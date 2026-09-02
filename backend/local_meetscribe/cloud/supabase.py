from __future__ import annotations

import json
import logging
import math
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urlencode, urlparse
from urllib.request import Request, urlopen

from local_meetscribe.config import Settings
from local_meetscribe.utils.errors import LocalMeetScribeError

LOGGER = logging.getLogger(__name__)
SIGNED_UPLOAD_TTL_SEC = 2 * 60 * 60


class SupabaseCloudError(LocalMeetScribeError):
    """A sanitized Supabase configuration or request failure."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class CloudHTTPResponse:
    status: int
    body: bytes
    headers: Mapping[str, str]


class SupabaseTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: int,
        output: BinaryIO | None = None,
    ) -> CloudHTTPResponse: ...


class UrlLibSupabaseTransport:
    """Small dependency-free HTTP transport, replaceable by a fake in tests."""

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: int,
        output: BinaryIO | None = None,
    ) -> CloudHTTPResponse:
        request = Request(url, data=body, headers=dict(headers), method=method)
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310 - configured URL.
                if output is None:
                    response_body = response.read()
                else:
                    response_body = b""
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                return CloudHTTPResponse(
                    status=response.status,
                    body=response_body,
                    headers={key.casefold(): value for key, value in response.headers.items()},
                )
        except HTTPError as exc:
            return CloudHTTPResponse(
                status=exc.code,
                body=exc.read(),
                headers={key.casefold(): value for key, value in exc.headers.items()},
            )
        except (OSError, URLError) as exc:
            raise SupabaseCloudError("Supabase could not be reached.") from exc


@dataclass(frozen=True)
class SignedUploadPart:
    part_number: int
    byte_start: int
    byte_end: int
    size_bytes: int
    object_path: str
    upload_url: str


@dataclass(frozen=True)
class SignedRecordingUpload:
    recording_id: str
    bucket_id: str
    object_path: str
    content_type: str
    parts: tuple[SignedUploadPart, ...]
    expires_in: int = SIGNED_UPLOAD_TTL_SEC


@dataclass(frozen=True)
class CloudRecording:
    id: str
    object_path: str
    original_filename: str
    mime_type: str
    size_bytes: int
    part_count: int
    part_size_bytes: int
    file_extension: str
    storage_status: str


@dataclass(frozen=True)
class CloudTranscriptSegment:
    start_sec: float
    end_sec: float
    text: str


@dataclass(frozen=True)
class RetentionCleanupResult:
    attempted: int
    deleted: int
    failed: int


class SupabaseCloudClient:
    """Server-side Supabase REST/Storage client.

    It intentionally uses only the service credential on the local backend. Browser
    clients receive path-scoped, two-hour upload signatures and never this key.
    """

    def __init__(
        self,
        *,
        url: str,
        service_role_key: str,
        bucket: str = "recordings",
        owner_id: str | None = None,
        part_size_bytes: int = 6 * 1024 * 1024,
        max_recording_bytes: int = 4 * 1024 * 1024 * 1024,
        timeout_sec: int = 30,
        transport: SupabaseTransport | None = None,
    ) -> None:
        normalized_url = url.strip().rstrip("/")
        parsed = urlparse(normalized_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise SupabaseCloudError("Supabase project URL must be an HTTPS URL.")
        if len(service_role_key.strip()) < 20:
            raise SupabaseCloudError("Supabase service role key is missing or too short.")
        if bucket != "recordings":
            raise SupabaseCloudError("Supabase Storage bucket must be 'recordings'.")
        if owner_id is not None:
            try:
                owner_id = str(uuid.UUID(owner_id))
            except ValueError as exc:
                raise SupabaseCloudError("Supabase owner ID must be a UUID.") from exc

        self.url = normalized_url
        self.service_role_key = service_role_key.strip()
        self.bucket = bucket
        self.owner_id = owner_id
        self.part_size_bytes = max(6 * 1024 * 1024, min(24 * 1024 * 1024, part_size_bytes))
        self.max_recording_bytes = max_recording_bytes
        self.timeout_sec = max(5, timeout_sec)
        self.transport = transport or UrlLibSupabaseTransport()

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        url: str | None = None,
        service_role_key: str | None = None,
        bucket: str | None = None,
        transport: SupabaseTransport | None = None,
    ) -> SupabaseCloudClient | None:
        if not settings.supabase_enabled and not (url and service_role_key):
            return None
        resolved_url = url or settings.supabase_url
        resolved_key = service_role_key or settings.supabase_service_role_key
        if not resolved_url or not resolved_key:
            LOGGER.warning(
                "Supabase upload is disabled because its local configuration is incomplete"
            )
            return None
        return cls(
            url=resolved_url,
            service_role_key=resolved_key,
            bucket=bucket or settings.supabase_bucket,
            owner_id=settings.supabase_owner_id,
            part_size_bytes=settings.supabase_part_size_bytes,
            max_recording_bytes=settings.supabase_max_recording_bytes,
            timeout_sec=settings.supabase_request_timeout_sec,
            transport=transport,
        )

    def create_upload_descriptor(
        self,
        *,
        filename: str,
        content_type: str,
        size_bytes: int,
    ) -> SignedRecordingUpload:
        if size_bytes <= 0:
            raise SupabaseCloudError("Recording size must be greater than zero.")
        if size_bytes > self.max_recording_bytes:
            raise SupabaseCloudError("Recording is larger than the configured cloud upload limit.")
        safe_filename = _safe_cloud_filename(filename)
        normalized_type = _normalized_audio_type(content_type, safe_filename)
        if not normalized_type.startswith("audio/"):
            raise SupabaseCloudError("Only audio recordings can use cloud upload.")

        recording_id = str(uuid.uuid4())
        extension = _file_extension(safe_filename, normalized_type)
        owner_folder = self.owner_id or "shared"
        object_prefix = f"{owner_folder}/{recording_id}"
        part_count = math.ceil(size_bytes / self.part_size_bytes)
        self._request_json(
            "POST",
            f"{self.url}/rest/v1/recordings",
            payload={
                "id": recording_id,
                "owner_id": self.owner_id,
                "bucket_id": self.bucket,
                "object_path": object_prefix,
                "original_filename": safe_filename,
                "mime_type": normalized_type,
                "size_bytes": size_bytes,
                "part_count": part_count,
                "part_size_bytes": self.part_size_bytes,
                "file_extension": extension,
                "storage_status": "pending_upload",
            },
            expected={201},
            prefer="return=minimal",
        )

        signed_parts: list[SignedUploadPart] = []
        try:
            for part_number in range(part_count):
                byte_start = part_number * self.part_size_bytes
                byte_end = min(size_bytes, byte_start + self.part_size_bytes)
                object_path = _part_object_path(object_prefix, part_number, extension)
                signed = self._request_json(
                    "POST",
                    self._storage_url("object/upload/sign", object_path),
                    payload={},
                    expected={200},
                )
                upload_url = _signed_upload_url(signed, self.url)
                signed_parts.append(
                    SignedUploadPart(
                        part_number=part_number,
                        byte_start=byte_start,
                        byte_end=byte_end,
                        size_bytes=byte_end - byte_start,
                        object_path=object_path,
                        upload_url=upload_url,
                    )
                )
        except SupabaseCloudError:
            self._mark_recording_failed(recording_id)
            raise

        return SignedRecordingUpload(
            recording_id=recording_id,
            bucket_id=self.bucket,
            object_path=object_prefix,
            content_type=normalized_type,
            parts=tuple(signed_parts),
        )

    def complete_recording(self, recording_id: str) -> CloudRecording:
        recording = self.get_recording(recording_id)
        if recording.storage_status == "ready":
            return recording
        if recording.storage_status != "pending_upload":
            raise SupabaseCloudError("Cloud recording is not awaiting upload completion.")
        for part_number in range(recording.part_count):
            object_path = _part_object_path(
                recording.object_path,
                part_number,
                recording.file_extension,
            )
            response = self.transport.request(
                "HEAD",
                self._storage_url("object", object_path),
                headers=self._headers(),
                body=None,
                timeout=self.timeout_sec,
            )
            if response.status != 200:
                raise SupabaseCloudError(
                    f"Cloud recording part {part_number + 1} is not available yet."
                )
            expected_size = min(
                recording.part_size_bytes,
                recording.size_bytes - part_number * recording.part_size_bytes,
            )
            raw_length = response.headers.get("content-length")
            if raw_length and raw_length.isdecimal() and int(raw_length) != expected_size:
                raise SupabaseCloudError(
                    f"Cloud recording part {part_number + 1} has an unexpected size."
                )

        self._request_json(
            "PATCH",
            f"{self.url}/rest/v1/recordings?{urlencode({'id': f'eq.{recording.id}'})}",
            payload={
                "storage_status": "ready",
                "upload_completed_at": datetime.now(UTC).isoformat(),
            },
            expected={204},
            prefer="return=minimal",
        )
        return CloudRecording(**{**recording.__dict__, "storage_status": "ready"})

    def get_recording(self, recording_id: str) -> CloudRecording:
        normalized_id = _recording_uuid(recording_id)
        query = urlencode(
            {
                "id": f"eq.{normalized_id}",
                "select": (
                    "id,object_path,original_filename,mime_type,size_bytes,part_count,"
                    "part_size_bytes,file_extension,storage_status"
                ),
            }
        )
        payload = self._request_json(
            "GET",
            f"{self.url}/rest/v1/recordings?{query}",
            expected={200},
        )
        if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
            raise SupabaseCloudError("Cloud recording was not found.")
        return _cloud_recording_from_row(payload[0])

    def download_recording(self, recording_id: str, destination: Path) -> CloudRecording:
        recording = self.get_recording(recording_id)
        if recording.storage_status != "ready":
            raise SupabaseCloudError("Cloud recording upload is not complete.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with destination.open("wb") as output:
                for part_number in range(recording.part_count):
                    object_path = _part_object_path(
                        recording.object_path,
                        part_number,
                        recording.file_extension,
                    )
                    response = self.transport.request(
                        "GET",
                        self._storage_url("object", object_path),
                        headers=self._headers(),
                        body=None,
                        timeout=self.timeout_sec,
                        output=output,
                    )
                    if response.status != 200:
                        raise SupabaseCloudError(
                            f"Cloud recording part {part_number + 1} could not be downloaded."
                        )
            if destination.stat().st_size != recording.size_bytes:
                raise SupabaseCloudError("Downloaded cloud recording has an unexpected size.")
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        return recording

    def sync_workflow_status(
        self,
        *,
        recording_id: str,
        workflow_id: str,
        status: str,
        stage: str,
        progress: float,
        error_message: str | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "recording_id": _recording_uuid(recording_id),
            "workflow_id": _workflow_identifier(workflow_id),
            "status": status,
            "stage": stage,
            "progress": max(0.0, min(1.0, progress)),
            "error_code": "workflow_failed" if error_message else None,
            "error_message": error_message[:500] if error_message else None,
        }
        if status in {"complete", "failed", "cancelled"}:
            payload["completed_at"] = datetime.now(UTC).isoformat()
        query = urlencode({"on_conflict": "workflow_id"})
        self._request_json(
            "POST",
            f"{self.url}/rest/v1/transcription_jobs?{query}",
            payload=payload,
            expected={201},
            prefer="resolution=merge-duplicates,return=minimal",
        )

    def try_sync_workflow_status(self, **values: Any) -> bool:
        try:
            self.sync_workflow_status(**values)
            return True
        except SupabaseCloudError as exc:
            LOGGER.warning(
                "Cloud status sync failed for workflow %s at stage %s (%s)",
                values.get("workflow_id"),
                values.get("stage"),
                type(exc).__name__,
            )
            return False

    def persist_transcript(
        self,
        *,
        recording_id: str,
        workflow_id: str,
        provider: str,
        model_name: str,
        text_raw: str,
        segments: list[CloudTranscriptSegment],
        suggested_filename: str,
    ) -> str:
        """Persist one immutable raw transcript and its ordered segments idempotently."""
        normalized_recording_id = _recording_uuid(recording_id)
        normalized_workflow_id = _workflow_identifier(workflow_id)
        jobs_query = urlencode(
            {
                "workflow_id": f"eq.{normalized_workflow_id}",
                "select": "id",
            }
        )
        jobs = self._request_json(
            "GET",
            f"{self.url}/rest/v1/transcription_jobs?{jobs_query}",
            expected={200},
        )
        if not isinstance(jobs, list) or len(jobs) != 1 or not isinstance(jobs[0], dict):
            raise SupabaseCloudError("Cloud transcription job was not found.")
        job_id = str(jobs[0].get("id") or "")
        _recording_uuid(job_id)

        existing_query = urlencode({"job_id": f"eq.{job_id}", "select": "id"})
        existing = self._request_json(
            "GET",
            f"{self.url}/rest/v1/transcripts?{existing_query}",
            expected={200},
        )
        transcript_id = _transcript_id_from_rows(existing)
        if transcript_id is None:
            try:
                created = self._request_json(
                    "POST",
                    f"{self.url}/rest/v1/transcripts",
                    payload={
                        "recording_id": normalized_recording_id,
                        "job_id": job_id,
                        "provider": provider,
                        "model_name": model_name,
                        "language": "unknown",
                        "text_raw": text_raw,
                        "text_clean": text_raw,
                        "metadata": {
                            "suggested_filename": suggested_filename,
                            "chunk_count": len(segments),
                        },
                    },
                    expected={201},
                    prefer="return=representation",
                )
                transcript_id = _transcript_id_from_rows(created)
                if transcript_id is None:
                    raise SupabaseCloudError("Supabase did not return the created transcript.")
            except SupabaseCloudError as exc:
                if exc.status != 409:
                    raise
                # A concurrent retry may have inserted the immutable transcript
                # after our initial lookup. Re-read it and continue with the
                # idempotent segment insert instead of treating that race as loss.
                concurrent = self._request_json(
                    "GET",
                    f"{self.url}/rest/v1/transcripts?{existing_query}",
                    expected={200},
                )
                transcript_id = _transcript_id_from_rows(concurrent)
                if transcript_id is None:
                    raise SupabaseCloudError(
                        "The concurrent cloud transcript could not be recovered."
                    ) from exc
        if segments:
            rows: list[dict[str, object]] = [
                {
                    "transcript_id": transcript_id,
                    "segment_index": index,
                    "start_ms": round(segment.start_sec * 1000),
                    "end_ms": round(segment.end_sec * 1000),
                    "speaker_label": "SPEAKER_00",
                    "language": "unknown",
                    "text_raw": segment.text,
                    "text_clean": segment.text,
                    "confidence": None,
                    "needs_review": False,
                    "overlap": False,
                    "words_raw": [],
                }
                for index, segment in enumerate(segments)
            ]
            self._request_json(
                "POST",
                (
                    f"{self.url}/rest/v1/transcript_segments?"
                    f"{urlencode({'on_conflict': 'transcript_id,segment_index'})}"
                ),
                payload=rows,
                expected={201},
                prefer="resolution=ignore-duplicates,return=minimal",
            )
        return transcript_id

    def try_persist_transcript(self, **values: Any) -> bool:
        try:
            self.persist_transcript(**values)
            return True
        except SupabaseCloudError as exc:
            LOGGER.warning(
                "Cloud transcript persistence failed for workflow %s (%s)",
                values.get("workflow_id"),
                type(exc).__name__,
            )
            return False

    def cleanup_expired_recordings(self, *, limit: int = 25) -> RetentionCleanupResult:
        """Delete a bounded batch of expired audio objects while preserving metadata.

        The database claim switches due rows to ``deleting`` and reclaims stale
        leases, so an interrupted cleanup is safely retryable.
        Transcript and job rows are preserved because only its retention state is
        finalized; the recording row itself is never deleted.
        """
        bounded_limit = max(1, min(100, limit))
        claimed = self._request_json(
            "POST",
            f"{self.url}/rest/v1/rpc/claim_expired_recordings",
            payload={"p_limit": bounded_limit},
            expected={200},
        )
        if not isinstance(claimed, list):
            raise SupabaseCloudError("Supabase returned invalid retention work.")
        rows: list[dict[str, object]] = []
        rows.extend(item for item in claimed if isinstance(item, dict))

        recordings: list[CloudRecording] = []
        seen: set[str] = set()
        for row in rows:
            if len(recordings) >= bounded_limit:
                break
            try:
                recording = _cloud_recording_from_row(row)
            except SupabaseCloudError:
                LOGGER.warning("Skipped invalid cloud retention metadata")
                continue
            if recording.id in seen or recording.storage_status != "deleting":
                continue
            seen.add(recording.id)
            recordings.append(recording)

        deleted = 0
        failed = 0
        for recording in recordings:
            try:
                self._delete_recording_parts(recording)
                completed = self._request_json(
                    "POST",
                    f"{self.url}/rest/v1/rpc/complete_recording_retention",
                    payload={"p_recording_id": recording.id},
                    expected={200},
                )
                if completed is not True:
                    raise SupabaseCloudError("Cloud retention completion was not accepted.")
                deleted += 1
            except SupabaseCloudError as exc:
                failed += 1
                LOGGER.warning(
                    "Cloud retention cleanup failed for recording %s (%s)",
                    recording.id,
                    type(exc).__name__,
                )
                try:
                    released = self._request_json(
                        "POST",
                        f"{self.url}/rest/v1/rpc/fail_recording_retention",
                        payload={
                            "p_recording_id": recording.id,
                            "p_error": "storage_delete_failed",
                        },
                        expected={200},
                    )
                    if released is not True:
                        raise SupabaseCloudError("Cloud retention claim was not released.")
                except SupabaseCloudError as release_exc:
                    LOGGER.warning(
                        "Could not release retention claim for recording %s (%s)",
                        recording.id,
                        type(release_exc).__name__,
                    )
        return RetentionCleanupResult(
            attempted=len(recordings),
            deleted=deleted,
            failed=failed,
        )

    def try_cleanup_expired_recordings(self, *, limit: int = 25) -> RetentionCleanupResult:
        try:
            return self.cleanup_expired_recordings(limit=limit)
        except SupabaseCloudError as exc:
            LOGGER.warning("Cloud retention cleanup could not start (%s)", type(exc).__name__)
            return RetentionCleanupResult(attempted=0, deleted=0, failed=0)

    def _delete_recording_parts(self, recording: CloudRecording) -> None:
        paths = [
            _part_object_path(
                recording.object_path,
                part_number,
                recording.file_extension,
            )
            for part_number in range(recording.part_count)
        ]
        for offset in range(0, len(paths), 100):
            self._request_json(
                "DELETE",
                f"{self.url}/storage/v1/object/{quote(self.bucket, safe='')}",
                payload={"prefixes": paths[offset : offset + 100]},
                expected={200},
            )

    def _mark_recording_failed(self, recording_id: str) -> None:
        try:
            self._request_json(
                "PATCH",
                f"{self.url}/rest/v1/recordings?{urlencode({'id': f'eq.{recording_id}'})}",
                payload={"storage_status": "failed"},
                expected={204},
                prefer="return=minimal",
            )
        except SupabaseCloudError:
            LOGGER.warning("Could not mark cloud recording %s as failed", recording_id)

    def _storage_url(self, operation: str, object_path: str) -> str:
        return (
            f"{self.url}/storage/v1/{operation}/"
            f"{quote(self.bucket, safe='')}/{quote(object_path, safe='/')}"
        )

    def _headers(self, *, prefer: str | None = None) -> dict[str, str]:
        headers = {
            "apikey": self.service_role_key,
            "content-type": "application/json",
        }
        # New `sb_secret_...` keys are opaque API keys, not JWT bearer tokens.
        # Legacy service_role JWTs still require Authorization for Storage/PostgREST.
        if self.service_role_key.startswith("eyJ"):
            headers["authorization"] = f"Bearer {self.service_role_key}"
        if prefer:
            headers["prefer"] = prefer
        return headers

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        payload: object | None = None,
        expected: set[int],
        prefer: str | None = None,
    ) -> Any:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        response = self.transport.request(
            method,
            url,
            headers=self._headers(prefer=prefer),
            body=body,
            timeout=self.timeout_sec,
        )
        if response.status not in expected:
            raise SupabaseCloudError(
                _response_error(response.status, response.body),
                status=response.status,
            )
        if not response.body:
            return None
        try:
            return json.loads(response.body)
        except json.JSONDecodeError as exc:
            raise SupabaseCloudError("Supabase returned an invalid response.") from exc


def _response_error(status: int, body: bytes) -> str:
    message = ""
    try:
        payload = json.loads(body)
        if isinstance(payload, dict):
            message = str(payload.get("message") or payload.get("error") or "")
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    suffix = f": {message[:200]}" if message else ""
    return f"Supabase request failed with HTTP {status}{suffix}"


def _transcript_id_from_rows(value: object) -> str | None:
    if not isinstance(value, list) or not value or not isinstance(value[0], dict):
        return None
    candidate = str(value[0].get("id") or "")
    try:
        return _recording_uuid(candidate)
    except SupabaseCloudError:
        return None


def _safe_cloud_filename(value: str) -> str:
    name = Path(value).name.strip()
    name = re.sub(r"[^\w.() -]+", "_", name, flags=re.UNICODE).strip(" .")
    return name[:180] or "recording.webm"


def _file_extension(filename: str, content_type: str) -> str:
    suffix = Path(filename).suffix.casefold().removeprefix(".")
    if re.fullmatch(r"[a-z0-9]{1,10}", suffix):
        return suffix
    return {
        "audio/mp4": "m4a",
        "audio/mpeg": "mp3",
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/webm": "webm",
        "audio/ogg": "ogg",
        "audio/aac": "aac",
        "audio/flac": "flac",
        "audio/3gpp": "3gp",
    }.get(content_type.partition(";")[0], "audio")


def _normalized_audio_type(content_type: str, filename: str) -> str:
    normalized = content_type.strip().casefold()
    if normalized and normalized != "application/octet-stream":
        return normalized
    return {
        ".m4a": "audio/mp4",
        ".mp4": "audio/mp4",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".aac": "audio/aac",
        ".ogg": "audio/ogg",
        ".webm": "audio/webm",
        ".flac": "audio/flac",
        ".3gp": "audio/3gpp",
    }.get(Path(filename).suffix.casefold(), normalized)


def _part_object_path(prefix: str, part_number: int, extension: str) -> str:
    return f"{prefix}/source.part{part_number:03d}.{extension}"


def _signed_upload_url(payload: object, project_url: str) -> str:
    if not isinstance(payload, dict):
        raise SupabaseCloudError("Supabase did not return a signed upload URL.")
    raw_url = payload.get("url") or payload.get("signedURL") or payload.get("signedUrl")
    if not isinstance(raw_url, str):
        raise SupabaseCloudError("Supabase did not return a signed upload URL.")
    if raw_url.startswith("/object/upload/sign/"):
        upload_url = f"{project_url}/storage/v1{raw_url}"
    elif raw_url.startswith("/storage/v1/object/upload/sign/"):
        upload_url = f"{project_url}{raw_url}"
    else:
        upload_url = raw_url
    parsed = urlparse(upload_url)
    project = urlparse(project_url)
    token = parse_qs(parsed.query).get("token", [""])[0]
    if (
        parsed.scheme != "https"
        or parsed.netloc != project.netloc
        or not parsed.path.startswith("/storage/v1/object/upload/sign/")
    ):
        raise SupabaseCloudError("Supabase returned an invalid signed upload URL.")
    if not token:
        raise SupabaseCloudError("Supabase did not return an upload token.")
    return upload_url


def _recording_uuid(value: str) -> str:
    try:
        return str(uuid.UUID(value))
    except ValueError as exc:
        raise SupabaseCloudError("Cloud recording ID is invalid.") from exc


def _cloud_recording_from_row(row: Mapping[str, object]) -> CloudRecording:
    try:
        recording = CloudRecording(
            id=_recording_uuid(str(row["id"])),
            object_path=str(row["object_path"]),
            original_filename=str(row["original_filename"]),
            mime_type=str(row["mime_type"]),
            size_bytes=int(str(row["size_bytes"])),
            part_count=int(str(row["part_count"])),
            part_size_bytes=int(str(row["part_size_bytes"])),
            file_extension=str(row["file_extension"]),
            storage_status=str(row["storage_status"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SupabaseCloudError("Cloud recording metadata is invalid.") from exc
    if (
        recording.size_bytes <= 0
        or not 1 <= recording.part_count <= 10_000
        or recording.part_size_bytes <= 0
        or not re.fullmatch(r"[a-z0-9]{1,10}", recording.file_extension)
        or not recording.object_path.endswith(f"/{recording.id}")
    ):
        raise SupabaseCloudError("Cloud recording metadata is invalid.")
    return recording


def _workflow_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", value):
        raise SupabaseCloudError("Cloud workflow ID is invalid.")
    return value
