from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import os
import re
import secrets
import shutil
import threading
import time
import unicodedata
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated, BinaryIO

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from local_meetscribe.cloud.supabase import (
    CloudTranscriptSegment,
    SupabaseCloudClient,
    SupabaseCloudError,
)
from local_meetscribe.config import Settings, ensure_runtime_dirs, get_settings
from local_meetscribe.db import JobStore
from local_meetscribe.pipeline.asr import has_cuda_runtime
from local_meetscribe.pipeline.export import write_exports
from local_meetscribe.pipeline.gemini import (
    GeminiTranscriptionProgress,
    get_gemini_progress,
    transcribe_gemini_package,
)
from local_meetscribe.pipeline.glossary import quick_scan_glossary
from local_meetscribe.pipeline.ingest import probe_media
from local_meetscribe.pipeline.optimizer import (
    OptimizerOverrides,
    OptimizerRequest,
    optimize_audio_package,
    recommend_optimization,
)
from local_meetscribe.pipeline.orchestrator import TranscriptionPipeline
from local_meetscribe.pipeline.prepare_audio import prepare_llm_audio
from local_meetscribe.schemas import (
    ExportKind,
    JobRecord,
    Transcript,
    TranscriptionRequest,
    TranscriptPatch,
    load_transcript,
)
from local_meetscribe.security import GeminiShareStore, SupabaseConfigStore
from local_meetscribe.utils.errors import LocalMeetScribeError

LOGGER = logging.getLogger(__name__)
STAGED_UPLOAD_TTL_SEC = 24 * 60 * 60
CLOUD_CLEANUP_BATCH_SIZE = 25
CLOUD_CLEANUP_INTERVAL_SEC = 15 * 60
CLOUD_MAINTENANCE_INTERVAL_SEC = 30
RECOVERABLE_WORKFLOW_STATUSES = frozenset({"queued", "optimizing", "transcribing"})


class CloudUploadDescriptorRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    content_type: str = Field(max_length=160)
    size_bytes: int = Field(gt=0)


class _WorkflowInputLease:
    """An OS-owned cross-process lock released automatically after a crash."""

    def __init__(self, handle: BinaryIO) -> None:
        self.handle = handle
        self.released = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        try:
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(  # type: ignore[attr-defined]
                    self.handle.fileno(),
                    fcntl.LOCK_UN,  # type: ignore[attr-defined]
                )
        except (OSError, ValueError):
            pass
        finally:
            self.handle.close()


def create_app(
    settings: Settings | None = None,
    *,
    supabase_client: SupabaseCloudClient | None = None,
) -> FastAPI:
    active_settings = settings or get_settings()
    ensure_runtime_dirs(active_settings)
    store = JobStore(active_settings)
    share_store = GeminiShareStore(active_settings.data_dir)
    supabase_store = SupabaseConfigStore(active_settings.data_dir)
    share_failures: dict[str, list[float]] = {}
    share_failure_lock = threading.Lock()
    remote_sessions: dict[str, float] = {}
    remote_session_lock = threading.Lock()
    active_workflow_inputs: set[str] = set()
    active_workflow_lock = threading.Lock()
    cloud_outbox_lock = threading.Lock()
    cloud_cleanup_lock = threading.Lock()
    cloud_cleanup_last_started = float("-inf")
    maintenance_stop = threading.Event()
    maintenance_wake = threading.Event()

    def current_supabase_client() -> SupabaseCloudClient | None:
        if supabase_client is not None:
            return supabase_client
        stored = None
        try:
            stored = supabase_store.load()
        except LocalMeetScribeError as exc:
            LOGGER.warning(
                "Saved Supabase configuration could not be loaded (%s)",
                type(exc).__name__,
            )
        url = active_settings.supabase_url or (stored.project_url if stored else None)
        key = active_settings.supabase_service_role_key or (
            stored.service_role_key if stored else None
        )
        bucket = (
            active_settings.supabase_bucket
            if active_settings.supabase_url or active_settings.supabase_service_role_key
            else (stored.bucket if stored else active_settings.supabase_bucket)
        )
        try:
            return SupabaseCloudClient.from_settings(
                active_settings,
                url=url,
                service_role_key=key,
                bucket=bucket,
            )
        except SupabaseCloudError as exc:
            LOGGER.warning("Supabase upload is disabled (%s)", type(exc).__name__)
            return None

    def run_cloud_cleanup_if_due(client: SupabaseCloudClient) -> None:
        nonlocal cloud_cleanup_last_started
        cleanup = getattr(client, "try_cleanup_expired_recordings", None)
        if not callable(cleanup):
            return
        now = time.monotonic()
        with cloud_cleanup_lock:
            if now - cloud_cleanup_last_started < CLOUD_CLEANUP_INTERVAL_SEC:
                return
            cloud_cleanup_last_started = now
        try:
            result = cleanup(limit=CLOUD_CLEANUP_BATCH_SIZE)
        except Exception as exc:  # noqa: BLE001 - maintenance must never affect uploads.
            LOGGER.warning("Cloud retention maintenance failed (%s)", type(exc).__name__)
            return
        LOGGER.info(
            "Cloud retention cleanup attempted %d recordings; deleted=%d failed=%d",
            result.attempted,
            result.deleted,
            result.failed,
        )

    def schedule_cloud_cleanup(_client: SupabaseCloudClient) -> None:
        # The lifespan-owned daemon performs cleanup. Waking it keeps descriptor
        # creation non-blocking while retaining the opportunistic behavior.
        maintenance_wake.set()

    def try_flush_cloud_outbox_file(
        path: Path,
        client: SupabaseCloudClient | None,
    ) -> bool:
        if client is None:
            return False
        try:
            payload = _read_json_object(path)
            workflow_id = str(payload.get("workflow_id") or "")
            recording_id = str(payload.get("recording_id") or "")
            package_id = str(payload.get("package_id") or "")
            status = str(payload.get("status") or "")
            stage = str(payload.get("stage") or status)
            if not re.fullmatch(r"[a-f0-9]{32}", workflow_id):
                raise LocalMeetScribeError("Cloud outbox workflow ID is invalid.")
            if not package_id or not recording_id or not status:
                raise LocalMeetScribeError("Cloud outbox entry is incomplete.")
            client.sync_workflow_status(
                recording_id=recording_id,
                workflow_id=workflow_id,
                status=status,
                stage=stage,
                progress=float(str(payload.get("progress") or 0.0)),
                error_message=(
                    str(payload.get("error_message"))[:500]
                    if payload.get("error_message")
                    else None
                ),
            )
            if bool(payload.get("include_transcript")):
                transcript = _read_json_object(
                    active_settings.data_dir / "optimized" / package_id / "gemini_transcript.json"
                )
                raw_chunks = transcript.get("chunks")
                chunks = raw_chunks if isinstance(raw_chunks, list) else []
                client.persist_transcript(
                    recording_id=recording_id,
                    workflow_id=workflow_id,
                    provider=str(transcript.get("provider") or "gemini"),
                    model_name=str(transcript.get("model") or active_settings.gemini_model),
                    text_raw=str(transcript.get("text") or ""),
                    segments=[
                        CloudTranscriptSegment(
                            start_sec=float(item.get("start_sec") or 0.0),
                            end_sec=float(item.get("end_sec") or 0.0),
                            text=str(item.get("text") or ""),
                        )
                        for item in chunks
                        if isinstance(item, dict)
                    ],
                    suggested_filename=str(transcript.get("suggested_filename") or "transcript"),
                )
            _update_json_object(
                _workflow_state_path(active_settings, workflow_id),
                {
                    "cloud_sync_complete": True,
                    "cloud_synced_at": time.time(),
                },
            )
            path.unlink(missing_ok=True)
            return True
        except Exception as exc:  # noqa: BLE001 - a durable outbox must survive all failures.
            LOGGER.warning(
                "Cloud outbox delivery failed for %s (%s)",
                path.stem,
                type(exc).__name__,
            )
            return False

    def persist_workflow_state(
        *,
        workflow_id: str,
        package_id: str,
        status: str,
        cloud_recording_id: str | None,
        cloud_client: SupabaseCloudClient | None,
        error: str | None = None,
        auto_exported: bool | None = None,
        auto_export_error: str | None = None,
        durable_fields: Mapping[str, object] | None = None,
        attempt_cloud_delivery: bool = True,
    ) -> None:
        state_path = _workflow_state_path(active_settings, workflow_id)
        progress_by_status = {
            "queued": 0.0,
            "optimizing": 0.15,
            "transcribing": 0.5,
            "complete": 1.0,
            "failed": 1.0,
        }
        if cloud_recording_id:
            with cloud_outbox_lock:
                _write_workflow_state(
                    state_path,
                    workflow_id=workflow_id,
                    package_id=package_id,
                    status=status,
                    error=error,
                    auto_exported=auto_exported,
                    auto_export_error=auto_export_error,
                    cloud_recording_id=cloud_recording_id,
                    cloud_sync_complete=False,
                    durable_fields=durable_fields,
                )
                outbox_path = _cloud_outbox_path(active_settings, workflow_id)
                _write_json_object(
                    outbox_path,
                    {
                        "schema_version": 1,
                        "workflow_id": workflow_id,
                        "recording_id": cloud_recording_id,
                        "package_id": package_id,
                        "status": status,
                        "stage": status,
                        "progress": progress_by_status.get(status, 0.0),
                        "error_message": error,
                        "include_transcript": status == "complete",
                        "updated_at": time.time(),
                    },
                )
                if attempt_cloud_delivery:
                    try_flush_cloud_outbox_file(outbox_path, cloud_client)
            maintenance_wake.set()
            return
        _write_workflow_state(
            state_path,
            workflow_id=workflow_id,
            package_id=package_id,
            status=status,
            error=error,
            auto_exported=auto_exported,
            auto_export_error=auto_export_error,
            durable_fields=durable_fields,
        )

    def require_share_passcode(request: Request, passcode: str | None) -> None:
        client_id = request.client.host if request.client else "unknown"
        now = time.monotonic()
        with share_failure_lock:
            recent = [
                attempted_at
                for attempted_at in share_failures.get(client_id, [])
                if now - attempted_at < 60.0
            ]
            share_failures[client_id] = recent
            if len(recent) >= 5:
                raise HTTPException(
                    status_code=429,
                    detail="비밀번호 확인 횟수를 초과했습니다. 1분 후 다시 시도하세요.",
                )
        if not passcode or not share_store.verify_passcode(passcode):
            with share_failure_lock:
                share_failures.setdefault(client_id, []).append(now)
            raise HTTPException(status_code=401, detail="공유 비밀번호가 맞지 않습니다.")
        with share_failure_lock:
            share_failures.pop(client_id, None)

    def issue_remote_session() -> str:
        now = time.time()
        token = secrets.token_urlsafe(32)
        with remote_session_lock:
            expired = [value for value, expires_at in remote_sessions.items() if expires_at <= now]
            for value in expired:
                remote_sessions.pop(value, None)
            remote_sessions[token] = now + active_settings.remote_session_ttl_sec
        return token

    def remote_session_is_valid(authorization: str | None) -> bool:
        scheme, separator, token = (authorization or "").partition(" ")
        if not separator or scheme.casefold() != "bearer" or not token:
            return False
        now = time.time()
        with remote_session_lock:
            expires_at = remote_sessions.get(token)
            if expires_at is None:
                return False
            if expires_at <= now:
                remote_sessions.pop(token, None)
                return False
        return True

    def run_transcription_workflow(
        workflow_id: str,
        package_id: str,
        upload_id: str | None,
        cloud_recording_id: str | None,
        cloud_client: SupabaseCloudClient | None,
        optimizer_request: OptimizerRequest,
        api_key: str,
        workflow_input_key: str,
        workflow_lease: _WorkflowInputLease,
    ) -> None:
        output_root = active_settings.data_dir / "optimized"
        output_dir = output_root / package_id
        phase = "optimizing" if upload_id or cloud_recording_id else "transcribing"
        cloud_download_dir: Path | None = None

        def persist_state(
            status: str,
            *,
            error: str | None = None,
            auto_exported: bool | None = None,
            auto_export_error: str | None = None,
        ) -> None:
            persist_workflow_state(
                workflow_id=workflow_id,
                package_id=package_id,
                status=status,
                error=error,
                auto_exported=auto_exported,
                auto_export_error=auto_export_error,
                cloud_recording_id=cloud_recording_id,
                cloud_client=cloud_client,
            )

        release_system_awake = _request_system_awake()
        try:
            if (upload_id or cloud_recording_id) and not _optimized_package_complete(output_dir):
                persist_state("optimizing")
                shutil.rmtree(output_dir, ignore_errors=True)
                if cloud_recording_id:
                    if cloud_client is None:
                        raise SupabaseCloudError("Supabase cloud upload is not configured.")
                    recording = cloud_client.get_recording(cloud_recording_id)
                    cloud_download_dir = active_settings.tmp_dir / "cloud-downloads" / workflow_id
                    source_path = cloud_download_dir / _safe_filename(recording.original_filename)
                    cloud_client.download_recording(cloud_recording_id, source_path)
                    staged_upload_dir = None
                else:
                    staged_upload_dir, source_path = _resolve_staged_upload(
                        active_settings.tmp_dir / "optimizer-uploads",
                        upload_id or "",
                    )
                optimize_audio_package(
                    source_path,
                    output_root,
                    active_settings,
                    optimizer_request,
                    package_id=package_id,
                )
                if not _optimized_package_complete(output_dir):
                    raise LocalMeetScribeError(
                        "The optimized recording package is incomplete. Select the recording again."
                    )
                if staged_upload_dir is not None:
                    shutil.rmtree(staged_upload_dir, ignore_errors=True)
                if cloud_download_dir is not None:
                    shutil.rmtree(cloud_download_dir, ignore_errors=True)

            if not _optimized_package_complete(output_dir):
                raise LocalMeetScribeError(
                    "The optimized recording package is incomplete. Select the recording again."
                )

            phase = "transcribing"
            persist_state("transcribing")
            transcript_result = transcribe_gemini_package(
                output_dir,
                active_settings,
                api_key=api_key,
            )
            auto_exported, auto_export_error = _auto_export_transcript(
                transcript_result.txt_path,
                transcript_result.suggested_filename,
                active_settings.auto_export_dir,
            )
            persist_state(
                "complete",
                auto_exported=auto_exported,
                auto_export_error=auto_export_error,
            )
        except Exception as exc:  # noqa: BLE001 - background work must persist failure state.
            if phase == "transcribing" and _gemini_outputs_complete(output_dir):
                LOGGER.info(
                    "Background workflow %s recovered completed transcript artifacts",
                    workflow_id,
                )
                auto_exported, auto_export_error = _auto_export_stored_transcript(
                    output_dir,
                    active_settings.auto_export_dir,
                )
                persist_state(
                    "complete",
                    auto_exported=auto_exported,
                    auto_export_error=auto_export_error,
                )
                return
            if phase == "optimizing":
                shutil.rmtree(output_dir, ignore_errors=True)
            LOGGER.error(
                "Background workflow %s failed during %s (%s)",
                workflow_id,
                phase,
                type(exc).__name__,
            )
            persist_state("failed", error=_background_error_message(exc))
        finally:
            if cloud_download_dir is not None:
                shutil.rmtree(cloud_download_dir, ignore_errors=True)
            release_system_awake()
            workflow_lease.release()
            with active_workflow_lock:
                active_workflow_inputs.discard(workflow_input_key)

    def try_reserve_workflow_input(
        workflow_input_key: str,
    ) -> tuple[_WorkflowInputLease | None, str | None]:
        with active_workflow_lock:
            if workflow_input_key in active_workflow_inputs:
                return None, "active"
            lease = _try_acquire_workflow_input_lease(
                active_settings.tmp_dir / "workflow-locks",
                workflow_input_key,
            )
            if lease is None:
                return None, "locked"
            active_workflow_inputs.add(workflow_input_key)
            return lease, None

    def recovery_api_key() -> str:
        configured = (active_settings.gemini_api_key or "").strip()
        if configured:
            return configured
        try:
            return (share_store.load_api_key() or "").strip()
        except LocalMeetScribeError as exc:
            LOGGER.warning("Saved Gemini key could not be loaded (%s)", type(exc).__name__)
            return ""

    def recover_workflows() -> None:
        workflow_dir = active_settings.tmp_dir / "workflows"
        workflow_dir.mkdir(parents=True, exist_ok=True)
        for state_path in sorted(workflow_dir.glob("*.json")):
            lease: _WorkflowInputLease | None = None
            workflow_input_key = ""
            try:
                state = _read_json_object(state_path)
                workflow_id = str(state.get("workflow_id") or "")
                package_id = str(state.get("package_id") or "")
                status = str(state.get("status") or "")
                if (
                    not re.fullmatch(r"[a-f0-9]{32}", workflow_id)
                    or state_path.stem != workflow_id
                    or not re.fullmatch(r"[a-f0-9]{32}", package_id)
                ):
                    raise LocalMeetScribeError("Workflow recovery state is invalid.")
                cloud_recording_id = str(state.get("cloud_recording_id") or "") or None
                if cloud_recording_id and not bool(state.get("cloud_sync_complete")):
                    with cloud_outbox_lock:
                        outbox_path = _cloud_outbox_path(active_settings, workflow_id)
                        if not outbox_path.exists():
                            _write_json_object(
                                outbox_path,
                                _cloud_outbox_payload_from_state(state),
                            )
                if status not in RECOVERABLE_WORKFLOW_STATUSES:
                    continue

                input_kind, input_id, workflow_input_key = _recovery_input(state, package_id)
                lease, reservation_error = try_reserve_workflow_input(workflow_input_key)
                if lease is None:
                    if reservation_error == "active":
                        persist_workflow_state(
                            workflow_id=workflow_id,
                            package_id=package_id,
                            status="failed",
                            error=(
                                "A newer recovery worker already owns this recording. "
                                "Retry the recording only if that workflow fails."
                            ),
                            cloud_recording_id=cloud_recording_id,
                            cloud_client=current_supabase_client(),
                            durable_fields={"error_code": "duplicate_recovery_worker"},
                            attempt_cloud_delivery=False,
                        )
                    continue

                output_dir = active_settings.data_dir / "optimized" / package_id
                if _gemini_outputs_complete(output_dir):
                    try:
                        auto_exported, auto_export_error = _auto_export_stored_transcript(
                            output_dir,
                            active_settings.auto_export_dir,
                        )
                        persist_workflow_state(
                            workflow_id=workflow_id,
                            package_id=package_id,
                            status="complete",
                            cloud_recording_id=cloud_recording_id,
                            cloud_client=current_supabase_client(),
                            auto_exported=auto_exported,
                            auto_export_error=auto_export_error,
                            durable_fields={
                                "attempt_count": int(str(state.get("attempt_count") or 0)) + 1,
                                "recovered_after_restart": True,
                            },
                            attempt_cloud_delivery=False,
                        )
                    finally:
                        lease.release()
                        lease = None
                        with active_workflow_lock:
                            active_workflow_inputs.discard(workflow_input_key)
                    continue

                api_key = recovery_api_key()
                cloud_client = current_supabase_client() if cloud_recording_id else None
                recovery_error: str | None = None
                error_code: str | None = None
                if not api_key:
                    recovery_error = (
                        "서버 재시작 후 전사를 계속하려면 서버 PC에 기본 Gemini API key를 "
                        "등록한 뒤 녹음을 다시 시작하세요. 요청 헤더의 API key는 저장되지 않습니다."
                    )
                    error_code = "recovery_key_unavailable"
                elif (
                    input_kind == "cloud"
                    and cloud_client is None
                    and not (_optimized_package_complete(output_dir))
                ):
                    recovery_error = (
                        "서버 재시작 후 클라우드 녹음을 다시 받으려면 서버 PC에서 "
                        "Supabase 연결을 복구한 뒤 다시 시도하세요."
                    )
                    error_code = "recovery_cloud_unavailable"
                if recovery_error:
                    try:
                        persist_workflow_state(
                            workflow_id=workflow_id,
                            package_id=package_id,
                            status="failed",
                            error=recovery_error,
                            cloud_recording_id=cloud_recording_id,
                            cloud_client=cloud_client,
                            durable_fields={
                                "error_code": error_code or "recovery_unavailable",
                                "requires_api_key": error_code == "recovery_key_unavailable",
                            },
                            attempt_cloud_delivery=False,
                        )
                    finally:
                        lease.release()
                        lease = None
                        with active_workflow_lock:
                            active_workflow_inputs.discard(workflow_input_key)
                    continue

                optimizer_request = _optimizer_request_from_payload(state.get("optimizer_request"))
                _update_json_object(
                    state_path,
                    {
                        "attempt_count": int(str(state.get("attempt_count") or 0)) + 1,
                        "recovered_after_restart": True,
                    },
                )
                recovery_thread = threading.Thread(
                    target=run_transcription_workflow,
                    args=(
                        workflow_id,
                        package_id,
                        input_id if input_kind == "upload" else None,
                        cloud_recording_id if input_kind == "cloud" else None,
                        cloud_client,
                        optimizer_request,
                        api_key,
                        workflow_input_key,
                        lease,
                    ),
                    name=f"phonescribe-recovery-{workflow_id[:8]}",
                    daemon=True,
                )
                recovery_thread.start()
                lease = None  # Ownership transferred to the recovery worker.
            except Exception as exc:  # noqa: BLE001 - one corrupt state cannot stop startup.
                if lease is not None:
                    lease.release()
                    with active_workflow_lock:
                        active_workflow_inputs.discard(workflow_input_key)
                LOGGER.warning(
                    "Skipped workflow recovery state %s (%s)",
                    state_path.stem,
                    type(exc).__name__,
                )

    def maintenance_loop() -> None:
        while not maintenance_stop.is_set():
            client = current_supabase_client()
            if client is not None:
                outbox_dir = active_settings.tmp_dir / "cloud-outbox"
                outbox_dir.mkdir(parents=True, exist_ok=True)
                for outbox_path in sorted(outbox_dir.glob("*.json")):
                    if maintenance_stop.is_set():
                        break
                    with cloud_outbox_lock:
                        try_flush_cloud_outbox_file(outbox_path, client)
                if not maintenance_stop.is_set():
                    run_cloud_cleanup_if_due(client)
            maintenance_wake.wait(CLOUD_MAINTENANCE_INTERVAL_SEC)
            maintenance_wake.clear()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        maintenance_stop.clear()
        maintenance_wake.clear()
        recover_workflows()
        maintenance_thread = threading.Thread(
            target=maintenance_loop,
            name="phonescribe-cloud-maintenance",
            daemon=True,
        )
        maintenance_thread.start()
        try:
            yield
        finally:
            maintenance_stop.set()
            maintenance_wake.set()
            maintenance_thread.join(timeout=2.0)

    app = FastAPI(title="LocalMeetScribe", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(active_settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def protect_remote_api(request: Request, call_next: Callable):
        public_api_paths = {
            "/api/health",
            "/api/runtime",
            "/api/gemini-share/verify",
        }
        requires_session = (
            active_settings.remote_access_enabled
            and request.method != "OPTIONS"
            and request.url.path.startswith("/api/")
            and request.url.path not in public_api_paths
            and not _is_loopback_request(request)
        )
        if requires_session and not remote_session_is_valid(request.headers.get("authorization")):
            return JSONResponse(
                status_code=401,
                content={"detail": "공유 비밀번호를 다시 확인하세요."},
            )
        return await call_next(request)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/runtime")
    def runtime(request: Request) -> dict[str, str | bool]:
        cuda = has_cuda_runtime()
        saved_share_key = share_store.api_key_configured
        cloud_upload_enabled = current_supabase_client() is not None
        return {
            "device": "cuda" if cuda else "cpu",
            "cuda": cuda,
            "fast_model": (
                active_settings.faster_whisper_cuda_model
                if cuda
                else active_settings.faster_whisper_cpu_model
            ),
            "accurate_model": active_settings.faster_whisper_cuda_model,
            "gemini_transcription_enabled": active_settings.enable_gemini_transcription,
            "gemini_api_key_configured": bool(active_settings.gemini_api_key or saved_share_key),
            "gemini_model": active_settings.gemini_model,
            "gemini_share_enabled": share_store.passcode_configured,
            "gemini_share_ready": bool(
                share_store.passcode_configured
                and (active_settings.gemini_api_key or saved_share_key)
            ),
            "supabase_configured": bool(
                active_settings.supabase_url and active_settings.supabase_service_role_key
            )
            or supabase_store.configured,
            "cloud_upload_enabled": cloud_upload_enabled,
            "local_admin": _is_loopback_request(request),
        }

    @app.post("/api/gemini-share/verify")
    def verify_gemini_share(
        request: Request,
        share_passcode: Annotated[str | None, Header(alias="X-LocalMeetScribe-Passcode")] = None,
    ) -> dict[str, object]:
        if not share_store.passcode_configured:
            raise HTTPException(status_code=404, detail="Shared Gemini access is not configured.")
        require_share_passcode(request, share_passcode)
        return {
            "valid": True,
            "key_ready": bool(active_settings.gemini_api_key or share_store.api_key_configured),
            "access_token": issue_remote_session(),
            "expires_in": active_settings.remote_session_ttl_sec,
        }

    @app.post("/api/admin/gemini-share-key")
    def configure_gemini_share_key(
        request: Request,
        api_key: Annotated[str, Form()],
        share_passcode: Annotated[str | None, Header(alias="X-LocalMeetScribe-Passcode")] = None,
    ) -> dict[str, bool]:
        if not _is_loopback_request(request):
            raise HTTPException(
                status_code=403,
                detail="기본 API key 등록은 서버 PC에서만 가능합니다.",
            )
        require_share_passcode(request, share_passcode)
        try:
            share_store.save_api_key(api_key)
        except LocalMeetScribeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"configured": True}

    @app.post("/api/admin/supabase-config")
    def configure_supabase(
        request: Request,
        project_url: Annotated[str, Form()],
        service_role_key: Annotated[str, Form()],
        share_passcode: Annotated[
            str | None,
            Header(alias="X-LocalMeetScribe-Passcode"),
        ] = None,
    ) -> dict[str, bool]:
        if not _is_loopback_request(request):
            raise HTTPException(
                status_code=403,
                detail="Supabase 설정은 서버 PC에서만 변경할 수 있습니다.",
            )
        if share_store.passcode_configured:
            require_share_passcode(request, share_passcode)
        try:
            supabase_store.save(
                project_url=project_url,
                service_role_key=service_role_key,
                bucket="recordings",
            )
            configured = current_supabase_client() is not None
            maintenance_wake.set()
        except LocalMeetScribeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"configured": configured, "cloud_upload_enabled": configured}

    @app.post("/api/cloud-recordings/upload-descriptor", status_code=201)
    def create_cloud_upload_descriptor(
        payload: CloudUploadDescriptorRequest,
    ) -> dict[str, object]:
        client = current_supabase_client()
        if client is None:
            raise HTTPException(
                status_code=503,
                detail="Supabase cloud upload is not configured. Use local upload instead.",
            )
        schedule_cloud_cleanup(client)
        try:
            descriptor = client.create_upload_descriptor(
                filename=payload.filename,
                content_type=payload.content_type,
                size_bytes=payload.size_bytes,
            )
        except SupabaseCloudError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {
            "recording_id": descriptor.recording_id,
            "bucket_id": descriptor.bucket_id,
            "object_path": descriptor.object_path,
            "content_type": descriptor.content_type,
            "expires_in": descriptor.expires_in,
            "parts": [
                {
                    "part_number": part.part_number,
                    "byte_start": part.byte_start,
                    "byte_end": part.byte_end,
                    "size_bytes": part.size_bytes,
                    "object_path": part.object_path,
                    "upload": {
                        "protocol": "signed-put",
                        "url": part.upload_url,
                        "headers": {
                            "content-type": descriptor.content_type,
                            "cache-control": "max-age=3600",
                            "x-upsert": "false",
                        },
                    },
                }
                for part in descriptor.parts
            ],
        }

    @app.post("/api/cloud-recordings/{recording_id}/complete")
    def complete_cloud_recording(recording_id: str) -> dict[str, str]:
        client = current_supabase_client()
        if client is None:
            raise HTTPException(status_code=503, detail="Supabase cloud upload is not configured.")
        try:
            recording = client.complete_recording(recording_id)
        except SupabaseCloudError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"recording_id": recording.id, "status": recording.storage_status}

    @app.post("/api/optimizer/analyze")
    def optimizer_analyze(
        file: Annotated[UploadFile, File()],
        destination: Annotated[str, Form()] = "gemini",
        openai_model: Annotated[str, Form()] = "gpt-4o-transcribe",
        word_timestamps: Annotated[bool, Form()] = False,
        codec: Annotated[str | None, Form()] = None,
        bitrate_kbps: Annotated[int | None, Form()] = None,
        chunk_minutes: Annotated[float | None, Form()] = None,
        remove_silence: Annotated[bool, Form()] = True,
        loudnorm: Annotated[bool, Form()] = True,
        speech_filter: Annotated[bool, Form()] = True,
        denoise: Annotated[bool, Form()] = False,
        language: Annotated[str, Form()] = "auto",
        run_quick_scan: Annotated[bool, Form(alias="quick_scan")] = True,
    ) -> dict[str, object]:
        if language not in {"auto", "ko", "en"}:
            raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")

        upload_id = uuid.uuid4().hex
        staged_root = active_settings.tmp_dir / "optimizer-uploads"
        _prune_staged_uploads(staged_root)
        upload_dir = staged_root / upload_id
        source_dir = upload_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        source_path = source_dir / _safe_filename(file.filename or "upload")
        scan_dir = upload_dir / "quick-scan"
        try:
            with source_path.open("wb") as output_file:
                shutil.copyfileobj(file.file, output_file)
            media_info = probe_media(source_path, active_settings)
            optimizer_request = _optimizer_request(
                destination=destination,
                openai_model=openai_model,
                word_timestamps=word_timestamps,
                codec=codec,
                bitrate_kbps=bitrate_kbps,
                chunk_minutes=chunk_minutes,
                remove_silence=remove_silence,
                loudnorm=loudnorm,
                speech_filter=speech_filter,
                denoise=denoise,
            )
            recommendation = recommend_optimization(
                media_info,
                source_path.stat().st_size,
                optimizer_request,
            )
            if not run_quick_scan:
                quick_scan = {
                    "glossary": [],
                    "preview_text": "",
                    "detected_language": "unknown",
                    "scan_seconds": 0,
                    "warning": None,
                }
            else:
                try:
                    scan = quick_scan_glossary(
                        source_path,
                        scan_dir,
                        active_settings,
                        language=language,  # type: ignore[arg-type]
                    )
                    quick_scan = {
                        "glossary": scan.terms,
                        "preview_text": scan.preview_text,
                        "detected_language": scan.detected_language,
                        "scan_seconds": scan.scan_seconds,
                        "warning": scan.warning,
                    }
                except Exception as exc:  # noqa: BLE001 - glossary scan is optional.
                    LOGGER.info(
                        "Quick scan unavailable for staged upload %s (%s)",
                        upload_id,
                        type(exc).__name__,
                    )
                    quick_scan = {
                        "glossary": [],
                        "preview_text": "",
                        "detected_language": "unknown",
                        "scan_seconds": 0,
                        "warning": "빠른 언어 스캔을 건너뛰었습니다.",
                    }
                finally:
                    shutil.rmtree(scan_dir, ignore_errors=True)

            return {
                "upload_id": upload_id,
                "source": media_info.__dict__,
                "original_bytes": source_path.stat().st_size,
                "recommendation": recommendation.__dict__,
                "quick_scan": quick_scan,
            }
        except LocalMeetScribeError as exc:
            shutil.rmtree(upload_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - API must convert failures to user errors.
            shutil.rmtree(upload_dir, ignore_errors=True)
            LOGGER.exception("Unexpected optimizer analysis failure for upload %s", upload_id)
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected optimizer analysis failure: {type(exc).__name__}: {exc}",
            ) from exc

    @app.post("/api/optimizer/recommend")
    def optimizer_recommend(
        file: Annotated[UploadFile, File()],
        destination: Annotated[str, Form()] = "gemini",
        openai_model: Annotated[str, Form()] = "gpt-4o-transcribe",
        word_timestamps: Annotated[bool, Form()] = False,
        codec: Annotated[str | None, Form()] = None,
        bitrate_kbps: Annotated[int | None, Form()] = None,
        chunk_minutes: Annotated[float | None, Form()] = None,
        remove_silence: Annotated[bool, Form()] = True,
        loudnorm: Annotated[bool, Form()] = True,
        speech_filter: Annotated[bool, Form()] = True,
        denoise: Annotated[bool, Form()] = False,
    ) -> dict[str, object]:
        scan_id = uuid.uuid4().hex
        scan_dir = active_settings.tmp_dir / f"optimizer-recommend-{scan_id}"
        scan_dir.mkdir(parents=True, exist_ok=True)
        source_path = scan_dir / _safe_filename(file.filename or "upload")
        try:
            with source_path.open("wb") as output_file:
                shutil.copyfileobj(file.file, output_file)
            media_info = probe_media(source_path, active_settings)
            request = _optimizer_request(
                destination=destination,
                openai_model=openai_model,
                word_timestamps=word_timestamps,
                codec=codec,
                bitrate_kbps=bitrate_kbps,
                chunk_minutes=chunk_minutes,
                remove_silence=remove_silence,
                loudnorm=loudnorm,
                speech_filter=speech_filter,
                denoise=denoise,
            )
            recommendation = recommend_optimization(media_info, source_path.stat().st_size, request)
            return {
                "source": media_info.__dict__,
                "original_bytes": source_path.stat().st_size,
                "recommendation": recommendation.__dict__,
            }
        except LocalMeetScribeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            shutil.rmtree(scan_dir, ignore_errors=True)

    @app.post("/api/optimizer/package")
    def optimizer_package(
        file: Annotated[UploadFile | None, File()] = None,
        destination: Annotated[str, Form()] = "gemini",
        openai_model: Annotated[str, Form()] = "gpt-4o-transcribe",
        word_timestamps: Annotated[bool, Form()] = False,
        codec: Annotated[str | None, Form()] = None,
        bitrate_kbps: Annotated[int | None, Form()] = None,
        chunk_minutes: Annotated[float | None, Form()] = None,
        remove_silence: Annotated[bool, Form()] = True,
        loudnorm: Annotated[bool, Form()] = True,
        speech_filter: Annotated[bool, Form()] = True,
        denoise: Annotated[bool, Form()] = False,
        upload_id: Annotated[str | None, Form()] = None,
    ) -> dict[str, object]:
        package_id = uuid.uuid4().hex
        output_root = active_settings.data_dir / "optimized"
        output_dir = output_root / package_id
        staged_upload_dir: Path | None = None
        source_dir: Path | None = None
        try:
            if upload_id:
                staged_upload_dir, source_path = _resolve_staged_upload(
                    active_settings.tmp_dir / "optimizer-uploads",
                    upload_id,
                )
            elif file is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                source_dir = output_dir / "source"
                source_dir.mkdir(parents=True, exist_ok=True)
                source_path = source_dir / _safe_filename(file.filename or "upload")
                with source_path.open("wb") as output_file:
                    shutil.copyfileobj(file.file, output_file)
            else:
                raise LocalMeetScribeError(
                    "The optimized package requires an uploaded file or staged upload ID."
                )
            request = _optimizer_request(
                destination=destination,
                openai_model=openai_model,
                word_timestamps=word_timestamps,
                codec=codec,
                bitrate_kbps=bitrate_kbps,
                chunk_minutes=chunk_minutes,
                remove_silence=remove_silence,
                loudnorm=loudnorm,
                speech_filter=speech_filter,
                denoise=denoise,
            )
            package = optimize_audio_package(
                source_path,
                output_root,
                active_settings,
                request,
                package_id=package_id,
            )
            if staged_upload_dir is not None:
                shutil.rmtree(staged_upload_dir, ignore_errors=True)
            elif source_dir is not None:
                shutil.rmtree(source_dir, ignore_errors=True)
            return {
                "id": package.id,
                "source": package.source.__dict__,
                "recommendation": package.recommendation.__dict__,
                "chunks": [chunk.__dict__ for chunk in package.chunks],
                "manifest_url": package.manifest_url,
                "package_url": package.package_url,
            }
        except LocalMeetScribeError as exc:
            shutil.rmtree(output_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - API must convert failures to user errors.
            shutil.rmtree(output_dir, ignore_errors=True)
            LOGGER.exception("Unexpected optimizer package failure")
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected optimizer package failure: {type(exc).__name__}: {exc}",
            ) from exc

    @app.post("/api/workflows", status_code=202)
    def start_transcription_workflow(
        background_tasks: BackgroundTasks,
        request: Request,
        upload_id: Annotated[str | None, Form()] = None,
        package_id: Annotated[str | None, Form()] = None,
        cloud_recording_id: Annotated[str | None, Form()] = None,
        destination: Annotated[str, Form()] = "gemini",
        openai_model: Annotated[str, Form()] = "gpt-4o-transcribe",
        word_timestamps: Annotated[bool, Form()] = False,
        codec: Annotated[str | None, Form()] = None,
        bitrate_kbps: Annotated[int | None, Form()] = None,
        chunk_minutes: Annotated[float | None, Form()] = None,
        remove_silence: Annotated[bool, Form()] = True,
        loudnorm: Annotated[bool, Form()] = True,
        speech_filter: Annotated[bool, Form()] = True,
        denoise: Annotated[bool, Form()] = False,
        gemini_api_key: Annotated[str | None, Header(alias="X-Gemini-API-Key")] = None,
        share_passcode: Annotated[
            str | None,
            Header(alias="X-LocalMeetScribe-Passcode"),
        ] = None,
    ) -> dict[str, object]:
        if sum(bool(value) for value in (upload_id, package_id, cloud_recording_id)) != 1:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Provide exactly one staged upload ID, optimized package ID, "
                    "or cloud recording ID."
                ),
            )

        optimizer_request = _optimizer_request(
            destination=destination,
            openai_model=openai_model,
            word_timestamps=word_timestamps,
            codec=codec,
            bitrate_kbps=bitrate_kbps,
            chunk_minutes=chunk_minutes,
            remove_silence=remove_silence,
            loudnorm=loudnorm,
            speech_filter=speech_filter,
            denoise=denoise,
        )
        if optimizer_request.destination != "gemini":
            raise HTTPException(
                status_code=400,
                detail="Background transcription workflows require the Gemini destination.",
            )

        workflow_cloud_client: SupabaseCloudClient | None = None
        if upload_id:
            try:
                _resolve_staged_upload(
                    active_settings.tmp_dir / "optimizer-uploads",
                    upload_id,
                )
            except LocalMeetScribeError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            resolved_package_id = uuid.uuid4().hex
        elif cloud_recording_id:
            workflow_cloud_client = current_supabase_client()
            if workflow_cloud_client is None:
                raise HTTPException(
                    status_code=503,
                    detail="Supabase cloud upload is not configured.",
                )
            try:
                cloud_recording = workflow_cloud_client.get_recording(cloud_recording_id)
            except SupabaseCloudError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            if cloud_recording.storage_status != "ready":
                raise HTTPException(
                    status_code=409,
                    detail="Cloud recording upload is not complete.",
                )
            cloud_recording_id = cloud_recording.id
            resolved_package_id = uuid.uuid4().hex
        else:
            resolved_package_id = package_id or ""
            if not re.fullmatch(r"[a-f0-9]{32}", resolved_package_id):
                raise HTTPException(status_code=404, detail="Optimized package not found.")
            if not (
                active_settings.data_dir / "optimized" / resolved_package_id / "manifest.json"
            ).exists():
                raise HTTPException(status_code=404, detail="Optimized package not found.")

        resolved_api_key = (gemini_api_key or "").strip()
        if share_store.passcode_configured:
            if not remote_session_is_valid(request.headers.get("authorization")):
                require_share_passcode(request, share_passcode)
            resolved_api_key = (
                active_settings.gemini_api_key or share_store.load_api_key() or ""
            ).strip()
        elif not resolved_api_key:
            resolved_api_key = (active_settings.gemini_api_key or "").strip()
        if not resolved_api_key:
            raise HTTPException(
                status_code=503,
                detail="Gemini API key가 필요합니다.",
            )

        workflow_id = uuid.uuid4().hex
        workflow_input_key = (
            f"upload:{upload_id}"
            if upload_id
            else (
                f"cloud:{cloud_recording_id}"
                if cloud_recording_id
                else f"package:{resolved_package_id}"
            )
        )
        workflow_lease, _reservation_error = try_reserve_workflow_input(workflow_input_key)
        if workflow_lease is None:
            raise HTTPException(
                status_code=409,
                detail="This recording already has an active transcription workflow.",
            )
        input_kind = "upload" if upload_id else ("cloud" if cloud_recording_id else "package")
        input_id = upload_id or cloud_recording_id or resolved_package_id
        credential_mode = (
            "ephemeral"
            if gemini_api_key
            and not share_store.passcode_configured
            and not active_settings.gemini_api_key
            else "server"
        )
        try:
            persist_workflow_state(
                workflow_id=workflow_id,
                package_id=resolved_package_id,
                status="queued",
                cloud_recording_id=cloud_recording_id,
                cloud_client=workflow_cloud_client,
                durable_fields={
                    "schema_version": 2,
                    "input_kind": input_kind,
                    "input_id": input_id or "",
                    "workflow_input_key": workflow_input_key,
                    "optimizer_request": _optimizer_request_payload(optimizer_request),
                    "credential_mode": credential_mode,
                    "attempt_count": 0,
                    "created_at": time.time(),
                },
            )
            background_tasks.add_task(
                run_transcription_workflow,
                workflow_id,
                resolved_package_id,
                upload_id,
                cloud_recording_id,
                workflow_cloud_client,
                optimizer_request,
                resolved_api_key,
                workflow_input_key,
                workflow_lease,
            )
        except Exception:
            workflow_lease.release()
            with active_workflow_lock:
                active_workflow_inputs.discard(workflow_input_key)
            raise
        return {
            "workflow_id": workflow_id,
            "package_id": resolved_package_id,
            "cloud_recording_id": cloud_recording_id,
            "status": "queued",
        }

    @app.get("/api/workflows/{workflow_id}")
    def transcription_workflow_status(workflow_id: str) -> dict[str, object]:
        if not re.fullmatch(r"[a-f0-9]{32}", workflow_id):
            raise HTTPException(status_code=404, detail="Workflow not found.")
        state_path = _workflow_state_path(active_settings, workflow_id)
        if not state_path.exists():
            raise HTTPException(status_code=404, detail="Workflow not found.")

        state = _read_json_object(state_path)
        package_id = str(state.get("package_id") or "")
        status = str(state.get("status") or "failed")
        package_dir = active_settings.data_dir / "optimized" / package_id
        if status == "failed" and _gemini_outputs_complete(package_dir):
            status = "complete"
            auto_exported, auto_export_error = _auto_export_stored_transcript(
                package_dir,
                active_settings.auto_export_dir,
            )
            state["auto_exported"] = auto_exported
            state["auto_export_error"] = auto_export_error
            try:
                persist_workflow_state(
                    workflow_id=workflow_id,
                    package_id=package_id,
                    status="complete",
                    auto_exported=auto_exported,
                    auto_export_error=auto_export_error,
                    cloud_recording_id=str(state.get("cloud_recording_id") or "") or None,
                    cloud_client=current_supabase_client(),
                    attempt_cloud_delivery=False,
                )
            except OSError:
                LOGGER.info("Could not persist recovered workflow %s", workflow_id)
        response: dict[str, object] = {
            "workflow_id": workflow_id,
            "package_id": package_id,
            "status": status,
            "error": None if status == "complete" else state.get("error"),
            "auto_exported": state.get("auto_exported"),
            "auto_export_error": state.get("auto_export_error"),
        }
        if state.get("cloud_recording_id"):
            response["cloud_recording_id"] = state["cloud_recording_id"]
        if (package_dir / "manifest.json").exists():
            response["package"] = _optimized_package_payload(package_dir, package_id)

        if status == "transcribing":
            progress = get_gemini_progress(package_dir)
            response["transcription_progress"] = _gemini_progress_payload(progress)
        elif status == "complete":
            response["transcription_progress"] = _gemini_progress_payload(
                get_gemini_progress(package_dir)
            )
            response["transcript"] = _stored_gemini_transcript_payload(
                package_dir,
                package_id,
            )
        return response

    @app.get("/api/optimizer/packages/{package_id}/{filename}")
    def download_optimizer_package_file(
        package_id: str,
        filename: str,
        download_name: str | None = None,
    ) -> FileResponse:
        if not re.fullmatch(r"[a-f0-9]{32}", package_id):
            raise HTTPException(status_code=404, detail="Optimized package file not found.")
        allowed = re.fullmatch(
            r"(chunk_\d{3}\.(mp3|m4a|ogg)|manifest\.json|optimized_package\.zip|"
            r"gemini_transcript\.(json|txt))",
            filename,
        )
        if not allowed:
            raise HTTPException(status_code=404, detail="Optimized package file not found.")
        path = active_settings.data_dir / "optimized" / package_id / filename
        if not path.exists():
            raise HTTPException(status_code=404, detail="Optimized package file not found.")
        return FileResponse(
            path,
            filename=_safe_download_filename(download_name, filename),
        )

    @app.post("/api/optimizer/packages/{package_id}/gemini-transcribe")
    def gemini_transcribe_package(
        package_id: str,
        request: Request,
        gemini_api_key: Annotated[str | None, Header(alias="X-Gemini-API-Key")] = None,
        share_passcode: Annotated[
            str | None,
            Header(alias="X-LocalMeetScribe-Passcode"),
        ] = None,
    ) -> dict[str, object]:
        if not re.fullmatch(r"[a-f0-9]{32}", package_id):
            raise HTTPException(status_code=404, detail="Optimized package not found.")
        package_dir = active_settings.data_dir / "optimized" / package_id
        if not (package_dir / "manifest.json").exists():
            raise HTTPException(status_code=404, detail="Optimized package not found.")
        try:
            resolved_api_key = gemini_api_key
            if share_store.passcode_configured:
                require_share_passcode(request, share_passcode)
                resolved_api_key = active_settings.gemini_api_key or share_store.load_api_key()
                if not resolved_api_key:
                    raise HTTPException(
                        status_code=503,
                        detail="서버 PC에서 기본 Gemini API key를 먼저 등록하세요.",
                    )
            result = transcribe_gemini_package(
                package_dir,
                active_settings,
                api_key=resolved_api_key,
            )
            return {
                "provider": result.provider,
                "model": result.model,
                "text": result.text,
                "suggested_filename": result.suggested_filename,
                "chunk_count": len(result.chunks),
                "chunks": [
                    {
                        "filename": chunk.filename,
                        "start_sec": chunk.start_sec,
                        "end_sec": chunk.end_sec,
                        "delivery": chunk.delivery,
                        "mime_type": chunk.mime_type,
                    }
                    for chunk in result.chunks
                ],
                "txt_url": f"/api/optimizer/packages/{package_id}/gemini_transcript.txt",
                "json_url": f"/api/optimizer/packages/{package_id}/gemini_transcript.json",
            }
        except HTTPException:
            raise
        except LocalMeetScribeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - API must convert failures to user errors.
            LOGGER.exception("Unexpected Gemini transcription failure for package %s", package_id)
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected Gemini transcription failure: {type(exc).__name__}: {exc}",
            ) from exc

    @app.get("/api/optimizer/packages/{package_id}/gemini-transcribe/progress")
    def gemini_transcription_progress(package_id: str) -> dict[str, object]:
        if not re.fullmatch(r"[a-f0-9]{32}", package_id):
            raise HTTPException(status_code=404, detail="Optimized package not found.")
        package_dir = active_settings.data_dir / "optimized" / package_id
        if not (package_dir / "manifest.json").exists():
            raise HTTPException(status_code=404, detail="Optimized package not found.")
        try:
            progress = get_gemini_progress(package_dir)
            return {
                "status": progress.status,
                "completed_chunks": progress.completed_chunks,
                "total_chunks": progress.total_chunks,
                "current_chunk": progress.current_chunk,
                "progress": progress.progress,
                "elapsed_sec": progress.elapsed_sec,
                "eta_sec": progress.eta_sec,
            }
        except LocalMeetScribeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/files/quick-scan")
    def quick_scan_file(
        file: Annotated[UploadFile, File()],
        language: Annotated[str, Form()] = "auto",
    ) -> dict[str, object]:
        scan_id = uuid.uuid4().hex
        scan_dir = active_settings.tmp_dir / f"quick-scan-{scan_id}"
        scan_dir.mkdir(parents=True, exist_ok=True)
        source_path = scan_dir / _safe_filename(file.filename or "upload")
        try:
            with source_path.open("wb") as output_file:
                shutil.copyfileobj(file.file, output_file)
            result = quick_scan_glossary(
                source_path,
                scan_dir,
                active_settings,
                language=language,  # type: ignore[arg-type]
            )
            return {
                "glossary": result.terms,
                "preview_text": result.preview_text,
                "detected_language": result.detected_language,
                "scan_seconds": result.scan_seconds,
                "warning": result.warning,
            }
        except LocalMeetScribeError as exc:
            return {
                "glossary": [],
                "preview_text": "",
                "detected_language": "unknown",
                "scan_seconds": 0,
                "warning": str(exc),
            }
        except Exception as exc:  # noqa: BLE001 - quick scan should not block upload.
            LOGGER.exception("Unexpected quick glossary scan failure")
            return {
                "glossary": [],
                "preview_text": "",
                "detected_language": "unknown",
                "scan_seconds": 0,
                "warning": f"Unexpected quick scan failure: {type(exc).__name__}: {exc}",
            }
        finally:
            shutil.rmtree(scan_dir, ignore_errors=True)

    @app.post("/api/files/prepare-audio")
    def prepare_audio_file(
        file: Annotated[UploadFile, File()],
        remove_silence: Annotated[bool, Form()] = True,
        max_minutes: Annotated[float | None, Form()] = None,
        bitrate_kbps: Annotated[int, Form()] = 32,
    ) -> dict[str, object]:
        prepare_id = uuid.uuid4().hex
        prepare_dir = active_settings.data_dir / "prepared" / prepare_id
        prepare_dir.mkdir(parents=True, exist_ok=True)
        source_path = prepare_dir / f"source_{_safe_filename(file.filename or 'upload')}"
        try:
            with source_path.open("wb") as output_file:
                shutil.copyfileobj(file.file, output_file)
            prepared = prepare_llm_audio(
                source_path,
                prepare_dir,
                active_settings,
                remove_silence=remove_silence,
                max_minutes=max_minutes,
                bitrate_kbps=bitrate_kbps,
            )
            source_path.unlink(missing_ok=True)
            return {
                "id": prepare_id,
                "filename": prepared.path.name,
                "download_url": f"/api/files/prepared/{prepare_id}",
                "original_duration_sec": prepared.original_info.duration_sec,
                "prepared_duration_sec": prepared.prepared_info.duration_sec,
                "original_bytes": prepared.original_bytes,
                "prepared_bytes": prepared.prepared_bytes,
                "compression_ratio": prepared.compression_ratio,
                "remove_silence": prepared.remove_silence,
                "max_minutes": prepared.max_minutes,
                "bitrate_kbps": prepared.bitrate_kbps,
            }
        except LocalMeetScribeError as exc:
            shutil.rmtree(prepare_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - API must convert failures to user errors.
            shutil.rmtree(prepare_dir, ignore_errors=True)
            LOGGER.exception("Unexpected LLM audio preparation failure")
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected LLM audio preparation failure: {type(exc).__name__}: {exc}",
            ) from exc

    @app.get("/api/files/prepared/{prepare_id}")
    def download_prepared_audio(prepare_id: str) -> FileResponse:
        if not re.fullmatch(r"[a-f0-9]{32}", prepare_id):
            raise HTTPException(status_code=404, detail="Prepared audio not found.")
        path = active_settings.data_dir / "prepared" / prepare_id / "llm_audio.m4a"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Prepared audio not found.")
        return FileResponse(path, filename="llm_audio.m4a", media_type="audio/mp4")

    @app.get("/api/jobs", response_model=list[JobRecord])
    def list_jobs() -> list[JobRecord]:
        return store.list_jobs()

    @app.post("/api/jobs", response_model=JobRecord)
    def create_job(
        background_tasks: BackgroundTasks,
        file: Annotated[UploadFile, File()],
        mode: Annotated[str, Form()] = "accurate",
        language: Annotated[str, Form()] = "auto",
        speakers: Annotated[int | None, Form()] = None,
        min_speakers: Annotated[int | None, Form()] = None,
        max_speakers: Annotated[int | None, Form()] = None,
        glossary: Annotated[str, Form()] = "",
        denoise: Annotated[bool, Form()] = False,
        loudness_normalize: Annotated[bool, Form()] = False,
        trim_silence: Annotated[bool, Form()] = False,
    ) -> JobRecord:
        job_id = uuid.uuid4().hex
        upload_dir = active_settings.uploads_dir / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        filename = _safe_filename(file.filename or "upload")
        source_path = upload_dir / filename
        with source_path.open("wb") as output_file:
            shutil.copyfileobj(file.file, output_file)
        output_dir = active_settings.job_dir(job_id) / "exports"
        request = TranscriptionRequest(
            mode=mode,  # type: ignore[arg-type]
            language=language,  # type: ignore[arg-type]
            speakers=speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            glossary=_parse_glossary(glossary),
            denoise=denoise,
            loudness_normalize=loudness_normalize,
            trim_silence=trim_silence,
        )
        job = store.create_job(job_id, source_path=source_path, output_dir=output_dir)
        background_tasks.add_task(
            _run_job, active_settings, job_id, source_path, output_dir, request
        )
        return job

    @app.get("/api/jobs/{job_id}", response_model=JobRecord)
    def get_job(job_id: str) -> JobRecord:
        return _get_job_or_404(store, job_id)

    @app.get("/api/jobs/{job_id}/transcript", response_model=Transcript)
    def get_transcript(job_id: str) -> Transcript:
        job = _get_job_or_404(store, job_id)
        if not job.transcript_path:
            raise HTTPException(status_code=404, detail="Transcript is not ready.")
        return load_transcript(Path(job.transcript_path))

    @app.patch("/api/jobs/{job_id}/transcript", response_model=Transcript)
    def patch_transcript(job_id: str, patch: TranscriptPatch) -> Transcript:
        job = _get_job_or_404(store, job_id)
        if not job.transcript_path or not job.output_dir:
            raise HTTPException(status_code=404, detail="Transcript is not ready.")
        transcript = load_transcript(Path(job.transcript_path))
        text_updates = {item.id: item.text_clean for item in patch.segments}
        speaker_updates = {item.id: item.display_name for item in patch.speakers}
        for segment in transcript.segments:
            if segment.id in text_updates:
                segment.text_clean = text_updates[segment.id]
        for speaker in transcript.speakers:
            if speaker.id in speaker_updates:
                speaker.display_name = speaker_updates[speaker.id]
        transcript = write_exports(transcript, Path(job.output_dir))
        transcript_path = Path(transcript.exports.json_path or job.transcript_path)
        store.update_job(job_id, transcript_path=transcript_path)
        return transcript

    @app.get("/api/jobs/{job_id}/exports/{kind}")
    def get_export(job_id: str, kind: ExportKind) -> FileResponse:
        job = _get_job_or_404(store, job_id)
        if not job.transcript_path:
            raise HTTPException(status_code=404, detail="Transcript is not ready.")
        transcript = load_transcript(Path(job.transcript_path))
        export_path = _export_path(transcript, kind)
        if not export_path or not Path(export_path).exists():
            raise HTTPException(status_code=404, detail=f"Export not found: {kind}")
        return FileResponse(export_path, filename=Path(export_path).name)

    @app.get("/api/jobs/{job_id}/audio")
    def get_audio(job_id: str) -> FileResponse:
        job = _get_job_or_404(store, job_id)
        if not job.source_path or not Path(job.source_path).exists():
            raise HTTPException(status_code=404, detail="Audio is not available.")
        return FileResponse(job.source_path, filename=Path(job.source_path).name)

    _mount_frontend(app)
    return app


def _run_job(
    settings: Settings,
    job_id: str,
    source_path: Path,
    output_dir: Path,
    request: TranscriptionRequest,
) -> None:
    store = JobStore(settings)

    def progress(stage: str, progress_value: float) -> None:
        store.update_job(job_id, status="running", stage=stage, progress=progress_value)

    try:
        store.update_job(job_id, status="running", stage="starting", progress=0.01)
        pipeline = TranscriptionPipeline(settings, progress_callback=progress)
        transcript = pipeline.run(
            source_path,
            output_dir=output_dir,
            request=request,
            job_id=job_id,
        )
        transcript_path = Path(transcript.exports.json_path or output_dir / "transcript.json")
        store.update_job(
            job_id,
            status="completed",
            stage="completed",
            progress=1.0,
            transcript_path=transcript_path,
        )
    except LocalMeetScribeError as exc:
        store.update_job(job_id, status="failed", stage="failed", progress=1.0, error=str(exc))
    except Exception as exc:  # noqa: BLE001 - API must convert background failures to job errors.
        LOGGER.exception("Unexpected pipeline failure for job %s", job_id)
        store.update_job(
            job_id,
            status="failed",
            stage="failed",
            progress=1.0,
            error=f"Unexpected pipeline failure: {type(exc).__name__}: {exc}",
        )


def _get_job_or_404(store: JobStore, job_id: str) -> JobRecord:
    try:
        return store.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc


def _parse_glossary(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def _optimizer_request(
    *,
    destination: str,
    openai_model: str,
    word_timestamps: bool,
    codec: str | None,
    bitrate_kbps: int | None,
    chunk_minutes: float | None,
    remove_silence: bool,
    loudnorm: bool,
    speech_filter: bool,
    denoise: bool,
) -> OptimizerRequest:
    if destination == "anthropic":
        raise LocalMeetScribeError("Claude cannot transcribe raw audio. Use it only after STT.")
    if destination not in {"gemini", "openai", "optimize"}:
        raise LocalMeetScribeError(f"Unsupported optimizer destination: {destination}")
    if openai_model not in {"gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"}:
        raise LocalMeetScribeError(f"Unsupported OpenAI transcription model: {openai_model}")
    if codec == "":
        codec = None
    if bitrate_kbps == 0:
        bitrate_kbps = None
    if chunk_minutes == 0:
        chunk_minutes = None
    return OptimizerRequest(
        destination=destination,  # type: ignore[arg-type]
        openai_model=openai_model,  # type: ignore[arg-type]
        word_timestamps=word_timestamps,
        overrides=OptimizerOverrides(
            codec=codec,  # type: ignore[arg-type]
            bitrate_kbps=bitrate_kbps,
            chunk_minutes=chunk_minutes,
            remove_silence=remove_silence,
            loudnorm=loudnorm,
            speech_filter=speech_filter,
            denoise=denoise,
        ),
    )


def _optimizer_request_payload(request: OptimizerRequest) -> dict[str, object]:
    return {
        "destination": request.destination,
        "openai_model": request.openai_model,
        "word_timestamps": request.word_timestamps,
        "overrides": {
            "codec": request.overrides.codec,
            "bitrate_kbps": request.overrides.bitrate_kbps,
            "chunk_minutes": request.overrides.chunk_minutes,
            "remove_silence": request.overrides.remove_silence,
            "loudnorm": request.overrides.loudnorm,
            "speech_filter": request.overrides.speech_filter,
            "denoise": request.overrides.denoise,
        },
    }


def _optimizer_request_from_payload(value: object) -> OptimizerRequest:
    if value is None:
        return _optimizer_request(
            destination="gemini",
            openai_model="gpt-4o-transcribe",
            word_timestamps=False,
            codec=None,
            bitrate_kbps=None,
            chunk_minutes=None,
            remove_silence=True,
            loudnorm=True,
            speech_filter=True,
            denoise=False,
        )
    if not isinstance(value, dict):
        raise LocalMeetScribeError("Workflow optimizer settings are invalid.")
    raw_overrides = value.get("overrides")
    overrides = raw_overrides if isinstance(raw_overrides, dict) else {}
    return _optimizer_request(
        destination=str(value.get("destination") or "gemini"),
        openai_model=str(value.get("openai_model") or "gpt-4o-transcribe"),
        word_timestamps=bool(value.get("word_timestamps", False)),
        codec=str(overrides["codec"]) if overrides.get("codec") else None,
        bitrate_kbps=(
            int(overrides["bitrate_kbps"]) if overrides.get("bitrate_kbps") is not None else None
        ),
        chunk_minutes=(
            float(overrides["chunk_minutes"])
            if overrides.get("chunk_minutes") is not None
            else None
        ),
        remove_silence=bool(overrides.get("remove_silence", True)),
        loudnorm=bool(overrides.get("loudnorm", True)),
        speech_filter=bool(overrides.get("speech_filter", True)),
        denoise=bool(overrides.get("denoise", False)),
    )


def _safe_filename(value: str) -> str:
    name = unicodedata.normalize("NFC", Path(value).name)
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name)
    return name.strip(" .") or "upload"


def _workflow_state_path(settings: Settings, workflow_id: str) -> Path:
    return settings.tmp_dir / "workflows" / f"{workflow_id}.json"


def _cloud_outbox_path(settings: Settings, workflow_id: str) -> Path:
    return settings.tmp_dir / "cloud-outbox" / f"{workflow_id}.json"


def _write_workflow_state(
    path: Path,
    *,
    workflow_id: str,
    package_id: str,
    status: str,
    error: str | None = None,
    auto_exported: bool | None = None,
    auto_export_error: str | None = None,
    cloud_recording_id: str | None = None,
    cloud_sync_complete: bool | None = None,
    durable_fields: Mapping[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {}
    if path.exists():
        with suppress(LocalMeetScribeError):
            payload.update(_read_json_object(path))
    if durable_fields:
        payload.update(durable_fields)
    payload.update(
        {
            "workflow_id": workflow_id,
            "package_id": package_id,
            "status": status,
            "error": error,
            "updated_at": time.time(),
        }
    )
    if auto_exported is not None:
        payload["auto_exported"] = auto_exported
    if auto_export_error is not None:
        payload["auto_export_error"] = auto_export_error
    if cloud_recording_id is not None:
        payload["cloud_recording_id"] = cloud_recording_id
    if cloud_sync_complete is not None:
        payload["cloud_sync_complete"] = cloud_sync_complete
    _write_json_object(path, payload)


def _write_json_object(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = json.dumps(dict(payload), ensure_ascii=False, indent=2)
    for attempt in range(5):
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(value, encoding="utf-8")
            temporary.replace(path)
            return
        except PermissionError:
            if attempt + 1 >= 5:
                raise
            time.sleep(0.05 * (2**attempt))
        finally:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


def _update_json_object(path: Path, updates: Mapping[str, object]) -> None:
    current = _read_json_object(path) if path.exists() else {}
    current.update(updates)
    _write_json_object(path, current)


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LocalMeetScribeError(f"Could not read workflow state: {path.name}") from exc
    if not isinstance(payload, dict):
        raise LocalMeetScribeError(f"Workflow state is invalid: {path.name}")
    return payload


def _cloud_outbox_payload_from_state(state: Mapping[str, object]) -> dict[str, object]:
    status = str(state.get("status") or "failed")
    return {
        "schema_version": 1,
        "workflow_id": str(state.get("workflow_id") or ""),
        "recording_id": str(state.get("cloud_recording_id") or ""),
        "package_id": str(state.get("package_id") or ""),
        "status": status,
        "stage": status,
        "progress": {
            "queued": 0.0,
            "optimizing": 0.15,
            "transcribing": 0.5,
            "complete": 1.0,
            "failed": 1.0,
        }.get(status, 0.0),
        "error_message": state.get("error"),
        "include_transcript": status == "complete",
        "updated_at": time.time(),
    }


def _recovery_input(
    state: Mapping[str, object],
    package_id: str,
) -> tuple[str, str, str]:
    schema_version = state.get("schema_version")
    if schema_version is not None and schema_version != 2:
        raise LocalMeetScribeError("Workflow recovery state version is unsupported.")
    input_kind = str(state.get("input_kind") or "")
    input_id = str(state.get("input_id") or "")
    input_key = str(state.get("workflow_input_key") or "")
    if not input_kind:
        cloud_recording_id = str(state.get("cloud_recording_id") or "")
        if cloud_recording_id:
            input_kind = "cloud"
            input_id = cloud_recording_id
        elif package_id:
            input_kind = "package"
            input_id = package_id
        input_key = f"{input_kind}:{input_id}"
    if input_kind not in {"upload", "cloud", "package"} or not input_id:
        raise LocalMeetScribeError("Workflow recovery input is missing.")
    expected_key = f"{input_kind}:{input_id}"
    if input_key != expected_key:
        raise LocalMeetScribeError("Workflow recovery input key is invalid.")
    return input_kind, input_id, input_key


def _try_acquire_workflow_input_lease(
    lock_dir: Path,
    workflow_input_key: str,
) -> _WorkflowInputLease | None:
    lock_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(workflow_input_key.encode("utf-8")).hexdigest()
    handle = (lock_dir / f"{digest}.lock").open("a+b")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(  # type: ignore[attr-defined]
                handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,  # type: ignore[attr-defined]
            )
        return _WorkflowInputLease(handle)
    except OSError:
        handle.close()
        return None


def _background_error_message(exc: Exception) -> str:
    if isinstance(exc, LocalMeetScribeError):
        return str(exc)[:500]
    return f"Unexpected background failure: {type(exc).__name__}"


def _gemini_outputs_complete(package_dir: Path) -> bool:
    try:
        txt_path = package_dir / "gemini_transcript.txt"
        json_path = package_dir / "gemini_transcript.json"
        if not all(path.is_file() and path.stat().st_size > 0 for path in (txt_path, json_path)):
            return False
        transcript = _read_json_object(json_path)
        transcript_text = transcript.get("text")
        raw_transcript_chunks = transcript.get("chunks")
        if not isinstance(transcript_text, str) or not isinstance(raw_transcript_chunks, list):
            return False
        if not raw_transcript_chunks or txt_path.read_text(encoding="utf-8") != transcript_text:
            return False

        manifest = _read_json_object(package_dir / "manifest.json")
        raw_manifest_chunks = manifest.get("chunks")
        if not isinstance(raw_manifest_chunks, list) or not raw_manifest_chunks:
            return False
        expected_filenames = {
            str(item.get("filename") or "")
            for item in raw_manifest_chunks
            if isinstance(item, dict)
        }
        transcript_filenames = {
            str(item.get("filename") or "")
            for item in raw_transcript_chunks
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        }
        return (
            len(expected_filenames) == len(raw_manifest_chunks)
            and len(transcript_filenames) == len(raw_transcript_chunks)
            and "" not in expected_filenames
            and expected_filenames == transcript_filenames
        )
    except (LocalMeetScribeError, OSError, UnicodeError):
        return False


def _optimized_package_complete(package_dir: Path) -> bool:
    try:
        manifest = _read_json_object(package_dir / "manifest.json")
        raw_chunks = manifest.get("chunks")
        if not isinstance(raw_chunks, list) or not raw_chunks:
            return False
        for item in raw_chunks:
            if not isinstance(item, dict):
                return False
            filename = str(item.get("filename") or "")
            if not filename or Path(filename).name != filename:
                return False
            chunk_path = package_dir / filename
            if not chunk_path.is_file() or chunk_path.stat().st_size <= 0:
                return False
        return True
    except (LocalMeetScribeError, OSError):
        return False


def _auto_export_stored_transcript(
    package_dir: Path,
    export_dir: Path | None,
) -> tuple[bool | None, str | None]:
    if export_dir is None:
        return None, None
    try:
        payload = _read_json_object(package_dir / "gemini_transcript.json")
        suggested_filename = str(payload.get("suggested_filename") or "transcript")
    except LocalMeetScribeError:
        suggested_filename = "transcript"
    return _auto_export_transcript(
        package_dir / "gemini_transcript.txt",
        suggested_filename,
        export_dir,
    )


def _auto_export_transcript(
    source_path: Path,
    suggested_filename: str,
    export_dir: Path | None,
) -> tuple[bool | None, str | None]:
    if export_dir is None:
        return None, None

    try:
        export_dir.mkdir(parents=True, exist_ok=True)
        safe_name = _safe_filename(suggested_filename)
        base_name = Path(safe_name).stem if safe_name.casefold().endswith(".txt") else safe_name
        destination = _reserve_unique_export_path(export_dir, base_name, ".txt")
        try:
            shutil.copy2(source_path, destination)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
    except OSError as exc:
        LOGGER.error("Automatic transcript export failed (%s)", type(exc).__name__)
        return False, "서버 PC의 Downloads\\PhoneScribe에 TXT를 저장하지 못했습니다."

    return True, None


def _reserve_unique_export_path(export_dir: Path, base_name: str, suffix: str) -> Path:
    for index in range(1, 10_000):
        numbered_suffix = "" if index == 1 else f"_{index}"
        candidate = export_dir / f"{base_name}{numbered_suffix}{suffix}"
        try:
            descriptor = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            continue
        os.close(descriptor)
        return candidate
    raise OSError("Could not reserve a unique transcript export path")


def _request_system_awake() -> Callable[[], None]:
    if os.name != "nt":
        return lambda: None
    try:
        import ctypes

        execution_state_continuous = 0x80000000
        execution_state_system_required = 0x00000001
        requested = ctypes.windll.kernel32.SetThreadExecutionState(
            execution_state_continuous | execution_state_system_required
        )
        if requested == 0:
            return lambda: None

        def release() -> None:
            ctypes.windll.kernel32.SetThreadExecutionState(execution_state_continuous)

        return release
    except (AttributeError, OSError):
        return lambda: None


def _optimized_package_payload(package_dir: Path, package_id: str) -> dict[str, object]:
    manifest = _read_json_object(package_dir / "manifest.json")
    return {
        "id": package_id,
        "source": manifest.get("source"),
        "recommendation": manifest.get("recommendation"),
        "chunks": manifest.get("chunks"),
        "manifest_url": f"/api/optimizer/packages/{package_id}/manifest.json",
        "package_url": f"/api/optimizer/packages/{package_id}/optimized_package.zip",
    }


def _gemini_progress_payload(
    progress: GeminiTranscriptionProgress,
) -> dict[str, object]:
    return {
        "status": progress.status,
        "completed_chunks": progress.completed_chunks,
        "total_chunks": progress.total_chunks,
        "current_chunk": progress.current_chunk,
        "progress": progress.progress,
        "elapsed_sec": progress.elapsed_sec,
        "eta_sec": progress.eta_sec,
    }


def _stored_gemini_transcript_payload(
    package_dir: Path,
    package_id: str,
) -> dict[str, object]:
    payload = _read_json_object(package_dir / "gemini_transcript.json")
    raw_chunks = payload.get("chunks")
    chunks = raw_chunks if isinstance(raw_chunks, list) else []
    public_chunks = [
        {
            "filename": item.get("filename"),
            "start_sec": item.get("start_sec"),
            "end_sec": item.get("end_sec"),
            "delivery": item.get("delivery"),
            "mime_type": item.get("mime_type"),
        }
        for item in chunks
        if isinstance(item, dict)
    ]
    return {
        "provider": "gemini",
        "model": payload.get("model"),
        "text": payload.get("text"),
        "suggested_filename": payload.get("suggested_filename"),
        "chunk_count": len(public_chunks),
        "chunks": public_chunks,
        "txt_url": f"/api/optimizer/packages/{package_id}/gemini_transcript.txt",
        "json_url": f"/api/optimizer/packages/{package_id}/gemini_transcript.json",
    }


def _resolve_staged_upload(staged_root: Path, upload_id: str) -> tuple[Path, Path]:
    if not re.fullmatch(r"[a-f0-9]{32}", upload_id):
        raise LocalMeetScribeError("Staged upload ID is invalid.")
    upload_dir = staged_root / upload_id
    source_dir = upload_dir / "source"
    if not source_dir.is_dir():
        raise LocalMeetScribeError(
            "The staged upload is no longer available. Select the recording again."
        )
    source_files = [path for path in source_dir.iterdir() if path.is_file()]
    if len(source_files) != 1:
        raise LocalMeetScribeError("The staged upload is incomplete. Select the recording again.")
    return upload_dir, source_files[0]


def _prune_staged_uploads(staged_root: Path) -> None:
    staged_root.mkdir(parents=True, exist_ok=True)
    root = staged_root.resolve()
    cutoff = time.time() - STAGED_UPLOAD_TTL_SEC
    protected_uploads: set[str] = set()
    workflow_dir = staged_root.parent / "workflows"
    if workflow_dir.is_dir():
        for state_path in workflow_dir.glob("*.json"):
            try:
                state = _read_json_object(state_path)
                if (
                    str(state.get("status") or "") in RECOVERABLE_WORKFLOW_STATUSES
                    and state.get("input_kind") == "upload"
                ):
                    protected_uploads.add(str(state.get("input_id") or ""))
            except LocalMeetScribeError:
                continue
    for candidate in staged_root.iterdir():
        if not candidate.is_dir() or not re.fullmatch(r"[a-f0-9]{32}", candidate.name):
            continue
        try:
            resolved = candidate.resolve()
            if (
                resolved.parent != root
                or candidate.name in protected_uploads
                or candidate.stat().st_mtime >= cutoff
            ):
                continue
            shutil.rmtree(resolved, ignore_errors=True)
        except OSError:
            LOGGER.info("Could not prune staged upload %s", candidate.name)


def _is_loopback_request(request: Request) -> bool:
    client_host = request.client.host if request.client else ""
    request_host = request.url.hostname or ""
    return _is_loopback_host(client_host) and _is_loopback_host(request_host)


def _is_loopback_host(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host.casefold() == "localhost"


def _safe_download_filename(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    candidate = _safe_filename(value)[:140].rstrip(" .")
    expected_suffix = Path(fallback).suffix.lower()
    if Path(candidate).suffix.lower() != expected_suffix:
        candidate = f"{Path(candidate).stem}{expected_suffix}"
    return candidate or fallback


def _export_path(transcript: Transcript, kind: ExportKind) -> str | None:
    if kind == "json":
        return transcript.exports.json_path
    return getattr(transcript.exports, kind)


def _mount_frontend(app: FastAPI) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    dist = repo_root / "frontend" / "dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")


app = create_app()
