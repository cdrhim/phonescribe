from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import BinaryIO

import pytest
from fastapi.testclient import TestClient
from local_meetscribe.api import app as app_module
from local_meetscribe.cloud.supabase import (
    CloudHTTPResponse,
    CloudRecording,
    CloudTranscriptSegment,
    RetentionCleanupResult,
    SignedRecordingUpload,
    SignedUploadPart,
    SupabaseCloudClient,
    SupabaseCloudError,
)
from local_meetscribe.config import Settings
from local_meetscribe.pipeline.optimizer import OptimizerRequest
from local_meetscribe.security import SupabaseConfigStore
from local_meetscribe.utils.errors import LocalMeetScribeError

from tests.helpers import make_test_settings

MIB = 1024 * 1024
SERVICE_KEY = "sb_secret_test_key_12345678901234567890"
RECORDING_ID = "8b28cc33-7e16-40bb-b42b-30db1e4558a7"
WORKFLOW_ID = "78f1e7f7-bc8a-4eea-972c-d05fc7117080"


class QueueTransport:
    def __init__(self, responses: list[CloudHTTPResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        body: bytes | None,
        timeout: int,
        output: BinaryIO | None = None,
    ) -> CloudHTTPResponse:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "body": body,
                "timeout": timeout,
            }
        )
        response = self.responses.pop(0)
        if output is not None:
            output.write(response.body)
            return CloudHTTPResponse(response.status, b"", response.headers)
        return response


def response(status: int, payload: object | None = None, **headers: str) -> CloudHTTPResponse:
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    return CloudHTTPResponse(status=status, body=body, headers=headers)


def make_client(transport: QueueTransport, **overrides: object) -> SupabaseCloudClient:
    values: dict[str, object] = {
        "url": "https://project-ref.supabase.co",
        "service_role_key": SERVICE_KEY,
        "part_size_bytes": 6 * MIB,
        "transport": transport,
    }
    values.update(overrides)
    return SupabaseCloudClient(**values)  # type: ignore[arg-type]


def test_signed_descriptor_splits_large_audio_into_scoped_put_urls() -> None:
    transport = QueueTransport(
        [
            response(201),
            response(200, {"url": "/object/upload/sign/recordings/a?token=token-0"}),
            response(200, {"url": "/object/upload/sign/recordings/b?token=token-1"}),
            response(200, {"url": "/object/upload/sign/recordings/c?token=token-2"}),
        ]
    )
    client = make_client(transport)

    descriptor = client.create_upload_descriptor(
        filename="phone recording.m4a",
        content_type="application/octet-stream",
        size_bytes=13 * MIB,
    )

    assert descriptor.content_type == "audio/mp4"
    assert [
        (part.part_number, part.byte_start, part.byte_end, part.size_bytes)
        for part in descriptor.parts
    ] == [
        (0, 0, 6 * MIB, 6 * MIB),
        (1, 6 * MIB, 12 * MIB, 6 * MIB),
        (2, 12 * MIB, 13 * MIB, MIB),
    ]
    assert [part.upload_url for part in descriptor.parts] == [
        "https://project-ref.supabase.co/storage/v1/object/upload/sign/recordings/a?token=token-0",
        "https://project-ref.supabase.co/storage/v1/object/upload/sign/recordings/b?token=token-1",
        "https://project-ref.supabase.co/storage/v1/object/upload/sign/recordings/c?token=token-2",
    ]
    insert = json.loads(transport.calls[0]["body"] or b"{}")
    assert insert["part_count"] == 3
    assert insert["object_path"].startswith("shared/")
    assert transport.calls[0]["headers"]["apikey"] == SERVICE_KEY  # type: ignore[index]
    assert "authorization" not in transport.calls[0]["headers"]  # type: ignore[operator]


def test_signed_descriptor_rejects_an_upload_url_for_another_origin() -> None:
    transport = QueueTransport(
        [
            response(201),
            response(200, {"url": "https://example.test/upload?token=not-trusted"}),
            response(204),
        ]
    )
    client = make_client(transport)

    with pytest.raises(SupabaseCloudError, match="invalid signed upload URL"):
        client.create_upload_descriptor(
            filename="meeting.wav",
            content_type="audio/wav",
            size_bytes=MIB,
        )

    failed_update = json.loads(transport.calls[-1]["body"] or b"{}")
    assert failed_update["storage_status"] == "failed"


def test_legacy_service_role_jwt_uses_bearer_authorization() -> None:
    transport = QueueTransport([])
    client = SupabaseCloudClient(
        url="https://project-ref.supabase.co",
        service_role_key="eyJlegacy-service-role-jwt-value",
        transport=transport,
    )

    headers = client._headers()  # noqa: SLF001 - verifies credential compatibility boundary.

    assert headers["apikey"] == "eyJlegacy-service-role-jwt-value"
    assert headers["authorization"] == "Bearer eyJlegacy-service-role-jwt-value"


def test_complete_then_download_reassembles_parts_without_loading_whole_file(
    tmp_path: Path,
) -> None:
    row = {
        "id": RECORDING_ID,
        "object_path": f"shared/{RECORDING_ID}",
        "original_filename": "meeting.webm",
        "mime_type": "audio/webm",
        "size_bytes": 6,
        "part_count": 2,
        "part_size_bytes": 3,
        "file_extension": "webm",
        "storage_status": "pending_upload",
    }
    transport = QueueTransport(
        [
            response(200, [row]),
            response(200, None, **{"content-length": "3"}),
            response(200, None, **{"content-length": "3"}),
            response(204),
            response(200, [{**row, "storage_status": "ready"}]),
            CloudHTTPResponse(200, b"abc", {}),
            CloudHTTPResponse(200, b"def", {}),
        ]
    )
    client = make_client(transport)

    completed = client.complete_recording(RECORDING_ID)
    destination = tmp_path / "meeting.webm"
    downloaded = client.download_recording(RECORDING_ID, destination)

    assert completed.storage_status == "ready"
    assert downloaded.storage_status == "ready"
    assert destination.read_text(encoding="utf-8") == "abcdef"
    get_urls = [
        str(call["url"])
        for call in transport.calls
        if call["method"] == "GET" and "/storage/v1/" in str(call["url"])
    ]
    assert get_urls[0].endswith("/source.part000.webm")
    assert get_urls[1].endswith("/source.part001.webm")


def test_status_and_transcript_persistence_do_not_log_or_mutate_raw_text() -> None:
    job_id = "9028074b-14a5-46a7-bd1f-b2fdeed01e3d"
    transcript_id = "264d43e8-751a-4a2d-bd5a-d9544a1ba1a8"
    transport = QueueTransport(
        [
            response(201),
            response(200, [{"id": job_id}]),
            response(200, []),
            response(201, [{"id": transcript_id}]),
            response(201),
        ]
    )
    client = make_client(transport)

    client.sync_workflow_status(
        recording_id=RECORDING_ID,
        workflow_id=WORKFLOW_ID,
        status="complete",
        stage="complete",
        progress=1.0,
    )
    result = client.persist_transcript(
        recording_id=RECORDING_ID,
        workflow_id=WORKFLOW_ID,
        provider="gemini",
        model_name="test-model",
        text_raw="private transcript",
        segments=[CloudTranscriptSegment(0.0, 2.5, "private transcript")],
        suggested_filename="meeting",
    )

    assert result == transcript_id
    transcript_row = json.loads(transport.calls[3]["body"] or b"{}")
    assert transcript_row["text_raw"] == "private transcript"
    assert transcript_row["text_clean"] == "private transcript"
    segment_rows = json.loads(transport.calls[4]["body"] or b"[]")
    assert segment_rows[0]["start_ms"] == 0
    assert segment_rows[0]["end_ms"] == 2500
    assert segment_rows[0]["text_raw"] == segment_rows[0]["text_clean"]


def test_transcript_retry_fills_segments_for_existing_transcript() -> None:
    job_id = "9028074b-14a5-46a7-bd1f-b2fdeed01e3d"
    transcript_id = "264d43e8-751a-4a2d-bd5a-d9544a1ba1a8"
    transport = QueueTransport(
        [
            response(200, [{"id": job_id}]),
            response(200, [{"id": transcript_id}]),
            response(201),
        ]
    )
    client = make_client(transport)

    result = client.persist_transcript(
        recording_id=RECORDING_ID,
        workflow_id=WORKFLOW_ID,
        provider="gemini",
        model_name="test-model",
        text_raw="private transcript",
        segments=[CloudTranscriptSegment(0.0, 1.0, "private transcript")],
        suggested_filename="meeting",
    )

    assert result == transcript_id
    assert len(transport.calls) == 3
    segment_call = transport.calls[2]
    assert "on_conflict=transcript_id%2Csegment_index" in str(segment_call["url"])
    assert segment_call["headers"]["prefer"] == (  # type: ignore[index]
        "resolution=ignore-duplicates,return=minimal"
    )


def test_transcript_insert_race_recovers_existing_row_and_fills_segments() -> None:
    job_id = "9028074b-14a5-46a7-bd1f-b2fdeed01e3d"
    transcript_id = "264d43e8-751a-4a2d-bd5a-d9544a1ba1a8"
    transport = QueueTransport(
        [
            response(200, [{"id": job_id}]),
            response(200, []),
            response(409, {"message": "duplicate key"}),
            response(200, [{"id": transcript_id}]),
            response(201),
        ]
    )

    result = make_client(transport).persist_transcript(
        recording_id=RECORDING_ID,
        workflow_id=WORKFLOW_ID,
        provider="gemini",
        model_name="test-model",
        text_raw="private transcript",
        segments=[CloudTranscriptSegment(0.0, 1.0, "private transcript")],
        suggested_filename="meeting",
    )

    assert result == transcript_id
    assert len(transport.calls) == 5
    assert transport.calls[3]["method"] == "GET"
    assert transport.calls[4]["headers"]["prefer"] == (  # type: ignore[index]
        "resolution=ignore-duplicates,return=minimal"
    )


def test_supabase_config_store_encrypts_secret(tmp_path: Path) -> None:
    store = SupabaseConfigStore(
        tmp_path,
        protect=lambda value: f"protected:{value[::-1]}",
        unprotect=lambda value: value.removeprefix("protected:")[::-1],
    )

    store.save(
        project_url="https://project-ref.supabase.co/",
        service_role_key=SERVICE_KEY,
        bucket="recordings",
    )

    saved = store.path.read_text(encoding="utf-8")
    loaded = store.load()
    assert store.configured is True
    assert loaded is not None
    assert loaded.project_url == "https://project-ref.supabase.co"
    assert loaded.service_role_key == SERVICE_KEY
    assert SERVICE_KEY not in saved


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI integration test")
def test_windows_dpapi_supabase_secret_round_trip(tmp_path: Path) -> None:
    store = SupabaseConfigStore(tmp_path)

    store.save(
        project_url="https://project-ref.supabase.co",
        service_role_key=SERVICE_KEY,
    )

    loaded = store.load()
    assert loaded is not None
    assert loaded.service_role_key == SERVICE_KEY
    assert SERVICE_KEY not in store.path.read_text(encoding="utf-8")


class FakeWorkflowCloudClient:
    def __init__(self) -> None:
        self.statuses: list[str] = []
        self.persisted = False

    def get_recording(self, _recording_id: str) -> CloudRecording:
        return CloudRecording(
            id=RECORDING_ID,
            object_path=f"shared/{RECORDING_ID}",
            original_filename="meeting.webm",
            mime_type="audio/webm",
            size_bytes=5,
            part_count=1,
            part_size_bytes=24 * MIB,
            file_extension="webm",
            storage_status="ready",
        )

    def download_recording(self, recording_id: str, destination: Path) -> CloudRecording:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"audio")
        return self.get_recording(recording_id)

    def sync_workflow_status(self, **values: object) -> None:
        self.statuses.append(str(values["status"]))

    def persist_transcript(self, **values: object) -> str:
        assert values["text_raw"] == "private transcript"
        self.persisted = True
        return "264d43e8-751a-4a2d-bd5a-d9544a1ba1a8"


class FakeDescriptorCloudClient(FakeWorkflowCloudClient):
    def create_upload_descriptor(self, **_values: object) -> SignedRecordingUpload:
        return SignedRecordingUpload(
            recording_id=RECORDING_ID,
            bucket_id="recordings",
            object_path=f"shared/{RECORDING_ID}",
            content_type="audio/webm",
            parts=(
                SignedUploadPart(
                    part_number=0,
                    byte_start=0,
                    byte_end=5,
                    size_bytes=5,
                    object_path=f"shared/{RECORDING_ID}/source.part000.webm",
                    upload_url=(
                        "https://project-ref.supabase.co/storage/v1/object/upload/sign/"
                        f"recordings/shared/{RECORDING_ID}/source.part000.webm?"
                        "token=scoped-token"
                    ),
                ),
            ),
        )

    def complete_recording(self, recording_id: str) -> CloudRecording:
        return self.get_recording(recording_id)


def write_optimized_fixture(package_dir: Path, *, transcript: bool = False) -> None:
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "chunk_001.mp3").write_bytes(b"audio")
    (package_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source": {"filename": "meeting.webm"},
                "recommendation": {},
                "chunks": [
                    {
                        "filename": "chunk_001.mp3",
                        "start_sec": 0,
                        "end_sec": 2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    if transcript:
        (package_dir / "gemini_transcript.txt").write_text(
            "private transcript",
            encoding="utf-8",
        )
        (package_dir / "gemini_transcript.json").write_text(
            json.dumps(
                {
                    "provider": "gemini",
                    "model": "test-model",
                    "suggested_filename": "meeting",
                    "chunks": [
                        {
                            "filename": "chunk_001.mp3",
                            "start_sec": 0,
                            "end_sec": 2,
                            "text": "private transcript",
                        }
                    ],
                    "text": "private transcript",
                }
            ),
            encoding="utf-8",
        )


def write_recoverable_state(
    settings: Settings,
    *,
    workflow_id: str,
    package_id: str,
    status: str,
    input_kind: str,
    input_id: str,
    cloud_recording_id: str | None = None,
) -> Path:
    state_dir = settings.tmp_dir / "workflows"
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / f"{workflow_id}.json"
    payload: dict[str, object] = {
        "schema_version": 2,
        "workflow_id": workflow_id,
        "package_id": package_id,
        "status": status,
        "error": None,
        "input_kind": input_kind,
        "input_id": input_id,
        "workflow_input_key": f"{input_kind}:{input_id}",
        "optimizer_request": {
            "destination": "gemini",
            "openai_model": "gpt-4o-transcribe",
            "word_timestamps": False,
            "overrides": {
                "codec": "mp3",
                "bitrate_kbps": 32,
                "chunk_minutes": 30,
                "remove_silence": True,
                "loudnorm": True,
                "speech_filter": True,
                "denoise": False,
            },
        },
        "credential_mode": "server",
        "attempt_count": 0,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    if cloud_recording_id:
        payload["cloud_recording_id"] = cloud_recording_id
        payload["cloud_sync_complete"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def wait_for_workflow_status(
    path: Path,
    expected: str,
    *,
    timeout: float = 2.0,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") == expected:
                return payload
        time.sleep(0.01)
    return json.loads(path.read_text(encoding="utf-8"))


def test_completed_gemini_artifacts_require_valid_json_and_full_chunk_coverage(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "package"
    write_optimized_fixture(package_dir, transcript=True)
    transcript_path = package_dir / "gemini_transcript.json"

    assert app_module._gemini_outputs_complete(package_dir) is True  # noqa: SLF001

    transcript_path.write_text("{not valid json", encoding="utf-8")
    assert app_module._gemini_outputs_complete(package_dir) is False  # noqa: SLF001

    transcript_path.write_text(
        json.dumps(
            {
                "provider": "gemini",
                "model": "test-model",
                "suggested_filename": "meeting",
                "chunks": [
                    {
                        "filename": "different_chunk.mp3",
                        "start_sec": 0,
                        "end_sec": 2,
                        "text": "private transcript",
                    }
                ],
                "text": "private transcript",
            }
        ),
        encoding="utf-8",
    )
    assert app_module._gemini_outputs_complete(package_dir) is False  # noqa: SLF001


def test_cloud_descriptor_api_matches_frontend_signed_put_contract(tmp_path: Path) -> None:
    client = TestClient(
        app_module.create_app(
            make_test_settings(tmp_path),
            supabase_client=FakeDescriptorCloudClient(),  # type: ignore[arg-type]
        )
    )

    response_value = client.post(
        "/api/cloud-recordings/upload-descriptor",
        json={"filename": "meeting.webm", "content_type": "", "size_bytes": 5},
    )

    assert response_value.status_code == 201
    payload = response_value.json()
    assert payload["recording_id"] == RECORDING_ID
    assert payload["expires_in"] == 7200
    assert payload["parts"][0]["byte_start"] == 0
    assert payload["parts"][0]["byte_end"] == 5
    assert payload["parts"][0]["upload"] == {
        "protocol": "signed-put",
        "url": (
            "https://project-ref.supabase.co/storage/v1/object/upload/sign/"
            f"recordings/shared/{RECORDING_ID}/source.part000.webm?token=scoped-token"
        ),
        "headers": {
            "content-type": "audio/webm",
            "cache-control": "max-age=3600",
            "x-upsert": "false",
        },
    }


def test_descriptor_schedules_throttled_cleanup_without_blocking(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()

    class MaintenanceCloudClient(FakeDescriptorCloudClient):
        def __init__(self) -> None:
            super().__init__()
            self.cleanup_calls = 0

        def try_cleanup_expired_recordings(self, *, limit: int) -> RetentionCleanupResult:
            assert limit == 25
            self.cleanup_calls += 1
            started.set()
            release.wait(timeout=2)
            return RetentionCleanupResult(attempted=0, deleted=0, failed=0)

    cloud = MaintenanceCloudClient()
    request_body = {
        "filename": "meeting.webm",
        "content_type": "audio/webm",
        "size_bytes": 5,
    }
    try:
        with TestClient(
            app_module.create_app(
                make_test_settings(tmp_path),
                supabase_client=cloud,  # type: ignore[arg-type]
            )
        ) as client:
            first = client.post("/api/cloud-recordings/upload-descriptor", json=request_body)
            assert first.status_code == 201
            assert started.wait(timeout=1)

            second = client.post("/api/cloud-recordings/upload-descriptor", json=request_body)
            assert second.status_code == 201
            assert cloud.cleanup_calls == 1
    finally:
        release.set()


def test_cloud_recording_runs_local_workflow_and_syncs_status(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    settings = replace(make_test_settings(tmp_path), gemini_api_key="test-gemini-key")
    cloud = FakeWorkflowCloudClient()

    def fake_optimize(
        _source: Path,
        output_root: Path,
        _settings: object,
        _request: object,
        *,
        package_id: str,
    ) -> None:
        package_dir = output_root / package_id
        package_dir.mkdir(parents=True)
        (package_dir / "chunk_001.mp3").write_bytes(b"audio")
        (package_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "source": {"filename": "meeting.webm"},
                    "recommendation": {},
                    "chunks": [
                        {
                            "filename": "chunk_001.mp3",
                            "start_sec": 0,
                            "end_sec": 2,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    def fake_transcribe(package_dir: Path, *_args: object, **_kwargs: object) -> object:
        chunk = SimpleNamespace(start_sec=0.0, end_sec=2.0, text="private transcript")
        txt_path = package_dir / "gemini_transcript.txt"
        json_path = package_dir / "gemini_transcript.json"
        txt_path.write_text("private transcript", encoding="utf-8")
        json_path.write_text(
            json.dumps(
                {
                    "provider": "gemini",
                    "model": "test-model",
                    "suggested_filename": "meeting",
                    "chunks": [asdict(CloudTranscriptSegment(0, 2, "private transcript"))],
                    "text": "private transcript",
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(
            provider="gemini",
            model="test-model",
            text="private transcript",
            suggested_filename="meeting",
            chunks=[chunk],
            txt_path=txt_path,
            json_path=json_path,
        )

    monkeypatch.setattr(app_module, "optimize_audio_package", fake_optimize)
    monkeypatch.setattr(app_module, "transcribe_gemini_package", fake_transcribe)
    client = TestClient(
        app_module.create_app(
            settings,
            supabase_client=cloud,  # type: ignore[arg-type]
        )
    )

    started = client.post(
        "/api/workflows",
        data={"destination": "gemini", "cloud_recording_id": RECORDING_ID},
    )

    assert started.status_code == 202, started.text
    workflow_id = started.json()["workflow_id"]
    state = json.loads(
        (settings.tmp_dir / "workflows" / f"{workflow_id}.json").read_text(encoding="utf-8")
    )
    assert state["status"] == "complete"
    assert state["cloud_recording_id"] == RECORDING_ID
    assert state["input_kind"] == "cloud"
    assert state["optimizer_request"]["destination"] == "gemini"
    assert "test-gemini-key" not in json.dumps(state)
    assert cloud.statuses == ["queued", "optimizing", "transcribing", "complete"]
    assert cloud.persisted is True


def test_startup_recovers_queued_cloud_workflow_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = replace(make_test_settings(tmp_path), gemini_api_key="server-gemini-key")
    package_id = "a" * 32
    workflow_id = "b" * 32
    state_path = write_recoverable_state(
        settings,
        workflow_id=workflow_id,
        package_id=package_id,
        status="queued",
        input_kind="cloud",
        input_id=RECORDING_ID,
        cloud_recording_id=RECORDING_ID,
    )

    class RecoveryCloud(FakeWorkflowCloudClient):
        def __init__(self) -> None:
            super().__init__()
            self.download_calls = 0

        def download_recording(
            self,
            recording_id: str,
            destination: Path,
        ) -> CloudRecording:
            self.download_calls += 1
            return super().download_recording(recording_id, destination)

    cloud = RecoveryCloud()
    optimize_calls = 0
    transcribe_calls = 0

    def fake_optimize(
        _source: Path,
        output_root: Path,
        _settings: object,
        request_value: OptimizerRequest,
        *,
        package_id: str,
    ) -> None:
        nonlocal optimize_calls
        optimize_calls += 1
        assert request_value.destination == "gemini"
        write_optimized_fixture(output_root / package_id)

    def fake_transcribe(
        package_dir: Path,
        _settings: object,
        *,
        api_key: str,
    ) -> object:
        nonlocal transcribe_calls
        transcribe_calls += 1
        assert api_key == "server-gemini-key"
        write_optimized_fixture(package_dir, transcript=True)
        chunk = SimpleNamespace(start_sec=0.0, end_sec=2.0, text="private transcript")
        return SimpleNamespace(
            provider="gemini",
            model="test-model",
            text="private transcript",
            suggested_filename="meeting",
            chunks=[chunk],
            txt_path=package_dir / "gemini_transcript.txt",
            json_path=package_dir / "gemini_transcript.json",
        )

    monkeypatch.setattr(app_module, "optimize_audio_package", fake_optimize)
    monkeypatch.setattr(app_module, "transcribe_gemini_package", fake_transcribe)
    with TestClient(
        app_module.create_app(settings, supabase_client=cloud)  # type: ignore[arg-type]
    ):
        state = wait_for_workflow_status(state_path, "complete")
        deadline = time.monotonic() + 1
        while not state.get("cloud_sync_complete") and time.monotonic() < deadline:
            time.sleep(0.01)
            state = json.loads(state_path.read_text(encoding="utf-8"))

    assert state["status"] == "complete"
    assert state["recovered_after_restart"] is True
    assert state["cloud_sync_complete"] is True
    assert cloud.download_calls == 1
    assert optimize_calls == 1
    assert transcribe_calls == 1
    assert cloud.persisted is True


def test_startup_recovers_local_upload_with_original_optimizer_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = replace(make_test_settings(tmp_path), gemini_api_key="server-gemini-key")
    package_id = "c" * 32
    workflow_id = "d" * 32
    upload_id = "e" * 32
    staged_dir = settings.tmp_dir / "optimizer-uploads" / upload_id
    source_dir = staged_dir / "source"
    source_dir.mkdir(parents=True)
    (source_dir / "meeting.m4a").write_bytes(b"audio")
    state_path = write_recoverable_state(
        settings,
        workflow_id=workflow_id,
        package_id=package_id,
        status="optimizing",
        input_kind="upload",
        input_id=upload_id,
    )
    seen_chunk_minutes: list[float | None] = []

    def fake_optimize(
        _source: Path,
        output_root: Path,
        _settings: object,
        request_value: OptimizerRequest,
        *,
        package_id: str,
    ) -> None:
        seen_chunk_minutes.append(request_value.overrides.chunk_minutes)
        write_optimized_fixture(output_root / package_id)

    def fake_transcribe(package_dir: Path, *_args: object, **_kwargs: object) -> object:
        write_optimized_fixture(package_dir, transcript=True)
        return SimpleNamespace(
            provider="gemini",
            model="test-model",
            text="private transcript",
            suggested_filename="meeting",
            chunks=[SimpleNamespace(start_sec=0.0, end_sec=2.0, text="private transcript")],
            txt_path=package_dir / "gemini_transcript.txt",
            json_path=package_dir / "gemini_transcript.json",
        )

    monkeypatch.setattr(app_module, "optimize_audio_package", fake_optimize)
    monkeypatch.setattr(app_module, "transcribe_gemini_package", fake_transcribe)
    with TestClient(app_module.create_app(settings)):
        state = wait_for_workflow_status(state_path, "complete")

    assert state["status"] == "complete"
    assert seen_chunk_minutes == [30.0]
    assert not staged_dir.exists()


def test_startup_transcribing_recovery_skips_completed_optimization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = replace(make_test_settings(tmp_path), gemini_api_key="server-gemini-key")
    package_id = "1" * 32
    workflow_id = "2" * 32
    upload_id = "3" * 32
    package_dir = settings.data_dir / "optimized" / package_id
    write_optimized_fixture(package_dir)
    state_path = write_recoverable_state(
        settings,
        workflow_id=workflow_id,
        package_id=package_id,
        status="transcribing",
        input_kind="upload",
        input_id=upload_id,
    )
    monkeypatch.setattr(
        app_module,
        "optimize_audio_package",
        lambda *_args, **_kwargs: pytest.fail("completed optimization must be reused"),
    )

    def fake_transcribe(package_dir: Path, *_args: object, **_kwargs: object) -> object:
        write_optimized_fixture(package_dir, transcript=True)
        return SimpleNamespace(
            provider="gemini",
            model="test-model",
            text="private transcript",
            suggested_filename="meeting",
            chunks=[SimpleNamespace(start_sec=0.0, end_sec=2.0, text="private transcript")],
            txt_path=package_dir / "gemini_transcript.txt",
            json_path=package_dir / "gemini_transcript.json",
        )

    monkeypatch.setattr(app_module, "transcribe_gemini_package", fake_transcribe)
    with TestClient(app_module.create_app(settings)):
        state = wait_for_workflow_status(state_path, "complete")

    assert state["status"] == "complete"


def test_startup_marks_request_only_key_recovery_as_actionable_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_test_settings(tmp_path)
    package_id = "4" * 32
    workflow_id = "5" * 32
    package_dir = settings.data_dir / "optimized" / package_id
    write_optimized_fixture(package_dir)
    state_path = write_recoverable_state(
        settings,
        workflow_id=workflow_id,
        package_id=package_id,
        status="transcribing",
        input_kind="package",
        input_id=package_id,
    )
    monkeypatch.setattr(
        app_module,
        "transcribe_gemini_package",
        lambda *_args, **_kwargs: pytest.fail("Gemini must not run without a stored key"),
    )

    with TestClient(app_module.create_app(settings)):
        state = wait_for_workflow_status(state_path, "failed")

    assert state["error_code"] == "recovery_key_unavailable"
    assert state["requires_api_key"] is True
    assert "API key" in str(state["error"])
    assert "server-gemini-key" not in state_path.read_text(encoding="utf-8")


def test_completed_cloud_workflow_outbox_retries_without_retranscribing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_test_settings(tmp_path)
    package_id = "6" * 32
    workflow_id = "7" * 32
    write_optimized_fixture(settings.data_dir / "optimized" / package_id, transcript=True)
    state_path = write_recoverable_state(
        settings,
        workflow_id=workflow_id,
        package_id=package_id,
        status="complete",
        input_kind="cloud",
        input_id=RECORDING_ID,
        cloud_recording_id=RECORDING_ID,
    )
    first_failed = threading.Event()
    retry_started = threading.Event()
    allow_retry = threading.Event()

    class RetryCloud(FakeWorkflowCloudClient):
        def __init__(self) -> None:
            super().__init__()
            self.persist_calls = 0

        def persist_transcript(self, **values: object) -> str:
            self.persist_calls += 1
            if self.persist_calls == 1:
                first_failed.set()
                raise SupabaseCloudError("temporary cloud failure", status=503)
            retry_started.set()
            allow_retry.wait(timeout=2)
            return super().persist_transcript(**values)

    cloud = RetryCloud()
    monkeypatch.setattr(app_module, "CLOUD_MAINTENANCE_INTERVAL_SEC", 0.02)
    monkeypatch.setattr(
        app_module,
        "transcribe_gemini_package",
        lambda *_args, **_kwargs: pytest.fail("terminal outbox recovery must not transcribe"),
    )
    outbox_path = settings.tmp_dir / "cloud-outbox" / f"{workflow_id}.json"

    with TestClient(
        app_module.create_app(settings, supabase_client=cloud)  # type: ignore[arg-type]
    ):
        assert first_failed.wait(timeout=1)
        assert retry_started.wait(timeout=1)
        assert outbox_path.exists()
        assert "private transcript" not in outbox_path.read_text(encoding="utf-8")
        allow_retry.set()
        deadline = time.monotonic() + 1
        while outbox_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert cloud.persist_calls >= 2
    assert not outbox_path.exists()
    assert state["cloud_sync_complete"] is True


def test_startup_retention_runs_during_first_minutes_after_os_boot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_started = threading.Event()

    class StartupCloud(FakeWorkflowCloudClient):
        def try_cleanup_expired_recordings(self, *, limit: int) -> RetentionCleanupResult:
            assert limit == 25
            cleanup_started.set()
            return RetentionCleanupResult(attempted=0, deleted=0, failed=0)

    class LowUptimeClock:
        @staticmethod
        def monotonic() -> float:
            return 5.0

        @staticmethod
        def time() -> float:
            return time.time()

        @staticmethod
        def sleep(seconds: float) -> None:
            time.sleep(seconds)

    monkeypatch.setattr(app_module, "time", LowUptimeClock())

    with TestClient(
        app_module.create_app(
            make_test_settings(tmp_path),
            supabase_client=StartupCloud(),  # type: ignore[arg-type]
        )
    ):
        assert cleanup_started.wait(timeout=1)


def test_startup_and_periodic_retention_cleanup_stop_with_lifespan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PeriodicCloud(FakeWorkflowCloudClient):
        def __init__(self) -> None:
            super().__init__()
            self.cleanup_calls = 0
            self.cleaned_twice = threading.Event()

        def try_cleanup_expired_recordings(self, *, limit: int) -> RetentionCleanupResult:
            assert limit == 25
            self.cleanup_calls += 1
            if self.cleanup_calls >= 2:
                self.cleaned_twice.set()
            return RetentionCleanupResult(attempted=0, deleted=0, failed=0)

    monkeypatch.setattr(app_module, "CLOUD_MAINTENANCE_INTERVAL_SEC", 0.01)
    monkeypatch.setattr(app_module, "CLOUD_CLEANUP_INTERVAL_SEC", 0.02)
    cloud = PeriodicCloud()

    with TestClient(
        app_module.create_app(
            make_test_settings(tmp_path),
            supabase_client=cloud,  # type: ignore[arg-type]
        )
    ):
        assert cloud.cleaned_twice.wait(timeout=1)

    stopped_at = cloud.cleanup_calls
    time.sleep(0.05)
    assert cloud.cleanup_calls == stopped_at


def test_workflow_input_os_lease_blocks_duplicates_and_releases(tmp_path: Path) -> None:
    lock_dir = tmp_path / "workflow-locks"
    first = app_module._try_acquire_workflow_input_lease(  # noqa: SLF001
        lock_dir,
        "upload:" + "8" * 32,
    )
    assert first is not None
    assert (
        app_module._try_acquire_workflow_input_lease(  # noqa: SLF001
            lock_dir,
            "upload:" + "8" * 32,
        )
        is None
    )
    first.release()
    reacquired = app_module._try_acquire_workflow_input_lease(  # noqa: SLF001
        lock_dir,
        "upload:" + "8" * 32,
    )
    assert reacquired is not None
    reacquired.release()


def test_runtime_and_descriptor_gracefully_fall_back_when_cloud_is_disabled(
    tmp_path: Path,
) -> None:
    client = TestClient(app_module.create_app(make_test_settings(tmp_path)))

    runtime = client.get("/api/runtime")
    descriptor = client.post(
        "/api/cloud-recordings/upload-descriptor",
        json={"filename": "meeting.m4a", "content_type": "audio/mp4", "size_bytes": 10},
    )

    assert runtime.status_code == 200
    assert runtime.json()["cloud_upload_enabled"] is False
    assert descriptor.status_code == 503


def test_retention_cleanup_retries_deleting_rows_and_preserves_metadata() -> None:
    stuck_id = "00a2546d-1aed-4c93-a6fc-bd904d31d5bc"
    claimed_id = "6398a75d-ea44-45d7-acf2-d53b92527c89"

    def deletion_row(recording_id: str) -> dict[str, object]:
        return {
            "id": recording_id,
            "object_path": f"shared/{recording_id}",
            "original_filename": "meeting.m4a",
            "mime_type": "audio/mp4",
            "size_bytes": 8,
            "part_count": 2,
            "part_size_bytes": 4,
            "file_extension": "m4a",
            "storage_status": "deleting",
        }

    transport = QueueTransport(
        [
            response(200, [deletion_row(stuck_id), deletion_row(claimed_id)]),
            response(200, []),
            response(200, True),
            response(500, {"message": "temporary storage failure"}),
            response(200, True),
        ]
    )
    client = make_client(transport)

    result = client.cleanup_expired_recordings(limit=2)

    assert result.attempted == 2
    assert result.deleted == 1
    assert result.failed == 1
    rpc_call = transport.calls[0]
    assert rpc_call["url"] == (
        "https://project-ref.supabase.co/rest/v1/rpc/claim_expired_recordings"
    )
    assert json.loads(rpc_call["body"] or b"{}")["p_limit"] == 2
    delete_call = transport.calls[1]
    assert delete_call["method"] == "DELETE"
    assert delete_call["url"] == ("https://project-ref.supabase.co/storage/v1/object/recordings")
    assert json.loads(delete_call["body"] or b"{}")["prefixes"] == [
        f"shared/{stuck_id}/source.part000.m4a",
        f"shared/{stuck_id}/source.part001.m4a",
    ]
    completion_call = transport.calls[2]
    assert completion_call["url"] == (
        "https://project-ref.supabase.co/rest/v1/rpc/complete_recording_retention"
    )
    assert json.loads(completion_call["body"] or b"{}")["p_recording_id"] == stuck_id
    failed_release = transport.calls[4]
    assert failed_release["url"] == (
        "https://project-ref.supabase.co/rest/v1/rpc/fail_recording_retention"
    )
    assert json.loads(failed_release["body"] or b"{}")["p_error"] == ("storage_delete_failed")
    assert not any(
        call["method"] in {"DELETE", "PATCH"} and "/rest/v1/recordings" in str(call["url"])
        for call in transport.calls
    )

    retry_transport = QueueTransport(
        [
            response(200, [deletion_row(claimed_id)]),
            response(200, []),
            response(200, True),
        ]
    )
    retried = make_client(retry_transport).cleanup_expired_recordings(limit=1)
    assert retried.attempted == 1
    assert retried.deleted == 1
    assert "claim_expired_recordings" in str(retry_transport.calls[0]["url"])


def test_supabase_bucket_is_fixed_to_recordings(tmp_path: Path) -> None:
    with pytest.raises(SupabaseCloudError, match="must be 'recordings'"):
        SupabaseCloudClient(
            url="https://project-ref.supabase.co",
            service_role_key=SERVICE_KEY,
            bucket="other-bucket",
        )

    store = SupabaseConfigStore(
        tmp_path,
        protect=lambda value: f"protected:{value}",
        unprotect=lambda value: value.removeprefix("protected:"),
    )
    with pytest.raises(LocalMeetScribeError, match="must be 'recordings'"):
        store.save(
            project_url="https://project-ref.supabase.co",
            service_role_key=SERVICE_KEY,
            bucket="other-bucket",
        )


def test_retention_object_deletes_are_batched() -> None:
    row = {
        "id": RECORDING_ID,
        "object_path": f"shared/{RECORDING_ID}",
        "original_filename": "meeting.m4a",
        "mime_type": "audio/mp4",
        "size_bytes": 101,
        "part_count": 101,
        "part_size_bytes": 1,
        "file_extension": "m4a",
        "storage_status": "deleting",
    }
    transport = QueueTransport(
        [
            response(200, [row]),
            response(200, []),
            response(200, []),
            response(200, True),
        ]
    )

    result = make_client(transport).cleanup_expired_recordings(limit=1)

    assert result.deleted == 1
    delete_calls = [call for call in transport.calls if call["method"] == "DELETE"]
    assert len(delete_calls) == 2
    assert len(json.loads(delete_calls[0]["body"] or b"{}")["prefixes"]) == 100
    assert len(json.loads(delete_calls[1]["body"] or b"{}")["prefixes"]) == 1
