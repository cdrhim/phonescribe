from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient
from local_meetscribe.api.app import create_app
from local_meetscribe.security import GeminiShareStore

from tests.helpers import make_test_settings


def test_remote_api_requires_passcode_session(tmp_path: Path) -> None:
    settings = replace(
        make_test_settings(tmp_path),
        gemini_api_key="test-gemini-key",
        remote_access_enabled=True,
        cors_origins=("https://phonescribe.vercel.app",),
    )
    share_store = GeminiShareStore(settings.data_dir)
    share_store.configure_passcode("98769876")
    client = TestClient(create_app(settings), base_url="https://phone.example.ts.net")

    assert client.get("/api/health").status_code == 200
    anonymous_runtime = client.get("/api/runtime")
    assert anonymous_runtime.status_code == 200
    assert anonymous_runtime.json()["remote_session_valid"] is False
    assert anonymous_runtime.json()["access_mode"] is None
    assert client.get("/api/jobs").status_code == 401
    assert client.get("/api/jobs", headers={"Authorization": "Bearer forged"}).status_code == 401
    assert (
        client.post(
            "/api/optimizer/analyze",
            files={"file": ("phone.m4a", b"recording", "audio/mp4")},
        ).status_code
        == 401
    )

    rejected = client.post(
        "/api/gemini-share/verify",
        headers={"X-LocalMeetScribe-Passcode": "1111"},
    )
    assert rejected.status_code == 401

    verified = client.post(
        "/api/gemini-share/verify",
        headers={"X-LocalMeetScribe-Passcode": "98769876"},
    )
    assert verified.status_code == 200
    assert verified.json()["access_mode"] == "record"
    token = verified.json()["access_token"]
    assert token and token != "98769876"
    authorized = {"Authorization": f"Bearer {token}"}
    authorized_runtime = client.get("/api/runtime", headers=authorized).json()
    assert authorized_runtime["remote_session_valid"] is True
    assert authorized_runtime["access_mode"] == "record"
    assert client.get("/api/jobs", headers=authorized).status_code == 200

    package_id = "a" * 32
    package_dir = settings.data_dir / "optimized" / package_id
    package_dir.mkdir(parents=True)
    (package_dir / "manifest.json").write_text(
        json.dumps({"source": {}, "recommendation": {}, "chunks": []}),
        encoding="utf-8",
    )
    resumed_session_workflow = client.post(
        "/api/workflows",
        data={"destination": "gemini", "package_id": package_id},
        headers=authorized,
    )
    assert resumed_session_workflow.status_code == 202

    remote_admin = client.post(
        "/api/admin/gemini-share-key",
        data={"api_key": "replacement-test-key-1234567890"},
        headers={
            **authorized,
            "X-LocalMeetScribe-Passcode": "98769876",
        },
    )
    assert remote_admin.status_code == 403


def test_remote_session_survives_app_restart_without_persisting_plaintext_token(
    tmp_path: Path,
) -> None:
    settings = replace(
        make_test_settings(tmp_path),
        remote_access_enabled=True,
        cors_origins=("https://phonescribe.vercel.app",),
    )
    GeminiShareStore(settings.data_dir).configure_passcode("98769876")

    first_client = TestClient(create_app(settings), base_url="https://phone.example.ts.net")
    verified = first_client.post(
        "/api/gemini-share/verify",
        headers={"X-LocalMeetScribe-Passcode": "98769876"},
    )
    assert verified.status_code == 200
    token = verified.json()["access_token"]

    session_path = settings.data_dir / "config" / "remote_sessions.json"
    stored_text = session_path.read_text(encoding="utf-8")
    stored = json.loads(stored_text)
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    assert token not in stored_text
    assert stored == {
        "sessions": {
            token_hash: {
                "access_mode": "record",
                "expires_at": stored["sessions"][token_hash]["expires_at"],
            }
        },
        "version": 2,
    }
    assert stored["sessions"][token_hash]["expires_at"] > time.time()

    restarted_client = TestClient(create_app(settings), base_url="https://phone.example.ts.net")
    restarted_response = restarted_client.get(
        "/api/jobs",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert restarted_response.status_code == 200


def test_remote_session_store_recovers_from_corrupted_file(tmp_path: Path) -> None:
    settings = replace(
        make_test_settings(tmp_path),
        remote_access_enabled=True,
        cors_origins=("https://phonescribe.vercel.app",),
    )
    GeminiShareStore(settings.data_dir).configure_passcode("98769876")
    session_path = settings.data_dir / "config" / "remote_sessions.json"
    session_path.write_text("{not valid json", encoding="utf-8")

    client = TestClient(create_app(settings), base_url="https://phone.example.ts.net")
    assert (
        client.get(
            "/api/jobs",
            headers={"Authorization": "Bearer previously-issued-token"},
        ).status_code
        == 401
    )
    verified = client.post(
        "/api/gemini-share/verify",
        headers={"X-LocalMeetScribe-Passcode": "98769876"},
    )
    assert verified.status_code == 200
    token = verified.json()["access_token"]
    assert (
        client.get(
            "/api/jobs",
            headers={"Authorization": f"Bearer {token}"},
        ).status_code
        == 200
    )
    assert json.loads(session_path.read_text(encoding="utf-8"))["version"] == 2


def test_expired_remote_sessions_are_rejected_and_pruned(tmp_path: Path) -> None:
    settings = replace(
        make_test_settings(tmp_path),
        remote_access_enabled=True,
        cors_origins=("https://phonescribe.vercel.app",),
    )
    expired_token = "expired-session-token"
    expired_hash = hashlib.sha256(expired_token.encode("utf-8")).hexdigest()
    session_path = settings.data_dir / "config" / "remote_sessions.json"
    session_path.parent.mkdir(parents=True, exist_ok=True)
    session_path.write_text(
        json.dumps(
            {
                "version": 1,
                "sessions": {expired_hash: time.time() - 1},
            }
        ),
        encoding="utf-8",
    )

    client = TestClient(create_app(settings), base_url="https://phone.example.ts.net")
    response = client.get(
        "/api/jobs",
        headers={"Authorization": f"Bearer {expired_token}"},
    )
    assert response.status_code == 401
    assert json.loads(session_path.read_text(encoding="utf-8"))["sessions"] == {}


def test_v1_remote_session_is_restored_as_record_mode(tmp_path: Path) -> None:
    settings = replace(
        make_test_settings(tmp_path),
        remote_access_enabled=True,
        cors_origins=("https://phonescribe.vercel.app",),
    )
    legacy_token = "legacy-v1-session-token"
    legacy_hash = hashlib.sha256(legacy_token.encode("utf-8")).hexdigest()
    session_path = settings.data_dir / "config" / "remote_sessions.json"
    session_path.parent.mkdir(parents=True, exist_ok=True)
    session_path.write_text(
        json.dumps(
            {
                "version": 1,
                "sessions": {legacy_hash: time.time() + 60},
            }
        ),
        encoding="utf-8",
    )

    client = TestClient(create_app(settings), base_url="https://phone.example.ts.net")
    runtime = client.get(
        "/api/runtime",
        headers={"Authorization": f"Bearer {legacy_token}"},
    )

    assert runtime.status_code == 200
    assert runtime.json()["remote_session_valid"] is True
    assert runtime.json()["access_mode"] == "record"


def test_access_codes_issue_mode_specific_durable_sessions(tmp_path: Path) -> None:
    settings = replace(
        make_test_settings(tmp_path),
        remote_access_enabled=True,
        cors_origins=("https://phonescribe.vercel.app",),
    )
    share_store = GeminiShareStore(settings.data_dir)
    share_store.configure_access_codes(record_code="1234", upload_code="5678")
    client = TestClient(create_app(settings), base_url="https://phone.example.ts.net")

    record_response = client.post(
        "/api/gemini-share/verify",
        headers={"X-LocalMeetScribe-Passcode": "1234"},
    )
    upload_response = client.post(
        "/api/gemini-share/verify",
        headers={"X-LocalMeetScribe-Passcode": "5678"},
    )

    assert record_response.status_code == 200
    assert record_response.json()["access_mode"] == "record"
    assert upload_response.status_code == 200
    assert upload_response.json()["access_mode"] == "upload"
    for response, expected_mode in (
        (record_response, "record"),
        (upload_response, "upload"),
    ):
        token = response.json()["access_token"]
        restarted_client = TestClient(
            create_app(settings),
            base_url="https://phone.example.ts.net",
        )
        runtime = restarted_client.get(
            "/api/runtime",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert runtime.status_code == 200
        assert runtime.json()["remote_session_valid"] is True
        assert runtime.json()["access_mode"] == expected_mode


def test_changing_share_passcode_invalidates_persisted_sessions(tmp_path: Path) -> None:
    settings = replace(
        make_test_settings(tmp_path),
        remote_access_enabled=True,
        cors_origins=("https://phonescribe.vercel.app",),
    )
    share_store = GeminiShareStore(settings.data_dir)
    share_store.configure_passcode("98769876")
    client = TestClient(create_app(settings), base_url="https://phone.example.ts.net")
    verified = client.post(
        "/api/gemini-share/verify",
        headers={"X-LocalMeetScribe-Passcode": "98769876"},
    )
    token = verified.json()["access_token"]
    authorized = {"Authorization": f"Bearer {token}"}
    assert client.get("/api/jobs", headers=authorized).status_code == 200

    share_store.configure_passcode("another-secure-passcode")

    assert client.get("/api/jobs", headers=authorized).status_code == 401
    assert not (settings.data_dir / "config" / "remote_sessions.json").exists()


def test_remote_cors_allows_only_configured_frontend(tmp_path: Path) -> None:
    settings = replace(
        make_test_settings(tmp_path),
        remote_access_enabled=True,
        cors_origins=("https://phonescribe.vercel.app",),
    )
    client = TestClient(create_app(settings), base_url="https://phone.example.ts.net")

    allowed = client.options(
        "/api/cloud-recordings/upload-descriptor",
        headers={
            "Origin": "https://phonescribe.vercel.app",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "https://phonescribe.vercel.app"

    unauthorized = client.post(
        "/api/cloud-recordings/upload-descriptor",
        headers={
            "Origin": "https://phonescribe.vercel.app",
            "Authorization": "Bearer expired",
        },
        json={
            "filename": "phone.webm",
            "content_type": "audio/webm",
            "size_bytes": 1,
        },
    )
    assert unauthorized.status_code == 401
    assert (
        unauthorized.headers["access-control-allow-origin"]
        == "https://phonescribe.vercel.app"
    )

    denied = client.options(
        "/api/cloud-recordings/upload-descriptor",
        headers={
            "Origin": "https://attacker.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" not in denied.headers
