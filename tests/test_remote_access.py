from __future__ import annotations

import json
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
    share_store.configure_passcode("35433543")
    client = TestClient(create_app(settings), base_url="https://phone.example.ts.net")

    assert client.get("/api/health").status_code == 200
    assert client.get("/api/runtime").status_code == 200
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
        headers={"X-LocalMeetScribe-Passcode": "0000"},
    )
    assert rejected.status_code == 401

    verified = client.post(
        "/api/gemini-share/verify",
        headers={"X-LocalMeetScribe-Passcode": "35433543"},
    )
    assert verified.status_code == 200
    token = verified.json()["access_token"]
    assert token and token != "35433543"
    authorized = {"Authorization": f"Bearer {token}"}
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
            "X-LocalMeetScribe-Passcode": "35433543",
        },
    )
    assert remote_admin.status_code == 403


def test_remote_cors_allows_only_configured_frontend(tmp_path: Path) -> None:
    settings = replace(
        make_test_settings(tmp_path),
        remote_access_enabled=True,
        cors_origins=("https://phonescribe.vercel.app",),
    )
    client = TestClient(create_app(settings), base_url="https://phone.example.ts.net")

    allowed = client.options(
        "/api/runtime",
        headers={
            "Origin": "https://phonescribe.vercel.app",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "https://phonescribe.vercel.app"

    denied = client.options(
        "/api/runtime",
        headers={
            "Origin": "https://attacker.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert "access-control-allow-origin" not in denied.headers
