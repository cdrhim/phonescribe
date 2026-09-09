from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from local_meetscribe.api.app import create_app
from local_meetscribe.pipeline.derivatives import (
    create_transcript_artifact,
    organize_transcript,
    read_stored_transcript_artifacts,
    summarize_transcript,
)
from local_meetscribe.utils.errors import LocalMeetScribeError

from tests.helpers import make_test_settings

SAMPLE_TRANSCRIPT = "\n".join(
    (
        "[00:00] SPEAKER_01: 오늘 목표는 신규 녹음 흐름을 간단하게 만드는 것입니다.",
        "[00:15] SPEAKER_02: 사용자는 녹음 종료 뒤 다른 버튼을 누르지 않아야 합니다.",
        "[00:28] SPEAKER_01: 원문 TXT는 자동으로 준비하기로 결정했습니다.",
        "[00:41] SPEAKER_02: 정리본과 요약본은 사용자가 원할 때만 만듭니다.",
        "[00:56] SPEAKER_01: 다음 일정은 9월 12일까지 모바일 테스트를 완료하는 것입니다.",
        "[01:10] SPEAKER_02: 제가 다운로드 버튼 작동을 검토하겠습니다.",
        "[01:24] SPEAKER_01: 네.",
        "[01:29] SPEAKER_02: 파일명 변경은 고급 메뉴에 두면 화면이 더 깔끔해집니다.",
        "[01:43] SPEAKER_01: 완료 화면에서는 미팅록 일부만 미리 보여줍니다.",
        "[01:55] SPEAKER_02: 전체 내용은 다운로드한 TXT에 유지됩니다.",
        "",
    )
)


def test_organized_version_preserves_each_spoken_line() -> None:
    organized = organize_transcript(SAMPLE_TRANSCRIPT)

    assert organized.startswith("미팅 정리본\n")
    assert "[00:00] · SPEAKER_01" in organized
    for spoken_line in (
        "오늘 목표는 신규 녹음 흐름을 간단하게 만드는 것입니다.",
        "원문 TXT는 자동으로 준비하기로 결정했습니다.",
        "전체 내용은 다운로드한 TXT에 유지됩니다.",
    ):
        assert spoken_line in organized


def test_summary_is_shorter_and_uses_only_source_sentences() -> None:
    summary = summarize_transcript(SAMPLE_TRANSCRIPT)

    assert summary.startswith("미팅 요약본\n")
    assert len(summary) < len(SAMPLE_TRANSCRIPT)
    assert "결정·후속" in summary
    bullet_lines = [line[2:] for line in summary.splitlines() if line.startswith("- ")]
    assert bullet_lines
    assert all(sentence in SAMPLE_TRANSCRIPT for sentence in bullet_lines)
    assert "네." not in bullet_lines


def test_artifacts_are_separate_and_raw_transcript_is_immutable(tmp_path: Path) -> None:
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    raw_path = package_dir / "gemini_transcript.txt"
    raw_path.write_text(SAMPLE_TRANSCRIPT, encoding="utf-8")
    before = hashlib.sha256(raw_path.read_bytes()).hexdigest()

    organized = create_transcript_artifact(package_dir, "organized")
    summary = create_transcript_artifact(package_dir, "summary")
    summary_bytes = summary.txt_path.read_bytes()
    repeated_summary = create_transcript_artifact(package_dir, "summary")

    assert organized.txt_path.name == "meeting_organized.txt"
    assert summary.txt_path.name == "meeting_summary.txt"
    assert repeated_summary.text == summary.text
    assert repeated_summary.txt_path.read_bytes() == summary_bytes
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == before
    stored = read_stored_transcript_artifacts(package_dir)
    assert set(stored) == {"organized", "summary"}
    assert stored["organized"].source_sha256 == before


def test_artifact_rejects_unknown_kind(tmp_path: Path) -> None:
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    (package_dir / "gemini_transcript.txt").write_text(SAMPLE_TRANSCRIPT, encoding="utf-8")

    with pytest.raises(LocalMeetScribeError, match="Unsupported"):
        create_transcript_artifact(package_dir, "unknown")


def test_artifact_api_returns_preview_and_download_without_changing_raw(tmp_path: Path) -> None:
    settings = make_test_settings(tmp_path)
    package_id = "a" * 32
    package_dir = settings.data_dir / "optimized" / package_id
    package_dir.mkdir(parents=True)
    raw_path = package_dir / "gemini_transcript.txt"
    raw_path.write_text(SAMPLE_TRANSCRIPT, encoding="utf-8")
    before = raw_path.read_bytes()
    client = TestClient(create_app(settings))

    response = client.post(
        f"/api/optimizer/packages/{package_id}/transcript-artifacts/summary"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] == "summary"
    assert payload["text"].startswith("미팅 요약본")
    assert payload["txt_url"].endswith("/meeting_summary.txt")
    download = client.get(payload["txt_url"])
    assert download.status_code == 200
    assert download.text == payload["text"]
    assert raw_path.read_bytes() == before

    unsupported = client.post(
        f"/api/optimizer/packages/{package_id}/transcript-artifacts/other"
    )
    assert unsupported.status_code == 404
