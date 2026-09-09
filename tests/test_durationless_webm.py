from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from local_meetscribe.api import app as app_module
from local_meetscribe.pipeline import optimizer as optimizer_module
from local_meetscribe.pipeline.ingest import MediaInfo, probe_media
from local_meetscribe.pipeline.optimizer import (
    OptimizerOverrides,
    OptimizerRequest,
    optimize_audio_package,
)
from local_meetscribe.utils.errors import LocalMeetScribeError

from tests.helpers import make_test_settings
from tests.test_supabase_cloud import (
    RECORDING_ID,
    FakeWorkflowCloudClient,
    write_optimized_fixture,
    write_recoverable_state,
)


def _wait_for_workflow_status(path: Path, expected: str) -> dict[str, object]:
    deadline = time.monotonic() + 2
    payload: dict[str, object] = {}
    while time.monotonic() < deadline:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            time.sleep(0.01)
            continue
        if payload.get("status") == expected:
            return payload
        time.sleep(0.01)
    return payload


def test_probe_media_uses_packet_timestamps_when_webm_has_no_container_duration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "browser-recording.webm"
    source.write_bytes(b"decodable webm")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if "-show_streams" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "streams": [
                            {
                                "codec_type": "audio",
                                "codec_name": "opus",
                                "sample_rate": "48000",
                                "channels": 1,
                            }
                        ],
                        "format": {"format_name": "matroska,webm"},
                    }
                ),
                "",
            )
        if "packet=pts_time,duration_time" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                "pts_time=-0.007000|duration_time=0.020000\n"
                "pts_time=17.800000|duration_time=0.060000\n",
                "",
            )
        pytest.fail(f"unexpected command: {command}")

    monkeypatch.setattr("local_meetscribe.pipeline.ingest.shutil.which", lambda value: value)
    monkeypatch.setattr("local_meetscribe.pipeline.ingest.subprocess.run", fake_run)

    info = probe_media(source, make_test_settings(tmp_path))

    assert info.duration_sec == pytest.approx(17.867)
    assert info.sample_rate == 48000
    assert info.channels == 1
    assert len(calls) == 2


def test_probe_media_decodes_for_duration_when_packets_have_no_timestamps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "browser-recording.webm"
    source.write_bytes(b"decodable webm")

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if "-show_streams" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "streams": [
                            {
                                "codec_type": "audio",
                                "sample_rate": "48000",
                                "channels": 1,
                            }
                        ],
                        "format": {},
                    }
                ),
                "",
            )
        if "packet=pts_time,duration_time" in command:
            return subprocess.CompletedProcess(command, 0, "", "")
        if "-progress" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                "out_time_us=17860000\nout_time_ms=17860000\nprogress=end\n",
                "",
            )
        pytest.fail(f"unexpected command: {command}")

    monkeypatch.setattr("local_meetscribe.pipeline.ingest.shutil.which", lambda value: value)
    monkeypatch.setattr("local_meetscribe.pipeline.ingest.subprocess.run", fake_run)

    info = probe_media(source, make_test_settings(tmp_path))

    assert info.duration_sec == pytest.approx(17.86)


def test_optimizer_refuses_zero_duration_before_writing_a_chunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "browser-recording.webm"
    source.write_bytes(b"nonempty")
    monkeypatch.setattr(
        optimizer_module,
        "probe_media",
        lambda *_args: MediaInfo(source.name, 0.0, 48000, 1),
    )
    monkeypatch.setattr(
        optimizer_module,
        "_run_ffmpeg_chunk",
        lambda **_kwargs: pytest.fail("a zero-duration source must not emit a chunk"),
    )

    with pytest.raises(LocalMeetScribeError, match="no decodable positive duration"):
        optimize_audio_package(
            source,
            tmp_path / "optimized",
            make_test_settings(tmp_path),
            OptimizerRequest(destination="gemini"),
            package_id="zero-duration",
        )


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required for the generated WebM integration fixture",
)
def test_durationless_generated_webm_optimizes_to_positive_audio(tmp_path: Path) -> None:
    source = tmp_path / "browser-recording.webm"
    generated = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.35",
            "-c:a",
            "libopus",
            "-f",
            "webm",
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    assert generated.returncode == 0
    source.write_bytes(generated.stdout)

    settings = make_test_settings(tmp_path)
    original = probe_media(source, settings)
    package = optimize_audio_package(
        source,
        settings.data_dir / "optimized",
        settings,
        OptimizerRequest(
            destination="gemini",
            overrides=OptimizerOverrides(
                remove_silence=False,
                loudnorm=False,
                speech_filter=False,
            ),
        ),
        package_id="durationless-webm",
    )

    assert original.duration_sec > 0.3
    assert package.source.duration_sec > 0.3
    assert len(package.chunks) == 1
    assert package.chunks[0].duration_sec > 0.3
    optimized = probe_media(package.output_dir / package.chunks[0].filename, settings)
    assert optimized.duration_sec > 0.3
    assert app_module._optimized_package_complete(package.output_dir) is True  # noqa: SLF001


def test_recovery_regenerates_legacy_zero_duration_cloud_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = replace(make_test_settings(tmp_path), gemini_api_key="server-gemini-key")
    package_id = "e" * 32
    workflow_id = "f" * 32
    package_dir = settings.data_dir / "optimized" / package_id
    package_dir.mkdir(parents=True)
    (package_dir / "chunk_001.mp3").write_bytes(b"header-only")
    (package_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source": {"filename": "meeting.webm", "duration_sec": 0.0},
                "chunks": [
                    {
                        "filename": "chunk_001.mp3",
                        "start_sec": 0.0,
                        "end_sec": 0.0,
                        "duration_sec": 0.0,
                        "bytes": 225,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    state_path = write_recoverable_state(
        settings,
        workflow_id=workflow_id,
        package_id=package_id,
        status="transcribing",
        input_kind="cloud",
        input_id=RECORDING_ID,
        cloud_recording_id=RECORDING_ID,
    )

    class RecoveryCloud(FakeWorkflowCloudClient):
        def __init__(self) -> None:
            super().__init__()
            self.download_calls = 0

        def download_recording(self, recording_id: str, destination: Path) -> object:
            self.download_calls += 1
            return super().download_recording(recording_id, destination)

    cloud = RecoveryCloud()
    optimize_calls = 0

    def fake_optimize(
        _source: Path,
        output_root: Path,
        _settings: object,
        _request: object,
        *,
        package_id: str,
    ) -> None:
        nonlocal optimize_calls
        optimize_calls += 1
        write_optimized_fixture(output_root / package_id)

    def fake_transcribe(package_dir: Path, *_args: object, **_kwargs: object) -> object:
        write_optimized_fixture(package_dir, transcript=True)
        return SimpleNamespace(
            suggested_filename="meeting",
            txt_path=package_dir / "gemini_transcript.txt",
        )

    monkeypatch.setattr(app_module, "optimize_audio_package", fake_optimize)
    monkeypatch.setattr(app_module, "transcribe_gemini_package", fake_transcribe)

    with TestClient(
        app_module.create_app(settings, supabase_client=cloud)  # type: ignore[arg-type]
    ):
        state = _wait_for_workflow_status(state_path, "complete")

    assert state["status"] == "complete"
    assert cloud.download_calls == 1
    assert optimize_calls == 1
    assert app_module._optimized_package_complete(package_dir) is True  # noqa: SLF001
