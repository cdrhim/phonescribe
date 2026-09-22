from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

from local_meetscribe import cli
from local_meetscribe.cli import app
from local_meetscribe.security import GeminiShareStore
from typer.testing import CliRunner

from tests.helpers import write_tone_wav


def test_cli_transcribe_uses_mock_engines(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("LOCAL_MEETSCRIBE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LOCAL_MEETSCRIBE_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("LOCAL_MEETSCRIBE_ALLOW_MOCKS", "true")
    monkeypatch.setattr(
        "local_meetscribe.pipeline.orchestrator.has_package",
        lambda _package: False,
    )
    wav_path = write_tone_wav(tmp_path / "meeting.wav")
    out_dir = tmp_path / "out"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "transcribe",
            str(wav_path),
            "--out",
            str(out_dir),
            "--mode",
            "cpu",
            "--language",
            "auto",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "mock-asr" in result.output
    assert (out_dir / "transcript.json").exists()


def test_cli_models_status(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("LOCAL_MEETSCRIBE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LOCAL_MEETSCRIBE_MODELS_DIR", str(tmp_path / "models"))
    runner = CliRunner()

    result = runner.invoke(app, ["models", "status"])

    assert result.exit_code == 0, result.output
    assert "Qwen3-ASR" in result.output or "qwen_asr" in result.output


def test_cli_configures_separate_access_codes(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    data_dir = tmp_path / "data"
    monkeypatch.setattr(cli, "get_settings", lambda: SimpleNamespace(data_dir=data_dir))
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["share", "configure-access-codes"],
        input="1234\n1234\n5678\n5678\n",
    )

    assert result.exit_code == 0, result.output
    store = GeminiShareStore(data_dir)
    assert store.access_mode_for_passcode("1234") == "record"
    assert store.access_mode_for_passcode("5678") == "upload"
    stored_text = store.path.read_text(encoding="utf-8")
    assert "1234" not in stored_text
    assert "5678" not in stored_text


def test_serve_uses_resilient_event_loop_factory(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    run_values: dict[str, object] = {}

    def fake_run(app_value: str, **values: object) -> None:
        run_values["app"] = app_value
        run_values.update(values)

    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=fake_run))

    cli.serve(host="127.0.0.1", port=8766)

    assert run_values["app"] == "local_meetscribe.api.app:app"
    assert run_values["loop"] == "local_meetscribe.cli:_server_loop_factory"
    loop = cli._server_loop_factory()  # noqa: SLF001
    try:
        if sys.platform == "win32":
            assert isinstance(loop, asyncio.SelectorEventLoop)
    finally:
        loop.close()
