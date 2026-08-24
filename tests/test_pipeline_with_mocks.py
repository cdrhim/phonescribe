from __future__ import annotations

from pathlib import Path

from local_meetscribe.pipeline.orchestrator import TranscriptionPipeline
from local_meetscribe.schemas import TranscriptionRequest, load_transcript

from tests.helpers import make_test_settings, write_tone_wav


def test_mock_pipeline_runs_without_models(tmp_path: Path) -> None:
    wav_path = write_tone_wav(tmp_path / "meeting.wav")
    settings = make_test_settings(tmp_path)
    out_dir = tmp_path / "out"
    pipeline = TranscriptionPipeline(settings)

    transcript = pipeline.run(
        wav_path,
        output_dir=out_dir,
        request=TranscriptionRequest(mode="accurate", language="auto"),
        job_id="job_test",
    )

    assert transcript.id == "job_test"
    assert transcript.config.asr_engine == "mock-asr"
    assert transcript.config.diarization_engine == "single-speaker"
    assert transcript.source.sample_rate == 16000
    assert transcript.segments
    assert transcript.segments[0].text_raw
    assert transcript.segments[0].text_clean
    assert transcript.segments[0].needs_review is True
    assert transcript.exports.json_path is not None
    assert Path(transcript.exports.json_path).exists()

    persisted = load_transcript(Path(transcript.exports.json_path))
    assert persisted.exports.md is not None
    assert Path(persisted.exports.md).exists()
