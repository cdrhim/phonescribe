from __future__ import annotations

from pathlib import Path

from local_meetscribe.pipeline.export import write_exports
from local_meetscribe.schemas import (
    SourceInfo,
    Speaker,
    Transcript,
    TranscriptConfig,
    TranscriptSegment,
    Word,
)


def test_exports_are_written(tmp_path: Path) -> None:
    transcript = Transcript(
        id="job_export",
        source=SourceInfo(filename="meeting.wav", duration_sec=2.0),
        config=TranscriptConfig(),
        speakers=[Speaker(id="SPEAKER_00", display_name="Speaker 1", total_sec=2.0)],
        segments=[
            TranscriptSegment(
                id="seg_000001",
                start=0.0,
                end=2.0,
                speaker="SPEAKER_00",
                language="en",
                text_raw="hello team",
                text_clean="Hello team.",
                words=[Word(word="Hello", start=0.0, end=0.5, speaker="SPEAKER_00")],
            )
        ],
    )

    exported = write_exports(transcript, tmp_path)

    assert exported.exports.json_path
    assert exported.exports.md
    assert exported.exports.txt
    assert exported.exports.srt
    assert exported.exports.vtt
    assert exported.exports.docx
    assert exported.exports.minutes_md
    assert Path(exported.exports.docx).read_bytes().startswith(b"PK")
    assert "SPEAKER_00: Hello team." in Path(exported.exports.md).read_text(encoding="utf-8")
