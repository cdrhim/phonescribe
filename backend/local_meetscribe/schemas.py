from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Mode = Literal["accurate", "fast", "cpu"]
Language = Literal["auto", "ko", "en"]
SegmentLanguage = Literal["ko", "en", "mixed", "unknown"]
JobStatus = Literal["queued", "running", "completed", "failed"]
ExportKind = Literal["json", "md", "txt", "srt", "vtt", "docx", "minutes_md"]


class SourceInfo(BaseModel):
    filename: str
    duration_sec: float = 0.0
    sample_rate: int = 16000
    channels: int = 1


class TranscriptConfig(BaseModel):
    mode: Mode = "accurate"
    asr_engine: str = "mock-asr"
    asr_model: str = "mock"
    diarization_engine: str = "single-speaker"
    language: Language = "auto"
    alignment_engine: str = "segment"
    vad_engine: str = "mock-vad"
    formatter_engine: str = "rule-based"


class Speaker(BaseModel):
    id: str
    display_name: str
    total_sec: float = 0.0


class Word(BaseModel):
    word: str
    start: float
    end: float
    confidence: float | None = None
    speaker: str = "SPEAKER_00"


class TranscriptSegment(BaseModel):
    id: str
    start: float
    end: float
    speaker: str = "SPEAKER_00"
    language: SegmentLanguage = "unknown"
    text_raw: str
    text_clean: str
    confidence: float | None = None
    needs_review: bool = False
    overlap: bool = False
    words: list[Word] = Field(default_factory=list)


class Exports(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    json_path: str | None = Field(default=None, alias="json")
    md: str | None = None
    txt: str | None = None
    srt: str | None = None
    vtt: str | None = None
    docx: str | None = None
    minutes_md: str | None = None


class Transcript(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source: SourceInfo
    config: TranscriptConfig
    speakers: list[Speaker]
    segments: list[TranscriptSegment]
    exports: Exports = Field(default_factory=Exports)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TranscriptionRequest(BaseModel):
    mode: Mode = "accurate"
    language: Language = "auto"
    speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    glossary: list[str] = Field(default_factory=list)
    denoise: bool = False
    loudness_normalize: bool = False
    trim_silence: bool = False
    allow_mock: bool = True

    @field_validator("speakers", "min_speakers", "max_speakers")
    @classmethod
    def positive_speaker_count(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("speaker counts must be positive")
        return value


class JobRecord(BaseModel):
    id: str
    status: JobStatus
    stage: str
    progress: float = 0.0
    source_path: str | None = None
    output_dir: str | None = None
    transcript_path: str | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class SegmentEdit(BaseModel):
    id: str
    text_clean: str


class SpeakerEdit(BaseModel):
    id: str
    display_name: str


class TranscriptPatch(BaseModel):
    segments: list[SegmentEdit] = Field(default_factory=list)
    speakers: list[SpeakerEdit] = Field(default_factory=list)


class EvalReport(BaseModel):
    english_wer: float | None = None
    korean_cer: float | None = None
    korean_spacing_normalized_cer: float | None = None
    segment_timestamp_mae_sec: float | None = None
    diarization_der: float | None = None
    combined_speaker_attributed_error: float | None = None
    warnings: list[str] = Field(default_factory=list)


def load_transcript(path: Path) -> Transcript:
    return Transcript.model_validate_json(path.read_text(encoding="utf-8"))
