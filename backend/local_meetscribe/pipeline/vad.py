from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from local_meetscribe.pipeline.ingest import MediaInfo
from local_meetscribe.utils.errors import MissingDependencyError


@dataclass(frozen=True)
class SpeechRegion:
    start: float
    end: float


class VADEngine(Protocol):
    name: str

    def detect(self, audio_path: Path, media_info: MediaInfo) -> list[SpeechRegion]: ...


class MockVADEngine:
    name = "mock-vad"

    def __init__(self, chunk_size_sec: float = 30.0) -> None:
        self.chunk_size_sec = chunk_size_sec

    def detect(self, audio_path: Path, media_info: MediaInfo) -> list[SpeechRegion]:
        del audio_path
        if media_info.duration_sec <= 0:
            return []
        regions: list[SpeechRegion] = []
        start = 0.0
        while start < media_info.duration_sec:
            end = min(media_info.duration_sec, start + self.chunk_size_sec)
            regions.append(SpeechRegion(start=start, end=end))
            start = end
        return regions


class SileroVADEngine:
    name = "silero-vad"

    def __init__(self) -> None:
        raise MissingDependencyError(
            "silero-vad",
            "Install a Silero VAD integration before selecting this engine.",
        )
