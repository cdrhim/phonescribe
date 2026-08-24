from __future__ import annotations

from pathlib import Path
from typing import Protocol

from local_meetscribe.utils.errors import MissingDependencyError


class EnhancementEngine(Protocol):
    name: str

    def enhance(self, audio_path: Path, output_dir: Path) -> Path: ...


class NoopEnhancementEngine:
    name = "none"

    def enhance(self, audio_path: Path, output_dir: Path) -> Path:
        del output_dir
        return audio_path


class DeepFilterNetEnhancementEngine:
    name = "deepfilternet"

    def __init__(self) -> None:
        raise MissingDependencyError(
            "deepfilternet",
            "Install the optional [enhance] dependencies and enable denoise only after "
            "ASR testing.",
        )

    def enhance(self, audio_path: Path, output_dir: Path) -> Path:
        del output_dir
        return audio_path
