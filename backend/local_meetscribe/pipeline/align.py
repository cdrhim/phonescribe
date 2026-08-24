from __future__ import annotations

from pathlib import Path
from typing import Protocol

from local_meetscribe.pipeline.asr import ASRResult, ASRSegment, ASRWord, has_package
from local_meetscribe.schemas import Language
from local_meetscribe.utils.errors import MissingDependencyError


class AlignmentEngine(Protocol):
    name: str

    def align(
        self, audio_path: Path, asr_result: ASRResult, *, language: Language
    ) -> ASRResult: ...


class SegmentAlignmentEngine:
    name = "segment"

    def align(self, audio_path: Path, asr_result: ASRResult, *, language: Language) -> ASRResult:
        del audio_path, language
        aligned: list[ASRSegment] = []
        for segment in asr_result.segments:
            if segment.words:
                aligned.append(segment)
                continue
            words = _words_from_segment(segment)
            aligned.append(
                ASRSegment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    language=segment.language,
                    confidence=segment.confidence,
                    words=words,
                )
            )
        return ASRResult(asr_result.engine_name, asr_result.model_name, aligned)


class WhisperXAlignmentEngine:
    name = "whisperx"

    def __init__(self) -> None:
        if not has_package("whisperx"):
            raise MissingDependencyError(
                "whisperx",
                "Install with `uv pip install -e .[whisper]` and ensure a supported device.",
            )

    def align(self, audio_path: Path, asr_result: ASRResult, *, language: Language) -> ASRResult:
        del audio_path, language
        # WhisperX has several device/model branches. This placeholder keeps the adapter
        # boundary explicit while the segment aligner remains the safe fallback.
        raise MissingDependencyError(
            "whisperx-alignment-adapter",
            "WhisperX is installed, but this adapter needs project-specific model/device wiring.",
        )


def _words_from_segment(segment: ASRSegment) -> list[ASRWord]:
    tokens = [token for token in segment.text.split() if token]
    if not tokens:
        return []
    duration = max(0.001, segment.end - segment.start)
    step = duration / len(tokens)
    return [
        ASRWord(
            word=token,
            start=segment.start + index * step,
            end=segment.start + (index + 1) * step,
            confidence=segment.confidence,
        )
        for index, token in enumerate(tokens)
    ]
