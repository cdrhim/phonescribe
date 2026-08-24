from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from local_meetscribe.pipeline.asr import has_package
from local_meetscribe.pipeline.ingest import MediaInfo
from local_meetscribe.schemas import TranscriptionRequest
from local_meetscribe.utils.errors import MissingDependencyError, ModelUnavailableError

PYANNOTE_COMMUNITY_1 = "pyannote/speaker-diarization-community-1"


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str


class DiarizationEngine(Protocol):
    name: str

    def diarize(
        self,
        audio_path: Path,
        media_info: MediaInfo,
        request: TranscriptionRequest,
    ) -> list[SpeakerTurn]: ...


class SingleSpeakerDiarizationEngine:
    name = "single-speaker"

    def diarize(
        self,
        audio_path: Path,
        media_info: MediaInfo,
        request: TranscriptionRequest,
    ) -> list[SpeakerTurn]:
        del audio_path, request
        return [
            SpeakerTurn(
                start=0.0,
                end=max(media_info.duration_sec, 0.001),
                speaker="SPEAKER_00",
            )
        ]


class PyannoteDiarizationEngine:
    name = "pyannote-community-1"

    def __init__(self, model_path_or_name: str, *, hf_token: str | None = None) -> None:
        if not has_package("pyannote.audio"):
            raise MissingDependencyError(
                "pyannote.audio",
                "Install with `uv pip install -e .[diarization]` or "
                "`pip install -e .[diarization]`.",
            )
        from pyannote.audio import Pipeline

        self.model_name = model_path_or_name
        local_path = Path(model_path_or_name).expanduser()
        kwargs: dict[str, str] = {}
        if hf_token and not local_path.exists():
            kwargs["token"] = hf_token
        try:
            self._pipeline = Pipeline.from_pretrained(model_path_or_name, **kwargs)
        except TypeError:
            legacy_kwargs: dict[str, str] = {}
            if hf_token and not local_path.exists():
                legacy_kwargs["use_auth_token"] = hf_token
            try:
                self._pipeline = Pipeline.from_pretrained(model_path_or_name, **legacy_kwargs)
            except Exception as legacy_exc:  # noqa: BLE001
                raise ModelUnavailableError(
                    PYANNOTE_COMMUNITY_1,
                    "Accept the model terms, set HF_TOKEN for first download, and run "
                    "`local-meetscribe models download --profile diarization`.",
                ) from legacy_exc
        except Exception as exc:  # noqa: BLE001 - pyannote raises mixed exception types.
            raise ModelUnavailableError(
                PYANNOTE_COMMUNITY_1,
                "Accept the model terms, set HF_TOKEN for first download, and run "
                "`local-meetscribe models download --profile diarization`.",
            ) from exc
        self._move_to_best_device()

    def diarize(
        self,
        audio_path: Path,
        media_info: MediaInfo,
        request: TranscriptionRequest,
    ) -> list[SpeakerTurn]:
        del media_info
        kwargs: dict[str, int] = {}
        if request.speakers is not None:
            kwargs["num_speakers"] = request.speakers
        if request.min_speakers is not None:
            kwargs["min_speakers"] = request.min_speakers
        if request.max_speakers is not None:
            kwargs["max_speakers"] = request.max_speakers
        output = self._pipeline(str(audio_path), **kwargs)
        annotation = getattr(output, "exclusive_speaker_diarization", None) or getattr(
            output, "speaker_diarization", output
        )
        turns: list[SpeakerTurn] = []
        speaker_map: dict[str, str] = {}
        for turn, label in _iter_speaker_turns(annotation):
            if label not in speaker_map:
                speaker_map[label] = f"SPEAKER_{len(speaker_map):02d}"
            turns.append(
                SpeakerTurn(
                    start=float(turn.start),
                    end=float(turn.end),
                    speaker=speaker_map[label],
                )
            )
        return turns

    def _move_to_best_device(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                self._pipeline = self._pipeline.to(torch.device("cuda"))
        except Exception:
            return


def _iter_speaker_turns(annotation: Any) -> list[tuple[Any, str]]:
    if hasattr(annotation, "itertracks"):
        return [(turn, str(label)) for turn, _, label in annotation.itertracks(yield_label=True)]
    return [(turn, str(label)) for turn, label in annotation]


class NemoSortformerDiarizationEngine:
    name = "nemo-sortformer"

    def __init__(self) -> None:
        raise MissingDependencyError(
            "nvidia-nemo-sortformer",
            "The NeMo Sortformer adapter is reserved for future integration and is not required.",
        )
