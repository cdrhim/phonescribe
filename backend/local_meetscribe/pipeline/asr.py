from __future__ import annotations

import importlib.util
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from local_meetscribe.pipeline.vad import SpeechRegion
from local_meetscribe.schemas import Language
from local_meetscribe.utils.errors import MissingDependencyError, ModelUnavailableError

QWEN_ASR_17B = "Qwen/Qwen3-ASR-1.7B"
QWEN_ASR_06B = "Qwen/Qwen3-ASR-0.6B"
QWEN_ALIGNER = "Qwen/Qwen3-ForcedAligner-0.6B"
FASTER_WHISPER_LARGE_V3_TURBO = "turbo"
FASTER_WHISPER_SMALL = "small"


@dataclass(frozen=True)
class FasterWhisperRuntime:
    device: str
    compute_type: str
    batch_size: int
    num_workers: int = 1
    cpu_threads: int = 0


@dataclass(frozen=True)
class ASRWord:
    word: str
    start: float
    end: float
    confidence: float | None = None


@dataclass(frozen=True)
class ASRSegment:
    start: float
    end: float
    text: str
    language: str | None = None
    confidence: float | None = None
    words: list[ASRWord] | None = None
    avg_logprob: float | None = None
    compression_ratio: float | None = None
    no_speech_prob: float | None = None


@dataclass(frozen=True)
class ASRResult:
    engine_name: str
    model_name: str
    segments: list[ASRSegment]


class ASREngine(Protocol):
    name: str
    model_name: str

    def transcribe(
        self,
        audio_path: Path,
        regions: list[SpeechRegion],
        *,
        language: Language,
        glossary: list[str],
    ) -> ASRResult: ...


def has_package(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def _language_for_qwen(language: Language) -> str | None:
    return {"auto": None, "ko": "Korean", "en": "English"}[language]


def _language_for_whisper(language: Language) -> str | None:
    return None if language == "auto" else language


class MockASREngine:
    name = "mock-asr"
    model_name = "mock"

    def transcribe(
        self,
        audio_path: Path,
        regions: list[SpeechRegion],
        *,
        language: Language,
        glossary: list[str],
    ) -> ASRResult:
        del audio_path, glossary
        segments: list[ASRSegment] = []
        for index, region in enumerate(regions or [SpeechRegion(0.0, 1.0)], start=1):
            text = (
                f"Mock transcript segment {index}. "
                "Configure local ASR models to transcribe real audio."
            )
            words = _space_words(text, region.start, region.end)
            segments.append(
                ASRSegment(
                    start=region.start,
                    end=region.end,
                    text=text,
                    language="unknown" if language == "auto" else language,
                    confidence=None,
                    words=words,
                )
            )
        return ASRResult(engine_name=self.name, model_name=self.model_name, segments=segments)


class FasterWhisperEngine:
    name = "faster-whisper"

    def __init__(
        self,
        model_path_or_name: str,
        *,
        runtime: FasterWhisperRuntime | None = None,
        beam_size: int = 1,
        vad_filter: bool = True,
        use_batched: bool = True,
        word_timestamps: bool = True,
        condition_on_previous_text: bool = False,
        without_timestamps: bool = False,
        temperature: float | list[float] = 0.0,
    ) -> None:
        if not has_package("faster_whisper"):
            raise MissingDependencyError(
                "faster-whisper",
                "Install with `uv pip install -e .[whisper]` or `pip install -e .[whisper]`.",
            )
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        self.model_name = model_path_or_name
        self.runtime = runtime or FasterWhisperRuntime(
            device="cpu",
            compute_type="int8",
            batch_size=4,
        )
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.use_batched = use_batched and self.runtime.batch_size > 1
        self.word_timestamps = word_timestamps
        self.condition_on_previous_text = condition_on_previous_text
        self.without_timestamps = without_timestamps
        self.temperature = temperature
        self._batched_pipeline: Any | None = None
        try:
            self._model = WhisperModel(
                model_path_or_name,
                device=self.runtime.device,
                compute_type=self.runtime.compute_type,
                num_workers=self.runtime.num_workers,
                cpu_threads=self.runtime.cpu_threads,
            )
            if self.use_batched:
                self._batched_pipeline = BatchedInferencePipeline(model=self._model)
        except Exception as exc:  # noqa: BLE001 - CTranslate2 raises runtime-specific errors.
            if _is_oom_error(exc) or _is_device_runtime_error(exc):
                raise ModelUnavailableError(
                    model_path_or_name,
                    "faster-whisper could not initialize with "
                    f"device={self.runtime.device}, compute_type={self.runtime.compute_type}. "
                    "Try CPU mode, int8 quantization, or a smaller model.",
                ) from exc
            raise

    def transcribe(
        self,
        audio_path: Path,
        regions: list[SpeechRegion],
        *,
        language: Language,
        glossary: list[str],
    ) -> ASRResult:
        del regions
        initial_prompt = "\n".join(glossary) if glossary else None
        try:
            segments_iter, info = self._transcribe_once(
                audio_path,
                language=language,
                initial_prompt=initial_prompt,
            )
            result_segments = self._collect_segments(segments_iter, info)
        except Exception as exc:  # noqa: BLE001 - CTranslate2 raises runtime-specific errors.
            if not _is_oom_error(exc):
                raise
            if self.runtime.device == "cuda" and self.runtime.compute_type == "float16":
                self.runtime = FasterWhisperRuntime(
                    "cuda",
                    "int8_float16",
                    8,
                    cpu_threads=self.runtime.cpu_threads,
                )
                self._reload_after_oom()
                segments_iter, info = self._transcribe_once(
                    audio_path,
                    language=language,
                    initial_prompt=initial_prompt,
                )
                result_segments = self._collect_segments(segments_iter, info)
            elif self.runtime.device == "cuda":
                self.runtime = FasterWhisperRuntime(
                    "cpu",
                    "int8",
                    8,
                    cpu_threads=self.runtime.cpu_threads,
                )
                self._reload_after_oom()
                segments_iter, info = self._transcribe_once(
                    audio_path,
                    language=language,
                    initial_prompt=initial_prompt,
                )
                result_segments = self._collect_segments(segments_iter, info)
            else:
                raise ModelUnavailableError(
                    self.model_name,
                    "faster-whisper ran out of memory on CPU. Try a smaller model "
                    "or shorter audio.",
                ) from exc
        return ASRResult(
            engine_name=self.name,
            model_name=(
                f"{self.model_name} "
                f"({self.runtime.device}/{self.runtime.compute_type}/batch={self.runtime.batch_size})"
            ),
            segments=result_segments,
        )

    def _transcribe_once(
        self,
        audio_path: Path,
        *,
        language: Language,
        initial_prompt: str | None,
    ) -> tuple[Any, Any]:
        transcriber = self._batched_pipeline or self._model
        kwargs: dict[str, Any] = {
            "language": _language_for_whisper(language),
            "task": "transcribe",
            "initial_prompt": initial_prompt,
            "word_timestamps": self.word_timestamps,
            "beam_size": self.beam_size,
            "temperature": self.temperature,
            "condition_on_previous_text": self.condition_on_previous_text,
        }
        if self._batched_pipeline is not None:
            kwargs["batch_size"] = self.runtime.batch_size
            kwargs["without_timestamps"] = self.without_timestamps
        else:
            kwargs["vad_filter"] = self.vad_filter
        return transcriber.transcribe(str(audio_path), **kwargs)

    def _reload_after_oom(self) -> None:
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        self._model = WhisperModel(
            self.model_name,
            device=self.runtime.device,
            compute_type=self.runtime.compute_type,
            num_workers=self.runtime.num_workers,
            cpu_threads=self.runtime.cpu_threads,
        )
        self._batched_pipeline = (
            BatchedInferencePipeline(model=self._model)
            if self.use_batched and self.runtime.batch_size > 1
            else None
        )

    @staticmethod
    def _collect_segments(segments_iter: Any, info: Any) -> list[ASRSegment]:
        result_segments: list[ASRSegment] = []
        for segment in segments_iter:
            avg_logprob = _safe_float(getattr(segment, "avg_logprob", None))
            words = [
                ASRWord(
                    word=w.word.strip(),
                    start=float(w.start),
                    end=float(w.end),
                    confidence=getattr(w, "probability", None),
                )
                for w in (segment.words or [])
                if w.word.strip()
            ]
            result_segments.append(
                ASRSegment(
                    start=float(segment.start),
                    end=float(segment.end),
                    text=segment.text.strip(),
                    language=getattr(info, "language", None),
                    confidence=_logprob_to_confidence(avg_logprob),
                    words=words,
                    avg_logprob=avg_logprob,
                    compression_ratio=_safe_float(getattr(segment, "compression_ratio", None)),
                    no_speech_prob=_safe_float(getattr(segment, "no_speech_prob", None)),
                )
            )
        return result_segments


class QwenASREngine:
    name = "qwen3-asr"

    def __init__(
        self,
        model_path_or_name: str,
        *,
        forced_aligner_path_or_name: str | None = None,
        device_map: str | None = None,
        dtype_name: str = "auto",
    ) -> None:
        if not has_package("qwen_asr"):
            raise MissingDependencyError(
                "qwen-asr",
                "Install with `uv pip install -e .[qwen]` or `pip install -e .[qwen]`.",
            )
        import torch
        from qwen_asr import Qwen3ASRModel

        dtype = _torch_dtype(torch, dtype_name)
        kwargs: dict[str, Any] = {
            "dtype": dtype,
            "max_inference_batch_size": 8,
            "max_new_tokens": 2048,
        }
        if device_map is not None:
            kwargs["device_map"] = device_map
        if forced_aligner_path_or_name:
            kwargs["forced_aligner"] = forced_aligner_path_or_name
            kwargs["forced_aligner_kwargs"] = {"dtype": dtype}
            if device_map is not None:
                kwargs["forced_aligner_kwargs"]["device_map"] = device_map
        self.model_name = model_path_or_name
        self._return_timestamps = forced_aligner_path_or_name is not None
        try:
            self._model = Qwen3ASRModel.from_pretrained(model_path_or_name, **kwargs)
        except Exception as exc:  # noqa: BLE001 - torch/transformers surface mixed errors.
            if _is_oom_error(exc):
                raise ModelUnavailableError(
                    model_path_or_name,
                    "Qwen3-ASR ran out of memory while loading. Try Qwen3-ASR-0.6B, "
                    "CPU mode, or a lower-memory runtime.",
                ) from exc
            raise

    def transcribe(
        self,
        audio_path: Path,
        regions: list[SpeechRegion],
        *,
        language: Language,
        glossary: list[str],
    ) -> ASRResult:
        del regions, glossary
        try:
            result = self._model.transcribe(
                audio=str(audio_path),
                language=_language_for_qwen(language),
                return_time_stamps=self._return_timestamps,
            )
        except Exception as exc:  # noqa: BLE001 - torch/transformers surface mixed errors.
            if _is_oom_error(exc):
                raise ModelUnavailableError(
                    self.model_name,
                    "Qwen3-ASR ran out of memory during inference. Try Qwen3-ASR-0.6B "
                    "or faster-whisper CPU mode.",
                ) from exc
            raise
        first = result[0] if isinstance(result, list) else result
        detected_language = _get_value(first, "language")
        text = str(_get_value(first, "text") or "").strip()
        timestamps = _get_value(first, "time_stamps") or _get_value(first, "timestamps") or []
        if timestamps:
            segments = [
                ASRSegment(
                    start=float(_get_value(item, "start_time", "start") or 0.0),
                    end=float(_get_value(item, "end_time", "end") or 0.0),
                    text=str(_get_value(item, "text", "word") or "").strip(),
                    language=str(detected_language) if detected_language else None,
                    confidence=None,
                    words=[
                        ASRWord(
                            word=str(_get_value(item, "text", "word") or "").strip(),
                            start=float(_get_value(item, "start_time", "start") or 0.0),
                            end=float(_get_value(item, "end_time", "end") or 0.0),
                            confidence=None,
                        )
                    ],
                )
                for item in timestamps
                if str(_get_value(item, "text", "word") or "").strip()
            ]
        else:
            segments = [
                ASRSegment(
                    start=0.0,
                    end=0.0,
                    text=text,
                    language=str(detected_language) if detected_language else None,
                    confidence=None,
                    words=[],
                )
            ]
        return ASRResult(engine_name=self.name, model_name=self.model_name, segments=segments)


def choose_compute() -> tuple[str, str]:
    if has_cuda_runtime():
        return "cuda", "float16"
    return "cpu", "int8"


def faster_whisper_runtime_plan(
    *,
    cpu: bool = False,
    cpu_threads: int = 0,
) -> list[FasterWhisperRuntime]:
    if cpu:
        return [FasterWhisperRuntime("cpu", "int8", 8, cpu_threads=cpu_threads)]
    if has_cuda_runtime():
        return [
            FasterWhisperRuntime("cuda", "float16", 16),
            FasterWhisperRuntime("cuda", "int8_float16", 8),
            FasterWhisperRuntime("cpu", "int8", 8, cpu_threads=cpu_threads),
        ]
    return [FasterWhisperRuntime("cpu", "int8", 8, cpu_threads=cpu_threads)]


def has_cuda_runtime() -> bool:
    if shutil.which("nvidia-smi"):
        return True
    try:
        import ctranslate2

        return bool(ctranslate2.get_cuda_device_count())
    except Exception:  # noqa: BLE001 - probing must never block CPU fallback.
        return False


def default_cpu_threads() -> int:
    count = os.cpu_count() or 0
    if count <= 2:
        return 0
    return max(1, count - 1)


def ensure_model_allowed(
    model: str,
    model_path: Path,
    *,
    allow_autodownload: bool,
) -> str:
    if model_path.exists():
        return str(model_path)
    if allow_autodownload:
        return model
    raise ModelUnavailableError(
        model,
        "Run `local-meetscribe models download --profile accurate|fast|diarization` or enable "
        "LOCAL_MEETSCRIBE_ALLOW_MODEL_AUTODOWNLOAD=true.",
    )


def _torch_dtype(torch_module: Any, dtype_name: str) -> Any:
    if dtype_name == "bfloat16" or dtype_name == "auto":
        return torch_module.bfloat16
    if dtype_name == "float16":
        return torch_module.float16
    if dtype_name == "float32":
        return torch_module.float32
    return torch_module.bfloat16


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _logprob_to_confidence(avg_logprob: float | None) -> float | None:
    if avg_logprob is None:
        return None
    return max(0.0, min(1.0, math.exp(avg_logprob)))


def _get_value(item: Any, *names: str) -> Any:
    for name in names:
        if isinstance(item, dict) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    return None


def _is_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "out of memory",
            "cuda oom",
            "cublas_status_alloc_failed",
            "std::bad_alloc",
            "failed to allocate",
        )
    )


def _is_device_runtime_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "cuda driver",
            "cudnn",
            "cublas",
            "no cuda",
            "invalid device",
        )
    )


def _space_words(text: str, start: float, end: float) -> list[ASRWord]:
    tokens = [token for token in re.split(r"\s+", text.strip()) if token]
    if not tokens:
        return []
    duration = max(0.001, end - start)
    step = duration / len(tokens)
    return [
        ASRWord(word=token, start=start + index * step, end=start + (index + 1) * step)
        for index, token in enumerate(tokens)
    ]
