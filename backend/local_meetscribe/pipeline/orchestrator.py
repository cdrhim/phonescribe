from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from pathlib import Path

from local_meetscribe.config import Settings, ensure_runtime_dirs, get_settings
from local_meetscribe.pipeline.align import SegmentAlignmentEngine
from local_meetscribe.pipeline.asr import (
    FASTER_WHISPER_LARGE_V3_TURBO,
    FASTER_WHISPER_SMALL,
    QWEN_ALIGNER,
    QWEN_ASR_06B,
    QWEN_ASR_17B,
    ASREngine,
    ASRResult,
    ASRSegment,
    FasterWhisperEngine,
    MockASREngine,
    QwenASREngine,
    default_cpu_threads,
    ensure_model_allowed,
    faster_whisper_runtime_plan,
    has_cuda_runtime,
    has_package,
)
from local_meetscribe.pipeline.diarize import (
    PYANNOTE_COMMUNITY_1,
    DiarizationEngine,
    PyannoteDiarizationEngine,
    SingleSpeakerDiarizationEngine,
)
from local_meetscribe.pipeline.enhance import DeepFilterNetEnhancementEngine, NoopEnhancementEngine
from local_meetscribe.pipeline.export import write_exports
from local_meetscribe.pipeline.format import (
    FormatterEngine,
    LocalLLMFormatterEngine,
    RuleBasedFormatterEngine,
)
from local_meetscribe.pipeline.ingest import MediaInfo, normalize_to_wav
from local_meetscribe.pipeline.merge import merge_transcript
from local_meetscribe.pipeline.vad import MockVADEngine
from local_meetscribe.schemas import Transcript, TranscriptConfig, TranscriptionRequest
from local_meetscribe.utils.errors import (
    LocalMeetScribeError,
    MissingDependencyError,
    ModelUnavailableError,
)

ProgressCallback = Callable[[str, float], None]

LOGGER = logging.getLogger(__name__)


class EngineFactory:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def create_asr(
        self,
        request: TranscriptionRequest,
        *,
        media_info: MediaInfo | None = None,
    ) -> ASREngine:
        errors: list[str] = []
        candidates: list[Callable[[], ASREngine]]
        if request.mode == "accurate":
            candidates = [
                lambda: self._qwen_engine(QWEN_ASR_17B),
                lambda: self._qwen_engine(QWEN_ASR_06B),
                lambda: self._faster_whisper_engine(request, media_info=media_info),
            ]
        elif request.mode == "fast":
            candidates = [lambda: self._faster_whisper_engine(request, media_info=media_info)]
        else:
            candidates = [
                lambda: self._faster_whisper_engine(request, media_info=media_info, cpu=True)
            ]

        for candidate in candidates:
            try:
                return candidate()
            except (MissingDependencyError, ModelUnavailableError) as exc:
                errors.append(str(exc))

        if self.settings.allow_mock_engines and request.allow_mock:
            LOGGER.info("Falling back to mock ASR because no real local model is available.")
            return MockASREngine()
        raise LocalMeetScribeError(
            "No ASR engine is available. "
            + " ".join(errors)
            + " Enable mocks with LOCAL_MEETSCRIBE_ALLOW_MOCKS=true for smoke tests."
        )

    def create_diarization(self, request: TranscriptionRequest) -> DiarizationEngine:
        if request.speakers == 1:
            return SingleSpeakerDiarizationEngine()
        if not has_package("pyannote.audio"):
            return SingleSpeakerDiarizationEngine()
        model_path = self.settings.model_path(PYANNOTE_COMMUNITY_1)
        if not model_path.exists() and not self.settings.allow_model_autodownload:
            return SingleSpeakerDiarizationEngine()
        model = str(model_path) if model_path.exists() else PYANNOTE_COMMUNITY_1
        try:
            return PyannoteDiarizationEngine(model, hf_token=self.settings.hf_token)
        except ModelUnavailableError:
            return SingleSpeakerDiarizationEngine()

    def create_formatter(self) -> FormatterEngine:
        if self.settings.enable_llm_cleanup:
            return LocalLLMFormatterEngine(
                base_url=self.settings.ollama_url,
                model=self.settings.ollama_model,
            )
        return RuleBasedFormatterEngine()

    def _qwen_engine(self, repo_id: str) -> QwenASREngine:
        if not has_package("qwen_asr"):
            raise MissingDependencyError(
                "qwen-asr",
                "Install with `uv pip install -e .[qwen]` or `pip install -e .[qwen]`.",
            )
        model = ensure_model_allowed(
            repo_id,
            self.settings.model_path(repo_id),
            allow_autodownload=self.settings.allow_model_autodownload,
        )
        aligner_path = self.settings.model_path(QWEN_ALIGNER)
        forced_aligner = str(aligner_path) if aligner_path.exists() else None
        return QwenASREngine(
            model,
            forced_aligner_path_or_name=forced_aligner,
            device_map=_default_device_map(),
        )

    def _faster_whisper_engine(
        self,
        request: TranscriptionRequest,
        *,
        media_info: MediaInfo | None = None,
        cpu: bool = False,
    ) -> FasterWhisperEngine:
        if not has_package("faster_whisper"):
            raise MissingDependencyError(
                "faster-whisper",
                "Install with `uv pip install -e .[whisper]` or `pip install -e .[whisper]`.",
            )
        use_cuda = has_cuda_runtime() and not cpu
        requested_model = (
            self.settings.faster_whisper_cuda_model
            if use_cuda
            else self.settings.faster_whisper_cpu_model
        )
        model_name = requested_model or (
            FASTER_WHISPER_LARGE_V3_TURBO if use_cuda else FASTER_WHISPER_SMALL
        )
        model = ensure_model_allowed(
            model_name,
            self.settings.model_path(model_name),
            allow_autodownload=self.settings.allow_model_autodownload,
        )
        errors: list[str] = []
        duration_sec = media_info.duration_sec if media_info else 0.0
        long_cpu_job = not use_cuda and duration_sec >= 20 * 60
        word_timestamps = use_cuda or (request.mode == "accurate" and not long_cpu_job)
        without_timestamps = not word_timestamps
        cpu_threads = self.settings.faster_whisper_cpu_threads or default_cpu_threads()
        for runtime in faster_whisper_runtime_plan(cpu=cpu, cpu_threads=cpu_threads):
            try:
                return FasterWhisperEngine(
                    model,
                    runtime=runtime,
                    beam_size=1,
                    word_timestamps=word_timestamps,
                    condition_on_previous_text=False,
                    without_timestamps=without_timestamps,
                    temperature=0.0,
                )
            except ModelUnavailableError as exc:
                errors.append(str(exc))
        raise ModelUnavailableError(
            model_name,
            "No faster-whisper runtime could initialize. " + " ".join(errors),
        )


class TranscriptionPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        engine_factory: EngineFactory | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        ensure_runtime_dirs(self.settings)
        self.engine_factory = engine_factory or EngineFactory(self.settings)
        self.progress_callback = progress_callback

    def run(
        self,
        input_path: Path,
        *,
        output_dir: Path | None = None,
        request: TranscriptionRequest | None = None,
        job_id: str | None = None,
    ) -> Transcript:
        options = request or TranscriptionRequest()
        current_job_id = job_id or uuid.uuid4().hex
        work_dir = self.settings.job_dir(current_job_id)
        output = output_dir or (work_dir / "exports")
        work_dir.mkdir(parents=True, exist_ok=True)
        self._progress("ingest", 0.08)
        normalized = normalize_to_wav(
            input_path,
            work_dir,
            self.settings,
            loudness_normalize=options.loudness_normalize,
            trim_silence=options.trim_silence,
        )

        self._progress("enhance", 0.16)
        if options.denoise:
            enhanced_path = DeepFilterNetEnhancementEngine().enhance(normalized.path, work_dir)
        else:
            enhanced_path = NoopEnhancementEngine().enhance(normalized.path, work_dir)

        self._progress("vad", 0.24)
        vad_engine = MockVADEngine()
        regions = vad_engine.detect(enhanced_path, normalized.info)

        self._progress("asr", 0.42)
        asr_engine = self.engine_factory.create_asr(options, media_info=normalized.info)
        asr_result = asr_engine.transcribe(
            enhanced_path,
            regions,
            language=options.language,
            glossary=options.glossary,
        )
        asr_result = _repair_zero_length_segments(asr_result, normalized.info.duration_sec)

        self._progress("alignment", 0.58)
        alignment_engine = SegmentAlignmentEngine()
        aligned = alignment_engine.align(enhanced_path, asr_result, language=options.language)

        self._progress("diarization", 0.70)
        diarization_engine = self.engine_factory.create_diarization(options)
        speaker_turns = diarization_engine.diarize(enhanced_path, normalized.info, options)

        self._progress("merge", 0.80)
        config = TranscriptConfig(
            mode=options.mode,
            asr_engine=asr_result.engine_name,
            asr_model=asr_result.model_name,
            diarization_engine=diarization_engine.name,
            language=options.language,
            alignment_engine=alignment_engine.name,
            vad_engine=vad_engine.name,
        )
        transcript = merge_transcript(
            job_id=current_job_id,
            media_info=normalized.info,
            config=config,
            asr_result=aligned,
            speaker_turns=speaker_turns,
        )

        self._progress("format", 0.88)
        formatter = self.engine_factory.create_formatter()
        transcript.config.formatter_engine = formatter.name
        transcript = formatter.clean(transcript, options.glossary)

        self._progress("export", 0.95)
        transcript = write_exports(transcript, output)
        self._progress("completed", 1.0)
        return transcript

    def _progress(self, stage: str, progress: float) -> None:
        if self.progress_callback:
            self.progress_callback(stage, progress)


def _repair_zero_length_segments(asr_result: ASRResult, duration_sec: float) -> ASRResult:
    if not asr_result.segments:
        return asr_result
    repaired: list[ASRSegment] = []
    for segment in asr_result.segments:
        end = segment.end
        if end <= segment.start:
            end = duration_sec
        repaired.append(
            ASRSegment(
                start=segment.start,
                end=end,
                text=segment.text,
                language=segment.language,
                confidence=segment.confidence,
                words=segment.words,
                avg_logprob=segment.avg_logprob,
                compression_ratio=segment.compression_ratio,
                no_speech_prob=segment.no_speech_prob,
            )
        )
    return ASRResult(asr_result.engine_name, asr_result.model_name, repaired)


def _default_device_map() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001 - device probing must not block fallback behavior.
        return "cpu"
    return "cpu"
