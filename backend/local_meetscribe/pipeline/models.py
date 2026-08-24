from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

from local_meetscribe.config import Settings
from local_meetscribe.pipeline.asr import (
    FASTER_WHISPER_LARGE_V3_TURBO,
    FASTER_WHISPER_SMALL,
    QWEN_ALIGNER,
    QWEN_ASR_06B,
    QWEN_ASR_17B,
)
from local_meetscribe.pipeline.diarize import PYANNOTE_COMMUNITY_1
from local_meetscribe.utils.errors import MissingDependencyError


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    profile: str
    package_module: str
    note: str


@dataclass(frozen=True)
class ModelStatus:
    repo_id: str
    profile: str
    local_path: Path
    downloaded: bool
    package_module: str
    package_available: bool
    note: str


MODEL_SPECS = [
    ModelSpec(QWEN_ASR_17B, "accurate", "qwen_asr", "Primary accurate ASR."),
    ModelSpec(QWEN_ASR_06B, "accurate", "qwen_asr", "Lower-resource Qwen ASR fallback."),
    ModelSpec(QWEN_ALIGNER, "accurate", "qwen_asr", "Qwen forced aligner for timestamps."),
    ModelSpec(
        FASTER_WHISPER_SMALL,
        "fast",
        "faster_whisper",
        "CPU-friendly faster-whisper model for long meetings.",
    ),
    ModelSpec(
        FASTER_WHISPER_LARGE_V3_TURBO,
        "fast",
        "faster_whisper",
        "CUDA-friendly faster-whisper turbo ASR.",
    ),
    ModelSpec(PYANNOTE_COMMUNITY_1, "diarization", "pyannote.audio", "Local speaker diarization."),
]

PROFILE_REPOS = {
    "accurate": [QWEN_ASR_17B, QWEN_ASR_06B, QWEN_ALIGNER],
    "fast": [FASTER_WHISPER_SMALL, FASTER_WHISPER_LARGE_V3_TURBO],
    "diarization": [PYANNOTE_COMMUNITY_1],
}


def package_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def get_model_status(settings: Settings) -> list[ModelStatus]:
    return [
        ModelStatus(
            repo_id=spec.repo_id,
            profile=spec.profile,
            local_path=settings.model_path(spec.repo_id),
            downloaded=settings.model_path(spec.repo_id).exists(),
            package_module=spec.package_module,
            package_available=package_available(spec.package_module),
            note=spec.note,
        )
        for spec in MODEL_SPECS
    ]


def download_profile(profile: str, settings: Settings) -> list[Path]:
    if profile not in PROFILE_REPOS:
        raise ValueError(f"Unknown profile: {profile}")

    downloaded: list[Path] = []
    for repo_id in PROFILE_REPOS[profile]:
        destination = settings.model_path(repo_id)
        destination.mkdir(parents=True, exist_ok=True)
        if repo_id == FASTER_WHISPER_LARGE_V3_TURBO:
            _download_faster_whisper_model(repo_id, destination)
            downloaded.append(destination)
            continue
        snapshot_download = _snapshot_download()
        token = settings.hf_token if repo_id.startswith("pyannote/") else None
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(destination),
            token=token,
        )
        downloaded.append(destination)
    return downloaded


def _download_faster_whisper_model(model_name: str, destination: Path) -> None:
    if not package_available("faster_whisper"):
        raise MissingDependencyError(
            "faster-whisper",
            "Install with `uv pip install -e .[whisper]` before downloading the fast profile.",
        )
    from faster_whisper.utils import download_model

    download_model(model_name, output_dir=str(destination))


def _snapshot_download():
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise MissingDependencyError(
            "huggingface-hub",
            "Install an optional extra that includes model downloads, for example "
            "`uv pip install -e .[qwen]`.",
        ) from exc
    return snapshot_download
