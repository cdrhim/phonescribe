from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    models_dir: Path
    allow_mock_engines: bool
    allow_model_autodownload: bool
    ffmpeg_binary: str
    ffprobe_binary: str
    ollama_url: str
    ollama_model: str
    enable_llm_cleanup: bool
    hf_token: str | None
    enable_gemini_transcription: bool = False
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash"
    gemini_api_base: str = "https://generativelanguage.googleapis.com/v1beta"
    faster_whisper_cpu_model: str = "small"
    faster_whisper_cuda_model: str = "turbo"
    faster_whisper_cpu_threads: int = 0

    @property
    def db_path(self) -> Path:
        return self.data_dir / "jobs.sqlite3"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def tmp_dir(self) -> Path:
        return self.data_dir / "tmp"

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def model_path(self, repo_id: str) -> Path:
        return self.models_dir / repo_id.replace("/", "--")


@lru_cache
def get_settings() -> Settings:
    data_dir = Path(os.getenv("LOCAL_MEETSCRIBE_DATA_DIR", "./data")).expanduser().resolve()
    models_dir = Path(os.getenv("LOCAL_MEETSCRIBE_MODELS_DIR", "./models")).expanduser().resolve()
    return Settings(
        data_dir=data_dir,
        models_dir=models_dir,
        allow_mock_engines=_env_bool("LOCAL_MEETSCRIBE_ALLOW_MOCKS", True),
        allow_model_autodownload=_env_bool("LOCAL_MEETSCRIBE_ALLOW_MODEL_AUTODOWNLOAD", False),
        ffmpeg_binary=os.getenv("LOCAL_MEETSCRIBE_FFMPEG") or _default_tool("ffmpeg"),
        ffprobe_binary=os.getenv("LOCAL_MEETSCRIBE_FFPROBE") or _default_tool("ffprobe"),
        ollama_url=os.getenv("LOCAL_MEETSCRIBE_OLLAMA_URL", "http://127.0.0.1:11434"),
        ollama_model=os.getenv("LOCAL_MEETSCRIBE_OLLAMA_MODEL", "qwen3:8b"),
        enable_llm_cleanup=_env_bool("LOCAL_MEETSCRIBE_ENABLE_LLM_CLEANUP", False),
        hf_token=os.getenv("HF_TOKEN") or None,
        enable_gemini_transcription=_env_bool(
            "LOCAL_MEETSCRIBE_ENABLE_GEMINI_TRANSCRIPTION", False
        ),
        gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
        gemini_model=os.getenv("LOCAL_MEETSCRIBE_GEMINI_MODEL", "gemini-3.5-flash"),
        gemini_api_base=os.getenv(
            "LOCAL_MEETSCRIBE_GEMINI_API_BASE",
            "https://generativelanguage.googleapis.com/v1beta",
        ).rstrip("/"),
        faster_whisper_cpu_model=os.getenv("LOCAL_MEETSCRIBE_FASTER_WHISPER_CPU_MODEL", "small"),
        faster_whisper_cuda_model=os.getenv("LOCAL_MEETSCRIBE_FASTER_WHISPER_CUDA_MODEL", "turbo"),
        faster_whisper_cpu_threads=_env_int("LOCAL_MEETSCRIBE_FASTER_WHISPER_CPU_THREADS", 0),
    )


def ensure_runtime_dirs(settings: Settings) -> None:
    for path in (
        settings.data_dir,
        settings.models_dir,
        settings.uploads_dir,
        settings.jobs_dir,
        settings.tmp_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _default_tool(name: str) -> str:
    discovered = shutil.which(name)
    if discovered:
        return discovered
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        winget_root = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        if winget_root.exists():
            matches = sorted(
                winget_root.glob(f"**/{name}.exe"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if matches:
                return str(matches[0])
    return name


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default
