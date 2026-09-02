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
    remote_access_enabled: bool = False
    remote_session_ttl_sec: int = 2 * 60 * 60
    cors_origins: tuple[str, ...] = (
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    )
    auto_export_dir: Path | None = None
    supabase_enabled: bool = False
    supabase_url: str | None = None
    supabase_service_role_key: str | None = None
    supabase_bucket: str = "recordings"
    supabase_owner_id: str | None = None
    supabase_part_size_bytes: int = 6 * 1024 * 1024
    supabase_max_recording_bytes: int = 4 * 1024 * 1024 * 1024
    supabase_request_timeout_sec: int = 30

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
        remote_access_enabled=_env_bool("LOCAL_MEETSCRIBE_REMOTE_ACCESS", False),
        remote_session_ttl_sec=max(
            300,
            _env_int("LOCAL_MEETSCRIBE_REMOTE_SESSION_TTL_SEC", 2 * 60 * 60),
        ),
        cors_origins=_env_csv(
            "LOCAL_MEETSCRIBE_CORS_ORIGINS",
            ("http://127.0.0.1:5173", "http://localhost:5173"),
        ),
        auto_export_dir=_env_optional_path("LOCAL_MEETSCRIBE_AUTO_EXPORT_DIR"),
        supabase_enabled=_env_bool("LOCAL_MEETSCRIBE_SUPABASE_ENABLED", False),
        supabase_url=(os.getenv("LOCAL_MEETSCRIBE_SUPABASE_URL") or "").strip().rstrip("/") or None,
        supabase_service_role_key=(
            os.getenv("LOCAL_MEETSCRIBE_SUPABASE_SERVICE_ROLE_KEY") or ""
        ).strip()
        or None,
        supabase_bucket="recordings",
        supabase_owner_id=(os.getenv("LOCAL_MEETSCRIBE_SUPABASE_OWNER_ID") or "").strip() or None,
        supabase_part_size_bytes=max(
            6 * 1024 * 1024,
            min(
                24 * 1024 * 1024,
                _env_int(
                    "LOCAL_MEETSCRIBE_SUPABASE_PART_SIZE_BYTES",
                    6 * 1024 * 1024,
                ),
            ),
        ),
        supabase_max_recording_bytes=max(
            1,
            _env_int(
                "LOCAL_MEETSCRIBE_SUPABASE_MAX_RECORDING_BYTES",
                4 * 1024 * 1024 * 1024,
            ),
        ),
        supabase_request_timeout_sec=max(
            5,
            _env_int("LOCAL_MEETSCRIBE_SUPABASE_REQUEST_TIMEOUT_SEC", 30),
        ),
    )


def ensure_runtime_dirs(settings: Settings) -> None:
    for path in (
        settings.data_dir,
        settings.models_dir,
        settings.uploads_dir,
        settings.jobs_dir,
        settings.tmp_dir,
        settings.auto_export_dir,
    ):
        if path is not None:
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


def _env_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None:
        return default
    parsed = tuple(item.strip().rstrip("/") for item in value.split(",") if item.strip())
    return parsed or default


def _env_optional_path(name: str) -> Path | None:
    value = (os.getenv(name) or "").strip()
    if not value:
        return None
    return Path(value).expanduser().resolve()
