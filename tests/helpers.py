from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

from local_meetscribe.config import Settings


def write_tone_wav(path: Path, *, duration_sec: float = 1.0, sample_rate: int = 16000) -> Path:
    frames = int(duration_sec * sample_rate)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for index in range(frames):
            value = int(8000 * math.sin(2 * math.pi * 440 * index / sample_rate))
            wav_file.writeframes(struct.pack("<h", value))
    return path


def make_test_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        allow_mock_engines=True,
        allow_model_autodownload=False,
        ffmpeg_binary="ffmpeg",
        ffprobe_binary="ffprobe",
        ollama_url="http://127.0.0.1:11434",
        ollama_model="qwen3:8b",
        enable_llm_cleanup=False,
        hf_token=None,
    )
