from __future__ import annotations

import json
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

from local_meetscribe.config import Settings
from local_meetscribe.utils.errors import ExternalToolError, LocalMeetScribeError


@dataclass(frozen=True)
class MediaInfo:
    filename: str
    duration_sec: float
    sample_rate: int
    channels: int


@dataclass(frozen=True)
class NormalizedAudio:
    path: Path
    info: MediaInfo


def _is_wav(path: Path) -> bool:
    return path.suffix.lower() == ".wav"


def _probe_wav(path: Path) -> MediaInfo:
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        frames = wav_file.getnframes()
        duration = frames / float(sample_rate) if sample_rate else 0.0
    return MediaInfo(
        filename=path.name,
        duration_sec=duration,
        sample_rate=sample_rate,
        channels=channels,
    )


def _require_tool(binary: str, install_hint: str) -> str:
    if shutil.which(binary):
        return binary
    raise ExternalToolError(binary, install_hint)


def probe_media(path: Path, settings: Settings) -> MediaInfo:
    if not path.exists():
        raise LocalMeetScribeError(f"Input file does not exist: {path}")
    if _is_wav(path):
        return _probe_wav(path)

    ffprobe = _require_tool(
        settings.ffprobe_binary,
        "Install ffmpeg/ffprobe and ensure it is on PATH, or provide a 16 kHz mono WAV.",
    )
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise LocalMeetScribeError(f"ffprobe failed for input file: {completed.stderr.strip()}")
    if not completed.stdout:
        raise LocalMeetScribeError(
            "ffprobe did not return media metadata. Check that the file is a valid "
            "audio/video file."
        )
    data = json.loads(completed.stdout)
    audio_stream = next(
        (stream for stream in data.get("streams", []) if stream.get("codec_type") == "audio"),
        None,
    )
    if audio_stream is None:
        raise LocalMeetScribeError("Input does not contain an audio stream.")
    duration_raw = audio_stream.get("duration") or data.get("format", {}).get("duration") or 0
    return MediaInfo(
        filename=path.name,
        duration_sec=float(duration_raw),
        sample_rate=int(audio_stream.get("sample_rate") or 0),
        channels=int(audio_stream.get("channels") or 1),
    )


def normalize_to_wav(
    input_path: Path,
    output_dir: Path,
    settings: Settings,
    *,
    loudness_normalize: bool = False,
    trim_silence: bool = False,
) -> NormalizedAudio:
    output_dir.mkdir(parents=True, exist_ok=True)
    original = probe_media(input_path, settings)
    output_path = output_dir / "normalized_16k_mono.wav"

    if (
        _is_wav(input_path)
        and original.sample_rate == 16000
        and original.channels == 1
        and not loudness_normalize
        and not trim_silence
    ):
        shutil.copy2(input_path, output_path)
        return NormalizedAudio(
            path=output_path,
            info=MediaInfo(
                filename=input_path.name,
                duration_sec=original.duration_sec,
                sample_rate=16000,
                channels=1,
            ),
        )

    ffmpeg = _require_tool(
        settings.ffmpeg_binary,
        "Install ffmpeg and ensure it is on PATH for non-WAV inputs or preprocessing options.",
    )
    filters: list[str] = []
    if loudness_normalize:
        filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")
    if trim_silence:
        filters.append("silenceremove=start_periods=1:start_duration=0.2:start_threshold=-45dB")

    cmd = [ffmpeg, "-y", "-i", str(input_path), "-ac", "1", "-ar", "16000"]
    if filters:
        cmd.extend(["-af", ",".join(filters)])
    cmd.append(str(output_path))
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise LocalMeetScribeError(f"ffmpeg normalization failed: {completed.stderr.strip()}")

    normalized = _probe_wav(output_path)
    return NormalizedAudio(
        path=output_path,
        info=MediaInfo(
            filename=input_path.name,
            duration_sec=normalized.duration_sec,
            sample_rate=16000,
            channels=1,
        ),
    )
