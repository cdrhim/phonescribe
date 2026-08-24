from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from local_meetscribe.config import Settings
from local_meetscribe.pipeline.ingest import MediaInfo, probe_media
from local_meetscribe.utils.errors import ExternalToolError, LocalMeetScribeError


@dataclass(frozen=True)
class PreparedAudio:
    path: Path
    original_info: MediaInfo
    prepared_info: MediaInfo
    original_bytes: int
    prepared_bytes: int
    remove_silence: bool
    max_minutes: float | None
    bitrate_kbps: int

    @property
    def compression_ratio(self) -> float:
        if self.original_bytes <= 0:
            return 0.0
        return self.prepared_bytes / self.original_bytes


def prepare_llm_audio(
    input_path: Path,
    output_dir: Path,
    settings: Settings,
    *,
    remove_silence: bool = True,
    max_minutes: float | None = None,
    bitrate_kbps: int = 32,
) -> PreparedAudio:
    if max_minutes is not None and max_minutes <= 0:
        raise LocalMeetScribeError("Max minutes must be empty or greater than 0.")
    if bitrate_kbps not in {24, 32, 48, 64}:
        raise LocalMeetScribeError("Bitrate must be one of 24, 32, 48, or 64 kbps.")

    original_info = probe_media(input_path, settings)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "llm_audio.m4a"
    ffmpeg = _require_tool(
        settings.ffmpeg_binary,
        "Install ffmpeg and ensure it is on PATH to prepare LLM audio.",
    )
    cmd = build_prepare_audio_command(
        ffmpeg=ffmpeg,
        input_path=input_path,
        output_path=output_path,
        remove_silence=remove_silence,
        max_minutes=max_minutes,
        bitrate_kbps=bitrate_kbps,
    )
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise LocalMeetScribeError(f"LLM audio preparation failed: {completed.stderr.strip()}")
    prepared_info = probe_media(output_path, settings)
    return PreparedAudio(
        path=output_path,
        original_info=original_info,
        prepared_info=prepared_info,
        original_bytes=input_path.stat().st_size,
        prepared_bytes=output_path.stat().st_size,
        remove_silence=remove_silence,
        max_minutes=max_minutes,
        bitrate_kbps=bitrate_kbps,
    )


def build_prepare_audio_command(
    *,
    ffmpeg: str,
    input_path: Path,
    output_path: Path,
    remove_silence: bool,
    max_minutes: float | None,
    bitrate_kbps: int,
) -> list[str]:
    filters: list[str] = []
    if remove_silence:
        filters.append(
            "silenceremove="
            "start_periods=1:"
            "start_duration=0.25:"
            "start_threshold=-45dB:"
            "stop_periods=-1:"
            "stop_duration=0.65:"
            "stop_threshold=-45dB"
        )
    if max_minutes is not None:
        filters.append(f"atrim=duration={max_minutes * 60:.3f}")
    if filters:
        filters.append("asetpts=PTS-STARTPTS")

    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
    ]
    if filters:
        cmd.extend(["-af", ",".join(filters)])
    cmd.extend(
        [
            "-c:a",
            "aac",
            "-b:a",
            f"{bitrate_kbps}k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    return cmd


def _require_tool(binary: str, install_hint: str) -> str:
    if shutil.which(binary):
        return binary
    raise ExternalToolError(binary, install_hint)
