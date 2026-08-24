from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from local_meetscribe.config import Settings
from local_meetscribe.pipeline.ingest import MediaInfo, probe_media
from local_meetscribe.utils.errors import ExternalToolError, LocalMeetScribeError

Destination = Literal["gemini", "openai", "optimize"]
OpenAIModel = Literal["gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"]
Codec = Literal["mp3", "m4a", "ogg"]

MB_PER_MINUTE = {
    64: 0.47,
    48: 0.35,
    32: 0.23,
    24: 0.18,
}

PROVIDER_MATRIX = {
    "gemini": {
        "label": "Gemini",
        "role": "native multimodal audio to text",
        "audio_input": True,
        "codec": "mp3",
        "sample_rate_hz": 16000,
        "channels": 1,
        "bitrate_kbps": 32,
        "max_audio_hours": 9.5,
        "inline_limit_mb": 20,
        "preferred_chunk_minutes": 30,
        "single_request_minutes": 45,
        "tokens_per_second": 32,
        "rationale": (
            "Gemini accepts long native audio; 16 kHz mono at about 32 kbps avoids "
            "uploading bytes Gemini will downsample internally. Long meetings use "
            "silence-aligned chunks so retries and transcript output remain manageable."
        ),
    },
    "openai": {
        "label": "OpenAI STT",
        "role": "dedicated /v1/audio/transcriptions endpoint",
        "audio_input": True,
        "codec": "m4a",
        "sample_rate_hz": 16000,
        "channels": 1,
        "bitrate_kbps": 32,
        "file_limit_mb": 25,
        "safe_file_limit_mb": 24,
        "preferred_chunk_minutes": 10,
        "models": {
            "gpt-4o-transcribe": {
                "cost_per_minute": 0.006,
                "duration_cap_sec": 1500,
                "word_timestamps": False,
            },
            "gpt-4o-mini-transcribe": {
                "cost_per_minute": 0.003,
                "duration_cap_sec": 1500,
                "word_timestamps": False,
            },
            "whisper-1": {
                "cost_per_minute": 0.006,
                "duration_cap_sec": None,
                "word_timestamps": True,
            },
        },
        "rationale": (
            "OpenAI STT has a 25 MB cap, and gpt-4o transcribe models also cap each "
            "request at 25 minutes; short silence-aware chunks are more reliable."
        ),
    },
    "anthropic": {
        "label": "Claude",
        "role": "post processor only",
        "audio_input": False,
        "rationale": (
            "Claude cannot transcribe raw audio; use it only after another STT result exists."
        ),
    },
    "optimize": {
        "label": "Optimize only",
        "role": "small portable local artifact",
        "audio_input": True,
        "codec": "ogg",
        "sample_rate_hz": 16000,
        "channels": 1,
        "bitrate_kbps": 24,
        "rationale": (
            "Opus/OGG at 16 kHz mono is the smallest portable local artifact for clear speech."
        ),
    },
}


@dataclass(frozen=True)
class OptimizerOverrides:
    codec: Codec | None = None
    bitrate_kbps: int | None = None
    chunk_minutes: float | None = None
    remove_silence: bool = True
    loudnorm: bool = True
    speech_filter: bool = True
    denoise: bool = False


@dataclass(frozen=True)
class OptimizerRequest:
    destination: Destination
    openai_model: OpenAIModel = "gpt-4o-transcribe"
    word_timestamps: bool = False
    overrides: OptimizerOverrides = OptimizerOverrides()


@dataclass(frozen=True)
class Recommendation:
    destination: Destination
    provider_label: str
    model: str | None
    codec: Codec
    sample_rate_hz: int
    channels: int
    bitrate_kbps: int
    chunk_count: int
    chunk_minutes: float | None
    projected_size_mb: float
    projected_chunk_mb: float
    estimated_tokens: int | None
    estimated_cost_usd: float | None
    delivery: str
    rationale: str
    warnings: list[str]
    prompt: str


@dataclass(frozen=True)
class OptimizedChunk:
    filename: str
    download_url: str
    start_sec: float
    end_sec: float
    duration_sec: float
    bytes: int


@dataclass(frozen=True)
class OptimizedPackage:
    id: str
    recommendation: Recommendation
    source: MediaInfo
    chunks: list[OptimizedChunk]
    manifest_url: str
    package_url: str
    output_dir: Path


def recommend_optimization(
    media_info: MediaInfo,
    input_bytes: int,
    request: OptimizerRequest,
) -> Recommendation:
    del input_bytes
    if request.destination == "openai":
        return _recommend_openai(media_info, request)
    if request.destination == "gemini":
        return _recommend_gemini(media_info, request.overrides)
    if request.destination == "optimize":
        return _recommend_optimize(media_info, request.overrides)
    raise LocalMeetScribeError(f"Unsupported destination: {request.destination}")


def optimize_audio_package(
    input_path: Path,
    output_root: Path,
    settings: Settings,
    request: OptimizerRequest,
    *,
    package_id: str,
) -> OptimizedPackage:
    output_dir = output_root / package_id
    output_dir.mkdir(parents=True, exist_ok=True)
    source_info = probe_media(input_path, settings)
    recommendation = recommend_optimization(source_info, input_path.stat().st_size, request)
    split_points = _split_points(input_path, settings, source_info, recommendation)

    chunks: list[OptimizedChunk] = []
    for index, (start, end) in enumerate(split_points, start=1):
        filename = f"chunk_{index:03d}.{recommendation.codec}"
        output_path = output_dir / filename
        _run_ffmpeg_chunk(
            input_path=input_path,
            output_path=output_path,
            settings=settings,
            recommendation=recommendation,
            overrides=request.overrides,
            start_sec=start,
            end_sec=end,
        )
        chunks.append(
            OptimizedChunk(
                filename=filename,
                download_url=f"/api/optimizer/packages/{package_id}/{filename}",
                start_sec=start,
                end_sec=end,
                duration_sec=max(0.0, end - start),
                bytes=output_path.stat().st_size,
            )
        )

    manifest = {
        "id": package_id,
        "source": asdict(source_info),
        "recommendation": asdict(recommendation),
        "chunks": [asdict(chunk) for chunk in chunks],
        "provider_constraints": _public_provider_matrix(),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    package_path = output_dir / "optimized_package.zip"
    with zipfile.ZipFile(package_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(manifest_path, arcname="manifest.json")
        for chunk in chunks:
            archive.write(output_dir / chunk.filename, arcname=chunk.filename)

    return OptimizedPackage(
        id=package_id,
        recommendation=recommendation,
        source=source_info,
        chunks=chunks,
        manifest_url=f"/api/optimizer/packages/{package_id}/manifest.json",
        package_url=f"/api/optimizer/packages/{package_id}/optimized_package.zip",
        output_dir=output_dir,
    )


def _recommend_openai(media_info: MediaInfo, request: OptimizerRequest) -> Recommendation:
    provider = PROVIDER_MATRIX["openai"]
    models = provider["models"]
    model_name = "whisper-1" if request.word_timestamps else request.openai_model
    model = models[model_name]
    bitrate = _bitrate(request.overrides.bitrate_kbps, int(provider["bitrate_kbps"]))
    codec = _codec(request.overrides.codec, str(provider["codec"]))
    chunk_minutes = request.overrides.chunk_minutes or float(provider["preferred_chunk_minutes"])
    max_by_size = float(provider["safe_file_limit_mb"]) / MB_PER_MINUTE[bitrate]
    duration_cap = model["duration_cap_sec"]
    if duration_cap:
        max_by_duration = float(duration_cap) / 60
        chunk_minutes = min(chunk_minutes, max_by_duration)
    chunk_minutes = min(chunk_minutes, max_by_size)
    chunk_minutes = max(1.0, chunk_minutes)
    duration_min = _minutes(media_info.duration_sec)
    chunk_count = max(1, math.ceil(duration_min / chunk_minutes))
    projected_chunk_mb = min(chunk_minutes, duration_min) * MB_PER_MINUTE[bitrate]
    cost = duration_min * float(model["cost_per_minute"])
    warnings = []
    if request.word_timestamps and request.openai_model != "whisper-1":
        warnings.append("Word-level timestamps require whisper-1; model changed for this package.")
    return Recommendation(
        destination="openai",
        provider_label=str(provider["label"]),
        model=model_name,
        codec=codec,
        sample_rate_hz=int(provider["sample_rate_hz"]),
        channels=int(provider["channels"]),
        bitrate_kbps=bitrate,
        chunk_count=chunk_count,
        chunk_minutes=chunk_minutes,
        projected_size_mb=duration_min * MB_PER_MINUTE[bitrate],
        projected_chunk_mb=projected_chunk_mb,
        estimated_tokens=None,
        estimated_cost_usd=cost,
        delivery="Upload chunks to /v1/audio/transcriptions.",
        rationale=str(provider["rationale"]),
        warnings=warnings,
        prompt=_prompt_for("openai", model_name),
    )


def _recommend_gemini(media_info: MediaInfo, overrides: OptimizerOverrides) -> Recommendation:
    provider = PROVIDER_MATRIX["gemini"]
    bitrate = _bitrate(overrides.bitrate_kbps, int(provider["bitrate_kbps"]))
    codec = _codec(overrides.codec, str(provider["codec"]))
    duration_min = _minutes(media_info.duration_sec)
    projected_size = duration_min * MB_PER_MINUTE[bitrate]
    inline_limit = float(provider["inline_limit_mb"])
    max_hours = float(provider["max_audio_hours"])
    preferred_chunk_minutes = float(provider["preferred_chunk_minutes"])
    single_request_minutes = float(provider["single_request_minutes"])
    warnings = []
    requested_chunk_minutes = overrides.chunk_minutes
    if requested_chunk_minutes is not None:
        chunk_minutes = max(1.0, min(requested_chunk_minutes, max_hours * 60))
    elif duration_min > single_request_minutes:
        chunk_minutes = preferred_chunk_minutes
    else:
        chunk_minutes = None
    chunk_count = 1 if chunk_minutes is None else max(1, math.ceil(duration_min / chunk_minutes))
    if chunk_count > 1:
        warnings.append(
            f"Long meeting: using {chunk_count} silence-aligned chunks for reliable retries."
        )
    projected_chunk_mb = (
        projected_size
        if chunk_minutes is None
        else min(duration_min, chunk_minutes) * MB_PER_MINUTE[bitrate]
    )
    delivery = (
        "Inline generateContent"
        if projected_chunk_mb <= inline_limit
        else "Gemini Files API"
    )
    return Recommendation(
        destination="gemini",
        provider_label=str(provider["label"]),
        model="gemini audio",
        codec=codec,
        sample_rate_hz=int(provider["sample_rate_hz"]),
        channels=int(provider["channels"]),
        bitrate_kbps=bitrate,
        chunk_count=chunk_count,
        chunk_minutes=chunk_minutes if chunk_count > 1 else None,
        projected_size_mb=projected_size,
        projected_chunk_mb=projected_chunk_mb,
        estimated_tokens=math.ceil(media_info.duration_sec * int(provider["tokens_per_second"])),
        estimated_cost_usd=None,
        delivery=delivery,
        rationale=str(provider["rationale"]),
        warnings=warnings,
        prompt=_prompt_for("gemini", None),
    )


def _recommend_optimize(media_info: MediaInfo, overrides: OptimizerOverrides) -> Recommendation:
    provider = PROVIDER_MATRIX["optimize"]
    bitrate = _bitrate(overrides.bitrate_kbps, int(provider["bitrate_kbps"]))
    codec = _codec(overrides.codec, str(provider["codec"]))
    duration_min = _minutes(media_info.duration_sec)
    return Recommendation(
        destination="optimize",
        provider_label=str(provider["label"]),
        model=None,
        codec=codec,
        sample_rate_hz=int(provider["sample_rate_hz"]),
        channels=int(provider["channels"]),
        bitrate_kbps=bitrate,
        chunk_count=1,
        chunk_minutes=None,
        projected_size_mb=duration_min * MB_PER_MINUTE[bitrate],
        projected_chunk_mb=duration_min * MB_PER_MINUTE[bitrate],
        estimated_tokens=None,
        estimated_cost_usd=None,
        delivery="Download a local optimized audio artifact.",
        rationale=str(provider["rationale"]),
        warnings=[],
        prompt=_prompt_for("optimize", None),
    )


def _split_points(
    input_path: Path,
    settings: Settings,
    media_info: MediaInfo,
    recommendation: Recommendation,
) -> list[tuple[float, float]]:
    if recommendation.chunk_count <= 1:
        return [(0.0, media_info.duration_sec)]
    max_chunk_sec = (recommendation.chunk_minutes or 10.0) * 60
    silences = _detect_silences(input_path, settings)
    points: list[tuple[float, float]] = []
    start = 0.0
    while start < media_info.duration_sec:
        hard_end = min(start + max_chunk_sec, media_info.duration_sec)
        if hard_end >= media_info.duration_sec:
            points.append((start, media_info.duration_sec))
            break
        target = hard_end
        min_end = start + max_chunk_sec * 0.7
        candidates = [
            (silence_start + silence_end) / 2
            for silence_start, silence_end in silences
            if min_end <= (silence_start + silence_end) / 2 <= hard_end
        ]
        end = min(candidates, key=lambda value: abs(value - target)) if candidates else hard_end
        points.append((start, end))
        start = end
    return points


def _detect_silences(input_path: Path, settings: Settings) -> list[tuple[float, float]]:
    ffmpeg = _require_tool(settings.ffmpeg_binary)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-nostats",
        "-i",
        str(input_path),
        "-af",
        "silencedetect=noise=-40dB:d=0.6",
        "-f",
        "null",
        "-",
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
        return []
    starts: list[float] = []
    ranges: list[tuple[float, float]] = []
    for line in completed.stderr.splitlines():
        start_match = re.search(r"silence_start:\s*([0-9.]+)", line)
        if start_match:
            starts.append(float(start_match.group(1)))
            continue
        end_match = re.search(r"silence_end:\s*([0-9.]+)", line)
        if end_match and starts:
            ranges.append((starts.pop(0), float(end_match.group(1))))
    return ranges


def _run_ffmpeg_chunk(
    *,
    input_path: Path,
    output_path: Path,
    settings: Settings,
    recommendation: Recommendation,
    overrides: OptimizerOverrides,
    start_sec: float,
    end_sec: float,
) -> None:
    ffmpeg = _require_tool(settings.ffmpeg_binary)
    filters = _audio_filters(overrides)
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_sec:.3f}",
        "-t",
        f"{max(0.001, end_sec - start_sec):.3f}",
        "-i",
        str(input_path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        str(recommendation.channels),
        "-ar",
        str(recommendation.sample_rate_hz),
    ]
    if filters:
        cmd.extend(["-af", ",".join(filters)])
    cmd.extend(_codec_args(recommendation.codec, recommendation.bitrate_kbps))
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
        raise LocalMeetScribeError(f"Optimizer ffmpeg failed: {completed.stderr.strip()}")


def _audio_filters(overrides: OptimizerOverrides) -> list[str]:
    filters: list[str] = []
    if overrides.speech_filter:
        filters.extend(["highpass=f=80", "lowpass=f=8000"])
    if overrides.loudnorm:
        filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")
    if overrides.denoise:
        filters.append("afftdn")
    if overrides.remove_silence:
        filters.append(
            "silenceremove="
            "start_periods=1:"
            "start_duration=0.25:"
            "start_threshold=-45dB:"
            "stop_periods=-1:"
            "stop_duration=0.65:"
            "stop_threshold=-45dB"
        )
    filters.append("asetpts=PTS-STARTPTS")
    return filters


def _codec_args(codec: Codec, bitrate_kbps: int) -> list[str]:
    if codec == "mp3":
        return ["-c:a", "libmp3lame", "-b:a", f"{bitrate_kbps}k"]
    if codec == "ogg":
        return ["-c:a", "libopus", "-b:a", f"{bitrate_kbps}k", "-vbr", "on"]
    return ["-c:a", "aac", "-b:a", f"{bitrate_kbps}k", "-movflags", "+faststart"]


def _bitrate(value: int | None, fallback: int) -> int:
    bitrate = value or fallback
    if bitrate not in MB_PER_MINUTE:
        raise LocalMeetScribeError("Bitrate must be one of 24, 32, 48, or 64 kbps.")
    return bitrate


def _codec(value: str | None, fallback: str) -> Codec:
    codec = value or fallback
    if codec not in {"mp3", "m4a", "ogg"}:
        raise LocalMeetScribeError("Codec must be mp3, m4a, or ogg.")
    return codec  # type: ignore[return-value]


def _minutes(seconds: float) -> float:
    return max(0.001, seconds / 60)


def _prompt_for(destination: str, model: str | None) -> str:
    if destination == "gemini":
        return (
            "Transcribe this meeting audio with timestamps, Korean/English text, rough speaker "
            "labels if clear, and no added summary. Do not invent inaudible content."
        )
    if destination == "openai":
        return (
            f"Use {model} to transcribe each chunk. Preserve chunk offsets from manifest.json "
            "when stitching timestamps. Do not send audio to Claude."
        )
    return (
        "Use this optimized local file as the input artifact for your chosen transcription "
        "system."
    )


def _public_provider_matrix() -> dict[str, object]:
    return {key: value for key, value in PROVIDER_MATRIX.items() if key != "anthropic"}


def _require_tool(binary: str) -> str:
    discovered = shutil.which(binary)
    if discovered:
        return discovered
    raise ExternalToolError(binary, "Install ffmpeg/ffprobe and ensure it is on PATH.")
