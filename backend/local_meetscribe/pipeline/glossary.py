from __future__ import annotations

import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from local_meetscribe.config import Settings
from local_meetscribe.pipeline.asr import (
    FASTER_WHISPER_SMALL,
    FasterWhisperEngine,
    FasterWhisperRuntime,
    default_cpu_threads,
    ensure_model_allowed,
)
from local_meetscribe.schemas import Language
from local_meetscribe.utils.errors import LocalMeetScribeError

DEFAULT_SCAN_SECONDS = 90.0

_ENGLISH_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "are",
    "because",
    "but",
    "can",
    "for",
    "from",
    "have",
    "here",
    "into",
    "just",
    "like",
    "meeting",
    "more",
    "not",
    "now",
    "our",
    "that",
    "the",
    "then",
    "there",
    "this",
    "with",
    "you",
}

_KOREAN_STOPWORDS = {
    "거기",
    "거는",
    "거를",
    "거에",
    "거죠",
    "거지",
    "같고",
    "것도",
    "것은",
    "것을",
    "것이",
    "그거",
    "그게",
    "그냥",
    "그런",
    "그럼",
    "그래서",
    "그리고",
    "근데",
    "관련",
    "네요",
    "다른",
    "되게",
    "되는",
    "되면",
    "되어",
    "들이",
    "미팅",
    "라고",
    "많이",
    "말씀",
    "뭔가",
    "보면",
    "보시면",
    "시간",
    "시간짜리",
    "사실",
    "아니",
    "아마",
    "약간",
    "어떤",
    "여기",
    "오늘",
    "우리",
    "위해",
    "이거",
    "이게",
    "이런",
    "이제",
    "있고",
    "있는",
    "있다",
    "있어",
    "있습니다",
    "있으셨나",
    "저는",
    "저희",
    "정도",
    "제가",
    "조금",
    "지금",
    "진짜",
    "하구",
    "하면",
    "하고",
    "하는",
    "하나",
    "회사",
}

_KOREAN_SUFFIXES = (
    "입니다",
    "이에요",
    "이라고",
    "이고",
    "이어서",
    "거든요",
    "잖아요",
    "셨나요",
    "셨나",
    "이라면",
    "라면",
    "에서",
    "해서",
    "으로",
    "에게",
    "하고",
    "하구요",
    "습니다",
    "까지",
    "부터",
    "처럼",
    "보다",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "도",
    "만",
    "에",
    "의",
    "와",
    "과",
    "로",
    "요",
)


@dataclass(frozen=True)
class GlossaryScanResult:
    terms: list[str]
    preview_text: str
    detected_language: str
    scan_seconds: float
    warning: str | None = None


def quick_scan_glossary(
    input_path: Path,
    work_dir: Path,
    settings: Settings,
    *,
    language: Language = "auto",
    scan_seconds: float = DEFAULT_SCAN_SECONDS,
    max_terms: int = 18,
) -> GlossaryScanResult:
    work_dir.mkdir(parents=True, exist_ok=True)
    clip_path = work_dir / "glossary_scan_16k.wav"
    _write_scan_clip(input_path, clip_path, settings, scan_seconds)

    model_name = settings.faster_whisper_cpu_model or FASTER_WHISPER_SMALL
    model = ensure_model_allowed(
        model_name,
        settings.model_path(model_name),
        allow_autodownload=settings.allow_model_autodownload,
    )
    cpu_threads = settings.faster_whisper_cpu_threads or default_cpu_threads()
    engine = FasterWhisperEngine(
        model,
        runtime=FasterWhisperRuntime(
            device="cpu",
            compute_type="int8",
            batch_size=8,
            cpu_threads=cpu_threads,
        ),
        beam_size=1,
        word_timestamps=False,
        condition_on_previous_text=False,
        without_timestamps=True,
        temperature=0.0,
    )
    result = engine.transcribe(clip_path, [], language=language, glossary=[])
    preview_text = " ".join(segment.text for segment in result.segments).strip()
    first_language = result.segments[0].language if result.segments else None
    detected_language = _detected_language(preview_text, first_language)
    return GlossaryScanResult(
        terms=extract_glossary_terms(preview_text, max_terms=max_terms),
        preview_text=preview_text,
        detected_language=detected_language,
        scan_seconds=scan_seconds,
    )


def extract_glossary_terms(text: str, *, max_terms: int = 18) -> list[str]:
    displays: dict[str, Counter[str]] = defaultdict(Counter)
    counts: Counter[str] = Counter()
    kinds: dict[str, str] = {}

    for raw in re.findall(r"\b[A-Za-z][A-Za-z0-9+._/-]{1,24}\b", text):
        token = raw.strip(".,!?;:()[]{}\"'")
        key = token.lower()
        if key in _ENGLISH_STOPWORDS:
            continue
        if _is_repetitive_noise(token):
            continue
        if not (_looks_technical_english(token) or len(token) >= 5):
            continue
        counts[key] += 1
        displays[key][token] += 1
        kinds[key] = "en"

    for raw in re.findall(r"[가-힣]{2,16}", text):
        token = _normalize_korean_term(raw)
        if len(token) < 2 or token in _KOREAN_STOPWORDS:
            continue
        if _is_repetitive_noise(token):
            continue
        key = f"ko:{token}"
        counts[key] += 1
        displays[key][token] += 1
        kinds[key] = "ko"

    ranked = sorted(
        counts,
        key=lambda key: (
            _term_score(key, counts[key], displays[key].most_common(1)[0][0], kinds[key]),
            displays[key].most_common(1)[0][0],
        ),
        reverse=True,
    )

    terms: list[str] = []
    for key in ranked:
        display = displays[key].most_common(1)[0][0]
        if kinds[key] == "ko" and counts[key] < 2 and len(display) < 4:
            continue
        terms.append(display)
        if len(terms) >= max_terms:
            break
    return terms


def _write_scan_clip(
    input_path: Path,
    output_path: Path,
    settings: Settings,
    scan_seconds: float,
) -> None:
    cmd = [
        settings.ffmpeg_binary,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-t",
        str(scan_seconds),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
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
        raise LocalMeetScribeError(f"quick glossary scan failed: {completed.stderr.strip()}")


def _looks_technical_english(token: str) -> bool:
    return (
        any(character.isdigit() for character in token)
        or any(character in "+._/-" for character in token)
        or token.upper() == token
    )


def _is_repetitive_noise(token: str) -> bool:
    compact = re.sub(r"\s+", "", token)
    if len(compact) < 6:
        return False
    if re.search(r"(.)\1{3,}", compact):
        return True
    if len(set(compact)) <= 2:
        return True
    for width in (2, 3, 4):
        chunks = [compact[index : index + width] for index in range(0, len(compact), width)]
        if len(chunks) >= 3 and Counter(chunks).most_common(1)[0][1] / len(chunks) > 0.6:
            return True
    return False


def _normalize_korean_term(token: str) -> str:
    normalized = token
    changed = True
    while changed:
        changed = False
        for suffix in _KOREAN_SUFFIXES:
            if len(normalized) - len(suffix) >= 2 and normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
                changed = True
                break
    return normalized


def _term_score(key: str, count: int, display: str, kind: str) -> float:
    del key
    score = float(count * 4)
    if kind == "en" and _looks_technical_english(display):
        score += 8
    if kind == "ko":
        score += min(len(display), 6)
    return score


def _detected_language(text: str, raw_language: str | None) -> str:
    language = (raw_language or "").lower()
    if language in {"ko", "korean"}:
        return "ko"
    if language in {"en", "english"}:
        return "en"
    if re.search(r"[가-힣]", text):
        return "ko"
    if re.search(r"[A-Za-z]", text):
        return "en"
    return "unknown"
