from __future__ import annotations

import json
import re
from typing import Protocol

from local_meetscribe.schemas import Transcript
from local_meetscribe.utils.errors import MissingDependencyError

TRANSCRIPT_CLEANING_SYSTEM_PROMPT = (
    "You are a transcript formatter for Korean/English meeting transcripts. Preserve the "
    "original meaning and wording. Do not add information. Do not remove uncertainty. Only fix "
    "punctuation, paragraphing, obvious casing, spacing, and speaker-readable formatting. Keep "
    "Korean and English as spoken. Do not translate. If a phrase is unclear, keep it and add [?]."
)


class FormatterEngine(Protocol):
    name: str

    def clean(self, transcript: Transcript, glossary: list[str]) -> Transcript: ...


class RuleBasedFormatterEngine:
    name = "rule-based"

    def clean(self, transcript: Transcript, glossary: list[str]) -> Transcript:
        cleaned = transcript.model_copy(deep=True)
        for segment in cleaned.segments:
            segment.text_clean = _apply_glossary_casing(
                _safe_spacing(segment.text_clean),
                glossary,
            )
        return cleaned


class LocalLLMFormatterEngine:
    name = "ollama-local"

    def __init__(self, *, base_url: str, model: str) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise MissingDependencyError(
                "httpx",
                "Install with `uv pip install -e .[llm]` or `pip install -e .[llm]`.",
            ) from exc
        self._httpx = httpx
        self.base_url = base_url.rstrip("/")
        self.model = model

    def clean(self, transcript: Transcript, glossary: list[str]) -> Transcript:
        payload_segments = [
            {
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "speaker": segment.speaker,
                "text_raw": segment.text_raw,
            }
            for segment in transcript.segments
        ]
        user_prompt = (
            "Raw segment list with timestamps and speaker labels:\n"
            f"{json.dumps(payload_segments, ensure_ascii=False)}\n"
            "Glossary:\n"
            f"{json.dumps(glossary, ensure_ascii=False)}\n"
            "Return JSON with the same segment ids and only the cleaned text."
        )
        response = self._httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": TRANSCRIPT_CLEANING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=120,
        )
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "")
        updates = _parse_cleaning_response(content)
        cleaned = transcript.model_copy(deep=True)
        for segment in cleaned.segments:
            candidate = updates.get(segment.id)
            if candidate and _cleaning_is_conservative(segment.text_raw, candidate, glossary):
                segment.text_clean = _apply_glossary_casing(_safe_spacing(candidate), glossary)
            else:
                segment.text_clean = _apply_glossary_casing(
                    _safe_spacing(segment.text_clean),
                    glossary,
                )
        return cleaned


def _safe_spacing(text: str) -> str:
    value = re.sub(r"\s+", " ", text).strip()
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    value = re.sub(r"([,.;:!?])([^\s])", r"\1 \2", value)
    return value


def _apply_glossary_casing(text: str, glossary: list[str]) -> str:
    value = text
    for term in sorted({item.strip() for item in glossary if item.strip()}, key=len, reverse=True):
        if not re.search(r"[A-Za-z]", term):
            continue
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", re.IGNORECASE)
        value = pattern.sub(term, value)
    return value


def _cleaning_is_conservative(raw: str, candidate: str, glossary: list[str]) -> bool:
    if not candidate.strip():
        return False
    if _compact_hangul(raw) != _compact_hangul(candidate):
        return False
    raw_latin = _latin_tokens(raw)
    glossary_latin = set().union(*(_latin_tokens(term) for term in glossary), set())
    introduced_latin = _latin_tokens(candidate) - raw_latin - glossary_latin
    return not introduced_latin


def _latin_tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9]+", text) if len(token.strip()) > 1}


def _compact_hangul(text: str) -> str:
    return "".join(re.findall(r"[\uac00-\ud7a3]", text))


def _parse_cleaning_response(content: str) -> dict[str, str]:
    parsed = json.loads(content)
    if isinstance(parsed, dict) and "segments" in parsed:
        parsed = parsed["segments"]
    if not isinstance(parsed, list):
        raise ValueError("Formatter response must be a list or object with a segments list.")
    updates: dict[str, str] = {}
    for item in parsed:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            value = item.get("text_clean") or item.get("cleaned_text") or item.get("text")
            if isinstance(value, str):
                updates[item["id"]] = value
    return updates
