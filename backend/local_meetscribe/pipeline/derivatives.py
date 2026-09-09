from __future__ import annotations

import hashlib
import math
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from local_meetscribe.utils.errors import LocalMeetScribeError

TranscriptArtifactKind = Literal["organized", "summary"]

ARTIFACT_FILENAMES: dict[TranscriptArtifactKind, str] = {
    "organized": "meeting_organized.txt",
    "summary": "meeting_summary.txt",
}

_TIMESTAMP_PREFIX = re.compile(r"^\s*\[([^\]\r\n]{1,32})\]\s*")
_SPEAKER_PREFIX = re.compile(
    r"^\s*((?:SPEAKER[_\s-]*\d+|speaker\s*\d+|\ud654\uc790\s*\d+|\ubc1c\uc5b8\uc790\s*\d+|\ucc38\uc11d\uc790\s*\d+))\s*:\s*",
    re.IGNORECASE,
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?\u3002\uff01\uff1f])\s+")
_WHITESPACE = re.compile(r"[ \t\f\v]+")
_LOW_VALUE_UTTERANCE = re.compile(
    r"^(?:\ub124|\uc608|\uc544|\uc5b4|\uc74c|\uc800|\uadf8\ub7ec\ub124\uc694|\uadf8\ub807\uad70\uc694|okay|ok|yes|yeah|right)[.!?\s]*$",
    re.IGNORECASE,
)
_ACTION_TERMS = (
    "\ub2f4\ub2f9",
    "\ud560 \uc77c",
    "\ud558\uaca0\uc2b5\ub2c8\ub2e4",
    "\ud574\uc57c",
    "\uae4c\uc9c0",
    "\ub9c8\uac10",
    "\ud6c4\uc18d",
    "\uac80\ud1a0",
    "\uc900\ube44",
    "\uc804\ub2ec",
    "\uacf5\uc720",
    "\ub2e4\uc74c",
    "action item",
    "follow up",
    "follow-up",
    "next step",
    "owner",
    "deadline",
    "due ",
)
_DECISION_TERMS = (
    "\uacb0\uc815",
    "\ud569\uc758",
    "\ud655\uc815",
    "\ud558\uae30\ub85c",
    "\uc9c4\ud589\ud558\uae30\ub85c",
    "decision",
    "decided",
    "agreed",
    "confirmed",
)
_TOPIC_TERMS = (
    "\ubaa9\ud45c",
    "\ubb38\uc81c",
    "\uc774\uc288",
    "\uc6b0\uc120",
    "\uc694\uad6c",
    "\uc81c\uc548",
    "\uacb0\uacfc",
    "\uc608\uc0b0",
    "\uc77c\uc815",
    "goal",
    "issue",
    "priority",
    "proposal",
    "result",
    "budget",
    "schedule",
)


@dataclass(frozen=True)
class TranscriptArtifactResult:
    kind: TranscriptArtifactKind
    text: str
    txt_path: Path
    source_sha256: str


def create_transcript_artifact(
    package_dir: Path,
    kind: TranscriptArtifactKind | str,
) -> TranscriptArtifactResult:
    if kind not in ARTIFACT_FILENAMES:
        raise LocalMeetScribeError("Unsupported transcript artifact kind.")
    artifact_kind = cast(TranscriptArtifactKind, kind)
    source_path = package_dir / "gemini_transcript.txt"
    try:
        source_bytes = source_path.read_bytes()
    except OSError as exc:
        raise LocalMeetScribeError("The transcript is not ready yet.") from exc
    if len(source_bytes) > 10 * 1024 * 1024:
        raise LocalMeetScribeError("The transcript is too large to prepare a derived version.")
    try:
        source_text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LocalMeetScribeError("The transcript text is not valid UTF-8.") from exc
    if not source_text.strip():
        raise LocalMeetScribeError("The transcript is empty.")

    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if artifact_kind == "organized":
        artifact_text = organize_transcript(source_text)
    else:
        artifact_text = summarize_transcript(source_text)

    target = package_dir / ARTIFACT_FILENAMES[artifact_kind]
    _atomic_write_text(target, artifact_text)
    try:
        current_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    except OSError as exc:
        target.unlink(missing_ok=True)
        raise LocalMeetScribeError("The transcript changed while preparing the result.") from exc
    if current_sha256 != source_sha256:
        target.unlink(missing_ok=True)
        raise LocalMeetScribeError("The transcript changed while preparing the result.")
    return TranscriptArtifactResult(
        kind=artifact_kind,
        text=artifact_text,
        txt_path=target,
        source_sha256=source_sha256,
    )


def organize_transcript(source_text: str) -> str:
    """Format every source utterance without summarizing or replacing its wording."""

    normalized = source_text.replace("\r\n", "\n").replace("\r", "\n")
    blocks: list[str] = []
    pending_plain: list[str] = []

    def flush_plain() -> None:
        if not pending_plain:
            return
        blocks.append(" ".join(pending_plain))
        pending_plain.clear()

    for raw_line in normalized.split("\n"):
        line = _normalize_inline_space(raw_line).strip()
        if not line:
            flush_plain()
            continue
        timestamp, speaker, spoken = _turn_parts(line)
        if timestamp or speaker:
            flush_plain()
            heading = " · ".join(part for part in (timestamp, speaker) if part)
            blocks.append(f"{heading}\n{spoken}" if spoken else heading)
        else:
            pending_plain.append(line)
    flush_plain()

    body = "\n\n".join(block for block in blocks if block.strip()).strip()
    return f"\ubbf8\ud305 \uc815\ub9ac\ubcf8\n\n{body}\n"


def summarize_transcript(source_text: str) -> str:
    """Create a concise extractive summary using only phrases present in the transcript."""

    sentences = _source_sentences(source_text)
    if not sentences:
        raise LocalMeetScribeError("The transcript does not contain text to summarize.")

    target_count = min(10, max(3, math.ceil(math.sqrt(len(sentences)) * 1.4)))
    target_count = min(target_count, len(sentences))
    ranked = sorted(
        range(len(sentences)),
        key=lambda index: (_summary_score(sentences[index], index, len(sentences)), -index),
        reverse=True,
    )
    chosen = sorted(ranked[:target_count])
    selected = [sentences[index] for index in chosen]

    follow_ups = [sentence for sentence in selected if _is_decision_or_action(sentence)]
    follow_up_keys = {sentence.casefold() for sentence in follow_ups}
    key_points = [sentence for sentence in selected if sentence.casefold() not in follow_up_keys]
    if not key_points:
        key_points = selected
        follow_ups = []

    lines = ["\ubbf8\ud305 \uc694\uc57d\ubcf8", "", "\ud575\uc2ec \ub0b4\uc6a9"]
    lines.extend(f"- {sentence}" for sentence in key_points)
    if follow_ups:
        lines.extend(["", "\uacb0\uc815\u00b7\ud6c4\uc18d"])
        lines.extend(f"- {sentence}" for sentence in follow_ups)
    return "\n".join(lines).strip() + "\n"


def read_stored_transcript_artifacts(package_dir: Path) -> dict[str, TranscriptArtifactResult]:
    source_path = package_dir / "gemini_transcript.txt"
    try:
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    except OSError:
        return {}

    artifacts: dict[str, TranscriptArtifactResult] = {}
    for kind, filename in ARTIFACT_FILENAMES.items():
        path = package_dir / filename
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if not text.strip():
            continue
        artifacts[kind] = TranscriptArtifactResult(
            kind=kind,
            text=text,
            txt_path=path,
            source_sha256=source_sha256,
        )
    return artifacts


def _source_sentences(source_text: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    normalized = source_text.replace("\r\n", "\n").replace("\r", "\n")
    for raw_line in normalized.split("\n"):
        line = _normalize_inline_space(raw_line).strip()
        if not line:
            continue
        _timestamp, _speaker, spoken = _turn_parts(line)
        content = spoken or line
        fragments = _SENTENCE_BOUNDARY.split(content)
        for fragment in fragments:
            sentence = fragment.strip(" -\u2022\t")
            if len(sentence) < 4 or _LOW_VALUE_UTTERANCE.fullmatch(sentence):
                continue
            key = sentence.casefold()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(sentence)
    return candidates


def _turn_parts(line: str) -> tuple[str, str, str]:
    timestamp = ""
    timestamp_match = _TIMESTAMP_PREFIX.match(line)
    if timestamp_match:
        timestamp = f"[{timestamp_match.group(1).strip()}]"
        line = line[timestamp_match.end() :].strip()

    speaker = ""
    speaker_match = _SPEAKER_PREFIX.match(line)
    if speaker_match:
        speaker = speaker_match.group(1).strip()
        line = line[speaker_match.end() :].strip()
    return timestamp, speaker, line


def _summary_score(sentence: str, index: int, total: int) -> float:
    lowered = sentence.casefold()
    score = 0.0
    score += sum(4.0 for term in _DECISION_TERMS if term in lowered)
    score += sum(3.0 for term in _ACTION_TERMS if term in lowered)
    score += sum(2.0 for term in _TOPIC_TERMS if term in lowered)
    if re.search(r"\b\d+(?:[./:-]\d+)*\b", sentence):
        score += 1.5
    if sentence.endswith(("?", "\uff1f")):
        score += 0.5
    length = len(sentence)
    if 20 <= length <= 220:
        score += 2.0
    elif length > 360:
        score -= 1.0
    score += max(0.0, 1.5 - (index / max(1, total - 1)))
    return score


def _is_decision_or_action(sentence: str) -> bool:
    lowered = sentence.casefold()
    return any(term in lowered for term in (*_DECISION_TERMS, *_ACTION_TERMS))


def _normalize_inline_space(value: str) -> str:
    return _WHITESPACE.sub(" ", value)


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        # Write bytes so the downloadable result is identical on Windows and POSIX.
        # Path.write_text() translates ``\n`` to ``\r\n`` on Windows by default,
        # while the JSON preview keeps ``\n`` unchanged.
        temporary.write_bytes(value.encode("utf-8"))
        temporary.replace(path)
    except OSError as exc:
        raise LocalMeetScribeError("Could not save the transcript result.") from exc
    finally:
        temporary.unlink(missing_ok=True)
