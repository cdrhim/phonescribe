from __future__ import annotations

import re
from collections import Counter, defaultdict

from local_meetscribe.pipeline.asr import ASRResult, ASRSegment
from local_meetscribe.pipeline.diarize import SpeakerTurn
from local_meetscribe.pipeline.ingest import MediaInfo
from local_meetscribe.schemas import (
    SegmentLanguage,
    SourceInfo,
    Speaker,
    Transcript,
    TranscriptConfig,
    TranscriptSegment,
    Word,
)


def merge_transcript(
    *,
    job_id: str,
    media_info: MediaInfo,
    config: TranscriptConfig,
    asr_result: ASRResult,
    speaker_turns: list[SpeakerTurn],
) -> Transcript:
    turns = speaker_turns or [
        SpeakerTurn(start=0.0, end=max(media_info.duration_sec, 0.001), speaker="SPEAKER_00")
    ]
    segments: list[TranscriptSegment] = []
    speaker_seconds: dict[str, float] = defaultdict(float)

    for index, asr_segment in enumerate(asr_result.segments, start=1):
        words = [
            Word(
                word=word.word,
                start=word.start,
                end=word.end,
                confidence=word.confidence,
                speaker=_speaker_for_interval(word.start, word.end, turns),
            )
            for word in (asr_segment.words or [])
        ]
        speaker = _segment_speaker(asr_segment.start, asr_segment.end, words, turns)
        duration = max(0.0, asr_segment.end - asr_segment.start)
        speaker_seconds[speaker] += duration
        overlap = _has_overlap(asr_segment.start, asr_segment.end, turns)
        needs_review = _needs_review(asr_segment, overlap, config.asr_engine)
        segments.append(
            TranscriptSegment(
                id=f"seg_{index:06d}",
                start=asr_segment.start,
                end=asr_segment.end,
                speaker=speaker,
                language=_segment_language(asr_segment.language, asr_segment.text),
                text_raw=asr_segment.text,
                text_clean=asr_segment.text,
                confidence=asr_segment.confidence,
                needs_review=needs_review,
                overlap=overlap,
                words=words,
            )
        )

    if not speaker_seconds:
        speaker_seconds["SPEAKER_00"] = media_info.duration_sec
    speakers = [
        Speaker(id=speaker_id, display_name=f"Speaker {idx + 1}", total_sec=seconds)
        for idx, (speaker_id, seconds) in enumerate(sorted(speaker_seconds.items()))
    ]
    return Transcript(
        id=job_id,
        source=SourceInfo(
            filename=media_info.filename,
            duration_sec=media_info.duration_sec,
            sample_rate=16000,
            channels=1,
        ),
        config=config,
        speakers=speakers,
        segments=segments,
    )


def _speaker_for_interval(start: float, end: float, turns: list[SpeakerTurn]) -> str:
    scores = _speaker_overlap_scores(start, end, turns)
    if not scores:
        midpoint = (start + end) / 2
        for turn in turns:
            if turn.start <= midpoint <= turn.end:
                return turn.speaker
        return "SPEAKER_00"
    return max(scores.items(), key=lambda item: item[1])[0]


def _segment_speaker(
    start: float,
    end: float,
    words: list[Word],
    turns: list[SpeakerTurn],
) -> str:
    scores = _speaker_overlap_scores(start, end, turns)
    if scores:
        return max(scores.items(), key=lambda item: item[1])[0]
    if words:
        durations: dict[str, float] = defaultdict(float)
        for word in words:
            durations[word.speaker] += max(0.0, word.end - word.start)
        if durations:
            return max(durations.items(), key=lambda item: item[1])[0]
    return "SPEAKER_00"


def _speaker_overlap_scores(
    start: float,
    end: float,
    turns: list[SpeakerTurn],
) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for turn in turns:
        amount = _overlap(start, end, turn.start, turn.end)
        if amount > 0:
            scores[turn.speaker] += amount
    return dict(scores)


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _has_overlap(start: float, end: float, turns: list[SpeakerTurn]) -> bool:
    active = {
        turn.speaker
        for turn in turns
        if _overlap(start, end, turn.start, turn.end) > max(0.05, (end - start) * 0.1)
    }
    return len(active) > 1


def _needs_review(segment: ASRSegment, overlap: bool, asr_engine: str) -> bool:
    if asr_engine == "mock-asr":
        return True
    text = segment.text
    if segment.confidence is not None and segment.confidence < 0.35:
        return True
    if segment.avg_logprob is not None and segment.avg_logprob < -1.0:
        return True
    if segment.no_speech_prob is not None and segment.no_speech_prob > 0.6:
        return True
    if segment.compression_ratio is not None and segment.compression_ratio > 2.4:
        return True
    if overlap:
        return True
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    if not normalized:
        return True
    duration = max(0.0, segment.end - segment.start)
    tokens = normalized.split()
    if duration > 15.0 and len(tokens) <= 2:
        return True
    if len(tokens) >= 8:
        top_count = Counter(tokens).most_common(1)[0][1]
        if top_count / len(tokens) > 0.45:
            return True
    compact = re.sub(r"\s+", "", normalized)
    if len(compact) >= 12:
        for width in (2, 3, 4):
            chunks = [compact[i : i + width] for i in range(0, len(compact), width)]
            if chunks and Counter(chunks).most_common(1)[0][1] / len(chunks) > 0.55:
                return True
    return False


def _segment_language(raw_language: str | None, text: str) -> SegmentLanguage:
    lang = (raw_language or "").lower()
    if lang in {"ko", "korean"}:
        return "ko"
    if lang in {"en", "english"}:
        return "en"
    has_hangul = bool(re.search(r"[\uac00-\ud7a3]", text))
    has_latin = bool(re.search(r"[A-Za-z]", text))
    if has_hangul and has_latin:
        return "mixed"
    if has_hangul:
        return "ko"
    if has_latin:
        return "en"
    return "unknown"
