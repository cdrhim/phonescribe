from __future__ import annotations

import math
import re
from pathlib import Path

from local_meetscribe.schemas import EvalReport, Transcript, load_transcript


def evaluate_transcripts(
    pred_path: Path,
    ref_path: Path,
    *,
    ref_rttm: Path | None = None,
) -> EvalReport:
    pred = load_transcript(pred_path)
    ref = load_transcript(ref_path)
    warnings: list[str] = []
    pred_text = _joined_clean_text(pred)
    ref_text = _joined_clean_text(ref)
    english_wer = _english_wer(pred_text, ref_text, warnings)
    korean_cer = _cer(_korean_chars(pred_text), _korean_chars(ref_text))
    spacing_cer = _cer(_spacing_normalized_korean(pred_text), _spacing_normalized_korean(ref_text))
    timestamp_mae = _timestamp_mae(pred, ref)
    der = _diarization_der(pred, ref_rttm, warnings)
    combined = _speaker_attributed_error(pred, ref)
    if korean_cer is None:
        warnings.append("No Korean characters found; Korean CER was not reported.")
    return EvalReport(
        english_wer=english_wer,
        korean_cer=korean_cer,
        korean_spacing_normalized_cer=spacing_cer,
        segment_timestamp_mae_sec=timestamp_mae,
        diarization_der=der,
        combined_speaker_attributed_error=combined,
        warnings=warnings,
    )


def _joined_clean_text(transcript: Transcript) -> str:
    return " ".join(segment.text_clean for segment in transcript.segments)


def _english_wer(pred_text: str, ref_text: str, warnings: list[str]) -> float | None:
    pred_en = " ".join(re.findall(r"[A-Za-z0-9']+", pred_text.lower()))
    ref_en = " ".join(re.findall(r"[A-Za-z0-9']+", ref_text.lower()))
    if not pred_en and not ref_en:
        return None
    try:
        from jiwer import wer
    except ImportError:
        warnings.append("Install the [dev] extra to compute English WER with jiwer.")
        return None
    return float(wer(ref_en, pred_en))


def _korean_chars(text: str) -> str:
    return "".join(re.findall(r"[\uac00-\ud7a3]", text))


def _spacing_normalized_korean(text: str) -> str:
    return re.sub(r"\s+", "", _korean_chars(text))


def _cer(pred: str, ref: str) -> float | None:
    if not pred and not ref:
        return None
    if not ref:
        return math.inf
    return _levenshtein(pred, ref) / len(ref)


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (ca != cb)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def _timestamp_mae(pred: Transcript, ref: Transcript) -> float | None:
    ref_by_id = {segment.id: segment for segment in ref.segments}
    errors: list[float] = []
    for segment in pred.segments:
        reference = ref_by_id.get(segment.id)
        if reference:
            errors.append(abs(segment.start - reference.start))
            errors.append(abs(segment.end - reference.end))
    if not errors:
        return None
    return sum(errors) / len(errors)


def _diarization_der(
    pred: Transcript,
    ref_rttm: Path | None,
    warnings: list[str],
) -> float | None:
    if ref_rttm is None:
        return None
    try:
        from pyannote.core import Annotation, Segment
        from pyannote.metrics.diarization import DiarizationErrorRate
    except ImportError:
        warnings.append("Install the [diarization] extra to compute DER with pyannote.metrics.")
        return None
    reference = _rttm_to_annotation(ref_rttm, Annotation, Segment)
    hypothesis = _transcript_to_annotation(pred, Annotation, Segment)
    if reference is None or hypothesis is None:
        warnings.append("DER was not reported because RTTM or hypothesis speaker turns were empty.")
        return None
    metric = DiarizationErrorRate()
    return float(metric(reference, hypothesis))


def _speaker_attributed_error(pred: Transcript, ref: Transcript) -> float | None:
    ref_by_id = {segment.id: segment for segment in ref.segments}
    total = 0
    errors = 0
    for segment in pred.segments:
        reference = ref_by_id.get(segment.id)
        if not reference:
            continue
        total += 1
        if (
            segment.speaker != reference.speaker
            or segment.text_clean.strip() != reference.text_clean.strip()
        ):
            errors += 1
    if total == 0:
        return None
    return errors / total


def _rttm_to_annotation(path: Path, annotation_cls, segment_cls):
    annotation = annotation_cls(uri=path.stem)
    any_turn = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        start = float(parts[3])
        duration = float(parts[4])
        speaker = parts[7]
        annotation[segment_cls(start, start + duration), "_"] = speaker
        any_turn = True
    return annotation if any_turn else None


def _transcript_to_annotation(transcript: Transcript, annotation_cls, segment_cls):
    annotation = annotation_cls(uri=transcript.id)
    any_turn = False
    for segment in transcript.segments:
        if segment.end <= segment.start:
            continue
        annotation[segment_cls(segment.start, segment.end), "_"] = segment.speaker
        any_turn = True
    return annotation if any_turn else None
