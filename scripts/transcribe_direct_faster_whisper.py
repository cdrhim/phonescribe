from __future__ import annotations

import argparse
import sys
from pathlib import Path

from local_meetscribe.config import get_settings
from local_meetscribe.pipeline.asr import ASRResult, ASRSegment, ASRWord
from local_meetscribe.pipeline.diarize import SpeakerTurn
from local_meetscribe.pipeline.export import write_exports
from local_meetscribe.pipeline.format import RuleBasedFormatterEngine
from local_meetscribe.pipeline.ingest import probe_media
from local_meetscribe.pipeline.merge import merge_transcript
from local_meetscribe.schemas import TranscriptConfig


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--word-timestamps", action="store_true")
    args = parser.parse_args()

    from faster_whisper import BatchedInferencePipeline, WhisperModel

    settings = get_settings()
    media_info = probe_media(args.input, settings)
    args.out.mkdir(parents=True, exist_ok=True)
    model_name = args.model or settings.faster_whisper_cpu_model
    model = WhisperModel(model_name, device="cpu", compute_type="int8", num_workers=1)
    pipeline = BatchedInferencePipeline(model=model)
    segments_iter, info = pipeline.transcribe(
        str(args.input),
        language=args.language,
        task="transcribe",
        batch_size=args.batch_size,
        beam_size=1,
        word_timestamps=args.word_timestamps,
        without_timestamps=not args.word_timestamps,
        condition_on_previous_text=False,
        temperature=0.0,
        vad_filter=True,
    )
    segments: list[ASRSegment] = []
    for segment in segments_iter:
        print(f"{segment.start:.1f}-{segment.end:.1f} {segment.text.strip()}", flush=True)
        words = [
            ASRWord(
                word=word.word.strip(),
                start=float(word.start),
                end=float(word.end),
                confidence=getattr(word, "probability", None),
            )
            for word in (segment.words or [])
            if word.word.strip()
        ]
        segments.append(
            ASRSegment(
                start=float(segment.start),
                end=float(segment.end),
                text=segment.text.strip(),
                language=getattr(info, "language", None),
                confidence=None,
                words=words,
                avg_logprob=getattr(segment, "avg_logprob", None),
                compression_ratio=getattr(segment, "compression_ratio", None),
                no_speech_prob=getattr(segment, "no_speech_prob", None),
            )
        )

    transcript = merge_transcript(
        job_id="direct-faster-whisper",
        media_info=media_info,
        config=TranscriptConfig(
            mode="cpu",
            asr_engine="faster-whisper",
            asr_model=f"{model_name} (cpu/int8/direct/batch={args.batch_size})",
            diarization_engine="single-speaker",
            language="auto",
        ),
        asr_result=ASRResult("faster-whisper", model_name, segments),
        speaker_turns=[
            SpeakerTurn(start=0.0, end=max(media_info.duration_sec, 0.001), speaker="SPEAKER_00")
        ],
    )
    transcript = RuleBasedFormatterEngine().clean(transcript, [])
    transcript = write_exports(transcript, args.out)
    print(f"TRANSCRIPT_JSON={transcript.exports.json_path}", flush=True)


if __name__ == "__main__":
    main()
