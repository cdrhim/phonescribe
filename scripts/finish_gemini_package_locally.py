from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from faster_whisper import BatchedInferencePipeline, WhisperModel
from local_meetscribe.pipeline.gemini import suggest_transcript_filename


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return payload


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _timestamp(seconds: float) -> str:
    whole = max(0, int(seconds))
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _chunk_text(segments: list[dict[str, object]], chunk_start: float) -> str:
    lines: list[str] = []
    for segment in segments:
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        absolute_start = chunk_start + float(str(segment.get("start") or 0.0))
        lines.append(f"[{_timestamp(absolute_start)}] SPEAKER_00: {text}")
    return "\n".join(lines).strip()


def _combine_chunks(chunks: list[dict[str, object]]) -> str:
    if len(chunks) == 1:
        return str(chunks[0].get("text") or "").strip()
    sections: list[str] = []
    for chunk in chunks:
        start = float(str(chunk.get("start_sec") or 0.0))
        end = float(str(chunk.get("end_sec") or 0.0))
        filename = str(chunk.get("filename") or "")
        text = str(chunk.get("text") or "").strip()
        sections.append(
            f"[{_timestamp(start)} - {_timestamp(end)} / {filename}]\n{text}"
        )
    return "\n\n".join(sections).strip()


def _checkpoint_payload(chunks: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "provider": "hybrid",
        "model": "gemini + faster-whisper-small",
        "updated_at": datetime.now(UTC).isoformat(),
        "chunks": chunks,
    }


def _write_checkpoint(path: Path, chunks: list[dict[str, object]]) -> None:
    _atomic_write_text(
        path,
        json.dumps(_checkpoint_payload(chunks), ensure_ascii=False, indent=2),
    )


def _artifact_chunk_names(payload: dict[str, Any]) -> list[str]:
    raw_chunks = payload.get("chunks")
    if not isinstance(raw_chunks, list):
        return []
    return [
        str(item.get("filename") or "")
        for item in raw_chunks
        if isinstance(item, dict)
    ]


def _completed_artifacts_match(
    txt_path: Path,
    json_path: Path,
    expected_names: list[str],
) -> bool:
    if not txt_path.is_file() or not json_path.is_file():
        return False
    try:
        text = txt_path.read_text(encoding="utf-8")
        payload = _read_object(json_path)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return False
    return bool(text) and payload.get("text") == text and _artifact_chunk_names(
        payload
    ) == expected_names


def _write_complete_progress(path: Path, total_chunks: int) -> None:
    payload = {
        "status": "complete",
        "completed_chunks": total_chunks,
        "total_chunks": total_chunks,
        "current_chunk": None,
        "elapsed_sec": 0.0,
        "eta_sec": 0.0,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def _heartbeat(
    stop: threading.Event,
    progress: dict[str, float | int],
) -> None:
    while not stop.wait(20.0):
        print(
            "local fallback heartbeat: "
            f"chunk {int(progress['chunk'])}/{int(progress['total'])}, "
            f"audio {progress['audio_sec']:.0f}/{progress['duration_sec']:.0f}s, "
            f"elapsed {time.monotonic() - progress['started_at']:.0f}s",
            flush=True,
        )


def finish_package(
    package_dir: Path,
    *,
    model_path: Path,
    cpu_threads: int,
    batch_size: int,
) -> None:
    manifest = _read_object(package_dir / "manifest.json")
    raw_manifest_chunks = manifest.get("chunks")
    if not isinstance(raw_manifest_chunks, list) or not raw_manifest_chunks:
        raise ValueError("The optimized package has no chunks.")
    manifest_chunks = [item for item in raw_manifest_chunks if isinstance(item, dict)]
    if len(manifest_chunks) != len(raw_manifest_chunks):
        raise ValueError("The optimized package has invalid chunk metadata.")

    final_txt = package_dir / "gemini_transcript.txt"
    final_json = package_dir / "gemini_transcript.json"
    progress_path = package_dir / "gemini_progress.json"
    expected_names = [str(item.get("filename") or "") for item in manifest_chunks]
    if final_txt.exists() and final_json.exists():
        if not _completed_artifacts_match(final_txt, final_json, expected_names):
            raise FileExistsError("Existing final transcript artifacts are inconsistent.")
        _write_complete_progress(progress_path, len(manifest_chunks))
        print("local fallback: valid final artifacts already exist", flush=True)
        return

    original_partial_path = package_dir / "gemini_transcript.partial.json"
    checkpoint_path = package_dir / "local_fallback.partial.json"
    if checkpoint_path.exists():
        checkpoint = _read_object(checkpoint_path)
        raw_completed = checkpoint.get("chunks")
    else:
        original_partial = _read_object(original_partial_path)
        raw_completed = original_partial.get("chunks")
    if not isinstance(raw_completed, list):
        raise ValueError("The partial transcript checkpoint is invalid.")
    completed = [dict(item) for item in raw_completed if isinstance(item, dict)]

    completed_by_name = {str(item.get("filename") or ""): item for item in completed}
    if any(name not in expected_names for name in completed_by_name):
        raise ValueError("The partial transcript contains an unexpected chunk.")
    missing = [
        item
        for item in manifest_chunks
        if str(item.get("filename")) not in completed_by_name
    ]
    if not missing:
        print("local fallback: all chunks already checkpointed", flush=True)
    else:
        print(
            f"local fallback: loading small model for {len(missing)} remaining chunks",
            flush=True,
        )
        model = WhisperModel(
            str(model_path),
            device="cpu",
            compute_type="int8",
            cpu_threads=cpu_threads,
            num_workers=1,
        )
        pipeline = BatchedInferencePipeline(model=model)
        source = manifest.get("source")
        source_filename = (
            str(source.get("filename") or "") if isinstance(source, dict) else ""
        )
        initial_prompt = Path(source_filename).stem or None

        for manifest_index, chunk_meta in enumerate(manifest_chunks, start=1):
            filename = str(chunk_meta.get("filename") or "")
            if filename in completed_by_name:
                continue
            chunk_path = package_dir / filename
            if not chunk_path.is_file() or chunk_path.stat().st_size <= 0:
                raise FileNotFoundError(f"Missing optimized chunk: {filename}")
            chunk_start = float(chunk_meta.get("start_sec") or 0.0)
            chunk_end = float(chunk_meta.get("end_sec") or 0.0)
            duration = max(0.0, chunk_end - chunk_start)
            started_at = time.monotonic()
            progress: dict[str, float | int] = {
                "chunk": manifest_index,
                "total": len(manifest_chunks),
                "audio_sec": 0.0,
                "duration_sec": duration,
                "started_at": started_at,
            }
            stop = threading.Event()
            heartbeat = threading.Thread(
                target=_heartbeat,
                args=(stop, progress),
                name="local-fallback-heartbeat",
                daemon=True,
            )
            heartbeat.start()
            print(
                f"local fallback: chunk {manifest_index}/{len(manifest_chunks)} started "
                f"({duration:.0f}s audio)",
                flush=True,
            )
            local_segments: list[dict[str, object]] = []
            try:
                segments, _info = pipeline.transcribe(
                    str(chunk_path),
                    language=None,
                    task="transcribe",
                    log_progress=False,
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    initial_prompt=initial_prompt,
                    word_timestamps=False,
                    without_timestamps=False,
                    multilingual=True,
                    vad_filter=True,
                    batch_size=batch_size,
                )
                next_report = 0.1
                for segment in segments:
                    segment_end = max(0.0, float(segment.end))
                    progress["audio_sec"] = min(duration, segment_end)
                    text = str(segment.text or "").strip()
                    if text:
                        local_segments.append(
                            {
                                "start": max(0.0, float(segment.start)),
                                "end": segment_end,
                                "text": text,
                            }
                        )
                    ratio = segment_end / duration if duration > 0 else 1.0
                    if ratio >= next_report:
                        print(
                            f"local fallback: chunk {manifest_index} audio progress "
                            f"{min(100, int(ratio * 100))}%",
                            flush=True,
                        )
                        next_report += 0.1
            finally:
                stop.set()
                heartbeat.join(timeout=1.0)
            text = _chunk_text(local_segments, chunk_start)
            if not text:
                raise ValueError(f"Local ASR returned no transcript for {filename}.")
            completed_chunk: dict[str, object] = {
                "filename": filename,
                "start_sec": chunk_start,
                "end_sec": chunk_end,
                "delivery": "local_cpu",
                "mime_type": "audio/mp3",
                "text": text,
                "model": "faster-whisper-small",
            }
            completed_by_name[filename] = completed_chunk
            completed = [
                completed_by_name[name] for name in expected_names if name in completed_by_name
            ]
            _write_checkpoint(checkpoint_path, completed)
            print(
                f"local fallback: chunk {manifest_index}/{len(manifest_chunks)} checkpointed "
                f"after {time.monotonic() - started_at:.0f}s",
                flush=True,
            )

    ordered = [completed_by_name[name] for name in expected_names if name in completed_by_name]
    if len(ordered) != len(expected_names):
        raise ValueError("Not every manifest chunk has a transcript checkpoint.")
    combined_text = _combine_chunks(ordered)
    if not combined_text:
        raise ValueError("The combined transcript is empty.")
    source = manifest.get("source")
    source_filename = (
        str(source.get("filename") or "transcript")
        if isinstance(source, dict)
        else "transcript"
    )
    suggested_filename = suggest_transcript_filename(source_filename, combined_text)
    models = list(dict.fromkeys(str(item.get("model") or "") for item in ordered))
    payload = {
        "provider": "hybrid",
        "model": " + ".join(model for model in models if model),
        "created_at": datetime.now(UTC).isoformat(),
        "source": manifest.get("source"),
        "recommendation": manifest.get("recommendation"),
        "suggested_filename": suggested_filename,
        "chunks": ordered,
        "text": combined_text,
    }
    if final_txt.exists() and final_txt.read_text(encoding="utf-8") != combined_text:
        raise FileExistsError("Existing final TXT does not match the recovered transcript.")
    if final_json.exists():
        existing_payload = _read_object(final_json)
        if (
            existing_payload.get("text") != combined_text
            or _artifact_chunk_names(existing_payload) != expected_names
        ):
            raise FileExistsError("Existing final JSON does not match the recovered transcript.")
    if not final_txt.exists():
        _atomic_write_text(final_txt, combined_text)
    if not final_json.exists():
        _atomic_write_text(final_json, json.dumps(payload, ensure_ascii=False, indent=2))
    _write_complete_progress(progress_path, len(manifest_chunks))
    print("local fallback: final TXT and JSON written successfully", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finish an interrupted Gemini package with local faster-whisper.",
    )
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    finish_package(
        args.package_dir.resolve(),
        model_path=args.model_path.resolve(),
        cpu_threads=max(1, args.cpu_threads),
        batch_size=max(1, args.batch_size),
    )


if __name__ == "__main__":
    main()
