from __future__ import annotations

import base64
import json
import re
import time
import uuid
from collections import Counter
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from local_meetscribe.config import Settings
from local_meetscribe.utils.errors import LocalMeetScribeError

GeminiDelivery = Literal["inline", "files_api"]
GeminiProgressStatus = Literal["idle", "transcribing", "complete", "failed"]

INLINE_LIMIT_BYTES = 20 * 1024 * 1024
MAX_REQUEST_ATTEMPTS = 3
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MODEL_FALLBACK_STATUS_CODES = RETRYABLE_STATUS_CODES | {400, 404}
GEMINI_FALLBACK_MODELS = ("gemini-3.5-flash", "gemini-3.5-flash-lite")
INTERACTION_POLL_ATTEMPTS = 120
PARTIAL_TRANSCRIPT_FILENAME = "gemini_transcript.partial.json"
PROGRESS_FILENAME = "gemini_progress.json"
DEFAULT_TRANSCRIPTION_PROMPT = (
    "Generate a faithful Korean/English meeting transcript from this audio. "
    "Keep spoken transcript text only. Use timestamps at segment starts. Use rough speaker "
    "labels only when they are clear from the audio. If speech is not recognizable, write "
    "[inaudible] instead of guessing. Do not invent missing words, summaries, action items, "
    "or topics that were not spoken."
)


@dataclass(frozen=True)
class GeminiChunkTranscript:
    filename: str
    start_sec: float
    end_sec: float
    delivery: GeminiDelivery
    mime_type: str
    text: str
    model: str | None = None


@dataclass(frozen=True)
class _GeminiGeneration:
    text: str
    model: str


@dataclass(frozen=True)
class GeminiTranscriptResult:
    provider: str
    model: str
    text: str
    suggested_filename: str
    chunks: list[GeminiChunkTranscript]
    txt_path: Path
    json_path: Path


@dataclass(frozen=True)
class GeminiTranscriptionProgress:
    status: GeminiProgressStatus
    completed_chunks: int
    total_chunks: int
    current_chunk: int | None
    progress: float
    elapsed_sec: float
    eta_sec: float | None


def transcribe_gemini_package(
    package_dir: Path,
    settings: Settings,
    *,
    api_key: str | None = None,
) -> GeminiTranscriptResult:
    request_api_key = (api_key or "").strip()
    if request_api_key:
        settings = replace(
            settings,
            enable_gemini_transcription=True,
            gemini_api_key=request_api_key,
        )
    if not settings.enable_gemini_transcription:
        raise LocalMeetScribeError(
            "Gemini transcription is off. Enter a Gemini API key in the page, or set "
            "LOCAL_MEETSCRIBE_ENABLE_GEMINI_TRANSCRIPTION=true and restart the server."
        )
    if not settings.gemini_api_key:
        raise LocalMeetScribeError("Set GEMINI_API_KEY before using Gemini transcription.")

    manifest_path = package_dir / "manifest.json"
    if not manifest_path.exists():
        raise LocalMeetScribeError("Optimized package manifest is missing.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks_meta = manifest.get("chunks")
    if not isinstance(chunks_meta, list) or not chunks_meta:
        raise LocalMeetScribeError("Optimized package does not contain audio chunks.")

    valid_chunks_meta = [item for item in chunks_meta if isinstance(item, dict)]
    if not valid_chunks_meta:
        raise LocalMeetScribeError("Optimized package does not contain valid audio chunks.")

    completed_result = _load_completed_transcript_result(
        package_dir,
        valid_chunks_meta,
        settings.gemini_model,
    )
    if completed_result is not None:
        _write_progress_safely(
            package_dir / PROGRESS_FILENAME,
            status="complete",
            completed_chunks=len(valid_chunks_meta),
            total_chunks=len(valid_chunks_meta),
            current_chunk=None,
            elapsed_sec=0.0,
            eta_sec=0.0,
            average_chunk_sec=None,
            started_at=datetime.now(UTC).isoformat(),
        )
        return completed_result

    httpx = _load_httpx()
    partial_path = package_dir / PARTIAL_TRANSCRIPT_FILENAME
    progress_path = package_dir / PROGRESS_FILENAME
    completed = _load_partial_transcripts(partial_path, settings.gemini_model)
    transcripts: list[GeminiChunkTranscript] = []
    manifest_filenames = {
        str(item.get("filename") or "") for item in valid_chunks_meta if item.get("filename")
    }
    completed_before_run = len(manifest_filenames.intersection(completed))
    completed_count = completed_before_run
    total_chunks = len(valid_chunks_meta)
    previous_average = _load_progress_average(progress_path)
    average_chunk_sec = previous_average
    run_started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    current_chunk = next(
        (
            index
            for index, item in enumerate(valid_chunks_meta, start=1)
            if str(item.get("filename") or "") not in completed
        ),
        None,
    )
    _write_progress_safely(
        progress_path,
        status="transcribing",
        completed_chunks=completed_count,
        total_chunks=total_chunks,
        current_chunk=current_chunk,
        elapsed_sec=0.0,
        eta_sec=(
            average_chunk_sec * (total_chunks - completed_count)
            if average_chunk_sec is not None
            else None
        ),
        average_chunk_sec=average_chunk_sec,
        started_at=started_at,
    )

    try:
        with httpx.Client(timeout=httpx.Timeout(60.0, read=900.0)) as client:
            for index, chunk_meta in enumerate(valid_chunks_meta, start=1):
                filename = str(chunk_meta.get("filename") or "")
                chunk_path = package_dir / filename
                if not filename or not chunk_path.exists():
                    raise LocalMeetScribeError(
                        f"Optimized chunk is missing: {filename or 'unknown'}"
                    )
                cached = completed.get(filename)
                if cached is not None:
                    transcripts.append(cached)
                    continue
                current_chunk = index
                elapsed_sec = time.monotonic() - run_started
                _write_progress_safely(
                    progress_path,
                    status="transcribing",
                    completed_chunks=completed_count,
                    total_chunks=total_chunks,
                    current_chunk=current_chunk,
                    elapsed_sec=elapsed_sec,
                    eta_sec=(
                        average_chunk_sec * (total_chunks - completed_count)
                        if average_chunk_sec is not None
                        else None
                    ),
                    average_chunk_sec=average_chunk_sec,
                    started_at=started_at,
                )
                mime_type = gemini_mime_type_for_path(chunk_path)
                prompt = _chunk_prompt(
                    float(chunk_meta.get("start_sec") or 0.0),
                    float(chunk_meta.get("end_sec") or 0.0),
                )
                if can_send_gemini_inline(chunk_path):
                    delivery: GeminiDelivery = "inline"
                    generation = _generate_inline(
                        client,
                        settings,
                        chunk_path,
                        mime_type,
                        prompt,
                    )
                else:
                    delivery = "files_api"
                    file_info = _upload_file(client, settings, chunk_path, mime_type)
                    try:
                        file_info = _wait_for_file_active(client, settings, file_info)
                        generation = _generate_from_file(
                            client,
                            settings,
                            file_info,
                            mime_type,
                            prompt,
                        )
                    finally:
                        _delete_uploaded_file(client, settings, file_info)
                transcripts.append(
                    GeminiChunkTranscript(
                        filename=filename,
                        start_sec=float(chunk_meta.get("start_sec") or 0.0),
                        end_sec=float(chunk_meta.get("end_sec") or 0.0),
                        delivery=delivery,
                        mime_type=mime_type,
                        text=generation.text.strip(),
                        model=generation.model,
                    )
                )
                _write_partial_transcripts(
                    partial_path,
                    settings.gemini_model,
                    transcripts,
                )
                completed_count += 1
                elapsed_sec = time.monotonic() - run_started
                completed_this_run = completed_count - completed_before_run
                if previous_average is not None and completed_before_run:
                    average_chunk_sec = (
                        previous_average * completed_before_run + elapsed_sec
                    ) / (completed_before_run + completed_this_run)
                else:
                    average_chunk_sec = elapsed_sec / completed_this_run
                _write_progress_safely(
                    progress_path,
                    status="transcribing",
                    completed_chunks=completed_count,
                    total_chunks=total_chunks,
                    current_chunk=None,
                    elapsed_sec=elapsed_sec,
                    eta_sec=average_chunk_sec * (total_chunks - completed_count),
                    average_chunk_sec=average_chunk_sec,
                    started_at=started_at,
                )
    except Exception:
        elapsed_sec = time.monotonic() - run_started
        _write_progress_safely(
            progress_path,
            status="failed",
            completed_chunks=completed_count,
            total_chunks=total_chunks,
            current_chunk=current_chunk,
            elapsed_sec=elapsed_sec,
            eta_sec=(
                average_chunk_sec * (total_chunks - completed_count)
                if average_chunk_sec is not None
                else None
            ),
            average_chunk_sec=average_chunk_sec,
            started_at=started_at,
        )
        raise

    if not transcripts:
        raise LocalMeetScribeError("Gemini transcription did not process any chunks.")

    text = _combine_chunk_text(transcripts)
    source = manifest.get("source")
    source_filename = (
        str(source.get("filename") or "transcript")
        if isinstance(source, dict)
        else "transcript"
    )
    suggested_filename = suggest_transcript_filename(source_filename, text)
    txt_path = package_dir / "gemini_transcript.txt"
    json_path = package_dir / "gemini_transcript.json"
    _atomic_write_text(txt_path, text)
    result_model = _result_model(transcripts, settings.gemini_model)
    payload = {
        "provider": "gemini",
        "model": result_model,
        "created_at": datetime.now(UTC).isoformat(),
        "source": manifest.get("source"),
        "recommendation": manifest.get("recommendation"),
        "suggested_filename": suggested_filename,
        "chunks": [asdict(chunk) for chunk in transcripts],
        "text": text,
    }
    _atomic_write_text(json_path, json.dumps(payload, ensure_ascii=False, indent=2))
    _best_effort_unlink(partial_path)
    _write_progress_safely(
        progress_path,
        status="complete",
        completed_chunks=total_chunks,
        total_chunks=total_chunks,
        current_chunk=None,
        elapsed_sec=time.monotonic() - run_started,
        eta_sec=0.0,
        average_chunk_sec=average_chunk_sec,
        started_at=started_at,
    )
    return GeminiTranscriptResult(
        provider="gemini",
        model=result_model,
        text=text,
        suggested_filename=suggested_filename,
        chunks=transcripts,
        txt_path=txt_path,
        json_path=json_path,
    )


def gemini_mime_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".mp3":
        return "audio/mp3"
    if suffix == ".wav":
        return "audio/wav"
    if suffix in {".aac", ".m4a"}:
        return "audio/aac"
    if suffix == ".ogg":
        return "audio/ogg"
    if suffix == ".flac":
        return "audio/flac"
    if suffix in {".aif", ".aiff"}:
        return "audio/aiff"
    raise LocalMeetScribeError(
        "Gemini transcription supports mp3, wav, aac/m4a, ogg, flac, and aiff packages."
    )


def can_send_gemini_inline(path: Path) -> bool:
    return path.stat().st_size <= INLINE_LIMIT_BYTES


def get_gemini_progress(package_dir: Path) -> GeminiTranscriptionProgress:
    manifest_path = package_dir / "manifest.json"
    if not manifest_path.exists():
        raise LocalMeetScribeError("Optimized package manifest is missing.")
    manifest = _read_json_object(manifest_path)
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list):
        raise LocalMeetScribeError("Optimized package does not contain audio chunks.")
    valid_chunks = [item for item in chunks if isinstance(item, dict)]
    total_chunks = len(valid_chunks)
    if total_chunks == 0:
        raise LocalMeetScribeError("Optimized package does not contain audio chunks.")

    if (
        (package_dir / "gemini_transcript.json").exists()
        and (package_dir / "gemini_transcript.txt").exists()
    ):
        return GeminiTranscriptionProgress(
            status="complete",
            completed_chunks=total_chunks,
            total_chunks=total_chunks,
            current_chunk=None,
            progress=1.0,
            elapsed_sec=0.0,
            eta_sec=0.0,
        )

    payload = _read_json_object(package_dir / PROGRESS_FILENAME)
    status_value = str(payload.get("status") or "")
    if status_value in {"idle", "transcribing", "complete", "failed"}:
        completed_chunks = min(
            total_chunks,
            max(0, int(payload.get("completed_chunks") or 0)),
        )
        current_value = payload.get("current_chunk")
        current_chunk = (
            int(current_value)
            if isinstance(current_value, int) and 1 <= current_value <= total_chunks
            else None
        )
        eta_value = _safe_nonnegative_float(payload.get("eta_sec"))
        return GeminiTranscriptionProgress(
            status=status_value,  # type: ignore[arg-type]
            completed_chunks=completed_chunks,
            total_chunks=total_chunks,
            current_chunk=current_chunk,
            progress=completed_chunks / total_chunks,
            elapsed_sec=_safe_nonnegative_float(payload.get("elapsed_sec")) or 0.0,
            eta_sec=eta_value,
        )

    if (package_dir / "gemini_transcript.json").exists():
        return GeminiTranscriptionProgress(
            status="complete",
            completed_chunks=total_chunks,
            total_chunks=total_chunks,
            current_chunk=None,
            progress=1.0,
            elapsed_sec=0.0,
            eta_sec=0.0,
        )

    partial = _read_json_object(package_dir / PARTIAL_TRANSCRIPT_FILENAME)
    partial_chunks = partial.get("chunks")
    filenames = {
        str(item.get("filename") or "")
        for item in valid_chunks
        if item.get("filename")
    }
    completed_chunks = 0
    if isinstance(partial_chunks, list):
        completed_chunks = len(
            {
                str(item.get("filename") or "")
                for item in partial_chunks
                if isinstance(item, dict)
                and str(item.get("filename") or "") in filenames
            }
        )
    return GeminiTranscriptionProgress(
        status="idle",
        completed_chunks=completed_chunks,
        total_chunks=total_chunks,
        current_chunk=None,
        progress=completed_chunks / total_chunks,
        elapsed_sec=0.0,
        eta_sec=None,
    )


def suggest_transcript_filename(source_filename: str, transcript: str) -> str:
    original_stem = _filename_fragment(Path(source_filename).stem) or "transcript"
    date_match = re.search(r"(?<!\d)(\d{6}|\d{8})(?!\d)", original_stem)
    date_prefix = date_match.group(1) if date_match else ""
    cleaned = re.sub(r"\[[^\]]{1,80}\]", " ", transcript)
    cleaned = re.sub(
        r"(?im)^\s*(speaker|화자|참석자)[ _-]*\d*\s*:\s*",
        " ",
        cleaned,
    )
    stopwords = {
        "about",
        "and",
        "are",
        "for",
        "from",
        "have",
        "just",
        "meeting",
        "okay",
        "our",
        "that",
        "the",
        "their",
        "there",
        "they",
        "this",
        "was",
        "were",
        "with",
        "yeah",
        "you",
        "그리고",
        "그냥",
        "그런",
        "그래서",
        "내용",
        "대한",
        "말씀",
        "부분",
        "오늘",
        "이런",
        "저희",
        "제가",
        "지금",
        "하는",
        "합니다",
        "관련",
        "있습니다",
    }
    tokens: list[tuple[str, str, int]] = []
    for position, match in enumerate(re.finditer(r"[A-Za-z가-힣][A-Za-z0-9가-힣]{1,24}", cleaned)):
        display = match.group(0)
        key = display.casefold()
        if key in stopwords:
            continue
        tokens.append((key, display, position))
    counts = Counter(key for key, _, _ in tokens)
    first_seen: dict[str, tuple[str, int]] = {}
    for key, display, position in tokens:
        first_seen.setdefault(key, (display, position))
    ranked = sorted(
        counts,
        key=lambda key: (-counts[key], first_seen[key][1]),
    )
    terms = [first_seen[key][0] for key in ranked[:3]]
    if not terms:
        return original_stem
    pieces = ([date_prefix] if date_prefix else []) + terms
    return _filename_fragment("_".join(pieces)) or original_stem


def _filename_fragment(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", value)
    cleaned = re.sub(r"[\s_]+", "_", cleaned).strip(" ._-")
    return cleaned[:96].rstrip(" ._-")


def _load_httpx() -> Any:
    try:
        import httpx
    except ImportError as exc:
        raise LocalMeetScribeError(
            "Gemini transcription requires optional dependencies. Install with: "
            "pip install -e .[llm]"
        ) from exc
    return httpx


def _chunk_prompt(start_sec: float, end_sec: float) -> str:
    return (
        DEFAULT_TRANSCRIPTION_PROMPT
        + "\nThis optimized chunk maps to original audio time "
        + f"{_format_timestamp(start_sec)} through {_format_timestamp(end_sec)}. "
        + "When possible, use original-time timestamps."
    )


def _generate_inline(
    client: Any,
    settings: Settings,
    audio_path: Path,
    mime_type: str,
    prompt: str,
) -> _GeminiGeneration:
    return _generate_interaction(
        client,
        settings,
        prompt,
        {
            "type": "audio",
            "mime_type": mime_type,
            "data": base64.b64encode(audio_path.read_bytes()).decode("ascii"),
        },
    )


def _upload_file(
    client: Any,
    settings: Settings,
    audio_path: Path,
    mime_type: str,
) -> dict[str, Any]:
    size = audio_path.stat().st_size
    start_response = _request_with_retry(
        client,
        "POST",
        _upload_url(settings),
        headers={
            **_api_headers(settings),
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(size),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json",
        },
        json={"file": {"display_name": audio_path.name}},
    )
    _raise_for_gemini_error(start_response)
    upload_url = start_response.headers.get("x-goog-upload-url")
    if not upload_url:
        raise LocalMeetScribeError("Gemini Files API did not return an upload URL.")

    with audio_path.open("rb") as audio_file:
        upload_response = client.post(
            upload_url,
            headers={
                "Content-Length": str(size),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            },
            content=audio_file,
        )
    _raise_for_gemini_error(upload_response)
    return upload_response.json()


def _wait_for_file_active(
    client: Any,
    settings: Settings,
    file_info: dict[str, Any],
) -> dict[str, Any]:
    file_obj = _file_obj(file_info)
    file_name = str(file_obj.get("name") or "")
    if not file_name:
        return file_info

    for _ in range(30):
        state = str(file_obj.get("state") or "ACTIVE").upper()
        if state == "ACTIVE":
            return {"file": file_obj}
        if state == "FAILED":
            raise LocalMeetScribeError("Gemini Files API failed to process the uploaded audio.")
        time.sleep(2)
        response = _request_with_retry(
            client,
            "GET",
            f"{settings.gemini_api_base}/{file_name}",
            headers=_api_headers(settings),
        )
        _raise_for_gemini_error(response)
        file_obj = _file_obj(response.json())
    raise LocalMeetScribeError("Gemini Files API upload is still processing. Try again shortly.")


def _generate_from_file(
    client: Any,
    settings: Settings,
    file_info: dict[str, Any],
    fallback_mime_type: str,
    prompt: str,
) -> _GeminiGeneration:
    file_obj = _file_obj(file_info)
    file_uri = file_obj.get("uri")
    if not file_uri:
        raise LocalMeetScribeError("Gemini Files API upload did not return a file URI.")
    mime_type = str(file_obj.get("mimeType") or file_obj.get("mime_type") or fallback_mime_type)
    return _generate_interaction(
        client,
        settings,
        prompt,
        {"type": "audio", "mime_type": mime_type, "uri": file_uri},
    )


def _generate_interaction(
    client: Any,
    settings: Settings,
    prompt: str,
    audio_input: dict[str, str],
) -> _GeminiGeneration:
    last_response: Any | None = None
    last_exception: LocalMeetScribeError | None = None
    for model in _model_candidates(settings.gemini_model):
        payload = {
            "model": model,
            "input": [
                {"type": "text", "text": prompt},
                audio_input,
            ],
        }
        try:
            response = _request_with_retry(
                client,
                "POST",
                _interactions_url(settings),
                headers=_api_headers(settings),
                json=payload,
            )
        except LocalMeetScribeError as exc:
            last_exception = exc
            continue
        if response.status_code < 400:
            try:
                interaction = _wait_for_interaction_completion(
                    client,
                    settings,
                    response.json(),
                )
                return _GeminiGeneration(
                    text=_extract_interaction_text(interaction),
                    model=model,
                )
            except LocalMeetScribeError as exc:
                last_exception = exc
                continue
        last_response = response
        if response.status_code not in MODEL_FALLBACK_STATUS_CODES:
            break

    if last_response is not None:
        _raise_for_gemini_error(last_response)
    if last_exception is not None:
        raise last_exception
    raise LocalMeetScribeError("Gemini did not process the audio request.")


def _delete_uploaded_file(client: Any, settings: Settings, file_info: dict[str, Any]) -> None:
    file_name = str(_file_obj(file_info).get("name") or "")
    if not file_name:
        return
    try:
        client.delete(f"{settings.gemini_api_base}/{file_name}", headers=_api_headers(settings))
    except Exception:
        return


def _extract_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise LocalMeetScribeError("Gemini did not return a transcript candidate.")
    parts = candidates[0].get("content", {}).get("parts", [])
    texts = [str(part["text"]) for part in parts if isinstance(part, dict) and "text" in part]
    text = "\n".join(text.strip() for text in texts if text.strip()).strip()
    if not text:
        raise LocalMeetScribeError("Gemini returned an empty transcript.")
    return text


def _extract_interaction_text(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    steps = payload.get("steps")
    texts: list[str] = []
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict) or step.get("type") != "model_output":
                continue
            texts.extend(_text_items(step.get("content")))

    outputs = payload.get("outputs")
    if isinstance(outputs, list):
        texts.extend(_text_items(outputs))

    text = "\n".join(texts).strip()
    if not text:
        status = str(payload.get("status") or "unknown")
        raise LocalMeetScribeError(
            f"Gemini returned no transcript text (interaction status: {status})."
        )
    return text


def _text_items(items: object) -> list[str]:
    if not isinstance(items, list):
        return []
    values: list[str] = []
    for item in items:
        if isinstance(item, dict) and item.get("type") == "text":
            value = str(item.get("text") or "").strip()
            if value:
                values.append(value)
    return values


def _wait_for_interaction_completion(
    client: Any,
    settings: Settings,
    payload: dict[str, Any],
) -> dict[str, Any]:
    status = str(payload.get("status") or "").casefold()
    if status != "in_progress":
        return payload
    interaction_id = str(payload.get("id") or "").strip()
    if not interaction_id:
        raise LocalMeetScribeError("Gemini interaction is in progress without a recovery ID.")

    for attempt in range(INTERACTION_POLL_ATTEMPTS):
        time.sleep(min(5.0, 1.0 + attempt * 0.25))
        response = _request_with_retry(
            client,
            "GET",
            f"{_interactions_url(settings)}/{interaction_id}",
            headers=_api_headers(settings),
        )
        _raise_for_gemini_error(response)
        payload = response.json()
        status = str(payload.get("status") or "").casefold()
        if status != "in_progress":
            return payload
    raise LocalMeetScribeError("Gemini interaction did not complete before the recovery timeout.")


def _raise_for_gemini_error(response: Any) -> None:
    if response.status_code < 400:
        return
    detail = f"Gemini API returned HTTP {response.status_code}."
    try:
        payload = response.json()
        message = payload.get("error", {}).get("message")
        if message:
            detail = f"Gemini API error: {message}"
    except Exception:
        pass
    if response.status_code == 429:
        detail += (
            " The Gemini free-tier quota may be exhausted. Wait for it to reset or check "
            "the active limits in Google AI Studio."
        )
    elif response.status_code in {500, 502, 503, 504}:
        detail = (
            "Gemini is temporarily unavailable after automatic retries. "
            "The optimized audio is saved; retry transcription to continue."
        )
    raise LocalMeetScribeError(detail)


def _request_with_retry(client: Any, method: str, url: str, **kwargs: Any) -> Any:
    for attempt in range(MAX_REQUEST_ATTEMPTS):
        try:
            response = client.request(method, url, **kwargs)
        except Exception as exc:
            if attempt + 1 >= MAX_REQUEST_ATTEMPTS:
                raise LocalMeetScribeError(
                    "Gemini API network request failed after automatic retries."
                ) from exc
            time.sleep(2**attempt)
            continue
        if (
            response.status_code not in RETRYABLE_STATUS_CODES
            or attempt + 1 >= MAX_REQUEST_ATTEMPTS
        ):
            return response
        time.sleep(_retry_delay(response, attempt))
    raise LocalMeetScribeError("Gemini API request failed after automatic retries.")


def _retry_delay(response: Any, attempt: int) -> float:
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            return min(30.0, max(1.0, float(retry_after)))
        except ValueError:
            pass
    return min(15.0, float(3 * (2**attempt)))


def _api_headers(settings: Settings) -> dict[str, str]:
    return {"x-goog-api-key": settings.gemini_api_key or ""}


def _interactions_url(settings: Settings) -> str:
    return f"{settings.gemini_api_base}/interactions"


def _model_candidates(configured_model: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys((configured_model, *GEMINI_FALLBACK_MODELS)))


def _result_model(chunks: list[GeminiChunkTranscript], configured_model: str) -> str:
    models = list(dict.fromkeys(chunk.model or configured_model for chunk in chunks))
    return " + ".join(models) if models else configured_model


def _upload_url(settings: Settings) -> str:
    return f"{_upload_base(settings)}/files"


def _upload_base(settings: Settings) -> str:
    return settings.gemini_api_base.replace("/v1beta", "/upload/v1beta", 1)


def _file_obj(file_info: dict[str, Any]) -> dict[str, Any]:
    file_obj = file_info.get("file", file_info)
    return file_obj if isinstance(file_obj, dict) else {}


def _combine_chunk_text(chunks: list[GeminiChunkTranscript]) -> str:
    if len(chunks) == 1:
        return chunks[0].text.strip()
    sections = []
    for chunk in chunks:
        sections.append(
            f"[{_format_timestamp(chunk.start_sec)} - {_format_timestamp(chunk.end_sec)} / "
            f"{chunk.filename}]\n{chunk.text.strip()}"
        )
    return "\n\n".join(sections).strip()


def _load_partial_transcripts(
    path: Path,
    model: str,
) -> dict[str, GeminiChunkTranscript]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if payload.get("model") != model or not isinstance(payload.get("chunks"), list):
        return {}
    completed: dict[str, GeminiChunkTranscript] = {}
    for item in payload["chunks"]:
        if not isinstance(item, dict):
            continue
        filename = str(item.get("filename") or "")
        delivery = str(item.get("delivery") or "")
        text = str(item.get("text") or "").strip()
        if not filename or not text or delivery not in {"inline", "files_api"}:
            continue
        completed[filename] = GeminiChunkTranscript(
            filename=filename,
            start_sec=float(item.get("start_sec") or 0.0),
            end_sec=float(item.get("end_sec") or 0.0),
            delivery=delivery,  # type: ignore[arg-type]
            mime_type=str(item.get("mime_type") or "audio/mp3"),
            text=text,
            model=str(item.get("model") or model),
        )
    return completed


def _load_completed_transcript_result(
    package_dir: Path,
    chunks_meta: list[dict[str, Any]],
    configured_model: str,
) -> GeminiTranscriptResult | None:
    json_path = package_dir / "gemini_transcript.json"
    txt_path = package_dir / "gemini_transcript.txt"
    if not json_path.exists() or not txt_path.exists():
        return None
    payload = _read_json_object(json_path)
    text = str(payload.get("text") or "").strip()
    raw_chunks = payload.get("chunks")
    if not text or not isinstance(raw_chunks, list):
        return None

    expected_filenames = {
        str(item.get("filename") or "") for item in chunks_meta if item.get("filename")
    }
    chunks: list[GeminiChunkTranscript] = []
    for item in raw_chunks:
        if not isinstance(item, dict):
            continue
        filename = str(item.get("filename") or "")
        delivery = str(item.get("delivery") or "")
        chunk_text = str(item.get("text") or "").strip()
        if not filename or not chunk_text or delivery not in {"inline", "files_api"}:
            continue
        chunks.append(
            GeminiChunkTranscript(
                filename=filename,
                start_sec=float(item.get("start_sec") or 0.0),
                end_sec=float(item.get("end_sec") or 0.0),
                delivery=delivery,  # type: ignore[arg-type]
                mime_type=str(item.get("mime_type") or "audio/mp3"),
                text=chunk_text,
                model=str(item.get("model") or configured_model),
            )
        )
    if {chunk.filename for chunk in chunks} != expected_filenames:
        return None

    return GeminiTranscriptResult(
        provider="gemini",
        model=str(payload.get("model") or configured_model),
        text=text,
        suggested_filename=str(payload.get("suggested_filename") or "transcript"),
        chunks=chunks,
        txt_path=txt_path,
        json_path=json_path,
    )


def _write_partial_transcripts(
    path: Path,
    model: str,
    chunks: list[GeminiChunkTranscript],
) -> None:
    payload = {
        "model": model,
        "updated_at": datetime.now(UTC).isoformat(),
        "chunks": [asdict(chunk) for chunk in chunks],
    }
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def _write_progress_safely(
    path: Path,
    *,
    status: GeminiProgressStatus,
    completed_chunks: int,
    total_chunks: int,
    current_chunk: int | None,
    elapsed_sec: float,
    eta_sec: float | None,
    average_chunk_sec: float | None,
    started_at: str,
) -> None:
    try:
        _write_progress(
            path,
            status=status,
            completed_chunks=completed_chunks,
            total_chunks=total_chunks,
            current_chunk=current_chunk,
            elapsed_sec=elapsed_sec,
            eta_sec=eta_sec,
            average_chunk_sec=average_chunk_sec,
            started_at=started_at,
        )
    except OSError:
        # Progress is advisory. A transient Windows file lock must not discard a transcript.
        return


def _write_progress(
    path: Path,
    *,
    status: GeminiProgressStatus,
    completed_chunks: int,
    total_chunks: int,
    current_chunk: int | None,
    elapsed_sec: float,
    eta_sec: float | None,
    average_chunk_sec: float | None,
    started_at: str,
) -> None:
    payload = {
        "status": status,
        "completed_chunks": completed_chunks,
        "total_chunks": total_chunks,
        "current_chunk": current_chunk,
        "progress": completed_chunks / total_chunks if total_chunks else 0.0,
        "elapsed_sec": round(max(0.0, elapsed_sec), 1),
        "eta_sec": round(max(0.0, eta_sec), 1) if eta_sec is not None else None,
        "average_chunk_sec": (
            round(max(0.0, average_chunk_sec), 1)
            if average_chunk_sec is not None
            else None
        ),
        "started_at": started_at,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def _atomic_write_text(path: Path, value: str) -> None:
    for attempt in range(5):
        temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary_path.write_text(value, encoding="utf-8")
            temporary_path.replace(path)
            return
        except PermissionError:
            if attempt + 1 >= 5:
                raise
            time.sleep(0.05 * (2**attempt))
        finally:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)


def _best_effort_unlink(path: Path) -> None:
    for attempt in range(5):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt + 1 >= 5:
                return
            time.sleep(0.05 * (2**attempt))


def _load_progress_average(path: Path) -> float | None:
    average = _safe_nonnegative_float(_read_json_object(path).get("average_chunk_sec"))
    return average if average is not None and average > 0 else None


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_nonnegative_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return max(0.0, number)


def _format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
