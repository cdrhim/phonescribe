from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import secrets
import shutil
import threading
import time
import unicodedata
import uuid
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Annotated

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from local_meetscribe.config import Settings, ensure_runtime_dirs, get_settings
from local_meetscribe.db import JobStore
from local_meetscribe.pipeline.asr import has_cuda_runtime
from local_meetscribe.pipeline.export import write_exports
from local_meetscribe.pipeline.gemini import (
    GeminiTranscriptionProgress,
    get_gemini_progress,
    transcribe_gemini_package,
)
from local_meetscribe.pipeline.glossary import quick_scan_glossary
from local_meetscribe.pipeline.ingest import probe_media
from local_meetscribe.pipeline.optimizer import (
    OptimizerOverrides,
    OptimizerRequest,
    optimize_audio_package,
    recommend_optimization,
)
from local_meetscribe.pipeline.orchestrator import TranscriptionPipeline
from local_meetscribe.pipeline.prepare_audio import prepare_llm_audio
from local_meetscribe.schemas import (
    ExportKind,
    JobRecord,
    Transcript,
    TranscriptionRequest,
    TranscriptPatch,
    load_transcript,
)
from local_meetscribe.security import GeminiShareStore
from local_meetscribe.utils.errors import LocalMeetScribeError

LOGGER = logging.getLogger(__name__)
STAGED_UPLOAD_TTL_SEC = 24 * 60 * 60


def create_app(settings: Settings | None = None) -> FastAPI:
    active_settings = settings or get_settings()
    ensure_runtime_dirs(active_settings)
    store = JobStore(active_settings)
    share_store = GeminiShareStore(active_settings.data_dir)
    share_failures: dict[str, list[float]] = {}
    share_failure_lock = threading.Lock()
    remote_sessions: dict[str, float] = {}
    remote_session_lock = threading.Lock()
    active_workflow_inputs: set[str] = set()
    active_workflow_lock = threading.Lock()

    def require_share_passcode(request: Request, passcode: str | None) -> None:
        client_id = request.client.host if request.client else "unknown"
        now = time.monotonic()
        with share_failure_lock:
            recent = [
                attempted_at
                for attempted_at in share_failures.get(client_id, [])
                if now - attempted_at < 60.0
            ]
            share_failures[client_id] = recent
            if len(recent) >= 5:
                raise HTTPException(
                    status_code=429,
                    detail="비밀번호 확인 횟수를 초과했습니다. 1분 후 다시 시도하세요.",
                )
        if not passcode or not share_store.verify_passcode(passcode):
            with share_failure_lock:
                share_failures.setdefault(client_id, []).append(now)
            raise HTTPException(status_code=401, detail="공유 비밀번호가 맞지 않습니다.")
        with share_failure_lock:
            share_failures.pop(client_id, None)

    def issue_remote_session() -> str:
        now = time.time()
        token = secrets.token_urlsafe(32)
        with remote_session_lock:
            expired = [value for value, expires_at in remote_sessions.items() if expires_at <= now]
            for value in expired:
                remote_sessions.pop(value, None)
            remote_sessions[token] = now + active_settings.remote_session_ttl_sec
        return token

    def remote_session_is_valid(authorization: str | None) -> bool:
        scheme, separator, token = (authorization or "").partition(" ")
        if not separator or scheme.casefold() != "bearer" or not token:
            return False
        now = time.time()
        with remote_session_lock:
            expires_at = remote_sessions.get(token)
            if expires_at is None:
                return False
            if expires_at <= now:
                remote_sessions.pop(token, None)
                return False
        return True

    def run_transcription_workflow(
        workflow_id: str,
        package_id: str,
        upload_id: str | None,
        optimizer_request: OptimizerRequest,
        api_key: str,
        workflow_input_key: str,
    ) -> None:
        state_path = _workflow_state_path(active_settings, workflow_id)
        output_root = active_settings.data_dir / "optimized"
        output_dir = output_root / package_id
        phase = "optimizing" if upload_id else "transcribing"
        release_system_awake = _request_system_awake()
        try:
            if upload_id:
                _write_workflow_state(
                    state_path,
                    workflow_id=workflow_id,
                    package_id=package_id,
                    status="optimizing",
                )
                staged_upload_dir, source_path = _resolve_staged_upload(
                    active_settings.tmp_dir / "optimizer-uploads",
                    upload_id,
                )
                optimize_audio_package(
                    source_path,
                    output_root,
                    active_settings,
                    optimizer_request,
                    package_id=package_id,
                )
                shutil.rmtree(staged_upload_dir, ignore_errors=True)

            phase = "transcribing"
            _write_workflow_state(
                state_path,
                workflow_id=workflow_id,
                package_id=package_id,
                status="transcribing",
            )
            transcribe_gemini_package(
                output_dir,
                active_settings,
                api_key=api_key,
            )
            _write_workflow_state(
                state_path,
                workflow_id=workflow_id,
                package_id=package_id,
                status="complete",
            )
        except Exception as exc:  # noqa: BLE001 - background work must persist failure state.
            if phase == "transcribing" and _gemini_outputs_complete(output_dir):
                LOGGER.info(
                    "Background workflow %s recovered completed transcript artifacts",
                    workflow_id,
                )
                _write_workflow_state(
                    state_path,
                    workflow_id=workflow_id,
                    package_id=package_id,
                    status="complete",
                )
                return
            if phase == "optimizing":
                shutil.rmtree(output_dir, ignore_errors=True)
            LOGGER.error(
                "Background workflow %s failed during %s (%s)",
                workflow_id,
                phase,
                type(exc).__name__,
            )
            _write_workflow_state(
                state_path,
                workflow_id=workflow_id,
                package_id=package_id,
                status="failed",
                error=_background_error_message(exc),
            )
        finally:
            release_system_awake()
            with active_workflow_lock:
                active_workflow_inputs.discard(workflow_input_key)

    app = FastAPI(title="LocalMeetScribe", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(active_settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def protect_remote_api(request: Request, call_next: Callable):
        public_api_paths = {
            "/api/health",
            "/api/runtime",
            "/api/gemini-share/verify",
        }
        requires_session = (
            active_settings.remote_access_enabled
            and request.method != "OPTIONS"
            and request.url.path.startswith("/api/")
            and request.url.path not in public_api_paths
            and not _is_loopback_request(request)
        )
        if requires_session and not remote_session_is_valid(
            request.headers.get("authorization")
        ):
            return JSONResponse(
                status_code=401,
                content={"detail": "공유 비밀번호를 다시 확인하세요."},
            )
        return await call_next(request)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/runtime")
    def runtime(request: Request) -> dict[str, str | bool]:
        cuda = has_cuda_runtime()
        saved_share_key = share_store.api_key_configured
        return {
            "device": "cuda" if cuda else "cpu",
            "cuda": cuda,
            "fast_model": (
                active_settings.faster_whisper_cuda_model
                if cuda
                else active_settings.faster_whisper_cpu_model
            ),
            "accurate_model": active_settings.faster_whisper_cuda_model,
            "gemini_transcription_enabled": active_settings.enable_gemini_transcription,
            "gemini_api_key_configured": bool(
                active_settings.gemini_api_key or saved_share_key
            ),
            "gemini_model": active_settings.gemini_model,
            "gemini_share_enabled": share_store.passcode_configured,
            "gemini_share_ready": bool(
                share_store.passcode_configured
                and (active_settings.gemini_api_key or saved_share_key)
            ),
            "local_admin": _is_loopback_request(request),
        }

    @app.post("/api/gemini-share/verify")
    def verify_gemini_share(
        request: Request,
        share_passcode: Annotated[str | None, Header(alias="X-LocalMeetScribe-Passcode")] = None,
    ) -> dict[str, object]:
        if not share_store.passcode_configured:
            raise HTTPException(status_code=404, detail="Shared Gemini access is not configured.")
        require_share_passcode(request, share_passcode)
        return {
            "valid": True,
            "key_ready": bool(active_settings.gemini_api_key or share_store.api_key_configured),
            "access_token": issue_remote_session(),
            "expires_in": active_settings.remote_session_ttl_sec,
        }

    @app.post("/api/admin/gemini-share-key")
    def configure_gemini_share_key(
        request: Request,
        api_key: Annotated[str, Form()],
        share_passcode: Annotated[str | None, Header(alias="X-LocalMeetScribe-Passcode")] = None,
    ) -> dict[str, bool]:
        if not _is_loopback_request(request):
            raise HTTPException(
                status_code=403,
                detail="기본 API key 등록은 서버 PC에서만 가능합니다.",
            )
        require_share_passcode(request, share_passcode)
        try:
            share_store.save_api_key(api_key)
        except LocalMeetScribeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"configured": True}

    @app.post("/api/optimizer/analyze")
    def optimizer_analyze(
        file: Annotated[UploadFile, File()],
        destination: Annotated[str, Form()] = "gemini",
        openai_model: Annotated[str, Form()] = "gpt-4o-transcribe",
        word_timestamps: Annotated[bool, Form()] = False,
        codec: Annotated[str | None, Form()] = None,
        bitrate_kbps: Annotated[int | None, Form()] = None,
        chunk_minutes: Annotated[float | None, Form()] = None,
        remove_silence: Annotated[bool, Form()] = True,
        loudnorm: Annotated[bool, Form()] = True,
        speech_filter: Annotated[bool, Form()] = True,
        denoise: Annotated[bool, Form()] = False,
        language: Annotated[str, Form()] = "auto",
        run_quick_scan: Annotated[bool, Form(alias="quick_scan")] = True,
    ) -> dict[str, object]:
        if language not in {"auto", "ko", "en"}:
            raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")

        upload_id = uuid.uuid4().hex
        staged_root = active_settings.tmp_dir / "optimizer-uploads"
        _prune_staged_uploads(staged_root)
        upload_dir = staged_root / upload_id
        source_dir = upload_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        source_path = source_dir / _safe_filename(file.filename or "upload")
        scan_dir = upload_dir / "quick-scan"
        try:
            with source_path.open("wb") as output_file:
                shutil.copyfileobj(file.file, output_file)
            media_info = probe_media(source_path, active_settings)
            optimizer_request = _optimizer_request(
                destination=destination,
                openai_model=openai_model,
                word_timestamps=word_timestamps,
                codec=codec,
                bitrate_kbps=bitrate_kbps,
                chunk_minutes=chunk_minutes,
                remove_silence=remove_silence,
                loudnorm=loudnorm,
                speech_filter=speech_filter,
                denoise=denoise,
            )
            recommendation = recommend_optimization(
                media_info,
                source_path.stat().st_size,
                optimizer_request,
            )
            if not run_quick_scan:
                quick_scan = {
                    "glossary": [],
                    "preview_text": "",
                    "detected_language": "unknown",
                    "scan_seconds": 0,
                    "warning": None,
                }
            else:
                try:
                    scan = quick_scan_glossary(
                        source_path,
                        scan_dir,
                        active_settings,
                        language=language,  # type: ignore[arg-type]
                    )
                    quick_scan = {
                        "glossary": scan.terms,
                        "preview_text": scan.preview_text,
                        "detected_language": scan.detected_language,
                        "scan_seconds": scan.scan_seconds,
                        "warning": scan.warning,
                    }
                except Exception as exc:  # noqa: BLE001 - glossary scan is optional.
                    LOGGER.info(
                        "Quick scan unavailable for staged upload %s (%s)",
                        upload_id,
                        type(exc).__name__,
                    )
                    quick_scan = {
                        "glossary": [],
                        "preview_text": "",
                        "detected_language": "unknown",
                        "scan_seconds": 0,
                        "warning": "빠른 언어 스캔을 건너뛰었습니다.",
                    }
                finally:
                    shutil.rmtree(scan_dir, ignore_errors=True)

            return {
                "upload_id": upload_id,
                "source": media_info.__dict__,
                "original_bytes": source_path.stat().st_size,
                "recommendation": recommendation.__dict__,
                "quick_scan": quick_scan,
            }
        except LocalMeetScribeError as exc:
            shutil.rmtree(upload_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - API must convert failures to user errors.
            shutil.rmtree(upload_dir, ignore_errors=True)
            LOGGER.exception("Unexpected optimizer analysis failure for upload %s", upload_id)
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected optimizer analysis failure: {type(exc).__name__}: {exc}",
            ) from exc

    @app.post("/api/optimizer/recommend")
    def optimizer_recommend(
        file: Annotated[UploadFile, File()],
        destination: Annotated[str, Form()] = "gemini",
        openai_model: Annotated[str, Form()] = "gpt-4o-transcribe",
        word_timestamps: Annotated[bool, Form()] = False,
        codec: Annotated[str | None, Form()] = None,
        bitrate_kbps: Annotated[int | None, Form()] = None,
        chunk_minutes: Annotated[float | None, Form()] = None,
        remove_silence: Annotated[bool, Form()] = True,
        loudnorm: Annotated[bool, Form()] = True,
        speech_filter: Annotated[bool, Form()] = True,
        denoise: Annotated[bool, Form()] = False,
    ) -> dict[str, object]:
        scan_id = uuid.uuid4().hex
        scan_dir = active_settings.tmp_dir / f"optimizer-recommend-{scan_id}"
        scan_dir.mkdir(parents=True, exist_ok=True)
        source_path = scan_dir / _safe_filename(file.filename or "upload")
        try:
            with source_path.open("wb") as output_file:
                shutil.copyfileobj(file.file, output_file)
            media_info = probe_media(source_path, active_settings)
            request = _optimizer_request(
                destination=destination,
                openai_model=openai_model,
                word_timestamps=word_timestamps,
                codec=codec,
                bitrate_kbps=bitrate_kbps,
                chunk_minutes=chunk_minutes,
                remove_silence=remove_silence,
                loudnorm=loudnorm,
                speech_filter=speech_filter,
                denoise=denoise,
            )
            recommendation = recommend_optimization(media_info, source_path.stat().st_size, request)
            return {
                "source": media_info.__dict__,
                "original_bytes": source_path.stat().st_size,
                "recommendation": recommendation.__dict__,
            }
        except LocalMeetScribeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            shutil.rmtree(scan_dir, ignore_errors=True)

    @app.post("/api/optimizer/package")
    def optimizer_package(
        file: Annotated[UploadFile | None, File()] = None,
        destination: Annotated[str, Form()] = "gemini",
        openai_model: Annotated[str, Form()] = "gpt-4o-transcribe",
        word_timestamps: Annotated[bool, Form()] = False,
        codec: Annotated[str | None, Form()] = None,
        bitrate_kbps: Annotated[int | None, Form()] = None,
        chunk_minutes: Annotated[float | None, Form()] = None,
        remove_silence: Annotated[bool, Form()] = True,
        loudnorm: Annotated[bool, Form()] = True,
        speech_filter: Annotated[bool, Form()] = True,
        denoise: Annotated[bool, Form()] = False,
        upload_id: Annotated[str | None, Form()] = None,
    ) -> dict[str, object]:
        package_id = uuid.uuid4().hex
        output_root = active_settings.data_dir / "optimized"
        output_dir = output_root / package_id
        staged_upload_dir: Path | None = None
        source_dir: Path | None = None
        try:
            if upload_id:
                staged_upload_dir, source_path = _resolve_staged_upload(
                    active_settings.tmp_dir / "optimizer-uploads",
                    upload_id,
                )
            elif file is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                source_dir = output_dir / "source"
                source_dir.mkdir(parents=True, exist_ok=True)
                source_path = source_dir / _safe_filename(file.filename or "upload")
                with source_path.open("wb") as output_file:
                    shutil.copyfileobj(file.file, output_file)
            else:
                raise LocalMeetScribeError(
                    "The optimized package requires an uploaded file or staged upload ID."
                )
            request = _optimizer_request(
                destination=destination,
                openai_model=openai_model,
                word_timestamps=word_timestamps,
                codec=codec,
                bitrate_kbps=bitrate_kbps,
                chunk_minutes=chunk_minutes,
                remove_silence=remove_silence,
                loudnorm=loudnorm,
                speech_filter=speech_filter,
                denoise=denoise,
            )
            package = optimize_audio_package(
                source_path,
                output_root,
                active_settings,
                request,
                package_id=package_id,
            )
            if staged_upload_dir is not None:
                shutil.rmtree(staged_upload_dir, ignore_errors=True)
            elif source_dir is not None:
                shutil.rmtree(source_dir, ignore_errors=True)
            return {
                "id": package.id,
                "source": package.source.__dict__,
                "recommendation": package.recommendation.__dict__,
                "chunks": [chunk.__dict__ for chunk in package.chunks],
                "manifest_url": package.manifest_url,
                "package_url": package.package_url,
            }
        except LocalMeetScribeError as exc:
            shutil.rmtree(output_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - API must convert failures to user errors.
            shutil.rmtree(output_dir, ignore_errors=True)
            LOGGER.exception("Unexpected optimizer package failure")
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected optimizer package failure: {type(exc).__name__}: {exc}",
            ) from exc

    @app.post("/api/workflows", status_code=202)
    def start_transcription_workflow(
        background_tasks: BackgroundTasks,
        request: Request,
        upload_id: Annotated[str | None, Form()] = None,
        package_id: Annotated[str | None, Form()] = None,
        destination: Annotated[str, Form()] = "gemini",
        openai_model: Annotated[str, Form()] = "gpt-4o-transcribe",
        word_timestamps: Annotated[bool, Form()] = False,
        codec: Annotated[str | None, Form()] = None,
        bitrate_kbps: Annotated[int | None, Form()] = None,
        chunk_minutes: Annotated[float | None, Form()] = None,
        remove_silence: Annotated[bool, Form()] = True,
        loudnorm: Annotated[bool, Form()] = True,
        speech_filter: Annotated[bool, Form()] = True,
        denoise: Annotated[bool, Form()] = False,
        gemini_api_key: Annotated[str | None, Header(alias="X-Gemini-API-Key")] = None,
        share_passcode: Annotated[
            str | None,
            Header(alias="X-LocalMeetScribe-Passcode"),
        ] = None,
    ) -> dict[str, object]:
        if bool(upload_id) == bool(package_id):
            raise HTTPException(
                status_code=400,
                detail="Provide one staged upload ID or one optimized package ID.",
            )

        optimizer_request = _optimizer_request(
            destination=destination,
            openai_model=openai_model,
            word_timestamps=word_timestamps,
            codec=codec,
            bitrate_kbps=bitrate_kbps,
            chunk_minutes=chunk_minutes,
            remove_silence=remove_silence,
            loudnorm=loudnorm,
            speech_filter=speech_filter,
            denoise=denoise,
        )
        if optimizer_request.destination != "gemini":
            raise HTTPException(
                status_code=400,
                detail="Background transcription workflows require the Gemini destination.",
            )

        if upload_id:
            try:
                _resolve_staged_upload(
                    active_settings.tmp_dir / "optimizer-uploads",
                    upload_id,
                )
            except LocalMeetScribeError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            resolved_package_id = uuid.uuid4().hex
        else:
            resolved_package_id = package_id or ""
            if not re.fullmatch(r"[a-f0-9]{32}", resolved_package_id):
                raise HTTPException(status_code=404, detail="Optimized package not found.")
            if not (
                active_settings.data_dir
                / "optimized"
                / resolved_package_id
                / "manifest.json"
            ).exists():
                raise HTTPException(status_code=404, detail="Optimized package not found.")

        resolved_api_key = (gemini_api_key or "").strip()
        if share_store.passcode_configured:
            require_share_passcode(request, share_passcode)
            resolved_api_key = (
                active_settings.gemini_api_key or share_store.load_api_key() or ""
            ).strip()
        elif not resolved_api_key:
            resolved_api_key = (active_settings.gemini_api_key or "").strip()
        if not resolved_api_key:
            raise HTTPException(
                status_code=503,
                detail="Gemini API key가 필요합니다.",
            )

        workflow_id = uuid.uuid4().hex
        workflow_input_key = (
            f"upload:{upload_id}" if upload_id else f"package:{resolved_package_id}"
        )
        with active_workflow_lock:
            if workflow_input_key in active_workflow_inputs:
                raise HTTPException(
                    status_code=409,
                    detail="This recording already has an active transcription workflow.",
                )
            active_workflow_inputs.add(workflow_input_key)
        state_path = _workflow_state_path(active_settings, workflow_id)
        try:
            _write_workflow_state(
                state_path,
                workflow_id=workflow_id,
                package_id=resolved_package_id,
                status="queued",
            )
            background_tasks.add_task(
                run_transcription_workflow,
                workflow_id,
                resolved_package_id,
                upload_id,
                optimizer_request,
                resolved_api_key,
                workflow_input_key,
            )
        except Exception:
            with active_workflow_lock:
                active_workflow_inputs.discard(workflow_input_key)
            raise
        return {
            "workflow_id": workflow_id,
            "package_id": resolved_package_id,
            "status": "queued",
        }

    @app.get("/api/workflows/{workflow_id}")
    def transcription_workflow_status(workflow_id: str) -> dict[str, object]:
        if not re.fullmatch(r"[a-f0-9]{32}", workflow_id):
            raise HTTPException(status_code=404, detail="Workflow not found.")
        state_path = _workflow_state_path(active_settings, workflow_id)
        if not state_path.exists():
            raise HTTPException(status_code=404, detail="Workflow not found.")

        state = _read_json_object(state_path)
        package_id = str(state.get("package_id") or "")
        status = str(state.get("status") or "failed")
        package_dir = active_settings.data_dir / "optimized" / package_id
        if status == "failed" and _gemini_outputs_complete(package_dir):
            status = "complete"
            try:
                _write_workflow_state(
                    state_path,
                    workflow_id=workflow_id,
                    package_id=package_id,
                    status="complete",
                )
            except OSError:
                LOGGER.info("Could not persist recovered workflow %s", workflow_id)
        response: dict[str, object] = {
            "workflow_id": workflow_id,
            "package_id": package_id,
            "status": status,
            "error": None if status == "complete" else state.get("error"),
        }
        if (package_dir / "manifest.json").exists():
            response["package"] = _optimized_package_payload(package_dir, package_id)

        if status == "transcribing":
            progress = get_gemini_progress(package_dir)
            response["transcription_progress"] = _gemini_progress_payload(progress)
        elif status == "complete":
            response["transcription_progress"] = _gemini_progress_payload(
                get_gemini_progress(package_dir)
            )
            response["transcript"] = _stored_gemini_transcript_payload(
                package_dir,
                package_id,
            )
        return response

    @app.get("/api/optimizer/packages/{package_id}/{filename}")
    def download_optimizer_package_file(
        package_id: str,
        filename: str,
        download_name: str | None = None,
    ) -> FileResponse:
        if not re.fullmatch(r"[a-f0-9]{32}", package_id):
            raise HTTPException(status_code=404, detail="Optimized package file not found.")
        allowed = re.fullmatch(
            r"(chunk_\d{3}\.(mp3|m4a|ogg)|manifest\.json|optimized_package\.zip|"
            r"gemini_transcript\.(json|txt))",
            filename,
        )
        if not allowed:
            raise HTTPException(status_code=404, detail="Optimized package file not found.")
        path = active_settings.data_dir / "optimized" / package_id / filename
        if not path.exists():
            raise HTTPException(status_code=404, detail="Optimized package file not found.")
        return FileResponse(
            path,
            filename=_safe_download_filename(download_name, filename),
        )

    @app.post("/api/optimizer/packages/{package_id}/gemini-transcribe")
    def gemini_transcribe_package(
        package_id: str,
        request: Request,
        gemini_api_key: Annotated[str | None, Header(alias="X-Gemini-API-Key")] = None,
        share_passcode: Annotated[
            str | None,
            Header(alias="X-LocalMeetScribe-Passcode"),
        ] = None,
    ) -> dict[str, object]:
        if not re.fullmatch(r"[a-f0-9]{32}", package_id):
            raise HTTPException(status_code=404, detail="Optimized package not found.")
        package_dir = active_settings.data_dir / "optimized" / package_id
        if not (package_dir / "manifest.json").exists():
            raise HTTPException(status_code=404, detail="Optimized package not found.")
        try:
            resolved_api_key = gemini_api_key
            if share_store.passcode_configured:
                require_share_passcode(request, share_passcode)
                resolved_api_key = active_settings.gemini_api_key or share_store.load_api_key()
                if not resolved_api_key:
                    raise HTTPException(
                        status_code=503,
                        detail="서버 PC에서 기본 Gemini API key를 먼저 등록하세요.",
                    )
            result = transcribe_gemini_package(
                package_dir,
                active_settings,
                api_key=resolved_api_key,
            )
            return {
                "provider": result.provider,
                "model": result.model,
                "text": result.text,
                "suggested_filename": result.suggested_filename,
                "chunk_count": len(result.chunks),
                "chunks": [
                    {
                        "filename": chunk.filename,
                        "start_sec": chunk.start_sec,
                        "end_sec": chunk.end_sec,
                        "delivery": chunk.delivery,
                        "mime_type": chunk.mime_type,
                    }
                    for chunk in result.chunks
                ],
                "txt_url": f"/api/optimizer/packages/{package_id}/gemini_transcript.txt",
                "json_url": f"/api/optimizer/packages/{package_id}/gemini_transcript.json",
            }
        except HTTPException:
            raise
        except LocalMeetScribeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - API must convert failures to user errors.
            LOGGER.exception("Unexpected Gemini transcription failure for package %s", package_id)
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected Gemini transcription failure: {type(exc).__name__}: {exc}",
            ) from exc

    @app.get("/api/optimizer/packages/{package_id}/gemini-transcribe/progress")
    def gemini_transcription_progress(package_id: str) -> dict[str, object]:
        if not re.fullmatch(r"[a-f0-9]{32}", package_id):
            raise HTTPException(status_code=404, detail="Optimized package not found.")
        package_dir = active_settings.data_dir / "optimized" / package_id
        if not (package_dir / "manifest.json").exists():
            raise HTTPException(status_code=404, detail="Optimized package not found.")
        try:
            progress = get_gemini_progress(package_dir)
            return {
                "status": progress.status,
                "completed_chunks": progress.completed_chunks,
                "total_chunks": progress.total_chunks,
                "current_chunk": progress.current_chunk,
                "progress": progress.progress,
                "elapsed_sec": progress.elapsed_sec,
                "eta_sec": progress.eta_sec,
            }
        except LocalMeetScribeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/files/quick-scan")
    def quick_scan_file(
        file: Annotated[UploadFile, File()],
        language: Annotated[str, Form()] = "auto",
    ) -> dict[str, object]:
        scan_id = uuid.uuid4().hex
        scan_dir = active_settings.tmp_dir / f"quick-scan-{scan_id}"
        scan_dir.mkdir(parents=True, exist_ok=True)
        source_path = scan_dir / _safe_filename(file.filename or "upload")
        try:
            with source_path.open("wb") as output_file:
                shutil.copyfileobj(file.file, output_file)
            result = quick_scan_glossary(
                source_path,
                scan_dir,
                active_settings,
                language=language,  # type: ignore[arg-type]
            )
            return {
                "glossary": result.terms,
                "preview_text": result.preview_text,
                "detected_language": result.detected_language,
                "scan_seconds": result.scan_seconds,
                "warning": result.warning,
            }
        except LocalMeetScribeError as exc:
            return {
                "glossary": [],
                "preview_text": "",
                "detected_language": "unknown",
                "scan_seconds": 0,
                "warning": str(exc),
            }
        except Exception as exc:  # noqa: BLE001 - quick scan should not block upload.
            LOGGER.exception("Unexpected quick glossary scan failure")
            return {
                "glossary": [],
                "preview_text": "",
                "detected_language": "unknown",
                "scan_seconds": 0,
                "warning": f"Unexpected quick scan failure: {type(exc).__name__}: {exc}",
            }
        finally:
            shutil.rmtree(scan_dir, ignore_errors=True)

    @app.post("/api/files/prepare-audio")
    def prepare_audio_file(
        file: Annotated[UploadFile, File()],
        remove_silence: Annotated[bool, Form()] = True,
        max_minutes: Annotated[float | None, Form()] = None,
        bitrate_kbps: Annotated[int, Form()] = 32,
    ) -> dict[str, object]:
        prepare_id = uuid.uuid4().hex
        prepare_dir = active_settings.data_dir / "prepared" / prepare_id
        prepare_dir.mkdir(parents=True, exist_ok=True)
        source_path = prepare_dir / f"source_{_safe_filename(file.filename or 'upload')}"
        try:
            with source_path.open("wb") as output_file:
                shutil.copyfileobj(file.file, output_file)
            prepared = prepare_llm_audio(
                source_path,
                prepare_dir,
                active_settings,
                remove_silence=remove_silence,
                max_minutes=max_minutes,
                bitrate_kbps=bitrate_kbps,
            )
            source_path.unlink(missing_ok=True)
            return {
                "id": prepare_id,
                "filename": prepared.path.name,
                "download_url": f"/api/files/prepared/{prepare_id}",
                "original_duration_sec": prepared.original_info.duration_sec,
                "prepared_duration_sec": prepared.prepared_info.duration_sec,
                "original_bytes": prepared.original_bytes,
                "prepared_bytes": prepared.prepared_bytes,
                "compression_ratio": prepared.compression_ratio,
                "remove_silence": prepared.remove_silence,
                "max_minutes": prepared.max_minutes,
                "bitrate_kbps": prepared.bitrate_kbps,
            }
        except LocalMeetScribeError as exc:
            shutil.rmtree(prepare_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - API must convert failures to user errors.
            shutil.rmtree(prepare_dir, ignore_errors=True)
            LOGGER.exception("Unexpected LLM audio preparation failure")
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected LLM audio preparation failure: {type(exc).__name__}: {exc}",
            ) from exc

    @app.get("/api/files/prepared/{prepare_id}")
    def download_prepared_audio(prepare_id: str) -> FileResponse:
        if not re.fullmatch(r"[a-f0-9]{32}", prepare_id):
            raise HTTPException(status_code=404, detail="Prepared audio not found.")
        path = active_settings.data_dir / "prepared" / prepare_id / "llm_audio.m4a"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Prepared audio not found.")
        return FileResponse(path, filename="llm_audio.m4a", media_type="audio/mp4")

    @app.get("/api/jobs", response_model=list[JobRecord])
    def list_jobs() -> list[JobRecord]:
        return store.list_jobs()

    @app.post("/api/jobs", response_model=JobRecord)
    def create_job(
        background_tasks: BackgroundTasks,
        file: Annotated[UploadFile, File()],
        mode: Annotated[str, Form()] = "accurate",
        language: Annotated[str, Form()] = "auto",
        speakers: Annotated[int | None, Form()] = None,
        min_speakers: Annotated[int | None, Form()] = None,
        max_speakers: Annotated[int | None, Form()] = None,
        glossary: Annotated[str, Form()] = "",
        denoise: Annotated[bool, Form()] = False,
        loudness_normalize: Annotated[bool, Form()] = False,
        trim_silence: Annotated[bool, Form()] = False,
    ) -> JobRecord:
        job_id = uuid.uuid4().hex
        upload_dir = active_settings.uploads_dir / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        filename = _safe_filename(file.filename or "upload")
        source_path = upload_dir / filename
        with source_path.open("wb") as output_file:
            shutil.copyfileobj(file.file, output_file)
        output_dir = active_settings.job_dir(job_id) / "exports"
        request = TranscriptionRequest(
            mode=mode,  # type: ignore[arg-type]
            language=language,  # type: ignore[arg-type]
            speakers=speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            glossary=_parse_glossary(glossary),
            denoise=denoise,
            loudness_normalize=loudness_normalize,
            trim_silence=trim_silence,
        )
        job = store.create_job(job_id, source_path=source_path, output_dir=output_dir)
        background_tasks.add_task(
            _run_job, active_settings, job_id, source_path, output_dir, request
        )
        return job

    @app.get("/api/jobs/{job_id}", response_model=JobRecord)
    def get_job(job_id: str) -> JobRecord:
        return _get_job_or_404(store, job_id)

    @app.get("/api/jobs/{job_id}/transcript", response_model=Transcript)
    def get_transcript(job_id: str) -> Transcript:
        job = _get_job_or_404(store, job_id)
        if not job.transcript_path:
            raise HTTPException(status_code=404, detail="Transcript is not ready.")
        return load_transcript(Path(job.transcript_path))

    @app.patch("/api/jobs/{job_id}/transcript", response_model=Transcript)
    def patch_transcript(job_id: str, patch: TranscriptPatch) -> Transcript:
        job = _get_job_or_404(store, job_id)
        if not job.transcript_path or not job.output_dir:
            raise HTTPException(status_code=404, detail="Transcript is not ready.")
        transcript = load_transcript(Path(job.transcript_path))
        text_updates = {item.id: item.text_clean for item in patch.segments}
        speaker_updates = {item.id: item.display_name for item in patch.speakers}
        for segment in transcript.segments:
            if segment.id in text_updates:
                segment.text_clean = text_updates[segment.id]
        for speaker in transcript.speakers:
            if speaker.id in speaker_updates:
                speaker.display_name = speaker_updates[speaker.id]
        transcript = write_exports(transcript, Path(job.output_dir))
        transcript_path = Path(transcript.exports.json_path or job.transcript_path)
        store.update_job(job_id, transcript_path=transcript_path)
        return transcript

    @app.get("/api/jobs/{job_id}/exports/{kind}")
    def get_export(job_id: str, kind: ExportKind) -> FileResponse:
        job = _get_job_or_404(store, job_id)
        if not job.transcript_path:
            raise HTTPException(status_code=404, detail="Transcript is not ready.")
        transcript = load_transcript(Path(job.transcript_path))
        export_path = _export_path(transcript, kind)
        if not export_path or not Path(export_path).exists():
            raise HTTPException(status_code=404, detail=f"Export not found: {kind}")
        return FileResponse(export_path, filename=Path(export_path).name)

    @app.get("/api/jobs/{job_id}/audio")
    def get_audio(job_id: str) -> FileResponse:
        job = _get_job_or_404(store, job_id)
        if not job.source_path or not Path(job.source_path).exists():
            raise HTTPException(status_code=404, detail="Audio is not available.")
        return FileResponse(job.source_path, filename=Path(job.source_path).name)

    _mount_frontend(app)
    return app


def _run_job(
    settings: Settings,
    job_id: str,
    source_path: Path,
    output_dir: Path,
    request: TranscriptionRequest,
) -> None:
    store = JobStore(settings)

    def progress(stage: str, progress_value: float) -> None:
        store.update_job(job_id, status="running", stage=stage, progress=progress_value)

    try:
        store.update_job(job_id, status="running", stage="starting", progress=0.01)
        pipeline = TranscriptionPipeline(settings, progress_callback=progress)
        transcript = pipeline.run(
            source_path,
            output_dir=output_dir,
            request=request,
            job_id=job_id,
        )
        transcript_path = Path(transcript.exports.json_path or output_dir / "transcript.json")
        store.update_job(
            job_id,
            status="completed",
            stage="completed",
            progress=1.0,
            transcript_path=transcript_path,
        )
    except LocalMeetScribeError as exc:
        store.update_job(job_id, status="failed", stage="failed", progress=1.0, error=str(exc))
    except Exception as exc:  # noqa: BLE001 - API must convert background failures to job errors.
        LOGGER.exception("Unexpected pipeline failure for job %s", job_id)
        store.update_job(
            job_id,
            status="failed",
            stage="failed",
            progress=1.0,
            error=f"Unexpected pipeline failure: {type(exc).__name__}: {exc}",
        )


def _get_job_or_404(store: JobStore, job_id: str) -> JobRecord:
    try:
        return store.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc


def _parse_glossary(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def _optimizer_request(
    *,
    destination: str,
    openai_model: str,
    word_timestamps: bool,
    codec: str | None,
    bitrate_kbps: int | None,
    chunk_minutes: float | None,
    remove_silence: bool,
    loudnorm: bool,
    speech_filter: bool,
    denoise: bool,
) -> OptimizerRequest:
    if destination == "anthropic":
        raise LocalMeetScribeError("Claude cannot transcribe raw audio. Use it only after STT.")
    if destination not in {"gemini", "openai", "optimize"}:
        raise LocalMeetScribeError(f"Unsupported optimizer destination: {destination}")
    if openai_model not in {"gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"}:
        raise LocalMeetScribeError(f"Unsupported OpenAI transcription model: {openai_model}")
    if codec == "":
        codec = None
    if bitrate_kbps == 0:
        bitrate_kbps = None
    if chunk_minutes == 0:
        chunk_minutes = None
    return OptimizerRequest(
        destination=destination,  # type: ignore[arg-type]
        openai_model=openai_model,  # type: ignore[arg-type]
        word_timestamps=word_timestamps,
        overrides=OptimizerOverrides(
            codec=codec,  # type: ignore[arg-type]
            bitrate_kbps=bitrate_kbps,
            chunk_minutes=chunk_minutes,
            remove_silence=remove_silence,
            loudnorm=loudnorm,
            speech_filter=speech_filter,
            denoise=denoise,
        ),
    )


def _safe_filename(value: str) -> str:
    name = unicodedata.normalize("NFC", Path(value).name)
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name)
    return name.strip(" .") or "upload"


def _workflow_state_path(settings: Settings, workflow_id: str) -> Path:
    return settings.tmp_dir / "workflows" / f"{workflow_id}.json"


def _write_workflow_state(
    path: Path,
    *,
    workflow_id: str,
    package_id: str,
    status: str,
    error: str | None = None,
) -> None:
    payload = {
        "workflow_id": workflow_id,
        "package_id": package_id,
        "status": status,
        "error": error,
        "updated_at": time.time(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    value = json.dumps(payload, ensure_ascii=False, indent=2)
    for attempt in range(5):
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(value, encoding="utf-8")
            temporary.replace(path)
            return
        except PermissionError:
            if attempt + 1 >= 5:
                raise
            time.sleep(0.05 * (2**attempt))
        finally:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LocalMeetScribeError(f"Could not read workflow state: {path.name}") from exc
    if not isinstance(payload, dict):
        raise LocalMeetScribeError(f"Workflow state is invalid: {path.name}")
    return payload


def _background_error_message(exc: Exception) -> str:
    if isinstance(exc, LocalMeetScribeError):
        return str(exc)[:500]
    return f"Unexpected background failure: {type(exc).__name__}"


def _gemini_outputs_complete(package_dir: Path) -> bool:
    try:
        return all(
            path.is_file() and path.stat().st_size > 0
            for path in (
                package_dir / "gemini_transcript.txt",
                package_dir / "gemini_transcript.json",
            )
        )
    except OSError:
        return False


def _request_system_awake() -> Callable[[], None]:
    if os.name != "nt":
        return lambda: None
    try:
        import ctypes

        execution_state_continuous = 0x80000000
        execution_state_system_required = 0x00000001
        requested = ctypes.windll.kernel32.SetThreadExecutionState(
            execution_state_continuous | execution_state_system_required
        )
        if requested == 0:
            return lambda: None

        def release() -> None:
            ctypes.windll.kernel32.SetThreadExecutionState(
                execution_state_continuous
            )

        return release
    except (AttributeError, OSError):
        return lambda: None


def _optimized_package_payload(package_dir: Path, package_id: str) -> dict[str, object]:
    manifest = _read_json_object(package_dir / "manifest.json")
    return {
        "id": package_id,
        "source": manifest.get("source"),
        "recommendation": manifest.get("recommendation"),
        "chunks": manifest.get("chunks"),
        "manifest_url": f"/api/optimizer/packages/{package_id}/manifest.json",
        "package_url": f"/api/optimizer/packages/{package_id}/optimized_package.zip",
    }


def _gemini_progress_payload(
    progress: GeminiTranscriptionProgress,
) -> dict[str, object]:
    return {
        "status": progress.status,
        "completed_chunks": progress.completed_chunks,
        "total_chunks": progress.total_chunks,
        "current_chunk": progress.current_chunk,
        "progress": progress.progress,
        "elapsed_sec": progress.elapsed_sec,
        "eta_sec": progress.eta_sec,
    }


def _stored_gemini_transcript_payload(
    package_dir: Path,
    package_id: str,
) -> dict[str, object]:
    payload = _read_json_object(package_dir / "gemini_transcript.json")
    raw_chunks = payload.get("chunks")
    chunks = raw_chunks if isinstance(raw_chunks, list) else []
    public_chunks = [
        {
            "filename": item.get("filename"),
            "start_sec": item.get("start_sec"),
            "end_sec": item.get("end_sec"),
            "delivery": item.get("delivery"),
            "mime_type": item.get("mime_type"),
        }
        for item in chunks
        if isinstance(item, dict)
    ]
    return {
        "provider": "gemini",
        "model": payload.get("model"),
        "text": payload.get("text"),
        "suggested_filename": payload.get("suggested_filename"),
        "chunk_count": len(public_chunks),
        "chunks": public_chunks,
        "txt_url": f"/api/optimizer/packages/{package_id}/gemini_transcript.txt",
        "json_url": f"/api/optimizer/packages/{package_id}/gemini_transcript.json",
    }


def _resolve_staged_upload(staged_root: Path, upload_id: str) -> tuple[Path, Path]:
    if not re.fullmatch(r"[a-f0-9]{32}", upload_id):
        raise LocalMeetScribeError("Staged upload ID is invalid.")
    upload_dir = staged_root / upload_id
    source_dir = upload_dir / "source"
    if not source_dir.is_dir():
        raise LocalMeetScribeError(
            "The staged upload is no longer available. Select the recording again."
        )
    source_files = [path for path in source_dir.iterdir() if path.is_file()]
    if len(source_files) != 1:
        raise LocalMeetScribeError(
            "The staged upload is incomplete. Select the recording again."
        )
    return upload_dir, source_files[0]


def _prune_staged_uploads(staged_root: Path) -> None:
    staged_root.mkdir(parents=True, exist_ok=True)
    root = staged_root.resolve()
    cutoff = time.time() - STAGED_UPLOAD_TTL_SEC
    for candidate in staged_root.iterdir():
        if not candidate.is_dir() or not re.fullmatch(r"[a-f0-9]{32}", candidate.name):
            continue
        try:
            resolved = candidate.resolve()
            if resolved.parent != root or candidate.stat().st_mtime >= cutoff:
                continue
            shutil.rmtree(resolved, ignore_errors=True)
        except OSError:
            LOGGER.info("Could not prune staged upload %s", candidate.name)


def _is_loopback_request(request: Request) -> bool:
    client_host = request.client.host if request.client else ""
    request_host = request.url.hostname or ""
    return _is_loopback_host(client_host) and _is_loopback_host(request_host)


def _is_loopback_host(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host.casefold() == "localhost"


def _safe_download_filename(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    candidate = _safe_filename(value)[:140].rstrip(" .")
    expected_suffix = Path(fallback).suffix.lower()
    if Path(candidate).suffix.lower() != expected_suffix:
        candidate = f"{Path(candidate).stem}{expected_suffix}"
    return candidate or fallback


def _export_path(transcript: Transcript, kind: ExportKind) -> str | None:
    if kind == "json":
        return transcript.exports.json_path
    return getattr(transcript.exports, kind)


def _mount_frontend(app: FastAPI) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    dist = repo_root / "frontend" / "dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")


app = create_app()
