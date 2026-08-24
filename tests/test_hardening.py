from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import local_meetscribe.api.app as app_module
import local_meetscribe.pipeline.gemini as gemini_module
import pytest
from fastapi.testclient import TestClient
from local_meetscribe.api.app import _safe_download_filename, _safe_filename
from local_meetscribe.pipeline.asr import (
    ASRResult,
    ASRSegment,
    ASRWord,
    faster_whisper_runtime_plan,
)
from local_meetscribe.pipeline.diarize import SpeakerTurn
from local_meetscribe.pipeline.format import RuleBasedFormatterEngine
from local_meetscribe.pipeline.gemini import (
    GeminiChunkTranscript,
    GeminiTranscriptResult,
    can_send_gemini_inline,
    gemini_mime_type_for_path,
    get_gemini_progress,
    suggest_transcript_filename,
    transcribe_gemini_package,
)
from local_meetscribe.pipeline.glossary import GlossaryScanResult, extract_glossary_terms
from local_meetscribe.pipeline.ingest import MediaInfo
from local_meetscribe.pipeline.merge import merge_transcript
from local_meetscribe.pipeline.optimizer import (
    OptimizedChunk,
    OptimizedPackage,
    OptimizerRequest,
    recommend_optimization,
)
from local_meetscribe.pipeline.prepare_audio import build_prepare_audio_command
from local_meetscribe.schemas import (
    SourceInfo,
    Speaker,
    Transcript,
    TranscriptConfig,
    TranscriptSegment,
)
from local_meetscribe.utils.errors import LocalMeetScribeError

from tests.helpers import make_test_settings


def test_optimizer_reuses_staged_upload_without_second_file_transfer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_test_settings(tmp_path)
    media_info = MediaInfo(
        filename="phone.m4a",
        duration_sec=65.0,
        sample_rate=48000,
        channels=1,
    )
    captured_source: list[Path] = []

    monkeypatch.setattr(app_module, "probe_media", lambda *_args: media_info)
    monkeypatch.setattr(
        app_module,
        "quick_scan_glossary",
        lambda *_args, **_kwargs: GlossaryScanResult(
            terms=["OASIS"],
            preview_text="",
            detected_language="ko",
            scan_seconds=30,
        ),
    )

    def fake_optimize(
        input_path: Path,
        output_root: Path,
        _settings: object,
        request: OptimizerRequest,
        *,
        package_id: str,
    ) -> OptimizedPackage:
        captured_source.append(input_path)
        recommendation = recommend_optimization(media_info, input_path.stat().st_size, request)
        return OptimizedPackage(
            id=package_id,
            recommendation=recommendation,
            source=media_info,
            chunks=[
                OptimizedChunk(
                    filename="chunk_001.mp3",
                    download_url=f"/api/optimizer/packages/{package_id}/chunk_001.mp3",
                    start_sec=0,
                    end_sec=65,
                    duration_sec=65,
                    bytes=128,
                )
            ],
            manifest_url=f"/api/optimizer/packages/{package_id}/manifest.json",
            package_url=f"/api/optimizer/packages/{package_id}/optimized_package.zip",
            output_dir=output_root / package_id,
        )

    monkeypatch.setattr(app_module, "optimize_audio_package", fake_optimize)
    client = TestClient(app_module.create_app(settings))

    analysis_response = client.post(
        "/api/optimizer/analyze",
        data={"destination": "gemini", "language": "auto"},
        files={"file": ("phone.m4a", b"phone recording", "audio/mp4")},
    )
    assert analysis_response.status_code == 200
    analysis = analysis_response.json()
    upload_id = analysis["upload_id"]
    staged_dir = settings.tmp_dir / "optimizer-uploads" / upload_id
    assert staged_dir.is_dir()
    assert analysis["quick_scan"]["glossary"] == ["OASIS"]

    package_response = client.post(
        "/api/optimizer/package",
        data={"destination": "gemini", "upload_id": upload_id},
    )
    assert package_response.status_code == 200
    assert captured_source and captured_source[0].name == "phone.m4a"
    assert not staged_dir.exists()


def test_background_workflow_survives_client_request_and_keeps_text_out_of_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = replace(
        make_test_settings(tmp_path),
        enable_gemini_transcription=True,
        gemini_api_key="test-gemini-key",
        auto_export_dir=tmp_path / "downloads" / "PhoneScribe",
    )
    media_info = MediaInfo(
        filename="phone.m4a",
        duration_sec=65.0,
        sample_rate=48000,
        channels=1,
    )
    monkeypatch.setattr(app_module, "probe_media", lambda *_args: media_info)
    monkeypatch.setattr(
        app_module,
        "quick_scan_glossary",
        lambda *_args, **_kwargs: pytest.fail("shared mode must skip the quick scan"),
    )

    def fake_optimize(
        input_path: Path,
        output_root: Path,
        _settings: object,
        request: OptimizerRequest,
        *,
        package_id: str,
    ) -> OptimizedPackage:
        recommendation = recommend_optimization(media_info, input_path.stat().st_size, request)
        output_dir = output_root / package_id
        output_dir.mkdir(parents=True, exist_ok=True)
        chunk = OptimizedChunk(
            filename="chunk_001.mp3",
            download_url=f"/api/optimizer/packages/{package_id}/chunk_001.mp3",
            start_sec=0,
            end_sec=65,
            duration_sec=65,
            bytes=5,
        )
        (output_dir / chunk.filename).write_bytes(b"audio")
        (output_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "id": package_id,
                    "source": asdict(media_info),
                    "recommendation": asdict(recommendation),
                    "chunks": [asdict(chunk)],
                }
            ),
            encoding="utf-8",
        )
        return OptimizedPackage(
            id=package_id,
            recommendation=recommendation,
            source=media_info,
            chunks=[chunk],
            manifest_url=f"/api/optimizer/packages/{package_id}/manifest.json",
            package_url=f"/api/optimizer/packages/{package_id}/optimized_package.zip",
            output_dir=output_dir,
        )

    def fake_transcribe(
        package_dir: Path,
        _settings: object,
        *,
        api_key: str | None = None,
    ) -> GeminiTranscriptResult:
        assert api_key == "test-gemini-key"
        chunk = GeminiChunkTranscript(
            filename="chunk_001.mp3",
            start_sec=0,
            end_sec=65,
            delivery="inline",
            mime_type="audio/mp3",
            text="background transcript",
        )
        txt_path = package_dir / "gemini_transcript.txt"
        json_path = package_dir / "gemini_transcript.json"
        txt_path.write_text("background transcript", encoding="utf-8")
        json_path.write_text(
            json.dumps(
                {
                    "provider": "gemini",
                    "model": "test-model",
                    "suggested_filename": "meeting",
                    "chunks": [asdict(chunk)],
                    "text": "background transcript",
                }
            ),
            encoding="utf-8",
        )
        return GeminiTranscriptResult(
            provider="gemini",
            model="test-model",
            text="background transcript",
            suggested_filename="meeting",
            chunks=[chunk],
            txt_path=txt_path,
            json_path=json_path,
        )

    monkeypatch.setattr(app_module, "optimize_audio_package", fake_optimize)
    monkeypatch.setattr(app_module, "transcribe_gemini_package", fake_transcribe)
    client = TestClient(app_module.create_app(settings))
    analysis = client.post(
        "/api/optimizer/analyze",
        data={"destination": "gemini", "language": "auto", "quick_scan": "false"},
        files={"file": ("phone.m4a", b"phone recording", "audio/mp4")},
    ).json()
    assert analysis["quick_scan"]["detected_language"] == "unknown"

    start_response = client.post(
        "/api/workflows",
        data={"destination": "gemini", "upload_id": analysis["upload_id"]},
        headers={"X-Gemini-API-Key": "test-gemini-key"},
    )
    assert start_response.status_code == 202
    workflow_id = start_response.json()["workflow_id"]

    status_response = client.get(f"/api/workflows/{workflow_id}")
    assert status_response.status_code == 200
    status = status_response.json()
    assert status["status"] == "complete"
    assert status["transcript"]["text"] == "background transcript"
    assert status["package"]["source"]["filename"] == "phone.m4a"
    assert status["auto_exported"] is True
    assert status["auto_export_error"] is None
    assert (settings.auto_export_dir / "meeting.txt").read_text(encoding="utf-8") == (
        "background transcript"
    )

    state_path = settings.tmp_dir / "workflows" / f"{workflow_id}.json"
    assert "background transcript" not in state_path.read_text(encoding="utf-8")


def test_merge_assigns_segment_by_maximum_time_overlap() -> None:
    transcript = merge_transcript(
        job_id="job_overlap",
        media_info=MediaInfo("meeting.wav", duration_sec=3.0, sample_rate=16000, channels=1),
        config=TranscriptConfig(asr_engine="faster-whisper"),
        asr_result=ASRResult(
            engine_name="faster-whisper",
            model_name="turbo",
            segments=[
                ASRSegment(
                    start=0.0,
                    end=2.0,
                    text="hello agenda",
                    confidence=0.9,
                    words=[
                        ASRWord("hello", 0.0, 0.8),
                        ASRWord("agenda", 0.8, 2.0),
                    ],
                )
            ],
        ),
        speaker_turns=[
            SpeakerTurn(0.0, 0.9, "SPEAKER_00"),
            SpeakerTurn(0.8, 2.0, "SPEAKER_01"),
        ],
    )

    assert transcript.segments[0].speaker == "SPEAKER_01"
    assert transcript.segments[0].overlap is True
    assert transcript.segments[0].needs_review is True


def test_needs_review_uses_confidence_and_repetition() -> None:
    transcript = merge_transcript(
        job_id="job_review",
        media_info=MediaInfo("meeting.wav", duration_sec=4.0, sample_rate=16000, channels=1),
        config=TranscriptConfig(asr_engine="faster-whisper"),
        asr_result=ASRResult(
            engine_name="faster-whisper",
            model_name="turbo",
            segments=[
                ASRSegment(
                    start=0.0,
                    end=4.0,
                    text="yes yes yes yes yes yes yes yes",
                    confidence=0.8,
                    compression_ratio=3.0,
                )
            ],
        ),
        speaker_turns=[SpeakerTurn(0.0, 4.0, "SPEAKER_00")],
    )

    assert transcript.segments[0].needs_review is True


def test_rule_formatter_applies_glossary_casing_without_invention() -> None:
    transcript = Transcript(
        id="job_glossary",
        source=SourceInfo(filename="meeting.wav", duration_sec=1.0),
        config=TranscriptConfig(),
        speakers=[Speaker(id="SPEAKER_00", display_name="Speaker 1")],
        segments=[
            TranscriptSegment(
                id="seg_000001",
                start=0.0,
                end=1.0,
                text_raw="gpu review",
                text_clean="gpu review",
            )
        ],
    )

    cleaned = RuleBasedFormatterEngine().clean(transcript, ["GPU", "Kubernetes"])

    assert cleaned.segments[0].text_clean == "GPU review"
    assert "Kubernetes" not in cleaned.segments[0].text_clean


def test_faster_whisper_cpu_plan_is_int8() -> None:
    plan = faster_whisper_runtime_plan(cpu=True)

    assert plan[0].device == "cpu"
    assert plan[0].compute_type == "int8"
    assert plan[0].batch_size == 8


def test_glossary_terms_come_from_scanned_transcript_text() -> None:
    terms = extract_glossary_terms(
        "AI 기반 B2G IR 미팅입니다. 투자 매출 투자 매출 스파클랩 스파클랩 "
        "B2G 시장과 B2C 전략을 봅니다. 하하하하하하하하하하하하하하"
    )

    assert "AI" in terms
    assert "B2G" in terms
    assert "스파클랩" in terms
    assert "없는단어" not in terms
    assert all("하하" not in term for term in terms)


def test_prepare_llm_audio_command_removes_silence_and_truncates() -> None:
    cmd = build_prepare_audio_command(
        ffmpeg="ffmpeg",
        input_path=Path("in.m4a"),
        output_path=Path("out.m4a"),
        remove_silence=True,
        max_minutes=90,
        bitrate_kbps=32,
    )
    joined = " ".join(cmd)

    assert "silenceremove=" in joined
    assert "atrim=duration=5400.000" in joined
    assert "-b:a 32k" in joined
    assert cmd[-1] == "out.m4a"


def test_openai_recommendation_chunks_90_minute_audio() -> None:
    rec = recommend_optimization(
        MediaInfo("meeting.m4a", duration_sec=90 * 60, sample_rate=48000, channels=2),
        input_bytes=70_000_000,
        request=OptimizerRequest(destination="openai", openai_model="gpt-4o-transcribe"),
    )

    assert rec.codec == "m4a"
    assert rec.sample_rate_hz == 16000
    assert rec.channels == 1
    assert rec.chunk_count == 9
    assert rec.chunk_minutes is not None and rec.chunk_minutes <= 25
    assert rec.projected_chunk_mb <= 24
    assert rec.estimated_cost_usd == 0.54


def test_gemini_recommendation_chunks_long_meetings_for_reliable_retries() -> None:
    rec = recommend_optimization(
        MediaInfo("meeting.m4a", duration_sec=90 * 60, sample_rate=48000, channels=2),
        input_bytes=70_000_000,
        request=OptimizerRequest(destination="gemini"),
    )

    assert rec.codec == "mp3"
    assert rec.chunk_count == 3
    assert rec.chunk_minutes == 30
    assert rec.delivery == "Inline generateContent"
    assert rec.projected_chunk_mb == pytest.approx(6.9)
    assert rec.estimated_tokens == 172800


def test_gemini_mime_mapping_accepts_phone_m4a_package() -> None:
    assert gemini_mime_type_for_path(Path("chunk_001.m4a")) == "audio/aac"
    assert gemini_mime_type_for_path(Path("chunk_001.mp3")) == "audio/mp3"


def test_gemini_inline_limit_is_size_based(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("local_meetscribe.pipeline.gemini.INLINE_LIMIT_BYTES", 10)
    small = tmp_path / "small.mp3"
    large = tmp_path / "large.mp3"
    small.write_bytes(b"1234567890")
    large.write_bytes(b"12345678901")

    assert can_send_gemini_inline(small) is True
    assert can_send_gemini_inline(large) is False


def test_gemini_transcription_is_disabled_by_default(tmp_path: Path) -> None:
    package_dir = tmp_path / "optimized" / "pkg"
    package_dir.mkdir(parents=True)
    (package_dir / "manifest.json").write_text(
        '{"chunks":[{"filename":"chunk_001.mp3","start_sec":0,"end_sec":1}]}',
        encoding="utf-8",
    )
    (package_dir / "chunk_001.mp3").write_bytes(b"not real audio")

    with pytest.raises(LocalMeetScribeError, match="Gemini transcription is off"):
        transcribe_gemini_package(package_dir, make_test_settings(tmp_path))


def test_gemini_request_key_is_explicit_opt_in(tmp_path: Path) -> None:
    package_dir = tmp_path / "optimized" / "pkg"
    package_dir.mkdir(parents=True)
    (package_dir / "manifest.json").write_text('{"chunks": []}', encoding="utf-8")

    with pytest.raises(LocalMeetScribeError, match="does not contain audio chunks"):
        transcribe_gemini_package(
            package_dir,
            make_test_settings(tmp_path),
            api_key="request-only-key",
        )


def test_gemini_transcription_resumes_completed_chunks(tmp_path: Path) -> None:
    package_dir = tmp_path / "optimized" / "pkg"
    package_dir.mkdir(parents=True)
    (package_dir / "manifest.json").write_text(
        json.dumps(
            {
                "chunks": [
                    {
                        "filename": "chunk_001.mp3",
                        "start_sec": 0,
                        "end_sec": 10,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (package_dir / "chunk_001.mp3").write_bytes(b"cached chunk")
    partial_path = package_dir / "gemini_transcript.partial.json"
    partial_path.write_text(
        json.dumps(
            {
                "model": "gemini-3.5-flash",
                "chunks": [
                    {
                        "filename": "chunk_001.mp3",
                        "start_sec": 0,
                        "end_sec": 10,
                        "delivery": "inline",
                        "mime_type": "audio/mp3",
                        "text": "[00:00] cached transcript",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = transcribe_gemini_package(
        package_dir,
        make_test_settings(tmp_path),
        api_key="request-only-key",
    )

    assert result.text == "[00:00] cached transcript"
    assert result.txt_path.exists()
    assert result.json_path.exists()
    assert not partial_path.exists()
    progress = get_gemini_progress(package_dir)
    assert progress.status == "complete"
    assert progress.completed_chunks == 1
    assert progress.total_chunks == 1
    assert progress.progress == 1.0
    assert progress.eta_sec == 0.0


def test_gemini_transcription_reuses_completed_result_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_dir = tmp_path / "optimized" / "pkg"
    package_dir.mkdir(parents=True)
    chunk = {
        "filename": "chunk_001.mp3",
        "start_sec": 0,
        "end_sec": 10,
        "delivery": "inline",
        "mime_type": "audio/mp3",
        "text": "[00:00] completed transcript",
        "model": "gemini-3.5-flash",
    }
    (package_dir / "manifest.json").write_text(
        json.dumps({"chunks": [{"filename": "chunk_001.mp3", "start_sec": 0, "end_sec": 10}]}),
        encoding="utf-8",
    )
    (package_dir / "gemini_transcript.txt").write_text(
        "[00:00] completed transcript",
        encoding="utf-8",
    )
    (package_dir / "gemini_transcript.json").write_text(
        json.dumps(
            {
                "provider": "gemini",
                "model": "gemini-3.5-flash",
                "suggested_filename": "completed",
                "chunks": [chunk],
                "text": "[00:00] completed transcript",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        gemini_module,
        "_load_httpx",
        lambda: pytest.fail("completed transcript must not call Gemini again"),
    )

    result = transcribe_gemini_package(
        package_dir,
        make_test_settings(tmp_path),
        api_key="request-only-key",
    )

    assert result.text == "[00:00] completed transcript"
    assert result.suggested_filename == "completed"
    progress = get_gemini_progress(package_dir)
    assert progress.status == "complete"
    assert progress.progress == 1.0


def test_gemini_progress_recovers_completed_partial_chunks(tmp_path: Path) -> None:
    package_dir = tmp_path / "optimized" / "pkg"
    package_dir.mkdir(parents=True)
    (package_dir / "manifest.json").write_text(
        json.dumps(
            {
                "chunks": [
                    {"filename": "chunk_001.mp3"},
                    {"filename": "chunk_002.mp3"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (package_dir / "gemini_transcript.partial.json").write_text(
        json.dumps({"chunks": [{"filename": "chunk_001.mp3", "text": "not returned"}]}),
        encoding="utf-8",
    )

    progress = get_gemini_progress(package_dir)

    assert progress.status == "idle"
    assert progress.completed_chunks == 1
    assert progress.total_chunks == 2
    assert progress.progress == 0.5
    assert progress.eta_sec is None


def test_gemini_interactions_falls_back_to_stable_audio_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        def __init__(self, status_code: int, payload: dict[str, object]) -> None:
            self.status_code = status_code
            self._payload = payload
            self.headers: dict[str, str] = {}

        def json(self) -> dict[str, object]:
            return self._payload

    class Client:
        def __init__(self) -> None:
            self.models: list[str] = []
            self.urls: list[str] = []

        def request(self, method: str, url: str, **kwargs: object) -> Response:
            assert method == "POST"
            payload = kwargs["json"]
            assert isinstance(payload, dict)
            model = str(payload["model"])
            self.models.append(model)
            self.urls.append(url)
            if model == "gemini-3.6-flash":
                return Response(500, {"error": {"message": "Internal error encountered."}})
            return Response(
                200,
                {
                    "steps": [
                        {
                            "type": "model_output",
                            "content": [{"type": "text", "text": "[00:00] test speech"}],
                        }
                    ]
                },
            )

    audio_path = tmp_path / "chunk.mp3"
    audio_path.write_bytes(b"small audio fixture")
    client = Client()
    monkeypatch.setattr(gemini_module.time, "sleep", lambda _seconds: None)

    generation = gemini_module._generate_inline(
        client,
        replace(make_test_settings(tmp_path), gemini_model="gemini-3.6-flash"),
        audio_path,
        "audio/mp3",
        "Transcribe faithfully.",
    )

    assert generation.model == "gemini-3.5-flash"
    assert generation.text == "[00:00] test speech"
    assert client.models == ["gemini-3.6-flash"] * 3 + ["gemini-3.5-flash"]
    assert all(url.endswith("/v1beta/interactions") for url in client.urls)


def test_gemini_interactions_falls_back_when_http_200_has_no_transcript(
    tmp_path: Path,
) -> None:
    class Response:
        status_code = 200
        headers: dict[str, str] = {}

        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def json(self) -> dict[str, object]:
            return self._payload

    class Client:
        def __init__(self) -> None:
            self.models: list[str] = []

        def request(self, method: str, _url: str, **kwargs: object) -> Response:
            assert method == "POST"
            payload = kwargs["json"]
            assert isinstance(payload, dict)
            model = str(payload["model"])
            self.models.append(model)
            if model == "gemini-3.6-flash":
                return Response({"id": "interaction-without-output", "status": "completed"})
            return Response(
                {
                    "status": "completed",
                    "steps": [
                        {
                            "type": "model_output",
                            "content": [{"type": "text", "text": "[00:00] recovered"}],
                        }
                    ],
                }
            )

    audio_path = tmp_path / "chunk.mp3"
    audio_path.write_bytes(b"small audio fixture")
    client = Client()

    generation = gemini_module._generate_inline(
        client,
        replace(make_test_settings(tmp_path), gemini_model="gemini-3.6-flash"),
        audio_path,
        "audio/mp3",
        "Transcribe faithfully.",
    )

    assert generation.model == "gemini-3.5-flash"
    assert generation.text == "[00:00] recovered"
    assert client.models == ["gemini-3.6-flash", "gemini-3.5-flash"]


def test_failed_workflow_recovers_when_completed_artifacts_exist(tmp_path: Path) -> None:
    settings = make_test_settings(tmp_path)
    package_id = "a" * 32
    workflow_id = "b" * 32
    package_dir = settings.data_dir / "optimized" / package_id
    package_dir.mkdir(parents=True)
    chunk = {
        "filename": "chunk_001.mp3",
        "start_sec": 0,
        "end_sec": 10,
        "delivery": "inline",
        "mime_type": "audio/mp3",
        "text": "completed output",
        "model": "gemini-3.5-flash",
    }
    (package_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source": {
                    "filename": "meeting.m4a",
                    "duration_sec": 10,
                    "sample_rate": 16000,
                    "channels": 1,
                },
                "recommendation": {},
                "chunks": [chunk],
            }
        ),
        encoding="utf-8",
    )
    (package_dir / "gemini_transcript.txt").write_text("completed output", encoding="utf-8")
    (package_dir / "gemini_transcript.json").write_text(
        json.dumps(
            {
                "provider": "gemini",
                "model": "gemini-3.5-flash",
                "suggested_filename": "meeting",
                "chunks": [chunk],
                "text": "completed output",
            }
        ),
        encoding="utf-8",
    )
    workflow_dir = settings.tmp_dir / "workflows"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / f"{workflow_id}.json").write_text(
        json.dumps(
            {
                "workflow_id": workflow_id,
                "package_id": package_id,
                "status": "failed",
                "error": "Unexpected background failure: PermissionError",
            }
        ),
        encoding="utf-8",
    )

    response = TestClient(app_module.create_app(settings)).get(f"/api/workflows/{workflow_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "complete"
    assert payload["error"] is None
    assert payload["transcription_progress"]["status"] == "complete"
    assert payload["transcript"]["text"] == "completed output"
    recovered_state = json.loads((workflow_dir / f"{workflow_id}.json").read_text(encoding="utf-8"))
    assert recovered_state["status"] == "complete"
    assert recovered_state["error"] is None


def test_gemini_retryable_error_keeps_optimized_audio_message() -> None:
    class Response:
        status_code = 500
        headers: dict[str, str] = {}

        @staticmethod
        def json() -> dict[str, object]:
            return {"error": {"message": "Internal error encountered."}}

    with pytest.raises(LocalMeetScribeError, match="optimized audio is saved"):
        gemini_module._raise_for_gemini_error(Response())


def test_transcript_filename_recommendation_uses_date_and_spoken_terms() -> None:
    suggested = suggest_transcript_filename(
        "260724 1500 서울글로벌센터 OASIS 5 Anika.m4a",
        (
            "[00:00] OASIS 창업 지원 프로그램을 안내합니다. "
            "OASIS 창업 지원 대상과 신청 방법을 설명합니다."
        ),
    )

    assert suggested == "260724_OASIS_창업_지원"


def test_download_filenames_preserve_korean_and_expected_extension() -> None:
    assert _safe_filename("서울글로벌센터 OASIS.m4a") == "서울글로벌센터 OASIS.m4a"
    assert (
        _safe_download_filename("260724_OASIS:창업지원.txt", "gemini_transcript.txt")
        == "260724_OASIS_창업지원.txt"
    )
