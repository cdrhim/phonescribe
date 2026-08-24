from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Annotated

import typer

from local_meetscribe.config import get_settings
from local_meetscribe.db import JobStore
from local_meetscribe.evaluation import evaluate_transcripts
from local_meetscribe.pipeline.models import download_profile, get_model_status
from local_meetscribe.pipeline.orchestrator import TranscriptionPipeline
from local_meetscribe.schemas import TranscriptionRequest
from local_meetscribe.utils.errors import LocalMeetScribeError
from local_meetscribe.utils.logging import configure_logging

app = typer.Typer(no_args_is_help=True)
models_app = typer.Typer(no_args_is_help=True)
app.add_typer(models_app, name="models")


@app.command()
def transcribe(
    input: Annotated[Path, typer.Argument(exists=True, readable=True, help="Audio/video input.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output directory.")],
    mode: Annotated[str, typer.Option(help="accurate, fast, or cpu.")] = "accurate",
    language: Annotated[str, typer.Option(help="auto, ko, or en.")] = "auto",
    speakers: Annotated[int | None, typer.Option("--speakers", help="Exact speaker count.")] = None,
    min_speakers: Annotated[int | None, typer.Option("--min-speakers")] = None,
    max_speakers: Annotated[int | None, typer.Option("--max-speakers")] = None,
    glossary: Annotated[Path | None, typer.Option("--glossary", help="Glossary text file.")] = None,
) -> None:
    """Transcribe a meeting recording locally."""
    configure_logging()
    settings = get_settings()
    store = JobStore(settings)
    job_id = uuid.uuid4().hex
    request = TranscriptionRequest(
        mode=mode,  # type: ignore[arg-type]
        language=language,  # type: ignore[arg-type]
        speakers=speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        glossary=_read_glossary(glossary),
    )
    store.create_job(job_id, source_path=input, output_dir=out)

    def progress(stage: str, progress_value: float) -> None:
        store.update_job(job_id, status="running", stage=stage, progress=progress_value)
        typer.echo(f"{job_id} {stage} {progress_value:.0%}")

    try:
        pipeline = TranscriptionPipeline(settings, progress_callback=progress)
        transcript = pipeline.run(input, output_dir=out, request=request, job_id=job_id)
        transcript_path = Path(transcript.exports.json_path or out / "transcript.json")
        store.update_job(
            job_id,
            status="completed",
            stage="completed",
            progress=1.0,
            transcript_path=transcript_path,
        )
    except LocalMeetScribeError as exc:
        store.update_job(job_id, status="failed", stage="failed", progress=1.0, error=str(exc))
        raise typer.Exit(code=2) from exc

    typer.echo(
        json.dumps(
            {
                "job_id": job_id,
                "transcript": transcript.exports.json_path,
                "output_dir": str(out.resolve()),
                "asr_engine": transcript.config.asr_engine,
                "diarization_engine": transcript.config.diarization_engine,
            },
            indent=2,
        )
    )


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Bind port.")] = 8765,
) -> None:
    """Serve the local web UI and API."""
    configure_logging()
    import uvicorn

    uvicorn.run("local_meetscribe.api.app:app", host=host, port=port, reload=False)


@models_app.command("status")
def models_status() -> None:
    """Show installed packages and downloaded local model directories."""
    settings = get_settings()
    rows = get_model_status(settings)
    typer.echo("profile      package          package?  downloaded?  local_path")
    for row in rows:
        typer.echo(
            f"{row.profile:<12} {row.package_module:<16} "
            f"{str(row.package_available):<9} {str(row.downloaded):<12} {row.local_path}"
        )


@models_app.command("download")
def models_download(
    profile: Annotated[str, typer.Option("--profile", help="accurate, fast, or diarization.")],
) -> None:
    """Download model files explicitly into the local models directory."""
    settings = get_settings()
    try:
        paths = download_profile(profile, settings)
    except LocalMeetScribeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    for path in paths:
        typer.echo(str(path))


@app.command()
def eval(
    pred: Annotated[Path, typer.Option("--pred", exists=True, readable=True)],
    ref: Annotated[Path, typer.Option("--ref", exists=True, readable=True)],
    ref_rttm: Annotated[Path | None, typer.Option("--ref-rttm", exists=True, readable=True)] = None,
) -> None:
    """Evaluate a transcript against a reference transcript JSON."""
    report = evaluate_transcripts(pred, ref, ref_rttm=ref_rttm)
    typer.echo(report.model_dump_json(indent=2))


def _read_glossary(path: Path | None) -> list[str]:
    if path is None:
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    app()
