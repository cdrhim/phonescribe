# Security and Privacy

LocalMeetScribe is designed for local-first transcription.

- No telemetry is included.
- No external transcription API is called by the default pipeline.
- Model downloads are explicit through `local-meetscribe models download`.
- Runtime files are stored under `LOCAL_MEETSCRIBE_DATA_DIR` and models under
  `LOCAL_MEETSCRIBE_MODELS_DIR`.
- Uploaded audio, normalized WAV files, transcripts, and exports should be treated as
  sensitive local data.
- Transcript contents are not logged by default. Logs should include job IDs and stages only.
- Optional local LLM cleanup is disabled by default and should point to a local endpoint such as
  Ollama on `127.0.0.1`.
- The pyannote diarization model can require a Hugging Face token for first download. Keep tokens
  in `.env` or your shell environment, never in committed files.
