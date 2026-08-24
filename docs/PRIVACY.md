# Privacy

LocalMeetScribe is local-first.

- Uploaded media stays under `LOCAL_MEETSCRIBE_DATA_DIR`.
- Models stay under `LOCAL_MEETSCRIBE_MODELS_DIR`.
- No telemetry is emitted.
- No cloud transcription APIs are used.
- Optional Ollama cleanup is disabled by default and should target a local endpoint.
- Logs should include job IDs and stages only.

Treat `data/`, `models/`, `outputs/`, `.env`, audio files, and transcript exports as sensitive.
They are ignored by Git.
