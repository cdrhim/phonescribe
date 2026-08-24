# Troubleshooting

## ffmpeg Missing

Install ffmpeg or provide a 16 kHz mono WAV file.

## Qwen Model Unavailable

Install the optional extra and download models explicitly:

```bash
uv pip install -e .[qwen]
local-meetscribe models download --profile accurate
```

## faster-whisper Missing

```bash
uv pip install -e .[whisper]
local-meetscribe models download --profile fast
```

## pyannote Skipped

Set `HF_TOKEN`, accept the model terms, install `[diarization]`, then download the profile.
If unavailable, LocalMeetScribe intentionally falls back to `SPEAKER_00`.

## CUDA OOM

Use `--mode cpu`, the Qwen 0.6B model, or faster-whisper CPU/int8.

## Mock Transcript Appears

The base install uses mocked ASR when real local models are unavailable and mocks are enabled.
Install extras and run `local-meetscribe models status`.

## Gemini Button Disabled

Gemini transcription is opt-in. Install `[llm]`, set
`LOCAL_MEETSCRIBE_ENABLE_GEMINI_TRANSCRIPTION=true`, set `GEMINI_API_KEY`, and restart the server.

## Gemini Rejects m4a

Use the default Gemini recommendation so the package is mp3. The app accepts phone m4a input, but
Gemini package output should stay on the provider default unless you are testing a specific format.
