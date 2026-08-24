# Install

## Base

```bash
uv venv
uv pip install -e .[dev]
```

Pip fallback:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

Install ffmpeg for non-WAV inputs and preprocessing:

```bash
winget install Gyan.FFmpeg
```

macOS:

```bash
brew install ffmpeg
```

Linux:

```bash
sudo apt-get install ffmpeg
```

## Frontend

```bash
cd frontend
npm install
npm run build
```

## Optional Gemini Transcription

Gemini dispatch is intentionally separate from the local default pipeline:

```bash
uv pip install -e .[llm]
set LOCAL_MEETSCRIBE_ENABLE_GEMINI_TRANSCRIPTION=true
set GEMINI_API_KEY=your_google_ai_studio_key
set LOCAL_MEETSCRIBE_GEMINI_MODEL=gemini-3.6-flash
```

Restart the server after changing these values. Phone `.m4a` recordings are accepted as input;
the optimizer re-encodes them into Gemini-ready mp3 by default.

## Serve

```bash
local-meetscribe serve --host 127.0.0.1 --port 8765
```

Hardware-specific setup is documented in [RUNTIME.md](RUNTIME.md).
