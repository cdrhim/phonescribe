# LocalMeetScribe

LocalMeetScribe is a local-first Korean/English meeting transcription app with a CLI,
FastAPI backend, React review UI, SQLite job metadata, speaker labels, timestamps, and exports.

The base install runs immediately with mocked engines. Real local transcription is enabled through
optional extras and explicit model downloads.

## Install

Recommended with `uv`:

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

Install frontend dependencies:

```bash
cd frontend
npm install
npm run build
cd ..
```

## Run With Mocked Engines

```bash
local-meetscribe transcribe meeting.wav --out outputs/demo --mode accurate --language auto
local-meetscribe serve --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765`.

## Real Local ASR Setup

Qwen accurate mode:

```bash
uv pip install -e .[qwen]
local-meetscribe models download --profile accurate
local-meetscribe transcribe meeting.wav --out outputs/qwen --mode accurate --language auto
```

The Qwen adapter follows the official `qwen-asr` package path for
`Qwen3ASRModel.from_pretrained(...)` and optional `Qwen/Qwen3-ForcedAligner-0.6B` timestamps.
See the [Qwen3-ASR model card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B).

faster-whisper fallback:

```bash
uv pip install -e .[whisper]
local-meetscribe models download --profile fast
local-meetscribe transcribe meeting.wav --out outputs/whisper --mode fast --language auto
```

The faster-whisper adapter now uses an adaptive fast path:

- CPU: `small`, `int8`, batched inference, segment timestamps for long meetings.
- CUDA: `turbo` / large-v3-turbo, `float16`, batched inference, word timestamps.

Override the defaults with `LOCAL_MEETSCRIBE_FASTER_WHISPER_CPU_MODEL` and
`LOCAL_MEETSCRIBE_FASTER_WHISPER_CUDA_MODEL`.

Diarization:

```bash
uv pip install -e .[diarization]
set HF_TOKEN=your_hugging_face_token
local-meetscribe models download --profile diarization
local-meetscribe transcribe meeting.wav --out outputs/diarized --speakers 3
```

If pyannote is not installed, the token is missing, or the model is unavailable,
LocalMeetScribe labels everything as `SPEAKER_00`.

## Optional Gemini Transcription

The optimizer can turn phone `.m4a` recordings into Gemini-ready audio packages locally. A
separate `Transcribe with Gemini` button can send that optimized package to Gemini after explicit
consent.

For local use, create a free key at <https://aistudio.google.com/apikey>, paste it into the page,
accept the Google upload notice, upload the recording, and select `Optimize + transcribe`. The key
remains only in the current browser tab and is sent to this local backend for that transcription
request.

For a shared deployment, configure the key on the server instead:

```powershell
$env:LOCAL_MEETSCRIBE_ENABLE_GEMINI_TRANSCRIPTION = "true"
$env:GEMINI_API_KEY = "your_google_ai_studio_key"
$env:LOCAL_MEETSCRIBE_GEMINI_MODEL = "gemini-3.5-flash"
local-meetscribe serve --host 127.0.0.1 --port 8765
```

For Gemini, the default package is mp3 at 16 kHz mono so iPhone/Android m4a inputs do not depend
on Gemini accepting the original m4a container. Google may use free-tier API content to improve
its products, so use local ASR or a paid Gemini project for confidential meetings.

Meetings longer than 45 minutes are split near silence into roughly 30-minute chunks. Completed
Gemini chunks are checkpointed locally, temporary quota/server failures retry automatically, and
pressing `Transcribe with Gemini` again resumes unfinished packages. While transcription is
running, the page shows completed chunks, the active chunk, percentage, and an ETA calculated
after the first chunk completes.

Gemini calls use the Interactions API. Retryable service failures fall back from the configured
model to the stable audio-capable `gemini-3.5-flash` and `gemini-3.5-flash-lite` models. Optimized
audio and completed chunks remain on disk, so a retry does not require another upload or repeat
successful chunks.

With the shared passcode flow, enter the passcode and choose a recording. As soon as the upload is
analyzed and the passcode is verified, optimization and transcription start automatically without
a second button press. On a phone, keep the page open until the initial upload and analysis finish.
The PC then performs optimization and Gemini transcription in the background. The phone screen may
turn off or the browser may close; reopening the same workflow URL restores progress and results.
When the page is visible after completion, the TXT transcript downloads automatically once. The
page requests a Screen Wake Lock while busy when the browser exposes that API. Standard Wake Lock
requires HTTPS, so it works with a Tailscale Serve URL but may be unavailable on a plain Wi-Fi
`http://` address.
While a background workflow is active on Windows, the server also requests that the PC stay awake;
the PC display may still turn off, and the request is released when the workflow finishes.

For Wi-Fi sharing, a local share passcode can unlock one server-side Gemini key. Configure the
passcode locally, open the app through `127.0.0.1`, enter the passcode and API key once, and select
`Save as this PC's default key`. Windows stores the key with user-scoped DPAPI encryption; LAN
browsers receive only the passcode field and never receive the API key. Five failed passcode
attempts from one client trigger a one-minute cooldown. This is a convenience lock for a trusted
LAN, not a substitute for HTTPS and authentication on an internet-facing deployment.

Start or verify the Wi-Fi server with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-network.ps1
```

For private access from anywhere, install Tailscale on this PC and the phone or laptop that will
open LocalMeetScribe, sign both into the same tailnet, then run:

```powershell
tailscale serve --bg http://127.0.0.1:8766
```

Tailscale provides a stable private HTTPS `*.ts.net` address. Keep Tailscale Funnel disabled:
Funnel makes the service public, while Serve limits it to devices and users allowed by the
tailnet.

After transcription, choose `Original filename`, `Auto recommendation`, or enter a custom basename.
The selected name is applied to the TXT, JSON, optimized ZIP, and a downloadable copy of the
original phone recording. Browser security does not allow the page to rename the source file
in place, so the original recording remains unchanged.

## CLI

```bash
local-meetscribe transcribe INPUT --out OUTDIR --mode accurate --language auto
local-meetscribe transcribe INPUT --out OUTDIR --mode fast --language ko --speakers 2
local-meetscribe serve --host 127.0.0.1 --port 8765
local-meetscribe models status
local-meetscribe models download --profile accurate
local-meetscribe eval --pred transcript.json --ref reference.json
```

## Privacy Defaults

- No telemetry.
- No cloud transcription APIs in the default pipeline.
- Gemini sends audio to Google only after a key, explicit consent, and the transcription button
  are used. A server-configured key remains optional.
- No transcript content in logs.
- Models download only through `local-meetscribe models download` unless
  `LOCAL_MEETSCRIBE_ALLOW_MODEL_AUTODOWNLOAD=true`.
- Optional local LLM cleanup is off by default.

Runtime data defaults to `./data`; models default to `./models`.

## Development

```bash
ruff check .
mypy backend/local_meetscribe
pytest
```

Frontend:

```bash
cd frontend
npm run build
```

## Known Limits

- The default pipeline uses mock ASR unless real extras and local models are installed.
- WhisperX alignment and NeMo Sortformer are adapter placeholders.
- DER calculation requires `pyannote.metrics` from the `[diarization]` extra and reference RTTM.

See [docs/RUNTIME.md](docs/RUNTIME.md) for CUDA, Apple Silicon, and CPU-only setup.
