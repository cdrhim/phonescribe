# Models

LocalMeetScribe does not require model downloads for mocked smoke tests.

## Accurate Profile

```bash
uv pip install -e .[qwen]
local-meetscribe models download --profile accurate
```

Downloads:

- `Qwen/Qwen3-ASR-1.7B`
- `Qwen/Qwen3-ASR-0.6B`
- `Qwen/Qwen3-ForcedAligner-0.6B`

## Fast Profile

```bash
uv pip install -e .[whisper]
local-meetscribe models download --profile fast
```

Downloads:

- faster-whisper `turbo`, the documented alias for the large-v3-turbo runtime

CUDA tries `float16`, then `int8_float16`. CPU mode uses `int8`.

## Diarization Profile

```bash
uv pip install -e .[diarization]
set HF_TOKEN=your_token
local-meetscribe models download --profile diarization
```

Downloads:

- `pyannote/speaker-diarization-community-1`

This can require accepted Hugging Face model terms.
After download, LocalMeetScribe loads the local path without a token or network access.

## Status

```bash
local-meetscribe models status
```

The command reports package availability and local model directories.
