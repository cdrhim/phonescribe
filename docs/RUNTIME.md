# Runtime Integration

LocalMeetScribe runs locally after explicit model download. The mocked engines remain the default
smoke-test path and do not download models.

## CUDA

Install the optional extras you need:

```bash
uv pip install -e .[qwen,whisper,diarization]
```

Recommended CUDA path:

```bash
local-meetscribe models download --profile accurate
local-meetscribe models download --profile fast
local-meetscribe models download --profile diarization
local-meetscribe transcribe meeting.wav --out outputs/cuda --mode accurate --language auto
```

Runtime behavior:

- Qwen3-ASR uses `device_map="cuda:0"` when PyTorch reports CUDA.
- faster-whisper tries CUDA `float16` with batch 16, then CUDA `int8_float16` with batch 8,
  then CPU `int8` with batch 8.
- pyannote community-1 loads on CPU by default and is moved to CUDA when available.
- If CUDA runs out of memory, LocalMeetScribe falls back where possible and emits an actionable
  model error when no local runtime fits.

## Apple Silicon

Recommended path:

```bash
uv pip install -e .[qwen,whisper]
local-meetscribe models download --profile accurate
local-meetscribe transcribe meeting.wav --out outputs/apple --mode accurate --language auto
```

Notes:

- Qwen3-ASR can use PyTorch MPS when available.
- CTranslate2/faster-whisper does not use PyTorch MPS through this adapter; use CPU `int8`.
- Start with `Qwen/Qwen3-ASR-0.6B` if memory is limited.

## CPU Only

```bash
uv pip install -e .[whisper]
local-meetscribe models download --profile fast
local-meetscribe transcribe meeting.wav --out outputs/cpu --mode cpu --language auto
```

CPU mode uses faster-whisper `small` with CTranslate2 `int8`, batch size 8, deterministic decoding,
and segment timestamps for long recordings. This is materially faster than loading `turbo` on CPU
while keeping useful Korean/English meeting quality. For the most accurate path, use CUDA with
`LOCAL_MEETSCRIBE_FASTER_WHISPER_CUDA_MODEL=turbo` or install the Qwen profile.

CPU tuning knobs:

```bash
set LOCAL_MEETSCRIBE_FASTER_WHISPER_CPU_MODEL=small
set LOCAL_MEETSCRIBE_FASTER_WHISPER_CPU_THREADS=0
```

Use `0` CPU threads to let CTranslate2 choose. Set a positive number only if you have benchmarked
the machine.

## Diarization

```bash
uv pip install -e .[diarization]
set HF_TOKEN=your_hugging_face_token
local-meetscribe models download --profile diarization
```

`pyannote/speaker-diarization-community-1` may require accepted Hugging Face terms for first
download. After the model exists under `LOCAL_MEETSCRIBE_MODELS_DIR`, the adapter loads the local
path without a token or network access.

Speaker constraints are passed through:

```bash
local-meetscribe transcribe meeting.wav --out outputs/diarized --speakers 3
local-meetscribe transcribe meeting.wav --out outputs/diarized --min-speakers 2 --max-speakers 5
```

## Glossary

Glossary terms are formatting hints only. They can preserve casing such as `GPU` or `Q4 OKR` when
the corresponding normalized words are already present. They are not inserted into transcript text
when absent.
