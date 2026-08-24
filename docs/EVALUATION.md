# Evaluation

Run:

```bash
local-meetscribe eval --pred transcript.json --ref reference.json
```

Metrics:

- English WER via `jiwer` when installed.
- Korean CER.
- Korean spacing-normalized CER.
- Segment timestamp mean absolute error when segment IDs match.
- DER when a reference RTTM is supplied and `pyannote.metrics` is installed.
- Combined speaker-attributed segment error.

Korean spacing makes WER misleading because spacing can be stylistically inconsistent while the
spoken content is still correct. Report CER for Korean, and use spacing-normalized CER when spacing
is not the evaluation target.

Private benchmark sets can be evaluated with:

```bash
python scripts/benchmark.py --cases path/to/cases.jsonl --out benchmark-results.json
```

Each JSONL case should contain:

```json
{"id": "meeting-001", "pred": "outputs/meeting-001/transcript.json", "ref": "refs/meeting-001.json"}
```

For DER, run the CLI directly with `--ref-rttm`:

```bash
local-meetscribe eval --pred transcript.json --ref reference.json --ref-rttm reference.rttm
```
