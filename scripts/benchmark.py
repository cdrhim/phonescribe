from __future__ import annotations

import argparse
import json
from pathlib import Path

from local_meetscribe.evaluation import evaluate_transcripts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    reports = []
    for line in args.cases.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        report = evaluate_transcripts(
            Path(case["pred"]),
            Path(case["ref"]),
            ref_rttm=Path(case["ref_rttm"]) if case.get("ref_rttm") else None,
        )
        reports.append({"case": case.get("id"), "report": report.model_dump()})

    args.out.write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
