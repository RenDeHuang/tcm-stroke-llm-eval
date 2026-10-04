#!/usr/bin/env python3
"""Verify that one fendo-aligned Task 1 model matrix is complete.

An API error is operationally incomplete and must be retried. A successful
response with an invalid/ambiguous model answer is retained as a model result
and is not silently converted into an API success.
"""
import argparse
import json
import os
import sys

BASE = "/Users/huangrende/Desktop/王健"
OUT = f"{BASE}/out/tcm_stroke_eval"
sys.path.insert(0, f"{BASE}/scripts/tcm_stroke_eval")
from runner import rkey  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=6906)
    args = ap.parse_args()

    incomplete = False
    for strategy in ("P1", "P2", "P3", "P4"):
        for lang in ("zh", "en"):
            key = rkey(args.model, "T1", strategy, lang, args.n, args.tag)
            path = os.path.join(OUT, "runs", f"{key}.jsonl")
            rows = []
            if os.path.exists(path):
                with open(path, encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            incomplete = True
            # Keep every attempt for auditability, but judge completion by the
            # latest attempt for each case. A transient earlier error is fine
            # once that same case has been successfully retried.
            latest = {}
            for row in rows:
                if row.get("case_id"):
                    latest[row["case_id"]] = row
            errors = sum(1 for row in latest.values() if row.get("err"))
            successful = {
                case_id for case_id, row in latest.items() if not row.get("err")
            }
            ok = len(successful) == args.n and errors == 0
            if not ok:
                incomplete = True
            print(
                f"{strategy}/{lang}: success={len(successful)}/{args.n} "
                f"errors={errors} {'OK' if ok else 'INCOMPLETE'}"
            )
    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
