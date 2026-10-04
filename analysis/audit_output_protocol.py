# -*- coding: utf-8 -*-
"""审计 Fendo-style 输出协议：最终答案、Tier、截断和解析分歧。"""
import argparse, glob, json, os
from collections import Counter, defaultdict

BASE = "/Users/huangrende/Desktop/王健"
OUT = f"{BASE}/out/tcm_stroke_eval"
RUNS = f"{OUT}/runs"


def audit_file(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    n = len(rows)
    if not n:
        return None
    def nonempty(k):
        return sum(bool(r.get(k)) for r in rows)
    return {
        "file": path,
        "n": n,
        "api_ok": sum(not r.get("err") for r in rows),
        "tier": dict(Counter(str(r.get("tier")) for r in rows)),
        "exempt": sum(bool(r.get("exempt")) for r in rows),
        "finish_reason": dict(Counter(str(r.get("finish_reason")) for r in rows)),
        "gen_params": dict(Counter(
            f"T{r.get('gen_params',{}).get('temperature')}/P{r.get('gen_params',{}).get('top_p')}/M{r.get('gen_params',{}).get('max_tokens')}"
            if r.get("gen_params") else "none" for r in rows)),
        "answer_present": nonempty("answer"),
        "pred_present": nonempty("pred"),
        "answer_pred_disagree": sum(bool(r.get("pred")) and (r.get("pred") != r.get("answer")) for r in rows),
        "raw_empty": sum(not r.get("raw") for r in rows),
        "error_count": sum(bool(r.get("err")) for r in rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*fendo_tcm_migration_v1*.jsonl")
    ap.add_argument("--out", default=f"{OUT}/protocol_audit_v1.json")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(RUNS, args.glob)))
    reports = [x for f in files if (x := audit_file(f))]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(reports, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps({"files": len(reports), "out": args.out}, ensure_ascii=False))
    for r in reports:
        print(os.path.basename(r["file"]), r["n"], "api_ok", r["api_ok"],
              "tier", r["tier"], "finish", r["finish_reason"],
              "exempt", r["exempt"], "gen", r["gen_params"],
              "pred", r["pred_present"], "disagree", r["answer_pred_disagree"])

if __name__ == "__main__":
    main()
