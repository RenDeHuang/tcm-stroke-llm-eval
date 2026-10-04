# -*- coding: utf-8 -*-
"""严格对齐版 P1-P4 + P5/P6 标准化干预分析。"""
import glob, json, os, sys
from collections import Counter
B = "/Users/huangrende/Desktop/王健"
OUT = f"{B}/out/tcm_stroke_eval"
RUNS = f"{OUT}/runs"
sys.path.insert(0, f"{B}/scripts/tcm_stroke_eval")
from equivalence import score_records
from metrics import accuracy, macro_f1, weighted_f1, per_class_f1, bootstrap_ci, mcnemar

VOCAB = json.load(open(f"{OUT}/top5_vocab.json"))
MODEL = "deepseek-v4-pro"
TAG = "aligned_v1"


def load_one(strategy, lang="zh", n=400, tag=TAG):
    pat = f"{RUNS}/{MODEL}__T1_{strategy}_{lang}__n{n}__{tag}__*.jsonl"
    fs = sorted(glob.glob(pat))
    if not fs:
        raise FileNotFoundError(pat)
    # 同一版本只保留最新文件；runner 的断点续跑始终写同一文件。
    p = fs[-1]
    seen = {}
    for line in open(p, encoding="utf-8"):
        try:
            r = json.loads(line)
            seen[r["case_id"]] = r
        except Exception:
            pass
    return list(seen.values()), p


def paired(groups):
    ok = None
    for recs in groups.values():
        ids = {r["case_id"] for r in recs if not r.get("err")}
        ok = ids if ok is None else ok & ids
    return {k: [r for r in v if r["case_id"] in ok] for k, v in groups.items()}, len(ok or set())


def summarize(recs):
    s = score_records(recs, VOCAB)
    yt = [d["gold"] for d in s["detail"]]
    yp = [d["mapped"] for d in s["detail"]]
    labels = sorted(set(yt), key=lambda x: VOCAB.index(x) if x in VOCAB else 999)
    mf, mlo, mhi = bootstrap_ci(yt, yp, labels, n=1000, seed=0)
    wf, wlo, whi = bootstrap_ci(yt, yp, labels, fn=weighted_f1, n=1000, seed=0)
    pc = per_class_f1(yt, yp, labels)
    api_ok = sum(1 for r in recs if not r.get("err"))
    parsed = sum(1 for r in recs if not r.get("err") and r.get("pred") is not None)
    oov = sum(1 for d in s["detail"] if d["answer"] and d["mapped"] is None)
    return dict(
        n=len(recs), api_ok=api_ok, api_success=api_ok / max(len(recs), 1),
        parsed=parsed, parse_rate=parsed / max(api_ok, 1),
        acc_L0=s["acc_L0"], acc_L1=s["acc_L1"], naming_gap=s["naming_gap"],
        macroF1=mf, macro_ci=[mlo, mhi], weightedF1=wf, weighted_ci=[wlo, whi],
        per_class=pc, oov=oov, oov_rate=oov / max(len(recs), 1),
    )


def add_row(lines, name, x):
    lines.append(
        f"| {name} | {x['n']} | {x['api_ok']} ({x['api_success']:.1%}) | "
        f"{x['acc_L0']:.4f} | {x['acc_L1']:.4f} | {x['macroF1']:.4f} "
        f"[{x['macro_ci'][0]:.4f},{x['macro_ci'][1]:.4f}] | {x['weightedF1']:.4f} "
        f"[{x['weighted_ci'][0]:.4f},{x['weighted_ci'][1]:.4f}] | {x['oov_rate']:.1%} |"
    )


def main():
    raw = {}
    files = {}
    for st in ["P1", "P2", "P3", "P4"]:
        raw[st], files[st] = load_one(st, "zh")
    pr, n = paired(raw)
    rows = {st: summarize(pr[st]) for st in pr}

    lines = [
        "# 中风病 LLM 证型辨识：严格对齐版结果", "",
        "## 设计", "",
        "P1–P4 共享同一 Top5 候选标签空间、同一输出协议、同一模型和同一 400 例；",
        "只改变参考框架、类别特异判定标准或逐步推理形式。API 失败病例不计入配对效应分析，",
        "但单独报告 API 成功率。", "",
        f"配对成功病例数：**{n}**", "",
        "## P1–P4 主要结果（L1 官方归一化口径）", "",
        "| 策略 | 总n | API成功 | L0严格 | L1归一 | macro-F1 (95% CI) | weighted-F1 (95% CI) | OOV率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for st in ["P1", "P2", "P3", "P4"]:
        add_row(lines, st, rows[st])

    lines += ["", "## 每类 F1", "", "| 策略 | " + " | ".join(VOCAB) + " |", "|---|" + "---:|" * len(VOCAB)]
    for st in ["P1", "P2", "P3", "P4"]:
        lines.append("| " + st + " | " + " | ".join(f"{rows[st]['per_class'].get(v, 0.0):.4f}" for v in VOCAB) + " |")

    lines += ["", "## McNemar（L1，配对病例）", "", "| 对照 | 净增益（相对P1） | p值 |", "|---|---:|---:|"]
    base = {d["case_id"]: d for d in score_records(pr["P1"], VOCAB)["detail"]}
    for st in ["P2", "P3", "P4"]:
        cur = {d["case_id"]: d for d in score_records(pr[st], VOCAB)["detail"]}
        ids = sorted(set(base) & set(cur))
        yt = [base[i]["gold"] for i in ids]
        pa = [base[i]["mapped"] for i in ids]
        pb = [cur[i]["mapped"] for i in ids]
        b, c, p = mcnemar(yt, pa, pb)
        lines.append(f"| P1 vs {st} | {(c-b)/len(ids):+.4f} | {p:.4g} |")

    lines += ["", "## 文件", ""]
    for st in ["P1", "P2", "P3", "P4"]:
        lines.append(f"- {st}: `{files[st]}`")

    out = f"{OUT}/REPORT_ALIGNED.md"
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump({"paired_n": n, "rows": rows, "files": files}, open(f"{OUT}/aligned_summary.json", "w"), ensure_ascii=False, indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
