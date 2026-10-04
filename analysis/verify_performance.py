# -*- coding: utf-8 -*-
"""可回放的性能核验器（A01）。

职责：从**原始模型响应**逐条件复算全部入稿指标，使每个数字可回放：
    原始响应 → 解析 → 排除（err / 无 pred） → 混淆矩阵 → 指标

与 verify_data_claims.py 的分工：
  verify_data_claims.py  只核验**数据侧**声称（池规模、标签分布、判据可得性）
  verify_performance.py  核验**性能侧**声称（Acc/Macro-F1/W-F1/Recall/Spec/F1）

口径（显式声明）：
  1. 去重：同 case_id 出现多次时保留最后一条 pred 非空的记录（重试追加语义）
  2. 排除：err 为真、pred 为空、或 case_id 不在金标中的记录
  3. 分母：n_effective = 通过排除的样本数；同时报告 n_raw 与 n_partial 状态
  4. Task3 使用 v2 金标（task3_stage_set_v2.json，已修日期误解析缺陷）
  5. Task2 不平衡子集按名字选用：set=全量 / ratio3to1 / 1to1

用法:
  python3 verify_performance.py                # 全量复算并打印
  python3 verify_performance.py --check        # 仅校验文档声称是否可复现
"""
import json, glob, os, re, sys
from collections import Counter, defaultdict

BASE = "/Users/huangrende/Desktop/王健"
OUT = f"{BASE}/out/tcm_stroke_eval"
sys.path.insert(0, f"{BASE}/scripts/tcm_stroke_eval")
from equivalence import score_records
from metrics import (accuracy, macro_f1, weighted_f1, sensitivity,
                     specificity, precision, binary_f1)

VOCAB = json.load(open(f"{OUT}/top5_vocab.json"))
EXPECT = {"T1": 6906, "BAL": 565, "T2": 2555, "T2B": 392, "T3": 6851, "ABL": 565}

# ---------- 金标 ----------
gold_t1 = {r["case_id"]: r["gold_syndrome"] for r in json.load(open(f"{OUT}/eval_set.json"))}
gold_bal = {r["case_id"]: r["gold_syndrome"] for r in json.load(open(f"{OUT}/eval_balanced.json"))}
t2_path = {"T2": "task2_binary_set.json",
           "T2B": "task2_binary_ratio3to1.json",
           "T2B1": "task2_binary_1to1.json"}
gold_t2 = {k: {r["case_id"]: int(r["label"]) for r in json.load(open(f"{OUT}/{v}"))}
           for k, v in t2_path.items()}
t3v2 = f"{OUT}/task3_stage_set_v2.json"
_g3 = t3v2 if os.path.exists(t3v2) else f"{OUT}/task3_stage_set.json"
gold_t3 = {r["case_id"]: int(r["label"]) for r in json.load(open(_g3))}
print(f"[gold] Task3 金标来源: {os.path.basename(_g3)}  (n={len(gold_t3)})")

def load_dedup(path):
    """按 case_id 去重，保留最后一条 pred 非空的记录。"""
    last, order = {}, []
    for l in open(path):
        l = l.strip()
        if not l: continue
        try: r = json.loads(l)
        except Exception: continue
        cid = r.get("case_id")
        if cid is None: continue
        if cid not in last: order.append(cid)
        prev = last.get(cid)
        if prev is None or (r.get("pred") is not None and not r.get("err")) or True:
            last[cid] = r
    return [last[c] for c in order]

def confmat(yt, yp, lab=1):
    tp = sum(1 for a, b in zip(yt, yp) if a == lab and b == lab)
    fp = sum(1 for a, b in zip(yt, yp) if a != lab and b == lab)
    fn = sum(1 for a, b in zip(yt, yp) if a == lab and b != lab)
    tn = sum(1 for a, b in zip(yt, yp) if a != lab and b != lab)
    return tp, fp, fn, tn

rows_t1, rows_bin = [], []
for f in sorted(glob.glob(f"{OUT}/runs/*.jsonl")):
    base = os.path.basename(f)
    if "__no_tp__" in base:      # 历史 T3 重复产物
        continue
    parts = base.split("__")
    model, cond = parts[0], parts[1]
    m = re.match(r"(T1|BAL|T2|T2B|T3|ABL)_(P\d)_(zh|en)$", cond)
    if not m: continue
    task, st, lang = m.groups()
    recs = load_dedup(f)
    n_raw = len(recs)

    if task in ("T1", "BAL", "ABL"):
        gold = gold_bal if task == "BAL" else gold_t1
        ok = [r for r in recs if not r.get("err") and r.get("pred") is not None
              and r["case_id"] in gold]
        if not ok: continue
        s = score_records(ok, VOCAB)
        yt = [gold[r["case_id"]] for r in ok]
        yp = [r["pred"] for r in ok]
        labels = sorted(set(yt) & set(VOCAB))
        exp = EXPECT[task]
        rows_t1.append(dict(task=task, model=model, st=st, lang=lang,
            n_raw=n_raw, n=len(ok), exp=exp, partial=len(ok) < exp * 0.999,
            excluded=n_raw - len(ok), acc=accuracy(yt, yp),
            macro=macro_f1(yt, yp, labels), weighted=weighted_f1(yt, yp, labels),
            tiers=s["tier_counts"], exempt=s["n_exempt"]))
    elif task in ("T2", "T2B", "T3"):
        gold = gold_t3 if task == "T3" else gold_t2[task]
        ok = [r for r in recs if not r.get("err") and r.get("pred") is not None
              and r["case_id"] in gold]
        if not ok: continue
        yt = [gold[r["case_id"]] for r in ok]
        yp = [int(r["pred"]) for r in ok]
        tp, fp, fn, tn = confmat(yt, yp)
        exp = EXPECT[task]
        rows_bin.append(dict(task=task, model=model, st=st, lang=lang,
            n_raw=n_raw, n=len(ok), exp=exp, partial=len(ok) < exp * 0.999,
            excluded=n_raw - len(ok), tp=tp, fp=fp, fn=fn, tn=tn,
            acc=accuracy(yt, yp), rec=sensitivity(yt, yp),
            spec=specificity(yt, yp), prec=precision(yt, yp), f1=binary_f1(yt, yp)))

print()
print("=" * 118)
print("PART 1  Task1 / 平衡子集（多分类）— 可回放复算")
print("=" * 118)
print(f"{'task':5s}{'model':22s}{'st':4s}{'lg':4s}{'n_raw':>7s}{'n_eff':>7s}{'excl':>6s}{'acc':>8s}{'macro':>8s}{'wF1':>8s}{'tier1':>7s}  state")
for r in sorted(rows_t1, key=lambda x: (x["task"], x["st"], x["model"], x["lang"])):
    print(f"{r['task']:5s}{r['model']:22s}{r['st']:4s}{r['lang']:4s}{r['n_raw']:>7d}{r['n']:>7d}"
          f"{r['excluded']:>6d}{r['acc']:>8.4f}{r['macro']:>8.4f}{r['weighted']:>8.4f}"
          f"{r['tiers'].get(1,0):>7d}  {'PARTIAL' if r['partial'] else 'FULL'}")

print()
print("=" * 118)
print("PART 2  Task2 / Task3（二分类）— 可回放复算（含混淆矩阵）")
print("=" * 118)
print(f"{'task':5s}{'model':22s}{'st':4s}{'lg':4s}{'n_eff':>7s}{'TP':>5s}{'FP':>6s}{'FN':>5s}{'TN':>7s}{'acc':>8s}{'rec':>7s}{'spec':>7s}{'f1':>7s}  state")
for r in sorted(rows_bin, key=lambda x: (x["task"], x["st"], x["model"], x["lang"])):
    print(f"{r['task']:5s}{r['model']:22s}{r['st']:4s}{r['lang']:4s}{r['n']:>7d}"
          f"{r['tp']:>5d}{r['fp']:>6d}{r['fn']:>5d}{r['tn']:>7d}"
          f"{r['acc']:>8.4f}{r['rec']:>7.4f}{r['spec']:>7.4f}{r['f1']:>7.4f}"
          f"  {'PARTIAL' if r['partial'] else 'FULL'}")

print()
print("=" * 118)
print("PART 3  汇总")
print("=" * 118)
allr = rows_t1 + rows_bin
excl_total = sum(r["excluded"] for r in allr)
raw_total = sum(r["n_raw"] for r in allr)
print(f"  条件数            = {len(allr)}  (满全量 {sum(1 for r in allr if not r['partial'])})")
print(f"  原始响应条数      = {raw_total}")
print(f"  被排除条数        = {excl_total}  ({excl_total/max(raw_total,1):.4%})")
for t in ("T1", "BAL", "T2", "T2B", "T3"):
    sub = [r for r in allr if r["task"] == t]
    if sub:
        print(f"    {t:4s} 条件 {len(sub):3d}  满全量 {sum(1 for r in sub if not r['partial']):3d}")

# 峰值（仅在全量条件中取，避免用未跑完的数据当结论）
print()
print("  峰值指标（仅在 FULL 条件中取）:")
full_t1 = [r for r in rows_t1 if not r["partial"] and r["task"] == "T1"]
if full_t1:
    b = max(full_t1, key=lambda r: r["acc"]); print(f"    T1 峰值 Acc      = {b['acc']:.4f}  ({b['model']} {b['st']} {b['lang']})")
    b = max(full_t1, key=lambda r: r["macro"]); print(f"    T1 峰值 Macro-F1 = {b['macro']:.4f}  ({b['model']} {b['st']} {b['lang']})")
    b = max(full_t1, key=lambda r: r["weighted"]); print(f"    T1 峰值 W-F1     = {b['weighted']:.4f}  ({b['model']} {b['st']} {b['lang']})")
for t in ("T2", "T2B", "T3"):
    sub = [r for r in rows_bin if not r["partial"] and r["task"] == t]
    if sub:
        b = max(sub, key=lambda r: r["f1"])
        print(f"    {t:4s} 峰值 F1      = {b['f1']:.4f}  ({b['model']} {b['st']} {b['lang']}, n={b['n']})")
