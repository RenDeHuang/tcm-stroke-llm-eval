# -*- coding: utf-8 -*-
"""结果盘点器：输出当前所有条件的完成度与指标，供成稿与评审使用。

权威口径：本脚本 + verify_data_claims.py 是数字的唯二来源。
未满全量的条件标记 PARTIAL，禁止当作结论引用。
用法: python3 inventory_results.py
"""
import json, glob, os, re, sys
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from equivalence import score_records
from metrics import accuracy, macro_f1, weighted_f1, sensitivity, specificity, precision, binary_f1

OUT = "/Users/huangrende/Desktop/王健/out/tcm_stroke_eval"
VOCAB = json.load(open(f"{OUT}/top5_vocab.json"))

# 理论规模
EXPECT = {"T1": 6906, "BAL": 565, "T2": 2555, "T2B": 392, "T3": 6851, "ABL": 565}

def load(f):
    recs = []
    seen = {}
    for l in open(f):
        l = l.strip()
        if not l: continue
        try: r = json.loads(l)
        except Exception: continue
        cid = r.get("case_id")
        if cid is None:
            recs.append(r); continue
        # 重试追加：保留最后一条有 pred 的记录
        if cid in seen:
            if r.get("pred") is not None or not r.get("err"):
                seen[cid] = r
        else:
            seen[cid] = r
    return list(seen.values())

t1_rows, bin_rows = [], []
for f in sorted(glob.glob(f"{OUT}/runs/*.jsonl")):
    base = os.path.basename(f)
    if "__no_tp__" in base:      # 历史 T3 重复产物，排除
        continue
    parts = base.split("__")
    model = parts[0]
    cond = parts[1]
    m = re.match(r"(T1|BAL|T2|T2B|T3|ABL)_(P\d)_(zh|en)$", cond)
    if not m: continue
    task, st, lang = m.groups()
    recs = load(f)
    ok = [r for r in recs if not r.get("err") and r.get("pred") is not None]
    if not ok: continue
    exp = EXPECT.get(task, 0)
    full = len(ok) >= exp * 0.999 if exp else False
    ratio = len(ok) / exp if exp else 0
    if task in ("T1", "BAL", "ABL"):
        s = score_records(ok, VOCAB)
        yt = [r["gold_syndrome"] for r in ok]
        yp = [r.get("pred") or "__NONE__" for r in ok]
        labels = sorted(set(yt) & set(VOCAB))
        t1_rows.append(dict(task=task, model=model, st=st, lang=lang, n=len(ok), exp=exp,
            full=full, ratio=ratio, acc=accuracy(yt, yp),
            macro=macro_f1(yt, yp, labels), weighted=weighted_f1(yt, yp, labels),
            tiers=s["tier_counts"], err=len(recs) - len(ok)))
    else:
        yt = [int(r["gold_syndrome"]) for r in ok]
        yp = [int(r["pred"]) for r in ok]
        bin_rows.append(dict(task=task, model=model, st=st, lang=lang, n=len(ok), exp=exp,
            full=full, ratio=ratio, acc=accuracy(yt, yp), rec=sensitivity(yt, yp),
            spec=specificity(yt, yp), prec=precision(yt, yp), f1=binary_f1(yt, yp),
            err=len(recs) - len(ok)))

print("=" * 104)
print("PART 1  Task1 / 平衡子集（多分类：Acc / Macro-F1 / Weighted-F1）")
print("=" * 104)
print(f"{'task':5s}{'model':22s}{'st':4s}{'lg':4s}{'n':>6s}{'exp':>6s}{'%':>7s}{'acc':>8s}{'macro':>8s}{'wF1':>8s}  flag")
for r in sorted(t1_rows, key=lambda x: (x["task"], x["st"], x["model"], x["lang"])):
    flag = "FULL" if r["full"] else "PARTIAL"
    print(f"{r['task']:5s}{r['model']:22s}{r['st']:4s}{r['lang']:4s}{r['n']:>6d}{r['exp']:>6d}"
          f"{r['ratio']:>7.1%}{r['acc']:>8.3f}{r['macro']:>8.3f}{r['weighted']:>8.3f}  {flag}")

print()
print("=" * 104)
print("PART 2  Task2 / Task2B / Task3（二分类：Acc / Recall / Specificity / F1）")
print("=" * 104)
print(f"{'task':5s}{'model':22s}{'st':4s}{'lg':4s}{'n':>6s}{'exp':>6s}{'%':>7s}{'acc':>8s}{'rec':>8s}{'spec':>8s}{'f1':>8s}  flag")
for r in sorted(bin_rows, key=lambda x: (x["task"], x["st"], x["model"], x["lang"])):
    flag = "FULL" if r["full"] else "PARTIAL"
    print(f"{r['task']:5s}{r['model']:22s}{r['st']:4s}{r['lang']:4s}{r['n']:>6d}{r['exp']:>6d}"
          f"{r['ratio']:>7.1%}{r['acc']:>8.3f}{r['rec']:>8.3f}{r['spec']:>8.3f}{r['f1']:>8.3f}  {flag}")

print()
print("=" * 104)
print("PART 3  完成度汇总")
print("=" * 104)
allr = t1_rows + bin_rows
full = [r for r in allr if r["full"]]
print(f"  条件总数       = {len(allr)}")
print(f"  已满全量       = {len(full)}  ({len(full)/max(len(allr),1):.1%})")
print(f"  未满（PARTIAL）= {len(allr)-len(full)}")
bytask = defaultdict(lambda: [0, 0])
for r in allr:
    bytask[r["task"]][0] += 1
    if r["full"]: bytask[r["task"]][1] += 1
for t in sorted(bytask):
    c, f_ = bytask[t]
    print(f"    {t:5s} {f_}/{c} 满全量")
print()
missing = []
for task in ("T1", "T2", "T3"):
    have = {f"{r['model']}|{r['st']}|{r['lang']}" for r in allr if r["task"] == task and r["full"]}
    models = ["deepseek-v4-flash", "glm-5.2", "openpangu-2.0-flash", "qwen3-32b"]
    for mo in models:
        for st in ("P1", "P2", "P3", "P4"):
            for lg in ("zh", "en"):
                k = f"{mo}|{st}|{lg}"
                if k not in have:
                    missing.append(f"{task} {k}")
print(f"  未满全量的条件共 {len(missing)} 个（Task1 理论 32 / Task2 32 / Task3 32）")
