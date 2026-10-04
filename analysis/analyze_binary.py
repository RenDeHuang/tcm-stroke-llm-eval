# -*- coding: utf-8 -*-
"""Task2/Task3 二分类结果分析（对齐 fendo 的 Acc / Recall / Specificity / F1）。

用法: python3 analyze_binary.py [glob]
"""
import json, glob, os, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import accuracy, sensitivity, specificity, precision, binary_f1

OUT = "/Users/huangrende/Desktop/王健/out/tcm_stroke_eval"
pattern = sys.argv[1] if len(sys.argv) > 1 else "*.jsonl"

def boot_ci(yt, yp, fn, n=1000, seed=0):
    import random
    rng = random.Random(seed); N = len(yt); vals = []
    for _ in range(n):
        idx = [rng.randrange(N) for _ in range(N)]
        a = [yt[i] for i in idx]; b = [yp[i] for i in idx]
        vals.append(fn(a, b))
    vals.sort()
    return vals[int(0.025*n)], vals[int(0.975*n)-1]

rows = []
for f in sorted(glob.glob(f"{OUT}/runs/{pattern}")):
    # T3（分期）为单一任务，只应统计 __full 子集；
    # 历史遗留的 __no_tp 子集是同一提示词的重复执行，必须排除以免重复计数。
    if "__T3_" in f and "__no_tp__" in f:
        continue
    recs = [json.loads(l) for l in open(f)]
    ok = [r for r in recs if not r.get("err") and r.get("pred") is not None]
    if not ok: continue
    base = os.path.basename(f)
    model, cond = base.split("__")[0], base.split("__")[1]
    task, st, lang = cond.split("_")[0], cond.split("_")[1], cond.split("_")[2]
    yt = [int(r["gold_syndrome"]) for r in ok]
    yp = [int(r["pred"]) for r in ok]
    lo, hi = boot_ci(yt, yp, binary_f1)
    rows.append(dict(task=task, model=model, st=st, lang=lang, n=len(ok),
                     acc=accuracy(yt, yp), rec=sensitivity(yt, yp),
                     spec=specificity(yt, yp), prec=precision(yt, yp),
                     f1=binary_f1(yt, yp), lo=lo, hi=hi,
                     unresolved=len(recs) - len(ok)))

if not rows:
    print("no binary results yet"); sys.exit(0)

print(f"{'task':5s}{'model':22s}{'st':5s}{'lg':4s}{'n':>5s}{'Acc':>8s}{'Recall':>8s}{'Spec':>8s}{'F1':>8s}{'95%CI':>18s}")
print("-"*92)
for r in rows:
    print(f"{r['task']:5s}{r['model']:22s}{r['st']:5s}{r['lang']:4s}{r['n']:>5d}"
          f"{r['acc']:>8.3f}{r['rec']:>8.3f}{r['spec']:>8.3f}{r['f1']:>8.3f}"
          f"{f'[{r[chr(108)+chr(111)]:.3f},{r[chr(104)+chr(105)]:.3f}]':>18s}")

print()
print("=== 按任务×模型汇总 ===")
agg = {}
for r in rows:
    k = (r["task"], r["model"])
    agg.setdefault(k, []).append(r)
for (t, m), rs in sorted(agg.items()):
    print(f"{t:5s}{m:22s} 条件={len(rs):2d} Acc={sum(x['acc'] for x in rs)/len(rs):.3f} "
          f"F1={sum(x['f1'] for x in rs)/len(rs):.3f} Recall={sum(x['rec'] for x in rs)/len(rs):.3f} "
          f"Spec={sum(x['spec'] for x in rs)/len(rs):.3f}")
