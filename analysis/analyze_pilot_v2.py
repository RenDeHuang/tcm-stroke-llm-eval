# -*- coding: utf-8 -*-
"""pilot 汇总：验证 fendo 对齐后的协议、Tier 分布与各条件指标。"""
import json, glob, os, sys
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from equivalence import score_records
from metrics import macro_f1, weighted_f1, accuracy

OUT = "/Users/huangrende/Desktop/王健/out/tcm_stroke_eval"
VOCAB = json.load(open(f"{OUT}/top5_vocab.json"))
PATTERN = sys.argv[1] if len(sys.argv) > 1 else "*v2fix_pilot*.jsonl"

rows = []
for f in sorted(glob.glob(f"{OUT}/runs/{PATTERN}")):
    base = os.path.basename(f)
    model, cond = base.split("__")[0], base.split("__")[1]
    st, lang = cond.split("_")[1], cond.split("_")[2]
    recs = [json.loads(l) for l in open(f)]
    ok = [r for r in recs if not r.get("err")]
    if not ok:
        continue
    s = score_records(ok, VOCAB)
    yt = [r["gold_syndrome"] for r in ok]
    yp = [r.get("pred") or "__NONE__" for r in ok]
    labels = sorted(set(yt) | set(yp) - {"__NONE__"})
    rows.append(dict(model=model, st=st, lang=lang, n=len(ok),
                     acc=accuracy(yt, yp),
                     macro=macro_f1(yt, yp, [l for l in labels if l != "__NONE__"]),
                     weighted=weighted_f1(yt, yp, [l for l in labels if l != "__NONE__"]),
                     tiers=s["tier_counts"], exempt=s["n_exempt"],
                     finish=Counter(r.get("finish_reason") for r in ok),
                     err=len(recs) - len(ok)))

print(f"{'model':22s}{'st':5s}{'lang':5s}{'n':>4s}{'acc':>7s}{'macroF1':>9s}{'wF1':>7s}{'tier1':>7s}{'tier2':>7s}{'tier3':>7s}{'length':>7s}")
print("-" * 92)
for r in rows:
    print(f"{r['model']:22s}{r['st']:5s}{r['lang']:5s}{r['n']:>4d}{r['acc']:>7.3f}{r['macro']:>9.3f}"
          f"{r['weighted']:>7.3f}{r['tiers'].get(1,0):>7d}{r['tiers'].get(2,0):>7d}"
          f"{r['tiers'].get(3,0):>7d}{r['finish'].get('length',0):>7d}")

agg = defaultdict(list)
tiers = defaultdict(Counter)
fin = defaultdict(Counter)
for r in rows:
    agg[r["model"]].append(r)
    for k, v in r["tiers"].items(): tiers[r["model"]][k] += v
    for k, v in r["finish"].items(): fin[r["model"]][k] += v

print()
print("=== 模型汇总 ===")
print(f"{'model':22s}{'条件':>4s}{'acc':>8s}{'macroF1':>9s}{'wF1':>8s}{'tier分布':>22s}{'length':>8s}")
for m in sorted(agg):
    a = agg[m]
    print(f"{m:22s}{len(a):>4d}{sum(x['acc'] for x in a)/len(a):>8.3f}"
          f"{sum(x['macro'] for x in a)/len(a):>9.3f}{sum(x['weighted'] for x in a)/len(a):>8.3f}"
          f"{str(dict(tiers[m])):>22s}{fin[m].get('length',0):>8d}")

tot = [r for r in rows]
print()
print(f"条件数={len(rows)}  总记录={sum(r['n'] for r in rows)}  "
      f"API错误={sum(r['err'] for r in rows)}  "
      f"截断={sum(r['finish'].get('length',0) for r in rows)}  "
      f"exempt={sum(r['exempt'] for r in rows)}")
