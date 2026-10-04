# -*- coding: utf-8 -*-
"""汇总 runs/*.jsonl -> 指标表 + bootstrap CI + 混淆矩阵 + 未解析分析。"""
import json, os, glob, sys, csv
from collections import defaultdict, Counter
import numpy as np
BASE="/Users/huangrende/Desktop/王健"; OUT=f"{BASE}/out/tcm_stroke_eval"; RUNS=f"{OUT}/runs"
sys.path.insert(0,f"{BASE}/scripts/tcm_stroke_eval")
from metrics import macro_f1, accuracy, bootstrap_ci, mcnemar

def load():
    d=defaultdict(list)
    for f in glob.glob(f"{RUNS}/*.jsonl"):
        for l in open(f):
            try: r=json.loads(l)
            except: continue
            d[(r["model"],r["task"],r["strategy"],r["lang"],r.get("subset",""))].append(r)
    return d

def main():
    runs=load()
    if not runs: print("无数据"); return
    rows=[]
    for (m,t,s,l,sub),recs in sorted(runs.items()):
        recs=sorted(recs,key=lambda x:x["case_id"])
        yt=[r["gold_syndrome"] for r in recs]; yp=[r["pred"] for r in recs]
        if t=="T2":
            # 二分类：macro-F1 over 2 labels
            f1,lo,hi=bootstrap_ci(yt,yp,["中经络","中脏腑"],fn=macro_f1,n=1000,seed=0)
        else:
            f1,lo,hi=bootstrap_ci(yt,yp,sorted(set(yt)),fn=macro_f1,n=1000,seed=0)
        acc=accuracy(yt,yp); res=sum(1 for r in recs if r["tier"]<3)/len(recs)
        rows.append(dict(task=t,model=m,strategy=s,lang=l,subset=sub,n=len(recs),
                         macroF1=round(f1,4),ci_lo=round(lo,4),ci_hi=round(hi,4),
                         acc=round(acc,4),resolved=round(res,3)))
    os.makedirs(OUT,exist_ok=True)
    with open(f"{OUT}/summary.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"{'task':5s}{'model':22s}{'st':4s}{'lang':5s}{'sub':7s}{'n':>5s}{'macroF1':>9s}{'95%CI':>18s}{'acc':>7s}{'resolved':>9s}")
    for r in rows:
        ci=f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]"
        print(f"{r['task']:5s}{r['model']:22s}{r['strategy']:4s}{r['lang']:5s}{r['subset']:7s}{r['n']:5d}{r['macroF1']:9.4f}{ci:>18s}{r['acc']:7.3f}{r['resolved']:9.2f}")
    # 未解析/词表外分析
    print("\n=== 未解析（Tier3）中的词表外预测 ===")
    oov=Counter()
    for (m,t,s,l,sub),recs in runs.items():
        for r in recs:
            if r["tier"]==3 and r.get("raw"): oov[r["raw"].strip()[:20]]+=1
    for k,v in oov.most_common(15): print(f"  {v:3d}  {k}")
    json.dump(rows,open(f"{OUT}/summary.json","w"),ensure_ascii=False,indent=1)

if __name__=="__main__": main()
