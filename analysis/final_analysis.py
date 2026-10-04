# -*- coding: utf-8 -*-
"""最终统一分析：主实验 + 均衡子集 + Task2/Task3 + fendo 对标。"""
import json, os, sys, glob
from collections import defaultdict, Counter
import numpy as np
B="/Users/huangrende/Desktop/王健"; OUT=f"{B}/out/tcm_stroke_eval"; RUNS=f"{OUT}/runs"
sys.path.insert(0,f"{B}/scripts/tcm_stroke_eval")
from equivalence import score_records, extract_answer, norm
from metrics import macro_f1, accuracy, bootstrap_ci, mcnemar

vocab=json.load(open(f"{OUT}/top5_vocab.json"))
TOP5=vocab

def load(fn, dedup=True):
    recs=[json.loads(l) for l in open(f"{RUNS}/{fn}.jsonl")]
    if dedup:
        seen={}
        for r in recs: seen[r["case_id"]]=r
        recs=list(seen.values())
    return recs

def paired(recs_dict):
    """取所有组都成功的公共 case。"""
    keys=list(recs_dict)
    common=None
    for k in keys:
        ok=set(r["case_id"] for r in recs_dict[k] if not r.get("err"))
        common=ok if common is None else common&ok
    return {k:[r for r in v if r["case_id"] in common] for k,v in recs_dict.items()}, len(common)

MAIN={"P1 自由":"deepseek-v4-pro__T1_P1_zh__n400__4449545d",
      "P5 闭集":"deepseek-v4-pro__T1_P5_zh__n400__539da249",
      "P6 闭集+定义":"deepseek-v4-pro__T1_P6_zh__n400__56f21376",
      "P7 few-shot":"deepseek-v4-pro__T1_P7_zh__n400__df863e26"}
BAL={"BAL-P1 自由":"deepseek-v4-pro__BAL_P1_zh__n565__54816f22",
     "BAL-P5 闭集":"deepseek-v4-pro__BAL_P5_zh__n565__c0dfdcdc",
     "BAL-P6 闭集+定义":"deepseek-v4-pro__BAL_P6_zh__n565__b133e5d3"}

def summarize(recs_dict, title):
    R={k:score_records(v,vocab) for k,v in recs_dict.items()}
    print(f"\n{'='*82}\n{title}\n{'='*82}")
    print(f"{'组':18s}{'n':>5s}{'L0严格':>9s}{'L1归一':>9s}{'命名成本':>10s}{'macroF1':>9s}{'OOV率':>8s}")
    for k,r in R.items():
        yt=[d["gold"] for d in r["detail"]]; yp=[d["mapped"] for d in r["detail"]]
        f1=macro_f1(yt,yp,sorted(set(yt)))
        oov=sum(r["oov_classes"].values())/r["n"]
        print(f"{k:18s}{r['n']:5d}{r['acc_L0']:9.4f}{r['acc_L1']:9.4f}{r['naming_gap']:+10.4f}{f1:9.4f}{oov:8.1%}")
    return R

def mcnemar_table(R, base):
    A={d["case_id"]:d for d in R[base]["detail"]}
    print(f"\n  配对检验（vs {base}, L1口径）：")
    for k,r in R.items():
        if k==base: continue
        Bv={d["case_id"]:d for d in r["detail"]}
        c=sorted(set(A)&set(Bv)); yt=[A[x]["gold"] for x in c]
        pa=[A[x]["mapped"] for x in c]
        pb=[Bv[x]["mapped"] for x in c]
        b,cc,p=mcnemar(yt,pa,pb)
        star="★★" if p<0.01 else ("★" if p<0.05 else "")
        print(f"    {k:18s} 净增益={(cc-b)/len(c):+.4f}  p={p:.3g} {star}")

if __name__=="__main__":
    # 主实验
    raw={k:load(v) for k,v in MAIN.items()}
    pr,n=paired(raw)
    print(f"主实验配对病例数: {n}")
    R1=summarize(pr,"主实验 · 自然分布（n≈346）")
    mcnemar_table(R1,"P1 自由")
    # 均衡
    raw2={k:load(v) for k,v in BAL.items()}
    pr2,n2=paired(raw2)
    print(f"\n均衡子集配对病例数: {n2}")
    R2=summarize(pr2,"均衡子集 · 每类约113例")
    mcnemar_table(R2,"BAL-P1 自由")
