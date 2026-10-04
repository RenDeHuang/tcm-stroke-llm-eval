# -*- coding: utf-8 -*-
"""Task2（中经络/中脏腑）与 Task3（舌脉消融）分析。"""
import json, glob, sys
from collections import defaultdict, Counter
import numpy as np
B="/Users/huangrende/Desktop/王健"; OUT=f"{B}/out/tcm_stroke_eval"; RUNS=f"{OUT}/runs"
sys.path.insert(0,f"{B}/scripts/tcm_stroke_eval")
from equivalence import extract_answer, norm
from metrics import macro_f1, accuracy, bootstrap_ci

def load(task):
    d=defaultdict(list)
    for f in glob.glob(f"{RUNS}/*.jsonl"):
        recs=[json.loads(l) for l in open(f)]
        if not recs or recs[0]["task"]!=task: continue
        r0=recs[0]
        seen={}
        for r in recs: seen[r["case_id"]]=r
        d[(r0["model"],r0["strategy"],r0["lang"],r0.get("subset",""))]=list(seen.values())
    return d

def main():
    L=["# Task2 / Task3 结果",""]
    # ---- Task2 ----
    t2=load("T2")
    if t2:
        L+=["## Task2：中经络 vs 中脏腑（规则型二分类）","",
            "| 模型 | 策略 | 语言 | n | Acc | macro-F1 | 阳性类召回 |","|---|---|---|---:|---:|---:|---:|"]
        for (m,st,l,sub),recs in sorted(t2.items()):
            yt=[r["gold_syndrome"] for r in recs]
            yp=[]
            for r in recs:
                a=norm(extract_answer(r.get("raw")))
                if "中经络" in a: yp.append("中经络")
                elif "中脏腑" in a: yp.append("中脏腑")
                else: yp.append(None)
            f1=macro_f1(yt,yp,["中经络","中脏腑"]); acc=accuracy(yt,yp)
            # 阳性（中脏腑）召回
            pos=sum(1 for t,p in zip(yt,yp) if t=="中脏腑" and p=="中脏腑")
            posn=sum(1 for t in yt if t=="中脏腑")
            L.append(f"| {m} | {st} | {l} | {len(recs)} | {acc:.4f} | {f1:.4f} | {pos/max(posn,1):.3f} |")
        L.append("")
    else:
        L.append("## Task2：尚无数据（未跑）\n")
    # ---- Task3 ----
    t3=load("T3")
    if t3:
        L+=["## Task3：舌脉消融（影响幅度）","",
            "| 模型 | 策略 | 语言 | n | 完整 Acc | 去舌脉 Acc | ΔAcc |","|---|---|---|---:|---:|---:|---:|"]
        # 按 (model,strategy,lang) 聚合 full/no_tp
        agg=defaultdict(dict)
        for (m,st,l,sub),recs in t3.items():
            yt=[r["gold_syndrome"] for r in recs]
            yp=[]
            for r in recs:
                a=norm(extract_answer(r.get("raw")))
                hit=[v for v in json.load(open(f"{OUT}/top5_vocab.json")) if norm(v) in a]
                yp.append(max(hit,key=len) if hit else None)
            agg[(m,st,l)][sub]=(accuracy(yt,yp), len(recs))
        for k,v in sorted(agg.items()):
            if "full" in v and "no_tp" in v:
                L.append(f"| {k[0]} | {k[1]} | {k[2]} | {v['full'][1]} | {v['full'][0]:.4f} | {v['no_tp'][0]:.4f} | {v['no_tp'][0]-v['full'][0]:+.4f} |")
        L.append("")
    else:
        L.append("## Task3：尚无数据（未跑）\n")
    open(f"{OUT}/REPORT_t2t3.md","w").write("\n".join(L))
    print("\n".join(L))

if __name__=="__main__": main()
