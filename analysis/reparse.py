# -*- coding: utf-8 -*-
"""离线重解析：不重跑 API，用最新解析器重算所有 runs/*.jsonl 的 pred/tier。

说明：raw 输出已落盘，解析规则改进后无需重新调用模型。
"""
import json, glob, os, sys
BASE="/Users/huangrende/Desktop/王健"; OUT=f"{BASE}/out/tcm_stroke_eval"; RUNS=f"{OUT}/runs"
sys.path.insert(0,f"{BASE}/scripts/tcm_stroke_eval")
from runner import parse_t1, parse_t2
vocab=json.load(open(f"{OUT}/top5_vocab.json"))

for f in sorted(glob.glob(f"{RUNS}/*.jsonl")):
    recs=[json.loads(l) for l in open(f)]
    changed=0
    for r in recs:
        old=(r.get("pred"),r.get("tier"))
        raw=r.get("raw","")
        if r["task"]=="T2": pred,tier=parse_t2(raw)
        else: pred,tier=parse_t1(raw,vocab)
        if (pred,tier)!=old: changed+=1
        r["pred"],r["tier"]=pred,tier
        r["_old_pred"]=old[0]
    with open(f,"w") as fh:
        for r in recs: fh.write(json.dumps(r,ensure_ascii=False)+"\n")
    res_before=sum(1 for r in recs if r.get("_old_pred") is not None)
    res_after=sum(1 for r in recs if r["pred"] is not None)
    print(f"{os.path.basename(f)[:60]:60s} n={len(recs):4d} 变化 {changed:4d}  解析 {res_before}->{res_after}")
