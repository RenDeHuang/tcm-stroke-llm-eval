# -*- coding: utf-8 -*-
"""完整结果矩阵 v2：T1/T2/T3 主矩阵 + BAL + T2B + ABL 舌脉消融。
按 case_id 终态去重；复用 equivalence.py 的权威归一（L1 官方口径）。"""
import json, os, re, sys
from collections import defaultdict, Counter
import numpy as np
B="/Users/huangrende/Desktop/王健"; OUT=f"{B}/out/tcm_stroke_eval"; RUNS=f"{OUT}/runs"
sys.path.insert(0,f"{B}/scripts/tcm_stroke_eval")
from equivalence import norm, to_official

VOCAB=json.load(open(f"{OUT}/top5_vocab.json"))

def load(fn):
    seen={}
    for l in open(f"{RUNS}/{fn}",encoding="utf8"):
        r=json.loads(l); seen[r["case_id"]]=r
    return list(seen.values())

def gold(name, field):
    return {r["case_id"]: r[field] for r in json.load(open(f"{OUT}/{name}"))}
G1=gold("eval_set.json","gold_syndrome")
GB=gold("eval_balanced.json","gold_syndrome")
G2=gold("task2_binary_set.json","label")
G2B=gold("task2_binary_ratio3to1.json","label")
G3=gold("task3_stage_set_v2.json","label")

def L1(pred):
    if pred is None: return None
    return to_official(str(pred), VOCAB)

def multiclass(preds, golds):
    if not golds: return dict(n=0, non_estimable="empty analysis set")
    y=[L1(p) if (L1(p) in VOCAB) else "__none__" for p in preds]
    g=[to_official(gg, VOCAB) if to_official(gg,VOCAB) in VOCAB else "__none__" for gg in golds]
    labels=[v for v in VOCAB]
    K=len(labels); idx={l:i for i,l in enumerate(labels)}
    cm=np.zeros((K,K),dtype=np.int64); acc=0
    for a,b in zip(g,y):
        if a==b: acc+=1
        if b in idx and a in idx: cm[idx[a],idx[b]]+=1
    f=[]
    for i in range(K):
        tp=cm[i,i]; fp=cm[:,i].sum()-tp; fn=cm[i,:].sum()-tp
        p=tp/(tp+fp) if tp+fp else 0.0; r=tp/(tp+fn) if tp+fn else 0.0
        f.append(2*p*r/(p+r) if p+r else 0.0)
    sup=cm.sum(1)
    per=[dict(label=labels[i],support=int(sup[i]),predicted=int(cm[:,i].sum()),
              precision=float(cm[i,i]/cm[:,i].sum()) if cm[:,i].sum() else 0.0,
              recall=float(cm[i,i]/sup[i]) if sup[i] else 0.0,
              f1=round(float(f[i]),4),
              ratio=round(float(cm[:,i].sum()/sup[i]),2) if sup[i] else 0.0) for i in range(K)]
    return dict(acc=acc/len(g), macro=float(np.mean(f)),
                wf=float((np.array(f)*sup).sum()/sup.sum()), per_class=per, cm=cm.tolist())

def to_bin(v):
    if v is None: return None
    s=str(v).strip().lower()
    if s in ("1","1.0","true"): return 1
    if s in ("0","0.0","false"): return 0
    return None

def binary(preds,golds):
    if not golds: return dict(n=0, non_estimable="empty analysis set")
    tp=fp=tn=fn=un=0
    for a,b in zip(golds,preds):
        b=to_bin(b)
        if b is None: un+=1; continue
        if a==1 and b==1: tp+=1
        elif a==0 and b==1: fp+=1
        elif a==0 and b==0: tn+=1
        elif a==1 and b==0: fn+=1
    n=tp+fp+tn+fn
    acc=(tp+tn)/n if n else 0.0
    p=tp/(tp+fp) if tp+fp else 0.0; r=tp/(tp+fn) if tp+fn else 0.0
    f1=2*p*r/(p+r) if p+r else 0.0; spec=tn/(tn+fp) if tn+fp else 0.0
    return dict(n=n,acc=acc,f1=float(f1),rec=float(r),spec=float(spec),unresolved=un)

PAT=re.compile(r'(?P<model>.+?)__(?P<task>ABL|T1|T2|T2B|T3|BAL)_(?P<prompt>P[1-4])_(?P<lang>zh|en)__n(?P<n>\d+)__fendo_tcm_migration_v2__(?P<variant>(?:full|no_tp)__)?[a-f0-9]+\.jsonl$')
conds=defaultdict(list)
for f in os.listdir(RUNS):
    m=PAT.match(f)
    if not m: continue
    d=m.groupdict(); v=(d['variant'] or '').rstrip('_')
    conds[(d['task'],d['model'],d['prompt'],d['lang'],v)].append(f)
EXP={"T1":6906,"BAL":565,"T2":2555,"T2B":392,"T3":6847,"ABL":6906}

results={}
for key,files in conds.items():
    task,model,prompt,lang,variant=key
    if len(files)>1:
        results[key]=dict(status="excluded_duplicate_identity",files=sorted(files)); continue
    recs=load(files[0])
    if task in ("T1","BAL","ABL"):
        GM = GB if task=="BAL" else G1
        preds=[];golds=[]
        for r in recs:
            cid=r["case_id"]
            if cid not in GM: continue
            preds.append(r.get("pred")); golds.append(GM[cid])
        m=multiclass(preds,golds); m["n"]=len(golds)
    elif task in ("T2","T2B"):
        GM=G2 if task=="T2" else G2B
        preds=[];golds=[]
        for r in recs:
            cid=r["case_id"]
            if cid not in GM: continue
            preds.append(r.get("pred")); golds.append(GM[cid])
        m=binary(preds,golds)
    elif task=="T3":
        preds=[];golds=[]
        for r in recs:
            cid=r["case_id"]
            if cid not in G3: continue
            preds.append(r.get("pred")); golds.append(G3[cid])
        m=binary(preds,golds)
    m["status"]="full" if m.get("n")==EXP[task] else ("partial" if variant=="full" else "partial_no_tp")
    m["source_file"]=files[0]
    results[key]=m

outp=f"{OUT}/final_matrix_v2.json"
json.dump({f"{k[0]}|{k[1]}|{k[2]}|{k[3]}|{k[4]}":v for k,v in results.items()},
          open(outp,"w",encoding="utf8"), ensure_ascii=False, indent=1)
print("wrote",outp,"conditions:",len(results))
print(Counter(k[0] for k in results))
