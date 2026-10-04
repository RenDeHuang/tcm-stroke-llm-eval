# -*- coding: utf-8 -*-
"""峰值条件的预登记自助区间（复刻 final_matrix_v2 口径，unit=record, n=2000, percentile 95%）。
仅消费冻结 runs/*.jsonl 与金标集；点值必须与 final_matrix_v2.json 逐字一致。"""
import json, os, re, sys, random
from collections import defaultdict
import numpy as np
B="/Users/huangrende/Desktop/王健"; OUT=f"{B}/out/tcm_stroke_eval"; RUNS=f"{OUT}/runs"
sys.path.insert(0,f"{B}/scripts/tcm_stroke_eval")
from equivalence import norm, to_official
VOCAB=json.load(open(f"{OUT}/top5_vocab.json"))
SEED=20261002; N=2000; ALPHA=0.05

def load(fn):
    seen={}
    for l in open(f"{RUNS}/{fn}",encoding="utf8"):
        r=json.loads(l); seen[r["case_id"]]=r
    return list(seen.values())
def gold(name,field):
    return {r["case_id"]:r[field] for r in json.load(open(f"{OUT}/{name}"))}
G1=gold("eval_set.json","gold_syndrome"); G2=gold("task2_binary_set.json","label"); G3=gold("task3_stage_set_v2.json","label")

def L1(p):
    if p is None: return None
    return to_official(str(p),VOCAB)
def multiclass(preds,golds):
    y=[L1(p) if (L1(p) in VOCAB) else "__none__" for p in preds]
    g=[to_official(gg,VOCAB) if to_official(gg,VOCAB) in VOCAB else "__none__" for gg in golds]
    labels=list(VOCAB); idx={l:i for i,l in enumerate(labels)}; K=len(labels)
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
    return dict(acc=acc/len(g), macro=float(np.mean(f)), wf=float((np.array(f)*sup).sum()/sup.sum()))
def to_bin(v):
    if v is None: return None
    s=str(v).strip().lower()
    if s in ("1","1.0","true"): return 1
    if s in ("0","0.0","false"): return 0
    return None
def binary(preds,golds):
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
    f1=2*p*r/(p+r) if p+r else 0.0
    return dict(n=n,acc=acc,f1=float(f1),rec=float(r),spec=(tn/(tn+fp) if tn+fp else 0.0))

def build(task,fn):
    recs=load(fn); GM=G1 if task=="T1" else (G2 if task=="T2" else G3)
    preds=[];golds=[]
    for r in recs:
        cid=r["case_id"]
        if cid not in GM: continue
        preds.append(r.get("pred")); golds.append(GM[cid])
    met=(lambda s,t: multiclass(s,t)) if task=="T1" else (lambda s,t: binary(s,t))
    return preds,golds,met
def boot(preds,golds,met,field):
    rng=random.Random(SEED); k=len(preds); vals=[]
    for _ in range(N):
        idx=[rng.randrange(k) for _ in range(k)]
        vals.append(met([preds[i] for i in idx],[golds[i] for i in idx])[field])
    vals.sort()
    return vals[int(ALPHA/2*N)], vals[int((1-ALPHA/2)*N)-1]

# peak conditions (from final_matrix_v2)
targets={
 "T1|deepseek-v4-flash|P2|zh":["T1","deepseek-v4-flash__T1_P2_zh__n6906__fendo_tcm_migration_v2__4046a5f8.jsonl",["macro","acc"]],
 "T2|qwen3-32b|P3|zh":["T2","qwen3-32b__T2_P3_zh__n2555__fendo_tcm_migration_v2__977a6306.jsonl",["f1","acc"]],
 "T3|openpangu-2.0-flash|P3|zh":["T3","openpangu-2.0-flash__T3_P3_zh__n6851__fendo_tcm_migration_v2__full__2a3fcef3.jsonl",["f1","acc"]],
}
# resolve T3 filename if hash differs
for key,val in list(targets.items()):
    task,fn,fields=val
    if not os.path.exists(f"{RUNS}/{fn}"):
        cands=[x for x in os.listdir(RUNS) if re.match(re.escape(task)+r"|", "") or ("__"+task+"_" in x and x.startswith(fn.split('__')[0]) and f"_{fields and ''}"=="")]
        import glob
        pat=f"{RUNS}/{fn.split('__')[0]}__{task}_P{fn.split('_P')[1].split('_')[0]}_zh__*.jsonl"
        got=glob.glob(pat)
        if got: targets[key]=[task,os.path.basename(got[0]),fields]
        else: print("MISS",key,fn)

out={"surface_kind":"peak_condition_bootstrap_ci","analysis_design":"unit=record, percentile 95%, n_resamples=2000, seed="+str(SEED),
     "authoritative_matrix":"out/tcm_stroke_eval/final_matrix_v2.json","note":"点值与 final_matrix_v2 同口径；区间为预登记 unit=record 自助（不做聚簇修正）。","conditions":{}}
for key,(task,fn,fields) in targets.items():
    preds,golds,met=build(task,fn)
    res={"task":task,"source_file":fn,"n_records":len(preds)}
    for fld in fields:
        pt=met(preds,golds)[fld]
        lo,hi=boot(preds,golds,met,fld)
        res[fld]={"point":round(pt,4),"ci95":[round(lo,4),round(hi,4)]}
    out["conditions"][key]=res
    print(key,json.dumps(res,ensure_ascii=False))
json.dump(out,open(f"{OUT}/peak_ci_v1.json","w",encoding="utf8"),ensure_ascii=False,indent=1)
print("saved peak_ci_v1.json")
