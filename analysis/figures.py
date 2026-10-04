# -*- coding: utf-8 -*-
"""图表：① 三档口径对比柱状 ② 命名偏差成本 ③ 策略×语言热力图 ④ 混淆矩阵。"""
import json, os, glob, sys
from collections import defaultdict, Counter
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
B="/Users/huangrende/Desktop/王健"; OUT=f"{B}/out/tcm_stroke_eval"; RUNS=f"{OUT}/runs"
sys.path.insert(0,f"{B}/scripts/tcm_stroke_eval")
from equivalence import score_records, official_map, norm
from metrics import macro_f1, accuracy

for cand in ["/System/Library/Fonts/PingFang.ttc","/System/Library/Fonts/Hiragino Sans GB.ttc"]:
    if os.path.exists(cand):
        try:
            font_manager.fontManager.addfont(cand)
            plt.rcParams["font.family"]=font_manager.FontProperties(fname=cand).get_name(); break
        except Exception: pass
plt.rcParams["axes.unicode_minus"]=False

def load_runs():
    d=defaultdict(list)
    for f in glob.glob(f"{RUNS}/*.jsonl"):
        recs=[json.loads(l) for l in open(f)]
        if not recs: continue
        r0=recs[0]
        if r0["task"]!="T1": continue
        # 去重
        seen={}
        for r in recs: seen[r["case_id"]]=r
        d[(r0["model"],r0["strategy"],r0["lang"])]=list(seen.values())
    return d

def main():
    runs=load_runs(); vocab=json.load(open(f"{OUT}/top5_vocab.json"))
    if not runs: print("无数据"); return
    rows=[]
    for (m,st,l),recs in runs.items():
        r=score_records(recs,vocab)
        rows.append(dict(model=m,strategy=st,lang=l,n=r["n"],
                         L0=r["acc_L0"],L1=r["acc_L1"],gap=r["naming_gap"]))
    json.dump(rows,open(f"{OUT}/scores_all.json","w"),ensure_ascii=False,indent=1)
    print("已写 scores_all.json，共",len(rows),"行")

    # Fig1: 口径对比（按模型×语言）
    models=sorted(set(r["model"] for r in rows))
    fig,ax=plt.subplots(figsize=(max(7,2.2*len(models)),4.6))
    labels=[];L0s=[];L1s=[]
    for m in models:
        for l in ["zh","en"]:
            sub=[r for r in rows if r["model"]==m and r["lang"]==l]
            if not sub: continue
            labels.append(f"{m.split('-')[0]}\n{l}")
            L0s.append(np.mean([r["L0"] for r in sub]))
            L1s.append(np.mean([r["L1"] for r in sub]))
    x=np.arange(len(labels)); w=0.38
    ax.bar(x-w/2,L0s,w,label="L0 严格字面",color="#d9534f")
    ax.bar(x+w/2,L1s,w,label="L1 官方归一",color="#5cb85c")
    for i,(a,b) in enumerate(zip(L0s,L1s)):
        ax.text(i-w/2,a+.01,f"{a:.3f}",ha="center",fontsize=8)
        ax.text(i+w/2,b+.01,f"{b:.3f}",ha="center",fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=8); ax.set_ylabel("准确率")
    ax.set_title("命名规范化前后的准确率（L0 vs L1）"); ax.legend(); ax.grid(axis="y",alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_calibration.png",dpi=160); print("-> fig1")

    # Fig2: 策略对比（含 P5/P6）
    fig,ax=plt.subplots(figsize=(8,4.6))
    sts=sorted(set(r["strategy"] for r in rows))
    mk={}; 
    for i,st in enumerate(sts):
        v=[r["L0"] for r in rows if r["strategy"]==st]
        ax.bar(i,np.mean(v) if v else 0,0.6,label=st)
        ax.text(i,(np.mean(v) if v else 0)+.005,f"{np.mean(v):.3f}" if v else "—",ha="center",fontsize=8)
    ax.set_xticks(range(len(sts))); ax.set_xticklabels(sts)
    ax.set_ylabel("L0 准确率"); ax.set_title("各提示策略的严格准确率（P5/P6 = 命名规范化干预）")
    ax.grid(axis="y",alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_strategy.png",dpi=160); print("-> fig2")

    # Fig3: 命名偏差成本
    fig,ax=plt.subplots(figsize=(7,4.2))
    gaps=[(r["model"].split("-")[0]+"/"+r["strategy"]+"/"+r["lang"], r["gap"]) for r in rows]
    gaps.sort(key=lambda t:-t[1])
    ax.barh([g[0] for g in gaps][:20],[g[1] for g in gaps][:20],color="#f0ad4e")
    ax.set_xlabel("命名偏差成本（L1-L0）"); ax.set_title("术语不规范造成的低估幅度")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_naming_gap.png",dpi=160); print("-> fig3")

if __name__=="__main__": main()
