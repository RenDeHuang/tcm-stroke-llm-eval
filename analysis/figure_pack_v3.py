# -*- coding: utf-8 -*-
"""中风病 LLM 证型评测 · 图件包 v3（对齐 Yang Figure 目录 + 我们的独特贡献）
数据源：out/tcm_stroke_eval/final_matrix_v2.json（129 条件，按 case_id 终态）
规范：MAS medical-figure-design —— 图是证据面；不推断超出数据的主张。
"""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

B="/Users/huangrende/Desktop/王健"; OUT=f"{B}/out/tcm_stroke_eval"
FIG=f"{OUT}/manuscript_v3/figures"; os.makedirs(FIG,exist_ok=True)
for c in ["/System/Library/Fonts/PingFang.ttc","/System/Library/Fonts/Hiragino Sans GB.ttc"]:
    if os.path.exists(c):
        font_manager.fontManager.addfont(c)
        plt.rcParams["font.family"]=font_manager.FontProperties(fname=c).get_name(); break
plt.rcParams["axes.unicode_minus"]=False
plt.rcParams.update({"font.size":9,"axes.titlesize":10,"figure.dpi":200})

D=json.load(open(f"{OUT}/final_matrix_v2.json"))
def sel(task):
    out=[]
    for k,v in D.items():
        t,m,p,l,var=k.split("|")
        if t==task and v.get("status")=="full" and "acc" in v: out.append((m,p,l,var,v))
    return out
MODELS=["deepseek-v4-flash","glm-5.2","openpangu-2.0-flash","qwen3-32b"]
SHORT={"deepseek-v4-flash":"DS-flash","glm-5.2":"GLM-5.2","openpangu-2.0-flash":"Pangu-2.0","qwen3-32b":"Qwen3-32B"}
C={"zh":"#2E6DA4","en":"#D9534F"}

# ---------- Fig 2: Task1 Weighted-F1 分布（模型 × Prompt × 语言） ----------
r=sel("T1")
fig,ax=plt.subplots(figsize=(9,4.2))
xs=[];labels=[]
i=0
for m in MODELS:
    for pm in ["P1","P2","P3","P4"]:
        for lg in ["zh","en"]:
            sub=[v["wf"] for (mm,pp,ll,vv,v) in r if mm==m and pp==pm and ll==lg]
            if not sub: continue
            ax.bar(i,sub[0],0.8,color=C[lg],edgecolor="white",linewidth=.5)
            xs.append(i); labels.append(f"{pm}\n{lg}"); i+=1
    i+=0.6
ax.set_xticks(xs); ax.set_xticklabels(labels,fontsize=6.5)
ax.set_ylabel("Weighted-F1"); ax.set_ylim(0,0.8)
ax.set_title("图2  Task1 证型五分类：Weighted-F1（模型 × Prompt × 语言，满全量条件）",fontsize=9.5)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C["zh"],label="中文"),Patch(color=C["en"],label="英文")],fontsize=8,loc="upper right")
ax.grid(axis="y",alpha=.25)
import matplotlib.transforms as mtrans
for j,m in enumerate(MODELS):
    ax.text(j*4.6+ (j*0.6) +1.5, 0.755, SHORT[m], ha="center", fontsize=8, fontweight="bold")
ax.axhline(0.3123,color="gray",ls="--",lw=.8)
ax.text(len(xs)-0.5,0.317,"多数类基线 0.3123",ha="right",fontsize=7,color="gray")
fig.tight_layout(); fig.savefig(f"{FIG}/fig2_task1_weighted_f1.png"); plt.close(fig)
print("fig2 ok")

# ---------- Fig 3: 最优条件混淆矩阵 ----------
best=max(r,key=lambda t:t[4]["macro"])
cm=np.array(best[4]["cm"]); labs=json.load(open(f"{OUT}/top5_vocab.json"))
fig,ax=plt.subplots(figsize=(5.2,4.4))
cmn=cm/cm.sum(axis=1,keepdims=True)
im=ax.imshow(cmn,cmap="Blues",vmin=0,vmax=1)
for i in range(len(labs)):
    for j in range(len(labs)):
        ax.text(j,i,f"{cmn[i,j]:.2f}",ha="center",va="center",fontsize=7,
                color="white" if cmn[i,j]>0.5 else "black")
ax.set_xticks(range(len(labs))); ax.set_xticklabels([l.replace("证","") for l in labs],rotation=45,ha="right",fontsize=8)
ax.set_yticks(range(len(labs))); ax.set_yticklabels([l.replace("证","") for l in labs],fontsize=8)
ax.set_xlabel("预测"); ax.set_ylabel("金标")
ax.set_title(f"图3  Task1 混淆矩阵（行归一）\n{best[0]} / {best[1]} / {best[2]}，Macro-F1={best[4]['macro']:.4f}",fontsize=9)
fig.colorbar(im,fraction=.046,pad=.04)
fig.tight_layout(); fig.savefig(f"{FIG}/fig3_task1_confusion.png"); plt.close(fig)
print("fig3 ok")

# ---------- Fig 4: 三任务 Prompt 阶梯 ----------
fig,axes=plt.subplots(1,3,figsize=(10,3.4))
for ax,(task,mk,ylab) in zip(axes,[("T1","macro","Macro-F1"),("T2","f1","F1"),("T3","f1","F1")]):
    rr=sel(task); means=[];ns=[]
    for pm in ["P1","P2","P3","P4"]:
        v=[x[4][mk] for x in rr if x[1]==pm]
        means.append(np.mean(v) if v else np.nan); ns.append(len(v))
    ax.plot(["P1","P2","P3","P4"],means,"o-",color="#2E6DA4",lw=2,ms=6)
    for i,(v,n) in enumerate(zip(means,ns)):
        if not np.isnan(v): ax.annotate(f"{v:.3f}\n(n={n})",(i,v),textcoords="offset points",xytext=(0,8),ha="center",fontsize=7)
    ax.set_title(f"{task}",fontsize=10); ax.set_ylabel(ylab); ax.grid(alpha=.25)
    ax.set_ylim(min([m for m in means if not np.isnan(m)])-0.06, max([m for m in means if not np.isnan(m)])+0.09)
fig.suptitle("图4  三任务的 Prompt 阶梯（满全量条件均值）",fontsize=10)
fig.tight_layout(); fig.savefig(f"{FIG}/fig4_prompt_ladder.png"); plt.close(fig)
print("fig4 ok")

# ---------- Fig 5: 舌脉消融（我们的独特贡献；Yang 未做） ----------
abl={};full={}
for k,v in D.items():
    t,m,p,l,var=k.split("|")
    if t=="ABL" and var=="no_tp" and v.get("status")=="full": abl[(m,p,l)]=v
    if t=="T1" and v.get("status")=="full" and "macro" in v: full[(m,p,l)]=v
pairs=[(k,full[k]) for k in sorted(abl) if k in full]
fig,ax=plt.subplots(figsize=(6.4,4.0))
labels=[f"{SHORT[k[0]]}\n{k[1]}-{k[2]}" for k,_ in pairs]
no=[v["macro"] for _,v in pairs]
fu=[f["macro"] for _,f in pairs]
x=np.arange(len(pairs)); w=0.38
ax.bar(x-w/2,no,w,label="剥离舌脉 (no_tp)",color="#D9534F")
ax.bar(x+w/2,fu,w,label="完整文本 (full)",color="#5CB85C")
for i,(a,b) in enumerate(zip(no,fu)):
    ax.annotate(f"{b-a:+.3f}",(i,(a+b)/2),ha="center",fontsize=7,color="#333",
                xytext=(0,0),textcoords="offset points")
    ax.text(i,max(a,b)+0.012,f"{b-a:+.3f}",ha="center",fontsize=6.5,color="#B03A2E")
ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=7)
ax.set_ylabel("Macro-F1"); ax.set_ylim(0,0.5)
ax.set_title("图5  舌脉消融配对比较（Task1）\n配对 Δ 均值 −0.0928，9/9 为负",fontsize=9.5)
ax.legend(fontsize=8); ax.grid(axis="y",alpha=.25)
fig.tight_layout(); fig.savefig(f"{FIG}/fig5_tongue_pulse_ablation.png"); plt.close(fig)
print("fig5 ok")

# ---------- Fig 6: 长尾过度预测 ----------
pc={c["label"]:c for c in best[4]["per_class"]}
labs2=list(labs)
fig,ax=plt.subplots(figsize=(6.6,3.8))
ratios=[pc[l]["ratio"] for l in labs2]
cols=["#B03A2E" if r>1.4 else "#5B9BD5" for r in ratios]
ax.bar(range(len(labs2)),ratios,0.6,color=cols)
ax.axhline(1.0,color="black",ls="--",lw=1)
ax.text(len(labs2)-0.4,1.03,"1.0 = 无系统性偏差",ha="right",fontsize=7)
for i,(l,r) in enumerate(zip(labs2,ratios)):
    ax.text(i,r+0.06,f"{r:.2f}×\nP={pc[l]['precision']:.3f}",ha="center",fontsize=7)
ax.set_xticks(range(len(labs2))); ax.set_xticklabels([l.replace("证","") for l in labs2],fontsize=8.5)
ax.set_ylabel("预测数 / 金标数"); ax.set_ylim(0,max(ratios)+0.7)
ax.set_title(f"图6  Task1 逐类预测倍率（{best[0]} / {best[1]} / {best[2]}）\n两个长尾证型被系统性过度预测",fontsize=9.5)
ax.grid(axis="y",alpha=.25)
fig.tight_layout(); fig.savefig(f"{FIG}/fig6_tail_overprediction.png"); plt.close(fig)
print("fig6 ok")

# ---------- Fig 7: 中英对比 ----------
fig,ax=plt.subplots(figsize=(6.2,3.6))
tasks=["T1","T2","T3","T2B"]; mk={"T1":"macro","T2":"f1","T3":"f1","T2B":"f1"}
x=np.arange(len(tasks)); w=0.35
zv=[];ev=[]
for t in tasks:
    rr=[x for x in sel(t)]
    z=[x[4][mk[t]] for x in rr if x[2]=="zh"]; e=[x[4][mk[t]] for x in rr if x[2]=="en"]
    zv.append(np.mean(z) if z else np.nan); ev.append(np.mean(e) if e else np.nan)
ax.bar(x-w/2,zv,w,label="中文",color=C["zh"]); ax.bar(x+w/2,ev,w,label="英文",color=C["en"])
for i,(a,b) in enumerate(zip(zv,ev)):
    ax.text(i-w/2,a+.008,f"{a:.3f}",ha="center",fontsize=7); ax.text(i+w/2,b+.008,f"{b:.3f}",ha="center",fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([f"{t}\n({'Macro' if t=='T1' else 'F1'})" for t in tasks],fontsize=8)
ax.set_ylabel("指标均值"); ax.legend(fontsize=8); ax.grid(axis="y",alpha=.25)
ax.set_title("图7  中英文条件对比（满全量均值）\n中文在三个主任务上均不低于英文",fontsize=9.5)
fig.tight_layout(); fig.savefig(f"{FIG}/fig7_language.png"); plt.close(fig)
print("fig7 ok")

# ---------- Fig 1: pipeline ----------
fig,ax=plt.subplots(figsize=(9,2.6)); ax.axis("off")
boxes=[("TCM-SD\n中风病记录",0.02),("清洗\n去泄漏/去重\n8696→7489",0.20),
       ("三任务构造\nT1 五分类 6906\nT2 编码 2555\nT3 分期 v2 6847",0.40),
       ("4 模型 × 2 语言\n× 4 Prompt\n× 3 任务",0.62),
       ("三层裁决\n格式/规则/裁决",0.80),("指标 + 95%CI\n+ McNemar",0.94)]
for txt,x in boxes:
    ax.add_patch(plt.Rectangle((x,0.3),0.14,0.4,facecolor="#EAF1F8",edgecolor="#2E6DA4",lw=1.2))
    ax.text(x+0.07,0.5,txt,ha="center",va="center",fontsize=7)
for i in range(len(boxes)-1):
    x0=boxes[i][1]+0.14; x1=boxes[i+1][1]
    ax.annotate("",xy=(x1,0.5),xytext=(x0,0.5),arrowprops=dict(arrowstyle="->",color="#2E6DA4",lw=1.2))
ax.set_xlim(0,1.1); ax.set_ylim(0,1)
ax.set_title("图1  评测流程（Yang 框架迁移版）",fontsize=10)
fig.tight_layout(); fig.savefig(f"{FIG}/fig1_pipeline.png"); plt.close(fig)
print("fig1 ok")
print("\n全部图件：", sorted(os.listdir(FIG)))
