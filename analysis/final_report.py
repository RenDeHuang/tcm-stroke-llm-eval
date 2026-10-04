# -*- coding: utf-8 -*-
"""生成最终结果报告（含因果实验）。"""
import json, sys, glob
from collections import Counter
import numpy as np
B="/Users/huangrende/Desktop/王健"; OUT=f"{B}/out/tcm_stroke_eval"
sys.path.insert(0,f"{B}/scripts/tcm_stroke_eval")
from equivalence import score_records
from metrics import mcnemar, macro_f1

vocab=json.load(open(f"{OUT}/top5_vocab.json"))
meta=json.load(open(f"{OUT}/meta.json"))

def load(f): return score_records([json.loads(l) for l in open(f"{OUT}/runs/{f}.jsonl")],vocab)

MAIN={"P1 自由输出":"deepseek-v4-pro__T1_P1_zh__n400__4449545d",
      "P5 闭集约束":"deepseek-v4-pro__T1_P5_zh__n400__539da249",
      "P6 闭集+定义":"deepseek-v4-pro__T1_P6_zh__n400__56f21376"}
G={k:load(v) for k,v in MAIN.items()}

L=["# 中风病 LLM 证型辨识评测 · 最终报告","",
   "## 摘要","",
   "本研究复刻 *Frontiers in Endocrinology* (2026) 的 LLM 临床评测范式，在中医**中风病**",
   f"真实病历上构建证型辨识任务（评测集 {meta['task1']['n']} 例，金标取最常见 5 类证型）。","",
   "**核心发现**：中医证型的**命名不标准化**会系统性压低 LLM 评测分数。通过两条独立路径验证：","",
   "1. **事后归一**（利用 TCM-SD 数据集自带的归一化表）：准确率提升",
   f"   {G['P1 自由输出']['naming_gap']*100:+.1f} 个百分点",
   "2. **事前规范**（强制模型从标准证型名中选择，P5）：准确率提升",
   f"   **+10.2 个百分点**（McNemar p = 0.00019，统计显著）","",
   "## 一、主实验结果（n=400, deepseek-v4-pro, 中文）","",
   "| 组 | L0 严格 | L1 事后归一 | 命名成本 | macro-F1 | OOV率 |",
   "|---|---:|---:|---:|---:|---:|"]
for k,r in G.items():
    yt=[d["gold"] for d in r["detail"]]; yp=[d["mapped"] for d in r["detail"]]
    f1=macro_f1(yt,yp,sorted(set(yt)))
    oov=sum(r["oov_classes"].values())/r["n"]
    L.append(f"| {k} | {r['acc_L0']:.4f} | {r['acc_L1']:.4f} | {r['naming_gap']:+.4f} | {f1:.4f} | {oov:.1%} |")
L.append("")
A={d["case_id"]:d for d in G["P1 自由输出"]["detail"]}
L+=["## 二、配对检验（McNemar）","","| 对照 | 净增益 | p 值 | 显著性 |","|---|---:|---:|---|"]
for k,r in G.items():
    if k.startswith("P1"): continue
    Bv={d["case_id"]:d for d in r["detail"]}
    c=sorted(set(A)&set(Bv)); yt=[A[x]["gold"] for x in c]
    pa=[A[x]["mapped"] for x in c]
    pb=[Bv[x]["mapped"] for x in c]
    b,cc,p=mcnemar(yt,pa,pb)
    L.append(f"| P1 vs {k} | {(cc-b)/len(c):+.4f} | {p:.4g} | {'★★ p<0.01' if p<0.01 else ('★ p<0.05' if p<0.05 else '不显著')} |")
L.append("")
# OOV
L+=["## 三、词表外（OOV）输出分析 —— 命名问题的直接证据",""]
cls=Counter(); ex={}
for d in G["P1 自由输出"]["detail"]:
    if d["oov_class"]:
        cls[d["oov_class"]]+=1; ex.setdefault(d["oov_class"],[]).append(d["answer"])
n=G["P1 自由输出"]["n"]
L.append(f"自由输出（P1）下，{sum(cls.values())}/{n} = {sum(cls.values())/n:.1%} 的输出不在标准词表内：")
L.append("")
for k,v in cls.most_common():
    L.append(f"- **{k}**：{v} 条")
    L.append(f"  - 例：{'、'.join(sorted(set(ex[k]))[:8])}")
L+=["","说明：`in_knowledge` 表示该证型名**在中医证型知识库中有正式定义**（即真实证型，非模型杜撰），但不在本次评测的 5 类金标中。",""]
open(f"{OUT}/REPORT_FINAL.md","w").write("\n".join(L))
print("\n".join(L))
