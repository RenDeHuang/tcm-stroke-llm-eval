# -*- coding: utf-8 -*-
"""生成 Markdown 结果报告。"""
import json, os, sys, glob
from collections import defaultdict, Counter
import numpy as np
B="/Users/huangrende/Desktop/王健"; OUT=f"{B}/out/tcm_stroke_eval"; RUNS=f"{OUT}/runs"
sys.path.insert(0,f"{B}/scripts/tcm_stroke_eval")
from equivalence import score_records, extract_answer, official_map

def main():
    meta=json.load(open(f"{OUT}/meta.json"))
    vocab=json.load(open(f"{OUT}/top5_vocab.json"))
    data=json.load(open(f"{OUT}/eval_set.json"))
    L=["# 中风病 LLM 证型辨识评测 · 结果报告","",
       "## 一、数据与设计","",
       f"- 病种：中风病（BNG080 中风病 + BNG081 中经络 + BNG082 中脏腑，共 {meta['pool_raw']} 例）",
       f"- 清洗后池：{meta['pool_clean']} 例（剔除泄漏 277、重复 930）",
       f"- Task1 评测集：**{meta['task1']['n']} 例**，金标取最常见 5 类证型",
       f"- Task2：{meta['task2']['n']} 例（中经络 vs 中脏腑，规则型二分类）",
       f"- Task3：{meta['task3']['n']} 例（舌脉消融）","",
       "### Task1 类别分布","","| 证型 | 例数 | 占比 |","|---|---:|---:|"]
    tot=sum(meta['task1']['dist'].values())
    for k,v in sorted(meta['task1']['dist'].items(),key=lambda t:-t[1]):
        L.append(f"| {k} | {v} | {v/tot:.1%} |")
    L.append("")
    # 结果
    runs=defaultdict(list)
    for f in glob.glob(f"{RUNS}/*.jsonl"):
        recs=[json.loads(l) for l in open(f)]
        if not recs: continue
        r0=recs[0]
        if r0["task"]!="T1": continue
        seen={}
        for r in recs: seen[r["case_id"]]=r
        runs[(r0["model"],r0["strategy"],r0["lang"])]=list(seen.values())
    L+=["## 二、三档口径结果（Task1）","",
        "| 模型 | 策略 | 语言 | n | L0 严格 | L1 官方归一 | 命名成本 |",
        "|---|---|---|---:|---:|---:|---:|"]
    rows=[]
    for (m,st,l),recs in sorted(runs.items()):
        r=score_records(recs,vocab)
        rows.append((m,st,l,r["n"],r["acc_L0"],r["acc_L1"],r["naming_gap"]))
        L.append(f"| {m} | {st} | {l} | {r['n']} | {r['acc_L0']:.4f} | {r['acc_L1']:.4f} | {r['naming_gap']:+.4f} |")
    L.append("")
    if rows:
        L.append(f"**平均**：L0 = {np.mean([r[4] for r in rows]):.4f}，L1 = {np.mean([r[5] for r in rows]):.4f}，"
                 f"命名成本 = {np.mean([r[6] for r in rows]):+.4f}")
        L.append("")
    # 词表外分析
    L+=["## 三、词表外（OOV）输出分析","",
        "模型给出的、不在评测词表内的证型归类：",""]
    allcls=Counter(); samples=defaultdict(list)
    for (m,st,l),recs in runs.items():
        r=score_records(recs,vocab)
        for k,v in r["oov_classes"].items(): allcls[k]+=v
        for d in r["detail"]:
            if d["oov_class"] and len(samples[d["oov_class"]])<8:
                samples[d["oov_class"]].append(d["answer"])
    for k,v in allcls.most_common():
        L.append(f"- **{k}**: {v} 条  例：{', '.join(sorted(set(samples[k]))[:6])}")
    L.append("")
    open(f"{OUT}/REPORT.md","w").write("\n".join(L))
    print(f"-> {OUT}/REPORT.md")
    print("\n".join(L[:40]))

if __name__=="__main__": main()
