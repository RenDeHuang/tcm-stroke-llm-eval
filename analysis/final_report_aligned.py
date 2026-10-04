# -*- coding: utf-8 -*-
"""生成严格对齐版最终报告：fendo-style P1-P4 + 标准化干预 P5/P6。"""
import glob, json, os, sys
from collections import Counter
B = "/Users/huangrende/Desktop/王健"
OUT = f"{B}/out/tcm_stroke_eval"
RUNS = f"{OUT}/runs"
sys.path.insert(0, f"{B}/scripts/tcm_stroke_eval")
from equivalence import score_records
from metrics import macro_f1, weighted_f1, per_class_f1, bootstrap_ci, mcnemar

VOCAB = json.load(open(f"{OUT}/top5_vocab.json"))

def load_path(path):
    seen = {}
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line); seen[r["case_id"]] = r
        except Exception:
            pass
    return list(seen.values())

def metric(recs):
    s = score_records(recs, VOCAB)
    yt = [d["gold"] for d in s["detail"]]
    yp = [d["mapped"] for d in s["detail"]]
    labels = [x for x in VOCAB if x in set(yt)]
    mf, lo, hi = bootstrap_ci(yt, yp, labels, n=1000, seed=0)
    wf, wlo, whi = bootstrap_ci(yt, yp, labels, fn=weighted_f1, n=1000, seed=0)
    return s, dict(n=len(recs), acc_L0=s["acc_L0"], acc_L1=s["acc_L1"],
                   macro=mf, macro_lo=lo, macro_hi=hi,
                   weighted=wf, weighted_lo=wlo, weighted_hi=whi,
                   per_class=per_class_f1(yt,yp,labels),
                   api_ok=sum(1 for r in recs if not r.get("err")),
                   oov=sum(1 for d in s["detail"] if d["answer"] and d["mapped"] is None))

def one(pattern):
    fs = sorted(glob.glob(pattern))
    if not fs: return None, None
    return load_path(fs[-1]), fs[-1]

def main():
    meta=json.load(open(f"{OUT}/meta.json"))
    aligned={}; paths={}
    for st in ["P1","P2","P3","P4"]:
        aligned[st], paths[st] = one(f"{RUNS}/deepseek-v4-pro__T1_{st}_zh__n400__aligned_v1__*.jsonl")
    if any(v is None for v in aligned.values()):
        missing=[k for k,v in aligned.items() if v is None]
        raise SystemExit(f"严格对齐实验尚未完成：{missing}")
    ok=set.intersection(*[{r['case_id'] for r in v if not r.get('err')} for v in aligned.values()])
    paired={k:[r for r in v if r['case_id'] in ok] for k,v in aligned.items()}
    am={k:metric(v)[1] for k,v in paired.items()}

    # 标准化干预：保留旧的开放式 P1 与 P5/P6 主实验结果。
    main_files={
        "开放式 P1":"deepseek-v4-pro__T1_P1_zh__n400__4449545d.jsonl",
        "P5 闭集":"deepseek-v4-pro__T1_P5_zh__n400__539da249.jsonl",
        "P6 闭集+定义":"deepseek-v4-pro__T1_P6_zh__n400__56f21376.jsonl",
    }
    standard={k:load_path(f"{RUNS}/{v}") for k,v in main_files.items()}
    sm={k:metric(v)[1] for k,v in standard.items()}

    L=["# 中风病 LLM 证型辨识评测 · 严格对齐版最终报告","",
       "## 研究定位","",
       "本研究采用 fendo 风格的真实世界临床文本评测框架，并增加中医证型术语标准化干预。",
       "P1–P4 是严格对齐基线；P5/P6 是额外的标准化干预，不冒充 fendo 原始 Prompt。","",
       "## 一、数据与标签","",
       f"- 病种：中风病（BNG080/081/082）",
       f"- 原始病例池：{meta['pool_raw']} 例；清洗后：{meta['pool_clean']} 例",
       f"- Task 1：{meta['task1']['n']} 例，Top5 证型覆盖 {meta['task1']['n']/meta['pool_clean']:.1%}",
       "- 金标：TCM-SD 数据集的 `norm_syndrome`；输入中剔除显式病名/证型泄漏",
       "- P1–P4 共同使用同一批前 400 例和同一输出协议。","",
       "## 二、严格对齐 P1–P4（配对病例）","",
       f"配对成功病例：**{len(ok)}**；API 失败单独报告，不作为模型错误。","",
       "| 策略 | API成功 | L0严格 | L1归一 | macro-F1 (95% CI) | weighted-F1 (95% CI) | OOV率 |",
       "|---|---:|---:|---:|---:|---:|---:|"]
    for st in ["P1","P2","P3","P4"]:
        x=am[st]
        L.append(f"| {st} | {x['api_ok']}/{x['n']} | {x['acc_L0']:.4f} | {x['acc_L1']:.4f} | {x['macro']:.4f} [{x['macro_lo']:.4f},{x['macro_hi']:.4f}] | {x['weighted']:.4f} [{x['weighted_lo']:.4f},{x['weighted_hi']:.4f}] | {x['oov']/x['n']:.1%} |")
    L += ["", "## 三、标准化干预（原主实验结果）", "",
          "| 条件 | n | L0严格 | L1归一 | macro-F1 (95% CI) | weighted-F1 (95% CI) | OOV率 |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ["开放式 P1","P5 闭集","P6 闭集+定义"]:
        x=sm[name]
        L.append(f"| {name} | {x['n']} | {x['acc_L0']:.4f} | {x['acc_L1']:.4f} | {x['macro']:.4f} [{x['macro_lo']:.4f},{x['macro_hi']:.4f}] | {x['weighted']:.4f} [{x['weighted_lo']:.4f},{x['weighted_hi']:.4f}] | {x['oov']/x['n']:.1%} |")
    L += ["", "## 四、解释边界", "",
          "- P1–P4 用于回答 fendo 风格 Prompt 如何迁移到中医证型任务。",
          "- P5/P6 用于回答标准证型词表和定义能否降低术语/标签空间造成的评测损失。",
          "- P1 严格匹配与官方归一化之间的差异，是命名标准化造成的评分损失；P5 相对开放式 P1 的差异还包含闭集标签空间约束。",
          "- OOV 输出区分为非 Top5 证型、官方未映射术语和空/格式失败，不能混称为模型医学知识错误。", "",
          "## 五、结果文件", ""]
    for st,p in paths.items(): L.append(f"- {st}: `{p}`")
    out=f"{OUT}/REPORT_FINAL_ALIGNED.md"
    open(out,"w",encoding="utf-8").write("\n".join(L)+"\n")
    print("\n".join(L))

if __name__ == "__main__": main()
