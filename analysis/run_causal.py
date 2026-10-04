# -*- coding: utf-8 -*-
"""跑 A/B/C/D 四组对照，并做配对统计检验。"""
import json, os, sys, argparse
from collections import defaultdict
B="/Users/huangrende/Desktop/王健"; sys.path.insert(0,f"{B}/scripts/tcm_stroke_eval")
OUT=f"{B}/out/tcm_stroke_eval"; RUNS=f"{OUT}/runs"
from equivalence import score_records, to_official, norm
from metrics import mcnemar

GROUPS={"A_free":"P1","B_closed":"P5","C_closed_def":"P6","D_cot":"P4"}

def load_group(model, strategy, lang, n):
    # 找到匹配的 run 文件
    import glob, re as _re
    vocab=json.load(open(f"{OUT}/top5_vocab.json"))
    best=None
    for f in glob.glob(f"{RUNS}/*.jsonl"):
        recs=[json.loads(l) for l in open(f)]
        if not recs: continue
        r0=recs[0]
        if r0["model"]==model and r0["strategy"]==strategy and r0["lang"]==lang and r0["task"]=="T1":
            best=recs
    return best

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--models",default="deepseek-v4-pro")
    ap.add_argument("--langs",default="zh")
    ap.add_argument("--n",type=int,default=150)
    a=ap.parse_args()
    vocab=json.load(open(f"{OUT}/top5_vocab.json"))

    print("="*78)
    print("因果验证实验：命名规范化对评测分数的影响")
    print("="*78)
    print(f"预注册假设 H1：闭集约束(P5)显著提升严格准确率 → 命名成本真实存在")
    print(f"零假设    H0：P5 相对 P1 无显著提升 → 核心假设被证伪\n")

    for model in a.models.split(","):
        for lang in a.langs.split(","):
            print(f"\n### {model} / {lang}")
            data={}
            for gname,st in GROUPS.items():
                recs=load_group(model,st,lang,a.n)
                if recs: data[gname]=recs[:a.n]
            if not data: print("  （无数据，跑完 runner 后再来）"); continue

            print(f"  {'组':16s}{'n':>5s}{'L0严格':>9s}{'L1官方归一':>11s}{'命名成本':>10s}{'未解析率':>9s}")
            res={}
            for gname,recs in data.items():
                r=score_records(recs,vocab)
                res[gname]=r
                unres=sum(1 for x in r["detail"] if not x["raw"])/r["n"]
                print(f"  {gname:16s}{r['n']:5d}{r['acc_L0']:9.4f}{r['acc_L1']:11.4f}{r['naming_gap']:+10.4f}{unres:9.2%}")

            # 配对检验：A vs B, A vs C
            if "A_free" in data and "B_closed" in data:
                A={d["case_id"]:d for d in res["A_free"]["detail"]}
                Bv={d["case_id"]:d for d in res["B_closed"]["detail"]}
                common=sorted(set(A)&set(Bv))
                yt=[A[c]["gold"] for c in common]
                pa=[A[c]["mapped"] for c in common]
                pb=[Bv[c]["mapped"] for c in common]
                b,cc,p=mcnemar(yt,pa,pb)
                print(f"\n  McNemar  A(自由) vs B(闭集)  n={len(common)}")
                print(f"    A对B错={b}  A错B对={cc}  p={p:.4g}  {'★显著' if p<0.05 else '不显著'}")
                d=(cc-b)/len(common)
                print(f"    闭集约束净增益 = {d:+.4f} ({d*100:+.1f} 个百分点)")
            if "A_free" in data and "C_closed_def" in data:
                A={d["case_id"]:d for d in res["A_free"]["detail"]}
                Cv={d["case_id"]:d for d in res["C_closed_def"]["detail"]}
                common=sorted(set(A)&set(Cv))
                yt=[A[c]["gold"] for c in common]
                pa=[A[c]["mapped"] for c in common]
                pc=[Cv[c]["mapped"] for c in common]
                b,cc,p=mcnemar(yt,pa,pc)
                print(f"\n  McNemar  A(自由) vs C(闭集+定义)  n={len(common)}")
                print(f"    A对C错={b}  A错C对={cc}  p={p:.4g}  {'★显著' if p<0.05 else '不显著'}")
                d=(cc-b)/len(common)
                print(f"    净增益 = {d:+.4f} ({d*100:+.1f} 个百分点)")

            json.dump({g:{"acc_L0":r["acc_L0"],"acc_L1":r["acc_L1"],"naming_gap":r["naming_gap"],"n":r["n"]}
                       for g,r in res.items()},
                      open(f"{OUT}/causal_{model}_{lang}.json","w"),ensure_ascii=False,indent=1)

if __name__=="__main__": main()
