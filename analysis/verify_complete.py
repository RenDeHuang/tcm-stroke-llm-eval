# -*- coding: utf-8 -*-
"""通用完成度核验：确认某个 (模型, 任务, 子集, 语言, 策略) 组合在给定样本量下
每个病例都有**非失败**记录。用于断点续跑循环的退出判据，替代人工目测。

判据是精确的：对每个 (strategy, lang) 组合，成功记录覆盖的 case_id 集合
必须等于数据集的完整 case_id 集合，且不存在任何 err 记录残留。
"""
import argparse, glob, json, os, sys

BASE = "/Users/huangrende/Desktop/王健"
OUT = f"{BASE}/out/tcm_stroke_eval"
RUNS = f"{OUT}/runs"

TASK_DATA = {
    "T1": "eval_set.json",
    "T2": "task2_binary_set.json",
    "T2B": "task2_binary_ratio3to1.json",
    "BAL": "eval_balanced.json",
    "T3": "task3_stage_set.json",
    "ABL": "eval_set.json",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=sorted(TASK_DATA))
    ap.add_argument("--subsets", default="", help="逗号分隔；空=全部")
    ap.add_argument("--strategies", default="P1,P2,P3,P4")
    ap.add_argument("--langs", default="zh,en")
    ap.add_argument("--n", type=int, default=0, help="0=全量")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    data = json.load(open(f"{OUT}/{TASK_DATA[a.task]}"))
    if a.n:
        data = data[:a.n]
    gold = {x["case_id"] for x in data}
    n = len(data)

    subsets = [s for s in a.subsets.split(",") if s] or [""]
    missing = []
    for st in [s for s in a.strategies.split(",") if s]:
        for lang in [l for l in a.langs.split(",") if l]:
            for sub in subsets:
                run_tag = f"{a.tag}__{sub}" if (a.tag and sub) else (a.tag or sub)
                pat = f"{RUNS}/{a.model}__{a.task}_{st}_{lang}__n{n}__{run_tag}__*.jsonl"
                files = sorted(glob.glob(pat))
                if not files:
                    missing.append(f"{st}/{lang}/{sub or '-'}: 无文件 ({pat})")
                    continue
                # 断点续跑会向同一文件追加：先前的失败记录与后续成功的记录
                # 共存是正常状态。判据必须以 case_id 的**最终状态**为准，
                # 而不是文件中出现过的失败次数。
                seen_ok, seen_err = set(), set()
                for l in open(files[-1], encoding="utf-8"):
                    try:
                        r = json.loads(l)
                    except Exception:
                        continue
                    if r.get("err"):
                        seen_err.add(r["case_id"])
                    else:
                        seen_ok.add(r["case_id"])
                miss = gold - seen_ok
                if miss:
                    missing.append(
                        f"{st}/{lang}/{sub or '-'}: 成功 {len(seen_ok)}/{n}, 缺 {len(miss)}, "
                        f"最终仍失败 {len(seen_err - seen_ok)}")
    if missing:
        print(f"[INCOMPLETE] {a.model} {a.task}")
        for m in missing:
            print("   -", m)
        sys.exit(1)
    print(f"[COMPLETE] {a.model} {a.task} n={n} subsets={subsets}")
    sys.exit(0)


if __name__ == "__main__":
    main()
