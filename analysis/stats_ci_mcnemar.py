# -*- coding: utf-8 -*-
"""P0 统计：bootstrap 95%CI + McNemar 两两检验（含 Holm 校正）。

纯计算，不调用 API。只对**满全量**条件出结果（未跑完的不进结论）。
用法: python3 stats_ci_mcnemar.py [T1|T2|T3]
"""
import json, glob, os, re, sys, random, itertools
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from equivalence import score_records
from metrics import accuracy, macro_f1, weighted_f1, binary_f1

BASE = "/Users/huangrende/Desktop/王健"; OUT = f"{BASE}/out/tcm_stroke_eval"
VOCAB = json.load(open(f"{OUT}/top5_vocab.json"))
EXPECT = {"T1": 6906, "T2": 2555, "T3": 6851}
random.seed(20260929)

gold_t1 = {r["case_id"]: r["gold_syndrome"] for r in json.load(open(f"{OUT}/eval_set.json"))}
_g3 = f"{OUT}/task3_stage_set_v2.json"
gold_t3 = {r["case_id"]: int(r["label"]) for r in json.load(open(_g3))}
gold_t2 = {r["case_id"]: int(r["label"]) for r in json.load(open(f"{OUT}/task2_binary_set.json"))}

def load(f):
    last, order = {}, []
    for l in open(f):
        l = l.strip()
        if not l: continue
        try: r = json.loads(l)
        except Exception: continue
        cid = r.get("case_id")
        if cid is None: continue
        if cid not in last: order.append(cid)
        last[cid] = r
    return [last[c] for c in order]

def boot(yt, yp, fn, n=2000):
    N = len(yt); vals = []
    for _ in range(n):
        idx = [random.randrange(N) for _ in range(N)]
        vals.append(fn([yt[i] for i in idx], [yp[i] for i in idx]))
    vals.sort()
    return vals[int(0.025*n)], vals[int(0.975*n)-1]

def mcnemar(ya, yb, yt):
    """McNemar 检验（连续性校正卡方 + 对数空间精确二项，避免大 n 溢出）。

    返回 (b, c, p_chi2, p_exact)。p_exact 用 log-sum-exp 计算，
    故 n 很大时也不溢出；n>1000 时精确检验与卡方几乎一致。
    """
    b = sum(1 for a, bb, t in zip(ya, yb, yt) if a == t and bb != t)
    c = sum(1 for a, bb, t in zip(ya, yb, yt) if a != t and bb == t)
    n = b + c
    if n == 0:
        return b, c, 1.0, 1.0
    from math import erfc, sqrt, lgamma, log, exp
    chi2 = (abs(b - c) - 1) ** 2 / n
    p_chi2 = erfc(sqrt(max(chi2, 0.0) / 2))
    # 精确二项双尾：2 * sum_{i=0}^{k} C(n,i) * 0.5^n，在 log 空间累加
    k = min(b, c)
    loghalf_n = -n * log(2.0)
    terms = [lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) + loghalf_n for i in range(0, k + 1)]
    mx = max(terms)
    p_exact = min(1.0, 2.0 * exp(mx + log(sum(exp(t - mx) for t in terms))))
    return b, c, p_chi2, p_exact

def holm(pvals):
    """Holm-Bonferroni：返回校正后的 p（同序）。"""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals); adj = [0.0] * m; prev = 0.0
    for rank, i in enumerate(idx):
        v = min(1.0, pvals[i] * (m - rank)); v = max(v, prev); prev = v; adj[i] = v
    return adj

task_filter = sys.argv[1] if len(sys.argv) > 1 else None
conds = []
for f in sorted(glob.glob(f"{OUT}/runs/*.jsonl")):
    base = os.path.basename(f)
    if "__no_tp__" in base: continue
    model, cond = base.split("__")[0], base.split("__")[1]
    m = re.match(r"(T1|T2|T3)_(P\d)_(zh|en)$", cond)
    if not m: continue
    task, st, lang = m.groups()
    if task_filter and task != task_filter: continue
    recs = load(f)
    gold = gold_t3 if task == "T3" else (gold_t2 if task == "T2" else gold_t1)
    ok = [r for r in recs if not r.get("err") and r.get("pred") is not None and r["case_id"] in gold]
    if len(ok) < EXPECT[task] * 0.999: continue      # 只收满全量
    if task == "T1":
        yt = [gold[r["case_id"]] for r in ok]; yp = [r["pred"] for r in ok]
    else:
        yt = [gold[r["case_id"]] for r in ok]; yp = [int(r["pred"]) for r in ok]
    conds.append((task, model, st, lang, yt, yp))

print(f"满全量条件数 = {len(conds)}")
by_task = defaultdict(list)
for c in conds: by_task[c[0]].append(c)

for task in ("T1", "T2", "T3"):
    sub = by_task.get(task, [])
    if not sub: continue
    print()
    print("=" * 100)
    print(f"Task {task} — bootstrap 95%CI（2000 次重采样）")
    print("=" * 100)
    fn = (lambda a, b: accuracy(a, b)) if task != "T1" else (lambda a, b: accuracy(a, b))
    t1fn = lambda a, b: macro_f1(a, b, sorted(set(a) & set(VOCAB)))
    print(f"{'model':22s}{'st':4s}{'lg':4s}{'n':>6s}{'Acc':>8s}{'95%CI':>18s}" +
          ("{:>10s}{:>18s}".format("MacroF1", "95%CI") if task == "T1" else
           "{:>10s}{:>18s}".format("F1", "95%CI")))
    for (t, model, st, lang, yt, yp) in sorted(sub, key=lambda x: (x[2], x[1], x[3])):
        lo, hi = boot(yt, yp, fn)
        line = f"{model:22s}{st:4s}{lang:4s}{len(yt):>6d}{accuracy(yt,yp):>8.4f}{f'[{lo:.4f},{hi:.4f}]':>18s}"
        if task == "T1":
            lo2, hi2 = boot(yt, yp, t1fn)
            line += f"{macro_f1(yt,yp,sorted(set(yt)&set(VOCAB))):>10.4f}{f'[{lo2:.4f},{hi2:.4f}]':>18s}"
        else:
            lo2, hi2 = boot(yt, yp, binary_f1)
            line += f"{binary_f1(yt,yp):>10.4f}{f'[{lo2:.4f},{hi2:.4f}]':>18s}"
        print(line)

    # McNemar：只比同 task 内的 zh 条件（避免跨语言混比），P1/P2/P3 两两
    print()
    print(f"Task {task} — McNemar 两两比较（同语言内，Holm 校正）")
    for lang in ("zh", "en"):
        grp = [c for c in sub if c[3] == lang]
        if len(grp) < 2: continue
        pairs, ps, labels = [], [], []
        for a, b in itertools.combinations(grp, 2):
            # 需同一批病例：按 case_id 对齐
            cids = None
            ids_a = {r for r in []} 
            # 用位置对齐：同一数据集同顺序 -> 直接比较
            if len(a[4]) != len(b[4]): continue
            bb, cc, p_chi, p_ex = mcnemar(a[5], b[5], a[4])
            pairs.append((f"{a[1]}/{a[2]} vs {b[1]}/{b[2]}", bb, cc, p_ex))
            ps.append(p_ex)
            labels.append(f"{a[1]}/{a[2]} vs {b[1]}/{b[2]}")
        if not ps: continue
        adj = holm(ps)
        print(f"  [{lang}]")
        for (name, bb, cc, p), pa in zip(pairs, adj):
            sig = "*" if pa < 0.05 else " "
            print(f"    {name:52s} b={bb:5d} c={cc:5d} p_exact={p:.3e} p_holm={pa:.3e} {sig}")
