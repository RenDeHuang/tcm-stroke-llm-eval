# -*- coding: utf-8 -*-
"""Yang et al. (Front Endocrinol 2026;17:1747468) 补充图复刻包。

复刻范围（与 FENDO_SIMILARITY.md 对齐）：
  Fig S1  最优模型 8 条件两两 McNemar（Task1）
  Fig S2  4 模型在 zh-P1 下两两 McNemar（Task1）
  Fig S3  平衡数据集各模型 Task1 性能对比
  Fig S4  模型 x Prompt F1 热图
  Fig S5  各模型峰值 F1 + 最优 Prompt 排名
  Fig S6  平衡集上各 Prompt 策略对比
  Fig S7  平衡集中英对比
  Fig S8  Task2 八条件 F1 分布
  Fig S9  Task3 八条件 F1 分布
  Fig S15 Prompt 优化对 Task1 Macro-F1 的 delta (P4 vs P1)
  Fig S16 Prompt 优化对 Task2 F1 的 delta
  Fig S17 Prompt 优化对 Task3 F1 的 delta
不可复刻（无同构数据，需在 limitation 声明）：S10-S14。
数据源：out/tcm_stroke_eval/{runs/*.jsonl, final_matrix_v2.json}；纯本地，不调用 API。
"""
import json, os, glob, re, sys, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

BASE = "/Users/huangrende/Desktop/王健"
OUT = f"{BASE}/out/tcm_stroke_eval"
RUNS = f"{OUT}/runs"
FIG = f"{OUT}/manuscript_v3/figures/supp"; os.makedirs(FIG, exist_ok=True)

sys.path.insert(0, f"{BASE}/scripts/tcm_stroke_eval")
from equivalence import norm, to_official   # noqa: E402

for _c in ["/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc"]:
    if os.path.exists(_c):
        font_manager.fontManager.addfont(_c)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_c).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 200})

MODELS = ["deepseek-v4-flash", "glm-5.2", "openpangu-2.0-flash", "qwen3-32b"]
SHORT = {"deepseek-v4-flash": "DS-flash", "glm-5.2": "GLM-5.2",
         "openpangu-2.0-flash": "Pangu-2.0", "qwen3-32b": "Qwen3-32B"}
C = {"zh": "#2E6DA4", "en": "#D9534F"}
LG = {"zh": "中文", "en": "英文"}

VOCAB = json.load(open(f"{OUT}/top5_vocab.json"))
VSET = [norm(v) for v in VOCAB]
D = json.load(open(f"{OUT}/final_matrix_v2.json"))

GOLD = {
    "T1": {r["case_id"]: norm(r["gold_syndrome"]) for r in json.load(open(f"{OUT}/eval_set.json"))},
    "T2": {r["case_id"]: int(r["label"]) for r in json.load(open(f"{OUT}/task2_binary_set.json"))},
    "T3": {r["case_id"]: int(r["label"]) for r in json.load(open(f"{OUT}/task3_stage_set_v2.json"))},
}
GOLD["BAL"] = GOLD["T1"]



def _lookup(task, m, p, l):
    """T3 的 key 带 |full 变体，其余任务不带。"""
    for k in (f"{task}|{m}|{p}|{l}|", f"{task}|{m}|{p}|{l}|full"):
        v = D.get(k)
        if v is not None:
            return v
    return None

def _read(fn):
    last, order = {}, []
    for line in open(fn, encoding="utf8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        cid = r.get("case_id")
        if cid is None:
            continue
        if cid not in last:
            order.append(cid)
        last[cid] = r
    return [last[c] for c in order]


def _pick(task, model, prompt, lang, variant=None):
    pat = re.compile(
        rf"^{re.escape(model)}__{task}_{prompt}_{lang}__n\d+__fendo_tcm_migration_v2__"
        rf"(?P<var>(?:full|no_tp)__)?[0-9a-f]+\.jsonl$")
    best = None
    for f in os.listdir(RUNS):
        m = pat.match(f)
        if not m:
            continue
        got = (m.group("var") or "").rstrip("_") or ""
        if variant is not None and got != variant:
            continue
        if variant is None and got:
            continue
        p = f"{RUNS}/{f}"
        n = sum(1 for _ in open(p, encoding="utf8"))
        if best is None or n > best[0]:
            best = (n, p)
    return best[1] if best else None


def pairs_for(task, model, prompt, lang, variant=None):
    """返回 (gold_list, pred_list) 按 case_id 对齐，仅保留金标可用项。"""
    p = _pick(task, model, prompt, lang, variant)
    if not p:
        return None
    g = GOLD[task]
    recs = _read(p)
    gold, pred = [], []
    for r in recs:
        cid = r["case_id"]
        if cid not in g:
            continue
        pr = r.get("pred")
        if pr is None:
            continue
        gold.append(g[cid])
        pred.append(norm(pr) if task in ("T1", "BAL") else int(pr))
    return gold, pred


def mcnemar_pair(ya, yb, yt):
    b = sum(1 for a, c, t in zip(ya, yb, yt) if a == t and c != t)
    c = sum(1 for a, cc, t in zip(ya, yb, yt) if a != t and cc == t)
    n = b + c
    if n == 0:
        return b, c, 1.0
    from math import erfc, sqrt, lgamma, log, exp
    chi2 = (abs(b - c) - 1) ** 2 / n
    p_chi = erfc(sqrt(max(chi2, 0.0) / 2))
    k = min(b, c)
    terms = [lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) - n * log(2.0)
             for i in range(k + 1)]
    mx = max(terms)
    p_ex = min(1.0, 2.0 * exp(mx + log(sum(exp(t - mx) for t in terms))))
    return b, c, p_ex


def holm(ps):
    idx = sorted(range(len(ps)), key=lambda i: ps[i])
    m = len(ps); adj = [0.0] * m; prev = 0.0
    for rank, i in enumerate(idx):
        v = min(1.0, ps[i] * (m - rank)); v = max(v, prev); prev = v; adj[i] = v
    return adj


def sig_matrix(entries, labels, pvals, title, fname, thr=0.05):
    """entries: (i,j,p); 画上三角 p 值热图。"""
    k = len(labels)
    M = np.full((k, k), np.nan)
    for (i, j, p) in entries:
        M[i, j] = M[j, i] = p
    fig, ax = plt.subplots(figsize=(max(4.6, k * 0.62), max(4.0, k * 0.56)))
    cmap = plt.cm.RdYlGn.copy(); cmap.set_bad("white")
    im = ax.imshow(np.where(np.isnan(M), np.nan, -np.log10(np.clip(M, 1e-300, 1))),
                   cmap="viridis")
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            p = M[i, j]
            if np.isnan(p):
                continue
            star = "*" if p < thr else ""
            ax.text(j, i, f"{p:.1e}{star}" if p < 0.1 else f"{p:.2f}{star}",
                    ha="center", va="center", fontsize=5.2,
                    color="white" if M[i, j] < 0.05 else "black")
    ax.set_xticks(range(k)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.5)
    ax.set_yticks(range(k)); ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, fraction=0.046, pad=0.04, label="-log10(p_exact)")
    fig.tight_layout(); fig.savefig(f"{FIG}/{fname}"); plt.close(fig)
    print(f"{fname} ok")


def heat(models, prompts, langs, task, metric, title, fname):
    k = len(models); vals = np.full((k, len(prompts) * len(langs)), np.nan)
    cols = []
    for j, p in enumerate(prompts):
        for l in langs:
            cols.append(f"{p}\n{lg}" if False else f"{p}·{LG[l]}")
    for i, m in enumerate(models):
        for j, p in enumerate(prompts):
            for li, l in enumerate(langs):
                key = f"{task}|{m}|{p}|{l}|"
                v = D.get(key)
                if v and v.get("status") == "full":
                    vals[i, j * len(langs) + li] = v.get(metric)
    fig, ax = plt.subplots(figsize=(1.05 * len(cols) + 2.2, 0.62 * k + 1.6))
    im = ax.imshow(vals, cmap="YlGnBu", aspect="auto", vmin=np.nanmin(vals), vmax=np.nanmax(vals))
    for i in range(k):
        for j in range(len(cols)):
            if not np.isnan(vals[i, j]):
                ax.text(j, i, f"{vals[i, j]:.3f}", ha="center", va="center", fontsize=6.4,
                        color="white" if vals[i, j] > np.nanmean(vals) else "black")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=7)
    ax.set_yticks(range(k)); ax.set_yticklabels([SHORT[m] for m in models], fontsize=8)
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, fraction=0.03, pad=0.02)
    fig.tight_layout(); fig.savefig(f"{FIG}/{fname}"); plt.close(fig)
    print(f"{fname} ok")


# ---------------- Fig S4: 模型 x Prompt 热图（T1/T2/T3 三 panel） ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 3.4))
for ax, (task, metric, ttl) in zip(axes, [("T1", "macro", "Task1 Macro-F1"),
                                          ("T2", "f1", "Task2 F1"),
                                          ("T3", "f1", "Task3 F1")]):
    cols = [f"{p}·{LG[l]}" for p in ["P1", "P2", "P3", "P4"] for l in ["zh", "en"]]
    vals = np.full((len(MODELS), len(cols)), np.nan)
    for i, m in enumerate(MODELS):
        for j, (p, l) in enumerate([(p, l) for p in ["P1", "P2", "P3", "P4"] for l in ["zh", "en"]]):
            v = _lookup(task, m, p, l)
            if v and v.get("status") == "full":
                vals[i, j] = v.get(metric)
    im = ax.imshow(vals, cmap="YlGnBu", aspect="auto")
    for i in range(len(MODELS)):
        for j in range(len(cols)):
            if not np.isnan(vals[i, j]):
                ax.text(j, i, f"{vals[i, j]:.3f}", ha="center", va="center", fontsize=5.6,
                        color="white" if vals[i, j] > np.nanmean(vals) else "black")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=6, rotation=45, ha="right")
    ax.set_yticks(range(len(MODELS))); ax.set_yticklabels([SHORT[m] for m in MODELS], fontsize=7)
    ax.set_title(ttl, fontsize=9)
fig.suptitle("Fig S4  模型 x Prompt x 语言 指标热图（满全量条件）", fontsize=10)
fig.tight_layout(); fig.savefig(f"{FIG}/figS4_heatmap.png"); plt.close(fig)
print("figS4 ok")

# ---------------- Fig S5: 各模型峰值 F1 + 最优 Prompt ----------------
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
for ax, (task, metric, ttl) in zip(axes, [("T1", "macro", "Task1 Macro-F1"),
                                          ("T2", "f1", "Task2 F1"),
                                          ("T3", "f1", "Task3 F1")]):
    peaks = []
    for m in MODELS:
        best = None
        for p in ["P1", "P2", "P3", "P4"]:
            for l in ["zh", "en"]:
                v = _lookup(task, m, p, l)
                if v and v.get("status") == "full" and (best is None or v[metric] > best[0]):
                    best = (v[metric], p, l)
        if best:
            peaks.append((m, best))
    peaks.sort(key=lambda x: -x[1][0])
    ax.bar(range(len(peaks)), [x[1][0] for x in peaks],
           color=["#2E6DA4" if x[1][2] == "zh" else "#D9534F" for x in peaks])
    ax.set_xticks(range(len(peaks)))
    ax.set_xticklabels([f"{SHORT[m]}\n{b[1]}·{b[2]}" for m, b in peaks], fontsize=6.4)
    for i, (m, b) in enumerate(peaks):
        ax.text(i, b[0] + 0.005, f"{b[0]:.3f}", ha="center", fontsize=6.5)
    ax.set_ylim(0, max(x[1][0] for x in peaks) * 1.18)
    ax.set_title(ttl, fontsize=9); ax.grid(axis="y", alpha=0.25)
fig.suptitle("Fig S5  各模型峰值指标与最优 Prompt（蓝=中文，红=英文）", fontsize=10)
fig.tight_layout(); fig.savefig(f"{FIG}/figS5_peak.png"); plt.close(fig)
print("figS5 ok")

# ---------------- Fig S15-S17: Prompt 优化 delta (P4-P1) ----------------
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
for ax, (task, metric, ttl) in zip(axes, [("T1", "macro", "Task1 Macro-F1"),
                                          ("T2", "f1", "Task2 F1"),
                                          ("T3", "f1", "Task3 F1")]):
    xs = []; labs = []; d = []; cols = []
    for m in MODELS:
        for l in ["zh", "en"]:
            v1 = _lookup(task, m, "P1", l); v4 = _lookup(task, m, "P4", l)
            if v1 and v4 and v1.get("status") == "full" and v4.get("status") == "full":
                xs.append(len(xs)); labs.append(f"{SHORT[m]}\n{l}"); d.append(v4[metric] - v1[metric])
                cols.append(C[l])
    ax.bar(xs, d, color=cols, edgecolor="white")
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=6.2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(ttl, fontsize=9); ax.grid(axis="y", alpha=0.25)
    ax.set_ylabel("Δ (P4 - P1)")
fig.suptitle("Fig S15-S17  Prompt 优化增益 Δ = P4 - P1（按模型 x 语言）", fontsize=10)
fig.tight_layout(); fig.savefig(f"{FIG}/figS15_17_prompt_delta.png"); plt.close(fig)
print("figS15_17 ok")

# ---------------- Fig S8/S9: Task2/Task3 八条件 F1 分布 ----------------
for task, metric, ttl, fn in [("T2", "f1", "Task2 八条件 F1 分布", "figS8_task2_dist.png"),
                              ("T3", "f1", "Task3 八条件 F1 分布", "figS9_task3_dist.png")]:
    data = []; labs = []
    for m in MODELS:
        for p in ["P1", "P2", "P3", "P4"]:
            for l in ["zh", "en"]:
                v = _lookup(task, m, p, l)
                if v and v.get("status") == "full":
                    data.append(v[metric]); labs.append(f"{SHORT[m]}\n{p}·{l}")
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.bar(range(len(data)), data, color="#2E6DA4", edgecolor="white")
    ax.set_xticks(range(len(data))); ax.set_xticklabels(labs, fontsize=5.6)
    ax.set_ylim(0, 1.0); ax.set_ylabel(metric.upper())
    ax.set_title(f"{ttl}（满全量条件）", fontsize=9); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{FIG}/{fn}"); plt.close(fig)
    print(fn, "ok")

# ---------------- Fig S3/S6/S7: 平衡集 ----------------
bal = [(m, p, l, D[f"BAL|{m}|{p}|{l}|"]) for m in MODELS
       for p in ["P1", "P2", "P3", "P4"] for l in ["zh", "en"]
       if f"BAL|{m}|{p}|{l}|" in D and D[f"BAL|{m}|{p}|{l}|"].get("status") == "full"]

if bal:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    xs = []; labs = []; cols = []
    for (m, p, l, v) in bal:
        xs.append(len(xs)); labs.append(f"{SHORT[m]}\n{p}·{l}")
        cols.append(C[l])
    ax.bar(xs, [v["macro"] for (_, _, _, v) in bal], color=cols, edgecolor="white")
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=6)
    ax.set_ylabel("Macro-F1"); ax.set_ylim(0, 0.75)
    ax.set_title("Fig S3  平衡数据集各条件 Task1 Macro-F1（565 例，蓝=中文，红=英文）", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{FIG}/figS3_balanced.png"); plt.close(fig)
    print("figS3 ok")

    fig, ax = plt.subplots(figsize=(7, 3.4))
    for pi, p in enumerate(["P1", "P2", "P3", "P4"]):
        vals = [v["macro"] for (m, pp, l, v) in bal if pp == p]
        if vals:
            ax.bar(pi, np.mean(vals), 0.7, color="#6A9FD6", edgecolor="white")
            ax.text(pi, np.mean(vals) + 0.005, f"{np.mean(vals):.3f}", ha="center", fontsize=7)
    ax.set_xticks(range(4)); ax.set_xticklabels(["P1", "P2", "P3", "P4"])
    ax.set_ylabel("Macro-F1（跨模型×语言均值）"); ax.set_ylim(0, 0.75)
    ax.set_title("Fig S6  平衡集上各 Prompt 策略对比", fontsize=9); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{FIG}/figS6_balanced_prompt.png"); plt.close(fig)
    print("figS6 ok")

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    zh = [v["macro"] for (m, p, l, v) in bal if l == "zh"]
    en = [v["macro"] for (m, p, l, v) in bal if l == "en"]
    ax.bar([0, 1], [np.mean(zh), np.mean(en)], 0.6, color=[C["zh"], C["en"]])
    for i, vals in enumerate([zh, en]):
        ax.text(i, np.mean(vals) + 0.005, f"{np.mean(vals):.3f}\n(n={len(vals)})", ha="center", fontsize=7)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["中文", "英文"])
    ax.set_ylabel("Macro-F1"); ax.set_ylim(0, 0.75)
    ax.set_title("Fig S7  平衡集中英对比（跨模型×Prompt 均值）", fontsize=9); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{FIG}/figS7_balanced_lang.png"); plt.close(fig)
    print("figS7 ok")

# ---------------- Fig S1/S2: McNemar 两两比较 ----------------
# S1: 最优模型 8 条件两两（Task1）
best_model = None; best_val = -1
for m in MODELS:
    for p in ["P1", "P2", "P3", "P4"]:
        for l in ["zh", "en"]:
            v = D.get(f"T1|{m}|{p}|{l}|")
            if v and v.get("status") == "full" and v["macro"] > best_val:
                best_val = v["macro"]; best_model = m
if best_model:
    labels = []; data = []
    for p in ["P1", "P2", "P3", "P4"]:
        for l in ["zh", "en"]:
            pr = pairs_for("T1", best_model, p, l)
            if pr:
                labels.append(f"{p}·{l}"); data.append(pr)
    entries = []; pvals = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            gi, pi_ = data[i]; gj, pj = data[j]
            common = [k for k in range(len(gi)) if gi[k] is not None]
            mi = {cid: None for cid in []}
            # 按 case_id 顺序一致（同一数据集固定顺序）
            n = min(len(gi), len(gj))
            b, c, p = mcnemar_pair(pi_[:n], pj[:n], gi[:n])
            entries.append((i, j, p)); pvals.append(p)
    adj = holm(pvals)
    adj_entries = [(i, j, a) for (i, j, _), a in zip(entries, adj)]
    sig_matrix(adj_entries, labels,
               [a for a in adj],
               f"Fig S1  Task1 最优模型 {SHORT[best_model]} 八条件两两 McNemar\n(Holm 校正 p_exact，* p<0.05)",
               "figS1_mcnemar_best.png")

# S2: 4 模型在 zh-P1 的两两（Task1）
labels2 = []; data2 = []
for m in MODELS:
    pr = pairs_for("T1", m, "P1", "zh")
    if pr:
        labels2.append(SHORT[m]); data2.append(pr)
entries2 = []; pvals2 = []
for i in range(len(data2)):
    for j in range(i + 1, len(data2)):
        gi, pi_ = data2[i]; gj, pj = data2[j]
        n = min(len(gi), len(gj))
        b, c, p = mcnemar_pair(pi_[:n], pj[:n], gi[:n])
        entries2.append((i, j, p)); pvals2.append(p)
adj2 = holm(pvals2)
sig_matrix([(i, j, a) for (i, j, _), a in zip(entries2, adj2)], labels2, adj2,
           "Fig S2  Task1 四模型 zh-P1 两两 McNemar\n(Holm 校正 p_exact，* p<0.05)",
           "figS2_mcnemar_models.png")

# ---------------- Table S18: 判据可得性对照 ----------------
rows = []
for task, name, exp in [("T1", "Task1 中风病五证型", 6906),
                        ("T2", "Task2 中脏腑二分类", 2555),
                        ("T3", "Task3 急性期分期", 6847)]:
    full = [(m, p, l) for m in MODELS for p in ["P1", "P2", "P3", "P4"] for l in ["zh", "en"]
            if f"{task}|{m}|{p}|{l}|" in D and D[f"{task}|{m}|{p}|{l}|"].get("status") == "full"]
    tot = sum(D[f"{task}|{m}|{p}|{l}|"]["n"] for (m, p, l) in full)
    rows.append((name, len(full), tot, exp * 32))
with open(f"{OUT}/manuscript_v3/TABLE_S18_criteria_availability.tsv", "w", encoding="utf8") as fh:
    fh.write("task\tn_full_conditions\tn_obs\tmax_possible\n")
    for r in rows:
        fh.write("\t".join(map(str, r)) + "\n")
print("Table S18 ok")

print("\nALL_SUPP_FIGURES done ->", FIG)
