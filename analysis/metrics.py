# -*- coding: utf-8 -*-
"""指标：macro-F1 / accuracy / bootstrap 95%CI / McNemar。"""
import numpy as np
from collections import defaultdict

def prf(y_true, y_pred, labels):
    """宏平均 precision/recall/F1（仅对两类都出现的标签计入，避免单类标签主导）。"""
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        if p is None:                     # 未解析 -> 记为错（不算任何类的 TP）
            fn[t] += 1; continue
        if t == p: tp[t] += 1
        else: fp[p] += 1; fn[t] += 1
    ps, rs, fs = [], [], []
    for l in labels:
        if tp[l] + fn[l] == 0: continue   # 金标未出现的类不计
        p = tp[l] / (tp[l] + fp[l]) if (tp[l] + fp[l]) else 0.0
        r = tp[l] / (tp[l] + fn[l])
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        ps.append(p); rs.append(r); fs.append(f)
    return (float(np.mean(ps)) if ps else 0.0,
            float(np.mean(rs)) if rs else 0.0,
            float(np.mean(fs)) if fs else 0.0)

def accuracy(y_true, y_pred):
    n = len(y_true)
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / max(n, 1)


# ---- 二分类指标（对齐 fendo Task2/Task3：Accuracy / Recall / Specificity / F1）----
def _bin_counts(y_true, y_pred, pos_label=1):
    tp = fp = fn = tn = 0
    for a, b in zip(y_true, y_pred):
        if a == pos_label and b == pos_label: tp += 1
        elif a != pos_label and b == pos_label: fp += 1
        elif a == pos_label and b != pos_label: fn += 1
        else: tn += 1
    return tp, fp, fn, tn

def sensitivity(y_true, y_pred, pos_label=1):
    """敏感度/召回 = TP / (TP + FN)。"""
    tp, fp, fn, tn = _bin_counts(y_true, y_pred, pos_label)
    return tp / (tp + fn) if (tp + fn) else 0.0

def specificity(y_true, y_pred, pos_label=1):
    """特异度 = TN / (TN + FP)。fendo Task2/3 的核心指标之一。"""
    tp, fp, fn, tn = _bin_counts(y_true, y_pred, pos_label)
    return tn / (tn + fp) if (tn + fp) else 0.0

def precision(y_true, y_pred, pos_label=1):
    tp, fp, fn, tn = _bin_counts(y_true, y_pred, pos_label)
    return tp / (tp + fp) if (tp + fp) else 0.0

def binary_f1(y_true, y_pred, pos_label=1):
    """阳性类 F1，对齐 fendo Task2/3 的 'F1 Score' 列。"""
    p = precision(y_true, y_pred, pos_label)
    r = sensitivity(y_true, y_pred, pos_label)
    return 2 * p * r / (p + r) if (p + r) else 0.0

def macro_f1(y_true, y_pred, labels):
    return prf(y_true, y_pred, labels)[2]

def bootstrap_ci(y_true, y_pred, labels, fn=macro_f1, n=1000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    yt = np.asarray(y_true, dtype=object); yp = np.asarray(y_pred, dtype=object)
    N = len(yt); vals = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        vals.append(fn(list(yt[idx]), list(yp[idx]), labels))
    lo, hi = np.percentile(vals, [100*alpha/2, 100*(1-alpha/2)])
    return float(fn(list(yt), list(yp), labels)), float(lo), float(hi)

def mcnemar(y_true, pred_a, pred_b):
    """成对比较；返回 (b, c, p)。b=A对B错, c=A错B对。"""
    b = sum(1 for t, a, c2 in zip(y_true, pred_a, pred_b) if a == t and c2 != t)
    c = sum(1 for t, a, c2 in zip(y_true, pred_a, pred_b) if a != t and c2 == t)
    n = b + c
    if n == 0: return b, c, 1.0
    # 精确二项检验
    from math import comb
    p = sum(comb(n, k) for k in range(min(b, c)+1)) / (2 ** n) * 2
    return b, c, min(1.0, p)

def binary_prf_multilabel(y_true_sets, y_pred_sets, elements):
    """多标签：每要素二分类的 macro P/R/F1。pred 为 None 表示未解析（全错）。"""
    ps, rs, fs = [], [], []
    for e in elements:
        tp = fp = fn = tn = 0
        for t, p in zip(y_true_sets, y_pred_sets):
            pt = e in t; pp = (e in p) if p is not None else False
            if pt and pp: tp += 1
            elif pt and not pp: fn += 1
            elif not pt and pp: fp += 1
            else: tn += 1
        p_ = tp/(tp+fp) if tp+fp else 0.0
        r_ = tp/(tp+fn) if tp+fn else 0.0
        f_ = 2*p_*r_/(p_+r_) if p_+r_ else 0.0
        ps.append(p_); rs.append(r_); fs.append(f_)
    return float(np.mean(ps)), float(np.mean(rs)), float(np.mean(fs))

def bootstrap_ci_multilabel(y_true_sets, y_pred_sets, elements, n=1000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    yt = list(y_true_sets); yp = list(y_pred_sets); N = len(yt); vals = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        vals.append(binary_prf_multilabel([yt[i] for i in idx], [yp[i] for i in idx], elements)[2])
    lo, hi = np.percentile(vals, [100*alpha/2, 100*(1-alpha/2)])
    return float(binary_prf_multilabel(yt, yp, elements)[2]), float(lo), float(hi)

def per_class_f1(y_true, y_pred, labels):
    """返回每个金标类别的 F1；未出现的金标类别记为 None。"""
    out = {}
    for l in labels:
        tp = fp = fn = 0
        for t, p in zip(y_true, y_pred):
            if p is None:
                if t == l: fn += 1
            elif t == l and p == l:
                tp += 1
            elif p == l and t != l:
                fp += 1
            elif t == l and p != l:
                fn += 1
        if tp + fn == 0:
            out[l] = None
        else:
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn)
            out[l] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return out


def weighted_f1(y_true, y_pred, labels):
    """按金标支持数加权的 F1，便于与 fendo 的 weighted-F1 对齐。"""
    pcs = per_class_f1(y_true, y_pred, labels)
    support = {l: sum(1 for t in y_true if t == l) for l in labels}
    total = sum(support.values())
    if not total:
        return 0.0
    return sum((support[l] / total) * (pcs[l] or 0.0) for l in labels if pcs[l] is not None)
