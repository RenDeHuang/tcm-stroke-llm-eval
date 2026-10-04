# -*- coding: utf-8 -*-
"""二级评分：严格口径（字面）+ 同义口径（文献公认异名）。

产出两种指标，用于量化"命名差异 vs 真实辨证错误"：
  strict  : 模型输出必须等于金标写法
  relaxed : 允许映射到文献公认异名
"""
import re, sys, json
B="/Users/huangrende/Desktop/王健"; sys.path.insert(0,f"{B}/scripts/tcm_stroke_eval")
from synonym_map import SYNONYMS, TARGET

def norm(s): return re.sub(r"[\s\u3000]","",s or "").strip("。.")

def resolve(raw, vocab):
    """返回 (canonical|None, how)  how ∈ {exact, synonym, oov}"""
    if not raw: return None, "empty"
    s = norm(raw)
    for v in sorted(vocab, key=len, reverse=True):
        if norm(v) in s or norm(v).rstrip("证") in s:
            return v, "exact"
    for k, v in sorted(SYNONYMS.items(), key=lambda t: -len(t[0])):
        if norm(k) in s:
            return v, "synonym"
    return None, "oov"

def score(records, vocab):
    strict = syn = oov = empty = wrong = 0
    detail = []
    for r in records:
        gold = r.get("gold_syndrome") or r.get("label")
        v, how = resolve(r.get("raw") or r.get("pred"), vocab)
        if how == "empty": empty += 1
        elif v is None: oov += 1
        elif v == gold:
            if how == "exact": strict += 1
            else: syn += 1
        else: wrong += 1
        detail.append(dict(case_id=r["case_id"], gold=gold, pred=v, how=how, raw=r.get("raw")))
    n = len(records)
    return dict(n=n, strict=strict, synonym=syn, oov=oov, empty=empty, wrong=wrong,
                acc_strict=round(strict/n,4), acc_relaxed=round((strict+syn)/n,4),
                oov_rate=round(oov/n,4), detail=detail)
