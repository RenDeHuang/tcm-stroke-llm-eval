# -*- coding: utf-8 -*-
"""证型等价判定与三档口径评分 —— 方法论核心。

核心问题：中医证型命名不标准化，会系统性低估 LLM 评测分数。

三档口径：
  L0 严格字面   模型最终答案（从输出中抽取）必须与金标逐字一致
  L1 官方归一   允许使用 TCM-SD **数据集自带**的归一化表（客观、非本研究者构造）
  L2 知识库同证 敏感分析（需文本判定，主观性较高，仅辅助）

关键设计：先抽"最终答案"再比对。
  - CoT 输出（P4）含大段推理，必须抽末行结论，否则字面比对全错
  - 闭集输出（P5/P6）本身即答案
"""
import re, json
B = "/Users/huangrende/Desktop/王健"

def norm(s):
    return re.sub(r"[\s\u3000]", "", s or "").strip("。.，,；;")

# ---------- 从模型输出中抽取"最终答案" ----------
_MARKERS = [
    r"证\s*型\s*[:：]\s*([^\n]{1,40})",
    r"[Ss]yndrome\s*[:：]\s*([^\n]{1,40})",
    r"分类\s*[:：]\s*([^\n]{1,20})",
    r"[Cc]lass\s*[:：]\s*([^\n]{1,30})",
    r"最终(?:的)?(?:结论|判断|证型)[是为:：\s]*([^\n]{1,40})",
]

def extract_answer(raw):
    t = re.sub(r"```(?:json)?", "", raw or "").strip()
    if not t: return ""
    for pat in _MARKERS:
        m = re.findall(pat, t)
        if m:
            return m[-1].strip().strip("*_-— ").strip()
    if len(t) <= 25:                      # 短输出即答案
        return t.strip()
    lines = [l.strip() for l in t.split("\n") if l.strip()]   # 否则取最后一行
    return lines[-1] if lines else t.strip()

# ---------- L1: 官方归一表（来自数据集自身）----------
def build_official_map():
    mp = {}
    files = [f"{B}/data/tcm_sd_full_20260923/train.json",
             f"{B}/data/tcm_sd_full_20260923/dev.json",
             f"{B}/data/ZY-BERT_TCM-SD/test.json"]
    for fp in files:
        try:
            for line in open(fp):
                line = line.strip()
                if not line: continue
                r = json.loads(line)
                a, b = r.get("syndrome"), r.get("norm_syndrome")
                if a and b and a != b: mp[a] = b
        except FileNotFoundError: pass
    return mp

_OFFICIAL = None
def official_map():
    global _OFFICIAL
    if _OFFICIAL is None: _OFFICIAL = build_official_map()
    return _OFFICIAL

def to_official(name, vocab):
    """按 L1 官方口径把证型名归到 vocab 内（支持链式归一）。"""
    if not name: return None
    n = norm(name)
    for v in vocab:
        if norm(v) == n: return v
    cur, seen = n, set()
    for _ in range(6):
        if cur in seen: break
        seen.add(cur)
        nxt = official_map().get(cur)
        if not nxt: break
        cur = norm(nxt)
        for v in vocab:
            if norm(v) == cur: return v
    for v in vocab:                        # 容忍缺「证」后缀
        if norm(v).rstrip("证") == n: return v
    return None

def load_knowledge():
    kn = set()
    try:
        for l in open(f"{B}/data/tcm_sd_full_20260923/syndrome_knowledge.json", encoding="utf-8"):
            l = l.strip()
            if l: kn.add(json.loads(l)["Name"])
    except FileNotFoundError: pass
    return kn

def classify_oov(name, knowledge):
    n = norm(name)
    if not n: return "empty"
    if n in official_map(): return "official_synonym"
    if n in knowledge: return "in_knowledge"
    stem = n.rstrip("证")
    for k in knowledge:
        if norm(k).rstrip("证") == stem: return "in_knowledge"
    return "unattested"

# ---------- 评分 ----------
_TIER_RANK = {1: 1, 2: 2, 3: 3}

def final_label(r):
    """v2 口径：返回 (label, tier, exempt)。

    评测的最终答案是"裁决后的最终标签"，即 runner 写入的 `pred`：
      - tier 1：闭集约束直接成功；
      - tier 2：规则解析唯一确定；
      - tier 3：预注册裁决规则给出（可能为 None = 未决，计为错误）。

    旧版优先使用 `answer`（自由文本抽取），在闭集约束下与 `pred` 语义不同，
    会把 tier 2/3 的裁决结果重新打散，因此 v2 统一以 `pred` 为准。
    对不含 tier 字段的 legacy 记录，回退到旧的 answer/pred 逻辑。
    """
    tier = r.get("tier")
    if tier is None:
        raw = r.get("raw") or ""
        if r.get("answer"):
            return r["answer"], None, bool(r.get("exempt"))
        if len(raw) >= 1200 and r.get("pred"):
            return r["pred"], None, bool(r.get("exempt"))
        return extract_answer(raw), None, bool(r.get("exempt"))
    return (r.get("pred"), int(tier), bool(r.get("exempt")))

def score_records(records, vocab, knowledge=None):
    if knowledge is None: knowledge = load_knowledge()
    n = len(records); L0 = L1 = 0; detail = []
    tier_counts = {1: 0, 2: 0, 3: 0}
    n_exempt = 0
    oov_cls = {}
    for r in records:
        gold = r.get("gold_syndrome") or r.get("label")
        ans, tier, exempt = final_label(r)
        if tier is not None:
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        n_exempt += int(exempt)
        na = norm(ans)
        hit0 = (na == norm(gold))
        mapped = to_official(na, vocab)          # 若 na 已在词表内，直接返回
        hit1 = (mapped == gold)
        L0 += hit0; L1 += hit1
        cls = None
        if mapped is None and na:
            cls = classify_oov(na, knowledge); oov_cls[na] = cls
        detail.append(dict(case_id=r["case_id"], gold=gold, answer=na, pred=r.get("pred"),
                           tier=tier, exempt=exempt,
                           mapped=mapped, L0=bool(hit0), L1=bool(hit1), oov_class=cls))
    return dict(n=n, acc_L0=round(L0/n,4), acc_L1=round(L1/n,4),
                naming_gap=round((L1-L0)/n,4),
                tier_counts=tier_counts, n_exempt=n_exempt,
                oov_classes={k: sum(1 for v in oov_cls.values() if v==k) for k in set(oov_cls.values())},
                detail=detail)
