# -*- coding: utf-8 -*-
"""入稿数字核验器：一次性复算 METHOD_LOCK / 稿件中所有对外声称的数字。

设计原则：每个数字都必须能从原始数据重新推导，且口径在代码中显式可见。
任何与文档不一致的数字，以本脚本输出为准（本脚本是数字的唯一权威来源）。

用法: python3 verify_data_claims.py
"""
import json, re, os
from collections import Counter

BASE = "/Users/huangrende/Desktop/王健"
OUT = f"{BASE}/out/tcm_stroke_eval"

def norm(s):
    return re.sub(r"\s+", "", s or "")

# ---------- 载入原始池 ----------
rows = []
for f in ["train.json", "dev.json"]:
    for l in open(f"{BASE}/data/tcm_sd_full_20260923/{f}"):
        l = l.strip()
        if l:
            rows.append(json.loads(l))
for l in open(f"{BASE}/data/ZY-BERT_TCM-SD/test.json"):
    l = l.strip()
    if l:
        rows.append(json.loads(l))

STROKE = {"BNG080", "BNG081", "BNG082"}
pool = [r for r in rows if r["lcd_id"] in STROKE]

# ---------- 清洗（与 build_eval_set.py 同口径）----------
clean, drop_leak, drop_dup = [], 0, 0
seen = set()
for r in pool:
    cc, de, dt = norm(r.get("chief_complaint")), norm(r.get("description")), norm(r.get("detection"))
    txt = cc + de + dt
    syn = r.get("norm_syndrome") or r["syndrome"]
    if ("中风" in txt) or ("中经络" in txt) or ("中脏腑" in txt) or (syn.replace("证", "") in txt):
        drop_leak += 1
        continue
    key = txt[:120]
    if key in seen:
        drop_dup += 1
        continue
    seen.add(key)
    clean.append(dict(code=r["lcd_id"], cc=cc, de=de, det=dt, txt=txt, syn=syn))

print("=" * 72)
print("CLAIM 1  数据池与清洗")
print("=" * 72)
print(f"  pool_raw (BNG080/081/082)      = {len(pool)}")
print(f"  drop_leak                      = {drop_leak}")
print(f"  drop_dup                       = {drop_dup}")
print(f"  pool_clean                     = {len(clean)}")

# ---------- CLAIM 2: Top5 覆盖率 ----------
c = Counter(x["syn"] for x in clean)
top5 = [s for s, _ in c.most_common(5)]
top5n = sum(v for k, v in c.items() if k in top5)
print()
print("=" * 72)
print("CLAIM 2  Task1 Top5 覆盖率")
print("=" * 72)
print(f"  top5 vocab                     = {top5}")
print(f"  top5_n / clean                 = {top5n}/{len(clean)} = {top5n/len(clean):.4%}")
print(f"  long_tail_fraction             = {(len(clean)-top5n)/len(clean):.4%}")
print(f"  [DOC EXPECTS 92.22%] -> {'ok' if abs(top5n/len(clean)-0.9222) <= 0.0005 else 'MISMATCH: update docs'}")
print(f"  [DOC EXPECTS 7.78%]  -> {'ok' if abs((len(clean)-top5n)/len(clean)-0.0778) <= 0.0005 else 'MISMATCH: update docs'}")

# ---------- CLAIM 3: Task1 标签分布与基线 ----------
t1 = json.load(open(f"{OUT}/eval_set.json"))
d1 = Counter(x["gold_syndrome"] for x in t1)
n1 = sum(d1.values())
print()
print("=" * 72)
print("CLAIM 3  Task1 分布与基线")
print("=" * 72)
for k, v in d1.most_common():
    print(f"  {k:8s} {v:5d}  {v/n1:.4%}")
print(f"  majority-class baseline Acc    = {max(d1.values())/n1:.4f}")
print(f"  uniform baseline Acc           = {1/len(d1):.4f}")

# ---------- CLAIM 4: Task2 标签质量（区分叙事/查体）----------
LAB = {"BNG081": "中经络", "BNG082": "中脏腑"}
t2raw = [x for x in clean if x["code"] in LAB]

PSYCH_KW = ["神志异常", "神志不清", "神志模糊", "意识障碍", "意识模糊", "意识不清",
            "嗜睡", "昏睡", "昏迷", "恍惚", "反应迟钝", "认知障碍", "不能对答",
            "神志蒙蔽", "神昏", "谵语", "烦躁"]
TPL = "神志清晰，精神尚可"      # 固定查体套话（逐字）
print()
print("=" * 72)
print("CLAIM 4  Task2 文本信号分离度（口径显式声明）")
print("=" * 72)
print(f"  叙事口径 = chief_complaint + description 命中任一 PSYCH_KW")
print(f"  查体口径 = detection 命中逐字套话 '{TPL}'")
print(f"  套话缺失率 = 1 - 查体口径")
for lab in ("中经络", "中脏腑"):
    sub = [x for x in t2raw if LAB[x["code"]] == lab]
    n = len(sub)
    nar = sum(1 for x in sub if any(k in (x["cc"] + x["de"]) for k in PSYCH_KW))
    tpl = sum(1 for x in sub if TPL in x["det"])
    print(f"  -- {lab} n={n}")
    print(f"     叙事含神志异常   {nar}/{n} = {nar/n:.2%}")
    print(f"     查体含套话       {tpl}/{n} = {tpl/n:.2%}")
    print(f"     查体缺套话       {n-tpl}/{n} = {(n-tpl)/n:.2%}")

# 简单规则基线
def rule(x):
    return 1 if (any(k in (x["cc"] + x["de"]) for k in PSYCH_KW) or (TPL not in x["det"])) else 0
y_true = [1 if LAB[x["code"]] == "中脏腑" else 0 for x in t2raw]
y_pred = [rule(x) for x in t2raw]
acc = sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)
print(f"  规则基线 Acc (叙事异常 OR 查体缺套话) = {acc:.4f}")
print(f"  多数类基线 Acc                       = {max(Counter(y_true).values())/len(y_true):.4f}")

# ---------- CLAIM 5: Task2/Task3 标签分布 ----------
print()
print("=" * 72)
print("CLAIM 5  Task2 / Task3 标签分布")
print("=" * 72)
t2 = json.load(open(f"{OUT}/task2_binary_set.json"))
d2 = Counter(x["label"] for x in t2)
print(f"  T2 n={len(t2)} {dict(d2)} pos_rate={d2[1]/len(t2):.4%} majority_baseline={max(d2.values())/len(t2):.4f}")
print(f"  T2 minority_class_n = {d2[1]}")
t3 = json.load(open(f"{OUT}/task3_stage_set.json"))
d3 = Counter(x["label"] for x in t3)
print(f"  T3 n={len(t3)} {dict(d3)} pos_rate={d3[1]/len(t3):.4%} majority_baseline={max(d3.values())/len(t3):.4f}")

# ---------- CLAIM 6: Task2/Task3 判据可得性 ----------
# 口径声明：判据可得性 = 在【清洗池 7489】中可解析出判据的比例（与 build_task3_stage.py 同源）
YR = re.compile(r"(\d+)\s*年"); MO = re.compile(r"(\d+)\s*个?月")
WK = re.compile(r"(\d+)\s*(?:周|星期)"); DY = re.compile(r"(\d+)\s*天|(\d+)\s*日")
HR = re.compile(r"(\d+)\s*(?:小时|h)")

def onset_days(text):
    head = text[:80]
    for pat, mult in ((YR, 365), (MO, 30), (WK, 7), (DY, 1), (HR, 0)):
        m = pat.search(head)
        if m:
            v = int(m.group(1) or (m.group(2) if m.lastindex and m.lastindex >= 2 else 0))
            return 0 if mult == 0 else v * mult
    return None

print()
print("=" * 72)
print("CLAIM 6  Task2/Task3 判据可得性与泄漏")
print("=" * 72)
parsed = sum(1 for x in clean if onset_days(x["cc"] + x["de"]) is not None)
print(f"  [T3] 判据=发病时长, 清洗池口径 {parsed}/{len(clean)} = {parsed/len(clean):.4%}")
print(f"  [DOC EXPECTS 91.48%] -> {'ok' if abs(parsed/len(clean)-0.9148) <= 0.0005 else 'MISMATCH: update docs'}")

PSYCH_ANY = ["神志", "意识", "昏", "嗜睡", "恍惚", "呆", "迟钝", "谵", "萎靡", "清楚", "清晰", "正常"]
psy_hit = sum(1 for x in t2raw if any(k in (x["cc"] + x["de"] + x["det"]) for k in PSYCH_ANY))
print(f"  [T2] 判据=神志描述, T2池口径 {psy_hit}/{len(t2raw)} = {psy_hit/len(t2raw):.4%}")
print(f"  [DOC EXPECTS 99.96%] -> {'ok' if abs(psy_hit/len(t2raw)-0.9996) <= 0.0005 else 'MISMATCH: update docs'}")

STAGE_KW = ["急性期", "恢复期", "后遗症期"]
leak = sum(1 for x in t3 if any(k in x["text"] for k in STAGE_KW))
print(f"  [T3] 分期词泄漏 = {leak}/{len(t3)} = {leak/len(t3):.4%}")

# ---------- CLAIM 7: 去重率 ----------
duprate = drop_dup / len(pool)
print()
print("=" * 72)
print("CLAIM 7  文本模板化（去重率）")
print("=" * 72)
print(f"  dup_drop / pool_raw            = {drop_dup}/{len(pool)} = {duprate:.4%}")
