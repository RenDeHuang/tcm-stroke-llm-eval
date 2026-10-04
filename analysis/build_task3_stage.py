# -*- coding: utf-8 -*-
"""构建 Task3：中风病分期二分类（是否处于急性期）。

对齐 Fendo Task2/Task3 的规则型二分类原型：
  - 判据来自权威指南的**可计算阈值**：发病时长
  - 参照《中风病中医诊疗指南》分期：急性期 ≤2 周，恢复期 2 周–6 月，后遗症期 >6 月
  - 输出闭集 1/0（1 = 急性期，0 = 非急性期）
  - 标签由判据派生，**不来自诊断编码**，与 Fendo 从 Admission Diagnosis 派生同构

清洗与 Task1/Task2 保持一致：去重 + 剔除显式病名/证型名泄漏。
"""
import json, re, os
from collections import Counter

BASE = "/Users/huangrende/Desktop/王健"
OUT = f"{BASE}/out/tcm_stroke_eval"
STROKE = {"BNG080", "BNG081", "BNG082"}

def norm(s): return re.sub(r"\s+", "", s or "")

rows = []
for f in ["train.json", "dev.json"]:
    for l in open(f"{BASE}/data/tcm_sd_full_20260923/{f}"):
        l = l.strip()
        if l: rows.append(json.loads(l))
for l in open(f"{BASE}/data/ZY-BERT_TCM-SD/test.json"):
    l = l.strip()
    if l: rows.append(json.loads(l))

pool = [r for r in rows if r["lcd_id"] in STROKE]
clean, seen = [], set()
for r in pool:
    txt = norm(r.get("chief_complaint")) + norm(r.get("description")) + norm(r.get("detection"))
    syn = r.get("norm_syndrome") or r["syndrome"]
    if "中风" in txt or "中经络" in txt or "中脏腑" in txt or syn.replace("证", "") in txt:
        continue
    if txt[:120] in seen:
        continue
    seen.add(txt[:120])
    clean.append(r)
print(f"清洗池 {len(clean)}")

# ---- 发病时长解析 ----
YR = re.compile(r"(\d+)\s*年")
MO = re.compile(r"(\d+)\s*个?月")
WK = re.compile(r"(\d+)\s*(?:周|星期)")
DY = re.compile(r"(\d+)\s*天|(\d+)\s*日")
HR = re.compile(r"(\d+)\s*(?:小时|h)")

def onset_days(text):
    """从主诉/现病史开头提取发病时长（天）。取最早出现的时长表达。"""
    head = text[:80]
    for pat, mult in ((YR, 365), (MO, 30), (WK, 7), (DY, 1), (HR, 0)):
        m = pat.search(head)
        if m:
            v = int(m.group(1) or (m.group(2) if m.lastindex and m.lastindex >= 2 else 0))
            if mult == 0:
                return 0            # 数小时内 -> 急性期
            return v * mult
    return None

ACUTE_DAYS = 14          # 指南：急性期 ≤2 周

out = []
for r in clean:
    narrative = norm(r.get("chief_complaint")) + norm(r.get("description"))
    d = onset_days(narrative)
    if d is None:
        continue
    label = 1 if d <= ACUTE_DAYS else 0
    txt = norm(r.get("chief_complaint")) + norm(r.get("description")) + norm(r.get("detection"))
    out.append(dict(
        case_id=f"T3S-{len(out)+1:05d}",
        text=txt,
        onset_days=d,
        stage=("急性期" if d <= ACUTE_DAYS else ("恢复期" if d <= 183 else "后遗症期")),
        label=label,
        label_name=("急性期" if label == 1 else "非急性期"),
        source_code=r["lcd_id"],
    ))

c = Counter(x["label"] for x in out)
print(f"Task3(分期二分类) n={len(out)}  急性期={c[1]} ({c[1]/len(out):.1%})  非急性期={c[0]} ({c[0]/len(out):.1%})")
print(f"三分类分布: {dict(Counter(x['stage'] for x in out))}")

# ---- 泄漏检查：答案是否直接写在文本里 ----
leak = sum(bool(re.search(r"急性期|恢复期|后遗症期", x["text"])) for x in out)
print(f"文本直接含分期词: {leak}/{len(out)} = {leak/len(out):.2%}  (仅作记录，病程本身是判据)")

json.dump(out, open(f"{OUT}/task3_stage_set.json", "w"), ensure_ascii=False, indent=1)
print(f"saved {OUT}/task3_stage_set.json")
