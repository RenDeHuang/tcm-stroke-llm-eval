# -*- coding: utf-8 -*-
"""构建「中风病证型辨识」评测集（复刻 fendo 范式）。

口径（已与项目负责人确认）：
  - 病种：中风病 = 医院编码 BNG080(中风病) + BNG081(中风-中经络) + BNG082(中风-中脏腑)
          （同一疾病的书写/分期变体；缺血性/出血性中风为国标 GB/T 15657 另一套编码，不计入）
  - Task 1 证型：取最常见 5 类（覆盖 92.2%，即 6906/7489；权威数字见 verify_data_claims.py）
  - Task 2 病性要素：从证型名规则派生（定义上正确，无需人工标注）
  - Task 3 舌脉消融：由 Task 1 样本派生（去掉舌脉段落）

防泄漏（对齐 fendo 的 label-leakage 防护）：
  - 剔除文本中出现病名（中风/中经络/中脏腑）或证型名的记录
  - 按文本前 120 字去重
"""
import json, os, re, random
from collections import Counter, defaultdict

random.seed(20260925)
BASE = "/Users/huangrende/Desktop/王健"
OUT = f"{BASE}/out/tcm_stroke_eval"
os.makedirs(OUT, exist_ok=True)

STROKE_CODES = {"BNG080", "BNG081", "BNG082"}
ELEMENTS = ["气虚", "血虚", "阴虚", "阳虚", "气滞", "血瘀", "痰", "湿", "热", "寒", "风", "火"]

def norm(s): return re.sub(r"\s+", "", s or "")

rows = []
for f in ["train.json", "dev.json"]:
    for l in open(f"{BASE}/data/tcm_sd_full_20260923/{f}"):
        l = l.strip()
        if l: rows.append(json.loads(l))
for l in open(f"{BASE}/data/ZY-BERT_TCM-SD/test.json"):
    l = l.strip()
    if l: rows.append(json.loads(l))

pool = [r for r in rows if r["lcd_id"] in STROKE_CODES]
print(f"中风病池: {len(pool)}")

# ---- 清洗 ----
clean, drop_leak, drop_dup = [], 0, 0
seen = set()
for r in pool:
    cc, de, dt = norm(r.get("chief_complaint")), norm(r.get("description")), norm(r.get("detection"))
    txt = cc + de + dt
    syn = r.get("norm_syndrome") or r["syndrome"]
    if ("中风" in txt) or ("中经络" in txt) or ("中脏腑" in txt) or (syn.replace("证", "") in txt):
        drop_leak += 1; continue
    key = txt[:120]
    if key in seen:
        drop_dup += 1; continue
    seen.add(key)
    clean.append(dict(lcd_id=r["lcd_id"], lcd_name=r["lcd_name"], text=txt,
                      chief=cc, desc=de, detection=dt, syndrome=syn))
print(f"清洗后 {len(clean)} (泄漏剔除 {drop_leak}, 重复剔除 {drop_dup})")

# ---- Task 1: top5 ----
c = Counter(x["syndrome"] for x in clean)
top5 = [s for s, _ in c.most_common(5)]
print("top5:", top5)
sub = [x for x in clean if x["syndrome"] in top5]
print(f"top5 子集 {len(sub)} / {len(clean)} = {len(sub)/len(clean):.1%}")

# ---- Task 2 金标：从证型名派生要素 ----
def elems_of(syn):
    return sorted({e for e in ELEMENTS if e in syn})
for x in sub:
    x["gold_elements"] = elems_of(x["syndrome"])
ec = Counter()
for x in sub:
    for e in x["gold_elements"]: ec[e] += 1
print("病性要素金标分布:", dict(ec.most_common()))

# ---- 分层：每类固定上限，保证稀有类全用 ----
PER_CLASS = None   # None = 不封顶，保留真实自然分布（对齐 fendo）
bysyn = defaultdict(list)
for x in sub: bysyn[x["syndrome"]].append(x)
final = []
for s in top5:
    lst = bysyn[s][:]
    random.shuffle(lst)
    final.extend(lst[:PER_CLASS] if PER_CLASS else lst)
random.shuffle(final)
print(f"\n最终评测集 {len(final)} 例")
print("类分布:", {s: sum(1 for x in final if x['syndrome'] == s) for s in top5})

# ---- Task 3: 舌脉消融（从文本剥离舌象/脉象句）----
TONGUE = re.compile(r"[^。；;，,]{0,10}(舌[^。；;，,]{0,20})")
PULSE = re.compile(r"[^。；;，,]{0,10}(脉[^。；；,，]{0,20})")
def strip_tongue_pulse(t):
    t2 = TONGUE.sub("", t); t2 = PULSE.sub("", t2)
    return re.sub(r"[，,]{2,}", "，", t2).strip("，。； ")

# ---- 落盘 ----
out = []
for i, x in enumerate(final, 1):
    out.append(dict(
        case_id=f"STROKE-{i:04d}",
        source_code=x["lcd_id"], source_name=x["lcd_name"],
        text=x["text"],
        text_no_tongue_pulse=strip_tongue_pulse(x["text"]),
        gold_syndrome=x["syndrome"],
        gold_elements=x["gold_elements"],
        tongue=x["detection"], # 原文保留，便于核查
    ))
json.dump(out, open(f"{OUT}/eval_set.json", "w"), ensure_ascii=False, indent=1)
json.dump(top5, open(f"{OUT}/top5_vocab.json", "w"), ensure_ascii=False, indent=1)
json.dump(ELEMENTS, open(f"{OUT}/elements.json", "w"), ensure_ascii=False, indent=1)

# 长尾（用于"长治分析"）
tail = [x for x in clean if x["syndrome"] not in top5]
json.dump([dict(case_id=f"TAIL-{i:04d}", text=x["text"], gold_syndrome=x["syndrome"],
                source_code=x["lcd_id"]) for i, x in enumerate(tail, 1)],
          open(f"{OUT}/tail_set.json", "w"), ensure_ascii=False, indent=1)
print(f"\n已保存: {OUT}/eval_set.json ({len(out)}), tail_set.json ({len(tail)} 长尾)")
# 校验消融确实去掉了舌脉
chk = sum(1 for x in out if not re.search(r"舌|脉", x["text_no_tongue_pulse"]))
print(f"消融文本已无舌脉的例数: {chk}/{len(out)} = {chk/len(out):.1%}")
