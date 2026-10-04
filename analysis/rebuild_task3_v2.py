# -*- coding: utf-8 -*-
"""Task3 v2：修正标签派生缺陷，并用现有预测重计分（不调用 API）。

缺陷（2026-09-29 发现，producer 自检 B01 与本审计共同确认）：
  v1 的 onset_days() 按【单位优先级】搜索（年→月→周→日），不是按文本位置。
  因此 "头晕乏力2天患者于2019年1月9日..." 会先命中 "2019年" -> 736935 天，
  把明确属于急性期的病例误标为非急性期。

修正（仅修 bug，不改构造定义）：
  1. 先剥离日期串（19xx/20xx 年、YYYY-M-D、YYYY年M月D日）
  2. 按【文本位置】取最早出现的时长表达
  3. 支持 半X（半年=180 / 半月=15 / 半天=0）
  4. 保留「首个时长=发病时点」的原构造语义（分期看中风发病时点）

产出：
  task3_stage_set_v2.json  —— 修正后金标
  task3_v1_v2_diff.json    —— 逐例差异
  重计分报告（stdout）
"""
import json, re, os, glob
from collections import Counter

BASE = "/Users/huangrende/Desktop/王健"
OUT = f"{BASE}/out/tcm_stroke_eval"

YR = re.compile(r"(\d+)\s*年"); MO = re.compile(r"(\d+)\s*个?月")
WK = re.compile(r"(\d+)\s*(?:周|星期)"); DY = re.compile(r"(\d+)\s*天|(\d+)\s*日")
HR = re.compile(r"(\d+)\s*(?:小时|h)")
HALF = re.compile(r"半\s*(年|个?月|周|天|日)")
PATS = [(YR, 365), (MO, 30), (WK, 7), (DY, 1), (HR, 0)]
DATE = re.compile(r"(19|20)\d{2}\s*年(?:\s*\d{1,2}\s*月)?(?:\s*\d{1,2}\s*[日号])?|\d{4}-\d{1,2}-\d{1,2}|\d{4}年\d{1,2}月\d{1,2}日")
ACUTE_DAYS = 14

def _val(m):
    return int(m.group(1) or (m.group(2) if m.lastindex and m.lastindex >= 2 else 0))

def onset_v2(text):
    """修正版：按位置取最早时长；剥离日期；支持半X。"""
    head = text[:80]
    stripped = DATE.sub("\u3000", head)
    cands = []
    for m in HALF.finditer(stripped):
        unit = m.group(1)
        v = 180 if "年" in unit else (15 if "月" in unit else 0)
        cands.append((m.start(), v))
    for pat, mult in PATS:
        for m in pat.finditer(stripped):
            cands.append((m.start(), 0 if mult == 0 else _val(m) * mult))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0])
    return cands[0][1]

# ---- 重建 ----
v1 = json.load(open(f"{OUT}/task3_stage_set.json"))
v2, diff = [], []
for r in v1:
    d = onset_v2(r["text"])
    if d is None:
        continue
    lab = 1 if d <= ACUTE_DAYS else 0
    n = dict(r)
    n["onset_days"] = d
    n["stage"] = "急性期" if d <= ACUTE_DAYS else ("恢复期" if d <= 183 else "后遗症期")
    n["label"] = lab
    n["label_name"] = "急性期" if lab == 1 else "非急性期"
    v2.append(n)
    if lab != r["label"]:
        diff.append(dict(case_id=r["case_id"], v1_days=r["onset_days"], v2_days=d,
                         v1_label=r["label"], v2_label=lab))
json.dump(v2, open(f"{OUT}/task3_stage_set_v2.json", "w"), ensure_ascii=False, indent=1)
json.dump(diff, open(f"{OUT}/task3_v1_v2_diff.json", "w"), ensure_ascii=False, indent=1)

c1 = Counter(x["label"] for x in v1); c2 = Counter(x["label"] for x in v2)
print("=" * 82)
print("Task3 标签重建")
print("=" * 82)
print(f"  v1 n={len(v1)}  急性期={c1[1]} ({c1[1]/len(v1):.2%})  最大onset={max(x['onset_days'] for x in v1)}")
print(f"  v2 n={len(v2)}  急性期={c2[1]} ({c2[1]/len(v2):.2%})  最大onset={max(x['onset_days'] for x in v2)}")
print(f"  标签翻转 = {len(diff)} ({len(diff)/len(v1):.2%})")
print(f"  多数类基线: v1={max(c1.values())/len(v1):.4f}  v2={max(c2.values())/len(v2):.4f}")

# ---- 重计分：现有预测 vs 新金标（无 API）----
GOLD2 = {x["case_id"]: x["label"] for x in v2}
def f1(yt, yp):
    tp = sum(1 for a, b in zip(yt, yp) if a == 1 == b)
    fp = sum(1 for a, b in zip(yt, yp) if b == 1 and a != 1)
    fn = sum(1 for a, b in zip(yt, yp) if a == 1 and b != 1)
    p = tp/(tp+fp) if tp+fp else 0.0; r = tp/(tp+fn) if tp+fn else 0.0
    return (2*p*r/(p+r) if p+r else 0.0), p, r

print()
print("=" * 82)
print("Task3 重计分（同一批预测，仅换金标，无 API 调用）")
print("=" * 82)
print(f"{'file':58s}{'n':>6s}{'v1_F1':>8s}{'v2_F1':>8s}{'Δ':>8s}")
rows = []
for f in sorted(glob.glob(f"{OUT}/runs/*__T3_*.jsonl")):
    if "__no_tp__" in f: continue
    recs = []
    seen = {}
    for l in open(f):
        l = l.strip()
        if not l: continue
        try: r = json.loads(l)
        except Exception: continue
        seen[r["case_id"]] = r
    recs = list(seen.values())
    ok = [r for r in recs if not r.get("err") and r.get("pred") is not None
          and r["case_id"] in GOLD2]
    if not ok: continue
    yt1 = [int(r["gold_syndrome"]) for r in ok]
    yt2 = [GOLD2[r["case_id"]] for r in ok]
    yp = [int(r["pred"]) for r in ok]
    f1a, _, _ = f1(yt1, yp); f1b, _, _ = f1(yt2, yp)
    name = os.path.basename(f).split("__fendo")[0]
    rows.append((name, len(ok), f1a, f1b))
    print(f"{name:58s}{len(ok):>6d}{f1a:>8.3f}{f1b:>8.3f}{f1b-f1a:>+8.3f}")
if rows:
    print()
    print(f"  F1 平均变化 = {sum(r[3]-r[2] for r in rows)/len(rows):+.4f}")
    print(f"  F1 最小变化 = {min(r[3]-r[2] for r in rows):+.4f}  最大变化 = {max(r[3]-r[2] for r in rows):+.4f}")
