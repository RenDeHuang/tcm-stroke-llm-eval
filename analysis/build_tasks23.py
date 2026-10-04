# -*- coding: utf-8 -*-
"""构建 Task2（中经络/中脏腑 二分类）与 Task3（舌脉消融）数据集。"""
import json, re, os, random
from collections import Counter
random.seed(20260925)
BASE="/Users/huangrende/Desktop/王健"; OUT=f"{BASE}/out/tcm_stroke_eval"

rows=[]
for f in ["train.json","dev.json"]:
    for l in open(f"{BASE}/data/tcm_sd_full_20260923/{f}"):
        l=l.strip()
        if l: rows.append(json.loads(l))
for l in open(f"{BASE}/data/ZY-BERT_TCM-SD/test.json"):
    l=l.strip()
    if l: rows.append(json.loads(l))
norm=lambda s:re.sub(r"\s+","",s or "")
st=[r for r in rows if r["lcd_id"] in {"BNG080","BNG081","BNG082"}]
clean,seen=[],set()
for r in st:
    txt=norm(r.get("chief_complaint"))+norm(r.get("description"))+norm(r.get("detection"))
    syn=r.get("norm_syndrome") or r["syndrome"]
    if "中风" in txt or "中经络" in txt or "中脏腑" in txt or syn.replace("证","") in txt: continue
    if txt[:120] in seen: continue
    seen.add(txt[:120])
    clean.append(dict(text=txt, code=r["lcd_id"], syndrome=syn))
print(f"清洗池 {len(clean)}")

# ---- Task2: 中经络(BNG081) vs 中脏腑(BNG082) ----
LAB={"BNG081":"中经络","BNG082":"中脏腑"}
t2=[dict(case_id=f"T2-{i:04d}", text=x["text"], code=x["code"], label=LAB[x["code"]])
    for i,x in enumerate([y for y in clean if y["code"] in LAB],1)]
c=Counter(x["label"] for x in t2)
print(f"Task2: {len(t2)}  分布 {dict(c)}  阳性率 {c['中脏腑']/len(t2):.2%}")
json.dump(t2, open(f"{OUT}/task2_set.json","w"), ensure_ascii=False, indent=1)

# 注：中脏腑仅 98 例，做类别均衡敏感性分析用
print(f"  -> 稀有类仅 {c['中脏腑']} 例，将做类别均衡抽样敏感性分析")

# ---- Task3: 舌脉消融（基于 Task1 评测集）----
t1=json.load(open(f"{OUT}/eval_set.json"))
TONGUE=re.compile(r"[^。；;，,]{0,12}舌[^。；;，,]{0,24}")
PULSE=re.compile(r"[^。；;，,]{0,12}脉[^。；;，,]{0,24}")
def strip_tp(t):
    t2=TONGUE.sub("",t); t2=PULSE.sub("",t2)
    return re.sub(r"[，,]{2,}","，",t2).strip("，。； ")
t3=[]
for x in t1:
    t3.append(dict(case_id=x["case_id"], gold_syndrome=x["gold_syndrome"],
                   text_full=x["text"], text_no_tp=strip_tp(x["text"])))
json.dump(t3, open(f"{OUT}/task3_set.json","w"), ensure_ascii=False, indent=1)
chk=sum(1 for x in t3 if not re.search(r"舌|脉",x["text_no_tp"]))
print(f"Task3: {len(t3)} 例, 消融成功率 {chk/len(t3):.1%}")

# ---- 汇总 ----
meta=dict(
  disease="中风病(BNG080+081+082)", pool_raw=8696, pool_clean=len(clean),
  task1=dict(n=len(t1), vocab=json.load(open(f"{OUT}/top5_vocab.json")),
             dist=dict(Counter(x["gold_syndrome"] for x in t1))),
  task2=dict(n=len(t2), labels=list(c.keys()), dist=dict(c)),
  task3=dict(n=len(t3), mode="tongue-pulse ablation"),
)
json.dump(meta, open(f"{OUT}/meta.json","w"), ensure_ascii=False, indent=1)
print("\n=== meta ===")
print(json.dumps(meta, ensure_ascii=False, indent=1)[:1200])
