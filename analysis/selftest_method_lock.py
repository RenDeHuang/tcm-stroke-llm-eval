# -*- coding: utf-8 -*-
"""方法锁定自检：验证 fendo 对齐后的 Prompt 结构、输出契约和三层裁决。

不调用任何 API；全部为本地离线检查，可随时重跑。
    python3 scripts/tcm_stroke_eval/selftest_method_lock.py
"""
import sys, json, re
BASE = "/Users/huangrende/Desktop/王健"
sys.path.insert(0, f"{BASE}/scripts/tcm_stroke_eval")
import prompts as P
from runner import parse_t1, _is_detailed_analysis

OUT = f"{BASE}/out/tcm_stroke_eval"
VOCAB = json.load(open(f"{OUT}/top5_vocab.json"))
TEXT = "患者神志清，半身不遂，言语謇涩，舌暗紫，苔白腻，脉弦滑。"
fails = []

def check(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)

print("=" * 80); print("1. additional_prompt：闭集 + 禁止冗余输出"); print("=" * 80)
zh, en = P.additional_prompt(VOCAB, "zh"), P.additional_prompt(VOCAB, "en")
check("不要添加其他文字" in zh, "中文 additional_prompt 含「不要添加其他文字」")
check(all(v in zh for v in VOCAB), "中文 additional_prompt 含全部 5 个标准证型")
check("without anything else" in en and "No additional explanation" in en,
      "英文 additional_prompt 含 'without anything else' / 'No additional explanation'")
check(all(v in en for v in VOCAB), "英文 additional_prompt 含全部 5 个标准证型")

print(); print("=" * 80); print("2. input_text 模板结构（对照 data sheet 3）"); print("=" * 80)
for st in ["P1", "P2", "P3", "P4"]:
    for lang in ["zh", "en"]:
        p = P.build_t1(st, TEXT, lang, VOCAB, None)
        tail = p.strip().split("\n")[-1]
        check("病历：" in p or "Medical record:" in p, f"{st}/{lang} 含病例字段前缀")
        check(("如果你是一位专业医生" in p) or ("If you are a professional doctor" in p),
              f"{st}/{lang} 含医生角色设定句")
        check(tail == P.additional_prompt(VOCAB, lang).strip(),
              f"{st}/{lang} 末行严格等于 additional_prompt（输出契约位置正确）")

print(); print("=" * 80); print("3. P1-P4 引导强度阶梯"); print("=" * 80)
lens = {st: len(P.build_t1(st, TEXT, "zh", VOCAB, None)) for st in ["P1", "P2", "P3", "P4"]}
print("   prompt 长度:", lens)
check(lens["P1"] < lens["P2"] < lens["P3"], "P1 < P2 < P3（引导强度递增）")
check(lens["P4"] > lens["P2"], "P4 含完整推理链，长度大于 P2")
p2 = P.build_t1("P2", TEXT, "zh", VOCAB, None)
check("《中风病中医诊疗指南》" in p2, "P2 只点名权威标准（不贴框架内容）")
p3 = P.build_t1("P3", TEXT, "zh", VOCAB, None)
check("支持：" in p3 and "鉴别：" in p3, "P3 含类别特异判定标准")
p4 = P.build_t1("P4", TEXT, "zh", VOCAB, None)
check("按照上述辨证思维链" in p4, "P4 以思维链收尾追问")

print(); print("=" * 80); print("4. 三层裁决协议"); print("=" * 80)
CASES = [
    ("风痰瘀阻证", 1, "风痰瘀阻证", "裸标签"),
    ("风痰瘀阻证。", 1, "风痰瘀阻证", "裸标签+标点"),
    ("证型：气虚血瘀证", 1, "气虚血瘀证", "规范前缀行"),
    ("综合来看，患者应属气虚血瘀证。", 2, "气虚血瘀证", "唯一候选"),
    ("可能是风痰瘀阻证，也可能是风痰入络证。", 3, "风痰入络证", "歧义->取最后"),
    ("无法确定该患者的证型。", 3, None, "无候选->未决"),
]
for raw, exp_t, exp_p, name in CASES:
    got_p, got_t = parse_t1(raw, VOCAB)
    check((got_t, got_p) == (exp_t, exp_p),
          f"tier{exp_t} {name}: got=({got_t},{got_p})")

print(); print("=" * 80); print("5. 医学模型长推理豁免"); print("=" * 80)
check(_is_detailed_analysis("首先分析病位，" + "患" * 400) is True, "长推理输出 -> exempt")
check(_is_detailed_analysis("风痰瘀阻证") is False, "闭集标签 -> 不豁免")
check(_is_detailed_analysis("患" * 400) is False, "长但无推理特征 -> 不豁免")

print(); print("=" * 80); print("6. Task2/Task3 二分类输出契约（fendo 式 1/0）"); print("=" * 80)
for st in ["P1", "P2", "P3", "P4"]:
    p2 = P.build_t2(st, TEXT, "zh"); p2e = P.build_t2(st, TEXT, "en")
    check("输出1，否则输出0" in p2 and "不需要额外的文字说明" in p2,
          f"Task2 {st} 中文闭集 1/0 契约")
    check("output 1 directly" in p2e and "without any additional text" in p2e,
          f"Task2 {st} 英文闭集 1/0 契约")
    p3 = P.build_t3(st, TEXT, "zh"); p3e = P.build_t3(st, TEXT, "en")
    check("输出1，否则输出0" in p3 and "不需要额外的文字说明" in p3,
          f"Task3 {st} 中文闭集 1/0 契约")
    check("output 1 directly" in p3e and "without any additional text" in p3e,
          f"Task3 {st} 英文闭集 1/0 契约")

print(); print("=" * 80); print("7. Task3 分期判据（fendo 式阈值）"); print("=" * 80)
p3p3 = P.build_t3("P3", TEXT, "zh")
check("2 周以内" in p3p3 and "6 个月" in p3p3, "Task3 P3 写入分期阈值判据")
p3p4 = P.build_t3("P4", TEXT, "zh")
check("推理流程" in p3p4 and "提取发病时间" in p3p4, "Task3 P4 写入推理链")

print(); print("=" * 80)
if fails:
    print(f"FAILED: {len(fails)} 项"); [print("  -", f) for f in fails]; sys.exit(1)
print("ALL CHECKS PASSED")
