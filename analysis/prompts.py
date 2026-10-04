# -*- coding: utf-8 -*-
"""中风病评测提示构造器（4 策略 × 中英 × 3 任务）。

严格按 fendo 补充材料 data sheet 3 的结构复刻：

    input_text = f"入院时情况：{admission_condition}\n"
                 f"住院经过：{hospitalization_process}\n"
                 f"以上是患者的入院时情况和住院经过，如果你是一位专业医生，"
                 f"{question}\n"
                 f"{additional_prompt}"

要点（与 fendo 原文一致）：
  1. `additional_prompt` 是固定的闭集输出约束，P1–P4 共用同一条；
  2. 该约束要求模型**直接输出一个标准标签且不得添加其他文字**；
  3. P1–P4 之间只有 `question` 不同，即引导强度不同。

参考：`refs/fendo_supplementary/data_sheet_3.docx`
"""
import json, os
BASE = "/Users/huangrende/Desktop/王健"

_KNOW = None
def know():
    global _KNOW
    if _KNOW is None:
        _KNOW = {}
        p = f"{BASE}/data/tcm_sd_full_20260923/syndrome_knowledge.json"
        for l in open(p, encoding="utf-8"):
            l = l.strip()
            if l:
                r = json.loads(l); _KNOW[r["Name"]] = r
    return _KNOW

def _defs(syns, maxlen=200):
    out = []
    for s in syns:
        r = know().get(s)
        d = (r.get("Definition") or "").strip().replace("\n", "") if r else ""
        out.append(f"- {s}：{d[:maxlen]}")
    return "\n".join(out)

# ---------------- Task 1: 5 类证型辨识 ----------------
# 严格对齐版：P1-P4 共用同一个 additional_prompt（闭集输出约束），
# 只改变 question 的引导强度。P5/P6/P7 是额外的标准化干预。


def additional_prompt(vocab, lang="zh"):
    """fendo `additional_prompt` 的中医版：固定闭集 + 禁止冗余输出。

    对应 fendo 原文：
      zh: 直接给出一个你诊断出的糖尿病类型，输出时只能选择这四种病型的其中一种：
          1型糖尿病，2型糖尿病，特殊类型糖尿病，妊娠期高血糖，且不要添加其他文字。
      en: You must choose exactly one option from the following: ... .
          Directly output your choice without anything else. No additional explanation is needed.

    证型名保留中文写法（金标准 `norm_syndrome` 为中文），以保证中英两套实验
    使用同一个可评分的标签空间。
    """
    names = "，".join(vocab)
    if lang == "zh":
        return (f"直接给出一个你诊断出的中医证型，输出时只能选择这五种证型的其中一种："
                f"{names}，且不要添加其他文字。")
    return ("You must choose exactly one option from the following: "
            + ", ".join(vocab) + ". "
            "Directly output your choice without anything else. No additional explanation is needed.")


def _wrap(text, question, vocab, lang="zh"):
    """按 fendo 的 input_text 模板拼接病例、医生角色设定、问题与输出约束。"""
    if lang == "zh":
        return (f"病历：{text}\n"
                f"以上是患者的病历，如果你是一位专业医生，"
                f"{question}\n"
                f"{additional_prompt(vocab, lang)}")
    return (f"Medical record: {text}\n"
            f"The above is the patient's medical record. If you are a professional doctor, "
            f"{question}\n"
            f"{additional_prompt(vocab, lang)}")


def _wrap_plain(text, question, extra, lang="zh"):
    """fendo 的通用 input_text 模板（Task2/Task3 用，输出约束可自定义）。"""
    if lang == "zh":
        return (f"病历：{text}\n"
                f"以上是患者的病历，如果你是一位专业医生，"
                f"{question}\n"
                f"{extra}")
    return (f"Medical record: {text}\n"
            f"The above is the patient's medical record. If you are a professional doctor, "
            f"{question}\n"
            f"{extra}")


def _criteria(vocab, lang="zh"):
    # 仅使用评测集之外的证型知识库/临床定义，提供类别特异的支持特征和鉴别点。
    items = {
        "气虚血瘀证": "支持：久病/年老/乏力、肢体无力，面色淡或少气懒言，舌淡或淡紫，脉细涩/无力；鉴别：以气虚为本并有血瘀表现，不以痰湿或神志昏蒙为主。",
        "风痰瘀阻证": "支持：半身不遂、口眼歪斜、言语謇涩，痰多或苔腻，舌暗/紫，脉弦滑或涩；鉴别：风、痰、瘀三者共同阻络，较风痰入络更突出瘀阻和半身不遂。",
        "风痰入络证": "支持：肢体麻木不仁、眩晕、口角流涎、感觉减退或轻瘫，苔腻，脉弦滑；鉴别：以风痰阻络和麻木为主，通常不以明显神志障碍为核心。",
        "痰浊蒙窍证": "支持：意识模糊、神志昏蒙或昏迷、语言不清、痰声/痰涎，苔白腻，脉滑；鉴别：核心是痰浊蒙闭清窍和神志异常。",
        "肝阳上亢证": "支持：头晕头痛、面红、急躁易怒、耳鸣，舌红，脉弦有力；鉴别：以肝阳偏亢扰动头目为主，不以痰浊蒙窍或明显气虚为主。",
    }
    if lang == "zh":
        return "\n".join(f"- {s}：{items.get(s, '')}" for s in vocab)
    # 英文实验沿用同一结构，证型名仍保留中文以便统一评分。
    return "\n".join(f"- {s}: {items.get(s, '')}" for s in vocab)


def _p4_reasoning_chain(lang="zh"):
    """P4 的 CoT 推理链，对应 fendo P4 中嵌入问句前的完整诊断流程。

    fendo 的 P4 不是“请按以下步骤分析”，而是把权威标准的完整推理流程
    写入 question，再以“按照上述诊断思维链的诊断流程……”收尾。
    """
    if lang == "zh":
        return (
            "根据中风病中医辨证流程，证型辨识的推理如下：\n"
            "1) 首先判断病位：神志清楚、仅见半身不遂、口舌歪斜、言语不利者，病位在经络；"
            "若见神志昏蒙、昏迷等神志障碍，病位在脏腑。\n"
            "2) 其次判断病性要素：结合起病形式与病程辨别风、痰、瘀、虚、火/阳亢等病性；"
            "急性起病、肢体动摇者多属风；苔腻、痰多、言语謇涩者多属痰；"
            "舌暗紫、痛有定处者多属瘀；久病、乏力、气短、脉细弱者多属虚；"
            "头晕头痛、面红、急躁易怒、脉弦有力者多属火/阳亢。\n"
            "3) 再结合神志、肢体与言语表现：肢体麻木不仁、感觉减退为主者提示风痰入络；"
            "半身不遂、口眼歪斜、言语謇涩并见痰瘀者提示风痰瘀阻；"
            "意识模糊、神志昏蒙、痰声者提示痰浊蒙窍。\n"
            "4) 最后综合舌象与脉象：舌淡或淡紫、脉细涩/无力支持气虚血瘀；"
            "舌暗紫、苔腻、脉弦滑或涩支持风痰瘀阻；苔腻、脉弦滑支持风痰入络；"
            "苔白腻、脉滑支持痰浊蒙窍；舌红、脉弦有力支持肝阳上亢。\n"
            "按照上述辨证思维链的流程，该患者应诊断为哪种中医证型？"
        )
    return (
        "According to the TCM stroke syndrome differentiation process, the reasoning is as follows:\n"
        "1) First determine the disease location: if consciousness is clear and only hemiplegia, "
        "deviated mouth and tongue, or speech difficulty is present, the location is the meridians; "
        "if clouded consciousness or coma is present, the location is the organs.\n"
        "2) Then determine the pathogenic nature: distinguish wind, phlegm, stasis, deficiency, "
        "and fire/yang hyperactivity from onset and course; acute onset with tremor suggests wind; "
        "greasy coating, profuse phlegm, and unclear speech suggest phlegm; "
        "dark purple tongue and fixed pain suggest stasis; chronic course, fatigue, shortness of breath, "
        "and weak fine pulse suggest deficiency; dizziness, headache, flushed face, irritability, "
        "and a forceful stringy pulse suggest fire/yang hyperactivity.\n"
        "3) Then integrate consciousness, limb, and speech findings: numbness and reduced sensation "
        "suggest wind-phlegm entering the collaterals; hemiplegia with deviated mouth and unclear speech "
        "plus phlegm and stasis suggests wind-phlegm obstructing the collaterals; "
        "clouded consciousness and phlegm sounds suggest turbid phlegm misting the orifices.\n"
        "4) Finally integrate tongue and pulse: pale or pale-purple tongue with weak fine pulse supports "
        "qi deficiency and blood stasis; dark purple tongue with greasy coating and stringy slippery pulse "
        "supports wind-phlegm with stasis; greasy coating with stringy slippery pulse supports "
        "wind-phlegm entering the collaterals; white greasy coating with slippery pulse supports "
        "turbid phlegm misting the orifices; red tongue with forceful stringy pulse supports "
        "liver yang hyperactivity.\n"
        "Based on the above differentiation reasoning, what TCM syndrome should this patient be diagnosed with?"
    )


def t1_p0(text, lang):
    """开放世界基线：不提供候选证型，但保持与 P1 相同的最终输出协议。"""
    if lang == "zh":
        return ("你是中医临床专家。请根据下面的中风病住院病历判断最符合的中医证型。\n\n"
                f"【病历】\n{text}\n\n"
                "可以使用你认为最准确的证型名称，不限制为预先给定的类别。"
                "最后一行必须严格写成：证型：<你的证型判断>。")
    return ("You are a TCM clinical expert. Determine the most appropriate syndrome from the stroke record below.\n\n"
            f"【Record】\n{text}\n\n"
            "You may use the syndrome name you consider most appropriate; no predefined label list is provided. "
            "The last line must be exactly: Syndrome: <your syndrome judgment>.")


def t1_p1(text, lang, vocab):
    """P1 Zero-shot direct query，对应 data sheet 3 的 prompt[1]。"""
    if lang == "zh":
        return _wrap(text, "该患者应诊断为哪种中医证型？", vocab, lang)
    return _wrap(text, "What TCM syndrome should this patient be diagnosed with?", vocab, lang)


def t1_p2(text, lang, vocab):
    """P2 Reference-guided，对应 data sheet 3 的 prompt[2]。

    fendo 的 P2 只在问句中**点名权威标准**（2025 ADA 建议），并不把标准内容
    贴进 prompt；内容层面的知识供给从 P3 才开始。
    """
    if lang == "zh":
        return _wrap(text, "根据《中风病中医诊疗指南》的证型分类标准，患者的证型属于哪一类型？", vocab, lang)
    return _wrap(text, "According to the TCM guideline for stroke syndrome classification, "
                       "which syndrome type does the patient have?", vocab, lang)


def t1_p3(text, lang, vocab):
    """P3 Criteria-driven，对应 data sheet 3 的 prompt[3]：把类别特异标准写入 question。"""
    crit = _criteria(vocab, lang)
    if lang == "zh":
        q = ("根据中风病中医辨证，各证型具有以下特征：\n"
             f"{crit}\n\n"
             "根据以上特征，该患者应诊断为哪种中医证型？")
        return _wrap(text, q, vocab, lang)
    q = ("According to TCM stroke syndrome differentiation, the syndrome types have the following characteristics:\n"
         f"{crit}\n\n"
         "Based on the above characteristics, which TCM syndrome should this patient be diagnosed with?")
    return _wrap(text, q, vocab, lang)


def t1_p4(text, lang, vocab):
    """P4 CoT，对应 data sheet 3 的 prompt[4]：把完整推理链写入 question 后再提问。"""
    return _wrap(text, _p4_reasoning_chain(lang), vocab, lang)


def build_t1(strategy, text, lang, vocab, exemplars=None):
    if strategy == "P0": return t1_p0(text, lang)
    if strategy == "P1": return t1_p1(text, lang, vocab)
    if strategy == "P2": return t1_p2(text, lang, vocab)
    if strategy == "P3": return t1_p3(text, lang, vocab)
    if strategy == "P5": return t1_p5(text, lang, vocab)
    if strategy == "P6": return t1_p6(text, lang, vocab)
    if strategy == "P7": return t1_p7(text, lang, vocab, exemplars)
    return t1_p4(text, lang, vocab)

# ---------------- Task 2: 中经络 vs 中脏腑（规则型二分类）----------------
# 对齐 fendo data sheet 3：二分类任务输出闭集 1/0，问句为"是否……"。
#   DKD : 患者是否诊断糖尿病肾病？       1=可诊断 / 0=否则
#   MetS: 患者是否诊断代谢综合征？       1=可诊断 / 0=否则
# 本项目同构改写为"是否属于 X"：
#   T2 : 患者是否属于中脏腑？            1=中脏腑 / 0=中经络
#   T3 : 患者是否处于急性期？            1=急性期 / 0=非急性期
_T2_EXTRA = {
    "zh": "只输出你的判断结果：如果属于中脏腑，直接输出1，否则输出0。直接输出1或0，不需要额外的文字说明。",
    "en": "Output only your result: if the patient belongs to Zhongzangfu, output 1 directly; "
          "otherwise output 0. Output only 1 or 0, without any additional text.",
}

def t2_p1(text, lang):
    """P1 Zero-shot，对应 fendo Task2/Task3 的 prompt[1]。"""
    q = ("该患者是否属于中风病中脏腑？" if lang == "zh"
         else "Does this patient belong to the Zhongzangfu type of stroke?")
    return _wrap_plain(text, q, _T2_EXTRA[lang], lang)

def t2_p2(text, lang):
    """P2 Reference-guided：只点名标准，不展开内容。"""
    q = ("根据中风病中医诊疗指南关于中经络与中脏腑的划分标准，该患者是否属于中脏腑？" if lang == "zh"
         else "According to the TCM stroke guideline distinction between meridian and organ types, "
              "does this patient belong to Zhongzangfu?")
    return _wrap_plain(text, q, _T2_EXTRA[lang], lang)

def t2_p3(text, lang):
    """P3 Criteria-driven：把判定标准写入 question。"""
    if lang == "zh":
        q = ("根据中风病中医判定标准：中经络为神志清楚，仅见半身不遂、口舌歪斜、言语不利等，"
             "无神志障碍；中脏腑为神志昏蒙或昏迷，可伴半身不遂、口舌歪斜。\n"
             "根据以上标准，该患者是否属于中脏腑？")
    else:
        q = ("According to TCM stroke criteria: Zhongjingluo means clear consciousness with only "
             "hemiplegia, deviated mouth and tongue, or speech difficulty, without consciousness "
             "impairment; Zhongzangfu means clouded consciousness or coma, possibly with hemiplegia "
             "and deviated mouth and tongue.\n"
             "Based on the above criteria, does this patient belong to Zhongzangfu?")
    return _wrap_plain(text, q, _T2_EXTRA[lang], lang)

def t2_p4(text, lang):
    """P4 CoT：把判别推理链写入 question。"""
    if lang == "zh":
        q = ("根据中风病中医分类的推理流程如下：\n"
             "1) 首先提取病历中与神志相关的描述；\n"
             "2) 其次提取肢体与言语症状；\n"
             "3) 最后依据「有无神志障碍」判定：神志清楚者为中经络，神志昏蒙或昏迷者为中脏腑。\n"
             "按照上述推理流程，该患者是否属于中脏腑？")
    else:
        q = ("The reasoning process for TCM stroke classification is as follows:\n"
             "1) First extract consciousness-related descriptions from the record;\n"
             "2) Then extract limb and speech signs;\n"
             "3) Finally classify by consciousness impairment: clear consciousness indicates "
             "Zhongjingluo, while clouded consciousness or coma indicates Zhongzangfu.\n"
             "Based on the above reasoning, does this patient belong to Zhongzangfu?")
    return _wrap_plain(text, q, _T2_EXTRA[lang], lang)


# ---------------- Task 3: 急性期 vs 非急性期（规则型二分类）----------------
_T3_EXTRA = {
    "zh": "只输出你的判断结果：如果处于急性期，直接输出1，否则输出0。直接输出1或0，不需要额外的文字说明。",
    "en": "Output only your result: if the patient is in the acute phase, output 1 directly; "
          "otherwise output 0. Output only 1 or 0, without any additional text.",
}

def t3_p1(text, lang):
    q = ("该患者是否处于中风病急性期？" if lang == "zh"
         else "Is this patient in the acute phase of stroke?")
    return _wrap_plain(text, q, _T3_EXTRA[lang], lang)

def t3_p2(text, lang):
    q = ("根据中风病中医诊疗指南关于分期的标准，该患者是否处于急性期？" if lang == "zh"
         else "According to the TCM stroke guideline staging criteria, "
              "is this patient in the acute phase?")
    return _wrap_plain(text, q, _T3_EXTRA[lang], lang)

def t3_p3(text, lang):
    if lang == "zh":
        q = ("根据中风病中医分期标准：急性期为发病 2 周以内；恢复期为发病 2 周至 6 个月；"
             "后遗症期为发病 6 个月以上。\n"
             "请根据病历中的发病时间判断，该患者是否处于急性期？")
    else:
        q = ("According to TCM stroke staging criteria: the acute phase is within 2 weeks of onset; "
             "the recovery phase is 2 weeks to 6 months; the sequelae phase is beyond 6 months.\n"
             "Based on the onset time documented in the record, is this patient in the acute phase?")
    return _wrap_plain(text, q, _T3_EXTRA[lang], lang)

def t3_p4(text, lang):
    if lang == "zh":
        q = ("根据中风病中医分期的推理流程如下：\n"
             "1) 首先从主诉和现病史中提取发病时间；\n"
             "2) 其次将发病时间与分期阈值比较：2 周以内为急性期，2 周至 6 个月为恢复期，"
             "6 个月以上为后遗症期；\n"
             "3) 最后依据上述分层判断患者当前所处的时期。\n"
             "按照上述推理流程，该患者是否处于急性期？")
    else:
        q = ("The reasoning process for TCM stroke staging is as follows:\n"
             "1) First extract the onset time from the chief complaint and history of present illness;\n"
             "2) Then compare the onset time with the staging thresholds: within 2 weeks is the acute "
             "phase, 2 weeks to 6 months is the recovery phase, and beyond 6 months is the sequelae phase;\n"
             "3) Finally determine the patient's current phase.\n"
             "Based on the above reasoning, is this patient in the acute phase?")
    return _wrap_plain(text, q, _T3_EXTRA[lang], lang)

def build_t3(strategy, text, lang):
    return {"P1": t3_p1, "P2": t3_p2, "P3": t3_p3, "P4": t3_p4}[strategy](text, lang)

def build_t2(strategy, text, lang):
    return {"P1": t2_p1, "P2": t2_p2, "P3": t2_p3, "P4": t2_p4}[strategy](text, lang)

# ---------------- 因果验证策略 P5/P6（命名规范化的效果）----------------
# 设计：控制变量，只改"输出侧术语规范"这一个因素。
#   P5 闭集约束  : 强制从 5 个标准证型名中选择（只改命名约束）
#   P6 闭集+定义 : P5 + 给出每个标准证型的规范定义（额外供给知识）

def t1_p5(text, lang, vocab):
    cands = "、".join(vocab) if lang == "zh" else ", ".join(vocab)
    if lang == "zh":
        return ("你是中医临床专家。请阅读下面的中风病住院病历，并**从下列标准证型中选出唯一的一个**。\n"
                f"【标准证型（必须从中选择，不得使用其他写法）】\n{cands}\n\n"
                f"【病历】\n{text}\n\n只能输出上述标准证型之一，不要输出其他任何内容。")
    return ("You are a TCM clinical expert. Read the following stroke record and choose exactly "
            "one syndrome **from the standard list below**.\n"
            f"【Standard syndromes (you MUST choose one of these, no other wording)】\n{cands}\n\n"
            f"【Record】\n{text}\n\nOutput only one of the standard syndrome names, nothing else.")

def t1_p6(text, lang, vocab):
    b = _defs(vocab)
    cands = "、".join(vocab) if lang == "zh" else ", ".join(vocab)
    if lang == "zh":
        return ("你是中医临床专家。请依据下列标准证型的规范定义，为中风病病历判断证型。\n"
                f"【标准证型及其定义】\n{b}\n\n"
                f"【强制要求】只能从以下名称中选择：{cands}\n\n"
                f"【病历】\n{text}\n\n只输出标准证型名称，不要输出其他任何内容。")
    return ("You are a TCM clinical expert. Using the standard syndrome definitions below, "
            "determine the syndrome of this stroke record.\n"
            f"【Standard syndromes with definitions】\n{b}\n\n"
            f"【Constraint】You MUST choose only from: {cands}\n\n"
            f"【Record】\n{text}\n\nOutput only the standard syndrome name, nothing else.")

# ---------------- P7: Few-shot（每类给 1 个真实范例）----------------
def t1_p7(text, lang, vocab, exemplars=None):
    """Few-shot：从训练集为每个证型取 1 个真实病历片段作为范例。"""
    if not exemplars:  # 无范例则退化为 P5
        return t1_p5(text, lang, vocab)
    blk = []
    for syn, ex in exemplars:
        blk.append(f"【范例·{syn}】\n{ex}\n→ 证型：{syn}")
    demo = "\n\n".join(blk)
    cands = "、".join(vocab) if lang == "zh" else ", ".join(vocab)
    if lang == "zh":
        return ("你是中医临床专家。下面先给出若干标注范例，请据此为新的中风病病历判断证型。\n\n"
                f"=== 标注范例 ===\n{demo}\n\n=== 待判断病历 ===\n{text}\n\n"
                f"【要求】只能从以下标准证型中选择：{cands}\n"
                "最后一行以「证型：」开头输出证型名称。")
    return ("You are a TCM clinical expert. Study the labeled examples, then classify the new "
            f"stroke record.\n\n=== Examples ===\n{demo}\n\n=== Record ===\n{text}\n\n"
            f"【Constraint】Choose only from: {cands}\n"
            "Last line: 'Syndrome: ' followed by the name.")
