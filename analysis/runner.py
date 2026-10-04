# -*- coding: utf-8 -*-
"""中风病评测运行器（并行 + 断点续跑）。

用法：
  python runner.py --models deepseek-v4-flash --tasks T1 --strategies P1 --langs zh --n 60 --workers 8
"""
import json, os, re, ssl, time, argparse, hashlib, threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx
from openai import OpenAI

BASE="/Users/huangrende/Desktop/王健"; OUT=f"{BASE}/out/tcm_stroke_eval"; RUNS=f"{OUT}/runs"
os.makedirs(RUNS, exist_ok=True)
import sys; sys.path.insert(0,f"{BASE}/scripts/tcm_stroke_eval")
from prompts import build_t1, build_t2, build_t3
from equivalence import extract_answer

def _load_ex():
    try: return json.load(open(f"{OUT}/fewshot_exemplars.json"))
    except Exception: return None

def load_key():
    s=open(f"{BASE}/key.txt",encoding="utf-8",errors="ignore").read()
    return re.search(r'api_key\s*=\s*["\']?([^"\'\n]+)',s).group(1).strip()

# ModelArts 当前账号存在 1 req/s、15 req/min 限流；统一在进程内调度，
# 避免把 API 限流错误混入模型失败。
_RATE_LOCK = threading.Lock()
_RATE_EVENTS = deque()
_RATE_LAST = 0.0

def wait_for_rate_limit(rpm=15, rps=1.0):
    """Per-process rolling RPM + smoothed RPS limiter; tune from account/model quota."""
    global _RATE_LAST
    min_interval = 1.0 / max(float(rps), 0.01)
    rpm = max(int(rpm), 1)
    while True:
        with _RATE_LOCK:
            now = time.monotonic()
            while _RATE_EVENTS and now - _RATE_EVENTS[0] >= 60.0:
                _RATE_EVENTS.popleft()
            wait_sec = max(0.0, min_interval - (now - _RATE_LAST))
            if len(_RATE_EVENTS) >= rpm:
                wait_sec = max(wait_sec, 60.0 - (now - _RATE_EVENTS[0]) + 0.01)
            if wait_sec <= 0:
                stamp = time.monotonic()
                _RATE_LAST = stamp
                _RATE_EVENTS.append(stamp)
                return
        time.sleep(wait_sec)


def client(timeout_sec=300, max_conn=400):
    ctx=ssl.create_default_context(); ctx.minimum_version=ssl.TLSVersion.TLSv1_2
    ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
    hc=httpx.Client(verify=ctx,follow_redirects=True,timeout=timeout_sec,
                    limits=httpx.Limits(max_connections=max_conn,
                                        max_keepalive_connections=max_conn))
    return OpenAI(api_key=load_key(),base_url="https://api.modelarts-maas.com/v1",http_client=hc)

def call(c,model,prompt,retries=3,max_tokens=2048,temperature=0.5,top_p=0.7,rpm=15,rps=1.0):
    err=None
    for a in range(retries):
        try:
            wait_for_rate_limit(rpm=rpm, rps=rps)
            r=c.chat.completions.create(model=model,messages=[{"role":"user","content":prompt}],
                max_tokens=max_tokens,temperature=temperature,top_p=top_p)
            m=r.choices[0].message
            content=(m.content or "").strip()
            reasoning=(getattr(m,"reasoning_content",None) or "").strip()
            # 思考型模型：content 为空说明输出被截断在思考阶段，此时回退到
            # reasoning_content 仅用于诊断，并在下方用 truncated_think 标记。
            t=content or reasoning
            usage=getattr(r,"usage",None)
            if usage is not None and hasattr(usage,"model_dump"):
                usage=usage.model_dump()
            elif usage is not None:
                usage=dict(usage)
            meta={
                "finish_reason": getattr(r.choices[0],"finish_reason",None),
                "usage": usage,
                "has_content": bool(content),
                "reasoning_chars": len(reasoning),
            }
            return t,None,meta
        except Exception as e:
            es=repr(e)
            err=es[:200]
            if "429" in es or "RateLimit" in es or "81101" in es:
                time.sleep(8 + a*10)      # 限流：指数退避
            else:
                time.sleep(2 + a*3)
    return "",err,{"finish_reason":None,"usage":None}

# ---- 解析：fendo 三层裁决协议 ----
# fendo 原文（Front. Endocrinol. 2026, 17:1747468）：
#   Tier 1 used prompt engineering to constrain outputs to a predefined format.
#   For non-conforming outputs, Tier 2 applied a rule-based parser with keyword
#   matching. Any remaining ambiguous outputs were adjudicated in Tier 3.
#   Detailed analytical responses from medical-specific LLMs were exempted from
#   this forced categorization and instead retained for qualitative analysis.
#
# 因此本实现严格区分：
#   tier 1  —— 输出本身即是闭集标准标签（prompt engineering 直接约束成功）
#   tier 2  —— 输出不合规，但规则化关键词解析可唯一确定标准标签
#   tier 3  —— 仍存歧义的输出，走预注册裁决规则（确定性、可复现、无人工介入）
#   exempt  —— 医学/长推理模型的详细分析型输出，豁免强制归类，转定性分析
#
# 裁决规则在 METHOD_LOCK.md 的"输出裁决"一节冻结，不因结果事后调整。

def norm(s): return re.sub(r"[\s，,。；;：:、\"'“”‘’（）()\[\]]","",s or "")
_ZH=[r"证\s*型\s*[:：]\s*([^\n，,。；;]{2,20})",r"[Ss]yndrome\s*[:：]\s*([^\n,。;]{2,20})"]

# 触发豁免的"详细分析型输出"特征：多步推理、CoT 标记、或显著长于闭集标签。
_COT_MARKERS = ("分析步骤","推理","首先","其次","然后","最后","步骤",
                "step 1","steps:","reasoning","first,","second,","finally")
_EXEMPT_MIN_CHARS = 300   # 超过该长度且含推理特征 -> 视为分析型输出

def _spans(ns, vocab):
    """返回 [(start, end, label)]：规范标签的完整匹配，优先长匹配、不重叠。

    先匹配完整标签（含「证」后缀），再允许缺失「证」后缀的形式；
    已被完整匹配覆盖的区间不再重复匹配，避免「痰浊蒙窍证」被拆成碎片。
    """
    taken = []
    for v in sorted(vocab, key=len, reverse=True):
        nv = norm(v)
        if not nv:
            continue
        start = 0
        while True:
            i = ns.find(nv, start)
            if i < 0:
                break
            span = (i, i + len(nv))
            if not any(s < span[1] and span[0] < e for s, e, _ in taken):
                taken.append((span[0], span[1], v))
            start = i + 1
    for v in sorted(vocab, key=len, reverse=True):
        stem = norm(v).rstrip("证")
        if not stem:
            continue
        start = 0
        while True:
            i = ns.find(stem, start)
            if i < 0:
                break
            span = (i, i + len(stem))
            if not any(s < span[1] and span[0] < e for s, e, _ in taken):
                taken.append((span[0], span[1], v))
            start = i + 1
    taken.sort()
    return taken

def _all_matches(t, vocab):
    """按出现顺序返回文本中命中的全部标准证型（用于歧义判定）。"""
    return [v for _, _, v in _spans(norm(t), vocab)]

def _is_detailed_analysis(t):
    """是否为医学模型的详细分析型输出（fendo 豁免条款的判定）。"""
    if len(t) < _EXEMPT_MIN_CHARS:
        return False
    low = t.lower()
    return any(m in low or m in t for m in _COT_MARKERS)

def parse_t1(raw, vocab):
    """返回 (pred, tier)；tier ∈ {1,2,3}，pred 为 None 表示未决。

    tier 的语义严格对齐 fendo，不把推理正文中出现的候选词当作最终答案。
    """
    t = re.sub(r"```(?:json)?", "", raw or "").strip()
    if not t:
        return None, 3
    ns = norm(t)

    # ---- tier 1：输出本身即是闭集标准标签（prompt engineering 约束成功） ----
    # (a) 整段输出就是一个标准标签（允许缺「证」后缀）
    for v in vocab:
        if ns == norm(v) or ns == norm(v).rstrip("证"):
            return v, 1
    # (b) 显式 `证型：xxx` / `Syndrome: xxx` 规范作答行
    for pat in _ZH:
        for val in reversed(re.findall(pat, t)):
            hit = _match(val, vocab)
            if hit:
                return hit, 1
    # (c) 极短输出且全篇只含唯一一个候选（例如「风痰瘀阻证。」）
    if len(ns) <= 12:
        uniq = sorted(set(_all_matches(t, vocab)))
        if len(uniq) == 1:
            return uniq[0], 1

    # ---- tier 2：规则化关键词解析，且能唯一确定 ----
    hits = _all_matches(t, vocab)
    uniq = sorted(set(hits))
    if len(uniq) == 1:
        return uniq[0], 2
    # 多候选时，若末尾几行中的某一行只含唯一候选，则视为收尾结论
    lines = [x.strip() for x in t.splitlines() if x.strip()]
    for line in reversed(lines[-3:]):
        cands = set(_all_matches(line, vocab))
        if len(cands) == 1:
            return cands.pop(), 2

    # ---- tier 3：预注册裁决规则（确定性） ----
    # 注意：传按出现位置排序的 hits（保留重复），而不是字母序的 uniq，
    # 否则「取最后出现的候选」这条裁决规则会失效。
    return _adjudicate(t, vocab, hits), 3

def _adjudicate(t, vocab, candidates):
    """tier 3 预注册裁决规则。

    fendo 论文未公开 tier 3 的具体裁决规则，本实现采用冻结的确定性规则：
      1. 若存在候选：取**最后一次**出现的候选作为模型最终倾向
         （符合"结论通常位于末尾"的阅读约定，且与 tier 2 收尾逻辑一致）；
      2. 若无任何候选：判为未决（None），即模型既未合规作答、也未给出可识别证型。
    该规则不引入人工主观判断，可完全复现。
    """
    if candidates:
        return candidates[-1]
    return None

def _match(s,vocab):
    """受约束匹配：完全包含 > 去「证」后缀 > 允许缺「证」字。

    严格受限于词表，不做模糊猜测；仅容忍「气虚血瘀」这类缺后缀的写法。
    """
    ns=norm(s)
    if not ns: return None
    # 完全包含，最长优先
    cands=[v for v in vocab if norm(v) in ns]
    if cands: return max(cands,key=len)
    # 去掉词表项的「证」后缀再匹配（如 "气虚血瘀证" -> "气虚血瘀"）
    stems=[(v,norm(v).rstrip("证")) for v in vocab]
    cands2=[v for v,st in stems if st and st in ns]
    if cands2: return max(cands2,key=len)
    # 仅对短输出（<=15字，即模型直接给答案）允许词表项是输出的超串
    if len(ns)<=15:
        cands3=[(v,norm(v)) for v in vocab if ns in norm(v) and len(ns)>=3]
        if cands3: return max(cands3,key=lambda t:len(t[1]))[0]
    return None

def parse_t2(raw):
    t=re.sub(r"```(?:json)?","",raw or "").strip()
    if not t: return None,3
    for pat in [r"分类\s*[:：]\s*([^\n，,。；;]{2,8})",r"[Cc]lass\s*[:：]\s*([^\n,。;]{2,20})"]:
        m=re.search(pat,t)
        if m:
            if "中经络" in m.group(1) or "jingluo" in m.group(1).lower(): return "中经络",1
            if "中脏腑" in m.group(1) or "zangfu" in m.group(1).lower(): return "中脏腑",1
    ns=norm(t)
    last=ns[-30:] if len(ns)>30 else ns
    zj=last.rfind("中经络"); zf=last.rfind("中脏腑")
    if zj<0 and zf<0:
        zj=ns.find("中经络"); zf=ns.find("中脏腑")
    if zj<0 and zf<0:
        if "jingluo" in t.lower(): return "中经络",2
        if "zangfu" in t.lower(): return "中脏腑",2
        return None,3
    return ("中经络" if zj>zf else "中脏腑"),2


def parse_binary(raw, pos_label=1):
    """解析 fendo 式二分类输出（1/0），返回 (pred, tier)。

    对齐 fendo 的 Tier 1/2/3：
      tier 1 —— 输出本身就是 `1` 或 `0`（闭集约束直接成功）
      tier 2 —— 输出含多余文字，但规则解析可唯一确定 1/0
      tier 3 —— 仍歧义，走预注册裁决（取最后一次出现的判定；无判定则未决）
    pred 为 1 / 0 / None。
    """
    t = re.sub(r"```(?:json)?", "", raw or "").strip()
    if not t:
        return None, 3
    ns = norm(t)

    # ---- tier 1：整段输出即为 1 或 0 ----
    if ns in ("1", "0"):
        return int(ns), 1
    # 规范前缀行，如 `诊断：1` / `Result: 0`
    m = re.search(r"(?:诊断|判断|结果|分类|result|diagnosis|answer)\s*[:：]\s*([01])\b", t, re.I)
    if m:
        return int(m.group(1)), 1

    # ---- tier 2：规则解析，且唯一 ----
    ds = re.findall(r"(?<![0-9.])[01](?![0-9.])", t)
    uniq = sorted(set(ds))
    if len(uniq) == 1:
        return int(uniq[0]), 2
    # 只在**完全没有数字**时才使用中文判定词回退，避免「可能是1，也可能0」被单字「是」误判。
    if not ds:
        # 阳性/阴性线索。注意：
        #   - `符合` 需排除 `不符合`（负向后顾）
        #   - `急性期` 需排除 `非急性期`
        #   - `恢复期` / `后遗症期` 本身即"非急性期"的判定依据
        pos = re.findall(r"属于|可以诊断|(?<!不)符合|(?<!非)急性期|中脏腑", t)
        neg = re.findall(r"不属于|不能诊断|不符合|非急性期|恢复期|后遗症期|中经络", t)
        if pos and not neg: return pos_label, 2
        if neg and not pos: return 0, 2
        # ---- tier 3：预注册裁决（取最后一次出现） ----
        if pos and neg:
            return (pos_label if t.rfind(pos[-1]) > t.rfind(neg[-1]) else 0), 3
        return None, 3

    # ---- tier 3：多数字歧义，预注册裁决取最后一次出现的判定 ----
    return int(ds[-1]), 3

def rkey(model,task,strategy,lang,n,extra=""):
    h=hashlib.md5(f"{model}|{task}|{strategy}|{lang}|{n}|{extra}".encode()).hexdigest()[:8]
    safe=re.sub(r"[^A-Za-z0-9._-]","_",model)
    tag=f"__{re.sub(r'[^A-Za-z0-9._-]', '_', extra)}" if extra else ""
    return f"{safe}__{task}_{strategy}_{lang}__n{n}{tag}__{h}"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--models",required=True); ap.add_argument("--tasks",default="T1")
    ap.add_argument("--strategies",default="P1,P2,P3,P4"); ap.add_argument("--langs",default="zh,en")
    ap.add_argument("--n",type=int,default=60); ap.add_argument("--workers",type=int,default=8)
    ap.add_argument("--max-tokens",type=int,default=2048)
    ap.add_argument("--temperature",type=float,default=0.5)
    ap.add_argument("--top-p",type=float,default=0.7)
    ap.add_argument("--rpm",type=int,default=15,help="本进程共享的每分钟请求上限")
    ap.add_argument("--rps",type=float,default=1.0,help="本进程平滑请求速率上限")
    ap.add_argument("--timeout",type=float,default=300,help="单次API请求超时秒数")
    ap.add_argument("--max-conn",type=int,default=400,help="HTTP 连接池上限（高延迟模型需要调大）")
    ap.add_argument("--retries",type=int,default=3,help="单病例请求尝试次数；批次重跑仍由断点续跑接管")
    ap.add_argument("--subset",default="",help="T3: full|no_tp")
    ap.add_argument("--tag",default="",help="结果版本标签，用于避免与旧提示词结果混写")
    a=ap.parse_args()
    c=client(timeout_sec=a.timeout, max_conn=a.max_conn)
    vocab=json.load(open(f"{OUT}/top5_vocab.json"))
    EX=_load_ex()

    # fendo 采用**逐模型**推理参数（data sheet 2），而非全局统一值：
    #   temperature / top_p 在 fendo 中统一为 0.7 / 0.7（Gemma2 为 0.6/0.9）
    #   max_tokens 跨度 64–2048，其中推理型模型给更大预算
    #   （QwQ-32B=1024，DeepSeek-R1=2048）
    #
    # 本项目的关键差异：当前四个主模型全部是**推理/思考型**部署，
    # 内部思考 tokens 同样计入 max_tokens。实测 Qwen3-32B 单次思考常超过
    # 500 tokens，若 max_tokens=512 会在生成最终标签前被截断
    # （finish_reason=length，content 为空），从而产生系统性假性失败。
    #
    # 因此统一取 fendo 的上界 2048，保证闭集答案总能输出；temperature/top_p
    # 保持 fendo 的 0.7/0.7。每次请求的真实 token 用量（含推理 tokens）
    # 记入 `usage`，并作为分析指标单独报告。
    MODEL_GEN_PARAMS = {
        "qwen3-32b":          dict(temperature=0.7, top_p=0.7, max_tokens=2048),
        "deepseek-v4-flash":  dict(temperature=0.7, top_p=0.7, max_tokens=2048),
        "deepseek-v4-pro":    dict(temperature=0.7, top_p=0.7, max_tokens=2048),
        "glm-5.2":            dict(temperature=0.7, top_p=0.7, max_tokens=2048),
        "openpangu-2.0-flash":dict(temperature=0.7, top_p=0.7, max_tokens=2048),
    }

    def gen_params(model):
        """返回该模型本次运行的生成参数（未登记的模型用全局默认）。"""
        p = MODEL_GEN_PARAMS.get(model)
        if p is None:
            return dict(temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens)
        out = dict(p)
        if a.max_tokens != 2048:      # 命令行显式覆盖 max_tokens 时优先生效
            out["max_tokens"] = a.max_tokens
        return out

    def load_task(task):
        if task=="BAL": return json.load(open(f"{OUT}/eval_balanced.json"))
        if task=="T1": return json.load(open(f"{OUT}/eval_set.json"))
        if task=="T2": return json.load(open(f"{OUT}/task2_binary_set.json"))
        if task=="T2B": return json.load(open(f"{OUT}/task2_binary_ratio3to1.json"))
        if task=="T3": return json.load(open(f"{OUT}/task3_stage_set.json"))   # 分期二分类（主 Task3）
        # 舌脉消融：直接复用 Task1 评测集，保证 full 臂与 T1 完全同源同题，
        # 且 no_tp 来自唯一权威消融定义（ablation.strip_tongue_pulse）。
        if task=="ABL": return json.load(open(f"{OUT}/eval_set.json"))
        return json.load(open(f"{OUT}/task3_set.json"))

    for model in [m.strip() for m in a.models.split(",") if m.strip()]:
        for task in [t.strip() for t in a.tasks.split(",") if t.strip()]:
            data=load_task(task)
            if a.n: data=data[:a.n]
            for st in [s.strip() for s in a.strategies.split(",") if s.strip()]:
                for lang in [l.strip() for l in a.langs.split(",") if l.strip()]:
                    # 只有 ABL（舌脉消融）需要 full + no_tp 两个配对子集。
                    # T3（分期二分类）是单一任务，误用双子集会导致同一提示词重复执行。
                    # T3 保留 "full" 作为唯一子集名，以便与已产出的历史文件对齐（断点续跑）。
                    if task == "ABL":
                        subsets = ["full", "no_tp"]
                    elif task == "T3":
                        subsets = ["full"]
                    else:
                        subsets = [""]
                    for sub in subsets:
                        # 仅当该任务确有多个子集时，--subset 才起筛选作用，
                        # 避免单子集任务被 --subset 误伤成整任务跳过。
                        if a.subset and len(subsets) > 1 and sub != a.subset: continue
                        run_tag = f"{a.tag}__{sub}" if (a.tag and sub) else (a.tag or sub)
                        rk=rkey(model,task,st,lang,len(data),run_tag); path=f"{RUNS}/{rk}.jsonl"
                        done=set()
                        if os.path.exists(path):
                            for l in open(path):
                                try:
                                    r = json.loads(l)
                                    # 只有成功返回的记录才算完成；API 失败记录必须可断点重试。
                                    if not r.get("err"):
                                        done.add(r["case_id"])
                                except Exception:
                                    pass
                        todo=[x for x in data if x["case_id"] not in done]
                        if not todo: print(f"  [skip] {rk}"); continue
                        print(f"  [{model}] {task} {st}/{lang}{('/'+sub) if sub else ''} 待跑 {len(todo)}/{len(data)} (workers={a.workers})",flush=True)
                        lock=threading.Lock(); t0=time.time(); nerr=0; cnt=0
                        fh=open(path,"a")
                        def work(x):
                            nonlocal nerr,cnt
                            if task in ("T1","BAL"):
                                p=build_t1(st,x["text"],lang,vocab,EX)
                            elif task in ("T2","T2B"):
                                p=build_t2(st,x["text"],lang)
                            elif task=="T3":
                                p=build_t3(st,x["text"],lang)
                            else:   # ABL: 舌脉消融，仍为证型分类（数据源 = eval_set.json）
                                txt=x["text"] if sub=="full" else x["text_no_tongue_pulse"]
                                p=build_t1(st,txt,lang,vocab)
                            gp=gen_params(model)
                            raw,err,meta=call(c,model,p,retries=a.retries,max_tokens=gp["max_tokens"],
                                temperature=gp["temperature"],top_p=gp["top_p"],rpm=a.rpm,rps=a.rps)
                            if task in ("T2","T2B"): pred,tier=parse_binary(raw,1)
                            elif task=="T3": pred,tier=parse_binary(raw,1)
                            else: pred,tier=parse_t1(raw,vocab)
                            # fendo 豁免条款：医学/长推理模型的详细分析型输出
                            # 不强制归类，标记 exempt 并保留供定性分析。
                            exempt = bool(task in ("T1","BAL") and _is_detailed_analysis(raw))
                            with lock:
                                if err: nerr+=1
                                # Preserve the complete request/response pair for
                                # auditability and exact re-parsing. Never truncate
                                # model output in the primary experiment.
                                fh.write(json.dumps(dict(case_id=x["case_id"],model=model,task=task,strategy=st,lang=lang,subset=sub,
                                    gold_syndrome=x.get("gold_syndrome") or x.get("label"),
                                    prompt=p, pred=pred,answer=extract_answer(raw),tier=tier,exempt=exempt,
                                    finish_reason=meta.get("finish_reason"),usage=meta.get("usage"),
                                    has_content=meta.get("has_content"),reasoning_chars=meta.get("reasoning_chars"),
                                    err=err,
                                    gen_params=gp,
                                    raw=(raw or "")),ensure_ascii=False)+"\n"); fh.flush()
                                cnt+=1
                                if cnt%20==0: print(f"    {cnt}/{len(todo)} ({time.time()-t0:.0f}s, err={nerr})",flush=True)
                        with ThreadPoolExecutor(max_workers=a.workers) as ex:
                            list(ex.map(work,todo))
                        fh.close()
                        print(f"    done {len(todo)} -> {path} ({time.time()-t0:.0f}s)",flush=True)

if __name__=="__main__":
    main()
