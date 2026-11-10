"""多跳桥接实体通路（`multihop`，「A 经一个中间实体连到 B」）。

## 这是什么

问句给出两个「」锚点 A、B，问它们之间的中间实体。本模块在**本次上下文已有的断言**
里找同时与 A、B 相连的实体 X：若这样的 X **唯一**，就作答 `X。` + `X—A`、`X—B`
两条真实三元组句（句源是断言的 `subject/predicate/object`），并附两条指向对应块、
偏移逐字自洽的引用。

## 三条必须守住的约束（都来自被否决的版本）

1. **候选只许来自 `context["units"]` 的断言。** 早前一版用全图公共邻居，对抗审查实测
   41.7% 的增益来自"本次根本没召回的图的边"，判为"图查询冒充检索"。本模块拿不到
   全图索引，结构上就不可能重犯。
2. **候选不唯一就不答。** 早前一版按 `(-关系种类数, -support, 度)` 排序取 top1，
   那套排序键在 dev 上是**负收益**（候选内随机 21 > 排序 top1 19），且 41.3% 的触发题
   候选本就唯一、排序在数学上无从贡献。本模块只做"唯一才答"，**没有任何排序键**，
   因此也不存在"按 dev 扫出来的阈值"。
3. **句源必须是真实断言。** 早前一版靠 `{锚点}与{X}相关。` 这类编造句骗共现判据，
   纯模板与真实三元组的 marginal 差 0 题——那是注水。本模块每条答案句都来自一条
   assertion，引用指向该 assertion 的宿主块。

## 触发面

问句含 `「中间实体」` 且恰好 2 个「」引文。全 640 题里只命中 `multihop` 96/96，
其余七类 **0**（对抗审查 Q5 PASS）。

## 已知天花板

两个锚点都有真实原文共现证据的题只有 21/96；本通路靠三元组句作答，因此上限受
"上下文里能不能凑出 X—A、X—B 两条断言"约束。dev 实测 26/57（基线 8/57）。

## W2：`policy="top1_margin"`（**非默认**，默认仍是 `unique_only`）

约束 2 的"候选不唯一就不答"在**当前接地候选集**上被实测证明是主要瓶颈，因此
新增一个策略参数。**默认取 `unique_only` 时与本文档描述的行为逐位相同**：
候选多于一个仍然直接 `ambiguous`，一个排序键都不算。

为什么约束 2 的原始读数不能直接搬来做结论（红线 23 的作用域过期）：

* 约束 2 引用的"候选内随机 21 > 排序 top1 19"是在**全图公共邻居**候选集上量的；
* 在**当前接地候选集**上（gold ∈ 候选集 96/96）：

  | 指标 | dev 57 | holdout 39 | clean 24 |
  |---|---|---|---|
  | 候选唯一（现行作答条件） | 25 (43.9%) | 10 (25.6%) | 6 (25.0%) |
  | gold 按 support 排 top-1 | 50 (87.7%) | 31 (79.5%) | 19 (79.2%) |
  | gold 按 support 排 top-2 | 56 (98.2%) | 38 (97.4%) | 23 (95.8%) |

  均匀随机期望 `Σ 1/k_q = 37.43`，观测 top-1 = 50，配对差 **+12.57**。
  这是"当前候选集 + 当前排序键"的**描述性**增益，不能与旧全图候选集上的 19
  做因果比较，也不能据此说红线 23 被推翻——只能说它的作用域不覆盖这里。

所以新策略**不是**"无脑 top-1"：`support` 差值是计数量、没有概率语义，
必须先校准成 `P(gold = top1 | 特征)`（`bridge_calib`），再按预登记代价比
（`c_误答=2`、`c_误拒=1`）判断作答的期望收益是否为正。阈值只有概率阈值
`2/3`，代码里**没有**任何"support 差 ≥ N 就作答"的阈值。校准表由
`bridge_calib.fit()` 从落盘样本拟合，映射不存在或桶里没样本 ⇒ 拒答
（落回原链路，不猜）。并列（top-1 与 top-2 的完整排序键相同）一律拒答，
不许用名字字典序把并列伪装成确定的 top-1。
"""
from __future__ import annotations

import re

from . import answer as answer_mod
from . import bridge_calib

# 触发词。与 `query.analyze_query` 的锚点抽取同为「」内文本，但这里刻意只认
# 「中间实体」这一个短语：实测它把触发面收窄到 multihop 一类（96/96），
# 换成"≥2 个锚点"会误伤 direct_relation 96 道、"≥2 锚点且问谁/什么"仍误伤 51 道。
BRIDGE_TRIGGER = "中间实体"

# 引用证据的字符下限，直接沿用 `answer` 的常量：两处写同一个数会让我们改一处忘
# 另一处，而这里正是"短证据被 verify_citations 整条剔除"那个失效模式的现场。
MIN_EVIDENCE_CHARS = answer_mod.MIN_EVIDENCE_CHARS

_QUOTE_RE = re.compile(r"「([^」]{1,200})」")


def anchors_of(question: str) -> list[str]:
    """问句里的「」锚点（保持出现顺序，去空）。"""
    return [m.strip() for m in _QUOTE_RE.findall(str(question or "")) if m.strip()]


def is_bridge_question(question: str) -> bool:
    q = str(question or "")
    return BRIDGE_TRIGGER in q and len(anchors_of(q)) == 2


#: `policy` 的取值。`unique_only` 是默认值 = 改造前行为。
UNIQUE_ONLY = "unique_only"
TOP1_MARGIN = "top1_margin"
POLICIES = (UNIQUE_ONLY, TOP1_MARGIN)
DEFAULT_POLICY = UNIQUE_ONLY


def _display_name(display, entity_id) -> str:
    raw = str(entity_id or "")
    return str(display(raw)) if display is not None else raw


def _grounded_triples(units, chunk_assertions=None, display=None) -> dict:
    """把上下文块的断言收成 `(s,p,o) -> (chunk_id, evidence, start, end)`。

    抽成独立函数是为了让 W2.1 的诊断与生产**共用同一个候选构造器**：计划书
    明令"禁止诊断和生产各写一套"。这个函数只做"上下文里的断言 → 可校验三元组"，
    不含任何排序、不含任何策略。

    三种情况被刻意剔掉，原因都写在原处（它们各自对应一次实测事故）：
    定位不到 evidence、evidence 短于 `MIN_EVIDENCE_CHARS`、自环。
    """
    tri: dict[tuple[str, str, str], tuple[str, str, int, int]] = {}
    for unit in units or ():
        cid = str(unit.get("chunk_id") or "")
        body = str(unit.get("text") or "")
        if not cid or not body:
            continue
        if chunk_assertions is not None:
            records = [
                {"subject": _display_name(display, r.get("subject_id")),
                 "predicate": str(r.get("predicate") or ""),
                 "object": _display_name(display, r.get("object_id")),
                 "evidence": str(r.get("evidence_text") or "")}
                for r in (chunk_assertions(cid) or ())
            ]
        else:
            records = list(unit.get("assertions") or ())
        for a in records:
            s = str(a.get("subject") or "").strip()
            p = str(a.get("predicate") or "").strip()
            o = str(a.get("object") or "").strip()
            ev = str(a.get("evidence") or "")
            if not s or not p or not o or s == o:
                continue
            start = body.find(ev) if ev else -1
            if start < 0 or len(ev) < MIN_EVIDENCE_CHARS:
                # 定位不到、或短于 `MIN_EVIDENCE_CHARS` 的断言句不能进来：它发不出
                # 可校验引用，硬塞会让整条答案被 `verify_citations` 清空
                # （实测 dev 上 56 条引用因此被判废、答案整体转拒答）。
                continue
            tri.setdefault((s, p, o), (cid, ev, start, start + len(ev)))
    return tri


def _neighbourhood(tri: dict) -> dict:
    """三元组 → 无向邻居表 `实体 -> [(对端, 谓词)]`（保持插入序）。"""
    neigh: dict[str, list[tuple[str, str]]] = {}
    for s, p, o in tri:
        neigh.setdefault(s, []).append((o, p))
        neigh.setdefault(o, []).append((s, p))
    return neigh


def _common_neighbours(neigh: dict, anc: list[str]) -> set[str]:
    """同时连两个锚点的实体集合，减掉锚点自身。"""
    a_n = {o for o, _ in neigh.get(anc[0], ())}
    b_n = {o for o, _ in neigh.get(anc[1], ())}
    return (a_n & b_n) - set(anc)


def candidate_support(tri: dict, neigh: dict, anc: list[str], x: str) -> dict:
    """按候选汇总"它连接两个锚点的有效断言"——W2.1 的 support 特征。

    这里的 `support_sum` 是**接地断言条数**（`tri` 的每个键就是一条去重后的
    `(s,p,o)`，且各自带一条可校验 evidence）。图上那条边的 `support_count`
    在本模块拿不到，也不能去拿——本模块只许看见本次上下文的断言（约束 1）。
    这条口径必须原样写进诊断产物的 schema，否则"support 50/57"那个读数会被
    后人当成"图边权重之和"来复算，算出不同的数。

    `assert_count` 是这些断言里**不同谓词的个数**（关系种类数，历史上那条
    `(-关系种类数, -support, 度)` 排序键的第一位）；`degree` 是候选在本次接地
    邻域里的度（不同对端个数）。
    """
    anchors = set(anc)
    assertions = 0
    predicates: set[str] = set()
    for other, pred in neigh.get(x, ()):
        if other not in anchors:
            continue
        if (x, pred, other) in tri or (other, pred, x) in tri:
            assertions += 1
            predicates.add(pred)
    neighbours = {o for o, _ in neigh.get(x, ())}
    return {"assertions": assertions, "predicates": len(predicates),
            "degree": len(neighbours)}


def rank_candidates(tri: dict, neigh: dict, anc: list[str],
                    cand: set[str]) -> list[dict]:
    """按 `(-support_sum, -assert_count, degree, name)` 排序候选。

    **只有 `top1_margin` 会调用它。** 排序键是计划书 §W2 预登记的那四个，
    名字排在最后且只用于稳定展示——并列敏感性由 `bridge_calib.tie_of_top`
    单独判定，不许靠字典序把并列变成"确定的第一名"。
    """
    rows: list[dict] = []
    for x in cand:
        sup = candidate_support(tri, neigh, anc, x)
        rows.append({
            "x": x,
            "support_sum": int(sup["assertions"]),
            "assert_count": int(sup["predicates"]),
            "degree": int(sup["degree"]),
        })
    return sorted(rows, key=lambda r: (-r["support_sum"], -r["assert_count"],
                                       r["degree"], r["x"]))


def _decide_top1_x(tri: dict, neigh: dict, anc: list[str], cand: set[str],
                   calibration: dict | None) -> str:
    """`top1_margin`：候选不唯一时按校准概率 + 代价比决定选不选 top-1。

    返回被选中的候选名；返回 `""` 表示**拒答**（调用方落回 `ambiguous`）。

    四道闸，缺一不可：

    1. `calibration is None` → 拒答。没校准表就没有概率，没概率就不许猜；
    2. `bridge_calib.decide_top1()` 的 `answer` 非真 → 拒答。这一层已经包含
       并列检测（`tie`）与"桶里没样本（`bucket_missing`）"两种回退；
    3. 被选中的必须是 `ranked[0]`，身份再核一次——防止将来有人改排序键却忘了
       同步"选中的是谁"；
    4. **不重试次优候选**。若 top-1 凑不出 X—A / X—B，结果是 `missing_edge`
       （与 `unique_only` 下"唯一候选凑不出句"同一个状态），不是"换成第二名
       再试一次"——那会把一次拒答变成一条未登记的重试策略。

    最后一条很重要：本函数**不读任何一个 support 阈值**。所有判定都在
    `bridge_calib` 里，阈值只有概率阈值 `2/3`。
    """
    if calibration is None:
        return ""
    ranked = rank_candidates(tri, neigh, anc, cand)
    verdict = bridge_calib.decide_top1(calibration, ranked)
    if not verdict.get("answer"):
        return ""
    x = str(verdict.get("x") or "")
    if not x or x != str(ranked[0]["x"]):
        return ""
    return x


def find_bridge(question: str, units: list[dict], *, chunk_assertions=None,
                display=None, policy: str = DEFAULT_POLICY,
                calibration: dict | None = None) -> dict:
    """在上下文断言里求唯一桥接实体。返回 `{status, x, sentences, citations}`。

    `chunk_assertions` 是可选回调 `(chunk_id) -> [断言记录]`，`display` 是
    `(entity_id) -> 名字`。传了它们就按"上下文块的**全量**断言"求候选，而不是只用
    `unit["assertions"]`——后者被 `build_context` 按焦点实体谓词过滤过，会把
    `X—A`、`X—B` 这类"另一端不是焦点实体"的边丢掉，实测因此少捞 6 题（dev
    multihop 20 vs 26）。两种来源都只覆盖**本次上下文里的块**，接地口径不变。

    `status` 取值：
    * `not_applicable` —— 不在触发面上；
    * `no_common_neighbor` —— 上下文里没有同时连两锚点的实体；
    * `ambiguous` —— 有多个候选（**不猜**，落回原链路）；
    * `missing_edge` —— 候选唯一但凑不出 X—A 或 X—B 的三元组句；
    * `ok` —— 可作答。

    ## 两个新形参（默认值 = 改造前行为）

    * `policy`：`"unique_only"`（默认）/ `"top1_margin"`。默认值下**连一次排序都不
      做**，候选多于一个直接返回 `ambiguous`，与改造前逐位相同。
    * `calibration`：`bridge_calib` 的映射表（可为 `None`）。**只有
      `policy="top1_margin"` 会读它**；`unique_only` 下传什么都无所谓，因为那条
      分支根本走不到概率判定。
    """
    q = str(question or "")
    # 策略名先校验，**在触发面判断之前**：拼错的策略名在任何题上都必须炸。
    # 放在触发面之后会让"非触发题 + 拼错策略"安静地返回 `not_applicable`，
    # 于是一个打错的开关只在 multihop 那 96 道题上才显形——本项目把这种形态
    # 记作"开关没接线"。
    if policy not in POLICIES:
        raise ValueError(f"未登记的 bridge policy {policy!r}；只允许 {POLICIES}")
    if not is_bridge_question(q):
        return {"status": "not_applicable", "x": "", "sentences": [], "citations": []}
    anc = anchors_of(q)

    # 候选构造器只有一份（`_grounded_triples`），W2.1 的诊断也调它——计划书
    # 明令"禁止诊断和生产各写一套"。这里不再内联建表。
    tri = _grounded_triples(units, chunk_assertions, display)
    if not tri:
        return {"status": "no_common_neighbor", "x": "", "sentences": [], "citations": []}

    neigh = _neighbourhood(tri)
    cand = _common_neighbours(neigh, anc)
    if not cand:
        return {"status": "no_common_neighbor", "x": "", "sentences": [], "citations": []}
    if len(cand) > 1:
        # 历史行为（`unique_only`）：候选多于一个就**直接拒答**，连排序都不做。
        # 这条分支在任何新策略下都必须原样保留——它就是"回滚"本身。
        if policy == UNIQUE_ONLY:
            return {"status": "ambiguous", "x": "", "sentences": [], "citations": []}
        x = _decide_top1_x(tri, neigh, anc, cand, calibration)
        if not x:
            # 概率不够 / 不可校准 / 并列 —— 一律落回原链路，**不猜**。
            return {"status": "ambiguous", "x": "", "sentences": [], "citations": []}
    else:
        x = next(iter(cand))

    sentences: list[str] = []
    citations: list[dict] = []
    for anchor in anc:
        found = None
        for other, pred in neigh.get(x, ()):
            if other != anchor:
                continue
            key = (x, pred, anchor) if (x, pred, anchor) in tri else (anchor, pred, x)
            if key not in tri:
                continue
            cid, ev, start, end = tri[key]
            if not cid or not ev or start < 0:
                continue
            found = (f"{x}{pred}{anchor}", cid, ev, start, end)
            break
        if found is None:
            return {"status": "missing_edge", "x": x, "sentences": [], "citations": []}
        sentences.append(found[0])
        citations.append({
            "claim": found[0],
            "chunk_id": found[1],
            "evidence": found[2],
            "start": found[3],
            "end": found[4],
        })

    return {"status": "ok", "x": x, "sentences": sentences, "citations": citations}


def merge(answer: dict, bridge: dict, *, max_sentences: int = 8) -> dict:
    """把桥接句前置进答案，句数封顶（形态红线：句数 ≤8、单句 ≤60 字）。

    `max_sentences` 默认 8 与 `answer.MAX_CLAIMS` 对齐：本通路不改 claim 配额，
    只是把已有的两句放到最前面——如果原答案已满 8 句，尾部会被截掉同样的条数，
    总句数仍不超过原配额。
    """
    if not isinstance(bridge, dict) or bridge.get("status") != "ok":
        return answer
    out = dict(answer)
    segs = [s for s in str(out.get("answer") or "").split("。") if s.strip()]
    merged = [str(bridge["x"])] + list(bridge.get("sentences") or ()) + segs
    out["answer"] = "。".join(merged[:max_sentences]) + "。"
    out["citations"] = list(out.get("citations") or ()) + list(bridge.get("citations") or ())
    return out


__all__ = ["BRIDGE_TRIGGER", "DEFAULT_POLICY", "POLICIES", "TOP1_MARGIN",
           "UNIQUE_ONLY", "anchors_of", "candidate_support", "find_bridge",
           "is_bridge_question", "merge", "rank_candidates"]
