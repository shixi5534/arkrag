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
"""
from __future__ import annotations

import re

from . import answer as answer_mod

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


def find_bridge(question: str, units: list[dict], *, chunk_assertions=None,
                display=None) -> dict:
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
    """
    q = str(question or "")
    if not is_bridge_question(q):
        return {"status": "not_applicable", "x": "", "sentences": [], "citations": []}
    anc = anchors_of(q)

    def name_of(eid: str) -> str:
        raw = str(eid or "")
        return str(display(raw)) if display is not None else raw

    # (subject, predicate, object) -> (chunk_id, evidence, start, end)：只收上下文里的断言。
    # 偏移在这里就按**宿主块自己的正文**回填，因为 `verify_citations` 要拿
    # `text[start:end] == evidence` 逐字比对；下游没有别的机会补这个偏移。
    tri: dict[tuple[str, str, str], tuple[str, str, int, int]] = {}
    for unit in units or ():
        cid = str(unit.get("chunk_id") or "")
        body = str(unit.get("text") or "")
        if not cid or not body:
            continue
        if chunk_assertions is not None:
            records = [
                {"subject": name_of(r.get("subject_id")),
                 "predicate": str(r.get("predicate") or ""),
                 "object": name_of(r.get("object_id")),
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
    if not tri:
        return {"status": "no_common_neighbor", "x": "", "sentences": [], "citations": []}

    neigh: dict[str, list[tuple[str, str]]] = {}
    for s, p, o in tri:
        neigh.setdefault(s, []).append((o, p))
        neigh.setdefault(o, []).append((s, p))

    a_n = {o for o, _ in neigh.get(anc[0], ())}
    b_n = {o for o, _ in neigh.get(anc[1], ())}
    cand = (a_n & b_n) - set(anc)
    if not cand:
        return {"status": "no_common_neighbor", "x": "", "sentences": [], "citations": []}
    if len(cand) > 1:
        return {"status": "ambiguous", "x": "", "sentences": [], "citations": []}

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


__all__ = ["BRIDGE_TRIGGER", "anchors_of", "find_bridge", "is_bridge_question", "merge"]
