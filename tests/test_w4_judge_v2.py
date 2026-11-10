"""判据 v2（三轴）的门：J6 四类支撑测试 + 同源证明 + 翻转分类 + loss 向量。

## 这个文件为什么这么写

计划书 W4 的验收 J6 只有一句：「构造"合法引用但不支持答案值""支持值但引用错位"
"额外候选无支撑""锚点被误计为候选"四类测试；结果必须符合预登记规则」。

四类里最容易写错的是最后一类：`multihop` 的答案里**必然**出现问句锚点
（`阿米娅交谈云兽` 里的 `云兽`），把它算成"第二个候选值"，就会把 v1 判对的答案
在 v2 里误拒。所以这里的第一条用例不是断言"通过"，而是先断言
`parse_values()` 只吐出 1 个值——**把解析器本身钉死**，再断言判定。

另外三条用例（同源证明、翻转分类、loss 向量）对应计划书 J1 / 判据表：
v2 的 `value_ok` 必须与 v1 的 `pass` 同源（唯一例外预登记在
`judge_v2.VALUE_OK_V1_EXCEPTIONS`），四类翻转必须计数正确且 `FP` 不许悄悄出现。

## 反向实测（写完真跑的，`tmp_w4a_mutate.py`，9/9 命中）

改动 `judge_v2.py` 的对应位置 → 指定用例红 → 还原后绿。逐笔清单见
`TestSuiteHygieneOfThisFile` 的 docstring（含两笔第一版"没变红"的坑）。
"""

from __future__ import annotations

import pytest

from arkrag import judge as v1_mod
from arkrag import judge_v2 as v2_mod


# ---------------------------------------------------------------- 夹具


def _item(answer, *, question="", acceptable=None):
    return {
        "question": question,
        "answer": answer,
        "acceptable_answers": list(acceptable or []),
    }


def _units(*units):
    """`{"units": [...]}` 形态的上下文（= `pipeline.ask_once` 的 `res["context"]`）。"""
    return {"units": list(units)}


def _unit(chunk_id, *, chapter="", title="", assertions=()):
    return {
        "chunk_id": chunk_id,
        "chapter": chapter,
        "title": title,
        "assertions": [dict(a) for a in assertions],
    }


def _assertion(aid, subject, predicate, obj, *, evidence="", chunk_id=""):
    rec = {
        "assertion_id": aid,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "evidence": evidence or f"{subject}{predicate}{obj}",
    }
    if chunk_id:
        rec["chunk_id"] = chunk_id
    return rec


def _cite(chunk_id, *, assertion_id="", start=0, end=1, evidence="x"):
    return {
        "chunk_id": chunk_id,
        "assertion_id": assertion_id,
        "start": start,
        "end": end,
        "evidence": evidence,
    }


def _ans(text, *citations):
    return {"answer": text, "citations": list(citations)}


Q_MH = "从「云兽」出发，经过一个中间实体可以到达「变形者集群」。这个中间实体叫什么？"
Q_CX = "「海洋」这个词在哪些章节的正文里出现过？"

#: `cross_chunk` 的章节正文：`act99` 刻意**不**提「海洋」（与 `tests/test_judge.py` 同一套）。
CHAPTERS = {
    "act17side": "海洋在他们脚下翻涌。",
    "main_ep16": "海洋干涸了。",
    "act99": "这一章讲的是别的东西。",
}


def _mh_context():
    """标准的双跳上下文：`阿米娅` 是唯一同时连两个锚点的实体。"""
    return _units(
        _unit(
            "mh#001",
            chapter="main_ep14",
            title="双跳测试块",
            assertions=[
                _assertion("a-mh-1", "阿米娅", "交谈", "云兽",
                           evidence="阿米娅与云兽交谈过"),
                _assertion("a-mh-2", "阿米娅", "交谈", "变形者集群",
                           evidence="阿米娅交谈变形者集群"),
            ],
        )
    )


def _mh_citations():
    return [
        _cite("mh#001", assertion_id="a-mh-1", start=0, end=9,
              evidence="阿米娅与云兽交谈过"),
        _cite("mh#001", assertion_id="a-mh-2", start=10, end=20,
              evidence="阿米娅交谈变形者集群"),
    ]


def _judge(category, answer, item, *, context=None, question=None,
           chapter_texts=None, chapter_cap=None):
    return v2_mod.judge_v2(
        category, answer, item, context=context, question=question,
        chapter_texts=chapter_texts, chapter_cap=chapter_cap)


# ---------------------------------------------------------------- J6-①


class TestJ6ValidCitationThatDoesNotSupportTheValue:
    """① 合法引用但不支持答案值 ⇒ `unsupported`，`support_ok=False`。

    这是 v1 的**唯一**缺陷形态：`judge_answer` 根本不看 citations，所以"答案里有
    一条格式合法的引用"这件事在 v1 里连一个比特都影响不了。v2 必须把它变成
    硬失败，而不是"有引用就算过"。
    """

    CHUNKS = {"mh#001": {"text": "凯尔希隶属罗德岛。", "chapter": "main_ep08"}}

    def _context(self):
        return _units(
            _unit("mh#001", chapter="main_ep08",
                  assertions=[_assertion("a-1", "凯尔希", "隶属", "罗德岛",
                                         evidence="凯尔希隶属罗德岛")]),
        )

    def test_citation_is_legal_by_v1_but_still_unsupported(self) -> None:
        cite = _cite("mh#001", assertion_id="a-1", start=0, end=8,
                     evidence="凯尔希隶属罗德岛")
        txt = "阿米娅交谈云兽。阿米娅交谈变形者集群。"
        # 先证明这条引用**格式上完全合法**（偏移逐字对上、章节可定位）：
        legal = v1_mod.citations_support({"citations": [cite]}, self.CHUNKS)
        assert legal["ok"] is True, "夹具坏了：这条引用在 v1 眼里就必须是合法的"
        r = _judge("multihop", _ans(txt, cite), _item("阿米娅", question=Q_MH),
                   context=self._context(), question=Q_MH)
        assert r["axes"]["support_ok"] is False, (
            "引用合法 ≠ 引用支持答案值：阿米娅 不在这条断言的任何一端")
        assert r["loss"]["unsupported_values"] == 1
        assert r["pass"] is False
        assert r["reason"].startswith("support:")
        # v1 对同一个答案判**通过**，正是要治的那处缺陷。
        assert v1_mod.judge_answer(
            "multihop", txt, _item("阿米娅", question=Q_MH), question=Q_MH)["pass"] is True

    def test_support_is_not_derived_from_gold(self) -> None:
        """答案值与 gold 一字不差、但上下文里没有支撑 ⇒ 仍判无支撑。

        这一条钉住"绝不从 gold 反推支撑"：如果实现里出现任何
        `if gold in ...` 之类的兜底，本用例会立刻变红。
        """
        r = _judge("multihop", _ans("阿米娅交谈云兽。阿米娅交谈变形者集群。"),
                   _item("阿米娅", question=Q_MH),
                   context=self._context(), question=Q_MH)
        assert r["axes"]["value_ok"] is True, "值是对的（v1 也会判对）"
        assert r["axes"]["support_ok"] is False, "没有任何引用 ⇒ 不许拿 gold 顶上"
        assert r["loss"]["unsupported_values"] == 1
        assert r["pass"] is False

    def test_context_missing_fails_closed_with_its_own_reason(self) -> None:
        """没传上下文 ⇒ 拒，但 reason 要与"值无支撑"区分开。"""
        r = _judge("multihop", _ans("阿米娅交谈云兽。阿米娅交谈变形者集群。"),
                   _item("阿米娅", question=Q_MH), question=Q_MH)
        assert r["pass"] is False, "缺输入不许被静默当成通过（计划书反证条件 2）"
        assert r["axes"]["support_ok"] is False
        assert "context_missing" in r["reason"]
        assert r["loss"]["unsupported_values"] == 1

    def test_plain_text_answer_without_citations_is_no_citations(self) -> None:
        """只给正文串、不给引用 ⇒ `support:no_citations`，不和"值无支撑"混为一谈。

        这条同时钉住一个**接线风险**：`eval_answer_quality.py` 是按位置把
        `text`（str）传进来的。若那条调用链不额外把 `ans["citations"]` 传进来，
        v2 的 support 轴会对**所有**题判 False。判据这边不许为了"让报告好看"
        而把无引用当有支撑——所以这里只做两件事：显式 fail-closed，并给一个
        能被单独计数的 reason。
        """
        r = _judge("multihop", "阿米娅交谈云兽。阿米娅交谈变形者集群。",
                   _item("阿米娅", question=Q_MH),
                   context=_mh_context(), question=Q_MH)
        assert r["pass"] is False
        assert r["reason"] == "support:no_citations"

    def test_explicit_citations_keyword_is_accepted_for_plain_text(self) -> None:
        """反向控制：正文串 + 显式 `citations=` ⇒ 同一答案立刻通过。"""
        r = v2_mod.judge_v2(
            "multihop", "阿米娅交谈云兽。阿米娅交谈变形者集群。",
            _item("阿米娅", question=Q_MH),
            context=_mh_context(), question=Q_MH, citations=_mh_citations())
        assert r["axes"]["support_ok"] is True
        assert r["pass"] is True


# ---------------------------------------------------------------- J6-②


class TestJ6SupportedValueButCitationPointsElsewhere:
    """② 支持值但引用错位（引用指向别的块）⇒ 同样要被抓到。

    这一条比 ① 严格：值确实在上下文里某条断言上，只是那条引用指的块不对。
    若实现只按 `assertion_id` 或只按 `chunk_id` 单边匹配，就会漏掉。
    """

    def _context(self):
        return _units(
            _unit("mh#001", chapter="main_ep14",
                  assertions=[_assertion("a-other", "阿米娅", "隶属", "罗德岛",
                                         evidence="阿米娅隶属罗德岛"),
                              _assertion("a-other-2", "凯尔希", "隶属", "罗德岛")]),
            _unit("mh#002", chapter="main_ep15",
                  assertions=[_assertion("a-mh-1", "阿米娅", "交谈", "云兽",
                                         evidence="阿米娅与云兽交谈过")]),
        )

    def test_assertion_id_on_the_wrong_chunk_is_misplaced(self) -> None:
        # 断言 id 是真的（`a-mh-1`），但它住在 `mh#002`；引用却声称在 `mh#001`。
        #
        # 夹具刻意让 `mh#001` 里**也有**一条以「阿米娅」为端点的断言（`a-other`）：
        # 这样"错位的那条引用到底算不算数"才可判别——若实现只把该断言排除、
        # 却仍然承认"`mh#001` 被引用过"，`a-other` 就会顶上把这条答案洗成有支撑。
        cite = _cite("mh#001", assertion_id="a-mh-1", start=0, end=9,
                     evidence="阿米娅与云兽交谈过")
        r = _judge("multihop", _ans("阿米娅交谈云兽。", cite),
                   _item("阿米娅", question=Q_MH),
                   context=self._context(), question=Q_MH)
        assert r["axes"]["support_ok"] is False, "引用指向别的块 ⇒ 整条引用都不算支撑"
        assert r["loss"]["unsupported_values"] == 1

    def test_explain_reports_the_misplaced_assertion(self) -> None:
        """错位必须能被审计看到，否则"为什么拒"只能靠猜。"""
        cite = _cite("mh#001", assertion_id="a-mh-1")
        detail = v2_mod.explain_v2(
            "multihop", _ans("阿米娅交谈云兽。", cite), _item("阿米娅", question=Q_MH),
            context=self._context(), question=Q_MH)
        support = detail["axes_detail"]["support"]
        assert support["misplaced_assertions"] == ["a-mh-1"]
        assert support["per_value"][0]["witness_level"] == ""

    def test_misplaced_assertion_cannot_be_revived_by_another_citation(self) -> None:
        """错位的那条断言不许被"另有一条合格引用"顺带救活。

        这是错位检查的**第二层**（`_CitedIndex.allows`）：第一层只看块，
        第二层还要把"被引错了位置的那条断言"本身排除掉。

        形状：唯一能以「阿米娅」为端点的断言 `a-mh-1` 住在 `mh#002`，而有一条引用
        声称它来自 `mh#001`；同时还有一条**合格**引用指向 `mh#002`。此时
        `mh#002` 确实"被引用过"，只做块级检查的实现就会让那条错位断言顶上成为
        witness，把这条答案洗成有支撑。
        """
        context = _units(
            _unit("mh#001", chapter="main_ep14",
                  assertions=[_assertion("a-x", "凯尔希", "隶属", "罗德岛")]),
            _unit("mh#002", chapter="main_ep15",
                  assertions=[_assertion("a-mh-1", "阿米娅", "交谈", "云兽",
                                         evidence="阿米娅与云兽交谈过")]),
        )
        citations = [
            _cite("mh#001", assertion_id="a-mh-1"),   # 错位：a-mh-1 住在 mh#002
            _cite("mh#002", assertion_id="a-mh-1"),   # 合格：同一条断言、正确的块
        ]
        detail = v2_mod.explain_v2(
            "multihop", _ans("阿米娅交谈云兽。", *citations),
            _item("阿米娅", question=Q_MH), context=context, question=Q_MH)
        support = detail["axes_detail"]["support"]
        assert support["misplaced_assertions"] == ["a-mh-1"]
        assert support["per_value"][0]["witness_level"] == "", (
            "错位断言被当成了 witness：它有一条引用指向 mh#001，不该作数")

    def test_citing_only_the_right_chunk_makes_it_supported(self) -> None:
        """反向控制：把引用改到正确的块 ⇒ 同一答案立刻有支撑。

        没有这一条，上面两条可能只是在断言"永远找不到 witness"。
        """
        cite = _cite("mh#002", assertion_id="a-mh-1", start=0, end=9,
                     evidence="阿米娅与云兽交谈过")
        r = _judge("multihop", _ans("阿米娅交谈云兽。", cite),
                   _item("阿米娅", question=Q_MH),
                   context=self._context(), question=Q_MH)
        assert r["axes"]["support_ok"] is True
        assert r["loss"]["unsupported_values"] == 0


# ---------------------------------------------------------------- J6-③


class TestJ6ExtraCandidateWithoutSupport:
    """③ 额外候选无支撑 ⇒ `extra_values` / `unsupported_values` 两个计数各自正确。

    `阿米娅` 是预登记答案，`凯尔希` 是"顺手多列的一个"。v1 对同一答案判**通过**
    （两个锚点都被 gold 连上了），v2 必须拒，且拒的理由要能分别数出来。
    """

    TXT = "阿米娅交谈云兽。阿米娅交谈变形者集群。凯尔希交谈变形者集群。"

    def test_extra_candidate_is_counted_and_unsupported(self) -> None:
        r = _judge("multihop", _ans(self.TXT, *_mh_citations()),
                   _item("阿米娅", question=Q_MH),
                   context=_mh_context(), question=Q_MH)
        assert r["axes"]["value_ok"] is True, "值轴本身是过的（gold 连上了两个锚点）"
        assert r["axes"]["cardinality_ok"] is False
        assert r["axes"]["support_ok"] is False, "凯尔希 在本次上下文里没有支撑"
        assert r["loss"]["extra_values"] == 1
        assert r["loss"]["unsupported_values"] == 1
        assert r["loss"]["cardinality_over"] == 1
        assert r["pass"] is False
        assert v1_mod.judge_answer(
            "multihop", self.TXT, _item("阿米娅", question=Q_MH),
            question=Q_MH)["pass"] is True, "v1 之所以判过，正是因为它没有精度项"

    def test_supported_extra_candidate_fails_only_on_cardinality(self) -> None:
        """额外候选**有支撑**时，仍要因基数被拒——两个失效面必须彼此独立。

        语境里补上 `凯尔希 交谈 变形者集群` 并给出它的引用 ⇒ support 过、
        cardinality 不过。这条钉住"基数不是靠无支撑顺带实现的"。
        """
        context = _units(
            _unit("mh#001", chapter="main_ep14", assertions=[
                _assertion("a-mh-1", "阿米娅", "交谈", "云兽",
                           evidence="阿米娅与云兽交谈过"),
                _assertion("a-mh-2", "阿米娅", "交谈", "变形者集群",
                           evidence="阿米娅交谈变形者集群"),
                _assertion("a-mh-3", "凯尔希", "交谈", "变形者集群",
                           evidence="凯尔希交谈变形者集群"),
            ]))
        citations = list(_mh_citations()) + [
            _cite("mh#001", assertion_id="a-mh-3", start=21, end=31,
                  evidence="凯尔希交谈变形者集群")]
        r = _judge("multihop", _ans(self.TXT, *citations),
                   _item("阿米娅", question=Q_MH),
                   context=context, question=Q_MH)
        assert r["axes"]["support_ok"] is True, "三个值都有支撑"
        assert r["axes"]["cardinality_ok"] is False
        assert r["loss"]["extra_values"] == 1
        assert r["loss"]["cardinality_over"] == 1
        assert r["loss"]["unsupported_values"] == 0


# ---------------------------------------------------------------- J6-④


class TestJ6AnchorsAreNeverCandidates:
    """④ 锚点被误计为候选 ⇒ **不**算成额外候选。本文件最容易写错的一条。

    `multihop` 的答案里必然出现锚点（`云兽`、`变形者集群`），它们来自**题干**，
    不是"这次提出的候选值"。把任一个算成候选，预登记答案就会被 v2 误拒。
    """

    def test_anchor_only_sentences_yield_exactly_one_value(self) -> None:
        txt = "云兽交谈阿米娅。变形者集群交谈阿米娅。"
        values = v2_mod.parse_values("multihop", txt, question=Q_MH)
        assert values == ["阿米娅"], (
            f"锚点不是候选值；实测取到 {values}（多出来的那一项就是把锚点当候选）")

    def test_anchor_led_answer_passes_every_axis(self) -> None:
        txt = "云兽交谈阿米娅。变形者集群交谈阿米娅。"
        r = _judge("multihop", _ans(txt, *_mh_citations()),
                   _item("阿米娅", question=Q_MH),
                   context=_mh_context(), question=Q_MH)
        assert r["axes"]["value_ok"] is True
        assert r["axes"]["cardinality_ok"] is True, "只有 1 个候选值，基数项必须过"
        assert r["loss"]["extra_values"] == 0
        assert r["loss"]["cardinality_over"] == 0
        assert r["pass"] is True

    def test_first_segment_that_is_an_anchor_is_not_a_bare_value(self) -> None:
        """首段就是锚点时不产生裸值候选（生产档首段是 `x`，不是锚点）。"""
        txt = "云兽。云兽交谈阿米娅。"
        values = v2_mod.parse_values("multihop", txt, question=Q_MH)
        assert values == ["阿米娅"], f"首段锚点被当成候选了：{values}"

    def test_background_claim_between_two_non_anchors_adds_nothing(self) -> None:
        """两端都不是锚点的句子是背景陈述，不是"提出了一个候选"。"""
        txt = "阿米娅交谈云兽。凯尔希隶属罗德岛。阿米娅交谈变形者集群。"
        values = v2_mod.parse_values("multihop", txt, question=Q_MH)
        assert values == ["阿米娅"], f"背景 claim 被算成候选了：{values}"

    def test_bare_first_segment_is_a_candidate_when_it_labels_the_answer(self) -> None:
        """桥接通路发的是 `x。x交谈A。x交谈B。`——首段裸值必须被认成候选。"""
        txt = "阿米娅。阿米娅交谈云兽。阿米娅交谈变形者集群。"
        values = v2_mod.parse_values("multihop", txt, question=Q_MH)
        assert values == ["阿米娅"]
        r = _judge("multihop", _ans(txt, *_mh_citations()),
                   _item("阿米娅", question=Q_MH),
                   context=_mh_context(), question=Q_MH)
        assert r["pass"] is True
        assert r["loss"]["extra_values"] == 0


# ---------------------------------------------------------------- 同源证明


#: `value_ok` 与 v1 `pass` 的逐类对照表。`expect_value_ok` 是 v2 的值轴期望值，
#: `registered_exception=True` 表示这一格是 `VALUE_OK_V1_EXCEPTIONS` 登记过的例外。
VALUE_PARITY_CASES = [
    # (说明, category, answer_text, item, chapter_texts, expect_v1, expect_value_ok, exception)
    ("single_chunk 命中", "single_chunk", "答案是凯尔希。", _item("凯尔希"),
     None, True, True, False),
    ("single_chunk 未命中", "single_chunk", "答案是阿米娅。", _item("凯尔希"),
     None, False, False, False),
    ("single_chunk 无 gold", "single_chunk", "随便什么", _item(""),
     None, False, False, False),
    ("alias 命中 acceptable", "alias", "时序演进", _item("不是矛盾", acceptable=["时序演进"]),
     None, True, True, False),
    ("direct_relation 单条断言含全部锚点", "direct_relation", "菈塔托丝隶属布朗陶家。",
     _item("隶属", question="「菈塔托丝」与「布朗陶」之间的关系是什么？"),
     None, True, True, False),
    ("direct_relation 主语不对", "direct_relation", "温蒂隶属罗德岛。",
     _item("隶属", question="「红」与「罗德岛」之间的关系是什么？"),
     None, False, False, False),
    ("multihop 连到每个锚点", "multihop", "阿米娅交谈云兽。阿米娅交谈变形者集群。",
     _item("阿米娅", question=Q_MH), None, True, True, False),
    ("multihop 只连一个锚点", "multihop", "阿米娅交谈变形者集群。变形者集群交谈凯尔希。",
     _item("阿米娅", question=Q_MH), None, False, False, False),
    ("multihop gold 就是锚点", "multihop", "凯尔希隶属罗德岛。",
     _item("凯尔希", question="「凯尔希」与「罗德岛」之间的中间实体是谁？"),
     None, False, False, False),
    ("cross_chunk 集合相等", "cross_chunk", "act17side、main_ep16",
     _item("act17side、main_ep16", question=Q_CX), CHAPTERS, True, True, False),
    ("cross_chunk 漏章节", "cross_chunk", "act17side",
     _item("act17side、main_ep16", question=Q_CX), CHAPTERS, False, False, False),
    ("cross_chunk 额外章节真实且有界", "cross_chunk", "act17side、main_ep16、act99",
     _item("act17side", question=Q_CX), CHAPTERS, False, False, False),
    ("cross_chunk 额外章节没提到实体", "cross_chunk", "act17side、main_ep16、act99",
     _item("act17side、main_ep16", question=Q_CX), CHAPTERS, False, False, False),
    # —— 唯一登记的例外：`|额外| <= |W|` 是基数规则，已搬到 cardinality 轴。
    ("cross_chunk 额外全真但超出 gold 数", "cross_chunk", "act17side、main_ep16、act99",
     _item("act17side", question=Q_CX), {"act17side": "海洋。", "main_ep16": "海洋。",
                                         "act99": "海洋。"},
     False, True, True),
    ("cross_chunk 没传正文（显式降级）", "cross_chunk", "act17side、main_ep16、act99",
     _item("act17side", question=Q_CX), None, False, False, False),
    ("未知类别不崩", "no_such_category", "x", _item("y"), None, False, False, False),
]


class TestValueOkIsSameOriginAsV1:
    """`value_ok` 必须与 v1 `pass` 同源（结构类之外）。

    "同源"在这里的机检形态是：对同一输入，`value_ok == v1.pass`，**除了**
    `judge_v2.VALUE_OK_V1_EXCEPTIONS` 逐条登记的那些格子。登记表本身也要被
    断言覆盖：例外表里列了 `cross_chunk`，那么至少得有一条用例落在那个格子上，
    否则登记表就是一条没人验证的声明（本项目的"门在测磁盘状态"同型失效）。
    """

    @pytest.mark.parametrize(
        "name,category,text,item,chapters,expect_v1,expect_value_ok,exception",
        VALUE_PARITY_CASES,
        ids=[c[0] for c in VALUE_PARITY_CASES])
    def test_value_axis_matches_v1(self, name, category, text, item, chapters,
                                   expect_v1, expect_value_ok, exception) -> None:
        q = item.get("question")
        v1 = v1_mod.judge_answer(category, text, item, question=q, chapter_texts=chapters)
        detail = v2_mod.explain_v2(category, text, item, question=q, chapter_texts=chapters)
        value_ok = detail["axes_detail"]["value"]["ok"]
        assert v1["pass"] is expect_v1, f"{name}: v1 判定与预期不符（{v1['reason']}）"
        assert value_ok is expect_value_ok, (
            f"{name}: v2 的 value_ok={value_ok} 与登记值 {expect_value_ok} 不符"
            f"（v1 reason={v1['reason']}）")
        if not exception:
            assert value_ok is v1["pass"], (
                f"{name}: 非登记例外格上 value_ok 必须与 v1 pass 一致")

    def test_registered_exception_is_exercised_by_at_least_one_case(self) -> None:
        """登记表里的每个类别，都必须被上表真打中一次（不许是死条目）。"""
        hit = {c[1] for c in VALUE_PARITY_CASES if c[7]}
        assert set(v2_mod.VALUE_OK_V1_EXCEPTIONS) <= hit, (
            f"登记表 {sorted(v2_mod.VALUE_OK_V1_EXCEPTIONS)} 里的类别没有对应用例："
            f"实测只覆盖 {sorted(hit)}")

    def test_the_registered_exception_is_still_rejected_by_v2(self) -> None:
        """搬走的那条规则**没有丢**：v2 整体仍必须拒（基数轴接住了它）。"""
        item = _item("act17side", question=Q_CX)
        chapters = {"act17side": "海洋。", "main_ep16": "海洋。", "act99": "海洋。"}
        r = _judge("cross_chunk", "act17side、main_ep16、act99", item,
                   question=Q_CX, chapter_texts=chapters)
        assert r["axes"]["value_ok"] is True, "值轴不再管基数"
        assert r["axes"]["cardinality_ok"] is False, "基数轴必须接住 |额外| > |W|"
        # over 记的是"超出多少"：|额外|=2、|W|=1 ⇒ 超 1；cap=4 下 |O|=3 没超。
        assert r["loss"]["cardinality_over"] == 1
        assert r["loss"]["extra_values"] == 2
        assert r["pass"] is False

    def test_unknown_category_is_rejected_not_crashing(self) -> None:
        r = _judge("no_such_category", "x", _item("y"))
        assert r["pass"] is False
        assert r["reason"].startswith("value:unknown_category")


# ---------------------------------------------------------------- 结构类


class TestStructuralCategoriesAreNotScored:
    """结构类返回 `pass=None`（剔分母），与 v1 同义。"""

    @pytest.mark.parametrize("cat", list(v2_mod.STRUCTURAL_CATEGORIES))
    def test_structural_returns_none(self, cat: str) -> None:
        r = _judge(cat, "时序演进", _item("时序演进"))
        assert r["pass"] is None, "结构类必须返回 None，不许当 0 也不许当 1"
        assert r["reason"] == v2_mod.REASON_STRUCTURAL
        assert "structural" in r["reason"]

    def test_structural_set_matches_v1(self) -> None:
        assert tuple(v2_mod.STRUCTURAL_CATEGORIES) == tuple(
            v1_mod.STRUCTURAL_CATEGORIES)
        assert set(v2_mod.SCORABLE_CATEGORIES).isdisjoint(
            set(v2_mod.STRUCTURAL_CATEGORIES))

    def test_empty_answer_is_false_not_none(self) -> None:
        r = _judge("single_chunk", "   ", _item("凯尔希"))
        assert r["pass"] is False
        assert r["reason"] == v2_mod.REASON_EMPTY
        assert r["loss"] == {"missing_or_wrong_values": 0, "extra_values": 0,
                             "unsupported_values": 0, "cardinality_over": 0}

    def test_structural_category_counts_as_na_in_flip_table(self) -> None:
        """结构类不进 PP/PF/FP/FF——否则四格之和与可判分母对不上。"""
        flip = v2_mod.classify_flip({"pass": None, "reason": "structural"},
                                    {"pass": None, "reason": "structural"},
                                    question_id="q-tc-000001", category="time_conflict")
        assert flip["cell"] == "NA"
        assert flip["cell"] not in v2_mod.CELLS


# ---------------------------------------------------------------- 翻转分类


def _verdict(passed, *, value_ok=True, support_ok=True, cardinality_ok=True):
    return {
        "pass": passed,
        "reason": "",
        "axes": {"value_ok": value_ok, "support_ok": support_ok,
                 "cardinality_ok": cardinality_ok},
        "loss": {"missing_or_wrong_values": 0, "extra_values": 0,
                 "unsupported_values": 0, "cardinality_over": 0},
    }


class TestFlipClassification:
    """`PP/PF/FP/FF` 计数正确，且 `PF` 的归因互斥。"""

    def test_four_cells(self) -> None:
        assert v2_mod.classify_flip(_verdict(True), _verdict(True))["cell"] == "PP"
        assert v2_mod.classify_flip(_verdict(True), _verdict(False))["cell"] == "PF"
        assert v2_mod.classify_flip(_verdict(False), _verdict(True))["cell"] == "FP"
        assert v2_mod.classify_flip(_verdict(False), _verdict(False))["cell"] == "FF"

    def test_pf_attributions_are_mutually_exclusive(self) -> None:
        """四个归因各自被一个构造打中，且 `PF` 总数等于归因之和。"""
        rows = [
            {"question_id": "q-a", "category": "single_chunk",
             "v1": _verdict(True), "v2": _verdict(False, support_ok=False)},
            {"question_id": "q-b", "category": "multihop",
             "v1": _verdict(True), "v2": _verdict(False, cardinality_ok=False)},
            {"question_id": "q-c", "category": "cross_chunk",
             "v1": _verdict(True), "v2": _verdict(False, cardinality_ok=False)},
            {"question_id": "q-d", "category": "alias",
             "v1": _verdict(True), "v2": _verdict(False, value_ok=False)},
            {"question_id": "q-e", "category": "single_chunk",
             "v1": _verdict(True), "v2": _verdict(True)},
            {"question_id": "q-f", "category": "direct_relation",
             "v1": _verdict(False), "v2": _verdict(False, support_ok=False)},
        ]
        rep = v2_mod.tally_flips(rows)
        assert rep["counts"] == {"PP": 1, "PF": 4, "FP": 0, "FF": 1}
        assert rep["pf_attribution"] == {
            "unsupported": 1, "entity_precision": 1,
            "chapter_cardinality": 1, "other": 1}
        assert sum(rep["pf_attribution"].values()) == rep["counts"]["PF"]
        assert rep["ok"] is True
        assert rep["problems"] == []

    def test_entity_precision_is_attributed_for_multihop_not_cross_chunk(self) -> None:
        """`multihop` 的基数失败记"实体精度"，`cross_chunk` 的记"章节基数"。

        这条钉住归因用的是**类别**，不是"哪根轴"。若不传类别（`judge_v2` 的
        返回里没有 `category` 字段），两者会全部塌成"章节基数"，`PF` 归因就
        失去意义。
        """
        mh = v2_mod.classify_flip(_verdict(True), _verdict(False, cardinality_ok=False),
                                  question_id="q-mh", category="multihop")
        cx = v2_mod.classify_flip(_verdict(True), _verdict(False, cardinality_ok=False),
                                  question_id="q-cx", category="cross_chunk")
        assert mh["attribution"] == "entity_precision"
        assert cx["attribution"] == "chapter_cardinality"
        assert mh["axes_failed"] == ["cardinality_ok"]

    def test_unregistered_fp_is_flagged(self) -> None:
        rows = [{"question_id": "q-x", "category": "multihop",
                 "v1": _verdict(False), "v2": _verdict(True)}]
        rep = v2_mod.tally_flips(rows)
        assert rep["counts"]["FP"] == 1
        assert rep["fp_unregistered"] == ["q-x"]
        assert rep["ok"] is False
        assert any("FP" in p for p in rep["problems"])

    def test_registered_fp_is_accepted_block(self) -> None:
        rows = [{"question_id": "q-x", "category": "multihop",
                 "v1": _verdict(False), "v2": _verdict(True)}]
        rep = v2_mod.tally_flips(rows, registered_fp={"q-x": "v1 的值判定漏了别名"})
        assert rep["fp_unregistered"] == []
        assert rep["ok"] is True
        assert rep["per_item"][0]["fp_registered"] is True
        assert rep["per_item"][0]["reason"].startswith("registered:")

    def test_registry_is_empty_by_default(self) -> None:
        """默认登记表必须为空：v2 在构造上取不到 FP（`v2 pass ⇒ v1 pass`）。"""
        assert v2_mod.REGISTERED_V1_FALSE_NEGATIVES == {}

    def test_structural_items_do_not_break_the_cell_sum(self) -> None:
        rows = [
            {"question_id": "q-1", "category": "single_chunk",
             "v1": _verdict(True), "v2": _verdict(True)},
            {"question_id": "q-2", "category": "time_conflict",
             "v1": {"pass": None}, "v2": {"pass": None}},
        ]
        rep = v2_mod.tally_flips(rows)
        assert rep["n_structural"] == 1
        assert rep["n_scorable"] == 1
        assert sum(rep["counts"].values()) == rep["n_scorable"]
        assert rep["ok"] is True


# ---------------------------------------------------------------- loss 向量


class TestLossVector:
    """四个计数各自的含义必须能单独验，不许互相遮住。"""

    def test_cross_chunk_missing_counts_chapters(self) -> None:
        r = _judge("cross_chunk", "act17side",
                   _item("act17side、main_ep16", question=Q_CX),
                   question=Q_CX, chapter_texts=CHAPTERS)
        assert r["loss"]["missing_or_wrong_values"] == 1
        assert r["loss"]["extra_values"] == 0
        assert r["pass"] is False

    def test_cross_chunk_extra_counts_added_chapters(self) -> None:
        """`extra_values` 只数"多列的章节"，与 `cardinality_over` 是两个量。

        这一笔刻意让 `|O| = 4 = cap`（cap 项不触发），只靠 v1 那条
        `|额外| <= |W|` 拒：多列了 3 章、gold 只有 1 章 ⇒ `over = 3 - 1 = 2`。
        """
        chapters = {"ch0": "海洋。", "ch1": "海洋。", "ch2": "海洋。", "ch3": "海洋。"}
        r = _judge("cross_chunk", "ch0、ch1、ch2、ch3",
                   _item("ch0", question=Q_CX),
                   question=Q_CX, chapter_texts=chapters)
        assert r["loss"]["missing_or_wrong_values"] == 0
        assert r["loss"]["extra_values"] == 3
        assert r["axes"]["cardinality_ok"] is False
        assert r["loss"]["cardinality_over"] == 2, "cap 项没触发，超出量只来自相对规则"

    def test_cross_chunk_cap_is_a_real_gate(self) -> None:
        """`|O| <= cap` 必须真的能拒——v1 的 `|O| <= 2|W|` 在 `|O| <= 4` 时恒真。"""
        chapters = {"act17side": "海洋。", "main_ep16": "海洋。", "act99": "海洋。"}
        item = _item("act17side、main_ep16", question=Q_CX)
        loose = _judge("cross_chunk", "act17side、main_ep16、act99", item,
                       question=Q_CX, chapter_texts=chapters, chapter_cap=4)
        tight = _judge("cross_chunk", "act17side、main_ep16、act99", item,
                       question=Q_CX, chapter_texts=chapters, chapter_cap=2)
        assert loose["axes"]["cardinality_ok"] is True, "cap=4 时 3 章没超"
        assert tight["axes"]["cardinality_ok"] is False, "cap=2 时 3 章超了"
        assert tight["loss"]["cardinality_over"] == 1

    def test_cap_does_not_replace_the_relative_rule(self) -> None:
        """cap 之下仍保留 v1 的 `|额外| <= |W|`（计划书原文，不能用 cap 替代）。"""
        chapters = {f"ch{i}": "海洋。" for i in range(8)}
        item = _item("ch0", question=Q_CX)   # |W| = 1
        r = _judge("cross_chunk", "ch0、ch1、ch2、ch3", item,
                   question=Q_CX, chapter_texts=chapters, chapter_cap=8)
        assert r["axes"]["cardinality_ok"] is False, (
            "cap=8 放得下 4 章，但 |额外|=3 > |W|=1，相对规则必须仍然拦得住")
        assert r["loss"]["cardinality_over"] == 2

    def test_loss_vector_keys_are_frozen(self) -> None:
        r = _judge("single_chunk", "答案是凯尔希。", _item("凯尔希"))
        assert set(r["loss"]) == {"missing_or_wrong_values", "extra_values",
                                  "unsupported_values", "cardinality_over"}
        assert set(r) == {"pass", "reason", "axes", "loss"}
        assert set(r["axes"]) == {"value_ok", "support_ok", "cardinality_ok"}

    def test_missing_or_wrong_is_one_not_zero_when_value_axis_fails(self) -> None:
        """值的计数必须跟着值轴走，不许因为"gold 串出现了"记 0。

        反例形状：gold（`阿米娅`）确实在答案里，但没连到全部锚点 ⇒ `value_ok`
        为 False。若 `missing_or_wrong_values` 用"gold 在不在文本里"来数，这里会
        记成 0，读报告的人就会看到"值没错、基数也没超，但整体不过"这种自相矛盾的行。
        """
        r = _judge("multihop", "阿米娅交谈变形者集群。",
                   _item("阿米娅", question=Q_MH), question=Q_MH)
        assert r["axes"]["value_ok"] is False
        assert r["loss"]["missing_or_wrong_values"] == 1

    def test_non_registered_categories_mark_cardinality_as_unchecked(self) -> None:
        detail = v2_mod.explain_v2("single_chunk", "答案是凯尔希。", _item("凯尔希"))
        assert detail["axes_detail"]["cardinality"]["ok"] is True
        assert detail["axes_detail"]["cardinality"]["reason"] == "not_registered"


# ---------------------------------------------------------------- 解析规则


class TestParseRulesAreFrozenPureFunctions:
    """解析规则是纯函数：同样的输入两次同结果、不改入参、不吃全局状态。"""

    def test_cross_chunk_uses_the_same_splitter_as_v1(self) -> None:
        """空格分隔的章节代号必须与 `judge.chapter_set` 完全一致（同一格坑）。"""
        text = "act33side main_ep14"
        assert v2_mod.parse_values("cross_chunk", text) == sorted(
            v1_mod.chapter_set(text))
        assert v2_mod.parse_values("cross_chunk", text) == ["act33side", "main_ep14"]

    def test_parse_values_is_deterministic_and_input_preserving(self) -> None:
        item = _item("阿米娅", question=Q_MH)
        snapshot = {k: (list(v) if isinstance(v, list) else v) for k, v in item.items()}
        first = v2_mod.parse_values("multihop", "阿米娅交谈云兽。", question=Q_MH, item=item)
        second = v2_mod.parse_values("multihop", "阿米娅交谈云兽。", question=Q_MH, item=item)
        assert first == second
        assert item == snapshot, "解析不许改入参"

    def test_long_value_absorbs_short_value(self) -> None:
        """`凯尔希` 与 `凯尔希医生` 同时出现时不是两个独立答案值。"""
        values = v2_mod.parse_values(
            "alias", "凯尔希医生", item=_item("凯尔希医生", acceptable=["凯尔希"]))
        assert values == ["凯尔希医生"], f"长值应该吸收短值，实测 {values}"

    def test_frozen_input_has_the_declared_shape(self) -> None:
        bundle = v2_mod.frozen_input(
            "cross_chunk", _ans("act17side、main_ep16", _cite("k#1")),
            _item("act17side、main_ep16", question=Q_CX),
            context=_units(_unit("k#1", chapter="act17side")))
        assert set(bundle) == {"schema", "category", "answer_text", "parsed_values",
                               "citations", "context_assertions", "context_chunks",
                               "gold", "anchors"}
        assert bundle["schema"] == v2_mod.INPUT_SCHEMA
        assert bundle["parsed_values"] == ["act17side", "main_ep16"]
        assert bundle["gold"]["values"] == ["act17side", "main_ep16"]
        assert bundle["anchors"] == ["海洋"]
        assert bundle["citations"][0]["chunk_id"] == "k#1"
        assert bundle["context_chunks"]["k#1"]["chapter"] == "act17side"

    def test_answer_dict_citations_are_normalized(self) -> None:
        cites = v2_mod.normalize_citations(
            [{"chunk_id": " k#1 ", "assertion_id": None, "evidence": " 甲 "}, "垃圾"])
        assert len(cites) == 1, "非 dict 的引用条目必须被丢掉而不是炸"
        assert cites[0]["chunk_id"] == "k#1"
        assert cites[0]["assertion_id"] == ""
        assert cites[0]["evidence"] == "甲"

    def test_evidence_level_is_the_weakest_accepted_layer(self) -> None:
        """`evidence` 兜底层：值出现在被引用断言的原文里也算见证，但层级要记弱。"""
        context = _units(_unit("c#1", assertions=[
            _assertion("a-1", "凯尔希", "隶属", "罗德岛",
                       evidence="凯尔希隶属罗德岛医院")]))
        detail = v2_mod.explain_v2(
            "single_chunk", _ans("医院", _cite("c#1", assertion_id="a-1")),
            _item("医院"), context=context)
        per_value = detail["axes_detail"]["support"]["per_value"]
        assert per_value[0]["witness_level"] in ("evidence", "chunk_field")
        assert per_value[0]["witness_level"] != "endpoint", (
            "「医院」不是这条断言的端点，不许被记成最强层级")


# ---------------------------------------------------------------- 文件自身卫生


class TestSuiteHygieneOfThisFile:
    """本文件自己的门：辅助函数必须会抛、每条用例必须能真红。

    ## 反向实测（`tmp_w4a_mutate.py`，**9 笔全命中**，每笔"改坏 → 指名用例红 → 还原"）

    ```
    基线 63 passed rc=0
    [1] support 恒记 0 个无支撑 → test_citation_is_legal_by_v1_but_still_unsupported 红
    [2] 删掉 _CitedIndex.allows 里的 misplaced 检查
        → test_misplaced_assertion_cannot_be_revived_by_another_citation 红
    [3] 错位的引用照样给它声称的块投票（删掉 continue）
        → test_assertion_id_on_the_wrong_chunk_is_misplaced 红
    [4] 单边锚点时把锚点自己也收进候选（即把锚点当候选）
        → test_anchor_only_sentences_yield_exactly_one_value 红
    [5] MULTIHOP_MAX_CANDIDATES = 2 → test_extra_candidate_is_counted_and_unsupported 红
    [6] cross_chunk 只用 cap、删掉 rel_over → test_cap_does_not_replace_the_relative_rule 红
    [7] PF 归因不分类别 → test_entity_precision_is_attributed_for_multihop_not_cross_chunk 红
    [8] 结构类也进翻转格子 → test_structural_category_counts_as_na_in_flip_table 红
    [9] missing_or_wrong 改回"gold 在不在文本里"
        → test_missing_or_wrong_is_one_not_zero_when_value_axis_fails 红
    合计 9/9 命中；还原后 63 passed rc=0
    ```

    ## 踩到的坑（写在这里，免得下一个人重犯）

    * **第 2 笔第一版没变红**：夹具里 `mh#001` 没有以「阿米娅」为端点的断言，于是
      "排除错位断言"与"不排除"结果一样（两条路径都找不到 witness）。
      补 `test_misplaced_assertion_cannot_be_revived_by_another_citation`
      才让这一层可判别——**用例要围着函数的判别面写，不能只围着"能过/不能过"写**。
    * **突变脚本第一版有两笔假 MISS**：改写后的文件与原文同大小、又落在同一秒内，
      `mtime+size` 两个字段都命中，CPython 直接跑旧 `.pyc`。脚本现在每轮先删
      `__pycache__/judge_v2*.pyc` 再跑。这与 `test_suite_hygiene.py` 记的
      「门在测磁盘状态，不在测实现」是同一种失效。
    """

    def test_helper_assertion_style_is_explicit(self) -> None:
        """本文件的用例如有问题必须 assert 失败，不靠"记录后继续"。"""
        r = _judge("multihop", "阿米娅交谈云兽。阿米娅交谈变形者集群。",
                   _item("阿米娅", question=Q_MH), question=Q_MH)
        assert r["pass"] is False
        assert isinstance(r["loss"]["unsupported_values"], int)

    def test_schema_version_is_pinned(self) -> None:
        """版本串写进报告，改它等于换口径——必须有人签字。"""
        assert v2_mod.SCHEMA == "arkrag.judge_v2/1"
        assert v2_mod.INPUT_SCHEMA == "arkrag.judge_v2.input/1"
        assert v2_mod.CROSS_CHUNK_CAP_DEFAULT == 4
        assert v2_mod.MULTIHOP_MAX_CANDIDATES == 1

    def test_support_policy_covers_every_scorable_category(self) -> None:
        missing = [c for c in v2_mod.SCORABLE_CATEGORIES
                   if c not in v2_mod.SUPPORT_POLICY]
        assert not missing, f"这些类没有登记 support 层级：{missing}"

    def test_judge_does_not_mutate_its_inputs(self) -> None:
        item = _item("阿米娅", question=Q_MH)
        context = _mh_context()
        before_item = {k: (list(v) if isinstance(v, list) else v) for k, v in item.items()}
        before_ctx = len(context["units"])
        _judge("multihop", _ans("阿米娅交谈云兽。", *_mh_citations()), item,
               context=context, question=Q_MH)
        assert item == before_item
        assert len(context["units"]) == before_ctx
