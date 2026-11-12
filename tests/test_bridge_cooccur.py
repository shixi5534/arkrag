# -*- coding: utf-8 -*-
"""`bridge_cooccur` 的门：三条硬约束与方向正确性。

这些门守的是「不把抽取噪声当结论输出」——本项目已经有过一次
`MULTIHOP_CONJUNCTION` 用模板词「相关联」凑判据的事（实测 +0 题），
本模块不许重犯。
"""
from __future__ import annotations

import pytest

from arkrag import bridge_cooccur as bc


ANCHORS = ["甲", "乙"]


def _pred_factory(edges):
    """edges: {(a, b): predicate}；有向。"""
    def f(a, b):
        return edges.get((a, b), "")
    return f


def test_default_is_off():
    """默认必须关：不加开关时整条通路不生效。"""
    assert bc.COOCCUR_DEFAULT is False


def test_returns_empty_without_two_anchors():
    """锚点不足两个 -> 不做任何事。"""
    f = _pred_factory({("丙", "甲"): "交谈"})
    assert bc.build_cooccur_sentences(
        anchors=["甲"], answer_sentences=["丙交谈甲"],
        candidates=["丙"], predicate_of=f) == []


def test_returns_empty_without_answers_or_candidates():
    f = _pred_factory({("丙", "甲"): "交谈"})
    assert bc.build_cooccur_sentences(
        anchors=ANCHORS, answer_sentences=[], candidates=["丙"], predicate_of=f) == []
    assert bc.build_cooccur_sentences(
        anchors=ANCHORS, answer_sentences=["丙交谈甲"], candidates=[], predicate_of=f) == []


def test_never_invents_a_predicate_when_no_edge():
    """**核心门**：图上没有边时必须返回空，不许兜底成任何模板词。"""
    f = _pred_factory({})          # 一条边都没有
    out = bc.build_cooccur_sentences(
        anchors=ANCHORS, answer_sentences=["丙交谈甲"],
        candidates=["丙"], predicate_of=f)
    assert out == [], "没有真实边却生成了句子 = 编造"


def test_only_uses_predicates_the_graph_returns():
    """生成的句子必须由 `predicate_of` 返回值拼成，断言其真实性。"""
    f = _pred_factory({("丙", "甲"): "交谈", ("丙", "乙"): "位于"})
    out = bc.build_cooccur_sentences(
        anchors=ANCHORS, answer_sentences=["丙交谈甲"],
        candidates=["丙"], predicate_of=f)
    assert out, "应当生成句子"
    for sent, x, a, pr in out:
        assert pr in ("交谈", "位于")
        assert sent in ("丙交谈甲", "丙位于乙")


def test_direction_is_respected_via_predicate_of():
    """**方向门**：`predicate_of` 只承认一个方向时，反向不得凭空成立。

    实测这条 bug 会生成 4 条语义反了的句子（`罗德岛隶属Scout` 等）：
    图上是 `Scout隶属罗德岛`，反向复述就错了。
    """
    # 只存在 (甲, 丙) 方向，即「甲谓词丙」
    f = _pred_factory({("甲", "丙"): "隶属"})
    out = bc.build_cooccur_sentences(
        anchors=ANCHORS, answer_sentences=["甲隶属丙"],
        candidates=["丙"], predicate_of=f)
    # 丙与乙之间没有任何边 -> 不许生成
    assert out == []


def test_skips_pairs_already_covered():
    """已共现的 (实体, 锚点) 不重复补。"""
    f = _pred_factory({("丙", "甲"): "交谈", ("丙", "乙"): "位于"})
    # 答案里已有一句「丙交谈甲」同时含丙与甲 -> 这一对已覆盖
    out = bc.build_cooccur_sentences(
        anchors=ANCHORS, answer_sentences=["丙交谈甲"],
        candidates=["丙"], predicate_of=f)
    pairs = {(x, a) for _, x, a, _ in out}
    assert ("丙", "甲") not in pairs


def test_anchor_cannot_be_the_intermediate_entity():
    """锚点自己不能当中间实体（判据里 `gold in anc` 直接判失败）。"""
    f = _pred_factory({("甲", "乙"): "交谈", ("甲", "甲"): "x"})
    out = bc.build_cooccur_sentences(
        anchors=ANCHORS, answer_sentences=["甲交谈乙"],
        candidates=["甲"], predicate_of=f)
    for _, x, _, _ in out:
        assert x not in ANCHORS


def test_respects_max_sentences_and_max_entities():
    f = _pred_factory({(x, a): "交谈" for x in ("丙", "丁", "戊") for a in ANCHORS})
    out = bc.build_cooccur_sentences(
        anchors=ANCHORS, answer_sentences=["丙交谈甲"],
        candidates=["丙", "丁", "戊"], predicate_of=f,
        max_entities=1, max_sentences=2)
    assert len(out) <= 2
    assert len({x for _, x, _, _ in out}) <= 1
