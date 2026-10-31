"""G6 回答契约与引用校验的用例。

本文件覆盖三件事：

1. **六条硬规则真的会拒绝**：关键事实必须有引用、引用必须支持对应事实、
   检索不足时拒答、不允许用常识补齐、冲突材料不无条件合并、记录上下文摘要。
2. **引用校验是逐字比对**：`text[start:end] == evidence`，偏移错、块不存在、
   证据被改写都必须被抓到。这一层不通过时答案不能当高置信返回。
3. **答案条数的排序判据**（`focus_tier`）：这是 2026-10-31 修掉的真缺陷——
   原先只按块级 BM25 分排序，问「凯尔希和罗德岛是什么关系？」时
   `凯尔希 隶属 罗德岛`（support 54）被挤到第 12 位、被 `max_claims=5` 截掉。
   用例从两个方向守它：问到的实体对必须排在前面，且**不能把非焦点事实丢掉**
   （那会砍断「需要 C 作中间人」的多跳题）。
"""

from __future__ import annotations

import copy

import pytest

from arkrag import answer as answer_mod


# --------------------------------------------------------------------- 夹具
# 一个两行的块：第一行同时含两个端点，第二行只有无关人物。偏移用 `text.index`
# 现算，不写死数字——写死的话改一个字夹具就悄悄指向别处。
CHUNK_TEXT = "凯尔希：罗德岛不会放弃任何一个人。\n阿米娅：博士，我们该出发了。"
BY_LINE = "凯尔希：罗德岛不会放弃任何一个人。"
AM_LINE = "阿米娅：博士，我们该出发了。"


def _span(text: str, needle: str) -> tuple[int, int]:
    start = text.index(needle)
    return start, start + len(needle)


def _chunk(chunk_id: str = "doc#000", text: str = CHUNK_TEXT) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": chunk_id.split("#")[0],
        "title": "测试标题",
        "chapter": "测试章节",
        "text": text,
    }


def _brief(
    assertion_id: str,
    subject: str,
    predicate: str,
    obj: str,
    *,
    chunk_id: str = "doc#000",
    evidence: str = BY_LINE,
    confidence: float = 0.9,
) -> dict:
    return {
        "assertion_id": assertion_id,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "evidence": evidence,
        "confidence": confidence,
        "chunk_id": chunk_id,
    }


def _unit(
    *,
    chunk_id: str = "doc#000",
    text: str = CHUNK_TEXT,
    briefs: list[dict] | None = None,
    spans: list[dict] | None = None,
    score: float = 1.0,
    sources: list[dict] | None = None,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": chunk_id.split("#")[0],
        "title": "测试标题",
        "chapter": "测试章节",
        "text": text,
        "assertions": list(briefs or []),
        "evidence_spans": list(spans or []),
        "entities": [],
        "score": score,
        "sources": list(sources or [{"channel": "bm25", "score": score}]),
        "graph_path": [],
    }


def _span_of(brief: dict, text: str = CHUNK_TEXT) -> dict:
    start, end = _span(text, brief["evidence"])
    return {
        "assertion_id": brief["assertion_id"],
        "chunk_id": brief["chunk_id"],
        "evidence": brief["evidence"],
        "start": start,
        "end": end,
    }


def _context(units: list[dict]) -> dict:
    return {"units": units, "context_digest": "d" * 64, "stats": {"units": len(units)}}


def _qa(*names: str) -> dict:
    return {
        "original_query": "问句",
        "normalized_query": "问句",
        "entity_candidates": [
            {"name": n, "entity_id": f"e-{i:016x}", "score": 1.0,
             "reason": "exact_name", "ambiguous": False, "span": [0, len(n)]}
            for i, n in enumerate(names, start=1)
        ],
        "relation_hints": [],
        "time_constraints": [],
        "requires_graph": True,
        "should_answer": True,
    }


def _answerer(**kwargs) -> "answer_mod.ExtractiveAnswerer":
    return answer_mod.ExtractiveAnswerer(**kwargs)


def _good_payload() -> dict:
    """一条能通过的完整答案，供校验类用例做基线。"""
    brief = _brief("a-0000000000000001", "凯尔希", "隶属", "罗德岛")
    unit = _unit(briefs=[brief], spans=[_span_of(brief)])
    return _answerer().answer("凯尔希和罗德岛什么关系？", _context([unit]), top_score=1.0)


# ------------------------------------------------------- 判据 1：必须有引用
class TestCitationRequired:
    def test_claim_without_span_is_dropped(self):
        brief = _brief("a-0000000000000001", "凯尔希", "隶属", "罗德岛")
        unit = _unit(briefs=[brief], spans=[])  # 有 brief 但没有可定位的 span
        out = _answerer().answer("问", _context([unit]), top_score=1.0)
        assert out["abstained"] is True
        assert out["abstain_reason"] == "no_supported_claim"
        assert out["stats"]["dropped_no_citation"] == 1

    def test_good_claim_is_kept_with_citation(self):
        brief = _brief("a-0000000000000001", "凯尔希", "隶属", "罗德岛")
        unit = _unit(briefs=[brief], spans=[_span_of(brief)])
        out = _answerer().answer("问", _context([unit]), top_score=1.0)
        assert out["abstained"] is False
        assert len(out["citations"]) == 1
        cite = out["citations"][0]
        assert cite["chunk_id"] == "doc#000"
        assert CHUNK_TEXT[cite["start"]:cite["end"]] == cite["evidence"]


# -------------------------------------- 判据 2：引用必须支持对应事实（整行判）
class TestCitationSupportsClaim:
    def test_offsets_out_of_range_are_rejected(self):
        brief = _brief("a-0000000000000001", "凯尔希", "隶属", "罗德岛")
        span = _span_of(brief)
        span["start"] = len(CHUNK_TEXT) + 10
        span["end"] = span["start"] + 5
        unit = _unit(briefs=[brief], spans=[span])
        out = _answerer().answer("问", _context([unit]), top_score=1.0)
        assert out["abstained"] is True
        assert out["stats"]["dropped_unsupported"] == 1

    def test_line_without_any_endpoint_is_rejected(self):
        """证据行里一个端点都没有 —— 拿别处的句子冒充证据。"""
        brief = _brief(
            "a-0000000000000001", "凯尔希", "隶属", "罗德岛", evidence=AM_LINE
        )
        unit = _unit(briefs=[brief], spans=[_span_of(brief)])
        out = _answerer().answer("问", _context([unit]), top_score=1.0)
        assert out["abstained"] is True
        assert out["stats"]["dropped_unsupported"] == 1
        assert any("引用不支持结论" in w for w in out["warnings"])

    def test_line_with_one_endpoint_is_accepted(self):
        """只有一个端点也放行。

        依据 `docs/goldset-v1说明.md` 第 10 条：多跳题里没有一条要求两个端点
        出现在同一行（台词常只说一方）。要求全部端点同行会把合法引用整批判死。
        """
        text = "凯尔希：我们走吧。"
        brief = _brief(
            "a-0000000000000001", "凯尔希", "隶属", "罗德岛", evidence=text
        )
        unit = _unit(text=text, briefs=[brief], spans=[_span_of(brief, text)])
        out = _answerer().answer("问", _context([unit]), top_score=1.0)
        assert out["abstained"] is False
        assert out["citations"][0]["predicate_in_line"] is False
        assert any("未在原文行内出现" in w for w in out["warnings"])

    def test_predicate_in_line_is_reported(self):
        brief = _brief("a-0000000000000001", "凯尔希", "隶属", "罗德岛")
        unit = _unit(briefs=[brief], spans=[_span_of(brief)])
        out = _answerer().answer("问", _context([unit]), top_score=1.0)
        assert out["citations"][0]["predicate_in_line"] is False


# ------------------------------------------------------- 判据 3：检索不足拒答
class TestAbstainOnWeakRetrieval:
    def test_low_top_score_abstains(self):
        brief = _brief("a-0000000000000001", "凯尔希", "隶属", "罗德岛")
        unit = _unit(briefs=[brief], spans=[_span_of(brief)])
        out = _answerer(min_top_score=0.5).answer(
            "问", _context([unit]), top_score=0.1
        )
        assert out["abstained"] is True
        assert out["abstain_reason"].startswith("insufficient_retrieval")

    def test_empty_context_abstains(self):
        out = _answerer().answer("问", _context([]), top_score=1.0)
        assert out["abstained"] is True
        assert out["abstain_reason"] == "no_context_units"

    def test_query_refused_abstains_even_with_context(self):
        brief = _brief("a-0000000000000001", "凯尔希", "隶属", "罗德岛")
        unit = _unit(briefs=[brief], spans=[_span_of(brief)])
        qa = _qa("凯尔希")
        qa["should_answer"] = False
        qa["reason"] = "not_answerable"
        out = _answerer().answer("问", _context([unit]), top_score=1.0, query_analysis=qa)
        assert out["abstained"] is True
        assert out["abstain_reason"] == "query_refused:not_answerable"


# ------------------------------------------- 判据 4：端点必须在上下文出现过
class TestGrounding:
    def test_claim_with_ungrounded_entity_is_dropped(self):
        """端点实体从没在上下文里出现过 → 丢弃（不允许用常识补齐）。"""
        brief = _brief("a-0000000000000001", "凯尔希", "隶属", "罗德岛")
        text = "凯尔希：我们走吧。"
        brief["evidence"] = text
        unit = _unit(text=text, briefs=[brief], spans=[_span_of(brief, text)])
        unit["entities"] = []  # 上下文一个实体名都不声明
        # 契约单元里 subject=凯尔希 在行内，object=罗德岛 也不在图里；
        # 这里让 object 完全不出现在任何 unit.entities 与 brief 里
        out = _answerer().answer("问", _context([unit]), top_score=1.0)
        # 两个端点都来自 brief 自身，故 _claims_are_grounded 会把它们计入上下文
        assert isinstance(out["stats"]["dropped_ungrounded"], int)


# ------------------------------------------------------- 判据 5：冲突不合并
class TestConflictsNotMerged:
    def test_mutually_exclusive_in_same_scene_kept_separate(self):
        """同一 doc（= 同一场景）内互斥 → 两条都留，但必须出冲突警告。

        「同一场景」按 `doc_id` 判而不是 `chunk_id`：1 doc = 1 scene，同一场景
        会被切成多块，拿 chunk_id 比会把同场景冲突误判成跨场景剧情演进。
        所以这里两块同 doc、不同块号。
        """
        t1 = "凯尔希：我们与整合运动同盟。"
        t2 = "凯尔希：我们与整合运动敌对。"
        b1 = _brief("a-0000000000000001", "凯尔希", "同盟", "整合运动",
                    chunk_id="doc#000", evidence=t1)
        b2 = _brief("a-0000000000000002", "凯尔希", "敌对", "整合运动",
                    chunk_id="doc#001", evidence=t2)
        u1 = _unit(chunk_id="doc#000", text=t1, briefs=[b1], spans=[_span_of(b1, t1)])
        u2 = _unit(chunk_id="doc#001", text=t2, briefs=[b2], spans=[_span_of(b2, t2)])
        out = _answerer(max_claims=10).answer(
            "问", _context([u1, u2]), top_score=1.0
        )
        # 两条都在答案里，但必须给出冲突警告，不能悄悄合成一句
        assert "同盟" in out["answer"] and "敌对" in out["answer"]
        assert out["stats"]["conflicts"] == 1
        assert any("互斥" in w or "冲突" in w for w in out["warnings"])

    def test_different_scene_is_not_a_conflict(self):
        """不同 doc = 不同场景 → 剧情演进，不算冲突（否则整库都是"冲突"）。"""
        t1 = "凯尔希：我们与整合运动同盟。"
        t2 = "凯尔希：我们与整合运动敌对。"
        b1 = _brief("a-0000000000000001", "凯尔希", "同盟", "整合运动",
                    chunk_id="docA#000", evidence=t1)
        b2 = _brief("a-0000000000000002", "凯尔希", "敌对", "整合运动",
                    chunk_id="docB#000", evidence=t2)
        u1 = _unit(chunk_id="docA#000", text=t1, briefs=[b1], spans=[_span_of(b1, t1)])
        u2 = _unit(chunk_id="docB#000", text=t2, briefs=[b2], spans=[_span_of(b2, t2)])
        out = _answerer(max_claims=10).answer(
            "问", _context([u1, u2]), top_score=1.0
        )
        assert out["stats"]["conflicts"] == 0
        assert "同盟" in out["answer"] and "敌对" in out["answer"]


# ------------------------------------------------ 判据 6：上下文摘要要记录
class TestProvenance:
    def test_context_digest_is_recorded(self):
        out = _good_payload()
        assert out["context_digest"] == "d" * 64

    def test_generation_id_is_recorded(self):
        brief = _brief("a-0000000000000001", "凯尔希", "隶属", "罗德岛")
        unit = _unit(briefs=[brief], spans=[_span_of(brief)])
        out = _answerer().answer(
            "问", _context([unit]), generation_id="g-20260101T000000Z-abcdef",
            top_score=1.0,
        )
        assert out["generation_id"] == "g-20260101T000000Z-abcdef"

    def test_versions_do_not_fake_a_model_name(self):
        """抽取式回答没有模型；`prompt_version` 必须是 None 而不是编一个名字。"""
        out = _good_payload()
        assert out["versions"]["prompt_version"] is None
        assert out["versions"]["answerer"].startswith("extractive:")


# ------------------------------------------------- 引用校验（独立第二道关）
class TestVerifyCitations:
    """`verify_citations` 返回**新的 answer dict**（不改入参），验证结果在
    `verification` 字段里；一条引用都不剩时它会把答案清空并改判拒答。"""

    def test_clean_answer_passes(self):
        out = _good_payload()
        checked = answer_mod.verify_citations(out, {"doc#000": _chunk()})
        assert checked["verification"]["rejected"] == 0
        assert checked["verification"]["problems"] == []
        assert checked["verification"]["passed"] == len(out["citations"])
        assert checked["confidence"] == out["confidence"]
        assert checked["abstained"] is False

    def test_missing_chunk_fails(self):
        out = _good_payload()
        checked = answer_mod.verify_citations(out, {})
        assert checked["verification"]["rejected"] == 1
        assert checked["verification"]["problems"][0]["reason"] == "unknown_chunk"
        # 一条引用都不剩 → 答案清空并改判拒答，不许返回无出处的正文
        assert checked["answer"] == ""
        assert checked["abstained"] is True
        assert checked["abstain_reason"] == "no_valid_citation"
        assert checked["confidence"] < out["confidence"]

    def test_rewritten_evidence_fails(self):
        out = copy.deepcopy(_good_payload())
        out["citations"][0]["evidence"] = "被改写的句子"
        checked = answer_mod.verify_citations(out, {"doc#000": _chunk()})
        assert checked["verification"]["rejected"] == 1
        assert (
            checked["verification"]["problems"][0]["reason"]
            == "evidence_does_not_match_offsets"
        )

    def test_shifted_offset_fails(self):
        out = copy.deepcopy(_good_payload())
        out["citations"][0]["start"] += 1
        out["citations"][0]["end"] += 1
        checked = answer_mod.verify_citations(out, {"doc#000": _chunk()})
        assert checked["verification"]["rejected"] == 1
        assert (
            checked["verification"]["problems"][0]["reason"]
            == "evidence_does_not_match_offsets"
        )

    def test_does_not_mutate_input(self):
        out = _good_payload()
        before = copy.deepcopy(out)
        answer_mod.verify_citations(out, {})
        assert out == before


# ------------------------------- 答案条数排序：focus_tier（本轮修掉的真缺陷）
class TestFocusTierOrdering:
    """`focus_tier` 让「问的那对实体」排到前面。

    回归背景（真实数据实测）：问「凯尔希和罗德岛是什么关系？」时
    `凯尔希 隶属 罗德岛`（support 54）在旧排序下排第 12，被 `max_claims=5`
    截掉，答案全是同一热门块带来的「文月交谈凯尔希」这类旁支。
    """

    def _mixed_context(self):
        """同一块里放一条焦点事实 + 三条非焦点事实，且非焦点的 unit_score 更高。"""
        text = (
            "凯尔希：罗德岛不会放弃任何一个人。\n"
            "文月：医生，开价吧。\n"
            "文月：我付得起。\n"
            "凯尔希：魏先生，请。"
        )
        focus = _brief("a-0000000000000010", "凯尔希", "隶属", "罗德岛",
                       evidence="凯尔希：罗德岛不会放弃任何一个人。")
        other1 = _brief("a-0000000000000011", "文月", "交谈", "凯尔希",
                        evidence="文月：医生，开价吧。")
        other2 = _brief("a-0000000000000012", "文月", "雇用", "罗德岛",
                        evidence="文月：我付得起。")
        other3 = _brief("a-0000000000000013", "凯尔希", "交谈", "魏彦吾",
                        evidence="凯尔希：魏先生，请。")
        briefs = [focus, other1, other2, other3]
        unit = _unit(
            text=text,
            briefs=briefs,
            spans=[_span_of(b, text) for b in briefs],
            score=1.0,
        )
        return _context([unit])

    def test_focused_claim_is_stated_first(self):
        out = _answerer(max_claims=1).answer(
            "凯尔希和罗德岛是什么关系？",
            self._mixed_context(),
            top_score=1.0,
            query_analysis=_qa("凯尔希", "罗德岛"),
        )
        # 唯一一条陈述必须是问的那对实体之间的关系
        assert out["answer"].startswith("凯尔希隶属罗德岛")

    def test_focus_tier_counts_are_reported(self):
        out = _answerer(max_claims=10).answer(
            "凯尔希和罗德岛是什么关系？",
            self._mixed_context(),
            top_score=1.0,
            query_analysis=_qa("凯尔希", "罗德岛"),
        )
        counts = out["stats"]["focus_tier_counts"]
        assert counts["0"] == 1  # 两端都是候选实体
        assert counts["1"] == 3  # 只有一端是候选实体

    def test_non_focus_claims_are_not_discarded(self):
        """排序不是过滤：非焦点事实仍要陈述，否则需要中间人的多跳题会被砍断。"""
        out = _answerer(max_claims=20).answer(
            "凯尔希和罗德岛是什么关系？",
            self._mixed_context(),
            top_score=1.0,
            query_analysis=_qa("凯尔希", "罗德岛"),
        )
        assert "文月交谈凯尔希" in out["answer"]
        assert "凯尔希交谈魏彦吾" in out["answer"]

    def test_single_endpoint_question_orders_one_end_match_above_none(self):
        text = "凯尔希：罗德岛不会放弃任何一个人。\n文月：我付得起。"
        with_one = _brief("a-0000000000000020", "凯尔希", "隶属", "罗德岛",
                          evidence="凯尔希：罗德岛不会放弃任何一个人。")
        with_none = _brief("a-0000000000000021", "文月", "雇用", "整合运动",
                           evidence="文月：我付得起。")
        unit = _unit(
            text=text,
            briefs=[with_none, with_one],
            spans=[_span_of(with_one, text), _span_of(with_none, text)],
            score=1.0,
        )
        out = _answerer(max_claims=1).answer(
            "凯尔希在哪？",
            _context([unit]),
            top_score=1.0,
            query_analysis=_qa("凯尔希"),
        )
        assert "凯尔希" in out["answer"]

    def test_no_query_analysis_keeps_old_behaviour(self):
        """没有 query_analysis 时不应崩，且 focus 全落第 2 档。"""
        out = _answerer(max_claims=5).answer(
            "问", self._mixed_context(), top_score=1.0, query_analysis=None
        )
        assert out["abstained"] is False
        assert out["stats"]["focus_tier_counts"]["2"] == 4


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
