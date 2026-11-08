"""两条「新答案形态」通路的门：标题通路与章节包通路。

**为什么必须单独建这个文件**：这两条通路各影响 52 / 32 道题，却在此前**零测试覆盖**
——`git grep TITLE_ANSWER_PATH -- tests/` 与 `CHAPTER_ANSWER_PATH` 都返回空。它们在
2026-11-05 就被写进台账说「已入库、干净口径 +22 题」，但开关一直是 `False`，一次都
没在生产里跑过。本项目把这种形态记作「**没接线的判据等于不存在**」。

所以这里钉两件事，缺一不可：

1. **开关必须是 `on`**。任何人把它改回 `False`，这一组必须立刻变红——否则「入库了」
   又会退化成一句无人验证的台账文字。
2. **打开后的行为本身要成立**：答案只能来自 `unit["title"]` / `unit["chapter"]`，
   引用必须自洽（`text[start:end] == evidence`）且能过 `verify_citations`，
   判据不成立时必须返回 `None` 落回原路径（不许猜）。

第 2 条的存在意义是：只钉开关不钉行为，等于允许有人用「把开关打开」把门撑绿。
"""

from __future__ import annotations

import pytest

from arkrag import answer as answer_mod

# 一律显式 import，不依赖 conftest 的路径注入假设：测试文件不许用相对 cwd 路径。
MIN = answer_mod.MIN_EVIDENCE_CHARS


def _chunk(chunk_id: str, *, title: str, chapter: str, text: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": chunk_id.split("#")[0],
        "title": title,
        "chapter": chapter,
        "text": text,
    }


def _unit(chunk: dict, *, score: float = 1.0) -> dict:
    return {
        "chunk_id": chunk["chunk_id"],
        "doc_id": chunk["doc_id"],
        "title": chunk["title"],
        "chapter": chunk["chapter"],
        "text": chunk["text"],
        "assertions": [],
        "evidence_spans": [],
        "entities": [],
        "score": score,
        "sources": [{"channel": "bm25", "score": score}],
        "graph_path": [],
    }


def _ctx(units: list[dict]) -> dict:
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


def _answer(question: str, units: list[dict], *, qa: dict | None = None) -> dict:
    out = answer_mod.ExtractiveAnswerer().answer(
        question, _ctx(units), top_score=1.0, query_analysis=qa
    )
    # 每条引用都必须自洽：这是 `verify_citations` 的硬判据，也是这里最容易写错的地方
    for cite in out["citations"]:
        chunk = next(c for c in units if c["chunk_id"] == cite["chunk_id"])
        got = chunk["text"][int(cite["start"]):int(cite["end"])]
        assert got == cite["evidence"], (
            f"引用不自洽：{cite['chunk_id']} [{cite['start']},{cite['end']}) "
            f"取到 {got!r}，却声明 evidence={cite['evidence']!r}"
        )
        assert len(cite["evidence"]) >= MIN, (
            f"引用证据 {cite['evidence']!r} 短于 MIN_EVIDENCE_CHARS={MIN}，"
            "会被 verify_citations 判 evidence_too_short"
        )
    return out


# ------------------------------------------------------------------ 开关在不在线
class TestSwitchesAreWired:
    """两个开关必须是 on。改回 False 就红——这是「没接线的判据等于不存在」的反向实测。"""

    def test_title_answer_path_is_on(self):
        assert answer_mod.TITLE_ANSWER_PATH is True, (
            "TITLE_ANSWER_PATH 被改回 False 了。这条通路在 06a741 上实测"
            "gold-v1 端到端 106→158（干净 17 题）、gold-v2 210→314，回退 0 题、"
            "应拒题正确拒答数不变。关掉等于把这 +17/+35 的干净增益丢回台账里当文字。"
        )

    def test_chapter_answer_path_is_on(self):
        assert answer_mod.CHAPTER_ANSWER_PATH is True, (
            "CHAPTER_ANSWER_PATH 被改回 False 了。这条通路在 06a741 上实测"
            "cross_chunk 0/32→12/32、章节代号精度 26/28、召回 26/26，"
            "gold-v2 净增益 +22 题（314→336）零回退、应拒题拒答数 10/52 不变。"
        )

    def test_title_triggers_are_the_four_metadata_skeletons(self):
        """触发词不许被放宽：放宽就会把 158 道含「」引文的非标题题一起吃进来。"""
        assert answer_mod.TITLE_TRIGGERS == (
            "请给出该块的标题",
            "请回答该段的标题",
            "出自哪个文本块",
            "来自哪一段剧情",
        )

    def test_chapter_trigger_is_a_single_narrow_word(self):
        assert answer_mod.CHAPTER_TRIGGERS == ("章节代号",)


# ------------------------------------------------------------------ 标题通路
class TestTitleAnswerPath:
    TEXT = (
        "凯尔希：罗德岛不会放弃任何一个人。\n"
        "阿米娅：博士，我们该出发了。"
    )
    QUOTE = "罗德岛不会放弃任何一个人"
    QUESTION = f"下面这段原文「{QUOTE}」出自哪个文本块？"

    def _one(self) -> dict:
        return _chunk("a#001", title="怒号光明·上", chapter="main_ep08", text=self.TEXT)

    def test_answers_with_the_unit_title(self):
        out = _answer(self.QUESTION, [_unit(self._one())])
        assert out["abstained"] is False
        assert out["answer"] == "怒号光明·上"
        assert out["stats"]["title_path"]["applied"] is True
        assert out["stats"]["title_path"]["title_source"] == "unit.title"

    def test_emits_a_line_citation_that_passes_verify(self):
        chunks = {"a#001": self._one()}
        out = _answer(self.QUESTION, [_unit(self._one())])
        assert len(out["citations"]) == 1
        checked = answer_mod.verify_citations(out, chunks)
        assert checked["answer"] == "怒号光明·上", (
            "引用没过校验，答案被清空改判拒答了 —— 这就是 q-cx-900024 的事故形态"
        )
        assert checked.get("abstain_reason", "") == ""
        assert checked["verification"]["rejected"] == 0, (
            f"有引用被剔除：{checked['verification']['problems']}"
        )

    def test_answer_never_comes_from_the_question_text(self):
        """答案只能取自 title：换一个 title，答案必须跟着换，问句一个字都不许进答案。"""
        other = _chunk("a#001", title="完全不同的标题", chapter="main_ep08", text=self.TEXT)
        out = _answer(self.QUESTION, [_unit(other)])
        assert out["answer"] == "完全不同的标题"
        assert self.QUOTE not in out["answer"]

    def test_ambiguous_title_falls_back_instead_of_guessing(self):
        """命中块的 title 不唯一 ⇒ 不触发（`titles` 长度必须为 1）。"""
        a = _chunk("a#001", title="标题甲", chapter="c1", text=self.TEXT)
        b = _chunk("b#001", title="标题乙", chapter="c2", text=self.TEXT)
        out = _answer(self.QUESTION, [_unit(a), _unit(b), _unit(b, score=0.5)])
        assert out["stats"]["title_path"]["applied"] is False
        assert out["answer"] not in ("标题甲", "标题乙")

    def test_quote_shorter_than_min_evidence_does_not_trigger(self):
        q = "「谢谢。」出自哪个文本块？"
        short = _chunk("a#001", title="标题甲", chapter="c1", text="凯尔希：谢谢。")
        out = _answer(q, [_unit(short)])
        assert out["stats"]["title_path"]["applied"] is False

    def test_non_metadata_question_does_not_trigger(self):
        """含「」引文但问的是别名规范名——158 道这类题必须一道都不触发。"""
        q = f"「{self.QUOTE}」这句话的说话人是谁？"
        out = _answer(q, [_unit(self._one())], qa=_qa("凯尔希"))
        assert out["stats"]["title_path"]["applied"] is False

    def test_every_matching_chunk_gets_its_own_citation(self):
        """同一 title 的两个块逐字节重复时，两块都要发引用（修 q-sc-000064 的回退）。"""
        a = _chunk("a#001", title="同标题", chapter="c1", text=self.TEXT)
        b = _chunk("b#002", title="同标题", chapter="c2", text=self.TEXT)
        out = _answer(self.QUESTION, [_unit(a), _unit(b, score=0.9)])
        assert out["answer"] == "同标题"
        assert {c["chunk_id"] for c in out["citations"]} == {"a#001", "b#002"}

    def test_abstain_gates_still_win(self):
        """三道拒答闸在通路之前：问句不可答时即使能定位也不许作答。"""
        qa = dict(_qa("凯尔希"))
        qa["should_answer"] = False
        qa["reason"] = "not_answerable"
        out = answer_mod.ExtractiveAnswerer().answer(
            self.QUESTION, _ctx([_unit(self._one())]), top_score=1.0, query_analysis=qa
        )
        assert out["abstained"] is True
        assert out["answer"] == ""
        assert out["abstain_reason"].startswith("query_refused")


# ------------------------------------------------------------------ 章节包通路
class TestChapterAnswerPath:
    QUESTION = "整合运动在哪些章节代号里出现过？"

    def _chunks(self) -> list[dict]:
        return [
            _chunk("a#001", title="T1", chapter="main_ep08",
                   text="塔露拉：整合运动不会后退。"),
            _chunk("b#002", title="T2", chapter="act17mini",
                   text="整合运动的旗帜被举起来了。"),
        ]

    def test_lists_chapters_from_unit_field(self):
        out = _answer(self.QUESTION, [_unit(c) for c in self._chunks()],
                      qa=_qa("整合运动"))
        assert out["abstained"] is False
        assert out["answer"] == "act17mini、main_ep08"  # sorted()
        assert out["stats"]["chapter_path"]["applied"] is True
        assert out["stats"]["chapter_path"]["chapter_source"] == "unit.chapter"

    def test_chapter_names_come_from_the_field_not_the_path(self):
        """`chunk_id` 路径第二段是 `a`/`b`，字段值是 `main_ep08`/`act17mini`。

        截路径只能答对 7/38，读字段 38/38 —— 这条约束是硬要求。
        """
        out = _answer(self.QUESTION, [_unit(c) for c in self._chunks()],
                      qa=_qa("整合运动"))
        assert "a" not in out["answer"].split("、")
        assert "b" not in out["answer"].split("、")

    def test_each_chapter_carries_a_citation(self):
        chunks = {c["chunk_id"]: c for c in self._chunks()}
        out = _answer(self.QUESTION, [_unit(c) for c in self._chunks()],
                      qa=_qa("整合运动"))
        assert len(out["citations"]) == 2
        checked = answer_mod.verify_citations(out, chunks)
        assert checked["verification"]["rejected"] == 0, (
            f"有引用被剔除：{checked['verification']['problems']}"
        )
        assert checked["answer"] == "act17mini、main_ep08"

    def test_no_focus_entity_in_text_does_not_trigger(self):
        """正文不含焦点实体名的块不进答案；一块都不剩时返回 None 不猜章节。"""
        units = [_unit(_chunk("a#001", title="T", chapter="main_ep08",
                              text="凯尔希：天气不错。"))]
        out = _answer(self.QUESTION, units, qa=_qa("整合运动"))
        assert out["stats"]["chapter_path"]["applied"] is False

    def test_chapter_without_usable_citation_is_refused(self):
        """能组出章节但一条够长的引用都发不出 ⇒ 返回 None（不许发空引用答案）。

        `凯尔希` 只有 3 字，短于 MIN_EVIDENCE_CHARS=4，那一行不能当引用。
        """
        units = [_unit(_chunk("a#001", title="T", chapter="main_ep08", text="凯尔希"))]
        out = self._run(units)
        assert out["stats"]["chapter_path"]["applied"] is False

    def _run(self, units: list[dict]) -> dict:
        return answer_mod.ExtractiveAnswerer().answer(
            self.QUESTION, _ctx(units), top_score=1.0, query_analysis=_qa("凯尔希")
        )

    def test_partial_match_keeps_only_matching_chapters(self):
        chunks = [
            _chunk("a#001", title="T1", chapter="main_ep08", text="整合运动不会后退。"),
            _chunk("b#002", title="T2", chapter="act17mini", text="凯尔希在读书。"),
        ]
        out = _answer(self.QUESTION, [_unit(c) for c in chunks], qa=_qa("整合运动"))
        assert out["answer"] == "main_ep08"

    def test_no_entity_candidates_does_not_trigger(self):
        out = _answer(self.QUESTION, [_unit(c) for c in self._chunks()], qa=_qa())
        assert out["stats"]["chapter_path"]["applied"] is False


# ------------------------------------------------------------------ 两条通路互斥
class TestPathsDoNotShadowEachOther:
    def test_title_question_does_not_reach_chapter_path(self):
        text = "凯尔希：罗德岛不会放弃任何一个人。"
        unit = _unit(_chunk("a#001", title="怒号光明·上", chapter="main_ep08", text=text))
        q = "「罗德岛不会放弃任何一个人」出自哪个文本块？"
        out = _answer(q, [unit], qa=_qa("凯尔希"))
        assert out["answer"] == "怒号光明·上"
        assert "chapter_path" not in out["stats"]

    def test_chapter_question_does_not_reach_title_path(self):
        unit = _unit(_chunk("a#001", title="怒号光明·上", chapter="main_ep08",
                            text="整合运动不会后退。"))
        out = _answer("整合运动在哪些章节代号里出现过？", [unit], qa=_qa("整合运动"))
        assert out["answer"] == "main_ep08"
        assert out["stats"]["title_path"]["applied"] is False


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
