"""`arkrag.judge` 的门：每条判据都必须能拦住它对应的退化解。

## 为什么每个反例都是「实测见过的那个错」
这个文件的每条 `test_*_rejects_*` 都对应一次真跑里判错过的案例，
不是臆想的边界值。写的时候顺手把来源记在 docstring 里。

## 反向实测（写完之后实际跑的，`tmp_mutate_judge.py`，8/8 命中）
cross_chunk 去掉「额外章节必须真实提及」→ `test_rejects_extra_chapter_it_never_mentions` 红；
去掉「额外章节数不得超过 gold」→ `test_rejects_listing_every_chapter` 红；
不允许超集（回到集合相等）→ `test_accepts_bounded_real_extra_chapters` 红；
direct_relation `all`→`any` → `test_rejects_claim_about_a_different_subject` 红；
multihop 放宽成「任一锚点」→ `test_rejects_gold_linked_to_only_one_anchor` 红；
删 `if gold in anc` → `test_rejects_gold_that_is_an_anchor` 红；
章节分隔符去掉空白字符类 → `test_space_separated_chapters_are_split` 红；
结构类 `None`→`False` → `test_structural_categories_return_none` 红。
"""

from __future__ import annotations

import pytest

from arkrag import judge as judge_mod


def _item(answer, *, question="", acceptable=None):
    return {
        "question": question,
        "answer": answer,
        "acceptable_answers": acceptable or [],
    }


#: 章节正文，供 cross_chunk 的"额外章节是否真的提到这个实体"用。
#: 刻意让 `main_ep16` 提到「海洋」、`act99` 是**没提到**的空壳章。
CHAPTERS = {
    "act17side": "海洋在他们脚下翻涌。",
    "memory_deepcl": "深海的海洋记录。",
    "main_ep16": "海洋干涸了。",
    "act99": "这一章讲的是别的东西。",
}


class TestEmptyAndStructural:
    """空答案与结构类的返回值不许混淆：空是 False，结构类是 None。"""

    def test_empty_answer_is_false_not_none(self) -> None:
        r = judge_mod.judge_answer("single_chunk", "   ", _item("凯尔希"))
        assert r["pass"] is False
        assert r["reason"] == "empty"

    def test_abstain_never_counts_as_correct(self) -> None:
        """拒答（空答案）即使 gold 也是空串，也不能算对。"""
        r = judge_mod.judge_answer("single_chunk", "", _item(""))
        assert r["pass"] is False

    @pytest.mark.parametrize("cat", list(judge_mod.STRUCTURAL_CATEGORIES))
    def test_structural_categories_return_none(self, cat: str) -> None:
        r = judge_mod.judge_answer(cat, "时序演进", _item("时序演进"))
        assert r["pass"] is None, "结构类必须返回 None（剔除分母），不许算 0 分"
        assert r["reason"].startswith("structural")

    def test_scorable_categories_do_not_overlap_structural(self) -> None:
        assert set(judge_mod.SCORABLE_CATEGORIES).isdisjoint(
            set(judge_mod.STRUCTURAL_CATEGORIES))
        assert set(judge_mod.SCORABLE_CATEGORIES) | set(
            judge_mod.STRUCTURAL_CATEGORIES) == set(judge_mod.CATEGORIES)

    def test_unknown_category_is_false_not_crash(self) -> None:
        r = judge_mod.judge_answer("no_such_category", "x", _item("y"))
        assert r["pass"] is False
        assert r["reason"].startswith("unknown_category")


class TestSubstringCategories:
    """single_chunk / alias：gold 是短专名，子串命中即等价。"""

    @pytest.mark.parametrize("cat", ["single_chunk", "alias"])
    def test_gold_substring_passes(self, cat: str) -> None:
        r = judge_mod.judge_answer(cat, "答案是凯尔希。", _item("凯尔希"))
        assert r["pass"] is True

    @pytest.mark.parametrize("cat", ["single_chunk", "alias"])
    def test_acceptable_answer_passes(self, cat: str) -> None:
        r = judge_mod.judge_answer(
            cat, "时序演进", _item("不是矛盾", acceptable=["时序演进"]))
        assert r["pass"] is True

    @pytest.mark.parametrize("cat", ["single_chunk", "alias"])
    def test_wrong_answer_fails(self, cat: str) -> None:
        r = judge_mod.judge_answer(cat, "答案是阿米娅。", _item("凯尔希"))
        assert r["pass"] is False
        assert r["reason"] == "gold_absent"

    @pytest.mark.parametrize("cat", ["single_chunk", "alias"])
    def test_no_gold_is_false(self, cat: str) -> None:
        r = judge_mod.judge_answer(cat, "随便什么", _item(""))
        assert r["pass"] is False
        assert r["reason"] == "no_gold"


class TestCrossChunkNeedsBoundedRealExtras:
    """cross_chunk：gold ⊆ 答案；每个额外章节必须真的提到该实体；且额外数 ≤ gold 数。

    这三条缺一不可：
      * 只要求集合相等 → 惩罚正确答案（图登记不全时 gold 系统性漏章节，实测 44/64 题）。
      * 只要求超集 + 额外真实 → 被普通名词打满（`海洋` 3 章 → 47 章，ratio 14.7）。
      * 规模上限是同时挡住"列全部章节"与"列全部真实提及"的最小约束。
    """

    Q = "「海洋」这个词在哪些章节的正文里出现过？"

    def test_exact_set_passes(self) -> None:
        r = judge_mod.judge_answer(
            "cross_chunk", "act17side、memory_deepcl",
            _item("act17side、memory_deepcl", question=self.Q),
            chapter_texts=CHAPTERS)
        assert r["pass"] is True

    def test_accepts_bounded_real_extra_chapters(self) -> None:
        """图漏登记导致 gold 少了 `main_ep16`（该章正文确实提到「海洋」）。

        额外 1 章 <= gold 2 章，且该章真实提及 → 应判对。
        这就是"集合相等在惩罚正确答案"的那 19 题之一。
        """
        r = judge_mod.judge_answer(
            "cross_chunk", "act17side、memory_deepcl、main_ep16",
            _item("act17side、memory_deepcl", question=self.Q),
            chapter_texts=CHAPTERS)
        assert r["pass"] is True
        assert r["reason"] == "extra_chapters_all_real_and_bounded"

    def test_rejects_extra_chapter_it_never_mentions(self) -> None:
        """`act99` 的正文里根本没有「海洋」，列它就是编的。"""
        r = judge_mod.judge_answer(
            "cross_chunk", "act17side、memory_deepcl、act99",
            _item("act17side、memory_deepcl", question=self.Q),
            chapter_texts=CHAPTERS)
        assert r["pass"] is False
        assert r["reason"] == "extra_chapter_does_not_mention_entity"

    def test_rejects_listing_every_chapter(self) -> None:
        """把全部章节都列出来必须判失败（旧超集判据下这是 38/38 全过）。"""
        r = judge_mod.judge_answer(
            "cross_chunk", "act17side、memory_deepcl、main_ep16、act99",
            _item("act17side、memory_deepcl", question=self.Q),
            chapter_texts=CHAPTERS)
        assert r["pass"] is False
        assert r["reason"] in ("extra_chapter_does_not_mention_entity",
                               "extra_chapters_outnumber_gold")

    def test_rejects_real_extras_that_outnumber_gold(self) -> None:
        """额外章节**全是真的**、但数量超过 gold 数，也必须拒。

        这条单独钉住规模上限，否则「额外必须真实」这一条会把它的功劳盖掉：
        实测把上限删掉时，本用例是唯一能变红的那条（`tmp_mutate_judge.py`）。
        来源：`海洋` 题 gold 3 章、正文真提及 47 章，光靠"真实"判据挡不住普通名词膨胀。
        """
        r = judge_mod.judge_answer(
            "cross_chunk", "act17side、memory_deepcl、main_ep16",
            _item("act17side", question=self.Q),
            chapter_texts=CHAPTERS)
        assert r["pass"] is False
        assert r["reason"] == "extra_chapters_outnumber_gold", (
            "两章额外都是真提及（memory_deepcl 与 main_ep16 正文都写了「海洋」），"
            "只能靠规模上限拒掉")

    def test_rejects_missing_chapters(self) -> None:
        r = judge_mod.judge_answer(
            "cross_chunk", "act17side", _item("act17side、memory_deepcl", question=self.Q),
            chapter_texts=CHAPTERS)
        assert r["pass"] is False
        assert r["reason"] == "chapter_set_missing"

    def test_without_chapter_texts_it_degrades_to_set_equality(self) -> None:
        """没传正文时不许假装做过真实性校验，必须显式降级并写明原因。"""
        r = judge_mod.judge_answer(
            "cross_chunk", "act17side、memory_deepcl、main_ep16",
            _item("act17side、memory_deepcl", question=self.Q))
        assert r["pass"] is False
        assert r["reason"] == "extra_chapters_unverified"

    def test_space_separated_chapters_are_split(self) -> None:
        """gold 里同时出现过 `、` 和**空格**分隔（`act33side main_ep14`）。

        第一版 `_CHAPTER_SPLIT` 漏了 `\\s`，于是要求答案里出现整串
        `act33side main_ep14`，38 道 cross_chunk 题全假失败。
        """
        assert judge_mod.chapter_set("act33side main_ep14") == {"act33side", "main_ep14"}
        r = judge_mod.judge_answer(
            "cross_chunk", "act33side、main_ep14", _item("act33side main_ep14"))
        assert r["pass"] is True

    def test_order_does_not_matter(self) -> None:
        r = judge_mod.judge_answer(
            "cross_chunk", "main_ep14、act33side", _item("act33side、main_ep14"))
        assert r["pass"] is True

    def test_duplicates_in_answer_do_not_break_set_equality(self) -> None:
        r = judge_mod.judge_answer(
            "cross_chunk", "act33side、act33side、main_ep14",
            _item("act33side、main_ep14"))
        assert r["pass"] is True

    def test_empty_answer_fails(self) -> None:
        r = judge_mod.judge_answer("cross_chunk", "", _item("act33side"))
        assert r["pass"] is False


class TestDirectRelationNeedsAllAnchorsInOneClaim:
    """direct_relation：**一条**断言必须同时含全部锚点 + gold。"""

    def test_one_claim_with_all_anchors_and_gold_passes(self) -> None:
        q = "「菈塔托丝」与「布朗陶」之间的关系是什么？"
        r = judge_mod.judge_answer(
            "direct_relation", "菈塔托丝隶属布朗陶家。", _item("隶属", question=q))
        assert r["pass"] is True

    def test_rejects_claim_about_a_different_subject(self) -> None:
        """来源：`q-dr-900010` 问「红」与「罗德岛」，答案里有一条「温蒂隶属罗德岛」
        就被旧判据判过了——主语完全不是问的那个。

        这条用例同时钉住"不能用 any(锚点)"：答案里确实有 gold、也确实有一个锚点
        （罗德岛），但缺另一个锚点（红）。
        """
        q = "「红」与「罗德岛」之间的关系是什么？"
        r = judge_mod.judge_answer(
            "direct_relation", "温蒂隶属罗德岛。", _item("隶属", question=q))
        assert r["pass"] is False
        assert r["reason"] == "no_single_claim_with_all_anchors"

    def test_anchors_split_across_two_claims_do_not_count(self) -> None:
        q = "「红」与「罗德岛」之间的关系是什么？"
        r = judge_mod.judge_answer(
            "direct_relation", "红离开龙门。温蒂隶属罗德岛。",
            _item("隶属", question=q))
        assert r["pass"] is False, "两个锚点分在两条断言里，没有任何一条回答了这个关系"

    def test_gold_in_one_claim_with_both_anchors_passes(self) -> None:
        q = "「红」与「罗德岛」之间的关系是什么？"
        r = judge_mod.judge_answer(
            "direct_relation", "红隶属罗德岛。温蒂隶属罗德岛。",
            _item("隶属", question=q))
        assert r["pass"] is True


class TestMultihopNeedsGoldLinkedToEveryAnchor:
    """multihop：gold 必须与**每个**锚点各自共现。

    旧判据只要"gold + ≥1 锚点"，实测 c12_12000 通过的 73 题里 **57 题只命中 1 个锚点**，
    没有一题体现"经中间实体连接两端"。
    """

    Q = "从「云兽」出发，经过一个中间实体可以到达「变形者集群」。这个中间实体叫什么？"

    def test_gold_linked_to_every_anchor_passes(self) -> None:
        r = judge_mod.judge_answer(
            "multihop", "阿米娅交谈云兽。阿米娅交谈变形者集群。",
            _item("阿米娅", question=self.Q))
        assert r["pass"] is True
        assert r["reason"] == "gold_linked_to_every_anchor"

    def test_rejects_gold_linked_to_only_one_anchor(self) -> None:
        """来源：`q-mh-000001` 的实跑答案里只有「阿米娅交谈变形者集群」，
        `云兽` 一个字都没出现，旧判据（≥1 锚点）却判过了。
        """
        r = judge_mod.judge_answer(
            "multihop", "阿米娅交谈变形者集群。变形者集群交谈凯尔希。",
            _item("阿米娅", question=self.Q))
        assert r["pass"] is False
        assert r["reason"] == "gold_not_linked_to_every_anchor"

    def test_rejects_when_no_anchor_is_linked(self) -> None:
        r = judge_mod.judge_answer(
            "multihop", "完全无关的一句话。", _item("阿米娅", question=self.Q))
        assert r["pass"] is False
        assert r["reason"] == "no_claim_links_gold_to_any_anchor"

    def test_rejects_gold_that_is_an_anchor(self) -> None:
        """gold 本身就是问句里的词时，这条题等于白送（题干自带答案）。"""
        q = "「凯尔希」与「罗德岛」之间的中间实体是谁？"
        r = judge_mod.judge_answer(
            "multihop", "凯尔希隶属罗德岛。", _item("凯尔希", question=q))
        assert r["pass"] is False
        assert r["reason"] == "gold_is_an_anchor"


class TestAnchorsExtraction:
    """锚点抽取：只认 `「」`，且要能处理重复与过短内容。"""

    def test_extracts_in_order_and_dedupes(self) -> None:
        assert judge_mod.anchors("「甲」与「乙」还有「甲」") == ["甲", "乙"]

    def test_empty_question_returns_empty(self) -> None:
        assert judge_mod.anchors("") == []
        assert judge_mod.anchors(None) == []

    def test_short_anchors_are_a_weak_signal_not_a_filter(self) -> None:
        """`short_anchors` 只按长度切，**不能**把关系名与短实体名分开。

        `negative_abstain` 的关系名（婚约/祖孙/邻居）都在 3 字以内，
        但短实体名（塔拉、红、陈）也在 3 字以内，两者混在同一个列表里。
        所以它只能给人看，不许拿来当"这句问的是关系不是实体"的判据——
        实测按它做拒答护栏 P1/P2 各自误伤 26 道应答题（其中 15 道原本答对）。
        """
        q = "「0374小队」与「塔拉」之间是否存在「婚约」关系？"
        got = judge_mod.short_anchors(q)
        assert "婚约" in got
        assert "0374小队" not in got
        assert "塔拉" in got, "短实体名也会被切进来，这正是它不能当判据的原因"


class TestCitationsSupport:
    """引用支撑：逐条查偏移与章节覆盖，不并入答对判据。"""

    CHUNKS = {
        "a#001": {"text": "凯尔希走进罗德岛的走廊。", "chapter": "main_ep08"},
        "b#002": {"text": "红离开了龙门。", "chapter": "main_ep14"},
    }

    def _cite(self, cid, start, end):
        rec = self.CHUNKS[cid]
        return {"chunk_id": cid, "evidence": rec["text"][start:end],
                "start": start, "end": end}

    def test_good_citation_passes(self) -> None:
        c = self._cite("a#001", 0, 6)
        r = judge_mod.citations_support({"citations": [c]}, self.CHUNKS)
        assert r["ok"] is True
        assert r["chapters_covered"] == ["main_ep08"]

    def test_offsets_mismatch_is_flagged(self) -> None:
        c = self._cite("a#001", 0, 6)
        c["evidence"] = "阿米娅"
        r = judge_mod.citations_support({"citations": [c]}, self.CHUNKS)
        assert r["ok"] is False
        assert r["problems"][0]["reason"] == "evidence_does_not_match_offsets"

    def test_unknown_chunk_is_flagged(self) -> None:
        r = judge_mod.citations_support(
            {"citations": [{"chunk_id": "zz#9", "evidence": "x", "start": 0, "end": 1}]},
            self.CHUNKS)
        assert r["ok"] is False
        assert r["problems"][0]["reason"] == "unknown_chunk"

    def test_missing_offsets_is_flagged(self) -> None:
        r = judge_mod.citations_support(
            {"citations": [{"chunk_id": "a#001", "evidence": "凯尔希"}]}, self.CHUNKS)
        assert r["ok"] is False
        assert r["problems"][0]["reason"] == "offsets_missing"

    def test_chapter_without_citation_is_flagged(self) -> None:
        """答案列了两章，但只给了第一章的引用 → 第二章读不出来。"""
        c = self._cite("a#001", 0, 6)
        r = judge_mod.citations_support(
            {"citations": [c]}, self.CHUNKS, chapters="main_ep08、main_ep14")
        assert r["ok"] is False
        assert r["problems"][-1]["reason"] == "chapters_without_citation"
        assert r["problems"][-1]["chapters"] == ["main_ep14"]

    def test_all_chapters_covered_passes(self) -> None:
        cites = [self._cite("a#001", 0, 6), self._cite("b#002", 0, 5)]
        r = judge_mod.citations_support(
            {"citations": cites}, self.CHUNKS, chapters="main_ep08、main_ep14")
        assert r["ok"] is True
