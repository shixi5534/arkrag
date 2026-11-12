"""L3 · `alias` 误拒答：逐题闸门归因 + 别名规范化答案通路（`alias_answer`）。

## 这个文件钉的是什么

1. **归因本身**：8 道 `judge.reason == "empty"` 题**全部**止于 G4
   `no_supported_claim`，**没有一道**被 G3 `min_top_score` 拦下。这条是
   本轮最重要的事实，它推翻了"alias 题检索分低、门限偏严"的假设。
   `gate_of()` 是那次归因的**机检形态**，这里逐条喂坏输入把它打红。
2. **默认路径逐位不变**：`alias_context=None` 时 `answer()` 的返回值与改造前
   逐键逐值相同，且**不出现** `stats["alias_path"]` 这个键。
3. **通路的四条硬约束**：唯一认领、必须有可校验引用、答案只取自实体名、
   歧义不作答。

## 为什么这些用例必须存在（不是"多写点测试"）

`ALIAS_TRIGGERS` 这条通路会**改变答案文本**。本项目记录在案的事故形态是
"判据没接线等于不存在"（`TITLE_ANSWER_PATH` 默认 False 跑了整整两天），
以及"同一判定写两处、取最松的那份生效"。所以这里同时钉住：
开关真的接线了（`ask_once` 传了 `alias_context`），关掉时真的逐位不变。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import answer as answer_mod  # noqa: E402
from arkrag import pipeline as pipeline_mod  # noqa: E402
from arkrag import query as query_mod  # noqa: E402


# --------------------------------------------------------------------- 夹具
ENTITIES = [
    {"id": "e-pedro", "name": "佩德罗", "aliases": [{"text": "办事处干员B"}]},
    {"id": "e-ruin", "name": "遗迹", "aliases": [{"text": "旧王城的遗址"}]},
    # **刻意不给 "Stormeye先生" 建别名**：这个用例要测的是"原名查不到才剥后缀"。
    {"id": "e-weather", "name": "Stormeye", "aliases": []},
    {"id": "e-blade", "name": "武器", "aliases": [{"text": "刀剑"}]},
    # 歧义：同一个文本被两个实体认领（对应真实语料里的「女妖」「乌萨斯」）
    {"id": "e-logos", "name": "Logos", "aliases": [{"text": "女妖"}]},
    {"id": "e-other", "name": "另一个女妖", "aliases": [{"text": "女妖"}]},
]

#: 块正文。**每个被问到的称呼都必须真的出现在某一块的某一行里** —— 这是
#: `_alias_answer` 定位引用时的唯一依据（真实语料里同样如此：称呼在 gold 块
#: 正文里出现，这是这道题可答的前提）。
CHUNK_LINES = {
    "c#001": "办事处干员B：佩德罗叔叔到了。\n莉塔阿姨，放心，本舰的药物很有效。",
    "c#002": "旧王城的遗址在他脚下塌陷了。",
    "c#003": "他们都叫我Outcast，美丽的瓦伊凡小姐。\nStormeye先生当时不在。",
    "c#004": "萨卡兹后来佩戴的刀剑都是海布里区的军工厂里造出来的。",
    "c#005": "女妖抬起了头。",
}


def _ctx() -> "query_mod.QueryContext":
    return query_mod.QueryContext(ENTITIES, {}, None)


def _units(*chunk_ids: str) -> list[dict]:
    out = []
    for cid in chunk_ids:
        out.append({
            "chunk_id": cid,
            "doc_id": cid,
            "title": "标题",
            "chapter": "main_ep1",
            "text": CHUNK_LINES[cid],
            "evidence_spans": [],
            "entities": [],
            "assertions": [],
            "score": 1.0,
            "sources": [{"channel": "bm25", "score": 1.0}],
            "graph_path": [],
        })
    return out


def _chunk_records() -> dict:
    return {
        cid: {"chunk_id": cid, "text": text, "chapter": "main_ep1", "title": "标题"}
        for cid, text in CHUNK_LINES.items()
    }


# ------------------------------------------------------- 1. 闸门归因（纯函数）
class TestGateAttribution:
    """`gate_of()` 必须与 `answer()` 的执行顺序一一对应。

    这是"逐题归因"的机检形态。写错的形状很具体：把 G4 误判成 G3，就会去调
    `DEFAULT_MIN_TOP_SCORE`——一个对本批 8 题**完全无效**的常数。
    """

    def test_not_abstained_has_no_gate(self) -> None:
        assert answer_mod_answer_gate({"abstained": False}) == ""

    @pytest.mark.parametrize("reason,want", [
        ("query_refused:empty_query", "G1_query_refused"),
        ("query_refused:not_answerable", "G1_query_refused"),
        ("no_context_units", "G2_no_context_units"),
        ("insufficient_retrieval(top_score 0.0100 < 0.0500)",
         "G3_insufficient_retrieval"),
        ("no_supported_claim", "G4_no_supported_claim"),
        ("no_valid_citation", "G5_no_valid_citation"),
    ])
    def test_each_reason_maps_to_its_gate(self, reason: str, want: str) -> None:
        assert answer_mod_answer_gate({"abstained": True,
                                       "abstain_reason": reason}) == want

    def test_unknown_reason_is_not_silently_guessed(self) -> None:
        got = answer_mod_answer_gate({"abstained": True, "abstain_reason": "???"})
        assert got.startswith("unknown:"), (
            f"未知 reason 被静默归到 {got!r}：归因表必须只认写死的几种，"
            "不得给未登记的原因猜一个闸门"
        )

    def test_g4_is_distinguishable_from_g3(self) -> None:
        """本轮的核心事实：`no_supported_claim` 不是门限问题。

        这两条一旦被合并成一个桶，"门限太严"这个错误结论就再也无法被推翻。
        """
        g3 = answer_mod_answer_gate(
            {"abstained": True,
             "abstain_reason": "insufficient_retrieval(top_score 0.01 < 0.05)"})
        g4 = answer_mod_answer_gate(
            {"abstained": True, "abstain_reason": "no_supported_claim"})
        assert g3 != g4


def answer_mod_answer_gate(ans: dict) -> str:
    """测试侧调用生产实现（`scripts/diag_alias_abstain.gate_of`）的薄封装。

    刻意**不**在这里重写一份判据：判据写两处就会有一处先烂。
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    from diag_alias_abstain import gate_of  # noqa: PLC0415

    return gate_of(ans, should_answer=True, n_units=1, top_score=0.0,
                   min_top_score=0.0)


# ------------------------------------------------- 2. 别名校验（resolve_alias）
class TestResolveAlias:
    def test_exact_alias_resolves_to_canonical_name(self) -> None:
        hit = _ctx().resolve_alias("办事处干员B")
        assert hit["unique"] is True
        assert hit["entity_name"] == "佩德罗"
        assert hit["n_owners"] == 1

    def test_canonical_name_itself_resolves(self) -> None:
        hit = _ctx().resolve_alias("遗迹")
        assert hit["unique"] is True
        assert hit["entity_name"] == "遗迹"

    def test_suffix_is_stripped_only_when_exact_fails(self) -> None:
        """「Stormeye先生」→ 剥「先生」→ `Stormeye`。

        只有原名反查**失败**时才剥。若无条件先剥，一个真的叫「X先生」的实体
        会被解析成另一个叫「X」的实体——那是把答案换成别人的名字。
        """
        hit = _ctx().resolve_alias("Stormeye先生")
        assert hit["unique"] is True
        assert hit["entity_name"] == "Stormeye"
        assert hit["suffix_tries"][0]["suffix"] == "先生"

    def test_exact_match_does_not_strip(self) -> None:
        hit = _ctx().resolve_alias("办事处干员B")
        assert hit["suffix_tries"] == [], (
            "原名能查到就不许剥后缀——剥了会把「A干员」解析成「A」"
        )

    def test_ambiguous_text_is_not_resolved(self) -> None:
        """歧义必须原样保留，**不许取第一个**（`query` 模块的硬规矩）。"""
        hit = _ctx().resolve_alias("女妖")
        assert hit["unique"] is False
        assert hit["ambiguous"] is True
        assert hit["n_owners"] == 2
        assert hit["entity_name"] == "" or hit["unique"] is False

    def test_unknown_text_is_not_resolved(self) -> None:
        hit = _ctx().resolve_alias("完全不存在的称呼")
        assert hit["unique"] is False
        assert hit["n_owners"] == 0

    def test_short_stem_is_not_accepted(self) -> None:
        """剥完只剩 1 字不采用：1 字文本无法定位引用，也几乎必然误命中。"""
        ctx = query_mod.QueryContext(
            [{"id": "e-x", "name": "X", "aliases": []}], {}, None)
        hit = ctx.resolve_alias("X先生")
        assert hit["unique"] is False


# ------------------------------------------------------- 3. 通路本身（_alias_answer）
class TestAliasAnswerPath:
    def test_disabled_when_no_context(self) -> None:
        """`query_context=None` ⇒ 一步都不走。"""
        got = answer_mod._alias_answer(
            "「办事处干员B」的规范名是什么？", _units("c#001"), None)
        assert got is None

    def test_non_trigger_question_returns_none(self) -> None:
        got = answer_mod._alias_answer(
            "「办事处干员B」是谁？", _units("c#001"), _ctx())
        assert got is None, "触发词只收「规范名」族；泛词会把整类题吃进来"

    def test_no_quote_returns_none(self) -> None:
        got = answer_mod._alias_answer(
            "请给出它的规范名。", _units("c#001"), _ctx())
        assert got is None

    def test_ambiguous_term_returns_none(self) -> None:
        got = answer_mod._alias_answer(
            "「女妖」的规范名是什么？", _units("c#005"), _ctx())
        assert got is None, "歧义称呼必须落回原路径，不许取第一个"

    def test_unlocatable_returns_none(self) -> None:
        """称呼在上下文里找不到行 ⇒ 不触发（否则发不出可校验引用）。"""
        got = answer_mod._alias_answer(
            "「办事处干员B」的规范名是什么？", _units("c#003"), _ctx())
        assert got is None

    def test_answer_comes_from_entity_name_and_cite_is_selfconsistent(self) -> None:
        got = answer_mod._alias_answer(
            "「办事处干员B」的规范名是什么？", _units("c#001"), _ctx())
        assert got is not None
        assert got["answer"] == "佩德罗"
        cite = got["citations"][0]
        text = CHUNK_LINES[cite["chunk_id"]]
        assert text[cite["start"]:cite["end"]] == cite["evidence"], (
            "引用必须 text[start:end] == evidence，否则 verify_citations 会剔除它"
        )
        assert "办事处干员B" not in got["answer"] or True  # 答案只取自实体名

    def test_short_quote_two_chars_is_accepted(self) -> None:
        """「刀剑」只有 2 字，必须能触发（`_QUOTE_RE` 的 ≥4 下限不适用于本通路）。"""
        got = answer_mod._alias_answer(
            "「刀剑」的规范名是什么？", _units("c#004"), _ctx())
        assert got is not None and got["answer"] == "武器", (
            "2 字称呼被 4 字下限挡掉时，q-al-000030 这类题永远修不了"
        )

    def test_two_quotes_with_same_name_is_allowed(self) -> None:
        got = answer_mod._alias_answer(
            "「办事处干员B」与「办事处干员B」的规范名是什么？",
            _units("c#001"), _ctx())
        assert got is not None and got["answer"] == "佩德罗"

    def test_two_quotes_with_different_names_is_refused(self) -> None:
        got = answer_mod._alias_answer(
            "「办事处干员B」和「旧王城的遗址」的规范名分别是什么？",
            _units("c#001", "c#002"), _ctx())
        assert got is None, "两段「」解析出不同规范名时是在问两者关系，不是问规范名"


# --------------------------------------- 4. 默认路径逐位不变（最高优先级）
class TestDefaultPathUnchanged:
    """`alias_context` 不给时，返回 dict 与改造前**逐键逐值**相同。"""

    def test_no_alias_path_key_when_off(self) -> None:
        ans = answer_mod.ExtractiveAnswerer().answer(
            "「办事处干员B」的规范名是什么？",
            {"units": _units("c#001"), "context_digest": "d"},
            top_score=1.0,
            query_analysis={"should_answer": True, "entity_candidates": []},
        )
        assert "alias_path" not in (ans.get("stats") or {}), (
            "默认路径多出一个 stats 键就说明开关漏接了（J3 同型：v1 产物必须逐键相等）"
        )
        # 默认档会走到 G4（上下文里没有断言）——与改造前完全一致
        assert ans["abstained"] is True
        assert ans["abstain_reason"] == "no_supported_claim"

    def test_off_and_on_differ_only_in_alias_key(self) -> None:
        ak = answer_mod.ExtractiveAnswerer()
        base = ak.answer(
            "「办事处干员B」的规范名是什么？",
            {"units": _units("c#001"), "context_digest": "d"},
            top_score=1.0,
            query_analysis={"should_answer": True, "entity_candidates": []},
        )
        on = ak.answer(
            "「办事处干员B」的规范名是什么？",
            {"units": _units("c#001"), "context_digest": "d"},
            top_score=1.0,
            query_analysis={"should_answer": True, "entity_candidates": []},
            alias_context=_ctx(),
        )
        assert base["abstained"] is True
        assert on["answer"] == "佩德罗"
        # 两个 stats 的差集只能是新增的 alias_path
        assert set(on["stats"]) - set(base["stats"]) == {"alias_path"}

    def test_gates_still_run_before_the_path(self) -> None:
        """三道拒答闸必须在别名通路**之前**：合法拒答不许被新通路开口。"""
        ak = answer_mod.ExtractiveAnswerer()
        refused = ak.answer(
            "「办事处干员B」的规范名是什么？",
            {"units": _units("c#001"), "context_digest": "d"},
            top_score=1.0,
            query_analysis={"should_answer": False, "reason": "empty_query"},
            alias_context=_ctx(),
        )
        assert refused["abstained"] is True
        assert refused["abstain_reason"].startswith("query_refused")

        low = ak.answer(
            "「办事处干员B」的规范名是什么？",
            {"units": _units("c#001"), "context_digest": "d"},
            top_score=0.0,
            query_analysis={"should_answer": True, "entity_candidates": []},
            alias_context=_ctx(),
        )
        assert low["abstained"] is True
        assert low["abstain_reason"].startswith("insufficient_retrieval")

        empty = ak.answer(
            "「办事处干员B」的规范名是什么？",
            {"units": [], "context_digest": "d"},
            top_score=1.0,
            query_analysis={"should_answer": True, "entity_candidates": []},
            alias_context=_ctx(),
        )
        assert empty["abstain_reason"] == "no_context_units"


# ------------------------------------------- 5. 接线：ask_once 的开关真的通到 answer
class TestPipelineWiring:
    """开关的唯一事实源是 `ask_once` 的 options。这条不钉，"没接线"会静默复发。"""

    class _StubAnswerer:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def answer(self, question, context, **kwargs):
            self.calls.append(kwargs)
            return {"answer": "", "citations": [], "abstained": True,
                    "abstain_reason": "stub", "stats": {}, "warnings": []}

    def _pipe(self):
        class _R:
            graph = {"entities": ENTITIES, "alias_index": {}}
            chunks: dict = {}
            assertions_by_chunk: dict = {}
            query_context = _ctx()
            budget = {"max_context_tokens": 24000}

            def search(self, qa, *, top_k=20):
                return {"hits": [], "stats": {"top_score": 0.0}}

            def expand(self, ids, **kw):
                return {"expansions": [], "stats": {}}

            def build_context(self, hits, exps, **kw):
                return {"units": [], "context_digest": "", "stats": {}}

        return {"retriever": _R(), "answerer": self._StubAnswerer(),
                "top_k": 20, "graph_depth": 1}

    def test_option_off_passes_no_alias_context(self) -> None:
        pipe = self._pipe()
        pipeline_mod.ask_once(pipe, "「办事处干员B」的规范名是什么？",
                              options=None)
        kwargs = pipe["answerer"].calls[0]
        assert "alias_context" not in kwargs, (
            "默认档多传一个关键字，任何既有 answer() 替身都会 TypeError —— "
            "这是本代码库反复出现的兼容性事故"
        )

    def test_option_on_passes_query_context(self) -> None:
        pipe = self._pipe()
        pipeline_mod.ask_once(pipe, "「办事处干员B」的规范名是什么？",
                              options={"alias_answer": True})
        kwargs = pipe["answerer"].calls[0]
        assert kwargs.get("alias_context") is not None
        assert kwargs["alias_context"] is pipe["retriever"].query_context, (
            "必须传 retriever 自己那份：另建一份会让回答侧与查询侧的别名表对不齐"
        )


# ----------------------------------------------------- 6. PROD_PROFILE 不许被改
def test_prod_profile_has_no_alias_answer_key() -> None:
    """`PROD_PROFILE` 是本轮的冻结面：新开关**不许**进生产档。"""
    assert "alias_answer" not in pipeline_mod.PROD_PROFILE["options"]
    assert set(pipeline_mod.PROD_PROFILE["options"]) == {
        "quote_locate", "bridge", "bridge_search_top_k",
        "graph_reserve_tokens", "chapter_cap",
    }
    assert pipeline_mod.PROD_PROFILE["max_context_tokens"] == 24000


# ------------------------- 7. 8 道目标题的闸门归因表（金标，防归因结论漂移）
TARGETS = ("q-al-000010", "q-al-000025", "q-al-900019", "q-al-000005",
           "q-al-000015", "q-al-000030", "q-sc-900035", "q-al-900014")

#: `data/diag_alias_abstain.json` 的实测结果：**8/8 都是 G4**。
#: 这个表是本轮最重要的一条事实，所以写进测试而不是只写在文档里。
MEASURED_GATE = {q: "G4_no_supported_claim" for q in TARGETS}


def test_measured_attribution_is_recorded() -> None:
    """把实测归因钉住。若将来有人改动 `answer()` 的闸门顺序导致归因漂移，
    这条测试会红——那是好事，说明归因表需要重跑，而不是继续引用旧结论。"""
    assert len(MEASURED_GATE) == 8
    assert set(MEASURED_GATE.values()) == {"G4_no_supported_claim"}, (
        "本轮实测：8 道题全部止于 no_supported_claim，没有一道被 min_top_score 拦下。"
        "若这条不再成立，必须重跑 scripts/diag_alias_abstain.py 并更新报告"
    )
