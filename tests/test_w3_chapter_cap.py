"""W3 门：章节答案的截断帽 `chapter_cap`。

## 这个文件守的是什么

W3 要让 `cross_chunk` 的答案**少列无关章节**：15 道失败里 7 道是
`extra_chapters_outnumber_gold`（`W ⊆ O` 但 `|O| > 2|W|`，R 全部 1.00）——
gold 章节一个不落全找到了，只因多列了几章被 `judge.judge_answer` 判错。
判据**不动**（`judge.py` 不在改动面内），只改答案本身。

三件事各自有一种静默失效的方式，所以分别钉：

1. **默认路径必须逐位等价。** `chapter_cap=0` 时答案串、引用列表、返回 dict 的
   键、`stats["chapter_path"]` 的键，都必须与改造前一致；连
   `_chapter_answer` 的**调用形状**都不许多一个关键字。
   失效方式：顺手把 `chapter_cap=0` 也当成「取前 0 章」或「取全部但重排」，
   历史台账读数集体漂移而无人察觉。

2. **截断是「按排序键取前 N 章，但呈现仍按字典序」。**
   失效方式：让截断改变答案里章节的排列顺序 —— `judge._CHAPTER_SPLIT` 虽然
   按集合比对不受影响，但台账里逐字引用答案的读法会全线漂移，而且
   `n_chapters == len(答案切分)` 这类自检会失配。

3. **⚠️ 排序键的质量是未知的，测试必须把这个未知原样钉住。**
   计划书 §「唯一必须命名的风险：排序键质量未知」写明：这个键**只被测过
   "当阈值用"**（`nh≥2` 掉到 2/38、`nh≥3` 归零），从没被测过"当序用"，要求先出
   `data/diag_cx_rank_keys.json` 的四键对照表再定 cap（判据 C5）。
   **那份对照表本轮没有产出**（生成它要跑全量世代，本工作项纪律禁止加载产物）。
   所以下面的 `TestRankKeyQualityIsUnproven` 不是"证明键是好的"，而是
   **把"它会选错"的那一面写死在用例里**：噪声章节（3 个弱单元）在
   `Σ score` 打平时靠单元数压过 gold（1 个强单元）。这条用例红＝有人偷偷换了
   排序键去让测试好看 —— 那是计划书明确禁止的「偷换排序键」。

## 为什么用合成上下文而不是真实世代

这里断言的是**装配逻辑的代数性质**（等价性、上限、稳定性、呈现顺序），不是
分数。真实世代的分数由 `scripts/eval_answer_quality.py` 负责，那是另一类证据，
而且一次 `load_generation()` 常驻 1.0–1.5 GB。
"""
from __future__ import annotations

import pytest

from arkrag import answer as answer_mod

MIN = answer_mod.MIN_EVIDENCE_CHARS
QUESTION = "整合运动在哪些章节代号里出现过？"


# --------------------------------------------------------------------- 夹具
def _chunk(chunk_id: str, *, chapter: str, text: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": chunk_id.split("#")[0],
        "title": f"T-{chapter}",
        "chapter": chapter,
        "text": text,
    }


def _unit(chunk: dict, *, score: float = 1.0, assertions=None, spans=None,
          entities=None) -> dict:
    return {
        "chunk_id": chunk["chunk_id"],
        "doc_id": chunk["doc_id"],
        "title": chunk["title"],
        "chapter": chunk["chapter"],
        "text": chunk["text"],
        "assertions": list(assertions or []),
        "evidence_spans": list(spans or []),
        "entities": list(entities or []),
        "score": score,
        "sources": [{"channel": "bm25", "score": score}],
        "graph_path": [],
    }


def _ctx(units: list[dict]) -> dict:
    return {"units": units, "context_digest": "d" * 64,
            "stats": {"units": len(units)}}


def _qa(*names: str) -> dict:
    return {
        "original_query": QUESTION,
        "normalized_query": QUESTION,
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


def _line_of(unit: dict, needle: str) -> tuple[int, int]:
    text = unit["text"]
    start = text.index(needle)
    return start, start + len(needle)


def _ask(units: list[dict], *, cap, qa=None, unit_factory=_unit):
    """调一次 `answer()`；`cap is None` 表示**完全不传** `chapter_cap`。

    这个区分是检查 1 的全部要点：`pipeline` 在默认档下真的不会传这个关键字，
    所以"不传"与"显式传 0"是两条同时存在的路径，必须逐位等价。
    """
    kwargs = {} if cap is None else {"chapter_cap": cap}
    return answer_mod.ExtractiveAnswerer().answer(
        QUESTION, _ctx(units), top_score=1.0,
        query_analysis=qa if qa is not None else _qa("整合运动"), **kwargs
    )


def _stable_view(out: dict) -> dict:
    """去掉 `versions.built_at` 后的整份返回（其余每个键都必须逐位相同）。

    `built_at` 是 `time.strftime(...)` 的**秒级**时间戳：两次调用跨过秒边界就会
    不同，把它算进比对会让这条门随机红——**随机红的门等于没有门**，而且会让
    人去关掉它而不是查缺陷。时间戳本身另有单独断言。
    """
    view = dict(out)
    versions = dict(view.get("versions") or {})
    versions.pop("built_at", None)
    view["versions"] = versions
    return view


def _spread_units(*, n_chapters: int, units_per_chapter=None,
                  scores_per_chapter=None) -> list[dict]:
    """造 `n_chapters` 个章节，每章 `units_per_chapter[i]` 个单元。

    每章的所有单元正文都含焦点实体名「整合运动」，所以都会通过
    `_chapter_answer` 的入选判据（`nh >= 1`）——这正是被截断的那个集合。
    """
    counts = list(units_per_chapter or [1] * n_chapters)
    scores = list(scores_per_chapter or [[1.0] * k for k in counts])
    out: list[dict] = []
    for i, (count, chapter_scores) in enumerate(zip(counts, scores)):
        chapter = f"main_ep{i:02d}"
        for j in range(count):
            text = f"整合运动在第{i}章的第{j}块里被提到了。"
            out.append(_unit(
                _chunk(f"doc{i:02d}/c#%03d" % j, chapter=chapter, text=text),
                score=chapter_scores[j],
            ))
    return out


def _chapters_of(out: dict) -> list[str]:
    return [c for c in str(out.get("answer") or "").split("、") if c]


# ------------------------------------------------------------------ 检查 1
class TestDefaultPathIsBitIdentical:
    """检查 1：`chapter_cap=0`（含不传）与改造前逐位等价。"""

    def test_explicit_zero_equals_omitted(self):
        """显式传 `chapter_cap=0` 必须与完全不传**逐位相同**。

        不是形式主义：`pipeline` 在默认档下不传这个关键字，于是两条路径真的会
        同时存在。若它们不等价，「默认关」就是假的。
        """
        units = _spread_units(n_chapters=5, units_per_chapter=[1, 2, 1, 3, 1])

        omitted = _ask(units, cap=None)
        explicit = _ask(units, cap=0)

        assert set(omitted) == set(explicit), "返回 dict 的键集合变了"
        assert _stable_view(omitted) == _stable_view(explicit)
        assert omitted["answer"] == explicit["answer"]
        assert omitted["citations"] == explicit["citations"]
        assert omitted["stats"] == explicit["stats"]
        assert omitted["confidence"] == explicit["confidence"]
        assert omitted["abstained"] is False

    def test_no_cap_keys_when_cap_is_zero(self):
        """cap=0 时 `chapter_path` 里不许出现任何 W3 新增的键。

        历史台账里有对 `stats["chapter_path"]` 做等值比较的读法，多一个键就会让
        它们全部失配（与 W1 的 `max_direct_units` / `capped_units` 同一条规矩）。
        """
        units = _spread_units(n_chapters=4)
        for cap in (None, 0):
            out = _ask(units, cap=cap)
            stats = out["stats"]["chapter_path"]
            assert stats["applied"] is True
            assert "chapter_cap" not in stats
            assert "n_chapters_found" not in stats
            assert "chapters_truncated" not in stats
            assert set(stats) == {"applied", "n_chapters", "chapter_source", "citations"}

    def test_internal_hit_dict_has_no_cap_keys_when_cap_is_zero(self):
        """`_chapter_answer` 自己的返回 dict：cap=0 时也不许多键。

        **这条是被反向实测逼出来的**：把 `_chapter_answer` 里
        `if cap > 0: hit["chapter_cap"] = …` 改成 `if True:` 之后，
        本文件其余 19 条**全部仍然是绿的** —— 因为 `answer()` 只在 `cap > 0` 时
        才去读 `n_chapters_found`，多出来的键永远浮不到公开返回值上。
        也就是说「不许加键」这条承诺此前只有一半有门。
        """
        units = _spread_units(n_chapters=3)
        qa = _qa("整合运动")
        for kwargs in ({}, {"chapter_cap": 0}):
            hit = answer_mod._chapter_answer(QUESTION, units, qa, **kwargs)
            assert hit is not None, "前置：这个上下文应当触发章节通路"
            assert set(hit) == {
                "answer", "chapters", "citations", "n_chapters", "chapter_source",
            }, f"cap=0 时返回 dict 多了键：{sorted(set(hit))}"

    def test_zero_does_not_pass_the_keyword_down(self, monkeypatch):
        """默认档的**调用形状**不许变：`chapter_cap == 0` 时不得拼这个关键字。

        这条比上一条更硬：等价性可以靠"多传一个 0 但实现忽略"蒙过去，而这条
        直接盯调用点。给 `_chapter_answer` 换成任何既有替身（测试桩、旧版实现）
        都不该因为多一个关键字而 TypeError。
        """
        seen: list[dict] = []
        real = answer_mod._chapter_answer

        def spy(question, units, query_analysis=None, **kwargs):
            seen.append(dict(kwargs))
            return real(question, units, query_analysis, **kwargs)

        monkeypatch.setattr(answer_mod, "_chapter_answer", spy)
        units = _spread_units(n_chapters=6)

        _ask(units, cap=None)
        _ask(units, cap=0)
        _ask(units, cap=2)

        assert seen[:2] == [{}, {}], f"默认档下多传了关键字：{seen[:2]}"
        assert seen[2] == {"chapter_cap": 2}


# ------------------------------------------------------------------ 检查 2
class TestCapTruncates:
    """检查 2：`chapter_cap=N` 的输出章节数 ≤ N，且呈现仍是字典序。"""

    @pytest.mark.parametrize("cap", [1, 2, 3, 4])
    def test_output_chapter_count_is_at_most_n(self, cap: int):
        units = _spread_units(
            n_chapters=6, units_per_chapter=[3, 1, 2, 1, 2, 4],
            scores_per_chapter=[[0.9, 0.1, 0.1], [0.8], [0.5, 0.2],
                                [0.4], [0.3, 0.1], [0.2, 0.1, 0.1, 0.1]],
        )
        full = _ask(units, cap=0)
        capped = _ask(units, cap=cap)

        assert len(_chapters_of(full)) == 6, "前置：6 个章节都该进未截断的答案"
        got = _chapters_of(capped)
        assert 0 < len(got) <= cap, f"cap={cap} 却输出了 {got}"
        assert capped["stats"]["chapter_path"]["chapters_truncated"] is True
        assert capped["stats"]["chapter_path"]["n_chapters_found"] == 6
        # 引用是「每块一条」（`_chapter_answer` 的既有粒度），不是「每章一条」；
        # 但**每个保留下来的章节**都必须至少有一条引用 —— 这是 `judge.
        # citations_support(chapters=…)` 的 `chapters_without_citation` 判据。
        by_id = {u["chunk_id"]: u for u in units}
        covered = {by_id[c["chunk_id"]]["chapter"] for c in capped["citations"]}
        assert covered == set(got), (
            f"引用覆盖的章节 {sorted(covered)} 与答案里的章节 {got} 不一致")
        assert len(capped["citations"]) < len(full["citations"]), (
            "截断后引用条数没有减少 —— 被砍掉的章节还在发引用")

    def test_kept_chapters_keep_dictionary_order(self):
        """截断只决定"留哪几章"，不许改"怎么排"。

        呈现顺序变了会让 `n_chapters == len(答案.split("、"))` 这类自检失配，
        也会让台账里逐字引用答案的读法漂移。

        **夹具必须让「排序键的顺序」≠「字典序」**，否则这条用例测不到任何东西：
        第一版用的分数让 top-3 恰好是 ep01/ep03/ep04、字典序也是 ep01/ep03/ep04，
        两边一模一样，把实现改成 `chapters = list(ranked[:cap])` 仍然是
        `19 passed` —— 一条永远绿的用例等于没有用例。现在让最高分落在
        字典序**最后**的 ep04 上，两个顺序才真的分得开：排序键给出
        ep04/ep01/ep03，答案必须仍呈现为 ep01/ep03/ep04。
        """
        units = _spread_units(
            n_chapters=5,
            units_per_chapter=[1, 1, 1, 1, 1],
            scores_per_chapter=[[0.1], [0.9], [0.2], [0.8], [0.95]],
        )
        out = _ask(units, cap=3)
        got = _chapters_of(out)
        assert len(got) == 3
        # 排序键的顺序（对比用）：ep04(0.95) > ep01(0.9) > ep03(0.8)
        ranked = sorted(
            [f"main_ep{i:02d}" for i in range(5)],
            key=lambda c: answer_mod._chapter_rank_key(
                c, [u for u in units if u["chapter"] == c]),
        )
        assert ranked[:3] == ["main_ep04", "main_ep01", "main_ep03"], (
            f"前置：排序键的 top-3 应当与字典序不同，实际 {ranked[:3]}")
        assert got == sorted(got), f"保留的章节没有按字典序呈现：{got}"
        assert got == ["main_ep01", "main_ep03", "main_ep04"], (
            f"应留下三个最高分的章节、并按字典序呈现，实际 {got}")

    def test_cap_at_or_above_count_is_identical_to_no_cap(self):
        """cap ≥ 章节点数时，结果必须与不截断逐位相同（含 stats 的 W3 键）。"""
        units = _spread_units(n_chapters=3)
        full = _ask(units, cap=0)
        loose = _ask(units, cap=3)
        assert loose["answer"] == full["answer"]
        assert loose["citations"] == full["citations"]
        assert loose["stats"]["chapter_path"]["chapters_truncated"] is False
        assert loose["stats"]["chapter_path"]["n_chapters_found"] == 3

    def test_every_citation_still_self_consistent(self):
        """截断后每条引用仍满足 `text[start:end] == evidence` 且够长。

        这是 `verify_citations` 的硬判据；引用不自洽会被静默剔除，答案被清空
        改判拒答（本项目记录在案的 `q-cx-900024` 事故形态）。
        """
        units = _spread_units(
            n_chapters=4, units_per_chapter=[2, 1, 3, 1],
            scores_per_chapter=[[0.5, 0.1], [0.9], [0.2, 0.1, 0.1], [0.4]],
        )
        out = _ask(units, cap=2)
        by_id = {u["chunk_id"]: u for u in units}
        assert out["citations"], "截断后一条引用都没有 —— 会把答案清空"
        for cite in out["citations"]:
            text = by_id[cite["chunk_id"]]["text"]
            assert text[cite["start"]:cite["end"]] == cite["evidence"]
            assert len(cite["evidence"]) >= MIN


# ------------------------------------------------------------------ 检查 3
class TestRankKeyQualityIsUnproven:
    """⚠️ 检查 3：排序键**不是**已被证明最优的，这里把它的失误形态钉死。

    排序键 = `(-Σ unit.score, -len(units), chapter)`（见
    `answer._chapter_rank_key` 的 docstring）。本节只断言"键确实是这个"以及
    "它会在什么情况下选错"，**不**断言它选得对。

    为什么用 0.25 / 0.75 这类可二进制精确表示的分数：`0.1*3 == 0.30000000000000004`
    这类浮点尾巴会让"打平"变成"差一点"，用例就测不到并列分支——
    那种用例看着绿，实际测的是另一条路径。
    """

    def test_rank_key_orders_by_total_score_first(self):
        """主键是 `Σ score`：分数高的章节排前面，与单元个数无关。"""
        gold = [{"score": 0.75}]
        noise = [{"score": 0.25}, {"score": 0.25}, {"score": 0.25}]
        gold_key = answer_mod._chapter_rank_key("gold_ch", gold)
        noise_key = answer_mod._chapter_rank_key("noise_ch", noise)
        # 0.75 == 0.75（精确），所以这一对**打平在主键上**
        assert gold_key[0] == noise_key[0]
        # 打平后由单元数决定：3 > 1 ⇒ 噪声章节排在前面
        assert noise_key < gold_key, (
            "Σ score 打平时应当由单元数打破并列（噪声 3 个单元胜 gold 1 个）")

    def test_strong_gold_wins_when_its_total_score_is_higher(self):
        """gold 1 个强单元 (0.75) vs 噪声 3 个弱单元 (0.2×3=0.6) ⇒ gold 留下。"""
        gold = _chunk("g#001", chapter="main_ep00", text="整合运动在 gold 章里。")
        noise = [
            _chunk("n#001", chapter="act99mini", text="整合运动在噪声里一句。"),
            _chunk("n#002", chapter="act99mini", text="整合运动在噪声里二句。"),
            _chunk("n#003", chapter="act99mini", text="整合运动在噪声里三句。"),
        ]
        units = [_unit(gold, score=0.75)] + [_unit(c, score=0.2) for c in noise]
        out = _ask(units, cap=1)
        assert _chapters_of(out) == ["main_ep00"], (
            f"Σ score 0.75 > 0.6，gold 章节应当留下，实际 {_chapters_of(out)}")

    def test_count_tiebreak_drops_the_gold_chapter_this_is_the_known_risk(self):
        """⚠️ **已知风险，原样钉死，不许为了让测试好看而改排序键。**

        构造：gold 章节只有 1 个单元（score 0.75），噪声章节有 3 个单元
        （各 0.25，Σ 也是 0.75，二进制精确相等）。`cap=1` 时主键打平，副键
        「单元数」把噪声章节排在前面 ⇒ **gold 被截掉**。

        这正是计划书 §「排序键质量未知」点名的形态：**多而弱压过少而强**。
        它没有被修掉，因为修它（比如改用平均分、或把单元数踢出键）等于在没有
        `data/diag_cx_rank_keys.json` 对照表的情况下换一个未经检验的键 ——
        计划书把「读完结果后新增第五个键追分」和「偷换排序键」都列为禁止行为。

        因此：这条用例转红＝有人换了排序键。换键必须有对照表，那时**连同本用例
        的期望值一起改**，并在报告里写明依据，而不是删掉它。
        """
        gold = _chunk("g#001", chapter="main_ep00", text="整合运动在 gold 章里。")
        noise = [
            _chunk("n#001", chapter="act99mini", text="整合运动在噪声里一句。"),
            _chunk("n#002", chapter="act99mini", text="整合运动在噪声里二句。"),
            _chunk("n#003", chapter="act99mini", text="整合运动在噪声里三句。"),
        ]
        units = [_unit(gold, score=0.75)] + [_unit(c, score=0.25) for c in noise]

        full = _ask(units, cap=0)
        assert _chapters_of(full) == ["act99mini", "main_ep00"], (
            "前置：不截断时两个章节都在答案里")

        capped = _ask(units, cap=1)
        assert _chapters_of(capped) == ["act99mini"], (
            f"排序键的行为变了：Σ score 打平时留下的是 {_chapters_of(capped)}，"
            f"而本门登记的是「按单元数打破并列 ⇒ 噪声章节胜」。"
            f"若这是有意换键，请先交 data/diag_cx_rank_keys.json 对照表，"
            f"再同步更新这条用例的期望值与注释。")

    def test_graph_only_units_rank_last(self):
        """⚠️ 另一条已知风险：纯图扩展单元的 `score` 恒 0.0。

        `build_context` 给纯图单元写的是 `{"channel": "graph", "score": 0.0}`，
        于是 `Σ score == 0`，哪怕正文真的提到实体也排在所有直接命中之后。
        gold 若只能经图扩展到达，会被先截掉。这条同样只记录、不修。
        """
        direct = _chunk("d#001", chapter="main_ep01", text="整合运动在直击块里。")
        graph_only = _chunk("x#001", chapter="main_ep02", text="整合运动在图块里。")
        g_unit = _unit(graph_only, score=0.0)
        g_unit["sources"] = [{"channel": "graph", "score": 0.0}]
        units = [_unit(direct, score=0.5), g_unit]

        out = _ask(units, cap=1)
        assert _chapters_of(out) == ["main_ep01"]
        assert answer_mod._chapter_rank_key("main_ep02", [g_unit])[0] == 0.0, (
            "纯图扩展单元的总分应当是 0.0 —— 它在这个排序键里没有分数证据")


class TestRankKeyIsDeterministic:
    """排序必须稳定：同分同数时按章节名字典序兜底，两次调用结果相同。"""

    def test_equal_keys_break_ties_by_chapter_name(self):
        a = [{"score": 0.5}]
        b = [{"score": 0.5}]
        assert answer_mod._chapter_rank_key("aaa", a) < answer_mod._chapter_rank_key("bbb", b), (
            "完全并列时必须由章节名字典序兜底，否则答案取决于 dict 插入顺序")

    def test_full_tie_is_order_independent(self):
        """三个章节分数与单元数全同 ⇒ 留下的顺序与上下文装配顺序无关。"""
        chapters = ["main_ep02", "main_ep00", "main_ep01"]
        forward = [
            _unit(_chunk(f"f#{i}", chapter=c, text=f"整合运动在 {c} 里。"), score=0.5)
            for i, c in enumerate(chapters)
        ]
        backward = list(reversed(forward))

        out_f = _ask(forward, cap=2)
        out_b = _ask(backward, cap=2)
        assert _chapters_of(out_f) == ["main_ep00", "main_ep01"]
        assert _chapters_of(out_f) == _chapters_of(out_b), (
            "同样的章节集合、不同的装配顺序给出了不同答案 ⇒ 排序不稳定")

    def test_repeated_calls_are_identical(self):
        units = _spread_units(
            n_chapters=5, units_per_chapter=[2, 1, 1, 3, 1],
            scores_per_chapter=[[0.5, 0.5], [1.0], [1.0], [0.5, 0.5, 0.5], [1.0]],
        )
        first = _ask(units, cap=2)
        second = _ask(units, cap=2)
        assert _stable_view(first) == _stable_view(second)


# ------------------------------------------------------------------ 检查 4
class TestNonChapterQuestionIsUnaffected:
    """检查 4：不触发章节通路的题，`chapter_cap` 一个字节都不许改。"""

    TEXT = "凯尔希：罗德岛不会放弃任何一个人。"

    def _claim_units(self) -> list[dict]:
        """一条能走 claim 通路的普通题上下文（与章节通路无关）。"""
        chunk = _chunk("a#001", chapter="main_ep08", text=self.TEXT)
        needle = "凯尔希：罗德岛不会放弃任何一个人"
        start, end = _line_of(_unit(chunk), needle)
        brief = {
            "assertion_id": "as-001",
            "subject": "凯尔希",
            "predicate": "隶属",
            "object": "罗德岛",
            "evidence": needle,
            "confidence": 0.9,
            "chunk_id": chunk["chunk_id"],
        }
        span = {
            "assertion_id": "as-001",
            "chunk_id": chunk["chunk_id"],
            "evidence": needle,
            "start": start,
            "end": end,
        }
        return [_unit(chunk, score=0.9, assertions=[brief], spans=[span],
                      entities=["凯尔希", "罗德岛"])]

    def test_normal_question_does_not_reach_the_chapter_path(self):
        units = self._claim_units()
        qa = _qa("凯尔希", "罗德岛")
        out = answer_mod.ExtractiveAnswerer().answer(
            "凯尔希和罗德岛是什么关系？", _ctx(units), top_score=1.0,
            query_analysis=qa, chapter_cap=2)
        assert out["stats"]["chapter_path"]["applied"] is False
        assert "chapter_cap" not in out["stats"]["chapter_path"], (
            "没走章节通路却记了 W3 的账 —— 台账会把它算成一次截断")

    def test_cap_does_not_change_a_normal_answer(self):
        units = self._claim_units()
        qa = _qa("凯尔希", "罗德岛")

        def ask(cap):
            return answer_mod.ExtractiveAnswerer().answer(
                "凯尔希和罗德岛是什么关系？", _ctx(units), top_score=1.0,
                query_analysis=qa,
                **({} if cap is None else {"chapter_cap": cap}))

        base = ask(None)
        for cap in (0, 1, 3, 8):
            got = ask(cap)
            assert _stable_view(got) == _stable_view(base), f"cap={cap} 改动了普通题"
            assert got["answer"] == base["answer"]
            assert got["citations"] == base["citations"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
