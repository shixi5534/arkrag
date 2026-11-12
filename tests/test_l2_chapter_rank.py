"""L2 门：`cross_chunk` 章节集合的**排序键开关** `chapter_rank`。

## 这个文件守的是什么

生产档 `chapter_cap=4` 下 64 道 `cross_chunk` 有 20 道判 `chapter_set_missing`。
`scripts/diag_cx_chapter_gap.py` 把那 20 道的每一个缺失 gold 章节逐章归因：

    A_chunk_absent             17 处   该章在本次上下文里一个单元都没有
    C_name_present_but_dropped 11 处   已过 `nh>=1` 进候选集，被排序键挤出前 4
    B_no_name_in_chapter        0 处   入选判据一处都没有误杀

所以这一类的**可装配部分**只有一个变量：截断时按什么排序。本文件钉四件事：

1. **默认档逐位等价。** `chapter_rank="score"`（含完全不传）时，答案串、引用列表、
   返回 dict 的键、`stats["chapter_path"]` 的键集合都必须与改造前一致；连
   `_chapter_answer` 的**调用形状**都不许多一个关键字。
   失效方式：顺手把 `"score"` 也拼进 `kwargs` —— 既有的 `_chapter_answer` 替身
   （测试桩、旧版实现）会因为多一个关键字 TypeError，历史读数集体漂移。

2. **换键只改"留哪几章"，不改"怎么排"。** 呈现顺序必须仍是字典序。
   失效方式：让 `focus_assertions` 顺序直接当答案顺序 —— `judge._CHAPTER_SPLIT`
   按集合比对不受影响（所以分数不会变），但台账里逐字引用答案的读法会全线漂移，
   `n_chapters == len(答案.split("、"))` 这类自检会失配。

3. **新键的机制必须可验证，且"它会选错"要写死。** `focus_assertions` 数的是
   「该章候选单元的断言端点命中焦点名的条数」。两条用例分别钉正反两面：
   - 正面：「1 个单元 3 条关于焦点的断言」必须压过「3 个单元各 1 条无关断言」——
     这正是诊断里 11 处 C 桶的形态（gold 章 1 个真命中 vs 噪声章多个弱命中）。
   - 反面：纯文本命中焦点名但**断言一条都没有**的章节会被排到最后
     （`Σscore` 与断言数都看不出它）。这是本键的已知失误形态，
     **不修、不美化**；把它改成"更好的键"＝换了实测依据，必须重新出 `data/diag_*.json`。

4. **`pipeline.ask_once` 的接线与门控。** 默认档不拼关键字；非默认档才拼；
   非章节题上即使传了也不该改变任何东西（`_chapter_answer` 本来就是 no-op）。

## 为什么用合成上下文而不是真实世代

这里断言的是**装配逻辑的代数性质**（等价性、稳定性、呈现顺序、机制方向），
不是分数。真实世代的分数由 `scripts/diag_l2_chapter_rank.py` 负责，那是另一类
证据，而且一次 `load_generation()` 常驻 1.0–1.5 GB。
"""
from __future__ import annotations

import pytest

from arkrag import answer as answer_mod

MIN = answer_mod.MIN_EVIDENCE_CHARS
QUESTION = "整合运动在哪些章节代号里出现过？"
FOCUS = "整合运动"


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


def _assertion(*, subject: str, obj: str, evidence: str, chunk_id: str,
               aid: str = "") -> dict:
    return {"assertion_id": aid, "subject": subject, "predicate": "参与",
            "object": obj, "evidence": evidence, "confidence": 1.0,
            "chunk_id": chunk_id}


def _ask(units: list[dict], *, cap, rank=None, qa=None):
    """调一次 `answer()`；`rank is None` 表示**完全不传** `chapter_rank`。"""
    kwargs: dict = {}
    if cap is not None:
        kwargs["chapter_cap"] = cap
    if rank is not None:
        kwargs["chapter_rank"] = rank
    return answer_mod.ExtractiveAnswerer().answer(
        QUESTION, _ctx(units), top_score=1.0,
        query_analysis=qa if qa is not None else _qa(FOCUS), **kwargs
    )


def _stable_view(out: dict) -> dict:
    """去掉 `versions.built_at` 后的整份返回（其余每个键都必须逐位相同）。

    `built_at` 是秒级时间戳：两次调用跨过秒边界就会不同，把它算进比对会让这条
    门随机红——**随机红的门等于没有门**。
    """
    view = dict(out)
    versions = dict(view.get("versions") or {})
    versions.pop("built_at", None)
    view["versions"] = versions
    return view


def _chapters_of(out: dict) -> list[str]:
    raw = str(out.get("answer") or "")
    return [x for x in raw.replace("，", "、").replace(",", "、").split("、") if x]


def _plain_units(chapters: list[str], *, text_has_focus: bool = True) -> list[dict]:
    """每章一个单元，正文都含焦点名（所以全过 `nh>=1`），无断言、同分。"""
    out = []
    for ch in chapters:
        body = f"{FOCUS}在这里出现了一句话。"
        if not text_has_focus:
            body = "这里只有一句与焦点无关的话。"
        out.append(_unit(_chunk(f"{ch}#001", chapter=ch, text=body), score=1.0))
        out[-1]["chapter"] = ch
        out[-1]["text"] = body
    return out


# ------------------------------------------------------------------ 检查 1
class TestDefaultPathIsByteIdentical:
    """检查 1：`chapter_rank="score"`（含不传）与改造前逐位等价。"""

    def test_default_and_explicit_score_are_identical(self):
        """不传 == 显式传 `"score"` == 改造前的键。"""
        units = _plain_units(["main_ep01", "main_ep02", "main_ep03",
                              "main_ep04", "main_ep05", "main_ep06"])
        reference = _ask(units, cap=4, rank=None)
        for rank in (None, "score"):
            got = _ask(units, cap=4, rank=rank)
            assert _stable_view(got) == _stable_view(reference), (
                f"chapter_rank={rank!r} 与不传的返回不一致")

    def test_default_path_stats_has_no_rank_key(self):
        """`stats["chapter_path"]` 的键集合：默认档不许出现 `chapter_rank`。"""
        units = _plain_units(["a01", "a02", "a03", "a04", "a05"])
        out = _ask(units, cap=4)
        stats = out["stats"]["chapter_path"]
        assert "chapter_rank" not in stats, f"默认档多写了键：{sorted(stats)}"
        assert set(stats) == {"applied", "n_chapters", "chapter_source", "citations",
                             "chapter_cap", "n_chapters_found", "chapters_truncated"}

    def test_internal_hit_dict_has_no_rank_key_by_default(self):
        """`_chapter_answer` 自己的返回 dict：默认档不许多键。"""
        units = _plain_units(["a01", "a02", "a03", "a04", "a05"])
        hit = answer_mod._chapter_answer(QUESTION, units, _qa(FOCUS), chapter_cap=4)
        assert hit is not None, "前置：该上下文应触发章节通路"
        assert "chapter_rank" not in hit, f"默认档多写了键：{sorted(hit)}"
        assert set(hit) == {"answer", "chapters", "citations", "n_chapters",
                            "chapter_source", "chapter_cap", "n_chapters_found",
                            "chapters_truncated"}

    def test_call_shape_does_not_gain_a_keyword_by_default(self, monkeypatch):
        """默认档的**调用形状**不许变。这条比等价性更硬。

        等价性可以靠"多传一个 `"score"` 但实现忽略"蒙过去；这条直接盯调用点。
        给 `_chapter_answer` 换成任何既有替身都不该因为多一个关键字 TypeError。
        """
        seen: list[dict] = []
        real = answer_mod._chapter_answer

        def spy(question, units, query_analysis=None, **kwargs):
            seen.append(dict(kwargs))
            return real(question, units, query_analysis, **kwargs)

        monkeypatch.setattr(answer_mod, "_chapter_answer", spy)
        units = _plain_units(["a01", "a02", "a03", "a04", "a05", "a06"])

        _ask(units, cap=4)                    # 默认档
        _ask(units, cap=4, rank="score")      # 显式默认值
        _ask(units, cap=4, rank="focus_assertions")

        assert seen[:2] == [{"chapter_cap": 4}, {"chapter_cap": 4}], (
            f"默认档下多传了关键字：{seen[:2]}")
        assert seen[2] == {"chapter_cap": 4, "chapter_rank": "focus_assertions"}

    def test_unknown_rank_falls_back_without_changing_answer(self):
        """未登记的键退回默认，**不悄悄换答案**。"""
        units = _plain_units(["a01", "a02", "a03", "a04", "a05", "a06"])
        base = _ask(units, cap=4)
        weird = _ask(units, cap=4, rank="no_such_key")
        assert weird["answer"] == base["answer"]
        assert "chapter_rank" not in weird["stats"]["chapter_path"]

    def test_rank_alone_changes_nothing(self):
        """cap=0 时 `chapter_rank` 不管传什么都不该改变一个字节。"""
        units = _plain_units(["a01", "a02", "a03"])
        base = _ask(units, cap=0)
        for rank in ("score", "focus_assertions"):
            got = _ask(units, cap=0, rank=rank)
            assert _stable_view(got) == _stable_view(base), (
                f"cap=0 但 chapter_rank={rank!r} 改变了返回")


# ------------------------------------------------------------------ 检查 2
class TestPresentationStaysLexicographic:
    """检查 2：换键只决定"留哪几章"，呈现仍是字典序。"""

    def test_focus_key_does_not_reorder_presentation(self):
        """夹具让两种顺序真的分得开，否则这条用例测不到任何东西。

        五章同分、每章一个单元，只有 `a05` 与 `a01` 的单元带焦点断言。
        `focus_assertions` 键给出 [a05, a01, a02, …]（断言数 2 > 1 > 0，并列
        按章节名），而答案必须仍呈现为字典序 `a01、a05`。
        第一版若把断言加在 `a01`/`a02` 上，排序顺序与字典序恰好相同，
        把实现改成 `chapters = list(ranked[:cap])` 仍然全绿 —— 一条永远绿的
        用例等于没有用例。
        """
        units: list[dict] = []
        for ch in ("a01", "a02", "a03", "a04", "a05"):
            n_focus = {"a01": 1, "a05": 2}.get(ch, 0)
            asserts = [
                _assertion(subject=FOCUS, obj=f"某对象{i}", evidence=f"{FOCUS}说了第{i}句。",
                           chunk_id=f"{ch}#001", aid=f"{ch}-{i}")
                for i in range(n_focus)
            ]
            u = _unit(_chunk(f"{ch}#001", chapter=ch,
                             text=f"{FOCUS}在这里出现了一句话。"),
                      score=1.0, assertions=asserts)
            units.append(u)

        out = _ask(units, cap=2, rank="focus_assertions")
        got = _chapters_of(out)
        assert got == sorted(got), f"保留的章节没有按字典序呈现：{got}"
        assert got == ["a01", "a05"], (
            f"应留下断言最多的两章并按字典序呈现，实际 {got}")

    def test_rank_order_differs_from_lexicographic_in_the_fixture(self):
        """前置自检：上一条的夹具确实让两种顺序不同（防止用例退化成空断言）。"""
        b = {
            "a01": [_unit(_chunk("a01#001", chapter="a01", text="x"), score=1.0,
                          assertions=[_assertion(subject=FOCUS, obj="o",
                                                 evidence="x", chunk_id="a01#001")])],
            "a05": [_unit(_chunk("a05#001", chapter="a05", text="x"), score=1.0,
                          assertions=[_assertion(subject=FOCUS, obj="o1",
                                                 evidence="x", chunk_id="a05#001"),
                                      _assertion(subject=FOCUS, obj="o2",
                                                 evidence="x", chunk_id="a05#001")])],
        }
        for ch in ("a02", "a03", "a04"):
            b[ch] = [_unit(_chunk(f"{ch}#001", chapter=ch, text="x"), score=1.0)]
        ranked = sorted(b, key=lambda c: answer_mod._chapter_rank_key_focus(
            c, b[c], {FOCUS}))
        assert ranked[:2] == ["a05", "a01"], (
            f"前置：focus_assertions 的 top-2 应当与字典序不同，实际 {ranked[:2]}")


# ------------------------------------------------------------------ 检查 3
class TestFocusAssertionKeyMechanism:
    """检查 3：新键的方向正确，且它的失误形态被钉死。"""

    def test_one_unit_with_many_focus_assertions_beats_many_units(self):
        """诊断里 C 桶 11 处的形态：gold 章 1 个真命中 vs 噪声章多个弱命中。

        噪声章 3 个单元各 1 条**无关**断言、同分；gold 章 1 个单元 3 条**关于
        焦点**的断言。`Σscore` 会把 gold 排到噪声章后面（个数 × 均分），
        这正是当前键在实测里丢掉 11 处 gold 章节的机制。
        """
        noise = [
            _unit(_chunk(f"noise#00{i}", chapter="noise", text=f"{FOCUS}在噪声章出现。"),
                  score=1.0,
                  assertions=[_assertion(subject="甲", obj="乙", evidence="x",
                                         chunk_id=f"noise#00{i}")])
            for i in range(3)
        ]
        gold = [_unit(_chunk("gold#001", chapter="gold_ch", text=f"{FOCUS}在 gold 章出现。"),
                      score=1.0,
                      assertions=[_assertion(subject=FOCUS, obj=f"o{i}", evidence="x",
                                             chunk_id="gold#001")
                                  for i in range(3)])]
        units = noise + gold

        by_chapter = {"noise": noise, "gold_ch": gold}
        old = sorted(by_chapter, key=lambda c: answer_mod._chapter_rank_key(
            c, by_chapter[c]))
        new = sorted(by_chapter, key=lambda c: answer_mod._chapter_rank_key_focus(
            c, by_chapter[c], {FOCUS}))
        assert old[0] == "noise", f"前置：当前键应当把噪声章排第一，实际 {old}"
        assert new[0] == "gold_ch", f"focus_assertions 没把 gold 章排第一：{new}"

        out = _ask(units, cap=1, rank="focus_assertions")
        assert _chapters_of(out) == ["gold_ch"], (
            "cap=1 时应当只留 gold_ch，实际 " + str(_chapters_of(out)))

    def test_one_unit_with_focus_assertion_is_not_starved_by_others(self):
        """并列时的兜底：断言数相同 ⇒ 退回 `-Σscore`，再退回章节名（全序）。"""
        a = [_unit(_chunk("a#001", chapter="a", text=f"{FOCUS}。"), score=0.5,
                  assertions=[_assertion(subject=FOCUS, obj="o", evidence="x",
                                         chunk_id="a#001")])]
        b = [_unit(_chunk("b#001", chapter="b", text=f"{FOCUS}。"), score=0.9,
                  assertions=[_assertion(subject=FOCUS, obj="o", evidence="x",
                                         chunk_id="b#001")])]
        by = {"a": a, "b": b}
        ranked = sorted(by, key=lambda c: answer_mod._chapter_rank_key_focus(
            c, by[c], {FOCUS}))
        assert ranked == ["b", "a"], f"断言数打平时应按 Σscore 降序，实际 {ranked}"

    def test_key_is_a_total_order(self):
        """同分同断言数的章节必须由章节名定序 —— 不许依赖 dict 插入顺序。"""
        units: list[dict] = []
        for ch in ("c03", "c01", "c02"):
            units.append(_unit(_chunk(f"{ch}#001", chapter=ch, text=f"{FOCUS}。"),
                               score=1.0))
        b: dict[str, list[dict]] = {}
        for u in units:
            b.setdefault(u["chapter"], []).append(u)
        forward = sorted(b, key=lambda c: answer_mod._chapter_rank_key_focus(
            c, b[c], {FOCUS}))
        # 反转插入顺序，结果必须一样
        b2 = {k: b[k] for k in reversed(list(b))}
        backward = sorted(b2, key=lambda c: answer_mod._chapter_rank_key_focus(
            c, b2[c], {FOCUS}))
        assert forward == backward == ["c01", "c02", "c03"]

    def test_known_failure_mode_text_hit_without_assertions_ranks_last(self):
        """**本键的已知失误形态**（不修、不美化，写死在这里）。

        某章正文确实提到焦点实体（所以 `nh>=1` 通过、它该被列出），但该块的
        断言一条都没命中焦点（图里这条边讲的是别人）。`focus_assertions` 对它
        记 0 条，于是它排到所有"有断言"的章后面 —— 若 cap 刚好卡在中间，这一章
        会被挤掉。**这就是换键的代价**：诊断阶段 B 里 `evidence` 键 +4/−1 的
        那一笔 `-1` 就是这一形态。

        改成"文本命中优先"能让这条用例红 —— 那是换掉实测依据，必须重新出
        `data/diag_l2_chapter_rank.json`，不许为了让测试好看而改。
        """
        text_only = [_unit(_chunk("t#001", chapter="text_only",
                                  text=f"{FOCUS}确实在这一章出现过。"), score=1.0)]
        with_assert = [_unit(_chunk("w#001", chapter="with_assert",
                                    text=f"{FOCUS}在这一章出现过。"), score=1.0,
                             assertions=[_assertion(subject=FOCUS, obj="o",
                                                    evidence="x", chunk_id="w#001")])]
        b = {"text_only": text_only, "with_assert": with_assert}
        ranked = sorted(b, key=lambda c: answer_mod._chapter_rank_key_focus(
            c, b[c], {FOCUS}))
        assert ranked[0] == "with_assert", (
            "已知风险被改掉了？focus_assertions 应当把「无焦点断言」的章排后")
        assert ranked[-1] == "text_only"

    def test_no_focus_name_in_names_set_does_not_crash(self):
        """`names` 为空集时不许抛异常（退化输入也必须给出全序）。"""
        b = {"a": [_unit(_chunk("a#001", chapter="a", text="x"), score=1.0)],
             "b": [_unit(_chunk("b#001", chapter="b", text="y"), score=2.0)]}
        ranked = sorted(b, key=lambda c: answer_mod._chapter_rank_key_focus(
            c, b[c], frozenset()))
        assert ranked == ["b", "a"], f"全部 0 条断言时应按 Σscore 降序，实际 {ranked}"

    def test_focus_name_uses_subject_and_object(self):
        """端点命中两个字段都算 —— 只认 subject 会把一半真命中漏掉。"""
        sub_case = [_unit(_chunk("s#001", chapter="s", text="x"), score=1.0,
                          assertions=[_assertion(subject=FOCUS, obj="o",
                                                 evidence="x", chunk_id="s#001")])]
        obj_case = [_unit(_chunk("o#001", chapter="o", text="x"), score=1.0,
                          assertions=[_assertion(subject="s", obj=FOCUS,
                                                 evidence="x", chunk_id="o#001")])]
        assert answer_mod._chapter_rank_key_focus("s", sub_case, {FOCUS})[0] == -1
        assert answer_mod._chapter_rank_key_focus("o", obj_case, {FOCUS})[0] == -1

    def test_assertion_counted_once_even_if_both_endpoints_match(self):
        """同一条断言的 subject 与 object 都是焦点名时只算 1 条。

        计 2 会让"自指断言"凭空获得双倍权重，那不是"该章说了几句关于它的话"。
        """
        u = [_unit(_chunk("x#001", chapter="x", text="x"), score=1.0,
                   assertions=[_assertion(subject=FOCUS, obj=FOCUS, evidence="x",
                                          chunk_id="x#001")])]
        assert answer_mod._chapter_rank_key_focus("x", u, {FOCUS})[0] == -1


# ------------------------------------------------------------------ 检查 4
class TestConstantsAreRegistered:
    """检查 4：默认值、登记表与 `PROD_PROFILE` 的口径。"""

    def test_default_rank_is_score(self):
        assert answer_mod.CHAPTER_RANK_DEFAULT == "score", (
            "默认值变了 —— 那会让所有历史 cross_chunk 读数漂移")

    def test_rank_registry_contains_both_keys(self):
        assert set(answer_mod.CHAPTER_RANK_KEYS) == {"score", "focus_assertions"}, (
            f"登记表被改了：{sorted(answer_mod.CHAPTER_RANK_KEYS)}")

    def test_prod_profile_does_not_carry_chapter_rank(self):
        """生产档不许带这个键 —— 带了就等于默认打开，违背"默认逐位不变"。"""
        from arkrag import pipeline as pipeline_mod
        opts = pipeline_mod.prod_options()
        assert "chapter_rank" not in opts, (
            f"PROD_PROFILE 里出现了 chapter_rank：{opts}")
        assert opts.get("chapter_cap") == 4, "生产档的 chapter_cap 不许动"


# ------------------------------------------------------------------ 检查 5
class TestPipelineWiring:
    """检查 5：`ask_once` 的接线与门控（用替身，不加载产物）。"""

    def _pipe(self, *, recorded: list[dict]):
        class _Ans:
            def answer(self, question, context, **kwargs):
                recorded.append(dict(kwargs))
                return {"answer": "", "citations": [], "abstained": True,
                        "abstain_reason": "stub", "warnings": [], "stats": {}}

        class _Ret:
            chunks: dict = {}
            graph: dict = {"entities": []}
            query_context = None
            budget = {"max_context_tokens": 24000}
            assertions_by_chunk: dict = {}
            display_name = staticmethod(lambda x: x)

            def search(self, qa, top_k=20):
                return {"hits": [], "stats": {"top_score": 1.0}}

            def expand(self, seed_ids, depth=1, query_analysis=None,
                       seed_chunk_ids=None):
                return {"expansions": [], "stats": {}}

            def build_context(self, hits, expansions, query_analysis=None, **kw):
                return {"units": [], "context_digest": "", "stats": {}}

        return {"retriever": _Ret(), "answerer": _Ans(), "top_k": 20,
                "graph_depth": 1, "vector_union": None, "entities": []}

    def test_default_options_do_not_pass_chapter_rank(self):
        from arkrag import pipeline as pipeline_mod
        seen: list[dict] = []
        pipeline_mod.ask_once(self._pipe(recorded=seen), QUESTION,
                              options={"chapter_cap": 4})
        assert "chapter_rank" not in seen[0], (
            f"默认档下多传了关键字：{sorted(seen[0])}")

    def test_non_default_rank_is_forwarded_only_when_cap_on(self):
        """非默认档才拼关键字；cap 关掉时传了也照样转发（不静默吞取值）。

        这一条与 `chapter_cap` 的处理**刻意不同**：那个是 cap=0 就归零（因为
        0 本身是"关"的语义），而 `chapter_rank` 是**排序键的名字**，吞掉它会让
        调用方以为开了、读数不变、去查别处。`_chapter_answer` 在 cap=0 时根本
        不读它，所以多传是空转，无害。
        """
        from arkrag import pipeline as pipeline_mod
        seen: list[dict] = []
        pipe = self._pipe(recorded=seen)
        pipeline_mod.ask_once(pipe, QUESTION,
                              options={"chapter_cap": 4,
                                       "chapter_rank": "focus_assertions"})
        assert seen[0].get("chapter_rank") == "focus_assertions", (
            f"非默认档没转发：{sorted(seen[0])}")

        seen.clear()
        pipeline_mod.ask_once(pipe, QUESTION,
                              options={"chapter_rank": "focus_assertions"})
        assert seen[0].get("chapter_rank") == "focus_assertions", (
            "cap 关闭时把 chapter_rank 吞掉了 —— 读数不变但调用方以为开了")


# ------------------------------------------------------------------ 检查 6
class TestNoCollateralDamage:
    """检查 6：非章节题的返回一个字节都不许改。"""

    def test_non_chapter_question_is_untouched(self):
        units = _plain_units(["a01", "a02", "a03"])
        q = "「整合运动」和「罗德岛」是什么关系？"
        base = answer_mod.ExtractiveAnswerer().answer(
            q, _ctx(units), top_score=1.0, query_analysis=_qa(FOCUS),
            chapter_cap=4)
        got = answer_mod.ExtractiveAnswerer().answer(
            q, _ctx(units), top_score=1.0, query_analysis=_qa(FOCUS),
            chapter_cap=4, chapter_rank="focus_assertions")
        assert _stable_view(got) == _stable_view(base), (
            "非章节题（不含触发词）的返回被 chapter_rank 改变了")
