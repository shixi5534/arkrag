"""multihop 多锚点合取装配（`answer._multihop_conjunction`）的门。

## 这个文件守的是什么

目标失败形态：`judge.judge_answer` 的 `multihop` 分支要求 gold 与**每一个**问句
锚点各自有一条断言共现。实测 480 道唯一可判题里该类 96 道、通过 41；55 道失败中
33 道是 `gold_not_linked_to_every_anchor`（gold 出现了，但只挂上一个锚点）。
本通路在 claim 列表之外补一条 `{实体}与{A}、{B}相关联` 的合取句救这一类。

三条性质各自有一种静默失效的方式，所以分别钉（这是本次要求的三条，缺一不可）：

1. **默认关闭时返回值逐位不变。** 开关是 `answer()` 的关键字参数，默认值读模块
   常量 `MULTIHOP_CONJUNCTION is False`。失效方式：顺手把 `stats` 也写进默认档——
   屏幕上答案一模一样，历史产物的字节却全变了，而且没有任何测试会叫。
   这里用 `repr` 全等比对（含 `stats`/`warnings`/`citations` 的每个键），
   并额外钉住"默认档 `stats` 里没有 `multihop_conjunction` 这个键"。

2. **开关打开且条件满足时合成 claim 出现，且 citation 指向该实体所在的块。**
   失效方式：句子生成了但引用指向第一个块（不是实体所在块），或
   `text[start:end] != evidence`。两者都会让 `verify_citations` 把引用整条剔除，
   答案退回"没有合取句"却看不出问题。所以这里逐条回读块正文验证偏移自洽。

3. **条件不满足时不生成。** 三个"不满足"必须都是落回原路径而不是猜：
   非多跳题（触发面外）、实体是锚点（判据 `gold in anc` 直接失败）、
   实体在上下文里定位不到行（发不出可校验引用）。另外还要钉两件事——
   **没有空位时不生成**（本通路只许加句、不许挤掉原 claim），以及
   **已经连全锚点的实体不进候选**（否则会拿已通过的题去换没通过的题）。

## 这个文件**不**证明什么

它不证明分数会涨，也不证明涨的部分不是注水。合取句是**装配**出来的句子，不是原文
三元组——`bridge.py` 约束 3 记的同型否决（编造句与真实三元组 marginal 差 0 题）
在这里同样适用。端到端读数与注水对照臂必须由 Lead 经 `run_guarded.py` 串行跑，
本文件的合成上下文只能证明"装配逻辑的代数性质"，不能替代那一步。
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from arkrag import answer as answer_mod
from arkrag import judge as judge_mod
from arkrag import pipeline as pipeline_mod

ROOT = Path(__file__).resolve().parents[1]
MIN = answer_mod.MIN_EVIDENCE_CHARS

#: 多跳题的问法骨架，与 `bridge.is_bridge_question` 的触发面一致（`中间实体` + 2 锚点）。
Q_MH = "已知「云兽」通过唯一一个中间实体与「变形者集群」相连。这个中间实体是谁？"
#: 触发面外：没有「中间实体」，只有两个锚点。
Q_DR = "「云兽」与「变形者集群」处于哪种关系？"

#: 金块正文：一行讲 阿米娅 与 云兽，一行讲 阿米娅 与 变形者集群。
#: 两行都含 阿米娅，所以合取句的引用指哪一行都自洽；测试断言的是"指向含该实体的块"。
TEXT_GOLD = (
    "阿米娅：云兽最近很不安分，你得看好它。\n"
    "阿米娅：变形者集群那边还没有消息。"
)
TEXT_OTHER = (
    "凯尔希：切尔诺伯格已经沦陷了。\n"
    "博士：罗德岛的航线不能改。"
)


def _chunk(chunk_id: str, text: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": chunk_id.split("#")[0],
        "title": "测试标题",
        "chapter": "测试章节",
        "text": text,
    }


def _brief(aid: str, subject: str, predicate: str, obj: str,
           evidence: str) -> dict:
    return {"assertion_id": aid, "subject": subject, "predicate": predicate,
            "object": obj, "evidence": evidence, "confidence": 0.9}


def _span_of(brief: dict, text: str) -> dict:
    start = text.index(brief["evidence"])
    return {"assertion_id": brief["assertion_id"], "evidence": brief["evidence"],
            "start": start, "end": start + len(brief["evidence"])}


def _unit(chunk_id: str, text: str, briefs: list[dict], *,
          score: float = 1.0, entities=None) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": chunk_id.split("#")[0],
        "title": "测试标题",
        "chapter": "测试章节",
        "text": text,
        "assertions": list(briefs),
        "evidence_spans": [_span_of(b, text) for b in briefs],
        "entities": list(entities or []),
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


def _gold_units() -> list[dict]:
    """只连上一个锚点的上下文：`阿米娅` 与 云兽 共现，变形者集群只出现在另一行。

    这正是 `gold_not_linked_to_every_anchor` 的形态——一条 claim 里
    `阿米娅 交谈 云兽`，探针句子另起一行提 `变形者集群`。
    """
    b1 = _brief("a-1", "阿米娅", "交谈", "云兽", "阿米娅：云兽最近很不安分，你得看好它。")
    return [_unit("doc#001", TEXT_GOLD, [b1], entities=["阿米娅", "云兽", "变形者集群"])]


def _asked(units: list[dict], question: str = Q_MH, *, qa=None, **kwargs) -> dict:
    return answer_mod.ExtractiveAnswerer().answer(
        question, _ctx(units), top_score=1.0,
        query_analysis=qa if qa is not None else _qa("云兽", "变形者集群"),
        **kwargs)


def _assert_citations_self_consistent(out: dict, units: list[dict]) -> None:
    """每条引用都必须能回读、且长度达标（`verify_citations` 的两条硬判据）。"""
    by_id = {u["chunk_id"]: u for u in units}
    for cite in out["citations"]:
        chunk = by_id.get(str(cite["chunk_id"]))
        assert chunk is not None, f"引用指向不存在的块 {cite['chunk_id']}"
        text = chunk["text"]
        got = text[int(cite["start"]):int(cite["end"])]
        assert got == cite["evidence"], (
            f"引用不自洽：{cite['chunk_id']} "
            f"[{cite['start']},{cite['end']}) 取到 {got!r}，"
            f"却声明 evidence={cite['evidence']!r}"
        )
        assert len(cite["evidence"]) >= MIN, (
            f"引用证据 {cite['evidence']!r} 短于 MIN_EVIDENCE_CHARS={MIN}，"
            "会被 verify_citations 判 evidence_too_short"
        )


# ------------------------------------------------------- 1. 默认关闭 = 逐位不变
class TestDefaultOffIsBitIdentical:
    """检查 1：开关默认关，返回 dict 逐位不变（含 stats 的键集合）。"""

    def test_switch_defaults_to_off(self) -> None:
        assert answer_mod.MULTIHOP_CONJUNCTION is False, (
            "MULTIHOP_CONJUNCTION 不是 False 了。这条通路的合取句是**装配**出来的，"
            "不是原文三元组；默认打开等于把一条带注水风险的句子直接送进生产答案，"
            "而没有任何端到端读数支持它。"
        )

    def test_source_literal_is_false_not_just_the_runtime_value(self) -> None:
        """源码字面量必须是 `MULTIHOP_CONJUNCTION = False`。

        只断言运行期取值不够：`answer()` 的默认值写成
        `None if not X else MULTIHOP_CONJUNCTION` 这类绕法，运行时照样能取到 False，
        但源码里那个数已经变了。这里按行扫源码，钉的是**那一行本身**。
        """
        src = (ROOT / "src" / "arkrag" / "answer.py").read_text(encoding="utf-8")
        decl = [ln.strip() for ln in src.splitlines()
                if ln.strip().startswith("MULTIHOP_CONJUNCTION =")]
        assert decl == ["MULTIHOP_CONJUNCTION = False"], (
            f"常量声明变了：{decl}。这条通路是实验通路，默认必须关。"
        )

    def test_explicit_false_equals_omitted(self) -> None:
        """显式传 `multihop_conjunction=False` 必须与完全不传逐位相同。"""
        units = _gold_units()
        omitted = _asked(units)
        explicit = _asked(units, multihop_conjunction=False)
        for key in ("answer", "citations", "confidence", "abstained",
                    "abstain_reason", "warnings", "stats"):
            assert repr(omitted[key]) == repr(explicit[key]), (
                f"默认档与显式 False 的 `{key}` 不同：\n"
                f"  omitted={omitted[key]!r}\n  explicit={explicit[key]!r}"
            )

    def test_no_new_stats_key_when_off(self) -> None:
        """默认档不许出现 `multihop_conjunction` 这个 stats 键。

        失效方式：顺手记账，屏幕上答案一模一样，历史产物的字节却全变了。
        """
        out = _asked(_gold_units())
        assert "multihop_conjunction" not in out["stats"], (
            f"默认档的 stats 里多了 multihop_conjunction 键：{sorted(out['stats'])}。"
            "默认档的返回结构必须与改造前逐位相同。"
        )

    def test_answer_text_has_no_conjunction_sentence_when_off(self) -> None:
        out = _asked(_gold_units())
        assert "相关联" not in out["answer"], (
            f"默认档的答案里出现了合取句式：「{out['answer']}」"
        )


# ---------------------------------------- 2. 打开且条件满足 = 句子 + 正确引用
class TestConjunctionIsAssembled:
    """检查 2：合取句出现、插在最前、citation 指向该实体所在的块。"""

    def _out(self, **kwargs) -> dict:
        return _asked(_gold_units(), multihop_conjunction=True, **kwargs)

    def test_sentence_names_the_entity_and_every_anchor(self) -> None:
        out = self._out()
        assert out["stats"]["multihop_conjunction"]["applied"] is True, (
            f"合取句没有生成：{out['stats']['multihop_conjunction']}"
        )
        entity = out["stats"]["multihop_conjunction"]["entities"][0]
        # 句子必须同时含实体与**全部**锚点——判据要的正是这个共现形态。
        assert f"{entity}与云兽、变形者集群" in out["answer"], (
            f"合取句没有同时含实体与两个锚点：「{out['answer']}」"
        )

    def test_judge_now_passes_on_the_assembled_answer(self) -> None:
        """这是这条通路**唯一**该拿到的收益：判据从"缺锚点"变成"连全"。

        判据本身一个字都没改——这里只是用 `judge.judge_answer` 验证
        `gold_not_linked_to_every_anchor → gold_linked_to_every_anchor`。
        """
        units = _gold_units()
        off = _asked(units)
        on = _asked(units, multihop_conjunction=True)
        item = {"answer": "阿米娅", "acceptable_answers": ["阿米娅"]}
        before = judge_mod.judge_answer("multihop", off["answer"], item, question=Q_MH)
        after = judge_mod.judge_answer("multihop", on["answer"], item, question=Q_MH)
        assert before["pass"] is False, (
            f"前提不成立：默认档本来就已经通过（{before}），这条用例失去判别力"
        )
        assert before["reason"] == "gold_not_linked_to_every_anchor", before
        assert after["pass"] is True, (
            f"装配后判据仍不通过：{after}；答案=「{on['answer']}」"
        )
        assert after["reason"] == "gold_linked_to_every_anchor"

    def test_citation_points_at_the_chunk_holding_the_entity(self) -> None:
        """引用必须指向**该实体所在的块**，不是第一个块、不是 gold 块以外的块。"""
        units = _gold_units()
        # 第二个块不含该实体，用来证明引用没有"顺手指向第一个块"。
        units.append(_unit("doc#002", TEXT_OTHER,
                           [_brief("a-2", "凯尔希", "隶属", "罗德岛",
                                   "凯尔希：切尔诺伯格已经沦陷了。")],
                           score=0.5, entities=["凯尔希", "罗德岛"]))
        out = _asked(units, multihop_conjunction=True)
        entity = out["stats"]["multihop_conjunction"]["entities"][0]
        conj = [c for c in out["citations"] if c["claim"].startswith(entity)]
        assert conj, f"找不到合取句的引用：{out['citations']}"
        cite = conj[0]
        by_id = {u["chunk_id"]: u for u in units}
        assert entity in by_id[str(cite["chunk_id"])]["text"], (
            f"引用指向的块 {cite['chunk_id']} 不含实体 {entity}"
        )
        _assert_citations_self_consistent(out, units)

    def test_conjunction_sentence_is_stated_first(self) -> None:
        """插在最前（不替换原 claim 列表）。"""
        out = self._out()
        first = out["answer"].split("。")[0]
        assert "相关联" in first, (
            f"合取句不在最前：「{out['answer']}」"
        )

    def test_answer_never_exceeds_max_claims(self) -> None:
        """句数仍受 `max_claims` 约束（与 `bridge.merge` 同一条形态红线）。"""
        units = _gold_units()
        for cap in (1, 2, 3, 8):
            out = answer_mod.ExtractiveAnswerer(max_claims=cap).answer(
                Q_MH, _ctx(units), top_score=1.0,
                query_analysis=_qa("云兽", "变形者集群"),
                multihop_conjunction=True,
            )
            n = len([s for s in out["answer"].split("。") if s.strip()])
            assert n <= cap, f"max_claims={cap} 时答案有 {n} 句：{out['answer']!r}"

    def test_coverage_pairs_are_never_reduced(self) -> None:
        """**核心不变式**：前插挤掉尾部，但覆盖对（判据数的那个东西）一句不许少。

        这是"结构上不可能 pass→fail"的机检形态：判据通过 ⟺ `(gold, a)` 对齐全，
        所以只要覆盖对集合是超集，原来通过的仍然通过。
        """
        # 造一个**顶满**的答案（max_claims=2，正好 2 条 claim）：
        #   claim 1 = `阿米娅交谈云兽`      → 覆盖 (1, 云兽)
        #   claim 2 = `凯尔希隶属罗德岛`    → 一个锚点都不覆盖
        # `阿米娅` 只连上 1/2 锚点，且答案已满 ⇒ 正是目标失败形态在饱和答案上的样子。
        text = (
            "阿米娅：云兽最近很不安分，你得看好它。\n"
            "凯尔希：罗德岛的航线不能改。\n"
            "变形者集群那边还没有消息。\n"
        )
        briefs = [
            _brief("a-1", "阿米娅", "交谈", "云兽", "阿米娅：云兽最近很不安分，你得看好它。"),
            _brief("a-2", "凯尔希", "隶属", "罗德岛", "凯尔希：罗德岛的航线不能改。"),
        ]
        units = [_unit("doc#010", text, briefs,
                       entities=["阿米娅", "云兽", "凯尔希", "罗德岛", "变形者集群"])]
        anchors = ["云兽", "变形者集群"]

        def run(on: bool) -> dict:
            return answer_mod.ExtractiveAnswerer(max_claims=2).answer(
                Q_MH, _ctx(units), top_score=1.0,
                query_analysis=_qa("云兽", "变形者集群"),
                multihop_conjunction=on)

        off, on = run(False), run(True)
        assert len([s for s in off["answer"].split("。") if s.strip()]) == 2
        before = answer_mod._coverage_pairs(
            [s for s in off["answer"].split("。") if s.strip()], anchors)
        after = answer_mod._coverage_pairs(
            [s for s in on["answer"].split("。") if s.strip()], anchors)
        assert before, "夹具失效：默认档没有覆盖对，不变式变成空转"
        assert before <= after, (
            f"截断丢掉了覆盖对：丢失 {sorted(before - after)}；\n"
            f"  之前={off['answer']!r}\n  之后={on['answer']!r}"
        )
        # 这条夹具下合取句应当生成（阿米娅只连 1/2 个锚点，且尾部那句不含锚点，
        # 挤掉它不损失任何覆盖对）。
        assert on["stats"]["multihop_conjunction"]["applied"] is True, (
            on["stats"]["multihop_conjunction"]
        )
        assert "阿米娅与云兽、变形者集群相关联" in on["answer"], on["answer"]

    def test_no_citation_points_at_a_sentence_missing_from_the_answer(self) -> None:
        """被挤掉的 claim，它的引用也必须一并不发。

        失效方式（反向实测漏掉的那一笔）：只截正文不截 `accepted`，于是
        `citations` 里留着一条 `claim`，而那句话已经不在答案里了——读者拿引用去
        找结论，找不到对应句子。这种形态 `verify_citations` **查不出来**：它只查
        偏移是否逐字对上，不查"这句话在不在答案里"。
        """
        text = (
            "阿米娅：云兽最近很不安分，你得看好它。\n"
            "凯尔希：罗德岛的航线不能改。\n"
            "变形者集群那边还没有消息。\n"
        )
        briefs = [
            _brief("a-1", "阿米娅", "交谈", "云兽", "阿米娅：云兽最近很不安分，你得看好它。"),
            _brief("a-2", "凯尔希", "隶属", "罗德岛", "凯尔希：罗德岛的航线不能改。"),
        ]
        units = [_unit("doc#010", text, briefs,
                       entities=["阿米娅", "云兽", "凯尔希", "罗德岛", "变形者集群"])]
        out = answer_mod.ExtractiveAnswerer(max_claims=2).answer(
            Q_MH, _ctx(units), top_score=1.0,
            query_analysis=_qa("云兽", "变形者集群"),
            multihop_conjunction=True)
        segs = [s for s in out["answer"].split("。") if s.strip()]
        orphans = [c["claim"] for c in out["citations"]
                   if not any(c["claim"] in seg for seg in segs)]
        assert not orphans, (
            f"这些引用指向了答案里没有的句子：{orphans}；\n  答案={out['answer']!r}"
        )
        _assert_citations_self_consistent(out, units)

    def test_stats_records_the_entity_source(self) -> None:
        """必须记下实体是哪来的——否则分不清"自己找的"和"外面给的"。"""
        out = self._out()
        block = out["stats"]["multihop_conjunction"]
        assert block["entity_source"] == "stated_claim_endpoint", block
        assert block["anchors"] == ["云兽", "变形者集群"], block

    def test_explicit_entity_arm_is_marked_as_explicit(self) -> None:
        """`multihop_entity=` 只用于诊断/上界臂，且必须被标成 explicit。"""
        out = _asked(_gold_units(), multihop_conjunction=True,
                     multihop_entity="阿米娅")
        block = out["stats"]["multihop_conjunction"]
        assert block["entity_source"] == "explicit", block
        assert block["entities"] == ["阿米娅"], block


# ------------------------------------------- 3. 条件不满足 = 不生成（不猜）
class TestNotGeneratedWhenConditionsFail:
    """检查 3：三种"做不到"都必须落回原路径。"""

    def test_non_multihop_question_does_not_generate(self) -> None:
        """触发面外（没有「中间实体」）：一句都不加，答案与默认档逐位相同。"""
        units = _gold_units()
        off = _asked(units, question=Q_DR, qa=_qa("云兽", "变形者集群"))
        on = _asked(units, question=Q_DR, qa=_qa("云兽", "变形者集群"),
                    multihop_conjunction=True)
        assert on["answer"] == off["answer"]
        assert on["stats"]["multihop_conjunction"]["applied"] is False
        assert on["stats"]["multihop_conjunction"]["reason"] == "not_multihop"

    def test_entity_that_is_an_anchor_does_not_generate(self) -> None:
        """锚点不许当实体：判据在 `gold in anc` 时直接判失败。

        两个入口都要钉：走候选池时由 `_multihop_entity_candidates` 排除；走
        `multihop_entity=` 显式臂时**池子不筛**（上界臂要能指任何实体），所以只能靠
        `_multihop_conjunction` 里那道守卫。只测前者的话那道守卫是死代码——
        反向实测正是这么发现的（删掉守卫、前一种用例仍然全绿）。
        """
        # 入口一：claims 里只剩锚点作为端点：`云兽 交谈 变形者集群`。
        text = "云兽与变形者集群走在一起。"
        b = _brief("a-3", "云兽", "交谈", "变形者集群", "云兽与变形者集群走在一起。")
        units = [_unit("doc#003", text, [b], entities=["云兽", "变形者集群"])]
        out = _asked(units, multihop_conjunction=True)
        block = out["stats"]["multihop_conjunction"]
        assert block["applied"] is False, block
        assert "相关联" not in out["answer"]

        # 入口二：显式臂直接点名一个锚点。池子不筛，守卫必须拦。
        out2 = _asked(_gold_units(), multihop_conjunction=True,
                      multihop_entity="云兽")
        block2 = out2["stats"]["multihop_conjunction"]
        assert block2["applied"] is False, (
            f"显式臂允许把锚点当实体了：{block2}"
        )
        assert block2["skipped"][0]["why"] == "is_anchor", block2
        assert "相关联" not in out2["answer"]

    def test_entity_without_locatable_line_does_not_generate(self) -> None:
        """定位不到自洽的行就不生成：发一条指不出处的句子会被 verify 整条剔除。"""
        units = _gold_units()
        out = _asked(units, multihop_conjunction=True,
                     multihop_entity="不存在的实体名")
        block = out["stats"]["multihop_conjunction"]
        assert block["applied"] is False, block
        assert block["skipped"][0]["why"] == "no_locatable_line", block
        assert "相关联" not in out["answer"]

    def test_no_sentence_room_does_not_generate(self) -> None:
        """顶满且前插会损失覆盖对时不生成——宁可漏救，不许暗换。

        这条钉的是**守卫**而不是"有空位才加"：`max_claims=1` 且唯一那句是
        `阿米娅—云兽`（含 1 个锚点）。前插合取句会把这句挤掉，而它正是覆盖对的
        唯一来源 ⇒ 守卫必须拒绝。
        """
        units = _gold_units()
        off = answer_mod.ExtractiveAnswerer(max_claims=1).answer(
            Q_MH, _ctx(units), top_score=1.0,
            query_analysis=_qa("云兽", "变形者集群"),
            multihop_conjunction=False,
        )
        on = answer_mod.ExtractiveAnswerer(max_claims=1).answer(
            Q_MH, _ctx(units), top_score=1.0,
            query_analysis=_qa("云兽", "变形者集群"),
            multihop_conjunction=True,
        )
        block = on["stats"]["multihop_conjunction"]
        assert block["applied"] is False, block
        assert block["reason"] == "coverage_would_regress", block
        assert block["skipped"][0]["why"] == "would_lose_coverage", block
        # 拒绝时答案必须与默认档逐位相同。
        assert on["answer"] == off["answer"]

    def test_entity_already_linked_to_every_anchor_is_not_a_candidate(self) -> None:
        """已经连全锚点的实体不进候选：给它补句子只会挤掉一条真 claim。"""
        # 两条 claim 把 `阿米娅` 与两个锚点都连上 ⇒ 判据本来就成立。
        b1 = _brief("a-1", "阿米娅", "交谈", "云兽", "阿米娅：云兽最近很不安分，你得看好它。")
        b2 = _brief("a-2", "阿米娅", "交谈", "变形者集群", "阿米娅：变形者集群那边还没有消息。")
        units = [_unit("doc#001", TEXT_GOLD, [b1, b2],
                       entities=["阿米娅", "云兽", "变形者集群"])]
        out = _asked(units, multihop_conjunction=True)
        block = out["stats"]["multihop_conjunction"]
        assert block["applied"] is False, (
            f"对已经连全锚点的实体仍生成了合取句：{block}"
        )
        assert block["entities"] == [], block


# ------------------------------------------------- 装配层的纯函数与接线门
class TestHelperContract:
    """`_multihop_entity_candidates` / `_locate_line` 的两条代数性质。"""

    def test_anchors_are_never_candidates(self) -> None:
        claims = [{"subject": "云兽", "object": "阿米娅"}]
        got = answer_mod._multihop_entity_candidates(
            claims, ["云兽", "变形者集群"])
        assert got == ["阿米娅"], got

    def test_partial_coverage_ranks_before_no_coverage(self) -> None:
        """只挂上部分锚点的实体排在"一个都没挂上"的前面——前者才是目标失败形态。"""
        claims = [
            {"subject": "无关甲", "object": "无关乙"},      # 一个锚点都没连
            {"subject": "阿米娅", "object": "云兽"},        # 连上 1/2 个锚点
        ]
        got = answer_mod._multihop_entity_candidates(
            claims, ["云兽", "变形者集群"])
        assert got.index("阿米娅") < got.index("无关甲"), got

    def test_locate_line_returns_none_for_short_lines(self) -> None:
        """短于 MIN_EVIDENCE_CHARS 的行不能当引用。"""
        units = [_unit("doc#009", "云。", [], entities=[])]
        assert answer_mod._locate_line(units, "云") is None

    def test_locate_line_offsets_are_self_consistent(self) -> None:
        units = [_unit("doc#001", TEXT_GOLD, [], entities=[])]
        span = answer_mod._locate_line(units, "云兽")
        assert span is not None
        assert span["text"][span["start"]:span["end"]] == span["line"]


class TestPipelineWiring:
    """`ask_once` 只在显式开启且是多跳题时才传这个关键字。

    这条门必须**真的调一次 `ask_once`**：只断言 `PROD_PROFILE` 里没有这个键，
    挡不住"接线写错方向"（例如无条件传 True）。反向实测正是这么发现的——
    把 `is_bridge_question` 门控删掉后，那条弱断言仍然全绿。
    """

    def _pipe(self, sink: list) -> dict:
        """一条最小链：检索/扩展返回空，答案器把收到的关键字记下来。"""

        class _Answerer:
            def answer(self, question, context, **kwargs):
                sink.append({"question": question, "kwargs": set(kwargs),
                             "mh": kwargs.get("multihop_conjunction")})
                return {"answer": "", "citations": []}

        class _Retriever:
            graph = {"entities": [], "alias_index": {}}
            assertions_by_chunk: dict = {}
            query_context = None
            budget: dict = {"max_context_tokens": 24000}

            def search(self, qa, top_k=0):
                return {"hits": [], "stats": {"top_score": 1.0}}

            def expand(self, seeds, depth=0, query_analysis=None, seed_chunk_ids=None):
                return {"expansions": [], "stats": {}}

            def build_context(self, hits, expansions, query_analysis=None, **kw):
                return {"units": [], "context_digest": "", "stats": {"units": 0}}

            def display_name(self, x):
                return str(x)

        return {"retriever": _Retriever(), "answerer": _Answerer(), "top_k": 5,
                "graph_depth": 1, "generation_id": "g-test"}

    def test_off_by_default_passes_no_keyword(self) -> None:
        seen: list = []
        pipeline_mod.ask_once(self._pipe(seen), Q_MH)
        assert seen, "`ask_once` 没有调用答案器"
        assert "multihop_conjunction" not in seen[0]["kwargs"], (
            f"默认档多传了关键字：{sorted(seen[0]['kwargs'])}"
        )

    def test_on_but_non_multihop_passes_no_keyword(self) -> None:
        seen: list = []
        pipeline_mod.ask_once(self._pipe(seen), Q_DR,
                              options={"multihop_conjunction": True})
        assert seen, "`ask_once` 没有调用答案器"
        assert "multihop_conjunction" not in seen[0]["kwargs"], (
            "非多跳题也接了合取通路：门控没生效"
        )

    def test_on_and_multihop_passes_the_keyword(self) -> None:
        seen: list = []
        pipeline_mod.ask_once(self._pipe(seen), Q_MH,
                              options={"multihop_conjunction": True})
        assert seen, "`ask_once` 没有调用答案器"
        assert seen[0]["mh"] is True, (
            f"多跳题打开开关后没有接线：{sorted(seen[0]['kwargs'])}"
        )

    def test_prod_profile_does_not_enable_it(self) -> None:
        assert pipeline_mod.PROD_PROFILE["options"].get("multihop_conjunction") is None, (
            "生产档里出现了 multihop_conjunction：这条通路是默认关的实验通路，"
            "进生产档必须有端到端读数与注水对照臂，两者本轮都没有"
        )
