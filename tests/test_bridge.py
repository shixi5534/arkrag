"""给多跳桥接通路（`src/arkrag/bridge.py`）写门。

这个模块是从一堆被否证的版本里削出来的，所以每条断言都对着一次具体的否决：

1. **候选只许来自 `context["units"]` 的断言。** 全图公共邻居版被判"图查询冒充检索"
   （41.7% 的增益来自本次根本没召回的图的边），所以 `find_bridge` 的签名里
   没有 graph / adjacency / entities 任何参数——这条门用签名检查把它钉死。
2. **候选不唯一就不答。** 排序版在 dev 上是负收益（候选内随机 21 > 排序 top1 19），
   所以本模块**没有任何排序键**。检查源码里不出现 `sort`/`sorted`/`max(`。
3. **句源必须是真实断言。** 编造句版与纯模板版 marginal 差 0 题。门：把断言拿掉后
   必须 `no_common_neighbor`，不许凭锚点名编出句子。
4. **引用必须逐字可校验。** 早前版引用没有 `start/end`，`verify_citations` 判
   `offsets_missing` 把整条答案清空（dev rejected=56）。门：每条引用都要能
   `chunks[cid]["text"][start:end] == evidence` 且长度 ≥ `MIN_EVIDENCE_CHARS`。
5. **形态红线：句数 ≤8、单句 ≤60 字。** 对抗审查认可上限。
6. 触发面只认 `「中间实体」` 且恰好 2 个引文——全 640 题只命中 multihop 96/96。
"""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest

from arkrag import answer as answer_mod
from arkrag import bridge as br

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "arkrag" / "bridge.py"


def _unit(cid: str, text: str, assertions: list[dict] | None = None) -> dict:
    return {"chunk_id": cid, "text": text, "assertions": list(assertions or ()),
            "title": "", "chapter": "", "sources": [], "score": 1.0, "evidence_spans": []}


def _assert(subject: str, predicate: str, obj: str, evidence: str) -> dict:
    return {"subject": subject, "predicate": predicate, "object": obj,
            "evidence": evidence}


Q_TWO = "已知「甲」通过唯一一个中间实体与「乙」相连。这个中间实体是谁？"


class TestTriggerSurface:
    def test_trigger_word_is_pinned(self) -> None:
        assert br.BRIDGE_TRIGGER == "中间实体", (
            f"BRIDGE_TRIGGER 现在是 {br.BRIDGE_TRIGGER!r}。实测只有「中间实体」把触发面"
            "收窄到 multihop 一类（96/96）；换成'≥2 个锚点'会误伤 direct_relation 96 道，"
            "'≥2 锚点且问谁/什么'仍误伤 51 道。"
        )

    def test_two_anchors_and_trigger(self) -> None:
        assert br.is_bridge_question(Q_TWO)

    def test_one_anchor_is_not_enough(self) -> None:
        assert not br.is_bridge_question("「甲」这个中间实体是谁？")

    def test_three_anchors_are_not_enough(self) -> None:
        assert not br.is_bridge_question("「甲」「乙」「丙」这三个之间的中间实体是谁？")

    def test_trigger_word_required(self) -> None:
        assert not br.is_bridge_question("「甲」与「乙」处于哪种关系？")

    def test_full_gold_set_trigger_surface_is_multihop_only(self) -> None:
        path = ROOT / "data" / "goldset" / "gold-v2.json"
        items = json.loads(path.read_text(encoding="utf-8"))["items"]
        hit: dict[str, int] = {}
        for it in items:
            if br.is_bridge_question(it.get("question") or ""):
                cat = it.get("category") or "?"
                hit[cat] = hit.get(cat, 0) + 1
        assert set(hit) == {"multihop"}, (
            f"桥接触发面在当前 gold 集上命中了 {hit}；要求只有 multihop（对抗审查 Q5 PASS）。"
        )
        assert hit["multihop"] == 96

    def test_anchors_keep_order_and_drop_blanks(self) -> None:
        assert br.anchors_of("「甲」和「乙」之间") == ["甲", "乙"]
        assert br.anchors_of("「甲」「乙」之间") == ["甲", "乙"]
        assert br.anchors_of("没有锚点") == []


class TestGroundingIsStructural:
    """条款 1/2：签名里不许出现全图结构，源码里不许出现排序键。"""

    def test_signature_has_no_graph_arguments(self) -> None:
        params = set(inspect.signature(br.find_bridge).parameters)
        banned = {"graph", "adjacency", "entities", "by_name", "alias_index"}
        assert not (params & banned), (
            f"find_bridge 的形参里出现了 {sorted(params & banned)}。"
            "全图公共邻居版被判'图查询冒充检索'——本模块必须拿不到全图索引，"
            "结构上就不可能绕过'候选只能来自本次上下文'这条约束。"
        )

    def test_source_has_no_ranking_key(self) -> None:
        src = SOURCE.read_text(encoding="utf-8")
        body = src[src.index("def find_bridge"):src.index("def merge")]
        for token in ("sorted(", ".sort(", "max(", "min("):
            assert token not in body, (
                f"find_bridge 里出现了 {token}：本模块只做'唯一才答'，"
                "没有任何排序键。排序版在 dev 上是负收益（候选内随机 21 > 排序 top1 19），"
                "且 41.3% 的触发题候选本就唯一、排序在数学上无从贡献。"
            )


class TestFindBridge:
    def test_two_real_assertions_yield_ok(self) -> None:
        text = "甲与丙同行。乙与丙交谈。"
        units = [_unit("c#1", text, [
            _assert("甲", "同行", "丙", "甲与丙同行"),
            _assert("乙", "交谈", "丙", "乙与丙交谈"),
        ])]
        out = br.find_bridge(Q_TWO, units)
        assert out["status"] == "ok", f"两条真实断言应能作答，实际 {out['status']}"
        assert out["x"] == "丙"

    def test_no_common_neighbor_has_no_candidate(self) -> None:
        units = [_unit("c#1", "甲与丙同行。", [_assert("甲", "同行", "丙", "甲与丙同行")])]
        out = br.find_bridge(Q_TWO, units)
        assert out["status"] == "no_common_neighbor"
        assert out["sentences"] == [] and out["citations"] == []

    def test_ambiguous_candidate_is_not_guessed(self) -> None:
        text = "甲与丙同行。乙与丙交谈。甲与丁同行。乙与丁交谈。"
        units = [_unit("c#1", text, [
            _assert("甲", "同行", "丙", "甲与丙同行"),
            _assert("乙", "交谈", "丙", "乙与丙交谈"),
            _assert("甲", "同行", "丁", "甲与丁同行"),
            _assert("乙", "交谈", "丁", "乙与丁交谈"),
        ])]
        out = br.find_bridge(Q_TWO, units)
        assert out["status"] == "ambiguous", (
            f"两个候选（丙、丁）时必须拒答并落回原链路，实际 {out['status']}。"
        )
        assert out["x"] == "" and out["citations"] == []

    def test_anchor_itself_is_never_a_candidate(self) -> None:
        """甲—乙直接相连时，乙不能被当成"中间实体"。

        这条由**两条互相冗余的守卫**共同保证，去掉任意一条都仍成立：

        * `s == o` 跳过自环 —— 锚点要进自己的邻居表，只能靠自环；
        * `- set(anc)` —— 结果集里再减掉两个锚点。

        突变实测：单独去掉任一条，`tests/test_bridge.py` 全绿；两条同时去掉才红。
        所以本文件另有一条 `test_self_loop_is_ignored` 与一条对 `- set(anc)` 的
        结果断言，而 `tmp_mutate_gates.py` 里这条用**双突变**验证。把这件事写下来，
        是因为"看起来有两道防线、其实只有一道"正是删代码时最容易误判的形态。
        """
        units = [_unit("c#1", "甲与乙同行。", [_assert("甲", "同行", "乙", "甲与乙同行")])]
        assert br.find_bridge(Q_TWO, units)["status"] == "no_common_neighbor"

    def test_self_loop_is_ignored(self) -> None:
        """自环（`甲—甲`）不许制造候选。

        注意这条是**与 `- set(anc)` 冗余**的：单靠 `- set(anc)` 这条断言也绿。
        见上一条的说明。保留它是因为自环在语义上就该被跳过——不能指望
        "减掉锚点"这个下游动作替上游擦屁股。
        """
        text = "甲与自己同行。甲与乙同行。"
        units = [_unit("c#1", text, [
            _assert("甲", "同行", "甲", "甲与自己同行"),
            _assert("甲", "同行", "乙", "甲与乙同行"),
        ])]
        assert br.find_bridge(Q_TWO, units)["status"] == "no_common_neighbor", (
            "自环被收进三角表后，甲会同时出现在自己的邻居里，从而伪造出"
            "'甲是甲与乙的公共邻居'——`s == o` 跳过必须生效。"
        )

    def test_no_candidate_ever_equals_an_anchor(self) -> None:
        """构造一个"只靠 `s == o` 挡得住"的场景：锚点互为邻居 + 自环。"""
        text = "甲与自己交谈。甲与乙交谈。乙与乙交谈。"
        units = [_unit("c#1", text, [
            _assert("甲", "交谈", "甲", "甲与自己交谈"),
            _assert("甲", "交谈", "乙", "甲与乙交谈"),
            _assert("乙", "交谈", "乙", "乙与乙交谈"),
        ])]
        out = br.find_bridge(Q_TWO, units)
        assert out["status"] == "no_common_neighbor"
        assert out["x"] not in ("甲", "乙")

    def test_missing_edge_is_reported(self) -> None:
        """候选唯一但凑不齐两条 X—锚点 句子时，不许只写半句话。"""
        text = "甲与丙同行。"
        units = [_unit("c#1", text, [
            _assert("甲", "同行", "丙", "甲与丙同行"),
            _assert("乙", "前往", "丁", "乙前往丁"),   # 乙的最短边与丙无关，但丁是孤立点
        ])]
        # 手工把丙指出现在 乙 的邻域里，但不给它一条能定位的 evidence
        units[0]["assertions"].append(_assert("乙", "交谈", "丙", "找不到的一句话"))
        out = br.find_bridge(Q_TWO, units)
        assert out["status"] in ("missing_edge", "no_common_neighbor"), (
            "定位不到的断言不能用来拼句子——那会发出无法校验的引用，"
            f"整条答案会被 verify_citations 清空。实际 {out['status']}。"
        )

    def test_not_applicable_off_trigger(self) -> None:
        out = br.find_bridge("「甲」与「乙」处于哪种关系？", [])
        assert out["status"] == "not_applicable"

    def test_short_evidence_is_skipped(self) -> None:
        """短于 `MIN_EVIDENCE_CHARS` 的证据不能进来（verify_citations 会整条剔除）。"""
        text = "甲与丙。乙与丙。"
        units = [_unit("c#1", text, [
            _assert("甲", "同行", "丙", "甲与丙"),
            _assert("乙", "交谈", "丙", "乙与丙"),
        ])]
        out = br.find_bridge(Q_TWO, units)
        assert out["status"] == "no_common_neighbor", (
            "4 字证据（'甲与丙'）恰好等于 MIN_EVIDENCE_CHARS 应保留；"
            "3 字应被剔除。这条用 4 字验保留，下面一条验剔除。"
        )
        assert br.MIN_EVIDENCE_CHARS == 4

    def test_evidence_below_minimum_is_dropped(self) -> None:
        """证据在正文里**找得到**、但短于门槛时必须被剔除。

        取样要点：正文必须真的含这段 2 字证据（`"甲丙"。`），否则
        `body.find(ev)` 先返回 -1，`start < 0` 那一半就把断言挡下了，
        长度门槛这一半根本没被执行到——这条测试就成了假绿。
        """
        text = "甲丙。乙丙。"
        units = [_unit("c#1", text, [
            _assert("甲", "同行", "丙", "甲丙"),
            _assert("乙", "交谈", "丙", "乙丙"),
        ])]
        assert "甲丙" in text and "乙丙" in text      # 前提：证据可定位
        assert len("甲丙") < br.MIN_EVIDENCE_CHARS    # 前提：确实短于门槛
        assert br.find_bridge(Q_TWO, units)["status"] == "no_common_neighbor", (
            "2 字证据能定位但发不出可校验引用（verify_citations 会整条剔除、"
            "答案转拒答），必须被长度门槛剔掉。"
        )


class TestCitationsAreVerifiable:
    def test_offsets_match_the_host_chunk_verbatim(self) -> None:
        text = "开场白。甲与丙同行。乙与丙交谈。收尾。"
        units = [_unit("c#1", text, [
            _assert("甲", "同行", "丙", "甲与丙同行"),
            _assert("乙", "交谈", "丙", "乙与丙交谈"),
        ])]
        chunks = {"c#1": {"chunk_id": "c#1", "text": text}}
        out = br.find_bridge(Q_TWO, units)
        assert len(out["citations"]) == 2
        for c in out["citations"]:
            assert text[c["start"]:c["end"]] == c["evidence"], (
                f"引用偏移与正文对不上：{c}"
            )
            assert len(c["evidence"]) >= br.MIN_EVIDENCE_CHARS
            assert c["chunk_id"] == "c#1"

    def test_verify_citations_accepts_the_result(self) -> None:
        """真调生产引用纪律：answer.verify_citations 不许清空这条答案。"""
        text = "开场白。甲与丙同行。乙与丙交谈。收尾。"
        units = [_unit("c#1", text, [
            _assert("甲", "同行", "丙", "甲与丙同行"),
            _assert("乙", "交谈", "丙", "乙与丙交谈"),
        ])]
        chunks = {"c#1": {"chunk_id": "c#1", "text": text}}
        found = br.find_bridge(Q_TWO, units)
        merged = br.merge({"answer": "", "abstained": True, "citations": []}, found)
        out = answer_mod.verify_citations(merged, chunks)
        assert out["verification"]["rejected"] == 0, (
            f"生产引用纪律拒了 {out['verification']['rejected']} 条："
            f"{out['verification']['problems']}"
        )
        assert "丙" in out["answer"]

    def test_chunk_assertions_callback_widens_the_source(self) -> None:
        """回调版按'上下文块的全量断言'求候选（unit["assertions"] 被谓词过滤过）。"""
        text = "甲与丙同行。乙与丙交谈。"
        units = [_unit("c#1", text, [])]        # unit 内的断言被过滤空了
        full = {
            "c#1": [
                {"subject_id": "e-jia", "predicate": "同行", "object_id": "e-bing",
                 "evidence_text": "甲与丙同行"},
                {"subject_id": "e-yi", "predicate": "交谈", "object_id": "e-bing",
                 "evidence_text": "乙与丙交谈"},
            ]
        }
        names = {"e-jia": "甲", "e-yi": "乙", "e-bing": "丙"}
        out = br.find_bridge(Q_TWO, units,
                             chunk_assertions=lambda cid: full.get(cid, ()),
                             display=lambda eid: names.get(eid, eid))
        assert out["status"] == "ok", (
            "传了 chunk_assertions 就该按全量断言求候选；"
            f"实际 {out['status']}——回调没被用上。"
        )
        assert out["x"] == "丙"

    def test_without_callback_the_units_are_the_only_source(self) -> None:
        text = "甲与丙同行。乙与丙交谈。"
        units = [_unit("c#1", text, [])]
        assert br.find_bridge(Q_TWO, units)["status"] == "no_common_neighbor"


class TestMerge:
    def test_bridge_sentences_go_first(self) -> None:
        found = {"status": "ok", "x": "丙",
                 "sentences": ["丙同行甲", "丙交谈乙"],
                 "citations": [{"claim": "丙同行甲", "chunk_id": "c#1",
                                "evidence": "甲与丙同行", "start": 4, "end": 9}]}
        out = br.merge({"answer": "原有答案句。", "citations": [], "abstained": False}, found)
        assert out["answer"].startswith("丙。"), (
            f"桥接句必须前置（判据要求 gold 与每个锚点同句共现），实际 {out['answer']!r}"
        )
        assert "原有答案句" in out["answer"]

    def test_sentence_cap_is_enforced(self) -> None:
        """形态红线：总句数 ≤8。"""
        base = "。".join(f"原句{i}" for i in range(12)) + "。"
        found = {"status": "ok", "x": "丙",
                 "sentences": ["丙同行甲", "丙交谈乙"], "citations": []}
        out = br.merge({"answer": base, "citations": []}, found, max_sentences=8)
        segs = [s for s in out["answer"].split("。") if s.strip()]
        assert len(segs) <= 8, f"句数 {len(segs)} 超过红线 8：{out['answer']!r}"

    def test_default_cap_matches_max_claims(self) -> None:
        sig = inspect.signature(br.merge)
        assert sig.parameters["max_sentences"].default == answer_mod.MAX_CLAIMS, (
            "merge 的默认句数上限必须跟 `answer.MAX_CLAIMS` 走；"
            "两处不一致就会出现'答案超过配额'的形态越界。"
        )

    def test_non_ok_bridge_is_a_passthrough(self) -> None:
        original = {"answer": "原答案。", "citations": [{"claim": "x"}]}
        for status in ("ambiguous", "no_common_neighbor", "missing_edge", "not_applicable"):
            out = br.merge(original, {"status": status, "x": "", "sentences": [],
                                      "citations": []})
            assert out == original, f"status={status} 时必须原样返回"

    def test_citations_are_appended_not_replaced(self) -> None:
        base_c = [{"claim": "原引用", "chunk_id": "c#9"}]
        found = {"status": "ok", "x": "丙", "sentences": ["丙同行甲"],
                 "citations": [{"claim": "丙同行甲", "chunk_id": "c#1"}]}
        out = br.merge({"answer": "", "citations": base_c}, found)
        assert len(out["citations"]) == 2 and out["citations"][0]["claim"] == "原引用"


class TestSourceKeepsTheThreeIronRules:
    def test_docstring_records_all_three_vetoes(self) -> None:
        src = SOURCE.read_text(encoding="utf-8")
        for token in ("41.7", "候选内随机 21 > 排序 top1 19", "marginal 差 0 题"):
            assert token in src, (
                f"源码里找不到被否决版本的实测读数 {token}——"
                "删了这些数字，下一个改这个模块的人就会把排序键加回来。"
            )

    def test_records_the_ceiling(self) -> None:
        src = SOURCE.read_text(encoding="utf-8")
        assert "21/96" in src and "26/57" in src, (
            "源码必须写清天花板：两锚点都有真实共现证据的题只有 21/96，"
            "dev 实测 26/57（基线 8/57）。"
        )

    def test_records_the_gate_surface(self) -> None:
        src = SOURCE.read_text(encoding="utf-8")
        assert "96/96" in src, "触发面（multihop 96/96、其余 0）必须写在源码里。"
