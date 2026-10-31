"""逐观察断言与聚合边（计划 E1 / E4）。

测试的重点不是"函数能跑"，而是三件会静默丢事实的事：
证据只留第一处、重复 id 被覆盖、聚合计功与断言数对不上。
"""
from __future__ import annotations

import copy

import pytest

from arkrag import assertion, config, contracts, norm_relation, vocab

# --------------------------------------------------------------------- 夹具


CHUNK_TEXT = (
    "凯尔希：博士，罗德岛不会放弃任何一个人。\n"
    "阿米娅：我明白了。\n"
    "凯尔希：罗德岛不会放弃任何一个人。"
)


def _chunk(chunk_id="doc_a#000", text=CHUNK_TEXT, **over):
    rec = {
        "chunk_id": chunk_id,
        "doc_id": chunk_id.split("#")[0],
        "scene_index": 0,
        "title": "测试标题",
        "chapter": "测试章节",
        "text": text,
    }
    rec.update(over)
    return rec


def _record(chunk_id, relations, entities=None):
    return {"chunk_id": chunk_id, "raw": {"entities": entities or [], "relations": relations}}


def _rel(source, relation, target, evidence, confidence=0.9):
    return {
        "source": source,
        "relation": relation,
        "target": target,
        "evidence": evidence,
        "confidence": confidence,
    }


IDENTITY = {"凯尔希": "凯尔希", "罗德岛": "罗德岛", "阿米娅": "阿米娅"}


def _build(relations, *, chunks=None, ident=None, **kw):
    chunks = chunks or [_chunk()]
    records = [_record(chunks[0]["chunk_id"], relations)]
    return assertion.build_assertions(
        chunks, records, entity_alias_map=IDENTITY if ident is None else ident, **kw
    )


# ------------------------------------------------------------------- id 身份


class TestAssertionId:
    def test_same_input_same_id(self):
        rels = [_rel("凯尔希", "隶属", "罗德岛", "罗德岛不会放弃任何一个人")]
        a, _ = _build(rels)
        b, _ = _build(rels)
        assert [x["assertion_id"] for x in a] == [x["assertion_id"] for x in b]

    def test_id_format(self):
        a, _ = _build([_rel("凯尔希", "隶属", "罗德岛", "罗德岛不会放弃任何一个人")])
        assert a[0]["assertion_id"].startswith("a-")
        assert len(a[0]["assertion_id"]) == 18

    def test_confidence_does_not_change_id(self):
        """契约 §1.1：非身份字段不许进哈希。置信度改一个事实的 id 会让回滚比对全废。"""
        low, _ = _build([_rel("凯尔希", "隶属", "罗德岛", "罗德岛不会放弃任何一个人", 0.6)])
        high, _ = _build([_rel("凯尔希", "隶属", "罗德岛", "罗德岛不会放弃任何一个人", 0.95)])
        assert low[0]["assertion_id"] == high[0]["assertion_id"]

    def test_evidence_start_changes_id(self):
        """同一句话在块里出现两次 → 两条独立断言，id 必须不同。
        这正是聚合边做不到的事：它只能留一条。"""
        text = "罗德岛不会放弃任何一个人。\n中间隔一句。\n罗德岛不会放弃任何一个人。"
        chunks = [_chunk(text=text)]
        # 两处出现：第一次 start=0，第二次在末尾
        rels = [_rel("凯尔希", "隶属", "罗德岛", "罗德岛不会放弃任何一个人")]
        got, stats = assertion.build_assertions(
            chunks, [_record(chunks[0]["chunk_id"], rels)], entity_alias_map=IDENTITY
        )
        assert len(got) == 1
        assert len(got[0]["evidence_all"]) == 2
        # 手工构造两条不同 start 的断言，id 必须不同
        id1 = assertion.assertion_id(
            chunk_id="c#000",
            subject_id="凯尔希",
            predicate="隶属",
            object_id="罗德岛",
            evidence_start=0,
            raw_relation="隶属",
        )
        id2 = assertion.assertion_id(
            chunk_id="c#000",
            subject_id="凯尔希",
            predicate="隶属",
            object_id="罗德岛",
            evidence_start=880,
            raw_relation="隶属",
        )
        assert id1 != id2

    def test_duplicate_id_is_reported_not_overwritten(self):
        """旧代码 `got[cid] = rec` 让后一条静默覆盖前一条。这里必须报出来。"""
        chunk = _chunk()
        rel = _rel("凯尔希", "隶属", "罗德岛", "罗德岛不会放弃任何一个人")
        records = [
            _record(chunk["chunk_id"], [rel]),
            _record(chunk["chunk_id"], [dict(rel)]),
        ]
        got, stats = assertion.build_assertions([chunk], records, entity_alias_map=IDENTITY)
        assert len(got) == 1, "重复观察不得变成两条"
        assert stats["duplicate_assertion_id"] == 1
        assert stats["duplicate_assertion_ids"], "必须留下可排查的样本"


# ------------------------------------------------------------------- 证据定位


class TestEvidence:
    def test_located_normally(self):
        got, _ = _build([_rel("凯尔希", "隶属", "罗德岛", "罗德岛不会放弃任何一个人")])
        a = got[0]
        assert CHUNK_TEXT[a["evidence_start"] : a["evidence_end"]] == "罗德岛不会放弃任何一个人"

    def test_all_occurrences_collected(self):
        """块内出现两次时 evidence_all 必须有两条，且两个偏移都能对上。"""
        got, stats = _build([_rel("凯尔希", "隶属", "罗德岛", "罗德岛不会放弃任何一个人")])
        a = got[0]
        assert len(a["evidence_all"]) == 2, a["evidence_all"]
        for occ in a["evidence_all"]:
            assert CHUNK_TEXT[occ["start"] : occ["end"]] == "罗德岛不会放弃任何一个人"
        assert stats["evidence_multiple_occurrences"] == 1

    def test_not_found_is_dropped(self):
        got, stats = _build([_rel("凯尔希", "隶属", "罗德岛", "这句话根本不在块里")])
        assert got == []
        assert stats["dropped:evidence_not_found"] == 1
        assert stats["samples"]["evidence_not_found"]

    def test_evidence_too_long_rejected(self):
        long_ev = "凯" * (contracts.EVIDENCE_MAX_CHARS + 1)
        got, stats = _build([_rel("凯尔希", "隶属", "罗德岛", long_ev)])
        assert got == []
        assert stats["dropped:evidence_invalid"] == 1

    def test_empty_evidence_rejected(self):
        got, stats = _build([_rel("凯尔希", "隶属", "罗德岛", "   ")])
        assert got == []
        assert stats["dropped:evidence_invalid"] == 1

    def test_evidence_from_another_chunk_rejected(self):
        """A 块的证据拿到 B 块定位时必须失败——不许拿别的块的证据顶替。"""
        chunk_a = _chunk("doc_a#000", text="这里只有关于龙门的一句话。")
        chunk_b = _chunk("doc_b#000", text="罗德岛不会放弃任何一个人。")
        records = [
            _record(chunk_a["chunk_id"], [_rel("凯尔希", "隶属", "罗德岛", "罗德岛不会放弃任何一个人")])
        ]
        got, stats = assertion.build_assertions(
            [chunk_a, chunk_b], records, entity_alias_map=IDENTITY
        )
        assert got == []
        assert stats["dropped:evidence_not_found"] == 1


# ----------------------------------------------------------------- 方向与场景


class TestDirectionAndScene:
    def test_direction_is_always_out(self):
        got, _ = _build([_rel("凯尔希", "隶属", "罗德岛", "罗德岛不会放弃任何一个人")])
        assert got[0]["direction"] == "out"

    def test_scene_id_derived_from_chunk_id(self):
        got, _ = _build([_rel("凯尔希", "隶属", "罗德岛", "罗德岛不会放弃任何一个人")])
        assert got[0]["scene_id"] == "doc_a"
        assert got[0]["doc_id"] == "doc_a"

    def test_adjudicated_drop_is_counted(self):
        """否定关系必须被丢弃（norm_relation 的第 3 步），且计数带 flag。"""
        got, stats = _build([_rel("凯尔希", "不隶属", "罗德岛", "罗德岛不会放弃任何一个人")])
        assert got == []
        assert stats["dropped:adjudicated:negation"] == 1

    def test_swap_direction_swaps_endpoints(self):
        """`dec.swap` 为真时必须换端点：被指导者不能变成指导者。

        注意：当前词表下 `norm_relation.adjudicate` **永远不会**返回 swap=True
        （实测穷举证明 `SWAP_MAP` 是不可达死分支，原因见
        `test_swap_map_is_unreachable_with_current_vocab`）。
        所以这里直接构造 Decision 来测本模块自己的换端点逻辑——
        它是一层防御，不能因为上游暂时不触发就删掉，也不能指望上游测试覆盖它。
        """
        ident = {"凯尔希": "E-K", "阿米娅": "E-A"}
        text = "阿米娅：我跟凯尔希学习。"

        original = norm_relation.adjudicate

        def fake(raw, source="", target=""):
            dec = norm_relation.Decision(raw, "指导", "adopt", "synonym_exact")
            dec.swap = True
            return dec

        norm_relation.adjudicate = fake
        try:
            got, stats = assertion.build_assertions(
                [_chunk("doc_a#000", text=text)],
                [_record("doc_a#000", [_rel("阿米娅", "师从", "凯尔希", "我跟凯尔希学习")])],
                entity_alias_map=ident,
            )
        finally:
            norm_relation.adjudicate = original

        assert got, stats
        a = got[0]
        assert a["predicate"] == "指导"
        assert a["subject_id"] == "E-K", "指导者应是凯尔希"
        assert a["object_id"] == "E-A", "被指导者应是阿米娅"
        assert stats["swapped_direction"] == 1

    def test_swap_map_is_unreachable_with_current_vocab(self):
        """记录一个实测事实：`norm_relation.SWAP_MAP` 当前不可达。

        三个键各有原因，且都不是"差一点就能触发"：

        1. `师从` 本身就在 `CANONICAL_RELATIONS` 里 → `adjudicate` 走 exact 分支直接 return，
           根本不会调到 `_direction_guard`。
        2. `受业于` 经 `RELATION_SYNONYMS` 归到 `师从`，而 `SWAP_MAP['受业于']` 写的是 `指导`，
           两者不相等 → `swap_needed` 恒为 False。
        3. `是……的学生` 含省略号，被 `RE_PUNCT` 命中 → 走 keep_flagged 直接 return。

        这不是本模块的 bug（`师从`/`指导` 本来就是 `INVERSE_PAIRS` 的一对，
        归一到自身是正确的），但它意味着 **`_direction_guard` 从未被执行过**。
        把它写成断言而不是留在注释里：将来有人改词表让分支可达时，这条会红，
        提醒他"这个分支终于活了，得补真正的测试"。
        """
        reachable = []
        for raw in (
            list(vocab.CANONICAL_RELATIONS)
            + list(vocab.RELATION_SYNONYMS)
            + list(norm_relation.SWAP_MAP)
        ):
            if norm_relation.adjudicate(raw, "A", "B").swap:
                reachable.append(raw)
        assert reachable == [], (
            "SWAP_MAP 变成可达了：请删掉这条断言并给换端点分支补真实测试。"
            f" 可触发输入={reachable}"
        )

    def test_endpoint_unknown_is_skipped(self):
        """映射里没有的名字意味着未消歧，不许拿原始名冒充 id。"""
        got, stats = _build(
            [_rel("凯尔希", "隶属", "罗德岛", "罗德岛不会放弃任何一个人")],
            ident={"凯尔希": "凯尔希"},
        )
        assert got == []
        assert stats["dropped:endpoint_unknown"] == 1

    def test_low_confidence_dropped(self):
        low = config.CONFIDENCE_THRESHOLD - 0.1
        got, stats = _build(
            [_rel("凯尔希", "隶属", "罗德岛", "罗德岛不会放弃任何一个人", low)]
        )
        assert got == []
        assert stats["dropped:low_confidence"] == 1

    def test_self_loop_dropped(self):
        got, stats = _build([_rel("凯尔希", "隶属", "凯尔希", "罗德岛不会放弃任何一个人")])
        assert got == []
        assert stats["dropped:self_loop"] == 1


# --------------------------------------------------------------------- 聚合


class TestAggregate:
    def _three(self):
        """三条观察、两条同边。

        这里的证据文本**必须互不相同**：`assertion_id` 含 `evidence_start`，
        两条同边观察若引用同一句话，会算出同一个 id 并被"重复 id"去重掉——
        那是夹具错误，不是实现错误（想让两条观察独立，就得让它们的证据位置不同）。
        """
        text = (
            "凯尔希是罗德岛的医疗主管。"
            "阿米娅在罗德岛长大。"
            "凯尔希与阿米娅都留在罗德岛。"
        )
        # 顺序刻意让低置信度那条在前：如果"代表证据"实现成取 items[0]，
        # 这条夹具就会暴露它（原来 0.8 在前，取第一条与取最可信碰巧同解）。
        rels = [
            _rel("凯尔希", "隶属", "罗德岛", "凯尔希与阿米娅都留在罗德岛", 0.7),
            _rel("凯尔希", "隶属", "罗德岛", "凯尔希是罗德岛的医疗主管", 0.8),
            _rel("阿米娅", "隶属", "罗德岛", "阿米娅在罗德岛长大", 0.9),
        ]
        return _build(rels, chunks=[_chunk(text=text)])

    def test_support_count_matches_ids(self):
        got, _ = self._three()
        edges = assertion.aggregate(got)
        for e in edges:
            assert e["support_count"] == len(e["assertion_ids"])

    def test_edge_groups_by_triple(self):
        got, _ = self._three()
        edges = assertion.aggregate(got)
        # 凯尔希-隶属-罗德岛 有 2 条，阿米娅-隶属-罗德岛 有 1 条
        assert len(edges) == 2
        top = edges[0]
        assert top["support_count"] == 2
        assert top["subject_id"] == "凯尔希"

    def test_confidence_summary(self):
        got, _ = self._three()
        edges = assertion.aggregate(got)
        top = edges[0]
        assert top["confidence_summary"]["min"] == pytest.approx(0.7)
        assert top["confidence_summary"]["max"] == pytest.approx(0.8)
        assert top["confidence_summary"]["mean"] == pytest.approx(0.75)

    def test_independent_scene_count(self):
        text = "罗德岛不会放弃任何一个人。"
        chunks = [
            _chunk("doc_a#000", text=text),
            _chunk("doc_b#000", text=text),
        ]
        records = [
            _record("doc_a#000", [_rel("凯尔希", "隶属", "罗德岛", text)]),
            _record("doc_b#000", [_rel("凯尔希", "隶属", "罗德岛", text)]),
        ]
        got, _ = assertion.build_assertions(chunks, records, entity_alias_map=IDENTITY)
        edges = assertion.aggregate(got)
        assert edges[0]["support_count"] == 2
        assert edges[0]["independent_scene_count"] == 2

    def test_independent_scene_count_is_not_support_count(self):
        """同一场景内两条观察：support_count=2 但 independent_scene_count 必须是 1。

        没有这条，"独立场景数"写成 `len(items)` 也照样过测试——
        而那个数字正是用来区分"两个人都这么说"和"同一段话说了两遍"的。
        """
        text = "凯尔希是罗德岛的医疗主管。凯尔希仍留在罗德岛。"
        chunk = _chunk("doc_a#000", text=text)
        records = [
            _record(
                "doc_a#000",
                [
                    _rel("凯尔希", "隶属", "罗德岛", "凯尔希是罗德岛的医疗主管"),
                    _rel("凯尔希", "隶属", "罗德岛", "凯尔希仍留在罗德岛"),
                ],
            )
        ]
        got, _ = assertion.build_assertions([chunk], records, entity_alias_map=IDENTITY)
        edges = assertion.aggregate(got)
        assert edges[0]["support_count"] == 2
        assert edges[0]["independent_scene_count"] == 1, "同一个场景不能算两票"

    def test_edge_evidence_is_traceable_to_an_assertion(self):
        """聚合边的 evidence 必须是同边某条真实断言的证据（L5.10 查的就是这件事）。"""
        got, _ = self._three()
        edges = assertion.aggregate(got)
        by_id = {a["assertion_id"]: a for a in got}
        for e in edges:
            assert e["evidence_assertion_id"] in e["assertion_ids"]
            assert by_id[e["evidence_assertion_id"]]["evidence_text"] == e["evidence"]

    def test_edge_evidence_comes_from_most_confident_assertion(self):
        """代表证据取最可信的那条。取 items[0] 也"可回溯"，但那是碰运气——
        观察顺序由 raw 文件行序决定，跟可信度无关。"""
        got, _ = self._three()
        edges = assertion.aggregate(got)
        top = edges[0]
        by_id = {a["assertion_id"]: a for a in got}
        lead = by_id[top["evidence_assertion_id"]]
        assert lead["confidence"] == pytest.approx(top["confidence_summary"]["max"])
        assert top["evidence"] == "凯尔希是罗德岛的医疗主管"

    def test_conflict_count_counts_mutually_exclusive_assertions(self):
        """`conflict_count` 是"这条边在原文里被别的观察反驳了几次"。

        没有被测到时写成恒 0 也照样绿——而那正是最容易被悄悄删掉的信息。
        """
        text = "凯尔希与罗德岛结盟。凯尔希与罗德岛反目。"
        chunk = _chunk("doc_a#000", text=text)
        records = [
            _record(
                "doc_a#000",
                [
                    _rel("凯尔希", "同盟", "罗德岛", "凯尔希与罗德岛结盟"),
                    _rel("凯尔希", "敌对", "罗德岛", "凯尔希与罗德岛反目"),
                ],
            )
        ]
        got, _ = assertion.build_assertions([chunk], records, entity_alias_map=IDENTITY)
        edges = assertion.aggregate(got)
        by_pred = {e["predicate"]: e for e in edges}
        assert "同盟" in by_pred and "敌对" in by_pred
        # 两条边各自都被对端的互斥谓词反驳了一次
        assert by_pred["同盟"]["conflict_count"] == 1
        assert by_pred["敌对"]["conflict_count"] == 1

    def test_conflict_count_zero_without_exclusive_predicate(self):
        """没有互斥谓词时必须真的是 0，不能恒返回 1 之类。"""
        got, _ = _build([_rel("凯尔希", "隶属", "罗德岛", "罗德岛不会放弃任何一个人")])
        edges = assertion.aggregate(got)
        assert edges[0]["conflict_count"] == 0

    def test_edge_id_stable_regardless_of_support_count(self):
        text = "罗德岛不会放弃任何一个人。"
        one, _ = _build([_rel("凯尔希", "隶属", "罗德岛", text)])
        two, _ = _build(
            [
                _rel("凯尔希", "隶属", "罗德岛", text, 0.7),
                _rel("凯尔希", "隶属", "罗德岛", text, 0.9),
            ]
        )
        assert assertion.aggregate(one)[0]["edge_id"] == assertion.aggregate(two)[0]["edge_id"]


# --------------------------------------------------------------------- 守恒


class TestConservation:
    def test_ok_when_complete(self):
        got, _ = _build([_rel("凯尔希", "隶属", "罗德岛", "罗德岛不会放弃任何一个人")])
        edges = assertion.aggregate(got)
        report = assertion.observation_conservation(got, edges)
        assert report["ok"] is True
        assert report["assertions"] == 1
        assert report["support_sum"] == 1

    def test_detects_unreferenced(self):
        """人为丢一条断言 → 守恒必须报出来（这是"静默丢事实"的检测器）。"""
        got, _ = _build(
            [
                _rel("凯尔希", "隶属", "罗德岛", "罗德岛不会放弃任何一个人"),
                _rel("阿米娅", "隶属", "罗德岛", "罗德岛不会放弃任何一个人"),
            ]
        )
        edges = assertion.aggregate(got)
        report = assertion.observation_conservation(got[:-1] + [], edges)
        # got[:-1] 少一条断言，边仍引用两条 → 出现未知引用
        assert report["ok"] is False
        assert report["unknown_references"]

    def test_detects_double_reference(self):
        got, _ = _build([_rel("凯尔希", "隶属", "罗德岛", "罗德岛不会放弃任何一个人")])
        edges = assertion.aggregate(got)
        edges.append(copy.deepcopy(edges[0]))
        edges[1]["edge_id"] = "r-different"
        report = assertion.observation_conservation(got, edges)
        assert report["ok"] is False
        assert report["multiply_referenced"]

    def test_detects_support_mismatch(self):
        got, _ = _build([_rel("凯尔希", "隶属", "罗德岛", "罗德岛不会放弃任何一个人")])
        edges = assertion.aggregate(got)
        edges[0]["support_count"] = 5
        report = assertion.observation_conservation(got, edges)
        assert report["ok"] is False
        assert report["support_mismatch"]

    def test_detects_support_mismatch_when_total_still_balances(self):
        """总和能对上、但每条的计数错了，也必须被抓住。

        上一条测试把 support_count 从 1 改成 5，顺带把 support_sum 也改了，
        于是"总和等于断言数"这条先命中——`support_mismatch` 单独那一项等于没被测到。
        这里把两条边的计数互换：总和不变，只有逐边核对能发现。
        """
        got, _ = _build(
            [
                _rel("凯尔希", "隶属", "罗德岛", "凯尔希是罗德岛的医疗主管"),
                _rel("阿米娅", "隶属", "罗德岛", "阿米娅在罗德岛长大"),
                _rel("阿米娅", "同行", "罗德岛", "阿米娅在罗德岛长大"),
            ],
            chunks=[
                _chunk(
                    text="凯尔希是罗德岛的医疗主管。阿米娅在罗德岛长大。"
                )
            ],
        )
        edges = assertion.aggregate(got)
        assert len(edges) == 3
        # 三条边各 1 条断言；把前两条的 support_count 互换不改变总和
        counts = [e["support_count"] for e in edges]
        assert counts == [1, 1, 1], counts
        edges[0]["support_count"], edges[1]["support_count"] = 2, 0
        report = assertion.observation_conservation(got, edges)
        assert report["support_sum"] == len(got), "总和仍然对得上"
        assert report["ok"] is False, "总和对了不代表每条都对"
        assert len(report["support_mismatch"]) == 2


# ----------------------------------------------------------------- 限定字段


class TestQualifiers:
    def test_defaults_filled(self):
        got, _ = _build([_rel("凯尔希", "隶属", "罗德岛", "罗德岛不会放弃任何一个人")])
        qual = got[0]["qualifiers"]
        assert set(qual) == set(assertion.QUALIFIER_DEFAULTS)
        assert qual["certainty"] == "asserted"
        assert qual["modality"] == "actual"
        assert qual["is_flashback"] is False
        assert qual["story_time"] == ""

    def test_scene_order_from_chunk(self):
        got, _ = _build([_rel("凯尔希", "隶属", "罗德岛", "罗德岛不会放弃任何一个人")])
        assert got[0]["qualifiers"]["scene_order"] == 0

    def test_override_applied(self):
        got, _ = _build(
            [_rel("凯尔希", "隶属", "罗德岛", "罗德岛不会放弃任何一个人")],
            qualifiers_override={"is_flashback": True, "certainty": "hedged"},
        )
        assert got[0]["qualifiers"]["is_flashback"] is True
        assert got[0]["qualifiers"]["certainty"] == "hedged"

    def test_unknown_override_key_raises(self):
        with pytest.raises(ValueError):
            assertion.default_qualifiers(not_a_field=1)

    def test_validate_type_errors(self):
        problems = assertion.validate_qualifiers(
            {
                **assertion.default_qualifiers(),
                "is_flashback": "yes",
                "scene_order": "0",
                "certainty": "maybe",
            }
        )
        assert len(problems) == 3
        assert any("is_flashback" in p for p in problems)
        assert any("scene_order" in p for p in problems)
        assert any("certainty" in p for p in problems)

    def test_validate_missing_and_unknown(self):
        problems = assertion.validate_qualifiers({"certainty": "asserted", "bogus": 1})
        assert any("缺字段" in p for p in problems)
        assert any("未知字段" in p for p in problems)

    def test_validate_bool_is_not_int(self):
        """Python 里 True 是 int 的子类；scene_order 不能把 bool 当序数收下。"""
        problems = assertion.validate_qualifiers(
            {**assertion.default_qualifiers(), "scene_order": True}
        )
        assert any("scene_order" in p for p in problems)


# -------------------------------------------------------------- 互斥关系共存


class TestMutualExclusion:
    def _pair(self, scene_a="doc_a", scene_b="doc_a"):
        text = "我们和他们结盟。"
        chunks = [
            _chunk(f"{scene_a}#000", text=text),
            _chunk(f"{scene_b}#000", text=text),
        ]
        records = [
            _record(f"{scene_a}#000", [_rel("凯尔希", "同盟", "罗德岛", text)]),
            _record(f"{scene_b}#000", [_rel("凯尔希", "敌对", "罗德岛", text)]),
        ]
        got, _ = assertion.build_assertions(chunks, records, entity_alias_map=IDENTITY)
        return got

    def test_different_scenes_kept_both(self):
        got = self._pair("doc_a", "doc_b")
        conflicts = assertion.check_qualifier_consistency(got)
        assert len(conflicts) == 1
        assert conflicts[0]["action"] == "kept_both"
        assert conflicts[0]["same_scene"] is False
        # 两条断言都还在——互斥关系不直接删除
        assert len(got) == 2

    def test_same_scene_needs_human(self):
        text = "我们和他们结盟。"
        chunk = _chunk("doc_a#000", text=text)
        records = [
            _record(
                "doc_a#000",
                [
                    _rel("凯尔希", "同盟", "罗德岛", text),
                    _rel("凯尔希", "敌对", "罗德岛", text),
                ],
            )
        ]
        got, _ = assertion.build_assertions([chunk], records, entity_alias_map=IDENTITY)
        conflicts = assertion.check_qualifier_consistency(got)
        assert len(conflicts) == 1
        assert conflicts[0]["action"] == "needs_human"
        assert "uncertain" in conflicts[0]["note"]

    def test_no_false_positive_on_unrelated_predicates(self):
        got, _ = _build([_rel("凯尔希", "隶属", "罗德岛", "罗德岛不会放弃任何一个人")])
        assert assertion.check_qualifier_consistency(got) == []


# ------------------------------------------------------------------ 工具函数


class TestHelpers:
    def test_find_all_overlapping(self):
        """重叠出现也要都找到。

        用 "啊啊啊啊" / "啊啊" 这种真重叠的例子：位置 0、1、2 都成立。
        注意 "凯尔希凯尔希" 里 "凯尔希" 出现在 0 和 3，**首尾相接但不重叠**，
        用它当用例的话，`pos + len(needle)` 的错误步进也能通过——测不出问题。
        """
        assert assertion.find_all_occurrences("啊啊啊啊", "啊啊") == [(0, 2), (1, 3), (2, 4)]

    def test_find_all_adjacent_occurrences(self):
        """首尾相接（非重叠）也必须都找到，且不能多算。"""
        assert assertion.find_all_occurrences("凯尔希凯尔希", "凯尔希") == [(0, 3), (3, 6)]

    def test_find_all_empty_needle(self):
        assert assertion.find_all_occurrences("任意文本", "") == []

    def test_find_all_no_match(self):
        assert assertion.find_all_occurrences("任意文本", "不存在") == []

    def test_sum_stats_shape(self):
        got, stats = _build(
            [
                _rel("凯尔希", "隶属", "罗德岛", "罗德岛不会放弃任何一个人"),
                _rel("凯尔希", "不隶属", "罗德岛", "罗德岛不会放弃任何一个人"),
            ]
        )
        summary = assertion.sum_stats(stats)
        assert summary["raw_relations"] == 2
        assert summary["accepted"] == 1
        assert summary["dropped"]["adjudicated:negation"] == 1
        assert summary["dropped_total"] == 1

    def test_chunk_unknown_counted(self):
        records = [
            _record("never_seen#999", [_rel("凯尔希", "隶属", "罗德岛", "罗德岛不会放弃任何一个人")])
        ]
        got, stats = assertion.build_assertions(
            [_chunk()], records, entity_alias_map=IDENTITY
        )
        assert got == []
        assert stats["dropped:chunk_unknown"] == 1

    def test_extraction_meta_copied(self):
        meta = {"schema_version": "ark-rag-1", "prompt_sha256": "abc", "generation_id": "g-1"}
        got, _ = _build(
            [_rel("凯尔希", "隶属", "罗德岛", "罗德岛不会放弃任何一个人")],
            extraction_meta=meta,
        )
        assert got[0]["extraction"] == meta
        # 必须是副本：改一条断言的 extraction 不该影响别的
        got[0]["extraction"]["schema_version"] = "changed"
        assert meta["schema_version"] == "ark-rag-1"


# --------------------------------------------------------------- 词表一致性

class TestVocabContract:
    def test_mutual_exclusion_matches_vocab(self):
        """模块内的互斥判定必须与词表一致，不能各写一份。"""
        for a, b in vocab.MUTUALLY_EXCLUSIVE_PAIRS:
            assert assertion._mutually_exclusive(a, b)
            assert assertion._mutually_exclusive(b, a)
        assert not assertion._mutually_exclusive("隶属", "同盟")
