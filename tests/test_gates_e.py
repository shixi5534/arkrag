"""E 阶段门禁的回归测试：L5.9 / L5.10 / L5.11 / L4.12 / L4.13 / L6.8。

这些门守的是本项目最贵的一类错误——**图说得对，但说不出凭什么**。
所以每一条门都必须能被注入的坏数据打红；只断言"干净数据时 PASS"
的门等于没有门（旧 `test_negative.py` 就是这么假通过的，记在台账里）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from arkrag import adjacency, gates  # noqa: E402

TEXT = "凯尔希隶属罗德岛。阿米娅隶属罗德岛。"
# 「凯尔希隶属罗德岛」= 0..8，「阿米娅隶属罗德岛」= 9..17
EV1 = (0, 8)
EV2 = (9, 17)


def _assertion(
    aid: str = "a-1",
    *,
    chunk_id: str = "d#001",
    subject: str = "凯尔希",
    obj: str = "罗德岛",
    evidence: str = "凯尔希隶属罗德岛",
    start: int = 0,
    end: int = 8,
    all_occurrences: list[dict] | None = None,
) -> dict:
    return {
        "assertion_id": aid,
        "chunk_id": chunk_id,
        "scene_id": "d",
        "subject_id": subject,
        "predicate": "隶属",
        "object_id": obj,
        "evidence_text": evidence,
        "evidence_start": start,
        "evidence_end": end,
        "evidence_all": all_occurrences if all_occurrences is not None else [{"start": start, "end": end}],
        "confidence": 0.9,
    }


def _edge(
    eid: str = "r-1",
    *,
    assertion_ids: list[str] | None = None,
    support: int = 1,
    evidence: str = "凯尔希隶属罗德岛",
    evidence_aid: str = "a-1",
    subject: str = "凯尔希",
    obj: str = "罗德岛",
) -> dict:
    return {
        "edge_id": eid,
        "subject_id": subject,
        "predicate": "隶属",
        "object_id": obj,
        "assertion_ids": assertion_ids if assertion_ids is not None else ["a-1"],
        "support_count": support,
        "independent_scene_count": 1,
        "confidence_summary": {"min": 0.9, "max": 0.9, "mean": 0.9},
        "evidence": evidence,
        "evidence_assertion_id": evidence_aid,
    }


def _chunks(text: str = TEXT) -> list[dict]:
    return [{"chunk_id": "d#001", "text": text}]


def _two_assertions() -> list[dict]:
    return [
        _assertion("a-1", start=EV1[0], end=EV1[1], evidence="凯尔希隶属罗德岛"),
        _assertion(
            "a-2",
            subject="阿米娅",
            start=EV2[0],
            end=EV2[1],
            evidence="阿米娅隶属罗德岛",
        ),
    ]


def _two_edges() -> list[dict]:
    return [
        _edge("r-1", assertion_ids=["a-1"]),
        _edge(
            "r-2",
            assertion_ids=["a-2"],
            subject="阿米娅",
            evidence="阿米娅隶属罗德岛",
            evidence_aid="a-2",
        ),
    ]


def _gate(run: gates.GateRun, prefix: str) -> gates.Gate:
    for g in run.gates:
        if g.name.startswith(prefix):
            return g
    raise AssertionError(f"没有以 {prefix!r} 开头的门；实际有 {[g.name for g in run.gates]}")


class TestL59EvidenceLocatable:
    def test_clean_data_passes(self):
        run = gates.l5_evidence_gates({}, _two_assertions(), _two_edges(), _chunks())
        assert _gate(run, "L5.9").verdict == "PASS"

    def test_wrong_offset_fails(self):
        bad = [_assertion("a-1", start=9, end=17)]
        run = gates.l5_evidence_gates({}, bad, _two_edges(), _chunks())
        assert _gate(run, "L5.9").verdict == "FAIL"

    def test_evidence_from_another_chunk_fails(self):
        """证据文本对，但它的块里没有这段原文——必须红。"""
        asserts = [_assertion("a-1", chunk_id="d#999")]
        run = gates.l5_evidence_gates({}, asserts, _two_edges(), _chunks())
        assert _gate(run, "L5.9").verdict == "FAIL"

    def test_evidence_all_with_one_bad_offset_fails(self):
        """主区间对、但 `evidence_all` 里有一条偏移错——也要红。

        这一条特别容易漏：建 Assertion 时算对一次，`evidence_all` 却可能是
        另一套逻辑填的。只查主区间等于只查了一半。
        """
        asserts = [
            _assertion(
                "a-1",
                all_occurrences=[{"start": 0, "end": 8}, {"start": 5, "end": 13}],
            )
        ]
        run = gates.l5_evidence_gates({}, asserts, _two_edges(), _chunks())
        assert _gate(run, "L5.9").verdict == "FAIL"

    def test_multiple_occurrences_all_correct_passes(self):
        text = "凯尔希隶属罗德岛。凯尔希隶属罗德岛。"
        asserts = [
            _assertion(
                "a-1",
                all_occurrences=[{"start": 0, "end": 8}, {"start": 9, "end": 17}],
            )
        ]
        run = gates.l5_evidence_gates({}, asserts, _two_edges(), _chunks(text))
        assert _gate(run, "L5.9").verdict == "PASS"


class TestL510AggregateIsDerived:
    def test_clean_data_passes(self):
        run = gates.l5_evidence_gates({}, _two_assertions(), _two_edges(), _chunks())
        assert _gate(run, "L5.10").verdict == "PASS"

    def test_edge_without_assertion_ids_fails(self):
        """旧模型形态：边自己带一条证据、没有 assertion_ids——必须红。"""
        edges = [_edge("r-1", assertion_ids=[])]
        run = gates.l5_evidence_gates({}, _two_assertions(), edges, _chunks())
        assert _gate(run, "L5.10").verdict == "FAIL"
        assert "缺 ids 1" in _gate(run, "L5.10").measured

    def test_support_count_mismatch_fails(self):
        edges = [_edge("r-1", assertion_ids=["a-1"], support=7)]
        run = gates.l5_evidence_gates({}, _two_assertions(), edges, _chunks())
        assert _gate(run, "L5.10").verdict == "FAIL"

    def test_representative_evidence_not_from_this_edge_fails(self):
        edges = [_edge("r-1", assertion_ids=["a-1"], evidence_aid="a-2")]
        run = gates.l5_evidence_gates({}, _two_assertions(), edges, _chunks())
        assert _gate(run, "L5.10").verdict == "FAIL"

    def test_representative_evidence_text_mismatch_fails(self):
        """代表证据的 assertion_id 属于本边，但存的文本和那条断言不一致。"""
        edges = [_edge("r-1", assertion_ids=["a-1"], evidence="罗德岛隶属凯尔希")]
        run = gates.l5_evidence_gates({}, _two_assertions(), edges, _chunks())
        assert _gate(run, "L5.10").verdict == "FAIL"


class TestL511ObservationConservation:
    def test_clean_data_passes(self):
        run = gates.l5_evidence_gates({}, _two_assertions(), _two_edges(), _chunks())
        assert _gate(run, "L5.11").verdict == "PASS"

    def test_unreferenced_assertion_fails(self):
        """两条断言、只有一条边引用其中一条——另一条凭空消失。"""
        run = gates.l5_evidence_gates(
            {}, _two_assertions(), [_edge("r-1", assertion_ids=["a-1"])], _chunks()
        )
        assert _gate(run, "L5.11").verdict == "FAIL"

    def test_multiply_referenced_fails(self):
        edges = [
            _edge("r-1", assertion_ids=["a-1"]),
            _edge(
                "r-2",
                assertion_ids=["a-1", "a-2"],
                subject="阿米娅",
                evidence="阿米娅隶属罗德岛",
                evidence_aid="a-2",
            ),
        ]
        run = gates.l5_evidence_gates({}, _two_assertions(), edges, _chunks())
        assert _gate(run, "L5.11").verdict == "FAIL"

    def test_unknown_reference_fails(self):
        edges = [_edge("r-1", assertion_ids=["a-1", "a-ghost"], support=2)]
        run = gates.l5_evidence_gates(
            {}, [_assertion("a-1")], edges, _chunks()
        )
        assert _gate(run, "L5.11").verdict == "FAIL"


class TestL412EntityIdIsNotName:
    def test_name_as_id_fails(self):
        """当前生产图正是这种形态：id == name。这条门就该红。"""
        graph = {"entities": [{"id": "凯尔希", "name": "凯尔希"}]}
        assert _gate(gates.l4_identity_gates(graph), "L4.12").verdict == "FAIL"

    def test_hashed_id_passes(self):
        graph = {"entities": [{"id": "e-0123456789abcdef", "name": "凯尔希"}]}
        assert _gate(gates.l4_identity_gates(graph), "L4.12").verdict == "PASS"

    def test_id_containing_name_fails(self):
        """id 里嵌名字也不行——改展示名仍会牵动身份。"""
        graph = {"entities": [{"id": "role:凯尔希", "name": "凯尔希"}]}
        assert _gate(gates.l4_identity_gates(graph), "L4.12").verdict == "FAIL"


class TestL413MergeAudit:
    def test_no_merge_is_pass(self):
        graph = {"entities": [{"id": "e-1", "name": "A", "merged_from": []}], "stats": {}}
        assert _gate(gates.l4_identity_gates(graph), "L4.13").verdict == "PASS"

    def test_merged_with_rule_counts_passes(self):
        graph = {
            "entities": [{"id": "e-1", "name": "A", "merged_from": ["B"]}],
            "stats": {"merge": {"alias_merged": 1}},
        }
        assert _gate(gates.l4_identity_gates(graph), "L4.13").verdict == "PASS"

    def test_merged_without_rule_counts_fails(self):
        """并了但没记按哪条规则并的——事后无法复核是否误并。"""
        graph = {
            "entities": [{"id": "e-1", "name": "A", "merged_from": ["B"]}],
            "stats": {},
        }
        assert _gate(gates.l4_identity_gates(graph), "L4.13").verdict == "FAIL"


class TestL64Direction:
    def _edges(self) -> list[dict]:
        return [
            _edge("r-1", subject="凯尔希", obj="罗德岛"),
            _edge("r-2", subject="凯尔希", obj="阿米娅") | {"predicate": "同盟"},
        ]

    def test_clean_adjacency_passes(self):
        edges = self._edges()
        adj = adjacency.build_directed_adjacency(edges)
        run = gates.l6_direction_gates(adj, edges)
        assert _gate(run, "L6.8").verdict == "PASS"

    def test_masqueraded_direction_fails(self):
        """入边塞进出边列表，并且把 direction 也改成 out——模拟旧 build_adjacency。

        这条用例最初是**通过的**（门没红），暴露了 L6.8 的真实缺口：
        只查"方向标记与所在列表一致"和"反向不沿用原谓词"，
        对"整体搬进 out 列表并改标记"这种形态无能为力。
        补上"两端落位"判据后才会红。记在这里，免得日后有人把那条判据当成冗余删掉。
        """
        edges = self._edges()
        adj = adjacency.build_directed_adjacency(edges)
        adj["罗德岛"]["out"].append(dict(adj["罗德岛"]["in"][0], direction="out"))
        adj["罗德岛"]["in"] = []
        run = gates.l6_direction_gates(adj, edges)
        assert _gate(run, "L6.8").verdict == "FAIL"
        assert "落位错误" in _gate(run, "L6.8").threshold

    def test_missing_subject_out_edge_fails(self):
        """主语侧出边被删掉，只留宾语侧入边——半张图，也算方向错误。"""
        edges = self._edges()
        adj = adjacency.build_directed_adjacency(edges)
        adj["凯尔希"]["out"] = [
            it for it in adj["凯尔希"]["out"] if it["edge_id"] != "r-1"
        ]
        run = gates.l6_direction_gates(adj, edges)
        assert _gate(run, "L6.8").verdict == "FAIL"

    def test_symmetric_edge_in_in_list_fails(self):
        """对称边被塞进入边列表——会让人以为"有个人指向我"。"""
        edges = self._edges()
        adj = adjacency.build_directed_adjacency(edges)
        sym = [it for it in adj["凯尔希"]["out"] if it["edge_id"] == "r-2"][0]
        adj["阿米娅"]["in"].append(dict(sym, neighbor_id="凯尔希", direction="in"))
        run = gates.l6_direction_gates(adj, edges)
        assert _gate(run, "L6.8").verdict == "FAIL"

    def test_reverse_using_original_predicate_fails(self, monkeypatch):
        edges = self._edges()
        adj = adjacency.build_directed_adjacency(edges)
        monkeypatch.setattr(adjacency, "inverse_of", lambda _p: "隶属")
        run = gates.l6_direction_gates(adj, edges)
        assert _gate(run, "L6.8").verdict == "FAIL"

    def test_missing_inverse_is_declared_not_failed(self):
        """无逆谓词只申报：这类边不能用于方向性表述，但它本身不是错误。"""
        edges = [_edge("r-1", subject="凯尔希", obj="罗德岛") | {"predicate": "前往"}]
        adj = adjacency.build_directed_adjacency(edges)
        run = gates.l6_direction_gates(adj, edges)
        assert _gate(run, "L6.8").verdict == "PASS"
        decl = _gate(run, "L6.8a")
        assert decl.verdict == "PASS"
        assert "1 条" in decl.measured


class TestAllEGatesCanFail:
    """元测试：每一条 E 阶段门都必须有能打红它的输入，否则不许合入。"""

    FAILING_INPUTS = {
        "L5.9": lambda: gates.l5_evidence_gates(
            {}, [_assertion("a-1", start=9, end=17)], _two_edges(), _chunks()
        ),
        "L5.10": lambda: gates.l5_evidence_gates(
            {}, _two_assertions(), [_edge("r-1", assertion_ids=[])], _chunks()
        ),
        "L5.11": lambda: gates.l5_evidence_gates(
            {}, _two_assertions(), [_edge("r-1", assertion_ids=["a-1"])], _chunks()
        ),
        "L4.12": lambda: gates.l4_identity_gates(
            {"entities": [{"id": "凯尔希", "name": "凯尔希"}]}
        ),
        "L4.13": lambda: gates.l4_identity_gates(
            {
                "entities": [{"id": "e-1", "name": "A", "merged_from": ["B"]}],
                "stats": {},
            }
        ),
        "L6.8": lambda: gates.l6_direction_gates(
            {
                "罗德岛": {
                    "out": [
                        {
                            "neighbor_id": "凯尔希",
                            "relation": "隶属",
                            # 伪装：入边塞进 out 列表、方向也标成 out。
                            # 单靠"方向与列表一致"这条判据抓不到，必须靠"两端落位"。
                            "direction": "out",
                            "inverse_relation": "包含",
                            "symmetric": False,
                            "edge_id": "r-1",
                            "assertion_ids": ["a-1"],
                            "weight": 0.9,
                            "support_count": 1,
                            "independent_scene_count": 1,
                            "reversible": True,
                        }
                    ],
                    "in": [],
                },
                "凯尔希": {"out": [], "in": []},
            },
            [_edge("r-1")],
        ),
    }

    @pytest.mark.parametrize("prefix", sorted(FAILING_INPUTS))
    def test_gate_actually_fails(self, prefix):
        run = self.FAILING_INPUTS[prefix]()
        assert _gate(run, prefix).verdict == "FAIL", (
            f"{prefix} 在坏输入下没有变红——这条门是假的"
        )

    def test_registry_covers_every_gate_produced(self):
        """所有 E 阶段门都必须登记在 FAILING_INPUTS 里，新增门不登记就红。"""
        produced = set()
        produced |= {g.name.split(" ")[0] for g in gates.l5_evidence_gates(
            {}, _two_assertions(), _two_edges(), _chunks()
        ).gates}
        produced |= {g.name.split(" ")[0] for g in gates.l4_identity_gates(
            {"entities": [{"id": "e-1", "name": "A", "merged_from": []}], "stats": {}}
        ).gates}
        edges = [_edge("r-1")]
        produced |= {g.name.split(" ")[0] for g in gates.l6_direction_gates(
            adjacency.build_directed_adjacency(edges), edges
        ).gates}
        # L6.8a 是纯申报门，不需要失败输入
        declared_informational = {"L6.8a"}
        missing = produced - set(self.FAILING_INPUTS) - declared_informational
        assert missing == set(), (
            f"这些门没有登记失败输入：{sorted(missing)}。"
            "新增门必须同时给出能打红它的输入，否则无法证明它不是假门。"
        )
