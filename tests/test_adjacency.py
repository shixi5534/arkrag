"""E2 有方向邻接的回归测试。

这套测试守的不是"邻接表能连上邻居"，而是**方向不被说反**：

* 出边用原谓词（`凯尔希 隶属 罗德岛`）；
* 入边必须挂逆谓词（从罗德岛这侧看是 `包含 凯尔希`）；
* 词表里没有逆谓词时 `inverse_relation=""`，**不许回退成原谓词**；
* 对称谓词（`同盟`/`交战`…）方向标 `undirected`，不许假装成有向边；
* 默认遍历只走出边，拼错的 direction 直接报错而不是默认放宽。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from arkrag import adjacency, vocab  # noqa: E402


def _edge(
    subject: str = "凯尔希",
    predicate: str = "隶属",
    obj: str = "罗德岛",
    *,
    edge_id: str = "r-aaa",
    support: int = 3,
    scenes: int = 2,
    mean: float = 0.8,
    assertions: list[str] | None = None,
) -> dict:
    return {
        "edge_id": edge_id,
        "subject_id": subject,
        "predicate": predicate,
        "object_id": obj,
        "assertion_ids": assertions if assertions is not None else ["a-1"],
        "support_count": support,
        "independent_scene_count": scenes,
        "confidence_summary": {"min": mean, "max": mean, "mean": mean},
    }


class TestInversePredicate:
    def test_known_inverse_is_returned(self):
        assert adjacency.inverse_of("隶属") == "包含"
        assert adjacency.inverse_of("包含") == "隶属"

    def test_inverse_is_symmetric_lookup(self):
        for a, b in vocab.INVERSE_PAIRS:
            assert adjacency.inverse_of(a) == b
            assert adjacency.inverse_of(b) == a

    def test_unknown_predicate_has_no_inverse(self):
        # 关键：不回退成原谓词。回退等于"反向说同一句话"。
        assert adjacency.inverse_of("注视") == ""

    def test_inverse_never_returns_original_predicate(self):
        """没有配对的谓词，逆值必须是空串而不是自身。"""
        for predicate in vocab.CANONICAL_RELATIONS:
            rev = adjacency.inverse_of(predicate)
            if not adjacency.is_symmetric(predicate):
                assert rev != predicate, f"{predicate} 的逆谓词不能是自身"


class TestDirectedAdjacency:
    def test_out_edge_keeps_predicate(self):
        adj = adjacency.build_directed_adjacency([_edge()])
        out = adj["凯尔希"]["out"]
        assert len(out) == 1
        assert out[0]["neighbor_id"] == "罗德岛"
        assert out[0]["relation"] == "隶属"
        assert out[0]["direction"] == "out"

    def test_in_edge_carries_inverse_relation(self):
        adj = adjacency.build_directed_adjacency([_edge()])
        inc = adj["罗德岛"]["in"]
        assert len(inc) == 1
        assert inc[0]["neighbor_id"] == "凯尔希"
        # 谓词是原事实"凯尔希 隶属 罗德岛"，但从罗德岛这侧描述要用"包含"
        assert inc[0]["relation"] == "隶属"
        assert inc[0]["inverse_relation"] == "包含"
        assert inc[0]["direction"] == "in"

    def test_out_and_in_are_separate_lists(self):
        """入边不许混进出边列表——混了就没法区分方向。"""
        adj = adjacency.build_directed_adjacency([_edge()])
        assert [i["neighbor_id"] for i in adj["凯尔希"]["out"]] == ["罗德岛"]
        assert [i["neighbor_id"] for i in adj["凯尔希"]["in"]] == []
        assert [i["neighbor_id"] for i in adj["罗德岛"]["in"]] == ["凯尔希"]
        assert [i["neighbor_id"] for i in adj["罗德岛"]["out"]] == []

    def test_edge_without_inverse_is_marked_not_reversible(self):
        adj = adjacency.build_directed_adjacency([_edge(predicate="注视")])
        inc = adj["罗德岛"]["in"][0]
        assert inc["inverse_relation"] == ""
        assert inc["reversible"] is False

    def test_items_carry_traceable_evidence(self):
        adj = adjacency.build_directed_adjacency(
            [_edge(assertions=["a-1", "a-2"], support=2, scenes=1)]
        )
        for node in adj.values():
            for item in node["out"] + node["in"]:
                assert item["edge_id"] == "r-aaa"
                assert item["assertion_ids"] == ["a-1", "a-2"]
                assert item["support_count"] == 2
                assert item["independent_scene_count"] == 1

    def test_build_is_deterministic(self):
        edges = [
            _edge(edge_id="r-bbb", support=1, obj="龙门"),
            _edge(edge_id="r-aaa", support=5, obj="切尔诺伯格"),
        ]
        first = adjacency.build_directed_adjacency(edges)
        second = adjacency.build_directed_adjacency(list(edges))
        assert first == second
        # 支持数高的排前面
        assert [i["neighbor_id"] for i in first["凯尔希"]["out"]] == [
            "切尔诺伯格",
            "龙门",
        ]


class TestSymmetricRelations:
    def test_symmetric_edge_is_undirected_not_fake_directed(self):
        adj = adjacency.build_directed_adjacency(
            [_edge(predicate="同盟", subject="凯尔希", obj="阿米娅")]
        )
        left = adj["凯尔希"]["out"][0]
        right = adj["阿米娅"]["out"][0]
        assert left["direction"] == "undirected"
        assert right["direction"] == "undirected"
        assert left["symmetric"] is True and right["symmetric"] is True
        # 对称关系两侧都可达，且都不许塞进 in 列表冒充"有人指向我"
        assert adj["阿米娅"]["in"] == []
        assert adj["凯尔希"]["in"] == []

    def test_symmetric_relation_inverse_is_itself(self):
        for predicate in vocab.SYMMETRIC_RELATIONS:
            item = adjacency.adjacency_item(
                neighbor_id="X",
                relation=predicate,
                direction="out",
                edge=_edge(predicate=predicate),
            )
            assert item["inverse_relation"] == predicate
            assert item["symmetric"] is True


class TestNeighbors:
    def test_default_is_out_only(self):
        edges = [
            _edge(edge_id="r-a", subject="凯尔希", obj="罗德岛"),
            _edge(edge_id="r-b", subject="阿米娅", predicate="隶属", obj="罗德岛"),
        ]
        adj = adjacency.build_directed_adjacency(edges)
        # 罗德岛这侧只有入边，默认 out 必须为空
        assert adjacency.neighbors(adj, "罗德岛") == []
        assert len(adjacency.neighbors(adj, "罗德岛", direction="in")) == 2
        assert len(adjacency.neighbors(adj, "罗德岛", direction="both")) == 2

    def test_bad_direction_raises(self):
        adj = adjacency.build_directed_adjacency([_edge()])
        with pytest.raises(ValueError, match="direction"):
            adjacency.neighbors(adj, "凯尔希", direction="outt")

    def test_unknown_entity_returns_empty(self):
        adj = adjacency.build_directed_adjacency([_edge()])
        assert adjacency.neighbors(adj, "不存在的人") == []

    def test_limit_is_respected(self):
        edges = [
            _edge(edge_id=f"r-{i}", obj=f"地点{i}", support=10 - i) for i in range(5)
        ]
        adj = adjacency.build_directed_adjacency(edges)
        assert len(adjacency.neighbors(adj, "凯尔希", limit=2)) == 2


class TestDirectionReport:
    def test_report_counts_out_and_in(self):
        adj = adjacency.build_directed_adjacency([_edge()])
        report = adjacency.direction_report(adj)
        assert report == {
            "entities": 2,
            "out_items": 1,
            "in_items": 1,
            "missing_inverse": 0,
            "fake_out": 0,
        }

    def test_report_flags_missing_inverse(self):
        adj = adjacency.build_directed_adjacency([_edge(predicate="注视")])
        assert adjacency.direction_report(adj)["missing_inverse"] == 1

    def test_report_detects_direction_masquerade(self):
        """把入边改成 out 造伪装，报告必须抓出来。"""
        adj = adjacency.build_directed_adjacency([_edge()])
        adj["罗德岛"]["out"].append(dict(adj["罗德岛"]["in"][0], direction="weird"))
        adj["罗德岛"]["in"] = []
        assert adjacency.direction_report(adj)["fake_out"] == 1

    def test_symmetric_undirected_does_not_count_as_fake(self):
        adj = adjacency.build_directed_adjacency([_edge(predicate="同盟")])
        assert adjacency.direction_report(adj)["fake_out"] == 0


class TestReverseSemantics:
    def test_no_problems_on_vocab_covered_edges(self):
        edges = [
            _edge(edge_id="r-a", predicate=a, subject="X", obj="Y")
            for a, _b in vocab.INVERSE_PAIRS[:4]
        ]
        assert adjacency.verify_reverse_semantics(edges) == []

    def test_missing_inverse_is_tolerated_but_symmetric_is_skipped(self):
        assert adjacency.verify_reverse_semantics([_edge(predicate="注视")]) == []
        assert adjacency.verify_reverse_semantics([_edge(predicate="同盟")]) == []

    def test_detects_inverse_that_equals_predicate(self, monkeypatch):
        """把逆谓词换回原谓词（正是旧 build_adjacency 的做法），必须报出。"""
        monkeypatch.setattr(adjacency, "inverse_of", lambda _p: "隶属")
        problems = adjacency.verify_reverse_semantics([_edge(predicate="隶属")])
        assert len(problems) == 1
        assert "沿用原谓词" in problems[0]


class TestTraversalDescription:
    def test_out_hop_and_in_hop_read_differently(self):
        out_line = adjacency.describe_traversal(
            {},
            [{"from_id": "凯尔希", "neighbor_id": "罗德岛", "relation": "隶属",
              "direction": "out"}],
        )[0]
        in_line = adjacency.describe_traversal(
            {},
            [{"from_id": "罗德岛", "neighbor_id": "凯尔希", "relation": "隶属",
              "direction": "in"}],
        )[0]
        assert out_line == "凯尔希 --隶属--> 罗德岛"
        assert in_line == "凯尔希 --隶属--> 罗德岛（反向读入）"
        assert out_line != in_line

    def test_undirected_hop(self):
        line = adjacency.describe_traversal(
            {},
            [{"from_id": "凯尔希", "neighbor_id": "阿米娅", "relation": "同盟",
              "direction": "undirected"}],
        )[0]
        assert "<--同盟-->" in line


class TestVocabCoverage:
    #: 词表里既无逆谓词、也不对称的谓词数量（实测固定值）。
    #: 它们的反向边 `inverse_relation=""`、`reversible=False`，
    #: 因此**不得用于顺着方向说事实**，只能当无向邻接用。
    ONE_WAY_ONLY = 17

    def test_unclassified_relations_are_counted_and_visible(self):
        """每个谓词要么有逆谓词、要么对称、要么被明确归为单向。

        单向谓词不是错误（"创建"确实没有自然的逆说法），但必须**数得清**：
        数量变了就说明有人往词表加了关系却没想方向，或者加了逆谓词忘了更新这里。
        """
        unclassified = sorted(
            p
            for p in vocab.CANONICAL_RELATIONS
            if not adjacency.inverse_of(p) and not adjacency.is_symmetric(p)
        )
        assert len(unclassified) == self.ONE_WAY_ONLY, (
            f"单向谓词数量从 {self.ONE_WAY_ONLY} 变成 {len(unclassified)}："
            f"{unclassified}。若新增了逆谓词请同步下调这个常量，"
            "若新增了关系请说明它的方向语义。"
        )

    def test_one_way_relations_are_not_reversible(self):
        for predicate in vocab.CANONICAL_RELATIONS:
            if adjacency.inverse_of(predicate) or adjacency.is_symmetric(predicate):
                continue
            item = adjacency.adjacency_item(
                neighbor_id="X",
                relation=predicate,
                direction="in",
                edge=_edge(predicate=predicate),
            )
            assert item["reversible"] is False
            assert item["inverse_relation"] == ""

    def test_every_inverse_pair_member_is_canonical_or_synonym(self):
        """逆谓词表里的名字必须能在词表里查到，否则反转换出来的是不可用的谓词。"""
        known = set(vocab.CANONICAL_RELATIONS) | set(vocab.RELATION_SYNONYMS)
        missing = sorted(
            {n for pair in vocab.INVERSE_PAIRS for n in pair} - known
        )
        assert missing == [], f"逆谓词表引用了词表里不存在的名字：{missing}"
