"""`ark-rag-3` 建图的端到端回归测试（E 阶段：稳定 id + 逐观察断言 + 方向邻接）。

这些测试守的是四件在真实语料上已被证明会静默出错的事：

1. **实体 id 不许是名字**。旧图 `id == 规范名`，改个显示名就等于换了个实体，
   所有引用它的边在下一代会集体换端点，而图上看起来一切正常。
2. **合并必须留审计**。并了却说不清按哪条规则并的，等于事后无法复核对错。
3. **待审别名不许可检索**。`identity.py` 记的实测事故：拒绝合并之后，
   别名表里仍写着 `review:"pending"`，而下游照样把它指过去——同一错误
   从"并实体"变成"并别名"，从图上更看不出来。
4. **聚合边不许脱离观察存在**。`assertion_ids` 被清空后，边还在、证据还写着，
   但它已经不再对应任何一次真实抽取——那是编出来的边。

每条门都必须能被注入的坏输入打红。只断言"干净数据时 PASS"的门等于没有门。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from arkrag import adjacency, config, gates, graph  # noqa: E402

# 两块正文，覆盖两种关系写法：一条同义谓词、一条需要换方向的被动式。
TEXT_A = "凯尔希：罗德岛需要你。\n阿米娅：我明白了。"
TEXT_B = "阿米娅：切尔诺伯格已经沦陷了。\n凯尔希：我们必须去龙门。"

CHUNKS = [
    {
        "chunk_id": "obt/main/level_main_01#000",
        "content_key": "ck0001",
        "text": TEXT_A,
        "title": "切尔诺伯格",
        "chapter": "第一章",
        "category": "obt/main",
        "scene_id": "obt/main/level_main_01",
        "doc_id": "obt/main/level_main_01",
    },
    {
        "chunk_id": "obt/main/level_main_01#001",
        "content_key": "ck0002",
        "text": TEXT_B,
        "title": "切尔诺伯格",
        "chapter": "第一章",
        "category": "obt/main",
        "scene_id": "obt/main/level_main_01",
        "doc_id": "obt/main/level_main_01",
    },
]

RECORDS = [
    {
        "chunk_id": "obt/main/level_main_01#000",
        "raw": {
            "entities": [
                {"name": "凯尔希", "type": "角色", "desc": "罗德岛医疗负责人"},
                # 别名指向另一个写法，用来验证别名进 id、规范名不进
                {"name": "阿米娅", "type": "角色", "aliases": ["阿米亚"]},
                {"name": "罗德岛", "type": "组织"},
            ],
            "relations": [
                {
                    "source": "凯尔希",
                    "relation": "隶属于",
                    "target": "罗德岛",
                    "evidence": "罗德岛需要你",
                    "confidence": 0.9,
                }
            ],
        },
    },
    {
        "chunk_id": "obt/main/level_main_01#001",
        "raw": {
            "entities": [
                {"name": "阿米娅", "type": "角色"},
                {"name": "凯尔希", "type": "角色"},
                {"name": "龙门", "type": "地点"},
            ],
            "relations": [
                {
                    "source": "凯尔希",
                    "relation": "隶属于",
                    "target": "龙门",
                    "evidence": "我们必须去龙门",
                    "confidence": 0.8,
                }
            ],
        },
    },
]


def _build(**kw):
    return graph.build_identity_graph(CHUNKS, RECORDS, **kw)


def _by_name(graph_doc: dict) -> dict[str, dict]:
    return {e["name"]: e for e in graph_doc["entities"]}


# ------------------------------------------------------------------ 1. 完整形态


class TestGraphShape:
    def test_graph_is_ark_rag_3_with_all_four_products(self):
        out = _build()
        g = out["graph"]
        assert g["schema_version"] == config.GRAPH_SCHEMA_IDENTITY == "ark-rag-3"
        # 四件套齐全：图、逐观察断言、聚合边、方向邻接
        assert out["assertions"] and out["edges"] and out["adjacency"]
        assert out["alias_map"]

    def test_entities_carry_stable_id_not_name(self):
        g = _build()["graph"]
        for ent in g["entities"]:
            assert re.fullmatch(r"e-[0-9a-f]{16}", ent["id"]), ent["id"]
            assert ent["name"] not in ent["id"]
            # 三张别名表必须同时存在：只给一张表就等于把"待审"混进"权威"
            assert {"aliases", "pending_aliases", "rejected_aliases"} <= set(ent)

    def test_alias_index_points_at_id_and_holds_aliases_only(self):
        g = _build()["graph"]
        by_name = _by_name(g)
        amiya = by_name["阿米娅"]
        # 规范名不进 alias_index（与旧图同一约定），别名进且指向 id
        assert "阿米娅" not in g["alias_index"]
        if amiya["aliases"]:
            text = amiya["aliases"][0]["text"]
            assert g["alias_index"][text] == amiya["id"]

    def test_edges_are_aggregated_from_assertions(self):
        out = _build()
        aids = {a["assertion_id"] for a in out["assertions"]}
        for edge in out["edges"]:
            # 每条聚合边都必须能逐条回溯到真实观察，且条数与 support_count 一致
            assert edge["assertion_ids"], "聚合边没有 assertion_ids 就是编出来的边"
            assert set(edge["assertion_ids"]) <= aids
            assert edge["support_count"] == len(edge["assertion_ids"])
            assert edge["evidence_assertion_id"] in edge["assertion_ids"]

    def test_adjacency_separates_out_and_in(self):
        out = _build()
        adj = out["adjacency"]
        assert adj, "方向邻接不能为空"
        for node in adj.values():
            assert set(node) == {"out", "in"}
            for item in node["out"]:
                assert item["direction"] in ("out", "undirected")
            for item in node["in"]:
                assert item["direction"] in ("in", "undirected")


# ------------------------------------------------------------------ 2. 旧路径不变


class TestLegacyGraphUntouched:
    def test_build_graph_still_returns_ark_rag_1(self):
        """`build_graph` 一行没改：它建出来的图必须还是 ark-rag-1 的形状。

        新函数并存不替代。这条测试防的是"顺手把新语义塞进旧函数"——
        那会让每一代已发布产物与冻结基线在下次重建时变形。
        """
        legacy = graph.build_graph(CHUNKS, RECORDS)
        assert legacy["schema_version"] == config.GRAPH_SCHEMA_LEGACY == "ark-rag-1"
        assert legacy["entities"]
        for ent in legacy["entities"]:
            assert ent["id"] == ent["name"], "旧图的 id 就是规范名，这一条不许变"
            assert all(isinstance(a, str) for a in ent["aliases"]), "旧图别名是 str 列表"
        for edge in legacy["edges"]:
            assert {"source", "relation", "target", "confidence"} <= set(edge)
            assert "assertion_ids" not in edge


# ------------------------------------------------------------------ 3. 端到端


class TestEndToEndGates:
    def test_ark_rag_3_passes_its_own_gates(self):
        out = _build()
        g = out["graph"]
        runs = [
            gates.l5_evidence_gates(g, out["assertions"], out["edges"], CHUNKS),
            gates.l4_identity_gates(g),
            gates.l6_direction_gates(out["adjacency"], out["edges"]),
        ]
        produced = {gate.name.split(" ")[0] for run in runs for gate in run.gates}
        assert {"L5.9", "L5.10", "L5.11", "L4.12", "L4.13", "L6.8"} <= produced
        fails = [gate for run in runs for gate in run.gates if gate.verdict == "FAIL"]
        assert not fails, [(gate.name, gate.measured, gate.note) for gate in fails]

    def test_l412_passes_because_ids_are_hashes(self):
        run = gates.l4_identity_gates(_build()["graph"])
        gate = next(g for g in run.gates if g.name.startswith("L4.12"))
        assert gate.verdict == "PASS"
        # measured 形如「0/3」：分子是命中数，必须是 0
        assert gate.measured.split("/")[0].strip() == "0"


# ------------------------------------------------------------------ 4. 喂坏输入必红


class TestBadInputTurnsGatesRed:
    def test_pending_alias_leaking_into_index_turns_l6_4c_red(self):
        """待审别名被塞进可检索表 → L6.4c 必须 FAIL。

        构造方式刻意贴近真实事故：不从 `aliases` 里塞（那样 `to_dict()` 早就把它
        归到 pending 了），而是**手工把 pending 的文本写进 `alias_index`**，
        模拟下游某个"顺手用全量别名建倒排"的实现。
        """
        g = _build()["graph"]
        victim = g["entities"][0]
        leaked = "库兰塔士兵"
        victim["pending_aliases"] = [{"text": leaked, "source": "upstream", "scenes": [], "review": "pending"}]
        g["alias_index"][leaked] = victim["id"]

        run = gates.l6_gates(
            {
                "graph_fp": g.get("fingerprint", ""),
                "chunk_count": len(CHUNKS),
                "corpus_fp": g.get("corpus_fp", ""),
                "vector_channels": "not_built_no_embedding_backend",
                "built_at": "2026-10-31T00:00:00Z",
            },
            g,
            len(CHUNKS),
            g.get("corpus_fp", ""),
            [],
        )
        gate = next(x for x in run.gates if x.name.startswith("L6.4c"))
        assert gate.verdict == "FAIL", f"待审别名漏进检索却没变红：{gate.measured}"
        assert leaked in gate.note

    def test_cleared_assertion_ids_turn_l5_10_red(self):
        """聚合边丢了 assertion_ids → L5.10 必须 FAIL（那条边已经无据可查）。"""
        out = _build()
        broken = [dict(edge) for edge in out["edges"]]
        broken[0]["assertion_ids"] = []
        run = gates.l5_evidence_gates(out["graph"], out["assertions"], broken, CHUNKS)
        gate = next(g for g in run.gates if g.name.startswith("L5.10"))
        assert gate.verdict == "FAIL"
        # 判据记在 measured 里：缺 ids 的边数必须非 0
        assert "缺 ids 1" in gate.measured
        assert broken[0]["edge_id"] in gate.note

    def test_masqueraded_direction_turns_l6_4_red(self):
        """入边塞进 out 列表且方向标成 out → L6.8 必须 FAIL。"""
        out = _build()
        adj = {k: {"out": list(v["out"]), "in": list(v["in"])} for k, v in out["adjacency"].items()}
        node_id = next(k for k, v in adj.items() if v["in"])
        # 把一条入边搬到 out 列表、方向也改成 out：单看"方向与列表一致"抓不到
        adj[node_id]["in"][0]["direction"] = "out"
        adj[node_id]["out"].append(adj[node_id]["in"].pop(0))
        run = gates.l6_direction_gates(adj, out["edges"])
        gate = next(g for g in run.gates if g.name.startswith("L6.8 "))
        assert gate.verdict == "FAIL"

    def test_merged_without_audit_turns_l4_13_red(self):
        g = _build()["graph"]
        g["entities"][0]["merged_from"] = [{"name": "某人", "reason": "alias", "scenes": []}]
        run = gates.l4_identity_gates(g)
        gate = next(x for x in run.gates if x.name.startswith("L4.13"))
        assert gate.verdict == "FAIL"


# ------------------------------------------------------------------ 5. 分派


class TestFieldDispatch:
    """gates/index 要同时读两代图的字段，且**判据只写一份**。

    这组测试守的是分派的两个方向都不能漏：只认旧形态会让新图整组门静默走错分支，
    只认新形态会让已发布世代的门失效。
    """

    def test_alias_texts_reads_both_shapes(self):
        legacy = {"aliases": ["甲", "乙"]}
        identity = {"aliases": [{"text": "甲", "review": "auto"}, {"text": "", "review": "auto"}]}
        assert gates.alias_texts(legacy) == ["甲", "乙"]
        assert gates.alias_texts(identity) == ["甲"]

    def test_merged_names_reads_both_shapes(self):
        assert gates.merged_names({"merged_from": ["甲"]}) == ["甲"]
        assert gates.merged_names(
            {"merged_from": [{"name": "甲", "reason": "alias", "scenes": []}]}
        ) == ["甲"]

    def test_edge_accessors_read_both_shapes(self):
        legacy_edge = {"source": "A", "target": "B", "relation": "隶属", "confidence": 0.7}
        identity_edge = {
            "subject_id": "A",
            "object_id": "B",
            "predicate": "隶属",
            "confidence_summary": {"min": 0.2, "max": 0.9, "mean": 0.5},
        }
        assert gates.edge_ends(legacy_edge) == gates.edge_ends(identity_edge) == ("A", "B")
        assert gates.edge_relation(legacy_edge) == gates.edge_relation(identity_edge) == "隶属"
        # 取 max 不取 mean：旧判据问的是"最可信那次够不够"
        assert gates.edge_confidence(legacy_edge) == pytest.approx(0.7)
        assert gates.edge_confidence(identity_edge) == pytest.approx(0.9)

    def test_l5_gates_accepts_ark_rag_3_graph(self):
        """把新图喂给**没改判据**的 L5：必须能读通，且不该因为形状不同而误判。"""
        g = _build()["graph"]
        g["corpus_fp"] = "corpusfp01234567"
        g["fingerprint"] = "fp0123456789abcdef"
        run = gates.l5_gates(g, CHUNKS, "corpusfp01234567")
        shape_fails = [
            gate.name
            for gate in run.gates
            if gate.verdict == "FAIL"
            and gate.name.split(" ")[0] in {"L5.4", "L5.5", "L5.6"}
        ]
        assert not shape_fails, f"新图被旧判据按形状误判：{shape_fails}"

    def test_index_builders_accept_both_shapes(self):
        from arkrag import index

        legacy_entities = [
            {"id": "凯尔希", "name": "凯尔希", "aliases": ["凯尔西"]},
            {"id": "阿米娅", "name": "阿米娅", "aliases": []},
        ]
        identity_entities = [
            {"id": "e-0000000000000001", "name": "凯尔希", "aliases": [{"text": "凯尔西"}]},
            {"id": "e-0000000000000002", "name": "阿米娅", "aliases": []},
        ]
        for ents, ok_id in ((legacy_entities, "凯尔希"), (identity_entities, "e-0000000000000001")):
            alias_index = index.build_alias_index(ents)
            assert alias_index["凯尔西"] == ok_id
            assert alias_index["凯尔希"] == ok_id

        legacy_edges = [{"source": "A", "target": "B", "relation": "隶属", "confidence": 0.7}]
        identity_edges = [
            {
                "subject_id": "A",
                "object_id": "B",
                "predicate": "隶属",
                "confidence_summary": {"max": 0.7},
            }
        ]
        assert index.build_adjacency(legacy_edges) == index.build_adjacency(identity_edges)
        assert index.build_adjacency(legacy_edges)["A"] == [("B", "隶属", 0.7)]


# ------------------------------------------------------------------ 6. 落盘产物


class TestSidecarArtifacts:
    def test_assertions_jsonl_is_sorted_and_round_trips(self, tmp_path):
        out = _build()
        path = tmp_path / "assertions.jsonl"
        count = graph.save_assertions(out["assertions"], path)
        assert count == len(out["assertions"])

        rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
        assert [r["assertion_id"] for r in rows] == sorted(
            a["assertion_id"] for a in out["assertions"]
        ), "断言必须按 assertion_id 排序，否则 assertion_artifact_id 不稳定"

    def test_adjacency_file_round_trips(self, tmp_path):
        out = _build()
        path = tmp_path / "adjacency.json"
        graph.save_adjacency(out["adjacency"], path)
        assert json.loads(path.read_text("utf-8")) == out["adjacency"]
