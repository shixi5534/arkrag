"""`arkrag.retrieve` 的判据测试 —— 每一条都对着一个会安静说错话的失效模式。

## 这个文件在守什么

检索与扩展出错的形态不是抛异常，而是「多给一点」或「少给一点」：

* 多给：把反向边当正向边用 → 事实说反（`隶属` 读成 `被隶属`）。
* 多给：为凑上下文塞进弱关系 / 无原句支撑的图结论 → 下游更容易编事实。
* 少给：预算被一条边独占 → 其余邻居一个都进不来，召回莫名其妙变差。
* 少给：截断时切开某个单元的 `evidence_spans` → 下游以为自己有证据。

五类判据：

1. **方向**（`TestDirection`）：默认只走出边；入边必须显式 `include_in=True`，
   且结果标 `used_in_edges=True`、`relation` 换成逆谓词。
2. **预算**（`TestBudget`）：四个上限各自能生效并写进 `truncated_reason`；
   单边断言上限防止一条边吃掉整个 `max_assertions`。
3. **证据**（`TestEvidenceAnchors`）：没有可定位证据的边不进扩展。
4. **上下文**（`TestBuildContext`）：同 chunk 合并、相同证据去重、直接证据优先、
   截断整单元丢弃且不切 span、`context_digest` 稳定。
5. **检索通道**（`TestSearch`）：两路分数各自可见，拒答问句直接空手返回。

## 隔离

所有用例把产物根与世代目录指到 `tmp_path` 并断言前缀。`Retriever` 本身是纯内存
对象，但纪律要求「将来有人在这里顺手加一次读盘，测试不能还静默通过」。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from arkrag import adjacency as adjacency_mod
from arkrag import config, index as index_mod, query, release, retrieve, vocab


# --------------------------------------------------------------------------
# 人造 ark-rag-3 图：3 实体、3 条边、3 个断言、3 个块
# --------------------------------------------------------------------------
CHUNK_A = "凯尔希：切尔诺伯格已经沦陷了。\n阿米娅：我们必须去罗德岛。"
CHUNK_B = "凯尔希：罗德岛需要你回来。\n阿米娅：我会听你的。"
CHUNK_C = "档案：今天天气不错，无人在场。"

ENT_KAL = "e-0000000000000001"
ENT_AMI = "e-0000000000000002"
ENT_RHO = "e-0000000000000003"


def _span(text: str, needle: str) -> tuple[int, int]:
    start = text.find(needle)
    assert start >= 0, f"造数据错误：{needle!r} 不在正文里"
    return start, start + len(needle)


def _chunk(cid: str, text: str, title: str = "测试篇") -> dict:
    return {
        "chunk_id": cid,
        "doc_id": cid.split("#")[0],
        "title": title,
        "chapter": "第一章",
        "text": text,
        "chars": len(text),
    }


def _fixture() -> tuple[dict, list[dict], list[dict], dict, list[dict]]:
    """返回 `(graph, chunks, assertions, adjacency, query_analysis)`。"""
    chunks = [
        _chunk("doc/a#000", CHUNK_A),
        _chunk("doc/a#001", CHUNK_B),
        _chunk("doc/b#000", CHUNK_C),
    ]
    entities = [
        {"id": ENT_KAL, "name": "凯尔希", "type": "角色",
         "aliases": [{"text": "凯尔希医生", "source": "upstream", "scenes": [], "review": "auto"}],
         "pending_aliases": [{"text": "老女人", "source": "upstream", "scenes": [], "review": "pending"}]},
        {"id": ENT_AMI, "name": "阿米娅", "type": "角色", "aliases": []},
        {"id": ENT_RHO, "name": "罗德岛", "type": "组织", "aliases": []},
    ]

    def _assertion(aid, cid, text, subj, pred, obj, evidence):
        start, end = _span(text, evidence)
        return {
            "assertion_id": aid,
            "subject_id": subj,
            "predicate": pred,
            "object_id": obj,
            "chunk_id": cid,
            "evidence_text": evidence,
            "evidence_start": start,
            "evidence_end": end,
            "confidence": 0.9,
        }

    assertions = [
        _assertion("a-0000000000000001", "doc/a#000", CHUNK_A, ENT_KAL, "隶属", ENT_RHO, "我们必须去罗德岛。"),
        # 这条是**非对称**谓词（保护 ↔ 受保护于），用来验方向：
        # `阿米娅 --保护--> 凯尔希` 从凯尔希这边看是入边，默认不许被读成
        # `凯尔希 --保护--> 阿米娅`。
        _assertion("a-0000000000000002", "doc/a#000", CHUNK_A, ENT_AMI, "保护", ENT_KAL, "阿米娅：我们必须去罗德岛。"),
        _assertion("a-0000000000000003", "doc/a#001", CHUNK_B, ENT_KAL, "指挥", ENT_AMI, "我会听你的。"),
        _assertion("a-0000000000000004", "doc/a#001", CHUNK_B, ENT_KAL, "交谈", ENT_AMI, "凯尔希：罗德岛需要你回来。"),
    ]
    edges = [
        {"edge_id": "r-0000000000000001", "subject_id": ENT_KAL, "predicate": "隶属",
         "object_id": ENT_RHO, "assertion_ids": ["a-0000000000000001"],
         "support_count": 5, "independent_scene_count": 3,
         "confidence_summary": {"min": 0.9, "max": 0.9, "mean": 0.9},
         "evidence": "我们必须去罗德岛。", "chunks": ["doc/a#000"], "scenes": ["doc/a"]},
        {"edge_id": "r-0000000000000002", "subject_id": ENT_AMI, "predicate": "保护",
         "object_id": ENT_KAL, "assertion_ids": ["a-0000000000000002"],
         "support_count": 2, "independent_scene_count": 1,
         "confidence_summary": {"min": 0.9, "max": 0.9, "mean": 0.9},
         "evidence": "阿米娅：我们必须去罗德岛。", "chunks": ["doc/a#000"], "scenes": ["doc/a"]},
        {"edge_id": "r-0000000000000003", "subject_id": ENT_KAL, "predicate": "指挥",
         "object_id": ENT_AMI, "assertion_ids": ["a-0000000000000003"],
         "support_count": 1, "independent_scene_count": 1,
         "confidence_summary": {"min": 0.9, "max": 0.9, "mean": 0.9},
         "evidence": "我会听你的。", "chunks": ["doc/a#001"], "scenes": ["doc/a"]},
        {"edge_id": "r-0000000000000004", "subject_id": ENT_KAL, "predicate": "交谈",
         "object_id": ENT_AMI, "assertion_ids": ["a-0000000000000004"],
         "support_count": 1, "independent_scene_count": 1,
         "confidence_summary": {"min": 0.9, "max": 0.9, "mean": 0.9},
         "evidence": "凯尔希：罗德岛需要你回来。", "chunks": ["doc/a#001"], "scenes": ["doc/a"]},
    ]
    graph = {
        "schema_version": config.GRAPH_SCHEMA_IDENTITY,
        "entities": entities,
        "edges": edges,
        "alias_index": index_mod.build_alias_index(entities),
        "fingerprint": "0123456789abcdef",
        "stats": {"entities": 3, "edges": 3},
    }
    adj = adjacency_mod.build_directed_adjacency(edges)
    qa = query.analyze_query("凯尔希和罗德岛是什么关系", entities=entities, vocab=vocab)
    return graph, chunks, assertions, adj, qa


def _make_retriever(chunks=None, assertions=None, budget=None) -> tuple[retrieve.Retriever, dict]:
    graph, all_chunks, all_assertions, adj, qa = _fixture()
    idx, _meta = index_mod.build(all_chunks, graph, "corpusfp01234567")
    ret = retrieve.Retriever(
        idx,
        graph,
        adj,
        chunks=chunks if chunks is not None else all_chunks,
        assertions=assertions if assertions is not None else all_assertions,
        budget=budget,
    )
    return ret, qa


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    root = (tmp_path / "artifacts").resolve()
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("ARKRAG_ARTIFACT_DIR", str(root))
    monkeypatch.setattr(release, "GENERATIONS_DIR", (root / "generations").resolve())
    monkeypatch.setattr(release, "RUNS_DIR", (root / "runs").resolve())
    monkeypatch.setattr(release, "CURRENT_PATH", (root / "current.json").resolve())
    assert str(config.artifact_root()).startswith(str(tmp_path.resolve()))
    assert Path(config.artifact_root()) != config.DATA_DIR
    return root


class TestSearch:
    def test_both_channels_are_visible_in_sources(self):
        ret, qa = _make_retriever()
        res = ret.search(qa, top_k=5)
        assert res["hits"], "三条块里至少『凯尔希/罗德岛』那条要命中"
        channels = {s["channel"] for h in res["hits"] for s in h["sources"]}
        assert retrieve.SOURCE_BM25 in channels
        assert retrieve.SOURCE_ENTITY in channels

    def test_entity_channel_boosts_chunk_with_names(self):
        """实体通道必须能抬高含问句实体的块。

        `doc/b#000` 与问句没有任何词重叠，实体通道也不该把它拉上来；如果
        实现把 entity 分数当成常数加给所有块，这里就会露出来。
        """
        ret, qa = _make_retriever()
        res = ret.search(qa, top_k=5)
        ids = [h["chunk_id"] for h in res["hits"]]
        assert "doc/a#000" in ids
        assert "doc/b#000" not in ids

    def test_refused_query_returns_no_hits(self):
        ret, _qa = _make_retriever()
        refused = query.analyze_query("", entities=_fixture()[0]["entities"], vocab=vocab)
        res = ret.search(refused, top_k=5)
        assert res["hits"] == []
        assert res["stats"]["skipped_reason"] == "empty_query"

    def test_stats_report_each_channel(self):
        ret, qa = _make_retriever()
        stats = ret.search(qa, top_k=5)["stats"]
        for key in ("bm25_terms", "bm25_top_score", "entity_channels", "entity_top_score",
                    "candidates", "top_score", "queried_text"):
            assert key in stats
        assert stats["entity_channels"] == len(qa["entity_candidates"])


class TestDirection:
    def test_default_expansion_walks_out_edges_only(self):
        """凯尔希出边只有「隶属 罗德岛」，入边是「阿米娅 --保护--> 凯尔希」。"""
        ret, qa = _make_retriever()
        out = ret.expand([ENT_KAL], depth=1, query_analysis=qa)
        relations = {(e["from_name"], e["relation"], e["neighbor_name"]) for e in out["expansions"]}
        assert ("凯尔希", "隶属", "罗德岛") in relations
        # 入边是「阿米娅 --保护--> 凯尔希」，默认不许把它读成「凯尔希 --保护--> 阿米娅」
        assert ("凯尔希", "保护", "阿米娅") not in relations
        assert out["stats"]["used_in_edges"] is False

    def test_include_in_marks_used_in_edges_and_uses_inverse_predicate(self):
        ret, qa = _make_retriever()
        out = ret.expand([ENT_KAL], depth=1, include_in=True, query_analysis=qa)
        assert out["stats"]["used_in_edges"] is True
        inbound = [e for e in out["expansions"] if e["direction"] == "in"]
        assert inbound, "include_in=True 时必须至少带回一条入边"
        for item in inbound:
            assert item["used_in_edges"] is True
            # 入边项的 relation 是原事实谓词，inverse_relation 才是从本侧描述时该用的词
            assert item["inverse_relation"] == adjacency_mod.inverse_of(item["relation"])
            assert item["inverse_relation"] != item["relation"]

    def test_symmetric_edge_is_not_counted_as_inbound_direction(self):
        """对称谓词两侧都是 out/undirected，不能因为进 in 列表就算方向性入边。"""
        graph, chunks, assertions, _adj, qa = _fixture()
        edges = copy.deepcopy(graph["edges"])
        edges.append(
            {"edge_id": "r-0000000000000009", "subject_id": ENT_KAL, "predicate": "战友",
             "object_id": ENT_AMI, "assertion_ids": ["a-0000000000000002"],
             "support_count": 1, "independent_scene_count": 1,
             "confidence_summary": {"min": 0.8, "max": 0.8, "mean": 0.8},
             "evidence": "阿米娅：我们必须去罗德岛。", "chunks": ["doc/a#000"], "scenes": ["doc/a"]}
        )
        graph["edges"] = edges
        adj = adjacency_mod.build_directed_adjacency(edges)
        idx, _ = index_mod.build(chunks, graph, "corpusfp01234567")
        ret = retrieve.Retriever(idx, graph, adj, chunks=chunks, assertions=assertions)
        out = ret.expand([ENT_KAL], depth=1, query_analysis=qa)
        sym = [e for e in out["expansions"] if e["relation"] == "战友"]
        if sym:  # 词表未把「战友」列为对称时跳过该断言
            for item in sym:
                assert item["direction"] in ("out", "undirected")


class TestBudget:
    def test_max_edges_is_enforced_and_reported(self):
        ret, qa = _make_retriever(budget={"max_edges": 1, "max_degree_per_entity": 99})
        out = ret.expand([ENT_KAL], depth=1, query_analysis=qa)
        assert len(out["expansions"]) <= 1
        assert out["stats"]["truncated_reason"].startswith("max_edges")

    def test_max_degree_per_entity_is_enforced(self):
        """这条守的是「一条边吃满整个预算」——没有度数上限时，一个实体就能
        用掉全部边预算，其余种子实体一条边都进不来。"""
        ret, qa = _make_retriever(budget={"max_degree_per_entity": 1, "max_edges": 50})
        out = ret.expand([ENT_KAL], depth=1, query_analysis=qa)
        assert len(out["expansions"]) == 1
        assert out["stats"]["truncated_reason"] == "max_degree_per_entity(1)"

    def test_max_assertions_per_edge_prevents_one_edge_from_eating_the_budget(self):
        """一条边挂 50 条断言时，预算 10 会被它一条吃光，其余邻居全进不来。

        判据不能只看 `assertions_used <= 10`——**没有单边上限时它恰好也等于 10**。
        必须看「有几条边进来了」：有上限（3）时第一条只吃 3 条，后面两条各 1 条，
        三条边都进得来；没有上限时第一条吃满 10 条，第二轮就撞上限 break。
        """
        graph, chunks, assertions, adj, qa = _fixture()
        many = []
        edge = copy.deepcopy(graph["edges"][0])
        edge["assertion_ids"] = []
        for i in range(50):
            aid = f"a-{i:016d}"
            many.append({"assertion_id": aid, "subject_id": ENT_KAL, "predicate": "隶属",
                         "object_id": ENT_RHO, "chunk_id": "doc/a#000",
                         "evidence_text": "我们必须去罗德岛。", "evidence_start": 18,
                         "evidence_end": 27, "confidence": 0.9})
            edge["assertion_ids"].append(aid)
        graph["edges"] = [edge, graph["edges"][2], graph["edges"][3]]
        adj = adjacency_mod.build_directed_adjacency(graph["edges"])
        idx, _ = index_mod.build(chunks, graph, "corpusfp01234567")
        ret = retrieve.Retriever(idx, graph, adj, chunks=chunks, assertions=many,
                                 budget={"max_assertions": 10, "max_assertions_per_edge": 3})
        out = ret.expand([ENT_KAL], depth=1, query_analysis=qa)
        assert out["stats"]["budget"]["max_assertions_per_edge"] == 3
        assert out["stats"]["assertions_used"] == 5, "3 + 1 + 1"
        assert len(out["expansions"]) == 3, "三条出边都必须进得来"

    def test_budget_is_echoed_with_actual_usage(self):
        ret, qa = _make_retriever()
        stats = ret.expand([ENT_KAL], depth=1, query_analysis=qa)["stats"]
        for key in ("max_depth", "max_nodes", "max_edges", "max_degree_per_entity",
                    "max_assertions", "max_assertions_per_edge", "timeout_ms", "support_high"):
            assert key in stats["budget"]
        assert stats["budget"]["depth_used"] == 1

    def test_priority_tiers_are_counted(self):
        ret, qa = _make_retriever()
        stats = ret.expand([ENT_KAL], depth=1, query_analysis=qa)["stats"]
        assert stats["priority_tiers"], "至少要有一档被命中"
        assert all(k in stats["priority_legend"] for k in stats["priority_tiers"])
        # 凯尔希是种子实体，其出边必然是第 1 档
        assert "1" in stats["priority_tiers"]

    def test_depth_zero_yields_nothing(self):
        ret, qa = _make_retriever()
        out = ret.expand([ENT_KAL], depth=0, query_analysis=qa)
        assert out["expansions"] == []


class TestEvidenceAnchors:
    def test_edge_without_any_evidence_never_enters_expansion(self):
        """图结论没有原句支撑时进上下文，只会让下游更容易编事实。"""
        graph, chunks, assertions, _adj, qa = _fixture()
        bare = {"edge_id": "r-0000000000000007", "subject_id": ENT_KAL, "predicate": "隶属",
                "object_id": ENT_RHO, "assertion_ids": [], "support_count": 9,
                "independent_scene_count": 1,
                "confidence_summary": {"min": 0.9, "max": 0.9, "mean": 0.9},
                "evidence": "", "chunks": [], "scenes": []}
        graph["edges"] = [bare]
        adj = adjacency_mod.build_directed_adjacency(graph["edges"])
        idx, _ = index_mod.build(chunks, graph, "corpusfp01234567")
        ret = retrieve.Retriever(idx, graph, adj, chunks=chunks, assertions=assertions)
        out = ret.expand([ENT_KAL], depth=1, query_analysis=qa)
        assert out["expansions"] == []

    def test_every_expansion_carries_chunk_id_and_evidence(self):
        ret, qa = _make_retriever()
        out = ret.expand([ENT_KAL], depth=1, query_analysis=qa)
        assert out["expansions"]
        for item in out["expansions"]:
            assert item["chunk_id"], item
            assert item["evidence"], item
            assert item["evidence_spans"], item
            for span in item["evidence_spans"]:
                assert span["start"] >= 0 and span["end"] > span["start"]

    def test_missing_chunk_data_degrades_offsets_to_minus_one_not_fake(self):
        """没有断言数据时允许退回边自带 evidence，但偏移必须记 -1。

        假装知道位置会让下游的 `text[start:end] == evidence` 校验变成一句空话。
        """
        graph, chunks, _assertions, adj, qa = _fixture()
        idx, _ = index_mod.build(chunks, graph, "corpusfp01234567")
        ret = retrieve.Retriever(idx, graph, adj, chunks=chunks, assertions=None)
        out = ret.expand([ENT_KAL], depth=1, query_analysis=qa)
        assert out["expansions"]
        for item in out["expansions"]:
            for span in item["evidence_spans"]:
                assert span["start"] == -1 and span["end"] == -1


class TestBuildContext:
    def test_same_chunk_appears_once_with_merged_sources(self):
        ret, qa = _make_retriever()
        hits = ret.search(qa, top_k=5)["hits"]
        exp = ret.expand([ENT_KAL], depth=1, query_analysis=qa)["expansions"]
        ctx = ret.build_context(hits, exp, query_analysis=qa)
        ids = [u["chunk_id"] for u in ctx["units"]]
        assert len(ids) == len(set(ids))

    def test_duplicate_evidence_is_merged(self):
        ret, qa = _make_retriever()
        hits = ret.search(qa, top_k=5)["hits"]
        exp = ret.expand([ENT_KAL], depth=1, query_analysis=qa)["expansions"]
        ctx = ret.build_context(hits, exp, query_analysis=qa)
        for unit in ctx["units"]:
            keys = [(s["evidence"], s["start"], s["end"]) for s in unit["evidence_spans"]]
            assert len(keys) == len(set(keys)), unit["chunk_id"]

    def test_direct_evidence_units_come_first(self):
        ret, qa = _make_retriever()
        hits = ret.search(qa, top_k=5)["hits"]
        exp = ret.expand([ENT_KAL], depth=1, query_analysis=qa)["expansions"]
        ctx = ret.build_context(hits, exp, query_analysis=qa)
        seen_graph_only = False
        for unit in ctx["units"]:
            is_direct = any(s["channel"] in (retrieve.SOURCE_BM25, retrieve.SOURCE_ENTITY)
                            for s in unit["sources"])
            if not is_direct:
                seen_graph_only = True
            elif seen_graph_only:
                pytest.fail("纯图背景块排在了直接证据块之前")

    def test_truncation_drops_whole_units_and_never_cuts_spans(self):
        """切一半的引用比没有引用更坏：下游会以为自己有证据。"""
        ret, qa = _make_retriever()
        hits = ret.search(qa, top_k=5)["hits"]
        exp = ret.expand([ENT_KAL], depth=1, query_analysis=qa)["expansions"]
        ctx = ret.build_context(hits, exp, token_budget=1, query_analysis=qa)
        assert ctx["units"] == []
        assert ctx["stats"]["truncated_chunks"]
        assert ctx["stats"]["evidence_spans"] == 0

    def test_context_digest_is_stable_and_input_sensitive(self):
        """摘要必须只由「最终留在上下文里的东西」决定。

        两次实测教训（都记在这里，免得后来人重犯）：

        * 少给一条 hit **不一定**改变摘要——那条 hit 指向的块可能仍被图扩展带回，
          最终单元集合与证据集合都没变。摘要不变才是对的。
        * 去掉全部扩展也不一定变——本次扩展带回的块与命中块重合。
        * 所以敏感度要用**真正进入单元集合的新块**来验（这里加 doc/b#000）。
        """
        ret, qa = _make_retriever()
        hits = ret.search(qa, top_k=5)["hits"]
        exp = ret.expand([ENT_KAL], depth=1, query_analysis=qa)["expansions"]
        first = ret.build_context(hits, exp, query_analysis=qa)["context_digest"]
        second = ret.build_context(hits, exp, query_analysis=qa)["context_digest"]
        assert first == second

        wider = hits + [{"chunk_id": "doc/b#000", "score": 0.1, "sources": []}]
        assert ret.build_context(wider, exp, query_analysis=qa)["context_digest"] != first

        # 被预算丢掉的单元不参与摘要
        starved = ret.build_context(hits, exp, token_budget=1, query_analysis=qa)
        assert starved["units"] == []
        assert starved["context_digest"] != first

    def test_missing_chunk_is_skipped_not_fabricated(self):
        ret, qa = _make_retriever(chunks=[_chunk("doc/a#000", CHUNK_A)])
        hits = [{"chunk_id": "doc/zzz#000", "score": 1.0, "sources": []}]
        ctx = ret.build_context(hits, [], query_analysis=qa)
        assert ctx["units"] == []
        assert ctx["stats"]["skipped_units"] == [
            {"chunk_id": "doc/zzz#000", "reason": "chunk_not_loaded"}
        ]

    def test_units_carry_everything_citation_checking_needs(self):
        ret, qa = _make_retriever()
        hits = ret.search(qa, top_k=5)["hits"]
        exp = ret.expand([ENT_KAL], depth=1, query_analysis=qa)["expansions"]
        ctx = ret.build_context(hits, exp, query_analysis=qa)
        assert ctx["units"]
        for unit in ctx["units"]:
            for key in ("chunk_id", "doc_id", "title", "chapter", "text",
                        "evidence_spans", "entities", "assertions", "score",
                        "sources", "graph_path", "tokens"):
                assert key in unit
            for span in unit["evidence_spans"]:
                assert unit["text"][span["start"]:span["end"]] == span["evidence"]

    def test_context_is_json_serialisable(self):
        ret, qa = _make_retriever()
        hits = ret.search(qa, top_k=5)["hits"]
        exp = ret.expand([ENT_KAL], depth=1, query_analysis=qa)["expansions"]
        ctx = ret.build_context(hits, exp, query_analysis=qa)
        assert json.loads(json.dumps(ctx, ensure_ascii=False))["stats"]["units"] >= 1


class TestEstimateTokens:
    def test_monotonic_in_text_length(self):
        short = retrieve.estimate_tokens("凯尔希")
        long = retrieve.estimate_tokens("凯尔希" * 10)
        assert long > short

    def test_empty_text_costs_zero(self):
        assert retrieve.estimate_tokens("") == 0
