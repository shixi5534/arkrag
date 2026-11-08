"""向量通道接线方式的门：**触发式尾部并集**。

## 这个文件在钉什么

向量通道必须常驻生产链路（用户硬约束），但「怎么接」不是自选动作——所有
**融合**式接法都已被 300 题实测否证（minmax 加权 / RRF k=1,2 / DBSF /
DBSF 尾部补位 / 头部锁定 + 向量重排余下），唯一不会伤到已答对 237 道题的
接法是「只在触发时把向量结果**追加到 base 尾部**」：追加块名次必然 ≥21，
base 已命中的题 gold 名次只可能不变 ⇒ R@20 单调不降。

所以这里钉四类东西：

1. **不触发时一个字节都不许变**。触发判据是 `top_score / score20 <= cut`；
   不触发时不许调 `embed_query`（那是一次 HTTP），也不许动 hits。
2. **触发时只许追加，不许重排**。已存在的 chunk 不许被改名次，新块
   `score` 必须是 0.0（向量余弦与归一化 bm25 不同量纲，混分就是被否证的融合）。
3. **`sources` 的键必须是 `channel`**。`Retriever.build_context` 的
   `direct()` 读 `s["channel"]`；写成 `source` 会在真实链路上 KeyError，
   而单测里如果只断言"有个 dict"就发现不了。
4. **embedding 的硬错误必须向上抛**。`EmbeddingError` 不许被吞成"没有向量"，
   否则「带向量的评测」和「不带向量的评测」会长得一模一样。

第 4 条是本项目栽过的坑：`index.build` 曾把 `vector_channels` 写成
`"not_built_no_embedding_backend"` 而没人发现，结果是台账里写着"有向量通道"、
`docs/baseline-metrics.md` 却明写"对外不能说具备语义检索"。
"""

from __future__ import annotations

import pytest

from arkrag import pipeline as pipeline_mod
from arkrag import vector_union as vu


# ------------------------------------------------------------------ 假件
class FakeVectorIndex:
    """只实现 `search`，够 `TriggeredVectorUnion` 用；不依赖 numpy。"""

    def __init__(self, results: list[tuple[str, float]], *, model="fake-embed",
                 dim=4):
        self.results = list(results)
        self.model = model
        self.dim = dim
        self.chunk_ids = [cid for cid, _ in self.results]
        self.queries: list[list[float]] = []

    def search(self, query_vector, *, top_k=20):
        self.queries.append(list(query_vector))
        return self.results[:top_k]


def _hits(n: int, *, top: float = 10.0, last: float = 1.0) -> list[dict]:
    """线性衰减的 hits：第 1 名 top，第 n 名 last。"""
    if n == 1:
        return [{"chunk_id": "c#0", "score": top}]
    step = (top - last) / (n - 1)
    return [{"chunk_id": f"c#{i:03d}", "score": top - step * i} for i in range(n)]


def _embedder(vec=(1.0, 0.0, 0.0, 0.0)):
    calls = []

    def embed(text):
        calls.append(text)
        return list(vec)

    embed.calls = calls
    return embed


# ------------------------------------------------------------------ 信号
class TestTriggerSignal:
    def test_ratio_is_top_over_twentieth(self):
        hits = _hits(20, top=10.0, last=2.0)
        assert vu.ratio_top_to_20(hits) == pytest.approx(5.0)

    def test_score_at_falls_back_to_last_when_shorter(self):
        hits = _hits(3, top=3.0, last=1.0)
        assert vu.score_at(hits, 20) == pytest.approx(1.0)
        assert vu.score_at(hits, 1) == pytest.approx(3.0)

    def test_ratio_is_infinite_when_twentieth_score_is_zero(self):
        # 分数衰减到底 = 完全没有把握，必须触发，不能被 0 除成 inf 之外的东西
        hits = _hits(20, top=10.0, last=0.0)
        assert vu.ratio_top_to_20(hits) == float("inf")
        assert vu.should_augment(hits) is False  # inf > cut，反而**不**触发

    def test_empty_hits_never_trigger(self):
        assert vu.ratio_top_to_20([]) == float("inf")
        assert vu.should_augment([]) is False

    def test_flat_scores_trigger(self):
        # 前 20 名同分 ⇒ ratio=1.0 ≤ cut ⇒ 排序毫无把握 ⇒ 触发
        hits = [{"chunk_id": f"c#{i:03d}", "score": 5.0} for i in range(20)]
        assert vu.should_augment(hits) is True


# ------------------------------------------------------------------ 主行为
class TestTriggeredTailUnion:
    def test_not_triggered_keeps_hits_and_never_embeds(self):
        idx = FakeVectorIndex([("v#1", 0.9)])
        embed = _embedder()
        u = vu.TriggeredVectorUnion(idx, embed_query=embed, cut=2.0)
        hits = _hits(20, top=10.0, last=1.0)  # ratio=10 > 2 → 不触发
        out = u.augment(hits, "问句")
        assert out["used"] is False
        assert out["reason"] == "not_triggered"
        assert out["hits"] == hits
        assert embed.calls == []
        assert idx.queries == []

    def test_triggered_appends_only_new_chunks_at_score_zero(self):
        idx = FakeVectorIndex([("c#000", 0.99), ("v#1", 0.80), ("v#2", 0.70)])
        u = vu.TriggeredVectorUnion(idx, embed_query=_embedder(), cut=2.0)
        hits = _hits(20, top=10.0, last=6.0)  # ratio≈1.67 ≤ 2 → 触发
        out = u.augment(hits, "问句")
        assert out["used"] is True and out["reason"] == "triggered"
        assert [h["chunk_id"] for h in out["hits"]] == (
            [h["chunk_id"] for h in hits] + ["v#1", "v#2"]
        )
        # c#000 已存在，不许重复追加
        assert out["added"] == ["v#1", "v#2"]
        for h in out["hits"][20:]:
            assert h["score"] == 0.0

    def test_base_head_order_and_scores_are_untouched(self):
        idx = FakeVectorIndex([("v#1", 0.9)])
        u = vu.TriggeredVectorUnion(idx, embed_query=_embedder(), cut=2.0)
        hits = _hits(20, top=10.0, last=6.0)
        before = [(h["chunk_id"], h["score"]) for h in hits]
        out = u.augment(hits, "问句")
        assert [(h["chunk_id"], h["score"]) for h in out["hits"][:20]] == before

    def test_input_hits_list_is_not_mutated(self):
        idx = FakeVectorIndex([("v#1", 0.9)])
        u = vu.TriggeredVectorUnion(idx, embed_query=_embedder(), cut=2.0)
        hits = _hits(20, top=10.0, last=6.0)
        snapshot = [dict(h) for h in hits]
        u.augment(hits, "问句")
        assert hits == snapshot

    def test_added_blocks_use_channel_key_for_build_context(self):
        # `build_context` 的 direct() 读 s["channel"]；写成 s["source"] 会 KeyError
        idx = FakeVectorIndex([("v#1", 0.9)])
        u = vu.TriggeredVectorUnion(idx, embed_query=_embedder(), cut=2.0)
        out = u.augment(_hits(20, top=10.0, last=6.0), "问句")
        src = out["hits"][-1]["sources"][0]
        assert src["channel"] == vu.SOURCE_VECTOR_UNION
        assert "source" not in src

    def test_every_vector_hit_already_in_base_is_not_an_error(self):
        # 向量 top-N 全落在 base 已召回的那批里：used=True 但 added=[]
        idx = FakeVectorIndex([("c#000", 0.9), ("c#001", 0.8)])
        u = vu.TriggeredVectorUnion(idx, embed_query=_embedder(), cut=2.0)
        out = u.augment(_hits(20, top=10.0, last=6.0), "问句")
        assert out["used"] is True
        assert out["added"] == []
        assert len(out["hits"]) == 20

    def test_query_text_is_what_gets_embedded(self):
        idx = FakeVectorIndex([("v#1", 0.9)])
        embed = _embedder()
        u = vu.TriggeredVectorUnion(idx, embed_query=embed, cut=2.0)
        u.augment(_hits(20, top=10.0, last=6.0), "凯尔希和罗德岛什么关系")
        assert embed.calls == ["凯尔希和罗德岛什么关系"]
        assert idx.queries == [[1.0, 0.0, 0.0, 0.0]]

    def test_embedding_error_propagates_never_swallowed(self):
        class EmbeddingError(RuntimeError):
            pass

        def boom(_text):
            raise EmbeddingError("ollama 挂了")

        u = vu.TriggeredVectorUnion(FakeVectorIndex([("v#1", 0.9)]),
                                    embed_query=boom, cut=2.0)
        with pytest.raises(EmbeddingError):
            u.augment(_hits(20, top=10.0, last=6.0), "问句")

    def test_stats_are_counted(self):
        idx = FakeVectorIndex([("v#1", 0.9)])
        u = vu.TriggeredVectorUnion(idx, embed_query=_embedder(), cut=2.0)
        u.augment(_hits(20, top=10.0, last=6.0), "问句")   # 触发
        u.augment(_hits(20, top=10.0, last=1.0), "问句")   # ratio=10 不触发
        info = u.info()
        assert info["mode"] == "triggered_tail_union"
        assert info["calls"] == 1 and info["triggered"] == 1
        assert info["added_total"] == 1
        assert info["vector_model"] == "fake-embed"

    def test_empty_base_hits_do_not_trigger(self):
        u = vu.TriggeredVectorUnion(FakeVectorIndex([("v#1", 0.9)]),
                                    embed_query=_embedder(), cut=2.0)
        out = u.augment([], "问句")
        assert out["used"] is False and out["reason"] == "no_base_hits"

    def test_constructor_rejects_bad_arguments(self):
        with pytest.raises(TypeError):
            vu.TriggeredVectorUnion(FakeVectorIndex([]), embed_query=None)
        with pytest.raises(ValueError):
            vu.TriggeredVectorUnion(FakeVectorIndex([]),
                                    embed_query=_embedder(), vector_k=0)


# ------------------------------------------------------------------ 管线接线
class _StubRetriever:
    """只够 `ask_once` 跑通的最小 retriever，用于验证接线而不是验证检索。"""

    def __init__(self, hits):
        self.graph = {"entities": [], "alias_index": {}}
        self.query_context = None
        self._hits = hits
        self.seen_top_k = None

    def search(self, query_analysis, *, top_k=20):
        self.seen_top_k = top_k
        return {"hits": [dict(h) for h in self._hits], "stats": {"top_score": 1.0,
                                                                "candidates": len(self._hits)}}

    def expand(self, entity_ids, *, depth=1, query_analysis=None, seed_chunk_ids=None):
        return {"expansions": [], "stats": {}}

    def build_context(self, hits, expansions, *, token_budget=None, query_analysis=None):
        units = [{"chunk_id": h["chunk_id"], "text": "x" * 20, "title": "",
                  "chapter": "", "sources": h.get("sources") or [],
                  "score": h.get("score") or 0.0, "assertions": [],
                  "evidence_spans": []} for h in hits]
        # 这里**真的**按 direct() 的判据读 s["channel"]——键写错就在这一步炸
        for u in units:
            any(s["channel"] in ("bm25", "entity") for s in u["sources"])
        return {"units": units, "context_digest": "d" * 64,
                "stats": {"units": len(units), "tokens_used": 1}}


class _StubAnswerer:
    def answer(self, question, context, **kwargs):
        return {"answer": "", "abstained": True, "citations": []}


def _stub_pipeline(hits, *, vector_union=None):
    return {"retriever": _StubRetriever(hits), "answerer": _StubAnswerer(),
            "top_k": 20, "graph_depth": 1, "vector_union": vector_union}


class TestPipelineWiring:
    def test_build_pipeline_defaults_to_no_vector_union(self):
        # 40 多个历史脚本都在调 build_pipeline；默认接向量会静默改掉它们的读数
        loaded = {
            "index": object(), "graph": {"entities": []}, "adjacency": {},
            "chunks": {}, "assertions": {},
        }
        pipe = pipeline_mod.build_pipeline(loaded, top_k=20, graph_depth=1)
        assert pipe["vector_union"] is None

    def test_ask_once_without_union_has_no_vector_stats(self):
        pipe = _stub_pipeline(_hits(20, top=10.0, last=6.0))
        res = pipeline_mod.ask_once(pipe, "凯尔希是谁")
        assert "vector_union" not in res["search"]["stats"]
        assert len(res["search"]["hits"]) == 20

    def test_ask_once_with_union_attaches_stats_and_grows_hits(self):
        idx = FakeVectorIndex([("v#1", 0.9)])
        u = vu.TriggeredVectorUnion(idx, embed_query=_embedder(), cut=2.0)
        pipe = _stub_pipeline(_hits(20, top=10.0, last=6.0), vector_union=u)
        res = pipeline_mod.ask_once(pipe, "凯尔希是谁")
        stats = res["search"]["stats"]["vector_union"]
        assert stats["used"] is True and stats["added"] == 1
        assert stats["cut"] == 2.0
        assert len(res["search"]["hits"]) == 21
        # R@20 的定义域不变：前 20 名一个都没动
        assert [h["chunk_id"] for h in res["search"]["hits"][:20]] == [
            f"c#{i:03d}" for i in range(20)
        ]

    def test_ask_once_with_union_but_not_triggered_is_noop(self):
        idx = FakeVectorIndex([("v#1", 0.9)])
        u = vu.TriggeredVectorUnion(idx, embed_query=_embedder(), cut=2.0)
        pipe = _stub_pipeline(_hits(20, top=10.0, last=1.0), vector_union=u)
        res = pipeline_mod.ask_once(pipe, "凯尔希是谁")
        stats = res["search"]["stats"]["vector_union"]
        assert stats["used"] is False and stats["reason"] == "not_triggered"
        assert len(res["search"]["hits"]) == 20

    def test_union_runs_before_expand_so_seeds_include_added_chunks(self):
        # 补进来的块必须参与图扩展，否则"白补"
        idx = FakeVectorIndex([("v#1", 0.9)])
        u = vu.TriggeredVectorUnion(idx, embed_query=_embedder(), cut=2.0)
        pipe = _stub_pipeline(_hits(20, top=10.0, last=6.0), vector_union=u)
        seen = {}

        class _SpyRetriever(_StubRetriever):
            def expand(self, entity_ids, *, depth=1, query_analysis=None,
                       seed_chunk_ids=None):
                seen["seeds"] = list(seed_chunk_ids or [])
                return {"expansions": [], "stats": {}}

        pipe["retriever"] = _SpyRetriever(_hits(20, top=10.0, last=6.0))
        pipeline_mod.ask_once(pipe, "凯尔希是谁")
        assert "v#1" in seen["seeds"]
