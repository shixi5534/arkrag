"""W1 门：`build_context(max_direct_units=)` 与它在 `pipeline` 里的门控。

## 这个文件守的是什么

W1 要**把「检索宽度」与「上下文宽度」解耦**：只限制*直接证据单元进入上下文*
的条数，不动检索召回了多少、也不动桥接的候选池。

三件事各自都有一种静默失效的方式，所以分别钉：

1. **默认路径必须逐位等价。** `max_direct_units` 默认 0，此时 `kept` 的
   chunk_id 序列、`stats` 的每个键、`context_digest` 都必须和改造前一致。
   这一条靠"同一输入跑两次、一次不传参一次显式传 0"来比对。
   失效方式：改动顺手把默认路径也换成 `break`，历史读数集体漂移而无人察觉。

2. **cap 路径必须是「有序前缀 + 首个超预算即停」。** 原实现在装不下时
   `continue` 试下一个更小的单元，于是名次更差的单元能挤进来（台账里
   193/350 那道悬案）。cap 路径承诺 `truncation_overtake == 0`。
   失效方式：把 cap 写成"过滤掉超出前 N 的"、装填仍用 `continue`，
   表面上条数对了，逆序照旧。

3. **桥接候选池不受 cap 影响。** `pipeline` 把 `context["units"]` 交给
   `bridge.find_bridge`；若只砍 `units`，桥接候选会被一并砍小，
   "解耦"就成了空话。`bridge_candidates` 必须保存 cap **之前**的完整集合。
   失效方式：`bridge_candidates` 在 cap 之后才拍快照。

## 为什么用合成语料而不是真实世代

这些门断言的是**装配逻辑的代数性质**（前缀性、等价性、候选集合不变），
不是分数。合成语料可控、毫秒级、不依赖 300 MB 产物，也不会有 OOM 风险。
真实世代的分数由 `scripts/eval_answer_quality.py` 负责，那是另一类证据。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from arkrag import adjacency as adjacency_mod
from arkrag import config
from arkrag import index as index_mod
from arkrag import pipeline as pipeline_mod
from arkrag import query as query_mod
from arkrag import retrieve
from arkrag import vocab

ROOT = Path(__file__).resolve().parents[1]

PAD = (
    "凯尔希：切尔诺伯格已经沦陷了，我们必须在天亮之前离开这里。"
    "阿米娅：可是罗德岛的人还在城里，我们不能丢下他们。"
    "凯尔希：博士，罗德岛不会为一个人的牺牲改变航向。"
)


def _text(n_chars: int) -> str:
    out: list[str] = []
    while len("".join(out)) < n_chars:
        out.append(PAD)
    return "".join(out)[:n_chars]


def _chunk(cid: str, text: str) -> dict:
    return {
        "chunk_id": cid,
        "doc_id": cid.split("#")[0],
        "title": "测试篇",
        "chapter": "第一章",
        "text": text,
        "chars": len(text),
    }


def _make(chunks: list[dict], *, budget: int = 12000):
    entities = [{"id": "e-0000000000000001", "name": "凯尔希",
                 "type": "角色", "aliases": []}]
    edges: list[dict] = []
    graph = {
        "schema_version": config.GRAPH_SCHEMA_IDENTITY,
        "entities": entities,
        "edges": edges,
        "alias_index": index_mod.build_alias_index(entities),
        "fingerprint": "0123456789abcdef",
        "stats": {"entities": 1, "edges": 0},
    }
    adj = adjacency_mod.build_directed_adjacency(edges)
    idx, _meta = index_mod.build(chunks, graph, "corpusfp01234567")
    ret = retrieve.Retriever(
        idx, graph, adj, chunks=chunks, assertions=[],
        budget={"max_context_tokens": budget}, vocab=vocab,
    )
    qa = query_mod.analyze_query("凯尔希和罗德岛是什么关系",
                                 entities=entities, vocab=vocab)
    return ret, qa


def _direct_hits(chunks: list[dict]) -> list[dict]:
    """全部当直接命中，分数递减以固定 `ordered` 的顺序。"""
    return [
        {"chunk_id": c["chunk_id"], "score": round(1.0 - i * 0.01, 4),
         "sources": [{"channel": retrieve.SOURCE_BM25, "score": 1.0}]}
        for i, c in enumerate(chunks)
    ]


def _graph_hits(chunks: list[dict]) -> list[dict]:
    """纯图扩展命中（只有 `graph` 通道 ⇒ 不算直接证据）。"""
    return [
        {"chunk_id": c["chunk_id"], "score": round(0.5 - i * 0.01, 4),
         "sources": [{"channel": retrieve.SOURCE_GRAPH, "score": 0.0}]}
        for i, c in enumerate(chunks)
    ]


def _ids(units: list[dict]) -> list[str]:
    return [str(u.get("chunk_id") or "") for u in units]


class TestDefaultPathIsBitIdentical:
    """检查 1：默认 0 = 与改造前逐位等价。"""

    def test_explicit_zero_equals_omitted(self):
        """显式传 `max_direct_units=0` 必须与完全不传逐位相同。

        这不是形式主义：`pipeline` 在 cap=0 时**不传这个关键字**，
        所以两条路径真的会同时存在。若它们不等价，那么"默认关"就是假的。
        """
        chunks = [_chunk("doc/a#%03d" % i, _text(600)) for i in range(12)]
        ret, qa = _make(chunks)
        hits = _direct_hits(chunks)

        omitted = ret.build_context(hits, [], query_analysis=qa)
        explicit = ret.build_context(hits, [], query_analysis=qa,
                                     max_direct_units=0)

        assert _ids(omitted["units"]) == _ids(explicit["units"])
        assert omitted["context_digest"] == explicit["context_digest"]
        assert omitted["stats"] == explicit["stats"]

    def test_no_new_stats_keys_when_cap_is_off(self):
        """cap=0 时不许出现 `max_direct_units` / `capped_units` / `bridge_candidates`。

        新键只允许出现在新开关开启的产物里（计划书"兼容性门禁"）。
        历史台账里有对 `stats` 做等值比较的读法，多一个键就会让它们全部失配。
        """
        chunks = [_chunk("doc/a#%03d" % i, _text(600)) for i in range(8)]
        ret, qa = _make(chunks)
        ctx = ret.build_context(_direct_hits(chunks), [], query_analysis=qa)

        assert "max_direct_units" not in ctx["stats"]
        assert "capped_units" not in ctx["stats"]
        assert "bridge_candidates" not in ctx

    def test_budget_override_still_identical(self):
        """带 `token_budget` 覆盖时也等价——历史脚本大量这样用。"""
        chunks = [_chunk("doc/a#%03d" % i, _text(600)) for i in range(12)]
        ret, qa = _make(chunks)
        hits = _direct_hits(chunks)

        a = ret.build_context(hits, [], token_budget=6000, query_analysis=qa)
        b = ret.build_context(hits, [], token_budget=6000, query_analysis=qa,
                              max_direct_units=0)
        assert _ids(a["units"]) == _ids(b["units"])
        assert a["context_digest"] == b["context_digest"]


class TestCapIsAnOrderedPrefix:
    """检查 2：cap 路径是「有序前缀 + 首个超预算即停」。"""

    def test_cap_limits_direct_units_to_n(self):
        """cap=3 时进入上下文的直接单元恰好是 ordered 的前 3 条。"""
        chunks = [_chunk("doc/a#%03d" % i, _text(400)) for i in range(12)]
        ret, qa = _make(chunks, budget=100000)  # 预算足够，只让 cap 起作用
        hits = _direct_hits(chunks)

        full = ret.build_context(hits, [], query_analysis=qa)
        capped = ret.build_context(hits, [], query_analysis=qa, max_direct_units=3)

        kept_full = [c for c in _ids(full["units"])]
        kept_capped = [c for c in _ids(capped["units"])]
        assert kept_capped == kept_full[:3], (
            f"cap=3 应当取 ordered 的前 3 条，实际 {kept_capped}")

    def test_capped_units_are_reported_not_dropped(self):
        """被 cap 排除的单元必须出现在 `stats["capped_units"]` 里。

        它们是"检索召回了但没进上下文"，不是"没召回"——两者混起来，
        台账就无法区分"检索不够"和"预算不够"。
        """
        chunks = [_chunk("doc/a#%03d" % i, _text(400)) for i in range(10)]
        ret, qa = _make(chunks, budget=100000)
        capped = ret.build_context(_direct_hits(chunks), [], query_analysis=qa,
                                   max_direct_units=4)
        assert len(capped["stats"]["capped_units"]) == 6
        assert capped["stats"]["max_direct_units"] == 4

    def test_cap_path_stops_at_first_over_budget_unit(self):
        """首个装不下的单元之后，**一个都不许再进**（逆序归零）。

        构造：预算刚好装下 2 个单元，第 3 个超预算，第 4 个又变小到能装下。
        旧的 `continue` 语义会让第 4 个挤进来；cap 路径必须停在第 3 个。
        这正是台账 193/350 那道"名次更差的单元反而进上下文"悬案的机制。
        """
        big = _text(2000)
        small = _text(100)
        chunks = [
            _chunk("doc/a#000", big),
            _chunk("doc/a#001", big),
            _chunk("doc/a#002", big),   # 超预算，应在此停止
            _chunk("doc/a#003", small),  # 能装下，但**不许**进
        ]
        ret, qa = _make(chunks)
        hits = _direct_hits(chunks)

        cost = retrieve.estimate_tokens(
            f"{chunks[0]['title']} {chunks[0]['chapter']} {chunks[0]['text']}",
            tokenizer_version=getattr(ret.index, "tokenizer_version",
                                      index_mod.TOKENIZER_VERSION),
            name_terms=getattr(ret.index, "name_terms", None),
        )
        # 预算：刚好 2 个 big，装不下第 3 个
        budget = cost * 2 + 1

        capped = ret.build_context(hits, [], token_budget=budget,
                                   query_analysis=qa, max_direct_units=10)
        kept = _ids(capped["units"])
        assert kept == ["doc/a#000", "doc/a#001"], (
            f"cap 路径必须首超即停，实际 {kept}——"
            f"若第 4 块（更小）进来了，说明逆序缺陷仍在")
        assert "doc/a#003" not in kept

    def test_truncation_overtake_is_zero_on_cap_path(self):
        """更形式化的同一断言：被砍的直接单元名次都劣于已装入的。"""
        chunks = [_chunk("doc/a#%03d" % i, _text(500 + i * 37)) for i in range(14)]
        ret, qa = _make(chunks)
        hits = _direct_hits(chunks)
        order = {c["chunk_id"]: i for i, c in enumerate(chunks)}

        capped = ret.build_context(hits, [], token_budget=4000,
                                   query_analysis=qa, max_direct_units=14)
        kept = _ids(capped["units"])
        dropped = set(capped["stats"]["truncated_chunks"]) - set(kept)
        if kept and dropped:
            worst_kept = max(order[c] for c in kept)
            best_dropped = min(order[c] for c in dropped)
            assert best_dropped > worst_kept, (
                f"被砍的名次 {best_dropped} 优于已装入的最差名次 {worst_kept}："
                f"cap 路径出现了逆序")


class TestBridgeCandidatePoolIsUnaffected:
    """检查 3：桥接候选池不受 cap 影响。"""

    def test_bridge_candidates_hold_the_full_grounded_set(self):
        """`bridge_candidates` = cap **之前**的全部单元（直接 + 图）。"""
        direct = [_chunk("doc/d#%03d" % i, _text(400)) for i in range(6)]
        graph = [_chunk("doc/g#%03d" % i, _text(400)) for i in range(4)]
        ret, qa = _make(direct + graph, budget=100000)

        capped = ret.build_context(
            _direct_hits(direct) + _graph_hits(graph), [],
            query_analysis=qa, max_direct_units=2)

        cand = _ids(capped["bridge_candidates"])
        assert len(capped["units"]) == 2 + 4, "上下文只留 2 个直接 + 4 个图单元"
        assert set(cand) == {c["chunk_id"] for c in direct + graph}, (
            "桥接候选池必须是 cap 之前的完整接地集合，否则 cap 会砍小桥接候选")
        assert len(cand) == 10

    def test_bridge_candidates_absent_when_cap_off(self):
        """cap=0 时**不许**有这个键（默认路径逐位等价的一部分）。"""
        chunks = [_chunk("doc/a#%03d" % i, _text(400)) for i in range(6)]
        ret, qa = _make(chunks, budget=100000)
        ctx = ret.build_context(_direct_hits(chunks), [], query_analysis=qa)
        assert "bridge_candidates" not in ctx

    def test_bridge_candidates_do_not_enter_context_digest(self):
        """摘要必须只描述回答器**真实看到**的证据。

        若 `bridge_candidates` 混进 `payload`，`context_digest` 就会声称
        回答器看到了它没被给的单元——那会让引用校验失去意义。
        """
        chunks = [_chunk("doc/a#%03d" % i, _text(400)) for i in range(8)]
        ret, qa = _make(chunks, budget=100000)
        hits = _direct_hits(chunks)

        capped = ret.build_context(hits, [], query_analysis=qa, max_direct_units=3)
        # 同一份"只看 3 条"的上下文，用 3 条直接命中重算，digest 应相等
        same = ret.build_context(hits[:3], [], query_analysis=qa)
        assert capped["context_digest"] == same["context_digest"], (
            "context_digest 受到了 bridge_candidates 的影响："
            "摘要只能描述实际装进上下文的单元")


class TestPipelineGating:
    """检查 4：`ask_once` 的门控——桥接题与章节题不 cap。"""

    def test_chapter_question_detection_uses_answer_constant(self):
        """`is_chapter_question` 直接转发 `answer.CHAPTER_TRIGGERS`，不另写一份。"""
        from arkrag import answer as answer_mod
        assert pipeline_mod.is_chapter_question("请给出它出现过的章节代号。") is True
        assert pipeline_mod.is_chapter_question("这个中间实体是谁？") is False
        # 常量改名后本门必须跟着红，而不是静默失效
        assert answer_mod.CHAPTER_TRIGGERS

    def test_max_direct_units_defaults_to_off_in_ask_once(self):
        """`ask_once` 的 options 默认值里不许出现 `max_direct_units`。

        默认开会让台账里 6000/12000/24000 三档的读数一起漂。
        """
        import inspect
        sig = inspect.signature(pipeline_mod.ask_once)
        assert sig.parameters["options"].default is None

    def test_prod_profile_does_not_enable_cap(self):
        """生产档**没有**开 cap——W1 还没验收，不许偷跑。"""
        assert "max_direct_units" not in pipeline_mod.PROD_PROFILE["options"]
