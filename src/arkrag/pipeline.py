"""检索问答管线的装配与单问执行——**唯一**一份实现。

为什么把它从 `scripts/ask.py` 搬进包内：J 阶段要加一个 HTTP 服务，服务必须
和命令行用**同一条**链路。如果 `scripts/serve.py` 自己写一遍"读哪些产物、
缺哪个算不合格、按什么顺序调 retriever/answerer"，那么"哪些产物是必需的"
这件事就有两份定义——本项目已经因为"同一判定写两处、取最松的那份生效"
出过三次同型事故（见 `tests/test_single_source_of_truth.py`）。

所以这里只做**搬移**，不改任何判定：

* `load_generation()` 的产物清单与检查顺序原样保留；
* `build_pipeline()` 的构造参数原样保留；
* `ask_once()` 的分支与调用顺序原样保留（`timings` 仍是可选旁路）。

`scripts/ask.py` 改成从这里 import 同名函数并继续对外暴露（`ask.load_generation`
等名字不变），所以既有的 `scripts/bench_h1.py` / `scripts/bench_h3.py` 里
`ask_mod.load_generation = …` 这类 monkeypatch 仍然生效——它们改的是
`scripts/ask.py` 模块对象上的名字，而 `ask.evaluate()` 是在 `ask.py` 里定义的，
解析的正是该模块的全局名。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import answer as answer_mod
from . import config
from . import index as index_mod
from . import query as query_mod
from . import release
from . import retrieve as retrieve_mod
from . import vocab


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    try:
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[2] {path} 不是合法 JSONL：{exc}") from exc
    return out


# ---------------------------------------------------------------- 块正文来源
class ChunksUnavailable(RuntimeError):
    """块正文找不到、或找到的那份证明不了是这一代吃进去的那批。

    为什么必须是异常而不是返回 None：块正文是**引用校验的判据**。
    拿错块不会报错——`text[start:end] != evidence` 只会让引用被静默剔除、
    置信度降低，表现为「答得短」。这种形态没人会去查，必须在这里拒。
    """


def resolve_chunks(gen, manifest: dict) -> Path:
    """决定这一代该从哪个文件读块正文，并**证明**那份文件就是它那一批。

    两代形态：

    1. 世代自带 `<gen>/chunks.jsonl`（K2 起的新形态）。直接用它，但仍要核
       `chunk_manifest_hash`——文件在盘上不等于内容对。
    2. 旧世代没有自带块文件。此时**不许默认"全局块文件就是它的"**：全局
       `data/arkrag/chunks.jsonl` 是 L1 的最新产物，下一次 L2 重建就会覆盖它，
       而世代不会跟着变。只有当它的 T1 子集能复现本代 `chunk_manifest_hash`
       时才允许借用——那不是猜测，是同一批块的证明。

    两种情况都证明不了就抛 `ChunksUnavailable`，由调用方决定怎么呈现。
    """
    want = str(manifest.get("chunk_manifest_hash") or "")

    def t1_of(records: list[dict]) -> list[dict]:
        return [r for r in records if config.in_t1(str(r.get("category") or ""))]

    own = gen.chunks_path
    if own.exists():
        records = t1_of(_read_jsonl(own))
        if want:
            got = release.chunk_manifest_hash(records)
            if got != want:
                raise ChunksUnavailable(
                    f"世代自带块正文与 Manifest 不符："
                    f"chunk_manifest_hash Manifest={want[:16]} 实际={got[:16]}"
                    f"（{own}，{len(records)} 块）"
                )
        if not records:
            raise ChunksUnavailable(f"世代自带块正文是空的：{own}")
        return own

    # ---- 旧世代：只有能证明是同一批块时才借全局文件
    shared = Path(config.chunks_path())
    if not shared.exists():
        raise ChunksUnavailable(f"找不到块文件 {shared}（引用校验需要块正文）")
    records = t1_of(_read_jsonl(shared))
    if want:
        got = release.chunk_manifest_hash(records)
        if got != want:
            raise ChunksUnavailable(
                f"这一代的块正文不在 {shared} 里："
                f"chunk_manifest_hash Manifest={want[:16]} 该文件实测={got[:16]}"
                f"（差 {len(records)} 块的那批）。拒绝用错块做引用校验——"
                "请给这一代补上自带的 chunks.jsonl，或从中断点重建。"
            )
    return shared


def load_generation(generation_id: str) -> dict:
    """读出评测/回答需要的全部产物。缺任何一件都抛 `SystemExit(2)`。

    这里刻意**不**做"缺断言就用边顶替"的降级：`ark-rag-3` 的回答要靠逐观察断言
    给出证据偏移，缺了它只能给出没有位置的引用，那种引用无法机检。
    """
    gen = release.Generation(generation_id)
    if not gen.dir.exists():
        raise SystemExit(f"[2] 世代不存在：{generation_id}（{gen.dir}）")
    manifest = gen.load()
    schema = str(manifest.get("graph_schema") or config.GRAPH_SCHEMA_LEGACY)
    if schema != config.GRAPH_SCHEMA_IDENTITY:
        raise SystemExit(
            f"[2] 世代 {generation_id} 的 graph_schema={schema}，"
            f"本工具只支持 {config.GRAPH_SCHEMA_IDENTITY}（ark-rag-1 没有断言，"
            "做不了带方向的扩展与可定位引用）"
        )
    for path, label in (
        (gen.graph_path, "graph/graph.json"),
        (gen.assertion_path, "graph/assertions.jsonl"),
        (gen.adjacency_path, "graph/adjacency.json"),
        (gen.index_path, "index/lexical.json"),
    ):
        if not path.exists():
            raise SystemExit(f"[2] 世代 {generation_id} 缺 {label}（{path}）")

    graph = json.loads(gen.graph_path.read_text("utf-8"))
    adjacency = json.loads(gen.adjacency_path.read_text("utf-8"))
    assertions = _read_jsonl(gen.assertion_path)
    idx, idx_meta = index_mod.load_index(gen.index_path)

    try:
        chunks_path = resolve_chunks(gen, manifest)
    except ChunksUnavailable as exc:
        raise SystemExit(f"[2] {exc}") from exc
    chunks: dict[str, dict] = {}
    for rec in _read_jsonl(chunks_path):
        cid = str(rec.get("chunk_id") or "")
        if cid:
            chunks[cid] = rec

    return {
        "generation_id": generation_id,
        "generation": gen,
        "manifest": manifest,
        "graph": graph,
        "adjacency": adjacency,
        "assertions": assertions,
        "index": idx,
        "index_meta": idx_meta,
        "chunks": chunks,
    }


# ---------------------------------------------------------------- 管线
def build_pipeline(loaded: dict, *, top_k: int, graph_depth: int) -> dict:
    retriever = retrieve_mod.Retriever(
        loaded["index"],
        loaded["graph"],
        loaded["adjacency"],
        chunks=loaded["chunks"],
        assertions=loaded["assertions"],
        vocab=vocab,
    )
    answerer = answer_mod.ExtractiveAnswerer()
    return {"retriever": retriever, "answerer": answerer, "top_k": top_k,
            "graph_depth": graph_depth}


def ask_once(pipeline: dict, question: str, *, timings: dict | None = None) -> dict:
    """一问一答的完整链路。返回答案 + 每一层的中间量（供排查与评测）。

    `timings` 传入一个 dict 时，逐阶段毫秒数写进它（H1 性能基线要用）。
    默认 `None` → 不测时间，返回结构与原来完全一致：**计时是可选旁路，
    不是返回值的一部分**，否则每个读 `result["search"]` 的调用方都得改。
    """
    retriever = pipeline["retriever"]
    entities = pipeline.get("entities") or retriever.graph.get("entities") or []
    _t0 = time.perf_counter()
    qa = query_mod.analyze_query(
        question,
        entities=entities,
        aliases=retriever.graph.get("alias_index"),
        vocab=vocab,
        # 复用 `Retriever` 建好的只读结构。**这一行是 H2 的主要提速点**
        # （每问省约 98 ms 的图级重建，见 `query.QueryContext` 的实测表）。
        # 必须传 `retriever.query_context` 而不是自己造：那样又变回每问重建。
        context=getattr(retriever, "query_context", None),
    )
    if timings is not None:
        timings["query_ms"] = (time.perf_counter() - _t0) * 1000.0
    if not qa.get("should_answer", True):
        if timings is not None:
            timings["total_ms"] = (time.perf_counter() - _t0) * 1000.0
            timings["abstained_by_query"] = 1
        return {
            "query_analysis": qa,
            "search": {"hits": [], "stats": {"skipped_reason": qa.get("reason")}},
            "expand": {"expansions": [], "stats": {"skipped_reason": qa.get("reason")}},
            "context": {"units": [], "context_digest": "", "stats": {}},
            "answer": pipeline["answerer"].answer(
                question, {"units": [], "context_digest": ""},
                generation_id=pipeline.get("generation_id", ""),
                top_score=0.0, query_analysis=qa,
            ),
        }

    found = retriever.search(qa, top_k=pipeline["top_k"])
    if timings is not None:
        timings["search_ms"] = (time.perf_counter() - _t0) * 1000.0 - timings["query_ms"]
    _t1 = time.perf_counter()
    seed_ids = [c["entity_id"] for c in qa.get("entity_candidates") or []]
    expanded = retriever.expand(
        seed_ids,
        depth=pipeline["graph_depth"],
        query_analysis=qa,
        seed_chunk_ids=[h["chunk_id"] for h in found["hits"]],
    )
    if timings is not None:
        timings["expand_ms"] = (time.perf_counter() - _t1) * 1000.0
        timings["expand_nodes"] = int(expanded["stats"].get("nodes_visited") or 0)
        timings["expand_edges"] = int(expanded["stats"].get("expansions") or 0)
    _t2 = time.perf_counter()
    context = retriever.build_context(found["hits"], expanded["expansions"], query_analysis=qa)
    if timings is not None:
        timings["context_ms"] = (time.perf_counter() - _t2) * 1000.0
    _t3 = time.perf_counter()
    ans = pipeline["answerer"].answer(
        question,
        context,
        generation_id=pipeline.get("generation_id", ""),
        top_score=float(found["stats"].get("top_score") or 0.0),
        query_analysis=qa,
        retrieval_stats=found["stats"],
    )
    if timings is not None:
        timings["answer_ms"] = (time.perf_counter() - _t3) * 1000.0
        timings["total_ms"] = (time.perf_counter() - _t0) * 1000.0
        timings["context_tokens"] = int(context["stats"].get("tokens_used") or 0)
        timings["context_units"] = int(context["stats"].get("units") or 0)
        timings["candidates"] = int(found["stats"].get("candidates") or 0)
    return {
        "query_analysis": qa,
        "search": found,
        "expand": expanded,
        "context": context,
        "answer": ans,
    }


__all__ = ["ask_once", "build_pipeline", "load_generation"]
