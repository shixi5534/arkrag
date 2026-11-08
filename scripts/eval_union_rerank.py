"""四种检索模式的同批对比：出质量数字，不是出延迟数字。

被比较的四条（同一代 g-20261103T132152Z-06a741、同一批题、同一输出窗 top_k=20）：

  base           现状生产链路：bm25 + 实体 + 说话人 + cooc（`Retriever.search`）
  vec_union      base 的 top20 与向量 top-N **取并集**（不排名次加权）
  base_rerank    base 的 top20 交给 CE 重排
  union_rerank   base 的 top20 ∪ 向量 top-N，交给 CE 重排   ← 这条从未被测过

`vec_union` 刻意不做 RRF：RRF 在本项目已被三轮否证（`docs/i阶段实测.md` §2.2–2.5），
而「并集 + 重排」是一条不同的接法——并集只负责"把候选拉进来"，排序交给 CE。
拿 RRF 的失败结论去否证它，是不成立的推理。

口径（与 `scripts/ask.py::evaluate` 一致）：
  · 每题 gold chunk 任一进入前 20 记 R@20 命中；前 5 记 R@5 命中。
  · MRR 取第一个 gold chunk 的名次倒数，未命中记 0。
  · 只有 R@20 与 MRR **同时不劣**才算改进（项目既有判据）。
  · 报 R@5 必附第 5 名转化率：gold 在第 5 名（而非更前）的题数 / 命中题数。

输出：`data/_tmp_union_rerank.json`，含四模式指标、逐题名次、延迟分位。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import (  # noqa: E402
    embedding,
    index as index_mod,
    pipeline,
    query as query_mod,
    retrieve,
    vector_store,
    vocab as vocab_mod,
)

try:
    from arkrag import rerank as rerank_mod
except Exception:  # pragma: no cover - 缺 onnxruntime 时明确报错
    rerank_mod = None

RERANK_DIR = Path(r"C:\Users\Lenovo\WorkBuddy\2026-07-27-16-45-03\roleplay-ai\models\bge-reranker-base-onnx")
TOP_K = 20
VECTOR_K = 20


def pct(values: list[float], q: float) -> float:
    """最近秩百分位，不插值（与 probe_rerank_opt 同口径）。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def rank_of(hits: list[dict], gold: set[str]) -> int | None:
    for i, h in enumerate(hits, 1):
        if str(h.get("chunk_id") or "") in gold:
            return i
    return None


def metrics(rows: list[dict], key: str, golds: dict[str, set[str]]) -> dict:
    n = len(rows)
    hit5 = hit20 = 0
    rr = 0.0
    fifth_only = 0
    for row in rows:
        rank = row[key]
        gold = golds[row["qid"]]
        if rank is None:
            continue
        if rank <= 20:
            hit20 += 1
        if rank <= 5:
            hit5 += 1
            if rank == 5:
                fifth_only += 1
        if rank:
            rr += 1.0 / rank
    return {
        "n": n,
        "recall@5": hit5 / n if n else 0.0,
        "recall@20": hit20 / n if n else 0.0,
        "mrr": rr / n if n else 0.0,
        # 第 5 名转化率：命中题里有几道是靠第 5 名才勉强进来的
        "rank5_share_of_hits": (fifth_only / hit20) if hit20 else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generation", default="g-20261103T132152Z-06a741")
    ap.add_argument("--gold-set", default="data/goldset/gold-v1.json")
    ap.add_argument("--limit", type=int, default=320)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--vector-k", type=int, default=VECTOR_K)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--intra-op", type=int, default=None)
    ap.add_argument("--use-int8", action="store_true")
    ap.add_argument("--modes", nargs="+",
                    default=["base", "vec_union", "base_rerank", "union_rerank"])
    ap.add_argument("--json", default="data/_tmp_union_rerank.json")
    args = ap.parse_args()

    if rerank_mod is None:
        print("读不到 arkrag.rerank（缺 onnxruntime/tokenizers？）", file=sys.stderr)
        return 2

    loaded = pipeline.load_generation(args.generation)
    index = loaded["index"]
    graph = loaded["graph"]
    chunks = loaded["chunks"]
    content_digest = (loaded["index_meta"] or {}).get("index_content_digest") or ""

    retriever = retrieve.Retriever(index, graph, {}, chunks=chunks, vocab=vocab_mod)
    entities = graph.get("entities") or []

    # 向量索引：缺失必须显式报错，不许静默退化
    model = embedding.model_name()
    bin_path, meta_path = vector_store.vector_paths(
        vector_store._slug(model), content_digest
    )
    if not (bin_path.exists() and meta_path.exists()):
        print(f"缺向量产物：{bin_path}", file=sys.stderr)
        return 3
    matrix, vmeta = vector_store.load_vectors(bin_path, meta_path)
    vec = vector_store.VectorIndex(matrix, vmeta)
    print(f"块 {len(chunks)}，实体 {len(retriever.entities_by_id)}，"
          f"向量 {len(vec.chunk_ids)}×{vec.dim} ({vec.model})", flush=True)

    need_rerank = any("rerank" in m for m in args.modes)
    reranker = None
    if need_rerank:
        t0 = time.perf_counter()
        reranker = rerank_mod.CrossEncoderReranker(
            RERANK_DIR,
            max_length=args.max_length,
            batch_size=args.batch_size,
            intra_op_threads=args.intra_op,
            use_int8=args.use_int8,
        )
        reranker.warmup()
        print(f"重排就绪 {reranker.info()} 加载 {1000*(time.perf_counter()-t0):.1f}ms",
              flush=True)

    gold = json.loads((ROOT / args.gold_set).read_text(encoding="utf-8"))
    # gold-v1/v2 的题在 `items`；兼容裸列表形态
    raw = gold.get("items") if isinstance(gold, dict) else gold
    items = [g for g in (raw or []) if g.get("gold_chunk_ids")]
    items = items[: args.limit]
    print(f"题 {len(items)}（有 gold_chunk_ids）", flush=True)

    golds: dict[str, set[str]] = {}
    rows: list[dict] = []
    lat: dict[str, list[float]] = {m: [] for m in args.modes}

    for i, item in enumerate(items, 1):
        qid = str(item.get("question_id") or i)
        q = str(item.get("question") or "")
        gset = {str(c) for c in item["gold_chunk_ids"]}
        golds[qid] = gset

        qa = query_mod.analyze_query(
            q, entities=entities, aliases=graph.get("alias_index"), vocab=vocab_mod
        )

        t0 = time.perf_counter()
        found = retriever.search(qa, top_k=args.top_k)
        base_lat = 1000 * (time.perf_counter() - t0)
        base_hits = found["hits"]
        lat["base"].append(base_lat)

        base_ids = [str(h["chunk_id"]) for h in base_hits]

        t0 = time.perf_counter()
        qv = embedding.embed_batch([q])[0]
        vhits = vec.search(qv, top_k=args.vector_k)
        vec_lat = 1000 * (time.perf_counter() - t0)

        # 并集：base 顺序在前（保留生产链路的排序偏好），向量新面孔追加在后
        union_ids = list(base_ids)
        seen = set(union_ids)
        for cid, _score in vhits:
            cid = str(cid)
            if cid not in seen:
                seen.add(cid)
                union_ids.append(cid)

        row = {
            "qid": qid,
            "gold_hit_base": bool(gset & set(base_ids)),
            "gold_hit_vec": bool(gset & {str(c) for c, _ in vhits}),
            "gold_hit_union": bool(gset & set(union_ids)),
            "pool_base": len(base_ids),
            "pool_union": len(union_ids),
            "base": rank_of(base_hits, gset),
        }

        def doc_text(cid: str) -> str:
            rec = chunks.get(cid) or {}
            return vector_store.chunk_text_for_embedding(rec)

        if "vec_union" in args.modes:
            t0 = time.perf_counter()
            lat_key = base_lat + vec_lat + 0.0
            lat["vec_union"].append(lat_key)
            row["vec_union"] = rank_of(
                [{"chunk_id": c} for c in union_ids], gset
            )

        if reranker is not None:
            if "base_rerank" in args.modes:
                t0 = time.perf_counter()
                order = reranker.rerank(q, [doc_text(c) for c in base_ids])
                lat["base_rerank"].append(1000 * (time.perf_counter() - t0))
                ranked = [base_ids[j] for j in order]
                row["base_rerank"] = rank_of([{"chunk_id": c} for c in ranked], gset)

            if "union_rerank" in args.modes:
                t0 = time.perf_counter()
                order = reranker.rerank(q, [doc_text(c) for c in union_ids])
                lat["union_rerank"].append(1000 * (time.perf_counter() - t0))
                ranked = [union_ids[j] for j in order]
                row["union_rerank"] = rank_of([{"chunk_id": c} for c in ranked], gset)

        rows.append(row)
        if i % 20 == 0 or i == len(items):
            print(f"  {i}/{len(items)}  "
                  f"base R@20={sum(1 for r in rows if r['base'] and r['base']<=20)/len(rows):.4f}",
                  flush=True)

    result = {
        "generation": args.generation,
        "gold_set": args.gold_set,
        "questions": len(items),
        "top_k": args.top_k,
        "vector_k": args.vector_k,
        "vector": {"model": vec.model, "dim": vec.dim, "count": len(vec.chunk_ids),
                   "digest": content_digest[:16]},
        "rerank": reranker.info() if reranker is not None else None,
        "metrics": {m: metrics(rows, m, golds) for m in args.modes},
        "latency": {
            m: {"p50_ms": pct(v, 0.5), "p95_ms": pct(v, 0.95),
                "mean_ms": statistics.fmean(v) if v else 0.0, "n": len(v)}
            for m, v in lat.items()
        },
        "pool": {
            "base_mean": statistics.fmean([r["pool_base"] for r in rows]),
            "union_mean": statistics.fmean([r["pool_union"] for r in rows]),
            "union_added_mean": statistics.fmean(
                [r["pool_union"] - r["pool_base"] for r in rows]
            ),
        },
        "rescue": {
            "base_only": sum(1 for r in rows if r["gold_hit_base"] and not r["gold_hit_vec"]),
            "vec_only": sum(1 for r in rows if r["gold_hit_vec"] and not r["gold_hit_base"]),
            "both": sum(1 for r in rows if r["gold_hit_base"] and r["gold_hit_vec"]),
            "neither": sum(1 for r in rows if not r["gold_hit_base"] and not r["gold_hit_vec"]),
            "union": sum(1 for r in rows if r["gold_hit_union"]),
        },
        "rows": rows,
    }

    out = ROOT / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n=== 质量 ===")
    print(f"{'模式':<14}{'R@5':>9}{'R@20':>9}{'MRR':>9}{'第5名转化':>11}")
    for m in args.modes:
        s = result["metrics"][m]
        print(f"{m:<14}{s['recall@5']:>9.4f}{s['recall@20']:>9.4f}"
              f"{s['mrr']:>9.4f}{s['rank5_share_of_hits']:>11.4f}")
    print("\n=== 延迟 ===")
    for m in args.modes:
        s = result["latency"][m]
        print(f"{m:<14}P50 {s['p50_ms']:>8.1f}ms   P95 {s['p95_ms']:>8.1f}ms")
    print("\n=== 候选池 ===")
    print(f"base 均值 {result['pool']['base_mean']:.1f}，"
          f"并集均值 {result['pool']['union_mean']:.1f}，"
          f"向量净增 {result['pool']['union_added_mean']:.1f}")
    print("\n=== gold 归属（top 窗内）===")
    r = result["rescue"]
    print(f"仅 base {r['base_only']} / 仅向量 {r['vec_only']} / 都有 {r['both']} / "
          f"都没有 {r['neither']} / 并集 {r['union']}")
    print(f"\n落盘 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
