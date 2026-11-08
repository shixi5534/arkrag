"""第二轮融合：DBSF / 低 k RRF / 分位数拉伸，以及「融合结果 + CE 重排」。

为什么另开一个脚本（而不是改 eval_combo.py）
------------------------------------------------
eval_combo.py 的十种组合全部失败，失败原因是同一个：向量分数与 bm25 分数
不在一个量纲上，而 minmax 归一化会把「向量独有块」压成 0 分去和 base 末名的
0 分抢位。本脚本换三种不改量纲前提的做法，每种都先验证「改完后 base 相对顺序
是否还活着」：

1. ``dbsf``  —— Distribution-Based Score Fusion。两路各自用「均值 ± 标准差」
   归一，而不是 min/max。依据：Qdrant「How to Tune Hybrid Search」称 5 个数据
   集里 3 个显著优于 RRF，且保留分数间距信息。minmax 的病灶是单点极值决定
   整个量纲，DBSF 用标准差加权，极值影响被摊薄。
2. ``rrf_k2`` —— RRF 但 k=2（Qdrant 默认）。本项目历史扫过 k=60→1，k=1 最好
   仍 ΔMRR −0.0071；k=2 在扫描网格里被 k=1 与 k=60 夹住，但没有单独记录读数，
   所以这里补上，避免「扫过了」与「测过了」混为一谈。
3. ``dbsf_tail`` —— DBSF 只用于决定尾部 k 位，头部 base 顺序锁死。这是前项目
   「补位不整体重排」的结构（docs/i阶段-前项目经验检索.md:674-692）。

然后是重排：``<fusion>_rr10`` / ``_rr20`` 表示对融合结果的前 10 / 20 名做 CE 重排，
其余位置原样保留。Qdrant「When Is a Reranker Worth It」明确要求先测 10 个候选，
只有赢了才加深；本项目历史上只测过「整体重排 20 个」，没测过「只重排前 10」。

判据（沿用 docs/i阶段实测.md 的五条口径）
------------------------------------------
* 输出窗固定 top_k=20。
* R@20 与 MRR 必须同时不劣，才算候选改进。
* 不显著不采纳（配对 bootstrap，4000 次重采样）。
* 报 R@5 必须同时报第 5 名转化率（rank5_share_of_hits），否则会掩盖「前 5 名变多
  但转化变差」。
* 延迟按检索阶段口径报 P50/P95，重排单独标注（它与检索阶段预算不是一回事）。
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import embedding as embedding_mod  # noqa: E402
from arkrag import pipeline as pipeline_mod  # noqa: E402
from arkrag import query as query_mod  # noqa: E402
from arkrag import vector_store  # noqa: E402
from arkrag import vocab as vocab_mod  # noqa: E402

DEFAULT_RERANK_DIR = Path(
    r"C:\Users\Lenovo\WorkBuddy\2026-07-27-16-45-03\roleplay-ai\models\bge-reranker-base-onnx"
)


def pct(values: list[float], q: float) -> float:
    """最近秩分位数，不做插值（项目口径）。"""
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def rank_of(ids: list[str], gold: set[str]) -> int | None:
    for i, cid in enumerate(ids, 1):
        if cid in gold:
            return i
    return None


def minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    if hi - lo <= 1e-12:
        # 全同分：不造顺序，一律给 0，由下游的稳定排序键决定。
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def dbsf(scores: dict[str, float]) -> dict[str, float]:
    """均值 ± 标准差归一。标准差退化为 0 时不造顺序。"""
    if not scores:
        return {}
    vals = list(scores.values())
    mean = statistics.fmean(vals)
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    if sd <= 1e-12:
        return {k: 0.0 for k in scores}
    lo, hi = mean - sd, mean + sd
    span = hi - lo
    out = {}
    for k, v in scores.items():
        x = (v - lo) / span
        # 落在 ±1σ 之外的值夹到 [0,1]：超出部分没有额外信息，只会让单个极值
        # 重新变成 minmax 那种「一个点定整个量纲」的局面。
        out[k] = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
    return out


def rrf(scores_by_channel: list[dict[str, float]], *, k: int) -> dict[str, float]:
    """名次倒数融合。同通道内按分数降序定名次，同分按 chunk_id 稳定。"""
    acc: dict[str, float] = {}
    for scores in scores_by_channel:
        for rank, cid in enumerate(
            sorted(scores, key=lambda c: (-scores[c], c)), start=1
        ):
            acc[cid] = acc.get(cid, 0.0) + 1.0 / (k + rank)
    return acc


def fused_order(
    base_scores: dict[str, float],
    vec_scores: dict[str, float],
    *,
    mode: str,
    top_k: int,
    base_ids: list[str],
    tail_k: int = 5,
    vec_weight: float = 1.0,
) -> list[str]:
    """返回融合后的 chunk_id 顺序（恰好 top_k 个，无重复）。"""
    base_rank = {cid: i for i, cid in enumerate(base_ids)}
    huge = len(base_ids) + 1

    if mode == "base":
        return base_ids[:top_k]

    if mode == "vec_only":
        return sorted(vec_scores, key=lambda c: (-vec_scores[c], c))[:top_k]

    if mode == "dbsf":
        bn, vn = dbsf(base_scores), dbsf(vec_scores)
        merged: dict[str, float] = {}
        for cid, s in bn.items():
            merged[cid] = s
        for cid, s in vn.items():
            merged[cid] = merged.get(cid, 0.0) + vec_weight * s
        return sorted(
            merged,
            key=lambda c: (-merged[c], base_rank.get(c, huge), c),
        )[:top_k]

    if mode == "dbsf_tail":
        bn, vn = dbsf(base_scores), dbsf(vec_scores)
        merged = dict(bn)
        for cid, s in vn.items():
            merged[cid] = merged.get(cid, 0.0) + vec_weight * s
        order = sorted(merged, key=lambda c: (-merged[c], base_rank.get(c, huge), c))
        head_n = max(0, top_k - tail_k)
        head = base_ids[:head_n]
        seen = set(head)
        tail = [c for c in order if c not in seen][: tail_k]
        for c in base_ids:
            if len(head) + len(tail) >= top_k:
                break
            if c not in seen and c not in tail:
                tail.append(c)
                seen.add(c)
        return (head + tail)[:top_k]

    if mode == "rrf_k1":
        merged = rrf([base_scores, vec_scores], k=1)
        return sorted(
            merged, key=lambda c: (-merged[c], base_rank.get(c, huge), c)
        )[:top_k]

    if mode == "rrf_k2":
        merged = rrf([base_scores, vec_scores], k=2)
        return sorted(
            merged, key=lambda c: (-merged[c], base_rank.get(c, huge), c)
        )[:top_k]

    raise SystemExit(f"未知 mode: {mode}")


def bootstrap_ci(deltas: list[float], *, times: int = 4000, seed: int = 20261106):
    """配对 bootstrap。返回 (观测均值, 2.5% 分位, 97.5% 分位, P(<=0))。"""
    if not deltas:
        return 0.0, 0.0, 0.0, 1.0
    obs = statistics.fmean(deltas)
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(times):
        means.append(statistics.fmean([deltas[rng.randrange(n)] for _ in range(n)]))
    means.sort()
    lo = means[int(0.025 * times)]
    hi = means[int(0.975 * times)]
    p = sum(1 for x in means if x <= 0) / times
    return obs, lo, hi, p


def metrics(rows: list[dict], key: str) -> dict:
    n = len(rows)
    r5 = sum(1 for r in rows if 0 < (r.get(key) or 0) <= 5)
    r20 = sum(1 for r in rows if 0 < (r.get(key) or 0) <= 20)
    mrr = sum(1.0 / r[key] for r in rows if r.get(key)) / n if n else 0.0
    # 第 5 名转化率：第 5 名这个位置上，gold 出现的比例。用来识别「前 5 名变多
    # 但都是靠前几名堆出来的、第 5 名本身在退化」这种假改进。
    at5 = sum(1 for r in rows if r.get(key) == 5)
    return {
        "recall@5": r5 / n if n else 0.0,
        "recall@20": r20 / n if n else 0.0,
        "mrr": mrr,
        "rank5_share_of_hits": at5 / r5 if r5 else 0.0,
        "n": n,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generation", default="g-20261103T132152Z-06a741")
    ap.add_argument("--gold-set", default="data/goldset/gold-v1.json")
    ap.add_argument("--limit", type=int, default=320)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--vector-k", type=int, default=100)
    ap.add_argument("--modes", nargs="+",
                    default=["base", "dbsf", "dbsf_tail", "rrf_k2"])
    ap.add_argument("--rerank-tops", nargs="*", type=int, default=[],
                    help="对融合结果的前 N 名做 CE 重排，如 10 20")
    ap.add_argument("--rerank-modes", nargs="*", default=None,
                    help="哪些模式参与重排，默认 = 第一个非 base 模式")
    ap.add_argument("--rerank-dir", default=str(DEFAULT_RERANK_DIR))
    ap.add_argument("--use-int8", action="store_true", default=True)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--json", default="data/eval_fusion2.json")
    args = ap.parse_args()

    t0 = time.perf_counter()
    loaded = pipeline_mod.load_generation(args.generation)
    index = loaded["index"]
    graph = loaded["graph"]
    chunks = loaded["chunks"]
    entities = graph.get("entities") or []
    gen = loaded["generation"]
    manifest = loaded["manifest"]
    content_digest = loaded["index_meta"]["index_content_digest"]

    # 块正文必须走 pipeline.resolve_chunks 这套唯一正解；直接用 loaded["chunks"]
    # 之前先让它跑一遍校验：世代自带 chunks.jsonl 与 manifest 的 chunk_manifest_hash
    # 必须对得上，对不上就抛（拒绝用错块算名次）。
    pipeline_mod.resolve_chunks(gen, manifest)
    print(f"世代 {args.generation} 实体 {len(entities)} 块 {len(chunks)} "
          f"({time.perf_counter()-t0:.1f}s)", flush=True)

    from arkrag.retrieve import Retriever

    retriever = Retriever(
        index, graph, loaded["adjacency"],
        chunks=chunks, assertions=loaded["assertions"], vocab=vocab_mod,
    )

    slug = vector_store._slug(embedding_mod.model_name())
    vec_bin, vec_meta = vector_store.vector_paths(slug, content_digest)
    if not (Path(vec_bin).exists() and Path(vec_meta).exists()):
        # 缺向量产物必须硬失败：静默退化成「没有向量」会让本轮结论变成假的。
        print(f"[x] 没有向量产物：{vec_bin}", flush=True)
        return 2
    matrix, vmeta = vector_store.load_vectors(vec_bin, vec_meta)
    vec = vector_store.VectorIndex(matrix, vmeta)
    if len(vec.chunk_ids) != len(chunks):
        print(f"[x] 向量条数 {len(vec.chunk_ids)} != 块数 {len(chunks)}", flush=True)
        return 2
    print(f"向量 {len(vec.chunk_ids)} × {vec.dim} ({vec.model})", flush=True)

    reranker = None
    rerank_tops = [k for k in args.rerank_tops if k > 0]
    if rerank_tops:
        from arkrag.rerank import CrossEncoderReranker

        t0 = time.perf_counter()
        reranker = CrossEncoderReranker(
            Path(args.rerank_dir),
            max_length=args.max_length,
            batch_size=args.batch_size,
            use_int8=args.use_int8,
        )
        reranker.warmup()
        print(f"重排就绪 {reranker.info()} 加载 "
              f"{1000*(time.perf_counter()-t0):.1f}ms", flush=True)

    gold = json.loads((ROOT / args.gold_set).read_text(encoding="utf-8"))
    raw = gold.get("items") if isinstance(gold, dict) else gold
    items = [g for g in (raw or []) if g.get("gold_chunk_ids")][: args.limit]
    print(f"题 {len(items)}（有 gold_chunk_ids）", flush=True)

    rerank_modes = args.rerank_modes or [
        m for m in args.modes if m not in ("base", "vec_only")
    ][:1]
    rerank_modes = [m for m in rerank_modes if m in args.modes]

    keys = list(args.modes)
    for m in rerank_modes:
        for n in rerank_tops:
            keys.append(f"{m}_rr{n}")

    rows: list[dict] = []
    lat: dict[str, list[float]] = {m: [] for m in keys}

    def doc_text(cid: str) -> str:
        return vector_store.chunk_text_for_embedding(chunks.get(cid) or {})

    for i, item in enumerate(items, 1):
        qid = str(item.get("question_id") or i)
        q = str(item.get("question") or "")
        gset = {str(c) for c in item["gold_chunk_ids"]}

        qa = query_mod.analyze_query(
            q, entities=entities, aliases=graph.get("alias_index"), vocab=vocab_mod
        )

        t0 = time.perf_counter()
        found = retriever.search(qa, top_k=args.top_k)
        base_lat = 1000 * (time.perf_counter() - t0)
        base_hits = found["hits"]
        base_ids = [str(h["chunk_id"]) for h in base_hits]
        base_scores = {str(h["chunk_id"]): float(h["score"]) for h in base_hits}

        t0 = time.perf_counter()
        qv = embedding_mod.embed_batch([q])[0]
        vhits = vec.search(qv, top_k=args.vector_k)
        vec_lat = 1000 * (time.perf_counter() - t0)
        vec_scores = {str(c): float(s) for c, s in vhits}

        row: dict = {"qid": qid}

        for m in args.modes:
            t0 = time.perf_counter()
            order = fused_order(
                base_scores, vec_scores,
                mode=m, top_k=args.top_k, base_ids=base_ids,
            )
            if m == "base":
                lat["base"].append(base_lat)
            else:
                lat[m].append(base_lat + vec_lat)
            row[m] = rank_of(order, gset)
            row[f"_{m}_ids"] = order

        if reranker is not None:
            for m in rerank_modes:
                order = row[f"_{m}_ids"]
                for n in rerank_tops:
                    t0 = time.perf_counter()
                    take = order[:n]
                    seq = reranker.rerank(q, [doc_text(c) for c in take])
                    ranked = [take[j] for j in seq] + order[n:]
                    key = f"{m}_rr{n}"
                    lat[key].append(base_lat + vec_lat + 1000 * (time.perf_counter() - t0))
                    row[key] = rank_of(ranked, gset)

        for m in keys:
            row.pop(f"_{m}_ids", None)
        # 重排键的临时 id 序列也要清掉
        for m in list(row):
            if m.endswith("_ids"):
                row.pop(m, None)
        rows.append(row)

        if i % 20 == 0 or i == len(items):
            print(f"  {i}/{len(items)}  "
                  f"base R@20={sum(1 for r in rows if 0 < (r.get('base') or 0) <= 20)/len(rows):.4f}",
                  flush=True)

    result = {
        "generation": args.generation,
        "gold_set": args.gold_set,
        "questions": len(items),
        "top_k": args.top_k,
        "vector_k": args.vector_k,
        "vector": {
            "model": vec.model, "dim": vec.dim,
            "count": len(vec.chunk_ids), "digest": content_digest[:16],
        },
        "rerank": reranker.info() if reranker is not None else None,
        "rerank_tops": rerank_tops,
        "rerank_modes": rerank_modes,
        "metrics": {m: metrics(rows, m) for m in keys},
        "latency": {
            m: {"p50_ms": pct(v, 0.5), "p95_ms": pct(v, 0.95),
                "mean_ms": statistics.fmean(v) if v else 0.0, "n": len(v)}
            for m, v in lat.items()
        },
        "rows": rows,
    }

    base_key = "base"
    sig = {}
    for m in keys:
        if m == base_key:
            continue
        dm = [1.0 / r[m] - 1.0 / r[base_key] if r.get(m) and r.get(base_key)
              else (1.0 / r[m] if r.get(m) else 0.0) - (1.0 / r[base_key] if r.get(base_key) else 0.0)
              for r in rows]
        d20 = [(1.0 if 0 < (r.get(m) or 0) <= 20 else 0.0)
               - (1.0 if 0 < (r.get(base_key) or 0) <= 20 else 0.0) for r in rows]
        d5 = [(1.0 if 0 < (r.get(m) or 0) <= 5 else 0.0)
              - (1.0 if 0 < (r.get(base_key) or 0) <= 5 else 0.0) for r in rows]
        om, lm, hm, pm = bootstrap_ci(dm)
        o20, l20, h20, p20 = bootstrap_ci(d20)
        o5, l5, h5, p5 = bootstrap_ci(d5)
        sig[m] = {
            "mrr": {"delta": om, "ci_lo": lm, "ci_hi": hm, "p_le_0": pm},
            "recall@20": {"delta": o20, "ci_lo": l20, "ci_hi": h20, "p_le_0": p20},
            "recall@5": {"delta": o5, "ci_lo": l5, "ci_hi": h5, "p_le_0": p5},
        }
    result["significance"] = sig

    out = ROOT / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    b = result["metrics"][base_key]
    print("\n=== 质量（输出窗 top-20），Δ 相对 base ===")
    print(f"{'模式':<16}{'R@5':>8}{'ΔR@5':>9}{'R@20':>8}{'ΔR@20':>9}"
          f"{'MRR':>9}{'ΔMRR':>9}{'第5名':>8}{'MRR p':>8}")
    for m in keys:
        s = result["metrics"][m]
        g = sig.get(m, {}).get("mrr", {})
        print(f"{m:<16}{s['recall@5']:>8.4f}{s['recall@5']-b['recall@5']:>+9.4f}"
              f"{s['recall@20']:>8.4f}{s['recall@20']-b['recall@20']:>+9.4f}"
              f"{s['mrr']:>9.4f}{s['mrr']-b['mrr']:>+9.4f}"
              f"{s['rank5_share_of_hits']:>8.4f}"
              f"{g.get('p_le_0', float('nan')):>8.4f}")

    print("\n=== MRR 95% CI（配对 bootstrap 4000 次）===")
    for m in keys:
        g = sig.get(m, {}).get("mrr")
        if not g:
            continue
        print(f"{m:<16}{g['delta']:+.4f}  [{g['ci_lo']:+.4f}, {g['ci_hi']:+.4f}]  "
              f"P(<=0)={g['p_le_0']:.4f}")

    print("\n=== 延迟（检索阶段口径；重排行含 CE 前向）===")
    for m in keys:
        s = result["latency"][m]
        print(f"{m:<16}P50 {s['p50_ms']:>8.1f}ms   P95 {s['p95_ms']:>8.1f}ms")

    print(f"\n落盘 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
