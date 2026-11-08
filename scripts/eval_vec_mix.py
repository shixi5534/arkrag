"""向量通道怎么进 top-20：三种接法的同批对比（只测排序，不测重排）。

前一版 `eval_union_rerank.py` 的 `vec_union` 模式有一个结构性缺陷，先说清楚：
它把并集拼成 `base_top20 + 向量新面孔`，于是向量带进来的块**名次必然 ≥21**，
R@20 在定义上不可能变。实测也确实一模一样（base 0.7900 / vec_union 0.7900）。
那个模式只能回答「向量能把多少题的 gold 拉进池子」（14 题），
**不能**回答「向量能不能把 gold 排进前 20」。

本脚本回答后者，三种接法：

  vec_only      只用向量 top-20（基线，验证向量单路上限）
  base          现状生产链路（bm25 + 实体 + 说话人 + cooc）
  mix_minmax    base 与向量的**分值各自 min-max 归一后加权求和**，再取 top-20
  mix_rank      两路名次倒数加权（等价于 k 很大的 RRF），再取 top-20

`mix_minmax` 的权重 `--weight` 是向量的权重：score = base_n + w * vec_n。
w=0 应当逐位复现 `base`——这是一条内置的自校验，不成立就说明归一路径有 bug。

口径同 `eval_union_rerank.py`：R@5 / R@20 / MRR / 第 5 名转化率，输出窗 top_k=20。
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
    pipeline,
    query as query_mod,
    retrieve,
    vector_store,
    vocab as vocab_mod,
)


def pct(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    if hi - lo <= 1e-12:
        # 全同分：归一成 0，不许造出顺序（并列就让它并列）
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def order(ids, gold):
    for i, cid in enumerate(ids, 1):
        if cid in gold:
            return i
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generation", default="g-20261103T132152Z-06a741")
    ap.add_argument("--gold-set", default="data/goldset/gold-v1.json")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--vector-k", type=int, default=100)
    ap.add_argument("--weights", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--json", default="data/eval_vec_mix.json")
    args = ap.parse_args()

    loaded = pipeline.load_generation(args.generation)
    index = loaded["index"]
    graph = loaded["graph"]
    chunks = loaded["chunks"]
    digest = (loaded["index_meta"] or {}).get("index_content_digest") or ""

    retriever = retrieve.Retriever(index, graph, {}, chunks=chunks, vocab=vocab_mod)
    entities = graph.get("entities") or []

    model = embedding.model_name()
    bin_path, meta_path = vector_store.vector_paths(
        vector_store._slug(model), digest
    )
    if not (bin_path.exists() and meta_path.exists()):
        print(f"缺向量产物：{bin_path}", file=sys.stderr)
        return 3
    matrix, vmeta = vector_store.load_vectors(bin_path, meta_path)
    vec = vector_store.VectorIndex(matrix, vmeta)
    print(f"块 {len(chunks)}，实体 {len(retriever.entities_by_id)}，"
          f"向量 {len(vec.chunk_ids)}×{vec.dim}", flush=True)

    gold = json.loads((ROOT / args.gold_set).read_text(encoding="utf-8"))
    raw = gold.get("items") if isinstance(gold, dict) else gold
    items = [g for g in (raw or []) if g.get("gold_chunk_ids")][: args.limit]
    print(f"题 {len(items)}", flush=True)

    modes = ["base", "vec_only"] + [f"mix_w{w:g}" for w in args.weights] + ["mix_rank"]
    rows = []
    lat = {m: [] for m in modes}

    for i, item in enumerate(items, 1):
        qid = str(item.get("question_id") or i)
        q = str(item.get("question") or "")
        gset = {str(c) for c in item["gold_chunk_ids"]}

        qa = query_mod.analyze_query(
            q, entities=entities, aliases=graph.get("alias_index"), vocab=vocab_mod
        )
        t0 = time.perf_counter()
        found = retriever.search(qa, top_k=args.top_k)
        t_base = 1000 * (time.perf_counter() - t0)
        base_hits = found["hits"]
        base_ids = [str(h["chunk_id"]) for h in base_hits]
        base_scores = {str(h["chunk_id"]): float(h.get("score") or 0.0) for h in base_hits}

        t0 = time.perf_counter()
        qv = embedding.embed_batch([q])[0]
        vec_hits = vec.search(qv, top_k=args.vector_k)
        t_vec = 1000 * (time.perf_counter() - t0)
        vec_ids = [str(c) for c, _ in vec_hits]
        vec_scores = {str(c): float(s) for c, s in vec_hits}

        row = {"qid": qid, "t_base": t_base, "t_vec": t_vec,
               "base": order(base_ids, gset),
               "vec_only": order(vec_ids, gset)}
        lat["base"].append(t_base)
        lat["vec_only"].append(t_base + t_vec)

        bn = minmax(base_scores)
        vn = minmax(vec_scores)
        base_rank = {cid: r for r, cid in enumerate(base_ids, 1)}
        vec_rank = {cid: r for r, cid in enumerate(vec_ids, 1)}
        for w in args.weights:
            merged = dict(bn)
            # w == 0 时绝不能把向量独有块以 0 分插进来：base 最末一名归一化后
            # 也是 0.0，两者并列后会按 chunk_id 抢位，把 base 的第 20 名挤掉
            # （实测 3/300 题因此翻转，自校验当场报错）。
            if w > 0.0:
                for cid, s in vn.items():
                    merged[cid] = merged.get(cid, 0.0) + w * s
            ranked = sorted(
                merged.items(),
                key=lambda kv: (
                    -kv[1],
                    base_rank.get(kv[0], 10**9),
                    vec_rank.get(kv[0], 10**9),
                    kv[0],
                ),
            )[: args.top_k]
            key = f"mix_w{w:g}"
            row[key] = order([c for c, _ in ranked], gset)
            lat[key].append(t_base + t_vec)

        # 名次倒数加权（k 取大 ⇒ 只做温和融合，不用 RRF 的小 k）
        inv = {}
        for r, cid in enumerate(base_ids, 1):
            inv[cid] = inv.get(cid, 0.0) + 1.0 / (60 + r)
        for r, cid in enumerate(vec_ids, 1):
            inv[cid] = inv.get(cid, 0.0) + 1.0 / (60 + r)
        ranked = sorted(inv.items(), key=lambda kv: (-kv[1], kv[0]))[: args.top_k]
        row["mix_rank"] = order([c for c, _ in ranked], gset)
        lat["mix_rank"].append(t_base + t_vec)

        rows.append(row)
        if i % 50 == 0 or i == len(items):
            print(f"  {i}/{len(items)}", flush=True)

    def metrics(key):
        n = len(rows)
        hit5 = hit20 = fifth = 0
        rr = 0.0
        for row in rows:
            rank = row[key]
            if rank is None:
                continue
            if rank <= 20:
                hit20 += 1
            if rank <= 5:
                hit5 += 1
                if rank == 5:
                    fifth += 1
            rr += 1.0 / rank
        return {"n": n, "recall@5": hit5 / n, "recall@20": hit20 / n,
                "mrr": rr / n, "rank5_share_of_hits": (fifth / hit20) if hit20 else 0.0}

    result = {
        "generation": args.generation,
        "gold_set": args.gold_set,
        "questions": len(items),
        "top_k": args.top_k,
        "vector_k": args.vector_k,
        "vector": {"model": vec.model, "dim": vec.dim, "count": len(vec.chunk_ids),
                   "digest": digest[:16]},
        "metrics": {m: metrics(m) for m in modes},
        "latency": {m: {"p50_ms": pct(lat[m], 0.5), "p95_ms": pct(lat[m], 0.95),
                        "mean_ms": statistics.fmean(lat[m]) if lat[m] else 0.0}
                    for m in modes},
        "rows": rows,
    }
    out = ROOT / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n=== 质量（输出窗 top-20）===")
    print(f"{'模式':<12}{'R@5':>9}{'R@20':>9}{'MRR':>9}{'第5名转化':>11}")
    for m in modes:
        s = result["metrics"][m]
        print(f"{m:<12}{s['recall@5']:>9.4f}{s['recall@20']:>9.4f}"
              f"{s['mrr']:>9.4f}{s['rank5_share_of_hits']:>11.4f}")
    print("\n=== 延迟 ===")
    for m in modes:
        s = result["latency"][m]
        print(f"{m:<12}P50 {s['p50_ms']:>7.1f}ms  P95 {s['p95_ms']:>7.1f}ms")
    print(f"\n落盘 {out}")

    # 自校验：w=0 必须逐位等于 base
    if 0.0 in args.weights:
        same = all(r["base"] == r["mix_w0"] for r in rows)
        print(f"自校验 mix_w0 == base 逐题名次相同：{same}")
        if not same:
            print("[!] 归一路径与 base 不一致，上面 mix_* 的数字不可信", file=sys.stderr)
            return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
