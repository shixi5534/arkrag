"""加权 RRF：解决「RRF 拉低 MRR」的结构性原因。

## RRF 为什么在本项目失败（实测已确认，不是猜测）

`sweep_fusion.py` 扫了 7 种通道子集 × 6 个 k = 42 个组合，**没有一个**能让
R@20 与 MRR 同时不劣于 baseline。原因是数学上的：

    RRF 分 = Σ_channel 1/(k + rank_c)

k=60 时，rank 1 → 1/61 = 0.01639，rank 30 → 1/91 = 0.01099，**只差 0.0054**；
而「多一路命中」直接再加一整个 1/(k+rank) ≈ 0.016 左右。
于是**在两路各排第 30 的块能压过只在单路排第 1 的块**。

本项目上下文只装得下 4.7 块（`max_context_tokens=6000`），所以头部名次
比"多捞到但进不了上下文"的候选值钱得多。等权 RRF 正好做反了这件事。

## 两条修法，都要量

1. **通道权重**（`weights`）：给 bm25 更大权重。`diagnose_channels.py` 实测
   bm25 单路 MRR 0.4248 > baseline 0.4123，而 vector 0.2101 / entity 0.2255，
   说明 bm25 的**名次**远比另两路可信。按名次可信度给权重，比等权更符合事实。

2. **头部保护**（`head_lock`）：bm25 的前 N 名直接锁定在结果最前面，
   融合只用于往后续位置排。这等于承认「bm25 的 top-N 已经很好，
   融合的作用是补召回而不是改前 N 的顺序」。

判据仍然只有一条：**R@20 与 MRR 同时不劣于 baseline**。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arkrag import (  # noqa: E402
    config,
    embedding,
    index as index_mod,
    query as query_mod,
    release,
    retrieve,
    vector_store,
    vocab as vocab_mod,
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def metrics(hits_per_q: list[list[str]], questions: list[dict]) -> dict:
    r5, r20, r50, rr = [], [], [], []
    for hits, item in zip(hits_per_q, questions):
        gold = [str(c) for c in item.get("gold_chunk_ids") or []]
        if not gold:
            continue
        gs = set(gold)
        r5.append(len(gs & set(hits[:5])) / len(gs))
        r20.append(len(gs & set(hits[:20])) / len(gs))
        r50.append(len(gs & set(hits[:50])) / len(gs))
        first = next((i for i, cid in enumerate(hits) if cid in gs), None)
        rr.append(0.0 if first is None else 1.0 / (first + 1))
    n = len(r20) or 1
    return {"r@5": round(sum(r5) / n, 4), "r@20": round(sum(r20) / n, 4),
            "r@50": round(sum(r50) / n, 4), "mrr": round(sum(rr) / n, 4), "n": len(r20)}


def weighted_rrf(channel_ranks: dict[str, list[str]], *, weights: dict[str, float],
                 k: float, top_k: int, head_lock: int = 0) -> list[str]:
    """带通道权重与头部锁定的 RRF。

    `head_lock=N` > 0 时，把 `bm25` 通道的前 N 名原样放在结果最前面
    （不足则按实际长度），其余位置由融合分填充（跳过已锁定的）。
    """
    scores: dict[str, float] = defaultdict(float)
    for name, ranked in channel_ranks.items():
        w = float(weights.get(name, 1.0))
        if w == 0:
            continue
        seen: set[str] = set()
        for rank, cid in enumerate(ranked, start=1):
            if cid in seen:
                continue
            seen.add(cid)
            scores[cid] += w / (k + rank)

    locked: list[str] = []
    if head_lock > 0:
        locked = list((channel_ranks.get("bm25") or [])[:head_lock])
        for cid in locked:
            scores.pop(cid, None)

    rest = [cid for cid, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]
    return (locked + rest)[:top_k]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="加权 RRF + 头部锁定")
    ap.add_argument("--generation", default="g-20261031T035231Z-4eafae")
    ap.add_argument("--gold-set", default="data/goldset/gold-v1.json")
    ap.add_argument("--limit", type=int, default=320)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--channel-k", type=int, default=100)
    ap.add_argument("--json")
    args = ap.parse_args(argv)

    gold = json.loads(Path(args.gold_set).read_text(encoding="utf-8"))
    items = gold["items"] if isinstance(gold, dict) else gold
    import importlib.util

    spec = importlib.util.spec_from_file_location("_ask", Path(__file__).resolve().parent / "ask.py")
    ask = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ask)
    questions = ask._stratified_sample(items, args.limit)
    print(f"题 {len(questions)}  top_k={args.top_k}  channel_k={args.channel_k}")

    gen = release.Generation(args.generation)
    chunks = {str(r["chunk_id"]): r for r in read_jsonl(config.chunks_path()) if r.get("chunk_id")}
    index, index_meta = index_mod.load_index(gen.index_path)
    digest = (index_meta or {}).get("index_content_digest") or ""
    graph = json.loads(gen.graph_path.read_text(encoding="utf-8"))
    retriever = retrieve.Retriever(index, graph, {}, chunks=chunks, vocab=vocab_mod)

    model = embedding.model_name()
    bin_path, meta_path = vector_store.vector_paths(vector_store._slug(model), digest)
    matrix, vmeta = vector_store.load_vectors(bin_path, meta_path)
    vec = vector_store.VectorIndex(matrix, vmeta)

    from arkrag import fusion

    fuser = fusion.FusionRetriever(
        index, vec, chunks=chunks, entities_by_id=retriever.entities_by_id,
        embed_query=lambda t: embedding.embed_batch([t])[0], channel_k=args.channel_k,
    )

    # 一次算好三路名次 + baseline
    per_q: list[dict[str, list[str]]] = []
    baseline_hits: list[list[str]] = []
    for i, item in enumerate(questions):
        q = str(item.get("question") or "")
        analysis = query_mod.analyze_query(
            q, entities=graph.get("entities") or [], aliases=graph.get("alias_index"),
            vocab=vocab_mod, context=retriever.query_context,
        )
        text = str(analysis.get("retrieval_query") or analysis.get("normalized_query") or "")
        per_q.append(fuser.channels(analysis, text=text))
        baseline_hits.append(
            [str(h["chunk_id"]) for h in retriever.search(analysis, top_k=args.top_k)["hits"]]
        )
        if (i + 1) % 80 == 0:
            print(f"  名次 {i + 1}/{len(questions)}", flush=True)

    base = metrics(baseline_hits, questions)
    print(f"\nbaseline（生产口径）  R@5 {base['r@5']:.4f}  R@20 {base['r@20']:.4f}  MRR {base['mrr']:.4f}")

    # 组合：通道权重 × k × 头部锁定
    configs = []
    for bw in (1.0, 2.0, 3.0, 5.0):
        for vw, ew in ((1.0, 1.0), (0.5, 0.5), (1.0, 0.5), (0.5, 0.25)):
            for k in (60, 10):
                for lock in (0, 5, 10, 20):
                    configs.append({"weights": {"bm25": bw, "vector": vw, "entity": ew},
                                    "k": k, "head_lock": lock})

    print(f"\n共 {len(configs)} 个组合")
    print(f"{'bm25/v/e':>12} {'k':>4} {'锁':>4} {'R@5':>8} {'R@20':>8} {'MRR':>8}  {'ΔR@20':>8} {'ΔMRR':>8}  判定")
    print("-" * 96)
    rows = []
    for cfg in configs:
        hits_per_q = []
        for chans in per_q:
            hits_per_q.append(
                weighted_rrf(chans, weights=cfg["weights"], k=cfg["k"],
                             top_k=args.top_k, head_lock=cfg["head_lock"])
            )
        m = metrics(hits_per_q, questions)
        d20 = round(m["r@20"] - base["r@20"], 4)
        dm = round(m["mrr"] - base["mrr"], 4)
        both = d20 >= 0 and dm >= 0
        verdict = "★两者都不劣" if both else ("召回升/MRR降" if d20 > 0 else
                                          ("召回降/MRR升" if dm > 0 else "两者都降"))
        w = cfg["weights"]
        rows.append({"weights": w, "k": cfg["k"], "head_lock": cfg["head_lock"], **m,
                     "d_r@20": d20, "d_mrr": dm, "both_ok": both, "verdict": verdict})
        label = f"{w['bm25']:g}/{w['vector']:g}/{w['entity']:g}"
        print(f"{label:>12} {cfg['k']:>4g} {cfg['head_lock']:>4} {m['r@5']:>8.4f} "
              f"{m['r@20']:>8.4f} {m['mrr']:>8.4f}  {d20:>+8.4f} {dm:>+8.4f}  {verdict}")

    winners = [r for r in rows if r["both_ok"]]
    print("\n" + "=" * 96)
    if winners:
        best = max(winners, key=lambda r: (r["d_mrr"] + r["d_r@20"], r["mrr"]))
        w = best["weights"]
        print(f"结论：{len(winners)} 个组合两者都不劣。最好：bm25/v/e={w['bm25']:g}/{w['vector']:g}/"
              f"{w['entity']:g}  k={best['k']:g}  head_lock={best['head_lock']}  "
              f"ΔR@20 {best['d_r@20']:+.4f}  ΔMRR {best['d_mrr']:+.4f}")
    else:
        print("结论：加权 RRF 与头部锁定都无法让两个指标同时不劣。"
              "RRF 在本项目应彻底放弃，转向其它排序改进。")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps({"generation": args.generation, "top_k": args.top_k,
                        "questions": len(questions), "baseline": base, "rows": rows},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
