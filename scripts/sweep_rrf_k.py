"""扫描 RRF 的平滑常数 k，看能否既拿召回增益又不牺牲头部精度。

## 为什么要扫

`RRF_K=60` 是论文与 ES/Weaviate 的默认值，但它假设各通道名次的可信度相近。
本项目实测（`data/eval_fusion_320_k20.json`）：k=60 时
* R@20 0.6575 → 0.6936（+0.0361，确实多捞到了 gold）
* MRR 0.4123 → 0.3370（−0.0753，前几名反而变差）

算一下就知道为什么：k=60 时 rank1 与 rank30 的贡献是 1/61=0.01639 与
1/91=0.01099，只差 0.0054；而"多一路命中"直接再加一份 1/(60+rank)。
于是**在两路各排第 30 的块能压过在单路排第 1 的块**。本项目上下文预算只装得下
4.7 块（G 阶段实测），前几名比"多捞到但进不了上下文"值钱得多，这个交易不划算。

k 越小越看重头部名次。这里把 k 从 60 扫到 1（含不对称的 1/5/10/20/30/60），
**结论只认「R@20 与 MRR 同时不劣于 baseline」的 k**，只有一个指标变好不算数。

## 为什么分两段

embedding 320 条要 30 秒以上，换 k 却只需重算一遍加法。所以先把每条通道的
**名次**算一次存下来，之后每个 k 都只在内存里重融合，不重复调模型。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arkrag import (  # noqa: E402
    config,
    embedding,
    fusion,
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


def metrics(hits_per_q: list[list[str]], questions: list[dict]) -> tuple[float, float, float, float]:
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
    return (
        round(sum(r5) / n, 4),
        round(sum(r20) / n, 4),
        round(sum(r50) / n, 4),
        round(sum(rr) / n, 4),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="扫描 RRF k")
    ap.add_argument("--generation", default="g-20261031T035231Z-4eafae")
    ap.add_argument("--gold-set", default="data/goldset/gold-v1.json")
    ap.add_argument("--limit", type=int, default=320)
    ap.add_argument("--top-k", type=int, default=20, help="生产口径是 20，比对必须同口径")
    ap.add_argument("--channel-k", type=int, default=100)
    ap.add_argument("--ks", nargs="+", type=float, default=[60, 30, 20, 10, 5, 3, 1])
    ap.add_argument("--json")
    args = ap.parse_args(argv)

    gold = json.loads(Path(args.gold_set).read_text(encoding="utf-8"))
    items = gold["items"] if isinstance(gold, dict) else gold
    import importlib.util

    spec = importlib.util.spec_from_file_location("_ask", Path(__file__).resolve().parent / "ask.py")
    ask = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ask)
    questions = ask._stratified_sample(items, args.limit)
    print(f"题 {len(questions)}  世代 {args.generation}  top_k={args.top_k}")

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
    print(f"块 {len(chunks)}  向量 {len(vec)}×{vec.dim}")

    fuser = fusion.FusionRetriever(
        index,
        vec,
        chunks=chunks,
        entities_by_id=retriever.entities_by_id,
        embed_query=lambda t: embedding.embed_batch([t])[0],
        channel_k=args.channel_k,
    )

    # ---- 第一段：算一次名次，存下来 ----
    per_q: list[dict[str, list[str]]] = []
    baseline_hits: list[list[str]] = []
    for i, item in enumerate(questions):
        q = str(item.get("question") or "")
        analysis = query_mod.analyze_query(
            q,
            entities=graph.get("entities") or [],
            aliases=graph.get("alias_index"),
            vocab=vocab_mod,
            context=retriever.query_context,
        )
        text = str(analysis.get("retrieval_query") or analysis.get("normalized_query") or "")
        chans = fuser.channels(analysis, text=text)
        per_q.append(chans)
        baseline_hits.append(
            [str(h["chunk_id"]) for h in retriever.search(analysis, top_k=args.top_k)["hits"]]
        )
        if (i + 1) % 40 == 0:
            print(f"  名次 {i + 1}/{len(questions)}", flush=True)

    base = metrics(baseline_hits, questions)
    print(f"\nbaseline                R@5 {base[0]:.4f}  R@20 {base[1]:.4f}  R@50 {base[2]:.4f}  MRR {base[3]:.4f}")

    # ---- 第二段：每个 k 只重融合 ----
    rows = []
    print(f"\n{'k':>6} {'通道':>10} {'R@5':>8} {'R@20':>8} {'R@50':>8} {'MRR':>8}  {'ΔR@20':>8} {'ΔMRR':>8}  判定")
    print("-" * 92)
    for k in args.ks:
        for label, use_vec in (("bm25+entity", False), ("三通道", True)):
            hits_per_q = []
            for chans in per_q:
                sub = {n: v for n, v in chans.items() if use_vec or n != "vector"}
                fused = fusion.rrf_fuse(sub, k=k, top_k=args.top_k)
                hits_per_q.append([f["chunk_id"] for f in fused])
            r5, r20, r50, mrr = metrics(hits_per_q, questions)
            d20, dm = round(r20 - base[1], 4), round(mrr - base[3], 4)
            if d20 >= 0 and dm >= 0:
                verdict = "两者都不劣"
            elif d20 > 0 and dm < 0:
                verdict = "召回升/MRR降"
            elif d20 < 0 and dm > 0:
                verdict = "召回降/MRR升"
            else:
                verdict = "两者都劣"
            rows.append(
                {"k": k, "channels": label, "r@5": r5, "r@20": r20, "r@50": r50, "mrr": mrr,
                 "d_r@20": d20, "d_mrr": dm, "verdict": verdict}
            )
            print(
                f"{k:>6g} {label:>10} {r5:>8.4f} {r20:>8.4f} {r50:>8.4f} {mrr:>8.4f}  "
                f"{d20:>+8.4f} {dm:>+8.4f}  {verdict}"
            )

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(
                {"generation": args.generation, "top_k": args.top_k, "questions": len(questions),
                 "baseline": {"r@5": base[0], "r@20": base[1], "r@50": base[2], "mrr": base[3]},
                 "rows": rows},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
