"""融合变体扫描：通道子集 × RRF k，全部在内存里重融合。

## 背景

`data/sweep_rrf_k.json`：k 从 60 扫到 1，**没有任何 k 能让 R@20 与 MRR 同时不劣于
baseline**。而 `diagnose_channels.py` 给出了原因——三条通道各自的单独质量差得远：

    baseline      R@20 0.6575  MRR 0.4123  (bm25+0.5*entity 加权)
    bm25 单路     R@20 0.6478  MRR 0.4248   ← MRR 比 baseline 还高
    vector 单路   R@20 0.3892  MRR 0.2101
    entity 单路   R@20 0.4650  MRR 0.2255   ← 只给 100 个候选，头部很糙

所以「三路同权投票」失败是必然的：两条质量 0.21 的通道各投一票，足以把
bm25 排在第 1 的块压下去。这不是 k 调得不好，是**投票结构**错了。

## 这一轮要回答三个问题

1. 去掉实体通道（bm25+vector）能不能救回 MRR？
2. 去掉向量通道（bm25+entity）呢？——用来确认拖后腿的到底是谁。
3. 有没有任何子集 × k 的组合能「R@20 与 MRR 同时不劣」？

判据仍是那句话：**只有一个指标变好不算数**。上下文只装 4.7 块，
MRR 掉 0.07 意味着首答落点整体后退，R@20 涨 0.03 补不回来。
"""

from __future__ import annotations

import argparse
import itertools
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
    return {
        "r@5": round(sum(r5) / n, 4),
        "r@20": round(sum(r20) / n, 4),
        "r@50": round(sum(r50) / n, 4),
        "mrr": round(sum(rr) / n, 4),
        "n": len(r20),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="融合通道子集 × k 扫描")
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

    fuser = fusion.FusionRetriever(
        index, vec, chunks=chunks, entities_by_id=retriever.entities_by_id,
        embed_query=lambda t: embedding.embed_batch([t])[0], channel_k=args.channel_k,
    )

    # ---- 一次算出所有通道名次 ----
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
    print(f"\nbaseline  R@5 {base['r@5']:.4f}  R@20 {base['r@20']:.4f}  MRR {base['mrr']:.4f}")

    subsets = [c for n in (1, 2, 3)
               for c in itertools.combinations(("bm25", "vector", "entity"), n)]
    ks = (60, 30, 20, 10, 5, 1)
    rows = []
    print(f"\n{'通道子集':<22} {'k':>4} {'R@5':>8} {'R@20':>8} {'MRR':>8}  {'ΔR@20':>8} {'ΔMRR':>8}  判定")
    print("-" * 96)
    for subset in subsets:
        for k in ks:
            hits_per_q = []
            for chans in per_q:
                sub = {n: v for n, v in chans.items() if n in subset}
                fused = fusion.rrf_fuse(sub, k=k, top_k=args.top_k)
                hits_per_q.append([f["chunk_id"] for f in fused])
            m = metrics(hits_per_q, questions)
            d20, dm = round(m["r@20"] - base["r@20"], 4), round(m["mrr"] - base["mrr"], 4)
            both_ok = d20 >= 0 and dm >= 0
            verdict = "★两者都≥baseline" if both_ok else (
                "召回升/MRR降" if d20 > 0 else ("召回降/MRR升" if dm > 0 else "两者都降"))
            rows.append({"subset": "+".join(subset), "k": k, **m, "d_r@20": d20, "d_mrr": dm,
                         "both_ok": both_ok, "verdict": verdict})
            print(f"{'+'.join(subset):<22} {k:>4} {m['r@5']:>8.4f} {m['r@20']:>8.4f} "
                  f"{m['mrr']:>8.4f}  {d20:>+8.4f} {dm:>+8.4f}  {verdict}")

    winners = [r for r in rows if r["both_ok"]]
    print("\n" + "=" * 96)
    if winners:
        best = max(winners, key=lambda r: (r["d_mrr"], r["d_r@20"]))
        print(f"结论：{len(winners)} 个组合两者都不劣，最好的是 {best['subset']} k={best['k']} "
              f"(ΔR@20 {best['d_r@20']:+.4f} / ΔMRR {best['d_mrr']:+.4f})")
    else:
        print("结论：没有任何通道子集 × k 能让 R@20 与 MRR 同时不劣于 baseline。")
        print("      RRF 在本项目上不适合作为默认检索融合——需要换机制（重排或加权 RRF）。")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps({"generation": args.generation, "top_k": args.top_k,
                        "questions": len(questions), "baseline": base, "rows": rows},
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\n写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
