"""量每一条通道各自单独的召回质量，判断哪一路在拖后腿。

## 为什么必须先量这个

`data/sweep_rrf_k.json` 显示：k 从 60 扫到 1，**没有任何一个 k 能让 R@20 与 MRR
同时不劣于 baseline**。最接近的是 k=1/三通道（R@20 +0.0125 / MRR −0.0071）。
而 `bm25+entity` 两路在 k=60 就把 MRR 从 0.4123 拉到 0.3410 —— 说明拖后腿的
很可能是实体通道，不是向量通道。

在调 k 或改融合公式之前，先确认每条通道**自己**的排序有多准。否则会陷入
「换个 k 再试一次」的无底洞：融合公式只是加权求和，通道本身不准，
任何权重都救不回来。

## 三个输出

1. `solo`：每条通道单独作为结果时的 R@5/20/50/MRR，以及「gold 块出现在该通道
   候选里的比例」（覆盖率上限——通道都没捞到的题，融合再准也没用）。
2. `verdict`：按 gold 是否被各通道捞到分类（全部捞到 / 只有部分 / 都没捞到），
   看融合增益到底来自哪一种情况。
3. `entity_diagnosis`：实体通道的候选规模与话题集中度，判断它是不是
   「谁提得多谁排前面」的通用名词垃圾。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arkrag import (  # noqa: E402
    embedding,
    fusion,
    index as index_mod,
    pipeline,
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
    r5, r20, r50, rr, cover = [], [], [], [], []
    for hits, item in zip(hits_per_q, questions):
        gold = [str(c) for c in item.get("gold_chunk_ids") or []]
        if not gold:
            continue
        gs = set(gold)
        r5.append(len(gs & set(hits[:5])) / len(gs))
        r20.append(len(gs & set(hits[:20])) / len(gs))
        r50.append(len(gs & set(hits[:50])) / len(gs))
        cover.append(1.0 if gs & set(hits) else 0.0)
        first = next((i for i, cid in enumerate(hits) if cid in gs), None)
        rr.append(0.0 if first is None else 1.0 / (first + 1))
    n = len(r20) or 1
    return {
        "n": len(r20),
        "r@5": round(sum(r5) / n, 4),
        "r@20": round(sum(r20) / n, 4),
        "r@50": round(sum(r50) / n, 4),
        "mrr": round(sum(rr) / n, 4),
        "hit_rate": round(sum(cover) / n, 4),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="分通道召回诊断")
    ap.add_argument("--generation", default="g-20261031T035231Z-4eafae")
    ap.add_argument("--gold-set", default="data/goldset/gold-v1.json")
    ap.add_argument("--limit", type=int, default=320)
    ap.add_argument("--channel-k", type=int, default=100)
    args = ap.parse_args(argv)

    gold = json.loads(Path(args.gold_set).read_text(encoding="utf-8"))
    items = gold["items"] if isinstance(gold, dict) else gold
    import importlib.util

    spec = importlib.util.spec_from_file_location("_ask", Path(__file__).resolve().parent / "ask.py")
    ask = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ask)
    questions = ask._stratified_sample(items, args.limit)
    print(f"题 {len(questions)}  channel_k={args.channel_k}")

    gen = release.Generation(args.generation)
    # 同上：分通道诊断必须用这一代自己的块正文，否则 vector 单路的命中率
    # 会被「空串向量」污染，而污染方向恰好是「向量通道看起来更差」。
    chunks_path = pipeline.resolve_chunks(gen, gen.load())
    chunks = {str(r["chunk_id"]): r for r in read_jsonl(chunks_path) if r.get("chunk_id")}
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

    solo: dict[str, list[list[str]]] = {"bm25": [], "vector": [], "entity": []}
    baseline_hits: list[list[str]] = []
    entity_sizes: list[int] = []
    entity_terms: Counter[str] = Counter()

    for i, item in enumerate(questions):
        q = str(item.get("question") or "")
        analysis = query_mod.analyze_query(
            q, entities=graph.get("entities") or [], aliases=graph.get("alias_index"),
            vocab=vocab_mod, context=retriever.query_context,
        )
        text = str(analysis.get("retrieval_query") or analysis.get("normalized_query") or "")
        chans = fuser.channels(analysis, text=text)
        for name in solo:
            solo[name].append(list(chans.get(name) or []))
        entity_sizes.append(len(chans.get("entity") or []))
        for cand in analysis.get("entity_candidates") or []:
            entity_terms[str(cand.get("name") or "")] += 1
        baseline_hits.append(
            [str(h["chunk_id"]) for h in retriever.search(analysis, top_k=20)["hits"]]
        )
        if (i + 1) % 80 == 0:
            print(f"  {i + 1}/{len(questions)}", flush=True)

    print("\n=== 单通道质量（各自单独当结果，channel_k=%d）===" % args.channel_k)
    print(f"{'通道':<8} {'R@5':>8} {'R@20':>8} {'R@50':>8} {'MRR':>8} {'gold命中率':>10} {'长度':>8}")
    print("-" * 66)
    base = metrics(baseline_hits, questions)
    print(f"{'baseline':<8} {base['r@5']:>8.4f} {base['r@20']:>8.4f} {base['r@50']:>8.4f} "
          f"{base['mrr']:>8.4f} {base['hit_rate']:>10.4f} {20:>8}")
    for name, hits in solo.items():
        m = metrics(hits, questions)
        avg_len = round(sum(len(h) for h in hits) / max(1, len(hits)), 1)
        print(f"{name:<8} {m['r@5']:>8.4f} {m['r@20']:>8.4f} {m['r@50']:>8.4f} "
              f"{m['mrr']:>8.4f} {m['hit_rate']:>10.4f} {avg_len:>8}")

    # ---- gold 被哪几路捞到 ----
    print("\n=== gold 块被各通道捞到的情况（决定融合上限）===")
    buckets: Counter[str] = Counter()
    for i, item in enumerate(questions):
        gold_ids = set(str(c) for c in item.get("gold_chunk_ids") or [])
        if not gold_ids:
            continue
        got = tuple(name for name in ("bm25", "vector", "entity") if gold_ids & set(solo[name][i]))
        buckets["+".join(got) if got else "无任何通道"] += 1
    for key, count in buckets.most_common():
        print(f"  {key:<24} {count:>3} 题")

    # ---- 实体通道诊断 ----
    print("\n=== 实体通道诊断 ===")
    sizes_sorted = sorted(entity_sizes)
    print(f"  候选数 P50 {sizes_sorted[len(sizes_sorted)//2]}  max {sizes_sorted[-1]}  "
          f"空候选题 {sum(1 for s in entity_sizes if s == 0)}")
    print(f"  问句里命中的实体（前 12）：{dict(entity_terms.most_common(12))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
