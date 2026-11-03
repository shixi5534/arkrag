"""对「权重 0.5 → 0.4」这类微小差异做显著性检验，防止把噪声当收益。

## 为什么需要这一步

`sweep_entity_weight.py` 的输出：

    w=0.50（生产值）  R@20 0.6842  MRR 0.4392
    w=0.40            R@20 0.6858  MRR 0.4406   ← ΔR@20 +0.0016 / ΔMRR +0.0014

看起来 0.4 更好。但这是在 **300 道题**上的均值，0.0014 的差距换成题目数就是
**平均每题 0.4 个名次的差别**——极可能只是几道题的抖动。若据此改生产常量，
就是把评测集噪声写进代码，下次换题集又会朝反方向调。

项目纪律写着「不许为迁就现状调低阈值」，同样地也不许为了凑数字去拟合 Gold Set。

## 判据

对每题算 `1/(rank_gold+1)` 的差（w=0.4 减 w=0.5），做**配对符号检验**：

* `wins` / `losses` / `ties`：0.4 更好的题数、更差的题数、持平的题数。
* 双侧 p 值（精确二项检验，n = wins+losses）：若 p ≥ 0.05，**差异不显著，
  维持原值 0.5**。
* 同时报 bootstrap 95% 区间（重采样题目，1,000 次），看区间是否跨 0。

这不是形式主义：n=300 时，即使两个配置完全等价，仅凭运气也会出现
wins 与 losses 差十几个题的情况。只有 p 值足够小才说明改动真的有效。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arkrag import (  # noqa: E402
    config,
    index as index_mod,
    query as query_mod,
    release,
    retrieve,
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


def reciprocal_ranks(hits_per_q: list[list[str]], questions: list[dict]) -> list[float | None]:
    """每题的 1/(rank+1)；无 gold 的题返回 None（不参与配对比较）。"""
    out: list[float | None] = []
    for hits, item in zip(hits_per_q, questions):
        gold = set(str(c) for c in item.get("gold_chunk_ids") or [])
        if not gold:
            out.append(None)
            continue
        first = next((i for i, cid in enumerate(hits) if cid in gold), None)
        out.append(0.0 if first is None else 1.0 / (first + 1))
    return out


def binom_two_sided(k: int, n: int) -> float:
    """精确二项检验（p=0.5）的双侧 p 值。n 最大几百，直接算组合数即可。"""
    if n == 0:
        return 1.0
    from math import comb

    total = 2 ** n
    # 双侧 = 2 × P(X ≤ min(k, n-k))，上限 1.0
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1))
    return min(1.0, 2 * tail / total)


def bootstrap_ci(diffs: list[float], *, reps: int = 1000, seed: int = 20261103) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(reps):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="通道权重差异显著性检验")
    ap.add_argument("--generation", default="g-20261031T035231Z-4eafae")
    ap.add_argument("--gold-set", default="data/goldset/gold-v1.json")
    ap.add_argument("--limit", type=int, default=320)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--weight-a", type=float, default=0.5, help="基准（当前生产值）")
    ap.add_argument("--weight-b", type=float, default=0.4, help="候选新值")
    args = ap.parse_args(argv)

    gold = json.loads(Path(args.gold_set).read_text(encoding="utf-8"))
    items = gold["items"] if isinstance(gold, dict) else gold
    import importlib.util

    spec = importlib.util.spec_from_file_location("_ask", Path(__file__).resolve().parent / "ask.py")
    ask = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ask)
    questions = ask._stratified_sample(items, args.limit)
    print(f"题 {len(questions)}  比较 w={args.weight_a}（基准） vs w={args.weight_b}（候选）")

    gen = release.Generation(args.generation)
    chunks = {str(r["chunk_id"]): r for r in read_jsonl(config.chunks_path()) if r.get("chunk_id")}
    index, _meta = index_mod.load_index(gen.index_path)
    graph = json.loads(gen.graph_path.read_text(encoding="utf-8"))
    retriever = retrieve.Retriever(index, graph, {}, chunks=chunks, vocab=vocab_mod)
    depth = args.top_k * retrieve.CANDIDATE_POOL_MULT

    bm25_all: list[dict[str, float]] = []
    entity_all: list[dict[str, float]] = []
    for i, item in enumerate(questions):
        q = str(item.get("question") or "")
        analysis = query_mod.analyze_query(
            q, entities=graph.get("entities") or [], aliases=graph.get("alias_index"),
            vocab=vocab_mod, context=retriever.query_context,
        )
        text = str(analysis.get("retrieval_query") or analysis.get("normalized_query") or "")
        bm25_all.append(retrieve._norm_scores(index.search(text, top_k=depth)) if text else {})
        ent: dict[str, float] = {}
        for cand in analysis.get("entity_candidates") or []:
            eid = str(cand.get("entity_id") or "")
            rec = retriever.entities_by_id.get(eid) or {}
            terms = [str(rec.get("name") or cand.get("name") or "")]
            for a in rec.get("aliases") or []:
                t = a.get("text") if isinstance(a, dict) else a
                if str(t or "").strip():
                    terms.append(str(t).strip())
            probe = " ".join(sorted({t for t in terms if t}))
            if not probe:
                continue
            w = float(cand.get("score") or 0.0)
            for cid, sc in retrieve._norm_scores(index.search(probe, top_k=depth)).items():
                ent[cid] = max(ent.get(cid, 0.0), sc * w)
        entity_all.append(ent)
        if (i + 1) % 80 == 0:
            print(f"  通道 {i + 1}/{len(questions)}", flush=True)

    def hits_for(w: float) -> list[list[str]]:
        out = []
        for bm25, ent in zip(bm25_all, entity_all):
            merged: dict[str, float] = defaultdict(float)
            for cid, sc in bm25.items():
                merged[cid] += sc
            for cid, sc in ent.items():
                merged[cid] += w * sc
            ranked = sorted(merged.items(), key=lambda kv: (-kv[1], kv[0]))
            out.append([cid for cid, _ in ranked[: args.top_k]])
        return out

    rr_a = reciprocal_ranks(hits_for(args.weight_a), questions)
    rr_b = reciprocal_ranks(hits_for(args.weight_b), questions)

    diffs = [b - a for a, b in zip(rr_a, rr_b) if a is not None and b is not None]
    wins = sum(1 for d in diffs if d > 1e-12)
    losses = sum(1 for d in diffs if d < -1e-12)
    ties = len(diffs) - wins - losses
    mean_a = sum(a for a in rr_a if a is not None) / max(1, sum(1 for a in rr_a if a is not None))
    mean_b = sum(b for b in rr_b if b is not None) / max(1, sum(1 for b in rr_b if b is not None))

    print(f"\nMRR：w={args.weight_a} → {mean_a:.4f}   w={args.weight_b} → {mean_b:.4f}   "
          f"Δ {mean_b - mean_a:+.5f}")
    print(f"配对结果：w={args.weight_b} 更好 {wins} 题 / 更差 {losses} 题 / 持平 {ties} 题")
    p = binom_two_sided(wins, wins + losses)
    lo, hi = bootstrap_ci(diffs)
    print(f"符号检验 双侧 p = {p:.4f}")
    print(f"bootstrap 95% 区间（MRR 差）= [{lo:+.5f}, {hi:+.5f}]")
    print()
    if p >= 0.05 or (lo <= 0 <= hi):
        print(f"结论：差异**不显著**（p={p:.3f}，区间跨 0）。维持 w={args.weight_a}，"
              f"不采纳 {args.weight_b}——那是评测集噪声，不是收益。")
    else:
        print(f"结论：差异显著（p={p:.4f}，区间不跨 0）。可以采纳 w={args.weight_b}。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
