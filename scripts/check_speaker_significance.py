"""对 speakers 结构化加成做显著性检验（R@20 与 MRR 分别验）。

## 为什么要单独验

`sweep_structured_boost.py` 给出 ws=0.1 时：

    R@20 0.6842 → 0.7019  (+0.0177)
    MRR  0.4392 → 0.4394  (+0.0002)

MRR 基本没动（这是好事：它说明加成没有破坏头部），R@20 涨了。但 **+0.0177 在
300 题上等于每题多命中 0.05 个 gold 块**——需要确认这不是少数题目带来的抖动。

因此分两个指标各自做配对检验：

* **R@20**：每题的 `|gold ∩ top20| / |gold|` 差值，做符号检验 + bootstrap。
* **MRR**：每题的 `1/(rank+1)` 差值，同样处理。

一个指标显著、另一个不显著，就如实这么写——**不许把"没变差"说成"变好了"**。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from math import comb
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


def _speakers_of(rec: dict) -> set[str]:
    raw = rec.get("speakers")
    if isinstance(raw, list):
        return {str(s) for s in raw if str(s).strip()}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw.replace("'", '"'))
            if isinstance(parsed, list):
                return {str(s) for s in parsed if str(s).strip()}
        except (json.JSONDecodeError, ValueError):
            return set()
    return set()


def binom_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1))
    return min(1.0, 2 * tail / (2 ** n))


def bootstrap_ci(diffs: list[float], *, reps: int = 2000, seed: int = 20261103) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def report(label: str, diffs: list[float]) -> None:
    wins = sum(1 for d in diffs if d > 1e-12)
    losses = sum(1 for d in diffs if d < -1e-12)
    ties = len(diffs) - wins - losses
    mean = sum(diffs) / max(1, len(diffs))
    p = binom_two_sided(wins, wins + losses)
    lo, hi = bootstrap_ci(diffs)
    print(f"\n--- {label} ---")
    print(f"  平均差 {mean:+.5f}   更好 {wins} 题 / 更差 {losses} 题 / 持平 {ties} 题")
    print(f"  符号检验双侧 p = {p:.4f}   bootstrap 95% 区间 [{lo:+.5f}, {hi:+.5f}]")
    if p < 0.05 and lo > 0:
        print("  → 显著提升")
    elif p < 0.05 and hi < 0:
        print("  → 显著下降")
    else:
        print("  → 不显著（区间跨 0 或 p ≥ 0.05），不能说它有变化")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="speakers 加成显著性")
    ap.add_argument("--generation", default="g-20261031T035231Z-4eafae")
    ap.add_argument("--gold-set", default="data/goldset/gold-v1.json")
    ap.add_argument("--limit", type=int, default=320)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--ws", type=float, default=0.1)
    args = ap.parse_args(argv)

    gold = json.loads(Path(args.gold_set).read_text(encoding="utf-8"))
    items = gold["items"] if isinstance(gold, dict) else gold
    import importlib.util

    spec = importlib.util.spec_from_file_location("_ask", Path(__file__).resolve().parent / "ask.py")
    ask = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ask)
    questions = ask._stratified_sample(items, args.limit)
    print(f"题 {len(questions)}  比较 ws=0 vs ws={args.ws}")

    gen = release.Generation(args.generation)
    chunks = {str(r["chunk_id"]): r for r in read_jsonl(config.chunks_path()) if r.get("chunk_id")}
    index, _meta = index_mod.load_index(gen.index_path)
    graph = json.loads(gen.graph_path.read_text(encoding="utf-8"))
    retriever = retrieve.Retriever(index, graph, {}, chunks=chunks, vocab=vocab_mod)
    depth = args.top_k * retrieve.CANDIDATE_POOL_MULT

    rows = []
    for i, item in enumerate(questions):
        q = str(item.get("question") or "")
        analysis = query_mod.analyze_query(
            q, entities=graph.get("entities") or [], aliases=graph.get("alias_index"),
            vocab=vocab_mod, context=retriever.query_context,
        )
        text = str(analysis.get("retrieval_query") or analysis.get("normalized_query") or "")
        bm25 = retrieve._norm_scores(index.search(text, top_k=depth)) if text else {}
        ent: dict[str, float] = {}
        names: dict[str, float] = {}
        for cand in analysis.get("entity_candidates") or []:
            eid = str(cand.get("entity_id") or "")
            rec = retriever.entities_by_id.get(eid) or {}
            terms = [str(rec.get("name") or cand.get("name") or "")]
            for a in rec.get("aliases") or []:
                t = a.get("text") if isinstance(a, dict) else a
                if str(t or "").strip():
                    terms.append(str(t).strip())
            for t in terms:
                names[t] = max(names.get(t, 0.0), float(cand.get("score") or 0.0))
            probe = " ".join(sorted({t for t in terms if t}))
            if not probe:
                continue
            w = float(cand.get("score") or 0.0)
            for cid, sc in retrieve._norm_scores(index.search(probe, top_k=depth)).items():
                ent[cid] = max(ent.get(cid, 0.0), sc * w)
        sp: dict[str, float] = {}
        cands = set(bm25) | set(ent)
        if names and cands:
            max_w = max(names.values()) or 1.0
            for cid in cands:
                spk = _speakers_of(chunks.get(cid) or {})
                if spk:
                    hit = sum(w for t, w in names.items() if t in spk)
                    if hit:
                        sp[cid] = hit / max_w
        rows.append((item, bm25, ent, sp))
        if (i + 1) % 80 == 0:
            print(f"  {i + 1}/{len(questions)}", flush=True)

    def run(use_speaker: bool) -> tuple[list[float], list[float]]:
        """返回每题的 (recall@20, 1/(rank+1))，只含有 gold 的题。"""
        rec, rr = [], []
        for item, bm25, ent, sp in rows:
            merged: dict[str, float] = defaultdict(float)
            for cid, sc in bm25.items():
                merged[cid] += sc
            for cid, sc in ent.items():
                merged[cid] += retrieve.ENTITY_CHANNEL_WEIGHT * sc
            if use_speaker:
                for cid, sc in sp.items():
                    merged[cid] += args.ws * sc
            ranked = [cid for cid, _ in sorted(merged.items(), key=lambda kv: (-kv[1], kv[0]))][
                : args.top_k
            ]
            golds = set(str(c) for c in item.get("gold_chunk_ids") or [])
            if not golds:
                continue
            rec.append(len(golds & set(ranked)) / len(golds))
            first = next((j for j, cid in enumerate(ranked) if cid in golds), None)
            rr.append(0.0 if first is None else 1.0 / (first + 1))
        return rec, rr

    rec_a, rr_a = run(False)
    rec_b, rr_b = run(True)
    print(f"\nn={len(rec_a)}  R@20 {sum(rec_a)/len(rec_a):.4f} → {sum(rec_b)/len(rec_b):.4f}   "
          f"MRR {sum(rr_a)/len(rr_a):.4f} → {sum(rr_b)/len(rr_b):.4f}")

    report("Recall@20 差值（ws=0.1 减 baseline）", [b - a for a, b in zip(rec_a, rec_b)])
    report("MRR 差值（ws=0.1 减 baseline）", [b - a for a, b in zip(rr_a, rr_b)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
