"""扫描实体通道权重 `ENTITY_CHANNEL_WEIGHT`（当前 0.5）。

## 为什么现在是重扫它的正确时机

这个权重当初定 0.5 时（G 阶段），候选池深度是 `top_k*3`。现在 `CANDIDATE_POOL_MULT`
已改成 10，**候选池变了，最优权重不一定还在 0.5**：池子更深意味着实体通道有更多
机会把弱匹配的块抬进来，权重偏高会更明显地伤害头部精度。

而且 `diagnose_channels.py` 已证明实体通道单独用很差（MRR 0.2255，而 bm25 是 0.4248），
说明它的价值主要在"补召回"，不在"定排名"。所以更可能的最优值比 0.5 更低。

## 口径

* 输出固定 top_k=20，`CANDIDATE_POOL_MULT` 固定 10（即生产口径）。
* 判据不变：**R@20 与 MRR 必须同时不劣**。在这里二者很可能同向（都在防止实体
  通道污染头部），所以若出现"召回降/MRR升"，意味着实体通道确实在补召回，
  那就需要给它**衰减排名而不是砍权重**（见 `sweep_entity_weight.py` 末尾说明）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="实体通道权重扫描")
    ap.add_argument("--generation", default="g-20261031T035231Z-4eafae")
    ap.add_argument("--gold-set", default="data/goldset/gold-v1.json")
    ap.add_argument("--limit", type=int, default=320)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--weights", nargs="+", type=float,
                    default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
    ap.add_argument("--json")
    args = ap.parse_args(argv)

    gold = json.loads(Path(args.gold_set).read_text(encoding="utf-8"))
    items = gold["items"] if isinstance(gold, dict) else gold
    import importlib.util

    spec = importlib.util.spec_from_file_location("_ask", Path(__file__).resolve().parent / "ask.py")
    ask = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ask)
    questions = ask._stratified_sample(items, args.limit)
    print(f"题 {len(questions)}  top_k={args.top_k}  pool_mult={retrieve.CANDIDATE_POOL_MULT}")

    gen = release.Generation(args.generation)
    chunks = {str(r["chunk_id"]): r for r in read_jsonl(config.chunks_path()) if r.get("chunk_id")}
    index, _meta = index_mod.load_index(gen.index_path)
    graph = json.loads(gen.graph_path.read_text(encoding="utf-8"))
    retriever = retrieve.Retriever(index, graph, {}, chunks=chunks, vocab=vocab_mod)

    depth = args.top_k * retrieve.CANDIDATE_POOL_MULT

    # 每题只算一次两条通道的归一分，之后各权重只重做加法（省掉重复检索）
    bm25_all: list[dict[str, float]] = []
    entity_all: list[dict[str, float]] = []
    for i, item in enumerate(questions):
        q = str(item.get("question") or "")
        analysis = query_mod.analyze_query(
            q, entities=graph.get("entities") or [], aliases=graph.get("alias_index"),
            vocab=vocab_mod, context=retriever.query_context,
        )
        text = str(analysis.get("retrieval_query") or analysis.get("normalized_query") or "")
        bm25_all.append(
            retrieve._norm_scores(index.search(text, top_k=depth)) if text else {}
        )
        ent: dict[str, float] = {}
        for cand in analysis.get("entity_candidates") or []:
            eid = str(cand.get("entity_id") or "")
            ent_rec = retriever.entities_by_id.get(eid) or {}
            terms = [str(ent_rec.get("name") or cand.get("name") or "")]
            for a in ent_rec.get("aliases") or []:
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

    print(f"\n{'权重':>6} {'R@5':>8} {'R@20':>8} {'R@50':>8} {'MRR':>8}  说明")
    print("-" * 62)
    rows = []
    base = None
    for w in args.weights:
        hits_per_q = []
        for bm25, ent in zip(bm25_all, entity_all):
            merged: dict[str, float] = defaultdict(float)
            for cid, sc in bm25.items():
                merged[cid] += sc
            for cid, sc in ent.items():
                merged[cid] += w * sc
            ranked = sorted(merged.items(), key=lambda kv: (-kv[1], kv[0]))
            hits_per_q.append([cid for cid, _ in ranked[: args.top_k]])
        m = metrics(hits_per_q, questions)
        if abs(w - 0.5) < 1e-9:
            base = m
        rows.append({"weight": w, **m})
        mark = "  ← 当前生产值" if abs(w - 0.5) < 1e-9 else ""
        print(f"{w:>6.2f} {m['r@5']:>8.4f} {m['r@20']:>8.4f} {m['r@50']:>8.4f} {m['mrr']:>8.4f}{mark}")

    if base:
        print("\n相对当前生产值 0.5 的差值：")
        for r in rows:
            if abs(r["weight"] - 0.5) < 1e-9:
                continue
            print(f"  w={r['weight']:<5.2f} ΔR@20 {r['r@20'] - base['r@20']:+.4f}  "
                  f"ΔMRR {r['mrr'] - base['mrr']:+.4f}")

    best = max(rows, key=lambda r: (r["mrr"], r["r@20"]))
    print(f"\nMRR 最优 w={best['weight']:.2f}  R@20 {best['r@20']:.4f}  MRR {best['mrr']:.4f}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps({"generation": args.generation, "top_k": args.top_k,
                        "questions": len(questions), "rows": rows},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
