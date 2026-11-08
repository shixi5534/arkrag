"""找出「什么时候该把向量通道叫进来」的查询期信号。

背景：全局融合（minmax / DBSF / RRF k=1,2 / 尾部补位）全部实测失败，原因是
任何全局改动都会碰坏已经答对的 237 道题。实测数据显示向量真正有用的情况是
一个子集：base top-20 里没有 gold 的 63 道题里，向量 top-20 救回 14 道
（且集中在 direct_relation 7 道、time_conflict 4 道）。

所以问题不是「怎么融合」，而是「什么时候才该让向量说话」。这个脚本只做一件事：
用**查询期就能拿到的信号**（不偷看 gold、不偷看 category）预测「base 会不会漏」，
并给出每个候选信号的 AUC 与不同阈值下的（救回 / 误伤）账。

误伤的定义要写死：把向量并集追加在 base 后面时，base 已经答对的题**不会**因为
并集掉出前 20（追加在尾部，名次只可能不变）。所以「误伤」在这个结构里真实含义是
「白白多算了向量的延迟」。这一点决定了：只要触发，R@20 单调不降。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import embedding as embedding_mod  # noqa: E402
from arkrag import pipeline as pipeline_mod  # noqa: E402
from arkrag import query as query_mod  # noqa: E402
from arkrag import vector_store  # noqa: E402
from arkrag import vocab as vocab_mod  # noqa: E402
from arkrag.retrieve import Retriever  # noqa: E402


def auc(scores: list[float], labels: list[int]) -> float:
    """Mann-Whitney 形式：正类分数高于负类的概率。并列按 0.5 计。"""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generation", default="g-20261103T132152Z-06a741")
    ap.add_argument("--gold-set", default="data/goldset/gold-v1.json")
    ap.add_argument("--limit", type=int, default=320)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--vector-k", type=int, default=100)
    ap.add_argument("--json", default="data/eval_gate_signals.json")
    args = ap.parse_args()

    loaded = pipeline_mod.load_generation(args.generation)
    index = loaded["index"]
    graph = loaded["graph"]
    chunks = loaded["chunks"]
    entities = graph.get("entities") or []
    content_digest = loaded["index_meta"]["index_content_digest"]
    pipeline_mod.resolve_chunks(loaded["generation"], loaded["manifest"])

    retriever = Retriever(
        index, graph, loaded["adjacency"],
        chunks=chunks, assertions=loaded["assertions"], vocab=vocab_mod,
    )

    slug = vector_store._slug(embedding_mod.model_name())
    vec_bin, vec_meta = vector_store.vector_paths(slug, content_digest)
    matrix, vmeta = vector_store.load_vectors(vec_bin, vec_meta)
    vec = vector_store.VectorIndex(matrix, vmeta)

    gold = json.loads((ROOT / args.gold_set).read_text(encoding="utf-8"))
    items = [g for g in (gold.get("items") or []) if g.get("gold_chunk_ids")][: args.limit]

    rows = []
    for i, item in enumerate(items, 1):
        q = str(item.get("question") or "")
        gset = {str(c) for c in item["gold_chunk_ids"]}
        cat = str(item.get("category") or "")

        qa = query_mod.analyze_query(
            q, entities=entities, aliases=graph.get("alias_index"), vocab=vocab_mod
        )
        found = retriever.search(qa, top_k=args.top_k)
        hits = found["hits"]
        stats = found["stats"]
        ids = [str(h["chunk_id"]) for h in hits]
        scores = [float(h["score"]) for h in hits]

        base_rank = next((j for j, c in enumerate(ids, 1) if c in gset), None)

        qv = embedding_mod.embed_batch([q])[0]
        vhits = vec.search(qv, top_k=args.vector_k)
        vids = [str(c) for c, _ in vhits]
        vec_hit20 = any(c in gset for c in vids[: args.top_k])

        top = scores[0] if scores else 0.0
        n = len(scores)
        s20 = scores[min(n, args.top_k) - 1] if n >= args.top_k else (scores[-1] if scores else 0.0)
        s2 = scores[1] if n > 1 else 0.0

        ecand = qa.get("entity_candidates") or []
        rows.append({
            "qid": str(item.get("question_id") or i),
            "category": cat,
            "base_rank": base_rank,
            "base_hit20": base_rank is not None,
            "vec_hit20": vec_hit20,
            "rescued": base_rank is None and vec_hit20,
            # —— 以下是查询期可得的信号 ——
            "top_score": top,
            "second_score": s2,
            "score20": s20,
            "margin_top_to_20": top - s20,
            "margin_top_to_2": top - s2,
            "ratio_top_to_20": (top / s20) if s20 > 1e-9 else float("inf"),
            "n_hits": n,
            "n_entity_cand": len(ecand),
            "entity_top_score": max([float(c.get("score") or 0.0) for c in ecand] or [0.0]),
            "bm25_terms": int(stats.get("bm25_terms") or 0),
            "speaker_boosted": int(stats.get("speaker_boosted") or 0),
            "cooc_boosted": int(stats.get("cooc_boosted") or 0),
            "candidates": int(stats.get("candidates") or 0),
        })
        if i % 25 == 0 or i == len(items):
            print(f"  {i}/{len(items)}", flush=True)

    out_path = ROOT / args.json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"generation": args.generation, "rows": rows},
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    n = len(rows)
    miss = [r for r in rows if not r["base_hit20"]]
    resc = [r for r in rows if r["rescued"]]
    print(f"\n题 {n}，base 漏 {len(miss)}，其中向量能救 {len(resc)}")
    from collections import Counter
    print("救回题的类别：", dict(Counter(r["category"] for r in resc)))
    print("漏掉题的类别：", dict(Counter(r["category"] for r in miss)))

    # 信号方向：我们希望「分数越低越该触发」。AUC 里把信号取负，
    # label=1 表示 base 漏。这样 AUC>0.5 表示低分确实预示漏。
    label = [0 if r["base_hit20"] else 1 for r in rows]
    signals = [
        ("top_score", -1), ("second_score", -1), ("score20", -1),
        ("margin_top_to_20", -1), ("margin_top_to_2", -1),
        ("ratio_top_to_20", -1), ("n_entity_cand", -1),
        ("entity_top_score", -1), ("bm25_terms", -1), ("candidates", -1),
    ]
    print("\n=== 信号 AUC（预测「base 会不会漏」，1=漏）===")
    scores_auc = {}
    for name, sign in signals:
        vals = []
        for r in rows:
            v = r[name]
            if isinstance(v, float) and v == float("inf"):
                v = 1e9
            vals.append(sign * float(v))
        a = auc(vals, label)
        scores_auc[name] = a
        print(f"  {name:<20}{a:.4f}")

    # 触发阈值扫描：触发题里包含多少救回题、总共触发多少题
    print("\n=== 单信号阈值扫描（触发=预测会漏）===")
    for name, sign in signals:
        vals = sorted({r[name] for r in rows if not (isinstance(r[name], float) and r[name] == float("inf"))})
        if len(vals) < 4:
            continue
        best = None
        for q in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
            cut = vals[int(q * (len(vals) - 1))]
            trig = [r for r in rows if (sign * float(r[name])) >= (sign * cut)]
            got = sum(1 for r in trig if r["rescued"])
            cost = len(trig)
            if got == len(resc) or (best is None and cost <= n * 0.35):
                pass
            rec = (got, cost, q, cut)
            if best is None or (rec[0], -rec[1]) > (best[0], -best[1]):
                best = rec
        if best:
            got, cost, q, cut = best
            print(f"  {name:<20} 分位 {q:.2f} cut={cut:<12.4g} "
                  f"触发 {cost:>3}/{n} 救回 {got}/{len(resc)}")

    print(f"\n落盘 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
