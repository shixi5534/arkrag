"""候选池深度实验：`Retriever.search` 内部把 bm25 候选拉多深。

## 为什么这个实验先做

`diagnose_ranking.py` 里我发现自己的脚本写错了一处口径：把「`search(top_k=100)`
再截前 20」标成了"生产口径"，实测 R@20 = 0.6850，而真实生产口径（`top_k=20`）
是 0.6575。**差 0.0275 全是候选池深度带来的**——`Retriever.search` 内部是
`index.search(text, top_k=top_k*3)`，所以 `top_k=20` 只拉 60 个候选，
`top_k=100` 拉 300 个，融合后截前 20 时后者能捞回不少。

这不是评测技巧，是**真实可用的杠杆**：用户看到的仍然是前 20 条，
只是我们内部多看了几眼再排序。代价只有排序开销（H2 已把 search 压到
P50 111 ms / P95 167 ms，多排序几十条几乎免费）。

## 要扫的东西

1. `bm25_mult`：内部 bm25 拉 `top_k * mult` 个候选（当前硬编码 3）。
2. 实体通道要不要一起加深（现在与 bm25 用同一个 `top_k*3`）。

## 纪律

* 输出 top_k 固定 20（生产口径），只改内部候选深度——否则比的是两件事。
* 只有 R@20 与 MRR **同时**不劣才算数。
* 这个实验**不引入任何新依赖、不改索引格式**，所以即使收益一般也可以安全采纳。
"""

from __future__ import annotations

import argparse
import json
import sys
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

ENTITY_WEIGHT = retrieve.ENTITY_CHANNEL_WEIGHT


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def metrics(hits_per_q: list[list[str]], questions: list[dict]) -> dict:
    r5, r20, rr = [], [], []
    for hits, item in zip(hits_per_q, questions):
        gold = [str(c) for c in item.get("gold_chunk_ids") or []]
        if not gold:
            continue
        gs = set(gold)
        r5.append(len(gs & set(hits[:5])) / len(gs))
        r20.append(len(gs & set(hits[:20])) / len(gs))
        first = next((i for i, cid in enumerate(hits) if cid in gs), None)
        rr.append(0.0 if first is None else 1.0 / (first + 1))
    n = len(r20) or 1
    return {"r@5": round(sum(r5) / n, 4), "r@20": round(sum(r20) / n, 4),
            "mrr": round(sum(rr) / n, 4), "n": len(r20)}


def search_with_mult(retriever, query_analysis: dict, *, top_k: int, mult: int,
                     entity_mult: int | None = None) -> list[str]:
    """**不改生产代码**的等价实现：与 `Retriever.search` 同公式，只把候选深度参数化。

    公式逐字照抄 `retrieve.py::search`：
        bm25 = norm(index.search(text, top_k=depth))
        entity_best[cid] = max(entity_best[cid], norm(index.search(probe, top_k=depth)) * weight)
        final = bm25 + 0.5 * entity
    这样实验结论能直接映射回生产代码的一行改动（`top_k * 3` → `top_k * mult`）。
    """
    em = mult if entity_mult is None else entity_mult
    if not query_analysis.get("should_answer", True):
        return []
    text = str(query_analysis.get("retrieval_query")
               or query_analysis.get("normalized_query") or "")
    depth = max(top_k, top_k * mult)
    bm25 = retrieve._norm_scores(retriever.index.search(text, top_k=depth)) if text else {}

    entity_best: dict[str, float] = {}
    for cand in query_analysis.get("entity_candidates") or []:
        eid = str(cand.get("entity_id") or "")
        ent = retriever.entities_by_id.get(eid) or {}
        terms = [str(ent.get("name") or cand.get("name") or "")]
        for item in ent.get("aliases") or []:
            t = item.get("text") if isinstance(item, dict) else item
            if str(t or "").strip():
                terms.append(str(t).strip())
        probe = " ".join(sorted({t for t in terms if t}))
        if not probe:
            continue
        weight = float(cand.get("score") or 0.0)
        edepth = max(top_k, top_k * em)
        for cid, score in retrieve._norm_scores(
            retriever.index.search(probe, top_k=edepth)
        ).items():
            entity_best[cid] = max(entity_best.get(cid, 0.0), score * weight)

    merged: dict[str, float] = defaultdict_float()
    for cid, score in bm25.items():
        merged[cid] += score
    for cid, score in entity_best.items():
        merged[cid] += ENTITY_WEIGHT * score
    ranked = sorted(merged.items(), key=lambda kv: (-kv[1], kv[0]))
    return [cid for cid, _ in ranked[:top_k]]


def defaultdict_float():
    from collections import defaultdict

    return defaultdict(float)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="候选池深度 × 实体通道深度")
    ap.add_argument("--generation", default="g-20261031T035231Z-4eafae")
    ap.add_argument("--gold-set", default="data/goldset/gold-v1.json")
    ap.add_argument("--limit", type=int, default=320)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--mults", nargs="+", type=int, default=[3, 5, 10, 20, 50])
    args = ap.parse_args(argv)

    gold = json.loads(Path(args.gold_set).read_text(encoding="utf-8"))
    items = gold["items"] if isinstance(gold, dict) else gold
    import importlib.util

    spec = importlib.util.spec_from_file_location("_ask", Path(__file__).resolve().parent / "ask.py")
    ask = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ask)
    questions = ask._stratified_sample(items, args.limit)
    print(f"题 {len(questions)}  top_k={args.top_k}（输出固定，只改内部候选深度）")

    gen = release.Generation(args.generation)
    chunks = {str(r["chunk_id"]): r for r in read_jsonl(config.chunks_path()) if r.get("chunk_id")}
    index, _meta = index_mod.load_index(gen.index_path)
    graph = json.loads(gen.graph_path.read_text(encoding="utf-8"))
    retriever = retrieve.Retriever(index, graph, {}, chunks=chunks, vocab=vocab_mod)

    analyses = []
    for item in questions:
        q = str(item.get("question") or "")
        analyses.append(
            query_mod.analyze_query(
                q, entities=graph.get("entities") or [], aliases=graph.get("alias_index"),
                vocab=vocab_mod, context=retriever.query_context,
            )
        )

    # 生产口径（mult=3）作为基准，用它自己的 metrics 当分母
    base_hits = [
        [str(h["chunk_id"]) for h in retriever.search(a, top_k=args.top_k)["hits"]]
        for a in analyses
    ]
    base = metrics(base_hits, questions)
    print(f"\n生产口径（内部 mult=3）  R@5 {base['r@5']:.4f}  R@20 {base['r@20']:.4f}  MRR {base['mrr']:.4f}")

    print(f"\n{'mult':>6} {'实体mult':>10} {'R@5':>8} {'R@20':>8} {'MRR':>8}  {'ΔR@20':>8} {'ΔMRR':>8}  判定")
    print("-" * 84)
    rows = []
    for mult in args.mults:
        for emult in (None, mult):
            label = mult if emult is None else f"{mult}/{emult}"
            hits = [
                search_with_mult(retriever, a, top_k=args.top_k, mult=mult, entity_mult=emult)
                for a in analyses
            ]
            m = metrics(hits, questions)
            d20, dm = round(m["r@20"] - base["r@20"], 4), round(m["mrr"] - base["mrr"], 4)
            verdict = ("★两者都≥基线" if d20 >= 0 and dm >= 0
                       else ("召回升/MRR降" if d20 > 0 else ("召回降/MRR升" if dm > 0 else "两者都降")))
            rows.append({"mult": mult, "entity_mult": emult if emult else "同左", **m,
                         "d_r@20": d20, "d_mrr": dm, "verdict": verdict})
            print(f"{mult:>6} {str(emult or '同左'):>10} {m['r@5']:>8.4f} {m['r@20']:>8.4f} "
                  f"{m['mrr']:>8.4f}  {d20:>+8.4f} {dm:>+8.4f}  {verdict}")

    winners = [r for r in rows if r["d_r@20"] >= 0 and r["d_mrr"] >= 0]
    print("\n" + "=" * 84)
    if winners:
        best = max(winners, key=lambda r: (r["d_r@20"] + r["d_mrr"]))
        print(f"结论：{len(winners)} 个组合两者都不劣；最好 {best['mult']}/{best['entity_mult']} "
              f"ΔR@20 {best['d_r@20']:+.4f} ΔMRR {best['d_mrr']:+.4f}")
    else:
        print("结论：加深候选池没有免费的午餐。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
