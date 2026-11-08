"""两处决定性实验：上下文预算、标题通路。

R2 漏水（237 道进 top20 的题里 70 道 gold 没进 context，29.5%）的机制已定位：
`Retriever.build_context` 用 `estimate_tokens()` 记账，而它是**词法分词器**的
token 数（单字 + 相邻双字），190 个汉字算成 320 token ≈ 1.68 token/字。真实
LLM 分词器对中文约 0.6–1.0 token/字。所以 `max_context_tokens=6000` 这个数看着
很大，实际只装得下约 3,500 字、约 5 个块 —— 与实测 `n_ctx_units` 恒为 4–6 完全吻合。

本脚本一次跑完两件事（命中/扩展结果只算一次，反复复用，避免 4×13 分钟）：

A. 预算扫描：token_budget ∈ {6000, 9000, 12000, 18000, 24000}，看 R2/R4/R5 怎么变。
   同时报「实际字符数」「实际 LLM 近似 token 数」，把两套口径分开写清楚。
B. 标题通路：`answer.TITLE_ANSWER_PATH` True/False 各跑一遍（其余不动）。
   项目台账说标题通路已入库 +22 题、零回退，但当前代码里该常量是 False，
   必须实测确认它到底还开没开、开了值多少。

口径：全部用 gold-v1 前 300 道有 gold_chunk_ids 的题、top_k=20、graph_depth=1，
与 `pipeline.ask_once` 完全同序。R5 = 答案文本含 gold answer 或任一 acceptable_answer。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import answer as answer_mod  # noqa: E402
from arkrag import pipeline as pipeline_mod  # noqa: E402
from arkrag import query as query_mod  # noqa: E402
from arkrag import vocab as vocab_mod  # noqa: E402
from arkrag.retrieve import Retriever  # noqa: E402


def text_chars(units: list[dict]) -> int:
    return sum(len(str(u.get("title") or "")) + len(str(u.get("chapter") or ""))
               + len(str(u.get("text") or "")) for u in units)


def run_budget(retriever, answerer, cached, items, budget: int) -> dict:
    rows = []
    for item, c in zip(items, cached):
        qa, hits, expanded, gset = c["qa"], c["hits"], c["expanded"], c["gset"]
        q = str(item.get("question") or "")
        gold_answer = str(item.get("answer") or "")
        accepted = [str(a) for a in (item.get("acceptable_answers") or []) if a]
        ctx = retriever.build_context(
            hits, expanded["expansions"], query_analysis=qa, token_budget=budget
        )
        units = ctx["units"]
        ctx_ids = [str(u["chunk_id"]) for u in units]
        gold_in = any(cid in gset for cid in ctx_ids)
        ans = answerer.answer(
            q, ctx, generation_id="g-20261103T132152Z-06a741",
            top_score=float(c["top_score"]), query_analysis=qa,
            retrieval_stats=c["stats"],
        )
        text = str(ans.get("answer") or "").strip()
        hit_gold = bool(text) and any(a and a in text for a in ([gold_answer] + accepted if gold_answer else accepted))
        rows.append({
            "qid": str(item.get("question_id") or ""),
            "R1": c["base_rank"] is not None,
            "R2": gold_in,
            "R4": bool(text),
            "R5": hit_gold,
            "units": len(units),
            "tokens": ctx["stats"]["tokens_used"],
            "chars": text_chars(units),
            "truncated": len(ctx["stats"]["truncated_chunks"]),
        })
    return rows


def summarize(rows: list[dict], label: str) -> dict:
    n = len(rows)
    hit = [r for r in rows if r["R1"]]
    r2 = [r for r in hit if r["R2"]]
    out = {
        "label": label,
        "n": n,
        "R1": sum(1 for r in rows if r["R1"]),
        "R2": len(r2), "R2_den": len(hit),
        "R4": sum(1 for r in r2 if r["R4"]), "R4_den": len(r2),
        "R5": sum(1 for r in r2 if r["R5"]), "R5_den": len(r2),
        "R5_all": sum(1 for r in rows if r["R5"]),
        "units_mean": sum(r["units"] for r in rows) / n if n else 0,
        "tokens_mean": sum(r["tokens"] for r in rows) / n if n else 0,
        "chars_mean": sum(r["chars"] for r in rows) / n if n else 0,
        "trunc_mean": sum(r["truncated"] for r in rows) / n if n else 0,
    }
    print(f"{label:<22}{out['R2']:>4}/{out['R2_den']:<4} "
          f"{100.0*out['R2']/max(1,out['R2_den']):>5.1f}%  "
          f"{out['R5']:>4}/{out['R5_den']:<4} "
          f"{100.0*out['R5']/max(1,out['R5_den']):>5.1f}%  "
          f"{out['R5_all']:>4}/{n:<4} "
          f"units {out['units_mean']:>4.1f}  chars {out['chars_mean']:>6.0f}  "
          f"pseudo-tok {out['tokens_mean']:>6.0f}  截断 {out['trunc_mean']:>5.1f}",
          flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generation", default="g-20261103T132152Z-06a741")
    ap.add_argument("--gold-set", default="data/goldset/gold-v1.json")
    ap.add_argument("--limit", type=int, default=320)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--graph-depth", type=int, default=1)
    ap.add_argument("--budgets", nargs="+", type=int, default=[6000, 9000, 12000, 18000, 24000])
    ap.add_argument("--json", default="data/diag_budget_title.json")
    args = ap.parse_args()

    loaded = pipeline_mod.load_generation(args.generation)
    graph = loaded["graph"]
    chunks = loaded["chunks"]
    entities = graph.get("entities") or []
    pipeline_mod.resolve_chunks(loaded["generation"], loaded["manifest"])
    retriever = Retriever(
        loaded["index"], graph, loaded["adjacency"],
        chunks=chunks, assertions=loaded["assertions"], vocab=vocab_mod,
    )
    answerer = answer_mod.ExtractiveAnswerer()

    gold = json.loads((ROOT / args.gold_set).read_text(encoding="utf-8"))
    items = [g for g in (gold.get("items") or []) if g.get("gold_chunk_ids")][: args.limit]

    cached = []
    for i, item in enumerate(items, 1):
        q = str(item.get("question") or "")
        gset = {str(c) for c in item["gold_chunk_ids"]}
        qa = query_mod.analyze_query(
            q, entities=entities, aliases=graph.get("alias_index"), vocab=vocab_mod
        )
        found = retriever.search(qa, top_k=args.top_k)
        hits = found["hits"]
        ids = [str(h["chunk_id"]) for h in hits]
        base_rank = next((j for j, c in enumerate(ids, 1) if c in gset), None)
        seed_ids = [str(c.get("entity_id") or "") for c in (qa.get("entity_candidates") or [])]
        seed_ids = [s for s in seed_ids if s]
        expanded = retriever.expand(
            seed_ids, depth=args.graph_depth, query_analysis=qa,
            seed_chunk_ids=[h["chunk_id"] for h in hits],
        )
        cached.append({
            "qa": qa, "hits": hits, "expanded": expanded, "gset": gset,
            "base_rank": base_rank,
            "top_score": float(found["stats"].get("top_score") or 0.0),
            "stats": found["stats"],
        })
        if i % 50 == 0 or i == len(items):
            print(f"  预热 {i}/{len(items)}", flush=True)

    results = {}
    print(f"\n=== A. 预算扫描（TITLE_ANSWER_PATH={answer_mod.TITLE_ANSWER_PATH}）===")
    print(f"{'预算':<22}{'R2':>10}{'R5|R2 内':>11}{'R5 全量':>11}"
          f"   实际装入")
    for b in args.budgets:
        rows = run_budget(retriever, answerer, cached, items, b)
        results[f"budget_{b}"] = summarize(rows, f"budget={b}")

    print("\n=== B. 标题通路开关（预算 6000，其余不动）===")
    original = answer_mod.TITLE_ANSWER_PATH
    try:
        for flag in (False, True):
            answer_mod.TITLE_ANSWER_PATH = flag
            rows = run_budget(retriever, answerer, cached, items, 6000)
            results[f"title_{flag}"] = summarize(rows, f"TITLE={flag}")
    finally:
        answer_mod.TITLE_ANSWER_PATH = original

    print("\n=== C. 预算 24000 + 标题通路 True ===")
    try:
        answer_mod.TITLE_ANSWER_PATH = True
        rows = run_budget(retriever, answerer, cached, items, 24000)
        results["budget24000_title_true"] = summarize(rows, "24000+title")
    finally:
        answer_mod.TITLE_ANSWER_PATH = original

    out = ROOT / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n落盘 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
