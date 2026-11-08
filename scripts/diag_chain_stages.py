"""测「检索已经进了前 20 之后」到底卡在哪一环。

动机（实测分布，不是猜的）：237 道 base 命中题里，gold 名次第 1 名 100 题(42.2%)、
前 3 名 144 题(60.8%)、前 5 名 171 题(72.2%)。检索层已经不是瓶颈，但整条链路的
作答正确率远低于此。所以必须把「命中」到「答对」之间的每一环拆开数清楚。

链路（`pipeline.ask_once` 的真实顺序）：
  hits(20) ──build_context──> units ──answer──> answer

`ExtractiveAnswerer` 不是 LLM：它的 claim **只来自 `unit["assertions"]`**，而
`build_context` 只给通过 `_assertion_relevant(rec, focus_ids, hint_predicates)` 的
assertion 挂到 unit 上。`focus_ids` 来自 `qa["entity_candidates"]`。**查询没识别出
实体 ⇒ focus_ids 为空 ⇒ 只有 hint_predicates 能救 ⇒ 大量 unit 挂不上 assertion
⇒ 无 claim ⇒ 拒答 `no_supported_claim`。**

本脚本逐题数这五环，每环都独立记一笔，从而指出第一处断掉的地方：
  R1 gold 是否进 hits 前 20            （检索）
  R2 gold 块是否进 context units       （组装/预算）
  R3 gold 块所在 unit 是否挂上 assertion（证据锚定）
  R4 是否产出非空答案                  （作答）
  R5 答案是否含 gold 答案串            （正确性）
并记录焦点实体数与拒答原因，用来验证上面那条「focus_ids 空 ⇒ 挂不上」的机制。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import pipeline as pipeline_mod  # noqa: E402
from arkrag import query as query_mod  # noqa: E402
from arkrag import vocab as vocab_mod  # noqa: E402
from arkrag.retrieve import Retriever  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generation", default="g-20261103T132152Z-06a741")
    ap.add_argument("--gold-set", default="data/goldset/gold-v1.json")
    ap.add_argument("--limit", type=int, default=320)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--graph-depth", type=int, default=1)
    ap.add_argument("--json", default="data/diag_chain_stages.json")
    args = ap.parse_args()

    loaded = pipeline_mod.load_generation(args.generation)
    index = loaded["index"]
    graph = loaded["graph"]
    chunks = loaded["chunks"]
    entities = graph.get("entities") or []
    pipeline_mod.resolve_chunks(loaded["generation"], loaded["manifest"])

    retriever = Retriever(
        index, graph, loaded["adjacency"],
        chunks=chunks, assertions=loaded["assertions"], vocab=vocab_mod,
    )
    from arkrag.answer import ExtractiveAnswerer

    answerer = ExtractiveAnswerer()

    gold = json.loads((ROOT / args.gold_set).read_text(encoding="utf-8"))
    items = [g for g in (gold.get("items") or []) if g.get("gold_chunk_ids")][: args.limit]

    rows = []
    for i, item in enumerate(items, 1):
        q = str(item.get("question") or "")
        gset = {str(c) for c in item["gold_chunk_ids"]}
        gold_answer = str(item.get("answer") or "")
        accepted_answers = [str(a) for a in (item.get("acceptable_answers") or []) if a]

        qa = query_mod.analyze_query(
            q, entities=entities, aliases=graph.get("alias_index"), vocab=vocab_mod
        )
        ecand = qa.get("entity_candidates") or []
        found = retriever.search(qa, top_k=args.top_k)
        hits = found["hits"]
        ids = [str(h["chunk_id"]) for h in hits]
        base_rank = next((j for j, c in enumerate(ids, 1) if c in gset), None)

        seed_ids = [str(c.get("entity_id") or "") for c in ecand]
        seed_ids = [s for s in seed_ids if s]
        expanded = retriever.expand(
            seed_ids, depth=args.graph_depth, query_analysis=qa,
            seed_chunk_ids=[h["chunk_id"] for h in hits],
        )
        context = retriever.build_context(
            hits, expanded["expansions"], query_analysis=qa
        )
        units = context["units"]
        ctx_ids = [str(u["chunk_id"]) for u in units]

        # R2：gold 进 context 了吗；若进了，它的 unit 挂了几条 assertion
        gold_in_ctx = [c for c in ctx_ids if c in gset]
        gold_unit_assertions = None
        gold_ctx_pos = None
        if gold_in_ctx:
            first = gold_in_ctx[0]
            gold_ctx_pos = ctx_ids.index(first) + 1
            for u in units:
                if str(u["chunk_id"]) == first:
                    gold_unit_assertions = len(u.get("assertions") or [])
                    break

        ans = answerer.answer(
            q, context,
            generation_id=args.generation,
            top_score=float(found["stats"].get("top_score") or 0.0),
            query_analysis=qa,
            retrieval_stats=found["stats"],
        )
        text = str(ans.get("answer") or "").strip()
        hit_gold = bool(text) and any(
            a and a in text for a in ([gold_answer] + accepted_answers if gold_answer else accepted_answers)
        )

        rows.append({
            "qid": str(item.get("question_id") or i),
            "category": str(item.get("category") or ""),
            "base_rank": base_rank,
            "R1_hit20": base_rank is not None,
            "n_entity_cand": len(ecand),
            "n_hint_pred": len(qa.get("relation_hints") or []),
            "should_answer": bool(qa.get("should_answer", True)),
            "n_ctx_units": len(units),
            "ctx_tokens": context["stats"]["tokens_used"],
            "truncated": len(context["stats"]["truncated_chunks"]),
            "R2_gold_in_ctx": bool(gold_in_ctx),
            "gold_ctx_pos": gold_ctx_pos,
            "gold_unit_assertions": gold_unit_assertions,
            "R3_gold_unit_has_assertion": bool((gold_unit_assertions or 0) > 0),
            "total_assertions_in_ctx": sum(len(u.get("assertions") or []) for u in units),
            "abstained": bool(ans.get("abstained")),
            "abstain_reason": str(ans.get("abstain_reason") or ""),
            "n_candidates": int(ans["stats"].get("candidates") or 0),
            "n_accepted": int(ans["stats"].get("accepted") or 0),
            "R4_has_answer": bool(text),
            "R5_answer_hits_gold": hit_gold,
            "dropped_no_citation": int(ans["stats"].get("dropped_no_citation") or 0),
            "dropped_unsupported": int(ans["stats"].get("dropped_unsupported") or 0),
            "dropped_ungrounded": int(ans["stats"].get("dropped_ungrounded") or 0),
        })
        if i % 25 == 0 or i == len(items):
            print(f"  {i}/{len(items)}", flush=True)

    out = ROOT / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"generation": args.generation, "top_k": args.top_k,
                    "graph_depth": args.graph_depth, "rows": rows},
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    n = len(rows)
    hit = [r for r in rows if r["R1_hit20"]]

    def rate(num, den, label):
        pct = 100.0 * num / den if den else 0.0
        print(f"  {label:<44}{num:>4}/{den:<4} = {pct:5.1f}%")

    print(f"\n=== 全 {n} 题，逐环漏水（括号内为相对于上一环）===")
    rate(sum(1 for r in rows if r["R1_hit20"]), n, "R1 gold 进 hits top20")
    rate(sum(1 for r in hit if r["R2_gold_in_ctx"]), len(hit), "R2 gold 进 context（在 R1 命中里）")
    r2 = [r for r in hit if r["R2_gold_in_ctx"]]
    rate(sum(1 for r in r2 if r["R3_gold_unit_has_assertion"]), len(r2),
         "R3 gold unit 挂上 assertion（在 R2 里）")
    rate(sum(1 for r in r2 if r["R4_has_answer"]), len(r2), "R4 产出了非空答案（在 R2 里）")
    rate(sum(1 for r in r2 if r["R5_answer_hits_gold"]), len(r2),
         "R5 答案含 gold 答案串（在 R2 里）")
    rate(sum(1 for r in rows if r["R5_answer_hits_gold"]), n, "R5 端到端答对（全量）")

    print("\n=== 拒答原因分布 ===")
    for k, v in Counter(r["abstain_reason"] or "(未拒答)" for r in rows).most_common():
        print(f"  {k:<40}{v:>4}")

    print("\n=== 焦点实体数 vs R3 挂上 assertion ===")
    for bucket, lo, hi in (("0 个", 0, 0), ("1 个", 1, 1), ("2 个", 2, 2), ("3+ 个", 3, 99)):
        grp = [r for r in rows if lo <= r["n_entity_cand"] <= hi]
        if not grp:
            continue
        has = sum(1 for r in grp if r["R3_gold_unit_has_assertion"])
        ans_ok = sum(1 for r in grp if r["R5_answer_hits_gold"])
        print(f"  焦点实体 {bucket:<6}n={len(grp):<4} "
              f"R3 挂上 {has:>3}/{len(grp):<4} 端到端答对 {ans_ok:>3}/{len(grp)}")

    print("\n=== gold 名次 vs 端到端答对 ===")
    for lo, hi in ((1, 1), (2, 3), (4, 5), (6, 10), (11, 20)):
        grp = [r for r in rows if r["base_rank"] and lo <= r["base_rank"] <= hi]
        if not grp:
            continue
        ok = sum(1 for r in grp if r["R5_answer_hits_gold"])
        r3 = sum(1 for r in grp if r["R3_gold_unit_has_assertion"])
        print(f"  名次 {lo:>2}-{hi:<2} n={len(grp):<4} R3 挂上 {r3:>3}  端到端答对 {ok:>3} "
              f"({100.0*ok/len(grp):5.1f}%)")

    print("\n=== gold 在 context 的位置分布（R2 命中的题）===")
    pos = Counter()
    for r in r2:
        p = r["gold_ctx_pos"] or 0
        key = ("1" if p == 1 else "2" if p == 2 else "3" if p == 3
               else "4-5" if 4 <= p <= 5 else "6-10" if 6 <= p <= 10
               else "11+" if p > 10 else "缺失")
        pos[key] += 1
    for k in ("1", "2", "3", "4-5", "6-10", "11+", "缺失"):
        if pos[k]:
            print(f"  第 {k:<5}{pos[k]:>4} 题  {100.0*pos[k]/len(r2):5.1f}%")

    print(f"\n落盘 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
