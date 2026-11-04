"""上下文预算截断量化（纯分析，不改参数、不做端到端实验）。

做法：对每道题**重放同一批 hits/expansions**，先用极大预算跑一次 `build_context`
拿到「完整有序单元表 + 每单元 token 成本」，再用同一个贪心循环在不同预算下重算。

这不等于端到端实测：没有重新检索、没有重新回答、没有验证回答层是否变好。
它精确回答的是「上下文装配这一步在不同预算下会放进哪些块」。

自检：预算 6000 的重算结果必须逐题等于 `data/a_triage_all320.jsonl` 里记录的
`context_units`，否则说明重放路径与生产不一致，结果作废。
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag.pipeline import build_pipeline, load_generation  # noqa: E402
from arkrag import query as query_mod  # noqa: E402
from arkrag import pipeline as pipeline_mod  # noqa: E402

GID = "g-20261103T132152Z-06a741"
TOP_K = 20
GRAPH_DEPTH = 1
TRACE = ROOT / "data" / "a_triage_all320.jsonl"

BIG = 10**9
BUDGETS = [6000, 7000, 8000, 10000, 12000, 15000]


def doc_of(chunk_id: str) -> str:
    return str(chunk_id).rsplit("#", 1)[0]


def replay(ordered: list[dict], budget: int) -> tuple[list[str], int, list[str]]:
    """完全复刻 src/arkrag/retrieve.py:855-869 的贪心填充。

    注意：装不下时是 `continue` 而不是 `break`，所以被跳过的单元不会终止扫描，
    后面更小的单元仍可能挤进来。这一点是本次分析的核心事实。
    """
    kept: list[str] = []
    truncated: list[str] = []
    used = 0
    for unit in ordered:
        cost = unit["tokens"]
        if used + cost > budget:
            truncated.append(unit["chunk_id"])
            continue
        used += cost
        kept.append(unit["chunk_id"])
    return kept, used, truncated


def main() -> int:
    rows = [json.loads(line) for line in TRACE.open(encoding="utf-8")]
    by_id = {r["question_id"]: r for r in rows}
    targets = [r for r in rows if r["primary"] == "上下文未采用"]
    print(f"目标题数（primary=上下文未采用）: {len(targets)}", flush=True)

    loaded = load_generation(GID)
    pipe = build_pipeline(loaded, top_k=TOP_K, graph_depth=GRAPH_DEPTH)
    retriever = pipe["retriever"]
    entities = pipe.get("entities") or retriever.graph.get("entities") or []
    vocab = pipeline_mod.vocab

    per_question: list[dict] = []
    selfcheck_fail: list[str] = []

    for i, rec in enumerate(targets, 1):
        qid = rec["question_id"]
        qa = query_mod.analyze_query(
            rec["question"],
            entities=entities,
            aliases=retriever.graph.get("alias_index"),
            vocab=vocab,
            context=getattr(retriever, "query_context", None),
        )
        hits = retriever.search(qa, top_k=TOP_K)["hits"]
        expansions = retriever.expand(
            [c["entity_id"] for c in qa.get("entity_candidates") or []],
            depth=GRAPH_DEPTH,
            query_analysis=qa,
            seed_chunk_ids=[h["chunk_id"] for h in hits],
        )["expansions"]

        full = retriever.build_context(hits, expansions, token_budget=BIG, query_analysis=qa)
        ordered = [dict(u) for u in full["units"]]

        kept6, used6, trunc6 = replay(ordered, 6000)
        recorded = list(rec["context_units"])
        if kept6 != recorded:
            selfcheck_fail.append(qid)

        gold = [g for g in rec["gold_chunk_ids"] if g]
        gold_set = set(gold)
        ctx_docs = {doc_of(c) for c in kept6}

        # gold 在有序表里的位置与被跳过的事实
        pos = {u["chunk_id"]: idx for idx, u in enumerate(ordered)}
        gold_info = {}
        for g in gold:
            gold_info[g] = {
                "pos": pos.get(g),
                "tokens": next((u["tokens"] for u in ordered if u["chunk_id"] == g), None),
                "kept@6000": g in kept6,
                "sibling_in_ctx": doc_of(g) in ctx_docs,
            }

        # 被跳过之后，是否有更低优先级的单元被保留（= 大块被小块顶掉）
        skipped_but_later_kept = []
        for idx, unit in enumerate(ordered):
            if unit["chunk_id"] not in trunc6:
                continue
            later = [u["chunk_id"] for u in ordered[idx + 1:] if u["chunk_id"] in kept6]
            if later:
                skipped_but_later_kept.append(
                    {"skipped": unit["chunk_id"], "tokens": unit["tokens"], "later_kept": later[:3]}
                )

        need = None
        for g in gold:
            if g in kept6:
                continue
            # gold 要进来，最小预算是多少（保持同一顺序、同一贪心）
            for b in range(6000, 40001, 100):
                if g in replay(ordered, b)[0]:
                    need = b if need is None else min(need, b)
                    break

        per_question.append(
            {
                "question_id": qid,
                "category": rec["category"],
                "gold": gold,
                "gold_info": gold_info,
                "ordered_all": [
                    {"chunk_id": u["chunk_id"], "tokens": u["tokens"], "score": round(u["score"], 4)}
                    for u in ordered
                ],
                "units_total": len(ordered),
                "tokens_all": sum(u["tokens"] for u in ordered),
                "kept@6000": kept6,
                "used@6000": used6,
                "truncated@6000": trunc6,
                "min_budget_for_gold": need,
                "skipped_but_later_kept": skipped_but_later_kept,
                "sibling_in_ctx_any": any(v["sibling_in_ctx"] for v in gold_info.values()),
                "by_budget": {
                    str(b): {
                        "kept": len(replay(ordered, b)[0]),
                        "used": replay(ordered, b)[1],
                        "gold_in": [g for g in gold if g in replay(ordered, b)[0]],
                    }
                    for b in BUDGETS
                },
            }
        )
        if i % 10 == 0:
            print(f"  ...{i}/{len(targets)}", flush=True)

    if selfcheck_fail:
        print(f"[自检失败] {len(selfcheck_fail)} 题重放与记录不一致: {selfcheck_fail[:5]}", file=sys.stderr)
        return 4
    print("[自检通过] 预算 6000 的重算逐题等于记录的 context_units", flush=True)

    # ---------------- 汇总结论 ----------------
    n = len(per_question)
    print(f"\n=== 1. 提高预算能救回多少题（gold 至少一块进上下文）===")
    print(f"{'预算':>8s} {'gold进上下文题数':>16s} {'占比':>8s}")
    for b in BUDGETS:
        c = sum(1 for r in per_question if r["by_budget"][str(b)]["gold_in"])
        print(f"{b:8d} {c:16d} {c / n:8.3f}")

    need_vals = [r["min_budget_for_gold"] for r in per_question if r["min_budget_for_gold"]]
    if need_vals:
        need_vals.sort()
        print(f"\n救回 1 题所需最小预算分布: 中位 {need_vals[len(need_vals)//2]}, "
              f"最小 {need_vals[0]}, 最大 {need_vals[-1]}")
        for cap in (7000, 8000, 10000, 12000, 15000, 20000):
            print(f"  预算≤{cap}: 可救 {sum(1 for x in need_vals if x <= cap)} 题")

    print(f"\n=== 2. 截断机制 ===")
    over = sum(1 for r in per_question if r["skipped_but_later_kept"])
    later_slots = sum(len(r["skipped_but_later_kept"]) for r in per_question)
    print(f"存在「被跳过而后面的块仍被保留」的题: {over}/{n}（占 {over/n:.3f}）")
    print(f"这类被跳过的单元总数: {later_slots}")
    examples = [r for r in per_question if r["skipped_but_later_kept"]][:3]
    for r in examples:
        s = r["skipped_but_later_kept"][0]
        print(f"  例 {r['question_id']}: 跳过 {s['skipped']}({s['tokens']}tok) 却保留 {s['later_kept'][0]}")

    big = [u for r in per_question for u in r["ordered_all"]]
    print(f"\n候选单元 token: 中位 {sorted(u['tokens'] for u in big)[len(big)//2]}, 均值 {sum(u['tokens'] for u in big)/len(big):.0f}")

    print(f"\n=== 3. 同 doc 兄弟块挤占 ===")
    sib = sum(1 for r in per_question if r["sibling_in_ctx_any"])
    print(f"gold 不在上下文、但同 doc 兄弟块在上下文内的题: {sib}/{n}（占 {sib/n:.3f}）")

    print(f"\n=== 单元总数 ===")
    print(f"候选单元数中位: {sorted(r['units_total'] for r in per_question)[n//2]}, "
          f"最大 {max(r['units_total'] for r in per_question)}")
    print(f"全部候选总 token 中位: {sorted(r['tokens_all'] for r in per_question)[n//2]}")
    print(f"6000 预算下保留单元数中位: {sorted(len(r['kept@6000']) for r in per_question)[n//2]}")

    out = ROOT / "data" / "a_context_budget_probe.json"
    json.dump(
        {
            "schema": "arkrag.a-context-budget-probe/1",
            "generation": GID,
            "top_k": TOP_K,
            "graph_depth": GRAPH_DEPTH,
            "split_policy": "pre-split-diagnostic（分析用，非基线非验收）",
            "method": "重放同一批 hits/expansions，用同一贪心循环在不同预算下重算；非端到端实测",
            "selfcheck": "pass" if not selfcheck_fail else "fail",
            "budgets": BUDGETS,
            "questions": per_question,
        },
        open(out, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=1,
    )
    print(f"\n逐题结果写入 {out.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
