"""既有缺陷量化：`evidence_too_short` 有没有把答案整条清空（→ 变成拒答）。

`diag_cite_failures.py` 实测：640 题里 13–14 题、3333 条引用里 16–17 条被
`verify_citations` 以 `evidence_too_short(2|3)` 剔除。剔除本身只是降置信度；
真正致命的是 `verify_citations` 里那条兜底——

    if not kept and answer.get("answer"):
        out["answer"] = ""; out["abstained"] = True
        out["abstain_reason"] = "no_valid_citation"

即「一条引用都不剩但还有答案正文 ⇒ 清空并改判拒答」。若某题只抽出 1 条 claim 且它的
证据只有 2–3 字，这题就会从「答对」变成「拒答」。本脚本数这个数。

同时验证一个候选修法：`_citation_span` 产出的 evidence 短于 `MIN_EVIDENCE_CHARS` 时，
把区间扩到断言所在整行（与 `CITATION_WHOLE_LINE` 同机制，但只在必要时触发）。
这修的是**同一个函数自己发出的东西被自己的校验器拒绝**这个自相矛盾，不是放宽判据：
`text[start:end] == evidence` 仍然逐条硬查。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import answer as answer_mod  # noqa: E402
from arkrag import pipeline as pipeline_mod  # noqa: E402
from arkrag import query as query_mod  # noqa: E402
from arkrag import vocab as vocab_mod  # noqa: E402
from arkrag.retrieve import Retriever  # noqa: E402


def head(*a):
    print("\n" + "=" * 96)
    print(*a)
    print("=" * 96)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generation", default="g-20261103T132152Z-06a741")
    ap.add_argument("--gold-set", default="data/goldset/gold-v2.json")
    ap.add_argument("--limit", type=int, default=640)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--graph-depth", type=int, default=1)
    ap.add_argument("--budget", type=int, default=6000)
    ap.add_argument("--json", default="data/diag_short_citation.json")
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
    items = list(gold.get("items") or [])[: args.limit]

    answer_mod.TITLE_ANSWER_PATH = True
    answer_mod.CHAPTER_ANSWER_PATH = True
    try:
        rows = []
        for i, item in enumerate(items, 1):
            q = str(item.get("question") or "")
            qa = query_mod.analyze_query(
                q, entities=entities, aliases=graph.get("alias_index"), vocab=vocab_mod
            )
            found = retriever.search(qa, top_k=args.top_k)
            hits = found["hits"]
            seed_ids = [str(c.get("entity_id") or "") for c in (qa.get("entity_candidates") or [])]
            seed_ids = [s for s in seed_ids if s]
            expanded = retriever.expand(
                seed_ids, depth=args.graph_depth, query_analysis=qa,
                seed_chunk_ids=[h["chunk_id"] for h in hits],
            )
            ctx = retriever.build_context(
                hits, expanded["expansions"], query_analysis=qa, token_budget=args.budget
            )
            ans = answerer.answer(
                q, ctx, generation_id=args.generation,
                top_score=float(found["stats"].get("top_score") or 0.0),
                query_analysis=qa, retrieval_stats=found["stats"],
            )
            checked = answer_mod.verify_citations(ans, chunks)
            na = len(ans.get("citations") or [])
            nk = len(checked.get("citations") or [])
            rows.append({
                "qid": str(item.get("question_id") or ""),
                "category": str(item.get("category") or ""),
                "should_abstain": bool(item.get("should_abstain")),
                "raw_answer": str(ans.get("answer") or ""),
                "raw_cites": na,
                "kept_cites": nk,
                "cleared": bool(ans.get("answer")) and not checked.get("answer"),
                "abstain_reason": str(checked.get("abstain_reason") or ""),
                "short_cites": sum(
                    1 for c in (ans.get("citations") or [])
                    if len(str(c.get("evidence") or "")) < answer_mod.MIN_EVIDENCE_CHARS
                ),
            })
            if i % 160 == 0:
                print(f"  {i}/{len(items)}", flush=True)
    finally:
        answer_mod.TITLE_ANSWER_PATH = False
        answer_mod.CHAPTER_ANSWER_PATH = False

    head("1. evidence_too_short 的实际后果")
    short = [r for r in rows if r["short_cites"] > 0]
    cleared = [r for r in rows if r["cleared"]]
    print(f"  含「过短引用」的题 {len(short)}/{len(rows)}；过短引用总条数 "
          f"{sum(r['short_cites'] for r in short)}")
    print(f"  因「引用全被剔」而被清空改判拒答的题：{len(cleared)}")
    for r in cleared[:10]:
        print(f"    {r['qid']} {r['category']:<14} raw={r['raw_cites']}条 "
              f"raw答案={r['raw_answer'][:34]!r}")
    print(f"  → 若为 0，则 `evidence_too_short` 只降置信度，不改答案/拒答判定")

    head("2. 这些题本来答对了吗（判断缺陷是否吃掉成绩）")
    for r in sorted(short, key=lambda x: -x["short_cites"])[:12]:
        print(f"    {r['qid']} {r['category']:<14} cites {r['raw_cites']}→{r['kept_cites']} "
              f"过短 {r['short_cites']} 清空={int(r['cleared'])}")

    out = ROOT / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n": len(rows),
        "n_with_short": len(short),
        "n_short_cites": sum(r["short_cites"] for r in short),
        "n_cleared": len(cleared),
        "cleared": cleared,
        "rows": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n落盘 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
