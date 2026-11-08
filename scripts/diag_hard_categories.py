"""剩下四类为什么全错：time_conflict 0/32、adversarial 0/16、negative_abstain 0/12，以及
章节通路 12 题的答案精度（含 gold 串 ≠ 答案正确，多余章节算不算错）。

前一轮实测（`diag_title_leak_chapter.json`）：
  single_chunk 0→52（其中 35 题题干自带答案，17 题干净）
  cross_chunk  0→12（章节通路，0 转坏）
  time_conflict 0/32、adversarial 0/16、negative_abstain 0/12 —— 这三类是纯粹的大洞。

本脚本要回答：
1. negative_abstain / adversarial 是不是**本来就该拒答**？如果 `should_abstain=True` 而系统
   拒答了，那不是错，是我的 R5 判据把它们错算了。必须按 `should_abstain` 分账。
2. time_conflict 32 题到底怎么错的：是拒答了、还是答成了另一时段的值？给出逐题对照。
3. 章节通路 12 题：答案列出的章节代号集合，与 gold 的集合比，精确率/召回各是多少。
   「答案含 gold 串」只能证明召回，不能证明没多列错的。
"""

from __future__ import annotations

import argparse
import json
import re
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
    print("\n" + "=" * 94)
    print(*a)
    print("=" * 94)


def chapters_in(text: str) -> set[str]:
    return set(re.findall(r"[a-z][a-z0-9_]{3,}", text or ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generation", default="g-20261103T132152Z-06a741")
    ap.add_argument("--gold-set", default="data/goldset/gold-v1.json")
    ap.add_argument("--limit", type=int, default=320)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--graph-depth", type=int, default=1)
    ap.add_argument("--budget", type=int, default=6000)
    ap.add_argument("--json", default="data/diag_hard_categories.json")
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

    orig_t, orig_c = answer_mod.TITLE_ANSWER_PATH, answer_mod.CHAPTER_ANSWER_PATH
    rows = []
    try:
        answer_mod.TITLE_ANSWER_PATH = True
        answer_mod.CHAPTER_ANSWER_PATH = True
        for i, item in enumerate(items, 1):
            q = str(item.get("question") or "")
            gset = {str(c) for c in item["gold_chunk_ids"]}
            gold_answer = str(item.get("answer") or "")
            accepted = [str(a) for a in (item.get("acceptable_answers") or []) if a]
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
            text = str(ans.get("answer") or "").strip()
            probes = ([gold_answer] + accepted) if gold_answer else accepted
            rows.append({
                "qid": str(item.get("question_id") or i),
                "category": str(item.get("category") or ""),
                "should_abstain": bool(item.get("should_abstain")),
                "question": q,
                "gold_answer": gold_answer,
                "acceptable": accepted,
                "answer": text,
                "R5": bool(text) and any(a and a in text for a in probes),
                "abstained": bool(ans.get("abstained")),
                "abstain_reason": str(ans.get("abstain_reason") or ""),
                "wrong_answer_in_q": bool(gold_answer) and gold_answer in q,
            })
            if i % 50 == 0 or i == len(items):
                print(f"  {i}/{len(items)}", flush=True)
    finally:
        answer_mod.TITLE_ANSWER_PATH = orig_t
        answer_mod.CHAPTER_ANSWER_PATH = orig_c

    n = len(rows)

    head("1. 按 should_abstain 分账（我原来的 R5 判据可能把该拒答的题算成错了）")
    abst = [r for r in rows if r["should_abstain"]]
    answ = [r for r in rows if not r["should_abstain"]]
    print(f"  should_abstain=True   {len(abst):>3} 题：其中真拒答 {sum(1 for r in abst if r['abstained']):>3}"
          f"、误作答 {sum(1 for r in abst if not r['abstained']):>3}")
    print(f"  should_abstain=False  {len(answ):>3} 题：答对 {sum(1 for r in answ if r['R5']):>3}"
          f"、答错 {sum(1 for r in answ if not r['R5']):>3}"
          f"、其中拒答 {sum(1 for r in answ if r['abstained']):>3}")
    ok_all = sum(1 for r in rows if (r["abstained"] if r["should_abstain"] else r["R5"]))
    print(f"  → 按「该拒就拒、该答就答」正确口径：{ok_all}/{n} = {100.0*ok_all/n:.1f}%"
          f"（原 R5 纯答题口径 {sum(1 for r in rows if r['R5'])}/{n}）")

    head("2. negative_abstain / adversarial 明细（前 12 题）")
    for cat in ("negative_abstain", "adversarial"):
        grp = [r for r in rows if r["category"] == cat]
        print(f"\n  --- {cat}：{len(grp)} 题，should_abstain=True 的 "
              f"{sum(1 for r in grp if r['should_abstain'])} 题 ---")
        for r in grp[:12]:
            print(f"    {r['qid']} abstain_q={r['should_abstain']} "
                  f"拒答={r['abstained']}({r['abstain_reason'][:26]}) "
                  f"R5={r['R5']}  Q={r['question'][:44]!r}")

    head("3. time_conflict 逐题（32 题全看）")
    tc = [r for r in rows if r["category"] == "time_conflict"]
    print(f"  答对 {sum(1 for r in tc if r['R5'])}/{len(tc)}；拒答 {sum(1 for r in tc if r['abstained'])}")
    for r in tc[:16]:
        print(f"    {r['qid']} R5={int(r['R5'])} 拒={int(r['abstained'])} "
              f"gold={r['gold_answer'][:26]!r} ans={r['answer'][:44]!r}")

    head("4. 章节通路 12 题的精度（答案含 gold ≠ 只含 gold）")
    cx = [r for r in rows if r["category"] == "cross_chunk" and r["R5"]]
    tot_prec_num = tot_prec_den = tot_rec_num = tot_rec_den = 0
    for r in cx:
        got = chapters_in(r["answer"])
        want = set()
        for a in ([r["gold_answer"]] + r["acceptable"]):
            want |= chapters_in(a)
        if not want:
            continue
        inter = got & want
        tot_prec_num += len(inter); tot_prec_den += len(got)
        tot_rec_num += len(inter); tot_rec_den += len(want)
        flag = "" if got == want else ("多余" if got > want else "缺")
        print(f"    {r['qid']} {flag:<4} got={sorted(got)} want={sorted(want)}")
    if tot_prec_den:
        print(f"\n  章节代号级 精确率 {tot_prec_num}/{tot_prec_den} = "
              f"{100.0*tot_prec_num/tot_prec_den:.1f}%   "
              f"召回 {tot_rec_num}/{tot_rec_den} = {100.0*tot_rec_num/tot_rec_den:.1f}%")

    out = ROOT / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n": n,
        "n_abstain_q": len(abst), "n_answer_q": len(answ),
        "ok_by_intent": ok_all,
        "r5_pure": sum(1 for r in rows if r["R5"]),
        "chapter_precision": [tot_prec_num, tot_prec_den],
        "chapter_recall": [tot_rec_num, tot_rec_den],
        "rows": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n落盘 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
