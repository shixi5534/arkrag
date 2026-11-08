"""A3 那 16 条引用定位失败是谁产生的：是既有缺陷，还是翻转开关引入的？

上一轮 `diag_abstain_risk.py` 的读数：
    全关        引用被剔 17
    TITLE 开    引用被剔 16
    TITLE+CHAPTER 开  引用被剔 16

**全关状态就已经有 17 条被剔** ⇒ 这不是翻转开关引入的，是既有缺陷。但我不能就这么
把它写成「不是我干的」——本项目记录在案的事故形态之一就是「判定顺序即口径」和
「静默降级」。必须逐条查：哪道题、哪条引用、失败原因是什么、这条引用是 claim 通路
发的还是两个新通路发的。

判据：
  * 若 16 条全部来自 claim 通路（`assertion_id` 非空） ⇒ 与新通路无关，属既有缺陷，
    单独记账、不许算进本次改动的成本，但**必须上报给用户且不许默认它不存在**。
  * 若有一条来自新通路（`granularity == "line"` 且 `claim` 以「标题：」/「章节：」开头）
    ⇒ 新通路有 bug，必须修完再谈翻转。
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
    ap.add_argument("--json", default="data/diag_cite_failures.json")
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

    for label, title, chapter in (
        ("全关", False, False),
        ("TITLE+CHAPTER 开", True, True),
    ):
        answer_mod.TITLE_ANSWER_PATH = title
        answer_mod.CHAPTER_ANSWER_PATH = chapter
        cases = []
        reason_counter = Counter()
        source_counter = Counter()
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
            ver = checked.get("verification") or {}
            for prob in ver.get("problems") or []:
                # 找出这条问题引用原先属于哪一种通路
                src = "claim"
                claim_text = ""
                for c in ans.get("citations") or []:
                    if (str(c.get("chunk_id")) == str(prob.get("chunk_id"))
                            and str(c.get("evidence")) == str(prob.get("evidence"))):
                        claim_text = str(c.get("claim") or "")
                        if str(c.get("granularity")) == "line":
                            if claim_text.startswith("标题："):
                                src = "title_path"
                            elif claim_text.startswith("章节："):
                                src = "chapter_path"
                        break
                cases.append({
                    "qid": str(item.get("question_id") or ""),
                    "category": str(item.get("category") or ""),
                    "chunk_id": prob.get("chunk_id"),
                    "reason": prob.get("reason"),
                    "source": src,
                    "claim": claim_text,
                    "evidence": str(prob.get("evidence") or "")[:70],
                })
                reason_counter[str(prob.get("reason"))] += 1
                source_counter[src] += 1
            if i % 160 == 0:
                print(f"  [{label}] {i}/{len(items)}", flush=True)

        head(f"[{label}] 引用定位失败 {len(cases)} 条")
        print(f"  按原因：{dict(reason_counter)}")
        print(f"  按通路：{dict(source_counter)}")
        for c in cases[:12]:
            print(f"    {c['qid']} {c['category']:<14} [{c['source']}] {c['reason']}")
            print(f"      chunk={c['chunk_id']}")
            print(f"      claim={c['claim'][:60]!r}")
            print(f"      evid ={c['evidence']!r}")

        out = ROOT / f"{args.json[:-5]}_{'both' if title else 'base'}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "label": label, "n_cases": len(cases),
            "by_reason": dict(reason_counter),
            "by_source": dict(source_counter),
            "cases": cases,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  落盘 {out}")

    answer_mod.TITLE_ANSWER_PATH = False
    answer_mod.CHAPTER_ANSWER_PATH = False
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
