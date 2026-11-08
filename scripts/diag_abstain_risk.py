"""翻转两个开关前必须补的最后一道验证：**拒答风险**。

为什么必须做：台账原文对 `CHAPTER_ANSWER_PATH` 的否决理由不是「没增益」，而是
「**6 道由拒答转作答的题答案全错，却各发 4–5 条定位通过的引用** ⇒ 不知道变成看起来
很确定的错，产品风险大于收益」。

而我此前在 gold-v1 上测的 300 题**全部 `should_abstain=False`**（实测 0 题 True），
所以那轮测试**结构上不可能看到这个风险**。拿一个看不到主要风险的样本去支持「翻转
默认值」，就是本项目记录在案的错误形态。

本脚本在 `gold-v2`（640 题，其中 52 题 `should_abstain=True`）上跑三态：
  全关 / TITLE 开 / TITLE+CHAPTER 开
并按 `should_abstain` 分账，逐题列出「应当拒答、却给了答案」的题。

判据（写死，不许事后改）：
  A1 应拒题正确拒答数：翻转后**不许下降**（任何一道从「正确拒答」变成「作答」都是回退）
  A2 应答题答对数：翻转后不许下降
  A3 引用硬失败：必须为 0（`text[start:end] == evidence` 逐条自检）
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
    ap.add_argument("--json", default="data/diag_abstain_risk.json")
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

    cached = []
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
        cached.append({
            "item": item, "qa": qa, "ctx": ctx,
            "top_score": float(found["stats"].get("top_score") or 0.0),
            "stats": found["stats"],
        })
        if i % 80 == 0 or i == len(items):
            print(f"  {i}/{len(items)}", flush=True)

    def run(title: bool, chapter: bool) -> dict:
        answer_mod.TITLE_ANSWER_PATH = title
        answer_mod.CHAPTER_ANSWER_PATH = chapter
        out = {}
        for c in cached:
            item = c["item"]
            q = str(item.get("question") or "")
            gold_answer = str(item.get("answer") or "")
            accepted = [str(a) for a in (item.get("acceptable_answers") or []) if a]
            ans = answerer.answer(
                q, c["ctx"], generation_id=args.generation,
                top_score=c["top_score"], query_analysis=c["qa"],
                retrieval_stats=c["stats"],
            )
            checked = answer_mod.verify_citations(ans, chunks)
            text = str(checked.get("answer") or "").strip()
            probes = ([gold_answer] + accepted) if gold_answer else accepted
            qid = str(item.get("question_id") or "")
            ver = checked.get("verification") or {}
            out[qid] = {
                "qid": qid,
                "category": str(item.get("category") or ""),
                "should_abstain": bool(item.get("should_abstain")),
                "question": q,
                "gold_answer": gold_answer,
                "answer": text,
                "abstained": bool(checked.get("abstained")),
                "abstain_reason": str(checked.get("abstain_reason") or ""),
                "R5": bool(text) and any(a and a in text for a in probes),
                "cite_passed": int(ver.get("passed") or 0),
                "cite_rejected": int(ver.get("rejected") or 0),
                "title_applied": bool(((ans.get("stats") or {}).get("title_path") or {}).get("applied")),
                "chapter_applied": bool(((ans.get("stats") or {}).get("chapter_path") or {}).get("applied")),
            }
        return out

    orig_t, orig_c = answer_mod.TITLE_ANSWER_PATH, answer_mod.CHAPTER_ANSWER_PATH
    try:
        base = run(False, False)
        t_on = run(True, False)
        both = run(True, True)
    finally:
        answer_mod.TITLE_ANSWER_PATH = orig_t
        answer_mod.CHAPTER_ANSWER_PATH = orig_c

    n = len(cached)
    abst = [q for q, v in base.items() if v["should_abstain"]]
    answ = [q for q, v in base.items() if not v["should_abstain"]]

    head(f"1. 样本构成（{args.gold_set}，{n} 题）")
    print(f"  should_abstain=True  {len(abst):>3} 题    should_abstain=False {len(answ):>3} 题")
    if not abst:
        print("  [!] 这个样本里没有应拒题 —— 无法验证拒答风险，不许据此翻转开关")

    head("2. A1 应拒题正确拒答数（不许下降）")
    for label, res in (("全关", base), ("TITLE 开", t_on), ("TITLE+CHAPTER 开", both)):
        ok = sum(1 for q in abst if res[q]["abstained"])
        print(f"  {label:<20} 正确拒答 {ok:>3}/{len(abst)}"
              f"   误作答 {len(abst)-ok:>3}")
    broke = [q for q in abst if base[q]["abstained"] and not both[q]["abstained"]]
    print(f"  → 从「正确拒答」变成「作答」的题：{len(broke)} 道")
    for q in broke[:20]:
        print(f"    {q} {both[q]['category']:<16} 答案={both[q]['answer'][:52]!r}"
              f"  引用 {both[q]['cite_passed']} 条")
    print(f"  A1 判定：{'PASS' if not broke else 'FAIL'}")

    head("3. A2 应答题答对数（不许下降）")
    for label, res in (("全关", base), ("TITLE 开", t_on), ("TITLE+CHAPTER 开", both)):
        ok = sum(1 for q in answ if res[q]["R5"])
        print(f"  {label:<20} 答对 {ok:>3}/{len(answ)} = {100.0*ok/max(1,len(answ)):.1f}%")
    worse = [q for q in answ if base[q]["R5"] and not both[q]["R5"]]
    print(f"  → 应答题回退 {len(worse)} 道")
    print(f"  A2 判定：{'PASS' if not worse else 'FAIL'}")

    head("4. A3 引用硬失败（必须 0）")
    for label, res in (("全关", base), ("TITLE 开", t_on), ("TITLE+CHAPTER 开", both)):
        rej = sum(v["cite_rejected"] for v in res.values())
        kept = sum(v["cite_passed"] for v in res.values())
        print(f"  {label:<20} 通过 {kept:>5}  被剔 {rej:>4}")
    total_rej = sum(v["cite_rejected"] for v in both.values())
    print(f"  A3 判定：{'PASS' if total_rej == 0 else 'FAIL（有 %d 条引用定位不过）' % total_rej}")

    head("5. 泄露拆分（标题通路 + 干净增益）")
    trig = [q for q, v in t_on.items() if v["title_applied"]]
    leaked = [q for q in trig if t_on[q]["gold_answer"] and t_on[q]["gold_answer"] in t_on[q]["question"]]
    clean = [q for q in trig if q not in set(leaked)]
    print(f"  标题触发 {len(trig)}  泄露 {len(leaked)}  干净 {len(clean)}")
    ch = [q for q, v in both.items() if v["chapter_applied"]]
    print(f"  章节触发 {len(ch)}  其中答对 {sum(1 for q in ch if both[q]['R5'])}")
    clean_gain = sum(1 for q in clean if t_on[q]["R5"] and not base[q]["R5"]) + \
                 sum(1 for q in ch if both[q]["R5"] and not t_on[q]["R5"])
    print(f"  干净净增益（不含题干泄露）= +{clean_gain} 题")

    verdict = {
        "A1_no_abstain_regression": not broke,
        "A2_no_answer_regression": not worse,
        "A3_zero_citation_failure": total_rej == 0,
    }
    verdict["all_pass"] = all(verdict.values())
    head("6. 判据总表")
    for k, v in verdict.items():
        print(f"  {k:<32} {'PASS' if v else 'FAIL'}")
    print(f"\n  → {'可以翻转默认值' if verdict['all_pass'] else '不许翻转默认值'}")

    out = ROOT / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "gold_set": args.gold_set, "n": n,
        "n_abstain": len(abst), "n_answer": len(answ),
        "base_r5": sum(1 for v in base.values() if v["R5"]),
        "title_r5": sum(1 for v in t_on.values() if v["R5"]),
        "both_r5": sum(1 for v in both.values() if v["R5"]),
        "abstain_ok": {
            "base": sum(1 for q in abst if base[q]["abstained"]),
            "title": sum(1 for q in abst if t_on[q]["abstained"]),
            "both": sum(1 for q in abst if both[q]["abstained"]),
        },
        "broke_abstain": broke,
        "regressed": worse,
        "citation_failures": total_rej,
        "clean_gain": clean_gain,
        "verdict": verdict,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"落盘 {out}")
    return 0 if verdict["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
