"""两件事：①标题通路 +52 题里有多少是「题干自带答案」的泄露；②章节通路对 cross_chunk 值多少。

为什么要查泄露：台账原文写着标题通路「两档并列：干净 22 题 / 含问句泄露 39 题」——
gold 里标题类题有两个骨架，模板 A 是 `在剧情《T》的正文里……`，**T 就是答案**。这类题
只要把问句里的书名号内容抄出来就答对了，跟检索/定位无关。所以 +52 这个数必须拆成
「真定位」与「题干泄露」两档，否则就是拿泄露当成绩上报。

为什么要查章节通路：`diag_title_verify.py` 实测 cross_chunk 现在是 **0/32**，而
`CHAPTER_ANSWER_PATH` 正是为它写的（`CHAPTER_TRIGGERS = ("章节代号",)`），台账记
「dev 350 上 +13/+6/+6，但 6 题由拒答转作答且答案全错」，于是默认关着。gold-v1 上
它到底值多少，必须实测而不是照抄 dev 350 的结论。

口径：300 题（gold-v1 有 gold_chunk_ids 的前 300）、top_k=20、graph_depth=1、budget=6000。
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
    print("\n" + "=" * 94)
    print(*a)
    print("=" * 94)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generation", default="g-20261103T132152Z-06a741")
    ap.add_argument("--gold-set", default="data/goldset/gold-v1.json")
    ap.add_argument("--limit", type=int, default=320)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--graph-depth", type=int, default=1)
    ap.add_argument("--budget", type=int, default=6000)
    ap.add_argument("--json", default="data/diag_title_leak_chapter.json")
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
            "item": item, "qa": qa, "ctx": ctx, "gset": gset,
            "top_score": float(found["stats"].get("top_score") or 0.0),
            "stats": found["stats"],
        })
        if i % 50 == 0 or i == len(items):
            print(f"  预热 {i}/{len(items)}", flush=True)

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
            text = str(ans.get("answer") or "").strip()
            probes = ([gold_answer] + accepted) if gold_answer else accepted
            st = ans.get("stats") or {}
            qid = str(item.get("question_id") or "")
            out[qid] = {
                "category": str(item.get("category") or ""),
                "question": q,
                "gold_answer": gold_answer,
                "answer": text,
                "R5": bool(text) and any(a and a in text for a in probes),
                "title_applied": bool((st.get("title_path") or {}).get("applied")),
                "chapter_applied": bool((st.get("chapter_path") or {}).get("applied")),
                "abstained": bool(ans.get("abstained")),
                "abstain_reason": str(ans.get("abstain_reason") or ""),
                "n_citations": len(ans.get("citations") or []),
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

    head("1. 三态总量（端到端 R5 / 300）")
    for label, res in (("开关全关（现状）", base), ("TITLE 开", t_on), ("TITLE+CHAPTER 开", both)):
        k = sum(1 for v in res.values() if v["R5"])
        print(f"  {label:<22}{k:>4}/{n} = {100.0*k/n:5.1f}%")

    head("2. 标题通路 +52 的成色拆解：题干里到底有没有 gold 答案")
    trig = [q for q, v in t_on.items() if v["title_applied"]]
    leaked = [q for q in trig if t_on[q]["gold_answer"] and t_on[q]["gold_answer"] in t_on[q]["question"]]
    clean = [q for q in trig if q not in set(leaked)]
    print(f"  标题通路触发 {len(trig)}/{n} 题")
    print(f"  其中「问句原文含 gold 答案」= {len(leaked)}/{len(trig)}"
          f"（{100.0*len(leaked)/max(1,len(trig)):.1f}%）← 这档是题干泄露，不能算检索/定位的功劳")
    print(f"  干净档（问句不含答案）= {len(clean)}/{len(trig)}")
    lk = sum(1 for q in leaked if t_on[q]["R5"])
    ck = sum(1 for q in clean if t_on[q]["R5"])
    print(f"  触发题答对：泄露档 {lk}/{len(leaked)}   干净档 {ck}/{len(clean)}")
    print(f"  → 净增益拆账：泄露档贡献 {lk - sum(1 for q in leaked if base[q]['R5'])} 题，"
          f"干净档贡献 {ck - sum(1 for q in clean if base[q]['R5'])} 题")
    if leaked:
        print("  泄露档样例（问句里直接出现答案）：")
        for q in leaked[:5]:
            print(f"    {q}  gold={t_on[q]['gold_answer']!r}  Q={t_on[q]['question'][:64]!r}")

    head("3. 章节通路：对 cross_chunk 值多少")
    cx = [q for q, v in base.items() if v["category"] == "cross_chunk"]
    print(f"  cross_chunk {len(cx)} 题：全关 {sum(1 for q in cx if base[q]['R5'])}"
          f" / TITLE 开 {sum(1 for q in cx if t_on[q]['R5'])}"
          f" / 两开 {sum(1 for q in cx if both[q]['R5'])}")
    ch_trig = [q for q, v in both.items() if v["chapter_applied"]]
    print(f"  章节通路触发 {len(ch_trig)}/{n} 题，类别分布 "
          f"{dict(Counter(both[q]['category'] for q in ch_trig))}")
    if ch_trig:
        print(f"  触发题：base {sum(1 for q in ch_trig if base[q]['R5'])} → 两开 "
              f"{sum(1 for q in ch_trig if both[q]['R5'])}")
        print("  触发题明细（转好的、转坏的都要看）：")
        for q in ch_trig:
            b, a = base[q]["R5"], both[q]["R5"]
            flag = "转好" if a and not b else ("转坏" if b and not a else ("都对" if a else "都错"))
            print(f"    {q} {both[q]['category']:<14} {flag}  Q={both[q]['question'][:46]!r}"
                  f"  ans={both[q]['answer'][:34]!r}")

    head("4. 逐题回退核查（TITLE+CHAPTER vs 全关）")
    worse = [q for q in base if base[q]["R5"] and not both[q]["R5"]]
    better = [q for q in base if not base[q]["R5"] and both[q]["R5"]]
    print(f"  转好 {len(better)}   转坏 {len(worse)}   都对 "
          f"{sum(1 for q in base if base[q]['R5'] and both[q]['R5'])}   都错 "
          f"{sum(1 for q in base if not base[q]['R5'] and not both[q]['R5'])}")
    for q in worse[:10]:
        print(f"    转坏 {q} {base[q]['category']:<14} base={base[q]['answer'][:36]!r}"
              f"  now={both[q]['answer'][:36]!r}")

    head("5. 按类别（三态）")
    cats = sorted({v["category"] for v in base.values()})
    print(f"  {'类别':<18}{'n':>4}{'全关':>7}{'TITLE':>7}{'两开':>7}{'Δ总':>6}")
    bycat = {}
    for cat in cats:
        qs = [q for q in base if base[q]["category"] == cat]
        a = sum(1 for q in qs if base[q]["R5"])
        b = sum(1 for q in qs if t_on[q]["R5"])
        c2 = sum(1 for q in qs if both[q]["R5"])
        bycat[cat] = {"n": len(qs), "base": a, "title": b, "both": c2}
        print(f"  {cat:<18}{len(qs):>4}{a:>7}{b:>7}{c2:>7}{c2-a:>+6}")

    out = ROOT / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n": n,
        "base_r5": sum(1 for v in base.values() if v["R5"]),
        "title_r5": sum(1 for v in t_on.values() if v["R5"]),
        "both_r5": sum(1 for v in both.values() if v["R5"]),
        "title_triggered": len(trig),
        "title_leaked": len(leaked), "title_clean": len(clean),
        "leaked_hit": lk, "clean_hit": ck,
        "chapter_triggered": len(ch_trig),
        "flip_better": better, "flip_worse": worse,
        "by_category": bycat,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n落盘 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
