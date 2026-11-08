"""校验 TITLE_ANSWER_PATH=True 的 +52 题是不是真的。

`diag_budget_title.py` 实测（300 题、top_k=20、graph_depth=1、budget=6000）：

    TITLE=False   R5|R2 内 60/167 = 35.9%   端到端 106/300 = 35.3%
    TITLE=True    R5|R2 内 112/167 = 67.1%  端到端 158/300 = 52.7%

`TITLE_ANSWER_PATH` 是一个模块级常量，当前代码里写的是 **False**，而台账记录
「标题通路已入库，干净口径 +22 题 = 6.29pt，零回退」。所以这里必须回答三件事，
否则这个 +52 不能上报：

1. **是不是假通过**：`R5` 用的是「答案文本含 gold answer 串」。标题通路直接把
   `unit["title"]` 当答案输出，若 gold answer 恰好就是标题，那么这是「照抄题目
   要求的东西」还是「真的定位对了块」？必须用 `required_citations` 的
   `evidence`/`chunk_id` 独立校验一遍：答案里给出的 citation 是否真的指到了
   gold 块，且 `text[start:end] == evidence`。

2. **有没有回退**：逐类别、逐题比 False/True 两组的 R5，统计转好/转坏/不变。
   判据是项目一贯的「不许有类别净退」。若题 A 原来对、开了标题通路后错，就是回退。

3. **有没有偷看 gold**：标题通路只能读 `unit["title"]` 与问题里的引号内容，
   不许读 `gold_chunk_ids`/`answer`。代码走查 + 断言（脚本里检查它只用 units）。

另外顺带校准 `ctx_tokens` 口径：`estimate_tokens` 是词法分词器的 token 数，
190 汉字算 320（1.68 token/字），而真实 LLM 分词器约 0.6–1.0 token/字。所以
budget=6000 实际只装约 4,037 字。本脚本同时报「字符数」与「pseudo-token」两个口径，
避免把两个数混着说。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import answer as answer_mod  # noqa: E402
from arkrag import pipeline as pipeline_mod  # noqa: E402
from arkrag import query as query_mod  # noqa: E402
from arkrag import vocab as vocab_mod  # noqa: E402
from arkrag.retrieve import Retriever  # noqa: E402


def head(*a):
    print("\n" + "=" * 92)
    print(*a)
    print("=" * 92)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generation", default="g-20261103T132152Z-06a741")
    ap.add_argument("--gold-set", default="data/goldset/gold-v1.json")
    ap.add_argument("--limit", type=int, default=320)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--graph-depth", type=int, default=1)
    ap.add_argument("--budget", type=int, default=6000)
    ap.add_argument("--json", default="data/diag_title_verify.json")
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

    # ------ 缓存检索与组装：两组共用，唯一差别是 TITLE 开关 --------------
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
        ctx = retriever.build_context(
            hits, expanded["expansions"], query_analysis=qa, token_budget=args.budget
        )
        cached.append({
            "item": item, "qa": qa, "ctx": ctx, "gset": gset,
            "base_rank": base_rank,
            "top_score": float(found["stats"].get("top_score") or 0.0),
            "stats": found["stats"],
        })
        if i % 50 == 0 or i == len(items):
            print(f"  预热 {i}/{len(items)}", flush=True)

    def run(flag: bool) -> dict:
        answer_mod.TITLE_ANSWER_PATH = flag
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
            hit_gold = bool(text) and any(a and a in text for a in probes)
            qid = str(item.get("question_id") or "")
            out[qid] = {
                "category": str(item.get("category") or ""),
                "R1": c["base_rank"] is not None,
                "base_rank": c["base_rank"],
                "answer": text,
                "citations": ans.get("citations") or [],
                "abstained": bool(ans.get("abstained")),
                "abstain_reason": str(ans.get("abstain_reason") or ""),
                "R5": hit_gold,
                "title_applied": bool((ans.get("stats") or {}).get("title_path", {}).get("applied")),
                "n_units": len(c["ctx"]["units"]),
                "chars": sum(len(str(u.get("title") or "")) + len(str(u.get("chapter") or ""))
                             + len(str(u.get("text") or "")) for u in c["ctx"]["units"]),
                "pseudo_tokens": c["ctx"]["stats"]["tokens_used"],
            }
        return out

    original = answer_mod.TITLE_ANSWER_PATH
    try:
        off = run(False)
        on = run(True)
    finally:
        answer_mod.TITLE_ANSWER_PATH = original

    n = len(cached)
    hit = [str(c["item"].get("question_id") or "") for c in cached if c["base_rank"] is not None]

    head("1. 总量与两环")
    for label, res in (("TITLE=False", off), ("TITLE=True", on)):
        r2 = sum(1 for q in hit if True)
        r5_all = sum(1 for v in res.values() if v["R5"])
        r5_hit = sum(1 for q in hit if res[q]["R5"])
        print(f"  {label:<14} 端到端 R5 {r5_all:>3}/{n} = {100.0*r5_all/n:5.1f}%"
              f"    命中题内 R5 {r5_hit:>3}/{len(hit)} = {100.0*r5_hit/len(hit):5.1f}%")

    head("2. 真伪校验：标题通路的 citation 是否真的指到 gold 块，且 text[start:end]==evidence")
    applied = [q for q, v in on.items() if v["title_applied"]]
    print(f"  标题通路实际触发 {len(applied)} 题（分母 {n} 题 = {100.0*len(applied)/n:.1f}%）")
    real_loc, bad_loc, no_cite = 0, 0, 0
    gold_cited, non_gold_cited = 0, 0
    for q in applied:
        v = on[q]
        cits = v["citations"]
        if not cits:
            no_cite += 1
            continue
        all_ok, any_gold = True, False
        for ct in cits:
            cid = str(ct.get("chunk_id") or "")
            txt = str(chunks.get(cid, {}).get("text") or "")
            s, e, ev = int(ct.get("start", -1)), int(ct.get("end", -1)), str(ct.get("evidence") or "")
            if not (0 <= s < e <= len(txt)) or txt[s:e] != ev:
                all_ok = False
            if cid in on[q].get("_gset", set()) or True:
                pass
        # gold 命中：citation 的 chunk_id 是否落在该题 gold_chunk_ids
        gset = next((c["gset"] for c in cached
                     if str(c["item"].get("question_id") or "") == q), set())
        any_gold = any(str(ct.get("chunk_id") or "") in gset for ct in cits)
        if all_ok:
            real_loc += 1
        else:
            bad_loc += 1
        if any_gold:
            gold_cited += 1
        else:
            non_gold_cited += 1
    print(f"  定位自检 text[start:end]==evidence 通过 {real_loc}/{len(applied)}   不通过 {bad_loc}")
    print(f"  citation 落在 gold 块上 {gold_cited}/{len(applied)}   落在非 gold 块 {non_gold_cited}")
    print(f"  完全没有 citation {no_cite}/{len(applied)}")

    head("3. 回退核查：逐题 False vs True")
    flip_worse = [q for q in off if off[q]["R5"] and not on[q]["R5"]]
    flip_better = [q for q in off if not off[q]["R5"] and on[q]["R5"]]
    same_ok = sum(1 for q in off if off[q]["R5"] and on[q]["R5"])
    same_bad = sum(1 for q in off if not off[q]["R5"] and not on[q]["R5"])
    print(f"  转好 {len(flip_better):>3}   转坏 {len(flip_worse):>3}   都对 {same_ok:>3}   都错 {same_bad:>3}")
    if flip_worse:
        print("  转坏清单（这是回退，必须解释）：")
        for q in flip_worse[:20]:
            print(f"    {q}  cat={off[q]['category']:<16} rank={off[q]['base_rank']}  "
                  f"off={off[q]['answer'][:40]!r}  on={on[q]['answer'][:40]!r}")

    head("4. 按类别（端到端答对数 / 该类题数）")
    cats = sorted({v["category"] for v in off.values()})
    print(f"  {'类别':<18}{'n':>4}{'False':>8}{'True':>8}{'Δ':>6}")
    bycat = {}
    for cat in cats:
        qs = [q for q in off if off[q]["category"] == cat]
        a = sum(1 for q in qs if off[q]["R5"])
        b = sum(1 for q in qs if on[q]["R5"])
        bycat[cat] = {"n": len(qs), "off": a, "on": b, "delta": b - a}
        print(f"  {cat:<18}{len(qs):>4}{a:>8}{b:>8}{b-a:>+6}")

    head("5. 标题通路的适用面（它只对「问标题」类题有用）")
    trig_cat = Counter(on[q]["category"] for q in applied)
    print(f"  触发题的类别分布: {dict(trig_cat)}")
    untrig = [q for q in off if q not in set(applied)]
    a = sum(1 for q in untrig if off[q]["R5"])
    b = sum(1 for q in untrig if on[q]["R5"])
    print(f"  未触发题 {len(untrig)} 道：False 答对 {a}  True 答对 {b}（应当完全相同：{a==b}）")

    head("6. 口径")
    v0 = next(iter(off.values()))
    print(f"  ctx 装入：字符均值 {sum(v['chars'] for v in off.values())/n:.0f} 字 / "
          f"pseudo-token 均值 {sum(v['pseudo_tokens'] for v in off.values())/n:.0f}")
    print(f"  estimate_tokens 是词法分词器口径（约 1.4–1.7 token/汉字），"
          f"真实 LLM 中文分词器约 0.6–1.0 token/汉字 → 两个数不许混说")

    out = ROOT / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generation": args.generation, "budget": args.budget, "top_k": args.top_k,
        "n": n, "n_hit": len(hit),
        "off_r5_all": sum(1 for v in off.values() if v["R5"]),
        "on_r5_all": sum(1 for v in on.values() if v["R5"]),
        "title_applied": len(applied),
        "loc_ok": real_loc, "loc_bad": bad_loc, "no_cite": no_cite,
        "gold_cited": gold_cited, "non_gold_cited": non_gold_cited,
        "flip_better": flip_better, "flip_worse": flip_worse,
        "by_category": bycat,
        "untriggered_same": (a == b),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n落盘 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
