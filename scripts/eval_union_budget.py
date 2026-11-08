"""向量补进来的块，能不能真的变成答对？

## 为什么要单独测这一层

`eval_vector_union.py` 的结果是三层全 0 变化：

    dev 350    hit20    R2（进上下文）   答对
    base       273      195              196
    向量尾部并集 273      195              196       ← 触发 103 题、追加 1414 块

原因不是"向量没用"，而是**位置**：并集只能把块追加在第 21 名以后，而
`build_context` 的 6000 pseudo-token 只装 4–6 块（`estimate_tokens` 是词法
口径，约 1.68 token/汉字）。救回的 15 道题，gold 全在 21 名以外，
**在进入上下文之前就被预算砍掉了**。

所以本轮不是重复测并集，而是问一个不同的问题：

    如果预算真的够装下并集补进来的块，那 15 道题会不会变成答对？

## 四个臂（同一批题、同一次检索，只换上下文预算与是否有并集）

    base@6000      生产现状
    union@6000     向量尾部并集（上一轮已测，三层 0 变化）
    base@18000     只放大预算
    union@18000    放大预算 + 并集

`build_context` 的预算取自 `retriever.budget["max_context_tokens"]`，
`ask_once` 里不传 `token_budget`，所以这里直接改这个字段。

## 报什么

每题记 `hit20 / ctx_rank / correct`，四臂并排；再单独看**并集触发题**这个
子集上的四臂对比——因为并集只可能影响这 103 道。

分母一律写清：`<分子>/<分母> [口径名]`。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import embedding  # noqa: E402
from arkrag import pipeline as pipeline_mod  # noqa: E402
from arkrag import vector_store  # noqa: E402
from arkrag import vector_union as union_mod  # noqa: E402

TOP_K = 20


def rank_of(units, gold: set[str]) -> int | None:
    for i, u in enumerate(units, 1):
        cid = u.get("chunk_id") if isinstance(u, dict) else u
        if str(cid) in gold:
            return i
    return None


def answer_hits_gold(answer: dict, item: dict) -> bool:
    text = str(answer.get("answer") or "").strip()
    if not text:
        return False
    cands = [str(item.get("answer") or "")] + [
        str(x) for x in (item.get("acceptable_answers") or []) if x
    ]
    return any(c and c in text for c in cands)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generation", default="g-20261103T132152Z-06a741")
    ap.add_argument("--gold-set", default="data/goldset/gold-v2.json")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--budgets", nargs="+", type=int, default=[6000, 18000])
    ap.add_argument("--cut", type=float, default=union_mod.DEFAULT_RATIO_CUT)
    ap.add_argument("--vector-k", type=int, default=union_mod.DEFAULT_VECTOR_K)
    ap.add_argument("--json", default="data/eval_union_budget.json")
    args = ap.parse_args()

    loaded = pipeline_mod.load_generation(args.generation)
    digest = (loaded["index_meta"] or {}).get("index_content_digest") or ""
    slug = vector_store._slug(embedding.model_name())
    bin_path, meta_path = vector_store.vector_paths(slug, digest)
    if not Path(bin_path).exists():
        raise SystemExit(f"[2] 缺向量产物 {bin_path}")
    matrix, vmeta = vector_store.load_vectors(bin_path, meta_path)
    vec = vector_store.VectorIndex(matrix, vmeta)

    pipe = pipeline_mod.build_pipeline(loaded, top_k=TOP_K, graph_depth=1)
    union = union_mod.TriggeredVectorUnion(
        vec, embed_query=lambda t: embedding.embed_batch([t])[0],
        cut=args.cut, vector_k=args.vector_k,
    )

    gold = json.loads((ROOT / args.gold_set).read_text(encoding="utf-8"))
    items = [g for g in (gold.get("items") or [])
             if str(g.get("split") or "") == args.split and g.get("gold_chunk_ids")]
    print(f"split={args.split} n={len(items)}", flush=True)

    rows = []
    for i, item in enumerate(items, 1):
        q = str(item.get("question") or "")
        gset = {str(c) for c in item["gold_chunk_ids"]}
        timings: dict = {}
        res = pipeline_mod.ask_once(pipe, q, timings=timings)
        base_hits = res["search"]["hits"]
        aug = union.augment(base_hits, q)
        row = {
            "qid": str(item.get("question_id") or i),
            "category": str(item.get("category") or ""),
            "ratio": aug["ratio"],
            "triggered": aug["used"],
            "added": len(aug["added"]),
            "hits": {}, "ctx": {}, "ok": {},
        }
        for name, hits in (("base", base_hits), ("union", aug["hits"])):
            row["hits"][name] = rank_of(hits, gset)
            for budget in args.budgets:
                pipe["retriever"].budget["max_context_tokens"] = budget
                saved_top = pipe["top_k"]
                expanded = pipe["retriever"].expand(
                    [c["entity_id"] for c in res["query_analysis"].get("entity_candidates") or []],
                    depth=1, query_analysis=res["query_analysis"],
                    seed_chunk_ids=[h["chunk_id"] for h in hits],
                )
                ctx = pipe["retriever"].build_context(
                    hits, expanded["expansions"], query_analysis=res["query_analysis"]
                )
                ans = pipe["answerer"].answer(
                    q, ctx,
                    generation_id=loaded["generation_id"],
                    top_score=float((res["search"]["stats"] or {}).get("top_score") or 0.0),
                    query_analysis=res["query_analysis"],
                )
                key = f"{name}@{budget}"
                row["ctx"][key] = rank_of(ctx["units"], gset)
                row["ok"][key] = answer_hits_gold(ans, item)
                pipe["top_k"] = saved_top
        rows.append(row)
        if i % 25 == 0 or i == len(items):
            print(f"  {i}/{len(items)}", flush=True)

    # 复原预算，别把进程状态带走
    pipe["retriever"].budget["max_context_tokens"] = 6000

    n = len(rows)
    arms = [f"{name}@{b}" for b in args.budgets for name in ("base", "union")]
    summary = {}
    for arm in arms:
        hit_name = arm.split("@")[0]
        summary[arm] = {
            "hit20": sum(1 for r in rows if r["hits"][hit_name] and r["hits"][hit_name] <= 20),
            "r2": sum(1 for r in rows if r["ctx"][arm]),
            "correct": sum(1 for r in rows if r["ok"][arm]),
        }
    trig = [r for r in rows if r["triggered"]]
    summary_triggered = {}
    for arm in arms:
        hit_name = arm.split("@")[0]
        summary_triggered[arm] = {
            "n": len(trig),
            "hit20": sum(1 for r in trig if r["hits"][hit_name] and r["hits"][hit_name] <= 20),
            "r2": sum(1 for r in trig if r["ctx"][arm]),
            "correct": sum(1 for r in trig if r["ok"][arm]),
        }

    out = {
        "generation": args.generation, "gold_set": args.gold_set,
        "split": args.split, "n": n, "cut": args.cut,
        "triggered": len(trig), "added_total": sum(r["added"] for r in rows),
        "arms": arms, "summary": summary, "summary_triggered": summary_triggered,
        "rows": rows,
    }
    path = ROOT / args.json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    def line(label, d, denom):
        return (f"{label:16s} hit20 {d['hit20']:>3}/{denom}  "
                f"R2 {d['r2']:>3}/{denom}  答对 {d['correct']:>3}/{denom}")

    print(f"\n== 全部 {n} 题", flush=True)
    for arm in arms:
        print("  " + line(arm, summary[arm], n), flush=True)
    print(f"\n== 并集触发题 {len(trig)} 题", flush=True)
    for arm in arms:
        print("  " + line(arm, summary_triggered[arm], len(trig)), flush=True)
    print(f"\n写出 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
