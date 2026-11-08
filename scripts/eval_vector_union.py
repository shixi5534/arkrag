"""向量通道接进生产链路后，三层指标的同批对比 + dev/holdout 交叉验证。

## 为什么必须分三层看，而不是只看 R@20

`R@20` 数的是「gold 落在检索输出的前 20 名里」。但 `build_context` 的
6000 pseudo-token 预算实际只装得下 4–6 个块（`estimate_tokens` 是词法口径，
约 1.68 token/汉字，见台账 2026-11-07 节），所以**第 11–20 名的块基本进不了
上下文**：实测分名次 context 转化率 rank1 100/100、rank2-3 44/44、
rank4-5 19/27、**rank6-10 3/34**、**rank11+ 1/32**。

于是「尾部并集」这种接法在 R@20 上必然是 0 变化（追加在尾部，名次只可能
不变），但在 R2（gold 进上下文）上也可能 0 变化（尾部本来就被预算砍掉）。
三层一起测，才能说清向量到底在哪一层起了作用、还是根本没起作用。

三层口径（每题独立真跑，不做重放）：

  R@20        gold 任一落在检索 hits 前 20 名
  R2          gold 任一落在 `build_context` 产出的 units 里
  answer      答案文本包含 gold 的 answer 或任一 acceptable_answers

## dev / holdout 分开报，不许合并

`DEFAULT_RATIO_CUT = 1.377` 是在 gold-v2 dev 350 上定的分位（第 40 百分位）。
holdout 238 只用于报告。合并报数等于用测试集调参——本项目已经因此栽过
（盲特征重排 train MRR +0.0161 / holdout −0.0091，判据不过）。

## 延迟单独报，且必须报「触发率」

向量通道只在触发时付 embed + 矩阵扫描的钱。不报触发率，延迟数字无法解读。

用法：
    py -3 scripts\\eval_vector_union.py --splits dev holdout
"""

from __future__ import annotations

import argparse
import json
import statistics
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


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


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
    accepted = [str(x) for x in (item.get("acceptable_answers") or []) if x]
    gold_answer = str(item.get("answer") or "")
    for cand in [gold_answer] + accepted:
        if cand and cand in text:
            return True
    return False


def run_group(loaded: dict, items: list[dict], *, use_union: bool,
              cut: float, vector_k: int) -> dict:
    pipe = pipeline_mod.build_pipeline(loaded, top_k=TOP_K, graph_depth=1)
    vec_index = None
    if use_union:
        digest = (loaded["index_meta"] or {}).get("index_content_digest") or ""
        slug = vector_store._slug(embedding.model_name())
        bin_path, meta_path = vector_store.vector_paths(slug, digest)
        if not Path(bin_path).exists():
            raise SystemExit(f"[2] 缺向量产物 {bin_path}")
        matrix, vmeta = vector_store.load_vectors(bin_path, meta_path)
        vec_index = vector_store.VectorIndex(matrix, vmeta)
        pipe["vector_union"] = union_mod.TriggeredVectorUnion(
            vec_index,
            embed_query=lambda t: embedding.embed_batch([t])[0],
            cut=cut, vector_k=vector_k,
        )

    rows = []
    lat: list[float] = []
    for i, item in enumerate(items, 1):
        q = str(item.get("question") or "")
        gold = {str(c) for c in (item.get("gold_chunk_ids") or [])}
        timings: dict = {}
        t0 = time.perf_counter()
        res = pipeline_mod.ask_once(pipe, q, timings=timings)
        lat.append((time.perf_counter() - t0) * 1000.0)
        hits = res["search"]["hits"]
        ctx = res["context"]["units"]
        vu = (res["search"]["stats"] or {}).get("vector_union") or {}
        rows.append({
            "qid": str(item.get("question_id") or i),
            "category": str(item.get("category") or ""),
            "hit_rank": rank_of(hits, gold),
            "ctx_rank": rank_of(ctx, gold),
            "correct": answer_hits_gold(res["answer"], item),
            "abstained": bool(res["answer"].get("abstained")),
            "union_used": bool(vu.get("used")),
            "union_added": int(vu.get("added") or 0),
            "candidates": int((res["search"]["stats"] or {}).get("candidates") or 0),
            "ctx_units": int((res["context"]["stats"] or {}).get("units") or 0),
        })
        if i % 50 == 0 or i == len(items):
            print(f"    {i}/{len(items)}", flush=True)

    n = len(rows)
    info = pipe["vector_union"].info() if use_union else {}
    return {
        "n": n,
        "hit20": sum(1 for r in rows if r["hit_rank"] and r["hit_rank"] <= 20),
        "r2": sum(1 for r in rows if r["ctx_rank"]),
        "correct": sum(1 for r in rows if r["correct"]),
        "triggered": sum(1 for r in rows if r["union_used"]),
        "added_total": sum(r["union_added"] for r in rows),
        "lat_p50": pct(lat, 0.50),
        "lat_p95": pct(lat, 0.95),
        "lat_mean": statistics.fmean(lat) if lat else 0.0,
        "union_info": info,
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generation", default="g-20261103T132152Z-06a741")
    ap.add_argument("--gold-set", default="data/goldset/gold-v2.json")
    ap.add_argument("--splits", nargs="+", default=["dev", "holdout"])
    ap.add_argument("--cut", type=float, default=union_mod.DEFAULT_RATIO_CUT)
    ap.add_argument("--vector-k", type=int, default=union_mod.DEFAULT_VECTOR_K)
    ap.add_argument("--json", default="data/eval_vector_union.json")
    args = ap.parse_args()

    loaded = pipeline_mod.load_generation(args.generation)
    gold = json.loads((ROOT / args.gold_set).read_text(encoding="utf-8"))
    all_items = [g for g in (gold.get("items") or []) if g.get("gold_chunk_ids")]

    out: dict = {"generation": args.generation, "gold_set": args.gold_set,
                 "cut": args.cut, "vector_k": args.vector_k, "splits": {}}
    for split in args.splits:
        items = [g for g in all_items if str(g.get("split") or "") == split]
        if not items:
            print(f"[!] split={split} 没有带 gold 的题，跳过")
            continue
        print(f"== split={split} n={len(items)}", flush=True)
        print("  base ...", flush=True)
        base = run_group(loaded, items, use_union=False, cut=args.cut,
                         vector_k=args.vector_k)
        print("  union ...", flush=True)
        uni = run_group(loaded, items, use_union=True, cut=args.cut,
                        vector_k=args.vector_k)

        flip_gain = [r["qid"] for b, r in zip(base["rows"], uni["rows"])
                     if not b["correct"] and r["correct"]]
        flip_loss = [r["qid"] for b, r in zip(base["rows"], uni["rows"])
                     if b["correct"] and not r["correct"]]
        out["splits"][split] = {
            "n": base["n"],
            "base": {k: v for k, v in base.items() if k != "rows"},
            "union": {k: v for k, v in uni.items() if k != "rows"},
            "correct_gain": len(flip_gain),
            "correct_loss": len(flip_loss),
            "gain_qids": flip_gain,
            "loss_qids": flip_loss,
            "rows": [{"qid": b["qid"], "category": b["category"],
                      "base_rank": b["hit_rank"], "union_rank": u["hit_rank"],
                      "base_ctx": b["ctx_rank"], "union_ctx": u["ctx_rank"],
                      "base_ok": b["correct"], "union_ok": u["correct"],
                      "union_used": u["union_used"], "added": u["union_added"]}
                     for b, u in zip(base["rows"], uni["rows"])],
        }
        print(f"  [{split}] base hit20 {base['hit20']}/{base['n']} "
              f"r2 {base['r2']}/{base['n']} correct {base['correct']}/{base['n']} "
              f"P50 {base['lat_p50']:.1f} P95 {base['lat_p95']:.1f}ms", flush=True)
        print(f"  [{split}] union hit20 {uni['hit20']}/{uni['n']} "
              f"r2 {uni['r2']}/{uni['n']} correct {uni['correct']}/{uni['n']} "
              f"P50 {uni['lat_p50']:.1f} P95 {uni['lat_p95']:.1f}ms "
              f"triggered {uni['triggered']}/{uni['n']} added {uni['added_total']}",
              flush=True)
        print(f"  [{split}] 答对翻转 +{len(flip_gain)}/-{len(flip_loss)}", flush=True)

    path = ROOT / args.json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写出 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
