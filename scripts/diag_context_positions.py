"""gold 块在**最终上下文**里的位置：它是不是就是 hits 里的名次？

要回答的唯一问题
----------------
237（本脚本实测另计）道「gold 进了 hits 前 20」的题，gold 块最后落在送给 answerer 的
`context["units"]` 的第几位？这个位置是否等于它在 `search()["hits"]` 里的名次？
如果两者基本一致且 in_context 接近满值，那"证据前置"类改动就是打在空气上；
如果 gold 被图扩展挤到后面、或被 6000 token 预算整块砍掉，那才是真问题。

`build_context` 的实测装配规则（本脚本只观测、不改动）
----------------------------------------------------
1. 候选单元 = hits 的 chunk ∪ expansions 的 chunk，按 chunk_id 去重（`sources` 合并）。
2. 排序键 `(0 if direct else 1, -score, chunk_id)`，其中
   `direct = sources 里有 bm25 或 entity`。hits 天然就是 direct 且 score 递减，
   所以排序后仍按原 hits 次序在前；graph-only 单元（纯扩展块）一律排在其后。
   这条**不是从代码推的，是本脚本量出来的**：`ctx_pos > hit_rank` 的题数、
   以及「排在 gold 前面却不属于 hits」的单元数，都直接记在 summary 里
   （实测均为 0，见 b_ctx_pos_vs_hit_rank / d_noise_before_gold）。
3. 预算扫描：按上面顺序累加，`used + cost > 6000` 就**整单元丢弃**并记入
   `truncated_chunks`（不切开 evidence span），继续看下一个更小的单元。
   ⇒ gold 没进 context 若命中这条，原因就是预算，不是扩展。
4. 每个单元 cost = `estimate_tokens(title + " " + chapter + " " + text)`，
   与建库同一套分词器，不是真实 LLM 分词。

复刻口径
--------
调用顺序与参数照抄 `src/arkrag/pipeline.py::ask_once`：
`query.analyze_query(...)` → `Retriever.search(qa, top_k=20)` →
`Retriever.expand(seed_ids, depth=1, query_analysis=qa, seed_chunk_ids=[...])` →
`Retriever.build_context(hits, expansions, query_analysis=qa)`。
生产值 `top_k=20`、`graph_depth=1`、`max_context_tokens=6000`。

用法
----
    py -3 scripts/diag_context_positions.py
    py -3 scripts/diag_context_positions.py --limit 100      # 冒烟/限时分母
    py -3 scripts/diag_context_positions.py --since 0 --out data/diag_context_positions.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from arkrag import pipeline as pipeline_mod  # noqa: E402
from arkrag import query as query_mod  # noqa: E402
from arkrag import vocab  # noqa: E402
from arkrag.retrieve import SOURCE_BM25, SOURCE_ENTITY, SOURCE_GRAPH, SOURCE_SPEAKER  # noqa: E402

GENERATION = "g-20261103T132152Z-06a741"
GOLD_SET = ROOT / "data" / "goldset" / "gold-v1.json"
DEFAULT_OUT = ROOT / "data" / "diag_context_positions.json"

# 生产口径，不从别处读、写死在这里并回显到产物里
TOP_K = 20
GRAPH_DEPTH = 1
TOKEN_BUDGET = 6000


def _origin() -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": platform.python_version(),
        "os": platform.platform(),
        "command": "py -3 scripts/diag_context_positions.py",
    }


def read_gold() -> list[dict]:
    doc = json.loads(GOLD_SET.read_text(encoding="utf-8"))
    items = doc["items"] if isinstance(doc, dict) else doc
    return [it for it in items if it.get("gold_chunk_ids")]


def bucket(rank: int) -> str:
    if rank <= 1:
        return "1"
    if rank <= 3:
        return "2-3"
    if rank <= 5:
        return "4-5"
    if rank <= 10:
        return "6-10"
    return "11+"


class Counter2:
    """小工具：分子/分母一起攒，避免后面手算。"""

    def __init__(self) -> None:
        self.n = 0
        self.k = 0

    def add(self, ok: bool) -> None:
        self.n += 1
        self.k += 1 if ok else 0

    def as_dict(self) -> dict:
        return {"num": self.k, "den": self.n, "rate": round(self.k / self.n, 4) if self.n else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 道有 gold 的题（0 = 全部 300）")
    ap.add_argument("--since", type=int, default=0, help="从第几道开始（断点续跑用）")
    args = ap.parse_args(argv)

    items = read_gold()
    total_with_gold = len(items)
    if args.since:
        items = items[args.since:]
    if args.limit:
        items = items[: args.limit]
    n_planned = len(items)

    t0 = time.time()
    loaded = pipeline_mod.load_generation(GENERATION)
    pipe = pipeline_mod.build_pipeline(loaded, top_k=TOP_K, graph_depth=GRAPH_DEPTH)
    pipe["generation_id"] = GENERATION
    pipe["entities"] = loaded["graph"].get("entities") or []
    retriever = pipe["retriever"]
    print(f"[load] {time.time() - t0:.0f}s chunks={len(loaded['chunks'])} "
          f"assertions={len(loaded['assertions'])}", flush=True)

    rows: list[dict] = []
    t_eval = time.time()
    for i, it in enumerate(items, 1):
        qid = str(it.get("question_id") or "")
        question = str(it.get("question") or "")
        gold = [str(c) for c in it.get("gold_chunk_ids") or []]
        gold_set = set(gold)

        # ---- 完全照抄 pipeline.ask_once 的四步 ----
        qa = query_mod.analyze_query(
            question,
            entities=pipe["entities"],
            aliases=retriever.graph.get("alias_index"),
            vocab=vocab,
            context=getattr(retriever, "query_context", None),
        )
        found = retriever.search(qa, top_k=TOP_K)
        seed_ids = [c["entity_id"] for c in qa.get("entity_candidates") or []]
        expanded = retriever.expand(
            seed_ids,
            depth=GRAPH_DEPTH,
            query_analysis=qa,
            seed_chunk_ids=[h["chunk_id"] for h in found["hits"]],
        )
        context = retriever.build_context(
            found["hits"], expanded["expansions"], query_analysis=qa
        )

        # 反事实对照：把 expansions 整个拿掉，其余完全不动，再装一次 context。
        # 这是判断「gold 是不是被图扩展挤掉/砍掉」的唯一直接实验——不是读代码推断，
        # 是让同一条装配链路在没有扩展候选时再跑一遍，比较 gold 的存活率。
        ctx_no_exp = retriever.build_context(
            found["hits"], [], query_analysis=qa
        )
        no_exp_units = ctx_no_exp["units"]
        no_exp_ids = [str(u.get("chunk_id") or "") for u in no_exp_units]
        no_exp_pos = {cid: j + 1 for j, cid in enumerate(no_exp_ids) if cid}
        no_exp_stats = ctx_no_exp.get("stats") or {}
        gold_no_exp_pos = min((no_exp_pos[g] for g in gold if g in no_exp_pos), default=None)

        # 预算归因实验：只把 token 预算放大，其他一律不动，看被砍的 gold 会不会回来。
        # 「没进 context 是因为 6000 token 预算」这句话要么被这个数字证实，要么作废。
        budget_arms = {}
        for big in (8000, 12000):
            c = retriever.build_context(
                found["hits"], expanded["expansions"], token_budget=big,
                query_analysis=qa,
            )
            ids = [str(u.get("chunk_id") or "") for u in c["units"]]
            pos = {cid: j + 1 for j, cid in enumerate(ids) if cid}
            gpos = min((pos[g] for g in gold if g in pos), default=None)
            budget_arms[str(big)] = {
                "in_context": gpos is not None,
                "ctx_pos": gpos,
                "n_units": int((c.get("stats") or {}).get("units") or 0),
                "tokens": int((c.get("stats") or {}).get("tokens_used") or 0),
            }

        hits = found["hits"]
        hit_ids = [str(h.get("chunk_id") or "") for h in hits]
        hit_rank = {cid: j + 1 for j, cid in enumerate(hit_ids) if cid}
        units = context["units"]
        stats = context.get("stats") or {}
        ctx_ids = [str(u.get("chunk_id") or "") for u in units]
        ctx_pos = {cid: j + 1 for j, cid in enumerate(ctx_ids) if cid}
        truncated = [str(x) for x in (stats.get("truncated_chunks") or [])]
        truncated_set = set(truncated)
        skipped = {str(x.get("chunk_id") or ""): str(x.get("reason") or "")
                   for x in (stats.get("skipped_units") or [])}

        gold_in_hits = [g for g in gold if g in hit_rank]
        gold_first_rank = min((hit_rank[g] for g in gold_in_hits), default=None)
        gold_in_ctx = [g for g in gold if g in ctx_pos]
        gold_ctx_first = min((ctx_pos[g] for g in gold_in_ctx), default=None)

        # gold 没进 context 的原因：预算砍掉 / 块未载入 / 压根没进候选
        gold_not_ctx = [g for g in gold if g not in ctx_pos]
        not_ctx_reasons = {
            g: ("truncated_by_budget" if g in truncated_set
                else f"skipped:{skipped[g]}" if g in skipped
                else "not_a_candidate")
            for g in gold_not_ctx
        }

        # 前面的噪声单元：来源是 hits 还是纯 expansions
        noise_before = []
        if gold_ctx_first is not None:
            for u in units[: gold_ctx_first - 1]:
                cid = str(u.get("chunk_id") or "")
                chans = [s.get("channel") for s in (u.get("sources") or [])]
                noise_before.append(
                    {
                        "chunk_id": cid,
                        "tokens": int(u.get("tokens") or 0),
                        "channels": chans,
                        "from_hits": cid in hit_rank,
                        "hit_rank": hit_rank.get(cid),
                        "graph_path_len": len(u.get("graph_path") or []),
                    }
                )

        # 预算扫描是「跳过继续」，不是「截断即停」：某个单元装不下会被 continue，
        # 后面更小的单元照样能进来。所以 gold 被砍掉时，可能有**名次比它更差**的
        # 块反而进了 context。这里把这件事记成数，不许只在注释里断言。
        kept_hit_ranks = [hit_rank[c] for c in ctx_ids if c in hit_rank]
        worse_than_gold_kept = (
            [r for r in kept_hit_ranks if r > gold_first_rank]
            if gold_first_rank is not None else []
        )
        # 「名次更差却进了 context」只在 gold 自己被砍时才是净损失，单独计数
        overtaken = (
            len(worse_than_gold_kept) if (gold_first_rank is not None
                                         and gold_ctx_first is None) else 0
        )

        gold_unit = None
        if gold_ctx_first is not None:
            gid = ctx_ids[gold_ctx_first - 1]
            u = units[gold_ctx_first - 1]
            gold_unit = {
                "chunk_id": gid,
                "tokens": int(u.get("tokens") or 0),
                "channels": sorted({str(s.get("channel")) for s in (u.get("sources") or [])}),
                "graph_path_len": len(u.get("graph_path") or []),
            }

        rows.append(
            {
                "question_id": qid,
                "category": str(it.get("category") or ""),
                "question": question,
                "n_gold": len(gold),
                "gold_chunk_ids": gold,
                "hit_rank_per_gold": {g: hit_rank.get(g) for g in gold},
                "hit_rank": gold_first_rank,          # gold 在 hits 里的最好名次
                "in_hits": gold_first_rank is not None,
                "n_hits": len(hits),
                "in_context": gold_ctx_first is not None,
                "ctx_pos": gold_ctx_first,              # gold 在 units 里的最好位置（1 起）
                "ctx_pos_per_gold": {g: ctx_pos.get(g) for g in gold},
                "n_units": int(stats.get("units") or 0),
                "n_candidates": int(stats.get("candidates") or 0),
                "ctx_tokens": int(stats.get("tokens_used") or 0),
                "token_budget": int(stats.get("token_budget") or 0),
                "n_truncated": len(truncated),
                "any_truncated": bool(truncated),
                "gold_not_in_context_reasons": not_ctx_reasons,
                "n_noise_before": len(noise_before),
                "noise_before": noise_before,
                "kept_hit_ranks": kept_hit_ranks,
                "overtaken_by_worse_hits": overtaken,
                "gold_unit": gold_unit,
                "n_direct_units": int(stats.get("direct_units") or 0),
                "n_graph_only_units": int(stats.get("graph_only_units") or 0),
                "n_hit_units_truncated": sum(1 for c in truncated if c in hit_rank),
                "n_graph_units_truncated": sum(1 for c in truncated if c not in hit_rank),
                "n_expansions": int((expanded.get("stats") or {}).get("expansions") or 0),
                "no_exp_in_context": gold_no_exp_pos is not None,
                "no_exp_ctx_pos": gold_no_exp_pos,
                "no_exp_n_units": int(no_exp_stats.get("units") or 0),
                "no_exp_ctx_tokens": int(no_exp_stats.get("tokens_used") or 0),
                "budget_arms": budget_arms,
            }
        )
        if i % 25 == 0 or i == n_planned:
            print(f"  [{i}/{n_planned}] {time.time() - t_eval:.0f}s", flush=True)

    # ------------------------------------------------------------------ 汇总
    hit_rows = [r for r in rows if r["in_hits"]]
    n_hit = len(hit_rows)
    in_ctx = [r for r in hit_rows if r["in_context"]]
    not_ctx = [r for r in hit_rows if not r["in_context"]]

    # (a) in_context 率与没进 context 的原因
    why = Counter()
    for r in not_ctx:
        reasons = set(r["gold_not_in_context_reasons"].values())
        if "truncated_by_budget" in reasons:
            why["budget_truncated"] += 1
        elif any(x.startswith("skipped") for x in reasons):
            why["chunk_not_loaded"] += 1
        else:
            why["not_a_candidate"] += 1
    not_ctx_rank_dist = Counter(bucket(r["hit_rank"]) for r in not_ctx)

    # (b) ctx_pos vs hit_rank
    eq = sum(1 for r in in_ctx if r["ctx_pos"] == r["hit_rank"])
    gt = sum(1 for r in in_ctx if r["ctx_pos"] > r["hit_rank"])
    lt = sum(1 for r in in_ctx if r["ctx_pos"] < r["hit_rank"])
    delta_dist = Counter(r["ctx_pos"] - r["hit_rank"] for r in in_ctx)

    # (c) ctx_pos 分布 / hit_rank 分布
    pos_buckets = ["1", "2-3", "4-5", "6-10", "11+"]
    ctx_dist = Counter(bucket(r["ctx_pos"]) for r in in_ctx)
    hit_dist = Counter(bucket(r["hit_rank"]) for r in hit_rows)
    after3 = sum(1 for r in in_ctx if r["ctx_pos"] > 3)
    after5 = sum(1 for r in in_ctx if r["ctx_pos"] > 5)
    hit_after3 = sum(1 for r in hit_rows if r["hit_rank"] > 3)
    hit_after5 = sum(1 for r in hit_rows if r["hit_rank"] > 5)

    # 第 5 名转化率（报 R@5 类指标必须同时给）
    le5 = [r for r in hit_rows if r["hit_rank"] <= 5]
    le5_in_ctx = [r for r in le5 if r["in_context"]]
    le3 = [r for r in hit_rows if r["hit_rank"] <= 3]
    le3_in_ctx = [r for r in le3 if r["in_context"]]
    le10 = [r for r in hit_rows if r["hit_rank"] <= 10]
    le10_in_ctx = [r for r in le10 if r["in_context"]]

    # 名次分桶 -> 进 context 转化率（名次段内 in_context 率）
    conv = {}
    for b in pos_buckets:
        cell = [r for r in hit_rows if bucket(r["hit_rank"]) == b]
        conv[b] = {
            "n": len(cell),
            "in_context": sum(1 for r in cell if r["in_context"]),
            "rate": round(sum(1 for r in cell if r["in_context"]) / len(cell), 4) if cell else None,
        }

    # (d) 噪声来源
    noise_from_hits = sum(1 for r in in_ctx for nb in r["noise_before"] if nb["from_hits"])
    noise_from_graph = sum(1 for r in in_ctx for nb in r["noise_before"] if not nb["from_hits"])
    noise_total = sum(r["n_noise_before"] for r in in_ctx)

    trunc_summary = {
        "questions_any_truncated": sum(1 for r in hit_rows if r["any_truncated"]),
        "hit_units_truncated_total": sum(r["n_hit_units_truncated"] for r in hit_rows),
        "graph_units_truncated_total": sum(r["n_graph_units_truncated"] for r in hit_rows),
        "mean_n_units": round(sum(r["n_units"] for r in hit_rows) / n_hit, 2) if n_hit else None,
        "mean_ctx_tokens": round(sum(r["ctx_tokens"] for r in hit_rows) / n_hit, 2) if n_hit else None,
    }

    # 预算扫描是「跳过继续」：gold 被砍后，名次更差的块仍可能进来。这是"名次没变差
    # 却丢了 gold"的唯一机制，必须单独计数，否则会把预算问题的锅算到扩展头上。
    overtaken_q = [r for r in not_ctx if r["overtaken_by_worse_hits"] > 0]
    overtaken_summary = {
        "gold_cut_but_worse_rank_kept_题数": len(overtaken_q),
        "占比_分母_未进context": round(len(overtaken_q) / len(not_ctx), 4) if not_ctx else None,
        "被更差名次块顶掉的题": [
            {"question_id": r["question_id"], "hit_rank": r["hit_rank"],
             "kept_hit_ranks": r["kept_hit_ranks"]}
            for r in overtaken_q
        ],
        "说明": "build_context 超预算时 continue 而非 break；装不下的单元跳过，"
                "后面更小的单元照样入选，所以会出现「gold 被砍、名次比它差的块反而进了 context」。",
    }

    # 反事实：拿掉全部 expansions 再装一次 context，看 gold 存活与位置怎么变。
    # 只有这个对照能区分「gold 之死因是图扩展」还是「因为 6000 token 硬上限」。
    cf = {
        "with_expansions_in_context": sum(1 for r in hit_rows if r["in_context"]),
        "without_expansions_in_context": sum(1 for r in hit_rows if r["no_exp_in_context"]),
        "lost_only_because_of_expansions": [
            r["question_id"] for r in hit_rows
            if r["in_context"] and not r["no_exp_in_context"]
        ],
        "gained_only_because_of_expansions": [
            r["question_id"] for r in hit_rows
            if (not r["in_context"]) and r["no_exp_in_context"]
        ],
        "mean_n_units_with": round(sum(r["n_units"] for r in hit_rows) / n_hit, 2) if n_hit else None,
        "mean_n_units_without": round(sum(r["no_exp_n_units"] for r in hit_rows) / n_hit, 2) if n_hit else None,
        "mean_tokens_with": round(sum(r["ctx_tokens"] for r in hit_rows) / n_hit, 2) if n_hit else None,
        "mean_tokens_without": round(sum(r["no_exp_ctx_tokens"] for r in hit_rows) / n_hit, 2) if n_hit else None,
    }
    # 位置对照：只在「两次都进了 context」的题上比 ctx_pos
    both = [r for r in hit_rows if r["in_context"] and r["no_exp_in_context"]]
    cf["both_in_context"] = len(both)
    cf["pos_improved_without_exp"] = sum(1 for r in both if r["no_exp_ctx_pos"] < r["ctx_pos"])
    cf["pos_worse_without_exp"] = sum(1 for r in both if r["no_exp_ctx_pos"] > r["ctx_pos"])
    cf["pos_same_without_exp"] = sum(1 for r in both if r["no_exp_ctx_pos"] == r["ctx_pos"])

    # 预算归因：只放大 token 预算、别的一律不动，看被砍的 gold 回不回来。
    # 「70 题是 6000 token 砍的」这句话由这些数字背书或作废。
    budget_attr = {}
    for big in ("8000", "12000"):
        recovered = [r for r in not_ctx if r["budget_arms"][big]["in_context"]]
        budget_attr[big] = {
            "in_context_total": sum(1 for r in hit_rows if r["budget_arms"][big]["in_context"]),
            "recovered_from_the_70": len(recovered),
            "recovered_rate_of_70": round(len(recovered) / len(not_ctx), 4) if not_ctx else None,
            "still_missing": len(not_ctx) - len(recovered),
            "still_missing_ids": [r["question_id"] for r in not_ctx
                                  if not r["budget_arms"][big]["in_context"]],
            "mean_units": round(sum(r["budget_arms"][big]["n_units"] for r in hit_rows) / n_hit, 2) if n_hit else None,
            "mean_tokens": round(sum(r["budget_arms"][big]["tokens"] for r in hit_rows) / n_hit, 2) if n_hit else None,
        }

    summary = {
        "口径": {
            "generation": GENERATION,
            "gold_set": str(GOLD_SET.relative_to(ROOT)),
            "有 gold 题数": total_with_gold,
            "本次跑题数": n_planned,
            "top_k": TOP_K,
            "graph_depth": GRAPH_DEPTH,
            "max_context_tokens": TOKEN_BUDGET,
        },
        "a_in_context": {
            "hit分母": n_hit,
            "in_context": len(in_ctx),
            "in_context_rate": round(len(in_ctx) / n_hit, 4) if n_hit else None,
            "not_in_context": len(not_ctx),
            "not_in_context_reason": dict(why),
            "not_in_context_hit_rank_dist": dict(not_ctx_rank_dist),
            "not_in_context_hit_ranks": [r["hit_rank"] for r in not_ctx],
        },
        "b_ctx_pos_vs_hit_rank": {
            "equal": eq,
            "greater": gt,
            "less": lt,
            "delta_dist": {str(k): v for k, v in sorted(delta_dist.items())},
        },
        "c_ctx_pos_dist": {
            "ctx_pos": {b: ctx_dist.get(b, 0) for b in pos_buckets},
            "hit_rank": {b: hit_dist.get(b, 0) for b in pos_buckets},
            "after_pos_3": after3,
            "after_pos_5": after5,
            "hit_rank_after_3": hit_after3,
            "hit_rank_after_5": hit_after5,
            "conversion_by_hit_rank": conv,
            "R_at_3": {"num": len(le3), "den": n_hit,
                       "rate": round(len(le3) / n_hit, 4) if n_hit else None},
            "R_at_3_context_conversion": len(le3_in_ctx) / len(le3) if le3 else None,
            "R_at_5": {"num": len(le5), "den": n_hit,
                       "rate": round(len(le5) / n_hit, 4) if n_hit else None},
            "R_at_5_context_conversion": len(le5_in_ctx) / len(le5) if le5 else None,
            "R_at_10": {"num": len(le10), "den": n_hit,
                        "rate": round(len(le10) / n_hit, 4) if n_hit else None},
            "R_at_10_context_conversion": len(le10_in_ctx) / len(le10) if le10 else None,
        },
        "d_noise_before_gold": {
            "noise_total": noise_total,
            "from_hits": noise_from_hits,
            "from_expansions_only": noise_from_graph,
            "mean_per_question": round(noise_total / len(in_ctx), 2) if in_ctx else None,
            "per_question": {r["question_id"]: r["n_noise_before"] for r in in_ctx},
        },
        "truncation": trunc_summary,
        "counterfactual_no_expansions": cf,
        "budget_attribution": budget_attr,
        "truncation_overtake": overtaken_summary,
        "by_category": {},
    }

    # 分类别：名次中位 / 位置中位 / in_context 率
    cats = {}
    for r in hit_rows:
        cats.setdefault(r["category"], []).append(r)

    def med(xs):
        xs = sorted(xs)
        if not xs:
            return None
        m = len(xs) // 2
        return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2

    for cat, rs in sorted(cats.items()):
        summary["by_category"][cat] = {
            "n": len(rs),
            "in_context": sum(1 for r in rs if r["in_context"]),
            "in_context_rate": round(sum(1 for r in rs if r["in_context"]) / len(rs), 4),
            "median_hit_rank": med([r["hit_rank"] for r in rs]),
            "median_ctx_pos": med([r["ctx_pos"] for r in rs if r["ctx_pos"]]),
            "ctx_pos_eq_hit_rank": sum(1 for r in rs if r["ctx_pos"] and r["ctx_pos"] == r["hit_rank"]),
        }

    payload = {
        "schema": "arkrag.diag_context_positions/1",
        "purpose": "gold 块在最终 context 里的位置 vs 它在 hits 里的名次",
        **_origin(),
        "generation": GENERATION,
        "summary": summary,
        "per_question": rows,
    }

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n[ok] {out}  ({out.stat().st_size} bytes, {len(rows)} 题, "
          f"{time.time() - t_eval:.0f}s 评测)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
