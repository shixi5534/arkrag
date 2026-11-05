# -*- coding: utf-8 -*-
"""预算容量与上下文填充的实测：名次转化、token 缺口、预算单调性、优先级反转、填充策略。

为什么要有这个脚本
------------------
P2 期间对「预算 6000 -> 8000」下过两次结论，两次都基于不完整的分母：

* 第一次（探针在「6000 下已失败的 74 题」上重放）结论「变差 0 题」——取样假象，
  那批题 gold 本来就不在上下文里，变差必然恒 0。
* 第二次（全 300 应答题）抓到 1 题变差，据此撤销预算改动。数量对，但没算代价比，
  也没区分「变差的是什么样的题」。

本脚本把四件事放在同一个确定性通路里量清楚，全部只用结构量（gold 块是否进上下文），
不跑问答——避免把「进上下文」与「答得出」两层混在一起：

1. 名次 -> 上下文 的**逐名次**转化率（历史文档用的是分桶口径，掩盖了第 5 名的断崖）
2. 「gold 名次 <=5 却没进上下文」的 **token 缺口**（是差一点点，还是根本装不下）
3. 预算 6000..12000 的 **keep 集合单调性**（预算加大反而丢块，是真实算法性质还是测量误差）
4. **优先级反转**（direct 被截断、却保留 graph-only）与 **三档填充策略**的对照

关键实现事实（踩过才知道）
--------------------------
* `Retriever.build_context(self, hits, expansions, *, token_budget=None, query_analysis=None)`
  —— 参数名是 `token_budget`，不是 `budget_tokens`。
* `pipeline.build_pipeline()` 返回 **dict**（取 `pipe["retriever"]`），不是对象。
* `query.analyze_query(text, *, entities, aliases, vocab, max_chars, context)` 没有 `loaded=`
  参数，实体表与别名表得从 retriever 上取。
* 单位是 `u["tokens"]`；来源通道在 `u["sources"][i]["channel"]`，direct 通道是
  `retrieve.SOURCE_BM25` / `retrieve.SOURCE_ENTITY`。

所有数字只对 dev 350 应答题成立，验收集 260（new-holdout 160 + v1-holdout 100）一次没碰。
"""
from __future__ import annotations

import json
import platform
import statistics as st
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from arkrag import config, pipeline, query as query_mod, retrieve as retrieve_mod, subproc, vocab  # noqa: E402

GENERATION = "g-20261103T132152Z-06a741"
GOLD = ROOT / "data" / "goldset" / "gold-v2.json"
SPLIT = ROOT / "data" / "goldset" / "split-v2.json"
ROUND1 = ROOT / "data" / "a_p2_round1.json"
WEIGHT_GRID = ROOT / "data" / "a_p2_weight_grid.json"
OUT = ROOT / "data" / "lead_budget_and_fill.json"

BUDGETS = (6000, 6400, 7000, 8000, 10000, 12000)
BIG = 10 ** 7
DIRECT_CHANNELS = (retrieve_mod.SOURCE_BM25, retrieve_mod.SOURCE_ENTITY)


def _git_revision() -> str:
    # 只有一行 40 位十六进制 ASCII，理论上不加 env 也不会坏；但 `env=` 一律走
    # `subproc.child_env()`，免得白名单越加越长——AST 门报红时先看是不是自己在滑坡。
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                             capture_output=True, text=True,
                             encoding=subproc.PIPE_ENCODING, errors="replace",
                             env=subproc.child_env())
        text = (out.stdout or "").strip().splitlines()
        return text[0] if text else "unknown"
    except Exception as exc:  # pragma: no cover - 取证降级
        return "unknown(%s)" % exc


def _provenance(command: str) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_revision(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "command": command,
    }


def load_dev() -> list[dict]:
    items = json.loads(GOLD.read_text(encoding="utf-8"))["items"]
    return [it for it in items
            if it.get("split") == "dev" and not it.get("should_abstain") and it.get("gold_chunk_ids")]


def prepare(n: int | None = None) -> dict:
    loaded = pipeline.load_generation(GENERATION)
    pipe = pipeline.build_pipeline(loaded, top_k=20, graph_depth=1)
    retriever = pipe["retriever"]
    entities = pipe.get("entities") or retriever.graph.get("entities") or []
    dev = load_dev()
    if n:
        dev = dev[:n]
    rows = []
    for i, it in enumerate(dev, 1):
        qa = query_mod.analyze_query(
            it["question"], entities=entities,
            aliases=retriever.graph.get("alias_index"), vocab=vocab,
            context=getattr(retriever, "query_context", None),
        )
        found = retriever.search(qa, top_k=20)
        hits = found["hits"]
        seeds = [c["entity_id"] for c in qa.get("entity_candidates") or []]
        expanded = retriever.expand(seeds, depth=1, query_analysis=qa,
                                    seed_chunk_ids=[h["chunk_id"] for h in hits])
        big = retriever.build_context(hits, expanded["expansions"],
                                     token_budget=BIG, query_analysis=qa)
        real = retriever.build_context(hits, expanded["expansions"], query_analysis=qa)
        units = [{"chunk_id": u["chunk_id"],
                  "tokens": int(u.get("tokens") or 0),
                  "score": float(u.get("score") or 0.0),
                  "is_direct": any(s["channel"] in DIRECT_CHANNELS for s in u["sources"])}
                 for u in big["units"]]
        gids = set(it["gold_chunk_ids"])
        ranks = [j for j, u in enumerate(units) if u["chunk_id"] in gids]
        rows.append({
            "question_id": it["question_id"], "category": it["category"],
            "origin": it.get("origin"), "gold": sorted(gids),
            "units": units,
            "best_rank": (min(ranks) + 1) if ranks else None,
            "real_kept": [u["chunk_id"] for u in real["units"]],
            "real_units": len(real["units"]),
            "real_used": int(real["stats"]["tokens_used"]),
        })
        if i % 50 == 0:
            print("  ... 准备 %d/%d" % (i, len(dev)), file=sys.stderr)
    print("准备完成 dev n =", len(rows), file=sys.stderr)
    return {"retriever": retriever, "budget": int(retriever.budget["max_context_tokens"]), "rows": rows}


def fill_current(units: list[dict], budget: int) -> tuple[list[dict], int]:
    used, kept = 0, []
    for u in units:
        if used + u["tokens"] > budget:
            continue
        used += u["tokens"]
        kept.append(u)
    return kept, used


def fill_two_phase(units: list[dict], budget: int, *, demote_retry: bool = False) -> tuple[list[dict], int]:
    used, kept = 0, []
    direct = [u for u in units if u["is_direct"]]
    graph = [u for u in units if not u["is_direct"]]
    leftover = []
    for u in direct:
        if used + u["tokens"] > budget:
            leftover.append(u)
            continue
        used += u["tokens"]
        kept.append(u)
    if demote_retry:
        for u in sorted(leftover, key=lambda x: (x["tokens"], -x["score"])):
            if used + u["tokens"] > budget:
                continue
            used += u["tokens"]
            kept.append(u)
    for u in graph:
        if used + u["tokens"] > budget:
            continue
        used += u["tokens"]
        kept.append(u)
    return kept, used


def section_rank_conversion() -> dict:
    """逐名次转化率。用 A 的两份独立产物交叉验证（同一配置 cooc0）。"""
    sources = {}
    for name, path in (("a_p2_round1", ROUND1), ("a_p2_weight_grid", WEIGHT_GRID)):
        if not path.exists():
            sources[name] = {"available": False}
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        per = data["per_question"]["cooc0"]
        table = {}
        for row in per:
            rank = row.get("gold_first_rank")
            if not rank:
                continue
            bucket = table.setdefault(int(rank), {"n": 0, "in_context": 0})
            bucket["n"] += 1
            if row.get("any_gold"):
                bucket["in_context"] += 1
        sources[name] = {"available": True, "by_rank": table}
    if not any(v.get("available") for v in sources.values()):
        return {"available": False, "note": "两份逐题产物都不在盘上，逐名次表无法复算"}

    by_rank = {}
    agree = None
    keys = [k for k, v in sources.items() if v.get("available")]
    first = sources[keys[0]]["by_rank"]
    for rank in sorted(first):
        entry = {"n": first[rank]["n"], "in_context": first[rank]["in_context"],
                 "rate": round(first[rank]["in_context"] / first[rank]["n"], 4)}
        if len(keys) > 1:
            other = sources[keys[1]]["by_rank"].get(rank)
            entry["cross_check_same"] = bool(other and other == first[rank])
            if agree is None:
                agree = True
            if not entry["cross_check_same"]:
                agree = False
        by_rank[rank] = entry
    cum_n = cum_hit = 0
    cumulative = {}
    for rank in sorted(by_rank):
        cum_n += by_rank[rank]["n"]
        cum_hit += by_rank[rank]["in_context"]
        cumulative[rank] = {"n": cum_n, "in_context": cum_hit, "rate": round(cum_hit / cum_n, 4)}
    le5 = cumulative.get(5, {})
    return {
        "available": True,
        "sources": list(keys),
        "cross_check_identical": agree,
        "by_rank": {str(k): v for k, v in by_rank.items()},
        "cumulative": {str(k): v for k, v in cumulative.items()},
        "le5": le5,
        "note": "分桶口径（1-3 / 4-5）会把第 5 名的断崖摊平；逐名次才看得见。",
    }


def section_shortfall(ctx: dict) -> dict:
    """gold 名次 <=5 却没进上下文：缺多少 token。"""
    budget = ctx["budget"]
    rows = []
    for r in ctx["rows"]:
        rank = r["best_rank"]
        if not rank or rank > 5:
            continue
        if set(r["real_kept"]) & set(r["gold"]):
            continue
        kept, used = fill_current(r["units"], budget)
        gold_unit = next(u for u in r["units"] if u["chunk_id"] in set(r["gold"]))
        rows.append({
            "question_id": r["question_id"], "category": r["category"],
            "best_rank": rank, "gold_tokens": gold_unit["tokens"],
            "tokens_used": used, "units": len(kept),
            "leftover": budget - used,
            "shortfall": used + gold_unit["tokens"] - budget,
        })
    gaps = [x["shortfall"] for x in rows]
    return {
        "n": len(rows),
        "rows": rows,
        "median_shortfall": int(st.median(gaps)) if gaps else None,
        "max_shortfall": max(gaps) if gaps else None,
        "note": "缺口 = 已用 + gold 块自身 token - 预算；正数表示差多少才能装进去。"
                "缺口远小于块长说明是「差一点点」，接近块长说明前面被更大的块吃掉配额。",
    }


def section_budget_curve(ctx: dict) -> dict:
    budget = ctx["budget"]
    grid = {}
    for b in BUDGETS:
        any_hit = all_hit = 0
        unit_sum = used_sum = 0
        for r in ctx["rows"]:
            kept, used = fill_current(r["units"], b)
            ids = {u["chunk_id"] for u in kept}
            gids = set(r["gold"])
            if ids & gids:
                any_hit += 1
            if gids <= ids:
                all_hit += 1
            unit_sum += len(kept)
            used_sum += used
        grid[b] = {
            "any_gold": any_hit, "all_gold": all_hit,
            "avg_units": round(unit_sum / len(ctx["rows"]), 3),
            "avg_used": round(used_sum / len(ctx["rows"]), 1),
        }
    # 相邻档救/丢
    adjacent = []
    for a, b in zip(BUDGETS, BUDGETS[1:]):
        gain = lose = 0
        g_all = l_all = 0
        lose_ids, lall_ids = [], []
        for r in ctx["rows"]:
            ka, _ = fill_current(r["units"], a)
            kb, _ = fill_current(r["units"], b)
            ia, ib = {u["chunk_id"] for u in ka}, {u["chunk_id"] for u in kb}
            gids = set(r["gold"])
            ha, hb = bool(ia & gids), bool(ib & gids)
            if hb and not ha:
                gain += 1
            if ha and not hb:
                lose += 1
                lose_ids.append(r["question_id"])
            if gids <= ia and not gids <= ib:
                l_all += 1
                lall_ids.append(r["question_id"])
            if gids <= ib and not gids <= ia:
                g_all += 1
        adjacent.append({"from": a, "to": b, "any_gain": gain, "any_lose": lose,
                         "any_net": gain - lose, "any_lose_ids": lose_ids,
                         "all_gain": g_all, "all_lose": l_all, "all_lose_ids": lall_ids})
    # 相对基线的对比
    versus_base = []
    base_kept = {}
    for r in ctx["rows"]:
        kept, _ = fill_current(r["units"], budget)
        base_kept[r["question_id"]] = {u["chunk_id"] for u in kept}
    for b in BUDGETS:
        if b == budget:
            continue
        gain = lose = g_all = l_all = 0
        lose_ids = []
        for r in ctx["rows"]:
            kept, _ = fill_current(r["units"], b)
            ids = {u["chunk_id"] for u in kept}
            gids = set(r["gold"])
            base = base_kept[r["question_id"]]
            if ids & gids and not base & gids:
                gain += 1
            if base & gids and not ids & gids:
                lose += 1
                lose_ids.append(r["question_id"])
            if gids <= ids and not gids <= base:
                g_all += 1
            if gids <= base and not gids <= ids:
                l_all += 1
        versus_base.append({"budget": b, "any_gain": gain, "any_lose": lose,
                            "any_net": gain - lose, "any_lose_ids": lose_ids,
                            "all_gain": g_all, "all_lose": l_all})
    # 非单调：某题在更高预算下反而丢了块
    nonmono = {"any": [], "all": []}
    for r in ctx["rows"]:
        prev_any = None
        prev_all = None
        trail = []
        for b in BUDGETS:
            kept, _ = fill_current(r["units"], b)
            ids = {u["chunk_id"] for u in kept}
            gids = set(r["gold"])
            cur_any, cur_all = bool(ids & gids), gids <= ids
            if prev_any is True and not cur_any:
                nonmono["any"].append({"question_id": r["question_id"], "budget": b})
            if prev_all is True and not cur_all:
                nonmono["all"].append({"question_id": r["question_id"], "budget": b})
            prev_any, prev_all = cur_any, cur_all
            trail.append((b, len(ids)))
    return {
        "grid": {str(k): v for k, v in grid.items()},
        "adjacent": adjacent,
        "versus_base": versus_base,
        "non_monotonic": {"any": nonmono["any"], "all": nonmono["all"]},
        "note": "keep 集合在更高预算下可能变差（direct 大块进来挤掉 direct 小块），"
                "这是贪心首适配 + `continue` 的算法性质；判据只能看 gold 侧救/丢，不能看 keep 集合本身。",
    }


def section_frontier(ctx: dict, shortfall: dict) -> dict:
    frontier = [x["question_id"] for x in shortfall["rows"]]
    by_q = {r["question_id"]: r for r in ctx["rows"]}
    out = []
    for b in BUDGETS:
        hit = []
        for qid in frontier:
            kept, _ = fill_current(by_q[qid]["units"], b)
            if {u["chunk_id"] for u in kept} & set(by_q[qid]["gold"]):
                hit.append(qid)
        out.append({"budget": b, "recovered": len(hit), "n": len(frontier)})
    return {"frontier": frontier, "by_budget": out,
            "note": "frontier = 名次 <=5 却因预算没进上下文的题；它们最便宜能救回来。"}


def section_inversion_and_fill(ctx: dict) -> dict:
    budget = ctx["budget"]
    inv = []
    gold_truncated = 0
    for r in ctx["rows"]:
        kept, used = fill_current(r["units"], budget)
        kept_ids = {u["chunk_id"] for u in kept}
        skipped = [u for u in r["units"] if u["chunk_id"] not in kept_ids]
        kept_graph = [u for u in kept if not u["is_direct"]]
        skipped_direct = [u for u in skipped if u["is_direct"]]
        if skipped_direct and kept_graph:
            inv.append({"question_id": r["question_id"], "category": r["category"],
                        "skipped_direct": len(skipped_direct),
                        "kept_graph": len(kept_graph),
                        "leftover": budget - used,
                        "gold_skipped": [u["chunk_id"] for u in skipped_direct
                                         if u["chunk_id"] in set(r["gold"])]})
        if any(u["chunk_id"] in set(r["gold"]) for u in skipped_direct):
            gold_truncated += 1
    policies = {}
    for name, fn in (("A_现状贪心", lambda u, b: fill_current(u, b)),
                     ("B_严格两段", lambda u, b: fill_two_phase(u, b)),
                     ("C_两段+降级补位", lambda u, b: fill_two_phase(u, b, demote_retry=True))):
        any_hit = all_hit = 0
        unit_sum = used_sum = 0
        for r in ctx["rows"]:
            kept, used = fn(r["units"], budget)
            ids = {u["chunk_id"] for u in kept}
            gids = set(r["gold"])
            if ids & gids:
                any_hit += 1
            if gids <= ids:
                all_hit += 1
            unit_sum += len(kept)
            used_sum += used
        policies[name] = {"any_gold": any_hit, "all_gold": all_hit,
                          "avg_units": round(unit_sum / len(ctx["rows"]), 3),
                          "avg_used": round(used_sum / len(ctx["rows"]), 1)}
    # A 档必须与真实现逐题一致，否则复算不可信
    mismatch = []
    for r in ctx["rows"]:
        kept, used = fill_current(r["units"], budget)
        if len(kept) != r["real_units"] or used != r["real_used"]:
            mismatch.append(r["question_id"])
    return {
        "inversion_n": len(inv),
        "inversion_rate": round(len(inv) / len(ctx["rows"]), 4),
        "inversion_rows": inv,
        "gold_truncated_n": gold_truncated,
        "fill_policies": policies,
        "A_match_real_mismatch": mismatch,
        "note": "优先级反转 = 有 direct 单元被截断、却有 graph-only 单元被保留；"
                "docstring 写「直接证据优先于图背景」，填充却用 continue。"
                "A 档与真实现逐题一致才说明复算可信。",
    }


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        # 不 reconfigure 的话，写出 cp936 编不出的字符会让进程以 1 退出，
        # 而这个 1 与「分析失败」长得一模一样。
        sys.stdout.reconfigure(encoding="utf-8")
    argv = list(sys.argv[1:] if argv is None else argv)
    limit = None
    if "--limit" in argv:
        limit = int(argv[argv.index("--limit") + 1])

    ctx = prepare(limit)
    shortfall = section_shortfall(ctx)
    rank_conv = section_rank_conversion()
    curve = section_budget_curve(ctx)
    frontier = section_frontier(ctx, shortfall)
    invfill = section_inversion_and_fill(ctx)

    print("\n=== 1 逐名次转化 ===")
    if rank_conv.get("available"):
        print("来源 %s  交叉验证逐位相同 = %s" % (rank_conv["sources"], rank_conv["cross_check_identical"]))
        for rank, v in rank_conv["by_rank"].items():
            print("  rank=%-3s n=%-4d 进上下文 %-4d = %.4f" % (rank, v["n"], v["in_context"], v["rate"]))
        print("  名次<=5 共 %s 题，进上下文 %s，转化率 %s"
              % (rank_conv["le5"].get("n"), rank_conv["le5"].get("in_context"), rank_conv["le5"].get("rate")))
    else:
        print("  不可用：", rank_conv.get("note"))

    print("\n=== 2 frontier 缺口（名次<=5 未进上下文）===")
    print("  n = %d  缺口中位 %s  最大 %s"
          % (shortfall["n"], shortfall["median_shortfall"], shortfall["max_shortfall"]))
    for row in shortfall["rows"]:
        print("  %-14s %-15s rank=%d gold_t=%d used=%d units=%d 缺口=%d"
              % (row["question_id"], row["category"], row["best_rank"],
                 row["gold_tokens"], row["tokens_used"], row["units"], row["shortfall"]))

    print("\n=== 3 预算曲线 ===")
    print("  预算    any_gold  all_gold  平均单元  平均已用")
    for b, v in curve["grid"].items():
        print("  %-7s %4d      %4d      %.2f      %.0f"
              % (b, v["any_gold"], v["all_gold"], v["avg_units"], v["avg_used"]))
    print("  相邻档：")
    for a in curve["adjacent"]:
        print("    %5d->%-5d any 救%d丢%d净%+d  all 救%d丢%d"
              % (a["from"], a["to"], a["any_gain"], a["any_lose"], a["any_net"],
                 a["all_gain"], a["all_lose"]))
    print("  相对基线 %d：" % ctx["budget"])
    for v in curve["versus_base"]:
        print("    %5d any 救%d丢%d净%+d  all 救%d丢%d  丢:%s"
              % (v["budget"], v["any_gain"], v["any_lose"], v["any_net"],
                 v["all_gain"], v["all_lose"], v["any_lose_ids"] or "-"))
    print("  gold 侧非单调（更高预算反而丢）：any %d 次  all %d 次"
          % (len(curve["non_monotonic"]["any"]), len(curve["non_monotonic"]["all"])))
    print("  frontier %d 题在各预算下的收复情况：" % len(frontier["frontier"]))
    for v in frontier["by_budget"]:
        print("    %5d 收复 %d/%d" % (v["budget"], v["recovered"], v["n"]))

    print("\n=== 4 优先级反转与填充策略 ===")
    print("  反转 %d/%d = %.4f  gold 块被截断 %d 题"
          % (invfill["inversion_n"], len(ctx["rows"]), invfill["inversion_rate"],
             invfill["gold_truncated_n"]))
    print("  A 档与真实现不一致题数 =", len(invfill["A_match_real_mismatch"]))
    for name, v in invfill["fill_policies"].items():
        print("  %-18s any %3d  all %3d  平均单元 %.2f  平均已用 %.0f"
              % (name, v["any_gold"], v["all_gold"], v["avg_units"], v["avg_used"]))

    payload = {
        "schema": 1,
        "purpose": "预算容量与上下文填充实测：逐名次转化、frontier 缺口、预算单调性、优先级反转与填充策略",
        **_provenance("py -3 scripts/lead_budget_and_fill.py"),
        "generation": GENERATION,
        "budget": ctx["budget"],
        "dev_n": len(ctx["rows"]),
        "1_名次转化": rank_conv,
        "2_frontier缺口": shortfall,
        "3_预算曲线": curve,
        "4_优先级反转与填充": invfill,
        "5_frontier收复": frontier,
        "6_限制": [
            "只对 dev 350 应答题成立，验收集 260 未观测，结论不得外推。",
            "全部是结构量（gold 块是否进上下文），不是真跑的答案与引用覆盖。",
            "逐名次转化表依赖 A 的两份逐题产物，本脚本只做复算与交叉验证。",
            "keep 集合非单调是真实算法性质；但 gold 侧非单调次数很少，不能拿个别题否定整体。",
        ],
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n写入", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
