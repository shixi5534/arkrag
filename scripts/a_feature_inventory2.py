#!/usr/bin/env python
"""第二遍特征盘点：只测两个**在线可得**的新信号（纯测量，不改排序代码）。

## 为什么单独一遍

`scripts/a_feature_inventory.py` 的第一遍量了 12 个维度，结论是：

* `rel_gold`（以 gold_answer 为关系词）净 +11，但**多数题 gold_answer 是实体名
  不是关系词**（多跳题问的就是中间实体名），且该值线上拿不到 —— 它是 oracle，
  只能证明「答案原文在 gold 块里」，不能当特征用。
* `rel_hint`（线上真能拿到的关系提示）净 **−8**，是负增益。
* 其余维度单维重排全部净 ≤ 0。

所以第一遍里唯一可实施的口径，是「答案原文在块里」这个**结构性事实**。本遍把它
换成**在线可算**的形式，并单独量它的上界。

## 两个被测量的信号

### S1 `common_neighbor`（多跳题专用，在线可得）

问句解析出两个端点实体 A、B → 在图邻接里求共同邻居集合
`N = neighbors(A) ∩ neighbors(B)` → 看块正文里出现了几个 `N` 的成员名。

不含 gold 答案，不需要任何标注。若共同邻居唯一且块里出现它，这块几乎必然
就是承载该跳的块。

同时量两个变体，用来分离「图里有没有这个结构」和「块里有没有这个名字」：
* `cn_count` —— 块里出现的共同邻居**个数**
* `cn_unique` —— 共同邻居**恰好唯一**且出现在块里（0/1）

### S2 `rel_line_both`（关系题专用，在线可得）

gold 关系词与两个端点**同一行**共现。这里用 gold_answer 作关系词只是为了让
测量跑起来——线上要用的是「问句里出现的关系词」，而 direct_relation 的问法
是「处于哪种关系」，句子里**没有**关系词，所以 S2 的 oracle 部分必须标清楚，
不能直接当特征。本遍的主要目的是量化「即使知道关系词，这个信号值不值得做」。

## 用法

    py -3 scripts/a_feature_inventory2.py
    py -3 scripts/a_feature_inventory2.py --limit 10
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import config, pipeline, query as query_mod, retrieve, vocab, subproc  # noqa: E402

CAND = "g-20261103T132152Z-06a741"
TRIAGE = ROOT / "data" / "a_triage_all320.jsonl"
GOLD_SET = ROOT / "data" / "goldset" / "gold-v1.json"
EVAL_TOP_K = 20


def _norm(s) -> str:
    return query_mod._normalize(str(s or ""))


def deep_search(retriever, query_analysis: dict, pool: int) -> dict:
    """同 a_failure_triage.deep_search：压 CANDIDATE_POOL_MULT 到 1，让 depth == pool。"""
    old = retrieve.CANDIDATE_POOL_MULT
    retrieve.CANDIDATE_POOL_MULT = 1
    try:
        return retriever.search(query_analysis, top_k=pool)
    finally:
        retrieve.CANDIDATE_POOL_MULT = old


def neighbor_ids(adjacency: dict, eid: str) -> set[str]:
    """无向邻居集合。`adjacency[eid]["out"]` 已含双向边（directions 里带
    `undirected`/`forward`/`backward`），这里把同样出现在 `in` 里的也算上，
    避免只走一个方向漏掉邻居。"""
    slot = adjacency.get(eid) or {}
    out: set[str] = set()
    for key in ("out", "in"):
        for e in slot.get(key) or []:
            nid = str(e.get("neighbor_id") or "")
            if nid:
                out.add(nid)
    out.discard(eid)
    return out


def analyze(args) -> int:
    t0 = time.time()
    rows = [json.loads(line) for line in TRIAGE.open(encoding="utf-8") if line.strip()]
    targets = [r for r in rows if str(r.get("primary") or "") in
               ("找到但未进前部", "上下文未采用")]
    if args.limit:
        targets = targets[: args.limit]

    print(f"题 {len(targets)}  世代 {args.generation}  深池 {args.pool}")
    loaded = pipeline.load_generation(args.generation)
    pipe = pipeline.build_pipeline(loaded, top_k=EVAL_TOP_K, graph_depth=1)
    rt = pipe["retriever"]
    entities = rt.graph.get("entities") or []
    adj = rt.adjacency
    alias_map = rt.graph.get("alias_index")

    per_question: list[dict] = []
    # 单维重排的救回/打坏计数
    stats = {k: {"rescued": 0, "broken": 0} for k in ("cn_count", "cn_unique", "hop_line", "rel_line_both")}

    for i, tgt in enumerate(targets, 1):
        qid = tgt["question_id"]
        cat = str(tgt.get("category") or "")
        qa = query_mod.analyze_query(
            tgt["question"], entities=entities, aliases=alias_map,
            vocab=vocab, context=getattr(rt, "query_context", None),
        )
        hits = (deep_search(rt, qa, args.pool).get("hits") or [])
        if not hits:
            continue
        order = [h["chunk_id"] for h in hits]
        rank_of = {cid: k + 1 for k, cid in enumerate(order)}
        gold_ids = [str(g) for g in (tgt.get("gold_chunk_ids") or [])]
        gold_ranks = {g: rank_of[g] for g in gold_ids if g in rank_of}
        if not gold_ranks:
            continue
        best_rank = min(gold_ranks.values())
        baseline_ok = best_rank <= EVAL_TOP_K

        # ---- S1 共同邻居 ----
        cs = list(qa.get("entity_candidates") or [])
        common_names: list[str] = []
        common_ids: set[str] = set()
        if len(cs) >= 2:
            a = str(cs[0].get("entity_id") or "")
            b = str(cs[1].get("entity_id") or "")
            if a and b:
                common_ids = neighbor_ids(adj, a) & neighbor_ids(adj, b)
                for eid in common_ids:
                    name = rt.display_name(eid)
                    if name:
                        common_names.append(name)

        rel_word = _norm(tgt.get("gold_answer") or "")
        # 两个端点的规范名（用于「同行」判定）
        end_names: list[str] = []
        for c in cs[:2]:
            ent = rt.entities_by_id.get(str(c.get("entity_id") or "")) or {}
            nm = _norm(ent.get("name") or c.get("name") or "")
            if nm:
                end_names.append(nm)

        def feats(cid: str) -> dict:
            rec = rt.chunks.get(cid) or {}
            text = str(rec.get("text") or "")
            present = [n for n in common_names if _norm(n) and _norm(n) in text]
            # S1b `hop_line`：共同邻居名与**至少一个端点**出现在同一行。
            #
            # 为什么必须加「同行」这一条：`cn_count`（只数块内出现几个共同邻居名）
            # 是净 −1 —— 阿米娅、罗德岛这类枢纽实体几乎每个块都出现过，计数等于
            # 在给枢纽块加分。加上同行约束之后，衡量的是「这一行同时提到了端点
            # 和那个中间人」，才是「这一行在讲这一跳」的证据。
            hop_line = 0
            cn_norm = [(_norm(n), n) for n in common_names]
            for line in text.splitlines():
                ln = _norm(line)
                if not any(e in ln for e in end_names):
                    continue
                if any(t and t in ln for t, _ in cn_norm):
                    hop_line = 1
                    break
            # S2 关系词与两端点同行
            rel_line = 0
            if rel_word and len(end_names) >= 2:
                for line in text.splitlines():
                    ln = _norm(line)
                    if rel_word in ln and all(n in ln for n in end_names):
                        rel_line = 1
                        break
            return {
                "cn_count": len(present),
                "cn_unique": 1 if (len(common_ids) == 1 and present) else 0,
                "hop_line": hop_line,
                "rel_line_both": rel_line,
            }

        pool_feats = {cid: feats(cid) for cid in order}
        gold_present = any(pool_feats[g]["cn_count"] > 0 for g in gold_ranks)
        rescued = {}
        for key in ("cn_count", "cn_unique", "hop_line", "rel_line_both"):
            reordered = sorted(order, key=lambda cid, k=key: (-pool_feats[cid][k], rank_of[cid]))
            ok = any(g in set(reordered[:EVAL_TOP_K]) for g in gold_ids)
            rescued[key] = ok
            if ok and not baseline_ok:
                stats[key]["rescued"] += 1
            if baseline_ok and not ok:
                stats[key]["broken"] += 1

        per_question.append({
            "question_id": qid,
            "category": cat,
            "primary": tgt.get("primary"),
            "question": tgt.get("question"),
            "gold_answer": tgt.get("gold_answer"),
            "best_gold_rank": best_rank,
            "baseline_in_top": baseline_ok,
            "n_common_neighbors": len(common_ids),
            "common_neighbor_names": sorted(common_names)[:12],
            "gold_block_has_common_neighbor": gold_present,
            "gold_feats": {g: pool_feats[g] for g in gold_ranks},
            "rerank_in_top": rescued,
        })
        if i % 20 == 0:
            print(f"  ... {i}/{len(targets)}  {time.time() - t0:.0f}s")

    if not per_question:
        print("[4] 自检失败：无逐题结果", file=sys.stderr)
        return 4

    n = len(per_question)
    by_cat: dict[str, dict] = {}
    for r in per_question:
        s = by_cat.setdefault(r["category"], {"n": 0, "gold_has_cn": 0, "cn_available": 0,
                                              "recall_by_dim": {k: 0 for k in stats}})
        s["n"] += 1
        s["gold_has_cn"] += 1 if r["gold_block_has_common_neighbor"] else 0
        s["cn_available"] += 1 if r["n_common_neighbors"] > 0 else 0
        for k in stats:
            s["recall_by_dim"][k] += 1 if r["rerank_in_top"][k] else 0

    summary = {
        "script": "scripts/a_feature_inventory2.py",
        "purpose": "共同邻居 / 关系词同行 两个在线信号的上界测量（纯测量，不改排序）",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "command": "py -3 scripts/a_feature_inventory2.py",
        "generation": args.generation,
        "gold_set": config.rel_to_root(GOLD_SET),
        "triage_source": config.rel_to_root(TRIAGE),
        "pool": args.pool,
        "eval_top_k": EVAL_TOP_K,
        "n_questions": n,
        "baseline_in_top": sum(1 for r in per_question if r["baseline_in_top"]),
        "common_neighbor_available": sum(1 for r in per_question if r["n_common_neighbors"] > 0),
        "gold_has_common_neighbor": sum(1 for r in per_question if r["gold_block_has_common_neighbor"]),
        "single_dim_rerank": {k: {"rescued": v["rescued"], "broken": v["broken"],
                                  "net": v["rescued"] - v["broken"]} for k, v in stats.items()},
        "by_category": by_cat,
        "questions": per_question,
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n=== n={n}，baseline 已在 top20 {summary['baseline_in_top']} 题 ===")
    print(f"两端点有共同邻居的题：{summary['common_neighbor_available']}")
    print(f"gold 块含某共同邻居名：{summary['gold_has_common_neighbor']}")
    print(f"\n{'维度':16s} {'救回':>5s} {'打坏':>5s} {'净':>5s}")
    for k, v in stats.items():
        print(f"{k:16s} {v['rescued']:5d} {v['broken']:5d} {v['rescued'] - v['broken']:+5d}")
    print("\n--- 按类（gold 块含共同邻居名 / 该类题数）---")
    for c in sorted(by_cat):
        s = by_cat[c]
        print(f"  {c:16s} {s['gold_has_cn']:3d}/{s['n']:3d}  有共同邻居 {s['cn_available']:3d}")
    print(f"\n写出 {config.rel_to_root(out)}  {time.time() - t0:.0f}s")
    return 0


def _git_commit() -> str:
    try:
        import subprocess
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                             capture_output=True, text=True, timeout=10,
                             env=subproc.child_env())
        return (out.stdout or "").strip() or "unknown"
    except Exception:
        return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generation", default=CAND)
    ap.add_argument("--pool", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-json", default=str(ROOT / "data" / "a_feature_inventory2.json"))
    return analyze(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
