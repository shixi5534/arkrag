#!/usr/bin/env python
"""gold 块与竞争对手的可解释特征盘点（**纯测量，不改排序、不改参数**）。

## 这份脚本回答什么问题

`data/a_triage_all320.jsonl` 已经量出：gold 进深池 0.99，进前 20 只有 0.7833。
于是真正的问题不是「再加个特征试试」，而是：

    **gold 块和非 gold 块，在可解释特征上到底差在哪？**

本脚本对每一道「块已经到了池里但没排进前部」的题，取 gold 块与它在深池里
**前后各 `WINDOW` 个竞争者**，逐块算一组可解释量，然后产出两张表：

1. **「gold 在哪一维上输给竞争对手」的分布表** —— 按维统计「gold 严格弱于
   窗口内最强对手」的题数。弱在哪一维，就说明哪一维有可用的区分信号。
2. **「只按某一维重排能救回多少」的上界估计** —— 单维重排的离线回放。
   这是上界，不是收益预测：单维重排会打坏本来排对的题，脚本同时给出
   「救回」和「打坏」两侧，两个数必须一起看。

## 纪律

* 只读。不写 `data/goldset/`，不改 `src/arkrag/retrieve.py`，不改任何预算常量。
* 单维重排的「上界」只用于判断**该维是否含信息**，不作为「实施该方案能涨多少」
  的依据。真正的收益必须由 B 冻结保留集后在真管线上测。
* 所有路径经 `config.rel_to_root()` 写出，不落绝对机台路径
  （`tests/test_portable_paths.py` 会判红）。

## 特征口径

分三组，含义不同，不要合成一个数：

**A. 端点命中**（问句里问到的实体，在块里出现没有）
    `subj_exact` / `obj_exact` —— 主语、宾语分别算，归一化后**精确**匹配，
    不是模糊匹配。`subj_alias` / `obj_alias` 走实体别名表。
    分开算的理由：多跳题的典型病是「一块里出现了端点 A 但没出现 B」，
    合成一个「命中数」会把这种信息抹掉。

**B. 关系词命中**（问的是什么关系，那个关系词在块里出现没有）
    `rel_gold` —— 以 gold_answer 为关系词（**oracle**，仅用于测量信号是否存在，
    线上拿不到这个值）。
    `rel_hint` —— 以 `query_analysis["relation_hints"]` 为关系词（**线上真的
    拿得到**）。两者之差就是「关系识别缺失造成的损失」。

**C. 结构与分数**
    `chars` / `tokens` / `bm25_raw` / `bm25_pct` / `score` / `score_pct`
    `title_hit` / `speaker_hit` / `chapter_hit`
    `cooccur` —— 两个端点是否在**同一行**共现（多跳/关系题的硬信号）。

`bm25_pct` / `score_pct` 是**在本题深池内的分位**（0=池底，1=池顶），
不是跨题可比量；跨题比较只看名次。

## 用法

    py -3 scripts/a_feature_inventory.py                     # 全量 136 题
    py -3 scripts/a_feature_inventory.py --limit 10           # 冒烟
    py -3 scripts/a_feature_inventory.py --window 8           # 换竞争者窗口
    py -3 scripts/a_feature_inventory.py --layers 上下文未采用  # 只测一类失败层

退出码：0 跑通；2 输入不可用；4 自检失败。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import config, pipeline, query as query_mod, retrieve, vocab, subproc  # noqa: E402

CAND = "g-20261103T132152Z-06a741"
TRIAGE = ROOT / "data" / "a_triage_all320.jsonl"
GOLD_SET = ROOT / "data" / "goldset" / "gold-v1.json"

# 生产评测口径（与 scripts/ask.py:evaluate() 第 233 行、a_failure_triage 对齐）
EVAL_TOP_K = 20
EVAL_GRAPH_DEPTH = 1

# 两个目标失败层。前者「块在池里但名次 > 20」，后者「进了前 20 但被预算截掉」。
LAYER_NOT_IN_TOP = "找到但未进前部"
LAYER_NOT_IN_CTX = "上下文未采用"


# ---------------------------------------------------------------- 特征计算
def _norm(s: str) -> str:
    return query_mod._normalize(str(s or ""))


def entity_terms(retriever, cand: dict) -> tuple[set[str], set[str]]:
    """返回 `(规范名集合, 别名集合)`，都已归一化。

    别名只取 `aliases`（已确认的别名表），不含 `pending_aliases`：待定别名
    没有经过人工确认，拿它当命中依据会把「疑似同名」当成「同名」。
    """
    eid = str(cand.get("entity_id") or "")
    ent = retriever.entities_by_id.get(eid) or {}
    canonical = {_norm(ent.get("name") or cand.get("name") or "")}
    canonical.discard("")
    aliases: set[str] = set()
    for item in ent.get("aliases") or []:
        text = item.get("text") if isinstance(item, dict) else item
        t = _norm(text)
        if t:
            aliases.add(t)
    return canonical, aliases


def endpoint_hit(text_norm: str, canonical: set[str], aliases: set[str]) -> tuple[int, int]:
    """`(精确命中, 别名命中)`。别名命中要求该别名**不在**规范名集合里重复计。"""
    exact = sum(1 for t in canonical if t and t in text_norm)
    alias = sum(1 for t in aliases if t and t not in canonical and t in text_norm)
    return exact, alias


def line_cooccur(text: str, terms: list[str]) -> bool:
    """两个端点是否出现在**同一行**。同行才是关系题的硬信号——
    一个 900 字的块里什么名字都能凑齐，按整块判没有区分度。"""
    for line in str(text or "").splitlines():
        if all(t and t in line for t in terms):
            return True
    return False


def chunk_features(
    retriever,
    *,
    rec: dict,
    qa: dict,
    gold_rel: str,
    bm25_raw: float,
    bm25_pool: list[float],
    merged_score: float,
    merged_pool: list[float],
    rank: int,
    pool_size: int,
) -> dict:
    """单块特征。`rank` 从 1 开始。"""
    text = str(rec.get("text") or "")
    text_norm = _norm(text)
    title_norm = _norm(rec.get("title") or "")
    chapter_norm = _norm(rec.get("chapter") or "")
    speakers_norm = [_norm(s) for s in (rec.get("speakers") or [])]

    cands = list(qa.get("entity_candidates") or [])
    subj_exact = subj_alias = obj_exact = obj_alias = 0
    if len(cands) >= 1:
        canon, alias = entity_terms(retriever, cands[0])
        subj_exact, subj_alias = endpoint_hit(text_norm, canon, alias)
    if len(cands) >= 2:
        canon, alias = entity_terms(retriever, cands[1])
        obj_exact, obj_alias = endpoint_hit(text_norm, canon, alias)

    # 关系词：gold_answer 是 oracle，relation_hints 是线上可得
    rel_gold_hit = 1 if gold_rel and _norm(gold_rel) in text_norm else 0
    hint_words = [str(h.get("canonical") or h.get("raw") or "") for h in (qa.get("relation_hints") or [])]
    hint_words = [w for w in hint_words if w]
    rel_hint_hit = 1 if any(_norm(w) in text_norm for w in hint_words) else 0

    # 端点同行共现（只在两端点都判过时才有意义）
    end_terms: list[str] = []
    for c in cands[:2]:
        canon, _ = entity_terms(retriever, c)
        end_terms.extend(sorted(canon))
    cooccur = 1 if len(end_terms) >= 2 and line_cooccur(text, end_terms) else 0

    def pct(value: float, pool_list: list[float]) -> float:
        if not pool_list:
            return 0.0
        below = sum(1 for v in pool_list if v < value)
        return round(below / len(pool_list), 6)

    return {
        # A 端点命中
        "subj_exact": subj_exact,
        "subj_alias": subj_alias,
        "obj_exact": obj_exact,
        "obj_alias": obj_alias,
        # B 关系词
        "rel_gold": rel_gold_hit,
        "rel_hint": rel_hint_hit,
        # C 结构
        "title_hit": 1 if any(_norm(t) in title_norm for t in end_terms if t) else 0,
        "speaker_hit": 1 if any(t and any(t == s for s in speakers_norm) for t in end_terms) else 0,
        "chapter_hit": 1 if any(t and t in chapter_norm for t in end_terms if t) else 0,
        "cooccur": cooccur,
        "chars": len(text),
        "tokens": retrieve.estimate_tokens(text),
        "bm25_raw": round(float(bm25_raw), 6),
        "bm25_pct": pct(float(bm25_raw), bm25_pool),
        "score": round(float(merged_score), 6),
        "score_pct": pct(float(merged_score), merged_pool),
        "rank": rank,
        "pool_size": pool_size,
    }


# 用于「gold 是否输给对手」比较的维度。都是「越大越好」。
COMPARE_DIMS = [
    "subj_exact", "subj_alias", "obj_exact", "obj_alias",
    "rel_gold", "rel_hint", "title_hit", "speaker_hit", "chapter_hit",
    "cooccur", "bm25_raw", "score",
]

# 单维重排候选维度。`rank` 与 `chars` 是「越小越好」，排序时取负。
RERANK_DIMS = {
    "subj_exact": 1, "obj_exact": 1, "subj_alias": 1, "obj_alias": 1,
    "rel_gold": 1, "rel_hint": 1, "cooccur": 1, "title_hit": 1,
    "speaker_hit": 1, "bm25_raw": 1, "score": 1,
    "chars_asc": -1,
}


# ---------------------------------------------------------------- 主流程
def deep_search(retriever, query_analysis: dict, pool: int) -> dict:
    """按**生产排序公式**取深池，只是不截到 top_k。

    做法与 `scripts/a_failure_triage.py:deep_search` 一致：把
    `CANDIDATE_POOL_MULT` 临时压到 1，让 `top_k` 自己承担深度，这样内部
    `depth = pool`，正好等于生产 `search(top_k=20)` 的 `depth = 20*10`，
    三路候选集与 `_norm_scores()` 归一分母与生产同源。

    直接调 `search(top_k=pool)` 会把 depth 顶到 `pool*10`，归一化基准漂移，
    前 20 排序就变了（a_failure_triage 初版踩过：20 题里 5 题不一致）。
    """
    old = retrieve.CANDIDATE_POOL_MULT
    retrieve.CANDIDATE_POOL_MULT = 1
    try:
        return retriever.search(query_analysis, top_k=pool)
    finally:
        retrieve.CANDIDATE_POOL_MULT = old


def bm25_lookup(retriever, qa: dict, pool: int) -> dict[str, float]:
    """本题深池内每个块的**原始** BM25 分（未归一化）。

    单独取一次的理由：`search()` 返回的 `sources[].score` 已被
    `_norm_scores()` 按该路最高分归一，跨题不可比；而「原始分在池内的分位」
    才是本题内的相对位置。代价是再扫一次倒排（实测 141 ms/题）。
    """
    text = str(qa.get("retrieval_query") or qa.get("normalized_query") or "")
    if not text:
        return {}
    return {str(cid): float(s) for cid, s in retriever.index.search(text, top_k=pool)}


def load_targets(path: Path, layers: set[str]) -> list[dict]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    return [r for r in rows if str(r.get("primary") or "") in layers]


def inventory(args) -> int:
    t0 = time.time()
    if not TRIAGE.exists():
        print(f"[2] 找不到 {config.rel_to_root(TRIAGE)}，请先跑 scripts/a_failure_triage.py",
              file=sys.stderr)
        return 2

    layers = {LAYER_NOT_IN_TOP, LAYER_NOT_IN_CTX}
    if args.layers:
        layers = set(args.layers)
    targets = load_targets(TRIAGE, layers)
    if not targets:
        print(f"[2] 没有 primary ∈ {sorted(layers)} 的题", file=sys.stderr)
        return 2
    if args.limit:
        targets = targets[: args.limit]

    print(f"题 {len(targets)}  世代 {args.generation}  深池 {args.pool}  窗口 ±{args.window}")
    loaded = pipeline.load_generation(args.generation)
    pipe = pipeline.build_pipeline(loaded, top_k=EVAL_TOP_K, graph_depth=EVAL_GRAPH_DEPTH)
    retriever = pipe["retriever"]
    entities = retriever.graph.get("entities") or []
    alias_map = retriever.graph.get("alias_index")

    per_question: list[dict] = []
    rescued = {d: 0 for d in RERANK_DIMS}
    broken = {d: 0 for d in RERANK_DIMS}
    lost_questions = 0

    for i, tgt in enumerate(targets, 1):
        qid = tgt["question_id"]
        qa = query_mod.analyze_query(
            tgt["question"],
            entities=entities,
            aliases=alias_map,
            vocab=vocab,
            context=getattr(retriever, "query_context", None),
        )
        found = deep_search(retriever, qa, args.pool)
        hits = found.get("hits") or []
        if not hits:
            lost_questions += 1
            continue

        gold_ids = [str(c) for c in (tgt.get("gold_chunk_ids") or [])]
        order = [h["chunk_id"] for h in hits]
        rank_of = {cid: idx + 1 for idx, cid in enumerate(order)}
        gold_ranks = {g: rank_of[g] for g in gold_ids if g in rank_of}
        if not gold_ranks:
            # gold 不在池里 —— 不属于本脚本的目标层，跳过并记账
            lost_questions += 1
            continue
        best_gold = min(gold_ranks, key=lambda g: gold_ranks[g])

        bm25_raw = bm25_lookup(retriever, qa, args.pool)
        bm25_pool = [bm25_raw.get(c, 0.0) for c in order]
        score_by_id = {h["chunk_id"]: float(h["score"]) for h in hits}
        merged_pool = [score_by_id[c] for c in order]

        # 竞争者窗口：gold 名次前后各 WINDOW 个（不含 gold 自己）
        best_rank = gold_ranks[best_gold]
        lo = max(1, best_rank - args.window)
        hi = min(len(order), best_rank + args.window)
        window_ids = [order[k - 1] for k in range(lo, hi + 1) if order[k - 1] not in gold_ids]

        def feats(cid: str) -> dict:
            rec = retriever.chunks.get(cid) or {}
            return chunk_features(
                retriever,
                rec=rec,
                qa=qa,
                gold_rel=str(tgt.get("gold_answer") or ""),
                bm25_raw=bm25_raw.get(cid, 0.0),
                bm25_pool=bm25_pool,
                merged_score=score_by_id.get(cid, 0.0),
                merged_pool=merged_pool,
                rank=rank_of.get(cid, 0),
                pool_size=len(order),
            )

        # 整池每块的特征只算一遍，下游 gold / 竞争者 / 重排全部查这张表，
        # 保证三条路径口径完全一致（不存在「比较时算一套、重排时算另一套」）。
        # 若按需重算，12 个重排维度 × 200 块 × 136 题 ≈ 32 万次 `chunk_features`。
        pool_feats = {cid: feats(cid) for cid in order}
        gold_feats = {g: pool_feats[g] for g in gold_ranks if g in pool_feats}
        comp_feats = {c: pool_feats[c] for c in window_ids if c in pool_feats}

        # ---- 表 1：gold 在哪一维输给最强对手 ----
        worse: dict[str, bool] = {}
        for dim in COMPARE_DIMS:
            g_vals = [f[dim] for f in gold_feats.values()]
            c_vals = [f[dim] for f in comp_feats.values()]
            g_best = max(g_vals) if g_vals else 0
            c_best = max(c_vals) if c_vals else 0
            worse[dim] = bool(c_vals) and c_best > g_best

        # ---- 表 2：单维重排的离线回放（上界估计）----
        single = {}
        baseline_ok = best_rank <= EVAL_TOP_K
        for name, sign in RERANK_DIMS.items():
            if name == "chars_asc":
                key = lambda cid: (-int(pool_feats[cid]["chars"]), rank_of[cid])  # noqa: E731
            else:
                key = lambda cid, n=name, s=sign: (-s * float(pool_feats[cid][n]), rank_of[cid])  # noqa: E731
            reordered = sorted(order, key=key)
            top = set(reordered[:EVAL_TOP_K])
            ok = any(g in top for g in gold_ids)
            single[name] = {"gold_in_top": ok, "delta_vs_baseline": (1 if ok else 0) - (1 if baseline_ok else 0)}
            if ok and not baseline_ok:
                rescued[name] += 1
            if baseline_ok and not ok:
                broken[name] += 1

        per_question.append({
            "question_id": qid,
            "category": tgt.get("category"),
            "primary": tgt.get("primary"),
            "question": tgt.get("question"),
            "gold_answer": tgt.get("gold_answer"),
            "gold_ranks": gold_ranks,
            "best_gold_rank": best_rank,
            "pool_size": len(order),
            "baseline_in_top": baseline_ok,
            "window_ids": window_ids,
            "gold_features": gold_feats,
            "competitor_features": comp_feats,
            "worse_than_best_competitor": worse,
            "relation_hints": [str(h.get("canonical") or h.get("raw") or "") for h in (qa.get("relation_hints") or [])],
            "single_dim_rerank": single,
        })

        if i % 20 == 0:
            print(f"  ... {i}/{len(targets)}  {time.time() - t0:.0f}s")

    if not per_question:
        print("[4] 自检失败：没有产生任何逐题结果", file=sys.stderr)
        return 4

    # ---- 汇总 ----
    n = len(per_question)
    worse_dist = {dim: sum(1 for r in per_question if r["worse_than_best_competitor"][dim]) for dim in COMPARE_DIMS}
    baseline_top = sum(1 for r in per_question if r["baseline_in_top"])

    by_category: dict[str, dict] = {}
    for r in per_question:
        c = str(r["category"])
        slot = by_category.setdefault(c, {"n": 0, "baseline_in_top": 0,
                                          "worse": {d: 0 for d in COMPARE_DIMS},
                                          "rescued": {d: 0 for d in RERANK_DIMS}})
        slot["n"] += 1
        slot["baseline_in_top"] += 1 if r["baseline_in_top"] else 0
        for d in COMPARE_DIMS:
            slot["worse"][d] += 1 if r["worse_than_best_competitor"][d] else 0
        for d in RERANK_DIMS:
            if r["single_dim_rerank"][d]["delta_vs_baseline"] > 0:
                slot["rescued"][d] += 1

    summary = {
        "script": "scripts/a_feature_inventory.py",
        "purpose": "gold 块与竞争者可解释特征盘点（纯测量；单维重排为上界估计，非收益预测）",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "command": "py -3 scripts/a_feature_inventory.py",
        "generation": args.generation,
        "gold_set": config.rel_to_root(GOLD_SET),
        "triage_source": config.rel_to_root(TRIAGE),
        "layers": sorted(layers),
        "pool": args.pool,
        "window": args.window,
        "eval_top_k": EVAL_TOP_K,
        "n_questions": n,
        "n_skipped_gold_not_in_pool": lost_questions,
        "baseline_in_top": baseline_top,
        "gold_worse_than_best_competitor": worse_dist,
        "single_dim_rerank": {
            d: {"rescued": rescued[d], "broken": broken[d], "net": rescued[d] - broken[d]}
            for d in RERANK_DIMS
        },
        "by_category": by_category,
        "questions": per_question,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n=== 汇总（n={n}，baseline 已在 top20 的 {baseline_top} 题）===")
    print("\n--- 表 1：gold 严格弱于窗口内最强对手的题数 ---")
    for d in sorted(worse_dist, key=lambda x: -worse_dist[x]):
        print(f"  {d:14s} {worse_dist[d]:4d}/{n}  {worse_dist[d] / n:.3f}")
    print("\n--- 表 2：单维重排（离线回放，上界估计）---")
    print(f"  {'维度':14s} {'救回':>5s} {'打坏':>5s} {'净':>5s}")
    for d in sorted(RERANK_DIMS, key=lambda x: -(rescued[x] - broken[x])):
        print(f"  {d:14s} {rescued[d]:5d} {broken[d]:5d} {rescued[d] - broken[d]:+5d}")
    print(f"\n写出 {config.rel_to_root(out_json)}  {time.time() - t0:.0f}s")
    return 0


def _git_commit() -> str:
    """当前 HEAD（只读）。取不到就写 unknown，不让报告因为缺 git 而失败。"""
    try:
        import subprocess
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
            capture_output=True, text=True, timeout=10,
            env=subproc.child_env(),
        )
        return (out.stdout or "").strip() or "unknown"
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generation", default=CAND)
    ap.add_argument("--pool", type=int, default=200, help="深池深度（默认 200，对齐归因脚本）")
    ap.add_argument("--window", type=int, default=5, help="gold 前后各取几个竞争者")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（冒烟用）")
    ap.add_argument("--layers", nargs="*", default=None,
                    help=f"只测这些 primary 层（默认 {'/'.join([LAYER_NOT_IN_TOP, LAYER_NOT_IN_CTX])}）")
    ap.add_argument("--out-json", default=str(ROOT / "data" / "a_feature_inventory.json"))
    args = ap.parse_args(argv)
    return inventory(args)


if __name__ == "__main__":
    raise SystemExit(main())
