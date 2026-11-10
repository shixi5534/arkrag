"""W0 · 测量 24000 档的「名次 → 上下文」转化（**只读测量，不改任何实现**）。

## 为什么需要这个文件

`data/diag_context_positions.json` 的 `budget_arms` 只到 8000 / 12000 两档，
而生产档跑的是 **24000**（均 19.5 个单元）。W1 要定的 cap 值直接依赖
"24000 档下 rank>10 的 gold 还有多少能进上下文"这张表，而这张表**不存在**。

`tmp_prod_*.json` 也补不了这个缺口：那里每题只留 `answer / judge / citations_ok`
等答案级字段，没有 `hit_rank` / `n_units` / `ctx_tokens` / 单元列表，
无法事后复算名次与上下文的对应关系。

## 这个脚本与 `diag_context_positions.py` 的关系

后者是 6000 档的模板，**不能直接复用**：它写死 `TOKEN_BUDGET = 6000`、
不走 `quote_locate` 前插、不区分桥接题的 `bridge_search_top_k=40`。
照抄它测出来的是**另一条链路**。

本文件严格照 `pipeline.ask_once(options=prod_options())` 的调用顺序走：

1. `analyze_query`
2. `retriever.search(qa, top_k=40 if 桥接题 else 20)` ← `bridge_search_top_k`
3. `quote_locate.locate` **前插**（分数 `max(hits)+1.0`）← `quote_locate`
4. `retriever.expand`
5. `retriever.build_context(..., graph_reserve_tokens=12000)`（仅桥接题）
6. 预算 24000

关键点：**名次是在前插之后的 `found["hits"]` 上算的**。前插会改变名次，
若在后插之前的列表上算 rank，测出的就不是生产链路上真实生效的名次。

## 产出

`data/diag_rank_24000.json`：

* `schema` / `generation` / `config_digest` / `run_digest` / `baseline_sha`
* `summary.<split>`：四个分母（全部输入 / 有 gold / 应答题 / gold 在 hits）
  + 逐名次 × 进上下文表 + 累计 presence + 单元 token 中位数 + 直接/纯图单元数
* `per_question`：每题的 `status` 与明细。**无 gold / gold 不在 hits /
  不可判分题一律保留**，用 `status` 区分，绝不静默从分母删除。

## 用法

    # 先小样本验证熔断与链路（3 题）
    py -3.14 scripts/run_guarded.py -- py -3.14 scripts/diag_rank_24000.py --limit 3

    # 全量（需要授权）
    py -3.14 scripts/run_guarded.py -- py -3.14 scripts/diag_rank_24000.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import pipeline as pipeline_mod  # noqa: E402
from arkrag import query as query_mod  # noqa: E402
from arkrag import quote_locate  # noqa: E402
from arkrag import retrieve as retrieve_mod  # noqa: E402
from arkrag import vocab  # noqa: E402

GENERATION = "g-20261103T132152Z-06a741"
GOLD_SET = ROOT / "data" / "goldset" / "gold-v2.json"
#: gold-v1 的题号集合。`clean` 分区**不是** gold-v2 里的一个 `split` 值，而是
#: "holdout 里且不在 gold-v1 里的题"（见 `scripts/eval_answer_quality.py::pick_items`）。
#: 不读这个文件，`clean` 分区就会是空的——本次实测踩过这个坑。
GOLD_V1 = ROOT / "data" / "goldset" / "gold-v1.json"
DEFAULT_OUT = ROOT / "data" / "diag_rank_24000.json"

#: 生产档口径：**只从 `PROD_PROFILE` 读，不在这里手抄一份**。
#: 抄一份就是第二个事实来源，两者迟早不一致——本项目已因这个形态出过三次事故。
BUDGET = int(pipeline_mod.PROD_PROFILE["max_context_tokens"])
OPTIONS = pipeline_mod.prod_options()

SPLITS = ("dev", "holdout", "clean")

#: 规范化 per_question 时要剔掉的字段（时间戳不参与 digest）。
_VOLATILE = {"elapsed_ms", "generated_at"}


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _digest(obj) -> str:
    """规范化 digest：`sort_keys` + 紧凑分隔符 + UTF-8。"""
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _canonical_rows(rows: list[dict]) -> list[dict]:
    """给 digest 用的规范形态：按 question_id 排序、剔掉易变字段。"""
    out = []
    for r in rows:
        out.append({k: v for k, v in r.items() if k not in _VOLATILE})
    return sorted(out, key=lambda r: (r.get("split") or "", r.get("question_id") or ""))


def read_gold() -> list[dict]:
    doc = json.loads(GOLD_SET.read_text(encoding="utf-8"))
    items = doc["items"] if isinstance(doc, dict) else doc
    return list(items or [])


def load_v1_ids() -> set[str]:
    """gold-v1 的题号集合。`clean` 分区的定义依赖它。"""
    if not GOLD_V1.exists():
        return set()
    doc = json.loads(GOLD_V1.read_text(encoding="utf-8"))
    items = doc.get("items") if isinstance(doc, dict) else doc
    return {str(it.get("question_id")) for it in items or []}


def pick_items(items: list[dict], split: str, v1_ids: set[str]) -> list[dict]:
    """按 split 选题。**逐字对齐 `scripts/eval_answer_quality.py::pick_items`。**

    为什么要抄这一份而不是各写各的：口径一旦不同，W0 的 `clean` 与台账里
    97/120 那个数就不是同一个分母，两张表无法并列。这里有三个真实陷阱：

    * `clean` **不是** `gold-v2` 里的一个 `split` 值。它是「holdout 里、有 gold、
      且题号不在 gold-v1 里」的题。按 `split == "clean"` 过滤会得到 0 题——
      本次首跑就是这么空的。
    * 每类都要求 `gold_chunk_ids` 非空。结构类题（`time_conflict` /
      `negative_abstain` / `adversarial`）没有 gold，本来就不进分母。
    * 因此 `有 gold` 这个分母**本来就等于**评测脚本的选题结果，
      不该再用"全部输入"当分母。
    """
    if split == "all":
        return [it for it in items if it.get("gold_chunk_ids")]
    if split == "clean":
        return [it for it in items
                if it.get("split") == "holdout" and it.get("gold_chunk_ids")
                and str(it.get("question_id")) not in v1_ids]
    return [it for it in items
            if it.get("split") == split and it.get("gold_chunk_ids")]
    """名次分桶。与 `diag_context_positions.json` 的口径**同名同义**（A1 要求）。"""
    if rank is None:
        return "无 gold 在 hits"
    if rank <= 1:
        return "1"
    if rank <= 3:
        return "2-3"
    if rank <= 5:
        return "4-5"
    if rank <= 10:
        return "6-10"
    return "11+"


def _pct(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def rank_bucket(rank: int | None) -> str:
    """名次分桶。与 `diag_context_positions.json` 的 `c_ctx_pos_dist` **同名同义**。

    `None` 单独成一桶（"无 gold 在 hits"），不并进 `11+`：
    前者是**检索层**没召回到，后者是**装配层**召回到了没装进去。混在一起，
    台账就无法区分这两种完全不同的失效。
    """
    if rank is None:
        return "无 gold 在 hits"
    if rank <= 1:
        return "1"
    if rank <= 3:
        return "2-3"
    if rank <= 5:
        return "4-5"
    if rank <= 10:
        return "6-10"
    return "11+"


def measure_one(pipe: dict, retriever, it: dict) -> dict:
    """跑一题，返回该题的完整测量记录。

    调用顺序**逐行对应** `pipeline.ask_once`，任何一处偏离都会让测量变成
    "另一条链路的读数"。见模块 docstring。
    """
    qid = str(it.get("question_id") or "")
    split = str(it.get("split") or "")
    category = str(it.get("category") or "")
    question = str(it.get("question") or "")
    gold = [str(c) for c in (it.get("gold_chunk_ids") or [])]
    should_abstain = bool(it.get("should_abstain"))

    base = {
        "question_id": qid, "split": split, "category": category,
        "question": question, "n_gold": len(gold),
        "gold_chunk_ids": gold, "should_abstain": should_abstain,
        # `selected_by` 记下**这道题被哪些分区选中**。`clean` ⊂ `holdout`，
        # 同一道题会同时属于两个分区；只记一个 `split` 字段就无法在 clean 下
        # 统计它，那正是首跑 clean 分区为空的原因。
        "selected_by": str(it.get("_selected_by") or split),
    }

    # ---- 结构性拒答题（时间冲突 / 负向 / 对抗）没有 gold，必须显式区分 ----
    if not gold:
        base.update({
            "status": "no_gold",
            "hit_rank": None, "in_hits": False, "in_context": False,
            "ctx_pos": None, "n_units": None, "ctx_tokens": None,
            "gold_unit_tokens": None, "channels": [], "n_graph_only_units": None,
            "rank_bucket": "无 gold 在 hits",
        })
        return base

    qa = query_mod.analyze_query(
        question,
        entities=pipe["entities"],
        aliases=retriever.graph.get("alias_index"),
        vocab=vocab,
        context=getattr(retriever, "query_context", None),
    )

    # ---- top_k：桥接题走 bridge_search_top_k（与 ask_once 同一判定） ----
    top_k = int(OPTIONS.get("search_top_k") or 0)
    if not top_k:
        top_k = pipe["top_k"]
        wide = int(OPTIONS.get("bridge_search_top_k") or 0)
        if wide and OPTIONS.get("bridge") and _is_bridge(question):
            top_k = wide
    found = retriever.search(qa, top_k=top_k)

    # ---- quote_locate 前插（分数 max(hits)+1.0）----
    quote_ids: list[str] = []
    if OPTIONS.get("quote_locate"):
        loc = quote_locate.locate(question, retriever.chunks)
        if loc["chunk_ids"]:
            top = max((float(h.get("score") or 0.0) for h in found["hits"]), default=0.0)
            front = [
                {"chunk_id": cid, "score": top + 1.0, "cooc_boost": 0.0,
                 "sources": [{"channel": retrieve_mod.SOURCE_BM25, "score": 1.0}]}
                for cid in loc["chunk_ids"]
            ]
            seen = {h["chunk_id"] for h in found["hits"]}
            found["hits"] = [h for h in front if h["chunk_id"] not in seen] + list(found["hits"])
            quote_ids = list(loc["chunk_ids"])

    # ---- 名次必须在**前插之后**算 ----
    hits = found["hits"]
    hit_ids = [str(h.get("chunk_id") or "") for h in hits]
    hit_rank = {cid: j + 1 for j, cid in enumerate(hit_ids) if cid}

    seed_ids = [c["entity_id"] for c in qa.get("entity_candidates") or []]
    expanded = retriever.expand(
        seed_ids,
        depth=pipe["graph_depth"],
        query_analysis=qa,
        seed_chunk_ids=[h["chunk_id"] for h in found["hits"]],
    )

    # ---- graph_reserve_tokens 只对桥接题生效（与 ask_once 同一门控）----
    reserve = int(OPTIONS.get("graph_reserve_tokens") or 0)
    if reserve and not _is_bridge(question):
        reserve = 0
    ctx_kwargs = {"graph_reserve_tokens": reserve} if reserve else {}
    context = retriever.build_context(
        found["hits"], expanded["expansions"], query_analysis=qa, **ctx_kwargs)

    units = context.get("units") or []
    stats = context.get("stats") or {}
    ctx_ids = [str(u.get("chunk_id") or "") for u in units]
    ctx_pos = {cid: j + 1 for j, cid in enumerate(ctx_ids) if cid}

    gold_in_hits = [g for g in gold if g in hit_rank]
    gold_in_ctx = [g for g in gold if g in ctx_pos]
    best_rank = min((hit_rank[g] for g in gold_in_hits), default=None)
    best_ctx = min((ctx_pos[g] for g in gold_in_ctx), default=None)

    # gold 单元自身 token / 通道：cap 的直接依据
    gold_units = [u for u in units if str(u.get("chunk_id") or "") in set(gold)]
    gold_tokens = [int(u.get("tokens") or 0) for u in gold_units]
    gold_channels = sorted({
        str(s.get("channel") or "")
        for u in gold_units for s in (u.get("sources") or [])
    })

    if not gold_in_hits:
        status = "gold_not_in_hits"
    elif not gold_in_ctx:
        status = "gold_in_hits_not_in_context"
    else:
        status = "ok"

    base.update({
        "status": status,
        "hit_rank": best_rank,
        "in_hits": bool(gold_in_hits),
        "in_context": bool(gold_in_ctx),
        "ctx_pos": best_ctx,
        "n_units": int(stats.get("units") or 0),
        "ctx_tokens": int(stats.get("tokens_used") or 0),
        "token_budget": int(stats.get("token_budget") or BUDGET),
        "n_candidates": int(stats.get("candidates") or 0),
        "n_hits": len(hit_ids),
        "gold_unit_tokens": int(statistics.median(gold_tokens)) if gold_tokens else None,
        "channels": gold_channels,
        "n_direct_units": int(stats.get("direct_units") or 0),
        "n_graph_only_units": int(stats.get("graph_only_units") or 0),
        "n_truncated": len(stats.get("truncated_chunks") or ()),
        "any_truncated": bool(stats.get("truncated_chunks")),
        "rank_bucket": rank_bucket(best_rank),
        "quote_locate_hits": quote_ids,
        "reserve_applied": reserve,
        "graph_reserve_tokens": int(stats.get("graph_reserve_tokens") or 0),
    })
    return base


def _is_bridge(question: str) -> bool:
    from arkrag import bridge
    return bridge.is_bridge_question(question)


def summarize(rows: list[dict], split: str) -> dict:
    """一个 split 的汇总。**分母逐个报出，不许只给一个比率。**

    `clean` ⊂ `holdout`，所以按 `selected_by` 筛而不是按 `split` 字段筛——
    同一道题可以同时属于两个分区。
    """
    sub = [r for r in rows if split in str(r.get("selected_by") or "")]
    n_all = len(sub)
    with_gold = [r for r in sub if int(r.get("n_gold") or 0) > 0]
    scorable = [r for r in with_gold if not r.get("should_abstain")]
    in_hits = [r for r in with_gold if r.get("in_hits")]
    in_ctx = [r for r in with_gold if r.get("in_context")]

    # 逐名次 × 进上下文（分母 = 该名次区间的题数，按本题 hits 实测）
    buckets: dict[str, dict] = {}
    for r in with_gold:
        b = r.get("rank_bucket") or "无 gold 在 hits"
        ent = buckets.setdefault(b, {"bucket": b, "n": 0, "in_context": 0})
        ent["n"] += 1
        if r.get("in_context"):
            ent["in_context"] += 1
    order = ["1", "2-3", "4-5", "6-10", "11+", "无 gold 在 hits"]
    bucket_rows = []
    for b in order:
        if b in buckets:
            ent = buckets[b]
            ent["rate"] = _pct(ent["in_context"], ent["n"])
            bucket_rows.append(ent)

    # 累计 presence：gold 名次 <= k 的题里，有多少进了上下文
    cumulative = []
    for k in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 40):
        reach = [r for r in with_gold
                 if r.get("hit_rank") is not None and r["hit_rank"] <= k]
        got = [r for r in reach if r.get("in_context")]
        cumulative.append({
            "k": k,
            "gold_rank_le_k": len(reach),
            "in_context": len(got),
            "rate_within_k": _pct(len(got), len(reach)),
        })

    # 关键量：rank>10 的题在 24000 档还有多少进上下文（反证条件 2 的分子）
    over10 = [r for r in with_gold
              if r.get("hit_rank") is not None and r["hit_rank"] > 10]
    over10_in = [r for r in over10 if r.get("in_context")]

    gold_tokens = [int(r["gold_unit_tokens"]) for r in with_gold
                   if r.get("gold_unit_tokens") is not None]
    unit_counts = [int(r["n_units"]) for r in in_hits if r.get("n_units") is not None]
    ctx_tokens = [int(r["ctx_tokens"]) for r in in_hits if r.get("ctx_tokens") is not None]
    direct_counts = [int(r["n_direct_units"]) for r in in_hits
                     if r.get("n_direct_units") is not None]
    graph_counts = [int(r["n_graph_only_units"]) for r in in_hits
                    if r.get("n_graph_only_units") is not None]

    def _med(vals: list[int]) -> int | None:
        return int(statistics.median(vals)) if vals else None

    def _mean(vals: list[int]) -> float | None:
        return round(sum(vals) / len(vals), 2) if vals else None

    return {
        "split": split,
        "denominators": {
            "全部输入": n_all,
            "有 gold": len(with_gold),
            "应答题（有 gold 且非结构类）": len(scorable),
            "gold 在 hits": len(in_hits),
            "gold 进上下文": len(in_ctx),
        },
        "rates": {
            "gold_in_hits_rate": _pct(len(in_hits), len(with_gold)),
            "in_context_rate": _pct(len(in_ctx), len(with_gold)),
        },
        "status_counts": {
            s: sum(1 for r in sub if r.get("status") == s)
            for s in ("ok", "gold_in_hits_not_in_context",
                      "gold_not_in_hits", "no_gold")
        },
        "by_rank_bucket": bucket_rows,
        "cumulative_presence": cumulative,
        "rank_gt_10": {
            "n": len(over10),
            "in_context": len(over10_in),
            "rate": _pct(len(over10_in), len(over10)),
            "question_ids": [r["question_id"] for r in over10_in],
            "note": "W0 反证条件 2 的分子：24000 档下 rank>10 仍进上下文的题数",
        },
        "units": {
            "n_units_median": _med(unit_counts),
            "n_units_mean": _mean(unit_counts),
            "n_units_min": min(unit_counts) if unit_counts else None,
            "n_units_max": max(unit_counts) if unit_counts else None,
            "ctx_tokens_median": _med(ctx_tokens),
            "ctx_tokens_mean": _mean(ctx_tokens),
            "gold_unit_tokens_median": _med(gold_tokens),
            "n_direct_units_mean": _mean(direct_counts),
            "n_graph_only_units_mean": _mean(graph_counts),
        },
        "by_category": _by_category(with_gold),
    }


def _by_category(with_gold: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for r in with_gold:
        cat = r.get("category") or ""
        ent = out.setdefault(cat, {"n": 0, "in_hits": 0, "in_context": 0})
        ent["n"] += 1
        if r.get("in_hits"):
            ent["in_hits"] += 1
        if r.get("in_context"):
            ent["in_context"] += 1
    for ent in out.values():
        ent["in_context_rate"] = _pct(ent["in_context"], ent["n"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="W0：24000 档名次→上下文转化测量")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0 = 全部）")
    ap.add_argument("--splits", default="dev,holdout,clean",
                    help="逗号分隔的 split 列表")
    ap.add_argument("--baseline", default="",
                    help="基线产物 SHA（默认自动取 tmp_prod_*.json 的 sha）")
    args = ap.parse_args(argv)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    v1_ids = load_v1_ids()
    all_items = read_gold()

    # 按 split 选题，**逐 split 调 `pick_items`**（口径与评测脚本一致）。
    # `clean` 与 `holdout` 有重叠（clean ⊂ holdout），所以不能简单按 `split`
    # 字段过滤再拼起来——那会让 holdout 的题在 clean 下重复计入，
    # 也让 clean 恒为空。这里给每道题**打上它被哪些分区选中**的标记。
    picked: list[dict] = []
    for sp in splits:
        for it in pick_items(all_items, sp, v1_ids):
            rec = dict(it)
            rec["_selected_by"] = sp
            picked.append(rec)
    # `clean` 是 holdout 的子集：同题会命中两个分区。按 (split, question_id) 去重，
    # 但把"被哪些分区选中"合并进一条记录，避免重复跑同一道题。
    merged: dict[str, dict] = {}
    for rec in picked:
        qid = str(rec.get("question_id") or "")
        if qid in merged:
            merged[qid]["_selected_by"] = ",".join(
                sorted({merged[qid]["_selected_by"], rec["_selected_by"]}))
            continue
        merged[qid] = rec

    items = list(merged.values())
    # 稳定顺序：dev → holdout → clean，同分区内按 gold 原序
    order = {sp: i for i, sp in enumerate(splits)}
    items.sort(key=lambda r: (order.get(str(r.get("split") or ""), 99),
                              str(r.get("question_id") or "")))
    if args.limit:
        items = items[: args.limit]

    print(f"[w0] 计划 {len(items)} 题，split={splits}，预算={BUDGET}", flush=True)
    print(f"[w0] 各分区选题数="
          f"{ {sp: sum(1 for r in items if sp in (r.get('_selected_by') or '')) for sp in splits} }",
          flush=True)
    print(f"[w0] options={json.dumps(OPTIONS, ensure_ascii=False)}", flush=True)

    t0 = time.time()
    loaded = pipeline_mod.load_generation(GENERATION)
    pipe = pipeline_mod.build_pipeline(
        loaded, top_k=20, graph_depth=1)
    pipe["generation_id"] = GENERATION
    pipe["entities"] = loaded["graph"].get("entities") or []
    retriever = pipe["retriever"]
    # 生产档预算落实例属性（与 apply_prod_profile 同一动作，但不改 DEFAULT_BUDGET）
    retriever.budget["max_context_tokens"] = BUDGET
    assert retriever.budget["max_context_tokens"] == BUDGET
    print(f"[load] {time.time() - t0:.1f}s chunks={len(loaded['chunks'])} "
          f"assertions={len(loaded['assertions'])}", flush=True)

    rows: list[dict] = []
    t1 = time.time()
    for i, it in enumerate(items, 1):
        try:
            rows.append(measure_one(pipe, retriever, it))
        except Exception as exc:  # noqa: BLE001 - 单题失败必须留痕，不许丢分母
            rows.append({
                "question_id": str(it.get("question_id") or ""),
                "split": str(it.get("split") or ""),
                "category": str(it.get("category") or ""),
                "question": str(it.get("question") or ""),
                "selected_by": str(it.get("_selected_by") or it.get("split") or ""),
                "status": "run_error",
                "error": f"{type(exc).__name__}: {exc}",
            })
        if i % 50 == 0:
            print(f"[w0] {i}/{len(items)} {time.time() - t1:.1f}s", flush=True)

    elapsed = time.time() - t1
    print(f"[w0] 完成 {len(rows)} 题，{elapsed:.1f}s", flush=True)

    baseline_sha = args.baseline
    if not baseline_sha:
        parts = {}
        for sp in splits:
            p = ROOT / f"tmp_prod_{sp}.json"
            if p.is_file():
                parts[sp] = _sha_file(p)
        baseline_sha = _digest(parts) if parts else ""

    payload = {
        "schema": "arkrag.diag_rank_24000/1",
        "purpose": "W0：测量 24000 档「名次 → 上下文」转化，为 W1 的 cap 值提供依据",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": platform.python_version(),
        "os": platform.platform(),
        "command": "py -3.14 scripts/run_guarded.py -- py -3.14 scripts/diag_rank_24000.py",
        "generation": GENERATION,
        "baseline_sha": baseline_sha,
        "config_digest": _digest({"budget": BUDGET, "options": OPTIONS,
                                  "top_k": 20, "graph_depth": 1}),
        "gold_set": str(GOLD_SET.relative_to(ROOT)),
        "gold_set_sha": _sha_file(GOLD_SET),
        "splits": splits,
        "elapsed_ms": round(elapsed * 1000.0, 1),
        "n_per_question": len(rows),
        "summary": {sp: summarize(rows, sp) for sp in splits},
        "per_question": _canonical_rows(rows),
    }
    payload["run_digest"] = _digest({
        "config_digest": payload["config_digest"],
        "per_question": payload["per_question"],
    })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")
    print(f"[w0] -> {out}")
    print(f"[w0] run_digest={payload['run_digest']}")
    for sp in splits:
        s = payload["summary"][sp]
        d = s["denominators"]
        u = s["units"]
        print(f"[w0] {sp}: 有gold {d['有 gold']} / 在hits {d['gold 在 hits']}"
              f" / 进ctx {d['gold 进上下文']}"
              f" | n_units 均 {u['n_units_mean']} 中位 {u['n_units_median']}"
              f" | ctx_tokens 均 {u['ctx_tokens_mean']}"
              f" | rank>10 进ctx {s['rank_gt_10']['in_context']}"
              f"/{s['rank_gt_10']['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
