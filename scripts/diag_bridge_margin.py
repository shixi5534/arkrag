"""W2.1 前置：桥接候选 margin 的真实样本落盘 + 校准表生成。

## 为什么必须由主会话跑

W2 的实现者（子代理）交付了 `bridge_calib.fit()` 与决策层，但**校准本身需要跑
全量世代**——一次 `load_generation()` 常驻 1.0–1.5 GB，且这台机器已因并发跑
这类进程崩过三次。所以：代码归子代理，数据归主会话，串行跑。

## 这个脚本做什么

对每道 `multihop` 题，取到桥接候选集后，为每个候选记一行样本：

    (support_sum, assert_count, degree, n_cands, raw_margin, is_gold_top1, x_is_gold)

* `support_sum` —— 该候选连接两锚点的**接地断言条数**（去重 `(s,p,o)`）。
  **口径由 `bridge.candidate_support()` 唯一决定**，本脚本不另算一套：
  W2 的实现者明确警告过，"我这边算断言条数、你那边算图边权重"会让桶对不上、
  决策大片落进 `bucket_missing`。这里直接调它的函数。
* `raw_margin` —— `support_sum(top1) - support_sum(top2)`
* `is_gold_top1` —— top-1 候选是否就是 gold 那个中间实体
* `x_is_gold` —— 每个候选各自是否 gold（用于算"指定 top-1 时对不对"）

然后调 `bridge_calib.fit()` 生成单调映射，落盘成 `data/bridge_calib.json`。

## gold 中间实体怎么定

`multihop` 题的 gold 是**块**，不是实体。判据要求"一句同时含两个锚点且含 gold 内容"。
这里用可机检的代理：**候选 x 是否出现在任一 gold 块的正文里**。
这是代理而非真值——写进产物，并在结论文档里标为代理口径。

## 用法

    python scripts/run_guarded.py -- python scripts/diag_bridge_margin.py
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

from arkrag import bridge as bridge_mod  # noqa: E402
from arkrag import bridge_calib  # noqa: E402
from arkrag import pipeline as pipeline_mod  # noqa: E402
from arkrag import query as query_mod  # noqa: E402
from arkrag import quote_locate  # noqa: E402
from arkrag import retrieve as retrieve_mod  # noqa: E402
from arkrag import vocab  # noqa: E402

GENERATION = "g-20261103T132152Z-06a741"
GOLD_SET = ROOT / "data" / "goldset" / "gold-v2.json"
GOLD_V1 = ROOT / "data" / "goldset" / "gold-v1.json"
DEFAULT_OUT = ROOT / "data" / "diag_bridge_margin.json"
CALIB_OUT = ROOT / "data" / "bridge_calib.json"

BUDGET = int(pipeline_mod.PROD_PROFILE["max_context_tokens"])
OPTIONS = pipeline_mod.prod_options()


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _digest(obj) -> str:
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_v1_ids() -> set[str]:
    if not GOLD_V1.exists():
        return set()
    doc = json.loads(GOLD_V1.read_text(encoding="utf-8"))
    items = doc.get("items") if isinstance(doc, dict) else doc
    return {str(it.get("question_id")) for it in items or []}


def pick_items(items: list[dict], split: str, v1_ids: set[str]) -> list[dict]:
    """逐字对齐 `scripts/eval_answer_quality.py::pick_items`。"""
    if split == "all":
        return [it for it in items if it.get("gold_chunk_ids")]
    if split == "clean":
        return [it for it in items
                if it.get("split") == "holdout" and it.get("gold_chunk_ids")
                and str(it.get("question_id")) not in v1_ids]
    return [it for it in items
            if it.get("split") == split and it.get("gold_chunk_ids")]


def measure_one(pipe: dict, retriever, it: dict) -> dict:
    """跑一题 multihop，返回候选样本。"""
    qid = str(it.get("question_id") or "")
    question = str(it.get("question") or "")
    gold_chunks = [str(c) for c in (it.get("gold_chunk_ids") or [])]

    qa = query_mod.analyze_query(
        question,
        entities=pipe["entities"],
        aliases=retriever.graph.get("alias_index"),
        vocab=vocab,
        context=getattr(retriever, "query_context", None),
    )
    top_k = int(OPTIONS.get("search_top_k") or 0)
    if not top_k:
        top_k = pipe["top_k"]
        wide = int(OPTIONS.get("bridge_search_top_k") or 0)
        if wide and OPTIONS.get("bridge") and bridge_mod.is_bridge_question(question):
            top_k = wide
    found = retriever.search(qa, top_k=top_k)

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

    seed_ids = [c["entity_id"] for c in qa.get("entity_candidates") or []]
    expanded = retriever.expand(
        seed_ids, depth=pipe["graph_depth"], query_analysis=qa,
        seed_chunk_ids=[h["chunk_id"] for h in found["hits"]],
    )
    reserve = int(OPTIONS.get("graph_reserve_tokens") or 0)
    if reserve and not bridge_mod.is_bridge_question(question):
        reserve = 0
    ctx_kwargs = {"graph_reserve_tokens": reserve} if reserve else {}
    context = retriever.build_context(
        found["hits"], expanded["expansions"], query_analysis=qa, **ctx_kwargs)
    units = context.get("units") or []

    base = {
        "question_id": qid,
        "split": str(it.get("split") or ""),
        "category": str(it.get("category") or ""),
        "question": question,
        "selected_by": str(it.get("_selected_by") or it.get("split") or ""),
        "n_units": len(units),
        "n_gold_chunks": len(gold_chunks),
    }

    if not bridge_mod.is_bridge_question(question):
        base.update({"status": "not_bridge", "samples": []})
        return base

    # 用「上下文块的全量断言」建候选（与 ask_once 的 bridge 分支同一口径）。
    # 四步都调 `bridge` 的**共享构造器**，不在这里另写一套：
    # W2 的实现者明确警告过，"我这边算断言条数、你那边算图边权重"会让桶对不上、
    # 决策大片落进 bucket_missing。所以这里只用它导出的函数。
    anc = bridge_mod.anchors_of(question)
    # 注意：`_grounded_triples(units, chunk_assertions, display)` 的后两个是
    # **位置参数**（不是 keyword-only）。传成关键字会静默落进默认值 `None`，
    # 于是只用 `unit["assertions"]`（被谓词过滤过），候选几乎全空——
    # 本次实测踩过：5 题全部 `no_candidates`。
    tri = bridge_mod._grounded_triples(
        units, retriever.assertions_by_chunk.get, retriever.display_name)
    neigh = bridge_mod._neighbourhood(tri)
    cand = bridge_mod._common_neighbours(neigh, anc)

    if not cand:
        # 区分两种"没候选"：tri 空 = 上下文里根本没建出接地三元组；
        # tri 非空但 cand 空 = 两个锚点没有共同邻居。这两者的含义完全不同，
        # 混成一个 `no_candidates` 会让"检索不到"和"图里真没有这条路径"无法区分。
        st = "no_grounded_triples" if not tri else "no_common_neighbor"
        base.update({"status": st, "samples": [], "tri_size": len(tri),
                     "neigh_size": len(neigh)})
        return base

    cands = bridge_mod.rank_candidates(tri, neigh, anc, cand)

    if not cands:
        base.update({"status": "no_candidates", "samples": []})
        return base
    # gold 代理：候选实体名是否出现在任一 gold 块正文里
    gold_text = ""
    for cid in gold_chunks:
        rec = retriever.chunks.get(cid) or {}
        gold_text += "\n" + str(rec.get("text") or "")

    n_cands = len(cands)
    top1 = cands[0]
    top2 = cands[1] if n_cands > 1 else None
    s1 = float(top1.get("support_sum") or 0)
    s2 = float(top2.get("support_sum") or 0) if top2 else 0.0
    margin = s1 - s2

    # 特征压成计划书登记的四个（直接调 bridge_calib 的实现，不另算）
    feats = bridge_calib.features_of_ranked(cands)

    samples = []
    for i, c in enumerate(cands):
        x = str(c.get("x") or "")
        name = str(retriever.display_name(x) or x)
        samples.append({
            "rank": i + 1,
            "x": x,
            "name": name,
            "support_sum": float(c.get("support_sum") or 0),
            "assert_count": int(c.get("assert_count") or 0),
            "degree": int(c.get("degree") or 0),
            "n_cands": n_cands,
            "raw_margin": margin,
            "is_top1": i == 0,
            # 代理口径：名字出现在任一 gold 块正文里
            "x_is_gold": bool(name) and name in gold_text,
        })

    top1_is_gold = bool(samples[0]["x_is_gold"]) if samples else False
    base.update({
        "status": "ok",
        "n_cands": n_cands,
        "features": feats,
        "support_sum_top1": s1,
        "support_sum_top2": s2,
        "raw_margin": margin,
        "top1_is_gold": top1_is_gold,
        # 现行策略（候选唯一才答）在这题上会不会作答
        "unique_only_answers": n_cands == 1,
        "unique_only_correct": bool(n_cands == 1 and top1_is_gold),
        "samples": samples,
    })
    return base


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("status") == "ok"]
    n = len(ok)
    out: dict = {
        "n_questions": len(rows),
        "n_ok": n,
        "status_counts": {},
        "unique_only": {},
        "top1_by_support": {},
    }
    for st in ("ok", "not_bridge", "no_grounded_triples", "no_common_neighbor",
               "no_candidates", "run_error"):
        out["status_counts"][st] = sum(1 for r in rows if r.get("status") == st)
    if not n:
        return out

    uniq_ans = [r for r in ok if r.get("unique_only_answers")]
    uniq_ok = [r for r in uniq_ans if r.get("unique_only_correct")]
    out["unique_only"] = {
        "answers": len(uniq_ans),
        "answers_correct": len(uniq_ok),
        "rate_of_all": round(len(uniq_ans) / n, 4),
        "precision": round(len(uniq_ok) / len(uniq_ans), 4) if uniq_ans else None,
    }

    top1_ok = [r for r in ok if r.get("top1_is_gold")]
    out["top1_by_support"] = {
        "correct": len(top1_ok),
        "rate": round(len(top1_ok) / n, 4),
        "unique_rate": round(len(uniq_ans) / n, 4),
        "headroom_pt": round(100.0 * (len(top1_ok) - len(uniq_ok)) / n, 2),
    }

    # margin 分布
    margins = [float(r.get("raw_margin") or 0.0) for r in ok]
    out["margin_dist"] = {
        "median": round(statistics.median(margins), 3) if margins else None,
        "mean": round(sum(margins) / len(margins), 3) if margins else None,
        "min": min(margins) if margins else None,
        "max": max(margins) if margins else None,
        "n_zero": sum(1 for m in margins if m == 0),
    }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="W2.1：桥接 margin 样本落盘与校准")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--calib-out", default=str(CALIB_OUT))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--splits", default="dev,holdout,clean")
    args = ap.parse_args(argv)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    v1_ids = load_v1_ids()
    doc = json.loads(GOLD_SET.read_text(encoding="utf-8"))
    all_items = doc["items"] if isinstance(doc, dict) else doc

    picked: list[dict] = []
    for sp in splits:
        for it in pick_items(all_items, sp, v1_ids):
            if str(it.get("category") or "") != "multihop":
                continue
            rec = dict(it)
            rec["_selected_by"] = sp
            picked.append(rec)
    merged: dict[str, dict] = {}
    for rec in picked:
        qid = str(rec.get("question_id") or "")
        if qid in merged:
            merged[qid]["_selected_by"] = ",".join(
                sorted({merged[qid]["_selected_by"], rec["_selected_by"]}))
            continue
        merged[qid] = rec
    items = list(merged.values())
    if args.limit:
        items = items[: args.limit]

    print(f"[w2] 计划 {len(items)} 题 multihop，budget={BUDGET}", flush=True)

    t0 = time.time()
    loaded = pipeline_mod.load_generation(GENERATION)
    pipe = pipeline_mod.build_pipeline(loaded, top_k=20, graph_depth=1)
    pipe["generation_id"] = GENERATION
    pipe["entities"] = loaded["graph"].get("entities") or []
    retriever = pipe["retriever"]
    retriever.budget["max_context_tokens"] = BUDGET
    print(f"[load] {time.time() - t0:.1f}s", flush=True)

    rows: list[dict] = []
    t1 = time.time()
    for i, it in enumerate(items, 1):
        try:
            rows.append(measure_one(pipe, retriever, it))
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "question_id": str(it.get("question_id") or ""),
                "split": str(it.get("split") or ""),
                "category": str(it.get("category") or ""),
                "selected_by": str(it.get("_selected_by") or ""),
                "status": "run_error",
                "error": f"{type(exc).__name__}: {exc}",
            })
        if i % 20 == 0:
            print(f"[w2] {i}/{len(items)} {time.time() - t1:.1f}s", flush=True)

    elapsed = time.time() - t1
    print(f"[w2] 完成 {len(rows)} 题，{elapsed:.1f}s", flush=True)

    summary = summarize(rows)
    for sp in splits:
        sub = [r for r in rows if sp in str(r.get("selected_by") or "")]
        summary[f"split_{sp}"] = summarize(sub)

    # ---- 校准：dev 上拟合，holdout/clean 只报不参与拟合 ----
    # 这是计划书"dev 用于校准，holdout 只做冻结后的确认"的直接落地。
    fit_rows = [r for r in rows
                if r.get("status") == "ok" and "dev" in str(r.get("selected_by") or "")]
    samples_for_fit = []
    for r in fit_rows:
        # 每题只贡献一行：top-1 是否 gold。这是"给定候选集，top-1 对不对"的问题。
        # 字段名与类型必须严格符合 `bridge_calib._check_sample` 的约定：
        #   * 标签键是 `gold_is_top1`（不是 is_gold_top1）；
        #   * 三个特征必须是**整数**，且 `support_top1 - support_top2` 必须等于
        #     `raw_margin`（它自己会校验，不一致就抛）；
        #   * `support_top1 >= support_top2`（按 support 降序排过）。
        # 这里不做任何"凑数"，缺什么就让它抛——fail-closed 是对的设计。
        f = r.get("features") or {}
        samples_for_fit.append({
            "n_cands": int(f.get("n_cands") or 1),
            "support_top1": int(f.get("support_top1") or 0),
            "support_top2": int(f.get("support_top2") or 0),
            "raw_margin": int(f.get("raw_margin") or 0),
            "gold_is_top1": bool(r.get("top1_is_gold")),
        })

    calib_report: dict = {"n_fit_samples": len(samples_for_fit)}
    if len(samples_for_fit) >= 10:
        mapping = bridge_calib.fit(samples_for_fit, keys=("raw_margin",))
        # 每个桶的实测精度（用于判断 p > 2/3 区域是否真的有把握）
        bucket_stats = {}
        for s in samples_for_fit:
            p = bridge_calib.probability(mapping, s)
            b = "missing" if p is None else round(float(p), 3)
            e = bucket_stats.setdefault(str(b), {"n": 0, "gold": 0})
            e["n"] += 1
            if s["gold_is_top1"]:
                e["gold"] += 1
        for k, v in bucket_stats.items():
            v["rate"] = round(v["gold"] / v["n"], 4) if v["n"] else None
        calib_report.update({
            "mapping": mapping,
            "bucket_stats": bucket_stats,
            "threshold": bridge_calib.threshold(),
        })
        try:
            # `to_json` 返回**字符串**（不是 dict）。直接写它，别 json.dumps 第二层，
            # 否则落盘的是一个 JSON 字符串而不是映射对象，`load_mapping` 读不回来。
            Path(args.calib_out).write_text(
                bridge_calib.to_json(mapping) + "\n", encoding="utf-8")
            # 立刻验证真能读回：写进去读不回来等于没写。
            _back = bridge_calib.load_mapping(args.calib_out)
            calib_report["calib_roundtrip_ok"] = bool(_back)
            calib_report["calib_path"] = str(args.calib_out)
            print(f"[w2] 校准表 -> {args.calib_out}（往返读回 OK）")
        except Exception as exc:  # noqa: BLE001
            calib_report["calib_error"] = f"{type(exc).__name__}: {exc}"
            print(f"[w2] 校准表写入/读回失败：{exc}")
    else:
        calib_report["skipped"] = f"dev 可用样本 {len(samples_for_fit)} < 10"

    payload = {
        "schema": "arkrag.diag_bridge_margin/1",
        "purpose": "W2.1：桥接候选 margin 真实样本 + 校准表，判 M0 可行性",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": platform.python_version(),
        "os": platform.platform(),
        "command": "python scripts/run_guarded.py -- python scripts/diag_bridge_margin.py",
        "generation": GENERATION,
        "gold_set": str(GOLD_SET.relative_to(ROOT)),
        "gold_set_sha": _sha_file(GOLD_SET),
        "config_digest": _digest({"budget": BUDGET, "options": OPTIONS}),
        "elapsed_ms": round(elapsed * 1000.0, 1),
        "n_per_question": len(rows),
        "gold_proxy": "候选实体名出现在任一 gold 块正文里（代理口径，非真值）",
        "calibration": calib_report,
        "summary": summary,
        "per_question": sorted(rows, key=lambda r: (r.get("selected_by") or "",
                                                    r.get("question_id") or "")),
    }
    payload["run_digest"] = _digest({"config_digest": payload["config_digest"],
                                     "per_question": payload["per_question"]})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")
    print(f"[w2] -> {out}")
    print(f"[w2] run_digest={payload['run_digest']}")
    for sp in splits:
        s = summary.get(f"split_{sp}") or {}
        if s.get("n_ok"):
            print(f"[w2] {sp}: n={s['n_ok']} 唯一作答={s['unique_only']['answers']}"
                  f" 唯一正确={s['unique_only']['answers_correct']}"
                  f" top1正确={s['top1_by_support']['correct']}"
                  f" ({s['top1_by_support']['rate']})"
                  f" 余量={s['top1_by_support']['headroom_pt']}pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
