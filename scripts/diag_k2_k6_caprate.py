"""W1 的 K2/K6 + W3 的 cap 触发率：一次装配、三项计数。

## 为什么合成一个脚本

K2（`truncation_overtake`）、K6（桥接候选池不变）、W3 的 cap 触发率，
三者的共同点是：**都只需要在真实数据上跑一遍装配链路，然后数计数**。
分成三个脚本就要各加载一次 302 MB 产物（1.0–1.5 GB 常驻），三次是纯浪费。

## K2：`truncation_overtake`（cap 路径是否首超即停）

计划书 K2：

> cap 触发题中，被装入的直接单元集合等于"cap 前缀按顺序、首超预算即停止"的结果；
> `truncation_overtake=0`

判定：对每道 cap 触发的题，检查 `units` 里的直接单元序列是否**严格等于**
`ordered` 的直接单元前缀（直到第一个装不下的为止）。若出现"名次更差的单元
进了上下文、名次更好的反而没进"，就是一次 overtake。

**注意**：这里只对 **cap 路径**承诺归零。默认路径（cap=0）的 `continue` 语义
本来就会产生 overtake（台账 193/350 那道悬案），不属本项范围。

## K6：桥接候选池不变

对每道桥接题，比较 `build_context(max_direct_units=N)` 返回的
`bridge_candidates` 与 `build_context()` 返回的 `units` 的候选 ID 集合。

期望：`bridge_candidates` 的集合 **等于或包含** 无 cap 时的 `units` 集合
（因为 cap 前的内容就是完整接地集）。

## W3：cap 触发率（非退化门禁）

对每道 `cross_chunk` 题，数 `_chapter_answer` 找到的章节数是否 > 4。
若触发率为 0，说明 cap 从未生效 —— 那是"开关没接线"的形态，必须报出来。
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import answer as answer_mod  # noqa: E402
from arkrag import bridge as bridge_mod  # noqa: E402
from arkrag import pipeline as pipeline_mod  # noqa: E402
from arkrag import query as query_mod  # noqa: E402
from arkrag import quote_locate  # noqa: E402
from arkrag import retrieve as retrieve_mod  # noqa: E402
from arkrag import vocab  # noqa: E402

GEN = "g-20261103T132152Z-06a741"
GOLD = ROOT / "data" / "goldset" / "gold-v2.json"
GOLD_V1 = ROOT / "data" / "goldset" / "gold-v1.json"
OUT = ROOT / "data" / "diag_k2_k6_caprate.json"

BUDGET = int(pipeline_mod.PROD_PROFILE["max_context_tokens"])
OPTIONS = pipeline_mod.prod_options()
CAP_N = int(OPTIONS.get("chapter_cap") or 0)
DIRECT_CAP = 14  # W1 的候选值，用来造"cap 触发"的场景


def load_v1_ids():
    if not GOLD_V1.exists():
        return set()
    doc = json.loads(GOLD_V1.read_text(encoding="utf-8"))
    items = doc.get("items") if isinstance(doc, dict) else doc
    return {str(it.get("question_id")) for it in items or []}


def pick(items, split, v1_ids):
    if split == "all":
        return [i for i in items if i.get("gold_chunk_ids")]
    if split == "clean":
        return [i for i in items if i.get("split") == "holdout"
                and i.get("gold_chunk_ids")
                and str(i.get("question_id")) not in v1_ids]
    return [i for i in items if i.get("split") == split and i.get("gold_chunk_ids")]


def build_ctx(retriever, pipe, question, *, direct_cap=0):
    """照 `ask_once` 的顺序装配一次上下文。"""
    qa = query_mod.analyze_query(
        question, entities=pipe["entities"],
        aliases=retriever.graph.get("alias_index"),
        vocab=vocab,
        context=getattr(retriever, "query_context", None))
    top_k = pipe["top_k"]
    wide = int(OPTIONS.get("bridge_search_top_k") or 0)
    if wide and OPTIONS.get("bridge") and bridge_mod.is_bridge_question(question):
        top_k = wide
    found = retriever.search(qa, top_k=top_k)
    if OPTIONS.get("quote_locate"):
        loc = quote_locate.locate(question, retriever.chunks)
        if loc["chunk_ids"]:
            top = max((float(h.get("score") or 0.0) for h in found["hits"]), default=0.0)
            front = [{"chunk_id": cid, "score": top + 1.0, "cooc_boost": 0.0,
                      "sources": [{"channel": retrieve_mod.SOURCE_BM25, "score": 1.0}]}
                     for cid in loc["chunk_ids"]]
            seen = {h["chunk_id"] for h in found["hits"]}
            found["hits"] = [h for h in front if h["chunk_id"] not in seen] + list(found["hits"])
    seed = [c["entity_id"] for c in qa.get("entity_candidates") or []]
    exp = retriever.expand(seed, depth=pipe["graph_depth"], query_analysis=qa,
                           seed_chunk_ids=[h["chunk_id"] for h in found["hits"]])
    reserve = int(OPTIONS.get("graph_reserve_tokens") or 0)
    if reserve and not bridge_mod.is_bridge_question(question):
        reserve = 0
    kw = {"graph_reserve_tokens": reserve} if reserve else {}
    if direct_cap:
        kw["max_direct_units"] = direct_cap
    return retriever.build_context(found["hits"], exp["expansions"],
                                   query_analysis=qa, **kw)


def is_direct(u):
    return any(s.get("channel") in (retrieve_mod.SOURCE_BM25, retrieve_mod.SOURCE_ENTITY)
               for s in (u.get("sources") or ()))


def measure_k2(ctx_capped, *, retriever, pipe, question):
    """K2：cap 路径的直接单元是否严格是有序前缀。"""
    units = ctx_capped.get("units") or []
    kept_direct = [u for u in units if is_direct(u)]
    cand = ctx_capped.get("bridge_candidates") or []
    ordered_direct = [u for u in cand if is_direct(u)]
    # 期望：kept_direct 是 ordered_direct 的前缀
    prefix_ok = True
    for i, u in enumerate(kept_direct):
        if i >= len(ordered_direct):
            prefix_ok = False
            break
        if u.get("chunk_id") != ordered_direct[i].get("chunk_id"):
            prefix_ok = False
            break
    return {
        "n_kept_direct": len(kept_direct),
        "n_ordered_direct": len(ordered_direct),
        "prefix_ok": prefix_ok,
        "overtake": 0 if prefix_ok else 1,
    }


def measure_k6(ctx_capped, ctx_plain):
    """K6：cap 前的完整接地候选 vs 无 cap 的 units。"""
    cand = {str(u.get("chunk_id") or "")
            for u in (ctx_capped.get("bridge_candidates") or ())}
    plain = {str(u.get("chunk_id") or "") for u in (ctx_plain.get("units") or ())}
    return {
        "n_bridge_candidates": len(cand),
        "n_plain_units": len(plain),
        "plain_subset_of_candidates": plain <= cand,
        "missing_from_candidates": sorted(plain - cand),
        "extra_in_candidates": len(cand - plain),
    }


def measure_cap_rate(retriever, pipe, question):
    """W3：章节答案路径找到的章节数，判断 cap 是否触发。"""
    ctx = build_ctx(retriever, pipe, question)
    units = ctx.get("units") or []
    chapters = {str(u.get("chapter") or "") for u in units if u.get("chapter")}
    chapters.discard("")
    return {"n_chapters": len(chapters), "cap": CAP_N,
            "triggers": len(chapters) > CAP_N if CAP_N else False}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="dev,holdout,clean")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    v1 = load_v1_ids()
    doc = json.loads(GOLD.read_text(encoding="utf-8"))
    all_items = doc["items"] if isinstance(doc, dict) else doc

    picked = []
    for sp in splits:
        for it in pick(all_items, sp, v1):
            rec = dict(it)
            rec["_sel"] = sp
            picked.append(rec)
    merged = {}
    for r in picked:
        q = str(r.get("question_id") or "")
        if q in merged:
            merged[q]["_sel"] = ",".join(sorted({merged[q]["_sel"], r["_sel"]}))
        else:
            merged[q] = r
    items = list(merged.values())
    if args.limit:
        items = items[:args.limit]
    print(f"[k2k6] {len(items)} 题，cap_n={CAP_N}, direct_cap={DIRECT_CAP}", flush=True)

    t0 = time.time()
    loaded = pipeline_mod.load_generation(GEN)
    pipe = pipeline_mod.build_pipeline(loaded, top_k=20, graph_depth=1)
    pipe["generation_id"] = GEN
    pipe["entities"] = loaded["graph"].get("entities") or []
    r = pipe["retriever"]
    r.budget["max_context_tokens"] = BUDGET
    print(f"[load] {time.time()-t0:.1f}s", flush=True)

    out_rows = []
    t1 = time.time()
    for i, it in enumerate(items, 1):
        q = str(it.get("question") or "")
        cat = str(it.get("category") or "")
        rec = {"question_id": it.get("question_id"), "category": cat,
               "selected_by": it.get("_sel")}
        try:
            if cat == "cross_chunk":
                rec["w3_cap"] = measure_cap_rate(r, pipe, q)
            # K2/K6 只在非桥接、非章节题上测（cap 的生效面）
            if not bridge_mod.is_bridge_question(q) and not pipeline_mod.is_chapter_question(q):
                cc = build_ctx(r, pipe, q, direct_cap=DIRECT_CAP)
                cp = build_ctx(r, pipe, q)
                rec["k2"] = measure_k2(cc, retriever=r, pipe=pipe, question=q)
                rec["k6"] = measure_k6(cc, cp)
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
        out_rows.append(rec)
        if i % 100 == 0:
            print(f"[k2k6] {i}/{len(items)} {time.time()-t1:.1f}s", flush=True)

    elapsed = time.time() - t1
    print(f"[k2k6] 完成 {len(out_rows)} 题，{elapsed:.1f}s", flush=True)

    k2_rows = [x for x in out_rows if "k2" in x]
    k6_rows = [x for x in out_rows if "k6" in x]
    c3_rows = [x for x in out_rows if "w3_cap" in x]

    summary = {
        "k2": {
            "n_tested": len(k2_rows),
            "n_overtake": sum(x["k2"]["overtake"] for x in k2_rows),
            "all_prefix_ok": all(x["k2"]["prefix_ok"] for x in k2_rows) if k2_rows else None,
        },
        "k6": {
            "n_tested": len(k6_rows),
            "n_plain_subset": sum(1 for x in k6_rows if x["k6"]["plain_subset_of_candidates"]),
            "all_subset": all(x["k6"]["plain_subset_of_candidates"] for x in k6_rows) if k6_rows else None,
            "total_missing": sum(len(x["k6"]["missing_from_candidates"]) for x in k6_rows),
        },
        "w3_cap_rate": {
            "n_cross_chunk": len(c3_rows),
            "n_triggers": sum(1 for x in c3_rows if x["w3_cap"]["triggers"]),
            "rate": round(sum(1 for x in c3_rows if x["w3_cap"]["triggers"]) / len(c3_rows), 4)
            if c3_rows else None,
            "chapter_counts": [x["w3_cap"]["n_chapters"] for x in c3_rows],
        },
    }

    payload = {
        "schema": "arkrag.diag_k2_k6_caprate/1",
        "purpose": "W1 K2/K6 + W3 cap 触发率：一次装配、三项计数",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": platform.python_version(),
        "os": platform.platform(),
        "command": "python scripts/run_guarded.py -- python scripts/diag_k2_k6_caprate.py",
        "generation": GEN,
        "chapter_cap": CAP_N,
        "direct_cap": DIRECT_CAP,
        "elapsed_ms": round(elapsed * 1000.0, 1),
        "summary": summary,
        "per_question": out_rows,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")
    print(f"[k2k6] -> {OUT}")
    import json as J
    print("[k2k6] K2:", J.dumps(summary["k2"], ensure_ascii=False))
    print("[k2k6] K6:", J.dumps(summary["k6"], ensure_ascii=False))
    print("[k2k6] W3 cap:", J.dumps({k: v for k, v in summary["w3_cap_rate"].items()
                                    if k != "chapter_counts"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
