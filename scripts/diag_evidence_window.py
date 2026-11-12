"""L4 结构量诊断：整块计价 vs 证据窗口计价，24000 预算能装多少单元。

**只读诊断，默认不动任何实现。** 它回答三个可机检的问题：

1. 同一道题、同一批命中与扩展，`build_context` 在「整块计价」与「证据窗口
   计价」两种口径下各装了几个单元、用了多少 token？
2. 窗口计价下 24000 预算的**单元容量**涨到多少（结构量，不是分数）？
3. 窗口有没有把任何一条断言的证据甩到窗口外（**必须为 0**）？

## 为什么这个脚本值得存在（即使实现日后被回滚）

端到端分数只能由 Lead 串行跑（一次 `load_generation()` 常驻 1.0–1.5 GB，
并发两份就会重演 58 GiB commit 事故）。而**结构容量**是纯算术：同一份索引、
同一份断言、同一批块，只换计价口径。容量翻不上去，端到端就不可能涨；
容量翻上去了而分数没涨，问题在回答侧，不在计价。这条推理链必须在没有
端到端读数的条件下也能走完，所以需要本文件。

## 与 `scripts/diag_rank_24000.py` 的关系

调用顺序**逐行照抄** `pipeline.ask_once(options=prod_options())` 的前半段
（`analyze_query` → `search(top_k=40 if 桥接题 else 20)` → `quote_locate` 前插
→ `expand` → `build_context`），因为换一条链路测出来的就不是生产读数。
`--verify-baseline` 会把「窗口关闭」那一臂的 `n_units` / `ctx_tokens` 与
`data/diag_rank_24000.json` 逐题比较——那是**默认路径逐位不变**的机检，
不需要跑评测，也不需要信任何自我声明。

## 用法

    # 3 题冒烟（先确认熔断与链路）
    py -3.14 scripts/run_guarded.py --max-commit-pct 92 --min-avail-mb 2048 \\
        --stdout _g_out.txt --stderr _g_err.txt -- \\
        py -3.14 scripts/diag_evidence_window.py --limit 3

    # 全量 588 题
    py -3.14 scripts/run_guarded.py --max-commit-pct 92 --min-avail-mb 2048 \\
        --stdout _g_out.txt --stderr _g_err.txt -- \\
        py -3.14 scripts/diag_evidence_window.py
"""
from __future__ import annotations

import argparse
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
GOLD_V1 = ROOT / "data" / "goldset" / "gold-v1.json"
DEFAULT_OUT = ROOT / "data" / "diag_evidence_window.json"
RANK_REF = ROOT / "data" / "diag_rank_24000.json"

BUDGET = int(pipeline_mod.PROD_PROFILE["max_context_tokens"])
OPTIONS = pipeline_mod.prod_options()
SPLITS = ("dev", "holdout", "clean")

#: 头部整块条数的扫档。12 是预登记默认值（`retrieve.WINDOW_HEAD_DEFAULT`），
#: 这里扫 0/4/8/12/16 是为了让 Lead 能看见「头部条数 → 容量」的整条曲线，
#: 而不是只拿到一个点。
HEAD_SWEEP = (0, 4, 8, 12, 16)


def _digest(obj) -> str:
    import hashlib
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def read_gold() -> list[dict]:
    doc = json.loads(GOLD_SET.read_text(encoding="utf-8"))
    return list((doc["items"] if isinstance(doc, dict) else doc) or [])


def load_v1_ids() -> set[str]:
    if not GOLD_V1.exists():
        return set()
    doc = json.loads(GOLD_V1.read_text(encoding="utf-8"))
    items = doc.get("items") if isinstance(doc, dict) else doc
    return {str(it.get("question_id")) for it in items or []}


def pick_items(items: list[dict], split: str, v1_ids: set[str]) -> list[dict]:
    """选题口径**逐字对齐** `scripts/eval_answer_quality.py::pick_items`。

    `clean` 不是 gold-v2 里的一个 split 值，而是「holdout 里、有 gold、
    且题号不在 gold-v1 里」——按 `split == "clean"` 过滤会得到 0 题。
    """
    if split == "all":
        return [it for it in items if it.get("gold_chunk_ids")]
    if split == "clean":
        return [it for it in items
                if it.get("split") == "holdout" and it.get("gold_chunk_ids")
                and str(it.get("question_id")) not in v1_ids]
    return [it for it in items
            if it.get("split") == split and it.get("gold_chunk_ids")]


def _is_bridge(question: str) -> bool:
    from arkrag import bridge
    return bridge.is_bridge_question(question)


def collect_candidates(pipe: dict, retriever, it: dict) -> dict | None:
    """照 `ask_once` 的前半段取 `(hits, expansions, qa)`。无 gold 的题返回 None。"""
    gold = [str(c) for c in (it.get("gold_chunk_ids") or [])]
    if not gold:
        return None
    question = str(it.get("question") or "")
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
        if wide and OPTIONS.get("bridge") and _is_bridge(question):
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
        seed_ids,
        depth=pipe["graph_depth"],
        query_analysis=qa,
        seed_chunk_ids=[h["chunk_id"] for h in found["hits"]],
    )
    reserve = int(OPTIONS.get("graph_reserve_tokens") or 0)
    if reserve and not _is_bridge(question):
        reserve = 0
    return {"gold": gold, "question": question, "qa": qa, "found": found,
            "expansions": expanded["expansions"], "reserve": reserve}


def arm(retriever, cand: dict, *, evidence_window: int,
        window_head_units: int | None) -> dict:
    """跑一臂 `build_context`，回收容量与断言安放情况。"""
    kw: dict = {}
    if cand["reserve"]:
        kw["graph_reserve_tokens"] = cand["reserve"]
    if evidence_window:
        kw["evidence_window"] = 1
        # 显式传头部条数（含 0）：0 也是一条要测的臂，不能用「不传」表示。
        kw["window_head_units"] = int(window_head_units or 0)
    ctx = retriever.build_context(
        cand["found"]["hits"], cand["expansions"],
        query_analysis=cand["qa"], **kw)
    units = ctx.get("units") or []
    stats = ctx.get("stats") or {}

    # ---- 断言安放：每一条 evidence span 必须完整落在窗口内 ------------------
    n_spans = 0
    n_out = 0
    # ---- `unit["text"]` 必须与块正文逐字等长（预登记反证条件 ③） ----------
    n_text_changed = 0
    for u in units:
        cid = str(u.get("chunk_id") or "")
        body = str(u.get("text") or "")
        src = str((retriever.chunks.get(cid) or {}).get("text") or "")
        if len(body) != len(src) or body != src:
            n_text_changed += 1
        lo = int(u.get("window_start", 0))
        hi = int(u.get("window_end", len(body)))
        for span in u.get("evidence_spans") or []:
            s = int(span.get("start", -1))
            e = int(span.get("end", -1))
            if s < 0 or e <= s:
                continue
            n_spans += 1
            if s < lo or e > hi:
                n_out += 1
        # 窗口正文必须与 `text[start:end]` 逐字一致（连续区间，不拼接）
        if u.get("window_applied"):
            if str(u.get("window_text") or "") != body[lo:hi]:
                n_out += 1

    gold = set(cand["gold"])
    ctx_ids = [str(u.get("chunk_id") or "") for u in units]
    return {
        "units": int(stats.get("units") or 0),
        "tokens_used": int(stats.get("tokens_used") or 0),
        "n_candidates": int(stats.get("candidates") or 0),
        "n_truncated": len(stats.get("truncated_chunks") or ()),
        "any_gold": bool(gold & set(ctx_ids)),
        "all_gold": bool(gold) and gold.issubset(set(ctx_ids)),
        "gold_units": sum(1 for c in ctx_ids if c in gold),
        "evidence_spans": n_spans,
        "spans_outside_window": n_out,
        "text_len_changed": n_text_changed,
        "windowed_units": int(stats.get("windowed_units") or 0),
        "window_full_units": int(stats.get("window_full_units") or 0),
        "window_saved_tokens": int(stats.get("window_saved_tokens") or 0),
        "window_chars_kept_pct": stats.get("window_chars_kept_pct"),
    }


def _med(vals):
    return statistics.median(vals) if vals else None


def _mean(vals):
    return round(sum(vals) / len(vals), 2) if vals else None


def summarize(rows: list[dict], arms: list[str]) -> dict:
    out: dict = {"n_questions": len(rows), "arms": {}}
    for a in arms:
        sub = [r["arms"][a] for r in rows if a in r.get("arms", {})]
        if not sub:
            continue
        out["arms"][a] = {
            "units_median": _med([s["units"] for s in sub]),
            "units_mean": _mean([s["units"] for s in sub]),
            "tokens_used_median": _med([s["tokens_used"] for s in sub]),
            "tokens_used_mean": _mean([s["tokens_used"] for s in sub]),
            "any_gold": sum(1 for s in sub if s["any_gold"]),
            "all_gold": sum(1 for s in sub if s["all_gold"]),
            "gold_units_mean": _mean([s["gold_units"] for s in sub]),
            "truncated_mean": _mean([s["n_truncated"] for s in sub]),
            "windowed_units_mean": _mean([s["windowed_units"] for s in sub]),
            "window_saved_tokens_mean": _mean([s["window_saved_tokens"] for s in sub]),
            "window_chars_kept_pct_median": _med(
                [s["window_chars_kept_pct"] for s in sub
                 if s.get("window_chars_kept_pct") is not None]),
            "spans_total": sum(s["evidence_spans"] for s in sub),
            "spans_outside_window": sum(s["spans_outside_window"] for s in sub),
            "text_len_changed": sum(s.get("text_len_changed") or 0 for s in sub),
        }
    return out


def capacity_table(rows: list[dict], arms: list[str], budget: int) -> list[dict]:
    """每臂的「单元容量」：token/单元 → 预算能装多少个。

    这是**结构量**，不是分数。`units` 是实际装进去的数量（受候选数上限约束），
    `capacity` 是按实测 token/单元外推的物理容量——两者必须一起报，
    只报后者会掩盖「候选池已经不够大」这件事。
    """
    out = []
    for a in arms:
        per = [r["arms"][a]["tokens_used"] / r["arms"][a]["units"]
               for r in rows
               if a in r.get("arms", {}) and r["arms"][a]["units"]]
        cap = [budget / p for p in per if p > 0]
        out.append({
            "arm": a,
            "token_per_unit_median": round(_med(per), 1) if per else None,
            "capacity_median": round(_med(cap), 1) if cap else None,
            "units_median": _med([r["arms"][a]["units"] for r in rows
                                  if a in r.get("arms", {})]),
            "n_candidates_median": _med([r["arms"][a]["n_candidates"] for r in rows
                                         if a in r.get("arms", {})]),
        })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="L4：证据窗口的容量对照（只读）")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0 = 全部）")
    ap.add_argument("--splits", default="dev,holdout,clean")
    ap.add_argument("--heads", default=",".join(str(x) for x in HEAD_SWEEP),
                    help="头部整块条数的扫档，逗号分隔")
    ap.add_argument("--no-verify-baseline", action="store_true",
                    help="跳过与 data/diag_rank_24000.json 的逐题对账")
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        # cp936 控制台 print 中文/箭头会 UnicodeEncodeError，让整轮以 1 退出。
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    heads = [int(x) for x in args.heads.split(",") if x.strip() != ""]

    v1_ids = load_v1_ids()
    all_items = read_gold()
    picked: list[dict] = []
    for sp in splits:
        for it in pick_items(all_items, sp, v1_ids):
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
    order = {sp: i for i, sp in enumerate(splits)}
    items.sort(key=lambda r: (order.get(str(r.get("split") or ""), 99),
                              str(r.get("question_id") or "")))
    if args.limit:
        items = items[: args.limit]

    arms = ["off"] + ["head%d" % h for h in heads]
    print(f"[l4] 计划 {len(items)} 题；预算 {BUDGET}；arms={arms}", flush=True)

    t0 = time.time()
    loaded = pipeline_mod.load_generation(GENERATION)
    pipe = pipeline_mod.build_pipeline(loaded, top_k=20, graph_depth=1)
    pipe["generation_id"] = GENERATION
    pipe["entities"] = loaded["graph"].get("entities") or []
    retriever = pipe["retriever"]
    retriever.budget["max_context_tokens"] = BUDGET
    print(f"[load] {time.time() - t0:.1f}s chunks={len(loaded['chunks'])} "
          f"assertions={len(loaded['assertions'])}", flush=True)

    rows: list[dict] = []
    t1 = time.time()
    for i, it in enumerate(items, 1):
        qid = str(it.get("question_id") or "")
        base = {"question_id": qid, "split": str(it.get("split") or ""),
                "category": str(it.get("category") or ""),
                "selected_by": str(it.get("_selected_by") or ""),
                "n_gold": len(it.get("gold_chunk_ids") or []), "arms": {}}
        try:
            cand = collect_candidates(pipe, retriever, it)
        except Exception as exc:  # noqa: BLE001 - 单题失败必须留痕，不许丢分母
            base["status"] = "collect_error"
            base["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(base)
            continue
        if cand is None:
            base["status"] = "no_gold"
            rows.append(base)
            continue
        # ---- 逐臂跑（每次都是同一份 hits/expansions，只换计价口径） --------
        try:
            base["arms"]["off"] = arm(retriever, cand, evidence_window=0,
                                      window_head_units=None)
            for h in heads:
                base["arms"]["head%d" % h] = arm(
                    retriever, cand, evidence_window=1, window_head_units=h)
        except Exception as exc:  # noqa: BLE001
            base["status"] = "arm_error"
            base["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(base)
            continue
        base["status"] = "ok"
        rows.append(base)
        if i % 50 == 0:
            print(f"[l4] {i}/{len(items)} {time.time() - t1:.1f}s", flush=True)
    elapsed = time.time() - t1
    print(f"[l4] 完成 {len(rows)} 题，{elapsed:.1f}s", flush=True)

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    payload = {
        "schema": "arkrag.diag_evidence_window/1",
        "purpose": "L4：整块计价 vs 证据窗口计价的容量对照（结构量，不是分数）",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": platform.python_version(),
        "os": platform.platform(),
        "generation": GENERATION,
        "gold_set": str(GOLD_SET.relative_to(ROOT)),
        "budget": BUDGET,
        "prod_options": dict(OPTIONS),
        "splits": splits,
        "heads": heads,
        "n_questions": len(rows),
        "n_ok": len(ok_rows),
        "elapsed_ms": round(elapsed * 1000.0, 1),
        "capacity": capacity_table(ok_rows, arms, BUDGET),
        "summary_all": summarize(ok_rows, arms),
        "summary_by_split": {
            sp: summarize([r for r in ok_rows if sp in str(r.get("selected_by") or "")], arms)
            for sp in splits
        },
        "text_len_changed_total": sum(int(r.get("text_len_changed") or 0) for r in ok_rows),
        "per_question": rows,
    }
    payload["run_digest"] = _digest({
        "budget": BUDGET, "heads": heads,
        "per_question": [{k: v for k, v in r.items()} for r in rows],
    })

    # ---- 与 `diag_rank_24000.json` 逐题对账（默认路径逐位不变的机检） -------
    if not args.no_verify_baseline and RANK_REF.exists():
        ref = json.loads(RANK_REF.read_text(encoding="utf-8"))
        refmap = {str(r.get("question_id")): r for r in ref.get("per_question") or []}
        mism = []
        cmp_n = 0
        for r in ok_rows:
            rref = refmap.get(r["question_id"])
            if not rref or rref.get("status") != "ok":
                continue
            cmp_n += 1
            got = r["arms"]["off"]
            if got["units"] != rref.get("n_units") or \
                    got["tokens_used"] != rref.get("ctx_tokens"):
                mism.append({
                    "question_id": r["question_id"],
                    "ref": [rref.get("n_units"), rref.get("ctx_tokens")],
                    "got": [got["units"], got["tokens_used"]],
                })
        payload["baseline_verify"] = {
            "ref": str(RANK_REF.relative_to(ROOT)),
            "compared": cmp_n,
            "mismatch": len(mism),
            "examples": mism[:10],
            "verdict": "PASS" if not mism else "FAIL",
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")

    # ---- 人读表 -------------------------------------------------------------
    print()
    print("=" * 92)
    print("L4 容量对照（%d 题，预算 %d，产物 = %s）" % (len(ok_rows), BUDGET, out.name))
    print("=" * 92)
    print("%-8s %8s %10s %10s %9s %9s %9s" % (
        "arm", "单元中位", "tok/单元", "容量中位", "候选中位", "窗口单元", "省tok均"))
    for row in payload["capacity"]:
        a = row["arm"]
        s = payload["summary_all"]["arms"].get(a) or {}
        print("%-8s %8s %10s %10s %9s %9s %9s" % (
            a, row["units_median"], row["token_per_unit_median"],
            row["capacity_median"], row["n_candidates_median"],
            s.get("windowed_units_mean"), s.get("window_saved_tokens_mean")))
    print()
    print("断言安放（必须 0 条落在窗口外）与 gold 覆盖：")
    for a in arms:
        s = payload["summary_all"]["arms"].get(a) or {}
        print("  %-8s spans=%6d  窗口外=%d  any_gold=%d/%d  all_gold=%d  "
              "窗口单元字符保留中位=%s%%" % (
                  a, s.get("spans_total") or 0, s.get("spans_outside_window") or 0,
                  s.get("any_gold") or 0, len(ok_rows), s.get("all_gold") or 0,
                  s.get("window_chars_kept_pct_median")))
    print()
    print("`unit[\"text\"]` 长度变化题数 = %d（预登记反证条件 ③：必须为 0）"
          % payload["text_len_changed_total"])
    if "baseline_verify" in payload:
        bv = payload["baseline_verify"]
        print("默认路径对账（vs %s）：比对 %d 题，不一致 %d 题 -> %s" % (
            bv["ref"], bv["compared"], bv["mismatch"], bv["verdict"]))
    print()
    print("按 split：")
    for sp in splits:
        s = payload["summary_by_split"][sp]
        if not s["n_questions"]:
            continue
        off = s["arms"].get("off") or {}
        h12 = s["arms"].get("head12") or {}
        print("  %-8s n=%3d  off 单元 %s / head12 单元 %s  容量 %s -> %s" % (
            sp, s["n_questions"], off.get("units_median"),
            h12.get("units_median"),
            next((c["capacity_median"] for c in payload["capacity"] if c["arm"] == "off"), None),
            next((c["capacity_median"] for c in payload["capacity"] if c["arm"] == "head12"), None)))
    print(f"[l4] -> {out}")
    print(f"[l4] run_digest={payload['run_digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
