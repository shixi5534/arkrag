"""W3 前置测量：章节排序键的质量对照表（判据 C5 的输入）。

## 这个脚本回答什么

W3 的 `chapter_cap` 要截断章节答案，但**按什么键排序**这件事此前没有实测依据。
`answer.py` 里那个 `_chapter_rank_key = (-Σscore, -len(units), chapter)` 是
**先验选择**——W3 的实现者在报告里明确写了「排序键质量未知，这是先验不是实测最优」。

本脚本就是补这个缺口的测量：对每一道 `cross_chunk` 题，枚举几个候选排序键，
看它们各自「把 gold 章节排进前 k」的比例。**只测量，不改实现。**

## 为什么必须单独测

`cap=4` 只消除 `|O| > 2|W|` 这一条判据失败，**不证明答案正确**。
如果排序键把 gold 章节排到第 5 位而噪声排到第 1 位，cap=4 反而会让
`W ⊄ O` —— 从"列太多"变成"该有的没列"，后者更坏。

所以必须先证明"排序键能把 gold 排到前面"，再谈截断。

## 候选排序键

| 键 | 定义 | 它会怎么错 |
|---|---|---|
| `sum_score` | `-Σ unit.score` | 多而弱压过少而强（3 个 0.25 赢 1 个 0.75） |
| `max_score` | `-max unit.score` | 单个高分块主导，忽略证据厚度 |
| `count` | `-len(units)` | 噪声章节天然块多 |
| `evidence` | `-Σ evidence_spans 数` | 引用密度高的块占优 |
| `sum_score_then_count` | `(-Σscore, -len)` | **当前实现用的键** |

## 产出

`data/diag_cx_rank_keys.json`：
- 每题：gold 章节集、各键下的 gold 最好名次、`hit_at_k` for k=2..8
- 汇总：每个键的 `hit@k` 曲线、gold 名次分布、以及"当前键 vs 最优键"的差距

## 用法

    python scripts/run_guarded.py -- python scripts/diag_cx_rank_keys.py
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

from arkrag import answer as answer_mod  # noqa: E402
from arkrag import pipeline as pipeline_mod  # noqa: E402
from arkrag import query as query_mod  # noqa: E402
from arkrag import quote_locate  # noqa: E402
from arkrag import retrieve as retrieve_mod  # noqa: E402
from arkrag import vocab  # noqa: E402

GENERATION = "g-20261103T132152Z-06a741"
GOLD_SET = ROOT / "data" / "goldset" / "gold-v2.json"
GOLD_V1 = ROOT / "data" / "goldset" / "gold-v1.json"
DEFAULT_OUT = ROOT / "data" / "diag_cx_rank_keys.json"

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


def rank_keys(by_chapter: dict[str, list[dict]]) -> dict[str, list[str]]:
    """对同一份 `by_chapter` 施加多个候选排序键，返回 {键名: 章节有序列表}。

    所有键都以「升序 = 好」为准，最后一个分量统一是章节名（确定性兜底）。
    """
    def s(units: list[dict]) -> float:
        return sum(float(u.get("score") or 0.0) for u in units)

    def mx(units: list[dict]) -> float:
        return max((float(u.get("score") or 0.0) for u in units), default=0.0)

    def ev(units: list[dict]) -> int:
        return sum(len(u.get("evidence_spans") or ()) for u in units)

    chapters = list(by_chapter.keys())
    keys = {
        "sum_score": lambda c: (-s(by_chapter[c]), 0, 0.0, c),
        "max_score": lambda c: (-mx(by_chapter[c]), 0, 0.0, c),
        "count": lambda c: (-len(by_chapter[c]), 0, 0.0, c),
        "evidence": lambda c: (-ev(by_chapter[c]), 0, 0.0, c),
        # 当前实现用的键：(-Σscore, -len(units), chapter)
        "sum_score_then_count": lambda c: (-s(by_chapter[c]), -len(by_chapter[c]), 0.0, c),
    }
    out: dict[str, list[str]] = {}
    for name, kf in keys.items():
        out[name] = sorted(chapters, key=kf)
    return out


def measure_one(pipe: dict, retriever, it: dict) -> dict:
    """跑一题，返回各排序键下 gold 章节的名次。"""
    qid = str(it.get("question_id") or "")
    question = str(it.get("question") or "")
    gold_chunks = {str(c) for c in (it.get("gold_chunk_ids") or [])}

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
        from arkrag import bridge as bridge_mod
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
    from arkrag import bridge as bridge_mod
    if reserve and not bridge_mod.is_bridge_question(question):
        reserve = 0
    ctx_kwargs = {"graph_reserve_tokens": reserve} if reserve else {}
    context = retriever.build_context(
        found["hits"], expanded["expansions"], query_analysis=qa, **ctx_kwargs)

    units = context.get("units") or []
    # 复刻 `_chapter_answer` 的聚合：按 unit["chapter"] 分组，
    # 只要该 unit 的块 id 在 gold 里，就把这个章节记为 gold 章节。
    by_chapter: dict[str, list[dict]] = {}
    for u in units:
        ch = str(u.get("chapter") or "")
        if not ch:
            continue
        by_chapter.setdefault(ch, []).append(u)

    gold_in_ctx = {cid for cid in gold_chunks
                   if any(str(u.get("chunk_id") or "") == cid for u in units)}

    base = {
        "question_id": qid,
        "split": str(it.get("split") or ""),
        "category": str(it.get("category") or ""),
        "question": question,
        "selected_by": str(it.get("_selected_by") or it.get("split") or ""),
        "n_units": len(units),
        "n_chapters": len(by_chapter),
        "n_gold_chunks": len(gold_chunks),
        "n_gold_in_ctx": len(gold_in_ctx),
    }

    if not by_chapter or not gold_in_ctx:
        base.update({"status": "no_gold_in_context", "gold_chapters": [],
                     "ranks": {}, "hit_at": {}})
        return base

    # gold 章节 = 含有 gold 块的章节
    gold_chapters = sorted({
        str(u.get("chapter") or "")
        for u in units
        if str(u.get("chunk_id") or "") in gold_chunks and u.get("chapter")
    })
    if not gold_chapters:
        base.update({"status": "no_gold_chapter", "gold_chapters": [],
                     "ranks": {}, "hit_at": {}})
        return base

    keys = rank_keys(by_chapter)
    ranks: dict[str, int | None] = {}
    hit_at: dict[str, dict[str, bool]] = {}
    for kname, ordered in keys.items():
        pos = {c: i + 1 for i, c in enumerate(ordered)}
        best = min((pos[g] for g in gold_chapters if g in pos), default=None)
        ranks[kname] = best
        hit_at[kname] = {str(k): (best is not None and best <= k)
                         for k in (2, 3, 4, 5, 6, 7, 8)}

    base.update({
        "status": "ok",
        "gold_chapters": gold_chapters,
        "chapters_ordered_by_sum_score": keys["sum_score_then_count"],
        "ranks": ranks,
        "hit_at": hit_at,
    })
    return base


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("status") == "ok"]
    keys = ["sum_score", "max_score", "count", "evidence", "sum_score_then_count"]
    out: dict = {"n_questions": len(rows), "n_ok": len(ok)}
    per_key: dict = {}
    for kname in keys:
        ranks = [r["ranks"][kname] for r in ok if r.get("ranks", {}).get(kname) is not None]
        hit = {}
        for k in (2, 3, 4, 5, 6, 7, 8):
            n = sum(1 for r in ok if r.get("hit_at", {}).get(kname, {}).get(str(k)))
            hit[str(k)] = {"n": n, "rate": round(n / len(ok), 4) if ok else None}
        per_key[kname] = {
            "gold_rank_median": int(statistics.median(ranks)) if ranks else None,
            "gold_rank_mean": round(sum(ranks) / len(ranks), 2) if ranks else None,
            "n_with_rank": len(ranks),
            "hit_at": hit,
        }
    out["per_key"] = per_key

    # 当前键相对各键的差距
    cur = per_key["sum_score_then_count"]["hit_at"]
    best_key = max(keys, key=lambda k: per_key[k]["hit_at"].get("4", {}).get("n", 0))
    out["current_key"] = "sum_score_then_count"
    out["best_key_at_4"] = best_key
    out["current_vs_best_at_4"] = {
        "current_n": cur.get("4", {}).get("n"),
        "best_n": per_key[best_key]["hit_at"].get("4", {}).get("n"),
        "best_key": best_key,
    }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="W3 前置：章节排序键质量对照")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--splits", default="dev,holdout,clean")
    args = ap.parse_args(argv)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    v1_ids = load_v1_ids()
    doc = json.loads(GOLD_SET.read_text(encoding="utf-8"))
    all_items = doc["items"] if isinstance(doc, dict) else doc

    # 只测 cross_chunk：W3 只影响这一类。
    picked: list[dict] = []
    for sp in splits:
        for it in pick_items(all_items, sp, v1_ids):
            if str(it.get("category") or "") != "cross_chunk":
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

    print(f"[cx] 计划 {len(items)} 题 cross_chunk，budget={BUDGET}", flush=True)

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
        except Exception as exc:  # noqa: BLE001 - 单题失败必须留痕
            rows.append({
                "question_id": str(it.get("question_id") or ""),
                "split": str(it.get("split") or ""),
                "category": str(it.get("category") or ""),
                "selected_by": str(it.get("_selected_by") or ""),
                "status": "run_error",
                "error": f"{type(exc).__name__}: {exc}",
            })
        if i % 20 == 0:
            print(f"[cx] {i}/{len(items)} {time.time() - t1:.1f}s", flush=True)

    elapsed = time.time() - t1
    print(f"[cx] 完成 {len(rows)} 题，{elapsed:.1f}s", flush=True)

    summary = summarize(rows)
    for sp in splits:
        sub = [r for r in rows if sp in str(r.get("selected_by") or "")]
        summary[f"split_{sp}"] = summarize(sub)

    payload = {
        "schema": "arkrag.diag_cx_rank_keys/1",
        "purpose": "W3 前置：证明章节排序键能把 gold 章节排到前面，再谈 cap",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": platform.python_version(),
        "os": platform.platform(),
        "command": "python scripts/run_guarded.py -- python scripts/diag_cx_rank_keys.py",
        "generation": GENERATION,
        "gold_set": str(GOLD_SET.relative_to(ROOT)),
        "gold_set_sha": _sha_file(GOLD_SET),
        "config_digest": _digest({"budget": BUDGET, "options": OPTIONS}),
        "elapsed_ms": round(elapsed * 1000.0, 1),
        "n_per_question": len(rows),
        "summary": summary,
        "per_question": sorted(rows, key=lambda r: (r.get("selected_by") or "",
                                                    r.get("question_id") or "")),
    }
    payload["run_digest"] = _digest({
        "config_digest": payload["config_digest"],
        "per_question": [{k: v for k, v in r.items()}
                         for r in payload["per_question"]],
    })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")
    print(f"[cx] -> {out}")
    print(f"[cx] run_digest={payload['run_digest']}")
    pk = summary["per_key"]
    print("[cx] 各键 gold 名次中位 / hit@4:")
    for k in pk:
        print("   %-22s median=%-4s hit@4=%s/%s (%s)"
              % (k, pk[k]["gold_rank_median"],
                 pk[k]["hit_at"].get("4", {}).get("n"),
                 summary["n_ok"],
                 pk[k]["hit_at"].get("4", {}).get("rate")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
