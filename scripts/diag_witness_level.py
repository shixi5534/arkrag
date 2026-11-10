"""W4：`witness_level` 分布审计 —— 在真实答案上逐值跑 `explain_v2`。

## 为什么需要单独跑

上一版从 `judge_both_dev.json` 里读 `explain`，但评测脚本**不存**这个字段
（它只存四个键的 verdict）。要审计 `witness_level`，必须拿真实答案重新调
`explain_v2()`。

## 这个数意味着什么

`support_ok` 的见证链按 `SUPPORT_POLICY` 优先级取第一个命中层：

| 层 | 含义 | 强度 |
|---|---|---|
| `endpoint` | 值 == 断言 subject/object | 强（真三元组端点） |
| `predicate` | 值 == 断言谓词 | 强（关系词答案） |
| `evidence` | 值出现在断言原文 | 中（子串，普通词与实体名不分） |
| `chunk_field` | 值 == 被引用块的 chapter/title | **弱（最弱一环）** |

**`chunk_field` 占比高 ⇒ `support_ok` 的严格度有很大一部分是空转的**：
它只确认"这个章节名在被引用的块里"，不确认"这个值有断言支撑"。

W4a 明确写过"这只是可观测，不是已解决"。本脚本就是去把那个可观测的数拉出来。
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import answer as answer_mod  # noqa: E402
from arkrag import bridge as bridge_mod  # noqa: E402
from arkrag import judge_v2  # noqa: E402
from arkrag import pipeline as pipeline_mod  # noqa: E402
from arkrag import query as query_mod  # noqa: E402
from arkrag import quote_locate  # noqa: E402
from arkrag import retrieve as retrieve_mod  # noqa: E402
from arkrag import vocab  # noqa: E402

GEN = "g-20261103T132152Z-06a741"
GOLD = ROOT / "data" / "goldset" / "gold-v2.json"
GOLD_V1 = ROOT / "data" / "goldset" / "gold-v1.json"
OUT = ROOT / "data" / "diag_witness_level.json"

BUDGET = int(pipeline_mod.PROD_PROFILE["max_context_tokens"])
OPTIONS = pipeline_mod.prod_options()
CAP = int(OPTIONS.get("chapter_cap") or 0)


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


def ask_with_context(pipe, retriever, question):
    """跑一次 ask_once，拿回 context 与 answer（供 explain_v2 用）。"""
    return pipeline_mod.ask_once(pipe, question, options=OPTIONS)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="dev,holdout,clean")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    v1 = load_v1_ids()
    doc = json.loads(GOLD.read_text(encoding="utf-8"))
    all_items = doc["items"] if isinstance(doc, dict) else doc
    picked = []
    for sp in splits:
        for it in pick(all_items, sp, v1):
            r = dict(it)
            r["_sel"] = sp
            picked.append(r)
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
    print(f"[witness] {len(items)} 题，cap={CAP}", flush=True)

    loaded = pipeline_mod.load_generation(GEN)
    pipe = pipeline_mod.build_pipeline(loaded, top_k=20, graph_depth=1)
    pipe["generation_id"] = GEN
    pipe["entities"] = loaded["graph"].get("entities") or []
    r = pipe["retriever"]
    r.budget["max_context_tokens"] = BUDGET
    chap = {}
    for cid, rec in (loaded["chunks"] or {}).items():
        ch = str((rec or {}).get("chapter") or "")
        if ch:
            chap[ch] = chap.get(ch, "") + str((rec or {}).get("text") or "")
    print(f"[witness] chunks={len(loaded['chunks'])} chapters={len(chap)}", flush=True)

    level_counter = Counter()
    cat_level = {}
    rows = []
    n_ok = 0
    n_err = 0
    for i, it in enumerate(items, 1):
        q = str(it.get("question") or "")
        cat = str(it.get("category") or "")
        try:
            res = ask_with_context(pipe, r, q)
            ans = res.get("answer") or {}
            ctx = res.get("context")
            exp = judge_v2.explain_v2(
                cat, ans, it, context=ctx, question=q,
                chapter_texts=chap, chapter_cap=CAP)
            # `explain_v2` 的逐值见证在 `axes_detail.support.per_value`。
            # 不是顶层 `values`——第一次写错时 `values=0`、`by_level={}`，
            # 那种"空但没报错"的形态最容易骗过自己。
            detail = (exp.get("axes_detail") or {}) if isinstance(exp, dict) else {}
            per_value = ((detail.get("support") or {}).get("per_value")) or []
            levels = []
            for v in per_value:
                lv = str((v or {}).get("witness_level") or "none")
                levels.append(lv)
                level_counter[lv] += 1
                cat_level.setdefault(cat, Counter())[lv] += 1
            rows.append({"question_id": it.get("question_id"), "category": cat,
                         "selected_by": it.get("_sel"), "levels": levels,
                         "v2_pass": (exp.get("verdict") or {}).get("pass")})
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            n_err += 1
            rows.append({"question_id": it.get("question_id"), "category": cat,
                         "error": f"{type(exc).__name__}: {exc}"})
        if i % 100 == 0:
            print(f"[witness] {i}/{len(items)}", flush=True)

    total = sum(level_counter.values())
    summary = {
        "n_questions": len(items),
        "n_ok": n_ok,
        "n_error": n_err,
        "n_values": total,
        "by_level": dict(level_counter),
        "by_level_pct": {k: round(100.0 * v / total, 2) for k, v in level_counter.items()}
        if total else {},
        "by_category": {c: dict(cnt) for c, cnt in cat_level.items()},
        "chunk_field_share_pct": round(100.0 * level_counter.get("chunk_field", 0) / total, 2)
        if total else None,
        "evidence_share_pct": round(100.0 * level_counter.get("evidence", 0) / total, 2)
        if total else None,
        "limitation": (
            "chunk_field 与 evidence 是两个最弱的层，它们的占比决定 support_ok "
            "有多少说服力。本统计只暴露这个数，不解决问题。"
        ),
    }

    payload = {
        "schema": "arkrag.diag_witness_level/2",
        "purpose": "W4：witness_level 分布审计（在真实答案上逐值跑 explain_v2）",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": platform.python_version(),
        "os": platform.platform(),
        "command": "python scripts/run_guarded.py -- python scripts/diag_witness_level.py",
        "generation": GEN,
        "chapter_cap": CAP,
        "support_policy": {k: list(v) for k, v in judge_v2.SUPPORT_POLICY.items()},
        "summary": summary,
        "per_question": rows,
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                              encoding="utf-8")
    print(f"[witness] -> {args.out}")
    print(f"[witness] ok={n_ok} err={n_err} values={total}")
    print(f"[witness] by_level={dict(level_counter)}")
    if total:
        print(f"[witness] chunk_field={summary['chunk_field_share_pct']}%  "
              f"evidence={summary['evidence_share_pct']}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
