"""回滚等价性证明：对同一批题求逐题摘要，用于「改动前 vs 改动后」逐位比对。

设计要求：**本脚本必须能在改动前和改动后的两份 `retrieve.py` 上各跑一次**，
所以它不许 import 任何新增符号（例如 `retrieve.COOC_BOOST_WEIGHT`）——那会让
「回到改动前」的那一次直接 AttributeError，证明就做不成了。

摘要取两样东西：

* 逐题的观测向量（`any_gold` / `all_gold` / `covered` / `abstained` / `units` /
  `tokens_used` / `citations` / `gold_first_rank` / 答案字面 / 上下文摘要），
* 一个覆盖全部题的总体 SHA-256。

两次运行的总体摘要相同 ⇒ 回滚后与改动前**逐位一致**，不是「差不多」。

用法（Windows，注意 `workdir` 里的中文目录）：

    py -3 scripts/a_rollback_check.py --out-json tmp_a_rollback_before.json
    # 把 retrieve.py 换回改动前版本后：
    py -3 scripts/a_rollback_check.py --out-json tmp_a_rollback_before.json

退出码 0 = 跑通。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from arkrag import config, pipeline as pipeline_mod  # noqa: E402

EVAL_TOP_K = 20
EVAL_GRAPH_DEPTH = 1


def load_dev(gold_path: Path) -> list[dict]:
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    return [
        item for item in (gold.get("items") or [])
        if item.get("split") == "dev"
        and not item.get("should_abstain")
        and (item.get("gold_chunk_ids") or [])
    ]


def coverage_of(answer: dict, reqs: list[dict]) -> bool:
    emitted = {
        (str(c.get("chunk_id") or ""), str(c.get("evidence") or ""))
        for c in (answer.get("citations") or [])
    }
    return bool(reqs) and all(
        any(cid == str(r.get("chunk_id") or "") and str(r.get("evidence") or "") in ev
            for cid, ev in emitted)
        for r in reqs
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generation", default="g-20261103T132152Z-06a741")
    ap.add_argument("--gold-v2", default="data/goldset/gold-v2.json")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--cooc-weight", type=float, default=None,
                    help=("把 retrieve.COOC_BOOST_WEIGHT 临时设成这个值。"
                          "改动前的代码没有这个常量，此时本参数只能是 None 或 0；"
                          "传了非 0 值而模块里没有该常量会直接报错，不静默忽略。"))
    args = ap.parse_args(argv)

    # 改动后的代码才有这个常量。用 getattr 探测而不是 try/except ImportError：
    # 「改动前那一份」必须能原样跑通，否则回滚证明无从比对。
    from arkrag import retrieve  # noqa: E402
    has_const = hasattr(retrieve, "COOC_BOOST_WEIGHT")
    if args.cooc_weight is not None:
        if not has_const:
            print("[2] 当前源码没有 COOC_BOOST_WEIGHT，不能设权重", file=sys.stderr)
            return 2
        retrieve.COOC_BOOST_WEIGHT = float(args.cooc_weight)

    gold_path = Path(args.gold_v2)
    if not gold_path.is_absolute():
        gold_path = ROOT / gold_path
    items = load_dev(gold_path)

    loaded = pipeline_mod.load_generation(args.generation)
    pipe = pipeline_mod.build_pipeline(loaded, top_k=EVAL_TOP_K, graph_depth=EVAL_GRAPH_DEPTH)

    rows = []
    for item in items:
        out = pipeline_mod.ask_once(pipe, item["question"])
        hits = [str(h.get("chunk_id") or "") for h in ((out.get("search") or {}).get("hits") or [])]
        ans = out.get("answer") or {}
        ctx = out.get("context") or {}
        units = [str(u.get("chunk_id") or "") for u in (ctx.get("units") or [])]
        gold = [str(g) for g in (item.get("gold_chunk_ids") or [])]
        rank = {cid: i + 1 for i, cid in enumerate(hits)}
        gold_ranks = [rank[g] for g in gold if g in rank]
        present = set(units)
        rows.append({
            "question_id": item["question_id"],
            "any_gold": any(g in present for g in gold),
            "all_gold": all(g in present for g in gold),
            "covered": coverage_of(ans, item.get("required_citations") or []),
            "abstained": bool(ans.get("abstained")),
            "units": len(units),
            "tokens_used": int((ctx.get("stats") or {}).get("tokens_used") or 0),
            "citations": len(ans.get("citations") or []),
            "gold_first_rank": min(gold_ranks) if gold_ranks else None,
            "answer": str(ans.get("answer") or ""),
            "unit_sequence": units,
            "context_digest": str(ctx.get("context_digest") or ""),
        })

    payload = "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    report = {
        "n_questions": len(rows),
        "overall_sha256": digest,
        "aggregate": {
            "any_gold": sum(1 for r in rows if r["any_gold"]),
            "all_gold": sum(1 for r in rows if r["all_gold"]),
            "covered": sum(1 for r in rows if r["covered"]),
            "abstained": sum(1 for r in rows if r["abstained"]),
            "citations": sum(r["citations"] for r in rows),
        },
        "python": platform.python_version(),
        "gold_set": config.rel_to_root(str(gold_path)),
        "rows": rows,
    }
    out_path = Path(args.out_json)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"n={len(rows)}  aggregate={report['aggregate']}")
    print(f"overall_sha256={digest}")
    print(f"写出 {out_path}  ({out_path.stat().st_size:,} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
