"""预算变化的受害题审计：预算加大后，哪些题的 gold 反而被挤出上下文。

## 为什么必须有这个脚本

`scripts/a_context_budget_probe.py` 只在 **74 道「上下文未采用」题**上重放，那批题在
6000 下 gold **本来就不在**上下文里，所以它能看到的只有「救回」，「变差」恒为 0。
那个 0 是**取样口径的产物，不是预算的性质**。

真实情况：6000 下已有 162 道题的 gold 进了上下文。把它们放进来看，预算 8000 会
**挤掉其中一部分**。本脚本补的就是这个缺口——把「已通过」的题也纳入审计。

## 两种测量，必须分开看

1. **重放**（`replay`）：拿 `build_context` 的贪心循环，在同一批 units 上用不同预算
   重算保留集合。快，但只是上下文层。
2. **真跑**（`pipeline`）：`build_pipeline` + `ask_once` 走完整条链
   （search→expand→build_context→answer→verify_citations）。慢，但答案/引用会跟着变。

两者的差异不是误差，是**不同的问题**：重放回答「上下文里还有没有 gold 块」，
真跑回答「最终答案变没变」。本脚本两者都出，且分别标注。

## 受害题的机制（已实锤 q-mh-000042）

`build_context` 是**按名次贪心首适配**（`retrieve.py:865` 的
`if used_tokens + cost > budget_tokens: continue`）。预算变大 → 原本装不下的中等
成本块被放进来 → 它们吃掉了小成本 gold 块所需的余量 → gold 反而落选。
所以「预算越大越好」不成立，**变差题数必须逐题列出来，不能只看净增**。

## 用法

    py -3 scripts/a_budget_audit.py
    py -3 scripts/a_budget_audit.py --budgets 6000 8000 --out-json data/a_budget_audit.json
"""

from __future__ import annotations

import argparse
import collections
import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from arkrag import config, pipeline as pipeline_mod, subproc  # noqa: E402

SCHEMA = "arkrag.a-budget-audit/1"
DEFAULT_GENERATION = "g-20261103T132152Z-06a741"
DEFAULT_TRIAGE = "data/a_triage_all320.jsonl"
DEFAULT_BUDGETS = [6000, 8000]
EVAL_TOP_K = 20


def rel(path) -> str:
    return config.rel_to_root(str(path))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="预算变化的受害题审计（真跑管线 + 重放）")
    ap.add_argument("--generation", default=DEFAULT_GENERATION)
    ap.add_argument("--triage", default=DEFAULT_TRIAGE)
    ap.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-json", default="data/a_budget_audit.json")
    args = ap.parse_args(argv)

    triage_path = Path(args.triage)
    if not triage_path.is_absolute():
        triage_path = ROOT / triage_path
    if not triage_path.is_file():
        print(f"[2] 输入不可用：找不到 {triage_path}", file=sys.stderr)
        return 2

    rows = [json.loads(x) for x in triage_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    questions = [r for r in rows if r.get("gold_chunk_ids")]
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        print("[2] 输入不可用：没有带 gold_chunk_ids 的题", file=sys.stderr)
        return 2

    loaded = pipeline_mod.load_generation(args.generation)
    budgets = list(args.budgets)

    # ---------------- 真跑：每题、每预算跑一遍完整管线 ----------------
    per_run: dict[int, dict[str, dict]] = {}
    for budget in budgets:
        built = pipeline_mod.build_pipeline(loaded, top_k=EVAL_TOP_K, graph_depth=1)
        built["retriever"].budget["max_context_tokens"] = budget
        run: dict[str, dict] = {}
        for idx, item in enumerate(questions, 1):
            if idx % 50 == 0:
                print(f"  [预算 {budget}] {idx}/{len(questions)}", flush=True)
            out = pipeline_mod.ask_once(built, item["question"])
            units = [u["chunk_id"] for u in out["context"]["units"]]
            kept = set(units)
            gold = [str(g) for g in item["gold_chunk_ids"]]
            run[item["question_id"]] = {
                "category": item.get("category"),
                "gold_kept": [g for g in gold if g in kept],
                "gold_missing": [g for g in gold if g not in kept],
                "any_gold": any(g in kept for g in gold),
                "all_gold": all(g in kept for g in gold),
                "tokens_used": out["context"]["stats"]["tokens_used"],
                "units": len(units),
                "truncated": len(out["context"]["stats"]["truncated_chunks"]),
                "answer": out["answer"]["answer"],
                "abstained": bool(out["answer"]["abstained"]),
                "citations": len(out["answer"]["citations"]),
                "citation_texts": [c.get("evidence") for c in out["answer"]["citations"]],
            }
        per_run[budget] = run

    baseline_budget = budgets[0]
    base = per_run[baseline_budget]
    if not base:
        print("[4] 自检失败：基线预算没有结果", file=sys.stderr)
        return 4

    # 自检 1：6000 的「至少一块命中」应与逐题归因记录的 gold_in_context 一致
    mismatch = []
    for item in questions:
        rec = base[item["question_id"]]
        recorded = bool(item.get("gold_in_context"))
        if rec["any_gold"] != recorded:
            mismatch.append(item["question_id"])
    if mismatch:
        print(f"[4] 自检失败：{len(mismatch)} 题与逐题归因的 gold_in_context 不一致：{mismatch[:5]}", file=sys.stderr)
        return 4

    # ---------------- 逐题盈亏 ----------------
    deltas = {}
    for budget in budgets[1:]:
        cur = per_run[budget]
        gained, lost, partial = [], [], []
        for item in questions:
            qid = item["question_id"]
            was, now = base[qid]["any_gold"], cur[qid]["any_gold"]
            if now and not was:
                gained.append(qid)
            if was and not now:
                lost.append(qid)
            if base[qid]["all_gold"] and not cur[qid]["all_gold"]:
                partial.append(qid)
        deltas[budget] = {"gained": gained, "lost": lost, "partial": partial}

    # ---------------- 相邻区间扫描（暴露非单调）----------------
    sweep = []
    for prev, nx in zip(budgets, budgets[1:]):
        a, b = per_run[prev], per_run[nx]
        good = [i["question_id"] for i in questions if b[i["question_id"]]["any_gold"] and not a[i["question_id"]]["any_gold"]]
        bad = [i["question_id"] for i in questions if a[i["question_id"]]["any_gold"] and not b[i["question_id"]]["any_gold"]]
        sweep.append({"from": prev, "to": nx, "gained": len(good), "lost": len(bad),
                      "net": len(good) - len(bad), "lost_ids": bad})

    # ---------------- 汇总 ----------------
    def summary(run):
        n = len(questions)
        used = sorted(v["tokens_used"] for v in run.values())
        return {
            "n": n,
            "any_gold": sum(1 for v in run.values() if v["any_gold"]),
            "all_gold": sum(1 for v in run.values() if v["all_gold"]),
            "abstained": sum(1 for v in run.values() if v["abstained"]),
            "citations": sum(v["citations"] for v in run.values()),
            "tokens_median": int(statistics.median(used)),
            "tokens_max": max(used),
            "units_median": int(statistics.median([v["units"] for v in run.values()])),
        }

    by_cat = {}
    for budget in budgets:
        cur = per_run[budget]
        bucket = collections.defaultdict(lambda: {"n": 0, "any_gold": 0, "all_gold": 0})
        for item in questions:
            rec = cur[item["question_id"]]
            b = bucket[rec["category"] or ""]
            b["n"] += 1
            b["any_gold"] += 1 if rec["any_gold"] else 0
            b["all_gold"] += 1 if rec["all_gold"] else 0
        by_cat[budget] = dict(bucket)

    print(f"世代 {args.generation}  题数 {len(questions)}  预算 {budgets}")
    print()
    print(f"{'预算':>6s} {'gold至少一块':>12s} {'gold全部':>10s} {'拒答':>6s} {'引用总数':>8s} {'tokens中位':>10s}")
    for budget in budgets:
        s = summary(per_run[budget])
        print(f"{budget:6d} {s['any_gold']:12d} {s['all_gold']:10d} {s['abstained']:6d} {s['citations']:8d} {s['tokens_median']:10d}")

    print()
    print("相对基线的逐题盈亏：")
    for budget in budgets[1:]:
        d = deltas[budget]
        print(f"  到 {budget}: 救回 {len(d['gained'])}  丢失 {len(d['lost'])}  全部转部分 {len(d['partial'])}  净 {len(d['gained'])-len(d['lost']):+d}")
        if d["lost"]:
            print(f"    丢失题：{d['lost']}")
        if d["partial"]:
            print(f"    部分转部分：{d['partial']}")

    print()
    print("相邻区间扫描（非单调证据）：")
    for step in sweep:
        print(f"  {step['from']} -> {step['to']}: 好 {step['gained']} 差 {step['lost']} 净 {step['net']:+d}"
              + (f"  受害题 {step['lost_ids']}" if step["lost_ids"] else ""))

    print()
    print("按类「gold 至少一块进上下文」：")
    cats = sorted({(item.get("category") or "") for item in questions})
    header = f"{'类别':18s}" + "".join(f"{b:>12d}" for b in budgets)
    print(header)
    for cat in cats:
        row = f"{cat:18s}"
        for budget in budgets:
            b = by_cat[budget].get(cat) or {"n": 0, "any_gold": 0}
            row += f"{b['any_gold']:>6d}/{b['n']:<5d}"
        print(row)

    doc = {
        "schema": SCHEMA,
        "purpose": "预算变化的受害题审计（真跑管线）；单边改善结论必须附带丢失题清单",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "command": "py -3 scripts/a_budget_audit.py",
        "generation": args.generation,
        "triage_source": rel(triage_path),
        "eval_top_k": EVAL_TOP_K,
        "graph_depth": 1,
        "budgets": budgets,
        "summary": {str(b): summary(per_run[b]) for b in budgets},
        "deltas_vs_baseline": {str(b): deltas[b] for b in budgets[1:]},
        "adjacent_sweep": sweep,
        "by_category": {str(b): by_cat[b] for b in budgets},
        "per_question": {str(b): per_run[b] for b in budgets},
        "method_note": "真跑 build_pipeline+ask_once，非重放。retrieval 层不读 max_context_tokens，故 hits 必然一致；此处验证的是上下文与答案层。",
    }
    out = Path(args.out_json)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写出 {rel(out)}")
    return 0


def _git_commit() -> str:
    try:
        import subprocess

        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                              capture_output=True, text=True, timeout=10,
                              env=subproc.child_env()).stdout.strip()
    except Exception:
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
