"""预算真跑：6000 / 7000 / 8000 三档，在 dev 350 应答题上量下游后果。

**为什么要这个脚本。** 预算 6000→8000 这件事被撤回过一次，撤销依据是「失败子集上探针回放」；
后来结构量审计（`docs/lead_预算容量与填充实测.md`）显示 6000→8000 是 `any_gold` 救 22 丢 2、
`all_gold` 救 22 丢 0，且 9 道 frontier 题 8000 下 9/9 全进。但结构量不是下游后果：
上下文里多进一块，回答器未必用得上、引用未必发得出。这个脚本把三档各真跑一遍全 dev，
数出**下游量**（覆盖 / 正确覆盖 / 答案含 gold / 拒答），回答「8000 值不值得开」。

**单变量。** 三档只差 `retriever.budget["max_context_tokens"]` 一个整数，代码其余部分
逐字节相同；`TITLE_ANSWER_PATH` 保持仓库默认（False），不做第二处改动。

**判据（预登记，先写后跑）。**
  K1 主判据：`correct_coverage` 净增 ≥ 10 题（覆盖成立且答案含 gold）。只涨 covered
             不涨 correct_coverage 的，只能记「口径修正」，不能记成绩。
  K2 副判据：`covered` 净增 ≥ 10 题。
  K3 天花板判据：frontier 9 题（gold 名次 ≤5 却被截断）收复 ≥ 6 题。
  R1 任一题 `correct_coverage` 由真变假即「撤」：必须逐题列题号，按类单列。
  R2 `abstain_accuracy` 任何下降即撤（应答题误拒会增加，预算加大不该改变拒答）。
  R3 逐条引用定位失败任何一条即撤（`text[start:end] == evidence`）。
  R4 两道已知非单调题（`q-mh-000042` / `q-tc-900022`）的变化必须单独列出——
     它们此前在 6000→6400→7000 之间来回跳，是「贪心首适配在预算边界上掷骰子」的活标本。

**三条口径纪律（踩过坑才写下来的）。**
  1. 三个分母并列报：`/350`（全部应答题）、`/reachable`、`/ungated`。P3 盯 `/ungated`。
  2. 「引用总条数」与「逐条定位通过率」分开报；分母变小会让通过率更容易拿满分，
     不是变好。
  3. 只跑 dev，不外推到 new-holdout 160；本轮结论只用来决定「要不要给 8000 开一个
     正式轮次」，不直接进发布门。

跑法
----
    py -3 scripts/lead_budget_realrun.py
    py -3 scripts/lead_budget_realrun.py --json data/lead_budget_realrun.json
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from arkrag import config, pipeline, provenance, subproc  # noqa: E402

#: 三档预算。6000 是仓库默认（`retrieve.DEFAULT_BUDGET["max_context_tokens"]`）。
BUDGETS: tuple[int, ...] = (6000, 7000, 8000)

#: 基线档：所有「救 / 丢」都以它为参照。
BASE_BUDGET = 6000

#: 两道已知非单调题：预算之间来回跳，单独列。
NON_MONOTONIC_WATCH: tuple[str, ...] = ("q-mh-000042", "q-tc-900022")

GENERATION = "g-20261103T132152Z-06a741"


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
            env=subproc.child_env(),
        )
        return out.stdout.strip() or "unknown"
    except Exception as exc:  # noqa: BLE001
        return f"unknown({exc})"


def load_dev(gold_path: Path) -> list[dict]:
    """dev 调参集里的应答题。验收集一律不碰。"""
    data = json.loads(gold_path.read_text(encoding="utf-8"))
    items = data.get("items") or []
    picked = [
        it
        for it in items
        if str(it.get("split") or "") == "dev"
        and not it.get("should_abstain")
        and (it.get("gold_chunk_ids") or [])
    ]
    picked.sort(key=lambda it: str(it.get("question_id") or ""))
    return picked


def covered_of(answer: dict, item: dict) -> bool:
    """与 `scripts/ask.py:283-298` 同口径：单向子串，引用可以更长。"""
    emitted = {
        (str(c.get("chunk_id") or ""), str(c.get("evidence") or ""))
        for c in answer.get("citations") or []
    }
    for req in item.get("required_citations") or []:
        cid = str(req.get("chunk_id") or "")
        ev = str(req.get("evidence") or "")
        if not any(c == cid and ev in emitted_ev for c, emitted_ev in emitted):
            return False
    return True


def citation_health(answer: dict, chunks: dict) -> tuple[int, int]:
    """(总条数, 定位失败条数)。判据：`text[start:end]` 逐字等于 `evidence`。"""
    total = 0
    bad = 0
    for cite in answer.get("citations") or []:
        total += 1
        rec = chunks.get(str(cite.get("chunk_id") or ""))
        text = str((rec or {}).get("text") or "")
        start, end = int(cite.get("start", -1)), int(cite.get("end", -1))
        if not (rec is not None and 0 <= start < end <= len(text) and text[start:end] == cite["evidence"]):
            bad += 1
    return total, bad


def run_one(loaded: dict, items: list[dict], budget: int) -> list[dict]:
    """一档预算跑一遍全 dev，返回逐题行。"""
    pipe = pipeline.build_pipeline(loaded, top_k=20, graph_depth=1)
    retriever = pipe["retriever"]
    retriever.budget["max_context_tokens"] = int(budget)
    chunks = loaded["chunks"] or {}
    rows: list[dict] = []
    for it in items:
        t0 = time.perf_counter()
        res = pipeline.ask_once(pipe, it["question"])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        ans = res.get("answer") or {}
        text = str(ans.get("answer") or "")
        hits = [str(h.get("chunk_id") or "") for h in (res.get("search") or {}).get("hits") or []]
        gold = {str(c) for c in it.get("gold_chunk_ids") or []}
        in_ctx = {str(u.get("chunk_id") or "") for u in (res.get("context") or {}).get("units") or []}
        ctx_stats = (res.get("context") or {}).get("stats") or {}
        first_rank = next((i + 1 for i, cid in enumerate(hits) if cid in gold), None)
        cite_total, cite_bad = citation_health(ans, chunks)
        accepted = [str(a) for a in it.get("acceptable_answers") or []]
        gold_answer = str(it.get("answer") or "")
        cands = [gold_answer] + accepted
        hits_gold = bool(text.strip()) and any(c and c in text for c in cands)
        cov = covered_of(ans, it)
        rows.append(
            {
                "question_id": str(it.get("question_id") or ""),
                "category": str(it.get("category") or ""),
                "origin": str(it.get("origin") or ""),
                "gold_in_hits": bool(gold & set(hits)),
                "all_gold": bool(gold) and gold.issubset(in_ctx),
                "gold_first_rank": first_rank,
                "required_in_context": bool(gold) and gold.issubset(in_ctx),
                "any_gold": bool(gold) and bool(gold & in_ctx),
                "covered": cov,
                "answer_hits_gold": hits_gold,
                "correct_coverage": bool(cov and hits_gold),
                "abstained": not text.strip(),
                "citations": cite_total,
                "cite_bad": cite_bad,
                "tokens_used": int(ctx_stats.get("tokens_used") or 0),
                "units": len(in_ctx),
                "elapsed_ms": round(elapsed_ms, 3),
            }
        )
    return rows


def _pct(a: int, b: int) -> str:
    return f"{a}/{b} = {a / b:.4f}" if b else f"{a}/{b}"


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "n": n,
        "gold_in_hits": sum(r["gold_in_hits"] for r in rows),
        "any_gold": sum(r["any_gold"] for r in rows),
        "all_gold": sum(r["all_gold"] for r in rows),
        "required_in_context": sum(r["required_in_context"] for r in rows),
        "covered": sum(r["covered"] for r in rows),
        "answer_hits_gold": sum(r["answer_hits_gold"] for r in rows),
        "correct_coverage": sum(r["correct_coverage"] for r in rows),
        "abstained": sum(r["abstained"] for r in rows),
        "citations": sum(r["citations"] for r in rows),
        "cite_bad": sum(r["cite_bad"] for r in rows),
        "avg_units": round(sum(r["units"] for r in rows) / n, 3) if n else 0.0,
        "avg_tokens": round(sum(r["tokens_used"] for r in rows) / n, 1) if n else 0.0,
        "p50_ms": round(sorted(r["elapsed_ms"] for r in rows)[n // 2], 3) if n else 0.0,
    }


def delta(base: list[dict], cand: list[dict], key: str) -> dict:
    """逐题救 / 丢。两档的题序一致（都按 question_id 排过）。"""
    gain = [b["question_id"] for b, c in zip(base, cand) if c[key] and not b[key]]
    lose = [b["question_id"] for b, c in zip(base, cand) if b[key] and not c[key]]
    return {"gain": gain, "lose": lose, "net": len(gain) - len(lose)}


def by_category(rows: list[dict], key: str) -> dict:
    out: dict[str, int] = {}
    for r in rows:
        if r[key]:
            out[r["category"]] = out.get(r["category"], 0) + 1
    return dict(sorted(out.items()))


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        # 不 reconfigure 的话，下面打印的「⇒」会让进程以 1 退出，
        # 而这个 1 与「分析失败」长得一模一样。
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(description="预算 6000/7000/8000 三档真跑（dev 350 应答题）")
    ap.add_argument("--gold", default="data/goldset/gold-v2.json")
    ap.add_argument("--json", default="")
    ap.add_argument("--generation", default=GENERATION)
    args = ap.parse_args(argv)

    gold_path = ROOT / args.gold
    if not gold_path.exists():
        print(f"输入不可用：{args.gold} 不存在", file=sys.stderr)
        return 2
    items = load_dev(gold_path)
    if len(items) != 350:
        print(f"自检失败：dev 应答题应为 350，实际 {len(items)}", file=sys.stderr)
        return 4

    loaded = pipeline.load_generation(args.generation)
    chunks = loaded["chunks"] or {}
    if not chunks:
        print("输入不可用：世代块正文为空", file=sys.stderr)
        return 2

    started = time.time()
    per_budget: dict[int, list[dict]] = {}
    for b in BUDGETS:
        t0 = time.time()
        per_budget[b] = run_one(loaded, items, b)
        print(f"[跑完] 预算 {b}：{time.time() - t0:.1f}s", flush=True)

    base = per_budget[BASE_BUDGET]
    base_sum = summarize(base)

    print()
    print("== 1 三档汇总（dev 350 应答题，TITLE_ANSWER_PATH 取仓库默认）")
    # 表头必须与下面 print 的取值顺序逐列对齐：hit20=gold_in_hits（检索层）、
    # any=进上下文的至少一块、all=全部块进上下文。三个名字不许混写——
    # 第一版本节表头曾把 any 标成 all_gold，读到的人会以为「全部 gold 都进了」。
    hdr = f"{'预算':>6} {'hit20':>9} {'any':>7} {'all':>7} {'covered':>8} {'correct':>8} {'ansHit':>7} {'abst':>5} {'引用条':>7} {'坏条':>4} {'单元':>5} {'已用tok':>8} {'p50ms':>7}"
    print(hdr)
    for b in BUDGETS:
        s = summarize(per_budget[b])
        print(
            f"{b:>6} {s['gold_in_hits']:>9} {s['any_gold']:>7} {s['all_gold']:>7} {s['covered']:>8} "
            f"{s['correct_coverage']:>8} {s['answer_hits_gold']:>7} {s['abstained']:>5} "
            f"{s['citations']:>7} {s['cite_bad']:>4} {s['avg_units']:>5} {s['avg_tokens']:>8} {s['p50_ms']:>7}"
        )

    print()
    print("== 2 相对基线 6000 的逐题救 / 丢")
    for b in BUDGETS:
        if b == BASE_BUDGET:
            continue
        print(f"-- {BASE_BUDGET} -> {b}")
        for key in ("gold_in_hits", "any_gold", "all_gold", "covered", "correct_coverage", "answer_hits_gold", "abstained"):
            d = delta(base, per_budget[b], key)
            tail = ""
            if d["gain"] or d["lose"]:
                tail = f"   救 {d['gain'][:6]}{' …' if len(d['gain']) > 6 else ''} 丢 {d['lose']}"
            print(f"   {key:<20} 救 {len(d['gain']):>3} 丢 {len(d['lose']):>3} 净 {d['net']:+d}{tail}")

    print()
    print("== 3 覆盖 / 正确覆盖按类（只列有变化的）")
    for b in BUDGETS:
        if b == BASE_BUDGET:
            continue
        print(f"-- {b}")
        for key in ("covered", "correct_coverage"):
            a, c = by_category(base, key), by_category(per_budget[b], key)
            diffs = {k: (a.get(k, 0), c.get(k, 0), c.get(k, 0) - a.get(k, 0)) for k in set(a) | set(c)}
            moved = {k: v for k, v in sorted(diffs.items()) if v[2] != 0}
            print(f"   {key}: {moved if moved else '无变化'}")

    print()
    print("== 4 两道已知非单调题（预算边界掷骰子的活标本）")
    idx = {r["question_id"]: i for i, r in enumerate(base)}
    for qid in NON_MONOTONIC_WATCH:
        if qid not in idx:
            print(f"   {qid}: 不在 dev 应答题里")
            continue
        i = idx[qid]
        trail = " -> ".join(
            f"{b}:any={int(per_budget[b][i]['any_gold'])}/all={int(per_budget[b][i]['all_gold'])}"
            f"/rank={per_budget[b][i]['gold_first_rank']}/units={per_budget[b][i]['units']}"
            f"/tok={per_budget[b][i]['tokens_used']}"
            for b in BUDGETS
        )
        print(f"   {qid}（{base[i]['category']}）{trail}")

    print()
    print("== 5 frontier 9 题（基线 gold 名次 ≤5 却未进上下文）的收复情况")
    frontier = [
        r["question_id"]
        for r in base
        if r["gold_first_rank"] is not None
        and r["gold_first_rank"] <= 5
        and not r["any_gold"]
    ]
    print(f"   基线 frontier 题数 {len(frontier)}：{frontier}")
    for b in BUDGETS:
        got = [q for q in frontier if per_budget[b][idx[q]]["any_gold"]]
        print(f"   预算 {b}: 收复 {len(got)}/{len(frontier)}   {got if got else ''}")

    print()
    print("== 6 判据（预登记，见模块 docstring）")
    cand = per_budget[8000]
    cand_sum = summarize(cand)
    k1 = delta(base, cand, "correct_coverage")
    k2 = delta(base, cand, "covered")
    frontier_8000 = [q for q in frontier if cand[idx[q]]["any_gold"]]
    r2 = cand_sum["abstained"] - base_sum["abstained"]
    print(f"   K1 correct_coverage 净增 {k1['net']:+d}（门槛 ≥ +10）-> {'PASS' if k1['net'] >= 10 else 'FAIL'}")
    print(f"   K2 covered 净增 {k2['net']:+d}（门槛 ≥ +10）-> {'PASS' if k2['net'] >= 10 else 'FAIL'}")
    if not frontier:
        k3_verdict = "N/A（frontier 空集，判据不成立，不许记 PASS）"
    else:
        k3_verdict = "PASS" if len(frontier_8000) >= 6 else "FAIL"
    print(f"   K3 frontier 收复 {len(frontier_8000)}/{len(frontier)}（门槛 ≥ 6）-> {k3_verdict}")
    print(f"   R1 correct_coverage 丢失 {len(k1['lose'])} 题 {k1['lose']} -> {'触发（撤）' if k1['lose'] else '未触发'}")
    print(f"   R2 拒答变化 {r2:+d} -> {'触发（撤）' if r2 > 0 else '未触发'}")
    print(f"   R3 定位失败条数 {cand_sum['cite_bad']} -> {'触发（撤）' if cand_sum['cite_bad'] else '未触发'}")
    print(f"   分母并列：covered {_pct(cand_sum['covered'], cand_sum['n'])}（/350）"
          f"；correct {_pct(cand_sum['correct_coverage'], cand_sum['n'])}（/350）")

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_revision(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "command": "py -3 scripts/lead_budget_realrun.py",
        "generation": args.generation,
        "gold_set": args.gold,
        "budgets": list(BUDGETS),
        "base_budget": BASE_BUDGET,
        "wall_seconds": round(time.time() - started, 2),
        "summary": {str(b): summarize(per_budget[b]) for b in BUDGETS},
        "delta_vs_base": {
            str(b): {
                key: delta(base, per_budget[b], key)
                for key in ("any_gold", "all_gold", "covered", "correct_coverage", "answer_hits_gold", "abstained")
            }
            for b in BUDGETS
            if b != BASE_BUDGET
        },
        "by_category": {
            str(b): {key: by_category(per_budget[b], key) for key in ("covered", "correct_coverage")}
            for b in BUDGETS
        },
        "criteria": {
            "K1_correct_coverage_net": delta(base, per_budget[BUDGETS[-1]], "correct_coverage")["net"],
            "K1_threshold": 10,
            "K1_verdict": "PASS" if delta(base, per_budget[BUDGETS[-1]], "correct_coverage")["net"] >= 10 else "FAIL",
            "K2_covered_net": delta(base, per_budget[BUDGETS[-1]], "covered")["net"],
            "K2_threshold": 10,
            "K2_verdict": "PASS" if delta(base, per_budget[BUDGETS[-1]], "covered")["net"] >= 10 else "FAIL",
            "K3_frontier_recovered": len(frontier_8000),
            "K3_frontier_total": len(frontier),
            "K3_threshold": 6,
            "K3_verdict": k3_verdict,
            "R1_correct_lost": delta(base, per_budget[BUDGETS[-1]], "correct_coverage")["lose"],
            "R2_abstained_delta": summarize(per_budget[BUDGETS[-1]])["abstained"] - summarize(base)["abstained"],
            "R3_cite_bad": summarize(per_budget[BUDGETS[-1]])["cite_bad"],
        },
        "frontier_at_base": frontier,
        "frontier_recovered": {str(b): [q for q in frontier if per_budget[b][idx[q]]["any_gold"]] for b in BUDGETS},
        "non_monotonic_watch": {
            q: [{str(b): per_budget[b][idx[q]]} for b in BUDGETS] if q in idx else "absent"
            for q in NON_MONOTONIC_WATCH
        },
        "per_question": {str(b): per_budget[b] for b in BUDGETS},
    }
    prov = provenance.collect(
        generation_id=args.generation,
        dataset=provenance.dataset_block(gold_path=ROOT / args.gold),
        inputs=provenance.inputs_block(
            [ROOT / args.gold, ROOT / "data" / "goldset" / "split-v2.json"]
        ),
        generated_at=payload["generated_at"],
    )
    payload["provenance_block"] = prov
    payload["provenance"] = {
        "seven_fields": provenance.verify(prov).get("verdict"),
        "detail": provenance.verify(prov),
    }

    if args.json:
        out = ROOT / args.json
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print()
        print(f"七字段 {payload['provenance']['seven_fields']}；JSON -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
