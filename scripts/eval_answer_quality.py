"""按题型判据评测「答对没有」——`arkrag.judge` 的唯一消费端。

## 这个脚本为什么必须存在
`src/arkrag/judge.py` 定稿后，如果没人调它，就只是又一份"写在库里没人用"的口径。
本项目已经因为"判据没接线等于不存在"栽过三次（`TITLE_ANSWER_PATH`、`CHAPTER_ANSWER_PATH`
都默认 False 而台账写着"已入库"）。所以这个脚本是判据的唯一生产消费端：
**除它以外不许再有任何地方自己写一份"答案算不算对"。**

## 三个分母必须并列
- `scorable`：剔掉结构类（`time_conflict` / `negative_abstain` / `adversarial`）后的分母。
- `all`：把结构类也算进分母（它们在判据里是 `None`，这里按"没答对"计入）。
两个数一起报，因为只看 scorable 会高估 8–15 个百分点。

## split 分开报，不合并
`dev` 上的任何阈值/口径调整都不能用 `holdout` 的数字去挑；`holdout` 里还有与
`gold-v1` 重复的题，所以额外给 `clean`（剔除重复题后的）一栏。

用法：
    py -3 scripts/eval_answer_quality.py --split dev
    py -3 scripts/eval_answer_quality.py --split holdout --json data/eval_quality_holdout.json
    py -3 scripts/eval_answer_quality.py --split clean --token-budget 6000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import config  # noqa: E402
from arkrag import judge as judge_mod  # noqa: E402
from arkrag import pipeline as pipeline_mod  # noqa: E402


def chapter_texts(chunks: dict) -> dict[str, str]:
    """`{章节代号: 该章全部块正文拼接}`，供 `judge_answer` 校验"额外章节是否真的提到实体"。

    只从本世代自带的块建（`load_generation` 已用 `resolve_chunks` 证明这些块属于本代），
    不许读全局 `data/arkrag/chunks.jsonl` —— 两者块数不同（13,798 vs 13,786），
    混用会让"这个章节里到底有没有这个名字"判错。
    """
    out: dict[str, list[str]] = {}
    for rec in chunks.values():
        ch = str(rec.get("chapter") or "")
        out.setdefault(ch, []).append(str(rec.get("text") or ""))
    return {k: "\n".join(v) for k, v in out.items()}


def pick_items(items: list[dict], split: str, v1_ids: set[str]) -> list[dict]:
    """按 split 选题。`clean` = holdout 里且不在 gold-v1 里。"""
    if split == "all":
        pool = [it for it in items if it.get("gold_chunk_ids")]
    elif split == "clean":
        pool = [it for it in items
                if it.get("split") == "holdout" and it.get("gold_chunk_ids")
                and str(it.get("question_id")) not in v1_ids]
    else:
        pool = [it for it in items if it.get("split") == split and it.get("gold_chunk_ids")]
    return pool


def load_v1_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    doc = json.loads(path.read_text("utf-8"))
    items = doc.get("items") if isinstance(doc, dict) else doc
    return {str(it.get("question_id")) for it in items or []}


def tally(rows: list[dict], arms: list[str]) -> dict:
    """按类别汇总，同时给 scorable 与 all 两个分母。"""
    per_cat: dict[str, dict] = {}
    for cat in judge_mod.CATEGORIES:
        sub = [r for r in rows if r["category"] == cat]
        if not sub:
            continue
        entry = {"n": len(sub), "scorable": cat in judge_mod.SCORABLE_CATEGORIES,
                 "arms": {}}
        for arm in arms:
            if cat in judge_mod.STRUCTURAL_CATEGORIES:
                entry["arms"][arm] = {"pass": None,
                                      "opened": sum(1 for r in sub if r["arms"][arm]["answered"])}
            else:
                entry["arms"][arm] = {"pass": sum(1 for r in sub if r["arms"][arm]["pass"]),
                                      "opened": sum(1 for r in sub if r["arms"][arm]["answered"])}
        per_cat[cat] = entry

    totals = {}
    for arm in arms:
        scorable = [r for r in rows if r["category"] in judge_mod.SCORABLE_CATEGORIES]
        ok_s = sum(1 for r in scorable if r["arms"][arm]["pass"])
        opened = sum(1 for r in rows if r["arms"][arm]["answered"])
        totals[arm] = {
            "scorable_n": len(scorable),
            "scorable_pass": ok_s,
            "scorable_rate": round(ok_s / len(scorable), 6) if scorable else 0.0,
            "all_n": len(rows),
            "all_rate": round(ok_s / len(rows), 6) if rows else 0.0,
            "opened": opened,
            "opened_rate": round(opened / len(rows), 6) if rows else 0.0,
        }
    return {"per_category": per_cat, "totals": totals}


def run(generation_id: str, gold_path: Path, *, split: str, limit: int,
        token_budget: int | None, prod: bool = False) -> dict:
    """跑一份 split。

    `prod=True` 时走 `pipeline.PROD_PROFILE`（生产档：引文定位 + 桥接 + 图证据预留
    + 24000 预算）。**默认 `False`，保证历史读数逐格可复现**——台账里 6000/12000 档
    的每一次对比都是在这个默认分支下测的，改成默认开生产档会让全部历史数字失去可比性。
    """
    loaded = pipeline_mod.load_generation(generation_id)
    pipe = pipeline_mod.build_pipeline(loaded, top_k=20, graph_depth=1)
    pipe["generation_id"] = generation_id
    pipe["entities"] = loaded["graph"].get("entities") or []
    options: dict | None = None
    if prod:
        options = pipeline_mod.apply_prod_profile(pipe)
    budget = int(token_budget) if token_budget is not None else \
        int(pipe["retriever"].budget["max_context_tokens"])
    pipe["retriever"].budget["max_context_tokens"] = budget

    doc = json.loads(gold_path.read_text("utf-8"))
    items = doc.get("items") if isinstance(doc, dict) else doc
    v1_ids = load_v1_ids(ROOT / "data" / "goldset" / "gold-v1.json")
    picked = pick_items(items, split, v1_ids)
    if limit:
        picked = picked[:limit]
    chap = chapter_texts(loaded["chunks"])

    rows: list[dict] = []
    t0 = time.perf_counter()
    for it in picked:
        q = str(it.get("question") or "")
        res = pipeline_mod.ask_once(pipe, q, options=options)
        ans = res["answer"] or {}
        text = str(ans.get("answer") or "")
        verdict = judge_mod.judge_answer(
            str(it.get("category") or ""), text, it, question=q, chapter_texts=chap)
        cites = judge_mod.citations_support(ans, loaded["chunks"])
        rows.append({
            "question_id": it.get("question_id"),
            "category": str(it.get("category") or ""),
            "split": it.get("split"),
            "question": q,
            "answer": text,
            "abstained": bool(ans.get("abstained")),
            "judge": verdict,
            "citations_ok": cites["ok"],
            "n_citations": cites["n_cites"],
            "arms": {"prod": {"pass": verdict["pass"] is True,
                              "reason": verdict["reason"],
                              "answered": bool(text.strip())}},
        })
    elapsed = (time.perf_counter() - t0) * 1000.0
    report = {
        "generation_id": generation_id,
        "gold_set": str(gold_path),
        "split": split,
        "token_budget": budget,
        "prod_profile": bool(prod),
        "prod_options": dict(options or {}),
        "n_items": len(rows),
        "elapsed_ms": round(elapsed, 1),
        "per_item_ms": round(elapsed / len(rows), 1) if rows else 0.0,
        "judge_schema": "arkrag.judge/1",
        "chapters_seen": len(chap),
        **(tally(rows, ["prod"])),
        "rows": rows,
    }
    return report


def print_report(rep: dict) -> None:
    print("=" * 84)
    print(f"按题型判据评测  世代 {rep['generation_id']}  split={rep['split']}"
          f"  budget={rep['token_budget']}  n={rep['n_items']}"
          f"{'  [生产档]' if rep.get('prod_profile') else ''}")
    print(f"平均 {rep['per_item_ms']} ms/题")
    print("-" * 84)
    print(f"{'类别':<18}{'n':>4}{'答对':>8}{'开口':>8}   说明")
    for cat, e in rep["per_category"].items():
        a = e["arms"]["prod"]
        if a["pass"] is None:
            print(f"{cat:<18}{e['n']:>4}{'-':>8}{a['opened']:>8}   结构类：拒判（剔分母）")
        else:
            print(f"{cat:<18}{e['n']:>4}{a['pass']:>8}{a['opened']:>8}   ")
    print("-" * 84)
    t = rep["totals"]["prod"]
    print(f"可判分母 {t['scorable_pass']}/{t['scorable_n']} = {t['scorable_rate'] * 100:.2f}%")
    print(f"全分数  {t['scorable_pass']}/{t['all_n']} = {t['all_rate'] * 100:.2f}%"
          f"（结构类按没答对计入）")
    print(f"开口率  {t['opened']}/{t['all_n']} = {t['opened_rate'] * 100:.2f}%")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="按题型判据评测答对率（arkrag.judge 唯一消费端）")
    ap.add_argument("--generation", default="g-20261103T132152Z-06a741")
    ap.add_argument("--gold-set", default=str(ROOT / "data" / "goldset" / "gold-v2.json"))
    ap.add_argument("--split", default="dev", choices=("dev", "holdout", "clean", "all"))
    ap.add_argument("--limit", type=int, default=0, help="0 表示不截断")
    ap.add_argument("--token-budget", type=int, default=None,
                    help="不传则吃 retrieve.DEFAULT_BUDGET（现 12000）；"
                         "复现 6000 档读数必须显式传 6000。"
                         "带 --prod 时不传则吃 pipeline.PROD_PROFILE（现 24000）")
    ap.add_argument("--prod", action="store_true",
                    help="按生产档跑（引文定位 + 桥接 + 图证据预留 + 预算取 PROD_PROFILE）；"
                         "不传则是历史默认档，6000/12000 档的旧读数靠它复现")
    ap.add_argument("--json", default="", help="报告落盘路径")
    args = ap.parse_args(argv)

    rep = run(args.generation, Path(args.gold_set), split=args.split,
              limit=args.limit, token_budget=args.token_budget, prod=args.prod)
    print_report(rep)
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n落盘 {out}（{out.stat().st_size} B）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
