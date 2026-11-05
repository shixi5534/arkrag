"""P3 第三轮方案稿的实测底数（只测量，不改任何生产代码、不动 gold）。

为什么要单独跑一遍
------------------
`scripts/a_p3_round3_prelim.py` 只留了闸门分区与画像，没有留逐题答案。写方案时三个靶子
都要「预期影响以题数说」，而**题数只能来自当前答案对不对**，所以这里补一次带答案的全量
跑（dev 380，约 40 s），把下列底数一次性落在 JSON 里：

1. 八类 × origin（v1/new）的闸门分布——回答「闸门1 的 120 题里 new 批占多少」；
2. 八类 × origin 的 `answer_hits_gold` 现状——靶子的**可救上界只能落在这批为假的题里**；
3. 三个靶子各自的「当前为假且上界可救」的题数（靶子一 cross_chunk 章节包、
   靶子二 multihop 中间实体、靶子三 alias 只分类不修）。

口径纪律（与全项目一致）
------------------------
* 只跑 `split == "dev"`（380 题）；**new-holdout 160 一题不碰**。
* 应答题 350 题 = `split=="dev"` ∧ 非 `should_abstain` ∧ `gold_chunk_ids` 非空。
* `answer_hits_gold` 用 `ask.py:283-298` 同一套 `acceptable_answers` 判据（本脚本
  `answer_hits_gold()` 逐字照抄），不另立志。
* `gold_set` 输出走 `config.rel_to_root()`，产物里 0 处绝对机台路径。

退出码
------
0 跑通；2 输入不可用；4 自检失败（自检：闸门分区必须闭合到应答题总数）。
"""

from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arkrag import config  # noqa: E402
from arkrag import pipeline  # noqa: E402

DEFAULT_GENERATION = "g-20261103T132152Z-06a741"
DEFAULT_GOLD = ROOT / "data" / "goldset" / "gold-v2.json"


def rel(path: Path) -> str:
    """机台路径 → 仓库相对路径（入库 JSON 不许出现 `E:\\...`）。"""
    try:
        return config.rel_to_root(str(path))
    except Exception:
        return str(path).replace(os.sep, "/").split("图rag/")[-1]


def git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True,
            text=True, encoding="utf-8", timeout=20, check=False,
            env=os.environ.copy(),
        ).stdout.strip()
    except Exception:
        return ""


def load_dev(gold_path: Path) -> list[dict]:
    data = json.loads(gold_path.read_text(encoding="utf-8"))
    return [it for it in data["items"] if it.get("split") == "dev"]


def answer_hits_gold(item: dict, answer: str) -> bool:
    """与 `scripts/ask.py` 的答案判据同口径：任一 acceptable_answer 出现在答案里。"""
    if not answer:
        return False
    cands = [str(item.get("answer") or "")]
    cands += [str(x) for x in (item.get("acceptable_answers") or [])]
    return any(c and c in answer for c in cands)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P3 第三轮方案稿的实测底数（只测量）")
    ap.add_argument("--generation", default=DEFAULT_GENERATION)
    ap.add_argument("--gold-set", default=str(DEFAULT_GOLD))
    ap.add_argument("--out-json", default=str(ROOT / "data" / "a_p3_round3_plan.json"))
    args = ap.parse_args(argv)

    gold_path = Path(args.gold_set)
    if not gold_path.is_absolute():
        gold_path = ROOT / gold_path
    if not gold_path.exists():
        print(f"[2] gold set 不可用：{gold_path}", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    loaded = pipeline.load_generation(args.generation)
    chunks = loaded["chunks"]
    pipe = pipeline.build_pipeline(loaded, top_k=20, graph_depth=1)
    items = load_dev(gold_path)
    answerable = [
        it for it in items
        if (not it.get("should_abstain")) and it.get("gold_chunk_ids")
    ]
    print(f"[load] dev={len(items)} 应答题={len(answerable)} 块={len(chunks)}")

    rows = []
    for it in answerable:
        out = pipeline.ask_once(pipe, str(it.get("question") or ""))
        ctx = out.get("context") or {}
        ans = out.get("answer") or {}
        ctx_ids = {str(u.get("chunk_id") or "") for u in ctx.get("units") or []}
        reqs = it.get("required_citations") or []
        emitted = {(str(c.get("chunk_id") or ""), str(c.get("evidence") or ""))
                   for c in ans.get("citations") or []}
        covered = bool(reqs) and all(
            any(cid == str(r.get("chunk_id") or "") and str(r.get("evidence") or "") in ev
                for cid, ev in emitted) for r in reqs)
        rows.append({
            "question_id": it["question_id"],
            "category": str(it.get("category") or ""),
            "origin": str(it.get("origin") or ""),
            "gold_answer": str(it.get("answer") or ""),
            "n_required": len(reqs),
            "covered": covered,
            "required_in_context": bool(reqs) and all(
                str(r.get("chunk_id") or "") in ctx_ids for r in reqs),
            "answer": str(ans.get("answer") or ""),
            "abstained": bool(ans.get("abstained")),
            "answer_hits_gold": answer_hits_gold(it, str(ans.get("answer") or "")),
            "ctx_ids": sorted(ctx_ids),
            "units": len(ctx.get("units") or []),
        })
    print(f"[run] 真管线 {len(rows)} 题 {time.perf_counter()-t0:.1f}s")

    # 闸门分区必须闭合
    n_cover = sum(1 for r in rows if r["covered"])
    n_ans_hit = sum(1 for r in rows if r["answer_hits_gold"])
    if n_cover + sum(1 for r in rows if not r["covered"]) != len(rows):
        print("[4] 闸门分区不闭合", file=sys.stderr)
        return 4

    # 八类 × origin：答案质量现状（靶子上界只能落在这里为假的题里）
    by_cat: dict[str, dict] = {}
    for cat in sorted({r["category"] for r in rows}):
        sub = [r for r in rows if r["category"] == cat]
        by_cat[cat] = {
            "n": len(sub),
            "origin": dict(collections.Counter(r["origin"] for r in sub)),
            "covered": sum(1 for r in sub if r["covered"]),
            "answer_hits_gold": sum(1 for r in sub if r["answer_hits_gold"]),
            "abstained": sum(1 for r in sub if r["abstained"]),
            "answer_hits_gold_by_origin": dict(collections.Counter(
                r["origin"] for r in sub if r["answer_hits_gold"])),
        }

    # 靶子一 / 二 的「当前为假」题号（可救上界必须从这里扣）
    def misses(cat: str) -> list[str]:
        return [r["question_id"] for r in rows
                if r["category"] == cat and not r["answer_hits_gold"]]

    doc = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_rev(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "command": "py -3 scripts/a_p3_round3_plan.py",
        "generation": args.generation,
        "gold_set": rel(gold_path),
        "split_policy": "pre-split-diagnostic/dev-only：dev 结论不外推到 new-holdout 160",
        "counts": {
            "dev": len(items),
            "answerable": len(answerable),
            "covered": n_cover,
            "answer_hits_gold": n_ans_hit,
        },
        "by_category": by_cat,
        "misses": {cat: misses(cat) for cat in by_cat},
        "per_question": [{k: v for k, v in r.items() if k != "ctx_ids"} for r in rows],
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n[八类 × origin] 答案质量现状（answer_hits_gold）")
    for cat, v in by_cat.items():
        print(f"  {cat:16s} n={v['n']:3d} origin={v['origin']} "
              f"covered={v['covered']:3d} hits={v['answer_hits_gold']:3d} "
              f"abstain={v['abstained']:3d}")
    print(f"\n总计 covered={n_cover}/{len(rows)} "
          f"answer_hits_gold={n_ans_hit}/{len(rows)}")
    print(f"输出: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
