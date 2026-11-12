"""L2 实测：`chapter_rank` 开关的**端到端**读数（不是重放上界）。

## 为什么要单独跑一遍

`scripts/diag_cx_chapter_gap.py` 阶段 B 是**重放**：换排序键、按新章节拼答案串、
再把串喂给判据。它证明"排序键选得对"，但**没有换引用** —— 真实通路会按新章节
重发引用，`verify_citations` 可能剔除某些引用、甚至把答案清空改判拒答
（`q-cx-900024` 记录在案的事故形态）。

所以这里用 `pipeline.ask_once` 真跑，`options` 只差 `chapter_rank` 一个键：

    A 基线   = 生产档（`PROD_PROFILE`，cap=4，rank="score"）
    B 实验   = 生产档 + `chapter_rank="focus_assertions"`

两份读数逐题配对，给出 `+ / −` 与逐题翻转清单。

## 判据与口径（一条都不动）

`judge.judge_answer` 原样调用；分母按 `judge.SCORABLE_CATEGORIES` 剔结构类；
split 分开报，不合并。

## 红线

* 只读 `src/`，不写 `src/`；产物 `data/diag_l2_chapter_rank.json`。
* `PROD_PROFILE` 一个字都不改：B 档只在 `options` 副本上加一个键。
* 一次 `load_generation()`，两档共用同一条 `pipe`。

用法：

    python scripts/run_guarded.py --max-commit-pct 92 --min-avail-mb 2048 -- \\
        python scripts/diag_l2_chapter_rank.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import answer as answer_mod  # noqa: E402
from arkrag import judge as judge_mod  # noqa: E402
from arkrag import pipeline as pipeline_mod  # noqa: E402

GEN = "g-20261103T132152Z-06a741"
GOLD = ROOT / "data" / "goldset" / "gold-v2.json"
GOLD_V1 = ROOT / "data" / "goldset" / "gold-v1.json"
OUT = ROOT / "data" / "diag_l2_chapter_rank.json"
TXT = ROOT / "data" / "diag_l2_chapter_rank.txt"

SPLITS = ("dev", "holdout", "clean")
ARMS = {
    "A_base": {},                                       # 生产档原样
    "B_focus": {"chapter_rank": "focus_assertions"},    # + 换排序键
}


def load_v1_ids() -> set[str]:
    if not GOLD_V1.exists():
        return set()
    doc = json.loads(GOLD_V1.read_text("utf-8"))
    items = doc.get("items") if isinstance(doc, dict) else doc
    return {str(it.get("question_id")) for it in items or []}


def pick(items: list[dict], split: str, v1: set[str]) -> list[dict]:
    if split == "clean":
        return [it for it in items if it.get("split") == "holdout"
                and it.get("gold_chunk_ids")
                and str(it.get("question_id")) not in v1]
    return [it for it in items if it.get("split") == split and it.get("gold_chunk_ids")]


def main() -> int:
    doc = json.loads(GOLD.read_text("utf-8"))
    items = doc.get("items") if isinstance(doc, dict) else doc
    v1 = load_v1_ids()

    loaded = pipeline_mod.load_generation(GEN)
    chunks = loaded["chunks"]
    ct: dict[str, list[str]] = {}
    for rec in chunks.values():
        ct.setdefault(str(rec.get("chapter") or ""), []).append(str(rec.get("text") or ""))
    chap_texts = {k: "\n".join(v) for k, v in ct.items()}

    pipe = pipeline_mod.build_pipeline(loaded, top_k=20, graph_depth=1)
    pipe["generation_id"] = GEN
    pipe["entities"] = loaded["graph"].get("entities") or []
    base_options = pipeline_mod.apply_prod_profile(pipe)

    out: dict = {"generation_id": GEN, "prod_profile": dict(pipeline_mod.PROD_PROFILE),
                 "arms": {}, "splits": {}}
    lines: list[str] = []
    p = lines.append
    p("=" * 92)
    p(f"L2 端到端实测：chapter_rank 开关   世代 {GEN}")
    p(f"生产档 {json.dumps(pipeline_mod.PROD_PROFILE, ensure_ascii=False)}")
    p("=" * 92)

    for split in SPLITS:
        picked = pick(items, split, v1)
        rows: list[dict] = []
        for it in picked:
            q = str(it.get("question") or "")
            cat = str(it.get("category") or "")
            rec = {"question_id": it.get("question_id"), "category": cat,
                   "split": it.get("split"), "arms": {}}
            for arm, extra in ARMS.items():
                opts = dict(base_options)
                opts.update(extra)
                res = pipeline_mod.ask_once(pipe, q, options=opts)
                ans = res.get("answer") or {}
                verified = answer_mod.verify_citations(ans, chunks)
                text = str(verified.get("answer") or "")
                v = judge_mod.judge_answer(cat, text, it, question=q,
                                           chapter_texts=chap_texts)
                rec["arms"][arm] = {
                    "pass": v.get("pass") is True,
                    "reason": v.get("reason"),
                    "answer": text,
                    "n_citations": len(verified.get("citations") or []),
                    "abstained": bool(verified.get("abstained")),
                    "chapter_path": ((ans.get("stats") or {}).get("chapter_path") or {}),
                }
            rows.append(rec)
        out["splits"][split] = rows

    # ---- 汇总 ----
    for split in SPLITS:
        rows = out["splits"][split]
        scorable = [r for r in rows if r["category"] in judge_mod.SCORABLE_CATEGORIES]
        per_arm = {}
        for arm in ARMS:
            per_arm[arm] = {
                "n": len(scorable),
                "pass": sum(1 for r in scorable if r["arms"][arm]["pass"]),
                "cx_n": sum(1 for r in scorable if r["category"] == "cross_chunk"),
                "cx_pass": sum(1 for r in scorable
                               if r["category"] == "cross_chunk" and r["arms"][arm]["pass"]),
            }
        gain = [r["question_id"] for r in scorable
                if r["arms"]["B_focus"]["pass"] and not r["arms"]["A_base"]["pass"]]
        loss = [r["question_id"] for r in scorable
                if r["arms"]["A_base"]["pass"] and not r["arms"]["B_focus"]["pass"]]
        out["arms"][split] = {"per_arm": per_arm, "gain": gain, "loss": loss}
        p("")
        p(f"[{split}]  可判 {per_arm['A_base']['n']}")
        for arm in ARMS:
            e = per_arm[arm]
            p(f"   {arm:<10} {e['pass']}/{e['n']} = {e['pass'] / e['n'] * 100:.2f}%   "
              f"cross_chunk {e['cx_pass']}/{e['cx_n']}")
        p(f"   +{len(gain)} / -{len(loss)}   gain={gain}  loss={loss}")
        for qid in gain + loss:
            r = next(x for x in rows if x["question_id"] == qid)
            p(f"     {qid}  A={r['arms']['A_base']['answer']!r} "
              f"({r['arms']['A_base']['reason']})")
            p(f"     {' ' * len(qid)}  B={r['arms']['B_focus']['answer']!r} "
              f"({r['arms']['B_focus']['reason']})")

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    print("\n".join(lines))
    print(f"\n-> {OUT}")
    print(f"-> {TXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
