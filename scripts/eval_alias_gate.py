"""L3 · 量化 `alias_answer` 开关的端到端效应（只读测量，不改任何实现默认值）。

## 口径（写死，三段都必须报）

1. **基线**：生产档（`PROD_PROFILE`），`alias_answer` 不开。必须复现
   dev 231/285、holdout 154/195、clean 98/120 与开口率 dev 347/350。
   复现不了就说明本轮改动污染了默认路径 —— 那是最高优先级的失败。
2. **实验**：生产档 + `options["alias_answer"] = True`。**唯一差异是这个键**。
3. **逐题差**：两臂的每题 `(pass, answered)` 逐题比对，给出
   `+题 / −题 / 开口 +题`，并列出每一道翻转的题号与答案。

## 反证条件（不满足就不许报正收益）

- **R1 默认路径逐位不变**：基线三份 split 的可判通过数与开口数必须与
  `data/diag_rank_24000`/`tmp_prod_*` 登记值完全一致。
- **R2 不许把拒答率整体拉低换分**：`should_abstain=True` 的题里，**正确拒答数
  不许下降**（A1）。新开口的题必须逐题列出，由人判断是不是假答。
- **R3 引用硬失败为 0**：新通路发出的每条引用必须 `text[start:end] == evidence`。

用法（必须走熔断器）：

    py -3.14 scripts/run_guarded.py --max-commit-pct 92 --min-avail-mb 2048 \
        --stdout _l3e_out.txt --stderr _l3e_err.txt \
        -- py -3.14 scripts/eval_alias_gate.py
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import answer as answer_mod  # noqa: E402
from arkrag import judge as judge_mod  # noqa: E402
from arkrag import pipeline as pipeline_mod  # noqa: E402

GENERATION = "g-20261103T132152Z-06a741"
GOLD_V2 = ROOT / "data" / "goldset" / "gold-v2.json"
GOLD_V1 = ROOT / "data" / "goldset" / "gold-v1.json"
DEFAULT_OUT = ROOT / "data" / "eval_alias_gate.json"

#: R1 登记值（`docs/计划-W0到W4-检索宽度解耦与决策兑现-20261108.md` 与
#: `tmp_prod_{dev,holdout}.json` 的 `totals.prod`）。**写死，不现场算**：
#: 现场算出来的"基线"与它自己比永远相等，那就不是一条判据。
BASELINE = {
    "dev": {"scorable_pass": 231, "scorable_n": 285, "opened": 347, "all_n": 350},
    "holdout": {"scorable_pass": 154, "scorable_n": 195, "opened": 233, "all_n": 238},
    "clean": {"scorable_pass": 98, "scorable_n": 120, "opened": None, "all_n": 144},
}


def log(*a) -> None:
    print(*a, flush=True)


def head(t: str) -> None:
    log("\n" + "=" * 100)
    log(t)
    log("=" * 100)


def pick_items(items: list[dict], split: str, v1_ids: set[str]) -> list[dict]:
    """逐字对齐 `scripts/eval_answer_quality.py::pick_items`（见 `diag_alias_abstain`）。"""
    if split == "clean":
        return [it for it in items
                if it.get("split") == "holdout" and it.get("gold_chunk_ids")
                and str(it.get("question_id")) not in v1_ids]
    if split == "all":
        return [it for it in items if it.get("gold_chunk_ids")]
    return [it for it in items
            if it.get("split") == split and it.get("gold_chunk_ids")]


def load_v1_ids() -> set[str]:
    if not GOLD_V1.exists():
        return set()
    doc = json.loads(GOLD_V1.read_text("utf-8"))
    items = doc.get("items") if isinstance(doc, dict) else doc
    return {str(it.get("question_id")) for it in items or []}


def chapter_texts(chunks: dict) -> dict[str, str]:
    out: dict[str, list[str]] = {}
    for rec in chunks.values():
        out.setdefault(str(rec.get("chapter") or ""), []).append(str(rec.get("text") or ""))
    return {k: "\n".join(v) for k, v in out.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="L3：alias_answer 开关的端到端效应")
    ap.add_argument("--generation", default=GENERATION)
    ap.add_argument("--gold-set", default=str(GOLD_V2))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--splits", default="dev,holdout,clean")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    gold = json.loads(Path(args.gold_set).read_text("utf-8"))
    items = list(gold.get("items") if isinstance(gold, dict) else gold)
    v1_ids = load_v1_ids()

    t0 = time.time()
    loaded = pipeline_mod.load_generation(args.generation)
    log(f"[load] {time.time() - t0:.1f}s chunks={len(loaded['chunks'])}")
    pipe = pipeline_mod.build_pipeline(loaded, top_k=20, graph_depth=1)
    pipe["generation_id"] = args.generation
    pipe["entities"] = loaded["graph"].get("entities") or []
    base_options = pipeline_mod.apply_prod_profile(pipe)
    chap = chapter_texts(loaded["chunks"])
    scorable = judge_mod.SCORABLE_CATEGORIES

    def run_arm(alias_on: bool) -> dict[str, dict]:
        opts = dict(base_options)
        if alias_on:
            opts["alias_answer"] = True
        else:
            opts.pop("alias_answer", None)
        out: dict[str, dict] = {}
        n = 0
        for sp in splits:
            for it in pick_items(items, sp, v1_ids):
                qid = str(it.get("question_id") or "")
                if args.limit and n >= args.limit:
                    break
                q = str(it.get("question") or "")
                res = pipeline_mod.ask_once(pipe, q, options=opts)
                ans = dict(res.get("answer") or {})
                ver = answer_mod.verify_citations(ans, loaded["chunks"])
                text = str(ver.get("answer") or "")
                verdict = judge_mod.judge_answer(
                    str(it.get("category") or ""), text, it, question=q,
                    chapter_texts=chap)
                cites = judge_mod.citations_support(ver, loaded["chunks"])
                key = f"{sp}:{qid}"
                if key in out:  # clean ⊂ holdout：同一题按 (split,qid) 各记一条
                    continue
                out[key] = {
                    "question_id": qid, "split": sp,
                    "category": str(it.get("category") or ""),
                    "should_abstain_gold": bool(it.get("should_abstain")),
                    "answer": text,
                    "answered": bool(text.strip()),
                    "pass": verdict["pass"] if verdict["pass"] is not None
                    else "structural",
                    "reason": verdict["reason"],
                    "n_citations": int(cites.get("n_cites") or 0),
                    "cites_ok": bool(cites.get("ok")),
                    "abstain_reason": str(ver.get("abstain_reason") or ""),
                    "alias_applied": bool(
                        ((ver.get("stats") or {}).get("alias_path") or {}).get("applied")),
                    "title_applied": bool(
                        ((ver.get("stats") or {}).get("title_path") or {}).get("applied")),
                    "cite_rejected": int(
                        (ver.get("verification") or {}).get("rejected") or 0),
                }
                n += 1
            if args.limit and n >= args.limit:
                break
        return out

    head("§1 基线臂（生产档，alias_answer 不开）")
    base = run_arm(False)
    head("§2 实验臂（生产档 + alias_answer=True）")
    exp = run_arm(True)

    def tally(arm: dict[str, dict]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for sp in splits:
            rows = [r for k, r in arm.items() if r["split"] == sp]
            sc = [r for r in rows if r["category"] in scorable]
            out[sp] = {
                "scorable_pass": sum(1 for r in sc if r["pass"] is True),
                "scorable_n": len(sc),
                "opened": sum(1 for r in rows if r["answered"]),
                "all_n": len(rows),
            }
        return out

    tb, te = tally(base), tally(exp)

    head("§3 R1 默认路径复现（基线臂 vs 登记值；不许有差异）")
    r1_ok = True
    for sp in splits:
        want = BASELINE.get(sp) or {}
        got = tb.get(sp) or {}
        diffs = []
        for k in ("scorable_pass", "scorable_n", "all_n"):
            if want.get(k) is not None and got.get(k) != want.get(k):
                diffs.append(f"{k}: 登记 {want[k]} ≠ 实测 {got[k]}")
        if want.get("opened") is not None and got.get("opened") != want.get("opened"):
            diffs.append(f"opened: 登记 {want['opened']} ≠ 实测 {got.get('opened')}")
        ok = not diffs
        r1_ok = r1_ok and ok
        log(f"  {sp:<9} 可判 {got.get('scorable_pass')}/{got.get('scorable_n')}"
            f"  开口 {got.get('opened')}/{got.get('all_n')}"
            f"   {'✅ 复现' if ok else '❌ ' + '; '.join(diffs)}")
    log(f"  R1 判定：{'PASS' if r1_ok else 'FAIL（默认路径被改动了，不许继续）'}")

    head("§4 两臂对照")
    for sp in splits:
        b, e = tb[sp], te[sp]
        log(f"  {sp:<9} 可判 {b['scorable_pass']}/{b['scorable_n']}"
            f" → {e['scorable_pass']}/{e['scorable_n']}"
            f" ({e['scorable_pass'] - b['scorable_pass']:+d} 题)")
        log(f"  {'':<9} 开口 {b['opened']}/{b['all_n']}"
            f" → {e['opened']}/{e['all_n']}"
            f" ({e['opened'] - b['opened']:+d} 题)")

    head("§5 逐题翻转")
    gain, loss, newly_opened, closed = [], [], [], []
    for k in sorted(base):
        b, e = base[k], exp.get(k) or {}
        if b["pass"] is not True and e.get("pass") is True:
            gain.append((k, b, e))
        elif b["pass"] is True and e.get("pass") is not True:
            loss.append((k, b, e))
        if not b["answered"] and e.get("answered"):
            newly_opened.append((k, b, e))
        elif b["answered"] and not e.get("answered"):
            closed.append((k, b, e))
    log(f"  答对 +{len(gain)} / −{len(loss)}    开口 +{len(newly_opened)} / −{len(closed)}")
    for label, rows in (("增益", gain), ("回退", loss),
                        ("新开口", newly_opened), ("新拒答", closed)):
        for k, b, e in rows:
            log(f"    [{label}] {k} {b['category']:<16}"
                f" pass {b['pass']}→{e.get('pass')}"
                f"  alias={e.get('alias_applied')} title={e.get('title_applied')}"
                f"  ncit={e.get('n_citations')}"
                f"  answer={(e.get('answer') or '')[:44]!r}")

    head("§6 R2 应拒题正确拒答数（不许下降）｜R3 引用硬失败（必须 0）")
    # ---- R2：必须在**全量** 588 题上测，不能只在 pick_items 的池子里测 ----
    # `pick_items` 只收 `gold_chunk_ids` 非空的题，而 gold-v2 的 52 道
    # `should_abstain=True`（全 negative_abstain）**一道都没有 gold**，
    # 所以它们根本不在 §1–§5 的池子里。只在池子里算 R2 会得到"应拒题 0 道"，
    # 那是一条**空真通过**的假判据（本项目记录在案的形态）。这里单独跑一遍。
    abst_report = {"n": 0, "base_correct": 0, "exp_correct": 0, "broke": []}
    if not args.limit:
        abst_items = [it for it in items if it.get("should_abstain")]
        abst_report["n"] = len(abst_items)
        for it in abst_items:
            q = str(it.get("question") or "")
            qid = str(it.get("question_id") or "")
            vals = []
            for alias_on in (False, True):
                opts = dict(base_options)
                if alias_on:
                    opts["alias_answer"] = True
                else:
                    opts.pop("alias_answer", None)
                res = pipeline_mod.ask_once(pipe, q, options=opts)
                ver = answer_mod.verify_citations(dict(res.get("answer") or {}),
                                                  loaded["chunks"])
                vals.append(bool(str(ver.get("answer") or "").strip()))
            if not vals[0]:
                abst_report["base_correct"] += 1
            if not vals[1]:
                abst_report["exp_correct"] += 1
            if not vals[0] and vals[1]:
                abst_report["broke"].append(qid)
        log(f"  R2 应拒题（全量，不入 pick_items 池）{abst_report['n']} 道："
            f"正确拒答 {abst_report['base_correct']} → {abst_report['exp_correct']}")
        log(f"     由「正确拒答」变成「作答」：{len(abst_report['broke'])} 道"
            f"{' ' + str(abst_report['broke'][:10]) if abst_report['broke'] else ''}")
    else:
        log("  R2 跳过（--limit 生效，应拒题不在池内，样本不完整）")

    # ---- R3：**两条都要**，缺一条就是假判据 -------------------------------
    # (a) 实验臂的引用剔除数**不许超过基线**（基线本身有历史残留剔除，不是 0，
    #     要求绝对 0 会把一条正确的改动判红——这是本脚本第一版犯的错）；
    # (b) alias 通路**自己发出的引用**必须逐条过 `verify_citations`（剔除 = 0）。
    rej_b = sum(r["cite_rejected"] for r in base.values())
    rej_e = sum(r["cite_rejected"] for r in exp.values())
    rej_alias = sum(r["cite_rejected"] for r in exp.values() if r.get("alias_applied"))
    n_alias = sum(1 for r in exp.values() if r.get("alias_applied"))
    log(f"  R3a 引用剔除总数：基线 {rej_b} → 实验 {rej_e}（不许上升）")
    log(f"  R3b alias 通路触发 {n_alias} 题，其中被剔引用 {rej_alias} 条（必须 0）")
    r2_ok = abst_report["exp_correct"] >= abst_report["base_correct"]
    r3_ok = (rej_e <= rej_b) and rej_alias == 0

    verdict = {"R1_default_path_identical": r1_ok,
               "R2_no_abstain_regression": r2_ok,
               "R3_zero_citation_failure": r3_ok}
    verdict["all_pass"] = all(verdict.values())
    head("§7 判定")
    for k, v in verdict.items():
        log(f"  {k:<34} {'PASS' if v else 'FAIL'}")

    payload = {
        "schema": "arkrag.eval_alias_gate/1",
        "generation": args.generation,
        "prod_options": base_options,
        "splits": splits,
        "baseline_registered": BASELINE,
        "baseline_measured": tb,
        "experiment_measured": te,
        "gain": [k for k, _, _ in gain],
        "loss": [k for k, _, _ in loss],
        "newly_opened": [{"key": k, "answer": (e.get("answer") or "")[:160],
                          "n_citations": e.get("n_citations"),
                          "cites_ok": e.get("cites_ok")}
                         for k, _, e in newly_opened],
        "closed": [k for k, _, _ in closed],
        "abstain_ok": abst_report,
        "citation_rejected": {"base": rej_b, "exp": rej_e,
                             "alias_path": rej_alias, "alias_triggered": n_alias},
        "verdict": verdict,
        "rows": {"base": base, "exp": exp},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with io.open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    log(f"\n落盘 {out}")
    return 0 if verdict["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
