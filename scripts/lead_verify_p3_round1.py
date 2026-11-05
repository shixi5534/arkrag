"""独立复核 A 的 P3 第一轮（整行引用）：不采信自报，自己从逐题结果重算。

要回答四件事：
1. `coverage_ok` 涨了多少、按什么分母算的、分母对不对；
2. 答案文本有没有变（单变量是否成立）；
3. 引用条数与逐条定位通过率的变化（塌缩代价）；
4. **答案质量涨了没有** —— 这是 Lead 定的硬口径：
   覆盖率涨而「答案文本含 gold 答案」的题数不涨，只能记口径修正，不是质量提升。
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import platform
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from arkrag import subproc  # noqa: E402


def git_rev() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=subproc.child_env(),
        )
        return (out.stdout or "").strip() or "unknown"
    except Exception as exc:  # noqa: BLE001
        return f"unknown({exc})"


def load_gold(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {it["question_id"]: it for it in data["items"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default="data/a_p3_line_citation.json")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    res = json.loads((ROOT / args.result).read_text(encoding="utf-8"))
    gold = load_gold(ROOT / res["gold_set"])
    per_q_raw = res["per_question"]
    # 实际结构：{"base": [行...], "cand": [行...]}，不是「题号 -> {base, cand}」
    per_q = {k: {row["question_id"]: row for row in per_q_raw[k]} for k in ("base", "cand")}
    qids = sorted(per_q["base"])
    print(f"逐题 {len(qids)} 道；variant = {res['variable']}")
    assert set(qids) == set(per_q["cand"]), "两个变体的题号集合不同，不能逐题比对"

    def pick(qid: str, key: str) -> dict:
        return per_q[key][qid]

    cov_gain, cov_lose = [], []
    ans_changed, ans_same = [], []
    cites_delta = collections.Counter()
    ans_hit_gain, ans_hit_lose = [], []
    cite_bad_base = cite_bad_cand = 0
    cite_total_base = cite_total_cand = 0
    collapse_gain = []
    for qid in qids:
        b, c = pick(qid, "base"), pick(qid, "cand")
        if b.get("covered") != c.get("covered"):
            (cov_gain if c.get("covered") else cov_lose).append(qid)
        if b.get("answer") != c.get("answer"):
            ans_changed.append(qid)
        else:
            ans_same.append(qid)
        cites_delta[int(c.get("citations") or 0) - int(b.get("citations") or 0)] += 1
        cite_total_base += int(b.get("citations") or 0)
        cite_total_cand += int(c.get("citations") or 0)
        cite_bad_base += int(b.get("cite_bad") or 0)
        cite_bad_cand += int(c.get("cite_bad") or 0)
        if int(b.get("citations_distinct") or 0) - int(c.get("citations_distinct") or 0) > 0:
            collapse_gain.append(qid)
        item = gold[qid]
        ga = str(item.get("answer") or "")
        ok_b = bool(ga) and ga in str(b.get("answer") or "")
        ok_c = bool(ga) and ga in str(c.get("answer") or "")
        if ok_c and not ok_b:
            ans_hit_gain.append(qid)
        if ok_b and not ok_c:
            ans_hit_lose.append(qid)

    # 答案质量：全量计数（含 should_abstain 之外的所有 dev 应答题）
    hit_base = sum(1 for q in qids if str(gold[q].get("answer") or "")
                   and str(gold[q]["answer"]) in str(pick(q, "base").get("answer") or ""))
    hit_cand = sum(1 for q in qids if str(gold[q].get("answer") or "")
                   and str(gold[q]["answer"]) in str(pick(q, "cand").get("answer") or ""))

    print("\n== 覆盖率 ==")
    print(f"  自报 base/cand: {res['coverage']['base']['coverage_ok']} -> "
          f"{res['coverage']['cand']['coverage_ok']}")
    print(f"  复算 base/cand: {sum(1 for q in qids if pick(q,'base').get('covered'))} -> "
          f"{sum(1 for q in qids if pick(q,'cand').get('covered'))}")
    print(f"  救 {len(cov_gain)} 丢 {len(cov_lose)} 净 {len(cov_gain)-len(cov_lose)}")
    if cov_lose:
        print(f"  丢失题号: {cov_lose}")

    print("\n== 单变量自检（答案文本是否变） ==")
    print(f"  答案文本相同 {len(ans_same)} / 变化 {len(ans_changed)}")
    if ans_changed:
        print(f"  变化题号（前 10）: {ans_changed[:10]}")

    print("\n== 引用条数 ==")
    print(f"  总数 base {cite_total_base} -> cand {cite_total_cand}（Δ{cite_total_cand-cite_total_base:+d}）")
    print(f"  逐条定位失败 base {cite_bad_base} -> cand {cite_bad_cand} ⇒ 通过率 "
          f"{(1-cite_bad_base/cite_total_base):.4f} -> {(1-cite_bad_cand/cite_total_cand):.4f}")
    print(f"  去重后减少的题数（塌缩）: {len(collapse_gain)}")
    print(f"  逐题引用数变化直方图（负=减少）: {dict(sorted(cites_delta.items()))}")

    print("\n== 答案质量（硬口径） ==")
    print(f"  答案文本含 gold 答案：base {hit_base} -> cand {hit_cand}"
          f"（救 {len(ans_hit_gain)} 丢 {len(ans_hit_lose)}）")
    if ans_hit_gain:
        print(f"  救回: {ans_hit_gain}")
    if ans_hit_lose:
        print(f"  丢失: {ans_hit_lose}")

    out = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_rev(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "command": "py -3 scripts/lead_verify_p3_round1.py",
        "source": args.result,
        "questions": len(qids),
        "coverage": {
            "base": sum(1 for q in qids if pick(q, "base").get("covered")),
            "cand": sum(1 for q in qids if pick(q, "cand").get("covered")),
            "gain": cov_gain, "lose": cov_lose,
            "selfreported_base": res["coverage"]["base"]["coverage_ok"],
            "selfreported_cand": res["coverage"]["cand"]["coverage_ok"],
            "denominators": {
                "all_answerable": res["coverage"]["base"]["den_all_answerable"],
                "span_reachable": res["coverage"]["base"]["den_span_reachable"],
                "two_gates": res["coverage"]["base"]["den_two_gates"],
            },
        },
        "single_variable": {"answer_same": len(ans_same), "answer_changed": ans_changed},
        "citations": {
            "total_base": cite_total_base, "total_cand": cite_total_cand,
            "bad_base": cite_bad_base, "bad_cand": cite_bad_cand,
            "accuracy_base": 1 - cite_bad_base / cite_total_base if cite_total_base else None,
            "accuracy_cand": 1 - cite_bad_cand / cite_total_cand if cite_total_cand else None,
            "collapsed_questions": len(collapse_gain),
            "delta_histogram": {str(k): v for k, v in sorted(cites_delta.items())},
        },
        "answer_quality": {
            "contains_gold_base": hit_base, "contains_gold_cand": hit_cand,
            "gain": ans_hit_gain, "lose": ans_hit_lose,
        },
    }
    if args.json:
        path = ROOT / args.json
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n已写 {path} ({path.stat().st_size} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
