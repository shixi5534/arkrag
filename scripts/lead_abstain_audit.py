# -*- coding: utf-8 -*-
"""审核「拒答正确率 ≥90%」这条门限：它的分母混了两类相反的错误。

背景
----
`scripts/ask.py:296-299` 把两类**方向相反**的错误放进同一个分母：

    abstained = bool(ans.get("abstained"))
    if question:
        abstain_ok.append(abstained == should_abstain)

于是「该答却拒」与「该拒却答」互相抵消。P3 第三轮 cross_chunk 的实测就是活例子：
`abstain_accuracy` 从 0.855263 升到 0.871053，而**应拒题正确拒答数 5/30 一个没变**——
涨的全是 6 道「拒答（错）→作答（也错）」，两类错误相抵后数字变好看了，系统没变好。
（同一形态已记入 `docs/p0_门报告可追溯.md` 教训表的第十八种。）

本脚本把这条指标拆成三行，并回答「要达 90% 到底需要什么」：
1. 三行拆解：应拒题正确拒答 a/N拒；应答题被拒 b/N答；b 里答案其实对的 c。
   判据：**只有 `N答 − b == c` 才算真改善**；`c=0` 而 b 下降 = 纯风险敞口。
2. 达到 90% 的**算术解空间**：给定当前 a、b、c，需要净改对多少题；
   以及分别走「多拒」与「多答」两条路各自要多少题。
3. 应拒题的可判别性初探：`negative_abstain` 那批问的是**词表外的关系标签**，
   这类题能不能靠「端点对在图上有没有相关边」判出来（只报结构量，不主张判据可用）。

边界
----
- 只读 dev（`split == "dev"` 与 `origin == "v1"` 两种口径各算一遍）；验收集不碰。
- 不改代码、不改 gold、不调参数。
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from arkrag import subproc  # noqa: E402

GOLD = ROOT / "data" / "goldset" / "gold-v2.json"
ROUND = ROOT / "data" / "a_p3_chapter_round.json"
QUOTE = re.compile(r"「([^」]+)」")


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=20, env=subproc.child_env(),
        )
    except Exception:  # noqa: BLE001
        return ""
    return (out.stdout or "").strip() if out.returncode == 0 else ""


def headroom(n_ans: int, b: int, c: int, a: int, n_ref: int, total: int) -> dict:
    """算术解空间：当前 (总正确) = a + (n_ans - b)。问 90% 需要几题。"""
    correct = a + (n_ans - b)
    need = (0.90 * total)
    gap = need - correct
    # 路线一：把 b 题里答案本来对的（c 题）改成「答且对」——纯免费
    free = min(b, c)
    ref_wrong_answered = n_ref - a          # 应拒却被答
    return {
        "分母_全部题": total,
        "当前正确": correct,
        "当前比例": round(correct / total, 6) if total else None,
        "90%所需正确": round(need, 3),
        "还差": round(gap, 3),
        "免费额度_b里答案本来对的": free,
        "免费后还差": round(gap - free, 3),
        "应拒却被答_可降额": ref_wrong_answered,
        "应答题被拒_可降额": b,
    }


def degenerate(n_ans: int, n_ref: int, total: int) -> dict:
    """两种退化策略在这条指标上的得分 —— 判「门限有没有鉴别力」。

    `abstain_accuracy` 只比对 `abstained == should_abstain`，**完全不看答案对不对**。
    所以「永不拒答」在应答题上全对、在应拒题上全错；「永不回答」反之。
    如果退化策略已经能过门限，这条门限就不具备鉴别力。
    """
    always_answer = n_ans / total
    always_abstain = n_ref / total
    return {
        "永不拒答_得分": round(always_answer, 6),
        "永不拒答_是否已过90%": always_answer >= 0.90,
        "永不回答_得分": round(always_abstain, 6),
        "应拒题占比": round(n_ref / total, 6),
        "说明": ("只要应拒题占比 < 10%，「永不拒答」就自动过 90% —— "
                 "这条门限把最差的策略也判成通过"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", dest="json_out", default="")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not GOLD.exists() or not ROUND.exists():
        raise SystemExit(f"输入不存在：{GOLD} / {ROUND}")
    gold = {it["question_id"]: it
            for it in json.loads(GOLD.read_text(encoding="utf-8"))["items"]}
    rnd = json.loads(ROUND.read_text(encoding="utf-8"))
    base = {r["question_id"]: r for r in rnd["per_question"]["base"]}

    # 注意：`data/a_p3_chapter_round.json` 只跑了 dev 的 380 题。v1_320 里有 100 道
    # 落在 v1-holdout，不在 base 里 —— 两个口径都必须先与 base 取交，并把被排除的
    # 题数如实报出来（否则「v1 320 题」这个名字与实际分母不符）。
    scopes = {}
    for name, sel in (
        ("dev_380", [q for q, it in gold.items() if it.get("split") == "dev"]),
        ("v1_320", [q for q, it in gold.items() if it.get("origin") == "v1"]),
    ):
        have = [q for q in sel if q in base]
        scopes[name] = (have, len(sel) - len(have))

    out: dict[str, dict] = {}
    for name, (qids, dropped) in scopes.items():
        total = len(qids)
        ref = [q for q in qids if gold[q].get("should_abstain")]
        ans = [q for q in qids if not gold[q].get("should_abstain")]
        a = sum(1 for q in ref if base[q]["abstained"])          # 应拒且拒
        ref_answered = sum(1 for q in ref if not base[q]["abstained"])  # 应拒却答
        b = sum(1 for q in ans if base[q]["abstained"])          # 应答却拒
        c = sum(1 for q in ans if base[q]["abstained"] and base[q]["answer_hits_gold"])
        answered_and_right = sum(1 for q in ans if not base[q]["abstained"] and base[q]["answer_hits_gold"])

        # 第三行：应拒却答那批题，答案有没有「碰巧对」（gold 常是「否」或不存在）
        ref_answered_right = sum(
            1 for q in ref if not base[q]["abstained"]
            and str(gold[q].get("answer") or "") in str(base[q]["answer"] or "")
        )
        reasons_ref = Counter(base[q]["abstain_reason"] for q in ref if base[q]["abstained"])
        reasons_ans = Counter(base[q]["abstain_reason"] for q in ans if base[q]["abstained"])
        by_cat = defaultdict(lambda: {"n": 0, "应拒": 0, "应拒且拒": 0, "应答": 0, "应答却拒": 0})

        for q in qids:
            cat = gold[q]["category"]
            t = by_cat[cat]
            t["n"] += 1
            if gold[q].get("should_abstain"):
                t["应拒"] += 1
                t["应拒且拒"] += base[q]["abstained"]
            else:
                t["应答"] += 1
                t["应答却拒"] += base[q]["abstained"]

        out[name] = {
            "总题数": total,
            "因不在base而排除的题数": dropped,
            "应拒题": len(ref),
            "应拒且拒_a": a,
            "应拒却答": ref_answered,
            "应拒却答_答案碰巧对": ref_answered_right,
            "应答题": len(ans),
            "应答却拒_b": b,
            "应答却拒_答案其实对_c": c,
            "应答且答案对": answered_and_right,
            "现值_混算": round((a + len(ans) - b) / total, 6),
            "真改善判据_N答减b等c": (len(ans) - b) == c,
            "应拒侧拒答率": round(a / len(ref), 6) if ref else None,
            "应答侧不误拒率": round((len(ans) - b) / len(ans), 6),
            "拒答原因_应拒侧": dict(reasons_ref),
            "拒答原因_应答侧": dict(reasons_ans),
            "算术解空间": headroom(len(ans), b, c, a, len(ref), total),
            "退化策略得分": degenerate(len(ans), len(ref), total),
            "按类": {k: dict(v) for k, v in sorted(by_cat.items())},
        }

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_revision(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "command": "py -3 scripts/lead_abstain_audit.py",
        "gold_set": "data/goldset/gold-v2.json",
        "answer_state": "CHAPTER_ANSWER_PATH=False / TITLE_ANSWER_PATH=False（仓库默认）",
        "口径": dict(out),
    }

    for name, s in out.items():
        print(f"===== {name} =====")
        print(f"  总 {s['总题数']}（已排除不在 base 的 {s['因不在base而排除的题数']} 题）"
              f"  应拒 {s['应拒题']}  应答 {s['应答题']}")
        print(f"  应拒且拒 a={s['应拒且拒_a']}  ({s['应拒侧拒答率']:.4f})"
              f"   应拒却答 {s['应拒却答']}（其中答案碰巧对 {s['应拒却答_答案碰巧对']}）")
        print(f"  应答却拒 b={s['应答却拒_b']}（其中答案其实对 c={s['应答却拒_答案其实对_c']}）"
              f"   应答侧不误拒率 {s['应答侧不误拒率']:.4f}")
        print(f"  混算现值 {s['现值_混算']:.6f}"
              f"   真改善判据 (N答−b==c) → {'成立' if s['真改善判据_N答减b等c'] else '不成立'}")
        h = s["算术解空间"]
        print(f"  90% 需要正确 {h['90%所需正确']:.1f} 题，当前 {h['当前正确']:.1f}，还差 {h['还差']:.1f}"
              f"；b 里本来对的 {h['免费额度_b里答案本来对的']} 题改对后还差 {h['免费后还差']:.1f}")
        print(f"  可降额：应拒却答 {h['应拒却被答_可降额']} + 应答题被拒 {h['应答题被拒_可降额']}")
        d = s["退化策略得分"]
        print(f"  退化策略：永不拒答 {d['永不拒答_得分']:.4f}"
              f"（已过 90%？{'是' if d['永不拒答_是否已过90%'] else '否'}）"
              f"  永不回答 {d['永不回答_得分']:.4f}")
        print(f"  拒答原因（应拒侧）{s['拒答原因_应拒侧']}")
        print(f"  拒答原因（应答侧）{s['拒答原因_应答侧']}")
        print("  按类：")
        for cat, t in s["按类"].items():
            print(f"    {cat:<16} n={t['n']:<4} 应拒 {t['应拒']:<3} 拒对 {t['应拒且拒']:<3}"
                  f" 应答 {t['应答']:<4} 误拒 {t['应答却拒']}")
        print()

    if args.json_out:
        p = Path(args.json_out)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                     encoding="utf-8", newline="\n")
        print(f"写出 {p}（{p.stat().st_size} B）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
