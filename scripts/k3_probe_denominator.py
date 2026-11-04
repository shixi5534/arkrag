# -*- coding: utf-8 -*-
"""K3：核对 `data/k3_recall_pairs.json` 里三行指标的**分母**。

口径 A（`scripts/k3_signif.py` 的三行表）里，r5 / r20 / rr 三行的均值分母并不相同，
而分母写错会让「Δ 与 p 值」看起来可跨行比较。所以这一段必须实测，不能引用。

段一 pair：逐指标数「两代都有 / 仅旧有 / 仅新有 / 都没有」，再把 RR 的两个合法口径
    各自的均值算出来（配对子集 229 与 0 填充 320）。
段二 mrr：把「0 填充」按分母分成两种（全部 320 / 有 gold 的 300），确认评测器打印的
    `MRR 0.4394 / 0.4312` 到底是哪个口径。

判据：**分母标注成立**——段一三行各自的「两代都有」必须与段二「有 gold 的题」自洽
（r5/r20 的配对分母 == 有 gold 题数），且两个 RR 口径都能算出非空均值。
退出码 0 = 分母标注成立 / 1 = 不成立（例如某指标一行全是 None）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PAIRS = ROOT / "data" / "k3_recall_pairs.json"


def section_pair(d: dict) -> tuple[int, dict[str, int]]:
    qs = d["questions"]
    print(f"共同题 {len(qs)}")

    both_counts: dict[str, int] = {}
    for metric in ("r5", "r20", "rr"):
        both = [q for q in qs if d["old"][q].get(metric) is not None and d["new"][q].get(metric) is not None]
        only_old = [q for q in qs if d["old"][q].get(metric) is not None and d["new"][q].get(metric) is None]
        only_new = [q for q in qs if d["old"][q].get(metric) is None and d["new"][q].get(metric) is not None]
        neither = [q for q in qs if d["old"][q].get(metric) is None and d["new"][q].get(metric) is None]
        both_counts[metric] = len(both)
        print(f"{metric:>4}：两代都有 {len(both)} / 仅旧有 {len(only_old)} / 仅新有 {len(only_new)} / 都没有 {len(neither)}")

    # rr 为 None 的两种成因分别多少
    no_gold = [q for q in qs if d["old"][q].get("r20") is None]
    print(f"\n无 gold 的题（r20 为 None）{len(no_gold)}")
    rr_none_old = [q for q in qs if d["old"][q].get("rr") is None]
    rr_none_new = [q for q in qs if d["new"][q].get("rr") is None]
    print(f"旧代 rr 为 None {len(rr_none_old)}（其中无 gold {len([q for q in rr_none_old if q in set(no_gold)])}）")
    print(f"新代 rr 为 None {len(rr_none_new)}（其中无 gold {len([q for q in rr_none_new if q in set(no_gold)])}）")

    # 两代 rr 都非 None == 229 吗？均值是不是 0.5700 / 0.5619
    both = [q for q in qs if d["old"][q].get("rr") is not None and d["new"][q].get("rr") is not None]
    a = sum(d["old"][q]["rr"] for q in both) / len(both)
    b = sum(d["new"][q]["rr"] for q in both) / len(both)
    print(f"\n配对子集 n={len(both)}  rr 旧 {a:.4f} / 新 {b:.4f} / Δ {b - a:+.4f}")

    # 0 填充口径
    zero_old = sum((d["old"][q]["rr"] or 0.0) for q in qs) / len(qs)
    zero_new = sum((d["new"][q]["rr"] or 0.0) for q in qs) / len(qs)
    print(f"0 填充口径 n={len(qs)}  rr 旧 {zero_old:.4f} / 新 {zero_new:.4f} / Δ {zero_new - zero_old:+.4f}")
    return len(qs), both_counts


def section_mrr(d: dict) -> int:
    qs = d["questions"]
    graded = [q for q in qs if d["old"][q].get("r20") is not None and d["new"][q].get("r20") is not None]
    print(f"有 gold 的题（两代 r20 都非 None）= {len(graded)}")

    for tag, pool in (("全部 320", qs), (f"有 gold 的 {len(graded)}", graded)):
        for gen in ("old", "new"):
            rr = sum((d[gen][q]["rr"] or 0.0) for q in pool) / len(pool)
            r5 = sum((d[gen][q]["r5"] or 0.0) for q in pool) / len(pool)
            r20 = sum((d[gen][q]["r20"] or 0.0) for q in pool) / len(pool)
            hits = sum(1 for q in pool if d[gen][q].get("rr") is not None)
            print(f"  {tag:>12} {gen}: R@5 {r5:.4f}  R@20 {r20:.4f}  MRR {rr:.4f}"
                  f"  （该子集内命中 {hits}）")
    return len(graded)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    d = json.loads(PAIRS.read_text("utf-8"))

    print("===== 段一：三行指标各自的配对样本量（pair）=====")
    total, both_counts = section_pair(d)
    print()
    print("===== 段二：0 填充口径按分母分成两种（mrr）=====")
    graded = section_mrr(d)

    # 判据：r5 / r20 的配对分母必须等于有 gold 的题数；两个 RR 口径都要非空
    ok = (both_counts.get("r5") == graded
          and both_counts.get("r20") == graded
          and 0 < both_counts.get("rr", 0) <= total)
    print()
    print(f"分母标注自洽：r5 配对 {both_counts.get('r5')} / r20 配对 {both_counts.get('r20')}"
          f" / 有 gold {graded} / rr 配对 {both_counts.get('rr')} / 共同题 {total}"
          f" → {'成立' if ok else '不成立'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
