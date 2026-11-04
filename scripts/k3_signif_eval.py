# -*- coding: utf-8 -*-
"""K3：从 data/k3_recall_pairs.json 重算配对显著性（不重新跑两代）。

口径 B —— 与 `scripts/ask.py` 的评估器对齐，否则算出来的均值对不上台账里的读数：
- 分母只取**有 gold_chunk_ids 的题**（320 题里 20 题没有 gold，一律剔除，n=300）。
- R@5 / R@20：命中比例，天然落在 [0,1]，无 null。
- RR：`ask.py:278-279` 写的是 `rr.append(0.0 if first is None else 1/(first+1))`
  ——**没命中记 0.0，不是丢掉**。所以这里同样 0 填充，否则均值会明显偏高。
- 符号检验：双侧精确二项尾概率（零假设 p=0.5），只看"新胜/旧胜"，持平不计入 n。
- bootstrap：按题重采样 5000 轮，取 Δ(new-old) 的 2.5% / 97.5% 分位。

与口径 A 的关系：`scripts/k3_signif.py` 是口径 A 的**生产者**（它顺序跑两代、写
`data/k3_recall_pairs.json`），其 `pairs()` 遇到 `None` 直接丢弃，所以它的 rr 均值分母
是 229 而不是 300。本脚本是口径 B 的**消费者**：只读 `data/k3_recall_pairs.json`，
不写任何文件，两套口径因此可以在同一份逐题明细上对照。

退出码：0 = 三指标均算出结果（判据成立）/ 1 = 有指标无数据可算（判据不成立）。
"""
from __future__ import annotations

import json
import random
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PAIRS = ROOT / "data" / "k3_recall_pairs.json"
SEED = 20261104
ROUNDS = 5000


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def bootstrap_ci(ps, seed=SEED, rounds=ROUNDS):
    rnd = random.Random(seed)
    n = len(ps)
    deltas = []
    for _ in range(rounds):
        pick = [ps[rnd.randrange(n)] for _ in range(n)]
        deltas.append(sum(p[1] for p in pick) / n - sum(p[0] for p in pick) / n)
    deltas.sort()
    return deltas[int(0.025 * rounds)], deltas[int(0.975 * rounds) - 1]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    d = json.loads(PAIRS.read_text("utf-8"))
    old, new = d["old"], d["new"]
    g = d["generations"]
    print(f"旧世代 {g['old']}")
    print(f"新世代 {g['new']}")
    common = sorted(set(old) & set(new))
    print(f"共同题 {len(common)}")

    # 有 gold 的题：两代 r20 都非 null（无 gold 时两代都是 null）
    graded = [q for q in common if old[q].get("r20") is not None and new[q].get("r20") is not None]
    print(f"有 gold 的题 {len(graded)}（无 gold 剔除 {len(common) - len(graded)}）\n")

    SPEC = {"r5": "R@5", "r20": "R@20", "rr": "RR"}
    empty: list[str] = []
    for metric in ("r5", "r20", "rr"):
        ps = []
        for q in graded:
            a = old[q].get(metric)
            b = new[q].get(metric)
            if a is None:
                a = 0.0   # 无 gold 时不会是 None（已在 graded 里剔过）；这里只剩"没命中"
            if b is None:
                b = 0.0
            ps.append((a, b))
        if not ps:
            empty.append(metric)
            print(f"[{SPEC[metric]}] 无可比题")
            continue
        x = sum(p[0] for p in ps) / len(ps)
        y = sum(p[1] for p in ps) / len(ps)
        wins = sum(1 for a, b in ps if b > a)
        losses = sum(1 for a, b in ps if b < a)
        ties = len(ps) - wins - losses
        p = sign_test(wins, losses)
        lo, hi = bootstrap_ci(ps)
        print(f"[{SPEC[metric]}] n={len(ps)}")
        print(f"  旧 {x:.4f} / 新 {y:.4f} / Δ {y - x:+.4f}")
        print(f"  新胜 {wins} / 旧胜 {losses} / 持平 {ties} → 符号检验 n={wins + losses}")
        print(f"  p = {p:.4f}")
        print(f"  bootstrap 95% [{lo:+.4f}, {hi:+.4f}]  跨 0={lo <= 0 <= hi}")
        print()

    # 对照：把"没命中"当缺失丢掉（这是错的，只为说明 0 填充的必要性）
    print("=== 对照：RR 把未命中题当缺失丢掉会怎样（错口径）===")
    ps = [(old[q]["rr"], new[q]["rr"]) for q in graded
          if old[q].get("rr") is not None and new[q].get("rr") is not None]
    if not ps:
        print("  无可比题（该口径不成立）")
        empty.append("rr-paired")
    else:
        x = sum(a for a, _ in ps) / len(ps)
        y = sum(b for _, b in ps) / len(ps)
        print(f"  n={len(ps)} 旧 {x:.4f} / 新 {y:.4f} / Δ {y - x:+.4f}（台账记的是 0.4394 / 0.4312，对不上）")

    return 0 if not empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
