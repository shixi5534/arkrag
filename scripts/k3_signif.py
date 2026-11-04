# -*- coding: utf-8 -*-
"""K3：新旧两代召回差异的配对显著性检验。

顺序跑两代（不同时载入，避免内存叠加），每题记 R@5 / R@20 / RR，
再用配对符号检验 + bootstrap 判断差异是否可分辨。
"""
from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
# Windows 控制台默认 cp936，不重配置会把中文表头写成乱码字节，
# 且 `> file` 重定向后文件不是 UTF-8（本项目脚本一律要求可被原样粘贴进文档）。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from arkrag import config, query as query_mod, vocab as vocab_mod  # noqa: E402

GOLD = ROOT / "data" / "goldset" / "gold-v1.json"
GENS = {
    "old": "g-20261031T035231Z-4eafae",
    "new": "g-20261103T132152Z-06a741",
}

spec = importlib.util.spec_from_file_location("ask_under_test", ROOT / "scripts" / "ask.py")
ask = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ask)

gold = json.loads(GOLD.read_text("utf-8"))
items = gold["items"] if isinstance(gold, dict) else gold
qs = ask._stratified_sample(items, 320)


def run_one(gid: str) -> dict[str, dict]:
    loaded = ask.load_generation(gid)
    pipe = ask.build_pipeline(loaded, top_k=20, graph_depth=1)
    pipe["generation_id"] = gid
    pipe["entities"] = loaded["graph"].get("entities") or []
    out = {}
    for it in qs:
        qid = str(it.get("question_id") or "")
        g = [str(c) for c in it.get("gold_chunk_ids") or []]
        res = ask.ask_once(pipe, str(it.get("question") or ""))
        hits = [str(h["chunk_id"]) for h in res["search"]["hits"]]
        gs = set(g)
        r5 = len(gs & set(hits[:5])) / len(gs) if gs else None
        r20 = len(gs & set(hits[:20])) / len(gs) if gs else None
        first = next((i for i, c in enumerate(hits) if c in gs), None)
        rr = None if (not gs or first is None) else 1.0 / (first + 1)
        out[qid] = {"r5": r5, "r20": r20, "rr": rr, "category": it.get("category")}
    del loaded, pipe
    return out


results = {}
for tag, gid in GENS.items():
    print(f"[{tag}] {gid} ...", flush=True)
    results[tag] = run_one(gid)
    print(f"[{tag}] done", flush=True)

common = sorted(set(results["old"]) & set(results["new"]))
print(f"\n共同题 {len(common)}")


def pairs(metric: str):
    out = []
    for qid in common:
        a = results["old"][qid].get(metric)
        b = results["new"][qid].get(metric)
        if a is None or b is None:
            continue
        out.append((a, b))
    return out


def sign_test(wins: int, losses: int) -> float:
    """双侧精确符号检验（零假设 p=0.5），用二项分布尾概率。"""
    n = wins + losses
    if n == 0:
        return 1.0
    from math import comb

    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def bootstrap_ci(pairs_list, seed=20261104, rounds=5000):
    rnd = random.Random(seed)
    n = len(pairs_list)
    deltas = []
    for _ in range(rounds):
        pick = [pairs_list[rnd.randrange(n)] for _ in range(n)]
        a = sum(p[0] for p in pick) / n
        b = sum(p[1] for p in pick) / n
        deltas.append(b - a)
    deltas.sort()
    lo = deltas[int(0.025 * rounds)]
    hi = deltas[int(0.975 * rounds) - 1]
    return lo, hi


print(f"\n{'指标':<6}{'旧均值':>10}{'新均值':>10}{'Δ':>10}{'新胜':>6}{'旧胜':>6}{'持平':>6}{'p':>9}  bootstrap 95%")
for metric in ("r5", "r20", "rr"):
    ps = pairs(metric)
    if not ps:
        print(f"{metric:<6} 无可比题")
        continue
    a = sum(p[0] for p in ps) / len(ps)
    b = sum(p[1] for p in ps) / len(ps)
    wins = sum(1 for x, y in ps if y > x)
    losses = sum(1 for x, y in ps if y < x)
    ties = len(ps) - wins - losses
    p = sign_test(wins, losses)
    lo, hi = bootstrap_ci(ps)
    print(f"{metric:<6}{a:>10.4f}{b:>10.4f}{b - a:>+10.4f}{wins:>6}{losses:>6}{ties:>6}{p:>9.4f}  [{lo:+.4f}, {hi:+.4f}]")

Path(ROOT / "data" / "k3_recall_pairs.json").write_text(
    json.dumps({"generations": GENS, "questions": common,
                "old": results["old"], "new": results["new"]},
               ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print("\n逐题明细已写 data/k3_recall_pairs.json")
