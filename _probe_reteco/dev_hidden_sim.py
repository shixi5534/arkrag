"""隐藏集模拟：把 dev 当总体，按官方口径（11 域宏平均、每域 ~18 turn、
turn 按 conversation 整簇入样）反复抽 200-turn 假隐藏集，
看同一个系统报出来的 nDCG@10 会飘多大，以及两个系统排名翻转的概率。

只用 dev_audit.json 的 per-turn 结果，不重跑检索。
"""
from __future__ import annotations
import json, math, random, statistics
from collections import defaultdict

SEED = 20261112
REPS = 4000
TARGET_TURNS = 200
N_DOMAINS = 11

d = json.load(open(r"E:\图rag\_probe_reteco\dev_audit.json", encoding="utf-8"))
rows = d["per_turn"]
for r in rows:
    r["domain"] = r["conv"].split("/")[0]

# 每个域：conversation -> [row indices]
by_dom = defaultdict(lambda: defaultdict(list))
for i, r in enumerate(rows):
    by_dom[r["domain"]][r["conv"]].append(i)
doms = sorted(by_dom)
per_dom_turns = TARGET_TURNS / N_DOMAINS  # ~18.2


def draw_domain(rng, dom, target):
    """整簇入样直到达到 target turn 数；簇内全部 turn 进样（官方口径：
    一题的 turn 不拆散，且一个对话不会跨 split）。"""
    groups = by_dom[dom]
    keys = list(groups)
    picked, n = [], 0
    while n < target:
        k = keys[rng.randrange(len(keys))]
        picked.extend(groups[k]); n += len(groups[k])
    return picked


def macro_score(rng, key, target_per_domain=per_dom_turns):
    vals = []
    for dom in doms:
        idx = draw_domain(rng, dom, target_per_domain)
        vals.append(sum(rows[i][key] for i in idx) / len(idx))
    return sum(vals) / len(vals)


def pct(s, lo=2.5, hi=97.5):
    s = sorted(s); n = len(s)
    return s[max(0, int(lo / 100 * n))], s[min(n - 1, int(hi / 100 * n))]


rng = random.Random(SEED)

print(f"dev: {len(rows)} turns / {len(set(r['conv'] for r in rows))} convs / {len(doms)} domains")
print(f"目标隐藏集: {TARGET_TURNS} turn, 每域 {per_dom_turns:.1f} turn")
print(f"每域平均 {sum(len(v) for v in by_dom[doms[0]].values())/len(by_dom[doms[0]]):.2f} turn/对话")
print(f"-> 每域约 {per_dom_turns/4.07:.1f} 个独立对话\n")

# ---- 1) 单系统的波动 ----
print("=== 同一系统在 200-turn 假隐藏集上的 nDCG@10 分布 ===")
summary = {}
for key, label in [("ndcg_base", "Base (current turn only)"), ("ndcg_hist", "+History")]:
    sims = [macro_score(rng, key) for _ in range(REPS)]
    lo, hi = pct(sims)
    point = sum(rows[i][key] for i in range(len(rows))) / len(rows)
    byd = defaultdict(list)
    for r in rows: byd[r["domain"]].append(r[key])
    dev_macro = sum(sum(v) / len(v) for v in byd.values()) / len(byd)
    summary[key] = {"dev_macro": round(dev_macro, 4), "sim_mean": round(statistics.mean(sims), 4),
                    "ci95": [round(lo, 4), round(hi, 4)], "width": round(hi - lo, 4),
                    "sd": round(statistics.pstdev(sims), 4),
                    "p05": round(sorted(sims)[int(.05*REPS)], 4),
                    "p95": round(sorted(sims)[int(.95*REPS)], 4),
                    "min": round(min(sims), 4), "max": round(max(sims), 4)}
    print(f"{label:24s} dev全量macro={dev_macro:.4f}")
    print(f"{'':24s} 200-turn 95%区间=[{lo:.4f},{hi:.4f}] 宽={hi-lo:.4f} SD={statistics.pstdev(sims):.4f}")
    print(f"{'':24s} 全距 [{min(sims):.4f},{max(sims):.4f}]")

# ---- 2) 两系统排名翻转概率 ----
print("\n=== 两系统在同一 200-turn 隐藏集上的差与翻转率 ===")
a_key, b_key = "ndcg_base", "ndcg_hist"
deltas, awins = [], 0
rng2 = random.Random(SEED + 1)
for _ in range(REPS):
    sa = macro_score(rng2, a_key)
    sb = macro_score(rng2, b_key)
    # 同一隐藏集应同簇共用；这里独立抽样，作为“两系统看同一批题”的近似上界
    deltas.append(sb - sa)
    if sb > sa: awins += 1
print(f"  Base -> +History 实测差 {statistics.mean(deltas):+.4f}，P(+History胜)={awins/REPS:.4f}")

# 真差为 delta 时，同簇配对差的正态近似
sd_single = summary[b_key]["sd"]
for true_delta in [0.00, 0.02, 0.05, 0.10, 0.15, 0.20]:
    # 配对（同一批题）时差异的 SD 远小于两个独立分数 SD 之和；实测配对 SD per turn=0.3758
    sd_paired_turn = 0.3758
    # 200 turn、每域 18.2 turn、簇内 4.07 turn -> 有效 turn 数
    n_eff = TARGET_TURNS / 4.48
    se = sd_paired_turn / math.sqrt(n_eff)
    power = 0.5 * (1 + math.erf((true_delta / se - 1.959964) / math.sqrt(2)))
    print(f"  真差={true_delta:.3f}  n_eff={n_eff:.1f}  SE={se:.4f}  "
          f"检测功效≈{power:.2%}  需真差≥{2.8016*sd_paired_turn/math.sqrt(n_eff):.3f} 才达 80%")

out = {"seed": SEED, "reps": REPS, "target_turns": TARGET_TURNS,
       "per_domain_turns": round(per_dom_turns, 2),
       "single_system": summary,
       "paired": {"observed_delta": round(statistics.mean(deltas), 4),
                  "p_hist_wins": round(awins / REPS, 4)}}
with open(r"E:\图rag\_probe_reteco\dev_hidden_sim.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("\nwrote dev_hidden_sim.json")
