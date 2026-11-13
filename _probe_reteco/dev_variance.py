"""三级方差分解：turn(iid) / conversation / domain 三套 bootstrap，
以及宏平均口径下隐藏集 (~200 turn, 11 域) 的 MDE。

仅用 dev 已有 per-turn 结果，不重跑检索。
"""
from __future__ import annotations
import json, math, random, statistics
from collections import defaultdict

SEED = 20261112
REPS = 4000
Z = 2.8016  # z_{.975}+z_{.80}

d = json.load(open(r"E:\图rag\_probe_reteco\dev_audit.json", encoding="utf-8"))
rows = d["per_turn"]
for r in rows:
    r["domain"] = r["conv"].split("/")[0]

def pct(samples, lo=2.5, hi=97.5):
    s = sorted(samples); n = len(s)
    return s[max(0, int(lo / 100 * n))], s[min(n - 1, int(hi / 100 * n))]

def boot_delta_iid(a, b, reps, rng):
    n = len(a)
    return [sum(b[i] - a[i] for i in (rng.randrange(n) for _ in range(n))) / n for _ in range(reps)]

def boot_delta_cluster(a, b, keys, reps, rng):
    g = defaultdict(list)
    for i, k in enumerate(keys): g[k].append(i)
    ks = list(g); G = len(ks)
    out = []
    for _ in range(reps):
        tot = cnt = 0
        for _ in range(G):
            idx = g[ks[rng.randrange(G)]]
            tot += sum(b[i] - a[i] for i in idx); cnt += len(idx)
        out.append(tot / cnt if cnt else 0.0)
    return out

def boot_macro_iid(a, b, doms, reps, rng):
    """official statistic: macro average over 11 domains of per-domain turn means."""
    by = defaultdict(list)
    for i, dm in enumerate(doms): by[dm].append(i)
    dm_list = list(by)
    out = []
    for _ in range(reps):
        vals = []
        for dm in dm_list:
            idx = by[dm]
            n = len(idx)
            vals.append(sum(b[idx[rng.randrange(n)]] - a[idx[rng.randrange(n)]] for _ in range(n)) / n)
        out.append(sum(vals) / len(vals))
    return out

def boot_macro_2level(a, b, doms, convs, reps, rng):
    """域内按 conversation 重采样，域固定（11 域全在隐藏集）。"""
    by = defaultdict(lambda: defaultdict(list))
    for i, (dm, cv) in enumerate(zip(doms, convs)): by[dm][cv].append(i)
    dm_list = list(by)
    out = []
    for _ in range(reps):
        vals = []
        for dm in dm_list:
            groups = by[dm]; ks = list(groups); G = len(ks)
            tot = cnt = 0
            for _ in range(G):
                idx = groups[ks[rng.randrange(G)]]
                tot += sum(b[i] - a[i] for i in idx); cnt += len(idx)
            vals.append(tot / cnt if cnt else 0.0)
        out.append(sum(vals) / len(vals))
    return out

a = [r["ndcg_base"] for r in rows]
b = [r["ndcg_hist"] for r in rows]
doms = [r["domain"] for r in rows]
convs = [r["conv"] for r in rows]
obs = sum(y - x for x, y in zip(a, b)) / len(a)

# official macro observed
byd = defaultdict(lambda: ([], []))
for i, dm in enumerate(doms):
    byd[dm][0].append(a[i]); byd[dm][1].append(b[i])
macro_obs = sum(sum(bb) / len(bb) - sum(aa) / len(aa) for aa, bb in byd.values()) / len(byd)

rng = random.Random(SEED)
print(f"turns={len(a)} convs={len(set(convs))} domains={len(set(doms))}")
print(f"pooled paired delta observed = {obs:+.4f}")
print(f"official-style macro delta observed = {macro_obs:+.4f}\n")

res = {}
for label, samples, point in [
    ("A iid @ turn", boot_delta_iid(a, b, REPS, rng), obs),
    ("B cluster @ conversation", boot_delta_cluster(a, b, convs, REPS, rng), obs),
    ("C macro, resample turns in domain", boot_macro_iid(a, b, doms, REPS, rng), macro_obs),
    ("D macro, resample conv in domain", boot_macro_2level(a, b, doms, convs, REPS, rng), macro_obs),
]:
    lo, hi = pct(samples)
    sd = statistics.pstdev(samples)
    res[label] = {"ci": [round(lo, 4), round(hi, 4)], "width": round(hi - lo, 4), "sd": round(sd, 4)}
    print(f"{label:34s} CI95=[{lo:+.4f},{hi:+.4f}] width={hi-lo:.4f} sd={sd:.4f}")

base_w = res["A iid @ turn"]["width"]
print("\n方差膨胀相对 iid@turn（CI 宽度比）:")
for k, v in res.items():
    print(f"  {k:34s} {v['width']/base_w:.2f}x   (SE 比 {v['sd']/res['A iid @ turn']['sd']:.2f}x)")

# ---- MDE under each口径, projected to 200-turn hidden test ----
sd_pair = statistics.pstdev([y - x for x, y in zip(a, b)])
dev_deff = d["cluster"]["pooled_deff"]
print(f"\nper-turn SD of paired difference = {sd_pair:.4f}")
print(f"measured pooled design effect (Kish) = {dev_deff:.2f}")
print(f"\n=== 隐藏集 ~200 turn / 11 域 的 MDE (alpha=.05 two-sided, power=.8) ===")
scen = [
    ("250 turns, iid", 250, 1.0),
    ("250 turns, conv-clustered (deff)", 250, dev_deff),
    ("200 turns, iid", 200, 1.0),
    ("200 turns, conv-clustered (deff)", 200, dev_deff),
    ("150 turns, conv-clustered (deff)", 150, dev_deff),
]
print(f"{'scenario':36s} {'n_eff':>7s} {'MDE':>8s}")
mde_rows = []
for label, n, df in scen:
    n_eff = n / df
    mde = Z * sd_pair / math.sqrt(n_eff)
    # per-domain MDE (18 turns/domain, ~4.5 convs/domain)
    per_dom_n = n / 11
    per_dom_neff = per_dom_n / df
    mde_dom = Z * sd_pair / math.sqrt(per_dom_neff)
    print(f"{label:36s} {n_eff:7.1f} {mde:8.4f}   (每域 {per_dom_n:.1f} turn -> 域级 MDE {mde_dom:.3f})")
    mde_rows.append({"scenario": label, "n": n, "deff": df, "n_eff": round(n_eff, 2),
                     "mde": round(mde, 4), "per_domain_turns": round(per_dom_n, 1),
                     "per_domain_mde": round(mde_dom, 4)})

out = {
    "seed": SEED, "reps": REPS,
    "observed": {"pooled": round(obs, 4), "macro": round(macro_obs, 4)},
    "ci": res,
    "sd_paired_per_turn": round(sd_pair, 4),
    "dev_deff": dev_deff,
    "mde": mde_rows,
}
with open(r"E:\图rag\_probe_reteco\dev_variance.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("\nwrote dev_variance.json")
