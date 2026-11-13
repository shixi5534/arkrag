"""四套口径的最终对照表（单一 seed / 单一 reps，供论文直接引用）。

口径定义：
  A  iid@turn           : 逐 turn 有放回重抽，池化均值
  B  cluster@conv       : 按对话整簇重抽，池化均值
  C  macro + turn       : 域内逐 turn 重抽，域均值再等权平均（官方统计量，但单位错）
  D  macro + conv       : 域内按对话整簇重抽，域均值再等权平均（官方统计量，正确单位）
"""
from __future__ import annotations
import io, sys, json, math, random, statistics
from collections import defaultdict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SEED = 20261112
REPS = 20000

d = json.load(open(r"E:\图rag\_probe_reteco\dev_audit.json", encoding="utf-8"))
rows = d["per_turn"]
for r in rows:
    r["domain"] = r["conv"].split("/")[0]
doms = sorted({r["domain"] for r in rows})
a = [r["ndcg_base"] for r in rows]
b = [r["ndcg_hist"] for r in rows]
conv = [r["conv"] for r in rows]
dom = [r["domain"] for r in rows]

idx_by_conv = defaultdict(list)
for i, c in enumerate(conv): idx_by_conv[c].append(i)
idx_by_dom_conv = defaultdict(lambda: defaultdict(list))
for i, (dm, c) in enumerate(zip(dom, conv)): idx_by_dom_conv[dm][c].append(i)

def diff(i): return b[i] - a[i]

def A(rng):
    n = len(a)
    return sum(diff(rng.randrange(n)) for _ in range(n)) / n

def B(rng):
    ks = list(idx_by_conv); G = len(ks)
    tot = cnt = 0
    for _ in range(G):
        g = idx_by_conv[ks[rng.randrange(G)]]
        tot += sum(diff(i) for i in g); cnt += len(g)
    return tot / cnt

def _macro(rng, sampler):
    vals = []
    for dm in doms:
        vals.append(sampler(idx_by_dom_conv[dm], rng))
    return sum(vals) / len(vals)

def C(rng):
    def s(groups, rng):
        allidx = [i for v in groups.values() for i in v]
        n = len(allidx)
        return sum(diff(allidx[rng.randrange(n)]) for _ in range(n)) / n
    return _macro(rng, s)

def D(rng):
    def s(groups, rng):
        ks = list(groups); G = len(ks)
        tot = cnt = 0
        for _ in range(G):
            g = groups[ks[rng.randrange(G)]]
            tot += sum(diff(i) for i in g); cnt += len(g)
        return tot / cnt
    return _macro(rng, s)

def pct(s, q):
    s = sorted(s); return s[max(0, min(len(s) - 1, int(q / 100 * len(s))))]

# 观测值
obs_pool = sum(diff(i) for i in range(len(a))) / len(a)
byd = defaultdict(list)
for i, dm in enumerate(dom): byd[dm].append(diff(i))
obs_macro = sum(sum(v) / len(v) for v in byd.values()) / len(byd)

print(f"turns={len(a)} convs={len(set(conv))} domains={len(doms)}  REPs={REPS} seed={SEED}")
print(f"observed pooled delta = {obs_pool:+.4f}   observed macro delta = {obs_macro:+.4f}\n")

rowsout = {}
print(f"{'口径':32s} {'Δ':>9s} {'CI95 low':>10s} {'CI95 high':>10s} {'宽度':>8s} {'SE':>8s}")
print("-" * 82)
for label, fn, obs in [
    ("A iid@turn (pooled)", A, obs_pool),
    ("B cluster@conv (pooled)", B, obs_pool),
    ("C macro + turn", C, obs_macro),
    ("D macro + conv (正确)", D, obs_macro),
]:
    rng = random.Random(SEED)
    s = [fn(rng) for _ in range(REPS)]
    lo, hi = pct(s, 2.5), pct(s, 97.5)
    sd = statistics.pstdev(s)
    print(f"{label:32s} {obs:+9.4f} {lo:+10.4f} {hi:+10.4f} {hi-lo:8.4f} {sd:8.4f}")
    rowsout[label] = {"delta": round(obs, 4), "ci95": [round(lo, 4), round(hi, 4)],
                      "width": round(hi - lo, 4), "se": round(sd, 5)}

wa = rowsout["A iid@turn (pooled)"]["width"]
print("\n相对 A 的宽度比 / SE 比:")
for k, v in rowsout.items():
    print(f"  {k:32s} {v['width']/wa:5.2f}x / {v['se']/rowsout['A iid@turn (pooled)']['se']:5.2f}x")

# 水平（非配对）的宏平均 SE：iid vs 聚类
print("\n=== 水平估计（非配对）的宏平均 SE ===")
for key, arr in (("Base", a), ("+History", b)):
    tot_iid = tot_clu = 0.0
    for dm in doms:
        idx = [i for v in idx_by_dom_conv[dm].values() for i in v]
        vals = [arr[i] for i in idx]
        n = len(vals); sd = statistics.pstdev(vals)
        kish = n * n / sum(len(v) ** 2 for v in idx_by_dom_conv[dm].values())
        tot_iid += (sd / math.sqrt(n)) ** 2
        tot_clu += (sd / math.sqrt(kish)) ** 2
    se_i, se_c = math.sqrt(tot_iid) / 11, math.sqrt(tot_clu) / 11
    print(f"  {key:9s} SE_iid={se_i:.5f} (半宽{1.96*se_i:.4f})  "
          f"SE_clu={se_c:.5f} (half{1.96*se_c:.4f})  膨胀={se_c/se_i:.2f}x")

print("\n=== per-turn nDCG 的 SD 区间（供 §2.2 引用）===")
for key, arr in (("Base", a), ("+History", b)):
    sds = []
    for dm in doms:
        idx = [i for v in idx_by_dom_conv[dm].values() for i in v]
        sds.append(statistics.pstdev([arr[i] for i in idx]))
    print(f"  {key:9s} per-domain SD 范围 [{min(sds):.4f}, {max(sds):.4f}]  池化 SD {statistics.pstdev(arr):.4f}")

with open(r"E:\图rag\_probe_reteco\ci_final.json", "w", encoding="utf-8") as f:
    json.dump({"seed": SEED, "reps": REPS, "observed_pooled": round(obs_pool, 4),
               "observed_macro": round(obs_macro, 4), "table": rowsout}, f,
              ensure_ascii=False, indent=2)
print("\nwrote ci_final.json")
