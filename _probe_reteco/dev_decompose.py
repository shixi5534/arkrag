"""方差来源分解 + 敏感性分析。

回答三个问题：
1. 200-turn 隐藏集的不确定性主要来自哪里（域间？簇间？turn 间？）
2. 官方宏平均口径下，聚类到底膨胀多少
3. 对 RECOR 报告的 CI/t 做敏感性分析（SD 不确定时结论怎么变）
"""
from __future__ import annotations
import io, sys, json, math, random, statistics
from collections import defaultdict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

d = json.load(open(r"E:\图rag\_probe_reteco\dev_audit.json", encoding="utf-8"))
rows = d["per_turn"]
for r in rows:
    r["domain"] = r["conv"].split("/")[0]
doms = sorted({r["domain"] for r in rows})
by_dom_conv = defaultdict(lambda: defaultdict(list))
for i, r in enumerate(rows):
    by_dom_conv[r["domain"]][r["conv"]].append(i)

print("=" * 74)
print("1. 方差分解：200-turn 隐藏集宏平均的方差从哪来")
print("=" * 74)

def macro_from(rng, key, mode, target=200 / 11):
    vals = []
    for dm in doms:
        groups = by_dom_conv[dm]
        ks = list(groups)
        if mode == "cluster":
            picked, n = [], 0
            while n < target:
                k = ks[rng.randrange(len(ks))]
                picked.extend(groups[k]); n += len(groups[k])
        elif mode == "turn":   # 逐 turn 抽（假装 turn 独立，簇被拆散）
            allidx = [i for k in ks for i in groups[k]]
            picked = [allidx[rng.randrange(len(allidx))] for _ in range(int(round(target)))]
        elif mode == "all":     # 用整域全量（只剩域内 turn 抽样噪声以外的东西）
            picked = [i for k in ks for i in groups[k]]
        vals.append(sum(rows[i][key] for i in picked) / len(picked))
    return sum(vals) / len(vals)

rng = random.Random(1)
for key in ("ndcg_base", "ndcg_hist"):
    row = {}
    for mode in ("cluster", "turn"):
        sims = [macro_from(rng, key, mode) for _ in range(3000)]
        row[mode] = statistics.pstdev(sims)
    print(f"\n{key}")
    print(f"  整簇入样(真实)  SD = {row['cluster']:.4f}   -> 95% 半宽 {1.96*row['cluster']:.4f}")
    print(f"  逐turn入样(错误) SD = {row['turn']:.4f}   -> 95% 半宽 {1.96*row['turn']:.4f}")
    print(f"  聚类膨胀（SD 比）= {row['cluster']/row['turn']:.2f}x；方差比 = {(row['cluster']/row['turn'])**2:.2f}x")

print()
print("=" * 74)
print("2. 官方宏平均口径下，'按对话聚类' 相对 '按 turn iid' 膨胀多少")
print("=" * 74)
print("  关键：官方统计量 = 先域内 turn 均值，再 11 域等权平均。")
print("  域内 turn 相关会在域内平均时被平掉一部分；域间差异不计入方差（11 域全在隐藏集）。")
print()
# analytic: var of domain mean under clustering ~ deff_domain * sigma^2 / n_domain
print(f"{'domain':22s} {'turns':>6s} {'convs':>6s} {'deff':>6s} {'sd':>7s} {'SE_iid':>8s} {'SE_clu':>8s} {'infl':>6s}")
tot_iid = tot_clu = 0.0
for dm in doms:
    idx = [i for k in by_dom_conv[dm] for i in by_dom_conv[dm][k]]
    v = [rows[i]["ndcg_hist"] for i in idx]
    n = len(v); G = len(by_dom_conv[dm])
    sd = statistics.pstdev(v)
    kish = n * n / sum(len(g) ** 2 for g in by_dom_conv[dm].values())
    deff = n / kish
    se_iid = sd / math.sqrt(n)
    se_clu = sd / math.sqrt(kish)
    tot_iid += se_iid ** 2
    tot_clu += se_clu ** 2
    print(f"{dm:22s} {n:6d} {G:6d} {deff:6.2f} {sd:7.4f} {se_iid:8.4f} {se_clu:8.4f} {se_clu/se_iid:6.2f}")
se_macro_iid = math.sqrt(tot_iid) / 11
se_macro_clu = math.sqrt(tot_clu) / 11
print(f"\n  macro SE (iid)     = {se_macro_iid:.4f}  -> CI95 半宽 {1.96*se_macro_iid:.4f}")
print(f"  macro SE (clustered)= {se_macro_clu:.4f}  -> CI95 半宽 {1.96*se_macro_clu:.4f}")
print(f"  膨胀 = {se_macro_clu/se_macro_iid:.2f}x (SE) / {(se_macro_clu/se_macro_iid)**2:.2f}x (方差)")

print()
print("=" * 74)
print("3. 对 RECOR 报告值的敏感性分析")
print("=" * 74)
print("  RECOR Table 17: H+R vs Baseline  D=+0.243  SE=0.0022  t=119.7")
print("                  H+R vs History   D=+0.039  SE=0.0013  t=30.2")
print("  报告 n = 23,768 = 2,971 turn x 8 retrievers")
print()
print("  我用 dev 实测的 per-turn 配对 SD（BM25 Base->Hist）= 0.3758 作为锚点。")
print("  注意：0.243 的差对应 SD 应更大，0.039 的差对应 SD 应更小，所以下面给区间。")
print()
hdr = f"{'SD假设':>8s} {'D':>7s} {'SE_iid(2971)':>13s} {'SE_clu(663)':>12s} {'t_clu':>7s} {'CI_clu':>9s} {'p<.05?':>7s}"
print(hdr)
print("-" * len(hdr))
for D in (0.243, 0.039):
    for sd in (0.20, 0.28, 0.3758, 0.45, 0.55):
        n_eff = 2971 / 4.48
        se = sd / math.sqrt(n_eff)
        t = D / se
        ci = 1.96 * se
        sig = "yes" if abs(t) > 1.96 else "NO"
        print(f"{sd:8.3f} {D:7.3f} {sd/math.sqrt(2971):13.5f} {se:12.4f} {t:7.2f} {ci:9.4f} {sig:>7s}")
    print()
print("  与报告值对比：")
print("    H+R vs Baseline: 报告 SE=0.0022 / CI±.004；聚类后 SE 约 0.0090~0.0246 -> CI±.018~.048")
print("                     即 CI 宽 4.5~12 倍。结论仍显著（t 约 10~27）。")
print("    H+R vs History : 报告 SE=0.0013 / CI±.004；聚类后 SE 约 0.0090~0.0246")
print("                     若真实 SD 与 Baseline 对比同量级，t 约 1.6~4.3 —— 可能不再显著。")
