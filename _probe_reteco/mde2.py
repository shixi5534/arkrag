import json, math, statistics
from collections import defaultdict
d=json.load(open(r"E:\图rag\_probe_reteco\dev_audit.json",encoding="utf-8"))
rows=d["per_turn"]
print("turns",len(rows),"convs",d["cluster"]["convs"],"kish",d["cluster"]["kish"],"deff",d["cluster"]["pooled_deff"])
print("paired delta observed", d["paired_delta"]["observed"], "sd_iid", d["paired_delta"]["iid"]["sd"], "sd_cluster", d["paired_delta"]["cluster"]["sd"])

# ICC(1) of the paired difference itself -- the quantity that governs clustered SE of a paired test
def icc1(groups):
    vals=[v for g in groups for v in g]; k=len(groups); N=len(vals)
    grand=sum(vals)/N
    ssb=sum(len(g)*(sum(g)/len(g)-grand)**2 for g in groups)
    ssw=sum((v-sum(g)/len(g))**2 for g in groups for v in g)
    msb=ssb/(k-1); msw=ssw/(N-k)
    k0=(N-sum(len(g)**2 for g in groups)/N)/(k-1)
    den=msb+(k0-1)*msw
    return (msb-msw)/den if den>0 else None, msb, msw, k0

g=defaultdict(list)
for r in rows: g[r["conv"]].append(r["ndcg_hist"]-r["ndcg_base"])
rho,msb,msw,k0=icc1(list(g.values()))
print()
print("ICC(1) of PAIRED DIFFERENCE (hist-base) = %.4f  (k0=%.2f)"%(rho,k0))
print("  -> clustered SE / iid SE = sqrt(1+(k0-1)*rho) = %.3f" % math.sqrt(1+(k0-1)*rho))

for name in ("ndcg_base","ndcg_hist"):
    g2=defaultdict(list)
    for r in rows: g2[r["conv"]].append(r[name])
    r2,_,_,k02=icc1(list(g2.values()))
    print("ICC(1) of %-10s = %.4f  k0=%.2f -> SE inflation %.3f"%(name,r2,k02,math.sqrt(1+(k02-1)*r2)))

# MDE using measured paired-difference SD
sd_paired = statistics.pstdev([r["ndcg_hist"]-r["ndcg_base"] for r in rows])
print()
print("SD of paired difference (per turn) = %.4f"%sd_paired)
print("Z_{.975}+Z_{.8} = 2.8016")
print()
print("=== MDE for hidden test ~200 turns, 11 domains ===")
deff=d["cluster"]["pooled_deff"]
for label, n_eff in [("iid (assume independence)", 200.0), ("clustered, measured deff=%.2f"%deff, 200.0/deff)]:
    for sd in [sd_paired]:
        mde=2.8016*sd/math.sqrt(n_eff)
        print(f"  {label:34s} n_eff={n_eff:6.1f} SD={sd:.4f} -> MDE={mde:.4f}")
print()
print("=== leaderboard-style neighbour gap context ===")
print("  published dev macro spread across domains (2a hist):")
vals=[(k,v["ndcg_hist"]) for k,v in d["per_domain"].items()]
vals.sort(key=lambda x:-x[1])
for k,v in vals: print(f"    {k:22s} {v:.4f}")
print("  domain range = %.4f (max-min)"%(vals[0][1]-vals[-1][1]))
