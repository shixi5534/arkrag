import json, math, statistics
from collections import defaultdict
d=json.load(open(r"E:\图rag\_probe_reteco\dev_audit.json",encoding="utf-8"))
rows=d["per_turn"]
for r in rows: r["domain"]=r["conv"].split("/")[0]
# official "macro" = per-domain turn mean, then mean over 11 domains
for key in ("ndcg_base","ndcg_hist"):
    byd=defaultdict(list)
    for r in rows: byd[r["domain"]].append(r[key])
    macro=sum(sum(v)/len(v) for v in byd.values())/len(byd)
    pooled=sum(r[key] for r in rows)/len(rows)
    # SE of macro under iid-within-domain turn resampling vs conv resampling
    print(f"{key}: pooled={pooled:.4f} macro={macro:.4f}  diff={macro-pooled:+.4f}")
    # per-domain SE contribution
    tot=0.0
    for dm,v in byd.items():
        sd=statistics.pstdev(v); tot += (sd**2)/len(v)
    se_macro_iid = math.sqrt(tot)/(11*11)*11/math.sqrt(11)*math.sqrt(11)  # placeholder
    print("   per-domain n and sd:")
    for dm,v in sorted(byd.items()):
        print(f"     {dm:22s} n={len(v):4d} mean={statistics.mean(v):.4f} sd={statistics.pstdev(v):.4f}")
    break
