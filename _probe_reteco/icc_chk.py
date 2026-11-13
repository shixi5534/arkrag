import io,sys,json,math,statistics
from collections import defaultdict
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8",errors="replace")
d=json.load(open(r"E:\图rag\_probe_reteco\dev_audit.json",encoding="utf-8"))
rows=d["per_turn"]
def icc1(groups):
    vals=[v for g in groups for v in g]; k=len(groups); N=len(vals)
    grand=sum(vals)/N
    ssb=sum(len(g)*(sum(g)/len(g)-grand)**2 for g in groups)
    ssw=sum((v-sum(g)/len(g))**2 for g in groups for v in g)
    msb=ssb/(k-1); msw=ssw/(N-k)
    k0=(N-sum(len(g)**2 for g in groups)/N)/(k-1)
    den=msb+(k0-1)*msw
    return (msb-msw)/den, k0
g=defaultdict(list)
for r in rows: g[r["conv"]].append(r["ndcg_hist"]-r["ndcg_base"])
rho,k0=icc1(list(g.values()))
print("paired-diff ICC=%.4f k0=%.3f -> SE inflation sqrt(1+(k0-1)rho)=%.3f"%(rho,k0,math.sqrt(1+(k0-1)*rho)))
for name in ("ndcg_base","ndcg_hist"):
    g2=defaultdict(list)
    for r in rows: g2[r["conv"]].append(r[name])
    r2,k02=icc1(list(g2.values()))
    print("%s ICC=%.4f k0=%.3f -> %.3f"%(name,r2,k02,math.sqrt(1+(k02-1)*r2)))
