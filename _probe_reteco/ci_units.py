import io,sys,json,statistics,math
from collections import defaultdict
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8",errors="replace")
d=json.load(open(r"E:\图rag\_probe_reteco\dev_audit.json",encoding="utf-8"))
rows=d["per_turn"]
for r in rows: r["domain"]=r["conv"].split("/")[0]
byd=defaultdict(list)
for r in rows: byd[r["domain"]].append(r["ndcg_hist"])
means=[statistics.mean(v) for v in byd.values()]
macro=statistics.mean(means)
print("my hist per-domain macro = %.4f"%macro)
print("SD across 11 domain means = %.4f"%statistics.stdev(means))
se_dom=statistics.stdev(means)/math.sqrt(11)
print("SE treating DOMAIN as unit = %.4f -> CI95 half=%.4f  (i.e. [%.3f, %.3f])"%(se_dom,1.96*se_dom,macro-1.96*se_dom,macro+1.96*se_dom))
print()
print("RECOR reports DIVER .584 (95%% CI: .520 to .649) -> half-width = %.4f"%(0.649-0.520)/2)
print("RECOR reports Average row 95%% CI = +-0.004")
print()
print("=== which sampling unit reproduces each reported CI width? ===")
for label,n,unit in [("turn iid, n=23768",23768,"turn"),("turn iid, n=2971",2971,"turn"),("conv clustered, n=2971/deff",2971/4.48,"conv"),("domain, n=11",11,"domain")]:
    se=0.3758/math.sqrt(n)
    print(f"  {label:28s} SE={se:.4f} half-width={1.96*se:.4f}")
