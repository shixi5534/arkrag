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
se_dom=statistics.stdev(means)/math.sqrt(11)
print("my hist per-domain macro = %.4f"%macro)
print("SD of 11 domain means    = %.4f"%statistics.stdev(means))
print("SE (domain as unit)      = %.4f -> CI95 [%.3f, %.3f]  half=%.4f"%(se_dom,macro-1.96*se_dom,macro+1.96*se_dom,1.96*se_dom))
print()
print("RECOR DIVER .584 CI [.520,.649] -> half-width = %.4f"%((0.649-0.520)/2))
print("RECOR Average row CI = +-0.004")
print()
print("=== CI half-width implied by each sampling unit (using my dev SD=0.3758) ===")
for label,n in [("turn iid n=23,768",23768),("turn iid n=2,971",2971),("conv clustered n=2,971/4.48",2971/4.48),("domain n=11",11)]:
    se=0.3758/math.sqrt(n)
    print("  %-30s SE=%.4f  half=%.4f"%(label,se,1.96*se))
