import io,sys,json,math,statistics
from collections import defaultdict
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8",errors="replace")
d=json.load(open(r"E:\图rag\_probe_reteco\dev_audit.json",encoding="utf-8"))
rows=d["per_turn"]
print("=== 核对补充计划书里引用的每个数字 ===")
print("per_turn rows:", len(rows))
convs={r["conv"] for r in rows}; print("convs:", len(convs))
sd=statistics.pstdev([r["ndcg_hist"]-r["ndcg_base"] for r in rows])
print("per-turn paired SD: %.4f"%sd)
print("macro base/hist:", d["macro_dev"])
print("cluster:", d["cluster"])
print("paired:", d["paired_delta"])
v=json.load(open(r"E:\图rag\_probe_reteco\dev_variance.json",encoding="utf-8"))
print("\nCI widths:", {k:x["width"] for k,x in v["ci"].items()})
for m in v["mde"]: print("  ", m["scenario"], "n_eff=%.1f MDE=%.4f"%(m["n_eff"],m["mde"]))
h=json.load(open(r"E:\图rag\_probe_reteco\dev_hidden_sim.json",encoding="utf-8"))
print("\nsingle_system:", json.dumps(h["single_system"], ensure_ascii=False, indent=1))
