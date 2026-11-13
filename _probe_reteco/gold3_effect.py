import io,sys,json
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8",errors="replace")
d=json.load(open(r"E:\图rag\_probe_reteco\dev_audit.json",encoding="utf-8"))
rows=d["per_turn"]
bad=[r for r in rows if r["qid"].startswith("ex_88512_") and r["domain" if "domain" in r else "conv"].startswith("politics")]
bad=[r for r in rows if r["conv"].endswith("/ex_88512")]
print("affected turns:", len(bad))
for r in bad:
    print("  ", r["qid"], "base=%.4f hist=%.4f"%(r["ndcg_base"],r["ndcg_hist"]),
          "gold_n=%d"%len(r["gold"]), "contains doc_3:", "politics_ex_88512_doc_3" in r["gold"])
print()
# if we drop doc_3 from gold and recompute IDCG, nDCG rises. Estimate effect on politics macro.
import math
def ndcg10_rerank(dcg_relevant, n_gold):
    ideal=sum(1.0/math.log2(i+2) for i in range(min(n_gold,10)))
    return dcg_relevant/ideal if ideal>0 else 0.0
for r in bad:
    n_bad=len(r["gold"]); n_ok=n_bad-1
    # recover dcg from reported ndcg (base)
    ideal_bad=sum(1.0/math.log2(i+2) for i in range(min(n_bad,10)))
    dcg=r["ndcg_base"]*ideal_bad
    new=ndcg10_rerank(dcg,n_ok)
    print("  %s base: %.4f -> %.4f (if doc_3 dropped)  delta=%+.4f"%(r["qid"],r["ndcg_base"],new,new-r["ndcg_base"]))
