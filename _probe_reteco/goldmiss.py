import json, os
from collections import defaultdict
root=r"E:\图rag\_probe_reteco\track2_recor"
doms=sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))])
tot_turns=0; tot_missing=0; affected_turns=0
print(f"{'domain':22s} {'turns':>6s} {'gold@bench':>10s} {'missing':>8s} {'turns_affected':>14s}")
for d in doms:
    ids=set()
    with open(os.path.join(root,d,"documents.jsonl"),encoding="utf-8") as f:
        for ln in f:
            if ln.strip(): ids.add(json.loads(ln)["doc_id"])
    data=json.load(open(os.path.join(root,d,"benchmark_dev.json"),encoding="utf-8"))
    n=0; nb=0; miss=0; aff=0
    for c in data:
        for t in c["turns"]:
            g=t.get("gold_doc_ids") or []
            if not g: continue
            n+=1; nb+=len(g)
            mm=[x for x in g if x not in ids]
            if mm: aff+=1; miss+=len(mm)
    tot_turns+=n; tot_missing+=miss; affected_turns+=aff
    print(f"{d:22s} {n:6d} {nb:10d} {miss:8d} {aff:14d}")
print(f"{'TOTAL':22s} {tot_turns:6d} {'':10s} {tot_missing:8d} {affected_turns:14d}")
print()
print("any gold id shared across domains? (id prefix check)")
