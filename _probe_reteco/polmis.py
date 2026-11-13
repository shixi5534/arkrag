import json, os
from collections import defaultdict
root=r"E:\图rag\_probe_reteco\track2_recor"
d="politics"
q=defaultdict(set)
for ln in open(os.path.join(root,d,"qrels_dev.txt"),encoding="utf-8"):
    if not ln.strip(): continue
    p=ln.split()
    if int(p[3])>0: q[p[0]].add(p[2])
data=json.load(open(os.path.join(root,d,"benchmark_dev.json"),encoding="utf-8"))
b={};meta={}
for c in data:
    for t in c["turns"]:
        g=t.get("gold_doc_ids") or []
        if g:
            b[f"{c['id']}_turn_{t['turn_id']}"]=set(g); meta[f"{c['id']}_turn_{t['turn_id']}"]=(c["id"],t["turn_id"])
for k in sorted(set(q)&set(b)):
    if q[k]!=b[k]:
        print("MISMATCH", k, "conv", meta[k][0], "turn", meta[k][1])
        print("  in qrels not bench:", sorted(q[k]-b[k]))
        print("  in bench not qrels:", sorted(b[k]-q[k]))
