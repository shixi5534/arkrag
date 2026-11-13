import json, os
from collections import defaultdict
root=r"E:\图rag\_probe_reteco\track2_recor"
doms=sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))])
for d in doms:
    qp=os.path.join(root,d,"qrels_dev.txt")
    if not os.path.exists(qp): print(f"{d:20s} qrels MISSING"); continue
    q=defaultdict(set)
    for ln in open(qp,encoding="utf-8"):
        if not ln.strip(): continue
        p=ln.split()
        if int(p[3])>0: q[p[0]].add(p[2])
    data=json.load(open(os.path.join(root,d,"benchmark_dev.json"),encoding="utf-8"))
    b={}
    for c in data:
        for t in c["turns"]:
            g=t.get("gold_doc_ids") or []
            if g: b[f"{c['id']}_turn_{t['turn_id']}"]=set(g)
    ok = set(q)==set(b)
    mism = [k for k in set(q)&set(b) if q[k]!=b[k]]
    print(f"{d:20s} qrels_topics={len(q):4d} bench_topics={len(b):4d} keyset_equal={ok} value_mismatch={len(mism)}")
