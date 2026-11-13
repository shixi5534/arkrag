import json, os, re, math, time
src = open(r"E:\图rag\_probe_reteco\tokcmp.py", encoding="utf-8").read()
head = src.split("for stem,stop,label in")[0]
ns = {}
exec(compile(head, "head", "exec"), ns)
tok_luceneish=ns["tok_luceneish"]; ids=ns["ids"]; texts=ns["texts"]; ndcg=ns["ndcg"]; root=ns["root"]
data=json.load(open(os.path.join(root,"benchmark_dev.json"),encoding="utf-8"))

def build(with_hist):
    queries={};gold={}
    for c in data:
        for t in c["turns"]:
            g=t.get("gold_doc_ids") or []
            if not g: continue
            parts=[t["query"]]
            if with_hist:
                h=t.get("conversation_history","")
                if h and h!="No previous conversation.":
                    parts.append("Conversation History:\n"+h)
            qid=f"{c['id']}_turn_{t['turn_id']}"
            queries[qid]="\n\n".join(parts); gold[qid]=set(g)
    return queries,gold

def run(with_hist,k1=0.9,b=0.4):
    queries,gold=build(with_hist)
    docs=[tok_luceneish(t,True,True) for t in texts]
    N=len(docs); dl=[len(d) for d in docs]; avg=sum(dl)/N
    tf=[];df={}
    for d in docs:
        c={}
        for w in d: c[w]=c.get(w,0)+1
        tf.append(c)
        for w in c: df[w]=df.get(w,0)+1
    idf={w: math.log(1+(N-n+0.5)/(n+0.5)) for w,n in df.items()}
    out=[]
    for qid,q in queries.items():
        qt=tok_luceneish(q,True,True)
        sc=[]
        for i in range(N):
            s=0.0;tfi=tf[i]
            for w in qt:
                f=tfi.get(w)
                if not f: continue
                s+=idf.get(w,0)*(f*(k1+1))/(f+k1*(1-b+b*dl[i]/avg))
            sc.append((ids[i],s))
        sc.sort(key=lambda x:(-x[1],x[0]))
        out.append(ndcg([d for d,_ in sc],gold[qid]))
    return sum(out)/len(out)

for wh,label in [(False,"2a Base (current turn)"),(True,"2a +History")]:
    t0=time.time(); v=run(wh)
    print(f"{label:24s} nDCG@10={v:.4f}  ({time.time()-t0:.0f}s)")
print()
print("official drones dev: Base=0.1695  +History=0.2954")
