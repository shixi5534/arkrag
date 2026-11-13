import json, os, re, math, time

# ---------- compact Porter stemmer (1980), stdlib only ----------
class Porter:
    def __init__(self):
        self.v = re.compile(r"[aeiouy]")
    def _cons(self,w,i):
        c=w[i]
        if c in "aeiou": return False
        if c=="y": return True if i==0 else not self._cons(w,i-1)
        return True
    def _m(self,s):
        n=0;i=0;L=len(s)
        while i<L and self._cons(s,i): i+=1
        while i<L:
            while i<L and not self._cons(s,i): i+=1
            if i>=L: break
            while i<L and self._cons(s,i): i+=1
            n+=1
        return n
    def _vowel_in(self,s): return bool(self.v.search(s))
    def _dbl(self,w):
        return len(w)>=2 and w[-1]==w[-2] and self._cons(w,len(w)-1)
    def _cvc(self,w):
        if len(w)<3: return False
        return (self._cons(w,len(w)-3) and not self._cons(w,len(w)-2) and self._cons(w,len(w)-1)
                and w[-1] not in "wxy")
    def stem(self,w):
        if len(w)<=2: return w
        # step1a
        if w.endswith("sses"): w=w[:-2]
        elif w.endswith("ies"): w=w[:-2]
        elif w.endswith("ss"): pass
        elif w.endswith("s"): w=w[:-1]
        # step1b
        flag=False
        if w.endswith("eed"):
            if self._m(w[:-3])>0: w=w[:-1]
        elif w.endswith("ed"):
            s=w[:-2]
            if self._vowel_in(s): w=s; flag=True
        elif w.endswith("ing"):
            s=w[:-3]
            if self._vowel_in(s): w=s; flag=True
        if flag:
            if w.endswith(("at","bl","iz")): w=w+"e"
            elif self._dbl(w) and w[-1] not in "lsz": w=w[:-1]
            elif self._m(w)==1 and self._cvc(w): w=w+"e"
        # step1c
        if w.endswith("y") and self._vowel_in(w[:-1]): w=w[:-1]+"i"
        # step2
        for suf,rep in [("ational","ate"),("tional","tion"),("enci","ence"),("anci","ance"),
                        ("izer","ize"),("bli","ble"),("alli","al"),("entli","ent"),("eli","e"),
                        ("ousli","ous"),("ization","ize"),("ation","ate"),("ator","ate"),
                        ("alism","al"),("iveness","ive"),("fulness","ful"),("ousness","ous"),
                        ("aliti","al"),("iviti","ive"),("biliti","ble"),("logi","log")]:
            if w.endswith(suf) and self._m(w[:-len(suf)])>0:
                w=w[:-len(suf)]+rep; break
        # step3
        for suf,rep in [("icate","ic"),("ative",""),("alize","al"),("iciti","ic"),
                        ("ical","ic"),("ful",""),("ness","")]:
            if w.endswith(suf) and self._m(w[:-len(suf)])>0:
                w=w[:-len(suf)]+rep; break
        # step4
        for suf in ["al","ance","ence","er","ic","able","ible","ant","ement","ment","ent",
                    "ion","ou","ism","ate","iti","ous","ive","ize"]:
            if w.endswith(suf):
                s=w[:-len(suf)]
                if self._m(s)>1 and (suf!="ion" or (s and s[-1] in "st")): w=s
                break
        # step5
        if w.endswith("e"):
            s=w[:-1]
            if self._m(s)>1 or (self._m(s)==1 and not self._cvc(s)): w=s
        if w.endswith("ll") and self._m(w)>1: w=w[:-1]
        return w

P = Porter()
STOP = set("""a about above after again against all am an and any are aren't as at be because been before being below
between both but by can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from
further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how
how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor not of off
on once only or other ought our ours ourselves out over own same shan't she she'd she'll she's should shouldn't so some
such than that that's the their theirs them themselves then there there's these they they'd they'll they're they've this
those through to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where
where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself
yourselves""".split())

root = r"E:\图rag\_probe_reteco\track2_recor\drones"
ids=[];texts=[]
with open(os.path.join(root,"documents.jsonl"),encoding="utf-8") as f:
    for ln in f:
        if ln.strip():
            d=json.loads(ln); ids.append(d["doc_id"]); texts.append(d["content"])

TOK=re.compile(r"[A-Za-z0-9]+")
def tok_luceneish(t, stem=True, stop=True):
    ws=[w.lower() for w in TOK.findall(t or "")]
    if stop: ws=[w for w in ws if w not in STOP]
    if stem: ws=[P.stem(w) for w in ws]
    return ws

data=json.load(open(os.path.join(root,"benchmark_dev.json"),encoding="utf-8"))
queries={};gold={}
for c in data:
    for t in c["turns"]:
        g=t.get("gold_doc_ids") or []
        if g:
            queries[f"{c['id']}_turn_{t['turn_id']}"]=t["query"]; gold[f"{c['id']}_turn_{t['turn_id']}"]=set(g)

def ndcg(ranked,g,k=10):
    dcg=sum(1.0/math.log2(i+2) for i,d in enumerate(ranked[:k]) if d in g)
    idcg=sum(1.0/math.log2(i+2) for i in range(min(len(g),k)))
    return dcg/idcg if idcg>0 else 0.0

def run(stem,stop,k1=0.9,b=0.4):
    docs=[tok_luceneish(t,stem,stop) for t in texts]
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
        qt=tok_luceneish(q,stem,stop)
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

for stem,stop,label in [(False,False,"raw [A-Za-z0-9]+"),(False,True,"+stopwords"),(True,False,"+porter"),(True,True,"+porter +stopwords")]:
    t0=time.time()
    v=run(stem,stop)
    print(f"{label:24s} nDCG@10={v:.4f}  ({time.time()-t0:.0f}s)")
print()
print("official drones 2a dev (Lucene analyzer + gensim LuceneBM25) = 0.1695")
