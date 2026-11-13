"""RETECO dev 全量预演：BM25(porter+stop) Base / +History，per-turn nDCG@10，
ICC(1) by conversation, iid vs clustered bootstrap, 方差膨胀。

只用标准库。array 压缩倒排索引以控制内存。
"""
from __future__ import annotations
import json, os, re, math, random, time, statistics
from array import array
from collections import defaultdict

ROOT = r"E:\图rag\_probe_reteco\track2_recor"
SEED = 20261112
REPS = 2000
K1, B = 0.9, 0.4

# ---------------- Porter (compact) ----------------
class Porter:
    def _cons(self, w, i):
        c = w[i]
        if c in "aeiou":
            return False
        if c == "y":
            return True if i == 0 else not self._cons(w, i - 1)
        return True
    def _m(self, s):
        n = 0; i = 0; L = len(s)
        while i < L and self._cons(s, i): i += 1
        while i < L:
            while i < L and not self._cons(s, i): i += 1
            if i >= L: break
            while i < L and self._cons(s, i): i += 1
            n += 1
        return n
    def _vowel_in(self, s): return any(c in "aeiou" for c in s) or (len(s) > 1 and "y" in s[1:])
    def _dbl(self, w): return len(w) >= 2 and w[-1] == w[-2] and self._cons(w, len(w) - 1)
    def _cvc(self, w):
        if len(w) < 3: return False
        return (self._cons(w, len(w) - 3) and not self._cons(w, len(w) - 2)
                and self._cons(w, len(w) - 1) and w[-1] not in "wxy")
    def stem(self, w):
        if len(w) <= 2: return w
        if w.endswith("sses"): w = w[:-2]
        elif w.endswith("ies"): w = w[:-2]
        elif w.endswith("ss"): pass
        elif w.endswith("s"): w = w[:-1]
        flag = False
        if w.endswith("eed"):
            if self._m(w[:-3]) > 0: w = w[:-1]
        elif w.endswith("ed"):
            s = w[:-2]
            if self._vowel_in(s): w = s; flag = True
        elif w.endswith("ing"):
            s = w[:-3]
            if self._vowel_in(s): w = s; flag = True
        if flag:
            if w.endswith(("at", "bl", "iz")): w = w + "e"
            elif self._dbl(w) and w[-1] not in "lsz": w = w[:-1]
            elif self._m(w) == 1 and self._cvc(w): w = w + "e"
        if w.endswith("y") and self._vowel_in(w[:-1]): w = w[:-1] + "i"
        for suf, rep in [("ational","ate"),("tional","tion"),("enci","ence"),("anci","ance"),
                         ("izer","ize"),("bli","ble"),("alli","al"),("entli","ent"),("eli","e"),
                         ("ousli","ous"),("ization","ize"),("ation","ate"),("ator","ate"),
                         ("alism","al"),("iveness","ive"),("fulness","ful"),("ousness","ous"),
                         ("aliti","al"),("iviti","ive"),("biliti","ble"),("logi","log")]:
            if w.endswith(suf) and self._m(w[:-len(suf)]) > 0:
                w = w[:-len(suf)] + rep; break
        for suf, rep in [("icate","ic"),("ative",""),("alize","al"),("iciti","ic"),
                         ("ical","ic"),("ful",""),("ness","")]:
            if w.endswith(suf) and self._m(w[:-len(suf)]) > 0:
                w = w[:-len(suf)] + rep; break
        for suf in ["al","ance","ence","er","ic","able","ible","ant","ement","ment","ent",
                    "ion","ou","ism","ate","iti","ous","ive","ize"]:
            if w.endswith(suf):
                s = w[:-len(suf)]
                if self._m(s) > 1 and (suf != "ion" or (s and s[-1] in "st")): w = s
                break
        if w.endswith("e"):
            s = w[:-1]
            if self._m(s) > 1 or (self._m(s) == 1 and not self._cvc(s)): w = s
        if w.endswith("ll") and self._m(w) > 1: w = w[:-1]
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
TOK = re.compile(r"[A-Za-z0-9]+")

def analyze(text):
    out = []
    for w in TOK.findall((text or "").lower()):
        if w in STOP: continue
        out.append(P.stem(w))
    return out

def ndcg_at_10(ranked, gold):
    dcg = 0.0
    for i, d in enumerate(ranked[:10]):
        if d in gold: dcg += 1.0 / math.log2(i + 2)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), 10)))
    return dcg / ideal if ideal > 0 else 0.0

# ---------------- per-domain index ----------------
class DomainIndex:
    def __init__(self, domain):
        d = os.path.join(ROOT, domain)
        self.doc_ids = []
        lens = []
        post = defaultdict(lambda: (array("i"), array("i")))  # term -> (docs, tfs)
        with open(os.path.join(d, "documents.jsonl"), encoding="utf-8") as f:
            for ln in f:
                if not ln.strip(): continue
                rec = json.loads(ln)
                idx = len(self.doc_ids)
                self.doc_ids.append(rec["doc_id"])
                toks = analyze(rec.get("content", ""))
                lens.append(len(toks))
                tf = defaultdict(int)
                for t in toks: tf[t] += 1
                for t, c in tf.items():
                    docs, tfs = post[t]
                    docs.append(idx); tfs.append(c)
        self.lens = lens
        self.N = len(lens)
        self.avgdl = (sum(lens) / self.N) if self.N else 1.0
        self.post = post
        self.idf = {t: math.log(1 + (self.N - len(p[0]) + 0.5) / (len(p[0]) + 0.5))
                    for t, p in post.items()}
        self.vocab = len(post)

    def search(self, qterms, topk=10):
        scores = defaultdict(float)
        N, avgdl, k1, b = self.N, self.avgdl, K1, B
        for t in qterms:
            p = self.post.get(t)
            if p is None: continue
            docs, tfs = p
            idf = self.idf[t]
            for i in range(len(docs)):
                di = docs[i]; f = tfs[i]
                scores[di] += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * self.lens[di] / avgdl))
        if not scores: return []
        ranked = sorted(scores.items(), key=lambda x: (-x[1], self.doc_ids[x[0]]))
        return [self.doc_ids[i] for i, _ in ranked[:topk]]

# ---------------- topics ----------------
def topics(domain, with_hist):
    d = os.path.join(ROOT, domain)
    data = json.load(open(os.path.join(d, "benchmark_dev.json"), encoding="utf-8"))
    rows = []
    for c in data:
        for t in c["turns"]:
            gold = t.get("gold_doc_ids") or []
            if not gold: continue
            parts = [t["query"]]
            if with_hist:
                h = t.get("conversation_history", "")
                if h and h != "No previous conversation.":
                    parts.append("Conversation History:\n" + h)
            rows.append({
                "qid": f"{c['id']}_turn_{t['turn_id']}",
                "conv": f"{domain}/{c['id']}",   # conversation ids are only unique per domain
                "turn_depth": int(t["turn_id"]) if str(t["turn_id"]).isdigit() else 1,
                "q": "\n\n".join(parts),
                "gold": set(gold),
            })
    return rows

# ---------------- stats ----------------
def kish(sizes):
    n = sum(sizes)
    return (n * n / sum(s * s for s in sizes)) if sizes else 0.0

def bca_free_ci(samples, lo=2.5, hi=97.5):
    s = sorted(samples)
    n = len(s)
    return s[max(0, int(lo / 100 * n))], s[min(n - 1, int(hi / 100 * n))]

def bootstrap_mean_iid(vals, reps, rng):
    n = len(vals)
    return [sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps)]

def bootstrap_mean_cluster(vals, clusters, reps, rng):
    groups = defaultdict(list)
    for v, c in zip(vals, clusters): groups[c].append(v)
    keys = list(groups)
    G = len(keys)
    out = []
    for _ in range(reps):
        tot, cnt = 0.0, 0
        for _ in range(G):
            g = groups[keys[rng.randrange(G)]]
            tot += sum(g); cnt += len(g)
        out.append(tot / cnt if cnt else 0.0)
    return out

def bootstrap_paired_delta_iid(a, b, reps, rng):
    n = len(a)
    out = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        out.append(sum(b[i] - a[i] for i in idx) / n)
    return out

def bootstrap_paired_delta_cluster(a, b, clusters, reps, rng):
    groups = defaultdict(list)
    for i, c in enumerate(clusters): groups[c].append(i)
    keys = list(groups); G = len(keys)
    out = []
    for _ in range(reps):
        d, cnt = 0.0, 0
        for _ in range(G):
            g = groups[keys[rng.randrange(G)]]
            d += sum(b[i] - a[i] for i in g); cnt += len(g)
        out.append(d / cnt if cnt else 0.0)
    return out

def icc1(groups):
    """ICC(1) one-way random effects, unbalanced."""
    vals = [v for g in groups for v in g]
    k = len(groups); N = len(vals)
    if k < 2 or N <= k: return None
    grand = sum(vals) / N
    ssb = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in groups)
    ssw = sum((v - sum(g) / len(g)) ** 2 for g in groups for v in g)
    msb = ssb / (k - 1); msw = ssw / (N - k)
    k0 = (N - sum(len(g) ** 2 for g in groups) / N) / (k - 1)
    den = msb + (k0 - 1) * msw
    return ((msb - msw) / den) if den > 0 else None

# ---------------- main ----------------
def main():
    rng = random.Random(SEED)
    domains = sorted([d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))])
    per_domain = {}
    all_rows = []
    t_start = time.time()
    for dom in domains:
        t0 = time.time()
        idx = DomainIndex(dom)
        base_rows = topics(dom, with_hist=False)
        hist_rows = topics(dom, with_hist=True)
        for rb, rh in zip(base_rows, hist_rows):
            assert rb["qid"] == rh["qid"]
            rb["ndcg_base"] = ndcg_at_10(idx.search(analyze(rb["q"])), rb["gold"])
            rb["ndcg_hist"] = ndcg_at_10(idx.search(analyze(rh["q"])), rh["gold"])
        mb = sum(r["ndcg_base"] for r in base_rows) / len(base_rows)
        mh = sum(r["ndcg_hist"] for r in base_rows) / len(base_rows)
        sizes = defaultdict(int)
        for r in base_rows: sizes[r["conv"]] += 1
        per_domain[dom] = {
            "turns": len(base_rows), "convs": len(sizes),
            "corpus": idx.N, "vocab": idx.vocab,
            "ndcg_base": round(mb, 4), "ndcg_hist": round(mh, 4),
            "deff": round(len(base_rows) / kish(list(sizes.values())), 2),
        }
        all_rows.extend(base_rows)
        print(f"[{dom:20s}] docs={idx.N:6d} turns={len(base_rows):4d} convs={len(sizes):3d} "
              f"base={mb:.4f} hist={mh:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        del idx

    macro_b = sum(v["ndcg_base"] for v in per_domain.values()) / len(per_domain)
    macro_h = sum(v["ndcg_hist"] for v in per_domain.values()) / len(per_domain)
    print(f"\nMACRO dev: Base={macro_b:.4f}  +History={macro_h:.4f}  "
          f"(official 0.1827 / 0.4379)")

    clusters = [r["conv"] for r in all_rows]
    sizes = defaultdict(int)
    for c in clusters: sizes[c] += 1
    kk = kish(list(sizes.values()))
    print(f"\nturns={len(all_rows)} convs={len(sizes)} mean_cluster={len(all_rows)/len(sizes):.2f} "
          f"Kish={kk:.1f} pooled_deff={len(all_rows)/kk:.2f}")

    for name in ("ndcg_base", "ndcg_hist"):
        groups = defaultdict(list)
        for r in all_rows: groups[r["conv"]].append(r[name])
        print(f"ICC(1) of per-turn {name}: {icc1(list(groups.values())):.4f}")

    a = [r["ndcg_base"] for r in all_rows]
    b = [r["ndcg_hist"] for r in all_rows]
    obs = sum(y - x for x, y in zip(a, b)) / len(a)
    d_iid = bootstrap_paired_delta_iid(a, b, REPS, rng)
    d_clu = bootstrap_paired_delta_cluster(a, b, clusters, REPS, rng)
    def stats(s):
        m = statistics.mean(s); sd = statistics.pstdev(s)
        lo, hi = bca_free_ci(s)
        return m, sd, lo, hi
    mi, sdi, loi, hii = stats(d_iid)
    mc, sdc, loc, hic = stats(d_clu)
    print(f"\npaired delta (Base->+History) observed = {obs:+.4f}")
    print(f"  iid     CI95=[{loi:+.4f},{hii:+.4f}]  width={hii-loi:.4f}  sd={sdi:.4f}")
    print(f"  cluster CI95=[{loc:+.4f},{hic:+.4f}]  width={hic-loc:.4f}  sd={sdc:.4f}  by conv")
    print(f"  variance inflation = {(sdc/sdi)**2:.2f}x   SE ratio = {sdc/sdi:.2f}x")
    print(f"\ntotal wall = {time.time()-t_start:.0f}s")

    out = {
        "seed": SEED, "reps": REPS,
        "bm25": {"k1": K1, "b": B, "tokenizer": "porter+stopwords(unigram)"},
        "per_domain": per_domain,
        "macro_dev": {"base": round(macro_b, 4), "hist": round(macro_h, 4)},
        "official_dev": {"base": 0.1827, "hist": 0.4379},
        "cluster": {"turns": len(all_rows), "convs": len(sizes),
                    "mean_cluster": round(len(all_rows)/len(sizes), 2),
                    "kish": round(kk, 2), "pooled_deff": round(len(all_rows)/kk, 2)},
        "paired_delta": {
            "observed": round(obs, 4),
            "iid": {"mean": round(mi,4), "sd": round(sdi,4), "ci95": [round(loi,4), round(hii,4)],
                    "width": round(hii-loi,4)},
            "cluster": {"mean": round(mc,4), "sd": round(sdc,4), "ci95": [round(loc,4), round(hic,4)],
                        "width": round(hic-loc,4), "cluster_key": "conversation_id"},
            "variance_inflation": round((sdc/sdi)**2, 3),
        },
        "per_turn": [{k: (list(v) if isinstance(v, set) else v) for k, v in r.items()} for r in all_rows],
    }
    with open(r"E:\图rag\_probe_reteco\dev_audit.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("wrote dev_audit.json")

if __name__ == "__main__":
    main()
