"""ArkRAG 评测口径对抗审查 · 第三轮（只读）

被审对象：把「两跳路径」作为**图扩展通道**实现的 multihop 方案
（X=公共邻居 top1 → 两条 X—锚点 assertion 构造成 expansion 项 → 喂 build_context → 答案层前置 claim 句）。

用法：
    py -3 scripts/audit_round3.py            # 跑全部
    py -3 scripts/audit_round3.py --only q5  # 单节

只读：不写 src/judge/eval/goldset，不改任何评测产物。
"""

import sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import argparse
import collections
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from arkrag import pipeline as P      # noqa: E402
from arkrag import judge as J         # noqa: E402
from arkrag import query as query_mod   # noqa: E402

GEN = "g-20261103T132152Z-06a741"
ANCHOR_RE = re.compile(r"「([^」]{1,30})」")
SCOR = set(J.SCORABLE_CATEGORIES)


def hr(t):
    print("\n" + "=" * 100)
    print(t)
    print("=" * 100)


L = P.load_generation(GEN)
graph, adj, chunks, assertions = L["graph"], L["adjacency"], L["chunks"], L["assertions"]
by_name = collections.defaultdict(list)
for e in graph["entities"]:
    by_name[e["name"]].append(e["id"])
    for a in (e.get("aliases") or []):
        if isinstance(a, str):
            by_name[a].append(e["id"])
name_of = {e["id"]: e["name"] for e in graph["entities"]}
aid2a = {a["assertion_id"]: a for a in assertions}
nbr = collections.defaultdict(dict)
for eid, d in adj.items():
    for item in (d.get("out") or []):
        nid = item["neighbor_id"]
        rec = nbr[eid].setdefault(nid, {"aids": [], "rels": set(), "supp": 0})
        rec["aids"].extend(item.get("assertion_ids") or [])
        rec["rels"].add(item.get("relation") or "")
        rec["supp"] = max(rec["supp"], int(item.get("support_count") or 0))

gold_doc = json.loads((ROOT / "data/goldset/gold-v2.json").read_text("utf-8"))
V1 = {it["question_id"] for it in
      json.loads((ROOT / "data/goldset/gold-v1.json").read_text("utf-8"))["items"]}
ITEMS = {it["question_id"]: it for it in gold_doc["items"]}
chap_acc = collections.defaultdict(list)
for cid, rec in chunks.items():
    chap_acc[str(rec.get("chapter") or "")].append(str(rec.get("text") or ""))
CHAP = {k: "\n".join(v) for k, v in chap_acc.items()}


def pick(split):
    if split == "clean":
        return [it for it in gold_doc["items"] if it.get("split") == "holdout"
                and it["question_id"] not in V1 and it.get("gold_chunk_ids")]
    return [it for it in gold_doc["items"]
            if it.get("split") == split and it.get("gold_chunk_ids")]


POOLS = {s: pick(s) for s in ("dev", "holdout", "clean")}
MH = [it for it in gold_doc["items"] if it["category"] == "multihop"]


def cands(A, B):
    """lead 的排序：(sum support 降序, 度数升序, id)。"""
    na, nb = collections.defaultdict(set), collections.defaultdict(set)
    for a in A:
        for x in nbr[a]:
            na[x].add(a)
    for b in B:
        for x in nbr[b]:
            nb[x].add(b)
    excl = set(A) | set(B)
    out = []
    for x in (set(na) & set(nb)):
        if x in excl:
            continue
        sup = sum(nbr[x][s]["supp"] for s in list(na[x]) + list(nb[x]) if s in nbr[x])
        out.append((x, sup, len(nbr[x])))
    out.sort(key=lambda t: (-t[1], t[2], t[0]))
    return out


def claim(x, tgt):
    for o in tgt:
        r = nbr[x].get(o)
        if not r:
            continue
        for aid in r["aids"]:
            a = aid2a.get(aid)
            if a:
                sn = name_of.get(a.get("subject_id"), "")
                on = name_of.get(a.get("object_id"), "")
                pr = a.get("predicate") or ""
                if sn and on and pr:
                    return f"{sn}{pr}{on}。", a
    return None, None


def boot(a, b, n=20000, seed=7):
    """逐题配对 bootstrap（单侧 H1：b 更好）。返回 (Δ, 2.5%CI, 97.5%CI, p)。"""
    d = [y - x for x, y in zip(a, b)]
    r = random.Random(seed)
    idx = list(range(len(d)))
    st = sorted(sum(d[r.choice(idx)] for _ in idx) for _ in range(n))
    return sum(d), st[int(n * .025)], st[int(n * .975)], sum(1 for s in st if s <= 0) / n


def mk_expansion(x, tgt_id, a):
    """把 assertion 包装成与 Retriever.expand() 输出同构的 expansion 项。"""
    r = nbr[x].get(tgt_id) or {"supp": 0, "rels": set()}
    cid = str(a.get("chunk_id") or "")
    st, en = a.get("evidence_start"), a.get("evidence_end")
    if not cid or st is None or en is None:
        return None
    ev = str(a.get("evidence_text") or "")
    rel = sorted(r["rels"])[0] if r["rels"] else str(a.get("predicate") or "")
    return {"from_id": x, "from_name": name_of.get(x, ""),
            "neighbor_id": tgt_id, "neighbor_name": name_of.get(tgt_id, ""),
            "relation": rel, "inverse_relation": rel, "direction": "undirected",
            "symmetric": True, "edge_id": f"{x}->{tgt_id}",
            "assertion_ids": [a["assertion_id"]],
            "support_count": r["supp"], "independent_scene_count": 0,
            "matched_tiers": [], "chunk_id": cid, "evidence": ev,
            "evidence_spans": [{"chunk_id": cid, "start": int(st), "end": int(en)}],
            "anchors": [{"assertion_id": a["assertion_id"], "chunk_id": cid,
                         "evidence": ev, "start": int(st), "end": int(en)}],
            "depth": 2, "priority_tier": 1, "used_in_edges": False}


PIPE = P.build_pipeline(L, top_k=40, graph_depth=1)
PIPE["generation_id"] = GEN
PIPE["entities"] = graph["entities"]
PIPE["retriever"].budget["max_context_tokens"] = 24000
PIPE["answerer"].max_claims = 8


def run_base(splits=("dev", "holdout", "clean"), gd=1):
    PIPE["top_k"] = 40
    PIPE["graph_depth"] = gd
    PIPE["retriever"].budget["max_context_tokens"] = 24000
    PIPE["answerer"].max_claims = 8
    out = {}
    for sp in splits:
        rows = {}
        for it in POOLS[sp]:
            res = P.ask_once(PIPE, it["question"])
            rows[it["question_id"]] = {
                "base": str((res.get("answer") or {}).get("answer") or ""),
                "cat": it["category"], "q": it["question"],
                "hits": {str(h.get("chunk_id") or "") for h in
                         (res.get("search") or {}).get("hits") or []},
                "units": {str(u.get("chunk_id") or "") for u in
                          (res.get("context") or {}).get("units") or []},
            }
        out[sp] = rows
    return out


def bridged(it, base):
    """lead 的答案层动作：缺共现句则把两条 claim 前置，句数封顶 8。"""
    anc = ANCHOR_RE.findall(str(it.get("question") or ""))
    if len(anc) < 2:
        return base, None
    A, B = by_name.get(anc[0], []), by_name.get(anc[1], [])
    if not (A and B):
        return base, None
    cs = cands(A, B)
    if not cs:
        return base, None
    x = cs[0][0]
    s1, _ = claim(x, A)
    s2, _ = claim(x, B)
    if not (s1 and s2):
        return base, None
    xn = name_of.get(x, "")
    cur = [s for s in base.split("。") if s.strip()]
    has = any(anc[0] in s and xn in s for s in cur) and \
          any(anc[1] in s and xn in s for s in cur)
    if has:
        return base, xn
    return "。".join(([s1.rstrip("。"), s2.rstrip("。")] + cur)[:8]) + "。", xn


# ------------------------------------------------------------------ Q5
def q5():
    hr("§Q5 门控命中集合：题面含 ≥2 个「」且含「中间实体」")
    hit = [it for it in gold_doc["items"]
           if len(ANCHOR_RE.findall(str(it.get("question") or ""))) >= 2
           and "中间实体" in str(it.get("question") or "")]
    bycat = collections.Counter(it["category"] for it in hit)
    bysplit = collections.Counter(it.get("split") for it in hit)
    print(f"   命中 {len(hit)}/{len(gold_doc['items'])} 题；逐类 {dict(bycat)}；"
          f"逐split {dict(bysplit)}")
    print(f"   multihop 总数 {len(MH)}；误伤 {sum(v for k,v in bycat.items() if k!='multihop')}；"
          f"漏 {len([it for it in MH if it not in hit])}")
    hit2 = [it for it in gold_doc["items"]
            if len(ANCHOR_RE.findall(str(it.get("question") or ""))) >= 2]
    print(f"   若只用『≥2 个「」』：{len(hit2)} 题，逐类 "
          f"{dict(collections.Counter(i['category'] for i in hit2))}")


# ------------------------------------------------------------------ Q1
def q1(B):
    hr("§Q1a 生产 depth-1 expand() 注入的非检索块 vs 本方案的桥接证据块")
    tot = inj = 0
    percat, pertot = collections.Counter(), collections.Counter()
    for qid, r in B["dev"].items():
        if r["cat"] not in SCOR:
            continue
        tot += len(r["units"])
        for c in r["units"]:
            if c not in r["hits"]:
                inj += 1
                percat[r["cat"]] += 1
        pertot[r["cat"]] += len(r["units"])
    print(f"   生产（depth-1，全部五类）：ctx unit {tot}，其中不在 search hits 的 {inj} "
          f"= {inj/tot*100:.1f}%")
    for c in sorted(pertot):
        print(f"      {c:18s} {percat[c]:>4d}/{pertot[c]:<5d} = {percat[c]/pertot[c]*100:4.1f}%")
    n = absent = 0
    for it in POOLS["dev"]:
        if it["category"] != "multihop":
            continue
        anc = ANCHOR_RE.findall(it["question"])
        if len(anc) < 2:
            continue
        A, Bn = by_name.get(anc[0], []), by_name.get(anc[1], [])
        if not (A and Bn):
            continue
        cs = cands(A, Bn)
        if not cs:
            continue
        x = cs[0][0]
        for tgt in (A, Bn):
            _, a = claim(x, tgt)
            if a:
                cid = str(a.get("chunk_id") or "")
                n += 1
                r = B["dev"][it["question_id"]]
                absent += (cid not in r["units"] and cid not in r["hits"])
    print(f"\n   本方案 dev multihop 桥接证据块 {n} 条，注入前既不在 hits 也不在 ctx 的 "
          f"{absent} 条 = {absent/max(1,n)*100:.1f}%")
    print(f"   ⇒ 与生产 {inj/tot*100:.1f}% 相比放大 {absent/max(1,n)/(inj/tot):.0f} 倍")


# ------------------------------------------------------------------ Q2/Q3
def q2q3(B):
    hr("§Q2/Q3 方案读数、形态、pass ≡ (X==gold)、bootstrap")
    res = {}
    for sp in ("dev", "holdout", "clean"):
        ok = n = fired = 0
        eq = collections.Counter()
        lens, ns, mx = [], [], []
        va, vb = [], []
        for qid, r in B[sp].items():
            if r["cat"] not in SCOR:
                continue
            txt = r["base"]
            xn = None
            if r["cat"] == "multihop":
                txt, xn = bridged(ITEMS[qid], r["base"])
                if xn:
                    fired += 1
                    p = J.judge_answer("multihop", txt, ITEMS[qid],
                                       question=r["q"], chapter_texts=CHAP)["pass"] is True
                    g = str(ITEMS[qid]["answer"])
                    eq["pass"] += p
                    eq["x_eq_gold"] += (xn == g)
                    eq["both"] += (p and xn == g)
                    eq["pass_not_gold"] += (p and xn != g)
                    eq["gold_not_pass"] += ((not p) and xn == g)
            p0 = J.judge_answer(r["cat"], r["base"], ITEMS[qid],
                                question=r["q"], chapter_texts=CHAP)["pass"] is True
            p1 = J.judge_answer(r["cat"], txt, ITEMS[qid],
                                question=r["q"], chapter_texts=CHAP)["pass"] is True
            va.append(int(p0)); vb.append(int(p1))
            n += 1
            ok += p1
            sl = [s for s in txt.split("。") if s.strip()]
            lens.append(len(txt)); ns.append(len(sl))
            mx.append(max((len(s) for s in sl), default=0))
        L2 = sorted(lens)
        res[sp] = (va, vb)
        print(f"   {sp:8s} 方案 {ok}/{n} = {ok/n*100:.2f}%  触发 {fired}")
        print(f"      形态：均长={sum(lens)/len(lens):.1f} P50={L2[len(L2)//2]} "
              f"P95={L2[int(len(L2)*.95)]} max={L2[-1]} 均句数={sum(ns)/len(ns):.2f}")
        print(f"      越界：总长>300字 {sum(1 for x in lens if x>300)}/{len(lens)}  "
              f"句数>8 {sum(1 for x in ns if x>8)}/{len(ns)}  "
              f"单句>60字 {sum(1 for m in mx if m>60)}/{len(mx)}")
        if r["cat"] == "multihop" or fired:
            print(f"      multihop：pass={eq['pass']} X==gold={eq['x_eq_gold']} "
                  f"同时={eq['both']} pass但X≠gold={eq['pass_not_gold']} "
                  f"X==gold但未pass={eq['gold_not_pass']}")
    print("\n   逐题配对 bootstrap（单侧 H1：方案更好，20000 次，seed=7）：")

    def boot(a, b):
        d = [y - x for x, y in zip(a, b)]
        r = random.Random(7)
        idx = list(range(len(d)))
        st = sorted(sum(d[r.choice(idx)] for _ in idx) for _ in range(20000))
        return sum(d), st[500], st[19500], sum(1 for s in st if s <= 0) / 20000

    for sp in ("dev", "holdout", "clean"):
        a, b = res[sp]
        obs, lo, hi, p = boot(a, b)
        ok = obs >= 1 and lo > 0 and p < .05
        print(f"     {sp:8s} {sum(a)}→{sum(b)}  Δ={obs:+d} ({obs/len(a)*100:+.2f}pp) "
              f"CI=[{lo:+d},{hi:+d}] p={p:.4f}  {'PASS' if ok else 'FAIL'}")


# ------------------------------------------------------------------ Q4
def q4():
    hr("§Q4 排序键扫描：同一候选集 9 种排序（判据 pass 口径）")
    KEYS = ["sum_sup", "nrel_then_sup", "deg_asc", "deg_desc", "id", "rand",
            "nrel_asc", "maxsup", "deg_over_sup"]

    def order(A, B, k):
        na, nb = collections.defaultdict(set), collections.defaultdict(set)
        for a in A:
            for x in nbr[a]:
                na[x].add(a)
        for b in B:
            for x in nbr[b]:
                nb[x].add(b)
        ex = set(A) | set(B)
        rows = []
        for x in (set(na) & set(nb)):
            if x in ex:
                continue
            sup = sum(nbr[x][s]["supp"] for s in list(na[x]) + list(nb[x]) if s in nbr[x])
            rels = set()
            for s in list(na[x]) + list(nb[x]):
                r = nbr[x].get(s)
                if r:
                    rels |= r["rels"]
            rows.append({"id": x, "sup": sup, "deg": len(nbr[x]), "nrel": len(rels)})
        key = {"sum_sup": lambda t: (-t["sup"], t["deg"], t["id"]),
               "nrel_then_sup": lambda t: (-t["nrel"], -t["sup"], t["deg"], t["id"]),
               "deg_asc": lambda t: (t["deg"], t["id"]),
               "deg_desc": lambda t: (-t["deg"], t["id"]),
               "id": lambda t: t["id"],
               "nrel_asc": lambda t: (t["nrel"], t["deg"], t["id"]),
               "maxsup": lambda t: (-t["sup"] / max(1, t["deg"]), t["id"]),
               "deg_over_sup": lambda t: (t["deg"] / max(1, t["sup"]), t["id"]),
               "rand": lambda t: (0,)}[k]
        if k == "rand":
            random.Random(20261107).shuffle(rows)
        else:
            rows.sort(key=key)
        return rows

    print(f"   {'排序键':16s} {'dev':>9s} {'holdout':>9s} {'clean':>9s} {'ALL':>8s}")
    tab = {}
    for k in KEYS:
        got = {"dev": 0, "holdout": 0, "clean": 0}
        den = {"dev": 0, "holdout": 0, "clean": 0}
        for it in MH:
            sp = "clean" if (it.get("split") == "holdout"
                             and it["question_id"] not in V1) else it.get("split")
            if sp not in got or not it.get("gold_chunk_ids"):
                continue
            den[sp] += 1
            anc = ANCHOR_RE.findall(it["question"])
            if len(anc) < 2:
                continue
            A, Bn = by_name.get(anc[0], []), by_name.get(anc[1], [])
            if not (A and Bn):
                continue
            rows = order(A, Bn, k)
            if not rows:
                continue
            x = rows[0]["id"]
            s1, _ = claim(x, A)
            s2, _ = claim(x, Bn)
            if not (s1 and s2):
                continue
            txt = f"{name_of.get(x,'')}。" + s1 + s2
            if J.judge_answer("multihop", txt, it, question=it["question"])["pass"] is True:
                got[sp] += 1
        tab[k] = got
        print(f"   {k:16s} " + " ".join(f"{got[s]:>3d}/{den[s]:<4}" for s in
                                       ("dev", "holdout", "clean"))
              + f" {sum(got.values()):>4d}/{sum(den.values())}")
    best = max(KEYS, key=lambda k: tab[k]["dev"])
    print(f"\n   dev 最优 = {best}（{tab[best]['dev']}）；lead 案 sum_sup dev = "
          f"{tab['sum_sup']['dev']} hold = {tab['sum_sup']['holdout']} "
          f"clean = {tab['sum_sup']['clean']}")
    print(f"   候选集内随机 dev = {tab['rand']['dev']}")


# ------------------------------------------------------------------ Q7
def _four_modes():
    """四模式真跑：base / 仅第4步注入 / 仅第5步前置 / 第4+5步。"""
    def ask(question, mode):
        qa = query_mod.analyze_query(
            question, entities=graph["entities"], aliases=graph.get("alias_index"),
            vocab=P.vocab, context=getattr(PIPE["retriever"], "query_context", None))
        found = PIPE["retriever"].search(qa, top_k=PIPE["top_k"])
        exps = list(PIPE["retriever"].expand(
            [c["entity_id"] for c in qa.get("entity_candidates") or []],
            depth=PIPE["graph_depth"], query_analysis=qa,
            seed_chunk_ids=[h["chunk_id"] for h in found["hits"]])["expansions"])
        info = None
        anc = ANCHOR_RE.findall(question)
        if len(anc) == 2:
            A, B = by_name.get(anc[0], []), by_name.get(anc[1], [])
            if A and B:
                cs = cands(A, B)
                if cs:
                    x = cs[0][0]
                    s1, a1 = claim(x, A)
                    s2, a2 = claim(x, B)
                    if s1 and s2:
                        info = {"x": name_of.get(x, ""), "n": len(cs), "s1": s1, "s2": s2}
                        if mode in ("inject", "both"):
                            e1 = mk_expansion(x, A[0], a1)
                            e2 = mk_expansion(x, B[0], a2)
                            if e1 and e2:
                                exps = [e1, e2] + exps
        ctx = PIPE["retriever"].build_context(found["hits"], exps, query_analysis=qa)
        ans = PIPE["answerer"].answer(
            question, ctx, generation_id=GEN,
            top_score=float(found["stats"].get("top_score") or 0.0),
            query_analysis=qa, retrieval_stats=found["stats"])
        txt = str((ans or {}).get("answer") or "")
        if mode in ("prepend", "both") and info:
            cur = [s for s in txt.split("。") if s.strip()]
            xn = info["x"]
            has = any(anc[0] in s and xn in s for s in cur) and \
                  any(anc[1] in s and xn in s for s in cur)
            if not has:
                txt = "。".join(([info["s1"].rstrip("。"), info["s2"].rstrip("。")]
                                + cur)[:8]) + "。"
        return txt, info, ctx

    out = {}
    for sp in ("dev", "holdout", "clean"):
        for mode in ("base", "inject", "prepend", "both"):
            vec, lens, sents, mxs = [], [], [], []
            for it in POOLS[sp]:
                if it["category"] not in SCOR:
                    continue
                txt, info, ctx = ask(it["question"], mode)
                p = J.judge_answer(it["category"], txt, it, question=it["question"],
                                   chapter_texts=CHAP)["pass"] is True
                vec.append(int(p))
                ss = [x for x in txt.split("。") if x.strip()]
                lens.append(len(txt))
                sents.append(len(ss))
                mxs.append(max((len(x) for x in ss), default=0))
            out[(sp, mode)] = {"vec": vec, "len": lens, "nsent": sents, "maxsent": mxs}
    return out


def q7():
    hr("§Q7 步骤归因：第4步(build_context 注入) vs 第5步(答案层前置) 的边际贡献")
    R = _four_modes()
    print(f"{'split':8s} {'mode':9s} {'pass':>10s}  {'Δ vs base':>10s}")
    for sp in ("dev", "holdout", "clean"):
        b = sum(R[(sp, "base")]["vec"])
        for mode in ("base", "inject", "prepend", "both"):
            v = sum(R[(sp, mode)]["vec"])
            print(f"{sp:8s} {mode:9s} {v:4d}/{len(R[(sp,mode)]['vec'])}  {v-b:+d}")
    hr("§Q7b bootstrap（逐题配对，单侧，20000 次，seed=7）")
    for sp in ("dev", "holdout", "clean"):
        for mode in ("inject", "prepend", "both"):
            obs, lo, hi, p = boot(R[(sp, "base")]["vec"], R[(sp, mode)]["vec"])
            ok = obs >= 1 and lo > 0 and p < .05
            print(f"  {sp:8s} vs {mode:9s} Δ={obs:+d} CI=[{lo:+d},{hi:+d}] "
                  f"p={p:.4f} {'PASS' if ok else 'FAIL'}")
    hr("§Q7c 形态红线（总长≤300 / 单句≤60 / 句数≤8）")
    for sp in ("dev", "holdout", "clean"):
        for mode in ("base", "both"):
            d = R[(sp, mode)]
            print(f"  {sp:8s} {mode:5s} 总长max={max(d['len'])} >300={sum(1 for x in d['len'] if x>300)}  "
                  f"句数max={max(d['nsent'])} >8={sum(1 for x in d['nsent'] if x>8)}  "
                  f"单句max={max(d['maxsent'])} >60={sum(1 for x in d['maxsent'] if x>60)}  "
                  f"n={len(d['len'])}")
    return R


# ------------------------------------------------------------------ Q10
def q10():
    hr("§Q10 X 候选来源：全图邻接（lead 新版） vs 本次 ctx 断言（接地版）")
    for sp in ("dev", "holdout", "clean"):
        c = collections.Counter()
        for it in POOLS[sp]:
            if it["category"] != "multihop":
                continue
            q = it["question"]
            anc = ANCHOR_RE.findall(q)
            if len(anc) != 2:
                continue
            A, B = by_name.get(anc[0], []), by_name.get(anc[1], [])
            if not (A and B):
                c["锚点不可映射"] += 1
                continue
            c["题数"] += 1
            g = str(it["answer"])
            cg = cands(A, B)
            c["全图有候选"] += bool(cg)
            if cg:
                c["全图top1==gold"] += (name_of.get(cg[0][0], "") == g)
            res = P.ask_once(PIPE, q)
            ctx = res.get("context") or {}
            trip = [(a, u["chunk_id"]) for u in ctx.get("units") or []
                    for a in (u.get("assertions") or [])]
            ent_of = collections.defaultdict(list)
            for i, (a, cid) in enumerate(trip):
                for nm in (a.get("subject"), a.get("object")):
                    if nm:
                        ent_of[nm].append(i)
            others = set(anc)
            def touching(name):
                o = collections.defaultdict(list)
                for i in ent_of.get(name, ()):
                    a = trip[i][0]
                    other = a.get("object") if a.get("subject") == name else a.get("subject")
                    if other and other not in others:
                        o[other].append(i)
                return o
            ta, tb = touching(anc[0]), touching(anc[1])
            common = set(ta) & set(tb)
            c["接地有候选"] += bool(common)
            if common:
                def key(x):
                    return (-sum(int(trip[i][0].get("support_count") or 0)
                                 for i in ta[x] + tb[x]), x)
                c["接地top1==gold"] += (sorted(common, key=key)[0] == g)
        n = c["题数"]
        print(f"  {sp:8s} n={n}  全图: 有候选 {c['全图有候选']}  top1==gold {c['全图top1==gold']}"
              f"   |   接地: 有候选 {c['接地有候选']}  top1==gold {c['接地top1==gold']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["q1", "q2", "q4", "q5", "q7", "q10"], default=None)
    main_a = ap.parse_args()
    need = main_a.only

    if need in (None, "q7"):
        q7()
    if need in (None, "q10"):
        q10()
    if need in (None, "q5"):
        q5()
    if need in (None, "q4"):
        q4()
    if need in (None, "q1", "q2"):
        B = run_base()
        if need in (None, "q1"):
            q1(B)
        if need in (None, "q2"):
            q2q3(B)


if __name__ == "__main__":
    main()



