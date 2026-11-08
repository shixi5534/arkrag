"""Red-team audit of four conclusions A/B/C/D about ArkRAG retrieval budget & claims.

Read-only w.r.t. src/**, tests/**, data/goldset/**.  Writes only:
  data/audit_redteam.json
All console output is ASCII (GBK console).

Sections
  S1  goldset overlap recount (gold-v1 vs gold-v2, clean holdout sizes)
  S2  cross_chunk gold completeness (gold chapters vs graph entity chapters vs text scan)
  S3  judge calibration ladder: same answers, several judges, per-category
  S4  structural-category feasibility probe (can answer.py-style templates do it?)
  S5  claims x budget matrix + paired bootstrap (dev / holdout / clean holdout)
  S6  end-to-end latency per budget (not just assembly)
"""
import io
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(r"E:\图rag")
sys.path.insert(0, str(ROOT / "src"))

GEN = "g-20261103T132152Z-06a741"
SEP = "\u3002"
QUOTE_RE = re.compile(r"\u300c([^\u300d]{1,40})\u300d")
TITLE_RE = re.compile(r"\u300a([^\u300b]{1,40})\u300b")
SPLITRE = r"[\u3001,\uff0c/\s]+"
STRUCTURAL = {"time_conflict", "adversarial", "negative_abstain"}

OUT = {}


def log(*a):
    print(*a, flush=True)


def cands(item):
    return [str(item.get("answer") or "")] + [str(x) for x in (item.get("acceptable_answers") or []) if x]


def boot(va, vb, iters=4000, seed=20261107):
    """Paired bootstrap over per-question 0/1 vectors; same scheme as the claims."""
    flags = [b - a for a, b in zip(va, vb)]
    n = len(flags)
    rnd = random.Random(seed)
    means = []
    for _ in range(iters):
        s = 0
        for _ in range(n):
            s += flags[rnd.randrange(n)]
        means.append(100.0 * s / n)
    means.sort()
    obs = 100.0 * sum(flags) / n
    return (obs, means[int(0.025 * iters)], means[int(0.975 * iters)],
            sum(1 for m in means if m <= 0) / iters)


def claims(text):
    return [s.strip() for s in text.split(SEP) if s.strip()]


def anchors(q):
    return [m.group(1).strip() for m in QUOTE_RE.finditer(q) if m.group(1).strip()]


def gold_chapter_set(item):
    want = set()
    for g in cands(item):
        want.update(x for x in re.split(SPLITRE, g) if x)
    return want


def got_chapter_set(text):
    return set(x for x in re.split(SPLITRE, text or "") if x)


# ---------------------------------------------------------------- judges
def judge_current(cat, text, item, q):
    """The judge the conclusions under audit actually use (tmp_scorecard/tmp_crossmode)."""
    cl = claims(text)
    gs = cands(item)
    if cat in ("single_chunk", "alias"):
        return any(g in text for g in gs)
    if cat == "cross_chunk":
        want = gold_chapter_set(item)
        got = got_chapter_set(text)
        return bool(want) and got == want
    if cat == "direct_relation":
        an = anchors(q)
        return any(any(g in s for g in gs) and an and all(a in s for a in an) for s in cl)
    if cat == "multihop":
        an = anchors(q)
        for s in cl:
            if not any(g in s for g in gs):
                continue
            if not any(a in s for a in an):
                continue
            if any(g in an for g in gs):
                return False
            return True
        return False
    return None


def judge_multihop_all_anchors(cat, text, item, q):
    """Same as current but multihop requires ALL anchors in the one claim."""
    if cat != "multihop":
        return judge_current(cat, text, item, q)
    cl = claims(text)
    gs = cands(item)
    an = anchors(q)
    for s in cl:
        if not any(g in s for g in gs):
            continue
        if not (an and all(a in s for a in an)):
            continue
        if any(g in an for g in gs):
            return False
        return True
    return False


def judge_exact(cat, text, item, q):
    """Un-gameable for short answers: a claim must BE the gold, not merely contain it."""
    cl = claims(text)
    gs = [c.strip() for c in cands(item) if c.strip()]
    if cat in ("single_chunk", "alias", "multihop"):
        for s in cl:
            t = s.strip().rstrip("\u3002\uff1b;")
            if t in gs:
                return True
            if any(t == g for g in gs):
                return True
        return False
    if cat == "cross_chunk":
        want = gold_chapter_set(item)
        got = got_chapter_set(text)
        return bool(want) and got == want
    if cat == "direct_relation":
        an = anchors(q)
        for s in cl:
            if not any(g in s for g in gs):
                continue
            if an and all(a in s for a in an):
                return True
        return False
    return None


def judge_strict_anchored(cat, text, item, q):
    """tmp_metric_audit.py 'strict': gold claim must also carry >=1 question anchor."""
    cl = claims(text)
    gs = cands(item)
    an = anchors(q)
    for s in cl:
        if not any(g in s for g in gs):
            continue
        if any(a in s for a in an):
            return True
    return False


JUDGES = {
    "current": judge_current,
    "multihop_all_anchors": judge_multihop_all_anchors,
    "exact": judge_exact,
    "strict_anchored": judge_strict_anchored,
}


# ---------------------------------------------------------------- S1
def s1_goldset():
    v1 = json.loads((ROOT / "data/goldset/gold-v1.json").read_text(encoding="utf-8"))["items"]
    v2 = json.loads((ROOT / "data/goldset/gold-v2.json").read_text(encoding="utf-8"))["items"]
    id1 = {str(x["question_id"]) for x in v1}
    id2 = {str(x["question_id"]) for x in v2}
    hold = [x for x in v2 if str(x.get("split")) == "holdout"]
    dev = [x for x in v2 if str(x.get("split")) == "dev"]
    hold_gold = [x for x in hold if x.get("gold_chunk_ids")]
    hold_contam = [x for x in hold_gold if str(x["question_id"]) in id1]
    clean = [x for x in hold if str(x["question_id"]) not in id1]
    clean_gold = [x for x in clean if x.get("gold_chunk_ids")]
    scorable = lambda xs: [x for x in xs if str(x.get("category")) not in STRUCTURAL]
    r = {
        "n_v1": len(v1), "n_v2": len(v2),
        "n_v1_ids_in_v2": len(id1 & id2),
        "n_v1_ids_not_in_v2": len(id1 - id2),
        "n_dev": len(dev), "n_dev_with_gold": sum(1 for x in dev if x.get("gold_chunk_ids")),
        "n_holdout": len(hold), "n_holdout_with_gold": len(hold_gold),
        "n_holdout_gold_also_in_v1": len(hold_contam),
        "n_clean_holdout": len(clean), "n_clean_holdout_with_gold": len(clean_gold),
        "n_clean_holdout_scorable": len(scorable(clean_gold)),
        "clean_holdout_cat": dict(Counter(str(x.get("category")) for x in clean_gold)),
        "holdout_cat": dict(Counter(str(x.get("category")) for x in hold_gold)),
        "dev_cat": dict(Counter(str(x.get("category")) for x in dev if x.get("gold_chunk_ids"))),
    }
    log("== S1 goldset")
    for k, v in r.items():
        log("   %-34s %s" % (k, v))
    OUT["s1_goldset"] = r
    return v1, v2


# ---------------------------------------------------------------- S2
def s2_cross_gold(v2):
    ch = {}
    with open(ROOT / "data/arkrag/chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            ch[o["chunk_id"]] = (o.get("chapter") or "", o.get("text") or "")
    g = json.loads((ROOT / "data/arkrag/graph.json").read_text(encoding="utf-8"))
    by_name = defaultdict(list)
    for e in g["entities"]:
        by_name[e["name"]].append(e)
    cx = [x for x in v2 if str(x.get("category")) == "cross_chunk"]
    rows = []
    for x in cx:
        q = str(x.get("question") or "")
        m = QUOTE_RE.search(q)
        nm = m.group(1) if m else None
        gold_ch = sorted({ch[c][0] for c in (x.get("gold_chunk_ids") or []) if c in ch})
        rec = {"qid": x["question_id"], "split": x["split"], "name": nm,
               "n_gold": len(gold_ch), "gold_chapters": gold_ch}
        ents = by_name.get(nm) or []
        if ents:
            e = ents[0]
            ech = set(e.get("chunks") or [])
            ent_ch = sorted({ch[c][0] for c in ech if c in ch})
            rec["n_entity"] = len(ent_ch)
            rec["entity_chapters"] = ent_ch
            rec["gold_chunks_subset_of_entity"] = set(x.get("gold_chunk_ids") or []) <= ech
            rec["entity_chapters_minus_gold"] = sorted(set(ent_ch) - set(gold_ch))
        else:
            rec["n_entity"] = None
            rec["entity_chapters"] = None
            rec["gold_chunks_subset_of_entity"] = None
            rec["entity_chapters_minus_gold"] = None
        # text scan
        keys = [nm] if nm else []
        if ents:
            keys = [k for k in ([nm] + list(ents[0].get("aliases") or [])) if k and len(k) >= 2]
        nmch, alch = set(), set()
        for cid, (chap, txt) in ch.items():
            if nm and nm in txt:
                nmch.add(chap)
            if any(k in txt for k in keys):
                alch.add(chap)
        rec["n_text_name"] = len(nmch)
        rec["n_text_alias"] = len(alch)
        rec["text_name_chapters"] = sorted(nmch)
        rows.append(rec)
    withE = [r for r in rows if r["n_entity"] is not None]
    summ = {
        "n_cx": len(rows), "n_with_graph_entity": len(withE),
        "n_no_graph_entity": len(rows) - len(withE),
        "gold_chapters_eq_entity_chapters": sum(1 for r in withE if set(r["gold_chapters"]) == set(r["entity_chapters"])),
        "entity_chapters_strictly_more_than_gold": sum(1 for r in withE if set(r["gold_chapters"]) < set(r["entity_chapters"])),
        "gold_chunks_subset_of_entity_chunks": sum(1 for r in withE if r["gold_chunks_subset_of_entity"]),
        "text_name_chapters_gt_gold": sum(1 for r in rows if r["n_text_name"] > r["n_gold"]),
        "text_name_chapters_eq_gold": sum(1 for r in rows if r["n_text_name"] == r["n_gold"]),
        "no_graph_entity_qids": [r["qid"] for r in rows if r["n_entity"] is None],
    }
    log("\n== S2 cross_chunk gold completeness")
    for k, v in summ.items():
        log("   %-40s %s" % (k, v))
    log("   %-16s %-8s %-12s %4s %5s %5s  entity_chapters_minus_gold" % ("qid", "split", "name", "nG", "nEnt", "nTxt"))
    for r in sorted(withE, key=lambda r: -(r["n_entity"] - r["n_gold"])):
        if r["entity_chapters_minus_gold"]:
            log("   %-16s %-8s %-12s %4d %5d %5d  %s" % (
                r["qid"], r["split"], str(r["name"])[:12], r["n_gold"], r["n_entity"],
                r["n_text_name"], ",".join(r["entity_chapters_minus_gold"])))
    OUT["s2_cross_gold"] = {"summary": summ, "rows": rows}
    return rows


# ---------------------------------------------------------------- S4
def s4_structural(v2):
    """Feasibility probe: can a template path (graph types / constant) answer these?"""
    g = json.loads((ROOT / "data/arkrag/graph.json").read_text(encoding="utf-8"))
    name2type = {}
    for e in g["entities"]:
        name2type.setdefault(e["name"], e.get("type") or "")
    dev = [x for x in v2 if str(x.get("split")) == "dev" and x.get("gold_chunk_ids")]

    def passes(text, item):
        gs = cands(item)
        return any(g in text for g in gs), any(c.strip() == text.strip() for c in gs)

    res = {}
    # adversarial: build answer purely from graph entity types
    adv = [x for x in dev if x.get("category") == "adversarial"]
    arows = []
    for x in adv:
        q = str(x["question"])
        qs = anchors(q)
        if len(qs) < 2:
            arows.append({"qid": x["question_id"], "skip": "anchors<2", "anchors": qs})
            continue
        a, b = qs[0], qs[1]
        ta, tb = name2type.get(a, "?"), name2type.get(b, "?")
        text = ("\u4e0d\u662f\u540c\u4e00\u4e2a\u5b9e\u4f53\uff1a\u300c%s\u300d\u7684\u7c7b\u578b\u662f%s\uff0c"
                "\u300c%s\u300d\u7684\u7c7b\u578b\u662f%s" % (a, ta, b, tb))
        hit, exact = passes(text, x)
        arows.append({"qid": x["question_id"], "a": a, "b": b, "type_a": ta, "type_b": tb,
                      "generated": text, "raw_hit": hit, "exact_hit": exact,
                      "gold": cands(x)[0]})
    res["adversarial"] = {
        "n": len(adv),
        "n_type_looked_up": sum(1 for r in arows if r.get("type_a") not in (None, "?")),
        "n_raw_hit": sum(1 for r in arows if r.get("raw_hit")),
        "n_exact_hit": sum(1 for r in arows if r.get("exact_hit")),
        "rows": arows,
    }
    log("\n== S4 structural feasibility")
    log("   adversarial: %d q, type looked up %d, raw-hit %d, exact-hit %d" % (
        res["adversarial"]["n"], res["adversarial"]["n_type_looked_up"],
        res["adversarial"]["n_raw_hit"], res["adversarial"]["n_exact_hit"]))

    # time_conflict: single constant template
    tc = [x for x in dev if x.get("category") == "time_conflict"]
    consts = Counter(cands(x)[0] for x in tc)
    best = consts.most_common(1)[0][0] if consts else ""
    tpass = sum(1 for x in tc if any(best in g or g in best for g in cands(x)))
    res["time_conflict"] = {
        "n": len(tc), "distinct_gold_answers": len(consts),
        "most_common_gold": best, "n_pass_with_single_constant": tpass,
        "question_templates": dict(Counter(
            "stage" if "\u4e0d\u540c\u9636\u6bb5" in str(x["question"]) else "other" for x in tc)),
        "n_should_abstain": sum(1 for x in tc if x.get("should_abstain")),
    }
    log("   time_conflict: %d q, distinct gold answers %d, one constant answers %d, should_abstain %d" % (
        len(tc), len(consts), tpass, res["time_conflict"]["n_should_abstain"]))

    # negative_abstain with gold
    na = [x for x in dev if x.get("category") == "negative_abstain"]
    nconst = Counter(cands(x)[0] for x in na)
    nbest = nconst.most_common(1)[0][0] if nconst else ""
    res["negative_abstain"] = {
        "n_with_gold": len(na), "distinct_gold_answers": len(nconst),
        "most_common_gold": nbest,
        "n_pass_with_single_constant": sum(1 for x in na if nbest in cands(x)),
        "n_should_abstain": sum(1 for x in na if x.get("should_abstain")),
    }
    log("   negative_abstain(with gold): %d q, distinct gold %d, one constant answers %d" % (
        len(na), len(nconst), res["negative_abstain"]["n_pass_with_single_constant"]))
    OUT["s4_structural"] = res
    return res


# ---------------------------------------------------------------- S5 + S3
ARMS = (("c5_6000", 5, 6000), ("c5_12000", 5, 12000),
        ("c12_6000", 12, 6000), ("c12_12000", 12, 12000), ("c12_40000", 12, 40000))


def s3_s5(v1, v2):
    from arkrag import pipeline as P
    log("\n== loading pipeline")
    t0 = time.perf_counter()
    loaded = P.load_generation(GEN)
    pipe = P.build_pipeline(loaded, top_k=20, graph_depth=1)
    log("   load_ms=%.0f" % ((time.perf_counter() - t0) * 1000))
    r = pipe["retriever"]

    v1ids = {str(x["question_id"]) for x in v1}
    items = [x for x in v2 if x.get("gold_chunk_ids")]
    log("   items=%d (dev+holdout with gold)" % len(items))

    rows = []
    tstart = time.perf_counter()
    for i, item in enumerate(items, 1):
        q = str(item.get("question") or "")
        cat = str(item.get("category") or "")
        res = P.ask_once(pipe, q)
        qa = res["query_analysis"]
        hits = res["search"]["hits"]
        expanded = r.expand([c["entity_id"] for c in qa.get("entity_candidates") or []],
                            depth=1, query_analysis=qa,
                            seed_chunk_ids=[h["chunk_id"] for h in hits])
        top = float((res["search"]["stats"] or {}).get("top_score") or 0.0)
        per = {}
        for tag, cmax, budget in ARMS:
            ctx = r.build_context(hits, expanded["expansions"], token_budget=budget, query_analysis=qa)
            pipe["answerer"].max_claims = cmax
            a = pipe["answerer"].answer(q, ctx, generation_id=GEN, top_score=top, query_analysis=qa)
            text = str(a.get("answer") or "")
            per[tag] = {"text": text, "units": len(ctx["units"]),
                        "chars": sum(len(str(u.get("text") or "")) for u in ctx["units"]),
                        "abstained": bool(a.get("abstained")),
                        "j": {k: f(cat, text, item, q) for k, f in JUDGES.items()}}
        pipe["answerer"].max_claims = 5
        rows.append({"qid": str(item["question_id"]), "cat": cat, "split": str(item.get("split")),
                     "in_v1": str(item["question_id"]) in v1ids,
                     "question": q, "gold": cands(item)[0],
                     "anchors": anchors(q), "per": per})
        if i % 50 == 0:
            log("   %d/%d  %.0fs" % (i, len(items), time.perf_counter() - tstart))

    scorable = [x for x in rows if x["cat"] not in STRUCTURAL]

    def subset(split=None, clean=False):
        s = [x for x in scorable if split is None or x["split"] == split]
        if clean:
            s = [x for x in s if not x["in_v1"]]
        return s

    def rate(sub, tag, jn):
        if not sub:
            return (0, 0, 0.0)
        k = sum(1 for x in sub if x["per"][tag]["j"][jn])
        return (k, len(sub), 100.0 * k / len(sub))

    matrices = {}
    log("\n== S5/S3 rates [judge=current, cross_chunk=set equality]")
    log("   %-22s %-6s %s" % ("subset", "n", " ".join("%14s" % t for t, _, _ in ARMS)))
    for name, sub in (("dev", subset("dev")), ("holdout", subset("holdout")),
                      ("clean_holdout", subset("holdout", clean=True))):
        matrices[name] = {"n": len(sub), "cells": {}}
        cells = []
        for tag, _, _ in ARMS:
            k, n, p = rate(sub, tag, "current")
            matrices[name]["cells"][tag] = {"pass": k, "n": n, "rate": round(p, 2)}
            cells.append("%5d/%-4d%6.2f" % (k, n, p))
        log("   %-22s %-6d %s" % (name, len(sub), " ".join(cells)))

    log("\n== S3 judge ladder (dev scorable n=%d)" % len(subset("dev")))
    dev = subset("dev")
    log("   %-16s %-22s %s" % ("arm", "judge", "pass/rate"))
    ladder = {}
    for tag, _, _ in ARMS:
        ladder[tag] = {}
        for jn in JUDGES:
            k, n, p = rate(dev, tag, jn)
            ladder[tag][jn] = {"pass": k, "n": n, "rate": round(p, 2)}
            log("   %-16s %-22s %d/%d %.2f%%" % (tag, jn, k, n, p))

    log("\n== S3 per-category pass counts under each judge (dev)")
    cats = ["single_chunk", "alias", "direct_relation", "multihop", "cross_chunk"]
    percat = {}
    for c in cats:
        sub = [x for x in dev if x["cat"] == c]
        percat[c] = {"n": len(sub), "cells": {}}
        for jn in JUDGES:
            row = []
            for tag, _, _ in ARMS:
                k = sum(1 for x in sub if x["per"][tag]["j"][jn])
                row.append(k)
            percat[c]["cells"][jn] = row
            log("   %-16s %-22s n=%2d  %s" % (c, jn, len(sub), " ".join("%3d" % v for v in row)))
    log("   arm order: " + " ".join(t for t, _, _ in ARMS))

    # ---- which questions flip
    flips = {}
    for a, b in (("c5_6000", "c12_12000"), ("c12_6000", "c12_12000"), ("c5_6000", "c12_6000")):
        g = [x for x in dev if x["per"][b]["j"]["current"] and not x["per"][a]["j"]["current"]]
        l = [x for x in dev if x["per"][a]["j"]["current"] and not x["per"][b]["j"]["current"]]
        gx = [x for x in dev if x["per"][b]["j"]["exact"] and not x["per"][a]["j"]["exact"]]
        lx = [x for x in dev if x["per"][a]["j"]["exact"] and not x["per"][b]["j"]["exact"]]
        flips["%s->%s" % (a, b)] = {
            "current_gain": [x["qid"] for x in g], "current_loss": [x["qid"] for x in l],
            "exact_gain": [x["qid"] for x in gx], "exact_loss": [x["qid"] for x in lx],
        }
        log("\n   %s->%s  current +%d/-%d   exact +%d/-%d" % (a, b, len(g), len(l), len(gx), len(lx)))

    # ---- bootstrap
    def boot(sub, a, b, jn, iters=4000, seed=20261107):
        if not sub:
            return None
        flags = [(1 if x["per"][b]["j"][jn] else 0) - (1 if x["per"][a]["j"][jn] else 0) for x in sub]
        rnd = random.Random(seed)
        n = len(flags)
        means = []
        for _ in range(iters):
            s = 0
            for _ in range(n):
                s += flags[rnd.randrange(n)]
            means.append(100.0 * s / n)
        means.sort()
        obs = 100.0 * sum(flags) / n
        return {"delta_pt": round(obs, 2),
                "ci_lo": round(means[int(0.025 * iters)], 2),
                "ci_hi": round(means[int(0.975 * iters)], 2),
                "p_le_0": round(sum(1 for m in means if m <= 0) / iters, 4),
                "n": n, "gain": sum(1 for f in flags if f > 0), "loss": sum(1 for f in flags if f < 0)}

    log("\n== S5 paired bootstrap (4000 iters, seed 20261107, judge=current)")
    boots = {}
    for name, sub in (("dev", subset("dev")), ("holdout", subset("holdout")),
                      ("clean_holdout", subset("holdout", clean=True))):
        boots[name] = {}
        for a, b in (("c5_6000", "c12_12000"), ("c12_6000", "c12_12000"), ("c5_6000", "c5_12000"),
                     ("c5_6000", "c12_6000")):
            for jn in ("current", "exact"):
                bs = boot(sub, a, b, jn)
                if bs is None:
                    continue
                boots[name]["%s->%s|%s" % (a, b, jn)] = bs
                log("   %-15s %-26s %-8s d=%+6.2f CI[%+6.2f,%+6.2f] p=%.4f gain %d loss %d n=%d" % (
                    name, "%s->%s" % (a, b), jn, bs["delta_pt"], bs["ci_lo"], bs["ci_hi"],
                    bs["p_le_0"], bs["gain"], bs["loss"], bs["n"]))

    OUT["s3_s5"] = {"judge_ladder_dev": ladder, "per_category_dev": percat,
                    "matrices": matrices, "flips": flips, "bootstrap": boots,
                    "rows": [{k: v for k, v in x.items() if k != "per"} |
                             {"per": {t: {"units": d["units"], "chars": d["chars"],
                                          "abstained": d["abstained"], "text": d["text"],
                                          "j": d["j"]} for t, d in x["per"].items()}}
                             for x in rows]}
    return pipe, r, items


# ---------------------------------------------------------------- S6
def s6_latency(items, n_sample=60):
    from arkrag import pipeline as P
    log("\n== S6 end-to-end latency per budget (ask_once, full pipeline)")
    sample = [x for x in items if x["split"] == "dev"][:n_sample]
    pipe = P.build_pipeline(P.load_generation(GEN), top_k=20, graph_depth=1)
    out = {}
    for budget in (6000, 12000, 40000):
        pipe["retriever"].budget["max_context_tokens"] = budget
        ts, ctxc, units = [], [], []
        for it in sample:
            t0 = time.perf_counter()
            res = P.ask_once(pipe, it["question"])
            ts.append((time.perf_counter() - t0) * 1000.0)
            ctxc.append(sum(len(str(u.get("text") or "")) for u in res["context"]["units"]))
            units.append(len(res["context"]["units"]))
        ts_sorted = sorted(ts)

        def pctl(p):
            return ts_sorted[max(0, min(len(ts_sorted) - 1, int(round(p / 100.0 * (len(ts_sorted) - 1)))))]
        out[str(budget)] = {
            "n": len(ts), "p50_ms": round(pctl(50), 1), "p95_ms": round(pctl(95), 1),
            "mean_ms": round(sum(ts) / len(ts), 1),
            "ctx_chars_mean": round(sum(ctxc) / len(ctxc)),
            "units_mean": round(sum(units) / len(units), 2),
        }
        log("   @%-6d p50 %6.1fms p95 %6.1fms mean %6.1fms  ctx_chars %6d units %5.2f" % (
            budget, out[str(budget)]["p50_ms"], out[str(budget)]["p95_ms"],
            out[str(budget)]["mean_ms"], out[str(budget)]["ctx_chars_mean"],
            out[str(budget)]["units_mean"]))
    pipe["retriever"].budget["max_context_tokens"] = 6000
    OUT["s6_latency"] = out


# ---------------------------------------------------------------- S7
def s7_followups(v1, v2):
    """Two decisive follow-ups that the first pass left open.

    (a) adversarial: is the 0/19 a *structural* impossibility or just a wrong
        output format in my first template?  Try every phrasing the goldset
        itself accepts.
    (b) clean holdout: the point estimate in the conclusion is 72.50->80.83
        but my eq-judge run gives 70.83->75.83.  Test the hypothesis that the
        quoted numbers are the SUPERSET judge by recomputing the clean-holdout
        subset straight out of the stored diag files.
    """
    g = json.loads((ROOT / "data/arkrag/graph.json").read_text(encoding="utf-8"))
    name2type = {}
    for e in g["entities"]:
        name2type.setdefault(e["name"], e.get("type") or "")
    dev = [x for x in v2 if str(x.get("split")) == "dev" and x.get("gold_chunk_ids")]
    adv = [x for x in dev if x.get("category") == "adversarial"]

    TMPLS = [
        "\u4e0d\u662f\u540c\u4e00\u4e2a\u5b9e\u4f53\uff1a\u300c%s\u300d\u7684\u7c7b\u578b\u662f%s\uff0c\u300c%s\u300d\u7684\u7c7b\u578b\u662f%s\uff08\u636e\u56fe\u8c31\uff09",
        "\u4e0d\u662f\u540c\u4e00\u4e2a\u5b9e\u4f53\uff0c%s\u662f%s\uff0c%s\u662f%s",
        "\u4e0d\u540c\u5b9e\u4f53\uff0c\u7c7b\u578b\u5206\u522b\u662f%s\u548c%s",
    ]
    log("\n== S7a adversarial: can a template that reads graph entity types pass?")
    log("   %-16s %-14s %-14s %-8s %-8s %s" % ("qid", "A", "B", "typeA", "typeB", "tmpl_hits"))
    arows = []
    for x in adv:
        q = str(x["question"])
        qs = anchors(q)
        if len(qs) < 2:
            arows.append({"qid": x["question_id"], "skip": "anchors<2"})
            continue
        a, b = qs[0], qs[1]
        ta, tb = name2type.get(a, "?"), name2type.get(b, "?")
        gs = cands(x)
        hits = []
        for k, t in enumerate(TMPLS, 1):
            if k == 1:
                cand_text = t % (a, ta, b, tb)
            elif k == 2:
                cand_text = t % (a, ta, b, tb)
            else:
                cand_text = t % (ta, tb)
            if any(gg in cand_text for gg in gs):
                hits.append(k)
        arows.append({"qid": x["question_id"], "a": a, "b": b, "type_a": ta, "type_b": tb,
                      "templates_passing": hits, "gold": gs[0]})
        log("   %-16s %-14s %-14s %-8s %-8s %s" % (
            x["question_id"], a[:14], b[:14], ta, tb, hits))
    n_any = sum(1 for r in arows if r.get("templates_passing"))
    log("   adversarial answerable by a graph-type template: %d/%d" % (n_any, len(adv)))
    OUT["s7a_adversarial_template"] = {"n": len(adv), "n_answerable": n_any, "rows": arows}

    # (b) clean holdout under superset vs equal, from stored diag files
    log("\n== S7b clean holdout: which judge produced the quoted 72.50%%->80.83%%?")
    v1ids = {str(x["question_id"]) for x in v1}
    res = {}
    for fn in ("diag_crossmode_holdout.json", "diag_holdout_holdout.json"):
        d = json.loads((ROOT / "data" / fn).read_text(encoding="utf-8"))
        rows = [x for x in d["rows"] if x["qid"] not in v1ids
                and str(x.get("cat", "")) not in STRUCTURAL]
        per_keys = sorted({k for x in rows for k in x["per"]})
        res[fn] = {"n_clean_scorable": len(rows), "keys": per_keys, "cells": {}}
        for k in per_keys:
            for jn in ("sup", "eq", "ok"):
                vals = [x["per"][k] for x in rows if jn in (x["per"][k] or {})]
                if not vals:
                    continue
                p = sum(1 for v in vals if v.get(jn))
                res[fn]["cells"]["%s|%s" % (k, jn)] = {"pass": p, "n": len(vals),
                                                       "rate": round(100.0 * p / len(vals), 2)}
        log("   %s  n_clean_scorable=%d" % (fn, len(rows)))
        for kk, vv in sorted(res[fn]["cells"].items()):
            log("      %-18s %3d/%-3d %6.2f%%" % (kk, vv["pass"], vv["n"], vv["rate"]))
    OUT["s7b_clean_holdout_judge"] = res


# ---------------------------------------------------------------- S8
def s8_judge_slack(v2):
    """Quantify how much of the 'current' pass comes from the two loose rules.

    (a) single_chunk / alias: the rule is 'gold string occurs anywhere in the
        answer'.  If the gold is already a phrase in the QUESTION, echoing the
        question satisfies it.  Measure how often that is even possible.
    (b) multihop: the rule is 'one claim holds gold + >=1 anchor'.  Report the
        anchor-count distribution inside the clause that actually matched, so
        the reader can see whether '>=1' is doing all the work.
    (c) clean-holdout delta re-derived from the stored crossmode file under
        BOTH judges, with a paired bootstrap on each, to identify which judge
        each number in the claim came from.
    """
    OUT["s8_judge_slack"] = {}
    rows = OUT["s3_s5"]["rows"]
    dev = [x for x in rows if x["split"] == "dev"]
    # BUGFIX: s3_s5 rows store "gold" as a SINGLE string (cands(item)[0]).
    # Iterating a string yields characters, so `g in question for g in gold`
    # is almost always true and would fake "gold already in question" at ~100%.
    # Always go back to the real goldset item and use the full candidate list.
    _by_qid = {str(it["question_id"]): it for it in v2}
    for _x in rows:
        _it = _by_qid.get(str(_x["qid"]))
        _x["gold_list"] = cands(_it) if _it is not None else [str(_x["gold"])]
    log("\n== S8a 'gold string occurs in the answer' vs 'gold string was already in the question'")
    log("   %-16s %-6s %-9s %-12s %-14s" % ("cat", "n", "goldInQ", "pass_inQ", "pass_notInQ"))
    resa = {}
    ARM = "c12_12000"
    for cat in ("single_chunk", "alias", "direct_relation"):
        rr = [x for x in dev if x["cat"] == cat]
        inq = [x for x in rr if any(g and g in x["question"] for g in x["gold_list"])]
        notq = [x for x in rr if not any(g and g in x["question"] for g in x["gold_list"])]
        pin = sum(1 for x in inq if x["per"][ARM]["j"]["current"])
        pnot = sum(1 for x in notq if x["per"][ARM]["j"]["current"])
        resa[cat] = {"n": len(rr), "gold_in_question": len(inq),
                     "pass_among_inq": pin, "pass_among_notinq": pnot,
                     "pass_total": pin + pnot}
        log("   %-16s %-6d %-9d %-12s %-14s" % (
            cat, len(rr), len(inq),
            "%d/%d" % (pin, len(inq)) if inq else "-",
            "%d/%d" % (pnot, len(notq)) if notq else "-"))
    OUT["s8_judge_slack"]["gold_in_question"] = resa

    log("\n== S8b multihop: anchors matched inside the clause that made it pass")
    hist = {}
    detail = []
    mh = [x for x in dev if x["cat"] == "multihop"]
    for x in mh:
        cell = x["per"][ARM]
        if not cell["j"]["current"]:
            continue
        text = cell["text"] or ""
        gs, an = x["gold_list"], x["anchors"]
        best = None
        for seg in re.split(r"[\u3002\uff1b\n]", text):
            if any(g in seg for g in gs):
                k = sum(1 for a in an if a in seg)
                if best is None or k > best:
                    best = k
        k = best if best is not None else 0
        hist[k] = hist.get(k, 0) + 1
        if k <= 1:
            detail.append({"qid": x["qid"], "anchors_matched": k, "n_anchors": len(an),
                           "gold": gs, "anchors": an,
                           "clause": (text[:160] if text else "")})
    npass = sum(hist.values())
    log("   multihop passing at %s: %d of %d" % (ARM, npass, len(mh)))
    for k in sorted(hist):
        log("      anchors_matched=%d : %d (%.1f%%)" % (k, hist[k], 100.0 * hist[k] / max(npass, 1)))
    log("   examples where the winning clause held <=1 of the question anchors:")
    for dd in detail[:12]:
        log("      %-16s %d/%d anchors  gold=%s" % (
            dd["qid"], dd["anchors_matched"], dd["n_anchors"], "/".join(dd["gold"])[:30]))
    OUT["s8_judge_slack"]["multihop_anchor_hist"] = {
        "arm": ARM, "n_pass": npass, "n_cat": len(mh), "hist": hist,
        "single_anchor_examples": detail}

    # (c) clean holdout: sup vs eq, with bootstrap
    d = json.loads((ROOT / "data/diag_crossmode_holdout.json").read_text(encoding="utf-8"))
    v1 = json.loads((ROOT / "data/goldset/gold-v1.json").read_text(encoding="utf-8"))
    v1ids = {str(x["question_id"]) for x in v1["items"]}
    clean = [x for x in d["rows"] if x["qid"] not in v1ids
             and str(x.get("cat", "")) not in STRUCTURAL]
    log("\n== S8c clean holdout, same 120 questions, both judges, paired bootstrap")
    out = {}
    for jn in ("sup", "eq"):
        for a, b in (("c5_6000", "c12_12000"), ("c12_6000", "c12_12000")):
            va = [1 if x["per"][a][jn] else 0 for x in clean]
            vb = [1 if x["per"][b][jn] else 0 for x in clean]
            pa, pb = sum(va), sum(vb)
            dd, lo, hi, pv = boot(va, vb)
            out["%s_%s_%s" % (a, b, jn)] = {
                "n": len(clean), "from": pa, "to": pb,
                "rate_from": round(100.0 * pa / len(clean), 2),
                "rate_to": round(100.0 * pb / len(clean), 2),
                "delta_pt": round(dd, 2), "ci": [round(lo, 2), round(hi, 2)],
                "p": round(pv, 4)}
            log("   %-9s %-10s->%-10s %3d/%d %6.2f%% -> %3d/%d %6.2f%%  d=%+.2f CI[%+.2f,%+.2f] p=%.4f" % (
                jn, a, b, pa, len(clean), 100.0 * pa / len(clean),
                pb, len(clean), 100.0 * pb / len(clean), dd, lo, hi, pv))
    OUT["s8_judge_slack"]["clean_holdout_both_judges"] = out


# ---------------------------------------------------------------- S9
def s9_null_baseline(v1, v2):
    """How much of the scoreboard can be earned with ZERO retrieval?

    The 'single_chunk' and 'alias' rules both say 'the gold string occurs
    somewhere in the answer'.  But the gold answer for those categories is very
    often the very title / alias that the question already quotes inside
    「」.  So an answerer that ignores retrieval entirely and just echoes the
    quoted span may already pass.  Measure it, per category and per subset.
    That number is the floor that any budget comparison inherits.
    """
    log("\n== S9 null baseline: answer = echo of the question's quoted span(s), no retrieval")
    dev_items = [x for x in v2 if x.get("gold_chunk_ids")]
    id1 = {str(x["question_id"]) for x in v1}

    def variants(item):
        q = str(item.get("question") or "")
        quotes = [m.group(1).strip() for m in QUOTE_RE.finditer(q)]
        titles = [m.group(1).strip() for m in TITLE_RE.finditer(q)]
        out = {}
        out["echo_first_quote"] = quotes[0] if quotes else ""
        out["echo_all_quotes"] = SEP.join(quotes)
        # single_chunk asks 'which chunk does this line come from? give the title'
        # and the question already names the work inside <>.  An answerer that
        # simply copies that span needs no retrieval at all.
        out["echo_titles"] = SEP.join(titles)
        return out

    VARIANTS = ("echo_first_quote", "echo_all_quotes", "echo_titles")

    subsets = {
        "dev": [x for x in dev_items if str(x.get("split")) == "dev"
                and str(x.get("category")) not in STRUCTURAL],
        "holdout": [x for x in dev_items if str(x.get("split")) == "holdout"
                    and str(x.get("category")) not in STRUCTURAL],
        "clean_holdout": [x for x in dev_items if str(x.get("split")) == "holdout"
                          and str(x.get("category")) not in STRUCTURAL
                          and str(x["question_id"]) not in id1],
    }
    res = {"subsets": {}, "per_category_dev": {}}
    for vn in VARIANTS:
        log("   baseline variant: %s" % vn)
        for sname, sub in subsets.items():
            k = 0
            for x in sub:
                txt = variants(x)[vn]
                if judge_current(str(x.get("category")), txt, x, str(x.get("question") or "")):
                    k += 1
            r = {"pass": k, "n": len(sub), "rate": round(100.0 * k / len(sub), 2) if sub else 0.0}
            res["subsets"]["%s|%s" % (vn, sname)] = r
            log("      %-14s %3d/%-3d %6.2f%%" % (sname, k, len(sub), r["rate"]))
        for cat in ("single_chunk", "alias", "direct_relation", "multihop", "cross_chunk"):
            sub = [x for x in subsets["dev"] if str(x.get("category")) == cat]
            k = sum(1 for x in sub if judge_current(
                cat, variants(x)[vn], x, str(x.get("question") or "")))
            res["per_category_dev"]["%s|%s" % (vn, cat)] = {
                "pass": k, "n": len(sub), "rate": round(100.0 * k / len(sub), 2) if sub else 0.0}
            log("      dev %-16s %2d/%-2d %6.2f%%" % (cat, k, len(sub),
                                                      100.0 * k / len(sub) if sub else 0.0))

    log("\n   per-category: does the gold answer already sit in the question text?")
    contam = {}
    for cat in ("single_chunk", "alias", "direct_relation", "multihop",
                "cross_chunk", "time_conflict", "adversarial", "negative_abstain"):
        sub = [x for x in v2 if str(x.get("split")) == "dev" and x.get("gold_chunk_ids")
               and str(x.get("category")) == cat]
        n = sum(1 for x in sub if any(g and g in str(x.get("question") or "") for g in cands(x)))
        contam[cat] = {"gold_in_question": n, "n": len(sub)}
        log("      %-16s %2d/%-2d" % (cat, n, len(sub)))
    res["gold_in_question_dev"] = contam
    OUT["s9_null_baseline"] = res


def s10_sup_eq_gap(v1, v2):
    """How much does the cross_chunk superset rule inflate, per arm and subset?

    Conclusion C claims superset judging inflates results.  Quantify the exact
    gap between 'got >= want' and 'got == want' on the same answers, so the
    claim is either confirmed by a number or refuted by one.
    """
    log("\n== S10 cross_chunk superset-vs-equality gap (same answers, two rules)")
    id1 = {str(x["question_id"]) for x in v1}
    d = json.loads((ROOT / "data/diag_crossmode_dev.json").read_text(encoding="utf-8"))
    h = json.loads((ROOT / "data/diag_crossmode_holdout.json").read_text(encoding="utf-8"))
    arms = sorted({k for x in d["rows"] for k in x["per"]})
    out = {}
    for name, rows in (("dev", d["rows"]), ("holdout", h["rows"])):
        for a in arms:
            sup = sum(1 for x in rows if x["per"][a]["sup"])
            eq = sum(1 for x in rows if x["per"][a]["eq"])
            n = len(rows)
            out.setdefault(name, {"n": n})[a] = {
                "sup": sup, "eq": eq, "gap_questions": sup - eq,
                "gap_pt": round(100.0 * (sup - eq) / n, 2)}
            log("   %-8s %-10s sup %2d/%-2d (%5.1f%%)  eq %2d/%-2d (%5.1f%%)  inflate %+d q = %+.2f pt" % (
                name, a, sup, n, 100.0 * sup / n, eq, n, 100.0 * eq / n, sup - eq,
                100.0 * (sup - eq) / n))
    OUT["s10_sup_eq_gap"] = out


def s11_extra_chapters_true(v2):
    """The decisive test for the cross_chunk judge.

    When the answer lists MORE chapters than gold, is that 'hallucinated
    inflation' or 'gold is incomplete'?  Ask the graph: for each extra chapter
    in the answer, does the entity actually have a chunk in that chapter?
    If it does, the equality rule is punishing a correct answer.
    """
    log("\n== S11 are the 'extra' chapters real? (graph entity.chunks says yes/no)")
    g = json.loads((ROOT / "data/arkrag/graph.json").read_text(encoding="utf-8"))
    ents = {e["name"]: e for e in g["entities"]}
    ch2chap = {}
    with (ROOT / "data/arkrag/chunks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            ch2chap[c["chunk_id"]] = c.get("chapter")
    rows = [x for x in OUT["s3_s5"]["rows"] if x["cat"] == "cross_chunk"]
    ARM = "c12_40000"
    detail = []
    tot_extra = tot_real = tot_fake = 0
    for x in rows:
        item = next(i for i in v2 if str(i["question_id"]) == x["qid"])
        q = str(item.get("question") or "")
        qs = [m.group(1).strip() for m in QUOTE_RE.finditer(q)]
        name = qs[0] if qs else ""
        ent = ents.get(name)
        if not ent:
            continue
        real = {ch2chap.get(c) for c in (ent.get("chunks") or []) if ch2chap.get(c)}
        want = gold_chapter_set(item)
        got = got_chapter_set(x["per"][ARM]["text"] or "")
        extra = got - want
        if not extra:
            continue
        yes = sorted(e for e in extra if e in real)
        no = sorted(e for e in extra if e not in real)
        tot_extra += len(extra)
        tot_real += len(yes)
        tot_fake += len(no)
        detail.append({"qid": x["qid"], "entity": name, "gold": sorted(want),
                       "extra_count": len(extra), "extra_chapters_that_are_real": yes,
                       "extra_chapters_that_are_not_real": no})
    log("   questions listing extra chapters at %s: %d" % (ARM, len(detail)))
    log("   extra chapter mentions total %d : real %d (%.0f%%) : not-real %d (%.0f%%)" % (
        tot_extra, tot_real, 100.0 * tot_real / max(tot_extra, 1),
        tot_fake, 100.0 * tot_fake / max(tot_extra, 1)))
    log("   %-16s %-12s %-6s %-6s %s" % ("qid", "entity", "extra", "real", "real_examples"))
    for dd in detail:
        log("   %-16s %-12s %-6d %-6d %s" % (
            dd["qid"], dd["entity"][:12], dd["extra_count"],
            len(dd["extra_chapters_that_are_real"]),
            ",".join(dd["extra_chapters_that_are_real"][:3])))
    OUT["s11_extra_chapters"] = {
        "arm": ARM, "n_q": len(detail), "extra_total": tot_extra,
        "extra_real": tot_real, "extra_not_real": tot_fake, "detail": detail}


def s12_gold_recall(v2):
    """Is the cross_chunk gold actually complete?  (attack point 5, decisive)

    S11 showed the 'extra' chapters the answer lists really do contain the
    entity name.  That can still be a substring artefact (e.g. '邙山' inside
    '邙山镇'), so this pass applies a strict containment test: a mention counts
    only if the matched span is not part of a longer known graph name/alias.

    Then: how many cross_chunk questions have at least one chapter where the
    entity genuinely appears but gold omits it?  That is the population for
    which the equality rule makes a correct answer fail.
    """
    log("\n== S12 cross_chunk gold recall under a strict mention test")
    g = json.loads((ROOT / "data/arkrag/graph.json").read_text(encoding="utf-8"))
    names = set()
    for e in g["entities"]:
        names.add(e["name"])
        for a in (e.get("aliases") or []):
            if isinstance(a, str) and a:
                names.add(a)
    ents = {e["name"]: e for e in g["entities"]}
    ch2chap, chap_text = {}, defaultdict(list)
    with (ROOT / "data/arkrag/chunks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            ch2chap[c["chunk_id"]] = c.get("chapter")
            chap_text[c.get("chapter")].append(c.get("text") or "")

    def clean_mention(nm, t):
        i = 0
        while True:
            i = t.find(nm, i)
            if i < 0:
                return False
            ext = False
            for L in range(len(nm) + 1, len(nm) + 4):
                if t[i:i + L] in names:
                    ext = True
                    break
            for L in range(1, 4):
                if i - L >= 0 and t[i - L:i + len(nm)] in names:
                    ext = True
                    break
            if not ext:
                return True
            i += 1

    rows = []
    n_q = n_with_entity = n_gold_missing = 0
    tot_missing = 0
    for item in v2:
        if str(item.get("category")) != "cross_chunk":
            continue
        n_q += 1
        q = str(item.get("question") or "")
        qs = [m.group(1).strip() for m in QUOTE_RE.finditer(q)]
        if not qs:
            continue
        nm = qs[0]
        ent = ents.get(nm)
        if not ent:
            continue
        n_with_entity += 1
        want = gold_chapter_set(item)
        ent_ch = {ch2chap.get(c) for c in (ent.get("chunks") or []) if ch2chap.get(c)}
        txt_ch = {ch for ch, ts in chap_text.items()
                  if any(clean_mention(nm, t) for t in ts)}
        missing = sorted(txt_ch - want)
        chain = {"qid": str(item["question_id"]), "split": str(item.get("split")),
                 "entity": nm, "gold": sorted(want),
                 "gold_eq_entity_chunks": (want == ent_ch),
                 "text_chapters": len(txt_ch),
                 "gold_missing_from_text": missing}
        rows.append(chain)
        if missing:
            n_gold_missing += 1
            tot_missing += len(missing)
    log("   cross_chunk questions %d, with graph entity %d" % (n_q, n_with_entity))
    log("   questions where the entity genuinely appears in a chapter gold omits: %d/%d"
        % (n_gold_missing, n_with_entity))
    log("   total omitted chapter mentions %d" % tot_missing)
    n_circ = sum(1 for r in rows if r["gold_eq_entity_chunks"])
    log("   questions where gold chapters == graph entity chunk chapters: %d/%d"
        % (n_circ, len(rows)))
    log("   (if that ratio is near 1.0, gold was DERIVED from the graph, so agreeing")
    log("    with the graph is not independent evidence that gold is complete)")
    log("   %-16s %-8s %-12s %-6s %-6s %s" % ("qid", "split", "entity", "gold", "text", "omitted"))
    for r in rows:
        if r["gold_missing_from_text"]:
            log("   %-16s %-8s %-12s %-6d %-6d %s" % (
                r["qid"], r["split"], r["entity"][:12], len(r["gold"]),
                r["text_chapters"], ",".join(r["gold_missing_from_text"][:4])))
    OUT["s12_gold_recall"] = {
        "n_cross_chunk": n_q, "n_with_entity": n_with_entity,
        "n_gold_missing_a_real_chapter": n_gold_missing,
        "total_omitted_chapters": tot_missing,
        "n_gold_equals_graph_chapters": n_circ, "rows": rows}


def s13_placebo_judge(v2):
    """Placebo test: score each answer against a FOREIGN question's gold.

    If the per-category rules only test "the gold string occurs somewhere in
    the answer", then an answer to question i can accidentally satisfy the gold
    of question j, because the golds are ordinary phrases (entity names,
    relation words).  A sound judge should give a cross-question pass rate near
    zero.  The gap between own-pass and foreign-pass is the amount of the score
    that is earned by string coincidence rather than by answering.

    Also: pass rate as a function of answer length, to show whether the loose
    rules pay for verbosity alone (the mechanism behind the MAX_CLAIMS and
    budget gains).
    """
    log("\n== S13 placebo judge: score answers against a foreign question's gold")
    ARM = "c12_12000"
    rows = [x for x in OUT["s3_s5"]["rows"] if x["split"] == "dev"]
    by_qid = {str(x["question_id"]): x for x in v2}
    by_cat = defaultdict(list)
    for x in rows:
        if x["cat"] in STRUCTURAL:
            continue
        by_cat[x["cat"]].append(x)

    def item_of(row):
        """The real goldset item, so gold_chunk_ids/acceptable_answers are exact."""
        return by_qid[str(row["qid"])]

    res = {"arm": ARM, "per_cat": {}}
    log("   %-16s %-6s %-10s %-12s %-12s %s" % (
        "cat", "n", "own_pass", "foreign_avg", "foreign_max", "reading"))
    mism = []
    for cat in ("single_chunk", "alias", "direct_relation", "multihop", "cross_chunk"):
        rr = by_cat.get(cat) or []
        if len(rr) < 4:
            continue
        own = 0
        for x in rr:
            it = item_of(x)
            got = judge_current(cat, x["per"][ARM]["text"] or "", it, x["question"])
            own += 1 if got else 0
            # cross-check: the row's gold must reproduce the stored verdict
            if bool(got) != bool(x["per"][ARM]["j"]["current"]):
                mism.append(x["qid"])
        # rotate the golds: answer of i judged against gold of i+shift
        frates = []
        for shift in (1, 2, 3):
            k = 0
            tot = 0
            for i, x in enumerate(rr):
                other = rr[(i + shift) % len(rr)]
                oit = item_of(other)
                tot += 1
                if judge_current(cat, x["per"][ARM]["text"] or "", oit,
                                 str(oit.get("question") or "")):
                    k += 1
            frates.append(round(100.0 * k / max(tot, 1), 2))
        own_rate = round(100.0 * own / len(rr), 2)
        favg = round(sum(frates) / len(frates), 2)
        res["per_cat"][cat] = {"n": len(rr), "own_pass": own, "own_rate": own_rate,
                               "foreign_rates": frates, "foreign_avg": favg}
        log("   %-16s %-6d %3d %6.2f%%  %6.2f%%      %6.2f%%      %s" % (
            cat, len(rr), own, own_rate, favg, max(frates),
            "LEAKY" if favg > 5.0 else "ok"))
    res["shim_mismatches"] = mism
    log("   gold cross-check: %d rows where the goldset item did not reproduce the"
        " stored verdict" % len(mism))
    if mism:
        log("      " + " ".join(mism[:20]))

    log("\n   pass rate vs answer length (dev, arm %s)" % ARM)
    log("   %-16s %-10s %-10s %-10s" % ("cat", "short<=2", "mid 3-5", "long>5"))
    for cat in ("single_chunk", "alias", "direct_relation", "multihop", "cross_chunk"):
        rr = by_cat.get(cat) or []
        buckets = {"short": [], "mid": [], "long": []}
        for x in rr:
            cell = x["per"][ARM]
            ncl = cell.get("units") or 0
            segs = [s for s in re.split(SEP, cell["text"] or "") if s.strip()]
            k = len(segs)
            key = "short" if k <= 2 else ("mid" if k <= 5 else "long")
            buckets[key].append(1 if cell["j"]["current"] else 0)
        cells = []
        for key in ("short", "mid", "long"):
            b = buckets[key]
            cells.append("%s %d/%d" % (key, sum(b), len(b)) if b else "%s -" % key)
        res["per_cat"].setdefault(cat, {})["length_buckets"] = {
            k: {"pass": sum(v), "n": len(v)} for k, v in buckets.items()}
        log("   %-16s %-10s %-10s %-10s" % (cat, cells[0], cells[1], cells[2]))
    OUT["s13_placebo_judge"] = res


def s14_best_effort(v1, v2):
    """Does the budget gain survive the most defensible judge combination?

    The current judge is loose where it should be strict and strict where it
    should be loose:
      * multihop  -- '>=1 anchor' lets a one-hop claim answer a two-hop question
                     (S8b: 42/42 passing multihop claims matched ONE anchor;
                     with all anchors the category scores 0).
      * cross_chunk -- 'got == want' requires equality against a gold that S12
                     shows omits real chapters in 38/60 questions, so it
                     punishes correct answers.
    So the most defensible combination is: multihop -> ALL anchors,
    cross_chunk -> superset, everything else unchanged.  Recompute both
    contrasts under it, with paired bootstrap, on all three subsets.
    """
    log("\n== S14 best-effort judge (multihop=all anchors, cross_chunk=superset)")
    try:
        sup = json.loads((ROOT / "data/diag_crossmode_holdout.json").read_text(encoding="utf-8"))
        supd = json.loads((ROOT / "data/diag_crossmode_dev.json").read_text(encoding="utf-8"))
        sup_map = {}
        for d in (sup, supd):
            for x in d["rows"]:
                sup_map.setdefault(str(x["qid"]), {})
                for k, v in x["per"].items():
                    sup_map[str(x["qid"])][k] = bool((v or {}).get("sup"))
    except Exception as e:
        log("   crossmode diag unavailable: %s" % e)
        sup_map = {}

    def judge_eff(cat, cell_cur, x, arm):
        if cat == "multihop":
            return cell_cur["j"]["multihop_all_anchors"]
        if cat == "cross_chunk":
            v = sup_map.get(str(x["qid"]), {}).get(arm)
            return bool(v) if v is not None else cell_cur["j"]["current"]
        return cell_cur["j"]["current"]

    rows = [x for x in OUT["s3_s5"]["rows"] if x["cat"] not in STRUCTURAL]
    subsets = {
        "dev": [x for x in rows if x["split"] == "dev"],
        "holdout": [x for x in rows if x["split"] == "holdout"],
        "clean_holdout": [x for x in rows if x["split"] == "holdout" and not x["in_v1"]],
    }
    res = {}
    log("   %-14s %-20s %-10s %-10s %-8s %-22s %s" % (
        "subset", "contrast", "from", "to", "delta", "ci95", "p"))
    for sname, sub in subsets.items():
        for jn in ("current", "best_effort"):
            for a, b in (("c5_6000", "c12_12000"), ("c12_6000", "c12_12000")):
                va, vb = [], []
                for x in sub:
                    if jn == "current":
                        va.append(1 if x["per"][a]["j"]["current"] else 0)
                        vb.append(1 if x["per"][b]["j"]["current"] else 0)
                    else:
                        va.append(1 if judge_eff(x["cat"], x["per"][a], x, a) else 0)
                        vb.append(1 if judge_eff(x["cat"], x["per"][b], x, b) else 0)
                pa, pb = sum(va), sum(vb)
                dd, lo, hi, pv = boot(va, vb)
                key = "%s|%s|%s_%s" % (sname, jn, a, b)
                res[key] = {"n": len(sub), "from": pa, "to": pb,
                            "rate_from": round(100.0 * pa / len(sub), 2),
                            "rate_to": round(100.0 * pb / len(sub), 2),
                            "delta_pt": round(dd, 2),
                            "ci": [round(lo, 2), round(hi, 2)], "p": round(pv, 4),
                            "gain": sum(1 for i in range(len(sub)) if va[i] == 0 and vb[i] == 1),
                            "loss": sum(1 for i in range(len(sub)) if va[i] == 1 and vb[i] == 0)}
                log("   %-14s %-20s %3d/%-4d %3d/%-4d %+7.2f [%+6.2f,%+6.2f] p=%.4f  +%d/-%d" % (
                    sname, "%s %s->%s" % (jn[:9], a, b), pa, len(sub), pb, len(sub),
                    dd, lo, hi, pv, res[key]["gain"], res[key]["loss"]))
    OUT["s14_best_effort"] = res

    # where does the budget gain come from, per category?
    log("\n   category attribution of the c12_6000 -> c12_12000 gain (current judge)")
    attr = {}
    for sname, sub in subsets.items():
        for a, b in (("c12_6000", "c12_12000"),):
            per = {}
            for cat in ("single_chunk", "alias", "direct_relation", "multihop", "cross_chunk"):
                rr = [x for x in sub if x["cat"] == cat]
                if not rr:
                    continue
                g = sum(1 for x in rr if not x["per"][a]["j"]["current"] and x["per"][b]["j"]["current"])
                l = sum(1 for x in rr if x["per"][a]["j"]["current"] and not x["per"][b]["j"]["current"])
                per[cat] = {"n": len(rr), "gain": g, "loss": l, "net": g - l}
            attr[sname] = per
            log("      %-14s %s" % (sname, " ".join(
                "%s:%+d" % (c, v["net"]) for c, v in per.items())))
    OUT["s14_best_effort"]["attribution"] = attr


def s15_denominator_and_lookup(v1, v2):
    """Two last attacks.

    (a) Denominator: D removes 65 questions.  That cannot change a PAIRED delta
        (both arms score 0 on them), but it does inflate the ABSOLUTE rate that
        A reports as the production number.  Compute both denominators.
    (b) Zero-retrieval lookup baseline: answer by picking the graph entity name
        that literally occurs in the question, and (for cross_chunk) the graph
        chapters of that entity.  No search, no context, no units.  If that
        scores well, the judge is rewarding lookup rather than retrieval.
    """
    log("\n== S15a denominator effect: what do the structural 65 do to the LEVEL?")
    rows = OUT["s3_s5"]["rows"]
    out = {"denominator": {}}
    for sname, pred in (("dev", lambda x: x["split"] == "dev"),
                        ("holdout", lambda x: x["split"] == "holdout"),
                        ("clean_holdout", lambda x: x["split"] == "holdout" and not x["in_v1"])):
        sub = [x for x in rows if pred(x)]
        for tag in ("c5_6000", "c12_6000", "c12_12000"):
            n_all = len(sub)
            k_all = sum(1 for x in sub if x["per"][tag]["j"]["current"] is True)
            sc = [x for x in sub if x["cat"] not in STRUCTURAL]
            k_sc = sum(1 for x in sc if x["per"][tag]["j"]["current"] is True)
            out["denominator"]["%s|%s" % (sname, tag)] = {
                "n_all": n_all, "pass_all": k_all,
                "rate_all": round(100.0 * k_all / n_all, 2),
                "n_scorable": len(sc), "pass_scorable": k_sc,
                "rate_scorable": round(100.0 * k_sc / len(sc), 2)}
            log("   %-14s %-10s all %3d/%-3d %6.2f%%   scorable %3d/%-3d %6.2f%%" % (
                sname, tag, k_all, n_all, 100.0 * k_all / n_all,
                k_sc, len(sc), 100.0 * k_sc / len(sc)))

    log("\n== S15b zero-retrieval lookup baseline (graph name found in the question)")
    g = json.loads((ROOT / "data/arkrag/graph.json").read_text(encoding="utf-8"))
    ents = {e["name"]: e for e in g["entities"]}
    ch2chap = {}
    with (ROOT / "data/arkrag/chunks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            ch2chap[c["chunk_id"]] = c.get("chapter")
    id1 = {str(x["question_id"]) for x in v1}

    def lookup_answer(item):
        q = str(item.get("question") or "")
        hits = []
        for nm in ents:
            if nm and nm in q:
                hits.append((len(nm), nm))
        hits.sort(reverse=True)
        names = [nm for _, nm in hits[:2]]
        cat = str(item.get("category") or "")
        if cat == "cross_chunk" and names:
            e = ents[names[0]]
            chs = sorted({ch2chap.get(c) for c in (e.get("chunks") or []) if ch2chap.get(c)})
            return "\u3001".join(chs)
        return SEP.join(names)

    res = {"subsets": {}, "per_cat": {}}
    subsets = {
        "dev": [x for x in v2 if x.get("gold_chunk_ids") and x.get("split") == "dev"
                and str(x.get("category")) not in STRUCTURAL],
        "holdout": [x for x in v2 if x.get("gold_chunk_ids") and x.get("split") == "holdout"
                    and str(x.get("category")) not in STRUCTURAL],
        "clean_holdout": [x for x in v2 if x.get("gold_chunk_ids") and x.get("split") == "holdout"
                          and str(x.get("category")) not in STRUCTURAL
                          and str(x["question_id"]) not in id1],
    }
    for sname, sub in subsets.items():
        k = sum(1 for x in sub if judge_current(str(x.get("category")), lookup_answer(x), x,
                                               str(x.get("question") or "")))
        res["subsets"][sname] = {"pass": k, "n": len(sub), "rate": round(100.0 * k / len(sub), 2)}
        log("   %-14s %3d/%-3d %6.2f%%" % (sname, k, len(sub), 100.0 * k / len(sub)))
    for cat in ("single_chunk", "alias", "direct_relation", "multihop", "cross_chunk"):
        sub = [x for x in subsets["dev"] if str(x.get("category")) == cat]
        k = sum(1 for x in sub if judge_current(cat, lookup_answer(x), x,
                                               str(x.get("question") or "")))
        res["per_cat"][cat] = {"pass": k, "n": len(sub),
                               "rate": round(100.0 * k / len(sub), 2) if sub else 0.0}
        log("   dev %-16s %2d/%-2d %6.2f%%" % (cat, k, len(sub), 100.0 * k / len(sub) if sub else 0.0))
    OUT["s15_denominator_and_lookup"] = out
    OUT["s15_denominator_and_lookup"]["lookup_baseline"] = res


def s16_eq_false_negatives(v1, v2):
    """How often does the equality rule kill an answer that is actually BETTER?

    S10 counted cells where sup=True but eq=False (the superset rule passes and
    equality fails).  S12 showed gold omits real chapters.  This section joins
    the two: of those sup-yes/eq-no cells, in how many does at least one of the
    'extra' chapters genuinely mention the entity in its actual text (strict
    containment test, so '邙山' inside '邙山镇' does NOT count)?

    Those cells are outright judge false negatives: the answer listed a chapter
    where the entity really appears, and the equality rule marked it wrong.
    """
    log("\n== S16 equality-rule false negatives (sup yes, eq no, extra chapter real)")
    g = json.loads((ROOT / "data/arkrag/graph.json").read_text(encoding="utf-8"))
    names = set()
    for e in g["entities"]:
        names.add(e["name"])
        for a in (e.get("aliases") or []):
            if isinstance(a, str) and a:
                names.add(a)
    chap_text = defaultdict(list)
    for line in (ROOT / "data/arkrag/chunks.jsonl").open(encoding="utf-8"):
        c = json.loads(line)
        chap_text[c.get("chapter")].append(c.get("text") or "")

    def clean_mention(nm, t):
        i = 0
        while True:
            i = t.find(nm, i)
            if i < 0:
                return False
            ext = False
            for L in range(len(nm) + 1, len(nm) + 4):
                if t[i:i + L] in names:
                    ext = True
                    break
            for L in range(1, 4):
                if i - L >= 0 and t[i - L:i + len(nm)] in names:
                    ext = True
                    break
            if not ext:
                return True
            i += 1

    mc = {}

    def chap_has(nm, ch):
        k = (nm, ch)
        if k not in mc:
            mc[k] = any(clean_mention(nm, t) for t in chap_text.get(ch, []))
        return mc[k]

    per = {}
    for f in ("diag_crossmode_dev.json", "diag_crossmode_holdout.json"):
        fp = ROOT / "data" / f
        if not fp.exists():
            continue
        for r in json.loads(fp.read_text(encoding="utf-8"))["rows"]:
            per[str(r["qid"])] = r["per"]
    byid = {str(x["question_id"]): x for x in v2}
    id1 = {str(x["question_id"]) for x in v1}
    textrows = {x["qid"]: x for x in OUT["s3_s5"]["rows"]}
    ARMS4 = ("c5_6000", "c12_6000", "c12_12000", "c12_40000")
    agg = {"cells": 0, "cells_real_extra": 0, "cells_all_extras_real": 0}
    qids_all, qids_arm = set(), defaultdict(set)
    detail = []
    for qid, p in per.items():
        item = byid.get(qid)
        if not item:
            continue
        qs = [m.group(1).strip() for m in QUOTE_RE.finditer(str(item.get("question") or ""))]
        if not qs:
            continue
        nm = qs[0]
        want = gold_chapter_set(item)
        for arm in ARMS4:
            cell = p.get(arm)
            if not cell or not cell.get("sup") or cell.get("eq"):
                continue
            tr = textrows.get(qid)
            if not tr:
                continue
            got = got_chapter_set(tr["per"][arm]["text"] or "")
            extra = sorted(got - want)
            real = [c for c in extra if chap_has(nm, c)]
            agg["cells"] += 1
            if real:
                agg["cells_real_extra"] += 1
                qids_all.add(qid)
                qids_arm[arm].add(qid)
                if len(real) == len(extra):
                    agg["cells_all_extras_real"] += 1
                detail.append({"qid": qid, "arm": arm, "entity": nm,
                               "split": str(item.get("split")),
                               "in_v1": qid in id1,
                               "extra": extra, "extra_that_really_mention": real})
    agg["distinct_questions"] = len(qids_all)
    agg["distinct_questions_per_arm"] = {a: len(qids_arm[a]) for a in ARMS4}
    agg["detail"] = detail
    log("   sup=yes/eq=no cells                        : %d" % agg["cells"])
    log("   ... in which an extra chapter really mentions the entity: %d (%.1f%%)"
        % (agg["cells_real_extra"], 100.0 * agg["cells_real_extra"] / max(agg["cells"], 1)))
    log("   ... in which EVERY extra chapter really mentions it     : %d"
        % agg["cells_all_extras_real"])
    log("   distinct questions affected                : %d" % agg["distinct_questions"])
    log("   per arm: %s" % agg["distinct_questions_per_arm"])
    log("   %-16s %-8s %-12s %-6s %s" % ("qid", "arm", "entity", "extra", "really-mentions"))
    for dd in detail:
        log("   %-16s %-8s %-12s %-6d %d/%d %s" % (
            dd["qid"], dd["arm"], dd["entity"][:12], len(dd["extra"]),
            len(dd["extra_that_really_mention"]), len(dd["extra"]),
            ",".join(dd["extra_that_really_mention"][:4])))
    OUT["s16_eq_false_negatives"] = agg


def s17_clean_cross_judge(v1, v2):
    """The only cross_chunk judge that is neither loose nor unfair.

    equality (eq)   -> punishes correct answers that list a real extra chapter
                       (S16: 18 questions at c12_12000 are killed this way)
    superset (sup)  -> rewards hallucinated chapters for free (S10: grows with
                       budget, +0.86pt -> +5.43pt)

    So judge cross_chunk as:  want subset of got  AND  every extra chapter
    genuinely mentions the entity (strict containment test).  A hallucinated
    chapter is still a failure; a genuinely-real omitted-by-gold chapter is not.
    Recompute the budget contrast under it, plus the same for every other arm.
    """
    from collections import defaultdict as _dd
    log("\n== S17 cross_chunk judge = superset AND no false-positive chapter")
    g = json.loads((ROOT / "data/arkrag/graph.json").read_text(encoding="utf-8"))
    names = set()
    for e in g["entities"]:
        names.add(e["name"])
        for a in (e.get("aliases") or []):
            if isinstance(a, str) and a:
                names.add(a)
    chap_text = _dd(list)
    for line in (ROOT / "data/arkrag/chunks.jsonl").open(encoding="utf-8"):
        c = json.loads(line)
        chap_text[c.get("chapter")].append(c.get("text") or "")

    def clean_mention(nm, t):
        i = 0
        while True:
            i = t.find(nm, i)
            if i < 0:
                return False
            ext = False
            for L in range(len(nm) + 1, len(nm) + 4):
                if t[i:i + L] in names:
                    ext = True
                    break
            for L in range(1, 4):
                if i - L >= 0 and t[i - L:i + len(nm)] in names:
                    ext = True
                    break
            if not ext:
                return True
            i += 1

    mc = {}

    def chap_has(nm, ch):
        k = (nm, ch)
        if k not in mc:
            mc[k] = any(clean_mention(nm, t) for t in chap_text.get(ch, []))
        return mc[k]

    byid = {str(x["question_id"]): x for x in v2}
    id1 = {str(x["question_id"]) for x in v1}
    ARMS4 = ("c5_6000", "c12_6000", "c12_12000", "c12_40000")
    verdict = {}
    stats = {"cells": 0, "extra_total": 0, "extra_real": 0, "extra_fake": 0}
    for qid, item in byid.items():
        if str(item.get("category")) != "cross_chunk":
            continue
        tr = next((x for x in OUT["s3_s5"]["rows"] if x["qid"] == qid), None)
        if not tr:
            continue
        qs = [m.group(1).strip() for m in QUOTE_RE.finditer(str(item.get("question") or ""))]
        nm = qs[0] if qs else ""
        want = gold_chapter_set(item)
        for arm in ARMS4:
            got = got_chapter_set(tr["per"][arm]["text"] or "")
            extra = got - want
            real = {c for c in extra if nm and chap_has(nm, c)}
            fake = extra - real
            stats["cells"] += 1
            stats["extra_total"] += len(extra)
            stats["extra_real"] += len(real)
            stats["extra_fake"] += len(fake)
            verdict[(qid, arm)] = bool(want) and bool(want <= got) and not fake
    log("   extra chapter mentions scanned %d : genuinely real %d : false positive %d"
        % (stats["extra_total"], stats["extra_real"], stats["extra_fake"]))
    res = {}
    rows = [x for x in OUT["s3_s5"]["rows"] if x["cat"] not in STRUCTURAL]
    subsets = {
        "dev": [x for x in rows if x["split"] == "dev"],
        "holdout": [x for x in rows if x["split"] == "holdout"],
        "clean_holdout": [x for x in rows if x["split"] == "holdout" and not x["in_v1"]],
    }
    log("   %-14s %-24s %-10s %-10s %-8s %-22s %s" % (
        "subset", "contrast", "from", "to", "delta", "ci95", "p"))
    for sname, sub in subsets.items():
        for a, b in (("c5_6000", "c12_12000"), ("c12_6000", "c12_12000")):
            va, vb = [], []
            for x in sub:
                if x["cat"] == "cross_chunk":
                    va.append(1 if verdict.get((x["qid"], a)) else 0)
                    vb.append(1 if verdict.get((x["qid"], b)) else 0)
                else:
                    va.append(1 if x["per"][a]["j"]["current"] else 0)
                    vb.append(1 if x["per"][b]["j"]["current"] else 0)
            pa, pb = sum(va), sum(vb)
            dd, lo, hi, pv = boot(va, vb)
            key = "%s|clean_cx|%s_%s" % (sname, a, b)
            res[key] = {"n": len(sub), "from": pa, "to": pb,
                        "rate_from": round(100.0 * pa / len(sub), 2),
                        "rate_to": round(100.0 * pb / len(sub), 2),
                        "delta_pt": round(dd, 2), "ci": [round(lo, 2), round(hi, 2)],
                        "p": round(pv, 4),
                        "gain": sum(1 for i in range(len(sub)) if va[i] == 0 and vb[i] == 1),
                        "loss": sum(1 for i in range(len(sub)) if va[i] == 1 and vb[i] == 0)}
            log("   %-14s %-24s %3d/%-4d %3d/%-4d %+7.2f [%+6.2f,%+6.2f] p=%.4f  +%d/-%d" % (
                sname, "%s -> %s" % (a, b), pa, len(sub), pb, len(sub), dd, lo, hi, pv,
                res[key]["gain"], res[key]["loss"]))
    # cross_chunk accuracy per arm under the three rules
    cc = [x for x in rows if x["cat"] == "cross_chunk"]
    per_arm = {}
    for arm in ARMS4:
        n = len(cc)
        eq = sum(1 for x in cc if x["per"][arm]["j"]["current"])
        clean = sum(1 for x in cc if verdict.get((x["qid"], arm)))
        per_arm[arm] = {"n": n, "eq": eq, "eq_rate": round(100.0 * eq / n, 2),
                        "clean": clean, "clean_rate": round(100.0 * clean / n, 2)}
    res["cross_chunk_per_arm"] = per_arm
    res["stats"] = stats
    log("   cross_chunk only, per arm: %s" % json.dumps(per_arm, ensure_ascii=False))
    OUT["s17_clean_cross_judge"] = res


def s18_defensible_judge(v1, v2):
    """The combined judge I would actually defend, and the budget contrast under it.

    multihop     : a passing answer must mention gold together with EVERY anchor
                   (the two hops may span different claims).  '>=1 anchor' (the
                   current rule) is passed by a one-hop claim, which answers half
                   the question; 'all anchors in ONE claim' is unrealistically
                   narrow, because a two-hop answer is naturally two claims.
    cross_chunk  : S17 rule -- superset of gold, and no extra chapter that does
                   not genuinely mention the entity.
    others       : unchanged (S13 says they are neither random nor obviously wrong).
    """
    from collections import defaultdict as _dd
    log("\n== S18 defensible combined judge")
    g = json.loads((ROOT / "data/arkrag/graph.json").read_text(encoding="utf-8"))
    names = set()
    for e in g["entities"]:
        names.add(e["name"])
        for a in (e.get("aliases") or []):
            if isinstance(a, str) and a:
                names.add(a)
    chap_text = _dd(list)
    for line in (ROOT / "data/arkrag/chunks.jsonl").open(encoding="utf-8"):
        c = json.loads(line)
        chap_text[c.get("chapter")].append(c.get("text") or "")

    def clean_mention(nm, t):
        i = 0
        while True:
            i = t.find(nm, i)
            if i < 0:
                return False
            ext = False
            for L in range(len(nm) + 1, len(nm) + 4):
                if t[i:i + L] in names:
                    ext = True
                    break
            for L in range(1, 4):
                if i - L >= 0 and t[i - L:i + len(nm)] in names:
                    ext = True
                    break
            if not ext:
                return True
            i += 1

    mc = {}

    def chap_has(nm, ch):
        k = (nm, ch)
        if k not in mc:
            mc[k] = any(clean_mention(nm, t) for t in chap_text.get(ch, []))
        return mc[k]

    ARMS4 = ("c5_6000", "c12_6000", "c12_12000", "c12_40000")
    byid = {str(x["question_id"]): x for x in v2}
    verdict = {}
    for x in OUT["s3_s5"]["rows"]:
        if x["cat"] in STRUCTURAL:
            continue
        item = byid.get(x["qid"])
        if not item:
            continue
        q = str(item.get("question") or "")
        gs = cands(item)
        an = anchors(q)
        for arm in ARMS4:
            text = x["per"][arm]["text"] or ""
            if x["cat"] == "multihop":
                ok = False
                if not any(gg in an for gg in gs):
                    for a_ in an:
                        if not any(any(gg in s for gg in gs) and a_ in s for s in claims(text)):
                            break
                    else:
                        ok = bool(an)
                verdict[(x["qid"], arm)] = ok
            elif x["cat"] == "cross_chunk":
                want = gold_chapter_set(item)
                got = got_chapter_set(text)
                qs = [m.group(1).strip() for m in QUOTE_RE.finditer(q)]
                nm = qs[0] if qs else ""
                fake = {c for c in (got - want) if not (nm and chap_has(nm, c))}
                verdict[(x["qid"], arm)] = bool(want) and bool(want <= got) and not fake
            else:
                verdict[(x["qid"], arm)] = bool(x["per"][arm]["j"]["current"])
    rows = [x for x in OUT["s3_s5"]["rows"] if x["cat"] not in STRUCTURAL]
    subsets = {
        "dev": [x for x in rows if x["split"] == "dev"],
        "holdout": [x for x in rows if x["split"] == "holdout"],
        "clean_holdout": [x for x in rows if x["split"] == "holdout" and not x["in_v1"]],
    }
    res = {"contrasts": {}, "per_arm_accuracy": {}, "per_category": {}}
    log("   %-14s %-22s %-10s %-10s %-8s %-22s %s" % (
        "subset", "contrast", "from", "to", "delta", "ci95", "p"))
    for sname, sub in subsets.items():
        for a, b in (("c5_6000", "c12_12000"), ("c12_6000", "c12_12000")):
            va = [1 if verdict.get((x["qid"], a)) else 0 for x in sub]
            vb = [1 if verdict.get((x["qid"], b)) else 0 for x in sub]
            pa, pb = sum(va), sum(vb)
            dd, lo, hi, pv = boot(va, vb)
            key = "%s|%s_%s" % (sname, a, b)
            res["contrasts"][key] = {
                "n": len(sub), "from": pa, "to": pb,
                "rate_from": round(100.0 * pa / len(sub), 2),
                "rate_to": round(100.0 * pb / len(sub), 2),
                "delta_pt": round(dd, 2), "ci": [round(lo, 2), round(hi, 2)],
                "p": round(pv, 4),
                "gain": sum(1 for i in range(len(sub)) if va[i] == 0 and vb[i] == 1),
                "loss": sum(1 for i in range(len(sub)) if va[i] == 1 and vb[i] == 0),
                "gain_qids": [sub[i]["qid"] for i in range(len(sub)) if va[i] == 0 and vb[i] == 1],
                "loss_qids": [sub[i]["qid"] for i in range(len(sub)) if va[i] == 1 and vb[i] == 0]}
            log("   %-14s %-22s %3d/%-4d %3d/%-4d %+7.2f [%+6.2f,%+6.2f] p=%.4f  +%d/-%d" % (
                sname, "%s -> %s" % (a, b), pa, len(sub), pb, len(sub), dd, lo, hi, pv,
                res["contrasts"][key]["gain"], res["contrasts"][key]["loss"]))
    for sname, sub in subsets.items():
        pa = {}
        for arm in ARMS4:
            p_ = sum(1 for x in sub if verdict.get((x["qid"], arm)))
            pa[arm] = {"pass": p_, "n": len(sub), "rate": round(100.0 * p_ / len(sub), 2)}
        res["per_arm_accuracy"][sname] = pa
        log("   %-14s %s" % (sname, " ".join("%s:%d/%d" % (k, v["pass"], v["n"]) for k, v in pa.items())))
    cats = ("single_chunk", "alias", "direct_relation", "multihop", "cross_chunk")
    for sname, sub in subsets.items():
        pc = {}
        for cat in cats:
            rr = [x for x in sub if x["cat"] == cat]
            if not rr:
                continue
            pc[cat] = {arm: sum(1 for x in rr if verdict.get((x["qid"], arm))) for arm in ARMS4}
            pc[cat]["n"] = len(rr)
        res["per_category"][sname] = pc
        log("   %-14s %s" % (sname, " ".join("%s=%s" % (c, pc[c]) for c in pc)))
    OUT["s18_defensible_judge"] = res


def s19_b_and_c(v1, v2):
    """Conclusion B (MAX_CLAIMS 5->12) and conclusion C, both under judge choice.

    B: the claim is 'rejected because holdout is only +3'.  Below the same
    contrast is recomputed under the current judge and under S18's defensible
    judge, so the verdict is not an artefact of the judge.
    C: 'superset inflates, equality is right, extra chapters grow with budget'.
    S17 already showed equality is the wrong rule; here the per-budget growth of
    extra / false-positive chapters is tabulated under the S17 accounting.
    """
    from collections import defaultdict as _dd
    log("\n== S19 conclusion B and C under judge choice")
    g = json.loads((ROOT / "data/arkrag/graph.json").read_text(encoding="utf-8"))
    names = set()
    for e in g["entities"]:
        names.add(e["name"])
        for a in (e.get("aliases") or []):
            if isinstance(a, str) and a:
                names.add(a)
    chap_text = _dd(list)
    for line in (ROOT / "data/arkrag/chunks.jsonl").open(encoding="utf-8"):
        c = json.loads(line)
        chap_text[c.get("chapter")].append(c.get("text") or "")

    def clean_mention(nm, t):
        i = 0
        while True:
            i = t.find(nm, i)
            if i < 0:
                return False
            ext = False
            for L in range(len(nm) + 1, len(nm) + 4):
                if t[i:i + L] in names:
                    ext = True
                    break
            for L in range(1, 4):
                if i - L >= 0 and t[i - L:i + len(nm)] in names:
                    ext = True
                    break
            if not ext:
                return True
            i += 1

    mc = {}

    def chap_has(nm, ch):
        k = (nm, ch)
        if k not in mc:
            mc[k] = any(clean_mention(nm, t) for t in chap_text.get(ch, []))
        return mc[k]

    ARMS4 = ("c5_6000", "c12_6000", "c12_12000", "c12_40000")
    byid = {str(x["question_id"]): x for x in v2}
    eff = {}
    for x in OUT["s3_s5"]["rows"]:
        item = byid.get(x["qid"])
        if not item:
            continue
        q = str(item.get("question") or "")
        gs = cands(item)
        an = anchors(q)
        for arm in ARMS4:
            text = x["per"][arm]["text"] or ""
            if x["cat"] in STRUCTURAL:
                eff[(x["qid"], arm)] = False
            elif x["cat"] == "multihop":
                ok = False
                if an and not any(gg in an for gg in gs):
                    ok = all(any(any(gg in s for gg in gs) and a_ in s for s in claims(text)) for a_ in an)
                eff[(x["qid"], arm)] = ok
            elif x["cat"] == "cross_chunk":
                want = gold_chapter_set(item)
                got = got_chapter_set(text)
                qs = [m.group(1).strip() for m in QUOTE_RE.finditer(q)]
                nm = qs[0] if qs else ""
                fake = {c for c in (got - want) if not (nm and chap_has(nm, c))}
                eff[(x["qid"], arm)] = bool(want) and bool(want <= got) and not fake
            else:
                eff[(x["qid"], arm)] = bool(x["per"][arm]["j"]["current"])

    rows = [x for x in OUT["s3_s5"]["rows"] if x["cat"] not in STRUCTURAL]
    subsets = {
        "dev": [x for x in rows if x["split"] == "dev"],
        "holdout": [x for x in rows if x["split"] == "holdout"],
        "clean_holdout": [x for x in rows if x["split"] == "holdout" and not x["in_v1"]],
    }
    res = {"B": {}, "C": {}, "A_absolutes": {}}
    log("   == B: MAX_CLAIMS 5 -> 12, same budget 6000, both judges")
    log("   %-14s %-10s %-10s %-10s %-8s %-22s %s" % (
        "subset", "judge", "from", "to", "delta", "ci95", "p"))
    for sname, sub in subsets.items():
        for jn in ("current", "defensible"):
            va, vb = [], []
            for x in sub:
                if jn == "current":
                    va.append(1 if x["per"]["c5_6000"]["j"]["current"] else 0)
                    vb.append(1 if x["per"]["c12_6000"]["j"]["current"] else 0)
                else:
                    va.append(1 if eff[(x["qid"], "c5_6000")] else 0)
                    vb.append(1 if eff[(x["qid"], "c12_6000")] else 0)
            pa, pb = sum(va), sum(vb)
            dd, lo, hi, pv = boot(va, vb)
            key = "%s|%s" % (sname, jn)
            res["B"][key] = {"n": len(sub), "from": pa, "to": pb,
                             "rate_from": round(100.0 * pa / len(sub), 2),
                             "rate_to": round(100.0 * pb / len(sub), 2),
                             "delta_pt": round(dd, 2), "ci": [round(lo, 2), round(hi, 2)],
                             "p": round(pv, 4),
                             "gain": sum(1 for i in range(len(sub)) if va[i] == 0 and vb[i] == 1),
                             "loss": sum(1 for i in range(len(sub)) if va[i] == 1 and vb[i] == 0),
                             "gain_qids": [sub[i]["qid"] for i in range(len(sub)) if va[i] == 0 and vb[i] == 1],
                             "loss_qids": [sub[i]["qid"] for i in range(len(sub)) if va[i] == 1 and vb[i] == 0]}
            log("   %-14s %-10s %3d/%-4d %3d/%-4d %+7.2f [%+6.2f,%+6.2f] p=%.4f  +%d/-%d" % (
                sname, jn, pa, len(sub), pb, len(sub), dd, lo, hi, pv,
                res["B"][key]["gain"], res["B"][key]["loss"]))
    log("\n   == C: cross_chunk accounting per budget (S17 rule)")
    cc = [x for x in rows if x["cat"] == "cross_chunk"]
    log("   %-10s %-8s %-8s %-8s %-8s %-10s %-10s" % (
        "arm", "n", "eq", "sup", "clean", "extraReal", "extraFake"))
    for arm in ARMS4:
        eq = sup = cl = 0
        er = ef = 0
        for x in cc:
            item = byid[x["qid"]]
            want = gold_chapter_set(item)
            got = got_chapter_set(x["per"][arm]["text"] or "")
            qs = [m.group(1).strip() for m in QUOTE_RE.finditer(str(item.get("question") or ""))]
            nm = qs[0] if qs else ""
            extra = got - want
            real = {c for c in extra if nm and chap_has(nm, c)}
            fake = extra - real
            er += len(real)
            ef += len(fake)
            if x["per"][arm]["j"]["current"]:
                eq += 1
            if want and want <= got:
                sup += 1
            if want and want <= got and not fake:
                cl += 1
        res["C"][arm] = {"n": len(cc), "eq": eq, "sup": sup, "clean": cl,
                         "extra_real": er, "extra_fake": ef,
                         "eq_rate": round(100.0 * eq / len(cc), 2),
                         "sup_rate": round(100.0 * sup / len(cc), 2),
                         "clean_rate": round(100.0 * cl / len(cc), 2)}
        log("   %-10s %-8d %-8d %-8d %-8d %-10d %-10d" % (arm, len(cc), eq, sup, cl, er, ef))
    log("\n   == A absolute levels under the defensible judge vs the quoted judge")
    for sname, sub in subsets.items():
        cur = {arm: sum(1 for x in sub if x["per"][arm]["j"]["current"]) for arm in ARMS4}
        dfn = {arm: sum(1 for x in sub if eff[(x["qid"], arm)]) for arm in ARMS4}
        res["A_absolutes"][sname] = {"n": len(sub), "current": cur, "defensible": dfn}
        log("   %-14s n=%d  quoted(current) %s   defensible %s" % (sname, len(sub), cur, dfn))
    OUT["s19_b_and_c"] = res


def s20_structural_reachability(v1, v2):
    """Can the EXISTING pipeline already compute a correct answer for the 65
    'structurally impossible' questions?  D says no; this measures it.

    time_conflict : answer() already calls _same_scene_conflicts, which DROPS
                    cross-document exclusive pairs as 'story progression'.  So the
                    discriminating evidence is computed and then thrown into
                    warnings.  Measure how often the retrieved context actually
                    contains an exclusive predicate pair on the question's entity
                    pair, and whether the two sides sit in DIFFERENT docs.
    negative_abstain : query.analyze_query already computes negated_hint; measure
                    how often it is set, and whether the quoted line is retrieved.
    adversarial   : measure whether both question entities resolve to distinct
                    graph entities with distinct types (template is then trivial).
    """
    from arkrag import pipeline as P
    from arkrag import vocab as V
    log("\n== S20 structural reachability: is the 65-question ceiling real, or a missing output path?")
    doc_of = {}
    for line in (ROOT / "data/arkrag/chunks.jsonl").open(encoding="utf-8"):
        c = json.loads(line)
        doc_of[str(c.get("chunk_id"))] = str(c.get("doc_id") or "")
    g = json.loads((ROOT / "data/arkrag/graph.json").read_text(encoding="utf-8"))
    name2type, name2id = {}, {}
    for e in g["entities"]:
        name2type.setdefault(e["name"], e.get("type") or "")
        name2id.setdefault(e["name"], e.get("id") or "")
    exc = {tuple(sorted(p)) for p in V.MUTUALLY_EXCLUSIVE_PAIRS}
    preds = sorted({p for pair in V.MUTUALLY_EXCLUSIVE_PAIRS for p in pair})

    dev = [x for x in v2 if str(x.get("split")) == "dev" and x.get("gold_chunk_ids")]
    tq = [x for x in dev if x.get("category") == "time_conflict"]
    nq = [x for x in dev if x.get("category") == "negative_abstain"]
    aq = [x for x in dev if x.get("category") == "adversarial"]
    if not (tq or nq or aq):
        log("   no structural items found; skipped")
        OUT["s20_structural_reachability"] = {"note": "no items"}
        return
    pipe = P.build_pipeline(P.load_generation(GEN), top_k=20, graph_depth=1)
    pipe["retriever"].budget["max_context_tokens"] = 12000
    pipe["answerer"].max_claims = 12

    def run(item):
        res = P.ask_once(pipe, str(item["question"]))
        return res

    res = {"time_conflict": {}, "negative_abstain": {}, "adversarial": {}}

    # ---- time_conflict -------------------------------------------------
    trows = []
    for x in tq:
        r = run(x)
        ans = r["answer"]
        qa = r["query_analysis"]
        # NOTE: anchors() returns every <quoted> span, which for time_conflict
        # includes the RELATION words ("weishu"/"beipan"), not just entities.
        # Requiring every span inside one claim made the check unsatisfiable.
        # Entities are the quoted spans that name a graph entity; predicates are
        # the quoted spans that are relation words.  Keep them apart.
        allq = anchors(str(x["question"]))
        ent_anc = [a for a in allq if a in name2id] or [a for a in allq if a not in preds]
        text = str(ans.get("answer") or "")
        # claim text -> doc ids it was drawn from
        cdoc = {}
        for c in (ans.get("citations") or []):
            cdoc.setdefault(str(c.get("claim") or ""), set()).add(
                doc_of.get(str(c.get("chunk_id")), ""))
        hits = []
        for cl in claims(text):
            ps = [p for p in preds if p in cl]
            on_pair = all(a in cl for a in ent_anc) if ent_anc else False
            if not on_pair:
                on_pair = len([a for a in ent_anc if a in cl]) >= 2
            for p in ps:
                hits.append({"claim": cl, "pred": p, "on_entity_pair": on_pair,
                             "docs": sorted(d for d in cdoc.get(cl, set()) if d)})
        pair_found, cross_doc, same_doc = False, False, False
        for i in range(len(hits)):
            for j in range(i + 1, len(hits)):
                a, b = hits[i], hits[j]
                if tuple(sorted((a["pred"], b["pred"]))) not in exc:
                    continue
                if not (a["on_entity_pair"] and b["on_entity_pair"]):
                    continue
                pair_found = True
                if a["docs"] and b["docs"] and not (set(a["docs"]) & set(b["docs"])):
                    cross_doc = True
                elif a["docs"] and b["docs"]:
                    same_doc = True
        trows.append({
            "qid": x["question_id"], "anchors": allq, "entity_anchors": ent_anc,
            "n_conflicts_reported": int((ans.get("stats") or {}).get("conflicts") or 0),
            "n_warnings": len(ans.get("warnings") or []),
            "exclusive_pair_in_answer": pair_found,
            "exclusive_pair_in_different_docs": cross_doc,
            "exclusive_pair_in_same_doc": same_doc,
            "should_abstain_reported": len(ans.get("warnings") or []),
            "answer_head": text[:100],
        })
    n = len(trows)
    res["time_conflict"] = {
        "n": n,
        "n_has_exclusive_pair": sum(1 for r in trows if r["exclusive_pair_in_answer"]),
        "n_pair_spans_different_docs": sum(1 for r in trows if r["exclusive_pair_in_different_docs"]),
        "n_pair_within_same_doc": sum(1 for r in trows if r["exclusive_pair_in_same_doc"]),
        "n_conflicts_reported_by_answerer": sum(1 for r in trows if r["n_conflicts_reported"] > 0),
        "rows": trows,
    }
    log("   time_conflict  n=%d  exclusive pair present in answer: %d  pair spans DIFFERENT docs "
        "(= 'story progression', the gold answer): %d  pair within same doc: %d" % (
            n, res["time_conflict"]["n_has_exclusive_pair"],
            res["time_conflict"]["n_pair_spans_different_docs"],
            res["time_conflict"]["n_pair_within_same_doc"]))
    log("   (answerer's own conflict counter fired on %d of %d)" % (
        res["time_conflict"]["n_conflicts_reported_by_answerer"], n))

    # ---- negative_abstain ----------------------------------------------
    nrows = []
    for x in nq:
        r = run(x)
        qa = r["query_analysis"]
        ans = r["answer"]
        q = str(x["question"])
        quoted = [m.group(1) for m in QUOTE_RE.finditer(q)]
        texts = [str(u.get("text") or "") for u in (r["context"].get("units") or [])]
        ctx_has_quote = any(quoted and quoted[0] and quoted[0] in t for t in texts)
        nrows.append({
            "qid": x["question_id"], "should_answer": qa.get("should_answer"),
            "negated_hint": bool(qa.get("negated_hint")),
            "negation_markers": qa.get("negation_markers"),
            "quoted_span_in_retrieved_context": ctx_has_quote,
            "answer_head": str(ans.get("answer") or "")[:80],
        })
    m = len(nrows)
    res["negative_abstain"] = {
        "n": m,
        "n_negated_hint_set": sum(1 for r in nrows if r["negated_hint"]),
        "n_quoted_span_retrieved": sum(1 for r in nrows if r["quoted_span_in_retrieved_context"]),
        "n_both": sum(1 for r in nrows
                      if r["negated_hint"] and r["quoted_span_in_retrieved_context"]),
        "rows": nrows,
    }
    log("   negative_abstain n=%d  negated_hint set: %d  quoted span retrieved: %d  both: %d" % (
        m, res["negative_abstain"]["n_negated_hint_set"],
        res["negative_abstain"]["n_quoted_span_retrieved"],
        res["negative_abstain"]["n_both"]))

    # ---- adversarial ----------------------------------------------------
    arows = []
    for x in aq:
        r = run(x)
        qa = r["query_analysis"]
        anc = anchors(str(x["question"]))
        got = {str(c.get("name") or "") for c in (qa.get("entity_candidates") or [])}
        a_, b_ = (anc + ["", ""])[:2]
        ta, tb = name2type.get(a_, ""), name2type.get(b_, "")
        arows.append({
            "qid": x["question_id"], "a": a_, "b": b_, "type_a": ta, "type_b": tb,
            "a_in_candidates": a_ in got, "b_in_candidates": b_ in got,
            "distinct_types": bool(ta and tb and ta != tb),
            "answerable_by_type_template": bool(ta and tb and ta != tb),
        })
    k = len(arows)
    res["adversarial"] = {
        "n": k,
        "n_distinct_types": sum(1 for r in arows if r["distinct_types"]),
        "n_both_entities_resolved_as_candidates": sum(
            1 for r in arows if r["a_in_candidates"] and r["b_in_candidates"]),
        "n_answerable_by_type_template": sum(1 for r in arows if r["answerable_by_type_template"]),
        "rows": arows,
    }
    log("   adversarial    n=%d  distinct graph types: %d  both entities are query candidates: %d" % (
        k, res["adversarial"]["n_distinct_types"],
        res["adversarial"]["n_both_entities_resolved_as_candidates"]))

    tc_ok = res["time_conflict"]["n_pair_spans_different_docs"]
    na_ok = res["negative_abstain"]["n_both"]
    ad_ok = res["adversarial"]["n_answerable_by_type_template"]
    res["verdict"] = {
        "reachable_total": tc_ok + na_ok + ad_ok,
        "of_total": n + m + k,
        "pct": round(100.0 * (tc_ok + na_ok + ad_ok) / max(1, n + m + k), 2),
        "note": ("'structurally impossible' is falsified if this is well above 0: "
                 "the evidence needed is already inside the retrieved context / "
                 "already computed by answer()."),
    }
    log("   => reachable with existing signal: %d of %d (%.1f%%)" % (
        res["verdict"]["reachable_total"], res["verdict"]["of_total"], res["verdict"]["pct"]))
    OUT["s20_structural_reachability"] = res


def main():
    reuse = len(sys.argv) > 1 and sys.argv[1] == "reuse"
    only = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] not in ("reuse",) else ""
    v1, v2 = s1_goldset()
    if only:
        # focused re-run: keep every stored section, recompute just this one
        prev = json.loads((ROOT / "data/audit_redteam.json").read_text(encoding="utf-8"))
        OUT.update(prev)
        if only == "s20":
            s20_structural_reachability(v1, v2)
        if only == "s8":
            s8_judge_slack(v2)
        if only == "s9":
            s9_null_baseline(v1, v2)
        if only == "s13":
            s13_placebo_judge(v2)
        if only == "s14":
            s14_best_effort(v1, v2)
        if only == "s15":
            s15_denominator_and_lookup(v1, v2)
        if only == "s16":
            s16_eq_false_negatives(v1, v2)
        if only == "s17":
            s17_clean_cross_judge(v1, v2)
        if only == "s18":
            s18_defensible_judge(v1, v2)
        if only == "s19":
            s19_b_and_c(v1, v2)
        p = ROOT / "data/audit_redteam.json"
        p.write_text(json.dumps(OUT, ensure_ascii=False, indent=1), encoding="utf-8")
        log("\nwrote %s (%.1f MB)" % (p, p.stat().st_size / 1e6))
        return
    v1, v2 = s1_goldset()
    s2_cross_gold(v2)
    s4_structural(v2)
    if reuse:
        prev = json.loads((ROOT / "data/audit_redteam.json").read_text(encoding="utf-8"))
        OUT["s3_s5"] = prev["s3_s5"]
        OUT["s6_latency"] = prev["s6_latency"]
        rows = prev["s3_s5"]["rows"]
        log("\n== reused S3/S5 rows (%d)" % len(rows))
        items = rows
    else:
        pipe, r, items = s3_s5(v1, v2)
        s6_latency(items)
    s7_followups(v1, v2)
    s8_judge_slack(v2)
    s9_null_baseline(v1, v2)
    s10_sup_eq_gap(v1, v2)
    s11_extra_chapters_true(v2)
    s12_gold_recall(v2)
    s13_placebo_judge(v2)
    s14_best_effort(v1, v2)
    s15_denominator_and_lookup(v1, v2)
    s16_eq_false_negatives(v1, v2)
    s17_clean_cross_judge(v1, v2)
    s18_defensible_judge(v1, v2)
    s19_b_and_c(v1, v2)
    s20_structural_reachability(v1, v2)
    p = ROOT / "data/audit_redteam.json"
    p.write_text(json.dumps(OUT, ensure_ascii=False, indent=1), encoding="utf-8")
    log("\nwrote %s (%.1f MB)" % (p, p.stat().st_size / 1e6))


if __name__ == "__main__":
    main()
