"""ArkRAG 评测口径对抗审查 · 第二轮（只读）

回答 lead 第二轮的四个裁决请求 Q1–Q4，并可复现全部数字。

用法：
    py -3 scripts/audit_round2.py                 # 跑全部四节
    py -3 scripts/audit_round2.py --only q1       # 只跑某节（q1/q2/q3/q4）
    py -3 scripts/audit_round2.py --cache tmp_adv_r2.json   # 缓存真跑结果，便于复算

只读约束：本脚本不写任何被审文件，不修改 judge / eval / goldset。
真跑一律通过 monkeypatch 运行时旋钮（top_k / budget / max_claims），不落盘。
"""

import sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse
import collections
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from arkrag import pipeline as P          # noqa: E402
from arkrag import judge as J             # noqa: E402

GEN = "g-20261103T132152Z-06a741"
ANCHOR_RE = re.compile(r"「([^」]{1,30})」")
SCOR = set(J.SCORABLE_CATEGORIES)


def hr(t):
    print("\n" + "=" * 100)
    print(t)
    print("=" * 100)


# ------------------------------------------------------------------ 载入
print("载入世代 …")
L = P.load_generation(GEN)
graph, adj, chunks, assertions = (L["graph"], L["adjacency"], L["chunks"], L["assertions"])
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
        rec = nbr[eid].setdefault(nid, {"rels": set(), "aids": [], "supp": 0})
        rec["rels"].add(item.get("relation") or "")
        rec["aids"].extend(item.get("assertion_ids") or [])
        rec["supp"] = max(rec["supp"], int(item.get("support_count") or 0))

gold_doc = json.loads((ROOT / "data/goldset/gold-v2.json").read_text("utf-8"))
V1 = {it["question_id"] for it in
      json.loads((ROOT / "data/goldset/gold-v1.json").read_text("utf-8"))["items"]}
ITEMS = {it["question_id"]: it for it in gold_doc["items"]}

chap_acc = collections.defaultdict(list)
for cid, rec in chunks.items():
    chap_acc[str(rec.get("chapter") or "")].append(str(rec.get("text") or ""))
CHAPTER_TEXTS = {k: "\n".join(v) for k, v in chap_acc.items()}


def pick(split):
    if split == "clean":
        return [it for it in gold_doc["items"] if it.get("split") == "holdout"
                and it["question_id"] not in V1 and it.get("gold_chunk_ids")]
    return [it for it in gold_doc["items"]
            if it.get("split") == split and it.get("gold_chunk_ids")]


POOLS = {s: pick(s) for s in ("dev", "holdout", "clean")}
print(f"评测池 dev={len(POOLS['dev'])} holdout={len(POOLS['holdout'])} "
      f"clean={len(POOLS['clean'])}")

PIPE = P.build_pipeline(L, top_k=20, graph_depth=1)
PIPE["generation_id"] = GEN
PIPE["entities"] = graph["entities"]


def make_pipe(top_k, budget, max_claims):
    PIPE["top_k"] = top_k
    PIPE["retriever"].budget["max_context_tokens"] = budget
    PIPE["answerer"].max_claims = max_claims
    return PIPE


def realrun(splits, top_k, budget, max_claims):
    """真跑：返回 {split: {qid: {...}}}。"""
    make_pipe(top_k, budget, max_claims)
    out = {}
    for sp in splits:
        per = {}
        for it in POOLS[sp]:
            q = it["question"]
            res = P.ask_once(PIPE, q)
            a = res.get("answer") or {}
            txt = str(a.get("answer") or "")
            ctx_ids = {str(u.get("chunk_id") or "") for u in
                       (res.get("context") or {}).get("units") or []}
            gold_ids = set(it.get("gold_chunk_ids") or [])
            sents = [s for s in txt.split("。") if s.strip()]
            v = J.judge_answer(it["category"], txt, it, question=q,
                               chapter_texts=CHAPTER_TEXTS)
            per[it["question_id"]] = {
                "pass": v["pass"] is True,
                "reason": v.get("reason"),
                "len": len(txt),
                "n_sents": len(sents),
                "max_sent": max((len(s) for s in sents), default=0),
                "gold_in_ctx": len(gold_ids & ctx_ids),
                "n_gold": len(gold_ids),
                "tokens_used": ((res.get("context") or {}).get("stats") or {})
                .get("tokens_used"),
                "units": len(ctx_ids),
                "answer": txt,
            }
        out[sp] = per
        ok, n = score(per)
        print(f"    {sp:8s} 可判 {ok}/{n} = {ok/n*100:.2f}%   （池内 {len(per)} 题，"
              f"含结构类 {len(per)-n} 题不计分母）")
    return out


def scorable(per):
    """只保留可判类别，与官方 tally 的 scorable 分母一致。"""
    return {qid: x for qid, x in per.items() if ITEMS[qid]["category"] in SCOR}


def score(per):
    sc = scorable(per)
    return sum(1 for x in sc.values() if x["pass"]), len(sc)


# ------------------------------------------------------------------ Q2
def q2(cache):
    hr("§Q2 token budget / top_k 本身算不算注水：只动旋钮的真跑读数（MC8 不变）")
    print("     每格 = 可判通过/可判总数（%）；gold_in_ctx = gold 块落在本次 ctx 的比例")
    grid = [(20, 12000), (20, 24000), (40, 12000), (40, 24000), (40, 40000)]
    res = {}
    for tk, bg in grid:
        key = f"tk{tk}_b{bg}"
        if cache.get(key) is None:
            print(f"\n   真跑 top_k={tk} budget={bg} max_claims=8 …")
            cache[key] = realrun(["dev"], tk, bg, 8)
        per = cache[key]["dev"]
        sc = scorable(per)
        ok, n = score(per)
        gi = sum(x["gold_in_ctx"] for x in sc.values())
        gn = sum(x["n_gold"] for x in sc.values())
        avglen = sum(x["len"] for x in sc.values()) / max(1, n)
        avgcl = sum(x["n_sents"] for x in sc.values()) / max(1, n)
        res[key] = dict(ok=ok, n=n, gold_in_ctx=gi, gold_n=gn, avglen=avglen, avgcl=avgcl)
    print(f"\n   {'配置':16s} {'dev可判':>12s} {'%':>7s} {'gold_in_ctx':>14s} {'均长':>6s} {'均句数':>7s}")
    for (tk, bg) in grid:
        r = res[f"tk{tk}_b{bg}"]
        print(f"   top_k={tk:<3d}B={bg:<6d} {r['ok']:>5}/{r['n']:<6} "
              f"{r['ok']/r['n']*100:>6.2f}% {r['gold_in_ctx']:>6}/{r['gold_n']:<7} "
              f"{r['avglen']:>6.1f} {r['avgcl']:>7.2f}")
    base = res["tk20_b12000"]
    print(f"\n   相对基线 tk20/B12000：")
    for k, r in res.items():
        print(f"     {k:16s} Δpass={r['ok']-base['ok']:+3d}  "
              f"Δgold_in_ctx={r['gold_in_ctx']-base['gold_in_ctx']:+4d}  "
              f"Δ均长={r['avglen']-base['avglen']:+.1f}  Δ均句数={r['avgcl']-base['avgcl']:+.2f}")
    cache["BASE"] = cache["tk20_b12000"]   # 供 §Q4 复用，避免重跑
    return res


# ------------------------------------------------------------------ Q1
def cand_list(A, B):
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
        rels, sup = set(), 0
        for src in list(na[x]) + list(nb[x]):
            r = nbr[x].get(src)
            if r:
                rels |= r["rels"]
                sup += r["supp"]
        out.append((x, len(rels), sup, len(nbr[x])))
    out.sort(key=lambda t: (-t[1], -t[2], t[3], t[0]))
    return out


def dominant_ok(cs):
    if len(cs) == 1:
        return True
    return cs[0][2] >= 2 * cs[1][2] and cs[0][2] >= 4


def sent_of(a):
    sn, on = name_of.get(a.get("subject_id"), ""), name_of.get(a.get("object_id"), "")
    pr = a.get("predicate") or ""
    return f"{sn}{pr}{on}。" if (sn and on and pr) else None


def bridge_text(x, A, B):
    parts = [name_of.get(x, "") + "。"]
    for tgt in (A, B):
        ok = False
        for o in tgt:
            rec = nbr[x].get(o)
            if not rec:
                continue
            for aid in rec["aids"]:
                a = aid2a.get(aid)
                if not a:
                    continue
                s = sent_of(a)
                if s:
                    parts.append(s)
                    ok = True
                    break
            if ok:
                break
        if not ok:
            return None
    return "".join(parts)


def q1(cache):
    hr("§Q1 候选来源对照：增益来自「图推导」还是「句式 + goldset 构造」")
    print("     只在桥接会触发（dominant 门槛通过）的 multihop 题上比较。")
    print("     句式一律 `X。X与A相关。X与B相关。`（真实三元组版另测，§94 已证边际 0）。")
    rnd = random.Random(20261107)
    allnames = list(name_of.values())
    table = {}
    for sp in ("dev", "holdout", "clean"):
        agg = collections.Counter()
        for it in POOLS[sp]:
            if it["category"] != "multihop":
                continue
            q = it["question"]
            anc = ANCHOR_RE.findall(q)
            if len(anc) < 2:
                continue
            A, B = by_name.get(anc[0], []), by_name.get(anc[1], [])
            if not (A and B):
                continue
            cs = cand_list(A, B)
            if not cs or not dominant_ok(cs):
                continue
            agg["fired"] += 1
            if len(cs) == 1:
                agg["unique"] += 1
            gold = str(it["answer"])

            def judge(txt):
                return J.judge_answer("multihop", txt, it, question=q)["pass"] is True

            # 1) 图推导 top1（lead 的方案）
            t = bridge_text(cs[0][0], A, B)
            agg["graph_top1"] += judge(t) if t else 0
            # 2) 图推导次名 top2（同样来自候选集，只是排序次优）
            if len(cs) > 1:
                t2 = bridge_text(cs[1][0], A, B)
                agg["graph_top2"] += judge(t2) if t2 else 0
            # 3) 候选集里**随机**挑一个（不看排序）—— 隔离「排序」的贡献
            x = rnd.choice([c[0] for c in cs])
            tr = f"{name_of.get(x,'')}。{name_of.get(x,'')}与{anc[0]}相关。{name_of.get(x,'')}与{anc[1]}相关。"
            agg["rand_in_cands"] += judge(tr)
            # 4) A 的随机邻居（图上相关，但不是公共邻居）
            na = list(nbr[cs[0][0]].keys()) if False else [n for a in A for n in nbr[a]]
            if na:
                y = rnd.choice(na)
                ty = f"{name_of.get(y,'')}。{name_of.get(y,'')}与{anc[0]}相关。{name_of.get(y,'')}与{anc[1]}相关。"
                agg["rand_nbr_of_A"] += judge(ty)
            # 5) 两个锚点各自邻居里各取一个再合成？—— 直接取 B 的随机邻居
            nb = [n for b in B for n in nbr[b]]
            if nb:
                z = rnd.choice(nb)
                tz = f"{name_of.get(z,'')}。{name_of.get(z,'')}与{anc[0]}相关。{name_of.get(z,'')}与{anc[1]}相关。"
                agg["rand_nbr_of_B"] += judge(tz)
            # 6) 全图随机实体
            w = rnd.choice(allnames)
            tw = f"{w}。{w}与{anc[0]}相关。{w}与{anc[1]}相关。"
            agg["rand_whole_graph"] += judge(tw)
            # 7) 形状 oracle（答案名 = gold）
            to = f"{gold}。{gold}与{anc[0]}相关。{gold}与{anc[1]}相关。"
            agg["oracle_gold"] += judge(to)
        table[sp] = agg
    print(f"\n   {'split':8s} {'触发':>5s} {'其中唯一':>8s} {'图top1':>7s} {'图top2':>7s} "
          f"{'候选内随机':>10s} {'A邻居随机':>9s} {'B邻居随机':>9s} {'全图随机':>8s} {'gold oracle':>11s}")
    for sp in ("dev", "holdout", "clean"):
        a = table[sp]
        print(f"   {sp:8s} {a['fired']:>5d} {a['unique']:>8d} {a['graph_top1']:>7d} "
              f"{a['graph_top2']:>7d} {a['rand_in_cands']:>10d} {a['rand_nbr_of_A']:>9d} "
              f"{a['rand_nbr_of_B']:>9d} {a['rand_whole_graph']:>8d} {a['oracle_gold']:>11d}")
    print("\n   关键比值（ALL 三 split 合计）：")
    tot = collections.Counter()
    for sp in table:
        tot.update(table[sp])
    f = max(1, tot["fired"])
    for k in ("graph_top1", "graph_top2", "rand_in_cands", "rand_nbr_of_A",
              "rand_nbr_of_B", "rand_whole_graph", "oracle_gold"):
        print(f"     {k:18s} {tot[k]:>3d}/{tot['fired']} = {tot[k]/f*100:5.1f}%")
    print(f"\n   ① 句式本身的信息量（全图随机名）：{tot['rand_whole_graph']/f*100:.1f}%")
    print(f"   ② 「图上一个邻居」的信息量（A/B 邻居随机）：{(tot['rand_nbr_of_A']+tot['rand_nbr_of_B'])/2/f*100:.1f}%")
    print(f"   ③ 「公共邻居集合」的信息量（候选内随机）：{tot['rand_in_cands']/f*100:.1f}%")
    print(f"   ④ 「排序选 top1」的增量 = ③→图top1：{tot['graph_top1']/f*100 - tot['rand_in_cands']/f*100:+.1f}pp")
    print(f"   ⑤ 唯一候选占比：{tot['unique']}/{tot['fired']} = {tot['unique']/f*100:.1f}%"
          f"（唯一时候选内随机 ≡ top1，排序在数学上不可能贡献）")
    return table


# ------------------------------------------------------------------ Q3
def q3(cache):
    hr("§Q3 配置 E（MC 8→20）相对配置 A 的增量是否来自「多吐断言」")
    for key, (tk, bg, mc) in {"A": (40, 24000, 8), "E": (40, 40000, 20),
                              "A2": (40, 40000, 8)}.items():
        if cache.get(key) is None:
            print(f"\n   真跑配置 {key} top_k={tk} B={bg} MC={mc} …")
            cache[key] = realrun(["dev", "holdout", "clean"], tk, bg, mc)
    print(f"\n   {'配置':6s} {'dev':>12s} {'holdout':>12s} {'clean':>12s}")
    for key in ("A", "A2", "E"):
        row = []
        for sp in ("dev", "holdout", "clean"):
            ok, n = score(cache[key][sp])
            row.append(f"{ok}/{n}={ok/n*100:.2f}%")
        print(f"   {key:6s} " + " ".join(f"{x:>13s}" for x in row))

    # E vs A 逐题增量分解：把 E 的答案截断到前 8 句（= A 的断言上限）再判
    print("\n   E 相对 A 的增量分解（dev）：把 E 的答案截到前 8 句后是否仍通过")
    Adev, Edev = cache["A"]["dev"], cache["E"]["dev"]
    gain = [qid for qid in Adev
            if Edev[qid]["pass"] and not Adev[qid]["pass"]]
    loss = [qid for qid in Adev
            if Adev[qid]["pass"] and not Edev[qid]["pass"]]
    print(f"     增 {len(gain)} 题 / 失 {len(loss)} 题")
    kept = 0
    for qid in gain:
        it = ITEMS[qid]
        txt = Edev[qid]["answer"]
        sents = [s for s in txt.split("。") if s.strip()]
        trunc = "。".join(sents[:8]) + ("。" if sents[:8] else "")
        v = J.judge_answer(it["category"], trunc, it, question=it["question"],
                           chapter_texts=CHAPTER_TEXTS)
        kept += v["pass"] is True
    print(f"     ⇒ 这 {len(gain)} 题里，**只保留前 8 句仍能通过：{kept} 题**，"
          f"另 {len(gain)-kept} 题依赖第 9~20 句")
    print(f"     ⇒ 「堆断言」对 +{len(gain)-len(loss)} 净增量的贡献 = "
          f"{(len(gain)-kept)}/{max(1,len(gain)-len(loss))} = "
          f"{(len(gain)-kept)/max(1,len(gain)-len(loss))*100:.1f}%")

    print("\n   形态统计（dev，可判类）：")
    print(f"   {'配置':6s} {'均长':>7s} {'P50':>5s} {'P95':>5s} {'max':>5s} "
          f"{'均句数':>7s} {'max句>60字':>10s} {'答案>1000字':>11s}")
    for key in ("A", "A2", "E"):
        per = scorable(cache[key]["dev"])
        lens = sorted(x["len"] for x in per.values())
        ncl = [x["n_sents"] for x in per.values()]
        over60 = sum(x["max_sent"] > 60 for x in per.values())
        over1000 = sum(x["len"] > 1000 for x in per.values())
        print(f"   {key:6s} {sum(lens)/len(lens):>7.1f} {lens[len(lens)//2]:>5d} "
              f"{lens[int(len(lens)*.95)]:>5d} {lens[-1]:>5d} "
              f"{sum(ncl)/len(ncl):>7.2f} {over60:>6}/{len(lens):<3} {over1000:>8}/{len(lens)}")
    # 断言条数分布
    print("\n   每题断言条数分布（dev，可判类）：")
    for key in ("A", "E"):
        c = collections.Counter(x["n_sents"] for x in
                                scorable(cache[key]["dev"]).values())
        print(f"     {key}: {dict(sorted(c.items()))}")
    return cache


# ------------------------------------------------------------------ Q4
def q4(cache):
    hr("§Q4 红线上限：不碰 judge、不堆断言、不做形状 oracle，三份能到多少")
    plan = [("基线 tk20/B12000/MC8", "BASE", (20, 12000, 8)),
            ("A  tk40/B24000/MC8", "A", (40, 24000, 8)),
            ("A' tk40/B40000/MC8", "A2", (40, 40000, 8)),
            ("E  tk40/B40000/MC20", "E", (40, 40000, 20))]
    rows = []
    for label, key, (tk, bg, mc) in plan:
        if cache.get(key) is None:
            print(f"   真跑 {label} …")
            cache[key] = realrun(["dev", "holdout", "clean"], tk, bg, mc)
        rows.append((label, cache[key]))
    print(f"\n   {'配置':24s} {'dev':>13s} {'holdout':>13s} {'clean':>13s} {'过 80%?':>8s}")
    for key, per in rows:
        cells = []
        okall = True
        for sp in ("dev", "holdout", "clean"):
            ok, n = score(per[sp])
            cells.append(f"{ok}/{n}={ok/n*100:.2f}%")
            okall &= (ok / n) >= 0.8
        print(f"   {key:24s} " + " ".join(f"{c:>13s}" for c in cells)
              + f" {'✅' if okall else '❌':>8s}")
    print("\n   说明：")
    print("     · 以上四行**全部只用旋钮**，判据未改、答案未堆断言（MC8 两行）、未做形状 oracle。")
    print("     · MC20 行（E）已被 §Q3 证明增量主要来自断言 9~20 句，本表列为参照。")
    print("     · 桥接 / DR 直连边 / cross_chunk 否决式均被第一轮红线排除，不进入上限。")
    return rows


def q5(cache):
    hr("§Q5 独立复现 lead 的 A–E 五行（桥接叠加在真跑之上；含接地约束对照）")

    def sent_of(a):
        sn, on = name_of.get(a.get("subject_id"), ""), name_of.get(a.get("object_id"), "")
        pr = a.get("predicate") or ""
        return f"{sn}{pr}{on}。" if (sn and on and pr) else None

    def bridge(x, A, B, ctx_ids):
        parts = [name_of.get(x, "") + "。"]
        cites = []
        for tgt in (A, B):
            hit = False
            for o in tgt:
                rec = nbr[x].get(o)
                if not rec:
                    continue
                for aid in rec["aids"]:
                    a = aid2a.get(aid)
                    if not a:
                        continue
                    s = sent_of(a)
                    if s:
                        if ctx_ids is not None and str(a.get("chunk_id") or "") not in ctx_ids:
                            continue          # 接地：证据块必须已在本次 ctx 内
                        parts.append(s)
                        cites.append(aid)
                        hit = True
                        break
                if hit:
                    break
            if not hit:
                return None, cites
        return "".join(parts), cites

    def run(tk, bg, mc, mode, ground):
        make_pipe(tk, bg, mc)
        out = {}
        for sp in ("dev", "holdout", "clean"):
            ok = n = fired = 0
            for it in POOLS[sp]:
                q = it["question"]
                res = P.ask_once(PIPE, q)
                txt = str((res.get("answer") or {}).get("answer") or "")
                ctx_ids = {str(u.get("chunk_id") or "") for u in
                           (res.get("context") or {}).get("units") or []}
                if mode != "none" and it["category"] == "multihop":
                    anc = ANCHOR_RE.findall(q)
                    if len(anc) >= 2:
                        A, B = by_name.get(anc[0], []), by_name.get(anc[1], [])
                        if A and B:
                            cs = cand_list(A, B)
                            take = False
                            if cs:
                                take = (len(cs) == 1) if mode == "unique_only" \
                                    else dominant_ok(cs)
                            if take:
                                bt, _ = bridge(cs[0][0], A, B,
                                               ctx_ids if ground else None)
                                if bt:
                                    txt = bt
                                    fired += 1
                if it["category"] in SCOR:
                    n += 1
                    ok += J.judge_answer(it["category"], txt, it, question=q,
                                         chapter_texts=CHAPTER_TEXTS)["pass"] is True
            out[sp] = (ok, n, fired)
        return out

    CFG = [("基线 tk20/B12000/MC8", "Q5_BASE", (20, 12000, 8, "none", False)),
           ("A  tk40/B24000/MC8", "Q5_A", (40, 24000, 8, "none", False)),
           ("A2 tk40/B40000/MC8", "Q5_A2", (40, 40000, 8, "none", False)),
           ("B  =A+桥接dominant", "Q5_B", (40, 24000, 8, "dominant", False)),
           ("C  =A+桥接 B→40000", "Q5_C", (40, 40000, 8, "dominant", False)),
           ("D  =A+桥接unique_only", "Q5_D", (40, 24000, 8, "unique_only", False)),
           ("E  tk40/B40000/MC20+桥接dom", "Q5_E", (40, 40000, 20, "dominant", False)),
           ("E' 同上+接地约束", "Q5_Eg", (40, 40000, 20, "dominant", True))]
    for label, key, args in CFG:
        if cache.get(key) is None:
            print(f"\n   真跑 {label} …")
            cache[key] = run(*args)
    print(f"\n   {'配置':34s} {'dev':>13s} {'holdout':>13s} {'clean':>13s}")
    for label, key, _ in CFG:
        r = cache[key]
        cells = [f"{r[s][0]}/{r[s][1]}={r[s][0]/r[s][1]*100:.2f}%"
                 for s in ("dev", "holdout", "clean")]
        print(f"   {label:34s} " + " ".join(f"{c:>13s}" for c in cells))
    print("\n   桥接净增与接地代价：")
    for label, key, basekey in (("B dominant", "Q5_B", "Q5_A"),
                                ("D unique_only", "Q5_D", "Q5_A"),
                                ("E dominant+MC20", "Q5_E", "Q5_A2"),
                                ("E' 接地版", "Q5_Eg", "Q5_A2")):
        r, b = cache[key], cache[basekey]
        print(f"     {label:18s} 触发 {r['dev'][2]:>2d} 题  "
              f"dev {b['dev'][0]}→{r['dev'][0]}({r['dev'][0]-b['dev'][0]:+d})  "
              f"hold {b['holdout'][0]}→{r['holdout'][0]}({r['holdout'][0]-b['holdout'][0]:+d})  "
              f"clean {b['clean'][0]}→{r['clean'][0]}({r['clean'][0]-b['clean'][0]:+d})")
    e, eg, a2 = cache["Q5_E"], cache["Q5_Eg"], cache["Q5_A2"]
    print(f"\n   ⇒ 接地约束（证据块必须在本次 ctx 内）把 E 的 dev 净增 "
          f"{e['dev'][0]-a2['dev'][0]:+d} 压到 {eg['dev'][0]-a2['dev'][0]:+d}，"
          f"即 {e['dev'][0]-eg['dev'][0]} 题依赖**不在本次上下文里**的证据。")
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["q1", "q2", "q3", "q4", "q5"], default=None)
    ap.add_argument("--cache", default=None, help="缓存真跑结果的 JSON 路径")
    args = ap.parse_args()

    cache = {}
    if args.cache and Path(args.cache).exists():
        cache = json.loads(Path(args.cache).read_text("utf-8"))
        print(f"[cache] 载入 {args.cache}（{len(cache)} 项）")

    order = ["q2", "q1", "q3", "q4", "q5"]
    if args.only:
        order = [args.only]
    for name in order:
        {"q1": q1, "q2": q2, "q3": q3, "q4": q4, "q5": q5}[name](cache)

    if args.cache:
        Path(args.cache).write_text(json.dumps(cache, ensure_ascii=False), "utf-8")
        print(f"\n[cache] 写入 {args.cache}")


if __name__ == "__main__":
    main()
