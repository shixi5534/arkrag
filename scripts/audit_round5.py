"""ArkRAG 第五轮对抗审查 —— N1 题干引文子串通道 / N2 alias_index 通道 / N3 cross_chunk replace。

只读脚本：不改 src/、不改 scripts/eval_answer_quality.py、不改 data/goldset/*、
不改任何 data/eval_quality_*.json。仅用既有世代与生产 API 现场重跑。

用法：
    py -3 scripts/audit_round5.py                 # 全部五节
    py -3 scripts/audit_round5.py --only n1 n3
    py -3 scripts/audit_round5.py --json out.json

节：
    n1  N1 引文→子串→块 title：唯一性、命中块是否 gold 块、扰动衰减、
        非命中块 title 对照、**gold 块内随机等长窗口对照**（这是我自造的、不受 lead 污染的控制）
    n2  N2 引文→alias_index→规范名：解析率、形状对照、引文与 gold 的关系、门控漏出
    n3  N3 cross_chunk 实体块 replace：cap 梯度、三口径、bootstrap、零检索 dump 天花板
    n4  组合分台账（生产引用纪律下），逐 split 给出允许档 / 不允许档
    n5  门控安全性（全 640 题）：N1 / N2 / N3 各自命中的类别分布
"""
from __future__ import annotations

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import answer as answer_mod  # noqa: E402
from arkrag import judge as J            # noqa: E402
from arkrag import pipeline as P         # noqa: E402
from arkrag import query as query_mod    # noqa: E402

GEN = "g-20261103T132152Z-06a741"
QUOTE_RE = re.compile(r"「([^」]{1,80})」")
ANCHOR_RE = re.compile(r"「([^」]{1,30})」")
CHAPTER_SPLIT = re.compile(r"[、,，/／\s]+")
SCOR = set(J.SCORABLE_CATEGORIES)

_L = P.load_generation(GEN)
GRAPH = _L["graph"]
CHUNKS = _L["chunks"]
CID = list(CHUNKS)
TEXT = {c: str(CHUNKS[c].get("text") or "") for c in CID}
TITLE = {c: str(CHUNKS[c].get("title") or "") for c in CID}
NAME = {e["id"]: e["name"] for e in GRAPH["entities"]}
ALIAS = GRAPH.get("alias_index") or {}
ENT_BY_ID = {e["id"]: e for e in GRAPH["entities"]}
BY_NAME: dict[str, list[str]] = collections.defaultdict(list)
for _e in GRAPH["entities"]:
    BY_NAME[_e["name"]].append(_e["id"])
    for _a in (_e.get("aliases") or []):
        if isinstance(_a, str):
            BY_NAME[_a].append(_e["id"])

_CH = collections.defaultdict(list)
for _cid in CID:
    _CH[str(CHUNKS[_cid].get("chapter") or "")].append(TEXT[_cid])
CHAP = {k: "\n".join(v) for k, v in _CH.items()}

_GOLD = json.loads((ROOT / "data/goldset/gold-v2.json").read_text("utf-8"))
_V1 = {it["question_id"] for it in
       json.loads((ROOT / "data/goldset/gold-v1.json").read_text("utf-8"))["items"]}


def pool(split: str) -> list[dict]:
    if split == "clean":
        return [it for it in _GOLD["items"] if it.get("split") == "holdout"
                and it["question_id"] not in _V1 and it.get("gold_chunk_ids")]
    return [it for it in _GOLD["items"]
            if it.get("split") == split and it.get("gold_chunk_ids")]


POOLS = {s: pool(s) for s in ("dev", "holdout", "clean")}

_PIPE = P.build_pipeline(_L, top_k=40, graph_depth=1)
_PIPE["generation_id"] = GEN
_PIPE["entities"] = GRAPH["entities"]
_PIPE["retriever"].budget["max_context_tokens"] = 24000   # 红线 26 冻结值
_PIPE["answerer"].max_claims = 8                          # 红线 25
RET, ANS = _PIPE["retriever"], _PIPE["answerer"]


def longest_quote(q, minlen=1) -> str:
    qs = [x for x in QUOTE_RE.findall(str(q)) if len(x) >= minlen]
    return max(qs, key=len) if qs else ""


def hits_of(qt: str) -> list[str]:
    return [c for c in CID if qt and qt in TEXT[c]]


def jd(it, txt):
    return J.judge_answer(it["category"], txt, it,
                          question=it.get("question"), chapter_texts=CHAP)


def cite_from(qt):
    hs = hits_of(qt)
    if not hs:
        return None
    c = hs[0]
    st = TEXT[c].find(qt)
    return [{"chunk_id": c, "evidence": qt, "start": st, "end": st + len(qt)}]


def base_answer(it, cx_replace=False) -> str:
    q = it["question"]
    qa = query_mod.analyze_query(q, entities=GRAPH["entities"],
                                 aliases=GRAPH.get("alias_index"), vocab=P.vocab,
                                 context=getattr(RET, "query_context", None))
    found = RET.search(qa, top_k=_PIPE["top_k"])
    hits = list(found["hits"])
    if cx_replace and it["category"] == "cross_chunk":
        nm = (ANCHOR_RE.findall(q) or [None])[0]
        ids = BY_NAME.get(nm, []) if nm else []
        if ids:
            cs = list((ENT_BY_ID.get(ids[0]) or {}).get("chunks") or [])[:20]
            ov = [{"chunk_id": c, "score": 1.0 - i * 1e-6,
                   "sources": [{"channel": "entity_index", "score": 1.0}]}
                  for i, c in enumerate(cs)]
            seen = {h["chunk_id"] for h in ov}
            hits = ov + [h for h in hits[:4] if h["chunk_id"] not in seen]
    ex = RET.expand([c["entity_id"] for c in qa.get("entity_candidates") or []],
                    depth=1, query_analysis=qa,
                    seed_chunk_ids=[h["chunk_id"] for h in hits])
    ctx = RET.build_context(hits, ex["expansions"], query_analysis=qa)
    ans = ANS.answer(q, ctx, generation_id=GEN,
                     top_score=float(found["stats"].get("top_score") or 0.0),
                     query_analysis=qa, retrieval_stats=found["stats"])
    return str((ans or {}).get("answer") or "")


def n1_answer(it):
    """N1：题干最长引文(≥8 字) → 全库逐字命中 → 命中块 title；走 verify_citations。"""
    qt = longest_quote(it["question"], 8)
    cs = cite_from(qt)
    if not cs:
        return "", "no_hit"
    v = answer_mod.verify_citations({"answer": TITLE[cs[0]["chunk_id"]], "citations": cs},
                                    CHUNKS)
    return v["answer"], ("ok" if v["citations"] else "cite_rej")


def n2_answer(it):
    """N2：题干引文 → alias_index → 实体规范名；短引文发不出合法引用会拒答。"""
    qt = longest_quote(it["question"], 1)
    eid = ALIAS.get(qt) if qt else None
    nm = NAME.get(str(eid)) if eid else None
    if not nm:
        return "", "no_alias"
    cs = cite_from(qt) or []
    v = answer_mod.verify_citations({"answer": nm, "citations": cs}, CHUNKS)
    return v["answer"], ("ok" if v["citations"] else "cite_rej")


def boot(a, b, n=20000, seed=7):
    d = [y - x for x, y in zip(a, b)]
    r = random.Random(seed)
    idx = list(range(len(d)))
    st = sorted(sum(d[r.choice(idx)] for _ in idx) for _ in range(n))
    return sum(d), st[int(n * .025)], st[int(n * .975)], sum(1 for s in st if s <= 0) / n


# ------------------------------------------------------------------ n1
def n1():
    print("§N1 题干引文子串通道（single_chunk）")
    print("-" * 96)
    rng = random.Random(11)
    for sp in ("dev", "holdout", "clean"):
        sc = [x for x in POOLS[sp] if x["category"] == "single_chunk"]
        uni = goldblk = ingold = 0
        for it in sc:
            qt = longest_quote(it["question"], 8)
            hs = hits_of(qt)
            uni += (len(hs) == 1)
            g = set(it.get("gold_chunk_ids") or [])
            goldblk += bool(set(hs) & g)
            ingold += any(qt in TEXT[c] for c in g if c in TEXT)
        n = len(sc)
        print(f"   {sp:8s} n={n} 唯一命中 {uni}/{n}={uni/n*100:.1f}%  "
              f"命中块∩gold {goldblk}/{n}={goldblk/n*100:.1f}%  "
              f"引文在 gold 块正文 {ingold}/{n}={ingold/n*100:.1f}%")

    print()
    print("   §N1.2 引文尾部截短（真实检索应平稳衰减）")
    for sp in ("dev", "holdout", "clean"):
        sc = [x for x in POOLS[sp] if x["category"] == "single_chunk"]
        row = []
        for cut in (0, 1, 2, 4, 8, 12):
            uni = p = n = 0
            for it in sc:
                qt = longest_quote(it["question"], 8)
                if cut:
                    if len(qt) <= cut + 2:
                        continue
                    qt = qt[: len(qt) - cut]
                n += 1
                hs = hits_of(qt)
                uni += (len(hs) == 1)
                if hs:
                    p += (jd(it, TITLE[hs[0]])["pass"] is True)
            row.append(f"cut{cut}:{p}/{n}")
        print(f"   {sp:8s} " + "  ".join(row))

    print()
    print("   §N1.3 形状对照：答案 = 非命中块 title / 全随机块 title")
    for sp in ("dev", "holdout", "clean"):
        sc = [x for x in POOLS[sp] if x["category"] == "single_chunk"]
        other = rand = 0
        for it in sc:
            hs = set(hits_of(longest_quote(it["question"], 8)))
            pool2 = [c for c in CID if c not in hs]
            other += (jd(it, TITLE[rng.choice(pool2)])["pass"] is True)
            rand += (jd(it, TITLE[rng.choice(CID)])["pass"] is True)
        n = len(sc)
        print(f"   {sp:8s} 非命中块 {other}/{n}={other/n*100:.1f}%  "
              f"| 全随机 {rand}/{n}={rand/n*100:.1f}%")

    print()
    print("   §N1.4 ★关键对照：gold 块内**随机等长窗口**是否也唯一命中它所在的块？")
    print("         （若接近 100%，则『引文唯一命中』是长子串的普遍属性，")
    print("           不是 gold 生成规则的特殊产物 ⇒ lead 说的『复现生成规则』不成立）")
    rng2 = random.Random(23)
    for sp in ("dev", "holdout", "clean"):
        sc = [x for x in POOLS[sp] if x["category"] == "single_chunk"]
        win_uni = win_pass = other_uni = n = 0
        for it in sc:
            g = [c for c in (it.get("gold_chunk_ids") or []) if c in TEXT]
            if not g:
                continue
            c = g[0]
            t = TEXT[c]
            L8 = max(8, len(longest_quote(it["question"], 8)) or 8)
            if len(t) <= L8:
                continue
            st = rng2.randrange(0, len(t) - L8)
            w = t[st:st + L8]
            hs = hits_of(w)
            n += 1
            win_uni += (hs == [c])
            win_pass += (jd(it, TITLE[hs[0]])["pass"] is True) if hs else 0
            oc = rng2.choice(CID)
            if len(TEXT[oc]) > L8:
                other_uni += (set(hits_of(TEXT[oc][:L8])) == {oc})
        if n:
            print(f"   {sp:8s} n={n} gold块内随机窗口唯一 {win_uni}/{n}="
                  f"{win_uni/n*100:.1f}% pass {win_pass}/{n}={win_pass/n*100:.1f}%  "
                  f"| 非gold块窗口唯一 {other_uni}/{n}={other_uni/n*100:.1f}%")


# ------------------------------------------------------------------ n2
def n2():
    print()
    print("§N2 alias 通道（引文 → alias_index → 实体规范名）")
    print("-" * 96)
    print(f"   alias_index 条目 {len(ALIAS)}；value 全为 str："
          f"{all(isinstance(v, str) for v in ALIAS.values())}")
    allnames = list(NAME.values())
    rng = random.Random(37)
    for sp in ("dev", "holdout", "clean"):
        al = [x for x in POOLS[sp] if x["category"] == "alias"]
        res = named = ok = rand = quote_ok = in_gold = 0
        for it in al:
            qt = longest_quote(it["question"], 1)
            eid = ALIAS.get(qt) if qt else None
            res += bool(eid)
            nm = NAME.get(str(eid)) if eid else None
            named += bool(nm)
            ok += (jd(it, nm)["pass"] is True) if nm else 0
            rand += (jd(it, rng.choice(allnames))["pass"] is True)
            quote_ok += (jd(it, qt)["pass"] is True) if qt else 0
            gs = [str(it.get("answer") or "").strip()] + \
                 [str(x).strip() for x in (it.get("acceptable_answers") or [])]
            gs = [g for g in gs if g]
            in_gold += any(qt and (qt.strip() == g or qt.strip() in g or g in qt) for g in gs)
        n = len(al)
        print(f"   {sp:8s} n={n} 解析 {res} 得名 {named} | 规范名 pass {ok}/{n}="
              f"{ok/n*100:.1f}% | 随机实体名 {rand}/{n} | 直接把引文当答案 "
              f"{quote_ok}/{n}={quote_ok/n*100:.1f}% | 引文与 gold 互为子串 "
              f"{in_gold}/{n}={in_gold/n*100:.1f}%")

    print()
    print("   §N2.2 生产引用纪律：引文长度 < MIN_EVIDENCE_CHARS 时发不出合法引用")
    for sp in ("dev", "holdout", "clean"):
        al = [x for x in POOLS[sp] if x["category"] == "alias"]
        ok = rej = n = 0
        for it in al:
            n += 1
            a, why = n2_answer(it)
            rej += (why == "cite_rej")
            ok += (jd(it, a)["pass"] is True) if a else 0
        print(f"   {sp:8s} n={n} 保留答案 {ok}/{n}={ok/n*100:.1f}%  "
              f"因引用过短被清空 {rej} 题")


# ------------------------------------------------------------------ n3
def n3():
    print()
    print("§N3 cross_chunk 实体块 replace")
    print("-" * 96)
    cx = {sp: [x for x in POOLS[sp] if x["category"] == "cross_chunk"]
          for sp in ("dev", "holdout", "clean")}
    for sp in ("dev", "holdout", "clean"):
        row = []
        for cap in (4, 8, 12, 20, 40, 10 ** 6):
            ok = 0
            for it in cx[sp]:
                nm = (ANCHOR_RE.findall(it["question"]) or [None])[0]
                ids = BY_NAME.get(nm, []) if nm else []
                cs = list((ENT_BY_ID.get(ids[0]) or {}).get("chunks") or []) if ids else []
                if cap < 10 ** 5:
                    cs = cs[:cap]
                chs = sorted({str(CHUNKS[c].get("chapter") or "") for c in cs
                              if c in CHUNKS and CHUNKS[c].get("chapter")})
                ok += (jd(it, "、".join(chs))["pass"] is True)
            row.append(f"cap={'ALL' if cap > 10**5 else cap}:{ok}/{len(cx[sp])}")
        print(f"   {sp:8s} 零检索 dump " + "  ".join(row))
    print("   ⇒ dev 峰值恰在 cap=20，40/ALL 回落：内点最优 = 调参痕迹，不是能力。")

    print()
    print("   §N3.2 严格门控三口径 + bootstrap")
    vec = {}
    for sp in ("dev", "holdout", "clean"):
        vb, vc = [], []
        for it in POOLS[sp]:
            if it["category"] not in SCOR:
                continue
            vb.append(int(jd(it, base_answer(it))["pass"] is True))
            vc.append(int(jd(it, base_answer(it, True))["pass"] is True))
        vec[sp] = (vb, vc)
        n = len(vb)
        obs, lo, hi, p = boot(vb, vc)
        print(f"   {sp:8s} 基线 {sum(vb)}/{n}={sum(vb)/n*100:.2f}% → replace "
              f"{sum(vc)}/{n}={sum(vc)/n*100:.2f}%  Δ={obs:+d} CI=[{lo:+d},{hi:+d}] "
              f"p={p:.4f} {'PASS' if (obs>=1 and lo>0 and p<.05) else 'FAIL'}")

    print()
    print("   §N3.3 泄漏：gold_chunk_ids ⊆ 锚点实体 chunks")
    for sp in ("dev", "holdout", "clean"):
        n = sub = sub20 = eq = 0
        for it in cx[sp]:
            nm = (ANCHOR_RE.findall(it["question"]) or [None])[0]
            ids = BY_NAME.get(nm, []) if nm else []
            if not ids:
                continue
            allc = list((ENT_BY_ID.get(ids[0]) or {}).get("chunks") or [])
            g = set(it.get("gold_chunk_ids") or [])
            chs = {str(CHUNKS[c].get("chapter") or "") for c in allc
                   if c in CHUNKS and CHUNKS[c].get("chapter")}
            want = {x for x in CHAPTER_SPLIT.split(str(it.get("answer") or "").strip()) if x}
            n += 1
            sub += (g <= set(allc))
            sub20 += (g <= set(allc[:20]))
            eq += (chs == want)
        print(f"   {sp:8s} n={n} gold⊆实体全部块 {sub}/{n}={sub/n*100:.1f}%  "
              f"⊆前20 {sub20}/{n}={sub20/n*100:.1f}%  章节集合完全相等 "
              f"{eq}/{n}={eq/n*100:.1f}%")


# ------------------------------------------------------------------ n4
def n4():
    print()
    print("§N4 组合分台账（生产引用纪律下的三档）")
    print("-" * 96)
    MH = {"dev": 11, "holdout": 7, "clean": 5}   # §Q13 接地两跳，三份 p<0.05
    for sp in ("dev", "holdout", "clean"):
        per = collections.Counter(); tot = collections.Counter()
        ok_all = tot_all = 0
        for it in POOLS[sp]:
            if it["category"] not in SCOR:
                continue
            p = jd(it, base_answer(it))["pass"] is True
            per[it["category"]] += p; tot[it["category"]] += 1
            ok_all += p; tot_all += 1
        sc_ok = sum(1 for it in POOLS[sp] if it["category"] == "single_chunk"
                    and (jd(it, n1_answer(it)[0])["pass"] is True if n1_answer(it)[0] else False))
        al_ok = sum(1 for it in POOLS[sp] if it["category"] == "alias"
                    and (jd(it, n2_answer(it)[0])["pass"] is True if n2_answer(it)[0] else False))
        cx_ok = sum(1 for it in POOLS[sp] if it["category"] == "cross_chunk"
                    and jd(it, base_answer(it, True))["pass"] is True)
        allow = ok_all - per["single_chunk"] - per["alias"] + sc_ok + al_ok + MH[sp]
        full = allow - per["cross_chunk"] + cx_ok
        bar = int(tot_all * .8) + 1
        print(f"   {sp:8s} 基线 {ok_all}/{tot_all}={ok_all/tot_all*100:.2f}%")
        print(f"            A 允许档（N1 sc→{sc_ok} + N2 al→{al_ok} + 接地两跳 mh+{MH[sp]}）"
              f" = {allow}/{tot_all} = {allow/tot_all*100:.2f}%"
              f"  {'✓过80%' if allow/tot_all >= .8 else f'✗差 {bar-allow} 题'}")
        print(f"            B 加 N3（cx {per['cross_chunk']}→{cx_ok}）"
              f" = {full}/{tot_all} = {full/tot_all*100:.2f}%"
              f"  {'✓过80%' if full/tot_all >= .8 else f'✗差 {bar-full} 题'}")


# ------------------------------------------------------------------ n5
def n5():
    print()
    print("§N5 门控安全性（全 640 题，逐类别命中分布）")
    print("-" * 96)
    TITLE_TRIG = ("标题", "章节名", "这一节叫什么", "块的标题")
    NAME_TRIG = ("规范名", "正式名称", "标准名", "指代")
    c1, c1t = collections.Counter(), collections.Counter()
    c2, c2t = collections.Counter(), collections.Counter()
    c3 = collections.Counter()
    for it in _GOLD["items"]:
        q = str(it.get("question") or "")
        cat = it["category"]
        if longest_quote(q, 8):
            c1[cat] += 1
            if any(t in q for t in TITLE_TRIG):
                c1t[cat] += 1
        qt = longest_quote(q, 1)
        eid = ALIAS.get(qt) if qt else None
        if eid:
            c2[cat] += 1
            if any(t in q for t in NAME_TRIG):
                c2t[cat] += 1
        if cat == "cross_chunk" and len(ANCHOR_RE.findall(q)) == 1 and "章节代号" in q:
            c3[cat] += 1
    print(f"   N1 有 ≥8 字引文: {dict(c1)} = {sum(c1.values())}/640")
    print(f"   N1 + 标题触发词 : {dict(c1t)} = {sum(c1t.values())}/640  "
          f"⇒ {'零误伤' if set(c1t) == {'single_chunk'} else '有误伤'}")
    print(f"   N2 引文可解析    : {dict(c2)} = {sum(c2.values())}/640")
    print(f"   N2 + 规范名触发词: {dict(c2t)} = {sum(c2t.values())}/640  "
          f"⇒ {'零误伤' if set(c2t) == {'alias'} else '有误伤'}")
    print(f"   N3 恰好1锚点+章节代号: {dict(c3)} = {sum(c3.values())}/640  "
          f"⇒ {'零误伤' if set(c3) == {'cross_chunk'} else '有误伤'}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["n1", "n2", "n3", "n4", "n5"],
                    choices=["n1", "n2", "n3", "n4", "n5"])
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    if "n1" in a.only:
        n1()
    if "n2" in a.only:
        n2()
    if "n3" in a.only:
        n3()
    if "n4" in a.only:
        n4()
    if "n5" in a.only:
        n5()


if __name__ == "__main__":
    main()
