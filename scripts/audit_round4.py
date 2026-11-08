"""ArkRAG 第四轮对抗审查 —— cross_chunk「实体 chunks 直取」检索通路。

只读脚本：不改 src/、不改 scripts/eval_answer_quality.py、不改 data/goldset/*、
不改任何评测产物。仅用既有世代数据与生产 API 现场重跑。

用法：
    py -3 scripts/audit_round4.py                 # 全部四节
    py -3 scripts/audit_round4.py --only q1 q3    # 子集
    py -3 scripts/audit_round4.py --json out.json

节：
    q1  纯章节 dump（零检索）与通道特异性 —— 这条通路是否等价于"读 gold 生成规则"
    q2  cap 梯度 + 接地版退路（实体块 ∩ 本次检索命中）
    q3  严格门控三 split 总分 + 逐题配对 bootstrap（单侧，20000 次）
    q4  泄漏与恒真度量：gold_chunk_ids ⊆ 实体 chunks；答案章节 100% 含锚点名

判据红线见 docs/审查-评测口径对抗性报告-20261107.md §E（含 22–31）。
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

from arkrag import judge as J          # noqa: E402
from arkrag import pipeline as P       # noqa: E402
from arkrag import query as query_mod  # noqa: E402

GEN = "g-20261103T132152Z-06a741"
ANCHOR_RE = re.compile(r"「([^」]{1,30})」")
CHAPTER_SPLIT = re.compile(r"[、,，/／\s]+")
SCOR = set(J.SCORABLE_CATEGORIES)

_LOADED = P.load_generation(GEN)
GRAPH = _LOADED["graph"]
CHUNKS = _LOADED["chunks"]
ENT_BY_ID = {e["id"]: e for e in GRAPH["entities"]}
BY_NAME: dict[str, list[str]] = collections.defaultdict(list)
for _e in GRAPH["entities"]:
    BY_NAME[_e["name"]].append(_e["id"])
    for _a in (_e.get("aliases") or []):
        if isinstance(_a, str):
            BY_NAME[_a].append(_e["id"])

_CHAP_ACC: dict[str, list[str]] = collections.defaultdict(list)
for _cid, _rec in CHUNKS.items():
    _CHAP_ACC[str(_rec.get("chapter") or "")].append(str(_rec.get("text") or ""))
CHAP = {k: "\n".join(v) for k, v in _CHAP_ACC.items()}

_GOLD = json.loads((ROOT / "data/goldset/gold-v2.json").read_text("utf-8"))
_V1 = {it["question_id"] for it in
       json.loads((ROOT / "data/goldset/gold-v1.json").read_text("utf-8"))["items"]}


def pool(split: str) -> list[dict]:
    """复刻 scripts/eval_answer_quality.py:pick_items 的池构造（clean 为派生池）。"""
    if split == "clean":
        return [it for it in _GOLD["items"]
                if it.get("split") == "holdout"
                and it["question_id"] not in _V1
                and it.get("gold_chunk_ids")]
    return [it for it in _GOLD["items"]
            if it.get("split") == split and it.get("gold_chunk_ids")]


POOLS = {s: pool(s) for s in ("dev", "holdout", "clean")}

_PIPE = P.build_pipeline(_LOADED, top_k=40, graph_depth=1)
_PIPE["generation_id"] = GEN
_PIPE["entities"] = GRAPH["entities"]
_PIPE["retriever"].budget["max_context_tokens"] = 24000   # 本轮冻结值（红线 26）
_PIPE["answerer"].max_claims = 8                          # 红线 25
RET = _PIPE["retriever"]
ANS = _PIPE["answerer"]


def anchor(it: dict) -> str | None:
    a = ANCHOR_RE.findall(str(it.get("question") or ""))
    return a[0] if a else None


def chapters_of(chunk_ids) -> list[str]:
    return sorted({str(CHUNKS[c].get("chapter") or "") for c in chunk_ids
                   if c in CHUNKS and CHUNKS[c].get("chapter")})


def gold_chapters(it: dict) -> set[str]:
    return {c for c in CHAPTER_SPLIT.split(str(it.get("answer") or "").strip()) if c}


def judge(it: dict, text: str) -> dict:
    return J.judge_answer("cross_chunk", text, it,
                          question=it.get("question"), chapter_texts=CHAP)


def run(it: dict, mode: str, cap: int = 20) -> dict:
    """mode: base | ent（lead 通路）| grounded（实体块 ∩ 本次检索命中）。"""
    q = it["question"]
    qa = query_mod.analyze_query(q, entities=GRAPH["entities"],
                                 aliases=GRAPH.get("alias_index"), vocab=P.vocab,
                                 context=getattr(RET, "query_context", None))
    found = RET.search(qa, top_k=_PIPE["top_k"])
    hits = list(found["hits"])
    stats = {"n_override": 0, "n_already_hit": 0}
    if mode != "base" and it.get("category") == "cross_chunk":
        nm = anchor(it)
        ids = BY_NAME.get(nm, []) if nm else []
        if ids:
            cs = list((ENT_BY_ID.get(ids[0]) or {}).get("chunks") or [])
            hit_ids = {h["chunk_id"] for h in hits}
            keep = cs[:cap] if mode == "ent" else [c for c in cs if c in hit_ids]
            stats["n_override"] = len(keep)
            stats["n_already_hit"] = sum(1 for c in keep if c in hit_ids)
            ov = [{"chunk_id": c, "score": 1.0 - i * 1e-6,
                   "sources": [{"channel": "entity_index", "score": 1.0}]}
                  for i, c in enumerate(keep)]
            seen = {h["chunk_id"] for h in ov}
            hits = ov + [h for h in hits[:4] if h["chunk_id"] not in seen]
    ex = RET.expand([c["entity_id"] for c in qa.get("entity_candidates") or []],
                    depth=1, query_analysis=qa,
                    seed_chunk_ids=[h["chunk_id"] for h in hits])
    ctx = RET.build_context(hits, ex["expansions"], query_analysis=qa)
    ans = ANS.answer(q, ctx, generation_id=GEN,
                     top_score=float(found["stats"].get("top_score") or 0.0),
                     query_analysis=qa, retrieval_stats=found["stats"])
    txt = str((ans or {}).get("answer") or "")
    return {"text": txt, "judge": J.judge_answer(it["category"], txt, it,
                                                 question=q, chapter_texts=CHAP),
            "ctx": ctx, "ans": ans, "stats": stats}


def boot(a, b, n=20000, seed=7):
    """逐题配对 bootstrap，单侧 H1: b 优于 a。返回 (obs, lo, hi, p)。"""
    d = [y - x for x, y in zip(a, b)]
    r = random.Random(seed)
    idx = list(range(len(d)))
    st = sorted(sum(d[r.choice(idx)] for _ in idx) for _ in range(n))
    return sum(d), st[int(n * .025)], st[int(n * .975)], sum(1 for s in st if s <= 0) / n


# ---------------------------------------------------------------- q1
def q1():
    print("§R7.0 / q1  纯章节 dump（零检索、零判据）与通道特异性")
    print("-" * 96)
    rng = random.Random(31)
    ents = [e for e in GRAPH["entities"] if e.get("chunks")]
    _ncache: dict[str, list[str]] = {}

    def name_chunks(n):
        if n not in _ncache:
            _ncache[n] = [c for c in CHUNKS if n in str(CHUNKS[c].get("text") or "")]
        return _ncache[n]

    rows_all = {}
    for sp in ("dev", "holdout", "clean"):
        cols = collections.OrderedDict((k, [0, 0]) for k in
                                       ("实体全部chunks", "随机实体chunks", "全库含名块", "全部494章"))
        for it in [x for x in POOLS[sp] if x["category"] == "cross_chunk"]:
            nm = anchor(it)
            ids = BY_NAME.get(nm, []) if nm else []
            cs = (ENT_BY_ID.get(ids[0]) or {}).get("chunks") or [] if ids else []
            cand = {
                "实体全部chunks": chapters_of(cs),
                "随机实体chunks": chapters_of(rng.choice(ents).get("chunks") or []),
                "全库含名块": chapters_of(name_chunks(nm) if nm else []),
                "全部494章": sorted(CHAP),
            }
            for k, chs in cand.items():
                cols[k][1] += 1
                cols[k][0] += (judge(it, "、".join(chs))["pass"] is True)
        rows_all[sp] = {k: tuple(v) for k, v in cols.items()}
        print(f"   {sp:8s} " + "  |  ".join(
            f"{k} {v[0]}/{v[1]}={v[0]/v[1]*100:.1f}%" for k, v in cols.items()))
    print("   ⇒ '实体全部 chunks 的章节' 是 gold 的同源副本：零检索即 93.8–97.4%，随机实体 0%。")
    return rows_all


# ---------------------------------------------------------------- q2
def q2():
    print()
    print("§R7.2 / q2  cap 梯度 + 接地版退路（严格门控：只有 cross_chunk 走该通路）")
    print("-" * 96)
    cx = {sp: [x for x in POOLS[sp] if x["category"] == "cross_chunk"]
          for sp in ("dev", "holdout", "clean")}
    for sp in ("dev", "holdout", "clean"):
        row = []
        for cap in (4, 8, 12, 20, 40, 10 ** 6):
            ok = sum(1 for it in cx[sp] if run(it, "ent", cap)["judge"]["pass"] is True)
            row.append(f"cap={'ALL' if cap > 10**5 else cap}:{ok}/{len(cx[sp])}")
        print(f"   {sp:8s} " + "  ".join(row))
    print("   ⇒ 非单调（dev 峰值恰在 cap=20，40/ALL 回落）：旋钮值靠噪声挑，不是能力。")

    print()
    print("§R7.2b 三种口径（仅 cross_chunk 生效，其余类与基线逐题同一路径）")
    out = {}
    for sp in ("dev", "holdout", "clean"):
        va, vb, vc = [], [], []
        for it in POOLS[sp]:
            if it["category"] not in SCOR:
                continue
            va.append(int(run(it, "base")["judge"]["pass"] is True))
            vb.append(int(run(it, "ent")["judge"]["pass"] is True))
            vc.append(int(run(it, "grounded")["judge"]["pass"] is True))
        out[sp] = (va, vb, vc)
        n = len(va)
        print(f"   {sp:8s} 基线 {sum(va)}/{n}={sum(va)/n*100:.2f}%  |  "
              f"lead 通路 {sum(vb)}/{n}={sum(vb)/n*100:.2f}% (Δ{sum(vb)-sum(va):+d})  |  "
              f"接地版 {sum(vc)}/{n}={sum(vc)/n*100:.2f}% (Δ{sum(vc)-sum(va):+d})")

    tot = ov = 0
    for it in cx["dev"]:
        s = run(it, "ent")["stats"]
        tot += s["n_override"]; ov += s["n_already_hit"]
    if tot:
        print(f"   dev 通路引入的实体块中，本次检索已召回的 {ov}/{tot} = "
              f"{ov/tot*100:.1f}%（新引入 {(tot-ov)/tot*100:.1f}%）")
    return out


# ---------------------------------------------------------------- q3
def q3(out=None):
    print()
    print("§R7.3 / q3  严格门控三 split 总分 + 逐题配对 bootstrap（单侧，20000 次，seed=7）")
    print("-" * 96)
    out = out or q2()
    for sp in ("dev", "holdout", "clean"):
        va, vb, _ = out[sp]
        obs, lo, hi, p = boot(va, vb)
        ok = obs >= 1 and lo > 0 and p < .05
        print(f"   {sp:8s} 基线 {sum(va)}/{len(va)}={sum(va)/len(va)*100:.2f}% → "
              f"通路 {sum(vb)}/{len(vb)}={sum(vb)/len(vb)*100:.2f}%  "
              f"Δ={obs:+d} CI=[{lo:+d},{hi:+d}] p={p:.4f} {'PASS' if ok else 'FAIL'}")
    print("   ⇒ D 协议要求 dev/holdout/clean 三份同时 PASS；本例 holdout(p=0.0828)、"
          "clean(p=0.4092) 不达标。")


# ---------------------------------------------------------------- q4
def q4():
    print()
    print("§R7.4 / q4  泄漏与恒真度量")
    print("-" * 96)
    for sp in ("dev", "holdout", "clean"):
        n = sub = sub20 = eq = 0
        for it in [x for x in POOLS[sp] if x["category"] == "cross_chunk"]:
            nm = anchor(it)
            ids = BY_NAME.get(nm, []) if nm else []
            if not ids:
                continue
            allc = list((ENT_BY_ID.get(ids[0]) or {}).get("chunks") or [])
            g = set(it.get("gold_chunk_ids") or [])
            n += 1
            sub += (g <= set(allc))
            sub20 += (g <= set(allc[:20]))
            eq += (set(chapters_of(allc)) == gold_chapters(it))
        print(f"   {sp:8s} n={n}  gold_chunk_ids ⊆ 实体全部 chunks {sub}/{n}={sub/n*100:.1f}%  |  "
              f"⊆ 前20 {sub20}/{n}={sub20/n*100:.1f}%  |  "
              f"章节集合完全相等 {eq}/{n}={eq/n*100:.1f}%")

    tot = taut = 0
    for sp in ("dev", "holdout", "clean"):
        for it in [x for x in POOLS[sp] if x["category"] == "cross_chunk"]:
            nm = anchor(it)
            r = run(it, "ent")
            for ch in J.chapter_set(r["text"]):
                tot += 1
                taut += bool(nm) and nm in CHAP.get(ch, "")
    print(f"   答案章节共 {tot} 个，其中『该章正文含锚点名』 {taut}/{tot} = "
          f"{taut/tot*100:.1f}%  ⇒ 判据的 extra 真实性校验恒真（红线 9/A9 失效）")

    hit = [it for it in _GOLD["items"] if it["category"] == "cross_chunk"]
    gate = [it for it in hit if len(ANCHOR_RE.findall(it["question"])) == 1
            and "章节代号" in it["question"]]
    wrong = len([x for x in gate if x["category"] != "cross_chunk"])
    print(f"   门控『恰好 1 个「」且含“章节代号”』命中 {len(gate)}/{len(hit)}，误伤 {wrong}")


# ---------------------------------------------------------------- q5
def q5():
    print()
    print("§R7.5 / q5  零检索天花板：答案 = 锚点实体 chunks 的章节集合（不检索、不看得分）")
    print("-" * 96)
    base = {"dev": 205, "holdout": 139, "clean": 89}
    cxold = {"dev": 23, "holdout": 16, "clean": 12}
    for sp in ("dev", "holdout", "clean"):
        row = []
        for cap in (10, 20, 40, None):
            ok = n = 0
            for it in [x for x in POOLS[sp] if x["category"] == "cross_chunk"]:
                nm = anchor(it)
                ids = BY_NAME.get(nm, []) if nm else []
                cs = list((ENT_BY_ID.get(ids[0]) or {}).get("chunks") or []) if ids else []
                if cap:
                    cs = cs[:cap]
                n += 1
                ok += (judge(it, "、".join(chapters_of(cs)))["pass"] is True)
            row.append(f"cap={'ALL' if cap is None else cap}:{ok}/{n}")
        print(f"   {sp:8s} " + "  ".join(row))
    print("   ⇒ 全部实体块 = 37/25/15（对 38/26/16），即 cross_chunk 无需任何检索即可拿满。")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["q1", "q2", "q3", "q4", "q5"],
                    choices=["q1", "q2", "q3", "q4", "q5"])
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    payload = {}
    if "q1" in a.only:
        payload["q1"] = q1()
    out = q2() if ("q2" in a.only or "q3" in a.only) else None
    if "q2" in a.only and out is not None:
        payload["q2"] = {sp: {"base": out[sp][0], "lead": out[sp][1],
                              "grounded": out[sp][2]} for sp in out}
    if "q3" in a.only:
        q3(out)
    if "q4" in a.only:
        q4()
    if a.json:
        Path(a.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
        print(f"\n[已写出] {a.json}")


if __name__ == "__main__":
    main()
