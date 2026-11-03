"""拆解 speaker 加成的两处改动各贡献了什么。

背景：`test_retrieve_speaker.py` 发现 `_speaker_boost` 的返回值能到 2.0，
而 docstring 声称 `(0,1]`。修法是「按实体去重 + 夹到 1.0」，但改完生产指标
R@20 从 0.7019 掉到 0.6919。**必须分清这两处改动各自的影响**，否则无法判断
"原先的 +0.0177 里有多少是真信号、有多少是分数无上界造成的"。

三种实现：
  A `orig`  ：按别名累加，无夹紧（评测出 0.7019 的那一版）
  B `dedup` ：按实体去重，无夹紧（上界 = 命中实体数）
  C `clamp` ：按实体去重 + 夹到 1.0（契约版）

对每种实现扫 ws 0 / 0.1 / 0.2 / 0.3 / 0.5 / 1.0，各报 R@5 / R@20 / MRR。
ws=0 是共同基线（加成关闭），用于计算 Δ。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arkrag import (  # noqa: E402
    config,
    index as index_mod,
    query as query_mod,
    release,
    retrieve,
    vocab as vocab_mod,
)

ENT = retrieve.ENTITY_CHANNEL_WEIGHT


def speaker_set(rec):
    return retrieve._speaker_set(rec)


def boost_orig(ret, cands, allowed):
    """原实现：按别名累加，无夹紧。"""
    if not cands or not allowed:
        return {}
    names = {}
    for cand in cands:
        eid = str(cand.get("entity_id") or "")
        ent = ret.entities_by_id.get(eid) or {}
        terms = [str(ent.get("name") or cand.get("name") or "")]
        for item in ent.get("aliases") or []:
            t = item.get("text") if isinstance(item, dict) else item
            if str(t or "").strip():
                terms.append(str(t).strip())
        w = float(cand.get("score") or 0.0)
        for term in terms:
            if term:
                names[term] = max(names.get(term, 0.0), w)
    if not names:
        return {}
    strongest = max(names.values()) or 1.0
    best = {}
    for cid in allowed:
        sp = speaker_set(ret.chunks.get(cid) or {})
        if not sp:
            continue
        hit = sum(w for term, w in names.items() if term in sp)
        if hit > 0:
            best[cid] = hit / strongest
    return best


def boost_dedup(ret, cands, allowed, *, clamp):
    """按实体去重；`clamp` 控制是否夹到 1.0。"""
    if not cands or not allowed:
        return {}
    by_entity = {}
    for cand in cands:
        eid = str(cand.get("entity_id") or "")
        ent = ret.entities_by_id.get(eid) or {}
        terms = {str(ent.get("name") or cand.get("name") or "").strip()}
        for item in ent.get("aliases") or []:
            t = item.get("text") if isinstance(item, dict) else item
            if str(t or "").strip():
                terms.add(str(t).strip())
        terms.discard("")
        if not terms:
            continue
        w = float(cand.get("score") or 0.0)
        prev = by_entity.get(eid)
        if prev is None or w > prev[1]:
            by_entity[eid] = (terms | (prev[0] if prev else set()), w)
        else:
            prev[0].update(terms)
    if not by_entity:
        return {}
    strongest = max(w for _t, w in by_entity.values()) or 1.0
    best = {}
    for cid in allowed:
        sp = speaker_set(ret.chunks.get(cid) or {})
        if not sp:
            continue
        hit = sum(w for terms, w in by_entity.values() if terms & sp)
        if hit > 0:
            best[cid] = min(1.0, hit / strongest) if clamp else hit / strongest
    return best


def search_with(ret, qa, *, top_k, ws, booster):
    depth = top_k * retrieve.CANDIDATE_POOL_MULT
    if not qa.get("should_answer", True):
        return []
    text = str(qa.get("retrieval_query") or qa.get("normalized_query") or "")
    bm25 = retrieve._norm_scores(ret.index.search(text, top_k=depth)) if text else {}
    entity = {}
    for cand in qa.get("entity_candidates") or []:
        eid = str(cand.get("entity_id") or "")
        ent = ret.entities_by_id.get(eid) or {}
        terms = [str(ent.get("name") or cand.get("name") or "")]
        for item in ent.get("aliases") or []:
            t = item.get("text") if isinstance(item, dict) else item
            if str(t or "").strip():
                terms.append(str(t).strip())
        probe = " ".join(sorted({t for t in terms if t}))
        if not probe:
            continue
        w = float(cand.get("score") or 0.0)
        for cid, sc in retrieve._norm_scores(ret.index.search(probe, top_k=depth)).items():
            entity[cid] = max(entity.get(cid, 0.0), sc * w)
    allowed = set(bm25) | set(entity)
    sp = booster(ret, qa.get("entity_candidates") or (), allowed) if ws else {}
    total = {}
    for cid, sc in bm25.items():
        total[cid] = total.get(cid, 0.0) + sc
    for cid, sc in entity.items():
        total[cid] = total.get(cid, 0.0) + ENT * sc
    for cid, sc in sp.items():
        total[cid] = total.get(cid, 0.0) + ws * sc
    ranked = sorted(total.items(), key=lambda kv: (-kv[1], kv[0]))
    return [cid for cid, _ in ranked[:top_k]]


def metrics(hits_all, qs):
    r5 = r20 = rr = 0.0
    n = 0
    for hits, it in zip(hits_all, qs):
        g = [str(c) for c in it.get("gold_chunk_ids") or []]
        if not g:
            continue
        gs = set(g)
        n += 1
        r5 += len(gs & set(hits[:5])) / len(gs)
        r20 += len(gs & set(hits[:20])) / len(gs)
        f = next((i for i, c in enumerate(hits) if c in gs), None)
        rr += 0.0 if f is None else 1.0 / (f + 1)
    return round(r5 / n, 4), round(r20 / n, 4), round(rr / n, 4)


def main() -> int:
    gen = release.Generation("g-20261031T035231Z-4eafae")
    chunks = {}
    with Path(config.chunks_path()).open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("chunk_id"):
                chunks[str(r["chunk_id"])] = r
    index, _ = index_mod.load_index(gen.index_path)
    graph = json.loads(gen.graph_path.read_text(encoding="utf-8"))
    rt = retrieve.Retriever(index, graph, {}, chunks=chunks, vocab=vocab_mod)
    gold = json.loads(Path("data/goldset/gold-v1.json").read_text(encoding="utf-8"))
    items = gold["items"] if isinstance(gold, dict) else gold
    spec = importlib.util.spec_from_file_location("_ask", Path("scripts/ask.py"))
    ask = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ask)
    qs = ask._stratified_sample(items, 320)
    analyses = [
        query_mod.analyze_query(
            str(it.get("question") or ""),
            entities=graph.get("entities") or [],
            aliases=graph.get("alias_index"),
            vocab=vocab_mod,
            context=rt.query_context,
        )
        for it in qs
    ]

    variants = [
        ("orig  (按别名累加,无夹紧)", boost_orig),
        ("dedup (按实体去重,无夹紧)", lambda r, c, a: boost_dedup(r, c, a, clamp=False)),
        ("clamp (按实体去重+夹1.0)", lambda r, c, a: boost_dedup(r, c, a, clamp=True)),
    ]
    WS = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)

    print(f"{'实现':<28} {'ws':>5} {'R@5':>8} {'R@20':>8} {'MRR':>8}  {'dR@20':>8} {'dMRR':>8}")
    print("-" * 84)
    for name, booster in variants:
        base = None
        for ws in WS:
            hits = [search_with(rt, a, top_k=20, ws=ws, booster=booster) for a in analyses]
            m = metrics(hits, qs)
            if ws == 0.0:
                base = m
            d20 = m[1] - base[1]
            dm = m[2] - base[2]
            tag = ""
            if ws == 0.1:
                tag = "  <-- 生产值"
            print(f"{name:<28} {ws:>5.2f} {m[0]:>8.4f} {m[1]:>8.4f} {m[2]:>8.4f}  "
                  f"{d20:>+8.4f} {dm:>+8.4f}{tag}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
