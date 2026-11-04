"""排序改动对下游的真跑验证：不预测，直接跑管线看 gold 是否真进上下文。

A 只测了 top-20 集合扰动（20 题里 8 题变了），机制上推「上下文单元集合也会变」。
但机制推断不等于下游结果。这里直接真跑：把 retriever.search 换成「深池重排后取前 20」，
对比基线（等价生产）与 pair0.3+hop0.2 两种排序下，gold 是否进上下文、引用是否覆盖。

只读：不改代码、不改配置常量、不写 gold。
"""

from __future__ import annotations

import argparse
import collections
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arkrag import pipeline as pipeline_mod, query as query_mod, retrieve, vocab  # noqa: E402


def _git_revision() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


GENERATION = "g-20261103T132152Z-06a741"
TRIAGE = "data/a_triage_all320.jsonl"
POOL = 200
TOP_K = 20

# (名字, w_pair, w_hop, w_cooc)
CONFIGS = [
    ("现状", 0.0, 0.0, 0.0),
    ("pair0.3+hop0.2", 0.3, 0.2, 0.0),
    ("hop0.5", 0.0, 0.5, 0.0),
    ("pair0.3+hop0.5", 0.3, 0.5, 0.0),
    ("cooc0.3", 0.0, 0.0, 0.3),
    ("pair0.3+hop0.2+cooc0.1", 0.3, 0.2, 0.1),
]


def neighbors(adj, eid):
    out = set()
    for key in ("out", "in"):
        for item in (adj.get(eid) or {}).get(key) or []:
            nid = str(item.get("neighbor_id") or "")
            if nid and nid != eid:
                out.add(nid)
    return out


def pair_relations(adj, a, b):
    rels = set()
    for item in (adj.get(a) or {}).get("out") or []:
        if str(item.get("neighbor_id")) == b:
            name = str(item.get("relation") or "")
            if name:
                rels.add(name)
    return rels


def build_signals(retriever, qa, hits):
    """每题算一次 pair/hop/cooc 逐块信号（与 a_ranking_replay 同口径）。"""
    adj = retriever.adjacency
    candidates = qa.get("entity_candidates") or []
    rels = set()
    common_names = []
    endpoint_names = []
    if len(candidates) >= 2:
        a = str(candidates[0].get("entity_id") or "")
        b = str(candidates[1].get("entity_id") or "")
        if a and b:
            rels = pair_relations(adj, a, b)
            common_names = [retriever.display_name(e) for e in (neighbors(adj, a) & neighbors(adj, b))]
            common_names = [x for x in common_names if x]
    for cand in candidates[:2]:
        ent = retriever.entities_by_id.get(str(cand.get("entity_id") or "")) or {}
        nm = ent.get("name") or cand.get("name")
        if nm:
            endpoint_names.append(nm)

    chunk_pred = {}
    for cid in [h["chunk_id"] for h in hits]:
        preds = set()
        for rec in (retriever.assertions_by_chunk.get(cid) or []):
            p = str(rec.get("predicate") or "")
            if p:
                preds.add(p)
        chunk_pred[cid] = preds

    hop, cooc = {}, {}
    for h in hits:
        cid = h["chunk_id"]
        text = str((retriever.chunks.get(cid) or {}).get("text") or "")
        hop_v = cooc_v = 0
        for line in text.splitlines():
            ep = [n for n in endpoint_names if n in line]
            if len(ep) >= 2:
                cooc_v = 1
            if ep and any(m in line for m in common_names):
                hop_v = 1
            if hop_v and cooc_v:
                break
        hop[cid] = hop_v
        cooc[cid] = cooc_v
    return rels, chunk_pred, hop, cooc


def make_patched_search(retriever, original, w_pair, w_hop, w_cooc, state, boost_score):
    """返回一个替换版 search：深池取回后按配置重排，切前 TOP_K 当 hits。

    `boost_score=True` 时把加成写进 hit["score"]。这很关键：`build_context`
    （src/arkrag/retrieve.py:850-853）按 `-u["score"]` 重新排序，只改 hits 顺序
    不改分，上下文顺序仍由原始 BM25 分决定 → 加成对上下文完全无效。
    """

    def patched(qa, top_k=None):
        saved = retrieve.CANDIDATE_POOL_MULT
        retrieve.CANDIDATE_POOL_MULT = 1
        try:
            payload = original(qa, top_k=POOL)
        finally:
            retrieve.CANDIDATE_POOL_MULT = saved
        hits = payload.get("hits") or []
        if not hits:
            return {"hits": [], "stats": payload.get("stats") or {}}
        order = [h["chunk_id"] for h in hits]
        hit_by_id = {h["chunk_id"]: h for h in hits}
        rank = {cid: i + 1 for i, cid in enumerate(order)}
        score = {h["chunk_id"]: float(h["score"]) for h in hits}
        rels, chunk_pred, hop, cooc = build_signals(retriever, qa, hits)
        state["rels"] = rels
        state["order"] = order

        def key(cid):
            return (
                -(score[cid]
                  + w_pair * (1 if (rels & chunk_pred.get(cid, set())) else 0)
                  + w_hop * hop.get(cid, 0)
                  + w_cooc * cooc.get(cid, 0)),
                rank[cid],
            )

        new_order = sorted(order, key=key)
        state["new_order"] = new_order
        state["pair_hits"] = sum(1 for c in order if rels & chunk_pred.get(c, set()))
        state["hop_hits"] = sum(1 for c in order if hop.get(c))
        keep = set(new_order[:TOP_K])
        new_hits = []
        for cid in new_order[:TOP_K]:
            h = hit_by_id[cid]
            h2 = dict(h)
            if boost_score:
                # 把加成折进 score，让 build_context 的内部排序也跟着变
                h2["score"] = score[cid] + w_pair * (1 if (rels & chunk_pred.get(cid, set())) else 0) \
                    + w_hop * hop.get(cid, 0) + w_cooc * cooc.get(cid, 0)
            new_hits.append(h2)
        return {"hits": new_hits, "stats": payload.get("stats") or {}}

    return patched


def gold_in_context(context_units, gold_ids):
    present = {u.get("chunk_id") for u in context_units}
    return {g: (g in present) for g in gold_ids}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-json", default="data/lead_seed_drift.json")
    args = ap.parse_args(argv)

    rows = [json.loads(x) for x in Path(ROOT / TRIAGE).read_text(encoding="utf-8").splitlines() if x.strip()]
    # triage 的 required_citations 只是条数，真正的 required 要从 gold 取
    gold = json.loads(Path(ROOT / "data/goldset/gold-v1.json").read_text(encoding="utf-8"))
    req_by_qid = {it["question_id"]: (it.get("required_citations") or []) for it in gold["items"]}
    # 只取应答题（有 gold_chunk_ids 且不应拒答）
    qs = [r for r in rows if r.get("gold_chunk_ids") and not r.get("should_abstain")]
    for r in qs:
        r["_reqs"] = req_by_qid.get(r["question_id"]) or []
    if args.limit:
        qs = qs[: args.limit]
    print(f"应答题 {len(qs)} 道")

    loaded = pipeline_mod.load_generation(GENERATION)
    base_pipe = pipeline_mod.build_pipeline(loaded, top_k=TOP_K, graph_depth=1)
    retriever = base_pipe["retriever"]

    results = {name: [] for name, _, _, _ in CONFIGS}
    original_search = retriever.search
    for idx, item in enumerate(qs, 1):
        if idx % 50 == 0:
            print(f"  ... {idx}/{len(qs)}", flush=True)
        gold_ids = [str(g) for g in item["gold_chunk_ids"]]
        for name, wp, wh, wc in CONFIGS:
            state = {}
            pipe = dict(base_pipe)
            pipe["retriever"] = retriever
            retriever.search = make_patched_search(retriever, original_search, wp, wh, wc, state, boost_score=True)
            try:
                out = pipeline_mod.ask_once(pipe, item["question"])
            finally:
                retriever.search = original_search
            units = (out.get("context") or {}).get("units") or []
            present = gold_in_context(units, gold_ids)
            # 真进上下文 = 至少一个 gold 块在 units 里
            any_gold = any(present.values())
            all_gold = all(present.values())
            # 引用覆盖（指标真判据：req.evidence in cite.evidence 且 chunk_id 相同）
            reqs = item.get("_reqs") or []
            cites = (out.get("answer") or {}).get("citations") or []
            emitted = {(str(c.get("chunk_id") or ""), str(c.get("evidence") or "")) for c in cites}
            covered = bool(reqs) and all(
                any(cid == str(r.get("chunk_id") or "") and str(r.get("evidence") or "") in ev
                    for cid, ev in emitted)
                for r in reqs
            )
            # 新前 20 里有几个 gold
            new_top20 = set(state.get("order") or []) and None
            results[name].append({
                "question_id": item["question_id"],
                "category": item.get("category"),
                "any_gold": any_gold,
                "all_gold": all_gold,
                "covered": covered,
                "units": len(units),
                "answer_len": len(str((out.get("answer") or {}).get("answer") or "")),
                "abstained": bool((out.get("answer") or {}).get("abstained")),
                "pair_signal_chunks": state.get("pair_hits"),
                "hop_signal_chunks": state.get("hop_hits"),
                "top20_changed": (state.get("new_order") or [])[:TOP_K] != (state.get("order") or [])[:TOP_K],
            })

    # 汇总
    summary = {}
    for name in results:
        rs = results[name]
        n = len(rs)
        summary[name] = {
            "n": n,
            "any_gold": sum(1 for r in rs if r["any_gold"]),
            "all_gold": sum(1 for r in rs if r["all_gold"]),
            "covered": sum(1 for r in rs if r["covered"]),
            "abstained": sum(1 for r in rs if r["abstained"]),
        }
    # 逐题迁移：现状 vs 每个配置
    base = {r["question_id"]: r for r in results["现状"]}
    transitions = {}
    for name in results:
        if name == "现状":
            continue
        gain, lose = [], []
        for r in results[name]:
            b = base.get(r["question_id"])
            if not b:
                continue
            if r["any_gold"] and not b["any_gold"]:
                gain.append(r["question_id"])
            elif b["any_gold"] and not r["any_gold"]:
                lose.append(r["question_id"])
        transitions[name] = {"gain": gain, "lose": lose, "net": len(gain) - len(lose)}

    out = {
        "schema": "arkrag.lead-seed-drift/1",
        "purpose": "验证排序改变是否真的改变 gold 进上下文：真跑 ask_once，不做预测",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_revision(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "command": "py -3 scripts/lead_seed_drift.py",
        "generation": GENERATION,
        "summary": summary,
        "transitions": transitions,
        "per_question": results,
    }
    Path(ROOT / args.out_json).write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print()
    print(f"{'配置':18s} {'n':>4s} {'进ctx':>6s} {'全进':>5s} {'覆盖':>5s} {'拒答':>5s}")
    for name in results:
        s = summary[name]
        print(f"{name:18s} {s['n']:4d} {s['any_gold']:6d} {s['all_gold']:5d} {s['covered']:5d} {s['abstained']:5d}")
    print()
    for name, t in transitions.items():
        print(f"{name:18s} 救回 {len(t['gain']):3d}  丢失 {len(t['lose']):3d}  净 {t['net']:+3d}")
        if t["lose"]:
            print(f"   丢失: {t['lose']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
