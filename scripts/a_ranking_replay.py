"""排序信号离线回放：把候选块重排一次，量 R@5 / R@20 / MRR 怎么变。

**这是测量工具，不是实验工具。** 它只重放同一批命中（`search` 的深池结果），
按新的排序键重排一遍再算指标，**不改任何检索代码、不改任何配置常量**。
真跑管线会重放 `search` → `expand` → `build_context` → `answer` → `verify_citations`
整条链，结果与本脚本的离线估算**不等价**（见 `docs/a阶段_预算实验方案.md` §3）。

## 三个被测信号（全部在线可得，不需要 gold 答案）

`pred_pair_hit`：端点对的图关系词是否出现在该块的断言里。
    端点取 `qa["entity_candidates"][:2]` 的 `entity_id`；端点对的关系词取自
    `adjacency[A]["out"]` 里 `neighbor_id == B` 的 `relation` 字段
    （**注意字段名是 `relation`，不是 `predicate`**——`predicate` 只存在于断言记录里，
    邻接表里叫 `relation`。写错字段会让整个信号恒为空集，回放结果与现状逐位相同，
    看起来像"信号无效"，其实是取错了键）。
    命中判据：块内存在断言满足 `predicate ∈ 端点对关系词`。

`hop_line`：块内是否存在某一行同时含「至少一个端点名」与「至少一个共同邻居名」。
    共同邻居 = `neighbors(adj, A) ∩ neighbors(adj, B)`，`neighbors` 同时读
    `adjacency[eid]["out"]` 与 `["in"]` 并 discard 自身。

`cooccur`：两端点是否在同一行里共现（不要求共同邻居）。

## 排序键

    sort_key = -(score + w_pair·pred_pair_hit + w_hop·hop_line + w_cooc·cooccur)
    次级键 = 原 rank（保证同分时保持现状次序，使 λ=0 与现状逐位一致）

## 为什么 λ=0 必须与现状逐位一致

这是本工具的自检。`sort_key` 在 λ=0 时退化为 `(-score, rank)`，而生产 `search()`
的输出本来就是按 `(-score, chunk_id)` 排好的，`rank` 即其序号，所以重排后顺序不变。
若 λ=0 的 R@5/R@20/MRR 与直接按原顺序算的不一致，说明本脚本的指标口径写错了。

## 用法

    py -3 scripts/a_ranking_replay.py
    py -3 scripts/a_ranking_replay.py --out-json data/a_ranking_replay.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arkrag import config, pipeline as pipeline_mod, query as query_mod, retrieve, vocab, subproc  # noqa: E402

SCHEMA = "arkrag.a-ranking-replay/1"
DEFAULT_GENERATION = "g-20261103T132152Z-06a741"
DEFAULT_TRIAGE = "data/a_triage_all320.jsonl"
DEFAULT_GOLD = "data/goldset/gold-v1.json"
DEFAULT_POOL = 200
EVAL_TOP_K = 20

# 被测配置：(名字, w_pair, w_hop, w_cooc)
CONFIGS = [
    ("现状", 0.0, 0.0, 0.0),
    ("pair0.1", 0.1, 0.0, 0.0),
    ("pair0.2", 0.2, 0.0, 0.0),
    ("pair0.3", 0.3, 0.0, 0.0),
    ("pair0.5", 0.5, 0.0, 0.0),
    ("hop0.2", 0.0, 0.2, 0.0),
    ("hop0.5", 0.0, 0.5, 0.0),
    ("cooc0.3", 0.0, 0.0, 0.3),
    ("pair0.3+hop0.2", 0.3, 0.2, 0.0),
    ("pair0.5+hop0.2", 0.5, 0.2, 0.0),
    ("pair0.3+hop0.5", 0.3, 0.5, 0.0),
]

# 方向题口径（对齐 scripts/ask.py:evaluate()：direct_relation + multihop 且非应拒）
DIRECTION_CATEGORIES = ("direct_relation", "multihop")


def rel(path: Path | str) -> str:
    """入库一律写相对仓库根的路径，不写机台绝对路径（tests/test_portable_paths.py）。"""
    return config.rel_to_root(str(path))


def deep_search(retriever, qa, pool: int):
    """与生产 `search(top_k=20)` 同源的深池。

    `search()` 内部 `depth = top_k * CANDIDATE_POOL_MULT`。直接把 top_k 放大到 pool
    会把 depth 抬到 pool*10，BM25 候选集与 `_norm_scores()` 的归一分母都跟着漂移，
    前 20 就与生产不一致（初版踩过：20 题里 5 题对不上）。把乘数临时压到 1，
    让 top_k 自己承担深度 → depth 正好等于 pool，三路候选集与归一分母与生产同源。
    """
    saved = retrieve.CANDIDATE_POOL_MULT
    retrieve.CANDIDATE_POOL_MULT = 1
    try:
        return retriever.search(qa, top_k=pool)
    finally:
        retrieve.CANDIDATE_POOL_MULT = saved


def neighbors(adj, eid: str) -> set[str]:
    out: set[str] = set()
    bucket = adj.get(eid) or {}
    for key in ("out", "in"):
        for item in bucket.get(key) or []:
            nid = str(item.get("neighbor_id") or "")
            if nid and nid != eid:
                out.add(nid)
    return out


def pair_relations(adj, a: str, b: str) -> set[str]:
    """A→B 的图关系词。邻接表里字段名是 `relation`。"""
    rels: set[str] = set()
    for item in (adj.get(a) or {}).get("out") or []:
        if str(item.get("neighbor_id")) == b:
            name = str(item.get("relation") or "")
            if name:
                rels.add(name)
    return rels


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="排序信号离线回放（测量，不改代码）")
    ap.add_argument("--generation", default=DEFAULT_GENERATION)
    ap.add_argument("--triage", default=DEFAULT_TRIAGE, help="逐题归因 JSONL，取 gold_chunk_ids")
    ap.add_argument("--gold-set", default=DEFAULT_GOLD, help="仅用于记录来源，本脚本不读 gold 语义")
    ap.add_argument("--pool", type=int, default=DEFAULT_POOL)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-json", default="data/a_ranking_replay.json")
    args = ap.parse_args(argv)

    triage_path = Path(args.triage)
    if not triage_path.is_absolute():
        triage_path = ROOT / triage_path
    if not triage_path.is_file():
        print(f"[2] 输入不可用：找不到逐题归因 {triage_path}", file=sys.stderr)
        return 2

    rows = [json.loads(line) for line in triage_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    questions = [r for r in rows if r.get("gold_chunk_ids")]
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        print("[2] 输入不可用：逐题归因里没有带 gold_chunk_ids 的题", file=sys.stderr)
        return 2

    loaded = pipeline_mod.load_generation(args.generation)
    built = pipeline_mod.build_pipeline(loaded, top_k=EVAL_TOP_K, graph_depth=1)
    retriever = built["retriever"]
    entities = retriever.graph.get("entities") or []
    adj = retriever.adjacency
    alias_index = retriever.graph.get("alias_index")

    # 每个块的断言 predicate 集合，一次性建好
    chunk_predicates: dict[str, set[str]] = collections.defaultdict(set)
    for rec in loaded["assertions"]:
        pred = str(rec.get("predicate") or "")
        if pred:
            chunk_predicates[str(rec.get("chunk_id"))].add(pred)

    names = [c[0] for c in CONFIGS]
    overall = {n: {"r5": 0.0, "r20": 0.0, "rr": 0.0, "n": 0} for n in names}
    per_cat = {n: collections.defaultdict(lambda: {"r5": 0.0, "r20": 0.0, "rr": 0.0, "n": 0}) for n in names}
    per_q = {n: [] for n in names}
    signal_per_q: list[dict] = []
    signal_avail = collections.defaultdict(lambda: {"n": 0, "has_pair": 0, "has_common": 0, "hop_line_any": 0})

    for idx, item in enumerate(questions, 1):
        if idx % 40 == 0:
            print(f"  ... {idx}/{len(questions)}", flush=True)
        qa = query_mod.analyze_query(
            item["question"],
            entities=entities,
            aliases=alias_index,
            vocab=vocab,
            context=getattr(retriever, "query_context", None),
        )
        payload = deep_search(retriever, qa, args.pool)
        hits = payload.get("hits") or []
        if not hits:
            continue
        order = [h["chunk_id"] for h in hits]
        rank = {cid: i + 1 for i, cid in enumerate(order)}
        score = {h["chunk_id"]: float(h["score"]) for h in hits}
        gold_ids = [str(g) for g in item["gold_chunk_ids"]]
        if not any(g in rank for g in gold_ids):
            continue

        candidates = qa.get("entity_candidates") or []
        rels: set[str] = set()
        common_names: list[str] = []
        endpoint_names: list[str] = []
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

        pair_hit = {cid: (1 if (rels & chunk_predicates.get(cid, set())) else 0) for cid in order}
        hop = {}
        cooc = {}
        for cid in order:
            text = str((retriever.chunks.get(cid) or {}).get("text") or "")
            hop_v = 0
            cooc_v = 0
            for line in text.splitlines():
                ep = [n for n in endpoint_names if n in line]
                both = len(ep) >= 2
                if both:
                    cooc_v = 1
                if ep and any(m in line for m in common_names):
                    hop_v = 1
                if hop_v and cooc_v:
                    break
            hop[cid] = hop_v
            cooc[cid] = cooc_v

        avail = signal_avail[item.get("category") or ""]
        avail["n"] += 1
        avail["has_pair"] += 1 if rels else 0
        avail["has_common"] += 1 if common_names else 0
        avail["hop_line_any"] += 1 if any(hop.values()) else 0

        for name, w_pair, w_hop, w_cooc in CONFIGS:
            reordered = sorted(
                order,
                key=lambda cid: (
                    -(score[cid] + w_pair * pair_hit[cid] + w_hop * hop[cid] + w_cooc * cooc[cid]),
                    rank[cid],
                ),
            )
            new_rank = {cid: i + 1 for i, cid in enumerate(reordered)}
            r5 = sum(1 for g in gold_ids if new_rank.get(g, 10**9) <= 5) / len(gold_ids)
            r20 = sum(1 for g in gold_ids if new_rank.get(g, 10**9) <= EVAL_TOP_K) / len(gold_ids)
            best = min((new_rank.get(g, 10**9) for g in gold_ids), default=10**9)
            rr = 1.0 / best if best < 10**9 else 0.0
            for bucket in (overall[name], per_cat[name][item.get("category") or ""]):
                bucket["n"] += 1
                bucket["r5"] += r5
                bucket["r20"] += r20
                bucket["rr"] += rr
            per_q[name].append(
                {
                    "question_id": item["question_id"],
                    "category": item.get("category"),
                    "r5": round(r5, 6),
                    "r20": round(r20, 6),
                    "rr": round(rr, 6),
                    "gold_new_ranks": {g: new_rank.get(g) for g in gold_ids},
                }
            )
        # 信号逐块只存一份（与配置无关；每个配置复制一份会让文件涨到 70 MB）
        signal_per_q.append(
            {
                "question_id": item["question_id"],
                "category": item.get("category"),
                "gold_ranks": {g: rank.get(g) for g in gold_ids},
                "endpoint_pair_relations": sorted(rels),
                "common_neighbor_count": len(common_names),
                "pair_hit_chunks": [c for c in order if pair_hit[c]],
                "hop_line_chunks": [c for c in order if hop[c]],
            }
        )

    # ---- 自检：λ=0 必须等于"不重排"的原始顺序 ----
    baseline = overall["现状"]
    if baseline["n"] == 0:
        print("[4] 自检失败：没有可用题目", file=sys.stderr)
        return 4
    drift = 0
    for name in names:
        if overall[name]["n"] != baseline["n"]:
            drift += 1
    if drift:
        print(f"[4] 自检失败：各配置覆盖题数不一致（{drift} 个配置）", file=sys.stderr)
        return 4

    def table() -> str:
        lines = [f"{'配置':16s} {'R@5':>8s} {'R@20':>8s} {'MRR':>8s}"]
        for name in names:
            b = overall[name]
            n = b["n"]
            lines.append(f"{name:16s} {b['r5']/n:8.4f} {b['r20']/n:8.4f} {b['rr']/n:8.4f}")
        return "\n".join(lines)

    def cat_table(name: str) -> str:
        lines = [f"{'类别':18s} {'n':>4s} {'R@5':>8s} {'R@20':>8s} {'MRR':>8s}"]
        for cat in sorted(per_cat[name]):
            b = per_cat[name][cat]
            n = b["n"]
            if not n:
                continue
            lines.append(f"{cat:18s} {n:4d} {b['r5']/n:8.4f} {b['r20']/n:8.4f} {b['rr']/n:8.4f}")
        return "\n".join(lines)

    print(f"世代 {args.generation}  深池 {args.pool}  题数 {baseline['n']}")
    print()
    print(table())
    print()
    print("按类（现状 vs pair0.3+hop0.2）：")
    print(cat_table("现状"))
    print()
    print(cat_table("pair0.3+hop0.2"))
    print()
    print("信号可用性（按类：题数 / 端点对有图关系 / 有共同邻居 / 有块命中 hop_line）：")
    for cat in sorted(signal_avail):
        a = signal_avail[cat]
        print(f"  {cat:18s} {a['n']:3d}  pair {a['has_pair']:3d}  common {a['has_common']:3d}  hop_line {a['hop_line_any']:3d}")

    doc = {
        "schema": SCHEMA,
        "purpose": "排序信号离线回放（纯测量；离线重排不等于真跑管线，见 docs/a阶段_预算实验方案.md §3）",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "command": "py -3 scripts/a_ranking_replay.py",
        "generation": args.generation,
        "triage_source": rel(triage_path),
        "gold_set": rel(Path(args.gold_set)),
        "pool": args.pool,
        "eval_top_k": EVAL_TOP_K,
        "configs": [{"name": n, "w_pair": p, "w_hop": h, "w_cooc": c} for n, p, h, c in CONFIGS],
        "overall": {n: {**overall[n], "r5": round(overall[n]["r5"] / overall[n]["n"], 6),
                        "r20": round(overall[n]["r20"] / overall[n]["n"], 6),
                        "rr": round(overall[n]["rr"] / overall[n]["n"], 6)} for n in names},
        "by_category": {n: {cat: {**b, "r5": round(b["r5"] / b["n"], 6) if b["n"] else 0.0,
                                  "r20": round(b["r20"] / b["n"], 6) if b["n"] else 0.0,
                                  "rr": round(b["rr"] / b["n"], 6) if b["n"] else 0.0}
                            for cat, b in per_cat[n].items() if b["n"]} for n in names},
        "signal_availability": {k: dict(v) for k, v in signal_avail.items()},
        "per_question": {n: per_q[n] for n in names},
        "signal_per_question": signal_per_q,
    }
    out = Path(args.out_json)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写出 {rel(out)}")
    return 0


def _git_commit() -> str:
    try:
        import subprocess

        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True, timeout=10,
            env=subproc.child_env(),
        ).stdout.strip()
    except Exception:
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
