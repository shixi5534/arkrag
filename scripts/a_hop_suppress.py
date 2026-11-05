"""hop_line 的共同邻居抑制：先测抑制，再谈权重（只测量，不改生产代码）。

## 为什么要这个脚本

`hop_line`（块内某行同时含端点规范名与某共同邻居名）单独用时在回放里
R@20 全场第二高，但 Lead 的真跑验证（`scripts/lead_seed_drift.py`）显示它
`hop0.5` 进上下文 162→162、引用覆盖 51→48，是洗牌不是增益。

根因是枢纽实体：阿米娅、罗德岛这类实体的共同邻居动辄几十个，
`hop_line` 会给几乎全部候选同时抬分，权重加多少都只是洗牌。
所以**抑制是前置条件，不是下一轮的待办**。

## 本脚本做两件事

1. `--stage dist`：只统计每题的共同邻居数 `cn` 与 `hop_line` 命中块数分布，
   回答「噪声有多大」。纯回放，秒级。
2. `--stage real`：对每组（权重, 抑制函数）真跑 `ask_once`，量
   **真进上下文题数 / 全部 gold 进 / 引用覆盖 / 救回 / 丢失**，按类分列。

## 两条纪律（本脚本刻意遵守）

* **加成必须折进 `hit["score"]`**。`src/arkrag/retrieve.py:850-853` 的
  `build_context` 只读 `-u["score"]`，**不读 hits 顺序**。只改顺序不改分，
  加成对上下文零影响，会得到「R@5/R@20 涨、下游一点不动」的假增益
  （Lead 第一版补丁就撞在这上面，四个配置逐位相同）。
* **不预测，真跑。** 名次→上下文的一阶模型看不见「进了之后引用发不发得出来」，
  所以验收一律用真跑数。回放只作诊断量，报告里必须标明。

## 口径

* 调参集 = `data/goldset/gold-v2.json` 里 `split == "dev"` 的 380 题，
  其中应答题（有 `gold_chunk_ids` 且 `should_abstain` 为假）350 题。
  验收集 260 题一律不碰。
* 引用覆盖判据与 `scripts/ask.py:283-298` 一致：
  `req.evidence in cite.evidence` 且 `chunk_id` 相同，`cites` 取自 `answer["citations"]`。
* 相对仓库根的路径一律走 `config.rel_to_root()`（`tests/test_portable_paths.py`）。

只读：不改进程外的任何文件，不写 gold，不动配置常量。
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from arkrag import config, pipeline as pipeline_mod, query as query_mod, retrieve, vocab, subproc  # noqa: E402

DEFAULT_GENERATION = "g-20261103T132152Z-06a741"
DEFAULT_GOLD_V2 = "data/goldset/gold-v2.json"
DEFAULT_POOL = 200
EVAL_TOP_K = 20


# ---------------------------------------------------------------- 抑制函数
# 输入 cn（两端点共同邻居数），返回 hop_line 加成的乘子。
# hop_line 只会在一题有 ≥1 个共同邻居时才可能命中，所以 cn >= 1 时才被调用；
# 其余情况乘子不参与计算。
def _none(cn: int) -> float:
    return 1.0


def _log1p(cn: int) -> float:
    """Lead 与我都提过的形式。注意 cn=1 时 1/log(2)=1.442，反而放大。"""
    return 1.0 / math.log(1 + cn)


def _log2p(cn: int) -> float:
    """+2 的偏移让 cn=1 时不放大（1/log3=0.910）。"""
    return 1.0 / math.log(2 + cn)


def _inv(cn: int) -> float:
    return 1.0 / (1 + cn)


def _sqrt(cn: int) -> float:
    return 1.0 / math.sqrt(1 + cn)


def _gate10(cn: int) -> float:
    return 1.0 if cn <= 10 else 0.0


def _gate5(cn: int) -> float:
    return 1.0 if cn <= 5 else 0.0


def _cap10(cn: int) -> float:
    return min(1.0, 10.0 / cn)


SUPPRESS = {
    "none": _none,
    "log1p": _log1p,
    "log2p": _log2p,
    "inv": _inv,
    "sqrt": _sqrt,
    "gate10": _gate10,
    "gate5": _gate5,
    "cap10": _cap10,
}


# (名字, w_pair, w_hop, w_cooc, 抑制函数名)
CONFIGS: list[tuple[str, float, float, float, str]] = [
    ("基线(不打补丁)", 0.0, 0.0, 0.0, "none"),
    ("基线(打补丁零权重)", 0.0, 0.0, 0.0, "none"),
    ("pair0.3", 0.3, 0.0, 0.0, "none"),
    ("pair0.3+hop0.2|none", 0.3, 0.2, 0.0, "none"),
    ("pair0.3+hop0.2|log1p", 0.3, 0.2, 0.0, "log1p"),
    ("pair0.3+hop0.2|log2p", 0.3, 0.2, 0.0, "log2p"),
    ("pair0.3+hop0.2|inv", 0.3, 0.2, 0.0, "inv"),
    ("pair0.3+hop0.2|sqrt", 0.3, 0.2, 0.0, "sqrt"),
    ("pair0.3+hop0.2|gate10", 0.3, 0.2, 0.0, "gate10"),
    ("pair0.3+hop0.2|cap10", 0.3, 0.2, 0.0, "cap10"),
    ("pair0.3+hop0.5|none", 0.3, 0.5, 0.0, "none"),
    ("pair0.3+hop0.5|log2p", 0.3, 0.5, 0.0, "log2p"),
    ("pair0.3+hop0.5|inv", 0.3, 0.5, 0.0, "inv"),
    ("pair0.3+hop0.5|gate10", 0.3, 0.5, 0.0, "gate10"),
]


# ---------------------------------------------------------------- 工具函数
def rel(path) -> str:
    """入库写相对仓库根的路径（tests/test_portable_paths.py）。"""
    return config.rel_to_root(str(path))


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
            capture_output=True, text=True, timeout=10,
            env=subproc.child_env(),
        ).stdout.strip()
    except Exception:
        return ""


def neighbors(adj, eid: str) -> set[str]:
    out: set[str] = set()
    for key in ("out", "in"):
        for item in (adj.get(eid) or {}).get(key) or []:
            nid = str(item.get("neighbor_id") or "")
            if nid and nid != eid:
                out.add(nid)
    return out


def pair_relations(adj, a: str, b: str) -> set[str]:
    """A→B 的图关系词。邻接表字段名是 `relation`（不是 `predicate`）。"""
    rels: set[str] = set()
    for item in (adj.get(a) or {}).get("out") or []:
        if str(item.get("neighbor_id")) == b:
            name = str(item.get("relation") or "")
            if name:
                rels.add(name)
    return rels


def deep_search(search_fn, qa, pool: int):
    """与生产 `search(top_k=20)` 同源的深池。

    `search()` 内部 `depth = top_k * CANDIDATE_POOL_MULT`；直接把 top_k 放大到 pool
    会把 depth 抬到 pool*10，BM25 候选集与 `_norm_scores()` 归一分母一起漂移。
    把乘数临时压到 1，让 top_k 自己承担深度 → depth 正好等于 pool。

    必须传入**原始** `search`：补丁生效时 `retriever.search` 已被替换，
    走 `retriever.search` 会无限递归（这版第一跑就撞上 RecursionError）。
    """
    saved = retrieve.CANDIDATE_POOL_MULT
    retrieve.CANDIDATE_POOL_MULT = 1
    try:
        return search_fn(qa, top_k=pool)
    finally:
        retrieve.CANDIDATE_POOL_MULT = saved


def question_signals(retriever, qa, hits):
    """每题算一次逐块信号：pair_hit / hop_line（含 cn）/ cooccur。"""
    adj = retriever.adjacency
    candidates = qa.get("entity_candidates") or []
    rels: set[str] = set()
    common_names: list[str] = []
    endpoint_names: list[str] = []
    cn = 0
    if len(candidates) >= 2:
        a = str(candidates[0].get("entity_id") or "")
        b = str(candidates[1].get("entity_id") or "")
        if a and b:
            rels = pair_relations(adj, a, b)
            common_ids = neighbors(adj, a) & neighbors(adj, b)
            cn = len(common_ids)
            common_names = [retriever.display_name(e) for e in common_ids]
            common_names = [x for x in common_names if x]
    for cand in candidates[:2]:
        ent = retriever.entities_by_id.get(str(cand.get("entity_id") or "")) or {}
        nm = ent.get("name") or cand.get("name")
        if nm:
            endpoint_names.append(str(nm))

    chunk_pred: dict[str, set[str]] = {}
    for h in hits:
        cid = h["chunk_id"]
        preds = {str(r.get("predicate") or "") for r in (retriever.assertions_by_chunk.get(cid) or [])}
        preds.discard("")
        chunk_pred[cid] = preds

    hop: dict[str, int] = {}
    cooc: dict[str, int] = {}
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
    return {"rels": rels, "chunk_pred": chunk_pred, "hop": hop, "cooc": cooc,
            "cn": cn, "common_names": common_names, "endpoint_names": endpoint_names}


def make_patched_search(retriever, original, w_pair: float, w_hop: float, w_cooc: float,
                        suppress: str, state: dict, pool: int = DEFAULT_POOL,
                        cache: dict | None = None):
    """返回替换版 search：深池取回后按配置重排，加成**折进 score**，切前 20。

    `suppress` 为 `none` 时与早先回放口径一致；其余按 `SUPPRESS[name](cn)` 缩放
    hop 加成。`state` 由调用方传入，用来把每题信号回传给主循环。

    `cache` 用来跨配置复用同一题的深池与信号：同一题 15 个配置的深池结果与逐块信号
    完全相同，重算是纯浪费（深池 141 ms + 200 块正文逐行扫）。缓存由主循环按题填。
    """
    sfn = SUPPRESS[suppress]

    def patched(qa, top_k=None):
        if cache is not None and cache.get("payload") is not None:
            payload = cache["payload"]
            sig = cache["signals"]
        else:
            payload = deep_search(original, qa, pool)
            sig = None
        hits = payload.get("hits") or []
        if not hits:
            return {"hits": [], "stats": payload.get("stats") or {}}
        if sig is None:
            sig = question_signals(retriever, qa, hits)
        order = [h["chunk_id"] for h in hits]
        hit_by_id = {h["chunk_id"]: h for h in hits}
        rank = {cid: i + 1 for i, cid in enumerate(order)}
        score = {h["chunk_id"]: float(h["score"]) for h in hits}
        cn = int(sig["cn"])
        hop_scale = sfn(cn) if cn > 0 else 0.0

        def boost(cid: str) -> float:
            b = 0.0
            if w_pair and (sig["rels"] & sig["chunk_pred"].get(cid, set())):
                b += w_pair
            if w_hop and sig["hop"].get(cid):
                b += w_hop * hop_scale
            if w_cooc and sig["cooc"].get(cid):
                b += w_cooc
            return b

        def key(cid: str):
            return (-(score[cid] + boost(cid)), rank[cid])

        new_order = sorted(order, key=key)
        new_hits = []
        for cid in new_order[:EVAL_TOP_K]:
            h2 = dict(hit_by_id[cid])
            # 关键：加成折进 score，否则 build_context 按原分重排，加成完全无效
            h2["score"] = score[cid] + boost(cid)
            new_hits.append(h2)
        stats = dict(payload.get("stats") or {})
        # 生产里 top_score 取排序后首条；折进 score 后应同步重算
        stats["top_score"] = float(new_hits[0]["score"]) if new_hits else 0.0
        state["order"] = order
        state["new_order"] = new_order
        state["cn"] = cn
        state["hop_scale"] = hop_scale
        state["hop_chunks_pool"] = sum(1 for c in order if sig["hop"].get(c))
        state["hop_chunks_top20"] = sum(1 for c in new_order[:EVAL_TOP_K] if sig["hop"].get(c))
        state["pair_chunks_top20"] = sum(
            1 for c in new_order[:EVAL_TOP_K] if sig["rels"] & sig["chunk_pred"].get(c, set())
        )
        state["top20_changed"] = new_order[:EVAL_TOP_K] != order[:EVAL_TOP_K]
        return {"hits": new_hits, "stats": stats}

    return patched


def coverage_of(answer: dict, reqs: list[dict]) -> bool:
    """与 scripts/ask.py:283-298 同口径。"""
    emitted = {
        (str(c.get("chunk_id") or ""), str(c.get("evidence") or ""))
        for c in (answer.get("citations") or [])
    }
    return bool(reqs) and all(
        any(cid == str(r.get("chunk_id") or "") and str(r.get("evidence") or "") in ev
            for cid, ev in emitted)
        for r in reqs
    )


def load_dev(gold_path: Path) -> list[dict]:
    """调参集：gold-v2 的 split == dev，且应答题。验收集一律不碰。"""
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    out = []
    for it in gold["items"]:
        if it.get("split") != "dev":
            continue
        if it.get("should_abstain"):
            continue
        if not it.get("gold_chunk_ids"):
            continue
        out.append(it)
    return out


# ---------------------------------------------------------------- 主流程
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="hop_line 共同邻居抑制测量（只读）")
    ap.add_argument("--generation", default=DEFAULT_GENERATION)
    ap.add_argument("--gold-v2", default=DEFAULT_GOLD_V2)
    ap.add_argument("--pool", type=int, default=DEFAULT_POOL)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stage", choices=("dist", "real"), default="dist")
    ap.add_argument("--configs", default="", help="逗号分隔的配置名过滤，空=全部")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args(argv)

    gold_path = Path(args.gold_v2)
    if not gold_path.is_absolute():
        gold_path = ROOT / gold_path
    if not gold_path.is_file():
        print(f"[2] 输入不可用：找不到 {gold_path}", file=sys.stderr)
        return 2

    questions = load_dev(gold_path)
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        print("[2] 输入不可用：调参集为空", file=sys.stderr)
        return 2
    print(f"调参集应答题 {len(questions)} 道（split=dev）", flush=True)

    loaded = pipeline_mod.load_generation(args.generation)
    base_pipe = pipeline_mod.build_pipeline(loaded, top_k=EVAL_TOP_K, graph_depth=1)
    retriever = base_pipe["retriever"]
    entities = retriever.graph.get("entities") or []
    alias_index = retriever.graph.get("alias_index")

    selected = [c for c in CONFIGS]
    if args.configs:
        want = {x.strip() for x in args.configs.split(",") if x.strip()}
        selected = [c for c in CONFIGS if c[0] in want]
        if not selected:
            print(f"[2] 输入不可用：--configs 没匹配到任何配置", file=sys.stderr)
            return 2

    # ---------------- stage dist：只量噪声规模 ----------------
    if args.stage == "dist":
        cn_hist = collections.Counter()
        hop_hist = collections.Counter()
        cooc_hist = collections.Counter()
        per_q = []
        for idx, item in enumerate(questions, 1):
            if idx % 50 == 0:
                print(f"  ... {idx}/{len(questions)}", flush=True)
            qa = query_mod.analyze_query(
                item["question"], entities=entities, aliases=alias_index,
                vocab=vocab, context=getattr(retriever, "query_context", None),
            )
            payload = deep_search(retriever.search, qa, args.pool)
            hits = payload.get("hits") or []
            if not hits:
                continue
            sig = question_signals(retriever, qa, hits)
            cn = sig["cn"]
            hop_pool = sum(1 for c in sig["hop"] if sig["hop"][c])
            hop_top = sum(1 for h in hits[:EVAL_TOP_K] if sig["hop"].get(h["chunk_id"]))
            cooc_top = sum(1 for h in hits[:EVAL_TOP_K] if sig["cooc"].get(h["chunk_id"]))
            cn_hist[cn] += 1
            hop_hist[hop_top] += 1
            cooc_hist[cooc_top] += 1
            per_q.append({
                "question_id": item["question_id"], "category": item["category"],
                "origin": item.get("origin"), "cn": cn,
                "hop_chunks_pool": hop_pool, "hop_chunks_top20": hop_top,
                "cooc_chunks_top20": cooc_top,
                "pool_size": len(hits),
            })
        print()
        print("=== cn（共同邻居数）分布 ===")
        buckets = [(0, 0), (1, 3), (4, 9), (10, 29), (30, 99), (100, 10**6)]
        for lo, hi in buckets:
            n = sum(v for k, v in cn_hist.items() if lo <= k <= hi)
            tag = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**6 else f">={lo}")
            print(f"  cn {tag:>9s}: {n:3d} 题")
        print()
        print("=== hop_line 在「前 20 内命中块数」分布 ===")
        hb = [(0, 0), (1, 3), (4, 9), (10, 10**6)]
        for lo, hi in hb:
            n = sum(v for k, v in hop_hist.items() if lo <= k <= hi)
            tag = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**6 else f">={lo}")
            print(f"  命中块数 {tag:>6s}: {n:3d} 题")
        vals = [r["hop_chunks_top20"] for r in per_q]
        if vals:
            vals_sorted = sorted(vals)
            mid = vals_sorted[len(vals_sorted) // 2]
            print(f"  中位 {mid}、均值 {sum(vals)/len(vals):.2f}、"
                  f"最大 {max(vals)}、=0 的题 {sum(1 for v in vals if v == 0)}/{len(vals)}")
        cn_vals = sorted(r["cn"] for r in per_q)
        if cn_vals:
            print(f"  cn 中位 {cn_vals[len(cn_vals)//2]}、均值 {sum(cn_vals)/len(cn_vals):.2f}、"
                  f"最大 {max(cn_vals)}、=0 的题 {sum(1 for v in cn_vals if v == 0)}/{len(cn_vals)}")

        # cooc 的稀疏度：用于「信号挑不挑块」这条判据（见 docs/a_P2_cooc实验方案与hop根因.md §1.3）
        print()
        print("=== cooc 在「前 20 内命中块数」分布 ===")
        for lo, hi in hb:
            n = sum(v for k, v in cooc_hist.items() if lo <= k <= hi)
            tag = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**6 else f">={lo}")
            print(f"  命中块数 {tag:>6s}: {n:3d} 题")
        cvals = [r["cooc_chunks_top20"] for r in per_q]
        if cvals:
            print(f"  中位 {sorted(cvals)[len(cvals)//2]}、均值 {sum(cvals)/len(cvals):.2f}、"
                  f"最大 {max(cvals)}、=0 的题 {sum(1 for v in cvals if v == 0)}/{len(cvals)}")

        out_path = args.out_json or "data/a_hop_suppress_dist.json"
        payload = {
            "schema": "arkrag.a-hop-suppress-dist/1",
            "purpose": "量 hop_line 与 cooc 的稠密度：共同邻居数与前 20 内命中块数分布",
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": _git_commit(), "python": platform.python_version(),
            "os": platform.platform(), "command": "py -3 scripts/a_hop_suppress.py --stage dist",
            "generation": args.generation,
            "gold_set": rel(gold_path), "split": "dev", "pool": args.pool,
            "n_questions": len(per_q),
            "cn_hist": {str(k): v for k, v in sorted(cn_hist.items())},
            "hop_top20_hist": {str(k): v for k, v in sorted(hop_hist.items())},
            "cooc_top20_hist": {str(k): v for k, v in sorted(cooc_hist.items())},
            "per_question": per_q,
        }
        (ROOT / out_path).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n写入 {out_path}")
        return 0

    # ---------------- stage real：真跑 ask_once ----------------
    original_search = retriever.search
    results: dict[str, list[dict]] = {c[0]: [] for c in selected}
    for idx, item in enumerate(questions, 1):
        if idx % 25 == 0:
            print(f"  ... {idx}/{len(questions)}", flush=True)
        gold_ids = [str(g) for g in item["gold_chunk_ids"]]
        reqs = item.get("required_citations") or []
        # 同一题的深池与逐块信号在 15 个配置间完全相同，按题建一次缓存
        base_search = retriever.search
        qa_cache = query_mod.analyze_query(
            item["question"], entities=entities, aliases=alias_index,
            vocab=vocab, context=getattr(retriever, "query_context", None),
        )
        _pool_payload = deep_search(base_search, qa_cache, args.pool)
        cache = {
            "payload": _pool_payload,
            "signals": question_signals(retriever, qa_cache, _pool_payload.get("hits") or []),
        }
        for name, wp, wh, wc, sup in selected:
            if name == "基线(不打补丁)":
                out = pipeline_mod.ask_once(base_pipe, item["question"])
                state = {"top20_changed": False, "cn": cache["signals"]["cn"], "hop_scale": None,
                         "hop_chunks_top20": 0, "pair_chunks_top20": 0, "hop_chunks_pool": 0}
                hits = (out.get("search") or {}).get("hits") or []
                order = [h["chunk_id"] for h in hits]
                rank = {cid: i + 1 for i, cid in enumerate(order)}
            else:
                state = {}
                retriever.search = make_patched_search(
                    retriever, original_search, wp, wh, wc, sup, state, args.pool, cache)
                try:
                    out = pipeline_mod.ask_once(base_pipe, item["question"])
                finally:
                    retriever.search = original_search
                order = state.get("new_order") or []
                rank = {cid: i + 1 for i, cid in enumerate(order)}
            units = (out.get("context") or {}).get("units") or []
            present = {str(u.get("chunk_id")) for u in units}
            any_gold = any(g in present for g in gold_ids)
            all_gold = all(g in present for g in gold_ids)
            ans = out.get("answer") or {}
            results[name].append({
                "question_id": item["question_id"],
                "category": item["category"],
                "origin": item.get("origin"),
                "any_gold": any_gold,
                "all_gold": all_gold,
                "covered": coverage_of(ans, reqs),
                "abstained": bool(ans.get("abstained")),
                "units": len(units),
                "answer_len": len(str(ans.get("answer") or "")),
                "citations": len(ans.get("citations") or []),
                "gold_first_rank": min((rank[g] for g in gold_ids if g in rank), default=None),
                "cn": state.get("cn"),
                "hop_scale": state.get("hop_scale"),
                "hop_chunks_top20": state.get("hop_chunks_top20"),
                "pair_chunks_top20": state.get("pair_chunks_top20"),
                "top20_changed": state.get("top20_changed"),
            })

    # 汇总
    base_name = "基线(不打补丁)"
    base_by_q = {r["question_id"]: r for r in results.get(base_name, [])}
    summary = {}
    for name in results:
        rs = results[name]
        gain = [r["question_id"] for r in rs
                if r["any_gold"] and not base_by_q.get(r["question_id"], {}).get("any_gold", False)]
        lose = [r["question_id"] for r in rs
                if base_by_q.get(r["question_id"], {}).get("any_gold", False) and not r["any_gold"]]
        summary[name] = {
            "n": len(rs),
            "any_gold": sum(1 for r in rs if r["any_gold"]),
            "all_gold": sum(1 for r in rs if r["all_gold"]),
            "covered": sum(1 for r in rs if r["covered"]),
            "abstained": sum(1 for r in rs if r["abstained"]),
            "gain": len(gain), "lose": len(lose), "net": len(gain) - len(lose),
            "gain_ids": gain, "lose_ids": lose,
        }

    # 按类
    by_cat: dict[str, dict[str, dict]] = {}
    for name in results:
        agg = collections.defaultdict(lambda: {"n": 0, "any_gold": 0, "covered": 0,
                                               "gain": 0, "lose": 0})
        for r in results[name]:
            a = agg[r["category"]]
            a["n"] += 1
            a["any_gold"] += int(r["any_gold"])
            a["covered"] += int(r["covered"])
            b = base_by_q.get(r["question_id"], {})
            if r["any_gold"] and not b.get("any_gold", False):
                a["gain"] += 1
            if b.get("any_gold", False) and not r["any_gold"]:
                a["lose"] += 1
        by_cat[name] = {k: dict(v) for k, v in agg.items()}

    out_path = args.out_json or "data/a_hop_suppress_real.json"
    payload = {
        "schema": "arkrag.a-hop-suppress-real/1",
        "purpose": "hop_line 共同邻居抑制的真跑测量：进上下文 / 引用覆盖 / 救回 / 丢失",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_commit(), "python": platform.python_version(),
        "os": platform.platform(),
        "command": "py -3 scripts/a_hop_suppress.py --stage real",
        "generation": args.generation,
        "gold_set": rel(gold_path), "split": "dev", "pool": args.pool,
        "n_questions": len(questions),
        "note": ("加成已折进 hit['score']：build_context（retrieve.py:850-853）只读 score 不读顺序，"
                 "只改顺序会得到「R@k 涨、下游不动」的假增益。stats['top_score'] 已按新排序重算。"),
        "configs": [{"name": n, "w_pair": wp, "w_hop": wh, "w_cooc": wc, "suppress": s}
                    for n, wp, wh, wc, s in selected],
        "summary": summary,
        "by_category": by_cat,
        "per_question": results,
    }
    (ROOT / out_path).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    print()
    hdr = f"{'配置':26s} {'进ctx':>6s} {'全进':>5s} {'覆盖':>5s} {'拒答':>5s} {'救':>3s} {'丢':>3s} {'净':>4s}"
    print(hdr)
    for name in results:
        s = summary[name]
        print(f"{name:26s} {s['any_gold']:6d} {s['all_gold']:5d} {s['covered']:5d} "
              f"{s['abstained']:5d} {s['gain']:3d} {s['lose']:3d} {s['net']:+4d}")
        if s["lose_ids"]:
            print(f"{'':26s} 丢失: {s['lose_ids']}")
    print(f"\n写入 {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
