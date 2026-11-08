"""召回组合对比（第二轮）：不动 base 排序的融合方式。

第一轮的结论（`data/eval_vec_mix.json`，300 题）：
  base            R@20 0.7900  MRR 0.4388
  vec_only        R@20 0.4833  MRR 0.2203
  mix_w0.25       R@20 0.7700  MRR 0.4396   ← 分值 minmax 加权，R@20 掉
  mix_w1          R@20 0.7300  MRR 0.4080
  mix_rank(RRF60) R@20 0.7600  MRR 0.3429
⇒ minmax 分值融合与名次倒数融合都在压坏 base 的排序。

关键事实：向量能把 14 道题的 gold 拉进池子（base 237 题命中 → 并集 251 题），
但那 14 块只能排在 base 20 名之后，所以「改变排序」一旦失手，收益立刻被
base 的损失吃掉。这一轮换三类不改 base 相对顺序的组合：

  slots_k*     —— 保留 base 前 (20-k) 名原封不动，尾部 k 个位置从向量榜补位
  gate_*       —— base 分值乘以 (1 + β·向量归一化分)，只在向量强烈偏好时才动
  lock3_vec    —— 锁死 base 前 3 名，其余 17 个位置按向量分重排（前项目「补位不整体重排」）
  rrf_k1       —— 名次倒数用 k=1（参比比核查报告：rrf_k=1 让 rank1/rank6 倍率 1.08×→3.50×）

用法：
  py -3 scripts/eval_combo.py --limit 320 --json data/eval_combo.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import embedding  # noqa: E402
from arkrag import pipeline  # noqa: E402
from arkrag import query as query_mod  # noqa: E402
from arkrag import retrieve  # noqa: E402
from arkrag import vector_store  # noqa: E402
from arkrag import vocab as vocab_mod  # noqa: E402

TOP_K = 20
VECTOR_K = 100


def pct(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    if hi - lo <= 1e-12:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def rank_of(ids, gold):
    for i, cid in enumerate(ids, 1):
        if cid in gold:
            return i
    return None


def slots(base_ids, vec_ids, k, top_k):
    """base 前 (top_k-k) 名原封不动，尾部 k 位优先从向量榜补（跳过已出现的），补不满再回填 base 尾巴。"""
    head = list(base_ids[: max(0, top_k - k)])
    seen = set(head)
    out = list(head)
    for cid in vec_ids:
        if len(out) >= top_k:
            break
        if cid in seen:
            continue
        seen.add(cid)
        out.append(cid)
    for cid in base_ids[len(head):]:
        if len(out) >= top_k:
            break
        if cid in seen:
            continue
        seen.add(cid)
        out.append(cid)
    return out


def gate(base_scores, vec_scores, beta, base_ids, vec_ids, top_k):
    """base 分 × (1 + β·向量归一化分)。向量榜上没有的块向量分记 0，不会被动。"""
    bn = minmax(base_scores)
    vn = minmax(vec_scores)
    merged = {}
    for cid, s in bn.items():
        merged[cid] = s * (1.0 + beta * vn.get(cid, 0.0))
    for cid, s in vn.items():
        if cid not in merged:
            merged[cid] = 0.0 * (1.0 + beta * s)
    brank = {cid: r for r, cid in enumerate(base_ids, 1)}
    vrank = {cid: r for r, cid in enumerate(vec_ids, 1)}
    ranked = sorted(
        merged.items(),
        key=lambda kv: (-kv[1], brank.get(kv[0], 10**9), vrank.get(kv[0], 10**9), kv[0]),
    )
    return [c for c, _ in ranked[:top_k]]


def lock3_vec(base_ids, vec_ids, vec_scores, top_k, lock=3, vec_pool=40):
    """锁死 base 前 lock 名；其余位置在 (base 余下 ∪ 向量前 vec_pool) 里按向量分重排。"""
    head = list(base_ids[:lock])
    seen = set(head)
    pool = []
    for cid in base_ids[lock:]:
        if cid not in seen:
            seen.add(cid)
            pool.append(cid)
    for cid in vec_ids[:vec_pool]:
        if cid not in seen:
            seen.add(cid)
            pool.append(cid)
    brank = {cid: r for r, cid in enumerate(base_ids, 1)}
    vrank = {cid: r for r, cid in enumerate(vec_ids, 1)}
    pool.sort(
        key=lambda cid: (
            -vec_scores.get(cid, -1.0),
            brank.get(cid, 10**9),
            vrank.get(cid, 10**9),
            cid,
        )
    )
    return head + pool[: max(0, top_k - lock)]


def rrf_k1(base_ids, vec_ids, top_k):
    """名次倒数融合，k=1（比核查报告里 Compendium 生产口径）。"""
    inv = {}
    for ids in (base_ids, vec_ids):
        for r, cid in enumerate(ids, 1):
            inv[cid] = inv.get(cid, 0.0) + 1.0 / (1 + r)
    ranked = sorted(inv.items(), key=lambda kv: (-kv[1], kv[0]))
    return [c for c, _ in ranked[:top_k]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generation", default="g-20261103T132152Z-06a741")
    ap.add_argument("--gold-set", default="data/goldset/gold-v1.json")
    ap.add_argument("--limit", type=int, default=320)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--vector-k", type=int, default=VECTOR_K)
    ap.add_argument("--json", default="data/eval_combo.json")
    args = ap.parse_args()

    loaded = pipeline.load_generation(args.generation)
    index = loaded["index"]
    graph = loaded["graph"]
    chunks = loaded["chunks"]
    digest = (loaded["index_meta"] or {}).get("index_content_digest") or ""

    retriever = retrieve.Retriever(index, graph, {}, chunks=chunks, vocab=vocab_mod)
    entities = graph.get("entities") or []

    model = embedding.model_name()
    bin_path, meta_path = vector_store.vector_paths(vector_store._slug(model), digest)
    if not (bin_path.exists() and meta_path.exists()):
        print(f"缺向量产物：{bin_path}", file=sys.stderr)
        return 3
    matrix, vmeta = vector_store.load_vectors(bin_path, meta_path)
    vec = vector_store.VectorIndex(matrix, vmeta)
    print(
        f"块 {len(chunks)}，实体 {len(retriever.entities_by_id)}，"
        f"向量 {len(vec.chunk_ids)}×{vec.dim} ({vec.model})",
        flush=True,
    )

    gold = json.loads((ROOT / args.gold_set).read_text(encoding="utf-8"))
    raw = gold.get("items") if isinstance(gold, dict) else gold
    items = [g for g in (raw or []) if g.get("gold_chunk_ids")][: args.limit]
    print(f"题 {len(items)}", flush=True)

    modes = [
        "base",
        "vec_only",
        "slots_k2",
        "slots_k3",
        "slots_k5",
        "gate_0.1",
        "gate_0.3",
        "gate_1.0",
        "lock3_vec",
        "rrf_k1",
    ]
    rows = []
    lat = {m: [] for m in modes}
    bad_combo = []

    for i, item in enumerate(items, 1):
        qid = str(item.get("question_id") or i)
        q = str(item.get("question") or "")
        gset = {str(c) for c in item["gold_chunk_ids"]}

        qa = query_mod.analyze_query(
            q, entities=entities, aliases=graph.get("alias_index"), vocab=vocab_mod
        )
        t0 = time.perf_counter()
        found = retriever.search(qa, top_k=args.top_k)
        t_base = 1000 * (time.perf_counter() - t0)
        base_ids = [str(h["chunk_id"]) for h in found["hits"]]
        base_scores = {str(h["chunk_id"]): float(h.get("score") or 0.0) for h in found["hits"]}

        t0 = time.perf_counter()
        qv = embedding.embed_batch([q])[0]
        vec_hits = vec.search(qv, top_k=args.vector_k)
        t_vec = 1000 * (time.perf_counter() - t0)
        vec_ids = [str(c) for c, _ in vec_hits]
        vec_scores = {str(c): float(s) for c, s in vec_hits}

        row = {"qid": qid, "base": rank_of(base_ids, gset), "vec_only": rank_of(vec_ids, gset)}
        lat["base"].append(t_base)
        lat["vec_only"].append(t_base + t_vec)

        combos = {
            "slots_k2": slots(base_ids, vec_ids, 2, args.top_k),
            "slots_k3": slots(base_ids, vec_ids, 3, args.top_k),
            "slots_k5": slots(base_ids, vec_ids, 5, args.top_k),
            "gate_0.1": gate(base_scores, vec_scores, 0.1, base_ids, vec_ids, args.top_k),
            "gate_0.3": gate(base_scores, vec_scores, 0.3, base_ids, vec_ids, args.top_k),
            "gate_1.0": gate(base_scores, vec_scores, 1.0, base_ids, vec_ids, args.top_k),
            "lock3_vec": lock3_vec(base_ids, vec_ids, vec_scores, args.top_k),
            "rrf_k1": rrf_k1(base_ids, vec_ids, args.top_k),
        }
        for name, ids in combos.items():
            row[name] = rank_of(ids, gset)
            lat[name].append(t_base + t_vec)
            if len(ids) != args.top_k or len(set(ids)) != args.top_k:
                bad_combo.append((qid, name, len(ids), len(set(ids))))

        rows.append(row)
        if i % 50 == 0 or i == len(items):
            print(f"  {i}/{len(items)}", flush=True)

    def metrics(key):
        n = len(rows)
        hit5 = hit20 = fifth = 0
        rr = 0.0
        for row in rows:
            rank = row[key]
            if rank is None:
                continue
            if rank <= 20:
                hit20 += 1
            if rank <= 5:
                hit5 += 1
                if rank == 5:
                    fifth += 1
            rr += 1.0 / rank
        return {
            "n": n,
            "recall@5": hit5 / n,
            "recall@20": hit20 / n,
            "mrr": rr / n,
            "rank5_share_of_hits": (fifth / hit20) if hit20 else 0.0,
        }

    result = {
        "generation": args.generation,
        "gold_set": args.gold_set,
        "questions": len(items),
        "top_k": args.top_k,
        "vector_k": args.vector_k,
        "vector": {
            "model": vec.model,
            "dim": vec.dim,
            "count": len(vec.chunk_ids),
            "digest": digest[:16],
        },
        "metrics": {m: metrics(m) for m in modes},
        "latency": {
            m: {
                "p50_ms": pct(lat[m], 0.5),
                "p95_ms": pct(lat[m], 0.95),
                "mean_ms": statistics.fmean(lat[m]) if lat[m] else 0.0,
            }
            for m in modes
        },
        "rows": rows,
    }
    out = ROOT / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    base_m = result["metrics"]["base"]
    print("\n=== 质量（输出窗 top-20），Δ 相对 base ===")
    print(f"{'模式':<12}{'R@5':>9}{'ΔR@5':>9}{'R@20':>9}{'ΔR@20':>9}{'MRR':>9}{'ΔMRR':>9}{'第5名':>9}")
    for m in modes:
        s = result["metrics"][m]
        print(
            f"{m:<12}{s['recall@5']:>9.4f}{s['recall@5'] - base_m['recall@5']:>+9.4f}"
            f"{s['recall@20']:>9.4f}{s['recall@20'] - base_m['recall@20']:>+9.4f}"
            f"{s['mrr']:>9.4f}{s['mrr'] - base_m['mrr']:>+9.4f}"
            f"{s['rank5_share_of_hits']:>9.4f}"
        )
    print("\n=== 延迟 ===")
    for m in modes:
        s = result["latency"][m]
        print(f"{m:<12}P50 {s['p50_ms']:>7.1f}ms  P95 {s['p95_ms']:>7.1f}ms")
    print(f"\n落盘 {out}")

    # 自校验：每条组合都必须恰好 top_k 个、无重复。长度不足会让 R@20 虚高
    # （缺的位置等于白送），所以这里必须当场报错而不是事后发现。
    if bad_combo:
        print(f"\n[!] 有 {len(bad_combo)} 条组合长度/重复不合规，前 5：{bad_combo[:5]}")
        return 4
    # base 前缀必须被 slots_* 原样保留（否则「改排序」与「补位」混在一起说不清）
    print("\n自校验：组合长度与唯一性 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
