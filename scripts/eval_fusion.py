"""I 阶段评测：基线（BM25+实体加权）vs RRF 三通道融合。

用法：

    py -3 scripts\\eval_fusion.py --limit 320
    py -3 scripts\\eval_fusion.py --limit 320 --json data\\eval_fusion.json
    py -3 scripts\\eval_fusion.py --limit 320 --modes bm25 rrf     # 只跑指定模式

## 为什么只测召回，不测答案

答案质量由 `answer.py` 决定，而它当前的瓶颈是「检索给出 42.6 个候选单元、上下文预算
只装得下 4.7 块」（G 阶段实测，引用覆盖率 0.1567 的根因）。融合改变的是**候选排序**，
所以这一轮该量的就是排序指标：Recall@5/20/50、MRR、以及"gold 块是否进入前 N"。
答案侧的收益要等上下文预算那一层改完才能干净地量出来——混在一起测，两个改动
的贡献会互相淹没。

## 四种模式的含义

* `baseline`：`retrieve.Retriever.search`（`bm25 + 0.5*entity`）。这是 G 阶段基线，
  **必须原地复现**，否则对比无意义。
* `bm25`：只用词法通道（RRF 融合里只有一路）。用来回答「融合的收益里有多少
  其实只是"去掉实体加权"带来的」。
* `rrf`：三通道 RRF（bm25 + vector + entity 字面匹配）。
* `rrf_novec`：两通道 RRF（bm25 + entity），用来**单独分离向量通道的贡献**。

`rrf` vs `rrf_novec` 的差就是向量通道的净收益。没有这个对照，"融合变好了"
分不清是融合机制的功劳还是新增通道的功劳。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# 本脚本会打印 U+2212（减号 −）与 Δ 一类的排印字符。Windows 控制台默认 cp936，
# 它编不出 U+2212，于是 `print` 自己抛 UnicodeEncodeError、进程以 1 退出——
# 看起来像「评测脚本跑挂了」，其实是控制台编码。2026-11-04 Gate 0 实测：
# 同样的字符串在 cp936 控制台下 rc=1，reconfigure 之后 rc=0。
# 守卫是必要的：pytest 捕获期间 `sys.stdout` 可能是不可重配置的文本流。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from arkrag import (  # noqa: E402
    embedding,
    fusion,
    index as index_mod,
    query as query_mod,
    release,
    retrieve,
    vector_store,
    vocab as vocab_mod,
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stratified(items: list[dict], limit: int | None, *, seed: int = 20261101) -> list[dict]:
    """按类别配额分层抽样。复用 `scripts/ask.py` 的口径，不在这里另写一份。

    为什么必须分层：`gold-v1.json` 里 single_chunk 排最前（64 条），
    按文件顺序截断会让任何局部评测只覆盖一个类别（G 阶段 §4.3 的教训）。
    """
    if limit is None or limit >= len(items):
        return list(items)
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_ask_for_sampling", Path(__file__).resolve().parent / "ask.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._stratified_sample(items, limit, seed=seed)


def percentile(values: list[float], pct: float) -> float | None:
    """nearest-rank。与 `bench_h1.percentile` 同一口径，空样本返 None。"""
    if not values:
        return None
    s = sorted(values)
    rank = max(1, min(len(s), int(-(-pct * len(s) // 100))))
    return s[rank - 1]


def metrics_for(hits_per_q: list[list[str]], questions: list[dict]) -> dict:
    """Recall@k / MRR。gold 为空的题单独计数，分母只算能算的。"""
    r5, r20, r50, rr = [], [], [], []
    skipped = 0
    for hits, item in zip(hits_per_q, questions):
        gold = [str(c) for c in item.get("gold_chunk_ids") or []]
        if not gold:
            skipped += 1
            continue
        gs = set(gold)
        r5.append(len(gs & set(hits[:5])) / len(gs))
        r20.append(len(gs & set(hits[:20])) / len(gs))
        r50.append(len(gs & set(hits[:50])) / len(gs))
        first = next((i for i, cid in enumerate(hits) if cid in gs), None)
        rr.append(0.0 if first is None else 1.0 / (first + 1))
    n = len(r20) or 1
    return {
        "n": len(r20),
        "skipped": skipped,
        "recall@5": round(sum(r5) / n, 4),
        "recall@20": round(sum(r20) / n, 4),
        "recall@50": round(sum(r50) / n, 4),
        "mrr": round(sum(rr) / n, 4),
    }


def per_category(hits_per_q: list[list[str]], questions: list[dict]) -> dict:
    buckets: dict[str, list[tuple[list[str], dict]]] = {}
    for hits, item in zip(hits_per_q, questions):
        buckets.setdefault(str(item.get("category") or "?"), []).append((hits, item))
    out = {}
    for cat, rows in sorted(buckets.items()):
        r20, rr = [], []
        for hits, item in rows:
            gold = [str(c) for c in item.get("gold_chunk_ids") or []]
            if not gold:
                continue
            gs = set(gold)
            r20.append(len(gs & set(hits[:20])) / len(gs))
            first = next((i for i, cid in enumerate(hits) if cid in gs), None)
            rr.append(0.0 if first is None else 1.0 / (first + 1))
        if r20:
            out[cat] = {
                "n": len(r20),
                "recall@20": round(sum(r20) / len(r20), 4),
                "mrr": round(sum(rr) / len(rr), 4),
            }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="I 阶段：基线 vs RRF 融合召回对比")
    ap.add_argument("--generation", help="世代 id；省略取 current 的 identity 世代")
    ap.add_argument("--gold-set", default="data/goldset/gold-v1.json")
    ap.add_argument("--limit", type=int, default=320)
    ap.add_argument("--top-k", type=int, default=50, help="每条通道/最终结果取多少（测 Recall@50 需要 50）")
    ap.add_argument("--channel-k", type=int, default=100, help="每条通道参与融合的候选数")
    ap.add_argument(
        "--rrf-k",
        type=float,
        default=float(fusion.RRF_K),
        help="RRF 平滑常数。越小越看重头部名次（k=60 会把第 1 名和第 30 名压平）",
    )
    ap.add_argument("--modes", nargs="+", default=["baseline", "bm25", "rrf_novec", "rrf"])
    ap.add_argument("--json", help="把完整结果写到这个路径")
    args = ap.parse_args(argv)

    gold = json.loads(Path(args.gold_set).read_text(encoding="utf-8"))
    items = gold["items"] if isinstance(gold, dict) else gold
    questions = stratified(items, args.limit)
    print(f"题集 {args.gold_set}  共 {len(items)} 题，本次评测 {len(questions)} 题")

    gid = args.generation
    if not gid:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_ask_for_gen", Path(__file__).resolve().parent / "ask.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        gid = mod._latest_identity_generation()
    print(f"世代 {gid}")

    gen = release.Generation(gid)
    from arkrag import config

    chunks_path = config.chunks_path()
    chunks = {str(r["chunk_id"]): r for r in read_jsonl(chunks_path) if r.get("chunk_id")}
    index, index_meta = index_mod.load_index(gen.index_path)
    content_digest = (index_meta or {}).get("index_content_digest") or ""
    graph = json.loads(gen.graph_path.read_text(encoding="utf-8"))

    retriever = retrieve.Retriever(index, graph, {}, chunks=chunks, vocab=vocab_mod)
    entities = graph.get("entities") or []
    print(f"块 {len(chunks)}，实体 {len(retriever.entities_by_id)}")

    # 向量索引：缺了就明确说缺，不许静默少一路
    vec = None
    model = embedding.model_name()
    bin_path, meta_path = vector_store.vector_paths(vector_store._slug(model), content_digest)
    if bin_path.exists() and meta_path.exists():
        try:
            matrix, vmeta = vector_store.load_vectors(bin_path, meta_path)
        except (ValueError, KeyError) as exc:
            print(f"[!] 向量产物读不出来：{exc}（跳过向量通道）", file=sys.stderr)
        else:
            vec = vector_store.VectorIndex(matrix, vmeta)
            print(f"向量 {len(vec)} 条 × {vec.dim} 维，model={vec.model}")
    else:
        print(f"[!] 没有向量产物 {bin_path.name}（先跑 build_vectors.py），rrf 将退化")

    def embed_query(text: str) -> list[float]:
        return embedding.embed_batch([text])[0]

    fuser = fusion.FusionRetriever(
        index,
        vec,
        chunks=chunks,
        entities_by_id=retriever.entities_by_id,
        embed_query=embed_query,
        channel_k=args.channel_k,
        rrf_k=args.rrf_k,
    )

    results: dict[str, dict] = {}
    for mode in args.modes:
        print(f"\n=== 模式 {mode} ===")
        hits_per_q: list[list[str]] = []
        lat: list[float] = []
        chan_stats: list[dict] = []
        for i, item in enumerate(questions):
            q = str(item.get("question") or "")
            t0 = time.perf_counter()
            analysis = query_mod.analyze_query(
                q,
                entities=entities,
                aliases=graph.get("alias_index"),
                vocab=vocab_mod,
                context=retriever.query_context,
            )
            if mode == "baseline":
                out = retriever.search(analysis, top_k=args.top_k)
            elif mode == "bm25":
                text = str(analysis.get("retrieval_query") or analysis.get("normalized_query") or "")
                ranked = [cid for cid, _ in index.search(text, top_k=args.top_k)]
                out = {"hits": [{"chunk_id": c} for c in ranked], "stats": {}}
            elif mode == "rrf_novec":
                saved = fuser.vector_index
                fuser.vector_index = None
                out = fuser.search(analysis, top_k=args.top_k)
                fuser.vector_index = saved
            elif mode == "rrf":
                out = fuser.search(analysis, top_k=args.top_k)
            else:
                raise SystemExit(f"未知模式 {mode}")
            lat.append((time.perf_counter() - t0) * 1000)
            hits_per_q.append([str(h["chunk_id"]) for h in out["hits"]])
            if out["stats"].get("channels_used"):
                chan_stats.append(out["stats"]["channels_used"])
            if (i + 1) % 40 == 0:
                print(f"  {i + 1}/{len(questions)}", flush=True)

        m = metrics_for(hits_per_q, questions)
        m["latency_p50_ms"] = round(percentile(lat, 50) or 0, 2)
        m["latency_p95_ms"] = round(percentile(lat, 95) or 0, 2)
        m["peak_hit_count"] = max((len(h) for h in hits_per_q), default=0)
        m["by_category"] = per_category(hits_per_q, questions)
        if chan_stats:
            agg: dict[str, list[int]] = {}
            for row in chan_stats:
                for k, v in row.items():
                    if isinstance(v, int):
                        agg.setdefault(k, []).append(v)
            m["channels_avg"] = {k: round(statistics.mean(v), 1) for k, v in agg.items()}
        results[mode] = m
        print(
            f"  R@5 {m['recall@5']:.4f}  R@20 {m['recall@20']:.4f}  R@50 {m['recall@50']:.4f}  "
            f"MRR {m['mrr']:.4f}  P50 {m['latency_p50_ms']:.1f}ms  P95 {m['latency_p95_ms']:.1f}ms  (n={m['n']})"
        )

    # ---- 对比表 ----
    print("\n" + "=" * 88)
    print(f"{'模式':<12} {'R@5':>8} {'R@20':>8} {'R@50':>8} {'MRR':>8} {'P50ms':>8} {'P95ms':>8}")
    print("-" * 88)
    base = results.get("baseline") or {}
    for mode, m in results.items():
        print(
            f"{mode:<12} {m['recall@5']:>8.4f} {m['recall@20']:>8.4f} {m['recall@50']:>8.4f} "
            f"{m['mrr']:>8.4f} {m['latency_p50_ms']:>8.1f} {m['latency_p95_ms']:>8.1f}"
        )
    if base:
        print("-" * 88)
        for mode, m in results.items():
            if mode == "baseline":
                continue
            d20 = m["recall@20"] - base["recall@20"]
            dm = m["mrr"] - base["mrr"]
            d50 = m["recall@50"] - base["recall@50"]
            sign = lambda x: f"{x:+.4f}"  # noqa: E731
            print(
                f"{mode:<12} ΔR@20 {sign(d20)}  ΔMRR {sign(dm)}  ΔR@50 {sign(d50)}"
            )

    # ---- 向量通道净收益 ----
    if "rrf" in results and "rrf_novec" in results:
        a, b = results["rrf"], results["rrf_novec"]
        print(
            f"\n向量通道净收益（rrf − rrf_novec）："
            f"ΔR@20 {a['recall@20'] - b['recall@20']:+.4f}  "
            f"ΔMRR {a['mrr'] - b['mrr']:+.4f}  "
            f"ΔR@50 {a['recall@50'] - b['recall@50']:+.4f}"
        )

    if args.json:
        out = {
            "generation": gid,
            "gold_set": args.gold_set,
            "questions": len(questions),
            "top_k": args.top_k,
            "channel_k": args.channel_k,
            "vector": {
                "model": vec.model if vec else None,
                "dim": vec.dim if vec else None,
                "count": len(vec) if vec else 0,
                "digest": (vec.model_digest[:16] if vec else None),
            },
            "results": results,
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n结果写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
