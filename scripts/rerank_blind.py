"""离线盲特征线性重排：唯一同时保住头部与预算的候选方案。

## 为什么做这个（而不是继续调 RRF）

I1 已经用 42 + 128 个组合证明：**RRF 在本项目救不回来**。前项目
（`roleplay-ai`，见 `docs/i阶段-前项目经验检索.md`）也独立撞过同一堵墙，他们的
结论是「调权重无效，调 k 也有限」——因为 RRF 只用名次，名次一旦被弱通道污染
就没有信息可恢复。

他们的出路是**盲特征线性重排**：把每个候选算成一排特征（余弦、词法覆盖、
IDF 加权覆盖、篇名覆盖、原分、实体度、块长度……），各自 z-score 后线性加权。
纯 numpy、零模型、p95 22.4 ms，实测 mrr6 0.6147→0.7599 且 holdout Δmrr6 +0.1221。

**关键差别（不能照抄的地方）**：他们的选优函数是 `(-hit6, -mrr6, name)`，
即 hit@6 优先、MRR 只当 tie-breaker。我方上下文只装 4.7 块，所以 MRR 才是主指标。
本脚本用 `(mrr, r20)` 优先——与他们相反，这是刻意的。

## 过拟合防线（这是本脚本最重要的部分）

320 题、10 个权重、每个权重 4 档网格 = 搜索空间 4^10 ≈ 10^6。**在这样的空间里
随便怎么拟合都能在训练集上刷出分数。** 所以：

1. 按类别分层切成 **train / holdout** 两半（不随机打乱类别比例）。
2. 权重只在 train 上做坐标上升。
3. **holdout 只在最后跑一次**，不看它调参。若 holdout 不涨，结论就是"这条路对我方无效"。
4. 报 `train` 与 `holdout` 两组数，并明确写清哪个是拟合出来的、哪个是没见过的。

## 特征清单（每个都要能独立解释，不搞黑箱）

| 特征 | 定义 | 为什么可能有信号 |
|---|---|---|
| `orig` | 生产口径的最终分 | 基线本身 |
| `bm25` | 词法通道归一分 | 纯词面匹配 |
| `entity` | 实体通道归一分 | 问句专名命中 |
| `speaker` | 说话人结构化命中 | 当事人是否在场 |
| `cos` | 查询向量与块向量余弦 | 语义相似（bge-m3，已归一化→点积） |
| `lex` | 问句 bigram 被块文本覆盖的比例 | 字面重合 |
| `lex_idf` | 同上但按 IDF 加权 | 罕见词重合更说明问题 |
| `in_text` | 问句实体名是否出现在块正文 | 主体提及 |
| `len` | 块 token 长度（z-score） | 过短块信息少，过长块稀释 |

`rank` 不作为特征（它由其它特征决定，放进去会把重排退化成恒等）。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arkrag import (  # noqa: E402
    config,
    embedding,
    index as index_mod,
    query as query_mod,
    release,
    retrieve,
    vector_store,
    vocab as vocab_mod,
)

FEATURES = ("orig", "bm25", "entity", "speaker", "cos", "lex", "lex_idf", "in_text", "len")
GRID = (0.0, 0.5, 1.0, 2.0)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def bigrams(text: str) -> set[str]:
    """汉字 bigram 集合。与索引的 t1 语义接近，但这里只用于覆盖率，不追求逐位一致。"""
    chars = [c for c in text if "\u4e00" <= c <= "\u9fff"]
    return {chars[i] + chars[i + 1] for i in range(len(chars) - 1)}


def zscore(values: list[float]) -> list[float]:
    """组内 z-score；零方差时全返 0（而不是 NaN——NaN 会静默污染整个排序）。"""
    if not values:
        return []
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var)
    if std < 1e-12:
        return [0.0] * n
    return [(v - mean) / std for v in values]


def collect(args) -> dict:
    gold = json.loads(Path(args.gold_set).read_text(encoding="utf-8"))
    items = gold["items"] if isinstance(gold, dict) else gold
    import importlib.util

    spec = importlib.util.spec_from_file_location("_ask", Path(__file__).resolve().parent / "ask.py")
    ask = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ask)
    questions = ask._stratified_sample(items, args.limit)

    gen = release.Generation(args.generation)
    chunks = {str(r["chunk_id"]): r for r in read_jsonl(config.chunks_path()) if r.get("chunk_id")}
    index, index_meta = index_mod.load_index(gen.index_path)
    graph = json.loads(gen.graph_path.read_text(encoding="utf-8"))
    retriever = retrieve.Retriever(index, graph, {}, chunks=chunks, vocab=vocab_mod)

    # 向量：用于 cos 特征
    digest = (index_meta or {}).get("index_content_digest") or ""
    model = embedding.model_name()
    bin_path, meta_path = vector_store.vector_paths(vector_store._slug(model), digest)
    matrix, vmeta = vector_store.load_vectors(bin_path, meta_path)
    vec = vector_store.VectorIndex(matrix, vmeta)

    # 词表 IDF（用块正文的 bigram 频次估）
    print("统计 IDF …", flush=True)
    df: dict[str, int] = defaultdict(int)
    chunk_bigrams: dict[str, set[str]] = {}
    for cid, rec in chunks.items():
        bg = bigrams(str(rec.get("text") or ""))
        chunk_bigrams[cid] = bg
        for g in bg:
            df[g] += 1
    n_chunks = len(chunks) or 1

    def idf(g: str) -> float:
        return math.log((n_chunks + 1) / (1 + df.get(g, 0))) + 1.0

    out: list[dict] = []
    for i, item in enumerate(questions):
        q = str(item.get("question") or "")
        analysis = query_mod.analyze_query(
            q, entities=graph.get("entities") or [], aliases=graph.get("alias_index"),
            vocab=vocab_mod, context=retriever.query_context,
        )
        result = retriever.search(analysis, top_k=args.pool)
        hits = result["hits"]
        if not hits:
            out.append({"item": item, "rows": []})
            continue

        # 查询向量（一次）
        text = str(analysis.get("retrieval_query") or analysis.get("normalized_query") or q)
        qvec = embedding.embed_batch([text])[0]
        qb = bigrams(q)
        qb_idf_total = sum(idf(g) for g in qb) or 1.0

        # 问句实体名（用于 in_text）
        ent_names = set()
        for cand in analysis.get("entity_candidates") or []:
            eid = str(cand.get("entity_id") or "")
            rec = retriever.entities_by_id.get(eid) or {}
            if rec.get("name"):
                ent_names.add(str(rec["name"]))
            for a in rec.get("aliases") or []:
                t = a.get("text") if isinstance(a, dict) else a
                if str(t or "").strip():
                    ent_names.add(str(t).strip())

        rows = []
        for h in hits:
            cid = str(h["chunk_id"])
            rec = chunks.get(cid) or {}
            body = str(rec.get("text") or "")
            srcs = {s["channel"]: s["score"] for s in h.get("sources") or []}
            bg = chunk_bigrams.get(cid) or set()
            inter = qb & bg
            rows.append({
                "chunk_id": cid,
                "orig": float(h["score"]),
                "bm25": float(srcs.get(retrieve.SOURCE_BM25, 0.0)),
                "entity": float(srcs.get(retrieve.SOURCE_ENTITY, 0.0)),
                "speaker": float(srcs.get(retrieve.SOURCE_SPEAKER, 0.0)),
                "cos": float(vec.matrix[vec._pos[cid]] @ qvec) if cid in vec._pos else 0.0,
                "lex": len(inter) / len(qb) if qb else 0.0,
                "lex_idf": sum(idf(g) for g in inter) / qb_idf_total if qb else 0.0,
                "in_text": 1.0 if any(nm in body for nm in ent_names) else 0.0,
                "len": float(len(body)),
            })
        out.append({"item": item, "rows": rows})
        if (i + 1) % 40 == 0:
            print(f"  特征 {i + 1}/{len(questions)}", flush=True)
    return {"questions": out, "pool": args.pool}


def stratify(questions: list[dict], *, holdout_ratio: float) -> tuple[list[int], list[int]]:
    """按类别分层切分。同类别内每隔 k 个取一个进 holdout，保证两半类别比例一致。"""
    by_cat: dict[str, list[int]] = defaultdict(list)
    for idx, q in enumerate(questions):
        by_cat[str(q["item"].get("category") or "")].append(idx)
    train, hold = [], []
    for cat in sorted(by_cat):
        idxs = by_cat[cat]
        step = max(2, round(1 / holdout_ratio))
        for j, idx in enumerate(idxs):
            (hold if j % step == 0 else train).append(idx)
    return sorted(train), sorted(hold)


def evaluate(data: dict, idxs: list[int], weights: dict[str, float]) -> dict:
    """按给定权重重排并算指标。只统计有 gold 的题。"""
    r5 = r20 = rr = 0.0
    n = 0
    for idx in idxs:
        entry = data["questions"][idx]
        gold = set(str(c) for c in entry["item"].get("gold_chunk_ids") or [])
        if not gold or not entry["rows"]:
            continue
        rows = entry["rows"]
        # 特征组内 z-score（每题独立，去掉块间量纲差异）
        scored = []
        for f in FEATURES:
            zs = zscore([r[f] for r in rows])
            for r, z in zip(rows, zs):
                r.setdefault("_z", {})[f] = z
        for r in rows:
            total = 0.0
            for f in FEATURES:
                w = weights.get(f, 0.0)
                if w:
                    total += w * r["_z"][f]
            scored.append((r["chunk_id"], total))
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        ranked = [cid for cid, _ in scored]
        n += 1
        r5 += len(gold & set(ranked[:5])) / len(gold)
        r20 += len(gold & set(ranked[:20])) / len(gold)
        first = next((j for j, cid in enumerate(ranked) if cid in gold), None)
        rr += 0.0 if first is None else 1.0 / (first + 1)
    if not n:
        return {"r@5": 0.0, "r@20": 0.0, "mrr": 0.0, "n": 0}
    return {"r@5": round(r5 / n, 4), "r@20": round(r20 / n, 4), "mrr": round(rr / n, 4), "n": n}


def coordinate_ascent(data: dict, idxs: list[int], *, rounds: int = 2) -> dict[str, float]:
    """坐标上升：每次只调一个权重，取使 (mrr, r@20) 字典序最大的值。

    **主目标是 MRR**（上下文只装 4.7 块，头部名次比"多捞到但进不去"值钱），
    R@20 作为次级键——这跟前项目的选择相反，见模块 docstring。
    从 `orig=1, 其它=0` 起步（即基线），保证拟合结果不会比基线更差。
    """
    weights = {f: (1.0 if f == "orig" else 0.0) for f in FEATURES}
    best = evaluate(data, idxs, weights)
    print(f"  起点 orig=1  MRR {best['mrr']:.4f}  R@20 {best['r@20']:.4f}")
    for rnd in range(rounds):
        for f in FEATURES:
            trials = []
            for v in GRID:
                if v == weights[f]:
                    continue
                cand = dict(weights)
                cand[f] = v
                m = evaluate(data, idxs, cand)
                trials.append((m["mrr"], m["r@20"], v, m))
            if not trials:
                continue
            trials.sort(key=lambda t: (-t[0], -t[1]))
            top_mrr, top_r20, top_v, top_m = trials[0]
            if (top_mrr, top_r20) > (best["mrr"], best["r@20"]):
                weights[f] = top_v
                best = top_m
                print(f"  r{rnd + 1} {f:>8} → {top_v:<4g} MRR {best['mrr']:.4f}  R@20 {best['r@20']:.4f}")
    return weights


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="离线盲特征线性重排")
    ap.add_argument("--generation", default="g-20261031T035231Z-4eafae")
    ap.add_argument("--gold-set", default="data/goldset/gold-v1.json")
    ap.add_argument("--limit", type=int, default=320)
    ap.add_argument("--pool", type=int, default=30, help="每题取前 N 个候选做重排")
    ap.add_argument("--holdout-ratio", type=float, default=0.5)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--cache", default="data/rerank_features.json")
    ap.add_argument("--json")
    args = ap.parse_args(argv)

    cache = Path(args.cache)
    if cache.exists():
        print(f"复用已采集特征 {cache}")
        data = json.loads(cache.read_text(encoding="utf-8"))
    else:
        data = collect(args)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        print(f"特征已落盘 {cache}")

    qs = data["questions"]
    train, hold = stratify(qs, holdout_ratio=args.holdout_ratio)
    print(f"\n题 {len(qs)}  train {len(train)}  holdout {len(hold)}  pool {data['pool']}")

    base_all = evaluate(data, list(range(len(qs))), {f: (1.0 if f == "orig" else 0.0) for f in FEATURES})
    print(f"\n基线（orig=1，即生产排序）  全量 MRR {base_all['mrr']:.4f}  R@20 {base_all['r@20']:.4f}  R@5 {base_all['r@5']:.4f}  n={base_all['n']}")
    base_tr = evaluate(data, train, {f: (1.0 if f == "orig" else 0.0) for f in FEATURES})
    base_ho = evaluate(data, hold, {f: (1.0 if f == "orig" else 0.0) for f in FEATURES})
    print(f"  train  MRR {base_tr['mrr']:.4f}  R@20 {base_tr['r@20']:.4f}  n={base_tr['n']}")
    print(f"  holdout MRR {base_ho['mrr']:.4f}  R@20 {base_ho['r@20']:.4f}  n={base_ho['n']}")

    print("\n坐标上升（只在 train 上）：")
    weights = coordinate_ascent(data, train, rounds=args.rounds)

    tr = evaluate(data, train, weights)
    ho = evaluate(data, hold, weights)
    allm = evaluate(data, list(range(len(qs))), weights)
    print(f"\n拟合权重：{json.dumps({k: v for k, v in weights.items() if v}, ensure_ascii=False)}")
    print(f"  train   MRR {tr['mrr']:.4f} (Δ{tr['mrr'] - base_tr['mrr']:+.4f})  "
          f"R@20 {tr['r@20']:.4f} (Δ{tr['r@20'] - base_tr['r@20']:+.4f})")
    print(f"  holdout MRR {ho['mrr']:.4f} (Δ{ho['mrr'] - base_ho['mrr']:+.4f})  "
          f"R@20 {ho['r@20']:.4f} (Δ{ho['r@20'] - base_ho['r@20']:+.4f})")
    print(f"  全量    MRR {allm['mrr']:.4f} (Δ{allm['mrr'] - base_all['mrr']:+.4f})  "
          f"R@20 {allm['r@20']:.4f} (Δ{allm['r@20'] - base_all['r@20']:+.4f})")

    print("\n判据：holdout 的 MRR 必须涨，且 R@20 不能掉。")
    if ho["mrr"] > base_ho["mrr"] and ho["r@20"] >= base_ho["r@20"]:
        print("→ holdout 通过。可以进入下一步（在真实管线里实现并全量复测）。")
    else:
        print("→ holdout **未通过**。这是在没见过的那一半题上失效，说明权重是拟合出来的，"
              "不是真实信号。如实记录为否，不做下一步。")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({
            "generation": args.generation, "pool": data["pool"],
            "questions": len(qs), "train": len(train), "holdout": len(hold),
            "baseline": {"all": base_all, "train": base_tr, "holdout": base_ho},
            "weights": weights,
            "fitted": {"train": tr, "holdout": ho, "all": allm},
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
