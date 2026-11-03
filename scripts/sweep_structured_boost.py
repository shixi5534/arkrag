"""结构化字段加成：用 `speakers` / `title` / `chapter` 而不是字面子串。

## 为什么这条路比 RRF 有希望

RRF 失败的原因是「两条弱通道各投一票，把 bm25 的第 1 名压下去」。但融合本身不是
错的方向——`diagnose_channels.py` 显示 bm25 单路 gold 命中率 0.9367，说明候选池
已经不缺 gold，缺的是**能把 gold 往上抬的可靠信号**。

`entity` 通道现在是「问句里的实体名，在块正文里字面出现过」——这个信号很糙：
一个块提到「凯尔希」三次不代表它在讲凯尔希的这件事（`entity` 单路 MRR 只有 0.2255
就是证据）。

而 `chunks.jsonl` 里有**结构化字段**：

    speakers: ['可萝尔', '村民', '格拉尼', ...]   ← 这个块里谁在说话
    title: '日正当中'                              ← 场景标题
    chapter: 'a001'                                ← 章节号
    category: 'activities'                         ← 剧情线

`speakers` 尤其有价值：**如果问句问的对象在这个块的 speakers 里，那这个块几乎
必然相关**（剧情是对话体，说话人就是当事人）。这是比子串匹配强得多的信号，
而且是结构化精确匹配，不是模糊匹配。

## 加成公式

    分 = bm25_norm + 0.5 * entity_norm            （现状，不动）
        + W_SPEAKER * (问句实体命中 speakers 的比例)
        + W_TITLE   * (问句实体命中 title 的比例)

加成项按「命中实体的候选分之和 / 问句里最强候选分」归一，量级与 bm25_norm 可比。

## 纪律

* 只加**精确匹配**的结构化信号，不做模糊匹配（那又会回到 entity 通道的老问题）。
* 判据仍然是 R@20 与 MRR 同时不劣。
* 每个加成项单独扫，收益可分离——否则不知道是哪一项在起作用。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def metrics(hits_per_q: list[list[str]], questions: list[dict]) -> dict:
    r5, r20, r50, rr = [], [], [], []
    for hits, item in zip(hits_per_q, questions):
        gold = [str(c) for c in item.get("gold_chunk_ids") or []]
        if not gold:
            continue
        gs = set(gold)
        r5.append(len(gs & set(hits[:5])) / len(gs))
        r20.append(len(gs & set(hits[:20])) / len(gs))
        r50.append(len(gs & set(hits[:50])) / len(gs))
        first = next((i for i, cid in enumerate(hits) if cid in gs), None)
        rr.append(0.0 if first is None else 1.0 / (first + 1))
    n = len(r20) or 1
    return {"r@5": round(sum(r5) / n, 4), "r@20": round(sum(r20) / n, 4),
            "r@50": round(sum(r50) / n, 4), "mrr": round(sum(rr) / n, 4), "n": len(r20)}


def _speakers_of(rec: dict) -> set[str]:
    raw = rec.get("speakers")
    if isinstance(raw, list):
        return {str(s) for s in raw if str(s).strip()}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw.replace("'", '"'))
            if isinstance(parsed, list):
                return {str(s) for s in parsed if str(s).strip()}
        except (json.JSONDecodeError, ValueError):
            return set()
    return set()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="结构化字段加成扫描")
    ap.add_argument("--generation", default="g-20261031T035231Z-4eafae")
    ap.add_argument("--gold-set", default="data/goldset/gold-v1.json")
    ap.add_argument("--limit", type=int, default=320)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--json")
    args = ap.parse_args(argv)

    gold = json.loads(Path(args.gold_set).read_text(encoding="utf-8"))
    items = gold["items"] if isinstance(gold, dict) else gold
    import importlib.util

    spec = importlib.util.spec_from_file_location("_ask", Path(__file__).resolve().parent / "ask.py")
    ask = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ask)
    questions = ask._stratified_sample(items, args.limit)
    print(f"题 {len(questions)}  top_k={args.top_k}")

    gen = release.Generation(args.generation)
    chunks = {str(r["chunk_id"]): r for r in read_jsonl(config.chunks_path()) if r.get("chunk_id")}
    index, _meta = index_mod.load_index(gen.index_path)
    graph = json.loads(gen.graph_path.read_text(encoding="utf-8"))
    retriever = retrieve.Retriever(index, graph, {}, chunks=chunks, vocab=vocab_mod)
    depth = args.top_k * retrieve.CANDIDATE_POOL_MULT

    # 质量自检：speakers 覆盖了多少块
    with_speakers = sum(1 for r in chunks.values() if _speakers_of(r))
    print(f"块 {len(chunks)}，其中有 speakers 的 {with_speakers} ({with_speakers/len(chunks):.1%})")

    # ---- 预计算每条问句的：bm25 分、entity 分、speaker 命中、title 命中 ----
    bm25_all: list[dict[str, float]] = []
    entity_all: list[dict[str, float]] = []
    speaker_all: list[dict[str, float]] = []
    title_all: list[dict[str, float]] = []
    chapter_all: list[dict[str, float]] = []

    for i, item in enumerate(questions):
        q = str(item.get("question") or "")
        analysis = query_mod.analyze_query(
            q, entities=graph.get("entities") or [], aliases=graph.get("alias_index"),
            vocab=vocab_mod, context=retriever.query_context,
        )
        text = str(analysis.get("retrieval_query") or analysis.get("normalized_query") or "")
        bm25 = retrieve._norm_scores(index.search(text, top_k=depth)) if text else {}
        bm25_all.append(bm25)

        ent: dict[str, float] = {}
        names: dict[str, float] = {}   # 问句实体 → 该实体的候选权重
        for cand in analysis.get("entity_candidates") or []:
            eid = str(cand.get("entity_id") or "")
            rec = retriever.entities_by_id.get(eid) or {}
            terms = [str(rec.get("name") or cand.get("name") or "")]
            for a in rec.get("aliases") or []:
                t = a.get("text") if isinstance(a, dict) else a
                if str(t or "").strip():
                    terms.append(str(t).strip())
            for t in terms:
                names[t] = max(names.get(t, 0.0), float(cand.get("score") or 0.0))
            probe = " ".join(sorted({t for t in terms if t}))
            if not probe:
                continue
            w = float(cand.get("score") or 0.0)
            for cid, sc in retrieve._norm_scores(index.search(probe, top_k=depth)).items():
                ent[cid] = max(ent.get(cid, 0.0), sc * w)
        entity_all.append(ent)

        # 候选集合 = bm25 ∪ entity 的块（只在这些里加结构化分，避免全表扫描）
        cands = set(bm25) | set(ent)
        sp: dict[str, float] = {}
        ti: dict[str, float] = {}
        ch: dict[str, float] = {}
        if names and cands:
            max_w = max(names.values()) or 1.0
            for cid in cands:
                rec = chunks.get(cid) or {}
                spk = _speakers_of(rec)
                if spk:
                    hit = sum(w for t, w in names.items() if t in spk)
                    if hit:
                        sp[cid] = hit / max_w
                title = str(rec.get("title") or "")
                if title:
                    hit = sum(w for t, w in names.items() if t in title)
                    if hit:
                        ti[cid] = hit / max_w
                chapter = str(rec.get("chapter") or "")
                if chapter and any(t == chapter for t in names):
                    ch[cid] = 1.0 / max_w
        speaker_all.append(sp)
        title_all.append(ti)
        chapter_all.append(ch)
        if (i + 1) % 80 == 0:
            print(f"  通道 {i + 1}/{len(questions)}", flush=True)

    # 统计信号出现频率（判断加成有没有机会生效）
    print(f"\n问句命中 speakers 的题：{sum(1 for s in speaker_all if s)} / {len(questions)}")
    print(f"问句命中 title 的题：{sum(1 for t in title_all if t)} / {len(questions)}")

    def run(ws: float, wt: float, wc: float) -> dict:
        hits_per_q = []
        for bm25, ent, sp, ti, ch in zip(bm25_all, entity_all, speaker_all, title_all, chapter_all):
            merged: dict[str, float] = defaultdict(float)
            for cid, sc in bm25.items():
                merged[cid] += sc
            for cid, sc in ent.items():
                merged[cid] += retrieve.ENTITY_CHANNEL_WEIGHT * sc
            for cid, sc in sp.items():
                merged[cid] += ws * sc
            for cid, sc in ti.items():
                merged[cid] += wt * sc
            for cid, sc in ch.items():
                merged[cid] += wc * sc
            ranked = sorted(merged.items(), key=lambda kv: (-kv[1], kv[0]))
            hits_per_q.append([cid for cid, _ in ranked[: args.top_k]])
        return metrics(hits_per_q, questions)

    base = run(0.0, 0.0, 0.0)
    print(f"\nbaseline（现状公式，无结构化加成）  R@5 {base['r@5']:.4f}  R@20 {base['r@20']:.4f}  MRR {base['mrr']:.4f}")

    print(f"\n{'ws':>5} {'wt':>5} {'wc':>5} {'R@5':>8} {'R@20':>8} {'MRR':>8}  {'ΔR@20':>8} {'ΔMRR':>8}  判定")
    print("-" * 74)
    rows = []
    grid = [(w, 0.0, 0.0) for w in (0.1, 0.2, 0.3, 0.5, 0.8, 1.0)]
    grid += [(w, w, 0.0) for w in (0.2, 0.3, 0.5)]
    grid += [(w, 0.0, w) for w in (0.2, 0.3, 0.5)]
    for ws, wt, wc in grid:
        m = run(ws, wt, wc)
        d20, dm = round(m["r@20"] - base["r@20"], 4), round(m["mrr"] - base["mrr"], 4)
        both = d20 >= 0 and dm >= 0
        verdict = "★两者都不劣" if both else ("召回升/MRR降" if d20 > 0 else
                                          ("召回降/MRR升" if dm > 0 else "两者都降"))
        rows.append({"ws": ws, "wt": wt, "wc": wc, **m, "d_r@20": d20, "d_mrr": dm,
                     "both_ok": both, "verdict": verdict})
        print(f"{ws:>5.1f} {wt:>5.1f} {wc:>5.1f} {m['r@5']:>8.4f} {m['r@20']:>8.4f} "
              f"{m['mrr']:>8.4f}  {d20:>+8.4f} {dm:>+8.4f}  {verdict}")

    winners = [r for r in rows if r["both_ok"]]
    print("\n" + "=" * 74)
    if winners:
        best = max(winners, key=lambda r: (r["d_mrr"] + r["d_r@20"], r["mrr"]))
        print(f"结论：{len(winners)} 个组合两者都不劣。最好 ws={best['ws']} wt={best['wt']} "
              f"wc={best['wc']}  ΔR@20 {best['d_r@20']:+.4f}  ΔMRR {best['d_mrr']:+.4f}")
    else:
        print("结论：结构化字段加成也无法让两个指标同时不劣。speakers/title 命中不足或信号太弱。")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps({"generation": args.generation, "top_k": args.top_k,
                        "questions": len(questions), "baseline": base,
                        "speaker_q": sum(1 for s in speaker_all if s),
                        "title_q": sum(1 for t in title_all if t), "rows": rows},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
