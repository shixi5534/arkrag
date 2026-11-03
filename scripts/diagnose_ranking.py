"""诊断：gold 块的排名是被「同场景兄弟块挤占」还是「相关性判断本身不行」。

## 为什么区分这两件事

G 阶段实测已有一条线索：`q-dr-000001` 的 gold 是 `obt/main/level_st_17-01#001`，
它不在前 20，但**同一个 doc 的 `#002`/`#003` 都在前 20**。这不是数据问题，是排序问题。
但如果两个兄弟块把两个名额占掉，我们就得用不同的工具：

* 若 **兄弟块挤占** 是主因 → 需要「块级去重 / 每 doc 名额上限 / MMR 多样性」，
  这些都很便宜，而且顺便省下上下文预算（现在只装得下 4.7 块，重复块等于浪费）。
* 若 **gold 排名本身就低** → 那是相关性判断不够，需要更好的打分信号
  （proximity / 字段权重 / 重排器），代价高得多。

两者对应的解法完全不同，所以先量清楚比例。混在一起改会导致「改了也没动」，
因为改的机制根本没对上主因。

## 三个输出

1. `rank_hist`：gold 块在 top-100 里的名次分布（1–5 / 6–10 / 11–20 / 21–50 / 51–100 / 未出现）。
2. `sibling`：gold 未进前 N 时，**同 doc 的其它块**占了多少个前 N 名额，以及
   「同 doc 最佳块」的名次——后者代表「如果按 doc 算，这个题其实早就对了」。
3. `per_doc_cap`：模拟「每个 doc 最多占 k 个名额」后 R@20/MRR 的变化。
   这是纯后处理，不重新检索，所以能立刻看出收益上限。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arkrag import config, index as index_mod, query as query_mod, release, retrieve, vocab as vocab_mod  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def doc_of(chunk_id: str) -> str:
    """`obt/main/level_st_17-01#001` → `obt/main/level_st_17-01`。

    按最后一个 `#` 切，因为 doc 名里不含 `#`，而块后缀一定是 `#NNN`。
    """
    cid = str(chunk_id or "")
    cut = cid.rfind("#")
    return cid[:cut] if cut > 0 else cid


def metrics(hits_per_q: list[list[str]], questions: list[dict]) -> dict:
    r5, r20, rr = [], [], []
    for hits, item in zip(hits_per_q, questions):
        gold = [str(c) for c in item.get("gold_chunk_ids") or []]
        if not gold:
            continue
        gs = set(gold)
        r5.append(len(gs & set(hits[:5])) / len(gs))
        r20.append(len(gs & set(hits[:20])) / len(gs))
        first = next((i for i, cid in enumerate(hits) if cid in gs), None)
        rr.append(0.0 if first is None else 1.0 / (first + 1))
    n = len(r20) or 1
    return {"r@5": round(sum(r5) / n, 4), "r@20": round(sum(r20) / n, 4),
            "mrr": round(sum(rr) / n, 4), "n": len(r20)}


def cap_per_doc(hits: list[str], cap: int) -> list[str]:
    """每个 doc 最多留 cap 个块，保持原顺序。

    这是**纯后处理**：不重新检索、不改打分，所以它衡量的只是
    「重复块占掉的名额值多少」。若加上它 R@20 反而升，说明当前排序里
    兄弟块确实在白占位置（这也意味着上下文预算在被浪费）。
    """
    if cap <= 0:
        return list(hits)
    seen: dict[str, int] = defaultdict(int)
    out = []
    for cid in hits:
        d = doc_of(cid)
        if seen[d] >= cap:
            continue
        seen[d] += 1
        out.append(cid)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="gold 排名归因诊断")
    ap.add_argument("--generation", default="g-20261031T035231Z-4eafae")
    ap.add_argument("--gold-set", default="data/goldset/gold-v1.json")
    ap.add_argument("--limit", type=int, default=320)
    ap.add_argument("--pool", type=int, default=100, help="候选池深度，要比 top_k 深才能看到真实名次")
    args = ap.parse_args(argv)

    gold = json.loads(Path(args.gold_set).read_text(encoding="utf-8"))
    items = gold["items"] if isinstance(gold, dict) else gold
    import importlib.util

    spec = importlib.util.spec_from_file_location("_ask", Path(__file__).resolve().parent / "ask.py")
    ask = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ask)
    questions = ask._stratified_sample(items, args.limit)
    print(f"题 {len(questions)}  候选池 {args.pool}")

    gen = release.Generation(args.generation)
    chunks = {str(r["chunk_id"]): r for r in read_jsonl(config.chunks_path()) if r.get("chunk_id")}
    index, _meta = index_mod.load_index(gen.index_path)
    graph = json.loads(gen.graph_path.read_text(encoding="utf-8"))
    retriever = retrieve.Retriever(index, graph, {}, chunks=chunks, vocab=vocab_mod)

    # ---- 计算每条题在「深候选池」里的名次 ----
    ranks: list[int | None] = []          # gold 首个命中名次（1 起；None=池里没有）
    sibling_best: list[int | None] = []   # 同 doc 最佳名次
    sibling_slots: list[int] = []         # 前 20 里被非 gold 的同 doc 块占掉的名额数
    pools: list[list[str]] = []

    for i, item in enumerate(questions):
        q = str(item.get("question") or "")
        analysis = query_mod.analyze_query(
            q, entities=graph.get("entities") or [], aliases=graph.get("alias_index"),
            vocab=vocab_mod, context=retriever.query_context,
        )
        pool = [str(h["chunk_id"]) for h in retriever.search(analysis, top_k=args.pool)["hits"]]
        pools.append(pool)
        golds = [str(c) for c in item.get("gold_chunk_ids") or []]
        gold_set, gold_docs = set(golds), {doc_of(g) for g in golds}

        first = next((j for j, cid in enumerate(pool) if cid in gold_set), None)
        ranks.append(None if first is None else first + 1)

        sib_first = next(
            (j for j, cid in enumerate(pool) if doc_of(cid) in gold_docs and cid not in gold_set),
            None,
        )
        sibling_best.append(None if sib_first is None else sib_first + 1)

        slots = sum(
            1 for cid in pool[:20]
            if doc_of(cid) in gold_docs and cid not in gold_set
        )
        sibling_slots.append(slots)

        if (i + 1) % 80 == 0:
            print(f"  {i + 1}/{len(questions)}", flush=True)

    has_gold = [i for i, it in enumerate(questions) if it.get("gold_chunk_ids")]
    def bucket(r):
        if r is None:
            return "池中无"
        if r <= 5: return "1-5"
        if r <= 10: return "6-10"
        if r <= 20: return "11-20"
        if r <= 50: return "21-50"
        return "51-100"

    print("\n=== gold 块名次分布（候选池深度 %d，n=%d）===" % (args.pool, len(has_gold)))
    hist: dict[str, int] = defaultdict(int)
    for i in has_gold:
        hist[bucket(ranks[i])] += 1
    for key in ("1-5", "6-10", "11-20", "21-50", "51-100", "池中无"):
        n = hist.get(key, 0)
        print(f"  {key:<8} {n:>4} 题  {n / max(1, len(has_gold)):>6.1%}")

    print("\n=== 同 doc 兄弟块的影响 ===")
    missed = [i for i in has_gold if ranks[i] is None or ranks[i] > 20]
    stole = [sibling_slots[i] for i in missed]
    print(f"  gold 未进前 20 的题：{len(missed)}")
    print(f"  其中同 doc 兄弟块挤进前 20 的题：{sum(1 for s in stole if s > 0)}")
    print(f"  这些题里被兄弟块占掉的名额：平均 {sum(stole) / max(1, len(stole)):.2f} 个（P50 "
          f"{sorted(stole)[len(stole)//2] if stole else 0}，max {max(stole) if stole else 0}）")
    sib_ok = sum(1 for i in missed if sibling_best[i] and sibling_best[i] <= 20)
    print(f"  若按 doc 判定（同 doc 最佳块进前 20）这批题里有 {sib_ok} 题其实已命中")

    # ---- 按 doc 设名额上限，纯后处理 ----
    print("\n=== 每 doc 名额上限（纯后处理，不重新检索）===")
    print(f"{'cap':>5} {'R@5':>8} {'R@20':>8} {'MRR':>8}  说明")
    print("-" * 52)
    base_m = None
    for cap in (0, 1, 2, 3, 4, 5):
        capped = [cap_per_doc(p[: args.pool], cap) if cap else p[:20]
                  for p in pools]
        # cap=0 表示「不设上限」，但为了与生产口径一致仍截到 20
        capped = [c[:20] for c in capped]
        m = metrics(capped, questions)
        if cap == 0:
            base_m = m
            note = "生产口径（不做处理）"
        else:
            d20 = m["r@20"] - base_m["r@20"]
            dm = m["mrr"] - base_m["mrr"]
            note = f"ΔR@20 {d20:+.4f}  ΔMRR {dm:+.4f}"
        print(f"{cap if cap else '∞':>5} {m['r@5']:>8.4f} {m['r@20']:>8.4f} {m['mrr']:>8.4f}  {note}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
