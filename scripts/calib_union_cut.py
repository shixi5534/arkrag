"""在 dev 上定向量通道的触发阈值，holdout 只用来报告。

## 为什么要单独一个脚本做这件事

`DEFAULT_RATIO_CUT` 是**超参数**。在评测集上挑阈值，再在同一批题上报成绩，
就是把测试集当训练集用。本项目已因这个死过一次：盲特征重排 train MRR +0.0161、
holdout −0.0091，判据不过（`docs/i阶段实测.md` §5.2）。

所以分成两件独立的事：

* 本脚本：**只读 dev**，算出每个候选阈值下的（触发数 / 救回数 / 空转数），
  按**事先写死的规则**选出阈值；
* `scripts/eval_vector_union.py`：拿这个阈值去 holdout 报成绩。

## 选阈值的规则（先写死，再看数字）

    score(cut) = 救回数 − λ × 空转数        λ = 0.05

为什么罚空转：追加在尾部时 R@20 单调不降，但空转会白付一次 embed + 矩阵扫描
（实测 ~100ms/次）。λ=0.05 的意思是「1 次白跑换 20 次救回才划算」——这是
工程判断，不是从数据里拟合出来的，写在这里是为了可被质疑。

并列时选**更小的 cut**（更保守，触发更少）。

## 口径

  救回   base top-20 里没有 gold，而向量 top-k 里有 gold
  空转   base top-20 里已经有 gold（触发也不会提升 R@20，只付延迟）
  漏检   base 没有 gold、向量也没有（触发与否都救不回来）

注意「救回」不等于「答对」：救回的块追加在第 21 名及以后，而 `build_context`
的预算只装 4–6 个块，所以它多半进不了上下文。这个落差必须由
`eval_vector_union.py` 的三层指标如实报出来，不能在这里用「救回」搪塞。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import embedding  # noqa: E402
from arkrag import pipeline as pipeline_mod  # noqa: E402
from arkrag import query as query_mod  # noqa: E402
from arkrag import vector_store  # noqa: E402
from arkrag import vector_union as union_mod  # noqa: E402
from arkrag import vocab as vocab_mod  # noqa: E402
from arkrag.retrieve import Retriever  # noqa: E402

#: 空转的代价系数（见模块头「选阈值的规则」）
EMPTY_PENALTY = 0.05


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generation", default="g-20261103T132152Z-06a741")
    ap.add_argument("--gold-set", default="data/goldset/gold-v2.json")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--vector-k", type=int, default=union_mod.DEFAULT_VECTOR_K)
    ap.add_argument("--json", default="data/calib_union_cut.json")
    args = ap.parse_args()

    loaded = pipeline_mod.load_generation(args.generation)
    graph, chunks = loaded["graph"], loaded["chunks"]
    digest = (loaded["index_meta"] or {}).get("index_content_digest") or ""
    slug = vector_store._slug(embedding.model_name())
    bin_path, meta_path = vector_store.vector_paths(slug, digest)
    if not Path(bin_path).exists():
        raise SystemExit(f"[2] 缺向量产物 {bin_path}")
    matrix, vmeta = vector_store.load_vectors(bin_path, meta_path)
    vec = vector_store.VectorIndex(matrix, vmeta)

    retriever = Retriever(loaded["index"], graph, loaded["adjacency"],
                          chunks=chunks, assertions=loaded["assertions"],
                          vocab=vocab_mod)
    entities = graph.get("entities") or []

    gold = json.loads((ROOT / args.gold_set).read_text(encoding="utf-8"))
    items = [g for g in (gold.get("items") or [])
             if str(g.get("split") or "") == args.split and g.get("gold_chunk_ids")]
    print(f"split={args.split} n={len(items)}", flush=True)

    rows = []
    for i, item in enumerate(items, 1):
        q = str(item.get("question") or "")
        gset = {str(c) for c in item["gold_chunk_ids"]}
        qa = query_mod.analyze_query(q, entities=entities,
                                     aliases=graph.get("alias_index"),
                                     vocab=vocab_mod)
        hits = retriever.search(qa, top_k=args.top_k)["hits"]
        base_hit = any(str(h.get("chunk_id")) in gset for h in hits)
        qv = embedding.embed_batch([q])[0]
        vids = [str(c) for c, _ in vec.search(qv, top_k=args.vector_k)]
        vec_hit = any(c in gset for c in vids)
        rows.append({
            "qid": str(item.get("question_id") or i),
            "ratio": union_mod.ratio_top_to_20(hits),
            "base_hit": base_hit,
            "vec_hit": vec_hit,
            "rescued": (not base_hit) and vec_hit,
        })
        if i % 50 == 0 or i == len(items):
            print(f"  {i}/{len(items)}", flush=True)

    n = len(rows)
    rescues = [r for r in rows if r["rescued"]]
    print(f"base 命中 {sum(1 for r in rows if r['base_hit'])}/{n}；"
          f"可救回 {len(rescues)}/{n}", flush=True)

    # 候选阈值：所有实际出现的 ratio 值（去重升序），外加 inf 兜底
    cands = sorted({r["ratio"] for r in rows})
    table = []
    for cut in cands + [float("inf")]:
        trig = [r for r in rows if r["ratio"] <= cut]
        empty = sum(1 for r in trig if r["base_hit"])
        got = sum(1 for r in trig if r["rescued"])
        table.append({
            "cut": None if cut == float("inf") else round(cut, 6),
            "triggered": len(trig),
            "rescued": got,
            "empty": empty,
            "score": round(got - EMPTY_PENALTY * empty, 6),
        })

    # 规则：score 最大；并列取更小 cut（更保守）。inf 视为最大 cut。
    best = None
    for row in table:
        key = (row["score"], -(row["cut"] if row["cut"] is not None else 1e18))
        if best is None or key > (best["score"],
                                  -(best["cut"] if best["cut"] is not None else 1e18)):
            best = row

    # 只要「能救回全部可救回题」的阈值里最小的那个也能被选出来更好解读，
    # 这里两个都记下来，选阈值仍按上面写死的规则。
    full = [row for row in table if row["rescued"] == len(rescues)]
    smallest_full = min(full, key=lambda r: r["triggered"]) if full else None

    out = {
        "generation": args.generation,
        "gold_set": args.gold_set,
        "split": args.split,
        "n": n,
        "base_hit": sum(1 for r in rows if r["base_hit"]),
        "vector_hit": sum(1 for r in rows if r["vec_hit"]),
        "rescuable": len(rescues),
        "rule": f"max(rescued - {EMPTY_PENALTY} * empty), tie -> smaller cut",
        "chosen": best,
        "smallest_that_gets_all": smallest_full,
        "table": table,
        "rows": rows,
    }
    path = ROOT / args.json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n选中 cut={best['cut']} 触发 {best['triggered']}/{n} "
          f"救回 {best['rescued']}/{len(rescues)} 空转 {best['empty']} "
          f"score={best['score']}")
    if smallest_full:
        print(f"（能救回全部的最小触发：cut={smallest_full['cut']} "
              f"触发 {smallest_full['triggered']}/{n} 空转 {smallest_full['empty']}）")
    print(f"写出 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
