"""L6 建索引：词法倒排 + 别名倒排 + 图邻接，并做新鲜度三查与检索自检。

与 L5 同一条纪律：不许在残缺输入上判 PASS。这里多做一层——**检索自检**。
倒排条目数对得上不代表检索对得上（切词错了、slot 串了、postings 写盘丢了，
计数都还是对的，但查询永远查不回来）。所以随机取若干「实体名 → 含该实体的块」，
拿名字去查，看那块能不能被捞回来；捞不回来就是索引坏了。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arkrag import cache, config, extract, gates, index  # noqa: E402
from arkrag.agent_tasks import AGENT_MODEL  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只索引前 N 块（调试用，非 0 时门会 FAIL）")
    ap.add_argument("--probes", type=int, default=200, help="检索自检抽样条数")
    ap.add_argument("--no-save", action="store_true", help="只算不写盘")
    args = ap.parse_args()

    chunks = [c for c in extract.load_chunks() if config.in_t1(c["category"])]
    missing = [c["chunk_id"] for c in chunks if not cache.is_cached_for_chunk(c, AGENT_MODEL)]
    print(f"T1 范围 {len(chunks)} 块，已入库 {len(chunks) - len(missing)}，缺 {len(missing)}")
    if missing:
        print("拒绝建索引：仍有块未抽取。索引落后于语料时不会报错，只会让新内容检索不到。",
              file=sys.stderr)
        return 2

    if args.limit:
        chunks = sorted(chunks, key=lambda c: c["chunk_id"])[: args.limit]

    graph_path = config.graph_path()
    if not graph_path.exists():
        print(f"拒绝建索引：{graph_path} 不存在，先跑 run_l5.py", file=sys.stderr)
        return 2
    graph = json.loads(graph_path.read_text("utf-8"))
    corpus_meta = json.loads(config.corpus_meta_path().read_text("utf-8"))
    corpus_fp = corpus_meta["corpus_fp"]

    print("建词法倒排……")
    idx, meta = index.build(chunks, graph, corpus_fp)
    print(f"倒排词条 {len(idx.postings):,}  文档 {idx.size:,}")

    # 检索自检两组探针：
    #  (a) token 探针——取该块里 df 最小的词回查该块。df=1 的词只出现在这一块，
    #      所以只要倒排正确就必定排第一。这是确定性回环，不命中即索引坏。
    #  (b) name 探针——拿实体名当查询捞回来源块，只统计手感，不判定（见 gates.py 注释）。
    rng = random.Random(20261029)
    by_id = {c["chunk_id"]: c for c in chunks}
    probes: list[dict] = []

    sample = rng.sample(chunks, min(args.probes, len(chunks)))
    for rec in sample:
        toks = [t for t in index.tokenize(f"{rec.get('title','')} {rec.get('chapter','')} {rec['text']}")
                if len(t) >= 2]
        rare = sorted(toks, key=lambda t: len(idx.postings.get(t, ())))
        picked = next((t for t in rare if len(idx.postings.get(t, ())) == 1), None)
        if picked is None:
            continue
        ranked = idx.search(picked, top_k=20)
        rank = next((i + 1 for i, (cid, _) in enumerate(ranked) if cid == rec["chunk_id"]), None)
        probes.append({"kind": "token", "query": picked, "chunk_id": rec["chunk_id"], "rank": rank})

    pool = [
        e for e in graph["entities"]
        if e.get("chunks") and any(cid in by_id for cid in e["chunks"]) and 2 <= len(e["name"]) <= 12
    ]
    for ent in rng.sample(pool, min(args.probes, len(pool))):
        want = next(cid for cid in ent["chunks"] if cid in by_id)
        ranked = idx.search(ent["name"], top_k=20)
        rank = next((i + 1 for i, (cid, _) in enumerate(ranked) if cid == want), None)
        probes.append({"kind": "name", "query": ent["name"], "chunk_id": want, "rank": rank})

    tok = [p for p in probes if p["kind"] == "token"]
    nm = [p for p in probes if p["kind"] == "name"]
    print(f"检索自检 token {sum(1 for p in tok if p['rank'])}/{len(tok)}  "
          f"name {sum(1 for p in nm if p['rank'])}/{len(nm)}")

    run = gates.l6_gates(meta, graph, len(chunks), corpus_fp, probes)
    gates.write_report(run, config.gates_path(), "L6")
    for g in run.gates:
        print(f"  [{g.verdict}] {g.name}  measured={g.measured}")

    if args.no_save:
        print("--no-save：未写索引文件")
        return 0
    index_path = config.index_path()
    index.save_index(idx, meta, index_path)
    size_mb = index_path.stat().st_size / 1e6
    print(f"已写 {index_path} ({size_mb:.1f} MB)  corpus_fp={meta['corpus_fp']} "
          f"graph_fp={meta['graph_fp']}")

    # 回读自检：写盘后必须能原样读回来，否则等于没写。
    reloaded, meta2 = index.load_index(index_path)
    print(f"回读校验 文档 {reloaded.size:,}（原 {idx.size:,}）  词条 {len(reloaded.postings):,}"
          f"  一致={reloaded.size == idx.size and len(reloaded.postings) == len(idx.postings)}")

    fails = [g for g in run.gates if g.verdict == "FAIL"]
    print(f"FAIL {len(fails)} / 共 {len(run.gates)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
