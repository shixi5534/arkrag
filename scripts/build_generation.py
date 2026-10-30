"""事务式构建 + 回读校验 + 原子发布（计划 C1 / C2）。

一条命令做完一件事：**要么得到一个已发布的 Generation，要么线上一个字节都没变。**

## 旧流程为什么会发布失败版本

`run_l5.py` 直接写 `data/arkrag/graph.json`，`run_l6.py` 直接写
`data/arkrag/index/lexical.json`，门禁是**在写完之后**才算的。所以：

* 门禁 FAIL 时，坏产物已经在正式路径上了，下一轮 L6 会读到它；
* 中途崩溃时，图是新的、索引是旧的，两者指纹对不上，但没人拦；
* 没有"上一代"，想退回只能重跑几小时。

## 这里的顺序（不可调换）

    固定输入快照 -> 建候选（写进 generations/<gid>/）
      -> 算门禁（写进同一个 generation 目录）
      -> 回读候选产物并核摘要 / 固定探针重放
      -> 全部硬门通过才 READY
      -> 原子替换 current.json

构建期间的任何一步失败，`current.json` 都不会被碰。门禁 FAIL 的世代留在盘上
（标 FAILED，附原因），用来事后追"那一代坏在哪"——而不是删掉。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arkrag import cache, config, extract, gates, graph, index, release  # noqa: E402
from arkrag.agent_tasks import AGENT_MODEL  # noqa: E402


def build_probes(idx, chunks: list[dict], built: dict, *, count: int = 200) -> list[dict]:
    """检索自检探针：确定性回环（token）+ 手感申报（name）。

    token 探针取该块里 df=1 的词回查来源块——df=1 只出现在一块，倒排正确就必排第一，
    不命中即索引坏。name 探针拿实体名查，常见名捞不回来源块是排序的正常表现，
    只申报不判定（见 gates.l6_gates 注释）。
    """
    import random

    rng = random.Random(20261029)
    by_id = {c["chunk_id"]: c for c in chunks}
    probes: list[dict] = []

    for rec in rng.sample(chunks, min(count, len(chunks))):
        toks = [
            t
            for t in index.tokenize(f"{rec.get('title','')} {rec.get('chapter','')} {rec['text']}")
            if len(t) >= 2
        ]
        rare = sorted(toks, key=lambda t: len(idx.postings.get(t, ())))
        picked = next((t for t in rare if len(idx.postings.get(t, ())) == 1), None)
        if picked is None:
            continue
        ranked = idx.search(picked, top_k=20)
        rank = next((i + 1 for i, (cid, _) in enumerate(ranked) if cid == rec["chunk_id"]), None)
        probes.append({"kind": "token", "query": picked, "chunk_id": rec["chunk_id"], "rank": rank})

    pool = [
        e
        for e in built.get("entities") or []
        if e.get("chunks") and any(cid in by_id for cid in e["chunks"]) and 2 <= len(e["name"]) <= 12
    ]
    for ent in rng.sample(pool, min(count, len(pool))):
        want = next(cid for cid in ent["chunks"] if cid in by_id)
        ranked = idx.search(ent["name"], top_k=20)
        rank = next((i + 1 for i, (cid, _) in enumerate(ranked) if cid == want), None)
        probes.append({"kind": "name", "query": ent["name"], "chunk_id": want, "rank": rank})
    return probes


def _fails(*runs) -> list[gates.Gate]:
    out = []
    for run in runs:
        out.extend([g for g in run.gates if g.verdict == "FAIL"])
    return out


def build(args: argparse.Namespace) -> int:
    t0 = time.time()
    run = release.Run()
    run.event("start", "开始构建候选 Generation", source="T1")

    # ---- 1. 固定输入快照 -------------------------------------------------
    chunks = [c for c in extract.load_chunks() if config.in_t1(c["category"])]
    missing = [c["chunk_id"] for c in chunks if not cache.is_cached_for_chunk(c, AGENT_MODEL)]
    run.event("snapshot", "读取块清单", chunks=len(chunks), missing=len(missing))
    if missing:
        run.error("snapshot", "仍有块未抽取，拒绝构建", f"{len(missing)} 块缺失")
        run.finish("failed", reason="missing_extractions", missing=len(missing))
        print(
            f"拒绝构建：{len(missing)} 块未抽取（示例 {missing[:3]}）。"
            "在残缺输入上建出来的图过不了任何一条语义门。",
            file=sys.stderr,
        )
        return 2

    records_list = graph.load_records(chunks)
    records = {r["chunk_id"]: r["raw"] for r in records_list}
    corpus_meta = json.loads((config.DATA_DIR / "corpus_meta.json").read_text("utf-8"))
    corpus_fp = corpus_meta["corpus_fp"]
    snapshot = release.capture_inputs(chunks=chunks, records=records, source_fp=corpus_fp)
    run.event(
        "snapshot",
        "输入快照已固定",
        chunk_manifest_hash=snapshot.chunk_manifest_hash[:16],
        extraction_manifest_hash=snapshot.extraction_manifest_hash[:16],
        git_revision=snapshot.git_revision,
    )

    # ---- 2. 建候选（全部写进 generations/<gid>/，不碰正式路径）-------------
    parent = (release.read_current() or {}).get("generation_id", "")
    gen = release.Generation(release.new_generation_id())
    gen.create(snapshot=snapshot, run_id=run.run_id, parent=parent)
    gen.advance(release.BUILDING)
    run.event("build", "候选目录已就位", generation_id=gen.generation_id, path=str(gen.dir))

    built = graph.build_graph(chunks, records_list)
    graph.attach_corpus_fp(built, corpus_fp)

    try:
        graph.save(built, gen.graph_path)
    except RuntimeError as exc:
        gen.fail("graph_save_refused", detail=str(exc))
        run.error("build", "建图落盘被拒", str(exc))
        run.finish("failed", reason="graph_save_refused", generation_id=gen.generation_id)
        print(f"建图失败：{exc}", file=sys.stderr)
        return 1

    idx, idx_meta = index.build(chunks, built, corpus_fp)
    probes = build_probes(idx, chunks, built, count=args.probes)
    index.save_index(idx, idx_meta, gen.index_path)
    stats = built["stats"]
    run.event(
        "build",
        "图与索引已写入候选目录",
        entities=stats["entities"],
        edges=stats["edges"],
        postings=len(idx.postings),
    )

    # ---- 3. 门禁（写进候选目录自己的 reports/）----------------------------
    l4 = gates.l4_gates(records_list, built)
    l5 = gates.l5_gates(built, chunks, corpus_fp)
    l6 = gates.l6_gates(idx_meta, built, len(chunks), corpus_fp, probes)
    for run_ in (l4, l5, l6):
        gates.write_report(run_, gen.gate_report_path, run_.gates[0].stage)

    hard_fails = _fails(l4, l5, l6)
    run.event(
        "gate",
        "门禁计算完成",
        fails=len(hard_fails),
        gates=len(l4.gates) + len(l5.gates) + len(l6.gates),
        failed_names=[g.name for g in hard_fails[:10]],
    )
    if hard_fails:
        detail = "; ".join(f"{g.name}={g.measured}" for g in hard_fails[:8])
        gen.fail("gate_fail", detail=detail)
        run.error("gate", f"门禁 FAIL {len(hard_fails)} 项，拒绝发布", detail)
        run.finish("failed", reason="gate_fail", generation_id=gen.generation_id)
        print(f"门禁 FAIL {len(hard_fails)} 项，未发布（候选留在 {gen.dir}）：", file=sys.stderr)
        for g in hard_fails:
            print(f"  FAIL {g.name}  实测[{g.measured}]", file=sys.stderr)
        return 1

    # ---- 4. 回读候选产物并核身份与摘要 ------------------------------------
    gen.advance(
        release.VALIDATING,
        graph_build_fp=built.get("fingerprint", ""),
        graph_artifact_id=release.sha256_file(gen.graph_path),
        index_artifact_id=release.sha256_file(gen.index_path),
        index_build_fp=idx_meta.get("index_build_fp", ""),
        index_content_digest=idx_meta.get("index_content_digest", ""),
        probe_digest=idx_meta.get("probe_digest", ""),
        gate_report_hash=release.sha256_file(gen.gate_report_path),
        stats={"entities": stats["entities"], "edges": stats["edges"], "isolated": stats["isolated"]},
    )
    # 真回读一次：读回来重算摘要，不看内存里那份
    reloaded, reloaded_meta = index.load_index(gen.index_path)
    replay = index.content_digest(reloaded)
    replay_probe = index.probe_digest(reloaded)
    run.event(
        "verify",
        "回读候选产物",
        docs=reloaded.size,
        postings=len(reloaded.postings),
        content_digest_match=replay == idx_meta["index_content_digest"],
        probe_digest_match=replay_probe == idx_meta["probe_digest"],
    )
    if replay != idx_meta["index_content_digest"] or reloaded.size != idx.size:
        gen.fail("readback_content_mismatch", detail=f"docs {reloaded.size} vs {idx.size}")
        run.error("verify", "回读内容与内存计算不一致，拒绝发布", "")
        run.finish("failed", reason="readback_content_mismatch", generation_id=gen.generation_id)
        print("回读不一致：写下的索引读回来内容变了，拒绝发布。", file=sys.stderr)
        return 1

    problems = release.verify_generation(gen)
    if problems:
        gen.fail("verify_failed", detail="; ".join(problems[:8]))
        run.error("verify", "Manifest 与产物不一致，拒绝发布", "; ".join(problems[:8]))
        run.finish("failed", reason="verify_failed", generation_id=gen.generation_id)
        print(f"发布前校验失败 {len(problems)} 项：", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    # ---- 5. 全部通过才 READY，再发布 ---------------------------------------
    gen.advance(release.READY)
    if args.no_publish:
        run.event("ready", "已到 READY，按 --no-publish 未切换指针", generation_id=gen.generation_id)
        run.finish("ready", generation_id=gen.generation_id)
        print(f"候选已 READY（未发布）：{gen.generation_id}")
        print(f"  目录 {gen.dir}")
        print(f"  发布：py -3 scripts\\release_cli.py publish {gen.generation_id}")
        return 0

    pointer = release.publish(gen, run=run)
    elapsed = time.time() - t0
    run.finish("published", generation_id=gen.generation_id, elapsed_seconds=round(elapsed, 1))

    print(f"已发布 {gen.generation_id}（上一代 {pointer['previous_generation'] or '(无)'}）")
    print(f"  实体 {stats['entities']}  边 {stats['edges']}  块 {len(chunks)}  "
          f"词条 {len(idx.postings):,}")
    print(f"  graph={pointer['graph_artifact_id'][:16]}  index={pointer['index_artifact_id'][:16]}")
    print(f"  门禁 {len(l4.gates) + len(l5.gates) + len(l6.gates)} 项全 PASS  "
          f"耗时 {elapsed:.1f}s  run={run.run_id}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="事务式构建并原子发布一代产物")
    ap.add_argument("--probes", type=int, default=200, help="检索自检抽样条数")
    ap.add_argument("--no-publish", action="store_true", help="只构建到 READY，不切换指针")
    return build(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
