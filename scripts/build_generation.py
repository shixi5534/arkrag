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

from arkrag import cache, config, extract, gates, graph, index, release, source  # noqa: E402
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


def _load_scenes() -> list[dict]:
    """读场景（只为算 scene_fingerprint，算不出就是空列表，不影响构建）。"""
    path = config.scenes_path()
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


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
    # 覆盖不全时只有一条路可走：建一个**明确标着 incomplete 的候选**，永不发布。
    # 不允许"补几个块让它看起来完整"——那是拿别的块的证据顶替。
    allow_incomplete = bool(getattr(args, "allow_incomplete", False))
    if allow_incomplete and not getattr(args, "no_publish", False):
        print("--allow-incomplete 必须与 --no-publish 同时使用：覆盖不全的世代不许发布。", file=sys.stderr)
        run.finish("failed", reason="allow_incomplete_requires_no_publish")
        return 2
    if missing and not allow_incomplete:
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
    corpus_meta = json.loads(config.corpus_meta_path().read_text("utf-8"))
    corpus_fp = corpus_meta["corpus_fp"]
    # 来源状态：权威是验证记录（`source.status_for_build()` 读的那份），
    # `corpus_meta.json` 里的同名字段只是 L1 抄下来的副本，只作兜底与交叉核对。
    # 完整理由见 `source.resolve_for_build` 的文档字符串——简版是：
    # 验证记录是**产出验证的地方**写的，下游缓存的副本不该反向当权威；
    # 而这份冻结的 corpus_meta.json（153 B，D4 之前的产物）压根没有这个字段，
    # 旧代码 `corpus_meta.get("source_status", "source_unverified")` 于是把
    # 一份实际已验证的语料判成未知来源、拒绝构建。缺字段 = 这个文件里没写，
    # 不等于"没验证过"。
    resolved = source.resolve_for_build(corpus_meta)
    source_status = resolved["status"]
    for line in resolved["conflicts"]:
        # 不一致不许静默：打出来、写进 run 日志，并已按更保守的一侧取值。
        print(f"警告：{line}", file=sys.stderr)
    run.event(
        "snapshot",
        "来源状态已裁定",
        source_status=source_status,
        authority_status=resolved["authority_status"],
        authority_path=resolved["authority_path"],
        authority_present=resolved["authority_present"],
        corpus_meta_copy=resolved["copy_status"] or "(无字段)",
        conflicts=len(resolved["conflicts"]),
    )
    if not resolved["verified"]:
        # 早拒：来源身份未知时建图要跑几分钟，且下游 verify_generation 反正会拦。
        # 在这里停，是把"为什么停"讲在花掉那几分钟之前。
        run.error(
            "snapshot",
            "来源未验证，拒绝构建",
            f"status={source_status} 权威={resolved['authority_path']}",
        )
        # reason 记实际状态而不是固定标签：如果状态是 "bogus" 这种笔误，
        # 记成 "source_unverified" 就把"写错了"和"确实没验证"混成一件事了。
        run.finish(
            "failed",
            reason=f"source_not_verified:{source_status}",
            source_status=source_status,
        )
        print(
            f"拒绝构建：来源状态 {source_status}（权威记录 {resolved['authority_path']}"
            f"{'' if resolved['authority_present'] else '，该文件不存在'}）。"
            "来源身份未知时建出来的图不得冒充 Release（计划 D4）。",
            file=sys.stderr,
        )
        return 2
    # 场景层身份：块内容没变但标题/章节/node_id 变了时，只有这一层能看出来
    scenes = _load_scenes()
    snapshot = release.capture_inputs(
        chunks=chunks,
        records=records,
        source_fp=corpus_fp,
        scenes=scenes,
        source_status=source_status,
    )
    run.event(
        "snapshot",
        "输入快照已固定",
        source_status=snapshot.source_status,
        scene_fp=snapshot.scene_fingerprint,
        chunk_fp=snapshot.chunk_fingerprint,
        chunk_manifest_hash=snapshot.chunk_manifest_hash[:16],
        extraction_manifest_hash=snapshot.extraction_manifest_hash[:16],
        prompt_sha256=snapshot.versions.get("prompt_sha256", "")[:16],
        vocab_sha=snapshot.versions.get("vocab_sha", ""),
        git_revision=snapshot.git_revision,
    )

    # ---- 2. 建候选（全部写进 generations/<gid>/，不碰正式路径）-------------
    parent = (release.read_current() or {}).get("generation_id", "")
    gen = release.Generation(release.new_generation_id())
    gen.create(snapshot=snapshot, run_id=run.run_id, parent=parent)
    # 覆盖申报写进 Manifest，让「这一代覆盖了哪些块」成为可核对的事实，
    # 而不是留在构建日志里。缺块时 release.publish() 会据此硬拒。
    gen.update(
        coverage_status=release.COVERAGE_INCOMPLETE if missing else release.COVERAGE_COMPLETE,
        coverage_declared=len(chunks),
        coverage_covered=len(chunks) - len(missing),
        coverage_missing=list(missing),
    )
    gen.advance(release.BUILDING)
    run.event("build", "候选目录已就位", generation_id=gen.generation_id, path=str(gen.dir))

    # 图模型分派（E 阶段）。默认 legacy，旧路径的行为逐字节不变：
    # `graph.build_graph` 一行没改，`ark-rag-1` 产物还是原来那份。
    graph_model = str(getattr(args, "graph_model", "legacy") or "legacy")
    ident = None
    if graph_model == "identity":
        ident = graph.build_identity_graph(
            chunks,
            records_list,
            extraction_meta={
                "extraction_manifest_hash": snapshot.extraction_manifest_hash,
                "generation_id": gen.generation_id,
            },
        )
        built = ident["graph"]
    else:
        built = graph.build_graph(chunks, records_list)

    graph.attach_corpus_fp(
        built,
        corpus_fp,
        identities={
            "source_fp": snapshot.source_fingerprint,
            "source_status": snapshot.source_status,
            "scene_fp": snapshot.scene_fingerprint,
            "chunk_fp": snapshot.chunk_fingerprint,
            "chunk_manifest_hash": snapshot.chunk_manifest_hash,
            "extraction_manifest_hash": snapshot.extraction_manifest_hash,
        },
    )

    try:
        graph.save(built, gen.graph_path)
    except RuntimeError as exc:
        gen.fail("graph_save_refused", detail=str(exc))
        run.error("build", "建图落盘被拒", str(exc))
        run.finish("failed", reason="graph_save_refused", generation_id=gen.generation_id)
        print(f"建图失败：{exc}", file=sys.stderr)
        return 1

    # ark-rag-3 的两份独立产物。写在 graph.json **之后**，因为它们的身份要由
    # 这一代自己的 Manifest 记下来；写失败就整代失败，不许留"图在、断言缺"的半成品。
    if ident is not None:
        try:
            graph.save_assertions(ident["assertions"], gen.assertion_path)
            graph.save_adjacency(ident["adjacency"], gen.adjacency_path)
        except RuntimeError as exc:
            gen.fail("identity_sidecar_refused", detail=str(exc))
            run.error("build", "断言/邻接落盘被拒", str(exc))
            run.finish("failed", reason="identity_sidecar_refused", generation_id=gen.generation_id)
            print(f"断言/邻接写盘失败：{exc}", file=sys.stderr)
            return 1

    # 索引绑定的是**图产物的内容摘要**，不是图配置指纹（计划 D3）：
    # 同一份配置重建两次图，指纹相同而产物未必相同，只有内容摘要能答"这份倒排对的是哪份图"。
    graph_artifact_id = release.sha256_file(gen.graph_path)
    idx, idx_meta = index.build(
        chunks, built, corpus_fp, graph_artifact_id=graph_artifact_id
    )
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

    # ark-rag-3 的三组 E 门：断言可回溯 / 实体身份 / 邻接方向。
    #
    # 必须**并进同 stage 的那个 GateRun**再落盘，不能各自 write_report 一次：
    # `gates.write_report` 是按 stage 整段替换的（`existing = [r for r in existing
    # if r["stage"] != stage]`），L4.12 与 L4.1 同属 stage "L4"，分两次写就是
    # 后写的那组把前一组从报告里抹掉。实测第一次真跑时 reports/gates.json 只剩
    # 7 行、显示"7 项全 PASS"，其实 L4/L5/L6 主组 30 项全被覆盖没了——
    # 门禁报告少一半还全绿，比门禁红更危险。
    gate_runs = [l4, l5, l6]
    if ident is not None:
        for target, extra in (
            (l4, gates.l4_identity_gates(built)),
            (l5, gates.l5_evidence_gates(built, ident["assertions"], ident["edges"], chunks)),
            (l6, gates.l6_direction_gates(ident["adjacency"], ident["edges"])),
        ):
            for gate in extra.gates:
                target.add(gate)

    for run_ in gate_runs:
        gates.write_report(run_, gen.gate_report_path, run_.gates[0].stage)

    hard_fails = _fails(*gate_runs)
    run.event(
        "gate",
        "门禁计算完成",
        fails=len(hard_fails),
        gates=sum(len(r.gates) for r in gate_runs),
        graph_model=graph_model,
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
    # ark-rag-3 的六个 Manifest 字段：图/断言/邻接三个 artifact 摘要 + 三个计数。
    # 只在 identity 分支写，legacy 分支的 Manifest 形态一个字段都不多——
    # 旧代的 Manifest 已经进过发布指针与冻结校验，补字段等于改旧产物的形状。
    identity_fields: dict = {}
    if ident is not None:
        identity_fields = {
            "graph_schema": config.GRAPH_SCHEMA_IDENTITY,
            "assertion_artifact_id": release.sha256_file(gen.assertion_path),
            "adjacency_artifact_id": release.sha256_file(gen.adjacency_path),
            "assertion_count": len(ident["assertions"]),
            "entity_count": len(built.get("entities") or []),
            "edge_count": len(built.get("edges") or []),
        }

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
        **identity_fields,
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
    if missing:
        print(
            f"注意：这一代覆盖不全——申报 {len(chunks)} 块，实际覆盖 {len(chunks) - len(missing)} 块，"
            f"缺 {len(missing)} 块（Manifest 已标 coverage_status=incomplete，release.publish() 会拒绝发布）。"
        )
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
    print(f"  门禁 {sum(len(r.gates) for r in gate_runs)} 项全 PASS  "
          f"耗时 {elapsed:.1f}s  run={run.run_id}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="事务式构建并原子发布一代产物")
    ap.add_argument("--probes", type=int, default=200, help="检索自检抽样条数")
    ap.add_argument("--no-publish", action="store_true", help="只构建到 READY，不切换指针")
    ap.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="允许在有块未抽取时构建候选（Manifest 标 incomplete，强制要求 --no-publish）",
    )
    ap.add_argument(
        "--graph-model",
        choices=("legacy", "identity"),
        default="legacy",
        help="建图模型：legacy=聚合边 ark-rag-1（默认，行为逐字节不变）；"
        "identity=逐观察断言 + 稳定 id + 方向邻接 ark-rag-3",
    )
    return build(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
