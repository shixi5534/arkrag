"""A 阶段：冻结 legacy 基线，并提供可失败的校验。

    py -3 scripts/snapshot_baseline.py            # 建快照（幂等；已存在则报错，除非 --force）
    py -3 scripts/snapshot_baseline.py --verify   # 校验当前产物是否仍与快照一致

产出：
    data/releases/legacy-baseline/manifest.json        产物清单（路径/字节/SHA-256/mtime）
    data/releases/legacy-baseline/gates.snapshot.json  75 项门禁的冻结副本
    docs/baseline-metrics.md                           人读的基线指标
    data/releases/legacy-baseline/metrics.json         机读的基线指标

`--verify` 退出码语义（必须能失败，见计划原则 5）：
    0  全部一致
    1  有产物被改动或丢失
    2  快照不存在
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arkrag import config, legacy  # noqa: E402


def _read_json(path: Path):
    return json.loads(Path(path).read_text("utf-8"))


def _count_jsonl(path: Path) -> int:
    n = 0
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def collect_metrics() -> dict:
    """现场指标。全部读实测，不许写估算值。"""
    m: dict = {
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "versions": {
            "schema_version": config.SCHEMA_VERSION,
            "prompt_version": config.PROMPT_VERSION,
            "normalizer_version": config.NORMALIZER_VERSION,
            "chunk_target_chars": config.CHUNK_TARGET_CHARS,
            "chunk_min_chars": config.CHUNK_MIN_CHARS,
            "chunk_overlap_turns": config.CHUNK_OVERLAP_TURNS,
            "t1_categories": list(config.T1_CATEGORIES),
            "confidence_threshold": config.CONFIDENCE_THRESHOLD,
        },
        "python": sys.version.split()[0],
    }

    # 语料。corpus_fp 的权威来源是 corpus_meta.json（L1 写），l0_manifest 记来源合同。
    l0_path = config.L0_MANIFEST_PATH
    corpus: dict = {}
    if l0_path.exists():
        l0 = _read_json(l0_path)
        corpus["baseline"] = l0.get("baseline")
        corpus["corpus_source"] = l0.get("corpus_source")
        corpus["dropped_files"] = l0.get("dropped_files")
    meta_path = config.DATA_DIR / "corpus_meta.json"
    if meta_path.exists():
        meta = _read_json(meta_path)
        corpus["corpus_fp"] = meta.get("corpus_fp")
        corpus["scenes"] = meta.get("scenes")
        corpus["turns"] = meta.get("turns")
    story_root = config.STORY_ROOT
    if story_root.exists():
        corpus["story_files_scanned"] = sum(
            1 for p in story_root.rglob("*.json") if p.name != "story_variables.json"
        )
    m["corpus"] = corpus

    # L1/L2
    scenes_path = config.DATA_DIR / "scenes.jsonl"
    if scenes_path.exists():
        m["scenes"] = {"count": _count_jsonl(scenes_path)}
        turns = 0
        with scenes_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    turns += len(json.loads(line).get("turns") or [])
        m["scenes"]["turns"] = turns

    chunks_path = config.DATA_DIR / "chunks.jsonl"
    if chunks_path.exists():
        per_category: dict[str, int] = {}
        t1 = 0
        total = 0
        with chunks_path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                c = json.loads(line)
                total += 1
                per_category[c["category"]] = per_category.get(c["category"], 0) + 1
                if config.in_t1(c["category"]):
                    t1 += 1
        m["chunks"] = {"total": total, "t1": t1, "per_category": per_category}

    # L3 缓存
    cache_dir = config.CACHE_DIR
    if cache_dir.exists():
        m["extraction_cache"] = legacy.dir_digest(cache_dir)

    # L5 图
    graph_path = config.GRAPH_PATH
    if graph_path.exists():
        g = _read_json(graph_path)
        m["graph"] = {
            "entities": len(g.get("entities") or []),
            "edges": len(g.get("edges") or []),
            "isolated": (g.get("stats") or {}).get("isolated"),
            "avg_degree": (g.get("stats") or {}).get("avg_degree"),
            "fingerprint": g.get("fingerprint"),
            "corpus_fp": g.get("corpus_fp"),
            "schema_version": g.get("schema_version"),
            "prompt_version": g.get("prompt_version"),
            "normalizer_version": g.get("normalizer_version"),
            "entity_rule_version": g.get("entity_rule_version"),
            "alias_count": len(g.get("alias_index") or {}),
        }

    # L6 索引
    index_path = config.INDEX_DIR / "lexical.json"
    if index_path.exists():
        idx = _read_json(index_path)
        m["index"] = {
            "bytes": index_path.stat().st_size,
            "meta": idx.get("meta"),
            "terms": len(idx.get("postings") or {}),
            "postings": sum(len(v) for v in (idx.get("postings") or {}).values()),
            "docs": len(idx.get("doc_id") or {}),
        }

    # 门禁
    gates_path = config.DATA_DIR / "gates.json"
    if gates_path.exists():
        gs = _read_json(gates_path)
        by_stage: dict[str, int] = {}
        by_verdict: dict[str, int] = {}
        for row in gs:
            by_stage[row.get("stage", "?")] = by_stage.get(row.get("stage", "?"), 0) + 1
            by_verdict[row.get("verdict", "?")] = by_verdict.get(row.get("verdict", "?"), 0) + 1
        m["gates"] = {
            "rows": len(gs),
            "by_stage": by_stage,
            "by_verdict": by_verdict,
            "non_pass": [r for r in gs if r.get("verdict") != "PASS"],
        }
    return m


def collect_artifacts() -> list[dict]:
    """产物清单：所有会进入 baseline 判断的文件都记 SHA-256。"""
    rows: list[dict] = []

    singles = [
        (config.GRAPH_PATH, "graph"),
        (config.INDEX_DIR / "lexical.json", "index"),
        (config.DATA_DIR / "gates.json", "gates"),
        (config.DATA_DIR / "chunks.jsonl", "chunks"),
        (config.DATA_DIR / "scenes.jsonl", "scenes"),
        (config.DATA_DIR / "shards.json", "shards"),
        (config.DATA_DIR / "l0_manifest.json", "l0_manifest"),
        (config.DATA_DIR / "l1_stats.json", "l1_stats"),
        (config.DATA_DIR / "l1_dropped.json", "l1_dropped"),
        (config.DATA_DIR / "corpus_meta.json", "corpus_meta"),
    ]
    for path, kind in singles:
        if Path(path).exists():
            rows.append(legacy.artifact_record(Path(path), kind))

    # 缓存目录用整体摘要 + 逐文件清单哈希，不逐文件堆 JSON
    if config.CACHE_DIR.exists():
        d = legacy.dir_digest(config.CACHE_DIR)
        rows.append(
            {
                "path": legacy._rel(config.CACHE_DIR),
                "kind": "extraction_cache_dir",
                "count": d["count"],
                "bytes": d["bytes"],
                "digest": d["digest"],
            }
        )
    return rows


def build_manifest() -> dict:
    metrics = collect_metrics()
    artifacts = collect_artifacts()
    gates_path = config.DATA_DIR / "gates.json"
    gate_rows = _read_json(gates_path) if gates_path.exists() else []
    return {
        "baseline_id": "legacy-baseline",
        "label": "legacy",
        "read_only": True,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": (
            "冻结的既有可用产物。后续实验禁止直接写入这些路径；"
            "新产物写 generations/<generation_id>/。"
            f"确需重建时设置 {legacy.ENV_ALLOW}=1。"
        ),
        "protected_files": list(legacy.PROTECTED_FILES),
        "protected_dirs": list(legacy.PROTECTED_DIRS),
        "versions": metrics["versions"],
        "metrics": metrics,
        "artifacts": artifacts,
        "gate_rows": len(gate_rows),
        "gate_report_sha256": legacy.sha256_file(gates_path) if gates_path.exists() else None,
    }


def write_snapshot(force: bool) -> int:
    legacy.BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    if legacy.BASELINE_MANIFEST.exists() and not force:
        print(
            f"快照已存在: {legacy._rel(legacy.BASELINE_MANIFEST)}\n"
            "冻结件只能建一次。确需重建基线请加 --force，并在台账里写明为什么旧基线作废。",
            file=sys.stderr,
        )
        return 2

    manifest = build_manifest()

    # gates 冻结副本
    gates_src = config.DATA_DIR / "gates.json"
    snapshot_path = legacy.BASELINE_DIR / "gates.snapshot.json"
    if gates_src.exists():
        snapshot_path.write_bytes(gates_src.read_bytes())

    metrics_path = legacy.BASELINE_DIR / "metrics.json"
    metrics_path.write_text(
        json.dumps(manifest["metrics"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    legacy.BASELINE_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    md = render_markdown(manifest)
    (config.ROOT / "docs" / "baseline-metrics.md").write_text(md, encoding="utf-8", newline="\n")

    print(f"已写 {legacy._rel(legacy.BASELINE_MANIFEST)}")
    print(f"已写 {legacy._rel(snapshot_path)}")
    print(f"已写 {legacy._rel(metrics_path)}")
    print("已写 docs/baseline-metrics.md")
    print(f"产物条目 {len(manifest['artifacts'])}  门禁行 {manifest['gate_rows']}")
    return 0


def render_markdown(manifest: dict) -> str:
    m = manifest["metrics"]
    v = m["versions"]
    L: list[str] = []
    L.append("# legacy-baseline 基线指标")
    L.append("")
    L.append("> 本文件由 `scripts/snapshot_baseline.py` 生成，**不要手改**。")
    L.append("> 冻结时间：`" + manifest["created_at"] + "`")
    L.append("> 校验命令：`py -3 scripts/snapshot_baseline.py --verify`（不一致时退出码 1）")
    L.append("")
    L.append("## 版本常量")
    L.append("")
    L.append("| 常量 | 值 |")
    L.append("|---|---|")
    for k in ("schema_version", "prompt_version", "normalizer_version"):
        L.append(f"| {k} | `{v[k]}` |")
    L.append(f"| chunk 目标/下限/重叠 | {v['chunk_target_chars']} / {v['chunk_min_chars']} / {v['chunk_overlap_turns']} |")
    L.append(f"| T1 范围 | {'、'.join(v['t1_categories'])} |")
    L.append(f"| 置信度阈值 | {v['confidence_threshold']} |")
    L.append(f"| Python | {m['python']} |")
    L.append("")

    if "corpus" in m:
        c = m["corpus"]
        L.append("## 语料")
        L.append("")
        L.append(f"- corpus_fp：`{c.get('corpus_fp')}`")
        if c.get("story_files_scanned"):
            L.append(f"- story 文件数：{c['story_files_scanned']}")
        src = c.get("corpus_source") or {}
        if src:
            L.append(f"- 来源仓库：`{src.get('repo')}`")
            L.append(f"- 来源 release：`{src.get('release_tag')}`")
            L.append(f"- ZIP：`{src.get('asset')}` {src.get('zip_bytes'):,} B sha256 `{src.get('zip_sha256')}`"
                     if isinstance(src.get("zip_bytes"), int) else f"- ZIP：`{src.get('asset')}`")
        base = c.get("baseline") or {}
        if base:
            L.append(f"- 源文件数：{base.get('story_files')}  源字节：{base.get('story_bytes'):,}"
                     if isinstance(base.get("story_bytes"), int) else f"- 源文件数：{base.get('story_files')}")
            bc = base.get("by_category") or {}
            if bc:
                L.append("- 源分类：" + "、".join(f"{k}={v}" for k, v in sorted(bc.items())))
        dropped = c.get("dropped_files") or []
        if dropped:
            L.append(f"- L0 丢弃文件：{len(dropped)} 个 → " + "、".join(
                f"`{(d.get('path') or d.get('file') or '?')}`" for d in dropped[:20]
            ))
        L.append("")
    if "scenes" in m:
        L.append("## L1 场景")
        L.append("")
        L.append(f"- 场景数：**{m['scenes']['count']:,}**")
        L.append(f"- 轮次数：**{m['scenes']['turns']:,}**")
        L.append("")
    if "chunks" in m:
        ch = m["chunks"]
        L.append("## L2 分块")
        L.append("")
        L.append(f"- 全量块数：**{ch['total']:,}**")
        L.append(f"- T1 块数：**{ch['t1']:,}**")
        L.append("")
        L.append("| category | 块数 |")
        L.append("|---|---|")
        for k in sorted(ch["per_category"]):
            L.append(f"| {k} | {ch['per_category'][k]:,} |")
        L.append("")
    if "extraction_cache" in m:
        e = m["extraction_cache"]
        L.append("## L3 抽取缓存")
        L.append("")
        L.append(f"- 缓存文件数：**{e['count']:,}**")
        L.append(f"- 总字节：{e['bytes']:,}")
        L.append(f"- 整体摘要：`{e['digest']}`")
        L.append("")
    if "graph" in m:
        g = m["graph"]
        L.append("## L5 图")
        L.append("")
        L.append("| 指标 | 值 |")
        L.append("|---|---|")
        L.append(f"| 实体数 | **{g['entities']:,}** |")
        L.append(f"| 边数 | **{g['edges']:,}** |")
        L.append(f"| 孤立实体 | {g['isolated']} |")
        L.append(f"| 平均度数 | {g['avg_degree']:.2f} |" if isinstance(g["avg_degree"], (int, float)) else f"| 平均度数 | {g['avg_degree']} |")
        L.append(f"| 别名条目 | {g['alias_count']:,} |")
        L.append(f"| graph_fp | `{g['fingerprint']}` |")
        L.append(f"| corpus_fp | `{g['corpus_fp']}` |")
        L.append(f"| ev / n | `{g['entity_rule_version']}` / `{g['normalizer_version']}` |")
        L.append("")
    if "index" in m:
        i = m["index"]
        L.append("## L6 索引")
        L.append("")
        L.append("| 指标 | 值 |")
        L.append("|---|---|")
        L.append(f"| 文件字节 | {i['bytes']:,}（{i['bytes'] / 1e6:.1f} MB） |")
        L.append(f"| 文档数 | {i['docs']:,} |")
        L.append(f"| 词条数 | {i['terms']:,} |")
        L.append(f"| 倒排项数 | {i['postings']:,} |")
        if i.get("meta"):
            L.append(f"| 索引图指纹 | `{i['meta'].get('graph_fp')}` |")
            L.append(f"| 向量通道 | `{i['meta'].get('vector_channels')}` |")
        L.append("")
    if "gates" in m:
        gs = m["gates"]
        L.append("## 门禁")
        L.append("")
        L.append(f"- 总行数：**{gs['rows']}**")
        L.append("- 分布：" + "、".join(f"{k}={v}" for k, v in sorted(gs["by_stage"].items())))
        L.append("- 判定：" + "、".join(f"{k}={v}" for k, v in sorted(gs["by_verdict"].items())))
        L.append("")
        if gs["non_pass"]:
            L.append("非 PASS 行：")
            L.append("")
            for r in gs["non_pass"]:
                L.append(f"- `{r.get('name')}` → {r.get('verdict')}")
        else:
            L.append("**全部 75 项 PASS。** 但依计划原则 8，全 PASS 不作为免检依据；")
            L.append("本基线只冻结「当时的面貌」，不表示已知盲区都被覆盖。")
        L.append("")
    L.append("## 产物清单（SHA-256）")
    L.append("")
    L.append("| path | kind | bytes | sha256 |")
    L.append("|---|---|---|---|")
    for a in manifest["artifacts"]:
        sha = a.get("sha256") or a.get("digest") or ""
        byt = a.get("bytes")
        byt_s = f"{byt:,}" if isinstance(byt, int) else str(byt)
        L.append(f"| `{a['path']}` | {a['kind']} | {byt_s} | `{sha}` |")
    L.append("")
    L.append("## 已知能力边界（继承自现有台账，不在本阶段修复）")
    L.append("")
    L.append("- L6 只有词法通道，`vector_channels=not_built_no_embedding_backend`。**对外不能说具备语义检索。**")
    L.append("- 图是聚合边模型，一条边只有一条 `evidence`，无法回答「这条边在第几段原文」。")
    L.append("- 图邻接双向且反向边沿用同一关系名，方向语义会丢。")
    L.append("- 无检索/问答/评测入口——系统目前只能建，不能答。")
    return "\n".join(L) + "\n"


def verify() -> int:
    manifest = legacy.load_baseline()
    if manifest is None:
        print(
            f"快照不存在: {legacy._rel(legacy.BASELINE_MANIFEST)}\n先跑 py -3 scripts/snapshot_baseline.py",
            file=sys.stderr,
        )
        return 2

    problems: list[str] = []
    for rec in manifest["artifacts"]:
        path = config.ROOT / rec["path"]
        if not path.exists():
            problems.append(f"产物丢失: {rec['path']}")
            continue
        if rec["kind"] == "extraction_cache_dir":
            d = legacy.dir_digest(path)
            if d["digest"] != rec["digest"]:
                problems.append(
                    f"缓存目录已变: {rec['path']} "
                    f"count {rec['count']}->{d['count']} digest {rec['digest'][:12]}->{d['digest'][:12]}"
                )
            continue
        got = legacy.sha256_file(path)
        if got != rec["sha256"]:
            problems.append(
                f"字节已变: {rec['path']} sha256 {rec['sha256'][:12]}->{got[:12]} "
                f"bytes {rec['bytes']}->{path.stat().st_size}"
            )

    # 门禁冻结副本也必须在
    snap = legacy.BASELINE_DIR / "gates.snapshot.json"
    if not snap.exists():
        problems.append(f"门禁冻结副本丢失: {legacy._rel(snap)}")

    if problems:
        print(f"legacy-baseline 校验 FAIL（{len(problems)} 项）:", file=sys.stderr)
        for p in problems:
            print("  - " + p, file=sys.stderr)
        print(
            "\n如果你确实重建了产物：确认新产物可用后重建基线（--force），"
            "并在 docs/升级台账.md 写明旧基线作废的理由。",
            file=sys.stderr,
        )
        return 1

    print(f"legacy-baseline 校验 PASS  产物 {len(manifest['artifacts'])} 项全部一致")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="校验当前产物是否与快照一致")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的快照（需在台账说明理由）")
    ap.add_argument("--print-metrics", action="store_true", help="只打印当前指标，不写盘")
    args = ap.parse_args()

    if args.print_metrics:
        print(json.dumps(collect_metrics(), ensure_ascii=False, indent=2))
        return 0
    if args.verify:
        return verify()
    return write_snapshot(args.force)


if __name__ == "__main__":
    raise SystemExit(main())
