"""把某个世代导出成一份 SQLite FTS5 索引（H3 第 ③ 步）。

## 用法

    py -3 scripts\\build_fts_index.py --generation g-20261031T035231Z-4eafae
    py -3 scripts\\build_fts_index.py --generation <gid> --force
    py -3 scripts\\build_fts_index.py --generation <gid> --no-graph

## 产物落在哪

`data/arkrag/fts/<generation_id>.sqlite`（可用 `--out` 改）。

刻意**不**放进 `generations/<gid>/`：世代目录带内容摘要与 legacy 写保护，
往里塞派生物会让 `snapshot_baseline.py` / `release_cli verify` 的目录摘要变化。
`fts_backend._refuse_inside_generation()` 会硬拒，这个脚本也有同样的自检。

## 退出码

* `0` —— 建成，或（未加 `--force` 时）产物已存在且内容可复用。
* `2` —— 硬错误：世代不存在、缺 `index/lexical.json`、目标落在世代目录里。
* `3` —— 库校验失败：建出来的库与源索引在块数/词条集合/抽样检索上对不上。
  这类失败必须是 3 而不是 0——「建了一份坏的但没人发现」正是本阶段要防的。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import config, fts_backend, index as index_mod, release, search_backend  # noqa: E402


def _fail(code: int, message: str) -> "SystemExit":
    print(f"[{code}] {message}", file=sys.stderr)
    return SystemExit(code)


def verify_database(db_path: Path, *, source: "index_mod.LexicalIndex", probe_queries: list[str]) -> dict:
    """建完必须自检，不能只看「文件写出来了」。

    三条检查，任何一条不过都抛 `RuntimeError`：
    1. 块数与源索引一致；
    2. 词条集合完全一致（多一个或少一个都算不一致——多出来的词条会让
       df 偏小、idf 偏大，而检索结果看起来仍然"正常"）；
    3. 逐块 `doc_len` 一致（长度归一化的输入）。

    再加一条更强的：用若干探针串分别走新旧后端，要求 `top_k` 列表与分数**逐位相同**。
    """
    conn = __import__("sqlite3").connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        db_docs = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        if db_docs != source.size:
            raise RuntimeError(f"块数不一致：库 {db_docs} vs 源 {source.size}")
        db_terms = int(conn.execute("SELECT COUNT(*) FROM postings").fetchone()[0])
        if db_terms != len(source.postings):
            raise RuntimeError(f"词条数不一致：库 {db_terms} vs 源 {len(source.postings)}")
        bad = 0
        for slot, chunk_id, doc_len in conn.execute("SELECT doc, chunk_id, doc_len FROM chunks"):
            if source.doc_id.get(slot) != chunk_id or int(source.doc_len.get(slot, 1)) != int(doc_len):
                bad += 1
                if bad <= 3:
                    print(
                        f"    不一致 slot={slot} 库=({chunk_id},{doc_len}) "
                        f"源=({source.doc_id.get(slot)},{source.doc_len.get(slot, 1)})",
                        file=sys.stderr,
                    )
        if bad:
            raise RuntimeError(f"{bad} 行的 chunk_id / doc_len 与源索引不一致")
    finally:
        conn.close()

    # 逐词条比对：把源倒排的每个 term 与库里的一一对上。这是 O(694k) 的循环，
    # 但只在建库时跑一次；跳过它等于放弃"词条集合一致"这个结论。
    conn = __import__("sqlite3").connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        db_terms_set = {row[0] for row in conn.execute("SELECT term FROM postings")}
    finally:
        conn.close()
    src_terms = set(source.postings)
    only_db = db_terms_set - src_terms
    only_src = src_terms - db_terms_set
    if only_db or only_src:
        raise RuntimeError(
            f"词条集合不一致：仅库 {len(only_db)}（如 {sorted(only_db)[:3]}），"
            f"仅源 {len(only_src)}（如 {sorted(only_src)[:3]}）"
        )

    backend = fts_backend.Fts5SearchBackend.open(db_path)
    try:
        mismatched = []
        for query in probe_queries:
            old = source.search(query, top_k=20)
            new = backend.search(query, top_k=20)
            if [c for c, _ in old] != [c for c, _ in new]:
                mismatched.append((query, "chunk_id 序列不同"))
                continue
            drift = max((abs(a[1] - b[1]) for a, b in zip(old, new)), default=0.0)
            # 1e-9 是"浮点运算次序不同"的量级上限，实测最大 1.4e-13。
            # 放到 1e-3 就等于允许排序在临界点上翻转而门禁不响。
            if drift > 1e-9:
                mismatched.append((query, f"分数漂移 {drift:.3e}"))
        if mismatched:
            raise RuntimeError(f"{len(mismatched)} 条探针检索与源索引不一致：{mismatched[:3]}")
        return {
            "probes": len(probe_queries),
            "max_abs_score_drift": max(
                (
                    max(
                        (abs(a[1] - b[1]) for a, b in zip(source.search(q, top_k=20), backend.search(q, top_k=20))),
                        default=0.0,
                    )
                    for q in probe_queries
                ),
                default=0.0,
            ),
        }
    finally:
        backend.close()


PROBE_QUERIES: tuple[str, ...] = (
    "凯尔希 罗德岛 关系 干员 博士",
    "阿米娅 龙门 整合运动 感染 乌萨斯 切尔诺伯格",
    "梦醒时分 俄勒冈 回答我 你觉得",
    "源石 天灾 感染者 矿石病 卡兹戴尔",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把世代导出成 SQLite FTS5 索引")
    parser.add_argument("--generation", required=True, help="世代 ID，如 g-20261031T035231Z-4eafae")
    parser.add_argument("--out", default="", help="输出 .sqlite 路径（默认 data/arkrag/fts/<gid>.sqlite）")
    parser.add_argument("--force", action="store_true", help="目标已存在时覆盖（默认拒绝）")
    parser.add_argument("--no-graph", action="store_true", help="不写实体/别名/断言/关系四张表")
    parser.add_argument("--skip-verify", action="store_true", help="跳过建库后自检（不推荐）")
    args = parser.parse_args(argv)

    try:
        gen = release.Generation(args.generation)
    except Exception as exc:  # noqa: BLE001 - 世代不存在时 release 抛什么就报什么
        raise _fail(2, f"世代 {args.generation} 取不到：{exc}") from exc
    if not gen.dir.exists():
        raise _fail(2, f"世代不存在：{gen.dir}")
    index_path = gen.index_path
    if not index_path.exists():
        raise _fail(2, f"世代缺 index/lexical.json：{index_path}")

    target = Path(args.out) if args.out else search_backend.default_db_path(args.generation)
    print(f"世代   ：{args.generation}")
    print(f"源索引 ：{index_path}（{index_path.stat().st_size / 1048576:.1f} MiB）")
    print(f"目标   ：{target}")

    if target.exists() and not args.force:
        print(f"产物已存在，跳过建库（加 --force 重建）：{target}（{target.stat().st_size / 1048576:.1f} MiB）")
        return 0

    t0 = time.perf_counter()
    # 索引源与自检共用同一份内存对象：解析 127 MB JSON 是这个脚本里最贵的一步
    # （实测 8.6 s），解析两次纯属浪费。
    source, source_meta = index_mod.load_index(index_path)
    print(
        f"源索引读出：{1000 * (time.perf_counter() - t0):.0f} ms  "
        f"块 {source.size}  词条 {len(source.postings)}  "
        f"分词 {source.tokenizer_version}  整词名 {len(source.name_terms)}"
    )

    graph = assertions = manifest = None
    if not args.no_graph:
        gp = gen.graph_path
        if gp.exists():
            graph = json.loads(gp.read_text("utf-8"))
        ap = gen.assertion_path
        if ap.exists():
            assertions = fts_backend._read_jsonl(ap)
        print(f"图产物：实体 {len((graph or {}).get('entities') or [])}  "
              f"边 {len((graph or {}).get('edges') or [])}  断言 {len(assertions or [])}")
    if gen.manifest_path.exists():
        manifest = json.loads(gen.manifest_path.read_text("utf-8"))

    stats = fts_backend.build_database(
        target,
        index=source,
        index_meta=source_meta,
        graph=graph,
        assertions=assertions,
        manifest=manifest,
        generation_id=args.generation,
    )
    del graph, assertions

    print()
    print("建库完成")
    print(f"  体积      ：{stats['mib']:.1f} MiB（{stats['bytes']:,} 字节）")
    print(f"  建库耗时  ：{stats['build_ms']:.0f} ms"
          f"（打包 {stats['pack_ms']:.0f} / FTS5 写入 {stats['fts_write_ms']:.0f} / 图写入 {stats['graph_write_ms']:.0f}）")
    print(f"  源 JSON   ：{index_path.stat().st_size / 1048576:.1f} MiB"
          f"  → 压缩比 {index_path.stat().st_size / max(1, stats['bytes']):.2f}x")
    print("  表行数    ：" + "  ".join(f"{k}={v}" for k, v in stats["tables"].items()))

    if args.skip_verify:
        print("已跳过自检（--skip-verify）")
        return 0

    t0 = time.perf_counter()
    try:
        result = verify_database(target, source=source, probe_queries=list(PROBE_QUERIES))
    except RuntimeError as exc:
        raise _fail(3, f"库校验失败：{exc}") from exc
    print(
        f"自检通过：块数/词条集合/doc_len 全一致；{result['probes']} 条探针 top-20 逐位相同，"
        f"最大分数漂移 {result['max_abs_score_drift']:.2e}（{1000 * (time.perf_counter() - t0):.0f} ms）"
    )
    print(f"提示：接线方式 `loaded['index'] = search_backend.open_backend(loaded['generation'].dir, kind='fts')`")
    print(f"提示：库位置 `{config.artifact_root() / 'fts'}`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
