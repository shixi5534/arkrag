"""L5 建图。默认要求 T1 范围内全部块都已入库——缺块就拒绝建图（不许在残缺输入上判 PASS）。

T1 范围取自 `config.T1_CATEGORIES`（主线 + 活动 + 干员密录 + 肉鸽），不再硬编码单个类别。

「已入库」的判据（计划 B3）：走 `cache.is_cached_for_chunk()`，带 chunk_id 身份核对。
旧版按 `content_key` 拼旧键查缓存，overlap 的同正文块会互相命中——某块其实没抽过，
也会被算成已入库，于是图在残缺输入上被判 PASS。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arkrag import cache, config, extract, gates, graph  # noqa: E402
from arkrag.agent_tasks import AGENT_MODEL  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true", help="只出诊断，不写图文件")
    args = ap.parse_args()

    chunks = [c for c in extract.load_chunks() if config.in_t1(c["category"])]
    missing = [c["chunk_id"] for c in chunks if not cache.is_cached_for_chunk(c, AGENT_MODEL)]
    print(f"T1 范围 {len(chunks)} 块（{'、'.join(config.T1_CATEGORIES)}），"
          f"已入库 {len(chunks) - len(missing)}，缺 {len(missing)}")
    if missing and not args.allow_partial:
        print("拒绝建图：仍有块未抽取。先跑完 L3，或加 --allow-partial 只看诊断。", file=sys.stderr)
        return 2

    records = graph.load_records(chunks)
    built = graph.build_graph(records and chunks or [], records)
    corpus_meta = json.loads(config.corpus_meta_path().read_text("utf-8"))
    # 语料指纹覆盖全部场景（不只是 T1），保证与 L6 三查同源
    graph.attach_corpus_fp(built, corpus_meta["corpus_fp"])
    stats = built["stats"]
    types = graph.entity_type_report(built)

    print(f"实体 {stats['entities']}  边 {stats['edges']}  平均度数 {stats['avg_degree']:.2f}  "
          f"孤立实体 {stats['isolated']} ({stats['isolated'] / max(1, stats['entities']):.1%})")
    print(f"类型分布 {types['type_distribution']}  词表外实体 {types['off_vocab_entities']}")
    print(f"噪声名剔除 {stats['dropped_names']}")
    print(f"合并 {stats['merge']}")
    print(f"关系裁决 {stats['relation_adjudication']}")
    print(f"侧车映射条目 {len(stats['sidecar_map'])}")
    print(f"指纹 corpus_fp={built['corpus_fp']} graph_fp={built['fingerprint']}")
    print(f"边均来源块数 {sum(len(e['chunks']) for e in built['edges']) / max(1, len(built['edges'])):.2f}")

    # L4（归一化）与 L5（建图）各自出闸。L4 没有独立产物，它是对着建好的图
    # 反查归一化那一步有没有漏闸；两项都写进 gates.json，不许只写在终端里。
    l4 = gates.l4_gates(records, built)
    l5 = gates.l5_gates(built, chunks, corpus_meta["corpus_fp"])
    for run_ in (l4, l5):
        for g in run_.gates:
            print(f"  [{g.verdict}] {g.name}  measured={g.measured}")

    if args.allow_partial:
        print("--allow-partial：仅诊断，未写 graph.json、未写门禁报告")
        return 0

    counts = graph.save(built, config.graph_path())
    print(f"已写 {config.graph_path()} -> {counts}")
    gates.write_report(l4, config.gates_path(), "L4")
    gates.write_report(l5, config.gates_path(), "L5")
    print(f"已写门禁报告 -> {config.gates_path()}（L4/L5）")

    fails = [g for g in (l4.gates + l5.gates) if g.verdict == "FAIL"]
    print(f"FAIL {len(fails)} / 共 {len(l4.gates) + len(l5.gates)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
