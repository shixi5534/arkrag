"""E1 真实数据验证：逐观察断言与聚合边在真实语料上的守恒与可定位性。

这是**一次性诊断脚本**，不是流水线的一环。它回答三个问题：

1. 逐观察拆分后有没有丢事实？（63,172 条 raw relations 的去向逐条对齐）
2. 每条断言的证据在自己的块里能不能定位？（计划 E1 的硬要求）
3. 聚合边能不能回溯到某条真实断言？（L5.10 要查的性质）

用法：
    py -3 scripts/e1_validate.py [--limit N] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from arkrag import assertion, config, extract, graph  # noqa: E402


def _alias_map(graph_doc: dict) -> dict[str, str]:
    """从现有图里取「名字/别名/被并名 -> 实体 id」映射。

    三处都必须收，少一处就会把"图上明明有的实体"算成 `dropped:endpoint_unknown`：

    * `entities[].name` —— 规范名；
    * `entities[].aliases` —— 归一化产出的别名；
    * `entities[].merged_from` —— 被并进来的原名（实测 627 条，最容易漏）。

    注意当前聚合图里实体 id 就是名字本身（E3 会换成稳定哈希 id），
    且 `alias_index` 的值是**规范名**而不是 id，所以要经 `by_name` 折一次。
    """
    out: dict[str, str] = {}
    by_name: dict[str, str] = {}
    for ent in graph_doc.get("entities") or []:
        eid = ent.get("id") or ent.get("name") or ""
        name = ent.get("name") or ""
        if not eid:
            continue
        if name:
            by_name.setdefault(name, eid)
            out.setdefault(name, eid)
        for alias in ent.get("aliases") or []:
            if alias:
                out.setdefault(alias, eid)
        for old in ent.get("merged_from") or []:
            if old:
                out.setdefault(old, eid)
    # alias_index: 别名 -> 规范名，再经 by_name 折成 id
    for alias, canonical in (graph_doc.get("alias_index") or {}).items():
        if not alias:
            continue
        target = by_name.get(canonical) or by_name.get(alias)
        if target:
            out.setdefault(alias, target)
    return out


def _identity_map(records: list[dict]) -> dict[str, str]:
    """raw 里出现过的全部实体名 -> 自身。

    这是 `--identity` 口径：把"消歧"这件事从验证里摘出去，
    只回答 E1 自己的问题——**观察守恒与证据定位**。
    用现图的别名表会让"图里本来就没有的名字"也被算进
    `dropped:endpoint_unknown`，那是 E3 的问题，不是 E1 的。
    """
    out: dict[str, str] = {}
    for rec in records:
        raw = rec.get("raw") or {}
        for ent in raw.get("entities") or []:
            name = (ent.get("name") or "").strip()
            if name:
                out.setdefault(name, name)
        for rel in raw.get("relations") or []:
            for key in ("source", "target"):
                name = (rel.get(key) or "").strip()
                if name:
                    out.setdefault(name, name)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 块（默认全量）")
    ap.add_argument("--json", default="", help="把结果写到这个 JSON 路径")
    ap.add_argument(
        "--identity",
        action="store_true",
        help=(
            "用 raw 里出现过的全部实体名做恒等映射（模拟 E3 之前的 Candidate 全集）。"
            "默认用现图的实体/别名表，于是『图里没有的名字』会进 dropped:endpoint_unknown。"
            "两种口径都要看：前者回答『观察守恒与否』，后者回答『现图漏了哪些端点』。"
        ),
    )
    args = ap.parse_args()

    chunks = [c for c in extract.load_chunks() if config.in_t1(c["category"])]
    if args.limit:
        chunks = chunks[: args.limit]
    print(f"块数 {len(chunks)}")

    records = graph.load_records(chunks)
    print(f"抽取记录 {len(records)}")

    if args.identity:
        alias_map = _identity_map(records)
        print(f"恒等映射 {len(alias_map)} 条（raw 里出现过的全部实体名）")
    else:
        graph_path = config.graph_path()
        graph_doc = json.loads(graph_path.read_text("utf-8"))
        alias_map = _alias_map(graph_doc)
        print(f"别名表 {len(alias_map)} 条（来自 {graph_path.name}）")

    extraction_meta = {
        "schema_version": config.SCHEMA_VERSION,
        "prompt_version": config.PROMPT_VERSION,
        "normalizer_version": config.NORMALIZER_VERSION,
        "entity_rule_version": graph.ENTITY_RULE_VERSION,
    }
    assertions, stats = assertion.build_assertions(
        chunks, records, entity_alias_map=alias_map, extraction_meta=extraction_meta
    )
    edges = assertion.aggregate(assertions)
    conservation = assertion.observation_conservation(assertions, edges)
    conflicts = assertion.check_qualifier_consistency(assertions)

    # 证据可定位性：逐条重新在自己的块里核一次（不信建 Assertion 时的结论）
    by_chunk = {c["chunk_id"]: c["text"] for c in chunks}
    unlocatable = []
    for a in assertions:
        text = by_chunk.get(a["chunk_id"], "")
        start, end = a["evidence_start"], a["evidence_end"]
        if text[start:end] != a["evidence_text"]:
            unlocatable.append(a["assertion_id"])
    # evidence_all 每一条也要核
    bad_all = []
    for a in assertions:
        text = by_chunk.get(a["chunk_id"], "")
        for occ in a["evidence_all"]:
            if text[occ["start"] : occ["end"]] != a["evidence_text"]:
                bad_all.append(a["assertion_id"])
                break

    # 每条聚合边的代表证据必须能在同边某条断言里找到
    by_aid = {a["assertion_id"]: a for a in assertions}
    untraceable = [
        e["edge_id"]
        for e in edges
        if e["evidence_assertion_id"] not in e["assertion_ids"]
        or by_aid.get(e["evidence_assertion_id"], {}).get("evidence_text") != e["evidence"]
    ]

    summary = {
        "chunks": len(chunks),
        "records": len(records),
        "raw_relations": stats["raw_relations"],
        "assertions": len(assertions),
        "edges": len(edges),
        "conservation": conservation,
        "stats": assertion.sum_stats(stats),
        "multi_occurrence_assertions": stats.get("evidence_multiple_occurrences", 0),
        "unlocatable": len(unlocatable),
        "bad_evidence_all": len(bad_all),
        "untraceable_edges": len(untraceable),
        "conflicts_total": len(conflicts),
        "conflicts_needs_human": sum(1 for c in conflicts if c["action"] == "needs_human"),
        "conflicts_kept_both": sum(1 for c in conflicts if c["action"] == "kept_both"),
        "support_histogram": dict(
            Counter(
                "1" if e["support_count"] == 1 else
                "2-5" if e["support_count"] <= 5 else
                "6-20" if e["support_count"] <= 20 else
                "21+"
                for e in edges
            )
        ),
        "max_support": max((e["support_count"] for e in edges), default=0),
    }

    print("\n=== 守恒 ===")
    for key in ("assertions", "edges", "support_sum", "ok"):
        print(f"  {key} = {conservation[key]}")
    for key in ("unreferenced", "multiply_referenced", "unknown_references", "support_mismatch"):
        if conservation[key]:
            print(f"  [!!] {key}: {conservation[key][:3]}")

    print("\n=== 丢弃去向 ===")
    dropped = summary["stats"]["dropped"]
    for reason, count in sorted(dropped.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:40s} {count}")
    print(f"  {'合计':40s} {summary['stats']['dropped_total']}")

    # 算术闭合：每一条 raw relation 要么成为断言，要么有书面丢弃理由。
    # 对不上就是有人在中途悄悄丢了东西——这正是本层要防的事。
    dup = summary["stats"]["duplicate_assertion_id"]
    dropped_total = summary["stats"]["dropped_total"]
    accounted = dropped_total + dup + summary["assertions"]
    print("\n=== 算术闭合 ===")
    print(f"  raw_relations          {summary['raw_relations']}")
    print(f"  - dropped              {dropped_total}")
    print(f"  - duplicate_assertion  {dup}")
    print(f"  = 断言                 {summary['assertions']}")
    print(f"  合计 {accounted} vs raw {summary['raw_relations']}: ", end="")
    print("一致" if accounted == summary["raw_relations"] else "[!!] 不一致，有观察去向不明")
    if dup:
        print(
            f"  注：{dup} 条是逐字重复的观察（同块同端点同谓词同证据位置），"
            "去重正确，但必须计数而不是静默忽略"
        )

    print("\n=== 证据定位（独立复核，不信建库时的结论）===")
    print(f"  定位失败断言      {summary['unlocatable']}")
    print(f"  evidence_all 不符 {summary['bad_evidence_all']}")
    print(f"  块内多处出现      {summary['multi_occurrence_assertions']}")
    print(f"  聚合边代表证据不可回溯 {summary['untraceable_edges']}")

    print("\n=== 聚合规模 ===")
    print(f"  断言 {summary['assertions']} -> 边 {summary['edges']}")
    print(f"  support 分布 {summary['support_histogram']}（最大 {summary['max_support']}）")
    print(f"  互斥冲突 {summary['conflicts_total']}（需人工 {summary['conflicts_needs_human']} / 跨场景保留 {summary['conflicts_kept_both']}）")

    ok = (
        conservation["ok"]
        and summary["unlocatable"] == 0
        and summary["bad_evidence_all"] == 0
        and summary["untraceable_edges"] == 0
    )
    print("\n=== 判定 ===")
    print("  PASS" if ok else "  FAIL")

    if args.json:
        Path(args.json).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  结果已写 {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
