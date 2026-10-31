"""E 阶段门禁的运行入口（L5.9/L5.10/L5.11/L4.12/L4.13/L6.8）。

**为什么不接进 `build_generation`**：这些门里有两条在当前图上必然 FAIL
（L4.12 实体 id 就是名字、L4.13 无规则计数），接进主链路会把每一次构建都打红，
而红着的门没人看。它们属于"新模型的门"——等 E 阶段落地、生产图切到
Assertion 模型之后，再整体接进 `gates.l5_gates` / `l4_gates` / `l6_gates`。

本脚本只跑、只打印、只写报告，不改任何产物。

用法：
    py -3 scripts/e_gates.py [--json path] [--stage L5|L4|L6|all]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from arkrag import adjacency as adjacency_mod  # noqa: E402
from arkrag import assertion, config, extract, gates, graph  # noqa: E402


def _alias_map(graph_doc: dict) -> dict[str, str]:
    """名字/别名/被并名 -> 实体 id。三处都要收（`merged_from` 最容易漏）。"""
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
    for alias, canonical in (graph_doc.get("alias_index") or {}).items():
        if not alias:
            continue
        target = by_name.get(canonical) or by_name.get(alias)
        if target:
            out.setdefault(alias, target)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="", help="把门禁行写到这个 JSON 路径")
    ap.add_argument(
        "--stage",
        default="all",
        choices=["L4", "L5", "L6", "all"],
        help="只跑某个 stage（默认全跑）",
    )
    args = ap.parse_args()

    chunks = [c for c in extract.load_chunks() if config.in_t1(c["category"])]
    records = graph.load_records(chunks)
    graph_doc = json.loads(config.graph_path().read_text("utf-8"))
    alias_map = _alias_map(graph_doc)
    print(f"块 {len(chunks)}  记录 {len(records)}  别名表 {len(alias_map)}", file=sys.stderr)

    extraction_meta = {
        "schema_version": config.SCHEMA_VERSION,
        "prompt_version": config.PROMPT_VERSION,
        "normalizer_version": config.NORMALIZER_VERSION,
        "entity_rule_version": graph.ENTITY_RULE_VERSION,
    }
    assertions, _stats = assertion.build_assertions(
        chunks, records, entity_alias_map=alias_map, extraction_meta=extraction_meta
    )
    edges = assertion.aggregate(assertions)
    print(f"断言 {len(assertions)}  边 {len(edges)}", file=sys.stderr)

    rows: list[dict] = []
    if args.stage in ("L5", "all"):
        rows += gates.l5_evidence_gates(graph_doc, assertions, edges, chunks).rows()
    if args.stage in ("L4", "all"):
        rows += gates.l4_identity_gates(graph_doc).rows()
    if args.stage in ("L6", "all"):
        adj = adjacency_mod.build_directed_adjacency(edges)
        rows += gates.l6_direction_gates(adj, edges).rows()

    fails = [r for r in rows if r["verdict"] == "FAIL"]
    for r in rows:
        print(f"[{r['verdict']:12s}] {r['name']}")
        print(f"              实测: {r['measured']}")
        if r["note"]:
            print(f"              说明: {r['note']}")
    print(
        f"\n合计 {len(rows)} 项，FAIL {len(fails)}，"
        f"NEEDS_HUMAN {sum(1 for r in rows if r['verdict'] == 'NEEDS_HUMAN')}"
    )

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"门禁行已写 {out}")

    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
