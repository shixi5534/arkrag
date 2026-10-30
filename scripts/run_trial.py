"""L3 抽检验收：对某一片（或某个 T0 标签、或 T1 全量）已入库的块逐条机检 + 产出人工对照清单。

用法：
  py -3 scripts/run_trial.py --label S00
  py -3 scripts/run_trial.py --label T1_full --all-t1 --no-doc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arkrag import agent_tasks, cache, config, extract, gates, prompt  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--shard", help="按分片选块；缺省按标签里的分片名推断")
    ap.add_argument("--all-t1", action="store_true", help="不做分片，直接对 T1 全量块机检")
    ap.add_argument("--no-doc", action="store_true", help="不产出人工对照清单（全量时必用）")
    args = ap.parse_args()

    shard = args.shard or (args.label if args.label.startswith("S") else None)
    if args.all_t1:
        chunks = [c for c in extract.load_chunks() if c.get("in_t1")]
    else:
        chunks = extract.load_chunks(shard=shard) if shard else None
    if chunks is None:
        inbox = sorted(agent_tasks.INBOX_DIR.glob(f"{args.label}.part*.jsonl")) or [
            agent_tasks.INBOX_DIR / f"{args.label}.jsonl"
        ]
        import json

        ids = []
        for path in inbox:
            if not path.exists():
                continue
            for line in path.read_text("utf-8").splitlines():
                if line.strip():
                    ids.append(json.loads(line)["chunk_id"])
        all_chunks = {c["chunk_id"]: c for c in extract.load_chunks()}
        chunks = [all_chunks[i] for i in ids if i in all_chunks]

    records, absent = [], []
    for chunk in chunks:
        raw = cache.read_for_chunk(chunk, agent_tasks.AGENT_MODEL)
        if raw is None:
            absent.append(chunk["chunk_id"])
            continue
        records.append({"chunk_id": chunk["chunk_id"], "text": chunk["text"], "raw": raw, "chunk": chunk})

    run = gates.l3_trial_gates(records, expected_blocks=len(chunks))
    if absent:
        run.add(
            gates.Gate(
                "L3.T18 该片块已入库数",
                "L3",
                f"{len(chunks)}",
                f"{len(records)} 缺 {len(absent)}",
                "FAIL",
                "; ".join(absent[:5]),
            )
        )
    gates.write_report(run, config.DATA_DIR / "gates.json", "L3")

    if args.no_doc:
        print(f"=== L3 抽检（{args.label}：{len(records)}/{len(chunks)} 块已入库）===")
        for g in run.gates:
            print(f"  {g.verdict:12s} {g.name:44s} 门线[{g.threshold}] 实测[{g.measured[:70]}]")
            if g.verdict != "PASS" and g.note:
                print(f"      note: {g.note[:220]}")
        fails = [g for g in run.gates if g.verdict == "FAIL"]
        print(f"FAIL {len(fails)} / 共 {len(run.gates)}")
        return 1 if fails else 0

    lines = [f"# {args.label} 抽取人工对照（已入库 {len(records)} 块，全量不抽样）", ""]
    for rec in records:
        raw = rec["raw"]
        ents = raw.get("entities") or []
        rels = raw.get("relations") or []
        lines.append(f"## {rec['chunk_id']} 〔{rec['chunk']['title']}｜{rec['chunk']['chapter']}〕")
        lines.append(
            f"- 实体 {len(ents)}：" + "、".join(f"{e['name']}〔{e['type']}〕" for e in ents[:40])
        )
        lines.append(
            f"- 关系 {len(rels)}："
            + "；".join(f"{r['source']}—{r['relation']}→{r['target']}‖{r['evidence']}" for r in rels[:30])
        )
        lines.append("")
    out = Path("docs") / f"人工抽验_{args.label}_{len(records)}块.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    print(f"=== L3 抽检（{args.label}：{len(records)}/{len(chunks)} 块已入库）===")
    for g in run.gates:
        print(f"  {g.verdict:12s} {g.name:30s} 门线[{g.threshold}] 实测[{g.measured[:70]}]")
        if g.verdict != "PASS" and g.note:
            print(f"      note: {g.note[:220]}")
    fails = [g for g in run.gates if g.verdict == "FAIL"]
    print(f"FAIL {len(fails)} / 共 {len(run.gates)}   清单 -> {out}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
