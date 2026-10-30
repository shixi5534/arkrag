"""L2 分块 + 章节装箱分片，产出 chunks.jsonl / shards.json 与 L2 门判定。"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arkrag import chunk as chunker  # noqa: E402
from arkrag import config, gates  # noqa: E402

T1_CATEGORIES = config.T1_CATEGORIES


def load_scenes(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def main() -> int:
    scenes = load_scenes(config.DATA_DIR / "scenes.jsonl")
    chunks = []
    for scene in scenes:
        chunks.extend(chunker.split_scene(scene))

    t1_chunks = [c for c in chunks if config.in_t1(c["category"])]
    shards_t1 = chunker.pack_shards(t1_chunks)
    shards_full = chunker.pack_shards(chunks)
    owner_t1 = chunker.shard_membership(t1_chunks, shards_t1)

    for c in chunks:
        c["shard"] = owner_t1.get(c["chunk_id"], "")
        c["in_t1"] = config.in_t1(c["category"])

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with (config.DATA_DIR / "chunks.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    (config.DATA_DIR / "shards.json").write_text(
        json.dumps(
            {
                "schema_version": config.SCHEMA_VERSION,
                "target_blocks_per_shard": 200,
                "t1_scope": list(T1_CATEGORIES),
                "t1": {
                    "chunks": len(t1_chunks),
                    "shards": shards_t1,
                    "concurrency_plan": "2 片并行 x 每片并发 4 = 全局 8",
                },
                "full": {"chunks": len(chunks), "shards": shards_full},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    run = gates.l2_gates(chunks, shards_t1, scenes)
    gates.write_report(run, config.DATA_DIR / "gates.json", "L2")

    total_chars = sum(len(t["text"]) for s in scenes for t in s["turns"])
    print("=== L2 结果 ===")
    print(f"总块数 {len(chunks):,}   T1 块数 {len(t1_chunks):,}   总字数 {total_chars:,}")
    print("T1 范围: " + "、".join(T1_CATEGORIES))
    sizes = Counter()
    for c in chunks:
        sizes[c["category"].split("/")[0]] += 1
    print("按类块数:", dict(sizes))
    print(f"T1 分片 {len(shards_t1)} 片:")
    for s in shards_t1:
        print(f"  {s['shard']} blocks={s['blocks']:4d} docs={s['docs']:4d} chapters={len(s['chapters'])} "
              + ",".join(x['chapter'].split('::')[1] for x in s['chapters'][:6]))
    print("\n=== L2 门判定 ===")
    for g in run.gates:
        print(f"  {g.verdict:13s} {g.name:34s} 门线[{g.threshold}] 实测[{g.measured[:60]}]")
        if g.note:
            print(f"      note: {g.note[:170]}")
    fails = [g for g in run.gates if g.verdict == "FAIL"]
    print(f"FAIL {len(fails)} / 共 {len(run.gates)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
