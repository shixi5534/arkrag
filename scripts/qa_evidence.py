"""独立质检：把已入库产物按「evidence 是否真的在正文里」重验一遍。

ingest_all.py 只跑 extract.validate_shape（结构），不验证据真伪——子代理自报"逐字命中"
不作为判据。这里直接拿 chunk 正文做字符串包含检查，顺带统计密度与词表占用，
用于写抽验报告。只读，不改任何产物。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import agent_tasks, cache, config, extract  # noqa: E402
from arkrag import prompt as prompt_mod  # noqa: E402
from arkrag import vocab  # noqa: E402


def body_of(c: dict) -> str:
    """证据的比对基准是块正文本身（chunk.text），不是 user_prompt 外壳。"""
    return c.get("text") or ""


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    chunks = extract.load_chunks()
    if limit:
        chunks = chunks[:limit]

    stats = {
        "chunks_total": 0,
        "chunks_cached": 0,
        "chunks_empty": 0,
        "entities": 0,
        "relations": 0,
        "rel_evidence_bad": 0,
        "rel_endpoint_missing": 0,
        "rel_off_vocab": 0,
        "rel_too_long": 0,
    }
    types = Counter()
    rels = Counter()
    bad_samples: list[str] = []

    allowed = set(vocab.CANONICAL_RELATIONS)

    for c in chunks:
        stats["chunks_total"] += 1
        raw = cache.read_for_chunk(c, agent_tasks.AGENT_MODEL)
        if raw is None:
            continue
        stats["chunks_cached"] += 1
        ents = raw.get("entities") or []
        rs = raw.get("relations") or []
        if not ents and not rs:
            stats["chunks_empty"] += 1
        stats["entities"] += len(ents)
        stats["relations"] += len(rs)
        for e in ents:
            types[e.get("type")] += 1
        body = body_of(c)
        names = {e.get("name") for e in ents}
        for r in rs:
            rels[r.get("relation")] += 1
            ev = r.get("evidence") or ""
            if allowed and r.get("relation") not in allowed:
                stats["rel_off_vocab"] += 1
            if len(ev) > 40:
                stats["rel_too_long"] += 1
            if ev and ev not in body:
                stats["rel_evidence_bad"] += 1
                if len(bad_samples) < 12:
                    bad_samples.append(f"{c['chunk_id']}: '{ev}'")
            if r.get("source") not in names or r.get("target") not in names:
                stats["rel_endpoint_missing"] += 1

    n_ent = max(1, stats["entities"])
    n_rel = max(1, stats["relations"])
    print(json.dumps({**stats, "report": {
        "executor": agent_tasks.AGENT_MODEL,
        "prompt_version": config.PROMPT_VERSION,
        "prompt_fingerprint": prompt_mod.prompt_fingerprint(),
        "entity_per_chunk": round(stats["entities"] / max(1, stats["chunks_cached"]), 2),
        "relation_per_chunk": round(stats["relations"] / max(1, stats["chunks_cached"]), 2),
        "avg_degree": round(2 * stats["relations"] / n_ent, 2),
        "type_distribution": dict(types),
        "relations_used_distinct": len(rels),
        "relations_vocab_size": len(allowed) if allowed else None,
        "evidence_bad_ratio": round(stats["rel_evidence_bad"] / n_rel, 4),
    }}, ensure_ascii=False, indent=2))
    if bad_samples:
        print("--- evidence 不在正文的样例:")
        for s in bad_samples:
            print("   " + s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
