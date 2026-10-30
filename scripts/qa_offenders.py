"""列出真正违规的关系行：词表外、证据不在正文、超长、端点不在实体表内。

`qa_evidence.py` 只给总数，这个给具体行，用来回退给产它的那条队列返修。
只读。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import agent_tasks, cache, config, extract, vocab  # noqa: E402


def main() -> int:
    allowed = set(vocab.CANONICAL_RELATIONS)
    off: list[tuple[str, str, str]] = []
    bad: list[tuple[str, str, str]] = []
    missing: list[tuple[str, str]] = []
    long_ev: list[tuple[str, str]] = []

    for c in extract.load_chunks():
        raw = cache.read_for_chunk(c, agent_tasks.AGENT_MODEL)
        if raw is None:
            continue
        ents = raw.get("entities") or []
        rs = raw.get("relations") or []
        names = {e.get("name") for e in ents}
        body = c.get("text") or ""
        for r in rs:
            rel = r.get("relation") or ""
            ev = r.get("evidence") or ""
            if rel not in allowed:
                off.append((c["chunk_id"], rel, ev))
            if ev and ev not in body:
                bad.append((c["chunk_id"], ev, body[:0]))
            if len(ev) > 40:
                long_ev.append((c["chunk_id"], ev))
            if r.get("source") not in names or r.get("target") not in names:
                missing.append((c["chunk_id"], f"{r.get('source')}->{r.get('target')}"))

    print(f"词表外 {len(off)}")
    for row in off[:40]:
        print("  ", row)
    print(f"证据不在正文 {len(bad)}")
    for row in bad[:40]:
        print("  ", row[0], "|", row[1])
    print(f"证据超 40 字 {len(long_ev)}")
    for row in long_ev[:20]:
        print("  ", row[0], "|", row[1])
    print(f"端点缺失 {len(missing)}")
    for row in missing[:20]:
        print("  ", row)
    print("违规块统计（按 chunk 前缀族）:", Counter(b[0].split("/")[0] for b in bad + [(o[0],) for o in off]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
