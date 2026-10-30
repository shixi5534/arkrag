"""诊断"证据属于隔壁块"的错位：判断是否是子代理输出行整体错位一位。

思路：对每条证据，先在块自身正文里找；找不到就在同一 doc 的前后块里找。
若某块的证据大多落在邻居块里，这张记录就是错位产物，不是抽错。
只读，只打印，不改任何产物。
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import agent_tasks, cache, config, extract  # noqa: E402


def main() -> int:
    chunks = extract.load_chunks()
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        by_doc[c["doc_id"]].append(c)
    for v in by_doc.values():
        v.sort(key=lambda x: x["chunk_id"])

    owner: dict[str, str] = {}
    for c in chunks:
        owner[c["chunk_id"]] = c["text"]

    misplaced = Counter()
    suspects: list[tuple[str, int, int, str]] = []

    for c in chunks:
        raw = cache.read_for_chunk(c, agent_tasks.AGENT_MODEL)
        if raw is None:
            continue
        rs = raw.get("relations") or []
        if not rs:
            continue
        own = c["text"]
        sibs = by_doc.get(c["doc_id"], [])
        idx = next((i for i, s in enumerate(sibs) if s["chunk_id"] == c["chunk_id"]), -1)
        neigh = []
        if idx > 0:
            neigh.append(("-1", sibs[idx - 1]["text"]))
        if 0 <= idx < len(sibs) - 1:
            neigh.append(("+1", sibs[idx + 1]["text"]))

        n_own = n_neigh = n_nowhere = 0
        which = Counter()
        for r in rs:
            ev = r.get("evidence") or ""
            if not ev:
                continue
            if ev in own:
                n_own += 1
                continue
            hit = next((tag for tag, txt in neigh if ev in txt), None)
            if hit:
                n_neigh += 1
                which[hit] += 1
            else:
                n_nowhere += 1
        if n_neigh and n_neigh >= n_own:
            misplaced[c["chunk_id"]] = n_neigh
            suspects.append((c["chunk_id"], n_own, n_neigh, dict(which)))

    cached = sum(1 for c in chunks if cache.is_cached_for_chunk(c, agent_tasks.AGENT_MODEL))
    print(f"疑似错位块 {len(suspects)} / 已入库 {cached}")
    for cid, a, b, w in suspects[:60]:
        print(f"  {cid}  own={a} neigh={b} {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
