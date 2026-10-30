"""新缓存键的回退验证脚本（B4 影子校验）。

计划风险 1：严格校验导致大量旧缓存失效。这里的做法是**影子校验**——
不采信旧键，而是对每个真实 Chunk 走新键 + legacy 身份核对，看有多少条能
被证明安全地复用、多少条必须重抽。

输出三类计数：current（新键命中）/ legacy（旧记录身份全部相符）/ miss / invalid。
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import cache, config  # noqa: E402

MODEL = "agent:qoder-subagent"


def load_chunks(limit: int | None = None) -> list[dict]:
    out = []
    with config.chunks_path().open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    if limit:
        random.Random(20261030).shuffle(out)
        out = out[:limit]
    return out


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    chunks = load_chunks(n or None)
    print(f"待查 Chunk {len(chunks):,}")

    layers = Counter()
    reasons = Counter()
    invalid_samples: list[str] = []
    legacy_samples: list[dict] = []

    for c in chunks:
        lk = cache.lookup_for_chunk(c, MODEL)
        layers[lk.layer] += 1
        if lk.layer == "invalid":
            reasons[lk.reason.split("(")[0]] += 1
            if len(invalid_samples) < 10:
                invalid_samples.append(f"{c['chunk_id']} :: {lk.reason}")
        elif lk.layer == "legacy" and len(legacy_samples) < 3:
            legacy_samples.append(
                {
                    "chunk_id": c["chunk_id"],
                    "content_key": c["content_key"],
                    "legacy_key": lk.key,
                    "entities": len(lk.raw.get("entities") or []),
                    "relations": len(lk.raw.get("relations") or []),
                }
            )

    total = sum(layers.values())
    reused = layers["current"] + layers["legacy"]
    print()
    print(f"current(新键)  {layers['current']:>7,}")
    print(f"legacy(可证安全){layers['legacy']:>7,}")
    print(f"miss            {layers['miss']:>7,}")
    print(f"invalid(须隔离) {layers['invalid']:>7,}")
    print(f"可复用比例      {reused}/{total} = {reused / total:.4%}" if total else "n/a")
    if reasons:
        print("\ninvalid 原因分布：")
        for k, v in reasons.most_common():
            print(f"  {k}: {v:,}")
    if invalid_samples:
        print("\ninvalid 样例：")
        for s in invalid_samples:
            print("  " + s)
    if legacy_samples:
        print("\nlegacy 复用样例：")
        for s in legacy_samples:
            print("  " + json.dumps(s, ensure_ascii=False))
    # 非零退出：任何 invalid 都必须被看见，不能安静通过
    return 1 if layers["invalid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
