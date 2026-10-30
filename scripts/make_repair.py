"""返修派发：把质检挑出来的坏块重新下发，并作废它们的旧缓存。

为什么必须作废缓存：`agent_tasks.ingest` 碰到已有缓存就算 hit 直接跳过，
不作废的话返修产物永远进不了库。

## 与旧版的两处区别（计划 C4）

1. **不作废整片、只作废这几块**：旧版还带 `--keep-cache` 之类的口子，容易被理解成
   「返修时可以先清空一批」；现在清理范围严格等于 `--list` 里的 chunk_id。
2. **新旧两种键都清**：只删新键会留下 legacy 记录，下次 `lookup()` 仍会把它当作
   「可证明安全的旧缓存」复用，返修看起来完全没生效。

用法：
    py -3 scripts/make_repair.py --list data/arkrag/repair_list.txt --label R01 [--batch-size 20]
会打印每批的 tag，并把 inbox/<label>.partNN.jsonl 写好。返修完照常 --ingest。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import agent_tasks, cache, extract  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--keep-cache", action="store_true", help="只下发不作废缓存（调试用）")
    args = ap.parse_args()

    wanted = [ln.strip() for ln in Path(args.list).read_text("utf-8").splitlines() if ln.strip()]
    by_id = {c["chunk_id"]: c for c in extract.load_chunks(ids=wanted)}
    missing = [w for w in wanted if w not in by_id]
    if missing:
        print(f"清单里 {len(missing)} 个 chunk_id 在 chunks.jsonl 里找不到：{missing[:5]}", file=sys.stderr)
        return 2

    if not args.keep_cache:
        removed = 0
        touched: list[str] = []
        for cid in wanted:
            c = by_id[cid]
            for p in cache.paths_for_chunk(c, agent_tasks.AGENT_MODEL):
                if p.exists():
                    p.unlink()
                    removed += 1
                    touched.append(p.name)
        print(f"已作废缓存 {removed} 个（含 legacy 键），涉及 {len(wanted)} 块")
        if touched:
            print("  示例:", ", ".join(touched[:5]))

    chunks = [by_id[cid] for cid in wanted]
    info = agent_tasks.prepare_batches(args.label, chunks, args.batch_size)
    print(json.dumps({"label": info["label"], "pending": info["pending"],
                      "tags": [b["tag"] for b in info["batches"]]}, ensure_ascii=False))
    print("规则文件：", Path(agent_tasks.INBOX_DIR) / "batch.system.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
