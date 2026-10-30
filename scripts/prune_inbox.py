"""清掉 inbox 里已入库的陈旧 part 文件，并重建作业分组。

旧口径遗留的 part 文件（S00/S04 等已抽完的片）如果留在 inbox 里，会被当成待抽重新派发，
白烧一轮算力。

## 判定口径（计划 B3）

旧版用 `cache.read_raw(cache_key(content_key, ...))` —— 键里没有 chunk_id。overlap
会让「同正文不同块」互相命中，于是没抽过的块也被判成已入库，陈旧的 part 文件被误删，
那一批块就再也没人派发了。

现在逐行取 `chunk_id` + `content_key` 走 `cache.is_cached()`，带身份核对。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import agent_tasks, cache, config  # noqa: E402

INBOX = config.DATA_DIR / "agent_inbox"


def main() -> int:
    removed = 0
    kept = 0
    unreadable = 0
    for path in sorted(INBOX.glob("S*.part*.jsonl")):
        rows: list[dict] = []
        bad = False
        with path.open(encoding="utf-8") as fh:
            for ln in fh:
                if not ln.strip():
                    continue
                try:
                    rec = json.loads(ln)
                except json.JSONDecodeError:
                    bad = True
                    break
                if not isinstance(rec, dict) or not rec.get("chunk_id") or not rec.get("content_key"):
                    bad = True
                    break
                rows.append(rec)
        if bad:
            # 任务文件自己坏了：不能凭它判断「都已入库」，保留不动，交给人看
            unreadable += 1
            kept += 1
            continue
        if rows and all(
            cache.is_cached(r["content_key"], r["chunk_id"], agent_tasks.AGENT_MODEL) for r in rows
        ):
            path.unlink()
            removed += 1
        else:
            kept += 1

    for pattern in ("_job_*.txt", "_queue_g*.txt", "_pending_index.txt"):
        for stale in INBOX.glob(pattern):
            stale.unlink()

    print(f"删除已全部入库的陈旧 part {removed} 个；保留待抽 {kept} 个（其中不可解析 {unreadable} 个）")
    return 1 if unreadable else 0


if __name__ == "__main__":
    raise SystemExit(main())
