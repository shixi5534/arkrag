"""按队列汇报真实剩余量：看每个作业里还有多少块没进缓存。

`--status` 只按片汇总，看不出"8 条队列各剩几个作业"。这里按 _job_*.txt 逐个算，
用来判断什么时候该重新摊派、哪条队列会先空。

判定口径（计划 B3）：旧版按 `content_key` 拼旧键查缓存，键里没有 chunk_id，
overlap 的同正文块会互相命中，剩余量会被系统性低估。现在带 chunk_id 核对。
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
    def cached(content_key: str, chunk_id: str) -> bool:
        return cache.is_cached(content_key, chunk_id, agent_tasks.AGENT_MODEL)

    job_left: dict[str, int] = {}
    job_total: dict[str, int] = {}
    for job_file in sorted(INBOX.glob("_job_*.txt")):
        tag = job_file.stem[len("_job_"):]
        left = 0
        total = 0
        with job_file.open(encoding="utf-8") as jf:
            parts = [ln.strip() for ln in jf if ln.strip()]
        for part in parts:
            task = INBOX / part
            if not task.exists():
                continue
            with task.open(encoding="utf-8") as tf:
                for ln in tf:
                    if not ln.strip():
                        continue
                    rec = json.loads(ln)
                    total += 1
                    if not cached(rec.get("content_key", ""), rec.get("chunk_id", "")):
                        left += 1
        job_left[tag] = left
        job_total[tag] = total

    grand_left = 0
    grand_total = 0
    for g in range(8):
        qpath = INBOX / f"_queue_g{g}.txt"
        if not qpath.exists():
            continue
        with qpath.open(encoding="utf-8") as qf:
            jobs = [ln.strip() for ln in qf if ln.strip()]
        left = sum(job_left.get(j, 0) for j in jobs)
        total = sum(job_total.get(j, 0) for j in jobs)
        open_jobs = sum(1 for j in jobs if job_left.get(j, 0) > 0)
        grand_left += left
        grand_total += total
        print(f"g{g}: 作业 {len(jobs):2d}（未完 {open_jobs:2d}）  剩余块 {left:5d} / 共 {total:5d}")

    print(f"合计剩余 {grand_left} / 共 {grand_total} 块")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
