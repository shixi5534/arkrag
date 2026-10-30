"""把待抽作业一分为二：原 8 路队列各留前一半，后一半交给新增 g8–g15。

为什么要拆：8 路跑完 227 个作业要 4 小时以上，扩到 16 路能压到 2 小时左右。
拆分必须一次性切干净——两个子代理同时追加同一个 part 文件会把行写交错，事后无法确认哪份可信。
所以这里先算出"还没开工"的作业全集，再按原队列各自的后半段整体搬走，保证同一 job 只有一个主人。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INBOX = ROOT / "data" / "arkrag" / "agent_inbox"
DONE = ROOT / "data" / "arkrag" / "agent_outbox" / "done"
OUTBOX = ROOT / "data" / "arkrag" / "agent_outbox"

KEEP_HEAD = 15  # 原队列各自保留的作业数


def parts_of(tag: str) -> list[str]:
    path = INBOX / f"_job_{tag}.txt"
    if not path.exists():
        return []
    return [ln.strip() for ln in path.open(encoding="utf-8") if ln.strip()]


def started(tag: str) -> bool:
    """作业里任何一个 part 已经产出或已归档，就算这活儿已经在某人手上。"""
    for part in parts_of(tag):
        if (OUTBOX / part).exists() or (DONE / part).exists():
            return True
    return False


def main() -> int:
    tails: list[list[str]] = [[] for _ in range(8)]
    kept: dict[str, list[str]] = {}

    for g in range(8):
        qpath = INBOX / f"_queue_g{g}.txt"
        jobs = [ln.strip() for ln in qpath.open(encoding="utf-8") if ln.strip()]
        head, tail = jobs[:KEEP_HEAD], jobs[KEEP_HEAD:]
        started_tail = [j for j in tail if started(j)]
        clean_tail = [j for j in tail if not started(j)]
        if started_tail:
            # 已经有人动过的留在原队列，避免两个子代理写同一个文件
            head = head + started_tail
        kept[qpath.name] = head
        # 尾部按队列编号轮流摊到 8 个新队列，让各新队的片种尽量均匀
        for i, job in enumerate(clean_tail):
            tails[i % 8].append(job)

    for name, jobs in kept.items():
        (INBOX / name).write_text("\n".join(jobs) + "\n", encoding="utf-8", newline="\n")

    total = 0
    for g in range(8):
        jobs = tails[g]
        total += len(jobs)
        (INBOX / f"_queue_g{g + 8}.txt").write_text(
            "\n".join(jobs) + "\n", encoding="utf-8", newline="\n"
        )
        print(f"_queue_g{g + 8}: {len(jobs)} 作业  首个 {jobs[0] if jobs else '-'}")

    print(f"原队列保留 {sum(len(v) for v in kept.values())} 作业；新队列接手 {total} 作业")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
