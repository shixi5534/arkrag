"""把待抽 part 文件按「目标行数」聚成作业（job），再把作业按轮转分给 6 个执行子代理。

产物：
  data/arkrag/agent_inbox/_job_<tag>.txt    每行一个 part 文件名（如 S11.part03.jsonl）
  data/arkrag/agent_inbox/_queue_g<N>.txt   该子代理的作业清单（每行一个 job tag）
作业内各 part 同属一片，便于按片 --ingest。
"""
from __future__ import annotations

import sys
from pathlib import Path

INBOX = Path(__file__).resolve().parents[1] / "data" / "arkrag" / "agent_inbox"
TARGET_LINES = 60
AGENTS = 8

OUTBOX_DONE = INBOX.parent / "agent_outbox" / "done"


def main() -> int:
    # 只按 inbox 里现存的 part 文件判定待抽：--plan 已把已入库的块排除掉。
    # 不能拿 agent_outbox/done/ 里同名文件做跳过依据——旧口径遗留的同名归档会让新任务被静默吞掉。
    parts: list[tuple[str, int]] = []
    for path in sorted(INBOX.glob("S*.part*.jsonl")):
        lines = sum(1 for ln in path.open(encoding="utf-8") if ln.strip())
        parts.append((path.name, lines))

    jobs: list[tuple[str, list[str], int]] = []
    cur: list[str] = []
    cur_lines = 0
    cur_shard = ""
    for name, lines in parts:
        shard = name.split(".")[0]
        if cur and (shard != cur_shard or cur_lines + lines > TARGET_LINES):
            jobs.append((cur[0].replace(".", "_"), cur, cur_lines))
            cur, cur_lines = [], 0
        cur.append(name)
        cur_lines += lines
        cur_shard = shard
    if cur:
        jobs.append((cur[0].replace(".", "_"), cur, cur_lines))

    for tag, files, lines in jobs:
        (INBOX / f"_job_{tag}.txt").write_text("\n".join(files) + "\n", encoding="utf-8", newline="\n")

    queues: dict[int, list[str]] = {i: [] for i in range(AGENTS)}
    for i, (tag, _files, _lines) in enumerate(jobs):
        queues[i % AGENTS].append(tag)
    for i in range(AGENTS):
        (INBOX / f"_queue_g{i}.txt").write_text(
            "\n".join(queues[i]) + "\n", encoding="utf-8", newline="\n"
        )

    total_lines = sum(l for _, l in parts)
    print(f"待抽 part {len(parts)} 个 / {total_lines} 行 -> {len(jobs)} 个作业")
    for i in range(AGENTS):
        print(f"  g{i}: {len(queues[i])} 作业  首个 {queues[i][0] if queues[i] else '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
