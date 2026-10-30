"""把 g8–g15 的作业并回 g0–g7。

起因：想扩到 16 路，但本会话的队友上限是 8，g8–g15 派不出去。那 107 个作业如果留在
没人读的队列文件里就会永远不跑，所以原样并回原队列末尾，让在手的人接着做。
并回顺序保持原队列的相对次序（g8 是 g0 的后半段，g9 是 g1 的后半段，以此类推）。
"""
from __future__ import annotations

from pathlib import Path

INBOX = Path(__file__).resolve().parents[1] / "data" / "arkrag" / "agent_inbox"


def read(name: str) -> list[str]:
    p = INBOX / name
    if not p.exists():
        return []
    return [ln.strip() for ln in p.open(encoding="utf-8") if ln.strip()]


def main() -> int:
    moved = 0
    for g in range(8):
        base = read(f"_queue_g{g}.txt")
        tail = read(f"_queue_g{g + 8}.txt")
        merged = base + tail
        (INBOX / f"_queue_g{g}.txt").write_text(
            "\n".join(merged) + "\n", encoding="utf-8", newline="\n"
        )
        (INBOX / f"_queue_g{g + 8}.txt").unlink(missing_ok=True)
        moved += len(tail)
        print(f"g{g}: {len(base)} + {len(tail)} = {len(merged)}")
    all_jobs = [j for g in range(8) for j in read(f"_queue_g{g}.txt")]
    print(f"并回 {moved} 个作业；合计 {len(all_jobs)} 条，去重后 {len(set(all_jobs))} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
