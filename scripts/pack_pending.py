# -*- coding: utf-8 -*-
"""按「缓存里还没有」的块重新打包任务文件。

与 pack_big.py 的区别：不看 outbox 文件是否完整，只看缓存里有没有这一块的产出。
这样半成品包（只写了一部分）会自动只补缺失的块，已入库的不会重复下发。

## 判定口径（计划 B3）

旧版用 `cache.read_raw(cache_key(content_key, ...))` 判缓存：键里没有 chunk_id，
overlap 造成的「同正文不同块」会互相命中，于是**这一块其实没抽过也会被算成已入库**，
永远补不下来。

现在走 `cache.is_cached(content_key, chunk_id, ...)`，带身份核对。

用法：
  py -3 scripts/pack_pending.py --per-pack 80 --label B02 --packs-file data/arkrag/_pending_packs.txt
产出：
  data/arkrag/agent_inbox/B02.partNN.jsonl
  data/arkrag/_packs_B02.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import agent_tasks, atomic, cache, config  # noqa: E402

INBOX = config.DATA_DIR / "agent_inbox"


def is_cached(content_key: str, chunk_id: str) -> bool:
    return cache.is_cached(content_key, chunk_id, agent_tasks.AGENT_MODEL)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-pack", type=int, default=80)
    ap.add_argument("--label", default="B02")
    ap.add_argument("--packs-file", default=str(config.DATA_DIR / "_pending_packs.txt"))
    a = ap.parse_args()

    packs_file = Path(a.packs_file)
    if not packs_file.exists():
        print(f"待打包清单不存在：{packs_file}", file=sys.stderr)
        return 2
    pack_names = [ln.strip() for ln in packs_file.read_text("utf-8").splitlines() if ln.strip()]

    # 收集所有待做块（按包顺序、包内保序），并跳过已缓存的
    rows: list[dict] = []
    seen: set[str] = set()
    missing_packs: list[str] = []
    for pn in pack_names:
        path = INBOX / pn
        if not path.exists():
            missing_packs.append(pn)
            continue
        with path.open(encoding="utf-8") as fh:
            for ln in fh:
                if not ln.strip():
                    continue
                r = json.loads(ln)
                cid = r["chunk_id"]
                if cid in seen:
                    continue
                seen.add(cid)
                if is_cached(r["content_key"], cid):
                    continue
                rows.append(r)

    for pn in missing_packs:
        print(f"WARN 清单里的包文件不存在：{pn}", file=sys.stderr)
    print("pending blocks needing extraction:", len(rows))

    # 按场景前缀（片号）分组，尽量同片同包
    def shard_of(cid: str) -> str:
        head = cid.split("/")[0]
        return head + "/" + (cid.split("/")[1] if "/" in cid else "")

    buckets: dict[str, list[dict]] = {}
    order: list[str] = []
    for r in rows:
        s = shard_of(r["chunk_id"])
        if s not in buckets:
            buckets[s] = []
            order.append(s)
        buckets[s].append(r)

    packs: list[list[dict]] = []
    cur: list[dict] = []
    for s in order:
        for r in buckets[s]:
            if cur and len(cur) >= a.per_pack:
                packs.append(cur)
                cur = []
            cur.append(r)
    if cur:
        packs.append(cur)

    out: list[str] = []
    for i, plist in enumerate(packs):
        name = f"{a.label}.part{i:02d}.jsonl"
        body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in plist)
        atomic.atomic_write_text(INBOX / name, body)
        out.append(f"{name}\t{len(plist)} blocks")

    atomic.atomic_write_text(config.DATA_DIR / f"_packs_{a.label}.txt", "\n".join(out) + "\n")
    print("packs:", len(packs))
    for line in out:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
