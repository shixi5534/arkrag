# -*- coding: utf-8 -*-
"""K3：核对触发点——`level_act16mini_st02#009` 的正文长度与尾部切块。

线上世代借全局 `data/arkrag/chunks.jsonl`；新候选自带 `chunks.jsonl`。三处都按同一场景
逐块打 `字数 / content_key`，用来核对「同一块正文变了 → content_key 变了 → 缓存键失效」。

判据：三处都能读出目标场景的块（全局 / 旧 / 新）才谈得上比对；旧世代按设计没有自带
chunks.jsonl（借全局），那不算失败。退出码 0 = 判据成立 / 1 = 全局或新候选缺块。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GEN = ROOT / "data" / "arkrag" / "generations"
OLD = GEN / "g-20261031T035231Z-4eafae"
NEW = GEN / "g-20261103T132152Z-06a741"
GLOBAL = ROOT / "data" / "arkrag" / "chunks.jsonl"

TARGET = "activities/act16mini/level_act16mini_st02"


def load(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            cid = r.get("chunk_id") or r.get("id")
            if cid and cid.startswith(TARGET):
                rows[cid] = r
    return rows


def show(tag: str, rows: dict[str, dict]) -> None:
    print(f"--- {tag} ---")
    for cid in sorted(rows):
        r = rows[cid]
        text = r.get("text") or ""
        ck = r.get("content_key") or r.get("content_hash") or ""
        print(f"  {cid}  字数 {len(text)}  content_key {ck}")
    print()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    print(f"目标场景 {TARGET}\n")

    print("=== 全局 data/arkrag/chunks.jsonl（线上世代借的就是它）===")
    global_rows = load(GLOBAL)
    show("global", global_rows)

    print("=== 旧世代 g-20261031T035231Z-4eafae ===")
    p = OLD / "chunks.jsonl"
    if p.exists():
        show("old 自带", load(p))
    else:
        print("  （该世代没有自带 chunks.jsonl，借全局文件）\n")

    print("=== 新候选 g-20261103T132152Z-06a741 ===")
    new_rows = load(NEW / "chunks.jsonl")
    show("new 自带", new_rows)

    # 判据：全局与新候选都得读出这个场景（旧世代借全局，不算失败）
    return 0 if (global_rows and new_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
