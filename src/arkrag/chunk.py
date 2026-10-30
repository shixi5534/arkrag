"""L2 分块：轮次序列 -> 抽取块；块不跨场景（场景=剧情文件，签字四）。"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from . import config

TARGET = config.CHUNK_TARGET_CHARS
MIN_CHARS = config.CHUNK_MIN_CHARS
OVERLAP_TURNS = config.CHUNK_OVERLAP_TURNS


def render_turn(turn: dict) -> str:
    """对话带说话人前缀；旁白/引文/标题卡按原文落进块正文——
    证据必须能在块文本里逐字定位，加装饰符号会让证据比对整体失效。"""
    if turn["kind"] == "dialogue" and turn["speaker"]:
        return f"{turn['speaker']}：{turn['text']}"
    return turn["text"]


def split_scene(scene: dict) -> list[dict]:
    items = [(render_turn(t), t) for t in scene["turns"]]
    chunks: list[list[tuple[str, dict]]] = []
    buf: list[tuple[str, dict]] = []
    size = 0
    for line, turn in items:
        if buf and size + len(line) + 1 > TARGET:
            chunks.append(buf)
            tail = buf[-OVERLAP_TURNS:] if OVERLAP_TURNS else []
            buf = tail + [(line, turn)]
            size = sum(len(x[0]) + 1 for x in buf)
        else:
            buf.append((line, turn))
            size += len(line) + 1
    if buf:
        chunks.append(buf)

    # 尾块过小并入前一块；丢弃等于丢内容
    merged: list[list[tuple[str, dict]]] = []
    for buf in chunks:
        body = "".join(line for line, _ in buf)
        if len(body) < MIN_CHARS and merged:
            merged[-1].extend(buf)
        else:
            merged.append(buf)

    out = []
    for index, buf in enumerate(merged):
        # 重叠轮次在同一片里可能出现两次，去重后才是本片真实覆盖的原文
        unique: list[tuple[str, dict]] = []
        seen: set[int] = set()
        for line, turn in buf:
            if turn["node_id"] in seen:
                continue
            seen.add(turn["node_id"])
            unique.append((line, turn))
        text = "\n".join(line for line, _ in unique)
        if not text.strip():
            continue
        speakers = sorted(
            {t["speaker"] for _, t in unique if t["kind"] == "dialogue" and t["speaker"]}
        )
        chunk_id = f"{scene['doc_id']}#{index:03d}"
        out.append(
            {
                "chunk_id": chunk_id,
                "doc_id": scene["doc_id"],
                "category": scene["category"],
                "chapter": scene["chapter"],
                "title": scene["title"],
                "scene_index": index,
                "speakers": speakers,
                "text": text,
                "chars": len(text),
                "raw_chars": sum(len(t["text"]) for _, t in unique),
                "label_chars": sum(len(line) - len(t["text"]) for line, t in unique),
                "turn_ids": sorted(seen),
                "content_key": hashlib.sha256(text.encode("utf-8")).hexdigest()[:24],
            }
        )
    return out


def chunk_all(scenes_path: Path) -> list[dict]:
    out: list[dict] = []
    with scenes_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.extend(split_scene(json.loads(line)))
    return out


def pack_shards(chunks: list[dict], target_per_shard: int = 200) -> list[dict]:
    """按章节装箱（LPT 贪心）：同一章节不许拆到两片，片间块数极差控制在 2 倍内。"""
    per_chapter: dict[str, dict] = defaultdict(lambda: {"blocks": 0, "docs": set(), "categories": set()})
    for c in chunks:
        key = f"{c['category']}::{c['chapter']}"
        entry = per_chapter[key]
        entry["blocks"] += 1
        entry["docs"].add(c["doc_id"])
        entry["categories"].add(c["category"])

    n_shards = max(1, round(len(chunks) / target_per_shard))
    bins: list[dict] = [
        {"shard": f"S{i:02d}", "blocks": 0, "chapters": [], "docs": 0} for i in range(n_shards)
    ]
    for key, entry in sorted(per_chapter.items(), key=lambda kv: (-kv[1]["blocks"], kv[0])):
        target = min(bins, key=lambda b: (b["blocks"], b["shard"]))
        target["blocks"] += entry["blocks"]
        target["docs"] += len(entry["docs"])
        target["chapters"].append({"chapter": key, "blocks": entry["blocks"], "docs": len(entry["docs"])})
    for b in bins:
        b["chapters"].sort(key=lambda x: (-x["blocks"], x["chapter"]))
    return [b for b in bins if b["blocks"] > 0]


def shard_membership(chunks: list[dict], shards: list[dict]) -> dict[str, str]:
    owner: dict[str, str] = {}
    for b in shards:
        for entry in b["chapters"]:
            category, chapter = entry["chapter"].split("::", 1)
            for c in chunks:
                if c["category"] == category and c["chapter"] == chapter:
                    owner[c["chunk_id"]] = b["shard"]
    return owner
