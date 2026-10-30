"""L2 分块：轮次序列 -> 抽取块；块不跨场景（场景=剧情文件，签字四）。

两个必须守住的失效点（计划 D1 / D2）：

D1 轮次身份。旧实现按 `node_id` 去重，而 `node_id` 是源文件给的编号，
会缺失（实测全库 201 轮 id 为 0）也可能重复；一旦同场景出现两个同 id
不同正文的轮次，后一个被静默 `continue` 掉——内容丢了，门还是 PASS。
本模块改用 `(scene_id, ordinal)` 作为轮次**实例身份**（ordinal 是轮次在
场景内的原始序号），`node_id` 降级为源元数据保留在 `node_ids` 里。

D2 尾块。旧实现 `if len(body) < MIN_CHARS and merged: merged[-1].extend(buf)`
无条件并入前一块，从不预测合并后长度——前块已接近硬上限时，一次合并就能
造出超长块，而超长块会让证据定位与上下文预算一起失准。本模块先预测总长，
超硬上限时先在相邻两块间重新平衡，仍不行就保留短尾块并显式写明原因
（`short_reason`），不再假装它不存在。
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from . import config

TARGET = config.CHUNK_TARGET_CHARS
MIN_CHARS = config.CHUNK_MIN_CHARS
MAX_CHARS = config.CHUNK_MAX_CHARS
OVERLAP_TURNS = config.CHUNK_OVERLAP_TURNS

# 短尾块的成因，全部可机械判读；空串表示"不是短块"
SHORT_NONE = ""
SHORT_NO_PREV = "no_previous_chunk"        # 整篇只有一块，场景是硬边界，无处可并
SHORT_MERGE_OVERFLOW = "merge_would_exceed_max"  # 并入前块会超硬上限，且平衡失败
SHORT_REBALANCE_FAILED = "rebalance_failed"      # 平衡尝试过但两侧仍有一侧不达标

# 尾块处置动作，写进块记录便于事后统计
TAIL_NONE = ""
TAIL_MERGED = "merged"
TAIL_REBALANCED = "rebalanced"
TAIL_KEPT = "kept_short"


def render_turn(turn: dict) -> str:
    """对话带说话人前缀；旁白/引文/标题卡按原文落进块正文——
    证据必须能在块文本里逐字定位，加装饰符号会让证据比对整体失效。"""
    if turn["kind"] == "dialogue" and turn["speaker"]:
        return f"{turn['speaker']}：{turn['text']}"
    return turn["text"]


def turn_ordinal(turn: dict, fallback: int) -> int:
    """轮次实例序号。老 scenes.jsonl 没有 ordinal 字段时退回位置序号——
    位置序号同样唯一且稳定，与解析期写入的 ordinal 在语义上等价。

    **注意 fallback 的语义**：调用方传进来的 fallback 必须是**场景内**的原始
    序号，不能是块内下标。这里曾经踩过：`_dedupe` 与 `turn_spans` 都用
    `enumerate(buf)` 传块内下标，于是老 scenes.jsonl（无 ordinal 字段）在
    每一块里都把 ordinal 从 0 重新数，40 轮的长场景只报出 7 个 ordinal，
    `turn_ordinals` 全部是 0..6。覆盖门看 `len(covered)` 时会以为是"轮次
    够用"，而按实例身份核对就会暴露大量轮次无人使用。现在 `split_scene`
    在入口处给每个轮次补齐场景级 ordinal，fallback 只作为兜底。
    """
    value = turn.get("ordinal")
    return value if isinstance(value, int) and value >= 0 else fallback


def _render(buf: list[tuple[str, dict]]) -> str:
    return "\n".join(line for line, _ in buf)


def _render_len(buf: list[tuple[str, dict]]) -> int:
    return len(_render(buf))


def _dedupe(buf: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    """按轮次实例身份去重。

    同一个 ordinal 在同一块里出现两次只可能来自重叠拼接的重复，按实例身份
    去掉是安全的；按 node_id 去重则会把"同编号不同正文"的真实轮次也当成重复
    ——那正是 D1 丢失内容的机理。
    """
    unique: list[tuple[str, dict]] = []
    seen: set[int] = set()
    for fallback, (line, turn) in enumerate(buf):
        key = turn_ordinal(turn, fallback)
        if key in seen:
            continue
        seen.add(key)
        unique.append((line, turn))
    return unique


def _rebalance(
    prev: list[tuple[str, dict]], tail: list[tuple[str, dict]]
) -> tuple[list[tuple[str, dict]], list[tuple[str, dict]]] | None:
    """把前块的尾部若干轮挪到尾块，让两块都达标且都不超硬上限。

    只在真的能达标时返回结果：搬完之后任意一块仍低于 MIN_CHARS 或高于
    MAX_CHARS 都算失败，交给调用方保留短尾块并写明原因。宁可显式留一个短块，
    也不静默造一个超长块，更不假装两块都合规。
    """
    if len(prev) < 2:
        return None
    for move in range(1, len(prev)):
        head = prev[:-move]
        merged_tail = prev[-move:] + tail
        head_len = _render_len(head)
        tail_len = _render_len(merged_tail)
        if head_len < MIN_CHARS or tail_len < MIN_CHARS:
            continue
        if head_len > MAX_CHARS or tail_len > MAX_CHARS:
            continue
        return head, merged_tail
    return None


def _with_ordinal(turn: dict, index: int) -> dict:
    """确保轮次带着**场景级** ordinal 往下走。

    老 `scenes.jsonl`（冻结基线就是这种）没有 `ordinal` 字段。如果放着不管，
    后面每个用到 `enumerate(buf)` 的地方都会拿**块内下标**当 fallback，
    于是每一块的 ordinal 都从 0 重新数 —— 轮次实例身份就退化成"块内位置"，
    `turn_ordinals` 覆盖门会以为覆盖完整，实际大量轮次根本没被任何块引用。

    这里在入口一次性补齐（返回浅拷贝，不改调用方传进来的 scene），
    下游所有 `turn_ordinal(turn, fallback)` 都能拿到稳定且全局唯一的序号。
    """
    value = turn.get("ordinal")
    if isinstance(value, int) and value >= 0:
        return turn
    patched = dict(turn)
    patched["ordinal"] = index
    return patched


def split_scene(scene: dict) -> list[dict]:
    items = [(render_turn(t), _with_ordinal(t, i)) for i, t in enumerate(scene["turns"])]
    pieces: list[list[tuple[str, dict]]] = []
    buf: list[tuple[str, dict]] = []
    size = 0
    # 记录每次开新块时是否放弃重叠，供事后统计"重叠让位于硬上限"的频率
    overlap_dropped = 0
    oversize_turn: list[int] = []
    for fallback, (line, turn) in enumerate(items):
        if buf and size + len(line) + 1 > TARGET:
            pieces.append(buf)
            # 重叠轮次是为了让跨块引用不至于断在边界上；但它本身占长度。
            # 如果一个长轮次 + 重叠就已经顶到硬上限，重叠必须让位——否则
            # 造出的超长块会让证据定位与上下文预算同时失准（计划 D2）。
            tail = buf[-OVERLAP_TURNS:] if OVERLAP_TURNS else []
            if tail and _render_len(tail + [(line, turn)]) > MAX_CHARS:
                tail = []
                overlap_dropped += 1
            buf = tail + [(line, turn)]
            size = sum(len(x[0]) + 1 for x in buf)
            if len(line) > MAX_CHARS:
                # 单轮本身超上限：轮次是原子单位，不许切成半句（切了就定位不到证据），
                # 只能显式登记，让门去判它。
                oversize_turn.append(turn_ordinal(turn, fallback))
        else:
            buf.append((line, turn))
            size += len(line) + 1
            if len(line) > MAX_CHARS:
                oversize_turn.append(turn_ordinal(turn, fallback))
    if buf:
        pieces.append(buf)

    # 尾块处置：先预测合并后长度，再决定合并 / 平衡 / 保留并标记原因
    merged: list[list[tuple[str, dict]]] = []
    tail_action: list[str] = []
    for buf in pieces:
        if not merged:
            merged.append(buf)
            tail_action.append(TAIL_NONE)
            continue
        if _render_len(buf) >= MIN_CHARS:
            merged.append(buf)
            tail_action.append(TAIL_NONE)
            continue
        prev = merged[-1]
        if _render_len(prev + buf) <= MAX_CHARS:
            merged[-1] = prev + buf
            tail_action[-1] = TAIL_MERGED
            continue
        balanced = _rebalance(prev, buf)
        if balanced is not None:
            merged[-1], new_tail = balanced
            merged.append(new_tail)
            tail_action[-1] = TAIL_REBALANCED
            tail_action.append(TAIL_REBALANCED)
            continue
        merged.append(buf)
        tail_action.append(TAIL_KEPT)

    out: list[dict] = []
    for index, buf in enumerate(merged):
        unique = _dedupe(buf)
        text = _render(unique)
        if not text.strip():
            continue
        # 逐轮字符偏移：证据定位（E1）与上下文组装都靠它，
        # 事后拿 evidence 文本在块里 find 只能拿到第一处，拿不到"这一轮在哪"。
        spans: list[dict] = []
        cursor = 0
        for fallback, (line, turn) in enumerate(unique):
            start = cursor
            end = start + len(line)
            spans.append(
                {
                    "ordinal": turn_ordinal(turn, fallback),
                    "node_id": turn.get("node_id"),
                    "start": start,
                    "end": end,
                }
            )
            cursor = end + 1  # "\n"
        speakers = sorted(
            {t["speaker"] for _, t in unique if t["kind"] == "dialogue" and t["speaker"]}
        )
        chars = len(text)
        short_reason = SHORT_NONE
        if chars < MIN_CHARS:
            if len(merged) == 1:
                short_reason = SHORT_NO_PREV
            elif tail_action[index] == TAIL_KEPT:
                short_reason = SHORT_MERGE_OVERFLOW
            else:
                short_reason = SHORT_REBALANCE_FAILED
        out.append(
            {
                "chunk_id": f"{scene['doc_id']}#{index:03d}",
                "doc_id": scene["doc_id"],
                "category": scene["category"],
                "chapter": scene["chapter"],
                "title": scene["title"],
                "scene_index": index,
                "speakers": speakers,
                "text": text,
                "chars": chars,
                "raw_chars": sum(len(t["text"]) for _, t in unique),
                "label_chars": sum(len(line) - len(t["text"]) for line, t in unique),
                "turn_ordinals": sorted({s["ordinal"] for s in spans}),
                "node_ids": sorted(
                    {s["node_id"] for s in spans if isinstance(s["node_id"], int)}
                ),
                "turn_spans": spans,
                "tail_action": tail_action[index] if index < len(tail_action) else TAIL_NONE,
                "short_reason": short_reason,
                "oversize_turn": False,
                "content_key": hashlib.sha256(text.encode("utf-8")).hexdigest()[:24],
            }
        )
    # 单轮超硬上限的块显式标记：轮次是原子单位，不许切成半句（切了就定位不到证据），
    # 所以这不是可以偷偷修掉的 bug，而是必须在门里被看见的事实。
    oversize_set = set(oversize_turn)
    for rec in out:
        rec["oversize_turn"] = bool({s["ordinal"] for s in rec["turn_spans"]} & oversize_set)
        if overlap_dropped:
            rec["overlap_dropped"] = overlap_dropped
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
