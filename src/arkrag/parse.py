"""L1 解析：storyList 节点时间轴 -> 规整轮次序列。"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from . import atomic, config
from .model import Scene, Turn
from .prop_spec import DROP_CLASS, KIND_BY_PROP, KNOWN_PROPS, TEXT_PROP_FIELD

TAG_RE = re.compile(r"<[^>]+>")
MASKED_SPEAKER_RE = re.compile(r"^[?？]+$")
INTERNAL_SPEAKER_RE = re.compile(r"^(char_|npc_|avg_|group_|system_)|#\d+$")
RE_MAIN_EP = re.compile(r"level_main_(\d{2})-")
RE_MAIN_FAMILY = re.compile(r"^level_([a-z]+)_(\d+)")
RE_MEMORY_OP = re.compile(r"^story_([a-z0-9]+)_")


def canon_doc_id(rel_stem: str) -> str:
    """doc_id 用折叠小写路径：索引文件里 obt/... 与磁盘目录 Obt/... 大小写不一致，
    折叠后文件与索引才能逐条对账；磁盘真实路径另行保存在 path 字段。"""
    return rel_stem.lower()


def chapter_of(rel_stem: str) -> str:
    """分片用的章节标识：主线按集号，活动按活动组，密录按干员，rogue 按 map。"""
    parts = rel_stem.split("/")
    head = parts[0].lower()
    stem = parts[-1]
    if head == "obt":
        sub = parts[1].lower() if len(parts) > 1 else ""
        if sub == "main":
            m = RE_MAIN_EP.search(stem)
            if m:
                return f"main_ep{m.group(1)}"
            fam = RE_MAIN_FAMILY.match(stem)
            if fam:
                return f"main_{fam.group(1)}{fam.group(2)}"
            # 剩余形态（如 level_tr28_end）按族名而非族名+编号归组，避免单文件成一章
            fallback = re.match(r"^level_([a-z]+)", stem)
            return f"main_{fallback.group(1)}" if fallback else "main_other"
        if sub == "memory":
            m = RE_MEMORY_OP.match(stem)
            return f"memory_{m.group(1)}" if m else "memory_other"
        if sub == "rogue":
            return "rogue_" + (parts[2] if len(parts) > 3 else (parts[2] if len(parts) > 2 else stem))
        if sub == "roguelike":
            return "roguelike_" + stem
        return f"{sub}_other"
    if head == "activities":
        return parts[1] if len(parts) > 2 else stem
    return parts[0]


def clean_text(raw: str) -> str:
    # NBSP 来自 story_variables.json 的 nbs 变量替换，会让同一句话产生两种字节形态
    return TAG_RE.sub("", raw).replace("\u00a0", " ").strip()


def classify_speaker(speaker: str) -> tuple[str, bool, bool]:
    masked = bool(MASKED_SPEAKER_RE.match(speaker))
    internal = bool(INTERNAL_SPEAKER_RE.search(speaker))
    return speaker, masked, internal


def parse_story(
    path: Path,
    stats: Counter,
    dropped_log: list | None = None,
    rel_stem: str | None = None,
) -> Scene | None:
    """rel_stem 可注入：负例测试用临时文件，不在 STORY_ROOT 下。"""

    def _stem(p: Path) -> str:
        if rel_stem is not None:
            return rel_stem
        return p.relative_to(config.STORY_ROOT).with_suffix("").as_posix()

    def _drop(reason: str) -> None:
        stats["drop_reason:" + reason] += 1
        if dropped_log is not None:
            dropped_log.append({"doc_id": canon_doc_id(_stem(path)), "reason": reason})

    rel = _stem(path)
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        stats["bom_files"] += 1
        raw = raw[3:]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        stats["decode_error"] += 1
        _drop("decode")
        return None
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        stats["json_error"] += 1
        _drop("json")
        return None

    nodes = doc.get("storyList")
    if not isinstance(nodes, list):
        _drop("no_storylist")
        return None

    category = rel.split("/")[0].lower() + "/" + (rel.split("/")[1].lower() if rel.count("/") > 1 else "")
    if rel.startswith("activities/"):
        category = "activities"
    scene = Scene(
        doc_id=canon_doc_id(rel),
        category=category,
        chapter=chapter_of(rel),
        title=(doc.get("storyName") or "").strip(),
        event_id=str(doc.get("eventid") or ""),
        entry_type=str(doc.get("entryType") or ""),
        story_code=str(doc.get("storyCode") or ""),
    )

    if "OPTIONTRACE" in doc:
        trace = doc.get("OPTIONTRACE")
        stats["drop_class:branch_condition"] += len(trace) if isinstance(trace, list) else 1
        stats["optiontrace_files"] += 1

    dropped: Counter = Counter()
    for node in nodes:
        prop = node.get("prop")
        if not isinstance(prop, str):
            stats["drop_class:malformed_node"] += 1
            dropped["malformed_node"] += 1
            continue
        low = prop.lower()
        stats["prop_seen:" + low] += 1
        if low not in KNOWN_PROPS:
            stats["drop_class:unknown_prop"] += 1
            dropped["unknown_prop:" + low] += 1
            continue
        field = TEXT_PROP_FIELD.get(low)
        if field is None:
            cls = DROP_CLASS[low]
            stats["drop_class:" + cls] += 1
            dropped[cls] += 1
            continue
        attrs = node.get("attributes") or {}
        value = attrs.get(field)
        if not isinstance(value, str) or not value.strip():
            stats["drop_class:empty_text"] += 1
            dropped["empty_text"] += 1
            continue
        body = clean_text(value)
        if not body:
            stats["drop_class:markup_only"] += 1
            dropped["markup_only"] += 1
            continue

        speaker_raw = attrs.get("name")
        speaker = speaker_raw.strip() if isinstance(speaker_raw, str) else ""
        base_kind = KIND_BY_PROP[low]
        if base_kind is None:
            kind = "dialogue" if speaker else "narration"
        else:
            kind = base_kind
        masked = False
        if speaker:
            _, masked, internal = classify_speaker(speaker)
            if internal:
                stats["speaker_internal"] += 1
            if masked:
                stats["speaker_masked"] += 1

        scene.turns.append(
            Turn(
                kind=kind,
                speaker=speaker,
                text=body,
                node_id=int(node.get("id") or 0),
                prop=low,
                speaker_masked=masked,
                ordinal=len(scene.turns),
            )
        )
        stats["turn_kind:" + kind] += 1
        stats["turn_chars:" + kind] += len(body)

    if not scene.turns:
        _drop("zero_turns")
        return None
    scene.dropped = dict(dropped)
    return scene


def iter_story_files() -> list[Path]:
    return sorted(
        p
        for p in config.STORY_ROOT.rglob("*.json")
        if p.name != "story_variables.json" and "__MACOSX" not in p.parts
    )


def parse_corpus(
    out_path: Path | None = None,
    limit: int | None = None,
    dropped_log: list | None = None,
) -> tuple[list[Scene], Counter]:
    """解析全语料。`out_path` 的写盘走 `atomic.atomic_write_stream`。

    历史教训（2026-10-30）：这里原来是 `out_path.open("w")` 直写，**绕过了
    `atomic`**，于是冻结写保护对 scenes.jsonl 完全失效——重跑 L1 把 71 MB 的
    冻结产物写成了 7 字节的 `rebuilt`，直到事后跑 `snapshot_baseline --verify`
    才发现。产物写入必须只有一条通道，否则"每个调用点记得调 guard"必然漏。
    """
    stats: Counter = Counter()
    scenes: list[Scene] = []
    files = iter_story_files()
    if limit:
        files = files[:limit]
    for path in files:
        scene = parse_story(path, stats, dropped_log)
        if scene is not None:
            scenes.append(scene)
    stats["files_scanned"] = len(files)
    stats["scenes_kept"] = len(scenes)
    if out_path is not None:
        def _writer(fh) -> None:
            for scene in scenes:
                fh.write(json.dumps(scene.as_dict(), ensure_ascii=False) + "\n")

        atomic.atomic_write_stream(out_path, _writer)
    return scenes, stats


def main() -> int:
    out = config.scenes_path()
    dropped_log: list = []
    scenes, stats = parse_corpus(out, dropped_log=dropped_log)
    atomic.atomic_write_json(config.l1_dropped_path(), dropped_log)
    atomic.atomic_write_json(config.l1_stats_path(), dict(stats.most_common()))
    kinds = {k.split(":", 1)[1]: v for k, v in stats.items() if k.startswith("turn_kind:")}
    total_turns = sum(kinds.values())
    dialogue = kinds.get("dialogue", 0)
    narration = kinds.get("narration", 0)
    unknown = stats.get("drop_class:unknown_prop", 0)
    print("=== L1 解析结果 ===")
    print(f"扫描文件数     : {stats['files_scanned']}")
    print(f"保留场景数     : {stats['scenes_kept']}   丢弃: {stats['files_scanned'] - stats['scenes_kept']}")
    for reason in sorted(k for k in stats if k.startswith("drop_reason:")):
        print(f"  {reason} = {stats[reason]}")
    print(f"轮次总数       : {total_turns:,}")
    for k in ("dialogue", "narration", "quote", "title_card"):
        share = kinds.get(k, 0) / max(1, total_turns)
        print(f"  {k:10s} {kinds.get(k, 0):8,}  {share:6.1%}   字数 {stats.get('turn_chars:'+k, 0):10,}")
    print(f"对话/旁白比    : {(dialogue / narration if narration else float('inf')):.2f}  (门线区间 1.5-4，见已知偏差)")
    print(f"旁白占比探针   : {narration / max(1, dialogue + narration):.3f}  (要求 <0.30)")
    print(f"掩码说话人     : {stats['speaker_masked']:,}   内部 id 说话人: {stats['speaker_internal']:,}")
    print("丢弃归类       : " + ", ".join(f"{k.split(':',1)[1]}:{v}" for k, v in sorted(stats.items()) if k.startswith("drop_class:")))
    print(f"unknown prop   : {unknown:,}  占比 {unknown / max(1, total_turns + sum(v for k, v in stats.items() if k.startswith('drop_class:'))):.5%}")
    print(f"scenes -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
