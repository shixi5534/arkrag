"""L0 语料准入体检：全量遍历，不做抽样。

产出 data/arkrag/l0_manifest.json 与基线数字，供后续阶段的门线比对。
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from . import config

BOM = "﻿"

# 污染三类的语料等价物：玩家选项 / 分支条件 / 玩法与标题屏节点
POLLUTION_PROPS = {
    "decision": "player_option",
    "predicate": "branch_condition",
    "startbattle": "gameplay",
    "tutorial": "gameplay",
    "skiptotis": "gameplay",
    "skiptothis": "gameplay",
    "header": "title_screen",
}


def _rel_key(path: Path) -> str:
    """story 根下的相对路径主干，作为 doc_id 的核心（与文件内容无关，折叠小写）。"""
    return path.relative_to(config.STORY_ROOT).with_suffix("").as_posix().lower()


def scan_file(path: Path) -> dict:
    raw = path.read_bytes()
    rec = {
        "doc_id": _rel_key(path),
        "bytes": len(raw),
        "has_bom": raw.startswith(BOM.encode("utf-8")),
        "crlf": b"\r\n" in raw,
        "decode_error": None,
    }
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        rec["decode_error"] = str(exc)
        return rec
    if text.startswith(BOM):
        text = text[1:]
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        rec["decode_error"] = "json:" + str(exc)
        return rec

    rec["top_keys"] = sorted(doc.keys())
    nodes = doc.get("storyList") or []
    rec["node_count"] = len(nodes)
    rec["eventid"] = doc.get("eventid")
    rec["story_name"] = doc.get("storyName")
    rec["entry_type"] = doc.get("entryType")
    rec["has_optiontrace"] = "OPTIONTRACE" in doc

    props_raw: Counter = Counter()
    props_lower: Counter = Counter()
    attr_keys: dict[str, Counter] = defaultdict(Counter)
    spoken = 0
    narrator = 0
    masked = 0
    spoken_chars = 0
    narrator_chars = 0
    playable_chars = 0  # 台词 + 旁白，不含字幕/标题卡，供字数口径的对话/旁白比
    speakers: Counter = Counter()
    content_chars = 0
    title_card_nodes = 0
    empty_content = 0
    pollution: Counter = Counter()

    for node in nodes:
        prop = node.get("prop")
        if not isinstance(prop, str):
            props_raw["<missing>"] += 1
            props_lower["<missing>"] += 1
            continue
        props_raw[prop] += 1
        low = prop.lower()
        props_lower[low] += 1
        attrs = node.get("attributes") or {}
        for key in attrs:
            attr_keys[low][key] += 1
        if low in POLLUTION_PROPS:
            pollution[POLLUTION_PROPS[low]] += 1
        if low == "name":
            content = attrs.get("content")
            if not isinstance(content, str) or not content.strip():
                empty_content += 1
                continue
            speaker = attrs.get("name")
            n = len(content.strip())
            if isinstance(speaker, str) and speaker.strip():
                spoken += 1
                spoken_chars += n
                playable_chars += n
                speakers[speaker.strip()] += 1
                if "？" in speaker or "?" in speaker:
                    masked += 1
            else:
                narrator += 1
                narrator_chars += n
                playable_chars += n
            content_chars += n
        elif low in ("subtitle", "multiline"):
            txt = attrs.get("text")
            if isinstance(txt, str) and txt.strip():
                title_card_nodes += 1
                content_chars += len(txt.strip())

    rec.update(
        {
            "props_raw": dict(props_raw),
            "props_lower": dict(props_lower),
            "attr_keys": {k: dict(v) for k, v in attr_keys.items()},
            "spoken": spoken,
            "narrator": narrator,
            "masked_speakers": masked,
            "spoken_chars": spoken_chars,
            "narrator_chars": narrator_chars,
            "playable_chars": playable_chars,
            "speakers": dict(speakers),
            "content_chars": content_chars,
            "title_cards": title_card_nodes,
            "empty_name_nodes": empty_content,
            "pollution": dict(pollution),
            "story_list_missing": "storyList" not in doc,
        }
    )
    return rec


def load_index(name: str) -> dict:
    path = config.SOURCE_DIR / name
    if not path.exists():
        return {}
    return json.loads(path.read_text("utf-8"))


def norm_index_key(key: str) -> str:
    """索引 key 形态 activities/a001/level_x 或 obt/main/level_x -> story 根相对主干。

    索引用小写 obt/，磁盘目录是 Obt/；doc_id 也折叠小写，两边才可比。
    """
    k = key.strip("/").lower()
    for prefix in ("gamedata/story/", "zh_cn/gamedata/story/"):
        if k.startswith(prefix):
            k = k[len(prefix) :]
    return k


def classify_empty(rec: dict) -> str:
    props = set(rec.get("props_lower", {}))
    if "video" in props:
        return "video_only"
    if props <= {"header", "skipnode", "stopmusic", "background", "dialog", "delay", "character", "theater"}:
        return "cutscene_scaffold_only"
    return "no_content_nodes"


def main() -> int:
    if not config.STORY_ROOT.exists():
        print(f"FAIL: story 根目录不存在 {config.STORY_ROOT}", file=sys.stderr)
        return 2

    files = sorted(p for p in config.STORY_ROOT.rglob("*.json") if p.name != "story_variables.json")
    records = [scan_file(p) for p in files]

    by_cat: Counter = Counter()
    for rec in records:
        head = rec["doc_id"].split("/")[0]
        by_cat[head] += 1

    prop_hist: Counter = Counter()
    prop_raw_hist: Counter = Counter()
    attr_key_hist: dict[str, Counter] = defaultdict(Counter)
    pollution_total: Counter = Counter()
    top_key_hist: Counter = Counter()
    decode_failures = []
    story_name_missing: list[str] = []
    bom_files = []
    crlf_files = []
    zero_turn_files = []
    speakers_all: Counter = Counter()
    nar_docs: list[str] = []
    spoken = narrator = masked = 0
    spoken_chars = narrator_chars = playable_chars = 0
    parsed_chars = 0

    for rec in records:
        if rec.get("decode_error"):
            decode_failures.append({"doc_id": rec["doc_id"], "error": rec["decode_error"]})
            continue
        if rec["has_bom"]:
            bom_files.append(rec["doc_id"])
        if rec["crlf"]:
            crlf_files.append(rec["doc_id"])
        for k, v in rec.get("props_raw", {}).items():
            prop_raw_hist[k] += v
        for k, v in rec.get("props_lower", {}).items():
            prop_hist[k] += v
        for prop, keys in rec.get("attr_keys", {}).items():
            for k, v in keys.items():
                attr_key_hist[prop][k] += v
        for k, v in rec.get("pollution", {}).items():
            pollution_total[k] += v
        for k in rec.get("top_keys", []):
            top_key_hist[k] += 1
        spoken += rec["spoken"]
        narrator += rec["narrator"]
        masked += rec["masked_speakers"]
        spoken_chars += rec["spoken_chars"]
        narrator_chars += rec["narrator_chars"]
        playable_chars += rec["playable_chars"]
        parsed_chars += rec["content_chars"]
        speakers_all.update(rec["speakers"])
        if rec["spoken"] + rec["narrator"] + rec["title_cards"] == 0:
            zero_turn_files.append(
                {"doc_id": rec["doc_id"], "nodes": rec["node_count"], "reason": classify_empty(rec)}
            )
        if rec["narrator"] > 0:
            nar_docs.append(rec["doc_id"])
        if not (rec.get("story_name") or "").strip():
            story_name_missing.append(rec["doc_id"])

    # 文件 <-> 索引对账（wordcount / storyinfo / summaries 三本索引，双向差集）
    indexes = {}
    for name in ("wordcount.json", "storyinfo.json", "summaries.json"):
        data = load_index(name)
        keys = set()
        if name == "wordcount.json":
            for per_event in data.values():
                keys |= set(per_event)
        else:
            keys = set(data)
        indexes[name] = {"entries": len(data), "keys": {norm_index_key(k) for k in keys}}

    doc_ids = {r["doc_id"] for r in records}
    extra_raw = load_index("extrastory.json")
    extra_ids = {
        norm_index_key(e.get("storyTxt", "")) for e in (extra_raw.get("extra") or []) if e.get("storyTxt")
    }

    def explain(doc_id: str) -> str:
        if doc_id in extra_ids:
            return "listed_in_extrastory.json（活动隐藏关，另册索引）"
        if doc_id.startswith("obt/rogue") or doc_id.startswith("obt/roguelike"):
            return "rogue/roguelike 内容不在该索引覆盖范围"
        return "UNEXPLAINED"

    recon = {}
    for name, idx in indexes.items():
        keys = idx["keys"]
        missing_file = sorted(keys - doc_ids)
        missing_index = sorted(doc_ids - keys)
        recon[name] = {
            "index_keys": len(keys),
            "files": len(doc_ids),
            "index_without_file": missing_file,
            "file_without_index": missing_index,
            "file_without_index_explained": {d: explain(d) for d in missing_index},
            "unexplained": [d for d in missing_index if explain(d) == "UNEXPLAINED"],
        }

    wordcount_total = sum(
        v for per_event in load_index("wordcount.json").values() for v in per_event.values()
    )

    manifest = {
        "schema_version": config.SCHEMA_VERSION,
        "source_dir": str(config.SOURCE_DIR),
        "corpus_source": {
            "repo": "3aKHP/ArknightsStoryJson",
            "release_tag": "akdp-26-10-14-13-45-10-99469376-v1",
            "asset": "zh_CN.zip",
            "zip_bytes": 35904420,
            "zip_sha256": "46b106207bac11af75fe70169e49b00f25e360da0f32d41824e29b52a01e0aeb",
            "compatibility_contract": "prts-story-v1",
            "factory_manifest_events": 485,
            "factory_manifest_chapters": 2032,
            "factory_manifest_json_files": 2360,
        },
        "baseline": {
            "story_files": len(records),
            "by_category": dict(by_cat),
            "story_bytes": sum(r["bytes"] for r in records),
            "dialogue_turns": spoken,
            "narration_turns": narrator,
            "dialogue_chars": spoken_chars,
            "narration_chars": narrator_chars,
            "playable_chars": playable_chars,
            "narration_docs": nar_docs,
            "total_docs": len(records),
            "masked_speaker_turns": masked,
            "title_card_turns": sum(r.get("title_cards", 0) for r in records),
            "parsed_content_chars": parsed_chars,
            "wordcount_index_chars": wordcount_total,
            "distinct_speakers": len(speakers_all),
            "top_speakers": speakers_all.most_common(40),
            "pollution_nodes": dict(pollution_total),
            "empty_name_nodes": sum(r.get("empty_name_nodes", 0) for r in records),
            "story_name_missing": len(story_name_missing),
            "story_name_missing_sample": story_name_missing[:10],
        },
        "encoding": {
            "bom_files": bom_files,
            "crlf_files_count": len(crlf_files),
            "crlf_files_sample": crlf_files[:20],
            "decode_failures": decode_failures,
        },
        "props": {
            "histogram_casefolded": dict(prop_hist.most_common()),
            "histogram_raw": dict(prop_raw_hist.most_common()),
            "distinct_raw": len(prop_raw_hist),
            "distinct_casefolded": len(prop_hist),
            "attribute_keys": {p: dict(c.most_common()) for p, c in attr_key_hist.items()},
        },
        "top_level_keys": dict(top_key_hist.most_common()),
        "reconciliation": recon,
        "dropped_files": zero_turn_files,
    }

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.L0_MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    b = manifest["baseline"]
    print("=== L0 基线 ===")
    print(f"剧情文件数           : {b['story_files']}  {dict(by_cat)}")
    print(f"story 目录字节       : {b['story_bytes']:,}")
    print(f"对话轮次             : {b['dialogue_turns']:,}")
    print(f"旁白轮次             : {b['narration_turns']:,}")
    print(f"掩码说话人轮次       : {b['masked_speaker_turns']:,}")
    print(f"标题卡轮次           : {b['title_card_turns']:,}")
    print(f"解析正文字数         : {b['parsed_content_chars']:,}  (wordcount 索引 {b['wordcount_index_chars']:,})")
    print(f"distinct 说话人      : {b['distinct_speakers']:,}")
    ratio = (b["dialogue_turns"] / b["narration_turns"]) if b["narration_turns"] else float("inf")
    print(f"对话/旁白比          : {ratio:.2f}   (计划书门线区间 1.5-4，本语料按签字五另加探针)")
    nar_share = b["narration_turns"] / max(1, b["narration_turns"] + b["dialogue_turns"])
    print(f"  探针 narration>0   : {b['narration_turns'] > 0}   narration 占比 {nar_share:.3f} (<0.30 要求)")
    print(f"污染节点             : {b['pollution_nodes']}")
    print(f"prop 种类(原样/折叠) : {manifest['props']['distinct_raw']} / {manifest['props']['distinct_casefolded']}")
    print("prop 直方图 top 30   : " + ", ".join(f"{k}:{v}" for k, v in list(manifest['props']['histogram_casefolded'].items())[:30]))
    print(f"零轮次文件           : {len(zero_turn_files)}  (每个必须查明原因)")
    print(f"BOM 文件             : {len(bom_files)}   解码失败: {len(decode_failures)}   CRLF 文件: {len(crlf_files)}")
    for name, r in recon.items():
        print(
            f"对账 {name:18s}: 索引键 {r['index_keys']:5d} 文件 {r['files']}  "
            f"有索引无文件 {len(r['index_without_file']):5d}  有文件无索引 {len(r['file_without_index']):5d}"
        )
    print(f"manifest -> {config.L0_MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
