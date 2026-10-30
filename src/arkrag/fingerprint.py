"""指纹：语料指纹逐轮次累进，图指纹按计划书五要素顺序（归一化版本必进）。"""
from __future__ import annotations

import hashlib
from typing import Iterable


def corpus_fingerprint(scenes: Iterable[dict]) -> str:
    h = hashlib.sha256()
    for scene in sorted(scenes, key=lambda s: s["doc_id"]):
        h.update(scene["doc_id"].encode("utf-8"))
        for turn in scene["turns"]:
            h.update(f"{turn['kind']}|{turn['speaker']}|{turn['text']}\x00".encode("utf-8"))
    return h.hexdigest()[:16]


def graph_fingerprint(
    schema_version: str,
    corpus_fp: str,
    model: str,
    prompt_version: str,
    chunk_chars: int,
    normalizer_version: str,
    entity_rule_version: str = "",
) -> str:
    parts = [
        schema_version,
        corpus_fp,
        model,
        prompt_version,
        str(chunk_chars),
        normalizer_version,
    ]
    # 实体合并规则（ev 版本）单独进指纹：改「谁能并进谁」既不动词表也不动提示词，
    # 却会让整张图的结构变掉。不把它写进指纹，同一个指纹就能对应两张不同的图，
    # 而指纹的全部用处就是「这个指纹只对应这一张图」。
    if entity_rule_version:
        parts.append(entity_rule_version)
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def artifact_sha16(path) -> str:
    from pathlib import Path

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
