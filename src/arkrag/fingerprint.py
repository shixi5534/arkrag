"""指纹：分层身份，每一层只描述自己那一层的输入。

计划 D3 要的是「分层指纹」，理由是一层的身份不能靠别层的版本来代表：

    source_fp      原始语料（ZIP / 解压树）到底是不是同一份
    scene_fp       解析出来的场景与轮次（含 doc_id/title/chapter/category/node_id/prop）
    chunk_fp       分块参数 + T1 范围 + 每块的 chunk_id/content_key
    extraction_fp  实际拿到的 raw 抽取结果（在 release.extraction_manifest_hash）
    graph_build_fp 建图配置：模型、**实际提示词内容**、词表、归一化与合并规则、分块参数
    graph_id       图产物的内容摘要（不是配置摘要）
    index_build_fp 建索引配置：字段、分词器、BM25 参数
    index_id       索引产物的内容摘要

两个必须分清的概念，混起来会导致「产物变了但身份没变」：

  * **build_fp 表配置**。改配置不改数据，它就该变；它变了不必然意味着图变了。
  * **artifact_id 表内容**。配置一模一样、重建一次，artifact_id 必须相同；
    如果它变了，说明有某个没进配置的东西也在影响产物——那正是要找的漏洞。

旧实现的两个具体缺陷（D3 修的就是这两条）：
  1. 图指纹吃的是 `PROMPT_VERSION` 这个**展示用字符串**。提示词改了忘改版本号，
     新图就顶着旧指纹发布，回滚锚点全部指错。现在吃 `prompt_sha256()` 的实际内容哈希。
  2. 词表（关系同义词 211 条、实体类型同义词 71 条、逆谓词、对称关系、互斥对、
     成员后缀……）完全不在指纹里。加一条同义词既不动代码也不动提示词，
     却能让整张图的结构变掉。现在吃 `vocab_digest()`。
"""
from __future__ import annotations

import hashlib
import json
from typing import Iterable


def _sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def corpus_fingerprint(scenes: Iterable[dict]) -> str:
    """语料指纹：逐场景逐轮次累进，与遍历顺序无关。"""
    h = hashlib.sha256()
    for scene in sorted(scenes, key=lambda s: s["doc_id"]):
        h.update(scene["doc_id"].encode("utf-8"))
        for turn in scene["turns"]:
            h.update(f"{turn['kind']}|{turn['speaker']}|{turn['text']}\x00".encode("utf-8"))
    return h.hexdigest()[:16]


def scene_fingerprint(scenes: Iterable[dict]) -> str:
    """场景层身份：比 corpus_fingerprint 多覆盖元数据。

    corpus_fingerprint 只吃 doc_id + 轮次正文，够用来判断"语料变没变"；
    但 title/chapter/category 会进索引字段、node_id 会进证据定位、prop 决定轮次类型，
    这些改了而正文没改时，corpus_fp 不变——所以场景层要单独有一个指纹。
    """
    h = hashlib.sha256()
    for scene in sorted(scenes, key=lambda s: s["doc_id"]):
        h.update(
            "\x1f".join(
                str(scene.get(k, ""))
                for k in ("doc_id", "title", "chapter", "category")
            ).encode("utf-8")
        )
        h.update(b"\x1e")
        for turn in scene["turns"]:
            h.update(
                f"{turn.get('ordinal', '')}|{turn.get('node_id', '')}|{turn.get('prop', '')}"
                f"|{turn['kind']}|{turn['speaker']}|{turn['text']}\x00".encode("utf-8")
            )
    return h.hexdigest()[:16]


def chunk_fingerprint(chunks: Iterable[dict], *, params: dict | None = None) -> str:
    """分块层身份：分块参数 + T1 范围 + 每块身份。

    只比块数是不够的——调小目标块长会让块数变多、调大重叠会让块数不变而内容全变。
    所以既吃参数，也吃每块的 chunk_id/content_key。
    """
    h = hashlib.sha256()
    if params:
        h.update(json.dumps(params, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        h.update(b"\x1e")
    for chunk in sorted(chunks, key=lambda c: c["chunk_id"]):
        h.update(f"{chunk['chunk_id']}\x00{chunk.get('content_key', '')}\x00".encode("utf-8"))
    return h.hexdigest()[:16]


def vocab_digest() -> str:
    """词表内容摘要。

    直接把 vocab 模块里那些会影响产物的常量序列化后取哈希，
    这样"改词表忘改版本号"就不再可能——词表本身就是指纹的一部分。
    """
    from . import vocab

    payload = {
        "ENTITY_RULE_VERSION": vocab.ENTITY_RULE_VERSION,
        "RELATION_VERSION": vocab.RELATION_VERSION,
        "ENTITY_TYPES": list(vocab.ENTITY_TYPES),
        "CANONICAL_RELATIONS": list(vocab.CANONICAL_RELATIONS),
        "RELATION_SYNONYMS": dict(sorted(vocab.RELATION_SYNONYMS.items())),
        "TYPE_SYNONYMS": dict(sorted(vocab.TYPE_SYNONYMS.items())),
        "INVERSE_PAIRS": [list(p) for p in vocab.INVERSE_PAIRS],
        "SYMMETRIC_RELATIONS": list(vocab.SYMMETRIC_RELATIONS),
        "MUTUALLY_EXCLUSIVE_PAIRS": [list(p) for p in vocab.MUTUALLY_EXCLUSIVE_PAIRS],
        "JUNK_RELATIONS": list(vocab.JUNK_RELATIONS),
        "GENERIC_PERSON_NOUNS": list(vocab.GENERIC_PERSON_NOUNS),
        "MEMBER_SUFFIXES": list(vocab.MEMBER_SUFFIXES),
        "ORG_SUFFIXES": list(vocab.ORG_SUFFIXES),
        "PRONOUN_SET": sorted(vocab.PRONOUN_SET),
        "PARTICLES": list(vocab.PARTICLES),
        "NEGATION_PREFIX": list(vocab.NEGATION_PREFIX),
        "DIRECTION_WATCH": list(vocab.DIRECTION_WATCH),
        "ACTION_VERBS": list(vocab.ACTION_VERBS),
        "MAX_NAME_CHARS": vocab.MAX_NAME_CHARS,
        "MAX_RELATION_CHARS": vocab.MAX_RELATION_CHARS,
    }
    return _sha16(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def prompt_digest() -> str:
    """提示词内容摘要（实际内容，不是展示用版本号）。"""
    from . import prompt

    return hashlib.sha256(
        (prompt.SYSTEM_PROMPT + "\x1f" + prompt.USER_TEMPLATE).encode("utf-8")
    ).hexdigest()


def graph_fingerprint(
    schema_version: str,
    corpus_fp: str,
    model: str,
    prompt_version: str,
    chunk_chars: int,
    normalizer_version: str,
    entity_rule_version: str = "",
    *,
    prompt_sha256: str = "",
    vocab_sha: str = "",
    chunk_params: dict | None = None,
    relation_version: str = "",
) -> str:
    """建图配置指纹（graph_build_fp）。

    保留 `prompt_version` 位置参数只为兼容旧调用；**实际参与哈希的是 prompt_sha256**。
    两者都在时用 sha（内容可信），只有旧调用没给 sha 时才退回版本字符串。
    """
    parts = [
        schema_version,
        corpus_fp,
        model,
        prompt_sha256 or prompt_version,
        str(chunk_chars),
        normalizer_version,
    ]
    # 实体合并规则（ev 版本）单独进指纹：改「谁能并进谁」既不动词表也不动提示词，
    # 却会让整张图的结构变掉。不把它写进指纹，同一个指纹就能对应两张不同的图，
    # 而指纹的全部用处就是「这个指纹只对应这一张图」。
    if entity_rule_version:
        parts.append(entity_rule_version)
    if relation_version:
        parts.append(f"rv={relation_version}")
    if vocab_sha:
        parts.append(f"vocab={vocab_sha}")
    if chunk_params:
        parts.append("chunk=" + json.dumps(chunk_params, sort_keys=True))
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def source_fingerprint(source_contract: dict | None, tree_digest: str = "") -> tuple[str, str]:
    """来源层身份（D4）。返回 (fp, status)。

    status 取值：
      verified_zip     ZIP 在场且字节与合同一致
      verified_tree    没有 ZIP，但解压树文件清单与合同一致（确定性清单可比）
      source_unverified 无法验证——**不得冒充 Release**，下游默认拒绝

    「无法验证」不是错误，只是身份未知；把它静默当成"已验证"才是错误。
    """
    contract = source_contract or {}
    parts = [
        str(contract.get("repo", "")),
        str(contract.get("release_tag", "")),
        str(contract.get("asset", "")),
        str(contract.get("zip_bytes", "")),
        str(contract.get("zip_sha256", "")),
        str(contract.get("compatibility_contract", "")),
        tree_digest,
    ]
    fp = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]
    # 状态取值引用 source 的常量，不在这里另抄一份字面量。
    # 这个函数目前没有调用方（D4 的真实来源判定在 source.verify 里，它才是权威），
    # 但留着两份各自独立的状态判定就是隐患：改了一处忘了另一处，
    # 就会出现"两个模块对同一份语料给出不同来源状态"。
    from . import source as _source

    if tree_digest and not tree_digest.startswith("unverified"):
        status = _source.VERIFIED_TREE
    elif contract.get("zip_sha256_verified"):
        status = _source.VERIFIED_ZIP
    else:
        status = _source.UNVERIFIED
    return fp, status


def artifact_sha16(path) -> str:
    from pathlib import Path

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def artifact_sha256(path) -> str:
    from pathlib import Path

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
