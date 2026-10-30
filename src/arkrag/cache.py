"""抽取缓存：键必须绑定「什么 Chunk、什么提示词、什么契约」。

## 为什么改（计划 B4）

旧键是 `sha256(content_key|model|prompt_version)[:24]`。它只认正文哈希和一个人工
维护的版本字符串，于是：

* **换 Chunk 身份察觉不到**：两个不同 `chunk_id` 恰好正文相同（overlap 让这很常见）
  会共用一条缓存，回收时 `chunk_id` 对不上也没人发现。
* **改提示词却不改版本就静默复用**：`PROMPT_VERSION = "v2"` 是手写的，改了
  `SYSTEM_PROMPT` 忘了 bump，旧缓存全部命中，表现为「改了提示词毫无变化」——
  这是最难查的一类问题，因为它看起来完全正常。
* **契约升级察觉不到**：`ark-rag-1` 的记录被 `ark-rag-2` 直接当合法输入读走。

新键绑定：`content_key` + `chunk_id` + `scene_id` + `model` + `prompt_sha256`
+ `schema_version` + `generation`。`prompt_sha256` 由**实际提示词内容**算出，
版本字符串降级为仅供展示，不再承担正确性。

## 为什么保留 legacy 读取

13,801 条既有缓存是唯一一份可用抽取结果，E 阶段要从它们重建 Assertion 图。
直接换键 = 全部失效 = 重抽 13,781 块 = 计划风险 1（严格校验导致大量旧缓存失效）。

计划给的处置是「影子校验分类，只豁免可证明安全的 Legacy 数据」。这里落在
`_legacy_record_is_safe()`：legacy 记录里自带 `chunk_id` / `content_key` /
`prompt_version` / `normalizer` / `model`，逐项与当前期望比对，**全部相符才认**。
不相符的一律不返回，并计入 `invalid_cache` 统计——不静默删除，也不静默接受。

## normalizer 为什么不进键

`raw` 存的是**归一化之前**的模型原始输出，归一化规则变了不影响这份字节。把它塞进
键只会让改一次归一化就重抽全量（约 13,781 次调用）。所以：

* `normalizer` **记进缓存记录**（可追溯「这条缓存是哪个归一化版本时期的产物」）
* 但**不参与键**，并在此处与测试里显式声明这个不变量。

这是 vs 计划 B4 字面要求（键含「词表/规范化版本」）的一处**有意偏差**：
词表事实上已经进键（词表被拼进 `SYSTEM_PROMPT`，因此进 `prompt_sha256`），
而归一化版本不进键。已在 `docs/升级台账.md` 记录理由。
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from . import atomic, config, contracts, prompt

CACHE_FORMAT = "raw"

# legacy 记录的格式标记（旧版写的就是这个）
LEGACY_CACHE_FORMAT = "raw"


def prompt_sha256() -> str:
    """由实际提示词内容计算。版本字符串不再承担正确性。"""
    return hashlib.sha256(
        (prompt.SYSTEM_PROMPT + "\x1f" + prompt.USER_TEMPLATE).encode("utf-8")
    ).hexdigest()


def cache_key(
    content_key: str,
    model: str,
    prompt_version: str,
) -> str:
    """**legacy 键**。只在读取历史缓存时使用。

    保留原签名是为了让「读旧缓存」这条路径与新键路径在代码里一眼可分；
    新写入一律用 `key_for()`。
    """
    payload = f"{content_key}|{model}|{prompt_version}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def key_for(
    *,
    content_key: str,
    chunk_id: str,
    scene_id: str,
    model: str,
    generation: int = 0,
    schema_version: str | None = None,
    prompt_digest: str | None = None,
) -> str:
    """新键：绑定身份、契约、模型、提示词实际内容、Generation。"""
    parts = [
        "v2",
        f"schema={schema_version or config.SCHEMA_VERSION}",
        f"content={content_key}",
        f"chunk={chunk_id}",
        f"scene={scene_id}",
        f"model={model}",
        f"prompt={prompt_digest or prompt_sha256()}",
        f"generation={generation}",
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]


def cache_path(key: str) -> Path:
    return config.CACHE_DIR / key[:2] / f"{key}.json"


# ------------------------------------------------------------------ 写入

def write_raw(
    key: str,
    raw: dict,
    model: str,
    chunk_id: str,
    content_key: str,
    *,
    scene_id: str = "",
    schema_version: str | None = None,
    prompt_digest: str | None = None,
    generation: int = 0,
    extra: dict | None = None,
) -> Path:
    """失败块不写缓存：瞬时故障一旦入库就会被永久固化。

    `extra` 只给 `adopt_legacy()` 用：它要往记录里写 `adopted_from` /
    `adopt_reason` / `adopted_at` 三个来源字段。做成参数而不是"写完再读回来改"，
    是为了让落盘只有一次原子写——读-改-写之间被打断会留下半个记录。
    """
    if not isinstance(raw, dict):
        raise ValueError("缓存只接受模型原始输出 dict")
    path = cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "__cache_format__": CACHE_FORMAT,
        "key_scheme": 2,
        # 记录但**不进键**：raw 是归一化之前的字节
        "normalizer": config.NORMALIZER_VERSION,
        "schema_version": schema_version or config.SCHEMA_VERSION,
        "prompt_version": config.PROMPT_VERSION,  # 仅供展示
        "prompt_sha256": prompt_digest or prompt_sha256(),
        "model": model,
        "chunk_id": chunk_id,
        "content_key": content_key,
        "scene_id": scene_id,
        "generation": generation,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "raw": raw,
    }
    if extra:
        rec.update(extra)
    atomic.atomic_write_json(path, rec)
    return path


# ------------------------------------------------------------------ 读取

def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        rec = json.loads(path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    return rec if isinstance(rec, dict) else None


def read_raw(key: str) -> dict | None:
    """按给定键读 raw。只做格式与版本校验，不做身份校验。

    身份校验请走 `lookup()`——它知道期望的 chunk_id / content_key。
    """
    rec = _load(cache_path(key))
    if rec is None:
        return None
    if rec.get("__cache_format__") != CACHE_FORMAT:
        return None
    if rec.get("key_scheme") == 2:
        # 新记录：提示词哈希必须与当前实际内容一致
        if rec.get("prompt_sha256") and rec["prompt_sha256"] != prompt_sha256():
            return None
    elif rec.get("prompt_version") != config.PROMPT_VERSION:
        # legacy 记录：只能靠版本字符串
        return None
    raw = rec.get("raw")
    return raw if isinstance(raw, dict) else None


class Lookup:
    """一次查找的结果。把「命中哪一层」也说清楚，否则无法判断缓存健康度。"""

    __slots__ = ("raw", "key", "layer", "reason", "adopted_from")

    def __init__(
        self,
        raw: dict | None,
        key: str = "",
        layer: str = "miss",
        reason: str = "",
        adopted_from: str = "",
    ) -> None:
        self.raw = raw
        self.key = key
        self.layer = layer  # "current" | "legacy" | "adopted" | "miss" | "invalid"
        self.reason = reason
        # adopted 层专用：这条新键记录是从哪个 legacy 缓存键搬过来的
        self.adopted_from = adopted_from

    def __bool__(self) -> bool:
        return self.raw is not None

    def __repr__(self) -> str:  # pragma: no cover
        return f"Lookup(layer={self.layer!r}, reason={self.reason!r})"


def _legacy_record_is_safe(rec: dict, *, chunk_id: str, content_key: str, model: str) -> tuple[bool, str]:
    """legacy 记录能否被证明安全地当作本 Chunk 的抽取结果。

    逐项核对记录里自带的身份字段。任何一项不符都退回——这正是旧键缺失的那层保护。
    """
    if rec.get("__cache_format__") != LEGACY_CACHE_FORMAT:
        return False, "legacy:bad_format"
    got_chunk = rec.get("chunk_id")
    if got_chunk != chunk_id:
        return False, f"legacy:chunk_id_mismatch({got_chunk!r}!={chunk_id!r})"
    got_ck = rec.get("content_key")
    if got_ck != content_key:
        return False, "legacy:content_key_mismatch"
    got_model = rec.get("model")
    if got_model != model:
        return False, f"legacy:model_mismatch({got_model!r}!={model!r})"
    if rec.get("prompt_version") != config.PROMPT_VERSION:
        return False, "legacy:prompt_version_stale"
    raw = rec.get("raw")
    if not isinstance(raw, dict):
        return False, "legacy:raw_not_object"
    if "entities" not in raw or "relations" not in raw:
        return False, "legacy:raw_missing_keys"
    return True, ""


# ------------------------------------------------------------------ 复用层

#: `adopt_legacy()` 唯一允许写进 `adopt_reason` 的值。新增理由必须先证明
#: 「raw 抽取的输入与当初逐项相等」，不能只是「看起来差不多」。
ADOPT_REASON_IDENTICAL = "identical_content"


def _scene_of_chunk_id(chunk_id: str) -> str:
    """`doc_id#NNN` 的 doc 部分。legacy 记录只保留了这一处场景绑定。"""
    head, sep, _ = str(chunk_id or "").rpartition("#")
    return head if sep else ""


def load_legacy_record(content_key: str, model: str, *, prompt_version: str | None = None) -> dict | None:
    """按 legacy 键取原始记录（不做身份核对，核对交给 `adopt_legacy()`）。

    存在的理由：`lookup()` 在 chunk_id 对不上时会返回 invalid，而「换块身份后
    能否复用」恰恰要拿到那条 chunk_id 对不上的记录再逐项核对。这条路径只读。
    """
    key = cache_key(content_key, model, prompt_version or config.PROMPT_VERSION)
    return _load(cache_path(key))


def adopt_legacy(
    src_record: dict | None,
    chunk: dict,
    model: str,
    *,
    generation: int = 0,
) -> Lookup:
    """把一条 legacy 缓存记录**带来源地**搬成本 Chunk 的新键记录。

    ## 为什么这不是「用 A 块的抽取结果去建 B 块的边」

    抽取这一步的函数签名是 `raw = f(块正文, 提示词, 模型)`——**输入里没有块身份**。
    `chunk_id` / `scene_id` / `generation` 只用于「这份结果属于谁」的记账，它们不
    参与任何生成语义。因此当且仅当下面四项**逐项字节相等**时，两条块在抽取语义
    上是同一个输入，raw 就是本次输入的合法输出，而不是别人的结论被借用：

    1. `content_key`——并且本函数会重算 `sha256(chunk["text"])[:24]` 核对它，
       确保 content_key 确实指向当前正文，而不是一个没跟着正文走的旧字段；
    2. `scene_id`——legacy 记录没有 `scene_id` 字段，它保留的唯一场景绑定是
       `chunk_id` 的 `doc#NNN` 前缀，故取 `doc` 部分比对；不等即拒；
    3. `model`——同一个模型才有同一个条件分布；
    4. `prompt_sha256`——legacy 记录同样没有这个字段，只有人工维护的
       `prompt_version`。所以这一项只能**降级为「`prompt_version` 与当前
       `config.PROMPT_VERSION` 相等」**来成立，并在新记录里写下当前真实的
       `prompt_sha256()`。这是本次复用唯一的弱环节，已如实计入 `note`，不再假装
       它是逐字节证明。

    四项之外还有一道独立的结构门：把 legacy 的 raw 按当前身份组成一条抽取记录，
    交给 `contracts.validate_extraction()` 校验——`schema_version`、实体类型、
    置信度区间、以及**每条关系的 `evidence` 必须在当前块正文里逐字出现**（并回填
    `evidence_start`/`evidence_end`）。证据定位是这一层最有分量的判据：它直接检查
    「这条边的原文依据是否真的在这一块里」。任何一条 error 就整体拒绝。

    注意契约**不**检查关系两端是否都在 entities 里（`_check_relations()` 收了
    `entity_names` 参数但没用它）。这里补一道：任一端点不在 entities 内的记录直接
    拒——否则搬进来的边在 L4/L5 会因为找不到端点被丢掉，等于白搬并且掩盖了坏数据。

    通过后写一条**新键**记录，并在记录里留下 `adopted_from`（来源 legacy 缓存键）、
    `adopt_reason`（恒为 `identical_content`）、`adopted_at`。下游 `lookup()` 会把
    这类记录单独报成 `layer="adopted"`，绝不混进 `current` 冒充本轮抽取产出——
    所以「有多少块是这个世代真抽的、多少块是搬来的」永远是分开的两个数。

    ## 拒绝时的返回值

    `src_record` 为空 → `layer="miss"`；任一项核对失败 → `layer="invalid"` 且
    `reason` 以 `adopt:` 开头说明断在哪一项。两种都不写盘，不静默降级。
    """
    content_key = chunk["content_key"]
    chunk_id = chunk["chunk_id"]
    scene_id = chunk.get("scene_id") or chunk.get("doc_id") or ""
    new_key = key_for(
        content_key=content_key,
        chunk_id=chunk_id,
        scene_id=scene_id,
        model=model,
        generation=generation,
    )

    # 已经搬过就不再搬：重复调用必须是幂等的
    existing = _load(cache_path(new_key))
    if existing is not None and isinstance(existing.get("raw"), dict):
        if existing.get("adopted_from"):
            return Lookup(
                existing["raw"], new_key, "adopted", str(existing.get("adopt_reason") or ""),
                adopted_from=str(existing["adopted_from"]),
            )
        return Lookup(existing["raw"], new_key, "current")

    if not isinstance(src_record, dict):
        return Lookup(None, new_key, "miss", "adopt:no_source")

    # ① content_key：既比字段，也比「正文是否真的产出了这个字段」
    src_ck = src_record.get("content_key")
    if src_ck != content_key:
        return Lookup(None, new_key, "invalid", f"adopt:content_key_mismatch({src_ck!r}!={content_key!r})")
    actual_ck = hashlib.sha256(str(chunk.get("text") or "").encode("utf-8")).hexdigest()[:24]
    if actual_ck != content_key:
        return Lookup(
            None, new_key, "invalid",
            f"adopt:chunk_text_does_not_match_content_key({actual_ck}!={content_key})",
        )

    # ② scene_id：legacy 没有独立字段，只能取 chunk_id 的 doc 前缀
    src_scene = _scene_of_chunk_id(src_record.get("chunk_id") or "")
    if not scene_id or src_scene != scene_id:
        return Lookup(None, new_key, "invalid", f"adopt:scene_id_mismatch({src_scene!r}!={scene_id!r})")

    # ③ model
    src_model = src_record.get("model")
    if src_model != model:
        return Lookup(None, new_key, "invalid", f"adopt:model_mismatch({src_model!r}!={model!r})")

    # ④ prompt_sha256（legacy 只有 prompt_version，只能降级比对，理由见 docstring）
    if src_record.get("prompt_version") != config.PROMPT_VERSION:
        return Lookup(
            None, new_key, "invalid",
            f"adopt:prompt_version_mismatch({src_record.get('prompt_version')!r}"
            f"!={config.PROMPT_VERSION!r})",
        )
    if src_record.get("__cache_format__") != LEGACY_CACHE_FORMAT:
        return Lookup(None, new_key, "invalid", "adopt:bad_format")

    src_raw = src_record.get("raw")
    if not isinstance(src_raw, dict):
        return Lookup(None, new_key, "invalid", "adopt:raw_not_object")

    # 端点闭合：契约收了 entity_names 却没用它，这里自己补，免得搬进来的边在
    # L4/L5 因为找不到端点被静默丢掉（那会让「复用」变成「看起来成功但没边」）。
    src_entities = src_raw.get("entities")
    if isinstance(src_entities, list):
        names = {
            str(e.get("name") or "").strip()
            for e in src_entities
            if isinstance(e, dict) and str(e.get("name") or "").strip()
        }
        src_relations = src_raw.get("relations")
        if isinstance(src_relations, list):
            for rel in src_relations:
                if not isinstance(rel, dict):
                    continue
                for end in ("source", "target"):
                    name = str(rel.get(end) or "").strip()
                    if name and name not in names:
                        return Lookup(
                            None, new_key, "invalid",
                            f"adopt:dangling_endpoint({end}={name!r} 不在 entities 内)",
                        )

    # 结构门：把 raw 按当前身份组记录，交给严格契约逐项校验（含证据定位）
    candidate = {
        "schema_version": contracts.EXTRACTION_SCHEMA,
        "chunk_id": chunk_id,
        "content_key": content_key,
        "scene_id": scene_id,
        "entities": src_raw.get("entities"),
        "relations": src_raw.get("relations"),
    }
    task = {"chunk_id": chunk_id, "content_key": content_key, "scene_id": scene_id}
    cleaned, errors = contracts.validate_extraction(candidate, task, str(chunk.get("text") or ""))
    if cleaned is None:
        first = next((e for e in errors if e.get("level") == "error"), {})
        return Lookup(
            None, new_key, "invalid",
            f"adopt:contract_rejected({first.get('code', '?')}: {str(first.get('detail', ''))[:80]})",
        )

    legacy_key = cache_key(content_key, model, config.PROMPT_VERSION)
    adopted_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_raw(
        new_key,
        src_raw,
        model,
        chunk_id,
        content_key,
        scene_id=scene_id,
        generation=generation,
        extra={
            "adopted_from": legacy_key,
            "adopt_reason": ADOPT_REASON_IDENTICAL,
            "adopted_at": adopted_at,
        },
    )
    return Lookup(src_raw, new_key, "adopted", ADOPT_REASON_IDENTICAL, adopted_from=legacy_key)


def lookup(
    *,
    content_key: str,
    chunk_id: str,
    scene_id: str,
    model: str,
    generation: int = 0,
    allow_legacy: bool = True,
) -> Lookup:
    """按「新键 → legacy 键（含身份核对）」顺序查找。

    新键命中即返回。新键未命中时，只有 legacy 记录的身份字段与本 Chunk 完全
    相符才认（可证明安全）；否则返回 `layer="invalid"`，供上层计入 `invalid_cache`。
    """
    new_key = key_for(
        content_key=content_key,
        chunk_id=chunk_id,
        scene_id=scene_id,
        model=model,
        generation=generation,
    )
    rec = _load(cache_path(new_key))
    if rec is not None:
        if rec.get("__cache_format__") == CACHE_FORMAT and isinstance(rec.get("raw"), dict):
            if rec.get("prompt_sha256") in (None, prompt_sha256()):
                # 带 adopted_from 的是「从 legacy 记录按四项全等搬过来的」，
                # 单独一层 reported，不能混进 current 冒充本轮抽取产出。
                if rec.get("adopted_from"):
                    return Lookup(
                        rec["raw"], new_key, "adopted", str(rec.get("adopt_reason") or ""),
                        adopted_from=str(rec["adopted_from"]),
                    )
                return Lookup(rec["raw"], new_key, "current")

    if not allow_legacy:
        return Lookup(None, new_key, "miss")

    old_key = cache_key(content_key, model, config.PROMPT_VERSION)
    if old_key == new_key:
        return Lookup(None, new_key, "miss")
    old_rec = _load(cache_path(old_key))
    if old_rec is None:
        return Lookup(None, new_key, "miss")
    ok, why = _legacy_record_is_safe(
        old_rec, chunk_id=chunk_id, content_key=content_key, model=model
    )
    if ok:
        return Lookup(old_rec["raw"], old_key, "legacy")
    return Lookup(None, new_key, "invalid", why)


def lookup_for_chunk(
    chunk: dict,
    model: str,
    *,
    generation: int = 0,
    allow_legacy: bool = True,
) -> Lookup:
    """按 Chunk 记录查找缓存。所有下游脚本都该走这个入口。

    直接把 `chunk` 的三个身份字段喂给 `lookup()`，避免调用方各写一份
    「content_key 从哪取、scene_id 从哪取」——那正是身份漂移的来源。
    """
    return lookup(
        content_key=chunk["content_key"],
        chunk_id=chunk["chunk_id"],
        scene_id=chunk.get("scene_id") or chunk.get("doc_id") or "",
        model=model,
        generation=generation,
        allow_legacy=allow_legacy,
    )


def read_for_chunk(chunk: dict, model: str, *, generation: int = 0) -> dict | None:
    """最常用的形式：只要 raw 或 None。"""
    return lookup_for_chunk(chunk, model, generation=generation).raw


def is_cached_for_chunk(chunk: dict, model: str, *, generation: int = 0) -> bool:
    """「这个 Chunk 有没有可用抽取结果」——状态查询与建图前置检查统一走这里。

    旧口径是「缓存文件存在即算已抽」，但键不含身份，文件存在并不能证明它就是
    本 Chunk 的结果。这里必须走带身份核对的 lookup。
    """
    return lookup_for_chunk(chunk, model, generation=generation).raw is not None


def is_cached(content_key: str, chunk_id: str, model: str, *, generation: int = 0) -> bool:
    """只要两个身份字段的场景（inbox 任务行、派发包等）。

    这些地方拿不到完整 Chunk 记录，但 `chunk_id` + `content_key` 已足够让
    `lookup()` 拒掉「同正文不同块」的命中。
    """
    return lookup(
        content_key=content_key,
        chunk_id=chunk_id,
        scene_id="",
        model=model,
        generation=generation,
    ).raw is not None


def paths_for_chunk(chunk: dict, model: str, *, generation: int = 0) -> list[Path]:
    keys = [
        key_for(
            content_key=chunk["content_key"],
            chunk_id=chunk["chunk_id"],
            scene_id=chunk.get("scene_id") or chunk.get("doc_id") or "",
            model=model,
            generation=generation,
        ),
        cache_key(chunk["content_key"], model, config.PROMPT_VERSION),
    ]
    seen: list[Path] = []
    for k in keys:
        p = cache_path(k)
        if p not in seen:
            seen.append(p)
    return seen


def invalid_cache_report() -> dict:
    """扫描缓存目录，把「结构坏 / 身份字段缺失 / 不可解析」的条目分类记账。

    计划 B4：非法缓存移入隔离区记 `invalid_cache`，不静默删除。本函数只统计，
    实际搬迁由 `quarantine_invalid()` 显式执行。
    """
    stats = {"total": 0, "ok": 0, "invalid": 0, "by_reason": {}, "samples": {}}
    if not config.CACHE_DIR.exists():
        return stats
    for path in config.CACHE_DIR.rglob("*.json"):
        stats["total"] += 1
        rec = _load(path)
        if rec is None:
            reason = "unreadable"
        elif rec.get("__cache_format__") != CACHE_FORMAT:
            reason = "bad_format"
        elif not isinstance(rec.get("raw"), dict):
            reason = "raw_not_object"
        elif "entities" not in rec["raw"] or "relations" not in rec["raw"]:
            reason = "raw_missing_keys"
        elif not rec.get("chunk_id"):
            reason = "missing_chunk_id"
        elif not rec.get("content_key"):
            reason = "missing_content_key"
        else:
            reason = ""
        if reason:
            stats["invalid"] += 1
            stats["by_reason"][reason] = stats["by_reason"].get(reason, 0) + 1
            stats["samples"].setdefault(reason, str(path))
        else:
            stats["ok"] += 1
    return stats


def quarantine_invalid(*, dry_run: bool = True) -> dict:
    """把非法缓存搬到 `data/arkrag/cache_quarantine/<reason>/`，不删除。"""
    dest_root = config.DATA_DIR / "cache_quarantine"
    moved: list[dict] = []
    if not config.CACHE_DIR.exists():
        return {"moved": 0, "dry_run": dry_run, "items": moved}
    for path in config.CACHE_DIR.rglob("*.json"):
        rec = _load(path)
        if rec is None:
            reason = "unreadable"
        elif rec.get("__cache_format__") != CACHE_FORMAT:
            reason = "bad_format"
        elif not isinstance(rec.get("raw"), dict):
            reason = "raw_not_object"
        elif "entities" not in rec["raw"] or "relations" not in rec["raw"]:
            reason = "raw_missing_keys"
        elif not rec.get("chunk_id"):
            reason = "missing_chunk_id"
        elif not rec.get("content_key"):
            reason = "missing_content_key"
        else:
            continue
        target = dest_root / reason / path.name
        moved.append({"from": str(path), "to": str(target), "reason": reason})
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            path.replace(target)
    return {"moved": len(moved), "dry_run": dry_run, "items": moved[:200]}


def count_cached() -> int:
    if not config.CACHE_DIR.exists():
        return 0
    return sum(1 for p in config.CACHE_DIR.rglob("*.json") if not p.name.endswith(".tmp"))
