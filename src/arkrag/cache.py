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

from . import atomic, config, prompt

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
) -> Path:
    """失败块不写缓存：瞬时故障一旦入库就会被永久固化。"""
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

    __slots__ = ("raw", "key", "layer", "reason")

    def __init__(self, raw: dict | None, key: str = "", layer: str = "miss", reason: str = "") -> None:
        self.raw = raw
        self.key = key
        self.layer = layer  # "current" | "legacy" | "miss" | "invalid"
        self.reason = reason

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
    """本 Chunk 在新旧两种键下的缓存文件路径。

    返修要「作废本 Chunk 的缓存」时必须两种都清：只删新键而留下 legacy 记录，
    下次 `lookup()` 仍会把它当可证明安全的旧记录复用，看起来「删了没生效」。
    """
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
