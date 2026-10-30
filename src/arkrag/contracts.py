"""抽取数据契约（计划 B1）。

现有 `extract.validate_shape` 只有 4 类检查，导致 `entities=""`、`entities=False`、
缺 `chunk_id`、重复 ID、置信度越界、证据根本不在 Chunk 里这类记录都能一路进图，
事后只能靠人眼在 13,781 个 Chunk 里找。

本模块提供两个入口，都是**逐字段定位**的严格校验：

    validate_task(task)                  → (cleaned, errors)
    validate_extraction(rec, task, text) → (cleaned, errors)

约定：返回 `errors` 非空即代表拒绝。调用方**不允许**在 errors 非空时把记录写进
正式产物；只能整条进隔离区。`errors` 每项含 `code`、`field`、`detail`，供汇总
统计与人工定位。

为什么 `cleaned` 与 `errors` 并存：结构合法但含可删项的记录（例如同一实体重复
出现两次）应当「删掉重复项并记录」而不是整条拒收——但**任何删减都要在 errors
里留下 `warn` 级条目**，否则就是静默改数据。
"""
from __future__ import annotations

import json
from typing import Any

TASK_SCHEMA = "ark-rag-task-2"
EXTRACTION_SCHEMA = "ark-rag-extraction-2"

# 证据最长允许长度：与 prompt 第 6 条「≤40 字连续原文」一致。超出说明模型编了整段。
EVIDENCE_MAX_CHARS = 60

ENTITY_TYPES = ("角色", "组织", "地点", "事件", "物品", "概念")


class Code:
    """机器可读错误码。禁止在调用方硬编码字符串——码表是唯一事实源。"""

    NOT_OBJECT = "not_object"
    MISSING_FIELD = "missing_field"
    WRONG_TYPE = "wrong_type"
    EMPTY_COLLECTION = "empty_collection"
    UNKNOWN_SCHEMA = "unknown_schema"
    UNKNOWN_FIELD = "unknown_field"
    CHUNK_ID_MISMATCH = "chunk_id_mismatch"
    CONTENT_KEY_MISMATCH = "content_key_mismatch"
    SCENE_ID_MISMATCH = "scene_id_mismatch"
    DUPLICATE_ENTITY_ID = "duplicate_entity_id"
    DUPLICATE_ENTITY_NAME = "duplicate_entity_name"
    DUPLICATE_RELATION = "duplicate_relation"
    BAD_CONFIDENCE = "bad_confidence"
    BAD_ENTITY_TYPE = "bad_entity_type"
    EMPTY_ENTITY_NAME = "empty_entity_name"
    EVIDENCE_NOT_FOUND = "evidence_not_found"
    EVIDENCE_TOO_LONG = "evidence_too_long"
    EVIDENCE_EMPTY = "evidence_empty"
    RELATION_UNKNOWN_PREDICATE = "relation_unknown_predicate"
    RELATION_SELF_LOOP = "relation_self_loop"
    SELF_REFERENCE = "self_reference"


def _err(code: str, field: str, detail: str, *, level: str = "error") -> dict:
    return {"code": code, "field": field, "detail": detail, "level": level}


def _is_str(v: Any) -> bool:
    return isinstance(v, str)


def _require(obj: dict, field: str, errors: list[dict], *, allow_empty: bool = False) -> Any:
    if field not in obj:
        errors.append(_err(Code.MISSING_FIELD, field, f"缺少字段 {field}"))
        return None
    v = obj[field]
    if v is None:
        errors.append(_err(Code.MISSING_FIELD, field, f"字段 {field} 为 null"))
        return None
    if _is_str(v) and not allow_empty and not v.strip():
        errors.append(_err(Code.MISSING_FIELD, field, f"字段 {field} 为空字符串"))
        return None
    return v


# --------------------------------------------------------------------------- task

def validate_task(task: Any) -> tuple[dict | None, list[dict]]:
    """校验一条抽取任务。返回 (cleaned_task, errors)。

    拒收条件（计划 B1 逐条落在这里）：
    非对象、未知 schema_version、缺 chunk_id / content_key / scene_id、
    缺 source_fingerprint / prompt_sha256 / model、chunk_id 与 content_key 为空。
    """
    errors: list[dict] = []
    if not isinstance(task, dict):
        errors.append(_err(Code.NOT_OBJECT, "", f"任务必须是对象，实际 {type(task).__name__}"))
        return None, errors

    schema = task.get("schema_version")
    if schema != TASK_SCHEMA:
        errors.append(
            _err(Code.UNKNOWN_SCHEMA, "schema_version", f"期望 {TASK_SCHEMA}，实际 {schema!r}")
        )
        return None, errors

    cleaned: dict = {"schema_version": TASK_SCHEMA}
    for field in (
        "run_id",
        "chunk_id",
        "content_key",
        "scene_id",
        "doc_id",
        "source_fingerprint",
        "prompt_sha256",
        "model",
    ):
        v = _require(task, field, errors)
        if v is not None:
            cleaned[field] = v

    # generation 允许为 0/空（首次构建），但字段必须在
    if "generation" not in task:
        errors.append(_err(Code.MISSING_FIELD, "generation", "缺少 generation（首次构建填 0）"))
    else:
        gen = task["generation"]
        if not isinstance(gen, int) or isinstance(gen, bool) or gen < 0:
            errors.append(
                _err(Code.WRONG_TYPE, "generation", f"generation 必须是非负整数，实际 {gen!r}")
            )
        else:
            cleaned["generation"] = gen

    text = task.get("text")
    if not _is_str(text) or not text.strip():
        errors.append(_err(Code.MISSING_FIELD, "text", "任务必须带非空 text（证据定位要用）"))
    else:
        cleaned["text"] = text

    for optional in ("title", "chapter", "category", "speakers", "node_ids"):
        if optional in task:
            cleaned[optional] = task[optional]

    return (None if errors else cleaned), errors


# --------------------------------------------------------------------- extraction

def _check_entities(items: Any, errors: list[dict]) -> list[dict]:
    if not isinstance(items, list):
        # 这正是现有代码 `or []` 把 ""/False 洗成 [] 的地方——必须报错而不是兜底
        errors.append(
            _err(Code.WRONG_TYPE, "entities", f"entities 必须是数组，实际 {type(items).__name__}")
        )
        return []
    if not items:
        errors.append(_err(Code.EMPTY_COLLECTION, "entities", "entities 为空：只列实体不建边也算没抽"))
        return []

    out: list[dict] = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    for i, e in enumerate(items):
        path = f"entities[{i}]"
        if not isinstance(e, dict):
            errors.append(_err(Code.WRONG_TYPE, path, f"实体必须是对象，实际 {type(e).__name__}"))
            continue
        name = e.get("name")
        if not _is_str(name) or not name.strip():
            errors.append(_err(Code.EMPTY_ENTITY_NAME, path + ".name", f"实体名为空: {name!r}"))
            continue
        name = name.strip()

        etype = e.get("type")
        if etype is not None and etype not in ENTITY_TYPES:
            # 未知类型不拒收整条，但要记 warn——类型归一化阶段会处理
            errors.append(
                _err(
                    Code.BAD_ENTITY_TYPE,
                    path + ".type",
                    f"实体类型不在词表: {etype!r}（将由 L4 归一化处理）",
                    level="warn",
                )
            )

        eid = e.get("id")
        if eid is not None:
            if not _is_str(eid) or not eid.strip():
                errors.append(_err(Code.WRONG_TYPE, path + ".id", f"实体 id 非法: {eid!r}"))
                continue
            if eid in seen_ids:
                errors.append(_err(Code.DUPLICATE_ENTITY_ID, path + ".id", f"重复实体 id: {eid}"))
                continue
            seen_ids.add(eid)

        if name in seen_names:
            # 同名重复：保留一条，但要留痕
            errors.append(
                _err(
                    Code.DUPLICATE_ENTITY_NAME,
                    path + ".name",
                    f"实体名重复已去重: {name}",
                    level="warn",
                )
            )
            continue
        seen_names.add(name)

        item = {"name": name}
        if etype is not None:
            item["type"] = etype
        if eid is not None:
            item["id"] = eid
        if "desc" in e:
            item["desc"] = e["desc"]
        out.append(item)

    return out


def _check_relations(items: Any, entity_names: set[str], errors: list[dict]) -> list[dict]:
    if not isinstance(items, list):
        errors.append(
            _err(Code.WRONG_TYPE, "relations", f"relations 必须是数组，实际 {type(items).__name__}")
        )
        return []
    # relations 允许为空数组（纯列实体），但 entities 空已经报错了

    out: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for i, r in enumerate(items):
        path = f"relations[{i}]"
        if not isinstance(r, dict):
            errors.append(_err(Code.WRONG_TYPE, path, f"关系必须是对象，实际 {type(r).__name__}"))
            continue

        ok = True
        for field in ("source", "relation", "target", "evidence"):
            v = r.get(field)
            if not _is_str(v) or not v.strip():
                errors.append(_err(Code.MISSING_FIELD, f"{path}.{field}", f"关系字段为空: {field}={v!r}"))
                ok = False
        if not ok:
            continue

        src, rel, tgt, ev = (r["source"].strip(), r["relation"].strip(), r["target"].strip(), r["evidence"])
        if src == tgt:
            errors.append(
                _err(Code.RELATION_SELF_LOOP, path, f"自环关系被拒: {src} -{rel}-> {tgt}")
            )
            continue

        if len(ev) > EVIDENCE_MAX_CHARS:
            errors.append(
                _err(
                    Code.EVIDENCE_TOO_LONG,
                    path + ".evidence",
                    f"证据 {len(ev)} 字超过上限 {EVIDENCE_MAX_CHARS}，疑似整段抄录",
                )
            )
            continue

        conf = r.get("confidence")
        if conf is not None:
            if isinstance(conf, bool) or not isinstance(conf, (int, float)):
                errors.append(_err(Code.BAD_CONFIDENCE, path + ".confidence", f"置信度非数值: {conf!r}"))
                continue
            if not 0.0 <= float(conf) <= 1.0:
                errors.append(
                    _err(Code.BAD_CONFIDENCE, path + ".confidence", f"置信度越界: {conf!r} 不在 [0,1]")
                )
                continue

        key = (src, rel, tgt)
        if key in seen:
            errors.append(
                _err(Code.DUPLICATE_RELATION, path, f"重复关系已去重: {src} -{rel}-> {tgt}", level="warn")
            )
            continue
        seen.add(key)

        item = {"source": src, "relation": rel, "target": tgt, "evidence": ev}
        if conf is not None:
            item["confidence"] = float(conf)
        out.append(item)

    return out


def validate_extraction(
    rec: Any,
    task: dict,
    chunk_text: str,
    *,
    known_predicates: set[str] | None = None,
) -> tuple[dict | None, list[dict]]:
    """校验一条抽取产物。

    `task` 是可信的任务记录（已经过 validate_task），用于核对身份字段。
    `chunk_text` 是该 Chunk 的完整正文，用于核对证据可定位。

    计划 B1 要求拒收的项全部在这里：
        非对象 / entities 缺失或非数组或为空 / entities="" 或 False /
        缺 chunk_id / content_key 不匹配 / chunk_id 不匹配 / scene_id 不匹配 /
        重复 ID / 非法置信度 / 证据无法定位 / 未知版本。
    """
    errors: list[dict] = []
    if not isinstance(rec, dict):
        errors.append(_err(Code.NOT_OBJECT, "", f"抽取记录必须是对象，实际 {type(rec).__name__}"))
        return None, errors

    schema = rec.get("schema_version")
    if schema != EXTRACTION_SCHEMA:
        errors.append(
            _err(Code.UNKNOWN_SCHEMA, "schema_version", f"期望 {EXTRACTION_SCHEMA}，实际 {schema!r}")
        )
        return None, errors

    # 身份三件套必须与任务一致，否则这条记录可能是别的 Chunk 的结果
    for field, code in (
        ("chunk_id", Code.CHUNK_ID_MISMATCH),
        ("content_key", Code.CONTENT_KEY_MISMATCH),
        ("scene_id", Code.SCENE_ID_MISMATCH),
    ):
        v = _require(rec, field, errors)
        if v is None:
            continue
        if task.get(field) and v != task[field]:
            errors.append(
                _err(code, field, f"{field} 与任务不符: 记录 {v!r} != 任务 {task[field]!r}")
            )

    # entities 不能用 `or []` 兜底：缺字段、null、""、False 都是四种不同的坏数据，
    # 现有代码把它们全洗成「没抽到实体」，事后无法区分「模型真的没抽」和「模型吐了个空串」。
    if "entities" not in rec:
        errors.append(_err(Code.MISSING_FIELD, "entities", "缺少 entities 字段"))
        entities_raw: Any = None
    else:
        entities_raw = rec["entities"]
        if entities_raw is None:
            errors.append(_err(Code.MISSING_FIELD, "entities", "entities 为 null"))
            entities_raw = None
        elif isinstance(entities_raw, bool) or entities_raw == "":
            errors.append(
                _err(
                    Code.WRONG_TYPE,
                    "entities",
                    f"entities 不能用 {entities_raw!r} 代替数组（会被静默当成「没抽到实体」）",
                )
            )
            entities_raw = None

    entities_out = _check_entities(entities_raw, errors) if entities_raw is not None else []
    relations = _check_relations(rec.get("relations", []), {e["name"] for e in entities_out}, errors)

    if known_predicates is not None:
        for i, r in enumerate(relations):
            if r["relation"] not in known_predicates:
                errors.append(
                    _err(
                        Code.RELATION_UNKNOWN_PREDICATE,
                        f"relations[{i}].relation",
                        f"关系名不在词表: {r['relation']}",
                        level="warn",
                    )
                )

    # 证据可定位：必须在**本 Chunk** 的正文里逐字出现
    located: list[dict] = []
    for i, r in enumerate(relations):
        ev = r["evidence"]
        pos = chunk_text.find(ev)
        if pos < 0:
            errors.append(
                _err(
                    Code.EVIDENCE_NOT_FOUND,
                    f"relations[{i}].evidence",
                    f"证据在本 Chunk 正文中找不到: {ev[:40]!r}",
                )
            )
            continue
        item = dict(r)
        item["evidence_start"] = pos
        item["evidence_end"] = pos + len(ev)
        located.append(item)

    if errors and any(e["level"] == "error" for e in errors):
        return None, errors

    cleaned = {
        "schema_version": EXTRACTION_SCHEMA,
        "chunk_id": rec.get("chunk_id"),
        "content_key": rec.get("content_key"),
        "scene_id": rec.get("scene_id"),
        "entities": entities_out,
        "relations": located,
    }
    if "generation" in rec:
        cleaned["generation"] = rec["generation"]
    if "model" in rec:
        cleaned["model"] = rec["model"]
    return cleaned, errors


class _Missing:
    def __repr__(self) -> str:  # pragma: no cover
        return "<missing>"


_MISSING = _Missing()


def summarize_errors(errors: list[dict]) -> dict:
    """把错误列表折成「按码计数 + 首个示例」，供收据与隔离报告使用。"""
    by_code: dict[str, int] = {}
    samples: dict[str, str] = {}
    for e in errors:
        if e["level"] != "error":
            continue
        by_code[e["code"]] = by_code.get(e["code"], 0) + 1
        samples.setdefault(e["code"], f"{e['field']}: {e['detail']}")
    warns: dict[str, int] = {}
    for e in errors:
        if e["level"] == "warn":
            warns[e["code"]] = warns.get(e["code"], 0) + 1
    return {"error_count": sum(by_code.values()), "by_code": by_code, "samples": samples, "warns": warns}


def dumps_errors(errors: list[dict]) -> str:
    return json.dumps(errors, ensure_ascii=False)
