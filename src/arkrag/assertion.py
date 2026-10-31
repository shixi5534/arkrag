"""逐观察断言（Assertion）与派生聚合边 —— 计划 E1 / E4。

## 这一层要消灭什么

改动前图里只有**聚合边**：`edges[(source, relation, target)]` 一条记录压掉同一条关系的
全部观察，只留一条 `evidence`。实测（T1 13,781 块）：

    63,172 raw relations
     −  402 建图期丢弃（endpoint_filtered 231 / self_loop 150 / low_confidence 21）
    = 62,770 进入聚合的观察  == sum(edge.repeat)
     −34,536 与同边其它观察共用一条 evidence
    = 28,234 聚合边

也就是说 **34,536 次观察在自己的图里没有证据位置**；9,747 条边的多条观察共用一条 evidence，
单边最多 116 条观察只留 1 条。回答"这个判断出自哪一段原文"时，聚合边只能给出一个代表，
剩下 115 次的出处永远丢了。

## 这里的结构

* `Assertion` 是**原子事实单元**：一次抽取产出的一条关系，带自己的 `chunk_id` 与字符偏移。
* 聚合边只是 Assertion 的**派生统计**（`assertion_ids` / `support_count` / `confidence_summary`），
  不再持有"唯一证据"的特权。

## 两条不能妥协的规矩

1. **证据定位失败就不许入库**。不做"找近似位置"、不许拿别的块的证据顶替、
   不许只留文本不留偏移。块内多处出现时必须收全部候选（`evidence_all`）——
   只留第一处等于谎报"这句话只在这块出现过一次"。
2. **守恒可查**。`sum(support_count)` 必须等于 assertion 总数，且每条 assertion 恰好被一条边引用。
   少一条就是静默丢事实，多引一次就是重复计功。`observation_conservation()` 把这件事变成可断言的数字。
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

from . import config, contracts, norm_relation, vocab

# Assertion 的 id 前缀。聚合边用 "r-"，实体用 "e-"（计划 E3），三者不互相冒充。
ASSERTION_PREFIX = "a-"
EDGE_PREFIX = "r-"

# 契约 §4：叙事限定字段的默认值。空串/默认值表示"抽取没有给"，**不等于"没有这个性质"**。
QUALIFIER_DEFAULTS: dict[str, object] = {
    "story_time": "",
    "valid_time": "",
    "scene_order": -1,
    "certainty": "asserted",
    "modality": "actual",
    "is_flashback": False,
    "is_hearsay": False,
    "is_hypothetical": False,
}

CERTAINTIES = ("asserted", "hedged", "uncertain", "negated")
MODALITIES = ("actual", "hypothetical", "reported", "remembered")


# --------------------------------------------------------------------------- id


def _sha16(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def assertion_id(
    *,
    chunk_id: str,
    subject_id: str,
    predicate: str,
    object_id: str,
    evidence_start: int,
    raw_relation: str,
) -> str:
    """稳定 id：只含身份字段（契约 §1.1）。

    刻意**不含** confidence / written_at / evidence_all：它们不是身份。把它们塞进来，
    同一个事实重跑一次就换 id，回滚比对与增量更新全部失效。
    两条观察即使文本一模一样，只要 `evidence_start` 不同就是两条独立 Assertion——
    那正是本层要保留的信息。
    """
    body = "\x1f".join(
        [
            chunk_id,
            subject_id,
            predicate,
            object_id,
            str(int(evidence_start)),
            raw_relation,
        ]
    )
    return ASSERTION_PREFIX + _sha16(body)


def edge_id(*, subject_id: str, predicate: str, object_id: str) -> str:
    """聚合边 id：由端点与谓词决定，与观察条数无关（多一条观察不该换 id）。"""
    body = "\x1f".join([subject_id, predicate, object_id])
    return EDGE_PREFIX + _sha16(body)


# ----------------------------------------------------------------- evidence 定位


def find_all_occurrences(text: str, needle: str) -> list[tuple[int, int]]:
    """找出 needle 在 text 里的**全部非重叠**出现位置（0 起，右开）。

    用 `find(needle, start)` 步进而不是正则：证据里有 `（` `·` 之类字符，
    正则要转义，而 `str.find` 天然按字面比较。步进用 `pos + 1` 而不是 `pos + len(needle)`，
    是为了让重叠出现也能被发现（"凯尔希凯尔希" 里 "凯尔希" 出现两次而且是重叠的）。
    """
    if not needle:
        return []
    out: list[tuple[int, int]] = []
    pos = text.find(needle)
    while pos >= 0:
        out.append((pos, pos + len(needle)))
        pos = text.find(needle, pos + 1)
    return out


# ----------------------------------------------------------------------- 限定字段


def default_qualifiers(**overrides) -> dict:
    """九个限定字段的默认值，可按 name 覆盖（测试与将来的时间抽取用）。"""
    unknown = set(overrides) - set(QUALIFIER_DEFAULTS)
    if unknown:
        raise ValueError(f"未知的限定字段: {sorted(unknown)}")
    out = dict(QUALIFIER_DEFAULTS)
    out.update(overrides)
    return out


def validate_qualifiers(qual: object) -> list[str]:
    """类型校验。错的**报出来**，不静默修正——静默修正会让错误数据看起来合规。"""
    problems: list[str] = []
    if not isinstance(qual, dict):
        return ["qualifiers 不是对象"]
    missing = set(QUALIFIER_DEFAULTS) - set(qual)
    if missing:
        problems.append(f"缺字段: {sorted(missing)}")
    unknown = set(qual) - set(QUALIFIER_DEFAULTS)
    if unknown:
        problems.append(f"未知字段: {sorted(unknown)}")
    for key in ("story_time", "valid_time"):
        if key in qual and not isinstance(qual[key], str):
            problems.append(f"{key} 应为字符串，实为 {type(qual[key]).__name__}")
    if "scene_order" in qual:
        value = qual["scene_order"]
        if isinstance(value, bool) or not isinstance(value, int):
            problems.append(f"scene_order 应为整数，实为 {type(value).__name__}")
        elif value < -1:
            problems.append(f"scene_order 不得小于 -1，实为 {value}")
    if "certainty" in qual and qual["certainty"] not in CERTAINTIES:
        problems.append(f"certainty 取值非法: {qual['certainty']!r}")
    if "modality" in qual and qual["modality"] not in MODALITIES:
        problems.append(f"modality 取值非法: {qual['modality']!r}")
    for key in ("is_flashback", "is_hearsay", "is_hypothetical"):
        if key in qual and not isinstance(qual[key], bool):
            problems.append(f"{key} 应为布尔，实为 {type(qual[key]).__name__}")
    return problems


# -------------------------------------------------------------------- 建 Assertion


def build_assertions(
    chunks: list[dict],
    records: list[dict],
    *,
    entity_alias_map: dict[str, str],
    extraction_meta: dict | None = None,
    min_confidence: float | None = None,
    qualifiers_override: dict | None = None,
) -> tuple[list[dict], dict]:
    """从「块 + raw 抽取 + 已消歧实体表」构建逐观察断言。

    `entity_alias_map` 是 E3 的产物（原始名 -> canonical entity id）。这里只接收、不造：
    映射里没有的名字意味着"这个提及还没被消歧"，此时该观察**跳过并计数**，
    绝不拿原始名冒充 id（那会绕过 E3 的全部防误并检查）。
    """
    by_chunk = {c["chunk_id"]: c for c in chunks}
    threshold = config.CONFIDENCE_THRESHOLD if min_confidence is None else float(min_confidence)

    assertions: list[dict] = []
    seen_ids: dict[str, int] = {}
    duplicate_ids: list[dict] = []
    samples: dict[str, list[dict]] = defaultdict(list)

    stats: Counter = Counter()
    stats["records"] = len(records)

    for rec in records:
        chunk_id = rec.get("chunk_id") or ""
        chunk = by_chunk.get(chunk_id)
        if chunk is None:
            stats["dropped:chunk_unknown"] += 1
            if len(samples["chunk_unknown"]) < 5:
                samples["chunk_unknown"].append({"chunk_id": chunk_id})
            continue
        text = chunk.get("text") or ""
        doc_id = chunk.get("doc_id") or chunk_id.split("#")[0]
        scene_id = chunk_id.split("#")[0]
        scene_order = chunk.get("scene_index", -1)
        raw = rec.get("raw") or {}

        for rel in raw.get("relations") or []:
            stats["raw_relations"] += 1
            raw_relation = (rel.get("relation") or "").strip()
            evidence = (rel.get("evidence") or "").strip()
            src_name = (rel.get("source") or "").strip()
            dst_name = (rel.get("target") or "").strip()

            if not evidence:
                stats["dropped:evidence_invalid"] += 1
                _sample(samples, "evidence_invalid", chunk_id, raw_relation, "空证据")
                continue
            if len(evidence) > contracts.EVIDENCE_MAX_CHARS:
                stats["dropped:evidence_invalid"] += 1
                _sample(samples, "evidence_invalid", chunk_id, raw_relation, f"证据 {len(evidence)} 字")
                continue

            subject_id = entity_alias_map.get(src_name) or ""
            object_id = entity_alias_map.get(dst_name) or ""
            if not subject_id or not object_id:
                stats["dropped:endpoint_unknown"] += 1
                _sample(
                    samples,
                    "endpoint_unknown",
                    chunk_id,
                    raw_relation,
                    f"{src_name!r}->{dst_name!r}",
                )
                continue

            dec = norm_relation.adjudicate(raw_relation, src_name, dst_name)
            if dec.action == "drop":
                stats[f"dropped:adjudicated:{dec.flag}"] += 1
                _sample(samples, f"adjudicated:{dec.flag}", chunk_id, raw_relation, dec.flag)
                continue
            predicate = dec.canonical or raw_relation
            if dec.swap:
                # 方向归一：谓词描述的其实是反方向。换端点而不是给端点加"反向"标记——
                # 契约 §1.3 规定 Assertion 上的 direction 恒为 out。
                subject_id, object_id = object_id, subject_id
                stats["swapped_direction"] += 1
            if dec.action == "keep_flagged":
                # 归一不了但也不是垃圾词。保留（它是真实观察），只计数，让 L4.6 去判。
                stats[f"kept_flagged:{dec.flag}"] += 1

            if subject_id == object_id:
                stats["dropped:self_loop"] += 1
                _sample(samples, "self_loop", chunk_id, raw_relation, subject_id)
                continue

            confidence = rel.get("confidence")
            if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
                stats["dropped:bad_confidence"] += 1
                _sample(samples, "bad_confidence", chunk_id, raw_relation, repr(confidence))
                continue
            confidence = float(confidence)
            if confidence < threshold:
                stats["dropped:low_confidence"] += 1
                continue

            occurrences = find_all_occurrences(text, evidence)
            if not occurrences:
                stats["dropped:evidence_not_found"] += 1
                _sample(samples, "evidence_not_found", chunk_id, raw_relation, evidence[:40])
                continue

            start, end = occurrences[0]
            if len(occurrences) > 1:
                stats["evidence_multiple_occurrences"] += 1

            qualifiers = default_qualifiers(**(qualifiers_override or {}))
            qualifiers["scene_order"] = scene_order

            a_id = assertion_id(
                chunk_id=chunk_id,
                subject_id=subject_id,
                predicate=predicate,
                object_id=object_id,
                evidence_start=start,
                raw_relation=raw_relation,
            )
            if a_id in seen_ids:
                # 重复 id 必须报出来。旧代码在这里是 `got[cid] = rec` 直接覆盖，
                # 于是"后一条赢"而前一条静默消失——本项目踩过两次的坑。
                stats["duplicate_assertion_id"] += 1
                if len(duplicate_ids) < 20:
                    duplicate_ids.append(
                        {
                            "assertion_id": a_id,
                            "chunk_id": chunk_id,
                            "first_index": seen_ids[a_id],
                            "raw_relation": raw_relation,
                        }
                    )
                continue
            seen_ids[a_id] = len(assertions)
            stats["accepted"] += 1

            assertions.append(
                {
                    "assertion_id": a_id,
                    "subject_id": subject_id,
                    "predicate": predicate,
                    "object_id": object_id,
                    "direction": "out",
                    "raw_relation": raw_relation,
                    "chunk_id": chunk_id,
                    "scene_id": scene_id,
                    "doc_id": doc_id,
                    "evidence_text": evidence,
                    "evidence_start": start,
                    "evidence_end": end,
                    "evidence_all": [{"start": s, "end": e} for s, e in occurrences],
                    "confidence": confidence,
                    "qualifiers": qualifiers,
                    "extraction": dict(extraction_meta or {}),
                }
            )

    out_stats = dict(stats)
    out_stats["duplicate_assertion_ids"] = duplicate_ids
    out_stats["samples"] = {k: v for k, v in samples.items()}
    out_stats["min_confidence"] = threshold
    return assertions, out_stats


def _sample(samples: dict, key: str, chunk_id: str, relation: str, detail: str) -> None:
    bucket = samples[key]
    if len(bucket) < 5:
        bucket.append({"chunk_id": chunk_id, "relation": relation, "detail": detail})


# ------------------------------------------------------------------------ 聚合


def aggregate(assertions: list[dict]) -> list[dict]:
    """把断言压成派生统计边。

    边**不再持有唯一证据的特权**：`evidence` 只是"代表值"，且必须能在同边某条 assertion 里找到
    （`L5.10` 就是查这件事）。真正的事实集合是 `assertion_ids`。
    """
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for a in assertions:
        groups[(a["subject_id"], a["predicate"], a["object_id"])].append(a)

    # 互斥谓词的冲突计数要跨边算：同盟的冲突对象是敌对，它们必然是两条边。
    by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for a in assertions:
        by_pair[(a["subject_id"], a["object_id"])].append(a)

    edges: list[dict] = []
    for (subject_id, predicate, object_id), items in groups.items():
        confidences = [float(a["confidence"]) for a in items]
        scenes = {a["scene_id"] for a in items}
        # 代表证据取"最可信的一条"，同分按 assertion_id 排，保证重跑结果一致。
        lead = sorted(items, key=lambda a: (-float(a["confidence"]), a["assertion_id"]))[0]

        exclusive = set()
        for pair in vocab.MUTUALLY_EXCLUSIVE_PAIRS:
            a, b = pair
            if predicate == a:
                exclusive.add(b)
            elif predicate == b:
                exclusive.add(a)
        conflict_count = sum(
            1
            for a in by_pair.get((subject_id, object_id), [])
            if a["predicate"] in exclusive
        )

        edges.append(
            {
                "edge_id": edge_id(
                    subject_id=subject_id, predicate=predicate, object_id=object_id
                ),
                "subject_id": subject_id,
                "predicate": predicate,
                "object_id": object_id,
                "direction": "out",
                "assertion_ids": [a["assertion_id"] for a in items],
                "support_count": len(items),
                "independent_scene_count": len(scenes),
                "confidence_summary": {
                    "min": min(confidences),
                    "max": max(confidences),
                    "mean": sum(confidences) / len(confidences),
                },
                "conflict_count": conflict_count,
                "evidence": lead["evidence_text"],
                "evidence_assertion_id": lead["assertion_id"],
                "chunks": sorted({a["chunk_id"] for a in items}),
                "scenes": sorted(scenes),
            }
        )
    edges.sort(key=lambda e: (-e["support_count"], e["edge_id"]))
    return edges


def observation_conservation(assertions: list[dict], edges: list[dict]) -> dict:
    """守恒检查：每条断言恰好被一条边引用，且边声明的 support_count 与引用数一致（契约 §5.1）。"""
    referenced: Counter = Counter()
    for e in edges:
        for a_id in e["assertion_ids"]:
            referenced[a_id] += 1
    known = {a["assertion_id"] for a in assertions}
    support_sum = sum(int(e.get("support_count") or 0) for e in edges)
    unreferenced = sorted(known - set(referenced))
    multiply = sorted(a for a, n in referenced.items() if n > 1)
    unknown_refs = sorted(set(referenced) - known)
    support_mismatch = [
        {
            "edge_id": e["edge_id"],
            "declared": int(e.get("support_count") or 0),
            "actual": len(e["assertion_ids"]),
        }
        for e in edges
        if int(e.get("support_count") or 0) != len(e["assertion_ids"])
    ]
    ok = (
        support_sum == len(assertions)
        and not unreferenced
        and not multiply
        and not unknown_refs
        and not support_mismatch
    )
    return {
        "assertions": len(assertions),
        "edges": len(edges),
        "support_sum": support_sum,
        "ok": ok,
        "unreferenced": unreferenced[:20],
        "multiply_referenced": multiply[:20],
        "unknown_references": unknown_refs[:20],
        "support_mismatch": support_mismatch[:20],
    }


# ------------------------------------------------------------------ 互斥与冲突


def check_qualifier_consistency(assertions: list[dict]) -> list[dict]:
    """互斥谓词的共存检查（契约 §4 末段）。

    互斥关系**不直接删除**：剧情里"先同盟后敌对"是完全合法的事实演进。
    所以这里只区分两件事：

    * 落在**不同场景** → 两条都保留，记一条 `kept_both` 冲突（时序演进的可能解释）。
    * 落在**同一场景** → 可能是抽取矛盾，记 `needs_human` 并要求该对里的 certainty 被标 uncertain。

    绝不做的事：悄悄挑一条留下、或按置信度高低砍掉一条。
    """
    # 按**无序**端点对分组：A-D 与 D-A 是同一对实体，互斥冲突要在同一组里看。
    # 用 frozenset 会让 (A,A) 退化，所以用排序后的元组。
    by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for a in assertions:
        key = tuple(sorted((a["subject_id"], a["object_id"])))
        by_pair[key].append(a)

    out: list[dict] = []
    for (left, right), items in sorted(by_pair.items()):
        for i, a in enumerate(items):
            for b in items[i + 1 :]:
                if not _mutually_exclusive(a["predicate"], b["predicate"]):
                    continue
                same_scene = a["scene_id"] == b["scene_id"]
                out.append(
                    {
                        "subject_id": a["subject_id"],
                        "object_id": a["object_id"],
                        "a": a["assertion_id"],
                        "b": b["assertion_id"],
                        "predicates": [a["predicate"], b["predicate"]],
                        "scenes": [a["scene_id"], b["scene_id"]],
                        "same_scene": same_scene,
                        "action": "needs_human" if same_scene else "kept_both",
                        "note": (
                            "同一场景内互斥谓词并存，须人工判定；该对 certainty 应标 uncertain"
                            if same_scene
                            else "不同场景的互斥谓词，按剧情时序演进保留两条"
                        ),
                    }
                )
    return out


def _mutually_exclusive(a: str, b: str) -> bool:
    for x, y in vocab.MUTUALLY_EXCLUSIVE_PAIRS:
        if {a, b} == {x, y}:
            return True
    return False


def sum_stats(stats: dict) -> dict:
    """把 Counter 风格的统计整理成可读摘要（写进图 stats，便于门禁与台账引用）。"""
    dropped = {
        k.split(":", 1)[1]: v
        for k, v in stats.items()
        if isinstance(k, str) and k.startswith("dropped:")
    }
    kept = {
        k.split(":", 1)[1]: v
        for k, v in stats.items()
        if isinstance(k, str) and k.startswith("kept_flagged:")
    }
    return {
        "raw_relations": stats.get("raw_relations", 0),
        "accepted": stats.get("accepted", 0),
        "dropped": dict(sorted(dropped.items())),
        "dropped_total": sum(dropped.values()),
        "kept_flagged": dict(sorted(kept.items())),
        "kept_flagged_total": sum(kept.values()),
        "swapped_direction": stats.get("swapped_direction", 0),
        "evidence_multiple_occurrences": stats.get("evidence_multiple_occurrences", 0),
        "duplicate_assertion_id": stats.get("duplicate_assertion_id", 0),
        "min_confidence": stats.get("min_confidence"),
    }
