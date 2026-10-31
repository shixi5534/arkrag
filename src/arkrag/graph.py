"""L5 建图：观察汇总 -> 实体合并 -> 建边 -> 提及计算 -> 落盘与指纹。

三条硬纪律：证据为空的边不许存在；空产物不许覆盖非空产物；归一化版本必须进指纹。
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from . import atomic, cache, config, fingerprint, legacy, norm_entity, norm_relation
from .agent_tasks import AGENT_MODEL
from .vocab import ENTITY_RULE_VERSION, ENTITY_TYPES, RELATION_VERSION


def load_records(chunks: list[dict]) -> list[dict]:
    """把已抽取的 raw 结果按 Chunk 读出来。

    走 `read_for_chunk` 而不是「用 content_key 拼旧键再读」：旧键不含 chunk_id，
    overlap 的两块正文相同时会互相命中，结果就是「用 A 块的抽取结果去建 B 块的边」。
    现在只有身份核对通过的记录才进入建图输入。
    """
    out = []
    for chunk in chunks:
        raw = cache.read_for_chunk(chunk, AGENT_MODEL)
        if raw is None:
            continue
        out.append({"chunk_id": chunk["chunk_id"], "raw": raw})
    return out


def build_graph(chunks: list[dict], records: list[dict]) -> dict:
    obs = norm_entity.collect(records)

    dropped_names: Counter = Counter()
    kept: dict[str, dict] = {}
    for name, slot in obs.items():
        etype = slot["type_votes"].most_common(1)[0][0]
        noise, why = norm_entity.is_noise_name(name, etype)
        if noise:
            dropped_names[why] += 1
            continue
        kept[name] = slot

    alias_edges = [
        (name, alias)
        for name, slot in kept.items()
        for alias in sorted(slot["aliases"])
    ]
    # 多人争用的称谓（殿下/将军/老爷/父亲…）不是任何人的别名：
    # 既不参与合并，也不写进 aliases（写进去只会让按别名检索随机命中错的人）。
    ambiguous = norm_entity.ambiguous_aliases(kept)
    groups, merge_stats = norm_entity.merge_names(kept, alias_edges)

    counts = {name: slot["mention_as_entity"] for name, slot in kept.items()}
    entities: dict[str, dict] = {}
    root_to_canon: dict[str, str] = {}
    for root, members in sorted(groups.items()):
        canon = norm_entity.canonical_pick(sorted(members), counts)
        for member in members:
            root_to_canon[member] = canon
        votes: Counter = Counter()
        descs: Counter = Counter()
        aliases: set[str] = set()
        chunks_seen: set[str] = set()
        mentions = 0
        for member in members:
            slot = kept[member]
            votes.update(slot["type_votes"])
            descs.update(slot["descs"])
            mentions += slot["mention_as_entity"]
            chunks_seen |= slot["chunks"]
            aliases |= slot["aliases"]
            aliases.add(member)
        aliases.discard(canon)
        for member in members:
            aliases.discard(member)
        aliases -= set(ambiguous)
        entities[canon] = {
            "id": canon,
            "name": canon,
            "type": votes.most_common(1)[0][0],
            "desc": descs.most_common(1)[0][0] if descs else "",
            "aliases": sorted(aliases),
            "mentions": mentions,
            "chunks": sorted(chunks_seen),
            "merged_from": sorted(m for m in members if m != canon),
        }

    edges: dict[tuple[str, str, str], dict] = {}
    rel_stats: Counter = Counter()
    sidecar: dict[str, dict] = {}
    for rec in records:
        chunk_id = rec["chunk_id"]
        for rel in rec["raw"].get("relations") or []:
            src = root_to_canon.get((rel.get("source") or "").strip())
            dst = root_to_canon.get((rel.get("target") or "").strip())
            raw_name = (rel.get("relation") or "").strip()
            conf = rel.get("confidence")
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                conf = 0.0
            if src is None or dst is None:
                rel_stats["endpoint_filtered"] += 1
                continue
            decision = norm_relation.adjudicate(raw_name, src, dst)
            if decision.action == "drop":
                rel_stats[f"dropped:{decision.flag}"] += 1
                continue
            if decision.action == "keep_flagged":
                rel_stats["kept_flagged"] += 1
            canonical = decision.canonical
            if decision.swap:
                src, dst = dst, src
            if src == dst:
                rel_stats["dropped:self_loop"] += 1
                continue
            if conf < config.CONFIDENCE_THRESHOLD:
                rel_stats["dropped:low_confidence"] += 1
                continue
            evidence = (rel.get("evidence") or "").strip()
            if not evidence:
                rel_stats["dropped:no_evidence"] += 1
                continue
            key = (src, canonical, dst)
            slot = edges.get(key)
            if slot is None:
                slot = {
                    "source": src,
                    "relation": canonical,
                    "target": dst,
                    "confidence": conf,
                    "evidence": evidence,
                    "chunks": [],
                    "repeat": 0,
                    "raw_names": Counter(),
                    "flags": set(),
                }
                edges[key] = slot
            slot["repeat"] += 1
            slot["confidence"] = max(slot["confidence"], conf)
            if chunk_id not in slot["chunks"]:
                slot["chunks"].append(chunk_id)
            if raw_name != canonical:
                slot["raw_names"][raw_name] += 1
                sidecar[raw_name] = {
                    "canonical": canonical,
                    "action": decision.action,
                    "flag": decision.flag,
                    "swap": decision.swap,
                }
            if decision.action == "keep_flagged":
                slot["flags"].add(decision.flag)
            # 提及次数必须把「作为关系端点出现」也算进去，否则只作端点的实体是永久盲区
            entities[src]["mentions"] += 1
            entities[dst]["mentions"] += 1

    for slot in edges.values():
        slot["raw_names"] = dict(slot["raw_names"])
        slot["flags"] = sorted(slot["flags"])

    degree: Counter = Counter()
    for (src, _rel, dst) in edges:
        degree[src] += 1
        degree[dst] += 1
    isolated = sum(1 for e in entities if degree.get(e, 0) == 0)

    graph = {
        "schema_version": config.SCHEMA_VERSION,
        "prompt_version": config.PROMPT_VERSION,
        "normalizer_version": config.NORMALIZER_VERSION,
        "entity_rule_version": ENTITY_RULE_VERSION,
        "executor": "agent:qoder-subagent",
        "entities": [entities[k] for k in sorted(entities)],
        "edges": [
            {**edges[k], "chunks": sorted(edges[k]["chunks"])}
            for k in sorted(edges, key=lambda x: (x[0], x[1], x[2]))
        ],
        "alias_index": {alias: ent["id"] for ent in entities.values() for alias in ent["aliases"]},
        "stats": {
            "entities": len(entities),
            "edges": len(edges),
            "isolated": isolated,
            "avg_degree": (2 * len(edges) / len(entities)) if entities else 0.0,
            "dropped_names": dict(dropped_names),
            "merge": merge_stats,
            "relation_adjudication": dict(rel_stats),
            "sidecar_map": sidecar,
        },
    }
    return graph


def build_identity_graph(
    chunks: list[dict],
    records: list[dict],
    *,
    pinned_ids: dict[str, str] | None = None,
    extraction_meta: dict | None = None,
) -> dict:
    """建 `ark-rag-3` 图（E 阶段）：稳定 id 实体 + 逐观察断言 + 有方向邻接。

    ## 与 `build_graph` 的关系

    **并存，不替代**。`build_graph` 是聚合边模型（`ark-rag-1`），已发布的每一代
    产物与冻结基线都是它建的，一个字节都不许变；本函数是逐观察模型，输出走另一套
    结构。所以这里是新函数而不是给 `build_graph` 加开关——加开关意味着要在同一个
    函数体里维护两套语义，任何一处忽略 schema 判断都会让旧代产物变形。

    ## 返回什么

    返回一个**四件套**，不是单个图对象：

        {"graph": {...}, "assertions": [...], "edges": [...], "adjacency": {...},
         "alias_map": {...}}

    * `graph` 落 `<gen>/graph/graph.json`（实体 + 聚合边 + 别名索引 + 统计）；
    * `assertions` 落 `<gen>/graph/assertions.jsonl`（**逐观察**，一行一条）；
    * `adjacency` 落 `<gen>/graph/adjacency.json`（out/in 分列）。

    为什么断言单独一个产物、不塞进 graph.json：断言是 6 万多行的逐观察事实，
    塞进图会让图从 17 MB 涨到不可读；而"图上的边凭哪一段原文"这个问题必须能
    单独回查、单独做摘要（`assertion_artifact_id`），两者绑在同一个文件里就
    无法回答"是图变了还是断言变了"。

    ## 为什么不再自己算别名表

    权威别名表只有一个来源：`identity.CanonicalEntity.to_dict()`。它的 `aliases`
    只含 `review ∈ {auto, approved}`，`pending_aliases` / `rejected_aliases` 是
    另外两张表。本函数**照抄** `to_dict()` 的结果，不在这里再过滤一遍——
    「同一判定写两处，只要一处松，整套就等于那处最松的」（见 identity.py:241 的
    实测事故）。`alias_index` 因此天然不含待审别名。
    """
    from . import adjacency as adjacency_mod
    from . import assertion as assertion_mod
    from . import identity

    alias_by_chunk = {c["chunk_id"]: c for c in chunks}
    have_raw = {r["chunk_id"] for r in records}

    # ---- 1. Mention 层：只搬原始提及，不做任何判定
    mentions: list[identity.Mention] = []
    edges_by_name: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for rec in records:
        chunk_id = rec["chunk_id"]
        raw = rec.get("raw") or {}
        for ent in raw.get("entities") or []:
            mentions.append(
                identity.Mention(
                    name=str(ent.get("name") or ""),
                    chunk_id=chunk_id,
                    raw_type=str(ent.get("type") or ""),
                    desc=str(ent.get("desc") or ""),
                    aliases=tuple(
                        str(a).strip() for a in (ent.get("aliases") or []) if str(a).strip()
                    ),
                )
            )
        # 第 6 道检查（冲突关系）需要「名字 -> 该名字的边」，只能在 raw 层攒
        for rel in raw.get("relations") or []:
            src = str(rel.get("source") or "").strip()
            dst = str(rel.get("target") or "").strip()
            rel_name = str(rel.get("relation") or "").strip()
            if src and rel_name:
                edges_by_name[src].append((rel_name, dst))
            if dst and rel_name:
                edges_by_name[dst].append((rel_name, src))

    # ---- 2. 候选层：类型归一 + 噪声剔除（判据仍是 norm_entity 那一份）
    candidates, cand_stats = identity.build_candidates(mentions, drop_noise=True)

    # ---- 3. 实体层：六道检查 + 合并，id 由 stable_id 固定
    entities, merge_stats = identity.merge_candidates(
        candidates,
        edges_by_name=edges_by_name,
        entity_rule_version=ENTITY_RULE_VERSION,
        pinned_ids=pinned_ids,
    )
    entity_docs = [e.to_dict() for e in entities]

    # ---- 4. 名字 -> id：实体名、被并名、权威别名（三处都收，别名最容易漏）
    alias_map: dict[str, str] = {}
    for doc in entity_docs:
        alias_map.setdefault(doc["name"], doc["id"])
    for doc in entity_docs:
        for item in doc["merged_from"]:
            alias_map.setdefault(item["name"], doc["id"])
    for doc in entity_docs:
        for item in doc["aliases"]:
            alias_map.setdefault(item["text"], doc["id"])

    # ---- 5. 逐观察断言 + 聚合边 + 有方向邻接
    meta = dict(extraction_meta or {})
    meta.setdefault("schema_version", config.SCHEMA_VERSION)
    meta.setdefault("prompt_sha256", fingerprint.prompt_digest())
    meta.setdefault("vocab_sha", fingerprint.vocab_digest())
    meta.setdefault("entity_rule_version", ENTITY_RULE_VERSION)
    meta.setdefault("relation_version", RELATION_VERSION)
    meta.setdefault("normalizer_version", config.NORMALIZER_VERSION)
    # 契约 §1.2 要求每条断言自带「哪批输入抽出来的」。这里填空串而不是猜：
    # 真实值是整批的 `extraction_manifest_hash`，由 build_generation 传进来。
    meta.setdefault("extraction_manifest_hash", "")
    meta.setdefault("generation_id", "")

    assertions, assertion_stats = assertion_mod.build_assertions(
        chunks, records, entity_alias_map=alias_map, extraction_meta=meta
    )
    edges = assertion_mod.aggregate(assertions)
    adjacency = adjacency_mod.build_directed_adjacency(edges)

    # ---- 6. 组装产物
    degree: Counter = Counter()
    for edge in edges:
        degree[edge["subject_id"]] += 1
        degree[edge["object_id"]] += 1
    isolated = sum(1 for doc in entity_docs if degree.get(doc["id"], 0) == 0)

    missing_raw = sorted(set(alias_by_chunk) - have_raw)
    alias_counts = {
        "authoritative": sum(len(d["aliases"]) for d in entity_docs),
        "pending": sum(len(d["pending_aliases"]) for d in entity_docs),
        "rejected": sum(len(d["rejected_aliases"]) for d in entity_docs),
    }
    # L4.13 的契约是「并了就要说清按哪条规则并的；没并就允许两边都为空」。
    # 所以没发生合并时 `stats.merge` 留空，完整统计放 `stats.identity`——
    # 把"没有合并记录"的统计塞进 merge 会让门永远判 FAIL（红久了没人看）。
    merged_total = sum(len(d["merged_from"]) for d in entity_docs)

    graph = {
        "schema_version": config.GRAPH_SCHEMA_IDENTITY,
        "prompt_version": config.PROMPT_VERSION,
        "normalizer_version": config.NORMALIZER_VERSION,
        "entity_rule_version": ENTITY_RULE_VERSION,
        "relation_version": RELATION_VERSION,
        "executor": "agent:qoder-subagent",
        "entities": entity_docs,
        "edges": [{**edge} for edge in edges],
        # 与旧图同一条约定：只存别名、不存规范名；value 是实体 id。
        # 数据源是 `to_dict()` 的 `aliases`（已排除 pending/rejected）。
        "alias_index": {
            item["text"]: doc["id"] for doc in entity_docs for item in doc["aliases"]
        },
        "stats": {
            "entities": len(entity_docs),
            "edges": len(edges),
            "isolated": isolated,
            "avg_degree": (2 * len(edges) / len(entity_docs)) if entity_docs else 0.0,
            "dropped_names": {
                k.split(":", 1)[1]: v
                for k, v in cand_stats.items()
                if str(k).startswith("dropped:")
            },
            "merge": dict(merge_stats) if merged_total else {},
            "identity": dict(merge_stats),
            "candidates": dict(cand_stats),
            "assertion": assertion_mod.sum_stats(assertion_stats),
            "assertions_total": len(assertions),
            "adjacency": adjacency_mod.direction_report(adjacency),
            "alias_counts": alias_counts,
            # 缺 raw 的块必须报出来：静默少边就等于"这张图看起来正常，其实少了一批事实"
            "chunks": len(alias_by_chunk),
            "records": len(records),
            "chunks_missing_raw": missing_raw[:50],
            "chunks_missing_raw_count": len(missing_raw),
        },
    }
    return {
        "graph": graph,
        "assertions": assertions,
        "edges": edges,
        "adjacency": adjacency,
        "alias_map": alias_map,
    }


def graph_fp_inputs() -> dict:
    """建图配置的全量输入（计划 D3）。

    这里刻意不只用 `config.PROMPT_VERSION` 这类展示字符串，而是吃三类**内容摘要**：
      * `prompt_sha256` —— 提示词实际内容。改了提示词忘改版本号，指纹照样会变。
      * `vocab_sha`     —— 词表内容（211 条同义词、逆谓词、对称关系、互斥对……）。
                           加一条同义词不动代码不动提示词，却能让整张图结构变掉。
      * `chunk_params`  —— 分块参数全组，不只目标块长（重叠轮次数同样改产物）。
    """
    return {
        "schema_version": config.SCHEMA_VERSION,
        "model": "agent:qoder-subagent",
        "prompt_version": config.PROMPT_VERSION,
        "prompt_sha256": fingerprint.prompt_digest(),
        "vocab_sha": fingerprint.vocab_digest(),
        "relation_version": RELATION_VERSION,
        "chunk_chars": config.CHUNK_TARGET_CHARS,
        "chunk_params": {
            "target_chars": config.CHUNK_TARGET_CHARS,
            "min_chars": config.CHUNK_MIN_CHARS,
            "max_chars": config.CHUNK_MAX_CHARS,
            "overlap_turns": config.CHUNK_OVERLAP_TURNS,
            "t1_categories": list(config.T1_CATEGORIES),
        },
        "normalizer_version": config.NORMALIZER_VERSION,
        "entity_rule_version": ENTITY_RULE_VERSION,
    }


def attach_corpus_fp(
    graph: dict,
    corpus_fp: str,
    *,
    identities: dict | None = None,
) -> None:
    """挂上建图配置指纹，并把分层身份一起写进产物（计划 D3）。

    `identities` 是上游各层的指纹（source_fp / scene_fp / chunk_fp）。图自己算不出
    它们——那是上一层的输入——但图必须**记录**它们，否则事后无法回答
    "这张图吃的是哪一批块、那一批块来自哪一版场景"。
    """
    fp = fingerprint.graph_fingerprint(**graph_fp_inputs(), corpus_fp=corpus_fp)
    graph["corpus_fp"] = corpus_fp
    graph["fingerprint"] = fp
    # 把配置输入本身也写进产物：事后要能回答"这张图是按什么配置建的"，
    # 只留一个 16 位指纹等于把答案丢掉。
    graph["build_config"] = {
        k: v for k, v in graph_fp_inputs().items() if k != "prompt_version"
    }
    graph["identities"] = dict(identities or {})


def save(graph: dict, path: Path) -> dict:
    """拒绝空产物覆盖非空产物；固定缩进/换行/编码 + 原子替换。

    受 legacy 写保护约束（`legacy.guard_write`）：写冻结路径会直接抛错。
    新产物请写 generations/<generation_id>/graph/graph.json。
    """
    path = Path(path)
    legacy.guard_write(path, reason="graph.save")
    path.parent.mkdir(parents=True, exist_ok=True)
    new_counts = {"entities": len(graph.get("entities") or []), "edges": len(graph.get("edges") or [])}
    if path.exists():
        old = json.loads(path.read_text("utf-8"))
        old_counts = {"entities": len(old.get("entities") or []), "edges": len(old.get("edges") or [])}
        for kind in ("entities", "edges"):
            if new_counts[kind] == 0 and old_counts[kind] > 0:
                raise RuntimeError(f"拒绝用空 {kind} 覆盖非空产物（旧 {old_counts[kind]} 条）")
            if old_counts[kind] and new_counts[kind] < old_counts[kind] * 0.5:
                raise RuntimeError(
                    f"{kind} 数量异常萎缩 {old_counts[kind]} -> {new_counts[kind]}，拒绝写入"
                )
        graph["previous"] = {"fingerprint": old.get("fingerprint"), **old_counts}
    atomic.atomic_write_json(path, graph)
    return new_counts


def save_assertions(assertions: list[dict], path: Path) -> int:
    """逐观察断言落盘：JSONL，一行一条，按 assertion_id 排序。

    排序是硬要求，不是洁癖：`assertion_artifact_id` 是这一代的身份之一，
    输入顺序一变摘要就变，"同一批输入重跑得到同一份产物"立刻不成立。
    """
    path = Path(path)
    legacy.guard_write(path, reason="graph.save_assertions")
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(assertions, key=lambda a: a["assertion_id"])
    lines = "".join(json.dumps(a, ensure_ascii=False, sort_keys=True) + "\n" for a in ordered)
    atomic.atomic_write_text(path, lines)
    return len(ordered)


def save_adjacency(adjacency: dict, path: Path) -> int:
    """有方向邻接落盘：`{entity_id: {"out": [...], "in": [...]}}`。

    空邻接不许覆盖非空邻接（同 `save()` 的理由）：一张图有实体却零邻接项，
    说明邻接这一步没跑，不是"这张图本来没有边"。
    """
    path = Path(path)
    legacy.guard_write(path, reason="graph.save_adjacency")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            old = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            old = {}
        if not adjacency and old:
            raise RuntimeError(f"拒绝用空邻接覆盖非空产物（旧 {len(old)} 个实体）")
        if old and len(adjacency) < len(old) * 0.5:
            raise RuntimeError(f"邻接实体数异常萎缩 {len(old)} -> {len(adjacency)}，拒绝写入")
    atomic.atomic_write_json(path, adjacency)
    return len(adjacency)


def entity_type_report(graph: dict) -> dict:
    dist = Counter(e["type"] for e in graph["entities"])
    off_vocab = [e["id"] for e in graph["entities"] if e["type"] not in ENTITY_TYPES]
    return {"type_distribution": dict(dist), "distinct_types": len(dist), "off_vocab_entities": len(off_vocab)}
