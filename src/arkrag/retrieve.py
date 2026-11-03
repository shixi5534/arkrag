"""检索与有界图扩展 —— 计划 G1 / G4 / G5。

## 分工

`query.analyze_query` 只说"问了什么"，本模块说"去库里找什么、找到多少就停"。
`answer` 只说"根据拿到的材料能说什么"。三件事分开写，是因为它们失败的后果不同：
问句解析错了会让整条链路答偏，检索错了会让答案没有证据，生成错了会编事实。

## 四条不许退让的规矩

1. **扩展必须带原文证据。** 每个扩展项都带 `chunk_id` + `evidence`（逐字来自
   `assertions.jsonl` 的 `evidence_text`）。只返回"凯尔希 --隶属--> 罗德岛"这种
   图结论、不返回那句话在哪里说的，等于要求下游相信一个不可查的断言。图里的
   每条边本来就是从具体观察聚合出来的，不拿它用，等于白扔 62,910 条断言。
2. **默认只走出边。** `adjacency` 把 in/out 分列，就是为了让"读反向"成为一个
   显式动作。要读入边必须 `include_in=True`，结果标 `used_in_edges=True`，
   并且每条入边的 `relation` 换成逆谓词（`adjacency.adjacency_item` 已经算好）。
   反向沿用原谓词是 E2 记录过的真实 bug，这里不重犯。
3. **预算硬性、用尽要说。** `max_depth / max_nodes / max_edges / max_degree_per_entity
   / max_assertions / max_context_tokens / timeout_ms` 全部可配且全部回显实际用量；
   任一项用尽就写 `truncated_reason`。悄悄多取一点，事后没人知道上下文缺了什么。
4. **同 chunk 只留一次，来源合并。** 词法命中与图扩展常指向同一块；拆成两条会让
   答案重复引用同一句话，看起来"证据充分"其实只有一条。

## 五档扩展优先级（`stats["priority_tiers"]` 各档计数）

| 档 | 判据 | 理由 |
|---|---|---|
| 1 | 边的一端**就是**查询实体，且深度 1 | 用户直接问到的关系最相关 |
| 2 | 谓词命中 `relation_hints` | 「凯尔希和罗德岛什么关系」里出现的关系词 |
| 3 | `support_count >= SUPPORT_HIGH` | 出现多次的关系比只出现一次的可信 |
| 4 | 边的 `chunks` 与已召回块有交集 | 已经找到的正文旁边的关系更可能是上下文 |
| 5 | 证据断言有 `qualifiers.scene_order` 且查询含时间词 | **只做标记，不做时序推理**（见下方「已知边界」） |

一条边可能同时命中多档，取**最小档号**作为它的 `priority_tier`，其余档号留在
`matched_tiers` 里——这样排序只看一个键，而计数仍能反映真实覆盖。

## 已知边界（如实写在这里，不靠判据掩盖）

* 第 5 档**不做真实时序推理**。语料里没有可靠的绝对时间轴，`scene_order` 是
  "本场景内第几条观察"，跨场景不可比。所以这里只标出"这条证据带场景序、查询
  提到了时间词"，把是否成立的判断留给 `answer` 的 warnings 与人工复核。
* **不排序不收敛的图游走**。深度上限 2，每实体邻居上限 20，节点上限 80；
  这些数字是为了"一次问答能在两秒内完成"，不是"覆盖整张图"。
* `token` 估算用与建库同一套分词器（`index.tokenize`）。它与任何真实 LLM 的
  分词都不同；写成文档是为了让"6000 token"这个数字可复现，不是声称精确。
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict

from . import adjacency as adjacency_mod
from . import index as index_mod
from . import query as query_mod

# 默认预算。每个数字都写在这里，调用方可逐个覆盖；没有任何一个藏在下游函数里。
DEFAULT_BUDGET: dict = {
    "max_depth": 2,
    "max_nodes": 80,
    "max_edges": 200,
    "max_degree_per_entity": 20,
    "max_assertions": 200,
    # 单条边最多取几条观察做锚点。没有这条上限时，实测第一个实体的一条
    # `交谈` 边就带着 121 条断言，直接吃掉整个 max_assertions 预算，于是
    # 整次扩展只产出 1 条边——预算被一条边独占，其余邻居一个都进不来。
    "max_assertions_per_edge": 5,
    "max_context_tokens": 6000,
    "timeout_ms": 2000,
    # 第 3 档的门槛：出现至少两次才算"可信度较高"。设成 2 而不是 1，是因为
    # 实测 28,625 条边里有相当一部分 support_count=1；全算高可信等于没有这一档。
    "support_high": 2,
}

# 实体加成通道的权重。选 0.5 的理由：词法命中是"正文里确实出现了这些词"，
# 图实体命中是"问句里出现了图里的名字"，后者更可能是问句命中了某个专名而不代表
# 那块的正文更相关。给一半权重正好让它能顶掉弱匹配，顶不掉强匹配。
ENTITY_CHANNEL_WEIGHT = 0.5

# 上下文单元的固定来源标记，避免各处手写字符串拼错。
SOURCE_BM25 = "bm25"
SOURCE_ENTITY = "entity"
SOURCE_GRAPH = "graph"


def estimate_tokens(text: str, *, tokenizer_version: str = index_mod.TOKENIZER_VERSION,
                    name_terms=None) -> int:
    """用建库同一套分词器估 token 数。

    与真实 LLM 分词不同，这一点在模块 docstring 里已如实声明。用同一套的好处是
    "上下文 6000 token"这个结论在测试里可复现，不会因为环境不同而漂。
    """
    if not text:
        return 0
    return len(index_mod.tokenize(text, version=tokenizer_version, name_terms=name_terms))


def _norm_scores(hits: list[tuple[str, float]]) -> dict[str, float]:
    """把一路得分归一到 [0, 1]。空输入返回空 dict。"""
    if not hits:
        return {}
    top = max(s for _, s in hits) or 1.0
    return {cid: s / top for cid, s in hits}


class Retriever:
    """检索 + 有界扩展。

    参数
    ----
    index
        `index.LexicalIndex`。分词版本与整词名表随索引走，查询侧不另造口径。
    graph
        `ark-rag-3` 的图 dict（需要 `entities` / `edges`）。
    adjacency
        `adjacency.build_directed_adjacency` 的输出。
    chunks
        `{chunk_id: 块}` 或块列表。**没有它就无法给出原文证据**，所以缺块时
        相关单元会被跳过并计入 `stats["skipped_units"]`，而不是伪造空正文。
    assertions
        `assertions.jsonl` 读出来的列表（可选）。有它才能给出 evidence 的
        绝对偏移；没有时扩展项仍带 `evidence`，但 `evidence_spans` 为空。
    budget
        覆盖 `DEFAULT_BUDGET` 中的任意键。
    """

    def __init__(
        self,
        index,
        graph: dict,
        adjacency: dict,
        *,
        chunks=None,
        assertions=None,
        budget: dict | None = None,
        query_context=None,
        vocab=None,
    ) -> None:
        self.index = index
        self.graph = graph or {}
        self.adjacency = adjacency or {}
        self.budget = dict(DEFAULT_BUDGET)
        self.budget.update(budget or {})

        self.entities_by_id: dict[str, dict] = {}
        self.entities_by_name: dict[str, dict] = {}
        self.alias_index: dict[str, str] = {}
        self._build_entity_tables()

        # 问句解析的只读预构建结构（`query.QueryContext`）。这里**惰性构造**：
        # 不传就用本图的 entities + 默认词表建一份，之后每问复用。
        #
        # 为什么放在 Retriever 上而不是 query 模块里做全局缓存：这些结构必须与
        # 本实例的 `graph` 成对，`Retriever` 已经绑定了那张图，生命周期天然对齐。
        # 见 `query.QueryContext` 的实测表（每问省约 98 ms）。
        if query_context is not None:
            self.query_context = query_context
        else:
            self.query_context = query_mod.QueryContext(
                self.graph.get("entities") or [],
                self.graph.get("alias_index"),
                vocab,
            )

        self.edges_by_id: dict[str, dict] = {}
        for edge in self.graph.get("edges") or []:
            eid = str(edge.get("edge_id") or "")
            if eid:
                self.edges_by_id[eid] = edge

        self.chunks: dict[str, dict] = {}
        if isinstance(chunks, dict):
            self.chunks = {str(k): v for k, v in chunks.items()}
        else:
            for rec in chunks or ():
                cid = str(rec.get("chunk_id") or "")
                if cid:
                    self.chunks[cid] = rec

        self.assertions_by_id: dict[str, dict] = {}
        self.assertions_by_chunk: dict[str, list[dict]] = defaultdict(list)
        for rec in assertions or ():
            aid = str(rec.get("assertion_id") or "")
            if not aid:
                continue
            self.assertions_by_id[aid] = rec
            cid = str(rec.get("chunk_id") or "")
            if cid:
                self.assertions_by_chunk[cid].append(rec)
        for items in self.assertions_by_chunk.values():
            items.sort(key=lambda r: str(r.get("assertion_id") or ""))

    # ------------------------------------------------------------ 实体表
    def _build_entity_tables(self) -> None:
        """建 `id->实体`、`name->实体`、`别名->id` 三张表。

        别名表直接用 `index.build_alias_index`——它按元素类型分派，两代图都能吃，
        且**不含 pending_aliases**（`identity.to_dict()` 已把它们分到另一张表）。
        在这里再写一遍过滤就是同一判定写两处，`tests/test_single_source_of_truth.py`
        记录的三次事故都是这个机理。
        """
        entities = list(self.graph.get("entities") or [])
        for ent in entities:
            eid = str(ent.get("id") or "")
            if eid:
                self.entities_by_id[eid] = ent
            name = str(ent.get("name") or "")
            if name:
                self.entities_by_name.setdefault(name, ent)
        self.alias_index = index_mod.build_alias_index(entities)

    def display_name(self, entity_id: str) -> str:
        ent = self.entities_by_id.get(str(entity_id)) or {}
        return str(ent.get("name") or entity_id)

    # ------------------------------------------------------------ 检索
    def search(self, query_analysis: dict, *, top_k: int = 20) -> dict:
        """BM25 命中，两路加权合并。

        通道与公式（写清是为了让分数可复算）：

        * `bm25`：`index.search(retrieval_query, top_k=top_k*3)`，再按该路最高分
          归一到 `[0,1]`。
        * `entity`：把每个实体候选的**规范名 + 权威别名**拼成一个查询串分别检索，
          每路同样归一，按候选 `score` 加权取该 chunk 的最大值。
        * 合并：`final = bm25_norm + ENTITY_CHANNEL_WEIGHT * entity_norm`。
          `ENTITY_CHANNEL_WEIGHT = 0.5` 的理由见常量注释。

        返回 `{hits:[{chunk_id, score, sources:[{channel, score}]}], stats:{...}}`。
        两路的原始分都留在 `sources` 里，谁贡献了多少可以逐条查。
        """
        top_k = max(1, int(top_k))
        offset = 0
        if not query_analysis.get("should_answer", True):
            return {
                "hits": [],
                "stats": {
                    "skipped_reason": str(query_analysis.get("reason") or "not_answerable"),
                    "bm25_terms": 0,
                    "entity_channels": 0,
                    "candidates": 0,
                    "top_k": top_k,
                },
            }
        text = str(
            query_analysis.get("retrieval_query")
            or query_analysis.get("normalized_query")
            or ""
        )
        bm25_raw = self.index.search(text, top_k=top_k * 3) if text else []
        bm25 = _norm_scores(bm25_raw)

        entity_best: dict[str, float] = {}
        entity_channels = 0
        for cand in query_analysis.get("entity_candidates") or []:
            eid = str(cand.get("entity_id") or "")
            ent = self.entities_by_id.get(eid) or {}
            terms = [str(ent.get("name") or cand.get("name") or "")]
            for item in ent.get("aliases") or []:
                t = item.get("text") if isinstance(item, dict) else item
                if str(t or "").strip():
                    terms.append(str(t).strip())
            probe = " ".join(sorted({t for t in terms if t}))
            if not probe:
                continue
            entity_channels += 1
            weight = float(cand.get("score") or 0.0)
            for cid, score in _norm_scores(self.index.search(probe, top_k=top_k * 3)).items():
                entity_best[cid] = max(entity_best.get(cid, 0.0), score * weight)

        merged: dict[str, dict] = {}
        for cid, score in bm25.items():
            merged.setdefault(cid, {"chunk_id": cid, "sources": []})["sources"].append(
                {"channel": SOURCE_BM25, "score": round(score, 6)}
            )
        for cid, score in entity_best.items():
            merged.setdefault(cid, {"chunk_id": cid, "sources": []})["sources"].append(
                {"channel": SOURCE_ENTITY, "score": round(score, 6)}
            )
        hits: list[dict] = []
        for cid, item in merged.items():
            total = 0.0
            for src in item["sources"]:
                if src["channel"] == SOURCE_BM25:
                    total += src["score"]
                else:
                    total += ENTITY_CHANNEL_WEIGHT * src["score"]
            hits.append(
                {
                    "chunk_id": cid,
                    "score": round(total, 6),
                    "sources": sorted(item["sources"], key=lambda s: s["channel"]),
                }
            )
        hits.sort(key=lambda h: (-h["score"], h["chunk_id"]))
        hits = hits[:top_k]

        # 词法通道单独贡献的 top1 分数，用于 answer 的"检索不足"判据
        top_score = hits[0]["score"] if hits else 0.0
        return {
            "hits": hits,
            "stats": {
                "bm25_terms": len(bm25),
                "bm25_top_score": round(max(bm25.values()), 6) if bm25 else 0.0,
                "entity_channels": entity_channels,
                "entity_top_score": round(max(entity_best.values()), 6) if entity_best else 0.0,
                "candidates": len(merged),
                "top_score": top_score,
                "top_k": top_k,
                "entity_weight": ENTITY_CHANNEL_WEIGHT,
                "skipped_reason": "",
                "queried_text": text,
            },
        }

    # ------------------------------------------------------------ 扩展
    def expand(
        self,
        entity_ids,
        *,
        depth: int = 1,
        budget: dict | None = None,
        include_in: bool = False,
        query_analysis: dict | None = None,
        seed_chunk_ids=None,
    ) -> dict:
        """从给定实体出发做有界扩展。

        只走出边；`include_in=True` 时才读入边，且入边项的 `relation` 换成逆谓词、
        结果标 `used_in_edges=True`。每条扩展项都带 `chunk_id` + `evidence`。

        返回 `{expansions:[...], stats:{...}}`。`stats` 里既有预算原始值也有实际用量，
        用尽时 `stats["truncated_reason"]` 非空。
        """
        merged_budget = dict(self.budget)
        merged_budget.update(budget or {})
        max_depth = max(0, int(merged_budget["max_depth"]))
        depth = min(max(0, int(depth)), max_depth)
        max_nodes = max(1, int(merged_budget["max_nodes"]))
        max_edges = max(0, int(merged_budget["max_edges"]))
        max_degree = max(1, int(merged_budget["max_degree_per_entity"]))
        max_assertions = max(0, int(merged_budget["max_assertions"]))
        max_per_edge = max(1, int(merged_budget.get("max_assertions_per_edge", 5)))
        support_high = int(merged_budget["support_high"])
        deadline = time.monotonic() + max(1, int(merged_budget["timeout_ms"])) / 1000.0

        qa = query_analysis or {}
        hint_predicates = {str(h.get("canonical") or h.get("raw") or "") for h in (qa.get("relation_hints") or [])}
        hint_predicates.discard("")
        time_constrained = bool(qa.get("time_constraints"))
        seeds = [str(e) for e in (entity_ids or []) if str(e)]
        seed_set = set(seeds)
        seed_chunks = {str(c) for c in (seed_chunk_ids or []) if str(c)}

        priority = {}   # 档号 -> 判据说明
        expansions: list[dict] = []
        seen_edge_keys: set[tuple[str, str]] = set()
        visited_nodes: list[str] = []
        truncated_reason = ""
        used_in_edges = False
        assertions_emitted = 0
        tier_counts = defaultdict(int)
        per_entity_taken: dict[str, int] = defaultdict(int)
        frontier = [(eid, 0) for eid in seeds] if depth > 0 else []
        visited = set(seeds)

        while frontier:
            if not frontier:
                break
            node_id, node_depth = frontier.pop(0)
            if node_id not in visited_nodes:
                visited_nodes.append(node_id)
            if len(visited_nodes) >= max_nodes:
                truncated_reason = truncated_reason or f"max_nodes({max_nodes})"
                break
            if time.monotonic() > deadline:
                truncated_reason = truncated_reason or "timeout_ms"
                break

            directions = ["out"] + (["in"] if include_in else [])
            candidates: list[tuple[dict, str]] = []
            for direction in directions:
                for item in adjacency_mod.neighbors(self.adjacency, node_id, direction=direction):
                    candidates.append((item, direction))
            # 先按档号、再按 support_count、最后按 edge_id 排——
            # 排序键只有这三个，避免"看起来更相关"这类没有判据的排序。
            scored: list[tuple[int, int, str, dict, str, list[int]]] = []
            for item, direction in candidates:
                edge_id = str(item.get("edge_id") or "")
                key = (node_id, edge_id) if direction == "out" else (edge_id, node_id)
                if key in seen_edge_keys:
                    continue
                edge = self.edges_by_id.get(edge_id) or {}
                tiers: list[int] = []
                if node_depth == 0 and node_id in seed_set:
                    tiers.append(1)
                if hint_predicates and item.get("relation") in hint_predicates:
                    tiers.append(2)
                if int(item.get("support_count") or 0) >= support_high:
                    tiers.append(3)
                if seed_chunks and set(edge.get("chunks") or []) & seed_chunks:
                    tiers.append(4)
                if time_constrained:
                    tiers.append(5)
                scored.append(
                    (
                        min(tiers) if tiers else 99,
                        -int(item.get("support_count") or 0),
                        edge_id,
                        item,
                        direction,
                        sorted(tiers),
                    )
                )
            scored.sort(key=lambda row: (row[0], row[1], row[2]))

            stopped = False
            for tier, _neg_support, edge_id, item, direction, tiers in scored:
                if len(expansions) >= max_edges:
                    truncated_reason = truncated_reason or f"max_edges({max_edges})"
                    stopped = True
                    break
                if per_entity_taken[node_id] >= max_degree:
                    # **只停止当前实体的展开，不终止整个遍历。**
                    # 原实现这里置 `stopped = True`，于是 `max_degree_per_entity` 一旦
                    # 用完就 `break` 出整个 while：实测问「凯尔希和罗德岛是什么关系？」
                    # 时凯尔希出边够 20 条即停，**第二个种子罗德岛从未被访问**
                    # （`nodes_visited=1`），问句里另一半实体对应的边根本不会出现在
                    # 上下文里。多实体问句必然踩到这条。
                    truncated_reason = truncated_reason or f"max_degree_per_entity({max_degree})"
                    break
                if time.monotonic() > deadline:
                    truncated_reason = truncated_reason or "timeout_ms"
                    stopped = True
                    break
                edge = self.edges_by_id.get(edge_id) or {}
                remaining = max_assertions - assertions_emitted
                anchors, used = self._edge_anchors(edge, min(max_per_edge, remaining))
                if not anchors:
                    # 没有可定位证据的边不进上下文：图结论没有原句支撑时，
                    # 写进上下文只会让下游更容易编事实。
                    continue
                if assertions_emitted + len(anchors) > max_assertions:
                    truncated_reason = truncated_reason or f"max_assertions({max_assertions})"
                    stopped = True
                    break
                assertions_emitted += len(anchors)
                per_entity_taken[node_id] += 1
                key = (node_id, edge_id) if direction == "out" else (edge_id, node_id)
                seen_edge_keys.add(key)
                if direction == "in":
                    used_in_edges = True
                tier_counts[tier] += 1
                priority.setdefault(tier, _tier_reason(tier))
                neighbor_id = str(item.get("neighbor_id") or "")
                expansions.append(
                    {
                        "from_id": node_id,
                        "from_name": self.display_name(node_id),
                        "neighbor_id": neighbor_id,
                        "neighbor_name": self.display_name(neighbor_id),
                        "relation": str(item.get("relation") or ""),
                        "inverse_relation": str(item.get("inverse_relation") or ""),
                        "direction": str(item.get("direction") or direction),
                        "symmetric": bool(item.get("symmetric")),
                        "edge_id": edge_id,
                        "assertion_ids": list(item.get("assertion_ids") or []),
                        "support_count": int(item.get("support_count") or 0),
                        "independent_scene_count": int(item.get("independent_scene_count") or 0),
                        "priority_tier": tier,
                        "matched_tiers": tiers,
                        "depth": node_depth + 1,
                        "used_in_edges": direction == "in" or bool(used),
                        "chunk_id": anchors[0]["chunk_id"],
                        "evidence": anchors[0]["evidence"],
                        "evidence_spans": [
                            {"chunk_id": a["chunk_id"], "start": a["start"], "end": a["end"]}
                            for a in anchors
                        ],
                        "anchors": anchors,
                    }
                )
                if neighbor_id and neighbor_id not in visited and node_depth + 1 <= depth:
                    visited.add(neighbor_id)
                    if node_depth + 1 < depth:
                        frontier.append((neighbor_id, node_depth + 1))
            if stopped:
                break

        stats = {
            "seeds": len(seeds),
            "nodes_visited": len(visited_nodes),
            "expansions": len(expansions),
            "assertions_used": assertions_emitted,
            "used_in_edges": used_in_edges,
            "include_in": bool(include_in),
            "priority_tiers": {str(k): tier_counts[k] for k in sorted(tier_counts)},
            "priority_legend": {str(k): v for k, v in sorted(priority.items())},
            "truncated_reason": truncated_reason,
            "budget": {
                "max_depth": max_depth,
                "depth_used": depth,
                "max_nodes": max_nodes,
                "max_edges": max_edges,
                "max_degree_per_entity": max_degree,
                "max_assertions": max_assertions,
                "max_assertions_per_edge": max_per_edge,
                "timeout_ms": int(merged_budget["timeout_ms"]),
                "support_high": support_high,
            },
        }
        return {"expansions": expansions, "stats": stats}

    def _edge_anchors(self, edge: dict, remaining: int) -> tuple[list[dict], bool]:
        """取这条边可定位的观察锚点。`remaining` 是还能用几条断言。

        锚点必须来自 `assertions.jsonl`：有 `chunk_id` 与 `evidence_start/end`。
        没有断言数据时退回边自带的 `evidence` + `chunks[0]`，但 `start/end` 记 -1，
        让下游的偏移校验能明确拒绝——不假装知道位置。
        """
        if remaining <= 0:
            return [], False
        anchors: list[dict] = []
        for aid in list(edge.get("assertion_ids") or [])[:remaining]:
            rec = self.assertions_by_id.get(str(aid))
            if not rec:
                continue
            anchors.append(
                {
                    "assertion_id": str(aid),
                    "chunk_id": str(rec.get("chunk_id") or ""),
                    "evidence": str(rec.get("evidence_text") or ""),
                    "start": int(rec.get("evidence_start", -1)),
                    "end": int(rec.get("evidence_end", -1)),
                }
            )
        if anchors:
            return anchors, True
        cid = str((edge.get("chunks") or [""])[0])
        ev = str(edge.get("evidence") or "")
        if cid and ev:
            return (
                [{"assertion_id": "", "chunk_id": cid, "evidence": ev, "start": -1, "end": -1}],
                False,
            )
        return [], False

    # ------------------------------------------------------------ 上下文
    def build_context(
        self,
        hits,
        expansions,
        *,
        token_budget: int | None = None,
        query_analysis: dict | None = None,
    ) -> dict:
        """把检索命中与图扩展拼成上下文单元。

        规则（每条都对着一个真实失效模式）：

        * **同 chunk 只留一次**，`sources` 合并。同一句话被两条通道各带回一次，
          在上下文里出现两遍，会让人误以为有两个独立证据。
        * **相同证据的多条 assertion 合并**成一个 `evidence` 组，`assertion_ids` 累加。
        * **直接证据优先于图背景**：词法命中块排在纯图扩展块之前（`sources` 里有
          `bm25`/`entity` 即算直接证据）。
        * **截断不截引用锚点**：超预算时整单元丢弃，绝不切开某个单元的
          `evidence_spans`。切一半的引用比没有引用更坏——下游会以为自己有证据。
        * 每个单元都带 `chunk_id/doc_id/title/chapter/text`，因为引用校验要拿
          `text[start:end]` 逐字比对。

        返回 `{units:[...], context_digest, stats:{...}}`。
        """
        budget_tokens = int(
            token_budget if token_budget is not None else self.budget["max_context_tokens"]
        )
        qa = query_analysis or {}
        hint_predicates = {
            str(h.get("canonical") or h.get("raw") or "") for h in (qa.get("relation_hints") or [])
        }
        hint_predicates.discard("")
        focus_ids = {str(c.get("entity_id") or "") for c in (qa.get("entity_candidates") or [])}
        focus_ids.discard("")

        units: dict[str, dict] = {}
        skipped: list[dict] = []

        def ensure(chunk_id: str) -> dict | None:
            cid = str(chunk_id or "")
            if not cid:
                return None
            if cid in units:
                return units[cid]
            rec = self.chunks.get(cid)
            if not rec:
                skipped.append({"chunk_id": cid, "reason": "chunk_not_loaded"})
                return None
            unit = {
                "chunk_id": cid,
                "doc_id": str(rec.get("doc_id") or ""),
                "title": str(rec.get("title") or ""),
                "chapter": str(rec.get("chapter") or ""),
                "text": str(rec.get("text") or ""),
                "evidence_spans": [],
                "entities": [],
                "assertions": [],
                "score": 0.0,
                "sources": [],
                "graph_path": [],
            }
            units[cid] = unit
            return unit

        # 1) 检索命中（直接证据）
        for hit in hits or []:
            unit = ensure(hit.get("chunk_id"))
            if unit is None:
                continue
            unit["score"] = max(unit["score"], float(hit.get("score") or 0.0))
            for src in hit.get("sources") or []:
                if src not in unit["sources"]:
                    unit["sources"].append(dict(src))
            # 这块里与焦点实体/关系提示相关的断言，作为证据锚点
            for rec in self.assertions_by_chunk.get(unit["chunk_id"], []):
                if not _assertion_relevant(rec, focus_ids, hint_predicates):
                    continue
                _push_span(unit, rec)
                unit["assertions"].append(_assertion_brief(rec, self.display_name))
                for eid in (rec.get("subject_id"), rec.get("object_id")):
                    name = self.display_name(str(eid or ""))
                    if name and name not in unit["entities"]:
                        unit["entities"].append(name)

        # 2) 图扩展（背景证据）
        for exp in expansions or []:
            unit = ensure(exp.get("chunk_id"))
            if unit is None:
                continue
            if exp not in unit["graph_path"]:
                unit["graph_path"].append(exp)
            for anchor in exp.get("anchors") or []:
                rec = self.assertions_by_id.get(str(anchor.get("assertion_id") or ""))
                if rec is not None:
                    # 证据偏移只在自己的块正文里有效。一条边可能由多个块里的断言支撑，
                    # 把 anchors[1:] 的偏移登记到 anchors[0] 那块上，就等于报一个指向
                    # 别处的位置——下游 `text[start:end] == evidence` 必然对不上
                    # （真实语料实测：105 条引用里 24 条这样坏掉）。所以按锚点自己的
                    # chunk_id 找单元再登记。
                    host_id = str(rec.get("chunk_id") or "")
                    target = (ensure(host_id) if host_id else None) or unit
                    _push_span(target, rec)
                    brief = _assertion_brief(rec, self.display_name)
                    if brief not in target["assertions"]:
                        target["assertions"].append(brief)
                    for eid in (rec.get("subject_id"), rec.get("object_id")):
                        name = self.display_name(str(eid or ""))
                        if name and name not in target["entities"]:
                            target["entities"].append(name)
            for name in (exp.get("from_name"), exp.get("neighbor_name")):
                if name and name not in unit["entities"]:
                    unit["entities"].append(name)
            if not any(s["channel"] == SOURCE_GRAPH for s in unit["sources"]):
                unit["sources"].append({"channel": SOURCE_GRAPH, "score": 0.0})

        def direct(unit: dict) -> bool:
            return any(s["channel"] in (SOURCE_BM25, SOURCE_ENTITY) for s in unit["sources"])

        ordered = sorted(
            units.values(),
            key=lambda u: (0 if direct(u) else 1, -u["score"], u["chunk_id"]),
        )

        kept: list[dict] = []
        used_tokens = 0
        truncated: list[str] = []
        for unit in ordered:
            cost = estimate_tokens(
                f"{unit['title']} {unit['chapter']} {unit['text']}",
                tokenizer_version=getattr(self.index, "tokenizer_version", index_mod.TOKENIZER_VERSION),
                name_terms=getattr(self.index, "name_terms", None),
            )
            unit["tokens"] = cost
            if used_tokens + cost > budget_tokens:
                truncated.append(unit["chunk_id"])
                continue
            used_tokens += cost
            kept.append(unit)

        payload = [
            {
                "chunk_id": u["chunk_id"],
                "evidence": [s["evidence"] for s in u["evidence_spans"]],
                "assertion_ids": [a["assertion_id"] for a in u["assertions"]],
            }
            for u in kept
        ]
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

        return {
            "units": kept,
            "context_digest": digest,
            "stats": {
                "units": len(kept),
                "candidates": len(units),
                "tokens_used": used_tokens,
                "token_budget": budget_tokens,
                "truncated_chunks": truncated,
                "skipped_units": skipped,
                "direct_units": sum(1 for u in kept if direct(u)),
                "graph_only_units": sum(1 for u in kept if not direct(u)),
                "evidence_spans": sum(len(u["evidence_spans"]) for u in kept),
            },
        }


def _tier_reason(tier: int) -> str:
    return {
        1: "查询实体的直接出边（深度 1）",
        2: "谓词命中 relation_hints",
        3: "support_count >= support_high",
        4: "与已召回块共现",
        5: "证据带 scene_order 且查询含时间词（仅标记，不做时序推理）",
        99: "未命中任何优先级档，仅因预算允许而纳入",
    }.get(tier, "未知档")


def _assertion_relevant(rec: dict, focus_ids: set[str], hint_predicates: set[str]) -> bool:
    """这条断言是否与当前问句相关。

    判据只有两条，都可机检：端点落在焦点实体上，或谓词命中关系提示。
    刻意不按"描述像不像"判断——那会把判据变成主观。
    """
    if str(rec.get("subject_id") or "") in focus_ids:
        return True
    if str(rec.get("object_id") or "") in focus_ids:
        return True
    if hint_predicates and str(rec.get("predicate") or "") in hint_predicates:
        return True
    return False


def _push_span(unit: dict, rec: dict) -> None:
    """登记一条断言的证据区间。相同证据只留一次，偏移取首次出现的合法位置。"""
    ev = str(rec.get("evidence_text") or "")
    start = int(rec.get("evidence_start", -1))
    end = int(rec.get("evidence_end", -1))
    for span in unit["evidence_spans"]:
        if span["evidence"] == ev and span["start"] == start and span["end"] == end:
            return
    unit["evidence_spans"].append(
        {
            "assertion_id": str(rec.get("assertion_id") or ""),
            "evidence": ev,
            "start": start,
            "end": end,
        }
    )


def _assertion_brief(rec: dict, display) -> dict:
    return {
        "assertion_id": str(rec.get("assertion_id") or ""),
        "subject": display(str(rec.get("subject_id") or "")),
        "predicate": str(rec.get("predicate") or ""),
        "object": display(str(rec.get("object_id") or "")),
        "evidence": str(rec.get("evidence_text") or ""),
        "confidence": rec.get("confidence"),
        "chunk_id": str(rec.get("chunk_id") or ""),
    }


__all__ = [
    "DEFAULT_BUDGET",
    "ENTITY_CHANNEL_WEIGHT",
    "SOURCE_BM25",
    "SOURCE_ENTITY",
    "SOURCE_GRAPH",
    "Retriever",
    "estimate_tokens",
]
