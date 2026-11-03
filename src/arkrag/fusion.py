"""RRF 三通道融合：BM25 + 向量 + 实体精确匹配。

## 为什么用 RRF 而不是加权求和

现在是 `final = bm25_norm + 0.5 * entity_norm`（`retrieve.ENTITY_CHANNEL_WEIGHT`）。
那套权重是**人工调出来的数字**：它假定两路分数的"尺度"可以对齐，而 BM25 是
无界正数、向量余弦在 [-1,1]、实体分数是乘过候选权重的归一分。三者量纲本来就不同，
归一到 [0,1] 再相加只是把量纲问题藏起来——换一个语料，最优权重就变了。

RRF（Reciprocal Rank Fusion）只用**名次**：

    score(d) = Σ_channel 1 / (k + rank_channel(d))

名次是无量纲的，所以三条通道能直接相加，不需要任何调参。代价是丢掉了分数的
"幅度"信息（第 1 名比第 2 名好多少）。计划 I1 原文就要求先用 RRF、
"不要一开始叠加复杂加权评分"——这个取舍是有意的，先用最不容易搞错的那个。

`RRF_K = 60` 是原论文（Cormack et al. 2009）的经验值，也是 Elasticsearch /
Weaviate 的默认值。它的作用是压平头部：k 越大，第 1 名与第 10 名的差距越小。
60 的效果是「进了某个通道的前几名都算数，但单靠一个通道当第一不足以霸榜」。

## 三条通道的定义（写清是为了让分数可复算）

* `bm25`：`index.search(retrieval_query, top_k=channel_k)` 的名次。
* `vector`：`vector_index.search(embed(query), top_k=channel_k)` 的名次。
* `entity`：**实体名在块正文里字面出现**的块。刻意用字面匹配而不是再跑一遍 BM25：
  BM25 已经覆盖了"词命中"这件事，如果实体通道也用 BM25，它就只是 bm25 的
  同分布复制品，融合后几乎不会带来新候选——两条几乎相同的通道做 RRF，
  等于给同一条证据投两票，把真正的多样性挤掉。

## 关于"没有向量通道时"

`vector_index=None` 时本模块仍可工作，只是少一路。**但调用方必须显式知道这件事**：
`stats` 里会写 `channels_used`，缺的那一路写 `vector: "absent"` 及原因。
理由与 FTS 后端、embedding 客户端一致——静默少一路，召回会掉一截而报告全绿。
"""

from __future__ import annotations

import time

# 原论文经验值，也是主流实现默认值。见模块 docstring。
RRF_K = 60

# 每条通道取多少候选参与融合。太大则 RRF 的 k 值失去意义（尾部名次几乎不贡献分数），
# 太小则融合无从发挥。取 100 与 `retrieve.search(top_k*3)` 在 top_k=20 时的量级相当。
DEFAULT_CHANNEL_K = 100

SOURCE_VECTOR = "vector"


def rrf_fuse(
    channel_ranks: dict[str, list[str]],
    *,
    k: int = RRF_K,
    top_k: int = 20,
) -> list[dict]:
    """把「通道 → 已排序 chunk_id 列表」融合成一条排序。

    `channel_ranks` 里每个列表必须是**已按该通道自身好坏排序**的候选（最优在前）。
    缺席的通道直接不传（或传空列表），不给它零票惩罚——缺席不等于不相关。

    返回 `[{"chunk_id", "score", "channels": {name: {rank, contrib}}}]`，降序。
    同分按 chunk_id 升序，与 `index.search` / `vector_store.search` 的 tie-break 一致，
    否则融合结果在真实同分场景下不可复现。
    """
    if k <= 0:
        raise ValueError(f"RRF k 必须为正（收到 {k}）")
    scores: dict[str, float] = {}
    detail: dict[str, dict] = {}
    for channel, ranked in channel_ranks.items():
        if not ranked:
            continue
        # 同一通道里重复出现的 chunk_id 只记首次名次：通道输出了重复项说明上游有 bug，
        # 但融合不该因此把它算两票（那就是奖励重复）。
        seen: set[str] = set()
        for pos, cid in enumerate(ranked):
            cid = str(cid)
            if cid in seen:
                continue
            seen.add(cid)
            # rank 从 1 开始（公式约定）。pos 从 0 开始，故 +1。
            rank = pos + 1
            contrib = 1.0 / (k + rank)
            scores[cid] = scores.get(cid, 0.0) + contrib
            detail.setdefault(cid, {})[channel] = {
                "rank": rank,
                "contrib": round(contrib, 8),
            }
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    out = []
    for cid, score in ordered[: max(1, int(top_k))]:
        out.append(
            {
                "chunk_id": cid,
                "score": round(score, 8),
                "channels": dict(sorted(detail[cid].items())),
            }
        )
    return out


class FusionRetriever:
    """三通道 RRF 检索。

    与 `retrieve.Retriever` 分开成独立类，而不是往 `Retriever.search` 里加分支：
    `Retriever.search` 的两路加权是 G 阶段评测过的基线，**召回数字必须可复现**。
    把融合塞进同一个方法，任何一次改动都可能悄悄改变旧路径的排序，而
    `bench`/`evaluate` 的对比就失去意义。

    代价是两边都要持有 chunks/图，重复一份内存。这里刻意只持**引用**
    （chunks dict 与 entity 表由调用方传入，不复制），所以不额外占内存。
    """

    def __init__(
        self,
        index,
        vector_index,
        *,
        chunks: dict[str, dict],
        entities_by_id: dict[str, dict] | None = None,
        embed_query=None,
        channel_k: int = DEFAULT_CHANNEL_K,
        rrf_k: int = RRF_K,
        synonyms: dict[str, list[str]] | None = None,
    ) -> None:
        self.index = index
        self.vector_index = vector_index
        self.chunks = chunks
        self.entities_by_id = entities_by_id or {}
        self.channel_k = max(1, int(channel_k))
        self.rrf_k = int(rrf_k)
        self._embed_query = embed_query
        self._name_terms: dict[str, list[str]] = {}
        for eid, ent in self.entities_by_id.items():
            terms = [str(ent.get("name") or "")]
            for item in ent.get("aliases") or []:
                t = item.get("text") if isinstance(item, dict) else item
                if str(t or "").strip():
                    terms.append(str(t).strip())
            self._name_terms[eid] = [t for t in dict.fromkeys(terms) if t]
        self._synonyms = dict(synonyms or {})
        self._haystack: dict[str, str] | None = None

    # ------------------------------------------------------------ 通道
    def _haystacks(self) -> dict[str, str]:
        """`{chunk_id: 可匹配文本}`，惰性建一次。

        必须复用 `vector_store.chunk_text_for_embedding`——那是「一个块被表示成
        什么文本」的唯一实现。若这里另拼一份，字面匹配看的文本与向量/BM25 看的
        文本就可能不一致，而那种偏差不会有任何报错。

        惰性建 + 缓存：13,781 块 × 约 900 字 ≈ 12 MB 字符串。每问重建一遍是
        13,781 次 f-string 拼接，实测比缓存慢两个数量级（H1 剖析里同类教训）。
        """
        if self._haystack is None:
            from .vector_store import chunk_text_for_embedding

            self._haystack = {
                cid: chunk_text_for_embedding(rec) for cid, rec in self.chunks.items()
            }
        return self._haystack

    def _chunk_contains(self, needle: str) -> list[str]:
        """字面匹配：文本里出现 needle 的块，按 chunk_id 升序。

        用 `chunk_order`（slot 序 = chunk_id 升序）而不是 dict 迭代序，
        保证结果稳定可复现。
        """
        if not needle:
            return []
        hay = self._haystacks()
        hits = []
        for cid in self.chunk_order:
            text = hay.get(cid)
            if text is not None and needle in text:
                hits.append(cid)
        return hits

    @property
    def chunk_order(self) -> list[str]:
        """稳定的块遍历序。由 `doc_id` 的 slot 序给出，与词法索引一致。"""
        if not hasattr(self, "_chunk_order"):
            doc_id = getattr(self.index, "doc_id", None) or {}
            if doc_id:
                self._chunk_order = [str(doc_id[s]) for s in sorted(doc_id)]
            else:
                self._chunk_order = sorted(self.chunks)
        return self._chunk_order

    def channels(self, query_analysis: dict, *, text: str) -> dict[str, list[str]]:
        """按优先级产出三条通道的候选名次。缺席的通道不进 dict。"""
        out: dict[str, list[str]] = {}
        if text:
            out["bm25"] = [cid for cid, _ in self.index.search(text, top_k=self.channel_k)]
        if self.vector_index is not None and self._embed_query is not None and text:
            qv = self._embed_query(text)
            out[SOURCE_VECTOR] = [
                cid for cid, _ in self.vector_index.search(qv, top_k=self.channel_k)
            ]
        named = self._entity_exact(query_analysis)
        if named:
            out["entity"] = named
        return out

    def _entity_exact(self, query_analysis: dict) -> list[str]:
        """问句里出现的实体，其规范名/别名在正文里字面出现的块。

        排序规则（决定名次，因而决定 RRF 贡献）：
        1. 命中实体越多越好——块同时含问句里的两个人名，比只含一个有说服力；
        2. 问句对该实体的候选分越高越好；
        3. chunk_id 升序。
        """
        score: dict[str, float] = {}
        for cand in query_analysis.get("entity_candidates") or []:
            eid = str(cand.get("entity_id") or "")
            weight = float(cand.get("score") or 0.0)
            terms = self._name_terms.get(eid) or []
            if not terms:
                name = str(cand.get("name") or "")
                terms = [name] if name else []
            # 同一实体的多个别名命中同一块只算一次，否则别名多的实体白拿分
            hit: set[str] = set()
            for term in terms:
                for cid in self._chunk_contains(term):
                    hit.add(cid)
            for cid in hit:
                score[cid] = score.get(cid, 0.0) + max(weight, 1e-6)
        return [cid for cid, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))][
            : self.channel_k
        ]

    # ------------------------------------------------------------ 检索
    def search(self, query_analysis: dict, *, top_k: int = 20) -> dict:
        """融合检索。返回结构与 `Retriever.search` 同形，另加 `channels` 明细。"""
        top_k = max(1, int(top_k))
        if not query_analysis.get("should_answer", True):
            return {
                "hits": [],
                "stats": {
                    "skipped_reason": str(query_analysis.get("reason") or "not_answerable"),
                    "channels_used": {},
                    "candidates": 0,
                    "top_k": top_k,
                    "fusion": "rrf",
                },
            }
        text = str(
            query_analysis.get("retrieval_query")
            or query_analysis.get("normalized_query")
            or ""
        )
        t0 = time.perf_counter()
        chans = self.channels(query_analysis, text=text)
        t_chan = (time.perf_counter() - t0) * 1000.0
        fused = rrf_fuse(chans, k=self.rrf_k, top_k=top_k)

        hits = []
        for item in fused:
            hits.append(
                {
                    "chunk_id": item["chunk_id"],
                    "score": item["score"],
                    "sources": [
                        {"channel": name, "score": info["contrib"]}
                        for name, info in sorted(item["channels"].items())
                    ],
                    "channels": item["channels"],
                }
            )
        used = {name: len(ranked) for name, ranked in chans.items()}
        if self.vector_index is None:
            used["vector"] = "absent_no_vector_index"
        elif self._embed_query is None:
            used["vector"] = "absent_no_embed_query"
        return {
            "hits": hits,
            "stats": {
                "channels_used": used,
                "candidates": len({c for ranked in chans.values() for c in ranked}),
                "top_score": hits[0]["score"] if hits else 0.0,
                "top_k": top_k,
                "fusion": "rrf",
                "rrf_k": self.rrf_k,
                "channel_k": self.channel_k,
                "channels_ms": round(t_chan, 3),
                "skipped_reason": "",
                "queried_text": text,
            },
        }


def overlap(ranked_a: list[str], ranked_b: list[str], *, top: int = 20) -> dict:
    """两组排序前 top 的重合度。评测融合增益时用：只看「新通道带来了多少新候选」。"""
    a = list(ranked_a[:top])
    b = list(ranked_b[:top])
    sa, sb = set(a), set(b)
    return {
        "top": top,
        "common": len(sa & sb),
        "only_a": len(sa - sb),
        "only_b": len(sb - sa),
        "jaccard": round(len(sa & sb) / len(sa | sb), 4) if (sa | sb) else 0.0,
    }
