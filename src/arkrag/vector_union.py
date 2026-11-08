"""向量通道的接线方式：**触发式尾部并集**，不是融合。

## 为什么不是融合（全部本机实测否证，300 题，top_k=20）

| 融合方式 | R@20 | MRR |
|---|---|---|
| base（无向量） | 0.7900 | 0.4388 |
| minmax 加权 w=0.25 / 0.5 / 1.0 | 0.7700 / 0.7500 / 0.7300 | 0.4396 / 0.4418 / 0.4080 |
| 名次倒数 RRF k=1 / k=2 | 0.7667 / 0.7667 | 0.3837 / 0.4061 |
| DBSF（均值±标准差归一） | 0.6900 | 0.3901 |
| DBSF 尾部补位 | 0.7733 | 0.4379 |
| 头部锁定 3 + 向量重排余下 | 0.6367 | 0.4154 |

唯一 MRR 正增益的 `gate_*`（`bn * (1 + β*vn)`）配对 bootstrap 4000 次：
β=1.0 时 MRR +0.0063、95%CI[−0.0097,+0.0233]、P(≤0)=0.2372；且 R@5 一律 −0.0100。
按项目判据（R@20 与 MRR 同时不劣 + 显著）一个都不能采纳。

结构原因：任何全局改动都会碰坏**已经答对的 237/300 道题**。而向量真正有用的
只是一个子集——base top-20 漏掉的 63 道里，向量 top-20 救回 14 道
（direct_relation 6、time_conflict 4、multihop 2、single_chunk 2）。

## 所以向量在这里的位置是「补漏」，不是「排序」

* 只在查询期信号判定「base 这次可能漏」时，才把向量 top-N **追加到 base 尾部**；
* 追加在尾部 ⇒ base 已命中的题 gold 名次只可能不变 ⇒ **R@20 / MRR 单调不降**；
* 代价只有延迟（一次 embed + 一次矩阵扫描），**没有质量回退风险**。

## 触发信号：`ratio_top_to_20 = 第 1 名分数 / 第 20 名分数`

分数在第 1 名与第 20 名之间几乎没有落差（ratio 接近 1）⇒ 排序没有把握 ⇒ 触发。
10 个查询期信号的 AUC（预测「base 会不会漏」，1=会漏；信号取负后 AUC>0.5 表示低值预示漏）：

| 信号 | AUC |
|---|---|
| **ratio_top_to_20** | **0.7605** |
| margin_top_to_20 | 0.7466 |
| margin_top_to_2 | 0.6681 |
| candidates | 0.5145 |
| bm25_terms | 0.5000 |
| top_score | 0.3945 |
| entity_top_score | 0.4401 |
| n_entity_cand | 0.3858 |
| second_score | 0.3430 |
| score20 | 0.2454 |

`DEFAULT_RATIO_CUT = 1.306328` 由 `scripts/calib_union_cut.py` **只在 dev 侧**
（gold-v2 dev 350）定出来：base 命中 273/350、向量可救回 15/350，规则
`max(救回 − 0.05 × 空转)` 选中它（触发 103/350、救回 15/15、空转 57）。
holdout 238 只报数不调参——本项目历史上因为「在评测集上选阈值」死过两次
（盲特征重排 train MRR +0.0161 / holdout −0.0091）。

## 为什么追加的块 `score` 记为 0.0

分值尺度不同（bm25 归一后与余弦不可比），混进 `score` 就是上面被否证的那种融合。
追加块只保证「在 hits 里存在」，靠列表位置表达优先级；`build_context` 的排序键
`(0 if direct else 1, -score, chunk_id)` 会把它们排在所有直证块之后——见
`score=0.0` 的注释。**它们进不进得了 context 由预算决定，这一点必须如实报。**
"""
from __future__ import annotations

from . import vector_store

#: 触发阈值：`top_score / score20 <= cut` 时认为排序没把握，叫向量来补
#:
#: 这个值由 `scripts/calib_union_cut.py` **只在 dev 上**定出来，holdout 只用来
#: 报告，不用来调参。dev（gold-v2，350 道带 gold）：base 命中 273/350、向量可
#: 救回 15/350；规则 `max(救回 − 0.05 × 空转)` 选中 **cut=1.306328**，触发
#: 103/350（29.4%）、救回 15/15、空转 57、score=12.15；且它正好是「能救回
#: 全部 15 道的最小触发阈值」，所以这个点没有可摘的余量。
#:
#: 上一版写的 1.377 是在 320 题 gold-v1 上目测的第 40 百分位，属于在评测集上
#: 挑阈值，已作废——本项目因同一形态死过一次（盲特征重排 train +0.0161 /
#: holdout −0.0091）。
DEFAULT_RATIO_CUT = 1.306328

#: 追加多少条向量结果（与 base 的 top_k 同量级；实测并集池 20→34.7）
DEFAULT_VECTOR_K = 20

#: 追加块在 hits 里的来源标记（与 retrieve.SOURCE_* 同一命名空间，但刻意
#: 不放进 `retrieve.SOURCE_*` 常量组：它不是检索器产出的分数，只是一个标记）
SOURCE_VECTOR_UNION = "vector_union"


def score_at(hits: list[dict], rank: int = 20) -> float:
    """取第 `rank` 名（1 起）的分；不足时退到末名。"""
    if not hits:
        return 0.0
    idx = min(len(hits), rank) - 1
    return float(hits[idx].get("score") or 0.0)


def ratio_top_to_20(hits: list[dict]) -> float:
    """第 1 名分 / 第 20 名分。第 20 名分为 0（分数已衰减到底）时返回 inf。"""
    if not hits:
        return float("inf")
    top = float(hits[0].get("score") or 0.0)
    s20 = score_at(hits, 20)
    if s20 <= 1e-9:
        return float("inf")
    return top / s20


def should_augment(hits: list[dict], *, cut: float = DEFAULT_RATIO_CUT) -> bool:
    """只做判定，不取向量——用于诊断脚本单独核对触发面。"""
    if not hits:
        return False
    return ratio_top_to_20(hits) <= cut


class TriggeredVectorUnion:
    """把向量通道作为 base 结果的**尾部补漏**接进检索链路。

    `embed_query(text) -> list[float]` 由调用方注入（生产是 Ollama
    `bge-m3`，测试是确定性假向量），这样本模块不依赖 HTTP，也不掩盖
    「embedding backend 不可用」这种硬错误——`EmbeddingError` 必须向上抛。
    """

    def __init__(self, vector_index, *, embed_query, cut: float = DEFAULT_RATIO_CUT,
                 vector_k: int = DEFAULT_VECTOR_K):
        if not callable(embed_query):
            raise TypeError("embed_query 必须是 callable(text) -> list[float]")
        if vector_k < 1:
            raise ValueError(f"vector_k 必须 >= 1，收到 {vector_k}")
        self.vector_index = vector_index
        self.embed_query = embed_query
        self.cut = float(cut)
        self.vector_k = int(vector_k)
        self.calls = 0
        self.triggered = 0
        self.added_total = 0

    # ---------------------------------------------------------------- 主入口
    def augment(self, hits: list[dict], query_text: str) -> dict:
        """返回新的 hits 列表与本次判定明细（**不就地改** 入参）。

        返回 `{"hits": [...], "used": bool, "ratio": float, "cut": float,
        "vec_hits": int, "added": [chunk_id...], "reason": str}`。
        """
        ratio = ratio_top_to_20(hits)
        out = {"hits": list(hits), "used": False, "ratio": ratio, "cut": self.cut,
               "vec_hits": 0, "added": [], "reason": ""}
        if not hits:
            out["reason"] = "no_base_hits"
            return out
        if ratio > self.cut:
            out["reason"] = "not_triggered"
            return out

        self.calls += 1
        qv = self.embed_query(query_text)
        vec_hits = self.vector_index.search(qv, top_k=self.vector_k)
        out["vec_hits"] = len(vec_hits)

        seen = {str(h.get("chunk_id") or "") for h in hits}
        merged = list(hits)
        for cid, _score in vec_hits:
            cid = str(cid)
            if not cid or cid in seen:
                continue
            seen.add(cid)
            merged.append({
                "chunk_id": cid,
                # 0.0 是刻意的：向量余弦与归一后的 bm25 不同量纲，混进 score
                # 就是已被否证的融合。追加块靠「在列表末尾」表达优先级。
                "score": 0.0,
                # 键必须叫 `channel`：`Retriever.build_context` 的 `direct()`
                # 读的是 `s["channel"]`。写成 `source` 会当场 KeyError——
                # 这是本项目 SOURCE_* 常量组之外的第二个命名，刻意沿用同一形状。
                "sources": [{"channel": SOURCE_VECTOR_UNION, "score": 0.0}],
            })
            out["added"].append(cid)

        self.triggered += 1
        self.added_total += len(out["added"])
        out["hits"] = merged
        out["used"] = True
        out["reason"] = "triggered"
        return out

    # ---------------------------------------------------------------- 自述
    def info(self) -> dict:
        return {
            "mode": "triggered_tail_union",
            "ratio_cut": self.cut,
            "vector_k": self.vector_k,
            "calls": self.calls,
            "triggered": self.triggered,
            "added_total": self.added_total,
            "vector_model": getattr(self.vector_index, "model", None),
            "vector_count": len(getattr(self.vector_index, "chunk_ids", []) or []),
            "vector_dim": getattr(self.vector_index, "dim", None),
        }


__all__ = [
    "DEFAULT_RATIO_CUT",
    "DEFAULT_VECTOR_K",
    "SOURCE_VECTOR_UNION",
    "TriggeredVectorUnion",
    "ratio_top_to_20",
    "score_at",
    "should_augment",
]
