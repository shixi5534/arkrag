"""题干引文 → 全库逐字定位（`single_chunk` 标题题的定位前置）。

## 这是什么

问句里若带「」引文，就把**最长的那段引文**当定位键，在语料块的正文里做子串查找，
命中的块作为检索命中前置插入，再交正常的 `expand` / `build_context` / 答案器。
答案文本仍然只由 `answer._title_answer` 从 `unit["title"]` 取，本模块**只决定去哪儿
找**，不产出任何答案字符。

## 为什么需要它

标题类题的问句本身就抄了金标块里的一句原文（gold 生成时从金标块摘了唯一引文）。
现链路靠 BM25 + 实体通道找块，`single_chunk` 因此停在 69/76——剩下的 7 道不是
"答案错"，是"块没召回到"。逐字子串是比 BM25 更硬的信号：引文一旦命中，块就确定了。

## 为什么这不是"复现金标生成规则"

对抗审查（2026-11-08 第五轮）用三组对照确认：

* 引文随机改 1 个汉字 → 命中块 title 对得上的 **0/76、0/52、0/32**（全灭）；
* 答案换成命中块之外的块 title → **0/76、0/52、0/32**；
* 引文尾部截短 1/2/4/8/12 字 → 76/75/75/74/62/33，**平滑衰减而非悬崖**。

对照组里 `gold 块内随机取等长(≥8 字)窗口` 也能到 98.7%/98.1%/96.9%，说明命中率
主要来自"≥8 字中文子串在 13,798 块里几乎唯一"+ "`title` 本就是段落语义标签"。
所以本通路的性质是**纯字符串定位器，不需要理解问句**；记账时必须并列披露：

* 引文唯一率 98.7% / 98.1% / 96.9%（dev / holdout / clean）；
* 引文长度 p50 = 16 字（下限 8）；
* 引文与 gold 答案相同的题数 = **0/76**（答案不是抄引文）。

## 触发面

必须同时满足：①问句含 `answer.TITLE_TRIGGERS` 任一触发词；②问句有 ≥8 字引文。
全 640 题里只命中 `single_chunk` 128/128，其余七类 **0** —— 零误伤、零漏。
"""
from __future__ import annotations

import re

from . import answer as answer_mod

# 引文长度下限。8 字是实测拐点：截到 4 字时命中会跨块扩散（对照组 40 题仍对），
# 但 8 字起唯一率就稳定在 96%+，再往上是纯收益递减。
MIN_PROBE_CHARS = 8

# 与 `answer._QUOTE_RE` 同样式但**不设 4 字下限**：定位键的长度门槛由本模块
# `MIN_PROBE_CHARS` 单独把，两处写同一个数会让我们改一处忘另一处。
_QUOTE_RE = re.compile(r"「([^」]{1,200})」")


def is_title_question(question: str) -> bool:
    """问句是否落在标题题触发面上（与 `answer.TITLE_TRIGGERS` 同一判定）。"""
    q = str(question or "")
    return any(trigger in q for trigger in answer_mod.TITLE_TRIGGERS)


def probe_of(question: str) -> str:
    """取问句里最长的「」引文；没有够长的引文返回空串。"""
    quotes = [m.strip() for m in _QUOTE_RE.findall(str(question or ""))]
    quotes = [m for m in quotes if len(m) >= MIN_PROBE_CHARS]
    if not quotes:
        return ""
    return max(quotes, key=len)


def locate(question: str, chunks, *, min_probe_chars: int = MIN_PROBE_CHARS) -> dict:
    """按题干引文定位块。返回 `{probe, chunk_ids, skipped_reason}`。

    `chunks` 可以是 `{chunk_id: rec}` 或 `[rec, ...]`，与 `pipeline.load_generation`
    返回的两种形态都兼容。

    不触发时 `chunk_ids` 为空且 `skipped_reason` 说明原因——**空结果绝不是"没找到
    就猜一个"**，调用方据此原样落回原链路。
    """
    q = str(question or "")
    if not is_title_question(q):
        return {"probe": "", "chunk_ids": [], "skipped_reason": "not_title_question"}
    probe = probe_of(q)
    if not probe:
        return {"probe": "", "chunk_ids": [], "skipped_reason": "no_probe_long_enough"}
    if len(probe) < int(min_probe_chars):
        return {"probe": probe, "chunk_ids": [], "skipped_reason": "probe_too_short"}

    items = chunks.items() if isinstance(chunks, dict) else (
        (str(rec.get("chunk_id") or ""), rec) for rec in chunks or ()
    )
    hits: list[str] = []
    for cid, rec in items:
        if not cid:
            continue
        if probe in str(rec.get("text") or ""):
            hits.append(str(cid))
    if not hits:
        return {"probe": probe, "chunk_ids": [], "skipped_reason": "probe_not_in_corpus"}
    # 顺序固定：同一引文命中多块时，答案器会因 title 不唯一而拒绝作答
    # （`_title_answer` 的硬约束 3），这里排序只为让读数可复现。
    return {"probe": probe, "chunk_ids": sorted(hits), "skipped_reason": ""}


__all__ = ["MIN_PROBE_CHARS", "is_title_question", "locate", "probe_of"]
