"""查询分析 —— 计划 G1「问句解析」与 G2「实体候选与关系提示」。

## 这一层只做一件事：把问句拆成**可被检索与扩展消费的结构**，不做判断

它不检索、不查图、不生成答案。理由：把"问了什么"和"去库里找什么"写在一起，
后续任何一个环节要改（换检索器、换扩展策略、加时间过滤）都得动同一个函数，
而这些改动彼此没有关系。

## 三条不许退让的规矩

1. **歧义必须原样保留。** 同一个别名被多个实体认领时，全部列出（各带
   `score` 与 `reason`），不许"取第一个"或"取提及次数最多的那个"。实测语料里
   「女妖」之于 Logos 与某个同名实体、「乌萨斯」之于「乌萨斯帝国」都真实存在；
   在查询分析阶段就强制唯一，等于把一个未经验证的合并结论当成前提带进下游。
2. **否定词与时间词不许丢。** 「玫瑰河畔不再参与」与「玫瑰河畔参与」在字符层面
   只差两个字，丢了否定就得到相反答案。它们进 `negated_hint` / `time_constraints`，
   并且**永不进停用词表**（见 `_STOPWORDS` 下方的运行时守卫）。
3. **丢弃要计数。** 每丢掉一个词都进 `stats["dropped_terms"]`。不记账的丢弃
   会让"为什么这条问句召回为 0"这种问题无法复盘。

## 为什么候选识别用 `index.NameTable` 而不是再写一套专名匹配

`NameTable` 已经在建库路径上被 13,781 个块各跑了一遍，它的行为（两字前缀求交 +
trie 最长匹配 + 位置优先不重叠）直接决定倒排里有哪些整词 token。查询侧若用另一套
匹配规则，就会出现"索引里有 `罗德岛制药公司` 这个 token，查询却拆成了两个名字"。
同一份专名表，建库与查询共用，才不会出现这种对不齐。

已知边界：`NameTable` 刻意跳过长度 1 的专名（单字已在单字 token 里）。所以单字
实体名不会被本函数识别成候选，只能靠词法召回命中。这是设计取舍，不是遗漏。
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

from . import index as index_mod

# 问句长度上限。超了就拒绝回答而不是截断后硬答：截断后的问句与用户问的不是一件事，
# 拿它检索再配一句"根据原文"是最坏的一种自信。
QUERY_MAX_CHARS = 512

# 否定标记。长的在前：先匹配「并非」再匹配「非」，避免把「并非」记成两个单字标记。
DEFAULT_NEGATION_MARKERS: tuple[str, ...] = (
    "并非", "不是", "不曾", "未曾", "并未", "没有",
    "不", "未", "没", "非", "无", "别", "勿", "莫",
)

# 时间标记。这些词改变答案的时间范围，必须原样带到下游。
DEFAULT_TIME_MARKERS: tuple[str, ...] = (
    "之前", "之后", "以前", "以后", "事先", "事后",
    "后来", "最初", "最终", "起初", "最后", "当时", "如今", "现在",
    "曾经", "随后", "接着", "先是", "然后", "此前", "此后",
)

# 出现这些词说明问的是关系/身份，光靠词法命中不够，要走图扩展。
DEFAULT_GRAPH_CUES: tuple[str, ...] = (
    "什么关系", "哪种关系", "之间", "相连", "关联", "中间实体", "谁是", "是谁",
    "身份", "属于谁", "效忠于", "隶属于", "类型是", "分别是什么",
)

# 检索用停用词。**只放对召回无贡献的结构词**，且下面有运行时守卫：
# 任何与否定/时间标记重叠的位置都不许被删。
_STOPWORDS: tuple[str, ...] = (
    "请问", "请", "问一下", "想问", "是否", "吗", "呢", "吧", "啊",
    "的", "了", "着", "过", "是", "和", "与", "跟", "在", "有",
    "什么", "怎么", "怎样", "哪", "哪个", "哪些", "谁", "为什么", "如何",
    "这个", "那个", "这些", "那些", "请问一下",
)

_WORDISH_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")


def _normalize(text: str) -> str:
    """全角折半角 + 折叠空白。不做小写化：中文无大小写，拉丁专名由 NameTable 处理。"""
    folded = unicodedata.normalize("NFKC", str(text or ""))
    return re.sub(r"\s+", " ", folded).strip()


def _alias_owner_map(entities, aliases) -> dict[str, list[dict]]:
    """`文本 -> [{entity_id, kind}]`，**保留一对多**。

    数据来源两处，优先级明确：

    * `entities` 是权威来源：每个实体的 `name` 与 `aliases`（`ark-rag-3` 的
      `to_dict()` 已经把待审/拒绝分到另外两张表，所以这里读到的就是可检索别名）。
    * `aliases`（图里的 `alias_index`）是补充：它是 `setdefault` 建的一对一表，
      会丢掉"同一文本被第二个实体认领"这件事——**所以不能用它当唯一来源**。
      凡是 `entities` 里已经有认领者的文本，一概以 `entities` 为准。
    """
    claims: dict[str, list[dict]] = defaultdict(list)
    for ent in entities or []:
        eid = str(ent.get("id") or ent.get("name") or "").strip()
        name = str(ent.get("name") or "").strip()
        if not eid or not name:
            continue
        claims[name].append({"entity_id": eid, "kind": "name"})
        for item in ent.get("aliases") or []:
            text = item.get("text") if isinstance(item, dict) else item
            text = str(text or "").strip()
            if text:
                claims[text].append({"entity_id": eid, "kind": "alias"})

    for text, eid in (aliases or {}).items():
        text = str(text or "").strip()
        eid = str(eid or "").strip()
        if not text or not eid:
            continue
        if text in claims:
            continue  # entities 已表态，不覆盖
        claims[text].append({"entity_id": eid, "kind": "alias"})

    # 去重：同一 (text, entity_id) 同时被 name 与 alias 认领时，保留 name
    out: dict[str, list[dict]] = {}
    for text, items in claims.items():
        best: dict[str, str] = {}
        for item in items:
            eid = item["entity_id"]
            if best.get(eid) != "name":
                best[eid] = item["kind"]
        out[text] = [
            {"entity_id": eid, "kind": kind} for eid, kind in sorted(best.items())
        ]
    return out


def _relation_lookup(vocab) -> tuple[dict[str, str], list[str]]:
    """`文本 -> 规范关系` 查表，以及被排除的单字同义词。

    排除单字同义词的理由不是洁癖：`RELATION_SYNONYMS` 里有「在」->位于、
    「任」->任职 这类单字键，它们会在「现在」「任何」里命中，把
    「现在凯尔希在哪」解析成「位于」。规范关系名本身最短两字，不受影响。
    """
    canonical = [str(r) for r in getattr(vocab, "CANONICAL_RELATIONS", ()) or ()]
    synonyms = dict(getattr(vocab, "RELATION_SYNONYMS", {}) or {})
    table: dict[str, str] = {}
    for name in canonical:
        table[name] = name
    dropped: list[str] = []
    for raw, canon in synonyms.items():
        raw = str(raw or "").strip()
        if not raw:
            continue
        if len(raw) < 2:
            dropped.append(raw)
            continue
        table.setdefault(raw, str(canon))
    return table, sorted(dropped)


def _find_terms(
    text: str,
    table: dict[str, str],
    *,
    blocked: list[tuple[int, int]] | None = None,
) -> tuple[list[dict], list[dict]]:
    """在 text 里找关系提示，长词优先、位置不重叠。返回 `(命中, 落在专名内的)`。

    专名内部的关系词不算提示：「罗德岛制药公司」含有「制造」这类字样时，问句
    并没有在问制造关系。挡下的照样回报，好在 stats 里看见。
    """
    if not table:
        return [], []
    taken = [False] * len(text)
    for span in blocked or ():
        for i in range(max(0, span[0]), min(len(text), span[1])):
            taken[i] = True
    keys = sorted(table, key=lambda k: (-len(k), k))
    hits: list[dict] = []
    skipped: list[dict] = []
    for key in keys:
        start = text.find(key)
        while start >= 0:
            end = start + len(key)
            if any(taken[start:end]):
                skipped.append(
                    {"raw": key, "canonical": table[key], "start": start, "end": end,
                     "reason": "inside_entity_or_overlap"}
                )
            else:
                for i in range(start, end):
                    taken[i] = True
                hits.append({"raw": key, "canonical": table[key], "start": start, "end": end})
            start = text.find(key, start + 1)
    hits.sort(key=lambda h: (h["start"], h["end"]))
    skipped.sort(key=lambda h: (h["start"], h["end"]))
    return hits, skipped


def _find_markers(
    text: str,
    markers,
    *,
    blocked: list[tuple[int, int]] | None = None,
) -> tuple[list[dict], list[dict]]:
    """找标记，长词优先、位置不重叠。返回 `(命中, 被专名挡下的)`。

    `blocked` 是**专名区间**。标记落在专名内部一律不算命中，理由是可机检的事实：
    真实语料里「巴别塔」含「别」、「乌萨斯」不含但「谢拉格」之类常有 —— 把专名
    内部的字当否定词，会把「玫瑰河畔是否参与了巴别塔相关的事务」判成否定句。
    被挡下的记进第二个返回值，好在 stats 里看见"挡了几次、挡的是谁"，
    而不是让它静默消失。
    """
    hits: list[dict] = []
    skipped: list[dict] = []
    taken = [False] * len(text)
    for spans in (blocked or ()):
        for i in range(max(0, spans[0]), min(len(text), spans[1])):
            taken[i] = True
    for marker in sorted({str(m) for m in markers or () if str(m)}, key=lambda k: (-len(k), k)):
        start = text.find(marker)
        while start >= 0:
            end = start + len(marker)
            if any(taken[start:end]):
                skipped.append({"marker": marker, "span": [start, end], "reason": "inside_entity"})
            else:
                for i in range(start, end):
                    taken[i] = True
                hits.append({"marker": marker, "span": [start, end]})
            start = text.find(marker, start + 1)
    hits.sort(key=lambda h: h["span"][0])
    skipped.sort(key=lambda h: h["span"][0])
    return hits, skipped


def _drop_stopwords(
    text: str,
    protected: list[tuple[int, int]],
) -> tuple[str, list[dict], list[dict]]:
    """删停用词，返回 `(检索用文本, 丢弃清单, 被守卫拦下的尝试)`。

    守卫是关键：与否定/时间标记或专名重叠的停用词**一律不删**。语料里「没有」含
    「有」、「不是」含「是」——按位置判重叠，不用字符串相等去猜。

    **丢弃的位置一律换成空格**，不是直接删掉。理由是可机检的：`凯尔希和罗德岛`
    删掉「和」会拼成 `凯尔希罗德岛`，分词时连出 `希罗` 这个假双字词，凭空给召回
    加了一个不存在的匹配项。留一个字符边界，跨词双字就断在这里。
    """
    if not text:
        return text, [], []
    taken = [False] * len(text)
    dropped: list[dict] = []
    blocked: list[dict] = []
    for word in sorted(_STOPWORDS, key=lambda w: (-len(w), w)):
        start = text.find(word)
        while start >= 0:
            end = start + len(word)
            inside = any(taken[start:end])
            overlaps_protected = any(
                start < p_end and p_start < end for p_start, p_end in protected
            )
            if overlaps_protected:
                blocked.append({"term": word, "span": [start, end]})
            elif not inside:
                for i in range(start, end):
                    taken[i] = True
                dropped.append({"term": word, "span": [start, end]})
            start = text.find(word, start + 1)
    chars: list[str] = []
    prev_dropped = False
    for i, ch in enumerate(text):
        if taken[i]:
            if not prev_dropped:
                chars.append(" ")
            prev_dropped = True
        else:
            chars.append(ch)
            prev_dropped = False
    kept = re.sub(r"\s+", " ", "".join(chars)).strip()
    return kept, dropped, blocked


def analyze_query(
    text: str,
    *,
    entities,
    aliases=None,
    vocab=None,
    max_chars: int = QUERY_MAX_CHARS,
) -> dict:
    """把问句解析成检索与扩展可直接消费的结构。

    参数
    ----
    text
        原始问句。全角字符会被折成半角，空白折叠。
    entities
        图里的实体列表（`graph["entities"]`）。**唯一权威的实体/别名来源**。
    aliases
        可选的 `{文本: entity_id}`（通常是 `graph["alias_index"]`）。只作补充，
        不会覆盖 `entities` 已经表态的文本——理由见 `_alias_owner_map`。
    vocab
        `arkrag.vocab` 或任何提供 `CANONICAL_RELATIONS` / `RELATION_SYNONYMS` 的
        对象。缺属性时按空处理，不抛错（便于测试只给最小对象）。
    max_chars
        超长阈值，默认 `QUERY_MAX_CHARS`。

    返回
    ----
    dict，字段见下方实现末尾的组装修正。`should_answer=False` 时 `reason` 说明原因，
    其余字段仍然给出（空/最小），这样调用方不必写两套分支。
    """
    original = str(text or "")
    normalized = _normalize(original)
    entities = list(entities or [])
    vocab = vocab if vocab is not None else _EMPTY_VOCAB

    negation_markers = tuple(getattr(vocab, "NEGATION_MARKERS", None) or DEFAULT_NEGATION_MARKERS)
    time_markers = tuple(getattr(vocab, "TIME_MARKERS", None) or DEFAULT_TIME_MARKERS)
    graph_cues = tuple(getattr(vocab, "GRAPH_CUES", None) or DEFAULT_GRAPH_CUES)

    # ---- 前置拒绝：空 / 纯标点 / 超长 -------------------------------------
    if not normalized:
        return _refusal(original, normalized, "empty_query")
    if not _WORDISH_RE.search(normalized):
        return _refusal(original, normalized, "no_word_characters")
    if len(normalized) > int(max_chars):
        return _refusal(
            original,
            normalized,
            f"query_too_long({len(normalized)}>{int(max_chars)})",
        )

    # ---- 实体候选：NameTable 单次扫描（先做，标记检测要拿它的区间做遮挡）----
    claims = _alias_owner_map(entities, aliases)
    table_terms = set(claims)
    name_table = index_mod.NameTable(table_terms) if table_terms else None
    spans = name_table.spans_in_chunk(normalized) if name_table else []
    entity_spans = [(s, e) for s, e in spans]

    # ---- 标记：否定 / 时间（专名内部的字不算标记）--------------------------
    negation_hits, negation_blocked = _find_markers(
        normalized, negation_markers, blocked=entity_spans
    )
    time_hits, time_blocked = _find_markers(
        normalized, time_markers, blocked=entity_spans
    )

    entity_candidates: list[dict] = []
    ambiguous: list[dict] = []
    for start, end in spans:
        matched = normalized[start:end]
        owners = claims.get(matched) or []
        if not owners:
            continue
        multi = len(owners) > 1
        if multi:
            ambiguous.append({"text": matched, "entity_ids": [o["entity_id"] for o in owners]})
        for owner in owners:
            is_name = owner["kind"] == "name"
            # 分数口径：整名命中 > 别名命中；长文本比短文本更可信（同一个 span 内
            # 命中的名字越长，被别的名字包含的可能性越低）。歧义不加分也不减分——
            # 歧义是事实，不是噪声。
            base = (1.0 if is_name else 0.85) + min(0.1, 0.01 * (len(matched) - 2))
            entity_candidates.append(
                {
                    "name": matched,
                    "entity_id": owner["entity_id"],
                    "score": round(min(1.0, base), 4),
                    "reason": "exact_name" if is_name else "alias",
                    "ambiguous": multi,
                    "span": [start, end],
                }
            )

    # 同一实体被多个匹配文本命中时保留最高分那条，但**不同实体一条都不删**
    best_by_id: dict[str, dict] = {}
    for cand in entity_candidates:
        prev = best_by_id.get(cand["entity_id"])
        if prev is None or cand["score"] > prev["score"]:
            best_by_id[cand["entity_id"]] = cand
    entity_candidates = sorted(
        best_by_id.values(), key=lambda c: (-c["score"], c["span"][0], c["entity_id"])
    )

    # ---- 关系提示 --------------------------------------------------------
    rel_table, dropped_synonyms = _relation_lookup(vocab)
    relation_hits, relation_skipped = _find_terms(normalized, rel_table, blocked=entity_spans)

    # ---- 停用词：带守卫地删 ----------------------------------------------
    protected = [(h["span"][0], h["span"][1]) for h in negation_hits + time_hits]
    protected += [(c["span"][0], c["span"][1]) for c in entity_candidates]
    # 关系提示也必须保护：「参与」中间夹着停用词「与」，删掉它会把关系词拆成
    # 「参」+「务」这种碎片，关系提示就等于丢了。这不是精度问题，是把一条
    # 能用于扩展检索的谓词凭空抹掉。
    protected += [(h["start"], h["end"]) for h in relation_hits]
    retrieval_text, dropped_terms, blocked_drops = _drop_stopwords(normalized, protected)

    requires_graph = bool(
        relation_hits
        or len(entity_candidates) >= 2
        or any(cue in normalized for cue in graph_cues)
    )

    return {
        "original_query": original,
        "normalized_query": normalized,
        "retrieval_query": retrieval_text or normalized,
        "entity_candidates": entity_candidates,
        "relation_hints": relation_hits,
        "time_constraints": time_hits,
        "negated_hint": bool(negation_hits),
        "negation_markers": [h["marker"] for h in negation_hits],
        "requires_graph": requires_graph,
        "should_answer": True,
        "reason": "",
        "stats": {
            "dropped_terms": dropped_terms,
            "blocked_drops": blocked_drops,
            "dropped_synonyms": dropped_synonyms,
            "negation_blocked": negation_blocked,
            "time_blocked": time_blocked,
            "relation_skipped": relation_skipped,
            "candidate_count": len(entity_candidates),
            "ambiguous_texts": ambiguous,
            "relation_hint_count": len(relation_hits),
            "time_marker_count": len(time_hits),
            "negation_marker_count": len(negation_hits),
            "name_table_terms": len(table_terms),
            "query_chars": len(normalized),
        },
    }


class _EmptyVocab:
    """最小 vocab 替身：全部按空处理。只用于测试与"没传 vocab"的兜底。"""

    CANONICAL_RELATIONS: tuple[str, ...] = ()
    RELATION_SYNONYMS: dict[str, str] = {}


_EMPTY_VOCAB = _EmptyVocab()


def _refusal(original: str, normalized: str, reason: str) -> dict:
    return {
        "original_query": original,
        "normalized_query": normalized,
        "retrieval_query": "",
        "entity_candidates": [],
        "relation_hints": [],
        "time_constraints": [],
        "negated_hint": False,
        "negation_markers": [],
        "requires_graph": False,
        "should_answer": False,
        "reason": reason,
        "stats": {
            "dropped_terms": [],
            "blocked_drops": [],
            "dropped_synonyms": [],
            "candidate_count": 0,
            "ambiguous_texts": [],
            "relation_hint_count": 0,
            "time_marker_count": 0,
            "negation_marker_count": 0,
            "name_table_terms": 0,
            "query_chars": len(normalized),
        },
    }


__all__ = [
    "QUERY_MAX_CHARS",
    "DEFAULT_NEGATION_MARKERS",
    "DEFAULT_TIME_MARKERS",
    "DEFAULT_GRAPH_CUES",
    "analyze_query",
]
