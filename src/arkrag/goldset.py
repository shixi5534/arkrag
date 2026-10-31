"""F3 Gold Set 第一版：从**已构建的图与语料**里机械挖掘评测题。

本模块只做一件事：把「图上已经存在、且能在语料正文里逐字定位」的事实，
按固定模板转成评测题，并给出可机检的校验函数。

## 为什么机械构造不等于「自证」

被评测系统若自己写标准答案，评测就退化成自证。所以这里的每一条题都锚在
**两个系统之外的独立对象**上：

1. 题目所问的事实必须写在 `gold_chunk_ids` 指向的语料块正文里，且
   `required_citations[].evidence` 必须是该块 `text` 的**逐字子串**
   （`str.find` 能定位，见 `validate_gold_item`）。语料块是抽取之前的输入，
   不是抽取的输出，改判据不会改写它。
2. 训练/评测用的答案字段（`answer`/`acceptable_answers`）刻意只用**语料侧或
   词表侧**能核对的字符串：块标题、章节名、词表里的规范谓词、实体规范名。
   没有任何一条题把「图的内部判断」当答案。
3. `should_abstain=True` 的题反过来用图做**否定证据**：要求两个实体都在图里、
   且图里确实不存在该谓词的任何边。这类题的「标准答案」是「拒答」，靠图证明
   「真没有」，而不是靠图声称「有什么」。

机检不能证明题目问得自然；它能证明的是「证据真实存在且题面所问的事实确实写在
那里」。这是本文件愿意担保的范围，超出范围的部分写进 `docs/goldset-v1说明.md`。

## 确定性

所有挖掘函数都接受 `seed`，内部用 `random.Random(seed)` 采样，不读时间、
不读环境变量。同样的输入 + 同样的 seed 必然产出同样的题（顺序也一致）。
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import atomic, config, vocab

# ---------------------------------------------------------------- 常量

#: 八类题目的类别名。顺序即报告顺序。
CATEGORIES: tuple[str, ...] = (
    "single_chunk",
    "alias",
    "direct_relation",
    "multihop",
    "cross_chunk",
    "time_conflict",
    "negative_abstain",
    "adversarial",
)

#: `question_id` 里用的类别缩写，必须全小写、互不相同、长度 ≤ 3。
CATEGORY_ABBR: dict[str, str] = {
    "single_chunk": "sc",
    "alias": "al",
    "direct_relation": "dr",
    "multihop": "mh",
    "cross_chunk": "cx",
    "time_conflict": "tc",
    "negative_abstain": "na",
    "adversarial": "ad",
}

#: 八类占比（计划 F3 原文）。合计 1.0；`build_gold_set` 用整数向下取整后再补齐余数。
CATEGORY_SHARE: dict[str, float] = {
    "single_chunk": 0.20,
    "alias": 0.15,
    "direct_relation": 0.15,
    "multihop": 0.15,
    "cross_chunk": 0.10,
    "time_conflict": 0.10,
    "negative_abstain": 0.10,
    "adversarial": 0.05,
}

#: 九个必填字段（计划 F3 原文「每条至少含」）。
REQUIRED_FIELDS: tuple[str, ...] = (
    "question_id",
    "question",
    "answer",
    "acceptable_answers",
    "gold_chunk_ids",
    "gold_assertion_ids",
    "required_citations",
    "difficulty",
    "category",
    "should_abstain",
)

QUESTION_ID_RE = re.compile(r"^q-(sc|al|dr|mh|cx|tc|na|ad)-\d{6}$")

#: 正文里的否定词（计划里的四个 + 两个同义形态，写在这里便于复核）。
NEGATION_WORDS: tuple[str, ...] = ("没有", "不会", "并未", "拒绝", "不曾", "不再")

# 关系词后面跟这些字时它是**名词的一部分**，不是谓词：
# `感染者`（感染）、`矿石病`（矿石/病）里的 `感染` 若当谓词处理，会产出
# 「源石虫不会感染矿石病」→ 问「源石虫感染矿石病了吗」这类题，而原文里
# 那两个词根本不是「感染」这个动作。
_NOUN_TAIL: frozenset[str] = frozenset("者病人们性物晶化")

# 实体名与否定词之间允许出现的字。实测中文里能落在这段空档的只有连词/副词/
# 时间副词；出现别的字（「他们」「既没有」之类）说明真正的主语不是这个名字。
_GLUE_CHARS: frozenset[str] = frozenset(
    "，、,；; 　并也都再曾就还又正是只从来没有至今几乎后期经已在的"
)

# 名字与否定词之间允许的敬称（「埃琳娜小姐并未离开」）。
_HONORIFICS: frozenset[str] = frozenset(
    ("小姐", "先生", "女士", "夫人", "大人", "殿下", "阁下")
)

# 单独出现时表示「并列否定」，此时主语不是前一个名词：
# 「他们既没有通讯设备也没有人指挥」问「通讯设备指挥了吗」是错题。
_PARALLEL_NEG: frozenset[str] = frozenset(("也", "又", "再", "还"))

#: 句中出现这些词就是疑问/反问句，此时「没有 X」不是在陈述事实。
#: 实测样本：`为什么我们可敬的法夫公爵没有感染，为什么…陛下没有感染`。
_INTERROGATIVE: tuple[str, ...] = ("为什么", "难道", "是否", "吗", "？", "?", "呢")

#: 关系词后面紧跟这些词说明它是时间从句的一部分（`在前往北方冰原之后`），
#: 不是谓词：`您的命运，并没有在前往北方冰原之后变得一片漆黑` 里「前往」是时间状语。
_TEMPORAL_TAIL: tuple[str, ...] = (
    "之后",
    "之前",
    "之时",
    "的时候",
    "时期",
    "期间",
    "过程中",
)

#: 句读切分符。用于把正文切成「句子」以便唯一定位证据。
_SENTENCE_SPLIT = re.compile(r"[。！？!?\n]")

#: 跨块跨章题允许列出的最大章节数。
#: 题目的答案要枚举**全部**出现章节，而每一章都必须有引用块支撑（否则解题者读完
#: 给出的块也推不出被列进去的章节 = 「答案对但引用没覆盖到」）。实测该章数分布为
#: `{2:720, 3:316, 4:192, 5:98, 6:77, 7:45, 8:48, 9:28, 10:33, 11:32, 12+:137}`，
#: 取 4 既保证引用能完整覆盖、又只放弃长尾。超出的一律不出题（如实少给）。
_CROSS_CHUNK_MAX_CHAPTERS = 4

#: 每条题里的 `difficulty` 取值。
DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")

#: 单块事实题可用的称谓模板。键是难度，值是带两个占位符的模板。
#: `{evidence}` 原文、`{title}` 块标题（来自语料，不是来自图）。
TEMPLATES: dict[str, tuple[str, ...]] = {
    "single_chunk": (
        "在剧情《{title}》的正文里，出现了这样一句话：「{evidence}」。这句话出自哪个文本块？请给出该块的标题。",
        "下面这段原文来自哪一段剧情？「{evidence}」请回答该段的标题。",
    ),
    "alias": (
        "在剧情正文里，「{alias}」被用来指代哪个角色或事物？请给出它的规范名。",
        "「{alias}」是《{title}》中出现的称呼，它的规范名（规范实体名）是什么？",
    ),
    "direct_relation": (
        "根据剧情，「{subject}」与「{object}」之间的关系是什么？请用词表里的一个关系词回答。",
        "「{subject}」对「{object}」处于哪种关系？",
    ),
    "multihop": (
        "已知「{a}」通过唯一一个中间实体与「{c}」相连（中间实体经由两条独立关系相连）。这个中间实体是谁？",
        "从「{a}」出发，经过一个中间实体可以到达「{c}」。这个中间实体叫什么？",
    ),
    "cross_chunk": (
        "「{name}」在哪些章节的正文里出现过？请列出全部章节代号。",
        "「{name}」这个实体横跨了多个章节。请给出它出现过的章节代号。",
    ),
    "time_conflict": (
        "「{a}」与「{b}」在剧情不同阶段分别被描述为「{r1}」和「{r2}」。这两条记录应如何理解？",
        "同两个实体「{a}」与「{b}」之间先后出现「{r1}」与「{r2}」两种相反关系。这是抽取矛盾还是剧情演进？请说明理由。",
    ),
    "negative_abstain": (
        "据原文「{sentence}」，「{entity}」是否{rel_phrase}？",
        "原文写道「{sentence}」。据此判断：「{entity}」{rel_phrase}这件事是否发生？",
    ),
    "negative_abstain_absent": (
        "「{a}」与「{b}」之间是否存在「{label}」关系？",
        "在图与语料里，「{a}」和「{b}」是否有「{label}」这层关系？",
    ),
    "adversarial": (
        "「{a}」与「{b}」名字相近，它们是同一个实体吗？若不是，请说明区别。",
        "有两个名字相近的实体：「{a}」与「{b}」。它们指同一个东西吗？请给出判断依据。",
    ),
}

#: 不可回答题里用的「词表不存在」的关系标签。
#: 每个都要过 `_absent_labels` 的机检（不在词表、且图里没有任何边用这个谓词）。
ABSENT_LABELS: tuple[str, ...] = (
    "婚约",
    "祖孙",
    "债主",
    "邻居",
    "寄养",
    "同乡",
    "笔友",
    "代言",
    "代购",
    "陪练",
)

#: 题面里描述否定断言时用的谓词短语补全（避免「是否交谈」这种生硬说法）。
_REL_PHRASE: dict[str, str] = {
    "治愈": "被治愈",
    "感染": "被感染",
    "救助": "被救助",
    "保护": "受到保护",
    "参与": "参与其中",
    "使用": "使用它",
    "交谈": "与对方交谈",
    "背叛": "背叛对方",
    "冲突": "与对方冲突",
    "离开": "离开",
    "前往": "前往",
    "持有": "持有它",
    "雇用": "被雇用",
    "隶属": "隶属于对方",
    "同盟": "与对方结盟",
    "敌对": "与对方敌对",
    "交战": "与对方交战",
}

#: 默认输出路径（**不在** `data/arkrag/` 之下：那是冻结产物区）。
DEFAULT_GOLD_PATH = "data/goldset/gold-v1.json"


# ---------------------------------------------------------------- 小工具


def _as_chunk_map(chunks: Any) -> dict[str, dict]:
    """把 `chunks` 归一成 `{chunk_id: 块}`。

    挖掘函数既接受 `list[dict]`（从 jsonl 读出来的形态），也接受已经是 dict 的
    查表结构——调用方常常已经建过一次索引，不该逼它重建。
    """
    if isinstance(chunks, dict):
        return chunks
    out: dict[str, dict] = {}
    for c in chunks or ():
        cid = c.get("chunk_id")
        if cid:
            out[cid] = c
    return out


def _as_assertion_list(assertions: Any) -> list[dict]:
    """把 `assertions` 归一成 list。dict 形态按 assertion_id 排序后取值，保证确定。"""
    if isinstance(assertions, dict):
        return [assertions[k] for k in sorted(assertions)]
    return list(assertions or ())


def _as_entity_list(entities: Any) -> list[dict]:
    """把 `entities` 归一成 list。"""
    if isinstance(entities, dict):
        return [entities[k] for k in sorted(entities)]
    return list(entities or ())


def _as_edge_list(edges: Any) -> list[dict]:
    """把 `edges` 归一成 list。"""
    if isinstance(edges, dict):
        return [edges[k] for k in sorted(edges)]
    return list(edges or ())


def _rng(seed: Any) -> random.Random:
    """固定 seed 的随机源。seed 为 None 时用 0，绝不用系统时间。"""
    return random.Random(0 if seed is None else seed)


def _stable_index(key: str, n: int) -> int:
    """把任意字符串稳定地映射到 ``[0, n)``。

    刻意不用内置 ``hash()``：Python 对 str 的哈希带每进程随机盐
    （PYTHONHASHSEED），同一条题在两个进程里会选到不同模板，
    「同 seed 同输出」的承诺就失效了。这里取 sha256 前缀，跨进程稳定。
    """
    if n <= 1:
        return 0
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % n


def _pick(rng: random.Random, pool: Sequence[dict], limit: int) -> list[dict]:
    """从 pool 里确定性抽样至多 `limit` 条。

    pool 先在调用方按稳定键排好序，因此 `sample` 的结果只取决于 seed 与 pool 内容。
    """
    if limit <= 0 or not pool:
        return []
    if len(pool) <= limit:
        return list(pool)
    return rng.sample(list(pool), limit)


def _entity_name(e: dict | None) -> str:
    """实体展示名。"""
    return (e or {}).get("name") or ""


def _authoritative_aliases(e: dict) -> list[str]:
    """只返回**权威**别名文本。

    权威 = `aliases` 里 `review ∈ {auto, approved}`。`pending_aliases` 与
    `rejected_aliases` 一律不进这个列表——待审别名本来就是「不确定是不是它」，
    拿它出题等于把未审判断写成标准答案。
    """
    out: list[str] = []
    for a in e.get("aliases") or ():
        text = a.get("text") if isinstance(a, dict) else a
        review = a.get("review") if isinstance(a, dict) else "auto"
        if not text:
            continue
        if review in ("auto", "approved"):
            out.append(text)
    return out


def _sentence_at(text: str, start: int, end: int) -> str:
    """取 `text[start:end]` 所在的那一句（按句读符切开，含两端内容）。

    `start`/`end` 落在同一句内时返回整句；找不到时退化为 `text[start:end]`。
    返回值一定是 `text` 的子串，因此仍能逐字定位。
    """
    if not text:
        return ""
    seg_start = 0
    for m in _SENTENCE_SPLIT.finditer(text):
        seg_end = m.start()
        if seg_start <= start and end <= seg_end:
            return text[seg_start:seg_end] or text[start:end]
        seg_start = m.end()
    if seg_start <= start:
        return text[seg_start:] or text[start:end]
    return text[start:end]


def _window_text(text: str, needle: str, *, before: int = 12, after: int = 20) -> str:
    """取 `needle` 第一次出现位置前后的一段原文（**整句截取**，保证是子串）。"""
    i = text.find(needle)
    if i < 0:
        return ""
    return _sentence_at(text, i, i + len(needle)) or text[
        max(0, i - before) : min(len(text), i + len(needle) + after)
    ]


def _unique_hit(chunk: dict | None, evidence: str) -> bool:
    """`evidence` 是否恰好在块正文里出现一次（唯一定位）。"""
    if not chunk or not evidence:
        return False
    return chunk.get("text", "").count(evidence) == 1


def _in_text(chunk: dict | None, needle: str) -> bool:
    """`needle` 是否出现在块正文里（不限次数）。"""
    return bool(chunk and needle and needle in chunk.get("text", ""))


def _is_generic(name: str) -> bool:
    """这个实体名是不是泛称（泛称不当题面主体）。"""
    from . import identity  # 延迟导入：identity 顶层 import 本模块不成立，但保守起见

    return identity.is_generic_anchor(name)


def _absent_labels(edges: Sequence[dict]) -> tuple[str, ...]:
    """筛出「词表里没有、图上也没有」的关系标签。

    `ABSENT_LABELS` 是手写的候选池，这里再机检一遍：只要图上真有一条边用了这个
    谓词、或它其实在 `CANONICAL_RELATIONS` 里，就不能用来构造「不可回答」题——
    否则那条题可能是有答案的。
    """
    used = {(e.get("predicate") or "") for e in edges}
    canon = set(vocab.CANONICAL_RELATIONS)
    return tuple(lab for lab in ABSENT_LABELS if lab not in canon and lab not in used)


def _edge_pair_set(edges: Sequence[dict]) -> set[tuple[str, str, str]]:
    """`{(subject_id, predicate, object_id)}`，也含反向（无向关系）。"""
    out: set[tuple[str, str, str]] = set()
    for e in edges:
        s, p, o = e.get("subject_id"), e.get("predicate"), e.get("object_id")
        if s and p and o:
            out.add((s, p, o))
            out.add((o, p, s))
    return out


# ---------------------------------------------------------------- 挖掘：单块事实


def mine_single_chunk_facts(
    assertions: Any,
    chunks: Any,
    *,
    limit: int = 64,
    seed: int = 0,
) -> list[dict]:
    """单块事实题：一句话 + 它所在的块标题。

    判据（全部机检，任一不过就丢）：
    * `evidence_text` 长度 8–40 字（太短定位不到、太长不像「一句话」）；
    * 该文本在该块正文里**恰好出现一次**（唯一定位，避免题目有多种正确答案）；
    * 该块在语料里真实存在。

    答案取块**标题**：它是语料字段，不是图的判断，且同一块内唯一。
    """
    cmap = _as_chunk_map(chunks)
    alist = _as_assertion_list(assertions)
    rng = _rng(seed)
    pool: list[dict] = []
    seen: set[str] = set()
    for a in alist:
        cid = a.get("chunk_id")
        ev = (a.get("evidence_text") or "").strip()
        if not cid or not (8 <= len(ev) <= 40):
            continue
        chunk = cmap.get(cid)
        if not _unique_hit(chunk, ev):
            continue
        title = (chunk.get("title") or "").strip()
        if not title or not (chunk.get("chapter") or "").strip():
            continue
        key = f"{cid}\x1f{ev}"
        if key in seen:  # 同块同句只出一题
            continue
        seen.add(key)
        pool.append({"chunk": chunk, "assertion": a, "evidence": ev, "title": title})
    pool.sort(key=lambda d: (d["chunk"]["chunk_id"], d["assertion"]["assertion_id"]))
    out: list[dict] = []
    for row in _pick(rng, pool, limit):
        chunk = row["chunk"]
        tmpl = TEMPLATES["single_chunk"][
            _stable_index(chunk["chunk_id"], len(TEMPLATES["single_chunk"]))
        ]
        out.append(
            {
                "question": tmpl.format(evidence=row["evidence"], title=row["title"]),
                "answer": row["title"],
                "acceptable_answers": sorted(
                    {
                        row["title"],
                        f"{chunk['chapter']}/{row['title']}",
                        chunk["chunk_id"],
                    }
                ),
                "gold_chunk_ids": [chunk["chunk_id"]],
                "gold_assertion_ids": [row["assertion"]["assertion_id"]],
                "required_citations": [
                    {"chunk_id": chunk["chunk_id"], "evidence": row["evidence"]}
                ],
                "difficulty": "easy",
                "category": "single_chunk",
                "should_abstain": False,
            }
        )
    return out


# ---------------------------------------------------------------- 挖掘：别名


def mine_alias_questions(
    entities: Any,
    chunks: Any = None,
    *,
    limit: int = 48,
    seed: int = 0,
) -> list[dict]:
    """别名与消歧题：`{alias} 指的是谁`。

    只用**权威**别名（`review ∈ {auto, approved}`）。`pending_aliases` 与
    `rejected_aliases` 不出题——它们是「待人工判定」与「已否决」，写成标准答案
    就是让评测去学一个本系统自己都不敢用的判断。

    证据要求：某个块的正文里同时出现该别名与其规范名（否则「这个别名指它」
    这句话在语料里没有落脚点，题目就变成了在复述图）。
    """
    cmap = _as_chunk_map(chunks)
    elist = _as_entity_list(entities)
    rng = _rng(seed)
    pool: list[dict] = []
    for e in elist:
        name = _entity_name(e)
        if not name:
            continue
        for alias in _authoritative_aliases(e):
            if alias == name or len(alias) < 2:
                continue
            for cid in e.get("chunks") or ():
                chunk = cmap.get(cid)
                if not chunk:
                    continue
                if _in_text(chunk, alias) and _in_text(chunk, name):
                    pool.append(
                        {
                            "entity": e,
                            "alias": alias,
                            "chunk": chunk,
                            "evidence": _window_text(chunk["text"], alias),
                        }
                    )
                    break
    pool.sort(key=lambda d: (d["entity"].get("id") or "", d["alias"]))
    out: list[dict] = []
    for row in _pick(rng, pool, limit):
        e, chunk, alias = row["entity"], row["chunk"], row["alias"]
        tmpls = TEMPLATES["alias"]
        tmpl = tmpls[_stable_index(alias, len(tmpls))]
        out.append(
            {
                "question": tmpl.format(alias=alias, title=chunk.get("title") or ""),
                "answer": e["name"],
                "acceptable_answers": sorted({e["name"]}),
                "gold_chunk_ids": [chunk["chunk_id"]],
                "gold_assertion_ids": [],
                "required_citations": [
                    {"chunk_id": chunk["chunk_id"], "evidence": row["evidence"]}
                ],
                "difficulty": "easy",
                "category": "alias",
                "should_abstain": False,
            }
        )
    return out


# ---------------------------------------------------------------- 挖掘：直接关系


def mine_direct_relations(
    edges: Any,
    entities: Any,
    chunks: Any = None,
    *,
    limit: int = 48,
    seed: int = 0,
) -> list[dict]:
    """两实体直接关系题：`{A} 与 {B} 是什么关系`，答案取词表里的 `predicate`。

    判据：
    * `support_count >= 2` 且 `independent_scene_count >= 2`（单次提及不算「关系」）；
    * `predicate` 在 `vocab.CANONICAL_RELATIONS` 内（答案必须能对上词表）；
    * 两端实体都在图里、都不是泛称（`医疗干员` 这类不当题面主体）；
    * 边自带的 `evidence` 在某个 chunk 里**唯一出现**（可逐字定位）；
    * 该 `evidence` 里同时出现两个端点的名字（否则证据支撑不了这条关系）。
    """
    from . import identity  # noqa: F401  仅为与 _is_generic 共用延迟导入风格

    cmap = _as_chunk_map(chunks)
    elist = _as_entity_list(entities)
    by_id = {e.get("id"): e for e in elist}
    canon = set(vocab.CANONICAL_RELATIONS)
    rng = _rng(seed)
    pool: list[dict] = []
    for e in _as_edge_list(edges):
        if (e.get("support_count") or 0) < 2:
            continue
        if (e.get("independent_scene_count") or 0) < 2:
            continue
        pred = e.get("predicate") or ""
        if pred not in canon:
            continue
        sub, obj = by_id.get(e.get("subject_id")), by_id.get(e.get("object_id"))
        if not sub or not obj:
            continue
        sname, oname = _entity_name(sub), _entity_name(obj)
        if not sname or not oname or sname == oname:
            continue
        if _is_generic(sname) or _is_generic(oname):
            continue
        ev = (e.get("evidence") or "").strip()
        if not ev or sname not in ev or oname not in ev:
            continue
        hit_chunk = None
        for cid in e.get("chunks") or ():
            if _unique_hit(cmap.get(cid), ev):
                hit_chunk = cmap[cid]
                break
        if not hit_chunk:
            continue
        pool.append(
            {"edge": e, "chunk": hit_chunk, "subject": sub, "object": obj, "evidence": ev}
        )
    pool.sort(key=lambda d: d["edge"].get("edge_id") or "")
    out: list[dict] = []
    for row in _pick(rng, pool, limit):
        e = row["edge"]
        tmpls = TEMPLATES["direct_relation"]
        tmpl = tmpls[_stable_index(e["edge_id"], len(tmpls))]
        out.append(
            {
                "question": tmpl.format(
                    subject=row["subject"]["name"], object=row["object"]["name"]
                ),
                "answer": e["predicate"],
                "acceptable_answers": sorted({e["predicate"]}),
                "gold_chunk_ids": [row["chunk"]["chunk_id"]],
                "gold_assertion_ids": list(e.get("assertion_ids") or [])[:1],
                "required_citations": [
                    {"chunk_id": row["chunk"]["chunk_id"], "evidence": row["evidence"]}
                ],
                "difficulty": "medium",
                "category": "direct_relation",
                "should_abstain": False,
            }
        )
    return out


# ---------------------------------------------------------------- 挖掘：多跳


def mine_multihop(
    adjacency: Any,
    edges: Any,
    entities: Any,
    chunks: Any = None,
    *,
    limit: int = 48,
    seed: int = 0,
) -> list[dict]:
    """多跳题：`A -> ? -> C` 的中间实体。

    硬约束：

    * **只走 `out`**。`in` 是「别人指向我」，走它等于把方向反过来，答出来的
      路径与题面要求的入射关系不符。`adjacency.json` 里每个节点的 `out` 已含
      无向关系的两份拷贝，所以正向可达性在 `out` 上是完整的。
    * `A != C`，且中间实体不等于两端。
    * 两条边的 `support_count >= 2`，`predicate` 在词表内。
    * 中间实体**唯一**：`A` 到 `C` 之间只经过一个中间实体。多解会让「标准答案」
      本身不成立，这类候选直接丢弃而不做随机取舍。

    `chunks` 可选：给了才能把 `required_citations` 指到**真正含该句**的块。
    边自带的 `chunks` 是全部支撑块，`evidence` 只来自其中某一个。
    """
    cmap = _as_chunk_map(chunks) if chunks is not None else {}
    elist = _as_entity_list(entities)
    by_id = {e.get("id"): e for e in elist}
    edge_by_id = {(e.get("edge_id") or ""): e for e in _as_edge_list(edges)}
    canon = set(vocab.CANONICAL_RELATIONS)
    adj = adjacency or {}
    mids: dict[tuple[str, str], set[str]] = defaultdict(set)
    first_edge: dict[tuple[str, str, str], dict] = {}
    for a_id, node in adj.items():
        if a_id not in by_id:
            continue
        for item in node.get("out") or ():
            e1 = edge_by_id.get(item.get("edge_id") or "")
            if not e1 or (e1.get("support_count") or 0) < 2:
                continue
            if (e1.get("predicate") or "") not in canon:
                continue
            mid = e1.get("object_id") or ""
            if not mid or mid == a_id or mid not in by_id:
                continue
            for item2 in (adj.get(mid) or {}).get("out", ()):
                e2 = edge_by_id.get(item2.get("edge_id") or "")
                if not e2 or (e2.get("support_count") or 0) < 2:
                    continue
                if (e2.get("predicate") or "") not in canon:
                    continue
                c_id = e2.get("object_id") or ""
                if not c_id or c_id in (a_id, mid) or c_id not in by_id:
                    continue
                mids[(a_id, c_id)].add(mid)
                first_edge[(a_id, c_id, mid)] = e1
                first_edge[(a_id, mid, c_id)] = e2
    rng = _rng(seed)
    pool: list[dict] = []
    for (a_id, c_id), mset in mids.items():
        if len(mset) != 1:
            continue
        mid = next(iter(mset))
        a, b, c = by_id[a_id], by_id[mid], by_id[c_id]
        if _is_generic(a["name"]) or _is_generic(b["name"]) or _is_generic(c["name"]):
            continue
        e1 = first_edge.get((a_id, c_id, mid))
        e2 = first_edge.get((a_id, mid, c_id))
        if not e1 or not e2:
            continue
        pool.append(
            {"a": a, "b": b, "c": c, "e1": e1, "e2": e2}
        )
    pool.sort(key=lambda d: (d["a"].get("id") or "", d["c"].get("id") or ""))
    out: list[dict] = []
    for row in _pick(rng, pool, limit):
        tmpls = TEMPLATES["multihop"]
        tmpl = tmpls[_stable_index(row["a"]["id"] + row["c"]["id"], len(tmpls))]
        cites: list[dict] = []
        chunks: list[str] = []
        for e in (row["e1"], row["e2"]):
            ev = (e.get("evidence") or "").strip()
            if not ev:
                continue
            # 不能直接取 `chunks[0]`：边的 `chunks` 是所有支撑块的集合，
            # 而 `evidence` 只来自其中**某一个**块（其 `evidence_assertion_id`
            # 所在块）。取 `chunks[0]` 会让引用指向一个根本不含该句的块，
            # 校验时才报「无法逐字定位」。这里按证据真实落点挑块。
            cid = None
            for cand in e.get("chunks") or ():
                if _unique_hit(cmap.get(cand), ev):
                    cid = cand
                    break
            if cid is None:
                for cand in e.get("chunks") or ():
                    if _in_text(cmap.get(cand), ev):
                        cid = cand
                        break
            if cid is None and not cmap:
                # 没给语料时无法判定落点，退回首个支撑块（调用方需自行保证可定位）
                cid = (e.get("chunks") or [None])[0]
            if cid:
                cites.append({"chunk_id": cid, "evidence": ev})
                chunks.append(cid)
        if not cites:
            continue
        out.append(
            {
                "question": tmpl.format(a=row["a"]["name"], c=row["c"]["name"]),
                "answer": row["b"]["name"],
                "acceptable_answers": sorted({row["b"]["name"]}),
                "gold_chunk_ids": sorted(set(chunks)),
                "gold_assertion_ids": [],
                "required_citations": cites,
                "difficulty": "hard",
                "category": "multihop",
                "should_abstain": False,
            }
        )
    return out


# ---------------------------------------------------------------- 挖掘：跨块跨章


def mine_cross_chunk(
    entities: Any,
    chunks: Any,
    *,
    limit: int = 32,
    seed: int = 0,
) -> list[dict]:
    """跨块跨章节综合题：`{name}` 在哪些章节出现过。

    判据：该实体在**至少两个不同 chapter** 的块正文里都出现了自己的名字。
    答案取章节代号集合——章节是语料字段，逐块可核。

    证据取每章一个块，`required_citations` 里给出该块中包含实体名的原文句子。
    """
    cmap = _as_chunk_map(chunks)
    elist = _as_entity_list(entities)
    rng = _rng(seed)
    pool: list[dict] = []
    for e in elist:
        name = _entity_name(e)
        if not name or len(name) < 2:
            continue
        by_chapter: dict[str, str] = {}
        for cid in e.get("chunks") or ():
            chunk = cmap.get(cid)
            if not chunk or not _in_text(chunk, name):
                continue
            chapter = (chunk.get("chapter") or "").strip()
            if not chapter:
                continue
            by_chapter.setdefault(chapter, cid)
        if len(by_chapter) < 2:
            continue
        # 只挖「章节数 ≤ 4」的实体：答案要把**全部**章节列出来，而题目又必须能被
        # 引用块完整支撑（否则就是「答案对但引用没覆盖到」——解题者读完给出的块
        # 也推不出被列进去的章节）。两个条件同时成立的唯一做法是：答案集合 ==
        # 引用块覆盖的章节集合，且这个集合小到可以逐章给引用。
        # 这是**进入候选池之前**的前置条件，不是抽完再筛——抽完再筛会让配额被
        # 白抽掉的候选占走（实测那样只有 26/32，凭空多出 6 个缺口）。
        # 代价：12+ 章的实体（实测 137 个）不出题，这是如实少给，不是判据放宽。
        if len(by_chapter) > _CROSS_CHUNK_MAX_CHAPTERS:
            continue
        pool.append({"entity": e, "by_chapter": by_chapter})
    pool.sort(key=lambda d: d["entity"].get("id") or "")
    out: list[dict] = []
    for row in _pick(rng, pool, limit):
        chapters = sorted(row["by_chapter"])
        # 只挖「章节数 ≤ 4」的实体：答案要把**全部**章节列出来，而题目又必须能被
        # 引用块完整支撑（否则就是「答案对但引用没覆盖到」——解题者读完给出的块
        # 也推不出被列进去的章节）。两个条件同时成立的唯一做法是：答案集合 ==
        # 引用块覆盖的章节集合，且这个集合本身小到可以枚举。
        # 代价是 12+ 章的实体（实测 137 个）不出题；这是如实少给，不是判据放宽。
        if len(chapters) > _CROSS_CHUNK_MAX_CHAPTERS:
            continue
        cites: list[dict] = []
        for chapter in chapters:  # 逐章都给引用，不截断
            cid = row["by_chapter"][chapter]
            chunk = cmap[cid]
            ev = _window_text(chunk["text"], row["entity"]["name"])
            if not ev:
                continue
            cites.append({"chunk_id": cid, "evidence": ev})
        if len(cites) != len(chapters):
            continue
        tmpls = TEMPLATES["cross_chunk"]
        tmpl = tmpls[_stable_index(row["entity"]["id"], len(tmpls))]
        out.append(
            {
                "question": tmpl.format(name=row["entity"]["name"]),
                "answer": "、".join(chapters),
                "acceptable_answers": sorted(
                    {"、".join(chapters), ",".join(chapters), " ".join(chapters)}
                ),
                "gold_chunk_ids": sorted({c["chunk_id"] for c in cites}),
                "gold_assertion_ids": [],
                "required_citations": cites,
                "difficulty": "medium",
                "category": "cross_chunk",
                "should_abstain": False,
            }
        )
    return out


# ---------------------------------------------------------------- 挖掘：时间冲突


def mine_time_conflict(
    assertions: Any,
    chunks: Any,
    entities: Any = None,
    *,
    limit: int = 32,
    seed: int = 0,
) -> list[dict]:
    """时间冲突 / 剧情阶段题：互斥谓词在不同场景并存。

    只取 `assertion.check_qualifier_consistency()` 判为 `action == "kept_both"` 的
    组合——它们落在**不同场景**，因此「先同盟后敌对」是合法的剧情演进，而不是
    抽取矛盾。`needs_human`（同场景互斥）不进金标准：那正是需要人工裁定、不能
    让评测去猜的东西。

    答案要求说明「时序演进」，因此 `should_abstain=False`。
    """
    from . import assertion as assertion_mod

    cmap = _as_chunk_map(chunks)
    alist = _as_assertion_list(assertions)
    by_aid = {a.get("assertion_id"): a for a in alist}
    elist = _as_entity_list(entities) if entities is not None else []
    by_eid = {e.get("id"): e for e in elist}
    conflicts = assertion_mod.check_qualifier_consistency(alist)
    rng = _rng(seed)
    pool: list[dict] = []
    for c in conflicts:
        if c.get("action") != "kept_both":
            continue
        s_id, o_id = c.get("subject_id"), c.get("object_id")
        a1, a2 = by_aid.get(c.get("a")), by_aid.get(c.get("b"))
        if not a1 or not a2:
            continue
        chunks_two = [a1.get("chunk_id"), a2.get("chunk_id")]
        if len({*chunks_two}) != 2:
            continue
        if not all(_in_text(cmap.get(cid), a1.get("evidence_text") or "") for cid in chunks_two[:1]):
            continue
        if not all(_in_text(cmap.get(cid), a2.get("evidence_text") or "") for cid in chunks_two[1:]):
            continue
        sname = _entity_name(by_eid.get(s_id)) or s_id or ""
        oname = _entity_name(by_eid.get(o_id)) or o_id or ""
        if not sname or not oname:
            continue
        r1 = (a1.get("predicate") or "").strip()
        r2 = (a2.get("predicate") or "").strip()
        if not r1 or not r2 or r1 == r2:
            continue
        pool.append(
            {
                "conflict": c,
                "sname": sname,
                "oname": oname,
                "r1": r1,
                "r2": r2,
                "a1": a1,
                "a2": a2,
            }
        )
    pool.sort(key=lambda d: (d["conflict"].get("a") or "", d["conflict"].get("b") or ""))
    # 同一 (实体对, 谓词对) 往往有**多组**互斥断言（不同 chunk 各一条）。
    # 不去重就会产出问句完全相同、只有 assertion_id 不同的重复题——校验里
    # 的重复问句检测会直接报错，而金标准里出现重复题本身就是缺陷。
    seen_key: set[tuple] = set()
    deduped: list[dict] = []
    for row in pool:
        key = (
            row["conflict"].get("subject_id"),
            row["conflict"].get("object_id"),
            tuple(sorted((row["r1"], row["r2"]))),
        )
        if key in seen_key:
            continue
        seen_key.add(key)
        deduped.append(row)
    out: list[dict] = []
    for row in _pick(rng, deduped, limit):
        cites = [
            {"chunk_id": row["a1"]["chunk_id"], "evidence": row["a1"]["evidence_text"]},
            {"chunk_id": row["a2"]["chunk_id"], "evidence": row["a2"]["evidence_text"]},
        ]
        tmpls = TEMPLATES["time_conflict"]
        tmpl = tmpls[_stable_index(row["conflict"]["a"] + row["conflict"]["b"], len(tmpls))]
        answer = "时序演进"
        out.append(
            {
                "question": tmpl.format(
                    a=row["sname"], b=row["oname"], r1=row["r1"], r2=row["r2"]
                ),
                "answer": answer,
                "acceptable_answers": sorted(
                    {
                        answer,
                        "剧情阶段变化，两条都成立",
                        "不是矛盾，是剧情先后变化",
                        f"先后出现{row['r1']}与{row['r2']}，不是矛盾",
                    }
                ),
                "gold_chunk_ids": sorted({c["chunk_id"] for c in cites}),
                "gold_assertion_ids": [row["a1"]["assertion_id"], row["a2"]["assertion_id"]],
                "required_citations": cites,
                "difficulty": "hard",
                "category": "time_conflict",
                "should_abstain": False,
            }
        )
    return out


# ---------------------------------------------------------------- 挖掘：否定与不可回答


def _negation_rows(chunks: Any, entities: Any, edges: Any) -> list[dict]:
    """扫语料找「否定词紧贴关系词、且主语实体就在否定词前面」的句子。

    判据（第一版只做了「否定词后 ≤2 字内出现关系词」一条，实测 344 条命中里
    绝大多数是假否定——「他没有离开过…」的真主语根本不在句子里、或者实体名
    被关系词吞掉，直接拿去出题会产出「原文并不支持『否』」的坏题）：

    1. 在**单句内**匹配（跨句的否定词与关系词没有句法联系）；
    2. 否定词与关系词之间最多隔 1 个字（「没有保护」「不会参与」「并未离开」）；
    3. 关系词后面不是名词化后缀（`感染者`/`矿石病` 里的 `感染` 不是谓词）；
    4. 实体名出现在否定词**之前**，且名字末尾到否定词之间最多 2 个字，且这些字
       只能是连词/副词（「侯爵没有保护」「霜星不会背叛」），或者整段是敬称
       （「埃琳娜小姐并未离开」）。这一条把「他没有离开」这类主语是人称代词、
       真主语在上一句的情况全部排掉；
    5. 该实体在图上**没有**这个谓词的任何边——否则题目就成了「原文否定、图肯定」，
       标准答案自相矛盾；
    6. 句子不是疑问/反问（含「为什么」「难道」「是否」「吗」「？」「呢」）。反问句
       里「为什么公爵没有感染」的答案并不是「否」，它压根不是在陈述事实；
    7. 名字与否定词之间的连词里不含「也/又/再/还」（并列否定）。「他们既没有通讯
       设备也没有人指挥」问「通讯设备指挥了吗」是错题——被否定的是并列结构里的
       一个宾语，不是主谓关系；
    8. 关系词后面不接时间从句（「在前往北方冰原之后」「大叛乱时期」）。此时关系词
       是时间状语的一部分，问「命运前往了吗」不通。

    实体解析需要 `entities`，故一并在这里做（调用方不必再挑实体）。
    """
    cmap = _as_chunk_map(chunks)
    rels = sorted(vocab.CANONICAL_RELATIONS, key=len, reverse=True)
    by_name: dict[str, dict] = {}
    for e in _as_entity_list(entities):
        name = _entity_name(e)
        if name and len(name) >= 2:
            by_name[name] = e
    has_rel: set[tuple[str, str]] = set()
    for e in _as_edge_list(edges):
        s, p, o = e.get("subject_id"), e.get("predicate"), e.get("object_id")
        if s and p:
            has_rel.add((s, p))
        if o and p:
            has_rel.add((o, p))
    rows: list[dict] = []
    for cid in sorted(cmap):
        chunk = cmap[cid]
        text = chunk.get("text") or ""
        for neg in NEGATION_WORDS:
            i = text.find(neg)
            while i >= 0:
                sentence = _sentence_at(text, i, i + 1)
                start = text.find(sentence) if sentence else -1
                if not (0 <= start <= i < start + len(sentence)) or len(sentence) > 120:
                    i = text.find(neg, i + 1)
                    continue
                rel = ""
                rel_pos = -1
                for r in rels:
                    k = sentence.find(r, i - start + len(neg))
                    gap = k - (i - start) - len(neg)
                    if 0 <= gap <= 1:
                        rel, rel_pos = r, k
                        break
                if not rel or sentence[rel_pos + len(rel) : rel_pos + len(rel) + 1] in _NOUN_TAIL:
                    i = text.find(neg, i + 1)
                    continue
                # 判据 8：关系词后面接时间从句时它是状语，不是谓词。
                #   两个形态：`在前往北方冰原之后`（关系词前有「在」、后面某处有「之后」）
                #   与 `大叛乱时期`（关系词后紧跟时间尾词）。
                if any(sentence[rel_pos + len(rel) :].startswith(t) for t in _TEMPORAL_TAIL):
                    i = text.find(neg, i + 1)
                    continue
                if sentence[:rel_pos].endswith("在") and any(
                    t in sentence[rel_pos : rel_pos + 16] for t in _TEMPORAL_TAIL
                ):
                    i = text.find(neg, i + 1)
                    continue
                # 判据 6：疑问/反问句里的「没有」不是在否定事实。
                if any(w in sentence for w in _INTERROGATIVE):
                    i = text.find(neg, i + 1)
                    continue
                head, sep, _rest = sentence.partition("：")
                body = sentence[len(head) + len(sep) :]
                body_start = start + len(head) + len(sep)
                neg_at = i - body_start
                best: tuple[int, str, dict, str] | None = None
                for name, ent in by_name.items():
                    p = body.find(name)
                    if p < 0 or p + len(name) > neg_at:
                        continue
                    between = body[p + len(name) : neg_at]
                    if len(between) > 2:
                        continue
                    # 判据 4：中间只能全是连词，或者整段就是敬称。
                    if between and not (
                        all(ch in _GLUE_CHARS for ch in between) or between in _HONORIFICS
                    ):
                        continue
                    # 判据 7：并列否定（「既没有…也没有…」）时主语不是前一个名词。
                    if any(ch in _PARALLEL_NEG for ch in between):
                        continue
                    if best is None or p + len(name) > best[0]:
                        best = (p + len(name), name, ent, between)
                if best is None:
                    i = text.find(neg, i + 1)
                    continue
                _end, name, ent, _between = best
                if (ent.get("id"), rel) in has_rel:
                    i = text.find(neg, i + 1)
                    continue
                if _is_generic(name):
                    i = text.find(neg, i + 1)
                    continue
                rows.append(
                    {
                        "chunk": chunk,
                        "negation": neg,
                        "relation": rel,
                        "pos": i,
                        "rel_pos": rel_pos,
                        "sentence": sentence,
                        "entity": ent,
                        "entity_name": name,
                    }
                )
                i = text.find(neg, i + 1)
    return rows


def mine_negative_and_unanswerable(
    chunks: Any,
    entities: Any,
    edges: Any,
    *,
    limit: int = 32,
    seed: int = 0,
) -> list[dict]:
    """否定题（是非，答案「否」）+ 不可回答题（`should_abstain=True`）。

    否定题的判据全部在 `_negation_rows()` 里（单句内、否定词与谓词间隔 ≤1 字、
    谓词后不接名词化后缀、主语实体紧贴否定词之前、且图上该实体没有这个谓词的
    任何边）。这里只负责把候选变成题。

    不可回答题判据：
    * 标签不在 `CANONICAL_RELATIONS`、图里也没有任何边用这个谓词（`_absent_labels`）；
    * 两个实体都在图里，且图上这两点之间**没有任何边**（连别名都没有）；
    * `should_abstain=True`、`gold_chunk_ids=[]`、`gold_assertion_ids=[]`。
    """
    cmap = _as_chunk_map(chunks)
    elist = _as_entity_list(entities)
    by_name: dict[str, dict] = {}
    for e in elist:
        name = _entity_name(e)
        if name and len(name) >= 2:
            by_name[name] = e
    pair = _edge_pair_set(_as_edge_list(edges))
    rng = _rng(seed)

    # ---- 否定题
    neg_pool: list[dict] = [
        row for row in _negation_rows(cmap, entities, edges) if row.get("sentence")
    ]
    neg_pool.sort(key=lambda d: (d["chunk"]["chunk_id"], d["pos"], d["entity_name"]))
    # 否定题与不可回答题同属一类配额。否定题池实测只有十几个（中文否定句的真主语
    # 大量是人称代词或跨句，机械规则能确证的很少），所以先按半分配额取否定题，
    # 剩下的**由同类里的不可回答题补满**——这不是「用别的类别凑数」。
    n_neg = min(len(neg_pool), max(0, limit // 2)) if limit else 0

    out: list[dict] = []
    for row in _pick(rng, neg_pool, n_neg):
        phrase = _REL_PHRASE.get(row["relation"], row["relation"])
        tmpls = TEMPLATES["negative_abstain"]
        tmpl = tmpls[_stable_index(row["chunk"]["chunk_id"] + str(row["pos"]), len(tmpls))]
        out.append(
            {
                "question": tmpl.format(
                    sentence=row["sentence"],
                    entity=row["entity_name"],
                    rel_phrase=phrase,
                ),
                "answer": "否",
                "acceptable_answers": ["否", "没有", "未发生"],
                "gold_chunk_ids": [row["chunk"]["chunk_id"]],
                "gold_assertion_ids": [],
                "required_citations": [
                    {"chunk_id": row["chunk"]["chunk_id"], "evidence": row["sentence"]}
                ],
                "difficulty": "medium",
                "category": "negative_abstain",
                "should_abstain": False,
            }
        )

    # ---- 不可回答题
    labels = _absent_labels(_as_edge_list(edges))
    if not labels:
        return out
    names = sorted(by_name)
    absent_pool: list[dict] = []
    for i, n1 in enumerate(names):
        e1 = by_name[n1]
        for n2 in names[i + 1 :]:
            e2 = by_name[n2]
            if e1["id"] == e2["id"]:
                continue
            if (e1["id"], "", e2["id"]) in pair:
                continue
            if any(
                (e1["id"], p, e2["id"]) in pair or (e2["id"], p, e1["id"]) in pair
                for p in vocab.CANONICAL_RELATIONS
            ):
                continue
            absent_pool.append({"a": e1, "b": e2})
            if len(absent_pool) >= 4000:
                break
        if len(absent_pool) >= 4000:
            break
    absent_pool.sort(key=lambda d: (d["a"]["id"], d["b"]["id"]))
    n_abs = max(0, limit - n_neg) if limit else 0
    for k, row in enumerate(_pick(rng, absent_pool, n_abs)):
        label = labels[k % len(labels)]
        tmpls = TEMPLATES["negative_abstain_absent"]
        tmpl = tmpls[k % len(tmpls)]
        out.append(
            {
                "question": tmpl.format(a=row["a"]["name"], b=row["b"]["name"], label=label),
                "answer": "应拒答：图中不存在这类关系的任何记录。",
                "acceptable_answers": sorted(
                    {
                        "应拒答：图中不存在这类关系的任何记录。",
                        "无法回答",
                        "拒答",
                        f"图中没有{label}关系",
                    }
                ),
                "gold_chunk_ids": [],
                "gold_assertion_ids": [],
                "required_citations": [],
                "difficulty": "medium",
                "category": "negative_abstain",
                "should_abstain": True,
                # 机检用的线索：`validate_gold_item` 靠这三个字段反查「两实体确实
                # 在图里、且图里确实没有这个谓词的任何边」。缺了它们，不可回答题
                # 的「图里真没有」就只是挖掘时的声称，校验等于没做。
                "_absent_label": label,
                "_absent_a_name": row["a"]["name"],
                "_absent_b_name": row["b"]["name"],
            }
        )
    return out


# ---------------------------------------------------------------- 挖掘：易混淆对抗


def mine_adversarial(
    entities: Any,
    chunks: Any,
    *,
    limit: int = 16,
    seed: int = 0,
) -> list[dict]:
    """易混淆对抗题：共享前缀/后缀但**类型不同**的实体对。

    判据：
    * 两名字一个包含另一个（共享前缀或后缀），因此字面极度相似；
    * `type` 不同（同名同类型往往是同一个东西的两次命名，不像「易混淆」）；
    * 存在某个块，其中出现了**短名**却没有出现长名——这样证据只支持短的那个，
      题目才有区分力。

    注意包含方向：短名是长名的子串，所以「含长名」必然「含短名」，反过来不成立。
    因此证据块必须按「有短名、无长名」找，不能按「有长名、无短名」找——后者
    永远为空（这正是本函数第一版返回 0 条的原因）。

    题面**不能**写出两边的类型。第一版模板是
    「「{a}」（类型：{ta}）与「{b}」（类型：{tb}）是同一个实体吗」，等于把
    判断依据直接印在题目里，答案退化成抄题。现在题面只说「名字相近」，类型差异
    留在答案里；答案另外必须说出「所引原文只出现了短名」这个可逐字复核的事实。
    """
    cmap = _as_chunk_map(chunks)
    elist = _as_entity_list(entities)
    rng = _rng(seed)
    by_name = {_entity_name(e): e for e in elist if len(_entity_name(e)) >= 2}
    names = sorted(by_name)
    pool: list[dict] = []
    for i, n1 in enumerate(names):
        for n2 in names[i + 1 :]:
            if not n1.startswith(n2) and not n2.startswith(n1):
                continue
            e1, e2 = by_name[n1], by_name[n2]
            if e1.get("type") == e2.get("type"):
                continue
            long_e, long_n, short_n = (e1, n1, n2) if len(n1) >= len(n2) else (e2, n2, n1)
            short_e = e2 if long_e is e1 else e1
            hit = None
            for cid in short_e.get("chunks") or ():
                chunk = cmap.get(cid)
                if not chunk:
                    continue
                if _in_text(chunk, short_n) and not _in_text(chunk, long_n):
                    hit = chunk
                    break
            if not hit:
                continue
            pool.append(
                {"long": long_e, "long_n": long_n, "short": short_e, "short_n": short_n, "chunk": hit}
            )
    pool.sort(key=lambda d: (d["long"].get("id") or "", d["short"].get("id") or ""))
    out: list[dict] = []
    for row in _pick(rng, pool, limit):
        ev = _window_text(row["chunk"]["text"], row["short_n"])
        if not ev:
            continue
        tmpls = TEMPLATES["adversarial"]
        tmpl = tmpls[_stable_index(row["long"]["id"] + row["short"]["id"], len(tmpls))]
        ta, tb = row["long"].get("type"), row["short"].get("type")
        answer = (
            f"不是同一个实体：「{row['long_n']}」的类型是{ta}，「{row['short_n']}」的类型是{tb}；"
            f"且所引原文只出现「{row['short_n']}」，没有出现「{row['long_n']}」。"
        )
        out.append(
            {
                "question": tmpl.format(a=row["long_n"], b=row["short_n"]),
                "answer": answer,
                "acceptable_answers": sorted(
                    {
                        answer,
                        f"不是同一个实体，{row['long_n']}是{ta}，{row['short_n']}是{tb}",
                        f"不同实体，类型分别是{ta}和{tb}",
                    }
                ),
                "gold_chunk_ids": [row["chunk"]["chunk_id"]],
                "gold_assertion_ids": [],
                "required_citations": [
                    {"chunk_id": row["chunk"]["chunk_id"], "evidence": ev}
                ],
                "difficulty": "hard",
                "category": "adversarial",
                "should_abstain": False,
                "_adv_long": row["long_n"],
                "_adv_short": row["short_n"],
            }
        )
    return out


# ---------------------------------------------------------------- 校验


def validate_gold_item(
    item: dict,
    chunks_by_id: dict[str, dict],
    *,
    edges: Sequence[dict] | None = None,
    entities_by_id: dict[str, dict] | None = None,
) -> list[str]:
    """校验一条题，返回**问题描述列表**（空列表 = 通过）。

    校验项（每项都能被构造输入打红，见 `tests/test_goldset.py`）：

    1. 九个必填字段齐全、类型正确、非空；
    2. `question_id` 匹配 `^q-(sc|al|dr|mh|cx|tc|na|ad)-\\d{6}$`；
    3. `category` 在八类内，且缩写与类别一致；
    4. `gold_chunk_ids` 每个都在语料里真实存在；
    5. `gold_assertion_ids` 是 list[str]；
    6. 每条 `required_citations` 的 `chunk_id` 必须落在 `gold_chunk_ids` 内，
       且 `evidence` 能在该块正文里用 `str.find` 定位（不接受「差不多」）；
    7. `should_abstain=True` → `gold_chunk_ids`、`gold_assertion_ids`、
       `required_citations` 三者都空；
    8. `should_abstain=False` → `gold_chunk_ids` 非空且 `required_citations` 非空；
    9. `acceptable_answers` 必须含 `answer`（否则正确答案自己都不算对）；
    10. 传入 `edges`/`entities_by_id` 时：不可回答题必须两实体都在图里、
        且图里不存在该谓词的任何边；
    11. 跨块跨章题：答案列出的章节集合必须恰好等于引用块覆盖的章节集合
        （答案列了某章、却没有那一章的引用块 = 不可解，属于「答案对但引用错」）；
    12. 对抗题：题面不得出现「类型」二字或具体类型词（否则判断依据直接印在
        题面上，答案退化成抄题），且引用块必须只支持短名。判据比较前先把两个
        实体名从题面剥掉——实体名本身可能含类型字（「…哗变事件」的类型是「事件」），
        不剥会把合法题判红。
    """
    errs: list[str] = []
    for f in REQUIRED_FIELDS:
        if f not in item:
            errs.append(f"缺少必填字段 {f}")
    if errs:
        return errs

    qid = item["question_id"]
    if not isinstance(qid, str) or not QUESTION_ID_RE.match(qid):
        errs.append(f"question_id 格式非法：{qid!r}")
    elif CATEGORY_ABBR.get(item.get("category")) and not qid.startswith(
        "q-" + CATEGORY_ABBR[item["category"]] + "-"
    ):
        errs.append(f"question_id 缩写与 category 不符：{qid!r} vs {item.get('category')!r}")

    if item.get("category") not in CATEGORIES:
        errs.append(f"category 不在八类内：{item.get('category')!r}")

    for f in ("question", "answer"):
        v = item.get(f)
        if not isinstance(v, str) or not v.strip():
            errs.append(f"{f} 必须是非空字符串")

    if item.get("difficulty") not in DIFFICULTIES:
        errs.append(f"difficulty 非法：{item.get('difficulty')!r}")

    acc = item.get("acceptable_answers")
    if not isinstance(acc, list) or not acc:
        errs.append("acceptable_answers 必须是非空 list")
    elif item.get("answer") not in acc:
        errs.append("acceptable_answers 未包含 answer（正确答案自己都判定为错）")

    gold = item.get("gold_chunk_ids")
    if not isinstance(gold, list):
        errs.append("gold_chunk_ids 必须是 list")
        gold = []
    for cid in gold:
        if cid not in chunks_by_id:
            errs.append(f"gold_chunk_ids 含不存在的块：{cid!r}")

    aids = item.get("gold_assertion_ids")
    if not isinstance(aids, list) or any(not isinstance(a, str) for a in aids):
        errs.append("gold_assertion_ids 必须是 list[str]")

    abstain = item.get("should_abstain")
    if not isinstance(abstain, bool):
        errs.append("should_abstain 必须是 bool")

    cites = item.get("required_citations")
    if not isinstance(cites, list):
        errs.append("required_citations 必须是 list")
        cites = []
    for c in cites:
        if not isinstance(c, dict):
            errs.append("required_citations 的元素必须是 dict")
            continue
        cid = c.get("chunk_id")
        ev = c.get("evidence")
        if cid not in gold:
            errs.append(f"引用块的 {cid!r} 不在 gold_chunk_ids 内")
            continue
        chunk = chunks_by_id.get(cid)
        if not chunk:
            errs.append(f"引用块不存在：{cid!r}")
            continue
        if not isinstance(ev, str) or not ev.strip():
            errs.append(f"引用证据为空：{cid!r}")
            continue
        if chunk.get("text", "").find(ev) < 0:
            errs.append(f"引用证据无法在该块正文中逐字定位：{cid!r} / {ev[:24]!r}")

    if abstain is True:
        if gold:
            errs.append("should_abstain=True 时 gold_chunk_ids 必须为空")
        if aids:
            errs.append("should_abstain=True 时 gold_assertion_ids 必须为空")
        if cites:
            errs.append("should_abstain=True 时 required_citations 必须为空")
        if edges is not None and entities_by_id is not None:
            label = item.get("_absent_label")
            if label is not None:
                if label in set(vocab.CANONICAL_RELATIONS):
                    errs.append(f"不可回答题用了词表内的谓词：{label!r}")
                if any(e.get("predicate") == label for e in edges):
                    errs.append(f"不可回答题的谓词在图上存在：{label!r}")
            by_name = {(e.get("name") or ""): e for e in entities_by_id.values()}
            ids: list[str] = []
            for slot in ("a", "b"):
                nm = item.get(f"_absent_{slot}_name")
                if not nm:
                    errs.append(f"不可回答题缺 _absent_{slot}_name，无法机检实体是否在图里")
                    continue
                ent = by_name.get(nm)
                if ent is None:
                    errs.append(f"不可回答题的实体不在图里：{nm!r}")
                else:
                    ids.append(ent.get("id") or "")
            # 「图里确实没有这类关系的任何记录」不能只是挖掘时的声称：
            # 直接按 id 反查，两实体之间不许存在该谓词的边（任一方向）。
            if len(ids) == 2 and ids[0] and ids[1] and label is not None:
                pair = _edge_pair_set(edges)
                if (ids[0], label, ids[1]) in pair or (ids[1], label, ids[0]) in pair:
                    errs.append(
                        "不可回答题的两实体在图上其实有该谓词的边："
                        f"{item.get('_absent_a_name')!r} / {label!r} / {item.get('_absent_b_name')!r}"
                    )
    elif abstain is False:
        if not gold:
            errs.append("should_abstain=False 时 gold_chunk_ids 不能为空")
        if not cites:
            errs.append("should_abstain=False 时 required_citations 不能为空")

    # 11. 跨块跨章题：答案列出的章节集合必须**恰好等于**引用块覆盖的章节集合。
    # 这条是「不接受答案对但引用错」在本题型上的具体形态——答案列了 act30side，
    # 而给出的引用块全在别的章节里，解题者读完块也推不出那一章，题就是不可解的。
    # 早期版本只取前 3 章给引用、答案却列全部章节，实测 38 处不覆盖。
    if item.get("category") == "cross_chunk" and isinstance(item.get("answer"), str):
        claimed = {c for c in item["answer"].split("、") if c}
        covered = {
            (chunks_by_id[cid].get("chapter") or "").strip()
            for cid in gold
            if cid in chunks_by_id
        }
        if len(covered - {""}) < 2:
            errs.append(f"跨章题的引用块只覆盖了 {len(covered - {''})} 个章节")
        missing = sorted(claimed - covered)
        if missing:
            errs.append(f"跨章题的答案章节未被引用块覆盖：{missing[:4]}")
        if claimed and claimed != covered - {""}:
            errs.append("跨章题的答案章节集合与引用块覆盖的章节集合不一致")

    # 12. 对抗题：题面不得直接给出判断依据。
    # 第一版模板把两个类型印在题面里（「「X」（类型：组织）与「Y」（类型：角色）
    # 是同一个实体吗」），答案于是退化成抄题——不需要读任何原文就能答对。
    # 这条只在挖出来的题上生效（靠 `_adv_long`/`_adv_short` 认出题型），
    # 手写题不受影响。
    long_n = item.get("_adv_long")
    short_n = item.get("_adv_short")
    if long_n and short_n and isinstance(item.get("question"), str):
        q = item["question"]
        # 先把两个实体名从题面里剥掉再找类型词：实体名本身可能就含类型字
        # （「第四集团军军队哗变**事件**」的类型是「事件」），不剥会把合法题判红。
        stripped = q.replace(long_n, "\u0000").replace(short_n, "\u0000")
        if "类型" in stripped:
            errs.append("对抗题在题面里写了「类型」，等于把判断依据直接给出")
        if entities_by_id is not None:
            by_name = {(e.get("name") or ""): e for e in entities_by_id.values()}
            for nm in (long_n, short_n):
                t = (by_name.get(nm) or {}).get("type")
                if t and t in stripped:
                    errs.append(f"对抗题的题面里出现了实体类型 {t!r}，答案可从题面抄出")
        # 证据块必须只支持短名（有短名、无长名），否则题目没有区分力
        for c in cites:
            chunk = chunks_by_id.get(c.get("chunk_id"))
            if chunk is None:
                continue
            body = chunk.get("text", "")
            if short_n not in body or long_n in body:
                errs.append(
                    f"对抗题的引用块同时/都没有出现两个名字，无法区分：{c.get('chunk_id')!r}"
                )
    return errs


# ---------------------------------------------------------------- 组装


def _quotas(total: int) -> dict[str, int]:
    """按 `CATEGORY_SHARE` 把 `total` 切成整数配额；余数补给占比最大者。"""
    quotas = {c: int(total * CATEGORY_SHARE[c]) for c in CATEGORIES}
    rest = total - sum(quotas.values())
    for c in sorted(CATEGORIES, key=lambda c: (-CATEGORY_SHARE[c], c)):
        if rest <= 0:
            break
        quotas[c] += 1
        rest -= 1
    return quotas


def _number(items: list[dict]) -> list[dict]:
    """给每条题补 `question_id`（按类别分组后顺序编号）。"""
    counters: dict[str, int] = defaultdict(int)
    for it in items:
        abbr = CATEGORY_ABBR[it["category"]]
        counters[abbr] += 1
        it["question_id"] = f"q-{abbr}-{counters[abbr]:06d}"
    return items


def find_duplicate_questions(items: Sequence[dict]) -> list[dict]:
    """找出问句完全相同的题（金标准里出现重复题本身就是缺陷）。

    返回 `[{question, first, again}]`，空列表表示没有重复。抽成独立函数是为了让
    这条判据能被单独喂坏输入打红——它原先内联在 `build_gold_set` 里，测试只能
    靠「真实数据上恰好 0 条」来"证明"，而 0 条既可能是判据生效、也可能是判据
    从来没被触发过。
    """
    seen: dict[str, str] = {}
    out: list[dict] = []
    for it in items:
        q = it.get("question") or ""
        if q in seen:
            out.append(
                {"question": q, "first": seen[q], "again": it.get("question_id") or ""}
            )
        else:
            seen[q] = it.get("question_id") or ""
    return out


def build_gold_set(
    assertions: Any,
    chunks: Any,
    edges: Any,
    entities: Any,
    adjacency: Any,
    *,
    total: int = 320,
    seed: int = 20261101,
) -> dict:
    """组装整套 Gold Set 与报告。

    配额按计划占比分配。**某类候选不足时如实少给并记缺口，绝不用别类凑数**——
    凑数会让「八类各占多少」这个报告字段永久失真，而失真比缺口更难发现。
    """
    cmap = _as_chunk_map(chunks)
    quotas = _quotas(total)
    miners: dict[str, Any] = {
        "single_chunk": lambda n: mine_single_chunk_facts(
            assertions, cmap, limit=n, seed=seed
        ),
        "alias": lambda n: mine_alias_questions(entities, cmap, limit=n, seed=seed),
        "direct_relation": lambda n: mine_direct_relations(
            edges, entities, cmap, limit=n, seed=seed
        ),
        "multihop": lambda n: mine_multihop(
            adjacency, edges, entities, cmap, limit=n, seed=seed
        ),
        "cross_chunk": lambda n: mine_cross_chunk(entities, cmap, limit=n, seed=seed),
        "time_conflict": lambda n: mine_time_conflict(
            assertions, cmap, entities, limit=n, seed=seed
        ),
        "negative_abstain": lambda n: mine_negative_and_unanswerable(
            cmap, entities, edges, limit=n, seed=seed
        ),
        "adversarial": lambda n: mine_adversarial(entities, cmap, limit=n, seed=seed),
    }
    all_items: list[dict] = []
    produced: dict[str, int] = {}
    for cat in CATEGORIES:
        got = miners[cat](quotas[cat])
        produced[cat] = len(got)
        all_items.extend(got)
    _number(all_items)

    by_id = {e.get("id"): e for e in _as_entity_list(entities)}
    failures: list[dict] = []
    for it in all_items:
        errs = validate_gold_item(
            it, cmap, edges=_as_edge_list(edges), entities_by_id=by_id
        )
        if errs:
            failures.append({"question_id": it.get("question_id"), "errors": errs})

    duplicates: list[dict] = find_duplicate_questions(all_items)

    report = {
        "total_requested": total,
        "total_actual": len(all_items),
        "quotas": quotas,
        "produced": produced,
        "gaps": {c: quotas[c] - produced[c] for c in CATEGORIES if quotas[c] > produced[c]},
        "share_actual": {
            c: (round(produced[c] / len(all_items), 4) if all_items else 0.0)
            for c in CATEGORIES
        },
        "validation_failures": failures,
        "duplicate_questions": duplicates,
    }
    return {
        "schema": "ark-goldset-1",
        "seed": seed,
        "total": len(all_items),
        "quotas": quotas,
        "report": report,
        "items": all_items,
    }


# ---------------------------------------------------------------- 存取


def default_path() -> Path:
    """默认 Gold Set 路径（`data/goldset/gold-v1.json`，不在冻结产物区内）。

    用 `config.ROOT` 拼绝对路径：`data/arkrag/` 是冻结区，评测集不该写进去；
    这里刻意独立成 `data/goldset/`。
    """
    return config.ROOT / DEFAULT_GOLD_PATH


def save_gold_set(gold: dict, path: str | Path | None = None) -> Path:
    """原子写出 Gold Set。返回实际写入路径。"""
    target = Path(path) if path is not None else default_path()
    atomic.atomic_write_json(target, gold, indent=2)
    return target


def load_gold_set(path: str | Path | None = None) -> dict:
    """读回 Gold Set（UTF-8 JSON）。"""
    target = Path(path) if path is not None else default_path()
    return json.loads(target.read_text("utf-8"))


__all__ = [
    "ABSENT_LABELS",
    "CATEGORIES",
    "CATEGORY_ABBR",
    "CATEGORY_SHARE",
    "DEFAULT_GOLD_PATH",
    "DIFFICULTIES",
    "NEGATION_WORDS",
    "QUESTION_ID_RE",
    "REQUIRED_FIELDS",
    "TEMPLATES",
    "build_gold_set",
    "default_path",
    "load_gold_set",
    "mine_adversarial",
    "mine_alias_questions",
    "mine_cross_chunk",
    "mine_direct_relations",
    "mine_multihop",
    "mine_negative_and_unanswerable",
    "mine_single_chunk_facts",
    "mine_time_conflict",
    "find_duplicate_questions",
    "save_gold_set",
    "validate_gold_item",
]
