"""分题型的答案判据：**唯一实现**。

## 为什么必须集中在一处
本轮评测里同一个"答对"被三种口径算过三遍，结论方向不同：
  * 「答案里出现 gold 串」——`MAX_CLAIMS` 5→12 时 dev +15 题，但严判据下只 +1 题，
    多出来的部分是"多堆几条断言、撞上高频专名"（`隶属` 出现在 102/350 题的答案里）。
  * `cross_chunk` 用「答案章节 ⊇ gold」——12000 档比 6000 档高 7 题；
    用「集合相等」时反过来低 1 题。**超集判据把"多列章节"记成了正确**，
    实测 12000 档的 46 个多列章节按语料正文判全部"确实出现过"，
    但按 `goldset.mine_cross_chunk` 的建题规则（`entity["chunks"]` ∩ 正文含该名）
    它们不属于 gold —— 也就是说超集判据在奖励"换了个定义作答"。
  * 「问句里自带的答案」——`single_chunk` 76 题里 46 题的 gold 就写在题干书名号里，
    零检索把标题抄出来就能过 46/76。任何检索侧的增益都必须减掉这个地板再报。

所以判据不能各脚本各写一份（那正是"同一判定写两处"的老事故），
这里给一份唯一的、逐类可机检的 `judge_answer()`，并配 `tests/test_judge.py`。

## 逐类判据（每一类的口径都必须能说清"为什么不是别的写法"）
| 类别 | 判据 | 为什么不能更松 |
|---|---|---|
| `single_chunk` / `alias` | gold 或某个 `acceptable_answers` 出现在答案里 | 这两类问的就是"这一句出自哪个块/这个称呼的规范名是什么"，gold 是短专名，子串命中即等价 |
| `cross_chunk` | gold ⊆ 答案；每个额外章节**必须真实提及**该实体；且**额外章节数不得超过 gold 章节数** | 只用「集合相等」会惩罚正确答案：`goldset.mine_cross_chunk` 只遍历 `entity["chunks"]` 且要求正文含**规范名**，图登记不全时 gold 系统性漏章节（实测 64 题里 44 题有真实提及章节被漏，共 317 章）。但只放开成「超集 + 额外真实」又会被普通名词打满：`海洋` 从 gold 的 3 章膨胀到 47 章（extra/gold = 14.7）、`巫术` 15.0、`集市` 12.5、`银行` 11.0。规模上限 `|额外| <= |gold|` 是同时挡住"列全部 494 章"和"列全部真实提及章节"的最小约束 |
| `direct_relation` | **一条**断言同时含**全部**问句锚点 + gold | 只要求"含任一锚点 + gold"时，`q-dr-900010` 问「红」与「罗德岛」会被「温蒂隶属罗德岛」判过——主语完全不是问的那个 |
| `multihop` | gold 与**每一个**问句锚点**各自**有一条断言共现（可分布在不同断言里），且 gold 本身不是锚点 | 只要求"≥1 锚点"时，实测 c12_12000 通过的 73 题里 **57 题只命中 1 个锚点**，没有一题体现"经中间实体连接两端"：`q-mh-000001` 问"从「云兽」经一中间实体到「变形者集群」"，答案里只有「阿米娅交谈变形者集群」，`云兽` 一个字都没出现 |
| `time_conflict` / `adversarial` / `negative_abstain` | 不给分数，返回 `structural` | 这三类要的是"时序演进""不是同一实体""不存在该关系"这类判断，抽取式答案器结构上产不出，硬判只会让分母虚高。**注意：这不等于"这类题无解"**——实测管线输出里已有 43/65 = 66.2% 的判别信号（`_same_scene_conflicts` 已算出"不同 doc = 剧情演进"、`negated_hint` 已置位、type 相异已在 entity_candidates 里），只是没被接进答案。剔除分母是为了"不假装它答对了"，不是为了"宣布它做不到" |

## 引用支撑是另一件事
`citations_support()` 单独判"每条引用是否落在它声称的那个章节、且偏移逐字对上"。
它不并入 `judge_answer()`：判据里混进引用检查就分不清"答错了"和"引用写错了"。
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "CATEGORIES",
    "STRUCTURAL_CATEGORIES",
    "SCORABLE_CATEGORIES",
    "anchors",
    "chapter_set",
    "citations_support",
    "judge_answer",
    "short_anchors",
]

#: 八类题目的类别名（与 `goldset.CATEGORIES` 同序，但这里不 import 它：
#: `goldset` 依赖 `identity`/`vocab`，判据不该把整条挖掘链拖进来）。
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

#: 抽取式答案器结构上产不出答案的类别：不给分数，只报"产出了非空答案没有"。
STRUCTURAL_CATEGORIES: tuple[str, ...] = (
    "time_conflict",
    "negative_abstain",
    "adversarial",
)

#: 能打分的那几类。
SCORABLE_CATEGORIES: tuple[str, ...] = tuple(
    c for c in CATEGORIES if c not in STRUCTURAL_CATEGORIES
)

#: 章节代号的分隔符。`cross_chunk` 的 `acceptable_answers` 里同时出现过
#: `、`、`,`、`，`、`/` **和空格**（例：`act33side main_ep14`）。第一版只按前四个切，
#: 于是要求答案里出现整串 `act33side main_ep14`，38 道题全判失败——那是判据的错。
_CHAPTER_SPLIT = re.compile(r"[、,，/／\s]+")

#: 问句里的锚点：`「」` 里的内容。
_ANCHOR_RE = re.compile(r"「([^」]{1,30})」")

#: 断言分隔符：答案是把断言用「。」接起来的串。
_SENTENCE_SPLIT = "。"


def _norm(text: object) -> str:
    """全角/半角归一后去首尾空白。判据里所有字面比较都过这一道。"""
    return unicodedata.normalize("NFKC", str(text or "")).strip()


def chapter_set(text: object) -> set[str]:
    """把一串章节代号切成集合。

    空串返回空集（调用方必须自己决定空集算不算通过——本模块里空集一律判失败）。
    """
    return {x for x in _CHAPTER_SPLIT.split(_norm(text)) if x}


def anchors(question: object) -> list[str]:
    """问句里 `「」` 括起来的锚点，按出现顺序去重。"""
    out: list[str] = []
    for m in _ANCHOR_RE.finditer(_norm(question)):
        a = m.group(1).strip()
        if a and a not in out:
            out.append(a)
    return out


def short_anchors(question: object, max_chars: int = 3) -> list[str]:
    """短锚点（≤`max_chars` 字）。

    `negative_abstain` 的关系名（「婚约」「祖孙」「邻居」）长度都在 3 字以内，
    而实体名基本更长。这个函数只供诊断用，不参与判分。
    """
    return [a for a in anchors(question) if len(a) <= max_chars]


def _sentences(text: str) -> list[str]:
    return [s for s in text.split(_SENTENCE_SPLIT) if s.strip()]


def _gold_strings(item: dict) -> list[str]:
    gold = _norm(item.get("answer"))
    out = [gold] if gold else []
    for x in item.get("acceptable_answers") or ():
        v = _norm(x)
        if v and v not in out:
            out.append(v)
    return out


def judge_answer(category: str, answer_text: object, item: dict,
                 *, question: object | None = None,
                 chapter_texts: dict | None = None) -> dict:
    """判一道题答对没有。

    返回 `{"pass": bool | None, "reason": str}`。`pass is None` 表示这类题
    本判据不打分（结构类），调用方必须把它从分母里剔除、单列，不许当 0 或当 1。

    `answer_text` 为空（含纯空白）时一律返回 `pass=False, reason="empty"`——
    **拒答不算答对**，哪怕 gold 也是空串。

    `chapter_texts` 只在 `cross_chunk` 用得上：`{章节代号: 该章全部块正文拼接}`。
    判"额外章节是不是真的提到这个实体"必须查语料正文，所以调用方要把它传进来。
    **不传时 `cross_chunk` 只能做集合相等**（并且 `reason` 会写明降级原因）——
    这是刻意的：不许在没查正文的情况下假装做过真实性校验。
    """
    text = _norm(answer_text)
    if not text:
        return {"pass": False, "reason": "empty"}
    cat = str(category or "")
    if cat in STRUCTURAL_CATEGORIES:
        return {"pass": None, "reason": "structural:not_supported"}
    golds = _gold_strings(item)
    if cat in ("single_chunk", "alias"):
        if not golds:
            return {"pass": False, "reason": "no_gold"}
        if any(g in text for g in golds):
            return {"pass": True, "reason": "gold_in_answer"}
        return {"pass": False, "reason": "gold_absent"}
    if cat == "cross_chunk":
        want = chapter_set(item.get("answer"))
        got = chapter_set(answer_text)
        if not want:
            return {"pass": False, "reason": "no_gold"}
        if got == want:
            return {"pass": True, "reason": "chapter_set_equal"}
        if not (want <= got):
            return {"pass": False, "reason": "chapter_set_missing"}
        extra = got - want
        if chapter_texts is None:
            return {"pass": False, "reason": "extra_chapters_unverified"}
        q = _norm(question if question is not None else item.get("question"))
        name = next((a for a in anchors(q) if a and a not in want), "")
        if not name:
            return {"pass": False, "reason": "no_entity_to_verify_extra"}
        if not all(ch in chapter_texts and name in chapter_texts[ch] for ch in extra):
            return {"pass": False, "reason": "extra_chapter_does_not_mention_entity"}
        if len(extra) > len(want):
            return {"pass": False, "reason": "extra_chapters_outnumber_gold"}
        return {"pass": True, "reason": "extra_chapters_all_real_and_bounded"}
    q = _norm(question if question is not None else item.get("question"))
    anc = anchors(q)
    gold = golds[0] if golds else ""
    sents = _sentences(text)
    if cat == "direct_relation":
        if not anc or not gold:
            return {"pass": False, "reason": "no_anchor_or_gold"}
        for s in sents:
            if all(a in s for a in anc) and gold in s:
                return {"pass": True, "reason": "one_claim_all_anchors_and_gold"}
        return {"pass": False, "reason": "no_single_claim_with_all_anchors"}
    if cat == "multihop":
        if not anc or not gold:
            return {"pass": False, "reason": "no_anchor_or_gold"}
        if gold in anc:
            return {"pass": False, "reason": "gold_is_an_anchor"}
        covered = [a for a in anc if any(a in s and gold in s for s in sents)]
        if not covered:
            return {"pass": False, "reason": "no_claim_links_gold_to_any_anchor"}
        if len(covered) < len(anc):
            return {"pass": False, "reason": "gold_not_linked_to_every_anchor"}
        return {"pass": True, "reason": "gold_linked_to_every_anchor"}
    return {"pass": False, "reason": f"unknown_category:{cat}"}


def citations_support(answer: dict, chunks: dict, chapters: object = None) -> dict:
    """每条引用是否落在它声称的章节、且偏移逐字对上。

    返回 `{ok, n_cites, problems, chapters_covered}`。
    `chapters` 给定时额外要求"这些章节里每一章至少有一条合格引用"——
    这是 `cross_chunk` 的引用支撑条件（答案列出 N 章就要有 N 条引用，
    否则解题者读完给出的块也推不出被列进去的那一章）。
    """
    cites = list(answer.get("citations") or [])
    problems: list[dict] = []
    covered: set[str] = set()
    for i, c in enumerate(cites):
        cid = str(c.get("chunk_id") or "")
        rec = chunks.get(cid) if isinstance(chunks, dict) else None
        if rec is None:
            problems.append({"index": i, "chunk_id": cid, "reason": "unknown_chunk"})
            continue
        text = str(rec.get("text") or "")
        ev = str(c.get("evidence") or "")
        start, end = c.get("start"), c.get("end")
        if start is None or end is None:
            problems.append({"index": i, "chunk_id": cid, "reason": "offsets_missing"})
            continue
        try:
            start, end = int(start), int(end)
        except (TypeError, ValueError):
            problems.append({"index": i, "chunk_id": cid, "reason": "offsets_invalid"})
            continue
        if not (0 <= start < end <= len(text)):
            problems.append({"index": i, "chunk_id": cid, "reason": "offsets_out_of_range"})
            continue
        if text[start:end] != ev:
            problems.append({"index": i, "chunk_id": cid,
                             "reason": "evidence_does_not_match_offsets"})
            continue
        covered.add(str(rec.get("chapter") or ""))
    want = chapter_set(chapters) if chapters is not None else None
    missing = sorted(want - covered) if want else []
    if missing:
        problems.append({"reason": "chapters_without_citation", "chapters": missing})
    return {
        "ok": not problems,
        "n_cites": len(cites),
        "problems": problems,
        "chapters_covered": sorted(x for x in covered if x),
    }
