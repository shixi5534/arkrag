"""L4 关系归一：十步裁决（顺序本身就是设计）+ 逆谓词方向裁决。

最容易写错的一点：第 3 步「否定前缀丢弃」必须在第 9 步「最长子串归约」之前，
否则「不隶属于」会被归约成「隶属」——把否定关系写成肯定关系，是事实错误而不是格式问题。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import vocab

RE_PUNCT = re.compile(r"[。！？；，、：:!?;,\u3000…—－\"'“”‘’()]")
RE_LATIN = re.compile(r"[A-Za-z]")
RE_DIGIT = re.compile(r"[0-9]")
PASSIVE_PREFIX = re.compile(r"^(被|受|遭|遇|为)")
LOCATIVE_SUFFIX = re.compile(r"(于|到|往|给|自|向|着|了|过|成|为)$")

# 需要换端点的归一：源词描述的是「反方向」的事实。师从/指导 两个方向都在词表里，
# 所以只有当自由谓语被归约到反向谓词时才需要换端点——这正是最容易写反的地方。
SWAP_MAP: dict[str, str] = {
    "师从": "指导",
    "是……的学生": "指导",
    "受业于": "指导",
}
# 方向语义一致、可以同向并入的白名单
SAME_DIRECTION_OK = {"隶属于", "从属于", "归属", "属于", "代表", "源自"}

JUNK_SET = set(vocab.JUNK_RELATIONS)
CANON_SET = set(vocab.CANONICAL_RELATIONS)
NEG_HEAD = vocab.NEGATION_PREFIX
# 否定不只有一种写法：拒绝/无法/未能 这类否定动词同样会把事实整个反过来，
# 而它们不在单字否定前缀里，必须单独拦，否则「拒绝帮助」会被归约成「救助」。
NEG_VERBS = (
    "拒绝", "无法", "未能", "不能", "不会", "不再", "没有", "没", "未", "勿", "避免",
    "防止", "放弃", "不愿", "不肯", "禁止", "难以", "未曾", "绝不", "并非",
)


def _is_negative(text: str) -> bool:
    return text.startswith(NEG_HEAD) or text.startswith(NEG_VERBS)


@dataclass
class Decision:
    raw: str
    canonical: str
    action: str  # adopt / drop / keep_flagged
    flag: str
    swap: bool = False


def _longest_substr_reduction(raw: str) -> tuple[str, str]:
    """剥被动前缀与方位后缀后，在「词表名子串」与「同义词键子串」里取**匹配更长**的那个。

    比较的是匹配长度而不是目标词长度：「从属于罗德岛」里 从属于(3)->隶属 必须赢过 属于(2)->属于，
    否则势力归属会被归到所有权关系上。
    """
    core = PASSIVE_PREFIX.sub("", raw)
    core = LOCATIVE_SUFFIX.sub("", core)
    if not core:
        return "", ""
    if core in CANON_SET:
        return core, "reduced_exact"
    best_len, best_target, best_how = 0, "", ""
    for name in CANON_SET:
        if name in core and len(name) > best_len:
            best_len, best_target, best_how = len(name), name, "reduced_substr"
    for key, val in vocab.RELATION_SYNONYMS.items():
        if val in CANON_SET and key in core and len(key) > best_len:
            best_len, best_target, best_how = len(key), val, "reduced_synonym_substr"
    if not best_target:
        return "", ""
    for _ in range(2):
        nxt = vocab.RELATION_SYNONYMS.get(best_target)
        if nxt and nxt in CANON_SET:
            best_target = nxt
        else:
            break
    return best_target, best_how


def adjudicate(raw: object, source: str = "", target: str = "") -> Decision:
    if not isinstance(raw, str) or not raw.strip():
        return Decision(str(raw), "", "drop", "empty")
    text = raw.strip()
    if text in JUNK_SET:
        return Decision(text, "", "drop", "junk_word")
    if _is_negative(text):
        # 必须在规则归约之前判定，否则「不隶属于」被救成「隶属」、「拒绝帮助」被救成「救助」
        return Decision(text, "", "drop", "negation")
    if RE_PUNCT.search(text):
        return Decision(text, text, "keep_flagged", "has_punct")
    if RE_LATIN.search(text) or RE_DIGIT.search(text):
        return Decision(text, text, "keep_flagged", "has_latin_or_digit")
    if len(text) > vocab.MAX_RELATION_CHARS:
        return Decision(text, text, "keep_flagged", "overlong")
    if text in CANON_SET:
        return Decision(text, text, "adopt", "exact")
    hit = vocab.RELATION_SYNONYMS.get(text)
    if hit:
        for _ in range(2):
            nxt = vocab.RELATION_SYNONYMS.get(hit)
            hit = nxt if nxt and nxt in CANON_SET else hit
        if hit in CANON_SET:
            return _direction_guard(Decision(text, hit, "adopt", "synonym_exact"), source, target)
    reduced, how = _longest_substr_reduction(text)
    if reduced:
        return _direction_guard(Decision(text, reduced, "adopt", how), source, target)
    return Decision(text, text, "keep_flagged", "unmapped")


def _direction_guard(dec: Decision, source: str, target: str) -> Decision:
    """归一可能把方向写反：需要换端点而目标在方向可疑名单里时，只有同向白名单放行。"""
    canonical = dec.canonical
    swap_needed = SWAP_MAP.get(dec.raw) == canonical and dec.raw != canonical
    if swap_needed:
        if dec.raw in SAME_DIRECTION_OK:
            return dec
        if canonical in vocab.DIRECTION_WATCH:
            dec.action = "keep_flagged"
            dec.flag = "direction_blocked"
            dec.canonical = dec.raw
            dec.swap = False
            return dec
        dec.swap = True
    return dec


def is_mutually_exclusive(a: str, b: str) -> bool:
    for x, y in vocab.MUTUALLY_EXCLUSIVE_PAIRS:
        if {a, b} == {x, y}:
            return True
    return False


def inverse_of(name: str) -> str | None:
    for a, b in vocab.INVERSE_PAIRS:
        if name == a:
            return b
        if name == b:
            return a
    return None
