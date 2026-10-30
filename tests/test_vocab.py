"""词表自检：结构错误必须在跑 L3 之前暴露，而不是在 L4 的门线上表现为不达标。"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arkrag import vocab  # noqa: E402

FAILURES: list[str] = []


def check(cond: bool, message: str) -> None:
    if not cond:
        FAILURES.append(message)


def main() -> int:
    canon = vocab.CANONICAL_RELATIONS
    dup = [k for k, v in Counter(canon).items() if v > 1]
    check(not dup, f"受控关系名重复: {dup}")
    check(len(set(vocab.ENTITY_TYPES)) == len(vocab.ENTITY_TYPES), "实体类型重复")

    for a, b in vocab.INVERSE_PAIRS:
        check(a in canon and b in canon, f"逆谓词对成员不在词表: {a}/{b}")
        check(a != b, f"逆谓词对两端同名: {a}")
    for r in vocab.SYMMETRIC_RELATIONS:
        check(r in canon, f"对称关系不在词表: {r}")
    for a, b in vocab.MUTUALLY_EXCLUSIVE_PAIRS:
        check(a in canon and b in canon, f"互斥关系不在词表: {a}/{b}")
    for r in vocab.DIRECTION_WATCH:
        check(r in canon, f"方向观察名单不在词表: {r}")

    # 逆谓词表要求「成对完整」：被列为一端的，另一端也必须能被同义词表捞到
    for src, dst in vocab.RELATION_SYNONYMS.items():
        check(dst in canon, f"同义词指向词表外: {src}->{dst}")
        check(src != dst, f"同义词自映射（应删，会让归约空转）: {src}")
        check(len(src) <= 10, f"同义词键过长，像句子不是关系: {src}")

    # 源文件级重复键：dict 字面量重复键会静默保留后者，必须显式抓出来
    text = Path("src/arkrag/vocab.py").read_text("utf-8")
    block = text.split("RELATION_SYNONYMS: dict[str, str] = {", 1)[1].split("\n}", 1)[0]
    keys = re.findall(r'^\s*"([^"]+)"\s*:', block, re.M)
    dup_keys = [k for k, v in Counter(keys).items() if v > 1]
    check(not dup_keys, f"同义词表存在重复键（后者静默覆盖前者）: {dup_keys}")

    latin = [k for k in vocab.RELATION_SYNONYMS if re.search(r"[A-Za-z]", k)]
    check(not latin, f"同义词表混入拉丁字母键: {latin}")

    for word in vocab.GENERIC_PERSON_NOUNS:
        check(len(word) <= 6, f"泛称词过长: {word}")
    generic_dup = [k for k, v in Counter(vocab.GENERIC_PERSON_NOUNS).items() if v > 1]
    check(not generic_dup, f"泛称表重复: {generic_dup}")

    for r in canon:
        check(2 <= len(r) <= vocab.MAX_RELATION_CHARS, f"关系名长度越界: {r}({len(r)})")

    if FAILURES:
        print("词表自检 FAIL:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print(
        f"词表自检 PASS  关系名 {len(canon)} 条 / 逆谓词对 {len(vocab.INVERSE_PAIRS)} / "
        f"对称 {len(vocab.SYMMETRIC_RELATIONS)} / 互斥 {len(vocab.MUTUALLY_EXCLUSIVE_PAIRS)} / "
        f"同义词 {len(vocab.RELATION_SYNONYMS)} 条 / 实体类型 {len(vocab.ENTITY_TYPES)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
