"""词表自检：结构错误必须在跑 L3 之前暴露，而不是在 L4 的门线上表现为不达标。

## 本文件为什么既有 `main()` 又有 `test_vocab()`

`check()` 曾经只往模块级 `FAILURES` 列表记、不抛异常，只有 `main()` 读它。
于是 `pytest tests/test_vocab.py` **收集到 0 个用例**却显示成功——它连这个
文件里有没有测试都不知道。这和 `tests/test_negative.py` 的假通过是同一类
问题（详见该文件 `expect()` 的 docstring）：CI 绿着，词表坏了没人知道。

现在 `check()` 直接抛 `AssertionError`，判据只有一份（`_check_all()`）：
`main()` 与 `test_vocab()` 都调它，两条运行路径不可能漂移。
`tests/test_suite_hygiene.py` 会机检这一点，防止有人把它改回记录式。
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import vocab  # noqa: E402


def check(cond: bool, message: str) -> None:
    """断言。失败时立刻抛异常——**不能只往列表里记一笔**。

    旧写法是 `if not cond: FAILURES.append(message)`，只有 `main()` 读它，
    于是 `pytest` 里这个文件能被收集、却没有任何断言真正生效。详见
    `tests/test_negative.py` 的 `expect()` 与 `tests/test_suite_hygiene.py`。
    """
    if not cond:
        raise AssertionError(message)


def _check_all() -> None:
    """全部词表检查。任一条不过就抛 AssertionError（带具体是哪一条）。"""
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

    # 源文件级重复键：dict 字面量重复键会静默保留后者，必须显式抓出来。
    # 用 ROOT 拼绝对路径而不是相对 cwd —— 换目录运行测试不该改变结论。
    text = (ROOT / "src" / "arkrag" / "vocab.py").read_text("utf-8")
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


def main() -> int:
    """命令行入口：跑一遍全部检查，打印结论，返回退出码。"""
    canon = vocab.CANONICAL_RELATIONS
    try:
        _check_all()
    except AssertionError as exc:
        print("词表自检 FAIL:")
        print("  -", exc)
        return 1
    print(
        f"词表自检 PASS  关系名 {len(canon)} 条 / 逆谓词对 {len(vocab.INVERSE_PAIRS)} / "
        f"对称 {len(vocab.SYMMETRIC_RELATIONS)} / 互斥 {len(vocab.MUTUALLY_EXCLUSIVE_PAIRS)} / "
        f"同义词 {len(vocab.RELATION_SYNONYMS)} 条 / 实体类型 {len(vocab.ENTITY_TYPES)}"
    )
    return 0


def test_vocab() -> None:
    """pytest 入口：跑同一套检查，让失败成为真正的测试失败。

    判据只有一份（`_check_all`），这里不做任何重复实现——否则两处判据迟早
    会漂移，而漂移的那一处总是更宽松的那一处。
    """
    _check_all()


def test_vocab_cli_exit_code_is_zero_on_clean_vocab() -> None:
    """`main()` 的退出码也必须是白的：CI 里常靠它判成败。"""
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
