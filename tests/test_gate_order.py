"""闸门判定必须按因果顺序：检索 → 准入 → 回答。

## 这条门在防什么

`scripts/lead_gate_recheck.py::classify_gate()` 给引用覆盖分六态。第一版把
`admit`（准入判据过没过）判在 `incl`（required 块进没进上下文）**之前**，
于是「块根本没进上下文」的题，只要那条断言也恰好过不了准入判据，就被记成
**闸门1b（抽到了但被相关度挡掉）**。

这是错的，而且错得很有害：

* 块不在上下文时，`build_context` 的第 1 步**根本没遍历到那个块**，
  `retrieve._assertion_relevant()` 一次都没被调用。说它「被准入判据挡掉」
  是给一件没发生的事编因果。
* 后果是把 P2 的活（检索没把块送进来，要改图扩展/排序）算进 P3 的口径
  （改准入判据），分母也跟着虚增：v1 300 应答题上 29 vs 19，多出 10 道。
  这 10 道里有 5 道 `multihop`（`q-mh-000010` / `-000015` / `-000017` /
  `-000021` / `-000043`），实测 required 块 `in_context = False`。

发现过程值得记：我用「29 题」去宣布旧分母 `51/72 = 0.7083` 作废、改成
`51/82 = 0.6220`；A 用因果顺序重排后复现出 25 题（dev 350）并指出 5 道
multihop 属闸门2，我据此修顺序重跑得 19 —— **19 + 2 + 51 = 72，旧分母本来是对的**。
一个「新拆出来的集合恰好让旧分母变成错的」时，要先怀疑新集合本身。

## 两条判据

* 块不在上下文 → 无论准入判据结果如何，都必须是**闸门2**。
* 块在上下文、断言过不了准入 → 必须是**闸门1b**（这才是「抽到了被挡掉」）。

## 怎么加载脚本

与 `tests/test_ask_generation_selection.py` / `tests/test_baseline_cache_inventory.py`
同法：`importlib.util.spec_from_file_location`。`scripts/` 不是包，不能 `import scripts.x`。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from arkrag import config

SCRIPTS = config.ROOT / "scripts"


def _load(name: str, filename: str):
    path = SCRIPTS / filename
    assert path.is_file(), f"脚本不存在：{path}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gate():
    return _load("lead_gate_recheck_under_test", "lead_gate_recheck.py")


# 一条断言：端点都不在焦点实体里、谓词也不在关系提示里 ⇒ 准入判据必定不过。
_BLOCKED_ASSERTION = {
    "assertion_id": "a-test",
    "chunk_id": "c1",
    "subject_id": "e-subject",
    "object_id": "e-object",
    "predicate": "交谈",
    "evidence_text": "我警告过你们不许进入森林。",
    "evidence_start": 0,
    "evidence_end": 13,
}

_BLOCK_TEXT = "我警告过你们不许进入森林。\n下一行。"

_GOLD_ITEM = {
    "question_id": "q-test",
    "category": "single_chunk",
    "question": "「我警告过你们不许进入森林。」出自哪个文本块？请给出该块的标题。",
    "answer": "某标题",
    "gold_chunk_ids": ["c1"],
    "required_citations": [{"chunk_id": "c1", "evidence": "我警告过你们不许进入森林。"}],
    "should_abstain": False,
}

_NO_CITATIONS = {"citations": [], "answer": "", "abstained": False}


def _classify(gate, *, kept, focus=frozenset({"e-elsewhere"}), hints=frozenset()) -> str:
    by_chunk = {cid: [_BLOCKED_ASSERTION] for cid in ("c1",)}
    chunks = {cid: {"text": _BLOCK_TEXT, "title": "某标题"} for cid in ("c1",)}
    return gate.classify_gate(
        _NO_CITATIONS, _GOLD_ITEM, by_chunk, chunks, set(focus), set(hints), set(kept)
    )


def test_block_missing_from_context_is_gate2_even_when_admission_also_fails(gate):
    """块不在上下文 + 断言过不了准入 ⇒ 必须记闸门2（先判 incl）。

    这条就是第一版记错的那一类：两个条件同时成立时被记成闸门1b，于是
    「检索没送进来」被记成「准入判据挡掉」，P2 的活算进 P3。
    """
    assert _classify(gate, kept=set()) == "闸门2 块没进上下文"


def test_block_in_context_with_blocked_assertion_is_gate1b(gate):
    """块在上下文、断言过不了准入 ⇒ 这才是闸门1b。反例侧控制。"""
    assert _classify(gate, kept={"c1"}) == "闸门1b 断言在但被相关度过滤"


def test_admission_passes_when_endpoint_is_a_focus_entity(gate):
    """端点落在焦点实体上时准入必须过 ⇒ 变成闸门3（块在手、断言在手、没发引用）。

    没有这一条，「闸门1b」可能只是「准入判据对谁都返回 False」的假象。
    """
    assert _classify(gate, kept={"c1"}, focus={"e-subject"}) == "闸门3 块在手没发引用"
