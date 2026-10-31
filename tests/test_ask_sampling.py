"""`scripts/ask.py --limit` 的分层抽样。

为什么单独立一个文件守它：`--limit` 的实现曾经是 `items[:limit]`，而
`gold-v1.json` 里 `single_chunk` 排在最前（64 条），于是 `--limit 40` /
`--limit 50` 实测只覆盖 `single_chunk` 一个类别。报告照常打印七项指标，看着
像"局部很好"，而多跳、对抗、时间冲突一条都没跑。**评测入口少跑几个类别不会
报错，只会让数字变好看**——这类缺陷必须由测试守，不能靠人记得。

本文件只测抽样逻辑本身（纯函数、不读盘、不建索引）。
"""

from __future__ import annotations

import importlib.util
import random
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _ask_module():
    """按路径加载 `scripts/ask.py`（它不是包的一部分）。"""
    path = ROOT / "scripts" / "ask.py"
    spec = importlib.util.spec_from_file_location("ask_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ask():
    return _ask_module()


# 与真实 gold-v1.json 同形的配额（八类 64/48/48/48/32/32/32/16 = 320）
QUOTAS = {
    "single_chunk": 64,
    "alias": 48,
    "direct_relation": 48,
    "multihop": 48,
    "cross_chunk": 32,
    "time_conflict": 32,
    "negative_abstain": 32,
    "adversarial": 16,
}


def _items(quotas: dict[str, int] | None = None) -> list[dict]:
    quotas = dict(quotas or QUOTAS)
    out: list[dict] = []
    for cat in quotas:  # 保持"某一类排最前"的原始形状
        for i in range(quotas[cat]):
            out.append(
                {
                    "question_id": f"q-{cat}-{i:06d}",
                    "category": cat,
                    "question": f"{cat} 第 {i} 题",
                }
            )
    return out


class TestEveryCategoryIsRepresented:
    def test_small_limit_does_not_collapse_to_one_category(self):
        items = _items()
        got = _ask_module()._stratified_sample(items, 40)
        cats = Counter(x["category"] for x in got)
        assert len(cats) == len(QUOTAS), f"只覆盖了 {sorted(cats)}"

    @pytest.mark.parametrize("limit", [8, 16, 40, 50, 120, 200])
    def test_all_categories_present_for_any_limit_above_category_count(self, limit):
        ask = _ask_module()
        got = ask._stratified_sample(_items(), limit)
        assert len(Counter(x["category"] for x in got)) == len(QUOTAS)

    def test_rarest_category_is_never_dropped(self):
        """`adversarial` 只有 16 条，占比 5% —— 最先被等比例取整砍掉的就是它。"""
        ask = _ask_module()
        for limit in (16, 20, 40, 50):
            got = ask._stratified_sample(_items(), limit)
            assert any(x["category"] == "adversarial" for x in got), limit


class TestQuotaProportions:
    def test_full_limit_returns_every_item_in_original_order(self):
        ask = _ask_module()
        items = _items()
        got = ask._stratified_sample(items, len(items))
        assert [x["question_id"] for x in got] == [x["question_id"] for x in items]

    def test_limit_above_total_returns_everything(self):
        ask = _ask_module()
        items = _items()
        assert len(ask._stratified_sample(items, 10_000)) == len(items)

    def test_counts_sum_to_limit(self):
        ask = _ask_module()
        items = _items()
        for limit in (8, 40, 50, 120, 200, 319):
            got = ask._stratified_sample(items, limit)
            assert len(got) == limit, limit

    def test_proportions_are_close_to_the_pool(self):
        ask = _ask_module()
        got = ask._stratified_sample(_items(), 160)
        cats = Counter(x["category"] for x in got)
        for cat, total in QUOTAS.items():
            expected = 160 * total / 320
            # 保底 1 条 + 取整会让小类别略微偏高，允许 ±2 条
            assert abs(cats[cat] - expected) <= 2, (cat, cats[cat], expected)

    def test_no_duplicates(self):
        ask = _ask_module()
        got = ask._stratified_sample(_items(), 120)
        ids = [x["question_id"] for x in got]
        assert len(ids) == len(set(ids))


class TestDeterminism:
    def test_same_limit_twice_picks_the_same_items(self):
        """同一个 `--limit` 两次跑必须抽到同一批题，否则指标不可比。"""
        ask = _ask_module()
        a = [x["question_id"] for x in ask._stratified_sample(_items(), 50)]
        b = [x["question_id"] for x in ask._stratified_sample(_items(), 50)]
        assert a == b

    def test_different_seed_picks_a_different_subset(self):
        ask = _ask_module()
        off = [x["question_id"] for x in ask._stratified_sample(_items(), 50)]
        on = [x["question_id"] for x in ask._stratified_sample(_items(), 50, seed=1)]
        assert off != on  # 打散确实生效，不是永远取每类前几条

    def test_shuffle_does_not_mutate_input_order(self):
        ask = _ask_module()
        items = _items()
        before = [x["question_id"] for x in items]
        ask._stratified_sample(items, 50)
        assert [x["question_id"] for x in items] == before


class TestEdgeCases:
    def test_limit_one_returns_one_item(self):
        ask = _ask_module()
        assert len(ask._stratified_sample(_items(), 1)) == 1

    def test_zero_or_negative_is_clamped_to_one(self):
        ask = _ask_module()
        assert len(ask._stratified_sample(_items(), 0)) == 1
        assert len(ask._stratified_sample(_items(), -5)) == 1

    def test_missing_category_field_is_tolerated(self):
        """类别字段缺失/为空时不能崩——归到一个空串桶里正常参与配额。"""
        ask = _ask_module()
        items = _items({"single_chunk": 10})
        items.append({"question_id": "q-x", "question": "没有 category"})
        got = ask._stratified_sample(items, 11)
        assert len(got) == 11

    def test_single_category_pool_is_honest(self):
        """池里只有一类时，抽出来的自然也只有一类——这是如实结果，不是缺陷。"""
        ask = _ask_module()
        got = ask._stratified_sample(_items({"only": 20}), 5)
        assert len(got) == 5
        assert {x["category"] for x in got} == {"only"}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
