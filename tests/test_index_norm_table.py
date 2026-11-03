"""BM25 预建归一化表的等价性门：排序必须与旧实现**逐位相同**。

## 为什么这个文件必须存在

H2 把 `search()` 内层循环里的

    length = self.doc_len.get(slot, 1)
    ... idf * (freq*(k1+1)) / (freq + k1*(1 - b + b*length/avg_len))

改成预建的 `norm_table()` 查表。这是纯优化——**排序结果一个字都不许变**。

但它有一个非常安静的失效形态：`avg_len` 与 `norm_table` 是两份缓存，
如果 `add()` 之后只失效了 `_avg_len` 没失效 `_norm`，新加的文档会用**旧的**
平均长度参与打分。分数只差千分之几，排序大体不变，召回也只掉一两个位置——
没有任何异常，没有日志，`verify` 也不一定红。

所以这里不测"能跑"，测的是**与冻结的旧实现逐位相等**。下面的
`_legacy_search` 是改动前 `search()` 的逐字副本，注释已注明它必须保持原样：
它就是这个门的标尺，改动它等于把尺子和被测物一起换掉。

## 还要覆盖的边界

* 有空洞的 slot（`doc_len` 缺某些 slot → 旧实现落 `len=1` 的默认值）
* 单文档（`avg_len` 退化）
* 空索引（不许抛异常）
* `add()` 使缓存失效
* `load_index` 往返后仍等价（JSON 把 int key 变字符串那条路径）
"""
from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arkrag import index as index_mod  # noqa: E402
from arkrag.index import LexicalIndex, tokenize  # noqa: E402


def _legacy_search(index: LexicalIndex, query: str, top_k: int = 20):
    """改动前 `LexicalIndex.search` 的**逐字副本**，作为等价性标尺。

    不要"顺手优化"这个函数。它唯一的职责就是保持旧行为不变，好让新实现
    有东西可比。它慢、它重复计算，都是故意的。
    """
    terms = index.name_terms
    n = max(1, index.size)
    q_terms = Counter(tokenize(query, version=index.tokenizer_version, name_terms=terms))
    avg_len = index.avg_len
    k1, b = 1.5, 0.75
    scores: dict[int, float] = defaultdict(float)
    for term, tf in q_terms.items():
        posting = index.postings.get(term)
        if not posting:
            continue
        df = len(posting)
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        for slot, freq in posting.items():
            length = index.doc_len.get(slot, 1)
            scores[slot] += idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * length / avg_len))
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], index.doc_id[kv[0]]))[:top_k]
    return [(index.doc_id[slot], score) for slot, score in ranked]


# ---------------------------------------------------------------- 语料
CORPUS = [
    ("c-001", "凯尔希站在罗德岛的甲板上，看着远处的切尔诺伯格。"),
    ("c-002", "罗德岛的干员们在龙门执行了一次秘密行动。"),
    ("c-003", "阿米娅是罗德岛的领袖，凯尔希是她的顾问。"),
    ("c-004", "整合运动的成员袭击了切尔诺伯格的核心城区。"),
    ("c-005", "博士在指挥中心调度了这次行动，感染者的数量超出预期。"),
    ("c-006", "乌萨斯的军队在边境集结，局势紧张。"),
    ("c-007", "罗德岛制药公司研发了一种新的抑制剂。"),
    ("c-008", "凯尔希拒绝了乌萨斯的提议，坚持留在罗德岛。"),
    ("c-009", "龙门的警司与罗德岛达成了临时合作。"),
    ("c-010", "ACT16 的记录显示感染者比例上升。"),
]

QUERIES = [
    "凯尔希和罗德岛的关系",
    "罗德岛在龙门做了什么",
    "整合运动袭击切尔诺伯格",
    "感染者",
    "乌萨斯 军队",
    "ACT16",
    "不存在的词汇在这句话里",
    "",
]


def _build(name_terms=None) -> LexicalIndex:
    idx = LexicalIndex()
    if name_terms:
        idx.set_name_terms(name_terms)
    for slot, (cid, text) in enumerate(CORPUS):
        idx.add(slot, cid, text, name_terms=name_terms)
    return idx


@pytest.fixture(scope="module")
def idx() -> LexicalIndex:
    return _build()


class TestSortOrderIsIdentical:
    """核心门：新实现的输出必须与旧实现逐位相同（含 score 的浮点值）。"""

    @pytest.mark.parametrize("query", QUERIES, ids=lambda q: q or "empty")
    @pytest.mark.parametrize("top_k", [5, 20, 100])
    def test_identical_to_legacy(self, idx: LexicalIndex, query: str, top_k: int) -> None:
        new = idx.search(query, top_k=top_k)
        old = _legacy_search(idx, query, top_k=top_k)
        assert new == old, (
            f"问句 {query!r} top_k={top_k} 排序或分数变了。\n"
            f"新实现：{new}\n"
            f"旧实现：{old}\n"
            f"这是纯优化，输出必须逐位相同。"
        )

    def test_scores_are_bitwise_equal(self, idx: LexicalIndex) -> None:
        """单独比 float 的 repr，防止 `==` 掩盖最后一位差异。"""
        for query in QUERIES:
            new = idx.search(query, top_k=100)
            old = _legacy_search(idx, query, top_k=100)
            assert [repr(s) for _, s in new] == [repr(s) for _, s in old], (
                f"{query!r} 的分数在浮点层面不同了"
            )


class TestNormTableContent:
    def test_matches_formula(self, idx: LexicalIndex) -> None:
        norm = idx.norm_table()
        avg_len = idx.avg_len
        k1, b = index_mod.BM25_K1, index_mod.BM25_B
        for slot, length in idx.doc_len.items():
            expected = k1 * (1 - b + b * length / avg_len)
            assert norm[slot] == pytest.approx(expected, rel=0, abs=0), (
                f"slot {slot} 的归一化项算错：{norm[slot]} vs {expected}"
            )

    def test_constants_are_the_documented_values(self) -> None:
        """参数值本身也要锁住：改了它们等于改所有世代的排序。"""
        assert index_mod.BM25_K1 == 1.5
        assert index_mod.BM25_B == 0.75

    def test_is_cached(self, idx: LexicalIndex) -> None:
        first = idx.norm_table()
        second = idx.norm_table()
        assert first is second, "norm_table 每次都重建，预建的意义就没了"


class TestCacheInvalidation:
    """`add()` 之后两张缓存都必须失效——这是最安静的失效形态。"""

    def test_add_invalidates_norm_table(self) -> None:
        idx = _build()
        before = idx.norm_table()
        # 加一个超长文档，把 avg_len 显著拉高
        idx.add(99, "c-099", "很长" * 500)
        after = idx.norm_table()
        assert after is not before, "add() 后 norm_table 没失效，新文档会用旧平均长度打分"

    def test_add_keeps_equivalence_with_legacy(self) -> None:
        idx = _build()
        idx.add(99, "c-099", "很长" * 500)
        for query in QUERIES:
            assert idx.search(query, top_k=100) == _legacy_search(idx, query, top_k=100), (
                f"{query!r} 在 add() 之后与旧实现分歧了"
            )

    def test_avg_len_still_invalidates(self) -> None:
        idx = _build()
        first = idx.avg_len
        idx.add(99, "c-099", "很长" * 500)
        assert idx.avg_len != first


class TestEdgeCases:
    def test_empty_index_does_not_raise(self) -> None:
        idx = LexicalIndex()
        assert idx.norm_table() == []
        assert idx.search("任何问句") == []

    def test_empty_index_legacy_agrees(self) -> None:
        idx = LexicalIndex()
        assert idx.search("任何问句") == _legacy_search(idx, "任何问句")

    def test_slot_gaps_fall_back_to_one(self) -> None:
        """`doc_len` 缺某个 slot 时，旧实现落 `len=1` 的默认值，新实现必须一致。"""
        idx = LexicalIndex()
        idx.add(0, "c-000", "凯尔希")
        idx.add(5, "c-005", "罗德岛在龙门")
        # 人为制造空洞：槽位 1..4 完全没有 doc_len
        norm = idx.norm_table()
        assert len(norm) == 6
        expected_default = index_mod.BM25_K1 * (
            1 - index_mod.BM25_B + index_mod.BM25_B * 1 / idx.avg_len
        )
        for slot in range(1, 5):
            assert norm[slot] == pytest.approx(expected_default, rel=0, abs=0)

    def test_gap_equivalence_with_legacy(self) -> None:
        idx = LexicalIndex()
        idx.add(0, "c-000", "凯尔希站在罗德岛的甲板上")
        idx.add(5, "c-005", "罗德岛在龙门执行行动")
        for query in ("罗德岛", "凯尔希 龙门"):
            assert idx.search(query, top_k=10) == _legacy_search(idx, query, top_k=10)

    def test_single_document(self) -> None:
        idx = LexicalIndex()
        idx.add(0, "c-000", "凯尔希")
        assert idx.search("凯尔希", top_k=5) == _legacy_search(idx, "凯尔希", top_k=5)


class TestRoundTrip:
    """落盘再读回后仍等价——`load_index` 把 int key 从 JSON 字符串转回来那条路径。"""

    def test_round_trip_equivalence(self, tmp_path: Path) -> None:
        idx = _build()
        path = tmp_path / "lexical.json"
        index_mod.save_index(idx, {"tokenizer_version": "t2"}, path)
        loaded, _meta = index_mod.load_index(path)

        assert loaded.norm_table() is not None
        for query in QUERIES:
            assert loaded.search(query, top_k=100) == _legacy_search(
                loaded, query, top_k=100
            ), f"{query!r} 在落盘往返后与旧实现分歧"

    def test_round_trip_preserves_avg_len(self, tmp_path: Path) -> None:
        idx = _build()
        path = tmp_path / "lexical.json"
        index_mod.save_index(idx, {"tokenizer_version": "t2"}, path)
        loaded, _ = index_mod.load_index(path)
        assert loaded.avg_len == idx.avg_len


class TestPerformanceClaim:
    """把"预建更快"变成机检，否则下次重构可能悄悄退回每问重算。"""

    def test_prebuilt_is_faster_on_large_index(self) -> None:
        import time

        idx = LexicalIndex()
        # 造 2000 个文档，让内层循环有足够量级
        for slot in range(2000):
            idx.add(slot, f"c-{slot:05d}", f"凯尔希 罗德岛 龙门 行动 记录 {slot} " * 12)

        query = "凯尔希 罗德岛"
        idx.search(query, top_k=20)  # 预热
        _legacy_search(idx, query, top_k=20)

        n = 5
        t0 = time.perf_counter()
        for _ in range(n):
            _legacy_search(idx, query, top_k=20)
        legacy_ms = (time.perf_counter() - t0) * 1000 / n

        t1 = time.perf_counter()
        for _ in range(n):
            idx.search(query, top_k=20)
        new_ms = (time.perf_counter() - t1) * 1000 / n

        assert new_ms < legacy_ms, (
            f"预建归一化表反而更慢：新 {new_ms:.2f} ms / 旧 {legacy_ms:.2f} ms"
        )

    def test_benchmark_scales_with_index(self) -> None:
        """规范条目数与 postings 条目数成正比——预建的关键前提。"""
        idx = _build()
        norm = idx.norm_table()
        total_postings = sum(len(p) for p in idx.postings.values())
        assert len(norm) <= max(idx.doc_id) + 1
        assert total_postings >= len(norm), (
            "postings 条目数应远大于文档数，否则预建收益无从谈起"
        )
