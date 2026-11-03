"""`QueryContext` 的正确性门：复用预构建结构必须与就地重建**逐位相同**。

## 为什么这个文件必须存在

H2 给 `analyze_query` 加了一条旁路：传入 `QueryContext` 就跳过实体表/专名表/
关系表的重建，直接复用。这条旁路每问省约 98 ms（320 题省约 29 秒），是 H2 的
主要提速点——**但它同时是一个"结果可能悄悄变了"的改动**。

失效形态很难发现：`_alias_owner_map` 建错一个别名、`NameTable` 少一个两字前缀，
结果都不是异常，而是**某个问句的实体候选少一个**。下游照常检索、照常作答，
只是召回从 20 掉到 19。评测分数会动，但没人能从一个下降的小数点看出是这里坏的。

所以两条路必须能证明等价。下面用的是**生产形状**的语料：多实体、一对多别名、
别名与规范名同文、含拉丁名的实体、需要 trie 最长匹配的前缀嵌套名字（「罗德岛」
与「罗德岛制药公司」），以及会跟否定词/停用词打架的字符。合成三五个玩具实体
测不出这些坑。

## 判定方式

不是"两条路都返回 dict"就算通过——那什么都不证明。这里要求**整个返回结构
深度相等**：`entity_candidates` 的顺序、`score` 的取值、`stats` 里每一条计数
（尤其是 `name_table_terms` 与 `dropped_terms`）都必须一模一样。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arkrag import query  # noqa: E402


# ---------------------------------------------------------------- 语料
# 刻意构造的实体图：覆盖一对多别名、别名撞规范名、前缀嵌套、拉丁名。
ENTITIES = [
    {
        "id": "e-kaltsit",
        "name": "凯尔希",
        "aliases": [{"text": "凯医生"}, {"text": "老女人"}],
    },
    {
        "id": "e-rhodes",
        "name": "罗德岛",
        "aliases": [{"text": "岛"}, {"text": "罗德岛制药公司"}],
    },
    {
        # 别名与另一个实体的规范名同文 —— `_alias_owner_map` 必须保留一对多
        "id": "e-rhodes-corp",
        "name": "罗德岛制药公司",
        "aliases": [{"text": "制药公司"}],
    },
    {
        "id": "e-amiya",
        "name": "阿米娅",
        "aliases": [],
    },
    {
        # 拉丁名走整词路径，不进 trie
        "id": "e-act16",
        "name": "ACT16",
        "aliases": [],
    },
    {
        # 与否定词「不」同位置开头，且含「有」——停用词守卫必须挡住
        "id": "e-noanswer",
        "name": "不有",
        "aliases": [],
    },
]

ALIASES = {
    "凯尔希": "e-kaltsit",       # 与 entities 重复，entities 应胜出
    "巴别塔": "e-babel",         # entities 里没有，走补充路径
    "罗德岛": "e-rhodes",
}


class _Vocab:
    """最小词表替身，但要**非空**——空词表下关系表为空，测不出重建差异。"""

    CANONICAL_RELATIONS = ("隶属", "位于", "参与", "制造")
    RELATION_SYNONYMS = {
        "属于": "隶属",
        "在": "位于",       # 单字同义词，应被 `_relation_lookup` 排除
        "任": "参与",       # 同上
        "参加": "参与",
    }


QUERIES = [
    "凯尔希和罗德岛是什么关系",
    "凯医生参加了什么行动",
    "罗德岛制药公司制造了什么",
    "阿米娅是否参与了整合运动的行动",
    "贝洛内属于哪个组织",
    "ACT16 里发生了什么",
    "巴别塔和罗德岛之间有联系吗",
    "凯尔希不在罗德岛",
    "老女人位于哪里",
    "",
    "？？？",
]


def _calls_without_context(text: str) -> dict:
    return query.analyze_query(
        text, entities=ENTITIES, aliases=ALIASES, vocab=_Vocab
    )


def _calls_with_context(text: str, ctx: query.QueryContext) -> dict:
    return query.analyze_query(
        text, entities=ENTITIES, aliases=ALIASES, vocab=_Vocab, context=ctx
    )


@pytest.fixture(scope="module")
def ctx() -> query.QueryContext:
    return query.QueryContext(ENTITIES, ALIASES, _Vocab)


class TestEquivalence:
    """两条路必须逐位相同——这是复用结构唯一的正确性依据。"""

    @pytest.mark.parametrize("text", QUERIES, ids=lambda t: t or "empty")
    def test_result_is_identical(self, text: str, ctx: query.QueryContext) -> None:
        rebuilt = _calls_without_context(text)
        reused = _calls_with_context(text, ctx)
        assert reused == rebuilt, (
            f"问句 {text!r} 在复用 QueryContext 后结果变了。\n"
            f"就地重建：{rebuilt}\n"
            f"复用上下文：{reused}\n"
            f"这两条路必须等价，否则 H2 的提速是拿正确性换的。"
        )

    def test_stats_counters_are_identical(self, ctx: query.QueryContext) -> None:
        """`stats` 单独断言一次：它最容易在重构里被算成两套口径。"""
        for text in QUERIES:
            rebuilt = _calls_without_context(text)["stats"]
            reused = _calls_with_context(text, ctx)["stats"]
            assert reused == rebuilt, f"{text!r} 的 stats 不一致：{reused} vs {rebuilt}"

    def test_entity_candidate_order_is_identical(self, ctx: query.QueryContext) -> None:
        """候选顺序决定 `seed_ids` 顺序，进而决定图扩展的遍历次序。"""
        for text in QUERIES:
            rebuilt = [
                c["entity_id"] for c in _calls_without_context(text)["entity_candidates"]
            ]
            reused = [
                c["entity_id"]
                for c in _calls_with_context(text, ctx)["entity_candidates"]
            ]
            assert reused == rebuilt, f"{text!r} 候选顺序变了：{reused} vs {rebuilt}"


class TestContextContent:
    """上下文本身的内容也要能挑出错误，不能只靠"两条路一致"。"""

    def test_term_count_matches_claim_table(self, ctx: query.QueryContext) -> None:
        assert ctx.term_count == len(ctx.claims)

    def test_single_char_synonyms_are_dropped(self, ctx: query.QueryContext) -> None:
        """单字同义词（「在」->位于）必须被排除，否则「现在」会被解析成「位于」。"""
        assert "在" in ctx.dropped_synonyms
        assert "任" in ctx.dropped_synonyms
        assert "在" not in ctx.rel_table

    def test_longer_synonym_survives(self, ctx: query.QueryContext) -> None:
        assert ctx.rel_table.get("属于") == "隶属"
        assert ctx.rel_table.get("参加") == "参与"

    def test_entities_win_over_alias_index(self, ctx: query.QueryContext) -> None:
        """`aliases` 是 `setdefault` 建的一对一表，不能覆盖 `entities` 的认领。"""
        owners = ctx.claims["凯尔希"]
        assert [o["entity_id"] for o in owners] == ["e-kaltsit"]
        assert owners[0]["kind"] == "name"

    def test_ambiguous_text_keeps_all_owners(self, ctx: query.QueryContext) -> None:
        """「罗德岛制药公司」同时是 e-rhodes 的别名与 e-rhodes-corp 的规范名。"""
        owners = ctx.claims["罗德岛制药公司"]
        ids = sorted(o["entity_id"] for o in owners)
        assert ids == ["e-rhodes", "e-rhodes-corp"], (
            f"一对多认领被合并了：{owners}。歧义是事实，不许在这里就取唯一。"
        )

    def test_alias_index_supplements_missing_entities(
        self, ctx: query.QueryContext
    ) -> None:
        """`entities` 里没有的文本，由 `aliases` 补上。"""
        assert ctx.claims["巴别塔"][0]["entity_id"] == "e-babel"

    def test_context_reports_its_size(self, ctx: query.QueryContext) -> None:
        assert "terms=" in repr(ctx)


class TestNoContextStillWorks:
    """不传 `context` 的旧路径必须保留可用——单元测试靠它只给最小实体表。"""

    def test_default_path_still_rebuilds(self) -> None:
        out = query.analyze_query("凯尔希和罗德岛", entities=ENTITIES, vocab=_Vocab)
        assert len(out["entity_candidates"]) == 2

    def test_empty_vocab_is_tolerated(self) -> None:
        """没传 vocab 时按空词表处理，不抛错。"""
        out = query.analyze_query("凯尔希", entities=ENTITIES)
        assert out["should_answer"] is True


class TestContextIsCheapToReuse:
    """把"复用比重建快"变成机检，否则下次重构可能悄悄又退回每问重建。"""

    def test_reuse_is_much_faster_than_rebuild(self) -> None:
        import time

        ctx = query.QueryContext(ENTITIES, ALIASES, _Vocab)
        text = "凯尔希和罗德岛是什么关系"

        # 预热，避免首次调用的解释器开销污染比较
        _calls_with_context(text, ctx)
        _calls_without_context(text)

        n = 50
        t0 = time.perf_counter()
        for _ in range(n):
            _calls_without_context(text)
        rebuild_ms = (time.perf_counter() - t0) * 1000 / n

        t1 = time.perf_counter()
        for _ in range(n):
            _calls_with_context(text, ctx)
        reuse_ms = (time.perf_counter() - t1) * 1000 / n

        assert reuse_ms < rebuild_ms, (
            f"复用 QueryContext 反而更慢：复用 {reuse_ms:.3f} ms / 重建 "
            f"{rebuild_ms:.3f} ms。提速点已经失效，H2 的收益是假的。"
        )
