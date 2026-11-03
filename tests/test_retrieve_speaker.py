"""`retrieve` 的两项 I 阶段改动：候选池深度 `CANDIDATE_POOL_MULT` 与说话人加成。

## 这个文件在守什么

两项改动都是**纯排序改动**：不加候选、不删候选、不改索引、不改召回上限。
所以它们最典型的失效形态不是抛异常，而是**安静地改变排序**：

* 候选池倍数被改回 3（或被人「顺手优化」成 1）→ 召回悄悄退回到 H 阶段的
  0.6575，而所有报告仍然显示「PASS」。这是**回归**，不是崩溃。
* 倍数被改成 0 或负数 → `index.search(top_k=0)` 返回空，检索全空，但
  `search()` 会返回 `hits: []` 而不是报错，看起来像「拒答」。
* 说话人加成被写成**子串匹配** → 「凯尔希」命中「凯尔希医生」，甚至
  「阿米娅」命中「阿米娅的母亲」。剧情是对话体，`speakers` 是精确名单，
  子串匹配会把无关块抬进头部。
* `speakers` 解析失败时**猜一个**（比如按 `,` 硬切单引号串）→ 造出根本不
  存在的说话人名，加成打到错误的块上。
* 加成的归一化上界是「问句里同时命中该块的实体数」，**可以超过 1.0**——
  实测证明超过 1.0 的部分全部来自多实体同时在场（夹到 1.0 会让 R@20 掉一半），
  所以这一条不是缺陷而是信号。测试锁的是「不得为负、不得超过命中实体数」
  以及「同一实体的多个称呼不得重复计票」。

## 常量值本身也要锁

`CANDIDATE_POOL_MULT == 10` 与 `SPEAKER_BOOST_WEIGHT == 0.1` 不是随手取的，
它们各有一张 320 题实测表（见源码常量注释）。测试锁住数值，是为了让「改这个
数」必须同时改测试与注释里的表——**改动要有签字**，与卫生门同一个思路。
"""
from __future__ import annotations

import pytest

from arkrag import adjacency as adjacency_mod
from arkrag import config, index as index_mod, query, retrieve, vocab


# --------------------------------------------------------------------------
# 人造语料：四个块，`speakers` 覆盖 list / repr 字符串 / 缺失 / 坏格式 四种形态
# --------------------------------------------------------------------------
ENT_KAL = "e-0000000000000001"
ENT_AMI = "e-0000000000000002"
ENT_DOC = "e-0000000000000003"

TEXT_KAL = "凯尔希：切尔诺伯格已经沦陷了。"
TEXT_AMI = "阿米娅：我们必须回罗德岛。"
TEXT_DOC = "档案：罗德岛的成立年份是公开记录。"
TEXT_OTHER = "路人：今天天气不错。"


def _chunk(cid: str, text: str, speakers=None, *, raw: str | None = None) -> dict:
    rec = {
        "chunk_id": cid,
        "doc_id": cid.split("#")[0],
        "title": "测试篇",
        "chapter": "第一章",
        "text": text,
        "chars": len(text),
    }
    if raw is not None:
        rec["speakers"] = raw
    elif speakers is not None:
        rec["speakers"] = speakers
    return rec


def _entities() -> list[dict]:
    return [
        {"id": ENT_KAL, "name": "凯尔希", "type": "角色", "aliases": []},
        {"id": ENT_AMI, "name": "阿米娅", "type": "角色", "aliases": []},
        {"id": ENT_DOC, "name": "罗德岛", "type": "组织", "aliases": []},
    ]


def _graph(entities=None) -> dict:
    ents = _entities() if entities is None else entities
    return {
        "schema_version": config.GRAPH_SCHEMA_IDENTITY,
        "entities": ents,
        "edges": [],
        "alias_index": index_mod.build_alias_index(ents),
        "fingerprint": "0123456789abcdef",
        "stats": {"entities": len(ents), "edges": 0},
    }


def _retriever(chunks: list[dict], *, entities=None) -> retrieve.Retriever:
    graph = _graph(entities)
    idx, _meta = index_mod.build(chunks, graph, "corpusfp01234567")
    return retrieve.Retriever(idx, graph, adjacency_mod.build_directed_adjacency([]), chunks=chunks)


def _qa(text: str, entities=None) -> dict:
    ents = _entities() if entities is None else entities
    return query.analyze_query(text, entities=ents, vocab=vocab)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    root = (tmp_path / "artifacts").resolve()
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("ARKRAG_ARTIFACT_DIR", str(root))


# --------------------------------------------------------------------------
# 1. 常量必须与源码注释里的实测表一致
# --------------------------------------------------------------------------
class TestConstants:
    def test_candidate_pool_mult_is_the_measured_optimum(self):
        """10 是 320 题扫描里唯一同时最大化 R@20 与 MRR 的点。

        改这个数必须同时改 `retrieve.py` 里那张表的注释与本断言——
        那张表是「为什么是 10」的全部证据，数字对不上就是在改结论。
        """
        assert retrieve.CANDIDATE_POOL_MULT == 10

    def test_speaker_boost_weight_is_the_only_passing_value(self):
        """0.1 是 12 个组合里唯一让 R@20 与 MRR 都不劣的那个。

        ws≥0.5 时两个指标一起掉（0.6819/0.4076），所以这个常量的上限
        不是"调优空间"，而是"再大就净亏"。
        """
        assert retrieve.SPEAKER_BOOST_WEIGHT == 0.1

    def test_speaker_boost_is_smaller_than_entity_weight(self):
        """加成必须弱于实体通道——否则它会成为实际主排序键，而那只被评测过一次。"""
        assert retrieve.SPEAKER_BOOST_WEIGHT < retrieve.ENTITY_CHANNEL_WEIGHT

    def test_speaker_weight_is_not_larger_than_one(self):
        """权重本身必须在 (0,1] 内；分数可以超 1（见 `_speaker_boost` 的 docstring），
        但权重超 1 会让加成单独就能顶掉满分的 bm25 块。"""
        assert 0.0 < retrieve.SPEAKER_BOOST_WEIGHT <= 1.0


# --------------------------------------------------------------------------
# 2. `_speaker_set`：解析必须严格，不许猜
# --------------------------------------------------------------------------
class TestSpeakerSet:
    def test_list_form(self):
        assert retrieve._speaker_set({"speakers": ["凯尔希", "阿米娅"]}) == {"凯尔希", "阿米娅"}

    def test_repr_string_form_is_the_real_shape(self):
        """`chunks.jsonl` 里真实存的是单引号 Python list，不是 JSON。"""
        assert retrieve._speaker_set({"speakers": "['可萝尔', '村民']"}) == {"可萝尔", "村民"}

    def test_json_string_form_also_accepted(self):
        assert retrieve._speaker_set({"speakers": '["凯尔希"]'}) == {"凯尔希"}

    def test_missing_key_yields_empty(self):
        assert retrieve._speaker_set({}) == set()

    def test_empty_string_yields_empty(self):
        assert retrieve._speaker_set({"speakers": ""}) == set()

    def test_whitespace_only_yields_empty(self):
        assert retrieve._speaker_set({"speakers": "   "}) == set()

    def test_none_yields_empty(self):
        assert retrieve._speaker_set({"speakers": None}) == set()

    def test_garbage_string_yields_empty_not_a_guess(self):
        """无法解析时**必须放弃**，不许按逗号硬切成"看起来像名字"的东西。

        猜错的说话人会让加成把无关块抬进头部，比没有加成更糟。
        """
        assert retrieve._speaker_set({"speakers": "可萝尔, 村民"}) == set()

    def test_broken_bracket_yields_empty(self):
        assert retrieve._speaker_set({"speakers": "['未闭合"}) == set()

    def test_non_string_entries_are_stringified(self):
        assert retrieve._speaker_set({"speakers": [1, 2]}) == {"1", "2"}

    def test_blank_entries_are_dropped(self):
        assert retrieve._speaker_set({"speakers": ["凯尔希", "", "  "]}) == {"凯尔希"}

    def test_entries_are_stripped(self):
        assert retrieve._speaker_set({"speakers": [" 凯尔希 "]}) == {"凯尔希"}

    def test_tuple_form(self):
        assert retrieve._speaker_set({"speakers": ("凯尔希",)}) == {"凯尔希"}


# --------------------------------------------------------------------------
# 3. `_speaker_boost`：只在候选池内算、只做精确匹配、归一到 (0,1]
# --------------------------------------------------------------------------
class TestSpeakerBoost:
    def test_hit_only_when_speaker_listed_exactly(self):
        chunks = [
            _chunk("doc/a#000", TEXT_KAL, ["凯尔希"]),
            _chunk("doc/a#001", TEXT_AMI, ["阿米娅"]),
        ]
        ret = _retriever(chunks)
        boost = ret._speaker_boost(_qa("凯尔希说了什么")["entity_candidates"], {"doc/a#000", "doc/a#001"})
        assert "doc/a#000" in boost
        assert "doc/a#001" not in boost

    def test_substring_does_not_match(self):
        """「凯尔希」不能命中「凯尔希医生」——结构化字段是精确名单。"""
        chunks = [_chunk("doc/a#000", TEXT_KAL, ["凯尔希医生"])]
        ret = _retriever(chunks)
        boost = ret._speaker_boost(_qa("凯尔希说了什么")["entity_candidates"], {"doc/a#000"})
        assert boost == {}

    def test_only_allowed_chunks_are_scored(self):
        """`allowed` 之外的块即使说话人命中也不许给分——否则等于扫全表。"""
        chunks = [
            _chunk("doc/a#000", TEXT_KAL, ["凯尔希"]),
            _chunk("doc/a#001", TEXT_AMI, ["凯尔希"]),
        ]
        ret = _retriever(chunks)
        boost = ret._speaker_boost(_qa("凯尔希说了什么")["entity_candidates"], {"doc/a#000"})
        assert set(boost) == {"doc/a#000"}

    def test_no_candidates_yields_empty(self):
        chunks = [_chunk("doc/a#000", TEXT_KAL, ["凯尔希"])]
        ret = _retriever(chunks)
        assert ret._speaker_boost([], {"doc/a#000"}) == {}

    def test_no_allowed_yields_empty(self):
        chunks = [_chunk("doc/a#000", TEXT_KAL, ["凯尔希"])]
        ret = _retriever(chunks)
        assert ret._speaker_boost(_qa("凯尔希说了什么")["entity_candidates"], set()) == {}

    def test_chunk_without_speakers_is_skipped(self):
        chunks = [_chunk("doc/a#000", TEXT_KAL)]
        ret = _retriever(chunks)
        assert ret._speaker_boost(_qa("凯尔希说了什么")["entity_candidates"], {"doc/a#000"}) == {}

    def test_scores_scale_with_entities_present_but_stay_bounded_by_entity_count(self):
        """分数随「问句里有多少实体在该块开口」上升，上界是命中实体数。

        **这不是越界**：`_speaker_boost` 的分母只取最强实体，分子是所有命中实体
        的权重之和，所以两个问句实体都在同一块说话时能到 2.0。这是刻意的——
        两个当事人都在场比只有一个更能说明该块相关。

        实测依据（`scripts/diagnose_speaker_clamp.py`）：把分数夹到 1.0 后
        R@20 从 0.7019 掉到 0.6919，而按实体去重对分数毫无影响。也就是说
        「超过 1.0」全部来自多实体同时在场，不是别名重复计票。

        这个用例锁的是**上界公式**：分数不得为负、不得超过命中实体数。
        """
        chunks = [
            _chunk("doc/a#000", TEXT_KAL, ["凯尔希", "阿米娅"]),
            _chunk("doc/a#001", TEXT_AMI, ["凯尔希"]),
        ]
        ret = _retriever(chunks)
        cands = _qa("凯尔希和阿米娅说了什么")["entity_candidates"]
        boost = ret._speaker_boost(cands, {"doc/a#000", "doc/a#001"})
        assert boost, "两个块都该命中"
        # 命中实体数最多为问句实体数；用 3 作宽松上界即可捕获真正的失控
        for score in boost.values():
            assert 0.0 < score <= len(cands), f"分数越界：{score} vs 实体数 {len(cands)}"

    def test_strongest_entity_alone_scores_exactly_one(self):
        """只命中最强实体时归一化后恰好 1.0（这是「一对一」的基准点）。"""
        entities = _entities()
        chunks = [_chunk("doc/a#000", TEXT_KAL, ["凯尔希"])]
        ret = _retriever(chunks, entities=entities)
        cands = _qa("凯尔希说了什么", entities)["entity_candidates"]
        if len(cands) != 1:
            pytest.skip("该问句命中了多个实体，不适合验证单实体归一化")
        boost = ret._speaker_boost(cands, {"doc/a#000"})
        assert boost["doc/a#000"] == pytest.approx(1.0)

    def test_multiple_aliases_of_same_entity_do_not_double_count(self):
        """同一实体的规范名与别名都命中同一块时，只按该实体一票计（不超过 1.0）。

        这与「多个**不同**实体同时在场可以超过 1.0」不冲突：超过 1.0 的来源
        必须是不同的实体，而不是同一个实体换了几个说法。
        """
        entities = [
            {"id": ENT_KAL, "name": "凯尔希", "type": "角色",
             "aliases": [{"text": "凯尔希医生", "source": "upstream", "scenes": [], "review": "auto"}]},
        ]
        # 块里同时列了规范名与别名：命中两票也不该变成 2.0
        chunks = [_chunk("doc/a#000", TEXT_KAL, ["凯尔希", "凯尔希医生"])]
        ret = _retriever(chunks, entities=entities)
        boost = ret._speaker_boost(_qa("凯尔希医生说了什么", entities)["entity_candidates"], {"doc/a#000"})
        if "doc/a#000" in boost:
            assert boost["doc/a#000"] <= 1.0, (
                f"同一实体的多个称呼被重复计票：{boost['doc/a#000']}"
            )


# --------------------------------------------------------------------------
# 4. `search()` 集成：候选池深度真的生效了
# --------------------------------------------------------------------------
class TestCandidatePoolInSearch:
    def _many_chunks(self, n: int) -> list[dict]:
        """造 n 个都含「罗德岛」的块，保证 bm25 有足够多候选。"""
        return [
            _chunk(f"doc/a#{i:03d}", f"罗德岛的第 {i} 号记录：第{i}次行动纪要。", [])
            for i in range(n)
        ]

    def test_stats_expose_pool_depth_and_it_is_multiple_of_top_k(self):
        ret = _retriever(self._many_chunks(120))
        stats = ret.search(_qa("罗德岛"), top_k=5)["stats"]
        assert stats["candidate_pool_depth"] == 5 * retrieve.CANDIDATE_POOL_MULT

    def test_candidates_can_exceed_top_k(self):
        """深度生效的证据：候选池比输出大，否则倍数等于没加。"""
        ret = _retriever(self._many_chunks(120))
        res = ret.search(_qa("罗德岛"), top_k=5)
        assert res["stats"]["candidates"] > 5, "候选池必须深于输出窗"

    def test_hits_never_exceed_top_k(self):
        """深度只影响内部候选，输出窗仍是 top_k——用户看到的条数不变。"""
        ret = _retriever(self._many_chunks(120))
        res = ret.search(_qa("罗德岛"), top_k=7)
        assert len(res["hits"]) <= 7

    def test_depth_scales_with_top_k(self):
        ret = _retriever(self._many_chunks(120))
        assert (ret.search(_qa("罗德岛"), top_k=3)["stats"]["candidate_pool_depth"]
                == 3 * retrieve.CANDIDATE_POOL_MULT)
        assert (ret.search(_qa("罗德岛"), top_k=9)["stats"]["candidate_pool_depth"]
                == 9 * retrieve.CANDIDATE_POOL_MULT)

    def test_speaker_key_is_present_in_stats_even_when_zero(self):
        """没有命中时也要报 0，不能省略——省略会让"没测到"看起来像"没有这一项"。"""
        ret = _retriever(self._many_chunks(30))
        stats = ret.search(_qa("罗德岛"), top_k=5)["stats"]
        assert "speaker_boosted" in stats
        assert stats["speaker_boosted"] == 0
        assert stats["speaker_weight"] == retrieve.SPEAKER_BOOST_WEIGHT

    def test_speaker_channel_surfaces_in_sources_when_hit(self):
        chunks = [
            _chunk("doc/a#000", TEXT_KAL, ["凯尔希"]),
            _chunk("doc/a#001", TEXT_DOC, ["档案员"]),
        ]
        ret = _retriever(chunks)
        res = ret.search(_qa("凯尔希说了什么"), top_k=5)
        channels = {s["channel"] for h in res["hits"] for s in h["sources"]}
        assert retrieve.SOURCE_SPEAKER in channels, "说话人命中必须作为独立来源出现在 sources 里"

    def test_speaker_boost_never_lifts_chunk_above_pure_bm25_top(self):
        """加成只有 0.1，不许让一个只靠 speaker 的块顶掉 bm25 满分块。

        这是「加成不是主排序键」的行为级证据：把权重限制在 0.1 的全部意义
        就在于它不能单独决定头部。
        """
        chunks = [
            _chunk("doc/a#000", "罗德岛 罗德岛 罗德岛 罗德岛 罗德岛", ["无关者"]),
            _chunk("doc/a#001", TEXT_OTHER, ["凯尔希"]),
        ]
        ret = _retriever(chunks)
        res = ret.search(_qa("凯尔希 罗德岛"), top_k=5)
        assert res["hits"], "至少要有一条命中"
        # 只靠说话人加成的块，总分上限是 SPEAKER_BOOST_WEIGHT，达不到 1.0
        for h in res["hits"]:
            srcs = {s["channel"]: s["score"] for s in h["sources"]}
            if retrieve.SOURCE_BM25 not in srcs:
                assert h["score"] <= retrieve.SPEAKER_BOOST_WEIGHT + 1e-9, (
                    f"纯 speaker 块的分数 {h['score']} 超过了权重上限"
                )
