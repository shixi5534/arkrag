"""`arkrag.query` 的判据测试 —— 每一条守的都是一个会静默出错的地方。

## 这个文件在守什么

`analyze_query` 只做一件事：把问句拆成下游能消费的结构。它最容易犯的错都不是
"抛异常"，而是**安静地少给一点东西**——少一个否定词、少一个歧义候选、少一条
被停用词连带删掉的关系词。所以下面每个用例都是"喂一份能触发该错法的输入，
断言结果里能看见它"。

四类判据：

1. **歧义不许被消解**（`TestAmbiguityPreserved`）——同一别名被两个实体认领时，
   两个都必须出现在 `entity_candidates` 里且各带 `score`/`reason`。取第一个
   就是把未经验证的合并结论当前提。
2. **否定/时间不许丢**（`TestMarkersSurvive`）——否定词进 `negated_hint`、
   时间词进 `time_constraints`，且它们的位置不被停用词删掉。
3. **专名遮挡**（`TestEntitySpansBlockMarkers`）——专名内部的「别」「未」不是
   否定词。「玫瑰河畔是否参与了巴别塔相关的事务」里的「别」在「巴别塔」里面。
4. **拒绝要给理由**（`TestRefusals`）——空/纯标点/超长三态。

## 隔离

`query.py` 本身不碰文件系统，但用例仍按仓库纪律把产物根指到 `tmp_path` 并断言
前缀：防止将来有人在这里加一个"顺手读一下图"的调用而测试还静默通过。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from arkrag import config, query, release, vocab


ENTITIES = [
    {
        "id": "e-0000000000000001",
        "name": "凯尔希",
        "type": "角色",
        "aliases": [{"text": "凯尔希医生", "source": "upstream", "scenes": [], "review": "auto"}],
    },
    {
        "id": "e-0000000000000002",
        "name": "罗德岛",
        "type": "组织",
        "aliases": [],
    },
    {
        # 「女妖」同时被这两个实体认领——真实语料里就是这样。
        "id": "e-0000000000000003",
        "name": "Logos",
        "type": "角色",
        "aliases": [{"text": "女妖", "source": "upstream", "scenes": [], "review": "auto"}],
    },
    {
        "id": "e-0000000000000004",
        "name": "女妖之王",
        "type": "角色",
        "aliases": [{"text": "女妖", "source": "upstream", "scenes": [], "review": "auto"}],
    },
    {
        "id": "e-0000000000000005",
        "name": "玫瑰河畔",
        "type": "角色",
        "aliases": [],
    },
    {
        "id": "e-0000000000000006",
        "name": "巴别塔",
        "type": "组织",
        "aliases": [],
    },
]


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """把产物根与世代目录都指到 tmp_path，并断言真的指过去了。"""
    root = (tmp_path / "artifacts").resolve()
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("ARKRAG_ARTIFACT_DIR", str(root))
    monkeypatch.setattr(release, "GENERATIONS_DIR", (root / "generations").resolve())
    monkeypatch.setattr(release, "RUNS_DIR", (root / "runs").resolve())
    monkeypatch.setattr(release, "CURRENT_PATH", (root / "current.json").resolve())
    assert str(config.artifact_root()).startswith(str(tmp_path.resolve()))
    assert Path(config.artifact_root()) != config.DATA_DIR
    return root


def _analyze(text, **kw):
    """默认带真实词表：`_EmptyVocab` 下 `relation_hints` 恒为空，测不出判据。"""
    kw.setdefault("vocab", vocab)
    return query.analyze_query(text, entities=ENTITIES, **kw)


class TestAmbiguityPreserved:
    """同一别名被多个实体认领时，候选一条都不许删。"""

    def test_alias_claimed_by_two_entities_yields_two_candidates(self):
        out = _analyze("女妖是谁")
        ids = {c["entity_id"] for c in out["entity_candidates"]}
        assert ids == {"e-0000000000000003", "e-0000000000000004"}
        assert out["stats"]["ambiguous_texts"] == [
            {"text": "女妖", "entity_ids": ["e-0000000000000003", "e-0000000000000004"]}
        ]

    def test_every_candidate_carries_score_and_reason(self):
        out = _analyze("女妖是谁")
        for cand in out["entity_candidates"]:
            assert 0.0 < cand["score"] <= 1.0
            assert cand["reason"] in {"exact_name", "alias"}
            assert cand["ambiguous"] is True

    def test_exact_name_outranks_alias(self):
        """整名命中必须比别名命中分高，否则别名会把真名挤到后面。"""
        out = _analyze("凯尔希医生与凯尔希")
        by_id = {c["entity_id"]: c for c in out["entity_candidates"]}
        # 同一实体被两个文本命中，只留最高分那条（整名）
        assert by_id["e-0000000000000001"]["reason"] == "exact_name"
        assert by_id["e-0000000000000001"]["name"] == "凯尔希"

    def test_alias_index_is_supplementary_not_authoritative(self):
        """alias_index 只补 entities 没表态的文本。

        这里喂一个与 entities 冲突的 alias_index（把「凯尔希」指向罗德岛），
        如果实现让 alias_index 覆盖 entities，就会多出一个错误的候选。
        """
        out = query.analyze_query(
            "凯尔希",
            entities=ENTITIES,
            aliases={"凯尔希": "e-0000000000000002"},
        )
        ids = [c["entity_id"] for c in out["entity_candidates"]]
        assert ids == ["e-0000000000000001"]


class TestMarkersSurvive:
    def test_negation_sets_hint(self):
        out = _analyze("凯尔希没有参与")
        assert out["negated_hint"] is True
        assert "没有" in out["negation_markers"]

    def test_negation_marker_is_not_dropped_as_stopword(self):
        """「不」在停用词表里没有，但「没」等标记必须原地保留在 retrieval_query。

        判据：把否定词删掉后 retrieval_query 会退化成「凯尔希 参与」——那就等于
        把「没参与」检索成「参与」。这里断言否定词仍在。
        """
        out = _analyze("凯尔希没有参与")
        assert "没有" in out["retrieval_query"]

    def test_time_markers_kept_as_original_fragments(self):
        out = _analyze("萨卢佐家族与贝洛内家族最初是什么关系，后来呢")
        texts = [t["marker"] for t in out["time_constraints"]]
        assert texts == ["最初", "后来"]
        for item in out["time_constraints"]:
            start, end = item["span"]
            assert out["normalized_query"][start:end] == item["marker"]

    def test_dropped_stopwords_are_counted(self):
        out = _analyze("请问凯尔希和罗德岛是什么关系")
        terms = [d["term"] for d in out["stats"]["dropped_terms"]]
        assert "请问" in terms
        # 每个被删的词都带原文区间，能回指到问句里它的位置
        for item in out["stats"]["dropped_terms"]:
            start, end = item["span"]
            assert out["normalized_query"][start:end] == item["term"]
        assert out["stats"]["query_chars"] == len(out["normalized_query"])


class TestEntitySpansBlockMarkers:
    def test_marker_inside_entity_name_is_not_negation(self):
        """「巴别塔」里的「别」是否定标记字面，但不构成否定。"""
        out = _analyze("玫瑰河畔是否参与了巴别塔相关的事务")
        assert out["negated_hint"] is False
        blocked = out["stats"]["negation_blocked"]
        assert any(b["marker"] == "别" and b["reason"] == "inside_entity" for b in blocked)

    def test_marker_outside_entity_still_counts(self):
        """确认上一条不是靠"把所有「别」都挡掉"通过的。"""
        out = _analyze("玫瑰河畔别参与巴别塔")
        assert out["negated_hint"] is True
        assert "别" in out["negation_markers"]


class TestRelationHints:
    def test_canonical_relation_is_hinted(self):
        out = _analyze("凯尔希隶属罗德岛吗")
        assert [h["canonical"] for h in out["relation_hints"]] == ["隶属"]

    def test_single_char_synonyms_are_excluded_and_reported(self):
        """单字同义词（在→位于、任→任职）会在「现在」「任何」里误命中。"""
        out = _analyze("凯尔希现在在哪")
        canon = [h.get("canonical") for h in out["relation_hints"]]
        assert "位于" not in canon
        assert out["stats"]["dropped_synonyms"], "排除清单必须被报告出来"

    def test_stopword_dropping_does_not_split_a_relation_hint(self):
        """「参与」不能被停用词删成「参」。

        这条守的是 `_drop_stopwords` 的 protected 区间包含关系提示位置。删错时
        关系提示还在 relation_hints 里，但 retrieval_query 已经碎掉，检索召回会
        莫名其妙变差——是最难查的一类回归。
        """
        out = _analyze("玫瑰河畔是否参与了巴别塔相关的事务")
        assert "参与" in out["retrieval_query"]

    def test_two_entities_force_graph_requirement(self):
        out = _analyze("凯尔希和罗德岛")
        assert out["requires_graph"] is True

    def test_single_entity_without_relation_does_not_require_graph(self):
        out = _analyze("凯尔希")
        assert out["requires_graph"] is False


class TestRefusals:
    def test_empty_query(self):
        out = _analyze("")
        assert out["should_answer"] is False
        assert out["reason"] == "empty_query"
        assert out["entity_candidates"] == []

    def test_punctuation_only(self):
        out = _analyze("？？？……")
        assert out["should_answer"] is False
        assert out["reason"] == "no_word_characters"

    def test_too_long_query_is_refused_not_truncated(self):
        out = _analyze("凯" * 600)
        assert out["should_answer"] is False
        assert out["reason"].startswith("query_too_long")
        assert "600" in out["reason"]

    def test_refusal_still_returns_all_keys(self):
        """调用方不该写两套分支：拒答时字段齐全，只是空。"""
        out = _analyze("")
        for key in (
            "original_query",
            "normalized_query",
            "retrieval_query",
            "entity_candidates",
            "relation_hints",
            "time_constraints",
            "requires_graph",
            "should_answer",
            "reason",
            "stats",
        ):
            assert key in out

    def test_fullwidth_is_normalized(self):
        out = _analyze("ＫＥＬＳＩ")  # 全角拉丁
        assert out["normalized_query"] == "KELSI"


class TestMinimalVocab:
    def test_missing_vocab_does_not_raise(self):
        out = query.analyze_query("凯尔希和罗德岛", entities=ENTITIES)
        assert out["should_answer"] is True
        assert out["relation_hints"] == []


class TestNoFilesystemWrites:
    def test_analysis_writes_nothing(self, tmp_path):
        before = sorted(p.name for p in tmp_path.rglob("*"))
        _analyze("凯尔希和罗德岛是什么关系")
        after = sorted(p.name for p in tmp_path.rglob("*"))
        assert before == after

    def test_result_is_json_serialisable(self):
        out = _analyze("凯尔希和罗德岛是什么关系")
        assert json.loads(json.dumps(out, ensure_ascii=False))["should_answer"] is True
