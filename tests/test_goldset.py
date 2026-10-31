"""F3 Gold Set 的机检：每条判据都必须能真的变红。

## 为什么这个文件里既有构造数据又有真实数据

构造数据负责证明「判据本身存在且会拒绝」——比如「待审别名不许出题」，只有
拿一个**只带 pending 别名的实体**喂进去、断言它产不出题，才能说这条判据生效；
在真实数据上看到「没有 pending 别名出的题」既可能是判据拦住了，也可能是这份
数据里根本没有 pending 别名（`data/arkrag` 里实测 pending 有 562 条，所以这次
不是空集，但依赖数据分布做结论本身就不稳）。

真实数据负责证明「判据没有把整类题全枪毙」——构造数据全绿、真实数据 0 条产出，
是这套门最典型的一种假绿。

## 关于「机械构造不等于自证」

本文件的真实数据断言全部落在**语料**上，不落在图上：`required_citations` 的
证据必须能用 `str.find` 在 `chunks.jsonl` 的正文里逐字定位；不可回答题必须
反查图里确实没有那条边。也就是说，即便图和 Gold Set 出自同一次构建，判据
仍然是「拿产物去对语料」，而语料是独立的一手事实。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from arkrag import goldset, release, vocab

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- 构造数据工具


def _chunk(cid: str, text: str, *, title: str = "标题", chapter: str = "ch1") -> dict:
    """最小可用语料块。"""
    return {
        "chunk_id": cid,
        "doc_id": cid.split("#")[0],
        "title": title,
        "chapter": chapter,
        "category": "activities",
        "scene_index": 0,
        "text": text,
    }


def _entity(eid: str, name: str, *, etype: str = "角色", chunks=("c#1",), **kw) -> dict:
    """最小可用实体；`aliases`/`pending_aliases` 默认空。"""
    row = {
        "id": eid,
        "name": name,
        "type": etype,
        "aliases": [],
        "pending_aliases": [],
        "rejected_aliases": [],
        "merged_from": [],
        "chunks": list(chunks),
    }
    row.update(kw)
    return row


def _edge(eid: str, s: str, p: str, o: str, *, support: int = 5, ev: str = "", chunks=("c#1",)) -> dict:
    """最小可用边。"""
    return {
        "edge_id": eid,
        "subject_id": s,
        "predicate": p,
        "object_id": o,
        "support_count": support,
        "independent_scene_count": support,
        "evidence": ev,
        "evidence_assertion_id": "x",
        "chunks": list(chunks),
        "scenes": ["s"],
    }


def _assertion(aid: str, cid: str, text: str, ev: str, *, s="", p="", o="") -> dict:
    """最小可用断言（单块事实题只用到 chunk_id/evidence_text/assertion_id）。"""
    start = text.find(ev)
    return {
        "assertion_id": aid,
        "chunk_id": cid,
        "scene_id": cid.split("#")[0],
        "doc_id": cid.split("#")[0],
        "subject_id": s,
        "predicate": p,
        "object_id": o,
        "evidence_text": ev,
        "evidence_start": start,
        "evidence_end": start + len(ev),
        "evidence_all": [{"start": start, "end": start + len(ev)}],
        "qualifiers": {},
    }


def _numbered(items: list[dict]) -> list[dict]:
    """给 miner 直接产出的题补 `question_id`。

    `question_id` 是**装配期**才有的（`build_gold_set` 里 `_number()` 按类别顺序
    编号）。miner 单独调用时不带它，所以单测里校验前必须先编号——否则
    `validate_gold_item` 会在「缺必填字段」处直接返回，后面十几条判据一条都跑不到，
    测试就变成了假绿。
    """
    goldset._number(items)
    return items


def _three_hop_fixture() -> tuple[list[dict], list[dict], dict, dict]:
    """三个实体 + 两条边 + 一份「正向」邻接 + 一份「只会被 in 看见」的邻接。

    返回 `(entities, edges, adj_out, adj_in_only)`。两条邻接含同一批边 ID，
    区别只在把边挂到 `out` 还是 `in` 上——这是「多跳只许走 out」唯一能被证明的
    办法：同一份边数据，换个方向就产不出题。
    """
    ents = [
        _entity("e-0000000000000001", "阿尔法"),
        _entity("e-0000000000000002", "贝塔"),
        _entity("e-0000000000000003", "伽马"),
    ]
    a, b, c = (e["id"] for e in ents)
    edges = [
        _edge("r1", a, "隶属", b, ev="阿尔法隶属于贝塔"),
        _edge("r2", b, "隶属", c, ev="贝塔隶属于伽马"),
    ]
    adj_out = {
        a: {"out": [{"edge_id": "r1"}], "in": []},
        b: {"out": [{"edge_id": "r2"}], "in": [{"edge_id": "r1"}]},
        c: {"out": [], "in": [{"edge_id": "r2"}]},
    }
    adj_in_only = {
        a: {"out": [], "in": [{"edge_id": "r1"}]},
        b: {"out": [], "in": [{"edge_id": "r2"}]},
        c: {"out": [], "in": []},
    }
    return ents, edges, adj_out, adj_in_only


# ---------------------------------------------------------------- 真实数据夹具


@pytest.fixture(scope="session")
def corpus() -> dict:
    """真实 `ark-rag-3` 世代产物 + 语料块。读不到就 skip，不伪造。"""
    gens_dir = release.GENERATIONS_DIR
    picked: Path | None = None
    if gens_dir.exists():
        for path in sorted(gens_dir.iterdir()):
            graph_path = path / "graph" / "graph.json"
            if not graph_path.is_file():
                continue
            if not (path / "graph" / "assertions.jsonl").is_file():
                continue
            doc = json.loads(graph_path.read_text("utf-8"))
            if doc.get("schema_version") == "ark-rag-3":
                picked = path
    if picked is None:
        pytest.skip("本机没有 ark-rag-3 世代目录，真实数据用例跳过（不伪造）")

    def read_jsonl(p: Path) -> list[dict]:
        rows: list[dict] = []
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    from arkrag import config

    graph = json.loads((picked / "graph" / "graph.json").read_text("utf-8"))
    return {
        "generation": picked.name,
        "graph": graph,
        "entities": graph.get("entities") or [],
        "edges": graph.get("edges") or [],
        "assertions": read_jsonl(picked / "graph" / "assertions.jsonl"),
        "adjacency": json.loads((picked / "graph" / "adjacency.json").read_text("utf-8")),
        "chunks": read_jsonl(config.chunks_path()),
    }


@pytest.fixture(scope="session")
def real_gold(corpus) -> dict:
    """真实数据上构建 40 题（每类 2–8 题，够证明「没有整类被枪毙」）。"""
    return goldset.build_gold_set(
        corpus["assertions"],
        corpus["chunks"],
        corpus["edges"],
        corpus["entities"],
        corpus["adjacency"],
        total=40,
        seed=20261101,
    )


@pytest.fixture(scope="session")
def real_chunk_map(corpus) -> dict[str, dict]:
    return {c["chunk_id"]: c for c in corpus["chunks"]}


# ---------------------------------------------------------------- 校验器本身


class TestValidateGoldItem:
    """`validate_gold_item` 的每一条判据都要能被构造输入打红。"""

    def _good(self) -> dict:
        return {
            "question_id": "q-sc-000001",
            "question": "以下引文出自哪个块？「阿尔法隶属于贝塔」",
            "answer": "标题",
            "acceptable_answers": ["标题", "ch1/标题"],
            "gold_chunk_ids": ["c#1"],
            "gold_assertion_ids": ["a1"],
            "required_citations": [{"chunk_id": "c#1", "evidence": "阿尔法隶属于贝塔"}],
            "difficulty": "easy",
            "category": "single_chunk",
            "should_abstain": False,
        }

    def test_good_item_passes(self) -> None:
        cmap = {"c#1": _chunk("c#1", "阿尔法隶属于贝塔。贝塔隶属于伽马。")}
        assert goldset.validate_gold_item(self._good(), cmap) == []

    def test_missing_field_is_reported(self) -> None:
        item = self._good()
        del item["answer"]
        errs = goldset.validate_gold_item(item, {"c#1": _chunk("c#1", "x")})
        assert any("缺少必填字段 answer" in e for e in errs)

    def test_bad_question_id_format(self) -> None:
        item = self._good()
        item["question_id"] = "Q1"
        errs = goldset.validate_gold_item(item, {"c#1": _chunk("c#1", "阿尔法隶属于贝塔")})
        assert any("question_id 格式非法" in e for e in errs)

    def test_question_id_abbr_must_match_category(self) -> None:
        item = self._good()
        item["question_id"] = "q-al-000001"
        errs = goldset.validate_gold_item(item, {"c#1": _chunk("c#1", "阿尔法隶属于贝塔")})
        assert any("缩写与 category 不符" in e for e in errs)

    def test_unknown_category(self) -> None:
        item = self._good()
        item["category"] = "vibes"
        errs = goldset.validate_gold_item(item, {"c#1": _chunk("c#1", "阿尔法隶属于贝塔")})
        assert any("category 不在八类内" in e for e in errs)

    def test_answer_must_be_in_acceptable_answers(self) -> None:
        item = self._good()
        item["acceptable_answers"] = ["别的答案"]
        errs = goldset.validate_gold_item(item, {"c#1": _chunk("c#1", "阿尔法隶属于贝塔")})
        assert any("未包含 answer" in e for e in errs)

    def test_nonexistent_gold_chunk(self) -> None:
        item = self._good()
        item["gold_chunk_ids"] = ["不存在#9"]
        errs = goldset.validate_gold_item(item, {"c#1": _chunk("c#1", "阿尔法隶属于贝塔")})
        assert any("含不存在的块" in e for e in errs)

    def test_tampered_evidence_is_rejected(self) -> None:
        """证据被改一个字就必须报错——这是「不接受答案对但引用错」的机检。"""
        item = self._good()
        item["required_citations"] = [
            {"chunk_id": "c#1", "evidence": "阿尔法隶属于伽马"}
        ]
        errs = goldset.validate_gold_item(item, {"c#1": _chunk("c#1", "阿尔法隶属于贝塔")})
        assert any("无法在该块正文中逐字定位" in e for e in errs)

    def test_citation_chunk_must_be_in_gold_chunks(self) -> None:
        item = self._good()
        item["gold_chunk_ids"] = ["c#2"]
        cmap = {
            "c#1": _chunk("c#1", "阿尔法隶属于贝塔"),
            "c#2": _chunk("c#2", "阿尔法隶属于贝塔"),
        }
        errs = goldset.validate_gold_item(item, cmap)
        assert any("不在 gold_chunk_ids 内" in e for e in errs)

    def test_abstain_must_not_carry_evidence(self) -> None:
        item = self._good()
        item["should_abstain"] = True
        errs = goldset.validate_gold_item(item, {"c#1": _chunk("c#1", "阿尔法隶属于贝塔")})
        assert any("should_abstain=True 时 gold_chunk_ids 必须为空" in e for e in errs)

    def test_answerable_must_carry_evidence(self) -> None:
        item = self._good()
        item["gold_chunk_ids"] = []
        item["required_citations"] = []
        errs = goldset.validate_gold_item(item, {"c#1": _chunk("c#1", "阿尔法隶属于贝塔")})
        assert any("should_abstain=False 时 gold_chunk_ids 不能为空" in e for e in errs)


# ---------------------------------------------------------------- 别名题


class TestAliasMining:
    """别名题只许用权威别名：`pending` / `rejected` 一律不出题。"""

    TEXT = "凯尔希：机密仓库就在下面。凯尔希打开了门。"

    def _cmap(self) -> dict[str, dict]:
        return {"c#1": _chunk("c#1", self.TEXT)}

    def test_authoritative_alias_produces_a_question(self) -> None:
        ent = _entity(
            "e-0000000000000001",
            "凯尔希",
            aliases=[{"text": "机密仓库", "source": "upstream", "scenes": ["c"], "review": "auto"}],
        )
        got = _numbered(goldset.mine_alias_questions([ent], self._cmap(), limit=5, seed=1))
        assert len(got) == 1
        assert got[0]["answer"] == "凯尔希"
        assert got[0]["category"] == "alias"
        assert goldset.validate_gold_item(got[0], self._cmap()) == []

    def test_pending_alias_in_pending_aliases_yields_nothing(self) -> None:
        ent = _entity(
            "e-0000000000000001",
            "凯尔希",
            pending_aliases=[
                {"text": "机密仓库", "source": "upstream", "scenes": ["c"], "review": "pending"}
            ],
        )
        assert goldset.mine_alias_questions([ent], self._cmap(), limit=5, seed=1) == []

    def test_pending_review_inside_aliases_yields_nothing(self) -> None:
        """`aliases` 里 review=pending 的行同样不许出题。

        `identity.to_dict()` 会把待审别名拆进 `pending_aliases`，但 dataclass 的
        `.aliases` 仍含全部条目；任何按位置而非按 `review` 取值的地方都会漏判，
        所以这里显式造一条「留在 aliases 里但 review=pending」的输入。
        """
        ent = _entity(
            "e-0000000000000001",
            "凯尔希",
            aliases=[{"text": "机密仓库", "source": "upstream", "scenes": ["c"], "review": "pending"}],
        )
        assert goldset.mine_alias_questions([ent], self._cmap(), limit=5, seed=1) == []

    def test_rejected_alias_yields_nothing(self) -> None:
        ent = _entity(
            "e-0000000000000001",
            "凯尔希",
            rejected_aliases=[{"text": "机密仓库", "source": "upstream", "scenes": ["c"], "review": "rejected"}],
        )
        assert goldset.mine_alias_questions([ent], self._cmap(), limit=5, seed=1) == []

    def test_alias_without_corpus_footing_yields_nothing(self) -> None:
        """别名与规范名不同块出现（语料里没有「这个别名指它」的落点）→ 不出题。"""
        ent = _entity(
            "e-0000000000000001",
            "凯尔希",
            aliases=[{"text": "勋爵", "source": "upstream", "scenes": ["c"], "review": "auto"}],
        )
        assert goldset.mine_alias_questions([ent], self._cmap(), limit=5, seed=1) == []


# ---------------------------------------------------------------- 多跳方向


class TestMultihopDirection:
    """多跳只许走 `out`：同一批边挂到 `in` 上就产不出题。"""

    def test_out_adjacency_produces_the_middle_entity(self) -> None:
        ents, edges, adj_out, _ = _three_hop_fixture()
        cmap = {"c#1": _chunk("c#1", "阿尔法隶属于贝塔。贝塔隶属于伽马。")}
        got = goldset.mine_multihop(adj_out, edges, ents, cmap, limit=5, seed=1)
        assert len(got) == 1
        assert got[0]["answer"] == "贝塔"
        assert got[0]["category"] == "multihop"

    def test_in_only_adjacency_produces_nothing(self) -> None:
        ents, edges, _, adj_in_only = _three_hop_fixture()
        cmap = {"c#1": _chunk("c#1", "阿尔法隶属于贝塔。贝塔隶属于伽马。")}
        assert goldset.mine_multihop(adj_in_only, edges, ents, cmap, limit=5, seed=1) == []

    def test_two_middles_are_dropped_not_guessed(self) -> None:
        """A→C 有两个中间实体时标准答案不唯一，必须丢弃而不是随机选一个。"""
        ents = [
            _entity("e-0000000000000001", "阿尔法"),
            _entity("e-0000000000000002", "贝塔"),
            _entity("e-0000000000000003", "伽马"),
            _entity("e-0000000000000004", "德尔塔"),
        ]
        a, b, c, d = (e["id"] for e in ents)
        edges = [
            _edge("r1", a, "隶属", b, ev="阿尔法隶属于贝塔"),
            _edge("r2", b, "隶属", c, ev="贝塔隶属于伽马"),
            _edge("r3", a, "隶属", d, ev="阿尔法隶属于德尔塔"),
            _edge("r4", d, "隶属", c, ev="德尔塔隶属于伽马"),
        ]
        adj = {
            a: {"out": [{"edge_id": "r1"}, {"edge_id": "r3"}], "in": []},
            b: {"out": [{"edge_id": "r2"}], "in": []},
            d: {"out": [{"edge_id": "r4"}], "in": []},
            c: {"out": [], "in": []},
        }
        cmap = {"c#1": _chunk("c#1", "阿尔法隶属于贝塔。贝塔隶属于伽马。")}
        assert goldset.mine_multihop(adj, edges, ents, cmap, limit=5, seed=1) == []


# ---------------------------------------------------------------- 不可回答题


class TestUnanswerable:
    """不可回答题的两条机检线索必须真的能被反查。"""

    def _entities(self) -> list[dict]:
        return [
            _entity("e-0000000000000001", "阿尔法"),
            _entity("e-0000000000000002", "贝塔"),
        ]

    def _items(self, edges: list[dict], entities: list[dict] | None = None) -> list[dict]:
        cmap = {"c#1": _chunk("c#1", "这一块里没有任何可用事实。")}
        return _numbered(
            goldset.mine_negative_and_unanswerable(
                cmap, entities or self._entities(), edges, limit=4, seed=7
            )
        )

    def test_items_carry_reverse_lookup_hints(self) -> None:
        got = self._items([])
        assert got, "空图上应能造出不可回答题"
        for it in got:
            assert it["should_abstain"] is True
            assert it["gold_chunk_ids"] == []
            assert it["gold_assertion_ids"] == []
            assert it["required_citations"] == []
            assert it["_absent_a_name"] and it["_absent_b_name"]
            assert it["_absent_label"] not in vocab.CANONICAL_RELATIONS

    def test_items_pass_validation_on_empty_graph(self) -> None:
        ents = self._entities()
        by_id = {e["id"]: e for e in ents}
        cmap = {"c#1": _chunk("c#1", "这一块里没有任何可用事实。")}
        for it in self._items([]):
            errs = goldset.validate_gold_item(it, cmap, edges=[], entities_by_id=by_id)
            assert errs == [], f"{it['question_id']} 不该失败：{errs}"

    def test_hint_absent_means_validation_skips_the_check(self) -> None:
        """去掉机检线索字段必须报错——否则「图里真没有」就只是挖掘时的声称。"""
        ents = self._entities()
        by_id = {e["id"]: e for e in ents}
        cmap = {"c#1": _chunk("c#1", "x")}
        it = self._items([])[0]
        stripped = dict(it)
        stripped.pop("_absent_a_name", None)
        errs = goldset.validate_gold_item(stripped, cmap, edges=[], entities_by_id=by_id)
        assert any("缺 _absent_a_name" in e for e in errs)

    def test_machine_check_catches_an_edge_that_actually_exists(self) -> None:
        """喂一条「该谓词其实有边」的图，必须报错。"""
        ents = self._entities()
        by_id = {e["id"]: e for e in ents}
        a, b = (e["id"] for e in ents)
        cmap = {"c#1": _chunk("c#1", "x")}
        it = self._items([])[0]
        label = it["_absent_label"]
        bogus = [_edge("r1", a, label, b)]
        errs = goldset.validate_gold_item(it, cmap, edges=bogus, entities_by_id=by_id)
        assert any("不可回答题的谓词在图上存在" in e for e in errs)
        assert any("其实有该谓词的边" in e for e in errs)

    def test_mining_skips_pairs_that_already_have_a_canonical_edge(self) -> None:
        """两实体之间已有正规关系边时，不许再审它们「没有关系」。"""
        ents = self._entities() + [_entity("e-0000000000000003", "伽马")]
        a, b, _c = (e["id"] for e in ents)
        got = self._items([_edge("r1", a, "隶属", b)], ents)
        assert got, "三实体应该还能造出不相邻的那一对"
        for it in got:
            assert {it["_absent_a_name"], it["_absent_b_name"]} != {"阿尔法", "贝塔"}

    def test_real_abstain_items_survive_reverse_lookup(self, corpus, real_gold) -> None:
        """真实数据：逐条按 name → id → 边 反查，不许有一条是有答案的。"""
        by_name = {e.get("name"): e for e in corpus["entities"]}
        pairs = {
            (e.get("subject_id"), e.get("predicate"), e.get("object_id"))
            for e in corpus["edges"]
        }
        abstain = [it for it in real_gold["items"] if it["should_abstain"]]
        assert abstain, "真实数据上一条不可回答题都造不出来，说明这一类被枪毙了"
        for it in abstain:
            a = by_name.get(it["_absent_a_name"])
            b = by_name.get(it["_absent_b_name"])
            assert a is not None and b is not None, f"实体不在图里：{it['_absent_a_name']}"
            label = it["_absent_label"]
            assert label not in vocab.CANONICAL_RELATIONS
            assert (a["id"], label, b["id"]) not in pairs
            assert (b["id"], label, a["id"]) not in pairs
            assert not any(e.get("predicate") == label for e in corpus["edges"])


# ---------------------------------------------------------------- 时间冲突


class TestTimeConflict:
    """互斥谓词跨场景时是「时序演进」，同场景才交人工。"""

    TEXT_A = "甲与乙结成同盟。"
    TEXT_B = "甲与乙互为敌对。"

    def _fixture(self, *, same_scene: bool) -> tuple[list, list, list]:
        chunks = [
            _chunk("c#1", self.TEXT_A, title="一"),
            _chunk("c#2", self.TEXT_B, title="二"),
        ]
        ents = [
            _entity("e-0000000000000001", "甲", chunks=("c#1", "c#2")),
            _entity("e-0000000000000002", "乙", chunks=("c#1", "c#2")),
        ]
        a1 = _assertion("a1", "c#1", self.TEXT_A, "甲与乙结成同盟", s=ents[0]["id"], p="同盟", o=ents[1]["id"])
        a1["scene_id"] = "s1"
        a2 = _assertion("a2", "c#2", self.TEXT_B, "甲与乙互为敌对", s=ents[0]["id"], p="敌对", o=ents[1]["id"])
        a2["scene_id"] = "s1" if same_scene else "s2"
        return ents, chunks, [a1, a2]

    def test_cross_scene_conflict_becomes_a_question(self) -> None:
        ents, chunks, asserts = self._fixture(same_scene=False)
        cmap = {c["chunk_id"]: c for c in chunks}
        got = goldset.mine_time_conflict(asserts, cmap, ents, limit=4, seed=1)
        assert len(got) == 1
        assert got[0]["category"] == "time_conflict"
        assert got[0]["should_abstain"] is False
        assert "时序演进" in got[0]["acceptable_answers"]
        assert len(got[0]["gold_chunk_ids"]) == 2

    def test_same_scene_conflict_is_left_to_humans(self) -> None:
        ents, chunks, asserts = self._fixture(same_scene=True)
        cmap = {c["chunk_id"]: c for c in chunks}
        assert goldset.mine_time_conflict(asserts, cmap, ents, limit=4, seed=1) == []

    def test_same_pair_repeated_across_chunks_yields_one_question(self) -> None:
        """同一 (实体对, 谓词对) 有多组断言时必须去重，否则产出重复问句。"""
        ents, chunks, asserts = self._fixture(same_scene=False)
        chunks.append(_chunk("c#3", self.TEXT_A, title="三"))
        a3 = dict(asserts[0], assertion_id="a3", chunk_id="c#3")
        cmap = {c["chunk_id"]: c for c in chunks}
        got = goldset.mine_time_conflict(asserts + [a3], cmap, ents, limit=4, seed=1)
        assert len(got) == 1
        assert goldset.find_duplicate_questions(_numbered(got)) == []


# ---------------------------------------------------------------- 否定题判据


class TestNegationRules:
    """`_negation_rows` 的 8 条判据逐条喂坏输入。"""

    def _rows(self, text: str, *, ents=None, edges=None) -> list[dict]:
        chunk = _chunk("c#1", text)
        pals = ents if ents is not None else [_entity("e-0000000000000001", "侯爵")]
        return goldset._negation_rows({"c#1": chunk}, pals, edges or [])

    def test_positive_case_is_mined(self) -> None:
        rows = self._rows("凯尔希：侯爵没有保护科学家们。")
        assert len(rows) == 1
        assert rows[0]["relation"] == "保护"
        assert rows[0]["entity_name"] == "侯爵"

    def test_interrogative_is_rejected(self) -> None:
        """反问句不是在陈述事实，答案未必是「否」。"""
        assert self._rows("凯尔希：为什么侯爵没有保护科学家们？") == []

    def test_parallel_negation_is_rejected(self) -> None:
        """并列否定：被否定的是并列结构里的宾语，不是主谓关系。"""
        ents = [_entity("e-0000000000000001", "通讯设备")]
        assert self._rows("凯尔希：他们既没有通讯设备也没有人指挥。", ents=ents) == []

    def test_temporal_clause_is_rejected(self) -> None:
        """「在前往北方冰原之后」里「前往」是时间状语，不是谓词。"""
        ents = [_entity("e-0000000000000001", "命运")]
        assert self._rows("凯尔希：命运并没有在前往北方冰原之后变好。", ents=ents) == []

    def test_entity_must_precede_the_negation(self) -> None:
        assert self._rows("凯尔希：没有保护科学家们的是侯爵。") == []

    def test_pronoun_subject_is_rejected(self) -> None:
        """「他没有离开」的真主语在上一句，不能牵到本块的实体上。"""
        ents = [_entity("e-0000000000000001", "选帝侯")]
        assert self._rows("凯尔希：他没有离开过。选帝侯还在。", ents=ents) == []

    def test_graph_edge_cancels_the_negation(self) -> None:
        """图里有这个谓词的边 → 标准答案自相矛盾，不许出题。"""
        ents = [_entity("e-0000000000000001", "侯爵")]
        edges = [_edge("r1", ents[0]["id"], "保护", "e-0000000000000009")]
        assert self._rows("凯尔希：侯爵没有保护科学家们。", ents=ents, edges=edges) == []

    def test_noun_suffix_is_rejected(self) -> None:
        """`感染者` 里的 `感染` 是名词成分，不是谓词。"""
        ents = [_entity("e-0000000000000001", "源石虫")]
        assert self._rows("香草：源石虫不会感染者。", ents=ents) == []


# ---------------------------------------------------------------- 缺口与重复


class TestCrossChunkCoverage:
    """跨块跨章题：答案列出的章节必须**全部**有引用块支撑。

    这类题的失效形态不是「答案错」，而是「答案对但引用没覆盖到」——答案里写着
    `act30side`，给出的引用块却在别的章节，解题者读完块根本推不出那一章。所以
    校验必须要求「答案章节集合 == 引用块覆盖的章节集合」，而不是只要求子集。
    """

    def _item(self, chapters_in_answer: list[str], cites: list[tuple[str, str, str]]) -> dict:
        """`cites` 是 `(chunk_id, chapter, text)`，据此现场造语料。"""
        chunks = {cid: _chunk(cid, text, chapter=ch) for cid, ch, text in cites}
        item = {
            "question_id": "q-cx-000001",
            "question": "「阿尔法」在哪些章节的正文里出现过？",
            "answer": "、".join(chapters_in_answer),
            "acceptable_answers": ["、".join(chapters_in_answer)],
            "gold_chunk_ids": [cid for cid, _c, _t in cites],
            "gold_assertion_ids": [],
            "required_citations": [
                {"chunk_id": cid, "evidence": "阿尔法"} for cid, _c, _t in cites
            ],
            "difficulty": "medium",
            "category": "cross_chunk",
            "should_abstain": False,
        }
        return {"item": item, "chunks": chunks}

    def test_covered_chapters_pass(self) -> None:
        case = self._item(
            ["act1", "act2"],
            [("c#1", "act1", "阿尔法在此"), ("c#2", "act2", "阿尔法在彼")],
        )
        errs = goldset.validate_gold_item(case["item"], case["chunks"])
        assert errs == []

    def test_answer_chapter_without_a_citation_is_rejected(self) -> None:
        """答案列了三章，只给出两章的引用块 → 必须报错。

        这正是修复前的真实缺陷：`mine_cross_chunk` 只取前 3 章给引用，答案却列
        全部章节，实测 38 处答案章节没有引用块覆盖。
        """
        case = self._item(
            ["act1", "act2", "act3"],
            [("c#1", "act1", "阿尔法在此"), ("c#2", "act2", "阿尔法在彼")],
        )
        errs = goldset.validate_gold_item(case["item"], case["chunks"])
        assert any("未被引用块覆盖" in e for e in errs), errs

    def test_citation_chapter_not_in_answer_is_rejected(self) -> None:
        """引用块覆盖了答案没列的章节 → 答案漏章，同样不一致。"""
        case = self._item(
            ["act1"],
            [("c#1", "act1", "阿尔法在此"), ("c#2", "act9", "阿尔法在彼")],
        )
        errs = goldset.validate_gold_item(case["item"], case["chunks"])
        assert any("不一致" in e for e in errs), errs

    def test_single_chapter_cross_chunk_item_is_rejected(self) -> None:
        case = self._item(
            ["act1"],
            [("c#1", "act1", "阿尔法在此"), ("c#2", "act1", "阿尔法仍在此")],
        )
        errs = goldset.validate_gold_item(case["item"], case["chunks"])
        assert any("只覆盖了" in e for e in errs), errs

    def test_miner_only_emits_chapters_it_can_cite(self) -> None:
        """挖出来的题必须逐条自证：每条的答案章节都被自己的引用块覆盖。"""
        chunk_list = [
            _chunk("c#1", "阿尔法出现在这里", chapter="act1"),
            _chunk("c#2", "阿尔法又出现在这里", chapter="act2"),
        ]
        ent = _entity("e-0000000000000001", "阿尔法", chunks=("c#1", "c#2"))
        items = _numbered(goldset.mine_cross_chunk([ent], chunk_list, limit=5, seed=1))
        assert items, "这个 fixture 应当至少挖出一条"
        cmap = {c["chunk_id"]: c for c in chunk_list}
        for it in items:
            assert goldset.validate_gold_item(it, cmap) == []
            claimed = set(it["answer"].split("、"))
            covered = {cmap[c["chunk_id"]]["chapter"] for c in it["required_citations"]}
            assert claimed == covered

    def test_entities_with_too_many_chapters_are_skipped_not_truncated(self) -> None:
        """章节数超过上限的实体不出题（而非「少列几章」）。

        截断答案会让题目变成「列出**部分**章节」而题面问「全部章节」，那是一个
        自相矛盾的题；正确处置是整条放弃。
        """
        chunk_list = [
            _chunk(f"c#{i}", f"阿尔法在第 {i} 处", chapter=f"act{i}") for i in range(1, 8)
        ]
        ent = _entity(
            "e-0000000000000001", "阿尔法", chunks=tuple(c["chunk_id"] for c in chunk_list)
        )
        items = goldset.mine_cross_chunk([ent], chunk_list, limit=5, seed=1)
        assert items == [], "超过章节上限的实体必须被整条跳过"

        few = chunk_list[:2]
        ent2 = _entity(
            "e-0000000000000002", "贝塔", chunks=tuple(c["chunk_id"] for c in few)
        )
        text_patch = [dict(c, text=c["text"].replace("阿尔法", "贝塔")) for c in few]
        got = goldset.mine_cross_chunk([ent2], text_patch, limit=5, seed=1)
        assert len(got) == 1
        assert set(got[0]["answer"].split("、")) == {"act1", "act2"}


class TestAdversarialLeak:
    """对抗题的题面不许把判断依据印出来。

    第一版模板是「「{a}」（类型：{ta}）与「{b}」（类型：{tb}）是同一个实体吗」——
    答案就是「不是，类型不同」，题面已经说了，不需要读任何原文。这类题在离线自评
    里会拿满分、在真实评测里完全无效。
    """

    def _pair(self) -> tuple[list[dict], list[dict]]:
        chunks = [
            _chunk("c#1", "卡拉出现在了这里", chapter="act1"),
        ]
        ents = [
            _entity("e-0000000000000001", "卡拉顿城镇议会", etype="组织", chunks=("c#1",)),
            _entity("e-0000000000000002", "卡拉", etype="角色", chunks=("c#1",)),
        ]
        return ents, chunks

    def test_miner_question_does_not_reveal_types(self) -> None:
        ents, chunks = self._pair()
        items = _numbered(goldset.mine_adversarial(ents, chunks, limit=4, seed=1))
        assert items, "这对实体应当能出题"
        for it in items:
            assert "类型" not in it["question"]
            for e in ents:
                assert e["type"] not in it["question"], it["question"]
            # 答案里仍然要说清类型差异 + 「原文只出现短名」这个可逐字复核的事实
            assert "组织" in it["answer"] and "角色" in it["answer"]
            assert "只出现" in it["answer"]

    def test_every_adversarial_template_is_leak_free(self) -> None:
        """模板表里**每一条**都不能带类型占位符或「类型」二字。

        只测「当前抽中的那条」是不够的：模板按 sha256 稳定索引轮换，改坏另一条
        在某个 fixture 上根本走不到（这正是第一版漏掉的地方）。
        """
        tmpls = goldset.TEMPLATES["adversarial"]
        assert len(tmpls) >= 2
        for t in tmpls:
            assert "{ta}" not in t and "{tb}" not in t, t
            assert "类型" not in t, t
            rendered = t.format(a="甲", b="乙")
            assert "甲" in rendered and "乙" in rendered

    def test_miner_output_validates_clean(self) -> None:
        ents, chunks = self._pair()
        cmap = {c["chunk_id"]: c for c in chunks}
        by_id = {e["id"]: e for e in ents}
        items = _numbered(goldset.mine_adversarial(ents, chunks, limit=4, seed=1))
        for it in items:
            errs = goldset.validate_gold_item(
                it, cmap, edges=[], entities_by_id=by_id
            )
            assert errs == [], errs

    def test_validator_rejects_type_word_in_question(self) -> None:
        """题面出现「类型」二字就拒——即使没写出具体类型词。

        必须用**不含具体类型词**的题面，否则「类型词泄漏」那条判据会先命中，
        这条判据等于没被测到（第一版就是这样假绿的）。
        """
        ents, chunks = self._pair()
        cmap = {c["chunk_id"]: c for c in chunks}
        by_id = {e["id"]: e for e in ents}
        it = _numbered(goldset.mine_adversarial(ents, chunks, limit=1, seed=1))[0]
        it["question"] = "「卡拉顿城镇议会」与「卡拉」的类型一样吗？"
        errs = goldset.validate_gold_item(it, cmap, edges=[], entities_by_id=by_id)
        assert any("判断依据" in e for e in errs), errs

    def test_validator_rejects_bare_type_word_in_question(self) -> None:
        """不写「类型：」，只把类型词写进题面也要拦。"""
        ents, chunks = self._pair()
        cmap = {c["chunk_id"]: c for c in chunks}
        by_id = {e["id"]: e for e in ents}
        it = _numbered(goldset.mine_adversarial(ents, chunks, limit=1, seed=1))[0]
        it["question"] = "「卡拉顿城镇议会」这个组织与「卡拉」这个角色是同一个吗？"
        errs = goldset.validate_gold_item(it, cmap, edges=[], entities_by_id=by_id)
        assert any("实体类型" in e for e in errs), errs

    def test_validator_rejects_citation_showing_both_names(self) -> None:
        """引用块里两个名字都有 → 证据无法区分两者。

        取一条合法题，只把引用块换成「两名同在」的块：这样测的是校验器本身，
        而不是「挖掘阶段恰好没产出」——后者会让测试在没跑到判据的情况下通过。
        """
        ents, chunks = self._pair()
        cmap = {c["chunk_id"]: c for c in chunks}
        by_id = {e["id"]: e for e in ents}
        it = _numbered(goldset.mine_adversarial(ents, chunks, limit=1, seed=1))[0]
        assert goldset.validate_gold_item(it, cmap, edges=[], entities_by_id=by_id) == []

        cid = "c#both"
        cmap[cid] = _chunk(cid, "卡拉顿城镇议会与卡拉同时出现", chapter="act1")
        it["gold_chunk_ids"] = [cid]
        it["required_citations"] = [{"chunk_id": cid, "evidence": "卡拉顿城镇议会与卡拉"}]
        errs = goldset.validate_gold_item(it, cmap, edges=[], entities_by_id=by_id)
        assert any("无法区分" in e for e in errs), errs


    def test_entity_name_containing_a_type_word_is_not_a_leak(self) -> None:
        """实体名里本来就有类型字（「第四集团军军队哗变事件」）不算泄漏。

        这是真实数据上跑出来的：校验器第一版直接 `t in question`，把这个实体的
        合法题判红了。判据必须先剥掉两个名字再比。
        """
        chunks = [_chunk("c#1", "第四集团军正在集结", chapter="act1")]
        ents = [
            _entity("e-0000000000000001", "第四集团军军队哗变事件", etype="事件", chunks=("c#1",)),
            _entity("e-0000000000000002", "第四集团军", etype="组织", chunks=("c#1",)),
        ]
        cmap = {c["chunk_id"]: c for c in chunks}
        by_id = {e["id"]: e for e in ents}
        items = _numbered(goldset.mine_adversarial(ents, chunks, limit=2, seed=1))
        assert items, "这对实体应当能出题"
        # 「事件」在长名里，但长名是合法题面的一部分，不是判断依据泄漏
        assert "事件" in items[0]["question"]
        for it in items:
            assert goldset.validate_gold_item(
                it, cmap, edges=[], entities_by_id=by_id
            ) == []


class TestGapAndDuplicates:
    """配额不足要如实报缺口；重复问句必须被检出。"""
    def test_quotas_sum_to_total(self) -> None:
        q = goldset._quotas(320)
        assert sum(q.values()) == 320
        assert q["single_chunk"] == 64
        assert q["alias"] == 48
        assert q["adversarial"] == 16

    def test_category_share_sums_to_one(self) -> None:
        assert abs(sum(goldset.CATEGORY_SHARE.values()) - 1.0) < 1e-9
        assert set(goldset.CATEGORY_SHARE) == set(goldset.CATEGORIES)

    def test_missing_category_is_reported_as_gap_not_backfilled(self) -> None:
        """只有语料和断言、没有边与实体：关系类必然 0 条，必须记缺口。"""
        text = "阿尔法隶属于贝塔。贝塔隶属于伽马。"
        chunk = _chunk("c#1", text, title="标题")
        gold = goldset.build_gold_set(
            [_assertion("a1", "c#1", text, "阿尔法隶属于贝塔")],
            [chunk],
            [],
            [],
            {},
            total=40,
            seed=1,
        )
        rep = gold["report"]
        assert rep["total_actual"] < rep["total_requested"]
        assert rep["gaps"], "缺口字段必须非空"
        for cat, gap in rep["gaps"].items():
            assert gap == rep["quotas"][cat] - rep["produced"][cat]
            assert gap > 0
        assert rep["produced"]["alias"] == 0
        assert sum(rep["produced"].values()) == rep["total_actual"]

    def test_duplicate_questions_are_detected(self) -> None:
        items = [
            {"question_id": "q-sc-000001", "question": "同样的问题"},
            {"question_id": "q-sc-000002", "question": "同样的问题"},
            {"question_id": "q-sc-000003", "question": "另一个问题"},
        ]
        dups = goldset.find_duplicate_questions(items)
        assert len(dups) == 1
        assert dups[0]["first"] == "q-sc-000001"
        assert dups[0]["again"] == "q-sc-000002"
        assert dups[0]["question"] == "同样的问题"

    def test_no_duplicates_on_unique_questions(self) -> None:
        items = [
            {"question_id": "q-sc-000001", "question": "甲"},
            {"question_id": "q-sc-000002", "question": "乙"},
        ]
        assert goldset.find_duplicate_questions(items) == []

    def test_real_gold_set_has_no_duplicates(self, real_gold) -> None:
        assert real_gold["report"]["duplicate_questions"] == []


# ---------------------------------------------------------------- 真实数据


class TestRealDataGoldSet:
    """真实数据：数量、八类覆盖、逐条校验。"""

    def test_total_and_no_validation_failures(self, real_gold) -> None:
        rep = real_gold["report"]
        assert rep["validation_failures"] == []
        assert rep["total_actual"] == real_gold["total"]
        assert real_gold["total"] == 40

    def test_first_twenty_items_validate(self, corpus, real_gold, real_chunk_map) -> None:
        by_id = {e["id"]: e for e in corpus["entities"]}
        for it in real_gold["items"][:20]:
            errs = goldset.validate_gold_item(
                it, real_chunk_map, edges=corpus["edges"], entities_by_id=by_id
            )
            assert errs == [], f"{it['question_id']} 校验失败：{errs}"

    def test_every_category_has_at_least_one_item(self, real_gold) -> None:
        cats = {it["category"] for it in real_gold["items"]}
        missing = set(goldset.CATEGORIES) - cats
        assert not missing, f"这些类别一条题都产不出来（可能被过紧的判据枪毙）：{missing}"

    def test_every_item_matches_required_fields(self, real_gold) -> None:
        for it in real_gold["items"]:
            for f in goldset.REQUIRED_FIELDS:
                assert f in it, f"{it.get('question_id')} 缺 {f}"
            assert goldset.QUESTION_ID_RE.match(it["question_id"])
            assert it["answer"] in it["acceptable_answers"]

    def test_question_ids_are_unique(self, real_gold) -> None:
        ids = [it["question_id"] for it in real_gold["items"]]
        assert len(ids) == len(set(ids))

    def test_citations_are_verbatim_in_corpus(self, real_gold, real_chunk_map) -> None:
        """逐条引用都必须在语料正文里能 `str.find` 到——不靠图的自述。"""
        n = 0
        for it in real_gold["items"]:
            for c in it["required_citations"]:
                text = real_chunk_map[c["chunk_id"]]["text"]
                assert text.find(c["evidence"]) >= 0, (
                    f"{it['question_id']} 的引用不在 {c['chunk_id']} 正文里"
                )
                n += 1
        assert n > 0, "一条引用都没有，等于没校验"

    def test_direct_relation_answers_are_in_vocabulary(self, real_gold) -> None:
        dr = [it for it in real_gold["items"] if it["category"] == "direct_relation"]
        assert dr, "直接关系题一条都没有"
        canon = set(vocab.CANONICAL_RELATIONS)
        for it in dr:
            assert it["answer"] in canon

    def test_negation_answers_are_no(self, real_gold) -> None:
        neg = [
            it
            for it in real_gold["items"]
            if it["category"] == "negative_abstain" and not it["should_abstain"]
        ]
        assert neg, "真实数据上一条否定题都产不出来"
        for it in neg:
            assert it["answer"] == "否"

    def test_deterministic_for_same_seed(self, corpus, real_gold) -> None:
        again = goldset.build_gold_set(
            corpus["assertions"],
            corpus["chunks"],
            corpus["edges"],
            corpus["entities"],
            corpus["adjacency"],
            total=40,
            seed=20261101,
        )
        assert [i["question_id"] for i in again["items"]] == [
            i["question_id"] for i in real_gold["items"]
        ]
        assert [i["question"] for i in again["items"]] == [
            i["question"] for i in real_gold["items"]
        ]


# ---------------------------------------------------------------- 存取


class TestPersistence:
    """`save_gold_set` / `load_gold_set` 必须可往返，且默认不写冻结区。"""

    def test_default_path_is_outside_frozen_area(self) -> None:
        p = goldset.default_path()
        assert p.name == "gold-v1.json"
        assert "goldset" in str(p)
        assert "arkrag" not in p.parts[-2:], "Gold Set 不该写进 data/arkrag/"

    def test_roundtrip_in_tmp_path(self, tmp_path) -> None:
        gold = {"schema": "ark-goldset-1", "seed": 1, "total": 0, "quotas": {}, "report": {}, "items": []}
        target = tmp_path / "gold-v1.json"
        wrote = goldset.save_gold_set(gold, target)
        assert str(wrote).startswith(str(tmp_path))
        assert goldset.load_gold_set(target) == gold

    def test_real_gold_set_roundtrip(self, tmp_path, real_gold) -> None:
        target = tmp_path / "sub" / "gold.json"
        target.parent.mkdir()
        wrote = goldset.save_gold_set(real_gold, target)
        assert str(wrote).startswith(str(tmp_path))
        back = goldset.load_gold_set(target)
        assert len(back["items"]) == len(real_gold["items"])
        assert back["report"]["validation_failures"] == []
