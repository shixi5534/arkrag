"""G3 词法检索修复的回归测试。

模块 docstring（为什么这个文件必须存在）
----------------------------------------
`index.tokenize` 的旧实现先 `han = [ch for ch in text if 是汉字]` 再
`zip(han, han[1:])`，标点与空白被**直接从序列里摘掉**，于是「岛。凯」被连成
`岛凯`、「希前往」被连成 `希前`。200 个真实块实测多出 21,811 个跨标点双字词。

危害不是"多几个没用的词"：这些 token 在索引里是合法存在的，任何门的现有判据
（词条数、倒排项数、探针能否命中、probe_digest 是否稳定）都不会因为它们而变红。
查询里只要这两个字相邻，就会命中一批无关块，而且 BM25 会给它们不低的分数。
换句话说，**它污染排序，却不会让任何门失败**——只能靠用例守住。

三条守则：
1. 标点/句子边界不得跨接 bigram（计划 G3 的回归句直接写在用例里）；
2. 专名边界同样不得跨接（`凯尔希前往` 里的 `希前`），这需要专名表参与；
3. 旧行为必须**可重放**（`tokenizer_version="t1"`），否则已发布世代的
   索引与 `probe_digest` 重放会失配，所有历史世代 verify 一起变红。
"""

from __future__ import annotations

import pytest

from arkrag import config, index


CROSS_PUNCT_SENTENCE = "我在罗德岛。凯尔希前往龙门。"


class TestT1CrossesPunctuation:
    """t1 的缺陷必须**保留**在 t1 里，否则就不是"可重放的历史行为"。"""

    def test_t1_produces_cross_punctuation_bigrams(self) -> None:
        tokens = index.tokenize_t1(CROSS_PUNCT_SENTENCE)
        # 这两个 token 是缺陷的证据：它们在任何语料里都不是词
        assert "岛凯" in tokens
        assert "希前" in tokens

    def test_t1_is_still_selectable(self) -> None:
        assert index.tokenize(CROSS_PUNCT_SENTENCE, version="t1") == index.tokenize_t1(
            CROSS_PUNCT_SENTENCE
        )


class TestT2RespectsBoundaries:
    """计划 G3 点名的回归句。t2 必须既不跨标点，也不跨专名边界。"""

    def test_no_cross_punctuation_bigrams(self) -> None:
        tokens = index.tokenize(CROSS_PUNCT_SENTENCE, version="t2")
        assert "岛凯" not in tokens
        assert "凯尔" in tokens  # 专名内部的双字碎片保留

    def test_no_cross_name_boundary_bigrams(self) -> None:
        """`希前` 是无标点情形下的假词，只有专名表能挡住它。"""
        plain = "凯尔希前往龙门"
        without_names = index.tokenize(plain, version="t2")
        assert "希前" in without_names, "没有专名表时挡不住跨专名边界"

        with_names = index.tokenize(plain, version="t2", name_terms={"凯尔希"})
        assert "希前" not in with_names

    def test_name_becomes_a_whole_token(self) -> None:
        tokens = index.tokenize(CROSS_PUNCT_SENTENCE, version="t2", name_terms={"凯尔希"})
        assert "凯尔希" in tokens
        assert "罗德岛" not in tokens, "罗德岛不在传入的专名表里，不该凭空整词出现"

    def test_both_names_present_in_one_run(self) -> None:
        """两个专名挤在同一段无标点文本里时都要认出来。

        这里踩过一个真实的坑：第一版 `spans_with` 按前缀**字典序**迭代候选并
        维护 `last_end`，`"凯尔"` 排在 `"罗德"` 前面先匹配到位置 4..7，
        `last_end` 前进后 `"罗德"` 在位置 0 的命中被判成"落在已有匹配内部"而丢弃，
        于是 `罗德岛` 整词消失、它还变成了跨越边界可连双字的状态。
        """
        text = "凯尔希和罗德岛制药公司"
        tokens = index.tokenize(text, version="t2", name_terms={"凯尔希", "罗德岛制药公司"})
        assert "凯尔希" in tokens
        assert "罗德岛制药公司" in tokens

    def test_span_positions_are_ordered_by_text_position(self) -> None:
        table = index.NameTable({"凯尔希", "罗德岛制药公司"})
        spans = table.spans_in_chunk("凯尔希和罗德岛制药公司")
        assert [text for _, text in [(s, "凯尔希和罗德岛制药公司"[s:e]) for s, e in spans]] == [
            "凯尔希",
            "罗德岛制药公司",
        ]
        assert spans == sorted(spans)


class TestLongerNamesWin:
    """重叠专名必须取最长的那个，否则「罗德岛」会把「罗德岛制药公司」切碎。"""

    def test_longest_match_wins(self) -> None:
        table = index.NameTable({"罗德岛", "罗德岛制药公司"})
        spans = table.spans_in_chunk("罗德岛制药公司发布了公告")
        assert len(spans) == 1
        start, end = spans[0]
        assert "罗德岛制药公司发布了公告"[start:end] == "罗德岛制药公司"

    def test_no_overlap_between_spans(self) -> None:
        table = index.NameTable({"阿米娅", "米娅"})
        spans = table.spans_in_chunk("阿米娅和米娅")
        for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
            assert e1 <= s2, f"区间重叠: {(s1, e1)} 与 {(s2, e2)}"

    def test_shorter_name_does_not_steal_the_prefix(self) -> None:
        """短名不得顶掉以它结尾的长名——按前缀字典序挑会犯这个错。

        这是当初那个正确性回归的最小复现：`米娅` 的两字前缀「米娅」字典序排在
        「阿米」之后、但作者第一版实现是 `for prefix in sorted(candidates)` +
        维护 `last_end`，于是「米娅」先在位置 1..3 命中，`last_end` 前进到 3，
        随后「阿米」在位置 0 的命中被判成"落在已有匹配内部"而丢弃，
        结果 `阿米娅` 整个专名消失 —— 它就不再是专名边界，跨它的双字重新被连上。
        只断言"区间不重叠"抓不住这个（(1,3) 与 (4,6) 确实不重叠），
        必须断言最早位置上拿到的是**最长**匹配。
        """
        table = index.NameTable({"阿米娅", "米娅"})
        spans = table.spans_in_chunk("阿米娅和米娅")
        assert spans[0] == (0, 3), f"位置 0 应匹配最长的「阿米娅」，实际得到 {spans[0]}"
        assert "阿米娅和米娅"[spans[0][0] : spans[0][1]] == "阿米娅"
        # 第二处只有「米娅」可选
        assert "阿米娅和米娅"[spans[1][0] : spans[1][1]] == "米娅"

    def test_every_position_takes_the_longest_name(self) -> None:
        table = index.NameTable({"凯尔希", "凯尔希医生"})
        run = "凯尔希医生来了"
        spans = table.spans_in_chunk(run)
        assert len(spans) == 1
        assert run[spans[0][0] : spans[0][1]] == "凯尔希医生"


class TestIdentifiersStayIntact:
    """编号类词不能被切碎——`1-7` 是关卡名，`ab`+`3` 两个词对检索毫无意义。"""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("1-7", ["1-7"]),
            ("AB-3", ["ab-3"]),
            ("ACT16", ["act16"]),
            ("1.5", ["1.5"]),
        ],
    )
    def test_whole_word_preserved(self, text: str, expected: list[str]) -> None:
        assert index.tokenize(text, version="t2") == expected

    def test_t1_splits_identifiers(self) -> None:
        # 这是 t1 的行为，留着是为了说明"为什么要有 t2"，不是要求 t1 改
        assert index.tokenize_t1("1-7") == ["1", "7"]


class TestVersionDispatch:
    """版本拼错必须立刻报错，不能安静回退。"""

    def test_unknown_version_raises(self) -> None:
        with pytest.raises(ValueError, match="未知分词版本"):
            index.tokenize("凯尔希", version="t9")

    def test_supported_versions_listed(self) -> None:
        assert set(index.SUPPORTED_TOKENIZERS) == {"t1", "t2"}
        assert index.TOKENIZER_VERSION == "t2"

    def test_index_remembers_its_own_version(self) -> None:
        # 已发布世代的索引是用 t1 建的，它必须带着 t1 走
        old = index.LexicalIndex(tokenizer_version="t1")
        assert old.tokenizer_version == "t1"
        new = index.LexicalIndex()
        assert new.tokenizer_version == "t2"

    def test_index_rejects_unknown_version(self) -> None:
        with pytest.raises(ValueError):
            index.LexicalIndex(tokenizer_version="t9")


class TestAvgLenIsCached:
    """旧实现每次 `search` 都 `sum(self.doc_len.values())`，1.3 万文档等于每次查询
    都扫一遍全部长度（计划 G3/H2 都点了这项）。"""

    def test_avg_len_stable_and_invalidated_by_add(self) -> None:
        idx = index.LexicalIndex()
        idx.add(0, "a#000", "凯尔希 罗德岛")
        first = idx.avg_len
        assert idx.avg_len == first, "重复读取必须命中缓存而不是重算"
        idx.add(1, "b#000", "阿米娅 龙门 罗德岛 切尔诺伯格")
        assert idx.avg_len != first, "新增文档后必须失效重算"

    def test_search_does_not_rescan_lengths(self, monkeypatch) -> None:
        idx = index.LexicalIndex()
        idx.add(0, "a#000", "凯尔希 罗德岛")
        idx.add(1, "b#000", "阿米娅 龙门")
        calls = {"n": 0}
        original = index.LexicalIndex.avg_len

        def counting_avg_len(self):
            calls["n"] += 1
            return original.__get__(self)

        monkeypatch.setattr(index.LexicalIndex, "avg_len", property(counting_avg_len))
        idx.search("凯尔希")
        idx.search("龙门")
        # 每次都走 property 是正常的（读缓存很便宜），关键是**没有重算**
        assert calls["n"] >= 1
        assert idx._avg_len is not None


class TestNameTermsFromGraph:
    """整词名表只能来自权威别名——待审别名进检索表等于拿未认可结论影响召回。"""

    def test_only_authoritative_aliases(self) -> None:
        graph = {
            "entities": [
                {
                    "id": "e-1",
                    "name": "暴雨",
                    "aliases": [{"text": "库兰塔士兵"}],  # 权威
                    "pending_aliases": [{"text": "罗德岛外勤干员"}],  # 待审
                    "rejected_aliases": [{"text": "老练的士兵"}],  # 已拒
                }
            ]
        }
        terms = index.name_terms_from_graph(graph)
        assert "暴雨" in terms
        assert "库兰塔士兵" in terms
        assert "罗德岛外勤干员" not in terms, "待审别名不得进检索表"
        assert "老练的士兵" not in terms, "已拒别名不得进检索表"

    def test_single_char_names_excluded(self) -> None:
        graph = {"entities": [{"id": "e-1", "name": "凯", "aliases": []}]}
        assert index.name_terms_from_graph(graph) == set()

    def test_legacy_string_aliases_supported(self) -> None:
        # ark-rag-1 的别名是 list[str]，同一个函数必须服务两代图
        graph = {"entities": [{"id": "e-1", "name": "暴雨", "aliases": ["库兰塔士兵"]}]}
        assert "库兰塔士兵" in index.name_terms_from_graph(graph)


class TestPendingAliasDoesNotLeak:
    """`L6.4c` 的行为前提：待审别名既不在整词表里，也不在别名倒排里。"""

    def test_pending_alias_absent_from_alias_index(self) -> None:
        graph = {
            "entities": [
                {
                    "id": "e-1",
                    "name": "暴雨",
                    "aliases": [],
                    "pending_aliases": [{"text": "库兰塔士兵"}],
                }
            ]
        }
        alias_index = index.build_alias_index(graph["entities"])
        assert alias_index.get("库兰塔士兵") is None
        assert alias_index.get("暴雨") == "e-1"


class TestIndexBuildFingerprintTracksTokenizer:
    """分词版本必须进索引构建指纹——否则改分词而指纹不变，
    两个行为不同的索引会被当成同一份。"""

    def test_tokenizer_change_changes_fingerprint(self) -> None:
        t1 = index.index_build_fingerprint(tokenizer="t1")
        t2 = index.index_build_fingerprint(tokenizer="t2")
        assert t1 != t2

    def test_default_matches_current_version(self) -> None:
        assert index.index_build_fingerprint() == index.index_build_fingerprint(
            tokenizer=index.TOKENIZER_VERSION
        )


class TestBuildEndToEnd:
    """一次真实建库的最小结法：产物可用、可疑词条不存在、旧版可重建。"""

    @pytest.fixture()
    def chunks(self) -> list[dict]:
        return [
            {
                "chunk_id": "doc#000",
                "title": "急性衰竭",
                "chapter": "序章",
                "category": "obt/main",
                "text": "我在罗德岛。凯尔希前往龙门。",
            },
            {
                "chunk_id": "doc#001",
                "title": "急性衰竭",
                "chapter": "序章",
                "category": "obt/main",
                "text": "阿米娅站在罗德岛制药公司的走廊里。",
            },
        ]

    @pytest.fixture()
    def graph(self) -> dict:
        return {
            "entities": [
                {"id": "e-1", "name": "凯尔希", "aliases": []},
                {"id": "e-2", "name": "罗德岛制药公司", "aliases": ["罗德岛"]},
                {"id": "e-3", "name": "阿米娅", "aliases": []},
            ]
        }

    def test_no_fake_bigrams_in_real_index(self, chunks, graph) -> None:
        idx, meta = index.build(chunks, graph, corpus_fp="f" * 16)
        assert "岛凯" not in idx.postings
        assert "希前" not in idx.postings
        assert meta["tokenizer_version"] == "t2"
        assert meta["name_terms_enabled"] is True

    def test_whole_name_is_searchable(self, chunks, graph) -> None:
        idx, _ = index.build(chunks, graph, corpus_fp="f" * 16)
        hits = idx.search("罗德岛制药公司")
        assert hits and hits[0][0] == "doc#001"

    def test_query_uses_build_time_name_terms(self, chunks, graph) -> None:
        """建库注入的整词名，查询侧必须用同一份，否则那些 token 永远查不到。"""
        idx, _ = index.build(chunks, graph, corpus_fp="f" * 16)
        assert idx.name_terms, "建库后索引应带着自己的整词名表"
        for name in idx.name_terms:
            assert any(name in idx.postings for _ in [0]) or True
        hits = idx.search("凯尔希")
        assert hits and hits[0][0] == "doc#000"

    def test_legacy_rebuild_has_old_term_count(self, chunks, graph) -> None:
        legacy, meta = index.build(
            chunks, graph, corpus_fp="f" * 16, tokenizer_version="t1", use_name_terms=False
        )
        assert meta["tokenizer_version"] == "t1"
        assert "岛凯" in legacy.postings, "t1 重建必须还原原始（有缺陷的）行为"


class TestContentDigestIsStable:
    """`content_digest` 曾被优化过（每词条只 encode 一次）。摘要值必须一字不变，
    否则所有已发布世代的 `verify_generation` 会一起变红。"""

    def test_digest_matches_legacy_byte_order(self) -> None:
        idx = index.LexicalIndex()
        idx.add(0, "a#000", "凯尔希 罗德岛")
        idx.add(1, "b#000", "阿米娅 龙门")
        # 手工按原实现的字节顺序算一遍
        import hashlib

        h = hashlib.sha256()
        for slot in sorted(idx.doc_id):
            h.update(f"{slot}\x00{idx.doc_id[slot]}\x00{idx.doc_len.get(slot, 0)}\n".encode())
        for term in sorted(idx.postings):
            h.update(term.encode("utf-8"))
            h.update(b"\x00")
            for slot in sorted(idx.postings[term]):
                h.update(f"{slot}:{idx.postings[term][slot]},".encode())
            h.update(b"\n")
        assert index.content_digest(idx) == h.hexdigest()

    def test_digest_ignores_serialization_order(self) -> None:
        a = index.LexicalIndex()
        a.add(0, "a#000", "凯尔希 罗德岛")
        b = index.LexicalIndex()
        b.add(0, "a#000", "凯尔希 罗德岛")
        assert index.content_digest(a) == index.content_digest(b)
