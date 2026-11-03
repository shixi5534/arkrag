"""`arkrag.vector_store` 与 `arkrag.fusion` 的判据测试。

## 这个文件在守什么

这两个模块都不会"崩"，它们会**静默给出一个看起来正常的错误答案**：

* 向量文件被截断 → `np.frombuffer` 不抛异常，只会得到一堆垃圾分数，
  表现是"召回莫名下降"，没有任何字段变红。所以 sha256 校验是必须的、
  且必须能真的变红（本文件用手工改坏一个字节来证明）。
* 块文本改了一个字而向量没重建 → 检索照常工作，但语义空间与正文脱节。
  所以 `chunk_text_sha256` 必须逐条比对。
* 块被删了而向量没删 → 检索可能返回一个**不存在的 chunk_id**，下游
  `chunks[cid]` 直接 KeyError，或者更糟——返回旧内容。
* RRF 里同一通道输出重复项 → 若按出现次数累加，等于奖励上游的 bug。
* 通道缺席时若不显式申报 → 报告里 `channels_used` 少一行，但没人会去数。

## 关于 numpy

所有依赖 numpy 的用例都 `pytest.importorskip("numpy")`。本项目 numpy 是
**可选依赖**（运行时零第三方依赖原则），所以测试也必须能缺它——
否则"没装 numpy 时测试失败"会被误读成"缺少向量通道是错的"。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arkrag import fusion, vector_store


# --------------------------------------------------------------------------
# 文本定义：全项目只有一处
# --------------------------------------------------------------------------
class TestChunkTextForEmbedding:
    def test_includes_title_chapter_text_in_that_order(self):
        rec = {"title": "篇章名", "chapter": "第一章", "text": "正文内容"}
        assert vector_store.chunk_text_for_embedding(rec) == "篇章名 第一章 正文内容"

    def test_missing_fields_degrade_to_empty_not_crash(self):
        assert vector_store.chunk_text_for_embedding({}) == "  "

    def test_extra_fields_are_ignored(self):
        """`chunk_id`/`doc_id`/`speakers` 等不进编码文本——否则同一段正文会因为
        元数据变化而被编成不同向量。"""
        rec = {"title": "t", "chapter": "c", "text": "x", "chunk_id": "doc/a#000",
               "speakers": "['凯尔希']", "chars": 99}
        assert vector_store.chunk_text_for_embedding(rec) == "t c x"

    def test_is_the_only_definition_used_by_build_and_verify(self):
        """`build_vectors.py` 与 `verify_vectors` 必须调同一个函数。

        用 AST 静态检查 `scripts/build_vectors.py` 里没有自己拼 f-string：
        若有人在那里另写一份，两处文本就可能不一致，而没有任何运行期报错。
        """
        import ast

        src_path = Path(__file__).resolve().parents[1] / "scripts" / "build_vectors.py"
        src = src_path.read_text(encoding="utf-8")
        assert "chunk_text_for_embedding" in src, "build_vectors.py 必须复用唯一文本定义"
        tree = ast.parse(src)
        # 反过来查：不许出现手写的 "title" + "chapter" + "text" 拼接
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.JoinedStr):
                seg = ast.get_source_segment(src, node) or ""
                if "title" in seg and "chapter" in seg and "text" in seg:
                    offenders.append(seg[:80])
        assert not offenders, f"build_vectors.py 里另拼了一份编码文本：{offenders}"


# --------------------------------------------------------------------------
# 侧车与路径
# --------------------------------------------------------------------------
class TestVectorPaths:
    def test_name_uses_content_digest_not_generation_id(self):
        """三个 ark-rag-3 世代内容全同的事实说明：内容相同就该复用同一份向量。"""
        bin_a, json_a = vector_store.vector_paths("bge-m3-latest", "c1fef3588e662271aaaa")
        bin_b, json_b = vector_store.vector_paths("bge-m3-latest", "c1fef3588e662271bbbb")
        assert bin_a == bin_b, "内容摘要相同（前 16 位）就必须指向同一份向量"
        assert bin_a.name.startswith("bge-m3-latest.")
        assert bin_a.suffix == ".bin"
        assert json_a.suffix == ".json"

    def test_different_digest_yields_different_file(self):
        bin_a, _ = vector_store.vector_paths("m", "aaaaaaaaaaaaaaaa")
        bin_b, _ = vector_store.vector_paths("m", "bbbbbbbbbbbbbbbb")
        assert bin_a != bin_b

    def test_different_model_yields_different_file(self):
        bin_a, _ = vector_store.vector_paths("model-a", "c1fef3588e662271")
        bin_b, _ = vector_store.vector_paths("model-b", "c1fef3588e662271")
        assert bin_a != bin_b

    def test_slug_sanitizes_colon(self):
        assert vector_store._slug("bge-m3:latest") == "bge-m3-latest"

    def test_slug_sanitizes_slash(self):
        assert "/" not in vector_store._slug("org/model:tag")

    def test_slug_never_returns_empty(self):
        assert vector_store._slug(":::") != ""


# --------------------------------------------------------------------------
# 落盘 / 读回的完整性
# --------------------------------------------------------------------------
class TestRoundTrip:
    @pytest.fixture(autouse=True)
    def _np(self):
        pytest.importorskip("numpy")

    def _save(self, tmp_path, n=4, dim=8, **over):
        vecs = [[float(i + j) / 10.0 for j in range(dim)] for i in range(n)]
        cids = [f"doc/a#{i:03d}" for i in range(n)]
        texts = [f"文本{i}" for i in range(n)]
        kw = dict(model="bge-m3:latest", model_digest="d" * 64, dim=dim,
                  generation_id="g-test", index_content_digest="c1fef3588e662271",
                  normalized=True, root=tmp_path)
        kw.update(over)
        return vector_store.save_vectors(vecs, cids, texts, **kw)

    def test_round_trip_preserves_values_and_order(self, tmp_path):
        bin_path, meta_path, meta = self._save(tmp_path)
        matrix, loaded = vector_store.load_vectors(bin_path, meta_path)
        assert list(loaded["chunk_ids"]) == [f"doc/a#{i:03d}" for i in range(4)]
        assert matrix.shape == (4, 8)
        assert loaded["model"] == "bge-m3:latest"
        assert loaded["dim"] == 8
        assert loaded["count"] == 4

    def test_meta_carries_identity_ground_truth(self, tmp_path):
        """模型 / digest / 维度三样必须落盘——它们定义向量空间。"""
        _bin, _meta_path, meta = self._save(tmp_path)
        assert meta["model"] == "bge-m3:latest"
        assert meta["model_digest"] == "d" * 64
        assert meta["dim"] == 8
        assert meta["schema_version"] == vector_store.VECTOR_SCHEMA_VERSION

    def test_meta_carries_per_chunk_text_hash(self, tmp_path):
        _bin, _meta_path, meta = self._save(tmp_path)
        assert len(meta["chunk_text_sha256"]) == 4
        assert meta["chunk_text_sha256"][0] == vector_store.text_sha256("文本0")

    def test_count_mismatch_is_rejected_before_writing(self, tmp_path):
        """条数不一致时必须**先失败再落盘**，不能留半成品。"""
        vecs = [[0.0] * 8 for _ in range(3)]
        with pytest.raises(ValueError, match="条数不一致"):
            vector_store.save_vectors(
                vecs, ["a", "b"], ["x", "y"], model="m", model_digest="",
                dim=8, generation_id="g", index_content_digest="c" * 16,
                normalized=True, root=tmp_path,
            )
        assert not list(tmp_path.glob("*.bin")), "校验失败时不许留下 .bin"

    def test_empty_vector_set_is_rejected(self, tmp_path):
        """空集不许落盘——那等于申报"有向量通道"却没有向量。"""
        with pytest.raises(ValueError, match="空向量集"):
            vector_store.save_vectors(
                [], [], [], model="m", model_digest="", dim=8, generation_id="g",
                index_content_digest="c" * 16, normalized=True, root=tmp_path,
            )

    def test_wrong_dimension_is_rejected(self, tmp_path):
        vecs = [[0.0] * 7 for _ in range(2)]
        with pytest.raises(ValueError, match="维度"):
            vector_store.save_vectors(
                vecs, ["a", "b"], ["x", "y"], model="m", model_digest="",
                dim=8, generation_id="g", index_content_digest="c" * 16,
                normalized=True, root=tmp_path,
            )


# --------------------------------------------------------------------------
# 损坏检测：这是本文最重要的一组
# --------------------------------------------------------------------------
class TestCorruptionDetection:
    @pytest.fixture(autouse=True)
    def _np(self):
        pytest.importorskip("numpy")

    def _save(self, tmp_path):
        vecs = [[float(i)] * 4 for i in range(3)]
        return vector_store.save_vectors(
            vecs, ["a", "b", "c"], ["x", "y", "z"], model="m", model_digest="",
            dim=4, generation_id="g", index_content_digest="c" * 16,
            normalized=False, root=tmp_path,
        )

    def test_truncated_matrix_is_detected(self, tmp_path):
        """截断后 `np.frombuffer` 不抛异常，只会得到垃圾分数——必须靠字节数发现。"""
        bin_path, meta_path, _meta = self._save(tmp_path)
        payload = bin_path.read_bytes()
        bin_path.write_bytes(payload[:-8])
        with pytest.raises(ValueError, match="字节数不符"):
            vector_store.load_vectors(bin_path, meta_path)

    def test_flipped_byte_is_detected(self, tmp_path):
        """改坏一个字节（字节数不变）只有 sha256 能发现。"""
        bin_path, meta_path, _meta = self._save(tmp_path)
        payload = bytearray(bin_path.read_bytes())
        payload[0] ^= 0xFF
        bin_path.write_bytes(bytes(payload))
        with pytest.raises(ValueError, match="sha256"):
            vector_store.load_vectors(bin_path, meta_path)

    def test_tampered_meta_count_is_detected(self, tmp_path):
        """侧车被改（count 变大）→ 元素数对不上，必须报错。"""
        bin_path, meta_path, _meta = self._save(tmp_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["count"] = 99
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ValueError, match="元素数"):
            vector_store.load_vectors(bin_path, meta_path)


# --------------------------------------------------------------------------
# 一一对应校验
# --------------------------------------------------------------------------
class TestVerifyVectors:
    def _meta(self, cids, texts):
        return {
            "count": len(cids),
            "chunk_ids": list(cids),
            "chunk_text_sha256": [vector_store.text_sha256(t) for t in texts],
        }

    def test_exact_match_passes(self):
        cids = ["a", "b"]
        texts = {"a": "文本 a", "b": "文本 b"}
        report = vector_store.verify_vectors(self._meta(cids, texts.values()), cids, texts)
        assert report["ok"] is True
        assert report["problems"] == []

    def test_missing_chunk_is_reported(self):
        """块新增了但没有向量 → 必须报出来，不许静默用旧向量凑。"""
        report = vector_store.verify_vectors(
            self._meta(["a"], ["文本 a"]), ["a", "b"], {"a": "文本 a", "b": "文本 b"}
        )
        assert report["ok"] is False
        assert report["missing"] == 1
        assert report["current_count"] == 2

    def test_orphan_vector_is_reported(self):
        """块删了但向量还在 → 检索可能返回不存在的 chunk_id。"""
        report = vector_store.verify_vectors(
            self._meta(["a", "b"], ["文本 a", "文本 b"]), ["a"], {"a": "文本 a"}
        )
        assert report["ok"] is False
        assert report["orphan"] == 1

    def test_changed_text_is_reported(self):
        """正文改了一个字而向量没重建 → 语义空间与正文脱节。"""
        meta = self._meta(["a"], ["旧正文"])
        report = vector_store.verify_vectors(meta, ["a"], {"a": "新正文"})
        assert report["ok"] is False
        assert report["mismatched"] == 1

    def test_problems_list_is_never_empty_when_not_ok(self):
        """`ok=False` 时必须有可读原因——空列表会让调用方无从报告。"""
        report = vector_store.verify_vectors(self._meta(["a"], ["x"]), ["a", "b"],
                                             {"a": "x", "b": "y"})
        assert report["ok"] is False
        assert report["problems"]


# --------------------------------------------------------------------------
# VectorIndex 检索语义
# --------------------------------------------------------------------------
class TestVectorIndexSearch:
    @pytest.fixture(autouse=True)
    def _np(self):
        self.np = pytest.importorskip("numpy")

    def _index(self, rows, cids):
        matrix = self.np.asarray(rows, dtype=self.np.float32)
        meta = {"chunk_ids": cids, "dim": len(rows[0]), "model": "m", "model_digest": ""}
        return vector_store.VectorIndex(matrix, meta)

    def test_highest_cosine_comes_first(self):
        idx = self._index([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], ["a", "b", "c"])
        ranked = idx.search([1.0, 0.0], top_k=3)
        assert ranked[0][0] == "a"

    def test_query_is_renormalized(self):
        """查询向量未归一化时也必须按余弦排序，不能按点积（长度会污染排序）。"""
        idx = self._index([[1.0, 0.0], [0.9, 0.9]], ["a", "b"])
        # 未归一化的查询，方向与 a 一致
        ranked_norm = idx.search([10.0, 0.0], top_k=2)
        ranked_unit = idx.search([1.0, 0.0], top_k=2)
        assert [c for c, _ in ranked_norm] == [c for c, _ in ranked_unit]

    def test_dimension_mismatch_raises(self):
        idx = self._index([[1.0, 0.0]], ["a"])
        with pytest.raises(ValueError, match="维度"):
            idx.search([1.0, 0.0, 0.0], top_k=1)

    def test_tie_break_is_chunk_id_ascending(self):
        """同分按 chunk_id 升序——与词法索引一致，否则排序不可复现。"""
        idx = self._index([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], ["c", "a", "b"])
        ranked = idx.search([1.0, 0.0], top_k=3)
        assert [c for c, _ in ranked] == ["a", "b", "c"]

    def test_top_k_larger_than_index_is_clamped(self):
        idx = self._index([[1.0, 0.0]], ["a"])
        assert len(idx.search([1.0, 0.0], top_k=50)) == 1

    def test_len_matches_chunk_count(self):
        idx = self._index([[1.0, 0.0], [0.0, 1.0]], ["a", "b"])
        assert len(idx) == 2


# --------------------------------------------------------------------------
# RRF 融合
# --------------------------------------------------------------------------
class TestRrfFuse:
    def test_single_channel_preserves_order(self):
        out = fusion.rrf_fuse({"a": ["x", "y", "z"]}, top_k=3)
        assert [r["chunk_id"] for r in out] == ["x", "y", "z"]

    def test_agreement_across_channels_wins(self):
        """两路都排第一的块必须压过只有一路排第一的块。"""
        out = fusion.rrf_fuse({"a": ["p", "q"], "b": ["p", "r"]}, top_k=3)
        assert out[0]["chunk_id"] == "p"

    def test_channel_present_in_more_lists_scores_higher_than_single_first(self):
        """两路各排第 30 的块得分高于一路排第 1 —— 这是 RRF 的设计后果，
        也是它在「上下文只装 4.7 块」的项目里失败的原因（记在这里防止有人误以为是 bug）。

        算术（k=60，生产默认）：rank1 单路贡献 1/61 = 0.01639；
        rank30 两路贡献 2 × 1/90 = 0.02222。**0.02222 > 0.01639。**
        k=1 时两者恰好都等于 0.5 —— 所以这个现象只在 k 较大时出现，
        这也是 `sweep_rrf_k.py` 里 k 越小越接近基线的原因。
        """
        def with_m_at_rank30(tag):
            return [f"{tag}top"] + [f"{tag}p{i}" for i in range(28)] + ["m"]

        out = fusion.rrf_fuse({"a": with_m_at_rank30("a"), "b": with_m_at_rank30("b")},
                              k=fusion.RRF_K, top_k=40)
        by_id = {r["chunk_id"]: r["score"] for r in out}
        assert by_id["m"] > by_id["atop"], (
            f"rank30 双路 {by_id['m']} 应大于 rank1 单路 {by_id['atop']}"
        )

    def test_duplicate_within_channel_counts_once(self):
        """同通道重复只算一次——否则等于奖励上游 bug。"""
        out = fusion.rrf_fuse({"a": ["x", "x", "x"]}, top_k=3)
        assert len(out) == 1
        assert out[0]["channels"]["a"]["rank"] == 1

    def test_missing_channel_gets_no_zero_penalty(self):
        only_a = fusion.rrf_fuse({"a": ["x"]}, top_k=1)
        with_b = fusion.rrf_fuse({"a": ["x"], "b": []}, top_k=1)
        assert only_a[0]["score"] == with_b[0]["score"]

    def test_top_k_is_respected(self):
        out = fusion.rrf_fuse({"a": ["1", "2", "3", "4", "5"]}, top_k=2)
        assert len(out) == 2

    def test_zero_k_raises(self):
        with pytest.raises(ValueError, match="k 必须为正"):
            fusion.rrf_fuse({"a": ["x"]}, k=0)

    def test_tie_break_is_chunk_id_ascending(self):
        out = fusion.rrf_fuse({"a": ["b", "a"]}, top_k=2)
        # 两者贡献相同（rank1 vs rank2 不同，故改造成同分场景）
        out2 = fusion.rrf_fuse({"a": ["z"], "b": ["a"]}, top_k=2)
        assert [r["chunk_id"] for r in out2] == ["a", "z"]

    def test_channels_detail_is_reported(self):
        out = fusion.rrf_fuse({"a": ["x"], "b": ["x"]}, top_k=1)
        assert set(out[0]["channels"]) == {"a", "b"}
        assert out[0]["channels"]["a"]["rank"] == 1

    def test_scores_are_deterministic(self):
        r1 = fusion.rrf_fuse({"a": ["x", "y"], "b": ["y", "x"]}, top_k=2)
        r2 = fusion.rrf_fuse({"a": ["x", "y"], "b": ["y", "x"]}, top_k=2)
        assert r1 == r2


class TestOverlap:
    def test_identical_lists(self):
        rep = fusion.overlap(["a", "b", "c"], ["a", "b", "c"], top=3)
        assert rep["common"] == 3
        assert rep["only_a"] == 0
        assert rep["jaccard"] == 1.0

    def test_disjoint_lists(self):
        rep = fusion.overlap(["a"], ["b"], top=1)
        assert rep["common"] == 0
        assert rep["jaccard"] == 0.0

    def test_top_truncates_both_side(self):
        rep = fusion.overlap(["a", "b", "c"], ["a", "x", "y"], top=1)
        assert rep["common"] == 1

    def test_empty_input_is_safe(self):
        assert fusion.overlap([], [], top=5)["jaccard"] == 0.0


# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------
class TestConstants:
    def test_schema_version_is_pinned(self):
        assert vector_store.VECTOR_SCHEMA_VERSION == "ark-rag-vec-1"

    def test_dtype_is_float32(self):
        """float32 是刻意的：float64 会让矩阵 112 MB 而召回无差别。"""
        assert vector_store.ELEMENT_DTYPE == "float32"
        assert vector_store.ELEMENT_BYTES == 4

    def test_rrf_k_default_is_paper_value(self):
        assert fusion.RRF_K == 60

    def test_channel_k_default(self):
        assert fusion.DEFAULT_CHANNEL_K == 100

    def test_vector_source_name(self):
        assert fusion.SOURCE_VECTOR == "vector"
