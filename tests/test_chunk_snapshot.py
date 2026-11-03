"""块正文必须跟着世代走，且消费者必须**证明**自己拿到的是那一批。

## 这个文件在守什么

K2 之前，块正文只有一份全局文件 `data/arkrag/chunks.jsonl`（L1 的最新产物），
世代目录里没有。于是「这一代声称吃了哪些块」与「消费者实际能读到的块正文」
是两份数据。实测后果（K2 候选 `27be38`，块数 13,798 vs 全局文件 13,781）：

* 17 块索引里有、全局文件里没有；
* 17 块两边都有但正文变了（`content_key` 与字数全变）；
* Gold Set 冒出 1 条引用硬失败（`q-sc-000024`），`text[95:114]` 切出来是
  另一句话。

坏引力在于**它不报错**：引用校验拿到对不上的切片只会把引用静默剔除、置信度
降一档，表现为「答得短」而不是「服务坏了」。所以这里守的是四条硬规矩：

1. 建造期把块正文写进 `<gen>/chunks.jsonl`——只有建造者知道吃的是哪批块。
2. 自检核这份文件的身份（`chunk_manifest_hash`）——文件在盘上不等于内容对。
3. 消费侧 `resolve_chunks()` 只认自带那份；旧世代借全局文件必须能证明是同一批，
   证明不了就 fail-closed，不许"先用了再说"。
4. 断言证据偏移**不许跨块登记**——宿主块载入不回来就不登记，绝不退回当前单元。
   退回等于把 A 块的偏移套在 B 块的正文上（同一条 `q-sc-000024`）。
"""
from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from arkrag import config, index as index_mod, pipeline, release, retrieve, vocab

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _load_build_script():
    spec = importlib.util.spec_from_file_location(
        "build_generation_for_chunks_test", ROOT / "scripts" / "build_generation.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CHUNKS = [
    {
        "chunk_id": f"doc/a#{i:03d}",
        "doc_id": "doc/a",
        "content_key": f"ck{i:04d}" * 4,
        "text": "凯尔希：切尔诺伯格已经沦陷了。\n阿米娅：我们必须去罗德岛。",
        "title": "切尔诺伯格",
        "chapter": "第一章",
        "category": "obt/main",
        "scene_id": f"doc/a#{i}",
        "chars": 30,
    }
    for i in range(3)
]


# ---------------------------------------------------------------- 消费侧：定位


class TestResolveChunks:
    def _gen(self, tmp_path, chunks, *, manifest_extra=None, own=True):
        gen = release.Generation("g-20260101T000000Z-cccccc")
        gen.dir.mkdir(parents=True, exist_ok=True)
        manifest = {"generation_id": gen.generation_id}
        if chunks is not None:
            manifest["chunk_manifest_hash"] = release.chunk_manifest_hash(chunks)
        manifest.update(manifest_extra or {})
        gen.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        if own:
            gen.chunks_path.write_text(
                "\n".join(json.dumps(c, ensure_ascii=False) for c in (chunks or [])) + "\n",
                encoding="utf-8",
            )
        return gen, manifest

    def test_prefers_the_generations_own_chunks(self, tmp_path, monkeypatch):
        monkeypatch.setattr(release, "GENERATIONS_DIR", tmp_path / "generations")
        shared = tmp_path / "shared" / "chunks.jsonl"
        shared.parent.mkdir(parents=True, exist_ok=True)
        shared.write_text("", encoding="utf-8")
        monkeypatch.setattr(config, "chunks_path", lambda: str(shared))

        gen, manifest = self._gen(tmp_path / "generations", CHUNKS)
        assert pipeline.resolve_chunks(gen, manifest) == gen.chunks_path

    def test_own_chunks_with_wrong_hash_is_refused(self, tmp_path, monkeypatch):
        """文件在盘上 ≠ 内容对。建造后被替换过就必须拒，不能"读得到就用"。"""
        monkeypatch.setattr(release, "GENERATIONS_DIR", tmp_path / "generations")
        gen, manifest = self._gen(tmp_path / "generations", CHUNKS)
        # 篡改正文但不改 Manifest
        tampered = [dict(c) for c in CHUNKS]
        tampered[0]["content_key"] = "ffffffffffffffffffffffff"
        gen.chunks_path.write_text(
            "\n".join(json.dumps(c, ensure_ascii=False) for c in tampered) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(pipeline.ChunksUnavailable) as exc:
            pipeline.resolve_chunks(gen, manifest)
        assert "不符" in str(exc.value)

    def test_own_chunks_empty_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(release, "GENERATIONS_DIR", tmp_path / "generations")
        gen, manifest = self._gen(tmp_path / "generations", [], own=True)
        with pytest.raises(pipeline.ChunksUnavailable):
            pipeline.resolve_chunks(gen, manifest)

    def test_legacy_generation_may_borrow_shared_when_hash_matches(self, tmp_path, monkeypatch):
        """旧世代没有自带块文件：能证明是同一批才许借。"""
        monkeypatch.setattr(release, "GENERATIONS_DIR", tmp_path / "generations")
        shared = tmp_path / "shared" / "chunks.jsonl"
        shared.parent.mkdir(parents=True, exist_ok=True)
        shared.write_text(
            "\n".join(json.dumps(c, ensure_ascii=False) for c in CHUNKS) + "\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(config, "chunks_path", lambda: str(shared))

        gen, manifest = self._gen(tmp_path / "generations", CHUNKS, own=False)
        assert pipeline.resolve_chunks(gen, manifest) == __import__("pathlib").Path(shared)

    def test_legacy_generation_with_mismatched_shared_is_refused(self, tmp_path, monkeypatch):
        """K2 的现场：候选世代 13,798 块，全局文件 13,781 块。必须拒。"""
        monkeypatch.setattr(release, "GENERATIONS_DIR", tmp_path / "generations")
        shared = tmp_path / "shared" / "chunks.jsonl"
        shared.parent.mkdir(parents=True, exist_ok=True)
        other = CHUNKS[:2]  # 少一块 = 另一批
        shared.write_text(
            "\n".join(json.dumps(c, ensure_ascii=False) for c in other) + "\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(config, "chunks_path", lambda: str(shared))

        gen, manifest = self._gen(tmp_path / "generations", CHUNKS, own=False)
        with pytest.raises(pipeline.ChunksUnavailable) as exc:
            pipeline.resolve_chunks(gen, manifest)
        assert "拒绝用错块做引用校验" in str(exc.value)

    def test_missing_shared_and_missing_own_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(release, "GENERATIONS_DIR", tmp_path / "generations")
        monkeypatch.setattr(config, "chunks_path", lambda: str(tmp_path / "nope.jsonl"))
        gen, manifest = self._gen(tmp_path / "generations", CHUNKS, own=False)
        with pytest.raises(pipeline.ChunksUnavailable):
            pipeline.resolve_chunks(gen, manifest)


# ---------------------------------------------------------------- 自检：身份


class TestVerifyChunksIdentity:
    def _generation_with_chunks(self, tmp_path, monkeypatch, chunks, *, write_own=True):
        monkeypatch.setattr(release, "GENERATIONS_DIR", tmp_path / "generations")
        gen = release.Generation("g-20260101T000000Z-dddddd")
        gen.dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "generation_id": gen.generation_id,
            "chunk_manifest_hash": release.chunk_manifest_hash(chunks),
        }
        gen.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        # `verify_generation` 的摘要门只在**文件缺失**时早退（`artifact_digests`
        # 抛 ReleaseError），早退了就永远走不到块正文那一段。这里补两个占位
        # 产物：它们的摘要与 Manifest 不符会各自记成问题，但流程继续。
        gen.graph_path.parent.mkdir(parents=True, exist_ok=True)
        gen.graph_path.write_text("{}", encoding="utf-8")
        gen.index_path.parent.mkdir(parents=True, exist_ok=True)
        gen.index_path.write_text("{}", encoding="utf-8")
        if write_own:
            gen.chunks_path.write_text(
                "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks) + "\n",
                encoding="utf-8",
            )
        return gen

    def test_matching_chunks_file_produces_no_chunk_problem(self, tmp_path, monkeypatch):
        """只有块正文这一层是干净的：图/索引缺件会各自报，但不能报块正文。"""
        gen = self._generation_with_chunks(tmp_path, monkeypatch, CHUNKS)
        problems = release.verify_generation(gen)
        assert not [p for p in problems if "块正文" in p], problems

    def test_tampered_chunks_file_is_reported(self, tmp_path, monkeypatch):
        gen = self._generation_with_chunks(tmp_path, monkeypatch, CHUNKS)
        tampered = [dict(c) for c in CHUNKS]
        tampered[1]["content_key"] = "000000000000000000000000"
        gen.chunks_path.write_text(
            "\n".join(json.dumps(c, ensure_ascii=False) for c in tampered) + "\n",
            encoding="utf-8",
        )
        problems = release.verify_generation(gen)
        assert any("块正文与 Manifest 不符" in p for p in problems), problems

    def test_unparseable_chunks_file_is_reported_not_raised(self, tmp_path, monkeypatch):
        """它的职责是报告这份产物不合格，自己崩掉就变成探针 500。"""
        gen = self._generation_with_chunks(tmp_path, monkeypatch, CHUNKS)
        gen.chunks_path.write_bytes(b"\xff\xfe not utf-8\x00")
        problems = release.verify_generation(gen)
        assert any("块正文不可解析" in p for p in problems), problems

    def test_generation_without_chunks_file_adds_no_problem(self, tmp_path, monkeypatch):
        """旧世代没有这个文件，不能因此判红——它们由消费侧 fail-closed。"""
        gen = self._generation_with_chunks(tmp_path, monkeypatch, CHUNKS, write_own=False)
        problems = release.verify_generation(gen)
        assert not [p for p in problems if "块正文" in p], problems


# ---------------------------------------------------------------- 建造期：落盘


class TestBuildWritesChunksSnapshot:
    def test_build_writes_chunks_that_reproduce_the_manifest_hash(self, tmp_path, monkeypatch):
        """端到端：建一代，自带块正文必须能复现 Manifest 的 chunk_manifest_hash。

        这条同时证明两件事：文件写了，且写的是**同一批**块（顺序、字段都对齐）。
        """
        mod = _load_build_script()
        data_dir = (tmp_path / "data").resolve()
        data_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(mod.config, "DATA_DIR", data_dir)
        # 世代/运行/指针必须一起隔离：只改 DATA_DIR 不够，`release` 的那三个
        # 常量是模块级的。**首版漏了这三行，真的把测试世代写进了
        # data/arkrag/generations**（`g-20261103T133302Z-eb3c1c`），清理时
        # 才发现的。
        monkeypatch.setattr(release, "GENERATIONS_DIR", data_dir / "generations")
        monkeypatch.setattr(release, "RUNS_DIR", data_dir / "runs")
        monkeypatch.setattr(release, "CURRENT_PATH", data_dir / "current.json")
        (data_dir / "generations").mkdir(parents=True, exist_ok=True)
        (data_dir / "runs").mkdir(parents=True, exist_ok=True)
        (data_dir / "source_verification.json").write_text(
            json.dumps({"status": "verified_zip", "checks": [], "problems": []}), encoding="utf-8"
        )
        (data_dir / "corpus_meta.json").write_text(
            json.dumps({"corpus_fp": "corpusfp01234567", "source_status": "verified_zip"}),
            encoding="utf-8",
        )
        monkeypatch.setattr(mod.extract, "load_chunks", lambda *a, **k: [dict(c) for c in CHUNKS])
        monkeypatch.setattr(mod.cache, "is_cached_for_chunk", lambda *a, **k: True)
        monkeypatch.setattr(
            mod.graph,
            "load_records",
            lambda chunks: [
                {
                    "chunk_id": c["chunk_id"],
                    "raw": {
                        "entities": [{"name": "凯尔希", "type": "角色"}, {"name": "罗德岛", "type": "组织"}],
                        "relations": [
                            {
                                "source": "凯尔希",
                                "relation": "隶属",
                                "target": "罗德岛",
                                "evidence": "我们必须去罗德岛。",
                                "confidence": 0.9,
                            }
                        ],
                    },
                }
                for c in chunks
            ],
        )

        from arkrag import gates

        def _passing(stage: str, n: int):
            def factory(*_a, **_k):
                run = gates.GateRun()
                for i in range(n):
                    run.add(gates.Gate(f"{stage}.{i}", stage, "替身", "1", "PASS", "替身门"))
                return run

            return factory

        monkeypatch.setattr(mod.gates, "l4_gates", _passing("L4", 1))
        monkeypatch.setattr(mod.gates, "l5_gates", _passing("L5", 1))
        monkeypatch.setattr(mod.gates, "l6_gates", _passing("L6", 1))

        import argparse

        args = argparse.Namespace(probes=3, no_publish=True, allow_incomplete=False, graph_model="identity")
        assert mod.build(args) == 0

        gens = [g for g in release.list_generations() if g["status"] == release.READY]
        assert len(gens) == 1, gens
        gen = release.Generation(gens[0]["generation_id"])
        assert gen.chunks_path.exists(), "建造期必须把块正文写进世代目录"

        manifest = gen.load()
        written = [
            json.loads(line)
            for line in gen.chunks_path.read_text("utf-8").splitlines()
            if line.strip()
        ]
        assert release.chunk_manifest_hash(written) == manifest["chunk_manifest_hash"]
        assert {c["chunk_id"] for c in written} == {c["chunk_id"] for c in CHUNKS}


# ---------------------------------------------------------------- 证据偏移不许跨块


def _chunk(cid: str, text: str) -> dict:
    return {
        "chunk_id": cid,
        "doc_id": cid.split("#")[0],
        "title": "测试篇",
        "chapter": "第一章",
        "text": text,
        "chars": len(text),
    }


CHUNK_HOST = "凯尔希：切尔诺伯格已经沦陷了。\n阿米娅：我们必须去罗德岛。"
CHUNK_OTHER = "档案：今天天气不错，无人在场。"


def _retriever(loaded_chunk_ids: list[str]):
    host = _chunk("doc/a#001", CHUNK_HOST)
    other = _chunk("doc/a#000", CHUNK_OTHER)
    all_chunks = [host, other]
    entities = [
        {"id": "e-0000000000000001", "name": "凯尔希", "type": "角色", "aliases": []},
        {"id": "e-0000000000000002", "name": "罗德岛", "type": "组织", "aliases": []},
    ]
    evidence = "我们必须去罗德岛。"
    start = CHUNK_HOST.find(evidence)
    asserts = [
        {
            "assertion_id": "a-0000000000000001",
            "subject_id": "e-0000000000000001",
            "predicate": "隶属",
            "object_id": "e-0000000000000002",
            "chunk_id": "doc/a#001",
            "evidence_text": evidence,
            "evidence_start": start,
            "evidence_end": start + len(evidence),
            "confidence": 0.9,
        }
    ]
    edges = [
        {
            "edge_id": "r-0000000000000001",
            "subject_id": "e-0000000000000001",
            "predicate": "隶属",
            "object_id": "e-0000000000000002",
            "assertion_ids": ["a-0000000000000001"],
            "support_count": 3,
            "independent_scene_count": 1,
            "confidence_summary": {"min": 0.9, "max": 0.9, "mean": 0.9},
            "evidence": evidence,
            "chunks": ["doc/a#001"],
            "scenes": ["doc/a"],
        }
    ]
    graph = {
        "schema_version": config.GRAPH_SCHEMA_IDENTITY,
        "entities": entities,
        "edges": edges,
        "alias_index": index_mod.build_alias_index(entities),
        "fingerprint": "0123456789abcdef",
        "stats": {"entities": 2, "edges": 1},
    }
    from arkrag import adjacency as adjacency_mod

    adj = adjacency_mod.build_directed_adjacency(edges)
    idx, _meta = index_mod.build(all_chunks, graph, "corpusfp01234567")
    chunks = {c["chunk_id"]: c for c in all_chunks if c["chunk_id"] in loaded_chunk_ids}
    return (
        retrieve.Retriever(idx, graph, adj, chunks=chunks, assertions=asserts, vocab=vocab),
        edge_report(),
    )


def edge_report() -> list[dict]:
    """模拟 `expand()` 的输出：一条边命中 `doc/a#000`，但它唯一的断言宿主是 `doc/a#001`。

    这正是缺陷现场：宿主块没载入，旧实现把 `#001` 的偏移登记到 `#000` 上。
    """
    evidence = "我们必须去罗德岛。"
    start = CHUNK_HOST.find(evidence)
    return [
        {
            "chunk_id": "doc/a#000",
            "from_name": "凯尔希",
            "relation": "隶属",
            "neighbor_name": "罗德岛",
            "anchors": [
                {
                    "assertion_id": "a-0000000000000001",
                    "chunk_id": "doc/a#001",
                    "evidence": evidence,
                    "start": start,
                    "end": start + len(evidence),
                }
            ],
        }
    ]


class TestEvidenceSpanNeverCrossesChunk:
    def test_host_chunk_not_loaded_registers_no_span(self):
        """宿主块载入不回来就不登记，绝不退回当前单元。

        退回的后果实测过：`text[95:114]` 在错块上切出另一句话，引用被静默剔除，
        表现为"答得短"。
        """
        ret, expansions = _retriever(["doc/a#000"])  # 宿主 #001 不在
        ctx = ret.build_context([], expansions)
        unit = next(u for u in ctx["units"] if u["chunk_id"] == "doc/a#000")
        assert unit["evidence_spans"] == [], "不许把 #001 的偏移登记到 #000 上"
        assert unit["assertions"] == []

    def test_host_chunk_loaded_registers_span_on_the_host(self):
        """正常路径：宿主块在，偏移必须登记到宿主块上（不是扩展命中的那块）。"""
        ret, expansions = _retriever(["doc/a#000", "doc/a#001"])
        ctx = ret.build_context([], expansions)
        by_id = {u["chunk_id"]: u for u in ctx["units"]}
        assert by_id["doc/a#001"]["evidence_spans"], "宿主块应当拿到证据区间"
        assert by_id["doc/a#000"]["evidence_spans"] == []
        span = by_id["doc/a#001"]["evidence_spans"][0]
        text = by_id["doc/a#001"]["text"]
        assert text[span["start"]:span["end"]] == span["evidence"]

    def test_every_span_slice_matches_its_own_chunk_text(self):
        """不变量：任何登记下来的区间，在它所在块的正文里都必须切出那句证据。

        这条不针对某个 bug，它直接把「引用定位正确率 1.0」这件事变成可断言的东西。
        """
        ret, expansions = _retriever(["doc/a#000", "doc/a#001"])
        ctx = ret.build_context([], expansions)
        checked = 0
        for unit in ctx["units"]:
            for span in unit["evidence_spans"]:
                assert unit["text"][span["start"]:span["end"]] == span["evidence"], (
                    unit["chunk_id"],
                    span,
                )
                checked += 1
        assert checked >= 1
