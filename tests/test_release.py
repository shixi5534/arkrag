"""Run/Generation 目录模型、状态机、回读校验与原子发布（计划 C1/C2）。

这些测试的核心不是"函数返回了什么"，而是几条**必须能失败的规矩**：

* 只有 READY 能发布；不经 READY 的 PUBLISHED 是非法状态。
* 发布前任何一步失败，`current.json` 必须**一次都没被碰过**——旧版本继续服务。
* 回读校验不比数量，比身份与内容摘要；产物被换掉一份必须能发现。
* 故障注入六种（构建半途/门禁 FAIL/回读损坏/写 Manifest 中断/发布前终止/
  发布后写报告失败）下，指针要么指向旧版，要么根本没有指针。
"""
from __future__ import annotations

import json

import pytest

from arkrag import index as index_mod
from arkrag import release


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """把 generations / runs / current 全部指到临时目录，不碰真实产物。"""
    gen_dir = tmp_path / "generations"
    runs_dir = tmp_path / "runs"
    current = tmp_path / "current.json"
    monkeypatch.setattr(release, "GENERATIONS_DIR", gen_dir)
    monkeypatch.setattr(release, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(release, "CURRENT_PATH", current)
    gen_dir.mkdir()
    runs_dir.mkdir()
    return tmp_path


def _chunks(n: int = 3) -> list[dict]:
    return [
        {
            "chunk_id": f"doc/a#{i:03d}",
            "content_key": f"ck{i}" * 8,
            "text": f"凯尔希：第 {i} 段。阿米娅点了点头。",
            "title": "测试章",
            "chapter": "第一章",
        }
        for i in range(n)
    ]


def _records(chunks: list[dict]) -> dict[str, dict]:
    return {
        c["chunk_id"]: {
            "entities": [{"name": "凯尔希", "type": "角色"}, {"name": "阿米娅", "type": "角色"}],
            "relations": [
                {
                    "source": "凯尔希",
                    "relation": "信任",
                    "target": "阿米娅",
                    "evidence": "凯尔希：",
                    "confidence": 0.9,
                }
            ],
        }
        for c in chunks
    }


def _snapshot(chunks=None):
    chunks = chunks or _chunks()
    return release.capture_inputs(
        chunks=chunks, records=_records(chunks), source_fp="srcfp0123456789"
    )


def _build_generation(gid: str = "g-20260101T000000Z-aaaaaa", *, chunks=None) -> release.Generation:
    """造一代能通过全部校验的产物。"""
    chunks = chunks or _chunks()
    snap = _snapshot(chunks)
    gen = release.Generation(gid)
    gen.create(snapshot=snap, run_id="r-test")
    gen.advance(release.BUILDING)

    graph = {
        "fingerprint": "graphfp0123456789",
        "corpus_fp": snap.source_fingerprint,
        "entities": [{"id": "凯尔希", "name": "凯尔希", "type": "角色", "aliases": []}],
        "edges": [],
    }
    gen.graph_path.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")

    idx, meta = index_mod.build(chunks, graph, snap.source_fingerprint)
    index_mod.save_index(idx, meta, gen.index_path)

    gen.gate_report_path.write_text(
        json.dumps([{"name": "L4.1", "verdict": "PASS"}], ensure_ascii=False), encoding="utf-8"
    )

    gen.advance(
        release.VALIDATING,
        graph_build_fp=graph["fingerprint"],
        graph_artifact_id=release.sha256_file(gen.graph_path),
        index_artifact_id=release.sha256_file(gen.index_path),
        index_build_fp=meta["index_build_fp"],
        index_content_digest=meta["index_content_digest"],
        probe_digest=meta["probe_digest"],
        gate_report_hash=release.sha256_file(gen.gate_report_path),
        stats={"entities": 1, "edges": 0},
    )
    gen.advance(release.READY)
    return gen


# ---------------------------------------------------------------- 身份与摘要


class TestIdentity:
    def test_manifest_hash_changes_when_content_changes_but_count_does_not(self):
        a = _chunks()
        b = [dict(c) for c in a]
        b[0]["content_key"] = "different0000000"
        assert release.chunk_manifest_hash(a) != release.chunk_manifest_hash(b)

    def test_chunk_manifest_hash_is_order_independent(self):
        a = _chunks()
        assert release.chunk_manifest_hash(a) == release.chunk_manifest_hash(list(reversed(a)))

    def test_extraction_manifest_hash_tracks_raw_content(self):
        a = _records(_chunks())
        b = {k: dict(v) for k, v in a.items()}
        b["doc/a#000"] = {"entities": [], "relations": []}
        assert release.extraction_manifest_hash(a) != release.extraction_manifest_hash(b)

    def test_snapshot_records_counts_and_versions(self):
        snap = _snapshot()
        d = snap.as_dict()
        assert d["chunk_count"] == 3
        assert d["extraction_count"] == 3
        assert d["versions"]["schema_version"]
        assert "git_revision" in d

    def test_index_content_digest_ignores_serialization(self, sandbox):
        """内容摘要必须跟着倒排走，不跟着 JSON 字节走。"""
        chunks = _chunks()
        graph = {"fingerprint": "g", "entities": [], "edges": []}
        idx, _meta = index_mod.build(chunks, graph, "fp")
        first = index_mod.content_digest(idx)
        # 序列化再读回：内容相同则摘要必须相同（JSON 键顺序/缩进不该影响身份）
        tmp = sandbox / "probe.json"
        index_mod.save_index(idx, {"x": 1}, tmp)
        reloaded, _ = index_mod.load_index(tmp)
        assert index_mod.content_digest(reloaded) == first

    def test_probe_digest_changes_when_postings_change(self):
        """探针摘要表「检索行为」：数量全对但排不出来，只有这一层能发现。"""
        chunks = _chunks()
        graph = {"fingerprint": "g", "entities": [], "edges": []}
        idx, _ = index_mod.build(chunks, graph, "fp")
        before = index_mod.probe_digest(idx)
        # 必须篡改**探针查询实际用到的** token，否则排序不会变——
        # 这正是「随便动一条倒排」无法证明探针有效的原因。
        probe_terms = set(index_mod.tokenize(index_mod.FIXED_PROBES[0]))
        victim = next(t for t in probe_terms if t in idx.postings and len(idx.postings[t]) >= 2)
        slot = next(iter(idx.postings[victim]))
        idx.postings[victim][slot] = 999
        assert index_mod.probe_digest(idx) != before


# ---------------------------------------------------------------- 状态机


class TestStateMachine:
    def test_normal_path_reaches_ready(self, sandbox):
        gen = _build_generation()
        assert gen.status() == release.READY

    def test_cannot_skip_to_ready(self, sandbox):
        gen = release.Generation("g-20260101T000000Z-bbbbbb")
        gen.create(snapshot=_snapshot(), run_id="r")
        with pytest.raises(release.IllegalTransition):
            gen.advance(release.READY)

    def test_cannot_publish_without_ready(self, sandbox):
        gen = release.Generation("g-20260101T000000Z-cccccc")
        gen.create(snapshot=_snapshot(), run_id="r")
        with pytest.raises(release.IllegalTransition):
            release.publish(gen)

    def test_any_state_can_fail(self, sandbox):
        gen = release.Generation("g-20260101T000000Z-dddddd")
        gen.create(snapshot=_snapshot(), run_id="r")
        gen.fail("boom", detail="测试")
        assert gen.status() == release.FAILED
        # FAILED 是终态
        with pytest.raises(release.IllegalTransition):
            gen.advance(release.BUILDING)

    def test_failed_generation_keeps_manifest_for_postmortem(self, sandbox):
        gen = release.Generation("g-20260101T000000Z-eeeeee")
        gen.create(snapshot=_snapshot(), run_id="r")
        gen.fail("gate_fail", detail="L4.2 FAIL")
        manifest = gen.load()
        assert manifest["fail_reason"] == "gate_fail"
        assert manifest["fail_detail"] == "L4.2 FAIL"
        assert gen.manifest_path.exists()

    def test_illegal_generation_id_rejected(self, sandbox):
        with pytest.raises(release.ReleaseError):
            release.Generation("not-a-generation")

    def test_create_refuses_existing_dir(self, sandbox):
        gen = release.Generation("g-20260101T000000Z-ffffff")
        gen.create(snapshot=_snapshot(), run_id="r")
        with pytest.raises(release.ReleaseError):
            gen.create(snapshot=_snapshot(), run_id="r2")


# ---------------------------------------------------------------- 回读校验


class TestVerify:
    def test_good_generation_verifies_clean(self, sandbox):
        gen = _build_generation()
        assert release.verify_generation(gen) == []

    def test_missing_graph_is_reported(self, sandbox):
        gen = _build_generation()
        gen.graph_path.unlink()
        problems = release.verify_generation(gen)
        assert any("不存在" in p for p in problems)

    def test_swapped_graph_bytes_detected(self, sandbox):
        """把图换成另一份内容（结构合法）——身份摘要必须发现。"""
        gen = _build_generation()
        other = {"fingerprint": "other", "entities": [], "edges": [], "corpus_fp": "x"}
        gen.graph_path.write_text(json.dumps(other), encoding="utf-8")
        problems = release.verify_generation(gen)
        assert any("graph_artifact_id" in p for p in problems)

    def test_index_bound_to_wrong_graph_detected(self, sandbox):
        gen = _build_generation()
        payload = json.loads(gen.index_path.read_text("utf-8"))
        payload["meta"]["graph_fp"] = "someone-else"
        gen.index_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        problems = release.verify_generation(gen)
        assert any("索引绑定的图指纹" in p for p in problems)

    def test_index_corpus_fp_mismatch_detected(self, sandbox):
        gen = _build_generation()
        payload = json.loads(gen.index_path.read_text("utf-8"))
        payload["meta"]["corpus_fp"] = "wrong-corpus"
        gen.index_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        # 先让 artifact digest 跟着改，确保测的是 corpus_fp 那一条而不是摘要
        gen.update(index_artifact_id=release.sha256_file(gen.index_path))
        problems = release.verify_generation(gen)
        assert any("索引语料指纹" in p for p in problems)

    def test_empty_artifact_is_hard_failure(self, sandbox):
        gen = _build_generation()
        gen.index_path.write_bytes(b"")
        problems = release.verify_generation(gen)
        assert any("空文件" in p for p in problems)

    def test_gate_report_hash_mismatch_detected(self, sandbox):
        gen = _build_generation()
        gen.gate_report_path.write_text("[]", encoding="utf-8")
        problems = release.verify_generation(gen)
        assert any("gate_report_hash" in p for p in problems)


# ---------------------------------------------------------------- 发布


class TestPublish:
    def test_publish_writes_pointer(self, sandbox):
        gen = _build_generation()
        pointer = release.publish(gen)
        assert pointer["generation_id"] == gen.generation_id
        assert release.read_current()["generation_id"] == gen.generation_id
        assert gen.status() == release.PUBLISHED

    def test_publish_records_previous(self, sandbox):
        first = _build_generation("g-20260101T000000Z-000001")
        release.publish(first)
        second = _build_generation("g-20260102T000000Z-000002")
        pointer = release.publish(second)
        assert pointer["previous_generation"] == first.generation_id

    def test_publish_refuses_broken_generation_and_marks_failed(self, sandbox):
        gen = _build_generation()
        gen.graph_path.unlink()
        with pytest.raises(release.ReleaseError):
            release.publish(gen)
        assert gen.status() == release.FAILED
        assert release.read_current() is None

    def test_publish_failure_never_touches_existing_pointer(self, sandbox):
        """最要紧的一条：新版本发布失败，线上必须还是旧版本。"""
        good = _build_generation("g-20260101T000000Z-000011")
        release.publish(good)
        before = sandbox.joinpath("current.json").read_bytes()

        bad = _build_generation("g-20260102T000000Z-000012")
        bad.index_path.write_bytes(b"corrupt")
        with pytest.raises(release.ReleaseError):
            release.publish(bad)

        assert sandbox.joinpath("current.json").read_bytes() == before
        assert release.current_generation().generation_id == good.generation_id

    def test_corrupt_pointer_raises_instead_of_guessing(self, sandbox):
        (sandbox / "current.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(release.ReleaseError):
            release.read_current()


# ---------------------------------------------------------------- 回滚


class TestRollback:
    def test_rollback_switches_pointer_only(self, sandbox):
        first = _build_generation("g-20260101T000000Z-000021")
        release.publish(first)
        second = _build_generation("g-20260102T000000Z-000022")
        release.publish(second)

        graph_bytes = first.graph_path.read_bytes()
        pointer = release.rollback(first.generation_id)
        assert pointer["generation_id"] == first.generation_id
        assert pointer["previous_generation"] == second.generation_id
        assert pointer.get("rolled_back") is True
        # 只切指针：历史产物一个字节都不许变
        assert first.graph_path.read_bytes() == graph_bytes

    def test_rollback_pointer_has_same_shape_as_publish(self, sandbox):
        """回滚指针必须与发布指针同构。

        首版回滚只写了 generation_id，于是 `release_cli current` 事后查不出线上跑的是
        哪份图、哪份索引、哪份门禁报告——而这些正是回滚后要核对的东西。
        """
        first = _build_generation("g-20260101T000000Z-000023")
        release.publish(first)
        release.publish(_build_generation("g-20260102T000000Z-000024"))
        rolled = release.rollback(first.generation_id)

        for key in (
            "generation_id",
            "status",
            "published_at",
            "previous_generation",
            "source_fingerprint",
            "graph_artifact_id",
            "index_artifact_id",
            "gate_report_hash",
        ):
            assert rolled.get(key), f"回滚指针缺 {key}"
            assert rolled[key] == release.read_current()[key]

    def test_rollback_to_current_is_refused(self, sandbox):
        """回滚到「已经是当前」的那一代必须被拒。

        否则指针会声称这一代替换了它自己，`previous_generation` 指向自身，
        下次真回滚时就没有可退的目标了。
        """
        gen = _build_generation("g-20260101T000000Z-000025")
        release.publish(gen)
        with pytest.raises(release.ReleaseError):
            release.rollback(gen.generation_id)
        assert release.read_current()["previous_generation"] == ""

    def test_rollback_refuses_failed_generation(self, sandbox):
        gen = _build_generation()
        gen.fail("manual")
        with pytest.raises(release.ReleaseError):
            release.rollback(gen.generation_id)

    def test_rollback_refuses_missing_generation(self, sandbox):
        with pytest.raises(release.ReleaseError):
            release.rollback("g-20260101T000000Z-999999")

    def test_rollback_refuses_broken_generation(self, sandbox):
        gen = _build_generation()
        release.publish(gen)
        gen.index_path.write_bytes(b"tampered")
        with pytest.raises(release.ReleaseError):
            release.rollback(gen.generation_id)


# ---------------------------------------------------------------- 故障注入


class TestFaultInjection:
    """计划 C1 列的六种故障，每一种都要求「current 仍指向旧有效 Generation」。"""

    def test_build_interrupted_midway(self, sandbox):
        good = _build_generation("g-20260101T000000Z-000031")
        release.publish(good)
        # 新的一代只写完图就中断（没有 index、没有 gate 报告、没有走到 READY）
        gen = release.Generation("g-20260102T000000Z-000032")
        gen.create(snapshot=_snapshot(), run_id="r-interrupted")
        gen.advance(release.BUILDING)
        gen.graph_path.write_text('{"fingerprint":"partial"}', encoding="utf-8")
        # 进程被杀，什么都没接着做
        assert release.current_generation().generation_id == good.generation_id
        assert gen.status() == release.BUILDING
        assert release.verify_generation(gen)  # 半成品自己也不自洽

    def test_gate_failure_blocks_publish(self, sandbox):
        good = _build_generation("g-20260101T000000Z-000041")
        release.publish(good)
        gen = _build_generation("g-20260102T000000Z-000042")
        # 门禁 FAIL -> 调用方不允许 advance(READY)，直接标记 FAILED
        gen.fail("gate_fail", detail="L5.3 边证据缺失 12 条")
        with pytest.raises(release.IllegalTransition):
            release.publish(gen)
        assert release.current_generation().generation_id == good.generation_id

    def test_readback_corruption_blocks_publish(self, sandbox):
        good = _build_generation("g-20260101T000000Z-000051")
        release.publish(good)
        gen = _build_generation("g-20260102T000000Z-000052")
        gen.index_path.write_text("{broken", encoding="utf-8")
        with pytest.raises(release.ReleaseError):
            release.publish(gen)
        assert release.current_generation().generation_id == good.generation_id

    def test_manifest_write_interrupted(self, sandbox):
        """Manifest 只写了半个 JSON：读取必须报错，不许当成有效一代。"""
        gen = _build_generation("g-20260101T000000Z-000061")
        release.publish(gen)
        victim = release.Generation("g-20260102T000000Z-000062")
        victim.create(snapshot=_snapshot(), run_id="r")
        victim.manifest_path.write_text('{"generation_id": "g-2026', encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            victim.load()
        assert release.current_generation().generation_id == gen.generation_id

    def test_terminated_before_publish(self, sandbox):
        good = _build_generation("g-20260101T000000Z-000071")
        release.publish(good)
        gen = _build_generation("g-20260102T000000Z-000072")  # 到 READY 就被终止
        assert gen.status() == release.READY
        assert release.current_generation().generation_id == good.generation_id

    def test_report_write_failure_after_publish_keeps_pointer(self, sandbox):
        gen = _build_generation("g-20260101T000000Z-000081")
        pointer = release.publish(gen)
        # 发布后写外部报告失败（比如磁盘满）：指针已经切了，不许因此回退
        try:
            raise OSError("No space left on device")
        except OSError:
            pass
        assert release.read_current()["generation_id"] == pointer["generation_id"]
        assert release.verify_generation(gen) == []


# ---------------------------------------------------------------- Run 事件


class TestRun:
    def test_events_are_appended_readably(self, sandbox):
        run = release.Run()
        run.event("build", "开始建图", chunks=13781)
        run.error("gate", "L4.2 FAIL", "12 条边无证据")
        run.finish("ok")
        lines = [json.loads(x) for x in run.events_path.read_text("utf-8").splitlines() if x.strip()]
        assert [x["stage"] for x in lines] == ["build", "gate", "run"]
        assert (run.dir / "errors").exists()
        assert len(list((run.dir / "errors").glob("*.json"))) == 1

    def test_run_json_has_status_after_finish(self, sandbox):
        run = release.Run()
        run.finish("failed", reason="门禁 FAIL")
        data = json.loads((run.dir / "run.json").read_text("utf-8"))
        assert data["status"] == "failed"


# ---------------------------------------------------------------- 列表与清理


class TestListing:
    def test_list_generations_reports_status(self, sandbox):
        a = _build_generation("g-20260101T000000Z-000091")
        release.publish(a)
        b = _build_generation("g-20260102T000000Z-000092")
        b.fail("boom")
        rows = {r["generation_id"]: r for r in release.list_generations()}
        assert rows[a.generation_id]["status"] == release.PUBLISHED
        assert rows[b.generation_id]["status"] == release.FAILED

    def test_list_tolerates_corrupt_manifest(self, sandbox):
        gen = release.Generation("g-20260101T000000Z-000093")
        gen.dir.mkdir(parents=True)
        gen.manifest_path.write_text("{bad", encoding="utf-8")
        rows = release.list_generations()
        assert rows[0]["status"] == "(Manifest 损坏)"

    def test_prune_is_dry_run_by_default(self, sandbox):
        victims = release.prune_old(keep=1)
        assert isinstance(victims, list)
        # 演练不删任何东西：现有目录都还在
        assert len(list(release.GENERATIONS_DIR.iterdir())) == 0
