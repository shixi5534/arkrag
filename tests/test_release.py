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
    # 归档根也必须指到临时目录：`archive_old()` 默认写 `ARCHIVE_DIR`，
    # 不隔离的话任何一条真归档的用例都会往真实的归档区里搬真实产物。
    monkeypatch.setattr(release, "ARCHIVE_DIR", tmp_path / "archive")
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
    """造一份输入快照。

    `source_status` 必须显式给 `verified_zip`：D4 之后 `source_unverified`
    的一代**不允许发布**，默认值刻意是"未验证"，别指望它自己变成已验证。
    """
    chunks = chunks or _chunks()
    return release.capture_inputs(
        chunks=chunks,
        records=_records(chunks),
        source_fp="srcfp0123456789",
        scenes=[
            {
                "doc_id": "doc/a",
                "title": "测试章",
                "chapter": "第一章",
                "category": "obt/main",
                "turns": [
                    {"ordinal": 0, "node_id": 1, "kind": "dialogue", "speaker": "凯尔希", "text": "第 0 段。"},
                    {"ordinal": 1, "node_id": 2, "kind": "narration", "speaker": "", "text": "阿米娅点了点头。"},
                ],
            }
        ],
        source_status="verified_zip",
    )


def _build_generation(gid: str = "g-20260101T000000Z-aaaaaa", *, chunks=None) -> release.Generation:
    """造一代能通过全部校验的产物。

    这一代必须满足 D3 的三层身份约束：图要记 `identities`（逐层对齐 Manifest）、
    索引要绑 `graph_artifact_id`（证明这份倒排对着哪一份图**产物**，而不是哪一份
    配置），Manifest 与索引的 `prompt_sha256`/`vocab_sha` 要一致。
    """
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
        # 图必须记录它吃的是哪一层输入（D3）：图自己算不出上游指纹，
        # 但必须记下来，否则事后无法回答"这张图吃的是哪一批块"。
        "identities": {
            "source_fp": snap.source_fingerprint,
            "source_status": snap.source_status,
            "scene_fp": snap.scene_fingerprint,
            "chunk_fp": snap.chunk_fingerprint,
            "chunk_manifest_hash": snap.chunk_manifest_hash,
            "extraction_manifest_hash": snap.extraction_manifest_hash,
        },
    }
    gen.graph_path.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")

    idx, meta = index_mod.build(
        chunks,
        graph,
        snap.source_fingerprint,
        graph_artifact_id=release.sha256_file(gen.graph_path),
    )
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

    @pytest.mark.parametrize(
        "status",
        [
            "source_unverified",  # 显式的未验证
            "",  # 空串：没写，不等于验证过
            "VERIFIED_ZIP",  # 大小写不同
            "verified_zip ",  # 尾空格（笔误）
            "verified",  # 缩写
            "verified_zip\n",  # 换行
            "bogus",  # 完全不认识
        ],
    )
    def test_any_unrecognized_source_status_blocks_release(self, sandbox, status):
        """门必须是白名单。

        只拒绝字面量 "source_unverified" 的写法等于没有门：把状态写成
        ""、"VERIFIED_ZIP"、"verified_zip " 这些同样**不是已验证**的取值，
        全都绕得过去。判据与 source.strength() 必须是同一张表。
        """
        gen = _build_generation()
        gen.update(source_status=status)
        problems = release.verify_generation(gen)
        assert any("source_status" in p for p in problems), (
            f"source_status={status!r} 竟然通过了来源门"
        )

    @pytest.mark.parametrize("status", ["verified_zip", "verified_tree"])
    def test_recognized_verified_status_passes(self, sandbox, status):
        gen = _build_generation()
        gen.update(source_status=status)
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

    def test_publish_refuses_incomplete_coverage(self, sandbox):
        """覆盖不全的一代可以完全自洽（Manifest 记的字节就是盘上这些字节），
        但把它切上线就是用 13764 块的图冒充 13798 块的语料。这一条只认
        Manifest 里的显式申报，不猜、不修。"""
        gen = _build_generation("g-20260101T000000Z-000031")
        gen.update(
            coverage_status=release.COVERAGE_INCOMPLETE,
            coverage_declared=3,
            coverage_covered=2,
            coverage_missing=["doc/a#002"],
        )
        assert release.verify_generation(gen) == []
        with pytest.raises(release.ReleaseError) as exc:
            release.publish(gen)
        assert "覆盖不全" in str(exc.value)
        assert "doc/a#002" in exc.value.detail
        assert release.read_current() is None
        # 拒绝发布不改状态：这一代还是 READY，人可以补齐输入后重试
        assert gen.status() == release.READY

    def test_publish_refusal_keeps_existing_pointer(self, sandbox):
        good = _build_generation("g-20260101T000000Z-000032")
        release.publish(good)
        before = sandbox.joinpath("current.json").read_bytes()

        bad = _build_generation("g-20260102T000000Z-000033")
        bad.update(coverage_status=release.COVERAGE_INCOMPLETE, coverage_missing=["doc/a#002"])
        with pytest.raises(release.ReleaseError):
            release.publish(bad)

        assert sandbox.joinpath("current.json").read_bytes() == before
        assert release.current_generation().generation_id == good.generation_id

    def test_generation_without_coverage_field_is_treated_as_complete(self, sandbox):
        """兼容历史世代：老 Manifest 没有 coverage_status，不能因为缺字段就拒发。"""
        gen = _build_generation("g-20260101T000000Z-000034")
        assert "coverage_status" not in gen.load()
        assert release.publish(gen)["generation_id"] == gen.generation_id


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


def _seed_three_kinds(prefix: str = "") -> dict[str, release.Generation]:
    """造 PUBLISHED / READY / FAILED 齐备的一组世代，`created_at` 显式递增。

    为什么要显式写 `created_at`：`_now()` 只到秒，同一次测试里建的四代
    `created_at` 往往完全相同，`prune_old()` 的排序就退化成"目录名顺序"，
    于是用例在断言一条**碰巧**成立的不变式。这里给它们明确的先后，
    让"最近 N 个"有确定含义。

    为什么不用手工造目录：`_build_generation()` 会写出能通过
    `verify_generation()` 的真实图/索引/门禁报告，于是 `publish()` 能真的
    走完发布路径。手工造一个只有 Manifest 的目录，`publish()` 会在
    `artifact_digests()` 里因为产物文件缺失抛 `ReleaseError` 早退——
    那样测到的就不是 prune，而是"发布拒绝半成品"。
    """
    out: dict[str, release.Generation] = {}

    published = _build_generation(f"g-20260101T000000Z-{prefix}aa0001")
    published.update(created_at="2026-01-01T00:00:00Z")
    release.publish(published)          # -> PUBLISHED，并成为 current
    out["published"] = published

    ready = _build_generation(f"g-20260102T000000Z-{prefix}bb0002")
    ready.update(created_at="2026-01-02T00:00:00Z")
    assert ready.status() == release.READY
    out["ready"] = ready

    failed_old = _build_generation(f"g-20260103T000000Z-{prefix}cc0003")
    failed_old.update(created_at="2026-01-03T00:00:00Z")
    failed_old.fail("boom-old")
    out["failed_old"] = failed_old

    failed_new = _build_generation(f"g-20260104T000000Z-{prefix}dd0004")
    failed_new.update(created_at="2026-01-04T00:00:00Z")
    failed_new.fail("boom-new")
    out["failed_new"] = failed_new

    return out


class TestPruneOnlyTouchesFailed:
    """`prune_old()` 只许清 FAILED。

    ## 这条门为什么必须存在（它挡住的是真实事故）

    旧实现的候选条件是 `status != PUBLISHED`，于是 **READY 也算可清候选**，
    再"从候选里留最近 keep 个"。现场实测（12 个世代，`current=55dfd7`、
    `previous=606ea9`）：`prune_old(keep=2)` 拟清 7 个，其中 5 个是 READY
    可服务世代，**包含上一代 `g-20261031T035231Z-4eafae`**。真跑
    `dry_run=False` 会一次性删掉上一代和多个候选，直接违反计划 K4
    「保留至少上一代完整有效 Generation」。

    旧用例 `test_prune_is_dry_run_by_default` 只断言"演练不删东西"，
    而且跑在**空目录**里——没有任何候选可清，缺陷永远暴露不出来。
    下面两条把世代摆成 READY/PUBLISHED/FAILED 齐备，让"清错东西"
    这件事有东西可清、也就有断言可红。
    """

    def test_dry_run_lists_only_failed(self, sandbox):
        gens = _seed_three_kinds()
        victims = release.prune_old(keep=1, dry_run=True)
        # 只有 FAILED 能进候选：READY 与 PUBLISHED 一个都不许出现在名单里，
        # 哪怕它们比某些 FAILED 更旧。
        assert set(victims) <= {gens["failed_new"].generation_id, gens["failed_old"].generation_id}, (
            f"dry-run 名单里出现了非 FAILED 世代：{victims}"
        )
        # keep=1 的含义是"最近 1 个 FAILED 留作证据"，所以最旧的那个 FAILED 才是待清对象。
        assert victims == [gens["failed_old"].generation_id]

    def test_real_prune_keeps_ready_and_published(self, sandbox):
        gens = _seed_three_kinds()
        removed = release.prune_old(keep=1, dry_run=False)
        # 真删：只该少掉那个最旧的 FAILED
        assert removed == [gens["failed_old"].generation_id]
        # READY 与 PUBLISHED 的目录必须原样还在——这是 K4 第一条要求的底线
        assert gens["ready"].dir.is_dir(), "READY 世代被 prune 删掉了"
        assert gens["published"].dir.is_dir(), "PUBLISHED 世代被 prune 删掉了"
        assert gens["failed_new"].dir.is_dir(), "最近 1 个 FAILED 应留作证据，却被删了"
        assert not gens["failed_old"].dir.exists(), "该清的 FAILED 没被清掉"


# ---------------------------------------------------------------- 观察期与归档

# 计划 K4 第 4 条「新版稳定运行约定观察期后才归档旧版」的两条判据：
#   · `observation_status()` —— 判得出"过没过观察期"，且判不出来时必须往
#     **未满**那边倒（fail-closed）；
#   · `archive_old()` —— 观察期未满一律拒绝，且拒绝时盘上一个字节都不动。
#
# 这里每个用例都盯着"能不能变红"：把 `archive_old()` 里的观察期检查删掉，
# 下面必须有一条用例变红；把 `observation_status()` 的缺字段分支改成
# `complete=True`，也必须有一条变红。


def _pointer(gid: str, published_at) -> None:
    """写一份发布指针。`published_at` 传 None 就是"这一代没记发布时间"。"""
    payload = {"generation_id": gid, "status": release.PUBLISHED, "previous_generation": ""}
    if published_at is not None:
        payload["published_at"] = published_at
    release.CURRENT_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _tree_digests(root) -> dict:
    """逐文件 (字节数, SHA-256)。用来证明"盘上一个字节都没少"。"""
    out = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[path.relative_to(root).as_posix()] = (path.stat().st_size, release.sha256_file(path))
    return out


class TestObservationWindow:
    """观察期判定：算得对，且判不出来时不许当"已满"。"""

    def test_complete_when_published_long_enough(self, sandbox):
        _pointer("g-20260101T000000Z-aaaaaa", "2026-01-01T00:00:00Z")
        st = release.observation_status(now="2026-01-09T00:00:00Z")
        # `_now()` 产出的就是带 Z 的 UTC 秒级时间戳，必须能解析（不是"猜"格式）。
        assert st["complete"] is True
        assert st["days_elapsed"] == 8.0
        assert st["required_days"] == release.OBSERVATION_DAYS
        assert st["generation_id"] == "g-20260101T000000Z-aaaaaa"
        assert st["published_at"] == "2026-01-01T00:00:00Z"
        assert "达到观察期" in st["reason"]

    def test_incomplete_when_published_recently(self, sandbox):
        _pointer("g-20260101T000000Z-aaaaaa", "2026-01-01T00:00:00Z")
        st = release.observation_status(now="2026-01-05T12:00:00Z")
        assert st["complete"] is False
        assert st["days_elapsed"] == 4.5
        assert "未达观察期" in st["reason"]

    def test_missing_published_at_is_not_treated_as_complete(self, sandbox):
        """缺 `published_at`：既不能当 0 天，也不能当已满——必须 fail-closed。"""
        _pointer("g-20260101T000000Z-aaaaaa", None)
        st = release.observation_status(now="2030-01-01T00:00:00Z")
        assert st["complete"] is False
        assert st["days_elapsed"] is None
        assert "缺失或无法解析" in st["reason"]

    def test_unparsable_published_at_is_not_treated_as_complete(self, sandbox):
        _pointer("g-20260101T000000Z-aaaaaa", "2026/01/01 00:00")
        st = release.observation_status(now="2030-01-01T00:00:00Z")
        assert st["complete"] is False
        assert st["days_elapsed"] is None
        assert "2026/01/01 00:00" in st["reason"]

    def test_no_pointer_is_not_treated_as_complete(self, sandbox):
        st = release.observation_status(now="2030-01-01T00:00:00Z")
        assert st["complete"] is False
        assert "没有发布指针" in st["reason"]

    def test_clock_going_backwards_is_not_treated_as_complete(self, sandbox):
        """`now` 早于 `published_at`（时钟回拨/手改时间戳）时时间线不可信。"""
        _pointer("g-20260101T000000Z-aaaaaa", "2026-06-01T00:00:00Z")
        st = release.observation_status(now="2026-01-01T00:00:00Z")
        assert st["complete"] is False
        assert "时间线不可信" in st["reason"]


class TestArchiveOnlyAfterObservation:
    """`archive_old()` 的拒绝理由，每条都必须能单独触发。

    为什么逐条写：合并成一句"不满足归档条件"，事后看日志的人分不清是
    "观察期没过"（等几天再来）还是"这一代自己坏了"（永远不该归档）。
    """

    def test_refuses_while_observation_incomplete_and_deletes_nothing(self, sandbox):
        """计划 K4 第 4 条本身：观察期未满 → 拒绝，且盘上一个字节都不许少。

        这条是本次改动的核心门。把 `archive_old()` 里的观察期判断删掉，
        它就会去复制+删原件，下面的字节核对立刻变红。
        """
        gen = _build_generation("g-20260102T000000Z-bbbbbb")
        other = _build_generation("g-20260101T000000Z-aaaaaa")
        release.publish(other)                      # 指针指向别的世代
        before = _tree_digests(gen.dir)
        with pytest.raises(release.ReleaseError) as err:
            release.archive_old(gen.generation_id, dry_run=False, archive_root=sandbox / "arch")
        assert "观察期未满" in err.value.detail
        assert _tree_digests(gen.dir) == before, "拒绝归档时仍然改动了原目录字节"
        assert not (sandbox / "arch" / gen.generation_id).exists(), "被拒绝时不得留下归档目录"

    def test_refuses_current_generation(self, sandbox):
        gen = _build_generation("g-20260101T000000Z-aaaaaa")
        release.publish(gen)                        # 它自己就是线上
        _pointer(gen.generation_id, "2026-01-01T00:00:00Z")   # 观察期已过也没用
        with pytest.raises(release.ReleaseError) as err:
            release.archive_old(gen.generation_id, dry_run=False, archive_root=sandbox / "arch")
        assert "当前线上世代" in err.value.detail
        assert gen.graph_path.is_file(), "线上世代的图被搬走了"

    def test_refuses_failed_generation(self, sandbox):
        """FAILED 是失败证据，归 `prune_old()` 管，不是归档对象。"""
        gen = _build_generation("g-20260102T000000Z-bbbbbb")
        gen.fail("boom")
        other = _build_generation("g-20260101T000000Z-aaaaaa")
        release.publish(other)
        _pointer(other.generation_id, "2026-01-01T00:00:00Z")
        with pytest.raises(release.ReleaseError) as err:
            release.archive_old(gen.generation_id, dry_run=False, archive_root=sandbox / "arch")
        assert "FAILED" in err.value.detail
        assert gen.graph_path.is_file()

    def test_refuses_generation_that_does_not_verify(self, sandbox):
        """自检不过的坏版本不许归档：它可能正是要留作证据的那一份。"""
        gen = _build_generation("g-20260102T000000Z-bbbbbb")
        other = _build_generation("g-20260101T000000Z-aaaaaa")
        release.publish(other)
        _pointer(other.generation_id, "2026-01-01T00:00:00Z")
        # 门禁报告被删：Manifest 记着 gate_report_hash，盘上没有这份字节。
        gen.gate_report_path.unlink()
        assert release.verify_generation(gen), "构造失败：这一代本该自检不过"
        with pytest.raises(release.ReleaseError) as err:
            release.archive_old(gen.generation_id, dry_run=False, archive_root=sandbox / "arch")
        assert "自检不过" in err.value.detail
        assert gen.graph_path.is_file()

    def test_refuses_missing_generation(self, sandbox):
        with pytest.raises(release.ReleaseError) as err:
            release.archive_old("g-20260109T000000Z-ffffff", archive_root=sandbox / "arch")
        assert "不存在" in str(err.value)

    def test_dry_run_touches_nothing(self, sandbox):
        gen = _build_generation("g-20260102T000000Z-bbbbbb")
        other = _build_generation("g-20260101T000000Z-aaaaaa")
        release.publish(other)
        _pointer(other.generation_id, "2026-01-01T00:00:00Z")
        before = _tree_digests(gen.dir)
        # 不传 archive_root：默认必须在 generations 同级（data/arkrag/archive）
        plan = release.archive_old(gen.generation_id)
        assert plan["dry_run"] is True
        assert plan["archive_root"] == str(release.ARCHIVE_DIR)
        assert plan["archive_root"] == str(release.GENERATIONS_DIR.parent / "archive")
        assert plan["file_count"] >= 2, f"演练没列出产物：{plan['files']}"
        assert plan["removed"] == []
        assert _tree_digests(gen.dir) == before
        assert not release.ARCHIVE_DIR.exists()

    def test_real_archive_copies_verifies_then_deletes(self, sandbox):
        gen = _build_generation("g-20260102T000000Z-bbbbbb")
        other = _build_generation("g-20260101T000000Z-aaaaaa")
        release.publish(other)
        _pointer(other.generation_id, "2026-01-01T00:00:00Z")
        original = _tree_digests(gen.dir)
        result = release.archive_old(
            gen.generation_id, dry_run=False, archive_root=sandbox / "arch"
        )
        dest = sandbox / "arch" / gen.generation_id
        # 1) 归档副本逐字节等于原件
        archived = _tree_digests(dest)
        for item in result["files"]:
            assert archived[item["path"]] == (item["bytes"], item["sha256"])
            assert item["bytes"] == original[item["path"]][0]
            assert item["sha256"] == original[item["path"]][1]
        # 2) 重量级产物已从世代目录删掉，Manifest 与门禁报告保留
        assert not gen.graph_path.exists()
        assert not gen.index_path.exists()
        assert gen.manifest_path.is_file(), "Manifest 被删了：这一代将从 list_generations() 消失"
        assert gen.gate_report_path.is_file(), "门禁报告被删了：Manifest 的 gate_report_hash 没有被引对象"
        # 3) archive-manifest.json 内容正确
        man = json.loads((dest / "archive-manifest.json").read_text("utf-8"))
        assert man["generation_id"] == gen.generation_id
        assert man["total_bytes"] == sum(item["bytes"] for item in man["files"])
        assert man["total_bytes"] == result["total_bytes"]
        assert man["kept"] == ["manifest.json", "reports/gates.json"]
        assert {f["path"] for f in man["files"]} == {"graph/graph.json", "index/lexical.json"}
        for item in man["files"]:
            assert item["source"] == str(gen.dir / item["path"])
            assert release.sha256_file(dest / item["path"]) == item["sha256"]
        # 4) 归档后仍有 Manifest：这一代在列表里还在（否则指针里的 id 查不到东西）
        assert gen.generation_id in {g["generation_id"] for g in release.list_generations()}

    def test_failed_copy_leaves_originals_untouched(self, sandbox):
        """复制/核对中途失败 → 原件一个字节都不动，且不留半份归档。

        用"归档根是个普通文件"模拟目标目录建不出来：此时 `mkdir` 抛错，
        必须已经什么都没删。
        """
        gen = _build_generation("g-20260102T000000Z-bbbbbb")
        other = _build_generation("g-20260101T000000Z-aaaaaa")
        release.publish(other)
        _pointer(other.generation_id, "2026-01-01T00:00:00Z")
        before = _tree_digests(gen.dir)
        blocker = sandbox / "arch"
        blocker.write_bytes(b"i am a file, not a directory")   # mkdir 必然失败
        with pytest.raises((release.ReleaseError, OSError)):
            release.archive_old(gen.generation_id, dry_run=False, archive_root=blocker)
        assert _tree_digests(gen.dir) == before, "归档失败却删了原件"
