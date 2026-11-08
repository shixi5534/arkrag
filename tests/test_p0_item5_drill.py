"""P0 第 5 项（隔离副本审批/发布/回滚/归档）的门。

这些用例盯的不是「`scripts/p0_item5_drill.py` 跑完了」，而是**几条必须能失败的规矩**：

* 报告里那句「真实归档路径本轮未走通，原因是观察期 X 天 < 7 天，不是实现问题」
  缺一个字就必须判红——没有这句话，下一代人会把「没走过」读成「验过了」。
* `real_archive_exercised` 只能 `false`。线上观察期没满，写 `true` 就是把没走过的
  路径记成走过。
* `unknown` 不算通过：七字段缺一项，整份报告就是不合格，不是「部分合格」。
* `dry_run=True` 之后，世代的每个字节与每个目录 mtime 都不许动；`dry_run=False`
  的顺序是「复制 → 逐文件核对 SHA-256 → 写归档清单 → 才删源」，删完归档副本
  **还原回去仍要能过 `verify_generation()`**。

反向实测（命令级，见卫生门 N 的注释）：删掉 `archive_old()` 里的观察期检查 →
`test_dry_run_refuses_while_observation_incomplete` 变红；把
`archive_old()` 改成「先删源再复制」→ `test_real_archive_copies_verifies_then_deletes`
变红；把 `check_report()` 里那句说明的判定删掉 →
`test_report_without_the_fixed_sentence_is_rejected` 变红。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from arkrag import index as index_mod
from arkrag import provenance, release

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    """按路径加载脚本模块（`scripts/` 不是包，不能 import）。"""
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DRILL = _load("p0_item5_drill_under_test", "scripts/p0_item5_drill.py")


# ---------------------------------------------------------------- 造数（照 test_release 的骨架）


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


def _snapshot(chunks: list[dict]):
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
                    {"ordinal": 0, "node_id": 1, "kind": "dialogue", "speaker": "凯尔希",
                     "text": "第 0 段。"},
                    {"ordinal": 1, "node_id": 2, "kind": "narration", "speaker": "",
                     "text": "阿米娅点了点头。"},
                ],
            }
        ],
        source_status="verified_zip",
    )


def _build_generation(gid: str, chunks: list[dict]) -> release.Generation:
    """造一代能通过 `verify_generation()` 的产物（三层身份 + 索引绑图产物）。"""
    snap = _snapshot(chunks)
    gen = release.Generation(gid)
    gen.create(snapshot=snap, run_id="r-test")
    gen.advance(release.BUILDING)

    graph = {
        "fingerprint": "graphfp0123456789",
        "corpus_fp": snap.source_fingerprint,
        "entities": [{"id": "凯尔希", "name": "凯尔希", "type": "角色", "aliases": []}],
        "edges": [],
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
        chunks, graph, snap.source_fingerprint,
        graph_artifact_id=release.sha256_file(gen.graph_path),
    )
    index_mod.save_index(idx, meta, gen.index_path)

    gen.gate_report_path.write_text(
        json.dumps([{"name": "L4.1", "verdict": "PASS"}], ensure_ascii=False),
        encoding="utf-8",
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


@pytest.fixture()
def box(tmp_path, monkeypatch):
    """把 release 的四个路径 + 归档根全部指到临时目录。"""
    gen_dir = tmp_path / "generations"
    runs_dir = tmp_path / "runs"
    monkeypatch.setattr(release, "GENERATIONS_DIR", gen_dir)
    monkeypatch.setattr(release, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(release, "CURRENT_PATH", tmp_path / "current.json")
    monkeypatch.setattr(release, "ARCHIVE_DIR", tmp_path / "archive")
    gen_dir.mkdir()
    runs_dir.mkdir()
    return tmp_path


def _pointer(path: Path, gid: str, published_at: str) -> None:
    payload = {
        "generation_id": gid,
        "status": release.PUBLISHED,
        "published_at": published_at,
        "previous_generation": "",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


# 观察期判定读的是**指针里的** `published_at`，而且拿真实当前时间算差值。
# 所以这两个日期必须是相对真实时间写好的一对，写死「未来」会让差值变负，
# `observation_status()` 会按「时间线不可信」判未满——那时用例红的原因就变成
# 时钟问题，而不是它想验的观察期规则。
_RECENT = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
_LONG_AGO = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
_NOT_TARGET = "g-20260102T000000Z-0a11ce"  # 指针指向「另一代」，否则会被「当前线上」先挡掉


# ---------------------------------------------------------------- 报告判据（纯函数）


def _good_report() -> dict:
    return {
        "schema": DRILL.SCHEMA,
        **provenance.collect(
            generation_id="g-20260101T000000Z-aaaaaa",
            inputs=provenance.inputs_block(["data/goldset/gold-v2.json"]),
        ),
        "real_archive_exercised": False,
        "real_archive_note": (
            "真实归档路径本轮未走通，原因是观察期 5.99 天 < 7 天，不是实现问题。"
            "本报告里的归档结论全部来自隔离副本上的同一份真实字节。"
        ),
        "checks": [{"id": "x", "ok": True, "detail": "d", "data": {}}],
    }


def test_a_complete_report_passes():
    assert DRILL.check_report(_good_report()) == []


def test_report_without_the_fixed_sentence_is_rejected():
    """那句「不是实现问题」是这份报告唯一能防住「演练通过」的措辞。"""
    report = _good_report()
    report["real_archive_note"] = "归档演练通过，全流程验证完成。"
    problems = DRILL.check_report(report)
    assert any("缺固定说明" in p for p in problems)


def test_the_fixed_sentence_tolerates_a_different_day_count():
    """天数每轮都在变，钉死 5.98 就会让下一天的报告变成假红。"""
    report = _good_report()
    report["real_archive_note"] = (
        "真实归档路径本轮未走通，原因是观察期 6.4 天 < 7 天，不是实现问题。"
    )
    assert not any("缺固定说明" in p for p in DRILL.check_report(report))


def test_claiming_the_real_archive_was_exercised_is_rejected():
    report = _good_report()
    report["real_archive_exercised"] = True
    problems = DRILL.check_report(report)
    assert any("real_archive_exercised 必须是 false" in p for p in problems)


def test_missing_provenance_field_makes_the_report_unknown_not_pass():
    """七字段缺一项 → 整份不合格。`unknown` 不许当通过。"""
    report = _good_report()
    report.pop("dataset")
    problems = DRILL.check_report(report)
    assert any("七字段未全部成立" in p for p in problems)


def test_a_failed_check_is_named_not_summarised():
    report = _good_report()
    report["checks"] = [
        {"id": "ok_one", "ok": True, "detail": "fine", "data": {}},
        {"id": "bad_one", "ok": False, "detail": "哈希不一致", "data": {}},
    ]
    problems = DRILL.check_report(report)
    assert len(problems) == 1 and "bad_one" in problems[0] and "哈希不一致" in problems[0]


def test_an_empty_check_list_cannot_pass():
    report = _good_report()
    report["checks"] = []
    assert any("checks 为空" in p for p in DRILL.check_report(report))


def test_machine_absolute_path_is_rejected():
    report = _good_report()
    report["sandbox"] = str(ROOT)
    assert any("绝对根路径" in p for p in DRILL.check_report(report))


# ---------------------------------------------------------------- 摘要差（纯函数）


def test_drift_names_missing_extra_and_changed_separately():
    """三种差必须分开列，「有变化」这句话没法排查。"""
    drift = DRILL.describe_drift(
        {"a": "1", "b": "2", "c": "3"},
        {"b": "9", "c": "3", "d": "4"},
    )
    assert drift["missing"] == ["a"]
    assert drift["extra"] == ["d"]
    assert drift["changed"] == ["b"]
    assert drift["identical"] is False


def test_drift_of_identical_trees_is_identical():
    same = {"a": "1", "b": "2"}
    assert DRILL.describe_drift(same, dict(same))["identical"] is True


def test_digest_tree_reads_real_bytes(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_bytes(b"alpha")
    (tmp_path / "sub" / "b.txt").write_bytes(b"beta")
    digests = DRILL.digest_tree(tmp_path)
    assert set(digests) == {"a.txt", "sub/b.txt"}
    assert digests["a.txt"] == release.sha256_bytes(b"alpha")


# ---------------------------------------------------------------- 归档两半（真跑）


def test_dry_run_refuses_while_observation_incomplete(box):
    """观察期没满就必须拒，且拒的时候盘上一个字节都不动。"""
    gid = "g-20260101T000000Z-aaaaaa"
    _build_generation(gid, _chunks())
    # 指针指向另一代、且是 1 天前发布：观察期 1 天 < 7 天。
    _pointer(release.CURRENT_PATH, _NOT_TARGET, _RECENT)
    before = DRILL.digest_tree(box)
    with pytest.raises(release.ReleaseError) as exc:
        release.archive_old(gid, dry_run=True)
    assert "观察期" in (str(exc.value) + exc.value.detail)
    assert DRILL.digest_tree(box) == before


def test_dry_run_touches_nothing_and_plans_every_heavy_artifact(box):
    """观察期已满时演练：给计划、建不了归档目录、字节不变。"""
    gid = "g-20260101T000000Z-aaaaaa"
    _build_generation(gid, _chunks())
    _pointer(release.CURRENT_PATH, _NOT_TARGET, _LONG_AGO)
    before = DRILL.digest_tree(box)
    plan = release.archive_old(gid, dry_run=True)
    assert plan["file_count"] >= 2
    assert not release.ARCHIVE_DIR.exists()
    assert DRILL.digest_tree(box) == before
    assert set(plan["kept"]) == {"manifest.json", "reports/gates.json"}


def test_real_archive_copies_verifies_then_deletes(box):
    """真归档：副本 SHA-256 全等 → 源只剩非产物 → **还原后仍能过自检**。"""
    gid = "g-20260101T000000Z-aaaaaa"
    gen = _build_generation(gid, _chunks())
    _pointer(release.CURRENT_PATH, _NOT_TARGET, _LONG_AGO)

    kept_dir = box / "kept" / gid
    for rel in ("manifest.json", "reports/gates.json"):
        (kept_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy2(gen.dir / rel, kept_dir / rel)

    res = release.archive_old(gid, dry_run=False)
    archive_dir = Path(res["archive_dir"])

    # ① 复制后源与归档副本逐文件 SHA-256 全等。
    for item in res["files"]:
        dst = archive_dir / item["path"]
        assert dst.is_file()
        assert release.sha256_file(dst) == item["sha256"], item["path"]

    # ② 核对通过才删源：删完源里只剩 `manifest.json` / `reports/gates.json`。
    #    实测过的真实世代里还会留下一个 0 字节的 `gates.json.lock`（写盘锁）——
    #    **它在不在取决于上一次写盘有没有留下锁**，所以判据是「除锁文件外一个产物都不剩」，
    #    而不是把锁文件也写死进期望值（那会让本用例在另一类世代上假红）。
    remaining_all = sorted(
        p.relative_to(gen.dir).as_posix() for p in gen.dir.rglob("*") if p.is_file()
    )
    remaining = [r for r in remaining_all if not r.endswith(".lock")]
    assert remaining == ["manifest.json", "reports/gates.json"]
    assert release._heavy_artifacts(release.Generation(gid)) == []

    # ③ 删源后归档副本还原成世代形状，仍要能过 verify_generation()。
    restore_root = box / "restore" / "generations"
    restore_root.mkdir(parents=True, exist_ok=True)
    DRILL.reassemble(archive_dir, kept_dir, restore_root)
    problems = release.verify_generation(release.Generation(gid, root=restore_root))
    assert problems == []


def test_reassemble_needs_the_kept_files(box):
    """还原必须带上源里保留的 `manifest.json`：少了它就一定过不了自检。

    这一条盯的是「归档可逆」这句话——如果只搬走产物、没有保留清单，
    还原出来的东西就是一包没有身份的字节。
    """
    gid = "g-20260101T000000Z-aaaaaa"
    _build_generation(gid, _chunks())
    _pointer(release.CURRENT_PATH, _NOT_TARGET, _LONG_AGO)
    res = release.archive_old(gid, dry_run=False)

    restore_root = box / "restore" / "generations"
    restore_root.mkdir(parents=True, exist_ok=True)
    empty_kept = box / "kept-empty" / gid
    empty_kept.mkdir(parents=True, exist_ok=True)
    DRILL.reassemble(Path(res["archive_dir"]), empty_kept, restore_root)

    with pytest.raises(release.ReleaseError):
        release.verify_generation(release.Generation(gid, root=restore_root))
