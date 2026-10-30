"""事务式构建脚本的端到端行为（计划 C1 / C2 的验收测试）。

这些测试跑的是 `scripts/build_generation.py` 本身，不是它的某个函数——
因为要证明的正是**顺序**：快照 -> 候选 -> 门禁 -> 回读 -> READY -> 发布。
函数级测试证明不了"门禁 FAIL 时指针没被动过"。

依赖全部用替身：`extract.load_chunks`、`graph.load_records` 喂人造数据，
`release.*` 路径指到临时目录。这样测试不读 40 MB 的真实产物，也不会写正式路径。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from arkrag import gates, release

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def script(tmp_path, monkeypatch):
    """加载 build_generation 脚本模块，并把所有写盘路径指到临时目录。"""
    spec = importlib.util.spec_from_file_location(
        "build_generation_under_test", ROOT / "scripts" / "build_generation.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    monkeypatch.setattr(release, "GENERATIONS_DIR", tmp_path / "generations")
    monkeypatch.setattr(release, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(release, "CURRENT_PATH", tmp_path / "current.json")
    (tmp_path / "generations").mkdir()
    (tmp_path / "runs").mkdir()
    # 交给 _wire 用，保证 config.DATA_DIR 也被隔离到同一个临时根
    mod.TEST_ROOT = tmp_path
    return mod


CHUNKS = [
    {
        "chunk_id": f"doc/a#{i:03d}",
        "content_key": f"ck{i:04d}",
        "text": "凯尔希：切尔诺伯格已经沦陷了。\n阿米娅：我们必须去罗德岛。",
        "title": "切尔诺伯格",
        "chapter": "第一章",
        "category": "obt/main",
        "scene_id": f"doc/a#{i}",
    }
    for i in range(4)
]

RECORDS = {
    c["chunk_id"]: {
        "entities": [
            {"name": "凯尔希", "type": "角色"},
            {"name": "阿米娅", "type": "角色"},
            {"name": "罗德岛", "type": "组织"},
        ],
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
    for c in CHUNKS
}


def _wire(mod, monkeypatch, tmp_path, *, gate_fails: bool = False):
    """把人造数据接到脚本的输入口上。

    门禁本身在这里换成替身：**这些测试的对象是发布顺序**（快照 -> 候选 -> 门禁
    -> 回读 -> READY -> 指针），不是 L4/L5/L6 的判据。真实判据要求平均度数 > 6、
    孤立实体率 <= 0.20 这类全量语料才有的规模，四块人造数据永远过不了——
    用假数据去满足它们只能把门放松，那正好是计划书第 5 条禁止的事。
    门禁语义由真实语料上的 run_l5/run_l6 与 A 阶段 75 行快照负责。
    """
    # DATA_DIR 必须一起指走：脚本从 `config.DATA_DIR / "corpus_meta.json"` 读语料指纹，
    # 不隔离就会把测试用的假指纹写进真实数据目录——这正是本测试首版干过的事，
    # 靠 A 阶段基线的 --verify 才发现（corpus_meta.json 字节已变）。
    data_dir = (tmp_path / "data").resolve()
    assert str(data_dir).startswith(str(tmp_path.resolve())), "DATA_DIR 必须隔离到临时目录"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(mod.config, "DATA_DIR", data_dir)
    (data_dir / "corpus_meta.json").write_text(
        json.dumps({"corpus_fp": "corpusfp01234567"}), encoding="utf-8"
    )

    monkeypatch.setattr(mod.extract, "load_chunks", lambda *a, **k: [dict(c) for c in CHUNKS])
    monkeypatch.setattr(mod.cache, "is_cached_for_chunk", lambda *a, **k: True)
    monkeypatch.setattr(
        mod.graph,
        "load_records",
        lambda chunks: [{"chunk_id": c["chunk_id"], "raw": RECORDS[c["chunk_id"]]} for c in chunks],
    )

    def _passing(stage: str, n: int):
        def factory(*_a, **_k):
            run = gates.GateRun()
            for i in range(n):
                run.add(
                    gates.Gate(f"{stage}.{i}", stage, "替身门", "1", "PASS", "发布顺序测试的替身")
                )
            if gate_fails:
                run.add(
                    gates.Gate(f"{stage}.99 注入的失败门", stage, "必须 0", "实测 3", "FAIL", "测试注入")
                )
            return run

        return factory

    monkeypatch.setattr(mod.gates, "l4_gates", _passing("L4", 2))
    monkeypatch.setattr(mod.gates, "l5_gates", _passing("L5", 3))
    monkeypatch.setattr(mod.gates, "l6_gates", _passing("L6", 2))


def _argv(**kw):
    import argparse

    defaults = {"probes": 5, "no_publish": False}
    defaults.update(kw)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------- 隔离守卫


class TestIsolation:
    def test_wire_points_data_dir_into_tmp(self, script, monkeypatch):
        """回归守卫：首版测试把假 corpus_fp 写进了真实 `data/arkrag/corpus_meta.json`，
        靠 A 阶段基线的 `--verify` 才发现。现在把「DATA_DIR 必须落在临时根下」
        变成断言——测试环境一旦漏了隔离，先红的是这条。"""
        from arkrag import config

        _wire(script, monkeypatch, script.TEST_ROOT)
        assert str(config.DATA_DIR).startswith(str(script.TEST_ROOT))
        assert Path(config.DATA_DIR) != (ROOT / "data" / "arkrag")

    def test_build_does_not_touch_real_corpus_meta(self, script, monkeypatch):
        real = ROOT / "data" / "arkrag" / "corpus_meta.json"
        if not real.exists():
            pytest.skip("真实数据目录不存在，跳过")
        before = real.read_bytes()
        _wire(script, monkeypatch, script.TEST_ROOT)
        assert script.build(_argv()) == 0
        assert real.read_bytes() == before


# ---------------------------------------------------------------- 正常发布


class TestHappyPath:
    def test_build_publishes_and_pointer_points_to_it(self, script, monkeypatch):
        _wire(script, monkeypatch, script.TEST_ROOT)
        assert script.build(_argv()) == 0

        cur = release.read_current()
        assert cur is not None
        gen = release.Generation(cur["generation_id"])
        assert gen.status() == release.PUBLISHED
        assert release.verify_generation(gen) == []
        # 产物在世代目录里，不在正式路径上
        assert gen.graph_path.exists() and gen.index_path.exists()
        assert gen.gate_report_path.exists()

    def test_no_publish_stops_at_ready(self, script, monkeypatch):
        _wire(script, monkeypatch, script.TEST_ROOT)
        assert script.build(_argv(no_publish=True)) == 0
        assert release.read_current() is None
        gens = release.list_generations()
        assert len(gens) == 1 and gens[0]["status"] == release.READY

    def test_second_build_records_parent(self, script, monkeypatch):
        _wire(script, monkeypatch, script.TEST_ROOT)
        assert script.build(_argv()) == 0
        first = release.read_current()["generation_id"]
        assert script.build(_argv()) == 0
        second = release.read_current()
        assert second["previous_generation"] == first

    def test_run_events_written_for_each_stage(self, script, monkeypatch):
        _wire(script, monkeypatch, script.TEST_ROOT)
        assert script.build(_argv()) == 0
        run_dirs = [p for p in (release.RUNS_DIR).iterdir() if p.is_dir()]
        assert run_dirs
        events = []
        for d in run_dirs:
            p = d / "events.jsonl"
            if p.exists():
                events.extend(json.loads(x) for x in p.read_text("utf-8").splitlines() if x.strip())
        stages = {e["stage"] for e in events}
        assert {"start", "snapshot", "build", "gate", "verify", "publish", "run"} <= stages


# ---------------------------------------------------------------- 必须能失败


class TestMustFail:
    def test_gate_failure_publishes_nothing(self, script, monkeypatch):
        """门禁 FAIL：候选留盘标 FAILED，指针不存在。"""
        _wire(script, monkeypatch, script.TEST_ROOT, gate_fails=True)
        assert script.build(_argv()) == 1
        assert release.read_current() is None
        gens = release.list_generations()
        assert len(gens) == 1
        assert gens[0]["status"] == release.FAILED

    def test_gate_failure_does_not_disturb_existing_release(self, script, monkeypatch):
        """最要紧的一条：新版本门禁 FAIL，线上必须还是旧版本。"""
        _wire(script, monkeypatch, script.TEST_ROOT)
        assert script.build(_argv()) == 0
        before = (release.CURRENT_PATH).read_bytes()
        good = release.read_current()["generation_id"]

        _wire(script, monkeypatch, script.TEST_ROOT, gate_fails=True)
        assert script.build(_argv()) == 1
        assert (release.CURRENT_PATH).read_bytes() == before
        assert release.read_current()["generation_id"] == good
        # 旧版本的自检仍然通得过
        assert release.verify_generation(release.Generation(good)) == []

    def test_missing_extractions_refuse_to_build(self, script, monkeypatch):
        _wire(script, monkeypatch, script.TEST_ROOT)
        monkeypatch.setattr(script.cache, "is_cached_for_chunk", lambda *a, **k: False)
        assert script.build(_argv()) == 2
        assert release.list_generations() == []
        assert release.read_current() is None

    def test_tampered_candidate_is_caught_by_readback(self, script, monkeypatch):
        """回读损坏：候选写完后被改动，发布前校验必须拦住。"""
        _wire(script, monkeypatch, script.TEST_ROOT)
        real_save = script.index.save_index

        def corrupt_save(idx, meta, path):
            real_save(idx, meta, path)
            Path(path).write_bytes(b'{"meta": {}, "doc_id": {}, "doc_len": {}, "postings": {}}')

        monkeypatch.setattr(script.index, "save_index", corrupt_save)
        assert script.build(_argv()) == 1
        assert release.read_current() is None
        assert release.list_generations()[0]["status"] == release.FAILED


# ---------------------------------------------------------------- 回滚链


class TestRollbackChain:
    def test_can_roll_back_to_previous_generation(self, script, monkeypatch):
        _wire(script, monkeypatch, script.TEST_ROOT)
        assert script.build(_argv()) == 0
        first = release.read_current()["generation_id"]
        assert script.build(_argv()) == 0
        second = release.read_current()["generation_id"]
        assert first != second

        pointer = release.rollback(first)
        assert pointer["generation_id"] == first
        assert release.read_current()["generation_id"] == first
        # 回滚不重建：第二代产物仍在，且仍自洽
        assert release.verify_generation(release.Generation(second)) == []

    def test_rolled_back_generation_still_serves_intact_artifacts(self, script, monkeypatch):
        _wire(script, monkeypatch, script.TEST_ROOT)
        assert script.build(_argv()) == 0
        first = release.read_current()["generation_id"]
        gen = release.Generation(first)
        graph_before = gen.graph_path.read_bytes()
        index_before = gen.index_path.read_bytes()

        assert script.build(_argv()) == 0
        release.rollback(first)

        assert gen.graph_path.read_bytes() == graph_before
        assert gen.index_path.read_bytes() == index_before
