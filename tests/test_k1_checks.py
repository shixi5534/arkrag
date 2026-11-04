"""K1 检查脚本的自检：**每一项都必须能失败**。

`scripts/k1_checks.py` 的价值不在"跑出九个 PASS"，而在"条件不满足时会变红"。
一个永远返回 True 的检查函数和没有这个函数是一样的，甚至更糟——它会让
台账里多出九行 PASS，而没人知道那九行是怎么来的。

所以这里对九项逐项构造「本该拦住」的场景，断言它确实返回 `ok=False`，
并且 `detail` 里说的是那件事（而不是碰巧因为别的原因红了）。

## 为什么用 monkeypatch 而不是真造坏数据

真造坏数据（改 Gold Set、删冻结基线、把磁盘填满）会污染工作区，而且
"把磁盘填满"在测试里做不到。这里的做法是：把**判据的输入**换掉，让判据自己
得出结论。它证明的是"判据写对了"，不证明"盘上现在是好的"——后者由手工
跑一次的实测输出负责（记在 `docs/k阶段实测.md`）。

## 与卫生门的关系

本文件进入 `TEST_FILES` 后会让 `tests/test_suite_hygiene.py` 的四个 parametrize
各多一项（+4），加上本文件的用例数，`N` 需要同步上调。这是有意设置的摩擦。
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_k1():
    spec = importlib.util.spec_from_file_location("k1_checks_under_test", ROOT / "scripts" / "k1_checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


k1 = _load_k1()


# ---------------------------------------------------------------- 夹具


class _FakeGeneration:
    """只实现 K1 检查用到的那几个属性的假世代。"""

    def __init__(self, manifest: dict, graph_payload: dict | None = None):
        self._manifest = manifest
        self._graph = graph_payload

    def load(self) -> dict:
        return dict(self._manifest)

    @property
    def graph_path(self) -> Path:
        # 指向一个必然不存在的路径：让 `graph_meta` 保持空，
        # 从而只比较 Manifest 记录的那一侧，把变量减到最少。
        return ROOT / "_k1_spec_fake_graph_that_does_not_exist.json"


def _empty_ctx(**over) -> dict:
    ctx = {
        "generations": [],
        "current": {"generation_id": "g-20260101T000000Z-aaaaaa", "status": "PUBLISHED"},
        "versions": {
            "schema_version": "ark-rag-1",
            "prompt_version": "v2",
            "prompt_sha256": "0" * 64,
            "normalizer_version": "n2",
            "task_schema": "ark-rag-task-2",
            "extraction_schema": "ark-rag-extraction-2",
            "entity_rule_version": "ev2",
            "relation_version": "rv2",
            "vocab_sha": "1" * 16,
        },
        "git_revision": "deadbeef",
    }
    ctx.update(over)
    return ctx


def _gen(gid: str, status: str, *, problems=(), files=7, size=254_005_061) -> dict:
    return {
        "generation_id": gid,
        "status": status,
        "path": str(ROOT / "_k1_spec_never_read"),
        "problems": list(problems),
        "files": files,
        "bytes": size,
    }


def _result_ok(res: dict) -> bool:
    return bool(res["ok"])


# ---------------------------------------------------------------- 1 Schema 冻结


class TestSchemaFrozenCanFail:
    def test_missing_identity_generation_fails(self):
        """盘上没有 ark-rag-3 世代 → 无法证明一致性 → 必须红。"""
        res = k1.check_schema_frozen(_empty_ctx())
        assert not _result_ok(res)
        assert "ark-rag-3" in res["detail"]

    def test_prompt_sha_mismatch_fails(self, monkeypatch):
        """代码的 prompt 摘要与产物记录的不一致 → 必须红，且要指名是哪一个键。"""
        recorded = dict(_empty_ctx()["versions"])
        recorded["prompt_sha256"] = "f" * 64
        fake = _FakeGeneration(
            {
                "graph_schema": "ark-rag-3",
                "versions": recorded,
                "created_at": "2026-11-03T00:00:00Z",
            }
        )
        monkeypatch.setattr(k1, "_newest_identity_generation", lambda rows: {"generation_id": "g-fake"})
        monkeypatch.setattr(k1.release, "Generation", lambda gid: fake)

        res = k1.check_schema_frozen(_empty_ctx())
        assert not _result_ok(res)
        assert "prompt_sha256" in res["detail"]
        assert res["data"]["mismatch"][0]["key"] == "prompt_sha256"

    def test_matching_versions_pass(self, monkeypatch):
        """反向对照：全一致时必须通过，否则上一条测试就只是"总是红"。"""
        # 只放产物**真的会记**的那几个键。`task_schema` / `extraction_schema`
        # 在真实 Manifest 与图里都不存在，把它们塞进 `versions` 会让
        # "未验证"那一条永远为空——测试就退化成不会失败的样子。
        recorded = {
            "schema_version": "ark-rag-1",
            "prompt_version": "v2",
            "prompt_sha256": "0" * 64,
            "normalizer_version": "n2",
            "vocab_sha": "1" * 16,
        }
        fake = _FakeGeneration(
            {
                "graph_schema": "ark-rag-3",
                "versions": recorded,
                "created_at": "2026-11-03T00:00:00Z",
            }
        )
        monkeypatch.setattr(k1, "_newest_identity_generation", lambda rows: {"generation_id": "g-fake"})
        monkeypatch.setattr(k1.release, "Generation", lambda gid: fake)

        ctx = _empty_ctx()
        # 代码侧的 vocab_sha 必须与假产物一致，否则会因别的原因红
        ctx["versions"] = dict(ctx["versions"])
        ctx["versions"]["vocab_sha"] = "1" * 16
        res = k1.check_schema_frozen(ctx)
        assert _result_ok(res), res["detail"]
        assert res["data"]["mismatch"] == []
        # 契约版本两端都不记，必须显式列为"未验证"而不是悄悄当成通过
        assert set(res["data"]["unverified_in_artifacts"]) == {"task_schema", "extraction_schema"}

    def test_prompt_sha_must_be_the_64_char_digest(self):
        """首版把 16 位展示指纹当成 64 位摘要，实测红了一次。

        这里把这条教训钉住：`_repo_versions()` 里的 `prompt_sha256` 长度必须是 64，
        且必须等于 `fingerprint.prompt_digest()`。取成 `prompt.prompt_fingerprint()`
        会得到 16 位，两次前 16 位相同 → 报出一条"代码=7583… 产物=7583…"的
        令人费解的不一致，方向完全指错。
        """
        versions = k1._repo_versions()
        assert len(versions["prompt_sha256"]) == 64
        from arkrag import fingerprint

        assert versions["prompt_sha256"] == fingerprint.prompt_digest()


# ---------------------------------------------------------------- 2/3 pytest 两项


class TestPytestBackedChecksCanFail:
    def test_contract_tests_fail_when_pytest_fails(self, monkeypatch):
        monkeypatch.setattr(
            k1,
            "_run_pytest",
            lambda paths: {
                "command": "pytest",
                "files": list(paths),
                "missing": [],
                "returncode": 1,
                "summary": "2 failed, 74 passed",
                "stdout_tail": ["2 failed"],
                "stderr_tail": [],
            },
        )
        res = k1.check_contract_tests(_empty_ctx())
        assert not _result_ok(res)
        assert "2 failed" in res["detail"]

    def test_contract_tests_fail_when_file_missing(self, monkeypatch):
        """测试文件被删掉时必须是红，而不是"没东西可跑所以过了"。"""
        monkeypatch.setattr(
            k1,
            "_run_pytest",
            lambda paths: {
                "command": "pytest",
                "files": [],
                "missing": ["tests/test_contracts.py"],
                "returncode": 0,
                "summary": "no tests ran",
                "stdout_tail": [],
                "stderr_tail": [],
            },
        )
        res = k1.check_contract_tests(_empty_ctx())
        assert not _result_ok(res)
        assert "缺测试文件" in res["detail"]

    def test_fault_injection_fail_when_pytest_fails(self, monkeypatch):
        monkeypatch.setattr(
            k1,
            "_run_pytest",
            lambda paths: {
                "command": "pytest",
                "files": list(paths),
                "missing": [],
                "returncode": 1,
                "summary": "1 failed, 145 passed",
                "stdout_tail": ["1 failed"],
                "stderr_tail": [],
            },
        )
        res = k1.check_fault_injection(_empty_ctx())
        assert not _result_ok(res)

    def test_declared_test_paths_exist(self):
        """K1 检查里点名的测试文件必须真的存在——否则那两项会永远"通过"。"""
        for path in k1.CONTRACT_TESTS + k1.FAULT_TESTS:
            assert (ROOT / path).exists(), path


# ---------------------------------------------------------------- 4 Gold Set 固定


class TestGoldSetCanFail:
    def test_changed_frozen_sha_fails(self, monkeypatch):
        """把冻结值改成别的值 → 真实题库必然对不上 → 红。"""
        monkeypatch.setattr(k1, "GOLD_SET_SHA256_16", "0" * 16)
        res = k1.check_gold_set_fixed(_empty_ctx())
        assert not _result_ok(res)
        assert "sha256" in res["detail"]

    def test_changed_quota_fails(self, monkeypatch):
        """只改配额、不改值 → 也必须红（防止"换题不换长"绕过 sha 检查）。"""
        quotas = dict(k1.GOLD_SET_QUOTAS)
        quotas["single_chunk"] = 65
        monkeypatch.setattr(k1, "GOLD_SET_QUOTAS", quotas)
        res = k1.check_gold_set_fixed(_empty_ctx())
        assert not _result_ok(res)
        assert "single_chunk" in res["detail"]

    def test_real_gold_set_passes(self):
        """反向对照：在当前盘上这份题库上必须是绿。"""
        res = k1.check_gold_set_fixed(_empty_ctx())
        assert _result_ok(res), res["detail"]
        assert res["data"]["actual_total"] == 320
        assert res["data"]["actual_sha256_16"] == "76d66073ff87aab1"


# ---------------------------------------------------------------- 5 旧产物已备份


class TestBackupCanFail:
    def test_single_servable_generation_fails(self, monkeypatch):
        """只剩一代可服务时"回滚"没有落点 → 必须红。"""
        monkeypatch.setattr(k1, "_sha256_file", k1._sha256_file)
        ctx = _empty_ctx(
            generations=[_gen("g-20260101T000000Z-aaaaaa", "PUBLISHED")],
        )
        monkeypatch.setattr(
            k1,
            "legacy",
            type(
                "L",
                (),
                {"BASELINE_MANIFEST": Path(str(ROOT / "README.md")), "baseline_paths": staticmethod(lambda: ())},
            ),
        )
        monkeypatch.setattr(
            k1.subprocess,
            "run",
            lambda *a, **kw: type(
                "R", (), {"returncode": 0, "stdout": "PASS", "stderr": ""}
            )(),
        )
        res = k1.check_old_artifacts_backed_up(ctx)
        assert not _result_ok(res)
        assert "可服务世代只有 1" in res["detail"]

    def test_broken_baseline_fails(self, monkeypatch):
        """冻结基线校验非零退出 → 必须红，并把退出码写进 data。"""
        monkeypatch.setattr(
            k1,
            "legacy",
            type(
                "L",
                (),
                {"BASELINE_MANIFEST": Path(str(ROOT / "README.md")), "baseline_paths": staticmethod(lambda: ("a",))},
            ),
        )
        monkeypatch.setattr(
            k1.subprocess,
            "run",
            lambda *a, **kw: type(
                "R", (), {"returncode": 3, "stdout": "产物 2 项不一致", "stderr": ""}
            )(),
        )
        ctx = _empty_ctx(
            generations=[
                _gen("g-20260101T000000Z-aaaaaa", "PUBLISHED"),
                _gen("g-20260102T000000Z-bbbbbb", "READY"),
            ]
        )
        res = k1.check_old_artifacts_backed_up(ctx)
        assert not _result_ok(res)
        assert res["data"]["snapshot_verify_returncode"] == 3
        assert "冻结基线" in res["detail"]


# ---------------------------------------------------------------- 6 磁盘


class TestDiskSpaceCanFail:
    def test_insufficient_free_space_fails(self, monkeypatch):
        """剩余空间低于阈值 → 必须红（阈值不是拍脑袋的，见常量注释）。"""
        monkeypatch.setattr(
            k1.shutil,
            "disk_usage",
            lambda path: type("U", (), {"total": 10 * 1024**3, "used": 10 * 1024**3, "free": 1024**3})(),
        )
        res = k1.check_disk_space(_empty_ctx())
        assert not _result_ok(res)
        assert "磁盘不足" in res["detail"]

    def test_large_generation_raises_requirement(self, monkeypatch):
        """单代产物特别大时阈值必须跟着抬——不能只用一个常数糊过去。"""
        monkeypatch.setattr(
            k1.shutil,
            "disk_usage",
            lambda path: type("U", (), {"total": 100 * 1024**3, "used": 0, "free": 3 * 1024**3})(),
        )
        # 2 GiB 的固定阈值下 3 GiB 是够的；但单代 1.5 GiB × 3 = 4.5 GiB 就不够了
        ctx = _empty_ctx(generations=[_gen("g-x", "READY", size=1536 * 1024**2)])
        res = k1.check_disk_space(ctx)
        assert not _result_ok(res)
        assert res["data"]["required_bytes"] == 3 * 1536 * 1024**2

    def test_real_disk_passes(self):
        res = k1.check_disk_space(_empty_ctx())
        assert _result_ok(res), res["detail"]
        assert res["data"]["required_bytes"] == k1.REQUIRED_FREE_BYTES


# ---------------------------------------------------------------- 7 模型预算


class TestModelBudgetCanFail:
    def test_llm_without_budget_fails(self, monkeypatch):
        """配了 LLM 端点却没声明预算 → 必须红（挡的是"悄悄开始烧钱"）。"""
        monkeypatch.setattr(k1.config, "LLM_MODEL", "qwen2.5:7b")
        monkeypatch.delenv("ARKRAG_K1_LLM_BUDGET", raising=False)
        res = k1.check_model_budget(_empty_ctx())
        assert not _result_ok(res)
        assert "ARKRAG_K1_LLM_BUDGET" in res["detail"]

    def test_llm_with_malformed_budget_fails(self, monkeypatch):
        """预算声明只有一段（缺币种/上限/次数）→ 必须红。"""
        monkeypatch.setattr(k1.config, "LLM_MODEL", "qwen2.5:7b")
        monkeypatch.setenv("ARKRAG_K1_LLM_BUDGET", "100")
        res = k1.check_model_budget(_empty_ctx())
        assert not _result_ok(res)
        assert "三段" in res["detail"]

    def test_llm_with_full_budget_passes(self, monkeypatch):
        monkeypatch.setattr(k1.config, "LLM_MODEL", "qwen2.5:7b")
        monkeypatch.setenv("ARKRAG_K1_LLM_BUDGET", "CNY/50/3200")
        res = k1.check_model_budget(_empty_ctx())
        assert _result_ok(res), res["detail"]

    def test_no_llm_endpoint_passes_as_zero_budget(self, monkeypatch):
        """本机现状：无端点 → 预算是 0，这是一个可验证的事实而不是"跳过"。"""
        monkeypatch.setattr(k1.config, "LLM_MODEL", "")
        monkeypatch.setattr(k1.config, "LLM_BASE_URL", "")
        res = k1.check_model_budget(_empty_ctx())
        assert _result_ok(res)
        assert res["data"]["budget_required"] is False


# ---------------------------------------------------------------- 8 可恢复


class TestResumableCanFail:
    def _patch_cache_clean(self, monkeypatch):
        monkeypatch.setattr(
            k1.cache,
            "invalid_cache_report",
            lambda: {"total": 10, "ok": 10, "invalid": 0, "by_reason": {}, "samples": {}},
        )

    def test_unknown_status_fails(self, monkeypatch):
        """出现未知状态字面量 → 必须红：没人知道这一代能不能发布。"""
        self._patch_cache_clean(monkeypatch)
        monkeypatch.setattr(k1.release, "publish", lambda gen: {})
        ctx = _empty_ctx(
            generations=[
                _gen("g-20260101T000000Z-aaaaaa", "PUBLISHED"),
                _gen("g-20260102T000000Z-bbbbbb", "ARCHIVED"),
            ]
        )
        res = k1.check_resumable(ctx)
        assert not _result_ok(res)
        assert "未知世代状态" in res["detail"]

    def test_publish_accepting_failed_generation_fails(self, monkeypatch):
        """`release.publish` 若接受了 FAILED 世代 → 必须红（现场证明隔离前提）。"""
        self._patch_cache_clean(monkeypatch)
        monkeypatch.setattr(k1.release, "publish", lambda gen: {"generation_id": "ok"})
        ctx = _empty_ctx(generations=[_gen("g-20260101T000000Z-aaaaaa", "FAILED")])
        res = k1.check_resumable(ctx)
        assert not _result_ok(res)
        assert res["data"]["publish_probe"] == "ACCEPTED"

    def test_current_generation_missing_on_disk_fails(self, monkeypatch):
        """指针指向的世代在盘上不存在 → 必须红（此时没人知道线上是哪一份）。"""
        self._patch_cache_clean(monkeypatch)
        monkeypatch.setattr(k1.release, "publish", self._raise_illegal())
        ctx = _empty_ctx(
            # 指针指 g-gone，盘上只有 aaaaaa
            current={"generation_id": "g-20260109T000000Z-gone00", "status": "PUBLISHED"},
            generations=[_gen("g-20260101T000000Z-aaaaaa", "PUBLISHED")],
        )
        res = k1.check_resumable(ctx)
        assert not _result_ok(res)
        assert "在盘上不存在" in res["detail"]

    def test_invalid_cache_count_fails(self, monkeypatch):
        """缓存里有非法条目 → 必须红。注意看 `invalid` 而不是 `total`。"""
        monkeypatch.setattr(
            k1.cache,
            "invalid_cache_report",
            lambda: {"total": 13801, "ok": 13800, "invalid": 1, "by_reason": {"bad_format": 1}, "samples": {}},
        )
        monkeypatch.setattr(k1.release, "publish", self._raise_illegal())
        ctx = _empty_ctx(generations=[_gen("g-20260101T000000Z-aaaaaa", "PUBLISHED")])
        res = k1.check_resumable(ctx)
        assert not _result_ok(res)
        assert "1 条非法记录" in res["detail"]

    def test_clean_invalid_cache_is_not_confused_with_total(self, monkeypatch):
        """13801 个文件、0 个坏的必须是绿。

        首版把 `total` 当成坏的数量，报出 "抽取缓存里有 13801 条非法记录"，
        而真相是一条都没有。这类错误最危险的地方是它看起来像个重大发现。
        """
        self._patch_cache_clean(monkeypatch)
        monkeypatch.setattr(k1.release, "publish", self._raise_illegal())
        ctx = _empty_ctx(generations=[_gen("g-20260101T000000Z-aaaaaa", "PUBLISHED")])
        res = k1.check_resumable(ctx)
        assert _result_ok(res), res["detail"]
        assert res["data"]["invalid_cache"]["total"] == 10
        assert res["data"]["invalid_cache"]["invalid"] == 0

    @staticmethod
    def _raise_illegal():
        def _publish(gen):
            raise k1.release.IllegalTransition("只有 READY 能发布")

        return _publish


# ---------------------------------------------------------------- 9 回滚已实测


class TestRollbackTestedCanFail:
    def test_needs_two_servable_generations(self, monkeypatch):
        ctx = _empty_ctx(generations=[_gen("g-20260101T000000Z-aaaaaa", "PUBLISHED")])
        res = k1.check_rollback_tested(ctx)
        assert not _result_ok(res)
        assert "不足 2" in res["detail"]

    def test_refused_rollback_fails(self, monkeypatch):
        """回滚命令本身抛错 → 必须红，而不是"演练过了"。"""
        ctx = _empty_ctx(
            generations=[
                _gen("g-20260101T000000Z-aaaaaa", "PUBLISHED"),
                _gen("g-20260102T000000Z-bbbbbb", "READY"),
            ]
        )

        def _boom(gid, **kw):
            raise k1.release.ReleaseError("回滚目标自检不过（2 项）")

        monkeypatch.setattr(k1.release, "rollback", _boom)
        res = k1.check_rollback_tested(ctx)
        assert not _result_ok(res)
        assert "检查自身抛异常" in res["detail"] or "回滚" in res["detail"]

    def test_pointer_key_mismatch_is_detected(self, monkeypatch):
        """回滚写出的指针键集缺 `rolled_back` → 必须红。

        探针目录里写临时指针，`release.CURRENT_PATH` 全程被换走再复原，
        线上指针一个字节都不动。
        """
        probe_dir = ROOT / "_k1_spec_rollback_probe"
        probe_dir.mkdir(exist_ok=True)
        ctx = _empty_ctx(
            generations=[
                _gen("g-20260101T000000Z-aaaaaa", "PUBLISHED"),
                _gen("g-20260102T000000Z-bbbbbb", "READY"),
            ]
        )
        original = k1.release.CURRENT_PATH

        def _fake_rollback(gid, **kw):
            pointer = {
                "generation_id": gid,
                "status": k1.release.PUBLISHED,
                "published_at": "2026-11-03T00:00:00Z",
                "previous_generation": "",
                "source_fingerprint": "",
                "graph_artifact_id": "",
                "index_artifact_id": "",
                "gate_report_hash": "",
            }
            target = Path(k1.release.CURRENT_PATH)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(pointer), encoding="utf-8")
            return pointer

        monkeypatch.setattr(k1.release, "rollback", _fake_rollback)
        monkeypatch.setattr(k1.config, "CIRCUIT_OPEN_PATH", str(probe_dir / "circuit.json"))
        try:
            res = k1.check_rollback_tested(ctx)
        finally:
            k1.release.CURRENT_PATH = original
            shutil.rmtree(probe_dir, ignore_errors=True)
        assert not _result_ok(res)
        assert "键集" in res["detail"]

    def test_probe_dir_is_unique_per_run_and_cleaned_up(self, monkeypatch):
        """探针目录必须每次运行独一份，且跑完删掉。

        原先写死成 `data/arkrag/_k1_rollback_probe/`，两个进程同时跑这一项就
        共用同一个 `current.json`：A 刚写完"当前是候选世代"，B 已经把它回滚成
        `e0aacc`，A 接着在 B 的指针上再回滚一次——要么被"已经是当前世代"拒掉
        （判成"回滚命令不可用"），要么被接受（判成"回滚到自身未被拒"）。
        两种都是假结论，且**单独跑永远绿**：实测并发 4 份 `--only rollback_tested`
        修正前 0/4 通过、修正后 4/4 通过。

        这条用例钉住的就是"不再共享"这个性质，不是"用了 mkdtemp"这个实现：
        跑两次拿到两个不同的目录，都不在 `CIRCUIT_OPEN_PATH` 的同级目录下，
        且跑完都不存在。
        """
        seen: list[str] = []

        def _fake_rollback(gid, **kw):
            seen.append(str(k1.release.CURRENT_PATH))
            pointer = {
                "generation_id": gid,
                "status": k1.release.PUBLISHED,
                "published_at": "2026-11-03T00:00:00Z",
                "previous_generation": "",
                "rolled_back": True,
                "source_fingerprint": "",
                "graph_artifact_id": "",
                "index_artifact_id": "",
                "gate_report_hash": "",
            }
            target = Path(k1.release.CURRENT_PATH)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(pointer), encoding="utf-8")
            return pointer

        monkeypatch.setattr(k1.release, "rollback", _fake_rollback)
        ctx = _empty_ctx(
            generations=[
                _gen("g-20260101T000000Z-aaaaaa", "PUBLISHED"),
                _gen("g-20260102T000000Z-bbbbbb", "READY"),
            ]
        )
        original = k1.release.CURRENT_PATH
        try:
            k1.check_rollback_tested(ctx)
            first_run = list(seen)
            k1.check_rollback_tested(ctx)
            second_run = list(seen)[len(first_run):]
        finally:
            k1.release.CURRENT_PATH = original

        assert first_run and second_run, (first_run, second_run)
        dirs_first = {Path(p).parent for p in first_run}
        dirs_second = {Path(p).parent for p in second_run}
        # 一次检查里 `rollback` 会被调两次（真回滚 + 回滚到自身的探针），
        # 所以按"每次检查"看目录集合，而不是按"每次调用"。
        assert len(dirs_first) == 1, f"同一次检查里换了探针目录：{dirs_first}"
        assert len(dirs_second) == 1, f"同一次检查里换了探针目录：{dirs_second}"
        assert dirs_first != dirs_second, f"两次运行共用了同一个探针目录：{dirs_first}"
        shared = Path(k1.config.CIRCUIT_OPEN_PATH).parent.resolve()
        for d in dirs_first | dirs_second:
            assert d.resolve().parent != shared, f"探针目录落在冻结目录里：{d}"
            assert not d.exists(), f"探针目录没被清掉：{d}"

    def test_real_pointer_is_never_written(self):
        """整个检查跑完后，真实指针的字节必须没变。

        这条是上面那条测试的护栏：探针万一漏了复原，会静默把线上指针指到别处，
        而唯一迹象是服务回答的世代变了。测试里直接比对字节。
        """
        pointer_path = k1.release.CURRENT_PATH
        before = pointer_path.read_bytes() if pointer_path.exists() else b""
        ctx = _empty_ctx(
            generations=[
                _gen("g-20260101T000000Z-aaaaaa", "PUBLISHED"),
                _gen("g-20260102T000000Z-bbbbbb", "READY"),
            ]
        )
        # 目标世代在盘上不存在，检查会失败，但指针仍然不许被写
        k1.check_rollback_tested(ctx)
        after = pointer_path.read_bytes() if pointer_path.exists() else b""
        assert before == after


# ---------------------------------------------------------------- run_checks 组装


class TestRunChecksAssembly:
    def test_unknown_only_id_is_rejected_by_main(self, capsys):
        code = k1.main(["--only", "no_such_check"])
        assert code == 2
        assert "未知检查 id" in capsys.readouterr().out

    def test_list_prints_nine_ids(self, capsys):
        code = k1.main(["--list"])
        out = capsys.readouterr().out
        assert code == 0
        for cid in k1.CHECK_IDS:
            assert cid in out
        assert len(k1.CHECK_IDS) == 9

    def test_check_that_raises_does_not_kill_the_round(self, monkeypatch):
        """单项抛异常不能拖垮整轮：那一项记红，其余照跑。"""

        def _boom(ctx):
            raise RuntimeError("这一项炸了")

        monkeypatch.setattr(k1, "CHECKS", (("schema_frozen", _boom, "假检查"), ("disk_space", k1.check_disk_space, "磁盘足够")))
        report = k1.run_checks(())
        ids = [c["id"] for c in report["checks"]]
        assert ids == ["schema_frozen", "disk_space"]
        assert report["checks"][0]["ok"] is False
        assert "这一项炸了" in report["checks"][0]["detail"]
        # 第二项仍然被跑到了
        assert report["checks"][1]["ok"] is True
        assert report["ok"] is False
        assert report["blocking_failed"] == ["schema_frozen"]


    def test_partial_run_must_not_claim_all_nine_passed(self, capsys):
        """`--only` 只跑子集时，结论绝不能说"九项全部通过"。

        实测踩到过：`--only schema_frozen,gold_set_fixed,disk_space` 跑了 3 项，
        却打印"九项 blocking 检查全部通过，可以进入 K2"。当时三项确实都绿，
        所以那句话不是"假的"，但它会让人以为九项都查过了——一个只查了三分之
        一的报告，最坏的结果是被当成放行凭据。
        """
        code = k1.main(["--only", "disk_space"])
        out = capsys.readouterr().out
        assert code == 0
        assert "九项 blocking 检查全部通过" not in out
        assert "只覆盖 1/9" in out or "只跑了 1/9" in out

    def test_full_run_still_claims_all_nine(self, monkeypatch):
        """反向对照：九项都跑完且都通过时，那句话必须还在。"""
        monkeypatch.setattr(
            k1,
            "CHECKS",
            tuple((cid, lambda ctx: k1._result(cid, name, True, "假通过", {}), name) for cid, _, name in k1.CHECKS),
        )
        report = k1.run_checks(())
        assert report["ok"] is True
        assert len(report["checks"]) == 9


# ---------------------------------------------------------------- 与实测对齐


class TestMatchesMeasuredFacts:
    """把 K1 首轮实测里出现的两个自伤错误钉成回归测试。"""

    def test_invalid_cache_report_contract(self):
        """`invalid_cache_report()` 的键集必须是 total/ok/invalid/by_reason/samples。

        这里不读真实缓存（那要扫 13801 个文件），只盯契约：
        谁是总数、谁是坏的数量。混淆这两者会得出"13801 条非法记录"的假重大发现。
        """
        from arkrag import cache

        report = cache.invalid_cache_report()
        assert {"total", "ok", "invalid", "by_reason", "samples"} <= set(report)
        assert report["ok"] + report["invalid"] == report["total"]

    def test_prompt_digest_and_display_fingerprint_are_different(self):
        """两个函数名字相近、前 16 位相同，但长度不同，用途不同。"""
        from arkrag import fingerprint, prompt

        digest = fingerprint.prompt_digest()
        display = prompt.prompt_fingerprint()
        assert len(digest) == 64
        assert len(display) == 16
        assert digest.startswith(display)
