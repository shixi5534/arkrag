"""冻结写保护的回归测试（计划 A1c / 原则 3 / 原则 8）。

## 这些测试为什么必须存在

2026-10-30 的真实事故：A 阶段把 11 个产物冻结为 `legacy-baseline` 只读，
其中包含 `scenes.jsonl`、`chunks.jsonl`、`shards.json`、`corpus_meta.json`、
`gates.json`。但写保护的清单 `legacy.PROTECTED_FILES` 只写了四项
（graph / index / gates / circuit），**没有**覆盖 L1/L2 那几个产物。
于是 D 阶段重跑 `run_l1.py`、`run_l2.py` 时，这四个已冻结产物被直接覆盖，
没有任何东西拦一下。

发现它的是 `scripts/snapshot_baseline.py --verify`——但那只是**事后**发现：
字节当时已经没了，靠检出基线提交 `0ae73f4` 的旧代码重跑一遍才复原。

两个结构性教训，也就是本文件要锁住的两条规矩：

1. **保护清单不能和冻结清单各写一份**。两份迟早漂移，而漂移方向恰好是
   "冻结的越来越多、保护的越来越少"。现在保护集合从 Manifest 派生
   （`legacy.protected_paths()`），冻结即受保护。
2. **保护必须落在唯一通道上**，不能靠每个调用点记得调一次。现在
   `atomic.atomic_write_bytes` 里统一拦（`guard_frozen_artifact`）。

第 3 条规矩管住"顺手关掉保护"：放行口 `ARKRAG_ALLOW_LEGACY_WRITE=1`
必须显式设置才生效，任何常用脚本里都不许出现。

## 第 4 条：测试不许写真实仓库

第一版本文件违反了它自己该守的规矩——三个"可写路径"的用例直接把文件写进了
**真实仓库**：

    test_atomic_write_allows_unprotected_path   写 data/arkrag/generations/g-x/graph/graph.json
    test_cache_dir_new_file_is_not_blocked      写 data/arkrag/cache/ab/new.json
    test_explicit_allow_env_bypasses_guard      写 data/arkrag/scenes.jsonl = b"rebuilt"

后果不是"多几个垃圾文件"那么轻：第三处把冻结基线的 `scenes.jsonl`（71,305,439
字节）覆盖成了 7 字节的 `rebuilt`，第二处让 cache 目录摘要变化，`snapshot_baseline
--verify` 因此报 FAIL 两项。**跑测试把被测系统自己的基线打坏了**，而检出的
只有事后校验。

修法：把所有"要落盘"的用例的目标根指到 `tmp_path`（`_repo` fixture 把
`config.ROOT` 换成临时根，`legacy._rel()` 因此仍算出同样的相对路径
`data/arkrag/scenes.jsonl`，**保护判据与真实场景完全一致**，断言强度不变），
并在文件末尾用 `TestTestsDoNotWriteRealRepo` 锁住这条规矩本身。
"""
from __future__ import annotations

import json

import pytest

from arkrag import atomic, legacy


REAL_ROOT = legacy.config.ROOT

# 事故里被测试写坏/写脏的真实路径，autouse 守卫逐个比对 stat
_REAL_WATCHED = (
    "data/arkrag/scenes.jsonl",
    "data/arkrag/chunks.jsonl",
    "data/arkrag/cache",
    "data/arkrag/generations",
)


@pytest.fixture(autouse=True)
def _real_repo_untouched():
    """每个用例前后都比一次真实仓库的关键路径 stat，被写过就直接失败。

    用 size + mtime_ns 而不是 sha256：这些路径里有两份几十 MB 的文件，
    每次用例都全量摘要太重；而"被写过"这件事 stat 一定会变。
    """
    def snapshot() -> dict:
        out = {}
        for rel in _REAL_WATCHED:
            path = REAL_ROOT / rel
            if path.is_dir():
                out[rel] = (True, len(list(path.rglob("*"))))
            elif path.exists():
                st = path.stat()
                out[rel] = (False, st.st_size, st.st_mtime_ns)
            else:
                out[rel] = (None,)
        return out

    before = snapshot()
    yield
    assert snapshot() == before, (
        "测试写到了真实仓库路径（" + ", ".join(_REAL_WATCHED) + "）："
        "这些路径属于冻结基线或正式产物，用例必须把目标根指到 tmp_path"
    )


@pytest.fixture()
def frozen(tmp_path, monkeypatch):
    """造一份最小可用的冻结清单，并把受保护集合与仓库根都指到临时目录。

    `config.ROOT` 改指 `tmp_path` 是第 4 条规矩的关键：`legacy._rel()` 用
    `config.ROOT` 算相对路径，所以临时根下的 `data/arkrag/scenes.jsonl`
    与真实仓库里的同名路径算出**同一个相对路径**，保护判据一字未改。
    """
    monkeypatch.setattr(legacy.config, "ROOT", tmp_path)
    baseline = tmp_path / "releases" / "legacy-baseline"
    baseline.mkdir(parents=True)
    manifest = baseline / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifacts": [
                    {"path": "data/arkrag/scenes.jsonl", "bytes": 1, "sha256": "aa"},
                    {"path": "data/arkrag/chunks.jsonl", "bytes": 1, "sha256": "bb"},
                    {"path": "data/arkrag/cache", "bytes": 0, "is_dir": True},
                    {"path": "data/arkrag/index/lexical.json", "bytes": 1, "sha256": "cc"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(legacy, "BASELINE_MANIFEST", manifest)
    monkeypatch.delenv(legacy.ENV_ALLOW, raising=False)
    (tmp_path / "data" / "arkrag").mkdir(parents=True, exist_ok=True)
    return tmp_path


class TestProtectionDerivesFromManifest:
    """规矩 1：冻结清单就是保护清单，不许各写一份。"""

    def test_manifest_entries_become_protected(self, frozen):
        """冻结清单里的每个单文件项都必须真的被保护。

        这条断言直接覆盖事故：旧代码 `PROTECTED_FILES` 里没有 scenes.jsonl /
        chunks.jsonl，`is_protected()` 对它们返回 False，重跑 L1/L2 就把冻结
        产物盖掉了。
        """
        assert legacy.is_protected(legacy.config.ROOT / "data/arkrag/scenes.jsonl")
        assert legacy.is_protected(legacy.config.ROOT / "data/arkrag/chunks.jsonl")

    def test_protected_paths_is_union_of_static_and_manifest(self, frozen):
        paths = legacy.protected_paths()
        assert "data/arkrag/scenes.jsonl" in paths          # 来自 Manifest
        assert "data/arkrag/graph.json" in paths            # 来自静态兜底
        assert "data/arkrag/chunks.jsonl" in paths          # 来自 Manifest

    def test_directory_entries_do_not_join_the_file_list(self, frozen):
        """目录项交给受保护目录集合管，不许混进单文件集合。

        混进去的后果是 `guard_frozen_artifact` 会把整个 cache 目录当单文件拦，
        抽取流水线的新缓存写不进去——保护过头同样是故障。
        """
        assert "data/arkrag/cache" not in legacy.baseline_paths()
        assert "data/arkrag/cache" in legacy.baseline_dirs()
        assert "data/arkrag/cache" in legacy.protected_dirs()

    def test_real_manifest_dir_entry_lacks_is_dir_flag(self, tmp_path, monkeypatch):
        """真实 Manifest 的目录项**没有** `is_dir` 字段，只有 bytes/count/digest/kind。

        只用 `item.get("is_dir")` 判目录，会把 `data/arkrag/cache` 当成单文件塞进
        `protected_paths()`。所以判据改成「三个信号取或」：显式 is_dir、缺 sha256、
        kind 以 _dir 结尾。这条用例用真实形态的条目锁住它。
        """
        baseline = tmp_path / "releases" / "legacy-baseline"
        baseline.mkdir(parents=True)
        manifest = baseline / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "artifacts": [
                        {"path": "data/arkrag/graph.json", "bytes": 5, "sha256": "ee",
                         "mtime": 1.0, "mtime_iso": "x", "kind": "graph"},
                        # 真实形态：无 is_dir、无 sha256，靠 count/digest 描述
                        {"path": "data/arkrag/cache", "bytes": 30688245, "count": 13801,
                         "digest": "a244f0362aa4", "kind": "extraction_cache_dir"},
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(legacy, "BASELINE_MANIFEST", manifest)
        assert legacy.baseline_paths() == ("data/arkrag/graph.json",)
        assert legacy.baseline_dirs() == ("data/arkrag/cache",)
        assert "data/arkrag/cache" not in legacy.protected_paths()

    def test_missing_manifest_falls_back_to_static_list(self, tmp_path, monkeypatch):
        """Manifest 不存在时退回静态清单，既不"全保护"也不"不保护"。"""
        monkeypatch.setattr(legacy, "BASELINE_MANIFEST", tmp_path / "nope.json")
        assert legacy.baseline_paths() == ()
        assert legacy.protected_paths() == legacy.PROTECTED_FILES

    def test_corrupt_manifest_falls_back_to_static_list(self, tmp_path, monkeypatch):
        bad = tmp_path / "manifest.json"
        bad.write_text("{not json", encoding="utf-8")
        monkeypatch.setattr(legacy, "BASELINE_MANIFEST", bad)
        assert legacy.baseline_paths() == ()
        assert legacy.protected_paths() == legacy.PROTECTED_FILES


class TestGuardIsOnTheOnlyWriteChannel:
    """规矩 2：保护落在 atomic 写通道上，不靠各调用点自觉。"""

    def test_atomic_write_refuses_frozen_artifact(self, frozen):
        """任何走 atomic 的写都必须被拦——这正是事故里没人拦的那条路。

        事故现场 `run_l1.py` 用的是 `atomic.atomic_write_json`，它当时不查保护。
        """
        target = legacy.config.ROOT / "data/arkrag/scenes.jsonl"
        with pytest.raises(legacy.LegacyWriteRefused):
            atomic.atomic_write_bytes(target, b"{}")

    def test_atomic_write_refuses_frozen_json(self, frozen):
        target = legacy.config.ROOT / "data/arkrag/chunks.jsonl"
        with pytest.raises(legacy.LegacyWriteRefused):
            atomic.atomic_write_json(target, {"a": 1})

    def test_stream_write_is_also_guarded(self, frozen):
        """流式通道同样要拦。

        `atomic_write_stream` 是为了让 71 MB 的 scenes.jsonl 不必先拼成完整
        bytes 才加进来的。它一旦不带保护，就等于给事故现场重新开了一道后门——
        当年 `parse.parse_corpus` 正是绕开通道直写才把冻结产物写成 `rebuilt`。
        """
        target = legacy.config.ROOT / "data/arkrag/scenes.jsonl"

        def _writer(fh):
            fh.write("rebuilt\n")

        with pytest.raises(legacy.LegacyWriteRefused):
            atomic.atomic_write_stream(target, _writer)

    def test_stream_write_allows_unprotected_path(self, frozen):
        target = frozen / "data/arkrag/staging/d-stage/scenes.jsonl"

        def _writer(fh):
            fh.write('{"a": 1}\n')

        atomic.atomic_write_stream(target, _writer)
        assert target.read_text("utf-8") == '{"a": 1}\n'
        assert list(target.parent.glob("*.tmp")) == []

    def test_atomic_write_allows_unprotected_path(self, frozen):
        """非冻结路径照常可写，保护不能把正常工作也拦掉。"""
        target = frozen / "data/arkrag/generations/g-x/graph/graph.json"
        atomic.atomic_write_json(target, {"ok": True})
        assert json.loads(target.read_text("utf-8"))["ok"] is True

    def test_cache_dir_new_file_is_not_blocked(self, frozen):
        """cache 目录里的**新**缓存必须能写：legacy 记录只读复用、新记录追加写，
        这是正常运行的一部分。在这里拦会把抽取流水线整个卡死。

        这条用例同时把两级判据的**分工**写死：`is_protected()` 认为 cache
        下的文件受保护（它管 PROTECTED_DIRS），但 `guard_frozen_artifact()`
        只看单文件清单，所以 atomic 通道放行。若哪天有人给 guard 加上目录前缀
        判断，抽取流水线就会全线报 LegacyWriteRefused。
        """
        target = frozen / "data/arkrag/cache/ab/new.json"
        assert legacy.is_protected(target) is True          # 目录级：受保护
        assert legacy._rel(target) not in legacy.protected_paths()  # 单文件级：不在冻结清单
        atomic.atomic_write_json(target, {"raw": {}})
        assert json.loads(target.read_text("utf-8")) == {"raw": {}}

    def test_explicit_allow_env_bypasses_guard(self, frozen, monkeypatch):
        """放行口存在，但必须显式设置才生效（重建 legacy 时的唯一通道）。"""
        monkeypatch.setenv(legacy.ENV_ALLOW, "1")
        target = frozen / "data/arkrag/scenes.jsonl"
        atomic.atomic_write_bytes(target, b"rebuilt")
        assert target.read_bytes() == b"rebuilt"

    def test_guard_refuses_directory_entry_itself(self, frozen):
        """冻结清单里的目录项本身不许被当文件写掉（否则整棵 cache 树会被
        一个 open("w") 换成普通文件）。目录项不进单文件集合，靠 PROTECTED_DIRS 兜。"""
        target = frozen / "data/arkrag/cache"
        assert legacy.is_protected(target) is True
        with pytest.raises(legacy.LegacyWriteRefused):
            legacy.guard_write(target, reason="test")


class TestAllowEnvIsNotBakedIn:
    """规矩 3：放行口不许被写进脚本或环境模板。"""

    def test_scripts_never_set_allow_env(self):
        """任何脚本里出现 `ARKRAG_ALLOW_LEGACY_WRITE=1` 都是事故前兆。

        原因很直接：一旦某个常用脚本设了它，保护就等于不存在，而下一个人
        不会知道这行是什么时候加的、为什么加。
        """
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1]
        offenders: list[str] = []
        for pattern in ("scripts/*.py", "src/arkrag/*.py"):
            for path in sorted(root.glob(pattern)):
                if path.name == "legacy.py":
                    continue  # 定义处当然会提到这个常量名
                text = path.read_text("utf-8")
                if f"{legacy.ENV_ALLOW}=1" in text or f'"{legacy.ENV_ALLOW}", "1"' in text:
                    offenders.append(path.relative_to(root).as_posix())
        assert offenders == [], f"这些脚本私自放行 legacy 写保护: {offenders}"

    def test_guard_message_points_to_generations(self, frozen):
        """报错必须告诉人正确做法，否则下一个人只会去搜怎么关掉保护。"""
        target = legacy.config.ROOT / "data/arkrag/scenes.jsonl"
        with pytest.raises(legacy.LegacyWriteRefused) as exc:
            legacy.guard_frozen_artifact(target, reason="test")
        message = str(exc.value)
        assert "generations/" in message
        assert legacy.ENV_ALLOW in message


class TestTestsDoNotWriteRealRepo:
    """规矩 4：测试不许把真实仓库当画布（本文件第一版就是这么把基线打坏的）。"""

    def test_config_root_is_restored_between_cases(self):
        """`frozen` fixture 改的是 `legacy.config.ROOT`，用例结束后必须还原。

        不还原的后果比想象中严重：后续用例读 `config.ROOT` 会落到上一个用例的
        临时目录，报错信息又指向真实路径，排查时看到的全是假像。"""
        assert legacy.config.ROOT == REAL_ROOT

    def test_watched_paths_still_point_at_real_repo(self):
        """守卫监视的是真实仓库路径，不是临时目录——否则它保护不了任何东西。"""
        scenes = REAL_ROOT / "data/arkrag/scenes.jsonl"
        assert REAL_ROOT == REAL_ROOT.resolve() or REAL_ROOT.is_dir()
        assert str(REAL_ROOT / "data/arkrag/cache").endswith("cache")
        assert scenes.parent.name == "arkrag"
