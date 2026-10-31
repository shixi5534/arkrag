"""可移植性：入库的 JSON 里不许出现绝对机台路径（A3「消除 `E:\\图rag` 硬编码」）。

为什么这是个缺陷而不只是"不好看"：`data/arkrag/current.json` 和
`data/arkrag/source_verification.json` 都进 git。它们一旦记下
`E:\\图rag\\data\\arkrag\\generations\\<gid>` 这种字符串，换台机器 clone 下来
那个路径指的是一片空地——而 Definition of Done 里写着"依赖命令部署方式可复现"。

三条必须能**变红**的规矩：

* `config.rel_to_root()`：仓库内路径写成相对仓库根（正斜杠），仓库外原样返回
  （不许写成 `../../..` 逃逸形式）。大小写只影响"判不判定在仓库内"，
  不影响返回值。
* 发布 / 回滚指针**不含** `path`：目录能从 `Generation(gid).dir` 推导，存一份
  就等于存一份会过期的绝对路径副本。两个指针必须保持同构——这条要机检，
  不能只靠 `rollback()` 里的一句注释。
* `read_current()` 读历史指针时把 `path` 当不存在，但**不许**顺手吃掉别的未知
  字段：那些可能是未来版本加的，静默丢弃等于让"降级一次就丢信息"变成常态。

所有落盘操作都在 `tmp_path` 里。`tests/test_build_generation.py` 记着一次真实
事故：测试写了假 `corpus_meta.json` 覆盖了真文件，是基线快照校验发现的。
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from arkrag import config
from arkrag import release
from arkrag import source
from test_release import _build_generation


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """把 generations / runs / current 指到 tmp，绝不碰真实产物。

    与 `tests/test_release.py` 的同名夹具同样做法：`release` 里这三个是模块级
    常量，monkeypatch 才会让函数体跟着变。
    """
    gen_dir = tmp_path / "generations"
    runs_dir = tmp_path / "runs"
    current = tmp_path / "current.json"
    monkeypatch.setattr(release, "GENERATIONS_DIR", gen_dir)
    monkeypatch.setattr(release, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(release, "CURRENT_PATH", current)
    gen_dir.mkdir()
    runs_dir.mkdir()
    return tmp_path


# ---------------------------------------------------------------- rel_to_root


class TestRelToRoot:
    def test_in_repo_path_becomes_relative_with_forward_slashes(self):
        got = config.rel_to_root(config.ROOT / "data" / "arkrag" / "current.json")
        assert got == "data/arkrag/current.json"
        assert "\\" not in got, "相对形式必须用正斜杠，反斜杠进 git 后在 POSIX 上是个普通字符"

    def test_out_of_repo_path_is_returned_unchanged(self):
        """仓库外原样返回。

        不写 `../../..`：那种字符串看着像相对路径，实际把"站在哪个目录"重新塞回
        语义里，换个 caller 就解析到别处——比绝对路径更糟。
        """
        outside = config.ROOT.parent / "somewhere-else" / "x.json"
        got = config.rel_to_root(outside)
        assert got == str(outside)
        assert ".." not in got

    def test_case_differing_in_repo_path_is_still_recognized(self):
        """Windows 上 `E:\\图rag` 与 `e:\\图rag` 是同一路径但字符串不等。

        判定必须 casefold；否则 `DATA` 写法的路径会被当成"仓库外"，进而被原样
        写成绝对路径——正是这个缺陷本身。
        """
        shouted = config.ROOT / "DATA" / "arkrag"
        got = config.rel_to_root(shouted)
        assert got == "data/arkrag", f"大小写不同的仓库内路径没被识别：{got!r}"

    def test_result_never_contains_parent_escape(self):
        """仓库内路径的相对结果里不许出现 `..`。"""
        probes = [
            config.ROOT,
            config.ROOT / "data",
            config.ROOT / "data" / "arkrag" / "generations",
            config.ROOT / "DATA" / "arkrag" / "NOPE_NOT_EXIST",
            config.ROOT / "data" / ".." / "data" / "arkrag",
        ]
        for probe in probes:
            got = config.rel_to_root(probe)
            assert ".." not in got, f"{probe} 的相对结果里带了父级逃逸：{got!r}"

    def test_sibling_dir_sharing_name_prefix_is_not_inside(self):
        """`E:\\图rag2\\x` 不是仓库内路径。

        按字符串前缀判定的实现会把它切成 `2/x` 这种错位的相对路径——所以判定
        必须按路径分量，边界要落在分隔符上。
        """
        sibling = Path(str(config.ROOT) + "2") / "x"
        got = config.rel_to_root(sibling)
        assert got == str(sibling)

    def test_dotdot_segments_are_resolved_before_comparison(self):
        got = config.rel_to_root(config.ROOT / "data" / ".." / "data" / "arkrag")
        assert got == "data/arkrag"


# ---------------------------------------------------------------- 发布指针


class TestPointerCarriesNoMachinePath:
    def test_publish_pointer_has_no_path_key(self, sandbox):
        """指针不存能推导的东西：`Generation(gid).dir` 就是那个目录。"""
        gen = _build_generation("g-20260101T000000Z-000091")
        pointer = release.publish(gen)
        assert "path" not in pointer
        # 落盘的那一份同样不许有（只看返回值会漏掉"返回值干净、写盘脏"的形态）
        on_disk = json.loads(release.CURRENT_PATH.read_text("utf-8"))
        assert "path" not in on_disk

    def test_rollback_pointer_has_no_path_key(self, sandbox):
        first = _build_generation("g-20260101T000000Z-000092")
        release.publish(first)
        release.publish(_build_generation("g-20260102T000000Z-000093"))
        pointer = release.rollback(first.generation_id)
        assert "path" not in pointer
        on_disk = json.loads(release.CURRENT_PATH.read_text("utf-8"))
        assert "path" not in on_disk

    def test_publish_and_rollback_pointers_are_isomorphic(self, sandbox):
        """两个指针的键集必须相同——这是"同构"的机检，不能只靠注释。

        `rolled_back` 是回滚独有的**语义**标记（它记录"这一代是被回滚切上来的"），
        不是身份字段，所以比较时排除。排除项写在这里而不是放宽断言：一旦 publish
        多了或少了一个身份字段，这条会红。
        """
        first = _build_generation("g-20260101T000000Z-000094")
        published = release.publish(first)
        release.publish(_build_generation("g-20260102T000000Z-000095"))
        rolled = release.rollback(first.generation_id)

        extra = {"rolled_back"}
        assert set(published) - extra == set(rolled) - extra, (
            f"publish 与 rollback 指针键集不同构："
            f"仅在 publish {set(published) - set(rolled)}，"
            f"仅在 rollback {set(rolled) - set(published)}"
        )

    def test_pointer_records_identity_fields_release_cli_prints(self, sandbox):
        """删掉 `path` 不能顺手删掉身份字段。

        `release_cli current` 打印的正是这几个键：少了任何一个，事后就查不出
        线上跑的是哪份图、哪份索引、哪份门禁报告。

        `previous_generation` 单独看：第一代发布时它**应当**是空串（没有前任），
        所以对它只要求"键在"，不要求非空；第二代发布后它才有值。
        """
        first = _build_generation("g-20260101T000000Z-000096")
        first_pointer = release.publish(first)
        assert "previous_generation" in first_pointer
        assert first_pointer["previous_generation"] == ""

        gen = _build_generation("g-20260102T000000Z-000097")
        pointer = release.publish(gen)
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
            assert pointer.get(key), f"发布指针缺身份字段 {key}"
        assert pointer["previous_generation"] == first.generation_id


# ---------------------------------------------------------------- 历史指针容忍


class TestReadCurrentToleratesLegacyPointer:
    def test_legacy_pointer_with_path_does_not_raise(self, sandbox):
        """历史指针里带绝对 `path`：当它不存在，不许因为多一个字段就报错。"""
        legacy = {
            "generation_id": "g-20260101T000000Z-aaaaaa",
            "path": "E:\\图rag\\data\\arkrag\\generations\\g-20260101T000000Z-aaaaaa",
            "status": release.PUBLISHED,
            "published_at": "2026-01-01T00:00:00Z",
        }
        release.CURRENT_PATH.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
        cur = release.read_current()
        assert cur is not None
        assert "path" not in cur
        assert cur["generation_id"] == "g-20260101T000000Z-aaaaaa"

    def test_unknown_fields_other_than_path_are_preserved(self, sandbox):
        """只针对 `path` 特判。别的未知字段可能是未来版本加的，不许吃掉。"""
        future = {
            "generation_id": "g-20260101T000000Z-bbbbbb",
            "path": "E:\\图rag\\data\\arkrag\\generations\\g-20260101T000000Z-bbbbbb",
            "status": release.PUBLISHED,
            "future_field": {"added_by": "later version", "n": 7},
            "another_one": [1, 2, 3],
        }
        release.CURRENT_PATH.write_text(json.dumps(future, ensure_ascii=False), encoding="utf-8")
        cur = release.read_current()
        assert cur["future_field"] == {"added_by": "later version", "n": 7}
        assert cur["another_one"] == [1, 2, 3]
        assert "path" not in cur

    def test_broken_pointer_still_refuses_to_guess(self, sandbox):
        """容忍遗留 `path` 不等于把"坏文件"也容忍掉。"""
        release.CURRENT_PATH.write_text("{ 这不是 JSON", encoding="utf-8")
        with pytest.raises(release.ReleaseError):
            release.read_current()


# ---------------------------------------------------------------- 来源证据


class TestSourceReportIsPortable:
    def test_verify_report_paths_are_repo_relative(self):
        """报告里的 `story_root` / `zip_path` 不许是绝对机台路径。

        用仓库内的小目录与一个不存在的 ZIP 来跑：只要报告字段成形，
        不必去哈希 135MB 的真实解压树——那会把这条门变成半分钟。
        """
        report = source.verify(
            story_root=config.ROOT / "scripts",
            zip_path=config.ROOT / "no-such-source.zip",
        )
        for key in ("story_root", "zip_path"):
            value = report[key]
            assert isinstance(value, str) and value
            assert "E:\\" not in value and "E:/" not in value, (
                f"{key} 仍是绝对机台路径：{value!r}"
            )
            assert not Path(value).is_absolute(), f"{key} 不该是绝对路径：{value!r}"
        assert report["story_root"] == "scripts"
        assert report["zip_path"] == "no-such-source.zip"

    def test_default_source_paths_are_repo_relative(self):
        """生产默认值（真实 STORY_ROOT 与仓库根 ZIP）走一遍同样的规则。

        这里只比字符串、不读盘：报告字段的形态与"有没有那份大文件"无关。
        """
        for probe in (config.STORY_ROOT, source.zip_path_from_env()):
            got = config.rel_to_root(probe)
            assert "E:\\" not in got and "E:/" not in got
            assert not Path(got).is_absolute()

    def test_verify_still_reads_from_the_real_absolute_root(self, tmp_path):
        """变的是**写进报告的字符串**，不是读盘逻辑。

        给一个带真实文件的解压树：`tree_files` 必须能数出来，否则就是"为了让
        路径好看把内部逻辑一起改坏了"。路径在仓库外，故报告里保留绝对形式。
        """
        root = tmp_path / "tree" / "story"
        root.mkdir(parents=True)
        (root / "a.json").write_text('{"x": 1}', encoding="utf-8")
        report = source.verify(story_root=root, zip_path=tmp_path / "absent.zip")
        assert report["tree_files"] == 1
        assert report["zip_present"] is False
        assert report["story_root"] == str(root)


# ------------------------------------------------ 入库 JSON 全库扫描（防回归）


#: 冻结基线产物：**允许**含绝对路径，因为它们的价值就是"逐字节不变的证据"。
#: 为了让一条检查变绿去改它们，等于把证据毁掉——所以豁免。但豁免本身由
#: `test_exemptions_are_really_frozen_artifacts` 盯着：名单里塞进一个没被冻结
#: 的文件，测试就红。否则这张名单会变成后门。
FROZEN_EXEMPT = (
    "data/arkrag/corpus_meta.json",
    "data/arkrag/l0_manifest.json",
)

#: Windows 盘符绝对路径、POSIX 绝对路径。刻意不匹配 `C:relative`——
#: 那种形式在 Windows 上另有语义，不属于"机台绝对路径"。
MACHINE_PATH_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|/)")


def _looks_absolute(value: str) -> bool:
    """一个**字符串值**是不是绝对机台路径。

    判据与 `scripts/repair_absolute_paths.py` 严格一致：只认真实存在的形态，
    且不含冒号路径（`C:relative` 在 Windows 上另有语义）。
    """
    if len(value) >= 3 and value[1] == ":" and value[2] in "\\/":
        return value[0].isalpha()
    return value.startswith("/")


def _walk_strings(obj):
    """产出 JSON 文档里所有字符串值的 (路径, 值)。"""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str):
                yield str(key), value
            else:
                for sub in _walk_strings(value):
                    yield sub
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, str):
                yield "[]", item
            else:
                for sub in _walk_strings(item):
                    yield sub


def _load_docs(rel: str) -> list:
    """把一个入库产物读成待扫描的文档列表。

    JSON 读成对象。JSONL 逐行读（一个坏行不该让整条门变哑巴——但它本身是
    另一个问题，这里记成可报告的 `__unparsable__` 字符串以便被看见）。
    """
    text = (config.ROOT / rel).read_text("utf-8")
    if rel.endswith(".jsonl"):
        docs = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError:
                docs.append({"__unparsable__": line[:200]})
        return docs
    return [json.loads(text)]


def _git_tracked_data_files() -> list[str]:
    """git 跟踪的 `data/` 下 JSON/JSONL 文件（相对仓库根，正斜杠）。

    用 `git ls-files` 而不是 `Path.rglob`：只有**入库**的文件才涉及
    "换台机器 clone 下来能不能用"。被 .gitignore 排除的缓存、索引、大图里
    出现本机路径是正常的，不该判红。
    """
    try:
        out = subprocess.run(
            ["git", "ls-files", "data/"],
            cwd=config.ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        pytest.skip("git 不可用，无法判定哪些文件入库")
    if out.returncode != 0:  # pragma: no cover
        pytest.skip(f"git ls-files 失败：{out.stderr.strip()[:200]}")
    return [rel for rel in out.stdout.split() if rel.endswith((".json", ".jsonl"))]


class TestNoMachinePathsInTrackedJson:
    def test_no_absolute_paths_in_tracked_data_json(self):
        """入库的 JSON 里不许出现绝对机台路径（A3 / DoD「部署方式可复现」）。

        这条是 `scripts/repair_absolute_paths.py` 那一轮的回归门：当时实测
        `agent_plan.json` 有 1,186 处、16 个 ledger 文件各 1 处，全是
        `E:\\图rag\\...`。产出点已改用 `config.rel_to_root()`，但若没人盯着，
        下次谁写一句 `str(path)` 就全回来了——而那时 git diff 里只看到一串
        JSON，看不出问题。

        **只判字符串值，不在 JSON 文本上跑正则**：第一版就是拿正则扫原文，
        结果把门禁报告里 `runs/`、`generations/<gid>/` 这类**相对**路径里的斜杠
        误判成绝对路径（实测 8 个文件报假阳性）。解析后再判，才分得清
        "值以 `/` 开头"与"值中间有个斜杠"——误报会让人学会忽略这条门。
        """
        offenders: list[str] = []
        for rel in _git_tracked_data_files():
            if rel in FROZEN_EXEMPT:
                continue
            for doc in _load_docs(rel):
                for key, value in _walk_strings(doc):
                    if _looks_absolute(value):
                        offenders.append(f"{rel} [{key}]: {value!r}")
        assert not offenders, (
            "入库 JSON 里仍有绝对机台路径（换机器 clone 后指向空地）：\n  "
            + "\n  ".join(offenders[:10])
            + "\n修法：产出点改用 config.rel_to_root()，历史文件跑 "
            "`py -3 scripts/repair_absolute_paths.py --apply`。"
        )

    def test_exemptions_are_really_frozen_artifacts(self):
        """豁免名单里每个文件都必须在 legacy-baseline 的登记清单里。

        否则这张名单就是后门：谁想绕过上面的检查，把自己的文件加进
        `FROZEN_EXEMPT` 就行。冻结是**唯一**的豁免理由。
        """
        from arkrag import legacy

        baseline = json.loads(legacy.BASELINE_MANIFEST.read_text("utf-8"))
        frozen = {item["path"] for item in baseline.get("artifacts", [])}
        not_frozen = [rel for rel in FROZEN_EXEMPT if rel not in frozen]
        assert not not_frozen, f"这些文件不在冻结基线清单里，不构成豁免理由：{not_frozen}"

    def test_scan_actually_finds_files(self):
        """扫描本身必须真的扫到了文件，否则"零命中"可能只是扫了个空集。"""
        files = _git_tracked_data_files()
        assert len(files) >= 20, f"只扫到 {len(files)} 个入库 JSON，判据可能坏了"

    def test_detector_recognizes_machine_paths_and_ignores_relative_ones(self):
        """判据本身的正反例：绝对路径必须被抓到，相对路径必须放过。

        没有这条，"全库零命中"无法区分「真干净」与「判据根本不工作」。
        """
        assert _looks_absolute("E:\\图rag\\data\\x.json")
        assert _looks_absolute("E:/图rag/data/x.json")
        assert _looks_absolute("/usr/local/x.json")
        assert not _looks_absolute("data/arkrag/x.json")
        assert not _looks_absolute("runs/r-20261030T114331Z-f487c2")
        assert not _looks_absolute("generations/g-20261030T114823Z-606ea9/reports/gates.json")
        assert not _looks_absolute("C:relative-only-on-windows")
        assert not _looks_absolute("")
        assert not _looks_absolute("每块最多 22 块 / 每场景最多 1 块")
