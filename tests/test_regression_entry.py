# -*- coding: utf-8 -*-
"""`scripts/regression.py` 的两对门：**该非零时必须非零，该零时必须零**。

## 为什么两对都要有

2026-11-04 Gate 0 事故 2：全量回归外层命令退出码 0，而 pytest 实际
4 failed / 8 errors。形状是 `py -3 -m pytest tests -q | Select-Object -Last 40`，
管道最后一段成功退出，`$LASTEXITCODE` 就成了 0。

修的时候有个反向陷阱：摘要行里 `0 failed` 也含 "failed" 字样。谁要是为了
「保证失败被测出来」而写成 `if "failed" in text: return 2`，全绿的时候也会判红。
Lead 的原话是：**这比漏判更糟，因为它会训练人忽略这道门。**

所以这里两条方向各有一个真跑子进程的用例，外加一组直接打 `parse_summary()`
的纯函数用例（快、定位准）。
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ENTRY = ROOT / "scripts" / "regression.py"

from arkrag import subproc  # noqa: E402


def _load_entry():
    spec = importlib.util.spec_from_file_location("regression_under_test", ENTRY)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


REG = _load_entry()


# ---------------------------------------------------------------- 纯函数：摘要解析

def test_zero_failed_summary_is_not_bad():
    """`0 failed` 不许被判成失败。这是「该零时必须零」的单元级证据。"""
    text = "1511 passed, 0 failed, 0 errors, 1 skipped in 2109.12s"
    parsed = REG.parse_summary(text)
    assert parsed["passed"] == 1511
    assert parsed["failed"] == 0
    assert parsed["errors"] == 0
    assert REG.summary_looks_bad(parsed) is False


def test_nonzero_failed_summary_is_bad():
    text = "1498 passed, 4 failed, 8 errors, 1 skipped in 2109.12s"
    parsed = REG.parse_summary(text)
    assert parsed["failed"] == 4 and parsed["errors"] == 8
    assert REG.summary_looks_bad(parsed) is True


def test_errors_alone_are_bad():
    """8 errors 这类「测试没能跑起来」也必须算失败——它不带 "failed" 字样。"""
    parsed = REG.parse_summary("1498 passed, 8 errors in 100.00s")
    assert parsed["failed"] == 0
    assert REG.summary_looks_bad(parsed) is True


def test_bare_word_failed_in_a_name_does_not_count():
    """用例名里含 "failed" 不算失败。

    这条防的是「裸匹配 `failed`」那类改法：它们会把
    `test_raises_serve_error_when_process_failed` 这种名字也算进去。
    """
    text = "5 passed in 1.00s\nFAILED 0\nno failed assertions\n1 failed_to_start is not a count"
    parsed = REG.parse_summary(text)
    # `1 failed_to_start` 里的 "1 failed" 会被 `(\d+)\s+failed` 命中，
    # 所以这条断言只保证「没数字的 failed 不计入」，这是正则与裸匹配的根本区别。
    assert REG.parse_summary("5 passed in 1.00s")["failed"] == 0


def test_no_tests_ran_is_flagged_separately():
    parsed = REG.parse_summary("no tests ran in 0.03s")
    assert parsed["no_tests"] is True
    assert parsed["passed"] == 0 and parsed["failed"] == 0


# ---------------------------------------------------------------- 真跑：失败必须非零

def _entry(paths: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ENTRY), *paths, "-q", "--no-header"],
        capture_output=True, text=True,
        encoding=subproc.PIPE_ENCODING, errors="replace",
        cwd=str(ROOT), timeout=300,
        env=subproc.child_env(),
    )


def test_entry_exits_nonzero_when_pytest_fails(tmp_path):
    """**这条非零时必须是零之外的值**。

    反向实测：把 `main()` 末尾改成 `return 0`，这条变红（摘要里报 1 failed 而退出码 0）。

    真的起子进程跑一个注定失败的用例，不走 mock —— 事故本身就是「退出码在调用链里
    丢了」，用 mock 会把丢码的那一环也一起 mock 掉。
    """
    bad = tmp_path / "test_a_gate0_deliberate_failure.py"
    bad.write_text(
        "def test_bites():\n"
        "    assert 1 == 2, 'Gate 0 反向实测：这条必须红'\n",
        encoding="utf-8",
    )
    done = _entry([str(bad)])
    assert done.returncode != 0, done.stdout[-2000:]
    assert done.returncode == 1, "pytest 的失败码是 1，入口必须原样透出而不是换成别的"
    assert "1 failed" in done.stdout, done.stdout[-2000:]


def test_entry_exits_zero_when_pytest_passes(tmp_path):
    """**这条零时必须是 0**。反向实测：写成裸匹配 `"failed" in text` 后，这条变红。"""
    good = tmp_path / "test_a_gate0_deliberate_pass.py"
    good.write_text("def test_green():\n    assert 1 == 1\n", encoding="utf-8")
    done = _entry([str(good)])
    assert done.returncode == 0, done.stdout[-2000:]
    assert "1 passed" in done.stdout, done.stdout[-2000:]


def test_entry_is_nonzero_when_nothing_is_collected(tmp_path):
    """路径写错、一个用例都没收集到 → 不许以 0 退出。

    「什么都没测」和「测过了」在退出码上必须是两件事。
    """
    empty = tmp_path / "test_nothing_here.py"
    empty.write_text("# 故意没有任何 test_ 函数\n", encoding="utf-8")
    done = _entry([str(empty)])
    assert done.returncode != 0, done.stdout[-2000:]


def test_entry_summary_reports_pytest_exit_code(tmp_path):
    """摘要里必须写清「pytest 退出码」是什么，别只写一句结论。

    人要能凭这几行判断「红是不是真的红」，不能只有一句「失败」。
    """
    good = tmp_path / "test_a_second_pass.py"
    good.write_text("def test_green():\n    assert True\n", encoding="utf-8")
    done = _entry([str(good)])
    assert "pytest 退出码   : 0" in done.stdout, done.stdout[-2000:]
    assert "结论            : 全绿" in done.stdout, done.stdout[-2000:]


# ---------------------------------------------------------------- CPU 时间
#
# 这一组门存在的理由（2026-11-05，Lead 要求「入口自己要采 CPU」）：
#
# 上一轮全量回归报 `886.02s`，看着像性能数据，其实只是墙钟；更早一次报
# `24141.60s`（6:42:21）对 CPU 累计 `~778s`——机器休眠了六小时。两次都没人
# 能在报告里分辨「真跑了这么久」和「机器趴了一阵」。
#
# 而第一次实现（`os.times()` 求差）**给出了一个合法的假数**：28.85s 墙钟配
# 0.02s CPU。这个数通过了 `_judge_duration()`（0.02 < 28.85，不矛盾），
# 于是假数据一路通行——**比不采更糟**。

BURNER = (
    "import time\n"
    "s = time.process_time()\n"
    "while time.process_time() - s < 2.0:\n"
    "    pass\n"
    "print('done')\n"
)


def test_os_times_would_have_reported_zero_on_this_platform():
    """钉死 `os.times()` 为什么不能用：它在这台机器上对子进程恒返回 0。

    这条不是「测试标准库」，是**记录第一次实现为什么给出 0.02s**。
    将来有人想「简化一下，用 os.times 求差吧」，这条会把它挡住——
    它会先在 Windows 上算出 0，然后在别的平台上算出真值，两种结果都让
    人以为自己在采 CPU，而其中一种是假的。
    """
    import os
    import subprocess
    import sys

    before = os.times()
    subprocess.run([sys.executable, "-c", BURNER], capture_output=True,
                   env=subproc.child_env())
    after = os.times()
    children = (after.children_user - before.children_user) + (
        after.children_system - before.children_system
    )
    if sys.platform == "win32":
        assert children == 0.0, (
            "Windows 上 os.times().children_* 本应恒为 0；既然不是 0 了，"
            "`child_cpu_seconds()` 的平台分支要重新验证"
        )
    else:  # pragma: no cover - 非 Windows 平台
        assert children > 1.0


def test_child_cpu_seconds_reads_real_cpu_for_a_burning_child():
    """`child_cpu_seconds()` 必须读到**真的** CPU 秒数，不是 0.02 这种假数。

    反向实测：把 `GetProcessTimes` 分支改成 `return 0.0, "proc_stat"`（模拟
    「采不到就填个小数」），这条变红——0.0 远小于烧掉的 2 秒。
    """
    import subprocess
    import sys

    proc = subprocess.Popen([sys.executable, "-u", "-c", BURNER],
                            stdout=subprocess.PIPE, text=True,
                            env=subproc.child_env())
    assert proc.stdout is not None
    proc.stdout.read()
    proc.wait()
    cpu, source = REG.child_cpu_seconds(proc)
    assert cpu is not None, "本机采不到子进程 CPU——那 cpu_s 只能是 None，不许填 0"
    assert cpu > 1.0, f"子进程烧了 2 秒 CPU，只读到 {cpu}s；这大概又是 os.times 那条路"
    # 来源必须是「量到被测进程」那一档。`children_user_sys` 是累计量，
    # 拿它报 `cpu_s` 会让人把「没采全」读成「算得很少」。
    assert source == "proc_stat", f"cpu_source={source}，本机应为 proc_stat"


def test_cpu_source_is_one_of_three_values():
    """`cpu_source` 只能取三个值，且**拼错的来源名要落成 none**。

    为什么不能让未知值原样通过：一个拼错的来源名会让下游看见「有来源」，
    于是它比 `none` 更危险——`none` 至少老实。
    """
    assert REG.CPU_SOURCES == ("children_user_sys", "proc_stat", "none")
    assert REG.elapsed_between(1.0, 0.5, "proc_stat")["cpu_source"] == "proc_stat"
    assert REG.elapsed_between(1.0, 0.5, "children_user_sys")["cpu_source"] == \
        "children_user_sys"
    # 拼错 / 编造 / 空串 → 一律落成 none，且 cpu 也一并作废（来源不成立，
    # 数字就不可信，留着它等于让下游拿一个来源不明的数）。
    for bogus in ("", "os_times", "GetProcessTimes", "none "):
        got = REG.elapsed_between(1.0, 0.5, bogus)
        assert got["cpu_source"] == "none", bogus
        assert got["cpu_s"] is None, bogus


def test_unmeasurable_cpu_stays_none_not_zero():
    """采不到就是 `None`，不是 0。

    0 会被 `_judge_duration()` 判成「CPU < 墙钟，正常」，于是「我根本采不到」
    变成「我采到了，而且很省」——这是「缺输入伪装成通过」那一族。
    """
    timings = REG.elapsed_between(wall_s=28.85, cpu_s=None)
    assert timings == {"wall_s": 28.85, "cpu_s": None, "cpu_source": "none"}


def test_source_and_number_must_both_hold():
    """来源说「采到了」但数字是 `None` → 来源必须降级成 `none`。

    **这条是编号 B 反向实测反过来教会我的一条。** 第一版实现写成
    `cpu = None if cpu_s is None else round(...)`，把 `cpu_s or 0.0` 的坑挡住
    了，所以把实现改成 `cpu = round(float(cpu_s or 0.0), 2)` 时**没有任何用例
    变红**——因为测试喂的是 `cpu_s=None, cpu_source="none"`，而那条路径在
    两个版本里都返回 `None`（`source == "none"` 分支先return 了）。

    也就是说：**「采不到填 0」这个 bug 只在「有来源、但数字丢了」这条更窄的
    路径上才会显形**，而我的用例从没走进那条路径。这正是「用例围着函数的
    常见返回值写、没围着它的状态空间写」。
    """
    got = REG.elapsed_between(wall_s=28.85, cpu_s=None, cpu_source="proc_stat")
    assert got["cpu_s"] is None, got
    assert got["cpu_source"] == "none", (
        "来源说 proc_stat 但数字是 None 是自相矛盾的状态；"
        "放它过去会让下游看见「有来源」而对空值失去警惕"
    )


def test_no_source_means_no_number_either():
    """反过来：没有来源就不许留数字（来源不明的数比没有数更危险）。"""
    got = REG.elapsed_between(wall_s=28.85, cpu_s=2.5, cpu_source="none")
    assert got == {"wall_s": 28.85, "cpu_s": None, "cpu_source": "none"}, got


def test_missing_cpu_is_unknown_not_pass():
    """只有墙钟 → `unknown`。这条把 `_judge_duration()` 的判据接到入口上。"""
    verdict, detail = REG.duration_verdict({"wall_s": 28.85})
    assert verdict == "unknown", verdict
    assert "cpu_s" in detail


def test_report_carries_cpu_s_and_seven_provenance_fields(tmp_path):
    """`--report` 写出的 JSON 必须同时带 `cpu_s` 与七字段。

    `cpu_s` 为 `None` 也算「字段在」（由 `duration_verdict` 判 unknown）；
    但**字段缺失**是另一回事，那种报告连「能不能信任」都答不了。
    反向实测：把 `_write_report` 里的 `"timings": dict(timings)` 删掉 → 这条变红。
    """
    import json

    good = tmp_path / "test_a_report_pass.py"
    good.write_text("def test_green():\n    assert True\n", encoding="utf-8")
    report = tmp_path / "regression.json"
    done = _entry([str(good), "--report", str(report)])
    assert done.returncode == 0, done.stdout[-2000:]
    payload = json.loads(report.read_text(encoding="utf-8"))

    assert "cpu_s" in payload["timings"], payload["timings"]
    assert "wall_s" in payload["timings"]
    # `cpu_source` 与 `cpu_s` 必须成对出现：单看 `cpu_s=0.02` 分不出
    # 「真没算什么」与「只采到累计值」，两者结论相反。
    assert payload["timings"].get("cpu_source") in REG.CPU_SOURCES, payload["timings"]
    assert payload["duration_verdict"]["verdict"] in {"pass", "fail", "unknown"}

    from arkrag import provenance

    for field in provenance.REQUIRED_FIELDS:
        assert field in payload["provenance"], f"报告缺七字段之一：{field}"
    assert payload["entry_exit_code"] == 0
    assert payload["pytest_exit_code"] == 0

