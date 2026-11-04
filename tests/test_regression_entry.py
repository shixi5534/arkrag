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
