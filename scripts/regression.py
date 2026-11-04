#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""回归入口：跑 pytest、逐行回显、**以 pytest 的退出码作为自己的退出码**。

## 为什么需要这个文件（2026-11-04 Gate 0 事故 2）

上一次全量回归，外层命令退出码是 0，而 pytest 实际上 4 failed / 8 errors。
形状是这样的：

    py -3 -m pytest tests -q | Select-Object -Last 40

管道右侧（`Select-Object`）成功退出，`$LASTEXITCODE` 就变成了它的退出码 0，
pytest 的失败被「最后一个命令成功」掩盖掉了。PowerShell 本身没毛病——它如实
报告了管道最后一段的退出码，是我们的调用链形状不对。

所以这里把「跑、收集、回显、退出」四件事收在一个进程里，退出码只有一个来源：

    py -3 scripts/regression.py

## 判据（两边都要有测试）

- pytest 说失败 → 本脚本非零。
- pytest 说全绿 → 本脚本必须**零**。

第二条同样重要。摘要行里 `0 failed` 也含 "failed" 字样，裸匹配 `"failed" in text`
会把成功判成失败——这种门比漏判更糟，因为它会训练人忽略这道门。所以下面只用
`(\\d+) failed` / `(\\d+) error(s)?` 且**捕获组整数 > 0** 才算失败。

## 为什么逐行 tee

全量 35 分钟。如果只在结束时打印摘要，中间红在哪一行没人看得见，人就只能
「等它跑完」而不是「边跑边判断要不要停」。逐行 tee 让每一次 stdout 刷新都可见。

## 用法

    py -3 scripts/regression.py                    # 全量 tests
    py -3 scripts/regression.py tests/test_k2_gray.py -q
    py -3 scripts/regression.py -- -q tests        # `--` 之后原样透传给 pytest

退出码：pytest 的退出码（0 全绿 / 1 有失败 / 2 被中断 / 3 内部错误 / 4 用法错误 /
5 没有收集到用例）。另外：本脚本自己起不来 pytest 时返回 4（与 pytest 的用法错误
区分不了，所以摘要里会写明是「入口自己失败」）。
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from arkrag import subproc  # noqa: E402

# pytest 的摘要行形如：`1511 passed, 4 failed, 8 errors, 1 skipped in 2109.12s`。
# 只认「数字 + 空格 + 词」并且**数字 > 0**——`0 failed` 不算失败。
FAILED_RE = re.compile(r"(\d+)\s+failed")
ERROR_RE = re.compile(r"(\d+)\s+error(?:s)?\b")
PASSED_RE = re.compile(r"(\d+)\s+passed")
COLLECTED_RE = re.compile(r"(\d+)\s+tests?\s+collected")
NO_TESTS_RE = re.compile(r"no\s+tests?\s+ran")

# 入口自己的失败码。pytest 用 1–5，这里用 90 起，避免和「pytest 失败」混淆。
EXIT_ENTRY_FAILED = 90


def parse_summary(text: str) -> dict:
    """从 pytest 输出里读出各计数。**不做裸字符串匹配**。

    返回 `{"passed", "failed", "errors", "collected", "no_tests", "summary_line"}`。
    计数缺失时给 0 —— 但 `no_tests` 单独标出来，因为「一个用例都没跑」和
    「全绿」看起来都是 0 failed，含义却完全不同。
    """
    failed = max((int(m.group(1)) for m in FAILED_RE.finditer(text)), default=0)
    errors = max((int(m.group(1)) for m in ERROR_RE.finditer(text)), default=0)
    passed = max((int(m.group(1)) for m in PASSED_RE.finditer(text)), default=0)
    collected = max((int(m.group(1)) for m in COLLECTED_RE.finditer(text)), default=0)
    no_tests = bool(NO_TESTS_RE.search(text))
    summary_line = ""
    for line in reversed(text.splitlines()):
        if PASSED_RE.search(line) or FAILED_RE.search(line) or ERROR_RE.search(line):
            summary_line = line.strip()
            break
    return {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "collected": collected,
        "no_tests": no_tests,
        "summary_line": summary_line,
    }


def summary_looks_bad(parsed: dict) -> bool:
    """摘要是否说明「有失败」。两个条件缺一不可：计数 > 0。

    这条函数就是「`0 failed` 不许判红」的唯一实现处，测试直接打它。
    """
    return parsed["failed"] > 0 or parsed["errors"] > 0


def _tee(stream, collected: list[str], echo: bool) -> None:
    for raw in iter(stream.readline, ""):
        collected.append(raw)
        if echo:
            sys.stdout.write(raw)
            sys.stdout.flush()


def run(args: list[str], echo: bool = True) -> tuple[int, dict, list[str]]:
    """跑 pytest，逐行回显，返回 (退出码, 解析结果, 完整输出行)。"""
    cmd = [sys.executable, "-m", "pytest", *args]
    if echo:
        sys.stdout.write("$ " + " ".join(cmd) + "\n")
        sys.stdout.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding=subproc.PIPE_ENCODING,
        errors="replace",
        bufsize=1,
        # 子进程按 UTF-8 写管道，父进程按 UTF-8 读——两边必须约定成同一个编码，
        # 否则 pytest 的中文摘要（失败用例的断言消息）会变成 \ufffd，人就看不到红在哪。
        env=subproc.child_env(),
    )
    lines: list[str] = []
    assert proc.stdout is not None
    _tee(proc.stdout, lines, echo)
    returncode = proc.wait()
    return returncode, parse_summary("".join(lines)), lines


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding=subproc.PIPE_ENCODING)

    parser = argparse.ArgumentParser(
        description="回归入口：跑 pytest 并以它的退出码作为本脚本的退出码"
    )
    parser.add_argument("paths", nargs="*", default=[], help="pytest 目标（默认 tests/）")
    parser.add_argument("--quiet", action="store_true", help="不回显 pytest 的逐行输出")
    args, passthrough = parser.parse_known_args(argv)

    targets = args.paths or ["tests"]
    extra = [a for a in passthrough if a != "--"]
    pytest_args = [*targets, *extra]

    try:
        returncode, parsed, lines = run(pytest_args, echo=not args.quiet)
    except OSError as exc:
        sys.stdout.write("[ENTRY FAILED] 起不来 pytest：%s: %s\n" % (type(exc).__name__, exc))
        return EXIT_ENTRY_FAILED

    sys.stdout.write("\n" + "=" * 72 + "\n")
    sys.stdout.write("回归摘要\n")
    sys.stdout.write("  命令            : %s\n" % " ".join(["py", "-3", "-m", "pytest", *pytest_args]))
    sys.stdout.write("  pytest 退出码   : %d\n" % returncode)
    sys.stdout.write("  摘要行          : %s\n" % (parsed["summary_line"] or "（没有可识别的摘要行）"))
    sys.stdout.write("  passed / failed / errors : %d / %d / %d\n"
                     % (parsed["passed"], parsed["failed"], parsed["errors"]))
    if parsed["no_tests"]:
        sys.stdout.write("  注意            : pytest 报告「没有跑任何用例」——"
                         "这不算通过，请先确认目标路径对不对。\n")

    # 退出码只有一个来源：pytest 的。摘要只用于**解释**这个码，不参与决定它。
    # 唯一例外是「pytest 说成功但一个用例都没跑」——那说明路径写错了，
    # 让它以成功退出等于把「什么都没测」当成「测过了」。
    if returncode == 0 and parsed["no_tests"]:
        sys.stdout.write("  结论            : 未跑任何用例，判入口失败（%d）\n" % EXIT_ENTRY_FAILED)
        return EXIT_ENTRY_FAILED

    if returncode != 0:
        sys.stdout.write("  结论            : 失败（pytest 退出码 %d）\n" % returncode)
    elif summary_looks_bad(parsed):
        # 退出码 0 但摘要说失败：两者矛盾时不许沉默。宁可让门红，因为它意味着
        # 有人把 pytest 的退出码弄丢了——这正是 Gate 0 事故 2。
        sys.stdout.write("  结论            : 矛盾——pytest 退出码 0 但摘要报失败，判入口失败（%d）\n"
                         % EXIT_ENTRY_FAILED)
        return EXIT_ENTRY_FAILED
    else:
        sys.stdout.write("  结论            : 全绿\n")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
