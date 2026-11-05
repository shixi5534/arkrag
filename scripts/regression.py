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
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from arkrag import provenance, subproc  # noqa: E402

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


# ---------------------------------------------------------------- 耗时：墙钟 + CPU
#
# 为什么必须两个都采（2026-11-05，Lead 抓的第二次）：
#
# 上一轮全量回归报 `886.02s (14:46)`。但这个数**只是墙钟**——`scripts/regression.py`
# 里 `process_time|cpu|perf_counter` 零命中。而更早一次报的是 `24141.60s`（6:42:21）
# 对进程 CPU 累计 `~778s`（13 分钟）：机器中途休眠了六小时。
#
# 只有墙钟的时候，这两种东西在报告里长得一模一样：
#   * 「这批用例真的跑了 6 小时 42 分」
#   * 「用例跑了 13 分钟，机器趴了 6 小时 29 分」
# 后果是任何「优化了多少秒」的结论都可能整段作废，而且没人看得出来。
#
# CPU 时间不会被休眠污染（睡眠期间没有 CPU 消耗），所以它是墙钟的**校验量**，
# 不是补充量。判据用 `provenance._judge_duration()`——那边已经有一条现成的门，
# 这里不复制一份逻辑，否则两处判据迟早分叉。

CPU_TRUST_RATIO_NOTE = (
    "CPU 时间远小于墙钟时，墙钟里混进了非计算时间（机器休眠 / 等盘 / 等网络），"
    "这种墙钟不能进性能报告"
)

#: `cpu_source` 的取值域，只有三个。**没有第四个兜底值**——采不到就是 `none`。
#:
#: 为什么要这个字段（Lead 2026-11-05 要求）：`cpu_s = 0.02` 这种数有两个完全
#: 不同的成因，而报告里它们长得一模一样：
#:   * 「这一轮真的几乎没算什么」      → 该信任它，可以据此说性能变好了
#:   * 「只采到了已回收子进程的累计值」→ 该丢掉它，它根本没覆盖被测进程
#: 单看 `cpu_s` 分不出是哪一种，于是「CPU 很低」会被读成「性能很好」。
#: `cpu_source` 就是让这两个成因在字段层面就不可混淆。
CPU_SOURCES = ("children_user_sys", "proc_stat", "none")


def child_cpu_seconds(proc) -> tuple[float | None, str]:
    """读**子进程**的 CPU 时间（用户 + 内核），返回 `(秒数, cpu_source)`。

    取不到返回 `(None, "none")`，**不返回 0**。

    ## cpu_source 的三个取值
    - `proc_stat`：直接读**子进程自身**的 CPU 计数器
      （Windows `GetProcessTimes`，POSIX `/proc/<pid>/stat`）。这是最可信的一档，
      因为它量的就是被测进程。
    - `children_user_sys`：读「**已回收子进程的累计** CPU」（POSIX
      `os.times()` 的 children 字段 / `resource.RUSAGE_CHILDREN`）。它是累计量、
      涵盖本进程所有子进程，不是单指被测进程。
    - `none`：采不到。

    ## 为什么这在本机是 `proc_stat` 而不是 `children_user_sys`
    2026-11-05 实测（`tmp_b_cpu_probe.py`，子进程烧 2 秒 CPU）：

        os.times() user Δ           = 0.0000
        os.times() children_user Δ  = 0.0000
        os.times() children_system Δ= 0.0000

    **Windows 上 `os.times()` 的四个字段全是 0**，包括那两个名字看起来专门为
    子进程准备的 `children_*`。用它算出来的 `cpu_s = 0.02` 是个**看起来合法的
    小数字**——比不采更糟，因为 `_judge_duration()` 只会判「CPU > 墙钟」矛盾，
    而 0.02 < 28.85 完全自洽，于是假数据一路通行。

    `GetProcessTimes` 在同一个实验里读回 **3.0625s**。它必须在 `proc.wait()`
    **之后**、句柄关闭**之前**读：运行中读到的是 `0.0`，因为子进程还没开始消耗。
    """
    if sys.platform == "win32":  # pragma: no cover - 平台分支
        try:
            import ctypes
            from ctypes import wintypes
        except ImportError:  # pragma: no cover
            return None, "none"
        handle = getattr(proc, "_handle", None)
        if not handle:
            return None, "none"
        creation, exit_, kernel, user = (wintypes.FILETIME() for _ in range(4))
        try:
            ok = ctypes.windll.kernel32.GetProcessTimes(  # type: ignore[attr-defined]
                wintypes.HANDLE(int(handle)),
                ctypes.byref(creation), ctypes.byref(exit_),
                ctypes.byref(kernel), ctypes.byref(user),
            )
        except (OSError, ValueError, TypeError):  # pragma: no cover
            return None, "none"
        if not ok:  # pragma: no cover
            return None, "none"

        def _to_s(ft) -> float:
            return ((ft.dwHighDateTime << 32) | ft.dwLowDateTime) / 1e7

        return round(_to_s(kernel) + _to_s(user), 4), "proc_stat"

    try:  # pragma: no cover - 平台分支
        import resource

        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        return round(float(usage.ru_utime + usage.ru_stime), 4), "children_user_sys"
    except (ImportError, OSError):  # pragma: no cover
        return None, "none"


def elapsed_between(wall_s: float, cpu_s: float | None, cpu_source: str = "none") -> dict:
    """拼出 `{"wall_s", "cpu_s", "cpu_source"}`。`cpu_s` 取不到就是 `None`，不填 0。

    纯函数，用例可以喂合成值。填 0 会让 `_judge_duration()` 判 PASS（0 < 墙钟），
    而真相是「根本采不到」——这正是「缺输入伪装成通过」那一族。

    `cpu_source` 不在 `CPU_SOURCES` 里时**直接判成 `none`**，不原样写进报告：
    一个拼错的来源名会让下游误以为「有来源」，比 `none` 更糟。

    同理，**来源说「采到了」但数字是 `None` 时，来源也要降级成 `none`**——
    「`proc_stat` 但没数」是个自相矛盾的状态，放它过去会让下游看见
    「有来源」而对一个空值失去警惕。降级之后 `cpu_s` 仍是 `None`，
    不会被填成 0（0 会被 `_judge_duration()` 读成「CPU 很低，性能很好」）。
    """
    source = cpu_source if cpu_source in CPU_SOURCES else "none"
    if cpu_s is None:
        # 来源与数字必须同时成立，缺一个就都作废。
        return {"wall_s": round(float(wall_s), 2), "cpu_s": None, "cpu_source": "none"}
    if source == "none":
        # 反过来：没有来源就不许留数字（来源不明的数比没有数更危险）。
        return {"wall_s": round(float(wall_s), 2), "cpu_s": None, "cpu_source": "none"}
    return {
        "wall_s": round(float(wall_s), 2),
        "cpu_s": round(float(cpu_s), 2),
        "cpu_source": source,
    }


def duration_verdict(timings: dict) -> tuple[str, str]:
    """把耗时两件套送进 `provenance._judge_duration()`，返回 (三态, 说明)。"""
    verdict, detail, missing = provenance._judge_duration(dict(timings))
    if missing:
        detail = detail + "（缺 " + "、".join(missing) + "）"
    return verdict, detail



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


def run(args: list[str], echo: bool = True) -> tuple[int, dict, list[str], dict]:
    """跑 pytest，逐行回显，返回 (退出码, 解析结果, 完整输出行, 耗时两件套)。"""
    cmd = [sys.executable, "-m", "pytest", *args]
    if echo:
        sys.stdout.write("$ " + " ".join(cmd) + "\n")
        sys.stdout.flush()
    t0 = time.perf_counter()
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
    wall_s = time.perf_counter() - t0
    # 必须在 wait() 之后、句柄关闭之前读：运行中读到恒为 0.0（见 child_cpu_seconds）。
    cpu_s, cpu_source = child_cpu_seconds(proc)
    timings = elapsed_between(wall_s, cpu_s, cpu_source)
    return returncode, parse_summary("".join(lines)), lines, timings


def _write_report(path: Path, pytest_args: list[str], returncode: int, parsed: dict,
                  timings: dict, verdict: str, detail: str, final_code: int) -> None:
    """把这一轮回归写成一份带七字段的报告：`cpu_s` 从这一刻起是**要求的字段**。

    `provenance.collect()` 负责 `generated_at` / `generation_id` / `git_commit` /
    `code_sha256_16` / `dataset` / `inputs` / `environment`；取不到的留空，
    由 `provenance.verify()` 判 unknown——**不在入口这里「尽力填」**。
    """
    payload = {
        "schema": "ark-regression-report-1",
        "provenance": provenance.collect(
            generation_id=parsed.get("generation_id") or "",
            inputs=[],
        ),
        "command": " ".join(["py", "-3", "-m", "pytest", *pytest_args]),
        "pytest_exit_code": returncode,
        "entry_exit_code": final_code,
        "summary_line": parsed["summary_line"],
        "counts": {
            "passed": parsed["passed"],
            "failed": parsed["failed"],
            "errors": parsed["errors"],
            "collected": parsed["collected"],
            "no_tests": parsed["no_tests"],
        },
        # 两个都写。只写 wall_s 的报告见 §「耗时：墙钟 + CPU」。
        "timings": dict(timings),
        "duration_verdict": {"verdict": verdict, "detail": detail},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding=subproc.PIPE_ENCODING)

    parser = argparse.ArgumentParser(
        description="回归入口：跑 pytest 并以它的退出码作为本脚本的退出码"
    )
    parser.add_argument("paths", nargs="*", default=[], help="pytest 目标（默认 tests/）")
    parser.add_argument("--quiet", action="store_true", help="不回显 pytest 的逐行输出")
    parser.add_argument("--report", metavar="PATH",
                        help="额外写一份带七字段与 cpu_s 的 JSON 报告到这里")
    args, passthrough = parser.parse_known_args(argv)

    targets = args.paths or ["tests"]
    extra = [a for a in passthrough if a != "--"]
    pytest_args = [*targets, *extra]

    try:
        returncode, parsed, lines, timings = run(pytest_args, echo=not args.quiet)
    except OSError as exc:
        sys.stdout.write("[ENTRY FAILED] 起不来 pytest：%s: %s\n" % (type(exc).__name__, exc))
        return EXIT_ENTRY_FAILED

    verdict, detail = duration_verdict(timings)

    sys.stdout.write("\n" + "=" * 72 + "\n")
    sys.stdout.write("回归摘要\n")
    sys.stdout.write("  命令            : %s\n" % " ".join(["py", "-3", "-m", "pytest", *pytest_args]))
    sys.stdout.write("  pytest 退出码   : %d\n" % returncode)
    sys.stdout.write("  摘要行          : %s\n" % (parsed["summary_line"] or "（没有可识别的摘要行）"))
    sys.stdout.write("  passed / failed / errors : %d / %d / %d\n"
                     % (parsed["passed"], parsed["failed"], parsed["errors"]))
    # 墙钟与 CPU 永远一起报。只报墙钟的耗时会被机器休眠污染，
    # 而污染过的那一行在报告里和「真跑了这么久」长得一模一样。
    # `cpu_source` 跟在后面：`cpu_s` 很低有两种成因（真没算什么 / 只采到累计值），
    # 不给来源就分不出是哪一种，而它们的结论完全相反。
    sys.stdout.write("  耗时            : 墙钟 %.2fs / CPU %s / 来源 %s（%s）\n"
                     % (timings["wall_s"],
                        "采不到" if timings["cpu_s"] is None else "%.2fs" % timings["cpu_s"],
                        timings["cpu_source"], verdict))
    if verdict != provenance.PASS:
        sys.stdout.write("  耗时说明        : %s\n" % detail)
    if parsed["no_tests"]:
        sys.stdout.write("  注意            : pytest 报告「没有跑任何用例」——"
                         "这不算通过，请先确认目标路径对不对。\n")

    # 退出码只有一个来源：pytest 的。摘要只用于**解释**这个码，不参与决定它。
    # 唯一例外是「pytest 说成功但一个用例都没跑」——那说明路径写错了，
    # 让它以成功退出等于把「什么都没测」当成「测过了」。
    if returncode == 0 and parsed["no_tests"]:
        sys.stdout.write("  结论            : 未跑任何用例，判入口失败（%d）\n" % EXIT_ENTRY_FAILED)
        final_code = EXIT_ENTRY_FAILED
    elif returncode != 0:
        sys.stdout.write("  结论            : 失败（pytest 退出码 %d）\n" % returncode)
        final_code = returncode
    elif summary_looks_bad(parsed):
        # 退出码 0 但摘要说失败：两者矛盾时不许沉默。宁可让门红，因为它意味着
        # 有人把 pytest 的退出码弄丢了——这正是 Gate 0 事故 2。
        sys.stdout.write("  结论            : 矛盾——pytest 退出码 0 但摘要报失败，判入口失败（%d）\n"
                         % EXIT_ENTRY_FAILED)
        final_code = EXIT_ENTRY_FAILED
    else:
        sys.stdout.write("  结论            : 全绿\n")
        final_code = returncode

    if args.report:
        _write_report(Path(args.report), pytest_args, returncode, parsed,
                      timings, verdict, detail, final_code)
        sys.stdout.write("  报告            : %s\n" % args.report)
    return final_code


if __name__ == "__main__":
    raise SystemExit(main())
