#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""并发跑 K1 的「回滚命令已实测」那一项，复现 2026-11-04 那个假红缺陷。

为什么要单独一个脚本：这一项在修正前会把探针目录写死在共享路径
`data/arkrag/_k1_rollback_probe/`，于是「两个进程同时跑这一项」就会互相踩对方的
`current.json`。串行跑一万遍都发现不了，必须并发。

判据（退出码）：
  0 = 4 份并发全部通过（修正后的行为）
  1 = 有任一份没通过（修正前的行为：实测 0/4）

用法：
  py -3 scripts/k1_race_check.py          # 默认 4 份
  py -3 scripts/k1_race_check.py --n 8    # 改并发数

注意：这是**检查脚手架本身**的检查，不是产品检查。它跑的是
`scripts/k1_checks.py --only rollback_tested`，那一项只在隔离指针上操作，
不写线上指针、不写真实世代。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from arkrag import subproc  # noqa: E402


def one(index: int) -> tuple[int, int, str]:
    """跑一份，返回 (序号, 退出码, 最后一行 [..] 输出)。"""
    proc = subprocess.run(
        [sys.executable, "scripts/k1_checks.py", "--only", "rollback_tested"],
        capture_output=True,
        # 必须显式 utf-8：Windows 下子进程输出是 UTF-8 字节，按 cp936 解码会在读线程里
        # 抛 UnicodeDecodeError，而且抛在终于还原之前（这个坑踩过一次）。
        text=True,
        encoding=subproc.PIPE_ENCODING,
        errors="replace",
        cwd=str(ROOT),
        # 但「父进程按 UTF-8 读」只有在子进程也按 UTF-8 写时才成立。k1_checks 打印的是
        # 中文结论行（`[ok] …`），少了 env 就会读成 \ufffd，这里挑「最后一个 [ 开头的行」
        # 仍能挑到，但内容不可读——和 K2 同源，一起堵掉。
        env=subproc.child_env(),
    )
    last = ""
    for line in (proc.stdout + proc.stderr).splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            last = stripped
    return index, proc.returncode, last[:160]


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="并发跑 K1 回滚检查（探针目录隔离的回归测试）")
    parser.add_argument("--n", type=int, default=4, help="并发份数，默认 4")
    args = parser.parse_args(argv)

    print(f"并发 {args.n} 份 scripts/k1_checks.py --only rollback_tested")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.n) as pool:
        results = sorted(pool.map(one, range(args.n)))

    passed = 0
    for index, code, line in results:
        good = code == 0
        passed += good
        print(f"  #{index} exit={code} {'ok' if good else 'FAIL'}  {line}")

    print(f"通过 {passed} / {len(results)}")
    if passed == len(results):
        print("结论：探针目录每次运行独一份，并发下互不影响")
        return 0
    print(
        "结论：有并发冲突。检查 scripts/k1_checks.py 里 check_rollback_tested() 的 "
        "tmp_dir 是否又变回了固定路径（应为 config.make_temp_dir(prefix=\"k1_rollback_probe_\")）"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
