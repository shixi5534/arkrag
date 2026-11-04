# -*- coding: utf-8 -*-
r"""反向实测的支架：**先看退出码，再清掉上一轮产物**。

## 为什么需要它（同一形态已经出现三次）

| 次数 | 形态 | 后果 |
|---|---|---|
| 1 | `py -3 -m pytest tests -q | Select-Object -Last 40` | 管道右侧成功退出，`$LASTEXITCODE` 变 0，测试红了但外层绿（Gate 0 修的就是这个） |
| 2 | 反向实测 harness 只看 JSON 产物、不看退出码 | patch 锚点写错让脚本崩了（退出码 1），harness 读到**上一轮残留**的 JSON，报出「结论没变，门是坏的」 |
| 3 | 邻接表字段名写错 | 回放与现状逐位相同，看着像「改造无影响」 |

三次都是**同一个形状**：证据链里有一个环节失败了，但它吐出的东西仍然长得像
「这一轮的结果」。第 1 次在外层命令，第 2 次在自查脚本，第 3 次在数据层。

## 两条规矩

1. **退出码非 0 时，产物一律视为无效。** 不看产物内容，也不做「它看起来对」的
   判断——脚本崩了还留着旧文件，那文件描述的不是这一轮。
2. **每轮开始前先删掉上一轮产物。** 光看退出码不够：脚本可能正常退出但没写
   文件（分支没走到），此时读到的是旧文件而退出码是 0。清残留把这种也堵住。

两条都要，缺一条都能造出「读到的不是本轮」的机会。

## 用法

    from arkrag import reverse_probe

    result = reverse_probe.run(
        "python 脚本",
        ["py", "-3", "scripts/p1c_leak_audit.py"],
        artifacts=["data/p1c_leak_audit.json"],
    )
    if not result["usable"]:
        ...  # 产物不可用，别据此下结论

`run()` 返回 `{command, returncode, stdout, stderr, artifacts, usable, reason}`。
`usable` 是唯一该被用来做判断的字段。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from . import config
from . import subproc

#: 退出码分档。反向实测里「脚本自己崩了」与「被测门变红了」是两件事：
#: 前者说明实验本身失败，后者才是实验结果。
EXIT_OK = 0


def clear(artifacts: list) -> list[str]:
    """删掉上一轮产物，返回实际删掉的相对路径。

    **只删显式列出的文件**，不按通配符扫目录——反向实测时有人的产物就躺在
    同一个 `data/` 里，扫目录会把它一起删掉。
    """
    removed: list[str] = []
    for item in artifacts:
        path = Path(item)
        if not path.is_absolute():
            path = Path(config.ROOT) / path
        try:
            if path.exists():
                path.unlink()
                removed.append(config.rel_to_root(path))
        except OSError:
            # 删不掉就如实不记（后面 `usable` 会因为「产物是旧的」而判无效，
            # 不在这里假装删成功了）。
            continue
    return removed


def run(
    label: str,
    argv: list[str],
    *,
    artifacts: list | None = None,
    timeout: float = 1800.0,
) -> dict:
    """跑一条命令，按「先退出码、再产物」的顺序判产物能不能用。

    `artifacts` 里的文件在命令开始**之前**被删除；命令结束后逐个检查是否重新
    出现。任一条不成立则 `usable=False`，并在 `reason` 里说明是哪一条。
    """
    arts = list(artifacts or [])
    cleared = clear(arts)
    proc = subprocess.run(
        argv,
        cwd=str(config.ROOT),
        capture_output=True,
        text=True,
        encoding=subproc.PIPE_ENCODING,
        errors="replace",
        timeout=timeout,
        env=subproc.child_env(),
        check=False,
    )
    present: list[dict] = []
    for item in arts:
        path = Path(item)
        if not path.is_absolute():
            path = Path(config.ROOT) / path
        present.append(
            {
                "path": config.rel_to_root(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else None,
            }
        )

    reasons: list[str] = []
    if proc.returncode != EXIT_OK:
        reasons.append(
            f"命令退出码 {proc.returncode} ≠ 0——**产物一律视为无效**"
            "（脚本崩了还留着旧文件，那文件描述的不是这一轮）"
        )
    missing = [row["path"] for row in present if not row["exists"]]
    if missing:
        reasons.append(
            f"命令结束后这些产物不存在：{'、'.join(missing)}"
            "（退出码 0 也可能是分支没走到，此时不能把旧文件当本轮结果）"
        )

    return {
        "label": label,
        "command": " ".join(argv),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "cleared": cleared,
        "artifacts": present,
        "usable": not reasons,
        "reason": "；".join(reasons) if reasons else "退出码 0 且产物是本轮新写的",
    }


def pytest_argv(*paths: str, extra: list[str] | None = None) -> list[str]:
    """跑 pytest 的标准写法。用 `sys.executable` 而不是 `py`：反向实测要跟自己
    正在用的那个解释器一致，否则「我的环境是好的」和「别人的环境是好的」会被混起来。"""
    argv = [sys.executable, "-m", "pytest", *paths, "-q"]
    if extra:
        argv.extend(extra)
    return argv
