"""起子进程时的管道编码约定（唯一事实源）。

## 为什么需要这个模块

Windows 下子进程的 `sys.stdout` 默认是 cp936（gbk）。父进程若用
`subprocess.run(..., encoding="utf-8")` 解码子进程输出，而子进程按 cp936 往管道里
写字节，中文就会变成不可逆的 `\\ufffd`（替换字符）。危害有两种，第二种更狠：

1. **假红**：所有「按中文匹配子进程输出」的判据恒不命中。门确实红了，但红的方向
   是错的 —— 它报「脚本没打印那一行」，而脚本其实打印了。本项目三次事故都是
   这种形状：检查自己写错，报出来的红方向是假的。
2. **子进程自己崩**：cp936 编不出的字符（emoji、部分符号）会让子进程的 `print`
   抛 `UnicodeEncodeError`，退出码 1。此时父进程看到的是「子进程失败了」，与真正
   的业务失败无法区分。

2026-11-04 Gate 0 的事故就是形状 1：`scripts/k2_gray.py` 起服务时没传
`PYTHONIOENCODING`，而它用 `re.search(r"监听 http://...")` 等这一行，于是
`tests/test_k2_gray.py` 的 12 个用例各等满 `BOOT_TIMEOUT_S=90` 秒超时才报错。

## 为什么不连 `PYTHONUTF8=1` 一起设

UTF-8 模式（`PYTHONUTF8=1` / `-X utf8`）不只改管道编码，还会改 `open()` 的默认
编码、文件系统路径的编解码、以及 `sys.stdin/stdout/stderr` 之外的一串行为。
我们只想修「父进程按 UTF-8 解码，所以子进程也得按 UTF-8 写管道」这一件事，
不该顺手改变子进程的整个 I/O 语义。这是**判断后不加**，不是漏了。

## 怎么用

::

    from arkrag.subproc import PIPE_ENCODING, child_env

    proc = subprocess.run(cmd, capture_output=True,
                          encoding=PIPE_ENCODING, errors="replace",
                          env=child_env())

`env=` 必须来自 :func:`child_env`，不要就地写 `dict(os.environ)` —— 那会漏掉
`PYTHONIOENCODING`，正是这次事故的样子。`tests/test_subprocess_encoding.py` 用
AST 静态扫描这条规矩，白名单在那边，且每条都带理由。
"""
from __future__ import annotations

import os
from typing import Mapping

#: 父进程解码子进程输出时用的编码，也是子进程必须写管道时用的编码。
#: 两边必须是同一个值，所以只在这一处定义。
PIPE_ENCODING = "utf-8"

#: 环境变量名。单独提出来，好让「谁改了这个约定」在 diff 里一眼可见。
ENV_PIPE_ENCODING = "PYTHONIOENCODING"


def child_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """从 `os.environ` 复制一份，再把管道编码强制成 UTF-8。

    参数 `base` 只给测试用（构造一个确定性的输入，不依赖跑测机器的环境）。

    三条约束：

    - **不清空整个环境**。子进程靠 `PATH` / `PYTHONPATH` / `ARKRAG_*` 活着；
      传一个只有 `PYTHONIOENCODING` 的 env 会让它找不到解释器之外的一切。
    - **强制覆盖而不是 setdefault**。若调用方的环境里已经是 `PYTHONIOENCODING=gbk`
      （例如从某个老批处理里继承来的），保留它就会重新造出「父进程按 UTF-8 解码
      cp936 字节」这个组合。两边一致比尊重外部设置更重要。
    - **不动 `PYTHONUTF8`**。理由见模块文档。
    """
    env = dict(os.environ if base is None else base)
    env[ENV_PIPE_ENCODING] = PIPE_ENCODING
    return env
