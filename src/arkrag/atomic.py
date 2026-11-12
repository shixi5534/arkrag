"""原子写与共享状态互斥（计划 C3）。

现有代码用 `path.with_suffix(".tmp")` 当临时文件，并且只 `os.replace` 不 fsync。
两个问题：

1. **固定临时文件名**：同目录下两个进程同时写同一产物会互相截断，最后落盘的是
   「两个进程各写了一半」的混合物——不是任何一方想要的内容。改用同目录 UUID 名。
2. **不 fsync**：`os.replace` 只保证目录项原子替换，不保证数据已落盘。断电或
   进程被 kill 后可能出现「文件名在、内容空」的产物，而门禁读到的就是它。
   顺序必须是 write → flush → fsync(file) → replace → fsync(dir)。

另外提供 `file_lock()`：`gates.json` 这类「读-改-写」共享状态必须串行，否则两个
stage 同时写会静默丢掉其中一方的行。
"""
from __future__ import annotations

import contextlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

from . import legacy

# Windows 上 msvcrt 可用；POSIX 上 fcntl 可用。两者都不可用时退化为进程内锁。
try:  # pragma: no cover - 平台分支
    import msvcrt
except ImportError:  # pragma: no cover
    msvcrt = None  # type: ignore[assignment]
try:  # pragma: no cover - 平台分支
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


def fsync_dir(path: Path) -> None:
    """把目录项也刷到盘上。Windows 不允许对目录开句柄，故静默跳过。"""
    if os.name == "nt":
        return
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """同目录 UUID 临时文件 + fsync + 原子替换 + 目录 fsync。

    这里是**所有产物写入的唯一通道**，因此也是冻结写保护的唯一关口
    （`legacy.guard_frozen_artifact`）。保护放在这里而不是各调用点，是因为
    2026-10-30 的事故证明"每个调用点各自记得调一次"必然漏：`run_l1.py` /
    `run_l2.py` 都没调，重跑就把已冻结产物覆盖了。
    """
    path = Path(path)
    legacy.guard_frozen_artifact(path, reason="atomic_write_bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:12]}.tmp"
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        fsync_dir(path.parent)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def atomic_write_text(path: Path, text: str, *, newline: str = "\n") -> None:
    if newline != "\n":
        text = text.replace("\n", newline)
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_stream(path: Path, writer) -> None:
    """流式版本：把已打开的临时文件句柄交给 `writer(fh)` 逐行写。

    为什么需要它：`scenes.jsonl` 有 71 MB，拼成一个 `bytes` 再交给
    `atomic_write_bytes` 意味着峰值内存翻几倍。但**不能因此绕开这条通道**——
    2026-10-30 的事故正是 `parse.parse_corpus` 用 `out_path.open("w")` 直写，
    保护完全没经过，冻结的 scenes.jsonl 被写成了 7 字节的 `rebuilt`。

    所以两个要求一起满足：保护照样在这里拦（第一行就查），写盘照样是
    write → flush → fsync → replace → fsync(dir)，只是内容由调用方流式灌进来。
    """
    path = Path(path)
    legacy.guard_frozen_artifact(path, reason="atomic_write_stream")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:12]}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            writer(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        fsync_dir(path.parent)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def atomic_write_json(path: Path, obj: Any, *, indent: int | None = 2) -> None:
    text = json.dumps(obj, ensure_ascii=False, indent=indent) + "\n"
    atomic_write_text(path, text)


def append_jsonl(path: Path, record: dict) -> None:
    """追加写一行。日志类文件要求「行不撕裂」，且**不许静默丢行**。

    ## 为什么不是单纯 `O_APPEND` + `os.write`

    原实现只靠 `O_APPEND` 保证"落在文件末尾"，这在 POSIX 上成立，但 **Windows
    上不成立**：CPython 用 `_O_APPEND` 让 CRT 在每次写前 seek 到末尾，而两个写者
    各自 `open` 时拿到的是同一个末尾偏移，于是**后写覆盖先写**。实测（8 线程 ×
    300 条）只落 1940～2072 行，丢 300～460 行，**无异常、无撕裂行、`os.write`
    报满长**——这种"成功返回但数据没了"是最坏的一类缺陷。

    `server.py` 的审计日志（`audit()` / `log_query()`）正是多线程写这里：
    管理操作的可追溯性会被静默破坏，而"回滚了但没人知道是谁做的"正是这个项目
    最不能接受的失败形态。

    ## 修法

    两层串行：进程内按路径加一把锁（挡住同进程多线程），外层再套 `file_lock()`
    （挡住多进程，Windows 走 `msvcrt.locking`、POSIX 走 `flock`）。
    代价是每行一次加锁，对审计/查询日志这个量级可以忽略；正确性优先。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")

    key = str(path)
    with _PROC_LOCKS_GUARD:
        proc_lock = _PROC_LOCKS.setdefault(key, _ProcLock())

    with proc_lock:
        with file_lock(path):
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                # 持锁期间显式 seek 到末尾：即便平台忽略 O_APPEND 语义，
                # 这一段也不会写到别处去。
                os.lseek(fd, 0, os.SEEK_END)
                os.write(fd, line)
                os.fsync(fd)
            finally:
                os.close(fd)


class _ProcLock:
    """进程内互斥（无平台文件锁时的退路）。"""

    def __init__(self) -> None:
        import threading

        self._lock = threading.RLock()

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, *exc):
        self._lock.release()
        return False


_PROC_LOCKS: dict[str, _ProcLock] = {}
_PROC_LOCKS_GUARD = __import__("threading").Lock()


@contextlib.contextmanager
def file_lock(target: Path, *, timeout: float = 60.0) -> Iterator[None]:
    """对 `target` 建一把独占锁（锁文件为 `<target>.lock`），跨进程串行。

    用于「读-改-写」共享状态（如 gates.json）。超时抛 TimeoutError，绝不静默
    跳过——静默跳过等于没锁。
    """
    target = Path(target)
    lock_path = target.parent / (target.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    key = str(lock_path)
    with _PROC_LOCKS_GUARD:
        proc_lock = _PROC_LOCKS.setdefault(key, _ProcLock())

    deadline = time.monotonic() + timeout
    fh = open(lock_path, "a+b")
    try:
        acquired = False
        if msvcrt is not None:
            while True:
                try:
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    acquired = True
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"取锁超时: {lock_path}")
                    time.sleep(0.02)
        elif fcntl is not None:
            while True:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"取锁超时: {lock_path}")
                    time.sleep(0.02)

        with proc_lock:
            yield
    finally:
        if acquired:
            with contextlib.suppress(OSError):
                if msvcrt is not None:
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                elif fcntl is not None:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()
