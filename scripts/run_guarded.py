"""带内存熔断的子进程运行器——**纯标准库**，不改任何历史脚本的默认值。

## 它解决的事故

2026-11-09 19:08–19:48 本机两次被外部 Python 进程打满**提交内存**（commit
charge）：

* 19:08:15 `python.exe` PID 20928（Python 3.14.3150.1013）启动；
* 2 分 40 秒内 CommitCharge = 62,174,048,256 B = **57.9 GiB**；
* 19:10:55 Windows 事件 2004：commit **69.2 / 69.3 GiB（99.8%）**，物理内存
  33.65 / 34.12 GB（98.6%），298 进程；
* Chromium 抛 `kOomExceptionCode = 0xE0000008`（ERROR_NOT_ENOUGH_MEMORY，定义在
  `partition_alloc/oom.h`），DSH 宿主 fail-fast：`dsh desktop host exited with
  3758096392`；
* 崩溃瞬间 DSH 自己两个进程各只占 0.4 GiB——**它是被连累的**。

根因是**进程外的内存冲击**，不是 DSH 泄漏、不是插件冲突，也不是 host stderr
尾部那堆 `ssh: 192.168.146.128 timed out`（那是 `rea-linux-mcp` 的日志噪音，
只因为弹窗只贴 stderr 尾部才显得像元凶）。

## 为什么必须用 Job Object，而不是"轮询到了再杀"

本项目一次 `load_generation()` 把 288 MB 磁盘产物读成 Python 对象：

| 文件 | 磁盘 |
|---|---|
| `index/lexical.json` | 106.7 MB |
| `chunks.jsonl` | 68.9 MB |
| `graph/assertions.jsonl` | 68.8 MB |
| `graph/graph.json` | 31.6 MB |
| `graph/adjacency.json` | 26.4 MB |

`json.loads` 之后常驻 1.0–1.5 GB。五个并发就是 5–8 GB。

**轮询后杀是来不及的**：一个进程从 1 GB 涨到 50 GB 只需要几十秒，而"涨"和
"整机换页卡死"之间几乎没有窗口——等到采样点发现超线，机器已经在疯狂换页，
杀进程的 `taskkill` 自己都可能因为内存不足而失败。本机实测过这一点：

    taskkill /F /T /PID <pid>  ->  ERROR: Access denied

熔断判对了、杀不掉，等于没有熔断。

`JOB_OBJECT_LIMIT_JOB_MEMORY` 是**内核级**硬上限：超限的分配请求根本不会被
满足，子进程当场收到 `MemoryError`。内存从没被真正占走，整机毫发无损。
本机实测（Python 3.14.3 / Windows）：

    Job 上限 64 MB，子进程尝试分配 200 MB
    -> 退出码 42，stdout 'MEMORY_ERROR_CAUGHT'

## 三层防护

1. **内核硬上限**（主）：`JOB_OBJECT_LIMIT_JOB_MEMORY` + `KILL_ON_JOB_CLOSE`。
   超限时子进程自己 `MemoryError`；本运行器崩溃时成员进程也被内核清理。
2. **轮询监督**（备）：每 2 秒看系统 commit / 可用物理内存；越线用
   `TerminateJobObject`（内核操作，不需要逐进程句柄权限）杀整棵树。
3. **跑前预估**：按产物磁盘体积 × 膨胀系数给峰值区间；超过预检上限直接拒绝。

## 一个必须记住的实现约束

子进程的 stdout/stderr **直接重定向到文件，不走管道**。上一版熔断器用
`os.read` 读管道，子进程不输出时读操作会永久阻塞，采样循环等于没启动——于是
"有熔断器"和"没有熔断器"在那次事故里没有任何区别。本文件因此把"不阻塞"当成
硬约束：管道一个都不开。

## 用法

    # 只看预估，不跑（最安全的检查）
    py -3.14 scripts/run_guarded.py --estimate-only

    # 跑评测（内核硬上限 4 GB + 轮询监督 + 熔断）
    py -3.14 scripts/run_guarded.py -- py -3.14 scripts/eval_answer_quality.py --split dev --prod

    # 跑测试（收紧到 3 GB、限时 10 分钟）
    py -3.14 scripts/run_guarded.py --job-memory-mb 3072 --timeout-s 600 -- py -3.14 -m pytest tests -q

退出码：子进程自己的退出码照原样透传（`MemoryError` 触发时是子进程自己的码）；
`90` 轮询越线被杀；`91` 超时被杀；`92` 预估超硬上限且未 `--force`；
`93` 已有另一个受管运行在跑。
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import tempfile
import time
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# `scripts/` 不是包的一部分，起子进程要用到 `arkrag.subproc` 的管道编码约定，
# 所以先把 `src` 插进 `sys.path`（与 `scripts/ask.py` 等既有脚本同一做法）。
sys.path.insert(0, str(ROOT / "src"))

from arkrag import subproc  # noqa: E402

#: 默认世代：本仓库当前唯一在用的那一代（与 `eval_answer_quality` 的默认一致）。
DEFAULT_GENERATION = "g-20261103T132152Z-06a741"

#: `load_generation` 真正读进内存的那几个文件（相对世代目录）。
ARTIFACT_FILES = (
    ("index", "lexical.json"),
    ("", "chunks.jsonl"),
    ("graph", "assertions.jsonl"),
    ("graph", "graph.json"),
    ("graph", "adjacency.json"),
)

#: 磁盘字节 -> Python 对象常驻字节的膨胀系数区间。
#:
#: 下限 3.0 对应"字段少、字符串短"的形态（graph.json 这类数字为主的表），
#: 上限 6.0 对应"每行都是小 dict、键名重复"的 JSONL（chunks / assertions）。
#: 实测口径：302.4 MB 磁盘 -> 1.0–1.5 GB 常驻，比值 3.4–5.0，落在区间内。
INFLATION_LOW = 3.0
INFLATION_HIGH = 6.0

#: 内核硬上限默认值（MB）。
#:
#: 本项目正常峰值 1.0–1.5 GB，4096 MB 留了约 2.7 倍余量：既不会误杀正常任务，
#: 又能在进程失控时把它按在 4 GB——而不是让它长到 58 GB 把整机拖死。
DEFAULT_JOB_MEMORY_MB = 4096

#: 轮询监督的阈值。
#:
#: * `max_commit_pct=70` —— **必须按 commit limit 的比例算，不能写死绝对 GB**。
#:   这是实测打出来的教训：事故时 commit limit 被页面文件撑到 69.3 GiB，而事后
#:   同一台机器的 limit 只有 38.0 GiB（页面文件被系统缩回去了）。写死 40 GiB 的
#:   阈值在 38 GiB 的 limit 下**永远不可能触发**——熔断器会静默失效，而这正是它
#:   存在的唯一理由。70% 在 38 GiB 上是 26.6 GiB，在 69 GiB 上是 48.4 GiB，
#:   两种形态下都在"还没开始疯狂换页"之前。
#: * `max_commit_gb=0`（0 = 不用绝对值）—— 绝对上限只在调用方明确知道本机
#:   limit 时才该给；默认交给比例判定。
#: * `min_avail_mb=4096` —— 物理内存可用低于 4 GB 时 Windows 已开始激进换页，
#:   Chromium 的分配失败通常就发生在这一带。
DEFAULT_MAX_COMMIT_PCT = 70.0
DEFAULT_MAX_COMMIT_GB = 0.0
DEFAULT_MIN_AVAIL_MB = 4096
DEFAULT_TIMEOUT_S = 1800
DEFAULT_INTERVAL_S = 2.0

#: 预检硬上限：预估峰值超过它就拒绝运行（`--force` 可越过）。
DEFAULT_PREFLIGHT_MAX_MB = 12288

EXIT_MEMORY = 90
EXIT_TIMEOUT = 91
EXIT_PREFLIGHT = 92
EXIT_LOCKED = 93


# ------------------------------------------------------- Windows Job Object
_IS_WIN = sys.platform == "win32"

if _IS_WIN:  # pragma: no cover - 平台分支，CI 上只会走一支
    _K32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _PSAPI = ctypes.WinDLL("psapi", use_last_error=True)

    _PROCESS_SET_QUOTA = 0x0100
    _PROCESS_TERMINATE = 0x0001
    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _PROCESS_VM_READ = 0x0010

    _JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
    _JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JobObjectExtendedLimitInformation = 9

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", ctypes.c_uint32),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.c_uint32),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", ctypes.c_uint32),
            ("SchedulingClass", ctypes.c_uint32),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    class _MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    class _PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    def system_memory() -> dict:
        """系统内存快照。`commit` 是 RAM + 页面文件的提交额度，事故的直接量。"""
        st = _MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if not _K32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return {}
        gib = 1024.0 ** 3
        used = st.ullTotalPageFile - st.ullAvailPageFile
        return {
            "commit_used_gb": round(used / gib, 3),
            "commit_limit_gb": round(st.ullTotalPageFile / gib, 3),
            "commit_pct": round(100.0 * used / st.ullTotalPageFile, 2)
            if st.ullTotalPageFile else 0.0,
            "total_phys_gb": round(st.ullTotalPhys / gib, 3),
            "avail_phys_mb": int(st.ullAvailPhys // (1024 * 1024)),
            "mem_load_pct": int(st.dwMemoryLoad),
        }

    def process_memory(pid: int) -> dict:
        """单进程的常驻与私有提交字节。取不到就返回空 dict（不猜）。"""
        handle = _K32.OpenProcess(
            _PROCESS_QUERY_LIMITED_INFORMATION | _PROCESS_VM_READ, False, int(pid))
        if not handle:
            return {}
        try:
            ctr = _PROCESS_MEMORY_COUNTERS_EX()
            ctr.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS_EX)
            if not _PSAPI.GetProcessMemoryInfo(handle, ctypes.byref(ctr), ctr.cb):
                return {}
            mb = 1024.0 * 1024.0
            return {"rss_mb": round(ctr.WorkingSetSize / mb, 1),
                    "private_mb": round(ctr.PrivateUsage / mb, 1)}
        finally:
            _K32.CloseHandle(handle)

    def create_job(memory_limit_mb: int):
        """建 Job Object：内存硬上限 + 句柄一关就杀光成员。

        返回 `(job, info)`；`job` 为 `None` 表示本机不支持（调用方退回轮询）。
        """
        job = _K32.CreateJobObjectW(None, None)
        if not job:
            return None, {"error": f"CreateJobObjectW: {ctypes.get_last_error()}"}
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        flags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if memory_limit_mb and memory_limit_mb > 0:
            # 同时设 process 与 job 两级：单进程上限管住一个失控进程，
            # job 上限管住"子进程又开了一堆孙进程"的合计。
            flags |= _JOB_OBJECT_LIMIT_PROCESS_MEMORY | _JOB_OBJECT_LIMIT_JOB_MEMORY
            limit = ctypes.c_size_t(int(memory_limit_mb) * 1024 * 1024)
            info.ProcessMemoryLimit = limit
            info.JobMemoryLimit = limit
        info.BasicLimitInformation.LimitFlags = flags
        ok = _K32.SetInformationJobObject(
            job, _JobObjectExtendedLimitInformation,
            ctypes.byref(info), ctypes.sizeof(info))
        if not ok:
            err = ctypes.get_last_error()
            _K32.CloseHandle(job)
            return None, {"error": f"SetInformationJobObject: {err}"}
        return job, {"memory_limit_mb": int(memory_limit_mb or 0),
                     "limit_flags": flags}

    def assign_to_job(job, pid: int, *, retries: int = 10) -> dict:
        """把子进程放进 Job。

        `AssignProcessToJobObject` **会偶发失败**（上游 `python_memory_limit` 因此
        写了 10 次重试）。这里照做：偶发失败不是"不支持"，直接放弃会静默降级成
        "没有硬上限"，那正是事故的形态。
        """
        attempts: list[dict] = []
        for i in range(max(1, retries)):
            handle = _K32.OpenProcess(
                _PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, int(pid))
            if not handle:
                attempts.append({"try": i, "error": "OpenProcess",
                                 "code": ctypes.get_last_error()})
                time.sleep(0.05)
                continue
            try:
                ok = _K32.AssignProcessToJobObject(job, handle)
                if ok:
                    attempts.append({"try": i, "ok": True})
                    return {"ok": True, "attempts": attempts}
                attempts.append({"try": i, "error": "AssignProcessToJobObject",
                                 "code": ctypes.get_last_error()})
            finally:
                _K32.CloseHandle(handle)
            time.sleep(0.05)
        return {"ok": False, "attempts": attempts}

    def terminate_job(job) -> bool:
        if not job:
            return False
        return bool(_K32.TerminateJobObject(job, 1))

    def close_job(job) -> None:
        if job:
            _K32.CloseHandle(job)

else:  # pragma: no cover - Linux/macOS 分支
    def system_memory() -> dict:
        info: dict[str, int] = {}
        try:
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    key, _, rest = line.partition(":")
                    parts = rest.split()
                    if parts and parts[0].isdigit():
                        info[key] = int(parts[0])  # kB
        except OSError:
            return {}
        if not info:
            return {}
        total_mb = info.get("MemTotal", 0) / 1024.0
        avail_mb = info.get("MemAvailable", info.get("MemFree", 0)) / 1024.0
        swap_total = info.get("SwapTotal", 0) / 1024.0
        swap_free = info.get("SwapFree", 0) / 1024.0
        limit_mb = total_mb + swap_total
        used_mb = limit_mb - (avail_mb + swap_free)
        return {
            "commit_used_gb": round(used_mb / 1024.0, 3),
            "commit_limit_gb": round(limit_mb / 1024.0, 3),
            "commit_pct": round(100.0 * used_mb / limit_mb, 2) if limit_mb else 0.0,
            "total_phys_gb": round(total_mb / 1024.0, 3),
            "avail_phys_mb": int(avail_mb),
            "mem_load_pct": int(round(100.0 * (1 - avail_mb / total_mb)))
            if total_mb else 0,
        }

    def process_memory(pid: int) -> dict:
        try:
            with open(f"/proc/{int(pid)}/status", encoding="utf-8") as fh:
                vals = {}
                for line in fh:
                    key, _, rest = line.partition(":")
                    if key in ("VmRSS", "VmSize"):
                        vals[key] = int(rest.split()[0])  # kB
        except (OSError, ValueError):
            return {}
        out = {}
        if "VmRSS" in vals:
            out["rss_mb"] = round(vals["VmRSS"] / 1024.0, 1)
        if "VmSize" in vals:
            out["private_mb"] = round(vals["VmSize"] / 1024.0, 1)
        return out

    def create_job(memory_limit_mb: int):
        return None, {"error": "platform", "detail": "Job Object 仅 Windows"}

    def assign_to_job(job, pid: int, *, retries: int = 10) -> dict:
        return {"ok": False, "attempts": [], "error": "platform"}

    def terminate_job(job) -> bool:
        return False

    def close_job(job) -> None:
        return None


# ------------------------------------------------------------------ 纯函数
def estimate_peak(total_bytes: int, *, low: float = INFLATION_LOW,
                  high: float = INFLATION_HIGH) -> dict:
    """按磁盘体积给峰值常驻区间。纯函数，便于单测。"""
    mb = max(0, int(total_bytes)) / (1024.0 * 1024.0)
    return {
        "artifact_mb": round(mb, 1),
        "low_mb": round(mb * float(low), 1),
        "high_mb": round(mb * float(high), 1),
        "inflation_low": float(low),
        "inflation_high": float(high),
    }


def commit_threshold_gb(sample: dict, limits: dict) -> float:
    """算出这一刻生效的 commit 阈值（GB）。0 = 不用这一项。

    **必须按 limit 比例算**：写死绝对 GB 会在页面文件被系统缩小后失效。
    事故时 limit 69.3 GiB、事后同一台机 38.0 GiB——绝对阈值 40 GiB 在后者上
    永远不触发，熔断器等于不存在。

    两项都给时取**更严**的那个（min），因为两个约束都得满足才算安全。
    """
    cands: list[float] = []
    pct = float(limits.get("max_commit_pct") or 0.0)
    limit = float(sample.get("commit_limit_gb") or 0.0)
    if pct and limit:
        cands.append(limit * pct / 100.0)
    absolute = float(limits.get("max_commit_gb") or 0.0)
    if absolute:
        cands.append(absolute)
    return min(cands) if cands else 0.0


def classify(sample: dict, limits: dict) -> str | None:
    """判定一次采样是否越线。返回原因串，未越线返回 `None`。

    `avail_phys_mb` 用 `-1` 表示"取不到"；取不到时**不判**这一项，避免把
    "读不到"当成"内存耗尽"（那样会在任何非 Windows 机器上误杀）。
    """
    cap = commit_threshold_gb(sample, limits)
    if cap and float(sample.get("commit_used_gb") or 0.0) >= cap:
        return "commit"
    floor = limits.get("min_avail_mb") or 0
    avail = sample.get("avail_phys_mb", -1)
    if floor and isinstance(avail, int) and avail >= 0 and avail < int(floor):
        return "avail_phys"
    child = limits.get("max_child_private_mb") or 0
    if child and float(sample.get("private_mb") or 0.0) >= float(child):
        return "child_private"
    return None


def artifact_bytes(generation_id: str, generations_root: Path | None = None) -> dict:
    """统计一代产物的磁盘体积。缺文件按 0 计——这里只做估算，不做出产校验。"""
    root = Path(generations_root) if generations_root is not None \
        else ROOT / "data" / "arkrag" / "generations"
    gen = root / generation_id
    files: dict[str, int] = {}
    total = 0
    for sub, name in ARTIFACT_FILES:
        path = (gen / sub / name) if sub else (gen / name)
        size = path.stat().st_size if path.is_file() else 0
        files[f"{sub}/{name}" if sub else name] = size
        total += size
    return {"generation": generation_id, "root": str(gen),
            "files": files, "total_bytes": total}


def format_estimate(est: dict, mem: dict) -> str:
    """人读的预估块。运行前打印的就是它。"""
    total_phys = float(mem.get("total_phys_gb") or 0.0)
    parts = [
        f"产物磁盘合计 {est['artifact_mb']:.1f} MB",
        f"预估峰值常驻 {est['low_mb']:.0f}–{est['high_mb']:.0f} MB"
        f"（膨胀系数 {est['inflation_low']:.1f}–{est['inflation_high']:.1f}×）",
    ]
    if total_phys > 0:
        pct_low = 100.0 * est["low_mb"] / (total_phys * 1024.0)
        pct_high = 100.0 * est["high_mb"] / (total_phys * 1024.0)
        parts.append(f"占物理内存 {pct_low:.1f}%–{pct_high:.1f}%"
                     f"（本机 {total_phys:.1f} GB）")
    return " / ".join(parts)


# ------------------------------------------------------------------ 进程控制
def _kill_everything(proc, job) -> dict:
    """杀整棵进程树。**优先 Job Object**（内核级），退回 `taskkill`，再退回
    `Popen.kill()`。

    实测教训：受沙箱约束的会话里 `taskkill /F /T` 返回 `ERROR: Access denied`，
    子进程照活不误——熔断判对了却杀不掉，等于没有熔断。
    """
    attempts: list[dict] = []
    killed = False
    if terminate_job(job):
        attempts.append({"method": "TerminateJobObject", "ok": True})
        killed = True
    else:
        attempts.append({"method": "TerminateJobObject", "ok": False})
    if not killed:
        try:
            if _IS_WIN:
                res = subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(int(proc.pid))],
                    capture_output=True, text=True, timeout=20,
                    # 管道编码必须与父进程解码一致，否则中文会变成 \\
                    # 见 `src/arkrag/subproc.py` 与 `tests/test_subprocess_encoding.py`。
                    encoding=subproc.PIPE_ENCODING, errors="replace",
                    env=subproc.child_env())
                ok = res.returncode == 0
                attempts.append({"method": "taskkill", "ok": ok,
                                 "returncode": res.returncode,
                                 "stderr": (res.stderr or "").strip()})
                killed = killed or ok
            else:
                os.killpg(os.getpgid(int(proc.pid)), 9)
                attempts.append({"method": "killpg", "ok": True})
                killed = True
        except Exception as exc:  # noqa: BLE001 - 杀不掉必须记下来
            attempts.append({"method": "exception", "ok": False,
                             "error": f"{type(exc).__name__}: {exc}"})
    if not killed:
        try:
            proc.kill()
            attempts.append({"method": "Popen.kill", "ok": True})
        except Exception as exc:  # noqa: BLE001
            attempts.append({"method": "Popen.kill", "ok": False,
                             "error": f"{type(exc).__name__}: {exc}"})
    return {"attempts": attempts}


def _reap(proc, *, timeout: float = 10.0) -> int | None:
    """等子进程真的退出并返回退出码。杀完拿不到退出码等于没验证杀掉。"""
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
        try:
            return proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            return None


def _acquire_lock(lock_path: Path) -> bool:
    """跨进程互斥：同一时刻只允许一个受管重型运行。"""
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(f"pid={os.getpid()}\ntime={time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    return True


def _release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except OSError:
        pass


# ------------------------------------------------------------------ 主流程
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="带内存熔断的子进程运行器（纯标准库，不改任何历史脚本默认值）",
        epilog="用 `--` 分隔本脚本参数与要跑的命令，例如："
               "`run_guarded.py --job-memory-mb 3072 -- py -3.14 -m pytest tests -q`",
    )
    parser.add_argument("--generation", default=DEFAULT_GENERATION,
                        help=f"按哪一代产物估算内存（默认 {DEFAULT_GENERATION}）")
    parser.add_argument("--estimate-only", action="store_true",
                        help="只打印预估并退出，不跑任何子进程")
    parser.add_argument("--job-memory-mb", type=int, default=DEFAULT_JOB_MEMORY_MB,
                        help="内核硬上限 MB：子进程超限时抛 MemoryError 而不是"
                             f"吃掉整机（默认 {DEFAULT_JOB_MEMORY_MB}；0 = 只杀不限）")
    parser.add_argument("--max-commit-pct", type=float,
                        default=DEFAULT_MAX_COMMIT_PCT,
                        # argparse 用 `%` 做格式替换，帮助文本里的百分号要写 `%%`。
                        help="轮询监督：系统提交内存占 commit limit 的百分比上限，"
                             f"超过即杀（默认 {DEFAULT_MAX_COMMIT_PCT}%%；0 = 不用）")
    parser.add_argument("--max-commit-gb", type=float,
                        default=DEFAULT_MAX_COMMIT_GB,
                        help="轮询监督：系统提交内存绝对上限 GB（默认 0 = 不用；"
                             "只在明确知道本机 limit 时给）")
    parser.add_argument("--min-avail-mb", type=int, default=DEFAULT_MIN_AVAIL_MB,
                        help="轮询监督：可用物理内存下限 MB，低于即杀"
                             f"（默认 {DEFAULT_MIN_AVAIL_MB}）")
    parser.add_argument("--max-child-private-mb", type=int, default=0,
                        help="轮询监督：单个子进程私有提交上限 MB（默认 0 = 不用，"
                             "交给 --job-memory-mb 的内核上限）")
    parser.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S,
                        help=f"墙钟超时秒数，0 = 不限（默认 {DEFAULT_TIMEOUT_S}）")
    parser.add_argument("--interval-s", type=float, default=DEFAULT_INTERVAL_S,
                        help=f"采样间隔秒（默认 {DEFAULT_INTERVAL_S}）")
    parser.add_argument("--preflight-max-mb", type=int,
                        default=DEFAULT_PREFLIGHT_MAX_MB,
                        help="预估峰值超过它直接拒绝运行（默认 "
                             f"{DEFAULT_PREFLIGHT_MAX_MB}）")
    parser.add_argument("--force", action="store_true",
                        help="预检超限也照跑（风险自担，仍会熔断）")
    parser.add_argument("--trace", default="tmp_run_guarded_trace.json",
                        help="采样轨迹落盘路径（默认 tmp_run_guarded_trace.json）")
    parser.add_argument("--stdout", default="tmp_run_guarded.out",
                        help="子进程 stdout 落盘路径")
    parser.add_argument("--stderr", default="tmp_run_guarded.err",
                        help="子进程 stderr 落盘路径")
    parser.add_argument("--no-lock", action="store_true",
                        help="不加跨进程互斥锁（不推荐）")
    parser.add_argument("cmd", nargs=argparse.REMAINDER,
                        help="要跑的命令（放在 `--` 之后）")
    args = parser.parse_args(argv)

    # ---- 闸一：跑之前先估 ----
    est = artifact_bytes(args.generation)
    mem = system_memory()
    peak = estimate_peak(est["total_bytes"])
    print(f"[guard] 世代 {args.generation}")
    print(f"[guard] {format_estimate(peak, mem)}")
    if mem:
        print(f"[guard] 当前 commit {mem.get('commit_used_gb')}/"
              f"{mem.get('commit_limit_gb')} GB"
              f"（{mem.get('commit_pct')}%），可用物理 "
              f"{mem.get('avail_phys_mb')} MB")
    print(f"[guard] 内核硬上限 {args.job_memory_mb} MB"
          + ("（Job Object：超限抛 MemoryError，不会吃满整机）"
             if _IS_WIN and args.job_memory_mb else "（未启用）"))
    if peak["high_mb"] > args.preflight_max_mb and not args.force:
        print(f"[guard] 拒绝运行：预估峰值 {peak['high_mb']:.0f} MB 超过预检上限 "
              f"{args.preflight_max_mb} MB。分批跑，或确认风险后加 --force。",
              file=sys.stderr)
        return EXIT_PREFLIGHT
    if args.estimate_only:
        print("[guard] --estimate-only：不运行任何子进程。")
        return 0

    cmd = [c for c in args.cmd if c != "--"]
    if not cmd:
        print("[guard] 没有要给运行的命令。用 `--` 分隔，例如："
              "`run_guarded.py -- py -3.14 -m pytest tests -q`", file=sys.stderr)
        return 2

    # ---- 闸二/三：跑的时候盯，越线就杀 ----
    lock_path = Path(tempfile.gettempdir()) / "arkrag_run_guarded.lock"
    if not args.no_lock and not _acquire_lock(lock_path):
        print(f"[guard] 已有另一个受管运行在跑（锁 {lock_path}）。"
              f"重型任务必须串行——并发正是那次 58 GiB 事故的成因。",
              file=sys.stderr)
        return EXIT_LOCKED

    limits = {
        "max_commit_pct": args.max_commit_pct,
        "max_commit_gb": args.max_commit_gb,
        "min_avail_mb": args.min_avail_mb,
        "max_child_private_mb": args.max_child_private_mb,
    }
    effective_commit = commit_threshold_gb(mem, limits)
    out_path = Path(args.stdout)
    err_path = Path(args.stderr)
    trace_path = Path(args.trace)
    samples: list[dict] = []
    verdict: str | None = None
    kill_info: dict = {}
    job = None
    job_info: dict = {}
    assign_info: dict = {}
    started = time.time()
    rc: int | None = None

    try:
        # 关键：输出重定向到**文件**，一个管道都不开（管道读会永久阻塞）。
        with out_path.open("wb") as out_fh, err_path.open("wb") as err_fh:
            proc = subprocess.Popen(
                cmd, stdout=out_fh, stderr=err_fh,
                # 即便 stdout/stderr 已重定向到文件，仍要显式给 env：
                # 子进程若自己 `print` 中文而环境是 cp936，会抛
                # `UnicodeEncodeError` 并退出码 1，与业务失败无法区分
                # （见 `src/arkrag/subproc.py` 模块头第 2 条危害）。
                env=subproc.child_env())
            print(f"[guard] 子进程 pid={proc.pid}，采样间隔 {args.interval_s}s")
            if effective_commit:
                print(f"[guard] 轮询阈值：commit≥{effective_commit:.1f} GB"
                      f"（= {args.max_commit_pct}% × limit "
                      f"{float(mem.get('commit_limit_gb') or 0):.1f} GB）"
                      f" / 可用物理<{args.min_avail_mb} MB")
            # 建 Job 并尽快把子进程放进去（Python 启动只占约 20 MB，
            # 这段窗口内不可能涨到危险值）。
            job, job_info = create_job(args.job_memory_mb)
            if job is not None:
                assign_info = assign_to_job(job, proc.pid)
                if assign_info.get("ok"):
                    print(f"[guard] Job Object 已生效：内存硬上限 "
                          f"{args.job_memory_mb} MB")
                else:
                    print("[guard] 警告：assign 到 Job 失败，内核硬上限未生效，"
                          "只剩轮询监督（来不及杀的风险）。"
                          f"详情：{json.dumps(assign_info, ensure_ascii=False)}",
                          file=sys.stderr)
            elif _IS_WIN:
                print(f"[guard] 警告：建 Job 失败，只剩轮询监督。"
                      f"详情：{json.dumps(job_info, ensure_ascii=False)}",
                      file=sys.stderr)

            while True:
                rc = proc.poll()
                if rc is not None:
                    break
                sample: dict = {"t_s": round(time.time() - started, 2)}
                sample.update(system_memory())
                sample.update(process_memory(proc.pid))
                samples.append(sample)
                verdict = classify(sample, limits)
                if verdict is None and args.timeout_s \
                        and (time.time() - started) > args.timeout_s:
                    verdict = "timeout"
                if verdict is not None:
                    kill_info = _kill_everything(proc, job)
                    rc = _reap(proc)
                    break
                time.sleep(max(0.05, float(args.interval_s)))
            rc = proc.returncode if rc is None else rc
    finally:
        close_job(job)
        if not args.no_lock:
            _release_lock(lock_path)

    elapsed = round(time.time() - started, 2)
    peak_commit = max((s.get("commit_used_gb") or 0.0) for s in samples) \
        if samples else (mem.get("commit_used_gb") or 0.0)
    peak_private = max((s.get("private_mb") or 0.0) for s in samples) \
        if samples else 0.0
    trace = {
        "schema": "arkrag.run_guarded/1",
        "command": cmd,
        "generation": args.generation,
        "estimate": peak,
        "artifact_bytes": est,
        "job": {"requested_memory_mb": args.job_memory_mb,
                "created": job is not None,
                "info": job_info,
                "assign": assign_info},
        "limits": limits,
        "timeout_s": args.timeout_s,
        "interval_s": args.interval_s,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
        "elapsed_s": elapsed,
        "child_returncode": rc,
        "verdict": verdict,
        "kill": kill_info,
        "peak_commit_used_gb": round(peak_commit, 3),
        "peak_child_private_mb": round(peak_private, 1),
        "n_samples": len(samples),
        "samples": samples,
        "stdout_path": str(out_path),
        "stderr_path": str(err_path),
    }
    try:
        trace_path.write_text(
            json.dumps(trace, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8")
        wrote = str(trace_path)
    except OSError as exc:
        wrote = f"（写不进去：{exc}）"

    print(f"[guard] 结束：{elapsed}s，子进程退出码 {rc}，"
          f"峰值 commit {peak_commit:.1f} GB / 子进程私有 {peak_private:.0f} MB，"
          f"{len(samples)} 次采样")
    print(f"[guard] 轨迹 -> {wrote}")
    print(f"[guard] 子进程输出 -> {out_path} / {err_path}")
    if verdict in ("commit", "avail_phys", "child_private"):
        print(f"[guard] 熔断（{verdict}）：已杀整棵进程树。", file=sys.stderr)
        return EXIT_MEMORY
    if verdict == "timeout":
        print("[guard] 熔断：墙钟超时，已杀整棵进程树。", file=sys.stderr)
        return EXIT_TIMEOUT
    return int(rc) if rc is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
