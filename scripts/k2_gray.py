# -*- coding: utf-8 -*-
"""K2 灰度查询：起**真实的 HTTP 服务实例**，对同一批固定探针题在两代上各问一遍，逐题比对。

    py -3 scripts\\k2_gray.py                                   # 候选 vs 上一代（默认世代）
    py -3 scripts\\k2_gray.py g-20261103T132152Z-06a741 g-20261031T035231Z-4eafae
    py -3 scripts\\k2_gray.py g-20261030T154606Z-55dfd7        # 演示用：该代不可服务 → 退出码 2

## 为什么要有这一步（漏项）

`docs/升级计划_GraphRAG架构.md:420-428` 的 K2 构建顺序末尾是
`… → 候选 READY → 灰度查询 → 正式发布`，而这两步一直没做：`docs/升级台账.md`
与 `docs/k阶段实测.md` 里「灰度查询」「正式发布」各 0 处。K4 演练里那个「灰度」
是**进程内**对 18 题跑一遍探针（`scripts/k4_drill.py`），既没有 HTTP，
也没有两个实例，更没有"两代同时在服务"的语义。

## 为什么比 HTTP 层，而不是只比进程内函数

`src/arkrag/server.py` 里有大量东西是**进程内调用完全绕过去**的：

* **路由**——`_route_get` / `_route_post` 的路径匹配、`/chunks/{id}` 那种
  "前缀之后整段都是 id"的解析（块 id 自带斜杠）、404 的 `not_found` 载荷；
* **鉴权**——`_require_admin()`：没配 `ARKRAG_ADMIN_TOKEN` 时管理端点必须
  `503 admin_disabled`，配了则要求 `Authorization: Bearer` 并对不上就 `401`；
* **限流**——`AppState.check_rate()` 每个客户端每窗口的计数，超限 `429`；
* **请求体排空**——`_BodyReader.read()` 必须在任何拒绝分支**之前**读掉请求体，
  否则 503 时连接被 RST，客户端拿到 `ConnectionAbortedError: [WinError 10053]`
  （这是 J1 修过的真实缺陷，只在真 socket 上复现）；
* **错误载荷**——`_error_payload()` 的 `{"error": {code, message}}` 形状与
  `sanitize_text()` 的路径/令牌清洗；
* **并发与墙钟**——`AppState.run_bounded()` 的信号量、504 `timeout`、503 `busy`；
* **载荷整形**——`_fit_payload()` 超 `max_response_bytes` 时的降级与 `truncated` 字段。

灰度要判断的恰恰是"**用户真正打到的东西**对不对"。绕过 socket 直接调
`pipeline.ask_once()`，上面每一条都不会被执行，于是"路由接错了""限流把灰度
流量挡了""错误载荷把内部路径泄了"这类事故在灰度里全都看不见。

## 两代是怎么起的（不动线上指针）

用 `scripts/serve.py --generation <gid> --host 127.0.0.1 --port 0` 起**子进程**：

* `--generation` 显式指定，所以**不读也不需要**发布指针来选世代；
* `--port 0` 让系统分配端口，不写死（同机可能有别的脚本在跑）；
* 本脚本**绝不写** `data/arkrag/current.json`，**绝不调** `/admin/rollback`。
  线上指针仍是 `g-20261030T154606Z-55dfd7`（`graph_schema=ark-rag-1`，当前链路
  加载不了），所以：

  * 基线取**上一代可服务**的 `g-20261031T035231Z-4eafae`，而不是线上那个；
  * 两代的 `/ready` 里 `serving_is_current` 都会是 `false`——这是**预期**的
    灰度形态（服务未发布世代），不是失败。

  `/release/current` 的指针值在演练前后各读一次并记进报告，用来证明"没动生产指针"。

## 探针题集是固定的

`PROBE_IDS` 写死在代码里（20 题，与 `scripts/k4_drill.py` 的 18 题同源，
另加 2 道 Gold Set 标注 `should_abstain=True` 的题，让"该拒却答"这个方向
也在集合里）。**不用随机、不用种子、不按类别现取**——否则"这次与上次"无法比较，
而且题少了会让判据自动变绿，所以还有一条"探针题数与固定题集一致"的门。

每题记录：是否回答 / 是否拒答 / 引用能否在该题命中块上定位 / 引用条数 /
命中 chunk_id 列表 / 是否命中 Gold Set 的 gold_chunk_ids。

## 判据（五条，都能失败）

1. `probe_set_fixed` —— 实际跑的题数 = `EXPECTED_PROBE_COUNT`，题 id 集合与
   `PROBE_IDS` 完全一致、无重复，且每题都能在 Gold Set 里找到（防"题变少了所以全过"）。
2. `probes_answered` —— 两代每道探针都拿到 `200`；出现任何非 200（含 429 限流、
   504 超时、503 未就绪）都算不成立。
3. `generation_identity` —— `/ready` 报的 `generation_id`、`/release/current` 报的
   `served_generation_id`、以及**每一条** `/query` 响应里的 `generation_id`，都必须
   等于"我请求的那一代"。防最隐蔽的那种错："我以为是候选在服务，其实是旧代"。
4. `abstain_flip` —— 同一题在两代上的**拒答决策**不能翻转（该答变该拒或反过来，
   用户能直接感知）。翻转 0 题才成立。
5. `citation_locate` —— 引用必须能在**该题命中的块**里定位：经 `GET /chunks/{id}`
   取回正文后，`text[start:end] == evidence` 且块确实属于本题命中块。失败数为 0 才成立。

Gold Set 里 `should_abstain=True` 的题被答了，属于**答案质量**问题（Gold Set 评测
那一关管），本次只把它记进 `abstain_flip` 的 `data` 与输出，不另设门——两代都有
同样表现说明这不是灰度引入的回归，混进灰度门会让这道门失去"这次和上次比"的语义。

## 内存与并发：模式是量的，不是猜的

`scripts/bench_h1.py` 实测每代加载后峰值 RSS 在 **1.7 GB** 量级，两个实例同时活着
约 3.4 GB。所以先量本机可用物理内存（`psutil` 不用——项目运行时零第三方依赖，
拿不到就写 `null` 不猜；Windows 走 `ctypes` 的 `GlobalMemoryStatusEx`）：

* 可用 ≥ 6144 MiB → `concurrent`：两个实例**同时**活着，逐题交替打（候选、基线、
  候选、基线…），是真正的并行灰度；
* 否则（或量不出来）→ `sequential`：先起候选、跑完探针、停；再起基线、跑完、停。

两种模式都把 `mode` 与实测 `available_memory_mib` 写进报告。

**顺次模式的局限**（如实记）：顺次时两个实例不会同时活着，所以它证明不了
"流量分流下两代互不干扰"、也测不出"两个实例同时吃内存时的表现"；它证明的是
"**同一批题在两代上的行为差异**"，而这正是本轮灰度要判的东西。要判"并行下是否
互相干扰"必须跑并发模式。

## 用法与退出码

    py -3 scripts\\k2_gray.py [候选世代] [基线世代]

* `0` —— 五条判据全部成立。
* `1` —— 起了服务，但有判据不成立（有 `FAIL` 行）。
* `2` —— 硬错误：服务起不来、世代加载/就绪失败、探针集缺失或与 Gold Set 对不上。

报告写 `data/k2_gray.json`（与 `data/k2_hard_gates.json`、`data/k4_drill.json` 同构）。
"""
from __future__ import annotations

import atexit
import ctypes
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from arkrag import atomic, config, subproc  # noqa: E402

# pytest 捕获期间 `sys.stdout` 是它自己的对象，不一定有 `reconfigure`。守卫必须有。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SERVE = ROOT / "scripts" / "serve.py"
GOLD = ROOT / "data" / "goldset" / "gold-v1.json"
OUT = ROOT / "data" / "k2_gray.json"

# 候选（READY、从未发布）与上一代**可服务**的世代。
# 线上那个 g-20261030T154606Z-55dfd7 是 ark-rag-1，当前链路加载不了，不能当基线。
CANDIDATE_GID = "g-20261103T132152Z-06a741"
BASELINE_GID = "g-20261031T035231Z-4eafae"

# 固定探针题集。**顺序也固定**，方便两次报告逐行对比。
# 前 18 题与 scripts/k4_drill.py 的固定探针同源（跨代可比）；
# 后 2 题是 Gold Set 标注 should_abstain=True 的，让"该拒"方向也在集合里。
PROBE_IDS: tuple[str, ...] = (
    "q-ad-000001", "q-ad-000002", "q-ad-000003",   # adversarial
    "q-al-000001", "q-al-000002",                  # alias
    "q-cx-000001", "q-cx-000002",                  # cross_chunk
    "q-dr-000001", "q-dr-000002",                  # direct_relation
    "q-mh-000001", "q-mh-000002",                  # multihop
    "q-na-000001", "q-na-000002", "q-na-000003",   # negative_abstain（gold 说不用拒）
    "q-sc-000001", "q-sc-000002",                  # single_chunk
    "q-tc-000001", "q-tc-000002",                  # time_conflict
    "q-na-000013", "q-na-000014",                  # negative_abstain（gold 说必须拒）
)

# 与 PROBE_IDS 分开写成字面量：删掉上面任何一题，这条门就会红。
# 写成 len(PROBE_IDS) 就等于自己跟自己比，恒真。
EXPECTED_PROBE_COUNT = 20

# 每代峰值 RSS 约 1.7 GB；两个实例同时活着约 3.4 GB。给 6 GB 门槛留出余量。
CONCURRENT_MIN_AVAIL_MIB = 6144

BOOT_TIMEOUT_S = 90.0      # 从起进程到打出「监听」行
READY_TIMEOUT_S = 300.0    # 冷启动要载入 121 MB 倒排 + 13798 块，实测 ~70 s
QUERY_TIMEOUT_S = 300.0
KILL_GRACE_S = 30.0

# 起服务时放宽限流：探针本身会顺便打 /chunks 复核引用，
# 默认 60 次/60 秒会把灰度流量自己挡成 429（那是限流生效，不是判据失败）。
SERVE_RATE_LIMIT = 2000

_LIVE: list["ServerProc"] = []


class ServeError(RuntimeError):
    """服务起不来 / 世代加载失败 / 传输层断开。一律映射到退出码 2。"""


# ------------------------------------------------------------------ 内存实测


class _MemoryStatusEx(ctypes.Structure):
    """`GlobalMemoryStatusEx` 的入参出参结构（照 scripts/bench_h1.py:67-110 的写法）。"""

    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def memory_status() -> dict:
    """本机物理内存读数。拿不到就写 None —— **不猜**。"""
    if sys.platform != "win32":
        return {
            "avail_mib": None,
            "total_mib": None,
            "load_percent": None,
            "source": "unsupported_platform",
        }
    try:
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx 返回 0")
        return {
            "avail_mib": int(status.ullAvailPhys // (1024 * 1024)),
            "total_mib": int(status.ullTotalPhys // (1024 * 1024)),
            "load_percent": int(status.dwMemoryLoad),
            "source": "GlobalMemoryStatusEx",
        }
    except Exception as exc:  # noqa: BLE001 - 量不出来不影响判据，只影响模式选择
        return {
            "avail_mib": None,
            "total_mib": None,
            "load_percent": None,
            "source": f"unavailable: {type(exc).__name__}",
        }


def pick_mode(avail_mib: int | None) -> str:
    """够 6 GB 就并行，否则（或量不出来）顺次。量不出来时选保守的那个。"""
    if avail_mib is not None and avail_mib >= CONCURRENT_MIN_AVAIL_MIB:
        return "concurrent"
    return "sequential"


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def process_peak_rss_mib(pid: int) -> int | None:
    """取某个**子进程**的峰值工作集。拿不到返回 None（不猜、不写 0）。"""
    if sys.platform != "win32":
        return None
    PROCESS_QUERY_INFORMATION = 0x0400
    PROCESS_VM_READ = 0x0010
    handle = None
    try:
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, int(pid)
        )
        if not handle:
            return None
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
        if not ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        ):
            return None
        return int(counters.PeakWorkingSetSize // (1024 * 1024))
    except Exception:  # noqa: BLE001
        return None
    finally:
        if handle:
            try:
                ctypes.windll.kernel32.CloseHandle(handle)
            except Exception:  # noqa: BLE001
                pass


# ------------------------------------------------------------------ HTTP 客户端


def http_call(base: str, method: str, path: str, body: dict | None = None,
              timeout: float = QUERY_TIMEOUT_S) -> tuple[int, dict]:
    """一发一收。传输层错误抛 `ServeError`（说明进程已经不行了），HTTP 错误码照样返回。"""
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(base + path, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, {"_raw": raw}
    except (urllib.error.URLError, OSError) as exc:
        raise ServeError(f"{method} {path} 传输层失败：{type(exc).__name__}: {exc}") from exc


def chunk_path(chunk_id: str) -> str:
    """块 id 自带斜杠，必须整段编码；服务端取出前缀后整段 unquote。"""
    return "/chunks/" + urllib.parse.quote(str(chunk_id), safe="")


class ServerProc:
    """一个 `scripts/serve.py` 子进程 + 它的 HTTP 客户端。

    生命周期有三重保险：`stop()`（正常路径，`finally` 里调）、`atexit`（异常退出）、
    以及 `stop()` 内部的 terminate → wait → kill 三级回收。**不许留孤儿进程。**
    """

    def __init__(self, generation_id: str, *, argv: list[str] | None = None,
                 boot_timeout_s: float = BOOT_TIMEOUT_S,
                 ready_timeout_s: float = READY_TIMEOUT_S) -> None:
        self.generation_id = generation_id
        self.boot_timeout_s = boot_timeout_s
        self.ready_timeout_s = ready_timeout_s
        self.argv = list(argv) if argv else [
            sys.executable, "-u", str(SERVE),
            "--generation", generation_id,
            "--host", "127.0.0.1",
            "--port", "0",                      # 系统分配，不写死端口
            "--rate-limit", str(SERVE_RATE_LIMIT),
        ]
        self.proc: subprocess.Popen | None = None
        self.pid: int | None = None
        self.base = ""
        self.port = 0
        self.boot_ms = 0
        self.ready_ms = 0
        self.ready_payload: dict = {}
        self.lines: list[str] = []
        self._buf: list[str] = []
        self._thread: threading.Thread | None = None

    # -- 起 ---------------------------------------------------------------
    def start(self) -> "ServerProc":
        # `env=` 不是可选项。子进程 stdout 默认是 cp936，我们下面按 UTF-8 解码，
        # 少了这一行中文就会变成 \ufffd，`pump()` 里那条「监听 http://…」的正则永不命中，
        # 于是每个用例都要干等满 `boot_timeout_s` 才报「没打出监听地址」——门红了，
        # 但红的方向是假的（脚本其实打印了）。2026-11-04 Gate 0 事故就这一行。
        proc = subprocess.Popen(
            self.argv, cwd=str(ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            encoding=subproc.PIPE_ENCODING, errors="replace", bufsize=1,
            env=subproc.child_env(),
        )
        self.proc = proc
        self.pid = proc.pid
        _LIVE.append(self)
        port_box: list[int] = []

        def pump() -> None:
            for line in proc.stdout:  # type: ignore[union-attr]
                line = line.rstrip("\n")
                self.lines.append(line)
                self._buf.append(line)
                del self._buf[:-400]          # 只留最近的诊断行
                m = re.search(r"监听 http://[^:\s]+:(\d+)", line)
                if m:
                    port_box.append(int(m.group(1)))

        self._thread = threading.Thread(target=pump, daemon=True)
        self._thread.start()

        t0 = time.time()
        while not port_box:
            if proc.poll() is not None:
                self.stop()
                raise ServeError(
                    f"世代 {self.generation_id} 的服务进程提前退出（退出码 {proc.returncode}）："
                    f"{self.tail()}"
                )
            if time.time() - t0 > self.boot_timeout_s:
                self.stop()
                raise ServeError(
                    f"世代 {self.generation_id} 在 {self.boot_timeout_s:g} 秒内没打出监听地址：{self.tail()}"
                )
            time.sleep(0.1)
        self.port = port_box[0]
        self.base = f"http://127.0.0.1:{self.port}"
        self.boot_ms = int((time.time() - t0) * 1000)

        # 轮询 /health 直到进程能应答（**不用 Start-Sleep 猜时间**）。
        while True:
            if proc.poll() is not None:
                self.stop()
                raise ServeError(
                    f"世代 {self.generation_id} 在 /health 就绪前退出（退出码 {proc.returncode}）："
                    f"{self.tail()}"
                )
            try:
                status, _ = http_call(self.base, "GET", "/health", timeout=10.0)
                if status == 200:
                    break
            except ServeError:
                pass
            if time.time() - t0 > self.boot_timeout_s:
                self.stop()
                raise ServeError(f"世代 {self.generation_id} 的 /health 一直不通：{self.tail()}")
            time.sleep(0.2)
        return self

    def wait_ready(self) -> dict:
        """轮询 /ready 直到 ready=True。**/health 通了不代表就绪**（服务端把两者刻意分开）。

        快速失败：`/ready` 返回 503 且**带着 checks 明细**，说明就绪判定已经算完并
        明确判否（加载阶段阻塞在请求里，不会先回 503 再回 200）。这种情况再等也不会
        变绿——世代真的不可服务（例如 graph_schema=ark-rag-1 的 manifest 缺字段），
        所以连测两次确认不是偶发后立即抛 `ServeError`（退出码 2），
        不干等到 `ready_timeout_s`。
        """
        t0 = time.time()
        last: dict = {}
        computed_not_ready = 0
        while True:
            if self.proc is not None and self.proc.poll() is not None:
                code = self.proc.returncode
                self.stop()
                raise ServeError(
                    f"世代 {self.generation_id} 在就绪前退出（退出码 {code}）：{self.tail()}"
                )
            try:
                status, payload = http_call(self.base, "GET", "/ready", timeout=READY_TIMEOUT_S)
            except ServeError as exc:
                raise ServeError(f"世代 {self.generation_id} 的 /ready 打不通：{exc}") from exc
            last = payload if isinstance(payload, dict) else {}
            if status == 200 and last.get("ready") is True:
                self.ready_payload = last
                self.ready_ms = int((time.time() - t0) * 1000)
                return last

            bad = [c for c in (last.get("checks") or [])
                   if not c.get("ok") and c.get("blocking", True)]
            detail = "；".join(f"{c.get('id')}={c.get('detail')}" for c in bad[:4]) or "无诊断"
            if status == 503 and last.get("checks"):
                computed_not_ready += 1
                if computed_not_ready >= 2:
                    self.stop()
                    raise ServeError(
                        f"世代 {self.generation_id} 不可服务：就绪判定算出 ready=false"
                        f"（{len(bad)} 条阻断项不成立）：{detail}"
                    )
            if time.time() - t0 > self.ready_timeout_s:
                self.stop()
                raise ServeError(
                    f"世代 {self.generation_id} 在 {self.ready_timeout_s:g} 秒内没就绪"
                    f"（HTTP {status}）：{detail}"
                )
            time.sleep(1.0)

    # -- 用 ---------------------------------------------------------------
    def get(self, path: str, timeout: float = QUERY_TIMEOUT_S) -> tuple[int, dict]:
        return http_call(self.base, "GET", path, timeout=timeout)

    def post(self, path: str, body: dict, timeout: float = QUERY_TIMEOUT_S) -> tuple[int, dict]:
        return http_call(self.base, "POST", path, body, timeout=timeout)

    def peak_rss_mib(self) -> int | None:
        if self.pid is None:
            return None
        return process_peak_rss_mib(self.pid)

    def tail(self, n: int = 12) -> str:
        return " | ".join(self._buf[-n:]) or "(无输出)"

    # -- 收 ---------------------------------------------------------------
    def stop(self) -> None:
        proc = self.proc
        if proc is None:
            return
        self.proc = None
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=KILL_GRACE_S)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(timeout=KILL_GRACE_S)
                    except subprocess.TimeoutExpired:
                        pass
        finally:
            if self in _LIVE:
                _LIVE.remove(self)

    def __enter__(self) -> "ServerProc":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()


@atexit.register
def _reap_all() -> None:  # pragma: no cover - 只在异常退出路径触发
    for proc in list(_LIVE):
        try:
            proc.stop()
        except Exception:  # noqa: BLE001
            pass


# ------------------------------------------------------------------ 判据（纯函数）


def check_probe_set(probe_ids: tuple[str, ...] | list[str],
                    gold_ids: set[str] | list[str],
                    run_ids: list[str]) -> dict:
    """题数/题集必须与写死的那份一致：题变少了、被换了、不在 Gold Set 里都不成立。"""
    gold = set(gold_ids)
    dupes = sorted({q for q in run_ids if run_ids.count(q) > 1})
    planned = list(probe_ids)
    missing = [q for q in planned if q not in set(run_ids)]
    extra = [q for q in run_ids if q not in set(planned)]
    not_in_gold = [q for q in planned if q not in gold]
    ok = (
        len(planned) == EXPECTED_PROBE_COUNT
        and len(run_ids) == EXPECTED_PROBE_COUNT
        and not missing and not extra and not dupes and not not_in_gold
        and set(run_ids) == set(planned)
    )
    if ok:
        detail = f"固定探针 {len(run_ids)} 题（期望 {EXPECTED_PROBE_COUNT}），题集一致且全部在 Gold Set 中"
    else:
        bits = []
        if len(planned) != EXPECTED_PROBE_COUNT:
            bits.append(f"写死题集 {len(planned)} 题 ≠ 期望 {EXPECTED_PROBE_COUNT}")
        if len(run_ids) != EXPECTED_PROBE_COUNT:
            bits.append(f"实跑 {len(run_ids)} 题 ≠ 期望 {EXPECTED_PROBE_COUNT}")
        if missing:
            bits.append(f"少了 {missing[:4]}")
        if extra:
            bits.append(f"多了 {extra[:4]}")
        if dupes:
            bits.append(f"重复 {dupes[:4]}")
        if not_in_gold:
            bits.append(f"Gold Set 里没有 {not_in_gold[:4]}")
        detail = "；".join(bits) or "题集不一致"
    return {
        "ok": ok,
        "detail": detail,
        "data": {
            "expected": EXPECTED_PROBE_COUNT,
            "planned": len(planned),
            "ran": len(run_ids),
            "missing": missing,
            "extra": extra,
            "duplicates": dupes,
            "not_in_gold": not_in_gold,
        },
    }


def check_probes_answered(probes: list[dict]) -> dict:
    """两代每题都要 200。429 限流、504 超时、503 未就绪都算不成立。"""
    bad: list[dict] = []
    total = 0
    for item in probes:
        for role in ("candidate", "baseline"):
            rec = item.get(role) or {}
            total += 1
            if int(rec.get("http_status") or 0) != 200:
                bad.append({
                    "question_id": item.get("question_id"),
                    "role": role,
                    "http_status": rec.get("http_status"),
                    "error": rec.get("error"),
                })
    ok = not bad and total > 0
    detail = (
        f"两代共 {total} 次探针请求全部 200"
        if ok
        else f"{len(bad)}/{total} 次探针请求不是 200："
             + "；".join(f"{b['question_id']}/{b['role']}={b['http_status']} {b.get('error') or ''}"
                         for b in bad[:4])
    )
    return {"ok": ok, "detail": detail, "data": {"total": total, "bad": bad}}


def check_generation_identity(records: list[dict]) -> dict:
    """世代标识三处核对：/ready、/release/current、每一条 /query 的回显。

    records 每项形如
    {"role","requested","ready_gid","served_gid","query_echoes":[...],"pointer_before","pointer_after"}。
    """
    problems: list[str] = []
    for rec in records:
        role = rec.get("role")
        want = str(rec.get("requested") or "")
        if str(rec.get("ready_gid") or "") != want:
            problems.append(f"{role} /ready 报 {rec.get('ready_gid')!r} ≠ 请求的 {want!r}")
        if str(rec.get("served_gid") or "") != want:
            problems.append(f"{role} /release/current 报 {rec.get('served_gid')!r} ≠ 请求的 {want!r}")
        echoes = [str(x) for x in (rec.get("query_echoes") or [])]
        wrong = [x for x in echoes if x != want]
        if not echoes:
            problems.append(f"{role} 没有任何 /query 世代回显可核对")
        elif wrong:
            problems.append(f"{role} {len(wrong)}/{len(echoes)} 条 /query 回显不是 {want!r}（如 {wrong[0]!r}）")
    ok = not problems and bool(records)
    detail = (
        "；".join(f"{r.get('role')} 三处标识均 = {r.get('requested')}" for r in records)
        if ok else "；".join(problems[:4])
    )
    return {"ok": ok, "detail": detail, "data": {"records": records, "problems": problems}}


def check_abstain_flip(probes: list[dict]) -> dict:
    """同一题两代拒答决策必须一致。翻转 = 用户能直接感知的行为变化。"""
    flips: list[dict] = []
    answered_but_should_abstain = {"candidate": [], "baseline": []}
    for item in probes:
        cand = item.get("candidate") or {}
        base = item.get("baseline") or {}
        if cand.get("http_status") != 200 or base.get("http_status") != 200:
            continue
        ca, ba = bool(cand.get("abstained")), bool(base.get("abstained"))
        if ca != ba:
            flips.append({
                "question_id": item.get("question_id"),
                "category": item.get("category"),
                "should_abstain": item.get("should_abstain"),
                "candidate_abstained": ca,
                "baseline_abstained": ba,
                "direction": "该答变该拒" if ca and not ba else "该拒变该答",
            })
        if item.get("should_abstain"):
            for role, rec in (("candidate", cand), ("baseline", base)):
                if not rec.get("abstained"):
                    answered_but_should_abstain[role].append(item.get("question_id"))
    ok = not flips
    if ok:
        detail = f"翻转 0 题（{len(probes)} 题两代一致）"
    else:
        detail = f"翻转 {len(flips)} 题：" + "；".join(
            f"{f['question_id']}（{f['direction']}）" for f in flips[:4]
        )
    # Gold 标注该拒却答了：答案质量问题，两代一样说明不是本次回归，只记录不设门。
    gold_note = {
        role: sorted(set(ids)) for role, ids in answered_but_should_abstain.items()
    }
    if any(gold_note.values()):
        detail += ("；另：Gold Set 标注应拒答的题被答了（两代相同，非灰度回归）"
                   f" 候选 {gold_note['candidate']} / 基线 {gold_note['baseline']}")
    return {
        "ok": ok,
        "detail": detail,
        "data": {
            "flips": flips,
            "flip_count": len(flips),
            "answered_but_should_abstain": gold_note,
        },
    }


def check_citation_locate(probes: list[dict]) -> dict:
    """引用必须能在**该题命中的块**上定位：`text[start:end] == evidence`。

    `unlocatable` = 块取不回来/偏移越界/原文对不上（产品缺陷）；
    `outside_hits` = 引用块不在本题命中块里（相当于引了没检索到的东西）。
    """
    unlocatable: list[dict] = []
    outside: list[dict] = []
    unverifiable: list[dict] = []
    for item in probes:
        hit_ids = set(item.get("hit_chunk_ids") or [])
        for role in ("candidate", "baseline"):
            rec = item.get(role) or {}
            for bad in rec.get("cite_failures") or []:
                row = {"question_id": item.get("question_id"), "role": role, **bad}
                unlocatable.append(row)
            for bad in rec.get("cite_outside_hits") or []:
                outside.append({"question_id": item.get("question_id"), "role": role, **bad})
            for bad in rec.get("cite_unverifiable") or []:
                unverifiable.append({"question_id": item.get("question_id"), "role": role, **bad})
        # 命中块集合为空时无法判断"引用在不在命中块里"，记进 unverifiable。
        if not hit_ids:
            for role in ("candidate", "baseline"):
                unverifiable.append({
                    "question_id": item.get("question_id"), "role": role,
                    "reason": "no_hit_chunks",
                })
    ok = not unlocatable and not outside
    total = sum(int(((item.get(r) or {}).get("citations") or 0)) for item in probes
                for r in ("candidate", "baseline"))
    if ok:
        detail = f"引用 {total} 条全部可在本题命中块上定位（越界/对不上 0，引到命中块外 0）"
    else:
        detail = (f"引用定位失败 {len(unlocatable)} 条、引到命中块外 {len(outside)} 条"
                  f"（共 {total} 条）："
                  + "；".join(f"{x['question_id']}/{x['role']} {x.get('chunk_id')} {x.get('reason')}"
                              for x in (unlocatable + outside)[:4]))
    return {
        "ok": ok,
        "detail": detail,
        "data": {
            "citations_total": total,
            "unlocatable": unlocatable,
            "outside_hits": outside,
            "unverifiable": unverifiable,
        },
    }


# ------------------------------------------------------------------ 探针采集


def _verify_citations(proc: ServerProc, citations: list[dict], hit_ids: set[str],
                      cache: dict[str, dict]) -> dict:
    """经 HTTP 取回被引用的块，再逐条比对偏移。**不复用进程内对象**——那正是本脚本要避开的。"""
    failures: list[dict] = []
    outside: list[dict] = []
    unverifiable: list[dict] = []
    for cite in citations:
        cid = str(cite.get("chunk_id") or "")
        evidence = str(cite.get("evidence") or "")
        start = int(cite.get("start", -1))
        end = int(cite.get("end", -1))
        if cid not in cache:
            status, payload = proc.get(chunk_path(cid))
            cache[cid] = {"status": status, "payload": payload if isinstance(payload, dict) else {}}
        got = cache[cid]
        if got["status"] != 200:
            failures.append({
                "chunk_id": cid,
                "reason": f"chunk_http_{got['status']}"
                          f"({(got['payload'].get('error') or {}).get('code')})",
            })
            continue
        text = str(got["payload"].get("text") or "")
        if got["payload"].get("text_truncated"):
            unverifiable.append({"chunk_id": cid, "reason": "chunk_text_truncated"})
            continue
        if start < 0 or end <= start or end > len(text):
            failures.append({
                "chunk_id": cid,
                "reason": f"offsets_out_of_range({start},{end},len={len(text)})",
            })
            continue
        if text[start:end] != evidence:
            failures.append({"chunk_id": cid, "reason": "evidence_does_not_match_offsets"})
            continue
        if hit_ids and cid not in hit_ids:
            outside.append({"chunk_id": cid, "reason": "cited_chunk_not_in_hits"})
    return {"failures": failures, "outside": outside, "unverifiable": unverifiable}


def probe_one(proc: ServerProc, item: dict, cache: dict[str, dict]) -> dict:
    """对一题打一次 `/query`，并把引用逐条经 HTTP 复核。"""
    t0 = time.time()
    try:
        status, body = proc.post("/query", {"question": item["question"]})
    except ServeError as exc:
        return {
            "http_status": 0, "error": f"{type(exc).__name__}: {exc}",
            "answered": False, "abstained": None, "abstain_reason": None,
            "citations": 0, "cite_ok": 0, "cite_failures": [], "cite_outside_hits": [],
            "cite_unverifiable": [], "gold_hit": False, "gen_echo": None,
            "ms": int((time.time() - t0) * 1000),
        }
    ms = int((time.time() - t0) * 1000)
    if status != 200:
        err = (body.get("error") or {}) if isinstance(body, dict) else {}
        return {
            "http_status": status,
            "error": f"{err.get('code')}：{err.get('message')}",
            "answered": False, "abstained": None, "abstain_reason": None,
            "citations": 0, "cite_ok": 0, "cite_failures": [], "cite_outside_hits": [],
            "cite_unverifiable": [], "gold_hit": False, "gen_echo": None, "ms": ms,
        }

    answer = body.get("answer") or {}
    citations = list(answer.get("citations") or [])
    units = [str(u.get("chunk_id") or "") for u in (body.get("context") or {}).get("units") or []]
    hits = [str(h.get("chunk_id") or "") for h in body.get("hits") or []]
    hit_ids = {x for x in (units + hits) if x}
    checked = _verify_citations(proc, citations, hit_ids, cache)
    gold_ids = [str(x) for x in item.get("gold_chunk_ids") or []]
    return {
        "http_status": status,
        "error": None,
        "answered": bool(not answer.get("abstained") and (answer.get("text") or "").strip()),
        "abstained": bool(answer.get("abstained")),
        "abstain_reason": answer.get("abstain_reason"),
        "confidence": answer.get("confidence"),
        "answer_chars": len(str(answer.get("text") or "")),
        "citations": len(citations),
        "cite_ok": len(citations) - len(checked["failures"]) - len(checked["unverifiable"]),
        "cite_failures": checked["failures"],
        "cite_outside_hits": checked["outside"],
        "cite_unverifiable": checked["unverifiable"],
        "hit_chunk_ids": units + hits,
        "gold_chunk_ids": gold_ids,
        "gold_hit": bool(set(gold_ids) & hit_ids),
        "gen_echo": body.get("generation_id"),
        "ms": ms,
    }


def collect_probes(procs: dict[str, ServerProc], items: dict[str, dict],
                   pattern: list[str]) -> list[dict]:
    """按 `pattern` 逐题交错打两代（concurrent 模式交错 = 两个实例真的同时在服务）。

    顺序固定：pattern 是固定列表，不随机。
    """
    caches = {role: {} for role in procs}
    out: list[dict] = []
    for qid in pattern:
        item = items[qid]
        row = {
            "question_id": qid,
            "category": item.get("category"),
            "should_abstain": bool(item.get("should_abstain")),
            "question": item.get("question"),
            "gold_chunk_ids": list(item.get("gold_chunk_ids") or []),
        }
        results: dict[str, dict] = {}
        for role in ("candidate", "baseline"):
            if role in procs:
                results[role] = probe_one(procs[role], item, caches[role])
        hit_union: set[str] = set()
        for rec in results.values():
            hit_union |= set(rec.get("hit_chunk_ids") or [])
        row["hit_chunk_ids"] = sorted(hit_union)
        for role, rec in results.items():
            row[role] = rec
        out.append(row)
        cand = row.get("candidate") or {}
        base = row.get("baseline") or {}
        flag = "FLIP" if cand.get("abstained") != base.get("abstained") else "    "
        print(f"  {flag} {qid:<12s} {str(row['category']):<18s} "
              f"候选 答{int(bool(cand.get('answered')))}/拒{int(bool(cand.get('abstained')))} "
              f"引{cand.get('citations', 0):>2d}(失败{len(cand.get('cite_failures') or [])}) "
              f"{cand.get('ms', 0):>6d}ms | "
              f"基线 答{int(bool(base.get('answered')))}/拒{int(bool(base.get('abstained')))} "
              f"引{base.get('citations', 0):>2d}(失败{len(base.get('cite_failures') or [])}) "
              f"{base.get('ms', 0):>6d}ms | gold命中{int(bool(cand.get('gold_hit')))}")
    return out


# ------------------------------------------------------------------ 主流程


def _load_gold() -> tuple[dict[str, dict], str]:
    try:
        raw = GOLD.read_text("utf-8")
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ServeError(f"Gold Set 读不出来（{config.rel_to_root(GOLD)}）：{exc}") from exc
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list) or not items:
        raise ServeError(f"Gold Set 形状不认识（{config.rel_to_root(GOLD)}）：没有 items")
    by_id = {str(it.get("question_id") or ""): it for it in items if isinstance(it, dict)}
    missing = [q for q in PROBE_IDS if q not in by_id]
    if missing:
        raise ServeError(f"探针集里有 {len(missing)} 题在 Gold Set 中找不到：{missing[:5]}")
    return by_id, str(payload.get("schema") or "") if isinstance(payload, dict) else ""


def _read_pointer(proc: ServerProc) -> dict:
    try:
        status, payload = proc.get("/release/current", timeout=30.0)
        if status != 200:
            return {"status": status}
        cur = payload.get("current") or {}
        return {
            "status": status,
            "generation_id": cur.get("generation_id"),
            "served_generation_id": payload.get("served_generation_id"),
            "serving_is_current": payload.get("serving_is_current"),
            "published_at": cur.get("published_at"),
        }
    except ServeError as exc:
        return {"status": 0, "error": str(exc)}


def _warm_up(proc: ServerProc, item: dict) -> int:
    """就绪后先打一题热身，把管道构建（实测 10~30 s）从探针耗时里摘出去。

    必须两代都热身，否则"候选首题 28 s、基线首题 0.3 s"这种差别会被误读成性能回归。
    """
    t0 = time.time()
    try:
        proc.post("/query", {"question": item["question"]})
    except ServeError:
        pass
    return int((time.time() - t0) * 1000)


def _start_all(gids: dict[str, str], mode: str) -> dict[str, ServerProc]:
    procs: dict[str, ServerProc] = {}
    try:
        if mode == "concurrent":
            for role in ("candidate", "baseline"):
                procs[role] = ServerProc(gids[role]).start()
            # 两代的冷启动并行等，否则要串行等两遍 ~70 s。
            errors: list[BaseException] = []

            def wait(role: str) -> None:
                try:
                    procs[role].wait_ready()
                except BaseException as exc:  # noqa: BLE001 - 收齐后统一抛
                    errors.append(exc)

            threads = [threading.Thread(target=wait, args=(r,)) for r in procs]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            if errors:
                raise errors[0]
        else:
            for role in ("candidate", "baseline"):
                procs[role] = ServerProc(gids[role]).start()
                procs[role].wait_ready()
    except BaseException:
        for proc in procs.values():
            proc.stop()
        raise
    return procs


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    cand_gid = args[0] if len(args) > 0 else CANDIDATE_GID
    base_gid = args[1] if len(args) > 1 else BASELINE_GID
    started = time.time()
    print("K2 灰度查询：两代真实 HTTP 实例，逐题比对")
    print(f"[i] 候选 {cand_gid}")
    print(f"[i] 基线 {base_gid}")
    print("[i] 只按 --generation 显式指定世代启动；不写发布指针，不调 /admin/rollback")

    mem = memory_status()
    mode = pick_mode(mem["avail_mib"])
    avail_txt = f"{mem['avail_mib']} MiB" if mem["avail_mib"] is not None else "量不出来"
    print(f"[i] 可用物理内存 {avail_txt}（总 {mem['total_mib']} MiB，负载 {mem['load_percent']}%）"
          f" 来源 {mem['source']}")
    print(f"[i] 模式 {mode}（门槛 {CONCURRENT_MIN_AVAIL_MIB} MiB："
          f"{'两个实例同时活着' if mode == 'concurrent' else '一先一后，串行跑'}）")
    if mode == "sequential":
        print("[i] 顺次模式局限：两代不同时活着，证明的是「同一批题在两代上的行为差异」，"
              "不是流量分流下的互不干扰")

    # ---- 1. 探针集自检（起服务之前就能判，硬错误直接 2） ----
    try:
        items, gold_schema = _load_gold()
    except ServeError as exc:
        print(f"[2] {exc}", file=sys.stderr)
        return 2
    checked_ids = [str(q) for q in PROBE_IDS if q in items]
    if len(checked_ids) != len(PROBE_IDS):
        missing = [q for q in PROBE_IDS if q not in items]
        print(f"[2] 探针集缺失 {len(missing)} 题：{missing[:5]}", file=sys.stderr)
        return 2

    results: list[dict] = []

    def record(cid: str, name: str, outcome: dict) -> None:
        row = {"id": cid, "name": name, "ok": bool(outcome["ok"]),
               "detail": outcome["detail"], "data": outcome.get("data")}
        results.append(row)
        print(f"[{'PASS' if row['ok'] else 'FAIL'}] {name:<20s} {row['detail']}")

    procs: dict[str, ServerProc] = {}
    probes: list[dict] = []
    identity_records: list[dict] = []
    rss: dict[str, int | None] = {}
    try:
        # ---- 2. 起两个实例并等就绪 ----
        print(f"[i] 起服务（serve.py --generation … --port 0，模式 {mode}）")
        procs = _start_all({"candidate": cand_gid, "baseline": base_gid}, mode)
        for role, proc in procs.items():
            ready = proc.ready_payload
            blocking_bad = [c for c in (ready.get("checks") or [])
                            if not c.get("ok") and c.get("blocking", True)]
            print(f"[i] {role} {proc.generation_id} 端口 {proc.port} 启动 {proc.boot_ms} ms / "
                  f"就绪 {proc.ready_ms} ms / 阻断项不符 {len(blocking_bad)}")
            # `serving_is_current=false` 是灰度期的**正常**形态（服务未发布世代），单独说清。
            vis = [c for c in (ready.get("checks") or []) if c.get("id") == "serving_is_current"]
            if vis and not vis[0].get("ok"):
                print(f"[i] {role} serving_is_current=false（预期）：{vis[0].get('detail')}")

        pointer_before = _read_pointer(procs["candidate"])

        # ---- 3. 热身（不记入探针） ----
        for role, proc in procs.items():
            print(f"[i] {role} 热身（管道构建 {proc.generation_id}）：{_warm_up(proc, items[PROBE_IDS[0]])} ms")

        # ---- 4. 逐题打两代 ----
        print("[i] 探针（固定题集，顺序固定；concurrent 模式下逐题交替打两代）")
        probes = collect_probes(procs, items, list(PROBE_IDS))
        # 对账：collect_probes 收到的是 PROBE_IDS，这里再确认没有静默丢题。
        ran_ids = [str(p["question_id"]) for p in probes]

        # ---- 5. 世代标识与指针（探针之后读，此时 served 已确定） ----
        pointer_after = _read_pointer(procs["candidate"])
        for role in ("candidate", "baseline"):
            proc = procs[role]
            echoes = [p[role].get("gen_echo") for p in probes
                      if p.get(role) and p[role].get("http_status") == 200]
            # 每个实例自己报的 served_generation_id 才是"这一代到底服务了谁"的证据。
            served = pointer_after.get("served_generation_id") if role == "candidate" \
                else _read_pointer(proc).get("served_generation_id")
            identity_records.append({
                "role": role,
                "requested": proc.generation_id,
                "ready_gid": (proc.ready_payload or {}).get("generation_id"),
                "served_gid": served,
                "query_echoes": echoes,
            })
        for role, proc in procs.items():
            rss[role] = proc.peak_rss_mib()

        # ---- 6. 判据 ----
        record("probe_set_fixed", "探针题数固定", check_probe_set(PROBE_IDS, set(items), ran_ids))
        record("probes_answered", "两代探针皆 200", check_probes_answered(probes))
        record("generation_identity", "世代标识三处核对", check_generation_identity(identity_records))
        record("abstain_flip", "拒答决策不翻转", check_abstain_flip(probes))
        record("citation_locate", "引用可定位", check_citation_locate(probes))
    except ServeError as exc:
        print(f"[2] 硬错误：{exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("[2] 被中断", file=sys.stderr)
        return 2
    finally:
        for proc in procs.values():
            proc.stop()

    # ---- 7. 汇总与落盘 ----
    bad = [r for r in results if not r["ok"]]
    print()
    for row in results:
        if not row["ok"]:
            print(f"未通过：{row['name']} —— {row['detail']}")
    print(f"灰度判据：{len(results) - len(bad)}/{len(results)} 通过")
    all_passed = not bad and len(results) == 5
    if all_passed:
        print(f"结论：候选 {cand_gid} 与基线 {base_gid} 在同一批 {len(probes)} 题上无拒答翻转、"
              f"引用全部可定位、世代标识正确（模式 {mode}）")
    else:
        print(f"结论：有 {len(bad)} 条判据不成立，候选不得进入正式发布")

    report = {
        "generation_id": cand_gid,
        "baseline_generation_id": base_gid,
        "mode": mode,
        "available_memory_mib": mem["avail_mib"],
        "total_memory_mib": mem["total_mib"],
        "memory_source": mem["source"],
        "probes": probes,
        "checks": results,
        "all_passed": all_passed,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": "scripts/k2_gray.py",
        "gold_schema": gold_schema,
        "probe_set": {
            "expected": EXPECTED_PROBE_COUNT,
            "ran": len(probes),
            "question_ids": [p["question_id"] for p in probes],
        },
        "servers": {
            role: {
                "generation_id": proc.generation_id,
                "port": proc.port,
                "boot_ms": proc.boot_ms,
                "ready_ms": proc.ready_ms,
                "peak_rss_mib": rss.get(role),
                # 用 self.pid 而不是 self.proc.pid：报告是在 finally 收完进程之后才组装的，
                # 那时 self.proc 已是 None，而 pid 是已经发生的事实，应当留下来。
                "pid": proc.pid,
            }
            for role, proc in procs.items()
        },
        "production_pointer_untouched": pointer_before.get("generation_id") == pointer_after.get("generation_id")
        if pointer_before.get("generation_id") else None,
        "production_pointer": {
            "before": pointer_before.get("generation_id"),
            "after": pointer_after.get("generation_id"),
        },
        "elapsed_s": round(time.time() - started, 1),
    }
    try:
        atomic.atomic_write_json(OUT, report)
        print(f"报告已写 {config.rel_to_root(OUT)}")
    except OSError as exc:
        print(f"[2] 报告写不出来（{config.rel_to_root(OUT)}）：{exc}", file=sys.stderr)
        return 2
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
