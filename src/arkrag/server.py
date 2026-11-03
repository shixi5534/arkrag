"""J1/J2/J3：最小 HTTP API、安全边界与可观测性。**只用标准库**。

## 为什么在这里而不是在脚本里

本项目运行时零第三方依赖（`requirements.txt` 只列 pytest 与它的传递依赖），所以
服务端用 `http.server`，不引 Flask/FastAPI。代价是没有现成的路由、参数校验、超时
与限流——这些全部在这里显式写出来，每条限制都有一个具名常量与环境变量，**不散落
在 handler 里的字面量**。

## 三条共同遵守的纪律

1. **不重写检索链。** 所有回答都走 `arkrag.pipeline.ask_once`，与
   `scripts/ask.py` 是同一条链路（同一个函数对象）。服务端多做一份"读哪些产物、
   按什么顺序调"就是给"必需产物清单"造第二个定义——本项目为此出过三次事故。
2. **不做静默降级。** 超限的参数返回 400 并说明上限，而不是悄悄夹到范围内；
   管理接口没有配 token 就 503 关闭，而不是默认放开。夹紧与放开都是"看起来能用"，
   而真正的问题是没有人知道线上跑的是什么参数。
3. **错误响应不含本机路径/密钥/堆栈。** 所有对外错误消息过一遍
   `sanitize_text()`；异常对象只进日志与审计（且日志本身也不写敏感原文）。

## Endpoints

=============== ===================================== ==================
方法            路径                                  说明
=============== ===================================== ==================
POST            /query                                检索问答（含引用校验）
GET             /chunks/{id}                          块正文（单条，可截断）
GET             /assertions/{id}                     逐观察断言
GET             /entities/{id}                       实体（含别名与指向它的边）
GET             /health                               存活探针（**不代表 ready**）
GET             /ready                                就绪探针（按盘上事实逐项核）
GET             /release/current                      发布指针
POST            /admin/rollback                       回滚（需管理令牌，审计）
POST            /admin/evaluate                       Gold Set 评测（需管理令牌，审计）
=============== ===================================== ==================

## Ready 的判据（计划 J1 原文四条 + 两条工程补充）

计划要求：current Manifest 完整；图索引身份一致；必要数据可读；门禁报告为
READY/PUBLISHED；**不因进程启动就返回 Ready**。

* `manifest_complete` —— Manifest 里那些"这一代到底是什么"的字段齐备
  （`generation_id`/`graph_schema`/`chunk_count`/`assertion_count`/四个 artifact id）。
  缺任何一个，后面所有检查都没有判据，所以在最前面拦。
* `graph_index_identity` —— 直接调 `release.verify_generation()`，**不在这里重写**。
  它已经按内容摘要逐层核对（图/索引/断言/邻接/gate_report_hash/分层身份/
  `index_content_digest`/`probe_digest`）。重写一份就是两份定义，取最松的那份生效。
* `data_readable` —— 五个必需文件存在、非空、可打开。
* `gates_pass` —— `reports/gates.json` 里每一条 `verdict == "PASS"`。
  注意该文件**没有顶层状态字段**，只有逐条 `verdict`，所以这里不查"状态"。
* `serving_is_current`（不阻断）—— 服务的世代是不是发布指针指向的那一代。
  这不是计划的 ready 条件：服务允许指向 READY 但未发布的世代（灰度期就是这个形态，
  见 K2）。但把它显式列出来，因为"线上服务跑的不是线上指针"这件事必须可见。
* `chunks_cover_generation`（阻断）—— 块正文文件的块集合必须覆盖这一代索引里的
  全部 chunk_id。`config.chunks_path()` 是 L1 的最新产物，未必等于某一代实际在用的
  块（D 阶段 13,798 vs 13,781 的分歧就是这么来的）。少块会让引用校验拿到 None，
  表现为"引用被静默剔除"而不是报错——所以它必须是阻断项。

## 明确做不到的事（写在缺省位置，不假装做到了）

* **无法中断已在执行的请求。** 超时用工作线程 + `future.result(timeout=)` 实现：
  到点返回 504，但那个线程仍会跑完（Python 没有安全抢占线程的手段）。并发信号量
  同时限制了这种滞留线程的数量上限。
* **无法保证响应字节数的硬上限。** `max_response_bytes` 达到时按"先丢最重的可选
  字段"的顺序降级，并置 `truncated`；这是一次性构造，不是流式截断。
* **无法做真正的鉴权体系。** 只有一个管理口令（`ARKRAG_ADMIN_TOKEN`），够用于
  本地/单机灰度；多用户与细粒度授权不在 J1 范围。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import secrets
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlparse

from . import atomic
from . import config
from . import index as index_mod
from . import pipeline as pipeline_mod
from . import query as query_mod
from . import release

# ---------------------------------------------------------------- 常量

#: 单条块正文在 API 响应里的默认上限（字符）。生产块正文 P95 为 915 字、
#: 最大 1,031 字，所以 4,096 对正常数据不截断；它是给"块正文里塞了异常长文本"
#: 这种情况的护栏，不是常态路径。
DEFAULT_MAX_RECORD_CHARS = 4096

#: 未指定时的缺省检索参数。与 `scripts/ask.py` 的 `--top-k 20` / `--graph-depth 1`
#: 一致：服务和命令行的默认行为必须一样，否则"同一问题两个入口给出不同答案"。
DEFAULT_TOP_K = 20
DEFAULT_GRAPH_DEPTH = 1

#: `top_k` 上限。基线评测与 CLI 都用 20；H 阶段测过 k=100 也只在 210 ms 量级，
#: 但超出 50 的候选对"上下文只装 4.7 块"没有意义，只是多占内存与延迟。
DEFAULT_MAX_TOP_K = 50

#: 图扩展深度上限。`retrieve.DEFAULT_BUDGET["max_depth"]` 是 2，计划 J2 要求
#: "限制图深度"，这里取同一个数而不是另编一个。
DEFAULT_MAX_GRAPH_DEPTH = 2

#: 单次请求的墙钟上限（秒）。检索链路 P95 在 210 ms 量级，20 s 是"明显已经卡死"的量级。
DEFAULT_TIMEOUT_S = 20.0

#: 同时在跑的请求数。`config.LLM_CONCURRENCY` 默认 4，但没有 LLM 时它不是瓶颈；
#: 检索是纯 CPU，超过 CPU 核数只会互相拖慢。
DEFAULT_CONCURRENCY = 4

#: 每客户端每窗口的请求数上限（滑动窗口，进程内计数）。
DEFAULT_RATE_LIMIT_CALLS = 60
DEFAULT_RATE_LIMIT_WINDOW_S = 60.0

#: `/admin/evaluate` 单次最多评多少题。320 是全量，跑一轮约 10 分钟。
MAX_EVAL_LIMIT = 320

#: 请求体上限。`/query` 的合法载荷只有一问 + 五个数字，64 KiB 已经极其宽松。
DEFAULT_MAX_BODY_BYTES = 64 * 1024

#: 响应上限。超了就降级（丢可选字段）并置 `truncated`。
DEFAULT_MAX_RESPONSE_BYTES = 256 * 1024

_ABSOLUTE_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/][^\s\"'<>|]*|\\\\[^\s\"'<>|]+|/(?:[^/\s\"'<>|]+/){2,}[^/\s\"'<>|]*)")


# ---------------------------------------------------------------- 设置


@dataclass
class Settings:
    """每个限制都有环境变量入口。**没有默认放开的项。**"""

    host: str = "127.0.0.1"
    port: int = 8765
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES
    max_record_chars: int = DEFAULT_MAX_RECORD_CHARS
    max_top_k: int = DEFAULT_MAX_TOP_K
    max_graph_depth: int = DEFAULT_MAX_GRAPH_DEPTH
    max_context_tokens: int = 6000
    timeout_s: float = DEFAULT_TIMEOUT_S
    concurrency: int = DEFAULT_CONCURRENCY
    rate_limit_calls: int = DEFAULT_RATE_LIMIT_CALLS
    rate_limit_window_s: float = DEFAULT_RATE_LIMIT_WINDOW_S
    admin_token: str = ""
    audit_path: Path = field(default_factory=lambda: config.DATA_DIR / "audit" / "server-audit.jsonl")
    query_log_path: Path = field(default_factory=lambda: config.DATA_DIR / "logs" / "queries.jsonl")
    log_query_text: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        def _int(name: str, default: int) -> int:
            raw = os.environ.get(name)
            if raw is None or not str(raw).strip():
                return default
            try:
                return int(raw)
            except ValueError as exc:
                raise ValueError(f"{name}={raw!r} 不是整数") from exc

        def _float(name: str, default: float) -> float:
            raw = os.environ.get(name)
            if raw is None or not str(raw).strip():
                return default
            try:
                return float(raw)
            except ValueError as exc:
                raise ValueError(f"{name}={raw!r} 不是数字") from exc

        return cls(
            host=os.environ.get("ARKRAG_SERVE_HOST", "127.0.0.1"),
            port=_int("ARKRAG_SERVE_PORT", 8765),
            max_body_bytes=_int("ARKRAG_SERVE_MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES),
            max_response_bytes=_int("ARKRAG_SERVE_MAX_RESPONSE_BYTES", DEFAULT_MAX_RESPONSE_BYTES),
            max_record_chars=_int("ARKRAG_SERVE_MAX_RECORD_CHARS", DEFAULT_MAX_RECORD_CHARS),
            max_top_k=_int("ARKRAG_SERVE_MAX_TOP_K", DEFAULT_MAX_TOP_K),
            max_graph_depth=_int("ARKRAG_SERVE_MAX_GRAPH_DEPTH", DEFAULT_MAX_GRAPH_DEPTH),
            max_context_tokens=_int("ARKRAG_SERVE_MAX_CONTEXT_TOKENS", 6000),
            timeout_s=_float("ARKRAG_SERVE_TIMEOUT_S", DEFAULT_TIMEOUT_S),
            concurrency=_int("ARKRAG_SERVE_CONCURRENCY", DEFAULT_CONCURRENCY),
            rate_limit_calls=_int("ARKRAG_SERVE_RATE_LIMIT_CALLS", DEFAULT_RATE_LIMIT_CALLS),
            rate_limit_window_s=_float(
                "ARKRAG_SERVE_RATE_LIMIT_WINDOW_S", DEFAULT_RATE_LIMIT_WINDOW_S
            ),
            # 密钥只从环境读，且**没有默认值**：没配就是关闭管理接口（fail-closed）。
            admin_token=os.environ.get("ARKRAG_ADMIN_TOKEN", ""),
            audit_path=Path(
                os.environ.get("ARKRAG_SERVE_AUDIT_LOG")
                or (config.DATA_DIR / "audit" / "server-audit.jsonl")
            ),
            query_log_path=Path(
                os.environ.get("ARKRAG_SERVE_QUERY_LOG")
                or (config.DATA_DIR / "logs" / "queries.jsonl")
            ),
            log_query_text=_truthy(os.environ.get("ARKRAG_SERVE_LOG_QUERY_TEXT")),
        )

    def describe(self) -> dict[str, Any]:
        """对外可说的限制摘要。**不含 token 本身**，只说它配没配。"""
        return {
            "max_query_chars": query_mod.QUERY_MAX_CHARS,
            "max_body_bytes": self.max_body_bytes,
            "max_response_bytes": self.max_response_bytes,
            "max_record_chars": self.max_record_chars,
            "max_top_k": self.max_top_k,
            "max_graph_depth": self.max_graph_depth,
            "max_context_tokens": self.max_context_tokens,
            "timeout_s": self.timeout_s,
            "concurrency": self.concurrency,
            "rate_limit_calls": self.rate_limit_calls,
            "rate_limit_window_s": self.rate_limit_window_s,
            "admin_enabled": bool(self.admin_token),
            "log_query_text": bool(self.log_query_text),
        }


def _truthy(raw: str | None) -> bool:
    if raw is None:
        return False
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------- 错误与清洗


class ApiError(Exception):
    """带 HTTP 状态码的对外错误。`message` 必须已经可以给外部看。"""

    def __init__(self, status: int, code: str, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.status = int(status)
        self.code = str(code)
        self.message = str(message)
        self.extra = extra


def sanitize_text(text: Any, *, settings: Settings | None = None) -> str:
    """把本机绝对路径与口令值从一段文本里抹掉。

    为什么不能只依赖"我不把路径写进消息"：`str(exc)` 里可能带着
    `FileNotFoundError: … 'E:\\图rag\\data\\arkrag\\x.json'`，而异常来自任何一层。
    统一在出口清一次，比在几十个 `raise` 点各记得清一次可靠。
    """
    out = str(text or "")
    if settings is not None and settings.admin_token:
        out = out.replace(settings.admin_token, "<redacted>")
    # 仓库根单独处理：它含中文目录名，正则未必吃得准，直接按字符串换。
    try:
        root = str(config.ROOT)
        for variant in {root, root.replace("\\", "/"), root.lower(), root.replace("\\", "/").lower()}:
            if variant:
                out = out.replace(variant, "<repo>")
    except Exception:  # pragma: no cover - ROOT 一定可算，这里只是不让清洗本身失败
        pass
    out = _ABSOLUTE_PATH_RE.sub("<path>", out)
    return out


def _error_payload(exc: ApiError, *, settings: Settings | None = None) -> dict:
    body = {"error": {"code": exc.code, "message": sanitize_text(exc.message, settings=settings)}}
    for key, value in exc.extra.items():
        # 下划线开头的键是服务端内部用的处置指令（如 `_drain` 表示请求体读没读完），
        # 不该出现在对外响应里——调用方看到一个 `_drain` 字段只会困惑。
        if key.startswith("_"):
            continue
        body["error"][key] = value
    return body


# ---------------------------------------------------------------- 世代载入与就绪


@dataclass
class _Served:
    generation_id: str
    loaded: dict
    pipeline: dict


class _VerificationCache:
    """`release.verify_generation()` 要重算 250 MB 产物的 sha256，约数秒。

    缓存键刻意用 **Manifest 的 mtime + size + generation_id，加上五个产物的
    mtime + size**：

    * Manifest 是这一代的身份文件，它一变（状态转 PUBLISHED、回滚后重新指向）
      就必须重算——只按世代 id 做键不够，因为同一代的 Manifest 会被
      `advance()`/`update()` 改写而产物没变；
    * 但只按 Manifest 做键**也不够**：产物被换掉时 Manifest 一个字都不变。
      那样 `/ready` 第一次说"逐项一致"，之后无论产物怎么改都继续这么说——
      探针从此只会说好话。实测这就是本缓存第一版的行为：把 `lexical.json`
      换掉之后第二次 `/ready` 仍然报 PASS。加上产物的 mtime/size 之后，
      任何一份产物被动过（哪怕只翻一个字节，size 可能不变但 mtime 会变）
      都会让下一次 `/ready` 重算。5 次 stat 的代价可以忽略。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: dict[tuple, list[str]] = {}

    def key(self, gen: release.Generation) -> tuple:
        def stamp(path: Path) -> tuple:
            try:
                st = Path(path).stat()
                return (st.st_mtime_ns, st.st_size)
            except OSError:
                return (0, 0)

        try:
            m = gen.manifest_path.stat()
            manifest_stamp = (m.st_mtime_ns, m.st_size)
        except OSError:
            manifest_stamp = (0, 0)
        return (
            gen.generation_id,
            manifest_stamp,
            stamp(gen.graph_path),
            stamp(gen.assertion_path),
            stamp(gen.adjacency_path),
            stamp(gen.index_path),
            stamp(gen.gate_report_path),
        )

    def get(self, gen: release.Generation) -> list[str]:
        key = self.key(gen)
        with self._lock:
            if key in self._rows:
                return list(self._rows[key])
        problems = release.verify_generation(gen)
        with self._lock:
            self._rows[key] = list(problems)
        return problems


class _CoverCache:
    """`chunks_cover_generation` 要载入 121 MB 的倒排才能拿到 `doc_id` 集合。

    实测这一项让 `/ready` 单次耗时 6 秒——探针不能这么慢，但也不能把结论记住不放
    （块正文被换掉之后必须能重新失败）。折中：按**两个输入文件的身份**缓存结论。
    只要索引文件与块正文文件都没被改动，覆盖关系就不可能变；任一文件被替换
    （mtime 或 size 变），下一次 `/ready` 必然重算。

    为什么不只按世代 id 缓存：`chunks.jsonl` 在世代目录之外，换掉它不会改任何
    世代的 Manifest——而那正是要防的情形。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: dict[tuple, tuple[bool, str]] = {}

    def key(self, gid: str) -> tuple:
        def stamp(path: Path) -> tuple:
            try:
                st = Path(path).stat()
                return (st.st_mtime_ns, st.st_size)
            except OSError:
                return (0, 0)

        gen = release.Generation(gid)
        try:
            chunks_path = pipeline_mod.resolve_chunks(gen, gen.load())
        except Exception:
            # 定位不出块正文时用 (0,0)：缓存里那条一定也是"不可用"，
            # 而下次若有文件补上，stamp 变成真实值自然就失配重算了。
            chunks_path = Path("")
        return (
            gid,
            stamp(gen.index_path),
            stamp(chunks_path),
        )

    def get(self, gid: str) -> tuple[bool, str]:
        key = self.key(gid)
        with self._lock:
            if key in self._rows:
                return self._rows[key]
        result = _compute_chunks_cover(gid)
        with self._lock:
            self._rows[key] = result
        return result


class AppState:
    """服务端持有的全部可变状态。**所有磁盘事实每次现查，不缓存"就绪"这个结论。**

    "不因进程启动就返回 Ready"的具体落实：`readiness()` 每次都重新读 Manifest、
    重新列门禁报告、重新查文件存在性；只有 sha256 那一层（代价数秒且只依赖产物字节）
    按 Manifest 指纹缓存。
    """

    def __init__(self, settings: Settings, *, generation_id: str = "") -> None:
        self.settings = settings
        self.served_generation_id = str(generation_id or "")
        self._served: _Served | None = None
        self._load_lock = threading.Lock()
        self._verify = _VerificationCache()
        self._cover = _CoverCache()
        self._rate_lock = threading.Lock()
        self._rate: dict[str, list[float]] = {}
        self._semaphore = threading.BoundedSemaphore(max(1, settings.concurrency))
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, settings.concurrency), thread_name_prefix="arkrag-serve"
        )
        self.started_at = time.time()
        self.stats = {"queries": 0, "errors": 0, "timeouts": 0, "rejected": 0}

    # ----------------------------------------------------------- 世代
    def resolve_generation_id(self) -> str:
        """`--generation` 优先；否则取发布指针。**不猜"最新的那个"。**"""
        if self.served_generation_id:
            return self.served_generation_id
        cur = release.read_current()
        gid = str((cur or {}).get("generation_id") or "")
        if not gid:
            raise ApiError(503, "no_current_generation", "没有发布指针，服务没有可服务的世代")
        return gid

    def served(self) -> _Served:
        if self._served is not None:
            return self._served
        with self._load_lock:
            if self._served is not None:
                return self._served
            gid = self.resolve_generation_id()
            try:
                loaded = pipeline_mod.load_generation(gid)
            except SystemExit as exc:
                # `load_generation` 用 SystemExit 报错（脚本语义）。这里必须转成
                # 对外错误：否则一个 `SystemExit` 会把工作线程带走，连接直接断掉，
                # 客户端只看到「连接被重置」，不知道是数据不齐。
                raise ApiError(503, "generation_not_servable", str(exc)) from exc
            # 默认参数与 `scripts/ask.py` 一致（top_k=20 / depth=1）。这两个数在这里
            # 出现是要给"未指定时的缺省"一个值，请求里显式给了就以请求为准。
            pipe = pipeline_mod.build_pipeline(loaded, top_k=DEFAULT_TOP_K, graph_depth=DEFAULT_GRAPH_DEPTH)
            pipe["generation_id"] = gid
            pipe["entities"] = loaded["graph"].get("entities") or []
            self._served = _Served(generation_id=gid, loaded=loaded, pipeline=pipe)
            return self._served

    def served_or_none(self) -> _Served | None:
        try:
            return self.served()
        except ApiError:
            return None

    def invalidate_served(self) -> None:
        """丢掉已载入的世代。回滚之后必须调用。

        为什么不靠"下次请求自然发现"：`_served` 一旦载入就不再查盘，指针切走之后
        服务会继续用旧那一代回答，而唯一的迹象是 `/ready` 的 `serving_is_current`
        变成 false——那是事后才发现，不是当时拒绝服务。
        """
        with self._load_lock:
            self._served = None

    # ----------------------------------------------------------- 就绪
    def readiness(self) -> dict:
        checks: list[dict] = []

        def add(cid: str, ok: bool, detail: str, *, blocking: bool = True) -> None:
            checks.append(
                {"id": cid, "ok": bool(ok), "blocking": bool(blocking), "detail": detail}
            )

        # ---- 1. 发布指针能不能读
        pointer: dict | None = None
        try:
            pointer = release.read_current()
        except release.ReleaseError as exc:
            add("current_pointer", False, f"发布指针不可解析：{exc}")
        else:
            add(
                "current_pointer",
                bool(pointer and pointer.get("generation_id")),
                "指针存在" if pointer else "没有发布指针（current.json 不存在）",
            )

        gid = self.served_generation_id or str((pointer or {}).get("generation_id") or "")
        if not gid:
            add("manifest_complete", False, "无法确定要服务的世代")
            return _ready_result(checks, gid="")

        gen = release.Generation(gid)

        # ---- 2. Manifest 完整性
        manifest: dict = {}
        try:
            manifest = gen.load()
        except Exception as exc:  # ReleaseError / OSError / JSONDecodeError
            add("manifest_complete", False, f"Manifest 读不出来：{exc}")
        else:
            need = (
                "generation_id",
                "graph_schema",
                "chunk_count",
                "assertion_count",
                "graph_artifact_id",
                "index_artifact_id",
                "assertion_artifact_id",
                "adjacency_artifact_id",
            )
            missing = [k for k in need if manifest.get(k) in (None, "", 0)]
            schema_ok = str(manifest.get("graph_schema") or "") == config.GRAPH_SCHEMA_IDENTITY
            if missing:
                detail = f"缺字段：{', '.join(missing)}"
            elif not schema_ok:
                detail = (
                    f"graph_schema={manifest.get('graph_schema')!r}"
                    f"，本服务只服务 {config.GRAPH_SCHEMA_IDENTITY}"
                )
            else:
                detail = "字段齐备且 schema 匹配"
            add("manifest_complete", not missing and schema_ok, detail)

        # ---- 3. 图索引身份一致（复用 release.verify_generation，不另写）
        if manifest:
            problems = self._verify.get(gen)
            add(
                "graph_index_identity",
                not problems,
                "逐项一致" if not problems else f"{len(problems)} 项不符：{problems[0]}",
            )
        else:
            add("graph_index_identity", False, "Manifest 不可用，跳过")

        # ---- 4. 必要数据可读
        unreadable: list[str] = []
        for label, path in (
            ("graph/graph.json", gen.graph_path),
            ("graph/assertions.jsonl", gen.assertion_path),
            ("graph/adjacency.json", gen.adjacency_path),
            ("index/lexical.json", gen.index_path),
            ("chunks.jsonl", config.chunks_path()),
        ):
            try:
                if not path.exists():
                    unreadable.append(f"{label} 不存在")
                elif path.stat().st_size <= 0:
                    unreadable.append(f"{label} 是空文件")
            except OSError as exc:
                unreadable.append(f"{label} 不可读：{exc}")
        add("data_readable", not unreadable, "；".join(unreadable) or "五个必需文件均非空")

        # ---- 5. 门禁报告全 PASS
        gate_rows, gate_problem = _gate_rows(gen)
        if gate_problem:
            add("gates_pass", False, gate_problem)
        else:
            bad = [r for r in gate_rows if str(r.get("verdict") or "") != "PASS"]
            add(
                "gates_pass",
                not bad,
                f"{len(gate_rows)} 条全部 PASS"
                if not bad
                else f"{len(bad)}/{len(gate_rows)} 条不是 PASS，首条 {bad[0].get('name')!r}",
            )

        # ---- 6. 块正文覆盖这一代的索引（阻断）
        cover, cover_detail = self._check_chunks_cover(gid)
        add("chunks_cover_generation", cover, cover_detail)
        # ---- 7. 服务的世代是不是发布指针那一代（**不阻断**，只作可见性）
        cur_gid = str((pointer or {}).get("generation_id") or "")
        add(
            "serving_is_current",
            bool(cur_gid) and cur_gid == gid,
            "服务的就是线上指针那一代" if cur_gid == gid else f"服务 {gid}，线上指针是 {cur_gid or '(无)'}",
            blocking=False,
        )

        result = _ready_result(checks, gid=gid)
        result["gate_rows"] = len(gate_rows)
        result["manifest_status"] = str(manifest.get("status") or "")
        result["coverage_status"] = str(manifest.get("coverage_status") or "")
        return result

    def _check_chunks_cover(self, gid: str) -> tuple[bool, str]:
        """块正文文件必须覆盖这一代索引里的每一个 chunk_id。

        为什么是阻断项：引用校验拿到 `None` 时不会报错，只会把引用判成"块不存在"
        然后剔除，置信度降一档。表现为"答得短"而不是"服务坏了"——静默失效的
        典型形态。D 阶段 13,798 块与线上 13,781 块的分歧正说明这个集合不是天然的。

        结论按「索引文件 + 块正文文件」的身份缓存，理由见 `_CoverCache`。实测不缓存
        时 `/ready` 单次 6 秒（要载入 121 MB 倒排），探针不能这么慢。
        """
        return self._cover.get(gid)

    # ----------------------------------------------------------- 限流
    def check_rate(self, client: str) -> None:
        now = time.monotonic()
        window = max(0.001, self.settings.rate_limit_window_s)
        limit = max(1, self.settings.rate_limit_calls)
        with self._rate_lock:
            stamps = [t for t in self._rate.get(client, []) if now - t < window]
            if len(stamps) >= limit:
                self._rate[client] = stamps
                raise ApiError(
                    429, "rate_limited", f"超过每 {window:g} 秒 {limit} 次的上限",
                    retry_after_s=round(window - (now - stamps[0]), 3),
                )
            stamps.append(now)
            self._rate[client] = stamps

    def run_bounded(self, fn: Callable[[], Any], *, timeout_s: float | None = None) -> Any:
        """在受限并发与墙钟上限内跑一段函数。

        超时返回 504，但工作线程**不会被终止**（Python 无法安全抢占线程）；
        信号量保证这种滞留线程最多 `concurrency` 个。这一点写在模块 docstring 里，
        不假装做到了。
        """
        limit = float(timeout_s if timeout_s is not None else self.settings.timeout_s)
        if not self._semaphore.acquire(timeout=limit):
            raise ApiError(503, "busy", "并发已满，请稍后重试")
        try:
            future = self._executor.submit(fn)
            try:
                return future.result(timeout=limit)
            except FutureTimeout as exc:
                self.stats["timeouts"] += 1
                raise ApiError(504, "timeout", f"请求超过 {limit:g} 秒未完成") from exc
        finally:
            self._semaphore.release()

    # ----------------------------------------------------------- 审计与日志
    def audit(self, action: str, *, actor: str, result: str, **fields: Any) -> None:
        record = {
            "at": _now(),
            "action": action,
            "actor": actor,
            "result": result,
            "pid": os.getpid(),
            **fields,
        }
        try:
            atomic.append_jsonl(self.settings.audit_path, record)
        except OSError:
            # 审计写不下去时**不能让业务静默继续**：管理操作必须在没有审计的情况下
            # 明确失败，否则"回滚了但没人知道是谁做的"。
            raise ApiError(500, "audit_unavailable", "审计日志不可写，拒绝执行管理操作")

    def log_query(self, record: dict) -> None:
        try:
            atomic.append_jsonl(self.settings.query_log_path, record)
        except OSError:
            # 查询日志写不下去只降级为计数，不阻断回答：日志是可观测性，
            # 而拒绝回答会把"磁盘满"升级成"服务不可用"，代价不对等。
            self.stats["errors"] += 1


def _compute_chunks_cover(gid: str) -> tuple[bool, str]:
    """真算一次覆盖关系。逻辑从 `AppState._check_chunks_cover` 原样挪到这里，
    只由 `_CoverCache` 在缓存未命中时调用。

    **块正文用 `pipeline.resolve_chunks()` 定，不自己挑文件。** 覆盖检查要问的
    是「消费者实际会拿到的那份块正文覆盖不覆盖索引」，而"消费者读哪份"这件事
    只有 `resolve_chunks()` 一处定义。这里若写成 `config.chunks_path()`，就又是
    两份判定：K2 实测该文件（L1 最新产物，13781 块）与候选世代（13798 块）不是
    同一批，而 `resolve_chunks()` 已经会 fail-closed 拒掉它——覆盖检查跟着拒，
    两边说的是同一句话。
    """
    gen = release.Generation(gid)
    try:
        idx, _meta = index_mod.load_index(gen.index_path)
    except Exception as exc:
        return False, f"索引读不出来：{exc}"
    want = {str(v) for v in idx.doc_id.values()}
    try:
        manifest = gen.load()
    except Exception as exc:
        return False, f"Manifest 读不出来：{exc}"
    try:
        chunks_path = pipeline_mod.resolve_chunks(gen, manifest)
    except pipeline_mod.ChunksUnavailable as exc:
        return False, f"块正文不可用：{exc}"
    except Exception as exc:
        return False, f"块正文定位失败：{exc}"
    have: set[str] = set()
    try:
        with Path(chunks_path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cid = str(rec.get("chunk_id") or "")
                if cid:
                    have.add(cid)
    except Exception as exc:
        return False, f"块正文读不出来：{exc}"
    missing = want - have
    if missing:
        sample = ", ".join(sorted(missing)[:3])
        return False, f"索引有 {len(want)} 块，块正文缺 {len(missing)} 块（如 {sample}）"
    return True, f"索引 {len(want)} 块均在块正文中（块正文 {config.rel_to_root(chunks_path)}，共 {len(have)} 条）"


def _gate_rows(gen: release.Generation) -> tuple[list[dict], str]:
    path = gen.gate_report_path
    if not path.exists():
        return [], "门禁报告不存在"
    try:
        payload = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [], f"门禁报告不可解析：{exc}"
    if isinstance(payload, dict):
        rows = payload.get("gates") or payload.get("rows") or []
    elif isinstance(payload, list):
        rows = payload
    else:
        return [], f"门禁报告形状不认识：{type(payload).__name__}"
    rows = [r for r in rows if isinstance(r, dict)]
    if not rows:
        return [], "门禁报告是空的（没有可判的条目）"
    return rows, ""


def _ready_result(checks: list[dict], *, gid: str) -> dict:
    blocking = [c for c in checks if c.get("blocking", True)]
    return {
        "ready": all(c["ok"] for c in blocking) and bool(blocking),
        "generation_id": gid,
        "checked_at": _now(),
        "checks": checks,
    }


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------- 载荷整形


def _clip_text(text: Any, limit: int, *, settings: Settings) -> tuple[str, bool]:
    raw = sanitize_text(text, settings=settings)
    if len(raw) <= limit:
        return raw, False
    return raw[:limit], True


def _chunk_payload(rec: dict, *, settings: Settings) -> dict:
    text, cut = _clip_text(rec.get("text") or "", settings.max_record_chars, settings=settings)
    return {
        "chunk_id": str(rec.get("chunk_id") or ""),
        "doc_id": str(rec.get("doc_id") or ""),
        "category": rec.get("category"),
        "chapter": rec.get("chapter"),
        "title": rec.get("title"),
        "scene_index": rec.get("scene_index"),
        "speakers": rec.get("speakers"),
        "chars": rec.get("chars"),
        "turn_ids": rec.get("turn_ids"),
        "text": text,
        "text_truncated": cut,
        # 剧情文本是**数据**，不是指令。显式标注，让任何下游（含以后接上的 LLM）
        # 都知道这一段不能当提示词解释。
        "untrusted": True,
    }


def _fit_payload(payload: dict, limit_bytes: int) -> tuple[dict, list[str]]:
    """超字节上限时按"先丢最重的可选字段"降级。

    顺序是刻意的：上下文单元（最长）→ 引用列表 → 各通道明细。被丢掉的字段名写进
    `truncated`，读的人一眼能看出"这不是全部"。
    """
    dropped: list[str] = []

    def size(obj: Any) -> int:
        return len(json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    if size(payload) <= limit_bytes:
        return payload, dropped

    for key, path in (
        ("context.units", ("context", "units")),
        ("citations", ("citations",)),
        ("channels", ("stats", "channels")),
        ("hits", ("hits",)),
    ):
        node: Any = payload
        ok = True
        for part in path[:-1]:
            if not isinstance(node, dict) or part not in node:
                ok = False
                break
            node = node[part]
        if not ok or not isinstance(node, dict) or path[-1] not in node:
            continue
        node[path[-1]] = []
        dropped.append(key)
        if size(payload) <= limit_bytes:
            break
    payload["truncated"] = dropped
    return payload, dropped


# ---------------------------------------------------------------- 请求体


class _BodyReader:
    """按上限读请求体。超限立刻 413，**不读到内存里再判断**。

    ## 为什么超限时还要把请求体读掉

    只回 413 然后就关连接，在 Windows 上会让客户端收到 `ConnectionAbortedError:
    [WinError 10053]`，而不是那条 413——因为接收缓冲区里还有未读字节，`close()`
    发的是 RST 而不是 FIN，客户端的内核会把已到达的响应一起丢掉。

    实测形态：`test_oversized_body_is_refused_before_reading` 单跑通过、全量跑
    偶发失败，报的正是 10053。单跑通过是因为那时客户端早把几百字节发完、
    数据已在缓冲区里被读走；全量跑时机不同就露出来。这种"只在测试里偶发"的
    失败最容易被当成 flaky 忽略掉，但线上同一个路径会把 413 变成连接重置，
    调用方看到的是"服务挂了"而不是"你的请求太大了"。

    所以超限时**先读掉**至多 `DRAIN_CAP` 字节（丢弃，不进内存累积），再抛 413。
    上限存在的原因是不能被一个声明了 10 GB 的请求拖住；超过就不读了，
    此时连接确实会被重置——但那已经是明确的滥用流量，不是正常客户端。
    """

    DRAIN_CAP = 1024 * 1024

    @staticmethod
    def _drain(handler: BaseHTTPRequestHandler, remaining: int) -> int:
        """读掉至多 `remaining` 字节并丢弃，返回实际读掉的字节数。"""
        dropped = 0
        while remaining > 0:
            piece = handler.rfile.read(min(remaining, 65536))
            if not piece:
                break
            dropped += len(piece)
            remaining -= len(piece)
        return dropped

    @staticmethod
    def read(handler: BaseHTTPRequestHandler, settings: Settings) -> bytes:
        raw_len = handler.headers.get("Content-Length")
        if raw_len is None:
            return b""
        try:
            length = int(raw_len)
        except ValueError as exc:
            raise ApiError(400, "bad_content_length", "Content-Length 不是整数") from exc
        if length < 0:
            raise ApiError(400, "bad_content_length", "Content-Length 为负")
        if length > settings.max_body_bytes:
            can_drain = length <= _BodyReader.DRAIN_CAP
            if can_drain:
                _BodyReader._drain(handler, length)
            raise ApiError(
                413,
                "payload_too_large",
                "请求体 %d 字节超过上限 %d" % (length, settings.max_body_bytes),
                # 声明了但没读掉的请求体，连接不能复用。
                # 键名以 `_` 开头：这是给 `_fail()` 看的处置指令，
                # `_error_payload()` 会把它从对外响应里剔掉。
                _drain=("done" if can_drain else "skipped_too_large"),
            )
        chunks: list[bytes] = []
        remaining = length
        while remaining > 0:
            piece = handler.rfile.read(min(remaining, 65536))
            if not piece:
                break
            chunks.append(piece)
            remaining -= len(piece)
        return b"".join(chunks)


def _json_body(handler: BaseHTTPRequestHandler, settings: Settings) -> dict:
    # 优先用 `_route_post()` 提前读好的字节：请求体必须在任何拒绝分支之前就读掉，
    # 否则 401/503/404/429 这些"还没轮到解析请求体"的响应会留下未读字节，
    # 连接被 RST，客户端看到的是连接重置而不是那个错误码。
    raw = getattr(handler, "_raw_body", None)
    if raw is None:
        raw = _BodyReader.read(handler, settings)
    if not raw:
        return {}
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ApiError(400, "bad_json", "请求体不是合法的 UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ApiError(400, "bad_json", "请求体必须是 JSON 对象")
    return payload


# ---------------------------------------------------------------- 参数校验


def _int_field(payload: dict, key: str, default: int, *, minimum: int, maximum: int) -> int:
    if key not in payload or payload[key] is None:
        return default
    raw = payload[key]
    if isinstance(raw, bool):
        raise ApiError(400, "bad_param", f"{key} 必须是整数，收到布尔")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ApiError(400, "bad_param", f"{key} 必须是整数") from exc
    if value < minimum or value > maximum:
        # **不夹紧**：夹紧会让调用方以为自己的参数生效了。
        raise ApiError(400, "param_out_of_range", f"{key}={value} 超出 [{minimum}, {maximum}]")
    return value


# ---------------------------------------------------------------- Handler


class Handler(BaseHTTPRequestHandler):
    server_version = "arkrag-serve/1"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    # 由 `serve()` 注入
    state: AppState

    # ---------------------------------------------------------- 基础工具
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 - 覆盖基类
        """关掉默认的 stderr 访问日志。

        理由：默认日志会把完整请求行写进去，问句原文就是敏感内容（计划 J3
        "默认不记录敏感完整内容"）。要访问日志就开 `ARKRAG_SERVE_LOG_QUERY_TEXT`，
        那时由 `log_query()` 写结构化记录，而不是往 stderr 喷自由文本。
        """
        return

    def _client(self) -> str:
        return str(self.client_address[0] if self.client_address else "")

    def _send(self, status: int, payload: dict, *, extra_headers: dict | None = None) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):  # pragma: no cover - 客户端已断
            pass

    def _fail(self, exc: ApiError) -> None:
        self.state.stats["errors"] += 1
        headers = {}
        if exc.status == 429 and "retry_after_s" in exc.extra:
            headers["Retry-After"] = str(max(1, int(float(exc.extra["retry_after_s"]) + 0.999)))
        if exc.extra.get("_drain") == "skipped_too_large":
            # 请求体没读完，这条连接不能复用：让客户端知道要重开，
            # 否则下一个请求会读到上一条请求残留的字节。
            headers["Connection"] = "close"
            self.close_connection = True
        self._send(exc.status, _error_payload(exc, settings=self.state.settings), extra_headers=headers)

    def _guard(self, fn: Callable[[], None]) -> None:
        try:
            fn()
        except ApiError as exc:
            self._fail(exc)
        except Exception as exc:  # noqa: BLE001 - 兜底，绝不把堆栈发出去
            self.state.stats["errors"] += 1
            self._send(
                500,
                {
                    "error": {
                        "code": "internal",
                        # 只给异常类型名，不给 str(exc)：后者常含路径。
                        "message": f"内部错误（{type(exc).__name__}），详情见服务端日志",
                    }
                },
            )
            self._log_internal(exc)

    def _log_internal(self, exc: BaseException) -> None:
        try:
            print(
                f"[arkrag-serve] internal {type(exc).__name__}: "
                f"{sanitize_text(exc, settings=self.state.settings)}",
                file=sys.stderr,
            )
        except Exception:  # pragma: no cover
            pass

    # ---------------------------------------------------------- 鉴权
    def _require_admin(self) -> str:
        token = self.state.settings.admin_token
        if not token:
            # fail-closed：没配口令 = 管理接口关闭。默认放开是这类接口最常见的事故源。
            raise ApiError(503, "admin_disabled", "未配置 ARKRAG_ADMIN_TOKEN，管理接口关闭")
        header = self.headers.get("Authorization") or ""
        prefix = "Bearer "
        got = header[len(prefix):].strip() if header.startswith(prefix) else ""
        if not got or not secrets.compare_digest(got, token):
            raise ApiError(401, "unauthorized", "管理接口需要有效的 Bearer 令牌")
        return "admin"

    # ---------------------------------------------------------- 路由
    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        self._guard(self._route_get)

    def do_POST(self) -> None:  # noqa: N802
        self._guard(self._route_post)

    def _route_get(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        self.state.check_rate(self._client())
        if path == "/health":
            return self._health()
        if path == "/ready":
            return self._ready()
        if path == "/release/current":
            return self._release_current()
        if path == "/limits":
            return self._send(200, {"limits": self.state.settings.describe()})
        for prefix, handler in (
            ("/chunks/", self._chunk),
            ("/assertions/", self._assertion),
            ("/entities/", self._entity),
        ):
            if path.startswith(prefix):
                # 不能用路径分段匹配：chunk_id 形如 `obt/main/level_main_07-02_end#002`，
                # 本身带斜杠。所以取前缀之后的全部剩余部分并按 URL 解码。
                return handler(unquote(path[len(prefix):]))
        raise ApiError(404, "not_found", f"没有这个路径：{path}")

    def _route_post(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        # 请求体必须在**任何**拒绝分支之前读掉。
        #
        # 实测踩到的形态：`_require_admin()` 在 `_json_body()` 之前跑，所以没配口令时
        # 直接 503、请求体一字未读，连接被 RST，客户端看到的是
        # `ConnectionAbortedError: [WinError 10053]` 而不是那条 503。
        # 单跑那条测试能过、全量跑偶发失败——因为是否存在未读字节取决于时序。
        #
        # 这既不是 413 也不是 401 的个性问题，而是所有"还没到解析请求体就拒绝"的
        # 错误路径的共性问题。放在路由入口统一读一次，就不用逐个拒绝点记得做这件事。
        # 读失败（413/400）时异常直接抛给 `_guard`，`_json_body()` 不会再读第二次。
        self._raw_body = _BodyReader.read(self, self.state.settings)
        self.state.check_rate(self._client())
        if path == "/query":
            return self._query()
        if path == "/admin/rollback":
            return self._admin_rollback()
        if path == "/admin/evaluate":
            return self._admin_evaluate()
        raise ApiError(404, "not_found", f"没有这个路径：{path}")

    # ---------------------------------------------------------- 探针
    def _health(self) -> None:
        """存活探针：只说明进程还能响应。**故意不查任何数据。**

        与 `/ready` 分开的理由：把二者合并的部署里，"活着但数据坏了"会被报成健康。
        """
        self._send(
            200,
            {
                "ok": True,
                "uptime_s": round(time.time() - self.state.started_at, 3),
                "pid": os.getpid(),
                "served_generation_id": self.state.served_generation_id,
                "stats": dict(self.state.stats),
                "note": "存活探针不代表就绪，请查 /ready",
            },
        )

    def _ready(self) -> None:
        result = self.state.readiness()
        self._send(200 if result["ready"] else 503, result)

    def _release_current(self) -> None:
        try:
            pointer = release.read_current()
        except release.ReleaseError as exc:
            raise ApiError(503, "current_unreadable", str(exc)) from exc
        served = self.state.served_generation_id or str((pointer or {}).get("generation_id") or "")
        self._send(
            200,
            {
                "current": pointer,
                "served_generation_id": served,
                "serving_is_current": bool(pointer)
                and str(pointer.get("generation_id") or "") == served,
                "read_at": _now(),
            },
        )

    # ---------------------------------------------------------- 单记录
    def _lookup_chunk(self, chunk_id: str) -> dict:
        served = self.state.served()
        rec = served.loaded["chunks"].get(chunk_id)
        if rec is None:
            raise ApiError(404, "chunk_not_found", f"没有这个块：{chunk_id}")
        return rec

    def _chunk(self, chunk_id: str) -> None:
        if not chunk_id:
            raise ApiError(400, "bad_request", "缺少 chunk id")
        rec = self._lookup_chunk(chunk_id)
        payload = _chunk_payload(rec, settings=self.state.settings)
        payload["generation_id"] = self.state.served().generation_id
        self._send(200, payload)

    def _assertion(self, assertion_id: str) -> None:
        if not assertion_id:
            raise ApiError(400, "bad_request", "缺少 assertion id")
        served = self.state.served()
        # 走 `Retriever` 已经建好的按 id 索引：它按 `assertion_id` 排序过，
        # 且已剔掉空 id 的记录。这里再线性扫一遍等于把同一件事做第二次。
        rec = served.pipeline["retriever"].assertions_by_id.get(assertion_id)
        if rec is None:
            raise ApiError(404, "assertion_not_found", f"没有这条断言：{assertion_id}")
        evidence, cut = _clip_text(
            rec.get("evidence_text") or "", self.state.settings.max_record_chars,
            settings=self.state.settings,
        )
        self._send(
            200,
            {
                "generation_id": served.generation_id,
                "assertion_id": str(rec.get("assertion_id") or ""),
                "subject_id": rec.get("subject_id"),
                "predicate": rec.get("predicate"),
                "object_id": rec.get("object_id"),
                "direction": rec.get("direction"),
                "chunk_id": rec.get("chunk_id"),
                "doc_id": rec.get("doc_id"),
                "scene_id": rec.get("scene_id"),
                "evidence_start": rec.get("evidence_start"),
                "evidence_end": rec.get("evidence_end"),
                "evidence_text": evidence,
                "evidence_truncated": cut,
                "confidence": rec.get("confidence"),
                "extraction": rec.get("extraction"),
                "untrusted": True,
            },
        )

    def _entity(self, entity_id: str) -> None:
        """按 id 查实体，**也接受规范名与权威别名**。

        为什么不止接受 id：实体 id 是 `identity.stable_id()` 算出来的稳定摘要
        （形如 `e-c7091cbe8cb6fe66`），人记不住、也搜不到。只认 id 的接口实际上
        只有机器能调——而"凯尔希有哪些关系"是最自然的一次人工查询。
        名字与别名的解析**复用图里已有的 `alias_index`**，不另建一份映射：
        那份表由 `identity.CanonicalEntity.to_dict()` 定，已排除待审/拒绝别名。
        """
        if not entity_id:
            raise ApiError(400, "bad_request", "缺少 entity id")
        served = self.state.served()
        graph = served.loaded["graph"]
        by_id = {str(e.get("id") or ""): e for e in graph.get("entities") or []}
        by_name = {str(e.get("name") or ""): e for e in graph.get("entities") or []}
        resolved_by = "id"
        rec = by_id.get(entity_id)
        if rec is None:
            rec = by_name.get(entity_id)
            resolved_by = "name"
        if rec is None:
            alias_owner = str((graph.get("alias_index") or {}).get(entity_id) or "")
            rec = by_id.get(alias_owner) if alias_owner else None
            resolved_by = "alias"
        if rec is None:
            raise ApiError(404, "entity_not_found", f"没有这个实体：{entity_id}")
        entity_id = str(rec.get("id") or "")
        edges = [
            e
            for e in graph.get("edges") or []
            if str(e.get("subject_id") or "") == entity_id
            or str(e.get("object_id") or "") == entity_id
        ]
        # 边可能很多（热门实体上千条）。按 support_count 降序给前 N 条，
        # 并显式告诉调用方总数——截断不说明就等于丢数据。
        limit = self.state.settings.max_top_k
        edges.sort(key=lambda e: (-int(e.get("support_count") or 0), str(e.get("edge_id") or "")))
        keep = edges[:limit]
        self._send(
            200,
            {
                "generation_id": served.generation_id,
                "resolved_by": resolved_by,
                "entity": {
                    "id": str(rec.get("id") or ""),
                    "name": rec.get("name"),
                    "type": rec.get("type"),
                    "desc": _clip_text(
                        rec.get("desc") or "", self.state.settings.max_record_chars,
                        settings=self.state.settings,
                    )[0],
                    "aliases": rec.get("aliases"),
                    "pending_aliases": rec.get("pending_aliases"),
                    "chunks": (rec.get("chunks") or [])[: self.state.settings.max_top_k],
                    "untrusted": True,
                },
                "edges": keep,
                "edges_total": len(edges),
                "edges_returned": len(keep),
                "edges_truncated": len(keep) < len(edges),
            },
        )

    # ---------------------------------------------------------- 查询
    def _query(self) -> None:
        settings = self.state.settings
        payload = _json_body(self, settings)
        question = payload.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ApiError(400, "bad_request", "question 必须是非空字符串")
        if len(question) > query_mod.QUERY_MAX_CHARS:
            # 与 `query.analyze_query()` 用的是同一个常量。两处各写一个数就会
            # 出现"HTTP 层放过去了、分析层再拒"的双重判定。
            raise ApiError(
                400, "param_out_of_range",
                f"question 长度 {len(question)} 超过上限 {query_mod.QUERY_MAX_CHARS}",
            )
        top_k = _int_field(payload, "top_k", 20, minimum=1, maximum=settings.max_top_k)
        depth = _int_field(payload, "graph_depth", 1, minimum=0, maximum=settings.max_graph_depth)
        ctx_tokens = _int_field(
            payload, "max_context_tokens", settings.max_context_tokens,
            minimum=200, maximum=settings.max_context_tokens,
        )

        served = self.state.served()
        pipe = dict(served.pipeline)
        pipe["top_k"] = top_k
        pipe["graph_depth"] = depth
        if ctx_tokens != served.pipeline["retriever"].budget.get("max_context_tokens"):
            budget = dict(served.pipeline["retriever"].budget)
            budget["max_context_tokens"] = ctx_tokens
            pipe["retriever"] = _with_budget(served.pipeline["retriever"], budget)
            # `Retriever` 有 `query_context`，重建实例会丢掉它 → 会把
            # `analyze_query` 拉回"每问重建图级结构"（H2 实测每问多 98 ms）。
            pipe["retriever"].query_context = served.pipeline["retriever"].query_context

        query_id = "q-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + secrets.token_hex(4)
        timings: dict = {}
        t0 = time.perf_counter()

        def work() -> dict:
            result = pipeline_mod.ask_once(pipe, question, timings=timings)
            return result

        result = self.state.run_bounded(work)
        wall_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        self.state.stats["queries"] += 1

        # 引用校验走 `answer.verify_citations`：与 CLI 用同一个函数，
        # 服务端不另写一套"什么算合格引用"。
        verified = pipeline_mod.answer_mod.verify_citations(
            result["answer"], served.loaded["chunks"]
        )
        ans = verified
        s_stats = result["search"].get("stats") or {}
        e_stats = result["expand"].get("stats") or {}
        c_stats = result["context"].get("stats") or {}

        body = {
            "query_id": query_id,
            "generation_id": served.generation_id,
            "question": question,
            "answer": {
                "text": ans.get("answer"),
                "citations": ans.get("citations") or [],
                "confidence": ans.get("confidence"),
                "abstained": bool(ans.get("abstained")),
                "abstain_reason": ans.get("abstain_reason"),
                "warnings": ans.get("warnings") or [],
                "verification": ans.get("verification"),
                "versions": ans.get("versions"),
            },
            "context": {
                "context_digest": result["context"].get("context_digest"),
                "units": [
                    {
                        "chunk_id": u.get("chunk_id"),
                        "score": u.get("score"),
                        "kind": u.get("kind") or u.get("source"),
                    }
                    for u in (result["context"].get("units") or [])
                ],
                "stats": c_stats,
            },
            "stats": {
                "candidates": s_stats.get("candidates"),
                "top_score": s_stats.get("top_score"),
                "bm25_terms": s_stats.get("bm25_terms"),
                "bm25_top_score": s_stats.get("bm25_top_score"),
                "entity_channels": s_stats.get("entity_channels"),
                "entity_top_score": s_stats.get("entity_top_score"),
                "speaker_boosted": s_stats.get("speaker_boosted"),
                "candidate_pool_depth": s_stats.get("candidate_pool_depth"),
                "channels": [
                    {"channel": src.get("channel"), "score": src.get("score")}
                    for hit in (result["search"].get("hits") or [])[:1]
                    for src in (hit.get("sources") or [])
                ],
                "expand_nodes": e_stats.get("nodes_visited"),
                "expand_edges": e_stats.get("expansions"),
                "expand_truncated_reason": e_stats.get("truncated_reason"),
                "context_tokens": c_stats.get("tokens_used"),
                "latency_ms": {**timings, "wall_ms": wall_ms},
                # 本机没有 LLM 端点，成本无从计算。写 null + 原因，
                # **不写 0**——0 会被当成"免费"，而真相是"没测"。
                "token_cost": None,
                "token_cost_reason": "本机无 LLM 端点，回答器为抽取式，无 token 计费",
            },
            "hits": [
                {
                    "chunk_id": h.get("chunk_id"),
                    "score": h.get("score"),
                    "channels": [s.get("channel") for s in h.get("sources") or []],
                }
                for h in (result["search"].get("hits") or [])
            ],
            "untrusted_content": True,
            "limits": {
                "top_k": top_k,
                "graph_depth": depth,
                "max_context_tokens": ctx_tokens,
            },
        }
        body, dropped = _fit_payload(body, settings.max_response_bytes)
        if dropped:
            body.setdefault("truncated", dropped)

        self._log_query_record(
            query_id=query_id,
            question=question,
            generation_id=served.generation_id,
            stats=body["stats"],
            answer=ans,
            verified=verified,
            top_k=top_k,
            depth=depth,
        )
        self._send(200, body)

    def _log_query_record(
        self,
        *,
        query_id: str,
        question: str,
        generation_id: str,
        stats: dict,
        answer: dict,
        verified: dict,
        top_k: int,
        depth: int,
    ) -> None:
        """J3 查询记录。**默认不写问句原文**，只写长度与摘要。

        为什么要摘要而不是原文：这个日志会长期留存，而问句本身就是用户内容；
        计划 J3 明确"默认不记录敏感完整内容"。要排查具体问题，用摘要去比对
        客户端手里的原问句即可。
        """
        settings = self.state.settings
        record = {
            "at": _now(),
            "query_id": query_id,
            "generation_id": generation_id,
            "query_chars": len(question),
            "query_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest()[:16],
            "top_k": top_k,
            "graph_depth": depth,
            "candidates": stats.get("candidates"),
            "channels": stats.get("channels"),
            "expand_nodes": stats.get("expand_nodes"),
            "expand_edges": stats.get("expand_edges"),
            "context_tokens": stats.get("context_tokens"),
            "latency_ms": stats.get("latency_ms"),
            "prompt_version": (answer.get("versions") or {}).get("prompt_version"),
            "answerer": (answer.get("versions") or {}).get("answerer"),
            "citation_verification": {
                "checked": (verified.get("verification") or {}).get("checked"),
                "accepted": (verified.get("verification") or {}).get("accepted"),
                "rejected": (verified.get("verification") or {}).get("rejected"),
            },
            "abstained": bool(answer.get("abstained")),
            "abstain_reason": answer.get("abstain_reason"),
            "token_cost": None,
        }
        if settings.log_query_text:
            record["question"] = question
        self.state.log_query(record)

    # ---------------------------------------------------------- 管理
    def _admin_rollback(self) -> None:
        actor = self._require_admin()
        payload = _json_body(self, self.state.settings)
        target = payload.get("generation_id")
        if not isinstance(target, str) or not target.strip():
            raise ApiError(400, "bad_request", "generation_id 必填")
        target = target.strip()
        if not release.GEN_ID_RE.match(target):
            raise ApiError(400, "bad_request", "generation_id 形状不对（应形如 g-YYYYMMDDTHHMMSSZ-xxxxxx）")

        before = (release.read_current() or {}).get("generation_id")
        try:
            pointer = release.rollback(target)
        except release.ReleaseError as exc:
            self.state.audit(
                "rollback", actor=actor, result="failed",
                from_generation=before, to_generation=target,
                error=sanitize_text(exc, settings=self.state.settings),
            )
            raise ApiError(409, "rollback_refused", str(exc)) from exc
        self.state.audit(
            "rollback", actor=actor, result="ok",
            from_generation=before, to_generation=pointer.get("generation_id"),
            gate_report_hash=pointer.get("gate_report_hash"),
        )
        # 回滚后服务端缓存的世代必须失效：否则"指针切了、服务还在用旧的那一代"，
        # 而且 /ready 会因为 `serving_is_current` 变 false 才被发现——太晚了。
        self.state.invalidate_served()
        self._send(200, {"rolled_back": True, "current": pointer, "invalidated_served": True})

    def _admin_evaluate(self) -> None:
        actor = self._require_admin()
        settings = self.state.settings
        payload = _json_body(self, settings)
        limit = _int_field(payload, "limit", 20, minimum=1, maximum=MAX_EVAL_LIMIT)
        raw_gold = payload.get("gold_set") or "data/goldset/gold-v1.json"
        if not isinstance(raw_gold, str):
            raise ApiError(400, "bad_request", "gold_set 必须是字符串路径")
        gold = _resolve_repo_path(raw_gold)
        if gold is None:
            # 只允许仓库内路径：否则 `/admin/evaluate` 变成"任意文件读取器"。
            raise ApiError(400, "bad_request", "gold_set 必须是仓库内路径")
        if not gold.exists():
            raise ApiError(400, "bad_request", f"gold_set 不存在：{config.rel_to_root(gold)}")

        served_gid = self.state.served_generation_id or self.state.resolve_generation_id()

        def work() -> dict:
            ask_mod = _ask_module()
            return ask_mod.evaluate(gold, limit=limit, generation_id=served_gid)

        t0 = time.perf_counter()
        report = self.state.run_bounded(work, timeout_s=max(settings.timeout_s, 600.0))
        wall_s = round(time.perf_counter() - t0, 3)
        self.state.audit(
            "evaluate", actor=actor, result="ok",
            generation_id=served_gid, gold_set=config.rel_to_root(gold),
            limit=limit, wall_s=wall_s,
            citation_verify_failures=report.get("citation_verify_failures"),
        )
        self._send(
            200,
            {
                "generation_id": report.get("generation_id"),
                "gold_set": config.rel_to_root(gold),
                "items_evaluated": report.get("items_evaluated"),
                "metrics": report.get("metrics"),
                "citation_verify_failures": report.get("citation_verify_failures"),
                "wall_s": wall_s,
            },
        )


def _with_budget(retriever, budget: dict):
    """复制一个 `Retriever`，只换预算。

    为什么复制而不原地改：`Retriever` 是并发共享对象，原地改预算会让同时进行的
    另一个请求读到别人的预算——"每请求可调预算"就会变成"最后一个请求说了算"。
    """
    from . import retrieve as retrieve_mod

    clone = retrieve_mod.Retriever(
        retriever.index,
        retriever.graph,
        retriever.adjacency,
        chunks=retriever.chunks,
        assertions=retriever.assertions_by_id.values() if hasattr(retriever, "assertions_by_id") else None,
        budget=budget,
        query_context=getattr(retriever, "query_context", None),
    )
    return clone


def _resolve_repo_path(raw: str) -> Path | None:
    """把一个字符串解析成仓库内的绝对路径；仓库外或越界返回 None。"""
    path = Path(raw)
    if not path.is_absolute():
        path = config.ROOT / path
    try:
        resolved = path.resolve()
        root = config.ROOT.resolve()
    except OSError:
        return None
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


_ASK_MODULE: Any = None
_ASK_LOCK = threading.Lock()


def _ask_module():
    """按路径加载 `scripts/ask.py`（它不是包的一部分）。

    为什么复用脚本而不是把评测逻辑搬进包内：`evaluate()` 与
    `_stratified_sample()` 是**同一个判定**的两半——抽样口径与指标口径必须同时
    一致，否则服务端跑出来的数会和 `ask.py --evaluate` 不可比。搬一份出来就会
    出现两份定义（本项目为此出过三次事故）。所以这里只是"按路径加载同一个文件"。
    """
    global _ASK_MODULE
    with _ASK_LOCK:
        if _ASK_MODULE is not None:
            return _ASK_MODULE
        path = config.ROOT / "scripts" / "ask.py"
        spec = importlib.util.spec_from_file_location("arkrag_ask_under_serve", path)
        if spec is None or spec.loader is None:  # pragma: no cover
            raise ApiError(500, "ask_module_missing", "找不到 scripts/ask.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _ASK_MODULE = mod
        return mod


# `_check_chunks_cover` 用到的索引模块已在文件顶部导入（`index as index_mod`），
# 这里不再做延迟注入：那会造出"同一个模块挂在两个名字上"的隐患。


# ---------------------------------------------------------------- 装配


def make_handler(state: AppState) -> type[Handler]:
    class _Bound(Handler):
        pass

    _Bound.state = state
    return _Bound


def serve(settings: Settings | None = None, *, generation_id: str = "") -> ThreadingHTTPServer:
    """建好并**不启动**的服务对象。调用方自己 `serve_forever()`。

    分成"建"与"跑"两步，是为了测试能拿到 `server.server_address` 里的实际端口
    （`port=0` 时由系统分配），再用真实 HTTP 请求打它——而不是绕过 HTTP 直接调
    handler，那样测不到路由、状态码与响应头。
    """
    settings = settings or Settings.from_env()
    state = AppState(settings, generation_id=generation_id)
    handler = make_handler(state)
    httpd = ThreadingHTTPServer((settings.host, settings.port), handler)
    httpd.daemon_threads = True
    httpd.state = state  # type: ignore[attr-defined]
    return httpd


__all__ = [
    "ApiError",
    "AppState",
    "DEFAULT_CONCURRENCY",
    "DEFAULT_MAX_BODY_BYTES",
    "DEFAULT_MAX_GRAPH_DEPTH",
    "DEFAULT_MAX_RECORD_CHARS",
    "DEFAULT_MAX_RESPONSE_BYTES",
    "DEFAULT_MAX_TOP_K",
    "DEFAULT_RATE_LIMIT_CALLS",
    "DEFAULT_RATE_LIMIT_WINDOW_S",
    "DEFAULT_TIMEOUT_S",
    "Handler",
    "MAX_EVAL_LIMIT",
    "Settings",
    "make_handler",
    "sanitize_text",
    "serve",
]
