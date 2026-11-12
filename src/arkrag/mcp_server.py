"""A2：MCP 协议实现（stdio + streamable HTTP）。**只用标准库。**

## 协议版本：本实现覆盖哪一版，为什么不覆盖更新那版

按 <https://modelcontextprotocol.io/specification/2025-06-18> 与
`2025-11-25` 两份修订实现（**两者语义相同**，都是 `initialize` 握手时代）：

* `initialize` / `notifications/initialized` 握手，版本协商（对不上就回服务端
  最新支持的那版，客户端自己决定是否断开）；
* `tools/list`（确定性顺序）、`tools/call`、`ping`；
* JSON-RPC 2.0 错误码 `-32700 / -32600 / -32601 / -32602 / -32603`；
* stdio：一行一条消息、**stdout 只许出现协议消息**、日志一律 stderr；
* streamable HTTP：`POST /mcp` 收 JSON-RPC 回 `application/json`，
  `Mcp-Session-Id` 管会话，`GET /mcp` 回 405 明说"本服务不提供 SSE 流"。

**未实现 `2026-07-28` 及之后的"无状态"修订**（该版取消了 `initialize`、
取消了 `Mcp-Session-Id`，改由每个请求的 `_meta` 携带版本与能力）。原因不是
"做得出来但不做"，而是它是**另一套语义**：没有握手就没有"会话协商到的版本"
这个状态，能力表要改成每请求判定。把两者混在一个实现里最容易出的错是
"看起来支持新协议，实际按旧语义处理参数"——那种半支持比明确不支持更糟。
客户端若按新协议来，会在 `initialize` 上收到明确的 `-32601`，
而不是被静默当成旧协议处理。详见 `docs/api-agent-接入规范-20261112.md` 的「未实现」。

## 为什么这一层不重写工具逻辑

工具定义与执行在 `arkrag.agent_api`（`TOOLS` 是唯一事实源）。这里只做
**传输与生命周期**：把字节读成 JSON-RPC、把 JSON-RPC 交给
`agent_api.handle_jsonrpc()`、把结果写回字节。任何"某个工具怎么校验参数"
的判定都不在这个文件里。

## stdio 的 stdout 纪律（为什么写成这样）

MCP 客户端把子进程的 stdout 当协议通道解析。任何一行非协议文本——哪怕是一句
`print("启动成功")`——都会让客户端解析失败、连接作废，而错误信息通常只说
"JSON 解析失败"，让人以为是自己发错了请求。

所以这里：

1. **不 print**。唯一写 stdout 的出口是 `_write_message()`，它只写 JSON 行。
2. 所有日志走 `_log()` → stderr。
3. 每行写完 `flush()`：stdio 是管道，不 flush 等于没发（客户端会一直等）。
4. 输出前把消息压成**单行**（`separators=(",", ":")`），因为协议规定消息内
   不得含换行。
"""
from __future__ import annotations

import json
import os
import secrets
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, TextIO
from urllib.parse import urlparse

from . import agent_api
from . import server as server_mod

# ---------------------------------------------------------------- 常量

#: stdio 单行上限。一问一答的最大载荷量级在几十 KB（响应上限 256 KiB），
#: 256 KiB 给 JSON-RPC 信封留足余量；超过这个长度的行不是本协议的正常输入。
MAX_STDIO_LINE_BYTES = 256 * 1024

#: MCP 的 HTTP 端点路径（Streamable HTTP 要求**单一**端点）。
MCP_ENDPOINT = "/mcp"

#: 原生 HTTP 入口（对应 `agent_api.build_openapi()` 里的 paths）。
#: 三种接入方式共用同一份 `TOOLS`，所以这些路径不需要单独维护。
TOOLS_PATH_PREFIX = "/tools/"
RPC_PATH = "/rpc"
OPENAPI_PATH = "/openapi.json"

#: 会话空闲上限（秒）。超过就回收：会话 ID 是唯一把状态挂到客户端的把手，
#: 不回收的话每个连过的客户端都会在内存里留一条。
DEFAULT_SESSION_TTL_S = 1800.0

#: 会话数上限。**有上限**是刻意的：会话表是内存里的字典，无上限等于一个
#: 可以远程撑爆进程的接口。
MAX_SESSIONS = 512

SESSION_HEADER = "Mcp-Session-Id"
PROTOCOL_HEADER = "MCP-Protocol-Version"


def _log(message: str) -> None:
    """唯一的日志出口。**永远 stderr**（stdio 下 stdout 是协议通道）。

    Windows 控制台默认用 cp936 之类的本地编码。中文日志写到那里，若 stderr 被
    重定向成管道或文件，`print` 会按本地编码编码，读端按 UTF-8 解就得到一个
    `UnicodeDecodeError`——**日志本身把宿主搞崩**是最没必要的一种事故
    （实测：`subprocess(..., text=True, encoding="utf-8")` 读 stderr 直接抛）。

    所以这里先把 stderr 重配成 UTF-8；做不到（非文本流）就退化成
    `errors="backslashreplace"`，宁可日志里有转义序列，也不让写日志失败。
    """
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
    except (AttributeError, OSError, ValueError):  # pragma: no cover - 非文本流
        pass
    try:
        print(f"[arkrag-mcp] {message}", file=sys.stderr, flush=True)
    except Exception:  # pragma: no cover - stderr 也该能写，兜底而已
        pass


def encode_message(message: Any) -> str:
    """压成**单行** JSON。协议规定 stdio 消息内不得含换行。"""
    return json.dumps(message, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


# ---------------------------------------------------------------- 会话（仅 HTTP）


@dataclass
class Session:
    session_id: str
    created_at: float = field(default_factory=time.time)
    touched_at: float = field(default_factory=time.time)
    protocol_version: str = ""
    initialized: bool = False

    def touch(self) -> None:
        self.touched_at = time.time()


class SessionStore:
    """HTTP 传输的会话表。

    ## 为什么要会话

    Streamable HTTP 的规范允许服务端在 `initialize` 响应里发 `Mcp-Session-Id`，
    之后要求客户端在每个请求上带着它。ArkRAG 的服务状态本来是无状态的
    （每个请求都重新读盘上事实），但**版本协商结果**是会话级的：一次会话协商到
    哪个协议版本，后续请求就该按哪个版本答复。没有会话就没有地方记这件事，
    于是只能"每次请求都重新猜"，那样协商等于没有发生。

    ## 为什么要有上限与 TTL

    表在内存里。没有上限 = 一个能远程把进程撑爆的接口；没有 TTL = 每个连过
    一次的客户端永久占一行。两者都是"看起来能用、长期就是事故"的形态。
    """

    def __init__(self, *, ttl_s: float = DEFAULT_SESSION_TTL_S, capacity: int = MAX_SESSIONS) -> None:
        self.ttl_s = float(ttl_s)
        self.capacity = int(capacity)
        self._lock = threading.Lock()
        self._rows: dict[str, Session] = {}

    def _reap(self, now: float) -> None:
        dead = [sid for sid, s in self._rows.items() if now - s.touched_at > self.ttl_s]
        for sid in dead:
            self._rows.pop(sid, None)

    def create(self, *, protocol_version: str) -> Session:
        now = time.time()
        with self._lock:
            self._reap(now)
            if len(self._rows) >= self.capacity:
                # 满了就淘汰**最久没动**的那个，而不是拒绝新会话：
                # 拒绝会让"最多同时 N 个客户端"变成硬失败，淘汰只是让最闲的那个重连。
                oldest = min(self._rows.values(), key=lambda s: s.touched_at)
                self._rows.pop(oldest.session_id, None)
            sid = secrets.token_urlsafe(24)
            session = Session(
                session_id=sid, created_at=now, touched_at=now, protocol_version=protocol_version
            )
            self._rows[sid] = session
            return session

    def get(self, session_id: str) -> Session | None:
        now = time.time()
        with self._lock:
            self._reap(now)
            session = self._rows.get(str(session_id or ""))
            if session is not None:
                session.touch()
            return session

    def drop(self, session_id: str) -> bool:
        with self._lock:
            return self._rows.pop(str(session_id or ""), None) is not None

    def __len__(self) -> int:
        with self._lock:
            self._reap(time.time())
            return len(self._rows)


# ---------------------------------------------------------------- stdio 传输


class StdioTransport:
    """一行一条 JSON-RPC 消息的 stdio 传输。

    `reader` / `writer` 可注入，这样测试能直接喂字符串、收字符串，
    不必真起子进程（真起子进程也能测，但那样断言"stdout 只有协议行"就要靠
    管道读取的时序，容易变成 flaky）。
    """

    def __init__(self, state: server_mod.AppState, *, reader: TextIO | None = None,
                 writer: TextIO | None = None) -> None:
        self.state = state
        self._reader = reader
        self._writer = writer
        self._lock = threading.Lock()

    # ---- 输入输出
    def _input(self) -> TextIO:
        if self._reader is not None:
            return self._reader
        stream = sys.stdin
        # Windows 上默认编码可能是 cp936；MCP 消息必须是 UTF-8。
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, OSError):  # pragma: no cover - 非文本流
            pass
        return stream

    def _output(self) -> TextIO:
        if self._writer is not None:
            return self._writer
        stream = sys.stdout
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, OSError):  # pragma: no cover
            pass
        return stream

    def write_message(self, message: Any) -> None:
        """**唯一**写 stdout 的出口。只写单行 JSON，写完立刻 flush。"""
        line = encode_message(message)
        with self._lock:
            out = self._output()
            out.write(line + "\n")
            try:
                out.flush()
            except (OSError, ValueError):  # pragma: no cover - 管道已断
                pass

    # ---- 主循环
    def handle_line(self, raw: str) -> list[dict]:
        """处理一行输入，返回要写回的消息列表（可能为空：通知/坏通知）。

        抽成纯函数（输入字符串 → 输出消息列表）是为了让"stdout 到底被写了什么"
        可以被逐条断言，而不是靠读管道去猜。
        """
        text = raw.strip()
        if not text:
            return []
        if len(text.encode("utf-8")) > MAX_STDIO_LINE_BYTES:
            _log(f"拒绝超长输入行（>{MAX_STDIO_LINE_BYTES} 字节）")
            return [agent_api.parse_error_response("输入行超过长度上限")]
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            # 位置信息足够定位，**不回显原始字节**（回显会把畸形输入变成日志污染源）。
            return [agent_api.parse_error_response(f"line {exc.lineno} column {exc.colno}")]
        response = agent_api.handle_jsonrpc(payload, self.state)
        if response is None:
            return []
        if isinstance(response, list):
            return list(response)
        return [response]

    def serve_forever(self) -> int:
        """读到 EOF 为止。返回进程退出码（0 = 正常关闭）。"""
        reader = self._input()
        _log(f"stdio 传输就绪（协议版本 {agent_api.LATEST_PROTOCOL_VERSION} 起）")
        try:
            while True:
                line = reader.readline()
                if not line:
                    break
                for message in self.handle_line(line):
                    self.write_message(message)
        except KeyboardInterrupt:  # pragma: no cover - 交互式
            pass
        finally:
            _log("stdio 传输结束")
        return 0


# ---------------------------------------------------------------- HTTP 传输


def _accept_allows_json(accept: str | None) -> bool:
    """`Accept` 是否允许我们回 `application/json`。

    规范要求客户端列出 `application/json` 与 `text/event-stream` **两者**。
    本服务只实现前者（不实现 SSE，见模块 docstring）。所以：

    * 没写 `Accept`（或空）→ 放行。规范要求写，但缺头不是"拒绝服务"的理由；
      我们回的是合法 JSON，客户端读得懂。
    * 写了且包含 `application/json` 或 `*/*` 或 `application/*` → 放行。
    * 只列了 `text/event-stream` → 返回 False（调用方回 406 并说明未实现 SSE）。
    """
    raw = str(accept or "").strip().lower()
    if not raw:
        return True
    for part in raw.split(","):
        media = part.split(";", 1)[0].strip()
        if media in ("application/json", "*/*", "application/*"):
            return True
        if media.endswith("+json"):
            return True
    return False


def _origin_allowed(origin: str | None, *, allow_any: bool = False) -> bool:
    """DNS rebinding 防护（规范对本传输的 MUST）。

    默认只接受"本机来源"或没有 `Origin`（非浏览器客户端不发这个头）。
    判定用 `urlparse` 取 host 再比对回环地址字面量，不做后缀匹配——
    `http://127.0.0.1.evil.com` 这类如果按 `startswith` 判就会混过去。
    """
    if allow_any:
        return True
    raw = str(origin or "").strip()
    if not raw:
        return True
    try:
        host = (urlparse(raw).hostname or "").lower()
    except ValueError:
        return False
    if not host:
        return False
    if host in ("localhost", "::1", "127.0.0.1"):
        return True
    # 127.0.0.0/8 整段都算本机
    if host.startswith("127.") and host.count(".") == 3:
        try:
            return all(0 <= int(part) <= 255 for part in host.split("."))
        except ValueError:
            return False
    return False


class McpHttpHandler(BaseHTTPRequestHandler):
    """Streamable HTTP 的 MCP 端点（`POST/GET/DELETE /mcp`）。

    协议要点（逐条对应规范）：

    * `POST`：请求 → `application/json` 单对象响应；通知/响应 → `202` 无体；
      批量请求 → JSON 数组。
    * `GET`：规范允许服务端回 `405 Method Not Allowed` 来表示"本端点不提供
      SSE 流"。本服务就回 405，并在体里说明原因（未实现 SSE 是**已知缺口**，
      不是错误配置）。
    * `DELETE`：客户端显式结束会话 → 丢会话，回 200。
    * `Mcp-Session-Id`：`initialize` 的响应里发；之后的请求必须带；不带 → 400；
      带了但不认识（过期/被删）→ 404（客户端收到 404 要重新 `initialize`）。
    * `MCP-Protocol-Version`：带了就必须是我们支持的版本，否则 400。
    * `Origin`：不是本机来源 → 403。
    """

    server_version = "arkrag-mcp/1"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    # 由 `serve_http()` 注入
    state: server_mod.AppState
    sessions: SessionStore
    allow_any_origin: bool = False

    # ---- 基础设施
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 - 覆盖基类
        """关掉默认访问日志：默认格式会把完整请求行写进 stderr，
        而 MCP 请求体里可能带用户问句。要排查就开 `_log()` 的结构化输出。"""
        return

    def _send_json(self, status: int, payload: Any, *, extra: dict | None = None) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):  # pragma: no cover - 客户端已断
            pass

    def _send_empty(self, status: int, *, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()

    def _fail(self, status: int, code: int, message: str, *, extra: dict | None = None) -> None:
        self._send_json(status, agent_api.jsonrpc_error(None, code, message), extra=extra)

    def _discard_body(self) -> None:
        """读掉并丢弃请求体（用于"还没解析就拒绝"的分支）。

        为什么不直接 `return` 了事：未读字节会让 `close()` 在 Windows 上发 RST
        而不是 FIN，客户端内核把已到达的错误响应一起丢掉，于是调用方看到的是
        "连接被重置"——一个把 403/404/406 全部伪装成"服务挂了"的形态。
        """
        try:
            self._read_body()
        except (OverflowError, ValueError):
            # 超限或长度非法时本来就没读完，但那条响应自己会说明原因。
            pass

    def _read_body(self) -> bytes:
        raw_len = self.headers.get("Content-Length")
        if raw_len is None:
            return b""
        try:
            length = int(raw_len)
        except ValueError:
            raise ValueError("Content-Length 不是整数") from None
        if length < 0:
            raise ValueError("Content-Length 为负")
        if length > self.state.settings.max_body_bytes:
            # 读掉再拒：只回 413 就关连接会让客户端收到 RST 而不是那条 413
            # （`server._BodyReader` 的模块注释里有实测记录）。
            remaining = min(length, 1024 * 1024)
            while remaining > 0:
                piece = self.rfile.read(min(remaining, 65536))
                if not piece:
                    break
                remaining -= len(piece)
            raise OverflowError(f"请求体 {length} 字节超过上限 {self.state.settings.max_body_bytes}")
        pieces = []
        remaining = length
        while remaining > 0:
            piece = self.rfile.read(min(remaining, 65536))
            if not piece:
                break
            pieces.append(piece)
            remaining -= len(piece)
        return b"".join(pieces)

    # ---- 路由
    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        if not _origin_allowed(self.headers.get("Origin"), allow_any=self.allow_any_origin):
            return self._fail(403, agent_api.INVALID_REQUEST, "Origin 不被允许（仅本机来源）")
        path = urlparse(self.path).path
        if path == OPENAPI_PATH:
            # OpenAPI 文档必须由**运行中的服务**给出，而不是一份可能过期的
            # 静态文件：`build_openapi()` 从 `TOOLS` 生成，所以这里发出去的
            # 路径与真正能调的工具永远一致。
            body = agent_api.openapi_json(settings=self.state.settings)
            raw = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):  # pragma: no cover
                pass
            return None
        if path != MCP_ENDPOINT:
            return self._fail(404, agent_api.INVALID_REQUEST, f"没有这个路径：{self.path}")
        # 规范：服务端可以不提供 SSE 流，此时回 405。**明确说原因**，
        # 否则客户端只会看到"这个方法不能用"，猜不出是版本还是特性缺失。
        return self._fail(
            405,
            agent_api.INVALID_REQUEST,
            "本服务未实现 SSE 流（GET 需要 text/event-stream）；"
            "请用 POST 发送 JSON-RPC，服务端会以 application/json 单条响应。",
            extra={"Allow": "POST, DELETE"},
        )

    def do_DELETE(self) -> None:  # noqa: N802
        if not _origin_allowed(self.headers.get("Origin"), allow_any=self.allow_any_origin):
            return self._fail(403, agent_api.INVALID_REQUEST, "Origin 不被允许（仅本机来源）")
        if urlparse(self.path).path != MCP_ENDPOINT:
            return self._fail(404, agent_api.INVALID_REQUEST, f"没有这个路径：{self.path}")
        sid = self.headers.get(SESSION_HEADER) or ""
        if not sid:
            return self._fail(400, agent_api.INVALID_REQUEST, "缺少 Mcp-Session-Id")
        self.sessions.drop(sid)
        return self._send_empty(200)

    def do_POST(self) -> None:  # noqa: N802
        try:
            self._handle_post()
        except OverflowError as exc:
            self._fail(413, agent_api.INVALID_REQUEST, str(exc))
        except ValueError as exc:
            self._fail(400, agent_api.PARSE_ERROR, str(exc))
        except Exception as exc:  # noqa: BLE001 - 兜底：绝不把 traceback 给客户端
            _log(f"http internal {type(exc).__name__}: "
                 f"{server_mod.sanitize_text(exc, settings=self.state.settings)}")
            self._fail(500, agent_api.INTERNAL_ERROR, f"内部错误（{type(exc).__name__}）")

    def _handle_post(self) -> None:
        if not _origin_allowed(self.headers.get("Origin"), allow_any=self.allow_any_origin):
            # 同 406 那条：先读掉请求体再拒，否则客户端收到的是连接重置而不是 403。
            self._discard_body()
            return self._fail(403, agent_api.INVALID_REQUEST, "Origin 不被允许（仅本机来源）")
        path = urlparse(self.path).path
        if path.startswith(TOOLS_PATH_PREFIX):
            return self._handle_tool_call(path[len(TOOLS_PATH_PREFIX):])
        if path == RPC_PATH:
            return self._handle_rpc()
        if path != MCP_ENDPOINT:
            # 请求体必须读掉再拒，理由同上（未读字节会让 close() 发 RST）。
            self._discard_body()
            return self._fail(404, agent_api.INVALID_REQUEST, f"没有这个路径：{self.path}")

        if not _accept_allows_json(self.headers.get("Accept")):
            # **先读掉请求体再拒**。只回 406 就关连接，接收缓冲区里还有未读字节，
            # Windows 下 close() 发的是 RST 而不是 FIN，客户端内核会把已到达的 406
            # 一起丢掉——客户端看到的是"连接被重置"，而不是"你要 SSE，我没实现"。
            # `server._BodyReader` 的模块注释里有实测记录（同一形态的 413 与 503）。
            self._discard_body()
            return self._fail(
                406,
                agent_api.INVALID_REQUEST,
                "本服务只返回 application/json，未实现 text/event-stream；"
                "请在 Accept 里包含 application/json。",
            )

        raw = self._read_body()
        if not raw:
            return self._fail(400, agent_api.PARSE_ERROR, "请求体是空的")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            detail = ""
            if isinstance(exc, json.JSONDecodeError):
                detail = f"line {exc.lineno} column {exc.colno}"
            message = "解析失败：请求不是合法的 UTF-8 JSON"
            if detail:
                message = f"{message}（{detail}）"
            return self._fail(400, agent_api.PARSE_ERROR, message)

        declared = self.headers.get(PROTOCOL_HEADER)
        if declared and declared not in agent_api.SUPPORTED_PROTOCOL_VERSIONS:
            return self._fail(
                400,
                agent_api.INVALID_REQUEST,
                f"不支持的 MCP-Protocol-Version：{declared}"
                f"（支持 {list(agent_api.SUPPORTED_PROTOCOL_VERSIONS)}）",
            )

        sid = self.headers.get(SESSION_HEADER) or ""
        is_initialize = isinstance(payload, dict) and payload.get("method") == "initialize"

        session: Session | None = None
        if sid:
            session = self.sessions.get(sid)
            if session is None:
                # 规范：客户端收到 404 必须重新 initialize。**不自动建会话**——
                # 自动建就等于"会话永不失效"，TTL 与容量上限全成摆设。
                return self._fail(
                    404, agent_api.INVALID_REQUEST,
                    "会话不存在或已过期，请重新发送 initialize",
                )
        elif not is_initialize:
            return self._fail(400, agent_api.INVALID_REQUEST, "缺少 Mcp-Session-Id（initialize 除外）")

        headers = {SESSION_HEADER: sid} if sid else {}

        # 批量里可能含通知；全部是通知/响应时回 202 无体。
        if isinstance(payload, list):
            has_request = any(isinstance(m, dict) and "id" in m and "method" in m for m in payload)
        else:
            has_request = isinstance(payload, dict) and "id" in payload and "method" in payload

        response = agent_api.handle_jsonrpc(payload, self.state)

        if is_initialize and session is None and isinstance(response, dict):
            negotiated = str((response.get("result") or {}).get("protocolVersion") or "")
            session = self.sessions.create(protocol_version=negotiated)
            session.initialized = True
            headers[SESSION_HEADER] = session.session_id

        if response is None or not has_request:
            return self._send_empty(202, extra=headers)
        return self._send_json(200, response, extra=headers)

    # ---- 原生 HTTP 入口（OpenAPI 那一侧）
    def _handle_tool_call(self, raw_name: str) -> None:
        """`POST /tools/<name>`：工具名即路径，请求体就是工具入参。

        这是 OpenAPI 文档里声明的那批路径的**实现**。它与 `/mcp` 走同一个
        `agent_api.call_tool()`，所以两条入口的能力、校验、错误码完全一致
        ——文档里写的行为就是真实行为，不是另一套。
        """
        name = raw_name.strip("/")
        spec = agent_api.tool_spec(name)
        if spec is None:
            try:
                self._read_body()
            except (OverflowError, ValueError):
                pass
            return self._fail(404, agent_api.INVALID_PARAMS, f"Unknown tool: {name}")
        raw = self._read_body()
        try:
            arguments = json.loads(raw.decode("utf-8")) if raw.strip() else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._fail(400, agent_api.PARSE_ERROR, "请求体不是合法的 UTF-8 JSON")
        if not isinstance(arguments, dict):
            return self._fail(400, agent_api.INVALID_PARAMS, "请求体必须是 JSON 对象")
        result = agent_api.call_tool(self.state, name, arguments)
        return self._send_json(200, result)

    def _handle_rpc(self) -> None:
        """`POST /rpc`：原生 JSON-RPC 2.0 端点（单条与批量）。

        与 `/mcp` 的区别只有一点：**不做会话管理**。MCP 的会话是为了记住
        "这次会话协商到哪个协议版本"，而按方法名直调工具不需要协商——
        每次请求都是自洽的。
        """
        raw = self._read_body()
        if not raw.strip():
            return self._fail(400, agent_api.PARSE_ERROR, "请求体是空的")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            detail = ""
            if isinstance(exc, json.JSONDecodeError):
                detail = f"line {exc.lineno} column {exc.colno}"
            message = "解析失败：请求不是合法的 UTF-8 JSON"
            if detail:
                message = f"{message}（{detail}）"
            return self._fail(400, agent_api.PARSE_ERROR, message)
        response = agent_api.handle_jsonrpc(payload, self.state)
        if response is None:
            return self._send_empty(202)
        return self._send_json(200, response)


def serve_http(
    settings: server_mod.Settings | None = None,
    *,
    generation_id: str = "",
    allow_any_origin: bool = False,
    session_ttl_s: float = DEFAULT_SESSION_TTL_S,
    state: server_mod.AppState | None = None,
) -> ThreadingHTTPServer:
    """建好但**不启动** MCP HTTP 服务（测试要拿 `server_address` 里的真实端口）。

    `state` 可注入：测试用假 `AppState` 打真实 HTTP，才能同时覆盖"传输层"与
    "工具层"，而不必先造一代真产物。不传就按 `settings` 现建一个。
    """
    settings = settings or server_mod.Settings.from_env()
    if state is None:
        state = server_mod.AppState(settings, generation_id=generation_id)
    sessions = SessionStore(ttl_s=session_ttl_s)

    class _Bound(McpHttpHandler):
        pass

    _Bound.state = state
    _Bound.sessions = sessions
    _Bound.allow_any_origin = allow_any_origin

    httpd = ThreadingHTTPServer((settings.host, settings.port), _Bound)
    httpd.daemon_threads = True
    httpd.state = state  # type: ignore[attr-defined]
    httpd.sessions = sessions  # type: ignore[attr-defined]
    return httpd


# ---------------------------------------------------------------- 装配入口


def run_stdio(
    settings: server_mod.Settings | None = None,
    *,
    generation_id: str = "",
    reader: TextIO | None = None,
    writer: TextIO | None = None,
) -> int:
    """stdio 模式主循环。返回退出码。"""
    settings = settings or server_mod.Settings.from_env()
    state = server_mod.AppState(settings, generation_id=generation_id)
    return StdioTransport(state, reader=reader, writer=writer).serve_forever()


def main(argv: list[str] | None = None) -> int:
    """`scripts/serve_mcp.py` 的实现体。参数解析在脚本里，这里只做分发。"""
    argv = list(sys.argv[1:] if argv is None else argv)
    transport = "stdio"
    host = os.environ.get("ARKRAG_MCP_HOST", "127.0.0.1")
    port = int(os.environ.get("ARKRAG_MCP_PORT", "8766"))
    generation_id = ""
    allow_any_origin = False

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--transport" and i + 1 < len(argv):
            transport = argv[i + 1]
            i += 2
        elif arg == "--host" and i + 1 < len(argv):
            host = argv[i + 1]
            i += 2
        elif arg == "--port" and i + 1 < len(argv):
            port = int(argv[i + 1])
            i += 2
        elif arg == "--generation" and i + 1 < len(argv):
            generation_id = argv[i + 1]
            i += 2
        elif arg == "--allow-any-origin":
            allow_any_origin = True
            i += 1
        elif arg in ("-h", "--help"):
            _log("用法：--transport stdio|http [--host H] [--port P] [--generation GID]")
            return 0
        else:
            _log(f"未知参数：{arg}")
            return 2

    if transport == "stdio":
        return run_stdio(generation_id=generation_id)

    settings = server_mod.Settings.from_env()
    settings.host = host
    settings.port = port
    httpd = serve_http(settings, generation_id=generation_id, allow_any_origin=allow_any_origin)
    bound_host, bound_port = httpd.server_address[0], httpd.server_address[1]
    _log(f"streamable HTTP 就绪：http://{bound_host}:{bound_port}{MCP_ENDPOINT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        httpd.server_close()
    return 0


__all__ = [
    "DEFAULT_SESSION_TTL_S",
    "MAX_SESSIONS",
    "MAX_STDIO_LINE_BYTES",
    "MCP_ENDPOINT",
    "McpHttpHandler",
    "OPENAPI_PATH",
    "PROTOCOL_HEADER",
    "RPC_PATH",
    "SESSION_HEADER",
    "Session",
    "SessionStore",
    "StdioTransport",
    "TOOLS_PATH_PREFIX",
    "encode_message",
    "main",
    "run_stdio",
    "serve_http",
]
