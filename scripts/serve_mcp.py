"""A3：MCP 服务启动入口。

    py -3 scripts\\serve_mcp.py                          # stdio（客户端当子进程拉起）
    py -3 scripts\\serve_mcp.py --transport http          # streamable HTTP，默认 :8766/mcp
    py -3 scripts\\serve_mcp.py --transport http --port 0  # 系统分配端口（测试用）
    py -3 scripts\\serve_mcp.py --generation g-20261031T035231Z-4eafae
    py -3 scripts\\serve_mcp.py --print-tools            # 打印能力表后退出

## 为什么 stdio 是默认

MCP 客户端（Claude Desktop / Cursor / 各类 IDE 插件）最常用、也最省心的接法是
把服务端当**子进程**拉起，用 stdin/stdout 通信。这条路不需要端口、不需要会话、
不需要防火墙规则——配置就是一个命令字符串。

## stdio 模式下 stdout 是协议通道

启动横幅、警告、错误——**全部走 stderr**。任何一行 `print()` 到 stdout 都会让
客户端解析失败，而它通常只报"JSON 解析失败"，让人以为是自己发错了请求。
本脚本因此没有任何 `print()`，输出统一经 `arkrag.mcp_server._log()`。

## HTTP 模式绑定 127.0.0.1

规范对本传输的要求（DNS rebinding 防护）：默认只监听回环、只接受本机 `Origin`。
要跨机接入必须显式 `--host 0.0.0.0` 与 `--allow-any-origin`，
并**自行**在前面加一层鉴权——本服务的 HTTP 传输没有鉴权（见文档「未实现」）。

## 退出码

* `0` —— 正常退出（stdin 关闭 / Ctrl+C）。
* `2` —— 硬错误：参数不对、端口占用、`ARKRAG_*` 环境变量不是数字。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arkrag import agent_api, mcp_server, server  # noqa: E402


def _log(message: str) -> None:
    """日志只走 stderr。**这里必须显式固定 UTF-8**。

    该脚本是 MCP 客户端的子进程；客户端通常用 `text=True, encoding="utf-8"`
    读 stderr。Windows 控制台默认 cp936，中文日志按本地编码写出、读端按 UTF-8
    解，客户端拿到的是 `UnicodeDecodeError`——**日志把宿主搞崩**是最没必要的
    一种故障（实测复现）。`mcp_server._log` 同样处理，这里不能只依赖它：
    本脚本在导入 `arkrag` 之后、调用 `mcp_server` 之前就会打日志。
    """
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
    except (AttributeError, OSError, ValueError):
        pass
    print(f"[arkrag-mcp] {message}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ArkRAG MCP 服务（可被 Claude Desktop / Cursor 等客户端直接接入）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--transport", choices=("stdio", "http"), default="stdio",
        help="stdio = 客户端当子进程拉起（默认）；http = streamable HTTP",
    )
    parser.add_argument("--host", default=None, help="HTTP 监听地址（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=None, help="HTTP 端口（默认 8766，0=系统分配）")
    parser.add_argument("--generation", default="", help="要服务的世代；省略则用发布指针")
    parser.add_argument(
        "--allow-any-origin", action="store_true",
        help="HTTP 模式接受任意 Origin（**默认只接受本机**；跨机接入需自行加鉴权）",
    )
    parser.add_argument(
        "--print-tools", action="store_true", help="打印能力表（JSON）后退出，不启动服务",
    )
    parser.add_argument(
        "--print-openapi", action="store_true", help="打印 OpenAPI 3.1 文档后退出",
    )
    args = parser.parse_args(argv)

    if args.print_tools:
        print(json.dumps(agent_api.tool_catalog(), ensure_ascii=False, indent=2))
        return 0
    if args.print_openapi:
        settings = server.Settings.from_env()
        if args.host is not None:
            settings.host = args.host
        if args.port is not None:
            settings.port = args.port
        print(agent_api.openapi_json(settings=settings))
        return 0

    problems = agent_api.validate_registry()
    if problems:
        # 能力表不自洽属于**开发期**错误，启动前就挡住：带病启动的结果是
        # `tools/list` 与 OpenAPI 各说各话，而客户端只会看到"某个工具行为怪异"。
        for problem in problems:
            _log(f"能力表不自洽：{problem}")
        return 2

    try:
        settings = server.Settings.from_env()
    except ValueError as exc:
        _log(f"配置错误：{exc}")
        return 2
    if args.host is not None:
        settings.host = args.host
    if args.port is not None:
        settings.port = args.port

    if not settings.admin_token:
        _log("管理类工具关闭（未配置 ARKRAG_ADMIN_TOKEN，fail-closed）")

    if args.transport == "stdio":
        return mcp_server.run_stdio(settings, generation_id=args.generation)

    try:
        httpd = mcp_server.serve_http(
            settings, generation_id=args.generation, allow_any_origin=args.allow_any_origin
        )
    except OSError as exc:
        _log(f"无法监听 {settings.host}:{settings.port}：{exc}")
        return 2

    host, port = httpd.server_address[0], httpd.server_address[1]
    _log(f"streamable HTTP 就绪：http://{host}:{port}{mcp_server.MCP_ENDPOINT}")
    _log("未实现 SSE 流；GET 会回 405 并说明原因")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - 交互式
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
