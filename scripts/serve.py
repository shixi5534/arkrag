"""J1：把检索问答链起成一个本地 HTTP 服务。

    py -3 scripts\\serve.py                                   # 用发布指针那一代
    py -3 scripts\\serve.py --generation g-20261031T035231Z-4eafae
    py -3 scripts\\serve.py --port 8765 --print-limits

启动后：

    GET  /health           存活（**不代表就绪**）
    GET  /ready            就绪（每次现查盘上事实）
    GET  /release/current  发布指针
    GET  /limits           本进程生效的限制摘要
    GET  /chunks/{id}      块正文
    GET  /assertions/{id}  断言
    GET  /entities/{id}    实体与其边
    POST /query            检索问答
    POST /admin/rollback   回滚（需 ARKRAG_ADMIN_TOKEN）
    POST /admin/evaluate   Gold Set 评测（需 ARKRAG_ADMIN_TOKEN）

## 这里刻意只做四件事

解析参数 → 组装 `Settings` → 调 `arkrag.server.serve()` → `serve_forever()`。
路由、校验、清洗、限流、审计全在 `arkrag.server` 里，因为**那些是要被测试直接
打的东西**（用真实 HTTP 打 `serve()` 返回的对象）。把任何一条规则写进这个脚本，
测试就够不着它了。

## 为什么默认 `--generation` 不填

空 = 用发布指针（`data/arkrag/current.json`）。**不填就不猜"最新的那个"**——
H 阶段查出的那个缺陷就是"取最新"实际取到了最旧，而三个世代内容恰好相同所以看不
出来。发布指针是唯一的事实源。

但要注意：线上指针现在指向的是 ark-rag-1 世代（`graph_schema` 为 None、没有
adjacency 与 assertions），它**不能**被检索链载入。所以本地验证要显式
`--generation g-20261031T035231Z-4eafae`（ark-rag-3，READY 未发布）。这时
`/ready` 的 `serving_is_current` 会是 false——这是**不阻断**的可见性检查，灰度期
服务未发布世代是计划 K2 允许的形态。

## 退出码

* `0` —— 正常退出（Ctrl+C）。
* `2` —— 硬错误：端口占用、`ARKRAG_*` 环境变量不是数字、指定世代不可服务。
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arkrag import config, server  # noqa: E402


def _build_settings(args: argparse.Namespace) -> server.Settings:
    """环境变量打底，命令行覆盖。**没有一项是"命令行不给就放开"。**"""
    base = server.Settings.from_env()
    if args.host is not None:
        base.host = args.host
    if args.port is not None:
        base.port = args.port
    if args.concurrency is not None:
        base.concurrency = max(1, args.concurrency)
    if args.timeout is not None:
        base.timeout_s = float(args.timeout)
    if args.rate_limit is not None:
        base.rate_limit_calls = max(1, args.rate_limit)
    return base


def _preflight(settings: server.Settings, generation_id: str) -> int:
    """启动前先把能一眼看出来是错的配置挡掉，别等到第一个请求才 500。

    这里**不做**"数据齐不齐"的检查：那是 `/ready` 的职责，而 `/ready` 的判据必须
    能在数据变坏之后重新失败一次。启动时查一遍然后记住结论，就变成"进程启动即
    Ready"——计划 J1 明确禁止。
    """
    if settings.admin_token:
        print("[i] 管理接口已启用（ARKRAG_ADMIN_TOKEN 已配置，令牌本身不打印）")
    else:
        print("[i] 管理接口关闭（未配置 ARKRAG_ADMIN_TOKEN）")

    gid = generation_id or str((_read_pointer() or {}).get("generation_id") or "")
    if not gid:
        print("[2] 没有发布指针，也没有 --generation；服务无世代可服务", file=sys.stderr)
        return 2
    print(f"[i] 目标世代 {gid}")
    if not generation_id:
        print("[i] 目标世代来自发布指针（未指定 --generation）")
    return 0


def _read_pointer() -> dict:
    path = config.DATA_DIR / "current.json"
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ArkRAG 检索问答 HTTP 服务（J1）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--generation", default="", help="要服务的世代；省略则用发布指针")
    parser.add_argument("--host", default=None, help=f"监听地址（默认 {server.Settings.host}）")
    parser.add_argument("--port", type=int, default=None, help="监听端口（默认 8765，0=系统分配）")
    parser.add_argument("--concurrency", type=int, default=None, help="并发上限")
    parser.add_argument("--timeout", type=float, default=None, help="单请求墙钟上限（秒）")
    parser.add_argument("--rate-limit", type=int, default=None, help="每窗口允许的调用次数")
    parser.add_argument("--print-limits", action="store_true", help="打印生效的限制后退出")
    args = parser.parse_args(argv)

    try:
        settings = _build_settings(args)
    except ValueError as exc:
        print(f"[2] 配置错误：{exc}", file=sys.stderr)
        return 2

    if args.print_limits:
        print(json.dumps(settings.describe(), ensure_ascii=False, indent=2))
        return 0

    code = _preflight(settings, args.generation)
    if code:
        return code

    try:
        httpd = server.serve(settings, generation_id=args.generation)
    except OSError as exc:
        print(f"[2] 无法监听 {settings.host}:{settings.port}：{exc}", file=sys.stderr)
        return 2

    host, port = httpd.server_address[0], httpd.server_address[1]
    print(f"[i] 监听 http://{host}:{port}   限制摘要见 GET /limits")
    print("[i] 就绪判据见 GET /ready（进程活着不等于就绪）")

    def _stop(_signum, _frame):
        # 在另一个线程里 shutdown：`serve_forever()` 正在本线程跑，
        # 从信号处理器里直接调会自锁。
        import threading

        threading.Thread(target=httpd.shutdown, daemon=True).start()

    for name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, _stop)
            except (ValueError, OSError):  # pragma: no cover - 非主线程/平台不支持
                pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - 上面的信号处理器通常先接管
        pass
    finally:
        httpd.server_close()
        print("[i] 已停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
