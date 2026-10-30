"""legacy 产物写保护与冻结校验。

计划 A1c：`graph.json`、`index/lexical.json`、`gates.json` 与当前 `agent_cache`
一旦冻结为 `legacy-baseline`，就不允许后续实验直接写。

保护必须能失败——只打印警告不算保护。两级机制：

1. **快照校验**（`scripts/snapshot_baseline.py --verify`）：冻结时记下的 SHA-256
   与当前字节不符，退出码非零。它防的是"产物被悄悄改过却没人发现"。
2. **写保护**（`guard_write`）：进程内对受保护路径的写入被拒绝，除非显式放行
   `ARKRAG_ALLOW_LEGACY_WRITE=1`。它防的是"新实验顺手覆盖了唯一一份可用产物"。

放行口是给「确实要重建 legacy 产物」的时刻用的，用完必须取消；不要把它写进
任何常用脚本或环境模板。
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from . import config

ENV_ALLOW = "ARKRAG_ALLOW_LEGACY_WRITE"

# 相对 ROOT 的受保护单文件产物
PROTECTED_FILES: tuple[str, ...] = (
    "data/arkrag/graph.json",
    "data/arkrag/index/lexical.json",
    "data/arkrag/gates.json",
    "data/arkrag/circuit_open.json",
)

# 相对 ROOT 的受保护目录（前缀匹配）
PROTECTED_DIRS: tuple[str, ...] = (
    "data/arkrag/cache",
)

RELEASES_DIR = config.DATA_DIR.parent / "releases"
BASELINE_DIR = RELEASES_DIR / "legacy-baseline"
BASELINE_MANIFEST = BASELINE_DIR / "manifest.json"


class LegacyWriteRefused(RuntimeError):
    """对已冻结 legacy 产物的写入被拒绝。"""


def _rel(path: Path) -> str:
    """把任意路径折成相对 ROOT 的 posix 形式；不在 ROOT 内时退化为绝对形式。"""
    p = Path(path)
    try:
        return p.resolve().relative_to(config.ROOT.resolve()).as_posix()
    except (ValueError, OSError):
        return p.as_posix()


def is_protected(path: Path) -> bool:
    rel = _rel(path)
    if rel in PROTECTED_FILES:
        return True
    return any(rel == d or rel.startswith(d + "/") for d in PROTECTED_DIRS)


def guard_write(path: Path, *, reason: str = "") -> None:
    """写入前调用。受保护路径且未放行时抛 LegacyWriteRefused。"""
    if not is_protected(path):
        return
    if os.environ.get(ENV_ALLOW) == "1":
        return
    raise LegacyWriteRefused(
        f"拒绝写入已冻结的 legacy 产物: {_rel(path)}"
        f"{'（' + reason + '）' if reason else ''}。"
        f"新产物请写进 generations/<generation_id>/，"
        f"确需重建 legacy 时显式设置 {ENV_ALLOW}=1。"
    )


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def artifact_record(path: Path, kind: str = "") -> dict:
    p = Path(path)
    st = p.stat()
    return {
        "path": _rel(p),
        "bytes": st.st_size,
        "sha256": sha256_file(p),
        "mtime": st.st_mtime,
        "mtime_iso": __import__("time").strftime(
            "%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime(st.st_mtime)
        ),
        "kind": kind,
    }


def dir_digest(root: Path) -> dict:
    """目录整体身份：按 (相对名, 字节数, SHA-256) 排序后折叠成一个摘要。

    逐文件记账 13,801 个缓存文件既臃肿又没人看；真正需要的是「这批缓存整体
    有没有变过」，所以记 count / bytes / digest 三件套，另存完整清单哈希。
    """
    root = Path(root)
    rows: list[str] = []
    total = 0
    biggest = 0
    if root.exists():
        for p in sorted(root.rglob("*")):
            if p.is_file():
                size = p.stat().st_size
                total += size
                biggest = max(biggest, size)
                rows.append(f"{p.relative_to(root).as_posix()}\x00{size}\x00{sha256_file(p)}")
    h = hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()
    return {"count": len(rows), "bytes": total, "largest_bytes": biggest, "digest": h}


def load_baseline() -> dict | None:
    if not BASELINE_MANIFEST.exists():
        return None
    return json.loads(BASELINE_MANIFEST.read_text("utf-8"))
