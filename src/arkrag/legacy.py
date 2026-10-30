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

# 相对 ROOT 的受保护单文件产物。
#
# 这一份是静态兜底；真正完整的保护集合由 `protected_paths()` 从
# `data/releases/legacy-baseline/manifest.json` 派生。原因是 2026-10-30 的现实
# 事故：冻结清单里明明有 scenes.jsonl / chunks.jsonl / shards.json /
# corpus_meta.json，但保护清单只写了 graph/index/gates/circuit 四项，
# 于是重跑 L1/L2 时把这四个已冻结产物直接覆盖掉了——**两份清单各写一份，
# 迟早会漂移**，而 `--verify` 虽然事后抓到了，但字节已经没了（靠旧提交的
# 代码重跑才复原）。现在两者共用同一事实源。
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


def _manifest_items() -> list[dict]:
    try:
        manifest = json.loads(BASELINE_MANIFEST.read_text("utf-8"))
    except (OSError, ValueError):
        return []
    return [item for item in (manifest.get("artifacts") or []) if isinstance(item, dict)]


def _is_dir_entry(item: dict) -> bool:
    """判断冻结清单里这一条是目录还是单文件。

    真实 manifest 里的目录项（`data/arkrag/cache`）**没有** `is_dir` 字段，
    键集是 `bytes/count/digest/kind/path`；单文件项则一定有 `sha256`。
    只按 `is_dir` 判会把缓存目录误当单文件塞进 `protected_paths()`——那等于
    说"路径 `data/arkrag/cache` 本身不许写"，但目录下的新键追加写仍然放行，
    保护对象就悄悄错了位。三个信号取或，任何一个成立即按目录处理。
    """
    if item.get("is_dir"):
        return True
    if "sha256" not in item:
        return True
    return str(item.get("kind") or "").endswith("_dir")


def baseline_paths() -> tuple[str, ...]:
    """冻结清单里逐条记录的**单文件**产物路径。

    Manifest 不存在或损坏时返回空元组——即退回 `PROTECTED_FILES` 静态兜底，
    而不是"保护全部"或"不保护任何"。两种极端都会让下一次事故更难判断。
    """
    rows: list[str] = []
    for item in _manifest_items():
        path = str(item.get("path") or "")
        if path and not _is_dir_entry(item):
            rows.append(path)
    return tuple(rows)


def baseline_dirs() -> tuple[str, ...]:
    """冻结清单里逐条记录的**目录**产物路径（如 `data/arkrag/cache`）。"""
    rows: list[str] = []
    for item in _manifest_items():
        path = str(item.get("path") or "")
        if path and _is_dir_entry(item):
            rows.append(path)
    return tuple(rows)


def protected_paths() -> tuple[str, ...]:
    """静态清单 + 冻结清单派生的并集，去重保序。"""
    seen: dict[str, None] = {}
    for path in PROTECTED_FILES + baseline_paths():
        seen.setdefault(path, None)
    return tuple(seen)


def protected_dirs() -> tuple[str, ...]:
    """受保护目录 = 静态清单 + 冻结清单派生的并集，去重保序。"""
    seen: dict[str, None] = {}
    for path in PROTECTED_DIRS + baseline_dirs():
        seen.setdefault(path, None)
    return tuple(seen)


def _rel(path: Path) -> str:
    """把任意路径折成相对 ROOT 的 posix 形式；不在 ROOT 内时退化为绝对形式。"""
    p = Path(path)
    try:
        return p.resolve().relative_to(config.ROOT.resolve()).as_posix()
    except (ValueError, OSError):
        return p.as_posix()


def is_protected(path: Path) -> bool:
    rel = _rel(path)
    if rel in protected_paths():
        return True
    return any(rel == d or rel.startswith(d + "/") for d in protected_dirs())


def guard_write(path: Path, *, reason: str = "") -> None:
    """写入前调用。受保护路径（含受保护目录）且未放行时抛 LegacyWriteRefused。"""
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


def guard_frozen_artifact(path: Path, *, reason: str = "") -> None:
    """只拦**冻结清单里的单文件产物**，不拦受保护目录。

    这是给 `atomic.atomic_write_bytes` 用的统一关口。用它是为了让保护落在
    "所有写入都必须经过的那一条通道"上，而不是每个调用点各自记得调一次——
    2026-10-30 的事故正是后者的直接后果：`run_l1.py` / `run_l2.py` 谁都没调
    `guard_write`，于是重跑时把已冻结的 scenes.jsonl / chunks.jsonl /
    shards.json / corpus_meta.json 直接覆盖了。

    刻意不拦 `PROTECTED_DIRS`（当前是 `data/arkrag/cache`）：缓存目录里
    legacy 记录只读复用、新记录以新键**追加**写入，这是正常运行的一部分；
    在这里拦会把抽取流水线整个卡死。要防的是"整份产物被覆盖"。
    """
    rel = _rel(path)
    if rel not in protected_paths():
        return
    if os.environ.get(ENV_ALLOW) == "1":
        return
    raise LegacyWriteRefused(
        f"拒绝写入已冻结的 legacy 产物: {rel}"
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
