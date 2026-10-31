"""把已入库 JSON 里的绝对机台路径改写成相对仓库根的形式（计划 A3 / DoD「部署方式可复现」）。

## 为什么需要单独一个脚本

`current.json` 与 `source_verification.json` 的路径问题已在 `repair_pointer_path.py`
那一轮修掉，当时我以为只有这两个文件。实际上 git 跟踪的 JSON 里还有三处会带
`E:\\图rag\\...`：

  * `data/arkrag/agent_plan.json` —— `S01.batches[*].inbox` / `.outbox`（1,186 处）
  * `data/arkrag/ledger/<label>.json` —— `task_file`（16 个文件各 1 处）
  * `data/arkrag/corpus_meta.json`、`data/arkrag/l0_manifest.json` —— `source_dir`（各 1 处）

前两类是**生成物**，其产出点已在 `agent_tasks.py` 改成 `config.rel_to_root()`；本脚本
负责把已落盘的历史文件按同一规则改写，让"代码已经改好"和"仓库里的文件已经合规"
两件事都成立——否则下一次 `git status` 干净只是因为没人重跑过。

后两类属于 A 阶段冻结的 legacy-baseline 产物（`data/releases/legacy-baseline/manifest.json`
逐字节登记），**本脚本刻意不动它们**：冻结的意义就是"一个字节都不许变"，为了让一条
检查变绿去改证据，等于把证据毁掉。它们由白名单显式豁免，理由写在白名单旁边。

## 改写规则

只用 `config.rel_to_root()`：仓库内路径 → 正斜杠相对形式；仓库外原样返回。
不做「把盘符替换成空串」这类字符串手术——那种写法会把
`E:\\图rag2\\x` 也切错，而 `rel_to_root` 是按路径分量比较的。

默认只演练。加 `--apply` 才写盘，写盘走 `atomic.atomic_write_json`（同目录临时文件 +
fsync + `os.replace`），所以中途失败不会留下半个文件。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arkrag import atomic, config, legacy  # noqa: E402

#: 参与改写的文件（相对仓库根）
TARGETS = (
    "data/arkrag/agent_plan.json",
    "data/arkrag/ledger/B01.json",
    "data/arkrag/ledger/S00.json",
    "data/arkrag/ledger/S01.json",
    "data/arkrag/ledger/S02.json",
    "data/arkrag/ledger/S03.json",
    "data/arkrag/ledger/S04.json",
    "data/arkrag/ledger/S05.json",
    "data/arkrag/ledger/S06.json",
    "data/arkrag/ledger/S07.json",
    "data/arkrag/ledger/S08.json",
    "data/arkrag/ledger/S09.json",
    "data/arkrag/ledger/S10.json",
    "data/arkrag/ledger/S11.json",
    "data/arkrag/ledger/T0.json",
    "data/arkrag/ledger/T0v2.json",
    "data/arkrag/ledger/_ingest.json",
)

#: 冻结基线产物：含绝对路径也**不许改**。理由见模块 docstring 的最后一段。
FROZEN_EXEMPT = (
    "data/arkrag/corpus_meta.json",
    "data/arkrag/l0_manifest.json",
)


def _looks_absolute(value: str) -> bool:
    """判断一个字符串是不是绝对机台路径。

    只认真实存在的形态（Windows 盘符 + 分隔符、POSIX 前导 `/`），不猜。
    `C:relative` 这种"盘符相对"形式 Windows 上有特殊语义，这里不算绝对。
    """
    if len(value) >= 3 and value[1] == ":" and value[2] in "\\/":
        return value[0].isalpha()
    return value.startswith("/")


def rewrite_paths(obj, *, changed: list[tuple[str, str, str]], prefix: str = ""):
    """递归改写 dict/list 里的绝对路径字符串，原地修改并记录改动三元组。"""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str) and _looks_absolute(value):
                new = config.rel_to_root(value)
                if new != value:
                    changed.append((f"{prefix}.{key}" if prefix else key, value, new))
                    obj[key] = new
            else:
                rewrite_paths(value, changed=changed, prefix=f"{prefix}.{key}" if prefix else key)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            if isinstance(value, str) and _looks_absolute(value):
                new = config.rel_to_root(value)
                if new != value:
                    changed.append((f"{prefix}[{index}]", value, new))
                    obj[index] = new
            else:
                rewrite_paths(value, changed=changed, prefix=f"{prefix}[{index}]")
    return obj


def repair_one(rel: str, *, apply: bool) -> tuple[int, list[tuple[str, str, str]]]:
    """改写一个文件，返回 (改动数, 改动清单)。文件不存在时返回 (0, [])。"""
    path = config.ROOT / rel
    if not path.exists():
        return 0, []
    doc = json.loads(path.read_text("utf-8"))
    changed: list[tuple[str, str, str]] = []
    rewrite_paths(doc, changed=changed)
    if changed and apply:
        # 走 atomic 通道：既拿到「同目录临时文件 + fsync + os.replace」，
        # 也顺带被 legacy 写保护检查一遍（这些文件不在冻结清单里，会正常放行）。
        atomic.atomic_write_json(path, doc)
    return len(changed), changed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="把已入库 JSON 的绝对路径改成相对仓库根")
    ap.add_argument("--apply", action="store_true", help="真正写盘（默认只演练）")
    ap.add_argument("--limit", type=int, default=3, help="每个文件最多打印几条样例")
    args = ap.parse_args(argv)

    frozen_present = [rel for rel in FROZEN_EXEMPT if (config.ROOT / rel).exists()]
    if frozen_present:
        print("豁免（冻结基线，逐字节登记，不得改写）:")
        for rel in frozen_present:
            print(f"  - {rel}")

    total = 0
    touched: list[str] = []
    for rel in TARGETS:
        count, changed = repair_one(rel, apply=args.apply)
        if not count:
            continue
        total += count
        touched.append(rel)
        print(f"{rel}: {count} 处")
        for where, old, new in changed[: args.limit]:
            print(f"    {where}: {old!r} -> {new!r}")
        if len(changed) > args.limit:
            print(f"    …… 另有 {len(changed) - args.limit} 处")

    if not total:
        print("没有需要改写的绝对路径。")
        return 0
    if args.apply:
        print(f"已改写 {total} 处，涉及 {len(touched)} 个文件。")
    else:
        print(f"演练：将改写 {total} 处，涉及 {len(touched)} 个文件。加 --apply 才写盘。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
