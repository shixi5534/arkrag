"""一次性修数据：把 `data/arkrag/current.json` 里的绝对机台路径摘掉。

## 背景

`current.json` 是发布指针，进 git。2026-10 之前的版本里有一个 `"path"` 字段，
存的是 `generations/<gid>` 的**绝对机台路径**（`E:\\图rag\\...`）。那个字段全仓库
没有任何一处读取——目录本来就能从 id 推导（`Generation(gid).dir`）。
`release.publish()` / `release.rollback()` 已经不再写它，于是盘上这份旧指针成了
唯一的残留。

## 为什么用脚本而不是手工编辑

手工改 JSON 会顺手改到别的东西，而且没人能证明"身份字段一个字节都没动"。
这个脚本按当前 `generation_id` 重建指针，走 `release` 的原子写通道，并在写盘
前后**逐字段比对身份字段**，比对不上就报错退出。

身份字段（必须与原来逐字相同）：

    generation_id / graph_artifact_id / index_artifact_id / gate_report_hash /
    source_fingerprint / previous_generation / published_at

## 用法

    py -3 scripts\\repair_pointer_path.py            # 演练：只报告将要改什么
    py -3 scripts\\repair_pointer_path.py --apply    # 真写

幂等：指针里已经没有 `path` 时什么都不做，退出码 0。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arkrag import legacy, release  # noqa: E402

# 指针里承载身份的那些键。它们的值一律不许因为这次修复而变化。
IDENTITY_KEYS = (
    "generation_id",
    "graph_artifact_id",
    "index_artifact_id",
    "gate_report_hash",
    "source_fingerprint",
    "previous_generation",
    "published_at",
)


def _read_raw() -> dict:
    """读**未**经 read_current() 处理的原始指针。

    `read_current()` 已经把 `path` 弹掉了——那正是它该做的，但这里要看的恰恰是
    "盘上到底还有没有这个字段"，所以自己读一遍。
    """
    if not release.CURRENT_PATH.exists():
        raise SystemExit(f"没有指针文件：{release.CURRENT_PATH}")
    try:
        raw = json.loads(release.CURRENT_PATH.read_text("utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"指针文件不可解析，拒绝猜测：{exc}") from exc
    if not isinstance(raw, dict):
        raise SystemExit(f"指针不是对象：{type(raw).__name__}")
    return raw


def main() -> int:
    ap = argparse.ArgumentParser(description="摘掉 current.json 里能推导的绝对 path 字段")
    ap.add_argument("--apply", action="store_true", help="真写盘（默认只演练）")
    args = ap.parse_args()

    before = _read_raw()
    if not before.get("generation_id"):
        raise SystemExit("指针里没有 generation_id，拒绝重建")

    if "path" not in before:
        print(f"指针里已经没有 path 字段，无需修复：{release.CURRENT_PATH}")
        return 0

    old_path = before["path"]
    print(f"指针     : {release.CURRENT_PATH}")
    print(f"世代     : {before['generation_id']}")
    print(f"现有 path: {old_path!r}")

    # 断言这个字段确实是"能从 id 推导的绝对路径"——否则它不是我们以为的那个东西，
    # 贸然删掉可能真的丢掉信息。
    gen = release.Generation(before["generation_id"])
    expected = gen.dir
    try:
        same = Path(old_path).resolve() == expected.resolve()
    except OSError:
        same = False
    if not same:
        raise SystemExit(
            f"指针里的 path 与 `Generation(gid).dir` 不一致，拒绝删除：\n"
            f"  指针记的 : {old_path!r}\n"
            f"  从 id 推导: {str(expected)!r}\n"
            f"两者不一致说明它不是「可推导的副本」，删掉可能丢信息。"
        )
    print(f"可推导   : {str(expected)!r}（与指针一致，故该字段是纯副本）")

    # 重建指针：保留原有键顺序，只去掉 path。
    after = {k: v for k, v in before.items() if k != "path"}

    # 身份字段必须逐字不变。用 == 而不是先比再写：这是写盘**前**的最后一道闸。
    drift = [k for k in IDENTITY_KEYS if before.get(k) != after.get(k)]
    if drift:
        raise SystemExit(f"重建后身份字段发生变化，拒绝写盘：{drift}")

    print(f"新键序   : {list(after)}")
    print(f"身份字段 : {len(IDENTITY_KEYS)} 项逐字不变")

    if not args.apply:
        print("\n（演练）加 --apply 才真写盘")
        return 0

    # 走 release 用的同一条原子写通道；它会顺带过 legacy 冻结检查。
    legacy.guard_frozen_artifact(release.CURRENT_PATH, reason="repair_pointer_path")
    release.atomic.atomic_write_json(release.CURRENT_PATH, after)

    # 写盘后回读比对：上面的检查只证明"我打算写什么"，这条证明"盘上现在是什么"。
    on_disk = _read_raw()
    if "path" in on_disk:
        raise SystemExit("写盘后回读，path 仍在——原子写没生效")
    for k in IDENTITY_KEYS:
        if on_disk.get(k) != before.get(k):
            raise SystemExit(
                f"写盘后 {k} 与原来不同：{on_disk.get(k)!r} != {before.get(k)!r}"
            )
    print(f"已写盘，回读确认：无 path，{len(IDENTITY_KEYS)} 项身份字段与原来逐字相同")
    return 0


if __name__ == "__main__":
    sys.exit(main())
