"""发布命令行：查看/发布/回滚/校验/清理 Generation（计划 C1 / K4）。

    py -3 scripts\\release_cli.py current           # 现在线上是哪一代
    py -3 scripts\\release_cli.py list              # 所有世代与状态
    py -3 scripts\\release_cli.py show <gid>        # 一代的 Manifest 与自检结果
    py -3 scripts\\release_cli.py publish <gid>     # 把 READY 的一代切成线上（先过发布前清单）
    py -3 scripts\\release_cli.py rollback <gid>    # 指针切回历史版本（不重建）
    py -3 scripts\\release_cli.py verify <gid>      # 只校验不改状态
    py -3 scripts\\release_cli.py prune --keep 2    # 清理旧 FAILED 世代（默认演练）

`rollback` 只切指针。历史产物一个字节都不动——重跑几小时的抽取才能造回来的东西，
不该在回滚时被"重新生成"一遍。

`publish` 在调 `release.publish()` 之前**先过发布前清单门**
（`scripts/k2_publish_check.py`）：五份门报告（K2 硬门 / K2 性能 / K2 灰度 /
K3 阈值审批 / K4 演练）必须都在、都过、**都绑这一代**。没有这一步，「五道门都跑过」
只是一句自觉——`release.publish()` 自己看不到那五份报告，一个 READY 世代照样能
带着「根本没跑」或「跑的是上一代」的报告上线。

这道清单**没有跳过开关**：需要跳过的时候，该做的是把那五道门跑完，而不是绕开门。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS.parent / "src"))
sys.path.insert(0, str(_SCRIPTS))

from arkrag import release  # noqa: E402

import k2_publish_check as publish_check  # noqa: E402  （发布前清单门）


def cmd_current(_args: argparse.Namespace) -> int:
    cur = release.read_current()
    if not cur:
        print("当前没有已发布的 Generation（current.json 不存在或为空）")
        return 2
    print(f"当前世代 {cur.get('generation_id')}")
    for key in ("status", "published_at", "previous_generation", "source_fingerprint"):
        if key in cur:
            print(f"  {key}={cur[key]}")
    for key in ("graph_artifact_id", "index_artifact_id", "gate_report_hash"):
        if cur.get(key):
            print(f"  {key}={cur[key][:24]}…")
    if cur.get("rolled_back"):
        print("  （这是一次回滚结果）")

    problems = release.verify_generation(release.Generation(cur["generation_id"]))
    if problems:
        print(f"\n自检 FAIL {len(problems)} 项：", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    print("\n自检 PASS：Manifest 与产物一致，图与索引绑同一份图")
    return 0


def cmd_list(_args: argparse.Namespace) -> int:
    rows = release.list_generations()
    if not rows:
        print("没有任何 Generation 目录")
        return 2
    cur = (release.read_current() or {}).get("generation_id", "")
    print(f"{'':2} {'generation_id':28} {'status':11} {'块':>6} {'实体':>6} {'边':>7}  创建时间")
    for r in rows:
        mark = "->" if r["generation_id"] == cur else "  "
        print(
            f"{mark} {r['generation_id']:28} {r['status']:11} "
            f"{r.get('chunk_count') or '-':>6} {r.get('entities') or '-':>6} "
            f"{r.get('edges') or '-':>7}  {r.get('created_at','')}"
        )
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    gen = release.Generation(args.generation_id)
    if not gen.manifest_path.exists():
        print(f"没有这一代：{gen.dir}", file=sys.stderr)
        return 2
    manifest = gen.load()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    problems = release.verify_generation(gen)
    print(f"\n自检：{'PASS' if not problems else f'FAIL {len(problems)} 项'}")
    for p in problems:
        print(f"  {p}")
    return 1 if problems else 0


def cmd_verify(args: argparse.Namespace) -> int:
    gen = release.Generation(args.generation_id)
    problems = release.verify_generation(gen)
    if problems:
        print(f"FAIL {len(problems)} 项：")
        for p in problems:
            print(f"  {p}")
        return 1
    print(f"PASS {gen.generation_id}：Manifest 与产物一致")
    return 0


def cmd_publish(args: argparse.Namespace) -> int:
    gen = release.Generation(args.generation_id)

    # 发布前清单门：必须先过，且它自己会写 data/k2_publish_check.json 留档。
    # 顺序不能反——先 publish 再查清单，等于事故已经发生才补检查。
    try:
        verdict = publish_check.evaluate_generation(gen.generation_id)
    except publish_check.PublishCheckError as exc:
        print(f"拒绝发布：发布前清单门无法判定（{exc}）", file=sys.stderr)
        return 1
    if not verdict["all_passed"]:
        print(f"拒绝发布：发布前清单 {len(verdict['failed'])} 条不成立", file=sys.stderr)
        for item in verdict["failed"]:
            print(f"  - {item['name']}: {item['detail']}", file=sys.stderr)
        print(
            "  清单明细见 data/k2_publish_check.json；"
            "先把上面点名的门跑完再发布（这道清单没有跳过开关）",
            file=sys.stderr,
        )
        return 1
    undetermined = [r for r in verdict["checks"] if r["verdict"] == publish_check.UNDETERMINED]
    if undetermined:
        # 无法判定不算失败，但它必须被说出来——否则「清单全过」会被读成
        # 「六条都验过了」，而实际有几条从来没被验过。
        print(f"清单通过（另有 {len(undetermined)} 条无法判定，见 data/k2_publish_check.json）：")
        for item in undetermined:
            print(f"  · {item['name']}: {item['detail']}")
    else:
        print("清单通过：五份门报告都在、都过、都绑这一代")

    run = release.Run()
    try:
        pointer = release.publish(gen, run=run)
    except release.ReleaseError as exc:
        # 发布被拒是正常业务结果（覆盖不全 / 状态不对 / 自检不过），
        # 不是程序崩溃——给人一句能读的话，再加 detail 指向排查入口，
        # 而不是甩一段 traceback 让人以为工具坏了。
        print(f"拒绝发布：{exc}", file=sys.stderr)
        if exc.detail:
            print(f"  明细：{exc.detail}", file=sys.stderr)
        print(f"  run={run.run_id}  日志 {run.dir / 'events.jsonl'}", file=sys.stderr)
        return 1
    print(f"已发布 {pointer['generation_id']}（上一代 {pointer['previous_generation'] or '(无)'}）")
    print(f"  run={run.run_id}  日志 {run.dir / 'events.jsonl'}")
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    run = release.Run()
    pointer = release.rollback(args.generation_id, run=run)
    print(f"已回滚到 {pointer['generation_id']}（原 {pointer['previous_generation'] or '(无)'}）")
    print("  只切换了指针；历史产物未重建、未修改")
    print(f"  run={run.run_id}")
    return 0


def cmd_prune(args: argparse.Namespace) -> int:
    victims = release.prune_old(keep=args.keep, dry_run=not args.apply)
    if not victims:
        print("没有可清理的世代")
        return 0
    verb = "已删除" if args.apply else "将删除（演练，未实际删除；加 --apply 执行）"
    print(f"{verb} {len(victims)} 个：")
    for gid in victims:
        print(f"  {gid}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Generation 发布与回滚")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("current", help="显示当前线上世代并自检").set_defaults(func=cmd_current)
    sub.add_parser("list", help="列出所有世代").set_defaults(func=cmd_list)

    for name, func, help_text in (
        ("show", cmd_show, "显示一代的 Manifest 与自检"),
        ("verify", cmd_verify, "只校验一代"),
        ("publish", cmd_publish, "发布 READY 的一代"),
        ("rollback", cmd_rollback, "回滚到历史世代"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("generation_id")
        p.set_defaults(func=func)

    p = sub.add_parser("prune", help="清理旧 FAILED 世代")
    p.add_argument("--keep", type=int, default=2, help="保留最近 N 个非发布世代")
    p.add_argument("--apply", action="store_true", help="真正删除（默认只演练）")
    p.set_defaults(func=cmd_prune)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
