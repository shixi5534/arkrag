# -*- coding: utf-8 -*-
"""`k2_publish_check` —— 发布前清单门：发布前必须证明「那几道门真的跑过，而且跑的是这一代」。

## 为什么要有这个脚本

`docs/升级计划_GraphRAG架构.md:420-428` 的 K2 构建顺序末尾是

    … → 全量硬门 → Gold Set 评测 → 性能评测 → 候选 READY → 灰度查询 → 正式发布

这几道门各自有脚本、各自写一份报告，但**没有任何一处要求「发布前把所有门都跑过，
且每份报告绑的是同一个候选世代」**。`release.publish()` 自己只查三件事：状态是
READY、覆盖完整、`verify_generation()` 没有不符项——它**看不到那五份门报告**。
也就是说，只要有人手快，一个 READY 世代就能上线，而上面五道门可能根本没跑，
或者跑的是上一代（报告是旧的）却没人对账。

这道门就是把「清单」变成可执行判定：五份报告在场、全过、全绑同一个世代、
候选自身状态与自检合格、报告不比候选更老。

## 五份报告与它们各自不同的世代字段（这是本脚本最容易写错的地方）

| 文件 | 世代字段 | 产出脚本 |
|---|---|---|
| `data/k2_hard_gates.json` | `generation_id` | `scripts/k2_hard_gates.py` |
| `data/k2_perf.json` | `generation_id` | `scripts/k2_perf.py` |
| `data/k2_gray.json` | `generation_id` | `scripts/k2_gray.py` |
| `data/k3_threshold_gate.json` | `generations.new` | `scripts/k3_thresholds.py` |
| `data/k4_drill.json` | `candidate` | `scripts/k4_drill.py` |

**不许把它们统一成同一个字段名**：那些字段是各脚本产出报告时写的，本脚本是消费者。
消费者去改生产者的报告格式，等于让报告与产生它的代码对不上。所以这里用一张
登记表把「哪份报告、世代 id 在第几个点后面」写死在一处（`REPORTS`），
不在五个地方各写一遍——同一判定写两处，迟早只更新一处。

## 三条判定纪律

1. **每一项都打印分母**（几份报告里的几份），没有分母的比率不许出现。
2. **无法判定就写「无法判定」，绝不写成通过**。时间戳这一条尤其：五份报告里只有
   `k2_perf.json` 与 `k2_gray.json` 带 `generated_at`，另外三份没有这个字段。
   对那三份只能记 `undetermined`（不算失败，也不算通过），并在 detail 里点名。
3. **失败要点名是哪一条、差在哪**：报告记的是哪个 id、要发布的是哪个 id，两个都打出来。

## 用法

    py -3 scripts\\k2_publish_check.py                     # 默认检查候选 06a741
    py -3 scripts\\k2_publish_check.py --generation g-...

## 退出码

* `0` —— 清单项全部成立（可以发布）；
* `1` —— 有清单项不成立（不得发布）；
* `2` —— 硬错误：参数非法、世代不存在/缺 Manifest、报告不是合法 JSON。

## 边界

* 本脚本**不改** `release.publish()`，也不代替它；它只是发布前的一道自查。
* 本脚本不写 Generation、不改产物、不动线上指针，只读报告与候选 Manifest。
* 「报告全过」不等于「这一代好」——它只证明那份门报告存在、通过、且绑的是同一代。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
# Windows 控制台默认 cp936，不重配置会把中文写成乱码字节，
# 且 `> file` 重定向后文件不是 UTF-8（本项目脚本一律要求输出可原样粘贴进文档）。
# 守卫是必要的：pytest 捕获期间 `sys.stdout` 可能是 `EncodedFile`/`CaptureFixture`
# 这类不可重配置的对象，直接调 `.reconfigure()` 会 AttributeError。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from arkrag import atomic, config, release  # noqa: E402

CANDIDATE_GID = "g-20261103T132152Z-06a741"
REPORT_PATH = ROOT / "data" / "k2_publish_check.json"
TOOL = "scripts/k2_publish_check.py"

# 五份门报告的登记表。`gid_field` 是点号路径（`generations.new` 表示先取
# `payload["generations"]` 再取 `["new"]`），`producer` 用于在缺报告时告诉人该跑哪个脚本。
REPORTS: tuple[dict, ...] = (
    {
        "key": "k2_hard_gates",
        "label": "K2 全量硬门（8 条）",
        "path": "data/k2_hard_gates.json",
        "gid_field": "generation_id",
        "producer": "py -3 scripts/k2_hard_gates.py",
    },
    {
        "key": "k2_perf",
        "label": "K2 性能评测（7 硬门 + 22 申报）",
        "path": "data/k2_perf.json",
        "gid_field": "generation_id",
        "producer": "py -3 scripts/k2_perf.py",
    },
    {
        "key": "k2_gray",
        "label": "K2 灰度查询（5 判据）",
        "path": "data/k2_gray.json",
        "gid_field": "generation_id",
        "producer": "py -3 scripts/k2_gray.py",
    },
    {
        "key": "k3_threshold_gate",
        "label": "K3 阈值与人工审批（8 delta + 9 申报）",
        "path": "data/k3_threshold_gate.json",
        "gid_field": "generations.new",
        "producer": "py -3 scripts/k3_thresholds.py",
    },
    {
        "key": "k4_drill",
        "label": "K4 回滚与灰度演练（17 门）",
        "path": "data/k4_drill.json",
        "gid_field": "candidate",
        "producer": "py -3 scripts/k4_drill.py",
    },
)

# 判定结果三态。`undetermined` 不是失败，但也**不是通过**——它表示这条判据
# 在现有报告字段下无法成立，必须原样写进报告让人看见。
PASS = "pass"
FAIL = "fail"
UNDETERMINED = "undetermined"


class PublishCheckError(RuntimeError):
    """清单门的硬错误：参数非法、世代不可读、报告不是合法 JSON。"""


# ---------------------------------------------------------------- 工具


def dig(payload: object, dotted: str) -> object:
    """按点号路径取嵌套字段；任一层缺失或不是 dict 就返回 `None`。

    这里刻意**不抛异常**：字段缺失本身就是一个判定结果（「这份报告没绑世代」），
    抛出去会让调用方少判一条。
    """
    cur = payload
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def parse_iso(text: object) -> datetime | None:
    """解析 `2026-11-04T07:14:40Z` 这类时间戳；解析不了返回 `None`（不猜当前时间）。"""
    if not isinstance(text, str) or not text.strip():
        return None
    raw = text.strip()
    if raw.endswith("Z") or raw.endswith("z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


# ---------------------------------------------------------------- 读报告


def load_reports(reports: tuple[dict, ...] | list[dict] = REPORTS) -> list[dict]:
    """读五份报告，逐份描出「在不在、是不是合法 JSON、绑的哪个世代、过没过、什么时候生成」。

    读不出来的报告不抛异常，只把原因记进 `error` 字段——门要能一次说清所有问题，
    而不是修一个再跑一次才看见下一个。
    """
    out: list[dict] = []
    for spec in reports:
        path = Path(spec["path"])
        if not path.is_absolute():
            path = ROOT / path
        record: dict = {
            "key": spec["key"],
            "label": spec.get("label", spec["key"]),
            "path": str(path),
            "rel_path": config.rel_to_root(path),
            "gid_field": spec["gid_field"],
            "producer": spec.get("producer", ""),
            "exists": path.exists(),
            "valid_json": False,
            "error": "",
            "payload": None,
            "generation_id": None,
            "all_passed": None,
            "generated_at": None,
        }
        if record["exists"]:
            try:
                record["payload"] = json.loads(path.read_text("utf-8"))
                record["valid_json"] = True
            except (OSError, ValueError) as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
        if record["valid_json"]:
            payload = record["payload"]
            record["generation_id"] = dig(payload, record["gid_field"])
            if isinstance(payload, dict):
                record["all_passed"] = payload.get("all_passed")
                record["generated_at"] = payload.get("generated_at")
        out.append(record)
    return out


# ---------------------------------------------------------------- 各条判据（纯函数）
#
# 六条判据全部吃「已读完的报告记录」与「候选的已知事实」，返回 (verdict, detail, data)。
# 纯函数是为了让 tests/ 能用合成数据逐条构造「本该拦住」的输入。


def check_reports_present(records: list[dict]) -> tuple[str, str, dict]:
    """五份报告都在场、都是合法 JSON。"""
    missing: list[str] = []
    broken: list[str] = []
    for rec in records:
        if not rec["exists"]:
            missing.append(f"{rec['rel_path']}（缺，跑 {rec['producer']}）")
        elif not rec["valid_json"]:
            broken.append(f"{rec['rel_path']}（{rec['error']}）")
    have = len(records) - len(missing) - len(broken)
    data = {"have": have, "total": len(records), "missing": missing, "broken": broken}
    if missing or broken:
        parts = []
        if missing:
            parts.append(f"缺 {len(missing)} 份：{'；'.join(missing)}")
        if broken:
            parts.append(f"坏 {len(broken)} 份：{'；'.join(broken)}")
        return FAIL, f"{have}/{len(records)} 份可用；" + "，".join(parts), data
    return PASS, f"{have}/{len(records)} 份报告在场且可解析", data


def check_reports_passed(records: list[dict]) -> tuple[str, str, dict]:
    """每份报告自己的 `all_passed` 都是 True。"""
    unusable = [r["rel_path"] for r in records if not r["valid_json"]]
    not_true: list[str] = []
    for rec in records:
        if not rec["valid_json"]:
            continue
        if rec["all_passed"] is not True:
            not_true.append(f"{rec['rel_path']}（all_passed={rec['all_passed']!r}）")
    judged = len(records) - len(unusable)
    data = {"passed": judged - len(not_true), "judged": judged, "not_true": not_true}
    if not_true:
        return FAIL, f"{data['passed']}/{judged} 份通过；未过：{'；'.join(not_true)}", data
    if unusable:
        return UNDETERMINED, f"{judged}/{len(records)} 份可判且全过，另 {len(unusable)} 份不可解析", data
    return PASS, f"{judged}/{judged} 份报告 all_passed=True", data


def check_reports_bind(records: list[dict], generation_id: str) -> tuple[str, str, dict]:
    """每份报告绑的都是本次要发布的那个世代。

    这一条专门拦「拿旧报告凑数」：报告都在、都过，但绑的是上一代——
    那就等于这一代根本没被测过。
    """
    unusable = [r["rel_path"] for r in records if not r["valid_json"]]
    mismatched: list[str] = []
    bound = 0
    for rec in records:
        if not rec["valid_json"]:
            continue
        got = rec["generation_id"]
        if got == generation_id:
            bound += 1
        else:
            mismatched.append(
                f"{rec['rel_path']}（字段 {rec['gid_field']} 记的是 {got!r}）"
            )
    judged = len(records) - len(unusable)
    data = {
        "bound": bound,
        "judged": judged,
        "total": len(records),
        "want": generation_id,
        "mismatched": mismatched,
    }
    if mismatched:
        return (
            FAIL,
            f"{bound}/{judged} 份绑对世代；要发布的是 {generation_id}，"
            f"不一致 {len(mismatched)} 份：{'；'.join(mismatched)}",
            data,
        )
    if unusable:
        return UNDETERMINED, f"{bound}/{judged} 份绑对，另 {len(unusable)} 份不可解析", data
    return PASS, f"{bound}/{judged} 份都绑 {generation_id}", data


def check_candidate_status(status: str | None, generation_id: str) -> tuple[str, str, dict]:
    """候选必须是 READY。"""
    data = {"status": status, "want": release.READY, "generation_id": generation_id}
    if status == release.READY:
        return PASS, f"候选 {generation_id} 状态 {status}", data
    return FAIL, f"候选 {generation_id} 状态是 {status!r}，只有 {release.READY} 能发布", data


def check_candidate_verify(problems: list[str], generation_id: str) -> tuple[str, str, dict]:
    """候选必须能通过 `verify_generation()`（Manifest 记的字节就是盘上这些字节）。"""
    data = {"problems": list(problems), "generation_id": generation_id}
    if not problems:
        return PASS, f"候选 {generation_id} 自检 0 项不符", data
    shown = "; ".join(str(p) for p in problems[:8])
    more = "" if len(problems) <= 8 else f"（共 {len(problems)} 项，只列前 8）"
    return FAIL, f"候选自检 {len(problems)} 项不符：{shown}{more}", data


def check_reports_freshness(records: list[dict], created_at: datetime | None) -> tuple[str, str, dict]:
    """报告不能早于候选世代建成。

    语义：一份在候选建成之前生成的报告，不可能是「对候选的评测」。
    三份报告没有 `generated_at` 字段，对它们只能记 `undetermined`——
    这条路必须原样写出来，不许默认通过。
    """
    stale: list[str] = []
    checked = 0
    undetermined: list[str] = []
    for rec in records:
        if not rec["valid_json"]:
            continue
        when = parse_iso(rec["generated_at"])
        if when is None:
            undetermined.append(rec["rel_path"])
            continue
        if created_at is None:
            undetermined.append(f"{rec['rel_path']}（候选无 created_at，无从比较）")
            continue
        checked += 1
        if when < created_at:
            stale.append(
                f"{rec['rel_path']}（生成于 {rec['generated_at']}，早于候选建成 "
                f"{created_at.strftime('%Y-%m-%dT%H:%M:%SZ')}）"
            )
    data = {
        "checked": checked,
        "undetermined": undetermined,
        "stale": stale,
        "candidate_created_at": (
            created_at.strftime("%Y-%m-%dT%H:%M:%SZ") if created_at else None
        ),
    }
    if stale:
        return FAIL, f"{len(stale)} 份报告早于候选建成：{'；'.join(stale)}", data
    if undetermined:
        return (
            UNDETERMINED,
            f"{checked} 份可比且都不早于候选建成；另 {len(undetermined)} 份无时间戳字段，"
            f"本条对其无法判定：{'；'.join(undetermined)}",
            data,
        )
    return PASS, f"{checked}/{len(records)} 份报告都不早于候选建成", data


# ---------------------------------------------------------------- 汇总


def evaluate(
    generation_id: str,
    records: list[dict],
    *,
    status: str | None,
    problems: list[str],
    created_at: datetime | None,
) -> list[dict]:
    """跑全部六条判据，返回 `{id, name, verdict, ok, detail, data}` 列表。

    `ok` 只对 `pass` 为真；`undetermined` 的 `ok` 是 `False`，但它**不使整道门失败**
    （由 `all_passed` 只统计 `FAIL` 保证）。这样「无法判定」既不会被算成通过，
    也不会把整道门拖红。
    """
    plan = [
        ("reports_present", "五份门报告在场且可解析", check_reports_present(records)),
        ("reports_passed", "每份报告自身 all_passed=True", check_reports_passed(records)),
        (
            "reports_bind_generation",
            "每份报告绑的是本次要发布的世代",
            check_reports_bind(records, generation_id),
        ),
        (
            "candidate_ready",
            "候选状态是 READY",
            check_candidate_status(status, generation_id),
        ),
        (
            "candidate_verify",
            "候选 verify_generation() 无不符项",
            check_candidate_verify(problems, generation_id),
        ),
        (
            "reports_not_stale",
            "报告不早于候选建成",
            check_reports_freshness(records, created_at),
        ),
    ]
    out: list[dict] = []
    for check_id, name, (verdict, detail, data) in plan:
        out.append(
            {
                "id": check_id,
                "name": name,
                "verdict": verdict,
                "ok": verdict == PASS,
                "detail": detail,
                "data": data,
            }
        )
    return out


# ---------------------------------------------------------------- 主流程


def evaluate_generation(
    generation_id: str, reports: tuple[dict, ...] | list[dict] | None = None
) -> dict:
    """把「读候选 + 读五份报告 + 跑六条判据」合成一次调用，返回一个纯数据字典。

    拆出这一层是为了让 `scripts/release_cli.py publish` 能在真正切指针之前复用同一条判定：
    发布路径与清单门必须判**同一件事**，不许各写一套（同一判定写两处，迟早只更新一处）。
    """
    try:
        gen = release.Generation(generation_id)
    except release.ReleaseError as exc:
        raise PublishCheckError(str(exc)) from exc
    if not gen.dir.exists():
        raise PublishCheckError(f"世代目录不存在：{config.rel_to_root(gen.dir)}")
    manifest = gen.load()
    status = str(manifest.get("status", "")) or None
    created_at = parse_iso(manifest.get("created_at"))
    problems = release.verify_generation(gen)
    records = load_reports(REPORTS if reports is None else reports)
    results = evaluate(
        generation_id, records, status=status, problems=problems, created_at=created_at
    )
    failed = [r for r in results if r["verdict"] == FAIL]
    return {
        "generation_id": generation_id,
        "manifest": manifest,
        "status": status,
        "created_at": created_at,
        "problems": problems,
        "records": records,
        "checks": results,
        "failed": failed,
        "all_passed": not failed,
    }


def report(generation_id: str) -> int:
    print("=" * 78)
    print(f"发布前清单门  Python {sys.version.split()[0]}  {sys.platform}")
    print(f"要发布的世代 {generation_id}")
    print(f"报告目录 {config.rel_to_root(ROOT / 'data')}")
    print("口径：五份门报告各自绑的世代字段不同（见 REPORTS 登记表），逐份比对")

    verdict = evaluate_generation(generation_id)
    manifest = verdict["manifest"]
    records = verdict["records"]
    results = verdict["checks"]

    print("-" * 78)
    print("逐份报告")
    for rec in records:
        if not rec["valid_json"]:
            print(f"  {rec['rel_path']:<40s} 不可用（{'不存在' if not rec['exists'] else rec['error']}）")
            continue
        print(
            f"  {rec['rel_path']:<40s} 世代字段 {rec['gid_field']:<18s} "
            f"= {rec['generation_id']}  all_passed={rec['all_passed']}  "
            f"generated_at={rec['generated_at']}"
        )

    print("-" * 78)
    print("清单判定")
    for item in results:
        tag = {PASS: "PASS", FAIL: "FAIL", UNDETERMINED: "无法判定"}[item["verdict"]]
        print(f"[{tag}] {item['name']:<28s} {item['detail']}")

    failed = verdict["failed"]
    undetermined = [r for r in results if r["verdict"] == UNDETERMINED]
    passed = [r for r in results if r["verdict"] == PASS]
    all_passed = verdict["all_passed"]

    print("-" * 78)
    print(
        f"清单 {len(passed)}/{len(results)} 条成立"
        f"（另有 {len(undetermined)} 条无法判定，不计为通过）"
    )
    if failed:
        print("不得发布，未成立：")
        for item in failed:
            print(f"  - {item['name']}: {item['detail']}")

    atomic.atomic_write_json(
        REPORT_PATH,
        {
            "generation_id": generation_id,
            "tool": TOOL,
            "candidate": {
                "status": verdict["status"],
                "created_at": manifest.get("created_at"),
                "graph_artifact_id": manifest.get("graph_artifact_id"),
                "index_artifact_id": manifest.get("index_artifact_id"),
                "verify_problems": list(verdict["problems"]),
            },
            "checks": results,
            "passed": [r["id"] for r in passed],
            "failed": [r["id"] for r in failed],
            "undetermined": [r["id"] for r in undetermined],
            "reports": [
                {
                    "key": rec["key"],
                    "rel_path": rec["rel_path"],
                    "gid_field": rec["gid_field"],
                    "producer": rec["producer"],
                    "exists": rec["exists"],
                    "valid_json": rec["valid_json"],
                    "generation_id": rec["generation_id"],
                    "all_passed": rec["all_passed"],
                    "generated_at": rec["generated_at"],
                    "error": rec["error"],
                }
                for rec in records
            ],
            "all_passed": all_passed,
            "note": (
                "all_passed 只统计「不成立」的判据；undetermined 的判据既不算通过也不算失败。"
                "本门只证明五份报告都在、都过、都绑这一代，不证明这一代本身好。"
            ),
        },
    )
    print(f"报告已写 {config.rel_to_root(REPORT_PATH)}")
    print("=" * 78)
    return 0 if all_passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="发布前清单门（五份门报告对账）")
    parser.add_argument(
        "--generation", default=CANDIDATE_GID, help=f"要发布的世代，默认 {CANDIDATE_GID}"
    )
    args = parser.parse_args(argv)
    try:
        return report(args.generation)
    except PublishCheckError as exc:
        print(f"[2] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
