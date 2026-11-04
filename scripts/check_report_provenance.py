# -*- coding: utf-8 -*-
"""体检一份（或一组）门报告的可追溯性，按 `arkrag.provenance` 的七字段判三态。

## 为什么要有这个命令

P0 之前，`data/` 下九份门报告各写各的字段：`k2_perf.json` 有 `generated_at`、
`k2_hard_gates.json` 连时间戳都没有；而 `scripts/k2_publish_check.py` 的
`check_reports_freshness()` 对查不到时间戳的报告记一个 `undetermined` 之后，
整道门**仍然能过**——「查不了」被安静地当成「不用查」。

这个命令把「查」这件事变成一个可以独立跑、有退出码的动作：报告缺字段就非零，
不是打印一行提示然后继续。

## 三态

* `pass`   —— 该字段填了、格式合法；
* `fail`   —— 填了但不合法（比如 commit 不是 40 位十六进制、口径已作废）；
* `unknown`—— 没填 / 无法判断。

`unknown` **不算通过**。因此退出码：

* `0` —— 这份报告的全部七个字段都是 `pass`，且它自己的 `all_passed`/`passed`
       为真（若该报告带这个键）；
* `1` —— 至少一个字段不是 `pass`，或报告自报未通过；
* `2` —— 命令行用法错误 / 文件读不了。

## 用法

```
py -3 scripts/check_report_provenance.py data/k2_perf.json
py -3 scripts/check_report_provenance.py data/*.json        # PowerShell 不展开，逐个给
py -3 scripts/check_report_provenance.py --json data/k4_drill.json
```

`--json` 把每份报告的判定打成 JSON（便于别的脚本吞），默认打人读的表。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from arkrag import config, provenance  # noqa: E402

EXIT_OK = 0
EXIT_NOT_PASS = 1
EXIT_USAGE = 2

#: 报告自报「整体是否通过」的键。两道门用了不同拼写，两个都认。
SELF_VERDICT_KEYS = ("all_passed", "passed")


def _self_verdict(payload: dict) -> bool | None:
    """报告自己说通没通过；没有这个键就返回 None（不适用，不是 unknown）。"""
    for key in SELF_VERDICT_KEYS:
        if key in payload:
            return bool(payload[key])
    return None


def inspect(path: Path) -> dict:
    """体检单份报告，返回一个可 JSON 化的字典。"""
    rel = config.rel_to_root(path)
    if not path.exists():
        return {
            "path": rel,
            "exists": False,
            "verdict": provenance.UNKNOWN,
            "reason": "报告文件不存在——不存在不能算通过",
        }
    try:
        payload = json.loads(path.read_text("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {
            "path": rel,
            "exists": True,
            "verdict": provenance.FAIL,
            "reason": f"解析失败：{exc}",
        }
    if not isinstance(payload, dict):
        return {
            "path": rel,
            "exists": True,
            "verdict": provenance.FAIL,
            "reason": f"顶层不是对象（是 {type(payload).__name__}）",
        }

    result = provenance.verify(payload)
    self_verdict = _self_verdict(payload)
    verdict = result["verdict"]
    # 报告自报未通过 → 整体至少是 fail。反过来「自报通过」不能把 unknown 提成 pass。
    if self_verdict is False and verdict == provenance.PASS:
        verdict = provenance.FAIL
    return {
        "path": rel,
        "exists": True,
        "verdict": verdict,
        "fields": result["fields"],
        "missing": result["missing"],
        "invalid": result["invalid"],
        "self_verdict": self_verdict,
        "gate_ok": provenance.gate_ok(payload),
    }


def _print_human(rows: list[dict]) -> None:
    for row in rows:
        name = row["path"]
        if not row.get("exists"):
            print(f"{name:34s} unknown   {row['reason']}")
            continue
        if "fields" not in row:
            print(f"{name:34s} fail      {row.get('reason', '')}")
            continue
        mark = {"pass": "pass    ", "fail": "fail    ", "unknown": "unknown "}[row["verdict"]]
        extra = []
        if row["missing"]:
            extra.append(f"缺 {len(row['missing'])} 项：{','.join(row['missing'])}")
        if row["invalid"]:
            extra.append(f"不合法：{','.join(row['invalid'])}")
        if row["self_verdict"] is False:
            extra.append("报告自报未通过")
        print(f"{name:34s} {mark} {';'.join(extra) or '七字段齐全'}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="按七字段 / 三态体检门报告的可追溯性；unknown 不算通过。",
    )
    ap.add_argument("reports", nargs="+", help="要体检的报告路径（相对仓库根或绝对路径）")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出，便于其它脚本消费")
    args = ap.parse_args(argv)

    rows = []
    for raw in args.reports:
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        rows.append(inspect(path))

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        _print_human(rows)

    if not rows:  # pragma: no cover - argparse 的 nargs='+' 已挡住
        return EXIT_USAGE
    return EXIT_OK if all(row["verdict"] == provenance.PASS for row in rows) else EXIT_NOT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
