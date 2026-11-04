# -*- coding: utf-8 -*-
"""K3 第 5 条判据「超阈值必须人工审批」的校验门。

## 为什么要有这个脚本

`docs/升级计划_GraphRAG架构.md:430-436` 的 K3 有五条判据，前四条（实体减少 /
实体增加 / 边减少 / 索引体积）已经在 `docs/升级台账.md` 的「## K3 阶段」里做完了
（四层差异全部归因、未解释 0）。第 5 条「超阈值必须人工审批」此前**一次都没落地**：
两份 md 里「人工审批」0 处，没有阈值定义、没有审批记录格式、没有「跨了就拦住」的
机制。而 K3 这一轮的实测恰好是「略降但不显著」，**从未触发过阈值**——这正是最危险
的状态：一条从未被执行过一次的判据，等于没有。

所以本脚本把这条判据变成一个会变红的门，而不是一句文档承诺。

## 判据（逐指标）

指标从 `data/k3_recall_pairs.json` **自己算**，不抄 `scripts/k3_signif*.py` 打印好的
数字——否则阈值比对的是二手数字，跨了也看不出来。

- 未跨阈值 → 不需要审批，PASS；
- 跨了阈值，且 `data/k3_approvals.json` 里有该指标的**有效**审批记录 → PASS；
- 跨了阈值，没有有效审批记录 → **FAIL**，并点名指标、实测多少、阈值多少、差多少。

「有效」= 七个字段齐全（metric / measured / threshold / approved_by / approved_at /
decision / rationale）、`approved_by` 是自然人（空串、TBD、agent、auto 之类占位符
一律无效——审批是人做的动作，程序不能自批）、`approved_at` 是可解析日期、
`decision` 是明确的通过/拒绝、`measured` 与本次实测差 ≤ 1e-9（审批只能覆盖它当时
看到的那一次实测，换一次实测就失效）。

`direction: "red_line"` 的行（未解释差异数）不接受审批：归因能力跟不上改动时，
签字不能替代归因。

## 两种口径必须显式区分

R@5 / R@20 在口径 A 与口径 B 下逐位相同；**RR 不同**（A 丢弃 null → n=229；
B 未命中填 0 → n=300）。阈值表里 RR 那行写明按**口径 B** 判（与 `scripts/ask.py`
的评测器同口径，否则门判的和线上量的不是一件事），口径 A 作为申报。

## 唯一事实源

阈值只在 `data/k3_thresholds.json` 里写一份。本脚本**只读不写**阈值，也不在代码里
藏任何数字常量（`1e-9` 是浮点比较容差，不是阈值）。本项目已经因为「同一判定写
两处、取最松者生效」出过三次同型事故，`tests/test_single_source_of_truth.py` 就是
为这个立的。

## 用法

    py -3 scripts/k3_thresholds.py                    # 正常跑，写 data/k3_threshold_gate.json
    py -3 scripts/k3_thresholds.py --thresholds X --pairs Y --approvals Z --out W
    py -3 scripts/k3_thresholds.py --unexplained 0    # 跳过重跑归因脚本，直接给未解释数

路径覆盖参数是为了反向实测与测试不必改动真实文件。

未解释差异数由 `scripts/k3_attribution.py`（12 秒左右）现跑现取，**不在这里重写一份
归因逻辑**——那会变成第二个事实源。解析不出这个数就硬错退出（退出码 2），不猜。

## 退出码

- `0` = 判据成立：没有指标跨阈值，或每个跨阈值的指标都有有效人工审批；
- `1` = 有指标跨阈值且没有有效审批记录（判据不成立，不许发布）；
- `2` = 硬错误：输入文件缺失/损坏、两套口径对 R@5/R@20 不一致、归因数取不到等。
"""
from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
# Windows 控制台默认 cp936，打印中文/箭头会抛 UnicodeEncodeError；
# `> file` 重定向会把输出写成 GBK 字节，与本项目其它脚本的「原样粘贴」约定冲突。
# 守卫是必要的：pytest 捕获期间 `sys.stdout` 可能不是可重配置的文本流。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_THRESHOLDS = ROOT / "data" / "k3_thresholds.json"
DEFAULT_APPROVALS = ROOT / "data" / "k3_approvals.json"
DEFAULT_PAIRS = ROOT / "data" / "k3_recall_pairs.json"
DEFAULT_GENS = ROOT / "data" / "arkrag" / "generations"
DEFAULT_OUT = ROOT / "data" / "k3_threshold_gate.json"
ATTRIBUTION = ROOT / "scripts" / "k3_attribution.py"

# 浮点比较容差（不是阈值）。阈值一律来自 data/k3_thresholds.json。
EPS = 1e-9

# approved_by 里出现这些值就说明「没有真人签字」。
PLACEHOLDERS = {
    "", "-", "n/a", "na", "none", "null", "tbd", "todo", "placeholder", "?",
    "agent", "auto", "bot", "system", "ci", "robot", "script",
    "自动化", "程序", "待定", "无",
}
APPROVED_DECISIONS = {"approved", "approve", "accepted", "accept", "批准", "通过", "接受"}
REJECTED_DECISIONS = {"rejected", "reject", "denied", "deny", "否决", "拒绝", "驳回"}


class GateError(Exception):
    """硬错误：判据无法评估。**不许吞掉当成通过。**"""


# ------------------------------------------------------------------ 读入


def load_json(path: Path, what: str) -> dict:
    if not path.exists():
        raise GateError(f"{what} 不存在：{path}（缺输入不等于判据成立）")
    try:
        with io.open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise GateError(f"{what} 不是合法 JSON：{path}：{exc}") from exc


# ------------------------------------------------------------------ 指标


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def compute_recall_metrics(pairs: dict) -> dict:
    """从逐题明细自己算 R@5 / R@20 / RR，两套口径都算。

    口径 A（`scripts/k3_signif.py`）：两代任一为 None 就丢弃该题。
    口径 B（`scripts/k3_signif_eval.py`）：分母只取有 gold 的题，未命中填 0.0。

    R@5 / R@20 天然无 null（有 gold 时就是命中比例），两套口径因此逐位相同——
    这不是假设，是这里算完再断言的事实；不相等就硬错，说明 pairs 文件形状变了。
    """
    old, new = pairs["old"], pairs["new"]
    common = sorted(set(old) & set(new))
    graded = [q for q in common
              if old[q].get("r20") is not None and new[q].get("r20") is not None]
    out: dict = {
        "n_common": len(common),
        "n_graded": len(graded),
        "generations": pairs.get("generations") or {},
        "calibers": {},
    }
    for caliber, name in (("A", "A"), ("B", "B")):
        block: dict = {"n": {}, "old": {}, "new": {}, "delta": {}}
        for metric in ("r5", "r20", "rr"):
            ps: list[tuple[float, float]] = []
            for q in graded:
                a, b = old[q].get(metric), new[q].get(metric)
                if caliber == "A":
                    if a is None or b is None:
                        continue          # 口径 A：丢弃
                    ps.append((float(a), float(b)))
                else:
                    ps.append((0.0 if a is None else float(a),
                               0.0 if b is None else float(b)))  # 口径 B：0 填充
            if not ps:
                raise GateError(f"口径 {name} 的 {metric} 无可比题（分母为 0）")
            x, y = _mean([p[0] for p in ps]), _mean([p[1] for p in ps])
            block["n"][metric] = len(ps)
            block["old"][metric] = x
            block["new"][metric] = y
            block["delta"][metric] = y - x
        out["calibers"][caliber] = block

    for metric in ("r5", "r20"):
        a = out["calibers"]["A"]["delta"][metric]
        b = out["calibers"]["B"]["delta"][metric]
        if abs(a - b) > EPS:
            raise GateError(
                f"{metric} 在口径 A/B 下 Δ 不一致（{a!r} vs {b!r}）——"
                f"pairs 文件的 null 形状变了，两套口径的前提不成立，不许继续判"
            )
    return out


def _pct(old: float, new: float) -> float:
    if old == 0:
        raise GateError("旧代计数为 0，百分比变化无定义")
    return (new - old) / old * 100.0


def compute_structure_metrics(gens_root: Path, generations: dict) -> dict:
    """两代 Manifest 的结构量与词法索引字节数。"""
    out: dict = {"old": {}, "new": {}}
    for side in ("old", "new"):
        gid = str((generations or {}).get(side) or "")
        if not gid:
            raise GateError(f"pairs 文件里没有 {side} 世代 id，无法比对结构量")
        gdir = gens_root / gid
        man = load_json(gdir / "manifest.json", f"{side} 世代 Manifest（{gid}）")
        lex = gdir / "index" / "lexical.json"
        if not lex.exists():
            raise GateError(f"{side} 世代的词法索引不存在：{lex}")
        out[side] = {
            "generation_id": gid,
            "entity_count": int(man["entity_count"]),
            "edge_count": int(man["edge_count"]),
            "lexical_bytes": lex.stat().st_size,
        }
    out["delta_pct"] = {
        "entity_count": _pct(out["old"]["entity_count"], out["new"]["entity_count"]),
        "edge_count": _pct(out["old"]["edge_count"], out["new"]["edge_count"]),
        "lexical_bytes": _pct(out["old"]["lexical_bytes"], out["new"]["lexical_bytes"]),
    }
    return out


def read_unexplained() -> int:
    """现跑 `scripts/k3_attribution.py`，取它自己打印的「未解释合计」。

    不在这里重写归因逻辑：那会变成第二个事实源，也是本项目三次同型事故的形状。
    解析不出就硬错——不猜、不当 0。
    """
    if not ATTRIBUTION.exists():
        raise GateError(f"归因脚本不存在：{ATTRIBUTION}")
    proc = subprocess.run(
        [sys.executable, str(ATTRIBUTION)],
        cwd=str(ROOT), capture_output=True, encoding="utf-8", errors="replace",
        timeout=600,
    )
    text = (proc.stdout or "") + (proc.stderr or "")
    hits = re.findall(r"未解释合计（边 \+ 断言）：(\d+)", text)
    if not hits:
        raise GateError(
            f"归因脚本没有打印「未解释合计」行（退出码 {proc.returncode}），"
            f"无法确认红线，判据不成立"
        )
    return int(hits[-1])


def measured_values(recall: dict, structure: dict, unexplained: int) -> dict:
    """指标名 → 本次实测值（**判的是 Δ，不是新代绝对值**）。"""
    return {
        "r20_delta": recall["calibers"]["A"]["delta"]["r20"],
        "r5_delta": recall["calibers"]["A"]["delta"]["r5"],
        "rr_delta_A": recall["calibers"]["A"]["delta"]["rr"],
        "rr_delta_B": recall["calibers"]["B"]["delta"]["rr"],
        "entity_count_pct": structure["delta_pct"]["entity_count"],
        "edge_count_pct": structure["delta_pct"]["edge_count"],
        "lexical_bytes_pct": structure["delta_pct"]["lexical_bytes"],
        "unexplained_differences": float(unexplained),
    }


# 阈值表的 metric 名 → measured_values 里的键。集中一处映射，避免各判据自己拼名字。
METRIC_KEY = {
    "r20_delta": "r20_delta",
    "r5_delta": "r5_delta",
    "rr_delta": "rr_delta_B",          # 阈值表声明按口径 B 判
    "entity_count_drop_pct": "entity_count_pct",
    "entity_count_rise_pct": "entity_count_pct",
    "edge_count_drop_pct": "edge_count_pct",
    "lexical_bytes_drop_pct": "lexical_bytes_pct",
    "unexplained_differences": "unexplained_differences",
}


def breaches(row: dict, value: float) -> bool:
    """这条阈值行被跨过了没有。方向语义只在阈值表里定义。"""
    direction, limit = row.get("direction"), row.get("threshold")
    if limit is None:
        raise GateError(f"阈值行 {row.get('metric')} 的 threshold 为空，无法判")
    if direction == "lower_is_worse":
        return value < float(limit) - EPS
    if direction == "higher_is_worse":
        return value > float(limit) + EPS
    if direction == "red_line":
        return value > float(limit) + EPS
    raise GateError(
        f"阈值行 {row.get('metric')} 的 direction={direction!r} 不认识"
        f"（只支持 lower_is_worse / higher_is_worse / red_line）"
    )


def gap_text(row: dict, value: float) -> str:
    """**跨线后**超了多少（带方向）。只在真跨线时用，措辞不能含糊。"""
    limit = float(row["threshold"])
    diff = value - limit
    if row.get("direction") in ("higher_is_worse", "red_line"):
        return f"超线 {diff:+.6f}（实测 {value:+.6f} vs 阈值 {limit:+.6f}）"
    return f"超线 {abs(diff):.6f}（实测 {value:+.6f} vs 阈值 {limit:+.6f}）"


def margin_text(row: dict, value: float) -> str:
    """**未跨线**时还剩多少余量。与 gap_text 分开，避免把余量说成「超线」。"""
    limit = float(row["threshold"])
    diff = value - limit
    # + 0.0 是为了把 -0.0 归一成 0.0：红线刚好压线时不该打印「距线尚有 -0.000000」。
    if row.get("direction") in ("higher_is_worse", "red_line"):
        return f"距线尚有 {-diff + 0.0:.6f}（实测 {value:+.6f} vs 阈值 {limit:+.6f}）"
    return f"距线尚有 {diff + 0.0:.6f}（实测 {value:+.6f} vs 阈值 {limit:+.6f}）"


# ------------------------------------------------------------------ 审批


def is_placeholder_approver(name) -> bool:
    if not isinstance(name, str):
        return True
    return name.strip().lower() in PLACEHOLDERS


def parse_approved_at(value) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    head = value.strip()[:10]
    try:
        _dt.date.fromisoformat(head)
    except ValueError:
        return None
    return head


def validate_approval(rec: dict, row: dict, value: float) -> list[str]:
    """返回问题清单；空清单 = 这条记录有效。"""
    problems: list[str] = []
    required = ("metric", "measured", "threshold", "approved_by",
                "approved_at", "decision", "rationale")
    missing = [f for f in required if f not in rec]
    if missing:
        problems.append(f"缺字段 {missing}")
    if is_placeholder_approver(rec.get("approved_by")):
        problems.append(
            f"approved_by={rec.get('approved_by')!r} 是占位符，审批必须是人"
        )
    if parse_approved_at(rec.get("approved_at")) is None:
        problems.append(f"approved_at={rec.get('approved_at')!r} 不是可解析日期")
    decision = str(rec.get("decision") or "").strip().lower()
    if decision in REJECTED_DECISIONS:
        problems.append(f"decision={rec.get('decision')!r} 是拒绝，不能放行")
    elif decision not in APPROVED_DECISIONS:
        problems.append(f"decision={rec.get('decision')!r} 不是明确的通过结论")
    if not str(rec.get("rationale") or "").strip():
        problems.append("rationale 为空（审批必须写为什么同意）")
    measured = rec.get("measured")
    if not isinstance(measured, (int, float)) or isinstance(measured, bool):
        problems.append(f"measured={measured!r} 不是数值")
    elif abs(float(measured) - value) > EPS:
        problems.append(
            f"记录里的 measured={float(measured)!r} 与本次实测 {value!r} 不符"
            f"（差 {abs(float(measured) - value)!r} > {EPS}）"
        )
    thr = rec.get("threshold")
    if thr is None or not isinstance(thr, (int, float)) or isinstance(thr, bool):
        problems.append(f"threshold={thr!r} 不是数值")
    elif abs(float(thr) - float(row["threshold"])) > EPS:
        problems.append(
            f"记录里的 threshold={thr!r} 与阈值表 {row['threshold']!r} 不一致"
            f"（审批必须针对当前登记的那条线）"
        )
    return problems


def judge(row: dict, value: float, approvals: list[dict]) -> dict:
    """单条阈值行的判定结果。"""
    metric = str(row.get("metric"))
    breach = breaches(row, value)
    check = {
        "metric": metric,
        "label": row.get("label"),
        "direction": row.get("direction"),
        "threshold": row.get("threshold"),
        "measured": value,
        "caliber": row.get("caliber"),
        "breached": breach,
        "approvals_seen": 0,
        "ok": True,
        "detail": "",
    }
    if not breach:
        check["detail"] = f"未跨阈值，不需要审批（{margin_text(row, value)}）"
        return check

    if row.get("direction") == "red_line":
        check["ok"] = False
        check["detail"] = (
            f"跨红线且红线不接受审批：{gap_text(row, value)}——"
            f"归因能力跟不上改动时，签字不能替代归因"
        )
        return check

    mine = [a for a in approvals
            if isinstance(a, dict) and str(a.get("metric") or "").strip() == metric]
    check["approvals_seen"] = len(mine)
    if not mine:
        check["ok"] = False
        check["detail"] = (
            f"跨阈值但没有任何该指标的审批记录：{gap_text(row, value)}；"
            f"指标 {metric}、实测 {value:+.6f}、阈值 {float(row['threshold']):+.6f}"
        )
        return check

    reasons: list[str] = []
    for rec in mine:
        problems = validate_approval(rec, row, value)
        if not problems:
            check["ok"] = True
            check["detail"] = (
                f"跨阈值（{gap_text(row, value)}），有有效人工审批："
                f"{rec.get('approved_by')} / {parse_approved_at(rec.get('approved_at'))} / "
                f"{rec.get('decision')}"
            )
            check["approval"] = {
                "approved_by": rec.get("approved_by"),
                "approved_at": rec.get("approved_at"),
                "decision": rec.get("decision"),
            }
            return check
        reasons.append(f"approved_by={rec.get('approved_by')!r}：" + "；".join(problems))

    check["ok"] = False
    check["detail"] = (
        f"跨阈值（{gap_text(row, value)}）且审批记录全部无效 → " + " | ".join(reasons)
    )
    return check


def judge_absolute(row: dict, recall: dict) -> dict:
    """absolute 行只申报：给当前值与差距，**判据不在 K3**。

    刻意不把它写成 PASS：`Recall@20 ≥ 0.90` 这类绝对门现在是**未达标**，
    写成 PASS 就是本项目明令禁止的「看起来像个重大发现」。
    """
    src = row.get("measured_from")
    value = None
    n = None
    if src and recall:
        value = recall["calibers"]["B"]["new"].get(src)
        n = recall["calibers"]["B"]["n"].get(src)
    limit = row.get("threshold")
    check = {
        "metric": row.get("metric"),
        "label": row.get("label"),
        "direction": row.get("direction"),
        "threshold": limit,
        "measured": value,
        "n": n,
        "k3_role": row.get("k3_role") or "申报，不判",
        "ok": True,                       # 不判 → 不参与通过/失败计数
        "declared": True,
        "met": None,                      # 没输入 → 不评估，既不是达标也不是未达标
    }
    if value is None:
        check["detail"] = "无实测输入（本门不评估这项，也不假装通过）"
        return check
    if limit is None:
        check["detail"] = f"当前 {value:.4f}（n={n}）；本行不设绝对阈值，仅申报基线"
        return check
    limit = float(limit)
    if row.get("direction") == "at_most":
        met = value <= limit + EPS
        check["detail"] = (
            f"当前 {value:.4f}（n={n}） vs ≤ {limit:.4f} → "
            + ("达标" if met else f"未达标 {value - limit:+.4f}")
        )
    else:
        met = value >= limit - EPS
        check["detail"] = (
            f"当前 {value:.4f}（n={n}） vs ≥ {limit:.4f} → "
            + ("达标" if met else f"未达标 {value - limit:+.4f}")
        )
    check["met"] = met
    return check


# ------------------------------------------------------------------ 主流程


def build_report(thresholds: dict, approvals_doc: dict, pairs: dict,
                 gens_root: Path, unexplained: int) -> dict:
    recall = compute_recall_metrics(pairs)
    structure = compute_structure_metrics(gens_root, recall["generations"])
    values = measured_values(recall, structure, unexplained)

    raw = approvals_doc.get("approvals")
    if not isinstance(raw, list):
        raise GateError("审批记录的 approvals 字段不是数组")
    approvals = [a for a in raw if isinstance(a, dict)]
    dropped = len(raw) - len(approvals)

    rows = thresholds.get("delta")
    if not isinstance(rows, list) or not rows:
        raise GateError("阈值表缺 delta 段（或为空）—— 没有阈值就谈不上审批")

    checks = [judge(row, values[METRIC_KEY[str(row["metric"])]], approvals)
              for row in rows]
    absolute = [judge_absolute(r, recall) for r in (thresholds.get("absolute") or [])]

    failed = [c for c in checks if not c["ok"]]
    return {
        "schema": "k3-threshold-gate-1",
        "thresholds_file": str(DEFAULT_THRESHOLDS.name),
        "generations": recall["generations"],
        "denominators": {
            "n_common_questions": recall["n_common"],
            "n_graded_questions": recall["n_graded"],
            "n_delta_metrics": len(checks),
            "n_absolute_declared": len(absolute),
            "n_approval_records": len(raw),
            "n_approval_records_usable": len(approvals),
            "n_approval_records_dropped_not_object": dropped,
            "n_unexplained": unexplained,
        },
        "recall": {
            side: {
                "n": recall["calibers"][side]["n"],
                "old": recall["calibers"][side]["old"],
                "new": recall["calibers"][side]["new"],
                "delta": recall["calibers"][side]["delta"],
            } for side in ("A", "B")
        },
        "structure": structure,
        "checks": checks,
        "absolute_declared": absolute,
        "all_passed": not failed,
        "failed_metrics": [c["metric"] for c in failed],
        "note": (
            "all_passed 只表示「K3 第 5 条判据成立」：没有指标跨阈值，或每个跨阈值的"
            "指标都有有效人工审批。它**不表示**审批流程被真实执行过——本轮没有指标"
            "跨阈值，因此没有发生任何人工审批。"
        ),
    }


def print_report(report: dict, thresholds: dict) -> None:
    den = report["denominators"]
    rec = report["recall"]
    st = report["structure"]
    print("=" * 78)
    print("K3 第 5 条判据：超阈值必须人工审批")
    print("=" * 78)
    print(f"阈值唯一事实源：data/k3_thresholds.json（schema={thresholds.get('schema')}）")
    print(f"旧代 {report['generations'].get('old')}")
    print(f"新代 {report['generations'].get('new')}")
    print()
    print("== 分母 ==")
    print(f"  共同题 {den['n_common_questions']}；有 gold 可评 {den['n_graded_questions']}"
          f"（无 gold 剔除 {den['n_common_questions'] - den['n_graded_questions']}）")
    print(f"  delta 判据 {den['n_delta_metrics']} 条；absolute 申报 {den['n_absolute_declared']} 条")
    print(f"  审批记录 {den['n_approval_records']} 条（可用 {den['n_approval_records_usable']}，"
          f"非对象被丢弃 {den['n_approval_records_dropped_not_object']}）")
    print()
    print("== 召回指标（两套口径逐条分开，Δ = 新 − 旧）==")
    for side, name in (("A", "口径 A（丢弃 null）"), ("B", "口径 B（0 填充，与 ask.py 同口径）")):
        blk = rec[side]
        print(f"  {name}")
        for metric, label in (("r5", "R@5"), ("r20", "R@20"), ("rr", "RR")):
            print(f"    {label:<5s} n={blk['n'][metric]:<4d} "
                  f"旧 {blk['old'][metric]:.6f} → 新 {blk['new'][metric]:.6f} "
                  f"Δ {blk['delta'][metric]:+.6f}")
    print()
    print("== 结构量 ==")
    for key, label in (("entity_count", "实体数"), ("edge_count", "边数"), ("lexical_bytes", "词法索引字节")):
        o, n = st["old"][key], st["new"][key]
        print(f"  {label:<12s} 旧 {o:>12,d} → 新 {n:>12,d}  "
              f"Δ {st['delta_pct'][key]:+.4f}%")
    print(f"  未解释差异 {den['n_unexplained']}（红线 0）")
    print()
    print("== delta 判据（逐指标）==")
    for c in report["checks"]:
        tag = "PASS" if c["ok"] else "FAIL"
        print(f"  [{tag}] {c['metric']:<24s} 阈值 {float(c['threshold']):+.4f}"
              f"  实测 {c['measured']:+.6f}  {c['detail']}")
    print()
    print("== absolute 声明（F4 的绝对门，K3 不重复判，只申报差距）==")
    for c in report["absolute_declared"]:
        print(f"  [申报] {str(c['metric']):<32s} {c['detail']}")
    print()
    bad = report["failed_metrics"]
    if bad:
        print(f"结论：{len(bad)}/{len(report['checks'])} 条 delta 判据不成立 → "
              f"{'、'.join(bad)} 跨阈值且无有效人工审批，不许发布")
    else:
        print(f"结论：{len(report['checks'])}/{len(report['checks'])} 条 delta 判据成立"
              f"（本轮无指标跨阈值，因此未发生人工审批；这不是「审批流程已验证通过」）")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="K3 第 5 条：超阈值必须人工审批")
    ap.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    ap.add_argument("--approvals", type=Path, default=DEFAULT_APPROVALS)
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--gens", type=Path, default=DEFAULT_GENS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--unexplained", type=int, default=None,
                    help="直接给未解释差异数，跳过重跑 scripts/k3_attribution.py")
    args = ap.parse_args(argv)

    try:
        thresholds = load_json(args.thresholds, "阈值表")
        approvals_doc = load_json(args.approvals, "审批记录")
        pairs = load_json(args.pairs, "配对召回明细")
        unexplained = args.unexplained if args.unexplained is not None else read_unexplained()
        report = build_report(thresholds, approvals_doc, pairs, args.gens, unexplained)
    except GateError as exc:
        print(f"[HARD ERROR] {exc}")
        print("退出码 2：判据无法评估。缺输入 / 算不出来一律不算通过。")
        return 2

    print_report(report, thresholds)

    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # 显式 newline="\n"：Windows 默认会把 \n 翻成 \r\n，产物要求 CRLF=0。
    with io.open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)

    print(f"\n写盘：{args.out.name}（all_passed={report['all_passed']}）")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
