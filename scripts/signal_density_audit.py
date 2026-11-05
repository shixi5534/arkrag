# -*- coding: utf-8 -*-
"""把「信号密度」变成报告里看得见的一列，并给出「密度 4+ 无效」的分桶实测。

## 为什么要有这个脚本

`data/a_p2_round1.json` 里有 `cooc_boosted` 与 `units` 两个数，但没有它们的比值。
Lead 2026-11-05 复核发现：cooc 加成的成败**完全由密度 `boosted / units` 决定**，
与题型无关 —— 密度 0–1 救 2、1–2 救 3、2–4 救 1、**4+ 的 35 题一分没挣到**。
机制是常数加成在命中块数远超上下文容量（units 中位 5）时全体同加同量，相对
顺序不变，等于没加；与 `hop_line` 的「20/20 全抬」是同一条规律。

那个比值在原始报告里算不出来，于是下一轮调参的人只能看到「R@20 涨了」。这个
脚本把它算出来、分桶、连同一对并列召回指标的包含关系一起写进报告，报告本身
要过 `arkrag.provenance` 的门。

## 为什么不改 `src/arkrag/retrieve.py`

`units` 只有 `build_context()` 之后才知道，`search()` 内部算不出来。把
`context_units` 硬塞进 `search().stats` 会造出一个「时有时无」的字段，而 0 会被
读成「密度真的是零」。所以密度只能在上层算 —— 这就是本脚本存在的位置。

## 用法

```
py -3 scripts/signal_density_audit.py                       # 读默认输入，写默认报告
py -3 scripts/signal_density_audit.py --in data/x.json --out data/y.json
py -3 scripts/signal_density_audit.py --check               # 只校验已写报告，不重算
```

退出码：`0` 合规（且 `--check` 时与磁盘一致）/ `1` 不合规或与磁盘不一致 /
`2` 输入不可用。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from arkrag import atomic, config, provenance  # noqa: E402

DEFAULT_INPUT = "data/a_p2_round1.json"
DEFAULT_OUTPUT = "data/signal_density_audit.json"

SCHEMA = "ark-signal-density-1"

#: 分桶边界。取自 Lead 2026-11-05 的实测表：0–1 / 1–2 / 2–4 / 4+，
#: 最后一桶是「净增益为 0」的那 35 题所在的区间。
BUCKETS: tuple[tuple[float, float | None, str], ...] = (
    (0.0, 1.0, "0-1"),
    (1.0, 2.0, "1-2"),
    (2.0, 4.0, "2-4"),
    (4.0, None, "4+"),
)

#: 两种分母口径，**必须分开列**。这是本项目栽过四次的同一个坑：分子和分母
#: 来自不同的集合，每个数都真、合起来意思不对。
#:
#: * `all_questions` —— 350 道题全算。`cooc_applied == 0` 的 215 题密度为 0，
#:   全部落进 0–1 桶（该桶 253 题），把「低密度有效」这个结论稀释成噪声。
#: * `boost_applied_only` —— 只算真正被加成过的 135 题。Lead 2026-11-05 的表
#:   就是这个口径（0–1 n=38 / 1–2 n=41 / 2–4 n=21 / 4+ n=35，合计 135）。
#:   它回答的问题是「在加成生效的题里，密度如何决定成败」——那才是要问的问题。
CALIBERS_DENSITY: tuple[str, ...] = ("all_questions", "boost_applied_only")

#: 需要并列给出包含关系的召回类维度。这三行**不是三份独立证据**：
#: `required_citations` 的块集恒等于 `gold_chunk_ids`，所以 `all_gold` 蕴含
#: `any_gold`，而单 gold 块题（217/350 = 62.0%）上两者数学恒等。
PARALLEL_DIMS: tuple[str, ...] = ("any_gold", "all_gold", "covered")

#: 分桶时要按哪些「救/丢」指标各算一遍。**必须分开算，不能只挑一个**：
#: Lead 2026-11-05 那张表的「1–2 桶救 3」是 `any_gold` 口径，而同一桶的
#: `covered` 只有救 2 —— 两个数都对，问的问题不同。只报一个，读者会拿另一个
#: 去对账然后发现对不上。
DENSITY_METRICS: tuple[str, ...] = ("any_gold", "all_gold", "covered")


def bucket_of(density: float | None) -> str:
    """密度落到哪个桶。`None`（没有上下文单元）单列，不混进 0–1。"""
    if density is None:
        return "n/a"
    for low, high, name in BUCKETS:
        if high is None:
            if density >= low:
                return name
        elif low <= density < high:
            return name
    return "n/a"


def density_rows(per_question: dict) -> dict:
    """为每个配置标签产出一张「每题的密度三件套 + 各指标的救/丢」的表。

    「救」= 候选该题指标为真而基线为假；「丢」反过来。三个指标（`any_gold` /
    `all_gold` / `covered`）各存一组，因为它们不是同一个问题：`any_gold` 问
    「gold 块有没有进上下文」，`covered` 问「回答里有没有把引用全发出来」。同一
    桶里 `any_gold` 救 3 而 `covered` 救 2 是很正常的（那题进了上下文但没发引用）。
    """
    tags = list(per_question or {})
    if len(tags) < 2:
        return {}
    base_tag = tags[0]
    base = {row["question_id"]: row for row in per_question[base_tag]}
    out = {}
    for tag in tags[1:]:
        rows = []
        for row in per_question[tag]:
            qid = row["question_id"]
            old = base.get(qid) or {}
            block = provenance.signal_density_block(
                {"cooc_boosted": row.get("cooc_boosted")},
                {"units": row.get("units")},
            )
            saved = {}
            lost = {}
            for metric in DENSITY_METRICS:
                saved[metric] = bool(row.get(metric)) and not bool(old.get(metric))
                lost[metric] = bool(old.get(metric)) and not bool(row.get(metric))
            rows.append({
                "question_id": qid,
                "category": row.get("category"),
                "origin": row.get("origin"),
                **block,
                # `signal_boosted` 是「命中了几块」，`signal_applied` 是「其中真正
                # 把加成写进 score 的有几块」。两者差 6 题（141 vs 135），差的那几
                # 题是命中了但增幅被夹成零的。分母口径必须用 `applied`：只有它才
                # 代表「这题的分数真的被动过」。
                "signal_applied": int(row.get("cooc_applied") or 0),
                "saved": saved,
                "lost": lost,
            })
        out[tag] = rows
    return out


def bucket_summary(rows: list[dict], *, caliber: str = "all_questions") -> list[dict]:
    """按桶聚合救/丢/净。每个桶给出三个指标各一组数。

    `caliber` 决定分母（见 `CALIBERS_DENSITY` 的注释）：
    `all_questions` 全算，`boost_applied_only` 只算 `signal_applied > 0` 的题。

    **用 `signal_applied` 而不是 `signal_boosted` 做筛选**：实测 141 题有命中、
    135 题真正把加成写进了 score（差 6 题是命中了但增幅被夹成零的）。只有后者
    才代表「这题的分数真的被动过」，用前者会把 6 道「其实没加成」的题算进
    「加成有效区」，与 Lead 2026-11-05 的表差 6 题（44 vs 38）。

    返回 `[{bucket, n, any_gold:{saved,lost,net}, all_gold:{...}, covered:{...}}]`。
    """
    if caliber not in CALIBERS_DENSITY:
        raise ValueError(f"未知口径 {caliber!r}，允许 {list(CALIBERS_DENSITY)}")
    order = [name for _, _, name in BUCKETS] + ["n/a"]

    def blank(name):
        cell = {"bucket": name, "n": 0}
        for metric in DENSITY_METRICS:
            cell[metric] = {"saved": 0, "lost": 0, "net": 0}
        return cell

    table = {name: blank(name) for name in order}
    for row in rows:
        if caliber == "boost_applied_only" and not int(row.get("signal_applied") or 0):
            continue
        cell = table[bucket_of(row.get("density"))]
        cell["n"] += 1
        for metric in DENSITY_METRICS:
            cell[metric]["saved"] += int(bool((row.get("saved") or {}).get(metric)))
            cell[metric]["lost"] += int(bool((row.get("lost") or {}).get(metric)))
    for cell in table.values():
        for metric in DENSITY_METRICS:
            cell[metric]["net"] = cell[metric]["saved"] - cell[metric]["lost"]
    return [table[name] for name in order if table[name]["n"] > 0]


def gain_set_rows(comparisons: dict) -> list[dict]:
    """从 `comparisons` 里摘出并列的召回类维度，带上增益题集。

    只取 `PARALLEL_DIMS` 里出现且有 `gain_ids` 的维度：其它维度（引用条数、
    延迟）不是召回类，混进来会把包含关系算成一堆无意义的空集。
    """
    rows = []
    for _tag, block in (comparisons or {}).items():
        dims = block.get("dims") or {}
        for name in PARALLEL_DIMS:
            dim = dims.get(name)
            if not isinstance(dim, dict):
                continue
            ids = [str(x) for x in dim.get("gain_ids") or []]
            rows.append({
                "metric": name,
                "gain": int(dim.get("gain") or 0),
                "lose": int(dim.get("lose") or 0),
                "net": int(dim.get("net") or 0),
                "gain_ids": ids,
                "lose_ids": [str(x) for x in dim.get("lose_ids") or []],
            })
    return rows


def build(payload: dict) -> dict:
    """纯函数：输入一份 P2 风格报告，输出本脚本的报告。"""
    per_question = payload.get("per_question") or {}
    comparisons = payload.get("comparisons") or {}
    rows_by_tag = density_rows(per_question)
    gains = gain_set_rows(comparisons)
    table = provenance.compute_gain_set_relations(gains)
    for row in gains:
        row["relation_table"] = table

    buckets = {
        tag: {caliber: bucket_summary(rows, caliber=caliber) for caliber in CALIBERS_DENSITY}
        for tag, rows in rows_by_tag.items()
    }
    report = {
        "schema": SCHEMA,
        "generated_at": provenance.now_iso(),
        "generation_id": str(payload.get("generation") or ""),
        "git_commit": provenance._git_commit(),
        "code_sha256_16": provenance.code_sha256_16(),
        "dataset": provenance.dataset_block(),
        "inputs": provenance.inputs_block([DEFAULT_INPUT]),
        "environment": provenance.environment_block(),
        "source_report": DEFAULT_INPUT,
        "calibers": list(CALIBERS_DENSITY),
        "buckets": BUCKETS_DOC(),
        "bucket_summary": buckets,
        "parallel_recall": gains,
        "gain_set_relations": table,
    }
    return report


def BUCKETS_DOC() -> list[dict]:
    return [
        {"low": low, "high": high, "name": name} for low, high, name in BUCKETS
    ]


def self_check(report: dict) -> list[str]:
    """报告自己必须过四道门：七字段、口径齐全、密度三件套、并列召回包含关系。

    「口径齐全」这一条是补的：单口径的旧报告结构上也能解析，如果只查密度三件套
    它会照样通过——而它缺的正是「两种分母」这件事本身。
    """
    problems = []
    verdict = provenance.verify(report)
    if verdict["verdict"] != provenance.PASS:
        problems.append(f"七字段：{verdict['detail']}（缺 {verdict['missing']}）")

    declared = report.get("calibers")
    if list(declared or []) != list(CALIBERS_DENSITY):
        problems.append(
            f"报告声明的口径是 {declared!r}，应为 {list(CALIBERS_DENSITY)}；"
            "只给一种分母的报告无法回答「密度效应是否被没加成的题稀释」"
        )
    for tag, by_caliber in (report.get("bucket_summary") or {}).items():
        if not isinstance(by_caliber, dict):
            problems.append(f"桶表 {tag} 不是按口径分组的结构（拿到 {type(by_caliber).__name__}）")
            continue
        for caliber in CALIBERS_DENSITY:
            if caliber not in by_caliber:
                problems.append(f"桶表 {tag} 缺口径 {caliber}")
        for caliber, rows in by_caliber.items():
            for cell in rows:
                for metric in DENSITY_METRICS:
                    row = {"signal_boosted": cell["n"]}
                    if cell["n"]:
                        row["context_units"] = cell["n"]
                        row["density"] = provenance.density_of(cell["n"], cell["n"])
                    result = provenance.verify_metric_row(row)
                    if result["verdict"] != provenance.PASS:
                        problems.append(
                            f"桶 {tag}/{caliber}/{cell['bucket']}/{metric} 的密度三件套不合格："
                            f"{result['detail']}"
                        )

    gain_result = provenance.verify_gain_sets(report.get("parallel_recall") or [])
    if gain_result["verdict"] != provenance.PASS:
        problems.append(f"并列召回指标：{gain_result['detail']}")
    return problems


def render_text(report: dict) -> str:
    """人读的摘要。故意把「4+ 净增益」单独点名，避免它被表格冲淡。

    两种分母口径都打出来，且**先打 `boost_applied_only`**：那是能看出密度效应
    的那个口径。只打全量口径会让人以为「0–1 桶 253 题只救 2 题，密度没什么用」。

    三个指标各一列：`any_gold` / `all_gold` / `covered`。只打一个的话，拿另一个
    口径去对账的人会发现数对不上（Lead 表里 1–2 桶救 3 是 `any_gold`，
    `covered` 只有 2）。
    """
    order = ("boost_applied_only", "all_questions")
    label = {
        "boost_applied_only": "只算真正被加成过的题（密度效应的正确口径）",
        "all_questions": "全部题（含 215 道完全没加成的，会把结论稀释）",
    }
    lines = ["信号密度分桶（boosted / units）", ""]
    for tag, by_caliber in (report.get("bucket_summary") or {}).items():
        for caliber in order:
            rows = (by_caliber or {}).get(caliber) or []
            if not rows:
                continue
            lines.append(f"[{tag}] 口径 {caliber} —— {label.get(caliber, '')}")
            head = f"  {'桶':<6}{'n':>5}"
            for metric in DENSITY_METRICS:
                head += f"{metric + ' 净':>14}"
            lines.append(head)
            for cell in rows:
                line = f"  {cell['bucket']:<6}{cell['n']:>5}"
                for metric in DENSITY_METRICS:
                    m = cell.get(metric) or {}
                    line += f"{m.get('net', 0):>14}"
                lines.append(line)
            lines.append("")
            lines.append("  （三分指标各一组净增益：any_gold=gold 块进了上下文、"
                         "all_gold=全部 gold 块都进了、covered=回答里引用发全了）")
            lines.append("")
    relations = report.get("gain_set_relations") or []
    if relations:
        lines.append("并列召回指标的包含关系（增益题集）")
        for rel in relations:
            flag = "嵌套" if rel["nested"] else "不嵌套"
            lines.append(
                f"  {rel['a']} × {rel['b']}：交集 {rel['intersection_n']}，"
                f"仅 {rel['a']} {rel['only_a'] or '—'}，仅 {rel['b']} {rel['only_b'] or '—'}（{flag}）"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="把信号密度做成报告里看得见的一列。")
    ap.add_argument("--in", dest="src", default=DEFAULT_INPUT)
    ap.add_argument("--out", dest="dst", default=DEFAULT_OUTPUT)
    ap.add_argument("--check", action="store_true", help="只校验已写报告，不重算")
    ap.add_argument("--json", action="store_true", help="打印报告全文")
    args = ap.parse_args(argv)

    src_path = Path(args.src)
    dst_path = Path(args.dst)
    if not src_path.is_absolute():
        src_path = ROOT / src_path
    if not dst_path.is_absolute():
        dst_path = ROOT / dst_path

    if args.check:
        if not dst_path.exists():
            print(f"报告不存在：{config.rel_to_root(dst_path)}", file=sys.stderr)
            return 1
        report = json.loads(dst_path.read_text("utf-8"))
        problems = self_check(report)
        if problems:
            for p in problems:
                print(f"[不合规] {p}", file=sys.stderr)
            return 1
        print(f"{config.rel_to_root(dst_path)} 合规")
        return 0

    if not src_path.exists():
        print(f"输入不存在：{config.rel_to_root(src_path)}", file=sys.stderr)
        return 2
    payload = json.loads(src_path.read_text("utf-8"))
    report = build(payload)
    problems = self_check(report)
    if problems:
        for p in problems:
            print(f"[不合规] {p}", file=sys.stderr)
        return 1
    atomic.atomic_write_json(dst_path, report)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render_text(report))
        print()
        print(f"写盘：{config.rel_to_root(dst_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
