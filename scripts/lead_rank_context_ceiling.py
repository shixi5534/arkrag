"""排名到上下文的转化上限：为什么 R@20 是不该优化的目标。

背景
----
`data/a_ranking_replay.json` 报告 pair0.3+hop0.2 让 R@20 从 0.7059 涨到 0.7262
（+2.02pt）。但上下文预算 6000 tokens、候选单元中位 1262 tokens，实际只装得下
约 5 个单元。于是有一个必须先问、而此前没人测过的问题：

    排名进到第几名，才算真的「进了上下文」？

如果只有前几名才有用，那么「R@20 涨了」和「检索质量真的变好了」是两件事，
把 R@20 当 P2 的主判据会优化一个对下游无影响的数字。

本脚本用生产实测的逐题排名与逐题 gold_in_context 建立这个映射，再用它去
估计各候选配置的真实收益。

口径
----
- 生产实测：`data/a_triage_all320.jsonl`（scripts/a_failure_triage.py 产物），
  每题有 `gold_first_rank`（gold 块在池内的最好名次）与 `gold_in_context`
  （真正进入上下文的 gold 块 id 列表）。
- 候选配置：`data/a_ranking_replay.json`（scripts/a_ranking_replay.py 产物），
  逐题 `gold_new_ranks` 是「该配置下每个 gold 块的新名次」。
- 「进上下文」是二值事实，不是概率；转化率表是**经验频率**。

重要限制（写在这里防止被当成定理）
----------------------------------
1. 转化率随排名变化，本质是「预算能装几块」与「名次和单元位置不完全对齐」
   共同造成的**经验关系**，不是算法保证。
2. 用基线转化率去预测新配置，隐含假设「新排名下的转化率与基线同分布」。
   本脚本因此同时给**不依赖该假设**的指标（R@3 / R@5：这两个区间的实测转化
   率是 1.0 与 0.70，接近确定性），把预测值只当参考。
3. 真验证必须真跑管线（改排序分 → 重建上下文 → 重新作答），replay 不算。
   见 docs/lead_引用覆盖漏斗实测.md 关于「重放 != 真跑」的实测。

用法
----
    py -3 scripts/lead_rank_context_ceiling.py
    py -3 scripts/lead_rank_context_ceiling.py --json data/lead_rank_context_ceiling.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

TRIAGE = DATA / "a_triage_all320.jsonl"
REPLAY = DATA / "a_ranking_replay.json"

# 生产实测的排名分桶。上界取 5 是因为 6000 预算实测只装约 5 个单元。
RANK_BUCKETS = [
    (1, 3, "1-3"),
    (4, 5, "4-5"),
    (6, 8, "6-8"),
    (9, 10, "9-10"),
    (11, 15, "11-15"),
    (16, 20, "16-20"),
    (21, 50, "21-50"),
    (51, 200, "51-200"),
    (201, 10**9, ">200"),
]


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        return (out.stdout or "").strip() or "unknown"
    except Exception:
        return "unknown"


def _origin() -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_revision(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "command": "py -3 scripts/lead_rank_context_ceiling.py",
    }


def load_triage() -> dict:
    rows = {}
    with TRIAGE.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows[rec["question_id"]] = rec
    return rows


def bucket_of(rank):
    if rank is None:
        return "池内无 gold"
    for lo, hi, label in RANK_BUCKETS:
        if lo <= rank <= hi:
            return label
    return ">200"


def section_conversion(triage: dict) -> dict:
    """生产实测：gold 名次 -> 是否真的进了上下文。"""
    answered = [r for r in triage.values() if not r.get("should_abstain")]
    table = defaultdict(lambda: {"n": 0, "in_ctx": 0})
    for rec in answered:
        rank = rec.get("gold_first_rank")
        key = bucket_of(rank)
        table[key]["n"] += 1
        if rec.get("gold_in_context"):
            table[key]["in_ctx"] += 1

    ordered = []
    total_n = 0
    total_ctx = 0
    for _lo, _hi, label in RANK_BUCKETS + [(0, 0, "池内无 gold")]:
        if label not in table:
            continue
        cell = table[label]
        n, ctx = cell["n"], cell["in_ctx"]
        total_n += n
        total_ctx += ctx
        ordered.append(
            {
                "bucket": label,
                "n": n,
                "in_context": ctx,
                "rate": round(ctx / n, 4) if n else 0.0,
            }
        )
    return {
        "应答题数": len(answered),
        "进上下文题数": total_ctx,
        "总体转化率": round(total_ctx / total_n, 4) if total_n else 0.0,
        "分桶": ordered,
    }


def section_ceiling_by_rank(triage: dict) -> dict:
    """累积：rank<=k 的题数，以及其中真的进了上下文的题数。"""
    answered = [r for r in triage.values() if not r.get("should_abstain")]
    out = []
    for k in (3, 5, 10, 20, 50, 100, 200):
        hit = [r for r in answered if (r.get("gold_first_rank") or 10**9) <= k]
        in_ctx = sum(1 for r in hit if r.get("gold_in_context"))
        out.append(
            {
                "k": k,
                "gold_rank_le_k": len(hit),
                "占应答题比": round(len(hit) / len(answered), 4) if answered else 0.0,
                "其中进上下文": in_ctx,
                "k 内转化率": round(in_ctx / len(hit), 4) if hit else 0.0,
            }
        )
    return {"应答题数": len(answered), "累积表": out}


def _ranks(rec: dict) -> list:
    return [v for v in (rec.get("gold_new_ranks") or {}).values() if isinstance(v, int)]


def _first(rec: dict):
    rs = _ranks(rec)
    return min(rs) if rs else None


def section_configs(replay: dict, triage: dict) -> dict:
    """各配置：R@3 / R@5 / R@20 与「预测进上下文题数」。

    预测用基线分桶转化率；同时给出不依赖该假设的 R@3 / R@5。
    """
    conv = {c["bucket"]: c["rate"] for c in section_conversion(triage)["分桶"]}
    per_q = replay["per_question"]
    rows = []
    for cfg in per_q:
        items = list(per_q[cfg])
        n = len(items)
        if not n:
            continue
        r3 = sum(1.0 if any(r <= 3 for r in _ranks(x)) else 0.0 for x in items) / n
        r5 = sum(1.0 if any(r <= 5 for r in _ranks(x)) else 0.0 for x in items) / n
        r20 = sum(x.get("r20") or 0.0 for x in items) / n
        pred = sum(conv.get(bucket_of(_first(x)), 0.0) for x in items) / n
        rows.append(
            {
                "config": cfg,
                "n": n,
                "R@3": round(r3, 4),
                "R@5": round(r5, 4),
                "R@20": round(r20, 4),
                "预测进上下文率": round(pred, 4),
            }
        )
    base = next((r for r in rows if r["config"] == "现状"), None)
    if base:
        for r in rows:
            r["dR@3_pt"] = round((r["R@3"] - base["R@3"]) * 100, 2)
            r["dR@5_pt"] = round((r["R@5"] - base["R@5"]) * 100, 2)
            r["dR@20_pt"] = round((r["R@20"] - base["R@20"]) * 100, 2)
            r["d预测进上下文_pt"] = round(
                (r["预测进上下文率"] - base["预测进上下文率"]) * 100, 2
            )
    rows.sort(key=lambda r: -r["预测进上下文率"])
    return {"配置表": rows}


def section_migration(replay: dict, base_cfg: str, target_cfg: str) -> dict:
    """把「名次变化」分成三类：真救进上下文 / 无效挪动 / 回退。

    - 真救：从「基本进不了上下文」的区间(>5)挪进「能进」的区间(<=5)
    - 无效：在 6-20 之间挪动，或从 >20 挪进 6-20（对上下文无影响）
    - 回退：从 <=5 掉到 >5
    """
    per_q = replay["per_question"]
    if base_cfg not in per_q or target_cfg not in per_q:
        return {"error": "配置不存在"}
    base = {x["question_id"]: x for x in per_q[base_cfg]}
    targ = {x["question_id"]: x for x in per_q[target_cfg]}

    def zone(rank):
        if rank is None:
            return "无"
        if rank <= 5:
            return "能进上下文(<=5)"
        if rank <= 20:
            return "评测算命中但进不了(6-20)"
        return "池内后段(>20)"

    trans = Counter()
    real_gain = waste = regress = 0
    for qid, b in base.items():
        if qid not in targ:
            continue
        fb, fn = _first(b), _first(targ[qid])
        if fb is None or fn is None:
            continue
        zb, zn = zone(fb), zone(fn)
        if zb == zn:
            trans[(zb, zn)] += 1
            continue
        trans[(zb, zn)] += 1
        if zb != "能进上下文(<=5)" and zn == "能进上下文(<=5)":
            real_gain += 1
        elif zb == "能进上下文(<=5)" and zn != "能进上下文(<=5)":
            regress += 1
        else:
            waste += 1

    ordered = [
        {"from": k[0], "to": k[1], "n": v}
        for k, v in sorted(trans.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    return {
        "基线配置": base_cfg,
        "目标配置": target_cfg,
        "真救进上下文_题数": real_gain,
        "无效挪动_题数": waste,
        "回退_题数": regress,
        "迁移矩阵": ordered,
    }


def section_units(triage: dict) -> dict:
    """上下文单元数分布：解释为什么只有前几名能进。"""
    answered = [r for r in triage.values() if not r.get("should_abstain")]
    counts = [len(r.get("context_units") or []) for r in answered]
    counts.sort()
    if not counts:
        return {}
    mid = counts[len(counts) // 2]
    return {
        "单元数中位": mid,
        "单元数最小": counts[0],
        "单元数最大": counts[-1],
        "单元数均值": round(sum(counts) / len(counts), 2),
        "说明": (
            "上下文预算 6000 tokens、候选单元中位约 1262 tokens，"
            "实际只能装约 5 个单元；这就是名次 6 以后基本进不了上下文的原因。"
        ),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default="", help="额外把结构化结果写到这个路径")
    ap.add_argument("--base", default="现状", help="迁移矩阵的基线配置名")
    ap.add_argument("--target", default="pair0.3+hop0.2", help="迁移矩阵的目标配置名")
    args = ap.parse_args(argv)

    if not TRIAGE.exists():
        print(f"[!] 缺少 {TRIAGE}", file=sys.stderr)
        return 2
    if not REPLAY.exists():
        print(f"[!] 缺少 {REPLAY}", file=sys.stderr)
        return 2

    triage = load_triage()
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))

    conv = section_conversion(triage)
    ceiling = section_ceiling_by_rank(triage)
    cfgs = section_configs(replay, triage)
    mig = section_migration(replay, args.base, args.target)
    units = section_units(triage)

    print("=" * 78)
    print("1 生产实测：gold 名次 -> 真的进了上下文吗")
    print("=" * 78)
    print(f"应答题 {conv['应答题数']}，其中 gold 进了上下文 {conv['进上下文题数']}"
          f"（{conv['总体转化率']:.4f}）")
    print(f"  {'名次段':<12}{'题数':>6}{'进上下文':>10}{'转化率':>10}")
    for row in conv["分桶"]:
        print(f"  {row['bucket']:<12}{row['n']:>6}{row['in_context']:>10}"
              f"{row['rate']:>10.4f}")

    print()
    print("=" * 78)
    print("2 累积视角：名次在 k 以内，有多少真的进了上下文")
    print("=" * 78)
    print(f"  {'k':>5}{'rank<=k 题数':>14}{'占比':>9}{'进上下文':>10}{'k内转化率':>11}")
    for row in ceiling["累积表"]:
        print(f"  {row['k']:>5}{row['gold_rank_le_k']:>14}{row['占应答题比']:>9.4f}"
              f"{row['其中进上下文']:>10}{row['k 内转化率']:>11.4f}")

    print()
    print("=" * 78)
    print("3 上下文单元数（解释第 1 节的分界）")
    print("=" * 78)
    print(f"  单元数 中位 {units.get('单元数中位')} / 均值 {units.get('单元数均值')}"
          f" / 范围 {units.get('单元数最小')}-{units.get('单元数最大')}")
    print(f"  {units.get('说明')}")

    print()
    print("=" * 78)
    print("4 各配置：R@20 与「真的能进上下文」的差距")
    print("=" * 78)
    print(f"  {'配置':<18}{'R@3':>7}{'R@5':>7}{'R@20':>7}{'预测进ctx':>10}"
          f"{'dR@5':>8}{'dR@20':>8}")
    for row in cfgs["配置表"]:
        print(f"  {row['config']:<18}{row['R@3']:>7.4f}{row['R@5']:>7.4f}"
              f"{row['R@20']:>7.4f}{row['预测进上下文率']:>10.4f}"
              f"{row.get('dR@5_pt', 0):>8.2f}{row.get('dR@20_pt', 0):>8.2f}")

    print()
    print("=" * 78)
    print(f"5 名次迁移分类：{args.base} -> {args.target}")
    print("=" * 78)
    if "error" in mig:
        print(f"  {mig['error']}")
    else:
        print(f"  真救进上下文（从 >5 挪进 <=5）: {mig['真救进上下文_题数']} 题")
        print(f"  无效挪动（在 6-20 内挪、或 >20 挪进 6-20）: {mig['无效挪动_题数']} 题")
        print(f"  回退（从 <=5 掉出）: {mig['回退_题数']} 题")
        print()
        for row in mig["迁移矩阵"]:
            print(f"    {row['from']:<28} -> {row['to']:<28} {row['n']:>4}")

    print()
    print("=" * 78)
    print("6 结论")
    print("=" * 78)
    z = {r["bucket"]: r for r in conv["分桶"]}
    first3 = z.get("1-3", {})
    four5 = z.get("4-5", {})
    six8 = z.get("6-8", {})
    print(f"  名次 1-3 进上下文率 {first3.get('rate')}（{first3.get('in_context')}/{first3.get('n')}）")
    print(f"  名次 4-5 进上下文率 {four5.get('rate')}（{four5.get('in_context')}/{four5.get('n')}）")
    print(f"  名次 6-8 进上下文率 {six8.get('rate')}（{six8.get('in_context')}/{six8.get('n')}）")
    print()
    print("  => 只有前 5 名左右能进上下文，6 名以后基本进不去。")
    print("     所以 R@20 涨而 R@5 不涨，对下游质量（引用覆盖、答案完整）**没有用**。")
    if "error" not in mig:
        print(f"     {args.target} 相对基线：真救 {mig['真救进上下文_题数']} 题、"
              f"无效挪动 {mig['无效挪动_题数']} 题、回退 {mig['回退_题数']} 题。")
    print()
    print("  => P2 的主判据应该是 R@5 / R@3 与「gold 进上下文题数」，")
    print("     R@20 只作为「有没有召回缺口」的辅助判据。")

    if args.json:
        payload = {
            "schema": "arkrag.lead_rank_context_ceiling/1",
            "purpose": "测量「排名进到第几才真的进了上下文」，并据此判断 R@20 是否适合作为 P2 主判据",
            **_origin(),
            "triage_source": str(TRIAGE.relative_to(ROOT)),
            "replay_source": str(REPLAY.relative_to(ROOT)),
            "1_名次到上下文转化": conv,
            "2_累积可达": ceiling,
            "3_单元数": units,
            "4_配置对比": cfgs,
            "5_迁移分类": mig,
        }
        out = Path(args.json)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n[ok] 结构化结果写入 {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
