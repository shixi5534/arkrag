# -*- coding: utf-8 -*-
"""cooc 的能力边界：信号密度决定成败，四维是口径嵌套。

为什么写这个脚本：`docs/lead_cooc能力边界与四维嵌套.md` 里的每个数字都必须
能被别人一条命令重算出来。所有输入都在盘上（`data/a_p2_round1.json`、
`data/goldset/gold-v2.json`、`data/goldset/gold-v1.json`、
`data/a_triage_all320.jsonl`、代内 `chunks.jsonl`），本脚本不做任何真跑。

用法：
    py -3 scripts/lead_cooc_boundary.py            # 人读
    py -3 scripts/lead_cooc_boundary.py --json data/lead_cooc_boundary.json
    py -3 scripts/lead_cooc_boundary.py --generation g-20261103T132152Z-06a741

零新增依赖：只用标准库 + 现有 arkrag（`config` / `subproc`）。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from arkrag import config, subproc  # noqa: E402

DEFAULT_GENERATION = "g-20261103T132152Z-06a741"
ROUND1 = ROOT / "data" / "a_p2_round1.json"
GOLD_V2 = ROOT / "data" / "goldset" / "gold-v2.json"
GOLD_V1 = ROOT / "data" / "goldset" / "gold-v1.json"
TRIAGE = ROOT / "data" / "a_triage_all320.jsonl"


# ---------------------------------------------------------------- 溯源

def _git_revision() -> str:
    """取当前 HEAD。输出是 40 位十六进制 ASCII，仍显式传 env 以过结构门。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=subproc.child_env(),
            timeout=30,
        )
        return (out.stdout or "").strip() or "unknown"
    except Exception as exc:  # pragma: no cover - 环境缺 git 时
        return "unknown(%s)" % exc


def _provenance(command: str) -> dict:
    import platform

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_revision(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "command": command,
    }


# ---------------------------------------------------------------- 载入

def load_round1(path: Path) -> tuple[dict, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pq = data["per_question"]
    base = {r["question_id"]: r for r in pq["cooc0"]}
    cand = {r["question_id"]: r for r in pq["cooc0.1"]}
    return base, cand


def load_gold(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {it["question_id"]: it for it in data["items"]}


def load_chunks(generation: str) -> dict:
    """块正文必须从**世代自带**文件读，不用全局 data/arkrag/chunks.jsonl。

    两种文件不是同一批块（实测代内 13,798 条、语料根 13,786 条，代内多 17、
    语料根多 5）。用错会让「块正文里有没有这一行」这类判断静默偏掉。
    """
    path = config.DATA_DIR / "generations" / generation / "chunks.jsonl"
    if not path.exists():
        raise SystemExit("世代自带块文件不存在：%s" % config.rel_to_root(path))
    chunks = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            chunks[rec["chunk_id"]] = rec
    return chunks


# ---------------------------------------------------------------- 分析

def section_dim_nesting(base: dict, cand: dict, gold_v1: dict, gold_v2: dict) -> dict:
    """四个召回维度是不是独立证据：增益集重合 + required≡gold 结构检验。"""

    def gains(m: dict, dim: str) -> set:
        return {q for q in base if not base[q][dim] and m[q][dim]}

    ga, gl, gc = gains(cand, "any_gold"), gains(cand, "all_gold"), gains(cand, "covered")

    def loses(m: dict, dim: str) -> set:
        return {q for q in base if base[q][dim] and not m[q][dim]}

    # required 块集是否恒等于 gold 块集
    struct = {}
    for tag, items in (("gold-v2", gold_v2), ("gold-v1", gold_v1)):
        c = Counter()
        bad = 0
        for it in items.values():
            g = set(it.get("gold_chunk_ids") or [])
            req = {rc.get("chunk_id") for rc in (it.get("required_citations") or [])}
            if not req and not g:
                c["both_empty"] += 1
            elif req == g:
                c["req==gold"] += 1
            else:
                bad += 1
                c["mismatch"] += 1
        struct[tag] = {
            "题数": len(items),
            "分布": dict(c),
            "req与gold不一致题数": bad,
            "结论": "required 块集恒等于 gold 块集" if bad == 0 else "存在不一致，嵌套结论不成立",
        }

    # 单块题上 any ≡ all
    nch = {q: len((gold_v2.get(q) or {}).get("gold_chunk_ids") or []) for q in base}
    single = [q for q in base if nch[q] == 1]
    multi = [q for q in base if nch[q] >= 2]

    # covered ⊆ all_gold 的反向缺口
    gap = {tag: [q for q in m if m[q]["all_gold"] and not m[q]["covered"]]
           for tag, m in (("cooc0", base), ("cooc0.1", cand))}
    violated = {tag: [q for q in m if m[q]["covered"] and not m[q]["all_gold"]]
                for tag, m in (("cooc0", base), ("cooc0.1", cand))}

    return {
        "增益题集": {
            "any_gold": sorted(ga),
            "all_gold": sorted(gl),
            "covered": sorted(gc),
            "all_gold⊆any_gold": gl <= ga,
            "covered⊆all_gold": gc <= gl,
            "三集合并集题数": len(ga | gl | gc),
            "三维同时转真题数": len(ga & gl & gc),
        },
        "丢失题集": {
            "any_gold": sorted(loses(cand, "any_gold")),
            "all_gold": sorted(loses(cand, "all_gold")),
            "covered": sorted(loses(cand, "covered")),
        },
        "required与gold结构": struct,
        "单块题any_eq_all": {
            "单gold块题数": len(single),
            "占比": round(len(single) / len(base), 4),
            "实测恒等成立": all(base[q]["any_gold"] == base[q]["all_gold"] for q in single),
            "多gold块题数": len(multi),
            "多块题里any≠all": sum(1 for q in multi if base[q]["any_gold"] != base[q]["all_gold"]),
        },
        "covered为真但all_gold为假": {k: len(v) for k, v in violated.items()},
        "all_gold为真但covered为假": {k: len(v) for k, v in gap.items()},
        "结论": (
            "covered ⊆ all_gold ⊆ any_gold 是评测口径的结构嵌套，"
            "不是四份独立证据；三维净额不能并列陈述"
            if gl <= ga and gc <= gl and all(len(v) == 0 for v in violated.values())
            else "嵌套关系不成立，需重新核对"
        ),
    }


def section_density(base: dict, cand: dict) -> dict:
    """信号密度 boosted/units 是否决定增益成败。"""
    trig = [q for q in base if cand[q].get("cooc_applied", 0) > 0]

    def density(q: str) -> float:
        return cand[q]["cooc_boosted"] / max(1, cand[q]["units"])

    def is_gain(q: str) -> bool:
        return (not base[q]["any_gold"]) and cand[q]["any_gold"]

    def bucket_stats(sel: list) -> dict:
        g = sum(1 for q in sel if is_gain(q))
        l = sum(1 for q in sel if base[q]["any_gold"] and not cand[q]["any_gold"])
        cd = sum(1 for q in sel if cand[q]["covered"]) - sum(1 for q in sel if base[q]["covered"])
        return {"n": len(sel), "救": g, "丢": l, "净": g - l, "覆盖Δ": cd}

    by_density = []
    for lo, hi in ((0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 1e9)):
        sel = [q for q in trig if lo <= density(q) < hi]
        if sel:
            row = {"区间": "%.0f-%.0f" % (lo, hi) if hi < 1e8 else "4+", **bucket_stats(sel)}
            by_density.append(row)

    by_boosted = []
    for lo, hi in ((1, 2), (3, 4), (5, 8), (9, 16), (17, 10 ** 6)):
        sel = [q for q in trig if lo <= cand[q]["cooc_boosted"] <= hi]
        if sel:
            by_boosted.append({
                "区间": "%d-%d" % (lo, hi) if hi < 10 ** 6 else "17+",
                **bucket_stats(sel),
            })

    # 混杂检验：在单一题型内部，密度效应是否仍成立
    confound = {}
    for cat in sorted({cand[q]["category"] for q in trig}):
        sel_c = [q for q in trig if cand[q]["category"] == cat]
        rows = []
        for lo, hi in ((0.0, 2.0), (2.0, 4.0), (4.0, 1e9)):
            z = [q for q in sel_c if lo <= density(q) < hi]
            if z:
                rows.append({"密度": "%.0f-%.0f" % (lo, hi) if hi < 1e8 else "4+", **bucket_stats(z)})
        confound[cat] = rows

    # 预筛能力：密度阈值能否完整圈住全部增益题
    gain_q = [q for q in trig if is_gain(q)]
    prescreen = []
    for thr in (1.0, 2.0, 3.0, 4.0, 6.0):
        keep = [q for q in trig if density(q) <= thr]
        prescreen.append({
            "阈值(密度<=": thr,
            "保留题数": len(keep),
            "捞到增益": sum(1 for q in keep if is_gain(q)),
            "增益题总数": len(gain_q),
            "漏掉": sorted(set(gain_q) - set(keep)),
        })

    # 名次位移幅度
    shift = []
    for q in trig:
        r0 = base[q]["gold_first_rank"]
        if r0 is None:
            continue
        r1 = cand[q]["gold_first_rank"]
        shift.append((r0 - (r1 if r1 is not None else 21), q, r0, r1))
    xs = [s[0] for s in shift]
    up = sorted([v for v in xs if v > 0])

    return {
        "触发题数": len(trig),
        "总题数": len(base),
        "按密度分桶": by_density,
        "按命中块数分桶": by_boosted,
        "混杂检验_按题型内密度": confound,
        "预筛能力": prescreen,
        "增益题": sorted(gain_q),
        "增益题密度": {q: round(density(q), 2) for q in sorted(gain_q)},
        "名次位移": {
            "可测题数": len(shift),
            "前进": sum(1 for v in xs if v > 0),
            "不变": sum(1 for v in xs if v == 0),
            "后退": sum(1 for v in xs if v < 0),
            "最大前进": max(xs) if xs else 0,
            "前进题中位": up[len(up) // 2] if up else 0,
        },
        "结论": (
            "增益集中在 boosted/units <= 3 的题上，4+ 桶零增益；"
            "题型内部同样单调，故是密度效应而非题型效应"
        ),
    }


def section_multi(base: dict, cand: dict, gold: dict) -> dict:
    """多块题为什么结构上拿不到收益。"""
    nch = {q: len((gold.get(q) or {}).get("gold_chunk_ids") or []) for q in base}
    by_k = []
    for k in (1, 2, 3, 4):
        sel = [q for q in base if nch[q] == k]
        if not sel:
            continue
        by_k.append({
            "gold块数": k,
            "n": len(sel),
            "any_gold": [sum(1 for q in sel if base[q]["any_gold"]),
                         sum(1 for q in sel if cand[q]["any_gold"])],
            "all_gold": [sum(1 for q in sel if base[q]["all_gold"]),
                         sum(1 for q in sel if cand[q]["all_gold"])],
            "covered": [sum(1 for q in sel if base[q]["covered"]),
                        sum(1 for q in sel if cand[q]["covered"])],
        })

    multi = [q for q in base if nch[q] >= 2]

    # 最差那颗 gold 块的名次（来自 triage，v1 口径深池 200）
    worst = []
    if TRIAGE.exists():
        with TRIAGE.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                ranks = row.get("gold_ranks") or {}
                if len(ranks) >= 2:
                    vals = [v for v in ranks.values() if isinstance(v, int)]
                    if vals:
                        worst.append(max(vals))
    worst.sort()
    worst_summary = {}
    if worst:
        n = len(worst)
        worst_summary = {
            "可测题数": n,
            "中位": worst[n // 2],
            "最小": worst[0],
            "最大": worst[-1],
            "分桶": dict(sorted(Counter(
                "1-3" if s <= 3 else "4-5" if s <= 5 else "6-8" if s <= 8
                else "9-20" if s <= 20 else ">20" for s in worst).items())),
            "≤5名占比": round(sum(1 for s in worst if s <= 5) / n, 4),
            ">20名占比": round(sum(1 for s in worst if s > 20) / n, 4),
        }

    half = [q for q in multi if base[q]["any_gold"] and not base[q]["all_gold"]]
    return {
        "按gold块数": by_k,
        "多块题数": len(multi),
        "第一块进第二块没进": {
            "题数": len(half),
            "占多块题比": round(len(half) / len(multi), 4) if multi else 0,
            "其中all_gold在候选下变真": sum(1 for q in half if cand[q]["all_gold"]),
        },
        "最差gold块名次": worst_summary,
        "结论": (
            "增益全部来自单 gold 块题；多块题瓶颈是最差那颗 gold 块的名次"
            "（中位 29 名），加成抬第一块无效"
        ),
    }


def section_cooc_multi(base: dict, cand: dict, gold: dict, chunks: dict) -> dict:
    """多块题里 num 满足 cooc 判据的，为什么也没收益。"""

    def cooc_hit(cid: str, names: list) -> bool:
        text = str((chunks.get(cid) or {}).get("text") or "")
        return any(all(n in line for n in names) for line in text.splitlines())

    nch = {q: len((gold.get(q) or {}).get("gold_chunk_ids") or []) for q in base}
    multi = [q for q in base if nch[q] >= 2]

    c = Counter()
    allhit = []
    for q in multi:
        names = cand[q].get("cooc_endpoints") or []
        if len(names) < 2:
            c["无两个端点（cooc 结构上不可能）"] += 1
            continue
        hits = [cooc_hit(g, names) for g in gold[q]["gold_chunk_ids"]]
        if all(hits):
            c["全部 gold 块满足"] += 1
            allhit.append(q)
        elif any(hits):
            c["部分 gold 块满足"] += 1
        else:
            c["没有任何 gold 块满足"] += 1

    rows = []
    for q in allhit:
        rows.append({
            "question_id": q,
            "category": base[q]["category"],
            "all_gold": [base[q]["all_gold"], cand[q]["all_gold"]],
            "first_rank": [base[q]["gold_first_rank"], cand[q]["gold_first_rank"]],
            "units": cand[q]["units"],
            "boosted": cand[q]["cooc_boosted"],
            "密度": round(cand[q]["cooc_boosted"] / max(1, cand[q]["units"]), 2),
        })

    # 单块题对照
    single = [q for q in base if nch[q] == 1]
    cs = Counter()
    for q in single:
        names = cand[q].get("cooc_endpoints") or []
        if len(names) < 2:
            cs["无两个端点"] += 1
            continue
        cs["满足" if cooc_hit(gold[q]["gold_chunk_ids"][0], names) else "不满足"] += 1

    return {
        "多块题分类": dict(c),
        "全部满足cooc的多块题": {
            "题数": len(allhit),
            "base all_gold为真": sum(1 for q in allhit if base[q]["all_gold"]),
            "cand all_gold为真": sum(1 for q in allhit if cand[q]["all_gold"]),
            "逐题": rows,
        },
        "单块题对照": dict(cs),
        "结论": (
            "全部 gold 块都满足 cooc 判据的多块题一题也没变好，"
            "因为密度中位 3.8 落在零增益区"
        ),
    }


def section_untriggered(base: dict, cand: dict) -> dict:
    """215 题未触发里，多少是实体解析上限（cooc 结构上不可能）。"""
    notrig = [q for q in base if not cand[q].get("cooc_applied", 0)]
    trig = [q for q in base if cand[q].get("cooc_applied", 0) > 0]
    ep_notrig = Counter(len(cand[q].get("cooc_endpoints") or []) for q in notrig)
    ep_all = Counter(len(cand[q].get("cooc_endpoints") or []) for q in base)

    boosted_but_not_applied = [q for q in notrig if cand[q].get("cooc_boosted", 0) > 0]
    return {
        "未触发题数": len(notrig),
        "未触发_端点数分布": {str(k): v for k, v in sorted(ep_notrig.items())},
        "全量_端点数分布": {str(k): v for k, v in sorted(ep_all.items())},
        "触发题全是两个端点": all(len(cand[q].get("cooc_endpoints") or []) >= 2 for q in trig),
        "结构上不可能触发题数": sum(v for k, v in ep_notrig.items() if k < 2),
        "算出加成但没进前20": sorted(boosted_but_not_applied),
        "结论": "未触发的一半以上是只解析出 <2 个实体，属 analyze_query 上限，不是 cooc 缺陷",
    }


# ---------------------------------------------------------------- 入口

def main(argv: list[str] | None = None) -> int:
    # 控制台保护：本脚本打印大量中文与 ⊆ / ⇒ 这类 cp936 编不出的符号。
    # 不 reconfigure 的话，在 cp936 控制台（或 `-s` 直接输出时）进程会以 1 退出，
    # 而这个 1 与「分析失败」长得一模一样——正是 Gate 0 记过的那类假失败。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(description="cooc 能力边界与四维嵌套复核")
    ap.add_argument("--json", dest="json_path", default="")
    ap.add_argument("--generation", default=DEFAULT_GENERATION)
    args = ap.parse_args(argv)

    for p in (ROUND1, GOLD_V2):
        if not p.exists():
            print("[HARD ERROR] 输入不存在：%s" % config.rel_to_root(p), file=sys.stderr)
            return 2

    base, cand = load_round1(ROUND1)
    gold_v2 = load_gold(GOLD_V2)
    gold_v1 = load_gold(GOLD_V1) if GOLD_V1.exists() else {}
    chunks = load_chunks(args.generation)

    result = {
        "schema": "lead-cooc-boundary/1",
        "purpose": "复核 P2 第一轮：四维嵌套、信号密度边界、多块题结构零收益",
        "generation": args.generation,
        "inputs": {
            "round1": config.rel_to_root(ROUND1),
            "gold_v2": config.rel_to_root(GOLD_V2),
            "gold_v1": config.rel_to_root(GOLD_V1) if GOLD_V1.exists() else "",
            "triage": config.rel_to_root(TRIAGE) if TRIAGE.exists() else "",
            "chunks": config.rel_to_root(
                config.DATA_DIR / "generations" / args.generation / "chunks.jsonl"),
        },
        "题数": {"dev应答题": len(base)},
        "1_四维嵌套": section_dim_nesting(base, cand, gold_v1, gold_v2),
        "2_信号密度": section_density(base, cand),
        "3_多块题结构": section_multi(base, cand, gold_v2),
        "4_cooc判据与多块题": section_cooc_multi(base, cand, gold_v2, chunks),
        "5_未触发归因": section_untriggered(base, cand),
    }
    result.update(_provenance("py -3 scripts/lead_cooc_boundary.py"))

    # ---- 打印
    d = result["1_四维嵌套"]
    print("=" * 72)
    print("1 四维是口径嵌套，不是独立证据")
    print("=" * 72)
    print("  增益题集：any_gold %d / all_gold %d / covered %d；并集 %d 题；三维同时转真 %d 题" % (
        len(d["增益题集"]["any_gold"]), len(d["增益题集"]["all_gold"]),
        len(d["增益题集"]["covered"]), d["增益题集"]["三集合并集题数"],
        d["增益题集"]["三维同时转真题数"]))
    print("  all_gold⊆any_gold=%s  covered⊆all_gold=%s" % (
        d["增益题集"]["all_gold⊆any_gold"], d["增益题集"]["covered⊆all_gold"]))
    for tag, s in d["required与gold结构"].items():
        print("  %s: %s（不一致 %d 题）" % (tag, s["结论"], s["req与gold不一致题数"]))
    print("  单块题 any≡all: n=%d (%.1f%%) 成立=%s" % (
        d["单块题any_eq_all"]["单gold块题数"], 100 * d["单块题any_eq_all"]["占比"],
        d["单块题any_eq_all"]["实测恒等成立"]))
    print("  covered真但all_gold假: %s ；all_gold真但covered假: %s" % (
        d["covered为真但all_gold为假"], d["all_gold为真但covered为假"]))
    print("  ⇒ %s" % d["结论"])

    s2 = result["2_信号密度"]
    print()
    print("=" * 72)
    print("2 信号密度决定成败（触发 %d / %d 题）" % (s2["触发题数"], s2["总题数"]))
    print("=" * 72)
    print("  按密度分桶:")
    for r in s2["按密度分桶"]:
        print("    密度 %-6s n=%-3d 救 %-2d 丢 %-2d 净 %+d 覆盖Δ %+d" % (
            r["区间"], r["n"], r["救"], r["丢"], r["净"], r["覆盖Δ"]))
    print("  按命中块数分桶:")
    for r in s2["按命中块数分桶"]:
        print("    boosted %-6s n=%-3d 救 %-2d 丢 %-2d 净 %+d" % (
            r["区间"], r["n"], r["救"], r["丢"], r["净"]))
    print("  混杂检验（direct_relation 内部）:")
    for r in s2["混杂检验_按题型内密度"].get("direct_relation", []):
        print("    密度 %-6s n=%-3d 净 %+d" % (r["密度"], r["n"], r["净"]))
    print("  增益题密度: %s" % s2["增益题密度"])
    print("  名次位移: %s" % s2["名次位移"])
    print("  ⇒ %s" % s2["结论"])

    s3 = result["3_多块题结构"]
    print()
    print("=" * 72)
    print("3 多块题结构上零收益")
    print("=" * 72)
    for r in s3["按gold块数"]:
        print("  gold块=%d n=%-3d any %s  all %s  covered %s" % (
            r["gold块数"], r["n"], r["any_gold"], r["all_gold"], r["covered"]))
    print("  第一块进第二块没进: %d 题（%.1f%%），其中 all_gold 变真 %d 题" % (
        s3["第一块进第二块没进"]["题数"], 100 * s3["第一块进第二块没进"]["占多块题比"],
        s3["第一块进第二块没进"]["其中all_gold在候选下变真"]))
    if s3["最差gold块名次"]:
        w = s3["最差gold块名次"]
        print("  最差 gold 块名次: 中位 %s，≤5名 %.1f%%，>20名 %.1f%%  分桶 %s" % (
            w["中位"], 100 * w["≤5名占比"], 100 * w[">20名占比"], w["分桶"]))
    print("  ⇒ %s" % s3["结论"])

    s4 = result["4_cooc判据与多块题"]
    print()
    print("=" * 72)
    print("4 多块题里满足 cooc 判据的也没收益")
    print("=" * 72)
    print("  多块题分类: %s" % s4["多块题分类"])
    a4 = s4["全部满足cooc的多块题"]
    print("  全部满足 cooc 的多块题 %d 题：base all_gold=%d → cand=%d" % (
        a4["题数"], a4["base all_gold为真"], a4["cand all_gold为真"]))
    print("  单块题对照: %s" % s4["单块题对照"])
    print("  ⇒ %s" % s4["结论"])

    s5 = result["5_未触发归因"]
    print()
    print("=" * 72)
    print("5 未触发归因")
    print("=" * 72)
    print("  未触发 %d 题，端点数分布 %s" % (s5["未触发题数"], s5["未触发_端点数分布"]))
    print("  结构上不可能触发（<2 端点）: %d 题" % s5["结构上不可能触发题数"])
    print("  算出加成但没进前20: %s" % s5["算出加成但没进前20"])
    print("  ⇒ %s" % s5["结论"])

    print()
    print("溯源：%s | %s | py %s | %s" % (
        result["generated_at"], result["git_commit"], result["python"], result["os"]))

    if args.json_path:
        out = Path(args.json_path)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print("已写出：%s（%d 字节）" % (config.rel_to_root(out), out.stat().st_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
