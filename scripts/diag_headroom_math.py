# -*- coding: utf-8 -*-
"""纯数学提分空间复算脚本（只读 data/ 既有 JSON，不加载产物、不跑模型）。

对应文档：docs/分析-纯数学提分空间-20261111.md

用法：
    python scripts/diag_headroom_math.py > _headroom.txt

设计约束（遵守交接文档 §2 的纪律）：
  * 不调用 load_generation()，只读已落盘的小 JSON（最大 575 KB）。
  * 不写任何既有产物，只打印到 stdout。
  * 不需要熔断器：峰值内存 < 200 MB。若将来有人加了大文件读取，必须改走
    scripts/run_guarded.py。
"""
from __future__ import annotations

import collections
import glob
import io
import json
import math
import os
import sys

OUT: list[str] = []


def p(*a) -> None:
    OUT.append(" ".join(str(x) for x in a))


def load(path: str):
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


STRUCT = {"time_conflict", "negative_abstain", "adversarial"}
SPLITS3 = ["dev", "holdout", "clean"]
CATS = ["single_chunk", "alias", "direct_relation", "multihop", "cross_chunk"]

W3 = "data/w3_evidence"
W4 = "data/w4_evidence"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def mcnemar_exact(b: int, c: int) -> float:
    """配对精确二项，双侧。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    s = sum(math.comb(n, i) for i in range(0, k + 1))
    return min(1.0, 2.0 * s / (2 ** n))


def passmap(path: str, arm: str = "prod") -> dict[str, bool]:
    d = load(path)
    out = {}
    for r in d["rows"]:
        if r["category"] in STRUCT:
            continue
        if arm == "v2":
            v = (r.get("judge_v2") or {}).get("pass")
        else:
            v = (r.get("judge") or {}).get("pass")
        out[r["question_id"]] = bool(v)
    return out


def report_cap(sp: str, cap: str) -> dict:
    return load("%s/%s_%s.json" % (W3, cap, sp))


# ---------------------------------------------------------------- 0. 分母
def section_denominator() -> None:
    p("=" * 78)
    p("0. 分母：三份 split 不是三份独立样本")
    p("=" * 78)
    ids = {}
    for sp in SPLITS3:
        d = report_cap(sp, "cap4")
        sc = [r["question_id"] for r in d["rows"] if r["category"] not in STRUCT]
        ids[sp] = set(sc)
        t = d["totals"]["prod"]
        p("  %-8s 题数=%3d 可判=%3d 通过=%3d 率=%.4f"
          % (sp, len(d["rows"]), t["scorable_n"], t["scorable_pass"], t["scorable_rate"]))
    p("  clean ⊆ holdout : %s ; clean∩dev = %d ; holdout\\clean = %d"
      % (ids["clean"] <= ids["holdout"], len(ids["clean"] & ids["dev"]),
         len(ids["holdout"] - ids["clean"])))
    union = ids["dev"] | ids["holdout"]
    naive = sum(len(ids[s]) for s in SPLITS3)
    p("  唯一题数 = %d ; 三份朴素相加 = %d ; 膨胀 x%.3f"
      % (len(union), naive, naive / len(union)))
    base = sum(1 for r in report_cap("dev", "cap4")["rows"]
               if r["category"] not in STRUCT and (r["judge"] or {}).get("pass"))
    base += sum(1 for r in report_cap("holdout", "cap4")["rows"]
                if r["category"] not in STRUCT and (r["judge"] or {}).get("pass"))
    p("  ★ 唯一可判 N = %d ; 通过 = %d ; 基准率 = %.4f" % (len(union), base, base / len(union)))
    p()
    # 结构类
    tot588 = sum(len(report_cap(s, "cap4")["rows"]) for s in ["dev", "holdout"])
    st = 0
    for sp in ["dev", "holdout"]:
        st += sum(1 for r in report_cap(sp, "cap4")["rows"] if r["category"] in STRUCT)
    p("  结构类（time_conflict/negative_abstain/adversarial）= %d/%d = %.2f%%"
      % (st, tot588, 100.0 * st / tot588))
    p("  报告口径 = %d/%d = %.2f%% ; 端到端口径 = %d/%d = %.2f%%"
      % (base, len(union), 100.0 * base / len(union), base, tot588, 100.0 * base / tot588))
    p("  ⇒ 差额 %.2f pt 全在『被判据排除计分』的那一档里"
      % (100.0 * base / len(union) - 100.0 * base / tot588))


# ---------------------------------------------------------------- 1. 失败分解
def section_partition() -> None:
    diag = load("data/diag_rank_24000.json")
    D = {r["question_id"]: r for r in diag["per_question"]}
    rows = {}
    for sp in ["dev", "holdout"]:
        for r in report_cap(sp, "cap4")["rows"]:
            if r["category"] not in STRUCT:
                rows[r["question_id"]] = r
    N = len(rows)
    base = sum(1 for r in rows.values() if (r["judge"] or {}).get("pass"))

    p()
    p("=" * 78)
    p("1. 失败的完全可加分解（N=%d，基准 %d = %.4f）" % (N, base, base / N))
    p("=" * 78)
    B = collections.Counter()
    bycat = collections.defaultdict(collections.Counter)
    for q, r in rows.items():
        if (r["judge"] or {}).get("pass"):
            continue
        d = D[q]
        if not d["in_hits"] and not d["in_context"]:
            b = "A_never_retrieved"
        elif not d["in_hits"] and d["in_context"]:
            b = "B_ctx_without_hit"
        elif d["in_hits"] and not d["in_context"]:
            b = "C_dropped_by_budget"
        else:
            b = "D_in_ctx_but_wrong"
        B[b] += 1
        bycat[b][r["category"]] += 1
    stage = {
        "A_never_retrieved": "RETRIEVAL",
        "B_ctx_without_hit": "RETRIEVAL",
        "C_dropped_by_budget": "BUDGET",
        "D_in_ctx_but_wrong": "ASSEMBLY",
    }
    order = ["A_never_retrieved", "B_ctx_without_hit", "C_dropped_by_budget", "D_in_ctx_but_wrong"]
    p("  %-24s %4s %8s  %s" % ("bucket", "n", "pt-upper", "by category"))
    for b in order:
        p("  %-24s %4d %+8.2f  %s" % (b, B[b], 100.0 * B[b] / N, dict(bycat[b])))
    p("  %-24s %4d %+8.2f" % ("total", sum(B.values()), 100.0 * sum(B.values()) / N))
    S = collections.Counter()
    for b in order:
        S[stage[b]] += B[b]
    p()
    for k in ["RETRIEVAL", "BUDGET", "ASSEMBLY"]:
        p("  %-12s %3d 题 = 上界 %+.2f pt" % (k, S[k], 100.0 * S[k] / N))
    p("  ★ 合计不可超过 %+.2f pt" % (100.0 * sum(B.values()) / N))
    return N, base, B


# ---------------------------------------------------------------- 2. 漏斗
def section_funnel() -> None:
    diag = load("data/diag_rank_24000.json")
    D = {r["question_id"]: r for r in diag["per_question"]}
    rows = {}
    for sp in ["dev", "holdout"]:
        for r in report_cap(sp, "cap4")["rows"]:
            if r["category"] not in STRUCT:
                rows[r["question_id"]] = r
    N = len(rows)
    p()
    p("=" * 78)
    p("2. 两级漏斗：P(pass | gold 已在上下文) 按类别")
    p("=" * 78)
    p("  %-16s %5s %7s %7s %9s  %s" % ("cat", "n", "in_ctx", "pass", "conv", "95% Wilson"))
    agg_ic = agg_pc = 0
    for c in CATS:
        ic = pc = 0
        n = 0
        for q, r in rows.items():
            if r["category"] != c:
                continue
            n += 1
            if D[q]["in_context"]:
                ic += 1
                if (r["judge"] or {}).get("pass"):
                    pc += 1
        lo, hi = wilson(pc, ic)
        agg_ic += ic
        agg_pc += pc
        p("  %-16s %5d %7d %7d %9.3f  [%.2f, %.2f]"
          % (c, n, ic, pc, (pc / ic) if ic else 0, lo, hi))
    lo, hi = wilson(agg_pc, agg_ic)
    p("  %-16s %5d %7d %7d %9.3f  [%.2f, %.2f]"
      % ("ALL", N, agg_ic, agg_pc, agg_pc / agg_ic, lo, hi))

    reach = sum(1 for q in rows if D[q]["in_hits"])
    inctx = sum(1 for q in rows if D[q]["in_context"])
    base = sum(1 for r in rows.values() if (r["judge"] or {}).get("pass"))
    p()
    p("  reach(gold 进 hits) = %d/%d = %.4f" % (reach, N, reach / N))
    p("  hit→ctx             = %d/%d = %.4f" % (inctx, reach, inctx / reach))
    p("  ctx→pass            = %d/%d = %.4f  （分母须用『gold 在 ctx 的通过数』）"
      % (agg_pc, agg_ic, agg_pc / agg_ic))
    p("  ★ 前两类（sc/dr）已封顶 0.985-0.992；空间全在 multihop 0.569 与 cross_chunk 0.737")


# ---------------------------------------------------------------- 3. 判据自由度
def section_judge() -> None:
    jb = load("%s/judge_both_dev.json" % W4)
    p()
    p("=" * 78)
    p("3. 判据自由度：v1 与 v2 的嵌套关系")
    p("=" * 78)
    M = collections.Counter()
    for r in jb["rows"]:
        if r["category"] in STRUCT:
            continue
        a = bool((r.get("judge") or {}).get("pass"))
        b = bool((r.get("judge_v2") or {}).get("pass"))
        M[(a, b)] += 1
    tt, tf, ft, ff = M[(True, True)], M[(True, False)], M[(False, True)], M[(False, False)]
    N = tt + tf + ft + ff
    p("  v1=1,v2=1 → %d ; v1=1,v2=0 → %d ; v1=0,v2=1 → %d ; v1=0,v2=0 → %d"
      % (tt, tf, ft, ff))
    po = (tt + ff) / N
    pe = ((tt + tf) / N) * ((tt + ft) / N) + ((ft + ff) / N) * ((tf + ff) / N)
    p("  N=%d ; v1=%.4f ; v2=%.4f ; Δ=%.2f pt ; 不一致=%d/%d=%.2f%% ; κ=%.4f"
      % (N, (tt + tf) / N, (tt + ft) / N, 100.0 * ((tt + ft) - (tt + tf)) / N,
         tf + ft, N, 100.0 * (tf + ft) / N, (po - pe) / (1 - pe)))
    p("  v1=0,v2=1 的次数 = %d ⇒ v2 是 v1 的严格子集（嵌套，非独立口径）" % ft)
    p("  ★ 判据选择一个旋钮 = %.2f pt，超过 W1+W2+W3+W4 净效应之和（1.25 pt）的 %.1f 倍"
      % (100.0 * tf / N, (100.0 * tf / N) / 1.25))
    p()
    p("  v2 逐类：")
    for c in CATS:
        n = v1 = v2 = card_fail = card_only = 0
        for r in jb["rows"]:
            if r["category"] != c:
                continue
            n += 1
            if (r.get("judge") or {}).get("pass"):
                v1 += 1
            j2 = r.get("judge_v2") or {}
            if j2.get("pass"):
                v2 += 1
            ax = j2.get("axes") or {}
            if not ax.get("cardinality_ok"):
                card_fail += 1
                if ax.get("value_ok") and ax.get("support_ok"):
                    card_only += 1
        p("    %-16s n=%3d v1=%3d v2=%3d | cardinal✗=%3d 其中只被基数否=%2d"
          % (c, n, v1, v2, card_fail, card_only))
    p("  ★ multihop 在 v2 下 %d/57 挂 cardinality 轴 ⇒ 该类整类 0 分" % 57)
    p("    若改成单值抽取，v2 的 dev 上限 = 204+57 = 261/285 = %.4f" % (261 / 285))


# ---------------------------------------------------------------- 4. 分辨率
def section_resolution() -> None:
    p()
    p("=" * 78)
    p("4. 分辨率：这份 goldset 能证明多大的效应")
    p("=" * 78)
    for sp, n, k, nm in [("dev", 285, 231, "dev"), ("holdout", 195, 154, "holdout"),
                         ("clean", 120, 98, "clean"), ("unique", 480, 385, "唯一 480")]:
        ph = k / n
        se = math.sqrt(ph * (1 - ph) / n)
        p("  %-10s n=%3d p=%.4f SE=%.2f pt 95%%半宽=±%.2f pt = ±%.1f 题"
          % (nm, n, ph, 100 * se, 196 * se, (196 * se) * n / 100.0))
    p()
    p("  精确 McNemar（对 cap=4 生产档）：")
    pairs = [
        ("dev", "cap4", "%s/cap4_dev.json" % W3, [
            ("w1 cap14", "%s/w1cap14_dev.json" % W3, "prod"),
            ("w2 margin", "%s/w2_dev.json" % W4, "prod"),
            ("judge v2", "%s/judge_both_dev.json" % W4, "v2")]),
        ("holdout", "cap4", "%s/cap4_holdout.json" % W3, [
            ("cap3", "%s/cap3_holdout.json" % W3, "prod"),
            ("cap5", "%s/cap5_holdout.json" % W3, "prod"),
            ("cap6", "%s/cap6_holdout.json" % W3, "prod"),
            ("w1 cap14", "%s/w1cap14_holdout.json" % W3, "prod"),
            ("w2 margin", "%s/w2_holdout.json" % W4, "prod"),
            ("judge v2", "%s/judge_both_holdout.json" % W4, "v2")]),
        ("clean", "cap4", "%s/cap4_clean.json" % W3, [
            ("cap3", "%s/cap3_clean.json" % W3, "prod"),
            ("cap5", "%s/cap5_clean.json" % W3, "prod"),
            ("cap6", "%s/cap6_clean.json" % W3, "prod"),
            ("w1 cap14", "%s/w1cap14_clean.json" % W3, "prod"),
            ("w2 margin", "%s/w2_clean.json" % W4, "prod"),
            ("judge v2", "%s/judge_both_clean.json" % W4, "v2")]),
    ]
    p("  %-8s %-12s %5s %4s %4s %5s %10s" % ("split", "arm", "base", "b", "c", "net", "p_exact"))
    sig = 0
    tot = 0
    for sp, _, basepath, arms in pairs:
        bm = passmap(basepath)
        bp = sum(bm.values())
        for name, path, arm in arms:
            am = passmap(path, arm)
            common = set(bm) & set(am)
            b = sum(1 for q in common if am[q] and not bm[q])
            c = sum(1 for q in common if bm[q] and not am[q])
            pv = mcnemar_exact(b, c)
            tot += 1
            if pv < 0.05:
                sig += 1
            p("  %-8s %-12s %5d %4d %4d %5d %10.4f" % (sp, name, bp, b, c, b - c, pv))
    p("  ★ %d/%d 个配对 p<0.05，占 %.0f%%" % (sig, tot, 100.0 * sig / tot))
    p()
    p("  显著性门槛（精确二项，双侧 α=.05，纯单向）：")
    for k in [1, 2, 3, 4, 5, 6]:
        p("    %d:0 → p=%.4f %s" % (k, mcnemar_exact(k, 0), "<0.05" if mcnemar_exact(k, 0) < 0.05 else ""))
    p("  ★ 6:0 才显著 ⇒ 净翻转 <6 题的改动在 α=.05 上不可证")
    p()
    p("  power=0.8 的检出概率（纯单向，N=480，门槛 6 题）：")
    for g in [4, 6, 8, 10, 15, 20]:
        ph = g / 480.0
        pw = sum(math.comb(480, i) * ph ** i * (1 - ph) ** (480 - i) for i in range(6, 481))
        p("    真实净翻转 %2d 题 (=%.2f pt) → 检出 %.3f %s"
          % (g, 100.0 * g / 480, pw, "OK" if pw >= 0.8 else ""))
    p("  ★ power=.8 的最小可检出净效应 ≈ 8 题 = 1.67 pt（且要求几乎零反向翻转）")


# ---------------------------------------------------------------- 5. 容量算术
def section_capacity() -> None:
    diag = load("data/diag_rank_24000.json")
    P = diag["per_question"]
    p()
    p("=" * 78)
    p("5. 物理约束：单元粒度 × 预算")
    p("=" * 78)
    n = len(P)
    nu = sum(r["n_units"] for r in P) / n
    nt = sum(r["n_truncated"] for r in P) / n
    nc = sum(r["n_candidates"] for r in P) / n
    nh = sum(r["n_hits"] for r in P) / n
    p("  n_units 均值 = %.2f ; n_truncated 均值 = %.2f" % (nu, nt))
    p("  候选被预算丢掉比例 = %.2f%%" % (100.0 * nt / (nu + nt)))
    p("  n_candidates 均值 = %.1f ; n_hits 均值 = %.2f（search top_k 先丢 %.1f）"
      % (nc, nh, nc - nh))
    p("  any_truncated 题数 = %d/%d = %.1f%%"
      % (sum(1 for r in P if r["any_truncated"]), n,
         100.0 * sum(1 for r in P if r["any_truncated"]) / n))
    p("  hit_rank>20 的题数 = %d" % sum(1 for r in P if r["hit_rank"] and r["hit_rank"] > 20))
    p()
    for sp in SPLITS3:
        u = diag["summary"][sp]["units"]
        per = u["ctx_tokens_mean"] / u["n_units_mean"]
        p("  %-8s token/单元 = %.0f ; 预算 24000 可装 %.1f 个" % (sp, per, 24000 / per))
    p()
    p("  ★ 粒度是唯一『不改 token 预算就能多装』的杠杆：")
    for gran in [1276, 638, 320]:
        p("    单元 %4d token → 24000 装 %5.1f 个 → 覆盖 rank ≤ %d"
          % (gran, 24000 / gran, int(24000 / gran)))
    p("  纯加预算：装下全部候选需 %.0f token (x%.1f)"
      % (nc * 1216, nc * 1216 / 24000))


# ---------------------------------------------------------------- 6. 基线缺口
def section_baseline_gap() -> None:
    p()
    p("=" * 78)
    p("6. 可复核性缺口")
    p("=" * 78)
    files = sorted(glob.glob("data/*.json")) + sorted(glob.glob("data/*_evidence/*.json"))
    n_rep = 0
    cap_off_24k = []
    for f in files:
        try:
            d = load(f)
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        t = (d.get("totals") or {}).get("prod") or {}
        if "scorable_pass" not in t:
            continue
        n_rep += 1
        if d.get("token_budget") == 24000 and "chapter_cap" not in (d.get("prod_options") or {}):
            cap_off_24k.append((os.path.basename(f), d.get("split"), t["scorable_pass"]))
    p("  带 totals 的评测报告 = %d 份" % n_rep)
    if cap_off_24k:
        p("  24000 档且未开 chapter_cap 的报告：")
        for x in cap_off_24k:
            p("    %s" % (x,))
    else:
        p("  ★ 24000 档且未开 chapter_cap 的报告：0 份")
        p("  ⇒ W3 对照的 228/152/97 三行无可复算产物，只能当转录数字。")
    p()
    p("  dev 上盘有的 24000 报告：")
    for f in sorted(glob.glob("%s/*_dev.json" % W3)) + sorted(glob.glob("%s/*_dev.json" % W4)):
        d = load(f)
        p("    %-24s pass=%-4d opts=%s"
          % (os.path.basename(f), d["totals"]["prod"]["scorable_pass"],
             ",".join(sorted((d.get("prod_options") or {}).keys()))))
    p("  ⇒ dev 缺 cap3 / cap5 / cap6 三份；W3 正文引用的 _w3_c{3,4,5,6}_dev.json 均不在库。")


def main() -> int:
    # 交接文档 §11.5 的坑：cp936 控制台 print 到 `⊆`/`∈` 会 UnicodeEncodeError。
    # 这里显式把 stdout 切到 utf-8 并用 replace 兜底，避免"整体崩在最后一行"。
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    section_denominator()
    section_partition()
    section_funnel()
    section_judge()
    section_resolution()
    section_capacity()
    section_baseline_gap()
    print("\n".join(OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
