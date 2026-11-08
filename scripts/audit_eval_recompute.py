"""ArkRAG 评测口径独立复算器（只读）

用途：一条命令输出「评测口径对抗性报告」所需的全部核对数字，并与官方读数对照；
      任何不一致都以 DIFF 行打印。本脚本**不写任何文件**、不修改评测产物。

用法：
    py -3 scripts/audit_eval_recompute.py
    py -3 scripts/audit_eval_recompute.py --json      # 机器可读
    py -3 scripts/audit_eval_recompute.py --bootstrap  # 追加 bootstrap 显著性

退出码：0 = 全部核对一致；1 = 存在 DIFF（用于 CI 卡口）。

设计约束（与报告 §C 一致）：
  * 只用标准库 + 本仓库 arkrag 包（零第三方运行时依赖）。
  * 官方数字一律从落盘的 data/eval_quality_*.json 的 arms.prod 里数，
    不信任任何汇总字段；三者必须自洽。
  * 判据复用 src/arkrag/judge.py，绝不在本脚本里重写一份"更宽松的判据"。
"""

import sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse
import collections
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data/goldset/gold-v2.json"
EVAL_FILES = {
    "dev": ROOT / "data/eval_quality_dev.json",
    "holdout": ROOT / "data/eval_quality_holdout.json",
    "clean": ROOT / "data/eval_quality_clean.json",
}

# 官方公布的读数（报告 §C 要核对的四个数）
OFFICIAL = {
    "dev": {"pass": 198, "scorable": 285, "pct": 69.47, "all_pass": 198, "all_n": 350,
            "answered": 346, "all_pct": 56.57, "answered_pct": 98.86},
    "holdout": {"pass": 132, "scorable": 195, "pct": 67.69},
    "clean": {"pass": 83, "scorable": 120, "pct": 69.17},
}

ANCHOR_RE = re.compile(r"「([^」]{1,30})」")
DIFFS = []


def diff(where, expected, actual, tol=0.02):
    ok = (abs(expected - actual) <= tol) if isinstance(expected, float) else (expected == actual)
    if not ok:
        DIFFS.append(f"{where}: 官方={expected} 复算={actual}")
    return ok


def hr(title):
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def pct(a, b):
    return f"{a}/{b} = {a / b * 100:.2f}%" if b else f"{a}/0 = n/a"


def load_json(p):
    return json.loads(p.read_text("utf-8"))


def scorable_categories():
    """优先用 judge.py 的自述，避免本脚本与判据脱钩。"""
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from arkrag import judge as jm
        return set(jm.SCORABLE_CATEGORIES), jm
    except Exception as exc:  # pragma: no cover - 环境缺包时降级
        print(f"[WARN] 无法 import arkrag.judge（{exc}）；分母类别回退到内置清单")
        return {"single_chunk", "alias", "direct_relation", "multihop", "cross_chunk"}, None


def build_chapter_texts():
    """复刻 scripts/eval_answer_quality.py:chapter_texts 的构建方式（只读，不 import 该脚本）。"""
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from arkrag import pipeline as pl
    except Exception as exc:
        print(f"[WARN] 无法 import arkrag.pipeline（{exc}）；§C2 将只比对非 cross_chunk 题")
        return None
    gen = None
    try:
        from arkrag import config
        gen = getattr(config, "DEFAULT_GENERATION", None) or getattr(config, "GENERATION", None)
    except Exception:
        pass
    if not gen:
        gen = "g-20261103T132152Z-06a741"
    try:
        loaded = pl.load_generation(gen)
    except Exception as exc:
        print(f"[WARN] load_generation({gen}) 失败：{exc}；§C2 跳过 cross_chunk")
        return None
    out = collections.defaultdict(list)
    for rec in loaded["chunks"].values():
        out[str(rec.get("chapter") or "")].append(str(rec.get("text") or ""))
    return {k: "\n".join(v) for k, v in out.items()}


# ---------------------------------------------------------------- C. 独立复算
def section_c(gold_items, scorable, chap=None):
    hr("§C 独立复算：从 data/eval_quality_dev.json 重算 per-category 与 totals")
    cat_set = scorable[0]
    for split, path in EVAL_FILES.items():
        if not path.exists():
            print(f"[SKIP] {path} 不存在")
            continue
        doc = load_json(path)
        rows = doc["rows"]
        p = collections.Counter()
        n = collections.Counter()
        all_pass = all_n = answered = 0
        reasons = collections.Counter()
        for r in rows:
            all_n += 1
            all_pass += bool(r["arms"]["prod"]["pass"])
            answered += bool(r["arms"]["prod"].get("answered"))
            if r["category"] not in cat_set:
                reasons[f"[非可判] {r['category']}"] += 1
                continue
            n[r["category"]] += 1
            ok = bool(r["arms"]["prod"]["pass"])
            p[r["category"]] += ok
            if not ok:
                reasons[f"{r['category']} / {r['arms']['prod'].get('reason') or '?'}"] += 1
        tp, tn = sum(p.values()), sum(n.values())
        print(f"\n--- split={split}  文件={path.name}  n_rows={all_n}  "
              f"budget={doc.get('token_budget')} ---")
        for cat in sorted(n):
            print(f"    {cat:16s} {pct(p[cat], n[cat]):>22}")
        print(f"    {'TOTAL(scorable)':16s} {pct(tp, tn):>22}")
        print(f"    {'TOTAL(all rows)':16s} {pct(all_pass, all_n):>22}")
        print(f"    {'answered':16s} {pct(answered, all_n):>22}")

        off = OFFICIAL.get(split)
        if off:
            diff(f"{split}.pass", off["pass"], tp)
            diff(f"{split}.scorable", off["scorable"], tn)
            diff(f"{split}.pct", off["pct"], round(tp / tn * 100, 2))
            if "all_pass" in off:
                diff(f"{split}.all_pass", off["all_pass"], all_pass)
                diff(f"{split}.all_n", off["all_n"], all_n)
                diff(f"{split}.all_pct", off["all_pct"], round(all_pass / all_n * 100, 2))
                diff(f"{split}.answered", off["answered"], answered)
                diff(f"{split}.answered_pct", off["answered_pct"],
                     round(answered / all_n * 100, 2))
        print(f"    失败理由分布（前 12）：")
        for k, v in reasons.most_common(12):
            print(f"       {v:>4}  {k}")

    # 用 judge.py 复核落盘 judge 字段是否与当前判据一致
    hr("§C2 落盘 judge 字段 vs 现场重跑 judge_answer（判据是否被改过）")
    if not scorable:
        return
    cat_set, jm = scorable[0], scorable[1]
    if jm is None:
        return
    items = {it["question_id"]: it for it in gold_items}
    for split, path in EVAL_FILES.items():
        if not path.exists():
            continue
        rows = load_json(path)["rows"]
        mism = 0
        sample = []
        for r in rows:
            it = items.get(r["question_id"])
            if it is None or r["category"] not in cat_set:
                continue
            v = jm.judge_answer(r["category"], r["answer"], it, question=r["question"],
                                chapter_texts=chap)
            if bool(v["pass"]) != bool(r["arms"]["prod"]["pass"]) or \
               (v.get("reason") or "") != (r["arms"]["prod"].get("reason") or ""):
                mism += 1
                if len(sample) < 5:
                    sample.append((r["question_id"], r["arms"]["prod"].get("reason"),
                                   v.get("reason")))
        total = sum(1 for r in rows if r["question_id"] in items and r["category"] in cat_set)
        print(f"    {split:8s} 不一致 {mism}/{total}  "
              f"{'✅ 判据未漂移' if mism == 0 else '❌ 判据已漂移'}")
        for s in sample:
            print(f"        {s[0]}  落盘={s[1]}  重跑={s[2]}")
        diff(f"{split}.judge_drift", 0, mism)


# ---------------------------------------------------------------- B. 泄漏审计
def derive_pools(gold_items):
    """复刻 scripts/eval_answer_quality.py:pick_items 的选题口径。

    gold-v2 里 split 只有 dev / holdout 两个值；`clean` 是**派生**出来的：
    holdout ∩ 不在 gold-v1 里 ∩ 有 gold_chunk_ids。
    """
    v1_path = ROOT / "data/goldset/gold-v1.json"
    v1 = set()
    if v1_path.exists():
        doc = load_json(v1_path)
        its = doc.get("items") if isinstance(doc, dict) else doc
        v1 = {str(it.get("question_id")) for it in its}
    pools = {
        "dev": [it for it in gold_items
                if it.get("split") == "dev" and it.get("gold_chunk_ids")],
        "holdout": [it for it in gold_items
                    if it.get("split") == "holdout" and it.get("gold_chunk_ids")],
        "clean": [it for it in gold_items
                  if it.get("split") == "holdout" and it.get("gold_chunk_ids")
                  and str(it.get("question_id")) not in v1],
    }
    return pools, v1


def section_b(gold_items, scorable, chap=None):
    hr("§B 数据泄漏审计")
    cat_set = scorable[0]
    pools, v1 = derive_pools(gold_items)
    by_split = {k: v for k, v in pools.items()}
    print(f"\n     评测池口径（复刻 pick_items）：dev={len(pools['dev'])} "
          f"holdout={len(pools['holdout'])} clean={len(pools['clean'])}"
          f"（clean = holdout ∩ 不在 gold-v1[{len(v1)} 题] ∩ 有 gold_chunk_ids）")

    print(f"\n[B1] question 文本里是否已含 gold 字符串（逐类别 × 逐 split，评测池口径）")
    print(f"     {'split':8s} {'类别':18s} {'含gold':>12s} {'占比':>10s}")
    stat = collections.defaultdict(lambda: [0, 0])   # (split, cat) -> [hit, n]
    for split, pool in by_split.items():
        for it in pool:
            if it["category"] not in cat_set:
                continue
            key = (split, it["category"])
            stat[key][1] += 1
            gold = str(it["answer"])
            stat[key][0] += bool(gold) and gold in it["question"]
    for split in ("dev", "holdout", "clean"):
        sp_hit = sp_n = 0
        for (s, c) in sorted(stat):
            if s != split:
                continue
            h, n = stat[(s, c)]
            sp_hit += h
            sp_n += n
            print(f"     {split:8s} {c:18s} {h:>5}/{n:<6} {h / n * 100:>9.2f}%")
        print(f"     {split:8s} {'小计（可判类）':18s} {sp_hit:>5}/{sp_n:<6} "
              f"{sp_hit / sp_n * 100 if sp_n else 0:>9.2f}%")

    print(f"\n[B2] 三份 split 的 question_id 是否互斥")
    sets = {s: {it["question_id"] for it in by_split.get(s, [])}
            for s in ("dev", "holdout", "clean")}
    for s, v in sets.items():
        print(f"     {s:8s} {len(v)} 个 question_id")
    print(f"     注意 clean 是 holdout 的**派生**子集（非独立抽样），两者必然重叠；")
    print(f"     真正的互斥要求是 dev 与 holdout / clean 无交集：")
    for a, b in (("dev", "holdout"), ("dev", "clean")):
        inter = sets[a] & sets[b]
        print(f"     {a} ∩ {b} = {len(inter)} {'✅互斥' if not inter else '❌重叠'}")
        diff(f"ids.{a}_{b}", 0, len(inter))
    inter = sets["holdout"] & sets["clean"]
    print(f"     holdout ∩ clean = {len(inter)}（= clean 全部，设计使然，"
          f"{'✅ clean ⊆ holdout' if inter == sets['clean'] else '⚠️ 非子集，异常'}）")

    print(f"\n[B3] dev × holdout 近似重复（字符 3-gram Jaccard ≥ 0.6）")

    def tri(s):
        s = re.sub(r"\s+", "", s or "")
        return {s[i:i + 3] for i in range(len(s) - 2)}

    dev = [(it["question_id"], it["category"], tri(it["question"]))
           for it in by_split.get("dev", [])]
    ho = [(it["question_id"], it["category"], tri(it["question"]))
          for it in by_split.get("holdout", [])]
    pairs = []
    for qid_a, ca, ta in dev:
        for qid_b, cb, tb in ho:
            if not ta or not tb:
                continue
            j = len(ta & tb) / len(ta | tb)
            if j >= 0.6:
                pairs.append((j, qid_a, ca, qid_b, cb))
    pairs.sort(reverse=True)
    same_cat = [p for p in pairs if p[2] == p[4]]
    cross_cat = [p for p in pairs if p[2] != p[4]]
    print(f"     ≥0.6 的对数：{len(pairs)}  （同类 {len(same_cat)} / 跨类 {len(cross_cat)}）")
    print(f"     分母：dev 池 {len(dev)} × holdout 池 {len(ho)} = {len(dev) * len(ho)} 对")
    for j, qa, ca, qb, cb in pairs[:8]:
        print(f"       {j:.3f}  {qa}({ca}) ~ {qb}({cb})")
    print(f"     按类别拆（同类 ≥0.6）：")
    cc = collections.Counter(p[2] for p in same_cat)
    for k, v in cc.most_common():
        print(f"       {k:18s} {v} 对")

    print(f"\n[B4] gold_chunk_ids 在 dev / holdout 之间的重叠")
    gc = collections.defaultdict(set)
    for it in by_split.get("dev", []):
        gc["dev"] |= set(it.get("gold_chunk_ids") or [])
    for it in by_split.get("holdout", []):
        gc["holdout"] |= set(it.get("gold_chunk_ids") or [])
    for it in by_split.get("clean", []):
        gc["clean"] |= set(it.get("gold_chunk_ids") or [])
    for s, v in gc.items():
        print(f"     {s:8s} gold_chunk_ids 并集 {len(v)} 块")
    for a, b in (("dev", "holdout"), ("dev", "clean"), ("holdout", "clean")):
        inter = gc[a] & gc[b]
        base = len(gc[b]) or 1
        print(f"     {a} ∩ {b} = {len(inter)} 块（占 {b} 的 {len(inter) / base * 100:.2f}%）")

    print(f"\n[B5] gold 集合完全相同（同 gold_chunk_ids）的题组，跨 split")
    sig = collections.defaultdict(list)
    for it in gold_items:
        gids = tuple(sorted(it.get("gold_chunk_ids") or []))
        if gids:
            sig[gids].append((it["split"], it["question_id"]))
    dup = {k: v for k, v in sig.items() if len(v) > 1}
    print(f"     共 {len(dup)} 组重复 gold 集合")
    for k, v in list(dup.items())[:5]:
        print(f"       {len(k)} 块 ← {v}")

    print(f"\n[B6] 被 pick_items 静默踢出分母的题（无 gold_chunk_ids）")
    raw = collections.Counter(it["split"] for it in gold_items)
    for split in ("dev", "holdout"):
        its = [it for it in gold_items if it["split"] == split]
        nog = sum(1 for it in its if not (it.get("gold_chunk_ids") or []))
        print(f"     {split:8s} 原始 {len(its)} 题，其中 {nog} 题无 gold_chunk_ids "
              f"→ 静默不进评测池（池 {len(its) - nog}）")
    print(f"     clean   派生自 holdout：holdout {raw.get('holdout', 0)} 题中 "
          f"{len(pools['clean'])} 题满足「非 gold-v1 ∩ 有 gold_chunk_ids」")
    print(f"     合计被静默剔除：{sum(1 for it in gold_items if not it.get('gold_chunk_ids'))}"
          f"/{len(gold_items)} 题（{sum(1 for it in gold_items if not it.get('gold_chunk_ids')) / len(gold_items) * 100:.2f}%）")

    print(f"\n[B7] 零检索地板：直接把题干当答案回显，judge 是否给分")
    if scorable[1] is None:
        return
    jm = scorable[1]
    items = {it["question_id"]: it for it in gold_items}
    for split, path in EVAL_FILES.items():
        if not path.exists():
            continue
        rows = load_json(path)["rows"]
        p = n = 0
        for r in rows:
            if r["category"] not in cat_set:
                continue
            it = items.get(r["question_id"])
            if it is None:
                continue
            n += 1
            v = jm.judge_answer(r["category"], r["question"], it, question=r["question"],
                                chapter_texts=chap)
            p += bool(v["pass"])
        print(f"     {split:8s} 回显题干 {pct(p, n)}  ← 这是任何改动的「零检索地板」")


# ---------------------------------------------------------------- D. bootstrap
def section_d(gold_items, scorable, n_boot=20000, seed=7):
    hr(f"§D bootstrap 显著性协议（逐题配对，{n_boot} 次重抽，单侧 H1: 改动更好）")
    cat_set = scorable[0]
    path = EVAL_FILES["dev"]
    if not path.exists():
        print("[SKIP] 缺 dev 读数")
        return
    rows = [r for r in load_json(path)["rows"] if r["category"] in cat_set]
    base = [1 if r["arms"]["prod"]["pass"] else 0 for r in rows]
    print("     把两条臂的逐题 0/1 向量喂进来即可复用；下面用 baseline vs baseline")
    print("     演示（Δ 应恒为 0），真实调用见各 tmp_adv_*.py 的同名函数。")
    obs, mean_pp, lo, hi, p_one = _paired_bootstrap(base, base, n_boot, seed)
    print(f"     基线自身：Δ={obs:+d} 题 ({mean_pp:+.2f}pp) 95%CI=[{lo:+d},{hi:+d}] "
          f"单侧 p={p_one:.4f}")
    print("\n     协议（任一改动必须全满足，否则判失败）：")
    for i, rule in enumerate([
        "dev 上 Δ ≥ +1 题（可判分母不变、题集不变、判据文件 hash 不变）",
        "holdout 上 Δ ≥ 0 题（不允许用 dev 挑方案后 holdout 变差）",
        "逐题配对 bootstrap 单侧 p < 0.05",
        "clean 上 Δ ≥ 0 题",
        "净增益 = pass − 零检索地板(§B7) 必须 > 0",
        "所有报告数字必须能被 scripts/audit_eval_recompute.py 复现（0 条 DIFF）",
    ], 1):
        print(f"       {i}. {rule}")


def _paired_bootstrap(a, b, n=20000, seed=7):
    d = [y - x for x, y in zip(a, b)]
    rnd = random.Random(seed)
    idx = list(range(len(d)))
    stats = sorted(sum(d[rnd.choice(idx)] for _ in idx) for _ in range(n))
    obs = sum(d)
    lo = stats[int(0.025 * n)]
    hi = stats[min(n - 1, int(0.975 * n))]
    p_one = sum(1 for s in stats if s <= 0) / n
    return obs, obs / len(d) * 100 if d else 0.0, lo, hi, p_one


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="额外打印机器可读汇总")
    ap.add_argument("--bootstrap", action="store_true", help="追加 bootstrap 演示节")
    ap.add_argument("--with-leak", action="store_true", help="追加 §B 泄漏审计（较慢）")
    args = ap.parse_args()

    print("ArkRAG 评测口径独立复算（只读）")
    print(f"仓库根：{ROOT}")
    if not GOLD.exists():
        print(f"[FATAL] 缺 {GOLD}")
        return 2
    gold_items = load_json(GOLD)["items"]
    print(f"gold-v2 题数：{len(gold_items)}")
    print(f"split 分布：{dict(collections.Counter(it['split'] for it in gold_items))}")
    print(f"类别分布：{dict(collections.Counter(it['category'] for it in gold_items))}")

    scorable = scorable_categories()
    if scorable[0]:
        print(f"可判类别（judge.SCORABLE_CATEGORIES）：{sorted(scorable[0])}")
    chap = build_chapter_texts() if scorable[1] is not None else None
    print(f"chapter_texts：{'已构建 %d 章' % len(chap) if chap else '不可用（cross_chunk 降级为集合相等）'}")

    section_c(gold_items, scorable, chap)
    if args.with_leak:
        section_b(gold_items, scorable, chap)
    if args.bootstrap:
        section_d(gold_items, scorable)

    hr("核对结果")
    if DIFFS:
        print(f"❌ 发现 {len(DIFFS)} 处与官方数字不一致：")
        for d in DIFFS:
            print("   DIFF " + d)
    else:
        print("✅ 全部官方数字复现成功，0 条 DIFF")
    if args.json:
        print(json.dumps({"diffs": DIFFS, "official": OFFICIAL}, ensure_ascii=False, indent=2))
    return 1 if DIFFS else 0


if __name__ == "__main__":
    sys.exit(main())
