r"""第六轮对抗审查复算器（只读）：审计**已接线的真实代码**。

被审对象：
  * `src/arkrag/quote_locate.py`（引文逐字定位，single_chunk）
  * `src/arkrag/bridge.py`（上下文内唯一公共邻居桥接，multihop）
  * `src/arkrag/retrieve.py` 的 `graph_reserve_tokens`（图证据预留额度）
  * `src/arkrag/pipeline.py` 的 `ask_once(options=...)` 接线

用法（Windows / pwsh）：
    $env:PYTHONPATH="E:\图rag\src"; py -3 scripts\audit_round6.py
    py -3 scripts\audit_round6.py --only r1          # 单节
    py -3 scripts\audit_round6.py --json out.json    # 落盘
    py -3 scripts\audit_round6.py --only r1 --nobridge-recompute   # 跳过最慢的逐题桥接

节：
    r1  独立复算 baseline vs bundle（口径 A = verify_citations 后判，红线 33）+ bootstrap
    r2  组件边际贡献（OFF / QL / BR / BR+reserve / BR+reserve(k20) / ALL）+ 逐类
    r3  reserve=0 vs 12000 的三条 Q2 判据（均长 / 断言数 / 召回）+ 上下文归因
    r4  bridge 结构性接地审计 + pass ≡ (X==gold) + 构造性命中率
    r5  quote_locate 定位器审计 + 门控全 640 题复核（用 goldset 原文）
    r6  形态红线 25 + 达标判定 + 基线口径对齐（同预算）

只读：不修改 src/、scripts/、data/goldset/、data/eval_quality_*.json。
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import inspect
import json
import random
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from arkrag import pipeline as P          # noqa: E402
from arkrag import judge as J             # noqa: E402
from arkrag import answer as answer_mod   # noqa: E402
from arkrag import bridge as bridge_mod   # noqa: E402
from arkrag import quote_locate as ql     # noqa: E402

GEN = "g-20261103T132152Z-06a741"
SCORABLE = set(J.SCORABLE_CATEGORIES)
TOP_K = 20
BUDGET = 24000
MAX_CLAIMS = 8

BUNDLE = {"quote_locate": True, "bridge": True,
          "graph_reserve_tokens": 12000, "bridge_search_top_k": 40}
BR_ONLY = {"bridge": True}
BR_RS = {"bridge": True, "graph_reserve_tokens": 12000, "bridge_search_top_k": 40}
BR_RS_K20 = {"bridge": True, "graph_reserve_tokens": 12000}
QL_ONLY = {"quote_locate": True}
QL_BR = {"quote_locate": True, "bridge": True}
OFF: dict = {}


# ---------------------------------------------------------------- 基础设施
def section(title: str) -> None:
    print()
    print("=" * 104)
    print(title)
    print("=" * 104)


class World:
    """一次装载：世代 + gold-v2 + clean 派生 + 章文本。"""

    def __init__(self) -> None:
        loaded = P.load_generation(GEN)
        self.loaded = loaded
        self.graph = loaded["graph"]
        self.chunks = loaded["chunks"]
        self.cid = list(self.chunks)
        chap = collections.defaultdict(list)
        for c in self.cid:
            chap[str(self.chunks[c].get("chapter") or "")].append(
                str(self.chunks[c].get("text") or ""))
        self.chapter_texts = {k: "\n".join(v) for k, v in chap.items()}
        gold = json.loads((ROOT / "data/goldset/gold-v2.json").read_text("utf-8"))
        self.items = gold["items"]
        v1 = {it["question_id"] for it in json.loads(
            (ROOT / "data/goldset/gold-v1.json").read_text("utf-8"))["items"]}
        self.pools = {
            "dev": [it for it in self.items
                    if it.get("split") == "dev" and it.get("gold_chunk_ids")],
            "holdout": [it for it in self.items
                        if it.get("split") == "holdout" and it.get("gold_chunk_ids")],
            "clean": [it for it in self.items
                      if it.get("split") == "holdout" and it["question_id"] not in v1
                      and it.get("gold_chunk_ids")],
        }
        self._pipes: dict[int, dict] = {}

    def pipe(self, budget: int = BUDGET) -> dict:
        if budget not in self._pipes:
            p = P.build_pipeline(self.loaded, top_k=TOP_K, graph_depth=1)
            p["generation_id"] = GEN
            p["entities"] = self.graph["entities"]
            p["retriever"].budget["max_context_tokens"] = budget
            p["answerer"].max_claims = MAX_CLAIMS
            self._pipes[budget] = p
        return self._pipes[budget]

    def answer(self, item: dict, opts: dict, budget: int = BUDGET) -> dict:
        """跑一次问答，返回 `{pass, reason, text, raw, res}`（口径 A = 红线 33）。"""
        r = P.ask_once(self.pipe(budget), item["question"], options=opts)
        ans = r["answer"] or {}
        raw = str(ans.get("answer") or "")
        cits = ans.get("citations") or []
        ver = answer_mod.verify_citations(ans, self.chunks) if cits else {
            "answer": raw, "problems": []}
        text = str(ver.get("answer") or "")
        jr = J.judge_answer(item["category"], text, item,
                            question=item.get("question"),
                            chapter_texts=self.chapter_texts)
        jr_raw = J.judge_answer(item["category"], raw, item,
                                question=item.get("question"),
                                chapter_texts=self.chapter_texts)
        return {"pass": jr.get("pass") is True, "reason": jr.get("reason"),
                "raw_pass": jr_raw.get("pass") is True, "text": text, "raw": raw,
                "cleared": bool(raw.strip()) and not text.strip(), "res": r}

    def scorable(self, split: str) -> list[dict]:
        return [it for it in self.pools[split] if it["category"] in SCORABLE]


def paired_bootstrap(better: list[int], worse: list[int],
                     n: int = 20000, seed: int = 7) -> dict:
    """逐题配对重抽，单侧 H1: better > worse。返回 Δ / CI / p。"""
    d = [b - a for a, b in zip(worse, better)]
    rng = random.Random(seed)
    idx = list(range(len(d)))
    sums = sorted(sum(d[rng.choice(idx)] for _ in idx) for _ in range(n))
    total = sum(d)
    lo = sums[int(n * .025)]
    hi = sums[int(n * .975)]
    p = sum(1 for s in sums if s <= 0) / n
    return {"delta": total, "ci_low": lo, "ci_high": hi, "p": p,
            "pass": total >= 1 and lo > 0 and p < .05}


REPORT: dict = {}


# ---------------------------------------------------------------- r1
def r1(world: World, nobridge: bool = False) -> None:
    section("§r1 独立复算：baseline vs bundle（tk20 / B24000 / MC8 / bridge_search_top_k=40 / reserve=12000）\n"
            "     口径 A = verify_citations 后判（红线 33）；口径 B = 不套引用纪律（并列披露）")
    out = {}
    for split in ("dev", "holdout", "clean"):
        items = world.scorable(split)
        per_b = collections.Counter()
        per_u = collections.Counter()
        tot = collections.Counter()
        vb, vu, vb_raw, vu_raw = [], [], [], []
        losses, cleared = [], 0
        for it in items:
            cat = it["category"]
            tot[cat] += 1
            b = world.answer(it, OFF)
            u = world.answer(it, BUNDLE)
            per_b[cat] += b["pass"]
            per_u[cat] += u["pass"]
            vb.append(int(b["pass"]))
            vu.append(int(u["pass"]))
            vb_raw.append(int(b["raw_pass"]))
            vu_raw.append(int(u["raw_pass"]))
            cleared += int(u["cleared"])
            if b["pass"] and not u["pass"]:
                losses.append((it["question_id"], u["reason"]))
        n = len(items)
        boot = paired_bootstrap(vu, vb)
        out[split] = {
            "n": n,
            "base": sum(per_b.values()), "union": sum(per_u.values()),
            "base_pct": round(sum(per_b.values()) / n * 100, 2),
            "union_pct": round(sum(per_u.values()) / n * 100, 2),
            "per_cat_base": dict(per_b), "per_cat_union": dict(per_u),
            "per_cat_total": dict(tot),
            "raw_rule_base": sum(vb_raw), "raw_rule_union": sum(vu_raw),
            "cleared_by_citations": cleared,
            "losses": losses, "bootstrap": boot,
        }
        print(f"  {split:8s} n={n}")
        print(f"     口径A 基线 {sum(per_b.values())}/{n}={sum(per_b.values())/n*100:.2f}%  "
              f"bundle {sum(per_u.values())}/{n}={sum(per_u.values())/n*100:.2f}%  "
              f"Δ={sum(per_u.values())-sum(per_b.values()):+d}")
        print(f"     口径B 基线 {sum(vb_raw)}/{n}  bundle {sum(vu_raw)}/{n}  "
              f"（两口径差 = {sum(vu_raw)-sum(per_u.values()):+d} 题，"
              f"被 verify_citations 清空 {cleared} 题）")
        print("     逐类A 基线 " + " ".join(f"{k}:{per_b[k]}/{tot[k]}" for k in sorted(tot)))
        print("     逐类A bundle " + " ".join(f"{k}:{per_u[k]}/{tot[k]}" for k in sorted(tot)))
        print(f"     回退/丢失 {len(losses)}: {losses}")
        print(f"     bootstrap Δ={boot['delta']:+d} CI=[{boot['ci_low']:+d},{boot['ci_high']:+d}] "
              f"p={boot['p']:.4f} {'PASS' if boot['pass'] else 'FAIL'}")
    REPORT["r1"] = out


# ---------------------------------------------------------------- r2
def r2(world: World) -> None:
    section("§r2 组件边际贡献（tk20 / B24000 / MC8；口径 A）")
    configs = {"OFF": OFF, "QL": QL_ONLY, "BR": BR_ONLY,
               "BR+RS12k": BR_RS, "BR+RS12k(k20)": BR_RS_K20, "ALL(bundle)": BUNDLE}
    out = {}
    for split in ("dev", "holdout", "clean"):
        items = world.scorable(split)
        print(f"  --- {split} ---")
        rows = {}
        for name, opts in configs.items():
            per = collections.Counter()
            tot = collections.Counter()
            for it in items:
                tot[it["category"]] += 1
                if world.answer(it, opts)["pass"]:
                    per[it["category"]] += 1
            n = len(items)
            rows[name] = (sum(per.values()), dict(per), dict(tot))
            print(f"    {name:16s} {sum(per.values())}/{n}={sum(per.values())/n*100:6.2f}%  "
                  + " ".join(f"{k[:3]}:{per[k]}" for k in
                             ("single_chunk", "alias", "direct_relation",
                              "multihop", "cross_chunk")))
        off = rows["OFF"][0]
        for name in configs:
            if name != "OFF":
                print(f"    边际 {name:16s} Δ={rows[name][0]-off:+d}")
        out[split] = {k: v[0] for k, v in rows.items()}
        out[split + "_per_cat"] = {k: v[1] for k, v in rows.items()}
    REPORT["r2"] = out


# ---------------------------------------------------------------- r3
def r3(world: World) -> None:
    section("§r3 reserve=0 vs 12000（仅桥接题）：Q2 三条判据 + 上下文归因")
    out = {}
    for split in ("dev", "holdout", "clean"):
        for name, opts in (("BR", BR_ONLY), ("BR+RS12k+tk40", BR_RS)):
            rows = []
            agg = collections.Counter()
            for it in world.scorable(split):
                if it["category"] != "multihop":
                    continue
                a = world.answer(it, opts)
                units = (a["res"]["context"] or {}).get("units") or []
                st = (a["res"]["context"] or {}).get("stats") or {}
                segs = [s for s in a["text"].split("。") if s.strip()]
                golds = [str(it.get("answer") or "")] + list(it.get("acceptable_answers") or [])
                rec = any(any(g and g in str(u.get("text") or "") for g in golds if len(g) >= 2)
                          for u in units)
                rows.append({"len": len(a["text"]), "sent": len(segs),
                             "claim": int(st.get("evidence_spans") or 0),
                             "recall": rec, "pass": a["pass"]})
                agg["units"] += int(st.get("units") or 0)
                agg["direct"] += int(st.get("direct_units") or 0)
                agg["graph"] += int(st.get("graph_only_units") or 0)
                agg["tok"] += int(st.get("tokens_used") or 0)
                agg["trunc"] += len(st.get("truncated_chunks") or ())
            n = len(rows)
            if not n:
                continue
            key = f"{split}/{name}"
            out[key] = {
                "n": n, "pass": sum(1 for r in rows if r["pass"]),
                "mean_len": round(sum(r["len"] for r in rows) / n, 1),
                "mean_sent": round(sum(r["sent"] for r in rows) / n, 2),
                "mean_claims": round(sum(r["claim"] for r in rows) / n, 2),
                "recall_pct": round(sum(1 for r in rows if r["recall"]) / n * 100, 1),
                "mean_tokens": round(agg["tok"] / n),
                "mean_units": round(agg["units"] / n, 2),
                "mean_direct": round(agg["direct"] / n, 2),
                "mean_graph": round(agg["graph"] / n, 2),
                "mean_trunc": round(agg["trunc"] / n, 2),
            }
            v = out[key]
            print(f"  {split:8s} {name:14s} n={n} pass={v['pass']} 均长={v['mean_len']:6.1f} "
                  f"均句={v['mean_sent']:5.2f} 均断言={v['mean_claims']:5.2f} "
                  f"召回={v['recall_pct']:5.1f}% 均tok={v['mean_tokens']:7d} "
                  f"| 均units={v['mean_units']:5.2f} direct={v['mean_direct']:5.2f} "
                  f"graph={v['mean_graph']:5.2f} trunc={v['mean_trunc']:5.2f}")
    # 同一 budget 的 bootstrap：BR vs BR+reserve
    print("  同一 budget(24000) 下 reserve+tk40 相对 BR-only 的增量 bootstrap：")
    for split in ("dev", "holdout", "clean"):
        x, y = [], []
        for it in world.scorable(split):
            x.append(int(world.answer(it, BR_ONLY)["pass"]))
            y.append(int(world.answer(it, BR_RS)["pass"]))
        boot = paired_bootstrap(y, x)
        out[f"boot/{split}"] = boot
        print(f"    {split:8s} BR {sum(x)} → BR+RS {sum(y)}  Δ={boot['delta']:+d} "
              f"CI=[{boot['ci_low']:+d},{boot['ci_high']:+d}] p={boot['p']:.4f} "
              f"{'PASS' if boot['pass'] else 'FAIL'}")
    REPORT["r3"] = out


# ---------------------------------------------------------------- r4
def r4(world: World, nobridge: bool = False) -> None:
    section("§r4 bridge 结构性接地 + pass ≡ (X==gold) + 构造性命中率")
    src = inspect.getsource(bridge_mod)
    struct = {"signature": str(inspect.signature(bridge_mod.find_bridge)),
              "keywords": {k: src.count(k) for k in
                           ("graph", "adjacency", "entities", "alias_index",
                            "neighbour", "neighbor_map")},
              "imports": [l.strip() for l in src.splitlines()
                          if l.startswith(("import", "from"))]}
    print(f"   签名 {struct['signature']}")
    print(f"   源码关键词计数 {struct['keywords']}")
    print(f"   import {struct['imports']}")
    print("   ⇒ 无 graph/adjacency 形参、源码 0 次图引用 ⇒ 结构上拿不到全图。")
    ret = world.pipe()["retriever"]
    out = {"struct": struct}
    for split in ("dev", "holdout", "clean"):
        st = collections.Counter()
        for it in world.scorable(split):
            if it["category"] != "multihop":
                continue
            a = world.answer(it, BUNDLE)
            if nobridge:
                continue
            br = bridge_mod.find_bridge(
                it["question"], (a["res"]["context"] or {}).get("units") or [],
                chunk_assertions=ret.assertions_by_chunk.get,
                display=ret.display_name)
            st["n"] += 1
            st["pass"] += a["pass"]
            if br["status"] != "ok":
                st["no_x"] += 1
                st["pass_no_x"] += a["pass"]
                continue
            golds = [str(it.get("answer") or "")] + list(it.get("acceptable_answers") or [])
            eq = any(g and br["x"] == g for g in golds)
            st["ok"] += 1
            st["x_eq_gold"] += eq
            st["pass_and_eq"] += (a["pass"] and eq)
            st["pass_not_eq"] += (a["pass"] and not eq)
            st["eq_not_pass"] += ((not a["pass"]) and eq)
        if nobridge:
            continue
        n = st["n"]
        out[split] = {k: v for k, v in st.items()}
        print(f"   {split:8s} n={n} pass={st['pass']} 唯一候选={st['ok']} "
              f"X==gold={st['x_eq_gold']} pass&XeqG={st['pass_and_eq']} "
              f"pass&X≠gold={st['pass_not_eq']} X==gold未pass={st['eq_not_pass']} "
              f"无候选={st['no_x']}(其中pass {st['pass_no_x']})")
        if st["ok"]:
            print(f"            条件命中率 {st['x_eq_gold']}/{st['ok']}"
                  f"={st['x_eq_gold']/st['ok']*100:.1f}%；"
                  f"占全部 {st['x_eq_gold']}/{n}={st['x_eq_gold']/n*100:.1f}%")
    REPORT["r4"] = out


# ---------------------------------------------------------------- r5
def r5(world: World) -> None:
    section("§r5 quote_locate 定位器审计 + 门控全 640 题复核")
    gate_loose = collections.Counter()
    gate = collections.Counter()
    by_split = collections.Counter()
    lens = []
    for it in world.items:
        q = it["question"]
        if ql.is_title_question(q):
            gate_loose[it["category"]] += 1
        loc = ql.locate(q, {})
        if loc["skipped_reason"] not in ("not_title_question", "no_probe_long_enough",
                                         "probe_too_short"):
            gate[it["category"]] += 1
            by_split[it.get("split")] += 1
            if loc["probe"]:
                lens.append(len(loc["probe"]))
    sample_q = next(it["question"] for it in world.items if it["category"] == "single_chunk")
    keys = sorted(ql.locate(sample_q, {}).keys())
    lens.sort()
    out = {
        "gate_trigger_only": dict(gate_loose),
        "gate_trigger_and_probe": dict(gate),
        "gate_by_split": dict(by_split),
        "probe_len": {"min": lens[0] if lens else 0,
                      "p50": lens[len(lens) // 2] if lens else 0,
                      "max": lens[-1] if lens else 0},
        "locate_keys": keys,
        "returns_title": "title" in keys,
        "title_triggers": list(answer_mod.TITLE_TRIGGERS),
    }
    print(f"   只用触发词            {dict(gate_loose)} 合计 {sum(gate_loose.values())}/640")
    print(f"   触发词 + ≥8字引文     {dict(gate)} 合计 {sum(gate.values())}/640")
    print(f"   逐 split              {dict(by_split)}")
    print(f"   引文长度              min={out['probe_len']['min']} p50={out['probe_len']['p50']} "
          f"max={out['probe_len']['max']}")
    print(f"   locate 返回键         {keys}  含 title: {out['returns_title']}")
    print(f"   TITLE_TRIGGERS        {out['title_triggers']}")
    print("   ⇒ 只返回 {probe, chunk_ids, skipped_reason}；title 由 answer._title_answer 从 unit 取。")
    REPORT["r5"] = out


# ---------------------------------------------------------------- r6
def r6(world: World) -> None:
    section("§r6 形态红线 25 + 达标判定 + 基线口径对齐（同预算）")
    out = {}
    for split in ("dev", "holdout", "clean"):
        rows = []
        for it in world.scorable(split):
            a = world.answer(it, BUNDLE)
            segs = [s for s in a["text"].split("。") if s.strip()]
            rows.append((len(a["text"]), len(segs), max((len(s) for s in segs), default=0)))
        n = len(rows)
        lens = sorted(r[0] for r in rows)
        v = {"n": n,
             "len_p50": lens[n // 2], "len_max": lens[-1],
             "over_300": sum(1 for r in rows if r[0] > 300),
             "sent_max": max(r[1] for r in rows),
             "over_8_sent": sum(1 for r in rows if r[1] > 8),
             "sent_len_max": max(r[2] for r in rows),
             "over_60": sum(1 for r in rows if r[2] > 60),
             "need_for_80pct": int(n * .8) + 1}
        out[split] = v
        print(f"  {split:8s} n={n} 总长 p50={v['len_p50']} max={v['len_max']} >300={v['over_300']} "
              f"| 句数 max={v['sent_max']} >8={v['over_8_sent']} "
              f"| 单句 max={v['sent_len_max']} >60={v['over_60']} "
              f"| 过 80% 需 ≥{v['need_for_80pct']} 题")
    print("  基线口径对齐（同 budget 才是同口径）：")
    for split in ("dev", "holdout", "clean"):
        items = world.scorable(split)
        a = sum(int(world.answer(it, OFF, 12000)["pass"]) for it in items)
        b = sum(int(world.answer(it, OFF, 24000)["pass"]) for it in items)
        c = sum(int(world.answer(it, BUNDLE, 24000)["pass"]) for it in items)
        out[f"budget/{split}"] = {"off_12000": a, "off_24000": b, "bundle_24000": c}
        print(f"    {split:8s} n={len(items)} OFF@12000 {a}  OFF@24000 {b}  BUNDLE@24000 {c}  "
              f"Δ(vs 24000 基线)={c-b:+d}  Δ(vs 12000 基线)={c-a:+d}")
    REPORT["r6"] = out


# ---------------------------------------------------------------- main
def sha256_of(rel: str) -> str:
    p = ROOT / rel
    if not p.exists():
        return "MISSING"
    return hashlib.sha256(p.read_bytes()).hexdigest().upper()


AUDITED = ["src/arkrag/judge.py", "scripts/eval_answer_quality.py",
           "data/goldset/gold-v2.json", "src/arkrag/quote_locate.py",
           "src/arkrag/bridge.py", "src/arkrag/pipeline.py",
           "src/arkrag/retrieve.py", "src/arkrag/answer.py"]

SECTIONS = {"r1": r1, "r2": r2, "r3": r3, "r4": r4, "r5": r5, "r6": r6}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="第六轮对抗审查复算（只读）")
    ap.add_argument("--only", default="", help="r1,r2,r3,r4,r5,r6 里选，逗号分隔；空 = 全部")
    ap.add_argument("--json", default="", help="把结果写到这个路径")
    ap.add_argument("--nobridge-recompute", action="store_true",
                    help="r1/r4 跳过逐题 find_bridge 复算（更快）")
    args = ap.parse_args(argv)

    want = [s.strip() for s in args.only.split(",") if s.strip()] or list(SECTIONS)
    world = World()

    print("被审文件 SHA-256：")
    digests = {}
    for rel in AUDITED:
        digests[rel] = sha256_of(rel)
        print(f"  {digests[rel]}  {rel}")

    for name in want:
        fn = SECTIONS.get(name)
        if fn is None:
            print(f"[skip] 未知节 {name}")
            continue
        if name in ("r1", "r4"):
            fn(world, nobridge=args.nobridge_recompute)
        else:
            fn(world)

    REPORT["sha256"] = digests
    if args.json:
        Path(args.json).write_text(json.dumps(REPORT, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print(f"\n已落盘 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
