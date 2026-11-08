r"""第七轮审查脚本：cross_chunk「章节收窄」（`src/arkrag/chapter_narrow.py`）的独立复算。

只读：不修改任何 src/ 或评测产物；只读取世代、goldset 与生产模块。
用法：
    $env:PYTHONPATH="E:\图rag\src"; py -3 scripts\audit_round7.py
    py -3 scripts\audit_round7.py --only r1,r3,r5      # 选节
    py -3 scripts\audit_round7.py --json out.json      # 落盘

节：
 r1 复现 bundle vs bundle+chapter_narrow（口径 A = answer.verify_citations 后判）+ 配对 bootstrap
 r2 gold ⊆ allow / 只删不加的定理检验
 r3 对照组（true_ent / rand_ent / rand_ent_plus_gold / fake_allow / name_rank / oracle）
 r4 判据退化：pass ↔ 输出==gold 交叉表
 r5 零检索「答案 = allow」分 + 总分上界
 r6 门控精度（全 640 题）+ ∩ctx 空操作检验 + 披露数（chunk 级 / 章节级 / 两种分母）

退出码：0 = 全部核对一致；1 = 与台账基线不符（可作 CI 卡口）。
"""
from __future__ import annotations

import sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, r"E:\图rag\src")

import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path

from arkrag import answer as answer_mod
from arkrag import judge as J
from arkrag import pipeline as P

try:                                    # 被审模块可能已被移除（本轮裁决之后确实被删）
    from arkrag import chapter_narrow as CN
    _CN_AVAILABLE = True
except ImportError:                     # pragma: no cover
    CN = None                           # type: ignore[assignment]
    _CN_AVAILABLE = False

ROOT = Path(r"E:\图rag")
GEN = "g-20261103T132152Z-06a741"
SPLIT_RE = re.compile(r"[、,，/／\s]+")

BUNDLE = {"quote_locate": True, "bridge": True,
          "graph_reserve_tokens": 12000, "bridge_search_top_k": 40}
NARROW = dict(BUNDLE, chapter_narrow=True)

# 台账基线（审查方独立复算值；改动须同步）
EXPECT = {
    "dev": {"n": 285, "bundle": 228, "narrow": 235},
    "holdout": {"n": 195, "bundle": 152, "narrow": 157},
    "clean": {"n": 120, "bundle": 97, "narrow": 98},
}

WATCH = [
    ("src/arkrag/judge.py", {"D84EF154E44610E1AC2A24E96967AE2A45890A22D3ABB193E773BA16B50C3468"}),
    ("scripts/eval_answer_quality.py",
     {"D3A19ED99BB4E25C80D84EDE77DEC65055210B2EFE04E8FA0E97A8626C6D3973"}),
    ("data/goldset/gold-v2.json",
     {"069375F8E355E4B9C3B4253E14959532F87E05BC129D5A7A6DA39E07F357A108"}),
    ("src/arkrag/chapter_narrow.py",
     {"37539D246362711D9ADF3EB3B63F814D258B1654E0BED76ECE3273231969BBA9"}),
    ("src/arkrag/quote_locate.py",
     {"AE007EC18BC78175FEA9B4F8046C384915A8A2B1B7D03403D8EC80FA4DE06525"}),
    ("src/arkrag/bridge.py",
     {"5BC426CBBE12FEB840856D183F9B5702EEC9AB1E8E4EB49E00D35624FB4CD70C"}),
    # pipeline.py 在本轮审查期间被改过一次（08:11:20，4B2C1C6C… → D3803D18…）。
    # 两个 hash 下的 §r1 读数逐格相同（235/157/98），但按红线 31 必须并列声明。
    ("src/arkrag/pipeline.py",
     {"4B2C1C6C5CBA247387F8E45DFA5ACC83D9C3B33A6603C8E58517726F8C97D5AA",
      "D3803D188D446526D2C031B87BCDB1F89731CC219A96E19D8923A32271812F07"}),
    ("src/arkrag/retrieve.py",
     {"6BBFA4165874FEF76BD52E30194B455E1E0D9221F3BE367A9923F472F315E27D"}),
    ("src/arkrag/answer.py",
     {"A764C48708CB525D915EC09FA43C4C51068CA8790F0C62DD3793F4C758D06F9A"}),
]


# --------------------------------------------------------------------------- 载入

def chap_set(s) -> set[str]:
    return {c.strip() for c in SPLIT_RE.split(str(s or "")) if c.strip()}


def load():
    L = P.load_generation(GEN)
    chunks = L["chunks"]
    graph = L["graph"]
    per_ch = collections.defaultdict(list)
    for cid, rec in chunks.items():
        per_ch[str(rec.get("chapter") or "")].append(str(rec.get("text") or ""))
    chapter_texts = {k: "\n".join(v) for k, v in per_ch.items()}

    items = json.loads((ROOT / "data/goldset/gold-v2.json").read_text("utf-8"))["items"]
    v1_path = ROOT / "data/goldset/gold-v1.json"
    v1 = set()
    if v1_path.exists():
        v1 = {it["question_id"] for it in
              json.loads(v1_path.read_text("utf-8"))["items"]}

    def pick(split):
        return [it for it in items
                if it.get("split") == split and it.get("gold_chunk_ids")]

    holdout = pick("holdout")
    pools = {
        "dev": pick("dev"),
        "holdout": holdout,
        "clean": [it for it in holdout if it["question_id"] not in v1],
    }
    p = P.build_pipeline(L, top_k=20, graph_depth=1)
    p["generation_id"] = GEN
    p["entities"] = graph["entities"]
    p["retriever"].budget["max_context_tokens"] = 24000
    p["answerer"].max_claims = 8
    return dict(L=L, chunks=chunks, graph=graph, chapter_texts=chapter_texts,
                items=items, pools=pools, pipe=p)


def run(ctx, it, opts):
    r = P.ask_once(ctx["pipe"], it["question"], options=opts)
    ans = dict(r["answer"] or {})
    cits = ans.get("citations") or []
    ver = (answer_mod.verify_citations(ans, ctx["chunks"]) if cits
           else {"answer": str(ans.get("answer") or ""), "citations": [], "problems": []})
    txt = str(ver.get("answer") or "")
    jr = J.judge_answer(it["category"], txt, it, question=it.get("question"),
                        chapter_texts=ctx["chapter_texts"])
    return {"pass": jr.get("pass") is True, "reason": jr.get("reason"),
            "text": txt, "ans": ans,
            "units": (r["context"] or {}).get("units") or []}


def judge(ctx, it, text):
    jr = J.judge_answer(it["category"], text, it, question=it.get("question"),
                        chapter_texts=ctx["chapter_texts"])
    return jr.get("pass") is True, jr.get("reason")


def boot(a, b, n=20000, seed=7):
    d = [y - x for x, y in zip(a, b)]
    rng = random.Random(seed)
    idx = list(range(len(d)))
    sums = sorted(sum(d[rng.choice(idx)] for _ in idx) for _ in range(n))
    return sum(d), sums[int(n * .025)], sums[int(n * .975)], sum(1 for s in sums if s <= 0) / n


def cxs(ctx, split):
    return [it for it in ctx["pools"][split] if it["category"] == "cross_chunk"]


def score_items(ctx, split):
    return [it for it in ctx["pools"][split] if it["category"] in set(J.SCORABLE_CATEGORIES)]


# --------------------------------------------------------------------------- 节

def r1(ctx, out):
    print("=" * 100)
    print("§r1 复现 bundle vs bundle+chapter_narrow（口径 A = verify_citations 后判，红线 33）")
    print("=" * 100)
    ok = True
    for split in ("dev", "holdout", "clean"):
        items = score_items(ctx, split)
        bv, nv = [], []
        per_b, per_n = collections.Counter(), collections.Counter()
        for it in items:
            x = run(ctx, it, BUNDLE)
            y = run(ctx, it, NARROW)
            bv.append(int(x["pass"])); nv.append(int(y["pass"]))
            per_b[it["category"]] += x["pass"]; per_n[it["category"]] += y["pass"]
        d, lo, hi, p = boot(bv, nv)
        e = EXPECT[split]
        same = (len(items) == e["n"] and sum(bv) == e["bundle"] and sum(nv) == e["narrow"])
        ok &= same
        print(f"  {split:8s} n={len(items)}  bundle {sum(bv)}/{len(items)}"
              f"={sum(bv)/len(items)*100:.2f}%  +narrow {sum(nv)}"
              f"={sum(nv)/len(items)*100:.2f}%  Δ={d:+d} CI=[{lo:+d},{hi:+d}] p={p:.4f} "
              f"{'PASS' if d >= 1 and lo > 0 and p < .05 else 'FAIL'}")
        print(f"           逐类 bundle " + " ".join(f"{k[:3]}:{per_b[k]}" for k in sorted(per_b)))
        print(f"           逐类 +nar   " + " ".join(f"{k[:3]}:{per_n[k]}" for k in sorted(per_n)))
        print(f"           台账核对 {'一致' if same else 'DIFF: 期望 n=%d bundle=%d narrow=%d' % (e['n'], e['bundle'], e['narrow'])}")
        out["r1"][split] = {"n": len(items), "bundle": sum(bv), "narrow": sum(nv),
                            "delta": d, "ci": [lo, hi], "p": p,
                            "per_category_bundle": dict(per_b),
                            "per_category_narrow": dict(per_n), "matches_ledger": same}
    return ok


def r2(ctx, out):
    print()
    print("=" * 100)
    print("§r2 gold ⊆ allow（红线 36）：只删不加 ⇒ 掉分是不可能的（定理）")
    print("=" * 100)
    for split in ("dev", "holdout", "clean"):
        sub = eq = n = 0
        sizes = []
        for it in cxs(ctx, split):
            gold = chap_set(it.get("answer"))
            allow = CN.allowed_chapters(it["question"], ctx["graph"], ctx["chunks"])
            n += 1
            sub += gold <= allow
            eq += gold == allow
            sizes.append(len(allow))
        sizes.sort()
        print(f"  {split:8s} n={n}  gold⊆allow {sub}/{n}={sub/n*100:.1f}%  "
              f"gold==allow {eq}/{n}  |allow| p50={sizes[n//2]} max={sizes[-1]}")
        out["r2"][split] = {"n": n, "gold_subset_allow": sub, "gold_eq_allow": eq}
    viol = checked = 0
    for it in cxs(ctx, "dev"):
        a = run(ctx, it, BUNDLE)
        if not a["pass"]:
            continue
        got = chap_set(a["ans"].get("answer"))
        gold = chap_set(it.get("answer"))
        extras = sorted(got - gold)
        if not extras:
            continue
        checked += 1
        pj, _ = judge(ctx, it, "、".join(sorted(got - {extras[0]})))
        viol += (not pj)
    print(f"  定理检验（dev）：删 1 个非 gold 章，检查 {checked} 题，掉分 {viol} 次 "
          f"⇒ {'不降成立' if viol == 0 else '不成立'}")
    out["r2"]["theorem_dev"] = {"checked": checked, "violations": viol}
    return True


def r3(ctx, out):
    print()
    print("=" * 100)
    print("§r3 对照组（红线 37）：把 gold 塞进过滤器后，来源是否还有信息")
    print("=" * 100)
    rng = random.Random(20261108)
    ents = ctx["graph"]["entities"]
    all_ch = set(ctx["chapter_texts"])
    for split in ("dev", "holdout", "clean"):
        acc = collections.Counter()
        n = 0
        for it in cxs(ctx, split):
            a = run(ctx, it, BUNDLE)
            units = a["units"]
            ctx_ch = {str(u.get("chapter") or "").strip() for u in units}
            ctx_ch.discard("")
            allow = CN.allowed_chapters(it["question"], ctx["graph"], ctx["chunks"])
            gold = chap_set(it.get("answer"))
            n += 1
            acc["base"] += a["pass"]

            def _try(aset, key):
                if not aset:
                    acc[key] += a["pass"]
                    return
                kept = {c for c in chap_set(a["ans"].get("answer"))
                        if c in aset and c in ctx_ch} or chap_set(a["ans"].get("answer"))
                acc[key] += judge(ctx, it, "、".join(sorted(kept)))[0]

            _try(allow, "true_ent")
            _try(all_ch, "noop")
            pe = ents[rng.randrange(len(ents))]
            re_ = CN._chapter_of_chunks(ctx["chunks"], pe.get("chunks") or [])
            _try(set(re_), "rand_ent")
            _try(set(re_) | gold, "rand_ent_plus_gold")
            _try(set(gold) | set(rng.sample(sorted(all_ch - gold),
                                            min(max(0, len(allow) - len(gold)),
                                                len(all_ch - gold)))),
                 "fake_allow")
            _try(set(gold), "oracle_gold")
        print(f"  {split:8s} n={n} " + "  ".join(
            f"{k}={acc[k]}" for k in ("base", "true_ent", "rand_ent", "rand_ent_plus_gold",
                                     "fake_allow", "noop", "oracle_gold")))
        marginal = (acc["true_ent"] - acc["rand_ent_plus_gold"])
        print(f"           ⇒ entity['chunks'] 边际贡献 = true_ent − rand_ent_plus_gold = "
              f"{acc['true_ent']} − {acc['rand_ent_plus_gold']} = {marginal:+d}")
        out["r3"][split] = dict(acc)
    return True


def r4(ctx, out):
    print()
    print("=" * 100)
    print("§r4 判据退化（红线 38）：收窄后 pass ↔ 输出==gold")
    print("=" * 100)
    for split in ("dev", "holdout", "clean"):
        st = collections.Counter()
        for it in cxs(ctx, split):
            a = run(ctx, it, BUNDLE)
            units = a["units"]
            ctx_ch = {str(u.get("chapter") or "").strip() for u in units}
            ctx_ch.discard("")
            got = chap_set(a["ans"].get("answer"))
            allow = CN.allowed_chapters(it["question"], ctx["graph"], ctx["chunks"])
            kept = CN.narrow(got, allow=allow, ctx_chapters=ctx_ch)
            pj, _ = judge(ctx, it, "、".join(sorted(kept)))
            eq = (kept == chap_set(it.get("answer")))
            st["n"] += 1; st["pass"] += pj; st["eq"] += eq
            st["pass_and_eq"] += (pj and eq)
            st["false_pos"] += (pj and not eq)
            st["false_neg"] += ((not pj) and eq)
        print(f"  {split:8s} n={st['n']} pass={st['pass']} out==gold={st['eq']} "
              f"pass&eq={st['pass_and_eq']} 假正={st['false_pos']} 假负={st['false_neg']}")
        out["r4"][split] = dict(st)
    return True


def r5(ctx, out):
    print()
    print("=" * 100)
    print("§r5 零检索「答案 = allow」（红线 34）")
    print("=" * 100)
    for split in ("dev", "holdout", "clean"):
        b = nz = d = 0
        for it in cxs(ctx, split):
            b += run(ctx, it, BUNDLE)["pass"]
            nz += run(ctx, it, NARROW)["pass"]
            allow = CN.allowed_chapters(it["question"], ctx["graph"], ctx["chunks"])
            if allow:
                d += judge(ctx, it, "、".join(sorted(allow)))[0]
        print(f"  {split:8s} n={len(cxs(ctx, split))} bundle={b} 收窄后={nz} "
              f"**答案=allow(零检索)={d}**")
        out["r5"][split] = {"bundle": b, "narrow": nz, "answer_is_allow": d,
                            "n": len(cxs(ctx, split))}
    return True


def r6(ctx, out):
    print()
    print("=" * 100)
    print("§r6 门控（红线 32）+ ∩ctx 空操作 + 披露数（红线 39/40）")
    print("=" * 100)
    print(f"  CHAPTER_TRIGGERS = {answer_mod.CHAPTER_TRIGGERS}")
    hit = collections.Counter()
    for it in ctx["items"]:
        if any(t in it["question"] for t in answer_mod.CHAPTER_TRIGGERS):
            hit[it["category"]] += 1
    print(f"  全 640 题命中 {sum(hit.values())}  逐类 {dict(hit)}  "
          f"⇒ {'零误伤零漏' if dict(hit) == {'cross_chunk': 64} else '有误伤或漏'}")
    out["r6"]["gate"] = dict(hit)

    for split in ("dev", "holdout", "clean"):
        vio = removed = 0
        for it in cxs(ctx, split):
            a = run(ctx, it, BUNDLE)
            got = chap_set(a["ans"].get("answer"))
            ctx_ch = {str(u.get("chapter") or "").strip() for u in a["units"]}
            ctx_ch.discard("")
            if not got <= ctx_ch:
                vio += 1
                removed += len(got - ctx_ch)
        print(f"  {split:8s} got⊄ctx 的题 {vio}  被 ∩ctx 删掉的章节 {removed} "
              f"⇒ {'∩ctx 恒为空操作' if removed == 0 else '∩ctx 有实际作用'}")
        out["r6"][f"ctx_noop_{split}"] = {"violations": vio, "chapters_removed": removed}

    by_name = collections.defaultdict(list)
    for e in ctx["graph"]["entities"]:
        by_name[str(e.get("name") or "").strip()].append(e)
    for group, splits in (("dev+holdout", ("dev", "holdout")),
                          ("dev+holdout+clean(/80)", ("dev", "holdout", "clean"))):
        c = collections.Counter()
        esz, msz = [], []
        for split in splits:
            for it in cxs(ctx, split):
                names = [x.strip() for x in re.findall(r"「([^」]{1,30})」", it["question"])
                         if x.strip()]
                eset = set()
                for nm in names:
                    for e in by_name.get(nm, []):
                        eset |= {str(x) for x in (e.get("chunks") or [])}
                if not eset:
                    continue
                mset = {cid for cid, rec in ctx["chunks"].items()
                        if any(nm in str(rec.get("text") or "") for nm in names)}
                ecap = {str(ctx["chunks"][x].get("chapter") or "")
                        for x in eset if x in ctx["chunks"]}
                mcap = {str(ctx["chunks"][x].get("chapter") or "")
                        for x in mset if x in ctx["chunks"]}
                gcap = chap_set(it.get("answer"))
                gcid = {str(x) for x in it["gold_chunk_ids"]}
                c["n"] += 1
                c["E_chunks⊆M_chunks"] += eset <= mset
                c["gold_chunks⊆E_chunks"] += gcid <= eset
                c["gold_chunks==E_chunks"] += gcid == eset
                c["E_chapters==gold"] += ecap == gcap
                c["gold⊆E_chapters"] += gcap <= ecap
                c["M_chapters==gold"] += mcap == gcap
                esz.append(len(eset)); msz.append(len(mset))
        n = c["n"]
        print(f"  [{group}] n={n}")
        for k in ("E_chunks⊆M_chunks", "gold_chunks⊆E_chunks", "gold_chunks==E_chunks",
                  "E_chapters==gold", "gold⊆E_chapters", "M_chapters==gold"):
            print(f"      {k:24s} {c[k]}/{n} = {c[k]/n*100:.1f}%")
        print(f"      平均|E|={sum(esz)/n:.1f} p50={sorted(esz)[n//2]}   "
              f"平均|M|={sum(msz)/n:.1f} p50={sorted(msz)[n//2]}")
        out["r6"][group] = {"n": n, **{k: c[k] for k in c}}
    return True


# --------------------------------------------------------------------------- 主

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="r1,r2,r3,r4,r5,r6")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    want = {s.strip() for s in args.only.split(",") if s.strip()}

    print("被审文件 SHA-256（与台账比对）")
    for rel, exps in WATCH:
        fp = ROOT / rel
        got = hashlib.sha256(fp.read_bytes()).hexdigest().upper() if fp.exists() else "MISSING"
        mark = "" if got in exps else "   ← 与台账不一致"
        print(f"  {got}  {rel}{mark}")
    print()

    ctx = load()
    out = {"r1": {}, "r2": {}, "r3": {}, "r4": {}, "r5": {}, "r6": {}}
    if not _CN_AVAILABLE:
        print("[SKIP] `arkrag.chapter_narrow` 不存在（已被移除）。")
        print("       本轮裁决：该模块判注水，已按结论删除；脚本保留以便回归时复查。")
        print("       需要复现历史数字时，请从裁决前的提交/备份恢复该模块后再跑。")
        if args.json:
            Path(args.json).write_text(
                json.dumps({"skipped": "chapter_narrow_missing"}, ensure_ascii=False, indent=2),
                "utf-8")
        return 2

    ok = True
    for name, fn in (("r1", r1), ("r2", r2), ("r3", r3),
                     ("r4", r4), ("r5", r5), ("r6", r6)):
        if name in want:
            ok &= bool(fn(ctx, out))
    if args.json:
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
        print(f"\n已落盘 {args.json}")
    print("\n结论：" + ("全部核对一致" if ok else "存在 DIFF，见上表"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
