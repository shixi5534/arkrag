#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cooc「信号作用范围」测算：真管线跑，只测范围规则，不改生产代码。

## 为什么不用名次代理

第一版用「gold 名次 ≤5 视作进上下文」做代理，自检直接失败：代理说基线 199 / 常数 202，
真跑是 189 / 195。根因是 `pipeline.ask_once` 里 `expand()` 会把图扩展单元也塞进上下文，
名次 6–8 的 gold 照样能进——`docs/a阶段_排序信号与预算方案.md` §8.6.1 那张转化率表里
「6–8 名 28 题只有 1 题进上下文」是**现状排序下**的观测，不是「名次决定上下文」的定律。
**代理不可用，只能真跑。** 这个坑值得留着：`hop_line` 那轮也用 rank≤5 代理（基线 193 vs 真跑 189，差 4 题），
当时没自检就报了数。

## 怎么做到「不改生产代码也能测范围规则」

`a_hop_suppress.py` 已经验证：进程内替换 `retriever.search`、把加成**折进 `hit["score"]`**，
再走**真** `pipeline.ask_once`，其 `any_gold`/`all_gold`/`covered`/`abstained`/`units`
与生产通路**逐题 0 不一致**（见 `docs/a_P2_round1_验收报告.md` §5）。所以这条路可信。

本脚本沿用同一手法，但加成规则从「权重 × 抑制函数」换成「作用范围」：

| 规则 | 规则内容 | 想验证的假设 |
|---|---|---|
| `none` | 不加成 | 基线 |
| `const0.1` | 所有命中块 +0.1（= 生产现状） | **自检必须逐题复现真跑** |
| `dens_le2/3/4` | 密度 > T 则整题不加成 | 密度门能否避开噪声 |
| `topk1/2/3` | 只抬原 rank 最靠前的 N 个命中块 | 「局部抬分」是否更好 |
| `inat_t2/3` | 加成按 `min(1, T/密度)` 反比缩放 | 反比缩放是否优于硬门 |
| `oracle_gold` | 只抬「命中且是 gold」的块（**线上拿不到**） | 范围规则的理论上界 |

`oracle_gold` 是纯上界参照，不是候选方案。

## 口径与纪律

- dev 350 道应答题（`split == dev`、非应拒、有 `gold_chunk_ids`）。**holdout 160 一次都没跑过，结论不许外推。**
- 四维都报：`any_gold` / `all_gold` / `covered` / `citations`，各报救/丢/净与题号（R2b' 通用规则）。
- 自检是硬要求且**零容差**：`const0.1` 必须与 `data/a_p2_round1.json` 的 `cooc0.1` **逐题完全一致**，
  否则退出码 4。抽样（`--limit`）时明确打印「未校验」，不作结论。

## 用法

    py -3 scripts/a_cooc_scope_probe.py                      # 全量 dev 350，11 规则
    py -3 scripts/a_cooc_scope_probe.py --limit 20 --rules none const0.1   # 冒烟
    py -3 scripts/a_cooc_scope_probe.py --rules const0.1 dens_le3 topk3 inat_t3

退出码：0 跑通 / 2 输入不可用 / 4 自检失败。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from arkrag import config, pipeline, query as query_mod, retrieve  # noqa: E402

DEFAULT_GENERATION = "g-20261103T132152Z-06a741"
DEFAULT_GOLD = ROOT / "data" / "goldset" / "gold-v2.json"
DEFAULT_BASELINE = ROOT / "data" / "a_p2_round1.json"
DEFAULT_OUT = ROOT / "data" / "a_cooc_scope_probe.json"

POOL = 200
TOP_K = 20
BASE_WEIGHT = 0.1


def rel(path) -> str:
    """机台路径一律转仓库相对——`tests/test_portable_paths.py` 判绝对路径。"""
    return config.rel_to_root(path)


def _git_commit() -> str | None:
    try:
        import subprocess
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
            capture_output=True, text=True, timeout=10,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def load_dev(gold_path: Path) -> list[dict]:
    """dev 调参集应答题。不做 `[:n]` 挑选（`ask.py --limit` 历史事故不许重演）。"""
    data = json.loads(gold_path.read_text(encoding="utf-8"))
    return [
        it for it in (data.get("items") or [])
        if it.get("split") == "dev"
        and not it.get("should_abstain")
        and (it.get("gold_chunk_ids") or [])
    ]


def deep_search(search_fn, qa, pool: int = POOL):
    """与生产 `search(top_k=20)` 同源的深池：压 `CANDIDATE_POOL_MULT` 让 `depth == pool`。

    必须传**原始** search 函数：补丁生效时 `retriever.search` 已被替换，走它会无限递归。
    """
    saved = retrieve.CANDIDATE_POOL_MULT
    retrieve.CANDIDATE_POOL_MULT = 1
    try:
        return search_fn(qa, top_k=pool)
    finally:
        retrieve.CANDIDATE_POOL_MULT = saved


def cooc_table(retriever, qa, hits) -> dict[str, int]:
    """与生产 `_cooc_boost` 同判据：任一行同时含两个端点规范名。

    自己实现一份而不调 `retriever._cooc_boost`，是为了脚本在「加之前 / 加之后」
    两份源码上都能跑（`a_rollback_check.py` 踩过这个坑）。
    """
    names: list[str] = []
    for cand in (qa.get("entity_candidates") or [])[:2]:
        ent = retriever.entities_by_id.get(str(cand.get("entity_id") or "")) or {}
        nm = ent.get("name") or cand.get("name")
        if nm:
            names.append(str(nm))
    if len(names) < 2:
        return {}
    out: dict[str, int] = {}
    for h in hits:
        cid = h["chunk_id"]
        text = str((retriever.chunks.get(cid) or {}).get("text") or "")
        for line in text.splitlines():
            if all(n in line for n in names):
                out[cid] = 1
                break
    return out


def strict_table(retriever, qa, hits) -> dict[str, int]:
    """cooc 的**收紧版**：同一行还要含一个「两端点的共同邻居」名。

    这是 hop_line 与 cooc 的交集，但性质不同：hop_line 是**独立加分通道**（已被否决，
    因为它对「任一端点 + 任一共同邻居」就点火，命中块数 20/20 全亮）；这里它是
    **cooc 通道的作用范围闸门**——先要求两端点同行（cooc 成立），再要求该行有中间实体。
    单个变量：只改 cooc 的适用范围，不引入新通道。
    """
    cands = qa.get("entity_candidates") or []
    names: list[str] = []
    for cand in cands[:2]:
        ent = retriever.entities_by_id.get(str(cand.get("entity_id") or "")) or {}
        nm = ent.get("name") or cand.get("name")
        if nm:
            names.append(str(nm))
    if len(names) < 2:
        return {}
    a = str(cands[0].get("entity_id") or "")
    b = str(cands[1].get("entity_id") or "")
    adj = retriever.adjacency
    common: set[str] = set()
    if a and b:
        def nb(eid: str) -> set[str]:
            rec = adj.get(eid) or {}
            s = {str(x.get("neighbor_id") or "") for x in (rec.get("out") or [])}
            s |= {str(x.get("neighbor_id") or "") for x in (rec.get("in") or [])}
            s.discard("")
            s.discard(eid)
            return s
        common = nb(a) & nb(b)
    common_names = [str(retriever.display_name(e) or "") for e in common]
    common_names = [x for x in common_names if x]
    out: dict[str, int] = {}
    for h in hits:
        cid = h["chunk_id"]
        text = str((retriever.chunks.get(cid) or {}).get("text") or "")
        for line in text.splitlines():
            if all(n in line for n in names) and any(m in line for m in common_names):
                out[cid] = 1
                break
    return out


def boosts_for(rule: str, hits, cooc: dict[str, int], gold: set[str],
               strict: dict[str, int] | None = None) -> tuple[dict[str, float], dict]:
    """按规则生成逐块加成。返回 (加成表, 诊断信息)。"""
    n = len(cooc)
    order = [h["chunk_id"] for h in hits]
    rank = {cid: i + 1 for i, cid in enumerate(order)}
    cooc_by_rank = sorted(cooc, key=lambda c: rank.get(c, 10 ** 9))
    diag = {"cooc_hits_pool": n}
    if strict is not None:
        diag["strict_hits_pool"] = len(strict)

    if rule == "none":
        return {}, diag
    if rule == "const0.1":
        return {c: BASE_WEIGHT for c in cooc}, diag
    if rule.startswith("dens_le"):
        t = float(rule[len("dens_le"):])
        # 密度用「上下文容量」作分母。真实 units 只有真跑后才知道，而加成必须在
        # 检索阶段就定——所以这里用生产预算下实测的容量中位 5 作常数分母，
        # 明确写进 diag 供复核（这不是 units，是容量常数）。
        dens = n / CAPACITY
        diag.update({"density_vs_capacity": round(dens, 4), "capacity_const": CAPACITY})
        return ({c: BASE_WEIGHT for c in cooc} if dens <= t else {}), diag
    if rule.startswith("topk"):
        k = int(rule[len("topk"):])
        diag["topk"] = k
        return {c: BASE_WEIGHT for c in cooc_by_rank[:k]}, diag
    if rule.startswith("inat_t"):
        t = float(rule[len("inat_t"):])
        dens = n / CAPACITY
        scale = min(1.0, t / dens) if dens > 0 else 0.0
        diag.update({"density_vs_capacity": round(dens, 4), "scale": round(scale, 4)})
        return {c: BASE_WEIGHT * scale for c in cooc}, diag
    if rule.startswith("strict"):
        # 收紧版命中表替代 cooc 命中表，其余同 const0.1——单变量：只改作用范围判据。
        # 权重沿用 0.1（不是新变量）；若签名里没传 strict 说明调用方漏了，直接报错而不是静默退化成 const。
        if strict is None:
            raise SystemExit("strict 规则需要 strict 命中表")
        return {c: BASE_WEIGHT for c in strict}, diag
    if rule == "oracle_gold":
        return {c: BASE_WEIGHT for c in cooc if c in gold}, diag
    raise SystemExit(f"未知规则：{rule}")


# 上下文容量常数：6000 token 预算 ÷ 单元 tokens 中位 ~1262 ≈ 4.8，实测单元数中位 5。
# 用于「密度」定义的分母（加成阶段拿不到真 units）。
CAPACITY = 5

RULES = [
    "none", "const0.1", "strict0.1",
    "dens_le2", "dens_le3", "dens_le4",
    "topk1", "topk2", "topk3",
    "inat_t2", "inat_t3",
    "oracle_gold",
]


def coverage_of(answer: dict, reqs: list[dict]) -> bool:
    """与 `scripts/ask.py:283-298` 同口径：required 全在引用里且 evidence 是子串。"""
    cites = answer.get("citations") or []
    for req in reqs:
        cid = req.get("chunk_id")
        ev = req.get("evidence") or ""
        if not any(c.get("chunk_id") == cid and ev in (c.get("evidence") or "") for c in cites):
            return False
    return bool(reqs)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--generation", default=DEFAULT_GENERATION)
    ap.add_argument("--gold", default=str(DEFAULT_GOLD))
    ap.add_argument("--baseline-json", default=str(DEFAULT_BASELINE))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rules", nargs="*", default=None)
    ap.add_argument("--out-json", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)

    gold_path = Path(args.gold)
    if not gold_path.exists():
        print(f"[2] gold 不存在：{rel(gold_path)}")
        return 2
    items = load_dev(gold_path)
    if args.limit:
        items = items[: args.limit]
    if not items:
        print("[2] dev 应答题为空")
        return 2
    rules = args.rules or RULES

    print(f"载入世代 {args.generation} …")
    loaded = pipeline.load_generation(args.generation)
    pipe = pipeline.build_pipeline(loaded, top_k=TOP_K, graph_depth=1)
    rt = pipe["retriever"]
    original_search = rt.search

    # 每题深池与 cooc 表只算一次，各规则复用（深池 ~140 ms + 200 块正文扫描，不缓存太慢）
    pool_cache: dict[str, dict] = {}
    for n, item in enumerate(items, 1):
        qid = item["question_id"]
        qa = query_mod.analyze_query(
            item["question"], entities=getattr(rt, "entities", None),
            aliases=getattr(rt, "alias_index", None),
            vocab=getattr(rt, "vocab", None),
            context=getattr(rt, "query_context", None),
        )
        payload = deep_search(original_search, qa, POOL)
        hits = payload.get("hits") or []
        pool_cache[qid] = {
            "qa": qa,
            "payload": payload,
            "hits": hits,
            "cooc": cooc_table(rt, qa, hits) if hits else {},
            "strict": strict_table(rt, qa, hits) if hits else {},
            "gold": {str(g) for g in (item.get("gold_chunk_ids") or [])},
        }
        if n % 50 == 0:
            print(f"  已算 {n}/{len(items)} 题深池与信号")

    per_rule: dict[str, dict] = {}
    for rule in rules:
        rows = []
        for item in items:
            qid = item["question_id"]
            c = pool_cache[qid]
            hits = c["hits"]
            gold = c["gold"]
            if not hits:
                rows.append({"question_id": qid, "category": item.get("category"),
                             "any_gold": False, "all_gold": False, "covered": False,
                             "citations": 0, "units": 0, "n_gold": len(gold)})
                continue
            boosts, diag = boosts_for(rule, hits, c["cooc"], gold, c.get("strict"))
            hit_by_id = {h["chunk_id"]: h for h in hits}
            order = [h["chunk_id"] for h in hits]
            rank = {cid: i + 1 for i, cid in enumerate(order)}
            # 关键：深池的 score 里**已经含有生产常量的 cooc 加成**（retrieve.py 现为 0.1）。
            # 不先扣掉就套规则，等于在「已加成」的基础上再加一次——本脚本第一版就是这么
            # 错的：`none` 跑出 195/147/63（= 生产 cooc0.1 的数），自检 15 处不一致才发现。
            # `cooc_boost` 字段由生产 search() 写入，可直接反推未加成前的分。
            raw = {cid: float(hit_by_id[cid]["score"]) - float(hit_by_id[cid].get("cooc_boost") or 0.0)
                   for cid in order}
            new_order = sorted(order, key=lambda cid: (-(raw[cid] + boosts.get(cid, 0.0)), rank[cid]))
            new_hits = []
            for cid in new_order[:TOP_K]:
                h2 = dict(hit_by_id[cid])
                # 加成折进 score：build_context 只读 score 不读顺序（retrieve.py:850-853）
                h2["score"] = raw[cid] + boosts.get(cid, 0.0)
                new_hits.append(h2)
            stats = dict(c["payload"].get("stats") or {})
            stats["top_score"] = float(new_hits[0]["score"]) if new_hits else 0.0

            def patched(qa, top_k=None, _nh=new_hits, _st=stats):
                return {"hits": list(_nh), "stats": dict(_st)}

            rt.search = patched
            try:
                out = pipeline.ask_once(pipe, item["question"])
            finally:
                rt.search = original_search

            ctx_units = [u.get("chunk_id") for u in (out["context"].get("units") or [])]
            ans = out["answer"]
            any_g = any(g in ctx_units for g in gold)
            all_g = bool(gold) and all(g in ctx_units for g in gold)
            cov = coverage_of(ans, item.get("required_citations") or [])
            rows.append({
                "question_id": qid,
                "category": item.get("category"),
                "origin": item.get("origin"),
                "n_gold": len(gold),
                "any_gold": any_g,
                "all_gold": all_g,
                "covered": cov,
                "citations": len(ans.get("citations") or []),
                "units": len(ctx_units),
                "abstained": bool(ans.get("abstained")),
                "gold_in_ctx": [g for g in gold if g in ctx_units],
                **diag,
            })
        agg = {k: sum(1 for r in rows if r.get(k)) for k in ("any_gold", "all_gold", "covered", "abstained")}
        agg["citations"] = sum(r.get("citations") or 0 for r in rows)
        per_rule[rule] = {"agg": agg, "rows": rows}
        print(f"  {rule:14s} any={agg['any_gold']:3d} all={agg['all_gold']:3d} "
              f"cov={agg['covered']:3d} cit={agg['citations']:4d} abst={agg['abstained']:2d}")

    # ---- 自检 1：const0.1 与真跑基线逐题一致（零容差）----
    baseline_path = Path(args.baseline_json)
    check = {"baseline_json": rel(baseline_path), "checked": False}
    if "const0.1" in per_rule and baseline_path.exists() and not args.limit:
        base = json.loads(baseline_path.read_text(encoding="utf-8"))
        ref = {r["question_id"]: r for r in base["per_question"]["cooc0.1"]}
        mine = {r["question_id"]: r for r in per_rule["const0.1"]["rows"]}
        diffs = []
        for qid, r in mine.items():
            o = ref.get(qid)
            if not o:
                diffs.append((qid, "missing"))
                continue
            for k in ("any_gold", "all_gold", "covered", "units"):
                if r.get(k) != o.get(k):
                    diffs.append((qid, k, r.get(k), o.get(k)))
        check.update({"checked": True, "n_diff": len(diffs), "diffs": diffs[:10]})
        print(f"\n[自检1] const0.1 vs 真跑 cooc0.1：逐题差异 {len(diffs)} 处")
        if diffs:
            print("[4] 补丁通路与生产通路不一致，本脚本所有数字不能用")
            return 4
    else:
        print("\n[自检1] 未跑 const0.1 或抽样 → 未校验补丁通路，本次数字不可作结论")

    # ---- 自检 2：none 规则应与真跑基线一致 ----
    if "none" in per_rule and baseline_path.exists() and not args.limit:
        base = json.loads(baseline_path.read_text(encoding="utf-8"))
        ref = {r["question_id"]: r for r in base["per_question"]["cooc0"]}
        mine = {r["question_id"]: r for r in per_rule["none"]["rows"]}
        diffs = [q for q, r in mine.items()
                 if any(r.get(k) != (ref.get(q) or {}).get(k) for k in ("any_gold", "all_gold", "covered", "units"))]
        check["none_diff"] = len(diffs)
        print(f"[自检2] none vs 真跑基线：逐题差异 {len(diffs)} 处")
        if diffs:
            print("[4] 基线复现失败")
            return 4

    # ---- 四维增量（R2b' 通用规则：各报救/丢/净 + 题号）----
    base_rows = {r["question_id"]: r for r in per_rule["none"]["rows"]}
    print("\n四维增量（相对 none）：")
    print("规则            any救 丢 净 | all救 丢 净 | cov救 丢 净 | citΔ")
    summary = {}
    for rule in rules:
        if rule == "none":
            continue
        rr = {r["question_id"]: r for r in per_rule[rule]["rows"]}
        dim = {}
        for k in ("any_gold", "all_gold", "covered"):
            gain = [q for q in rr if not base_rows[q].get(k) and rr[q].get(k)]
            lose = [q for q in rr if base_rows[q].get(k) and not rr[q].get(k)]
            dim[k] = {"gain": len(gain), "lose": len(lose), "net": len(gain) - len(lose),
                      "gain_ids": gain, "lose_ids": lose}
        dcit = per_rule[rule]["agg"]["citations"] - per_rule["none"]["agg"]["citations"]
        summary[rule] = {"dims": dim, "d_citations": dcit}
        d = dim
        print(f"  {rule:12s} {d['any_gold']['gain']:4d} {d['any_gold']['lose']:3d} {d['any_gold']['net']:+3d} |"
              f" {d['all_gold']['gain']:4d} {d['all_gold']['lose']:3d} {d['all_gold']['net']:+3d} |"
              f" {d['covered']['gain']:4d} {d['covered']['lose']:3d} {d['covered']['net']:+3d} | {dcit:+4d}")

    out = {
        "schema": "a-cooc-scope-probe-2",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "command": "py -3 scripts/a_cooc_scope_probe.py",
        "generation": args.generation,
        "gold_set": rel(gold_path),
        "baseline_json": rel(baseline_path),
        "split_policy": ("dev-only-pre-split-diagnostic：仅 dev 调参集 350 道应答题；"
                         "结论不得外推到 new-holdout 160（一次都未观测）。"),
        "n_questions": len(items),
        "pool": POOL, "top_k": TOP_K, "base_weight": BASE_WEIGHT, "capacity_const": CAPACITY,
        "method": ("进程内替换 retriever.search，加成折进 hit['score']，再走真 pipeline.ask_once；"
                   "自检要求 const0.1 与 data/a_p2_round1.json 的 cooc0.1 逐题一致。"),
        "self_check": check,
        "summary": summary,
        "per_rule": per_rule,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写出 {rel(out_path)}  ({out_path.stat().st_size:,} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
