#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""排序侧「洗牌量」测算：回答「新规则会不会又变成 hop_line 那种洗牌」。

## 为什么单独测这个

`hop0.5` 的教训：R@20 涨了 3.03pt（评测台上「命中」），但真进上下文净 0，
因为抬起来的块互相抵消、只把题从第 60 名挪到第 15 名——**名次动了，下游没动**。

判断一个新规则会不会重演这件事，有两个纯算术量（不需要任何代理假设）：

1. **top-20 集合变动率**：有多少题的 top-20 成员集合被改变。只改集合之外的名次
   （例如 60 名挪到 15 名而 20 名以内成员不变）对上下文毫无意义。
2. **gold 跨过容量线的题数**：有多少题的某块 gold 从容量外挪进容量内、或反之。
   这是「有希望真正进上下文」的上界——**注意它只是上界**：`pipeline.ask_once` 里
   `expand()` 会把图扩展单元也塞进来，名次 6–8 的 gold 有时也能进（实测
   `q-mh-000044` 之类），所以真进上下文只能由 `a_cooc_scope_probe.py` 真跑给出。

把「跨线题数」与真跑「实际进上下文题数」并排，就能看出 expand 效应有多大。

## 用法

    py -3 scripts/a_rank_churn.py
    py -3 scripts/a_rank_churn.py --rules const0.1 dens_le3 topk3 oracle_gold --out-json data/a_rank_churn.json

退出码：0 跑通 / 2 输入不可用 / 4 自检失败（基线行必须与深池原顺序逐位相同）。
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

from a_cooc_scope_probe import (  # noqa: E402
    BASE_WEIGHT, CAPACITY, DEFAULT_GENERATION, DEFAULT_GOLD, POOL, TOP_K,
    boosts_for, cooc_table, deep_search, load_dev, rel, strict_table,
)

DEFAULT_OUT = ROOT / "data" / "a_rank_churn.json"
DEFAULT_PROBE = ROOT / "data" / "a_cooc_scope_probe.json"

# 容量线：证据单元数中位 5（`data/lead_rank_context_ceiling.json` 的 `3_单元数.中位`）。
# 名次 ≤ 5 的粗口径。**只是上界的口径，不是定律**——理由见模块 docstring。
LINE = 5


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


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--generation", default=DEFAULT_GENERATION)
    ap.add_argument("--gold", default=str(DEFAULT_GOLD))
    ap.add_argument("--probe-json", default=str(DEFAULT_PROBE),
                    help="a_cooc_scope_probe.json，用来并排真跑进上下文题数")
    ap.add_argument("--rules", nargs="*",
                    default=["const0.1", "dens_le2", "dens_le3", "dens_le4",
                             "topk2", "topk3", "inat_t2", "oracle_gold"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-json", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)

    gold_path = Path(args.gold)
    if not gold_path.exists():
        print(f"[2] gold 不存在：{rel(gold_path)}")
        return 2
    items = load_dev(gold_path)
    if args.limit:
        items = items[: args.limit]

    print(f"载入世代 {args.generation} …")
    loaded = pipeline.load_generation(args.generation)
    pipe = pipeline.build_pipeline(loaded, top_k=TOP_K, graph_depth=1)
    rt = pipe["retriever"]
    original_search = rt.search

    cache: dict[str, dict] = {}
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
        if hits:
            # 与 a_cooc_scope_probe 一致：深池 score 已含生产常量的 cooc 加成，先扣回
            raw = {h["chunk_id"]: float(h["score"]) - float(h.get("cooc_boost") or 0.0) for h in hits}
        else:
            raw = {}
        cache[qid] = {
            "hits": hits,
            "raw": raw,
            "cooc": cooc_table(rt, qa, hits) if hits else {},
            "strict": strict_table(rt, qa, hits) if hits else {},
            "gold": {str(g) for g in (item.get("gold_chunk_ids") or [])},
        }
        if n % 100 == 0:
            print(f"  已算 {n}/{len(items)} 题")

    def order_of(qid: str, rule: str):
        c = cache[qid]
        hits = c["hits"]
        if not hits:
            return None
        boosts, diag = boosts_for(rule, hits, c["cooc"], c["gold"], c.get("strict"))
        order = [h["chunk_id"] for h in hits]
        rank = {cid: i + 1 for i, cid in enumerate(order)}
        new_order = sorted(order, key=lambda cid: (-(c["raw"][cid] + boosts.get(cid, 0.0)), rank[cid]))
        return new_order, rank, diag

    # ---- 自检：none 必须与深池原顺序逐位相同 ----
    bad = []
    for item in items:
        qid = item["question_id"]
        r = order_of(qid, "none")
        if not r:
            continue
        new_order, rank, _ = r
        if new_order != sorted(cache[qid]["raw"], key=lambda c: (-cache[qid]["raw"][c], rank[c])):
            bad.append(qid)
    print(f"[自检] none 与原分排序一致：差异 {len(bad)} 题")
    if bad:
        print("[4] 自检失败")
        return 4

    probe = {}
    probe_path = Path(args.probe_json)
    if probe_path.exists():
        try:
            probe = json.loads(probe_path.read_text(encoding="utf-8"))["per_rule"]
        except Exception:
            probe = {}

    print("\n洗牌量（350 题分母）：")
    print("规则            top20集合变动  Jaccard均值  跨线进 跨线出  真跑进ctxΔ")
    out_rules = {}
    base_top = {}
    new_order_baseline: dict[str, list[str]] = {}
    for item in items:
        qid = item["question_id"]
        r = order_of(qid, "none")
        if r:
            new_order_baseline[qid] = r[0]
            base_top[qid] = set(r[0][:TOP_K])

    for rule in args.rules:
        changed = 0
        jac_sum = 0.0
        jac_n = 0
        cross_in, cross_out = [], []
        for item in items:
            qid = item["question_id"]
            r = order_of(qid, rule)
            if not r:
                continue
            new_order, rank, _ = r
            new_top = set(new_order[:TOP_K])
            old_top = base_top.get(qid) or set()
            if new_top != old_top:
                changed += 1
            if old_top or new_top:
                inter = len(old_top & new_top)
                union = len(old_top | new_top) or 1
                jac_sum += inter / union
                jac_n += 1
            gold = cache[qid]["gold"]
            # 原顺序名次（= 未加成时的名次）
            old_pos = {cid: i + 1 for i, cid in enumerate(new_order_baseline[qid])}
            # 重排后名次
            new_pos = {cid: i + 1 for i, cid in enumerate(new_order)}
            old_r = [old_pos[g] for g in gold if g in old_pos]
            new_r = [new_pos[g] for g in gold if g in new_pos]
            if old_r and new_r:
                if min(old_r) > LINE and min(new_r) <= LINE:
                    cross_in.append(qid)
                if min(old_r) <= LINE and min(new_r) > LINE:
                    cross_out.append(qid)
        d_ctx = None
        if rule in probe:
            pr = {x["question_id"]: x for x in probe[rule]["rows"]}
            pn = probe.get("none")
            if pn:
                bn = {x["question_id"]: x for x in pn["rows"]}
                d_ctx = sum(1 for q in pr if pr[q]["any_gold"] and not (bn.get(q) or {}).get("any_gold"))
                d_ctx -= sum(1 for q in pr if (bn.get(q) or {}).get("any_gold") and not pr[q]["any_gold"])
        out_rules[rule] = {
            "top20_changed": changed,
            "jac_mean": round(jac_sum / jac_n, 4) if jac_n else None,
            "cross_in": cross_in, "cross_out": cross_out,
            "n_cross_in": len(cross_in), "n_cross_out": len(cross_out),
            "real_d_any_gold": d_ctx,
        }
        print(f"  {rule:12s} {changed:9d}/{jac_n:<5d} {out_rules[rule]['jac_mean']!s:>8s}"
              f" {len(cross_in):7d} {len(cross_out):6d}  "
              f"{('—' if d_ctx is None else f'{d_ctx:+d}'):>6s}")

    out = {
        "schema": "a-rank-churn-1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "command": "py -3 scripts/a_rank_churn.py",
        "generation": args.generation,
        "gold_set": rel(gold_path),
        "probe_json": rel(probe_path) if probe_path.exists() else None,
        "split_policy": ("dev-only-pre-split-diagnostic：仅 dev 调参集 350 道应答题；"
                         "结论不得外推到 new-holdout 160（一次都未观测）。"),
        "n_questions": len(items),
        "capacity_line": LINE,
        "capacity_const": CAPACITY,
        "caveat": ("跨线进/出是「有希望真正进上下文」的上界，不是结果：ask_once 的 expand() "
                   "会把图扩展单元也塞进上下文，名次 6–8 的 gold 有时也能进。"
                   "真进上下文只能由 a_cooc_scope_probe.py 真跑给出。"),
        "rules": out_rules,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写出 {rel(out_path)}  ({out_path.stat().st_size:,} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
