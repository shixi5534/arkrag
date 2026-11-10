"""用聚类 bootstrap 重算三条历史通路的显著性（计划书 §W4 J4 要求）。

## 计划书原文

> 历史通路 `quote_locate` p=0.0021、`bridge` CI[+5,+18]、`graph_reserve` hold p=0.1562
> 均须用冻结簇定义和同一配对答案重新计算；**在新 cluster CI 出来前，
> 不预写「仍显著/不被推翻」的结论**。

## 三条通路各自的对照

| 通路 | 实验臂 | 对照臂 |
|---|---|---|
| `quote_locate` | `quote_locate=True` | `quote_locate=False` |
| `bridge` | `bridge=True` + `graph_reserve_tokens=12000` | 两者都关 |
| `graph_reserve` | `graph_reserve_tokens=12000` | `graph_reserve_tokens=0` |

三组都在**其余生产档开关不变**的前提下比。臂间唯一差别就是要测的那个开关
（`graph_reserve` 那组的 bridge 两边都开）。

## 为什么必须重算

计划书说 cluster CI 的宽度会膨胀。dev 上 W4 已实测：v1→v2 的配对 Δ
方差膨胀 **4.4497**，CI 宽度 6.66pt → 14.07pt（2.11 倍）。
**历史 p 值若按 iid 算，是被低估的。**

## 产出

`data/diag_hist_pvalues.json`：每条通路的 iid 与 cluster 两套 Δ/CI/p，
以及方差膨胀。
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import cluster_ci  # noqa: E402
from arkrag import judge as judge_mod  # noqa: E402
from arkrag import pipeline as pipeline_mod  # noqa: E402

GEN = "g-20261103T132152Z-06a741"
GOLD = ROOT / "data" / "goldset" / "gold-v2.json"
GOLD_V1 = ROOT / "data" / "goldset" / "gold-v1.json"
OUT = ROOT / "data" / "diag_hist_pvalues.json"

BUDGET = int(pipeline_mod.PROD_PROFILE["max_context_tokens"])
BASE = pipeline_mod.prod_options()

#: 三条通路：(名字, 实验臂覆盖, 对照臂覆盖)
ARMS = [
    ("quote_locate",
     {"quote_locate": True},
     {"quote_locate": False}),
    ("bridge+reserve",
     {"bridge": True, "graph_reserve_tokens": 12000},
     {"bridge": False, "graph_reserve_tokens": 0}),
    ("graph_reserve",
     {"graph_reserve_tokens": 12000},
     {"graph_reserve_tokens": 0}),
]


def load_v1_ids():
    if not GOLD_V1.exists():
        return set()
    doc = json.loads(GOLD_V1.read_text(encoding="utf-8"))
    items = doc.get("items") if isinstance(doc, dict) else doc
    return {str(it.get("question_id")) for it in items or []}


def pick(items, split, v1_ids):
    if split == "all":
        return [i for i in items if i.get("gold_chunk_ids")]
    if split == "clean":
        return [i for i in items if i.get("split") == "holdout"
                and i.get("gold_chunk_ids")
                and str(i.get("question_id")) not in v1_ids]
    return [i for i in items if i.get("split") == split and i.get("gold_chunk_ids")]


def chapter_texts(chunks):
    out = {}
    for cid, rec in (chunks or {}).items():
        ch = str((rec or {}).get("chapter") or "")
        if ch:
            out[ch] = out.get(ch, "") + str((rec or {}).get("text") or "")
    return out


def run_arm(pipe, items, chap, opts):
    """跑一臂，返回逐题 pass 向量（不可判为 None）。"""
    vec = []
    for it in items:
        q = str(it.get("question") or "")
        cat = str(it.get("category") or "")
        res = pipeline_mod.ask_once(pipe, q, options=opts)
        ans = res.get("answer") or {}
        text = str(ans.get("answer") or "")
        verdict = judge_mod.judge_answer(cat, text, it, question=q,
                                         chapter_texts=chap)
        vec.append(verdict.get("pass"))
    return vec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="dev,holdout,clean")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    v1 = load_v1_ids()
    doc = json.loads(GOLD.read_text(encoding="utf-8"))
    all_items = doc["items"] if isinstance(doc, dict) else doc

    by_split = {}
    for sp in splits:
        picked = pick(all_items, sp, v1)
        if args.limit:
            picked = picked[: args.limit]
        by_split[sp] = picked
        print(f"[hist] {sp}: {len(picked)} 题", flush=True)

    t0 = time.time()
    loaded = pipeline_mod.load_generation(GEN)
    pipe = pipeline_mod.build_pipeline(loaded, top_k=20, graph_depth=1)
    pipe["generation_id"] = GEN
    pipe["entities"] = loaded["graph"].get("entities") or []
    pipe["retriever"].budget["max_context_tokens"] = BUDGET
    chap = chapter_texts(loaded["chunks"])
    print(f"[hist] load {time.time()-t0:.1f}s", flush=True)

    results = {}
    for name, exp_over, ctl_over in ARMS:
        results[name] = {}
        for sp in splits:
            items = by_split[sp]
            if not items:
                continue
            o_exp = dict(BASE)
            o_exp.update(exp_over)
            o_ctl = dict(BASE)
            o_ctl.update(ctl_over)
            t = time.time()
            v_exp = run_arm(pipe, items, chap, o_exp)
            v_ctl = run_arm(pipe, items, chap, o_ctl)
            # 只在两臂都能判的题上配对
            pairs = [(e, c) for e, c in zip(v_exp, v_ctl)
                     if e is not None and c is not None]
            before = [1 if c is True else 0 for _, c in pairs]
            after = [1 if e is True else 0 for e, _ in pairs]
            keys = cluster_ci.cluster_keys(
                [str(it.get("question") or "") for it, (e, c) in
                 zip(items, zip(v_exp, v_ctl)) if e is not None and c is not None])
            if not pairs:
                results[name][sp] = {"n_paired": 0, "note": "无可配对的题"}
                continue
            iid = cluster_ci.paired_bootstrap_delta(before, after, keys, clustered=False)
            clu = cluster_ci.paired_bootstrap_delta(before, after, keys, clustered=True)
            infl = cluster_ci.variance_inflation(iid, clu)
            results[name][sp] = {
                "n_paired": len(pairs),
                "n_clusters": len(set(keys)),
                "exp_pass": sum(after), "ctl_pass": sum(before),
                "delta_pp": iid.get("delta_pp"),
                "iid": iid, "cluster": clu,
                "variance_inflation": infl,
                "elapsed_s": round(time.time() - t, 1),
            }
            print(f"[hist] {name} {sp}: exp={sum(after)} ctl={sum(before)} "
                  f"Δ={iid.get('delta_pp'):+.2f}pt  "
                  f"iid CI[{iid.get('ci_low_pp'):.2f},{iid.get('ci_high_pp'):.2f}] "
                  f"clu CI[{clu.get('ci_low_pp'):.2f},{clu.get('ci_high_pp'):.2f}] "
                  f"膨胀={infl.get('value')}", flush=True)

    payload = {
        "schema": "arkrag.diag_hist_pvalues/1",
        "purpose": "用聚类 bootstrap 重算三条历史通路的显著性（计划书 J4）",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": platform.python_version(),
        "os": platform.platform(),
        "command": "python scripts/run_guarded.py -- python scripts/diag_hist_pvalues.py",
        "generation": GEN,
        "base_options": {k: (v if not isinstance(v, dict) else "<calib>")
                         for k, v in BASE.items()},
        "arms": [{"name": n, "exp": e, "ctl": c} for n, e, c in ARMS],
        "planned_reference": {
            "quote_locate_p": 0.0021,
            "bridge_ci": [5, 18],
            "graph_reserve_holdout_p": 0.1562,
            "note": ("这三个是历史读数，用 iid 口径算的。本文件的 cluster 列才是"
                     "可比的；在新 cluster CI 出来前不预写「仍显著/不被推翻」。"),
        },
        "results": results,
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                              encoding="utf-8")
    print(f"[hist] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
