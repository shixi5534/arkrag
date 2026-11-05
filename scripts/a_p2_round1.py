"""P2 第一轮：`cooc0.1` 上生产代码后的真跑验收（dev 350）。

与 `scripts/a_hop_suppress.py` 的分工说清楚，别混用：

* `a_hop_suppress.py` 是**进程内替换 `retriever.search`** 的离线加成扫描，
  用来在代码没改之前估上界。它的 `boost` 折进 `hit["score"]`，但没有走生产的
  `_cooc_boost()`、没有进 `stats`、也没有 `cooc_boost` 字段。
* 本脚本**不替换任何函数**，直接调 `pipeline.ask_once`，跑的是已经合入
  `src/arkrag/retrieve.py` 的 `COOC_BOOST_WEIGHT` 那条通路。唯一动的是这个常量的
  数值（`--weights` 给多个值时逐个 monkeypatch），用来在同一次进程里拿到
  「同一份代码、只差权重」的对照——**这才是能写进验收的基线**。

为什么基线必须在本脚本里重跑、不能引用 `a_cooc03_real.json` 的 189/143/58/30：
那份数是**替换版 search** 打出来的。现在生产 `search()` 已经改了（多了
`_cooc_boost()` 的逐块正文扫描，`stats` 多了三个键，`hits` 每项多了
`cooc_boost` 字段）。虽然权重为 0 时理论上结果应逐位相同，但这条「理论上」
必须**实测**，不能假定——见 `--verify-zero`。

## 四维分列（Lead 的 R2b' 通用规则）

任何配置必须分开报 `any_gold` / `all_gold` / `coverage_ok` / `citations`，
每维各报救/丢与题号。**净额为正不能抵消某一维的回退。**

## 退出码

0 = 跑通；2 = 输入不可用；4 = 自检失败（零权重与关掉信号不一致 / 题数不对）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from arkrag import config, pipeline as pipeline_mod, query as query_mod, retrieve  # noqa: E402

DEFAULT_GENERATION = "g-20261103T132152Z-06a741"
DEFAULT_GOLD = "data/goldset/gold-v2.json"
EVAL_TOP_K = 20
EVAL_GRAPH_DEPTH = 1


def rel(path) -> str:
    """入库一律写相对仓库根的路径（`tests/test_portable_paths.py`）。"""
    return config.rel_to_root(str(path))


def load_dev(gold_path: Path) -> list[dict]:
    """调参集：`split == dev` 且应答题（有 gold_chunk_ids、不要求拒答）。

    验收集（holdout 260）**一律不碰**——本函数只按 `split` 取值，不做别的挑选，
    避免哪天顺手写了 `[:n]` 把验收集漏进来（`ask.py --limit` 的历史事故）。
    """
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    items = []
    for item in gold.get("items") or []:
        if item.get("split") != "dev":
            continue
        if item.get("should_abstain"):
            continue
        if not (item.get("gold_chunk_ids") or []):
            continue
        items.append(item)
    return items


def coverage_of(answer: dict, reqs: list[dict]) -> bool:
    """关键事实引用覆盖，与 `scripts/ask.py:283-298` 同口径。

    注意是 `all(...)` + `req.evidence in cite.evidence`：每条 required 都要被
    某条引用覆盖，且 required 的 evidence 是引用 evidence 的**子串**
    （引用可以更长，包含整行说话人前缀）。
    """
    emitted = {
        (str(c.get("chunk_id") or ""), str(c.get("evidence") or ""))
        for c in (answer.get("citations") or [])
    }
    return bool(reqs) and all(
        any(cid == str(r.get("chunk_id") or "") and str(r.get("evidence") or "") in ev
            for cid, ev in emitted)
        for r in reqs
    )


def citation_hard_failures(answer: dict, chunks: dict) -> int:
    """引用定位硬失败数：`text[start:end]` 必须逐字等于 `evidence`。

    与 `scripts/ask.py:262-270` 同口径。生产里 `verify_citations` 会先把不合格的
    引用剔除，所以正常应是 0；这里再核一遍是为了抓「剔除后仍有漏网」。
    """
    bad = 0
    for cite in answer.get("citations") or []:
        rec = chunks.get(str(cite.get("chunk_id") or ""))
        text = str((rec or {}).get("text") or "")
        start, end = int(cite.get("start", -1)), int(cite.get("end", -1))
        if rec is None or not (0 <= start < end <= len(text)) or text[start:end] != cite.get("evidence"):
            bad += 1
    return bad


def run_one(pipeline, item: dict, chunks: dict) -> dict:
    """跑一题，返回逐题一行。字段取名与 `a_triage_all320.jsonl` 对齐，便于交叉引用。"""
    t0 = time.perf_counter()
    out = pipeline_mod.ask_once(pipeline, item["question"])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    search = out.get("search") or {}
    hits = [str(h.get("chunk_id") or "") for h in (search.get("hits") or [])]
    ans = out.get("answer") or {}
    ctx = out.get("context") or {}
    units = [str(u.get("chunk_id") or "") for u in (ctx.get("units") or [])]
    gold = [str(g) for g in (item.get("gold_chunk_ids") or [])]
    reqs = item.get("required_citations") or []

    rank = {cid: i + 1 for i, cid in enumerate(hits)}
    gold_ranks = [rank[g] for g in gold if g in rank]
    present = set(units)

    return {
        "question_id": item["question_id"],
        "category": item["category"],
        "origin": item.get("origin"),
        "gold_first_rank": min(gold_ranks) if gold_ranks else None,
        "any_gold": any(g in present for g in gold),
        "all_gold": all(g in present for g in gold),
        "covered": coverage_of(ans, reqs),
        "abstained": bool(ans.get("abstained")),
        "units": len(units),
        "tokens_used": int((ctx.get("stats") or {}).get("tokens_used") or 0),
        "citations": len(ans.get("citations") or []),
        "citation_hard_failures": citation_hard_failures(ans, chunks),
        "answer_len": len(str(ans.get("answer") or "")),
        "n_required": len(reqs),
        "cooc_boosted": int((search.get("stats") or {}).get("cooc_boosted") or 0),
        "cooc_applied": sum(1 for h in (search.get("hits") or []) if float(h.get("cooc_boost") or 0.0) > 0),
        "cooc_endpoints": list((search.get("stats") or {}).get("cooc_endpoints") or []),
        "elapsed_ms": round(elapsed_ms, 2),
    }


DIMS = ("any_gold", "all_gold", "covered")


def compare(base: dict, cand: dict) -> dict:
    """四维分列救/丢 + 题号。这是 R2b' 的机器可读形态。"""
    out = {}
    for dim in DIMS:
        gain = [q for q in base if not base[q][dim] and cand[q][dim]]
        lose = [q for q in base if base[q][dim] and not cand[q][dim]]
        out[dim] = {
            "base": sum(1 for q in base if base[q][dim]),
            "cand": sum(1 for q in cand if cand[q][dim]),
            "gain": len(gain),
            "lose": len(lose),
            "net": len(gain) - len(lose),
            "gain_ids": gain,
            "lose_ids": lose,
        }
    for field in ("citations", "citation_hard_failures", "abstained"):
        out[field] = {
            "base": sum(int(base[q][field]) for q in base),
            "cand": sum(int(cand[q][field]) for q in cand),
        }
    loss = [q for q in base
            if base[q]["elapsed_ms"] and cand[q]["elapsed_ms"] / base[q]["elapsed_ms"] > 1.5
            and cand[q]["elapsed_ms"] - base[q]["elapsed_ms"] > 20]
    out["latency"] = {
        "base_median_ms": round(statistics.median([base[q]["elapsed_ms"] for q in base]), 2),
        "cand_median_ms": round(statistics.median([cand[q]["elapsed_ms"] for q in cand]), 2),
        "base_p90_ms": round(sorted(base[q]["elapsed_ms"] for q in base)[int(len(base) * 0.9)], 2),
        "cand_p90_ms": round(sorted(cand[q]["elapsed_ms"] for q in cand)[int(len(cand) * 0.9)], 2),
        "slow_questions": len(loss),
        "slow_ids": loss,
    }
    return out


def per_category(base: dict, cand: dict) -> dict:
    """按类二维（any_gold / covered）分列，回退题号必须能看到。"""
    cats = sorted({base[q]["category"] for q in base})
    out = {}
    for cat in cats:
        ids = [q for q in base if base[q]["category"] == cat]
        b_any = sum(1 for q in ids if base[q]["any_gold"])
        c_any = sum(1 for q in ids if cand[q]["any_gold"])
        b_cov = sum(1 for q in ids if base[q]["covered"])
        c_cov = sum(1 for q in ids if cand[q]["covered"])
        out[cat] = {
            "n": len(ids),
            "any_gold": [b_any, c_any, c_any - b_any],
            "covered": [b_cov, c_cov, c_cov - b_cov],
            "any_gold_lose": [q for q in ids if base[q]["any_gold"] and not cand[q]["any_gold"]],
            "covered_lose": [q for q in ids if base[q]["covered"] and not cand[q]["covered"]],
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generation", default=DEFAULT_GENERATION)
    ap.add_argument("--gold-v2", default=DEFAULT_GOLD)
    ap.add_argument("--weights", nargs="+", type=float, default=[0.0, 0.1],
                    help="要跑的 COOC_BOOST_WEIGHT 序列；第一个视作基线")
    ap.add_argument("--limit", type=int, default=0, help="仅调试用；正式验收不许传")
    ap.add_argument("--out-json", default="data/a_p2_round1.json")
    args = ap.parse_args(argv)

    gold_path = Path(args.gold_v2)
    if not gold_path.is_absolute():
        gold_path = ROOT / gold_path
    if not gold_path.is_file():
        print(f"[2] 找不到 gold 文件：{gold_path}", file=sys.stderr)
        return 2

    items = load_dev(gold_path)
    if args.limit:
        print(f"[!] --limit {args.limit} 只用于调试，正式验收必须全量", file=sys.stderr)
        items = items[: args.limit]
    if not items:
        print("[2] dev 应答题为 0，检查 split 字段", file=sys.stderr)
        return 2

    print(f"载入世代 {args.generation} …", flush=True)
    loaded = pipeline_mod.load_generation(args.generation)
    pipe = pipeline_mod.build_pipeline(loaded, top_k=EVAL_TOP_K, graph_depth=EVAL_GRAPH_DEPTH)
    retriever = pipe["retriever"]
    chunks = loaded["chunks"]

    original_weight = retrieve.COOC_BOOST_WEIGHT
    per_weight: dict[str, list[dict]] = {}
    try:
        for w in args.weights:
            retrieve.COOC_BOOST_WEIGHT = float(w)
            tag = f"cooc{w:g}"
            print(f"  跑 {tag} …", flush=True)
            rows = [run_one(pipe, item, chunks) for item in items]
            per_weight[tag] = rows
            n_any = sum(1 for r in rows if r["any_gold"])
            n_all = sum(1 for r in rows if r["all_gold"])
            n_cov = sum(1 for r in rows if r["covered"])
            n_cit = sum(r["citations"] for r in rows)
            print(f"    any_gold {n_any} / all_gold {n_all} / covered {n_cov} / citations {n_cit}",
                  flush=True)
    finally:
        retrieve.COOC_BOOST_WEIGHT = original_weight

    tags = list(per_weight)
    base_tag, base_rows = tags[0], per_weight[tags[0]]
    base = {r["question_id"]: r for r in base_rows}

    # 自检 1：题数
    if len(base_rows) != len(items):
        print(f"[4] 自检失败：基线行数 {len(base_rows)} != 题数 {len(items)}", file=sys.stderr)
        return 4
    # 自检 2：基线必须是合法的「关掉信号」状态。
    # 注意 `cooc_boosted`（stats 里的信号命中块数）与权重无关，权重 0 时它照样 >0，
    # 拿它做判据会误报。要看的是 `cooc_applied`——真正被加进 score 的块数。
    if any(r["cooc_applied"] for r in base_rows):
        print("[4] 自检失败：权重 0 时仍有块被加进分，常量没被读取点接上", file=sys.stderr)
        return 4

    comparisons = {}
    for tag in tags[1:]:
        rows = per_weight[tag]
        cand = {r["question_id"]: r for r in rows}
        if set(cand) != set(base):
            print(f"[4] 自检失败：{tag} 的题号集合与基线不同", file=sys.stderr)
            return 4
        comparisons[tag] = {
            "dims": compare(base, cand),
            "per_category": per_category(base, cand),
        }

    report = {
        "schema": "ark-p2-round1-1",
        "split_policy": "dev-only-pre-split-diagnostic",
        "note": (
            "P2 第一轮真跑。跑的是合入 src/arkrag/retrieve.py 的 COOC_BOOST_WEIGHT 通路，"
            "不替换任何函数。dev 350 应答题（split==dev 且非应拒），验收集未使用。"
            "四维分列口径见 R2b'：净额为正不能抵消某一维的回退。"
        ),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generation": args.generation,
        "gold_set": rel(gold_path),
        "n_questions": len(items),
        "weights": args.weights,
        "baseline_tag": base_tag,
        "config": {
            "eval_top_k": EVAL_TOP_K,
            "eval_graph_depth": EVAL_GRAPH_DEPTH,
            "max_context_tokens": retrieve.DEFAULT_BUDGET["max_context_tokens"],
            "candidate_pool_mult": retrieve.CANDIDATE_POOL_MULT,
            "speaker_boost_weight": retrieve.SPEAKER_BOOST_WEIGHT,
            "entity_channel_weight": retrieve.ENTITY_CHANNEL_WEIGHT,
            "cooc_boost_weight_original": original_weight,
        },
        "summary": {
            tag: {
                "any_gold": sum(1 for r in rows if r["any_gold"]),
                "all_gold": sum(1 for r in rows if r["all_gold"]),
                "covered": sum(1 for r in rows if r["covered"]),
                "abstained": sum(1 for r in rows if r["abstained"]),
                "citations": sum(r["citations"] for r in rows),
                "citation_hard_failures": sum(r["citation_hard_failures"] for r in rows),
                "median_ms": round(statistics.median([r["elapsed_ms"] for r in rows]), 2),
            }
            for tag, rows in per_weight.items()
        },
        "comparisons": comparisons,
        "per_question": per_weight,
        "provenance": {
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "os": f"{platform.system()}-{platform.release()}-{platform.version()}",
            "command": "py -3 scripts/a_p2_round1.py",
            "host_sha256_of_gold": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
        },
    }

    out_path = Path(args.out_json)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写出 {out_path}  ({out_path.stat().st_size:,} B)")

    # 控制台摘要：四维分列，不许只报净额
    for tag, cmp_ in comparisons.items():
        d = cmp_["dims"]
        print(f"\n{base_tag} → {tag}")
        for dim in DIMS:
            x = d[dim]
            print(f"  {dim:9s} {x['base']:>4} → {x['cand']:>4}  救 {x['gain']:>2} 丢 {x['lose']:>2} "
                  f"净 {x['net']:+d}" + (f"  丢={x['lose_ids']}" if x["lose_ids"] else ""))
        c = d["citations"]
        print(f"  citations {c['base']:>4} → {c['cand']:>4}")
        f = d["citation_hard_failures"]
        print(f"  引用硬失败 {f['base']:>3} → {f['cand']:>3}")
        lat = d["latency"]
        print(f"  耗时中位 {lat['base_median_ms']:.1f} → {lat['cand_median_ms']:.1f} ms，"
              f"P90 {lat['base_p90_ms']:.1f} → {lat['cand_p90_ms']:.1f} ms，"
              f"变慢>1.5x 且 >20ms 的题 {lat['slow_questions']}")
    return 0


def _git_commit() -> str:
    """尽力取当前 commit；取不到就如实写 unknown，不编。"""
    head = ROOT / ".git" / "HEAD"
    try:
        ref = head.read_text(encoding="utf-8").strip()
        if ref.startswith("ref: "):
            target = ROOT / ".git" / ref[5:]
            return target.read_text(encoding="utf-8").strip()
        return ref
    except OSError:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
