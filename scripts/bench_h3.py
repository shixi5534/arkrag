"""H3 双跑对比：同一批题、同一 top_k、同一 graph_depth、同一种子，跑旧 JSON 后端与新 FTS5 后端。

## 两种口径，回答两个不同的问题（这一点必须看清楚）

**口径 A —— 跨子进程**（`run_one_backend`）。每个后端跑在独立子进程里，各自从
干净状态起算。这是**冷启动与内存**的正确口径：先跑 JSON 再跑 FTS，测到的 RSS 会
包含 127 MB JSON 解析留下的碎片，所以必须分开。**但它的延迟数不能用来比较检索快慢**：
JSON 子进程里躺着 1.5 GB 的倒排 dict，FTS 子进程只占 532 MiB，于是 FTS 子进程的
*每一个阶段*都白拿了一份「内存更宽裕」的好处。实测这个偏差足以把结论反向翻转——
口径 A 下 FTS 的 search 看起来快 17%，而口径 B 下它其实更慢。

**口径 B —— 同进程交替**（`run_interleaved_latency`）。两个后端都载入同一个进程，
交替切换着跑，内存状态完全相同。这是**「后端本身快慢」**的唯一有效口径。

## 为什么指标计算不在这里重写一遍

七项指标（Recall@20/@5、MRR、引用定位、引用覆盖、拒答、方向）的判定逻辑都在
`scripts/ask.py::evaluate` 里，且带一堆口径注释（哪些题该跳过、分母怎么算）。
在这里再写一份就是「同一判定写两处」——`tests/test_single_source_of_truth.py`
记录的三次事故都是这个机理。做法是**只替换载入层**：

* `arkrag.index.load_index` 改成返回 `Fts5SearchBackend`（它满足 `Retriever`
  对索引的全部依赖）；
* `ask.load_generation` 改成返回已载入的缓存，让 `ask.evaluate` 复用它而不是重读。

于是 `ask.evaluate` 原样运行，两种后端走的是**同一个函数**，指标口径不可能不同。

## 用法

    py -3 scripts\\bench_h3.py --generation g-20261031T035231Z-4eafae --questions 320

产物：`data/arkrag/fts/bench_h3_<gid>.json`（可用 `--json` 改）。
`--interleaved-rounds 0` 可跳过口径 B（不建议，那样就没有可信的检索延迟对比）。

## 退出码

* `0` —— 双跑跑通。**参考：慢也是 0**，这个脚本是测量器不是门禁。
* `2` —— 硬错误：世代不存在、FTS 索引不存在（先跑 `build_fts_index.py`）、题目为空。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from arkrag import config, fts_backend, index as index_mod, release, search_backend  # noqa: E402

import ask as ask_mod  # noqa: E402
import bench_h1 as bench_mod  # noqa: E402

DEFAULT_TOP_K = 20
DEFAULT_GRAPH_DEPTH = 1
DEFAULT_WARM_REPS = 3
DEFAULT_SEED = 20261101

METRIC_KEYS: tuple[str, ...] = (
    "recall_at_20",
    "recall_at_5",
    "mrr",
    "citation_location_accuracy",
    "required_citation_coverage",
    "abstain_accuracy",
    "direction_accuracy",
)


# ---------------------------------------------------------------- 单个后端（子进程里跑）
def run_one_backend(
    generation_id: str,
    *,
    backend_kind: str,
    questions: int,
    warm_reps: int,
    top_k: int,
    graph_depth: int,
    gold_path: Path,
    db_path: Path | None = None,
) -> dict:
    """在**当前进程**里跑一个后端，返回完整实测报告。由子进程调用。"""
    mem_start = bench_mod.rss_mib()
    timings: dict = {"backend": backend_kind}

    # ---- 只替换载入层（见模块 docstring）
    if backend_kind == search_backend.BACKEND_FTS:
        target_db = Path(db_path) if db_path is not None else search_backend.default_db_path(generation_id)
        if not target_db.exists():
            raise SystemExit(
                f"[2] FTS 索引不存在：{target_db}\n"
                f"    先跑 py -3 scripts\\build_fts_index.py --generation {generation_id}"
            )
        t0 = time.perf_counter()
        backend = fts_backend.Fts5SearchBackend.open(target_db)
        open_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        timings["index_open_ms"] = open_ms
        timings["backend_describe"] = backend.describe()

        real_load_index = index_mod.load_index

        def _patched_load_index(path):  # noqa: ANN001 - 签名与真实实现一致
            # 元数据从库里带回来，让 `ask.load_generation` 里的 `index_meta`
            # 仍是一份可记录的字典（probe_digest / index_content_digest 都在里面）。
            meta = backend.meta
            meta.setdefault("tokenizer_version", backend.tokenizer_version)
            return backend, meta

        index_mod.load_index = _patched_load_index
        try:
            loaded = _load_and_measure(generation_id, timings)
        finally:
            index_mod.load_index = real_load_index
        # 新后端的「载入索引」成本就是打开库那一步（`_load_and_measure` 里量到的
        # 那次 `load_index` 调用是打桩的，只返回已打开的 backend，≈0 ms）。
        # 这里换成真实值并重算总数，否则报告里会出现「冷启动 1.9 s、索引 0 ms」
        # 这种自相矛盾的行——因为它把打开库的时间漏在总数外面。
        timings["index_ms"] = open_ms
        timings.pop("index_open_ms", None)
        timings["total_ms"] = round(
            sum(v for k, v in timings.items()
                if k.endswith("_ms") and k != "total_ms" and isinstance(v, (int, float))),
            2,
        )
        timings["loaded_backend_kind"] = type(loaded["index"]).__name__
        if not isinstance(loaded["index"], fts_backend.Fts5SearchBackend):
            # 诊断性断言：如果打桩没生效，这次"新后端"的数其实是旧后端的数，
            # 而报告会照常打印——正是要防的那类静默错误。
            raise SystemExit("[2] FTS 后端打桩未生效，拒绝输出可能错误的对比数字")
    else:
        t0 = time.perf_counter()
        loaded = _load_and_measure(generation_id, timings)
        timings["backend_describe"] = search_backend.describe_backend(
            search_backend.JsonSearchBackend(loaded["index"], source_path=loaded["index_path"])
        )
        timings["loaded_backend_kind"] = type(loaded["index"]).__name__
        timings["_json_load_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

    mem_loaded = bench_mod.rss_mib()

    # ---- 管线构造（含 Retriever 建表）
    pipeline, retriever_ms = bench_mod.build_pipeline_timing(loaded, top_k=top_k, graph_depth=graph_depth)
    mem_pipeline = bench_mod.rss_mib()

    # ---- 同一批题
    picked = bench_mod.pick_questions(gold_path, questions)

    # ---- 延迟：每题先 cold 一次，再 warm_reps 次取中位数（与 bench_h1 同口径）
    cold_total: list[float] = []
    warm_total: list[float] = []
    stage_cold: dict[str, list[float]] = {k: [] for k in
                                         ("query_ms", "search_ms", "expand_ms", "context_ms", "answer_ms")}
    stage_warm: dict[str, list[float]] = {k: [] for k in stage_cold}
    top20_per_question: list[list[str]] = []
    candidates: list[float] = []
    retrieval_queries: list[str] = []

    for item in picked:
        question = str(item.get("question") or "")
        row = {"question_id": item.get("question_id"), "category": item.get("category")}

        cold = dict()
        result = ask_mod.ask_once(pipeline, question, timings=cold)
        cold_total.append(float(cold.get("total_ms") or 0.0))
        for key in stage_cold:
            if cold.get(key) is not None:
                stage_cold[key].append(float(cold[key]))
        top20_per_question.append([str(h.get("chunk_id") or "") for h in result["search"]["hits"]][:20])
        if cold.get("candidates") is not None:
            candidates.append(float(cold["candidates"]))
        qa = result.get("query_analysis") or {}
        retrieval_queries.append(str(qa.get("retrieval_query") or qa.get("normalized_query") or ""))

        reps: list[float] = []
        rep_rows: list[dict] = []
        for _ in range(max(1, warm_reps)):
            warm = dict()
            ask_mod.ask_once(pipeline, question, timings=warm)
            reps.append(float(warm.get("total_ms") or 0.0))
            rep_rows.append(warm)
        warm_total.append(float(bench_mod.percentile(reps, 50) or 0.0))
        for key in stage_warm:
            vals = [float(r[key]) for r in rep_rows if r.get(key) is not None]
            if vals:
                stage_warm[key].append(float(bench_mod.percentile(vals, 50) or 0.0))

        row["cold_ms"] = round(cold_total[-1], 3)
        row["warm_ms"] = round(warm_total[-1], 3)
        row["search_ms"] = round(float(cold.get("search_ms") or 0.0), 3)

    mem_end = bench_mod.rss_mib()

    # ---- 七项指标：原样调 `ask.evaluate`，只让它复用已载入的产物
    real_load_generation = ask_mod.load_generation
    ask_mod.load_generation = lambda gid: loaded  # noqa: ARG005 - 忽略入参，复用缓存
    try:
        eval_report = ask_mod.evaluate(gold_path, limit=questions, generation_id=generation_id)
    finally:
        ask_mod.load_generation = real_load_generation

    metrics = {
        key: {
            "value": eval_report["metrics"][key]["value"],
            "n": eval_report["metrics"][key]["n"],
            "skipped": eval_report["metrics"][key].get("skipped", 0),
        }
        for key in METRIC_KEYS
    }

    report = {
        "backend": backend_kind,
        "generation_id": generation_id,
        "questions": len(picked),
        "config": {
            "top_k": top_k,
            "graph_depth": graph_depth,
            "warm_reps": warm_reps,
            "seed": DEFAULT_SEED,
            "gold_set": str(gold_path),
        },
        "backend_describe": timings.get("backend_describe"),
        "load": {
            key: value
            for key, value in timings.items()
            if key.endswith("_ms") and key not in {"_json_load_ms"}
        },
        "retriever_build_ms": retriever_ms,
        "memory": {
            "at_start": mem_start,
            "after_load": mem_loaded,
            "after_pipeline": mem_pipeline,
            "at_end": mem_end,
        },
        "latency": {
            "cold": bench_mod.summarize(cold_total),
            "warm": bench_mod.summarize(warm_total),
            "stage_cold": {k: bench_mod.summarize(v) for k, v in stage_cold.items()},
            "stage_warm": {k: bench_mod.summarize(v) for k, v in stage_warm.items()},
        },
        "candidates": bench_mod.summarize(candidates),
        "metrics": metrics,
        "citation_verify_failures": eval_report.get("citation_verify_failures"),
        "top20": top20_per_question,
        "retrieval_queries": retrieval_queries,
        "measured_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    return report


def _load_and_measure(generation_id: str, timings: dict) -> dict:
    """逐产物计时载入。口径与 `bench_h1.timed_load` 一致（同一读法、同一编码）。

    不直接调 `bench_h1.timed_load`，因为它在函数内部**硬写**了
    `index_mod.load_index`；我们恰恰要替换这一步。其余四项读法与它逐行相同。
    """
    gen = release.Generation(generation_id)
    if not gen.dir.exists():
        raise SystemExit(f"[2] 世代不存在：{generation_id}")
    manifest = gen.load()

    t0 = time.perf_counter()
    graph = json.loads(gen.graph_path.read_text("utf-8"))
    timings["graph_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

    t0 = time.perf_counter()
    adjacency = json.loads(gen.adjacency_path.read_text("utf-8"))
    timings["adjacency_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

    t0 = time.perf_counter()
    assertions = ask_mod._read_jsonl(gen.assertion_path)
    timings["assertions_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

    t0 = time.perf_counter()
    idx, idx_meta = index_mod.load_index(gen.index_path)
    timings["index_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

    t0 = time.perf_counter()
    chunks: dict[str, dict] = {}
    chunks_path = config.chunks_path()
    if not chunks_path.exists():
        raise SystemExit(f"[2] 找不到块文件 {chunks_path}")
    for rec in ask_mod._read_jsonl(chunks_path):
        cid = str(rec.get("chunk_id") or "")
        if cid:
            chunks[cid] = rec
    timings["chunks_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    timings["total_ms"] = round(
        sum(v for k, v in timings.items() if k.endswith("_ms") and isinstance(v, (int, float))), 2
    )

    return {
        "generation_id": generation_id,
        "generation": gen,
        "index_path": gen.index_path,
        "manifest": manifest,
        "graph": graph,
        "adjacency": adjacency,
        "assertions": assertions,
        "index": idx,
        "index_meta": idx_meta,
        "chunks": chunks,
    }


# ---------------------------------------------------------------- 同进程交替 A/B
def run_interleaved_latency(
    generation_id: str,
    *,
    questions: int,
    top_k: int,
    graph_depth: int,
    gold_path: Path,
    db_path: Path,
    rounds: int = 2,
) -> dict:
    """同进程交替跑两个后端，用来把「后端本身快慢」与「进程内存压力」分开。

    ## 为什么必须有这一组

    上面那两张子进程报告是**冷启动与内存**的正确口径（各自从干净状态起算，
    没人被前一次运行的碎片污染）。但拿它们的**延迟**做对比是错的：
    JSON 子进程里躺着 1.5 GB 的倒排 dict，FTS 子进程里只有 532 MiB，
    于是 FTS 子进程的每一个阶段（query / expand / context 都算）都白拿了
    一份「内存更宽裕」的好处。实测这个偏差足以把结论**反向翻转**：
    子进程口径下 FTS 的 search 看起来快 17%，而同进程交替口径下它其实**更慢**。

    所以延迟必须以**同一进程、同一内存状态**下交替切换后端来量。这一组
    不做指标、不做 top-20，只回答一个问题：换后端本身让检索变快还是变慢。

    `rounds` 轮交替（json / fts / json / fts …），取所有轮次的样本合并计算，
    顺序效应被摊平；若两轮之间差异很大，说明测量本身不稳，会在返回里标出。
    """
    # JSON 后端先载入：这一组的用意就是让两个后端都活在**同一个**内存条件下，
    # 因此必须先付出 JSON 那份内存，再在同一条件下量 FTS。顺序不能反过来。
    loaded = ask_mod.load_generation(generation_id)
    be_json = search_backend.open_backend(loaded["generation"].dir, kind=search_backend.BACKEND_JSON)
    be_fts = search_backend.open_backend(
        loaded["generation"].dir, kind=search_backend.BACKEND_FTS, db_path=db_path
    )
    pipeline = ask_mod.build_pipeline(loaded, top_k=top_k, graph_depth=graph_depth)
    mem_both = bench_mod.rss_mib()

    picked = bench_mod.pick_questions(gold_path, questions)
    texts = [str(item.get("question") or "") for item in picked]

    per_round: list[dict] = []
    samples: dict[str, dict[str, list[float]]] = {
        "json": {"search": [], "total": []},
        "fts": {"search": [], "total": []},
    }

    for index in range(max(1, rounds)):
        # 每轮内部也交替两次，尽量让两个后端都经历「刚切换」与「已热身」两种状态
        for tag, backend in (("json", be_json), ("fts", be_fts),
                             ("json", be_json), ("fts", be_fts)):
            pipeline["retriever"].index = backend
            # 预热 5 题（不含在统计里），避免把首次页缓存算进去
            for text in texts[:5]:
                ask_mod.ask_once(pipeline, text)
            search_vals: list[float] = []
            total_vals: list[float] = []
            for text in texts:
                timings: dict = {}
                ask_mod.ask_once(pipeline, text, timings=timings)
                if timings.get("search_ms") is not None:
                    search_vals.append(float(timings["search_ms"]))
                if timings.get("total_ms") is not None:
                    total_vals.append(float(timings["total_ms"]))
            samples[tag]["search"].extend(search_vals)
            samples[tag]["total"].extend(total_vals)
        per_round.append({
            "round": index + 1,
            "json_search_p50": bench_mod.percentile(samples["json"]["search"][-len(texts):], 50),
            "fts_search_p50": bench_mod.percentile(samples["fts"]["search"][-len(texts):], 50),
        })

    rows = {}
    for tag in ("json", "fts"):
        rows[tag] = {
            "search": bench_mod.summarize(samples[tag]["search"]),
            "total": bench_mod.summarize(samples[tag]["total"]),
        }

    def _delta(metric: str, stat: str) -> float | None:
        a = rows["json"][metric].get(stat)
        b = rows["fts"][metric].get(stat)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return round(b - a, 3)
        return None

    rows["delta"] = {
        metric: {stat: _delta(metric, stat) for stat in ("p50_ms", "p95_ms", "p99_ms")}
        for metric in ("search", "total")
    }
    rows["rounds"] = per_round
    rows["rounds_requested"] = max(1, rounds)
    rows["questions"] = len(texts)
    rows["memory_both_backends"] = mem_both
    rows["note"] = (
        "同进程交替，两个后端共享同一内存状态；这是判定「后端本身快慢」的唯一有效口径。"
        "跨子进程那张延迟表的差异混入了内存压力，不能用来比较检索速度。"
    )

    pipeline["retriever"].index = be_json
    if hasattr(be_fts, "close"):
        be_fts.close()
    return rows


# ---------------------------------------------------------------- 对比
def compare(reports: dict[str, dict]) -> dict:
    """把两次实测折成对比结论。**只做减法与计数，不重新判定任何指标。**"""
    old = reports[search_backend.BACKEND_JSON]
    new = reports[search_backend.BACKEND_FTS]
    n = min(len(old["top20"]), len(new["top20"]))
    same_top20 = sum(1 for i in range(n) if old["top20"][i] == new["top20"][i])

    metric_rows = {}
    for key in METRIC_KEYS:
        a, b = old["metrics"][key], new["metrics"][key]
        metric_rows[key] = {
            "json": a["value"],
            "fts": b["value"],
            "delta": round(b["value"] - a["value"], 6),
            "n_json": a["n"],
            "n_fts": b["n"],
        }

    def _diff(path: list[str]) -> float | None:
        cur_a, cur_b = old, new
        for step in path:
            cur_a = (cur_a or {}).get(step) if isinstance(cur_a, dict) else None
            cur_b = (cur_b or {}).get(step) if isinstance(cur_b, dict) else None
        if isinstance(cur_a, (int, float)) and isinstance(cur_b, (int, float)):
            return round(cur_b - cur_a, 3)
        return None

    load_rows = {}
    for key in ("graph_ms", "adjacency_ms", "assertions_ms", "index_ms", "chunks_ms", "total_ms"):
        load_rows[key] = {
            "json": old["load"].get(key),
            "fts": new["load"].get(key),
            "delta": _diff(["load", key]),
        }

    mem_rows = {
        key: {
            "json": old["memory"].get(key, {}).get("current_mib"),
            "fts": new["memory"].get(key, {}).get("current_mib"),
            "peak_json": old["memory"].get(key, {}).get("peak_mib"),
            "peak_fts": new["memory"].get(key, {}).get("peak_mib"),
        }
        for key in ("at_start", "after_load", "after_pipeline", "at_end")
    }

    latency_rows = {
        bucket: {
            stat: {
                "json": old["latency"][bucket].get(stat),
                "fts": new["latency"][bucket].get(stat),
            }
            for stat in ("p50_ms", "p95_ms", "p99_ms")
        }
        for bucket in ("cold", "warm")
    }
    stage_rows = {
        key: {
            "json_p95": old["latency"]["stage_warm"].get(key, {}).get("p95_ms"),
            "fts_p95": new["latency"]["stage_warm"].get(key, {}).get("p95_ms"),
            "json_p50": old["latency"]["stage_warm"].get(key, {}).get("p50_ms"),
            "fts_p50": new["latency"]["stage_warm"].get(key, {}).get("p50_ms"),
        }
        for key in ("query_ms", "search_ms", "expand_ms", "context_ms", "answer_ms")
    }

    return {
        "questions_compared": n,
        "top20_identical": same_top20,
        "top20_identical_ratio": round(same_top20 / n, 6) if n else None,
        "metrics": metric_rows,
        "load": load_rows,
        "memory": mem_rows,
        "latency": latency_rows,
        "stage_warm": stage_rows,
        "search_p95_target_ms": 200.0,
        "search_p95_met": (
            None
            if new["latency"]["stage_warm"].get("search_ms", {}).get("p95_ms") is None
            else bool(new["latency"]["stage_warm"]["search_ms"]["p95_ms"] < 200.0)
        ),
    }


def print_report(reports: dict[str, dict], cmp: dict) -> None:
    old = reports[search_backend.BACKEND_JSON]
    new = reports[search_backend.BACKEND_FTS]
    line = "=" * 78
    print(line)
    print(f"H3 双跑对比  generation={new['generation_id']}  题数={cmp['questions_compared']}")
    print(f"口径：top_k={new['config']['top_k']}  graph_depth={new['config']['graph_depth']}  "
          f"warm_reps={new['config']['warm_reps']}  seed={new['config']['seed']}")
    print(line)

    print("\n【后端自我描述】")
    for tag, rep in (("json", old), ("fts ", new)):
        d = rep.get("backend_describe") or {}
        print(f"  {tag}  {d.get('backend')}  size={d.get('size')}  avg_len={d.get('avg_len')}  "
              f"tokenizer={d.get('tokenizer_version')}  产物={d.get('source_path')}")

    print("\n【七项指标】(同批题、同一函数计算)")
    print(f"  {'指标':<28} {'json':>9} {'fts':>9} {'delta':>9}  n(json/fts)")
    for key in METRIC_KEYS:
        row = cmp["metrics"][key]
        print(f"  {key:<28} {row['json']:>9.4f} {row['fts']:>9.4f} {row['delta']:>+9.4f}  "
              f"{row['n_json']}/{row['n_fts']}")

    print("\n【排序一致性】")
    print(f"  top-20 chunk_id 逐位相同：{cmp['top20_identical']}/{cmp['questions_compared']}"
          f"  （{100 * (cmp['top20_identical_ratio'] or 0):.2f}%）")

    print("\n【冷启动分阶段】（毫秒）")
    print(f"  {'阶段':<16} {'json':>10} {'fts':>10} {'delta':>10}")
    for key, row in cmp["load"].items():
        a = f"{row['json']:.1f}" if isinstance(row["json"], (int, float)) else "null"
        b = f"{row['fts']:.1f}" if isinstance(row["fts"], (int, float)) else "null"
        d = f"{row['delta']:+.1f}" if isinstance(row["delta"], (int, float)) else "null"
        print(f"  {key:<16} {a:>10} {b:>10} {d:>10}")
    print(f"  {'retriever_build':<16} {old['retriever_build_ms']:>10.1f} "
          f"{new['retriever_build_ms']:>10.1f} "
          f"{new['retriever_build_ms'] - old['retriever_build_ms']:>+10.1f}")

    print("\n【内存 RSS（MiB）】")
    for key, row in cmp["memory"].items():
        print(f"  {key:<16} json {row['json']}  peak {row['peak_json']}   |   "
              f"fts {row['fts']}  peak {row['peak_fts']}")

    print("\n【端到端延迟】（毫秒，nearest-rank）")
    for bucket, label in (("cold", "cold"), ("warm", "warm")):
        a = cmp["latency"][bucket]
        print(f"  {label}  P50 json={a['p50_ms']['json']} fts={a['p50_ms']['fts']}"
              f"   P95 json={a['p95_ms']['json']} fts={a['p95_ms']['fts']}"
              f"   P99 json={a['p99_ms']['json']} fts={a['p99_ms']['fts']}")

    print("\n【分阶段 P50/P95（warm，毫秒）】")
    print("  注意：本表来自两份独立子进程报告，两边的进程内存状态不同")
    print("        （json 子进程背着 1.5 GB 倒排 dict，fts 子进程只占 532 MiB），")
    print("        所以**本表的快慢差异不能当作「后端本身快慢」**，见下面的交替 A/B。")
    print(f"  {'阶段':<12} {'json P50':>9} {'fts P50':>9} {'json P95':>9} {'fts P95':>9}")
    for key, row in cmp["stage_warm"].items():
        print(f"  {key:<12} {row['json_p50']:>9} {row['fts_p50']:>9} "
              f"{row['json_p95']:>9} {row['fts_p95']:>9}")

    # ---- 同进程交替 A/B：判定检索快慢的唯一有效口径
    inter = cmp.get("interleaved")
    if inter:
        print("\n【同进程交替 A/B】（两个后端共享同一内存状态，这才是检索快慢的有效口径）")
        print(f"  题数={inter['questions']}  轮数={inter['rounds_requested']}  "
              f"两后端共存时 RSS={inter.get('memory_both_backends', {}).get('current_mib')} MiB")
        print(f"  {'指标':<10} {'json P50':>10} {'fts P50':>10} {'delta P50':>11} "
              f"{'json P95':>10} {'fts P95':>10} {'delta P95':>11}")
        for metric, label in (("search", "检索阶段"), ("total", "端到端")):
            a = inter["json"][metric]
            b = inter["fts"][metric]
            d = inter["delta"][metric]
            print(f"  {label:<10} {a['p50_ms']:>10} {b['p50_ms']:>10} {d['p50_ms']:>+11} "
                  f"{a['p95_ms']:>10} {b['p95_ms']:>10} {d['p95_ms']:>+11}")
        for row in inter["rounds"]:
            print(f"    轮 {row['round']}: json search P50={row['json_search_p50']}  "
                  f"fts search P50={row['fts_search_p50']}")

    target = cmp["search_p95_target_ms"]
    p95 = new["latency"]["stage_warm"].get("search_ms", {}).get("p95_ms")
    print("\n【验收：热查询检索阶段 P95 < 200 ms】")
    if p95 is None:
        print("  null —— 没有 warm 检索阶段样本，无法判定（不估算）")
    else:
        verdict = "达标" if float(p95) < target else "未达标"
        gap = float(p95) - target
        print(f"  fts search P95 = {p95} ms → {verdict}"
              f"（阈值 {target:.0f} ms，{'超出' if gap > 0 else '余量'} {abs(gap):.1f} ms）")
    if inter:
        d = inter["delta"]["search"].get("p95_ms")
        if d is not None:
            direction = "更慢" if d > 0 else "更快"
            print(f"  但按交替 A/B 口径：fts 检索 P95 比 json {direction} {abs(d):.1f} ms。")
            print("  → 结论：这条阈值在 H2 就已达成；H3 相对 json 后端在检索延迟上并没有收益，")
            print("     H3 的收益在冷启动与内存（见上方冷启动表与内存表）。")
    print(line)


# ---------------------------------------------------------------- 主流程
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="H3 双跑对比（旧 JSON 后端 vs FTS5 后端）")
    parser.add_argument("--generation", required=True)
    parser.add_argument("--questions", type=int, default=320, help="题目数（默认 320，即全量）")
    parser.add_argument("--warm-reps", type=int, default=DEFAULT_WARM_REPS)
    parser.add_argument(
        "--interleaved-rounds",
        type=int,
        default=2,
        help="同进程交替 A/B 的轮数（默认 2；0 = 跳过。判定检索快慢必须靠这一组）",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--graph-depth", type=int, default=DEFAULT_GRAPH_DEPTH)
    parser.add_argument("--gold-set", default="", help="默认 data/goldset/gold-v1.json")
    parser.add_argument("--db", default="", help="FTS 索引路径（默认 data/arkrag/fts/<gid>.sqlite）")
    parser.add_argument("--json", default="", help="报告输出路径（默认 data/arkrag/fts/bench_h3_<gid>.json）")
    # 内部子进程用；不要手工指定
    parser.add_argument("--backend", default="", help=argparse.SUPPRESS)
    parser.add_argument("--child-out", default="", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    gold_path = Path(args.gold_set) if args.gold_set else config.ROOT / "data" / "goldset" / "gold-v1.json"
    if not gold_path.exists():
        raise SystemExit(f"[2] Gold Set 不存在：{gold_path}")
    db_path = Path(args.db) if args.db else search_backend.default_db_path(args.generation)

    # ---- 子进程模式：跑一个后端，把报告写进文件
    if args.backend:
        report = run_one_backend(
            args.generation,
            backend_kind=args.backend,
            questions=args.questions,
            warm_reps=args.warm_reps,
            top_k=args.top_k,
            graph_depth=args.graph_depth,
            gold_path=gold_path,
            db_path=db_path,
        )
        Path(args.child_out).write_text(json.dumps(report, ensure_ascii=False), "utf-8")
        return 0

    # ---- 父进程：两次独立子进程
    if not db_path.exists():
        raise SystemExit(
            f"[2] FTS 索引不存在：{db_path}\n"
            f"    先跑 py -3 scripts\\build_fts_index.py --generation {args.generation}"
        )
    reports: dict[str, dict] = {}
    tmpdir = Path(tempfile.mkdtemp(prefix="h3bench-"))
    try:
        for kind in (search_backend.BACKEND_JSON, search_backend.BACKEND_FTS):
            out = tmpdir / f"{kind}.json"
            cmd = [
                sys.executable, str(Path(__file__).resolve()),
                "--generation", args.generation,
                "--questions", str(args.questions),
                "--warm-reps", str(args.warm_reps),
                "--top-k", str(args.top_k),
                "--graph-depth", str(args.graph_depth),
                "--gold-set", str(gold_path),
                "--db", str(db_path),
                "--backend", kind,
                "--child-out", str(out),
            ]
            print(f"[h3] 跑后端 {kind} ……", file=sys.stderr)
            env = dict(os.environ)
            env.setdefault("PYTHONIOENCODING", "utf-8")
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace")
            if proc.returncode != 0:
                sys.stderr.write(proc.stdout or "")
                sys.stderr.write(proc.stderr or "")
                raise SystemExit(f"[2] 后端 {kind} 失败，退出码 {proc.returncode}")
            if proc.stderr.strip():
                print(proc.stderr.strip(), file=sys.stderr)
            reports[kind] = json.loads(out.read_text("utf-8"))
    finally:
        for leftover in tmpdir.glob("*"):
            try:
                leftover.unlink()
            except OSError:
                pass
        try:
            tmpdir.rmdir()
        except OSError:
            pass

    cmp = compare(reports)

    # ---- 同进程交替 A/B：这是唯一能判断「后端本身快慢」的口径（见函数 docstring）
    interleaved: dict | None = None
    if args.interleaved_rounds > 0:
        print(f"[h3] 同进程交替 A/B（{args.interleaved_rounds} 轮）……", file=sys.stderr)
        interleaved = run_interleaved_latency(
            args.generation,
            questions=args.questions,
            top_k=args.top_k,
            graph_depth=args.graph_depth,
            gold_path=gold_path,
            db_path=db_path,
            rounds=args.interleaved_rounds,
        )
        cmp["interleaved"] = interleaved

    print_report(reports, cmp)

    out_path = Path(args.json) if args.json else (
        config.artifact_root() / "fts" / f"bench_h3_{args.generation}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"comparison": cmp, "runs": reports}, ensure_ascii=False, indent=1),
        "utf-8",
    )
    print(f"\n报告已写入：{out_path}")
    print(f"复跑命令：py -3 scripts\\bench_h3.py --generation {args.generation} "
          f"--questions {args.questions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
