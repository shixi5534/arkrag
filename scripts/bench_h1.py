"""`bench_h1` —— H 阶段 H1 性能基线：把"慢"量成可复跑的数字。

## 为什么要有这个脚本

计划 H1 原文只给了一个数：127.2 MB JSON、`json.loads` 约 23.88 秒、峰值约 692.4 MiB。
那是一个**孤立测量**——它没说测的是哪个文件、冷还是热、进程里还有没有别的东西，
所以它既不能被复现，也不能被用来判断优化有没有效。没有基线就不能声称"变快了"。

本脚本产出一份 JSON，字段即 H3 的验收对表：

| 组 | 字段 | 用途 |
|---|---|---|
| 加载 | `load.*_ms` 逐产物 | 冷启动成本拆到每个文件 |
| 内存 | `memory.peak_rss_mib` | 与 692.4 MiB 对比 |
| 延迟 | `latency.cold` / `latency.warm` 的 P50/P95/P99 | H3 目标「热查询 P95 < 200 ms」 |
| Top-K | `top_k[*]` | 多取多少块、多花多少时间 |
| 图扩展 | `expansion.*` | 节点/边实际用量 |
| Token | `tokens.*` | 单问题上下文 token；**成本不算**（见下） |

## 五条口径，写死在代码里

1. **冷查与热查分开报。** 第一次查询要付 page fault、dict 预热、`avg_len` 惰性计算
   的账；把它混进均值，会把"热查询 P95"这个指标做低。冷查 = 本进程对该题的第一遍，
   热查 = 其后各遍。两个数都报，不取平均。
2. **分位数用 nearest-rank，不用插值。** 样本少时线性插值会造出不存在的数；
   nearest-rank 的每个结果都是某个真实样本。同时报 `n`，让"P99 只有 1 个样本"
   这件事自己暴露出来。
3. **每问都记 `query_id` 与题目类别**，可以把慢点定位到题面类型，而不是只报一个总数。
4. **LLM Token 与成本：本机无 LLM 端点**，所以 `tokens.llm_*` 一律写 `null` 并注明
   原因。上下文 token 用建库同一套分词器估算（与 `retrieve.estimate_tokens` 一致，
   已在模块 docstring 声明不是真实 LLM 分词）。**不拿估算值冒充实测成本。**
5. **构建耗时取自真实 run 事件日志**（`runs/<run_id>/events.jsonl` 的 `at` 时间戳），
   不重跑一遍构建来"顺便测"——重跑会写新 Generation，污染产物目录。

## 用法

    py -3 scripts\\bench_h1.py                            # 默认 60 题、每热查 3 遍
    py -3 scripts\\bench_h1.py --questions 320 --warm-reps 5
    py -3 scripts\\bench_h1.py --generation g-...-4eafae --json reports\\bench_h1.json

## 退出码

* `0` —— 跑完（**慢也是 0**）。它是测量器不是门禁；把"慢"当失败会逼人先改阈值。
* `2` —— 硬错误：世代不存在/缺产物、Gold Set 缺失、参数非法。
"""
from __future__ import annotations

import argparse
import ctypes
import json
import random
import sys
import time
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from arkrag import config, release  # noqa: E402

import ask as ask_mod  # noqa: E402  （scripts/ask.py，不是包的一部分）


# ---------------------------------------------------------------- 内存
class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def rss_mib() -> dict:
    """当前与峰值 RSS（MiB）。Windows 用 `GetProcessMemoryInfo`，其它平台返回 null。

    不用 psutil：项目运行时承诺零第三方依赖（`requirements.txt` 只有测试用的
    pytest），为一个测量脚本引入运行时依赖会让"部署只需要标准库"这条失效。
    拿不到就写 `null`，**不猜**。
    """
    if sys.platform != "win32":
        return {"current_mib": None, "peak_mib": None, "source": "unsupported_platform"}
    try:
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessMemoryCounters), wintypes.DWORD]
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if not get_info(handle, ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error())
        mib = 1024 * 1024
        return {
            "current_mib": round(counters.WorkingSetSize / mib, 2),
            "peak_mib": round(counters.PeakWorkingSetSize / mib, 2),
            "source": "GetProcessMemoryInfo",
        }
    except Exception as exc:  # pragma: no cover - 平台相关
        return {"current_mib": None, "peak_mib": None, "source": f"error:{exc}"}


# ---------------------------------------------------------------- 分位数
def percentile(values: list[float], pct: float) -> float | None:
    """nearest-rank 分位数（ceil）。空样本返回 None，**不返回 0.0**。

    返回 0.0 会让"没测到"看起来像"极快"，这是本仓库已记录过的同族陷阱。
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    rank = max(1, min(len(ordered), int(-(-pct * len(ordered) // 100))))
    return round(ordered[rank - 1], 3)


def summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None,
                "min_ms": None, "max_ms": None, "mean_ms": None}
    return {
        "n": len(values),
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
        "mean_ms": round(sum(values) / len(values), 3),
    }


# ---------------------------------------------------------------- 加载计时
def timed_load(generation_id: str) -> tuple[dict, dict]:
    """逐个产物计时加载，返回 (loaded, load_timings)。

    复用 `ask.load_generation` 之外单独写这里，是因为它一次性把五件产物都读了，
    分不出谁慢。这里按同一份产物清单逐个读，**读法必须与 `ask.load_generation`
    一致**（同一函数、同一编码），否则量到的是另一条路径的成本。
    """
    from arkrag import index as index_mod

    gen = release.Generation(generation_id)
    if not gen.dir.exists():
        raise SystemExit(f"[2] 世代不存在：{generation_id}")
    manifest = gen.load()
    timings: dict = {}
    sizes: dict = {}

    t0 = time.perf_counter()
    graph = json.loads(gen.graph_path.read_text("utf-8"))
    timings["graph_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    sizes["graph_mib"] = round(gen.graph_path.stat().st_size / 1048576, 2)

    t0 = time.perf_counter()
    adjacency = json.loads(gen.adjacency_path.read_text("utf-8"))
    timings["adjacency_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    sizes["adjacency_mib"] = round(gen.adjacency_path.stat().st_size / 1048576, 2)

    t0 = time.perf_counter()
    assertions = ask_mod._read_jsonl(gen.assertion_path)
    timings["assertions_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    sizes["assertions_mib"] = round(gen.assertion_path.stat().st_size / 1048576, 2)

    t0 = time.perf_counter()
    idx, _meta = index_mod.load_index(gen.index_path)
    timings["index_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    sizes["index_mib"] = round(gen.index_path.stat().st_size / 1048576, 2)

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
    sizes["chunks_mib"] = round(chunks_path.stat().st_size / 1048576, 2)

    loaded = {
        "generation_id": generation_id,
        "generation": gen,
        "manifest": manifest,
        "graph": graph,
        "adjacency": adjacency,
        "assertions": assertions,
        "index": idx,
        "index_meta": _meta,
        "chunks": chunks,
    }
    timings["total_ms"] = round(sum(v for k, v in timings.items() if k.endswith("_ms")), 2)
    timings["artifact_mib"] = sizes
    return loaded, timings


def build_pipeline_timing(loaded: dict, *, top_k: int, graph_depth: int) -> tuple[dict, float]:
    """`Retriever` 构造本身也要计时：它建实体表/边表/断言索引，是冷启动的一部分。"""
    t0 = time.perf_counter()
    pipeline = ask_mod.build_pipeline(loaded, top_k=top_k, graph_depth=graph_depth)
    pipeline["generation_id"] = loaded["generation_id"]
    pipeline["entities"] = loaded["graph"].get("entities") or []
    return pipeline, round((time.perf_counter() - t0) * 1000.0, 2)


# ---------------------------------------------------------------- 构建耗时
def build_timings(generation_id: str) -> dict:
    """从真实 run 事件日志算各阶段耗时。**不重跑构建**（会写新 Generation）。"""
    gen = release.Generation(generation_id)
    manifest = gen.load()
    run_id = str(manifest.get("run_id") or "")
    if not run_id:
        return {"available": False, "reason": "manifest 无 run_id"}
    events_path = config.DATA_DIR / "runs" / run_id / "events.jsonl"
    if not events_path.exists():
        return {"available": False, "reason": f"找不到 {events_path}"}
    events: list[dict] = []
    with events_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not events:
        return {"available": False, "reason": "事件日志为空"}

    def _ts(ev: dict) -> float | None:
        raw = str(ev.get("at") or "")
        try:
            return time.mktime(time.strptime(raw, "%Y-%m-%dT%H:%M:%SZ"))
        except ValueError:
            return None

    stamps = [(str(ev.get("stage") or ""), str(ev.get("message") or ""), _ts(ev)) for ev in events]
    valid = [t for _, _, t in stamps if t is not None]
    total_s = round(max(valid) - min(valid), 2) if len(valid) >= 2 else None

    # 按 stage 汇总：同一 stage 首末事件的时间差
    by_stage: dict[str, dict] = {}
    for stage, message, ts in stamps:
        if ts is None:
            continue
        slot = by_stage.setdefault(stage, {"first": ts, "last": ts, "messages": []})
        slot["first"] = min(slot["first"], ts)
        slot["last"] = max(slot["last"], ts)
        if message:
            slot["messages"].append(message)
    stages = {
        stage: {
            "span_s": round(slot["last"] - slot["first"], 2),
            "steps": len(slot["messages"]),
            "first_message": slot["messages"][0] if slot["messages"] else "",
            "last_message": slot["messages"][-1] if slot["messages"] else "",
        }
        for stage, slot in by_stage.items()
    }
    return {
        "available": True,
        "run_id": run_id,
        "events_path": str(events_path),
        "event_count": len(events),
        "total_span_s": total_s,
        "by_stage_s": stages,
        "note": "时间戳精度为秒；total_span 是首末事件差，含进程启动前的等待，不等于纯计算耗时",
    }


# ---------------------------------------------------------------- 题目选择
def pick_questions(gold_path: Path, count: int, *, seed: int = 20261101) -> list[dict]:
    """按类别分层取题，与 `ask.evaluate` 同口径，保证两次跑的是同一批题。

    直接复用 `ask._stratified_sample`：这里再写一份配额逻辑就是同一判定写两处
    （`tests/test_single_source_of_truth.py` 记录的三次事故都是这个机理）。
    """
    if not gold_path.exists():
        raise SystemExit(f"[2] Gold Set 不存在：{gold_path}")
    doc = json.loads(gold_path.read_text("utf-8"))
    items = doc.get("items") if isinstance(doc, dict) else doc
    if not isinstance(items, list) or not items:
        raise SystemExit(f"[2] Gold Set 里没有 items：{gold_path}")
    chosen = ask_mod._stratified_sample(items, count, seed=seed)
    return [i for i in chosen if str(i.get("question") or "").strip()]


# ---------------------------------------------------------------- 主流程
def run(generation_id: str, *, questions: int, warm_reps: int, top_k: int,
        graph_depth: int, gold_path: Path) -> dict:
    mem_start = rss_mib()
    loaded, load_timings = timed_load(generation_id)
    mem_loaded = rss_mib()
    pipeline, retriever_ms = build_pipeline_timing(loaded, top_k=top_k, graph_depth=graph_depth)
    mem_ready = rss_mib()

    picked = pick_questions(gold_path, questions)

    cold: list[float] = []
    warm: list[float] = []
    per_stage_cold: dict[str, list[float]] = {}
    per_stage_warm: dict[str, list[float]] = {}
    rows: list[dict] = []
    nodes: list[int] = []
    edges: list[int] = []

    for item in picked:
        question = str(item.get("question") or "")
        cold_t: dict = {}
        ask_mod.ask_once(pipeline, question, timings=cold_t)
        cold.append(cold_t.get("total_ms", 0.0))
        for key in ("query_ms", "search_ms", "expand_ms", "context_ms", "answer_ms"):
            if key in cold_t:
                per_stage_cold.setdefault(key, []).append(cold_t[key])
        if "expand_nodes" in cold_t:
            nodes.append(cold_t["expand_nodes"])
            edges.append(cold_t["expand_edges"])

        reps: list[float] = []
        last: dict = {}
        for _ in range(warm_reps):
            warm_t: dict = {}
            ask_mod.ask_once(pipeline, question, timings=warm_t)
            reps.append(warm_t.get("total_ms", 0.0))
            last = warm_t
            for key in ("query_ms", "search_ms", "expand_ms", "context_ms", "answer_ms"):
                if key in warm_t:
                    per_stage_warm.setdefault(key, []).append(warm_t[key])
        # 热查单题取中位数：单次可能被 GC 撞一下，中位数比均值稳
        if reps:
            warm.append(percentile(reps, 50) or 0.0)
        rows.append(
            {
                "question_id": item.get("question_id"),
                "category": item.get("category"),
                "cold_total_ms": round(cold_t.get("total_ms", 0.0), 3),
                "warm_p50_ms": round(percentile(reps, 50) or 0.0, 3) if reps else None,
                "search_ms": round(last.get("search_ms", 0.0), 3) if last else None,
                "context_units": last.get("context_units"),
                "context_tokens": last.get("context_tokens"),
                "candidates": last.get("candidates"),
                "expand_nodes": cold_t.get("expand_nodes"),
                "expand_edges": cold_t.get("expand_edges"),
            }
        )

    # Top-K 成本：同一批题、不同 top_k，只测 search 一步（其它步骤与 top_k 无关）
    top_k_cost: dict[str, dict] = {}
    for k in sorted({5, 20, 50, 100, top_k}):
        times: list[float] = []
        cands: list[int] = []
        for item in picked[: min(len(picked), 20)]:
            qa = ask_mod.query_mod.analyze_query(
                str(item.get("question") or ""),
                entities=pipeline["entities"],
                aliases=pipeline["retriever"].graph.get("alias_index"),
                vocab=ask_mod.vocab,
            )
            t0 = time.perf_counter()
            found = pipeline["retriever"].search(qa, top_k=k)
            times.append((time.perf_counter() - t0) * 1000.0)
            cands.append(int(found["stats"].get("candidates") or 0))
        top_k_cost[str(k)] = {
            "n": len(times),
            "p50_ms": percentile(times, 50),
            "p95_ms": percentile(times, 95),
            "mean_candidates": round(sum(cands) / len(cands), 1) if cands else None,
        }

    mem_end = rss_mib()

    report = {
        "h_stage": "H1",
        "measured_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generation_id": generation_id,
        "generation_status": str(loaded["manifest"].get("status") or ""),
        "graph_schema": str(loaded["manifest"].get("graph_schema") or ""),
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "config": {
            "top_k": top_k,
            "graph_depth": graph_depth,
            "questions": len(picked),
            "warm_reps": warm_reps,
            "gold_set": str(gold_path),
            "category_mix": _category_mix(picked),
        },
        "load": load_timings,
        "retriever_build_ms": retriever_ms,
        "memory": {
            "at_start": mem_start,
            "after_load": mem_loaded,
            "after_pipeline": mem_ready,
            "at_end": mem_end,
            "plan_reference_peak_mib": 692.4,
            "note": "peak_mib 是进程生命周期高水位，含解释器与库；不与单文件 23.88s 的孤立测量直接相减",
        },
        "latency": {
            "cold": summarize(cold),
            "warm": summarize(warm),
            "stage_cold": {k: summarize(v) for k, v in sorted(per_stage_cold.items())},
            "stage_warm": {k: summarize(v) for k, v in sorted(per_stage_warm.items())},
            "note": "cold=本进程对该题首遍；warm=其后各遍取中位数；分位数为 nearest-rank",
        },
        "top_k": top_k_cost,
        "expansion": {
            "nodes_p50": percentile([float(n) for n in nodes], 50),
            "nodes_p95": percentile([float(n) for n in nodes], 95),
            "nodes_max": max(nodes) if nodes else None,
            "edges_p50": percentile([float(e) for e in edges], 50),
            "edges_p95": percentile([float(e) for e in edges], 95),
            "edges_max": max(edges) if edges else None,
            "budget": dict(pipeline["retriever"].budget),
        },
        "tokens": {
            "context_tokens_p50": percentile(
                [float(r["context_tokens"]) for r in rows if r.get("context_tokens") is not None], 50
            ),
            "context_tokens_p95": percentile(
                [float(r["context_tokens"]) for r in rows if r.get("context_tokens") is not None], 95
            ),
            "context_units_p50": percentile(
                [float(r["context_units"]) for r in rows if r.get("context_units") is not None], 50
            ),
            "llm_prompt_tokens": None,
            "llm_completion_tokens": None,
            "llm_cost_usd": None,
            "llm_unavailable_reason": (
                "本机无 LLM 端点（ARKRAG_LLM_BASE_URL / ARKRAG_LLM_MODEL 为空，"
                "arkrag.llm 缺配置直接抛 RuntimeError），回答器为抽取式 ExtractiveAnswerer；"
                "上下文 token 由建库同一套分词器估算，不是真实 LLM 分词，不能当成本依据"
            ),
        },
        "build": build_timings(generation_id),
        "per_question": rows,
    }
    return report


def _category_mix(items: list[dict]) -> dict:
    out: dict[str, int] = {}
    for item in items:
        key = str(item.get("category") or "")
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def print_report(report: dict) -> None:
    print("=" * 78)
    print(f"H1 性能基线  世代 {report['generation_id']}  Python {report['python_version']}")
    print(f"题量 {report['config']['questions']}  热查重复 {report['config']['warm_reps']}  "
          f"top_k {report['config']['top_k']}  graph_depth {report['config']['graph_depth']}")
    print("类别：" + "  ".join(f"{k}={v}" for k, v in report["config"]["category_mix"].items()))

    ld = report["load"]
    sizes = ld.get("artifact_mib") or {}
    print("-" * 78)
    print("加载（逐个产物）")
    for key in ("graph_ms", "adjacency_ms", "assertions_ms", "index_ms", "chunks_ms"):
        mib = sizes.get(key.replace("_ms", "_mib"), "?")
        print(f"  {key:<16} {ld.get(key):>9.1f} ms   {mib} MiB")
    print(f"  {'合计':<16} {ld.get('total_ms'):>9.1f} ms")
    print(f"  Retriever 构造   {report['retriever_build_ms']:>9.1f} ms")

    mem = report["memory"]
    print("-" * 78)
    print(f"内存  起始 {mem['at_start']['current_mib']} MiB  "
          f"加载后 {mem['after_load']['current_mib']} MiB  "
          f"管线就绪 {mem['after_pipeline']['current_mib']} MiB")
    print(f"      峰值高水位 {mem['at_end']['peak_mib']} MiB  （计划参照值 {mem['plan_reference_peak_mib']} MiB）")

    lat = report["latency"]
    print("-" * 78)
    print("延迟（ms，nearest-rank）")
    for name in ("cold", "warm"):
        s = lat[name]
        print(f"  {name:<6} n={s['n']:<4} P50 {s['p50_ms']}  P95 {s['p95_ms']}  "
              f"P99 {s['p99_ms']}  min {s['min_ms']}  max {s['max_ms']}")
    print("  分阶段（热）：" + "  ".join(
        f"{k.replace('_ms','')} P95={v['p95_ms']}" for k, v in lat["stage_warm"].items()
    ))

    print("-" * 78)
    print("Top-K 成本（仅 search 一步）")
    for k, v in report["top_k"].items():
        print(f"  k={k:<4} n={v['n']:<3} P50 {v['p50_ms']}  P95 {v['p95_ms']}  "
              f"平均候选 {v['mean_candidates']}")

    ex = report["expansion"]
    print("-" * 78)
    print(f"图扩展  节点 P50 {ex['nodes_p50']} P95 {ex['nodes_p95']} max {ex['nodes_max']}   "
          f"边 P50 {ex['edges_p50']} P95 {ex['edges_p95']} max {ex['edges_max']}")

    tk = report["tokens"]
    print(f"Token   上下文 P50 {tk['context_tokens_p50']} P95 {tk['context_tokens_p95']}   "
          f"块数 P50 {tk['context_units_p50']}")
    print("        LLM token/成本：不可测（无 LLM 端点，抽取式回答器）")

    b = report["build"]
    print("-" * 78)
    if b.get("available"):
        print(f"构建  run {b['run_id']}  总跨度 {b['total_span_s']} s  （{b['event_count']} 条事件）")
        for stage, v in b["by_stage_s"].items():
            print(f"  {stage:<12} {v['span_s']:>8.1f} s  {v['steps']} 步  {v['last_message'][:40]}")
    else:
        print(f"构建  不可得：{b.get('reason')}")
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="H1 性能基线")
    parser.add_argument("--generation", default="", help="世代 id，默认取最新 ark-rag-3")
    parser.add_argument("--questions", type=int, default=60, help="题量，默认 60")
    parser.add_argument("--warm-reps", type=int, default=3, help="热查每題重复次数，默认 3")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--graph-depth", type=int, default=1)
    parser.add_argument("--gold-set", default="", help="默认 data/goldset/gold-v1.json")
    parser.add_argument("--json", default="", help="把完整报告写到该路径")
    args = parser.parse_args(argv)

    if args.questions < 1:
        raise SystemExit("[2] --questions 必须 >= 1")
    if args.warm_reps < 1:
        raise SystemExit("[2] --warm-reps 必须 >= 1")
    if args.top_k < 1:
        raise SystemExit("[2] --top-k 必须 >= 1")

    gid = args.generation or ask_mod._latest_identity_generation()
    gold = Path(args.gold_set) if args.gold_set else ROOT / "data" / "goldset" / "gold-v1.json"
    if not gold.is_absolute():
        gold = (ROOT / gold).resolve()

    report = run(
        gid,
        questions=args.questions,
        warm_reps=args.warm_reps,
        top_k=args.top_k,
        graph_depth=args.graph_depth,
        gold_path=gold,
    )
    print_report(report)
    if args.json:
        out = Path(args.json)
        if not out.is_absolute():
            out = (ROOT / out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"完整报告已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
