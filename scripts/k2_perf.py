# -*- coding: utf-8 -*-
"""`k2_perf` —— K2 构建顺序里最后那个漏项「性能评测」：把两代放在同一条尺子上量。

## 为什么要有这个脚本

`docs/升级计划_GraphRAG架构.md` 的 K2 构建顺序要求「… → Gold Set 评测 → **性能评测**
→ 候选 READY」，而这一步从来没做过：`docs/k阶段实测.md` 里「性能」「bench」「P95」
各 0 处，只有 2 处 `ms/题` 说的是「耗时不是可复现数字，15–44 ms/题浮动、不可复现」。
DoD 的「性能与运维」块要求「不再依赖每次启动解析 127 MB 巨型 JSON；**查询和构建性能
有持续记录**」——没有这两个数，这句话无法被核对。

H 阶段已经留下了两把尺子：`scripts/bench_h1.py`（单代冷/热延迟、峰值 RSS、加载分解）
与 `scripts/bench_h3.py`（后端对比）。本脚本**复用**它们：

* `bench_h1.percentile` / `bench_h1.summarize` —— nearest-rank 分位数，这里绝不另造一套；
* `bench_h1.rss_mib` —— `GetProcessMemoryInfo` 取峰值 RSS，非 Windows 写 `null` 不猜；
* `bench_h1.build_timings` —— 构建耗时取自真实 run 事件日志，**不重跑构建**；
* `bench_h1.pick_questions` —— 与 `ask.evaluate` 同一套分层抽样，两代吃同一批题；
* `bench_h3` 的口径 A：**跨子进程**测量。冷启动成本与峰值 RSS 只有在全新进程里才是真的；
  同一进程先后载入两代，第二代会踩第一代的页缓存、峰值 RSS 也会互相叠加，
  量出来的是"我们的测量方法"而不是"这一代有多重"。

## 为什么门必须分成两类（本轮最要紧的一条）

单题耗时在本机**不可复现到具体值**：同一份代码、同一批题，重跑一次 P95 就可能挪几十
毫秒（`docs/k阶段实测.md` 已记 15–44 ms/题 的浮动），此刻还有别的进程在抢 CPU 与内存。
把这种数字做成门线（"候选不许比上一代慢 5%"）等于让门**随机变红**，而随机红的门活不过
两周——它会被顺手放宽，然后连"慢了十倍"也拦不住。所以：

1. **硬门（`checks`）**：只放**确定性事实**。索引与加载字节、索引绑定的摘要、
   冷加载走的函数、峰值 RSS 上界、每代报告字段完整性。这些数字不随机器忙不忙变，
   条件不成立时必红。
2. **申报项（`declared`）**：冷/热 P50/P95/P99、加载耗时、构建耗时。**只记录、不判红**，
   `enforced=false` 写在字段里，并明确它不进 `all_passed`。理由：它是噪声，
   做成门就是让门随机红；但持续记录它才能回答"变快了没有"。
3. **兜底门（`p95_backstop`）**：唯一一条碰时序的门，阈值取 **2 倍**——只为抓
   "慢了十倍"这种灾难性回退。不取 1.1 倍的理由见 `P95_BACKSTOP_FACTOR` 旁注释。

另外三条纪律：

* **每一项都打印分母**（`n=` 样本数、题数、字节数），没有分母的比率不许出现；
* **不拿估算值冒充实测**：本机无 LLM 端点，`llm_*` 一律 `null` 并注明原因（照 `bench_h1`）；
* **字段缺失返回 1，不当 0 处理**：`report_fields` 门逐字段点名缺了什么，缺就是红。

## 用法

    py -3 scripts\\k2_perf.py                        # 候选 vs 上一代，默认 40 题
    py -3 scripts\\k2_perf.py --questions 120 --warm-reps 3
    py -3 scripts\\k2_perf.py --generation g-... --baseline g-...

## 退出码

* `0` —— 全部门成立（申报项不参与判定）；
* `1` —— 有门不成立；
* `2` —— 硬错误：世代不存在/缺产物、Gold Set 缺失、子进程失败、参数非法。

## 边界

* 报告里的时序数字**只用于自测**。正式读数需在**无并发**时重跑：本仓库同一时刻可能有
  别的进程在跑 pytest 或建库，P95 会被它们推高。
* 本脚本不写 Generation、不改产物，只读 `data/arkrag/generations/<gid>` 与 run 事件日志。
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
# Windows 控制台默认 cp936，不重配置会把中文写成乱码字节，
# 且 `> file` 重定向后文件不是 UTF-8（本项目脚本一律要求输出可原样粘贴进文档）。
# 守卫是必要的：pytest 捕获期间 `sys.stdout` 可能是 `EncodedFile`/`CaptureFixture`
# 这类不可重配置的对象，直接调 `.reconfigure()` 会 AttributeError。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from arkrag import atomic, config, index as index_mod, pipeline, release  # noqa: E402

import ask as ask_mod  # noqa: E402
import bench_h1  # noqa: E402  （口径来源：percentile / summarize / rss_mib / build_timings / pick_questions）

CANDIDATE_GID = "g-20261103T132152Z-06a741"
BASELINE_GID = "g-20261031T035231Z-4eafae"
REPORT_PATH = ROOT / "data" / "k2_perf.json"
TOOL = "scripts/k2_perf.py"

# ---------------------------------------------------------------- 门线常量
#
# 峰值 RSS 上界。实测参照：`data/bench_h1_4eafae.json` 记上一代**单代**满载峰值
# 1717.59 MiB（含解释器与库）。4096 MiB 给它 2.4 倍余量——目的是抓"把 127 MB JSON
# 又整份 `json.loads` 了一遍"这种量级事故（那会直接多出一份 ~1.5 GiB），
# 而不是把正常波动做成门。
RSS_CEILING_MIB = 4096.0
# 候选峰值不许比上一代高 1.5 倍以上：同一份加载代码、两代产物字节数相差不到 5%，
# 1.5 倍已经不是波动而是换了加载路径。
RSS_RATIO_CEILING = 1.5
# 唯一一条碰时序的兜底门。为什么是 2 倍而不是 1.1 倍：
#   a) 同一代重跑，热查 P95 实测就能挪几十毫秒（本机 15–44 ms/题 的既有记载）；
#   b) 本仓库同一时刻常有别的子代理在跑 pytest/建库，P95 会被推高；
#   c) 门要抓的是"慢了十倍"这种灾难性回退——1.1～2 倍之间的差异不可归因，
#      做窄了只能逼人调阈值。
# 2 倍是能容忍 a/b 的最小整数倍，同时 10 倍级回退必红。
P95_BACKSTOP_FACTOR = 2.0
# 两代 `index/lexical.json` 的实测字节数。这是**产物身份事实**，不是估算：
# 世代 id 里带时间戳，重建不会改到同一个 gid 上。数值变了说明产物被替换过，
# 必须重新解释（`k2_hard_gates` 的第 5 条同族）。
EXPECTED_INDEX_BYTES = {
    CANDIDATE_GID: 106_685_068,
    BASELINE_GID: 127_244_898,
}
# 每代报告必须带的字段（缺一个就红，绝不静默当 0）。
MEASURE_FIELDS = (
    "generation_id",
    "graph_schema",
    "load_total_ms",
    "pipeline_ms",
    "index_tokenizer",
    "artifact_bytes",
    "chunks_path",
    "questions",
    "category_mix",
    "cold",
    "warm",
    "memory",
    "tokens",
)
LATENCY_FIELDS = ("n", "p50_ms", "p95_ms", "p99_ms")


class PerfError(RuntimeError):
    """硬错误：世代不存在、缺产物、子进程失败。对应退出码 2。"""


# ---------------------------------------------------------------- 子进程
def _spawn(child_args: list[str], out_path: Path) -> dict:
    """起一个子进程做测量，读回它写的 JSON。

    口径 A（跨子进程）在这里落地：每代一次全新进程，冷启动与峰值 RSS 才可比。
    子进程 stdout 用 UTF-8，避免中文在 cp936 管道里被替换成乱码字节。
    """
    cmd = [sys.executable, str(Path(__file__).resolve()), *child_args, "--out", str(out_path)]
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=env)
    if proc.returncode != 0:
        raise PerfError(
            f"子进程失败 rc={proc.returncode}：{' '.join(child_args)}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    if not out_path.exists():
        raise PerfError(f"子进程没写出 {out_path}：{' '.join(child_args)}")
    return json.loads(out_path.read_text("utf-8"))


# ---------------------------------------------------------------- 测量：整代
def measure_one(generation_id: str, *, questions: int, warm_reps: int, top_k: int,
                graph_depth: int, gold_path: Path) -> dict:
    """在一个全新进程里量一代：加载 → 建管线 → 冷查 → 热查。

    读法必须与生产链路一致：走 `pipeline.load_generation()`（它自己会核
    graph_schema、缺产物早退、并按 `resolve_chunks()` 证明块正文属于这一代），
    而不是自己挑文件——消费者读哪份块只有那一处定义。
    """
    gen = release.Generation(generation_id)
    if not gen.dir.exists():
        raise PerfError(f"世代不存在：{generation_id}（{gen.dir}）")
    manifest = gen.load()

    mem_start = bench_h1.rss_mib()
    t0 = time.perf_counter()
    loaded = pipeline.load_generation(generation_id)
    load_total_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    mem_loaded = bench_h1.rss_mib()

    t0 = time.perf_counter()
    pipe = pipeline.build_pipeline(loaded, top_k=top_k, graph_depth=graph_depth)
    pipeline_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    mem_ready = bench_h1.rss_mib()

    chunks_path = pipeline.resolve_chunks(gen, manifest)
    artifact_bytes = {
        "graph": gen.graph_path.stat().st_size,
        "adjacency": gen.adjacency_path.stat().st_size,
        "assertions": gen.assertion_path.stat().st_size,
        "index": gen.index_path.stat().st_size,
        "chunks": chunks_path.stat().st_size,
    }
    index_meta = loaded.get("index_meta") or {}

    picked = bench_h1.pick_questions(gold_path, questions)
    if not picked:
        raise PerfError(f"没取到任何题目：{gold_path}")

    cold: list[float] = []
    warm: list[float] = []
    rows: list[dict] = []
    for item in picked:
        question = str(item.get("question") or "")
        cold_t: dict = {}
        pipeline.ask_once(pipe, question, timings=cold_t)
        cold.append(float(cold_t.get("total_ms") or 0.0))
        reps: list[float] = []
        for _ in range(warm_reps):
            warm_t: dict = {}
            pipeline.ask_once(pipe, question, timings=warm_t)
            reps.append(float(warm_t.get("total_ms") or 0.0))
        # 热查单题取中位数：单次可能被 GC 撞一下，中位数比均值稳（同 bench_h1）
        warm.append(bench_h1.percentile(reps, 50) or 0.0)
        rows.append({"question_id": item.get("question_id"),
                     "category": item.get("category"),
                     "cold_total_ms": round(cold[-1], 3),
                     "warm_p50_ms": round(warm[-1], 3)})

    mem_end = bench_h1.rss_mib()
    return {
        "generation_id": generation_id,
        "graph_schema": str(manifest.get("graph_schema") or ""),
        "generation_status": str(manifest.get("status") or ""),
        "load_total_ms": load_total_ms,
        "pipeline_ms": pipeline_ms,
        "index_tokenizer": str(index_meta.get("tokenizer_version") or ""),
        "index_chunk_count": index_meta.get("chunk_count"),
        "artifact_bytes": artifact_bytes,
        "chunks_path": config.rel_to_root(chunks_path),
        "questions": len(picked),
        "category_mix": bench_h1._category_mix(picked),
        "cold": bench_h1.summarize(cold),
        "warm": bench_h1.summarize(warm),
        "warm_reps": warm_reps,
        "top_k": top_k,
        "graph_depth": graph_depth,
        "memory": {
            "at_start": mem_start,
            "after_load": mem_loaded,
            "after_pipeline": mem_ready,
            "at_end": mem_end,
        },
        "tokens": {
            "context_tokens_p50": None,
            "context_tokens_p95": None,
            "llm_prompt_tokens": None,
            "llm_completion_tokens": None,
            "llm_cost_usd": None,
            "llm_unavailable_reason": (
                "本机无 LLM 端点（ARKRAG_LLM_BASE_URL / ARKRAG_LLM_MODEL 为空，"
                "arkrag.llm 缺配置直接抛 RuntimeError），回答器是抽取式 ExtractiveAnswerer；"
                "拿估算值当成本依据就是造假，故一律 null（同 bench_h1 口径 4）"
            ),
        },
        "per_question": rows,
        "note": "本进程为该世代全新进程（口径 A）；时序数字只用于自测，正式读数需在无并发时重跑",
    }


# ---------------------------------------------------------------- 测量：索引路径
def probe_index_path(generation_id: str) -> dict:
    """只载入 `index/lexical.json`，并用 audit hook **证明**读了哪些文件。

    为什么要证明而不是声明：DoD 要求「不再依赖每次启动解析 127 MB 巨型 JSON」。
    「我们有 `load_index()`」是声明；「这一跑真的只打开了 `lexical.json`、
    一个字节都没碰 `graph/graph.json`」才是证据。`sys.addaudithook` 能看见进程里
    每一次 `open`，若将来有人把 `load_index` 改回解析 graph.json，这条门就红。
    """
    gen = release.Generation(generation_id)
    if not gen.dir.exists():
        raise PerfError(f"世代不存在：{generation_id}（{gen.dir}）")
    if not gen.index_path.exists():
        raise PerfError(f"缺索引：{gen.index_path}")

    opened: list[str] = []

    def hook(event: str, args: tuple) -> None:
        if event != "open" or not args:
            return
        first = args[0]
        if not isinstance(first, (str, bytes, os.PathLike)):
            return  # 文件描述符形态，不是路径，与本次证明无关
        try:
            opened.append(os.fsdecode(first))
        except (TypeError, ValueError):
            return  # 无法解码的路径记不进证据，但也不许中断被观测的调用

    sys.addaudithook(hook)

    t0 = time.perf_counter()
    idx, meta = index_mod.load_index(gen.index_path)
    index_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    index_peak = bench_h1.rss_mib()
    # 回读摘要：证明载入的**内容**就是 Manifest 记录那一份，而不是某个同名旧文件。
    # 用 probe_digest（10 条固定探针的 top-10 排序）而不是 content_digest：
    # 后者实测单代 16.2 秒，只为绑定身份不值当；前者 0.3 秒且同样能变。
    recomputed_probe = index_mod.probe_digest(idx)

    def norm(p: str) -> str:
        return os.path.normcase(os.path.abspath(p))

    json_paths = sorted({norm(p) for p in opened if p.lower().endswith((".json", ".jsonl"))})
    json_bytes = 0
    for p in json_paths:
        try:
            json_bytes += os.path.getsize(p)
        except OSError:
            json_bytes += 0  # 已消失的文件不计入字节数；它仍留在路径清单里被点名

    manifest = gen.load()
    return {
        "generation_id": generation_id,
        "index_path": config.rel_to_root(gen.index_path),
        "index_bytes": gen.index_path.stat().st_size,
        "graph_path": config.rel_to_root(gen.graph_path),
        "index_ms": index_ms,
        "index_peak_mib": index_peak.get("peak_mib"),
        "opened_json_paths": [config.rel_to_root(Path(p)) for p in json_paths],
        "opened_json_count": len(json_paths),
        "opened_json_bytes": json_bytes,
        "opened_lexical": norm(str(gen.index_path)) in json_paths,
        "opened_graph": norm(str(gen.graph_path)) in json_paths,
        "doc_count": len(idx.doc_id),
        "term_count": len(idx.postings),
        "tokenizer_version": str(meta.get("tokenizer_version") or "t1"),
        "meta_chunk_count": meta.get("chunk_count"),
        "meta_probe_digest": str(meta.get("probe_digest") or ""),
        "manifest_probe_digest": str(manifest.get("probe_digest") or ""),
        "recomputed_probe_digest": recomputed_probe,
        "note": "index_ms 是全新进程里首次读取的耗时（页缓存冷）；audit hook 在整个进程内有效，进程结束即消失",
    }


# ---------------------------------------------------------------- 判据
def check_generations_ok(mc: dict, mb: dict) -> tuple[bool, str, dict]:
    """两代必须是当前链路能加载的形态，且四件核心产物齐备非空。"""
    problems: list[str] = []
    for tag, m in (("候选", mc), ("上一代", mb)):
        gid = m.get("generation_id")
        if m.get("graph_schema") != config.GRAPH_SCHEMA_IDENTITY:
            problems.append(f"{tag} {gid} graph_schema={m.get('graph_schema')} 不是 "
                            f"{config.GRAPH_SCHEMA_IDENTITY}")
        if m.get("generation_status") != release.READY:
            problems.append(f"{tag} {gid} status={m.get('generation_status')} 不是 {release.READY}")
        for name, size in (m.get("artifact_bytes") or {}).items():
            if not isinstance(size, int) or size <= 0:
                problems.append(f"{tag} {gid} 产物 {name} 字节数={size}（应为正）")
    if mc.get("generation_id") == mb.get("generation_id"):
        problems.append(f"候选与上一代是同一个世代 {mc.get('generation_id')}，对比无意义")
    ok = not problems
    detail = (f"候选 {mc.get('generation_id')} schema={mc.get('graph_schema')} "
              f"status={mc.get('generation_status')} / 上一代 {mb.get('generation_id')} "
              f"schema={mb.get('graph_schema')} status={mb.get('generation_status')} / "
              f"5 件产物均 > 0 字节：{ok}")
    if problems:
        detail = "；".join(problems)
    return ok, detail, {
        "candidate": {"generation_id": mc.get("generation_id"),
                      "graph_schema": mc.get("graph_schema"),
                      "status": mc.get("generation_status")},
        "baseline": {"generation_id": mb.get("generation_id"),
                     "graph_schema": mb.get("graph_schema"),
                     "status": mb.get("generation_status")},
        "problems": problems,
    }


def check_index_size_delta(pc: dict, pb: dict, mc: dict, mb: dict) -> tuple[bool, str, dict]:
    """索引规模的事实对比：候选更小，且这件事被成因解释得住（t1→t2）。

    这是确定性事实，不随机器忙不忙变。三个方向同时核：
      1. 候选 `lexical.json` 字节数 < 上一代；
      2. 候选文档数（块数）**不比上一代少**——不然"文件小了"可能只是"内容少了"；
      3. 候选倒排词条数 < 上一代——成因是 t2 分词产出更少更长的词条。
    另外核两代字节数与已记录的产物身份一致，防止"换了文件还当同一代对比"。
    """
    cb, bb = pc["index_bytes"], pb["index_bytes"]
    shrink_pct = round((1 - cb / bb) * 100.0, 2) if bb else None
    problems: list[str] = []
    if not cb < bb:
        problems.append(f"候选 {cb:,} B 不小于上一代 {bb:,} B（方向不符）")
    if not pc["doc_count"] >= pb["doc_count"]:
        problems.append(f"候选文档数 {pc['doc_count']} < 上一代 {pb['doc_count']}："
                        "文件变小可能来自内容变少，而不是分词变省")
    if not pc["term_count"] < pb["term_count"]:
        problems.append(f"候选倒排词条 {pc['term_count']:,} 不少于上一代 "
                        f"{pb['term_count']:,}：与 t1→t2 的成因不符")
    for tag, pc_or_pb in (("候选", pc), ("上一代", pb)):
        want = EXPECTED_INDEX_BYTES.get(pc_or_pb["generation_id"])
        got = pc_or_pb["index_bytes"]
        if want is not None and got != want:
            problems.append(f"{tag} lexical.json 实测 {got:,} B != 已记录 {want:,} B"
                            "（产物身份变化，需重新解释）")
    if (mc.get("index_chunk_count") != pc["doc_count"]
            or mb.get("index_chunk_count") != pb["doc_count"]):
        problems.append(f"索引 meta.chunk_count 与实际文档数不符：候选 "
                        f"{mc.get('index_chunk_count')}/{pc['doc_count']}，上一代 "
                        f"{mb.get('index_chunk_count')}/{pb['doc_count']}")
    ok = not problems
    detail = (f"lexical.json 候选 {cb:,} B → 上一代 {bb:,} B，缩小 {shrink_pct}%"
              f"（n=2 代）；倒排词条 {pc['term_count']:,} ← {pb['term_count']:,}"
              f"（{round((1 - pc['term_count'] / pb['term_count']) * 100.0, 2)}% 更少，"
              f"成因 t1→t2）；文档数 {pc['doc_count']} ← {pb['doc_count']}"
              f"（+{pc['doc_count'] - pb['doc_count']} 块，文件反而更小）；"
              f"每千词条字节 候选 {round(cb / pc['term_count'] / 1000, 3)} KiB vs "
              f"上一代 {round(bb / pb['term_count'] / 1000, 3)} KiB")
    if problems:
        detail = "；".join(problems)
    return ok, detail, {
        "candidate_bytes": cb, "baseline_bytes": bb, "shrink_pct": shrink_pct,
        "candidate_term_count": pc["term_count"], "baseline_term_count": pb["term_count"],
        "candidate_doc_count": pc["doc_count"], "baseline_doc_count": pb["doc_count"],
        "candidate_tokenizer": pc["tokenizer_version"],
        "baseline_tokenizer": pb["tokenizer_version"],
        "expected_bytes": EXPECTED_INDEX_BYTES, "problems": problems,
    }


def check_cold_load_path(pc: dict, pb: dict) -> tuple[bool, str, dict]:
    """冷加载必须走 `index.load_index()`，不许整份 `json.loads(graph.json)`。

    证据是 audit hook 抓到的真实 `open` 清单：只应有 `index/lexical.json` 一份，
    读入字节数应恰好等于该文件字节数，且 `graph/graph.json` **一次都没被打开**。
    两代都核——上一代是"如果换回旧读法会不会偷偷解析图"的对照。
    """
    problems: list[str] = []
    for tag, p in (("候选", pc), ("上一代", pb)):
        if not p["opened_lexical"]:
            problems.append(f"{tag} 没有打开 {p['index_path']}（载入路径变了）")
        if p["opened_graph"]:
            problems.append(f"{tag} 打开了 {p['graph_path']}："
                            "冷加载回到了「整份解析 graph.json」的老路")
        if p["opened_json_bytes"] != p["index_bytes"]:
            problems.append(f"{tag} 打开了 {p['opened_json_count']} 份 JSON、共 "
                            f"{p['opened_json_bytes']:,} B，而 lexical.json 是 "
                            f"{p['index_bytes']:,} B（差 "
                            f"{p['opened_json_bytes'] - p['index_bytes']:,} B）")
    ok = not problems

    def one(p: dict) -> str:
        return (f"{p['generation_id'][-6:]} 打开 {p['opened_json_count']} 份 JSON / "
                f"{p['opened_json_bytes']:,} B（lexical.json {p['index_bytes']:,} B）")

    detail = (f"audit hook 实测：{one(pc)}；{one(pb)}；"
              f"graph.json 打开次数 候选 {int(pc['opened_graph'])} / 上一代 {int(pb['opened_graph'])}"
              f"（n=2 代 × 1 次冷加载）")
    if problems:
        detail = "；".join(problems)
    return ok, detail, {
        "candidate": {k: pc[k] for k in ("opened_json_paths", "opened_json_count",
                                         "opened_json_bytes", "index_bytes",
                                         "opened_lexical", "opened_graph")},
        "baseline": {k: pb[k] for k in ("opened_json_paths", "opened_json_count",
                                        "opened_json_bytes", "index_bytes",
                                        "opened_lexical", "opened_graph")},
        "problems": problems,
    }


def check_index_binding(pc: dict, pb: dict) -> tuple[bool, str, dict]:
    """载入的索引内容必须就是 Manifest 记录的那一份（回读摘要比对）。

    重算 `probe_digest`（10 条固定探针的 top-10 排序序列摘要）与两处记录对表：
    索引 meta 里写的、Manifest 里写的。三者不一致说明盘上这份文件不是这一代的。
    判据只吃传入的字典、**不读盘**——这样测试能用合成数据证明它会红。
    """
    problems: list[str] = []
    for tag, p in (("候选", pc), ("上一代", pb)):
        recomputed = p.get("recomputed_probe_digest")
        in_meta = p.get("meta_probe_digest") or ""
        want_manifest = p.get("manifest_probe_digest") or ""
        if not recomputed:
            problems.append(f"{tag} 未重算 probe_digest（索引载入失败？）")
        if not in_meta:
            problems.append(f"{tag} 索引 meta 缺 probe_digest（旧索引形态，无法绑定）")
        elif recomputed and recomputed != in_meta:
            problems.append(f"{tag} 重算 probe_digest={recomputed[:16]} != 索引 meta "
                            f"{in_meta[:16]}")
        if not want_manifest:
            problems.append(f"{tag} Manifest 缺 probe_digest（无期望值可比）")
        elif in_meta and want_manifest != in_meta:
            problems.append(f"{tag} 索引 meta probe_digest={in_meta[:16]} != Manifest "
                            f"{want_manifest[:16]}")
    ok = not problems
    detail = (f"probe_digest 重算一致：候选 {pc.get('recomputed_probe_digest', '')[:16]}、"
              f"上一代 {pb.get('recomputed_probe_digest', '')[:16]}"
              f"（n=2 代 × 10 条固定探针 top-10）")
    if problems:
        detail = "；".join(problems)
    return ok, detail, {
        "candidate": {"recomputed": pc.get("recomputed_probe_digest", ""),
                      "meta": pc.get("meta_probe_digest", ""),
                      "manifest": pc.get("manifest_probe_digest", "")},
        "baseline": {"recomputed": pb.get("recomputed_probe_digest", ""),
                     "meta": pb.get("meta_probe_digest", ""),
                     "manifest": pb.get("manifest_probe_digest", "")},
        "problems": problems,
    }


def check_rss_ceiling(mc: dict, mb: dict) -> tuple[bool, str, dict]:
    """峰值 RSS 上界：绝对上界 + 相对上一代 1.5 倍，两条都要。**

    量的是**每代各自全新进程**的峰值（口径 A），不是两代叠加——叠加测的是我们的
    测量方法。上界只抓量级事故（例如又把 127 MB JSON 整份 `json.loads` 一遍），
    不把正常波动做成门。
    """
    peak_c = ((mc.get("memory") or {}).get("at_end") or {}).get("peak_mib")
    peak_b = ((mb.get("memory") or {}).get("at_end") or {}).get("peak_mib")
    problems: list[str] = []
    for tag, peak in (("候选", peak_c), ("上一代", peak_b)):
        if peak is None:
            problems.append(f"{tag} 峰值 RSS 不可得（非 Windows 或 psapi 失败），无法判上界")
        elif peak > RSS_CEILING_MIB:
            problems.append(f"{tag} 峰值 {peak} MiB > 上界 {RSS_CEILING_MIB} MiB")
    if peak_c is not None and peak_b is not None:
        if peak_b > 0 and peak_c > peak_b * RSS_RATIO_CEILING:
            problems.append(f"候选峰值 {peak_c} MiB > 上一代 {peak_b} MiB × "
                            f"{RSS_RATIO_CEILING}（加载路径变了量级）")
    ok = not problems
    ratio = round(peak_c / peak_b, 3) if (peak_c and peak_b) else None
    detail = (f"峰值 RSS 候选 {peak_c} MiB / 上一代 {peak_b} MiB（比 {ratio}×，"
              f"n=2 进程）；上界 {RSS_CEILING_MIB} MiB、相对上限 {RSS_RATIO_CEILING}×")
    if problems:
        detail = "；".join(problems)
    return ok, detail, {"candidate_peak_mib": peak_c, "baseline_peak_mib": peak_b,
                        "ratio": ratio, "ceiling_mib": RSS_CEILING_MIB,
                        "ratio_ceiling": RSS_RATIO_CEILING, "problems": problems}


def check_report_fields(mc: dict, mb: dict) -> tuple[bool, str, dict]:
    """每代报告字段完整性：缺字段就是红，**不许静默当 0**。

    这条门防的是本仓库的老毛病：把"没测到"写成 0.0，于是一个测不出来的一代
    看起来又快又省。分位数尤其如此——`percentile` 对空样本返回 `None` 而不是 0.0，
    这里跟着把 `None` 当缺失处理。
    """
    problems: list[str] = []
    for tag, m in (("候选", mc), ("上一代", mb)):
        for field in MEASURE_FIELDS:
            if field not in m:
                problems.append(f"{tag} 缺字段 {field}")
        for kind in ("cold", "warm"):
            block = m.get(kind) or {}
            if not block:
                problems.append(f"{tag} 缺 {kind} 分位块")
                continue
            for field in LATENCY_FIELDS:
                if block.get(field) is None:
                    problems.append(f"{tag} {kind}.{field} 为 None（空样本或未测）")
        n = (m.get("cold") or {}).get("n")
        if n is not None and n != m.get("questions"):
            problems.append(f"{tag} 冷查样本 n={n} != 题数 {m.get('questions')}")
        if m.get("questions") in (None, 0):
            problems.append(f"{tag} 题数 {m.get('questions')}，分母为 0")
        tokens = m.get("tokens") or {}
        if tokens.get("llm_prompt_tokens") is not None:
            problems.append(f"{tag} llm_prompt_tokens 有值：本机无端点，必须是 null")
    ok = not problems
    detail = (f"候选 {len(MEASURE_FIELDS)} 字段齐 + 冷/热各 {len(LATENCY_FIELDS)} 项分位齐"
              f"（n={mc.get('questions')} 题）/ 上一代同（n={mb.get('questions')} 题）")
    if problems:
        detail = "；".join(problems)
    return ok, detail, {"required_fields": list(MEASURE_FIELDS),
                        "latency_fields": list(LATENCY_FIELDS), "problems": problems}


def check_p95_backstop(mc: dict, mb: dict) -> tuple[bool, str, dict]:
    """唯一的时序兜底门：候选热查 P95 超过上一代 2 倍才红。

    阈值为什么是 2 倍写在 `P95_BACKSTOP_FACTOR` 旁：1.1～2 倍之间的差异在本机
    不可归因（同一代重跑就能挪几十毫秒，何况还有别的进程在抢 CPU）。
    这条门只负责"慢了十倍"这种灾难性回退，申报项负责把细节记下来。
    """
    p95_c = (mc.get("warm") or {}).get("p95_ms")
    p95_b = (mb.get("warm") or {}).get("p95_ms")
    if p95_c is None or p95_b is None:
        return False, ("热查 P95 缺失（候选 "
                       f"{p95_c} / 上一代 {p95_b}），兜底门无法判定"), {
            "candidate_p95_ms": p95_c, "baseline_p95_ms": p95_b, "problems": ["P95 缺失"]}
    ratio = round(p95_c / p95_b, 3) if p95_b else None
    ok = bool(p95_b) and p95_c <= p95_b * P95_BACKSTOP_FACTOR
    detail = (f"热查 P95 候选 {p95_c} ms / 上一代 {p95_b} ms = {ratio}×"
              f"（阈值 {P95_BACKSTOP_FACTOR}×，n=候选 {mc.get('questions')} 题 / "
              f"上一代 {mb.get('questions')} 题）")
    if not ok:
        detail = (f"热查 P95 候选 {p95_c} ms 超过上一代 {p95_b} ms 的 "
                  f"{P95_BACKSTOP_FACTOR} 倍（实测 {ratio}×）：疑似灾难性回退，"
                  "须先查加载路径与检索复杂度，不许放宽阈值")
    return ok, detail, {"candidate_p95_ms": p95_c, "baseline_p95_ms": p95_b,
                        "ratio": ratio, "threshold": P95_BACKSTOP_FACTOR}


# ---------------------------------------------------------------- 打印与汇总
def _fmt_latency(block: dict) -> str:
    if not block:
        return "缺失"
    return (f"n={block.get('n')} P50 {block.get('p50_ms')} P95 {block.get('p95_ms')} "
            f"P99 {block.get('p99_ms')} min {block.get('min_ms')} max {block.get('max_ms')}")


def declared_items(mc: dict, mb: dict, pc: dict, pb: dict) -> list[dict]:
    """申报项：时序数字，**只记录、不判红**，不进 `all_passed`。

    每一项都标 `enforced=False`，免得将来有人把它当门读。
    """
    items: list[dict] = []

    def add(gid: str, name: str, value, unit: str, note: str = "") -> None:
        items.append({"id": f"{gid}:{name}", "name": name, "generation_id": gid,
                      "value": value, "unit": unit, "enforced": False, "note": note})

    for gid, m, p in ((mc["generation_id"], mc, pc), (mb["generation_id"], mb, pb)):
        for kind in ("cold", "warm"):
            block = m.get(kind) or {}
            for field in ("p50_ms", "p95_ms", "p99_ms"):
                add(gid, f"{kind}_{field}", block.get(field), "ms",
                    "nearest-rank；本机不可复现到具体值，仅供同环境重跑比较")
        add(gid, "load_total_ms", m.get("load_total_ms"), "ms",
            "生产链路 pipeline.load_generation() 全程，含五件产物")
        add(gid, "index_ms", p.get("index_ms"), "ms",
            "全新进程首次 index.load_index()，页缓存冷")
        add(gid, "pipeline_ms", m.get("pipeline_ms"), "ms", "Retriever 构造")
        add(gid, "peak_rss_mib", ((m.get("memory") or {}).get("at_end") or {}).get("peak_mib"),
            "MiB", "GetProcessMemoryInfo 进程高水位")
        build = m.get("build") or {}
        add(gid, "build_total_span_s", build.get("total_span_s"), "s",
            build.get("note") or "取自 run 事件日志，不重跑构建")
    return items


def report(candidate_gid: str, baseline_gid: str, *, questions: int, warm_reps: int,
           top_k: int, graph_depth: int, gold_path: Path) -> int:
    print("=" * 78)
    print(f"K2 性能评测（候选 vs 上一代）  Python {sys.version.split()[0]}  {sys.platform}")
    print(f"候选 {candidate_gid}")
    print(f"上一代 {baseline_gid}")
    print(f"题量 {questions}  热查重复 {warm_reps}  top_k {top_k}  graph_depth {graph_depth}")
    print("口径来源：scripts/bench_h1.py（nearest-rank 分位数 / GetProcessMemoryInfo / "
          "run 事件日志构建耗时）+ scripts/bench_h3.py 口径 A（跨子进程）")
    print("注意：本机此刻可能有别的进程在抢 CPU/内存，时序数字只用于自测，"
          "正式读数需在无并发时重跑")

    tmpdir = Path(tempfile.mkdtemp(prefix="k2perf-"))
    try:
        print("-" * 78)
        print("测量中（每代 2 个全新子进程：整代测量 + 索引路径证明）…")
        mc = _spawn(["--mode", "measure", "--generation", candidate_gid,
                     "--questions", str(questions), "--warm-reps", str(warm_reps),
                     "--top-k", str(top_k), "--graph-depth", str(graph_depth),
                     "--gold-set", str(gold_path)], tmpdir / "cand.json")
        mb = _spawn(["--mode", "measure", "--generation", baseline_gid,
                     "--questions", str(questions), "--warm-reps", str(warm_reps),
                     "--top-k", str(top_k), "--graph-depth", str(graph_depth),
                     "--gold-set", str(gold_path)], tmpdir / "base.json")
        pc = _spawn(["--mode", "probe-index", "--generation", candidate_gid],
                    tmpdir / "cand_idx.json")
        pb = _spawn(["--mode", "probe-index", "--generation", baseline_gid],
                    tmpdir / "base_idx.json")
    finally:
        for leftover in tmpdir.glob("*.json"):
            leftover.unlink(missing_ok=True)
        tmpdir.rmdir()

    # 构建耗时（申报项）在父进程取：读事件日志，不重跑构建
    for m in (mc, mb):
        m["build"] = bench_h1.build_timings(m["generation_id"])

    results: list[dict] = []

    def record(gid: str, name: str, outcome: tuple[bool, str, dict]) -> None:
        ok, detail, data = outcome
        results.append({"id": gid, "name": name, "ok": ok, "detail": detail, "data": data})
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<22s} {detail}")

    print("-" * 78)
    print("确定性事实（硬门，条件不成立必红）")
    record("generations_ok", "两代可加载且产物齐备", check_generations_ok(mc, mb))
    record("index_size_delta", "索引字节与成因核对", check_index_size_delta(pc, pb, mc, mb))
    record("cold_load_path", "冷加载只走 load_index", check_cold_load_path(pc, pb))
    record("index_binding", "索引内容绑定 Manifest", check_index_binding(pc, pb))
    record("rss_ceiling", "峰值 RSS 上界", check_rss_ceiling(mc, mb))
    record("report_fields", "每代报告字段完整性", check_report_fields(mc, mb))
    record("p95_backstop", "热查 P95 兜底（2×）", check_p95_backstop(mc, mb))

    decl = declared_items(mc, mb, pc, pb)
    print("-" * 78)
    print("申报项（时序数字，只记录不判红，不进 all_passed）")
    for tag, m in (("候选  ", mc), ("上一代", mb)):
        print(f"  {tag} 冷查 {_fmt_latency(m.get('cold'))}")
        print(f"  {tag} 热查 {_fmt_latency(m.get('warm'))}")
        print(f"  {tag} 加载 {m.get('load_total_ms')} ms（含五件产物）  "
              f"管线构造 {m.get('pipeline_ms')} ms")
        mem = m.get("memory") or {}
        print(f"  {tag} RSS 起始 {((mem.get('at_start') or {}).get('current_mib'))} MiB  "
              f"加载后 {((mem.get('after_load') or {}).get('current_mib'))} MiB  "
              f"就绪 {((mem.get('after_pipeline') or {}).get('current_mib'))} MiB  "
              f"峰值 {((mem.get('at_end') or {}).get('peak_mib'))} MiB")
        mix = m.get("category_mix") or {}
        print(f"  {tag} 题量 n={m.get('questions')}  类别 " +
              "  ".join(f"{k}={v}" for k, v in mix.items()))
        b = m.get("build") or {}
        if b.get("available"):
            print(f"  {tag} 构建 run {b.get('run_id')} 总跨度 {b.get('total_span_s')} s "
                  f"（{b.get('event_count')} 条事件，取自事件日志不重跑）")
        else:
            print(f"  {tag} 构建 不可得：{b.get('reason')}")
        print(f"  {tag} 索引路径 {m.get('chunks_path')}")
    print(f"  申报项合计 {len(decl)} 项，全部 enforced=False")
    print("  LLM token/成本：不可测（本机无 LLM 端点，抽取式回答器），一律 null")

    bad = [r for r in results if not r["ok"]]
    print("-" * 78)
    print(f"硬门 {len(results) - len(bad)}/{len(results)} 通过（申报项 {len(decl)} 项不参与判定）")
    if bad:
        print("未通过：")
        for r in bad:
            print(f"  - {r['name']}: {r['detail']}")

    out = REPORT_PATH
    atomic.atomic_write_json(out, {
        "generation_id": candidate_gid,
        "baseline_generation_id": baseline_gid,
        "tool": TOOL,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "cpu_count": os.cpu_count(),
            "concurrency_warning": "读数可能受同机并发负载影响；正式读数需在无并发时重跑",
        },
        "config": {"questions": questions, "warm_reps": warm_reps, "top_k": top_k,
                   "graph_depth": graph_depth,
                   "gold_set": config.rel_to_root(gold_path)},
        "checks": results,
        "declared": decl,
        "measurements": {candidate_gid: mc, baseline_gid: mb},
        "index_probes": {candidate_gid: pc, baseline_gid: pb},
        "all_passed": not bad,
    })
    print(f"报告已写 {config.rel_to_root(out)}")
    print("=" * 78)
    return 0 if not bad else 1


# ---------------------------------------------------------------- 入口
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="K2 性能评测（候选 vs 上一代）")
    parser.add_argument("--generation", default=CANDIDATE_GID, help=f"候选世代，默认 {CANDIDATE_GID}")
    parser.add_argument("--baseline", default=BASELINE_GID, help=f"上一代，默认 {BASELINE_GID}")
    parser.add_argument("--questions", type=int, default=40, help="题量，默认 40")
    parser.add_argument("--warm-reps", type=int, default=2, help="热查每题重复，默认 2")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--graph-depth", type=int, default=1)
    parser.add_argument("--gold-set", default="", help="默认 data/goldset/gold-v1.json")
    # 子进程模式（不是给人用的入口，父进程调用）
    parser.add_argument("--mode", default="compare",
                        choices=("compare", "measure", "probe-index"))
    parser.add_argument("--out", default="", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.questions < 1:
        print("[2] --questions 必须 >= 1", file=sys.stderr)
        return 2
    if args.warm_reps < 1:
        print("[2] --warm-reps 必须 >= 1", file=sys.stderr)
        return 2
    if args.top_k < 1:
        print("[2] --top-k 必须 >= 1", file=sys.stderr)
        return 2
    if args.graph_depth < 0:
        print("[2] --graph-depth 不能为负", file=sys.stderr)
        return 2
    if args.mode != "compare" and not args.out:
        print("[2] 子进程模式必须给 --out", file=sys.stderr)
        return 2

    gold = Path(args.gold_set) if args.gold_set else ROOT / "data" / "goldset" / "gold-v1.json"
    if not gold.is_absolute():
        gold = (ROOT / gold).resolve()

    try:
        if args.mode == "measure":
            if not gold.exists():
                raise PerfError(f"Gold Set 不存在：{gold}")
            payload = measure_one(args.generation, questions=args.questions,
                                  warm_reps=args.warm_reps, top_k=args.top_k,
                                  graph_depth=args.graph_depth, gold_path=gold)
            Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                      encoding="utf-8", newline="\n")
            return 0
        if args.mode == "probe-index":
            payload = probe_index_path(args.generation)
            Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                      encoding="utf-8", newline="\n")
            return 0
        if not gold.exists():
            raise PerfError(f"Gold Set 不存在：{gold}")
        return report(args.generation, args.baseline, questions=args.questions,
                      warm_reps=args.warm_reps, top_k=args.top_k,
                      graph_depth=args.graph_depth, gold_path=gold)
    except PerfError as exc:
        print(f"[2] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
