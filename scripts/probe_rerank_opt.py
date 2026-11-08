"""重排延迟优化实测：能不能把 95 ms/对压到 10 ms/对量级（20 候选 ≤ 200 ms）。

## 要回答的问题

`docs/i阶段实测.md` §5.1 的读数是：真实块正文截 512 token 时 **95.1 ms/对**，
重排 20 个候选要 1.9 秒。目标是「一次查询的重排开销 ≤ 200 ms」，即 10 ms/对。
本脚本把 7 类手段逐一实测，每格都给**本次跑出来的** P50/P95，不引用旧数。

## 为什么是「组内轮转交错」而不是「一个配置测完再测下一个」

第一版是按配置顺序测的，结果同一份配置（`int8/512/16 threads=默认`）在
baseline 组测得 **1089 ms**、在 threads 组末尾测得 **5218 ms**——差 4.8 倍。
两次之间没有改任何参数。原因不是配置，是这台机器：DeepSeek Harness / node /
msedge / wps 等进程在抢 20 个逻辑核（实测 LoadPercentage 41%），加上长时间连续
满载会降频。**顺序测量的结果里混着"第几个测的"，不是配置的效果。**

所以本版的协议是：

1. **同组的多个配置同时建好 session**，然后按 `(轮, 题, 配置)` 三重循环轮转采样，
   让每个配置的样本在时间上均匀铺开——组内比较不再受漂移影响。
2. 每个配置先 `warmup()` 一次并丢弃（实测首次执行真实更慢：连测 12 次得
   `[5337, 5337, 3777, 2100, 2080, 2108, 2048, 2207, 2177, 2046, 2102, 2159]` ms）。
3. 每组都重测一次**同一份基准配置**（`int8/512/16/bucket/default`）作为锚点，
   报告里可以直接看到跨组漂移有多大——漂移大的组，组内结论也要打折看。
4. 每行都写 `n_samples`。P95 在小样本上就是接近最大值，不许藏着。

「一次调用」= 一次 `score_pairs` 覆盖该题的 20 个候选，所以 `p50_ms`/`p95_ms`
就是**一轮重排的端到端耗时**，`per_pair_ms` = 该值 / 候选数。

## 数据

* 块正文：`data/arkrag/generations/g-20261103T132152Z-06a741/chunks.jsonl`（13,798 块）。
  **不用** `data/arkrag/chunks.jsonl`——那份少 17 块，且 `current.json` 指向的是
  另一代（`g-20261030T154606Z-55dfd7`），拿它测等于测错语料。
* 问句：`data/goldset/gold-v1.json` 的 `items` 前 40 条的 `question`。
* 候选池：每题 = 该题 `gold_chunk_ids` 的块正文（1–4 条）+ 同一 `doc_id` 内按
  `chunk_id` 排序的邻近块，补到 20 条。**全部是真实块正文，一个字都不截**——
  截断只由 `max_length` 负责，它是被测变量，不能提前破坏。

  用同文档邻近块补池的理由：延迟与相关性无关，ρ 是**同一池内**不同配置的排序比较，
  池怎么造都不影响这两件事；而邻近块比随机块更接近真实难负例。这是本脚本的构造
  选择，报告里不冒充"真实检索候选"。

## GraphOptimizationLevel 那一格

`CrossEncoderReranker` 的公开签名不带图优化档位（生产只有一个正确选择：ORT 默认
ALL）。为测四档，本脚本**用同一份权重另建一个 ORT session 并替换 reranker 内部的
`_sess`**，打分仍然走生产代码 `score_pairs`——不是另写一份推理逻辑。这样测出的差值
才是同一份 tokenize/pad/分桶代码下的差值。这一点必须写在报告里，否则别人会以为
生产代码暴露了那个开关。
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag.rerank import CrossEncoderReranker  # noqa: E402

#: 模型目录。仓库外（本机 roleplay-ai 项目的模型包），所以读的时候是绝对路径，
#: 但**写进 JSON 的只有目录名**——入库产物里不许出现机台路径
#: （见 tests/test_portable_paths.py）。
MODEL_DIR = Path(
    r"C:\Users\Lenovo\WorkBuddy\2026-07-27-16-45-03\roleplay-ai\models\bge-reranker-base-onnx"
)

#: 硬绑这一代。走 `config.chunks_path()` 会读到 current.json 指向的另一代（少 17 块）。
CHUNKS_PATH = ROOT / "data" / "arkrag" / "generations" / "g-20261103T132152Z-06a741" / "chunks.jsonl"
GOLDSET_PATH = ROOT / "data" / "goldset" / "gold-v1.json"
OUT_PATH = ROOT / "data" / "_tmp_rerank_opt.json"

CANDIDATES = 20
WARMUP_RUNS = 1


# ---------------------------------------------------------------- 环境与数据


def raise_priority() -> str:
    """把本进程提到 HIGH 优先级。

    绑核那条路走不通：实测 `SetProcessAffinityMask` 返回 0（失败），
    `GetProcessAffinityMask` 返回 0xfffff（20 核全掩码）。所以只能提优先级。
    """
    try:
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.kernel32.SetPriorityClass(handle, 0x80)  # HIGH_PRIORITY_CLASS
    except Exception as exc:  # pragma: no cover - 非 Windows
        return f"优先级提升失败（{type(exc).__name__}: {exc}）"
    return "HIGH" if ok else "提升失败（SetPriorityClass 返回 0）"


def read_chunks(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["chunk_id"]] = rec
    return out


def build_pools(chunks: dict[str, dict], items: list[dict], n_questions: int):
    """每题造一个 (question, docs) 池。docs 全是真实块正文，长度不动。"""
    by_doc: dict[str, list[str]] = {}
    for cid, rec in chunks.items():
        by_doc.setdefault(str(rec.get("doc_id") or ""), []).append(cid)
    for ids in by_doc.values():
        ids.sort()

    pools = []
    n_gold = n_neighbour = 0
    for item in items[:n_questions]:
        gold_ids = [cid for cid in (item.get("gold_chunk_ids") or []) if cid in chunks]
        picked = list(gold_ids)
        n_gold += len(picked)
        if picked:
            neighbours = by_doc.get(str(chunks[picked[0]].get("doc_id") or ""), [])
        else:
            # 该题 gold_chunk_ids 为空（应弃答题）：退回全库按 id 排序的前若干块。
            neighbours = sorted(chunks)
        for cid in neighbours:
            if len(picked) >= CANDIDATES:
                break
            if cid not in picked:
                picked.append(cid)
        n_neighbour += len(picked) - len(gold_ids)
        docs = [str(chunks[cid].get("text") or "") for cid in picked]
        pools.append((item.get("question_id", ""), item["question"], [d for d in docs if d]))
    stats = {
        "questions": len(pools),
        "candidates_per_question_min": min(len(p[2]) for p in pools),
        "candidates_per_question_max": max(len(p[2]) for p in pools),
        "docs_from_gold": n_gold,
        "docs_from_neighbour": n_neighbour,
        "doc_chars_mean": round(statistics.mean(len(d) for p in pools for d in p[2])),
        "doc_chars_min": min(len(d) for p in pools for d in p[2]),
        "doc_chars_max": max(len(d) for p in pools for d in p[2]),
    }
    return pools, stats


# ---------------------------------------------------------------- 统计工具


def pct(values: list[float], p: float) -> float:
    """最近秩法分位数。

    **不做线性插值**：n=20 的时候插值出来的 P95 是两个样本之间的虚构数字，
    它比"第 19 大"听起来精确，实际是编的。
    """
    if not values:
        raise ValueError("空样本")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    rank = int(round(p / 100.0 * len(xs) + 0.5))
    return xs[max(1, min(len(xs), rank)) - 1]


def spearman(a: list[float], b: list[float]) -> float:
    """秩相关，同分取平均秩（与 scripts/probe_reranker_budget.py 同一定义）。"""

    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        out = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    ra, rb = ranks(a), ranks(b)
    n = len(ra)
    if n < 2:
        return 1.0
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else 1.0


def mean_rho(ref: dict[str, list[float]], got: dict[str, list[float]]) -> tuple[float, float]:
    """逐题算 ρ 再取均值与最差题。

    不把 40 题拼成一条长向量算一个 ρ：那会把"题间分数尺度不同"混进系数里，
    测出来的其实是尺度差异，不是排序差异。
    """
    rhos = []
    for qid, base in ref.items():
        cur = got.get(qid)
        if cur is None or len(cur) != len(base) or len(base) < 3:
            continue
        rhos.append(spearman(base, cur))
    if not rhos:
        return float("nan"), float("nan")
    return round(statistics.mean(rhos), 4), round(min(rhos), 4)


# ---------------------------------------------------------------- 交错测量


def measure(rrs: dict, pools, questions: int, reps: int, collect_scores: bool = False):
    """组内轮转交错采样。

    循环顺序是 `(轮, 题, 配置)`：同一个题在几乎同一时刻被所有配置各跑一遍，
    于是任何时刻的机器状态对每个配置的贡献相同。这是本脚本唯一一条抗漂移措施，
    去掉它表格立刻变成"第几个测的"排行榜（实测同一配置两次差 4.8 倍）。
    """
    names = list(rrs)
    samples = {n: [] for n in names}
    scores: dict[str, dict[str, list[float]]] = {n: {} for n in names}
    for name, rr in rrs.items():
        for _ in range(WARMUP_RUNS):
            q = pools[0][1]
            rr.score_pairs([(q, d) for d in pools[0][2]])
    order = list(range(questions))
    for rep in range(reps):
        for qi in order:
            qid, q, docs = pools[qi]
            pairs = [(q, d) for d in docs]
            for name in names:
                rr = rrs[name]
                t0 = time.perf_counter()
                got = rr.score_pairs(pairs)
                dt = (time.perf_counter() - t0) * 1000.0
                samples[name].append(dt)
                if collect_scores and rep == reps - 1:
                    scores[name][qid] = got
    return samples, scores


def measure_single(rrs: dict, pools, questions: int, reps: int):
    """单对延迟：一次调用只喂 1 个候选。端到端 20 候选掩盖了"每对多少钱"。"""
    names = list(rrs)
    samples = {n: [] for n in names}
    for name, rr in rrs.items():
        for _ in range(WARMUP_RUNS):
            rr.score_pairs([(pools[0][1], pools[0][2][0])])
    for rep in range(reps):
        for qi in range(questions):
            q = pools[qi][1]
            doc = pools[qi][2][0]
            for name in names:
                t0 = time.perf_counter()
                rrs[name].score_pairs([(q, doc)])
                samples[name].append((time.perf_counter() - t0) * 1000.0)
    return samples


def measure_scaling(rr, pools, questions: int, reps: int, sizes):
    """候选条数扫描：一次调用喂 n 个候选，n 从 1 到 40。

    这是回答「20 候选能不能进 200 ms」的唯一硬办法。只测 n=20 一个点，
    分不清「每对多少钱」和「每次调用有多少固定开销」；扫描出来的斜率是每对成本，
    截距是固定开销。若截距本身已经超过 200 ms，那么无论怎么截长度、怎么调线程，
    20 候选都不可能进 200 ms——这个结论只能靠斜率/截距分解得到。

    与 measure() 同理用 (轮, 题, n) 轮转：n 的样本在时间上铺开，漂移对每个 n
    的贡献相同，所以 n 之间的差值（斜率）不受漂移影响。
    """
    sizes = list(sizes)
    samples = {n: [] for n in sizes}
    for _ in range(WARMUP_RUNS):
        rr.score_pairs([(pools[0][1], d) for d in pools[0][2][:max(sizes)]])
    for _rep in range(reps):
        for qi in range(questions):
            qtext, docs = pools[qi][1], pools[qi][2]
            for n in sizes:
                pairs = [(qtext, d) for d in docs[:n]]
                t0 = time.perf_counter()
                rr.score_pairs(pairs)
                samples[n].append((time.perf_counter() - t0) * 1000.0)
    return samples


def fit_line(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """最小二乘 y = slope*x + intercept，返回 (slope, intercept)。"""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    slope = num / den if den else 0.0
    return slope, my - slope * mx


def measure_breakdown(rr, pools, questions: int, reps: int):
    """把一次 20 候选调用拆成「tokenize+分桶定宽」与「ORT 前向」两段。

    目的不是优化，是定位：若 tokenize 段占比很小，那么所有长度/批大小调优的天花板
    都由 ORT 前向决定，分桶能省下的那点 padding 也就在这段里，天然只能省几个百分点。
    """
    tok_t, inf_t = [], []
    for _ in range(WARMUP_RUNS):
        rr.score_pairs([(pools[0][1], d) for d in pools[0][2]])
    for _rep in range(reps):
        for qi in range(questions):
            qtext, docs = pools[qi][1], pools[qi][2]
            pairs = [(qtext, d) for d in docs]
            t0 = time.perf_counter()
            enc = rr._encode(pairs)
            t1 = time.perf_counter()
            for idxs, width in rr._plan_buckets(enc):
                for start in range(0, len(idxs), rr._batch_size):
                    chunk = idxs[start : start + rr._batch_size]
                    ids, mask = rr._pad_batch(enc, chunk, width)
                    rr._run(ids, mask)
            t2 = time.perf_counter()
            tok_t.append((t1 - t0) * 1000.0)
            inf_t.append((t2 - t1) * 1000.0)
    return tok_t, inf_t


def row_from(group: str, name: str, rr, samples: list[float], *, candidates: int,
             extra: dict | None = None) -> dict:
    info = rr.info() if rr is not None else {}
    per_pair = [s / candidates for s in samples]
    row = {
        "group": group,
        "config": name,
        "use_int8": info.get("use_int8"),
        "max_length": info.get("max_length"),
        "batch_size": info.get("batch_size"),
        "bucket_width": info.get("bucket_width"),
        "intra_op_threads": info.get("intra_op_threads"),
        "candidates": candidates,
        "n_samples": len(samples),
        "p50_ms": round(pct(samples, 50), 1),
        "p95_ms": round(pct(samples, 95), 1),
        "min_ms": round(min(samples), 1),
        "max_ms": round(max(samples), 1),
        "per_pair_p50_ms": round(pct(per_pair, 50), 2),
        "per_pair_p95_ms": round(pct(per_pair, 95), 2),
    }
    if extra:
        row.update(extra)
    return row


class Recorder:
    def __init__(self, meta: dict):
        self.meta = meta
        self.rows: list[dict] = []

    def add(self, row: dict) -> None:
        self.rows.append(row)
        flush(self.rows, self.meta)
        if "p50_ms" in row:
            tail = ""
            if "spearman_mean" in row:
                tail = f" | rho {row['spearman_mean']}"
            print(
                f"  [{row['group']:<11}] {row['config']:<38} "
                f"P50 {row['p50_ms']:>8.1f}  P95 {row['p95_ms']:>8.1f} ms "
                f"| 每对 {row.get('per_pair_p50_ms', 0.0):>7.2f} ms | n={row['n_samples']}" + tail
            )
        else:
            # 拟合行没有 p50_ms（它本身就是从 p50 推出来的两个参数），单独打。
            print(f"  [{row['group']:<11}] {row['config']:<38} {row.get('note', '')}")


def flush(rows: list[dict], meta: dict) -> None:
    """每格都落一次盘：中途被杀也留得下已测出的真实读数。"""
    payload = dict(meta)
    payload["rows"] = rows
    # 表也一起落盘：只给 rows 的话，读的人还得自己复算分位数口径。
    payload["markdown"] = render_markdown(rows)
    OUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )


def build(model_dir=MODEL_DIR, **kw) -> CrossEncoderReranker:
    return CrossEncoderReranker(model_dir, **kw)


def swap_session(rr: CrossEncoderReranker, graph_level) -> None:
    """换一个带指定 GraphOptimizationLevel 的 session，其余完全不变。

    生产代码不暴露这个开关（生产只有默认 ALL 是对的），所以这里直接调 ORT。
    打分仍走 `score_pairs`，否则测的就不是同一份 tokenize/pad/分桶代码了。
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = graph_level
    rr._sess = ort.InferenceSession(
        str(rr._weights_path), opts, providers=["CPUExecutionProvider"]
    )


def cleanup(*objs) -> None:
    for o in objs:
        del o
    gc.collect()


# ---------------------------------------------------------------- 主流程


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=8, help="延迟矩阵用的题数")
    ap.add_argument("--reps", type=int, default=3, help="轮数（样本数 = 题数 × 轮数）")
    ap.add_argument("--rho-questions", type=int, default=40, help="算 ρ 用的题数")
    ap.add_argument("--groups", default="", help="只跑这些组（逗号分隔），默认全跑")
    ap.add_argument("--skip-fp32", action="store_true")
    args = ap.parse_args(argv)
    only = {g.strip() for g in args.groups.split(",") if g.strip()}

    def want(g: str) -> bool:
        return not only or g in only

    prio = raise_priority()
    print(f"进程优先级：{prio}    os.cpu_count()={os.cpu_count()}")
    if not CHUNKS_PATH.is_file():
        print(f"FATAL 找不到块正文 {CHUNKS_PATH}（任务要求硬绑这一代）")
        return 2

    t0 = time.perf_counter()
    chunks = read_chunks(CHUNKS_PATH)
    items = json.loads(GOLDSET_PATH.read_text(encoding="utf-8"))["items"]
    n_q = max(args.questions, args.rho_questions)
    pools, pool_stats = build_pools(chunks, items, n_q)
    print(
        f"数据：{len(chunks)} 块 / {len(items)} 题（用前 {n_q}）/ 每题 {CANDIDATES} 候选 / "
        f"块正文均值 {pool_stats['doc_chars_mean']} 字（{pool_stats['doc_chars_min']}–"
        f"{pool_stats['doc_chars_max']}），读取 {(time.perf_counter() - t0) * 1000:.0f} ms"
    )

    meta = {
        "model_dir": MODEL_DIR.name,
        "onnx_session_providers": [],
        "chunks_source": CHUNKS_PATH.relative_to(ROOT).as_posix(),
        "goldset_source": GOLDSET_PATH.relative_to(ROOT).as_posix(),
        "pool_stats": pool_stats,
        "protocol": {
            "warmup_runs_per_config": WARMUP_RUNS,
            "questions": args.questions,
            "reps": args.reps,
            "rho_questions": args.rho_questions,
            "sampling": "组内 (轮, 题, 配置) 三重循环轮转，抵消机器漂移",
            "percentile": "最近秩法，不插值",
        },
        "process_priority": prio,
        "cpu_count": os.cpu_count(),
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "notes": [
            "每行 p50_ms/p95_ms = 一次完整重排调用（candidates 个候选）的端到端耗时。",
            "per_pair_ms = 该值 / candidates。",
            "ρ = 逐题 Spearman 后取均值（同分平均秩），基准见 spearman_ref。",
            "graph_opt 组用同一权重另建 ORT session 并替换 _sess，打分仍走生产 score_pairs。",
            "baseline_anchor 组的行是同一配置在不同时间段的重测，用来看跨组漂移。",
        ],
    }
    rec = Recorder(meta)

    # ---- 基准配置一直在场：组内做参照，也当跨组漂移的锚点 ----
    print("\n=== 0. 基准配置 ===")
    base = build(max_length=512, batch_size=16, use_int8=True)
    meta["onnx_session_providers"] = base.info()["providers"]
    meta["baseline_info"] = base.info()
    t_warm = base.warmup()
    meta["baseline_first_inference_ms"] = round(t_warm, 1)
    print(f"  {json.dumps(base.info(), ensure_ascii=False)}")
    print(f"  首次推理（warmup）{t_warm:.1f} ms —— 这一下必须发生在启动期，不许落在请求路径上")
    s, _ = measure({"base": base}, pools, args.questions, args.reps)
    rec.add(row_from("baseline", "int8/512/16/bucket32/threads=默认", base, s["base"],
                     candidates=CANDIDATES))

    # ---- 1. fp32 vs int8 ----
    if want("dtype") and not args.skip_fp32:
        print("\n=== 1. fp32 vs int8（同类，只换权重）===")
        fp32 = build(max_length=512, batch_size=16, use_int8=False)
        meta["fp32_info"] = fp32.info()
        print(f"  fp32 加载 {fp32.info()['load_ms']} ms / 首次推理 {fp32.warmup():.1f} ms")
        s, sc = measure({"int8": base, "fp32": fp32}, pools, args.questions, args.reps,
                        collect_scores=True)
        for key, rr, label in (("int8", base, "int8/512/16"), ("fp32", fp32, "fp32/512/16")):
            rec.add(row_from("dtype", label, rr, s[key], candidates=CANDIDATES))
        # 单对（1 个候选）P50/P95——交付要的两个数
        ssp = measure_single({"int8": base, "fp32": fp32}, pools, args.questions, args.reps)
        for name, rr in (("int8/512/16 单对", base), ("fp32/512/16 单对", fp32)):
            key = name.split("/")[0]
            sub = ssp[key]
            rec.add({
                "group": "single_pair", "config": name, "use_int8": rr.info()["use_int8"],
                "max_length": 512, "batch_size": 16, "bucket_width": rr.info()["bucket_width"],
                "intra_op_threads": None, "candidates": 1, "n_samples": len(sub),
                "p50_ms": round(pct(sub, 50), 2), "p95_ms": round(pct(sub, 95), 2),
                "min_ms": round(min(sub), 2), "max_ms": round(max(sub), 2),
                "per_pair_p50_ms": round(pct(sub, 50), 2),
                "per_pair_p95_ms": round(pct(sub, 95), 2),
            })
        # fp32 与 int8 的排序一致性：直接量"量化有没有改变名次"
        rho_mean, rho_min = mean_rho(sc["fp32"], sc["int8"])
        meta["int8_vs_fp32_rho"] = {"mean": rho_mean, "min": rho_min,
                                    "note": "逐题 Spearman(int8 分数, fp32 分数)，题数= questions"}
        print(f"  int8 与 fp32 的逐题排序一致性：ρ 均值 {rho_mean}，最差题 {rho_min}")
        cleanup(fp32)

    # ---- 2. 线程数 ----
    if want("threads"):
        print("\n=== 2. intra_op_num_threads（组内交错）===")
        rrs = {}
        for th in (1, 2, 4, 8, 14, None):
            label = "None(ORT默认)" if th is None else str(th)
            # `None` 复用基准那个 session（ORT 默认线程数），其余各建一个。
            rrs[label] = (
                base
                if th is None
                else build(max_length=512, batch_size=16, use_int8=True, intra_op_threads=th)
            )
        # 线程少的配置单次调用慢一个数量级，样本数按比例缩小，否则这一组要跑十几分钟
        q = max(2, args.questions // 4)
        s, _ = measure(rrs, pools, q, 1)
        for label, rr in rrs.items():
            rec.add(row_from("threads", f"int8/512/16 threads={label}", rr, s[label],
                             candidates=CANDIDATES, extra={"note": f"样本 {q} 题 × 1 轮，线程少时缩短"}))
        best_label = min(rrs, key=lambda k: pct(s[k], 50))
        meta["threads_best"] = {"label": best_label, "p50_ms": round(pct(s[best_label], 50), 1)}
        print(f"  最快线程档：{best_label}")
        for label, rr in list(rrs.items()):
            if label != "None(ORT默认)":
                cleanup(rr)

    # ---- 3. GraphOptimizationLevel ----
    if want("graph_opt"):
        print("\n=== 3. GraphOptimizationLevel（同一权重另建 session，替换 _sess）===")
        import onnxruntime as ort

        saved = base._sess
        levels = {}
        for label, attr in (("DISABLE_ALL", "ORT_DISABLE_ALL"), ("BASIC", "ORT_ENABLE_BASIC"),
                            ("EXTENDED", "ORT_ENABLE_EXTENDED"), ("ALL", "ORT_ENABLE_ALL")):
            try:
                swap_session(base, getattr(ort.GraphOptimizationLevel, attr))
                base.warmup()
                levels[label] = base._sess
            except Exception as exc:
                rec.add({"group": "graph_opt", "config": label,
                         "error": f"{type(exc).__name__}: {exc}"})
        q = max(2, args.questions // 2)
        # 交错：每轮换 session 再测，避免"优化档"与"时间"混在一起
        per = {label: [] for label in levels}
        for _ in range(WARMUP_RUNS):
            for label, sess in levels.items():
                base._sess = sess
                base.score_pairs([(pools[0][1], d) for d in pools[0][2]])
        for _ in range(1):
            for qi in range(q):
                qq, qtext, docs = pools[qi]
                pairs = [(qtext, d) for d in docs]
                for label, sess in levels.items():
                    base._sess = sess
                    t0 = time.perf_counter()
                    base.score_pairs(pairs)
                    per[label].append((time.perf_counter() - t0) * 1000.0)
        for label in levels:
            rec.add(row_from("graph_opt", f"int8/512/16 graph_opt={label}", base, per[label],
                             candidates=CANDIDATES, extra={"note": "另建 session 替换 _sess"}))
        base._sess = saved
        cleanup(levels)

    # ---- 4. 分桶 vs 整批 padding ----
    if want("bucketing"):
        print("\n=== 4. 长度分桶 vs 整批 padding ===")
        whole = build(max_length=512, batch_size=16, use_int8=True, bucket_width=None)
        s, _ = measure({"分桶32": base, "整批padding": whole}, pools, args.questions, args.reps)
        rec.add(row_from("bucketing", "int8/512/16 整批padding", whole, s["整批padding"],
                         candidates=CANDIDATES))
        rec.add(row_from("bucketing", "int8/512/16 分桶32", base, s["分桶32"],
                         candidates=CANDIDATES))
        cleanup(whole)

    # ---- 5. max_length 与 ρ ----
    rho_rows = {}
    if want("max_length"):
        print("\n=== 5. max_length（ρ 用全部题，逐题 Spearman 取均值；基准 512）===")
        rrs = {}
        for ml in (512, 384, 256, 192, 128):
            rrs[ml] = base if ml == 512 else build(max_length=ml, batch_size=16, use_int8=True)
        s, sc = measure({str(k): v for k, v in rrs.items()}, pools,
                        args.rho_questions, 1, collect_scores=True)
        ref = sc["512"]
        for ml in rrs:
            rho_mean, rho_min = mean_rho(ref, sc[str(ml)])
            rho_rows[ml] = (rho_mean, rho_min)
            rec.add(row_from("max_length", f"int8/{ml}/16 分桶32", rrs[ml], s[str(ml)],
                             candidates=CANDIDATES,
                             extra={"spearman_mean": rho_mean, "spearman_min": rho_min,
                                    "spearman_ref": 512,
                                    "note": f"样本 {args.rho_questions} 题 × 1 轮"}))
        meta["rho_vs_512"] = {str(k): {"mean": v[0], "min": v[1]} for k, v in rho_rows.items()}
        for ml, rr in rrs.items():
            if ml != 512:
                cleanup(rr)

    # ---- 6. batch_size ----
    if want("batch_size"):
        print("\n=== 6. batch_size ===")
        rrs = {}
        for bs in (1, 8, 16, 32):
            rrs[bs] = base if bs == 16 else build(max_length=512, batch_size=bs, use_int8=True)
        s, _ = measure({str(k): v for k, v in rrs.items()}, pools, args.questions, args.reps)
        for bs, rr in rrs.items():
            rec.add(row_from("batch_size", f"int8/512/{bs} 分桶32", rr, s[str(bs)],
                             candidates=CANDIDATES))
        for bs, rr in rrs.items():
            if bs != 16:
                cleanup(rr)

    # ---- 7. 只重排 top-N + 候选条数扫描 ----
    if want("top_n"):
        print("\n=== 7. 只重排 top-N（端到端一轮）===")
        # 第一版按 5→8→10→20 顺序各测一遍，结果 5 个候选 2586 ms、20 个候选 648 ms——
        # 候选少的反而慢 4 倍。那是机器漂移，不是配置效果（同一时刻组内轮转就能避免）。
        # 所以改成一次扫描：n = 1..40 全部在同几轮里轮转采样，5/8/10/20 从扫描结果里取。
        sizes = (1, 2, 4, 5, 8, 10, 16, 20, 30, 40)
        s = measure_scaling(base, pools, args.questions, args.reps, sizes)
        xs = [float(n) for n in sizes]
        ys = [pct(s[n], 50) for n in sizes]
        slope, intercept = fit_line(xs, ys)
        meta["scaling_fit"] = {
            "sizes": list(sizes),
            "p50_by_size": {str(n): round(ys[i], 1) for i, n in enumerate(sizes)},
            "slope_ms_per_candidate": round(slope, 2),
            "intercept_ms": round(intercept, 1),
            "note": "对 p50(候选数) 做最小二乘；斜率=每多一个候选的边际成本，截距=固定开销",
        }
        for n in (5, 8, 10, 20):
            rec.add(row_from("top_n", f"int8/512/16 只喂 {n} 个候选", base, s[n], candidates=n))
        print(f"  拟合：每候选 {slope:.2f} ms，固定开销 {intercept:.1f} ms")
        # 固定开销与每对成本分开报，才能判断"进 200 ms"是否有理论可能。
        rec.add({
            "group": "top_n", "config": "拟合(斜率/截距)", "candidates": None,
            "n_samples": len(s[sizes[0]]),
            "slope_ms_per_candidate": round(slope, 2), "intercept_ms": round(intercept, 1),
            "note": "截距即候选数趋于 0 时的开销，含 tokenize 与 ORT 调用固定成本",
        })

    # ---- 7b. 一次 20 候选调用的耗时拆分 ----
    if want("breakdown"):
        print("\n=== 7b. 20 候选调用耗时拆分（tokenize vs ORT 前向）===")
        tok_t, inf_t = measure_breakdown(base, pools, args.questions, args.reps)
        for label, vals in (("tokenize+分桶", tok_t), ("ORT 前向", inf_t)):
            rec.add({
                "group": "breakdown", "config": f"int8/512/16 20候选 {label}",
                "candidates": CANDIDATES, "n_samples": len(vals),
                "p50_ms": round(pct(vals, 50), 1), "p95_ms": round(pct(vals, 95), 1),
                "min_ms": round(min(vals), 1), "max_ms": round(max(vals), 1),
                "note": "同一进程内手工分段计时，两段之和略小于 score_pairs 总耗时（有循环与回填开销）",
            })

    # ---- 8. 最优组合复核 + 锚点重测 ----
    if want("best"):
        print("\n=== 8. 锚点重测 + 组合候选（看跨组漂移与叠加收益）===")
        anchors = {"基准重测(int8/512/16/bucket)": base}
        th = meta.get("threads_best", {}).get("label")
        th_val = None if th in (None, "None(ORT默认)") else int(th)
        if th_val:
            anchors[f"int8/512/16 threads={th_val}"] = build(
                max_length=512, batch_size=16, use_int8=True, intra_op_threads=th_val)
        for ml in (384, 256):
            kw = {"max_length": ml, "batch_size": 16, "use_int8": True}
            if th_val:
                kw["intra_op_threads"] = th_val
            anchors[f"int8/{ml}/16" + (f" threads={th_val}" if th_val else "")] = build(**kw)
        s, sc = measure(anchors, pools, args.questions, args.reps, collect_scores=True)
        ref_scores = sc["基准重测(int8/512/16/bucket)"]
        for name, rr in anchors.items():
            extra = {}
            if name != "基准重测(int8/512/16/bucket)":
                rho_mean, rho_min = mean_rho(ref_scores, sc[name])
                extra = {"spearman_mean": rho_mean, "spearman_min": rho_min, "spearman_ref": 512}
            rec.add(row_from("anchor", name, rr, s[name], candidates=CANDIDATES, extra=extra))
        for name, rr in anchors.items():
            if name != "基准重测(int8/512/16/bucket)":
                cleanup(rr)

    flush(rec.rows, meta)
    print(f"\n已写入 {OUT_PATH.relative_to(ROOT).as_posix()}（{len(rec.rows)} 行）")
    print_markdown(rec.rows)
    return 0


def render_markdown(rows: list[dict]) -> str:
    """把 rows 渲染成 markdown 表。

    每格都是本次实测的毫秒数，没有一列是估算；`spearman_mean` 空着表示该格
    没测 ρ（只有 max_length 组和锚点组测），不是"ρ=0"。
    """
    cols = ("group", "config", "candidates", "n_samples", "p50_ms", "p95_ms",
            "per_pair_p50_ms", "per_pair_p95_ms", "spearman_mean")
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def print_markdown(rows: list[dict]) -> None:
    print("\n" + render_markdown(rows))


if __name__ == "__main__":
    raise SystemExit(main())
