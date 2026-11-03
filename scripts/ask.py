"""`ask` —— 计划 G7 的命令行入口：单问回答 + Gold Set 离线评测。

## 两种用法

单问::

    py -3 scripts\\ask.py "凯尔希和罗德岛是什么关系？" --generation g-20261031T035231Z-4eafae
    py -3 scripts\\ask.py "..." --generation g-... --top-k 30 --graph-depth 2 --json out.json

评测::

    py -3 scripts\\ask.py --evaluate --gold-set data\\goldset\\gold-v1.json --limit 50

## 退出码

* `0` —— 链路跑通。**注意：分数低也是 0**。评测器不是门禁，它负责如实报告数字；
  把"分数不够高"当成失败会让每次调参都必须先改退出码判据。
* `2` —— 硬错误：世代不存在/缺产物、Gold Set 文件缺失或损坏、`--limit` 非法。
* `3` —— 引用校验硬失败：有引用在 `text[start:end]` 上对不上。这是产品缺陷
  （`answer.verify_citations` 已把不合格引用剔除并降置信度），不能当成"答得不好"，
  所以单独一个码。

## 为什么评测要打印分母

每一项指标都会打印「能算 N 条 / 跳过 M 条」。理由：`Recall@5` 只在有 `gold_chunk_ids`
的题上可算，`拒答题正确率` 只在该题有确定的期望行为时可算。只报一个百分数而不报分母，
会把「只有 12 条能算」和「320 条都算了」显示成同一个数字。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import (  # noqa: E402
    adjacency as adjacency_mod,
    answer as answer_mod,
    config,
    index as index_mod,
    query as query_mod,
    release,
    retrieve as retrieve_mod,
    source,
    vocab,
)


# ---------------------------------------------------------------- 载入世代
def load_generation(generation_id: str) -> dict:
    """读出评测/回答需要的全部产物。缺任何一件都抛 `SystemExit(2)`。

    这里刻意**不**做"缺断言就用边顶替"的降级：`ark-rag-3` 的回答要靠逐观察断言
    给出证据偏移，缺了它只能给出没有位置的引用，那种引用无法机检。
    """
    gen = release.Generation(generation_id)
    if not gen.dir.exists():
        raise SystemExit(f"[2] 世代不存在：{generation_id}（{gen.dir}）")
    manifest = gen.load()
    schema = str(manifest.get("graph_schema") or config.GRAPH_SCHEMA_LEGACY)
    if schema != config.GRAPH_SCHEMA_IDENTITY:
        raise SystemExit(
            f"[2] 世代 {generation_id} 的 graph_schema={schema}，"
            f"本工具只支持 {config.GRAPH_SCHEMA_IDENTITY}（ark-rag-1 没有断言，"
            "做不了带方向的扩展与可定位引用）"
        )
    for path, label in (
        (gen.graph_path, "graph/graph.json"),
        (gen.assertion_path, "graph/assertions.jsonl"),
        (gen.adjacency_path, "graph/adjacency.json"),
        (gen.index_path, "index/lexical.json"),
    ):
        if not path.exists():
            raise SystemExit(f"[2] 世代 {generation_id} 缺 {label}（{path}）")

    graph = json.loads(gen.graph_path.read_text("utf-8"))
    adjacency = json.loads(gen.adjacency_path.read_text("utf-8"))
    assertions = _read_jsonl(gen.assertion_path)
    idx, idx_meta = index_mod.load_index(gen.index_path)

    chunks: dict[str, dict] = {}
    chunks_path = config.chunks_path()
    if not chunks_path.exists():
        raise SystemExit(f"[2] 找不到块文件 {chunks_path}（引用校验需要块正文）")
    for rec in _read_jsonl(chunks_path):
        cid = str(rec.get("chunk_id") or "")
        if cid:
            chunks[cid] = rec

    return {
        "generation_id": generation_id,
        "generation": gen,
        "manifest": manifest,
        "graph": graph,
        "adjacency": adjacency,
        "assertions": assertions,
        "index": idx,
        "index_meta": idx_meta,
        "chunks": chunks,
    }


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    try:
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[2] {path} 不是合法 JSONL：{exc}") from exc
    return out


# ---------------------------------------------------------------- 管线
def build_pipeline(loaded: dict, *, top_k: int, graph_depth: int) -> dict:
    retriever = retrieve_mod.Retriever(
        loaded["index"],
        loaded["graph"],
        loaded["adjacency"],
        chunks=loaded["chunks"],
        assertions=loaded["assertions"],
        vocab=vocab,
    )
    answerer = answer_mod.ExtractiveAnswerer()
    return {"retriever": retriever, "answerer": answerer, "top_k": top_k,
            "graph_depth": graph_depth}


def ask_once(pipeline: dict, question: str, *, timings: dict | None = None) -> dict:
    """一问一答的完整链路。返回答案 + 每一层的中间量（供排查与评测）。

    `timings` 传入一个 dict 时，逐阶段毫秒数写进它（H1 性能基线要用）。
    默认 `None` → 不测时间，返回结构与原来完全一致：**计时是可选旁路，
    不是返回值的一部分**，否则每个读 `result["search"]` 的调用方都得改。
    """
    retriever = pipeline["retriever"]
    entities = pipeline.get("entities") or retriever.graph.get("entities") or []
    _t0 = time.perf_counter()
    qa = query_mod.analyze_query(
        question,
        entities=entities,
        aliases=retriever.graph.get("alias_index"),
        vocab=vocab,
        # 复用 `Retriever` 建好的只读结构。**这一行是 H2 的主要提速点**
        # （每问省约 98 ms 的图级重建，见 `query.QueryContext` 的实测表）。
        # 必须传 `retriever.query_context` 而不是自己造：那样又变回每问重建。
        context=getattr(retriever, "query_context", None),
    )
    if timings is not None:
        timings["query_ms"] = (time.perf_counter() - _t0) * 1000.0
    if not qa.get("should_answer", True):
        if timings is not None:
            timings["total_ms"] = (time.perf_counter() - _t0) * 1000.0
            timings["abstained_by_query"] = 1
        return {
            "query_analysis": qa,
            "search": {"hits": [], "stats": {"skipped_reason": qa.get("reason")}},
            "expand": {"expansions": [], "stats": {"skipped_reason": qa.get("reason")}},
            "context": {"units": [], "context_digest": "", "stats": {}},
            "answer": pipeline["answerer"].answer(
                question, {"units": [], "context_digest": ""},
                generation_id=pipeline.get("generation_id", ""),
                top_score=0.0, query_analysis=qa,
            ),
        }

    found = retriever.search(qa, top_k=pipeline["top_k"])
    if timings is not None:
        timings["search_ms"] = (time.perf_counter() - _t0) * 1000.0 - timings["query_ms"]
    _t1 = time.perf_counter()
    seed_ids = [c["entity_id"] for c in qa.get("entity_candidates") or []]
    expanded = retriever.expand(
        seed_ids,
        depth=pipeline["graph_depth"],
        query_analysis=qa,
        seed_chunk_ids=[h["chunk_id"] for h in found["hits"]],
    )
    if timings is not None:
        timings["expand_ms"] = (time.perf_counter() - _t1) * 1000.0
        timings["expand_nodes"] = int(expanded["stats"].get("nodes_visited") or 0)
        timings["expand_edges"] = int(expanded["stats"].get("expansions") or 0)
    _t2 = time.perf_counter()
    context = retriever.build_context(found["hits"], expanded["expansions"], query_analysis=qa)
    if timings is not None:
        timings["context_ms"] = (time.perf_counter() - _t2) * 1000.0
    _t3 = time.perf_counter()
    ans = pipeline["answerer"].answer(
        question,
        context,
        generation_id=pipeline.get("generation_id", ""),
        top_score=float(found["stats"].get("top_score") or 0.0),
        query_analysis=qa,
        retrieval_stats=found["stats"],
    )
    if timings is not None:
        timings["answer_ms"] = (time.perf_counter() - _t3) * 1000.0
        timings["total_ms"] = (time.perf_counter() - _t0) * 1000.0
        timings["context_tokens"] = int(context["stats"].get("tokens_used") or 0)
        timings["context_units"] = int(context["stats"].get("units") or 0)
        timings["candidates"] = int(found["stats"].get("candidates") or 0)
    return {
        "query_analysis": qa,
        "search": found,
        "expand": expanded,
        "context": context,
        "answer": ans,
    }


# ---------------------------------------------------------------- 输出
def print_answer(result: dict, *, generation_id: str) -> None:
    ans = result["answer"]
    qa = result["query_analysis"]
    print("=" * 74)
    print(f"问：{qa.get('original_query')}")
    print(f"世代：{generation_id}")
    print(
        f"问句解析：候选实体 {len(qa.get('entity_candidates') or [])} 个"
        f"  关系提示 {len(qa.get('relation_hints') or [])} 个"
        f"  时间词 {len(qa.get('time_constraints') or [])} 个"
        f"  否定提示 {qa.get('negated_hint')}"
        f"  需要图扩展 {qa.get('requires_graph')}"
    )
    for cand in qa.get("entity_candidates") or []:
        flag = "（歧义）" if cand.get("ambiguous") else ""
        print(f"    - {cand['name']} → {cand['entity_id']}  "
              f"score {cand['score']}  {cand['reason']}{flag}")

    s = result["search"]["stats"]
    print(
        f"检索：命中 {len(result['search']['hits'])} 块"
        f"  bm25词 {s.get('bm25_terms')}  实体通道 {s.get('entity_channels')}"
        f"  top_score {s.get('top_score')}"
    )
    top_hits = result["search"]["hits"][:5]
    for hit in top_hits:
        chans = ",".join(src["channel"] for src in hit.get("sources") or [])
        print(f"    {hit['score']:<9.4f} {hit['chunk_id']}  [{chans}]")

    e = result["expand"]["stats"]
    print(
        f"扩展：深度 {e.get('budget', {}).get('depth_used')}"
        f"  访问节点 {e.get('nodes_visited')}  边 {e.get('expansions')}"
        f"  断言 {e.get('assertions_used')}"
        f"  走反向边 {e.get('used_in_edges')}"
        f"  截断原因 {e.get('truncated_reason') or '（未截断）'}"
    )
    tiers = e.get("priority_tiers") or {}
    if tiers:
        print("    优先级档命中：" + "  ".join(f"档{k}={v}" for k, v in sorted(tiers.items())))

    c = result["context"]["stats"]
    print(
        f"上下文：单元 {c.get('units')}  证据锚点 {c.get('evidence_spans')}"
        f"  tokens {c.get('tokens_used')}/{c.get('token_budget')}"
        f"  直接证据块 {c.get('direct_units')}  纯图背景块 {c.get('graph_only_units')}"
        f"  摘要 {str(result['context'].get('context_digest'))[:16]}"
    )

    print("-" * 74)
    if ans.get("abstained"):
        print(f"拒答：{ans.get('abstain_reason')}")
    else:
        print(f"答：{ans.get('answer')}")
    print(f"置信度 {ans.get('confidence')}   生成器 {ans.get('versions', {}).get('answerer')}")
    print(f"引用 {len(ans.get('citations') or [])} 条：")
    for cite in ans.get("citations") or []:
        print(f"    [{cite['start']}:{cite['end']}] {cite['chunk_id']}  「{cite['evidence']}」")
        print(f"        支撑：{cite['claim']}"
              f"  谓词在原文行内：{cite.get('predicate_in_line')}")
    if ans.get("warnings"):
        print("提示：")
        for w in ans["warnings"]:
            print(f"    - {w}")


# ---------------------------------------------------------------- 评测
def _hit_ids(hits: list[dict]) -> list[str]:
    return [str(h.get("chunk_id") or "") for h in hits]


def _stratified_sample(items: list[dict], limit: int, *, seed: int = 20261101) -> list[dict]:
    """按类别配额抽样，保证局部评测不退化成一个类别。

    原先是 `items[:limit]` —— 按文件顺序截断。而 `gold-v1.json` 里 `single_chunk`
    排最前（64 条），所以 `--limit 40`/`--limit 50` 实测只测到 `single_chunk`
    一个类别，报告看起来"单类很好"，实际多跳/对抗/时间冲突一条没跑。这是评测
    入口的真实缺陷：局部评测的数字不能代表整体，而人会拿它当结论。

    做法：按类别等比例分配配额，同类内部用**固定种子**打散后取前 N ——
    固定种子保证同一个 `--limit` 两次跑抽到同一批题，指标可比；不打散则总是
    取每类最前面几条，等于换了个口径偏。

    两个边界必须说清楚：
    * `limit` 小于类别数时不能"每类保底 1 条"——那会让 `--limit 1` 返回 8 条，
      总量超出去，评测规模不再等于用户要的规模。此时按类别池大小从大到小取
      前 `limit` 个类别，每个 1 条。
    * 配额不足的类别如实给实际条数（不借别的类别的题填），但只要有类别还有
      余量就补齐到 `limit`，所以正常情况下条数正好等于 `limit`。
    """
    total = len(items)
    want = max(1, min(int(limit), total))
    if want >= total:
        return list(items)

    by_cat: dict[str, list[dict]] = {}
    for item in items:
        by_cat.setdefault(str(item.get("category") or ""), []).append(item)

    rng = random.Random(seed)
    for rows in by_cat.values():
        rng.shuffle(rows)

    # 类别按池大小降序：数量少的类别（adversarial 16 条）也更容易保住，
    # 因为等比例取整最先砍的就是它们。
    cats = sorted(by_cat, key=lambda c: (-len(by_cat[c]), c))

    if want < len(cats):
        # 名额比类别还少：一类一条，优先给池子大的类别
        return [by_cat[cat][0] for cat in cats[:want]]

    # 每类先保底 1 条，剩余名额按占比分配
    quota: dict[str, int] = {cat: 1 for cat in cats}
    remaining = want - len(cats)
    if remaining:
        shares = {cat: remaining * len(by_cat[cat]) / total for cat in cats}
        for cat in cats:
            quota[cat] += int(shares[cat])
        # 取整丢掉的零头按小数部分从大到小补
        left = want - sum(quota.values())
        for cat in sorted(cats, key=lambda c: (-(shares[c] - int(shares[c])), c)):
            if left <= 0:
                break
            quota[cat] += 1
            left -= 1
    # 某类不够自己的配额时，把缺口转给还有余量的类别，保证总数等于 want
    deficit = 0
    for cat in cats:
        if quota[cat] > len(by_cat[cat]):
            deficit += quota[cat] - len(by_cat[cat])
            quota[cat] = len(by_cat[cat])
    for cat in cats:
        if deficit <= 0:
            break
        room = len(by_cat[cat]) - quota[cat]
        if room > 0:
            add = min(room, deficit)
            quota[cat] += add
            deficit -= add

    picked: list[dict] = []
    for cat in cats:
        picked.extend(by_cat[cat][: quota[cat]])
    return picked


def evaluate(gold_path: Path, *, limit: int, generation_id: str = "") -> dict:
    """跑 Gold Set 并返回七项指标 + 每项的分母。"""
    if not gold_path.exists():
        raise SystemExit(f"[2] Gold Set 不存在：{gold_path}")
    try:
        doc = json.loads(gold_path.read_text("utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[2] Gold Set 不是合法 JSON：{exc}") from exc
    items = doc.get("items") if isinstance(doc, dict) else doc
    if not isinstance(items, list) or not items:
        raise SystemExit(f"[2] Gold Set 里没有 items：{gold_path}")

    gid = generation_id or _latest_identity_generation()
    loaded = load_generation(gid)
    pipeline = build_pipeline(loaded, top_k=20, graph_depth=1)
    pipeline["generation_id"] = gid
    pipeline["entities"] = loaded["graph"].get("entities") or []

    selected = _stratified_sample(items, limit)

    recall20: list[float] = []
    recall5: list[float] = []
    rr: list[float] = []
    cite_total = 0
    cite_ok = 0
    coverage: list[bool] = []
    abstain_ok: list[bool] = []
    direction_ok: list[bool] = []
    cite_verify_failures = 0
    rows: list[dict] = []
    skipped: dict[str, int] = {"recall": 0, "coverage": 0, "abstain": 0, "direction": 0}

    for item in selected:
        question = str(item.get("question") or "")
        gold_chunks = [str(c) for c in item.get("gold_chunk_ids") or []]
        should_abstain = bool(item.get("should_abstain"))
        category = str(item.get("category") or "")

        result = ask_once(pipeline, question)
        hits = _hit_ids(result["search"]["hits"])
        ans = result["answer"]

        # 引用定位正确率：逐条看 text[start:end] 是否逐字等于 evidence
        for cite in ans.get("citations") or []:
            cite_total += 1
            rec = loaded["chunks"].get(str(cite.get("chunk_id") or ""))
            text = str((rec or {}).get("text") or "")
            start, end = int(cite.get("start", -1)), int(cite.get("end", -1))
            if rec is not None and 0 <= start < end <= len(text) and text[start:end] == cite["evidence"]:
                cite_ok += 1
            else:
                cite_verify_failures += 1

        # Recall@k / MRR：只在有 gold_chunk_ids 的题上可算
        if gold_chunks:
            inter20 = len(set(gold_chunks) & set(hits[:20]))
            inter5 = len(set(gold_chunks) & set(hits[:5]))
            recall20.append(inter20 / len(gold_chunks))
            recall5.append(inter5 / len(gold_chunks))
            first = next((i for i, cid in enumerate(hits) if cid in set(gold_chunks)), None)
            rr.append(0.0 if first is None else 1.0 / (first + 1))
        else:
            skipped["recall"] += 1

        # 关键事实引用覆盖率：应答题的每条 required_citations 都要被某条引用覆盖
        required = item.get("required_citations") or []
        if should_abstain:
            skipped["coverage"] += 1
        else:
            emitted = {
                (str(c.get("chunk_id") or ""), str(c.get("evidence") or ""))
                for c in ans.get("citations") or []
            }
            covered = all(
                any(cid == str(req.get("chunk_id") or "")
                    and str(req.get("evidence") or "") in ev
                    for cid, ev in emitted)
                for req in required
            )
            coverage.append(bool(covered))

        # 拒答题正确率：期望行为确定（该拒 / 该答）的题才能算
        abstained = bool(ans.get("abstained"))
        if question:
            abstain_ok.append(abstained == should_abstain)

        # 方向题正确率：direct_relation 与 multihop 都依赖方向语义
        if category in ("direct_relation", "multihop") and not should_abstain:
            gold_answer = str(item.get("answer") or "")
            ok = (not abstained) and gold_answer and gold_answer in str(ans.get("answer") or "")
            direction_ok.append(bool(ok))
        else:
            skipped["direction"] += 1

        rows.append(
            {
                "question_id": item.get("question_id"),
                "category": category,
                "abstained": abstained,
                "should_abstain": should_abstain,
                "citations": len(ans.get("citations") or []),
                "top1": hits[0] if hits else "",
                "answer": str(ans.get("answer") or "")[:120],
            }
        )

    def _mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 6) if values else 0.0

    report = {
        "generation_id": gid,
        "gold_set": str(gold_path),
        "items_total_in_file": len(items),
        "items_evaluated": len(selected),
        "metrics": {
            "recall_at_20": {"value": _mean(recall20), "n": len(recall20),
                             "skipped": skipped["recall"]},
            "recall_at_5": {"value": _mean(recall5), "n": len(recall5),
                            "skipped": skipped["recall"]},
            "mrr": {"value": _mean(rr), "n": len(rr), "skipped": skipped["recall"]},
            "citation_location_accuracy": {
                "value": round(cite_ok / cite_total, 6) if cite_total else 0.0,
                "n": cite_total, "ok": cite_ok, "skipped": 0,
            },
            "required_citation_coverage": {
                "value": round(sum(1 for x in coverage if x) / len(coverage), 6) if coverage else 0.0,
                "n": len(coverage), "covered": sum(1 for x in coverage if x),
                "skipped": skipped["coverage"],
            },
            "abstain_accuracy": {
                "value": round(sum(1 for x in abstain_ok if x) / len(abstain_ok), 6) if abstain_ok else 0.0,
                "n": len(abstain_ok), "correct": sum(1 for x in abstain_ok if x),
                "skipped": skipped["abstain"],
            },
            "direction_accuracy": {
                "value": round(sum(1 for x in direction_ok if x) / len(direction_ok), 6) if direction_ok else 0.0,
                "n": len(direction_ok), "correct": sum(1 for x in direction_ok if x),
                "skipped": skipped["direction"],
            },
        },
        "citation_verify_failures": cite_verify_failures,
        "rows": rows,
    }
    return report


def _latest_identity_generation() -> str:
    """没指定 `--generation` 时取**最新**一个 ark-rag-3 世代。

    原实现写成 `for info in release.list_generations(): ... return`——只要遇到第一个
    ark-rag-3 就返回了。而 `list_generations()` 按目录名升序遍历，目录名以
    `g-<时间戳>-<hash>` 开头，升序即**最旧在前**。于是"取最新"实际取到的是最旧那个。

    这个缺陷当时看不出后果：磁盘上三个 ark-rag-3 世代内容完全一致
    （`index_content_digest` / `probe_digest` / `graph_build_fp` / `chunk_manifest_hash`
    全部相同，见 `docs/h阶段实测.md`），跑谁的数字都一样。**但只要出现第四个内容不同的
    世代，`--evaluate` 不带 `--generation` 就会静默评测一个旧世代**，报告照常打印七个
    指标、看不出任何异常——与台账里 `--limit` 只测一类、`focus_tier` 被当过滤属同一族。

    修法：按 `created_at` 取最大者，缺失时退回 `generation_id` 字典序
    （该 id 形如 `g-<UTC 时间戳>-<hash>`，字典序即时间序）。两种键用同一个 `max`。

    另修一处同源隐患：原实现对 `list_generations()` 返回的每一项直接 `.load()`，
    而该函数会把「无 Manifest」「Manifest 损坏」的目录也列出来，此时 `.load()`
    抛 `FileNotFoundError`——一个缺 Manifest 的残留目录能让整个默认路径报错退出。
    这里先看 `list_generations()` 已给的 status，缺 Manifest 的直接跳过。
    """
    candidates: list[tuple[str, str]] = []
    for info in release.list_generations():
        gid = str(info.get("generation_id") or "")
        status = str(info.get("status") or "")
        if not gid or status.startswith("(无 Manifest") or status.startswith("(Manifest 损坏"):
            continue
        manifest = release.Generation(gid).load()
        if str(manifest.get("graph_schema") or "") != config.GRAPH_SCHEMA_IDENTITY:
            continue
        created = str(info.get("created_at") or manifest.get("created_at") or "")
        candidates.append((created, gid))
    if not candidates:
        raise SystemExit("[2] 找不到任何 ark-rag-3 世代，请用 --generation 指定")
    # 先按 created_at 排；为空者用 gid 顶替，两种键都单调于创建时间
    candidates.sort(key=lambda row: (row[0] or row[1], row[1]))
    return candidates[-1][1]


def print_evaluation(report: dict) -> None:
    m = report["metrics"]
    print("=" * 74)
    print(f"Gold Set 评测  世代 {report['generation_id']}")
    print(f"题集 {report['gold_set']}  文件内 {report['items_total_in_file']} 条"
          f"  本次评测 {report['items_evaluated']} 条")
    print("-" * 74)
    print(f"{'指标':<28}{'值':<12}{'能算':<8}{'跳过':<8}说明")
    print(f"{'Recall@20':<28}{m['recall_at_20']['value']:<12.4f}"
          f"{m['recall_at_20']['n']:<8}{m['recall_at_20']['skipped']:<8}"
          "gold_chunk_ids 在召回前20中的平均占比")
    print(f"{'Recall@5':<28}{m['recall_at_5']['value']:<12.4f}"
          f"{m['recall_at_5']['n']:<8}{m['recall_at_5']['skipped']:<8}"
          "同上，取前5")
    print(f"{'MRR':<28}{m['mrr']['value']:<12.4f}"
          f"{m['mrr']['n']:<8}{m['mrr']['skipped']:<8}"
          "首个 gold 块名次的倒数均值")
    cl = m["citation_location_accuracy"]
    print(f"{'引用定位正确率':<28}{cl['value']:<12.4f}"
          f"{cl['n']:<8}{cl['skipped']:<8}"
          "text[start:end]==evidence 的引用占全部引用")
    rc = m["required_citation_coverage"]
    print(f"{'关键事实引用覆盖率':<28}{rc['value']:<12.4f}"
          f"{rc['n']:<8}{rc['skipped']:<8}"
          "应答题里 required_citations 全被覆盖的比例")
    ab = m["abstain_accuracy"]
    print(f"{'拒答题正确率':<28}{ab['value']:<12.4f}"
          f"{ab['n']:<8}{ab['skipped']:<8}"
          "拒答行为与 should_abstain 一致的比例")
    di = m["direction_accuracy"]
    print(f"{'方向题正确率':<28}{di['value']:<12.4f}"
          f"{di['n']:<8}{di['skipped']:<8}"
          "direct_relation/multihop 答案含标准谓词的比例")
    print("-" * 74)
    print(f"引用校验硬失败 {report['citation_verify_failures']} 条")
    by_cat: dict[str, list[dict]] = {}
    for row in report["rows"]:
        by_cat.setdefault(row["category"], []).append(row)
    print("分类明细（类别：条数 / 拒答数 / 平均引用数）：")
    for cat, rows in sorted(by_cat.items()):
        n_abs = sum(1 for r in rows if r["abstained"])
        avg_c = sum(r["citations"] for r in rows) / max(1, len(rows))
        print(f"    {cat:<18}{len(rows):>4} 条   拒答 {n_abs:>3}   平均引用 {avg_c:.2f}")


# ---------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="离线检索问答：单问回答或 Gold Set 评测（默认抽取式，不联网）"
    )
    parser.add_argument("question", nargs="?", default="", help="要问的问题")
    parser.add_argument("--generation", default="", help="世代 id（评测时默认取最近的 ark-rag-3）")
    parser.add_argument("--top-k", type=int, default=20, help="检索命中块数，默认 20")
    parser.add_argument("--graph-depth", type=int, default=1, help="图扩展深度，默认 1")
    parser.add_argument("--json", default="", help="把完整结果写到这个文件")
    parser.add_argument("--evaluate", action="store_true", help="跑 Gold Set 评测")
    parser.add_argument("--gold-set", default="", help="Gold Set 路径（评测必填）")
    parser.add_argument("--limit", type=int, default=50, help="评测题数上限，默认 50")
    args = parser.parse_args(argv)

    if args.evaluate:
        if not args.gold_set:
            print("[2] --evaluate 需要 --gold-set", file=sys.stderr)
            return 2
        if args.limit < 1:
            print("[2] --limit 必须 >= 1", file=sys.stderr)
            return 2
        report = evaluate(Path(args.gold_set), limit=args.limit,
                          generation_id=args.generation)
        print_evaluation(report)
        if args.json:
            Path(args.json).write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(f"完整结果 -> {args.json}")
        return 3 if report["citation_verify_failures"] else 0

    question = args.question.strip()
    if not question:
        print("[2] 需要给一个问题（或用 --evaluate）", file=sys.stderr)
        return 2
    if args.graph_depth < 0:
        print("[2] --graph-depth 不能为负", file=sys.stderr)
        return 2
    gid = args.generation or _latest_identity_generation()
    loaded = load_generation(gid)
    pipeline = build_pipeline(loaded, top_k=args.top_k, graph_depth=args.graph_depth)
    pipeline["generation_id"] = gid
    pipeline["entities"] = loaded["graph"].get("entities") or []

    result = ask_once(pipeline, question)
    verified = answer_mod.verify_citations(result["answer"], loaded["chunks"])
    result["answer_verified"] = verified
    result["answer"] = verified
    print_answer(result, generation_id=gid)

    if args.json:
        payload = {
            "generation_id": gid,
            "question": question,
            "query_analysis": result["query_analysis"],
            "search": result["search"],
            "expand": result["expand"],
            "context": {
                "context_digest": result["context"]["context_digest"],
                "stats": result["context"]["stats"],
            },
            "answer": verified,
        }
        Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"完整结果 -> {args.json}")

    if verified["verification"]["rejected"]:
        print(
            f"[3] 引用校验硬失败：{verified['verification']['rejected']} 条引用"
            f"在 text[start:end] 上对不上（已降置信度并剔除）",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
