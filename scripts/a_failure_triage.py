# -*- coding: utf-8 -*-
"""A 阶段：分题型失败样本分层归因（检索与回答质量）。

## 这个脚本要回答的问题

候选世代 `g-20261103T132152Z-06a741` 在 Gold Set 320 题上 R@20 只有 0.6989，
而 Lead 的诊断是「gold 基本都在候选池里，排不进前 20」。**这一句必须逐题落证**，
否则后面所有「加召回 / 加融合 / 加重排」的讨论都建在印象上。

本项目已经因为「凭感觉加组件」栽过多次（RRF、向量融合、盲特征重排三次负结果，
见 `data/sweep_fusion.json` / `data/sweep_rrf_k.json` / `data/sweep_weighted_rrf.json`
/ `data/rerank_blind.json`）。所以这个脚本的第一产出不是分数，是**逐题一张表**：
每题 gold 到底在不在池里、排第几、进没进上下文、答案覆没覆盖、败在哪一层。

## 归因口径（一条题只给一个 primary）

按管线顺序判定，**第一个失败的阶段就是 primary**，因为下游阶段再准也救不回上游：

| 阶段 | primary 标签 | 判据 |
|---|---|---|
| 0 数据 | `标注错误` | required_citations 的 evidence 在块正文里定位不到 |
| 1 检索 | `检索未找到` | gold 不在候选池（`top_k=pool` 的合并结果里） |
| 2 排序 | `找到但未进前部` | gold 在池里，但名次 > top_k |
| 3 上下文 | `上下文未采用` | gold 进了前 top_k，但没进 context.units（预算截断） |
| 4 抽取 | `抽取缺事实` | gold 进了上下文，但 required 引用位置**没有断言 span 覆盖**，抽取式回答器无从引用 |
| 5 回答 | `回答未覆盖` | gold 进了上下文，但答案文本不含可接受答案 |
| 6 引用/拒答 | `引用或拒答错误` | 拒答行为与 should_abstain 不一致，或引用校验硬失败 |

**顺序不是随手排的，`抽取缺事实` 必须在上下文之后**：如果 gold 块压根没被
检索到，那「这块里没有断言」是个无从谈起的问题——它的 primary 是排序。
初版脚本把它排在检索之前，结果对抗题 3/3 全被判成 `抽取缺事实`，而该类
实测 R@20 只有 0.4375，一眼就是假的。先问「块到了没」，再问「块里的东西
能不能被用上」，这个次序不能颠倒。

另有一个**修饰标签** `合法同义证据漏标`：不单独做 primary（它不是管线故障，
而是 gold 标的那个块不是唯一能回答的块），只在「检索/排序失败」时附加，
提示「就算把 gold 顶上去，也可能本来就有别的块能答」。

`标注错误` 与 `合法同义证据漏标` 都**只标出来、不改 gold**——Gold Set 校准是
评测侧的活（工程师 B），这里只做技术归因。

## 池内名次怎么量（这里踩过一次，写清楚免得后人重犯）

直觉做法是「调 `search(top_k=200)` 看深池」，**这是错的**：
`retrieve.search()` 内部 `depth = top_k * CANDIDATE_POOL_MULT`，`top_k=200`
会把 depth 顶到 2000，于是 BM25 候选集变大、`_norm_scores()` 的归一化基准
（该路最高分）也跟着变，合并后前 20 的排序与生产 `search(top_k=20)`（depth=200）
不一致——初版脚本自检 20 题里 5 题不一致，读数不可信。

正确做法：把 `CANDIDATE_POOL_MULT` 临时压到 1，让 `top_k` 自己去承担深度，
于是 `search(top_k=200)` 的 depth 正好是生产 top_k=20 时的 200，BM25 / 实体 /
说话人三路的候选集与归一化基准与生产**完全同一条**，只是末尾不截到 20。

这样既不用另写一份合并公式（本项目已因「同一判定写两处、取最松那份生效」
出过三次同型事故），又保证前 20 与生产逐条一致——脚本 `selfcheck` 段会验，
不一致就以退出码 4 报错，不让你拿污染读数去写结论。

## 用法

    py -3 scripts\\a_failure_triage.py --probe-set union --pool 200
    py -3 scripts\\a_failure_triage.py --probe-set all  --pool 200 \\
        --out-jsonl data\\a_triage_all320.jsonl

退出码：0 = 跑通（分数低也是 0，与 `scripts/ask.py --evaluate` 同口径）；
2 = 输入不可用；4 = 自检失败（深池前 20 条与生产不一致，读数不可信）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import answer as answer_mod  # noqa: E402
from arkrag import config  # noqa: E402
from arkrag import pipeline as pipeline_mod  # noqa: E402
from arkrag import release, retrieve  # noqa: E402

# 生产评测口径（`scripts/ask.py:evaluate` 里写死的那两个数）。改动必须同步那边，
# 否则「归因用的管线」与「出分的管线」不是同一条，读数不可比。
EVAL_TOP_K = 20
EVAL_GRAPH_DEPTH = 1

LABELS = (
    "标注错误",
    "抽取缺事实",
    "检索未找到",
    "找到但未进前部",
    "上下文未采用",
    "回答未覆盖",
    "引用或拒答错误",
)
MODIFIER = "合法同义证据漏标"


# ---------------------------------------------------------------- 小工具
def answer_form(row_answers: dict) -> str:
    """答案形态：决定抽取式回答器**结构上**能不能给出这个答案。

    这不是分类癖好。抽取式回答器只会输出「主体 谓词 客体」三元组 claim
    （`answer.py` 的 `_claim_supported_by_line` 只认端点同行），所以：

    * `needs_metadata`（问「出自哪个块 / 该段标题」）——答案是一个块路径或标题，
      回答器没有「元数据回显」通道，结构上不可能命中；
    * `needs_list`（问「出现过哪些章节代号」）——答案是多值列表且要求**全部**，
      claim 是单条事实，拼不出完整集合；
    * `needs_judgment`（问「这是抽取矛盾还是剧情演进」）——答案是一句判断加理由，
      回答器没有生成式表述能力。

    这三类归因只能落 `回答未覆盖`（七类里没有更贴的），但**根因完全不同于
    「检到了但答案没写对」**——把它们的失败算进检索指标，会让人误以为要加召回。
    所以单独记 `answer_form`，做分层报告时按形态拆开看。
    """
    acceptable = [str(x) for x in (row_answers.get("acceptable_answers") or [])]
    question = str(row_answers.get("question") or "")
    if any(("#" in x or "/" in x) for x in acceptable):
        return "needs_metadata"
    for x in acceptable:
        toks = x.replace(",", " ").split()
        if len(toks) >= 2 and all(re.fullmatch(r"[a-z0-9_]+", t) for t in toks):
            return "needs_list"
    if "请给出该块的标题" in question or "请回答该段的标题" in question:
        return "needs_metadata"
    if "请列出全部章节代号" in question or "章节代号" in question:
        return "needs_list"
    if "矛盾" in question or "如何理解" in question:
        return "needs_judgment"
    return "needs_entity"


def quoted_span(question: str) -> str:
    """问句里用「」括起来的原文片段。single_chunk 类靠它判同义证据。"""
    m = re.search(r"「([^」]{4,})」", str(question or ""))
    return m.group(1) if m else ""


def doc_of(chunk_id: str) -> str:
    """`obt/main/level_st_17-01#001` → `obt/main/level_st_17-01`。

    按最后一个 `#` 切：doc 名里不含 `#`，块后缀一定是 `#NNN`。
    与 `scripts/diagnose_ranking.py:doc_of` 同口径（同 doc 兄弟块统计要用）。
    """
    cid = str(chunk_id or "")
    cut = cid.rfind("#")
    return cid[:cut] if cut > 0 else cid


def read_gold(path: Path) -> list[dict]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return doc["items"] if isinstance(doc, dict) else doc


def probe_ids(which: str) -> list[str]:
    """取固定探针题。

    `k1` = `scripts/k1_checks.py` 的 20 题（同时记在 `data/k2_gray.json`
    的 `probe_set.question_ids` 里）；`k4` = `scripts/k4_drill.py` 的 18 题
    （实测是 k1 的子集，union 仍是 20 题）；`union` = 两者并集去重。

    读产物而不是重新调 `pick_probes()`：探针清单本身是被台账引用过的口径，
    重新算一遍等于换了一个分母。
    """
    k1 = json.loads((ROOT / "data" / "k2_gray.json").read_text(encoding="utf-8"))
    k1_ids = [str(x) for x in k1["probe_set"]["question_ids"]]
    if which == "k1":
        return k1_ids
    k4 = json.loads((ROOT / "data" / "k4_drill.json").read_text(encoding="utf-8"))
    k4_ids = [str(x) for x in k4["probes"]]
    if which == "k4":
        return k4_ids
    return list(dict.fromkeys(k1_ids + k4_ids))


def span_covered(assertions_by_chunk: dict[str, list[dict]], chunk_text: str,
                 chunk_id: str, evidence: str) -> str:
    """required citation 的 (chunk_id, evidence) 有没有被某条断言的证据 span 覆盖。

    三种命中算覆盖：证据字符串与 `evidence_text` 互为子串，或区间有重叠。
    「区间有重叠」这一条是必要的：断言记的 evidence 往往比 gold 引用的那句话长
    （一句话里有两条关系时更明显），只比字符串会把覆盖判成没覆盖。

    返回 `covered` / `covered_by_assertion` / `no_assertion` / `not_in_text`。
    """
    pos = chunk_text.find(evidence)
    if not evidence or pos < 0:
        return "not_in_text"
    end = pos + len(evidence)
    rows = assertions_by_chunk.get(chunk_id) or []
    if not rows:
        return "no_assertion"
    for row in rows:
        try:
            start = int(row.get("evidence_start", -1))
            stop = int(row.get("evidence_end", -1))
        except (TypeError, ValueError):
            start = stop = -1
        if start >= 0 and stop > start and start < end and pos < stop:
            return "covered_by_assertion"
        text = str(row.get("evidence_text") or "")
        if text and (evidence in text or text in evidence):
            return "covered_by_assertion"
    return "no_assertion"


def deep_search(retriever, query_analysis: dict, pool: int) -> dict:
    """按**生产排序公式**取深池，只是不截到 top_k。

    做法：把 `CANDIDATE_POOL_MULT` 临时压到 1，让 `top_k` 自己承担深度。
    这样内部 `depth = pool * 1 = pool`，正好等于生产 `search(top_k=20)` 的
    `depth = 20 * CANDIDATE_POOL_MULT`，三路候选集与归一分母与生产同源；
    而如果直接调 `search(top_k=pool)`，depth 会被顶到 `pool * 10`，归一化
    基准漂移，前 20 排序就变了（初版脚本踩过，20 题里 5 题不一致）。

    `retrieve.CANDIDATE_POOL_MULT` 是模块级常量（`search()` 运行时读取），
    所以在调用前后赋值即可生效；用 `try/finally` 保证异常时也还原，
    不给后续调用留一个被改过的全局。
    """
    old = retrieve.CANDIDATE_POOL_MULT
    retrieve.CANDIDATE_POOL_MULT = 1
    try:
        return retriever.search(query_analysis, top_k=pool)
    finally:
        retrieve.CANDIDATE_POOL_MULT = old


def line_hits(text: str, names: list[str]) -> int:
    """有多少个名字出现在**同一行**里。

    用来判「这个块其实也回答了这道题」：关系类问题只要两个端点同行就能回答，
    不必是 gold 标的那一块。按行判而不是按整块判，否则一个几千字的块里
    什么名字都能凑齐。
    """
    best = 0
    for line in str(text or "").splitlines():
        hit = sum(1 for n in names if n and n in line)
        if hit > best:
            best = hit
            if best >= len(names):
                break
    return best


# ---------------------------------------------------------------- 主流程
def triage(args) -> int:
    t_start = time.time()
    gold_items = read_gold(Path(args.gold_set))
    if args.probe_set == "all":
        selected = list(gold_items)
    else:
        want = set(probe_ids(args.probe_set))
        selected = [it for it in gold_items if str(it.get("question_id")) in want]
        missing = want - {str(it.get("question_id")) for it in selected}
        if missing:
            print(f"[2] 探针里有 {len(missing)} 题不在 Gold Set 里：{sorted(missing)[:5]}",
                  file=sys.stderr)
            return 2
    if args.limit:
        selected = selected[: args.limit]
    print(f"题 {len(selected)}  世代 {args.generation}  深池 {args.pool}  "
          f"评测口径 top_k={EVAL_TOP_K} depth={EVAL_GRAPH_DEPTH}", flush=True)

    loaded = pipeline_mod.load_generation(args.generation)
    pipe = pipeline_mod.build_pipeline(loaded, top_k=EVAL_TOP_K,
                                       graph_depth=EVAL_GRAPH_DEPTH)
    pipe["generation_id"] = args.generation
    pipe["entities"] = loaded["graph"].get("entities") or []
    retriever = pipe["retriever"]
    chunks = loaded["chunks"]
    assertions_by_chunk: dict[str, list[dict]] = defaultdict(list)
    for row in loaded["assertions"]:
        assertions_by_chunk[str(row.get("chunk_id") or "")].append(row)
    print(f"载入完成 {time.time() - t_start:.0f}s  chunks={len(chunks)}  "
          f"assertions={len(loaded['assertions'])}", flush=True)

    rows: list[dict] = []
    selfcheck_bad = 0
    t_eval = time.time()
    for i, item in enumerate(selected):
        qid = str(item.get("question_id") or "")
        category = str(item.get("category") or "")
        question = str(item.get("question") or "")
        gold = [str(c) for c in item.get("gold_chunk_ids") or []]
        required = item.get("required_citations") or []
        should_abstain = bool(item.get("should_abstain"))

        result = pipeline_mod.ask_once(pipe, question)
        qa = result["query_analysis"]
        hits = [str(h.get("chunk_id") or "") for h in result["search"]["hits"]]
        ans = result["answer"]

        # ---- 深池名次：走生产排序公式，只是不截到 20 ----
        deep = deep_search(retriever, qa, args.pool)
        pool = [str(h.get("chunk_id") or "") for h in deep["hits"]]
        if pool[:EVAL_TOP_K] != hits:
            selfcheck_bad += 1

        gold_set = set(gold)
        gold_rank: dict[str, int | None] = {}
        for g in gold:
            gold_rank[g] = next((j + 1 for j, cid in enumerate(pool) if cid == g), None)
        ranks = [r for r in gold_rank.values() if r]
        first_rank = min(ranks) if ranks else None
        in_pool = first_rank is not None
        in_topk = bool(first_rank and first_rank <= EVAL_TOP_K)

        # 同 doc 兄弟块挤占：gold 未进前 top_k 时，同 doc 的其它块占了多少名额
        gold_docs = {doc_of(g) for g in gold}
        sibling_slots = sum(
            1 for cid in pool[:EVAL_TOP_K]
            if doc_of(cid) in gold_docs and cid not in gold_set
        )
        sibling_best = next(
            (j + 1 for j, cid in enumerate(pool)
             if doc_of(cid) in gold_docs and cid not in gold_set), None
        )

        # ---- 上下文是否采用 ----
        ctx_ids = [str(u.get("chunk_id") or "") for u in result["context"].get("units") or []]
        gold_in_ctx = [g for g in gold if g in ctx_ids]
        truncated = [str(x) for x in
                     (result["context"].get("stats") or {}).get("truncated_chunks") or []]
        gold_truncated = [g for g in gold if g in truncated]

        # ---- 标注层自检 ----
        cite_problems: list[dict] = []
        cover_states = Counter()
        for req in required:
            cid_ = str(req.get("chunk_id") or "")
            ev = str(req.get("evidence") or "")
            rec = chunks.get(cid_)
            text = str((rec or {}).get("text") or "")
            if rec is None:
                cover_states["chunk_missing"] += 1
                cite_problems.append({"chunk_id": cid_, "kind": "chunk_missing"})
                continue
            state = span_covered(assertions_by_chunk, text, cid_, ev)
            cover_states[state] += 1
            if state == "not_in_text":
                cite_problems.append({"chunk_id": cid_, "kind": "evidence_not_in_text",
                                      "evidence": ev[:60]})
        gold_not_cited = [g for g in gold
                          if required and g not in {str(r.get("chunk_id") or "") for r in required}]

        # ---- 回答层 ----
        answer_text = str(ans.get("answer") or "")
        acceptable = [str(x) for x in (item.get("acceptable_answers") or [])]
        if not acceptable:
            acceptable = [str(item.get("answer") or "")]
        acceptable = [x for x in acceptable if x]
        answer_covers = any(x in answer_text for x in acceptable) if answer_text else False
        abstained = bool(ans.get("abstained"))
        abstain_ok = abstained == should_abstain

        # 引用定位：与 `ask.evaluate` 同口径（text[start:end] 逐字相等）
        cite_total = cite_ok = 0
        for cite in ans.get("citations") or []:
            cite_total += 1
            rec = chunks.get(str(cite.get("chunk_id") or ""))
            text = str((rec or {}).get("text") or "")
            start, end = int(cite.get("start", -1)), int(cite.get("end", -1))
            if rec is not None and 0 <= start < end <= len(text) and text[start:end] == cite.get("evidence"):
                cite_ok += 1
        cite_bad = cite_total - cite_ok

        # 关键事实引用覆盖：与 `ask.evaluate` 同口径
        coverage_ok: bool | None = None
        if not should_abstain:
            emitted = {(str(c.get("chunk_id") or ""), str(c.get("evidence") or ""))
                       for c in ans.get("citations") or []}
            coverage_ok = all(
                any(cid_ == str(req.get("chunk_id") or "")
                    and str(req.get("evidence") or "") in ev for cid_, ev in emitted)
                for req in required
            )

        # ---- 合法同义证据漏标 ----
        # 只在「gold 没进前 top_k」时才有意义：gold 已经在前面时无所谓漏标。
        #
        # 判定分两种问法，不能一刀切：
        #   * 一般问法：池内前 20 有别的块**同一行**命中全部问句实体，或正文里
        #     出现可接受答案原文 → 那块其实也答得了。
        #   * 元数据问法（`needs_metadata`，问「这句话出自哪个块」）：答案本来就是
        #     块标题/路径，拿答案原文去块里找**永远找不到**（初版就这么误判过）。
        #     正确的判据是「池内前 20 有别的块正文里逐字含着问句引用的那句原文」——
        #     含着就说明那句话在多个块里重复出现，gold 不是唯一出处。
        names = [str(c.get("name") or "") for c in (qa.get("entity_candidates") or [])]
        names = [n for n in names if n]
        form = answer_form({"acceptable_answers": acceptable, "question": question})
        quote = quoted_span(question)
        synonym: dict | None = None
        if gold and not in_topk:
            for cid in pool[:EVAL_TOP_K]:
                if cid in gold_set:
                    continue
                text = str((chunks.get(cid) or {}).get("text") or "")
                hit = line_hits(text, names) if names else 0
                if form == "needs_metadata" and quote:
                    if quote in text:
                        synonym = {"chunk_id": cid, "rank": pool.index(cid) + 1,
                                   "kind": "quoted_text_repeated",
                                   "line_name_hits": hit, "answer_in_text": False}
                        break
                    continue
                ans_hit = any(len(x) >= 2 and x in text for x in acceptable)
                if (names and hit >= len(names)) or (ans_hit and hit >= 1) or (ans_hit and not names):
                    synonym = {"chunk_id": cid, "rank": pool.index(cid) + 1,
                               "kind": "line_entity_or_answer",
                               "line_name_hits": hit, "answer_in_text": bool(ans_hit)}
                    break

        # ---- 分层归因：按管线顺序取第一个失败阶段 ----
        #
        # `抽取缺事实` 的判据要收紧一层：**span 缺口必须真的挡住了答案**。
        # alias 类实测 22 题里有 21 题答案是**对的**（靠同块别的 claim 撞上了规范名），
        # 只是 required 指的那句证据底下没有断言，所以那条引用永远产不出来。
        # 这类是「引用覆盖」失败，不是「抽取能力不足」；混为一谈会让人去加抽取规则，
        # 而真正该动的是引用锚点选择。所以这里要求 `not answer_covers` 才成立，
        # 只挡引用不挡答案的情形落到最后一层 `引用或拒答错误`。
        layers: list[str] = []
        has_span_gap = bool(required) and "no_assertion" in cover_states
        span_blocks_answer = has_span_gap and not answer_covers
        if cite_problems:
            layers.append("标注错误")
        if gold and not in_pool:
            layers.append("检索未找到")
        if gold and in_pool and not in_topk:
            layers.append("找到但未进前部")
        if gold and in_topk and not gold_in_ctx:
            layers.append("上下文未采用")
        if span_blocks_answer and gold_in_ctx:
            layers.append("抽取缺事实")
        if (not should_abstain) and (not abstained) and gold_in_ctx and not answer_covers:
            layers.append("回答未覆盖")
        if (not abstain_ok) or cite_bad or coverage_ok is False:
            layers.append("引用或拒答错误")

        # 无 gold 的题（negative_abstain 的一部分）没有检索层，归因只能落到拒答层
        if not gold:
            layers = [x for x in layers if x not in
                      ("检索未找到", "找到但未进前部", "上下文未采用", "回答未覆盖")]
            if not layers and not abstain_ok:
                layers = ["引用或拒答错误"]

        primary = layers[0] if layers else ""

        row = {
            "question_id": qid,
            "category": category,
            "difficulty": item.get("difficulty"),
            "question": question,
            "gold_answer": str(item.get("answer") or ""),
            "acceptable_answers": acceptable,
            "should_abstain": should_abstain,
            "gold_chunk_ids": gold,
            "gold_ranks": gold_rank,
            "gold_first_rank": first_rank,
            "in_pool": in_pool,
            "in_top_k": in_topk,
            "pool_size": len(pool),
            "sibling_slots_in_top_k": sibling_slots,
            "sibling_best_rank": sibling_best,
            "gold_in_context": gold_in_ctx,
            "gold_truncated_from_context": gold_truncated,
            "context_units": ctx_ids,
            "context_stats": {
                "units": (result["context"].get("stats") or {}).get("units"),
                "tokens_used": (result["context"].get("stats") or {}).get("tokens_used"),
                "token_budget": (result["context"].get("stats") or {}).get("token_budget"),
            },
            "abstained": abstained,
            "abstain_reason": str(ans.get("abstain_reason") or ""),
            "abstain_ok": abstain_ok,
            "answer": answer_text,
            "answer_covers": answer_covers,
            "answer_form": form,
            "citations": cite_total,
            "citation_bad": cite_bad,
            "coverage_ok": coverage_ok,
            "required_citations": len(required),
            "required_cover_states": dict(cover_states),
            "span_gap": has_span_gap,
            # span 缺口是**只挡住了引用**还是**连答案都答不出**，这是两件事：
            # alias 21 题答案其实对（靠别的 claim 撞上了规范名），但 required 指的
            # 那句证据底下没有断言，所以引用永远产不出来 → 这是
            # 「关键事实引用覆盖 0.17」的直接成因，不是答不对。
            # 不拆开就会把引用指标问题误判成抽取能力不足，去加抽取规则。
            "span_gap_blocks_answer": bool(has_span_gap and not answer_covers),
            "annotation_problems": cite_problems,
            "gold_chunk_not_in_required_citations": gold_not_cited,
            "synonym_evidence": synonym,
            "layers": layers,
            "primary": primary,
            # 前 5 条命中留标题，供人工复核时不必回查块表
            "top5": [
                {"chunk_id": cid,
                 "title": str((chunks.get(cid) or {}).get("title") or "")[:40]}
                for cid in pool[:5]
            ],
            "gold_titles": [str((chunks.get(g) or {}).get("title") or "")[:40] for g in gold],
            "in_probe_set": qid in set(probe_ids("k1")),
        }
        rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(selected)}  {time.time() - t_eval:.0f}s", flush=True)

    if selfcheck_bad:
        print(f"[4] 自检失败：深池前 {EVAL_TOP_K} 条与生产 search 不一致 {selfcheck_bad} 题，"
              f"读数不可信，请先查 retrieve.search 是否被改动", file=sys.stderr)
        return 4

    # ---- 汇总 ----
    summary = summarize(rows)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_jsonl = Path(args.out_jsonl) if args.out_jsonl else (
        ROOT / "data" / f"a_triage_{args.probe_set}_{stamp}.jsonl")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    out_json = Path(args.out_json) if args.out_json else out_jsonl.with_suffix(".json")
    out_json.write_text(
        json.dumps({"schema": "a-triage-v1", "generation": args.generation,
                    # 相对仓库根写：绝对机台路径进 git 后换机器就错，
                    # `tests/test_portable_paths.py` 会为此判红（已实测踩过）。
                    "gold_set": config.rel_to_root(args.gold_set),
                    "probe_set": args.probe_set,
                    "pool": args.pool, "top_k": EVAL_TOP_K,
                    "graph_depth": EVAL_GRAPH_DEPTH,
                    "items_evaluated": len(rows), "summary": summary},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_summary(summary)
    print(f"\n逐题 -> {out_jsonl}\n汇总 -> {out_json}")
    print(f"总耗时 {time.time() - t_start:.0f}s")
    return 0


def summarize(rows: list[dict]) -> dict:
    """按类别汇总七层归因分布 + 每一项的通过率。"""
    by_cat: dict[str, dict] = {}
    for cat in sorted({r["category"] for r in rows}):
        sub = [r for r in rows if r["category"] == cat]
        graded = [r for r in sub if r["gold_chunk_ids"]]
        n_g = len(graded)
        dist = Counter(r["primary"] for r in sub if r["primary"])
        by_cat[cat] = {
            "n": len(sub),
            "n_with_gold": n_g,
            "pool_hit": round(sum(1 for r in graded if r["in_pool"]) / n_g, 4) if n_g else None,
            "top_k_hit": round(sum(1 for r in graded if r["in_top_k"]) / n_g, 4) if n_g else None,
            "context_hit": round(
                sum(1 for r in graded if r["gold_in_context"]) / n_g, 4) if n_g else None,
            "answer_cover": round(
                sum(1 for r in sub if not r["should_abstain"] and r["answer_covers"])
                / max(1, sum(1 for r in sub if not r["should_abstain"])), 4),
            "abstain_ok": round(sum(1 for r in sub if r["abstain_ok"]) / len(sub), 4),
            "coverage_ok": round(
                sum(1 for r in sub if r["coverage_ok"]) /
                max(1, sum(1 for r in sub if r["coverage_ok"] is not None)), 4),
            "primary_dist": dict(dist.most_common()),
            "synonym_flagged": sum(1 for r in sub if r["synonym_evidence"]),
            "annotation_flagged": sum(1 for r in sub if r["annotation_problems"]),
        }
    # 按答案形态汇总。这是**给优化方向用的**，与按类别汇总不是一回事：
    # 同一类里可能混着两种形态（negative_abstain 的 20 题要拒答、12 题要实体名），
    # 而 `needs_metadata` / `needs_list` / `needs_judgment` 三类共 128 题的失败
    # 与检索质量无关，混进整体 R@20 会误导「该加召回」。
    by_form: dict[str, dict] = {}
    for form_ in sorted({r["answer_form"] for r in rows}):
        sub = [r for r in rows if r["answer_form"] == form_]
        graded = [r for r in sub if r["gold_chunk_ids"]]
        n_g = len(graded)
        by_form[form_] = {
            "n": len(sub),
            "n_with_gold": n_g,
            "pool_hit": round(sum(1 for r in graded if r["in_pool"]) / n_g, 4) if n_g else None,
            "top_k_hit": round(sum(1 for r in graded if r["in_top_k"]) / n_g, 4) if n_g else None,
            "context_hit": round(
                sum(1 for r in graded if r["gold_in_context"]) / n_g, 4) if n_g else None,
            "answer_cover": round(
                sum(1 for r in sub if not r["should_abstain"] and r["answer_covers"])
                / max(1, sum(1 for r in sub if not r["should_abstain"])), 4),
            "primary_dist": dict(Counter(r["primary"] for r in sub if r["primary"]).most_common()),
        }
    flat = Counter(r["primary"] for r in rows if r["primary"])
    return {
        "by_category": by_cat,
        "by_answer_form": by_form,
        "primary_dist_total": dict(flat.most_common()),
        "n_failed": sum(1 for r in rows if r["primary"]),
        "n_total": len(rows),
    }


def print_summary(summary: dict) -> None:
    print("\n=== 分层归因（每类：题数 / 池命中 / 进前20 / 进上下文 / 答案覆盖 / 主因分布）===")
    head = f"{'类别':<18}{'题':>4}{'池':>8}{'前20':>8}{'上下文':>8}{'答案':>8}  主因分布"
    print(head)
    print("-" * 78)
    for cat, v in summary["by_category"].items():
        def f(x):
            return "  -  " if x is None else f"{x:.2f}"
        dist = "  ".join(f"{k}×{n}" for k, n in v["primary_dist"].items()) or "（全通过）"
        print(f"{cat:<18}{v['n']:>4}{f(v['pool_hit']):>8}{f(v['top_k_hit']):>8}"
              f"{f(v['context_hit']):>8}{f(v['answer_cover']):>8}  {dist}")
    print("-" * 78)
    print(f"总计 {summary['n_total']} 题，有主因 {summary['n_failed']} 题")
    print("主因合计：" + ("  ".join(f"{k}×{n}" for k, n in summary["primary_dist_total"].items())
                        or "（无失败）"))
    print("\n=== 按答案形态（决定抽取式回答器结构上能否给出这个答案）===")
    print(f"{'形态':<18}{'题':>4}{'池':>8}{'前20':>8}{'上下文':>8}{'答案':>8}  主因分布")
    print("-" * 78)
    for form_, v in summary["by_answer_form"].items():
        def g(x):
            return "  -  " if x is None else f"{x:.2f}"
        dist = "  ".join(f"{k}×{n}" for k, n in v["primary_dist"].items()) or "（全通过）"
        print(f"{form_:<18}{v['n']:>4}{g(v['pool_hit']):>8}{g(v['top_k_hit']):>8}"
              f"{g(v['context_hit']):>8}{g(v['answer_cover']):>8}  {dist}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="分题型失败样本分层归因")
    ap.add_argument("--generation", default="g-20261103T132152Z-06a741")
    ap.add_argument("--gold-set", default=str(ROOT / "data" / "goldset" / "gold-v1.json"))
    ap.add_argument("--probe-set", default="union", choices=("k1", "k4", "union", "all"),
                    help="k1/k4/union 只用固定探针；all 跑整个 Gold Set（会碰保留集风险）")
    ap.add_argument("--pool", type=int, default=200, help="深池深度，必须 ≥ top_k")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-jsonl", default="")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args(argv)
    if args.pool < EVAL_TOP_K:
        print(f"[2] --pool 必须 >= {EVAL_TOP_K}", file=sys.stderr)
        return 2
    if not Path(args.gold_set).is_file():
        print(f"[2] 找不到 Gold Set：{args.gold_set}", file=sys.stderr)
        return 2
    return triage(args)


if __name__ == "__main__":
    raise SystemExit(main())
