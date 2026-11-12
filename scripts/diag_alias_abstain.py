"""L3 · `alias` / `single_chunk` 误拒答的**逐题闸门归因**（只读诊断，不改任何实现）。

## 要回答的问题（唯一目的）

480 道可判题里有 8 道 `judge.reason == "empty"`（答案空、`n_citations=0`），其中
6 道的 gold 就在 rank 1–2。本脚本对**每一道**题回答：

> 它被 `ExtractiveAnswerer.answer()` 里**哪一道**闸拦下的？

不允许给汇总数字了事 —— 汇总无法区分"门限太严"和"装配失败"，而这两件事的
修法完全相反（前者调门限，后者调 claim 生成）。

## `answer()` 里一共有 **五** 个 `abstained=True` 出口，不是三个

按代码执行顺序（`src/arkrag/answer.py`）：

| # | 位置 | `abstain_reason` |
|---|---|---|
| G1 | L682 `if not bool(qa.get("should_answer", True))` | `query_refused:<reason>` |
| G2 | L686 `if not units` | `no_context_units` |
| G3 | L690 `if float(top_score) < self.min_top_score` | `insufficient_retrieval(...)` |
| G4 | L878 `if not accepted`（claim 全被丢） | `no_supported_claim` |
| G5 | `verify_citations` L567（引用全被剔除） | `no_valid_citation` |

任务书写的是"三道闸"（G1/G2/G3），但 `judge.reason == "empty"` 只说明**答案文本为空**，
它对 G1–G5 五条路**完全不区分**。所以必须把五条都测，否则会把 G4/G5 误记成
"门限太严"，然后去调一个根本没起作用的常数。

## 两段测量

**第一段（逐题）**：对 8 道题走**生产档**（`pipeline.PROD_PROFILE`，含
`chapter_cap=4`）完整跑一遍 `ask_once` → `verify_citations`，逐题打印
`should_answer / top_score / min_top_score / len(units) / abstain_reason`，
并做一次**反事实重放**（`min_top_score=0.0`，其余一字不动）：如果松开 G3 后
答案非空且含 gold，那 G3 就是可修的；如果松开后改报 `no_supported_claim`，
那 G3 不是瓶颈，真正的瓶颈在 G4。

**第二段（全量）**：把 gold-v2 的全部 588 题（三 split 去重）跑一遍生产档，
记录每题的 `top_score`、类别、`gold` 是否在上下文、基线 `abstain_reason`。
用途是给"能不能放宽门限"提供**分母与反证**：只测 8 道题无法回答
"放宽之后会多开多少题、其中多少题会变成假答"。

## 为什么"反事实"只改 `self.min_top_score`

`answer()` 里 G1/G2 在 G3 之前，G4/G5 在 G3 之后。只改阈值、**不碰任何全局
常量、不碰 `quote_locate` / `bridge` / `chapter_cap`**，就保证反事实与基线的
唯一差异是 G3 这一道闸。这是本项目单变量的最低要求。

## 用法（必须走熔断器，一次 `load_generation()` 常驻 1.0–1.5 GB）

    py -3.14 scripts/run_guarded.py --max-commit-pct 92 --min-avail-mb 2048 \
        --stdout _diag_alias_out.txt --stderr _diag_alias_err.txt \
        -- py -3.14 scripts/diag_alias_abstain.py

    # 只跑 8 题（不跑全量扫描）
    ... scripts/diag_alias_abstain.py --no-sweep
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import answer as answer_mod  # noqa: E402
from arkrag import judge as judge_mod  # noqa: E402
from arkrag import pipeline as pipeline_mod  # noqa: E402

GENERATION = "g-20261103T132152Z-06a741"
GOLD_V2 = ROOT / "data" / "goldset" / "gold-v2.json"
GOLD_V1 = ROOT / "data" / "goldset" / "gold-v1.json"
DEFAULT_OUT = ROOT / "data" / "diag_alias_abstain.json"

#: 任务给定的 8 道 `judge.reason == "empty"` 题。**写死**，不从任何产物反推——
#: 反推的话，"这 8 道是哪 8 道"就成了一份会随基线漂移的清单，归因无法复核。
TARGETS: tuple[str, ...] = (
    "q-al-000010", "q-al-000025", "q-al-900019",
    "q-al-000005", "q-al-000015", "q-al-000030",
    "q-sc-900035", "q-al-900014",
)


def log(*a) -> None:
    print(*a, flush=True)


def head(title: str) -> None:
    log("\n" + "=" * 100)
    log(title)
    log("=" * 100)


def pick_items(items: list[dict], split: str, v1_ids: set[str]) -> list[dict]:
    """**逐字对齐** `scripts/eval_answer_quality.py::pick_items`。

    抄一份而不是 import：`eval_answer_quality` 是脚本不是包，import 它会连带跑
    argparse 与其它副作用。这里只抄"怎么选分母"这一条口径，并在测试里钉住
    它与原实现同义（`tests/test_l3_alias_gate.py`）。
    """
    if split == "clean":
        return [it for it in items
                if it.get("split") == "holdout" and it.get("gold_chunk_ids")
                and str(it.get("question_id")) not in v1_ids]
    if split == "all":
        return [it for it in items if it.get("gold_chunk_ids")]
    return [it for it in items
            if it.get("split") == split and it.get("gold_chunk_ids")]


def load_v1_ids() -> set[str]:
    if not GOLD_V1.exists():
        return set()
    doc = json.loads(GOLD_V1.read_text("utf-8"))
    items = doc.get("items") if isinstance(doc, dict) else doc
    return {str(it.get("question_id")) for it in items or []}


def chapter_texts(chunks: dict) -> dict[str, str]:
    """与 `eval_answer_quality.chapter_texts` 同义（judge 的 cross_chunk 要它）。"""
    out: dict[str, list[str]] = {}
    for rec in chunks.values():
        out.setdefault(str(rec.get("chapter") or ""), []).append(str(rec.get("text") or ""))
    return {k: "\n".join(v) for k, v in out.items()}


def gold_probe(item: dict) -> list[str]:
    """判"答案里有没有 gold"用的探针串（与 `judge._gold_strings` 同序）。"""
    probes = [str(item.get("answer") or "")]
    for x in item.get("acceptable_answers") or ():
        v = str(x or "")
        if v and v not in probes:
            probes.append(v)
    return [p for p in probes if p]


def gate_of(ans: dict, *, should_answer: bool, n_units: int,
            top_score: float, min_top_score: float) -> str:
    """从**返回的 dict** 反推是哪一道闸。纯函数，便于单测。

    只在 `abstained=True` 时有意义；`abstained=False` 返回 `""`。

    判定顺序必须与 `answer()` 的执行顺序一致：先 G1 再 G2 再 G3，之后才轮到
    G4/G5 —— 后者由 `abstain_reason` 自报（`no_supported_claim` /
    `no_valid_citation`），不靠猜。
    """
    if not ans.get("abstained"):
        return ""
    reason = str(ans.get("abstain_reason") or "")
    if reason.startswith("query_refused"):
        return "G1_query_refused"
    if reason == "no_context_units":
        return "G2_no_context_units"
    if reason.startswith("insufficient_retrieval"):
        return "G3_insufficient_retrieval"
    if reason == "no_supported_claim":
        return "G4_no_supported_claim"
    if reason == "no_valid_citation":
        return "G5_no_valid_citation"
    return f"unknown:{reason}"


def unit_digest(units: list[dict], gold_ids: set[str]) -> list[dict]:
    """上下文单元的速写（供报告逐题核对，不参与任何判据）。"""
    out = []
    for u in units:
        cid = str(u.get("chunk_id") or "")
        out.append({
            "chunk_id": cid,
            "is_gold": cid in gold_ids,
            "score": round(float(u.get("score") or 0.0), 6),
            "channels": sorted({str(s.get("channel") or "") for s in (u.get("sources") or [])}),
            "n_assertions": len(u.get("assertions") or ()),
            "n_evidence_spans": len(u.get("evidence_spans") or ()),
            "n_entities": len(u.get("entities") or ()),
            "text_chars": len(str(u.get("text") or "")),
        })
    return out


def probe_relevance(retriever, qa: dict, gold_ids: set[str], units: list[dict]) -> dict:
    """拆开 `build_context` 的断言过滤：`_assertion_relevant` 到底放过了什么、挡了什么。

    这是本脚本最关键的一段诊断，因为 §1 的结论把失败点指到了"上下文单元里
    一条断言都没有"。而断言进不进单元**只由** `retrieve._assertion_relevant`
    决定，它的判据是「断言的 subject_id 或 object_id ∈ 焦点实体 id」或
    「谓词 ∈ 关系提示」。所以必须逐条问：这些断言本可以进，是被哪个条件挡的。

    只读：不构造假设性上下文，只对 `assertions_by_chunk` 与 `_assertion_relevant`
    做只读调用。返回按块分组的计数与逐条明细（上限 8 条/块，避免报告膨胀）。
    """
    from arkrag import retrieve as retrieve_mod

    focus_ids = {str(c.get("entity_id") or "") for c in (qa.get("entity_candidates") or [])}
    focus_ids.discard("")
    hint_predicates = {
        str(h.get("canonical") or h.get("raw") or "")
        for h in (qa.get("relation_hints") or ())
    }
    hint_predicates.discard("")

    # 只查 gold 块与带断言的上下文块：全 19 个块逐条 dump 太吵。
    ctx_with_assert = [u for u in units if u.get("assertions")]
    watch = list(gold_ids) + [str(u.get("chunk_id") or "") for u in ctx_with_assert]
    out: dict[str, dict] = {}
    for cid in dict.fromkeys(c for c in watch if c):
        recs = retriever.assertions_by_chunk.get(cid) or []
        chunk_text = str((retriever.chunks.get(cid) or {}).get("text") or "")
        kept, dropped = [], []
        for rec in recs:
            ok = retrieve_mod._assertion_relevant(rec, focus_ids, hint_predicates)
            item = {
                "assertion_id": str(rec.get("assertion_id") or ""),
                "subject": retriever.display_name(str(rec.get("subject_id") or "")),
                "predicate": str(rec.get("predicate") or ""),
                "object": retriever.display_name(str(rec.get("object_id") or "")),
                "subject_id": str(rec.get("subject_id") or ""),
                "object_id": str(rec.get("object_id") or ""),
                "evidence": str(rec.get("evidence_text") or ""),
                "start": int(rec.get("evidence_start", -1)),
                "end": int(rec.get("evidence_end", -1)),
                # 偏移自洽性：这是 `verify_citations` 的硬判据，提前测一次
                "offset_ok": (
                    int(rec.get("evidence_start", -1)) >= 0
                    and chunk_text[int(rec.get("evidence_start", 0)):int(rec.get("evidence_end", 0))]
                    == str(rec.get("evidence_text") or "")
                ),
            }
            (kept if ok else dropped).append(item)
        out[cid] = {
            "is_gold": cid in gold_ids,
            "n_assertions": len(recs),
            "n_relevant_kept": len(kept),
            "n_relevant_dropped": len(dropped),
            "kept": kept[:8],
            "dropped_head": dropped[:8],
        }
    return {
        "focus_entity_ids": sorted(focus_ids),
        "hint_predicates": sorted(hint_predicates),
        "by_chunk": out,
    }


#: 问句里「称呼 → 规范名」这一问法的触发词。**故意只收「规范名」族**：
#: 本批 8 题里 5 道 alias 全是这个模板，而 `cross_chunk` / `multihop` 的问法
#: 一个都不含它。多收一个泛词（如「是什么」）会把整个题库吃进来。
ALIAS_NAME_TRIGGERS: tuple[str, ...] = ("规范名", "规范实体名")

#: 称呼后缀。语料里同一个实体在不同场合被叫「X」「X先生」「X叔叔」「X医生」。
#: 只在**原名解析失败**时逐条剥离重试，且剥完必须仍 ≥2 字（更短的无法作为引用）。
NAME_SUFFIXES: tuple[str, ...] = (
    "先生", "女士", "小姐", "叔叔", "阿姨", "干员", "医生", "大人", "阁下", "教官",
)


def quoted_terms(question: str) -> list[str]:
    """问句里「」框住的全部引文（保持原序、去重）。"""
    out: list[str] = []
    q = str(question or "")
    i = 0
    while True:
        a = q.find("「", i)
        if a < 0:
            break
        b = q.find("」", a + 1)
        if b < 0:
            break
        term = q[a + 1:b]
        if term and term not in out:
            out.append(term)
        i = b + 1
    return out


def resolve_alias(claims: dict, display, term: str) -> dict:
    """把一个称呼解析成**唯一**规范实体；不唯一或解析不到就如实返回。

    这是本次修复候选的核心判据，写死在这里而不是散在实现里：只有
    `unique=True` 时才允许作答。歧义（「女妖」之于两个实体）在 `query` 模块里
    被明确要求原样保留，所以这里也**不许取第一个**。
    """
    tries: list[dict] = []

    def attempt(text: str, kind: str) -> list[dict]:
        owners = claims.get(text) or []
        return [{"entity_id": str(o.get("entity_id") or ""),
                 "kind": str(o.get("kind") or ""),
                 "entity_name": display(str(o.get("entity_id") or "")),
                 "matched_text": text, "match": kind} for o in owners]

    owners = attempt(term, "exact")
    if not owners:
        for suffix in NAME_SUFFIXES:
            if term.endswith(suffix) and len(term) - len(suffix) >= 2:
                stem = term[: -len(suffix)]
                hit = attempt(stem, f"strip:{suffix}")
                if hit:
                    tries.append({"stem": stem, "suffix": suffix, "owners": hit})
                    owners = hit
                    break
    ids = {o["entity_id"] for o in owners if o["entity_id"]}
    return {
        "term": term,
        "owners": owners,
        "unique": len(ids) == 1,
        "n_owners": len(ids),
        "entity_id": (next(iter(ids)) if len(ids) == 1 else ""),
        "entity_name": (owners[0]["entity_name"] if owners else ""),
        "suffix_tries": tries,
        "ambiguous": len(ids) > 1,
    }


def locate_quote_line(units: list[dict], term: str) -> dict | None:
    """在上下文里找一行含该称呼、且偏移自洽的行（发引用用）。

    与 `_title_answer` 的定位法同形：命中多块时**不猜**，只取第一个能自洽定位的，
    并把候选数记下来。`text.find(line)` 之后必须回验 `text[start:start+len] == line`，
    否则同一行在块内重复出现时会定位到别处（`q-sc-000064` 的失效形态）。
    """
    hits: list[dict] = []
    for unit in units:
        text = str(unit.get("text") or "")
        if not text or term not in text:
            continue
        for line in text.splitlines():
            if len(line) < answer_mod.MIN_EVIDENCE_CHARS:
                continue
            if term not in line:
                continue
            start = text.find(line)
            if start < 0 or text[start:start + len(line)] != line:
                continue
            hits.append({"chunk_id": str(unit.get("chunk_id") or ""),
                         "start": start, "end": start + len(line), "evidence": line})
            break
    return {"n_candidates": len(hits), "first": (hits[0] if hits else None)}


def probe_title_path(question: str, units: list[dict]) -> dict:
    """`_title_answer` 为什么没触发——逐条查它自己的三个前置条件。

    `q-sc-900035` 明明含 `TITLE_TRIGGERS[0]`（「请给出该块的标题」）却没有走标题
    通路，必须查清是"没有引文"、"定位不到"、还是"title 不唯一"。这三者的修法
    完全不同，不许合并成一句"没触发"。
    """
    triggers = [t for t in answer_mod.TITLE_TRIGGERS if t in str(question or "")]
    quotes = [m for m in answer_mod._QUOTE_RE.findall(str(question or "")) if m.strip()]
    located: list[dict] = []
    for unit in units:
        title = str(unit.get("title") or "").strip()
        text = str(unit.get("text") or "")
        if not title or not text:
            continue
        for line in text.splitlines():
            if not line.strip() or len(line) < answer_mod.MIN_EVIDENCE_CHARS:
                continue
            if not any(quote in line for quote in quotes):
                continue
            start = text.find(line)
            if start < 0 or text[start:start + len(line)] != line:
                continue
            located.append({"chunk_id": str(unit.get("chunk_id") or ""),
                            "title": title, "evidence": line[:60]})
            break
    titles = sorted({x["title"] for x in located})
    return {
        "path_enabled": bool(answer_mod.TITLE_ANSWER_PATH),
        "triggers_hit": triggers,
        "quotes": quotes,
        "n_located": len(located),
        "titles": titles,
        "title_unique": len(titles) == 1,
        "located": located[:6],
        "would_fire": bool(triggers) and bool(quotes) and len(titles) == 1,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="L3：alias/single_chunk 误拒答的逐题闸门归因")
    ap.add_argument("--generation", default=GENERATION)
    ap.add_argument("--gold-set", default=str(GOLD_V2))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=0, help="全量扫描只跑前 N 题（0 = 全部）")
    ap.add_argument("--no-sweep", action="store_true",
                    help="只跑 8 题逐题归因，不跑全量扫描")
    args = ap.parse_args(argv)

    gold_doc = json.loads(Path(args.gold_set).read_text("utf-8"))
    items = list(gold_doc.get("items") if isinstance(gold_doc, dict) else gold_doc)
    by_id = {str(it.get("question_id") or ""): it for it in items}

    missing = [q for q in TARGETS if q not in by_id]
    if missing:
        log(f"[!] gold-v2 里找不到这些题号：{missing}")
        return 2

    # ---- 全量扫描的题池：三 split 并集、按 question_id 去重（clean ⊂ holdout） ----
    v1_ids = load_v1_ids()
    pool: dict[str, dict] = {}
    for sp in ("dev", "holdout", "clean"):
        for it in pick_items(items, sp, v1_ids):
            qid = str(it.get("question_id") or "")
            rec = pool.setdefault(qid, dict(it))
            rec.setdefault("_selected_by", set())
            rec["_selected_by"].add(sp)
    if args.limit:
        keep = list(pool)[: args.limit]
        pool = {k: pool[k] for k in keep}
    log(f"[l3] 全量池 {len(pool)} 题；目标 8 题全部在池内="
        f"{all(q in pool for q in TARGETS)}")

    # ---- 加载（一次，常驻 1.0–1.5 GB；熔断器已在外层） -------------------
    t0 = time.time()
    loaded = pipeline_mod.load_generation(args.generation)
    log(f"[load] {time.time() - t0:.1f}s chunks={len(loaded['chunks'])} "
        f"assertions={len(loaded['assertions'])}")
    pipe = pipeline_mod.build_pipeline(loaded, top_k=20, graph_depth=1)
    pipe["generation_id"] = args.generation
    pipe["entities"] = loaded["graph"].get("entities") or []
    options = pipeline_mod.apply_prod_profile(pipe)
    chap = chapter_texts(loaded["chunks"])
    answerer = pipe["answerer"]
    log(f"[prod] budget={pipe['retriever'].budget['max_context_tokens']} "
        f"options={json.dumps(options, ensure_ascii=False, sort_keys=True)} "
        f"min_top_score={answerer.min_top_score}")

    rows: list[dict] = []

    def measure(it: dict) -> dict:
        qid = str(it.get("question_id") or "")
        q = str(it.get("question") or "")
        gold = {str(c) for c in (it.get("gold_chunk_ids") or [])}
        probes = gold_probe(it)
        res = pipeline_mod.ask_once(pipe, q, options=options)
        qa = res.get("query_analysis") or {}
        stats = (res.get("search") or {}).get("stats") or {}
        units = (res.get("context") or {}).get("units") or []
        top_score = float(stats.get("top_score") or 0.0)
        ans = dict(res.get("answer") or {})
        base_gate = gate_of(ans, should_answer=bool(qa.get("should_answer", True)),
                            n_units=len(units), top_score=top_score,
                            min_top_score=answerer.min_top_score)
        verified = answer_mod.verify_citations(ans, loaded["chunks"])
        vtext = str(verified.get("answer") or "")
        vgate = gate_of(verified, should_answer=bool(qa.get("should_answer", True)),
                        n_units=len(units), top_score=top_score,
                        min_top_score=answerer.min_top_score)

        # ---- 反事实：只松开 G3（min_top_score → 0.0），别的一字不动 --------
        cf = {}
        if top_score < answerer.min_top_score:
            saved = answerer.min_top_score
            try:
                answerer.min_top_score = 0.0
                # `chapter_cap` 的门控必须与 `ask_once` 逐字一致（只对章节题生效），
                # 否则反事实与基线的差异就不止 G3 这一道闸。
                cf_cap = int(options.get("chapter_cap") or 0)
                if cf_cap and not pipeline_mod.is_chapter_question(q):
                    cf_cap = 0
                cf_ans = answerer.answer(
                    q, (res.get("context") or {}), generation_id=args.generation,
                    top_score=top_score, query_analysis=qa,
                    retrieval_stats=stats,
                    **({"chapter_cap": cf_cap} if cf_cap else {}),
                )
            finally:
                answerer.min_top_score = saved
            cf_ver = answer_mod.verify_citations(cf_ans, loaded["chunks"])
            cf_text = str(cf_ver.get("answer") or "")
            cf["relax_min_top_score"] = {
                "abstained": bool(cf_ver.get("abstained")),
                "abstain_reason": str(cf_ver.get("abstain_reason") or ""),
                "gate": gate_of(cf_ver, should_answer=bool(qa.get("should_answer", True)),
                                n_units=len(units), top_score=top_score,
                                min_top_score=0.0),
                "answer_chars": len(cf_text),
                "answer_head": cf_text[:120],
                "n_citations": len(cf_ver.get("citations") or ()),
                "gold_in_answer": bool(cf_text) and any(p and p in cf_text for p in probes),
                "accepted": int((cf_ans.get("stats") or {}).get("accepted") or 0),
                "candidates": int((cf_ans.get("stats") or {}).get("candidates") or 0),
                "dropped_no_citation": int(
                    (cf_ans.get("stats") or {}).get("dropped_no_citation") or 0),
                "dropped_unsupported": int(
                    (cf_ans.get("stats") or {}).get("dropped_unsupported") or 0),
                "dropped_ungrounded": int(
                    (cf_ans.get("stats") or {}).get("dropped_ungrounded") or 0),
            }

        # ---- 深度归因：G4 的"claim 为零"要从**四个上游**分别查 ----------------
        # `answer()` 的 `stats` 把候选数与三个丢弃原因分开了，但只有原样记下来
        # 才能区分"根本没有候选"(candidates=0) 与"有候选但没有引用锚点"
        # (dropped_no_citation>0) 与"引用不支持结论"(dropped_unsupported)。
        gold_units = [u for u in units if str(u.get("chunk_id") or "") in gold]
        gold_text = str((gold_units[0].get("text") if gold_units else "") or "")
        qa_cands = [
            {"name": str(c.get("name") or ""),
             "entity_id": str(c.get("entity_id") or ""),
             "reason": str(c.get("reason") or ""),
             "score": float(c.get("score") or 0.0)}
            for c in (qa.get("entity_candidates") or ())
        ]
        # 问句里被「」框起来的那段称呼（alias 题的题面形态），以及它**是否**
        # 被 NameTable 认成了实体候选。这两件事分开记，因为"问句里有没有这个称呼"
        # 与"系统知不知道它是谁"是 G4 成败的分水岭。
        quoted = []
        qstr = str(q)
        i = 0
        while True:
            a = qstr.find("「", i)
            b = qstr.find("」", a + 1) if a >= 0 else -1
            if a < 0 or b < 0:
                break
            quoted.append(qstr[a + 1:b])
            i = b + 1
        cand_names = {c["name"] for c in qa_cands}
        # ---- 修复候选探针：称呼能否解析成唯一规范名 + 能否定位一行做引用 ----
        claims = pipe["retriever"].query_context.claims
        display = pipe["retriever"].display_name
        alias_probe = []
        for term in quoted_terms(q):
            r = resolve_alias(claims, display, term)
            r["answer_in_gold"] = bool(str(it.get("answer") or "")) and \
                r["entity_name"] == str(it.get("answer") or "")
            r["locate"] = locate_quote_line(units, term)
            alias_probe.append(r)
        deep = {
            "qa_entity_candidates": qa_cands,
            "qa_relation_hints": [str(h.get("canonical") or h.get("raw") or "")
                                  for h in (qa.get("relation_hints") or ())],
            "quoted_terms": quoted,
            "quoted_in_candidates": [t for t in quoted if t in cand_names],
            "alias_probe": alias_probe,
            "title_probe": probe_title_path(q, units) if any(
                t in q for t in answer_mod.TITLE_TRIGGERS) else None,
            "base_stats": {
                "candidates": int((ans.get("stats") or {}).get("candidates") or 0),
                "accepted": int((ans.get("stats") or {}).get("accepted") or 0),
                "dropped_no_citation": int(
                    (ans.get("stats") or {}).get("dropped_no_citation") or 0),
                "dropped_unsupported": int(
                    (ans.get("stats") or {}).get("dropped_unsupported") or 0),
                "dropped_ungrounded": int(
                    (ans.get("stats") or {}).get("dropped_ungrounded") or 0),
            },
            "gold_unit_text_chars": len(gold_text),
            "gold_unit_has_answer": bool(gold_text) and str(it.get("answer") or "") in gold_text,
            "n_context_units_with_assertions": sum(
                1 for u in units if u.get("assertions")),
            "n_context_units_with_spans": sum(
                1 for u in units if u.get("evidence_spans")),
            # gold 块的断言数：为 0 说明 `_assertion_relevant` 把这块的断言全滤掉了，
            # 那是"焦点实体没对上"，不是"引用不支持结论"。
            "gold_unit_assertions": len((gold_units[0].get("assertions") if gold_units else []) or ()),
            "gold_unit_evidence_spans": len(
                (gold_units[0].get("evidence_spans") if gold_units else []) or ()),
            "gold_unit_entities": list((gold_units[0].get("entities") if gold_units else []) or ()),
        }

        return {
            "deep": deep,
            # 逐题都算：`q-al-000005` / `q-al-000015` 是"有 1 条断言但被
            # `_claim_supported_by_line` 丢掉"，与"0 条断言"是两种失效，必须并列可见。
            "relevance": probe_relevance(pipe["retriever"], qa, gold, units),
            "question_id": qid,
            "split": str(it.get("split") or ""),
            "selected_by": ",".join(sorted(it.get("_selected_by") or
                                           {str(it.get("split") or "")})),
            "category": str(it.get("category") or ""),
            "question": q,
            "gold_answer": str(it.get("answer") or ""),
            "should_abstain_gold": bool(it.get("should_abstain")),
            # ---- 三道（实为五道）闸的输入量 ----
            "should_answer": bool(qa.get("should_answer", True)),
            "qa_reason": str(qa.get("reason") or ""),
            "n_units": len(units),
            "top_score": round(top_score, 6),
            "min_top_score": float(answerer.min_top_score),
            "top_score_minus_min": round(top_score - answerer.min_top_score, 6),
            "n_hits": len((res.get("search") or {}).get("hits") or ()),
            "hit_rank": _hit_rank((res.get("search") or {}).get("hits") or (), gold),
            "gold_in_context": any(str(u.get("chunk_id") or "") in gold for u in units),
            # ---- 归因 ----
            "base_abstained": bool(ans.get("abstained")),
            "base_abstain_reason": str(ans.get("abstain_reason") or ""),
            "base_abstain_gate": base_gate,
            "base_answer_chars": len(str(ans.get("answer") or "")),
            "base_accepted": int((ans.get("stats") or {}).get("accepted") or 0),
            "base_candidates": int((ans.get("stats") or {}).get("candidates") or 0),
            "verified_abstained": bool(verified.get("abstained")),
            "verified_abstain_reason": str(verified.get("abstain_reason") or ""),
            "verified_abstain_gate": vgate,
            "verified_answer_chars": len(vtext),
            "n_citations": len(verified.get("citations") or ()),
            "cite_rejected": int((verified.get("verification") or {}).get("rejected") or 0),
            "gold_in_answer": bool(vtext) and any(p and p in vtext for p in probes),
            "counterfactual": cf,
            "units": unit_digest(units, gold),
        }

    def _hit_rank(hits: list[dict], gold: set[str]) -> int | None:
        for i, h in enumerate(hits, 1):
            if str(h.get("chunk_id") or "") in gold:
                return i
        return None

    # ---- 8 题逐题 ----
    head("§1 目标 8 题逐题闸门归因（生产档）")
    targets: list[dict] = []
    for qid in TARGETS:
        row = measure(by_id[qid])
        targets.append(row)
        log(f"\n--- {qid}  {row['category']}  split={row['split']} "
            f"gold={row['gold_answer']!r}")
        log(f"    should_answer={row['should_answer']} (reason={row['qa_reason']!r})"
            f"  n_units={row['n_units']}  n_hits={row['n_hits']}"
            f"  hit_rank={row['hit_rank']}  gold_in_ctx={row['gold_in_context']}")
        log(f"    top_score={row['top_score']:.6f}  min_top_score={row['min_top_score']}"
            f"  Δ={row['top_score_minus_min']:+.6f}")
        log(f"    => base_abstained={row['base_abstained']}"
            f"  reason={row['base_abstain_reason']!r}  gate={row['base_abstain_gate']}")
        log(f"    => after verify_citations: abstained={row['verified_abstained']}"
            f"  reason={row['verified_abstain_reason']!r}"
            f"  gate={row['verified_abstain_gate']}"
            f"  n_citations={row['n_citations']}"
            f"  cite_rejected={row['cite_rejected']}")
        dp = row["deep"]
        st = dp["base_stats"]
        log(f"    [深查] claim 漏斗: candidates={st['candidates']}"
            f" → accepted={st['accepted']}"
            f"  (no_citation {st['dropped_no_citation']} /"
            f" unsupported {st['dropped_unsupported']} /"
            f" ungrounded {st['dropped_ungrounded']})")
        log(f"           问句「」引文={dp['quoted_terms']}"
            f"  被认成实体候选={dp['quoted_in_candidates']}")
        log(f"           实体候选={[c['name'] + '(' + c['reason'] + ')' for c in dp['qa_entity_candidates']]}")
        log(f"           上下文带断言的单元 {dp['n_context_units_with_assertions']}/{row['n_units']}"
            f"   带证据偏移 {dp['n_context_units_with_spans']}/{row['n_units']}")
        log(f"           gold 块：断言 {dp['gold_unit_assertions']}"
            f"  证据偏移 {dp['gold_unit_evidence_spans']}"
            f"  实体 {dp['gold_unit_entities']}"
            f"  正文 {dp['gold_unit_text_chars']} 字"
            f"  含 gold={dp['gold_unit_has_answer']}")
        if row["counterfactual"]:
            cf = row["counterfactual"]["relax_min_top_score"]
            log(f"    [反事实] min_top_score→0.0: abstained={cf['abstained']}"
                f"  reason={cf['abstain_reason']!r}  gate={cf['gate']}"
                f"  accepted={cf['accepted']}/{cf['candidates']}"
                f"  (no_cite {cf['dropped_no_citation']} / unsupported "
                f"{cf['dropped_unsupported']} / ungrounded {cf['dropped_ungrounded']})")
            log(f"             answer={cf['answer_head']!r}"
                f"  gold_in_answer={cf['gold_in_answer']}"
                f"  n_citations={cf['n_citations']}")
        else:
            log("    [反事实] 未触发（top_score 未低于门限，G3 未拦）")
        rel = row.get("relevance")
        if rel:
            log(f"    [断言相关性] 焦点实体 id={rel['focus_entity_ids']}"
                f"  关系提示={rel['hint_predicates']}")
            for cid, info in rel["by_chunk"].items():
                tag = "GOLD" if info["is_gold"] else "ctx "
                log(f"           [{tag}] {cid}  断言 {info['n_assertions']}"
                    f"  kept={info['n_relevant_kept']}"
                    f" dropped={info['n_relevant_dropped']}")
                for a in info["kept"][:3]:
                    log(f"                 KEPT  {a['subject']}-{a['predicate']}-{a['object']}"
                        f"  off_ok={a['offset_ok']}  ev={a['evidence'][:40]!r}")
                for a in info["dropped_head"][:3]:
                    log(f"                 DROP  {a['subject']}-{a['predicate']}-{a['object']}"
                        f"  off_ok={a['offset_ok']}  ev={a['evidence'][:40]!r}")
            for p in dp.get("alias_probe") or []:
                loc = p.get("locate") or {}
                log(f"    [称呼解析] 「{p['term']}」→ 认领者 {p['n_owners']}"
                    f"  唯一={p['unique']}  规范名={p['entity_name']!r}"
                    f"  ==gold答案 {p['answer_in_gold']}"
                    f"  可定位行={loc.get('n_candidates')}")
                if p.get("suffix_tries"):
                    log(f"               剥后缀：{p['suffix_tries'][0]['stem']!r}"
                        f"（去 {p['suffix_tries'][0]['suffix']}）才命中")
            tp = dp.get("title_probe")
            if tp:
                log(f"    [标题通路] 触发词={tp['triggers_hit']} 引文={tp['quotes']}"
                    f" 定位到 {tp['n_located']} 块  title={tp['titles']}"
                    f" 唯一={tp['title_unique']} 会触发={tp['would_fire']}")

    # ---- 全量扫描 ----
    sweep: list[dict] = []
    if not args.no_sweep:
        head(f"§2 全量扫描 {len(pool)} 题（生产档）")
        t1 = time.time()
        for i, qid in enumerate(pool, 1):
            try:
                sweep.append(measure(pool[qid]))
            except Exception as exc:  # noqa: BLE001 - 单题失败必须留痕，不许丢分母
                sweep.append({"question_id": qid,
                              "category": str(pool[qid].get("category") or ""),
                              "error": f"{type(exc).__name__}: {exc}"})
            if i % 50 == 0:
                log(f"  {i}/{len(pool)}  {time.time() - t1:.1f}s")
        log(f"  完成 {len(sweep)} 题，{time.time() - t1:.1f}s")

    payload = {
        "schema": "arkrag.diag_alias_abstain/1",
        "generation": args.generation,
        "gold_set": str(args.gold_set),
        "prod_options": options,
        "token_budget": pipe["retriever"].budget["max_context_tokens"],
        "min_top_score": float(answerer.min_top_score),
        "targets": targets,
        "sweep": sweep,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with io.open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    log(f"\n落盘 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
