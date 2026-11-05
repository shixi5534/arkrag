"""重算引用覆盖闸门：把「断言在但被相关度判据挡掉」单列成闸门 1b。

为什么需要这一节
================
`docs/lead_引用覆盖漏斗实测.md` 的四闸门把「断言层没有 gold 那句话」当成一道闸。
但现场代码里 gold 那句话可能**已经抽出来了**，却因为
`src/arkrag/retrieve.py::_assertion_relevant()` 的判据（端点落在焦点实体上，
或谓词命中关系提示）不成立而**根本没有被登记**——它不是「抽取没抽到」，
也不是「回答层不发引用」，而是**准入判据挡掉了正确答案**。

这两件事对 P3 是完全不同的活：
    * 「抽取没抽到」→ 补抽取（改 goldset / 改抽取器）
    * 「抽到了但被挡」→ 改准入判据（改 `_assertion_relevant` 的输入侧）

不分开数，就会把一件代码层面的活记成抽取缺陷，然后拿它去当「天花板」。

本节只量、不改。所有数字都来自现场代码真跑 `pipeline.ask_once`。
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from arkrag import answer as answer_mod  # noqa: E402
from arkrag import pipeline, release, retrieve, subproc  # noqa: E402

QUOTE_RE = re.compile(r"「([^」]+)」")


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=subproc.child_env(),
        )
        return (out.stdout or "").strip() or "unknown"
    except Exception as exc:  # noqa: BLE001
        return f"unknown({exc})"


def load_gold(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {it["question_id"]: it for it in data["items"]}


def load_chunks(gen: release.Generation) -> dict[str, dict]:
    path = pipeline.resolve_chunks(gen, gen.load())
    out = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            out[rec["chunk_id"]] = rec
    return out


def load_assertions(gen: release.Generation) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = collections.defaultdict(list)
    with gen.assertion_path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            out[rec["chunk_id"]].append(rec)
    return out


def load_triage(path: Path) -> dict[str, dict]:
    out = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            out[rec["question_id"]] = rec
    return out


def coverage_ok(ans: dict, gold_item: dict) -> bool:
    """与 `scripts/ask.py:283-298` 同口径：单向 `req.evidence in cite.evidence`。"""
    emitted = [
        (str(c.get("chunk_id") or ""), str(c.get("evidence") or ""))
        for c in (ans.get("citations") or [])
    ]
    return all(
        any(
            cid == str(rc.get("chunk_id") or "") and str(rc.get("evidence") or "") in ev
            for cid, ev in emitted
        )
        for rc in (gold_item.get("required_citations") or [])
    )


def line_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    left = text.rfind("\n", 0, start) + 1
    right = text.find("\n", end)
    return left, (len(text) if right < 0 else right)


def reach_level(gold_item, by_chunk, chunks) -> str:
    """取最坏一条 required 的可达层级：span（断言短语直接含）> line（整行含）> none。

    这是「改引用粒度能不能救」的判据，与「抽取有没有抽到」是两件事：
        * span：现状管线就能发这条引用
        * line：断言落在这一行，但 `evidence_text` 只切了行内片段
          ⇒ 改发整行可救，零 gold 改动
        * none：这一行根本没有断言落在上面 ⇒ 抽取缺口，改粒度救不动
    """
    worst = "span"
    for rc in gold_item.get("required_citations") or []:
        cid = rc["chunk_id"]
        req = str(rc.get("evidence") or "")
        recs = by_chunk.get(cid, [])
        if any(req in str(a.get("evidence_text") or "") for a in recs):
            continue
        text = str((chunks.get(cid) or {}).get("text") or "")
        line_ok = False
        for a in recs:
            s, e = int(a.get("evidence_start", -1)), int(a.get("evidence_end", -1))
            if s < 0 or e <= s or e > len(text):
                continue
            left, right = line_bounds(text, s, e)
            if req in text[left:right]:
                line_ok = True
                break
        if not line_ok:
            return "none"
        worst = "line"
    return worst


def classify_gate(ans, gold_item, by_chunk, chunks, focus, hints, kept) -> str:
    """六态互斥：covered / 闸门3 / 闸门2 / 闸门1b / 闸门1L / 闸门1。"""
    if coverage_ok(ans, gold_item):
        return "covered"
    level = reach_level(gold_item, by_chunk, chunks)
    if level == "none":
        return "闸门1 抽取缺（行都没被抽）"
    if level == "line":
        return "闸门1L 仅整行可救"
    admit = True
    incl = True
    for rc in gold_item.get("required_citations") or []:
        cid = rc["chunk_id"]
        req = str(rc.get("evidence") or "")
        matched = [a for a in by_chunk.get(cid, []) if req in str(a.get("evidence_text") or "")]
        if not any(retrieve._assertion_relevant(a, focus, hints) for a in matched):
            admit = False
        if cid not in kept:
            incl = False
    # 因果顺序：块没进上下文时，`_assertion_relevant` 根本没机会执行——那条断言
    # 不进 `unit["assertions"]` 是因为**块不在**，不是因为被判不相关。先判 1b
    # 会把「检索层没送进来」错记成「准入门挡掉」，于是把 P2 的活算进 P3 的口径，
    # 分母也跟着虚增（v1 上 29 vs 19）。必须先判 incl。
    if not incl:
        return "闸门2 块没进上下文"
    if not admit:
        return "闸门1b 断言在但被相关度过滤"
    return "闸门3 块在手没发引用"


def frag_rescuable(gold_item, by_chunk, chunks) -> bool:
    """整行粒度能救的题：required 行有断言，但 required 句不是断言短语的子串。"""
    for rc in gold_item.get("required_citations") or []:
        cid = rc["chunk_id"]
        req = str(rc.get("evidence") or "")
        text = str((chunks.get(cid) or {}).get("text") or "")
        if any(req in str(a.get("evidence_text") or "") for a in by_chunk.get(cid, [])):
            continue
        for a in by_chunk.get(cid, []):
            s, e = int(a.get("evidence_start", -1)), int(a.get("evidence_end", -1))
            if s < 0 or e <= s or e > len(text):
                continue
            left, right = line_bounds(text, s, e)
            if req in text[left:right]:
                return True
    return False


def run_all(pipe, answerable, gold, by_chunk, chunks, retriever) -> dict[str, dict]:
    out = {}
    for qid in answerable:
        item = gold[qid]
        res = pipeline.ask_once(pipe, item["question"])
        qa = res["query_analysis"] or {}
        ans = res["answer"] or {}
        focus = {str(c.get("entity_id") or "") for c in (qa.get("entity_candidates") or [])}
        hints = {str(h.get("canonical") or h.get("raw") or "") for h in (qa.get("relation_hints") or [])}
        focus.discard("")
        hints.discard("")
        kept = {u["chunk_id"] for u in (res["context"] or {}).get("units") or []}
        gate = classify_gate(ans, item, by_chunk, chunks, focus, hints, kept)
        out[qid] = {
            "gate": gate,
            "category": item["category"],
            "coverage_ok": gate == "covered",
            "n_citations": len(ans.get("citations") or []),
            "abstained": bool(ans.get("abstained")),
            "n_focus": len(focus),
            "n_hints": len(hints),
            "_res": res,
        }
    return out


def section_gate_funnel(base: dict, cand: dict) -> dict:
    order = ["covered", "闸门3 块在手没发引用", "闸门2 块没进上下文",
             "闸门1b 断言在但被相关度过滤", "闸门1L 仅整行可救", "闸门1 抽取缺（行都没被抽）"]
    rows = []
    for name, data in (("baseline(cooc0.0)", base), ("cooc0.1", cand)):
        cnt = collections.Counter(v["gate"] for v in data.values())
        rows.append({"配置": name, **{k: cnt.get(k, 0) for k in order}, "合计": sum(cnt.values())})
    moves = [
        {"question_id": q, "baseline": base[q]["gate"], "cooc0.1": cand[q]["gate"]}
        for q in base
        if base[q]["gate"] != cand[q]["gate"]
    ]
    return {
        "说明": "六态互斥，合计恒等于应答题数；闸门1L 与闸门1 把「改粒度能救」与「抽取缺口」分开",
        "各配置": rows,
        "两档归类不同的题": moves,
        "covered 增量": sum(1 for v in cand.values() if v["gate"] == "covered")
                       - sum(1 for v in base.values() if v["gate"] == "covered"),
    }


def section_gate1b(base: dict, gold: dict, by_chunk: dict, chunks: dict, retriever) -> dict:
    qids = [q for q, v in base.items() if v["gate"] == "闸门1b 断言在但被相关度过滤"]
    rows = []
    for qid in sorted(qids):
        item = gold[qid]
        rc = (item.get("required_citations") or [{}])[0]
        cid = rc.get("chunk_id", "")
        text = str((chunks.get(cid) or {}).get("text") or "")
        title = str((chunks.get(cid) or {}).get("title") or "")
        quotes = QUOTE_RE.findall(item["question"])
        # 强行放行后 claim 层能不能立得住。
        # 注意：`_claim_supported_by_line` 要的是 claim 里带 start/end（先验过偏移，
        # 偏移非法直接 ok=False）。只传 subject/object/predicate 会让它**恒返回 False**，
        # 那是假的「立不住」——第一版就是这么写错的，和之前的
        # `str(res["answer"])` 是同一类：给函数喂了它读不到的字段。
        claim_ok_any = False
        # 只该看「含 gold 那句话的那条断言」——放行后能发出去的就是它。
        matched = [a for a in by_chunk.get(cid, [])
                   if str(rc.get("evidence") or "") in str(a.get("evidence_text") or "")]
        for a in matched:
            subj = retriever.display_name(str(a.get("subject_id") or ""))
            obj = retriever.display_name(str(a.get("object_id") or ""))
            claim = {
                "subject": subj,
                "object": obj,
                "predicate": a.get("predicate"),
                "start": int(a.get("evidence_start", -1)),
                "end": int(a.get("evidence_end", -1)),
            }
            if answer_mod._claim_supported_by_line(claim, text).get("ok"):
                claim_ok_any = True
                break
        rows.append({
            "question_id": qid,
            "category": item["category"],
            "n_matched_assertions": len(matched),
            "answer_equals_chunk_title": item.get("answer") == title,
            "required_quoted_in_question": str(rc.get("evidence") or "") in item["question"],
            "n_quotes_in_question": len(quotes),
            "claim_layer_can_stand": claim_ok_any,
        })
    n = len(rows)
    return {
        "题数": n,
        "按题型": dict(collections.Counter(r["category"] for r in rows)),
        "答案恰等于块 title": sum(1 for r in rows if r["answer_equals_chunk_title"]),
        "required 被问句原样引用": sum(1 for r in rows if r["required_quoted_in_question"]),
        "问句含「」引文": sum(1 for r in rows if r["n_quotes_in_question"]),
        "强行放行后 claim 层仍立不住": n - sum(1 for r in rows if r["claim_layer_can_stand"]),
        "机制": ("`retrieve._assertion_relevant()` 只认「端点落在焦点实体」或「谓词命中关系提示」；"
                 "这类问句把要引的原句直接放在「」里，里面既不是实体名也不是关系词，"
                 "于是 gold 那条断言根本不被登记 ⇒ 回答层连候选都拿不到"),
        "逐题": rows,
    }


def section_gate1b_convert(base: dict, cand: dict, gold: dict, by_chunk: dict,
                            chunks: dict, ans_by_qid: dict) -> dict:
    """闸门1b 的 29 题：准入放行之后，**指标会不会过**、**答案对不对**。

    这两件事必须分开报。`required_citation_coverage` 的判据是
    「`req.evidence` 是某条已发引用 `evidence` 的子串」，
    而那 24 道题里 `required.evidence` 恰好**等于**某条断言的 `evidence_text`。
    所以只要那条断言被准入、被引用，覆盖率就会变真——
    **哪怕答案文本仍是「主体 谓词 客体」，根本不是题目要的那个标题。**

    只报覆盖率会得出「修准入判据就能提 24 题」；只报答案对错会漏掉
    「指标涨了但答的还是错的」这个更难发现的问题。两个都报。
    """
    qids = [q for q, v in base.items() if v["gate"] == "闸门1b 断言在但被相关度过滤"]
    rows = []
    for qid in sorted(qids):
        item = gold[qid]
        res = ans_by_qid[qid]
        kept = {u["chunk_id"] for u in (res["context"] or {}).get("units") or []}
        ans_text = str((res["answer"] or {}).get("answer") or "")
        gold_ans = str(item.get("answer") or "")
        cov_if_admitted = True
        blocks_kept = True
        for rc in item.get("required_citations") or []:
            cid = rc["chunk_id"]
            req = str(rc.get("evidence") or "")
            if cid not in kept:
                blocks_kept = False
            if not any(req in str(a.get("evidence_text") or "") for a in by_chunk.get(cid, [])):
                cov_if_admitted = False
        answer_hits = bool(gold_ans) and (gold_ans in ans_text)
        rows.append({
            "question_id": qid,
            "category": item["category"],
            "all_required_blocks_in_context": blocks_kept,
            "coverage_would_pass_if_admitted": blocks_kept and cov_if_admitted,
            "answer_text_contains_gold_answer": answer_hits,
            "gold_answer": gold_ans,
            "answer_text": ans_text[:80],
        })
    n = len(rows)
    return {
        "题数": n,
        "required 块已在上下文": sum(1 for r in rows if r["all_required_blocks_in_context"]),
        "准入放行后覆盖率会变真": sum(1 for r in rows if r["coverage_would_pass_if_admitted"]),
        "准入放行后答案文本仍不含 gold 答案": sum(
            1 for r in rows if not r["answer_text_contains_gold_answer"]),
        "既会过覆盖率又答不上": sum(
            1 for r in rows if r["coverage_would_pass_if_admitted"]
            and not r["answer_text_contains_gold_answer"]),
        "判读": ("「覆盖率会变真」与「答案对不对」是两件事：多数题的 gold 答案就是块标题，"
                 "而抽取式回答器只会输出三元组，于是覆盖率能过、答案仍是错的。"
                 "这类增益不能算作回答质量提升，只能算口径修正"),
        "逐题": rows,
    }


def section_disjoint(base: dict, gold: dict, by_chunk: dict, chunks: dict) -> dict:
    """三集合互斥。**口径必须一致**：都用「取最坏一条 required」。

    上一版这里用 `frag_rescuable()`（任一 required 整行可救）去比
    `classify_gate()`（全部 required 都 span 可达），两个分母不同，
    于是算出 `frag ∩ 闸门1 = 55` 而不是 0——这不是发现，是口径错配。
    现在统一到「最坏一条」：level=line 才算 frag，level=none 才算抽取缺。
    """
    levels = {q: reach_level(gold[q], by_chunk, chunks) for q in base}
    frag = {q for q, lv in levels.items() if lv == "line"}
    none = {q for q, lv in levels.items() if lv == "none"}
    g1b = {q for q, v in base.items() if v["gate"] == "闸门1b 断言在但被相关度过滤"}
    ok = {q for q, v in base.items() if v["gate"] == "covered"}
    return {
        "整行粒度可救(frag, 最坏一条)": len(frag),
        "抽取缺口(none, 最坏一条)": len(none),
        "闸门1b(span 可达但被准入判据挡)": len(g1b),
        "已覆盖": len(ok),
        "frag ∩ 闸门1b": len(frag & g1b),
        "frag ∩ 抽取缺口": len(frag & none),
        "闸门1b ∩ 抽取缺口": len(g1b & none),
        "三者合计": len(frag) + len(none) + len(g1b),
        "结论": ("口径统一到「最坏一条 required」后，frag / 抽取缺口 / 闸门1b 三者互斥："
                 "frag 要改引用粒度、抽取缺口要补抽取、闸门1b 要改准入判据。"
                 "这是三件不同的活，任何把它们合成一个「天花板」的说法都会导致修错地方"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generation", default="g-20261103T132152Z-06a741")
    ap.add_argument("--gold", default="data/goldset/gold-v1.json")
    ap.add_argument("--triage", default="data/a_triage_all320.jsonl")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    gen = release.Generation(args.generation)
    loaded = pipeline.load_generation(gen.generation_id)
    pipe = pipeline.build_pipeline(loaded, top_k=20, graph_depth=1)
    retriever = pipe["retriever"]

    gold = load_gold(ROOT / args.gold)
    chunks = load_chunks(gen)
    by_chunk = load_assertions(gen)
    triage = load_triage(ROOT / args.triage)
    answerable = [q for q, t in triage.items() if not t.get("should_abstain")]

    print(f"应答题 {len(answerable)}，块 {len(chunks)}，有断言的块 {len(by_chunk)}")

    old = retrieve.COOC_BOOST_WEIGHT
    try:
        retrieve.COOC_BOOST_WEIGHT = 0.0
        base = run_all(pipe, answerable, gold, by_chunk, chunks, retriever)
        retrieve.COOC_BOOST_WEIGHT = 0.1
        cand = run_all(pipe, answerable, gold, by_chunk, chunks, retriever)
    finally:
        retrieve.COOC_BOOST_WEIGHT = old

    ans_by_qid = {q: v.pop("_res") for q, v in base.items()}

    out = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_revision(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "command": "py -3 scripts/lead_gate_recheck.py",
        "generation": gen.generation_id,
        "answerable": len(answerable),
        "1_闸门漏斗": section_gate_funnel(base, cand),
        "2_闸门1b画像": section_gate1b(base, gold, by_chunk, chunks, retriever),
        "2b_闸门1b兑现": section_gate1b_convert(base, cand, gold, by_chunk, chunks, ans_by_qid),
        "3_三集合互斥": section_disjoint(base, gold, by_chunk, chunks),
    }

    print("\n== 1 闸门漏斗 ==")
    for row in out["1_闸门漏斗"]["各配置"]:
        print("  " + "  ".join(f"{k}={v}" for k, v in row.items()))
    print(f"  covered 增量 {out['1_闸门漏斗']['covered 增量']:+d}")
    for mv in out["1_闸门漏斗"]["两档归类不同的题"]:
        print(f"    {mv['question_id']}: {mv['baseline']} -> {mv['cooc0.1']}")

    g1b = out["2_闸门1b画像"]
    print("\n== 2 闸门1b 画像 ==")
    for k in ("题数", "答案恰等于块 title", "required 被问句原样引用", "问句含「」引文",
              "强行放行后 claim 层仍立不住"):
        print(f"  {k}: {g1b[k]}")
    print(f"  按题型: {g1b['按题型']}")

    dj = out["3_三集合互斥"]
    print("\n== 3 三集合互斥 ==")
    for k, v in dj.items():
        print(f"  {k}: {v}")

    cv = out["2b_闸门1b兑现"]
    print("\n== 2b 闸门1b 兑现（准入放行之后） ==")
    for k in ("题数", "required 块已在上下文", "准入放行后覆盖率会变真",
              "准入放行后答案文本仍不含 gold 答案", "既会过覆盖率又答不上"):
        print(f"  {k}: {cv[k]}")

    if args.json:
        path = ROOT / args.json
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n已写 {path} ({path.stat().st_size} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
