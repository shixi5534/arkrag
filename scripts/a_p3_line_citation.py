"""P3 第一轮：引用粒度（断言短语 → 断言所在整行）真跑验收。

唯一变量
========
`answer.CITATION_WHOLE_LINE`：False = 发断言短语（现状），True = 发整行。
同一次进程、同一份数据、同一份代码，只翻这一个常量。

为什么这算单变量（必须自证，不能只论证）
======================================
Lead 提醒：`_claim_supported_by_line` 的判据是「端点名出现在 evidence 所在整行」，
扩 span 后它的输入可能变。**源码事实**：它的入参是 `claim` 与 `span["text"]`，读的是
`claim["start"]/["end"]`——那是**断言 span**，不是引用。本脚本用三条硬证据验证：

1. 两侧 `answer` 正文必须逐字相同（claim 取舍没变）；
2. 两侧 `stats.dropped_unsupported` / `dropped_no_citation` / `dropped_ungrounded`
   / `accepted` 必须逐题相同（判据输入没变）；
3. 两侧 claim 条数与 `assertion_id` 集合必须逐题相同。

三条任一不成立 ⇒ 这不是单变量实验，脚本 exit 4 并拒绝出结论。

必须同时报的四个数（Lead 明确要求）
==================================
1. `coverage_ok` 题数，**三个分母并列**：全部应答题 / 断言层可达 / 两道上游闸都过；
2. 引用定位正确率 —— **两侧各自的引用条数 + 独立同侧的逐条通过率**（塌缩会让分母变小，
   1.0000 不必然是变好）；
3. 引用硬失败数；
4. 救回 / 丢失逐题清单（救的按题型、丢的给题号）。

另报「塌缩」：`(chunk_id, evidence)` 去重后的引用条数，衡量同行多条断言被压成一条。
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import answer as answer_mod, config, pipeline, subproc  # noqa: E402

DEFAULT_GEN = "g-20261103T132152Z-06a741"
DEFAULT_GOLD = ROOT / "data" / "goldset" / "gold-v2.json"


def rel(path: Path) -> str:
    return config.rel_to_root(path)


def git_rev() -> str:
    try:
        import subprocess

        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
            capture_output=True, text=True, timeout=20, check=False,
            # 子进程输出按 UTF-8 解码，父子两端编码必须一致，见 src/arkrag/subproc.py
            env=subproc.child_env(),
        )
        return (out.stdout or "").strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def load_dev(gold_path: Path) -> list[dict]:
    """dev 调参集：**全量**，不切片。

    `ask.py --limit` 曾在按类聚簇的 gold 上只覆盖到 single_chunk 一类，此后所有
    探针一律全量。这里只按 split 取 dev，再做任何 `[:n]` 都是错。
    """
    doc = json.loads(gold_path.read_text(encoding="utf-8"))
    items = [it for it in doc["items"] if it.get("split") == "dev"]
    items.sort(key=lambda it: it["question_id"])
    return items


def required_chunks(item: dict) -> list[str]:
    return [str(rc.get("chunk_id") or "") for rc in item.get("required_citations") or []]


def span_reachable(item: dict, by_chunk: dict[str, list[dict]]) -> bool:
    """每条 required 的 evidence 是否是某条断言 `evidence_text` 的子串。

    与 `scripts/ask.py:283-298` 的**单向子串**真判据同口径（不是我自己发明的量）。
    """
    for rc in item.get("required_citations") or []:
        cid = str(rc.get("chunk_id") or "")
        req = str(rc.get("evidence") or "")
        if not any(req in str(rec.get("evidence_text") or "")
                   for rec in by_chunk.get(cid) or []):
            return False
    return True


def coverage_of(ans: dict, required: list[dict]) -> bool:
    """与 `scripts/ask.py:283-298` 逐字同口径。"""
    emitted = {
        (str(c.get("chunk_id") or ""), str(c.get("evidence") or ""))
        for c in ans.get("citations") or []
    }
    return all(
        any(cid == str(req.get("chunk_id") or "") and str(req.get("evidence") or "") in ev
            for cid, ev in emitted)
        for req in required
    )


def citation_scan(ans: dict, chunks: dict) -> tuple[int, int, int]:
    """(条数, 逐条定位通过数, 硬失败数)，与 `scripts/ask.py:262-270` 同口径。"""
    total = ok = bad = 0
    for cite in ans.get("citations") or []:
        total += 1
        rec = chunks.get(str(cite.get("chunk_id") or ""))
        text = str((rec or {}).get("text") or "")
        start, end = int(cite.get("start", -1)), int(cite.get("end", -1))
        if rec is not None and 0 <= start < end <= len(text) and text[start:end] == cite.get("evidence"):
            ok += 1
        else:
            bad += 1
    return total, ok, bad


def run_config(pipeline_obj, items, chunks, by_chunk, whole_line: bool) -> list[dict]:
    answer_mod.CITATION_WHOLE_LINE = bool(whole_line)
    rows: list[dict] = []
    for it in items:
        t0 = time.perf_counter()
        res = pipeline.ask_once(pipeline_obj, str(it.get("question") or ""))
        ms = (time.perf_counter() - t0) * 1000.0
        ans = res["answer"]
        required = it.get("required_citations") or []
        ctx_ids = {str(u.get("chunk_id") or "") for u in res["context"]["units"]}
        cite_total, cite_ok, cite_bad = citation_scan(ans, chunks)
        distinct = len({(str(c.get("chunk_id") or ""), str(c.get("evidence") or ""))
                        for c in ans.get("citations") or []})
        st = ans.get("stats") or {}
        rows.append({
            "question_id": it["question_id"],
            "category": it.get("category") or "",
            "origin": it.get("origin") or "",
            "should_abstain": bool(it.get("should_abstain")),
            "n_required": len(required),
            "required_in_context": bool(required) and all(
                c in ctx_ids for c in required_chunks(it)),
            "span_reachable": span_reachable(it, by_chunk),
            "covered": (not it.get("should_abstain")) and bool(required)
            and coverage_of(ans, required),
            "abstained": bool(ans.get("abstained")),
            "answer": str(ans.get("answer") or ""),
            "answer_sha": hashlib.sha256(
                str(ans.get("answer") or "").encode("utf-8")).hexdigest()[:16],
            "citations": cite_total,
            "citations_distinct": distinct,
            "cite_ok": cite_ok,
            "cite_bad": cite_bad,
            "claim_ids": sorted(str(c.get("assertion_id") or "")
                                for c in ans.get("citations") or []),
            "dropped_unsupported": int(st.get("dropped_unsupported") or 0),
            "dropped_no_citation": int(st.get("dropped_no_citation") or 0),
            "dropped_ungrounded": int(st.get("dropped_ungrounded") or 0),
            "accepted": int(st.get("accepted") or 0),
            "candidate_claims": int(st.get("candidates") or 0),
            "elapsed_ms": round(ms, 3),
        })
    return rows


def diff_sets(base_rows, cand_rows, field: str) -> dict:
    b = {r["question_id"] for r in base_rows if r[field]}
    c = {r["question_id"] for r in cand_rows if r[field]}
    return {
        "base": len(b), "cand": len(c),
        "gain": sorted(c - b), "lose": sorted(b - c), "net": len(c) - len(b),
        "gain_by_category": dict(collections.Counter(
            r["category"] for r in cand_rows if r["question_id"] in (c - b))),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P3 第一轮：引用粒度真跑验收")
    ap.add_argument("--generation", default=DEFAULT_GEN)
    ap.add_argument("--gold-set", default=str(DEFAULT_GOLD))
    ap.add_argument("--out-json", default=str(ROOT / "data" / "a_p3_line_citation.json"))
    args = ap.parse_args(argv)

    gold_path = Path(args.gold_set)
    if not gold_path.is_absolute():
        gold_path = ROOT / gold_path
    if not gold_path.exists():
        print(f"[2] 输入不可用：{args.gold_set}", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    loaded = pipeline.load_generation(args.generation)
    chunks = loaded["chunks"]
    by_chunk: dict[str, list[dict]] = collections.defaultdict(list)
    for rec in loaded["assertions"]:
        by_chunk[str(rec.get("chunk_id") or "")].append(rec)
    pipe = pipeline.build_pipeline(loaded, top_k=20, graph_depth=1)

    items = load_dev(gold_path)
    answerable = [it for it in items if not it.get("should_abstain")
                  and (it.get("gold_chunk_ids") or [])]
    print(f"[load] dev={len(items)} 应答题={len(answerable)} 块={len(chunks)}")

    t1 = time.perf_counter()
    base = run_config(pipe, items, chunks, by_chunk, whole_line=False)
    print(f"[run] baseline {time.perf_counter()-t1:.1f}s")
    t1 = time.perf_counter()
    cand = run_config(pipe, items, chunks, by_chunk, whole_line=True)
    print(f"[run] whole-line {time.perf_counter()-t1:.1f}s")

    # ---- 单变量自证（三条，任一不过即拒绝出结论） ----
    checks = []
    same_answer = [r["question_id"] for b, r in zip(base, cand)
                   if b["answer_sha"] != r["answer_sha"]]
    checks.append(("answer 正文逐题相同", not same_answer, same_answer[:5]))
    stat_fields = ("dropped_unsupported", "dropped_no_citation",
                   "dropped_ungrounded", "accepted", "candidate_claims")
    stat_bad = {f: [b["question_id"] for b, r in zip(base, cand) if b[f] != r[f]]
                for f in stat_fields}
    checks.append(("stats 判据计数逐题相同", not any(stat_bad.values()),
                   {k: v[:5] for k, v in stat_bad.items() if v}))
    ids_bad = [b["question_id"] for b, r in zip(base, cand) if b["claim_ids"] != r["claim_ids"]]
    checks.append(("claim 的 assertion_id 集合逐题相同", not ids_bad, ids_bad[:5]))

    print("\n[单变量自证]")
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  {detail}"))

    # ---- 四个必须报的数 ----
    def cov_block(rows):
        ans_rows = [r for r in rows if r["question_id"] in {a["question_id"] for a in answerable}]
        n_all = len(ans_rows)
        n_span = sum(1 for r in ans_rows if r["span_reachable"])
        n_gate = sum(1 for r in ans_rows if r["span_reachable"] and r["required_in_context"])
        cov = sum(1 for r in ans_rows if r["covered"])
        return {
            "coverage_ok": cov,
            "den_all_answerable": n_all,
            "den_span_reachable": n_span,
            "den_two_gates": n_gate,
            "r_all": round(cov / n_all, 6) if n_all else 0.0,
            "r_span": round(cov / n_span, 6) if n_span else 0.0,
            "r_gates": round(cov / n_gate, 6) if n_gate else 0.0,
        }

    cov_base, cov_cand = cov_block(base), cov_block(cand)
    cite_base = {
        "total": sum(r["citations"] for r in base),
        "distinct": sum(r["citations_distinct"] for r in base),
        "ok": sum(r["cite_ok"] for r in base),
        "bad": sum(r["cite_bad"] for r in base),
    }
    cite_cand = {
        "total": sum(r["citations"] for r in cand),
        "distinct": sum(r["citations_distinct"] for r in cand),
        "ok": sum(r["cite_ok"] for r in cand),
        "bad": sum(r["cite_bad"] for r in cand),
    }
    cite_base["accuracy"] = round(cite_base["ok"] / cite_base["total"], 6) if cite_base["total"] else 0.0
    cite_cand["accuracy"] = round(cite_cand["ok"] / cite_cand["total"], 6) if cite_cand["total"] else 0.0

    cov_diff = diff_sets(base, cand, "covered")

    # 塌缩：同行多条断言被压成同一串 evidence
    collapse = {
        "base_total_minus_distinct": cite_base["total"] - cite_base["distinct"],
        "cand_total_minus_distinct": cite_cand["total"] - cite_cand["distinct"],
        "per_q_more_collapse": sorted(
            r["question_id"] for b, r in zip(base, cand)
            if (r["citations"] - r["citations_distinct"])
            > (b["citations"] - b["citations_distinct"])),
    }

    # 验收样本逐条（Lead 指定的 6 条 alias）
    samples = ["q-al-000012", "q-al-000014", "q-al-000022",
               "q-al-000033", "q-al-000037", "q-al-000038"]
    bmap = {r["question_id"]: r for r in base}
    cmap = {r["question_id"]: r for r in cand}
    sample_rows = []
    for qid in samples:
        b, c = bmap.get(qid), cmap.get(qid)
        if not b or not c:
            sample_rows.append({"question_id": qid, "error": "不在 dev 集"})
            continue
        sample_rows.append({
            "question_id": qid,
            "category": b["category"],
            "n_required": b["n_required"],
            "before": {k: b[k] for k in ("covered", "citations", "citations_distinct",
                                         "cite_ok", "cite_bad", "abstained", "answer")},
            "after": {k: c[k] for k in ("covered", "citations", "citations_distinct",
                                        "cite_ok", "cite_bad", "abstained", "answer")},
        })

    lat = sorted(r["elapsed_ms"] for r in cand)
    doc = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_rev(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "command": "py -3 scripts/a_p3_line_citation.py",
        "generation": args.generation,
        "gold_set": rel(gold_path),
        "split": "dev",
        "split_policy": "pre-split-diagnostic/dev-only：dev 结论不外推到 new-holdout 160",
        "variable": {"name": "CITATION_WHOLE_LINE", "base": False, "cand": True},
        "counts": {"dev_questions": len(items), "answerable": len(answerable)},
        "single_variable_selfcheck": [
            {"check": n, "pass": ok, "detail": d} for n, ok, d in checks
        ],
        "coverage": {"base": cov_base, "cand": cov_cand, "diff": cov_diff},
        "citations": {"base": cite_base, "cand": cite_cand, "collapse": collapse},
        "acceptance_samples": sample_rows,
        "latency_ms": {
            "p50": lat[len(lat) // 2] if lat else 0.0,
            "p90": lat[min(len(lat) - 1, int(len(lat) * 0.9))] if lat else 0.0,
            "max": lat[-1] if lat else 0.0,
        },
        "per_question": {"base": base, "cand": cand},
        "wall_seconds": round(time.perf_counter() - t0, 2),
    }

    out = Path(args.out_json)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[覆盖率] 三分母并列")
    for tag, blk in (("base", cov_base), ("cand", cov_cand)):
        print(f"  {tag}: {blk['coverage_ok']}  "
              f"/{blk['den_all_answerable']}={blk['r_all']}  "
              f"/{blk['den_span_reachable']}={blk['r_span']}  "
              f"/{blk['den_two_gates']}={blk['r_gates']}")
    print(f"  救回 {len(cov_diff['gain'])} 丢失 {len(cov_diff['lose'])} 净 {cov_diff['net']}")
    print(f"  救回按题型 {cov_diff['gain_by_category']}")
    if cov_diff["lose"]:
        print(f"  丢失题号 {cov_diff['lose']}")

    print("\n[引用] 条数 / 逐条定位 / 硬失败")
    for tag, blk in (("base", cite_base), ("cand", cite_cand)):
        print(f"  {tag}: 条数={blk['total']} 去重后={blk['distinct']} "
              f"通过={blk['ok']} 失败={blk['bad']} 通过率={blk['accuracy']}")
    print(f"  塌缩（条数-去重后）: base {collapse['base_total_minus_distinct']} "
          f"→ cand {collapse['cand_total_minus_distinct']}")
    print(f"  输出: {rel(out)}")

    failed = [c for c in checks if not c[1]]
    if failed:
        print("\n[4] 单变量自证失败——这不是单变量实验，拒绝出结论", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
