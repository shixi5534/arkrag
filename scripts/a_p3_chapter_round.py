"""P3 第三轮·靶子一 cross_chunk 章节包：真跑验收 + 反向实测 + 回滚等价比对。

**这是生产通路真跑**，不是离线回放：直接调 `pipeline.ask_once`，唯一差别是模块级开关
`answer.CHAPTER_ANSWER_PATH` 的取值。同一份代码、只差这一个布尔量。

## 与 `a_title_path_round.py` 的关系

结构照抄（同一套判据函数与同口径），但**变量不是同一个旋钮**：那个脚本验
`TITLE_ANSWER_PATH`（single_chunk 的标题形态），本脚本验 `CHAPTER_ANSWER_PATH`
（cross_chunk 的章节代号形态）。混在一个脚本里会让「本轮改了什么」说不清。

## 三个 stage

- `--stage real`：基线（False）vs 候选（True）各跑全量 dev 380，出**三道闸**三行数
  （答案对 / 引用覆盖 / 正确覆盖），并在每一行后写明**下一道闸兑现了几题**。
- `--stage reverse`：R8'-a 换实体（换成不存在的名字，必须落回原路径且不猜章节）、
  R8'-b 触发词纯度（「章节代号」在 v2 640 题里只出现在 cross_chunk）、
  R8'-c 章节来源（答案里每个章节都必须来自某个上下文单元的 `chapter` 字段）。
- `--stage rollback`：四轮子进程比对观测向量总体 SHA-256（HEAD 版 / 新源码 False /
  新源码 True ×2），子进程必须回显开关实际取值。

## 五条硬规矩

1. `load_dev()` 不做 `[:n]` 挑选——dev 顺序按类聚簇，截前缀只覆盖一类（`ask.py --limit`
   历史事故）。抽样必须跨类并打印「本次数字不可作结论」。
2. 判据函数与 `scripts/ask.py:262-302` 逐字同口径，不自己发明。
3. 子进程一律 `env=subproc.child_env()`（Gate 0 事故的规矩，AST 门会扫这条）。
4. **章节名只读 `unit["chapter"]`**，不许截 `chunk_id` 路径（main 线路径第二段只有
   `main`、真值 `main_ep14`；截路径 7/38、读字段 38/38）。
5. **churn 判据不适用要写明理由**，不能空白：本通路不经 `search`，top-20 集合变动 = 0。

退出码：0 跑通 / 2 输入不可用 / 4 自检失败（自检失败时拒绝出结论）。
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import answer as answer_mod, config, pipeline, subproc  # noqa: E402

DEFAULT_GEN = "g-20261103T132152Z-06a741"
DEFAULT_GOLD = ROOT / "data" / "goldset" / "gold-v2.json"

# 观测向量：回滚等价性比对用。任何一个变了都说明「开关关掉后不是原样」。
OBSERVE_FIELDS = (
    "covered", "correct_coverage", "answer_hits_gold", "abstained",
    "citations", "citations_distinct", "cite_bad", "units", "answer_sha",
    "chapter_path_applied",
)


def rel(path: Path) -> str:
    return config.rel_to_root(path)


def git_rev() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
            capture_output=True, text=True, timeout=20, check=False,
            env=subproc.child_env(),
        )
        return (out.stdout or "").strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def load_dev(gold_path: Path, limit: int | None = None) -> list[dict]:
    """dev 全集：split==dev，**保留应拒题**（R3' 要证明没把该拒的翻成作答）。"""
    data = json.loads(gold_path.read_text(encoding="utf-8"))
    rows = [it for it in data["items"] if it.get("split") == "dev"]
    rows.sort(key=lambda it: it["question_id"])
    if limit:
        by_cat: dict[str, list[dict]] = {}
        for it in rows:
            by_cat.setdefault(str(it.get("category") or ""), []).append(it)
        picked: list[dict] = []
        while len(picked) < limit and any(by_cat.values()):
            for cat in sorted(by_cat):
                if by_cat[cat] and len(picked) < limit:
                    picked.append(by_cat[cat].pop(0))
        print(f"[warn] --limit {limit}：跨类抽样，本次数字**不可作结论**")
        return picked
    return rows


def answerable_of(items: list[dict]) -> list[dict]:
    return [it for it in items
            if not it.get("should_abstain") and (it.get("gold_chunk_ids") or [])]


# ---------------------------------------------------------------- 判据（与 ask.py 同口径）


def required_chunks(item: dict) -> list[str]:
    return [str(rc.get("chunk_id") or "") for rc in item.get("required_citations") or []]


def span_reachable(item: dict, by_chunk: dict[str, list[dict]]) -> bool:
    """每条 required 的 evidence 是否是某条断言 `evidence_text` 的子串。"""
    for rc in item.get("required_citations") or []:
        cid = str(rc.get("chunk_id") or "")
        req = str(rc.get("evidence") or "")
        if not any(req in str(rec.get("evidence_text") or "")
                   for rec in by_chunk.get(cid) or []):
            return False
    return True


def coverage_of(item: dict, citations: list[dict]) -> bool:
    """引用覆盖：required 全部命中。与 `scripts/ask.py:283-298` 逐字同口径。"""
    reqs = item.get("required_citations") or []
    if not reqs:
        return False
    for req in reqs:
        cid = str(req.get("chunk_id") or "")
        ev = str(req.get("evidence") or "")
        if not any(str(c.get("chunk_id") or "") == cid
                   and ev in str(c.get("evidence") or "") for c in citations or []):
            return False
    return True


def answer_hits_gold(item: dict, answer: str) -> bool:
    """答案质量：gold 答案或任一 acceptable 出现在答案里。"""
    text = str(answer or "")
    if not text:
        return False
    cands = [str(item.get("answer") or "")]
    cands += [str(x) for x in (item.get("acceptable_answers") or [])]
    return any(c and c in text for c in cands)


def citation_scan(units: list[dict], citations: list[dict]) -> dict:
    """`text[start:end] == evidence` 逐条比对。与 `scripts/ask.py:262-270` 同口径。"""
    by_id = {str(u.get("chunk_id") or ""): u for u in units or []}
    total = failed = 0
    for cite in citations or []:
        total += 1
        unit = by_id.get(str(cite.get("chunk_id") or ""))
        if unit is None:
            failed += 1
            continue
        text = str(unit.get("text") or "")
        start, end = int(cite.get("start", -1)), int(cite.get("end", -1))
        if start < 0 or end < start or text[start:end] != str(cite.get("evidence") or ""):
            failed += 1
    return {"total": total, "failed": failed}


# ---------------------------------------------------------------- 跑一题 / 跑一轮


def run_one(pipe: dict, item: dict, question: str | None = None) -> dict:
    q = str(item.get("question") or "") if question is None else str(question)
    t0 = time.perf_counter()
    out = pipeline.ask_once(pipe, q)
    ms = (time.perf_counter() - t0) * 1000.0

    ctx = out.get("context") or {}
    ans = out.get("answer") or {}
    units = ctx.get("units") or []
    cites = ans.get("citations") or []
    st = ans.get("stats") or {}
    cp = st.get("chapter_path") or {}

    required = item.get("required_citations") or []
    ctx_ids = {str(u.get("chunk_id") or "") for u in units}
    scan = citation_scan(units, cites)
    covered = (not item.get("should_abstain")) and bool(required) and coverage_of(item, cites)
    hits_gold = answer_hits_gold(item, str(ans.get("answer") or ""))

    line_cites = [c for c in cites if str(c.get("granularity") or "") == "line"]
    span_cites = [c for c in cites if str(c.get("granularity") or "") != "line"]

    answer_text = str(ans.get("answer") or "")
    # 答案里的章节必须真的来自某个上下文单元的 chapter 字段（R8'-c）
    ctx_chapters = {str(u.get("chapter") or "").strip() for u in units}
    ctx_chapters.discard("")
    if bool(cp.get("applied")):
        answer_chapters = [c for c in answer_text.split("、") if c]
        ch_from_unit = all(c in ctx_chapters for c in answer_chapters)
    else:
        answer_chapters = []
        ch_from_unit = True  # 未启用本通路时不适用

    return {
        "question_id": item["question_id"],
        "category": str(item.get("category") or ""),
        "origin": str(item.get("origin") or ""),
        "should_abstain": bool(item.get("should_abstain")),
        "has_gold": bool(item.get("gold_chunk_ids") or []),
        "n_required": len(required),
        "required_in_context": bool(required) and all(c in ctx_ids for c in required_chunks(item)),
        "covered": covered,
        "answer_hits_gold": hits_gold,
        "correct_coverage": covered and hits_gold,
        "abstained": bool(ans.get("abstained")),
        "abstain_reason": str(ans.get("abstain_reason") or ""),
        "answer": answer_text,
        "answer_sha": hashlib.sha256(answer_text.encode("utf-8")).hexdigest()[:16],
        "citations": len(cites),
        "citations_distinct": len({(str(c.get("chunk_id") or ""), str(c.get("evidence") or ""))
                                   for c in cites}),
        "citations_line": len(line_cites),
        "citations_span": len(span_cites),
        "cite_ok": scan["total"] - scan["failed"],
        "cite_bad": scan["failed"],
        "chapter_path_applied": bool(cp.get("applied")),
        "chapter_path_source": str(cp.get("chapter_source") or ""),
        "chapter_path_n": int(cp.get("n_chapters") or 0),
        "chapter_answer_chapters": answer_chapters,
        "chapter_from_unit": ch_from_unit,
        "dropped_unsupported": int(st.get("dropped_unsupported") or 0),
        "dropped_no_citation": int(st.get("dropped_no_citation") or 0),
        "dropped_ungrounded": int(st.get("dropped_ungrounded") or 0),
        "accepted": int(st.get("accepted") or 0),
        "candidate_claims": int(st.get("candidates") or 0),
        "units": len(units),
        "unit_sequence": [str(u.get("chunk_id") or "") for u in units],
        "elapsed_ms": round(ms, 3),
    }


def run_config(pipe: dict, items: list[dict], enabled: bool) -> list[dict]:
    answer_mod.CHAPTER_ANSWER_PATH = bool(enabled)
    try:
        return [run_one(pipe, it) for it in items]
    finally:
        answer_mod.CHAPTER_ANSWER_PATH = False


def build_all():
    loaded = pipeline.load_generation(DEFAULT_GEN)
    chunks = loaded["chunks"]
    by_chunk: dict[str, list[dict]] = collections.defaultdict(list)
    for rec in loaded["assertions"]:
        by_chunk[str(rec.get("chunk_id") or "")].append(rec)
    pipe = pipeline.build_pipeline(loaded, top_k=20, graph_depth=1)
    return loaded, chunks, by_chunk, pipe


# ---------------------------------------------------------------- 统计


def cov_block(rows: list[dict], by_chunk: dict[str, list[dict]],
              items_by_id: dict[str, dict]) -> dict:
    ans_rows = [r for r in rows if not r["should_abstain"] and r["has_gold"]]
    n_all = len(ans_rows)
    n_span = sum(1 for r in ans_rows
                 if span_reachable(items_by_id[r["question_id"]], by_chunk))
    n_gate = sum(1 for r in ans_rows
                 if span_reachable(items_by_id[r["question_id"]], by_chunk)
                 and r["required_in_context"])
    cov = sum(1 for r in ans_rows if r["covered"])
    cor = sum(1 for r in ans_rows if r["correct_coverage"])
    return {
        "coverage_ok": cov, "correct_coverage": cor,
        "den_all_answerable": n_all, "den_span_reachable": n_span, "den_two_gates": n_gate,
        "r_all": round(cov / n_all, 6) if n_all else 0.0,
        "r_span": round(cov / n_span, 6) if n_span else 0.0,
        "r_gates": round(cov / n_gate, 6) if n_gate else 0.0,
        "c_all": round(cor / n_all, 6) if n_all else 0.0,
        "c_span": round(cor / n_span, 6) if n_span else 0.0,
        "c_gates": round(cor / n_gate, 6) if n_gate else 0.0,
    }


def absten_block(rows: list[dict]) -> dict:
    should = [r for r in rows if r["should_abstain"]]
    answerable = [r for r in rows if not r["should_abstain"]]
    return {
        "n_should_abstain": len(should), "n_answerable": len(answerable),
        "abstained_should": sum(1 for r in should if r["abstained"]),
        "abstained_answerable": sum(1 for r in answerable if r["abstained"]),
        "abstained_total": sum(1 for r in rows if r["abstained"]),
        "abstain_accuracy": round(
            (sum(1 for r in should if r["abstained"])
             + sum(1 for r in answerable if not r["abstained"])) / len(rows), 6) if rows else 0.0,
        "should_abstain_but_answered": sorted(
            r["question_id"] for r in should if not r["abstained"]),
        "answerable_but_abstained": sorted(
            r["question_id"] for r in answerable if r["abstained"]),
    }


def cite_block(rows: list[dict]) -> dict:
    total = sum(r["citations"] for r in rows)
    distinct = sum(r["citations_distinct"] for r in rows)
    ok = sum(r["cite_ok"] for r in rows)
    bad = sum(r["cite_bad"] for r in rows)
    return {
        "total": total, "distinct": distinct,
        "line": sum(r["citations_line"] for r in rows),
        "span": sum(r["citations_span"] for r in rows),
        "ok": ok, "bad": bad,
        "accuracy": round(ok / total, 6) if total else 0.0,
        "collapse_total_minus_distinct": total - distinct,
    }


def diff_set(base, cand, field) -> dict:
    b = {r["question_id"] for r in base if r[field]}
    c = {r["question_id"] for r in cand if r[field]}
    cat_of = {r["question_id"]: r["category"] for r in cand}
    return {
        "base": len(b), "cand": len(c),
        "gain": sorted(c - b), "lose": sorted(b - c), "net": len(c) - len(b),
        "gain_by_category": dict(collections.Counter(cat_of[q] for q in c - b)),
        "lose_by_category": dict(collections.Counter(cat_of[q] for q in b - c)),
    }


def per_category(base, cand, field) -> dict:
    out: dict[str, dict] = {}
    bmap = {r["question_id"]: r for r in base}
    for r in cand:
        slot = out.setdefault(r["category"],
                             {"n": 0, "base": 0, "cand": 0, "gain": [], "lose": []})
        slot["n"] += 1
        b = bmap[r["question_id"]]
        slot["base"] += 1 if b[field] else 0
        slot["cand"] += 1 if r[field] else 0
        if r[field] and not b[field]:
            slot["gain"].append(r["question_id"])
        if b[field] and not r[field]:
            slot["lose"].append(r["question_id"])
    for slot in out.values():
        slot["delta"] = slot["cand"] - slot["base"]
    return dict(sorted(out.items()))


# ---------------------------------------------------------------- stage: real


def stage_real(gold_path: Path, out_json: Path, items: list[dict]) -> int:
    loaded, chunks, by_chunk, pipe = build_all()
    items_by_id = {it["question_id"]: it for it in items}
    answerable = answerable_of(items)
    print(f"[load] dev={len(items)} 应答题={len(answerable)} "
          f"应拒={len(items)-len(answerable)} 块={len(chunks)}")

    t = time.perf_counter()
    base = run_config(pipe, items, enabled=False)
    print(f"[run] 基线（CHAPTER_ANSWER_PATH=False）{time.perf_counter()-t:.1f}s")
    t = time.perf_counter()
    cand = run_config(pipe, items, enabled=True)
    print(f"[run] 候选（CHAPTER_ANSWER_PATH=True）{time.perf_counter()-t:.1f}s")

    bmap = {r["question_id"]: r for r in base}
    cmap = {r["question_id"]: r for r in cand}

    # 触发集合（设计上应等于 cross_chunk 里「在上下文能定位到焦点实体」的题）
    trig = sorted(q for q, r in cmap.items() if r["chapter_path_applied"])

    checks: list[dict] = []
    # 自检 1：开关 False 时不该有任何题触发
    leaked = sorted(q for q, r in bmap.items() if r["chapter_path_applied"])
    checks.append({"check": "开关 False 时无题触发章节通路", "pass": not leaked,
                   "detail": leaked[:5]})

    # 自检 2：触发的题必须全为 cross_chunk
    wrong_cat = sorted(q for q in trig if cmap[q]["category"] != "cross_chunk")
    checks.append({"check": "触发题全为 cross_chunk", "pass": not wrong_cat,
                   "detail": wrong_cat[:5]})

    # 自检 3：非触发题必须与基线逐字相同（答案 sha + 三闸全等）
    changed = sorted(
        q for q in cmap
        if q not in set(trig) and (
            cmap[q]["answer_sha"] != bmap[q]["answer_sha"]
            or cmap[q]["covered"] != bmap[q]["covered"]
            or cmap[q]["correct_coverage"] != bmap[q]["correct_coverage"]
            or cmap[q]["abstained"] != bmap[q]["abstained"]))
    checks.append({"check": "非触发题行为与基线逐题一致", "pass": not changed,
                   "detail": changed[:5]})

    # 自检 4：R8'-c 章节来源——答案里每个章节都必须来自某个上下文单元的 chapter
    bad_src = sorted(q for q in trig if not cmap[q]["chapter_from_unit"])
    checks.append({"check": "答案章节全部来自 unit.chapter（R8'-c）", "pass": not bad_src,
                   "detail": bad_src[:5]})

    # 自检 5：引用硬失败必须为 0
    bad_cite = sorted(q for q in trig if cmap[q]["cite_bad"] > 0)
    checks.append({"check": "触发题引用定位零失败", "pass": not bad_cite,
                   "detail": bad_cite[:5]})

    cov_base = cov_block(base, by_chunk, items_by_id)
    cov_cand = cov_block(cand, by_chunk, items_by_id)
    cite_base, cite_cand = cite_block(base), cite_block(cand)
    abs_base, abs_cand = absten_block(base), absten_block(cand)
    ans_diff = diff_set(base, cand, "answer_hits_gold")
    cov_diff = diff_set(base, cand, "covered")
    cor_diff = diff_set(base, cand, "correct_coverage")

    # ---- 三道闸：每行后写明下一道闸兑现几题（Lead m04575 硬约束一）
    def next_gate(gain: list[str], nxt: dict) -> int:
        return sum(1 for q in gain if q in set(nxt["gain"]))

    gates = {
        "gate1_answer": {
            "label": "第一道闸：答案含 gold（answer_hits_gold）",
            "base": ans_diff["base"], "cand": ans_diff["cand"],
            "gain": ans_diff["gain"], "lose": ans_diff["lose"], "net": ans_diff["net"],
            "next_gate_fulfilled": next_gate(ans_diff["gain"], cov_diff),
            "next_gate_label": "其中同时把引用覆盖做真的题数",
        },
        "gate2_coverage": {
            "label": "第二道闸：引用覆盖（covered）",
            "base": cov_diff["base"], "cand": cov_diff["cand"],
            "gain": cov_diff["gain"], "lose": cov_diff["lose"], "net": cov_diff["net"],
            "next_gate_fulfilled": next_gate(cov_diff["gain"], cor_diff),
            "next_gate_label": "其中同时答案也对（→ 正确覆盖）的题数",
        },
        "gate3_correct": {
            "label": "第三道闸：正确覆盖（covered ∧ 答案含 gold）",
            "base": cor_diff["base"], "cand": cor_diff["cand"],
            "gain": cor_diff["gain"], "lose": cor_diff["lose"], "net": cor_diff["net"],
            "next_gate_fulfilled": None,
            "next_gate_label": "终点",
        },
    }

    # 转作答清单
    turned = sorted(q for q, r in cmap.items()
                    if r["should_abstain"] is False and bmap[q]["abstained"] and not r["abstained"])
    should_which = sorted(q for q in turned if bmap[q]["should_abstain"])

    lat = sorted(r["elapsed_ms"] for r in cand)
    lat_b = sorted(r["elapsed_ms"] for r in base)

    doc = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_rev(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "command": "py -3 scripts/a_p3_chapter_round.py --stage real",
        "generation": DEFAULT_GEN,
        "gold_set": rel(gold_path),
        "split": "dev",
        "split_policy": "pre-split-diagnostic/dev-only：dev 结论不外推到 new-holdout 160",
        "variable": {"name": "CHAPTER_ANSWER_PATH", "base": False, "cand": True},
        "counts": {"dev_questions": len(items), "answerable": len(answerable),
                   "should_abstain": len(items) - len(answerable)},
        "churn": {
            "applicable": False,
            "reason": "本通路不经 search、不重排 hits：top-20 集合变动 = 0，"
                      "gold 名次不变；洗牌判据（真跑Δ ÷ top-20 变动集合数 < 0.1）不适用。",
        },
        "selfcheck": checks,
        "chapter_path": {"triggered": len(trig), "triggered_ids": trig},
        "gates": gates,
        "coverage": {
            "base": cov_base, "cand": cov_cand,
            "answer_diff": ans_diff, "covered_diff": cov_diff, "correct_diff": cor_diff,
            "correct_by_category": per_category(base, cand, "correct_coverage"),
            "answer_by_category": per_category(base, cand, "answer_hits_gold"),
            "covered_by_category": per_category(base, cand, "covered"),
        },
        "citations": {
            "base": cite_base, "cand": cite_cand,
            "granularity_note": "line = 章节通路自建的整行引用；span = 断言 span 引用",
        },
        "abstain": {"base": abs_base, "cand": abs_cand,
                    "turned_to_answer": turned,
                    "turned_but_should_abstain": should_which},
        "latency_ms": {
            "base": {"p50": lat_b[len(lat_b) // 2] if lat_b else 0.0,
                     "p90": lat_b[min(len(lat_b) - 1, int(len(lat_b) * 0.9))] if lat_b else 0.0,
                     "max": lat_b[-1] if lat_b else 0.0},
            "cand": {"p50": lat[len(lat) // 2] if lat else 0.0,
                     "p90": lat[min(len(lat) - 1, int(len(lat) * 0.9))] if lat else 0.0,
                     "max": lat[-1] if lat else 0.0},
        },
        "per_question": {"base": base, "cand": cand},
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[自检]")
    for c in checks:
        print(f"  {'PASS' if c['pass'] else 'FAIL'}  {c['check']}"
              + ("" if c["pass"] else f"  {c['detail']}"))

    print("\n[三道闸] 每行后写明下一道闸兑现几题")
    for key in ("gate1_answer", "gate2_coverage", "gate3_correct"):
        g = gates[key]
        extra = ("" if g["next_gate_fulfilled"] is None
                 else f"  → 下一道闸兑现 {g['next_gate_fulfilled']} 题（{g['next_gate_label']}）")
        print(f"  {g['label']}: {g['base']} → {g['cand']}  "
              f"救 {len(g['gain'])} 丢 {len(g['lose'])} 净 {g['net']}{extra}")
        if g["lose"]:
            print(f"    丢失题号: {g['lose']}")

    print("\n[覆盖率 / 正确覆盖] 三分母并列（分母为应答题）")
    for tag, blk in (("base", cov_base), ("cand", cov_cand)):
        print(f"  {tag}: 覆盖 {blk['coverage_ok']}  正确覆盖 {blk['correct_coverage']}  "
              f"/{blk['den_all_answerable']}={blk['c_all']}  "
              f"/{blk['den_span_reachable']}={blk['c_span']}  "
              f"/{blk['den_two_gates']}={blk['c_gates']}")

    print("\n[引用] 条数 / 逐条定位 / 两种粒度分列")
    for tag, blk in (("base", cite_base), ("cand", cite_cand)):
        print(f"  {tag}: 条数={blk['total']} 去重={blk['distinct']} "
              f"整行={blk['line']} span={blk['span']} "
              f"通过={blk['ok']} 失败={blk['bad']} 通过率={blk['accuracy']}")

    print("\n[拒答]")
    for tag, blk in (("base", abs_base), ("cand", abs_cand)):
        print(f"  {tag}: 应拒题中拒答 {blk['abstained_should']}/{blk['n_should_abstain']}  "
              f"应答题中拒答 {blk['abstained_answerable']}/{blk['n_answerable']}  "
              f"拒答正确率 {blk['abstain_accuracy']}")
    print(f"  拒答转作答 {len(turned)} 题，其中「本该拒答」{len(should_which)} 题 {should_which}")
    print(f"  输出: {rel(out_json)}")

    if any(not c["pass"] for c in checks):
        print("\n[4] 自检失败——拒绝出结论", file=sys.stderr)
        return 4
    if should_which:
        print("\n[4] 有本该拒答的题被转成作答（R3' 触发条件）", file=sys.stderr)
        return 4
    return 0


# ---------------------------------------------------------------- stage: reverse


def stage_reverse(gold_path: Path, out_json: Path, items: list[dict]) -> int:
    """R8'-a 换实体 / R8'-b 触发词纯度 / R8'-c 章节来源。零容忍，任一失败 exit 4。"""
    loaded, chunks, by_chunk, pipe = build_all()
    answer_mod.CHAPTER_ANSWER_PATH = True
    try:
        # --- R8'-b：触发词纯度（全库集合等式，不只 dev）
        full = json.loads(gold_path.read_text(encoding="utf-8"))["items"]
        trigger = answer_mod.CHAPTER_TRIGGERS[0]
        with_trigger = {it["question_id"]: str(it.get("category") or "")
                        for it in full if trigger in str(it.get("question") or "")}
        non_cx = sorted(q for q, cat in with_trigger.items() if cat != "cross_chunk")
        purity_ok = not non_cx
        print(f"[R8'-b] 「{trigger}」出现在 {len(with_trigger)} 题，"
              f"其中非 cross_chunk {len(non_cx)} 题 → {'PASS' if purity_ok else 'FAIL'}")
        if non_cx:
            print(f"  混入: {non_cx[:10]}")

        # --- 触发题清单（dev 里本通路真正生效的）
        base = run_config(pipe, items, enabled=False)
        answer_mod.CHAPTER_ANSWER_PATH = True
        cand = [run_one(pipe, it) for it in items]
        bmap = {r["question_id"]: r for r in base}
        trig = [r for r in cand if r["chapter_path_applied"]]
        print(f"[R8'] dev 触发 {len(trig)} 题")

        # --- R8'-a：把问句里的焦点实体换成不存在的名字，必须落回原路径且不猜章节
        import re as _re
        rows_a = []
        for r in trig:
            it = next(x for x in items if x["question_id"] == r["question_id"])
            q = str(it["question"])
            # 把「」引文整体替换成不存在的名字（cross_chunk 问句里实体常以「」形式出现）
            mutated = _re.sub(r"「[^」]+」", "「完全不存在的东西甲乙丙」", q)
            if mutated == q:
                mutated = q + "（其中涉及「完全不存在的东西甲乙丙」）"
            out = pipeline.ask_once(pipe, mutated)
            ans = out.get("answer") or {}
            st = ans.get("stats") or {}
            applied = bool((st.get("chapter_path") or {}).get("applied"))
            rows_a.append({
                "question_id": it["question_id"],
                "mutated_question": mutated,
                "chapter_path_applied": applied,
                "answer": str(ans.get("answer") or ""),
            })
        bad_a = [r for r in rows_a if r["chapter_path_applied"]]
        a_ok = not bad_a
        print(f"[R8'-a] 换实体后仍走章节通路 {len(bad_a)}/{len(rows_a)} 题 → "
              f"{'PASS' if a_ok else 'FAIL'}")
        for r in bad_a[:5]:
            print(f"  {r['question_id']}: {r['answer'][:60]}")

        # --- R8'-c：章节来源（只测真实问题，不测变异问句）
        bad_c = sorted(r["question_id"] for r in trig if not r["chapter_from_unit"])
        c_ok = not bad_c
        print(f"[R8'-c] 答案章节非来自 unit.chapter {len(bad_c)} 题 → "
              f"{'PASS' if c_ok else 'FAIL'}")
        if bad_c:
            print(f"  {bad_c[:10]}")

        # --- 引用定位（触发题零失败）
        bad_cite = sorted(r["question_id"] for r in trig if r["cite_bad"] > 0)
        d_ok = not bad_cite
        print(f"[引用] 触发题定位失败 {len(bad_cite)} 题 → {'PASS' if d_ok else 'FAIL'}")

        doc = {
            "generated_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": git_rev(),
            "python": platform.python_version(),
            "os": platform.platform(),
            "command": "py -3 scripts/a_p3_chapter_round.py --stage reverse",
            "generation": DEFAULT_GEN,
            "gold_set": rel(gold_path),
            "split_policy": "pre-split-diagnostic/dev-only：dev 结论不外推到 new-holdout 160",
            "r8b_trigger_purity": {
                "trigger": trigger, "n_with_trigger": len(with_trigger),
                "n_non_cross_chunk": len(non_cx), "intruders": non_cx[:20],
                "pass": purity_ok,
            },
            "r8a_swap_entity": {"n": len(rows_a), "n_still_applied": len(bad_a),
                                "rows": rows_a, "pass": a_ok},
            "r8c_chapter_source": {"n_bad": len(bad_c), "bad": bad_c, "pass": c_ok},
            "citation_locate": {"n_bad": len(bad_cite), "bad": bad_cite, "pass": d_ok},
            "all_pass": bool(a_ok and c_ok and d_ok and purity_ok),
            "n_triggered": len(trig),
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  输出: {rel(out_json)}")
        return 0 if doc["all_pass"] else 4
    finally:
        answer_mod.CHAPTER_ANSWER_PATH = False


# ---------------------------------------------------------------- stage: rollback


def digest_of(rows: list[dict]) -> str:
    blob = json.dumps(
        [{k: r[k] for k in ("question_id",) + OBSERVE_FIELDS} for r in rows],
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _one_run(mode: str, out_path: Path) -> int:
    """子进程内部：跑一轮全量 dev，打印摘要与**实际生效的开关值**（必须回显）。"""
    items = load_dev(DEFAULT_GOLD)
    _, _, _, pipe = build_all()
    if mode == "false":
        answer_mod.CHAPTER_ANSWER_PATH = False
    elif mode == "true":
        answer_mod.CHAPTER_ANSWER_PATH = True
    elif mode != "orig":
        print(f"[2] 未知 mode {mode}", file=sys.stderr)
        return 2
    effective = bool(getattr(answer_mod, "CHAPTER_ANSWER_PATH", False))
    rows = [run_one(pipe, it) for it in items]
    out_path.write_text(
        json.dumps({"mode": mode, "effective_chapter_answer_path": effective,
                    "digest": digest_of(rows), "n": len(rows),
                    "aggregate": {
                        "covered": sum(1 for r in rows if r["covered"]),
                        "correct_coverage": sum(1 for r in rows if r["correct_coverage"]),
                        "answer_hits_gold": sum(1 for r in rows if r["answer_hits_gold"]),
                        "abstained": sum(1 for r in rows if r["abstained"]),
                        "citations": sum(r["citations"] for r in rows),
                    }},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"CHAPTER_ANSWER_PATH={effective} DIGEST={digest_of(rows)}")
    return 0


def stage_rollback(out_json: Path) -> int:
    """四轮：HEAD 版 answer.py（无该开关）/ 新源码 False / 新源码 True ×2。

    原版用旁路 src 副本 + PYTHONPATH 切导入来源——**不许 import 被验的改动**。
    """
    tmp = Path(tempfile.mkdtemp(prefix="a_chapterrollback_"))
    runs: list[dict] = []
    try:
        orig_src = tmp / "src"
        shutil.copytree(ROOT / "src", orig_src)
        head = subprocess.run(
            ["git", "show", "HEAD:src/arkrag/answer.py"], cwd=str(ROOT),
            capture_output=True, check=False, env=subproc.child_env(),
        )
        if head.returncode != 0:
            print("[2] 取 HEAD 版 answer.py 失败", file=sys.stderr)
            return 2
        (orig_src / "arkrag" / "answer.py").write_bytes(head.stdout)
        runs.append(_spawn(tmp, orig_src, "orig"))
        for i, mode in enumerate(("false", "true", "true")):
            runs.append(_spawn(tmp, ROOT / "src", mode, tag=f"{mode}{i}"))

        orig, false_run = runs[0], runs[1]
        true_runs = runs[2:]
        rollback_equal = orig["digest"] == false_run["digest"]
        stable = len({r["digest"] for r in true_runs}) == 1
        echo_ok = (orig["effective_chapter_answer_path"] is False
                   and false_run["effective_chapter_answer_path"] is False
                   and all(r["effective_chapter_answer_path"] is True for r in true_runs))

        doc = {
            "generated_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": git_rev(),
            "python": platform.python_version(),
            "os": platform.platform(),
            "command": "py -3 scripts/a_p3_chapter_round.py --stage rollback",
            "observed_fields": list(OBSERVE_FIELDS),
            "runs": runs,
            "orig_equals_false": rollback_equal,
            "true_runs_stable": stable,
            "switch_echo_ok": echo_ok,
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

        print("[回滚等价性] 观测向量总体 SHA-256")
        for r in runs:
            print(f"  {r['tag']:>8}  mode={r['mode']:<5} "
                  f"CHAPTER_ANSWER_PATH={r['effective_chapter_answer_path']}  {r['digest']}")
        print(f"  改动前 == 新源码 False ? {rollback_equal}")
        print(f"  True 两次相同 ? {stable}")
        print(f"  开关实际取值回显正确 ? {echo_ok}")
        print(f"  输出: {rel(out_json)}")
        if not (rollback_equal and stable and echo_ok):
            print("\n[4] 回滚等价性不成立——拒绝出结论", file=sys.stderr)
            return 4
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _spawn(tmp: Path, src_dir: Path, mode: str, tag: str = "") -> dict:
    out_path = tmp / f"run_{tag or mode}.json"
    env = subproc.child_env()
    env["PYTHONPATH"] = str(src_dir)
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--stage", "_one",
         "--mode", mode, "--out-json", str(out_path)],
        cwd=str(ROOT), capture_output=True, text=True,
        encoding=subproc.PIPE_ENCODING, errors="replace",
        timeout=1800, check=False, env=env,
    )
    if not out_path.exists():
        raise SystemExit(f"[2] 子进程未产出 {mode}: rc={proc.returncode} "
                         f"{(proc.stderr or '')[-800:]}")
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    payload["tag"] = tag or mode
    payload["rc"] = proc.returncode
    return payload


# ---------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P3 第三轮·cross_chunk 章节包：真跑/反向/回滚")
    ap.add_argument("--stage", choices=("real", "reverse", "rollback", "_one"), default="real")
    ap.add_argument("--mode", default="", help="_one 用：orig / false / true")
    ap.add_argument("--gold-set", default=str(DEFAULT_GOLD))
    ap.add_argument("--out-json", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    gold_path = Path(args.gold_set)
    if not gold_path.is_absolute():
        gold_path = ROOT / gold_path
    if not gold_path.exists():
        print(f"[2] 输入不可用：{args.gold_set}", file=sys.stderr)
        return 2

    default_out = {
        "real": ROOT / "data" / "a_p3_chapter_round.json",
        "reverse": ROOT / "data" / "a_p3_chapter_reverse.json",
        "rollback": ROOT / "data" / "a_p3_chapter_rollback.json",
        "_one": None,
    }[args.stage]
    out_json = Path(args.out_json) if args.out_json else default_out
    if out_json is not None and not out_json.is_absolute():
        out_json = ROOT / out_json

    if args.stage == "_one":
        if out_json is None:
            print("[2] _one 需要 --out-json", file=sys.stderr)
            return 2
        return _one_run(args.mode, out_json)
    if args.stage == "rollback":
        return stage_rollback(out_json)

    items = load_dev(gold_path, args.limit or None)
    if args.stage == "real":
        return stage_real(gold_path, out_json, items)
    return stage_reverse(gold_path, out_json, items)


if __name__ == "__main__":
    raise SystemExit(main())
