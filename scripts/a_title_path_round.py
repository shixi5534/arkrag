"""P3 标题通路：真跑验收 + 反向实测 + 回滚等价比对。

**这是生产通路真跑**，不是进程内替换 search 的离线回放：直接调 `pipeline.ask_once`，
唯一的差别是模块级开关 `answer.TITLE_ANSWER_PATH` 的取值。同一份代码、只差这一个
布尔量，因此「候选 - 基线」是干净的单变量对照。

## 为什么单开一个脚本而不再往 a_p3_line_citation.py 里加

那个脚本的变量是 `CITATION_WHOLE_LINE`（引用粒度），本脚本的变量是
`TITLE_ANSWER_PATH`（新增答案形态）。两者虽然都落在 `citations` 上，但**不是同一个
旋钮**，混在一个脚本里会让「本轮改了什么」在输出里说不清。Lead m03916 明确要求
「整行引用条数」与「断言 span 引用条数」分开计，这个脚本就是干这件事的。

## 三个 stage

- `--stage real`：基线（开关 False）vs 候选（开关 True）各跑一遍全量 dev，出五数 +
  两档并列（干净 22 / 含问句泄露 39）+ 转作答清单 + 应拒题清单 + 两种引用粒度分列。
- `--stage reverse`：R7'-a 换标题、R7'-b 换引文。这两笔是零容忍测试，失败即撤销改动。
- `--stage rollback`：四轮子进程各跑一遍全量 dev，比对观测向量总体 SHA-256：
  改动前源码（HEAD 版 answer.py，无该开关） / 新源码 False / 新源码 True（跑两次）。

## 三条硬规矩（都是本项目踩过的坑）

1. `load_dev()` **不做 `[:n]` 挑选**——dev 顺序按类聚簇，截前缀只覆盖一类
   （`ask.py --limit` 历史事故）。抽样必须跨类，且打印「本次数字不可作结论」。
2. 判据函数（`coverage_of` / `citation_scan` / `answer_hits_gold`）与
   `scripts/ask.py:262-302` 逐字同口径，不自己发明。
3. 子进程一律 `env=subproc.child_env()`（Gate 0 事故的规矩，AST 门会扫这条）。

退出码：0 跑通 / 2 输入不可用 / 4 自检失败（自检失败时**拒绝出结论**）。
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import hashlib
import json
import platform
import re
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

# R7'-a 用的假标题：必须与任何真实块标题都不同，且不可能是问句里原有内容。
FAKE_TITLE = "完全不存在的假标题"
# R7'-b 用的假引文：足够长、足够不像语料里的话，确保在该块里找不到。
FAKE_QUOTE = "这段引文在该块里绝对不存在请勿当真"


def rel(path: Path) -> str:
    return config.rel_to_root(path)


def git_rev() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
            capture_output=True, text=True, timeout=20, check=False,
            # 子进程输出按 UTF-8 解码，父子两端编码必须一致，见 src/arkrag/subproc.py
            env=subproc.child_env(),
        )
        return (out.stdout or "").strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def load_dev(gold_path: Path, limit: int | None = None) -> list[dict]:
    """dev 全集：split==dev，**不按应拒/非应拒筛**。

    保留应拒题是刻意的：R3' 要证明「标题通路没把该拒的翻成作答」，如果只跑应答题，
    这条判据就无从检验（Lead m03916 要求单列那 5 道应拒题的题号）。

    **不做 `[:n]` 挑选**；`--limit` 时按类跨类抽样，并打印「不可作结论」。
    """
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
    tp = st.get("title_path") or {}

    required = item.get("required_citations") or []
    ctx_ids = {str(u.get("chunk_id") or "") for u in units}
    scan = citation_scan(units, cites)
    covered = (not item.get("should_abstain")) and bool(required) and coverage_of(item, cites)
    hits_gold = answer_hits_gold(item, str(ans.get("answer") or ""))

    # 两种引用粒度分列（Lead m03916 要求）：标题通路自建的是整行引用，其余是断言 span
    line_cites = [c for c in cites if str(c.get("granularity") or "") == "line"]
    span_cites = [c for c in cites if str(c.get("granularity") or "") != "line"]

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
        "answer": str(ans.get("answer") or ""),
        "answer_sha": hashlib.sha256(
            str(ans.get("answer") or "").encode("utf-8")).hexdigest()[:16],
        "citations": len(cites),
        "citations_distinct": len({(str(c.get("chunk_id") or ""), str(c.get("evidence") or ""))
                                   for c in cites}),
        "citations_line": len(line_cites),
        "citations_span": len(span_cites),
        "cite_ok": scan["total"] - scan["failed"],
        "cite_bad": scan["failed"],
        "title_path_applied": bool(tp.get("applied")),
        "title_path_source": str(tp.get("title_source") or ""),
        "title_path_located": int(tp.get("located_units") or 0),
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
    answer_mod.TITLE_ANSWER_PATH = bool(enabled)
    try:
        return [run_one(pipe, it) for it in items]
    finally:
        answer_mod.TITLE_ANSWER_PATH = False


def build_all(gold_path: Path):
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
    """三分母并列。分母只在**应答题**上取（与 P3 第一轮同口径）。"""
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
        "coverage_ok": cov,
        "correct_coverage": cor,
        "den_all_answerable": n_all,
        "den_span_reachable": n_span,
        "den_two_gates": n_gate,
        # 覆盖率口径（覆盖率为主判据的历史口径）
        "r_all": round(cov / n_all, 6) if n_all else 0.0,
        "r_span": round(cov / n_span, 6) if n_span else 0.0,
        "r_gates": round(cov / n_gate, 6) if n_gate else 0.0,
        # 正确覆盖口径（覆盖成立 ∧ 答案含 gold）
        "c_all": round(cor / n_all, 6) if n_all else 0.0,
        "c_span": round(cor / n_span, 6) if n_span else 0.0,
        "c_gates": round(cor / n_gate, 6) if n_gate else 0.0,
    }


def absten_block(rows: list[dict]) -> dict:
    should = [r for r in rows if r["should_abstain"]]
    answerable = [r for r in rows if not r["should_abstain"]]
    return {
        "n_should_abstain": len(should),
        "n_answerable": len(answerable),
        "abstained_should": sum(1 for r in should if r["abstained"]),
        "abstained_answerable": sum(1 for r in answerable if r["abstained"]),
        "abstained_total": sum(1 for r in rows if r["abstained"]),
        "correct_should": sum(1 for r in should if r["abstained"]),
        "correct_answerable": sum(1 for r in answerable if not r["abstained"]),
        "abstain_accuracy": round(
            (sum(1 for r in should if r["abstained"])
             + sum(1 for r in answerable if not r["abstained"])) / len(rows), 6) if rows else 0.0,
        # 该拒却答了的题（回到作答）——R3' 的核心清单
        "should_abstain_but_answered": sorted(
            r["question_id"] for r in should if not r["abstained"]),
        # 该答却拒了的题
        "answerable_but_abstained": sorted(
            r["question_id"] for r in answerable if r["abstained"]),
    }


def cite_block(rows: list[dict]) -> dict:
    total = sum(r["citations"] for r in rows)
    distinct = sum(r["citations_distinct"] for r in rows)
    ok = sum(r["cite_ok"] for r in rows)
    bad = sum(r["cite_bad"] for r in rows)
    return {
        "total": total,
        "distinct": distinct,
        "line": sum(r["citations_line"] for r in rows),
        "span": sum(r["citations_span"] for r in rows),
        "ok": ok,
        "bad": bad,
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
        cat = r["category"]
        slot = out.setdefault(cat, {"n": 0, "base": 0, "cand": 0, "gain": [], "lose": []})
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
    loaded, chunks, by_chunk, pipe = build_all(gold_path)
    items_by_id = {it["question_id"]: it for it in items}
    answerable = answerable_of(items)
    print(f"[load] dev={len(items)} 应答题={len(answerable)} 应拒={len(items)-len(answerable)} "
          f"块={len(chunks)}")

    t = time.perf_counter()
    base = run_config(pipe, items, enabled=False)
    print(f"[run] 基线（TITLE_ANSWER_PATH=False）{time.perf_counter()-t:.1f}s")
    t = time.perf_counter()
    cand = run_config(pipe, items, enabled=True)
    print(f"[run] 候选（TITLE_ANSWER_PATH=True）{time.perf_counter()-t:.1f}s")

    bmap = {r["question_id"]: r for r in base}
    cmap = {r["question_id"]: r for r in cand}

    # 两档拆分与零命中清单（先读，自检要用）
    fa = json.loads((ROOT / "data" / "a_title_feasible_split.json").read_text(encoding="utf-8"))
    feas_rows = json.loads(
        (ROOT / "data" / "a_title_impl_feasible.json").read_text(encoding="utf-8"))
    clean = set(fa["clean_no_leak"])
    leak = set(fa["question_leak"])
    zero_hit = {r["qid"] for r in feas_rows if int(r.get("n_hit") or 0) == 0}

    # ---- 自检 1：开关 False 时不该有任何 title_path 生效
    leaked = sorted(q for q, r in bmap.items() if r["title_path_applied"])
    checks = [{"check": "开关 False 时无题触发标题通路", "pass": not leaked, "detail": leaked[:5]}]

    # ---- 自检 2：触发题数必须等于「在线可定位」的题数（61），不是模板题数（76）
    # 76 是触发模板的题数；其中 15 道引文在上下文里找不到，按设计**不触发**（落回原路径）。
    trig = sorted(q for q, r in cmap.items() if r["title_path_applied"])
    expect_trig = len(fa["feasible_all"])
    if len(items) == 380:
        checks.append({"check": f"触发题数 == {expect_trig}（在线可定位）",
                       "pass": len(trig) == expect_trig, "detail": f"实际 {len(trig)}"})
        # 15 道模板题必须全部「不触发」且行为与基线一致
        no_hit = [q for q in (set(fa["feasible_all"]) | set(fa["question_leak"])
                              | set(fa["clean_no_leak"]))
                  if q not in set(fa["feasible_all"])]
        silent_bad = sorted(q for q in no_hit
                            if q in cmap and (cmap[q]["title_path_applied"]
                                              or cmap[q]["answer_sha"] != bmap[q]["answer_sha"]))
        checks.append({"check": "零命中 15 题不触发且行为与基线一致",
                       "pass": not silent_bad, "detail": silent_bad[:5]})
    # ---- 自检 3：触发的题必须全是 single_chunk
    wrong_cat = sorted(q for q in trig if cmap[q]["category"] != "single_chunk")
    checks.append({"check": "触发题全为 single_chunk", "pass": not wrong_cat,
                   "detail": wrong_cat[:5]})

    cov_base = cov_block(base, by_chunk, items_by_id)
    cov_cand = cov_block(cand, by_chunk, items_by_id)
    cite_base, cite_cand = cite_block(base), cite_block(cand)
    abs_base, abs_cand = absten_block(base), absten_block(cand)
    cor_diff = diff_set(base, cand, "correct_coverage")
    cov_diff = diff_set(base, cand, "covered")
    ans_diff = diff_set(base, cand, "answer_hits_gold")

    # ---- 两档并列：干净（问句不含答案）在前，含泄露在后
    clean = set(fa["clean_no_leak"])
    leak = set(fa["question_leak"])
    clean_gain = [q for q in cor_diff["gain"] if q in clean]
    leak_gain = [q for q in cor_diff["gain"] if q in leak]
    unfired = [q for q in (clean | leak) if q in cmap and not cmap[q]["title_path_applied"]]

    # ---- 结构性自检（Lead m04046 第三节的教训：加法校验抓不到对调，结构性等式才抓得住）
    # 干净 ⟺ 模板 B ∩ 可行、泄露 ⟺ 模板 A ∩ 可行，这两条必须在**两层**上都严格成立。
    # 只验「干净 + 泄露 = 可行」是不够的：两个数对调后加法照样成立。
    formA_re = re.compile(r"在剧情《[^》]*》的正文里")
    formB_re = re.compile(r"下面这段原文来自哪一段剧情")
    form_of = {}
    for it in items:
        q = str(it.get("question") or "")
        if formA_re.search(q):
            form_of[it["question_id"]] = "A"
        elif formB_re.search(q):
            form_of[it["question_id"]] = "B"
    formA_feas = {q for q in fa["feasible_all"] if form_of.get(q) == "A"}
    formB_feas = {q for q in fa["feasible_all"] if form_of.get(q) == "B"}
    checks.append({
        "check": "干净集 == 模板B ∩ 可行（结构性等式，防对调）",
        "pass": set(clean) == formB_feas,
        "detail": {"only_clean": sorted(set(clean) - formB_feas)[:5],
                   "only_formB": sorted(formB_feas - set(clean))[:5]},
    })
    checks.append({
        "check": "泄露集 == 模板A ∩ 可行（结构性等式，防对调）",
        "pass": set(leak) == formA_feas,
        "detail": {"only_leak": sorted(set(leak) - formA_feas)[:5],
                   "only_formA": sorted(formA_feas - set(leak))[:5]},
    })
    # 触发题的增益必须全部分布在 干净 ∪ 泄露 里（不许有来路不明的增益）
    stray_gain = [q for q in cor_diff["gain"] if q not in (clean | leak)]
    checks.append({"check": "正确覆盖增益全部落在 干净∪泄露 内",
                   "pass": not stray_gain, "detail": stray_gain[:5]})
    if len(clean_gain) + len(leak_gain) != len(cor_diff["gain"]):
        checks.append({"check": "干净 + 泄露 == 总增益",
                       "pass": False,
                       "detail": f"{len(clean_gain)}+{len(leak_gain)} != {len(cor_diff['gain'])}"})

    # ---- 转作答清单（基线拒答 → 候选作答）
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
        "command": "py -3 scripts/a_title_path_round.py --stage real",
        "generation": DEFAULT_GEN,
        "gold_set": rel(gold_path),
        "split": "dev",
        "split_policy": "pre-split-diagnostic/dev-only：dev 结论不外推到 new-holdout 160",
        "variable": {"name": "TITLE_ANSWER_PATH", "base": False, "cand": True},
        "counts": {"dev_questions": len(items), "answerable": len(answerable),
                   "should_abstain": len(items) - len(answerable)},
        "selfcheck": checks,
        "title_path": {
            "triggered": len(trig), "triggered_ids": trig,
            "gain_in_clean": clean_gain, "gain_in_leak": leak_gain,
            "clean_or_leak_but_not_fired": unfired,
        },
        "coverage": {
            "base": cov_base, "cand": cov_cand,
            "correct_diff": cor_diff, "covered_diff": cov_diff, "answer_diff": ans_diff,
            "correct_by_category": per_category(base, cand, "correct_coverage"),
            "covered_by_category": per_category(base, cand, "covered"),
        },
        "citations": {
            "base": cite_base, "cand": cite_cand,
            # 两种粒度分列（Lead m03916 要求）
            "granularity_note": "line = 标题通路自建的整行引用；span = 断言 span 引用",
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
    out_json.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[自检]")
    for c in checks:
        print(f"  {'PASS' if c['pass'] else 'FAIL'}  {c['check']}"
              + ("" if c["pass"] else f"  {c['detail']}"))

    print("\n[覆盖率 / 正确覆盖] 三分母并列（分母为应答题）")
    for tag, blk in (("base", cov_base), ("cand", cov_cand)):
        print(f"  {tag}: 覆盖 {blk['coverage_ok']}  正确覆盖 {blk['correct_coverage']}  "
              f"/{blk['den_all_answerable']}={blk['c_all']}  "
              f"/{blk['den_span_reachable']}={blk['c_span']}  "
              f"/{blk['den_two_gates']}={blk['c_gates']}")
    print(f"  正确覆盖救 {len(cor_diff['gain'])} 丢 {len(cor_diff['lose'])} "
          f"净 {cor_diff['net']}  按题型 {cor_diff['gain_by_category']}")
    if cor_diff["lose"]:
        print(f"  正确覆盖丢失 {cor_diff['lose']}")
    print(f"  覆盖救 {len(cov_diff['gain'])} 丢 {len(cov_diff['lose'])} 净 {cov_diff['net']}")
    print(f"  答案含 gold 救 {len(ans_diff['gain'])} 丢 {len(ans_diff['lose'])} "
          f"净 {ans_diff['net']}")

    print("\n[两档并列] 干净在前")
    print(f"  干净（模板 B，答案不在问句）: 救 {len(clean_gain)} 题")
    print(f"  含问句泄露（模板 A）      : 救 {len(leak_gain)} 题（只作第二行）")
    if unfired:
        print(f"  预期可兑现但未触发: {unfired}")

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
    print(f"  拒答转作答 {len(turned)} 题，其中「本该拒答」{len(should_which)} 题 "
          f"{should_which}")
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
    """R7'-a 换标题 / R7'-b 换引文。零容忍：任一失败即 exit 4。"""
    fa = json.loads((ROOT / "data" / "a_title_feasible_split.json").read_text(encoding="utf-8"))
    gold_by_id = {it["question_id"]: it for it in items}

    # 只选模板 A（问句里含《真标题》）的可行题——模板 B 的问句里没有《》，换标题是空转。
    picks: list[str] = []
    for qid in fa["feasible_all"]:
        it = gold_by_id.get(qid)
        if not it:
            continue
        q = str(it.get("question") or "")
        answer = str(it.get("answer") or "")
        if f"《{answer}》" in q:
            picks.append(qid)
        if len(picks) == 5:
            break
    if len(picks) < 5:
        print(f"[2] 模板 A 可行题不足 5 道（实际 {len(picks)}）", file=sys.stderr)
        return 2

    _, _, _, pipe = build_all(gold_path)
    answer_mod.TITLE_ANSWER_PATH = True
    rows: list[dict] = []
    try:
        for qid in picks:
            it = gold_by_id[qid]
            answer = str(it.get("answer") or "")
            q0 = str(it.get("question") or "")

            # R7'-a：把《真标题》换成《假标题》，句式与引文不动
            qa = q0.replace(f"《{answer}》", f"《{FAKE_TITLE}》")
            ok_a_constructed = qa != q0
            ra = run_one(pipe, it, question=qa)

            # R7'-b：把「」引文换成一句该块里没有的话
            quotes = answer_mod._QUOTE_RE.findall(q0)
            qb = q0
            for quote in quotes:
                qb = qb.replace(f"「{quote}」", f"「{FAKE_QUOTE}」")
            ok_b_constructed = qb != q0 and FAKE_QUOTE in qb
            rb = run_one(pipe, it, question=qb)

            # 「落回原路径」的正确对照物是**同一改后问句在开关关闭时的结果**，
            # 不是原问句的结果——原问句在开关打开时会走标题通路，拿它当基线是错的
            # （我第一版就这么写，5/5 假红）。
            answer_mod.TITLE_ANSWER_PATH = False
            try:
                rb_off = run_one(pipe, it, question=qb)
            finally:
                answer_mod.TITLE_ANSWER_PATH = True

            # 基线（原问句、开关打开）用于展示「本来会答出 title」
            r0 = run_one(pipe, it)

            rows.append({
                "question_id": qid,
                "gold_answer": answer,
                "question_original": q0,
                "r7a": {
                    "constructed": ok_a_constructed,
                    "question": qa,
                    "applied": ra["title_path_applied"],
                    "answer": ra["answer"],
                    "answer_is_real_title": ra["answer"].strip() == answer.strip(),
                    "pass": ok_a_constructed and ra["title_path_applied"]
                    and ra["answer"].strip() == answer.strip(),
                },
                "r7b": {
                    "constructed": ok_b_constructed,
                    "question": qb,
                    "applied": rb["title_path_applied"],
                    "answer": rb["answer"],
                    "answer_same_as_path_off": rb["answer_sha"] == rb_off["answer_sha"],
                    "pass": ok_b_constructed and not rb["title_path_applied"]
                    and rb["answer_sha"] == rb_off["answer_sha"],
                },
                "baseline": {"applied": r0["title_path_applied"], "answer": r0["answer"]},
            })
    finally:
        answer_mod.TITLE_ANSWER_PATH = False

    a_pass = sum(1 for r in rows if r["r7a"]["pass"])
    b_pass = sum(1 for r in rows if r["r7b"]["pass"])
    doc = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_rev(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "command": "py -3 scripts/a_title_path_round.py --stage reverse",
        "generation": DEFAULT_GEN,
        "gold_set": rel(gold_path),
        "split_policy": "pre-split-diagnostic/dev-only",
        "fake_title": FAKE_TITLE,
        "fake_quote": FAKE_QUOTE,
        "picks": picks,
        "r7a": {"n": len(rows), "pass": a_pass,
                "meaning": "换掉问句里的《真标题》后，答案必须仍是块真实 title"},
        "r7b": {"n": len(rows), "pass": b_pass,
                "meaning": "换掉「」引文后，必须落回原路径（不触发标题通路）且不编答案"},
        "per_question": rows,
    }
    out_json.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[R7'-a 换标题]（答案必须跟块 title 走，不许抄问句）")
    for r in rows:
        tag = "PASS" if r["r7a"]["pass"] else "FAIL"
        print(f"  {tag}  {r['question_id']}  应用={r['r7a']['applied']}  "
              f"答案={r['r7a']['answer']!r}  期望={r['gold_answer']!r}")
    print(f"  合计 {a_pass}/{len(rows)}")
    print("[R7'-b 换引文]（定位不到必须落回原路径，不许瞎猜）")
    for r in rows:
        tag = "PASS" if r["r7b"]["pass"] else "FAIL"
        print(f"  {tag}  {r['question_id']}  应用={r['r7b']['applied']}  "
              f"开关关闭时同名问句结果一致={r['r7b']['answer_same_as_path_off']}  "
              f"答案={r['r7b']['answer'][:40]!r}")
    print(f"  合计 {b_pass}/{len(rows)}")
    print(f"  输出: {rel(out_json)}")

    if a_pass < len(rows) or b_pass < len(rows):
        print("\n[4] 反向实测失败（R7' 触发）——K1' 作废、改动必须撤销", file=sys.stderr)
        return 4
    return 0


# ---------------------------------------------------------------- stage: rollback


OBSERVE_FIELDS = ("answer_sha", "abstained", "citations",
                  "citations_line", "citations_span",
                  "covered", "correct_coverage", "unit_sequence", "title_path_applied")


def digest_of(rows: list[dict]) -> str:
    blob = json.dumps(
        [{k: r[k] for k in ("question_id",) + OBSERVE_FIELDS} for r in rows],
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _one_run(mode: str, out_path: Path) -> int:
    """子进程内部：跑一轮全量 dev，打印摘要与**实际生效的开关值**（必须回显）。"""
    gold_path = DEFAULT_GOLD
    items = load_dev(gold_path)
    _, _, _, pipe = build_all(gold_path)
    if mode == "false":
        answer_mod.TITLE_ANSWER_PATH = False
    elif mode == "true":
        answer_mod.TITLE_ANSWER_PATH = True
    elif mode != "orig":
        print(f"[2] 未知 mode {mode}", file=sys.stderr)
        return 2
    effective = bool(getattr(answer_mod, "TITLE_ANSWER_PATH", False))
    rows = [run_one(pipe, it) for it in items]
    out_path.write_text(
        json.dumps({"mode": mode, "effective_title_answer_path": effective,
                    "digest": digest_of(rows), "n": len(rows),
                    "aggregate": {
                        "covered": sum(1 for r in rows if r["covered"]),
                        "correct_coverage": sum(1 for r in rows if r["correct_coverage"]),
                        "abstained": sum(1 for r in rows if r["abstained"]),
                        "citations": sum(r["citations"] for r in rows),
                    }},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TITLE_ANSWER_PATH={effective} DIGEST={digest_of(rows)}")
    return 0


def stage_rollback(out_json: Path) -> int:
    """四轮：HEAD 版 answer.py（原版，无该开关） / 新源码 False / 新源码 True ×2。

    原版用旁路 src 副本 + PYTHONPATH 切换导入来源——**不许 import 被验的改动**
    （P3 第一轮踩过：回滚探针直接读 `retrieve.COOC_BOOST_WEIGHT`，在改动前的源码上
    立刻 AttributeError，等于验不成）。
    """
    tmp = Path(tempfile.mkdtemp(prefix="a_titlerollback_"))
    runs: list[dict] = []
    try:
        # 1) 原版：src 副本，answer.py 换成 HEAD 版
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
        # 2) 新源码 False / 3-4) 新源码 True 两次
        for i, mode in enumerate(("false", "true", "true")):
            runs.append(_spawn(tmp, ROOT / "src", mode, tag=f"{mode}{i}"))

        orig = runs[0]
        false_run = runs[1]
        true_runs = [r for r in runs[2:]]
        rollback_equal = orig["digest"] == false_run["digest"]
        stable = len({r["digest"] for r in true_runs}) == 1
        # 开关值必须真的生效（否则三次全同，看起来像「功能无效」）
        echo_ok = (orig["effective_title_answer_path"] is False
                   and false_run["effective_title_answer_path"] is False
                   and all(r["effective_title_answer_path"] is True for r in true_runs))

        doc = {
            "generated_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": git_rev(),
            "python": platform.python_version(),
            "os": platform.platform(),
            "command": "py -3 scripts/a_title_path_round.py --stage rollback",
            "observed_fields": list(OBSERVE_FIELDS),
            "runs": runs,
            "orig_equals_false": rollback_equal,
            "true_runs_stable": stable,
            "switch_echo_ok": echo_ok,
        }
        out_json.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

        print("[回滚等价性] 观测向量总体 SHA-256")
        for r in runs:
            print(f"  {r['tag']:>8}  mode={r['mode']:<4} "
                  f"TITLE_ANSWER_PATH={r['effective_title_answer_path']}  "
                  f"{r['digest']}")
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
    """在子进程里跑一轮，读回它写的 json。子进程必须回显开关实际取值。"""
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
    line = ""
    for out_line in (proc.stdout or "").splitlines():
        if "DIGEST=" in out_line:
            line = out_line.strip()
    if not out_path.exists():
        raise SystemExit(f"[2] 子进程未产出 {mode}: rc={proc.returncode} "
                         f"{(proc.stderr or '')[-500:]}")
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    payload["tag"] = tag or mode
    payload["rc"] = proc.returncode
    payload["stdout_tail"] = line
    return payload


# ---------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P3 标题通路：真跑 / 反向实测 / 回滚等价")
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
        "real": ROOT / "data" / "a_title_path_round.json",
        "reverse": ROOT / "data" / "a_title_path_reverse.json",
        "rollback": ROOT / "data" / "a_title_path_rollback.json",
        "_one": None,
    }[args.stage]
    out_json = Path(args.out_json) if args.out_json else default_out
    if out_json is not None and not out_json.is_absolute():
        out_json = ROOT / out_json
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)

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
