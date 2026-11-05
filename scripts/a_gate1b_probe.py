"""闸门 1b 前置测量：被 `_assertion_relevant` 挡掉的断言长什么样、放行后能不能答对。

**本脚本只测量，不改任何生产代码。** 放行规则通过进程内临时替换
`arkrag.retrieve._assertion_relevant` 实现，跑完自动还原（try/finally）。

背景（Lead `docs/lead_引用覆盖五闸门实测.md` 的五闸门六态）：引用覆盖失败的题里有一类
「断言就在上下文块里、也落在 required 指定的那句证据上，但 `build_context` 第 1 步用
`_assertion_relevant` 把它过滤掉了，回答层连候选都拿不到」。这一类在 v1 300 应答题上是
29 题。本脚本要回答两个问题：

1. 挡掉它们的到底是谁——端点不在焦点实体上，还是谓词不在关系提示里？
2. 如果放行，覆盖率与**答案质量**各变多少？（分标题题 / 非标题题两群看）

## 两个已踩过的坑（照做，别重犯）

* **口径必须同一**：题级的「有 required」= 该题 `required_citations` 里**全部**落在
  上下文（`all(...)`）；条级的「断言在」= 某条断言的 `evidence_text` **包含**该 required
  的 `evidence`。两处混用会算出假交集（Lead 上一次用「任一」对「全部」得 55，真值 0）。
* **喂判据函数它读不到的字段 = 永远落默认分支**：`answer._claim_supported_by_line` 读
  `claim["start"]/["end"]`，不给偏移就恒返回 False。

判据同源（逐字对齐，不自己发挥）：
* `scripts/ask.py:283-298` 引用覆盖 `coverage_of()`
* `scripts/ask.py:262-270` 引用定位硬失败 `citation_scan()`
* 答案质量：`gold.answer` 或任一 `acceptable_answers` 出现在 `answer` 里

用法：
    py -3 scripts/a_gate1b_probe.py --stage diag            # 只看被挡断言的画像
    py -3 scripts/a_gate1b_probe.py --stage rules           # 放行规则真跑（全量 dev）
    py -3 scripts/a_gate1b_probe.py --stage rules --limit 12  # 冒烟（数字不可作结论）

退出码：0 跑通 / 2 输入不可用 / 4 自检失败。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arkrag import config, pipeline, retrieve, subproc  # noqa: E402

GEN = "g-20261103T132152Z-06a741"
EVAL_TOP_K = 20
EVAL_GRAPH_DEPTH = 1


# ---------------------------------------------------------------- 工具


def rel_to_root(path: Path) -> str:
    """入库 JSON 里不许出现绝对机台路径（test_portable_paths.py 会判红）。"""
    try:
        return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def git_rev() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=20,
            # 子进程输出按 UTF-8 解码，父子两端编码必须一致，见 src/arkrag/subproc.py
            env=subproc.child_env(),
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def load_dev(gold_path: Path, limit: int | None) -> list[dict]:
    """dev 调参集里的应答题：split==dev、非应拒、有 gold 块。

    **不做 `[:n]` 挑选**——dev 顺序按类聚簇（前 76 题全是 single_chunk），
    截前缀只覆盖一类，历史上已经坑过一次（`ask.py --limit`）。
    """
    data = json.loads(gold_path.read_text(encoding="utf-8"))
    rows = [
        it
        for it in data["items"]
        if it.get("split") == "dev"
        and not it.get("should_abstain")
        and (it.get("gold_chunk_ids") or [])
    ]
    rows.sort(key=lambda it: it["question_id"])
    if limit:
        # 抽样也必须跨类，否则数字会只代表一类
        by_cat: dict[str, list[dict]] = {}
        for it in rows:
            by_cat.setdefault(str(it.get("category") or ""), []).append(it)
        picked: list[dict] = []
        while len(picked) < limit and any(by_cat.values()):
            for cat in sorted(by_cat):
                if by_cat[cat] and len(picked) < limit:
                    picked.append(by_cat[cat].pop(0))
        return picked
    return rows


def answer_hits_gold(item: dict, answer: str) -> bool:
    """答案质量判据：gold 答案或任一 acceptable 出现在答案里。"""
    text = str(answer or "")
    if not text:
        return False
    cands = [str(item.get("answer") or "")]
    cands += [str(x) for x in (item.get("acceptable_answers") or [])]
    return any(c and c in text for c in cands)


def coverage_of(item: dict, citations: list[dict]) -> bool:
    """引用覆盖：required 全部命中（chunk_id 相等且 evidence 是引用 evidence 的子串）。

    与 `scripts/ask.py:283-298` 逐字同口径。
    """
    reqs = item.get("required_citations") or []
    if not reqs:
        return False
    for req in reqs:
        cid = str(req.get("chunk_id") or "")
        ev = str(req.get("evidence") or "")
        ok = False
        for cite in citations or []:
            if str(cite.get("chunk_id") or "") == cid and ev in str(cite.get("evidence") or ""):
                ok = True
                break
        if not ok:
            return False
    return True


def citation_scan(context_units: list[dict], citations: list[dict]) -> dict:
    """引用定位：`text[start:end] == evidence` 逐条比对。与 `scripts/ask.py:262-270` 同口径。"""
    by_id = {str(u.get("chunk_id") or ""): u for u in context_units or []}
    total = 0
    failed = 0
    for cite in citations or []:
        total += 1
        unit = by_id.get(str(cite.get("chunk_id") or ""))
        if unit is None:
            failed += 1
            continue
        text = str(unit.get("text") or "")
        start = int(cite.get("start", -1))
        end = int(cite.get("end", -1))
        if start < 0 or end < start or text[start:end] != str(cite.get("evidence") or ""):
            failed += 1
    return {"total": total, "failed": failed}


# ---------------------------------------------------------------- 放行规则

def make_rule(name: str, question: str, required_evidence: list[str]):
    """造一个替换版的 `_assertion_relevant`。

    **纯在线可得**（不看 gold）的规则只有 `quote` / `any_predicate` / `admit_all`；
    `gold_required` 是**上界参照**——它按 required 的 evidence 直接放行，线上拿不到，
    只用来量「即便放行也救不回来」的那部分。

    为什么需要 `required_evidence`：规则里要做证据比对，但那是 gold 侧信息，
    只有 `gold_required` 用得到；其余规则一律不碰它（防越界）。
    """
    evs = [e for e in required_evidence if e]

    def rule(rec: dict, focus_ids: set[str], hint_predicates: set[str]) -> bool:
        if name == "none":
            return retrieve._assertion_relevant_orig(rec, focus_ids, hint_predicates)
        if name == "admit_all":
            return True
        if name == "quote":
            # 问句里「」引用的原文，若就是这条断言的证据句，放行。
            # 这是用户提问时自己贴的证据，属在线可得信息。
            ev = str(rec.get("evidence_text") or "")
            if len(ev) >= 4 and ev in question:
                return True
            return retrieve._assertion_relevant_orig(rec, focus_ids, hint_predicates)
        if name == "any_predicate":
            # 放宽到「谓词是任意合法关系」——相当于去掉端点在焦点实体上的要求。
            # 这是「放宽准入」最激进但仍可辩护的形态：不断言端点归属，只信谓词表。
            if str(rec.get("predicate") or ""):
                return True
            return retrieve._assertion_relevant_orig(rec, focus_ids, hint_predicates)
        if name == "gold_required":
            ev = str(rec.get("evidence_text") or "")
            for req in evs:
                if req and (req in ev or ev in req):
                    return True
            return retrieve._assertion_relevant_orig(rec, focus_ids, hint_predicates)
        raise SystemExit(f"[2] 未知规则 {name}")

    return rule


# ---------------------------------------------------------------- 主流程


def run_one(pipe: dict, item: dict, rule_name: str, need_required_ev: bool) -> dict:
    """跑一题。规则在这里临时换上，跑完立刻还原。"""
    qid = item["question_id"]
    reqs = item.get("required_citations") or []
    req_ev = [str(r.get("evidence") or "") for r in reqs] if need_required_ev else []
    rule = make_rule(rule_name, str(item.get("question") or ""), req_ev)
    retrieve._assertion_relevant = rule
    try:
        t0 = time.perf_counter()
        out = pipeline.ask_once(pipe, str(item.get("question") or ""))
        ms = (time.perf_counter() - t0) * 1000.0
    finally:
        retrieve._assertion_relevant = retrieve._assertion_relevant_orig

    ctx = out.get("context") or {}
    ans = out.get("answer") or {}
    units = ctx.get("units") or []
    cites = ans.get("citations") or []
    unit_ids = {str(u.get("chunk_id") or "") for u in units}
    req_ids = [str(r.get("chunk_id") or "") for r in reqs]
    covered = coverage_of(item, cites)
    scan = citation_scan(units, cites)
    return {
        "question_id": qid,
        "category": str(item.get("category") or ""),
        "origin": str(item.get("origin") or ""),
        "gold_answer": str(item.get("answer") or ""),
        "answer": str(ans.get("answer") or ""),
        "answer_hits_gold": answer_hits_gold(item, str(ans.get("answer") or "")),
        "covered": covered,
        "correct_coverage": covered and answer_hits_gold(item, str(ans.get("answer") or "")),
        "abstained": bool(ans.get("abstained")),
        "n_required": len(reqs),
        "required_in_context": bool(req_ids) and all(c in unit_ids for c in req_ids),
        "required_in_context_any": any(c in unit_ids for c in req_ids),
        "citations": len(cites),
        "citation_failed": scan["failed"],
        "units": len(units),
        "n_hit_assertions": sum(len(u.get("assertions") or []) for u in units),
        "answer_len": len(str(ans.get("answer") or "")),
        "elapsed_ms": round(ms, 1),
    }


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    covered = sum(1 for r in rows if r["covered"])
    correct = sum(1 for r in rows if r["correct_coverage"])
    cites = sum(r["citations"] for r in rows)
    failed = sum(r["citation_failed"] for r in rows)
    return {
        "n": n,
        "covered": covered,
        "correct_coverage": correct,
        "answer_hits_gold": sum(1 for r in rows if r["answer_hits_gold"]),
        "abstained": sum(1 for r in rows if r["abstained"]),
        "required_in_context": sum(1 for r in rows if r["required_in_context"]),
        "citations": cites,
        "citation_failed": failed,
        "locate_pass_rate": round((cites - failed) / cites, 4) if cites else None,
        "answer_len_median": sorted(r["answer_len"] for r in rows)[n // 2] if n else 0,
    }


def compare(base: dict, cand: dict) -> dict:
    """四维 + 正确覆盖，各报救/丢题号（R2b' 通用规则：净额不能抵消任一维回退）。"""
    bmap = {r["question_id"]: r for r in base}
    cmap = {r["question_id"]: r for r in cand}
    out = {}
    for dim in ("covered", "correct_coverage", "answer_hits_gold", "required_in_context"):
        gain = [q for q in bmap if cmap[q][dim] and not bmap[q][dim]]
        lose = [q for q in bmap if bmap[q][dim] and not cmap[q][dim]]
        out[dim] = {
            "base": sum(1 for q in bmap if bmap[q][dim]),
            "cand": sum(1 for q in cmap if cmap[q][dim]),
            "gain": len(gain),
            "lose": len(lose),
            "net": len(gain) - len(lose),
            "gain_ids": sorted(gain),
            "lose_ids": sorted(lose),
        }
    # 交集 / 差集（不许只并列净额）
    cov_gain = set(out["covered"]["gain_ids"])
    cor_gain = set(out["correct_coverage"]["gain_ids"])
    out["gain_set_relations"] = {
        "covered_only": sorted(cov_gain - cor_gain),
        "correct_only": sorted(cor_gain - cov_gain),
        "both": sorted(cov_gain & cor_gain),
        "covered_gain_n": len(cov_gain),
        "correct_gain_n": len(cor_gain),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="闸门 1b 前置测量（只测不改）")
    ap.add_argument("--generation", default=GEN)
    ap.add_argument("--gold-v2", default=str(ROOT / "data" / "goldset" / "gold-v2.json"))
    ap.add_argument("--stage", choices=("diag", "rules"), default="diag")
    ap.add_argument(
        "--rules",
        default="none,quote,any_predicate,admit_all,gold_required",
        help="逗号分隔，执行顺序即结果顺序",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()
    if args.stage == "diag":
        args.out_json = args.out_json or str(ROOT / "data" / "a_gate1b_diag.json")
    else:
        args.out_json = args.out_json or str(ROOT / "data" / "a_gate1b_rules.json")

    gold_path = Path(args.gold_v2)
    if not gold_path.exists():
        print(f"[2] 找不到 gold set：{gold_path}", file=sys.stderr)
        return 2

    items = load_dev(gold_path, args.limit or None)
    if not items:
        print("[2] dev 应答题为空", file=sys.stderr)
        return 2
    sampled = bool(args.limit)
    print(f"[1] dev 应答题 {len(items)} 题" + ("（抽样，数字不可作结论）" if sampled else ""))

    # 备份原函数，供规则回退
    retrieve._assertion_relevant_orig = retrieve._assertion_relevant

    t0 = time.perf_counter()
    loaded = pipeline.load_generation(args.generation)
    pipe = pipeline.build_pipeline(loaded, top_k=EVAL_TOP_K, graph_depth=EVAL_GRAPH_DEPTH)
    print(f"[1] 载入世代 {args.generation} 用时 {time.perf_counter() - t0:.1f}s")

    result: dict = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_rev(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "command": "py -3 scripts/a_gate1b_probe.py " + " ".join(sys.argv[1:]),
        "generation": args.generation,
        "gold_set": rel_to_root(gold_path),
        "split": "dev",
        "split_policy": "pre-split-diagnostic（只跑 dev 380 里的应答题；验收集 260 未碰；结论不许外推到 new-holdout 160）",
        "eval_top_k": EVAL_TOP_K,
        "eval_graph_depth": EVAL_GRAPH_DEPTH,
        "sampled": sampled,
    }

    # 基线
    base_rows = [run_one(pipe, it, "none", False) for it in items]
    result["baseline"] = aggregate(base_rows)
    print(f"[1] 基线：covered={result['baseline']['covered']} correct={result['baseline']['correct_coverage']}")

    if args.stage == "diag":
        # 画像：对每题被挡掉的 required 断言，量端点/谓词状态
        detail = []
        for item in items:
            qid = item["question_id"]
            row = next(r for r in base_rows if r["question_id"] == qid)
            if row["covered"] or not row["required_in_context"]:
                continue
            out = pipeline.ask_once(pipe, str(item.get("question") or ""))
            qa = out.get("query_analysis") or {}
            focus = {str(c.get("entity_id") or "") for c in (qa.get("entity_candidates") or [])}
            focus.discard("")
            hints = {str(h.get("canonical") or h.get("raw") or "") for h in (qa.get("relation_hints") or [])}
            hints.discard("")
            reqs = item.get("required_citations") or []
            per_req = []
            for req in reqs:
                cid = str(req.get("chunk_id") or "")
                ev = str(req.get("evidence") or "")
                recs = pipe["retriever"].assertions_by_chunk.get(cid) or []
                matched = [r for r in recs if ev and ev in str(r.get("evidence_text") or "")]
                blocked = [r for r in matched if not retrieve._assertion_relevant_orig(r, focus, hints)]
                per_req.append(
                    {
                        "chunk_id": cid,
                        "evidence": ev,
                        "n_matched": len(matched),
                        "n_blocked": len(blocked),
                        "blocked_detail": [
                            {
                                "assertion_id": str(r.get("assertion_id") or ""),
                                "predicate": str(r.get("predicate") or ""),
                                "subject": pipe["retriever"].display_name(str(r.get("subject_id") or "")),
                                "object": pipe["retriever"].display_name(str(r.get("object_id") or "")),
                                "subj_in_focus": str(r.get("subject_id") or "") in focus,
                                "obj_in_focus": str(r.get("object_id") or "") in focus,
                                "pred_in_hints": str(r.get("predicate") or "") in hints,
                                "evidence_text": str(r.get("evidence_text") or ""),
                            }
                            for r in blocked
                        ],
                    }
                )
            n_blocked = sum(p["n_blocked"] for p in per_req)
            n_matched = sum(p["n_matched"] for p in per_req)
            if n_matched == 0 or n_blocked == 0:
                continue  # 不属于闸门 1b
            title = ""
            for u in (out.get("context") or {}).get("units") or []:
                if str(u.get("chunk_id")) in {p["chunk_id"] for p in per_req}:
                    title = str(u.get("title") or "")
                    break
            detail.append(
                {
                    "question_id": qid,
                    "category": str(item.get("category") or ""),
                    "question": str(item.get("question") or ""),
                    "gold_answer": str(item.get("answer") or ""),
                    "required_title": title,
                    "answer_equals_title": bool(title) and str(item.get("answer") or "") == title,
                    "focus_ids_n": len(focus),
                    "hint_predicates": sorted(hints),
                    "n_matched": n_matched,
                    "n_blocked": n_blocked,
                    "per_req": per_req,
                    "answer": str((out.get("answer") or {}).get("answer") or ""),
                }
            )
        result["gate1b_diag"] = {
            "n": len(detail),
            "by_category": _count(detail, "category"),
            "answer_equals_title": sum(1 for d in detail if d["answer_equals_title"]),
            # 挡掉原因分解（条级）
            "blocked_reason": {
                "endpoint_not_focus": sum(
                    1
                    for d in detail
                    for p in d["per_req"]
                    for b in p["blocked_detail"]
                    if not b["subj_in_focus"] and not b["obj_in_focus"]
                ),
                "predicate_not_hint": sum(
                    1
                    for d in detail
                    for p in d["per_req"]
                    for b in p["blocked_detail"]
                    if not b["pred_in_hints"]
                ),
                "total_blocked": sum(d["n_blocked"] for d in detail),
            },
            "per_question": detail,
        }
        print(
            f"[1] 闸门1b 候选 {len(detail)} 题，"
            f"答案等于块标题 {result['gate1b_diag']['answer_equals_title']} 题"
        )

    if args.stage == "rules":
        rules = [r.strip() for r in args.rules.split(",") if r.strip()]
        if rules and rules[0] != "none":
            rules = ["none"] + rules
        rule_rows: dict[str, list[dict]] = {"none": base_rows}
        for name in rules:
            if name == "none":
                continue
            t0 = time.perf_counter()
            rows = [run_one(pipe, it, name, name == "gold_required") for it in items]
            rule_rows[name] = rows
            agg = aggregate(rows)
            cmp = compare(base_rows, rows)
            print(
                f"[1] {name}: covered={agg['covered']} correct={agg['correct_coverage']} "
                f"answer_hits_gold={agg['answer_hits_gold']} "
                f"cov救{cmp['covered']['gain']}/丢{cmp['covered']['lose']} "
                f"正确救{cmp['correct_coverage']['gain']}/丢{cmp['correct_coverage']['lose']} "
                f"({time.perf_counter() - t0:.0f}s)"
            )
        result["rules"] = {
            name: {"aggregate": aggregate(rows), "vs_baseline": compare(base_rows, rows)}
            for name, rows in rule_rows.items()
        }
        result["per_question"] = {name: rows for name, rows in rule_rows.items()}

    # 自检：规则跑完后基线口径必须还在（防止替换没还原）
    sanity = run_one(pipe, items[0], "none", False)
    if sanity["question_id"] != items[0]["question_id"]:
        print("[4] 自检失败：还原后基线抖动", file=sys.stderr)
        return 4
    result["sanity"] = {"restored_ok": True}

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[0] 写出 {out_path}（{out_path.stat().st_size} 字节）")
    return 0


def _count(rows: list[dict], key: str) -> dict:
    out: dict[str, int] = {}
    for r in rows:
        out[str(r.get(key) or "")] = out.get(str(r.get(key) or ""), 0) + 1
    return out


if __name__ == "__main__":
    raise SystemExit(main())
