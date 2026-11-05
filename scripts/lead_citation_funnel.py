"""引用覆盖（required_citation_coverage）的漏斗实测 —— 只读分析，零新增依赖。

为什么需要这个脚本
=================
P1-b 最初把「88 条 no_assertion」定性成「断言 span 粒度与 gold 引用整句粒度不匹配，
改判定口径就能提分」。实测否掉了这个诊断，并且发现三件事：

1. 指标真判据是**单向**子串：`scripts/ask.py:283-298` 里
   `str(req["evidence"]) in str(cite["evidence"])`。
   `cite["evidence"]` 来自 `answer.py:350` 的 `span["evidence"]`
   ← `retrieve.py:928` 的 `rec["evidence_text"]`，也就是断言短语本身。
   任何用「双向子串」或「区间重叠」算的覆盖率都比指标宽，算出来的天花板不是指标的。

2. `no_assertion` 只出现在四类，且正好是 gold 挖掘时不填 `gold_assertion_ids`
   的那四类：alias(:468) / cross_chunk(:743) / negative_abstain(:1069) /
   adversarial(:1195)。它们的 required evidence 是
   `_window_text(chunk["text"], needle)` 从块正文截出的**整句**，
   与断言 `evidence_text`（抽取器切出的短语）是两条独立产出路径。
   direct_relation(:536) 与 multihop(:619) 之所以 100% 可达，
   是因为 required 直接取自边的 `evidence_assertion_id`，与断言同源。

3. 因此「覆盖率低」不是一件事，而是至少四件：
   标注指错行 / 抽取缺事实 / 块没进上下文 / 块进了但没选中那条 claim。
   四者对应完全不同的工作，混在一起算总分只会得出「不可达」的错误结论。

用法
====
    py -3 scripts/lead_citation_funnel.py                # 全部段落，人读
    py -3 scripts/lead_citation_funnel.py --json out.json # 同时落结构化结果

所有数字都在运行时从数据与源码现场的产物算出，不写死任何结论值。
"""
from __future__ import annotations

import argparse
import collections
import difflib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from arkrag import config, subproc  # noqa: E402

GENERATION = "g-20261103T132152Z-06a741"
SPEAKER_RE = re.compile(r"^[^：:]{1,12}[：:]")


# ---------------------------------------------------------------- 数据装载
def load_generation(gen_id: str) -> tuple[dict[str, list[dict]], dict[str, str]]:
    base = config.ROOT / "data" / "arkrag" / "generations" / gen_id
    by_chunk: dict[str, list[dict]] = collections.defaultdict(list)
    with open(base / "graph" / "assertions.jsonl", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            by_chunk[rec["chunk_id"]].append(rec)
    chunks: dict[str, str] = {}
    with open(base / "chunks.jsonl", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            chunks[rec.get("chunk_id")] = rec.get("text") or ""
    return by_chunk, chunks


def load_gold() -> list[dict]:
    path = config.ROOT / "data" / "goldset" / "gold-v1.json"
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["items"]


# ---------------------------------------------------------------- 判据
def metric_reach_span(item: dict, by_chunk) -> bool:
    """指标真判据：每条 required 的 evidence 是否是某条断言 evidence_text 的子串。"""
    for rc in item.get("required_citations") or []:
        cid, req = rc["chunk_id"], rc["evidence"]
        if not any(req in (rec.get("evidence_text") or "") for rec in by_chunk.get(cid) or []):
            return False
    return True


def _line_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """与 src/arkrag/answer.py:60 同义：把 [start,end) 扩到整行。"""
    start = max(0, min(start, len(text)))
    end = max(start, min(end, len(text)))
    left = text.rfind("\n", 0, start) + 1
    right = text.find("\n", end)
    return left, (len(text) if right < 0 else right)


def metric_reach_line(item: dict, by_chunk, chunks) -> bool:
    """假设口径：required 是否落在某条断言所在的整行内。"""
    for rc in item.get("required_citations") or []:
        cid, req = rc["chunk_id"], rc["evidence"]
        text = chunks.get(cid, "")
        ok = False
        for rec in by_chunk.get(cid) or []:
            s, e = int(rec.get("evidence_start", -1)), int(rec.get("evidence_end", -1))
            if s < 0 or e <= s or e > len(text):
                continue
            a, b = _line_bounds(text, s, e)
            if req in text[a:b]:
                ok = True
                break
        if not ok:
            return False
    return True


def classify_citation(rc: dict, by_chunk) -> str:
    """单条 required 相对断言的三种形态（真判据口径）。"""
    cid, req = rc["chunk_id"], rc["evidence"]
    texts = [(rec.get("evidence_text") or "") for rec in by_chunk.get(cid) or []]
    texts = [t for t in texts if t]
    if not texts:
        return "块内无断言"
    if any(req in t for t in texts):
        return "ok"                 # 断言含 gold 那句
    if any(t in req for t in texts):
        return "fragment"           # 断言是 gold 句的碎片
    return "different"              # 毫无关系


def answer_supported(item: dict, by_chunk) -> bool:
    """答案（或其 acceptable）是否出现在该块某条断言的端点/证据里。

    用来区分「gold 引用指错行」（有断言支持，只是 required 指向别处）
    与「抽取真缺事实」（断言层根本没有这个事实）。
    """
    needles = [item.get("answer") or ""] + list(item.get("acceptable_answers") or [])
    needles = [n.strip() for n in needles if n and len(n.strip()) >= 2]
    if not needles:
        return False
    for rc in item.get("required_citations") or []:
        for rec in by_chunk.get(rc["chunk_id"]) or []:
            hay = " ".join(str(rec.get(k) or "") for k in
                           ("evidence_text", "subject_id", "object_id"))
            if any(n in hay for n in needles):
                return True
    return False


# ---------------------------------------------------------------- 段落
def section_metric_vs_loose(answerable, by_chunk) -> dict:
    """真判据 vs 双向子串：两个口径差多少。"""
    st = collections.Counter()
    for it in answerable:
        for rc in it.get("required_citations") or []:
            st[classify_citation(rc, by_chunk)] += 1
    reach = sum(1 for it in answerable if metric_reach_span(it, by_chunk))
    return {
        "条级形态": dict(st),
        "条级合计": sum(st.values()),
        "题级全可达": reach,
        "题级总数": len(answerable),
        "题级天花板": round(reach / len(answerable), 4) if answerable else 0.0,
    }


def section_by_category(answerable, by_chunk, gold_empty_ids) -> list[dict]:
    """按题型拆：哪些类的 required 永远对不上断言，以及为什么。"""
    rows = []
    for cat in sorted({it["category"] for it in answerable}):
        items = [it for it in answerable if it["category"] == cat]
        ok = sum(1 for it in items if metric_reach_span(it, by_chunk))
        empty = gold_empty_ids.get(cat)
        rows.append({
            "category": cat,
            "题数": len(items),
            "全可达": ok,
            "可达率": round(ok / len(items), 4) if items else 0.0,
            "gold_assertion_ids为空": empty,
        })
    return rows


def section_granularity_myth(answerable, by_chunk) -> dict:
    """「粒度差」检验：在**对不上断言的**那批 required 上，两者到底差多远。

    必须限定在不可达的那批上算。混进「ok」的 279 条会把中位拉高，
    看起来像「长短差一点」，从而掩盖「完全是两行」这个事实。
    """
    lens, zero, contains = [], 0, 0
    req_lens, et_lens = [], []
    n_fragment = n_different = 0
    for it in answerable:
        for rc in it.get("required_citations") or []:
            req = rc["evidence"]
            recs = by_chunk.get(rc["chunk_id"]) or []
            texts = [t for t in ((r.get("evidence_text") or "") for r in recs) if t]
            if not texts:
                continue                      # 块内无断言，单列在别处
            if any(req in t for t in texts):
                continue                      # 可达，不是本段要回答的问题
            if any(t in req for t in texts):
                # 断言只是 gold 句的一个碎片：改引用粒度可能救得回
                n_fragment += 1
                continue
            # 断言与 gold 句毫无包含关系：完全不同的两行
            n_different += 1
            best = 0
            for et in texts:
                if et in req:
                    contains += 1
                m = difflib.SequenceMatcher(None, req, et).find_longest_match(
                    0, len(req), 0, len(et))
                best = max(best, m.size)
            if best == 0:
                zero += 1
            lens.append(best)
            req_lens.append(len(req))
            et_lens.append(min(len(t) for t in texts))

    def med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else 0

    return {
        "检验范围": "仅对不上断言的那批 required（已剔除可达的那些）",
        "断言是gold句碎片(et⊂req)": n_fragment,
        "断言与gold句无关(不同行)": n_different,
        "不同行的最长公共子串_中位": med(lens),
        "不同行的最长公共子串_最大": max(lens) if lens else 0,
        "不同行中公共子串为0的条数": zero,
        "gold句字数中位": med(req_lens),
        "断言evidence_text字数中位": med(et_lens),
        "解读": "碎片类可靠改引用粒度救；不同行类改粒度救不了，属抽取或标注问题",
    }


def section_attribution(answerable, by_chunk) -> dict:
    """不可达题二分：gold 指错行 vs 抽取真缺。"""
    st = collections.Counter()
    detail = {"gold引用指错行": [], "抽取真缺事实": []}
    for it in answerable:
        if metric_reach_span(it, by_chunk):
            continue
        if answer_supported(it, by_chunk):
            st["gold引用指错行"] += 1
            detail["gold引用指错行"].append(it["question_id"])
        else:
            st["抽取真缺事实"] += 1
            detail["抽取真缺事实"].append(it["question_id"])
    return {"计数": dict(st), "题号": detail}


def section_mining_origin(answerable, by_chunk) -> dict:
    """决定性检验：required 与断言的关系是否由 gold 挖掘路径决定。

    如果可达的类里 required **恰好等于**某条断言的 evidence_text，
    而不可达的类里 required 与断言毫无关系，那说明这个指标量的是
    「gold 挖掘时是否从断言里取证据」，而不是「系统引用得好不好」。
    这两个结论对 P3 的含义完全不同：前者要改挖掘，后者才要改回答。
    """
    stats: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for it in answerable:
        cat = it["category"]
        for rc in it.get("required_citations") or []:
            req = rc["evidence"]
            texts = [t for t in ((r.get("evidence_text") or "")
                                 for r in by_chunk.get(rc["chunk_id"]) or []) if t]
            if any(req == t for t in texts):
                stats[cat]["恰好等于断言"] += 1
            elif any(req in t for t in texts):
                stats[cat]["是断言的子串"] += 1
            elif any(t in req for t in texts):
                stats[cat]["断言是其碎片"] += 1
            else:
                stats[cat]["与断言无关"] += 1
    rows = []
    for cat in sorted(stats):
        d = stats[cat]
        total = sum(d.values())
        same = d["恰好等于断言"]
        rows.append({
            "category": cat,
            "required条数": total,
            "恰好等于断言": same,
            "等于断言占比": round(same / total, 4) if total else 0.0,
            "与断言无关": d["与断言无关"],
        })
    same_total = sum(r["恰好等于断言"] for r in rows)
    all_total = sum(r["required条数"] for r in rows)
    return {
        "按题型": rows,
        "合计": {
            "required条数": all_total,
            "恰好等于断言": same_total,
            "占比": round(same_total / all_total, 4) if all_total else 0.0,
        },
        "解读": ("引用覆盖的可达性由 gold 挖掘路径决定：从断言里取证据的类接近 100% 可达，"
                 "从块正文截整句的类接近 0%"),
    }


def section_funnel(answerable, by_chunk, triage) -> dict:
    """四道互斥闸门：成功 / 断言不含 / 块没进上下文 / 块进了没选中。"""
    st = collections.Counter()
    by_cat = collections.defaultdict(collections.Counter)
    for it in answerable:
        qid = it["question_id"]
        cat = it["category"]
        if not (it.get("required_citations") or []):
            st["无required"] += 1
            by_cat[cat]["无required"] += 1
            continue
        if not metric_reach_span(it, by_chunk):
            st["闸门1 断言不含gold句"] += 1
            by_cat[cat]["闸门1 断言不含gold句"] += 1
            continue
        ctx = set((triage.get(qid) or {}).get("gold_in_context") or [])
        req_ids = [rc["chunk_id"] for rc in it["required_citations"]]
        if not all(c in ctx for c in req_ids):
            st["闸门2 required块没进上下文"] += 1
            by_cat[cat]["闸门2 required块没进上下文"] += 1
            continue
        covered = (triage.get(qid) or {}).get("coverage_ok")
        if covered:
            st["成功覆盖"] += 1
            by_cat[cat]["成功覆盖"] += 1
        else:
            st["闸门3 块在手却没发这条引用"] += 1
            by_cat[cat]["闸门3 块在手却没发这条引用"] += 1
    return {
        "题级": dict(st),
        "按题型": {c: dict(v) for c, v in sorted(by_cat.items())},
    }


def section_line_hypothesis(answerable, by_chunk, chunks, triage) -> dict:
    """若引用改发「断言所在整行」：结构天花板与真正可兑现数。"""
    span_ok = line_ok = 0
    newly = []
    for it in answerable:
        s = metric_reach_span(it, by_chunk)
        l = metric_reach_line(it, by_chunk, chunks)
        span_ok += s
        line_ok += l
        if l and not s:
            newly.append(it)
    convertible = []
    for it in newly:
        ctx = set((triage.get(it["question_id"]) or {}).get("gold_in_context") or [])
        req_ids = [rc["chunk_id"] for rc in it["required_citations"]]
        if all(c in ctx for c in req_ids):
            convertible.append(it)
    return {
        "现状天花板": f"{span_ok}/{len(answerable)}",
        "整行假设天花板": f"{line_ok}/{len(answerable)}",
        "整行新救回": len(newly),
        "零回退": not (span_ok and line_ok < span_ok),
        "真正可兑现": len(convertible),
        "可兑现按题型": dict(collections.Counter(it["category"] for it in convertible)),
        "新救回按题型": dict(collections.Counter(it["category"] for it in newly)),
    }


def section_required_line_split(answerable, by_chunk, chunks) -> dict:
    """把「对不上的 required」按「required 那一行有没有被抽成断言」拆两群。

    为什么需要这一节
    ================
    B 在 P1-c 把 13 条 dev 侧 `needs_extra` 判成「答案在同块另一行，但那一行不含
    别名，所以不能只改 required 指向」，并据此算出「追加 required 只能救 2 条」。
    判据没错，但 `needs_extra` 不是一个同质集合：里面混了两种完全不同的东西。

    - **frag**：required 那一行上**有**断言，只是抽取器切的是行内片段。
      改引用粒度（发整行）就能通，不动 gold 一个字节。
    - **none**：required 那一行**根本没被抽成任何断言**。这是抽取缺口，
      改 gold 改不动，改引用粒度也改不动。

    三种修法（改粒度 / 追加 required / 补抽取）能救的题数与代价完全不同，
    合成一个「2/13」会把最便宜的那条路算没了。
    """
    per_cit = collections.Counter()
    by_cat = collections.defaultdict(collections.Counter)
    rows = []
    for it in answerable:
        states = []
        for rc in it.get("required_citations") or []:
            cid, req = rc["chunk_id"], rc["evidence"]
            text = chunks.get(cid, "")
            ok = any(req in (rec.get("evidence_text") or "")
                     for rec in by_chunk.get(cid) or [])
            line_hit = False
            if not ok:
                for rec in by_chunk.get(cid) or []:
                    s, e = int(rec.get("evidence_start", -1)), int(rec.get("evidence_end", -1))
                    if s < 0 or e <= s or e > len(text):
                        continue
                    a, b = _line_bounds(text, s, e)
                    if req in text[a:b]:
                        line_hit = True
                        break
            state = "ok" if ok else ("frag" if line_hit else "none")
            per_cit[state] += 1
            by_cat[it["category"]][state] += 1
            states.append({"chunk_id": cid, "state": state})
        if all(s["state"] == "ok" for s in states):
            continue
        rows.append({
            "question_id": it["question_id"],
            "category": it["category"],
            "split": it.get("split", ""),
            "states": [s["state"] for s in states],
            "worst": "none" if any(s["state"] == "none" for s in states) else "frag",
        })

    # 与 B 的 P1-c 判定表交叉：`needs_extra` 那群到底落在哪一侧
    cross = {}
    p1c = config.ROOT / "data" / "p1c_leak_audit.json"
    if p1c.exists():
        try:
            doc = json.loads(p1c.read_text(encoding="utf-8"))
            ac = doc.get("a_class_requirements") or {}
            mine = {r["question_id"]: r for r in rows}
            bucket = collections.defaultdict(list)
            for r in ac.get("rows") or []:
                qid, verdict, split = r.get("question_id"), r.get("verdict"), r.get("split")
                if verdict != "needs_extra":
                    continue
                side = (mine.get(qid) or {}).get("worst", "not_in_answerable")
                bucket[side].append({"question_id": qid, "split": split})
            cross = {
                "来源": config.rel_to_root(p1c),
                "needs_extra_按我的两群拆": {k: len(v) for k, v in bucket.items()},
                "明细": {k: sorted(v, key=lambda x: x["question_id"]) for k, v in bucket.items()},
                "结论": ("needs_extra 混了两群，不许当成一个集合算行数"
                         if len([k for k in bucket if k in ("frag", "none")]) == 2
                         else "只有一群或口径不同，需人工核对"),
            }
        except (OSError, ValueError) as exc:
            cross = {"status": "unreadable", "detail": str(exc)}

    return {
        "逐条口径": dict(per_cit),
        "按题型": {k: dict(v) for k, v in sorted(by_cat.items())},
        "未全达标的题数": len(rows),
        "按最坏一条分类": dict(collections.Counter(r["worst"] for r in rows)),
        "与 p1c 判定表交叉": cross,
        "三种修法各自能救": {
            "改引用粒度（发整行）": per_cit.get("frag", 0),
            "补抽取": per_cit.get("none", 0),
            "改 gold 标注": "见 B 的 P1-c 探针：追加 required 只救 2/13，本脚本不重复测",
        },
    }


def section_speaker_lines(by_chunk, chunks) -> dict:
    """检验「说话人标签行被抽取器系统性跳过」这个猜想（结论：不成立）。"""
    tag_total = tag_mined = other_total = other_mined = 0
    for cid, text in chunks.items():
        mined = {(rec.get("evidence_text") or "") for rec in by_chunk.get(cid) or []}
        for line in text.split("\n"):
            s = line.strip()
            if not s:
                continue
            hit = any(ev and ev in s for ev in mined)
            if SPEAKER_RE.match(s):
                tag_total += 1
                tag_mined += hit
            else:
                other_total += 1
                other_mined += hit
    return {
        "说话人标签行总数": tag_total,
        "被抽成断言": tag_mined,
        "标签行覆盖率": round(tag_mined / tag_total, 4) if tag_total else 0.0,
        "其他行总数": other_total,
        "其他行被抽": other_mined,
        "其他行覆盖率": round(other_mined / other_total, 4) if other_total else 0.0,
        "结论": "标签行覆盖率高于其他行，不是系统性跳过" if tag_total and other_total
                and tag_mined / tag_total > other_mined / other_total
                else "标签行覆盖率低于其他行",
    }


# ---------------------------------------------------------------- main
def os_name() -> str:
    """与既有产物同口径的 OS 描述串。"""
    try:
        import platform
        return platform.platform()
    except Exception:  # pragma: no cover - platform 理论上不会抛
        return ""


def _git_revision() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                             capture_output=True, text=True, timeout=10,
                             env=subproc.child_env())
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default="", help="把结构化结果写到这个路径")
    ap.add_argument("--generation", default=GENERATION)
    ap.add_argument("--skip-heavy", action="store_true",
                    help="跳过说话人标签行全语料扫描（要读 13,798 块正文）")
    args = ap.parse_args(argv)

    by_chunk, chunks = load_generation(args.generation)
    gold = load_gold()
    answerable = [it for it in gold if not it.get("should_abstain")]

    triage = {}
    tpath = config.ROOT / "data" / "a_triage_all320.jsonl"
    if tpath.exists():
        with open(tpath, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                triage[rec.get("question_id")] = rec

    gold_empty_ids = {}
    for it in gold:
        gold_empty_ids.setdefault(it["category"], None)
        gold_empty_ids[it["category"]] = not bool(it.get("gold_assertion_ids"))

    out = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_revision(),
        "python": "%d.%d.%d" % sys.version_info[:3],
        "os": os_name(),
        "command": "py -3 scripts/lead_citation_funnel.py",
        "generation": args.generation,
        "应答题数": len(answerable),
        "1_判据与天花板": section_metric_vs_loose(answerable, by_chunk),
        "2_按题型": section_by_category(answerable, by_chunk, gold_empty_ids),
        "3_粒度差检验": section_granularity_myth(answerable, by_chunk),
        "4_不可达归因": section_attribution(answerable, by_chunk),
        "4b_挖掘路径决定可达性": section_mining_origin(answerable, by_chunk),
    }
    if triage:
        out["5_四闸门漏斗"] = section_funnel(answerable, by_chunk, triage)
        out["6_整行引用假设"] = section_line_hypothesis(answerable, by_chunk, chunks, triage)
    out["6b_required行是否被抽成断言"] = section_required_line_split(answerable, by_chunk, chunks)
    if not args.skip_heavy:
        out["7_说话人标签行"] = section_speaker_lines(by_chunk, chunks)

    print("=" * 68)
    print("引用覆盖漏斗实测   generation=%s   应答题=%d" % (args.generation, len(answerable)))
    print("=" * 68)
    for key in out:
        if key.startswith(("generated_at", "generation", "应答题数")):
            continue
        print()
        print("## %s" % key)
        print(json.dumps(out[key], ensure_ascii=False, indent=1))

    if args.json:
        target = Path(args.json)
        if not target.is_absolute():
            target = config.ROOT / target
        target.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n",
                          encoding="utf-8")
        print()
        print("[i] 结构化结果写到 %s" % config.rel_to_root(target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
