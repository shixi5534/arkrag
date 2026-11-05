"""P3 第三轮靶子上界实测：题型路由可行性、cross_chunk 章节包、multihop 中间实体、闸门2 三层归因。

为什么写这个脚本
----------------
P3 第一轮（整行引用）与第二轮（标题通路）都只动 single_chunk，其余七类增益为 0。
在决定第三轮往哪打之前，必须先用**在线可得的信息**（问句 + 上下文 + 图，不用 gold）
把三件事的上界量出来，否则又会拿一个「结构性不可达」的方向去做一周：

1. 「按题型路由」在实现上可不可行——问句骨架与 category 是不是一一映射；
2. cross_chunk 要的「章节代号」能不能从块元数据机械拼出来，拼出来对不对；
3. multihop 要的「共同中间实体」能不能从邻接表机械求出来，求出来对不对；
4. 闸门2 的 203 题到底该归谁——检索层（gold 挤不进前 20）、容量层（进了前 20 被预算截断）、
   还是多块题只进了一部分。这三者对应完全不同的工作。

本节所有数字都可以用一条命令复算：
    py -3 scripts/lead_p3_targets.py --json data/lead_p3_targets.json

口径纪律
--------
- 只读：不改 gold、不改检索参数、不碰 src/。
- 块正文与图一律从**世代自带**文件读，不许用全局 data/arkrag/chunks.jsonl
  （两文件不是同一批块：代内 13,798 / 语料根 13,786）。
- 「上界」是从答案侧反推的分数上限，不是实现承诺；实现还要过单变量、引用定位、拒答三道门。
"""

from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import pathlib
import platform
import re
import statistics
import subprocess
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from arkrag import config, pipeline, retrieve, subproc  # noqa: E402

GENERATION = "g-20261103T132152Z-06a741"
GOLD_SET = "data/goldset/gold-v2.json"
QUOTE_RE = re.compile(r"「([^」]+)」")


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
            env=subproc.child_env(),
        )
        return (out.stdout or "").strip() or "unknown"
    except Exception as exc:  # noqa: BLE001
        return f"unknown({exc})"


# ---------------------------------------------------------------- 基础装载


def load_retriever():
    loaded = pipeline.load_generation(GENERATION)
    pipe = pipeline.build_pipeline(loaded, top_k=20, graph_depth=1)
    return pipe, pipe["retriever"]


def load_dev(path: str = GOLD_SET) -> list[dict[str, Any]]:
    data = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return [
        it
        for it in data["items"]
        if it.get("split") == "dev" and not it.get("should_abstain")
    ]


def entity_index(retriever) -> tuple[dict[str, dict], dict[str, set[str]]]:
    ents: dict[str, dict] = {}
    for e in retriever.graph.get("entities") or []:
        eid = str(e.get("entity_id") or e.get("id") or "")
        if eid:
            ents[eid] = e
    name2ids: dict[str, set[str]] = collections.defaultdict(set)
    for eid, e in ents.items():
        name = str(e.get("name") or "")
        if name:
            name2ids[name].add(eid)
    return ents, name2ids


def neighbours(retriever, eid: str) -> dict[str, float]:
    """邻接表是 {entity_id: {"out": [ {neighbor_id, weight, ...} ], "in": [...]}}。

    返回 {邻居 id: 最高权重}（同一邻居多条边取最大值，权重只用于排序）。
    """
    node = retriever.adjacency.get(eid) or {}
    best: dict[str, float] = {}
    for direction in ("out", "in"):
        for nb in node.get(direction) or []:
            nid = str(nb.get("neighbor_id") or "")
            if not nid:
                continue
            w = float(nb.get("weight") or 0.0)
            if w > best.get(nid, -1.0):
                best[nid] = w
    return best


# ---------------------------------------------------------------- 1 题型路由


def skeleton(question: str) -> str:
    q = QUOTE_RE.sub("「Q」", question)
    q = re.sub(r"《[^》]*》", "《T》", q)
    return q


def section_router(gold: dict[str, Any]) -> dict[str, Any]:
    sk: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for it in gold["items"]:
        sk[skeleton(it["question"])][it["category"]] += 1
    per_cat: dict[str, int] = collections.Counter()
    for it in gold["items"]:
        per_cat[it["category"]] += 1
    cross = {k: dict(v) for k, v in sk.items() if len(v) > 1}
    per_cat_sk: dict[str, int] = collections.Counter(
        it_cat for it_cat in {c: None for c in per_cat}
    )
    cat_skeletons: dict[str, set[str]] = collections.defaultdict(set)
    for it in gold["items"]:
        cat_skeletons[it["category"]].add(skeleton(it["question"]))
    examples = {}
    for it in gold["items"]:
        examples.setdefault(it["category"], skeleton(it["question"]))
    return {
        "问题总数": len(gold["items"]),
        "骨架总数": len(sk),
        "跨类骨架数": len(cross),
        "跨类骨架": cross,
        "每类骨架数": {c: len(s) for c, s in sorted(cat_skeletons.items())},
        "每类题数": dict(sorted(per_cat.items())),
        "每类骨架示例": examples,
        "判据": "跨类骨架数 == 0 ⇒ 纯规则路由可行，不需要模型",
    }


# ---------------------------------------------------------------- 2 cross_chunk


def chapter_of(chunk_id: str, by_chunk: dict[str, dict]) -> str:
    """章节代号必须读块记录的 `chapter` 字段，不能从 chunk_id 路径里截。

    路径第二段只是目录名（`activities/act33side/...` → `act33side`），
    而 main 线真实章节是 `main_ep14` / `main_st17` / `memory_whitw2` 这种，
    路径里根本没有。用路径截会得到 `main` / `memory`，13/64 精确命中；
    读字段则 38/38 精确命中——同一批题，差 25 题全在 main/memory 线上。
    """
    rec = by_chunk.get(chunk_id) or {}
    return str(rec.get("chapter") or "")


def section_cross_chunk(
    retriever, items: list[dict[str, Any]], by_chunk: dict[str, dict]
) -> dict[str, Any]:
    ents, name2ids = entity_index(retriever)

    def chapters_from_path(chunk_id: str) -> str:
        parts = [p for p in chunk_id.split("#")[0].split("/") if p]
        return parts[1] if len(parts) >= 2 else ""

    by_gold_chunk = 0
    by_path = 0
    by_assertion_endpoint = 0
    by_assertion_set = 0
    hit_gold_chunk = 0
    rows = []
    for it in items:
        if it.get("category") != "cross_chunk":
            continue
        gold_chaps = [chapter_of(c, by_chunk) for c in it["gold_chunk_ids"]]
        gold_chapters = "、".join(dict.fromkeys(c for c in gold_chaps if c))
        ok_by_gold = gold_chapters == it["answer"]
        if ok_by_gold:
            by_gold_chunk += 1
        path_chapters = "、".join(
            dict.fromkeys(chapters_from_path(c) for c in it["gold_chunk_ids"])
        )
        if path_chapters == it["answer"]:
            by_path += 1
        names = QUOTE_RE.findall(it["question"])
        name = names[0] if names else ""
        ids = name2ids.get(name, set())
        ch_assert: set[str] = set()
        for cid, ch in by_chunk.items():
            recs = ch.get("assertions") or []
            if not recs:
                continue
            if any(
                str(rec.get("subject_id") or "") in ids
                or str(rec.get("object_id") or "") in ids
                for rec in recs
            ):
                chap = chapter_of(cid, by_chunk)
                if chap:
                    ch_assert.add(chap)
        predicted = "、".join(sorted(ch_assert))
        ok_assert = predicted == it["answer"]
        if ok_assert:
            by_assertion_endpoint += 1
        gold_set = {c for c in gold_chaps if c}
        ok_set = gold_set == ch_assert
        if ok_set:
            by_assertion_set += 1
        if gold_set & ch_assert:
            hit_gold_chunk += 1
        rows.append(
            {
                "question_id": it["question_id"],
                "answer": it["answer"],
                "gold_chapters": gold_chapters,
                "path_chapters": path_chapters,
                "predicted_chapters_sorted": predicted,
                "predicted_chapters_set_equal": sorted(gold_set) == sorted(ch_assert),
                "ok_by_gold_chunk_field": ok_by_gold,
                "ok_by_chunk_id_path": path_chapters == it["answer"],
                "ok_by_assertion_ordered": ok_assert,
                "ok_by_assertion_set": ok_set,
                "n_predicted": len(ch_assert),
            }
        )
    both = sum(
        1 for r in rows if r["ok_by_gold_chunk_field"] and r["ok_by_assertion_set"]
    )
    return {
        "题数": len(rows),
        "按gold块chapter字段_精确命中": by_gold_chunk,
        "按gold块chunk_id路径截_精确命中": by_path,
        "按断言端点拼_同序精确命中": by_assertion_endpoint,
        "按断言端点拼_集合相等": by_assertion_set,
        "至少命中一个gold章节": hit_gold_chunk,
        "字段法与断言法皆成立": both,
        "候选章节数中位": statistics.median([r["n_predicted"] for r in rows]) if rows else None,
        "候选章节数最大": max((r["n_predicted"] for r in rows), default=None),
        "逐题": rows,
        "判据": "chapter 必须读块记录字段而不是从 chunk_id 路径截（路径里 main 线只剩 main，"
        "真实章节是 main_ep14/main_st17/memory_whitw2）。「按 gold 块拼」证明答案可从元数据"
        "机械生成；「按断言端点拼」是不用 gold 时的上界，且只能比集合、不能比顺序。",
    }


# ---------------------------------------------------------------- 3 multihop


def section_multihop(
    retriever,
    items: list[dict[str, Any]],
    required_in_context: dict[str, bool],
    answer_hits_gold: dict[str, bool],
) -> dict[str, Any]:
    ents, name2ids = entity_index(retriever)
    parsed = 0
    contains = 0
    uniq = 0
    uniq_ok = 0
    top1 = 0
    top1_ok = 0
    sizes = []
    rows = []
    for it in items:
        if it.get("category") != "multihop":
            continue
        names = QUOTE_RE.findall(it["question"])
        if len(names) < 2:
            continue
        A = name2ids.get(names[0], set())
        B = name2ids.get(names[1], set())
        if not A or not B:
            continue
        parsed += 1
        score: dict[str, float] = {}
        for a in A:
            for m, w1 in neighbours(retriever, a).items():
                for b, w2 in neighbours(retriever, m).items():
                    if b in B:
                        score[m] = max(score.get(m, 0.0), w1 + w2)
        pred = {}
        for m in score:
            nm = str(ents.get(m, {}).get("name") or "")
            if nm:
                pred[nm] = max(pred.get(nm, 0.0), score[m])
        sizes.append(len(pred))
        qid = it["question_id"]
        if it["answer"] in pred:
            contains += 1
        if len(pred) == 1:
            uniq += 1
            if it["answer"] in pred:
                uniq_ok += 1
        if pred:
            best = max(pred.items(), key=lambda kv: (kv[1], kv[0]))[0]
            top1 += 1
            if best == it["answer"]:
                top1_ok += 1
        rows.append(
            {
                "question_id": qid,
                "answer": it["answer"],
                "n_candidates": len(pred),
                "contains_answer": it["answer"] in pred,
                "top1": best if pred else None,
                "required_in_context": required_in_context.get(qid),
                "answer_hits_gold_now": answer_hits_gold.get(qid),
            }
        )
    return {
        "题数": len(rows),
        "可解析双端点": parsed,
        "候选集含gold答案": contains,
        "候选唯一": uniq,
        "候选唯且正确": uniq_ok,
        "加权top1可产出": top1,
        "加权top1正确": top1_ok,
        "候选数中位": statistics.median(sizes) if sizes else None,
        "候选数最大": max(sizes) if sizes else None,
        "required_in_context为真": sum(1 for r in rows if r["required_in_context"]),
        "当前answer_hits_gold": sum(1 for r in rows if r["answer_hits_gold_now"]),
        "判据": "中间实体可从邻接表机械求；但 multihop 的 required 块全部不在上下文，"
        "所以这一路只能抬答案质量，抬不动引用覆盖。",
        "逐题": rows,
    }


# ---------------------------------------------------------------- 4 闸门2 归因


def section_gate2(
    pipe,
    dev: list[dict[str, Any]],
    ctx_rows: dict[str, dict],
) -> dict[str, Any]:
    buckets: collections.Counter = collections.Counter()
    by_cat: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    examples: dict[str, list[str]] = collections.defaultdict(list)
    for it in dev:
        qid = it["question_id"]
        if ctx_rows.get(qid, {}).get("required_in_context"):
            continue
        out = pipeline.ask_once(pipe, it["question"])
        hits = [h["chunk_id"] for h in out["search"]["hits"]]
        units = {u["chunk_id"] for u in out["context"]["units"]}
        trunc = set(out["context"]["stats"].get("truncated_chunks") or [])
        req = {c["chunk_id"] for c in (it.get("required_citations") or [])}
        in_units = req & units
        if not (req & set(hits)):
            k = "检索层_gold不在前20"
        elif in_units == req:
            k = "口径层_全在units却判False"
        elif in_units:
            k = "部分层_多块题只进一部分"
        elif req & trunc:
            k = "容量层_前20但全被预算截断"
        else:
            k = "其它"
        buckets[k] += 1
        by_cat[it["category"]][k] += 1
        if len(examples[k]) < 8:
            examples[k].append(qid)
    return {
        "闸门2题数": sum(buckets.values()),
        "三层归因": dict(buckets.most_common()),
        "按类": {c: dict(v) for c, v in sorted(by_cat.items())},
        "样例": dict(examples),
        "判据": "三层对应三种完全不同的活：检索层=排序/召回；容量层=预算或填充；"
        "部分层=多块题的第二块名次。混成一个「闸门2 203」会导致修错地方。",
    }


# ---------------------------------------------------------------- 5 按类表


def section_by_category(
    dev: list[dict[str, Any]],
    ctx_rows: dict[str, dict],
) -> dict[str, Any]:
    agg: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for it in dev:
        r = ctx_rows.get(it["question_id"]) or {}
        a = agg[it["category"]]
        a["n"] += 1
        a["required_in_context"] += 1 if r.get("required_in_context") else 0
        a["covered"] += 1 if r.get("covered") else 0
        a["answer_hits_gold"] += 1 if r.get("answer_hits_gold") else 0
        if not r.get("covered") and r.get("answer_hits_gold"):
            a["答案对但未覆盖"] += 1
            if r.get("required_in_context"):
                a["其中required在上下文"] += 1
    cols = ["n", "required_in_context", "covered", "answer_hits_gold", "答案对但未覆盖", "其中required在上下文"]
    total = collections.Counter()
    for a in agg.values():
        total.update(a)
    return {
        "列": cols,
        "表": {c: {k: int(a[k]) for k in cols} for c, a in sorted(agg.items())},
        "合计": {k: int(total[k]) for k in cols},
        "判据": "「答案对但未覆盖」才是回答层真缺口；其中 required 已在上下文的才可能靠改回答救回。",
    }


# ---------------------------------------------------------------- main


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        # 不 reconfigure 的话，本脚本打印的中文在某些控制台下会抛 UnicodeEncodeError，
        # 进程以 1 退出，而这个 1 与「分析失败」长得一模一样。
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", dest="json_path", default="")
    ap.add_argument("--gold", default=GOLD_SET)
    args = ap.parse_args()

    gold = json.loads((ROOT / args.gold).read_text(encoding="utf-8"))
    dev = load_dev(args.gold)
    pipe, retriever = load_retriever()

    ctx_path = ROOT / "data/a_title_path_round.json"
    ctx_rows: dict[str, dict] = {}
    if ctx_path.exists():
        payload = json.loads(ctx_path.read_text(encoding="utf-8"))
        for row in payload["per_question"]["cand"]:
            ctx_rows[row["question_id"]] = row
    else:
        print("警告: 缺少 data/a_title_path_round.json，第 4/5 节改用现场真跑", file=sys.stderr)
        for it in dev:
            out = pipeline.ask_once(pipe, it["question"])
            units = {u["chunk_id"] for u in out["context"]["units"]}
            req = {c["chunk_id"] for c in (it.get("required_citations") or [])}
            ctx_rows[it["question_id"]] = {
                "question_id": it["question_id"],
                "category": it["category"],
                "required_in_context": bool(req) and req <= units,
                "covered": False,
                "answer_hits_gold": False,
            }

    by_chunk: dict[str, dict] = {}
    gen_chunks = config.DATA_DIR / "generations" / GENERATION / "chunks.jsonl"
    if not gen_chunks.exists():
        raise SystemExit(f"世代自带块文件不存在: {gen_chunks}")
    for line in gen_chunks.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        cid = str(rec.get("chunk_id") or "")
        if cid:
            by_chunk[cid] = rec
    # 断言挂到块上
    gen_assert = config.DATA_DIR / "generations" / GENERATION / "graph" / "assertions.jsonl"
    if gen_assert.exists():
        for line in gen_assert.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = str(rec.get("chunk_id") or "")
            if cid in by_chunk:
                by_chunk[cid].setdefault("assertions", []).append(rec)

    ric = {qid: bool(r.get("required_in_context")) for qid, r in ctx_rows.items()}
    ahg = {qid: bool(r.get("answer_hits_gold")) for qid, r in ctx_rows.items()}

    report: dict[str, Any] = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_revision(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "command": "py -3 scripts/lead_p3_targets.py --json data/lead_p3_targets.json",
        "generation": GENERATION,
        "gold_set": args.gold,
        "1_题型路由": section_router(gold),
        "2_cross_chunk章节包": section_cross_chunk(retriever, dev, by_chunk),
        "3_multihop中间实体": section_multihop(retriever, dev, ric, ahg),
        "4_闸门2归因": section_gate2(pipe, dev, ctx_rows),
        "5_按类表": section_by_category(dev, ctx_rows),
    }

    print("=" * 78)
    print("1 题型路由")
    r = report["1_题型路由"]
    print(f"  题 {r['问题总数']} / 骨架 {r['骨架总数']} / 跨类骨架 {r['跨类骨架数']}")
    print(f"  每类骨架数 {r['每类骨架数']}")
    print("=" * 78)
    print("2 cross_chunk 章节包")
    c = report["2_cross_chunk章节包"]
    print(
        f"  题 {c['题数']} / 按gold块chapter字段 {c['按gold块chapter字段_精确命中']} / "
        f"按路径截 {c['按gold块chunk_id路径截_精确命中']} / "
        f"按断言端点(同序) {c['按断言端点拼_同序精确命中']} / "
        f"按断言端点(集合) {c['按断言端点拼_集合相等']} / "
        f"至少命中一个gold章节 {c['至少命中一个gold章节']}"
    )
    print(f"  候选章节数 中位 {c['候选章节数中位']} 最大 {c['候选章节数最大']}")
    print("=" * 78)
    print("3 multihop 共同中间实体")
    m = report["3_multihop中间实体"]
    print(
        f"  题 {m['题数']} / 可解析 {m['可解析双端点']} / 候选含金 {m['候选集含gold答案']} / "
        f"候选唯一 {m['候选唯一']}(对 {m['候选唯且正确']}) / 加权top1 正确 {m['加权top1正确']}/{m['加权top1可产出']}"
    )
    print(f"  required_in_context 为真 {m['required_in_context为真']} / 当前 answer_hits_gold {m['当前answer_hits_gold']}")
    print("=" * 78)
    print("4 闸门2 归因")
    g2 = report["4_闸门2归因"]
    print(f"  题 {g2['闸门2题数']}")
    for k, v in g2["三层归因"].items():
        print(f"  {k:28}{v:>4}")
    print("=" * 78)
    print("5 按类表")
    t = report["5_按类表"]
    print("  " + "类".ljust(16) + "".join(col.rjust(10) for col in t["列"][1:]))
    for cat, row in t["表"].items():
        print("  " + cat.ljust(16) + "".join(str(row[col]).rjust(10) for col in t["列"][1:]))
    print("  " + "合计".ljust(16) + "".join(str(t["合计"][col]).rjust(10) for col in t["列"][1:]))
    print("=" * 78)

    if args.json_path:
        out = ROOT / args.json_path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"已写出 {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
