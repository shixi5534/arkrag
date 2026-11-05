"""真跑 13 条 `needs_extra` 题，记录「引用实际发到了哪一行」。

## 为什么要一个入库脚本

`docs/p1c_漏标候选判定表.md` 第十节把 13 条 `needs_extra` 拆成 frag 6 / none 7 两群。
拆分的**判据**（required 那一行在块里有没有被抽成断言）只用 gold + 块正文就能算，
`scripts/p1c_leak_audit.py` 里就是这么算的，跑得很快。

但「引用实际发到了哪一行」这件事（`required_line_cited` / `carrier_lines_cited` /
`emitted_n` / `units`）**必须真跑一次管线**才知道。2026-11-05 第一次是在
`%TEMP%` 里的临时脚本算的 —— 临时文件既不入库也不可复现，等于「只有我知道」。
所以搬成这个脚本：它读同一批 gold 与世代，真跑 13 次 `ask_once`，把观测写进
`data/p1c_needs_extra_citation.json`，任何人都能用同一条命令重算。

    py -3 scripts/needs_extra_citation_probe.py            # 重算并写盘
    py -3 scripts/needs_extra_citation_probe.py --check    # 只比对，不写盘

**它不判定、不改 gold**，只记录「这一次真跑里，引用落在哪一行」。判定在
`scripts/p1c_leak_audit.py` 里。

## 为什么要分成两个字段而不是一个布尔

`required_line_cited` 与 `carrier_lines_cited` 回答的是两个不同的问题：

* `required_line_cited=True` → 引用已经指向 gold 标的那一行，覆盖判据应该成立；
* `carrier_lines_cited=[True]` → 引用指向了**另一行**，而那一行才是承载答案的。

第二种情况正是「跨行证据」的形态：答案被引用了，但引的不是 gold 那句话。
把两者合成一个「引用覆盖到了没有」的布尔，就会把这两种完全不同的失败混在一起
——本项目已经因为这种合并栽过好几次。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from arkrag import config, pipeline, provenance  # noqa: E402
import p1c_leak_audit as P1C  # noqa: E402

OUT = ROOT / "data" / "p1c_needs_extra_citation.json"
TOP_K = 20
GRAPH_DEPTH = 1


def _cited_lines(answer_obj: dict) -> list[dict]:
    """这次回答实际发出的引用：`(chunk_id, evidence, start, end)`。"""
    out = []
    for cite in answer_obj.get("citations") or []:
        out.append({
            "chunk_id": str(cite.get("chunk_id") or ""),
            "evidence": str(cite.get("evidence") or ""),
            "start": cite.get("start"),
            "end": cite.get("end"),
        })
    return out


def _matches_span(cites: list[dict], chunk_id: str, span: tuple[int, int] | None) -> bool:
    """有没有一条引用落在 `span` 这个区间上（区间相交即算）。"""
    if not span:
        return False
    for cite in cites:
        if cite["chunk_id"] != chunk_id:
            continue
        try:
            left, right = int(cite["start"]), int(cite["end"])
        except (TypeError, ValueError):
            continue
        if left < span[1] and span[0] < right:
            return True
    return False


def group_of(req_span: tuple[int, int] | None,
             assertions: list[dict]) -> tuple[str, list[str]]:
    """`frag` / `none` 的分界线：**required 那一行在块里有没有被抽成断言**。

    返回 `(group, 落在那一行内的断言 id 列表)`。

    ## 为什么不能用「答案在不在 required 行」当分界线

    `needs_extra` 的定义**就是**「答案在另一行」。拿它当分界线，每一条都会判成
    `none`，`frag` 群永远是空的——一个恒空的群看起来像「结论是这类不存在」，
    实际是判据与定义重复。2026-11-05 第一版脚本的分群就是按 `supports` 列表写的，
    跑出来 frag 6 / none 10 只对上了 6，剩下 3 条 holdout 被算错边。

    分界线要看的是**抽取覆盖**（那一行有没有被抽成断言），不是**答案位置**。
    """
    if not req_span:
        return "none", []
    left, right = req_span
    inside = []
    for a in assertions:
        try:
            a_left = int(a.get("evidence_start") or 0)
            a_right = int(a.get("evidence_end") or 0)
        except (TypeError, ValueError):
            continue
        if a_left >= left and a_right <= right:
            inside.append(str(a.get("assertion_id") or ""))
    return ("frag" if inside else "none"), inside


def tally_rows(rows: list[dict]) -> dict:
    """四格必须**互斥且覆盖全部**：`required` 行与承载行可能被同一次回答一起引用，
    所以只有「只有 required / 两行都有 / 只有承载 / 都没有」四格加起来才等于 n。

    第一版只写了三格（漏掉「两行都有」），三个数相加比总数多 1 ——
    这正是「同一件事被算了两遍」的形态，而且因为多出来的那一格恰好是「看起来最
    正常」的那种（两行都引到了），肉眼几乎发现不了。
    """
    return {
        "n": len(rows),
        "required_line_cited": sum(1 for r in rows if r["required_line_cited"]),
        "required_and_carrier": sum(1 for r in rows
                                    if r["required_line_cited"] and r["any_carrier_cited"]),
        "carrier_cited_only": sum(1 for r in rows
                                  if r["any_carrier_cited"] and not r["required_line_cited"]),
        "required_cited_only": sum(1 for r in rows
                                   if r["required_line_cited"] and not r["any_carrier_cited"]),
        "neither_cited": sum(1 for r in rows
                             if not r["required_line_cited"] and not r["any_carrier_cited"]),
    }


def build() -> dict:
    gold_v1 = json.loads(P1C.GOLD_V1.read_text("utf-8"))
    gold_v2 = json.loads(P1C.GOLD_V2.read_text("utf-8"))
    split_map = {it["question_id"]: it["split"] for it in gold_v2["items"]}
    corpus = P1C.load_corpus()

    states = P1C.required_states(gold_v1["items"], corpus)
    a_ids = P1C.a_class_ids(gold_v1["items"], corpus, states)
    a_req = P1C.a_class_requirements(gold_v1["items"], corpus, states, a_ids, split_map)
    targets = [r for r in a_req["rows"] if r["verdict"] == "needs_extra"]

    loaded = pipeline.load_generation(P1C.GENERATION)
    pipe = pipeline.build_pipeline(loaded, top_k=TOP_K, graph_depth=GRAPH_DEPTH)
    by_id = {str(it["question_id"]): it for it in gold_v1["items"]}

    rows = []
    for row in targets:
        qid = row["question_id"]
        item = by_id[qid]
        res = pipeline.ask_once(pipe, str(item.get("question") or ""))
        cites = _cited_lines(res.get("answer") or {})
        context_units = int((res.get("context") or {}).get("stats", {}).get("units") or 0)

        req = (row.get("requirements") or [{}])[0]
        chunk_id = str(req.get("chunk_id") or "")
        req_span = None
        chunk_text = str((corpus["chunks"].get(chunk_id) or {}).get("text") or "")
        evidence = str(item.get("required_citations", [{}])[0].get("evidence") or "")
        if evidence:
            found = P1C._line_span(chunk_text, evidence)
            req_span = (found[0], found[1]) if found else None

        # 分群用纯函数，测试可以直接喂合成数据验证它（见
        # `tests/test_needs_extra_citation.py::TestGroupOfIsAPureFunction`）。
        group, on_line_assertions = group_of(req_span, corpus["by_chunk"].get(chunk_id) or [])

        carrier_lines = [s.get("line") or "" for s in (req.get("supporters_other_line") or [])]
        carrier_flags = []
        for sup in req.get("supporters_other_line") or []:
            span = sup.get("span") or (sup.get("line") and None)
            if isinstance(span, list) and len(span) == 2:
                carrier_flags.append(_matches_span(cites, chunk_id, (int(span[0]), int(span[1]))))
            else:
                # 支撑信息里没有精确区间时退化成「那条断言原文有没有被引用」
                carrier_flags.append(any(
                    str(sup.get("evidence_text") or "") in c["evidence"] for c in cites))
        req_cited = _matches_span(cites, chunk_id, req_span)

        rows.append({
            "question_id": qid,
            "group": group,
            "assertions_on_required_line": on_line_assertions,
            "category": row.get("category"),
            "split": row.get("split"),
            "answer": row.get("answer"),
            "chunk_id": chunk_id,
            "required_line": (req_span and P1C._line_span(chunk_text, evidence)[2].strip()[:160]) or None,
            "required_line_cited": bool(req_cited),
            "carrier_lines": carrier_lines,
            "carrier_lines_cited": carrier_flags,
            "any_carrier_cited": bool(any(carrier_flags)) if carrier_flags else False,
            "emitted_n": len(cites),
            "units": context_units,
            "abstained": bool((res.get("answer") or {}).get("abstained")),
        })

    total = tally_rows(rows)
    cover = total["required_cited_only"]
    both = total["required_and_carrier"]
    carrier_only = total["carrier_cited_only"]
    neither = total["neither_cited"]

    def _tally(subset):
        return tally_rows(subset)

    return {
        "schema": "arkrag.p1c-needs-extra-citation/1",
        "generated_by": "scripts/needs_extra_citation_probe.py",
        "generation_id": P1C.GENERATION,
        "top_k": TOP_K,
        "graph_depth": GRAPH_DEPTH,
        "chunks_source_api": "pipeline.resolve_chunks(gen, manifest)",
        "chunks_loaded": len(corpus["chunks"]),
        "targets_n": len(rows),
        # 四个互斥格：只有 required 行 / 两行都有 / 只有承载行 / 都没有。
        # 四格之和必须等于 targets_n。
        "if_appended_would_cover_n": cover + both,
        "both_lines_cited_n": both,
        "carrier_cited_but_required_not_n": carrier_only,
        "required_cited_but_carrier_not_n": cover,
        "neither_cited_n": neither,
        # 分口径的两个拆分：**按划分**（dev 可改侧 / holdout 只记录）与
        # **按群**（frag 行已被抽成断言 / none 行没被抽）。两个拆分都要给，
        # 因为「13 条 dev 里 frag 6 条」这种说法只有同时给出两个维度才对得上账。
        "by_split": {s: _tally([r for r in rows if r["split"] == s])
                     for s in ("dev", "holdout")},
        "by_group": {g: _tally([r for r in rows if r["group"] == g])
                     for g in ("frag", "none")},
        "by_split_and_group": {
            f"{s}/{g}": _tally([r for r in rows if r["split"] == s and r["group"] == g])
            for s in ("dev", "holdout") for g in ("frag", "none")
        },
        "detail": rows,
    }


def annotate(payload: dict) -> dict:
    """补上七道可追溯字段。取不到的留空，不填假值。"""
    # `inputs` 必须是 `inputs_block()` 的产物（每条带 sha16），不能直接传一串路径：
    # 只传路径时每条只有 path 字段、没有 sha16，`judge_inputs` 会判 unknown 且
    # 报错信息看不出是「没算哈希」还是「没登记输入」——这正是我在
    # `tests/test_provenance.py` 里补的那条形状洞。
    block = provenance.collect(
        generation_id=payload["generation_id"],
        inputs=provenance.inputs_block([
            "data/goldset/gold-v1.json",
            "data/goldset/gold-v2.json",
            "data/a_triage_all320.jsonl",
        ]),
    )
    payload = dict(payload)
    payload.update(block)
    return payload


def _observations(payload: dict) -> dict:
    """只留「观测到的引用落点」，去掉每次运行都合法地变化的溯源字段。

    `generated_at` / `git_commit` / `code_sha256_16` 三者每跑一次都变，把它们
    算进 `--check` 的比较里，这道命令就永远是红的——而它要回答的是
    「引用落点变了吗」，不是「文件字节一样吗」。所以比较前剥掉。
    """
    volatile = set(provenance.REQUIRED_FIELDS) | {"schema", "generated_by"}
    return {k: v for k, v in payload.items() if k not in volatile}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    check = "--check" in args
    payload = annotate(build())
    body = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if check:
        old_raw = OUT.read_text("utf-8") if OUT.exists() else ""
        try:
            old_payload = json.loads(old_raw) if old_raw else {}
        except json.JSONDecodeError:
            old_payload = {}
        if old_payload and _observations(old_payload) == _observations(payload):
            print("引用观测与当前数据一致：", payload["targets_n"], "条")
            return 0
        print("引用观测与当前数据不一致（重跑不带 --check 即可刷新）")
        return 1
    OUT.write_text(body, encoding="utf-8")
    print("写盘：", config.rel_to_root(OUT))
    print(f"目标 {payload['targets_n']} 条 | required 行被引用 "
          f"{payload['if_appended_would_cover_n']} | 只有承载行被引用 "
          f"{payload['carrier_cited_but_required_not_n']} | 只有 required 行被引用 "
          f"{payload['required_cited_but_carrier_not_n']} | 两边都没被引用 "
          f"{payload['neither_cited_n']}")
    for name, block in payload["by_split"].items():
        print(f"  {name}: n={block['n']} required行被引用={block['required_line_cited']} "
              f"仅承载行={block['carrier_cited_only']} 都没={block['neither_cited']}")
    for name, block in payload["by_group"].items():
        print(f"  {name}: n={block['n']} required行被引用={block['required_line_cited']} "
              f"仅承载行={block['carrier_cited_only']} 都没={block['neither_cited']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
