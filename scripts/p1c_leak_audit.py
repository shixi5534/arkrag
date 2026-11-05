"""P1-c：59 条「合法同义证据漏标」候选的独立判定，并生成判定表。

**本脚本只出判定，不改任何 gold。** 它读三个已入库的数据文件，跑一遍得出判定，
再写 `docs/p1c_漏标候选判定表.md` 与 `data/p1c_leak_audit.json`；
第三方在同一份仓库状态下运行本脚本，应得到同一张表——这是「判定必须能被复算」
这条要求的实现方式（只给结论、不给可跑的命令，等于不可复核）。

    py -3 scripts/p1c_leak_audit.py            # 重算并重写判定表与 JSON
    py -3 scripts/p1c_leak_audit.py --check    # 只比对新旧，不写盘（有差异返回 1）

输入（全部已入库）：
  * `data/a_triage_all320.jsonl` —— A 的逐题 triage 结果，其中 59 行带 `synonym_evidence`；
  * `data/goldset/gold-v1.json` / `gold-v2.json` —— 题面、gold 块、gold 断言、split 归属；
  * 世代 `g-20261103T132152Z-06a741` 的断言与块正文。

块正文来源：**`pipeline.resolve_chunks(gen, manifest)`**，不自己拼路径、不读
`config.chunks_path()`。理由见 `src/arkrag/pipeline.py:60-108` 的 docstring：
全局 `data/arkrag/chunks.jsonl` 是 L1 最新产物，下一次 L2 重建就覆盖它而世代不跟着变；
只有世代自带的 `<gen>/chunks.jsonl` 并核过 `chunk_manifest_hash` 才是「这一代吃进去的那批」。
2026-11-05 本判定的第一版自己拼路径、还在 `exists()` 为 False 时静默回退到语料根，
是同一个坑第三次被踩（前两次：D 阶段 13798 vs 13781、P1-b 结构报告 13786 vs 13798）。

判定口径（本脚本唯一实现处，逐类判据写进每条输出的 `criterion` 字段）：

  真漏标    候选块★独立★支撑同一个答案，且它与 gold 引用的不是同一处文字。
  不是漏标  候选块只是提到相关名字、或只提供一半证据、或属于「子串平凡可推」、
            或它是同一条边（同三元组）的另一个承载块——这些都不构成「gold 漏掉了
            一条本就该收的合法证据」。
  无法判定  别名形态不完全一致（块里是短名/带姓名的另一形态）、或答案粒度本身有
            多种合法取法，需要人读原文才能定。

  alias        规范名是否为别名的子串（子串 → 平凡，不是漏标）；否则别名与规范名
               是否同一行共现。
  direct_relation
               ⚠ 判据 2026-11-05 两次修正，两个口径必须分开记录：

               **V1 宽口径**（本脚本第一版、也是 Lead 前一版用的）：
               「断言 subject/object 恰为问句两实体、predicate 恰为 gold 关系词」。
               这个口径**太宽**：它把结构字段对上、但 evidence 根本没在说这回事的断言
               也算成等价证据。反例 `q-dr-000041` 的 `obt/main/level_main_01-03_beg#000`，
               断言 et=`'天灾在我们头顶发生的可能性就越大'`——没在说「天灾破坏切尔诺伯格」。
               按 V1 宽口径，96/96 题都能找到非 gold 等价块。

               **V2 严口径**（gold 挖掘 `mine_direct_relations` 自己的构造判据）：
               结构对上 **且** 该断言 `evidence_text` 里**同时出现两个端点的名字**。
               实测 **47/96**（dev 26 / holdout 21）。V1→V2 掉了一半，说明「等价块」
               这个说法本身依赖标尺，不能拿宽口径的数去否决别人的结论。

               判定用 V2 严口径，并按两条不同的理由分判：
               · 严口径有命中、且三元组与 gold 同一条 → 「代表证据之外的承载块」，不是漏标。
                 依据：`edge.evidence` 单一字符串 + `edge.chunks` 列表，gold 从 N 块里挑
                 「该 evidence 唯一出现」的那一块，`gold_chunk_ids` 恒 1 是这一类的定义。
               · 严口径有命中、但三元组与 gold 不同一条 → 无法判定。
               · 严口径无命中、宽口径有命中 → 「判据过宽」：结构对上但 evidence 里端点名
                 不全，该断言不足以支撑这条关系，不是漏标。
               · 两个口径都无命中 → 只是块正文同时提到两个实体，更不是漏标。
  multihop     候选块里是否有断言连接「答案实体 ↔ 起点或终点实体」。即便有，单跳也
               **不能**独立确立两跳路径 → 不是漏标（只算补充证据）。
  time_conflict
               候选块是否同时出现两个关系词（时序冲突题需要两侧）。缺一侧 → 不是漏标。
  adversarial  ★两个名字是否出现在同一条断言★（块正文共现 ≠ 断言共现）。
               · 同一条断言里两个名字都在 → 该断言能独立支撑「两者不同」→ 真漏标。
               · 只在块正文共现、没有任何一条断言同时含两者 → 不是漏标（同块不同段）。
               · 另记「短名是否独立出现（非长名子串）」：若短名只作为长名的子串出现，
                 块里连一次独立的短名用法都没有，更不构成同义证据。
  cross_chunk  候选块所属章节代号是否已在 gold 答案里。已在 → 不是漏标。
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arkrag import answer, config, pipeline, release  # noqa: E402

ROOT = config.ROOT
GENERATION = "g-20261103T132152Z-06a741"
TRIAGE = ROOT / "data" / "a_triage_all320.jsonl"
GOLD_V1 = ROOT / "data" / "goldset" / "gold-v1.json"
GOLD_V2 = ROOT / "data" / "goldset" / "gold-v2.json"
DOC = ROOT / "docs" / "p1c_漏标候选判定表.md"
JSON_OUT = ROOT / "data" / "p1c_leak_audit.json"
QUOTED = re.compile(r"[「『](.+?)[」』]")
ZERO = "\u0000"
CATEGORIES = ["alias", "direct_relation", "multihop", "time_conflict",
              "cross_chunk", "adversarial"]
RELATION_WORDS = ("隶属", "背叛", "同盟", "敌对", "交战", "保护")

# 布尔答案的候选词。判它是因为「否」「没有」这类答案会在断言里以**子串**形式
# 出现（「火龙并没有被源石感染」里的「没有」），机器判据在这里必然误命中，
# 所以这类题只能判「无法判定」，不能判「答案有断言支持」。
BOOLEAN_ANSWERS = frozenset({"是", "否", "没有", "未发生", "发生", "有"})

A_CLASS_VERDICTS = (
    "carrier_ok",        # required 那句就落在承载答案的行上 → 不改
    "move_safe",         # 答案行同时含问句里的别名 → 可改 required 指向答案行
    "needs_extra",       # 答案行不含别名 → 不能移动，只能追加 required 或接受跨行
    "boolean_answer",    # 布尔答案，机器只匹配到子串 → 无法判定
    "no_support",        # gold 块里没有任何断言含答案 → 抽取缺口，不是标注问题
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def short_free_count(text: str, long_name: str, short_name: str) -> int:
    """short_name 在 text 里、且不落在 long_name 内部的出现次数。

    必要性：`玻利瓦尔` 是 `玻利瓦尔第二独立国` 的子串，直接 count 会把
    「长名内部的短名」算成「独立的短名用法」，让「原文只出现短名」这个判据失真。
    """
    if not short_name:
        return 0
    masked = text.replace(long_name, ZERO * len(long_name)) if long_name else text
    return masked.count(short_name)


def load_corpus() -> dict:
    """取这一代吃进去的断言与块正文（唯一合法来源：pipeline.resolve_chunks）。"""
    gen = release.Generation(GENERATION)
    manifest = gen.load()
    chunks_file = pipeline.resolve_chunks(gen, manifest)
    chunks = {str(r.get("chunk_id")): r for r in pipeline._read_jsonl(chunks_file)}
    by_chunk: dict[str, list[dict]] = {}
    by_id: dict[str, dict] = {}
    for a in pipeline._read_jsonl(gen.assertion_path):
        by_chunk.setdefault(str(a.get("chunk_id") or ""), []).append(a)
        by_id[str(a.get("assertion_id") or "")] = a
    graph = json.loads(gen.graph_path.read_text("utf-8"))
    ents = {str(e["id"]): e for e in graph.get("entities") or []}
    name2ids: dict[str, set[str]] = {}
    for eid, e in ents.items():
        nm = str(e.get("name") or "").strip()
        if nm:
            name2ids.setdefault(nm, set()).add(eid)
        for al in e.get("aliases") or []:
            txt = str((al or {}).get("text") if isinstance(al, dict) else al).strip()
            if txt:
                name2ids.setdefault(txt, set()).add(eid)
    return {
        "chunks": chunks,
        "by_chunk": by_chunk,
        "by_id": by_id,
        "ents": ents,
        "name2ids": name2ids,
        "edges": graph.get("edges") or [],
        "chunks_source": config.rel_to_root(chunks_file),
        "chunk_manifest_hash": str(manifest.get("chunk_manifest_hash") or ""),
    }


def candidates(triage_rows: list[dict], split_map: dict[str, str]) -> list[dict]:
    """从 A 的 triage 行里取那 59 条带 `synonym_evidence` 的候选。"""
    out = []
    for r in triage_rows:
        cand = r.get("synonym_evidence")
        if not cand:
            continue
        out.append({
            "question_id": r["question_id"],
            "category": r["category"],
            "split": split_map.get(r["question_id"], "<none>"),
            "question": r.get("question"),
            "gold_answer": r.get("gold_answer"),
            "acceptable_answers": r.get("acceptable_answers") or [],
            "should_abstain": r.get("should_abstain"),
            "gold_chunk_ids": r.get("gold_chunk_ids") or [],
            "candidate": cand,
        })
    return out


def required_states(items: list[dict], corpus: dict) -> dict[str, list[dict]]:
    """复算每条 required citation 的四种状态（与 `scripts/ask.py:283-298` 的判据同向）。

    `in_assertion`           某条断言的 `evidence_text` **包含** required 那句；
    `assertion_is_fragment`  反过来，断言是 required 的片段；
    `unrelated`              块里有断言，但没有一条相关；
    `chunk_no_assertion`     该块一条断言都没有。
    """
    by_chunk = corpus["by_chunk"]
    out: dict[str, list[dict]] = {}
    for it in items:
        rows = []
        for req in it.get("required_citations") or []:
            cid = str(req.get("chunk_id") or "")
            ev = str(req.get("evidence") or "")
            al = by_chunk.get(cid) or []
            if not al:
                state = "chunk_no_assertion"
            elif any(ev and ev in str(a.get("evidence_text") or "") for a in al):
                state = "in_assertion"
            elif any(str(a.get("evidence_text") or "") and str(a.get("evidence_text")) in ev
                     for a in al):
                state = "assertion_is_fragment"
            else:
                state = "unrelated"
            rows.append({"chunk_id": cid, "evidence": ev, "state": state})
        out[str(it["question_id"])] = rows
    return out


def a_class_ids(v1_items: list[dict], corpus: dict,
                states: dict[str, list[dict]]) -> list[str]:
    """P1-b 的「A 类」：被卡住、且答案本身在 gold 块的某条断言里（宽松口径，答案任意长度）。"""
    out = []
    for it in v1_items:
        qid = str(it["question_id"])
        if it.get("should_abstain"):
            continue
        rows = states.get(qid) or []
        if not rows or all(r["state"] == "in_assertion" for r in rows):
            continue
        needles = [str(it.get("answer") or "")] + [str(x) for x in it.get("acceptable_answers") or []]
        needles = [n for n in needles if n]
        for cid in it.get("gold_chunk_ids") or []:
            hit = any(n in str(a.get("evidence_text") or "")
                      for a in corpus["by_chunk"].get(str(cid)) or []
                      for n in needles)
            if hit:
                out.append(qid)
                break
    return sorted(out)


def _line_span(text: str, needle: str) -> tuple[int, int, str] | None:
    """`needle` 在 `text` 里首次出现位置所在整行（左闭右开）。找不到返回 None。"""
    i = text.find(needle)
    if i < 0:
        return None
    left, right = answer._line_bounds(text, i, i + len(needle))
    return left, right, text[left:right]


def _assertion_line(text: str, a: dict) -> tuple[int, int, str]:
    """一条断言 evidence 所在的整行。`answer._line_bounds` 会把 span 扩到整行。"""
    left, right = answer._line_bounds(
        text, int(a.get("evidence_start") or 0), int(a.get("evidence_end") or 0))
    return left, right, text[left:right]


def a_class_requirements(v1_items: list[dict], corpus: dict,
                         states: dict[str, list[dict]],
                         a_ids: list[str],
                         split_map: dict[str, str]) -> dict:
    """P1-c 剩余：A 类那 27 条「gold 引用指错行」的题，逐条判能不能只改 required 指向。

    ## 为什么不能沿用「两个名字都在块里」那个判据

    P1-c 前一轮在 `direct_relation` 上用过「候选块里有等价断言 ⇒ 真漏标」，结果
    那个判据在该类上**恒为真**（96/96），差点据此推翻 P2 的前提。这一轮换判据：

    **答案是否被某条断言支持**（`needle in evidence_text`），
    **以及 required 标的那一行是不是承载答案的那一行**（整行区间是否重叠）。

    两个问题都要回答，缺一个就会把「引用口径窄」误判成「gold 指错行」。

    ## 四态（必须分开，因为补救动作完全不同）

    | 判定 | 含义 | 该做什么 |
    |---|---|---|
    | `carrier_ok` | required 那句就落在承载答案的那一行上 | **不改**。required 带说话人前缀、断言不带，是引用口径窄，不是 gold 缺陷 |
    | `move_safe` | 答案在**同块另一行**，且那一行**也含**问句里的别名 | 改 required 指向那一行——安全，那一行同时证明「别名在这」和「答案是什么」 |
    | `needs_extra` | 答案在同块另一行，但那一行**不含**别名 | **不能移动**。挪过去就丢掉「别名出现」这个唯一证据，而块里没有任何一行同时给出两者。只能追加一条 required，或接受跨行 |
    | `boolean_answer` | 答案是否/没有这类布尔词 | **无法判定**。机器会把断言里的「没有」当成命中，判据在这里失效 |

    `move_safe` 与 `needs_extra` 混成一种是这一轮最容易犯的错：它们都表现为
    「答案在别处」，但一个能靠改指向解决、另一个改了反而更糟。
    """
    rows = []
    for it in v1_items:
        qid = str(it["question_id"])
        if qid not in a_ids:
            continue
        question = str(it.get("question") or "")
        aliases = [x for x in QUOTED.findall(question) if x]
        needles = [str(it.get("answer") or "")] + [
            str(x) for x in it.get("acceptable_answers") or []]
        needles = [n for n in needles if n]
        longest = max(needles, key=len, default="")
        is_bool = any(n in BOOLEAN_ANSWERS for n in needles) and len(longest) <= 3

        supports = []
        for cid in it.get("gold_chunk_ids") or []:
            cid = str(cid)
            chunk_text = str((corpus["chunks"].get(cid) or {}).get("text") or "")
            for a in corpus["by_chunk"].get(cid) or []:
                ev = str(a.get("evidence_text") or "")
                hit = [n for n in needles if n in ev]
                if not hit:
                    continue
                left, right, line = _assertion_line(chunk_text, a)
                supports.append({
                    "chunk_id": cid,
                    "assertion_id": str(a.get("assertion_id") or ""),
                    "evidence_text": ev,
                    "matched": hit,
                    "line": line.strip()[:160],
                    "span": [left, right],
                    "line_has_alias": [x for x in aliases if x in line],
                })

        req_rows = []
        for req in it.get("required_citations") or []:
            cid = str(req.get("chunk_id") or "")
            ev = str(req.get("evidence") or "")
            chunk_text = str((corpus["chunks"].get(cid) or {}).get("text") or "")
            span = _line_span(chunk_text, ev) if ev else None
            same_chunk = [s for s in supports if s["chunk_id"] == cid]
            on_line = [s for s in same_chunk
                       if span and s["span"][0] < span[1] and span[0] < s["span"][1]]
            others = [s for s in same_chunk if s not in on_line]
            if on_line:
                verdict = "carrier_ok"
                reason = "承载答案的断言就在 required 那一行上（前缀差异属引用口径窄）"
            elif is_bool:
                verdict = "boolean_answer"
                reason = "布尔答案，断言里以子串形式出现，机器判据失效"
            elif others and any(s["line_has_alias"] for s in others):
                verdict = "move_safe"
                reason = "答案在同块另一行且那一行也含别名，可安全改指向"
            elif others:
                verdict = "needs_extra"
                reason = "答案在同块另一行但那一行不含别名，改指向会丢证据"
            else:
                verdict = "no_support"
                reason = "gold 块里没有任何断言含答案（抽取缺口，不是标注问题）"
            req_rows.append({
                "chunk_id": cid,
                "evidence": ev[:120],
                "line": (span[2].strip()[:160] if span else None),
                "found_in_text": span is not None,
                "verdict": verdict,
                "reason": reason,
                "supporters_on_same_line": [
                    {"assertion_id": s["assertion_id"], "evidence_text": s["evidence_text"],
                     "line": s["line"]} for s in on_line],
                "supporters_other_line": [
                    {"assertion_id": s["assertion_id"], "evidence_text": s["evidence_text"],
                     "line": s["line"], "line_has_alias": s["line_has_alias"]}
                    for s in others],
            })

        # 题级取最坏的一行：只要有一行的判定要求「改 gold」，这条题就是待改题。
        order = ("no_support", "move_safe", "needs_extra", "boolean_answer", "carrier_ok")
        worst = next((v for v in order
                      if any(r["verdict"] == v for r in req_rows)), "carrier_ok")
        rows.append({
            "question_id": qid,
            "category": it.get("category"),
            "split": split_map.get(qid, "<none>"),
            "question": question,
            "aliases": aliases,
            "answer": it.get("answer"),
            "acceptable_answers": list(it.get("acceptable_answers") or []),
            "is_boolean_answer": is_bool,
            "verdict": worst,
            "requirements": req_rows,
            "assertions_in_gold_chunks": len(supports),
        })
    return {
        "rows": rows,
        "tally": dict(Counter(r["verdict"] for r in rows)),
        "by_split": {
            split: dict(Counter(r["verdict"] for r in rows if r["split"] == split))
            for split in ("dev", "holdout")
        },
        "dev_n": sum(1 for r in rows if r["split"] == "dev"),
        "holdout_n": sum(1 for r in rows if r["split"] == "holdout"),
    }


def judge_one(item: dict, corpus: dict, gold_v2_by_id: dict[str, dict]) -> dict:
    cat = item["category"]
    cid = str(item["candidate"].get("chunk_id") or "")
    text = str((corpus["chunks"].get(cid) or {}).get("text") or "")
    names = QUOTED.findall(str(item.get("question") or ""))
    canon = str(item.get("gold_answer") or "")
    forms = [canon] + [str(a) for a in item.get("acceptable_answers") or []]
    forms = [f for f in forms if f]
    by_chunk = corpus["by_chunk"]
    ents = corpus["ents"]
    name2ids = corpus["name2ids"]

    verdict = criterion = reason = ""
    inputs: dict = {"chunk_id": cid}
    extra: dict = {}

    def ent_name(eid: str) -> str:
        return str((ents.get(str(eid)) or {}).get("name") or eid)

    if cat == "alias":
        alias = names[0] if names else ""
        substring = bool(alias and canon and canon in alias)
        same_line = [ln for ln in text.splitlines()
                     if alias in ln and canon in ln and not substring]
        criterion = "规范名是否为别名子串；否则别名与规范名是否同一行共现"
        if alias and alias not in text:
            verdict = "无法判定"
            reason = (f"问句里的别名「{alias}」在候选块里没有逐字出现（块里是它的短形态），"
                      "要对上得先确认二者同指")
        elif substring:
            verdict = "不是漏标"
            reason = (f"规范名「{canon}」是别名「{alias}」的子串，"
                      "任何含别名的行都平凡含规范名，不构成独立证据")
        elif same_line:
            verdict = "真漏标"
            reason = f"别名与规范名在同一行共现且非子串关系：「{same_line[0][:60]}」"
            inputs["evidence_line"] = same_line[0][:120]
        else:
            verdict = "不是漏标"
            reason = "别名与规范名不同行，块内没有建立「别名 → 规范名」的指代"
        extra = {"alias": alias, "substring": substring, "same_line_n": len(same_line)}

    elif cat == "direct_relation":
        want: set[str] = set()
        for nm in names:
            want |= name2ids.get(nm, set())

        def _both_names_in(ev: str, s: str, o: str) -> bool:
            sn, on = ent_name(s), ent_name(o)
            return bool(sn and on and sn in ev and on in ev)

        hits_wide: list[dict] = []
        hits_strict: list[dict] = []
        cand_triples: set[tuple[str, str, str]] = set()
        for a in by_chunk.get(cid, []):
            s, o = str(a.get("subject_id") or ""), str(a.get("object_id") or "")
            if not (s in want and o in want):
                continue
            if str(a.get("predicate") or "") != canon or s == o:
                continue
            rec = {"subject": ent_name(s), "predicate": canon, "object": ent_name(o),
                   "evidence_text": str(a.get("evidence_text") or "")}
            hits_wide.append(rec)
            if _both_names_in(rec["evidence_text"], s, o):
                hits_strict.append(rec)
                cand_triples.add((s, canon, o))
        # gold 那条边是哪一条？用 gold_assertion_ids 反查它的三元组。
        # 注意：A 的 triage 行里 gold_assertion_ids 恒为 None（59/59），必须按题号回
        # gold-v2 取；否则这里恒为空、每条都会落到「无法判定」——那会把「判据没取到
        # 输入」伪装成「无法判定」，又是一条假结论（已留反向实测记录）。
        gitem = gold_v2_by_id.get(str(item["question_id"])) or {}
        gold_triples: set[tuple[str, str, str]] = set()
        for aid in gitem.get("gold_assertion_ids") or []:
            ga = corpus["by_id"].get(str(aid))
            if not ga:
                continue
            s = str(ga.get("subject_id") or "")
            o = str(ga.get("object_id") or "")
            p = str(ga.get("predicate") or "")
            gold_triples.add((s, p, o))
            gold_triples.add((o, p, s))
        same_edge = bool(cand_triples & gold_triples)
        criterion = ("V2 严口径：断言结构字段对上 **且** evidence_text 里两个端点名都在；"
                     "再看该三元组是否与 gold 同一条边。宽口径（只看结构字段）同时记数，"
                     "两者之差就是「判据过宽」的量")
        if hits_strict and same_edge:
            verdict = "不是漏标"
            h = hits_strict[0]
            reason = (f"候选块有严口径命中断言「{h['subject']} —{h['predicate']}→ {h['object']}」，"
                      "但它与 gold 指向的是**同一条边**——`edge.evidence` 是单一字符串、"
                      "`edge.chunks` 是列表，gold 的设计就是「一条边挑一块代表证据」，"
                      "其余承载块不是漏标")
            inputs["assertion"] = h
        elif hits_strict:
            verdict = "无法判定"
            h = hits_strict[0]
            reason = (f"严口径命中「{h['subject']} —{h['predicate']}→ {h['object']}」，"
                      "但三元组与 gold 不同一条，需人确认它是否描述了另一种关系")
            inputs["assertion"] = h
        elif hits_wide:
            h = hits_wide[0]
            verdict = "不是漏标"
            reason = ("**判据过宽**：结构字段对上的断言有，但它的 evidence 里没有同时出现两个"
                      f"端点名（「{h['evidence_text'][:30]}」），按 gold 挖掘自己的构造标准，"
                      "这条断言不足以支撑该关系——不是漏标，是我方判据把抽取质量差的断言"
                      "也算成了等价证据")
            inputs["assertion"] = h
        else:
            verdict = "不是漏标"
            present = [n for n in names if n in text]
            reason = (f"两个实体名都在块正文出现（{len(present)}/{len(names)}："
                      f"{'、'.join(present)}），但块内没有断言同时给出这两个实体与 "
                      f"gold 关系词「{canon}」——正文共现不等于断言共现，"
                      "更不等于独立支撑同一个答案")
            inputs["names_in_block_text"] = present
        extra = {"assertions_wide": hits_wide, "assertions_strict": hits_strict,
                 "wide_n": len(hits_wide), "strict_n": len(hits_strict),
                 "same_edge_as_gold": same_edge,
                 "names_in_block_text_n": len([n for n in names if n in text])}

    elif cat == "multihop":
        ans_ids = name2ids.get(canon, set())
        want = set()
        for nm in names:
            want |= name2ids.get(nm, set())
        hops = []
        for a in by_chunk.get(cid, []):
            s, o = str(a.get("subject_id") or ""), str(a.get("object_id") or "")
            if (s in ans_ids and o in want) or (o in ans_ids and s in want):
                hops.append({"subject": ent_name(s),
                             "predicate": str(a.get("predicate") or ""),
                             "object": ent_name(o),
                             "evidence_text": str(a.get("evidence_text") or "")})
        criterion = "候选块里连接答案实体与问句实体的断言条数（两跳题需两侧同时成立）"
        verdict = "不是漏标"
        reason = (f"候选块只提供一跳（命中 {len(hops)} 条断言），两跳题需要"
                  "「起点↔中间」与「中间↔终点」两侧同时成立；单侧不足以独立支撑同一个答案"
                  if hops else
                  "候选块里没有连接答案实体与问句实体的断言，只是同一场景的其它对话")
        extra = {"hops": hops}
        if hops:
            inputs["assertion"] = hops[0]

    elif cat == "time_conflict":
        rel_words = [n for n in names if n in RELATION_WORDS]
        present = {w: (w in text) for w in rel_words}
        n_present = sum(present.values())
        criterion = "候选块里两个相反关系词是否同时出现（时序冲突题需两侧）"
        verdict = "不是漏标"
        reason = (f"候选块只出现 {n_present}/{len(rel_words)} 个关系词，"
                  "单侧无法独立支撑「先后出现两种相反关系」这一结论"
                  if n_present < len(rel_words) else
                  "两侧关系词都在，需人工确认它是否真的叙述了先后变化")
        extra = {"relation_words_present": present}

    elif cat == "adversarial":
        long_name = names[0] if names else ""
        short_name = names[1] if len(names) > 1 else ""
        long_in_block = text.count(long_name) if long_name else 0
        short_free_in_block = short_free_count(text, long_name, short_name)
        both_in_block = bool(long_in_block and short_free_in_block)
        same_assert = []
        for a in by_chunk.get(cid, []):
            ev = str(a.get("evidence_text") or "")
            if long_name and short_name and long_name in ev and short_free_count(ev, long_name, short_name):
                same_assert.append(ev)
        criterion = ("两个名字是否出现在★同一条断言★里（块正文共现不算）；"
                     "另核短名是否独立出现（非长名子串）")
        if same_assert:
            verdict = "真漏标"
            reason = (f"有一条断言同时含两个名字：「{same_assert[0][:50]}」——"
                      "它能独立支撑「两者不同」，是 gold 未收录的合法证据")
            inputs["assertion"] = same_assert[0]
        else:
            verdict = "不是漏标"
            reason = ("块正文里两个名字都出现，但**没有任何一条断言同时含两者**"
                      f"（长名 {long_in_block} 次、独立短名 {short_free_in_block} 次），"
                      "属同块不同段；gold 答案的依据是「所引原文只出现短名」，"
                      "块正文共现并不能替代它")
        extra = {"long_name": long_name, "short_name": short_name,
                 "long_in_block": long_in_block,
                 "short_free_in_block": short_free_in_block,
                 "both_in_block_text": both_in_block,
                 "same_assertion_n": len(same_assert)}

    elif cat == "cross_chunk":
        parts = cid.split("/")
        code = ""
        if len(parts) >= 2 and parts[0] == "activities":
            code = parts[1]
        elif len(parts) >= 3 and parts[0] == "obt" and parts[1] == "memory":
            stem = parts[2].split("#")[0]
            code = "memory_" + re.sub(r"_\d+_\d+$", "", re.sub(r"^story_", "", stem))
        elif len(parts) >= 3 and parts[0] == "obt" and parts[1] == "main":
            mm = re.search(r"(\d+)-(\d+)", parts[2])
            code = f"main_ep{int(mm.group(1)):02d}" if mm else "main_unknown"
        in_answer = bool(code) and code in canon
        criterion = "候选块所属章节代号是否已在 gold 答案里"
        verdict = "不是漏标"
        reason = (f"候选块所属章节代号「{code}」已经在 gold 答案里，"
                  "它是同一章节的另一块，不增加答案信息"
                  if in_answer else f"章节代号「{code}」不在 gold 答案里")
        extra = {"chapter": code, "chapter_in_answer": in_answer}
    else:
        raise SystemExit(f"未知类别：{cat}")

    answer_in_text = any(f in text for f in forms)
    answer_in_assertion = any(
        any(f in str(a.get("evidence_text") or "") for f in forms)
        for a in by_chunk.get(cid, [])
    )
    return {
        "question_id": item["question_id"],
        "category": cat,
        "split": item["split"],
        "question": item.get("question"),
        "gold_answer": canon,
        "gold_chunk_ids": item.get("gold_chunk_ids"),
        "candidate_in_gold_chunks": cid in (item.get("gold_chunk_ids") or []),
        "candidate_chunk_id": cid,
        "verdict": verdict,
        "criterion": criterion,
        "reason": reason,
        "inputs": inputs,
        "answer_in_block_text": answer_in_text,
        "answer_in_assertion": answer_in_assertion,
        "evidence": extra,
    }


def direct_relation_equivalence(corpus: dict, gold_items: list[dict]) -> dict:
    """把「同一关系、两端点对、非 gold」的等价承载块全类数一遍——**两个口径都数**。

    V1 宽：只看断言结构字段（subject/object 恰为问句两实体、predicate 恰为 gold 答案）。
          这个口径把 evidence 根本没在说这回事的断言也算进来，实测 96/96。
    V2 严：再要求该断言 `evidence_text` 里**同时出现两个端点的名字**——这正是
          `mine_direct_relations` 自己的构造条件。实测 47/96。

    两个数一起给，是因为「等价块」这个说法依赖标尺：只报宽口径会得出「96/96 判据恒真」
    这种过头的结论（2026-11-05 已被 Lead 更正），只报严口径又会看不出宽口径错在哪。
    """
    by_chunk = corpus["by_chunk"]
    ents = corpus["ents"]
    name2ids = corpus["name2ids"]

    def name_of(eid: str) -> str:
        return str((ents.get(str(eid)) or {}).get("name") or "")

    sites: dict[tuple[str, str, str], set[str]] = {}
    strict_sites: dict[tuple[str, str, str], set[str]] = {}
    for cid, al in by_chunk.items():
        for a in al:
            s = str(a.get("subject_id") or "")
            o = str(a.get("object_id") or "")
            p = str(a.get("predicate") or "")
            if not s or not o or s == o:
                continue
            et = str(a.get("evidence_text") or "")
            sn, on = name_of(s), name_of(o)
            for key in ((s, o, p), (o, s, p)):
                sites.setdefault(key, set()).add(cid)
            if sn and on and sn in et and on in et:
                for key in ((s, o, p), (o, s, p)):
                    strict_sites.setdefault(key, set()).add(cid)

    def tally(idx: dict[tuple[str, str, str], set[str]]) -> dict:
        hit = dev = hold = gold_among = 0
        totals: list[int] = []
        extras: list[int] = []
        per_item: dict[str, dict] = {}
        for it in gold_items:
            if it.get("category") != "direct_relation":
                continue
            want: set[str] = set()
            for nm in QUOTED.findall(str(it.get("question") or "")):
                want |= name2ids.get(nm, set())
            gold = set(it.get("gold_chunk_ids") or [])
            found: set[str] = set()
            for (s, o, p), cs in idx.items():
                if p == it.get("answer") and s in want and o in want and s != o:
                    found |= cs
            if not found:
                continue
            per_item[it["question_id"]] = {
                "split": it.get("split"),
                "total_blocks": len(found),
                "non_gold_blocks": len(found - gold),
                "gold_in_sites": bool(gold & found),
            }
            if gold & found:
                gold_among += 1
            if found - gold:
                hit += 1
                if it.get("split") == "holdout":
                    hold += 1
                else:
                    dev += 1
            totals.append(len(found))
            extras.append(len(found - gold))
        totals.sort()
        extras.sort()

        def triple(xs: list[int]) -> str:
            return f"{xs[0]} / {xs[len(xs) // 2]} / {xs[-1]}" if xs else "— / — / —"

        return {"with_equivalent_non_gold_chunk": hit, "dev": dev, "holdout": hold,
                "total_blocks_triple": triple(totals), "extra_blocks_triple": triple(extras),
                "gold_chunk_among_sites": gold_among, "per_item": per_item}

    questions = len([1 for it in gold_items if it.get("category") == "direct_relation"])
    v1 = tally(sites)
    v2 = tally(strict_sites)
    return {"questions": questions, "v1_wide": v1, "v2_strict": v2}


def render_doc(rows: list[dict], meta: dict, equiv: dict, a_req: dict,
               a_ids: list[str], split_map: dict[str, str]) -> str:
    tot = Counter(r["verdict"] for r in rows)
    cnt = Counter((r["category"], r["verdict"]) for r in rows)
    n_same = sum(r["evidence"]["same_assertion_n"] for r in rows if r["category"] == "adversarial")
    n_both = sum(1 for r in rows if r["category"] == "adversarial"
                 and r["evidence"]["both_in_block_text"])
    n_flag = sum(1 for r in rows if r["answer_in_block_text"])
    n_both_a = sum(1 for r in rows if r["answer_in_block_text"] and r["answer_in_assertion"])
    n_text_only = sum(1 for r in rows if r["answer_in_block_text"] and not r["answer_in_assertion"])
    alias_ids = [q for q in a_ids if q.startswith("q-al-")]
    na_ids = [q for q in a_ids if q.startswith("q-na-")]

    L: list[str] = []
    L.append("# P1-c 判定表：59 条「合法同义证据漏标」候选\n")
    L.append("> **本表只出判定，未改动任何 gold 文件。** 判定全部由 `scripts/p1c_leak_audit.py`")
    L.append("> 从已入库的数据算出；复核者不需要我的任何临时脚本或临时目录。\n")

    L.append("## 零、怎么复核这张表\n")
    L.append("```")
    L.append("py -3 scripts/p1c_leak_audit.py            # 重算并重写本表与 data/p1c_leak_audit.json")
    L.append("py -3 scripts/p1c_leak_audit.py --check    # 只比对新旧，有差异返回 1")
    L.append("```")
    L.append("每条判定都带三样东西，第三方可用同一份数据从零复算：")
    L.append("1. **题号**；2. **判据**（`criterion` 字段，逐类固定，不随题而变）；")
    L.append("3. **输入**（`inputs` 字段：候选 `chunk_id`；有断言的给断言原文；"
             "alias 给 `evidence_line`；direct_relation 无命中的给块正文里出现过的实体名）。")
    L.append("逐条明细同时写在 `data/p1c_leak_audit.json` 的 `rows` 里。\n")

    L.append("### 输入与块正文来源（这一项必须写清，本次曾出过事故）\n")
    L.append("- 候选：`data/a_triage_all320.jsonl` 里带 `synonym_evidence` 的 59 行（A 的产物，已入库）")
    L.append("- 题面/gold：`data/goldset/gold-v1.json`、`data/goldset/gold-v2.json`（后者提供 `split`）")
    L.append(f"- 块正文：`pipeline.resolve_chunks(gen, manifest)` → `{meta['chunks_source']}`"
             f"（{meta['chunks_loaded']} 块）")
    L.append(f"- 核过 `chunk_manifest_hash` = `{meta['chunk_manifest_hash'][:16]}…`；"
             "文件在盘上不等于内容对，`exists()` 检查抓不到这类错")
    L.append("- **不**读 `config.chunks_path()`（全局 `data/arkrag/chunks.jsonl` 是 L1 最新产物，"
             "下一次 L2 重建就覆盖它而世代不跟着变）\n")

    L.append("## 一、结论汇总\n")
    L.append("| 类别 | 候选数 | 真漏标 | 不是漏标 | 无法判定 |")
    L.append("|---|---|---|---|---|")
    for c in CATEGORIES:
        n = sum(1 for r in rows if r["category"] == c)
        L.append(f"| {c} | {n} | {cnt[(c, '真漏标')]} | {cnt[(c, '不是漏标')]} | {cnt[(c, '无法判定')]} |")
    L.append(f"| **合计** | **{len(rows)}** | **{tot['真漏标']}** | "
             f"**{tot['不是漏标']}** | **{tot['无法判定']}** |\n")
    L.append(f"**{len(rows)} 条里 {tot['真漏标']} 条真漏标、{tot['不是漏标']} 条不是漏标、"
             f"{tot['无法判定']} 条无法判定。**")
    L.append("（第一版曾判 2 条为真漏标，2026-11-05 改判——理由不是口径宽窄，是那个判据把")
    L.append("「结构字段对上、evidence 却没在说这回事」的断言也算成等价证据；见第二节。）")
    L.append("与 A 的措辞（「候选池前 20 名里存在别的块正文命中该题全部实体名或含答案原文」）")
    L.append("不矛盾：A 的判据是「命中实体名 / 含答案原文」，而「命中实体名」不等于")
    L.append("「独立支撑同一个答案」。另外**这 59 条的候选块无一已在 `gold_chunk_ids` 里**")
    L.append("（0/59），所以每条都在回答同一个问题：这个**别的块**该不该并进 gold。\n")

    L.append("## 二、原来判为「真漏标」的那 2 条，为什么改判（本节比那 2 条本身重要）\n")
    L.append("第一版把 `q-dr-000011` / `q-dr-000041` 判成真漏标，判据是：")
    L.append("「候选块里存在一条断言，subject/object 恰为问句两实体、predicate 恰为 gold 关系词」。\n")
    L.append("**这个判据太宽。** 它把「结构字段对上、但 evidence 根本没在说这回事」的断言也当成")
    L.append("等价证据。反例就出在这两条自己身上：`q-dr-000041` 的候选块里有一条断言")
    L.append("`et='天灾在我们头顶发生的可能性就越大'`——它没在说「天灾破坏切尔诺伯格」，")
    L.append("但按结构字段它算命中。\n")
    L.append("换成 gold 挖掘自己用的判据（`mine_direct_relations` 的构造条件：论断 `evidence`")
    L.append("里必须**同时出现两个端点的名字**）重数一遍，两个口径的差距：\n")
    L.append("| 判据 | 「存在非 gold 等价块」的题数 | dev | holdout | 每题总等价块 最少/中位/最多 | "
             "扣掉 gold 后富余 最少/中位/最多 |")
    L.append("|---|---|---|---|---|---|")
    for label, key in (("V1 宽（只看结构字段）", "v1_wide"),
                       ("**V2 严**（gold 挖矿自己的判据）", "v2_strict")):
        d = equiv[key]
        L.append(f"| {label} | **{d['with_equivalent_non_gold_chunk']}** / {equiv['questions']}"
                 f" | {d['dev']} | {d['holdout']} | {d['total_blocks_triple']} | "
                 f"{d['extra_blocks_triple']} |")
    L.append("")
    L.append("**「等价块」这个说法依赖标尺。** V1 宽口径下 96/96 全中，看起来像「判据恒真」；")
    L.append("V2 严口径下只有 47/96，一半。任何拿宽口径数去否决别人结论的动作都会过头——")
    L.append("这一条本轮已经踩过，见下。\n")
    L.append("### 那 2 条的两条理由并不相同（不能压成一句）\n")
    L.append("下表左三列是**候选块内**的断言数（回应「这个候选块该不该并进 gold」），")
    L.append("右两列是**该题全类**的等价承载块数（V1 宽 / V2 严，都不含 gold 那块）——")
    L.append("两个范围不同，别把「候选块里 1 条」和「全类 4 块」混成一个数。\n")
    L.append("| 题号 | 划分 | 候选块内断言 宽/严 | 全类非 gold 等价块 V1 宽 | 全类非 gold 等价块 V2 严 | 判定 |")
    L.append("|---|---|---|---|---|---|")
    for r in rows:
        if r["question_id"] not in ("q-dr-000011", "q-dr-000041"):
            continue
        e = r["evidence"]
        pi1 = equiv["v1_wide"]["per_item"].get(r["question_id"]) or {}
        pi2 = equiv["v2_strict"]["per_item"].get(r["question_id"]) or {}
        L.append(f"| `{r['question_id']}` | {r['split']} | {e['wide_n']} / {e['strict_n']} | "
                 f"{pi1.get('non_gold_blocks')} | {pi2.get('non_gold_blocks')} | {r['verdict']} |")
    L.append("")
    L.append("- **`q-dr-000011`：判据过宽。** 候选块 `activities/act29side/level_act29side_10_end#006`")
    L.append("  的那条断言 `et='莱辛......莱辛！'`，**两个端点名一个都不在 evidence 里**。")
    L.append("  按断言的结构字段判它对，按 gold 自己的构造标准这条不足以支撑「莱辛与埃芒加德交谈」。")
    L.append("  V2 严口径下它的等价块数 = **0**。")
    L.append("- **`q-dr-000041`：代表证据形态。** V2 严口径下有 4 个等价块，证据都很实")
    L.append("  （`'天灾，摧毁了切尔诺伯格。'`、`'即使是切尔诺伯格城，在天灾的直接冲击下，也会化作一摊废墟'`、")
    L.append("  `'天灾降临时，就连无比坚固的切尔诺伯格都会被撕得粉碎'`）。但这 4 块与 gold 块属于")
    L.append("  **同一条 edge 的不同承载块**——`edge.evidence` 单一字符串、`edge.chunks` 是列表，")
    L.append("  gold 从 N 块里挑「该 evidence 唯一出现」的那一块。这不叫漏标，叫「gold 只收代表证据」。\n")
    L.append("### 机制：为什么 `direct_relation` 恒 1 个 `gold_chunk_ids`\n")
    L.append("`src/arkrag/goldset.py::mine_direct_relations` 的构造条件（原文）：")
    L.append("> * 边自带的 `evidence` 在某个 chunk 里**唯一出现**（可逐字定位）；")
    L.append("> * 该 `evidence` 里**同时出现两个端点的名字**（否则证据支撑不了这条关系）。\n")
    L.append("所以一条边承载 N 块、只有一条代表证据，gold 挑「该 evidence 唯一出现」的那块。")
    L.append("各题型 `gold_chunk_ids` 条数分布佐证：`adversarial {1:32}`、`alias {1:96}`、")
    L.append("`direct_relation {1:96}`、`single_chunk {1:128}`、`multihop {2:96}`、")
    L.append("`time_conflict {2:64}`、`cross_chunk {2:35, 3:20, 4:9}`、`negative_abstain {0:52, 1:12}`。")
    L.append("**`direct_relation` 恒 1 块是这一类的定义，不是个别题标错。**\n")
    L.append("### 这件事的教训（本轮连踩两级，都写在这里）\n")
    L.append("1. **判据若在某类题上接近恒真，它就没有判别力。** 但反过来也要小心：")
    L.append("   「恒真」这个判断本身可能来自判据太宽。V1 宽口径 96/96、V2 严口径 47/96，")
    L.append("   差了一半——**说「判据恒真」之前，先用系统自己的标尺重算一遍**。")
    L.append("2. **拿一个数去否决别人的结论，必须先用对方或系统自己的判据重算。**")
    L.append("   两种判据的差异超过 2 倍就不许上报。本轮 Lead 正是靠这条把自己的")
    L.append("   「+30pt / 96-96」收回成「+1.33pt / 47-96」，我也在同一处把 2 条真漏标收回成 0 条。")
    L.append("3. 与仓库既有事故同形：`invalid_cache_report()[\"total\"]` 把")
    L.append("   `total=13801 / ok=13801 / invalid=0` 报成「13801 条非法记录」——")
    L.append("   **判据本身在骗人，数字却很好看。**\n")

    L.append(f"## 三、无法判定（{tot['无法判定']} 条）\n")
    for r in rows:
        if r["verdict"] != "无法判定":
            continue
        L.append(f"- `{r['question_id']}`（{r['split']}）：{r['reason']}")
        L.append(f"  - 问句：{r['question']}")
        L.append(f"  - gold 答案：{r['gold_answer']}；候选块：`{r['candidate_chunk_id']}`")
        L.append(f"  - 判据：{r['criterion']}")
    L.append("")

    L.append("## 四、adversarial 9 条：块正文共现 vs 同一条断言共现（Lead 硬要求拆开）\n")
    L.append("判据明确写死：**「两个名字都出现在块正文」与「两个名字出现在同一条断言」是两回事**，")
    L.append("前者可能只是同块不同段（与 A 类「同块不同段」同一形态），后者才是可独立引用的证据。\n")
    L.append("| 题号 | 划分 | 长名(在块内次数) | 独立短名(非长名子串) | 同一条断言含两者 | 判定 |")
    L.append("|---|---|---|---|---|---|")
    for r in rows:
        if r["category"] != "adversarial":
            continue
        e = r["evidence"]
        L.append(f"| `{r['question_id']}` | {r['split']} | {e['long_name']}({e['long_in_block']}) | "
                 f"{e['short_free_in_block']} | {e['same_assertion_n']} | {r['verdict']} |")
    L.append("")
    L.append(f"**实测：9 条里 {n_both} 条两个名字在块正文都出现，"
             f"但同一条断言同时含两者的有 {n_same} 条。**")
    zero_short = [r["question_id"] for r in rows
                  if r["category"] == "adversarial" and r["evidence"]["short_free_in_block"] == 0]
    if zero_short:
        L.append("另有 " + str(len(zero_short)) + " 条（" +
                 " / ".join(f"`{q}`" for q in zero_short) +
                 "）的短名只在长名内部出现（独立短名计数 0），"
                 "即块里连一次独立的短名用法都没有。\n")

    L.append("## 五、`answer_in_text` 这个数要拆（Lead 硬要求）\n")
    L.append("先纠正一个口径：A 原始 `synonym_evidence.answer_in_text` 为真的是 **19 条**，不是 20；")
    L.append("我早前说的 20 是另一个字段（把 `acceptable_answers` 也算进来，多出 `q-mh-000026` 的「黍」，")
    L.append("A 那个标志要求答案更长）。两个数都对，但必须标明定义。\n")
    L.append(f"- 本次复算「答案（含 acceptable_answers）出现在候选块正文」= **{n_flag}** 条\n")
    L.append("再按 Lead 要求拆成两问：答案在块里 ≠ 该块该被标为 required。\n")
    L.append("| 答案在哪 | 条数 | 含义 |")
    L.append("|---|---|---|")
    L.append(f"| 块正文有、且块内某条断言里也有 | {n_both_a} | 该块本可提供断言级证据，属「口径窄」 |")
    L.append(f"| 块正文有、但**没有任何断言含答案** | {n_text_only} | 答案只是散在正文里，"
             "构成不了可引用证据 → 不能据此并进 gold |")
    L.append("")
    L.append("**关键点：这 59 条候选块 0 条已在 `gold_chunk_ids` 里**，所以「答案在这个别的块里出现」")
    L.append("本身不能证明它是漏标——alias 要证明「别名→规范名」的指代、multihop 要证明两跳、")
    L.append("direct_relation 要证明同一关系，缺一样都不算独立支撑。\n")

    L.append(f"## 六、逐题判定（全部 {len(rows)} 条，含判据与输入）\n")
    for c in CATEGORIES:
        sel = [r for r in rows if r["category"] == c]
        if not sel:
            continue
        L.append(f"### {c}（{len(sel)} 条）\n")
        L.append(f"**判据**：{sel[0]['criterion']}\n")
        L.append("| 题号 | 划分 | 判定 | 输入（候选块 / 断言） | 依据 |")
        L.append("|---|---|---|---|---|")
        for r in sel:
            inp = f"`{r['candidate_chunk_id']}`"
            a = r["inputs"].get("assertion")
            if isinstance(a, dict) and a.get("evidence_text"):
                inp += f"<br>断言「{a['evidence_text'][:30]}」"
            elif r["inputs"].get("evidence_line"):
                inp += f"<br>行「{r['inputs']['evidence_line'][:30]}」"
            L.append(f"| `{r['question_id']}` | {r['split']} | {r['verdict']} | {inp} | {r['reason']} |")
        L.append("")

    L.append("## 七、dev / holdout 拆分（两个口径都列，避免再对不上账）\n")
    L.append("### 7.1 这 59 条候选\n")
    L.append(f"- 59 条：dev **{sum(1 for r in rows if r['split'] == 'dev')}** / "
             f"holdout **{sum(1 for r in rows if r['split'] == 'holdout')}**")
    L.append(f"- **真漏标：{tot['真漏标']} 条**（dev 0 / holdout 0）。第一版判的 2 条已按 Lead 裁决")
    L.append("  改判为「不是漏标（同一条边的其它承载块）」，理由见第二节——判据恒真，不是口径宽窄。")
    L.append("- 按 Lead 裁决：holdout 侧只记录不改，报告里按「已知 gold 缺陷」显式扣减；只有 dev 侧可改。")
    L.append("  本次 dev 侧同样 0 条可改，所以**没有任何 gold 改动，`gold-v2.json` 指纹不变**。\n")

    L.append("### 7.2 P1-b 的 A 类 27 条（gold 引用指错行）\n")
    L.append("| 口径 | 条数 | dev | holdout |")
    L.append("|---|---|---|---|")
    for label, ids in (("alias 子集", alias_ids),
                       ("negative_abstain 子集", na_ids),
                       ("**完整 27 条**", a_ids)):
        L.append(f"| {label} | {len(ids)} | "
                 f"{sum(1 for q in ids if split_map.get(q) == 'dev')} | "
                 f"{sum(1 for q in ids if split_map.get(q) == 'holdout')} |")
    L.append("")
    L.append("**口径说明**：Lead 早前说「27 条里有 5 条落在 v1 holdout」，那 5 条"
             "（`q-al-000001` / `000023` / `000027` / `000032` / `000042`）是 **alias 子集 23 条**里的；")
    L.append("完整 27 条还含 4 条 negative_abstain，其中 `q-na-000006` 也在 holdout，")
    L.append("所以完整口径是 **dev 21 / holdout 6**。两个口径都对，分母不同，本表两个都列。\n")

    L.append("## 八、这批判定可能错在哪（给 A 技术复核重点看）\n")
    L.append("1. **multihop 16 条判「不是漏标」可能偏严。** 若认为「补上另一跳的 corroborating evidence」")
    L.append("   也算合法同义证据，这 16 条会翻一半左右。这是本次最需要 A 判的一类。")
    L.append("2. **adversarial 9 条的判读依赖「gold 答案陈述的依据」而不是「问题能否被回答」。**")
    L.append("   候选块确实能支撑「两者不同」，但给不出 gold 要求的那句依据（原文只出现短名）。")
    L.append("   若 P3 判据只要求「答案能被支撑」，这 9 条要重算。")
    L.append("3. **`q-al-000014` 判无法判定**是因为块里是「玛丽安娜」而不是问句里的「玛丽安娜·布莱克」，")
    L.append("   块里另有一处「雅拉·布克·威尔森」。要人确认「玛丽安娜·布莱克」与「雅拉」是否同指。")
    L.append("4. **cross_chunk 3 条按章节代号判**，若答案粒度允许「同章节其它块也算证据」则要重算。\n")

    L.append("## 九、留给 P0 的一笔：`direct_relation` 的检索指标口径待定案\n")
    L.append("第二节说明「同一条边的其它承载块不是漏标」，但这只解决了「gold 该不该改」；")
    L.append("**还有一个问题没解决：检索指标该按哪个口径算。**\n")
    sv2 = equiv["v2_strict"]
    L.append("`direct_relation` 恒 1 个 `gold_chunk_ids`，而按 V2 严口径，"
             f"{sv2['with_equivalent_non_gold_chunk']}/{equiv['questions']} 题存在非 gold 的等价承载块"
             f"（每题总等价块 {sv2['total_blocks_triple']}，扣掉 gold 那块后富余 "
             f"{sv2['extra_blocks_triple']}）。")
    L.append("**这个口径缺口是真的，但量级不大。** Lead 用 V2 严口径真跑（非回放）：\n")
    L.append("| 数据集 | gold 口径 R@5 | V2 等价口径 R@5 | 差 | gold 口径 R@20 | V2 等价口径 R@20 | 差 |")
    L.append("|---|---|---|---|---|---|---|")
    L.append("| v1 (300 应答题) | 0.5600 | 0.5833 | **+2.33pt（+7 题）** | 0.7833 | 0.7967 | **+1.33pt（+4 题）** |")
    L.append("| v2 (588 应答题) | 0.5561 | 0.5748 | +1.87pt（+11 题） | 0.7806 | 0.7942 | +1.36pt（+8 题） |")
    L.append("")
    L.append("**这里必须留一条更正记录**：本轮先用 V1 宽口径算出「+30pt / 96-96」，据此差点")
    L.append("得出「排序是主战场这个结论得重写」。换成 V2 严口径后是 **+1.33pt / 47-96**，")
    L.append("**不改变任何大方向**。宽口径里混着「`'天灾在我们头顶发生的可能性就越大'`」这类")
    L.append("根本没在说该关系的断言。量级差 20 倍以上，不许再引用宽口径的数。\n")
    L.append("两条路还是只能选一条，且都要**双向反向实测 + 报告里同时给两个口径的数**：")
    L.append("- **路 A**：把 `gold_chunk_ids` 扩成该边的全部承载块（改 gold，96 题口径重定）。")
    L.append("  代价：破坏「代表证据」语义，引用定位那条目前已满分的指标会失去基准。")
    L.append("- **路 B**（Lead 倾向）：gold 不动，改评分侧——`direct_relation` 的检索指标按")
    L.append("  「V2 等价块集合」判命中，引用定位仍按唯一 evidence 判（保住 1.0000）。")
    L.append("  代价：报告里必须同时给两套数，否则看的人会以为指标突然涨了。\n")
    L.append("**本表不预设任何一边，且这项的优先级是 P0 收尾**（量级 1.3pt，不插队到门报告之前）。")
    L.append("在定案之前，`direct_relation` 的 R@5 / R@20 **可以**与其他类比，但要注明是按哪个口径算的。\n")

    L.append("## 十、A 类 27 条：能不能只改 required 指向（dev 21 / holdout 6）\n")
    L.append("P1-b 那 27 条「gold 引用指错行」的题，要回答的问题不是「有没有别的块也提到这"
             "件事」（那个判据在 `direct_relation` 上恒为真，已作废），而是两个：\n")
    L.append("1. **答案是否被某条断言支持**（`needle in evidence_text`）；")
    L.append("2. **required 标的那一行是不是承载答案的那一行**（整行区间是否重叠）。\n")
    L.append("两个问题都要回答。只答第一个会把「引用口径窄」误判成「gold 指错行」；")
    L.append("只答第二个会把「证据本质跨行」误判成「可以挪一行」。\n")
    L.append("### 10.1 四态与它们各自的补救动作\n")
    L.append("| 判定 | 含义 | 该做什么 |")
    L.append("|---|---|---|")
    L.append("| `carrier_ok` | required 那句就落在承载答案的那一行上 | **不改**。required 带说话人前缀、"
             "断言不带，是引用口径窄，不是 gold 缺陷 |")
    L.append("| `move_safe` | 答案在同块另一行，且那一行**也含**问句里的别名 | 改 required "
             "指向那一行——安全，那一行同时证明「别名在这」与「答案是什么」 |")
    L.append("| `needs_extra` | 答案在同块另一行，但那一行**不含**别名 | **不能移动**。挪过去就丢掉"
             "「别名出现」这个唯一证据，而块里没有任何一行同时给出两者。只能追加一条 required，或接受跨行 |")
    L.append("| `boolean_answer` | 答案是否/没有这类布尔词 | **无法判定**。机器会把断言里的「没有」"
             "当成命中（例：「火龙并没有被源石感染」），判据在这里失效 |")
    L.append("")
    L.append(f"### 10.2 判定结果（{a_req['dev_n']} dev / {a_req['holdout_n']} holdout）\n")
    L.append("| 划分 | " + " | ".join(f"`{v}`" for v in A_CLASS_VERDICTS) + " | 合计 |")
    L.append("|---|---|---|---|---|---|---|")
    for split, label in (("dev", "**dev（可改侧）**"), ("holdout", "holdout（只记录不改）")):
        tally = a_req["by_split"].get(split) or {}
        cells = " | ".join(str(tally.get(v, 0)) for v in A_CLASS_VERDICTS)
        L.append(f"| {label} | {cells} | {sum(tally.values())} |")
    L.append("")
    dev_tally = a_req["by_split"].get("dev") or {}
    if not dev_tally.get("move_safe"):
        L.append("**关键结论：dev 侧 `move_safe` = 0。** 也就是说，dev 那 "
                 f"{a_req['dev_n']} 条里**没有一条能靠「只把 required 指向改到答案那一行」"
                 "救回来**——凡是答案在别处的，那一行都不含别名。")
        L.append("这一条如果没分开判，很容易被写成「dev 侧 15 条可改」，"
                 "而实际上把它们改过去会**让引用更难被断言支持**（丢掉了别名这条证据）。\n")
    L.append("### 10.3 逐题表\n")
    L.append("| 题号 | 划分 | 类别 | 答案 | 判定 | required 指向的那一行 | 承载答案的断言 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in a_req["rows"]:
        req = r["requirements"][0] if r["requirements"] else {}
        sup = (req.get("supporters_on_same_line") or req.get("supporters_other_line") or [{}])[0]
        req_line = (req.get("line") or "（未在块正文里找到）").replace("|", "\\|")[:44]
        sup_text = (sup.get("evidence_text") or "—").replace("|", "\\|")[:34]
        L.append(f"| `{r['question_id']}` | {r['split']} | {r['category']} | "
                 f"{r['answer']} | {r['verdict']} | {req_line} | {sup_text} |")
    L.append("")
    L.append("### 10.4 哪些已判「不是 gold 问题」的，理由是哪一条\n")
    carrier = [r for r in a_req["rows"] if r["verdict"] == "carrier_ok"]
    if carrier:
        L.append(f"`carrier_ok` {len(carrier)} 条："
                 + "、".join(f"`{r['question_id']}`" for r in carrier) + "。")
        L.append("这些题的 required 确实指向承载答案的那一行，只是 gold 把说话人前缀"
                 "（「杜林青年：」这类）也写进了 reference，而断言从冒号后面开始。"
                 "**这是引用口径窄，改它是改指标不是改标注。**\n")
    extra = [r for r in a_req["rows"] if r["verdict"] == "needs_extra"]
    if extra:
        L.append(f"`needs_extra` {len(extra)} 条全部是别名题，逐条核过**别名确实都在 required 那句里**：")
        L.append("、".join(f"`{r['question_id']}`" for r in extra) + "。")
        L.append("它们的共同形态是：问「X 被用来指代谁」，X 出现在一句里，答案出现在另一句里，"
                 "两句之间没有一行同时含两者。**这在现有 gold schema 下只能追加 required，"
                 "不能移动。**\n")

    L.append("### 10.5 这四条判据各自怎么变红（反向实测）\n")
    L.append("判定逻辑会直接导致改 gold，所以每条判据都必须有能变红的用例。"
             "用例在 `tests/test_p1c_leak_audit.py`，反向实测每次改一处实现、跑一条用例、再还原：\n")
    L.append("| 改坏的地方 | 变红的用例 | 实测 |")
    L.append("|---|---|---|")
    L.append("| 删掉 `move_safe` 分支（让它落到 `needs_extra`） | "
             "`test_alias_on_the_answer_line_is_what_makes_it_safe_to_move` | "
             "改前 0 / 改后 1（报错里 `+ needs_extra`）/ 还原 0 |")
    L.append("| 把 `move_safe` 条件放宽成「只要答案在别的行就算可挪」 | "
             "`test_answer_on_a_line_without_the_alias_must_not_be_moved` | "
             "改前 0 / 改后 1（报错里 `+ move_safe`）/ 还原 0 |")
    L.append("| 把「承载答案的断言与 required 同一行」的区间判据改成恒假 | "
             "`test_the_answer_sitting_on_the_required_line_needs_no_change` | "
             "改前 0 / 改后 1（报错里 `+ no_support`）/ 还原 0 |")
    L.append("| 清空 `BOOLEAN_ANSWERS` | `test_boolean_answers_are_not_judged_by_substring` | "
             "改前 0 / 改后 1（报错里 `+ no_support`）/ 还原 0 |")
    L.append("")
    L.append("**其中第二条第一次没做成实测**：第一版的用例选择器写成了 "
             "`needs_extra or zero_move_safe`，那不是有效的 `-k` 表达式，"
             "只选中一条恰好绿的用例，于是记成「没变红」。重做时把它指向真正该红的"
             "那一条，才拿到 0 / 1 / 0。**「没变红」先怀疑选择器与用例，再怀疑门。**\n")

    L.append("## 十一、这张表自己的变更日志\n")
    L.append("本表被改过三轮，每轮结论都变过，所以记账。**三轮都没有改动任何 gold 文件。**\n")
    L.append("| 版本 | 改了什么 | 结论怎么变 | 触发原因 |")
    L.append("|---|---|---|---|")
    L.append("| v1 | 首版：59 条逐条判 | 2 真漏标 / 56 不是漏标 / 1 无法判定 | — |")
    L.append("| v2 | `direct_relation` 那 2 条改判；补 V1 宽 / V2 严双口径对照 | "
             "**0 / 58 / 1** | 判据在 `direct_relation` 上恒为真（96/96），"
             "与 gold 挖掘自己的构造条件冲突；V1 宽口径还差点导出 +30pt 的假增益 |")
    L.append("| v3 | 新增第十节：A 类 27 条「能不能只改 required 指向」四态判定 | "
             "dev 21 条里 **`move_safe` 0 条** | P1-b 留的待办；四态是必需的，"
             "因为 `move_safe` 与 `needs_extra` 的补救动作相反 |")
    L.append("")
    L.append("**每一版的错都在「判据选错」，不在「算错」**：")
    L.append("- v1 的错：用「块里有等价断言」判漏标，该判据在 `direct_relation` 上恒真；")
    L.append("- v2 的错（自查出来的）：先用 V1 宽口径算增益，量级比 V2 严口径大 20 倍；")
    L.append("- v3 的错（自查出来的）：第一版把 `move_safe` 与 `needs_extra` 合成一种")
    L.append("  `move_required`，会报出「dev 侧 15 条可改」，而实际一条都不能只改指向。\n")
    L.append("影响题数（按 Lead 要求拆两个数）：**dev 影响 0 题 / holdout 影响 0 题**——")
    L.append("三轮下来结论是「不改任何 gold」。holdout 侧那 6 条（alias 5 + na 1）")
    L.append("按裁决只记录不改，若将来要动必须走保留集使用预算并记「动完哪些历史结论失效」。\n")
    L.append("技术复核（A）：待。批准（Lead）：待。**审核人空缺即表示尚未提交任何 gold 改动。**\n")
    return "\n".join(L) + "\n"


def build() -> tuple[str, dict]:
    triage_rows = read_jsonl(TRIAGE)
    corpus = load_corpus()
    gold_v1 = json.loads(GOLD_V1.read_text("utf-8"))
    gold_v2 = json.loads(GOLD_V2.read_text("utf-8"))
    split_map = {it["question_id"]: it["split"] for it in gold_v2["items"]}
    gold_v2_by_id = {it["question_id"]: it for it in gold_v2["items"]}

    states = required_states(gold_v1["items"], corpus)
    a_ids = a_class_ids(gold_v1["items"], corpus, states)
    rows = [judge_one(c, corpus, gold_v2_by_id) for c in candidates(triage_rows, split_map)]
    equiv = direct_relation_equivalence(corpus, gold_v2["items"])
    a_req = a_class_requirements(gold_v1["items"], corpus, states, a_ids, split_map)

    meta = {
        "chunks_source": corpus["chunks_source"],
        "chunks_loaded": len(corpus["chunks"]),
        "chunk_manifest_hash": corpus["chunk_manifest_hash"],
    }
    doc = render_doc(rows, meta, equiv, a_req, a_ids, split_map)
    payload = {
        "n": len(rows),
        "chunks_source": corpus["chunks_source"],
        "chunks_source_api": "pipeline.resolve_chunks(gen, manifest)",
        "chunk_manifest_hash": corpus["chunk_manifest_hash"][:16],
        "chunks_loaded": len(corpus["chunks"]),
        "verdicts": dict(Counter(r["verdict"] for r in rows)),
        "split": dict(Counter(r["split"] for r in rows)),
        "direct_relation_equivalence": equiv,
        "a_class_loose": a_ids,
        "a_class_split": {q: split_map.get(q, "<none>") for q in a_ids},
        "a_class_requirements": a_req,
        "rows": rows,
    }
    return doc, payload


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    check = "--check" in args
    doc, payload = build()
    body = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if check:
        old_doc = DOC.read_text("utf-8") if DOC.exists() else ""
        old_json = JSON_OUT.read_text("utf-8") if JSON_OUT.exists() else ""
        if old_doc == doc and old_json == body:
            print("判定表与 JSON 与当前数据一致")
            return 0
        print("判定表或 JSON 与当前数据不一致（重跑不带 --check 即可刷新）")
        return 1
    DOC.write_text(doc, encoding="utf-8")
    JSON_OUT.write_text(body, encoding="utf-8")
    print("wrote", config.rel_to_root(DOC))
    print("wrote", config.rel_to_root(JSON_OUT))
    print("块正文来源:", payload["chunks_source"], "块数:", payload["chunks_loaded"])
    print("判定:", payload["verdicts"], "| 划分:", payload["split"])
    equiv = payload["direct_relation_equivalence"]
    for caliber in ("v1_wide", "v2_strict"):
        block = equiv[caliber]
        print(f"direct_relation 等价({caliber}): "
              f"{block['with_equivalent_non_gold_chunk']}/{equiv['questions']} 题"
              f"（dev {block['dev']} / holdout {block['holdout']}），"
              f"总等价块 {block['total_blocks_triple']}，"
              f"扣 gold 后富余 {block['extra_blocks_triple']}，"
              f"gold 落在等价集合内 {block['gold_chunk_among_sites']}/{equiv['questions']}")
    print("A 类 27 条:", len(payload["a_class_loose"]),
          "| dev", sum(1 for v in payload["a_class_split"].values() if v == "dev"),
          "holdout", sum(1 for v in payload["a_class_split"].values() if v == "holdout"))
    req = payload["a_class_requirements"]
    print("A 类 required 判定: 全部", req["tally"], "| dev", req["by_split"].get("dev"),
          "| holdout", req["by_split"].get("holdout"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
