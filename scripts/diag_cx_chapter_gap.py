"""L2 诊断：`cross_chunk` 的 `chapter_set_missing` 到底缺在哪一步（**只读**）。

# 阶段 A：逐题把「缺章」拆成三个互斥的地方

`judge.judge_answer` 在 `cross_chunk` 上判 `chapter_set_missing` 当且仅当
`W ⊄ O`（`W` = gold 答案切出的章节集合，`O` = 本次答案切出的集合）。
生产档 `chapter_cap=4`、而这 20 道题的 gold 章节数最大 4 —— 所以**不是 cap 上限不够**。
阶段 A 回答的是：**`O` 为什么少了几章**，逐章归到三桶：

| 桶 | 含义 | 谁能修 |
|---|---|---|
| `A_chunk_absent` | 该章在**本次上下文**里一个单元都没有 | 召回/预算，答案装配救不了 |
| `B_no_name_in_chapter` | 该章有单元在上下文，但正文都不含焦点实体名（`nh=0`，被 `_chapter_answer` 的 `nh>=1` 滤掉） | 装配：入选判据太严 |
| `C_name_present_but_dropped` | 该章有单元且正文含焦点名（`nh>=1` 成立），却被 `chapter_cap` 的排序键挤出前 4 | 装配：**排序键精度** |

# 阶段 B：换排序键能把多少题救回来（上界实测，不是估计）

阶段 A 给出 `C` 桶的题，但"能救"必须证明：**存在一个不用 gold 的键**，在
`cap=4` 下把这些题的 gold 章节全部留在前 4 名，且不弄丢现在答对的题。

阶段 B 对**全部 64 道 cross_chunk 题**（dev 38 + holdout 26）跑生产档装配，
拿到 `_chapter_answer` 的 `by_chapter`（`nh>=1` 的那个集合），然后对每个候选键
重放一遍「排序 → 取前 4 → 按字典序呈现」，并把答案串喂给**真正的判据函数**
`judge.judge_answer`（只读，不改）。这就是「只换排序键、别的什么都不动」的
端到端上界。

**复现自检**：当前键（`sum_score_then_count`）的重放结果必须逐题等于生产档
实测判据，否则阶段 B 的所有数字作废 —— 脚本会把不一致逐题列出来并给出
`replication_ok`。这一条是阶段 B 可信的唯一依据。

口径与红线：

* **只读**：不写 `src/`，不改 `judge*`/`data/goldset/`。产物只有
  `data/diag_cx_chapter_gap.json` 与 `data/diag_cx_chapter_gap.txt`。
* 失败题选取**不自己判**：直接读 `data/w3_evidence/cap4_<split>.json` 里
  `judge.reason == "chapter_set_missing"` 的 `question_id`。复现的是生产档实测
  失败集，不是脚本作者以为的失败集。
* 上下文用**生产档原样装配**（`apply_prod_profile` + `ask_once`），不重写装配链。
* 候选键只消费 `unit["score"] / len(units) / evidence_spans`，**不读 gold、
  不读 `entity["chunks"]`**。gold 只用于事后判对错。

用法（必须走熔断器）：

    python scripts/run_guarded.py --max-commit-pct 92 --min-avail-mb 2048 -- \\
        python scripts/diag_cx_chapter_gap.py
"""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import answer as answer_mod  # noqa: E402
from arkrag import judge as judge_mod  # noqa: E402
from arkrag import pipeline as pipeline_mod  # noqa: E402

GEN = "g-20261103T132152Z-06a741"
GOLD = ROOT / "data" / "goldset" / "gold-v2.json"
EVID = ROOT / "data" / "w3_evidence"
OUT_JSON = ROOT / "data" / "diag_cx_chapter_gap.json"
OUT_TXT = ROOT / "data" / "diag_cx_chapter_gap.txt"

#: 问句里的锚点（与 `judge._ANCHOR_RE` 同一个式样，但这里**只做诊断**，
#: 不参与任何判分）。
ANCHOR_RE = re.compile(r"「([^」]{1,30})」")

SPLITS = ("dev", "holdout")
CAP = 4  # 生产档值；阶段 B 用它做"取前 4 章"


# ------------------------------------------------------------------ 候选排序键
def _sum_score(units: list[dict]) -> float:
    return sum(float(u.get("score") or 0.0) for u in units)


def _max_score(units: list[dict]) -> float:
    return max((float(u.get("score") or 0.0) for u in units), default=0.0)


def _n_spans(units: list[dict]) -> int:
    return sum(len(u.get("evidence_spans") or ()) for u in units)


def _n_assertions(units: list[dict]) -> int:
    return sum(len(u.get("assertions") or ()) for u in units)


def _n_focus_assertions(units: list[dict], names: set[str]) -> int:
    """单元里 subject/object 命中焦点名的断言条数（**只看本次上下文**）。"""
    n = 0
    for u in units:
        for brief in u.get("assertions") or ():
            if any(str(brief.get(k) or "") in names for k in ("subject", "object")):
                n += 1
    return n


def _n_direct(units: list[dict]) -> int:
    """直接证据单元数（`sources` 里有 bm25/entity 通道的才算；纯图扩展单元不算）。"""
    n = 0
    for u in units:
        chans = {str(s.get("channel") or "") for s in (u.get("sources") or ())}
        if chans & {"bm25", "entity", "speaker"}:
            n += 1
    return n


def _n_entities(units: list[dict]) -> int:
    return sum(len(u.get("entities") or ()) for u in units)


#: `{键名: 排序键}`，统一「升序 = 好的在前」，末分量一律章节名（确定性兜底）。
#:
#: **预登记组**：前五个与 `scripts/diag_cx_rank_keys.py` 的四键 + 当前键
#: **同名同义**（当前实现用的是 `sum_score_then_count`）。这一组的结果必须能
#: 与那份已入库的对照表对照，所以不许改定义。
#:
#: **探索组**（本轮新增，标注在这里以免被当成预登记）：全部只消费
#: `unit` 上现成的字段（`score / evidence_spans / assertions / entities / sources`），
#: **不读 gold、不读 `entity["chunks"]`**。加它们的理由：`C` 桶 11 例的形态是
#: 「gold 章 1 个单元被 3–5 个噪声章各 1 个单元挤掉」，而 `Σscore` 本质是
#: 「个数 × 均分」——所以要测不看总量、改看**单个单元证据厚度**的键。
KEYS: dict[str, object] = {
    # ---- 预登记组 ----
    "sum_score_then_count": lambda c, b, n: (-_sum_score(b[c]), -len(b[c]), str(c)),
    "sum_score": lambda c, b, n: (-_sum_score(b[c]), 0.0, str(c)),
    "max_score": lambda c, b, n: (-_max_score(b[c]), 0.0, str(c)),
    "count": lambda c, b, n: (-len(b[c]), 0.0, str(c)),
    "evidence": lambda c, b, n: (-_n_spans(b[c]), 0.0, str(c)),
    # ---- 探索组 ----
    "evidence_then_score": lambda c, b, n: (
        -_n_spans(b[c]), -_sum_score(b[c]), -len(b[c]), str(c)),
    "mean_score": lambda c, b, n: (
        -(_sum_score(b[c]) / max(1, len(b[c]))), -_max_score(b[c]), str(c)),
    "max_score_then_count": lambda c, b, n: (
        -_max_score(b[c]), -len(b[c]), str(c)),
    "assertions": lambda c, b, n: (-_n_assertions(b[c]), -_sum_score(b[c]), str(c)),
    "focus_assertions": lambda c, b, n: (
        -_n_focus_assertions(b[c], n), -_sum_score(b[c]), str(c)),
    "direct_units": lambda c, b, n: (-_n_direct(b[c]), -_sum_score(b[c]), str(c)),
    "entities_mentioned": lambda c, b, n: (
        -_n_entities(b[c]), -_sum_score(b[c]), str(c)),
}
CURRENT_KEY = "sum_score_then_count"


# ------------------------------------------------------------------ 阶段 A 工具
def _warnings(found: int, got: set[str], truncated: bool, cap: int) -> list[str]:
    """自检：`chapter_path` 报的 `n_chapters_found` 必须解释得通答案里的章节数。

    `cap=0` 时 `n_chapters_found` 不存在（键只在 cap 打开时写），跳过。
    非 0 时应有 `len(got) == min(found, cap)`；对不上就是记账和答案脱节，
    必须点名——否则本诊断的"缺章"归因会建立在一个错的 `O` 上。
    """
    if not cap:
        return []
    want_len = min(found, cap)
    if found and want_len != len(got):
        return [f"n_chapters_found={found} cap={cap} 与 |O|={len(got)} 不自洽"
                f"（截断={truncated}）"]
    return []


def load_items() -> list[dict]:
    doc = json.loads(GOLD.read_text("utf-8"))
    return doc.get("items") if isinstance(doc, dict) else doc


def failing_ids(split: str) -> list[str]:
    """从生产档实测报告里取 `chapter_set_missing` 的 question_id（按报告顺序）。"""
    path = EVID / f"cap4_{split}.json"
    if not path.exists():
        raise SystemExit(f"缺报告 {path}（先跑 W3 的 cap4 实测）")
    rep = json.loads(path.read_text("utf-8"))
    out = []
    for row in rep.get("rows") or []:
        if str(row.get("category") or "") != "cross_chunk":
            continue
        if (row.get("judge") or {}).get("reason") == "chapter_set_missing":
            out.append(str(row.get("question_id") or ""))
    return out


def by_chapter_of(units: list[dict], names: set[str]) -> dict[str, list[dict]]:
    """复刻 `answer._chapter_answer` 的聚合：`nh>=1`（正文含焦点名）才收。"""
    out: dict[str, list[dict]] = {}
    for u in units:
        ch = str(u.get("chapter") or "").strip()
        if not ch:
            continue
        text = str(u.get("text") or "")
        if not text or not any(n in text for n in names):
            continue
        out.setdefault(ch, []).append(u)
    return out


# ------------------------------------------------------------------ 主流程
def main() -> int:
    items = {str(it.get("question_id") or ""): it for it in load_items()}
    fail_targets: list[tuple[str, str]] = []
    for split in SPLITS:
        for qid in failing_ids(split):
            fail_targets.append((split, qid))

    # 阶段 B 要全部 cross_chunk 题（dev + holdout，去重后 64 道）
    all_cx: list[dict] = []
    seen: set[str] = set()
    for split in SPLITS:
        rep = json.loads((EVID / f"cap4_{split}.json").read_text("utf-8"))
        for row in rep.get("rows") or []:
            if str(row.get("category") or "") != "cross_chunk":
                continue
            qid = str(row.get("question_id") or "")
            if qid in seen:
                continue
            seen.add(qid)
            all_cx.append({"question_id": qid, "split": split,
                           "prod_pass": bool((row.get("judge") or {}).get("pass") is True),
                           "prod_reason": (row.get("judge") or {}).get("reason")})

    loaded = pipeline_mod.load_generation(GEN)
    chunks = loaded["chunks"]
    cid2chapter = {cid: str(rec.get("chapter") or "") for cid, rec in chunks.items()}
    ct: dict[str, list[str]] = {}
    for rec in chunks.values():
        ct.setdefault(str(rec.get("chapter") or ""), []).append(str(rec.get("text") or ""))
    chap_texts = {k: "\n".join(v) for k, v in ct.items()}

    pipe = pipeline_mod.build_pipeline(loaded, top_k=20, graph_depth=1)
    pipe["generation_id"] = GEN
    pipe["entities"] = loaded["graph"].get("entities") or []
    options = pipeline_mod.apply_prod_profile(pipe)
    cap = int(options.get("chapter_cap") or 0) or CAP

    # 先把 64 道全跑一遍，题目信息与上下文都留下来给两个阶段用
    facts: dict[str, dict] = {}
    for rec in all_cx:
        qid = rec["question_id"]
        it = items.get(qid)
        if it is None:
            continue
        q = str(it.get("question") or "")
        res = pipeline_mod.ask_once(pipe, q, options=options)
        ans = res.get("answer") or {}
        text = str(ans.get("answer") or "")
        verdict = judge_mod.judge_answer("cross_chunk", text, it,
                                         question=q, chapter_texts=chap_texts)
        qa = res.get("query_analysis") or {}
        cands = (qa.get("entity_candidates") or [])[:2]
        names = {str(c.get("name") or "").strip() for c in cands}
        names.discard("")
        units = list((res.get("context") or {}).get("units") or [])
        facts[qid] = {
            "item": it, "question": q, "prod_text": text, "prod_verdict": verdict,
            "units": units, "names": names,
            "anchors": [a.strip() for a in ANCHOR_RE.findall(q) if a.strip()],
            "by_chapter": by_chapter_of(units, names),
            "chapter_path": ((ans.get("stats") or {}).get("chapter_path") or {}),
            "n_entity_candidates_total": len(qa.get("entity_candidates") or []),
            "rec": rec,
        }

    # ================================================= 阶段 A：逐题归因
    rows: list[dict] = []
    for split, qid in fail_targets:
        f = facts.get(qid)
        if f is None:
            rows.append({"question_id": qid, "split": split,
                         "error": "gold_item_missing_or_not_run"})
            continue
        it, q = f["item"], f["question"]
        text, verdict = f["prod_text"], f["prod_verdict"]
        units, names, by_chapter = f["units"], f["names"], f["by_chapter"]
        anchors = f["anchors"]

        want = judge_mod.chapter_set(it.get("answer"))
        got = judge_mod.chapter_set(text)
        missing = sorted(want - got)
        extra = sorted(got - want)

        by_ch: dict[str, list[dict]] = {}
        for u in units:
            by_ch.setdefault(str(u.get("chapter") or ""), []).append(u)

        gold_cids = [str(c) for c in (it.get("gold_chunk_ids") or [])]
        ctx_cids = {str(u.get("chunk_id") or "") for u in units}
        gold_ch_cids = {cid: cid2chapter.get(cid, "") for cid in gold_cids}
        gold_in_ctx = [c for c in gold_cids if c in ctx_cids]

        stats = f["chapter_path"]
        found = int(stats.get("n_chapters_found") or 0)
        truncated = bool(stats.get("chapters_truncated"))

        per_missing: list[dict] = []
        for ch in missing:
            us = by_ch.get(ch) or []
            named = by_chapter.get(ch) or []
            anchor_hits: list[dict] = []
            for u in us:
                body = str(u.get("text") or "")
                hit = [a for a in anchors if a in body]
                if hit:
                    idx = body.find(hit[0])
                    anchor_hits.append({
                        "chunk_id": str(u.get("chunk_id") or ""),
                        "anchor": hit[0],
                        "snippet": body[max(0, idx - 20): idx + 60].replace("\n", " "),
                    })
            full_has = {a: (a in chap_texts.get(ch, "")) for a in anchors}
            if not us:
                bucket = "A_chunk_absent"
            elif not named:
                bucket = "B_no_name_in_chapter"
            else:
                bucket = "C_name_present_but_dropped"
            # 若该章进了候选集（`nh>=1`），它在各键下排第几？
            rank_now = rank_focus = None
            n_focus_assert = n_assert = n_spans = None
            if named:
                ord_now = sorted(by_chapter,
                                 key=lambda c: KEYS[CURRENT_KEY](c, by_chapter, names))
                rank_now = ord_now.index(ch) + 1
                ord_focus = sorted(by_chapter,
                                   key=lambda c: KEYS["focus_assertions"](c, by_chapter, names))
                rank_focus = ord_focus.index(ch) + 1
                n_focus_assert = _n_focus_assertions(named, names)
                n_assert = _n_assertions(named)
                n_spans = _n_spans(named)
            per_missing.append({
                "chapter": ch,
                "bucket": bucket,
                "n_units_in_ctx": len(us),
                "n_units_with_focus_name": len(named),
                "n_units_with_anchor": len(anchor_hits),
                "rank_under_current_key": rank_now,
                "rank_under_focus_assertions_key": rank_focus,
                "n_assertions": n_assert,
                "n_focus_assertions": n_focus_assert,
                "n_evidence_spans": n_spans,
                "anchor_hits": anchor_hits[:3],
                "chapter_corpus_has_anchor": full_has,
                "gold_chunks_in_ctx_of_this_chapter": [
                    c for c in gold_in_ctx if cid2chapter.get(c) == ch],
                "unit_scores": [round(float(u.get("score") or 0.0), 4) for u in us][:6],
            })

        rows.append({
            "question_id": qid,
            "split": split,
            "question": q,
            "want": sorted(want),
            "got": sorted(got),
            "missing": missing,
            "extra": extra,
            "n_want": len(want),
            "n_got": len(got),
            "judge_reason": verdict.get("reason"),
            "answer_text": text,
            "chapter_path": stats,
            "n_chapters_found": found,
            "chapters_truncated": truncated,
            "cap": cap,
            "focus_names": sorted(names),
            "anchors": anchors,
            "n_entity_candidates_total": f["n_entity_candidates_total"],
            "ctx_chapters": sorted(k for k in by_ch if k),
            "ctx_n_units": len(units),
            "ctx_chapters_with_focus_name": sorted(by_chapter),
            "candidate_set_size": len(by_chapter),
            "gold_chunk_ids": gold_cids,
            "gold_chapters_from_chunks": sorted({v for v in gold_ch_cids.values() if v}),
            "gold_in_ctx": gold_in_ctx,
            "gold_chapters_covered_by_ctx": sorted(
                {cid2chapter.get(c, "") for c in gold_in_ctx} - {""}),
            "w_subset_of_candidates": sorted(want) and set(want) <= set(by_chapter),
            "max_recoverable_by_reselection": bool(
                want and set(want) <= set(by_chapter) and len(want) <= cap),
            "per_missing_chapter": per_missing,
            "warnings": _warnings(found, got, truncated, cap),
        })

    ok_rows = [r for r in rows if "error" not in r]
    bucket_counts: dict[str, int] = {}
    for r in ok_rows:
        for pm in r["per_missing_chapter"]:
            bucket_counts[pm["bucket"]] = bucket_counts.get(pm["bucket"], 0) + 1
    q_level = {"gold_all_in_ctx": 0, "gold_partly_in_ctx": 0, "gold_none_in_ctx": 0}
    for r in ok_rows:
        want_ch = set(r["want"])
        cov = set(r["gold_chapters_covered_by_ctx"])
        if want_ch <= cov:
            q_level["gold_all_in_ctx"] += 1
        elif cov:
            q_level["gold_partly_in_ctx"] += 1
        else:
            q_level["gold_none_in_ctx"] += 1

    recoverable = [r["question_id"] for r in ok_rows if r["max_recoverable_by_reselection"]]

    # ================================================= 阶段 C：上界与不可修的部分
    #
    # 阶段 B 只证明「换某个**不看 gold** 的键能救 4 题」。阶段 C 回答两个
    # 必须同时报出来的问题，否则 +4 会被误读成「只差一个排序键」：
    #
    # 1. **重选章节这条路的天花板**：给一个 **oracle 键**（把 gold 章节排最前），
    #    在同样 `cap=4` 下能救几题。若 oracle 也只有 7 题，说明剩下 13 题
    #    **不是排序问题**，换任何键都救不回来。它是上界，不是方案。
    # 2. **`A` 桶到底缺在哪**：gold 章在本次上下文里没有单元。用**实体索引
    #    直读**（`graph.entities[].chunks` → 章节）看那章是否属于焦点实体自己
    #    的章节集合。命中 ⇒ 信息在产物里、只是没进上下文（属召回面）；
    #    不命中 ⇒ gold 章节与实体索引都不一致（属 gold/索引面）。
    #    ⚠️ 索引直读是本项目对抗红线 34 要求必须标注的口径，这里**只用于诊断**，
    #    不参与任何可交付通路。
    entity_chapters: dict[str, set[str]] = {}
    for ent in loaded["graph"].get("entities") or []:
        nm = str(ent.get("name") or "")
        if not nm:
            continue
        cs = {cid2chapter.get(str(c)) for c in (ent.get("chunks") or [])}
        cs.discard("")
        cs.discard(None)
        if cs:
            entity_chapters.setdefault(nm, set()).update(cs)

    oracle_res: dict[str, dict] = {}
    a_bucket_rows: list[dict] = []
    for split, qid in fail_targets:
        f = facts.get(qid)
        if f is None:
            continue
        it = f["item"]
        want = judge_mod.chapter_set(it.get("answer"))
        b = f["by_chapter"]
        # --- oracle 键：gold 章节排最前（**上界，不是方案**）
        oracle_text = None
        if b:
            ordered = sorted(b, key=lambda c: (0 if c in want else 1, str(c)))
            kept = set(ordered[:cap]) if len(ordered) > cap else set(ordered)
            chapters = [c for c in sorted(b) if c in kept]
            oracle_text = "、".join(chapters)
            v = judge_mod.judge_answer("cross_chunk", oracle_text, it,
                                       question=f["question"], chapter_texts=chap_texts)
            oracle_res[qid] = {"text": oracle_text, "pass": v.get("pass") is True,
                               "reason": v.get("reason")}
        else:
            oracle_res[qid] = {"text": f["prod_text"],
                               "pass": f["prod_verdict"].get("pass") is True,
                               "reason": f["prod_verdict"].get("reason")}
        # --- A 桶逐章：实体索引里有没有这一章
        gname = set()
        for nm in f["names"]:
            gname |= entity_chapters.get(nm, set())
        for ch in sorted(want - judge_mod.chapter_set(f["prod_text"])):
            in_ctx = any(str(u.get("chapter") or "") == ch for u in f["units"])
            if in_ctx:
                continue
            a_bucket_rows.append({
                "question_id": qid,
                "split": split,
                "chapter": ch,
                "in_entity_index_chapters": ch in gname,
            })

    n_oracle = sum(1 for x in oracle_res.values() if x["pass"])
    a_in_idx = sum(1 for r in a_bucket_rows if r["in_entity_index_chapters"])
    stage_c = {
        "oracle_key_upper_bound": {
            "n_pass": n_oracle,
            "n_questions": len(oracle_res),
            "prod_pass_on_same_set": sum(
                1 for qid in oracle_res if facts[qid]["rec"]["prod_pass"]),
            "n_fail": len(oracle_res),
            "note": "oracle 键 = gold 章节排最前，**不是可用方案**，"
                    "只用来界定「重选章节」这条路的天花板；"
                    "它同时说明剩余题不是排序问题",
            "still_failing": sorted(qid for qid, x in oracle_res.items()
                                    if not x["pass"]),
        },
        "a_bucket_chapter_misses": {
            "n": len(a_bucket_rows),
            "in_entity_index": a_in_idx,
            "not_in_entity_index": len(a_bucket_rows) - a_in_idx,
            "rows": a_bucket_rows,
            "note": "gold 章在本次上下文里无单元。`in_entity_index_chapters=True` "
                    "⇒ 该章属于焦点实体自己的 chunks（信息在产物里、没进上下文，"
                    "属召回面）；False ⇒ 连实体索引里也没有，属 gold/索引面。"
                    "该口径是**索引直读**，红线 34 要求标注，仅用于诊断。",
        },
    }


    key_results: dict[str, dict] = {}
    per_q_by_key: dict[str, dict[str, dict]] = {}
    for kname, kf in KEYS.items():
        per_q: dict[str, dict] = {}
        for qid, f in facts.items():
            it, q = f["item"], f["question"]
            b = f["by_chapter"]
            if not b:
                # 没有候选章节 ⇒ 真实通路落回 claim 通路，答案不是章节串。
                # 换键对它**一点影响都没有**，直接记为生产档实测结果。
                per_q[qid] = {
                    "text": f["prod_text"],
                    "pass": f["prod_verdict"].get("pass") is True,
                    "reason": f["prod_verdict"].get("reason"),
                    "n_chapters": len(judge_mod.chapter_set(f["prod_text"])),
                    "changed_from_prod": False,
                }
                continue
            ordered = sorted(b, key=lambda c: kf(c, b, f["names"]))
            kept = set(ordered[:cap]) if len(ordered) > cap else set(ordered)
            chapters = [c for c in sorted(b) if c in kept]  # 呈现仍按字典序
            sim_text = "、".join(chapters)
            v = judge_mod.judge_answer("cross_chunk", sim_text, it,
                                       question=q, chapter_texts=chap_texts)
            per_q[qid] = {
                "text": sim_text,
                "pass": v.get("pass") is True,
                "reason": v.get("reason"),
                "n_chapters": len(chapters),
                "changed_from_prod": sim_text != f["prod_text"],
            }
        per_q_by_key[kname] = per_q
        n_pass = sum(1 for x in per_q.values() if x["pass"])
        gain = [qid for qid, x in per_q.items()
                if x["pass"] and not facts[qid]["rec"]["prod_pass"]]
        loss = [qid for qid, x in per_q.items()
                if facts[qid]["rec"]["prod_pass"] and not x["pass"]]
        per_split = {}
        for sp in SPLITS:
            sub = [qid for qid, x in per_q.items() if facts[qid]["rec"]["split"] == sp]
            per_split[sp] = {
                "n": len(sub),
                "pass": sum(1 for qid in sub if per_q[qid]["pass"]),
                "prod_pass": sum(1 for qid in sub if facts[qid]["rec"]["prod_pass"]),
            }
        key_results[kname] = {
            "n": len(per_q),
            "pass": n_pass,
            "prod_pass": sum(1 for f in facts.values() if f["rec"]["prod_pass"]),
            "gain": len(gain), "loss": len(loss),
            "gain_ids": sorted(gain), "loss_ids": sorted(loss),
            "changed_ids": sorted(qid for qid, x in per_q.items()
                                  if x["changed_from_prod"]),
            "per_split": per_split,
        }

    # 复现自检：当前键重放 vs 生产档实测判据
    cur = per_q_by_key[CURRENT_KEY]
    mismatch = [qid for qid in cur
                if cur[qid]["pass"] != facts[qid]["rec"]["prod_pass"]]
    text_diff = [qid for qid in cur
                 if cur[qid]["text"] != facts[qid]["prod_text"]]

    summary = {
        "generation_id": GEN,
        "prod_options": dict(options),
        "cap": cap,
        "n_failing": len(ok_rows),
        "split_counts": {s: sum(1 for r in ok_rows if r["split"] == s) for s in SPLITS},
        "per_split_bucket": {},
        "per_missing_chapter_buckets": bucket_counts,
        "question_level_gold_coverage": q_level,
        "want_size_hist": {},
        "got_size_hist": {},
        "recoverable_by_reselection": {
            "n": len(recoverable), "ids": recoverable,
            "note": "W ⊆ `_chapter_answer` 候选集 且 |W| ≤ cap ⇒ 只换排序键就可能全中",
        },
        "stage_b": {
            "n_questions": len(facts),
            "replication_ok": not mismatch,
            "replication_mismatches": sorted(mismatch),
            "replication_text_diffs": sorted(text_diff),
            "replication_note": "当前键的重放判据必须逐题等于生产档实测；"
                                "不相等则本阶段的换键结论全部作废",
            "per_key": key_results,
        },
        "stage_c": stage_c,
    }
    for s in SPLITS:
        sub = [r for r in ok_rows if r["split"] == s]
        bc: dict[str, int] = {}
        for r in sub:
            for pm in r["per_missing_chapter"]:
                bc[pm["bucket"]] = bc.get(pm["bucket"], 0) + 1
        summary["per_split_bucket"][s] = {
            "n": len(sub), "buckets": bc,
            "gold_all_in_ctx": sum(
                1 for r in sub if set(r["want"]) <= set(r["gold_chapters_covered_by_ctx"])),
            "recoverable_by_reselection": sum(
                1 for r in sub if r["max_recoverable_by_reselection"]),
        }
    for r in ok_rows:
        summary["want_size_hist"][str(r["n_want"])] = \
            summary["want_size_hist"].get(str(r["n_want"]), 0) + 1
        summary["got_size_hist"][str(r["n_got"])] = \
            summary["got_size_hist"].get(str(r["n_got"]), 0) + 1

    OUT_JSON.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")

    # ---------------------------------------------------------- 文本报告
    lines: list[str] = []
    p = lines.append
    p("=" * 96)
    p(f"L2 诊断：cross_chunk chapter_set_missing   世代 {GEN}   生产档 cap={cap}")
    p(f"失败题 {len(ok_rows)}  分列 {summary['split_counts']}")
    p(f"gold 章节数分布 {summary['want_size_hist']}   答案章节数分布 {summary['got_size_hist']}")
    p(f"缺口分桶（按**缺失章节**计数）{bucket_counts}")
    p(f"题级 gold 块在上下文里的覆盖 {q_level}")
    p(f"只换排序键就可能全中的题 {len(recoverable)}: {recoverable}")
    p("=" * 96)
    p("")
    p("## 阶段 B：换排序键的端到端上界（判据用真的 `judge.judge_answer`）")
    p(f"   复现自检 replication_ok = {not mismatch}  "
      f"(判据不一致 {len(mismatch)} 题：{sorted(mismatch)})")
    if text_diff:
        p(f"   答案串不一致 {len(text_diff)} 题：{sorted(text_diff)}")
    base = key_results[CURRENT_KEY]["prod_pass"]
    p(f"   生产档实测答对 {base}/{len(facts)}")
    p(f"   {'键':<24}{'答对':>6}{'+':>5}{'-':>5}   {'dev':>9}{'holdout':>11}")
    for kname in KEYS:
        r = key_results[kname]
        dsp = r["per_split"]["dev"]
        hsp = r["per_split"]["holdout"]
        p(f"   {kname:<24}{r['pass']:>6}{r['gain']:>5}{r['loss']:>5}   "
          f"{dsp['pass']}/{dsp['n']:<7}{hsp['pass']}/{hsp['n']:<9}")
    best = max(KEYS, key=lambda k: (key_results[k]["pass"], -key_results[k]["loss"]))
    p(f"   净最好：{best}  {key_results[best]['pass']}/{len(facts)}  "
      f"(+{key_results[best]['gain']} / -{key_results[best]['loss']})")
    p("")
    p("## 阶段 B 逐题翻转明细（净最好的那个键）")
    for qid in key_results[best]["gain_ids"]:
        f = facts[qid]
        p(f"   + {qid} ({f['rec']['split']})  生产: {f['prod_text']!r}")
        p(f"     -> {per_q_by_key[best][qid]['text']!r}   "
          f"gold={sorted(judge_mod.chapter_set(f['item'].get('answer')))}")
    for qid in key_results[best]["loss_ids"]:
        f = facts[qid]
        p(f"   - {qid} ({f['rec']['split']})  生产: {f['prod_text']!r}")
        p(f"     -> {per_q_by_key[best][qid]['text']!r}   "
          f"gold={sorted(judge_mod.chapter_set(f['item'].get('answer')))}")
    p("")
    p("## 阶段 C：重选章节这条路的天花板 + A 桶缺章的来源")
    ou = stage_c["oracle_key_upper_bound"]
    p(f"   oracle 键（gold 排最前，**上界不是方案**）: {ou['n_pass']}/{ou['n_questions']} "
      f"（生产档在同一批上 {ou['prod_pass_on_same_set']}）")
    p(f"   仍然失败的 {len(ou['still_failing'])} 题: {ou['still_failing']}")
    ab = stage_c["a_bucket_chapter_misses"]
    p(f"   A 桶缺章 {ab['n']} 处：属于焦点实体自己的章节 {ab['in_entity_index']} 处，"
      f"连实体索引里也没有 {ab['not_in_entity_index']} 处")
    for r in ab["rows"]:
        p(f"      {r['question_id']:<14} {r['split']:<8} {r['chapter']:<16} "
          f"实体索引里 = {r['in_entity_index_chapters']}")
    p("")
    p("")
    p("## 阶段 A 逐题")
    for r in rows:
        if "error" in r:
            p(f"\n[{r['split']}] {r['question_id']}  ⚠ {r['error']}")
            continue
        p("")
        p(f"[{r['split']}] {r['question_id']}  判据={r['judge_reason']}")
        p(f"  问句: {r['question']}")
        p(f"  W(gold) = {r['want']}")
        p(f"  O(答案) = {r['got']}     缺 = {r['missing']}   多 = {r['extra']}")
        p(f"  答案串 = {r['answer_text'][:120]!r}")
        p(f"  章节通路: {r['chapter_path']}")
        p(f"  候选集(|{CURRENT_KEY}|) = {r['candidate_set_size']} 章   "
          f"W ⊆ 候选集 = {r['w_subset_of_candidates']}   "
          f"只换键可救 = {r['max_recoverable_by_reselection']}")
        p(f"  焦点名(entity_candidates[:2]) = {r['focus_names']}  "
          f"(全候选 {r['n_entity_candidates_total']} 个)  问句锚点 = {r['anchors']}")
        p(f"  上下文: {r['ctx_n_units']} 单元 / {len(r['ctx_chapters'])} 章 {r['ctx_chapters']}")
        p(f"  gold 块在上下文的 = {len(r['gold_in_ctx'])}/{len(r['gold_chunk_ids'])} "
          f"-> 覆盖章 {r['gold_chapters_covered_by_ctx']}")
        for pm in r["per_missing_chapter"]:
            p(f"    缺章 {pm['chapter']:<16} -> {pm['bucket']}"
              f"  (当前键名次 {pm['rank_under_current_key']} / "
              f"focus_assertions 键名次 {pm['rank_under_focus_assertions_key']})")
            p(f"       该章在上下文的单元 {pm['n_units_in_ctx']} 个，"
              f"含焦点名 {pm['n_units_with_focus_name']} 个，"
              f"含问句锚点 {pm['n_units_with_anchor']} 个")
            if pm["n_units_with_focus_name"]:
                p(f"       该章候选内：断言 {pm['n_assertions']} 条 / "
                  f"其中端点命中焦点名 {pm['n_focus_assertions']} 条 / "
                  f"evidence_spans {pm['n_evidence_spans']}")
            p(f"       该章全语料正文含锚点: {pm['chapter_corpus_has_anchor']}")
            if pm["unit_scores"]:
                p(f"       该章上下文单元分数: {pm['unit_scores']}")
            if pm["gold_chunks_in_ctx_of_this_chapter"]:
                p(f"       ★ gold 块就在这些单元里: "
                  f"{pm['gold_chunks_in_ctx_of_this_chapter']}")
            for h in pm["anchor_hits"]:
                p(f"       锚点命中 {h['chunk_id']}: …{h['snippet']}…")

    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    print("\n".join(lines))
    print(f"\n-> {OUT_JSON}")
    print(f"-> {OUT_TXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
