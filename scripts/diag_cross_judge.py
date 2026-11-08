"""`diag_cross_judge` —— cross_chunk 判据诊断：gold 到底是不是"全部章节"、
`_chapter_answer` 多列的章节到底是真章节还是无关章节、以及一条不能靠多列通过的判据。

用法::

    py -3 scripts\\diag_cross_judge.py                 # 默认 6000/12000 两档
    py -3 scripts\\diag_cross_judge.py --budgets 6000

## 三个口径（全部用真图真块算，不查库、不猜）

* `G_name`  = 图里该实体 `chunks` 中**正文确实含该实体规范名**的块所属章节集合。
  这与 `goldset.mine_cross_chunk` 的挖掘判据逐字一致（它也是 `for cid in e["chunks"]
  if _in_text(chunk, name)`），所以它是"gold 应当等于什么"的机检口径。
* `G_all`   = 图里该实体 `chunks` 全部块所属章节集合（不过滤正文）。
* `C_text`  = 语料里**任意**块的正文含该实体规范名的章节集合（完全绕开图的归属）。
  这是"正文出现过"的最宽口径；普通名词（`巫术`/`银行`）在这个口径下会炸到几十章。

## 判定边界

`_chapter_answer` 的入选判据是"该章节下至少一个**上下文单元**的正文含焦点名"。
所以它列出的章节必然在 `C_text` 里（名字确实出现在该章正文），但**不一定**在
`G_name` 里 —— 图没把那个块归给该实体（归属缺口），或者命中的是第 2 个实体候选。

脚本对每个越界章节分四类（互斥，按优先级）::

    gold_gap           章节在 G_name 里而 gold 没列（gold 不全）
    alias_only         章节在 G_all 里（有块归属该实体），但该章正文不含规范名
    unattributed_real  章节不在 G_all，但该章正文含规范名（图归属缺口）
    unrelated          该章正文根本不含规范名（与焦点实体无关）

## 纪律

打印一律 ASCII（Windows 控制台 GBK 会炸），不写 U+2212 / U+2192 / U+21D2 这类符号。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import answer as answer_mod  # noqa: E402
from arkrag import pipeline as pipeline_mod  # noqa: E402

GENERATION_ID = "g-20261103T132152Z-06a741"
GOLD_PATH = ROOT / "data" / "goldset" / "gold-v2.json"
OUT_JSON = ROOT / "data" / "diag_cross_judge.json"

QUOTE_RE = re.compile(r"「([^」]{2,})」")
SPLIT_RE = re.compile(r"[、,，;；\s]+")


def split_chapters(text: str) -> list[str]:
    """把 `a、b` / `a b` / `a,b` 切成章节列表（保持首次出现顺序）。"""
    out: list[str] = []
    for part in SPLIT_RE.split(str(text or "")):
        part = part.strip()
        if part and part not in out:
            out.append(part)
    return out


def norm_name(name: str) -> str:
    return str(name or "").strip()


def build_chapter_text_index(chunks: dict) -> dict[str, str]:
    """`chapter -> 该章全部块正文的拼接`（只用于 `name in text` 判定）。"""
    by_ch: dict[str, list[str]] = defaultdict(list)
    for rec in chunks.values():
        ch = str(rec.get("chapter") or "").strip()
        if not ch:
            continue
        by_ch[ch].append(str(rec.get("text") or ""))
    return {ch: "\n".join(texts) for ch, texts in by_ch.items()}


def rank(passed: int, total: int, label: str) -> str:
    """统一口径写法 `<分子>/<分母> [口径名]`。"""
    return f"{passed}/{total} [{label}]"


def run_one(pipe: dict, question: str) -> dict:
    """跑一次生产链路，原样返回（不改任何 src 行为）。"""
    return pipeline_mod.ask_once(pipe, question)


def _occurrence_spans(text: str, needle: str) -> list[tuple[int, int]]:
    """`needle` 在 `text` 里的全部出现位置（左闭右开）。"""
    out: list[tuple[int, int]] = []
    if not needle:
        return out
    i = text.find(needle)
    while i >= 0:
        out.append((i, i + len(needle)))
        i = text.find(needle, i + 1)
    return out


def standalone_occurs(text: str, name: str, longer: list[str]) -> bool:
    """`name` 是否在 `text` 里**独立**出现（至少一次不被更长的实体名包住）。

    生产 `_chapter_answer` 用的是裸子串 `name in unit["text"]`，所以「帕特」会命中
    「帕特里奇昂」、「蒂姆」会命中「卡蒂姆」——那是**另一个实体**。这里把这种
    "只在更长实体名内部命中" 的情况单独识别出来。
    """
    if not name:
        return False
    spans = _occurrence_spans(text, name)
    if not spans:
        return False
    covered: list[tuple[int, int]] = []
    for longer_name in longer:
        if len(longer_name) <= len(name):
            continue
        covered.extend(_occurrence_spans(text, longer_name))
    for start, end in spans:
        if not any(cs <= start and end <= ce for cs, ce in covered):
            return True
    return False


def variant_chapters(
    units: list[dict],
    names: set[str],
    *,
    require_attributed: bool = False,
    require_text: bool = True,
    require_assertion: bool = False,
    require_standalone: bool = False,
    longer: dict[str, list[str]] | None = None,
    allowed_chunks: set[str] | None = None,
    chapter_whitelist: set[str] | None = None,
) -> list[str]:
    """在**同一批上下文单元**上重放 `_chapter_answer` 的章节判据（只读，不改 src）。

    生产判据 = `require_text=True, require_attributed=False`
    （`_chapter_answer`: 章节入选 <=> 该章有单元正文含焦点名）。

    变体用来回答"通路该怎么收紧"：

    * `require_attributed=True` —— 还要求该单元的 `entities` 里出现焦点名。
      `unit["entities"]` 只由**该块里与焦点实体/关系提示相关的断言**填充
      （见 `retrieve.build_context` 步骤 1），所以它是"图把这块归给了该实体"
      的上下文内代理。
    * `require_assertion=True` —— 还要求该块有一条断言的 subject/object 是焦点名。
    * `require_standalone=True` —— 还要求正文里有一次**不被更长实体名包住**的
      命中（挡掉「帕特」命中「帕特里奇昂」这类跨实体子串误配）。
    * `require_text=False` —— 只要求归属，不要求正文出现（等价于 na>=1 一类）。
    * `allowed_chunks` —— 只接受落在该白名单里的块（用图实体 `chunks` 字段
      做归属，而不是上下文里稀疏的 `unit["entities"]`）。
    """
    longer = longer or {}
    out: list[str] = []
    for unit in units:
        chapter = str(unit.get("chapter") or "").strip()
        if not chapter:
            continue
        text = str(unit.get("text") or "")
        ents = {str(e) for e in (unit.get("entities") or [])}
        briefs = unit.get("assertions") or []
        assert_names: set[str] = set()
        for brief in briefs:
            for key in ("subject", "object"):
                val = str(brief.get(key) or "")
                if val:
                    assert_names.add(val)

        if require_text and not any(n and n in text for n in names):
            continue
        if allowed_chunks is not None and str(unit.get("chunk_id") or "") not in allowed_chunks:
            continue
        if chapter_whitelist is not None and chapter not in chapter_whitelist:
            continue
        if require_standalone and not any(
            standalone_occurs(text, n, longer.get(n) or []) for n in names
        ):
            continue
        if require_attributed and not (names & ents):
            continue
        if require_assertion and not (names & assert_names):
            continue
        if chapter not in out:
            out.append(chapter)
    return sorted(out)


def find_extra_evidence(
    units: list[dict],
    chapter: str,
    names: set[str],
    longer: dict[str, list[str]] | None = None,
    limit: int = 2,
) -> list[dict]:
    """取出让 `chapter` 入选的那些单元里、含焦点名的原文行（供人工复核）。"""
    longer = longer or {}
    out: list[dict] = []
    for unit in units:
        if str(unit.get("chapter") or "").strip() != chapter:
            continue
        text = str(unit.get("text") or "")
        ents = [str(e) for e in (unit.get("entities") or [])]
        matched = [n for n in sorted(names) if n and n in text]
        if not matched:
            continue
        standalone = {
            n: standalone_occurs(text, n, longer.get(n) or []) for n in matched
        }
        lines = [ln for ln in text.splitlines() if any(n in ln for n in matched)]
        out.append(
            {
                "chunk_id": str(unit.get("chunk_id") or ""),
                "unit_chapter": chapter,
                "names_in_text": matched,
                "standalone": standalone,
                "unit_entities_has_focus": bool(names & set(ents)),
                "sample_lines": [ln[:80] for ln in lines[:limit]],
            }
        )
        if len(out) >= limit:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="6000,12000")
    ap.add_argument("--json", default=str(OUT_JSON))
    ap.add_argument("--generation", default=GENERATION_ID)
    args = ap.parse_args()
    budgets = [int(x) for x in str(args.budgets).split(",") if x.strip()]

    gold_doc = json.loads(GOLD_PATH.read_text("utf-8"))
    items = [
        it
        for it in gold_doc.get("items") or []
        if it.get("split") == "dev" and it.get("category") == "cross_chunk"
    ]
    items.sort(key=lambda it: str(it.get("question_id")))
    total = len(items)
    print(f"[load] cross_chunk dev items: {total}")

    loaded = pipeline_mod.load_generation(args.generation)
    chunks = loaded["chunks"]
    entities = loaded["graph"].get("entities") or []
    by_name: dict[str, list[dict]] = defaultdict(list)
    for ent in entities:
        by_name[norm_name(ent.get("name"))].append(ent)
    chapter_text = build_chapter_text_index(chunks)
    chunk_chapter = {cid: str(rec.get("chapter") or "") for cid, rec in chunks.items()}
    chapter_code_set = {c for c in chunk_chapter.values() if c}

    # 更长的实体名索引：用来识别「帕特」命中「帕特里奇昂」这类跨实体子串误配。
    all_ent_names = sorted(
        {str(e.get("name")) for e in entities if e.get("name")}, key=len
    )
    longer_map: dict[str, list[str]] = {
        nm: [o for o in all_ent_names if o != nm and nm in o] for nm in all_ent_names
    }

    pipe = pipeline_mod.build_pipeline(loaded, top_k=20, graph_depth=1)
    pipe["generation_id"] = args.generation
    pipe["entities"] = entities
    retriever = pipe["retriever"]
    print(f"[pipe] top_k={pipe['top_k']} graph_depth={pipe['graph_depth']} "
          f"chapter_path={answer_mod.CHAPTER_ANSWER_PATH}")

    # ---- 每题的真值三口径 + 单题元数据 -------------------------------------
    meta: dict[str, dict] = {}
    q1_name_hit = 0
    q1_all_hit = 0
    q1_text_hit = 0
    for it in items:
        qid = str(it.get("question_id"))
        names = QUOTE_RE.findall(it.get("question") or "")
        name = norm_name(names[0]) if names else ""
        cand = sorted(by_name.get(name) or [], key=lambda e: str(e.get("id")))
        gold = sorted(split_chapters(it.get("answer")))
        rec = {
            "question_id": qid,
            "focus_name": name,
            "gold": gold,
            "entity_ids": [str(e.get("id")) for e in cand],
            "G_name": [],
            "G_all": [],
            "C_text": [],
            "n_entity_chunks": 0,
            "entity_chunk_ids": [],
        }
        if cand:
            ent = cand[0]
            ents_chunks = [str(c) for c in ent.get("chunks") or ()]
            rec["n_entity_chunks"] = len(ents_chunks)
            rec["entity_chunk_ids"] = ents_chunks
            g_all = sorted(
                {chunk_chapter[c] for c in ents_chunks if chunk_chapter.get(c)}
            )
            g_name = sorted(
                {
                    chunk_chapter[c]
                    for c in ents_chunks
                    if chunk_chapter.get(c) and name in str((chunks.get(c) or {}).get("text") or "")
                }
            )
            rec["G_all"] = g_all
            rec["G_name"] = g_name
        rec["C_text"] = sorted(ch for ch, text in chapter_text.items() if name and name in text)
        if set(rec["G_name"]) == set(gold):
            q1_name_hit += 1
        if set(rec["G_all"]) == set(gold):
            q1_all_hit += 1
        if set(rec["C_text"]) == set(gold):
            q1_text_hit += 1
        meta[qid] = rec

    print()
    print("== Q1: gold vs graph truth ==")
    print("  " + rank(q1_name_hit, total, "gold == G_name (entity chunks AND name in text)"))
    print("  " + rank(q1_all_hit, total, "gold == G_all (entity chunks, no text filter)"))
    print("  " + rank(q1_text_hit, total, "gold == C_text (any chunk text in chapter)"))

    # ---- 单边判据的可作弊性（用 gold 自身穷举，不是实测产出） ---------------
    # 这一节证明 superset / subset / 只列一章 这几条判据不能当验收：退化解都能拿高分。
    corpus_chapters = sorted(set(chunk_chapter.values()))
    game_super = game_sub = game_equal = game_super_equal = 0
    for qid, md in meta.items():
        gold = set(md["gold"])
        game_super += int(bool(gold) and gold <= set(corpus_chapters))
        game_super_equal += int(bool(gold) and set(corpus_chapters) == gold)
        first = sorted(gold)[0] if gold else ""
        game_sub += int(bool(first) and {first} <= gold)
        game_equal += int(bool(first) and {first} == gold)

    print("  -- degenerate answers scored by one-sided judges (proves they are not usable) --")
    print("  " + rank(game_super, total, "list_all_chapters_in_corpus passes [superset]"))
    print("  " + rank(game_sub, total, "list_one_chapter_any_gold passes [subset]"))
    print("  " + rank(game_equal, total, "list_one_chapter_any_gold passes [set_equal]"))
    print("  " + rank(
        game_super_equal, total, "list_all_chapters_in_corpus passes [set_equal]"
    ))
    print("  " + rank(
        max(game_super_equal, game_equal), total,
        "best_degenerate_answer passes [final_criterion (set_equal core)]"
    ))

    # ---- 两档预算下的生产产出 + 越界章节逐章分类 ---------------------------
    per_budget: dict[str, dict] = {}
    for budget in budgets:
        retriever.budget["max_context_tokens"] = int(budget)
        bkey = str(budget)
        rows: list[dict] = []

        m_equal = m_super = m_sub = 0
        m_cite_each = 0      # 相等 + 每章都有引用
        m_cite_name = 0      # 相等 + 每章引用行含焦点名
        m_proposed = 0       # 相等 + 每章引用行含焦点名 + 引用偏移自检通过
        m_strict = 0         # proposed + 答案全部 token 都是真实章节码
        n_bad_tokens = 0
        m_abstain = 0
        m_path = 0
        m_ceiling = 0
        n_ceiling_fail = 0
        ceiling_recall_sum = 0
        ceiling_denom_sum = 0
        cond_denom = cond_exact = cond_v1_exact = 0
        gp_recall_hit = gp_recall_den = gp_recall_v1_hit = 0
        gp_prec_bad = gp_prec_den = 0
        extra_not_in_ctx = 0
        cls_totals: dict[str, int] = defaultdict(int)
        vscore: dict[str, dict[str, int]] = {
            v: {"equal": 0, "super": 0, "n_extra": 0, "n_miss": 0, "n_listed": 0}
            for v in (
                "v0_text",
                "v1_text_attributed",
                "v2_text_assertion",
                "v3_attributed_only",
                "v4_text_standalone",
                "v5_text_standalone_attributed",
                "v6_graph_chunks",
                "v7_graph_chunks_named",
                "v8_graph_chapters_in_ctx",
                "v9_graph_chapters_in_ctx_named",
            )
        }
        n_extra_q = 0
        n_miss_q = 0
        extra_sum = 0
        miss_sum = 0
        list_sum = 0

        for it in items:
            qid = str(it.get("question_id"))
            md = meta[qid]
            gold = set(md["gold"])
            g_name = set(md["G_name"])
            g_all = set(md["G_all"])

            result = run_one(pipe, it.get("question") or "")
            ans = result["answer"]
            units = result.get("context", {}).get("units") or []
            qa = result.get("query_analysis") or {}
            stats = (ans.get("stats") or {}).get("chapter_path") or {}
            applied = bool(stats.get("applied"))

            answer_text = str(ans.get("answer") or "")
            got = split_chapters(answer_text)
            gotset = set(got)
            abstained = bool(ans.get("abstained"))
            if abstained:
                m_abstain += 1
            if applied:
                m_path += 1

            exact = (not abstained) and gotset == gold and bool(gotset)
            superset = (not abstained) and gold <= gotset and bool(gotset)
            subset = (not abstained) and gotset <= gold and bool(gotset)
            m_equal += int(exact)
            m_super += int(superset)
            m_sub += int(subset)

            # 引用 -> 章节
            cites = ans.get("citations") or []
            cite_by_ch: dict[str, list[dict]] = defaultdict(list)
            cite_ok_by_ch: dict[str, list[dict]] = defaultdict(list)
            cite_bad = 0
            for ci in cites:
                cid = str(ci.get("chunk_id") or "")
                ch = chunk_chapter.get(cid, "")
                rec = chunks.get(cid) or {}
                text = str(rec.get("text") or "")
                start, end = int(ci.get("start", -1)), int(ci.get("end", -1))
                ok = (
                    rec is not None
                    and 0 <= start < end <= len(text)
                    and text[start:end] == str(ci.get("evidence") or "")
                )
                cite_by_ch[ch].append({**ci, "ok": ok})
                if ok:
                    cite_ok_by_ch[ch].append({**ci, "ok": ok})
                else:
                    cite_bad += 1

            names = {norm_name(c.get("name")) for c in (qa.get("entity_candidates") or [])[:2]}
            names.discard("")

            def has_name(ch: str, pool: dict[str, list[dict]]) -> bool:
                for ci in pool.get(ch) or []:
                    ev = str(ci.get("evidence") or "")
                    if any(n and n in ev for n in names):
                        return True
                return False

            each_ok = exact and all(c in cite_by_ch for c in gotset)
            name_ok = exact and all(has_name(c, cite_by_ch) for c in gotset)
            proposed = exact and all(has_name(c, cite_ok_by_ch) for c in gotset) and cite_bad == 0

            # ---- 待定判据：集合相等 + 每章有引用 + 引用偏移自检 + 答案全为真章节码 ----
            # 前四项即 proposed。第五项 additional: answer 里不得出现非章节码的碎片。
            tokens = [t for t in re.split(r"[、,，/;\s]+", answer_text) if t.strip()]
            tokens_valid = bool(tokens) and all(t in chapter_code_set for t in tokens)
            strict_proposed = proposed and tokens_valid
            m_strict += int(strict_proposed)
            if not tokens_valid:
                n_bad_tokens += 1
            m_cite_each += int(each_ok)
            m_cite_name += int(name_ok)
            m_proposed += int(proposed)

            extra = sorted(gotset - gold)
            miss = sorted(gold - gotset)

            # 上下文覆盖到的章节（供 v8/v9 与天花板共用）
            unit_chapters = {str(u.get("chapter") or "").strip() for u in units}
            unit_chapters.discard("")

            # ---- 通路变体网格：同一批单元上重放不同严度 ------------------
            variants = {
                "v0_text": variant_chapters(units, names, longer=longer_map),
                "v1_text_attributed": variant_chapters(
                    units, names, require_attributed=True, longer=longer_map
                ),
                "v2_text_assertion": variant_chapters(
                    units, names, require_assertion=True, longer=longer_map
                ),
                "v3_attributed_only": variant_chapters(
                    units,
                    names,
                    require_text=False,
                    require_attributed=True,
                    longer=longer_map,
                ),
                "v4_text_standalone": variant_chapters(
                    units, names, require_standalone=True, longer=longer_map
                ),
                "v5_text_standalone_attributed": variant_chapters(
                    units,
                    names,
                    require_standalone=True,
                    require_attributed=True,
                    longer=longer_map,
                ),
                # v6/v7: 不用稀疏的 unit["entities"]，改用图实体归属（entity.chunks）。
                # 这正是 gold 的生成口径，且只受"上下文里有没有该块"限制。
                "v6_graph_chunks": variant_chapters(
                    units, names, allowed_chunks=set(md["entity_chunk_ids"])
                ),
                "v7_graph_chunks_named": variant_chapters(
                    units,
                    names,
                    allowed_chunks=set(md["entity_chunk_ids"]),
                    require_text=True,
                ),
                # v8/v9: 图实体的**章节级**归属。G_name 是"实体 chunks 里正文含
                # 规范名的块所属章节"集合；这里只保留**上下文里确实有单元**的那些
                # 章节（否则无法给出引用），不再要求单元 chunk_id 本身在 entity.chunks
                # 里 —— v6 的块级白名单过严，会把"该章有实体块、但上下文抽到的是
                # 该章另一个块"的真章节误杀。
                "v8_graph_chapters_in_ctx": sorted(
                    set(md["G_name"]) & set(unit_chapters)
                ),
                "v9_graph_chapters_in_ctx_named": variant_chapters(
                    units,
                    names,
                    allowed_chunks=None,
                    require_text=True,
                    longer=longer_map,
                    chapter_whitelist=set(md["G_name"]),
                ),
            }

            # ---- 抽取式通路的上下文天花板 --------------------------------
            # 上下文里根本没出现的 gold 章节，任何"只读上下文"的章节过滤器都拿不到。
            # 天花板 = gold 中有多少章节至少有一个单元落在该章。
            unit_chapters = {str(u.get("chapter") or "").strip() for u in units}
            unit_chapters.discard("")
            gold_in_ctx = sorted(gold & unit_chapters)
            gold_no_unit = sorted(gold - unit_chapters)
            ceiling_ok = bool(gold) and not gold_no_unit
            m_ceiling += int(ceiling_ok)
            ceiling_recall_sum += len(gold_in_ctx)
            ceiling_denom_sum += len(gold)
            if gold_no_unit:
                n_ceiling_fail += 1

            # ---- 条件在"上下文够不够"之下的通路效率 ----------------------
            # ceiling_ok = 该题所有 gold 章节都在上下文里；此时分数纯由通路判据决定。
            if ceiling_ok:
                cond_denom += 1
                cond_exact += int(exact)
                cond_v1_exact += int(
                    bool(set(variants["v1_text_attributed"]))
                    and set(variants["v1_text_attributed"]) == gold
                )
            # 章节级：上下文可及的 gold 章节有多少被列出（通路召回）
            gp_recall_hit += len(set(gold_in_ctx) & gotset)
            gp_recall_den += len(gold_in_ctx)
            gp_recall_v1_hit += len(set(gold_in_ctx) & set(variants["v1_text_attributed"]))
            # 章节级：上下文可及但不在 gold 的章节有多少被误列（通路精度）
            ctx_non_gold = unit_chapters - gold
            gp_prec_bad += len(gotset & ctx_non_gold)
            gp_prec_den += len(extra)
            extra_not_in_ctx += len([c for c in extra if c not in unit_chapters])
            if extra:
                n_extra_q += 1
            if miss:
                n_miss_q += 1
            extra_sum += len(extra)
            miss_sum += len(miss)
            list_sum += len(gotset)

            extras_detail = []
            q_classes: set[str] = set()
            for ch in extra:
                chap_text = chapter_text.get(ch, "")
                # 分类按**单元级**判定（生产通路就是按单元入选的），
                # 章节级拼接文本只作为参考，避免"别处有过一次独立命中"
                # 就把整章算成真章节。
                chap_units = [
                    u for u in units if str(u.get("chapter") or "").strip() == ch
                ]
                unit_standalone = any(
                    standalone_occurs(
                        str(u.get("text") or ""),
                        md["focus_name"],
                        longer_map.get(md["focus_name"]) or [],
                    )
                    for u in chap_units
                )
                standalone = unit_standalone
                if ch in g_name:
                    kind = "gold_gap"
                elif ch in g_all:
                    kind = "alias_only"
                elif standalone:
                    kind = "unattributed_standalone"
                elif md["focus_name"] and md["focus_name"] in chap_text:
                    kind = "substring_other_entity"
                else:
                    kind = "unrelated"
                cls_totals[kind] += 1
                q_classes.add(kind)
                matched = [n for n in sorted(names) if n and n in chap_text]
                extras_detail.append(
                    {
                        "chapter": ch,
                        "kind": kind,
                        "name_standalone": standalone,
                        "names_in_chapter_text": matched,
                        "cite": bool(cite_by_ch.get(ch)),
                        "unit_evidence": find_extra_evidence(
                            units, ch, names, longer_map
                        ),
                    }
                )

            # ---- 通路变体网格：同一批单元上重放不同严度 ------------------
            for vname, vlist in variants.items():
                vset = set(vlist)
                vscore[vname]["equal"] += int(bool(vset) and vset == gold)
                vscore[vname]["super"] += int(bool(vset) and gold <= vset)
                vscore[vname]["n_extra"] += len(vset - gold)
                vscore[vname]["n_miss"] += len(gold - vset)
                vscore[vname]["n_listed"] += len(vset)

            rows.append(
                {
                    "question_id": qid,
                    "focus_name": md["focus_name"],
                    "gold": md["gold"],
                    "answer": answer_text,
                    "answer_chapters": got,
                    "abstained": abstained,
                    "abstain_reason": str(ans.get("abstain_reason") or ""),
                    "chapter_path_applied": applied,
                    "n_units": len(units),
                    "tokens_used": (result.get("context", {}).get("stats") or {}).get("tokens_used"),
                    "token_budget": budget,
                    "exact": exact,
                    "superset": superset,
                    "subset": subset,
                    "cite_each_ok": each_ok,
                    "cite_name_ok": name_ok,
                    "proposed_ok": proposed,
                    "variants": {k: v for k, v in variants.items()},
                    "n_citations": len(cites),
                    "cite_bad": cite_bad,
                    "extra": extra,
                    "extra_detail": extras_detail,
                    "extra_classes": sorted(q_classes),
                    "miss": miss,
                    "gold_in_ctx": gold_in_ctx,
                    "gold_no_unit": gold_no_unit,
                    "ceiling_ok": ceiling_ok,
                    "g_name": md["G_name"],
                    "g_all": md["G_all"],
                    "c_text_size": len(md["C_text"]),
                    "merged_from": [str(it.get("question_id"))],
                }
            )

        n = total
        per_budget[bkey] = {
            "budget": budget,
            "metrics": {
                "set_equal": rank(m_equal, n, "set_equal"),
                "superset": rank(m_super, n, "superset"),
                "subset": rank(m_sub, n, "subset"),
                "equal_and_cite_each": rank(m_cite_each, n, "equal_and_cite_each"),
                "equal_and_cite_line_has_focus": rank(m_cite_name, n, "equal_and_cite_line_has_focus"),
                "proposed": rank(m_proposed, n, "proposed_cross_chunk"),
                "final_criterion": rank(m_strict, n, "final_criterion_cross_chunk"),
                "answer_token_not_a_chapter_code": rank(
                    n_bad_tokens, n, "answer_token_not_a_chapter_code"
                ),
                "abstained": rank(m_abstain, n, "abstained"),
                "chapter_path_applied": rank(m_path, n, "chapter_path_applied"),
                "ceiling_all_gold_chapters_in_context": rank(
                    m_ceiling, n, "ceiling_all_gold_chapters_in_context"
                ),
            },
            "ceiling": {
                "gold_chapters_in_context": f"{ceiling_recall_sum}/{ceiling_denom_sum}",
                "gold_chapters_in_context_rate": (
                    round(ceiling_recall_sum / ceiling_denom_sum, 4)
                    if ceiling_denom_sum
                    else 0.0
                ),
                "n_questions_blocked_by_context": n_ceiling_fail,
            },
            "conditional": {
                "on_reachable_questions_set_equal": rank(
                    cond_exact, cond_denom, "on_reachable_questions_set_equal"
                ),
                "on_reachable_questions_v1_exact": rank(
                    cond_v1_exact, cond_denom, "on_reachable_questions_v1_exact"
                ),
                "reachable_gold_chapter_recall_v0": rank(
                    gp_recall_hit, gp_recall_den, "reachable_gold_chapter_recall_v0"
                ),
                "reachable_gold_chapter_recall_v1": rank(
                    gp_recall_v1_hit, gp_recall_den, "reachable_gold_chapter_recall_v1"
                ),
                "extra_chapters_with_unit_in_context": rank(
                    gp_prec_bad, gp_prec_den, "extra_chapters_with_unit_in_context"
                ),
                "extra_chapters_absent_from_context": rank(
                    extra_not_in_ctx, gp_prec_den, "extra_chapters_absent_from_context"
                ),
            },
            "extra_class_totals": dict(cls_totals),
            "variants": {
                v: {
                    "set_equal": rank(vs["equal"], n, v),
                    "superset": rank(vs["super"], n, v + "_superset"),
                    "mean_listed": round(vs["n_listed"] / n, 4) if n else 0.0,
                    "mean_extra": round(vs["n_extra"] / n, 4) if n else 0.0,
                    "mean_miss": round(vs["n_miss"] / n, 4) if n else 0.0,
                }
                for v, vs in vscore.items()
            },
            "n_questions_with_extra": n_extra_q,
            "n_questions_with_miss": n_miss_q,
            "mean_listed": round(list_sum / n, 4) if n else 0.0,
            "mean_extra": round(extra_sum / n, 4) if n else 0.0,
            "mean_miss": round(miss_sum / n, 4) if n else 0.0,
            "rows": rows,
        }

        print()
        print(f"== budget {budget} ==")
        for k, v in per_budget[bkey]["metrics"].items():
            print(f"  {v}")
        print(f"  extra class totals: {json.dumps(dict(cls_totals), ensure_ascii=True)}")
        print(f"  ceiling: gold chapters reachable in context = "
              f"{per_budget[bkey]['ceiling']['gold_chapters_in_context']} "
              f"({per_budget[bkey]['ceiling']['gold_chapters_in_context_rate']}), "
              f"questions blocked by context = "
              f"{per_budget[bkey]['ceiling']['n_questions_blocked_by_context']}/{n}")
        print(f"  questions with extra={n_extra_q} with_miss={n_miss_q} "
              f"mean_listed={per_budget[bkey]['mean_listed']} "
              f"mean_extra={per_budget[bkey]['mean_extra']} "
              f"mean_miss={per_budget[bkey]['mean_miss']}")
        print("  -- chapter-attribution variants (same units, stricter gate) --")
        for v, vd in per_budget[bkey]["variants"].items():
            print(f"    {vd['set_equal']} (superset {vd['superset']}) "
                  f"listed={vd['mean_listed']} extra={vd['mean_extra']} miss={vd['mean_miss']}")
        print("  -- conditional on reachable context --")
        for k, v in per_budget[bkey]["conditional"].items():
            print(f"    {v}")

    # ---- 与 gold 不一致的题逐题清单（Q1 反例） ----------------------------
    q1_counterexamples = []
    for qid, md in meta.items():
        if set(md["G_name"]) != set(md["gold"]) or set(md["G_all"]) != set(md["gold"]):
            q1_counterexamples.append(
                {
                    "question_id": qid,
                    "focus_name": md["focus_name"],
                    "gold": md["gold"],
                    "G_name": md["G_name"],
                    "G_all": md["G_all"],
                    "extra_in_G_all_not_in_gold": sorted(
                        set(md["G_all"]) - set(md["gold"])
                    ),
                    "extra_in_G_name_not_in_gold": sorted(
                        set(md["G_name"]) - set(md["gold"])
                    ),
                }
            )

    payload = {
        "generation_id": args.generation,
        "gold_set": str(GOLD_PATH),
        "category": "cross_chunk",
        "split": "dev",
        "n_items": total,
        "q1": {
            "gold_equals_G_name": rank(q1_name_hit, total, "gold == G_name"),
            "gold_equals_G_all": rank(q1_all_hit, total, "gold == G_all"),
            "gold_equals_C_text": rank(q1_text_hit, total, "gold == C_text"),
            "counterexamples": q1_counterexamples,
            "degenerate": {
                "corpus_n_chapters": len(corpus_chapters),
                "list_all_chapters_in_corpus_passes_superset": rank(
                    game_super, total, "list_all_chapters_in_corpus passes superset"
                ),
                "list_one_chapter_any_gold_passes_subset": rank(
                    game_sub, total, "list_one_chapter_any_gold passes subset"
                ),
                "list_one_chapter_any_gold_passes_set_equal": rank(
                    game_equal, total, "list_one_chapter_any_gold passes set_equal"
                ),
                "list_all_chapters_in_corpus_passes_set_equal": rank(
                    game_super_equal, total,
                    "list_all_chapters_in_corpus passes set_equal"
                ),
            },
        },
        "per_budget": per_budget,
        "items_meta": list(meta.values()),
        "criterion": {
            "name": "cross_chunk 集合相等 + 逐章引用支撑",
            "pseudo_assert": (
                "not ans.abstained and "
                "chapters(ans.answer) == chapters(item.answer) and "
                "for each ch in chapters(ans.answer): "
                "exists cite in ans.citations with chunk_chapter(cite.chunk_id)==ch "
                "and focus_name in cite.evidence "
                "and chunk_text[cite.start:cite.end]==cite.evidence"
            ),
        },
        "criterion_final": {
            "name": "final_criterion_cross_chunk",
            "definition_zh": (
                "answers 被解析成章节码集合 S；判据要求以下五项同时成立："
                "(1) 未拒答；(2) S 非空；(3) S == set(gold 的章节集合)，双向相等，"
                "多列和漏列都判错；(4) S 中每个章节都至少有一条引用落在该章节"
                "（chunk_chapter(cite.chunk_id) == ch）；(5) 每条引用的 "
                "chunk.text[cite.start:cite.end] == cite.evidence，且该引用行含焦点实体名。"
                "另加：答案被切分后的每个 token 必须本身是一个合法章节码，"
                "禁止在答案里塞非章节码碎片来靠模糊匹配蒙过去。"
            ),
            "machine_checkable": [
                "ans = ask_once(pipeline, item.question)['answer']",
                "S = set(t for t in re.split(r'[、,，/;\\s]+', ans['answer']) if t)",
                "assert not ans['abstained']",
                "assert S != set()",
                "assert S == set(split_chapters(item['answer']))",
                "assert S <= chapter_code_set",
                "for ch in S:",
                "    ok = [c for c in ans['citations']",
                "          if chunk_chapter[c['chunk_id']] == ch",
                "          and chunk_text(c['chunk_id'])[c['start']:c['end']] == c['evidence']",
                "          and item_focus_name in c['evidence']]",
                "    assert ok",
            ],
            "why_not_cheatable": (
                "把语料全部章节列出来 -> S 是真超集，第 (3) 项失败；"
                "只列一章 -> S 是真子集，第 (3) 项失败。"
                "两项退化解在第 (3) 项同时被挡（实测 38/38 超集退化会通过旧 superset 判据，"
                "但在本判据下 0/38 通过）。多列必须逐章有引用支撑，凭空补章节过不了第 (4)(5) 项。"
            ),
        },
    }
    outp = Path(args.json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"[write] {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
