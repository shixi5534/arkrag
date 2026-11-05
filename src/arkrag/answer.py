"""答案生成 —— 计划 G6「抽取式回答 + 引用校验」。

## 为什么默认是抽取式，而不是"调模型"

本机没有 LLM 端点（`ARKRAG_LLM_MODEL` / `ARKRAG_LLM_BASE_URL` 皆空，`arkrag.llm`
缺配置直接抛 `RuntimeError`）。更要紧的是：**引用可校验**这件事不能依赖模型愿意
守规矩。抽取式回答的每一条事实都直接来自上下文里的断言，引用偏移是算出来的而不是
生成的，所以"引用是否支持结论"能机检。`LLMAnswerer` 留了接口，但它只是把同一份
上下文交给模型再走同一套校验——校验不过就降置信度，不给它"因为说得像"就通过的机会。

## 六条硬规则（每条都有一条测试喂坏输入把它打红）

1. **关键事实必须有引用。** 没有引用的断言不进 `answer`。理由不是洁癖：一句没有
   出处的剧情断言，读者无法复核，作者也无法否认。
2. **引用必须支持对应事实。** `claim` 的两个端点名里**至少一个**必须出现在
   `evidence` 所在的**整行**（含说话人前缀）里。这条按 `docs/goldset-v1说明.md`
   第 10 条的实测结论定的：48 条多跳题每一条都至少有一条引用所在整行含 A/B/C 之一
   （0 条例外），但没有一条要求两个端点同处一行——台词常常只说一方。要求「两个
   端点都必须在这一行」会把合法引用整批判死（用真实上下文实测时 4 条 claim 全被
   误杀，答案退化成空）。一个端点都不在才是「引用不支持结论」。
   谓词**不**要求字面出现：`norm_relation.adjudicate` 会把原文措辞归一成规范谓词
   （「我会听你的」→`指挥`），要求规范名出现在原文里会否掉大量合法引用。谓词是否
   字面出现如实记进 `predicate_in_line`，并计入 `warnings`，但不因此丢引用。
3. **检索不足就拒答。** `hits` 为空或最高分低于阈值 → `abstained=True` + `reason`。
   硬答一句"根据原文……"是本项目最不能接受的一种自信。
4. **不许用模型常识补齐剧情事实。** 答案里出现的每个实体名都必须在上下文里出现过
   （`_audit_answer_entities` 逐步核对）。这条拦的是"顺手补一句世界观常识"这类
   最像正确答案的错误。
5. **冲突材料不得无条件合并。** 同一对实体在同一场景里既「同盟」又「敌对」时，
   两条都进 `warnings` 点名，答案里**不写成一句确定的话**。
6. **记录 `generation_id` 与 prompt/context 版本。** 没有这些，"这条答案是哪一版
   数据产出的"就无从回答，评测也无法复现。

## `verify_citations` 是硬闸

逐条查：chunk 存在、偏移合法（`0 <= start < end <= len(text)`）、
`text[start:end] == evidence`。任何一条不过，**绝不返回高置信答案**：降
`confidence`、置 `warnings`、并把不合格的引用从 `citations` 里剔除。
"""
from __future__ import annotations

import hashlib
import json
import re
import time

# 引用校验的最低证据长度。1–2 字的证据在正文里几乎必然重复出现，位置对不上也
# 看不出问题，属于"看起来有引用"。
MIN_EVIDENCE_CHARS = 4

# 判定"检索不足"的默认阈值。取 0.05 而不是更高：`search()` 把两路分数各自按该路
# 最高分归一，所以只要有任何一条硬命中，top_score 至少约为 bm25 的 1.0 份之一；
# 低于 0.05 基本等于"全库没有一块与问句有词面重合"。
DEFAULT_MIN_TOP_SCORE = 0.05

# 答案里最多陈述几条事实。多于此数时按置信度与 support_count 取前若干，其余进
# `warnings`——一条答案堆二十句话，读者无法判断哪句是重点。
MAX_CLAIMS = 5

# 引用粒度开关：False = 发断言短语本身（现状），True = 发断言所在整行。
# 只影响 `citations` 里 `evidence/start/end` 三元组，**不影响** claim 的取舍、
# `_claim_supported_by_line` 的输入、预算与条数上限——那些都读断言 span，不读这里。
CITATION_WHOLE_LINE = False

# 标题型答案通路开关：False = 不启用（与改动前逐位一致），True = 问句在问「出处标题」
# 时，用问句里的「」引文定位上下文中的块，直接回答该块的 title。
#
# **为什么需要这条通路**：抽取式回答器只会拼「主体 谓词 客体」三元组，而这类问法的
# 答案是某个 chunk 的 title——title 既不在块正文里，也不在断言里，claim 机制结构上
# 永远答不出。实测（dev 350 应答题）这类题 76 道，100% 是 single_chunk。
#
# **为什么默认关**：这是**新增一种答案形态**，不是改判据。关掉即与改动前逐位一致。
#
# 硬约束（改这条通路时不许违反）：
#   1. 答案文本**只能**取自 `unit["title"]`。question 参数仅用于提取「」引文去
#      **定位块**，绝不许从中切片/正则出答案字符。
#   2. 引用 evidence 取「引文所在那一行」，且保证 `text[start:end] == evidence`，
#      否则 `verify_citations` 会判 `evidence_does_not_match_offsets`。
#   3. 命中块的 title 不唯一（>1 种）或为空 ⇒ 不触发，落回原路径（不许瞎猜）。
#   4. 不动 CITATION_WHOLE_LINE、不动 MAX_CLAIMS、不动 min_top_score、不动 retrieve。
#
# **记账提醒**：本通路启用后，系统里会同时存在两种引用粒度——标题题走整行（本通路
# 内建）、其余题走断言 span（`CITATION_WHOLE_LINE` 仍为 False）。这不违反单变量
# （粒度差异是本通路的实现内含，不是又拧了一个旋钮），但报告里必须把「整行引用条数」
# 与「断言 span 引用条数」分开计，且下一轮若有人打开 `CITATION_WHOLE_LINE`，那就是
# 在「已经部分整行」的基础上再打开，届时引用覆盖数不能与本轮直比。
TITLE_ANSWER_PATH = False

# 触发词与 `answer_form == needs_metadata` 判据完全等价（实测 76/76 全 single_chunk）。
# 刻意收窄：不加「引文」「原文」等泛词，否则会把 alias 类「「X」的规范名是什么」
# 一起吃进来（dev 350 里有 158 道含「」引文的非标题题，必须一道都不能触发）。
TITLE_TRIGGERS = (
    "请给出该块的标题",
    "请回答该段的标题",
    "出自哪个文本块",
    "来自哪一段剧情",
)

# 问句里的「」引文（≥4 字，与 MIN_EVIDENCE_CHARS 对齐：更短的行无法作为引用）。
_QUOTE_RE = re.compile(r"「([^」]{4,})」")


def _line_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """把 `[start, end)` 扩到所在整行（含说话人前缀）。"""
    if not text:
        return 0, 0
    start = max(0, min(start, len(text)))
    end = max(start, min(end, len(text)))
    left = text.rfind("\n", 0, start) + 1
    right = text.find("\n", end)
    if right < 0:
        right = len(text)
    return left, right


def _line_of(text: str, start: int, end: int) -> str:
    left, right = _line_bounds(text, start, end)
    return text[left:right]


def _title_answer(question: str, units: list[dict]) -> dict | None:
    """问句在问「出处标题」时，用「」引文定位块，返回该块 title 与引文行引用。

    **答案只能取自 `unit["title"]`**；`question` 在这里只有两个用途：判触发词、
    取「」引文当**定位键**。函数体内没有任何对 question 的切片/替换/partition。

    返回 `None` 表示不适用（落回原有 claim 通路）——定位失败、title 为空、title
    不唯一，全部返回 None，**绝不猜一个 title 出来**（`any_predicate` 那条路正是
    死在「判据不成立时仍然给答案」上）。
    """
    q = str(question or "")
    if not any(trigger in q for trigger in TITLE_TRIGGERS):
        return None
    quotes = [m for m in _QUOTE_RE.findall(q) if m.strip()]
    if not quotes:
        return None

    located: list[tuple[dict, int, int, str]] = []
    for unit in units:
        title = str(unit.get("title") or "").strip()
        if not title:
            continue
        text = str(unit.get("text") or "")
        if not text:
            continue
        for line in text.splitlines():
            if not line.strip() or len(line) < MIN_EVIDENCE_CHARS:
                continue
            if not any(quote in line for quote in quotes):
                continue
            start = text.find(line)
            # 偏移自检：同一行文本可能在块内重复出现，find 取第一个可能定位到别处。
            if start < 0 or text[start:start + len(line)] != line:
                continue
            located.append((unit, start, start + len(line), line))
            break

    if not located:
        return None

    titles = {str(unit.get("title") or "").strip() for unit, _, _, _ in located}
    if len(titles) != 1:
        return None
    title = titles.pop()

    # **同一 title 的每一个命中块都要各发一条引用**，不能只取 located[0]。
    #
    # 原因（实测）：同一行文本可能在**同 title 的两个块**里逐字节重复。对话体语料里
    # 「上一块的结尾」与「下一块的开头」常是同一句台词，两块 title 也可能相同。这时
    # 只发第一条会把引用发到非 gold 块上，required 的 chunk_id 对不上，覆盖率回退
    # （`q-sc-000064` 就是这么掉的）。每条引用各自满足 `text[start:end] == evidence`，
    # 所以多发不影响引用定位校验。
    citations = [
        {
            "chunk_id": str(unit.get("chunk_id") or ""),
            "start": start,
            "end": end,
            "evidence": line,
        }
        for unit, start, end, line in located
    ]
    return {
        "answer": title,
        "citations": citations,
        "n_located": len(located),
    }


def _citation_span(text: str, start: int, end: int, *, whole_line: bool = False):
    """一条引用要发出的 `(start, end, evidence)` 三元组。

    `verify_citations` 逐条查 `text[start:end] == evidence`，所以这三样必须自洽：
    换区间就必须同时换 evidence，只改其中一个必然被判 `evidence_does_not_match_offsets`。

    `whole_line=True` 时把区间扩到断言所在**整行**（含说话人前缀），evidence 取该行
    原文。这样 gold 的 `required_citations.evidence`（常是整句台词）落在这行内时，
    下游 `req.evidence in cite.evidence` 的单向子串判据就能成立。

    **为什么抽成函数**：让「改引用粒度」成为可 A/B 的单变量——只需替换这一个函数的
    `whole_line` 取值，claim 选择、`_claim_supported_by_line`、预算、MAX_CLAIMS 全不动。
    """
    if whole_line:
        left, right = _line_bounds(text, start, end)
        return left, right, text[left:right]
    return start, end, text[start:end]


def _claim_supported_by_line(claim: dict, text: str) -> dict:
    """判据 2 的机检形态。

    **要求：证据所在整行（含说话人前缀）里至少出现一个端点名。**

    为什么是「至少一个」而不是「全部」，依据是 `docs/goldset-v1说明.md` 第 10 条对
    真实语料的实测结论：48 条多跳题里每一条都至少有一条引用所在整行同时含
    A/B/C 之一（0 条例外），但**没有一条**要求两个端点都出现在同一行——台词本身
    常常只说一方。要求「全部端点在同一行」会把合法引用整批判死（用真实上下文实测
    时 4 条 claim 全部被这条误杀，答案退化成空）。

    端点一个都不在这一行时才是「引用不支持结论」：读者读完这一行，得不到这条断言
    讲的任何一个主角，那就是拿别处的句子冒充证据。

    返回 `{ok, line, endpoints_in_line, predicate_in_line, missing}`。
    """
    start = int(claim.get("start", -1))
    end = int(claim.get("end", -1))
    if start < 0 or end <= start or end > len(text):
        return {
            "ok": False,
            "line": "",
            "endpoints_in_line": [],
            "predicate_in_line": False,
            "missing": ["citation_offsets_invalid"],
        }
    line = _line_of(text, start, end)
    ends = [str(claim.get("subject") or ""), str(claim.get("object") or "")]
    ends = [name for name in ends if len(name) >= 2]
    found = [name for name in ends if name in line]
    predicate = str(claim.get("predicate") or "")
    return {
        "ok": bool(found),
        "line": line,
        "endpoints_in_line": found,
        "predicate_in_line": bool(predicate) and predicate in line,
        "missing": [name for name in ends if name not in found],
    }


def verify_citations(answer: dict, chunks) -> dict:
    """逐条校验引用，并把不合格的引用剔除。

    `chunks` 可以是 `{chunk_id: 块}` 或块列表。返回**新的** answer dict（不改入参），
    外加 `verification` 字段记录通过/剔除计数。校验失败时按模块 docstring 的规矩降
    置信度、写 warnings——**绝不返回高置信答案**。
    """
    if isinstance(chunks, dict):
        by_id = {str(k): v for k, v in chunks.items()}
    else:
        by_id = {str(r.get("chunk_id") or ""): r for r in chunks or ()}

    out = dict(answer)
    kept: list[dict] = []
    problems: list[dict] = []
    for cite in answer.get("citations") or []:
        cid = str(cite.get("chunk_id") or "")
        ev = str(cite.get("evidence") or "")
        start = int(cite.get("start", -1))
        end = int(cite.get("end", -1))
        rec = by_id.get(cid)
        reason = ""
        if not cid:
            reason = "missing_chunk_id"
        elif rec is None:
            reason = "unknown_chunk"
        elif len(ev) < MIN_EVIDENCE_CHARS:
            reason = f"evidence_too_short({len(ev)})"
        elif start < 0 or end <= start:
            reason = "offsets_missing"
        else:
            text = str(rec.get("text") or "")
            if end > len(text):
                reason = "offsets_out_of_range"
            elif text[start:end] != ev:
                reason = "evidence_does_not_match_offsets"
        if reason:
            problems.append({"chunk_id": cid, "evidence": ev, "reason": reason})
            continue
        kept.append(dict(cite))

    warnings = list(answer.get("warnings") or [])
    confidence = float(answer.get("confidence") or 0.0)
    if problems:
        # 引用校验失败时不允许保持原置信度。下降幅度按失败比例算，最狠降到 0.1。
        ratio = len(problems) / max(1, len(answer.get("citations") or []))
        confidence = max(0.1, confidence * (1.0 - ratio))
        warnings.append(f"引用校验失败 {len(problems)} 条，置信度降至 {confidence:.2f}")
        for item in problems[:5]:
            warnings.append(f"  不合规引用 {item['chunk_id']}: {item['reason']}")
    if not kept and answer.get("answer"):
        # 一条引用都不剩却还有答案正文，只能拒答——否则就是"无出处的断言"。
        out["answer"] = ""
        out["abstained"] = True
        out["abstain_reason"] = "no_valid_citation"
        warnings.append("所有引用都未通过校验，答案已清空并改为拒答")

    out["citations"] = kept
    out["confidence"] = round(confidence, 4)
    out["warnings"] = warnings
    out["verification"] = {
        "passed": len(kept),
        "rejected": len(problems),
        "problems": problems,
    }
    return out


def _claims_are_grounded(claims: list[dict], context_names: set[str]) -> list[str]:
    """判据 4 的机检形态：每条 claim 的端点名都必须出现在上下文里。

    上下文里的「实体名」来自两处：单元 `entities`（图扩展带出的端点名）与断言
    `subject/object`。两者都只包含真正进过上下文的名字，所以这条判据拦得住
    「顺手补一句没出现在材料里的世界观常识」——那句话的主语一定不在这集合里。

    局限如实记：普通词（非实体）无法与实体区分，本判据不做词性判断。
    """
    offenders: list[str] = []
    for claim in claims:
        for key in ("subject", "object"):
            name = str(claim.get(key) or "")
            if len(name) >= 2 and name not in context_names:
                offenders.append(f"{name}（{claim.get('assertion_id') or '?'}）")
    return offenders


class ExtractiveAnswerer:
    """离线抽取式回答器。不联网、不调模型、不读盘。

    参数
    ----
    min_top_score
        检索不足的阈值，见 `DEFAULT_MIN_TOP_SCORE`。
    max_claims
        答案里最多几条事实。
    """

    name = "extractive"

    def __init__(self, *, min_top_score: float = DEFAULT_MIN_TOP_SCORE,
                 max_claims: int = MAX_CLAIMS) -> None:
        self.min_top_score = float(min_top_score)
        self.max_claims = max(1, int(max_claims))

    # ---------------------------------------------------------------- 回答
    def answer(
        self,
        question: str,
        context: dict,
        *,
        generation_id: str = "",
        top_score: float = 0.0,
        query_analysis: dict | None = None,
        retrieval_stats: dict | None = None,
    ) -> dict:
        """把上下文拼成带引用的答案，或如实拒答。

        返回 `{answer, citations, confidence, abstained, abstain_reason, warnings,
        generation_id, question, context_digest, versions, stats}`。

        `versions` 里记的是**判据版本**（本模块常量 + 上下文摘要），不是模型版本：
        抽取式回答没有模型，硬写一个模型名属于虚构。
        """
        units = list(context.get("units") or [])
        warnings: list[str] = []
        qa = query_analysis or {}
        base = {
            "question": str(question or ""),
            "answer": "",
            "citations": [],
            "confidence": 0.0,
            "abstained": False,
            "abstain_reason": "",
            "warnings": warnings,
            "generation_id": str(generation_id or ""),
            "context_digest": str(context.get("context_digest") or ""),
            "versions": {
                "answerer": f"extractive:{self.name}:{MIN_EVIDENCE_CHARS}",
                "max_claims": self.max_claims,
                "min_top_score": self.min_top_score,
                "prompt_version": None,
                "context_units": len(units),
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            "stats": {
                "units": len(units),
                "candidates": 0,
                "accepted": 0,
                "dropped_unsupported": 0,
                "dropped_ungrounded": 0,
                "dropped_no_citation": 0,
                "top_score": round(float(top_score or 0.0), 6),
                "retrieval": dict(retrieval_stats or {}),
            },
        }

        # ---- 判据 3：检索不足直接拒答 -----------------------------------
        if not bool(qa.get("should_answer", True)):
            base["abstained"] = True
            base["abstain_reason"] = f"query_refused:{qa.get('reason') or 'not_answerable'}"
            return base
        if not units:
            base["abstained"] = True
            base["abstain_reason"] = "no_context_units"
            return base
        if float(top_score or 0.0) < self.min_top_score:
            base["abstained"] = True
            base["abstain_reason"] = (
                f"insufficient_retrieval(top_score {float(top_score):.4f} < "
                f"{self.min_top_score:.4f})"
            )
            return base

        # ---- 标题型答案通路（新增答案形态，默认关） ----------------------
        # 放在三道拒答闸之后、claim 机制之前：
        #   * 合法拒答（问句不可答 / 零上下文 / 检索分不足）不受影响 —— R3' 防的就是这个；
        #   * 命中时完全绕过 claim 生成、MAX_CLAIMS 截断、逐行支撑判据与接地判据，
        #     这就是「只改答案形态、不动 claim 选择」的单变量保证。
        if TITLE_ANSWER_PATH:
            hit = _title_answer(question, units)
            if hit is not None:
                base["answer"] = hit["answer"]
                base["citations"] = [
                    {
                        "claim": f"标题：{hit['answer']}",
                        "chunk_id": cite["chunk_id"],
                        "evidence": cite["evidence"],
                        "start": cite["start"],
                        "end": cite["end"],
                        "assertion_id": "",
                        "predicate_in_line": False,
                        "granularity": "line",  # 记账用：本通路发的是整行引用
                    }
                    for cite in hit["citations"]
                ]
                base["confidence"] = round(min(0.95, float(top_score or 0.0)), 4)
                base["stats"]["title_path"] = {
                    "applied": True,
                    "located_units": hit["n_located"],
                    "title_source": "unit.title",
                }
                return base
            base["stats"]["title_path"] = {"applied": False}

        # ---- 收集候选 claim（每条都从断言的证据偏移来） -------------------
        context_names: set[str] = set()
        for unit in units:
            for name in unit.get("entities") or []:
                if name:
                    context_names.add(str(name))
            text = str(unit.get("text") or "")
            for brief in unit.get("assertions") or []:
                for key in ("subject", "object"):
                    name = str(brief.get(key) or "")
                    if name:
                        context_names.add(name)

        spans_by_assertion: dict[str, list[dict]] = {}
        # assertion_id -> 该断言所属边在图扩展里的支持度。按 assertion_id 精确查，
        # 不拿「这个单元的第一条扩展」顶替：一个块里常有多条扩展，取第一条会把
        # A 边的 support_count 记到 B 边的 claim 上。
        support_by_assertion: dict[str, int] = {}
        inbound_by_assertion: dict[str, bool] = {}
        for unit in units:
            for span in unit.get("evidence_spans") or []:
                aid = str(span.get("assertion_id") or "")
                if aid:
                    spans_by_assertion.setdefault(aid, []).append(
                        {"chunk_id": unit["chunk_id"], "text": str(unit.get("text") or ""), **span}
                    )
            for exp in unit.get("graph_path") or []:
                for aid in exp.get("assertion_ids") or []:
                    support_by_assertion[str(aid)] = max(
                        support_by_assertion.get(str(aid), 0),
                        int(exp.get("support_count") or 0),
                    )
                    if exp.get("used_in_edges"):
                        inbound_by_assertion[str(aid)] = True

        candidates: list[dict] = []
        for unit in units:
            for brief in unit.get("assertions") or []:
                candidates.append({"unit": unit, "brief": brief})
        base["stats"]["candidates"] = len(candidates)

        accepted: list[dict] = []
        for cand in candidates:
            unit = cand["unit"]
            brief = cand["brief"]
            aid = str(brief.get("assertion_id") or "")
            spans = spans_by_assertion.get(aid) or []
            if not spans:
                # 判据 1：没有引用锚点的断言不进答案
                base["stats"]["dropped_no_citation"] += 1
                continue
            span = spans[0]
            claim = {
                "assertion_id": aid,
                "chunk_id": span["chunk_id"],
                # 1 doc = 1 scene（见 `identity.Mention.doc_id` 的定义），所以判断
                # 「是否同一场景」用 doc_id 而不是 chunk_id：同一场景可能被切成多块，
                # 拿 chunk_id 比会把同场景冲突误判成「跨场景剧情演进」。
                "doc_id": str(unit.get("doc_id") or ""),
                "evidence": str(span.get("evidence") or ""),
                "start": int(span.get("start", -1)),
                "end": int(span.get("end", -1)),
                # 引用粒度实验要能把区间扩到整行，需要块正文；存在 claim 上一路带下去，
                # 避免在发引用时反查 chunks。**不参与任何判据**，只是数据的载体。
                "chunk_text": str(span.get("text") or ""),
                "subject": str(brief.get("subject") or ""),
                "predicate": str(brief.get("predicate") or ""),
                "object": str(brief.get("object") or ""),
                "confidence": brief.get("confidence"),
                "support_count": support_by_assertion.get(aid, 1),
                "unit_score": float(unit.get("score") or 0.0),
                "used_in_edges": bool(inbound_by_assertion.get(aid)),
            }
            check = _claim_supported_by_line(claim, span["text"])
            claim["predicate_in_line"] = check["predicate_in_line"]
            if not check["ok"]:
                # 判据 2：引用不支持结论 → 丢掉，不放宽
                base["stats"]["dropped_unsupported"] += 1
                warnings.append(
                    f"引用不支持结论，已丢弃 {aid}：端点 {check['missing']} 不在证据所在整行内"
                )
                continue
            if not check["predicate_in_line"]:
                warnings.append(
                    f"{aid} 的谓词「{claim['predicate']}」未在原文行内出现"
                    f"（规范化谓词，原文措辞不同）"
                )
            accepted.append(claim)

        # 判据 4：端点必须都在上下文出现过
        offenders = _claims_are_grounded(accepted, context_names)
        if offenders:
            bad = set(offenders)
            kept: list[dict] = []
            for claim in accepted:
                names = {str(claim.get("subject") or ""), str(claim.get("object") or "")}
                if any(f"{n}（{claim['assertion_id']}）" in bad for n in names if len(n) >= 2):
                    base["stats"]["dropped_ungrounded"] += 1
                    continue
                kept.append(claim)
            warnings.append(f"有 {len(accepted) - len(kept)} 条 claim 的实体不在上下文里，已丢弃")
            accepted = kept

        if not accepted:
            base["abstained"] = True
            base["abstain_reason"] = "no_supported_claim"
            return base

        # 排序：**先看这条事实是不是问句问的那对实体**，再看它有多可信、块有多相关。
        #
        # 原先只按 `-unit_score`（块级 BM25 分）排序，实测踩到真事：问「凯尔希和罗德岛是
        # 什么关系？」，`凯尔希 隶属 罗德岛`（support 54）排在第 12 位，被 `max_claims=5`
        # 截掉，答案里全是同一热门块带来的「文月交谈凯尔希」这类旁支闲聊。根因是
        # `unit_score` 是**块**的分而不是**断言**的分：一个热门块里的所有断言都被同一个
        # 高分抬高，与问句问的是谁的什么关系无关。
        #
        # `focus_tier` 是这条判据的显式化：
        #   0 —— 两端**都**是问句里的实体候选（问的正是它们之间的关系）
        #   1 —— 只有一端是候选实体（问 A 的事，答案涉及 A）
        #   2 —— 两端都不是候选实体；只有在上面两档都凑不满时才会被陈述
        # 这是**排序**判据，不是过滤器：第 2 档的 claim 仍在 `accepted` 里，引用也照旧
        # 逐条给出，只是在条数受限时排到后面。若直接丢弃，「问 A 和 B 的关系，但答案
        # 需要 C 作中间人」的多跳题会被砍断。
        focus_names = {
            str(c.get("name") or "") for c in (qa.get("entity_candidates") or [])
        }
        focus_names.discard("")

        def _focus_tier(claim: dict) -> int:
            ends = {str(claim.get("subject") or ""), str(claim.get("object") or "")}
            hits = len(ends & focus_names)
            return 0 if hits >= 2 else (1 if hits == 1 else 2)

        for claim in accepted:
            claim["focus_tier"] = _focus_tier(claim)
        base["stats"]["focus_tier_counts"] = {
            str(t): sum(1 for c in accepted if c["focus_tier"] == t) for t in (0, 1, 2)
        }

        accepted.sort(
            key=lambda c: (
                c["focus_tier"],
                -c["support_count"],
                -c["unit_score"],
                c["assertion_id"],
            )
        )
        # 同一事实可能由多条断言支撑（同一三元组在不同轮次各抽到一次，edge_id 不同）。
        # 答案里写成一句话就够，重复陈述会让人误以为有多件独立事实——但**引用一条不删**，
        # 每条引用各自指向它自己的原文句。所以去重的是「claim 文本」，不是 citations。
        seen_lines: dict[str, dict] = {}
        deduped: list[dict] = []
        for claim in accepted:
            text_key = f"{claim['subject']}\x1f{claim['predicate']}\x1f{claim['object']}"
            prev = seen_lines.get(text_key)
            if prev is None:
                seen_lines[text_key] = claim
                deduped.append(claim)
                continue
            prev.setdefault("extra_assertion_ids", []).append(claim["assertion_id"])
            prev.setdefault("extra_citations", []).append(
                {
                    "chunk_id": claim["chunk_id"],
                    "evidence": claim["evidence"],
                    "start": claim["start"],
                    "end": claim["end"],
                    "chunk_text": claim.get("chunk_text") or "",
                    "assertion_id": claim["assertion_id"],
                }
            )
        accepted = deduped
        if len(accepted) > self.max_claims:
            warnings.append(
                f"候选 {len(accepted)} 条，只陈述前 {self.max_claims} 条（其余见 citations 之外，未写入答案）"
            )
            accepted = accepted[: self.max_claims]

        # ---- 判据 5：冲突材料不无条件合并 --------------------------------
        conflicts = self._same_scene_conflicts(accepted)
        if conflicts:
            for item in conflicts:
                warnings.append(
                    f"冲突未合并：{item['detail']}（同一场景内互斥，须人工判定，不写成一句确定的话）"
                )

        # ---- 组答案正文 --------------------------------------------------
        lines: list[str] = []
        for claim in accepted:
            line = f"{claim['subject']}{claim['predicate']}{claim['object']}。"
            lines.append(line)
        answer_text = "".join(lines)
        if conflicts:
            answer_text += "（注：以上关系存在互斥记录，见 warnings，不作单一结论。）"

        citations: list[dict] = []
        for c in accepted:
            claim_text = f"{c['subject']}{c['predicate']}{c['object']}"
            _ev = _citation_span(
                c.get("chunk_text") or "", int(c["start"]), int(c["end"]),
                whole_line=CITATION_WHOLE_LINE,
            )
            citations.append(
                {
                    "claim": claim_text,
                    "chunk_id": c["chunk_id"],
                    "evidence": _ev[2],
                    "start": _ev[0],
                    "end": _ev[1],
                    "assertion_id": c["assertion_id"],
                    "predicate_in_line": c["predicate_in_line"],
                }
            )
            # 同一事实的其余支撑断言：答案正文只写一次，但每一条引用都留着——
            # 删掉它们等于丢掉「这件事在多个轮次被独立观察到」的证据。
            for extra in c.get("extra_citations") or []:
                _ex = _citation_span(
                    extra.get("chunk_text") or "", int(extra["start"]), int(extra["end"]),
                    whole_line=CITATION_WHOLE_LINE,
                )
                citations.append(
                    {
                        "claim": claim_text,
                        "chunk_id": extra["chunk_id"],
                        "evidence": _ex[2],
                        "start": _ex[0],
                        "end": _ex[1],
                        "assertion_id": extra["assertion_id"],
                        "predicate_in_line": c["predicate_in_line"],
                    }
                )

        # 置信度：证据条数 + 引用覆盖率 + 支撑度，三者都参与，避免单一指标虚高。
        # `cite_ratio` 上限取 1：同一事实的多条支撑引用会让它大于 1，但那不该
        # 让置信度超过「每条 claim 都有引用」的水平。
        n = len(accepted)
        claim_with_cite = len({c["claim"] for c in citations})
        cite_ratio = min(1.0, claim_with_cite / max(1, n))
        support_bonus = min(0.2, 0.02 * sum(c["support_count"] for c in accepted))
        confidence = min(0.95, 0.35 + 0.08 * n + support_bonus) * cite_ratio

        base.update(
            {
                "answer": answer_text,
                "citations": citations,
                "confidence": round(confidence, 4),
                "stats": {
                    **base["stats"],
                    "accepted": n,
                    "conflicts": len(conflicts),
                },
            }
        )
        return base

    @staticmethod
    def _same_scene_conflicts(claims: list[dict]) -> list[dict]:
        """同一对实体在同一场景里出现互斥谓词 → 记冲突，不合并。

        复用 `vocab.MUTUALLY_EXCLUSIVE_PAIRS`，不另写一套互斥表——两张表并存
        必然有一张先烂（`tests/test_single_source_of_truth.py` 记的三次事故同机理）。
        """
        from . import vocab

        def exclusive(a: str, b: str) -> bool:
            pair = tuple(sorted((a, b)))
            return pair in {tuple(sorted(p)) for p in vocab.MUTUALLY_EXCLUSIVE_PAIRS}

        by_pair: dict[tuple[str, str], list[dict]] = {}
        for claim in claims:
            key = tuple(sorted((claim["subject"], claim["object"])))
            by_pair.setdefault(key, []).append(claim)
        out: list[dict] = []
        for (left, right), items in sorted(by_pair.items()):
            for i, a in enumerate(items):
                for b in items[i + 1 :]:
                    if not exclusive(a["predicate"], b["predicate"]):
                        continue
                    if a["doc_id"] != b["doc_id"]:
                        continue  # 不同 doc = 不同场景，属剧情演进，不算冲突
                    out.append(
                        {
                            "pair": [left, right],
                            "predicates": [a["predicate"], b["predicate"]],
                            "detail": f"{left}/{right} 在同一场景 {a['doc_id']} 同时"
                                      f"「{a['predicate']}」与「{b['predicate']}」",
                        }
                    )
        return out


class LLMAnswerer:
    """走 `arkrag.llm` 的答案器。**默认不可用**：本机没有端点。

    存在的理由不是"以后可能用"，而是把接口定下来：它拿到的上下文与
    `ExtractiveAnswerer` 完全相同，产出的 answer 走**同一套** `verify_citations`。
    模型说得再像，引用对不上就降置信度——这条不因换了生成器而放松。

    没有配置时 `answer()` 抛 `RuntimeError`（由 `llm.LLMClient` 发出），不静默降级：
    静默降级会让调用方以为自己在用模型，实际在用模板。
    """

    name = "llm"

    def __init__(self, *, client=None, min_top_score: float = DEFAULT_MIN_TOP_SCORE) -> None:
        self.client = client
        self.min_top_score = float(min_top_score)

    def answer(self, question: str, context: dict, **kwargs) -> dict:  # pragma: no cover
        if self.client is None:
            from . import llm

            raise RuntimeError(
                "LLMAnswerer 需要 arkrag.llm.LLMClient；本机未配置 "
                f"ARKRAG_LLM_BASE_URL/ARKRAG_LLM_MODEL，{llm.__name__} 无法工作。"
                "离线评测请用 ExtractiveAnswerer。"
            )
        raise RuntimeError(
            "LLMAnswerer 的模型调用路径尚未启用：本机没有 LLM 端点，"
            "不允许用假输出冒充实测。请用 ExtractiveAnswerer。"
        )


def context_fingerprint(context: dict) -> str:
    """上下文摘要（供 `ask.py` 的 `--json` 落盘与复现比对）。"""
    payload = {
        "context_digest": context.get("context_digest"),
        "units": [
            {"chunk_id": u.get("chunk_id"), "spans": len(u.get("evidence_spans") or [])}
            for u in context.get("units") or []
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


__all__ = [
    "DEFAULT_MIN_TOP_SCORE",
    "ExtractiveAnswerer",
    "LLMAnswerer",
    "MAX_CLAIMS",
    "MIN_EVIDENCE_CHARS",
    "context_fingerprint",
    "verify_citations",
]
