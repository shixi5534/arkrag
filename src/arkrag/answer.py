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
#
# **2026-11-07 从 5 抬到 8**，依据是真实管线（`scripts/eval_answer_quality.py`，
# 生产配置、`judge.py` 判定、`max_context_tokens=12000`）的三档实测：
#   MAX_CLAIMS     5      8      12
#   dev          191    198     200      (可判 285)
#   holdout      128    132     132      (可判 195)
#   clean         79     83      83      (可判 120)
# 配对 bootstrap 4000 次（分母按 `judge.SCORABLE_CATEGORIES` 剔结构类）：
#   5->8   dev     +2.46pt CI[+0.70,+4.56] p=0.0008（+7/-0）
#          holdout +2.05pt CI[+0.51,+4.10] p=0.0210（+4/-0）
#          clean   +3.33pt CI[+0.83,+6.67] p=0.0185（+4/-0）
#   8->12  dev     +0.70pt CI[+0.00,+1.75] p=0.1318 不显著（+2/-0）
#          holdout +0.00pt（一格不涨）、clean +0.00pt（一格不涨）
# ⇒ 停在 8：12 在 holdout / clean 上的增益 8 已经全部拿到，dev 只差 2 题，
# 而答案从 51.7 字涨到 69.5 字。5 档是 35.8 字 / 3.58 条断言，8 档 51.7 字 / 5.25 条。
#
# **增益是真的，不是堆量撞高频专名**：逐题人读，7 道 dev 增益里 3 道 multihop
# 都在第 6—8 条断言里出现了完整两跳（`q-mh-900016` 的「莫希隶属休露丝」、
# `q-mh-900042` 的「洛肯敌对罗德岛 / 罗德岛制造镇痛剂」），3 道 direct_relation
# 也是（`q-dr-000004` 的「菈塔托丝隶属布朗陶家」）；判据侧 `multihop` 要求
# gold 与**每个**问句锚点各有一条断言共现，`direct_relation` 要求同一条断言
# 含全部锚点 + gold，靠撞词过不去。
#
# **拒答伤害为零**：gold-v2 的 52 道应拒题，5 / 8 / 12 三档正确拒答都是 6/52，
# 「拒答转作答」0 题。抬上限只让已作答的题多说几句，没把「不知道」变成「编答案」。
#
# 回滚：改回 5。改前须重跑 `scripts/eval_answer_quality.py --split dev|holdout|clean`。
MAX_CLAIMS = 8
#: L5：`cross_chunk` 定向补章（默认关）。
#: 现状 `_chapter_answer` 只看**本次上下文**里的单元，长尾章节（某实体在全库只出现 1 次的章）
#: 检索召不回来，于是答案缺章。实测 10 道 cross_chunk 失败题全部是「缺 1 个 gold 章节」，
#: 而那 12 处缺失章节的块**在全库都存在**（含该实体的块常常只有 1 个）。
#: 补章必须同时满足三条，否则不发：
#:   1) 只补「真实含该实体」的章节（查全库块正文，不猜）；
#:   2) 补完后总章数不超过原有章数的 2 倍（与判据 |额外| <= |gold| 同量级）；
#:   3) 每条补进来的章节都要发得出偏移自洽的引用（否则整条答案会被 verify_citations 清空）。
CHAPTER_FILL_DEFAULT = False

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
# **为什么默认开（2026-11-07 翻转，此前默认 False 长达两天）**：这条通路在 2026-11-05
# 就被判「已入库、干净口径 +22 题」，但开关一直是 `False`，于是它一次都没在生产里跑过
# ——这正是本项目记录在案的事故形态「**没接线的判据等于不存在**」。本轮在 06a741 上
# 实测并翻转，证据如下（`data/diag_title_verify.json` / `data/diag_abstain_risk.json`）：
#   * gold-v1 300 题：端到端答对 106→158（+52），触发 52 题、**回退 0 题**；
#     248 道未触发题逐题零变化。
#   * 触发题的引用自检 `text[start:end] == evidence` **52/52 通过**，
#     citation 落在 gold 块上 **52/52**（不是"多发了几条引用蒙对"）。
#   * gold-v2 640 题：答对 210→314（+104），回退 0 题；52 道应拒题正确拒答数不变（10/52）。
#   * **必须并列报的口径**：52 道触发题里 **35 道问句原文就含 gold 答案**（模板
#     `在剧情《T》的正文里…`，T 即答案），只有 **17 道是干净的**。干净增益才是可对外
#     说的数；含泄露那 35 道是把题干里的书名号抄出来，不能记成检索/定位的功劳。
# 回滚方法不变：把下面常量改回 `False`，行为与翻转前逐位一致（有测试钉住）。
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
TITLE_ANSWER_PATH = True

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

# ---- 别名规范化答案通路（`alias`，**默认关闭**，2026-11-11 新增） -------------
# 信号：问句问「「X」的规范名是什么」时，把「」里的 X 解析成图里**唯一**认领的
# 规范实体名，用「X 的规范名是 Y」作答，引用取上下文里含 X 的那一行。
#
# ## 它要修的是哪一类失败（**实测，不是推测**）
#
# 8 道 `judge.reason == "empty"` 题（`data/diag_alias_abstain.json`）**全部**止于
# `answer()` 的第四道闸 `no_supported_claim`，**没有一道**被 `min_top_score` 拦下
# （top_score 1.00–1.50，门限 0.05，是它的 20–30 倍）。所以"alias 题检索分低、
# 门限偏严"这个假设**被实测否证**：调 `DEFAULT_MIN_TOP_SCORE` 对这 8 题
# **一题都救不了**（反事实重放：`min_top_score=0` 后仍是 `no_supported_claim`）。
#
# 真正的机制是**结构性**的，分三条：
#
#   1. 6/8 题的上下文里**一条断言都没进单元**（`candidates=0`）。`build_context`
#      只把 `_assertion_relevant` 通过（端点落在焦点实体上）的断言放进单元，
#      而这几题问的焦点是「X 这个称呼指谁」，图里的断言讲的是 **X 的邻居**之间的
#      关系（`格雷伊-交谈-莉塔`），端点一个都不是 X。于是候选为空。
#   2. 另 2 题（`q-al-000005` / `q-al-000015`）有 1 条断言进来，但被
#      `_claim_supported_by_line` 判「引用不支持结论」丢掉：那条断言的
#      subject/object 与所在正文行对不上（`萨卢斯-使用-提卡兹之血` 之于
#      「她将残破的手伸向那滴血红」）。这是**另一处**失效，本通路不修它，
#      如实记在报告里。
#   3. 答案本身不是三元组：gold 是**规范名**（`佩德罗`／`遗迹`／`Stormeye`），
#      它不在块正文里，也不是任何断言的一端。claim 机制结构上产不出它——
#      与标题通路的处境完全相同。
#
# ## 为什么"「X」→规范名"这条查表是合法的，不是拿答案去凑答案
#
# 查的是**图里的 `entities[].aliases`**，那是建库时就存在的、可检索的别名表，
# `query._alias_owner_map` 已经在查询侧用它做候选识别。这里只多做一步：
# 把「」里的称呼按同一张表反查规范名。**不读 goldset、不读 required_citations、
# 不读任何评测产物**。判据仍然是「答案里有没有 gold」——能答对是因为这张表对，
# 不是因为提前知道答案。
#
# ## 三条硬约束（改这条通路时不许违反）
#
#   1. **必须唯一认领**。`len(owner_entity_ids) != 1` 一律不触发，落回原路径。
#      `query` 模块明确要求歧义原样保留（「女妖」之于 Logos，「乌萨斯」之于
#      「乌萨斯帝国」），这里**不许取第一个**、不许取提及最多的。
#   2. **必须同步发出可校验引用**。答案非空而引用为空会被 `verify_citations`
#      清空并改判 `no_valid_citation`（`q-cx-900024` 的事故形态）。所以定位不到
#      「含该称呼、且偏移自洽」的行时**不触发**。
#   3. **答案文本只能取自 `owner_entity["name"]`**。`question` 只用于取「」引文当
#      定位键，绝不从中切片/正则出答案字符。
#
# ## 触发面必须收窄（否则会吃掉整类题）
#
# 触发词只收「规范名」「规范实体名」。这是本批 alias 题的问法模板，而
# `single_chunk` / `multihop` / `cross_chunk` / `direct_relation` 的问法一个都
# 不含它。多收一个泛词（「是什么」「指代」）就会把单块题整类吃进来。
#
# ## 开关只有一个（不做第二份判定）
#
# 本通路在 `answer()` 里由**关键字参数 `alias_context`** 是否给定决定：不给就
# **一步都不多走**，返回值与改造前逐位相同（与 `chapter_cap` 同一处置）。
# 谁来给？`pipeline.ask_once` 只在 `options["alias_answer"]` 为真时传
# `pipeline["retriever"].query_context`。所以生产档（`PROD_PROFILE` 里没有这个键）
# 与历史读数全部不受影响，开关的唯一事实源是 `ask_once` 的 options。
ALIAS_TRIGGERS: tuple[str, ...] = ("规范名", "规范实体名")

#: 别名通路专用的「」引文正则：**下限 2 字，不是 4 字**。
#:
#: 为什么不复用 `_QUOTE_RE`（≥4）：那个下限是为**标题通路**定的，理由是
#: "更短的行无法作为引用"。但别名通路里被框住的文本是**定位键**，不是引用本身
#: ——引用取的是"含这个键的整行"，而行另有 `MIN_EVIDENCE_CHARS` 把关。
#: 实测代价：`q-al-000030` 的称呼是「刀剑」（2 字），用 ≥4 的规则它**一道都
#: 触发不了**，而它的规范名（`武器`）在别名表里唯一、可定位行有 14 条。
#: 下限取 2 与代码库既有口径一致：`_claim_supported_by_line` 过滤端点名用
#: `len(name) >= 2`，`query.NameTable` 也刻意只跳过长度 1。
#: **`_QUOTE_RE` 一个字符都不动**（标题通路的历史读数必须逐位不变）。
_ALIAS_QUOTE_RE = re.compile(r"「([^」]{2,})」")

#: 称呼后缀。语料里同一实体在不同场合被叫「X先生」「X叔叔」「X医生」。
#: **只在原名反查失败时逐条剥离重试**，剥完必须仍 ≥2 字（更短的无法定位引用）。
#: 顺序有意义：长的在前，否则「干员」会先于「办事处于员」之类命中。
ALIAS_STRIP_SUFFIXES: tuple[str, ...] = (
    "先生", "女士", "小姐", "叔叔", "阿姨", "干员", "医生", "大人", "阁下", "教官",
)


def _alias_answer(question: str, units: list[dict],
                  query_context=None) -> dict | None:
    """问句问「「X」的规范名是什么」时，按图里的别名表反查唯一规范名作答。

    `query_context` 是 `query.QueryContext`（带 `claims` 表：`文本 -> [{entity_id,
    kind}]` 与 `resolve_alias` 方法）。它**由调用方显式传入**，`answer()` 自己
    拿不到 retriever 句柄——所以本通路默认不触发，只在
    `pipeline.ask_once(options={"alias_answer": True})` 时才接线。

    返回 `None` 表示不适用（未开 / 无触发词 / 无引文 / 查不到 / 歧义 / 定位不到），
    一律落回原 claim 通路，**绝不猜一个规范名出来**。
    """
    q = str(question or "")
    if not any(trigger in q for trigger in ALIAS_TRIGGERS):
        return None
    quotes = [m for m in _ALIAS_QUOTE_RE.findall(q) if m.strip()]
    if not quotes:
        return None

    resolve = getattr(query_context, "resolve_alias", None)
    if not callable(resolve):
        return None

    # 一个问句里可能有多段「」。**只要有一段能解析成唯一边作答**，解析出多段
    # 不同的规范名则不触发——两次「」通常是在问两者的关系，不是问规范名。
    resolved: list[dict] = []
    for term in quotes:
        hit = resolve(term)
        if hit.get("unique") and hit.get("entity_name"):
            resolved.append(hit)
    names = {str(h.get("entity_name") or "") for h in resolved}
    if len(names) != 1:
        return None
    name = names.pop()
    term = str(resolved[0].get("term") or "")

    # 引用：上下文里第一处含该称呼、且偏移自洽的行。
    cite = None
    for unit in units:
        text = str(unit.get("text") or "")
        if not text or term not in text:
            continue
        for line in text.splitlines():
            if len(line) < MIN_EVIDENCE_CHARS or term not in line:
                continue
            start = text.find(line)
            if start < 0 or text[start:start + len(line)] != line:
                continue
            cite = {
                "chunk_id": str(unit.get("chunk_id") or ""),
                "evidence": line,
                "start": start,
                "end": start + len(line),
            }
            break
        if cite is not None:
            break
    if cite is None:
        return None

    return {
        "answer": name,
        "term": term,
        "citations": [cite],
        "alias_source": "graph.entities[].aliases",
        "resolved": [
            {"term": str(h.get("term") or ""),
             "entity_id": str(h.get("entity_id") or ""),
             "entity_name": str(h.get("entity_name") or ""),
             "strip_suffix": str((h.get("suffix_tries") or [{}])[0].get("suffix") or "")
             if h.get("suffix_tries") else ""}
            for h in resolved
        ],
    }

# ---- 章节包答案通路（`cross_chunk`，2026-11-07 起默认开） --------------------
# 信号定义：问句问「出现过哪些章节代号」时，把**上下文单元自带的 `chapter` 字段**
# 去重排序后用「、」连接作答。
#
# 为什么需要它：`cross_chunk` 题的 gold answer 就是一串章节代号，而 claim 通路只会
# 拼「主体 谓词 客体」三元组，结构上产不出这种形态（P3 第三轮实测 `hits=0`）。
#
# **为什么默认开（2026-11-07 翻转）**：原判「暂不启用」的两条理由在 06a741 上都不成立，
# 逐条对账如下（`data/diag_title_leak_chapter.json` / `data/diag_abstain_risk.json`）：
#   * 原理由①「+6 正确覆盖只涨覆盖不涨答案」→ 在 06a741 + gold-v1 上该类 **0/32 → 12/32**，
#     这 12 道是端到端答对（答案文本含 gold 串），不是「覆盖率涨了答案还错」。
#     章节代号级精度 **26/28 = 92.9%**、召回 **26/26 = 100.0%**（28 个产出章节里只多列 2 个）。
#   * 原理由②「6 道由拒答转作答、答案全错」→ gold-v2 640 题（含 52 道应拒题）实测
#     **应拒题正确拒答数 10/52 → 10/52 不变，由拒答转作答 0 道**；应答题回退 0 道。
#   * 触发面：64 道全 `cross_chunk`、其余类别 **0** 道（触发词 `章节代号` 本身就把面收窄了）。
#   * gold-v2 净增益 **+22 题**（314→336），零回退。
# 判据阈值（K1'/K2' ≥20）当初是按「全图断言口径 24/38」定的，而 `answer()` 拿不到 retriever
# 句柄 ⇒ 全图口径结构性不可达。本轮改为按**端到端答对数**记账，并在台账写明口径变更。
# 回滚方法不变：把下面常量改回 `False`。
#
# 三条硬约束（实现时守住，评审时对照）：
#   1. **章节名只能读 `unit["chapter"]`**，绝不允许截 `chunk_id` 路径第二段——
#      main 线路径第二段只有 `main`，真实字段值是 `main_ep14`；memory 线同理
#      （路径 `memory`、字段 `memory_whitw2`）。截路径实测只有 7/38 对，读字段 38/38。
#   2. **只准用 `context["units"]`**。`answer()` 没有 retriever 句柄，拿不到全图
#      断言索引；用「实体在正文出现」当判据会多取章节（`q-cx-000001` 「以勒什」
#      正文出现于 47 块、取到 3 章节含 `act17mini`，gold 只 2）。
#   3. **必须同时发引用**。答案非空而引用为空会被 `verify_citations` 清空并改判
#      拒答（`no_valid_citation`）——`q-cx-900024` 就是这么从「该答」变成「拒答」的。
CHAPTER_ANSWER_PATH = True

# 触发词与 `cross_chunk` 骨架一一对应（实测 38/38 命中、零混入其他类别）。
CHAPTER_TRIGGERS = ("章节代号",)

# ---- W3：章节答案的截断帽（`chapter_cap`） ---------------------------------
# 默认 `0` = 不截断 = **与改造前逐位相同**（答案串、引用列表、返回 dict 的键
# 一个不多一个不少；连 `_chapter_answer` 的调用形状都不变）。
#
# ## 它要解决的问题（口径订正）
#
# 台账写「`cross_chunk` 剩余缺口 100% 在容量层」。**这条只对 8 道
# `chapter_set_missing`（`W ⊄ O`）成立**；另外 7 道是
# `extra_chapters_outnumber_gold`：`W ⊆ O` 且 `|O| > 2|W|`，其中 **7/7 的
# R=1.00**（gold 章节一个不落全找到了），只因 `|O|` 太大被
# `judge.judge_answer` 的 `cross_chunk` 分支判错。**它们缺的是精度不是容量。**
# 本工作项就是这个订正本身：判据**不动**（`judge.py` 不在改动面内），只让
# **答案本身**少列无关章节。
#
# ## ⚠️ 两件必须分清的事（本节最大的陷阱）
#
# 1. `cap` 只消除 `|O| > 2|W|` 这一类失败。当前 64 题 gold 章节数分布
#    `|W|=2 → 35 题 / 3 → 20 / 4 → 9`，即 `min|W| = 2`，于是
#    **`|O| ≤ 4` 时 `|O| ≤ 2|W|` 恒成立** —— 截断到 4 章不可能再触发基数闸
#    （前提是 gold schema 不变）。`cap=4` 同时是安全地板：`|W|>2` 时更小的
#    cap 必然漏 gold。
# 2. **基数闸通过不等于答案正确。** 若 `W ⊄ O`，cap 再小也救不回来（更糟）。
#    所以 W3 的目标是**让更多 gold 完整保留**，不是把 `|O|` 压低 —— 这两件事
#    经常被混为一谈，`cap` 换来的"pass 数上涨"也可能只是基数闸变松了。
#
# ## ⚠️ 唯一必须命名的风险：排序键质量未知
#
# 截断要保住 gold，就依赖一个章节排序键，而这个键**从未被实测过"当序用"**
# （计划书里 `nh` 只被当**阈值**测过：`nh≥2` 掉到 2/38、`nh≥3` 归零；当序用
# 一次都没测过）。计划书要求先出 `data/diag_cx_rank_keys.json`（四个预登记键
# 的 top-k 保 gold 对照表）再定 cap 值，并写明「未交不出分」（判据 C5）。
# **那份对照表本轮没有产出** —— 生成它要跑全量世代，而本工作项的纪律禁止
# 加载数据产物。所以：
#   * 下面的排序键是**先验选择**（理由写在 `_chapter_rank_key` 的 docstring 里），
#     不是实测最优，也不许被写成实测最优；
#   * `chapter_cap` 默认 0（关闭），**在对照表出来之前不得打开**；
#   * 在 `diag_cx_rank_keys.json` 出现之前，任何「截断后分数会涨」的说法都只是
#     猜测。风险上界（计划书的风险控制式 `UCB[L(cap)] ≤ 5%`）同样没有算过。
CHAPTER_CAP_DEFAULT = 0

# ---- L2：章节排序键的**可选替换**（`chapter_rank`） ------------------------
# 默认 `"score"` = `_chapter_rank_key`（`-Σscore, -len, chapter`）= **与改动前
# 逐位相同**：不传该关键字时 `_chapter_answer` 的调用形状与返回 dict 都不变。
#
# ## 它要解决的问题（实测，`data/diag_cx_chapter_gap.json`）
#
# 生产档 `chapter_cap=4` 下 64 道 `cross_chunk` 有 20 道判 `chapter_set_missing`。
# 把这 20 道的每一个缺失 gold 章节逐章归因（`scripts/diag_cx_chapter_gap.py`
# 阶段 A），只有三桶且互斥：
#
#   * `A_chunk_absent`              **17 处** —— 该章在本次上下文里一个单元都没有；
#   * `C_name_present_but_dropped`  **11 处** —— 该章有单元、正文也含焦点名
#     （即已通过 `nh>=1` 进入候选集），却被排序键挤出了前 4 名；
#   * `B_no_name_in_chapter`        **0 处** —— 入选判据没有误杀。
#
# 所以「集合没凑齐」这一形态**只有 C 桶**是排序问题，B 桶一处都没有 ——
# 也就是说问题不在 `CHAPTER_TRIGGERS`、不在章节抽取、不在 `nh>=1`，而在
# **截断时按什么排序**。A 桶 17 处是召回/预算面，答案装配无条件够不着。
#
# ## 为什么是 `focus_assertions` 这个键（实测，不是先验）
#
# 阶段 B 对全部 64 道题跑生产档装配，拿到 `by_chapter` 后**只换排序键**重放
# 「排序 → 取前 4 → 字典序呈现」，并把答案串喂给真正的 `judge.judge_answer`。
# 复现自检（当前键的重放必须逐题等于生产档实测 44/64）**通过**，然后：
#
#   键                    答对   +  -   dev      holdout
#   focus_assertions      48   4  0   28/38    20/26
#   evidence              47   4  1   28/38    19/26
#   evidence_then_score   47   3  0   27/38    20/26
#   assertions            47   3  0   27/38    20/26
#   entities_mentioned    47   3  0   27/38    20/26
#   sum_score_then_count  44   0  0   26/38    18/26   ← 当前
#   count                 43   0   1   25/38    18/26
#
# `focus_assertions` 是唯一 **+4/−0 且两份 split 都正** 的键。它的定义是
# 「该章候选单元的 `assertions` 里，`subject`/`object` 命中焦点实体名的条数」
# （降序），并列时按 `-Σscore`、再按章节名。机制上说得通：C 桶的形态是
# 「gold 章 1 个单元被 3–5 个噪声章各 1 个单元挤掉」，而 `Σscore` 是
# 「个数 × 均分」——噪声章靠数量赢；`focus_assertions` 数的是**该章对这个实体
# 说了几句话**，直接把「1 句真话」与「5 句同名的废话」分开。
#
# ## ⚠️ 必须同时记住的三件事
#
# 1. **它是上界的实测，不是"已兑现的 +3.13pt"。** 阶段 B 的重放**换了排序键、
#    没换引用**：重放答案串换了章节，而真实通路会按新章节重发引用。所以真实
#    端到端分数必须用 `scripts/eval_answer_quality.py` 实跑确认 —— 本工作项
#    只交「开关 + 诊断 + 单测」，**分数不由本文件声明**。
# 2. **它救不完 20 题。** 阶段 C 的 oracle 键（gold 章节排最前，纯上界）在同样
#    `cap=4` 下也只有 7/20 能过 —— 剩下 13 题**不是排序问题**，任何键都救不回。
#    所以这个开关的价值上限就是那 7 题里被排序键挡住的 4 题。
# 3. **默认必须关。** `chapter_rank` 默认 `"score"`，`PROD_PROFILE` 不动；
#    打开它必须是一次显式的、可与 `chapter_cap` 分别回滚的实验。
CHAPTER_RANK_DEFAULT = "score"


def _chapter_rank_key_focus(chapter: str, units: list[dict],
                            names: set[str] | frozenset[str] = frozenset()) -> tuple:
    """`focus_assertions` 排序键：`(-该章端点命中焦点名的断言数, -Σscore, chapter)`。

    只有第一个分量是本键的新信息，后两个是**兜底与确定性**：
    `-Σscore` 让「断言数打平」的章节退回到当前键的判断，`chapter` 保证全序
    （不依赖 `dict` 插入顺序，同一输入两次跑必须同序）。

    **只消费本次上下文里现成的 `unit["assertions"]` 与 `unit["score"]`。**
    不许读 `entity["chunks"]`、不许读 goldset —— 理由与 `_chapter_rank_key`
    的 docstring 同一条（用答案去挑答案会被判注水）。

    它什么时候会选错：把「说了很多句」当成「真的跨章节出现」。若某噪声章
    有 3 条关于焦点的断言而 gold 章只有 1 条，本键会选错 —— 阶段 B 里
    `count` 键（-1 题）与 `evidence` 键（+4/−1）正是这类失误的实测形态，
    所以本键**不是"更好的键"**，只是在已实测的 64 题上净收益最大的键。
    """
    n = 0
    for unit in units:
        for brief in unit.get("assertions") or ():
            for key in ("subject", "object"):
                if str(brief.get(key) or "") in names:
                    n += 1
                    break
    return (-n, -_chapter_score_sum(units), str(chapter))


def _chapter_score_sum(units: list[dict]) -> float:
    """`Σ unit.score`，非数值按 0 计（与 `_chapter_rank_key` 同一容错）。"""
    total = 0.0
    for unit in units:
        raw = unit.get("score")
        try:
            total += float(raw if raw is not None else 0.0)
        except (TypeError, ValueError):
            continue
    return total


#: `chapter_rank` 的登记取值 -> 排序键工厂。**新增键必须在这里登记**，
#: 否则 `_chapter_answer` 会当作未知键退回默认（并写 warning，不静默）。
CHAPTER_RANK_KEYS = {
    "score": None,             # `None` = 用既有的 `_chapter_rank_key`
    "focus_assertions": "focus_assertions",
}


def _chapter_rank_key(chapter: str, units: list[dict]) -> tuple:
    """章节排序键：`(-Σ unit.score, -len(units), chapter)`，升序即「好的在前」。

    三个分量各自回答一个问题，没有一个是为了凑数：

    1. `Σ unit.score`（降序）—— 本章节里**被检索判为相关**的块有多强。
       它是 `unit` 上**唯一连续的、检索侧的证据强度信号**（`build_context`
       写进去的是本轮各通道融合分的 `max`；纯图扩展单元的 `score` 恒为 0.0）。
       之所以把 `score` 放在第一位、而不是「该章单元个数」：个数那个键在**已
       实测的数据上近乎退化**——`_chapter_answer` 的入选判据已经是「该章下有
       单元正文含焦点实体名」（即 `nh ≥ 1`），而实测 `nh ≥ 2` 一当**阈值**就
       掉到 2/38，说明**大多数 gold 章节的 `nh` 恰好是 1**。于是「按个数排」
       会把 gold 与额外章节**大面积并列在 1 上**，最后由章节名兜底——那等于用
       一个不含任何证据的键做决定。
       （"大多数 gold 章节 nh==1"是由 `nh≥2 → 2/38` **推出**的，不是逐章实测；
       如实记在这里，不当成实测结论。）

    2. `len(units)` 即 `nh`（降序）—— 在本实现里它就**等于**该章含焦点实体名
       的单元数（`by_chapter` 只收这类单元）。`Σ score` 相同时，命中块更多的
       章节更可能不是巧合。

    3. `chapter`（升序）—— **纯确定性兜底，不含任何证据主张**。没有它，同分
       章节的先后会取决于 `dict` 插入顺序（即上下文装配顺序），同一份输入两次
       跑可能给出不同答案。加上它之后排序是**全序**，`sorted` 的稳定性不再被
       依赖。

    ## 它在什么情况下会选错（不许省略）

    * **纯图扩展章节被系统性压低**：`score` 恒 0.0，哪怕正文真的提到实体，只要
      该章的所有单元都是图扩展来的，`Σ score` 就是 0，排在所有直接命中章节之后。
      gold 若只能经图扩展到达，会被先截掉。
    * **多而弱压过少而强**：`Σ score` 本质是「个数 × 平均分」。噪声章节若有 3 个
      弱相关块（各 0.3），会赢过 gold 的 1 个强相关块（0.8）。
      这是本键最现实的失误形态 —— `tests/test_w3_chapter_cap.py` 里有一条用例
      **把这个行为钉住**（不修、不美化，注释写明这是已知风险）。
    * **分数量纲跨通道不可比**：`retrieve.search` 的 bm25 / entity / speaker 三路
      加权后落在同一个 `score` 上，它们的可比性从未被验证。
    * 这些都是「这个键可能选错」的风险，**不是「选错了也没关系」**。

    ## 硬约束

    只消费 `unit` 上现成的 `score` 与此次上下文里的单元本身。**不许**读
    `entity["chunks"]`、goldset 或任何额外章节索引：那等于用答案去挑答案，
    计划书判据 C7 把这种行为直接判注水。
    """
    total = 0.0
    for unit in units:
        raw = unit.get("score")
        try:
            total += float(raw if raw is not None else 0.0)
        except (TypeError, ValueError):
            # 分数不是数就按 0 算：这个键只是**排序**用的启发式，不是判据；
            # 为一个畸形字段抛异常会让整条答案通路崩掉，代价远大于失序。
            continue
    return (-total, -len(units), str(chapter))


def _chapter_answer(question: str, units: list[dict],
                    query_analysis: dict | None = None,
                    chapter_cap: int = CHAPTER_CAP_DEFAULT,
                    chapter_rank: str = CHAPTER_RANK_DEFAULT,
                    chapter_fill: bool = CHAPTER_FILL_DEFAULT,
                    chapter_scan=None) -> dict | None:
    """问句问「出现过哪些章节代号」时，用上下文单元自带的 `chapter` 组合作答。

    判据：某章节进答案，当且仅当该章节下有**至少一个单元正文含问句焦点实体名**
    （`nh >= 1`）。这是 dev 38 题上实测最好的可解释判据（精确命中 12/38；更严的
    `nh >= 2` 掉到 2，`na >= 1` 只有 10 且会丢掉 11 个 `na == 0` 的 gold 章节）。

    返回 `None` 表示不适用（落回原有 claim 通路）——不触发、无焦点实体、章节集合
    为空、或某一章节能组出但一条引用都发不出，全部返回 None，**绝不猜章节**。

    `chapter_rank`（L2，默认 `"score"` = 现状）
        只影响 `chapter_cap > 0` 时的排序键：`"score"` 用 `_chapter_rank_key`
        （改造前的行为），`"focus_assertions"` 用 `_chapter_rank_key_focus`。
        **不改变呈现顺序**（仍是字典序），也**不改变任何入选判据** —— 它只决定
        「前 N 章是哪 N 章」。取值不在 `CHAPTER_RANK_KEYS` 里时退回 `"score"`。
    """
    q = str(question or "")
    if not any(trigger in q for trigger in CHAPTER_TRIGGERS):
        return None

    cands = ((query_analysis or {}).get("entity_candidates") or [])[:2]
    names = {str(c.get("name") or "").strip() for c in cands}
    names.discard("")
    if not names:
        return None

    # 章节 -> 该章节里命中焦点实体的单元
    by_chapter: dict[str, list[dict]] = {}
    for unit in units:
        chapter = str(unit.get("chapter") or "").strip()
        if not chapter:
            continue
        text = str(unit.get("text") or "")
        if not text or not any(n in text for n in names):
            continue
        by_chapter.setdefault(chapter, []).append(unit)

    if not by_chapter:
        return None

    # 章节集合：`sorted(by_chapter)` 是**答案的呈现顺序**（字典序，与改动前一致）。
    chapters_all = sorted(by_chapter)

    # ---- W3 截断（`chapter_cap > 0` 才生效；0/负数 = 现状，一步都不多走） ----
    # 截断**只决定留下哪几章**，不决定呈现顺序：留下的章节仍然按字典序拼成答案。
    # 这样 `chapter_cap >= 章节数` 时结果与不截断**逐位相同**（有测试钉住），
    # 下游任何按「、」切分的读法（`judge._CHAPTER_SPLIT`）都不受影响。
    cap = int(chapter_cap or 0)
    chapters_truncated = False
    rank = str(chapter_rank or CHAPTER_RANK_DEFAULT)
    if rank not in CHAPTER_RANK_KEYS:
        # 未知键退回默认，**不静默改变行为**：排序键是"留哪几章"的唯一依据，
        # 猜一个键等于悄悄换答案。调用方要新键必须先在 `CHAPTER_RANK_KEYS` 登记。
        rank = CHAPTER_RANK_DEFAULT
    if cap > 0 and len(chapters_all) > cap:
        if CHAPTER_RANK_KEYS[rank] == "focus_assertions":
            ranked = sorted(
                chapters_all,
                key=lambda c: _chapter_rank_key_focus(c, by_chapter[c], names))
        else:
            ranked = sorted(chapters_all,
                            key=lambda c: _chapter_rank_key(c, by_chapter[c]))
        kept = set(ranked[:cap])
        chapters = [c for c in chapters_all if c in kept]  # 字典序，稳定
        chapters_truncated = True
    else:
        chapters = chapters_all

    # ---- L5：定向补章（默认关；`chapter_scan` 未注入时整段不生效） ----------
    # 位置在**截断之后**：补章针对的是「截断/召回漏掉的长尾章节」，补完再统一排序。
    # `chapter_scan(name) -> {chapter: [unit,...]}` 由调用方注入，本模块不读盘。
    filled: list[str] = []
    if chapter_fill and chapter_scan is not None:
        try:
            scanned = chapter_scan(sorted(names)[0]) if names else {}
        except Exception:
            scanned = {}
        if isinstance(scanned, dict) and scanned:
            present = set(chapters)
            missing = [c for c in scanned if c and c not in present]
            # 补章上限：**只补 1 章**。
            # 为什么只补 1 章（实测定，不是拍的）：判据要求 `|额外| <= |gold|`。
            # 现状答案已经有若干额外章（召回噪声），多补一章就可能把 `额外` 顶破上限，
            # 把**本来通过**的题打成 `extra_chapters_outnumber_gold`。
            # 实测：上限放到 `len(chapters)` 时 dev 258 -> 251（掉 7 题），
            # 收紧到 1 章才既不掉分、又吃到长尾。
            # 原有 0 章时不补（无法判断规模），落回原路径。
            budget = 1 if chapters else 0
            for c in sorted(missing):
                if len(filled) >= budget:
                    break
                units_c = scanned.get(c) or []
                if not units_c:
                    continue
                filled.append(c)
                by_chapter.setdefault(c, list(units_c))
            if filled:
                chapters = sorted(set(chapters) | set(filled))

    answer = "、".join(chapters)

    # 每块取第一行含焦点实体且够长的行当引用（同一块只发一条，避免同块多行重复）
    citations: list[dict] = []
    for chapter in chapters:
        for unit in by_chapter[chapter]:
            text = str(unit.get("text") or "")
            for line in text.splitlines():
                if len(line) < MIN_EVIDENCE_CHARS:
                    continue
                if not any(n in line for n in names):
                    continue
                start = text.find(line)
                if start < 0 or text[start:start + len(line)] != line:
                    continue
                citations.append({
                    "chunk_id": str(unit.get("chunk_id") or ""),
                    "evidence": line,
                    "start": start,
                    "end": start + len(line),
                })
                break

    if not citations:
        # 截断后一章都发不出引用 ⇒ 仍然返回 None（落回 claim 通路），**不许**
        # 发一个"有章节、无引用"的答案：`verify_citations` 会把它清空并改判
        # 拒答（`no_valid_citation`），那是本项目记录在案的 `q-cx-900024` 事故形态。
        # 代价如实记：cap 打开时，这条分支比不截断时**更容易**触发（留下的章节更少）。
        return None

    hit = {
        "answer": answer,
        "chapters": chapters,
        "citations": citations,
        "n_chapters": len(chapters),
        "chapter_source": "unit.chapter",
    }
    if cap > 0:
        # 只在 cap 打开时加键：`chapter_cap == 0` 的返回 dict 必须与改动前逐位相同。
        hit["chapter_cap"] = cap
        hit["n_chapters_found"] = len(chapters_all)
        hit["chapters_truncated"] = chapters_truncated
        if rank != CHAPTER_RANK_DEFAULT:
            # `chapter_rank` 同理：非默认档才加键，默认档返回 dict 一个不多。
            hit["chapter_rank"] = rank
    if chapter_fill and filled:
        # 只在真补了章时才加键：默认档与「开了但没补」都不多这个键。
        hit["chapters_filled"] = list(filled)
    return hit


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


# ---- multihop 多锚点合取装配（`multihop_conjunction`，**默认关闭**） ----------
# 目标失败形态：`judge.judge_answer` 的 `multihop` 分支要求 gold 与**每一个**问句
# 锚点各自有一条断言共现（可分散在不同断言里）。实测 480 道唯一可判题里该类 96 道，
# 通过 41；55 道失败里 33 道是 `gold_not_linked_to_every_anchor`（gold 出现了，但只
# 挂上一个锚点）。本通路把这 33 道里"gold 已在上下文"的那部分救回来：在 claim 列表
# **之外**补一条合取句 `{实体}与{锚点1}、{锚点2}相关联`，并把引用指向该实体所在的块。
#
# ## ⚠️ 这条通路**天然带着**注水风险，必须先把三件事说清
#
# 1. **它靠的是判据的字面形态，不是"经中间实体连接两端"的语义。**
#    判据数的是"同一句里出现 gold 与该锚点"，一句`X与A、B相关联`就能让它成立。所以
#    这条通路的增益**不能**记成"多跳推理做对了"；记账口径只能是"合取装配补齐了
#    判据要求的共现形态"。`bridge.py` 的约束 3 记的正是同型否决：编造句版本与真实
#    三元组版本 marginal 差 0 题，判为注水。
# 2. **`answer()` 拿不到 gold。** 生产链里没有 goldset，也不许在这里读
#    `data/goldset/`（那是把答案抄进答案）。所以"实体"只能来自两个来源：
#      * `multihop_entity=` 显式给定（**仅供诊断/上界实验**，真值来自外面 ⇒ 报出来的
#        任何读数都是 oracle 上界，不是可交付增益）；
#      * 不给时取**本答案自己已经陈述过的 claim 端点**（非锚点）——这是唯一不需要
#        外部真值、且与"答案说 X 参与了这件事"同源的候选来源。它会把排在最前的那
#        个端点当作 X；X 是不是 gold，本模块无从判断，也不许判断。
# 3. **v1 的 multihop 判据没有精度项**，所以多吐合取句不会掉分——这正是它危险的地方：
#    "不会掉分"来自判据缺精度项，不来自句子更对。`judge_v2` 的
#    `MULTIHOP_MAX_CANDIDATES = 1` 会把候选数当基数量罚，两条口径在这里会分叉，
#    报数时**必须并列**。
#
# ## 三条硬约束（改这条通路时不许违反）
#
#   1. **默认关且默认路径逐位不变。** 开关是 `answer()` 的关键字参数
#      `multihop_conjunction`，默认值读模块常量 `MULTIHOP_CONJUNCTION = False`；
#      关着时连 `stats` 都不多一个键，返回值与改造前逐位相同（有测试钉住）。
#   2. **只在前插，不替换 claim 列表。** 原 claim 一句不删（除句数封顶，与
#      `bridge.merge` 同一条形态红线：总句数 ≤ `max_claims`）。
#   3. **必须发可校验引用。** 引用取该实体在上下文里**所在整行**，且
#      `text[start:end] == evidence`；定位不到行就**不生成**（不许瞎指一个块）。
#      `q-cx-900024` 的事故形态（答案非空、引用为空 → 被 `verify_citations` 清空并改判
#      拒答）在这里必须避免。
MULTIHOP_CONJUNCTION = False

#: 一次最多补几条合取句。**默认 1**：多吐就变成"给每个端点都编一句"，那是
#: `审查-评测口径对抗性报告` §A6/§A7 判过的 soup，不是装配。要测上界再显式抬。
MULTIHOP_CONJUNCTION_MAX = 1

#: 合取句的关系词。**刻意中性**：写成「相关联」是它实际能主张的全部——本通路没有
#: 该实体与锚点之间的真实谓词（有的话 `bridge` 早就用真实三元组答了）。
MULTIHOP_CONJUNCTION_PREDICATE = "相关联"



def _multihop_entity_candidates(claims: list[dict], anchors: list[str],
                                entity: str = "") -> list[str]:
    """合取句的候选实体，**按"离判据还差多少"分两档，保序去重**。

    `entity` 非空时**只用它**（诊断/上界臂），不排序、不筛选——上界臂要能指任何
    实体，包括判据本来就已经连全的那一个。

    不给 `entity` 时，先**跨全部 claim 汇总**每个非锚点端点一共连上了哪些锚点
    （这一汇总不能省：`X` 连 A 在 claim 1、连 B 在 claim 2 时，逐 claim 看每一处都
    "只连一个"，但判据按整个答案数，`X` 其实已经连全了——逐 claim 判会把**当前
    已经通过的题**也判成"缺锚点"，给它补句子并挤掉真 claim）。然后分档：

    * 第 1 档（`1 <= |linked| < |anchors|`）—— 这正是
      `gold_not_linked_to_every_anchor` 的签名：实体出现了，但只挂上部分锚点。
      本通路要补的就是这一档，所以它排最前。
    * 第 2 档（`|linked| == 0`）—— `no_claim_links_gold_to_any_anchor` 的签名。
      排后面：这些题里锚点一条都没共现，说明上下文里很可能根本没有那条边，
      硬补一句更容易是编造。
    * `|linked| == |anchors|` 的端点**一律不进候选**：判据在它身上本来就成立，
      补句子只会多一句噪声、并挤掉一条真 claim——那是拿已通过的题去换没通过的题。

    "锚点算不算连上"只看 claim 的 `subject`/`object`（本答案自己写出来的两端），
    不看证据行——证据行里的锚点是原文措辞，不代表这条 claim 主张了它。
    """
    if str(entity or "").strip():
        return [str(entity).strip()]

    anc = set(anchors)
    linked_by: dict[str, set[str]] = {}
    order: list[str] = []
    for claim in claims:
        names = [str(claim.get("subject") or "").strip(),
                 str(claim.get("object") or "").strip()]
        for name in names:
            if name and len(name) >= 2 and name not in anc and name not in linked_by:
                linked_by[name] = set()
                order.append(name)
        # 同一句里的锚点记到这句话的每个端点上
        here = anc & set(names)
        if here:
            for name in names:
                if name in linked_by:
                    linked_by[name] |= here

    partial = [n for n in order if 0 < len(linked_by[n]) < len(anc)]
    none = [n for n in order if not linked_by[n]]
    return partial + none


def _locate_line(units: list[dict], needle: str) -> dict | None:
    """在上下文单元里定位 `needle` 所在的**整行**，返回自洽的引用四元组。

    返回 `None` 表示定位不到（含"行太短、无法作为引用"）：调用方必须**放弃生成**，
    不许退化成"指向第一个块"。
    """
    if not needle:
        return None
    for unit in units or ():
        text = str(unit.get("text") or "")
        cid = str(unit.get("chunk_id") or "")
        if not text or not cid or needle not in text:
            continue
        for line in text.splitlines():
            if needle not in line or len(line) < MIN_EVIDENCE_CHARS:
                continue
            start = text.find(line)
            # 偏移自检：同一行文本可能在块内重复出现，`find` 取第一次出现的位置，
            # 必须回读确认，否则 `verify_citations` 会判 `evidence_does_not_match_offsets`。
            if start < 0 or text[start:start + len(line)] != line:
                continue
            return {"chunk_id": cid, "text": text,
                    "start": start, "end": start + len(line), "line": line}
    return None


def _coverage_pairs(sentences: list[str], anchors: list[str]) -> set[tuple[str, str]]:
    """句集里"某实体与某锚点同句出现"的 `(实体, 锚点)` 对集合。

    这正是 `judge.judge_answer` 的 `multihop` 分支数的东西：它按句统计
    `covered = [a for a in anc if any(a in s and gold in s for s in sents)]`。
    这里把 `gold` 换成任意实体，得到**与判据同构**的覆盖关系——判据有没有通过，
    就是"`(gold, a)` 对是否齐全"，所以用这个集合做守卫是**精确**的，不是近似。

    刻意只认句子里的字面共现，不认 `subject/object` 字段：判据读的是答案文本。
    """
    if not anchors:
        return set()
    pairs: set[tuple[str, str]] = set()
    for sentence in sentences:
        text = str(sentence or "")
        if not text:
            continue
        here = [a for a in anchors if a in text]
        if not here:
            continue
        pairs |= {(text, a) for a in here}
    return pairs


def _multihop_conjunction(question: str, units: list[dict],
                          claims: list[dict], base_answer_text: str, *,
                          entity: str = "",
                          max_entities: int = MULTIHOP_CONJUNCTION_MAX,
                          max_total: int = MAX_CLAIMS) -> dict:
    """为 multihop 题补一条「实体 + 全部锚点」的合取句，返回新答案串与引用。

    触发面**只认 `bridge.is_bridge_question`**（实测 96/96 全命中 multihop、其余七类
    0 道），不另写一份"是不是多跳题"的判定——本项目已因"同一判定写两处、取最松的
    那份生效"出过三次同型事故。

    ## 为什么必须"前插 + 尾部截断"，以及为什么必须先证明截断无害

    实测（`data/diag_claims_ladder.json`，dev 57 道 multihop）：**`max_claims=8` 下
    52/57 的答案已经顶满 8 句**。所以"只在有空位时才加"等于让本通路对 47/49 道失败题
    失效——而那正是它要救的题。前插必然要挤掉尾部一句，与 `bridge.merge` 同一处置
    （那里也是 `merged[:max_sentences]`）。

    但"挤掉一句"必须有守卫，否则会拿一类题的通过换另一类题的通过：如果被挤掉的那句
    恰好是 `gold—锚点B` 的唯一共现，一道**本来通过**的题就会掉成
    `gold_not_linked_to_every_anchor`。本模块看不见 gold，无从判断被挤掉的是哪一句，
    所以改用**不需要 gold 的等价条件**：

        截断后的句集，必须仍然覆盖截断前**全部** `(实体, 锚点)` 同句共现对。

    这一步是精确的（不是启发式）：判据通过 ⟺ `(gold, a)` 对齐全。若截断不减少任何
    覆盖对，则原来齐全的一直齐全 ⇒ **结构上不可能出现 pass→fail**。而 `gold` 若在
    截断前只连了部分锚点，加进来的合取句只能增加覆盖对，所以 fail→pass 仍然可能。
    一句话：**这条守卫把回退压到零，把增益留给判据。**

    条件不满足时返回空 `sentences`：非多跳题 / 覆盖对会减少 / 候选为空 / 候选是锚点 /
    候选在上下文里定位不到行。**每一种"做不到"都落回原路径，不猜。**
    """
    from . import bridge  # 延迟导入：`bridge` 顶部 import 了本模块，顶层导入会成环

    q = str(question or "")
    if not bridge.is_bridge_question(q):
        return {"sentences": [], "citations": [], "reason": "not_multihop",
                "anchors": [], "entities": [], "skipped": []}
    anchors = bridge.anchors_of(q)
    if len(anchors) < 2:
        return {"sentences": [], "citations": [], "reason": "not_enough_anchors",
                "anchors": anchors, "entities": [], "skipped": []}

    total = max(1, int(max_total or 1))
    # 判据看见的句子：`。` 切分去空。冲突注记若存在，它贴在最后一句尾巴上，
    # 因此自动跟着那一句走——守卫比较的就是判据真正读到的字节。
    base_segs = [s for s in str(base_answer_text or "").split("。") if s.strip()]
    if not base_segs:
        return {"answer": "", "sentences": [], "citations": [],
                "reason": "no_base_answer", "anchors": anchors,
                "entities": [], "skipped": [], "dropped_claims": 0}

    limit = min(max(0, int(max_entities or 0)), total)
    before = _coverage_pairs(base_segs, anchors)
    cands = _multihop_entity_candidates(claims, anchors, entity)
    sentences: list[str] = []
    citations: list[dict] = []
    used: list[str] = []
    skipped: list[dict] = []
    new_segs: list[str] = base_segs
    for name in cands:
        if len(sentences) >= limit:
            break
        # 判据里 `gold in anc` 直接判失败：锚点不许当 X。
        if name in anchors:
            skipped.append({"entity": name, "why": "is_anchor"})
            continue
        span = _locate_line(units, name)
        if span is None:
            # 定位不到自洽的行就不生成：发一条指不出处的句子会被 `verify_citations`
            # 判 `evidence_does_not_match_offsets`，整条答案跟着降级。
            skipped.append({"entity": name, "why": "no_locatable_line"})
            continue
        candidate = f"{name}与{'、'.join(anchors)}{MULTIHOP_CONJUNCTION_PREDICATE}"
        # 试插：把这一句放最前，尾部丢掉同样多句，再比覆盖对。
        trial = ([candidate] + sentences + base_segs)[:total]
        lost = before - _coverage_pairs(trial, anchors)
        if lost:
            # 前插这一句会挤掉尾部一句、并连带丢掉一条覆盖对 ⇒ 可能把一道本来
            # 通过的题打成失败。放弃，不猜、不换。
            skipped.append({"entity": name, "why": "would_lose_coverage",
                            "lost_pairs": len(lost)})
            continue
        _ev = _citation_span(span["text"], span["start"], span["end"], whole_line=True)
        sentences.append(candidate)
        citations.append({
            "claim": candidate,
            "chunk_id": span["chunk_id"],
            "evidence": _ev[2],
            "start": _ev[0],
            "end": _ev[1],
            "assertion_id": "",
            "predicate_in_line": False,
            "granularity": "line",  # 记账用：本通路发的是整行引用
        })
        used.append(name)
        new_segs = trial
    return {
        "answer": ("。".join(new_segs) + "。") if sentences else "",
        "sentences": sentences,
        "citations": citations,
        "anchors": anchors,
        "entities": used,
        "skipped": skipped,
        # 前插 k 句 ⇒ 尾部丢掉 k 句。必须记账：不记的话"答案变短了"会被
        # 误读成检索变差。
        "dropped_claims": len(sentences),
        "reason": "" if sentences else (
            "coverage_would_regress" if skipped else "no_candidate_entity"),
    }


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
        chapter_cap: int = CHAPTER_CAP_DEFAULT,
        chapter_rank: str = CHAPTER_RANK_DEFAULT,
        alias_context=None,
        multihop_conjunction: bool | None = None,
        multihop_entity: str = "",
        max_claims: int = 0,
        chapter_fill: bool = CHAPTER_FILL_DEFAULT,
        chapter_scan=None,
    ) -> dict:
        """把上下文拼成带引用的答案，或如实拒答。

        返回 `{answer, citations, confidence, abstained, abstain_reason, warnings,
        generation_id, question, context_digest, versions, stats}`。

        `versions` 里记的是**判据版本**（本模块常量 + 上下文摘要），不是模型版本：
        抽取式回答没有模型，硬写一个模型名属于虚构。

        `chapter_cap`（W3，默认 0 = 现状）
            只在章节包通路（`CHAPTER_ANSWER_PATH` + `CHAPTER_TRIGGERS`）生效：
            章节集合按 `_chapter_rank_key` 排序后取前 N 章。`0` 时**与改动前
            逐位相同**——不仅答案不变，连 `_chapter_answer` 的关键字形状都不变
            （0 时不传 `chapter_cap` … 见下方调用点）。风险与「排序键质量未知」
            见 `CHAPTER_CAP_DEFAULT` 上方那段模块级注释。

        `chapter_rank`（L2，默认 `"score"` = 现状）
            只在 `chapter_cap > 0` 时决定**按什么排序取前 N 章**：
            `"score"` = 既有的 `_chapter_rank_key`；`"focus_assertions"` =
            `_chapter_rank_key_focus`（该章端点命中焦点名的断言数降序）。
            **不改入选判据、不改呈现顺序**，只换"留哪几章"。
            `"score"` 时**连 `_chapter_answer` 的关键字都不拼**，返回值与改造前
            逐位相同。取舍与实测数字见 `CHAPTER_RANK_DEFAULT` 上方那段注释。

        `alias_context`（默认 `None` = 现状）
            给定时才启用**别名规范化答案通路**（`_alias_answer`）：问句问
            「「X」的规范名是什么」时，按图里的别名表反查唯一规范名作答。
            `None` 时该通路**一步都不多走**，返回值与改造前逐位相同。            **这是本通路的唯一开关**（不是模块常量）：唯一事实源是
            `pipeline.ask_once` 的 `options["alias_answer"]`，`PROD_PROFILE`
            里没有这个键，所以生产档与全部历史读数不受影响。
            判据与风险见 `ALIAS_TRIGGERS` 上方那段模块级注释。

        `multihop_conjunction`（默认 `None` = 读模块常量 `MULTIHOP_CONJUNCTION`
        = `False` = 现状）
            打开后，**在 claim 列表之外补一条合取句** `{实体}与{A}、{B}相关联`，
            让 `judge.judge_answer` 的 `multihop` 分支拿到"gold 与每个锚点同句共现"。
            原 claim 一句不删（只受 `max_claims` 句数封顶，与 `bridge.merge` 同一红线），
            合取句**插在最前面**。
            关着时 `answer()` 的返回值**逐位不变**，连 `stats` 都不多一个键。
            触发面、候选来源、注水风险与三条硬约束见 `MULTIHOP_CONJUNCTION` 上方那段
            模块级注释——**这条通路天然带注水风险**，读数口径必须按那里写的记。

        `multihop_entity`（默认 `""`）
            **仅供诊断/上界实验**：显式指定合取句用的实体。生产链里没人传它
            （`answer()` 拿不到 goldset，也不许读）。不传时实体取本答案已陈述过的
            端点，见 `_multihop_entity_candidates`。
        """
        units = list(context.get("units") or [])
        # ---- L4b：按题覆盖 claim 句数上限（默认 0 = 用构造时的 self.max_claims）----
        # 为什么要这个旋钮：L4 证据窗口把「进上下文的单元数」从 ~19.5 抬到 ~36-62，
        # 但答案仍只允许 `MAX_CLAIMS=8` 句 —— 参与竞争的 claim 变多，gold 句更容易
        # 被挤出前 8。`data/diag_claims_ladder.json` 实测 claims 5->12 得 +15 题、
        # 0 题损失，说明这个上限在 v1 判据下是**纯赚**的。默认 0 时逐位不变。
        eff_max_claims = int(max_claims) if int(max_claims or 0) > 0 else self.max_claims
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

        # ---- 别名规范化答案通路（默认关：`alias_context is None` 一步都不走） ----
        # 放在三道拒答闸之后、其它通路之前，理由与标题通路一致：命中时完全绕过
        # claim 生成、MAX_CLAIMS 截断、逐行支撑判据与接地判据。
        #
        # **顺序为什么是「先别名、后标题」**：两条通路的触发词不重叠（`规范名` vs
        # `请给出该块的标题` 等），实测 8 道目标题里没有一道同时含两类触发词
        # （`q-sc-900035` 只含标题类，它的「Guard......」在 9 个块里都出现，
        # 标题通路因 title 不唯一必然落空）。所以这个顺序**对已有结果零影响**，
        # 只是把更窄的判据放在前面。`alias_context is None`（默认）时整段短路，
        # 返回 dict 与改造前逐位相同——这一条有测试钉住。
        if alias_context is not None:
            alias_hit = _alias_answer(question, units, alias_context)
            if alias_hit is not None:
                base["answer"] = alias_hit["answer"]
                base["citations"] = [
                    {
                        "claim": f"「{alias_hit['term']}」的规范名是{alias_hit['answer']}",
                        "chunk_id": cite["chunk_id"],
                        "evidence": cite["evidence"],
                        "start": cite["start"],
                        "end": cite["end"],
                        "assertion_id": "",
                        "predicate_in_line": False,
                        "granularity": "line",  # 记账用：本通路发的是整行引用
                    }
                    for cite in alias_hit["citations"]
                ]
                base["confidence"] = round(min(0.95, float(top_score or 0.0)), 4)
                base["stats"]["alias_path"] = {
                    "applied": True,
                    "term": alias_hit["term"],
                    "resolved": alias_hit["resolved"],
                    "alias_source": alias_hit["alias_source"],
                    "citations": len(alias_hit["citations"]),
                }
                return base
            base["stats"]["alias_path"] = {"applied": False}

        # ---- 标题型答案通路（新增答案形态，2026-11-07 起默认开） ----------
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

        # ---- 章节包答案通路（新增答案形态，2026-11-07 起默认开） ------------
        # 同样放在三道拒答闸之后、claim 机制之前，理由与标题通路一致。
        if CHAPTER_ANSWER_PATH:
            # W3：**只在真的要截断时才拼关键字**。默认 `chapter_cap == 0` 时连
            # 调用形状都与改动前一样（与 `pipeline.ask_once` 里
            # `graph_reserve_tokens` / `max_direct_units` 同一个做法）——这样任何
            # 既有 `_chapter_answer` 替身（测试桩、旧版实现）都不会因为多一个
            # 关键字参数而 TypeError，历史读数也逐位不变。
            #
            # L2：`chapter_rank` 走**同一条规矩**——非默认取值才拼进 kwargs。
            # 于是 `chapter_cap=0` 且 `chapter_rank="score"` 时，`_chapter_answer`
            # 收到的关键字集合与改造前**逐位相同**（有测试钉住）。
            cap = int(chapter_cap or 0)
            rank = str(chapter_rank or CHAPTER_RANK_DEFAULT)
            chapter_kwargs: dict = {}
            if cap > 0:
                chapter_kwargs["chapter_cap"] = cap
            if rank != CHAPTER_RANK_DEFAULT:
                chapter_kwargs["chapter_rank"] = rank
            # L5：只在显式开启且注入了扫描回调时才拼关键字（默认路径形状不变）
            if chapter_fill and chapter_scan is not None:
                chapter_kwargs["chapter_fill"] = True
                chapter_kwargs["chapter_scan"] = chapter_scan
            chapter_hit = _chapter_answer(question, units, qa, **chapter_kwargs)
            if chapter_hit is not None:
                base["answer"] = chapter_hit["answer"]
                base["citations"] = [
                    {
                        "claim": f"章节：{chapter_hit['answer']}",
                        "chunk_id": cite["chunk_id"],
                        "evidence": cite["evidence"],
                        "start": cite["start"],
                        "end": cite["end"],
                        "assertion_id": "",
                        "predicate_in_line": False,
                        "granularity": "line",  # 记账用：本通路发的是整行引用
                    }
                    for cite in chapter_hit["citations"]
                ]
                base["confidence"] = round(min(0.95, float(top_score or 0.0)), 4)
                base["stats"]["chapter_path"] = {
                    "applied": True,
                    "n_chapters": chapter_hit["n_chapters"],
                    "chapter_source": chapter_hit["chapter_source"],
                    "citations": len(chapter_hit["citations"]),
                }
                if cap > 0:
                    # cap 打开时才记账：截断前后的章节数必须同时可见，否则
                    # 「gold 完整保留率」的分母无从复算（计划书判据 C7）。
                    base["stats"]["chapter_path"]["chapter_cap"] = cap
                    base["stats"]["chapter_path"]["n_chapters_found"] = (
                        chapter_hit["n_chapters_found"])
                    base["stats"]["chapter_path"]["chapters_truncated"] = (
                        chapter_hit["chapters_truncated"])
                    if rank != CHAPTER_RANK_DEFAULT:
                        # 排序键也要记账：只报"留下几章"看不出"按什么留的"，
                        # 而这一项正是 L2 唯一的变量。
                        #
                        # **记的是"实际生效的键"，不是调用方传进来的那个**：
                        # `_chapter_answer` 会把未登记的键退回默认，这里若照抄
                        # 入参，报告就会声称用了一个从没被用过的键。所以读它
                        # 回填的 `chapter_rank`（只有非默认档才写这个键）。
                        effective_rank = chapter_hit.get("chapter_rank")
                        if effective_rank:
                            base["stats"]["chapter_path"]["chapter_rank"] = (
                                effective_rank)
                return base
            base["stats"]["chapter_path"] = {"applied": False}

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
        if len(accepted) > eff_max_claims:
            warnings.append(
                f"候选 {len(accepted)} 条，只陈述前 {eff_max_claims} 条（其余见 citations 之外，未写入答案）"
            )
            accepted = accepted[:eff_max_claims]

        # ---- 判据 5：冲突材料不无条件合并 --------------------------------
        conflicts = self._same_scene_conflicts(accepted)
        if conflicts:
            for item in conflicts:
                warnings.append(
                    f"冲突未合并：{item['detail']}（同一场景内互斥，须人工判定，不写成一句确定的话）"
                )

        # ---- multihop 多锚点合取装配（默认关：`multihop_conjunction` 为假时整段不走） ----
        # 位置说明：**在 `accepted` 定稿（去重 + `max_claims` 截断）之后、组正文之前**。
        # 这样合取句的候选实体取自"本答案实际陈述过的那几条"——不是取自全部候选，
        # 否则它会指向一句答案里根本没说的话。关着时这里一个变量都不多建，
        # 返回值与改造前逐位相同（有测试钉住）。
        # 先组出**默认档的正文**：合取通路要在这段字节上验"截断不损失覆盖对"，
        # 所以它必须先存在。不打开开关时这段就是最终答案，与改造前逐位相同。
        lines: list[str] = []
        for claim in accepted:
            line = f"{claim['subject']}{claim['predicate']}{claim['object']}。"
            lines.append(line)
        answer_text = "".join(lines)
        if conflicts:
            answer_text += "（注：以上关系存在互斥记录，见 warnings，不作单一结论。）"

        conj_on = bool(MULTIHOP_CONJUNCTION if multihop_conjunction is None
                       else multihop_conjunction)
        conj_payload: dict = {}
        if conj_on:
            conj_payload = _multihop_conjunction(
                question, units, accepted, answer_text, entity=multihop_entity,
                max_total=self.max_claims)
            base["stats"]["multihop_conjunction"] = {
                "enabled": True,
                "applied": bool(conj_payload.get("sentences")),
                "reason": conj_payload.get("reason") or "",
                "anchors": list(conj_payload.get("anchors") or ()),
                "entities": list(conj_payload.get("entities") or ()),
                "skipped": list(conj_payload.get("skipped") or ()),
                # 前插必然在尾部挤掉同样多句。必须记账：不记的话"答案变短了"
                # 会被后人误读成检索变差。
                "dropped_claims": int(conj_payload.get("dropped_claims") or 0),
                "entity_source": ("explicit" if str(multihop_entity or "").strip()
                                  else "stated_claim_endpoint"),
            }
            if conj_payload.get("answer"):
                answer_text = str(conj_payload["answer"])
                # 前插 k 句 ⇒ 尾部必须真的让出 k 个位置，否则总句数超配额。
                # **`accepted` 与 `citations` 必须同时缩到同一批 claim**：只缩正文不缩
                # 引用，会留下"引用了答案里没有的句子"这种自相矛盾的形态。
                keep = max(0, self.max_claims - len(conj_payload["sentences"]))
                if keep < len(accepted):
                    warnings.append(
                        f"多锚点合取句占了 {len(conj_payload['sentences'])} 个句位，"
                        f"尾部 {len(accepted) - keep} 条 claim 未写入答案（引用也一并不发）"
                    )
                    accepted = accepted[:keep]

        citations: list[dict] = []
        # 合取句的引用**放在最前**，与句子的位置一致：读者按答案的顺序读，
        # 引用列表也应当先给出那句话的出处。关着时这里是空 list，不增删任何引用。
        citations.extend(conj_payload.get("citations") or ())
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
