"""判据 v2：三轴判据（value / support / cardinality）。

## 这个文件存在的理由

`judge.py` 是 W4 之前唯一的判据实现，它的三处结构性缺陷是本工作项要治的对象
（计划书 `## W4 · 判据 v2 双口径`）：

1. **无精度项** ⇒ `multihop` 的最优策略是"全列候选"，唯一刹车是 `merge` 的 8 句。
2. **通过判定与 P/R 改进不一致** ⇒ 实测 345 个配对里有 5 个逆序对
   （`q-cx-000011` F1=0.857 判 FAIL、`q-cx-900027` F1=0.667 判 PASS）。
3. **基数闸退化** ⇒ `min|W|=2` 使 `|O|≤2|W|` 在 `|O|≤4` 时恒真。

处置方式是**并列新增**，不是改判据：`judge.py` 一个字节都不动（红线 1），
本模块只是 import 它的辅助函数，把同一个"什么算对"重新拆成三条可机检的轴。

```
judge_v2(category, answer, item, *, context=None, question=None,
         chapter_texts=None, chapter_cap=None, citations=None)
  -> {"pass": bool | None, "reason": str,
      "axes": {"value_ok": bool, "support_ok": bool, "cardinality_ok": bool},
      "loss": {"missing_or_wrong_values": int, "extra_values": int,
               "unsupported_values": int, "cardinality_over": int}}
```

`pass = value_ok AND support_ok AND cardinality_ok`。结构类
（`time_conflict` / `negative_abstain` / `adversarial`）返回 `pass=None`，
与 v1 同义：**剔除分母，不许当 0 也不许当 1**。

## 与 v1 的分工（不许把两件事混在一起）

| 轴 | 判定内容 | 与 v1 的关系 |
|---|---|---|
| `value_ok` | "值对不对"：gold 是否出现在答案里、断言是否覆盖全部锚点、章节是否覆盖 gold 章节、额外章节是否真的提到该实体 | **同源**：逐类照抄 `judge.judge_answer` 的值逻辑，用它的 `_norm` / `_gold_strings` / `chapter_set` / `anchors` / `_sentences`，不另发明一套 |
| `support_ok` | "这个值是不是本次上下文推得出来的"：`答案值 → 本次上下文 assertion_id → citation` | v2 新增。v1 **完全不消费 citations**（`judge_answer` 的签名里没有它），所以形状 oracle 能拿 100% |
| `cardinality_ok` | `multihop`：桥接答案只许 1 个候选值；`cross_chunk`：`|O| ≤ cap` | v2 新增（v1 的 `multihop` 无基数项，`cross_chunk` 的 `|O|≤2|W|` 恒真） |

**唯一一处"值逻辑搬家"（预登记，见 `VALUE_OK_V1_EXCEPTIONS`）**：v1 的
`cross_chunk` 里那条"额外章节数不得超过 gold 章节数"（`|额外| ≤ |W|`）是**基数**
规则，不是值规则，所以它被移到 `cardinality_ok` 里（与 `|O| ≤ cap` 并列保留，
**不是被 cap 替代**）。结果是 `value_ok` 与 `v1 pass` 在"额外章节全为真且
`|额外| > |W|`"这一个格子上不一致——这是唯一登记在案的例外，由
`tests/test_w4_judge_v2.py::TestValueOkIsSameOrigin` 逐条钉住。

## 结构化评测输入（**先定义、再冻结**）

计划书要求先定义 `answer_text / parsed_values / citations / context_assertions /
category / gold` 这套输入并冻结解析规则。`frozen_input()` 就是那份定义，
返回：

```
{"schema": "arkrag.judge_v2.input/1",
 "category": str,
 "answer_text": str,                  # NFKC + strip，与 v1 同一个归一
 "parsed_values": list[str],          # 解析规则见下，纯函数 parse_values()
 "citations": [{"chunk_id","evidence","assertion_id","start","end"}, ...],
 "context_assertions": [{...}, ...],  # 归一后的本次上下文断言
 "context_chunks": {chunk_id: {"chapter","title"}},
 "gold": {"answer","acceptable_answers","values"},
 "anchors": list[str]}                # 只来自 question，与 v1 同一个抽取
```

### 解析规则（冻结；`parse_values()` 是纯函数、可单测）

| 类别 | `parsed_values` |
|---|---|
| `cross_chunk` | `sorted(judge.chapter_set(answer_text))`——与 v1 同一个章节切分器（含空格分隔符那一格） |
| `multihop` | 见下"候选值抽取"，**只从首段裸值 + 关系句的非锚点端点取**，锚点永远不是候选 |
| `single_chunk` / `alias` | 答案里出现的 gold/acceptable 串（与 v1 的"子串命中即等价"同源）；一个都没出现时退回首段原文——**这个回退只为让 `loss` 与 `reason` 有对象可指**，`value_ok` 仍然由 gold 子串判，不吃这里 |
| `direct_relation` | 同上（答案是词表里的关系词，单值） |
| 其他 / 未知 | 同 `single_chunk` 的回退规则 |

三条附加规则（都是为了"不许把同一件事数两遍"）：

* **长值吸收短值**：若 `A ⊂ B` 且两者都在列表里，丢掉 `A`。`凯尔希` 与
  `凯尔希医生` 同时出现时它们不是两条独立答案值。
* **锚点不是候选**：`multihop` 的关系证据句里出现的问句锚点（`「」` 里的东西）
  **一定**不计候选值——这是本文件最容易写错的一条，
  `TestJ6AnchorsAreNeverCandidates` 钉住。
* **背景 claim 不是候选**：一句话两端都不是问句锚点时，它是背景陈述，不是
  "把某个实体当作中间答案提出来"，不计候选（fail-open 方向：少算候选，不多算）。

## `support_ok` 的映射（v2 的核心）

对每个答案值 `v`，要求存在一条链

```
v --(witness)--> 本次上下文里的 assertion / 块属性 --(cited)--> 答案的 citations
```

`witness` 的四个层级（冻结；`SUPPORT_POLICY` 决定每类允许哪几层）：

| 层级 | 判据 | 强度 |
|---|---|---|
| `endpoint` | `v` 等于该断言的 `subject` 或 `object` | 最强：值就是三元组的一端 |
| `predicate` | `v` 等于该断言的 `predicate` | 强：`direct_relation` 的答案就是关系词 |
| `chunk_field` | `v` 等于**被引用的块**的 `chapter` 或 `title` | 中：`single_chunk` 问的是块标题、`cross_chunk` 列的是章节代号，两者都不是三元组端点，没有这一层会让这两类答案**无条件**判无支撑 |
| `evidence` | `v` 出现在该断言的 `evidence` 原文里 | 最弱：提到 ≠ 断言，只作兜底 |

`cited` 的判据（冻结，两层都要过）：

1. **块层**：答案的 citations 里必须有一条指向该断言**所在的块**。
2. **断言层**：若那条 citation 还写了 `assertion_id`，它必须与断言自己的 id 一致；
   对不上（把 A 块的断言说成在 B 块）时**整条引用作废**——既不给它声称的块计入
   "被引用"，也不让那条断言成为 witness。少了第 2 层，一条合格引用就能把错位
   断言重新洗成 witness（反向实测第 2 笔就是这么漏的）。

**缺映射计 `unsupported`，绝不从 gold 反推支撑**：`support_ok` 的实现里没有任何
一处读 `item["answer"]` / `acceptable_answers`。`gold` 只出现在 `value_ok` 与
`parsed_values` 里（后者只是为了定位答案值本身）。

**引用从哪来**：`answer` 是 dict 时取它自己的 `citations`；是正文串时可以另传
`citations=`。**两者都没有 ⇒ `support_ok=False`（`support:no_citations`）**。
`eval_answer_quality.py` 那条调用链传的是答案 dict，所以生产路径上引用是齐的；
但任何只传正文串的调用方都必须知道这个降级是**可数**的（reason 单独一个 token），
不是"悄悄算过"。

**什么情况会漏**（不许省略）：

* `evidence` 层只做子串，普通词与实体名不分（与 `answer._claims_are_grounded` 同一个局限）。
* `chunk_field` 层是最弱的一层：一个块标题被引用，就足以让"这个标题"这个值过关，
  哪怕引用那行根本没提到它。`explain_v2()` 逐值报 `witness_level`，就是要让
  这一层的占比可以被审计，而不是被平均掉。
* 上下文里没带 `chapter` 时（只有断言、没有 units），`cross_chunk` 的章节值
  **全部**判无支撑——这是 fail-closed，不是"通过"。
* `context` 完全没传时同样 fail-closed，且 reason 用 `support:context_missing`
  与真正的"值无支撑"区分开（避免"缺输入"被静默算成"答错了"）。

## `cardinality_ok`

* `multihop`：`MULTIHOP_MAX_CANDIDATES = 1`（预登记的桥接答案只允许 1 个候选值），
  `cardinality_over = max(0, len(parsed_values) - 1)`。
* `cross_chunk`：`|O| ≤ CROSS_CHUNK_CAP_DEFAULT`（W3 校准，当前生产档 4），
  **且**保留 v1 的 `|额外| ≤ |W|`（不能用 cap 替代它）；
  `cardinality_over = max(|O|-cap, |额外|-|W|)`（各自先钳到 ≥0 再取大）。
* 其余三类（`single_chunk` / `alias` / `direct_relation`）：计划书没有预登记基数
  规则，`cardinality_ok` 恒 `True` 且 reason 记 `cardinality:not_registered`。
  **这是一个已知的宽松面**：这三类里"多写几条无关 claim"不受 v2 约束。

## 内置证伪：翻转分类

`classify_flip(v1_verdict, v2_verdict)` 返回 `PP` / `PF` / `FP` / `FF`（结构类或
任一侧 `pass is None` 时返回 `NA`，**不计入四个格子**，否则
`PP+PF+FP+FF` 就与可判分母对不上——计划书反证条件 2）。

`PF`（v1 过、v2 拒）必须能按**互斥归因**分类，取值只有四个（计划书原文）：
`unsupported`（无支撑）/ `entity_precision`（实体精度）/ `chapter_cardinality`
（章节基数）/ `other`（其他明确原因）。互斥靠**一条固定优先级阶梯**实现：
先看 support，再看 multihop 基数，再看章节基数，最后才落 `other`。
归因是单值的；`axes_failed` 同时列出全部未过的轴，供人工复核"主因是不是选错了"。

`FP`（v1 拒、v2 过）**默认必须为 0**。当前设计下 `FP` 在构造上就取不到：
`value_ok` 与 v1 的值逻辑同源、`cardinality_ok` 只收紧不放松，所以
`v2 pass ⇒ v1 pass`（唯一例外是上面那条搬家规则，它把 `|额外|≤|W|` 移进
cardinality 后仍然拦得住）。`REGISTERED_V1_FALSE_NEGATIVES` 是一个空注册表：
将来若真有人让 v2 修正了 v1 的假阴性，必须**逐题**登记 `question_id -> 理由`，
否则 `tally_flips()` 的 `ok` 为 False。这条是计划书 J1 的"`FP=0` 或有逐题批准
的 v1 假阴性修正"的机检形态。

## 已知局限（本工作项没有验证的部分）

1. **没有在真实答案集上跑过。** 本工作项的红线禁止任何加载数据产物的命令
   （`load_generation()` 一次常驻 1.0–1.5 GB，机器已因并发崩过三次），
   所以 `support_ok` 在真语料上的通过率、`witness_level` 的分布、
   `PF` 的归因计数**全部未知**。这里只有合成测试。
2. **`CROSS_CHUNK_CAP_DEFAULT = 4` 是抄 W3 生产档的值，不是本文件校准出来的。**
   W3 的排序键质量本身未经实测（见 `answer.CHAPTER_CAP_DEFAULT` 上方注释），
   所以 cap 一旦被证明会漏 gold，这里的基数轴会跟着一起错。
3. **`multihop` 候选值抽取是启发式**：靠"关系词切分 + 首段裸值 + 非锚点端点"。
   名字里含关系词（`感染者`）、或答案用散文转述（"中间人是阿米娅，她与云兽…"）
   都取不出候选，此时 `cardinality_ok` 退化为 True（fail-open，不冤枉人，也不拦人）。
4. **`value_ok` 与 v1 的同源是"照抄 + 测试钉住"，不是"调用同一个函数"。**
   两者迟早会漂移；`TestValueOkIsSameOrigin` 用一张逐类语料表把漂移变成红，
   但它只覆盖表里那些形状。
"""

from __future__ import annotations

from typing import Any

from . import judge as judge_mod
from . import vocab

__all__ = [
    "CELLS",
    "CROSS_CHUNK_CAP_DEFAULT",
    "INPUT_SCHEMA",
    "MULTIHOP_MAX_CANDIDATES",
    "PF_ATTRIBUTIONS",
    "REGISTERED_V1_FALSE_NEGATIVES",
    "SCHEMA",
    "SUPPORT_LEVELS",
    "SUPPORT_POLICY",
    "VALUE_OK_V1_EXCEPTIONS",
    "classify_flip",
    "context_assertions",
    "explain_v2",
    "frozen_input",
    "judge_v2",
    "normalize_citations",
    "normalize_context",
    "parse_values",
    "parse_values_detail",
    "tally_flips",
]

#: 判据版本串。报告里要与 `judge_schema` 并列记，否则分不清哪份数来自哪套判据。
SCHEMA = "arkrag.judge_v2/1"

#: 结构化输入的 schema 串。
INPUT_SCHEMA = "arkrag.judge_v2.input/1"

#: `cross_chunk` 的章节数上限。**抄 W3 生产档**（`pipeline.PROD_PROFILE` 的
#: `chapter_cap=4`），本文件没有校准它，也不许在这里改标定值——改要改 W3 的
#: 对照表（`data/diag_cx_rank_keys.json`，尚未产出）。
CROSS_CHUNK_CAP_DEFAULT = 4

#: `multihop` 允许的候选值个数：预登记的桥接答案**只允许 1 个**。
MULTIHOP_MAX_CANDIDATES = 1

#: 答案值的最短长度。1 个字的"值"在中文里无法与语气词区分，且会把
#: `contains_anchor` 之类的子串判断变成噪声。
MIN_VALUE_CHARS = 2

#: 裸值（首段候选）的长度上限。生产档桥接通路发的是 `x。x交谈A。x交谈B。`，
#: x 是实体名；超过 30 字的首段不是"候选值"而是句子。
MAX_BARE_VALUE_CHARS = 30

#: `evidence` 层只认原文子串；低于这个长度不做子串匹配（防止单字命中）。
MIN_EVIDENCE_MATCH_CHARS = 2

#: 结构类（`pass is None`），与 v1 同源，不重写一份。
STRUCTURAL_CATEGORIES: tuple[str, ...] = tuple(judge_mod.STRUCTURAL_CATEGORIES)

#: 可判类（要打分的）。
SCORABLE_CATEGORIES: tuple[str, ...] = tuple(judge_mod.SCORABLE_CATEGORIES)

#: 四个翻转格子。结构类不进这四个格子（见 `classify_flip`）。
CELLS: tuple[str, ...] = ("PP", "PF", "FP", "FF")

#: `PF` 的四个互斥归因（计划书原文：无支撑 / 实体精度 / 章节基数 / 其他明确原因）。
PF_ATTRIBUTIONS: tuple[str, ...] = (
    "unsupported",
    "entity_precision",
    "chapter_cardinality",
    "other",
)

#: 预登记的 v1 假阴性修正表：`question_id -> 理由`。
#: **默认空**：当前 v2 在构造上取不到 FP（见模块 docstring）。
REGISTERED_V1_FALSE_NEGATIVES: dict[str, str] = {}

#: `value_ok` 与 `v1 pass` 的预登记例外（搬迁到别的轴的规则）。
#: 键是类别，值是一句人话解释。测试逐条断言"例外只发生在这些格子上"。
VALUE_OK_V1_EXCEPTIONS: dict[str, str] = {
    "cross_chunk": (
        "v1 的 `|额外章节| <= |gold 章节|` 是基数规则，已移入 cardinality 轴"
        "（与 `|O| <= cap` 并列保留）。于是当额外章节全为真且 |额外| > |W| 时，"
        "v1 判 False 而 v2 的 value_ok 为 True —— 这是唯一登记的例外。"
    ),
}

#: `support_ok` 的层级名（强 → 弱只用于解释，判定是"任一层成立即可"）。
SUPPORT_LEVELS: tuple[str, ...] = ("endpoint", "predicate", "chunk_field", "evidence")

#: 每类允许的 witness 层级（冻结）。**顺序即优先级**，`explain_v2` 报首个命中层。
SUPPORT_POLICY: dict[str, tuple[str, ...]] = {
    # 问的是"这句话出自哪个文本块/这个称呼的规范名是什么"：标题与规范名都不是
    # 三元组端点，必须有 `chunk_field`，否则这两类会被无条件判无支撑。
    "single_chunk": ("endpoint", "chunk_field", "evidence"),
    "alias": ("endpoint", "chunk_field", "evidence"),
    # 答案是词表里的关系词 ⇒ 只能是断言自己的 predicate。
    "direct_relation": ("predicate", "endpoint", "evidence"),
    # 桥接实体必须是一个端点（或至少出现在被引用的证据原文里）；
    # 不接受 `predicate`/`chunk_field`——"某个块叫这个名字"不是桥接支撑。
    "multihop": ("endpoint", "evidence"),
    # 列的是章节代号 ⇒ 章节属性是主层级。
    "cross_chunk": ("chunk_field", "endpoint", "evidence"),
}

#: `reason` 串里的固定片段。与 v1 的词表尽量同名，便于两套判定并列对照。
REASON_EMPTY = "empty"
REASON_STRUCTURAL = "structural:not_supported"
REASON_ALL_PASS = "all_axes_pass"

_TRIM = " \t\r\n，,、；;：:。！？!?（）()「」“”\"'"

_ZERO_LOSS = {
    "missing_or_wrong_values": 0,
    "extra_values": 0,
    "unsupported_values": 0,
    "cardinality_over": 0,
}


# ---------------------------------------------------------------- 归一工具


def _norm(value: Any) -> str:
    """交给 v1 的归一（NFKC + strip），保证两套判据对同一串字面等价。"""
    return judge_mod._norm(value)


def _gold_values(category: str, item: dict) -> list[str]:
    """注册答案的"值"列表。

    `cross_chunk` 用章节集合（与 v1 同一个 `chapter_set`），其余类用
    `judge._gold_strings`（gold + acceptable_answers，去重保序）。
    """
    if category == "cross_chunk":
        return sorted(judge_mod.chapter_set((item or {}).get("answer")))
    return judge_mod._gold_strings(item or {})


def _answer_text(answer: Any) -> str:
    """从 `str` 或答案 dict 里取答案正文。"""
    if isinstance(answer, dict):
        raw = answer.get("answer")
        if raw is None:
            raw = answer.get("text")
        return _norm(raw)
    return _norm(answer)


def normalize_citations(raw: Any) -> list[dict]:
    """把答案的 citations 归一成冻结形状。

    只做"形状归一 + 去噪"，**不校验偏移**：偏移是否逐字对上、引用是否落在它
    声称的章节，是 `judge.citations_support` 的职责（单一事实源，不许在这里
    再写一份）。这里只关心"这条引用指向哪个块的哪条断言"。
    """
    out: list[dict] = []
    for entry in raw or ():
        if not isinstance(entry, dict):
            continue
        out.append(
            {
                "chunk_id": _norm(entry.get("chunk_id")),
                "evidence": _norm(entry.get("evidence")),
                "assertion_id": _norm(entry.get("assertion_id")),
                "start": entry.get("start"),
                "end": entry.get("end"),
            }
        )
    return out


def _assertion_record(raw: dict, chunk_id: str, chunk_meta: dict) -> dict | None:
    if not isinstance(raw, dict):
        return None
    rec = {
        "assertion_id": _norm(raw.get("assertion_id")),
        "chunk_id": _norm(raw.get("chunk_id")) or _norm(chunk_id),
        "chapter": _norm(raw.get("chapter")) or _norm((chunk_meta or {}).get("chapter")),
        "title": _norm(raw.get("title")) or _norm((chunk_meta or {}).get("title")),
        "subject": _norm(raw.get("subject")),
        "predicate": _norm(raw.get("predicate")),
        "object": _norm(raw.get("object")),
        "evidence": _norm(raw.get("evidence") or raw.get("evidence_text")),
    }
    if not any(rec[k] for k in ("subject", "predicate", "object", "evidence")):
        # 什么都不含的记录不构成任何 witness，留着只会让计数虚高。
        return None
    return rec


def normalize_context(context: Any) -> dict:
    """把"本次上下文"归一成 `{assertions: [...], chunks: {chunk_id: {chapter,title}}}`。

    接受四种形态（生产链路上都会出现）：

    * `None` —— 没给上下文（fail-closed，全部值判无支撑）；
    * `list[dict]` —— 直接一串断言；
    * `{"units": [{"chunk_id","chapter","title","assertions":[...]}, ...]}`
      —— `pipeline.ask_once` 的 `res["context"]`；
    * `{"assertions": [...]}` —— 只要断言、不要块属性。

    断言记录里自带的 `chapter` / `title` 会被并进 `chunks`，这样只有断言列表的
    调用方也能用上 `chunk_field` 层（否则 `cross_chunk` 的章节值会整批判无支撑，
    而那是输入形状问题，不是答案问题——但这种降级必须**显式**，见 reason）。
    """
    assertions: list[dict] = []
    chunks: dict[str, dict] = {}

    def add_assertion(raw: Any, chunk_id: str = "", meta: dict | None = None) -> None:
        rec = _assertion_record(raw, chunk_id, meta or {})
        if rec is None:
            return
        assertions.append(rec)
        cid = rec["chunk_id"]
        if cid and (rec["chapter"] or rec["title"]):
            slot = chunks.setdefault(cid, {"chapter": "", "title": ""})
            slot["chapter"] = slot["chapter"] or rec["chapter"]
            slot["title"] = slot["title"] or rec["title"]

    if context is None:
        return {"assertions": [], "chunks": {}}
    if isinstance(context, (list, tuple)):
        for raw in context:
            add_assertion(raw)
    elif isinstance(context, dict):
        units = context.get("units")
        if isinstance(units, (list, tuple)):
            for unit in units:
                if not isinstance(unit, dict):
                    continue
                cid = _norm(unit.get("chunk_id"))
                meta = {
                    "chapter": _norm(unit.get("chapter")),
                    "title": _norm(unit.get("title")),
                }
                if cid and (meta["chapter"] or meta["title"]):
                    slot = chunks.setdefault(cid, {"chapter": "", "title": ""})
                    slot["chapter"] = slot["chapter"] or meta["chapter"]
                    slot["title"] = slot["title"] or meta["title"]
                for raw in unit.get("assertions") or ():
                    add_assertion(raw, cid, meta)
        for raw in context.get("assertions") or ():
            add_assertion(raw)

    assertions.sort(
        key=lambda r: (
            r["chunk_id"],
            r["assertion_id"],
            r["subject"],
            r["predicate"],
            r["object"],
            r["evidence"],
        )
    )
    # 完全重复的断言（同 id 同内容）只留一条：它们不提供额外证据，只会让
    # "有几条支撑"这类计数虚高。
    deduped: list[dict] = []
    for rec in assertions:
        if not deduped or deduped[-1] != rec:
            deduped.append(rec)
    return {"assertions": deduped, "chunks": chunks}


def context_assertions(context: Any) -> list[dict]:
    """`normalize_context()` 的断言投影（供冻结输入使用）。"""
    return normalize_context(context)["assertions"]


# ---------------------------------------------------------------- 解析规则


def _segments(text: str) -> list[str]:
    """按 `。` 与换行切段，保留顺序。

    分隔符与 v1 的 `judge._SENTENCE_SPLIT`（`。`）对齐；换行也要切，因为
    `bridge.merge` 之外的答案形态里换行同样表示"另一句"。
    """
    out: list[str] = []
    for chunk in text.replace("\r\n", "\n").replace("\r", "\n").split("。"):
        for line in chunk.split("\n"):
            piece = line.strip()
            if piece:
                out.append(piece)
    return out


def _first_segment(text: str) -> str:
    segs = _segments(text)
    return segs[0].strip(_TRIM) if segs else ""


def _is_anchor(name: str, anchors: list[str]) -> bool:
    return bool(name) and name in anchors


def _contains_anchor(text: str, anchors: list[str]) -> bool:
    return any(a and a in text for a in anchors)


def _split_relation(segment: str) -> tuple[str, str, str] | None:
    """把一句话切成 `(主语, 关系词, 宾语)`。

    关系词表**只从 `vocab.CANONICAL_RELATIONS` 取**，不在本文件里手抄一份
    （本项目的"同一判定写两处"事故形态）。取"最早出现"的关系词；同一位置取更长的
    那个，这样 `受保护于` 不会被 `保护` 抢先切开。
    主语为空（关系词在句首）时返回 `None`——没有主语就没有候选值。
    """
    best: tuple[int, str] | None = None
    for word in vocab.CANONICAL_RELATIONS:
        pos = segment.find(word)
        if pos <= 0:
            continue
        if best is None or (pos, -len(word)) < (best[0], -len(best[1])):
            best = (pos, word)
    if best is None:
        return None
    pos, word = best
    subject = segment[:pos].strip(_TRIM)
    obj = segment[pos + len(word):].strip(_TRIM)
    if not subject or not obj:
        return None
    return subject, word, obj


def _collapse(values: list[dict]) -> list[dict]:
    """去重 + 长值吸收短值（保序）。

    `凯尔希` 与 `凯尔希医生` 同时出现时，短的那个不构成独立答案值——留着会让
    `unsupported_values` / `extra_values` 两个计数同时虚高。
    """
    kept: list[dict] = []
    for rec in values:
        value = _norm(rec["value"])
        if not value:
            continue
        if any(value == k["value"] for k in kept):
            continue
        kept.append({"value": value, "source": rec["source"]})
    final: list[dict] = []
    for rec in kept:
        absorbed = any(rec["value"] != other["value"] and rec["value"] in other["value"]
                       for other in kept)
        if not absorbed:
            final.append(rec)
    return final


def _multihop_values(text: str, anchors: list[str]) -> list[dict]:
    """`multihop` 的候选值抽取（冻结规则，见模块 docstring）。

    只有两个来源：

    1. **首段裸值**：第一段不含关系词、不是锚点、长度 ≤ `MAX_BARE_VALUE_CHARS`。
       生产档桥接通路发的正是这个形状（`x。x交谈A。x交谈B。`）。
    2. **关系句的非锚点端点**：一句话两端一端是问句锚点、另一端不是 ⇒ 另一端
       就是这次提出的候选。两端都不是锚点的是背景 claim，不计候选。

    **锚点永远不是候选**：这一条是本文件最容易写错的地方——把
    `变形者集群交谈凯尔希` 里的 `变形者集群` 算成"额外候选"，就会把 v1 判对的
    答案在 v2 里误拒（`TestJ6AnchorIsNotCandidate`）。
    """
    out: list[dict] = []
    for index, segment in enumerate(_segments(text)):
        piece = segment.strip(_TRIM)
        if not piece or piece in anchors:
            continue
        split = _split_relation(piece)
        if split is None:
            if index == 0 and len(piece) <= MAX_BARE_VALUE_CHARS:
                out.append({"value": piece, "source": "bare_segment"})
            continue
        subject, _predicate, obj = split
        subject_is_anchor = _is_anchor(subject, anchors)
        object_is_anchor = _is_anchor(obj, anchors)
        if subject_is_anchor and object_is_anchor:
            continue
        if subject_is_anchor:
            candidate = obj
        elif object_is_anchor:
            candidate = subject
        else:
            # 两端都不是问句锚点：这是背景陈述，不是"把某个实体当作中间答案提出来"。
            continue
        if _is_anchor(candidate, anchors) or _contains_anchor(candidate, anchors):
            continue
        if len(candidate) < MIN_VALUE_CHARS:
            continue
        out.append({"value": candidate, "source": "relation_endpoint"})
    return _collapse(out)


def parse_values_detail(
    category: str,
    answer_text: Any,
    *,
    question: Any = None,
    item: dict | None = None,
) -> list[dict]:
    """解析答案值并保留来源（`source`）。纯函数：不读盘、不改入参、无全局状态。"""
    cat = str(category or "")
    text = _norm(answer_text)
    if not text:
        return []
    if cat == "cross_chunk":
        return [{"value": ch, "source": "chapter"} for ch in sorted(judge_mod.chapter_set(text))]

    q_source = question if question is not None else (item or {}).get("question")
    anchors = judge_mod.anchors(_norm(q_source))
    if cat == "multihop":
        return _multihop_values(text, anchors)

    golds = _gold_values(cat, item or {})
    hits = [{"value": g, "source": "gold_substring"} for g in golds if g and g in text]
    if hits:
        return _collapse(hits)
    fallback = _first_segment(text)
    if fallback:
        return [{"value": fallback, "source": "answer_segment"}]
    return []


def parse_values(
    category: str,
    answer_text: Any,
    *,
    question: Any = None,
    item: dict | None = None,
) -> list[str]:
    """`parse_values_detail()` 的值投影（冻结输入里的 `parsed_values`）。"""
    return [rec["value"] for rec in parse_values_detail(
        category, answer_text, question=question, item=item)]


def frozen_input(
    category: str,
    answer: Any,
    item: dict,
    *,
    context: Any = None,
    question: Any = None,
    citations: Any = None,
) -> dict:
    """构造冻结的结构化评测输入（计划书要求先定义的那一套形状）。

    这是**唯一的**输入归一入口：`judge_v2` 与 `explain_v2` 都只吃它的输出，
    测试也可以直接对它的字段做断言。

    `answer` 是 `dict` 时引用从它里面取；是 `str` 时引用只能来自显式的
    `citations` 参数。**两者都没有 ⇒ `citations=[]` ⇒ `support_ok=False`**。
    这条降级是刻意的（支撑不能凭空造），但调用方必须知道：只传答案正文等于
    宣布"这条答案没有引用支撑"，`reason` 会写 `support:no_citations`。
    """
    cat = str(category or "")
    it = dict(item or {})
    text = _answer_text(answer)
    if citations is not None:
        raw_citations = citations
    elif isinstance(answer, dict):
        raw_citations = answer.get("citations")
    else:
        raw_citations = None
    cites = normalize_citations(raw_citations)
    ctx = normalize_context(context)
    q_source = question if question is not None else it.get("question")
    return {
        "schema": INPUT_SCHEMA,
        "category": cat,
        "answer_text": text,
        "parsed_values": parse_values(cat, text, question=q_source, item=it),
        "citations": cites,
        "context_assertions": ctx["assertions"],
        "context_chunks": ctx["chunks"],
        "gold": {
            "answer": _norm(it.get("answer")),
            "acceptable_answers": [_norm(x) for x in (it.get("acceptable_answers") or ())],
            "values": _gold_values(cat, it),
        },
        "anchors": judge_mod.anchors(_norm(q_source)),
    }


# ---------------------------------------------------------------- 三条轴


def _value_axis(
    category: str,
    text: str,
    item: dict,
    *,
    question: Any,
    chapter_texts: dict | None,
) -> tuple[bool, str]:
    """`value_ok`：逐类照抄 v1 的值逻辑（同源），返回 `(ok, reason_token)`。

    **故意不包含任何基数规则**：`cross_chunk` 的 `|额外| ≤ |W|` 已登记搬到
    `cardinality_ok`（`VALUE_OK_V1_EXCEPTIONS`），`multihop` 的"1 个候选值"
    更是 v2 新增的。如果把基数混进来，`PF` 就没法按轴归因了。
    """
    cat = str(category or "")
    golds = _gold_values(cat, item)
    if cat in ("single_chunk", "alias"):
        if not golds:
            return False, "no_gold"
        if any(g in text for g in golds):
            return True, "gold_in_answer"
        return False, "gold_absent"
    if cat == "cross_chunk":
        want = judge_mod.chapter_set((item or {}).get("answer"))
        got = judge_mod.chapter_set(text)
        if not want:
            return False, "no_gold"
        if got == want:
            return True, "chapter_set_equal"
        if not (want <= got):
            return False, "chapter_set_missing"
        extra = got - want
        if chapter_texts is None:
            # 与 v1 同一条规矩：不许在没查正文时假装做过真实性校验。
            return False, "extra_chapters_unverified"
        q = _norm(question if question is not None else (item or {}).get("question"))
        name = next((a for a in judge_mod.anchors(q) if a and a not in want), "")
        if not name:
            return False, "no_entity_to_verify_extra"
        texts = chapter_texts or {}
        for ch in extra:
            if ch not in texts or name not in texts[ch]:
                return False, "extra_chapter_does_not_mention_entity"
        return True, "extra_chapters_all_real"

    q = _norm(question if question is not None else (item or {}).get("question"))
    anc = judge_mod.anchors(q)
    gold = golds[0] if golds else ""
    sents = judge_mod._sentences(text)
    if cat == "direct_relation":
        if not anc or not gold:
            return False, "no_anchor_or_gold"
        for sentence in sents:
            if all(a in sentence for a in anc) and gold in sentence:
                return True, "one_claim_all_anchors_and_gold"
        return False, "no_single_claim_with_all_anchors"
    if cat == "multihop":
        if not anc or not gold:
            return False, "no_anchor_or_gold"
        if gold in anc:
            return False, "gold_is_an_anchor"
        covered = [a for a in anc if any(a in s and gold in s for s in sents)]
        if not covered:
            return False, "no_claim_links_gold_to_any_anchor"
        if len(covered) < len(anc):
            return False, "gold_not_linked_to_every_anchor"
        return True, "gold_linked_to_every_anchor"
    return False, f"unknown_category:{cat}"


class _CitedIndex:
    """答案的 citations 折算成"哪些块被引用过、哪些引用自相矛盾"。

    一条引用只要同时写了 `chunk_id` 与 `assertion_id`，两者就必须指向同一个块；
    对不上时这条引用**一个比特都不贡献**——既不给它声称的块计入"被引用"，也不让
    那条断言成为 witness。理由：一条"我引的是 X 块"而证据其实住在 Y 块的引用，
    它声称的"X 块支持这个答案"本身就不成立，把它算进去等于给错位洗白。
    """

    def __init__(self, citations: list[dict], by_id: dict[str, dict]) -> None:
        self.chunks: set[str] = set()
        self.misplaced: set[str] = set()
        for cite in citations:
            cid = cite.get("chunk_id") or ""
            aid = cite.get("assertion_id") or ""
            if aid and aid in by_id:
                owner_chunk = by_id[aid].get("chunk_id") or ""
                if cid and owner_chunk and cid != owner_chunk:
                    self.misplaced.add(aid)
                    continue
            if cid:
                self.chunks.add(cid)

    def allows(self, assertion: dict) -> bool:
        """这条断言能不能作 witness：它所在的块必须被引用过，且 id 没被引错位置。"""
        aid = assertion.get("assertion_id") or ""
        if aid and aid in self.misplaced:
            return False
        cid = assertion.get("chunk_id") or ""
        return bool(cid) and cid in self.chunks


def _level_hits(
    level: str,
    value: str,
    assertion: dict,
) -> bool:
    if level == "endpoint":
        return bool(value) and value in (assertion.get("subject") or "",
                                        assertion.get("object") or "")
    if level == "predicate":
        return bool(value) and value == (assertion.get("predicate") or "")
    if level == "evidence":
        evidence = assertion.get("evidence") or ""
        return bool(evidence) and len(value) >= MIN_EVIDENCE_MATCH_CHARS and value in evidence
    return False


def _witness(
    value: str,
    category: str,
    assertions: list[dict],
    chunks: dict[str, dict],
    cited: _CitedIndex,
) -> dict:
    """给一个答案值找 witness，返回 `{witness_level, assertion_id, chunk_id}`。

    找不到时 `witness_level` 为空串（**不是** `None`：`explain_v2` 与 counting
    两边都按"空串 = 没找到"判，混进 None 会让 `bool()` 判断悄悄反过来）。
    """
    policy = SUPPORT_POLICY.get(category, ("endpoint", "evidence"))
    for level in policy:
        if level == "chunk_field":
            for cid in sorted(cited.chunks):
                meta = chunks.get(cid) or {}
                fields = {meta.get("chapter") or "", meta.get("title") or ""} - {""}
                if value and value in fields:
                    return {"witness_level": level, "assertion_id": "", "chunk_id": cid}
            continue
        for assertion in assertions:
            if not cited.allows(assertion):
                continue
            if _level_hits(level, value, assertion):
                return {
                    "witness_level": level,
                    "assertion_id": assertion.get("assertion_id") or "",
                    "chunk_id": assertion.get("chunk_id") or "",
                }
    return {"witness_level": "", "assertion_id": "", "chunk_id": ""}


def _support_axis(
    category: str,
    values: list[str],
    bundle: dict,
) -> tuple[bool, str, list[dict], int]:
    """`support_ok`：每条答案值都要有"本次上下文 → 被引用"的 witness。

    返回 `(ok, reason_token, per_value, unsupported_count)`。
    **实现里不读 gold**——support 只能从 `context` 与 `citations` 推出来。
    """
    assertions = bundle["context_assertions"]
    chunks = bundle["context_chunks"]
    citations = bundle["citations"]
    by_id = {a["assertion_id"]: a for a in assertions if a.get("assertion_id")}
    cited = _CitedIndex(citations, by_id)

    if not values:
        # 没有答案值就没有"要支撑的东西"：空真通过。**不能**把这个当成"查过了"——
        # `judge_v2` 的 `loss.unsupported_values` 因此也是 0，读报告的人要配合
        # `parsed_values` 一起看（解析失败会让两个数同时为 0）。
        return True, "", [], 0
    if not assertions and not chunks:
        # 上下文没给（或给空）：fail-closed，且用独立 reason 与"值无支撑"区分开。
        per_value = [{"value": v, "witness_level": "", "assertion_id": "", "chunk_id": ""}
                     for v in values]
        return False, "context_missing", per_value, len(values)
    if not citations:
        per_value = [{"value": v, "witness_level": "", "assertion_id": "", "chunk_id": ""}
                     for v in values]
        return False, "no_citations", per_value, len(values)

    per_value: list[dict] = []
    unsupported = 0
    for value in values:
        hit = _witness(value, category, assertions, chunks, cited)
        per_value.append({"value": value, **hit})
        if not hit["witness_level"]:
            unsupported += 1
    if unsupported:
        return False, f"unsupported_values={unsupported}", per_value, unsupported
    return True, "", per_value, 0


def _cardinality_axis(
    category: str,
    text: str,
    values: list[str],
    item: dict,
    *,
    cap: int,
) -> tuple[bool, str, int, dict]:
    """`cardinality_ok`：只对 `multihop` / `cross_chunk` 有预登记规则。

    返回 `(ok, reason_token, over, detail)`。
    """
    cat = str(category or "")
    if cat == "multihop":
        over = max(0, len(values) - MULTIHOP_MAX_CANDIDATES)
        detail = {"allowed": MULTIHOP_MAX_CANDIDATES, "got": len(values), "over": over}
        if over:
            return False, f"over={over}", over, detail
        return True, "", 0, detail
    if cat == "cross_chunk":
        want = judge_mod.chapter_set((item or {}).get("answer"))
        got = judge_mod.chapter_set(text)
        extra = got - want
        cap_over = max(0, len(got) - cap)
        # v1 的 `|额外| <= |W|` **不能**被 cap 替代（计划书原文），两条一起保留。
        rel_over = max(0, len(extra) - len(want)) if want else 0
        over = max(cap_over, rel_over)
        detail = {
            "cap": cap,
            "got": len(got),
            "want": len(want),
            "extra": len(extra),
            "cap_over": cap_over,
            "relative_over": rel_over,
            "over": over,
        }
        if over:
            return False, f"over={over}", over, detail
        return True, "", 0, detail
    # 其余三类没有预登记的基数规则：axis 恒 True，但 reason 里标注出来，
    # 免得读报告的人以为"基数项过了"意味着这里查过什么。
    return True, "not_registered", 0, {"allowed": None, "got": len(values), "over": 0}


def _loss(
    category: str,
    text: str,
    values: list[str],
    item: dict,
    *,
    value_ok: bool,
    unsupported: int,
    cardinality_over: int,
) -> dict:
    """可解释的计数向量（不引入未登记权重，计划书原文）。

    `missing_or_wrong_values` 的口径分两种，因为两类答案的"值"根本不是同一种东西：

    * `cross_chunk`：值是章节集合，可以逐章数，`missing = |W \\ O|`。
    * 其余四类：注册答案是**单值**，"漏了几个值"没有意义，只有"这个值成没成"。
      所以这里记 **0/1**：值轴站稳记 0，否则记 1。**不用"gold 串在不在文本里"**——
      那会把"gold 出现了但没连到全部锚点"记成 0（看起来没错），
      而 `value_ok` 明明为 False。
    """
    cat = str(category or "")
    if cat == "cross_chunk":
        want = judge_mod.chapter_set((item or {}).get("answer"))
        got = judge_mod.chapter_set(text)
        missing = len(want - got)
        extra = len(got - want)
    else:
        missing = 0 if value_ok else 1
        extra = (max(0, len(values) - MULTIHOP_MAX_CANDIDATES)
                 if cat == "multihop" else 0)
    return {
        "missing_or_wrong_values": missing,
        "extra_values": extra,
        "unsupported_values": unsupported,
        "cardinality_over": cardinality_over,
    }


# ---------------------------------------------------------------- 公开判据


def _result(pass_value: bool | None, reason: str, value_ok: bool, support_ok: bool,
            cardinality_ok: bool, loss: dict) -> dict:
    """装配冻结返回形状。**键集合固定这四个**，多一个都不加（消费端按它取数）。"""
    return {
        "pass": pass_value,
        "reason": reason,
        "axes": {
            "value_ok": value_ok,
            "support_ok": support_ok,
            "cardinality_ok": cardinality_ok,
        },
        "loss": {
            "missing_or_wrong_values": int(loss.get("missing_or_wrong_values", 0)),
            "extra_values": int(loss.get("extra_values", 0)),
            "unsupported_values": int(loss.get("unsupported_values", 0)),
            "cardinality_over": int(loss.get("cardinality_over", 0)),
        },
    }


def judge_v2(
    category: str,
    answer: Any,
    item: dict,
    *,
    context: Any = None,
    question: Any = None,
    chapter_texts: dict | None = None,
    chapter_cap: int | None = None,
    citations: Any = None,
) -> dict:
    """三轴判据。`pass = value_ok AND support_ok AND cardinality_ok`。

    参数
    ----
    category
        八类之一。结构类返回 `pass=None`（剔分母，与 v1 同义）。
    answer
        答案正文 `str`，或答案 dict（`{"answer"/"text", "citations"}`）。
    item
        gold 题条（至少要 `answer` / `acceptable_answers` / `question`）。
    context
        **本次**上下文。它决定 `support_ok`，不给就 fail-closed。
    question
        覆盖 `item["question"]`（调用方常把题面另存一处）。
    chapter_texts
        `{章节代号: 该章正文}`，只有 `cross_chunk` 的"额外章节是否真的提到实体"用。
        不传时与 v1 同样显式降级成 `extra_chapters_unverified`。
    chapter_cap
        `cross_chunk` 的章节上限；`None` 时取 `CROSS_CHUNK_CAP_DEFAULT`。
        W1/W3 的 cap 扫档通过这个参数切换，**不改模块常量**。
    citations
        `answer` 只给了正文串时，引用从这里取（`answer` 是 dict 时忽略本参数）。
        **不传且 answer 是 str ⇒ 视为没有引用 ⇒ `support_ok=False`**，
        `reason` 写 `support:no_citations`。这条不是宽容失败：没有引用就是没有
        支撑证据，判过才是错的（v1 的缺陷正是"不消费 citations"）。

    空答案一律 `pass=False, reason="empty"`（与 v1 同一个 token）——拒答不算答对，
    哪怕 gold 也是空串。
    """
    cat = str(category or "")
    bundle = frozen_input(cat, answer, item or {}, context=context,
                          question=question, citations=citations)
    text = bundle["answer_text"]
    if not text:
        return _result(False, REASON_EMPTY, False, False, False, dict(_ZERO_LOSS))
    if cat in STRUCTURAL_CATEGORIES:
        # 结构类不打分。三根轴填 False 只是占位：**消费端只许看 `pass is None`**，
        # 不许读这里的 axes 去统计"结构类也支持不住"。
        return _result(None, REASON_STRUCTURAL, False, False, False, dict(_ZERO_LOSS))

    it = dict(item or {})
    value_ok, value_reason = _value_axis(
        cat, text, it, question=question, chapter_texts=chapter_texts)
    support_ok, support_reason, _, unsupported = _support_axis(
        cat, bundle["parsed_values"], bundle)
    cap = CROSS_CHUNK_CAP_DEFAULT if chapter_cap is None else int(chapter_cap)
    cardinality_ok, card_reason, over, _ = _cardinality_axis(
        cat, text, bundle["parsed_values"], it, cap=cap)

    loss = _loss(cat, text, bundle["parsed_values"], it,
                 value_ok=value_ok, unsupported=unsupported,
                 cardinality_over=over)

    passed = bool(value_ok and support_ok and cardinality_ok)
    if passed:
        reason = REASON_ALL_PASS
    else:
        parts: list[str] = []
        if not value_ok:
            parts.append(f"value:{value_reason}")
        if not support_ok:
            parts.append(f"support:{support_reason}")
        if not cardinality_ok:
            parts.append(f"cardinality:{card_reason}")
        reason = "|".join(parts)
    return _result(passed, reason, value_ok, support_ok, cardinality_ok, loss)


def explain_v2(
    category: str,
    answer: Any,
    item: dict,
    *,
    context: Any = None,
    question: Any = None,
    chapter_texts: dict | None = None,
    chapter_cap: int | None = None,
    citations: Any = None,
) -> dict:
    """完整审计轨迹：冻结输入 + 逐值 witness + 两条基数明细 + **v1 并列判定**。

    这是给复核用的：`judge_v2` 只回四键，任何"为什么不通过 / 凭什么通过"
    都得能从这里逐项看出，尤其是 `witness_level` 的分布（`chunk_field` 与
    `evidence` 是最弱的两层，它们的占比决定 `support_ok` 有多少说服力）。
    """
    cat = str(category or "")
    verdict = judge_v2(
        cat, answer, item, context=context, question=question,
        chapter_texts=chapter_texts, chapter_cap=chapter_cap, citations=citations)
    bundle = frozen_input(cat, answer, item or {}, context=context,
                          question=question, citations=citations)
    it = dict(item or {})
    cap = CROSS_CHUNK_CAP_DEFAULT if chapter_cap is None else int(chapter_cap)

    _, support_reason, per_value, unsupported = _support_axis(
        cat, bundle["parsed_values"], bundle)
    _, card_reason, over, card_detail = _cardinality_axis(
        cat, bundle["answer_text"], bundle["parsed_values"], it, cap=cap)
    value_ok, value_reason = (False, "empty")
    if bundle["answer_text"] and cat not in STRUCTURAL_CATEGORIES:
        value_ok, value_reason = _value_axis(
            cat, bundle["answer_text"], it, question=question, chapter_texts=chapter_texts)

    v1 = judge_mod.judge_answer(
        cat, bundle["answer_text"], it,
        question=question if question is not None else it.get("question"),
        chapter_texts=chapter_texts)
    by_id = {a["assertion_id"]: a for a in bundle["context_assertions"] if a.get("assertion_id")}
    cited = _CitedIndex(bundle["citations"], by_id)
    return {
        "schema": SCHEMA,
        "category": cat,
        "input": bundle,
        "verdict": verdict,
        "axes_detail": {
            "value": {"ok": value_ok, "reason": value_reason,
                      "gold_values": bundle["gold"]["values"]},
            "support": {
                "ok": verdict["axes"]["support_ok"],
                "reason": support_reason,
                "unsupported": unsupported,
                "per_value": per_value,
                "cited_chunks": sorted(cited.chunks),
                "misplaced_assertions": sorted(cited.misplaced),
            },
            "cardinality": {"ok": verdict["axes"]["cardinality_ok"],
                            "reason": card_reason, "over": over, **card_detail},
        },
        "v1": {"pass": v1.get("pass"), "reason": v1.get("reason")},
    }


# ---------------------------------------------------------------- 内置证伪


def _pf_attribution(v2_verdict: dict, category: str = "") -> tuple[str, list[str]]:
    """`PF` 的互斥归因：一条固定优先级阶梯，返回 `(单值归因, 全部未过轴)`。

    优先级理由：`support` 是 v2 唯一**新引入的硬门槛**，它是最常见的收紧原因，
    也最容易和别的轴同时失败；先归它，才能让"无支撑"这个数不被基数项吞掉。
    其后依次是 multihop 的实体精度、cross_chunk 的章节基数，最后才是 `other`
    （走到 `other` 基本只有两种情形：值逻辑真的变了，或某个轴自己不自洽）。

    `category` 由调用方显式给：`judge_v2` 的返回**只有四个键**（`pass` /
    `reason` / `axes` / `loss`），里面没有 `category`，从 verdict 里取会拿到空串
    而把所有 multihop 的基数失败误归成"章节基数"。
    """
    axes = (v2_verdict or {}).get("axes") or {}
    category = str(category or (v2_verdict or {}).get("category") or "")
    failed = [name for name in ("value_ok", "support_ok", "cardinality_ok")
              if not axes.get(name, True)]
    if not axes.get("support_ok", True):
        return "unsupported", failed
    if not axes.get("cardinality_ok", True):
        if category == "multihop":
            return "entity_precision", failed
        return "chapter_cardinality", failed
    return "other", failed


def classify_flip(
    v1_verdict: dict | None,
    v2_verdict: dict | None,
    *,
    question_id: str = "",
    category: str = "",
    registered_fp: dict | None = None,
) -> dict:
    """把一对 (v1, v2) 判定归进 `PP/PF/FP/FF`，并给 `PF` 做互斥归因。

    * 任一侧 `pass is None`（结构类）⇒ `cell="NA"`，**不进四个格子**：
      它们本来就剔分母，混进去会让 `PP+PF+FP+FF` 与可判分母对不上
      （计划书反证条件 2）。
    * `FP`（v1 拒、v2 过）默认必须为 0；出现时 `fp_registered` 为 False
      ⇒ 必须走逐题批准流程（`REGISTERED_V1_FALSE_NEGATIVES`）。
    """
    registry = REGISTERED_V1_FALSE_NEGATIVES if registered_fp is None else dict(registered_fp)
    p1 = (v1_verdict or {}).get("pass")
    p2 = (v2_verdict or {}).get("pass")
    base = {
        "question_id": question_id,
        "category": category or str((v2_verdict or {}).get("category") or ""),
        "v1_pass": p1,
        "v2_pass": p2,
    }
    if p1 is None or p2 is None:
        return {**base, "cell": "NA", "attribution": "structural",
                "axes_failed": [], "fp_registered": False,
                "reason": "structural:not_scored"}
    if p1 and p2:
        return {**base, "cell": "PP", "attribution": "", "axes_failed": [],
                "fp_registered": False, "reason": "both_pass"}
    if p1 and not p2:
        attribution, failed = _pf_attribution(v2_verdict, base["category"])
        return {**base, "cell": "PF", "attribution": attribution,
                "axes_failed": failed, "fp_registered": False,
                "reason": f"v1_pass_v2_reject:{attribution}"}
    if (not p1) and p2:
        registered = question_id in registry
        return {**base, "cell": "FP", "attribution": "v1_false_negative",
                "axes_failed": [], "fp_registered": registered,
                "reason": ("registered:" + registry[question_id]) if registered
                          else "unregistered_fp_needs_review"}
    return {**base, "cell": "FF", "attribution": "both_reject", "axes_failed": [],
            "fp_registered": False, "reason": "both_reject"}


def tally_flips(rows: list[dict], *, registered_fp: dict | None = None) -> dict:
    """逐题翻转表汇总。**顺带机检两类反证条件**（计划书 W4 反证条件 1、2）。

    `rows` 每项形如
    `{"question_id", "category", "v1": {...}, "v2": {...}}`。

    返回值里 `ok` 为真的条件是：
      * `PP+PF+FP+FF` 恰好等于可判题数（结构类记在 `n_structural`，不许并进来）；
      * 没有未登记的 `FP`。

    不满足任一条 ⇒ `problems` 会点名是哪一条，**不许用"总分没降"把它盖过去**。
    """
    registry = REGISTERED_V1_FALSE_NEGATIVES if registered_fp is None else dict(registered_fp)
    counts = {cell: 0 for cell in CELLS}
    pf_attr = {name: 0 for name in PF_ATTRIBUTIONS}
    per_item: list[dict] = []
    unregistered_fp: list[str] = []
    n_structural = 0
    for row in rows or ():
        qid = str(row.get("question_id") or "")
        v1 = row.get("v1") or {}
        v2 = row.get("v2") or {}
        flip = classify_flip(v1, v2, question_id=qid,
                             category=str(row.get("category") or ""),
                             registered_fp=registry)
        per_item.append(flip)
        if flip["cell"] == "NA":
            n_structural += 1
            continue
        counts[flip["cell"]] += 1
        if flip["cell"] == "PF":
            pf_attr[flip["attribution"]] += 1
        if flip["cell"] == "FP" and not flip["fp_registered"]:
            unregistered_fp.append(qid or "<no-question-id>")

    n_scorable = len(per_item) - n_structural
    total_cells = sum(counts.values())
    problems: list[str] = []
    if total_cells != n_scorable:
        problems.append(
            f"PP+PF+FP+FF={total_cells} 与可判题数 {n_scorable} 对不上（实现有 bug）")
    if unregistered_fp:
        problems.append(f"未登记的 FP：{unregistered_fp}（必须逐题审查或修正规则）")
    if sum(pf_attr.values()) != counts["PF"]:
        problems.append(
            f"PF 归因计数 {sum(pf_attr.values())} != PF 总数 {counts['PF']}（归因不互斥）")
    return {
        "schema": SCHEMA,
        "n_total": len(per_item),
        "n_scorable": n_scorable,
        "n_structural": n_structural,
        "counts": counts,
        "pf_attribution": pf_attr,
        "fp_unregistered": unregistered_fp,
        "per_item": per_item,
        "problems": problems,
        "ok": not problems,
    }
