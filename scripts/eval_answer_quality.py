"""按题型判据评测「答对没有」——`arkrag.judge` 的唯一消费端。

## 这个脚本为什么必须存在
`src/arkrag/judge.py` 定稿后，如果没人调它，就只是又一份"写在库里没人用"的口径。
本项目已经因为"判据没接线等于不存在"栽过三次（`TITLE_ANSWER_PATH`、`CHAPTER_ANSWER_PATH`
都默认 False 而台账写着"已入库"）。所以这个脚本是判据的唯一生产消费端：
**除它以外不许再有任何地方自己写一份"答案算不算对"。**

## 三个分母必须并列
- `scorable`：剔掉结构类（`time_conflict` / `negative_abstain` / `adversarial`）后的分母。
- `all`：把结构类也算进分母（它们在判据里是 `None`，这里按"没答对"计入）。
两个数一起报，因为只看 scorable 会高估 8–15 个百分点。

## split 分开报，不合并
`dev` 上的任何阈值/口径调整都不能用 `holdout` 的数字去挑；`holdout` 里还有与
`gold-v1` 重复的题，所以额外给 `clean`（剔除重复题后的）一栏。

## W4b：判据 v2 双口径（`--judge v1|v2|both`，默认 `v1`）

计划书 §W4 要求同一批答案**两套判据并行记分、两套数并列报**，并且把统计口径
一起升级。四件事各自有一条"静默失效"的形状，所以分别钉：

1. **`--judge v1` 必须逐位等于改造前**（J3 硬门）。因此 `judge_mode == "v1"`
   时 `run()` 返回的 dict **一个键都不多**：不写 `judge_mode`、不写 `flips`、
   不写 `ci`、不写 `floors`。行内也不多 `judge_v2` / `arms["v2"]`。
   任何"顺手把新块也塞进默认档"的改动都会让历史产物变字节，而分数一个字不变——
   这正是 J3 要拦的东西（`tests/test_w4_eval_dual.py::TestV1IsByteIdentical`）。
   地板三行（J5）走**报告模板**（`print_report`）无条件打印，v1/v2 都有；
   JSON 里只在 `v2`/`both` 档附带。
2. **两套数并列报**（J1 前置）。`v2`/`both` 档位下 `tally()` 同时给 `prod`（v1）
   与 `v2` 两个臂，打印时两列并排；只报一套等于选择性披露（红线 11 同型）。
3. **翻转四类计数**（J1）：`PP/PF/FP/FF`，且 `PF` 按**互斥原因**分组。
   `PP+PF+FP+FF` 必须等于"两套判据都能判"的分母；不能判的（结构类、v2 返回
   非 bool）单列，**不许当成 False 计入**（反证条件 2）。
4. **统计口径升级**（J4）：iid 配对 bootstrap 保留，另外按"冻结模板簇"整簇重抽
   （`arkrag.cluster_ci`）。同时报**簇数、有效簇数（Kish）、有效样本量**。
   方差膨胀**只在真有配对样本（v1/v2 同一批题）时计算**，算不出来就标"不可估计"，
   **绝不**用计划书里 `m̄≈2.5, ρ≈0.6` 反推的 ≈1.9 顶替实测。

`judge_v2` 由另一个工作项提供（`src/arkrag/judge_v2.py`，**本项不改它**）。
接口约定是：

    judge_v2.judge_answer_v2(category, answer_text, item,
                             question=q, chapter_texts=chap, context=ctx)

**实际落地的模块把入口叫 `judge_v2.judge_v2`（模块名与函数名同名）**，所以这里
按固定优先级认这两个名字（`resolve_v2_entry`），并把选中的那个写进报告
（`judge_v2_entry`）——静默任选一个会让"报告说自己用了 v2"这件事无法核对。

返回 `{"pass": bool|None, "reason": str, "axes": {...}, "loss": {...}}`。
`import` 不到时 `--judge v2/both` **在启动阶段就明确报错退出**（退出码 2），
不静默退回 v1——退回会产出一份"看着是双口径、其实两列同数"的假报告。

用法：
    py -3 scripts/eval_answer_quality.py --split dev
    py -3 scripts/eval_answer_quality.py --split holdout --json data/eval_quality_holdout.json
    py -3 scripts/eval_answer_quality.py --split clean --token-budget 6000
    py -3 scripts/eval_answer_quality.py --split dev --prod --judge both \\
        --json data/eval_quality_dev_dual.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import cluster_ci as cluster_mod  # noqa: E402
from arkrag import config  # noqa: E402
from arkrag import judge as judge_mod  # noqa: E402
from arkrag import pipeline as pipeline_mod  # noqa: E402

#: `--judge` 的三个取值。默认 `v1`：`judge_v2.py` 删掉就完全回到改造前（计划书 §回滚）。
JUDGE_MODES: tuple[str, ...] = ("v1", "v2", "both")

#: `judge_v2` 的 import 失败原因（供启动时报错与报告留痕，不静默）。
_JUDGE_V2_ERROR: str = ""

#: J5：地板三行的**登记常量**。
#:
#: 三个数都是历史实测，**不在本脚本里重新量**（量它们要么要读 gold 产物、要么要
#: 跑零检索通路，两者都不是评测脚本该干的事）。分母一律是 dev 的**可判分母 285**，
#: 与 `--split dev` 的可判分母同源；换 split 时三行照报，但校正指标会因分母不同
#: 自动标"不可估计"（见 `floor_report()`）。
#:
#: **不许相加**：三者是同一个分母上的三条**替代通路**（回显题干 / 形状 oracle /
#: 用 gold 顶替答案），它们不是互不相交的题集，相加会算出毫无意义的数。
FLOORS: tuple[dict, ...] = (
    {
        "key": "echo_question",
        "label": "回显题干",
        "count": 68,
        "n": 285,
        "rate": 0.2386,
        "method": "零检索：把题干里自带的 gold 原样回显即判过（逐题判定=judge_answer）",
        "source": "docs/审查-评测口径对抗性报告-20261107.md §B6",
    },
    {
        "key": "shape_oracle",
        "label": "形状 oracle",
        "count": 285,
        "n": 285,
        "rate": 1.0,
        "method": "零检索：把「问句锚点 + 正确 gold」拼成一句，不含任何检索内容",
        "source": "docs/审查-评测口径对抗性报告-20261107.md §A20",
    },
    {
        "key": "gold_verbatim",
        "label": "gold 原文",
        "count": 171,
        "n": 285,
        "rate": 0.6,
        "method": "零检索：答案直接等于 gold 原文（判据下反而低于生产 80.00%）",
        "source": "docs/审查-评测口径对抗性报告-20261107.md §A20",
    },
)

#: J5 校正指标的定义域：`0 ≤ floor ≤ pass ≤ N` 之外一律"不可估计"。
#: 计划书写的是 `(pass − floor) / (N − floor)`，只在 `0≤floor≤pass≤N` 时报告。
FLOOR_DOMAIN_NOTE = (
    "定义域 0 ≤ floor ≤ pass ≤ N；越界一律标「不可估计」，"
    "不夹到 0、不出负数、不把三行相加"
)


def chapter_texts(chunks: dict) -> dict[str, str]:
    """`{章节代号: 该章全部块正文拼接}`，供 `judge_answer` 校验"额外章节是否真的提到实体"。

    只从本世代自带的块建（`load_generation` 已用 `resolve_chunks` 证明这些块属于本代），
    不许读全局 `data/arkrag/chunks.jsonl` —— 两者块数不同（13,798 vs 13,786），
    混用会让"这个章节里到底有没有这个名字"判错。
    """
    out: dict[str, list[str]] = {}
    for rec in chunks.values():
        ch = str(rec.get("chapter") or "")
        out.setdefault(ch, []).append(str(rec.get("text") or ""))
    return {k: "\n".join(v) for k, v in out.items()}


def pick_items(items: list[dict], split: str, v1_ids: set[str]) -> list[dict]:
    """按 split 选题。`clean` = holdout 里且不在 gold-v1 里。"""
    if split == "all":
        pool = [it for it in items if it.get("gold_chunk_ids")]
    elif split == "clean":
        pool = [it for it in items
                if it.get("split") == "holdout" and it.get("gold_chunk_ids")
                and str(it.get("question_id")) not in v1_ids]
    else:
        pool = [it for it in items if it.get("split") == split and it.get("gold_chunk_ids")]
    return pool


def load_v1_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    doc = json.loads(path.read_text("utf-8"))
    items = doc.get("items") if isinstance(doc, dict) else doc
    return {str(it.get("question_id")) for it in items or []}


def tally(rows: list[dict], arms: list[str]) -> dict:
    """按类别汇总，同时给 scorable 与 all 两个分母。"""
    per_cat: dict[str, dict] = {}
    for cat in judge_mod.CATEGORIES:
        sub = [r for r in rows if r["category"] == cat]
        if not sub:
            continue
        entry = {"n": len(sub), "scorable": cat in judge_mod.SCORABLE_CATEGORIES,
                 "arms": {}}
        for arm in arms:
            if cat in judge_mod.STRUCTURAL_CATEGORIES:
                entry["arms"][arm] = {"pass": None,
                                      "opened": sum(1 for r in sub if r["arms"][arm]["answered"])}
            else:
                entry["arms"][arm] = {"pass": sum(1 for r in sub if r["arms"][arm]["pass"]),
                                      "opened": sum(1 for r in sub if r["arms"][arm]["answered"])}
        per_cat[cat] = entry

    totals = {}
    for arm in arms:
        scorable = [r for r in rows if r["category"] in judge_mod.SCORABLE_CATEGORIES]
        ok_s = sum(1 for r in scorable if r["arms"][arm]["pass"])
        opened = sum(1 for r in rows if r["arms"][arm]["answered"])
        totals[arm] = {
            "scorable_n": len(scorable),
            "scorable_pass": ok_s,
            "scorable_rate": round(ok_s / len(scorable), 6) if scorable else 0.0,
            "all_n": len(rows),
            "all_rate": round(ok_s / len(rows), 6) if rows else 0.0,
            "opened": opened,
            "opened_rate": round(opened / len(rows), 6) if rows else 0.0,
        }
    return {"per_category": per_cat, "totals": totals}


# ------------------------------------------------------------------ W4b 判据 v2
def load_judge_v2():
    """按需 import `arkrag.judge_v2`；拿不到就返回 `None` 并把原因留在模块变量里。

    刻意**不**在模块顶部无条件 import：`judge_v2.py` 是另一个工作项的产物，
    它不在时 v1 通路必须照常可用（计划书 §回滚：删掉 `judge_v2.py` 即完全恢复）。
    但 `--judge v2|both` 是**显式要求双口径**，那时拿不到模块必须报错退出，
    不许静默退回 v1 —— 静默退回会产出一份"两列同数"的假双口径报告。
    """
    global _JUDGE_V2_ERROR
    try:
        from arkrag import judge_v2  # noqa: F401
    except Exception as exc:  # ImportError 及其依赖缺失都算"没就绪"
        _JUDGE_V2_ERROR = f"{type(exc).__name__}: {exc}"
        return None
    _JUDGE_V2_ERROR = ""
    return judge_v2


def resolve_v2_entry(judge_v2):
    """在 `judge_v2` 模块上找到判分入口，返回 `(函数, 实际用的名字)`。

    接口约定写的是 `judge_answer_v2`，而实际落地的模块把入口叫 `judge_v2.judge_v2`
    （模块名与函数名同名）。这里**按固定优先级**认两个名字，并把选中的那个写进
    报告（`judge_v2_entry`）——静默任选一个会让"报告说自己用了 v2"这件事无法核对。

    两个都没有就返回 `(None, "")`，由调用方决定是"不可判"还是启动报错。
    """
    for name in ("judge_answer_v2", "judge_v2"):
        fn = getattr(judge_v2, name, None)
        if callable(fn):
            return fn, name
    return None, ""


#: 落地的 `judge_v2.judge_v2` 把第二参数写成 `answer`，并且**从答案 dict 里取
#: citations**（`frozen_input`：`answer` 是 dict 时引用从它里面取，是 str 时
#: 只能来自显式的 `citations` 参数）。只传正文等于宣布"这条答案没有引用支撑"，
#: `support_ok` 会恒为 False —— 那样两套判据的差就全是"引用没接上"这个假区别。
#: 所以对**这个入口**传答案 dict；对约定里的 `judge_answer_v2` 仍按约定传正文。
SHIPPED_V2_ENTRY = "judge_v2"


def judge_mode_arms(judge_mode: str) -> list[str]:
    """每个档位要出现在报告里的臂。`v1` 只有历史那一臂（J3：不许变动）。"""
    return ["prod"] if judge_mode == "v1" else ["prod", "v2"]


def v2_call_shape(entry_name: str) -> dict:
    """这次调用会怎么传参——写进报告，免得"v2 用了什么输入"只能靠读源码。"""
    if entry_name == SHIPPED_V2_ENTRY:
        return {"answer_form": "dict",
                "citations_from": "answer['citations']",
                "chapter_cap": "passed_from_options"}
    return {"answer_form": "str", "citations_from": "none（约定签名没有引用参数）",
            "chapter_cap": "not_supported"}


def _v2_call(judge_v2, category: str, text: str, item: dict, ans: dict, *,
             question: str, chapter_texts: dict, context: dict | None,
             chapter_cap: int | None = None) -> dict:
    """调 v2 判据，并把**任何**异常收敛成一个显式的"不可判"记录。

    收敛而不是往外抛，是因为一道题让 v2 崩掉时，整份报告的其余 284 题读数仍然
    有价值；但这条记录必须带着 `error` 字段进 `v2_undecidable` 单列，
    **不许**当成 False 计入分母（反证条件 2：缺失输入被静默当成通过/失败都算 bug）。
    """
    entry, entry_name = resolve_v2_entry(judge_v2)
    if entry is None:
        return {"pass": None, "reason": "v2_error:no_entry_point", "axes": {},
                "loss": {}, "error": "judge_v2 模块里没有可调用的判分入口"}
    payload = ans if entry_name == SHIPPED_V2_ENTRY else text
    kwargs: dict = {"question": question, "chapter_texts": chapter_texts,
                    "context": context}
    if entry_name == SHIPPED_V2_ENTRY and chapter_cap:
        kwargs["chapter_cap"] = int(chapter_cap)
    try:
        res = entry(category, payload, item, **kwargs)
    except Exception as exc:
        return {"pass": None, "reason": f"v2_error:{type(exc).__name__}",
                "axes": {}, "loss": {}, "error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(res, dict):
        return {"pass": None, "reason": "v2_error:not_a_dict", "axes": {}, "loss": {},
                "error": f"{entry_name} 返回了 {type(res).__name__}"}
    out = {"pass": res.get("pass"), "reason": str(res.get("reason") or ""),
           "axes": dict(res.get("axes") or {}), "loss": dict(res.get("loss") or {})}
    if out["pass"] is not None and not isinstance(out["pass"], bool):
        # `pass` 只允许 bool|None。别的类型（0/1、"true"）当作实现违约记下来。
        out["error"] = f"pass 类型非法：{type(out['pass']).__name__}"
        out["pass"] = None
    return out


#: `PF` 的互斥归因顺序。**必须**是固定顺序：同一道题只能落进一个桶，
#: 否则"PF 按互斥原因分组"就是假的（各组相加会超过 PF 总数）。
PF_AXIS_ORDER: tuple[str, ...] = ("support_ok", "cardinality_ok", "value_ok")

#: 轴名缺位时退回 `loss` 的键（同样固定顺序）。这两张表是**归因的唯一来源**：
#: 不从 `reason` 字符串里猜（判据的 reason 文案会改，改了分组就漂移）。
PF_LOSS_ORDER: tuple[str, ...] = (
    "unsupported_values", "cardinality_over", "extra_values", "missing_or_wrong_values",
)


def pf_bucket(rec: dict) -> str:
    """一条 v1=True / v2=False 的题，归到**唯一**一个原因桶。

    取第一条为 False 的轴（`PF_AXIS_ORDER`）；轴表里没有可读信息时退到 `loss`
    里第一个非零计数（`PF_LOSS_ORDER`）；两者都没有就是 `unattributed` ——
    这个桶非空即表示"翻转无法归因"，按计划书反证条件 1，那是设计失败。
    """
    axes = rec.get("axes") or {}
    for name in PF_AXIS_ORDER:
        if axes.get(name) is False:
            return name
    loss = rec.get("loss") or {}
    for name in PF_LOSS_ORDER:
        try:
            if int(loss.get(name) or 0) > 0:
                return name
        except (TypeError, ValueError):
            continue
    return "unattributed"


def flip_counts(rows: list[dict]) -> dict:
    """`PP / PF / FP / FF` 四类翻转 + 不可判单列。

    口径（写死，测试钉住）：

    * 只在**两套判据都给 bool** 的题上计四类：`PP=(T,T)`、`PF=(T,F)`、
      `FP=(F,T)`、`FF=(F,F)`；因此 `PP+PF+FP+FF == 两套都能判的题数`。
    * v2 给 `None`（含 v2 自己崩掉）的题进 `v2_undecidable`，**不**当 False。
    * v1 给 `None`（结构类）的题进 `structural`，**不**参与翻转。

    `PF` 额外给按 `pf_bucket()` 分组的计数，和 `PF` 逐题清单。
    """
    counts = {"PP": 0, "PF": 0, "FP": 0, "FF": 0}
    pf_groups: dict[str, int] = {}
    pf_items: list[dict] = []
    undecidable: list[dict] = []
    structural = 0
    for row in rows:
        v1 = (row.get("judge") or {}).get("pass")
        v2_rec = row.get("judge_v2")
        if v2_rec is None:
            continue
        v2 = v2_rec.get("pass")
        if v1 is None:
            structural += 1
            continue
        if v2 is None:
            undecidable.append({
                "question_id": row.get("question_id"),
                "category": row.get("category"),
                "v1_pass": bool(v1),
                "v2_reason": v2_rec.get("reason"),
                "error": v2_rec.get("error"),
            })
            continue
        key = ("P" if v1 else "F") + ("P" if v2 else "F")
        counts[key] += 1
        if key == "PF":
            bucket = pf_bucket(v2_rec)
            pf_groups[bucket] = pf_groups.get(bucket, 0) + 1
            pf_items.append({
                "question_id": row.get("question_id"),
                "category": row.get("category"),
                "bucket": bucket,
                "v1_reason": (row.get("judge") or {}).get("reason"),
                "v2_reason": v2_rec.get("reason"),
                "v2_loss": v2_rec.get("loss") or {},
            })
    decidable = sum(counts.values())
    return {
        "counts": counts,
        "decidable_n": decidable,
        "sum_matches_decidable": decidable == sum(counts.values()),
        "pf_by_reason": dict(sorted(pf_groups.items())),
        "pf_items": pf_items,
        "v2_undecidable": undecidable,
        "v2_undecidable_n": len(undecidable),
        "structural": structural,
        "parse_failures": _parse_failure_report(rows),
        "fp_items": [
            {"question_id": r.get("question_id"), "category": r.get("category"),
             "v2_reason": (r.get("judge_v2") or {}).get("reason")}
            for r in rows
            if (r.get("judge") or {}).get("pass") is False
            and (r.get("judge_v2") or {}).get("pass") is True
        ],
    }


def _parse_failure_report(rows: list[dict]) -> dict:
    """**解析失败单独计数**——计划书 W4 反证条件 4 点名的形态。

    原文：`v2 把大量 FF 通过解析失败统一拒绝，却隐藏解析失败数 ⇒ 属于判据退化，
    必须单列并修复，不能用低分证明严格。`

    为什么必须单列：W4a 的实现报告里明确警告过——
    `parsed_values == []` 时，`support_ok` 与 `cardinality_ok` 会**空真通过**，
    `loss` 的四个计数也全是 0。于是"解析器挂了"和"答案完美"在 `loss` 上长得一模一样。
    不单独数一下，就无法区分"v2 严格"和"v2 根本没解析出来"。

    判定口径（可机检，不依赖 v2 内部）：
    逐题看 `judge_v2.loss` 四项是否**全为 0**且该题**非空答案**。
    非空答案 + 四项全零 = 解析器没抓到任何值（因为任何真解析都会在
    `missing_or_wrong` 或 `extra` 上留下非零），记一次 `suspect_parse_failure`。

    **这不是断言"它一定解析失败了"**，是"这一格需要人看一眼"。所以字段名叫
    `suspect_*` 而不是 `parse_failed_*`——一个答得完美的题也会四项全零，
    两者只能靠抽审分开，不能靠这个计数分开。
    """
    sub = _scorable(rows)
    v2_rows = [r for r in sub if isinstance(r.get("judge_v2"), dict)]
    suspects: list[dict] = []
    empty_answers = 0
    for r in v2_rows:
        text = str(r.get("answer") or "")
        if not text.strip():
            empty_answers += 1
            continue
        loss = (r.get("judge_v2") or {}).get("loss") or {}
        vals = [int(loss.get(k) or 0) for k in
                ("missing_or_wrong_values", "extra_values",
                 "unsupported_values", "cardinality_over")]
        if vals and all(v == 0 for v in vals):
            suspects.append({
                "question_id": r.get("question_id"),
                "category": r.get("category"),
                "v1_pass": (r.get("judge") or {}).get("pass"),
                "v2_pass": (r.get("judge_v2") or {}).get("pass"),
                "v2_reason": (r.get("judge_v2") or {}).get("reason"),
                "answer_head": text[:60],
            })
    return {
        "n_v2_rows": len(v2_rows),
        "n_empty_answer": empty_answers,
        "n_zero_loss": len(suspects),
        "zero_loss_rate": round(len(suspects) / len(v2_rows), 4) if v2_rows else None,
        "suspect_zero_loss_items": suspects,
        "limitation": (
            "四项 loss 全零可能是解析器没抓到值，也可能是答案真的完美——"
            "本计数只负责把这一格暴露出来，不判定它是哪一种。"
            "必须抽审 suspect_zero_loss_items 才能区分，"
            "这也是计划书反证条件 4 要求「不能用低分证明严格」的直接落地。"
        ),
    }


# ------------------------------------------------------------------- J5 地板
def corrected_rate(pass_count: int, n: int, floor_count: int, floor_n: int) -> dict:
    """地板校正指标 `(pass − floor) / (N − floor)`，**带定义域保护**。

    只在 `0 ≤ floor ≤ pass ≤ N`（且 `floor_n == n`，即两个数同一个分母）时给数；
    否则返回 `estimable=False` + 原因。**不夹到 0、不出负数**：
    `floor > pass` 时"校正后通过率"在数学上是个负数，它没有任何解释力，
    把它打印出来只会让人以为"改进是负的"，而实际含义是"整条通路不如一条地板"。

    为什么还要 `floor_n == n`：三个地板数都是在 dev 可判分母 285 上量的。
    换 split（holdout 195 / clean 120）后，分子可以照报（那也是历史实测），
    但 `(pass − 68)/(195 − 68)` 这种把 285 分母的地板塞进 195 分母的算式
    **没有定义**——两个分母不同的数不能相减。
    """
    if int(floor_n) != int(n):
        return {"estimable": False, "value": None,
                "reason": f"分母不同：地板 {floor_n} vs 本次可判 {n}，不能相减"}
    if not (0 <= int(floor_count) <= int(pass_count) <= int(n)):
        return {"estimable": False, "value": None,
                "reason": f"越界：floor={floor_count} pass={pass_count} N={n}"
                          f"（要求 0≤floor≤pass≤N）"}
    denom = int(n) - int(floor_count)
    if denom <= 0:
        return {"estimable": False, "value": None,
                "reason": f"N−floor = {denom} ≤ 0，分母为零或负"}
    return {"estimable": True,
            "value": (int(pass_count) - int(floor_count)) / denom,
            "reason": f"({pass_count}−{floor_count})/({n}−{floor_count})"}


def floor_report(pass_count: int, n: int, *, arm: str) -> dict:
    """J5 三行：每行带**自己的分母与估计方法**，并附各自的校正指标。

    三行**不默认相加**：它们是同一分母上的三条替代通路（不是三个互斥题集）。
    `sum_of_floors` 这个键刻意**不存在**——一旦提供，早晚有人会去加。
    """
    lines = []
    for f in FLOORS:
        lines.append({
            "key": f["key"],
            "label": f["label"],
            "floor_count": f["count"],
            "floor_n": f["n"],
            "floor_rate": f["rate"],
            "method": f["method"],
            "source": f["source"],
            "arm": arm,
            "pass_count": pass_count,
            "n": n,
            "corrected": corrected_rate(pass_count, n, f["count"], f["n"]),
        })
    return {"lines": lines, "domain_note": FLOOR_DOMAIN_NOTE,
            "additive": False,
            "note": "三行是并列的替代通路，不是互斥题集；相加无意义，故不提供合计"}


# ------------------------------------------------------------------ J4 统计
def _scorable(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["category"] in judge_mod.SCORABLE_CATEGORIES]


def _arm_vector(rows: list[dict], arm: str) -> list[int]:
    """一臂的逐题 0/1 向量（只取可判类；`pass` 不是 True 一律算 0）。"""
    return [1 if r["arms"][arm]["pass"] else 0 for r in rows]


def stats_block(rows: list[dict], arms: list[str]) -> dict:
    """iid CI 与 cluster CI 并列，附簇数与两种"有效量"。

    * `iid`：保留既有口径（按题有放回重抽）。
    * `cluster`：按冻结模板簇整簇重抽。
    * `variance_inflation`：**只在真有配对样本时**才算。这里的配对样本是
      v1/v2 同一批题（`--judge v2|both` 才有）；v1 单臂没有配对，如实标不可估计。
    """
    sub = _scorable(rows)
    questions = [str(r.get("question") or "") for r in sub]
    keys = cluster_mod.cluster_keys(questions) if questions else []
    out: dict = {
        "denominator": len(sub),
        "cluster": cluster_mod.cluster_report(questions) if questions else {
            "n_items": 0, "unique_questions": 0, "n_clusters": 0,
            "cluster_sizes": [], "effective_clusters": 0.0,
            "n_gram": cluster_mod.DEFAULT_NGRAM,
            "jaccard_threshold": cluster_mod.DEFAULT_JACCARD,
        },
        "arms": {},
        "variance_inflation": {"estimable": False, "value": None,
                               "reason": "没有配对样本（单臂读数无法做方差分解）"},
        "effective_sample_size": {"estimable": False, "value": None,
                                  "reason": "方差膨胀不可估计"},
        "reference": {
            "note": "计划书里 m̄≈2.5、ρ≈0.6 推出方差膨胀≈1.9 只是设计量级估算，"
                    "不作为实测值出现；本块任何数都来自本次配对样本",
        },
    }
    for arm in arms:
        vec = _arm_vector(sub, arm)
        iid = cluster_mod.iid_bootstrap_mean(vec)
        clu = cluster_mod.cluster_bootstrap_mean(vec, keys) if keys else {
            "mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "var": 0.0,
            "reps": 0, "degenerate": True, "n_clusters": 0, "note": "no_observations"}
        out["arms"][arm] = {"iid": iid, "cluster": clu}
    if "prod" in arms and "v2" in arms:
        # 逐题配对（只在两臂都能判的题上）：这是本脚本唯一存在的配对样本。
        paired_rows = [r for r in sub
                       if (r.get("judge") or {}).get("pass") is not None
                       and (r.get("judge_v2") or {}).get("pass") is not None]
        pkeys = cluster_mod.cluster_keys([str(r.get("question") or "")
                                         for r in paired_rows]) if paired_rows else []
        before = [1 if (r["judge"]["pass"] is True) else 0 for r in paired_rows]
        after = [1 if (r["judge_v2"]["pass"] is True) else 0 for r in paired_rows]
        if paired_rows:
            iid_d = cluster_mod.paired_bootstrap_delta(
                before, after, pkeys, clustered=False)
            clu_d = cluster_mod.paired_bootstrap_delta(
                before, after, pkeys, clustered=True)
            infl = cluster_mod.variance_inflation(iid_d, clu_d)
            out["paired_v1_v2"] = {
                "n_paired": len(paired_rows),
                "n_clusters": len(set(pkeys)),
                "iid": iid_d,
                "cluster": clu_d,
            }
            out["variance_inflation"] = infl
            out["effective_sample_size"] = cluster_mod.effective_sample_size(
                len(paired_rows), infl)
        else:
            out["paired_v1_v2"] = {"n_paired": 0,
                                   "note": "两臂没有同时可判的题，配对样本为空"}
            out["variance_inflation"] = {
                "estimable": False, "value": None,
                "reason": "配对样本为空（无同时可判的题）"}
    return out


def context_note(res: dict) -> dict:
    """`ask_once` 返回的 context 是否真的能交给 v2（`units[].assertions`）。

    `ask_once` 在所有分支都会返回 `context` 键（拒答分支给
    `{"units": [], "context_digest": "", "stats": {}}`），所以正常情况下
    `context_available=True`。但**不能默认它在**：旧产物 / 替身返回的 `res`
    可能没有这个键，那时 v2 的 `support_ok` 只能拿不到断言——
    这个限制必须写进报告，不许让"支撑不足"看起来像"答案错了"。
    """
    ctx = res.get("context")
    if not isinstance(ctx, dict):
        return {"context_available": False,
                "context_note": "ask_once 没有返回 context dict；"
                                "v2 的 support 轴拿不到本次断言，"
                                "该维度不可信（不是答案错了）"}
    units = ctx.get("units")
    n_units = len(units) if isinstance(units, list) else 0
    n_assert = sum(len(u.get("assertions") or ()) for u in (units or ())
                   if isinstance(u, dict))
    return {"context_available": True,
            "context_units": n_units, "context_assertions": n_assert,
            "context_note": ""}


def _options_for_report(options: dict | None) -> dict:
    """报告里记 options 的可打印形态。

    `bridge_calibration` 是一整张映射表（几十个桶），原样塞进报告会让每份 JSON
    多出十几 KB，且它本身可以从 `--calibration` 指的路径重新读到。
    这里只记一个摘要（键名 + 桶数 + 阈值），既保留可追溯性又不灌水。
    """
    if not options:
        return {}
    out = dict(options)
    calib = out.get("bridge_calibration")
    if isinstance(calib, dict):
        out["bridge_calibration"] = {
            "feature_keys": list(calib.get("feature_keys") or ()),
            "n_buckets": len(calib.get("buckets") or ()),
            "digest": hashlib.sha256(
                json.dumps(calib, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()[:16],
        }
    return out


def run(generation_id: str, gold_path: Path, *, split: str, limit: int,
        token_budget: int | None, prod: bool = False,
        extra_options: dict | None = None,
        calibration_path: Path | None = None,
        judge_mode: str = "v1") -> dict:
    """跑一份 split。

    `prod=True` 时走 `pipeline.PROD_PROFILE`（生产档：引文定位 + 桥接 + 图证据预留
    + 24000 预算）。**默认 `False`，保证历史读数逐格可复现**——台账里 6000/12000 档
    的每一次对比都是在这个默认分支下测的，改成默认开生产档会让全部历史数字失去可比性。

    `judge_mode`（W4b）默认 `"v1"`，此时返回值与改造前**逐键逐值相同**。
    `"v2"` / `"both"` 才会（a）在每题上多跑一遍 `judge_v2`、（b）在报告里加
    `judge_mode` / `flips` / `stats` / `context_support` / `floors` 几块。
    `"v2"` / `"both"` 而 `arkrag.judge_v2` 不可用时**抛 `RuntimeError`**，
    不静默退回 v1（`main()` 会在启动阶段先拦一次，这里是防被直接调用的第二道）。
    """
    if judge_mode not in JUDGE_MODES:
        raise ValueError(f"judge_mode 只能是 {JUDGE_MODES}，收到 {judge_mode!r}")
    judge_v2 = None
    if judge_mode != "v1":
        judge_v2 = load_judge_v2()
        if judge_v2 is None:
            raise RuntimeError(
                f"judge_mode={judge_mode!r} 需要 `arkrag.judge_v2`，但它不可用："
                f"{_JUDGE_V2_ERROR or '模块缺失'}")
    loaded = pipeline_mod.load_generation(generation_id)
    pipe = pipeline_mod.build_pipeline(loaded, top_k=20, graph_depth=1)
    pipe["generation_id"] = generation_id
    pipe["entities"] = loaded["graph"].get("entities") or []
    options: dict | None = None
    if prod:
        options = pipeline_mod.apply_prod_profile(pipe)
    # ---- 实验开关（默认空 = 与历史读数逐位等价）----------------------------
    # 只允许**追加/覆盖** `ask_once` 的 options 键，不允许改任何默认值。
    # `extra_options` 为 `None` 或空 dict 时，`options` 与改造前完全相同。
    # 这条是为 W1 的 cap 扫档服务的：不同 cap 必须能在同一条命令上切换，
    # 而不是各自改一份脚本（改脚本就会让"实验"和"实现"变成两份事实源）。
    if extra_options:
        options = dict(options or {})
        options.update(extra_options)
    # ---- W2 校准表（默认不读，给了路径才注入）------------------------------
    # **不在 pipeline 里隐式读文件**：那会让"同一份 options"在不同机器上行为不同
    # （一条路径存在与否就能改结果），也让复现性无法从 options 本身证明。
    # 这里显式读、显式注入，路径进报告。
    if calibration_path is not None:
        from arkrag import bridge_calib as _bridge_calib
        options = dict(options or {})
        options["bridge_calibration"] = _bridge_calib.load_mapping(calibration_path)
    budget = int(token_budget) if token_budget is not None else \
        int(pipe["retriever"].budget["max_context_tokens"])
    pipe["retriever"].budget["max_context_tokens"] = budget

    doc = json.loads(gold_path.read_text("utf-8"))
    items = doc.get("items") if isinstance(doc, dict) else doc
    v1_ids = load_v1_ids(ROOT / "data" / "goldset" / "gold-v1.json")
    picked = pick_items(items, split, v1_ids)
    if limit:
        picked = picked[:limit]
    chap = chapter_texts(loaded["chunks"])

    rows: list[dict] = []
    t0 = time.perf_counter()
    for it in picked:
        q = str(it.get("question") or "")
        res = pipeline_mod.ask_once(pipe, q, options=options)
        ans = res["answer"] or {}
        text = str(ans.get("answer") or "")
        verdict = judge_mod.judge_answer(
            str(it.get("category") or ""), text, it, question=q, chapter_texts=chap)
        cites = judge_mod.citations_support(ans, loaded["chunks"])
        row = {
            "question_id": it.get("question_id"),
            "category": str(it.get("category") or ""),
            "split": it.get("split"),
            "question": q,
            "answer": text,
            "abstained": bool(ans.get("abstained")),
            "judge": verdict,
            "citations_ok": cites["ok"],
            "n_citations": cites["n_cites"],
            "arms": {"prod": {"pass": verdict["pass"] is True,
                              "reason": verdict["reason"],
                              "answered": bool(text.strip())}},
        }
        # ---- W4b：v2 只在显式要求时才算，且**只在此时才多写键** ----------
        # `judge_mode == "v1"` 时这一整段不执行，`row` 的字面量与改造前逐位相同
        # （J3：v1 产物必须逐题逐字段相等，多一个 `"judge_v2": null` 也算不等）。
        if judge_mode != "v1":
            ctx = res.get("context")
            v2 = _v2_call(judge_v2, row["category"], text, it, ans, question=q,
                          chapter_texts=chap, context=ctx,
                          chapter_cap=(options or {}).get("chapter_cap"))
            row["judge_v2"] = v2
            row["arms"]["v2"] = {"pass": v2["pass"] is True,
                                 "reason": v2["reason"],
                                 "answered": bool(text.strip()),
                                 "undecidable": v2["pass"] is None}
            row.update(context_note(res))
        rows.append(row)
    elapsed = (time.perf_counter() - t0) * 1000.0
    arms = judge_mode_arms(judge_mode)
    report = {
        "generation_id": generation_id,
        "gold_set": str(gold_path),
        "split": split,
        "token_budget": budget,
        "prod_profile": bool(prod),
        "prod_options": _options_for_report(options),
        "n_items": len(rows),
        "elapsed_ms": round(elapsed, 1),
        "per_item_ms": round(elapsed / len(rows), 1) if rows else 0.0,
        "judge_schema": "arkrag.judge/1",
        "chapters_seen": len(chap),
        **(tally(rows, arms)),
        "rows": rows,
    }
    # ---- W4b 附加块（`v1` 档一个都不加） ---------------------------------
    if judge_mode != "v1":
        rep_extra: dict = {
            "judge_mode": judge_mode,
            "judge_arms": arms,
            "judge_schema_v2": "arkrag.judge_v2/1",
            "judge_v2_entry": resolve_v2_entry(judge_v2)[1],
            "judge_v2_call": {
                **v2_call_shape(resolve_v2_entry(judge_v2)[1]),
                "chapter_cap_option": int((options or {}).get("chapter_cap") or 0),
                "note": "chapter_cap 为 0（未开 W3 截断）时不传该参数，"
                        "由 judge_v2 用自己的登记默认值；"
                        "开了就传同一个数，否则两套判据在 cross_chunk 上口径不同",
            },
            "flips": flip_counts(rows),
            "stats": stats_block(rows, arms),
            "context_support": {
                "rows_with_context": sum(1 for r in rows if r.get("context_available")),
                "rows_without_context": sum(
                    1 for r in rows if not r.get("context_available")),
                "limitation": "context 缺失时 v2 的 support 轴拿不到本次断言，"
                              "该轴失败不代表答案错",
            },
        }
        if "v2" in arms:
            rep_extra["floors"] = floor_report(
                int(report["totals"]["v2"]["scorable_pass"]),
                int(report["totals"]["v2"]["scorable_n"]), arm="v2")
        report.update(rep_extra)
    return report


def print_report(rep: dict) -> None:
    print("=" * 84)
    print(f"按题型判据评测  世代 {rep['generation_id']}  split={rep['split']}"
          f"  budget={rep['token_budget']}  n={rep['n_items']}"
          f"{'  [生产档]' if rep.get('prod_profile') else ''}"
          f"{'  [判据 ' + str(rep['judge_mode']) + ']' if rep.get('judge_mode') else ''}")
    print(f"平均 {rep['per_item_ms']} ms/题")
    print("-" * 84)
    arms = judge_mode_arms(str(rep.get("judge_mode") or "v1"))
    if len(arms) == 1:
        # ---- J3：v1 单臂的打印必须与改造前逐字相同 ------------------------
        print(f"{'类别':<18}{'n':>4}{'答对':>8}{'开口':>8}   说明")
        for cat, e in rep["per_category"].items():
            a = e["arms"]["prod"]
            if a["pass"] is None:
                print(f"{cat:<18}{e['n']:>4}{'-':>8}{a['opened']:>8}   结构类：拒判（剔分母）")
            else:
                print(f"{cat:<18}{e['n']:>4}{a['pass']:>8}{a['opened']:>8}   ")
        print("-" * 84)
        t = rep["totals"]["prod"]
        print(f"可判分母 {t['scorable_pass']}/{t['scorable_n']} = {t['scorable_rate'] * 100:.2f}%")
        print(f"全分数  {t['scorable_pass']}/{t['all_n']} = {t['all_rate'] * 100:.2f}%"
              f"（结构类按没答对计入）")
        print(f"开口率  {t['opened']}/{t['all_n']} = {t['opened_rate'] * 100:.2f}%")
    else:
        _print_dual(rep, arms)
    # ---- J5：地板三行**每次必报**（v1 / v2 / both 都有） -------------------
    _print_floors(rep)


def _print_dual(rep: dict, arms: list[str]) -> None:
    """两套判据并列打印。只报一套就是选择性披露（红线 11 同型）。"""
    print(f"{'类别':<18}{'n':>4}{'v1答对':>8}{'v2答对':>9}{'v1开口':>8}{'v2开口':>8}")
    for cat, e in rep["per_category"].items():
        a1, a2 = e["arms"].get("prod"), e["arms"].get("v2") or {}
        if a1["pass"] is None:
            print(f"{cat:<18}{e['n']:>4}{'-':>8}{'-' if a2.get('pass') is None else a2['pass']:>9}"
                  f"{a1['opened']:>8}{a2.get('opened', 0):>8}   结构类：拒判（剔分母）")
        else:
            print(f"{cat:<18}{e['n']:>4}{a1['pass']:>8}{a2.get('pass', 0):>9}"
                  f"{a1['opened']:>8}{a2.get('opened', 0):>8}")
    print("-" * 84)
    t1, t2 = rep["totals"]["prod"], rep["totals"]["v2"]
    print(f"可判分母  v1 {t1['scorable_pass']}/{t1['scorable_n']}"
          f" = {t1['scorable_rate'] * 100:.2f}%"
          f"   |   v2 {t2['scorable_pass']}/{t2['scorable_n']}"
          f" = {t2['scorable_rate'] * 100:.2f}%")
    print(f"全分数    v1 {t1['scorable_pass']}/{t1['all_n']} = {t1['all_rate'] * 100:.2f}%"
          f"   |   v2 {t2['scorable_pass']}/{t2['all_n']} = {t2['all_rate'] * 100:.2f}%")
    print(f"开口率    v1 {t1['opened']}/{t1['all_n']} = {t1['opened_rate'] * 100:.2f}%"
          f"   |   v2 {t2['opened']}/{t2['all_n']} = {t2['opened_rate'] * 100:.2f}%")
    fl = rep.get("flips") or {}
    c = fl.get("counts") or {}
    print("-" * 84)
    print(f"翻转四类  PP={c.get('PP', 0)} PF={c.get('PF', 0)} "
          f"FP={c.get('FP', 0)} FF={c.get('FF', 0)}"
          f"   合计 {fl.get('decidable_n', 0)}（= 两套都能判的题数）")
    if fl.get("v2_undecidable_n"):
        print(f"  ! v2 不可判 {fl['v2_undecidable_n']} 题（不计入四类，"
              f"也不当 False——缺失输入不许静默变成判定）")
    if fl.get("pf_by_reason"):
        print("  PF 按互斥原因：" + "  ".join(
            f"{k}={v}" for k, v in (fl.get("pf_by_reason") or {}).items()))
    if fl.get("fp_items"):
        print(f"  ! FP={len(fl['fp_items'])} 题（v1 拒、v2 放行）：" + ", ".join(
            str(x.get("question_id")) for x in fl["fp_items"][:12])
            + (" ..." if len(fl["fp_items"]) > 12 else ""))
    st = rep.get("stats") or {}
    cl = st.get("cluster") or {}
    print("-" * 84)
    print(f"簇结构    题数 {cl.get('n_items', 0)}  唯一题面 {cl.get('unique_questions', 0)}"
          f"  模板簇 {cl.get('n_clusters', 0)}"
          f"  有效簇数(Kish) {cl.get('effective_clusters', 0)}"
          f"  簇大小 {cl.get('cluster_sizes', [])}"
          f"（{cl.get('n_gram', 0)}-gram Jaccard >= {cl.get('jaccard_threshold', 0)}）")
    for arm in arms:
        a = (st.get("arms") or {}).get(arm) or {}
        iid, clu = a.get("iid") or {}, a.get("cluster") or {}
        print(f"CI[{arm}]  iid {iid.get('mean', 0) * 100:.2f}%"
              f" CI[{iid.get('ci_low', 0) * 100:.2f},{iid.get('ci_high', 0) * 100:.2f}]"
              f"   cluster {clu.get('mean', 0) * 100:.2f}%"
              f" CI[{clu.get('ci_low', 0) * 100:.2f},{clu.get('ci_high', 0) * 100:.2f}]")
    pair = st.get("paired_v1_v2") or {}
    if pair.get("n_paired"):
        pi, pc = pair.get("iid") or {}, pair.get("cluster") or {}
        print(f"配对Δ(v1->v2) n={pair['n_paired']}  簇 {pair.get('n_clusters', 0)}")
        print(f"  iid     Δ={pi.get('delta_pp', 0):+.2f}pt"
              f" CI[{pi.get('ci_low_pp', 0):+.2f},{pi.get('ci_high_pp', 0):+.2f}]"
              f" p={pi.get('p_one_sided', 1):.4f}")
        print(f"  cluster Δ={pc.get('delta_pp', 0):+.2f}pt"
              f" CI[{pc.get('ci_low_pp', 0):+.2f},{pc.get('ci_high_pp', 0):+.2f}]"
              f" p={pc.get('p_one_sided', 1):.4f}")
    infl = st.get("variance_inflation") or {}
    ess = st.get("effective_sample_size") or {}
    if infl.get("estimable"):
        print(f"方差膨胀  {infl['value']:.4f}（{infl.get('reason', '')}）"
              f"   有效样本量 {ess.get('value', 0):.2f}")
    else:
        # 计划书明令：算不出来就标不可估计，不许用 m-bar≈2.5,rho≈0.6 的假设值
        print(f"方差膨胀  不可估计（{infl.get('reason', '')}）；有效样本量 不可估计")


def _print_floors(rep: dict) -> None:
    """J5：三条地板**每次必报**，各带自己的分母与估计方法，不做合计。"""
    print("-" * 84)
    print("地板三行（每次必报；各自分母与估计方法随行；不可相加）")
    # 单臂 v1 时没有 `floors` 块（J3 不给 v1 的 JSON 加键），就地现算一份供打印。
    block = rep.get("floors")
    if block is None:
        t = rep["totals"]["prod"]
        block = floor_report(int(t["scorable_pass"]), int(t["scorable_n"]), arm="prod")
    for line in block["lines"]:
        corr = line.get("corrected") or {}
        got = (f"{corr['value'] * 100:.2f}%" if corr.get("estimable")
               else f"不可估计（{corr.get('reason', '')}）")
        print(f"  {line['label']:<10} {line['floor_count']}/{line['floor_n']}"
              f" = {line['floor_rate'] * 100:.2f}%"
              f"   [分母 {line['floor_n']}；{line['method']}]")
        print(f"             校正 ({line['pass_count']}-{line['floor_count']})"
              f"/({line['n']}-{line['floor_count']}) = {got}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="按题型判据评测答对率（arkrag.judge 唯一消费端）")
    ap.add_argument("--generation", default="g-20261103T132152Z-06a741")
    ap.add_argument("--gold-set", default=str(ROOT / "data" / "goldset" / "gold-v2.json"))
    ap.add_argument("--split", default="dev", choices=("dev", "holdout", "clean", "all"))
    ap.add_argument("--limit", type=int, default=0, help="0 表示不截断")
    ap.add_argument("--token-budget", type=int, default=None,
                    help="不传则吃 retrieve.DEFAULT_BUDGET（现 12000）；"
                         "复现 6000 档读数必须显式传 6000。"
                         "带 --prod 时不传则吃 pipeline.PROD_PROFILE（现 24000）")
    ap.add_argument("--prod", action="store_true",
                    help="按生产档跑（引文定位 + 桥接 + 图证据预留 + 预算取 PROD_PROFILE）；"
                         "不传则是历史默认档，6000/12000 档的旧读数靠它复现")
    ap.add_argument("--judge", default="v1", choices=JUDGE_MODES, dest="judge_mode",
                    help="W4b 判据档位。`v1`（默认）只用既有判据，产物与改造前逐位相同；"
                         "`v2` 只用 judge_v2；`both` 两套并行记分并报翻转与聚类 CI。"
                         "v2/both 需要 arkrag.judge_v2 可 import，否则启动即退出 2，"
                         "不会静默退回 v1。")
    ap.add_argument("--json", default="", help="报告落盘路径")
    ap.add_argument("--option", action="append", default=[], metavar="K=V",
                    help="实验开关，可重复：追加/覆盖 ask_once 的 options 键。"
                         "默认空 = 与历史读数逐位等价。"
                         "例：`--option max_direct_units=14`（W1 cap）。"
                         "值会被解析成 int/bool/str（'true'/'false' -> bool）。")
    ap.add_argument("--calibration", default="",
                    help="W2：bridge_calib 映射表 JSON 路径。给了就注入 "
                         "options['bridge_calibration']。不传则 options 里没有这个键"
                         "（`top1_margin` 会因此全部 fail-closed 拒答）。")
    args = ap.parse_args(argv)

    # ---- W4b：v2/both 但 judge_v2 没就绪 → 启动即明确报错退出（退出码 2）----
    # 不许静默退回 v1：那会产出一份"两列同数"的假双口径报告，比不跑更糟。
    # `--judge` 的默认值写死在 argparse 里，argparse 自己还会挡住非法取值
    # （`choices=JUDGE_MODES`），所以这里是"取值合法但依赖缺失"的唯一拦点。
    judge_v2 = None
    if args.judge_mode != "v1":
        judge_v2 = load_judge_v2()
        if judge_v2 is None:
            print(f"[2] --judge {args.judge_mode} 需要 `arkrag.judge_v2`，"
                  f"但它不可用：{_JUDGE_V2_ERROR or '模块缺失'}。"
                  f"不会静默退回 v1——那会产出一份两列同数的假双口径报告。",
                  file=sys.stderr)
            return 2

    extra: dict = {}
    for raw in args.option:
        if "=" not in raw:
            print(f"[2] --option 需要 K=V 形式，收到 {raw!r}", file=sys.stderr)
            return 2
        k, _, v = raw.partition("=")
        k, v = k.strip(), v.strip()
        if not k:
            print(f"[2] --option 的键为空：{raw!r}", file=sys.stderr)
            return 2
        low = v.lower()
        if low in ("true", "false"):
            extra[k] = (low == "true")
        else:
            try:
                extra[k] = int(v)
            except ValueError:
                extra[k] = v

    rep = run(args.generation, Path(args.gold_set), split=args.split,
              limit=args.limit, token_budget=args.token_budget, prod=args.prod,
              extra_options=extra or None,
              calibration_path=Path(args.calibration) if args.calibration else None,
              judge_mode=args.judge_mode)
    print_report(rep)
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n落盘 {out}（{out.stat().st_size} B）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
