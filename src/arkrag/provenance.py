# -*- coding: utf-8 -*-
"""门报告的「可追溯」契约：一份报告必须能回答五件事，否则不算通过。

## 为什么要有这个模块

到 P0 为止，仓库里已经有九份门报告（`k1_checks` / `k2_hard_gates` / `k2_perf` /
`k2_gray` / `k3_threshold_gate` / `k4_drill` / `k2_publish_check` / `k3_signif` /
`p1c_leak_audit`），每份自己决定写哪些字段。实测结果：

| 报告 | `generated_at` | 代码版本 | 数据版本 | 机器 |
|---|---|---|---|---|
| `k2_perf.json` | 有 | 无 | 有（config 里写 gold_set 路径） | 有（python/platform/cpu_count） |
| `k2_gray.json` | 有 | 无 | 无 | 部分 |
| `k2_hard_gates.json` | **无** | 无 | 无 | 无 |
| `k3_threshold_gate.json` | **无** | 无 | 无 | 无 |
| `k4_drill.json` | **无** | 无 | 无 | 无 |

后果不是「报告不好看」，是**三个月后没人能回答「这个 0.7 是哪份代码、哪个候选、
哪套数据跑出来的」**。项目已经因为「同一判定写两处、取最松者生效」栽过三次，
`tests/test_single_source_of_truth.py` 守的就是这个；本模块是同一原则在
「报告字段」上的落地：**字段名与判定语义只在这里定义一次**，各脚本调用它。

## 七个字段

Lead 定的清单，逐条对应一个「能不能复现」的问题：

| 字段 | 回答的问题 | 缺失时 |
|---|---|---|
| `generated_at` | 什么时候跑的 | 无法判断报告新旧 |
| `generation_id` | 跑的是哪个候选世代 | 报告绑错世代也没人发现 |
| `git_commit` | 哪份代码 | 无法区分「代码改了」与「数据变了」 |
| `code_sha256_16` | 代码树指纹（含未提交改动） | 同上，且未提交时 commit 号不变 |
| `dataset` | 哪套题（gold / split + 指纹） | 换题库前后两份报告无法区分 |
| `inputs` | 指标吃进去的文件各自 sha256 | 指标不可重算 |
| `environment` | 机器 / Python / OS | 性能类指标跨机不可比 |

`code_sha256_16` 与 `git_commit` **不是重复**：后者在有未提交改动时不变，
前者变。Lead 这一轮踩过的「改完代码忘了提交、两份报告 commit 号相同但结果不同」
就是这个缺口。

## 三态：pass / fail / unknown

**`unknown` 不是通过。** 这是本模块唯一想用一条断言钉死的事：

* `pass`   —— 字段齐、值合法、能复现；
* `fail`   —— 字段在，但值不合法（如 `caliber` 写了已作废的 `equiv_v1_wide`）；
* `unknown` —— 字段**缺**，或值存在但为空（`git_commit=""` 表示没装 git / 不在仓库里）。

区别很重要：`fail` 是「查出错了」，`unknown` 是「查不了」。把后者算成通过，
等于用「没查」冒充「查过没问题」——本项目三次事故里有两次是这个形态。

历史包袱：`scripts/k2_publish_check.py` 早于本模块，用的是 `undetermined`
而不是 `unknown`。那是同一概念的不同拼写，本模块**不改它**（消费者改生产者的
报告格式，等于让报告与产生它的代码对不上），只在新代码里统一用 `unknown`，
并在 `LEGACY_ALIASES` 里登记这个已知差异。

## 指标口径（`caliber`）

`direct_relation` 恒 1 个 `gold_chunk_ids`，而一条边可能承载 N 个块
（`edge.chunks` 是列表、`edge.evidence` 是单一字符串），所以「检索有没有命中」
取决于按哪个口径判。两个可用口径：

* `gold_only` —— 只认 `gold_chunk_ids`（现行，代表证据语义）；
* `equiv_v2` —— 认「同一关系、两端点对、且该断言 evidence 里两个端点名都在」的
  全部承载块（V2 严口径，47/96 题存在非 gold 等价块）。

**`equiv_v1_wide` 已作废，不许进报告。** 它把「结构字段对上、但 evidence 根本没
在说这回事」的断言也算成等价证据（反例：`'天灾在我们头顶发生的可能性就越大'`
被算成「天灾破坏切尔诺伯格」的证据），据此算出的 +30pt 是假的。

三条硬约束，全部做成可判定的：

1. 任何出现 recall / R@n 的行，必须带 `caliber`，缺这个字段判 **unknown**；
2. `caliber` 值不在白名单（比如写了 `equiv_v1_wide`）判 **fail**；
3. 只报 R@n、不并列下游量的行判 **fail** —— 实测反例：`hop0.5` 配置 R@20 全场
   第二高（0.7379），但「真进上下文题数」净增益 0、引用覆盖 51→48。只看 R@20
   的报告会给出「这个配置该上」的错误建议。

## 用法

    from arkrag import provenance

    block = provenance.collect(generation_id=gid, dataset=..., inputs=[...])
    payload = {"passed": True, **block, "checks": [...]}
    ok, problems = provenance.verify(payload)

`collect()` 只读环境与文件，不写盘；`verify()` 是纯函数，给测试喂合成 payload。
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import config

# ---------------------------------------------------------------- 常量

#: 七字段。顺序即报告里建议的书写顺序（人读的那份也按这个顺序排）。
#: 七字段。顺序即报告里的展示顺序，也是 `verify()` 里逐条判定的顺序。
REQUIRED_FIELDS: tuple[str, ...] = (
    "generated_at",
    "generation_id",
    "git_commit",
    "code_sha256_16",
    "dataset",
    "inputs",
    "environment",
)

#: 三态。`unknown` 不是通过——见模块 docstring。
PASS = "pass"
FAIL = "fail"
UNKNOWN = "unknown"

VERDICTS: tuple[str, ...] = (PASS, FAIL, UNKNOWN)

#: `k2_publish_check.py` 的历史拼写，同一概念。登记在这里是为了让「两种拼写同时
#: 存在」成为一个**已知且被记录**的事实，而不是某天被人当成 bug 顺手改掉。
LEGACY_ALIASES: dict[str, str] = {"undetermined": UNKNOWN}

#: 允许出现在报告里的指标口径。作废的 `equiv_v1_wide` 刻意**不在**这里。
CALIBERS: tuple[str, ...] = ("gold_only", "equiv_v2")

#: 已作废、若出现必须判 fail 的口径。留单独常量是为了报错时能点名叫出它。
RETIRED_CALIBERS: tuple[str, ...] = ("equiv_v1_wide", "v1_wide", "equiv_v1")

#: 「只看 R@n 不够」——这些键出现时，同一行/同一块里必须另有下游量。
RECALL_KEYS: tuple[str, ...] = ("r5", "r10", "r20", "recall", "recall_at_5", "recall_at_20")

#: 认可的下游量。任选其一即可，但必须有一个。
DOWNSTREAM_KEYS: tuple[str, ...] = (
    "context_hits",          # 真进上下文的题数
    "citation_covered",      # 引用覆盖成立的题数
    "answer_cover",          # 答案覆盖
    "downstream",            # 泛用口袋
    "downstream_metric",
)

_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$")
_SHA16_RE = re.compile(r"^[0-9a-f]{16}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")


class ProvenanceError(RuntimeError):
    """构造 provenance 块时的硬错误（被查文件不存在等）。"""


# ---------------------------------------------------------------- 代码/数据指纹


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def code_sha256_16(roots: tuple[str, ...] = ("src/arkrag", "scripts")) -> str:
    """代码树指纹：把若干目录下所有 `*.py` 的**相对路径 + 内容**按固定顺序喂给 sha256。

    为什么不用 `git rev-parse HEAD`：工作区有未提交改动时它不变。Lead 这一轮
    踩过——改完代码没提交，两份报告 commit 号相同、结果不同，看报告的人会以为
    「同样的代码跑出两个数」。

    路径用相对形式（`config.rel_to_root`），换机器不变；排序固定，否则同一份树
    两次算出不同的值。
    """
    h = hashlib.sha256()
    for root in roots:
        base = Path(config.ROOT) / root
        if not base.is_dir():
            continue
        for path in sorted(base.glob("*.py")):
            rel = config.rel_to_root(path).replace("\\", "/")
            h.update(rel.encode("utf-8"))
            h.update(b"\x00")
            h.update(path.read_bytes())
            h.update(b"\x00")
    return h.hexdigest()[:16]


def file_sha16(path) -> str:
    """单个文件的 sha256 前 16 位；文件不存在返回 `""`（由调用方判 unknown）。"""
    p = Path(path)
    if not p.exists():
        return ""
    return _sha256_file(p)[:16]


def dataset_block(
    *,
    gold_path=None,
    split_path=None,
    gold_total: int | None = None,
) -> dict:
    """数据集版本：题库与切分的指纹 + 题数。

    `gold_total` 允许调用方显式给（报告里写死过 640），与文件实测不一致时
    由 `verify()` 报 fail ——「报告说 640、文件里 638」这种必须有人叫。
    """
    gold = Path(gold_path) if gold_path else Path(config.ROOT) / "data" / "goldset" / "gold-v2.json"
    split = Path(split_path) if split_path else Path(config.ROOT) / "data" / "goldset" / "split-v2.json"
    out: dict = {
        "gold_path": config.rel_to_root(gold),
        "gold_sha16": file_sha16(gold),
        "split_path": config.rel_to_root(split),
        "split_sha16": file_sha16(split),
        "gold_total": gold_total,
    }
    if gold_total is None and gold.exists():
        try:
            payload = json.loads(gold.read_text("utf-8"))
            items = payload.get("items") if isinstance(payload, dict) else None
            out["gold_total"] = len(items) if isinstance(items, list) else None
        except (OSError, ValueError):
            out["gold_total"] = None
    return out


def inputs_block(paths: list) -> list[dict]:
    """指标输入文件清单：每个文件一条 `{path, sha16, bytes}`，缺文件时 sha16 为空。

    **不给默认值**：调用方必须显式列出「这个指标吃了哪些文件」。默认去猜
    「大概是 gold 吧」正是本项目三次「两个合法数据源选错一个」的起点。
    """
    rows: list[dict] = []
    for path in paths:
        p = Path(path)
        if not p.is_absolute():
            p = Path(config.ROOT) / p
        rows.append(
            {
                "path": config.rel_to_root(p),
                "sha16": file_sha16(p),
                "bytes": p.stat().st_size if p.exists() else None,
            }
        )
    return rows


def environment_block() -> dict:
    """机器与解释器。性能类指标跨机不可比，所以这一项对 `k2_perf` 是硬要求。"""
    return {
        "python": platform.python_version(),
        "platform": sys.platform,
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": _cpu_count(),
    }


def _cpu_count() -> int | None:
    import os

    try:
        return os.cpu_count()
    except (NotImplementedError, OSError):  # pragma: no cover - 极端环境
        return None


def _git_commit() -> str:
    """当前 HEAD。不在仓库里 / 没装 git 时返回 `""` ——**空串表示 unknown，不表示无**。

    刻意不写 `release.git_revision()` 的别名：那个函数属于发布路径，本模块属于
    报告契约，两者生命周期不同。但两者都遵守同一条：取不到就留空，不猜。
    """
    try:
        from . import release

        return str(release.git_revision() or "")
    except Exception:  # noqa: BLE001 - 报告字段取不到不该让主流程崩
        return ""


def now_iso() -> str:
    """UTC ISO8601，秒级，`Z` 结尾——与仓库既有报告（`k2_perf` / `k2_gray`）同格式。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- 构造


def collect(
    *,
    generation_id: str,
    dataset: dict | None = None,
    inputs: list | None = None,
    generated_at: str | None = None,
) -> dict:
    """拼出完整的七字段块。**任何一项取不到就留空/留 None，不填假值。**

    留空的后果由 `verify()` 判 `unknown`（不是 pass）。这是刻意的：如果
    `collect()` 自己「尽力填」，缺失就永远看不出来了。
    """
    return {
        "generated_at": generated_at or now_iso(),
        "generation_id": str(generation_id or ""),
        "git_commit": _git_commit(),
        "code_sha256_16": code_sha256_16(),
        "dataset": dataset if dataset is not None else dataset_block(),
        "inputs": inputs if inputs is not None else [],
        "environment": environment_block(),
    }


# ---------------------------------------------------------------- 逐字段判定


def _blank(value) -> bool:
    """「没填」的统一判据：None / 空白串 / 空容器。`0` 与 `False` 不算没填。"""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


def judge_generated_at(value) -> tuple[str, str]:
    if _blank(value):
        return UNKNOWN, "缺 generated_at：无法判断这份报告是新的还是上一轮的残留"
    if not isinstance(value, str) or not _ISO_RE.match(value.strip()):
        return FAIL, f"generated_at 不是 ISO8601（拿到 {value!r}），无法与其他报告比较先后"
    return PASS, f"generated_at={value}"


def judge_generation_id(value) -> tuple[str, str]:
    if _blank(value):
        return UNKNOWN, "缺 generation_id：报告没绑世代，拿错代的报告也没人会发现"
    text = str(value)
    if not re.match(r"^g-\d{8}T\d{6}Z-[0-9a-f]{6}$", text):
        return FAIL, f"generation_id 形状不对（拿到 {text!r}），应为 g-YYYYMMDDTHHMMSSZ-xxxxxx"
    return PASS, f"generation_id={text}"


def judge_git_commit(value) -> tuple[str, str]:
    if _blank(value):
        return UNKNOWN, "缺 git_commit（不在仓库里或没装 git）：无法确定是哪份代码"
    if not _COMMIT_RE.match(str(value)):
        return FAIL, f"git_commit 不像提交号（拿到 {value!r}）"
    return PASS, f"git_commit={value}"


def judge_code_sha(value) -> tuple[str, str]:
    if _blank(value):
        return UNKNOWN, "缺 code_sha256_16：工作区有未提交改动时 commit 号不变，只有这一项能区分"
    if not _SHA16_RE.match(str(value)):
        return FAIL, f"code_sha256_16 应为 16 位小写十六进制（拿到 {value!r}）"
    return PASS, f"code_sha256_16={value}"


def judge_dataset(value) -> tuple[str, str]:
    if _blank(value) or not isinstance(value, dict):
        return UNKNOWN, "缺 dataset：无法判断这份指标是换题库之前还是之后跑的"
    problems: list[str] = []
    if _blank(value.get("gold_path")):
        problems.append("没有 gold_path")
    if _blank(value.get("gold_sha16")):
        problems.append("gold_sha16 为空（文件不在盘上？）")
    if value.get("gold_total") is None:
        problems.append("没有 gold_total 题数")
    if problems:
        return UNKNOWN, "dataset 不完整：" + "；".join(problems)
    return PASS, f"dataset gold={value.get('gold_path')} sha16={value.get('gold_sha16')} n={value.get('gold_total')}"


def judge_inputs(value) -> tuple[str, str]:
    """指标输入清单必须每一行都带 sha16。

    **两种「缺输入」都要判 unknown，不许判 pass：**

    1. `inputs` 整个没填 / 空列表；
    2. 行里算不出 sha16（文件不在盘上，或调用方只传了路径没传 sha16）。

    第 2 条的第二种形态是 P0 实测出来的：`collect()` 允许直接传路径列表，
    于是行是**纯字符串**。早期实现只检查 `isinstance(row, dict)` 的行，
    字符串行一条都不检查 → 返回 `("pass", "inputs 1 个文件都已登记 sha16")`，
    **一个 sha16 都没登记，却报「都已登记」**。这类「跳过不认识的形状」的判据
    必须再补一条：「不认识的形状本身算不算合格」。答案是——不算。
    """
    if _blank(value) or not isinstance(value, list):
        return UNKNOWN, "缺 inputs：没有列出指标吃了哪些文件，指标不可重算"
    missing: list[str] = []
    unknown_shape = 0
    for row in value:
        if isinstance(row, dict):
            if _blank(row.get("sha16")):
                missing.append(str(row.get("path") or "?"))
            continue
        if isinstance(row, str) and row.strip():
            # 纯路径串：连 sha16 字段都没有，指标不可重算。
            missing.append(row.strip())
            continue
        # 数字 / None / 嵌套列表之类：既没路径也没 sha16。
        unknown_shape += 1
    if missing:
        return UNKNOWN, (
            f"inputs 里有 {len(missing)} 个文件算不出 sha16（只在清单里写了路径不够，"
            f"指标不可重算）：{'、'.join(missing)}"
        )
    if unknown_shape:
        return UNKNOWN, (
            f"inputs 里有 {unknown_shape} 行既不是路径串也没有 sha16，无法确认是哪份文件"
        )
    return PASS, f"inputs {len(value)} 个文件都已登记 sha16"


def judge_environment(value) -> tuple[str, str]:
    if _blank(value) or not isinstance(value, dict):
        return UNKNOWN, "缺 environment：性能类指标跨机不可比"
    problems = [k for k in ("python", "platform", "os") if _blank(value.get(k))]
    if problems:
        return UNKNOWN, "environment 缺字段：" + "、".join(problems)
    return PASS, f"environment python={value.get('python')} os={value.get('os')}"


_JUDGES = {
    "generated_at": judge_generated_at,
    "generation_id": judge_generation_id,
    "git_commit": judge_git_commit,
    "code_sha256_16": judge_code_sha,
    "dataset": judge_dataset,
    "inputs": judge_inputs,
    "environment": judge_environment,
}


# ---------------------------------------------------------------- 汇总


def _fold(verdicts: list[str]) -> str:
    """多条判定的合并规则。

    **fail 优先于 unknown**：能确定错了就先说错了，别让「另一项查不了」
    把确定的错误稀释成「查不了」。两者都没有才 pass。
    这就是「unknown 不许算通过」在合并层的实现——它只在**没有** fail 时
    才体现为「不算通过的通过」。
    """
    if FAIL in verdicts:
        return FAIL
    if UNKNOWN in verdicts:
        return UNKNOWN
    return PASS


def verify(payload: dict) -> dict:
    """检查一份报告是否满足七字段 + 三态契约。纯函数，不读盘、不抛异常。

    返回 `{verdict, fields, missing, invalid, detail}`：

    * `fields` —— 每个字段各自的 `{field, verdict, detail}`，**缺的是哪个字段
      直接可查**（不是笼统一句「无法判定」）；
    * `verdict` —— `fail` / `unknown` / `pass`；
    * `missing` / `invalid` —— 两类问题分开列，因为处置不同：缺的要补采集，
      不合法的是写错了。

    这个「能反查到缺哪个字段」的形状，来自 P1-c 踩到的坑：`gold_assertion_ids`
    恒为 None 时脚本安静地把每条降级成「无法判定」，退出码 0、不报错，看上去
    跟「真的判不了」一模一样。**缺输入必须与真判定不出来是两种不同的输出。**
    """
    if not isinstance(payload, dict):
        return {
            "verdict": FAIL,
            "fields": [],
            "missing": list(REQUIRED_FIELDS),
            "invalid": [],
            "detail": f"报告不是 dict（拿到 {type(payload).__name__}），无法检查任何字段",
        }
    fields: list[dict] = []
    missing: list[str] = []
    invalid: list[str] = []
    for name in REQUIRED_FIELDS:
        verdict, detail = _JUDGES[name](payload.get(name))
        fields.append({"field": name, "verdict": verdict, "detail": detail})
        if verdict == UNKNOWN:
            missing.append(name)
        elif verdict == FAIL:
            invalid.append(name)
    verdict = _fold([f["verdict"] for f in fields])
    if verdict == PASS:
        detail = "七字段齐备且合法"
    elif verdict == FAIL:
        detail = f"{len(invalid)} 项不合法：{'、'.join(invalid)}"
        if missing:
            detail += f"；另有 {len(missing)} 项缺失：{'、'.join(missing)}"
    else:
        detail = f"{len(missing)} 项缺失，无法判定：{'、'.join(missing)}"
    return {
        "verdict": verdict,
        "fields": fields,
        "missing": missing,
        "invalid": invalid,
        "detail": detail,
    }


# ---------------------------------------------------------------- 口径


def judge_caliber(value) -> tuple[str, str]:
    """单独判 `caliber`。

    与七字段分开，因为它只对**含 recall 的行**是硬要求：一份只测吞吐的报告
    没有口径可言，不该因此被判 unknown。调用方用 `verify_metric_row()` 组合。
    """
    if _blank(value):
        return UNKNOWN, "缺 caliber：同一个 R@n 在不同口径下不是同一个数"
    text = str(value)
    if text in RETIRED_CALIBERS:
        return FAIL, (
            f"caliber={text} 已作废：它把「结构字段对上、evidence 却没在说这回事」"
            "的断言也算成等价证据，据此算出的 +30pt 是假的（Lead 2026-11-05 已撤回）"
        )
    if text not in CALIBERS:
        return FAIL, f"caliber={text} 不在白名单 {list(CALIBERS)}"
    return PASS, f"caliber={text}"


def _has_any(row: dict, keys: tuple[str, ...]) -> bool:
    return any(k in row and not _blank(row.get(k)) for k in keys)


def verify_metric_row(row: dict) -> dict:
    """检查**一行指标**是否合规：口径 + 下游量。

    一行里只要出现 recall 类键，就必须：

    1. 带 `caliber`（缺 → unknown，作废值 → fail）；
    2. 并列至少一个下游量（缺 → fail）。

    第 2 条的理由是实测反例：`hop0.5` 配置 R@20 = 0.7379（六配置里第二高），
    但「真进上下文题数」净增益 0、引用覆盖 51→48。只看 R@20 的报告会建议上这
    个配置。**R@n 单独一行时，它不构成「更好」的证据。**

    第二条判 fail 而不是 unknown：报告里明明能看到这一行、也能看到没有下游量，
    这是**看得出来的缺口**，不是「查不了」。
    """
    if not isinstance(row, dict):
        return {"verdict": FAIL, "detail": f"指标行不是 dict（{type(row).__name__}）",
                "missing": [], "caliber": None, "has_recall": False, "has_downstream": False}
    has_recall = _has_any(row, RECALL_KEYS)
    has_down = _has_any(row, DOWNSTREAM_KEYS)
    if not has_recall:
        return {"verdict": PASS, "detail": "不含 recall 键，不适用口径检查",
                "missing": [], "caliber": row.get("caliber"),
                "has_recall": False, "has_downstream": has_down}
    caliber_verdict, caliber_detail = judge_caliber(row.get("caliber"))
    if caliber_verdict != PASS:
        # `missing` 与 `verify()` 同形：判不了的时候必须能反查到缺的是哪一项。
        # P1-c 那次的教训是「缺输入」安静地变成了「无法判定」，两种输出要分得开。
        return {"verdict": caliber_verdict,
                "detail": f"{caliber_detail}；本行含 recall 键，口径是硬要求",
                "missing": ["caliber"] if caliber_verdict == UNKNOWN else [],
                "caliber": row.get("caliber"), "has_recall": True, "has_downstream": has_down}
    if not has_down:
        return {
            "verdict": FAIL,
            "detail": (
                "本行只报了 recall、没有并列下游量。实测反例：`hop0.5` 的 R@20=0.7379"
                "（六配置第二高），但真进上下文净增益 0、引用覆盖 51→48——"
                "只有 R@n 的行会给出相反的建议。请补 "
                f"{list(DOWNSTREAM_KEYS)} 中至少一项"
            ),
            "missing": ["downstream"],
            "caliber": row.get("caliber"),
            "has_recall": True,
            "has_downstream": False,
        }
    return {"verdict": PASS, "detail": f"{caliber_detail}，且并列了下游量",
            "missing": [], "caliber": row.get("caliber"),
            "has_recall": True, "has_downstream": True}


def verify_metric_rows(rows) -> dict:
    """批量。返回 `{verdict, rows, bad, unknown_rows}`——坏行逐条列出，不汇总成一个数。"""
    if not isinstance(rows, (list, tuple)):
        return {"verdict": FAIL, "rows": [], "bad": [], "unknown_rows": [],
                "detail": f"指标集合应为 list，拿到 {type(rows).__name__}"}
    results = [verify_metric_row(r) for r in rows]
    verdicts = [r["verdict"] for r in results]
    bad = [i for i, r in enumerate(results) if r["verdict"] == FAIL]
    unknown_rows = [i for i, r in enumerate(results) if r["verdict"] == UNKNOWN]
    verdict = _fold(verdicts)
    detail = f"{len(rows)} 行指标，{len(bad)} 行不合规、{len(unknown_rows)} 行无法判定" if rows else "没有指标行"
    return {"verdict": verdict, "rows": results, "bad": bad,
            "unknown_rows": unknown_rows, "detail": detail}


# ---------------------------------------------------------------- 与发布门对接


def gate_ok(payload: dict) -> bool:
    """报告能不能算「过」。

    **`unknown` 一律不算过。** 这个函数存在的唯一目的就是让调用方不必自己
    记这条规则——每个脚本各写一遍 `if all_passed:` 正是漏掉 unknown 的地方。
    """
    if not isinstance(payload, dict):
        return False
    provenance_verdict = verify(payload)["verdict"]
    if provenance_verdict != PASS:
        return False
    return payload.get("all_passed") is True or payload.get("passed") is True
