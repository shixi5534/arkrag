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

#: 「排序加成」类指标：出现这些键时，同一行必须同时给下面三个数。
#: 名字带 `signal_` 前缀而不是直接复用 `cooc_boosted`，是为了让这类判据能覆盖
#: A 之后可能加的其它信号（hop_line / speaker / entity），不必每加一个信号就改门。
BOOST_KEYS: tuple[str, ...] = (
    "cooc_boosted",
    "hop_boosted",
    "speaker_boosted",
    "entity_boosted",
    "signal_boosted",
)

#: 信号密度必须能反查的三件套（缺一即 unknown）。
#:
#: 只报 `density` 不够：0.8 可能是 4/5，也可能是 8/10，后者在「密度 4+ 无效」的
#: 分界另一侧——同一个 0.8 在两个不同的结论区间里。
DENSITY_FIELDS: tuple[str, ...] = ("signal_boosted", "context_units", "density")

#: 「一次运行的累计耗时」。出现这些键时，同一行必须另有 CPU 时间。
WALL_CLOCK_KEYS: tuple[str, ...] = (
    "elapsed_s",
    "wall_s",
    "duration_s",
    "total_s",
    "wall_clock_s",
)

#: 认可的 CPU / 进程时间。`process_time` 是 Python 的 `time.process_time()`。
CPU_KEYS: tuple[str, ...] = (
    "cpu_s",
    "cpu_time_s",
    "process_time_s",
    "process_s",
)

#: 「引用覆盖」类指标：出现这些键时，同一行必须**并列三个分母**。
#:
#: 同一个「引用覆盖 0.17」有三种分母，三者答的是不同的问题：
#:   mixed     /300 —— 全部应答题。把三种互不相干的失败混在一个分母里：
#:                    断言层没抽到 gold 那句、required 块压根没进上下文、
#:                    块在手却没发这条引用。它涨落时看不出该修哪一层。
#:   reachable /194 —— 只算「gold 那句确实被抽成某条断言」的题，即断言层天花板。
#:   ungated   /72  —— 只算「断言可达 **且** required 块全进了上下文」的题，
#:                    两道上游闸都过了，只剩「发不发这条引用」。
#: 只报一个数，读者无法判断这次涨落发生在哪一层——这正是 P3 最需要的信息。
COVERAGE_KEYS: tuple[str, ...] = (
    "citation_coverage",
    "required_citation_coverage",
    "coverage",
)

#: 三个分母的名字。顺序有语义：从最宽到最窄，题数必须单调不增
#: （窄分母的题集是宽分母题集的子集）。写成常量是为了让报告与判据用同一套字。
COVERAGE_DENOMINATORS: tuple[str, ...] = ("mixed", "reachable", "ungated")

#: P3 盯的那一个。换成 mixed 会让「上游没抽到」和「下游没发引用」两类失败
#: 在同一个数字里互相掩盖，前者是抽取/挖掘的活，后者才是 P3 的活。
COVERAGE_TARGET = "ungated"

#: 「真跑净增益」÷「top-K 变动集合大小」= 信号转换率。
SHUFFLE_GAIN_KEYS: tuple[str, ...] = ("net_gain", "gain")
SHUFFLE_CHANGED_KEYS: tuple[str, ...] = ("top20_changed", "changed_questions", "changed_set_size")

#: 转换率低于它即判「洗牌」：名次动了，结果没动。
SHUFFLE_THRESHOLD = 0.1

#: 洗牌唯一允许的用途：**淘汰**这一版，不是**预测**下一版。
SHUFFLE_USAGE_OK = "elimination_only"

#: 「名次进前 5」类指标出现时，必须附带的第 5 名转化率。
#: 实测：名次 1–4 转化率 1.0000（181/181），**第 5 名 5/12 = 0.4167**，
#: 第 6 名 1/18 = 0.0556，第 7–20 名 0/44。断崖在第 4 与第 5 名之间。
#: 不附这个数，读者会把「R@5 高」读成「进了前 5 就能用」。
RANK5_KEYS: tuple[str, ...] = ("r5", "recall_at_5", "top5", "top_5")
RANK5_CONVERSION = 0.4167

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


# ---------------------------------------------------------------- 信号密度


def density_of(boosted, units):
    """`boosted / units`。`units` 为 0（或非数）时返回 `None`，不返回 0 也不抛。

    返回 `None` 而不是 0：0 会被读成「密度真的是零」，而真实情况是「这道题没有
    上下文单元」——两者在统计分桶里落点完全不同。
    """
    try:
        b = float(boosted)
        u = float(units)
    except (TypeError, ValueError):
        return None
    if u <= 0:
        return None
    return round(b / u, 4)


def _judge_density(row: dict) -> tuple[str, str, list[str]]:
    """「排序加成」行必须带 `signal_boosted` / `context_units` / `density`。

    ## 为什么要这条

    A 的 P2 第一轮真跑实测：cooc 加成的成败完全由密度 `boosted / units` 决定，
    与题型无关。密度 0–1 救 2、1–2 救 3、2–4 救 1、**4+ 的 35 题一分没挣到**；
    `direct_relation` 内部混杂检验同向，所以是密度效应不是题型效应。机制：命中
    给常数加成，命中块数远超上下文容量（units 中位 5）时全体同加同量 → 相对
    顺序不变 → 等于没加。与 `hop_line` 的「20/20 全抬」是同一条规律。

    报告里没有这个数，下一轮调参的人就还会拿「R@20 涨了」当成功。

    ## 为什么 `context_units` 是必填而不是「上层补算」

    `units` 只有 `build_context()` 之后才知道，`search()` 内部算不出来。所以这个
    字段**只能**在上层（`ask_once` / pipeline）填，`search().stats` 里不该有它。
    把它当成可选字段，就会有人塞进 `search()` 造出时有时无的值。
    """
    if not _has_any(row, BOOST_KEYS):
        return PASS, "不含排序加成键，密度检查不适用", []
    missing = [name for name in DENSITY_FIELDS if _blank(row.get(name))]
    if missing:
        return UNKNOWN, (
            "本行报了排序加成，却缺 " + "、".join(missing) + "。"
            "实测反例：cooc 加成在密度 `boosted/units` 4+ 的 35 道题上净增益为 0"
            "（常数加成在命中块数远超上下文容量时互相抵消），所以不报密度就没法"
            "判断这次「涨了」是不是只涨在低密度题上"
        ), missing
    expected = density_of(row.get("signal_boosted"), row.get("context_units"))
    if expected is None:
        return FAIL, (
            f"context_units={row.get('context_units')!r} 无法当分母（密度定义是 "
            "signal_boosted / context_units，分母必须 > 0）；没有上下文单元的行"
            "不该出现在指标里"
        ), []
    reported = density_of(row.get("density"), 1)
    if reported is None or abs(reported - expected) > 1e-6:
        return FAIL, (
            f"density={row.get('density')!r} 与 "
            f"signal_boosted/context_units={expected} 不一致——"
            "派生量要能被反查，手填的漂亮数字不算"
        ), []
    return PASS, f"密度三件套齐全，density={expected}", []


# ---------------------------------------------------------------- 耗时：墙钟 vs CPU


def _judge_duration(row: dict) -> tuple[str, str, list[str]]:
    """只报墙钟耗时的行判 unknown：那个数**可能是**真的，只是无法证实。

    ## 实测反例（2026-11-05）

    `py -3 scripts/regression.py tests -q` 报 24141.60s（6:42:21），而该进程 CPU
    累计只有 ~778s（13 分钟）——机器中途休眠了 6 小时。P0 的主题正是「有个数字
    不等于那个数字是测量的」，这段墙钟若进性能报告，会把一次 13 分钟的回归说成
    6 小时 42 分。

    ## 为什么 unknown 而不是 fail

    报告里能看到这个数字、也看不到 CPU 时间，这是「查不了」。判 fail 会暗示
    「这个数一定是假的」，而我们并不知道——CLI 在插着电的机器上跑，它就是真的。

    ## 为什么单请求延迟不算

    `p95_ms` / `median_ms` / `qps` 天生是墙钟，且不会被休眠污染（睡眠期间没有
    请求在跑）。把这类行也要求配 CPU 时间就是误伤，误伤会教人忽略这道门。
    """
    if not _has_any(row, WALL_CLOCK_KEYS):
        return PASS, "不含累计耗时键，墙钟/CPU 检查不适用", []
    has_cpu = _has_any(row, CPU_KEYS)
    if not has_cpu:
        return UNKNOWN, (
            "本行只报了墙钟耗时，没有 CPU 时间。实测反例：一次全量回归报 "
            "24141.60s，而进程 CPU 累计只有 ~778s——机器中途休眠了 6 小时，"
            "墙钟被休眠污染。请补 " + str(list(CPU_KEYS)) + " 中任一项"
        ), ["cpu_s"]
    wall = row.get([k for k in WALL_CLOCK_KEYS if not _blank(row.get(k))][0])
    cpu = row.get([k for k in CPU_KEYS if not _blank(row.get(k))][0])
    try:
        w = float(wall)
        c = float(cpu)
    except (TypeError, ValueError):
        return FAIL, f"耗时不是数：wall={wall!r} cpu={cpu!r}", []
    if c > w + 1.0:
        return FAIL, (
            f"CPU 时间 {c} 大于墙钟 {w}——这在单进程里不可能，说明两个数不是同一次"
            "运行采到的（多进程会超 1 倍以上，那时请用 process 级累计并注明）"
        ), []
    return PASS, f"墙钟 {w}s / CPU {c}s", []


# ---------------------------------------------------------------- 引用覆盖的三个分母


def _coverage_denominators(row: dict) -> dict | None:
    """从一行里取出三个分母。三种取法都认，取到一种就够：

    * `denominators: {mixed: 300, reachable: 194, ungated: 72}`
    * `coverage_denominator: {mixed_n: 300, ...}`
    * 平铺的 `coverage_denominators: {mixed: 300, ...}`

    取不到返回 `None`（调用方据此判 unknown，而不是猜一个分母）。
    """
    for key in ("denominators", "coverage_denominators", "coverage_denominator"):
        block = row.get(key)
        if isinstance(block, dict):
            got = {}
            for name in COVERAGE_DENOMINATORS:
                value = block.get(name, block.get(f"{name}_n"))
                if not _blank(value):
                    got[name] = value
            if got:
                return got
    return None


def _judge_coverage_denominators(row: dict) -> tuple[str, str, list[str]]:
    """「引用覆盖」行必须并列三个分母，并且题数必须单调不增。

    ## 为什么要三个

    同一个「引用覆盖 0.17」有三种分母，三者回答不同的问题：
    `/300` 混合了三种互不相干的失败、`/194` 只算断言层可达、`/72` 只算两道上游
    闸都过。只报一个数，读者无法判断这次涨落发生在抽取层、检索层还是回答层——
    而这正是 P3 唯一需要的信息。

    ## 为什么判 unknown 而不是 fail

    报告里只有 `coverage: 0.17` 一行时，我们**不知道**作者的分子分母是什么，
    可能他内部算了三个却没写出来，也可能只有一个。这是「查不了」。

    ## 为什么要求单调不增

    窄分母的题集是宽分母题集的子集（`ungated ⊆ reachable ⊆ mixed`），所以题数
    必须单调不增。反过来说明三个分母不是同一批数据的嵌套口径，而是三份不同的
    东西——那种情况下三个数并列也仍然没有可比性。
    """
    if not _has_any(row, COVERAGE_KEYS):
        return PASS, "不含引用覆盖键，三分母检查不适用", []
    denoms = _coverage_denominators(row)
    if not denoms:
        return UNKNOWN, (
            "本行报了引用覆盖，却没有并列三个分母。实测反例：同一个「0.17」可以是 "
            "51/300（混三种失败）、51/194（断言层可达）、51/72（两闸都过）——"
            "三者答的是不同的问题，只报一个数就分不清涨落发生在哪一层。"
            f"请补 denominators 里的 {list(COVERAGE_DENOMINATORS)}"
        ), ["denominators"]
    missing = [name for name in COVERAGE_DENOMINATORS if name not in denoms]
    if missing:
        return UNKNOWN, (
            "三个分母只给了 " + str(sorted(denoms)) + "，缺 " + "、".join(missing) +
            "。三者必须同时出现——少给最窄的那个，报告就会把「上游没抽到」"
            "当成「P3 没做好」"
        ), [f"denominators.{name}" for name in missing]
    try:
        values = {name: int(denoms[name]) for name in COVERAGE_DENOMINATORS}
    except (TypeError, ValueError):
        return FAIL, f"分母不是整数：{denoms!r}", []
    if values["mixed"] <= 0:
        return FAIL, f"mixed 分母 {values['mixed']} 不是正数", []
    if not (values["mixed"] >= values["reachable"] >= values["ungated"] > 0):
        return FAIL, (
            f"三个分母不是单调不增：{values}。窄分母的题集必须是宽分母题集的子集"
            "（ungated ⊆ reachable ⊆ mixed），不满足说明三个数不是同一批数据的"
            "嵌套口径，并列也没有可比性"
        ), []
    return PASS, (f"三个分母齐全 {values}，P3 盯 {COVERAGE_TARGET}"), []


# ---------------------------------------------------------------- 洗牌判据


def shuffle_ratio(net_gain, changed) -> float | None:
    """`真跑净增益 / top-K 变动集合大小`。分母为 0 或非数 → `None`。

    这个数回答的是「动了多少名次，换回来多少结果」。它天然是一个**淘汰**判据：
    低于门槛说明这一版名次变动主要是噪声，不配当下一版的预测依据。
    """
    try:
        g = float(net_gain)
        c = float(changed)
    except (TypeError, ValueError):
        return None
    if c <= 0:
        return None
    return round(g / c, 4)


def _judge_shuffle(row: dict) -> tuple[str, str, list[str]]:
    """同时报了「净增益」与「变动集合大小」的行，必须给转换率并声明用途。"""
    if not (_has_any(row, SHUFFLE_GAIN_KEYS) and _has_any(row, SHUFFLE_CHANGED_KEYS)):
        return PASS, "不同时含净增益与变动集合，洗牌判据不适用", []
    gain = row.get([k for k in SHUFFLE_GAIN_KEYS if not _blank(row.get(k))][0])
    changed = row.get([k for k in SHUFFLE_CHANGED_KEYS if not _blank(row.get(k))][0])
    ratio = shuffle_ratio(gain, changed)
    if ratio is None:
        return FAIL, f"净增益 {gain!r} / 变动集合 {changed!r} 算不出转换率（变动集合必须 > 0）", []
    reported = row.get("shuffle_ratio")
    if _blank(reported):
        return UNKNOWN, (
            f"报了净增益 {gain} 与变动集合 {changed}，却没有给转换率。"
            f"实算转换率 = {ratio}（阈值 {SHUFFLE_THRESHOLD}）："
            "不写这个数，读者会把「名次动了」读成「效果好了」"
        ), ["shuffle_ratio"]
    if abs(float(reported) - ratio) > 1e-4:
        return FAIL, f"shuffle_ratio={reported} 与 {gain}/{changed}={ratio} 不一致", []
    usage = str(row.get("shuffle_usage") or "")
    if ratio < SHUFFLE_THRESHOLD and usage != SHUFFLE_USAGE_OK:
        return FAIL, (
            f"转换率 {ratio} 低于 {SHUFFLE_THRESHOLD}，判为洗牌；"
            f"这一版只能当淘汰用（shuffle_usage={SHUFFLE_USAGE_OK}），"
            "不能写成「下一版会更好」的依据"
        ), []
    if ratio < SHUFFLE_THRESHOLD:
        return PASS, f"转换率 {ratio} 低于阈值，已声明只作淘汰用", []
    return PASS, f"转换率 {ratio} 达到阈值", []


# ---------------------------------------------------------------- 第 5 名断崖


def _judge_rank5(row: dict) -> tuple[str, str, list[str]]:
    """报「名次进前 5」的行必须附第 5 名转化率。

    实测（dev 350，两份独立逐题产物交叉验证逐位相同）：名次 1–4 的转化率是
    **1.0000**（181/181），第 5 名 **5/12 = 0.4167**，第 6 名 1/18，第 7–20 名 0/44。
    断崖落在第 4 与第 5 名之间。所以「R@5 = 0.79」不意味着「进了前 5 就能用」，
    它里面有一档只有四成能进上下文。
    """
    if not _has_any(row, RANK5_KEYS):
        return PASS, "不含前 5 名指标，断崖提示不适用", []
    value = row.get("rank5_conversion")
    if _blank(value):
        return UNKNOWN, (
            f"本行报了前 5 名内的指标，却没有附第 5 名转化率（应为 {RANK5_CONVERSION}）。"
            "实测：名次 1–4 转化率 1.0000，第 5 名只有 5/12=0.4167，断崖在第 4 与"
            "第 5 名之间。不附这个数，读者会把「进了前 5」读成「能用」"
        ), ["rank5_conversion"]
    try:
        got = float(value)
    except (TypeError, ValueError):
        return FAIL, f"rank5_conversion={value!r} 不是数", []
    if abs(got - RANK5_CONVERSION) > 5e-4:
        return FAIL, (
            f"rank5_conversion={got} 与实测 {RANK5_CONVERSION} 不符——"
            "这个数是实测出来的，不是估计值；改它之前请附新的逐名次实测"
        ), []
    return PASS, f"已附第 5 名转化率 {RANK5_CONVERSION}", []


def verify_metric_row(row: dict) -> dict:
    """检查**一行指标**是否合规。

    分两层：

    **A. 与 recall 无关的判据**（出现相应键就查，不看有没有 recall）：

    1. 排序加成 → 必须带 `signal_boosted` / `context_units` / `density`；
    2. 累计耗时 → 必须另有 CPU 时间（墙钟会被机器休眠污染）；
    3. 引用覆盖 → 必须并列三个分母（`/300` 混合、`/194` 断言层可达、`/72`
       两闸都过），且题数单调不增；
    4. 同时有净增益与 top-K 变动集合 → 必须给转换率，低于 0.1 只能声明淘汰用；
    5. 报前 5 名内的指标 → 必须附第 5 名转化率（实测 0.4167）。

    **B. 与 recall 有关的判据**（`r5`/`r20`/`recall` 等键出现时）：

    6. 带 `caliber`（缺 → unknown，作废值 → fail）；
    7. 并列至少一个下游量（缺 → fail）。

    第 7 条的理由是实测反例：`hop0.5` 配置 R@20 = 0.7379（六配置里第二高），
    但「真进上下文题数」净增益 0、引用覆盖 51→48。只看 R@20 的报告会建议上这
    个配置。**R@n 单独一行时，它不构成「更好」的证据。**

    判 fail 还是 unknown 的分界：报告里明明能看到这一行、也能看到缺什么 ——
    那是**看得出来的缺口**，判 fail；看不出作者到底算了什么 —— 那是「查不了」，
    判 unknown。两者都不算通过。
    """
    if not isinstance(row, dict):
        return {"verdict": FAIL, "detail": f"指标行不是 dict（{type(row).__name__}）",
                "missing": [], "caliber": None, "has_recall": False, "has_downstream": False}
    has_recall = _has_any(row, RECALL_KEYS)
    has_down = _has_any(row, DOWNSTREAM_KEYS)

    # 这五条判据**都不依赖 recall**：一行可以只报加成、只报覆盖、只报耗时。
    # 放在 recall 的「不适用就直接放行」分支之前，否则这类行永远不被检查。
    extras = [
        _judge_density(row),
        _judge_duration(row),
        _judge_coverage_denominators(row),
        _judge_shuffle(row),
        _judge_rank5(row),
    ]

    if not has_recall:
        verdict = _fold([p[0] for p in extras])
        details = [p[1] for p in extras if p[0] != PASS]
        missing = [name for p in extras for name in p[2]]
        return {"verdict": verdict,
                "detail": ("；".join(details) if details else "不含 recall 键，不适用口径检查"),
                "missing": missing, "caliber": row.get("caliber"),
                "has_recall": False, "has_downstream": has_down}
    caliber_verdict, caliber_detail = judge_caliber(row.get("caliber"))
    if caliber_verdict != PASS:
        # `missing` 与 `verify()` 同形：判不了的时候必须能反查到缺的是哪一项。
        # P1-c 那次的教训是「缺输入」安静地变成了「无法判定」，两种输出要分得开。
        extra_detail = "；".join(p[1] for p in extras if p[0] != PASS)
        extra_missing = [name for p in extras for name in p[2]]
        return {"verdict": _fold([caliber_verdict] + [p[0] for p in extras]),
                "detail": (f"{caliber_detail}；本行含 recall 键，口径是硬要求"
                           + (f"；{extra_detail}" if extra_detail else "")),
                "missing": (["caliber"] if caliber_verdict == UNKNOWN else []) + extra_missing,
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
            "missing": ["downstream"] + [name for p in extras for name in p[2]],
            "caliber": row.get("caliber"),
            "has_recall": True,
            "has_downstream": False,
        }
    extra_detail = "；".join(p[1] for p in extras if p[0] != PASS)
    return {"verdict": _fold([PASS] + [p[0] for p in extras]),
            "detail": (f"{caliber_detail}，且并列了下游量"
                       + (f"；{extra_detail}" if extra_detail else "")),
            "missing": [name for p in extras for name in p[2]],
            "caliber": row.get("caliber"),
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


# ---------------------------------------------------------------- 密度三件套的构造器


def signal_density_block(search_stats, context_stats) -> dict:
    """从一次问答的两段 stats 拼出 `{signal_boosted, context_units, density}`。

    ## 为什么必须由「上层」来填

    `signal_boosted`（这题命中给加成的块数）在 `search()` 内部就知道，但
    `context_units`（真正进了上下文的单元数）只有 `build_context()` 之后才知道。
    所以这两个数的比值**不可能**在 `search()` 里算出来——硬塞进去会造出一个
    「有时有、有时是 0」的字段，而 0 会被读成「密度真的是零」。

    调用方（`ask_once` 的调用者 / 评测脚本）拿 `result["search"]["stats"]` 与
    `result["context"]["stats"]` 各一份喂进来即可。

    ## 为什么加成数取 max 而不是求和

    `search().stats` 里每个信号各自一个计数（`cooc_boosted` / `speaker_boosted`
    / ...）。加成的**效果**是这些加量叠加进同一个 score，所以「这题有多少块被
    加成过」取最大值而不是求和——同一块被两个信号同时加成，仍只是一块。
    求和会把「两块各被两路加成」算成 4，密度虚高一倍，正好把 4+ 无效区间填满。

    取不到任何加成计数时返回 `signal_boosted=0`：这是真实的「这次没有加成」，
    不是「不知道」。`density_of(0, n)` 给 0.0，落在「0–1 无效」区间的左端，
    是诚实的位置。
    """
    counts = []
    if isinstance(search_stats, dict):
        for key in BOOST_KEYS:
            value = search_stats.get(key)
            if _blank(value):
                continue
            try:
                counts.append(int(value))
            except (TypeError, ValueError):
                continue
    boosted = max(counts) if counts else 0
    units = 0
    if isinstance(context_stats, dict):
        try:
            units = int(context_stats.get("units") or 0)
        except (TypeError, ValueError):
            units = 0
    return {
        "signal_boosted": boosted,
        "context_units": units,
        "density": density_of(boosted, units),
    }


# ---------------------------------------------------------------- 并列召回指标的包含关系


def compute_gain_set_relations(rows) -> list[dict]:
    """两两算增益题集的交集/差集。**这是「多个召回指标并列」时唯一诚实的写法。**

    返回 `[{a, b, intersection_n, only_a, only_b, nested}, ...]`，`a`/`b` 取每行的
    `metric` 名（没有就用 `row_<i>`）。`nested` 表示其中一个是另一个的子集。
    """
    named = []
    for i, row in enumerate(rows or []):
        if not isinstance(row, dict):
            continue
        ids = row.get("gain_ids")
        if not isinstance(ids, (list, tuple)) or not ids:
            continue
        name = str(row.get("metric") or f"row_{i}")
        named.append((name, set(str(x) for x in ids)))
    out = []
    for i in range(len(named)):
        for j in range(i + 1, len(named)):
            na, sa = named[i]
            nb, sb = named[j]
            out.append({
                "a": na,
                "b": nb,
                "intersection_n": len(sa & sb),
                "only_a": sorted(sa - sb),
                "only_b": sorted(sb - sa),
                "nested": (sa <= sb) or (sb <= sa),
            })
    return out


def verify_gain_sets(rows) -> dict:
    """并列的召回指标必须给出增益题集的包含关系。

    ## 真实案例（2026-11-05）

    A 的 P2 第一轮报 `any_gold +6` / `all_gold +4` / `covered +5`。这不是三份独立
    证据：`required_citations` 的块集恒等于 `gold_chunk_ids`（gold-v2 640 题、
    gold-v1 320 题均 0 例外），所以「所有 required 块在上下文」等价于「所有 gold
    块在上下文」，`covered` 只比 `all_gold` 多一层「回答层发不发引用」；单 gold
    块题占 217/350 = 62.0%，其上 `any_gold` 与 `all_gold` **数学恒等**。

    三个增益题集的实际差集只有一题（`q-tc-000023`）。

    只并列三个 +N，读者会以为三个机制同时改善 —— 与 `hop0.5` 的 R@20 虚高同类：
    **数字真，但读者会得出错误结论**。

    ## 判据

    * 少于两个带 `gain_ids` 的行 → 不适用（pass）；
    * 有 ≥2 行但没有任何一行给 `relation_table` → unknown（不是 fail：报告可能
      只是没写，我们不知道它到底是不是嵌套）；
    * 给了 `relation_table` 但与实际集合算出来的不一致 → fail（手填糊弄）；
    * 一致 → pass，并在 `nested_pairs` 里点名哪几对是嵌套关系。
    """
    if not isinstance(rows, (list, tuple)):
        return {"verdict": FAIL, "applicable": False, "nested_pairs": [],
                "detail": f"指标集合应为 list，拿到 {type(rows).__name__}"}
    gain_rows = [
        r for r in rows
        if isinstance(r, dict) and isinstance(r.get("gain_ids"), (list, tuple)) and r.get("gain_ids")
    ]
    if len(gain_rows) < 2:
        return {"verdict": PASS, "applicable": False, "nested_pairs": [],
                "detail": f"只有 {len(gain_rows)} 行带增益题集，不涉及并列"}
    expected = compute_gain_set_relations(gain_rows)
    nested = [f"{e['a']} ⊇ {e['b']}" if not e["only_b"] else f"{e['b']} ⊇ {e['a']}"
              for e in expected if e["nested"]]
    provided = None
    for row in gain_rows:
        table = row.get("relation_table")
        if isinstance(table, list) and table:
            provided = table
            break
    if provided is None:
        return {"verdict": UNKNOWN, "applicable": True, "nested_pairs": nested,
                "detail": (
                    "并列了 " + str(len(gain_rows)) + " 个召回类指标，却没有给增益题集的"
                    "包含关系。真实案例：any_gold +6 / all_gold +4 / covered +5 三者"
                    "实际差集只有一题（q-tc-000023）——required_citations 的块集恒等于"
                    "gold_chunk_ids，这个嵌套是结构性的，不是三个机制同时改善。"
                    "请用 compute_gain_set_relations() 生成 relation_table"
                )}
    mismatches = []
    for exp in expected:
        got = None
        for item in provided:
            if not isinstance(item, dict):
                continue
            if {str(item.get("a")), str(item.get("b"))} == {exp["a"], exp["b"]}:
                got = item
                break
        if got is None:
            mismatches.append(f"{exp['a']}×{exp['b']} 缺失")
            continue
        if int(got.get("intersection_n", -1)) != exp["intersection_n"]:
            mismatches.append(
                f"{exp['a']}×{exp['b']} 交集应为 {exp['intersection_n']}，表里写 {got.get('intersection_n')}"
            )
        if sorted(str(x) for x in got.get("only_a") or []) != exp["only_a"]:
            mismatches.append(f"{exp['a']}×{exp['b']} only_a 与实测不符")
        if sorted(str(x) for x in got.get("only_b") or []) != exp["only_b"]:
            mismatches.append(f"{exp['a']}×{exp['b']} only_b 与实测不符")
    if mismatches:
        return {"verdict": FAIL, "applicable": True, "nested_pairs": nested,
                "detail": "relation_table 与实际增益集不符：" + "；".join(mismatches)}
    return {"verdict": PASS, "applicable": True, "nested_pairs": nested,
            "detail": f"并列 {len(gain_rows)} 个召回指标，包含关系已给出"
                      + (f"（嵌套：{'、'.join(nested)}）" if nested else "")}


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
