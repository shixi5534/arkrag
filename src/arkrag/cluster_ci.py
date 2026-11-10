"""W4b：聚类 bootstrap 与「冻结模板簇」的簇键 —— 纯函数，不读任何产物。

## 这个模块为什么必须存在

`scripts/eval_answer_quality.py` 现在只有点估计：`pass / 分母`。历史上所有 CI 都是
**按题 iid 重抽**的配对 bootstrap（`scripts/audit_round6.py::paired_bootstrap` 等）。
计划书 §W4「评测口径同时升级」指出这条路的前提是假的：

> 实测 `dev × holdout` 3-gram Jaccard ≥0.6 的近似重复 **1,802 对**
> （alias 739 / cross_chunk 493 / multihop 322），题面是模板生成的。
> 按 `m̄≈2.5, ρ≈0.6` 估，**方差膨胀 ≈1.9，CI 宽度膨胀 ≈1.4**

同一模板生成的一批题在判据上高度相关（答案句式一样、判据分支一样），把 285 道题
当 285 份独立证据会让 CI 偏窄。所以要**按簇整簇重抽**。

## 本模块不做的事（守红线）

* **不重算那 1,802 对。** 那要读 gold（1 GB 级产物）且是已实测的事实，重算只会让
  这个模块变成第二个"读产物才能跑"的地方。本模块只实现**簇键计算函数**，
  对当前这次评测的题面算簇，并单测它（相同模板归同簇、不同模板分簇）。
* **不假设方差膨胀。** `m̄≈2.5, ρ≈0.6` 是**设计量级估算**，计划书明令
  「不许用假设值冒充实测」。`variance_inflation()` 只在**真有两臂配对样本**时
  才给数，否则返回 `estimable=False` + 原因。
* **不做任何 I/O。** 没有 `load_generation()`、没有文件读、没有环境变量。

## 三个概念必须分开（本项目栽过四次"分母口径不同"的坑）

| 量 | 定义 | 为什么不能互相替代 |
|---|---|---|
| `N` | 本次评测的题数 | —— |
| `unique_questions` | 题面归一后去重 | gold 相同不等于模板相同；同一模板下答案不同 |
| `n_clusters` | 3-gram Jaccard ≥ 0.6 连通分量个数 | 近似重复不是完全相同：去重键看不见"改几个字的同模板题" |

去重键（`unique_questions`）与簇键（`n_clusters`）**分别定义、分别报**，
不许用一个数替另一个。
"""
from __future__ import annotations

import math
import random
import statistics
import unicodedata

#: 3-gram：计划书给的是 3-gram Jaccard ≥0.6（实测 1,802 对）。
DEFAULT_NGRAM = 3

#: 近似重复阈值。与计划书同一口径，**不是**本模块调出来的。
DEFAULT_JACCARD = 0.6

#: bootstrap 重抽次数。与仓库里既有的 `paired_bootstrap(n=20000)` 相比更小是
#: 有意的：这里要跑「逐题 + 整簇」两套、每份 split 两臂，20000 次会让一个
#: 评测报告多等好几分钟，而分位点估计在 2000 次时已稳定到 3 位小数。
DEFAULT_REPS = 2000

#: 固定种子：报告里的 CI 必须能逐位复现，否则"这次 8.1 那次 8.3"无从对账。
SEED = 20261110


# --------------------------------------------------------------- 题面归一与 3-gram
def norm_question(text: object) -> str:
    """题面归一：NFKC + 去掉所有空白。

    只去空白、**不去标点**：判据（`judge.anchors`）看的是 `「」` 与字面子串，
    把标点抹掉会让"同模板"与"不同模板"的边界漂移。全角/半角由 NFKC 折平。
    """
    s = unicodedata.normalize("NFKC", str(text or ""))
    return "".join(ch for ch in s if not ch.isspace())


def ngrams(text: object, n: int = DEFAULT_NGRAM) -> frozenset[str]:
    """归一后取字符 n-gram 集合。短于 `n` 的串返回 `{整串}`（不返回空集）。"""
    s = norm_question(text)
    if not s:
        return frozenset()
    if len(s) <= n:
        return frozenset({s})
    return frozenset(s[i:i + n] for i in range(len(s) - n + 1))


def question_jaccard(a: object, b: object, n: int = DEFAULT_NGRAM) -> float:
    """两道题的 n-gram Jaccard。任一方为空集时返回 0.0（不许当 1）。"""
    sa, sb = ngrams(a, n), ngrams(b, n)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    return inter / len(sa | sb)


# --------------------------------------------------------------------- 簇键
def near_duplicate_pairs(texts: list, *, n: int = DEFAULT_NGRAM,
                         threshold: float = DEFAULT_JACCARD) -> list[tuple[int, int, float]]:
    """所有 Jaccard ≥ `threshold` 的下标对，按 (i, j) 升序。

    返回相似度而不是只返回布尔：报告里要能说"这两道多像"，
    而阈值恰好相等时的误判只有相似度能解释。
    """
    out: list[tuple[int, int, float]] = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = question_jaccard(texts[i], texts[j], n)
            if sim >= threshold:
                out.append((i, j, sim))
    return out


def cluster_keys(texts: list, *, n: int = DEFAULT_NGRAM,
                 threshold: float = DEFAULT_JACCARD) -> list[int]:
    """**簇键**：按题面近似重复把题聚成簇，返回每题的簇号。

    定义（可单测的纯函数）：把每个下标当结点，`Jaccard ≥ threshold` 的对连边，
    取**连通分量**；簇号 = 该分量里最小下标（决定性的，与输入顺序无关的稳定性
    只保证"同输入同输出"）。

    为什么用连通分量而不是"每个模板一个代表"：模板复发会成链（A≈B、B≈C 而
    A≉C），单链接连通分量是聚类里唯一不需要额外参数的取法，也不会因为
    代表元选谁而改簇内成员。

    复杂度 O(k² · n-gram 数)：k=350 时约 6 万次比对，毫秒级。
    """
    parent = list(range(len(texts)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # 小的当根：让簇号等于簇内最小下标，报告里的簇号才稳定可读
            if ra > rb:
                ra, rb = rb, ra
            parent[rb] = ra

    for i, j, _sim in near_duplicate_pairs(texts, n=n, threshold=threshold):
        union(i, j)
    return [find(i) for i in range(len(texts))]


def cluster_sizes(keys: list[int]) -> list[int]:
    """每个簇的题数，按簇号升序。"""
    counts: dict[int, int] = {}
    for k in keys:
        counts[int(k)] = counts.get(int(k), 0) + 1
    return [counts[k] for k in sorted(counts)]


def kish_effective_n(sizes: list[int]) -> float:
    """Kish 有效簇数 `(Σn)² / Σn²`。

    簇大小不齐时，"有几个簇"会高估信息量：一个 40 题的簇与一个 1 题的簇
    各算一票是不对的。Kish 给出的是**有效簇数**，不是有效样本量——
    有效样本量需要实测方差膨胀，见 `effective_sample_size()`。
    """
    total = sum(int(s) for s in sizes)
    sq = sum(int(s) * int(s) for s in sizes)
    if total <= 0 or sq <= 0:
        return 0.0
    return (total * total) / sq


def unique_question_count(texts: list) -> int:
    """题面归一后去重的唯一题数（**去重键**，与簇键不是一回事）。"""
    return len({norm_question(t) for t in texts})


def cluster_report(texts: list, *, n: int = DEFAULT_NGRAM,
                   threshold: float = DEFAULT_JACCARD) -> dict:
    """簇结构的一整块读数：题数 / 唯一题数 / 簇数 / 簇大小 / 有效簇数。"""
    keys = cluster_keys(texts, n=n, threshold=threshold)
    sizes = cluster_sizes(keys)
    return {
        "n_items": len(texts),
        "unique_questions": unique_question_count(texts),
        "n_clusters": len(sizes),
        "cluster_sizes": sizes,
        "effective_clusters": round(kish_effective_n(sizes), 4),
        "n_gram": int(n),
        "jaccard_threshold": float(threshold),
    }


# ------------------------------------------------------------------ bootstrap
def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(len(sorted_vals) * q)
    return sorted_vals[min(len(sorted_vals) - 1, max(0, idx))]


def _summarise(samples: list[float], observed: float, *, reps: int,
               note: str) -> dict:
    samples = sorted(samples)
    var = statistics.pvariance(samples) if len(samples) > 1 else 0.0
    return {
        "mean": observed,
        "ci_low": _percentile(samples, 0.025),
        "ci_high": _percentile(samples, 0.975),
        "var": var,
        "reps": reps,
        "degenerate": bool(var == 0.0),
        "note": note,
    }


def iid_bootstrap_mean(values: list[float], *, reps: int = DEFAULT_REPS,
                       seed: int = SEED) -> dict:
    """**保留的既有口径**：按题有放回重抽（假设题与题独立）。"""
    vals = [float(v) for v in values]
    if not vals:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "var": 0.0,
                "reps": 0, "degenerate": True, "note": "no_observations"}
    rng = random.Random(seed)
    k = len(vals)
    observed = sum(vals) / k
    samples = [sum(vals[rng.randrange(k)] for _ in range(k)) / k
               for _ in range(reps)]
    return _summarise(samples, observed, reps=reps, note="iid_resample_questions")


def cluster_bootstrap_mean(values: list[float], keys: list[int],
                           *, reps: int = DEFAULT_REPS,
                           seed: int = SEED) -> dict:
    """**新增口径**：按簇整簇有放回重抽（一个簇要么全进、要么全不进）。

    `values[i]` 与 `keys[i]` 必须一一对应（同一题的观测与簇号）。
    重抽后有放回地拼接各簇的值，再算均值——簇内的相关性因此被保留，
    这是 cluster bootstrap 与 iid bootstrap 唯一的区别。

    三种退化情形都必须显式标出来，不许安静地给一个好看的区间：
    * 没有簇（`values` 空）；
    * 所有簇的均值相同 → 方差 0，CI 退化为一个点；
    * `reps` 为 0。
    """
    vals = [float(v) for v in values]
    if len(vals) != len(keys):
        raise ValueError(f"values/keys 长度不一致：{len(vals)} vs {len(keys)}")
    if not vals:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "var": 0.0,
                "reps": 0, "degenerate": True, "n_clusters": 0,
                "note": "no_observations"}
    members: dict[int, list[float]] = {}
    for v, k in zip(vals, keys):
        members.setdefault(int(k), []).append(v)
    cluster_ids = sorted(members)
    observed = sum(vals) / len(vals)
    if reps <= 0 or len(cluster_ids) < 2:
        return {"mean": observed, "ci_low": observed, "ci_high": observed,
                "var": 0.0, "reps": 0, "degenerate": True,
                "n_clusters": len(cluster_ids),
                "note": "single_cluster_or_zero_reps"}
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(reps):
        pool: list[float] = []
        for _ in range(len(cluster_ids)):
            pool.extend(members[cluster_ids[rng.randrange(len(cluster_ids))]])
        samples.append(sum(pool) / len(pool))
    out = _summarise(samples, observed, reps=reps, note="cluster_resample")
    out["n_clusters"] = len(cluster_ids)
    return out


def paired_diffs(before: list[float], after: list[float]) -> list[int]:
    """逐题配对差 `after - before`（两臂必须同序、同长）。"""
    if len(before) != len(after):
        raise ValueError(f"配对向量的长度不一致：{len(before)} vs {len(after)}")
    return [int(b) - int(a) for a, b in zip(before, after)]


def variance_inflation(iid: dict, cluster: dict) -> dict:
    """方差膨胀系数 = 簇口径 bootstrap 方差 / iid 口径 bootstrap 方差。

    **只在真有配对样本、且两套方差都算得出来时才给数**：`iid["var"] == 0`
    （例如全部题同分）时比值无定义，返回 `estimable=False`；簇数 < 2 同理。
    计划书明令不许用 `m̄≈2.5, ρ≈0.6` 反推的假设值（≈1.9）冒充实测，
    所以这里**没有**任何默认值。
    """
    vi = float(iid.get("var") or 0.0)
    vc = float(cluster.get("var") or 0.0)
    n_clusters = int(cluster.get("n_clusters") or 0)
    if iid.get("reps", 0) == 0 or cluster.get("reps", 0) == 0:
        return {"estimable": False, "value": None,
                "reason": "某一套 bootstrap 没有样本（无配对样本或单簇）"}
    if n_clusters < 2:
        return {"estimable": False, "value": None,
                "reason": f"簇数 {n_clusters} < 2，整簇重抽没有自由度"}
    if vi <= 0.0:
        return {"estimable": False, "value": None,
                "reason": "iid 方差为 0（逐题差值全相同），比值无定义"}
    return {"estimable": True, "value": vc / vi,
            "reason": f"簇方差 {vc:.6g} / iid 方差 {vi:.6g}"}


def effective_sample_size(n_items: int, inflation: dict) -> dict:
    """有效样本量 = N / 方差膨胀。膨胀不可估计时**不给数**。"""
    n = int(n_items)
    if not inflation.get("estimable"):
        return {"estimable": False, "value": None,
                "reason": "方差膨胀不可估计，有效样本量同样不可估计"}
    ratio = float(inflation["value"])
    if ratio <= 0 or n <= 0:
        return {"estimable": False, "value": None, "reason": "输入不合法"}
    return {"estimable": True, "value": n / ratio,
            "reason": f"{n} / {ratio:.4f}"}


def one_sided_p(samples_mean_nonpositive: int, reps: int) -> float:
    """单侧 p = P(Δ ≤ 0)。`reps<=0` 时返回 1.0（不假装显著）。"""
    if reps <= 0:
        return 1.0
    return samples_mean_nonpositive / reps


def paired_bootstrap_delta(before: list[float], after: list[float], keys: list[int],
                           *, reps: int = DEFAULT_REPS, seed: int = SEED,
                           clustered: bool = True) -> dict:
    """配对 Δ 的 bootstrap：`clustered=True` 走整簇重抽，否则走 iid 逐题重抽。

    Δ 以**通过率差的百分点**表示（`mean(diffs) * 100`），与项目里其余
    bootstrap 报告同量纲。同时给 `delta_count = Σdiffs`（净增题数）。
    """
    diffs = paired_diffs(before, after)
    if not diffs:
        return {"delta_pp": 0.0, "delta_count": 0, "ci_low_pp": 0.0,
                "ci_high_pp": 0.0, "p_one_sided": 1.0, "reps": 0,
                "note": "no_observations"}
    boot = (cluster_bootstrap_mean(diffs, keys, reps=reps, seed=seed)
            if clustered else iid_bootstrap_mean(diffs, reps=reps, seed=seed))
    # 单侧 p 需要自己的抽样序列：均值 ≤ 0 的比例
    rng = random.Random(seed)
    counts_nonpositive = 0
    for _ in range(reps):
        if clustered:
            members: dict[int, list[int]] = {}
            for v, k in zip(diffs, keys):
                members.setdefault(int(k), []).append(v)
            ids = sorted(members)
            pool: list[int] = []
            for _ in range(len(ids)):
                pool.extend(members[ids[rng.randrange(len(ids))]])
            m = sum(pool) / len(pool)
        else:
            k = len(diffs)
            m = sum(diffs[rng.randrange(k)] for _ in range(k)) / k
        if m <= 0:
            counts_nonpositive += 1
    return {
        "delta_pp": boot["mean"] * 100.0,
        "delta_count": sum(diffs),
        "ci_low_pp": boot["ci_low"] * 100.0,
        "ci_high_pp": boot["ci_high"] * 100.0,
        "p_one_sided": one_sided_p(counts_nonpositive, reps),
        "reps": boot["reps"],
        "var": boot["var"],
        "n_clusters": boot.get("n_clusters", 0),
        "degenerate": boot["degenerate"],
        "note": boot["note"],
    }


__all__ = [
    "DEFAULT_JACCARD",
    "DEFAULT_NGRAM",
    "DEFAULT_REPS",
    "SEED",
    "cluster_bootstrap_mean",
    "cluster_keys",
    "cluster_report",
    "cluster_sizes",
    "effective_sample_size",
    "iid_bootstrap_mean",
    "kish_effective_n",
    "near_duplicate_pairs",
    "ngrams",
    "norm_question",
    "one_sided_p",
    "paired_bootstrap_delta",
    "paired_diffs",
    "question_jaccard",
    "unique_question_count",
    "variance_inflation",
]
