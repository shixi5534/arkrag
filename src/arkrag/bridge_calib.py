"""W2.1：桥接 `top1_margin` 的**校准映射**与**代价比闸门**——纯逻辑，不碰数据。

## 这个模块做什么、不做什么

计划书 `docs/计划-W0到W4-检索宽度解耦与决策兑现-20261108.md` 的 §W2.1 把
"敢不敢选 top-1" 拆成三段，本模块只承担后两段：

1. **产样本**（不在本模块，也不允许在本模块里做）：按生产同源候选构造器逐题落盘
   `{question_id, candidate_id, n_cands, support_top1, support_top2, raw_margin,
   tie, gold_is_top1}` → `data/diag_bridge_margin.json`。那一步要读 1 GB 级产物，
   必须经 `scripts/run_guarded.py` 串行跑。
2. **拟合单调映射** `p = P(gold = top1 | 特征)`：`fit()`。输入就是上面那些样本，
   输出一份可以从 JSON 读回来的映射表（`to_json` / `from_json` / `save_mapping` /
   `load_mapping`）。
3. **按代价比决策**：`decide_top1()` / `should_answer()`。

因此本模块**没有任何文件 I/O 的隐式依赖**（`load_mapping()` 只读你显式给的那
一个 JSON），可以纯单测，也不需要任何产物。

## 为什么原始 support 差值不能当概率

计划书要求得很明确："原始差值不能直接当概率"。`raw_margin = support_top1 −
support_top2` 是一个**计数量**，它的量纲与"答对的概率"无关；dev 上 top-1 命中率
87.7% 是这一批候选集的**边际**命中率，不是"margin=2 时命中的概率"。所以代码里
**不存在**任何形如 `raw_margin >= 2 → 作答` 的阈值。唯一的阈值是**概率**阈值，
且它由预登记的业务代价比算出来，不是扫出来的。

## 代价比与阈值（预登记常量，不许改）

```
c_误答 = ANSWER_COST = 2      # 答错：把"不知道"变成"看起来确定的错"
c_误拒 = REFUSE_COST = 1      # 误拒：落回原链路，只是没赚
作答 ⟺ p·c_误拒 > (1 − p)·c_误答  ⟺  p > c_误答 / (c_误答 + c_误拒) = 2/3
```

注意这里是**严格大于**：`p == 2/3` 时期望增益恰为 0，按"拿不到正收益就不动"
的保守口径判**拒答**（与计划书 §W2.1 的 `p ≥ τ*` 只差这一个边界点；两边的
差别在报告里必须写明，本项目不允许把边界口径悄悄改掉）。

实现上，判定只走 `should_answer()` 一条路，它比的是 `p > threshold()`。
**不用** `expected_gain(p) > 0` 当判定条件——两者数学等价但浮点不等价：
`2/3` 不可二进制表示，`expected_gain(2/3)` 算出 `−1.1e-16`，
于是"边界拒答"会变成舍入巧合（把 `>` 改成 `>=` 测试仍是绿的，本次反向实测
抓到过这一笔）。`expected_gain()` 只作报告字段。

## 单调性怎么保证

计划书："使用单调校准（候选分数越高，预测概率不得越低）；样本不足的区间回退拒答"。

* 单调沿 `raw_margin` 递增方向，**在同一层（stratum）内**成立。层的定义由
  `fit(keys=...)` 决定：默认只用 `("raw_margin",)`；如果要把 `n_cands` 也分进去，
  传 `("n_cands", "raw_margin")`，单调性就只在同一个 `n_cands` 内部要求——跨
  `n_cands` 强行拉平是错的（候选越多，同样 margin 越不可信，方向未必单调，
  那是一条**未经测量**的假设）。
* 强制手段是加权保序回归（PAVA，pool-adjacent-violators）。被合并的桶记同一个
  `p`，并带上 `pool` 编号，报告里能看出"哪几个 margin 桶被拉平了"。
* **样本不足的回退没有阈值常数**：桶里一个样本都没有 ⇒ 映射里就没有这个桶 ⇒
  `probability()` 返回 `None` ⇒ `decide_top1()` 拒答。用一个 `n >= k` 的 k 就成了
  又一条"未登记的超参数"，所以这里刻意不设。样本数 `n` 逐桶落盘，由人来看。
"""
from __future__ import annotations

import json
from pathlib import Path

#: 映射表 schema。改字段必须改这里。
SCHEMA = "arkrag.bridge_calib/1"

#: 预登记业务代价（计划书 §W2.1，**不再调整**）。
ANSWER_COST = 2      # c_误答
REFUSE_COST = 1      # c_误拒

#: 计划书允许进入校准的**全部**特征。多一个就是"未登记特征"。
FEATURES = ("n_cands", "support_top1", "support_top2", "raw_margin")

#: 单调约束的方向特征。它必须在 `fit(keys=...)` 里出现。
MONOTONE_IN = "raw_margin"


def threshold() -> float:
    """概率阈值 `τ* = c_误答 / (c_误答 + c_误拒)`（= 2/3，由代价比算出，不硬编码）。"""
    return ANSWER_COST / (ANSWER_COST + REFUSE_COST)


def expected_gain(p: float) -> float:
    """按代价比算"作答"相对"拒答"的期望增益 `p·c_误拒 − (1 − p)·c_误答`。

    **它不参与判定**，只作报告字段与人工核对（敏感性报告要报原始期望增益）。
    判定统一走 `should_answer()`，理由写在那个函数的 docstring 里：
    这个式子在 `p = 2/3` 处因浮点误差给出 `−1.1e-16` 而不是 `0.0`，
    拿它当判定条件会让"边界拒答"变成舍入巧合。
    """
    q = float(p)
    return q * float(REFUSE_COST) - (1.0 - q) * float(ANSWER_COST)


def should_answer(p: float) -> bool:
    """作答闸门：`p > τ*`，`p == τ*` 时**不作答**（边界保守）。

    为什么这里比的是 `p > threshold()` 而不是 `expected_gain(p) > 0`：

    后者在数学上完全等价，但**浮点上不等价**。`τ* = 2/3` 不是二进制可表示的
    小数，`expected_gain(2/3)` 算出来是 `−1.1e-16` 而不是 `0.0`——于是
    "边界拒答"这条语义就变成了"浮点舍入恰好把我救了一下"，把 `>` 改成 `>=`
    测试仍然是绿的（本次反向实测抓到过这一笔）。改成直接比阈值之后，
    `2/3` 与 `threshold()` 是同一个浮点数，运算符的差别立刻可见。

    代价常量依然真的参与决策：`threshold()` 由 `ANSWER_COST` / `REFUSE_COST`
    算出来，改常量会改阈值。`expected_gain()` 保留给敏感性报告与人工核对。
    """
    return float(p) > threshold()


# --------------------------------------------------------------------- 特征
def features_of_ranked(ranked: list[dict]) -> dict:
    """把排好序的候选列表压成计划书登记的四个特征。

    `support_top2` 在只有一个候选时记 0（没有第二名的 support 可言），
    `raw_margin` 因此等于 `support_top1` —— 这是定义，不是拟合。
    """
    if not ranked:
        raise ValueError("features_of_ranked 需要至少一个候选")
    top1 = int(ranked[0]["support_sum"])
    top2 = int(ranked[1]["support_sum"]) if len(ranked) > 1 else 0
    return {
        "n_cands": len(ranked),
        "support_top1": top1,
        "support_top2": top2,
        "raw_margin": top1 - top2,
    }


def bucket_key(features: dict, keys: tuple[str, ...] | list[str]) -> dict:
    """按 `keys` 从特征里取桶键（返回可 JSON 化的 dict，不用 tuple 以免序列化歧义）。"""
    if not keys:
        raise ValueError("bucket_key 至少要一个特征键")
    out: dict[str, int] = {}
    for name in keys:
        if name not in FEATURES:
            raise ValueError(f"未登记的特征 {name!r}；只允许 {FEATURES}")
        if name not in features:
            raise ValueError(f"特征 {name!r} 缺失，无法落桶（fail closed）")
        out[str(name)] = int(features[name])
    if MONOTONE_IN not in out:
        raise ValueError(
            f"桶键必须包含 {MONOTONE_IN!r}：单调性只沿这个方向有定义，"
            "缺了它就没有任何可检验的单调约束")
    return out


# --------------------------------------------------------------------- 拟合
def _check_sample(row: dict, index: int) -> dict:
    """校验一条样本，并把 `raw_margin` 统一成由两个 support 相减得到的值。"""
    if not isinstance(row, dict):
        raise ValueError(f"第 {index} 条样本不是 dict：{type(row).__name__}")
    for name in ("n_cands", "support_top1", "support_top2"):
        if name not in row:
            raise ValueError(f"第 {index} 条样本缺特征 {name!r}（fail closed）")
        try:
            row_int = int(row[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"第 {index} 条样本的 {name!r} 不是整数：{row[name]!r}") from exc
        if row_int < 0:
            raise ValueError(f"第 {index} 条样本的 {name!r} 为负：{row_int}")
    if row.get("gold_is_top1") is None:
        raise ValueError(f"第 {index} 条样本缺标签 gold_is_top1（fail closed）")
    n_cands = int(row["n_cands"])
    if n_cands < 1:
        raise ValueError(f"第 {index} 条样本的 n_cands={n_cands}，至少为 1")
    margin = int(row["support_top1"]) - int(row["support_top2"])
    if "raw_margin" in row and row["raw_margin"] is not None:
        if int(row["raw_margin"]) != margin:
            raise ValueError(
                f"第 {index} 条样本的 raw_margin={row['raw_margin']} 与 "
                f"support_top1−support_top2={margin} 不一致——"
                "特征之间自相矛盾的样本不许进校准")
    if margin < 0:
        raise ValueError(
            f"第 {index} 条样本的 raw_margin={margin} 为负：候选是按 support 降序排的，"
            "top1 的 support 不可能小于 top2——这条样本与生产排序不同源")
    return {
        "n_cands": n_cands,
        "support_top1": int(row["support_top1"]),
        "support_top2": int(row["support_top2"]),
        "raw_margin": margin,
        "gold_is_top1": bool(row["gold_is_top1"]),
        "tie": bool(row.get("tie") or False),
    }


def _pool_stratum(points: list[dict]) -> None:
    """加权 PAVA：沿 margin 升序把违反单调的相邻块合并，`p` 就地写回。

    `points` 必须已按 margin 升序。合并后同一块内的桶拿同一个 `p`，
    并共享 `pool` 编号——报告里能直接看出"哪几个桶被拉平了"。
    """
    blocks: list[dict] = []
    for pt in points:
        blocks.append({"n": pt["n"], "correct": pt["correct"], "members": [pt]})
        while len(blocks) > 1:
            prev, last = blocks[-2], blocks[-1]
            if prev["correct"] * last["n"] <= last["correct"] * prev["n"]:
                break
            blocks[-2:] = [{
                "n": prev["n"] + last["n"],
                "correct": prev["correct"] + last["correct"],
                "members": prev["members"] + last["members"],
            }]
    for pool_id, blk in enumerate(blocks):
        p = blk["correct"] / blk["n"]
        for pt in blk["members"]:
            pt["p"] = p
            pt["pool"] = pool_id


def fit(samples: list[dict], *, keys: tuple[str, ...] = (MONOTONE_IN,)) -> dict:
    """用 `(特征, 是否 gold_top1)` 样本拟合单调映射。

    * `tie=True` 的样本**不参与拟合**（计划书：并列一律拒答，它永远走不到
      "按概率决定"这一步，把它算进频率会把校准偏向"并列题"），但计数落盘。
    * 桶按 `keys` 分组；没有样本的桶**不存在于映射里**，查不到即拒答。
    * 单调性沿 `MONOTONE_IN` 在每个 stratum（`keys` 去掉 margin 后的其余键取值）
      内部用 PAVA 强制。

    返回可 JSON 化的 dict（见 `SCHEMA`）。
    """
    keys = tuple(str(k) for k in keys)
    if MONOTONE_IN not in keys:
        raise ValueError(f"keys 必须包含 {MONOTONE_IN!r}，否则没有可强制的单调方向")
    checked = [_check_sample(row, i) for i, row in enumerate(samples or [])]

    excluded_tie = [row for row in checked if row["tie"]]
    used = [row for row in checked if not row["tie"]]
    if not used:
        raise ValueError("没有任何可用样本（全部被 tie 排除或样本为空），无法校准")

    groups: dict[tuple, dict] = {}
    for row in used:
        key = bucket_key(row, keys)
        sig = tuple(key[name] for name in keys)
        slot = groups.setdefault(sig, {"key": key, "n": 0, "correct": 0,
                                       "margin": int(row[MONOTONE_IN])})
        slot["n"] += 1
        slot["correct"] += int(bool(row["gold_is_top1"]))

    strata: dict[tuple, list[dict]] = {}
    for sig, slot in groups.items():
        key = slot["key"]
        stratum = tuple(int(key[name]) for name in keys if name != MONOTONE_IN)
        strata.setdefault(stratum, []).append(slot)

    buckets: list[dict] = []
    for stratum in sorted(strata):
        pts = sorted(strata[stratum], key=lambda s: s["margin"])
        _pool_stratum(pts)
        for slot in pts:
            buckets.append({
                "key": slot["key"],
                "n": int(slot["n"]),
                "correct": int(slot["correct"]),
                "p": float(slot["p"]),
                "pool": int(slot["pool"]),
            })

    buckets.sort(key=lambda b: tuple(int(b["key"][name]) for name in keys))
    return {
        "schema": SCHEMA,
        "feature_keys": list(keys),
        "monotone_in": MONOTONE_IN,
        "cost": {"answer": ANSWER_COST, "refuse": REFUSE_COST},
        "threshold": threshold(),
        "n_samples": len(used),
        "n_excluded_tie": len(excluded_tie),
        "buckets": buckets,
    }


# --------------------------------------------------------------------- 查询
def _check_mapping(mapping: dict) -> dict:
    if not isinstance(mapping, dict):
        raise ValueError(f"映射必须是 dict，实际 {type(mapping).__name__}")
    if not mapping.get("buckets"):
        raise ValueError("映射里没有 buckets，无法查询")
    cost = mapping.get("cost") or {}
    if cost:
        if int(cost.get("answer", ANSWER_COST)) != ANSWER_COST or \
                int(cost.get("refuse", REFUSE_COST)) != REFUSE_COST:
            raise ValueError(
                f"映射里的代价 {cost} 与预登记常量 "
                f"{{'answer': {ANSWER_COST}, 'refuse': {REFUSE_COST}}} 不一致——"
                "代价比是预登记业务常量，不许被映射文件改写")
    keys = mapping.get("feature_keys")
    if not keys:
        raise ValueError("映射里没有 feature_keys，无法知道按哪些特征落桶")
    return mapping


def probability(mapping: dict, features: dict) -> float | None:
    """查表得到 `P(gold = top1 | 特征)`；桶不存在（样本不足）返回 `None`。

    **返回 None 不是 0**：`0.0` 会说服调用方"这个桶测过、概率是零"，
    而事实是"没样本"。调用方必须把 `None` 当"不可校准 ⇒ 拒答"。
    """
    m = _check_mapping(mapping)
    key = bucket_key(features, tuple(m["feature_keys"]))
    for bucket in m["buckets"]:
        if all(int(bucket["key"].get(name, -1)) == key[name] for name in key):
            return float(bucket["p"])
    return None


# --------------------------------------------------------------------- 决策
def tie_of_top(ranked: list[dict]) -> bool:
    """top-1 与 top-2 的**完整排序键**（除名字外）是否完全相同。

    计划书："遇到影响 top-1 的完全相同排序键必须先标 tie 并拒答"。
    名字只用于稳定展示，**不许**拿字典序把并列伪装成确定的 top-1。
    """
    if len(ranked) < 2:
        return False
    def sig(cand: dict) -> tuple:
        return (int(cand["support_sum"]), int(cand.get("assert_count") or 0),
                int(cand.get("degree") or 0))
    return sig(ranked[0]) == sig(ranked[1])


def decide_top1(mapping: dict | None, ranked: list[dict]) -> dict:
    """按校准概率 + 代价比决定"答不答"，并给出被选中的 top-1。

    返回值里的 `reason` 是**唯一**可核对的判定原因，取值：
    `no_candidate / single_candidate / tie / uncalibrated / bucket_missing /
     calibrated_answer / calibrated_abstain`。
    """
    if not ranked:
        return {"answer": False, "x": "", "p": None, "gain": None,
                "reason": "no_candidate", "features": {}}

    features = features_of_ranked(ranked)
    if len(ranked) == 1:
        # 候选唯一：与 `unique_only` 完全同一条路，不做任何概率判断。
        return {"answer": True, "x": str(ranked[0]["x"]), "p": None, "gain": None,
                "reason": "single_candidate", "features": features}
    if tie_of_top(ranked):
        return {"answer": False, "x": "", "p": None, "gain": None,
                "reason": "tie", "features": features}
    if mapping is None:
        return {"answer": False, "x": "", "p": None, "gain": None,
                "reason": "uncalibrated", "features": features}

    p = probability(mapping, features)
    if p is None:
        return {"answer": False, "x": "", "p": None, "gain": None,
                "reason": "bucket_missing", "features": features}
    # 判定**只有一条**：`should_answer()`。这里不再写第二份 `p > 2/3`，
    # 本项目已经因为"同一判定写两处、取最松的那份生效"出过三次同型事故。
    # `gain` 只作报告字段（敏感性报告要报原始期望增益），不参与判定。
    answer = should_answer(p)
    return {
        "answer": answer,
        "x": str(ranked[0]["x"]) if answer else "",
        "p": p,
        "gain": expected_gain(p),
        "reason": "calibrated_answer" if answer else "calibrated_abstain",
        "features": features,
    }


# --------------------------------------------------------------------- 持久化
def to_json(mapping: dict) -> str:
    """映射表 → JSON 文本（键序固定，便于逐字比对两次拟合结果）。"""
    _check_mapping(mapping)
    return json.dumps(mapping, ensure_ascii=False, sort_keys=True, indent=2)


def from_json(text: str) -> dict:
    """JSON 文本 → 映射表。schema 不符直接抛，不降级。"""
    mapping = json.loads(text)
    if not isinstance(mapping, dict) or mapping.get("schema") != SCHEMA:
        raise ValueError(
            f"映射 schema 不符：期望 {SCHEMA!r}，"
            f"实际 {mapping.get('schema') if isinstance(mapping, dict) else type(mapping).__name__!r}")
    return _check_mapping(mapping)


def save_mapping(path: str | Path, mapping: dict) -> Path:
    """写盘（UTF-8）。**只写调用方显式给的路径**，不猜任何 data/ 位置。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_json(mapping) + "\n", encoding="utf-8")
    return out


def load_mapping(path: str | Path) -> dict:
    """读回映射表。文件不存在就抛——静默退回"不可校准"会让 W2 变成永不触发。"""
    return from_json(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "ANSWER_COST", "FEATURES", "MONOTONE_IN", "REFUSE_COST", "SCHEMA",
    "bucket_key", "decide_top1", "expected_gain", "features_of_ranked", "fit",
    "from_json", "load_mapping", "probability", "save_mapping", "should_answer",
    "threshold", "tie_of_top", "to_json",
]
