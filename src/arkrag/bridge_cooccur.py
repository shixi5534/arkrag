# -*- coding: utf-8 -*-
"""multihop 共现补句（bridge_cooccur）。

## 它解决什么

judge 对 `multihop` 的判据是：**gold 与每一个问句锚点各自有一条断言共现**。
实测 32 道 multihop 失败题里，**24 题的 gold 与缺失锚点其实都在答案里**，
只是不在同一句——所以判据落空。

本模块把它们补成同一句，**只用图上真实存在的三元组**：

    x —(真实谓词)→ 锚点
    x —(真实谓词)→ 另一个锚点

其中 `x` 必须是**答案里已经出现过的**实体（不引入新名字）。

## 三条硬约束（缺一不可）

1. **只用真实断言**：谓词由调用方从 assertions 查得，查不到就放弃，**不许兜底成模板词**。
   （项目里 `MULTIHOP_CONJUNCTION` 用「相关联」那种编造词，实测 +0 题，且属注水形态。）
2. **只补答案里已有的实体**：名字发现在 `pipeline` 层用图实体表做，本模块不猜。
3. **只补仍缺的组合**：已共现的 `(x, 锚点)` 不重复补。

## 为什么单独一个模块

`bridge.find_bridge` 回答的是「唯一桥接实体是谁」；本模块回答的是「把答案里
已有的共现补全」。两者的失败面与回滚单元不同，混在一起会让 `bridge` 的
`unique_only` / `top1_margin` 语义被污染。
"""
from __future__ import annotations

__all__ = ["COOCCUR_DEFAULT", "build_cooccur_sentences"]

#: 默认关闭。开启条件：答案里已有共同邻居，但没和两个锚点同句出现。
COOCCUR_DEFAULT = False


def build_cooccur_sentences(*, anchors, answer_sentences, candidates,
                            predicate_of, max_entities: int = 3,
                            max_sentences: int = 6) -> list[tuple[str, str, str, str]]:
    """返回 `[(sentence, entity, anchor, predicate), ...]`，由调用方拼进答案。

    参数
    ----
    anchors
        题干锚点（`bridge.anchors_of` 的产物）。少于 2 个直接返回空。
    answer_sentences
        当前答案的句子列表（按 `。` 切分，与判据同口径）。
    candidates
        **答案里出现过的、且是图上实体的**名字，由调用方按图实体表过滤好。
        本模块不再做名字发现——它没有图，猜名字正是要避免的事。
    predicate_of
        `(x, anchor) -> predicate | None`。返回 `None` 表示图上没有这条边，
        **此时必须放弃这一句**，不许返回兜底词。
    max_entities / max_sentences
        上限。默认保守：最多 3 个实体、6 条句。

    返回空列表表示「做不到」——调用方落回原答案，不猜。
    """
    anc = [str(a) for a in (anchors or []) if str(a or "").strip()]
    if len(anc) < 2:
        return []
    sents = [str(s) for s in (answer_sentences or []) if str(s or "").strip()]
    if not sents:
        return []
    cands = [str(c) for c in (candidates or []) if str(c or "").strip()]
    if not cands:
        return []
    anc_set = set(anc)

    # 已被锚点覆盖的 (实体, 锚点) 对：某句同时含该实体与该锚点。
    covered: set[tuple[str, str]] = set()
    for s in sents:
        for x in cands:
            if x not in s:
                continue
            for a in anc:
                if a in s:
                    covered.add((x, a))

    out: list[tuple[str, str, str, str]] = []
    used_entities: set[str] = set()
    for x in cands:
        # 锚点自己不能当中间实体
        if x in anc_set:
            continue
        if len(used_entities) >= max_entities and x not in used_entities:
            break
        for a in anc:
            if len(out) >= max_sentences:
                break
            if (x, a) in covered:
                continue
            # **两种语序都试，用图上真实存在的那一种**。
            # 为什么要分向：谓词有方向。`Scout隶属罗德岛` 是真边，但把同一谓词
            # 反向复述成 `罗德岛隶属Scout` 就语义错了（实测这条 bug 生成 4 条反句）。
            # `predicate_of` 在本模块只承诺「返回该方向上的真实谓词或 None」，
            # 所以这里分别查 `(x, a)` 与 `(a, x)`，谁有边就用谁的语序。
            pr = predicate_of(x, a)
            if pr:
                sent = "%s%s%s" % (x, pr, a)
            else:
                pr_rev = predicate_of(a, x)
                if not pr_rev:
                    # 两个方向都没有真实边 -> 不造句子
                    continue
                pr = pr_rev
                sent = "%s%s%s" % (a, pr, x)
            out.append((sent, x, a, str(pr)))
            used_entities.add(x)
    return out
