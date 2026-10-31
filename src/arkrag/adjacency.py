"""有方向的邻接表 —— 计划 E2。

## 旧实现的错在哪

`index.build_adjacency` 把每条边往两个方向各塞一份，**反向那一份沿用同一个关系名**：

```python
adj[src].append((dst, edge["relation"], weight))
adj[dst].append((src, edge["relation"], weight))   # 关系名没变
```

于是「凯尔希 隶属 罗德岛」在图扩展时会被读成「罗德岛 隶属 凯尔希」。
这不是精度问题，是把事实说反了——而检索侧完全看不出来，因为它拿到的只是一串
(邻居, 关系名, 权重) 三元组，没有方向标记可以自查。

（附带事实：那个函数在本仓库**没有任何调用方**，属于从未被执行过的代码。
所以下面的实现不是"修一个 bug"，而是"第一次真正建邻接"。）

## 这一层的规矩

1. **出边与入边分开存**，不能混在一个列表里。
2. 反向边必须换成**逆谓词**（`隶属` ↔ `包含`）。词表里没有逆谓词时，
   `inverse_relation=""` 且该反向边不得用于"顺着方向说事实"——只能当无向邻接用。
3. **对称谓词**（`交战`/`同盟`/`亲属`…）反向仍是自身，标记 `symmetric=True`、
   方向记 `undirected`。
4. 每个邻接项都带 `edge_id` 与 `assertion_ids`，可以一路回溯到原文证据。
5. 默认图扩展**只走出边**；要走入边必须显式声明（`direction="in"` 或 `"both"`）。

## 与检索的关系

这里不做权重排序的业务判断，只保证"方向、谓词、可回溯性"三件事正确。
优先级打分留给 G4 的图扩展预算。
"""
from __future__ import annotations

from collections import defaultdict

from . import vocab

DIRECTION_OUT = "out"
DIRECTION_IN = "in"
DIRECTION_UNDIRECTED = "undirected"


def inverse_of(predicate: str) -> str:
    """逆谓词。没有配对时返回空串——**不回退成原谓词**。"""
    for a, b in vocab.INVERSE_PAIRS:
        if predicate == a:
            return b
        if predicate == b:
            return a
    return ""


def is_symmetric(predicate: str) -> bool:
    return predicate in vocab.SYMMETRIC_RELATIONS


def adjacency_item(
    *,
    neighbor_id: str,
    relation: str,
    direction: str,
    edge: dict,
) -> dict:
    """单个邻接项。字段是契约 §2 规定的全集。"""
    rev = inverse_of(relation)
    symmetric = is_symmetric(relation)
    if symmetric:
        # 对称关系没有反向语义，方向如实记 undirected，不要假装成 out
        direction = DIRECTION_UNDIRECTED
    return {
        "neighbor_id": neighbor_id,
        "relation": relation,
        "direction": direction,
        "inverse_relation": relation if symmetric else rev,
        "symmetric": symmetric,
        "edge_id": edge["edge_id"],
        "assertion_ids": list(edge.get("assertion_ids") or []),
        "weight": float(edge.get("confidence_summary", {}).get("mean") or 0.0),
        "support_count": int(edge.get("support_count") or 0),
        "independent_scene_count": int(edge.get("independent_scene_count") or 0),
        "reversible": bool(symmetric or rev),
    }


def build_directed_adjacency(edges: list[dict]) -> dict[str, dict]:
    """`{entity_id: {"out": [...], "in": [...]}}`。

    出边用原谓词；入边用逆谓词（没有逆谓词就留空串），并且**单独一个列表**，
    这样调用方不可能在不知情的情况下把入边当出边用。

    对称谓词（`交战`/`同盟`/…）没有方向可言：两侧都记 `out` 且方向标
    `undirected`。把对称边塞进 `in` 列表会暗示"有个人指向我"，
    而检索侧一旦按 `in` 做方向性回答就会说错——这是比丢边更坏的错。
    """
    adj: dict[str, dict] = defaultdict(lambda: {"out": [], "in": []})
    for edge in edges:
        subject = edge["subject_id"]
        obj = edge["object_id"]
        predicate = edge["predicate"]
        if is_symmetric(predicate):
            adj[subject]["out"].append(
                adjacency_item(
                    neighbor_id=obj,
                    relation=predicate,
                    direction=DIRECTION_UNDIRECTED,
                    edge=edge,
                )
            )
            adj[obj]["out"].append(
                adjacency_item(
                    neighbor_id=subject,
                    relation=predicate,
                    direction=DIRECTION_UNDIRECTED,
                    edge=edge,
                )
            )
            continue
        adj[subject]["out"].append(
            adjacency_item(
                neighbor_id=obj,
                relation=predicate,
                direction=DIRECTION_OUT,
                edge=edge,
            )
        )
        adj[obj]["in"].append(
            adjacency_item(
                neighbor_id=subject,
                relation=predicate,
                # 入边：这条边从邻居指向我。谓词保持原样（"邻居 隶属 我"是原事实），
                # 但从我这一侧出发去描述它时要用逆谓词，见 inverse_relation。
                direction=DIRECTION_IN,
                edge=edge,
            )
        )
    # 排序固定下来：同一份图重复构建必须得到同一份邻接（否则 fingerprint 会漂）
    for node in adj.values():
        node["out"].sort(key=lambda it: (-it["support_count"], it["edge_id"]))
        node["in"].sort(key=lambda it: (-it["support_count"], it["edge_id"]))
    return {k: v for k, v in adj.items()}


def neighbors(
    adj: dict[str, dict],
    entity_id: str,
    *,
    direction: str = "out",
    limit: int = 0,
) -> list[dict]:
    """取邻居。`direction` 只接受 out / in / both —— 拼错就报错，不默认放宽。"""
    if direction not in ("out", "in", "both"):
        raise ValueError(f"direction 只能是 out/in/both，收到 {direction!r}")
    node = adj.get(entity_id) or {"out": [], "in": []}
    if direction == "out":
        items = list(node["out"])
    elif direction == "in":
        items = list(node["in"])
    else:
        items = list(node["out"]) + list(node["in"])
    return items[:limit] if limit else items


def describe_traversal(adj: dict[str, dict], path: list[dict]) -> list[str]:
    """把一条扩展路径翻译成人能读的句子。

    存在的理由：图扩展如果只返回"实体 + 关系名"，回答时很容易顺着反向边说反话。
    把路径渲成句子，至少让每一步的方向错误在日志与测试里看得见——
    例如 `凯尔希 --隶属--> 罗德岛` 与 `罗德岛 <-隶属-- 凯尔希` 必须写成不同的话。
    """
    lines: list[str] = []
    for hop in path:
        relation = hop.get("relation") or ""
        if hop.get("direction") == DIRECTION_IN:
            subject = hop.get("neighbor_id", "")
            obj = hop.get("from_id") or hop.get("subject_id") or ""
            lines.append(f"{subject} --{relation}--> {obj}（反向读入）")
        elif hop.get("direction") == DIRECTION_UNDIRECTED:
            lines.append(
                f"{hop.get('from_id', '')} <--{relation}--> {hop.get('neighbor_id', '')}"
            )
        else:
            lines.append(
                f"{hop.get('from_id', '')} --{relation}--> {hop.get('neighbor_id', '')}"
            )
    return lines


def direction_report(adj: dict[str, dict]) -> dict:
    """邻接表的自检摘要（给门禁用）。

    要回答的问题：有没有邻接项把入边伪装成出边？有逆谓词缺失的反向边占多少？

    对称谓词的邻接项两列表里都可能出现、方向如实记 `undirected`，
    所以两列表都接受 `undirected`——只把"方向与所在列表不一致"算成伪装。
    """
    out_items = 0
    in_items = 0
    missing_inverse = 0
    fake_out = 0
    for node in adj.values():
        for item in node["out"]:
            out_items += 1
            if item["direction"] not in (DIRECTION_OUT, DIRECTION_UNDIRECTED):
                fake_out += 1
        for item in node["in"]:
            in_items += 1
            if item["direction"] not in (DIRECTION_IN, DIRECTION_UNDIRECTED):
                fake_out += 1
            if not item["symmetric"] and not item["inverse_relation"]:
                missing_inverse += 1
    return {
        "entities": len(adj),
        "out_items": out_items,
        "in_items": in_items,
        "missing_inverse": missing_inverse,
        "fake_out": fake_out,
    }


def verify_reverse_semantics(edges: list[dict]) -> list[str]:
    """逐边核对：反向描述必须换成逆谓词，不许沿用原谓词。

    返回问题列表（空 = 没问题）。造一个"反向边"直接喂给检查，
    这样门禁不需要依赖检索侧的调用路径就能验这件事。
    """
    problems: list[str] = []
    for edge in edges:
        predicate = edge["predicate"]
        if is_symmetric(predicate):
            continue
        rev = inverse_of(predicate)
        if not rev:
            # 没有逆谓词的边：允许存在，但必须能被识别出来（inverse_relation 为空）
            continue
        item = adjacency_item(
            neighbor_id=edge["subject_id"],
            relation=predicate,
            direction=DIRECTION_IN,
            edge=edge,
        )
        if item["inverse_relation"] == predicate:
            problems.append(
                f"{edge['edge_id']}: 反向沿用原谓词 {predicate!r}（应为 {rev!r}）"
            )
        elif item["inverse_relation"] != rev:
            problems.append(
                f"{edge['edge_id']}: 逆谓词是 {item['inverse_relation']!r}，词表要求 {rev!r}"
            )
    return problems
