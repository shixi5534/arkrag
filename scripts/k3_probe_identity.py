# -*- coding: utf-8 -*-
"""K3：身份机理——`stable_id` 与别名分组。

要回答的问题是：为什么 `莱伊 → 劫车人` 这条「同 id 换名」不是改名 bug。
`e-6fa1624b83f766b0` 的哈希对象从来是 `劫车人`（`identity._UnionFind` 合并时根取组内
字典序最小的成员名，`stable_id()` 哈希这个根），而**显示名是另一套算法**
（`canonical_pick`：出现次数多 → 含 `·` → 短 → 字典序）。旧代 `莱伊` 出现更多所以显示名
取 `莱伊`；新代两名不再合并，id 回落到出生名 `劫车人`，`莱伊` 另拿新 id。

只读两代 graph.json 与 `arkrag.identity`。

退出码：0 = 两代 entity_rule_version 一致且两个 stable_id 都能算出（判据成立）/ 1 = 不成立。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from arkrag import identity  # noqa: E402

GEN = ROOT / "data" / "arkrag" / "generations"
OLD = GEN / "g-20261031T035231Z-4eafae"
NEW = GEN / "g-20261103T132152Z-06a741"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    og = json.loads((OLD / "graph" / "graph.json").read_text("utf-8"))
    ng = json.loads((NEW / "graph" / "graph.json").read_text("utf-8"))
    print("old entity_rule_version:", og.get("entity_rule_version"), "new:", ng.get("entity_rule_version"))
    for tag, g in (("old", og), ("new", ng)):
        for e in g["entities"]:
            if e["id"] == "e-6fa1624b83f766b0":
                print(tag, json.dumps({k: e[k] for k in ("id", "name", "type", "aliases", "merged_from", "mentions")}, ensure_ascii=False))
        # any entity named 莱伊 or 劫车人
        for e in g["entities"]:
            if e["name"] in ("莱伊", "劫车人"):
                print(tag, "name-hit", e["id"], e["name"], "aliases=", [a.get("text") for a in e.get("aliases") or []], "chunks=", len(e.get("chunks") or []))

    rule = ng.get("entity_rule_version")
    ids = {}
    for nm in ("劫车人", "莱伊"):
        ids[nm] = identity.stable_id(nm, entity_rule_version=rule)
        print("stable_id(", nm, ") =", ids[nm])
    print("sorted pair:", sorted(["莱伊", "劫车人"]))

    # 判据：两代规则版本一致（否则 stable_id 不可跨代比），且两个名字都给出 id
    ok = (og.get("entity_rule_version") == ng.get("entity_rule_version")
          and all(ids.values()))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
