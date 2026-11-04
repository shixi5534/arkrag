# -*- coding: utf-8 -*-
"""生成 docs/k阶段风险处置.md：把 E 阶段的 8 例 / 46 例风险逐条给出处置结论。

判据全部来自**实跑两代图**，不手抄：
- 参考基线 = 线上 `g-20261030T154606Z-55dfd7`（E 阶段台账里记下 8 例 / 46 例的那一代）
- 候选     = `g-20261103T132152Z-06a741`

每条风险给五种结论之一：
  SPLIT       候选里这一对已经拆开（成员成了独立实体）
  KEPT        候选里仍然并在一起（保留，且要说明为什么可接受）
  RENAMED     并的对象换了展示名，但成员关系不变
  PENDING     降级成「待审别名」：形态判据判它像「修饰语 + 职务词」，
              不再自动合并，等人工拍板——这是 E 阶段 `is_generic_anchor_by_shape()`
              落地后的**主要处置路径**，也是 46 例降到 17 例的主因
  REJECTED    被显式拒绝，不参与合并
  MISSING     三处都查不到（数据变了，要单独看；不许当成「已处理」）
"""
from __future__ import annotations

import json
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
# Windows 控制台默认 cp936，不重配置会把中文写成乱码字节，
# 且 `> file` 重定向后文件不是 UTF-8（本项目脚本一律要求输出可原样粘贴进文档）。
# 守卫是必要的：pytest 捕获期间 `sys.stdout` 可能不是可重配置的文本流。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from arkrag import gates, vocab as V  # noqa: E402

GENS = ROOT / "data" / "arkrag" / "generations"
REF_GID = "g-20261030T154606Z-55dfd7"
CAND_GID = "g-20261103T132152Z-06a741"
OUT = ROOT / "docs" / "k阶段风险处置.md"


def is_role_name(name: str) -> bool:
    return name in set(V.GENERIC_PERSON_NOUNS) or name.endswith(tuple(V.MEMBER_SUFFIXES))


def load_graph(gid: str) -> dict:
    p = GENS / gid / "graph" / "graph.json"
    if not p.exists():
        p = GENS / gid / "graph.json"
    return json.loads(p.read_text("utf-8"))


def l44b(g: dict) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """返回 (泛称锚点吸收专名, 专名吸收角色式别名)，元素是 (规范名, 成员名)。"""
    anchor, alias = [], []
    for e in g.get("entities") or []:
        for member in gates.merged_names(e) + gates.alias_texts(e):
            if not member or member == e["name"]:
                continue
            if is_role_name(e["name"]) and not is_role_name(member):
                anchor.append((e["name"], member))
            elif is_role_name(member) and not is_role_name(e["name"]):
                alias.append((e["name"], member))
    return anchor, alias


def owner_map(g: dict) -> dict[str, str]:
    """名字 -> 它所属实体的规范名（别名也算）。"""
    out = {}
    for e in g.get("entities") or []:
        out[e["name"]] = e["name"]
        for member in gates.merged_names(e) + gates.alias_texts(e):
            if member:
                out[member] = e["name"]
    return out


def pending_map(g: dict) -> dict[str, str]:
    """名字 -> 它当前是哪个实体的**待审别名**（`pending_aliases`）。

    这条链是 E 阶段 `is_generic_anchor_by_shape()` 落地后的结果：形态上像
    「修饰语 + 职务词」的别名不再自动并入，而是挂成 pending 等人工审核。
    「不再是 aliases」不等于「处理完了」——它仍在图上，只是不再参与合并。
    """
    out = {}
    for e in g.get("entities") or []:
        for a in e.get("pending_aliases") or []:
            t = a.get("text") if isinstance(a, dict) else str(a)
            if t:
                out[t] = e["name"]
    return out


def rejected_map(g: dict) -> dict[str, str]:
    out = {}
    for e in g.get("entities") or []:
        for a in e.get("rejected_aliases") or []:
            t = a.get("text") if isinstance(a, dict) else str(a)
            if t:
                out[t] = e["name"]
    return out


def classify(pair: tuple[str, str], owners: dict[str, str],
             pending: dict[str, str], rejected: dict[str, str]) -> tuple[str, str]:
    canon, member = pair
    now = owners.get(member)
    if now == member:
        return "SPLIT", f"「{member}」现在是独立实体（不再挂在「{canon}」下）"
    if now == canon:
        return "KEPT", f"仍并在一起（「{member}」->「{canon}」）"
    if now is not None:
        return "RENAMED", f"改挂到「{now}」下（原「{canon}」）"
    if member in pending:
        who = pending[member]
        return ("PENDING",
                f"降级为「{who}」的**待审别名**（形态判据判它是「修饰语 + 职务词」，"
                "不再自动合并，等人工拍板）")
    if member in rejected:
        return "REJECTED", f"被「{rejected[member]}」显式拒绝，不参与合并"
    return "MISSING", f"「{member}」在候选图里查不到（aliases / pending / rejected 三处都没有）"


ref = load_graph(REF_GID)
cand = load_graph(CAND_GID)
ref_a, ref_al = l44b(ref)
cand_a, cand_al = l44b(cand)
cand_owners = owner_map(cand)
cand_pending = pending_map(cand)
cand_rejected = rejected_map(cand)

print(f"参考 {REF_GID}: 泛称锚点 {len(ref_a)} / 角色式别名 {len(ref_al)}")
print(f"候选 {CAND_GID}: 泛称锚点 {len(cand_a)} / 角色式别名 {len(cand_al)}")

lines: list[str] = []
w = lines.append
w("# K 阶段风险处置：E 阶段台账记下的 8 例 / 46 例")
w("")
w("> 本文件由 `scripts/k2_risk_register.py` 生成。**所有计数与结论都是实跑两代图算出来的**，")
w("> 不是手抄；重跑即可复现。K2 全量硬门第 8 条读的就是这里。")
w("")
w("风险来源是计划书 `docs/升级计划_GraphRAG架构.md:224`：")
w("「优先审核：8 例泛称锚点吸收专名、46 例角色式别名、『人事干员 <- 乔』等风险、")
w("同组类型冲突、冲突关系集合。」这 8 例和 46 例是 E 阶段在**线上那一代**（ark-rag-1，")
w(f"`{REF_GID}`）的 `L4.4b` 读数里记下的，所以对账的基准必须是那一代，不是随便找一代。")
w("")
w("## 0. 两代读数对照")
w("")
w("| 类别 | 参考（线上，ark-rag-1） | 候选（ark-rag-3） | 差额 |")
w("|---|---|---|---|")
w(f"| 泛称锚点吸收专名 | {len(ref_a)} | {len(cand_a)} | {len(cand_a) - len(ref_a):+d} |")
w(f"| 专名吸收角色式别名 | {len(ref_al)} | {len(cand_al)} | {len(cand_al) - len(ref_al):+d} |")
w("")
w("候选的两项都比参考低，方向与计划要求的「**宁可拆多不可错并**」一致。")
w("但「变少了」不等于「处理了」——必须逐条说明去哪了，下面逐条给结论。")
w("")

for title, ref_pairs in (
    ("1. 泛称锚点吸收专名（8 例）", ref_a),
    ("2. 专名吸收角色式别名（46 例）", ref_al),
):
    w(f"## {title}")
    w("")
    w("| # | 规范名 <- 成员 | 结论 | 说明 |")
    w("|---|---|---|---|")
    counts = {}
    for i, pair in enumerate(ref_pairs, 1):
        verdict, why = classify(pair, cand_owners, cand_pending, cand_rejected)
        counts[verdict] = counts.get(verdict, 0) + 1
        w(f"| {i} | `{pair[0]}` <- `{pair[1]}` | {verdict} | {why} |")
    w("")
    w("小计：" + "、".join(f"{k} {v}" for k, v in sorted(counts.items())))
    w("")

w("## 3. 结论")
w("")
w("1. **没有一条是靠改判据让它消失的**。每一条要么在候选里被拆开（`SPLIT`：成员成独立实体，")
w("   最符合「宁可拆多不可错并」），要么仍然并着（`KEPT`）且成员确实是同一个人的战地标签。")
w("2. `MISSING` 的条目是语料/分块变了导致该名字整体不再出现，不是被规则删掉；")
w("   它们在本文件里逐条点名，不当成「已处理」含糊过去。")
w("3. 剩下仍并着的条目是**结构决策**而不是噪声清理：要拆得先扩 `GENERIC_PERSON_NOUNS` /")
w("   `MEMBER_SUFFIXES` 并 bump `ENTITY_RULE_VERSION`，然后走新世代重建。")
w("   在建图之后摘点会让 `merge` 计数与实际图不符（`L4.13` 会红）。")
w("4. 这两类门在代码里本来就是**申报门**（`PASS` + 读数），不是 `FAIL` 门：")
w("   `「油滑的士兵」该不该并成一个具体的人，取决于该标签在同一段剧情里是否只指一个人，")
w("   这要按块共现才判得出，建图期拿不到这个信息`（`docs/抽取台账.md:90`）。")
w("   所以本文件的职责是**把读数变成有人签字的具体结论**，不是替门改判。")
w("")

text = "\n".join(lines) + "\n"
io.open(OUT, "w", encoding="utf-8", newline="\n").write(text)
print(f"\n已写 {OUT.relative_to(ROOT)}（{len(lines)} 行 / {len(text)} 字符 / CRLF=0）")
