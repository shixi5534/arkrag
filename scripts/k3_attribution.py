# -*- coding: utf-8 -*-
"""K3：两代图差异的**全量**归因——未解释栏必须为 0。

口径：
1. 块层：按 chunk_id 比 content_key，`旧 ck != 新 ck` 即"新增或变更块"。
2. 实体层：按 id 比（同 id 换名 = 改名，不是增删）。
3. 边层：edge_id 由 subject_id/object_id 派生，一次改名会连带改 edge_id，
   所以边必须**按展示名三元组**比，否则一次改名会伪造出一堆"消失+新增"。
4. 断言层：按 assertion_id 比，再分「落在新块上」与「改名涟漪」。
5. 每个差异必须落进 NEW_CHUNK / RENAME / NEW_ENTITY / GONE_ENTITY，
   落不进的一律计入 UNEXPLAINED——**这一栏非 0 就不许发布**。

§5 把口径 3 那句「边不能按 edge_id 比」也跑一遍读数：按 edge_id 比 vs 按展示名
三元组比各得到多少差异、用 edge_id 回查同一条 rename 边在两代各叫什么名字、
同名三元组在两代里的 edge_id 是否一致。**两种键在这份数据上都会给出错形状的
报告**，§5 就是这句话的证据，不是补充说明。

退出码：0 = 未解释栏为 0（判据成立）/ 1 = 存在未解释差异（不许发布）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GEN = ROOT / "data" / "arkrag" / "generations"
OLD = GEN / "g-20261031T035231Z-4eafae"
NEW = GEN / "g-20261103T132152Z-06a741"
OLD_CHUNKS = ROOT / "data" / "arkrag" / "chunks.jsonl"
NEW_CHUNKS = ROOT / "data" / "arkrag" / "staging" / "k2" / "chunks.jsonl"

RENAME = {"莱伊", "劫车人"}  # 同一人物的两个名字（互指）


def chunk_ids(path: Path) -> dict[str, str]:
    out = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[str(rec["chunk_id"])] = str(rec.get("content_key") or "")
    return out


def edges_by_name(g: dict) -> dict[tuple[str, str, str], dict]:
    by_id = {e["id"]: e for e in g["entities"]}
    out = {}
    for ed in g["edges"]:
        out[(by_id[ed["subject_id"]]["name"], ed["predicate"], by_id[ed["object_id"]]["name"])] = ed
    return out


def asserts(path: Path) -> dict[str, dict]:
    out = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                a = json.loads(line)
                out[a["assertion_id"]] = a
    return out


def split(rows, by_id, new_blocks):
    on_new = [a for a in rows if a.get("chunk_id") in new_blocks]
    rest = [a for a in rows if a.get("chunk_id") not in new_blocks]
    rename = [a for a in rest
              if by_id.get(a.get("subject_id"), {}).get("name") in RENAME
              or by_id.get(a.get("object_id"), {}).get("name") in RENAME]
    other = [a for a in rest if a not in rename]
    return on_new, rename, other


# ---------------------------------------------------------------- §5 边键对照
def edge_key_section(old: dict, new: dict, eo: dict, en: dict) -> None:
    """§5：为什么边按展示名比而不是按 edge_id 比——两种键各自的读数。"""
    print("\n===== §5 边为什么按展示名比 =====")

    # ---- §5.1 按 edge_id 比 vs 按展示名三元组比 ----
    print("=== 按 edge_id 比 ===")
    a = {e["edge_id"] for e in old["edges"]}
    b = {e["edge_id"] for e in new["edges"]}
    print("  旧 edge_id %d / 新 edge_id %d / 仅旧 %d / 仅新 %d"
          % (len(a), len(b), len(a - b), len(b - a)))
    print()
    print("=== 按展示名三元组比（同 attribution.py 口径）===")

    def trips(g, em):
        s = []
        for e in g["edges"]:
            sn = em.get(e["subject_id"], {}).get("name", e["subject_id"])
            on = em.get(e["object_id"], {}).get("name", e["object_id"])
            s.append((sn, e["predicate"], on))
        return s

    to, tn = trips(old, eo), trips(new, en)
    ao, an = set(to), set(tn)
    print("  旧三元组 %d / 新三元组 %d / 仅旧 %d / 仅新 %d"
          % (len(ao), len(an), len(ao - an), len(an - ao)))

    # ---- §5.2 用 edge_id 追踪同一条 rename 边 ----
    def nm(em, e):
        return (em.get(e["subject_id"], {}).get("name", "?"), e["predicate"],
                em.get(e["object_id"], {}).get("name", "?"))

    bo = {e["edge_id"]: e for e in old["edges"]}
    bn = {e["edge_id"]: e for e in new["edges"]}
    print()
    print("=== 7 条 rename 关系：在两代里各叫什么名字（按 edge_id 追踪同一条边）===")
    targets = [("劫车人", "交谈", "小锅盖"), ("劫车人", "交谈", "阿兰娜"), ("劫车人", "前往", "咧嘴谷"),
               ("劫车人", "持有", "弩"), ("劫车人", "救助", "小锅盖"), ("阿兰娜", "交谈", "劫车人")]
    print("-- 新代这 6 条，用 edge_id 回查旧代 --")
    for t in targets:
        hit = [e for e in new["edges"] if nm(en, e) == t]
        for e in hit:
            eid = e["edge_id"]
            oldn = nm(eo, bo[eid]) if eid in bo else None
            print("  新 %-22s edge_id %s → 旧代同 id 的名字 %s"
                  % (f"{t[0]}-{t[1]}->{t[2]}", eid, oldn))
    print("-- 旧代那 1 条，用 edge_id 回查新代 --")
    for e in old["edges"]:
        if nm(eo, e) == ("莱伊", "持有", "弩"):
            eid = e["edge_id"]
            print("  旧 莱伊-持有->弩        edge_id %s → 新代同 id 的名字 %s"
                  % (eid, nm(en, bn[eid]) if eid in bn else None))
    print()
    print("=== 结论用数：两种键各得到多少「差异」 ===")
    ao2 = {e["edge_id"] for e in old["edges"]}
    an2 = {e["edge_id"] for e in new["edges"]}
    to2 = {nm(eo, e) for e in old["edges"]}
    tn2 = {nm(en, e) for e in new["edges"]}
    print("  按 edge_id ：仅旧 %d / 仅新 %d（合计 %d）"
          % (len(ao2 - an2), len(an2 - ao2), len(ao2 - an2) + len(an2 - ao2)))
    print("  按名三元组 ：仅旧 %d / 仅新 %d（合计 %d）"
          % (len(to2 - tn2), len(tn2 - to2), len(to2 - tn2) + len(tn2 - to2)))

    # ---- §5.3 同名三元组的 edge_id 是否一致 ----
    bo5 = {}
    bn5 = {}
    for e in old["edges"]:
        bo5[nm(eo, e)] = e["edge_id"]
    for e in new["edges"]:
        bn5[nm(en, e)] = e["edge_id"]
    same = [t for t in bo5 if t in bn5]
    diff = [t for t in same if bo5[t] != bn5[t]]
    print()
    print("同名三元组在两代都出现：%d 条" % len(same))
    print("  其中 edge_id 相同：%d" % (len(same) - len(diff)))
    print("  其中 edge_id 不同：%d" % len(diff))
    for t in diff[:12]:
        print("    %-30s 旧 %s / 新 %s" % (f"{t[0]}-{t[1]}->{t[2]}", bo5[t], bn5[t]))
    print()
    # 看 diff 的三元组是否涉及 莱伊 / 劫车人
    involve = [t for t in diff if "莱伊" in t or "劫车人" in t]
    print("edge_id 不同且涉及 莱伊/劫车人：%d" % len(involve))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    old_ck = chunk_ids(OLD_CHUNKS)
    new_ck = chunk_ids(NEW_CHUNKS)
    new_blocks = {cid for cid, ck in new_ck.items() if old_ck.get(cid) != ck}
    print("===== 块 =====")
    print(f"  旧块 {len(old_ck):,} / 新块 {len(new_ck):,} / 新增或变更块 {len(new_blocks)}")

    old = json.loads((OLD / "graph" / "graph.json").read_text("utf-8"))
    new = json.loads((NEW / "graph" / "graph.json").read_text("utf-8"))

    old_by_id = {e["id"]: e for e in old["entities"]}
    new_by_id = {e["id"]: e for e in new["entities"]}
    old_names = {e["name"] for e in old["entities"]}
    new_names = {e["name"] for e in new["entities"]}

    only_id_old = set(old_by_id) - set(new_by_id)
    only_id_new = set(new_by_id) - set(old_by_id)
    renamed = [(i, old_by_id[i]["name"], new_by_id[i]["name"])
               for i in set(old_by_id) & set(new_by_id)
               if old_by_id[i]["name"] != new_by_id[i]["name"]]

    print("\n===== 实体 =====")
    print(f"  仅旧 id {len(only_id_old)} / 仅新 id {len(only_id_new)} / 同 id 换名 {len(renamed)}")
    for i, a, b in renamed:
        print(f"    改名 {i}: {a} -> {b}（mentions {old_by_id[i]['mentions']} -> {new_by_id[i]['mentions']}）")
    for i in sorted(only_id_old):
        e = old_by_id[i]
        print(f"    - {i} {e['name']} ({e['type']}) chunks={len(e.get('chunks') or [])}")
    for i in sorted(only_id_new):
        e = new_by_id[i]
        print(f"    + {i} {e['name']} ({e['type']}) chunks={len(e.get('chunks') or [])}")
    print(f"  实体数 old={len(old['entities']):,} new={len(new['entities']):,}"
          f"（{len(new['entities']) - len(old['entities']):+d}）")

    # ---------- 边 ----------
    oe, ne = edges_by_name(old), edges_by_name(new)
    only_e_new, only_e_old = set(ne) - set(oe), set(oe) - set(ne)
    print("\n===== 边（按展示名三元组比）=====")
    print(f"  旧边 {len(oe):,} / 新边 {len(ne):,} / 仅新 {len(only_e_new)} / 仅旧 {len(only_e_old)}")

    def classify(triple, ed):
        s, _, o = triple
        if set(ed.get("chunks") or []) & new_blocks:
            return "NEW_CHUNK"
        if s in RENAME or o in RENAME:
            return "RENAME"
        if s not in old_names or o not in old_names:
            return "NEW_ENTITY"
        if s not in new_names or o not in new_names:
            return "GONE_ENTITY"
        return "UNEXPLAINED"

    buckets, unexplained = {}, []
    for label, triples, pool in (("仅新", only_e_new, ne), ("仅旧", only_e_old, oe)):
        for t in sorted(triples):
            b = classify(t, pool[t])
            buckets.setdefault(f"{label}/{b}", []).append(t)
            if b == "UNEXPLAINED":
                unexplained.append((label, t, (pool[t].get("chunks") or [])[:3]))
    for k in sorted(buckets):
        print(f"  {k}: {len(buckets[k])}")
        for t in buckets[k]:
            print(f"      {t[0]} -{t[1]}-> {t[2]}")
    print(f"  未解释边：{len(unexplained)}")

    # ---------- 断言 ----------
    old_a = asserts(OLD / "graph" / "assertions.jsonl")
    new_a = asserts(NEW / "graph" / "assertions.jsonl")
    new_only = [new_a[k] for k in set(new_a) - set(old_a)]
    old_only = [old_a[k] for k in set(old_a) - set(new_a)]
    by_id_new = {e["id"]: e for e in new["entities"]}
    by_id_old = {e["id"]: e for e in old["entities"]}

    n_new, n_ren, n_other = split(new_only, by_id_new, new_blocks)
    o_new, o_ren, o_other = split(old_only, by_id_old, new_blocks)
    print("\n===== 断言 =====")
    print(f"  旧 {len(old_a):,} / 新 {len(new_a):,}（{len(new_a) - len(old_a):+d}）")
    print(f"  仅新 {len(new_only)} = 落新块 {len(n_new)} + 改名涟漪 {len(n_ren)} + 未解释 {len(n_other)}")
    print(f"  仅旧 {len(old_only)} = 落新块 {len(o_new)} + 改名涟漪 {len(o_ren)} + 未解释 {len(o_other)}")
    for tag, rows in (("仅新未解释", n_other), ("仅旧未解释", o_other)):
        for a in rows[:10]:
            print(f"      {tag}: chunk={a.get('chunk_id')} subj={by_id_new.get(a.get('subject_id'), by_id_old.get(a.get('subject_id'), {})).get('name')}")

    total_unexplained = len(unexplained) + len(n_other) + len(o_other)
    print(f"\n未解释合计（边 + 断言）：{total_unexplained}")
    print("结论：" + ("全部差异已归因，未解释 0" if total_unexplained == 0 else "存在未解释差异，不许发布"))

    # ---------- §5 边键对照 ----------
    edge_key_section(old, new, old_by_id, new_by_id)

    return 0 if total_unexplained == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
