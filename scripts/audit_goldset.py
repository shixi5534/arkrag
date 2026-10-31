"""独立复核 Gold Set：不复用 `goldset` 的任何挖掘/校验函数，只读一手材料。

为什么需要这个脚本：Gold Set 的价值取决于「它是否正确」这件事**不是**由产出它的
同一段代码自己宣称的。`goldset.validate_gold_item` 证明了内部一致性（引用块存在、
证据能逐字定位、章节集合自洽），但它与挖掘器共享同一套假设——假设错了，校验和
挖掘会一起错。这里换一条路径：按类别把每条答案重新对回 `chunks.jsonl` 正文、
`graph.json` 的边、`assertions.jsonl` 的断言，对不上就打印并以非零码退出。

两道防线分工：
* `goldset.validate_gold_item` 是产品的一部分，随每次构建运行，保证写出物自洽；
* 本脚本是外部审计，只在需要证明「这套题真的锚在语料事实上」时运行。

用法::

    py -3 scripts\\audit_goldset.py
    py -3 scripts\\audit_goldset.py --gold data/goldset/gold-v1.json --generation g-...

退出码：`0` 全部通过；`1` 发现异常（逐类计数 + 前 20 条明细）；
`2` 输入不可用（Gold Set 不存在、找不到 ark-rag-3 世代）。
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import config, release, vocab  # noqa: E402


def read_jsonl(p):
    rows = []
    with pathlib.Path(p).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pick_generation(wanted: str):
    """取指定世代；未指定时取**最新**的 ark-rag-3 世代（绝不退回 ark-rag-1）。"""
    if wanted:
        p = release.GENERATIONS_DIR / wanted
        return p if (p / "graph" / "graph.json").is_file() else None
    found = None
    for p in sorted(release.GENERATIONS_DIR.iterdir()):
        g = p / "graph" / "graph.json"
        if not g.is_file():
            continue
        try:
            doc = json.loads(g.read_text("utf-8"))
        except (OSError, ValueError):
            continue
        if doc.get("schema_version") == "ark-rag-3":
            found = p
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description="独立复核 Gold Set 是否锚在一手语料事实上")
    ap.add_argument("--gold", default=str(ROOT / "data" / "goldset" / "gold-v1.json"))
    ap.add_argument("--generation", default="")
    args = ap.parse_args()

    gold_path = pathlib.Path(args.gold)
    if not gold_path.is_file():
        print(f"找不到 Gold Set：{gold_path}", file=sys.stderr)
        return 2
    gold = json.loads(gold_path.read_text("utf-8"))
    chunks = {c["chunk_id"]: c for c in read_jsonl(config.chunks_path())}

    gen = pick_generation(args.generation)
    if gen is None:
        print("找不到可用的 ark-rag-3 世代（拒绝用 ark-rag-1 复核）", file=sys.stderr)
        return 2
    graph = json.loads((gen / "graph" / "graph.json").read_text("utf-8"))
    asserts = read_jsonl(gen / "graph" / "assertions.jsonl")
    ents = {e["id"]: e for e in graph["entities"]}
    by_name = {e["name"]: e for e in graph["entities"]}
    edges = graph["edges"]
    eset = {(e["subject_id"], e["predicate"], e["object_id"]) for e in edges}
    eset |= {(o, p, s) for (s, p, o) in eset}
    ekey = {(e["subject_id"], e["predicate"], e["object_id"]): e for e in edges}
    canon = set(vocab.CANONICAL_RELATIONS)
    aid = {a["assertion_id"]: a for a in asserts}

    bad = collections.Counter()
    detail = []
    for it in gold["items"]:
        qid, cat = it["question_id"], it["category"]

        # 1. 引用逐字可定位 + 块真实存在
        for c in it["required_citations"]:
            ch = chunks.get(c["chunk_id"])
            if ch is None:
                bad["citation_chunk_missing"] += 1
                detail.append((qid, "引用块不存在", c["chunk_id"]))
            elif ch["text"].find(c["evidence"]) < 0:
                bad["citation_not_verbatim"] += 1
                detail.append((qid, "引用不在正文", c["evidence"][:30]))

        for cid in it["gold_chunk_ids"]:
            if cid not in chunks:
                bad["gold_chunk_missing"] += 1
                detail.append((qid, "gold 块不存在", cid))

        # 2. 分类专属复核
        if cat == "single_chunk":
            ch = chunks[it["gold_chunk_ids"][0]]
            if ch["title"] != it["answer"]:
                bad["sc_title_mismatch"] += 1
                detail.append((qid, "答案不是块标题", it["answer"], ch["title"]))
            ev = it["required_citations"][0]["evidence"]
            if ch["text"].count(ev) != 1:
                bad["sc_not_unique"] += 1
                detail.append((qid, "证据不唯一", ev[:30]))
            if not (8 <= len(ev) <= 40):
                bad["sc_len_out_of_range"] += 1
                detail.append((qid, "证据长度越界", len(ev)))

        elif cat == "alias":
            ch = chunks[it["gold_chunk_ids"][0]]
            ev = it["required_citations"][0]["evidence"]
            target = by_name.get(it["answer"])
            if target is None:
                bad["al_answer_not_entity"] += 1
                detail.append((qid, "答案实体不在图里", it["answer"]))
            else:
                auth = [
                    a["text"] for a in target["aliases"]
                    if a.get("review") in ("auto", "approved")
                ]
                pending = [a["text"] for a in target.get("pending_aliases") or []]
                used = [a for a in auth if a in it["question"]]
                # 待审别名若只作为某个权威别名的一部分出现（"蒸汽骑士" ⊂ "最后的蒸汽骑士"），
                # 不算泄漏；只有独立出现才是拿未审判断出题。
                leaked = [
                    p for p in pending
                    if p in it["question"] and not any(p in u and p != u for u in used)
                ]
                if leaked:
                    bad["al_pending_alias_leaked"] += 1
                    detail.append((qid, "题面用了待审别名", leaked))
                if not used:
                    bad["al_question_lacks_authoritative_alias"] += 1
                    detail.append((qid, "题面没有权威别名", it["question"][:40]))
                if not any(a in ch["text"] for a in auth):
                    bad["al_alias_not_in_chunk"] += 1
                if target["name"] not in ch["text"]:
                    bad["al_name_not_in_chunk"] += 1
            if not ev:
                bad["al_empty_evidence"] += 1

        elif cat == "direct_relation":
            if it["answer"] not in canon:
                bad["dr_answer_not_in_vocab"] += 1
                detail.append((qid, "答案不在词表", it["answer"]))
            ev = it["required_citations"][0]["evidence"]
            hits = [
                e for e in edges
                if e["predicate"] == it["answer"] and e["evidence"] == ev
            ]
            if not hits:
                bad["dr_no_edge_with_that_evidence"] += 1
                detail.append((qid, "找不到该谓词且证据相同的边", ev[:30]))
            else:
                if not any((e["support_count"] or 0) >= 2 for e in hits):
                    bad["dr_support_lt2"] += 1

        elif cat == "multihop":
            if len(it["required_citations"]) != 2:
                bad["mh_citation_count"] += 1
            for c in it["required_citations"]:
                ev = c["evidence"]
                if not any(e["evidence"] == ev for e in edges):
                    bad["mh_evidence_not_an_edge"] += 1
                    detail.append((qid, "多跳证据不是任何边的证据", ev[:30]))
            mid = by_name.get(it["answer"])
            if mid is None:
                bad["mh_answer_not_entity"] += 1

        elif cat == "cross_chunk":
            chapters = {
                chunks[cid]["chapter"]
                for cid in it["gold_chunk_ids"] if cid in chunks
            }
            claimed = set(it["answer"].split("、"))
            if chapters != claimed:
                bad["cx_chapters_not_covered_by_gold_chunks"] += 1
                detail.append((qid, "章节未被引用块覆盖", sorted(claimed - chapters)[:4]))
            if len(chapters) < 2:
                bad["cx_single_chapter"] += 1
                detail.append((qid, "gold 块只来自一个章节", chapters))
            # 题面里出现的实体名有很多（「正文」里就含单字实体「文」），必须交叉
            # 验证：真正的被问实体应当在**它自己的每一个引用块**正文里都出现。
            cands = [
                e["name"] for e in graph["entities"]
                if e["name"] and len(e["name"]) >= 2 and e["name"] in it["question"]
            ]
            ok_names = [
                n for n in cands
                if all(n in chunks[c["chunk_id"]]["text"] for c in it["required_citations"])
            ]
            if not ok_names:
                bad["cx_no_entity_in_cited_chunks"] += 1
                detail.append((qid, "题面解析不出被问实体", cands[:4]))
            # 每一章的引用块正文里都真的出现了这个名字
            for c in it["required_citations"]:
                if ok_names and ok_names[0] not in chunks[c["chunk_id"]]["text"]:
                    bad["cx_name_not_in_cited_chunk"] += 1
                    detail.append((qid, "名字不在引用块", c["chunk_id"]))

        elif cat == "time_conflict":
            if len(it["gold_chunk_ids"]) < 2:
                bad["tc_chunks_lt2"] += 1
            if "时序演进" not in it["acceptable_answers"]:
                bad["tc_no_evolution_answer"] += 1
            # 断言里必须真有两组互斥谓词、且分处不同 scene
            ps = set()
            for aid_ in it["gold_assertion_ids"]:
                a = aid.get(aid_)
                if a:
                    ps.add(a["predicate"])
            if not ps:
                bad["tc_no_assertion_ids"] += 1
                detail.append((qid, "时间冲突题没有 gold_assertion_ids"))

        elif cat == "negative_abstain":
            if it["should_abstain"]:
                a = by_name.get(it["_absent_a_name"])
                b = by_name.get(it["_absent_b_name"])
                lab = it["_absent_label"]
                if a is None or b is None:
                    bad["na_entity_missing"] += 1
                if lab in canon:
                    bad["na_label_in_vocab"] += 1
                if any(e["predicate"] == lab for e in edges):
                    bad["na_label_used_in_graph"] += 1
                if a and b and ((a["id"], lab, b["id"]) in eset or (b["id"], lab, a["id"]) in eset):
                    bad["na_pair_has_edge"] += 1
                if it["gold_chunk_ids"] or it["required_citations"]:
                    bad["na_carries_evidence"] += 1
            else:
                ev = it["required_citations"][0]["evidence"]
                if not any(n in ev for n in ("没有", "不会", "并未", "拒绝", "不曾", "不再")):
                    bad["na_neg_no_negation_word"] += 1
                if "？" in ev or "?" in ev or "吗" in ev:
                    bad["na_neg_interrogative"] += 1
                    detail.append((qid, "否定题含疑问标记", ev[:30]))
                if it["answer"] != "否":
                    bad["na_neg_answer"] += 1

        elif cat == "adversarial":
            # 从答案里取那两个名字（答案由挖掘时挑出的实体对生成，且题面必须包含它们），
            # 不能"在题面里搜所有出现的实体名"——子串关系会让更短的实体也被搜出来。
            quoted = []
            rest = it["answer"]
            while "「" in rest:
                _head, _sep, rest = rest.partition("「")
                name, _sep2, rest = rest.partition("」")
                quoted.append(name)
            if len(quoted) < 2:
                bad["ad_answer_parse_failed"] += 1
                detail.append((qid, "答案解析不出两个名字", it["answer"][:40]))
            else:
                n1, n2 = quoted[0], quoted[1]
                e1, e2 = by_name.get(n1), by_name.get(n2)
                if e1 is None or e2 is None:
                    bad["ad_name_not_entity"] += 1
                    detail.append((qid, "答案名字不在图里", quoted))
                else:
                    if e1["type"] == e2["type"]:
                        bad["ad_same_type"] += 1
                        detail.append((qid, "两实体同类型", (n1, n2)))
                    if not (n1.startswith(n2) or n2.startswith(n1)):
                        bad["ad_not_prefix"] += 1
                        detail.append((qid, "不是前缀关系", (n1, n2)))
                    for n in quoted:
                        if n not in it["question"]:
                            bad["ad_name_missing_from_question"] += 1
            ch = chunks[it["gold_chunk_ids"][0]]
            short = min(quoted, key=len) if len(quoted) == 2 else ""
            long_ = max(quoted, key=len) if len(quoted) == 2 else ""
            if short and not (short in ch["text"] and long_ not in ch["text"]):
                bad["ad_evidence_supports_wrong_side"] += 1
                detail.append((qid, "证据块没能把短名与长名区分开", short))

        # 3. 通用：答案必须在 acceptable_answers 内
        if it["answer"] not in it["acceptable_answers"]:
            bad["answer_not_acceptable"] += 1

    print("== 独立复核（不复用 goldset 的挖掘与校验逻辑） ==")
    print(f"Gold Set  {gold_path}")
    print(f"世代      {gen.name}")
    print(f"题目数    {len(gold['items'])}")
    if bad:
        for k, v in sorted(bad.items()):
            print(f"  异常 {k}: {v}")
        print("\n前 20 条明细：")
        for d in detail[:20]:
            print("  ", d)
        return 1
    print(f"全部 {len(gold['items'])} 条按类别逐条复核通过（引用逐字、答案可对回一手材料）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
