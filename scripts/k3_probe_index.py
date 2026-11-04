# -*- coding: utf-8 -*-
"""K3：索引体积变化的归因——词项数下降 ≠ 语料缩水。

三件事分别量：
1. 两代索引的文档/词项/条目数与文件字节；
2. 同一批正文在 t1 / t2 下的 token 数之比（量纲是否真的缩了）；
3. t2 把专名整词入表带来的词项贡献 + 专名边界不连双字的影响。

判据：三件事的读数必须全部拿到——索引能载入、`in_t1` 块非空（否则 token 比无从算）、
专名表非空。任何一项缺失都退出 1，**不许把「没算出来」当成「没变化」**。
退出码 0 = 判据成立 / 1 = 有量缺失。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from arkrag import index as index_mod  # noqa: E402

GEN = ROOT / "data" / "arkrag" / "generations"
OLD = GEN / "g-20261031T035231Z-4eafae"
NEW = GEN / "g-20261103T132152Z-06a741"
NEW_CHUNKS = ROOT / "data" / "arkrag" / "staging" / "k2" / "chunks.jsonl"

T1_SAMPLE = 300  # 同一批正文取前 N 块 T1 块做 token 比


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    old_idx, old_meta = index_mod.load_index(OLD / "index" / "lexical.json")
    new_idx, new_meta = index_mod.load_index(NEW / "index" / "lexical.json")

    print("=== 两代索引 meta ===")
    print(f"  old tokenizer_version = {old_meta.get('tokenizer_version')!r}  doc_count={old_meta.get('doc_count')}")
    print(f"  new tokenizer_version = {new_meta.get('tokenizer_version')!r}  doc_count={new_meta.get('doc_count')}")

    print("\n=== 倒排规模 ===")
    rows = []
    for tag, gen, idx in (("old", OLD, old_idx), ("new", NEW, new_idx)):
        terms = len(idx.postings)
        entries = sum(len(p) for p in idx.postings.values())
        size = (gen / "index" / "lexical.json").stat().st_size
        rows.append((tag, len(idx.doc_id), terms, entries, size))
        print(f"  {tag}: 文档 {len(idx.doc_id)} / 词项 {terms:,} / 条目 {entries:,} / "
              f"平均每词 {entries / terms:.3f} / lexical.json {size:,} B")

    o, n = rows[0], rows[1]
    print(f"\n  文档 {o[1]:,} → {n[1]:,}（{n[1] - o[1]:+d}）")
    print(f"  词项 {o[2]:,} → {n[2]:,}（{n[2] - o[2]:+,d}，{n[2] / o[2] - 1:+.2%}）")
    print(f"  条目 {o[3]:,} → {n[3]:,}（{n[3] - o[3]:+,d}，{n[3] / o[3] - 1:+.2%}）")
    print(f"  字节 {o[4]:,} → {n[4]:,}（{n[4] / o[4] - 1:+.2%}）")

    # ---------- 2. 同一批正文的 t1 / t2 token 数 ----------
    texts = []
    with NEW_CHUNKS.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("in_t1"):
                texts.append(f"{rec.get('title', '')} {rec.get('chapter', '')} {rec.get('text', '')}")
    texts = texts[:T1_SAMPLE]

    graph = json.loads((NEW / "graph" / "graph.json").read_text("utf-8"))
    names = set()
    for e in graph.get("entities") or []:
        names.add(e["name"])
        for a in e.get("aliases") or []:
            names.add(a.get("text") if isinstance(a, dict) else a)
    names.discard(None)
    table = index_mod._as_table(frozenset(names))
    print(f"\n=== 同一批正文在 t1 / t2 下的 token 数（前 {len(texts)} 块 T1）===")
    print(f"  专名表 {len(names)} 项")
    tot = {}
    for ver in ("t1", "t2"):
        total = sum(len(index_mod.tokenize(t, version=ver, name_terms=table)) for t in texts)
        tot[ver] = total
        print(f"  {ver}: 总 token {total} / 平均每块 {total / len(texts):.1f}")
    print(f"  t2/t1 = {tot['t2'] / tot['t1']:.4f}（正文量纲只缩了 {1 - tot['t2'] / tot['t1']:.2%}）")

    # ---------- 3. 专名整词入表的贡献 ----------
    print("\n=== 专名整词入表贡献（t2 独有）===")
    t2_no_table = sum(len(index_mod.tokenize(t, version="t2")) for t in texts)
    print(f"  t2 传 name_terms 总 token {tot['t2']} / 不传 {t2_no_table}")
    print(f"  差 {t2_no_table - tot['t2']}（专名整词并入后减少的分词条目数）+ 专名整词本身 {len(table)} 类词项")
    print(f"  专名表项数 = {len(table)}")

    # 判据：token 比有效（分母非 0）且专名表非空
    ok = bool(texts) and tot.get("t1", 0) > 0 and len(table) > 0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
