"""H2 等价性核验（真实世代，不进 CI）：证明两项提速改动没有改变任何结果。

## 为什么要一个独立脚本

`tests/test_query_context.py` 与 `tests/test_index_norm_table.py` 用的是**合成语料**——
快、可控、能进 CI，但它们只证明"逻辑对"。合成语料永远不会有真实的
6,582 个实体、8,254 条别名文本、257 个关系前缀、694,304 个词项与
10,525,428 条 postings；而 H2 两项改动的失效形态恰恰都是**只在真实数据规模上
才显现的静默偏差**（少一条别名 → 某问句候选少一个；缓存漏失效 → 分数差千分之几）。

所以这里在**真实世代**上把三件事逐条核到"逐位"：

1. `QueryContext` 复用 vs `analyze_query` 就地重建 —— 整个返回结构深度相等。
2. 新 `search()` vs 冻结的旧实现 —— `(chunk_id, score)` 序列逐位相同（含 float `repr`）。
3. `norm_table()` vs 按公式逐项复算 —— 13,781 项全等。

## 为什么不放进 pytest

要加载 127 MB 索引、跑 320 题两遍，单次三分多钟；它属于"每次改动检索/解析路径后
手工跑一次"的核验，不是每次提交都跑的单元测试。放进 CI 只会让人想办法跳过它。

## 用法

    py -3 scripts\\verify_h2_equivalence.py
    py -3 scripts\\verify_h2_equivalence.py --generation g-20261031T035231Z-4eafae --limit 320

退出码：0 = 三项全等；1 = 有任何一处不等（会打印前几处差异）；2 = 硬错误。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import config, index as index_mod, query as query_mod  # noqa: E402
from arkrag import release  # noqa: E402


def _legacy_search(index, query: str, top_k: int = 20):
    """`LexicalIndex.search()` 在 H2 改动**之前**的逐字副本。

    这是标尺，不是候选实现。**不要顺手优化它**——它一旦被改成"更好"的写法，
    这个脚本就再也证明不了任何事了。
    """
    terms = index.name_terms
    n = max(1, index.size)
    q_terms = Counter(
        index_mod.tokenize(query, version=index.tokenizer_version, name_terms=terms)
    )
    avg_len = index.avg_len
    k1, b = 1.5, 0.75
    scores = defaultdict(float)
    for term, tf in q_terms.items():
        posting = index.postings.get(term)
        if not posting:
            continue
        df = len(posting)
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        for slot, freq in posting.items():
            length = index.doc_len.get(slot, 1)
            scores[slot] += idf * (freq * (k1 + 1)) / (
                freq + k1 * (1 - b + b * length / avg_len)
            )
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], index.doc_id[kv[0]]))[:top_k]
    return [(index.doc_id[slot], score) for slot, score in ranked]


def check_norm_table(idx) -> tuple[bool, str]:
    """逐项按公式复算 `norm_table()`。"""
    norm = idx.norm_table()
    k1, b = index_mod.BM25_K1, index_mod.BM25_B
    bad = 0
    first = ""
    for slot, length in idx.doc_len.items():
        expect = k1 * (1 - b + b * length / idx.avg_len)
        if slot >= len(norm) or norm[slot] != expect:
            bad += 1
            if not first:
                got = norm[slot] if slot < len(norm) else "<越界>"
                first = f"slot={slot} len={length} 期望={expect!r} 实际={got!r}"
    ok = bad == 0
    detail = f"{len(idx.doc_len)} 项，不一致 {bad} 项" + (f"（首处：{first}）" if first else "")
    return ok, detail


def check_search(idx, questions) -> tuple[bool, str]:
    """新 `search` vs 旧实现：`(chunk_id, score)` 序列逐位相同。"""
    diff = 0
    first = ""
    for q in questions:
        new = idx.search(q, top_k=20)
        old = _legacy_search(idx, q, top_k=20)
        if [c for c, _ in new] != [c for c, _ in old] or [
            repr(s) for _, s in new
        ] != [repr(s) for _, s in old]:
            diff += 1
            if not first:
                first = f"问句={q!r} 新={new[:2]} 旧={old[:2]}"
    ok = diff == 0
    detail = f"{len(questions)} 题，不一致 {diff} 题" + (f"（首处：{first}）" if first else "")
    return ok, detail


def check_query_context(entities, aliases, vocab, questions) -> tuple[bool, str]:
    """`QueryContext` 复用 vs 就地重建：整个返回结构深度相等。"""
    ctx = query_mod.QueryContext(entities, aliases, vocab)
    diff = 0
    first = ""
    for q in questions:
        with_ctx = query_mod.analyze_query(
            q, entities=entities, aliases=aliases, vocab=vocab, context=ctx
        )
        without = query_mod.analyze_query(q, entities=entities, aliases=aliases, vocab=vocab)
        if with_ctx != without:
            diff += 1
            if not first:
                a = json.dumps(with_ctx, ensure_ascii=False, sort_keys=True)
                b = json.dumps(without, ensure_ascii=False, sort_keys=True)
                pos = next((i for i, (ca, cb) in enumerate(zip(a, b)) if ca != cb), 0)
                first = (
                    f"问句={q!r}\n"
                    f"      有ctx : ...{a[max(0, pos - 60):pos + 100]}...\n"
                    f"      无ctx : ...{b[max(0, pos - 60):pos + 100]}..."
                )
    ok = diff == 0
    term_count = getattr(ctx, "term_count", 0)
    detail = (
        f"{len(questions)} 题（terms={term_count}），不等 {diff} 题"
        + (f"（首处：{first}）" if first else "")
    )
    return ok, detail


def main() -> int:
    ap = argparse.ArgumentParser(description="H2 等价性核验（真实世代）")
    ap.add_argument("--generation", default="", help="世代 id；省略则用线上 current")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0 = 全部）")
    args = ap.parse_args()

    gid = args.generation or _current_generation()
    gen_dir = release.GENERATIONS_DIR / gid
    index_path = gen_dir / "index" / "lexical.json"
    if not index_path.is_file():
        print(f"[2] 找不到索引：{index_path}", file=sys.stderr)
        return 2

    print(f"世代 {gid}")
    idx, _meta = index_mod.load_index(index_path)
    print(
        f"索引 {len(idx.doc_id)} 文档 / {len(idx.postings)} 词项 / "
        f"{sum(len(p) for p in idx.postings.values())} postings 条目 / "
        f"avg_len={idx.avg_len:.4f}"
    )

    questions = _questions(gen_dir, args.limit)
    print(f"题 {len(questions)} 道")
    print()

    api = _load_graph(gen_dir)
    entities = api.get("entities") or []
    aliases = api.get("alias_index")
    vocab = _load_vocab()

    results = [
        ("norm_table 逐项复算", *check_norm_table(idx)),
        ("新 search vs 旧实现", *check_search(idx, questions)),
        ("QueryContext 复用 vs 重建", *check_query_context(entities, aliases, vocab, questions)),
    ]

    failed = 0
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}：{detail}")
        if not ok:
            failed += 1

    print()
    if failed:
        print(f"{failed} 项不一致——H2 的提速改动改变了结果，必须排查。")
        return 1
    print("三项全部逐位相同：H2 的提速改动没有改变任何结果。")
    return 0


def _current_generation() -> str:
    cur = config.artifact_root() / "current.json"
    if cur.is_file():
        return str(json.loads(cur.read_text("utf-8")).get("generation_id") or "")
    raise SystemExit("[2] 没有 current.json，请用 --generation 指定世代")


def _load_graph(gen_dir: Path) -> dict:
    path = gen_dir / "graph" / "graph.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text("utf-8"))


def _load_vocab():
    """建库/查询用的词表对象。

    `scripts/ask.py` 直接把 `arkrag.vocab` **模块**当 vocab 传下去（`analyze_query`
    只读它的 `CANONICAL_RELATIONS` 等属性），这里必须用同一个对象，否则
    `_relation_lookup` 产出的关系表不同，核验就会假红。
    """
    from arkrag import vocab

    return vocab


def _questions(gen_dir: Path, limit: int) -> list[str]:
    gold = ROOT / "data" / "goldset" / "gold-v1.json"
    if not gold.is_file():
        raise SystemExit(f"[2] 找不到题集：{gold}")
    items = json.loads(gold.read_text("utf-8"))["items"]
    qs = [str(q["question"]) for q in items]
    return qs[:limit] if limit else qs


if __name__ == "__main__":
    raise SystemExit(main())
