"""SQLite FTS5 后端的测试（H3 第 ③ 步）。

## 这里最要紧的一条

`test_exact_mode_matches_lexical_index_bitwise` 断言的是**新旧后端的排序与分数逐位
相同**。它不是"差不多对齐"的自证，而是本次迁移的验收条件本身：只要 FTS5 侧的
BM25 复算在任何一个细节上与 `LexicalIndex.search` 分叉（遍历顺序、乘除结合、
`doc_len` 兜底值、同分次序），这条就会红。

为了让这条门真的能失败，用例里的语料是**故意构造**的：

* `c-03` 分数最高但 `chunk_id` 字典序最大 —— 任何"顺手按 id 排"会红；
* 有两个块分数完全相同 —— 同分次序错了会红；
* 一个块只含单字、一个块含超高频词 —— `doc_len` 兜底与 df 计算错了会红。

## 另外几条守的是纪律，不是功能

* `test_no_bm25_literals_in_source` —— BM25 参数只能 import，不许写字面量。
  本项目明令要求，且已有过一次"两处各写一份、改一处忘一处"的事故。
* `test_build_refuses_path_inside_generation` —— 纪律 #6，新产物不许进世代目录。
* `test_schema_version_mismatch_raises` —— 库与代码不同源时必须报错，
  不许按"看起来能跑"继续算。
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

from arkrag import fts_backend, index as index_mod

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- 夹具
def _corpus_index() -> "index_mod.LexicalIndex":
    """精心构造的小语料，专治"排序看起来对"的假通过。见模块 docstring。"""
    idx = index_mod.LexicalIndex(tokenizer_version="t1")
    idx.add(0, "c-01", "alpha")                                    # 最短
    idx.add(1, "c-02", "alpha beta 龙门 罗德岛")                     # 中等
    idx.add(2, "c-03", "alpha alpha alpha alpha 凯尔希 龙门")        # 分最高、id 最大
    idx.add(3, "c-04", "beta beta 阿米娅")                          # 与 c-05 同分
    idx.add(4, "c-05", "beta beta 阿米娅")                          # 与 c-04 同分
    idx.add(5, "c-06", "的 的 的 的 的 的 的 的")                     # 超高频词
    return idx


def _build(tmp_path: Path, idx: "index_mod.LexicalIndex", **kwargs) -> Path:
    db = tmp_path / "idx.sqlite"
    fts_backend.build_database(db, index=idx, index_meta={"index_content_digest": "x"}, **kwargs)
    return db


# ---------------------------------------------------------------- 打分口径（核心门）
def test_exact_mode_matches_lexical_index_bitwise(tmp_path: Path) -> None:
    """新旧后端 top_k 与分数必须逐位相同。这是本次迁移的验收条件本体。"""
    idx = _corpus_index()
    db = _build(tmp_path, idx)
    backend = fts_backend.Fts5SearchBackend.open(db)
    try:
        queries = [
            "alpha",
            "beta",
            "beta 阿米娅",          # 同分文档，考同分次序
            "alpha 龙门",
            "凯尔希",
            "的",
            "不存在的词",            # 空结果
            "",                     # 空查询
            "alpha alpha alpha",    # 重复查询词
        ]
        for query in queries:
            expected = idx.search(query, top_k=10)
            got = backend.search(query, top_k=10)
            assert [c for c, _ in got] == [c for c, _ in expected], (
                f"{query!r}：chunk_id 序列不同\n  旧={[c for c, _ in expected]}\n  新={[c for c, _ in got]}"
            )
            # 逐位相等，不是 approx：两边是同一个公式、同一运算次序
            assert [s for _, s in got] == [s for _, s in expected], (
                f"{query!r}：分数不是逐位相同\n  旧={[s for _, s in expected]}\n  新={[s for _, s in got]}"
            )
    finally:
        backend.close()


def test_exact_mode_matches_on_top_k_boundary(tmp_path: Path) -> None:
    """截断位置也要一致：top_k 恰好切在同分文档中间时最容易翻。"""
    idx = _corpus_index()
    db = _build(tmp_path, idx)
    backend = fts_backend.Fts5SearchBackend.open(db)
    try:
        for k in (1, 2, 3, 5, 6, 100):
            expected = idx.search("beta", top_k=k)
            got = backend.search("beta", top_k=k)
            assert [c for c, _ in got] == [c for c, _ in expected], f"top_k={k} 时截断位置不同"
    finally:
        backend.close()


def test_norm_table_matches_index_norm_table(tmp_path: Path) -> None:
    """长度归一化项必须与 `LexicalIndex.norm_table()` 逐位相同。

    `norm_table()` 是 H2 为消除「每问重算一遍同样的除法」引入的；FTS5 后端
    又建了一份自己用的扁平表。两份的公式一旦分叉，分数会在 1e-16 量级上漂，
    而「逐位相同」那条门就会变成时不时红一次的不稳定门——那种门最终会被
    人加 `approx` 绕过。所以这一条单独钉住。

    ## 为什么语料里**故意留一个空洞槽位**

    第一版语料所有槽位都连续、每个槽都有长度，于是「空洞槽位的兜底值」这条
    分支根本没被走到——把兜底值改成 0，测试照样全绿（变异测试发现的假门）。
    真实索引里空洞是会出现的：`LexicalIndex.add()` 遇到**分词结果为空**的块会
    直接 `return`，不登记 `doc_id` / `doc_len`，槽位因此被跳过。这里复现那种洞。
    """
    idx = index_mod.LexicalIndex(tokenizer_version="t1")
    idx.add(0, "c-01", "alpha")
    idx.add(2, "c-03", "alpha beta")  # 故意跳过 slot 1
    idx.add(3, "c-04", "beta 龙门")
    assert 1 not in idx.doc_id, "夹具没造出空洞槽位，这条门就测不到兜底分支"

    db = _build(tmp_path, idx)
    backend = fts_backend.Fts5SearchBackend.open(db)
    try:
        upstream = idx.norm_table()
        mine = backend._build_norm()  # noqa: SLF001 - 故意测私有实现的一致性
        assert len(mine) == len(upstream), f"长度不同：{len(mine)} vs {len(upstream)}"
        for slot, (a, b) in enumerate(zip(upstream, mine)):
            assert a == b, f"slot {slot} 的归一化项不同：{a!r} vs {b!r}"
        # 兜底分支单独断言：空洞槽位的值必须等于「长度按 1 计」的那个数，
        # 而不是 0。改成 0 会让该槽位在 BM25 里被过度奖励。
        assert mine[1] != 0.0, "空洞槽位的兜底归一化项是 0"
        assert mine[1] == upstream[1]
    finally:
        backend.close()


def test_bm25_params_come_from_index_module(tmp_path: Path) -> None:
    """参数值必须真来自 `arkrag.index`，而不是本模块自己写死的 1.5 / 0.75。

    做法是**在运行时改一下常量**，看后端行为是否跟着变。若它在别处复制了
    字面量，改常量不会影响它，这条就红。
    """
    idx = _corpus_index()
    db = _build(tmp_path, idx)
    backend = fts_backend.Fts5SearchBackend.open(db)
    try:
        before = backend.search("alpha beta", top_k=10)
        assert before, "夹具语料上应当有命中"
        original = index_mod.BM25_K1
        try:
            index_mod.BM25_K1 = original * 2.0
            backend_after = fts_backend.Fts5SearchBackend.open(db)
            try:
                after = backend_after.search("alpha beta", top_k=10)
            finally:
                backend_after.close()
        finally:
            index_mod.BM25_K1 = original
        # 改 k1 后分数必须变：若没变，说明用的是别的来源的参数
        assert [s for _, s in after] != [s for _, s in before], (
            "改 index.BM25_K1 后分数没变 —— 说明 FTS5 后端没有真的用这个常量"
        )
    finally:
        backend.close()


def test_no_bm25_numeric_literals_in_source() -> None:
    """静态守门：代码里不许出现 BM25 参数的字面量。

    `1.5` / `0.75` 这两个数只允许出现在 `arkrag/index.py` 里（那里是唯一定义处）。
    必须从 `arkrag.index.BM25_K1` / `BM25_B` import。

    ## 为什么用 tokenize 而不是 grep 整行

    第一版写成「去掉 `#` 之后的内容再找子串」，结果被文档里的 `1.5 GB` 判成违规
    ——**这是一个假阳性**。假阳性比漏检更危险：人会把它当噪声，然后放宽规则或删测试。
    所以这里用标准库 `tokenize` 只取**真正的数字字面量**，注释与字符串一律不参与。
    这样门既不会误报，也不会因为「参数写成 1.50」而漏报（`float(tok)==1.5` 也判）。
    """
    import io
    import tokenize as py_tokenize

    source = (ROOT / "src" / "arkrag" / "fts_backend.py").read_text("utf-8")
    banned = {index_mod.BM25_K1: "BM25_K1", index_mod.BM25_B: "BM25_B"}
    offenders: list[str] = []
    for tok in py_tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type != py_tokenize.NUMBER:
            continue
        try:
            value = float(tok.string.rstrip("jJ"))
        except ValueError:
            continue
        if value in banned:
            offenders.append(f"第 {tok.start[0]} 行 {tok.string!r}（= {banned[value]}）")
    assert not offenders, (
        "fts_backend.py 的代码里出现了 BM25 参数字面量；"
        "必须从 arkrag.index.BM25_K1 / BM25_B import：" + "；".join(offenders)
    )


# ---------------------------------------------------------------- 词条与元数据
def test_token_set_identical_to_source_index(tmp_path: Path) -> None:
    """库里的词条集合必须与源索引完全一致。

    多一个词条会让 df 偏小、idf 偏大；少一个会让某些查询命中不了。
    两种情况检索结果都"看起来正常"。
    """
    idx = _corpus_index()
    db = _build(tmp_path, idx)
    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        db_terms = {row[0] for row in conn.execute("SELECT term FROM postings")}
    finally:
        conn.close()
    assert db_terms == set(idx.postings)


def test_doc_len_and_chunk_ids_round_trip(tmp_path: Path) -> None:
    idx = _corpus_index()
    db = _build(tmp_path, idx)
    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        rows = dict((doc, (cid, dl)) for doc, cid, dl in conn.execute("SELECT doc, chunk_id, doc_len FROM chunks"))
    finally:
        conn.close()
    assert len(rows) == idx.size
    for slot, chunk_id in idx.doc_id.items():
        assert rows[slot][0] == chunk_id
        assert rows[slot][1] == idx.doc_len[slot]


def test_avg_len_matches_source(tmp_path: Path) -> None:
    idx = _corpus_index()
    db = _build(tmp_path, idx)
    backend = fts_backend.Fts5SearchBackend.open(db)
    try:
        assert backend.avg_len == idx.avg_len
        assert backend.size == idx.size
    finally:
        backend.close()


def test_all_required_tables_exist_with_rows(tmp_path: Path) -> None:
    """计划 H3 的表清单必须一张不缺。缺表而功能照跑是最难发现的坏。"""
    idx = _corpus_index()
    graph = {
        "entities": [
            {"id": "e-1", "name": "凯尔希", "type": "角色", "desc": "", "mentions": 2,
             "aliases": [{"text": "凯太后"}]},
        ],
        "edges": [
            {"edge_id": "r-1", "subject_id": "e-1", "predicate": "所属", "object_id": "e-2",
             "direction": "out", "support_count": 1, "conflict_count": 0,
             "confidence_summary": {"max": 0.9}},
        ],
        "alias_index": {"凯": "e-1"},
    }
    assertions = [
        {"assertion_id": "a-1", "chunk_id": "c-01", "subject_id": "e-1", "predicate": "所属",
         "object_id": "e-2", "confidence": 0.8, "evidence_text": "凯尔希"},
    ]
    db = _build(tmp_path, idx, graph=graph, assertions=assertions, generation_id="g-test")
    counts = fts_backend.table_counts(db)
    for name in fts_backend.REQUIRED_TABLES:
        assert name in counts, f"缺表 {name}"
        assert counts[name] > 0, f"表 {name} 是空的"
    assert counts["entities"] == 1
    assert counts["relations"] == 1
    assert counts["assertions"] == 1
    # alias_index 与 aliases 数组都该进 aliases 表（两个来源，同一张表）
    assert counts["aliases"] == 2


# ---------------------------------------------------------------- 安全与形态版本
def test_build_refuses_path_inside_generation(tmp_path: Path) -> None:
    """纪律 #6：新产物不许写进世代目录。"""
    idx = _corpus_index()
    inside = tmp_path / "generations" / "g-20260101T000000Z-abcdef" / "fts.sqlite"
    inside.parent.mkdir(parents=True)
    with pytest.raises(ValueError, match="世代目录") as excinfo:
        fts_backend.build_database(inside, index=idx, index_meta={})
    # 报错要指出该写哪里，否则下次还会有人往世代目录里写
    assert "fts" in str(excinfo.value)
    assert not inside.exists(), "被拒绝后不应留下任何文件"


def test_schema_version_mismatch_raises(tmp_path: Path) -> None:
    """库与代码不同源时必须报错，不许按"看起来能跑"继续算分。"""
    idx = _corpus_index()
    db = _build(tmp_path, idx)
    conn = sqlite3.connect(db)
    try:
        conn.execute("UPDATE meta SET value=? WHERE key='fts_schema_version'", ('"ark-rag-fts-999"',))
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(ValueError, match="形态版本") as excinfo:
        fts_backend.Fts5SearchBackend.open(db)
    assert "999" in str(excinfo.value), "报错里要带出库里实际的版本号，便于判断是库旧还是代码旧"


def test_unknown_tokenizer_version_raises(tmp_path: Path) -> None:
    idx = _corpus_index()
    db = _build(tmp_path, idx)
    conn = sqlite3.connect(db)
    try:
        conn.execute("UPDATE meta SET value=? WHERE key='tokenizer_version'", ('"t9"',))
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(ValueError, match="分词版本") as excinfo:
        fts_backend.Fts5SearchBackend.open(db)
    assert "t9" in str(excinfo.value)


def test_open_missing_db_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope.sqlite"
    with pytest.raises(FileNotFoundError) as excinfo:
        fts_backend.Fts5SearchBackend.open(missing)
    # 报错要指向建库脚本，否则调用方只会看到"文件不存在"然后自己去猜
    assert "build_fts_index" in str(excinfo.value)


def test_score_mode_rejects_unknown(tmp_path: Path) -> None:
    idx = _corpus_index()
    db = _build(tmp_path, idx)
    with pytest.raises(ValueError, match="打分模式") as excinfo:
        fts_backend.Fts5SearchBackend.open(db, score_mode="magic")
    # 报错要把合法取值列出来，否则调用方只能去读源码才知道能填什么
    assert "exact" in str(excinfo.value) and "fts5" in str(excinfo.value)
    # 已知模式必须都接受，且自报的模式名要与请求一致
    for mode in fts_backend.SCORE_MODES:
        backend = fts_backend.Fts5SearchBackend.open(db, score_mode=mode)
        try:
            assert backend.describe()["score_mode"] == mode
        finally:
            backend.close()


# ---------------------------------------------------------------- FTS5 原生 bm25（另一套口径）
def test_fts5_native_mode_is_flagged_as_different_scoring(tmp_path: Path) -> None:
    """原生 `bm25()` 是**另一套口径**，必须能自证不同，且不得冒充等价。

    这条不是"要求两边一样"——恰恰相反：它确认两边**确实不一样**，
    于是任何把 `score_mode` 悄悄换成 `fts5` 却仍宣称"排序逐位相同"的报告都是假的。
    差异来自 idf 公式：SQLite 用 `log((N-df+0.5)/(df+0.5))`，
    本项目用 `log(1+(N-df+0.5)/(df+0.5))`。
    """
    idx = _corpus_index()
    db = _build(tmp_path, idx)
    exact = fts_backend.Fts5SearchBackend.open(db, score_mode=fts_backend.SCORE_MODE_EXACT)
    native = fts_backend.Fts5SearchBackend.open(db, score_mode=fts_backend.SCORE_MODE_FTS5)
    try:
        same_scores = 0
        total = 0
        for query in ("alpha", "beta", "alpha beta", "beta 阿米娅"):
            a = exact.search(query, top_k=10)
            b = native.search(query, top_k=10)
            total += 1
            if [s for _, s in a] == [s for _, s in b]:
                same_scores += 1
        assert same_scores < total, (
            "两种模式的分数完全一致 —— 说明 SCORE_MODE_FTS5 其实没走原生 bm25()，"
            "报告里『原生 bm25』那一栏会是假的"
        )
        # 原生模式至少要有结果，不许因为口径不同就返回空
        assert native.search("beta", top_k=10), "原生 bm25 模式返回空结果"
        assert native.describe()["score_mode"] == fts_backend.SCORE_MODE_FTS5
        assert exact.describe()["score_mode"] == fts_backend.SCORE_MODE_EXACT
    finally:
        exact.close()
        native.close()


# ---------------------------------------------------------------- 建库与可重入
def test_build_is_deterministic(tmp_path: Path) -> None:
    """同一份索引建两次，内容（块/词条/文档长度/分数）必须一致。

    体积与字节不要求相同（页分配与 optimize 的时序会差几 KB），
    但**检索行为**必须一致——那才是"同一份索引"的定义。
    """
    idx = _corpus_index()
    db1 = tmp_path / "a.sqlite"
    db2 = tmp_path / "b.sqlite"
    fts_backend.build_database(db1, index=idx, index_meta={})
    fts_backend.build_database(db2, index=idx, index_meta={})
    b1 = fts_backend.Fts5SearchBackend.open(db1)
    b2 = fts_backend.Fts5SearchBackend.open(db2)
    try:
        assert b1.size == b2.size
        assert b1.avg_len == b2.avg_len
        for query in ("alpha", "beta 阿米娅", "龙门"):
            assert b1.search(query, top_k=10) == b2.search(query, top_k=10)
    finally:
        b1.close()
        b2.close()


def test_build_from_generation_refuses_overwrite(tmp_path: Path, monkeypatch) -> None:
    """返修走新产物：已存在时默认拒绝，必须显式 force。"""
    idx = _corpus_index()
    gen_dir = tmp_path / "g-20260101T000000Z-abcdef"
    (gen_dir / "index").mkdir(parents=True)
    index_mod.save_index(idx, {"tokenizer_version": idx.tokenizer_version}, gen_dir / "index" / "lexical.json")

    db = tmp_path / "out.sqlite"
    stats = fts_backend.build_from_generation(gen_dir, db_path=db, with_graph=False)
    assert stats["bytes"] > 0
    with pytest.raises(FileExistsError):
        fts_backend.build_from_generation(gen_dir, db_path=db, with_graph=False)
    # force 才允许覆盖
    again = fts_backend.build_from_generation(gen_dir, db_path=db, with_graph=False, force=True)
    assert again["bytes"] > 0


def test_build_from_generation_requires_index_file(tmp_path: Path) -> None:
    gen_dir = tmp_path / "g-20260101T000000Z-abcdef"
    gen_dir.mkdir()
    target = tmp_path / "x.sqlite"
    with pytest.raises(FileNotFoundError, match="lexical.json") as excinfo:
        fts_backend.build_from_generation(gen_dir, db_path=target)
    assert "lexical.json" in str(excinfo.value)
    assert not target.exists(), "缺源索引时不应产出任何库文件"


def test_open_is_read_only(tmp_path: Path) -> None:
    """后端只读打开：意外写入应当直接失败，而不是悄悄改库。"""
    idx = _corpus_index()
    db = _build(tmp_path, idx)
    backend = fts_backend.Fts5SearchBackend.open(db)
    try:
        with pytest.raises(sqlite3.OperationalError) as excinfo:
            backend._conn.execute("INSERT INTO meta(key, value) VALUES('x','1')")  # noqa: SLF001
        assert "readonly" in str(excinfo.value).lower(), (
            f"失败原因不是只读约束，而是别的问题：{excinfo.value}"
        )
    finally:
        backend.close()


def test_fts_tokenize_spec_is_valid_sqlite() -> None:
    """`FTS_TOKENIZE` 必须是本机 sqlite 认得的指令。

    `tokenize="unicode61 remove_diacritics 0 tokenchars '...'"` 这种**两个选项
    同时出现**的写法在 sqlite 3.50.4 上报 `parse error in tokenize directive`。
    这条把"本机能不能建"钉成断言，而不是靠人记住。
    """
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(f'CREATE VIRTUAL TABLE t USING fts5(x, tokenize="{fts_backend.FTS_TOKENIZE}")')
        conn.execute("INSERT INTO t(rowid, x) VALUES(1, 'act16 1-7 a.b')")
        conn.execute("CREATE VIRTUAL TABLE v USING fts5vocab(t, 'row')")
        terms = {row[0] for row in conn.execute("SELECT term FROM v")}
    finally:
        conn.close()
    # 词内连接符必须保住：t2 的 token 里 `1-7` 是一个整体，
    # 被 unicode61 默认规则拆成 `1` / `7` 就会与源索引的词条集合对不上。
    assert "1-7" in terms, f"连字符被切开了：{sorted(terms)}"
    assert "a.b" in terms, f"句点被切开了：{sorted(terms)}"
    assert "act16" in terms
