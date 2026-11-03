"""检索后端抽象层的测试（H3 第 ① ② 步）。

## 这些测试要能失败

每个用例都先问过「如果实现退化成最笨的版本，这条会不会红」：

* `test_json_backend_matches_lexical_index_bitwise` —— 如果适配器加了任何"聪明"
  的后处理（重排、截断、round 分数），浮点值或顺序就会有差，用例红。
* `test_open_backend_fts_without_db_raises` —— 如果工厂在缺库时静默退回 JSON，
  用例红。这条最重要：静默降级会让「我测的是新后端」这句话变成假的，
  而两个后端的数看起来都正常。
* `test_auto_picks_fts_only_when_db_exists` —— 如果 `auto` 永远选 JSON，
  或者永远选 FTS，其中必有一条会红。

刻意不写「只把观察结果 append 进列表」的写法：那正是本项目记录过的三次事故
（pytest 全绿而缺陷躺在那里）。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from arkrag import index as index_mod, search_backend

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- 夹具
def _tiny_index() -> "index_mod.LexicalIndex":
    """一个够小、但能区分「按分数排」与「按 chunk_id 排」的索引。

    chunk_id 故意与分数顺序相反（`c-03` 分最高但 id 最大），这样任何
    「顺手按 id 排」的实现都会被抓到。
    """
    idx = index_mod.LexicalIndex(tokenizer_version="t1")
    idx.add(0, "c-01", "alpha")
    idx.add(1, "c-02", "beta 龙门")
    idx.add(2, "c-03", "alpha alpha alpha 凯尔希")
    idx.add(3, "c-04", "alpha 龙门 罗德岛")
    return idx


def _fake_generation(tmp_path: Path, idx: "index_mod.LexicalIndex") -> Path:
    """在临时目录里造一个最小的世代目录（只有 index/lexical.json）。"""
    gen_dir = tmp_path / "g-20260101T000000Z-abcdef"
    index_dir = gen_dir / "index"
    index_dir.mkdir(parents=True)
    meta = {
        "tokenizer_version": idx.tokenizer_version,
        "chunk_count": idx.size,
        "index_content_digest": "deadbeef" * 8,
        "probe_digest": "cafebabe" * 8,
    }
    index_mod.save_index(idx, meta, index_dir / "lexical.json")
    return gen_dir


# ---------------------------------------------------------------- 协议与适配器
def test_json_backend_matches_lexical_index_bitwise(tmp_path: Path) -> None:
    """适配器必须逐位转发：chunk_id 顺序与分数浮点值都不能变。"""
    idx = _tiny_index()
    gen_dir = _fake_generation(tmp_path, idx)
    backend = search_backend.load_json_backend(gen_dir / "index" / "lexical.json")

    for query in ("alpha", "alpha 龙门", "凯尔希 alpha", "不存在的词"):
        expected = idx.search(query, top_k=10)
        got = backend.search(query, top_k=10)
        assert [c for c, _ in got] == [c for c, _ in expected], f"{query!r} 的 chunk_id 顺序变了"
        assert [s for _, s in got] == [s for _, s in expected], f"{query!r} 的分数不是逐位相同"
    assert backend.size == idx.size
    assert backend.avg_len == idx.avg_len
    assert backend.tokenizer_version == idx.tokenizer_version


def test_json_backend_satisfies_protocol(tmp_path: Path) -> None:
    idx = _tiny_index()
    gen_dir = _fake_generation(tmp_path, idx)
    backend = search_backend.load_json_backend(gen_dir / "index" / "lexical.json")
    assert isinstance(backend, search_backend.SearchBackend)
    # 协议里刻意没有 postings —— 有它就会逼所有新后端都造一份内存倒排
    assert not hasattr(search_backend.SearchBackend, "postings")


def test_describe_backend_always_fills_required_fields(tmp_path: Path) -> None:
    """报告里不许出现空描述：没有 describe() 的第三方后端也要有最小描述。"""
    idx = _tiny_index()
    gen_dir = _fake_generation(tmp_path, idx)
    backend = search_backend.load_json_backend(gen_dir / "index" / "lexical.json")
    described = search_backend.describe_backend(backend)
    for key in ("backend", "tokenizer_version", "name_terms", "size", "avg_len"):
        assert key in described, f"描述缺字段 {key}"
    assert described["backend"] == search_backend.BACKEND_JSON
    assert described["size"] == idx.size

    class _Bare:
        """只满足协议、没有 describe() 的假后端。"""

        tokenizer_version = "t1"
        name_terms = frozenset()

        @property
        def size(self) -> int:
            return 7

        @property
        def avg_len(self) -> float:
            return 3.5

        def search(self, query, top_k=20, *, name_terms=None):  # noqa: ANN001, ARG002
            return []

    bare = search_backend.describe_backend(_Bare())
    assert bare["size"] == 7
    assert bare["backend"] == "_Bare"


# ---------------------------------------------------------------- 工厂
def test_open_backend_rejects_unknown_kind(tmp_path: Path) -> None:
    idx = _tiny_index()
    gen_dir = _fake_generation(tmp_path, idx)
    with pytest.raises(ValueError) as excinfo:
        search_backend.open_backend(gen_dir, kind="sqlite")
    # 消息里必须列出合法取值：拼错后端名的人要能立刻看出该写什么
    assert "json" in str(excinfo.value) and "fts" in str(excinfo.value)


def test_open_backend_fts_without_db_raises(tmp_path: Path) -> None:
    """缺库时**必须**抛错。静默退回 JSON 会让双跑对比全是假的。"""
    idx = _tiny_index()
    gen_dir = _fake_generation(tmp_path, idx)
    missing = tmp_path / "fts" / "nope.sqlite"
    assert not missing.exists()
    with pytest.raises(FileNotFoundError) as excinfo:
        search_backend.open_backend(gen_dir, kind=search_backend.BACKEND_FTS, db_path=missing)
    assert "nope.sqlite" in str(excinfo.value)


def test_open_backend_auto_falls_back_to_json_without_db(tmp_path: Path) -> None:
    idx = _tiny_index()
    gen_dir = _fake_generation(tmp_path, idx)
    backend = search_backend.open_backend(
        gen_dir, kind=search_backend.BACKEND_AUTO, db_path=tmp_path / "fts" / "nope.sqlite"
    )
    assert isinstance(backend, search_backend.JsonSearchBackend)


def test_open_backend_auto_picks_fts_when_db_exists(tmp_path: Path) -> None:
    from arkrag import fts_backend

    idx = _tiny_index()
    gen_dir = _fake_generation(tmp_path, idx)
    db_path = tmp_path / "fts" / "tiny.sqlite"
    fts_backend.build_database(db_path, index=idx, index_meta={"index_content_digest": "x"})

    backend = search_backend.open_backend(gen_dir, kind=search_backend.BACKEND_AUTO, db_path=db_path)
    assert isinstance(backend, fts_backend.Fts5SearchBackend), "auto 有库时必须选 fts"
    # 再确认它真的在查库而不是悄悄读了 JSON：库里的分数与 JSON 路径逐位相同，
    # 但类型与 describe() 的 backend 字样必须是 fts。
    assert backend.describe()["backend"] == "fts"
    backend.close()


def test_default_db_path_stays_outside_generation_dirs() -> None:
    """纪律 #6：新产物不许落进 `generations/<gid>/`。"""
    path = search_backend.default_db_path("g-20260101T000000Z-abcdef")
    parts = [p.lower() for p in Path(path).parts]
    assert "generations" not in parts, f"默认产物路径落在世代目录里：{path}"
    assert "fts" in parts
    assert Path(path).name == "g-20260101T000000Z-abcdef.sqlite"


def test_backend_kinds_are_exhaustive() -> None:
    """工厂接受的取值与常量表必须一致（拼错一个名字会静默变成别的后端）。"""
    assert set(search_backend.BACKEND_KINDS) == {"json", "fts", "auto"}
    assert search_backend.BACKEND_JSON in search_backend.BACKEND_KINDS
    assert search_backend.BACKEND_FTS in search_backend.BACKEND_KINDS
