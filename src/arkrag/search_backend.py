"""检索后端抽象（H3）：让「词法检索」这一层可以被整体替换而不动调用方。

## 为什么需要这一层

H1/H2 剖析的结论是：`LexicalIndex` 自身的算法没错（H2 已经把单次检索 P95 压到
167 ms），真正的问题是**每次冷启动都要把 127 MB 的 `lexical.json` 解析成 Python
对象**（实测 8.6–9.7 s，占冷加载 12.1 s 的 82%），以及那份对象本身常驻内存。

要换掉这个存储，就必须先有一个**调用方只认接口、不认实现**的缝。否则「换后端」
会变成在 `ask.py` / `retrieve.py` 里到处 `if 后端 == ...`，而这两个文件正是本轮
最不该被反复改动的部分。

## 接口小到什么程度是刻意的

`src/arkrag/retrieve.py` 对索引的依赖只有一个方法（`Retriever.search` 里两处
`self.index.search(text, top_k=...)`），外加两个只读属性
（`tokenizer_version` / `name_terms`，`build_context` 用它算 token 数）。
所以协议就按这个**实际用到的面**定，不多不少：

* `search()` —— 唯一的检索入口，返回 `[(chunk_id, score)]`，同分按 chunk_id 升序。
* `tokenizer_version` / `name_terms` —— 口径自证：查询侧必须用建库时同一份分词器
  与同一份整词名表，否则倒排里的整词名 token 永远查不到，而它们的 `doc_len`
  贡献仍在，等于凭空抬高每个文档的长度（`index.py` 里那段注释是这个坑的原始记录）。
* `size` / `avg_len` —— 便于诊断与复算 BM25 的 idf / 长度归一化。

协议里刻意**不**放 `postings` / `doc_len` 这类实现细节：一旦放进去，任何新后端都
被迫造一份内存倒排，那就等于什么都没换。

## 后端怎么选

工厂 `open_backend()` 支持三种取值：

* `"json"` —— 旧路径：`arkrag.index.load_index()` 读 127 MB JSON。**必须保留**
  （计划 H3 第 ⑧ 步「保留旧 Generation 回滚」）。
* `"fts"` —— 新路径：`arkrag.fts_backend.Fts5SearchBackend` 读 `.sqlite`。
* `"auto"` —— 有 `.sqlite` 就用 fts，没有就退回 json。

`auto` 的默认方向是「有新品就用新品、没有就老路照跑」，而不是「没有就报错」：
回滚时人只要删掉/改掉 `--backend json` 就能退回去，不需要同时改代码。
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from . import config, index as index_mod

# 后端名。写成常量而不是散落的字面量，是因为它们会出现在命令行、报告 JSON
# 与测试断言里三处；拼错一次就会静默退回自动选择，而"跑的是哪个后端"正是
# 双跑对比最不能弄错的一件事。
BACKEND_JSON = "json"
BACKEND_FTS = "fts"
BACKEND_AUTO = "auto"
BACKEND_KINDS: tuple[str, ...] = (BACKEND_JSON, BACKEND_FTS, BACKEND_AUTO)


@runtime_checkable
class SearchBackend(Protocol):
    """检索后端协议。`runtime_checkable` 只校验方法/属性存在，不校验语义。"""

    #: 这个后端**自己**的分词版本，不是当前代码的默认版本
    tokenizer_version: str
    #: 建库时注入的整词名表（查询侧必须用同一份）
    name_terms: frozenset[str]

    @property
    def size(self) -> int:
        """可检索的文档数（= 进索引的块数）。"""
        ...

    @property
    def avg_len(self) -> float:
        """平均文档长度，BM25 长度归一化用。"""
        ...

    def search(
        self,
        query: str,
        top_k: int = 20,
        *,
        name_terms: "set[str] | frozenset[str] | None" = None,
    ) -> list[tuple[str, float]]:
        """检索。返回 `[(chunk_id, score)]`，按分数降序、同分按 chunk_id 升序。"""
        ...


class JsonSearchBackend:
    """旧路径适配器：把 `LexicalIndex` 包成协议，行为**逐位不变**。

    这个类刻意只做转发，不加任何逻辑。加逻辑就有两个实现要对齐，而
    「同一判定写两处」正是 `tests/test_single_source_of_truth.py` 记录的三次
    事故的机理。转发是唯一不会漂移的写法。

    `search()` 的签名与 `LexicalIndex.search` 一致，所以双跑对比里
    「旧后端的数」就是 H1/H2 已经测过的那些数，可以逐项对上。
    """

    def __init__(self, index: "index_mod.LexicalIndex", *, source_path: Path | None = None) -> None:
        self._index = index
        self.tokenizer_version = index.tokenizer_version
        self.name_terms = index.name_terms
        self.source_path = Path(source_path) if source_path is not None else None

    @property
    def index(self) -> "index_mod.LexicalIndex":
        """底层 `LexicalIndex`。给诊断脚本用，检索路径不依赖它。"""
        return self._index

    @property
    def size(self) -> int:
        return self._index.size

    @property
    def avg_len(self) -> float:
        return self._index.avg_len

    def search(
        self,
        query: str,
        top_k: int = 20,
        *,
        name_terms: "set[str] | frozenset[str] | None" = None,
    ) -> list[tuple[str, float]]:
        return self._index.search(query, top_k=top_k, name_terms=name_terms)

    def describe(self) -> dict:
        """供双跑报告记录的自我描述。"""
        return {
            "backend": BACKEND_JSON,
            "tokenizer_version": self.tokenizer_version,
            "name_terms": len(self.name_terms),
            "size": self.size,
            "avg_len": round(self.avg_len, 6),
            "source_path": str(self.source_path) if self.source_path else "",
        }


def load_json_backend(index_path: Path) -> JsonSearchBackend:
    """从 `lexical.json` 载入旧后端。这是冷启动里那 8.6–9.7 s 的来源。"""
    idx, _meta = index_mod.load_index(Path(index_path))
    return JsonSearchBackend(idx, source_path=index_path)


def open_backend(
    generation_dir: Path,
    *,
    kind: str = BACKEND_AUTO,
    db_path: Path | None = None,
) -> "SearchBackend":
    """按某种后端打开一个世代。**这是接线用的唯一入口。**

    用法（在 `scripts/ask.py::load_generation` 的最后加一行，或调用处覆盖）：

        loaded["index"] = search_backend.open_backend(loaded["generation"].dir, kind="fts")

    `generation_dir` 是世代目录（含 `index/lexical.json`）。`db_path` 默认取
    `config.artifact_root() / "fts" / "<世代目录名>.sqlite"`。

    `kind="auto"` 时：`.sqlite` 存在就用 fts，否则退回 json。**不静默降级**——
    选了 `"fts"` 而库不存在就直接抛 `FileNotFoundError`，因为"以为在测新后端、
    其实还在跑旧后端"会让双跑对比得出完全错误的结论。
    """
    generation_dir = Path(generation_dir)
    kind = str(kind or BACKEND_AUTO)
    if kind not in BACKEND_KINDS:
        raise ValueError(f"未知后端 {kind!r}，只支持 {BACKEND_KINDS}")

    if kind == BACKEND_AUTO:
        candidate = Path(db_path) if db_path is not None else default_db_path(generation_dir.name)
        kind = BACKEND_FTS if candidate.exists() else BACKEND_JSON

    if kind == BACKEND_JSON:
        return load_json_backend(generation_dir / "index" / "lexical.json")

    # 延迟 import：只跑旧路径时不必加载 sqlite 相关代码
    from . import fts_backend

    target = Path(db_path) if db_path is not None else default_db_path(generation_dir.name)
    return fts_backend.Fts5SearchBackend.open(target)


def default_db_path(generation_id: str) -> Path:
    """新后端的默认产物位置：`<artifact_root>/fts/<generation_id>.sqlite`。

    刻意**不在** `generations/<gid>/` 里面：世代目录是发布产物、有内容摘要与
    legacy 写保护，把新索引塞进去会让「世代目录摘要」变化，从而让
    `scripts/snapshot_baseline.py` 与 `release_cli verify` 变红。
    """
    return config.artifact_root() / "fts" / f"{generation_id}.sqlite"


def describe_backend(backend: "SearchBackend") -> dict:
    """统一描述。没有 `describe()` 的后端也给一份最小描述，报告里不留空。"""
    describe = getattr(backend, "describe", None)
    if callable(describe):
        return dict(describe())
    return {
        "backend": type(backend).__name__,
        "tokenizer_version": getattr(backend, "tokenizer_version", ""),
        "name_terms": len(getattr(backend, "name_terms", ()) or ()),
        "size": int(getattr(backend, "size", 0) or 0),
        "avg_len": round(float(getattr(backend, "avg_len", 0.0) or 0.0), 6),
        "source_path": "",
    }
