"""SQLite FTS5 检索后端（H3）：把「127 MB JSON + Python 层 BM25」换成「一个 .sqlite」。

## 这一层到底换了什么

不变的是**打分口径**：idf、长度归一化、同分按 chunk_id 升序，全部与
`arkrag.index.LexicalIndex.search` 逐位相同，BM25 参数直接从
`arkrag.index.BM25_K1` / `BM25_B` import（本项目明令禁止在别处复制这两个字面量，
见 `tests/test_single_source_of_truth.py` 记录的事故）。

变的是**倒排住在哪**：

| | 旧（`lexical.json`） | 新（`.sqlite`） |
|---|---|---|
| 冷启动 | 解析 127 MB JSON → Python 对象，实测 8.6–9.7 s | `sqlite3.connect()` 打开，实测 17–235 ms |
| 内存 | 694,304 个 token × 各自的 dict，实测常驻约 1.5 GB | 只有 `doc_ids` / `doc_len` / `norm` 三张扁平表（13,781 条） |
| 单次检索 | 扫 13–23 万条 postings，Python 逐对算 BM25 | 同左（口径一致），取数走 BLOB 反序列化，**实测比内存 dict 慢** |

**关于最后一行，必须说清，这是个反直觉的实测结果。** 同一进程里交替切换两个后端、
同一批 320 题、`top_k=60`，实测：

| 后端 | 检索 P50 | 检索 P95 |
| --- | --- | --- |
| `lexical.json`（内存 dict） | 86.3 – 101.7 ms | 130.0 – 146.0 ms |
| `.sqlite`（本模块 exact 模式） | 132.4 – 138.9 ms | 199.6 – 203.0 ms |

**FTS 后端在「单次检索」这一项上更慢（P50 +27%~+60%，P95 +38%~+56%）。**
原因不难理解：json 后端装好之后 `postings` 就是一个热 dict，`dict.get(term)` 直接命中，
Python 层没有任何反序列化；而本模块每题要为命中的 ~74 个 term 各做一次
`array("I").frombytes()` 重建数组对象，再走 BLOB 读取，这笔开销超过了「少扫一点」的收益。

**所以 H3 的收益不在检索延迟，而在冷启动与内存**（见 `docs/h3-FTS5迁移.md` §5）。
不要因为「检索慢了」就以为迁移失败，也不要反过来把冷启动的收益算成检索变快。

## 两种打分模式，默认 `exact`

* `SCORE_MODE_EXACT`（默认）—— **FTS5 只当倒排存储**。打分仍由本模块按
  `LexicalIndex.search` 的公式逐位复算，因此 `top_k` 列表与旧后端**完全一致**，
  可以在测试里直接断言「两边 chunk_id 列表与分数浮点值都相同」。这是可验证的
  那条路，也是默认值。
* `SCORE_MODE_FTS5` —— 直接调 FTS5 原生 `bm25()`。**已实现，但实测两头都不占**：
  与 exact 模式同一进程交替实测 P50 110–116 ms / P95 214–219 ms，**比 exact 模式
  （P50 132–139 / P95 200–203）在 P50 上略快、P95 上更慢**，且
  **idf 公式与现有实现不同**：SQLite 用
  `log((N - df + 0.5) / (df + 0.5))`，本项目用 `log(1 + (N - df + 0.5) / (df + 0.5))`
  = `log((N + 1) / (df + 0.5))`。分子不同，排序必然不同。
  实测 320 题 top-20 与 json 后端的**平均重合率仅 0.5919**。
  结论：**不作默认，也不作推荐**。它是一个独立的打分口径，要用就得单独报召回数字；
  想拿它「换速度」是不成立的——两个口径的检索耗时都在同一个量级。

## 为什么 `chunk_texts` 存的是「token 串」而不是原文

SQLite 没有中文分词器（`unicode61` 会把整段汉字当一个词）。若把原文直接塞进
FTS5，中文查询会**一条都命中不了**。所以写入的是
`tokenize()` 已经产出的 token，用空格拼起来，FTS5 侧只负责建 b-tree。
这样分词口径 100% 由 `arkrag.index.tokenize` 决定，FTS5 不参与切词，
两种模式看到的词条集合与旧后端完全相同（构建后逐个 token 校验过）。

同理，`chunk_texts` 里 token **按词条排序**而不是按原文顺序：bm25 只需要词频，
不需要位置；本项目也没有短语查询。代价是失去了位置信息（若要支持短语检索要改），
这一点写在限制里，不藏着。

## 建库为什么从 `lexical.json` 读，而不是重新分词

因为「重新分词」意味着在第二个地方实现一遍「哪些块进索引、用哪个分词器、
用哪份整词名表、文本怎么拼（title + chapter + text）」。那正是纪律 #2 禁止的
「同一判定写两处」——只要哪天两边有一处改动忘了同步，两份索引就会静默不同，
而**没有任何门能发现**（因为各自的检索都自洽）。

从已发布的 `lexical.json` 翻译，等价性由构造成立：同一份 postings、同一份
`doc_len`、同一份 `doc_id`。代价是建库时要先把 127 MB 解析一遍（8.6 s，一次性），
这是值得的：它换来「不需要证明两个分词器一致」。
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import time
import uuid
from array import array
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from . import index as index_mod, legacy

# 数据库形态版本。表结构或语义一改就要升号：旧库被新代码打开时能一眼看出
# 「这份字节不是这套代码写的」，而不是用错语义算出看似正常的分数。
FTS_SCHEMA_VERSION = "ark-rag-fts-1"

# FTS5 的分词指令。**不用 `unicode61` 默认的空格+标点切分来做中文**，
# 这里只是把已经切好的 token 串按空格读回来，所以真正需要保证的是
# 「写进去的 token 与取出来时的字面量一致」。
#
# `tokenchars '_-./:'` 是给 t2 用的：t2 的 token 允许词内连接符
# （`ACT16`、`1-7`、`A.B`），若让 unicode61 按默认规则再切一次，
# `1-7` 会被拆成 `1` 和 `7`，两边词条集合就对不上了。
# 实测 `tokenize="unicode61 remove_diacritics 0 tokenchars '...'"` 这种
# **两个选项同时出现**的写法在本机 sqlite 3.50.4 上报
# `parse error in tokenize directive`，只能二选一；这里选 tokenchars——
# token 完整性比去掉变音符重要（本项目 token 已统一 lower()，且以中文与
# ASCII 编号为主）。
FTS_TOKENIZE = "unicode61 tokenchars '_-./:'"

# 打分模式。
SCORE_MODE_EXACT = "exact"
SCORE_MODE_FTS5 = "fts5"
SCORE_MODES: tuple[str, ...] = (SCORE_MODE_EXACT, SCORE_MODE_FTS5)

# 建库时每个事务提交的 postings 行数。太大则内存尖峰高，太小则 fsync 次数多。
_WRITE_BATCH = 20000

# 表名清单。测试逐表断言存在，避免「建表漏了一张、功能照样跑」。
REQUIRED_TABLES: tuple[str, ...] = (
    "meta",
    "chunks",
    "chunk_texts",
    "postings",
    "entities",
    "aliases",
    "assertions",
    "relations",
    "artifacts",
    "runs",
)


# ---------------------------------------------------------------- 建库
def _connect_for_write(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    # 建库是一次性批处理：关掉回滚日志与同步能省掉大量 fsync。
    # 产物是可通过 `build_fts_index.py` 重建的派生物，不需要崩溃可恢复性。
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA cache_size=-262144")  # 256 MiB page cache
    return conn


def _create_schema(conn: sqlite3.Connection) -> None:
    """按计划 H3 的表清单建表。

    `chunk_texts` 用 `content=''`（无正文模式）：我们只需要倒排，
    不需要把 token 串再存一遍——那会让库大出 40% 而没有任何查询会去读它。
    无正文模式的代价是不支持 `highlight()` / `snippet()`，本项目不用。
    """
    conn.executescript(
        f"""
        CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;

        CREATE TABLE chunks(
            doc        INTEGER PRIMARY KEY,   -- 槽位，与旧索引的 slot 逐一对齐
            chunk_id   TEXT NOT NULL UNIQUE,
            doc_id     TEXT NOT NULL DEFAULT '',
            scene_id   TEXT NOT NULL DEFAULT '',
            title      TEXT NOT NULL DEFAULT '',
            chapter    TEXT NOT NULL DEFAULT '',
            doc_len    INTEGER NOT NULL       -- 该块的 token 总数（BM25 长度归一化用）
        );

        -- FTS5 虚拟表：存 token 串，仅供 SCORE_MODE_FTS5 与人工排查使用。
        CREATE VIRTUAL TABLE chunk_texts USING fts5(
            tokens,
            content='',
            tokenize="{FTS_TOKENIZE}"
        );

        -- 倒排本体：一个 token 一行，文档号与词频各存一个紧凑数组。
        -- 为什么不是 (term, doc, freq) 三列的普通表：那样每题要回传十几万行
        -- 交给 Python 逐行组装，实测比读内存 dict 还慢（见文档「原型淘汰记录」）。
        -- BLOB 只回传 74 个 token 的几 MB 字节，反序列化在 C 层完成。
        CREATE TABLE postings(
            term  TEXT PRIMARY KEY,
            docs  BLOB NOT NULL,   -- array('I')：升序槽位
            freqs BLOB NOT NULL    -- array('I')：与 docs 一一对应的词频
        ) WITHOUT ROWID;

        CREATE TABLE entities(
            entity_id TEXT PRIMARY KEY,
            name      TEXT NOT NULL DEFAULT '',
            type      TEXT NOT NULL DEFAULT '',
            desc      TEXT NOT NULL DEFAULT '',
            mentions  INTEGER NOT NULL DEFAULT 0,
            aliases   TEXT NOT NULL DEFAULT '[]'   -- JSON 数组，仅作展示，检索不读
        ) WITHOUT ROWID;

        CREATE TABLE aliases(
            alias     TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            PRIMARY KEY(alias, entity_id)
        ) WITHOUT ROWID;

        CREATE TABLE assertions(
            assertion_id   TEXT PRIMARY KEY,
            chunk_id       TEXT NOT NULL DEFAULT '',
            doc_id         TEXT NOT NULL DEFAULT '',
            scene_id       TEXT NOT NULL DEFAULT '',
            subject_id     TEXT NOT NULL DEFAULT '',
            predicate      TEXT NOT NULL DEFAULT '',
            object_id      TEXT NOT NULL DEFAULT '',
            direction      TEXT NOT NULL DEFAULT '',
            confidence     REAL,
            evidence_start INTEGER,
            evidence_end   INTEGER,
            evidence_text  TEXT NOT NULL DEFAULT '',
            qualifiers     TEXT NOT NULL DEFAULT '{{}}'
        ) WITHOUT ROWID;

        CREATE TABLE relations(
            edge_id        TEXT PRIMARY KEY,
            subject_id     TEXT NOT NULL DEFAULT '',
            predicate      TEXT NOT NULL DEFAULT '',
            object_id      TEXT NOT NULL DEFAULT '',
            direction      TEXT NOT NULL DEFAULT '',
            support_count  INTEGER,
            conflict_count INTEGER,
            confidence_max REAL
        ) WITHOUT ROWID;

        CREATE TABLE artifacts(
            role        TEXT PRIMARY KEY,
            path        TEXT NOT NULL DEFAULT '',
            artifact_id TEXT NOT NULL DEFAULT '',
            bytes       INTEGER,
            sha256      TEXT NOT NULL DEFAULT ''
        ) WITHOUT ROWID;

        CREATE TABLE runs(
            run_id        TEXT PRIMARY KEY,
            generation_id TEXT NOT NULL DEFAULT '',
            created_at    TEXT NOT NULL DEFAULT '',
            status        TEXT NOT NULL DEFAULT '',
            graph_schema  TEXT NOT NULL DEFAULT '',
            git_revision  TEXT NOT NULL DEFAULT ''
        ) WITHOUT ROWID;

        CREATE INDEX chunks_by_chunk_id ON chunks(chunk_id);
        CREATE INDEX assertions_by_chunk ON assertions(chunk_id);
        CREATE INDEX assertions_by_subject ON assertions(subject_id);
        CREATE INDEX relations_by_subject ON relations(subject_id);
        CREATE INDEX relations_by_object ON relations(object_id);
        CREATE INDEX aliases_by_entity ON aliases(entity_id);
        """
    )


def _write_meta(conn: sqlite3.Connection, meta: dict[str, Any]) -> None:
    rows = [(str(k), json.dumps(v, ensure_ascii=False, sort_keys=True)) for k, v in meta.items()]
    conn.executemany("INSERT INTO meta(key, value) VALUES(?, ?)", rows)


def build_database(
    db_path: Path,
    *,
    index: "index_mod.LexicalIndex",
    index_meta: dict,
    graph: dict | None = None,
    assertions: Iterable[dict] | None = None,
    manifest: dict | None = None,
    generation_id: str = "",
) -> dict:
    """把一份**已存在的** `LexicalIndex` 翻译成 .sqlite。返回构建统计。

    这是一个纯函数式的落盘：输入是内存里的索引对象，输出是一个文件。
    它不关心索引是怎么来的，所以测试可以喂一个十几条的小索引进来，
    不必碰 40 MB 的块文件。
    """
    db_path = Path(db_path)
    _refuse_inside_generation(db_path)
    legacy.guard_write(db_path, reason="fts_backend.build_database")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # 先写临时文件再原子替换：中途失败不会留下半张表，
    # 而"半张表能打开、能查、只是少了一半文档"是最难发现的那种坏。
    tmp = db_path.with_name(f"{db_path.name}.tmp-{uuid.uuid4().hex[:8]}")
    t_start = time.perf_counter()
    stats: dict[str, Any] = {"db_path": str(db_path), "tables": {}}
    try:
        conn = _connect_for_write(tmp)
        try:
            _create_schema(conn)

            # --- chunks + postings
            t0 = time.perf_counter()
            slots = sorted(index.doc_id)
            conn.executemany(
                "INSERT INTO chunks(doc, chunk_id, doc_len) VALUES(?,?,?)",
                [(s, index.doc_id[s], int(index.doc_len.get(s, 1))) for s in slots],
            )
            chunk_rows = len(slots)

            # 同时按文档累积 token 串（顺序无所谓：bm25 只用词频）。
            n_slots = (max(slots) + 1) if slots else 0
            doc_tokens: list[list[str]] = [[] for _ in range(n_slots)]
            batch: list[tuple[str, bytes, bytes]] = []
            t_pack = time.perf_counter()
            for term, posting in index.postings.items():
                docs = array("I")
                freqs = array("I")
                ap_doc, ap_freq = docs.append, freqs.append
                for slot, freq in posting.items():
                    ap_doc(slot)
                    ap_freq(int(freq))
                    doc_tokens[slot].append(term)
                batch.append((term, docs.tobytes(), freqs.tobytes()))
                if len(batch) >= _WRITE_BATCH:
                    conn.executemany("INSERT INTO postings(term, docs, freqs) VALUES(?,?,?)", batch)
                    batch.clear()
            if batch:
                conn.executemany("INSERT INTO postings(term, docs, freqs) VALUES(?,?,?)", batch)
                batch.clear()
            pack_ms = (time.perf_counter() - t_pack) * 1000.0
            stats["tables"]["chunks"] = chunk_rows
            stats["tables"]["postings"] = len(index.postings)

            # --- chunk_texts（FTS5）。写入的是 token 串，并重复 token 以保住词频：
            # bm25 的 f(qi,D) 就是靠同一 token 在串里出现的次数算的。
            t0 = time.perf_counter()
            conn.executemany(
                "INSERT INTO chunk_texts(rowid, tokens) VALUES(?,?)",
                ((slot, " ".join(toks)) for slot, toks in enumerate(doc_tokens) if toks),
            )
            fts_ms = (time.perf_counter() - t0) * 1000.0
            stats["tables"]["chunk_texts"] = sum(1 for toks in doc_tokens if toks)
            stats["tokens_written"] = sum(len(toks) for toks in doc_tokens)
            del doc_tokens

            # --- 图相关的三张表（检索不走它们，但计划 H3 的表清单要求，
            # 且它们是将来把图查询也搬进 SQL 的前提）
            t0 = time.perf_counter()
            graph = graph or {}
            ent_rows = []
            alias_rows = []
            for ent in graph.get("entities") or []:
                eid = str(ent.get("id") or "")
                if not eid:
                    continue
                aliases = ent.get("aliases") or []
                ent_rows.append(
                    (
                        eid,
                        str(ent.get("name") or ""),
                        str(ent.get("type") or ""),
                        str(ent.get("desc") or ""),
                        int(ent.get("mentions") or 0),
                        json.dumps(aliases, ensure_ascii=False),
                    )
                )
                for item in aliases:
                    text = item.get("text") if isinstance(item, dict) else item
                    text = str(text or "").strip()
                    if text:
                        alias_rows.append((text, eid))
            # `alias_index` 是图里已算好的「别名 -> 实体」，把它并进去：
            # 两代图的别名结构不同，`build_alias_index` 已经处理过这类分派，
            # 这里不再自己判断一遍（纪律 #2）。
            for alias, eid in (graph.get("alias_index") or {}).items():
                alias = str(alias or "").strip()
                if alias:
                    alias_rows.append((alias, str(eid)))
            conn.executemany(
                "INSERT OR REPLACE INTO entities(entity_id,name,type,desc,mentions,aliases)"
                " VALUES(?,?,?,?,?,?)",
                ent_rows,
            )
            conn.executemany("INSERT OR IGNORE INTO aliases(alias, entity_id) VALUES(?,?)", alias_rows)

            rel_rows = []
            for edge in graph.get("edges") or []:
                summary = edge.get("confidence_summary") or {}
                rel_rows.append(
                    (
                        str(edge.get("edge_id") or ""),
                        str(edge.get("subject_id") or edge.get("source") or ""),
                        str(edge.get("predicate") or edge.get("relation") or ""),
                        str(edge.get("object_id") or edge.get("target") or ""),
                        str(edge.get("direction") or ""),
                        int(edge.get("support_count") or 0),
                        int(edge.get("conflict_count") or 0),
                        float(summary.get("max") or 0.0) if isinstance(summary, dict) else None,
                    )
                )
            rel_rows = [r for r in rel_rows if r[0]]
            conn.executemany(
                "INSERT OR REPLACE INTO relations(edge_id,subject_id,predicate,object_id,"
                "direction,support_count,conflict_count,confidence_max) VALUES(?,?,?,?,?,?,?,?)",
                rel_rows,
            )

            asrt_rows = []
            for row in assertions or ():
                asrt_rows.append(
                    (
                        str(row.get("assertion_id") or ""),
                        str(row.get("chunk_id") or ""),
                        str(row.get("doc_id") or ""),
                        str(row.get("scene_id") or ""),
                        str(row.get("subject_id") or ""),
                        str(row.get("predicate") or ""),
                        str(row.get("object_id") or ""),
                        str(row.get("direction") or ""),
                        float(row["confidence"]) if row.get("confidence") is not None else None,
                        int(row["evidence_start"]) if row.get("evidence_start") is not None else None,
                        int(row["evidence_end"]) if row.get("evidence_end") is not None else None,
                        str(row.get("evidence_text") or ""),
                        json.dumps(row.get("qualifiers") or {}, ensure_ascii=False, sort_keys=True),
                    )
                )
            asrt_rows = [r for r in asrt_rows if r[0]]
            conn.executemany(
                "INSERT OR REPLACE INTO assertions(assertion_id,chunk_id,doc_id,scene_id,"
                "subject_id,predicate,object_id,direction,confidence,evidence_start,"
                "evidence_end,evidence_text,qualifiers) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                asrt_rows,
            )
            graph_ms = (time.perf_counter() - t0) * 1000.0
            stats["tables"]["entities"] = len(ent_rows)
            stats["tables"]["aliases"] = len(alias_rows)
            stats["tables"]["relations"] = len(rel_rows)
            stats["tables"]["assertions"] = len(asrt_rows)

            # --- artifacts / runs（身份留痕：这份 .sqlite 对着哪一份产物建的）
            manifest = manifest or {}
            artifact_rows = [
                (role, "", str(manifest.get(key) or ""), None, "")
                for role, key in (
                    ("graph", "graph_artifact_id"),
                    ("index", "index_artifact_id"),
                    ("assertion", "assertion_artifact_id"),
                    ("adjacency", "adjacency_artifact_id"),
                )
            ]
            conn.executemany(
                "INSERT OR REPLACE INTO artifacts(role,path,artifact_id,bytes,sha256)"
                " VALUES(?,?,?,?,?)",
                artifact_rows,
            )
            conn.execute(
                "INSERT OR REPLACE INTO runs(run_id,generation_id,created_at,status,"
                "graph_schema,git_revision) VALUES(?,?,?,?,?,?)",
                (
                    str(manifest.get("run_id") or ""),
                    generation_id or str(manifest.get("generation_id") or ""),
                    str(manifest.get("created_at") or ""),
                    str(manifest.get("status") or ""),
                    str(manifest.get("graph_schema") or ""),
                    str(manifest.get("git_revision") or ""),
                ),
            )
            stats["tables"]["artifacts"] = len(artifact_rows)
            stats["tables"]["runs"] = 1

            _write_meta(
                conn,
                {
                    "fts_schema_version": FTS_SCHEMA_VERSION,
                    "generation_id": generation_id or str(manifest.get("generation_id") or ""),
                    "tokenizer_version": index.tokenizer_version,
                    "name_terms": sorted(index.name_terms),
                    "doc_count": index.size,
                    "avg_len": repr(index.avg_len),
                    # BM25 参数记的是**产生这份库时**代码里的值，供追溯；
                    # 检索时仍从 arkrag.index import，不从库里读——
                    # 否则改常量后旧库会继续用旧值，两种后端就悄悄分叉了。
                    "bm25_k1": repr(index_mod.BM25_K1),
                    "bm25_b": repr(index_mod.BM25_B),
                    "index_fields": list(index_mod.INDEX_FIELDS),
                    "index_content_digest": str(index_meta.get("index_content_digest") or ""),
                    "probe_digest": str(index_meta.get("probe_digest") or ""),
                    "index_build_fp": str(index_meta.get("index_build_fp") or ""),
                    "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "built_by": "scripts/build_fts_index.py",
                },
            )
            conn.commit()
            # 合并 b-tree 段：批量插入后段很多，检索要逐段扫。
            conn.execute("INSERT INTO chunk_texts(chunk_texts) VALUES('optimize')")
            conn.commit()
        finally:
            conn.close()

        os.replace(tmp, db_path)
    except BaseException:
        for leftover in (tmp, tmp.with_name(tmp.name + "-journal")):
            try:
                leftover.unlink()
            except OSError:
                pass
        raise

    stats["build_ms"] = round((time.perf_counter() - t_start) * 1000.0, 3)
    stats["pack_ms"] = round(pack_ms, 3)
    stats["fts_write_ms"] = round(fts_ms, 3)
    stats["graph_write_ms"] = round(graph_ms, 3)
    stats["bytes"] = db_path.stat().st_size
    stats["mib"] = round(stats["bytes"] / 1048576, 3)
    return stats


def _refuse_inside_generation(db_path: Path) -> None:
    """新产物必须落在世代目录之外（纪律 #6）。

    世代目录是发布产物、带内容摘要；往里塞一份 200 MB 的 .sqlite 会让
    `snapshot_baseline.py` / `release_cli verify` 的目录摘要变化，
    于是「重建了一次派生物」看起来像「发布产物被篡改了」。
    """
    resolved = Path(db_path).resolve()
    parts = [p.lower() for p in resolved.parts]
    if "generations" in parts:
        raise ValueError(
            f"拒绝把 FTS 索引写进世代目录：{resolved}。"
            "新产物应写进 data/arkrag/fts/<generation_id>.sqlite。"
        )


def build_from_generation(
    generation_dir: Path,
    *,
    db_path: Path | None = None,
    force: bool = False,
    with_graph: bool = True,
) -> dict:
    """从世代目录（含 `index/lexical.json`）建一份 .sqlite。

    `with_graph=False` 时跳过实体/断言/关系三张表——纯粹为了在建库脚本里
    提供「只要倒排」的快速路径（表结构仍然完整，只是空表）。
    """
    generation_dir = Path(generation_dir)
    index_path = generation_dir / "index" / "lexical.json"
    if not index_path.exists():
        raise FileNotFoundError(f"世代缺少 index/lexical.json：{index_path}")

    target = Path(db_path) if db_path is not None else _default_db_path(generation_dir.name)
    if target.exists() and not force:
        raise FileExistsError(f"{target} 已存在；确需重建请加 --force（返修走新产物，不覆盖旧产物）")

    index, index_meta = index_mod.load_index(index_path)

    graph = assertions = manifest = None
    if with_graph:
        graph_path = generation_dir / "graph" / "graph.json"
        if graph_path.exists():
            graph = json.loads(graph_path.read_text("utf-8"))
        asrt_path = generation_dir / "graph" / "assertions.jsonl"
        if asrt_path.exists():
            assertions = _read_jsonl(asrt_path)
    manifest_path = generation_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text("utf-8"))

    return build_database(
        target,
        index=index,
        index_meta=index_meta,
        graph=graph,
        assertions=assertions,
        manifest=manifest,
        generation_id=generation_dir.name,
    )


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _default_db_path(generation_id: str) -> Path:
    from . import config

    return config.artifact_root() / "fts" / f"{generation_id}.sqlite"


# ---------------------------------------------------------------- 读库
class Fts5SearchBackend:
    """SQLite FTS5 后端。构造后行为与 `JsonSearchBackend` 同接口。"""

    def __init__(
        self,
        conn: sqlite3.Connection,
        meta: dict[str, Any],
        *,
        db_path: Path,
        score_mode: str = SCORE_MODE_EXACT,
    ) -> None:
        if score_mode not in SCORE_MODES:
            raise ValueError(f"未知打分模式 {score_mode!r}，只支持 {SCORE_MODES}")
        self._conn = conn
        self._meta = meta
        self.db_path = Path(db_path)
        self.score_mode = score_mode
        self.tokenizer_version = str(meta.get("tokenizer_version") or "")
        if self.tokenizer_version not in index_mod.SUPPORTED_TOKENIZERS:
            raise ValueError(
                f"库里的分词版本 {self.tokenizer_version!r} 不在 "
                f"{index_mod.SUPPORTED_TOKENIZERS}；这份库不能按当前代码重放"
            )
        self.name_terms: frozenset[str] = frozenset(meta.get("name_terms") or ())

        # 扁平表：槽位即下标，比 dict 快，也正好是 norm 表要的形状
        rows = conn.execute("SELECT doc, chunk_id, doc_len FROM chunks ORDER BY doc").fetchall()
        if not rows:
            raise ValueError(f"{self.db_path} 里没有任何块；库是空的或是坏的")
        biggest = max(r[0] for r in rows)
        self._doc_ids: list[str] = [""] * (biggest + 1)
        self._doc_len: list[int] = [0] * (biggest + 1)
        for doc, chunk_id, doc_len in rows:
            self._doc_ids[doc] = chunk_id
            self._doc_len[doc] = int(doc_len)
        self._size = len(rows)
        self._avg_len = (sum(self._doc_len) / self._size) if self._size else 1.0
        self._norm = self._build_norm()

    # 与 `index.LexicalIndex.norm_table()` 同一公式：口径一致是硬要求，
    # 一旦这里另写一遍，两种后端的分数就会在 1e-16 量级上分叉，
    # 而「排序逐位相同」这条验收就不再可断言。
    def _build_norm(self) -> list[float]:
        k1, b = index_mod.BM25_K1, index_mod.BM25_B
        avg_len = self._avg_len
        default = k1 * (1.0 - b + b * 1 / avg_len)
        norm = [default] * len(self._doc_ids)
        for slot, length in enumerate(self._doc_len):
            if length:
                norm[slot] = k1 * (1.0 - b + b * length / avg_len)
        return norm

    @classmethod
    def open(cls, db_path: Path, *, score_mode: str = SCORE_MODE_EXACT) -> "Fts5SearchBackend":
        """只读打开。用 `mode=ro` 是为了让「后端意外写了库」直接报错。"""
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"FTS 索引不存在：{db_path}（先跑 scripts/build_fts_index.py）")
        uri = f"file:{db_path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        try:
            meta = {
                str(k): json.loads(v) for k, v in conn.execute("SELECT key, value FROM meta").fetchall()
            }
        except sqlite3.DatabaseError as exc:
            conn.close()
            raise ValueError(f"{db_path} 不是本模块的 FTS 库：{exc}") from exc
        if str(meta.get("fts_schema_version") or "") != FTS_SCHEMA_VERSION:
            conn.close()
            raise ValueError(
                f"{db_path} 的形态版本是 {meta.get('fts_schema_version')!r}，"
                f"当前代码是 {FTS_SCHEMA_VERSION!r}；请重建索引"
            )
        return cls(conn, meta, db_path=db_path, score_mode=score_mode)

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    def __enter__(self) -> "Fts5SearchBackend":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -------------------------------------------------- 协议实现
    @property
    def size(self) -> int:
        return self._size

    @property
    def avg_len(self) -> float:
        return self._avg_len

    @property
    def meta(self) -> dict[str, Any]:
        return dict(self._meta)

    def search(
        self,
        query: str,
        top_k: int = 20,
        *,
        name_terms: "set[str] | frozenset[str] | None" = None,
    ) -> list[tuple[str, float]]:
        top_k = max(1, int(top_k))
        if self.score_mode == SCORE_MODE_FTS5:
            return self._search_fts5(query, top_k, name_terms=name_terms)
        return self._search_exact(query, top_k, name_terms=name_terms)

    # -------------------------------------------------- exact：口径与旧后端逐位相同
    def _search_exact(
        self,
        query: str,
        top_k: int,
        *,
        name_terms: "set[str] | frozenset[str] | None",
    ) -> list[tuple[str, float]]:
        """复算 `LexicalIndex.search` 的同一公式，数据来自 postings 表。

        三处刻意的写法，都是为了让浮点结果**逐位**相同：
        1. 查询词按 `tokenize()` 产出的首次出现顺序遍历（`Counter` 保序），
           与旧实现 `q_terms.items()` 的顺序一致；
        2. 同一 token 内按槽位升序累加，与旧实现 `posting.items()` 的插入序一致；
        3. 乘除结合顺序照抄：`idf * (freq * k1p1) / (freq + norm)`。
        """
        terms = self.name_terms if name_terms is None else name_terms
        q_terms = Counter(
            index_mod.tokenize(query, version=self.tokenizer_version, name_terms=terms)
        )
        if not q_terms:
            return []

        order = list(q_terms)
        placeholders = ",".join("?" * len(order))
        rows = self._conn.execute(
            f"SELECT term, docs, freqs FROM postings WHERE term IN ({placeholders})", order
        ).fetchall()
        if not rows:
            return []
        blobs = {row[0]: (row[1], row[2]) for row in rows}

        n = max(1, self._size)
        k1 = index_mod.BM25_K1
        k1p1 = k1 + 1.0
        norm = self._norm
        scores: dict[int, float] = {}
        for term in order:  # 顺序即口径，见 docstring
            packed = blobs.get(term)
            if not packed:
                continue
            docs_blob, freqs_blob = packed
            df = len(docs_blob) // 4
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            docs = array("I")
            docs.frombytes(docs_blob)
            freqs = array("I")
            freqs.frombytes(freqs_blob)
            acc = scores
            for i, slot in enumerate(docs):
                freq = freqs[i]
                acc[slot] = acc.get(slot, 0.0) + idf * (freq * k1p1) / (freq + norm[slot])

        ids = self._doc_ids
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], ids[kv[0]]))[:top_k]
        return [(ids[slot], score) for slot, score in ranked]

    # -------------------------------------------------- fts5：原生 bm25（另一套口径）
    def _search_fts5(
        self,
        query: str,
        top_k: int,
        *,
        name_terms: "set[str] | frozenset[str] | None",
    ) -> list[tuple[str, float]]:
        """直接调 FTS5 `bm25()`。**与旧后端不是同一套打分**，见模块 docstring。"""
        terms = self.name_terms if name_terms is None else name_terms
        tokens = index_mod.tokenize(query, version=self.tokenizer_version, name_terms=terms)
        uniq = list(dict.fromkeys(t for t in tokens if t))
        if not uniq:
            return []
        match = " OR ".join(_quote_fts(t) for t in uniq)
        try:
            rows = self._conn.execute(
                "SELECT rowid, bm25(chunk_texts) AS s FROM chunk_texts"
                " WHERE chunk_texts MATCH ? ORDER BY s LIMIT ?",
                (match, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            # 词里全是 FTS5 语法字符时可能整句非法；此时退回逐词 OR 会得到
            # 同样结果，没必要在这里猜。直接如实报空，不假装有结果。
            return []
        ids = self._doc_ids
        # bm25() 是「越小越好」且为负值；取负变成「越大越好」，与协议约定一致。
        return [(ids[rowid], -float(score)) for rowid, score in rows if 0 <= rowid < len(ids)]

    # -------------------------------------------------- 诊断
    def describe(self) -> dict:
        return {
            "backend": "fts",
            "score_mode": self.score_mode,
            "tokenizer_version": self.tokenizer_version,
            "name_terms": len(self.name_terms),
            "size": self.size,
            "avg_len": round(self.avg_len, 6),
            "source_path": str(self.db_path),
            "db_mib": round(self.db_path.stat().st_size / 1048576, 3) if self.db_path.exists() else None,
            "built_at": str(self._meta.get("built_at") or ""),
            "index_content_digest": str(self._meta.get("index_content_digest") or ""),
            "probe_digest": str(self._meta.get("probe_digest") or ""),
        }


def _quote_fts(token: str) -> str:
    """把 token 包成 FTS5 字符串字面量。内部引号翻倍转义。"""
    return '"' + token.replace('"', '""') + '"'


def table_counts(db_path: Path) -> dict[str, int]:
    """各表行数，供建库脚本与测试核对。"""
    conn = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    try:
        out: dict[str, int] = {}
        for name in REQUIRED_TABLES:
            out[name] = int(conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
        return out
    finally:
        conn.close()
