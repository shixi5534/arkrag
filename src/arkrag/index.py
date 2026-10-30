"""L6 索引：词法倒排、别名倒排、图邻接，以及「有嵌入后端才建」的向量索引位。

元数据新鲜度三查是硬要求：索引落后于语料时不报错、只是让新内容检索不到，
是计划书实测过近一半索引失效的那种事故。

## 身份（计划 D3）

此前只有 `graph_fp`：它证明「输入配置相同」，不证明「这份字节就是那次构建产出的」。
现在分开两件事：

* `INDEX_BUILD_FP` —— 构建配置指纹（字段组成、分词版本、模型参数）。表配置。
* `content_digest()` —— 产物内容摘要。表最终字节。

Manifest 同时绑两者：配置变了要重建，内容被换掉了要能发现。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from . import atomic, config, fingerprint, legacy

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# 分词实现版本。任何改变 token 集合的改动都必须升这个号并进指纹——
# 否则「切词变了」和「索引坏了」在指纹上完全一样，只能靠人眼分辨。
TOKENIZER_VERSION = "t1"

# 参与检索的字段。标题与章节进索引是为了让「切城篇里凯尔希说了什么」能命中，
# 只嵌正文会分不清同章节的不同场景（原有注释保留）。
INDEX_FIELDS = ("title", "chapter", "text")


def tokenize(text: str) -> list[str]:
    """中文按单字 + 相邻双字切分，拉丁/数字按词切分：不依赖任何分词库或模型。"""
    out: list[str] = []
    for segment in re.split(r"\s+", text):
        for m in TOKEN_RE.finditer(segment):
            out.append(m.group().lower())
    han = [ch for ch in text if "\u4e00" <= ch <= "\u9fff"]
    out.extend(han)
    out.extend(a + b for a, b in zip(han, han[1:]))
    return out


def index_build_fingerprint(**overrides) -> str:
    """索引构建配置指纹。字段组成、分词版本、BM25 参数都在里面。"""
    parts = {
        "fields": "|".join(INDEX_FIELDS),
        "tokenizer": TOKENIZER_VERSION,
        "bm25_k1": "1.5",
        "bm25_b": "0.75",
        "schema_version": config.SCHEMA_VERSION,
        "normalizer_version": config.NORMALIZER_VERSION,
    }
    parts.update({k: str(v) for k, v in overrides.items()})
    body = "\x1f".join(f"{k}={v}" for k, v in sorted(parts.items()))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def content_digest(index: "LexicalIndex") -> str:
    """产物内容摘要：把倒排本身摘要一遍，而不是摘要文件字节。

    用内容而不是文件字节，是因为 JSON 的键顺序、缩进、浮点格式化都能让同一个
    索引写出两份不同字节。我们要发现的是"索引内容变了"，不是"序列化变了"。
    """
    h = hashlib.sha256()
    for slot in sorted(index.doc_id):
        h.update(f"{slot}\x00{index.doc_id[slot]}\x00{index.doc_len.get(slot, 0)}\n".encode())
    for term in sorted(index.postings):
        h.update(term.encode("utf-8"))
        h.update(b"\x00")
        for slot in sorted(index.postings[term]):
            h.update(f"{slot}:{index.postings[term][slot]},".encode())
        h.update(b"\n")
    return h.hexdigest()


class LexicalIndex:
    def __init__(self) -> None:
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)
        self.doc_len: dict[int, int] = {}
        self.doc_id: dict[int, str] = {}

    def add(self, slot: int, chunk_id: str, text: str) -> None:
        counts = Counter(tokenize(text))
        if not counts:
            return
        self.doc_len[slot] = sum(counts.values())
        self.doc_id[slot] = chunk_id
        for token, freq in counts.items():
            self.postings[token][slot] = freq

    @property
    def size(self) -> int:
        return len(self.doc_id)

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        n = max(1, self.size)
        q_terms = Counter(tokenize(query))
        avg_len = (sum(self.doc_len.values()) / n) if n else 1.0
        k1, b = 1.5, 0.75
        scores: dict[int, float] = defaultdict(float)
        for term, tf in q_terms.items():
            posting = self.postings.get(term)
            if not posting:
                continue
            df = len(posting)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for slot, freq in posting.items():
                length = self.doc_len.get(slot, 1)
                scores[slot] += idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * length / avg_len))
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], self.doc_id[kv[0]]))[:top_k]
        return [(self.doc_id[slot], score) for slot, score in ranked]


def build_adjacency(edges: list[dict]) -> dict[str, list[tuple[str, str, float]]]:
    adj: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    for edge in edges:
        src, dst = edge["source"], edge["target"]
        weight = float(edge.get("confidence") or 0.0)
        adj[src].append((dst, edge["relation"], weight))
        # 检索期游走要双向连通；方向语义由关系名本身携带，不在这里反转事实
        adj[dst].append((src, edge["relation"], weight))
    return {k: sorted(v) for k, v in adj.items()}


def build_alias_index(entities: list[dict]) -> dict[str, str]:
    out: dict[str, str] = {}
    for ent in entities:
        out[ent["name"]] = ent["id"]
        for alias in ent["aliases"]:
            out.setdefault(alias, ent["id"])
    return out


def build(
    chunk_records: list[dict],
    graph: dict,
    corpus_fp: str,
    *,
    graph_artifact_id: str = "",
) -> tuple[LexicalIndex, dict]:
    index = LexicalIndex()
    for slot, rec in enumerate(sorted(chunk_records, key=lambda r: r["chunk_id"])):
        # 块向量/词法文本都带标题前缀：正文常缺主语与场景，只嵌正文会分不清同章节的不同场景
        index.add(slot, rec["chunk_id"], f"{rec.get('title','')} {rec.get('chapter','')} {rec['text']}")
    meta = {
        "schema_version": config.SCHEMA_VERSION,
        "corpus_fp": corpus_fp,
        # graph_fp 是**配置**指纹，graph_artifact_id 是**内容**摘要（计划 D3）。
        # 只绑配置不够：同一份配置重建两次图，指纹一样而图未必一样；
        # 索引必须能回答"我这份倒排是对着哪一份图产物建的"，那只有内容摘要能答。
        "graph_fp": graph.get("fingerprint", ""),
        "graph_artifact_id": graph_artifact_id,
        "prompt_version": config.PROMPT_VERSION,
        "prompt_sha256": fingerprint.prompt_digest(),
        "vocab_sha": fingerprint.vocab_digest(),
        "normalizer_version": config.NORMALIZER_VERSION,
        # 配置指纹与内容摘要分开记：前者表"该不该重建"，后者表"字节是不是这些字节"
        "index_build_fp": index_build_fingerprint(),
        "tokenizer_version": TOKENIZER_VERSION,
        "index_fields": list(INDEX_FIELDS),
        "index_content_digest": content_digest(index),
        "probe_digest": probe_digest(index),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "chunk_count": index.size,
        "entity_count": len(graph.get("entities") or []),
        "edge_count": len(graph.get("edges") or []),
        # 向量通道状态必须写进元数据：静默缺席就等于将来没人发现它落后
        "vector_channels": "not_built_no_embedding_backend",
        "embedding_model": None,
        "embedding_dim": None,
    }
    return index, meta


def save_index(index: LexicalIndex, meta: dict, path: Path) -> None:
    """受 legacy 写保护约束；用 UUID 临时文件 + fsync + 原子替换。"""
    path = Path(path)
    legacy.guard_write(path, reason="index.save_index")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "doc_id": index.doc_id,
        "doc_len": index.doc_len,
        "postings": {k: v for k, v in sorted(index.postings.items())},
    }
    atomic.atomic_write_json(path, payload, indent=None)


def load_index(path: Path) -> tuple[LexicalIndex, dict]:
    payload = json.loads(path.read_text("utf-8"))
    index = LexicalIndex()
    index.doc_id = {int(k): v for k, v in payload["doc_id"].items()}
    index.doc_len = {int(k): v for k, v in payload["doc_len"].items()}
    index.postings = {k: {int(s): f for s, f in v.items()} for k, v in payload["postings"].items()}
    return index, payload["meta"]


def freshness_check(meta: dict, current_chunk_count: int, current_corpus_fp: str) -> list[str]:
    """启动三查：任一不等就报错，不许静默使用。"""
    problems = []
    if meta.get("chunk_count") != current_chunk_count:
        problems.append(f"索引块数 {meta.get('chunk_count')} != 当前块数 {current_chunk_count}")
    if meta.get("corpus_fp") != current_corpus_fp:
        problems.append(f"索引语料指纹 {meta.get('corpus_fp')} != 当前 {current_corpus_fp}")
    if not meta.get("built_at"):
        problems.append("索引缺少构建时间")
    return problems


# 固定探针：每次都查这几条，用来发现"计数全对但检索排不出来"的失效。
# 选词理由：覆盖高/中/低频与多字词，且都是确定出现在语料里的实体名与常用词。
FIXED_PROBES: tuple[str, ...] = (
    "凯尔希",
    "阿米娅",
    "罗德岛",
    "龙门",
    "切尔诺伯格",
    "整合运动",
    "干员",
    "博士",
    "感染",
    "乌萨斯",
)


def probe_digest(index: "LexicalIndex", probes: tuple[str, ...] = FIXED_PROBES, top_k: int = 10) -> str:
    """把固定探针的排序结果摘要成一个值。

    计划 C2 要求 L6 回读不能只比数量。这里补的是"检索行为"这一层：
    同一个索引重算一遍，10 条固定查询的 top-10 结果序列必须逐字相同。
    数量相同、内容不同、甚至倒排写错 slot——只要检索结果变了，这里就会变。
    """
    h = hashlib.sha256()
    for q in probes:
        ranked = index.search(q, top_k=top_k)
        h.update(q.encode("utf-8"))
        h.update(b"\x00")
        for chunk_id, score in ranked:
            h.update(f"{chunk_id}:{score:.6f},".encode())
        h.update(b"\n")
    return h.hexdigest()
