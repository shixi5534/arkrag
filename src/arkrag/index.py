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
from functools import lru_cache
from pathlib import Path

from . import atomic, config, fingerprint, legacy

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# t2 用的两个正则。与 `TOKEN_RE` 分开保留，因为 t1 的语义必须能原样重放。
#
# `WORD_T2_RE` 允许词内连接符，让 `ACT16`、`1-7`、`AB-3`、`A.B` 这类编号与专名
# **整词保留**（计划 G3 要求「保留英文数字专有编号」）。t1 的 `[A-Za-z0-9]+`
# 会把它们切成 `act` / `16`，于是查 `ACT16` 会命中所有含 `16` 的块。
WORD_T2_RE = re.compile(r"[A-Za-z0-9]+(?:[-_.:/][A-Za-z0-9]+)*")

# 连续汉字串。t2 的双字词只允许在**同一个串内**生成，这是与 t1 的关键差别。
CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")

# 分词实现版本。任何改变 token 集合的改动都必须升这个号并进指纹——
# 否则「切词变了」和「索引坏了」在指纹上完全一样，只能靠人眼分辨。
#
# t1 -> t2（计划 G3）：
#   1. 跨标点/跨句的双字词被禁止。t1 先把全文汉字抽成一个数组再 `zip` 相邻，
#      标点被直接跳过，于是 `"我在罗德岛。凯尔希前往龙门。"` 产出 `岛凯`、`希前`
#      两个语料里根本不存在的词。200 个真实块实测 t1 多出 **21,811** 个这种词。
#      危害不是"多几个无用词"：查询里只要这两个字相邻，就会命中一批毫无关系的块，
#      BM25 还给它不低的分数，而**没有任何门能发现**——这些 token 在索引里合法存在。
#   2. 编号整词保留（见 `WORD_T2_RE`）。
#   3. 实体名与权威别名可作为完整 Token 入索引（见 `tokenize` 的 `name_terms`）。
TOKENIZER_VERSION = "t2"

# 历史分词版本表。`LexicalIndex` 记住自己是用哪版建的，查询时用同一版切词，
# 这样旧世代索引的 `probe_digest` 重放结果与它建库时逐字相同——
# 否则改一次分词就会让**所有已发布世代的 `release_cli verify` 变红**，
# 而那种红是假红（索引没坏，只是重放用错了分词器）。
SUPPORTED_TOKENIZERS: tuple[str, ...] = ("t1", "t2")

# BM25 参数（H2 提为具名常量）。
#
# 原本是 `search()` 里的局部变量 `k1, b = 1.5, 0.75`。提出来是因为 H3 的 SQLite
# FTS5 后端必须用**同一组参数**，否则两边打分不可比、"排序逐位相同"这个验收条件
# 就无从谈起；而让别的模块去 import 一个函数内部的名字是不可能的，只能复制字面量，
# 那就又成了「同一判定写两处」（见 `tests/test_single_source_of_truth.py`）。
#
# 数值**没有改动**，只是从局部变量搬到模块级。改这两个值等于改所有世代的检索
# 排序，属于必须走新世代 + 全量评测的变更。
BM25_K1: float = 1.5
BM25_B: float = 0.75

# 参与检索的字段。标题与章节进索引是为了让「切城篇里凯尔希说了什么」能命中，
# 只嵌正文会分不清同章节的不同场景（原有注释保留）。
INDEX_FIELDS = ("title", "chapter", "text")


def tokenize_t1(text: str) -> list[str]:
    """t1：全文汉字抽平后连双字（跨标点也连）。保留以支持旧索引重放。"""
    out: list[str] = []
    for segment in re.split(r"\s+", text):
        for m in TOKEN_RE.finditer(segment):
            out.append(m.group().lower())
    han = [ch for ch in text if "\u4e00" <= ch <= "\u9fff"]
    out.extend(han)
    out.extend(a + b for a, b in zip(han, han[1:]))
    return out


class NameTable:
    """专名表。用来在一次切词里找出「文本中出现了哪些专名」。

    为什么需要它：只按标点断句不够。`"凯尔希前往龙门"` 里没有任何标点，但它是
    「专名 凯尔希」+「动词 前往」，t1 会连出 `希前`。计划 G3 的回归句
    `"我在罗德岛。凯尔希前往龙门。"` 同时包含跨标点（`岛凯`）与跨专名边界（`希前`）
    两种假词，只修前者测试仍不过。

    为什么不逐条 `term in text`：实体名 + 权威别名实测 **8,177** 条，13,781 个块
    逐条子串搜索是一亿多次调用。六次实现实测（全量建库耗时，t1 基线 15–27 秒，
    机器有噪声，故同时记下同一轮测到的 t1 值）：

    | 实现 | 耗时 | t1 同轮 | 问题 |
    |---|---|---|---|
    | 首字分桶 + `bytearray` 占位 | 286.0s | 25.4s | 每位置 `any(taken[i:end])` 都新建切片 |
    | 按长度 `frozenset` 查表 | 67.9s | 25.4s | 长度循环空转十几次 `continue` |
    | 首字 `dict.get` + 逐候选 `startswith` | 56.4s | 17.2s | **3,970 万次** `startswith` 调用 |
    | 专名表编成一条交替正则 | 176.6s | 28.4s | 8,177 个分支让 `re` 在每位置线性尝试，比不用还慢 |
    | 字典 trie 逐字符走 | 45.1s | 24.7s | 每位置一次 `dict` 查找，多数位置立刻跳出，但仍是 Python 级 |
    | **两字前缀与块内 bigram 集合求交** | **17.1s** | 14.9s | 最终版：C 层集合求交定候选，只对候选做 `str.find` |

    教训：前四版都在"减少候选"上打转，但每字符仍在 Python 层做多次函数调用。
    真正的解法是**完全不在字符层扫**——任何专名都以它自己的两字前缀开头，所以
    把「表里所有两字前缀」与「本块出现的全部双字」求交（`frozenset & set` 是 C 层
    操作），一次就得到可能开头的少数几个前缀，再只对这些做 `str.find`。Python 层
    的循环次数从"每字符一次"降到"每个命中候选一次"。

    这不只是性能问题：分词是实现细节，却决定了索引体积（词条 694,304 → 390,507）
    与建库时间。若不是先量了一遍代价，这个十倍会一直留在构建链路里。
    """

    __slots__ = ("_root", "_prefix2", "_words", "_count")

    _END = ""  # 词尾标记，不会与任何单字键冲突（空串不是合法字）

    def __init__(self, terms) -> None:
        root: dict[str, dict] = {}
        words: set[str] = set()
        cjk_terms: list[str] = []
        for term in terms or ():
            term = str(term or "").strip()
            if len(term) < 2:
                # 单字专名不进整词表：单字 token 已覆盖它，整词再加一遍是重复计数
                continue
            if CJK_RUN_RE.fullmatch(term):
                cjk_terms.append(term)
                node = root
                for ch in term:
                    node = node.setdefault(ch, {})
                node[self._END] = {}
            else:
                # 含拉丁/数字/标点的名字（`ACT16`、`A-3`）走整词路径比较
                words.add(term.lower())
        self._root = root
        # 名字前缀索引：任何专名都必然以它自己的两字前缀开头。于是"这个位置有没有
        # 可能是专名开头"可退化成"这两字是不是某个专名的前两字"。实测 1,480 个不同
        # 首字覆盖了 48.7% 的字符（在 Python 层做字符判断只省几个百分点），而两字
        # 前缀只有 **2,261** 个 —— 先把块里的 bigram 集合与它求交，候选就只剩少数。
        self._prefix2 = frozenset(t[:2] for t in cjk_terms)
        self._words = frozenset(words)
        self._count = len(cjk_terms) + len(words)

    def __len__(self) -> int:
        return self._count

    def __bool__(self) -> bool:
        """必须显式定义。

        踩过的坑：没写 `__bool__` 时，`tokenize_t2` 里每处理一个汉字串都会执行
        `if table` / `if table else`，而 Python 的真值判定会回落到 `__len__`。
        `__len__` 原本每次现算 `sum(len(v) for v in self._by_len.values())`
        （约 39 万项），全量建库因此调用了 141 万次 `__len__`、耗时 26 秒，
        `__len__` 一度是热点榜第二。表的"非空"与"有多少条"是两件事，
        不该让真值判定去数条目。
        """
        return self._count > 0

    def spans_with(self, run: str, bigrams) -> list[tuple[int, int]]:
        """在 `run` 里找专名区间，复用调用方**已经算好的** `bigrams`。

        核心思路：任何专名都以它自己的两字前缀开头。所以先用 `_prefix2` 与这个
        run 的双字集合求交（`frozenset & set` 是 C 层操作），得到"可能开头的两字"，
        再对这些少数候选做 `str.find` 定位 —— 于是 Python 层的循环次数从
        "每个字符一次"降到"每个命中的候选一次"。命中的位置再沿 trie 走最长匹配。

        不重叠的取舍必须**按位置顺序**做。踩过的坑：第一版按前缀字典序迭代
        候选并维护 `last_end`，「罗德岛」没被认出来 —— `"凯尔"` 排在 `"罗德"`
        前面先匹配到位置 6..9，`last_end` 前进到 9，随后 `"罗德"` 在位置 2 的
        命中被判成"落在已有匹配内部"而丢弃。前缀字典序与文本位置序无关，
        所以这里先收全部候选命中，再按 `(start, -length)` 排序后贪心取不重叠集。

        为什么不用字符类正则：试过，48.0s vs 46.2s，等于没改。1,480 个不同首字
        覆盖 48.7% 的字符，字符级过滤本来就没有多少可省的。
        """
        root = self._root
        if not root:
            return []
        candidates = self._prefix2 & set(bigrams)
        if not candidates:
            return []
        end_mark = self._END
        n = len(run)
        hits: list[tuple[int, int]] = []
        for prefix in candidates:
            start = run.find(prefix)
            while start >= 0:
                node = root[run[start]]
                best = None
                j = start + 1
                while True:
                    if end_mark in node:
                        best = j
                    if j >= n:
                        break
                    nxt = node.get(run[j])
                    if nxt is None:
                        break
                    node = nxt
                    j += 1
                if best is not None:
                    hits.append((start, best))
                start = run.find(prefix, start + 1)
        if not hits:
            return []
        # 位置优先、同起点取最长，然后贪心去重叠
        hits.sort(key=lambda sp: (sp[0], sp[0] - sp[1]))
        spans: list[tuple[int, int]] = []
        last_end = -1
        for start, end in hits:
            if start >= last_end:
                spans.append((start, end))
                last_end = end
        return spans

    def spans_in_run(self, run: str) -> list[tuple[int, int]]:
        """同 `spans_with`，但自己算双字。建库热路径请直接调 `spans_with`。"""
        return self.spans_with(run, [run[i:i + 2] for i in range(len(run) - 1)])

    def spans_in_chunk(self, text: str) -> list[tuple[int, int]]:
        """在整块文本上找专名区间（含绝对偏移）。"""
        if not self._root:
            return []
        n = len(text)
        bigrams = [text[i:i + 2] for i in range(n - 1)]
        return self.spans_with(text, bigrams)

    def latin_terms_in(self, words) -> list[str]:
        """返回文本里以整词形式出现的非汉字专名（`words` 是已小写的词集合）。"""
        return sorted(self._words & set(words))


@lru_cache(maxsize=4)
def _table_for(frozen: frozenset) -> NameTable:
    """专名表的缓存。建库时 13,781 次切词共用同一份表，不该每次重建。"""
    return NameTable(frozen)


def _as_table(name_terms) -> NameTable | None:
    """把 `set` / `frozenset` / `NameTable` 统一成 `NameTable`。

    已经是 `NameTable` 时原样返回，**不做任何复制**。建库时这个函数每块调一次，
    如果每次都 `frozenset(name_terms)`，8,177 个名字会被复制 13,781 遍——
    实测这一项单独占 11.3 秒（占全量建库 43.1s 的四分之一），比 trie 扫描本身还贵。
    所以调用方应当先转一次再传下来。
    """
    if not name_terms:
        return None
    if isinstance(name_terms, NameTable):
        return name_terms
    return _table_for(frozenset(name_terms))


def tokenize_t2(text: str, *, name_terms=None) -> list[str]:
    """t2：单字 + 串内相邻双字；编号整词；专名整词入表**且不在专名边界连双字**。

    边界规则为什么必要
    ------------------
    只按标点断句还不够。`"凯尔希前往龙门"` 里没有任何标点，但它显然是
    「专名 凯尔希」+「动词 前往」，而 t1 会连出 `希前` —— 这两个字在任何语料里
    都不是一个词。计划 G3 给的回归句 `"我在罗德岛。凯尔希前往龙门。"` 同时包含
    这两种情形：`岛凯` 是跨标点，`希前` 是跨专名边界。只修标点那条，
    回归测试仍然不过。

    所以：已识别的专名区间两侧**不连**跨边界双字（专名内部的双字碎片仍保留，
    它们能帮上部分匹配，且不会跨到别的实体上去）。
    """
    table = _as_table(name_terms)
    out: list[str] = []
    words: set[str] = set()
    for m in WORD_T2_RE.finditer(text):
        w = m.group().lower()
        words.add(w)
        out.append(w)
    # 专名扫描每块做一次（绝对偏移），不再逐汉字 run 调用 —— 见 spans_in_chunk。
    spans = table.spans_in_chunk(text) if table else []
    # cuts 存**绝对位置**：`p in cuts` 表示 text[p-1] 与 text[p] 之间是边界
    cuts: set[int] = set()
    for start, end in spans:
        cuts.add(start)
        cuts.add(end)
    present: list[str] = []
    for m in CJK_RUN_RE.finditer(text):
        run = m.group()
        out.extend(run)
        s = m.start()
        # 双字用切片而不是 `run[i] + run[i+1]`：后者每对都新建两个单字对象再拼接，
        # 13,781 个块累计约 825 万次，实测量级与 trie 扫描本身相当。
        bigrams = [run[i:i + 2] for i in range(len(run) - 1)]
        if cuts:
            bigrams = [b for i, b in enumerate(bigrams) if (s + i + 1) not in cuts]
        out.extend(bigrams)
    if table:
        for start, end in spans:
            present.append(text[start:end])
        out.extend(present)
        out.extend(table.latin_terms_in(words))
    return out


def tokenize(
    text: str,
    *,
    version: str = TOKENIZER_VERSION,
    name_terms=None,
) -> list[str]:
    """按指定版本切词。中文单字 + 双字、拉丁/数字整词，不依赖任何分词库或模型。

    `version` 必须是 `SUPPORTED_TOKENIZERS` 之一——拼错即抛 `ValueError`。
    不默认回退到最新版：安静地用错分词器，会让旧索引的检索结果与建库时不一致，
    而 `probe_digest` 只会说"不符"，不会说"你用错了分词器"。

    `name_terms` 只在 t2 生效，接受 `set`/`frozenset`/`NameTable`。它解决 t1 的
    两个老问题：单字+双字切分处理不了三字以上专名（「罗德岛制药公司」只能靠碎片
    被查到），以及专名边界会连出假词（`希前`）。刻意不做全量注入——只把文本里
    **确实出现**的专名整词加入，否则每个块都背上全部 8 千个名字。
    """
    if version not in SUPPORTED_TOKENIZERS:
        raise ValueError(
            f"未知分词版本 {version!r}，只支持 {SUPPORTED_TOKENIZERS}"
        )
    if version == "t1":
        return tokenize_t1(text)
    return tokenize_t2(text, name_terms=name_terms)


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

    性能：每个词条**只 encode 一次**，槽位串用 `",".join` 拼好再一起喂给哈希。
    逐槽 `h.update(...)` 的写法在真实数据上要 940 万次 `encode`+`update`
    （实测 19.0 秒，占全量建库 43.1s 的 44%），而摘要本身只要 694,304 次调用。
    摘要值不变——拼出来的字节序列与逐段 update 完全一致。
    """
    h = hashlib.sha256()
    h.update(
        "".join(
            f"{slot}\x00{index.doc_id[slot]}\x00{index.doc_len.get(slot, 0)}\n"
            for slot in sorted(index.doc_id)
        ).encode()
    )
    for term in sorted(index.postings):
        posting = index.postings[term]
        # 每槽都带尾随逗号（保持与原实现在字节层面完全一致，摘要值才不变）
        pairs = "".join(f"{slot}:{posting[slot]}," for slot in sorted(posting))
        h.update(term.encode("utf-8"))
        h.update(b"\x00")
        h.update(pairs.encode())
        h.update(b"\n")
    return h.hexdigest()


class LexicalIndex:
    """词法倒排。

    `tokenizer_version` 是这个索引**自己**的分词版本，不是当前代码的版本。查询
    与 `probe_digest` 重放都用它，这样旧世代索引的行为与它建库时逐字一致。
    `avg_len` 在 `add()` 之后惰性算一次并缓存：旧实现每次 `search` 都
    `sum(self.doc_len.values())`，1.3 万个文档时等于每次查询都扫一遍全部长度
    （计划 G3/H2 都点名了这项）。
    """

    def __init__(self, tokenizer_version: str = TOKENIZER_VERSION) -> None:
        if tokenizer_version not in SUPPORTED_TOKENIZERS:
            raise ValueError(
                f"未知分词版本 {tokenizer_version!r}，只支持 {SUPPORTED_TOKENIZERS}"
            )
        self.tokenizer_version = tokenizer_version
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)
        self.doc_len: dict[int, int] = {}
        self.doc_id: dict[int, str] = {}
        self._avg_len: float | None = None
        # BM25 长度归一化分母项，按 slot 索引的扁平表（H2）。与 `_avg_len` 同时
        # 失效——它们依赖同一批 `doc_len`。
        self._norm: list[float] | None = None
        # 建库时用的整词名表。必须随手带着走：查询侧要用同一份，
        # 否则倒排里那些整词名 token 永远查不到，白白拉高 doc_len。
        self.name_terms: frozenset[str] = frozenset()

    def set_name_terms(self, terms: "set[str] | None") -> None:
        self.name_terms = frozenset(terms or ())

    def add(
        self,
        slot: int,
        chunk_id: str,
        text: str,
        *,
        name_terms: "set[str] | None" = None,
    ) -> None:
        counts = Counter(tokenize(text, version=self.tokenizer_version, name_terms=name_terms))
        if not counts:
            return
        self.doc_len[slot] = sum(counts.values())
        self.doc_id[slot] = chunk_id
        self._avg_len = None  # 新增文档即失效
        self._norm = None
        for token, freq in counts.items():
            self.postings[token][slot] = freq

    @property
    def size(self) -> int:
        return len(self.doc_id)

    @property
    def avg_len(self) -> float:
        """平均文档长度。惰性计算一次并缓存，`add()` 时失效。"""
        if self._avg_len is None:
            n = max(1, self.size)
            self._avg_len = (sum(self.doc_len.values()) / n) if n else 1.0
        return self._avg_len

    def norm_table(self) -> list[float]:
        """BM25 长度归一化项 `k1*(1-b+b*len/avg_len)`，按 slot 预建一次（H2）。

        ## 为什么值得预建

        H1 实测：真实索引 13,781 文档 / 694,304 词项 / **10,525,428 个 postings 条目**，
        每问平均命中 56.7 个 query term、内层循环迭代 **19.3 万次**。旧实现在这 19.3 万次
        迭代里每次都做 `self.doc_len.get(slot, 1)` 再算一遍 `length/avg_len`
        ——而这个除法**只与文档有关，与问句无关**，`doc_len` 在一个世代内又不变。

        预建后每问只剩一次列表索引与一个乘加，除法从 19.3 万次降到 1.4 万次（每代一次）。
        真实索引 A/B（320 题）：P50 150.789 → 93.378 ms（−38.1%）、
        P95 228.979 → 138.992 ms（−39.3%）。

        ## 为什么是 list 而不是 dict

        slot 由 `enumerate(sorted(chunk_records, ...))` 连续分配（0..n-1），
        用 list 下标就是 slot 本身，比 dict 哈希快。有空洞时按原实现的
        `doc_len.get(slot, 1)` 落 `len=1` 的默认值补齐——**保持与旧实现逐位等价**，
        这是本次优化的验收条件，不是"差不多就行"。

        ## 数值等价性

        `k1*(1-b+b*len/avg_len)` 与旧实现 `freq + k1*(1-b+b*length/avg_len)`
        里的同一子表达式**逐位相同**（同一表达式、同一运算次序、同一 float 精度）。
        唯一的差别是把"每问算一次"改成"每代算一次"，因此排序结果必须完全一致
        （`tests/test_index_norm_table.py` 对此逐位断言）。
        """
        if self._norm is not None:
            return self._norm
        avg_len = self.avg_len
        k1, b = BM25_K1, BM25_B
        if not self.doc_id:
            self._norm = []
            return self._norm
        biggest = max(self.doc_id)
        # 默认值对应旧实现里 `doc_len.get(slot, 1)` 的兜底分支。
        default = k1 * (1.0 - b + b * 1 / avg_len)
        norm = [default] * (biggest + 1)
        for slot, length in self.doc_len.items():
            if 0 <= slot <= biggest:
                norm[slot] = k1 * (1.0 - b + b * length / avg_len)
        self._norm = norm
        return norm

    def search(
        self,
        query: str,
        top_k: int = 20,
        *,
        name_terms: "set[str] | None" = None,
    ) -> list[tuple[str, float]]:
        """BM25 检索。返回 `[(chunk_id, score)]`，排序稳定（同分按 chunk_id）。

        `name_terms` 不传时用索引自带的 `self.name_terms`——这是正确的默认值：
        建库时注入了哪些整词名，查询时就得用同一份，否则那些 token 永远查不到，
        而倒排里它们的 `doc_len` 贡献仍在，等于凭空抬高每个文档的长度。
        """
        terms = self.name_terms if name_terms is None else name_terms
        n = max(1, self.size)
        q_terms = Counter(tokenize(query, version=self.tokenizer_version, name_terms=terms))
        norm = self.norm_table()
        k1 = BM25_K1
        k1p1 = k1 + 1.0
        scores: dict[int, float] = defaultdict(float)
        for term, tf in q_terms.items():
            posting = self.postings.get(term)
            if not posting:
                continue
            df = len(posting)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            # 内层循环是 H1 剖析里的第一热点（81 万次迭代）。这里省掉每轮的
            # `doc_len.get` 与除法，只留一次列表索引与一个乘加。
            for slot, freq in posting.items():
                scores[slot] += idf * (freq * k1p1) / (freq + norm[slot])
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], self.doc_id[kv[0]]))[:top_k]
        return [(self.doc_id[slot], score) for slot, score in ranked]


def build_adjacency(edges: list[dict]) -> dict[str, list[tuple[str, str, float]]]:
    """旧形态的检索期邻接：双向塞、关系名不变。

    这是 `ark-rag-1` 的语义，**保留原样**（已发布世代的索引就是这么建的，
    改它等于让旧代的索引与建它的代码不再同源）。`ark-rag-3` 的方向邻接不走这里，
    走 `arkrag.adjacency.build_directed_adjacency`——那里 out/in 分列、反向换逆谓词。

    字段按元素类型分派（两代边字段名不同，见 `gates.edge_ends` 的注释）：
    这里的映射是**无损的**，只换字段名不改语义。
    """
    adj: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    for edge in edges:
        src = edge.get("source")
        dst = edge.get("target")
        if src is None:
            src = edge.get("subject_id")
        if dst is None:
            dst = edge.get("object_id")
        src, dst = str(src or ""), str(dst or "")
        rel = edge.get("relation")
        if rel is None:
            rel = edge.get("predicate")
        weight = edge.get("confidence")
        if weight is None:
            summary = edge.get("confidence_summary") or {}
            weight = summary.get("max") if isinstance(summary, dict) else None
        adj[src].append((dst, str(rel or ""), float(weight or 0.0)))
        # 检索期游走要双向连通；方向语义由关系名本身携带，不在这里反转事实
        adj[dst].append((src, str(rel or ""), float(weight or 0.0)))
    return {k: sorted(v) for k, v in adj.items()}


def build_alias_index(entities: list[dict]) -> dict[str, str]:
    """实体名 + 权威别名 -> 实体 id。

    `aliases` 有两代形态：`ark-rag-1` 是 list[str]，`ark-rag-3` 是 list[dict]
    且**只含 review ∈ {auto, approved}**（`identity.to_dict()` 已把待审/拒绝
    分到另外两张表）。这里按元素类型取文本，不按 schema 分支——同一个函数
    服务两代图，判据不重复写。

    为什么不在新图那边额外过滤 `pending_aliases`：`to_dict()` 的输出里根本没有
    把待审别名放进 `aliases`，本函数再过滤一遍就是"同一判定写两处"，
    而那正是 `tests/test_single_source_of_truth.py` 记录的三次事故的共同机理。
    """
    out: dict[str, str] = {}
    for ent in entities:
        eid = ent["id"]
        out[ent["name"]] = eid
        for item in ent.get("aliases") or []:
            text = str(item.get("text") or "") if isinstance(item, dict) else str(item or "")
            text = text.strip()
            if text:
                out.setdefault(text, eid)
    return out


def name_terms_from_graph(graph: dict) -> set[str]:
    """从图里取「可以被整词检索的名字」：实体名 + 权威别名。

    **只用 `aliases`，不碰 `pending_aliases` / `rejected_aliases`。** 待审别名
    进检索表，就等于拿一个还没被认可的合并结论去影响召回——`ark-rag-3` 的
    `identity.to_dict()` 已经把两张表分开，这里再读一遍待审表属于"同一判定写两处"，
    而 `tests/test_single_source_of_truth.py` 记录的三次事故都是这个机理。

    返回的是**候选词表**，不是"每个块都要注入的词"。`tokenize` 只把文本里确实
    出现的名字整词入索引，所以 8 千个名字不会让每个块都背上全部词。
    """
    terms: set[str] = set()
    for ent in graph.get("entities") or []:
        name = str(ent.get("name") or "").strip()
        if len(name) >= 2:
            terms.add(name)
        for item in ent.get("aliases") or []:
            text = str(item.get("text") or "") if isinstance(item, dict) else str(item or "")
            text = text.strip()
            # 单字别名不进整词表：单字已经在单字 token 里了，再整词加一遍是重复计数
            if len(text) >= 2:
                terms.add(text)
    return terms


def build(
    chunk_records: list[dict],
    graph: dict,
    corpus_fp: str,
    *,
    graph_artifact_id: str = "",
    tokenizer_version: str = TOKENIZER_VERSION,
    use_name_terms: bool = True,
) -> tuple[LexicalIndex, dict]:
    """建词法索引。

    `tokenizer_version` 默认用当前版本；显式传 `"t1"` 可以重建历史语义的索引
    （用于对照旧世代的检索行为）。`use_name_terms` 关掉时退化成纯单字+双字切分，
    便于量化"整词名"这一项单独带来多少收益（计划 I2 要求收益可分离测量）。
    """
    index = LexicalIndex(tokenizer_version=tokenizer_version)
    name_terms = name_terms_from_graph(graph) if use_name_terms else None
    index.set_name_terms(name_terms)
    # 表只转一次再逐块传下去。每块都传原始 set 的话，`_as_table` 会为 13,781 个块
    # 各复制一遍 8,177 个名字，实测 11.3 秒——纯粹是重复劳动。
    table = _as_table(name_terms)
    for slot, rec in enumerate(sorted(chunk_records, key=lambda r: r["chunk_id"])):
        # 块向量/词法文本都带标题前缀：正文常缺主语与场景，只嵌正文会分不清同章节的不同场景
        index.add(
            slot,
            rec["chunk_id"],
            f"{rec.get('title','')} {rec.get('chapter','')} {rec['text']}",
            name_terms=table,
        )
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
        "index_build_fp": index_build_fingerprint(tokenizer=index.tokenizer_version),
        "tokenizer_version": index.tokenizer_version,
        "name_terms_enabled": bool(name_terms),
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
    """受 legacy 写保护约束；用 UUID 临时文件 + fsync + 原子替换。

    `tokenizer_version` 与 `name_terms` 必须一起落盘：查询与 `probe_digest` 重放都要
    用它们。只存倒排不存这两样，回读后就只能按"当前代码的默认分词器"重放，于是
    每次改分词都会让**所有已发布世代**的 `verify` 变红——那种红是假红。
    `name_terms` 约 8 千条、JSON 后约数十 KB，相对 127 MB 的倒排可以忽略。
    """
    path = Path(path)
    legacy.guard_write(path, reason="index.save_index")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "tokenizer_version": index.tokenizer_version,
        "name_terms": sorted(index.name_terms),
        "doc_id": index.doc_id,
        "doc_len": index.doc_len,
        "postings": {k: v for k, v in sorted(index.postings.items())},
    }
    atomic.atomic_write_json(path, payload, indent=None)


def load_index(path: Path) -> tuple[LexicalIndex, dict]:
    payload = json.loads(path.read_text("utf-8"))
    # 旧索引（ark-rag-1 那一代）没有这两个字段。缺 `tokenizer_version` 时按 t1 恢复，
    # 因为那批索引确实是 t1 建的；缺 `name_terms` 时为空集，t1 本来也不用它。
    version = payload.get("tokenizer_version") or "t1"
    index = LexicalIndex(tokenizer_version=version)
    index.set_name_terms(payload.get("name_terms") or None)
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
