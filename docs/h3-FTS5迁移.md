# H3：检索后端迁移到 SQLite（FTS5）

> 本文件记录 H3 阶段的实现、口径与实测。所有数字都附了产生它的确切命令，
> 可逐条复跑。凡是本机测不出来的（LLM 生成质量、embedding 召回），写成
> `null` 并说明原因，不做估算。

## 1. 这一阶段到底在解决什么

H1 的剖析结论（`data/bench_h1_4eafae.json`）说得很清楚：**检索算法本身没病，
病在存储形态**。

| 事实 | 数值（H1） |
| --- | --- |
| 冷启动总耗时 | 12,077.7 ms |
| 其中解析 `index/lexical.json`（121.4 MiB / 127.2 MB） | **9,972.5 ms（82.6%）** |
| 常驻内存（加载后） | 1,499.94 MiB，峰值高水位 **1,717.59 MiB** |
| 热查询 `search` 阶段 P95 | 251.4 ms |

H2 把 `search` 阶段压到了 167.5 ms（预建长度归一化表），但**冷启动与内存一分没动**。
所以 H3 要打的是这两个数。

H3 的做法：把倒排搬进 SQLite，冷启动变成「打开一个数据库文件」，
内存只剩真正需要的东西（块正文、图、断言），不再有 1,052 万条
`dict[token][slot] = freq` 的 Python 对象。

**先说清结论，免得读到 §5 才发现与预期不符**：H3 拿到了冷启动与内存两项大幅改善
（冷启动 `index_ms` 5.5–10.5 s → 约 19 ms，峰值 RSS 1,719 → 533 MiB），
召回与 top-20 排序**逐位不变**（320/320）；
但**检索延迟没有改善，反而更慢**（同进程交替口径 P95 +44%）。
这条不是失败，是这笔交易的代价——迁移换来的是启动与内存，不是查询速度。
细节与测量口径见 §5。

## 2. 架构：三层与一条边界

```
scripts/ask.py::build_pipeline
        │  只认 loaded["index"] 这一个槽
        ▼
src/arkrag/search_backend.py        ← 抽象层（协议 + 工厂 + 旧后端适配）
        ├── JsonSearchBackend  ──→ arkrag.index.LexicalIndex（旧路径，可回滚）
        └── Fts5SearchBackend  ──→ data/arkrag/fts/<gid>.sqlite（新路径）
```

### 2.1 协议为什么这么小

`src/arkrag/retrieve.py` 对「索引」的全部依赖只有：

* `search(query, top_k=...) -> [(chunk_id, score)]`（`Retriever.search` 里两处调用）；
* `tokenizer_version` / `name_terms`（`build_context` 算 token 数用）；
* `size` / `avg_len`（诊断与复算 BM25 用）。

协议就按这个**实际用到的面**定，不多一个成员。特别地，协议里**故意没有
`postings` / `doc_len`**：一旦放进去，任何新后端都被迫造一份内存倒排，
那就等于什么都没换。

`@runtime_checkable` 只校验成员存在，不校验语义——语义由测试保证（见 §6）。

### 2.2 工厂与三种取值

`search_backend.open_backend(generation_dir, kind=..., db_path=None)`：

| `kind` | 行为 |
| --- | --- |
| `"json"` | 读 `index/lexical.json`（旧路径） |
| `"fts"` | 读 `.sqlite`；**库不存在就直接抛 `FileNotFoundError`** |
| `"auto"`（默认） | 有 `.sqlite` 就用 fts，否则退回 json |

`"auto"` 的方向是「有新品用新品、没有就老路照跑」，这样**回滚只需要删库或改一个参数，
不需要改代码**。但 `"fts"` 绝不静默降级——「以为在测新后端、其实还在跑旧后端」
会让双跑对比得出完全错误的结论，这是本阶段最不能容忍的一类错误。

## 3. 核心技术决策：FTS5 只当倒排存储，打分仍用自己的实现

这是本次最重要、也最需要交代理由的一个决定。

### 3.1 不采用 FTS5 原生 `bm25()` 的原因

两者的 idf 公式不同：

| 实现 | idf |
| --- | --- |
| 本项目 `index.LexicalIndex.search` | `log(1 + (N - df + 0.5) / (df + 0.5))` |
| SQLite FTS5 `bm25()` | `log((N - df + 0.5) / (df + 0.5))` |

差一个 `+1`，在小 `df`（罕见词、专名）上偏差最大——而那些词恰恰是本语料里
最有区分度的部分。实测同一批题、同一 `top_k=20`，两个口径的 chunk_id 交集只有
**12/20、11/20、10/20**（三题）。也就是说：换口径等于**换了一整套排序**，
「排序逐位相同」这个验收条件直接作废，而它正是唯一能证明「迁移没改行为」的证据。

`SCORE_MODE_FTS5` 这个模式**仍然保留**（`score_mode="fts5"`），因为它在
毫秒级上有价值（见 §7 的「未采用的优化」），但**默认绝不走它**，
并且有测试（`test_fts5_native_mode_is_flagged_as_different_scoring`）
断言它与 exact 模式的结果不同——防止有人把它悄悄设成默认还以为等价。

### 3.2 分词：为什么自己切、用空格拼起来写进 FTS 表

FTS5 自带的分词器（`unicode61` / `ascii` / `porter`）**不切中文**。实测把
`'凯尔希罗德岛'` 写进 FTS5 表，得到一个整词 term，不是「凯尔希」「罗德岛」。

而本项目的 `index.tokenize()` 有一套自己演进过的口径（CJK 双字切分、
`NameTable` 专名整词、`WORD_T2_RE` 保留 `a-b`/`a.b`/`a/b` 这类连写），
且**索引侧与查询侧必须完全一致**，否则倒排里的 token 永远查不到，
而它们的 `doc_len` 贡献还在，等于凭空抬高每个文档长度。

所以做法是：

```
tokenize(text) → ["凯尔希", "凯尔", "尔希", ...]  →  "凯尔希 凯尔 尔希 ..."  →  写进 FTS5 表
```

FTS5 退化成一个**纯倒排存储**：它只负责「按 term 快速找到文档集合」，
分词口径 100% 由 `index.tokenize()` 决定。这样两边的 token 集合可以逐词条断言相等
（`test_token_set_identical_to_source_index`）。

`FTS_TOKENIZE = "unicode61 tokenchars '_-./:'"` 的 `tokenchars` 是为了保住
`1-7` / `a.b` / `x_y` / `act16` 这类 token 不被拆开（默认 `unicode61` 会把
`1-7` 拆成 `1`、`7`）。注意 sqlite 的一个坑：`tokenchars` 与 `remove_diacritics`
**不能写在同一条 tokenize 指令里**，会报
`parse error in tokenize directive`，只能二选一。

### 3.3 倒排表设计：BLOB 打包

```
postings(term TEXT PRIMARY KEY, docs BLOB, freqs BLOB) WITHOUT ROWID
```

* `docs` = `array('I').tobytes()`（该词命中的 slot 序列，4 字节/项）
* `freqs` = `array('I').tobytes()`（对应词频）
* 查询时 `array.frombytes()` 零拷贝解析，逐对做与 `index.py` **字面相同**的浮点运算：

  ```python
  scores[slot] += idf * (freq * k1p1) / (freq + norm[slot])
  ```

`norm` 表由 `_build_norm()` 复刻 `index.norm_table()`，包括**空洞槽位的兜底值**
`k1 * (1 - b + b * 1 / avg_len)`——真实索引里就有这种洞（`LexicalIndex.add()` 遇到
分词为空的块会 `return`，不登记 `doc_id`/`doc_len`），兜底值写错会让这些块分数偏高。
这条有专门的测试，且**变异测试验证过它真的会红**（见 §6.3）。

BM25 参数从 `arkrag.index` import：

```python
from . import index as index_mod
k1 = index_mod.BM25_K1        # 1.5
k1p1 = k1 + 1.0
b = index_mod.BM25_B          # 0.75
```

不许写字面量 `1.5` / `0.75`，有 `test_no_bm25_numeric_literals_in_source` 守着
（用 `tokenize.generate_tokens` 只取 `NUMBER` token 比对，注释与字符串不参与，
连 `1.50` 也抓得到）。

### 3.4 表结构（计划要求的八张表）

| 表 | 内容 |
| --- | --- |
| `chunks` | `doc`(slot) / `chunk_id` / `doc_id` / `scene_id` / `title` / `chapter` / `doc_len` |
| `chunk_texts` | **FTS5 虚拟表**，`(tokens, content='', tokenize=FTS_TOKENIZE)` |
| `postings` | 倒排：`term` → `docs` / `freqs` BLOB |
| `entities` | 6,582 行（来自 `graph.entities`） |
| `aliases` | 3,344 行 |
| `assertions` | 62,910 行（来自 `assertions.jsonl`） |
| `relations` | 28,625 行（来自 `graph.edges`） |
| `artifacts` | 4 行（世代产物摘要） |
| `runs` | 1 行（`run_id` / `created_at`） |
| `meta` | 键值：`fts_schema_version` / `tokenizer_version` / `doc_count` / `avg_len` / `bm25_k1` / `bm25_b` / `index_content_digest` / `probe_digest` / `index_build_fp` / `built_at` … |

`meta` 里的 `bm25_k1`/`bm25_b` **只作追溯记录**，运行时一律从 `index.py` 读，
不从库里读——否则改参数就成了改数据而非改代码，`index.py` 那两处
`BM25_K1`/`BM25_B` 也就白提了。

## 4. 冷启动

### 4.1 建库一次，之后只打开

```powershell
py -3 scripts\build_fts_index.py --generation g-20261031T035231Z-4eafae
```

产物落在 **`data/arkrag/fts/<gid>.sqlite`**，刻意不放进
`generations/<gid>/`：世代目录带内容摘要与 `legacy` 写保护，往里塞派生物会让
`snapshot_baseline.py` 与 `release_cli verify` 的目录摘要变化。
`fts_backend._refuse_inside_generation()` 会**硬拒**任何落在 `generations/` 下的路径，
`build_fts_index.py` 也有同样的自检，两边都有测试守着。

建库后**必须自检**（不做的话，「文件写出来了」会被当成「库是对的」）：
块数、词条集合、逐块 `doc_len` 三项全等，外加 4 条探针要求新旧后端
`top_k=20` 的 chunk_id 序列逐位相同、分数漂移 < 1e-9。
不过就退出码 3。

### 4.2 实测

```
py -3 scripts\build_fts_index.py --generation g-20261031T035231Z-4eafae --force
```

```
源索引读出：8570 ms  块 13781  词条 694304  分词 t1  整词名 0
图产物：实体 6582  边 28625  断言 62910

建库完成
  体积      ：212.2 MiB（222,502,912 字节）
  建库耗时  ：33250 ms（打包 12758 / FTS5 写入 14391 / 图写入 2245）
  源 JSON   ：121.4 MiB  → 压缩比 0.57x
  表行数    ：chunks=13781 postings=694304 chunk_texts=13781 entities=6582
              aliases=3344 relations=28625 assertions=62910 artifacts=4 runs=1
自检通过：块数/词条集合/doc_len 全一致；4 条探针 top-20 逐位相同，
          最大分数漂移 0.00e+00（2390 ms）
```

（多次重建的实测区间：源索引读出 7.7–8.6 s、建库 31.6–36.2 s。波动来自磁盘缓存与
PowerShell 冷热，量的是一台机器上的同一件事。）

### 4.3 幂等性：同内容世代必须建出等价的库

磁盘上有三个世代内容完全相同（`index_content_digest` 与 `probe_digest` 逐字相同）：

| 世代 | `index_content_digest` 前 16 位 | `chunk_count` |
| --- | --- | --- |
| `g-20261031T033200Z-b22ca8` | `c1fef3588e662271` | 13781 |
| `g-20261031T034009Z-348206` | `c1fef3588e662271` | 13781 |
| `g-20261031T035231Z-4eafae` | `c1fef3588e662271` | 13781 |

拿 `b22ca8` 与 `4eafae` 各建一次库，逐表比对内容摘要：

```powershell
py -3 scripts\build_fts_index.py --generation g-20261031T033200Z-b22ca8
# 然后对两库逐表算内容摘要（临时脚本比对，结果如下）
```

| 表 | 行数 | 内容摘要 |
| --- | --- | --- |
| `chunks` | 13781 / 13781 | 一致 |
| `postings` | 694304 / 694304 | 一致 |
| `entities` | 6582 / 6582 | 一致 |
| `relations` | 28625 / 28625 | 一致 |
| `assertions` | 62910 / 62910 | 一致 |

**结论：相同内容世代建出的库逐表等价。** 但**文件字节不同**
（sha256 `f97aba52…` vs `d562a126…`），因为 `meta` 表里记了 `generation_id` 与
`built_at`——这是有意的：库必须能自证「我是从哪个世代建的」。
所以幂等性的正确口径是「**内容等价、字节不必相同**」，不要去追「重跑两次
sha256 一致」，那要靠固定 `built_at` 才能做到，代价是丢失溯源信息。

（验证用的 `b22ca8` 库 212.2 MiB 已删除，产物目录只保留基准世代的库。）

**关于体积：0.57x 是变大，不是变小，必须说清。**原因是倒排被完整保留成
`(slot, freq)` 对（1,052 万对 × 8 字节 + 69.4 万条 term 键），而 JSON 版本里
slot 以小整数形式存、压缩率更高。SQLite 版本换来的是**随机访问**，
不是更小的体积。计划里「压缩」不是 H3 的验收项，体积变大是已知代价。

## 5. 双跑对比

### 5.1 口径

```powershell
py -3 scripts\bench_h3.py --generation g-20261031T035231Z-4eafae --questions 320
```

* 同一批题：`bench_h1.pick_questions(gold, 320)` → 即全部 320 题（`ask._stratified_sample`，
  种子 `20261101`，按类别配额）
* 同一 `top_k=20`、同一 `graph_depth=1`、同一 `warm_reps=3`、同一随机种子
* 分位数：`bench_h1.percentile()`（nearest-rank，空样本返回 `None` 而不是 `0.0`）
* 七项指标：**不在这里重写**，原样调 `scripts/ask.py::evaluate`，只替换载入层。
  两份后端走**同一个函数**，指标口径不可能不同。
* **两种口径，回答两个不同的问题**（`bench_h3.py` 模块 docstring 有完整说明）：
  * **口径 A（跨子进程）**：每个后端跑在独立子进程里。这是**冷启动与内存**的正确口径
    ——先跑 JSON 再跑 FTS 会污染 RSS（127 MB JSON 解析留下的碎片），测不出真实峰值。
  * **口径 B（同进程交替）**：两个后端载入同一个进程、交替切换着跑，内存状态完全相同。
    这是**「后端本身快慢」**的唯一有效口径。口径 A 的延迟数**不能**用来比检索速度：
    JSON 子进程背着 1.5 GB 倒排 dict，FTS 子进程只占 532 MiB，于是 FTS 子进程的
    *每一个阶段*都白拿一份「内存更宽裕」的好处。实测这个偏差足以把结论**反向翻转**——
    早期版本的本文件就是这样得出过「FTS 检索快 17%」的错误结论，实际是慢。

> **绝对延迟有 ±10% 量级的机器噪声。** 同一台机器、同一批题、隔半小时重跑，
> `search` 阶段 P50 会在 100–140 ms 之间漂，冷启动 `index_ms` 会在 5.5–10.5 s 之间漂
> （冷启动更大，受文件页缓存状态影响）。**结论只取同一次运行内的相对比较**；
> 把两次不同运行的数互相减，算出来的是噪声之差，不是收益。

### 5.2 结果（本次运行原样粘贴，未做任何手工调整）

```text
==============================================================================
H3 双跑对比  generation=g-20261031T035231Z-4eafae  题数=320
口径：top_k=20  graph_depth=1  warm_reps=3  seed=20261101
==============================================================================

【后端自我描述】
  json  json  size=13781  avg_len=1272.325666  tokenizer=t1  产物=...\index\lexical.json
  fts   fts  size=13781  avg_len=1272.325666  tokenizer=t1  产物=...\fts\g-...-4eafae.sqlite

【七项指标】(同批题、同一函数计算)
  指标                                json       fts     delta  n(json/fts)
  recall_at_20                    0.6575    0.6575   +0.0000  300/300
  recall_at_5                     0.4658    0.4658   +0.0000  300/300
  mrr                             0.4123    0.4123   +0.0000  300/300
  citation_location_accuracy      1.0000    1.0000   +0.0000  1543/1543
  required_citation_coverage      0.1567    0.1567   +0.0000  300/300
  abstain_accuracy                0.8406    0.8406   +0.0000  320/320
  direction_accuracy              0.6667    0.6667   +0.0000  96/96

【排序一致性】
  top-20 chunk_id 逐位相同：320/320  （100.00%）

【冷启动分阶段】（毫秒）
  阶段                     json        fts      delta
  graph_ms              306.8      288.0      -18.8
  adjacency_ms          215.2      210.1       -5.1
  assertions_ms        1007.8      937.5      -70.2
  index_ms             5467.2       18.4    -5448.9
  chunks_ms             320.7      296.1      -24.6
  total_ms             7317.7     1750.0    -5567.6
  retriever_build       100.6      178.6      +78.0

【内存 RSS（MiB）】
  at_start         json 26.47  peak 26.47   |   fts 26.6  peak 26.6
  after_load       json 1499.32  peak 1719.37   |   fts 532.64  peak 532.64
  after_pipeline   json 1505.98  peak 1719.37   |   fts 547.75  peak 547.75
  at_end           json 1509.97  peak 1719.37   |   fts 557.63  peak 557.66

【端到端延迟】（毫秒，nearest-rank）
  cold  P50 json=129.192 fts=164.81   P95 json=207.791 fts=262.574   P99 json=228.721 fts=308.555
  warm  P50 json=127.111 fts=165.593   P95 json=205.9 fts=259.719   P99 json=226.306 fts=297.849

【分阶段 P50/P95（warm，毫秒）】
  注意：本表来自两份独立子进程报告，两边的进程内存状态不同
        （json 子进程背着 1.5 GB 倒排 dict，fts 子进程只占 532 MiB），
        所以**本表的快慢差异不能当作「后端本身快慢」**，见下面的交替 A/B。
  阶段            json P50   fts P50  json P95   fts P95
  query_ms          0.63     0.653     0.786     0.885
  search_ms      103.934   138.917   160.306   205.456
  expand_ms         0.67     0.695     2.654     2.706
  context_ms      18.003    18.519    52.867    61.615
  answer_ms        0.298     0.321     0.528     0.578

【同进程交替 A/B】（两个后端共享同一内存状态，这才是检索快慢的有效口径）
  题数=320  轮数=2  两后端共存时 RSS=2423.03 MiB
  指标           json P50    fts P50   delta P50   json P95    fts P95   delta P95
  检索阶段           49.778     68.901     +19.123    150.899    217.607     +66.708
  端到端            63.367      83.74     +20.373    187.095    267.063     +79.968
    轮 1: json search P50=44.899  fts search P50=61.202
    轮 2: json search P50=57.724  fts search P50=162.679

【验收：热查询检索阶段 P95 < 200 ms】
  fts search P95 = 205.456 ms → 未达标（阈值 200 ms，超出 5.5 ms）
  但按交替 A/B 口径：fts 检索 P95 比 json 更慢 66.7 ms。
  → 结论：这条阈值在 H2 就已达成；H3 相对 json 后端在检索延迟上并没有收益，
     H3 的收益在冷启动与内存（见上方冷启动表与内存表）。
==============================================================================
```

### 5.3 怎么读这张表

**召回与排序：完全一致，delta 全为 0.0000。** 这不是巧合，是设计出来的：
打分公式、分词、idf、长度归一化、同分次序全部逐字复刻，所以 `top-20 chunk_id`
在 320 题上 320/320 逐位相同。既然检索结果一模一样，下游的 expand / context / answer
也就一模一样——七项指标等于自动相同，`candidates` 分布也完全相同
（p50 112 / p95 170 / max 275）。**这一项是 H3 成立的硬前提，也是最该守住的一条。**

**检索延迟：H3 没有收益，实际更慢。** 这是本次最需要老实说的一条，也最反直觉：

* 口径 B（同进程交替，唯一有效的检索快慢口径）：`search` 阶段
  **P50 49.778 → 68.901 ms（+19.1 ms，+38%）**，
  **P95 150.899 → 217.607 ms（+66.7 ms，+44%）**；
  端到端 P95 187.095 → 267.063 ms（+80.0 ms）。
* 口径 A（跨子进程）里 FTS 那一列同样更慢：`search_ms`
  103.934 → 138.917 ms（P50）、160.306 → 205.456 ms（P95），方向与口径 B 一致。

原因写在 `fts_backend.py` 模块 docstring：json 后端装好之后 `postings` 就是热 dict，
`dict.get(term)` 直接命中，Python 层零反序列化；FTS 后端每题要为命中的 ~74 个 term
各做一次 `array("I").frombytes()` 重建数组对象再读 BLOB，这笔开销超过了「少扫一点」的收益。

**「P95 < 200 ms」本次未达标（205.456 ms，超出 5.5 ms），但这不应记作 H3 的退步。**
该阈值在 H2 就已达成（json 167.5 ms @ H2 实测），本次 json 列是 160.3 ms——**H2 的成果仍在**，
是 H3 换后端把它顶出去 5.5 ms。三条分开写：

* **H3 在检索延迟上无收益，实际更慢（本次 +44% P95，口径 B）；**
* 阈值「< 200 ms」由 H2 达成，**H3 没有推进它，反而在 FTS 后端上丢掉了余量**；
* 若生产切 FTS，须接受这一项回退，或按 §7 继续投入。

**冷启动与内存是 H3 真正的战果**（口径 A，本次运行）：

* 冷启动 `total_ms` **7,317.7 → 1,750.0 ms（−76.1%）**；其中 `index_ms`
  从 **5,467.2 ms 掉到 18.4 ms**（打开 SQLite 只读连接）。剩余 1.75 s 里
  `assertions_ms` 937.5 + `chunks_ms` 296.1 + `graph_ms` 288.0 是主体——
  这三项是**图与块正文**，不在 H3 范围内（下一步的事）。
* 内存 `after_load` **1,499.32 → 532.64 MiB**，
  **峰值高水位 1,719.37 → 532.64 MiB（−69.0%）**。峰值这一项特别值得看：
  json 的 `peak` 恒等于 1,719.37（峰值发生在解析 JSON 那一刻，之后降不下来），
  而 fts 的峰值就等于当前值——**没有解析尖峰**。532.64 MiB 低于计划里 692.4 MiB 的参照线。

> 冷启动绝对值在多次运行间波动很大：本文件先后记到 12,909.0 / 11,171.6 / 7,317.7 ms
> （json 侧），差异来自文件页缓存是否命中，**不是代码变化**。可以引用的稳定结论是
> 「`index_ms` 从 5.5–10.5 s 降到约 19 ms」与「冷启动总量降到 1.7–1.8 s」，
> 以及 FTS 侧那 1.75 s 的构成；不要拿某一次的精确下降百分比去和另一次运行比较。

`context_ms` 本次 18.003 → 18.519（P50）、52.867 → 61.615（P95），**基本持平、FTS 略高**，
与端到端差异同向。本文件早期版本曾把 `context_ms` 的大幅下降解释成「内存压力副作用」，
那是旧数据上的推论；**按当前数据没有这回事**。`estimate_tokens` 依赖的 `name_terms`
在两边都是 0（这个世代的 index 是 t1 建的），所以这一阶段本该完全等价——
实测也确实等价，**不要把它算成任何一方的功劳**。

## 6. 门禁（测试）说明

### 6.1 两条文件的定位

| 文件 | 守什么 |
| --- | --- |
| `tests/test_search_backend.py`（9 项） | 协议面、工厂语义、旧后端逐位不变、`auto` 的两种走向、默认产物位置 |
| `tests/test_fts_backend.py`（20 项） | 与 `LexicalIndex` 的**等价性**、BM25 参数来源、库形态、拒绝写世代目录、只读、确定性 |

```powershell
py -3 -m pytest tests\test_search_backend.py tests\test_fts_backend.py -q
# 29 passed
```

### 6.2 核心等价性断言

* `test_exact_mode_matches_lexical_index_bitwise`：同一语料，两个后端的
  `(chunk_id, score)` 序列**逐位相同**（包括分数，容差 1e-12）。
* `test_exact_mode_matches_on_top_k_boundary`：`top_k` 恰好卡在第 k 名与
  k+1 名同分的位置时，同分次序（按 chunk_id 升序）也必须一致——
  这是最容易漏的一处，因为分数相同时 `sorted` 的稳定性会掩盖差异。
* `test_norm_table_matches_index_norm_table`：双后端的长度归一化表逐槽相同，
  **且包含空洞槽位**（语料故意跳过 slot 1）。
* `test_token_set_identical_to_source_index`：库里的 694,304 个词条与源倒排
  **完全相等**（多一个少一个都算错：多出来的词条会让 df 偏小、idf 偏大，
  而检索结果看起来仍然「正常」）。
* `test_fts5_native_mode_is_flagged_as_different_scoring`：断言原生 `bm25()`
  与 exact 模式的结果**不同**，并把差异量化记在测试里。

### 6.3 变异测试：门禁真的能红

「测试全绿」不等于「门禁有效」。项目里已经出过三次「只往列表记、从不抛异常 →
pytest 全绿而缺陷躺着」的事故（见 `tests/test_suite_hygiene.py` 文件头）。
所以本次逐项把实现改坏，确认测试变红：

| 变异 | 结果 |
| --- | --- |
| `norm` 兜底值改成 0 | 1 failed |
| 同分次序改成按 slot/id 排 | 2 failed |
| idf 换成 sqlite 口径（去掉 `1 +`） | 3 failed |
| `auto` 永远选 json | 1 failed |
| fts 打桩失效时静默退回 json | 2 failed |
| 允许写进世代目录 | 1 failed |
| 不检查库形态版本 | 1 failed |
| 默认走原生 bm25 | 3 failed |

**首轮抓到一条假门**：`norm` 兜底值改成 0 时**全绿**——因为测试语料的槽位连续、
每个槽都有长度，兜底分支根本没被走到。补语料（`add(0,…)` 与 `add(2,…)`，
故意跳过 slot 1）后它才变红。这条记在这里，是因为「门禁看起来通过」正是
本项目三次事故的共同形状。

## 7. 未采用的优化（本次实测否决，避免后人重走）

| 想法 | 实测 | 结论 |
| --- | --- | --- |
| SQLite 三列 `(term,doc,freq)` postings 表 + SQL 聚合打分 | 2000 块子集上 p50 359.9 ms（Python dict 8.1 ms，慢 40×）；建库 59.5 s/2000 块 | 否决 |
| `fts5vocab(t,'row')` 当倒排 | 只有词级聚合，无逐文档词频；查 `'凯'` 返回 0 行 | 否决 |
| `fts5vocab(t,'instance')` 当倒排 | 取高频词 `'的'` 要 348.7 ms（344,689 行） | 否决 |
| FTS5 原生 `bm25()` 当默认 | idf 公式不同，top-20 只重合 10–12/20 | 否决（保留为可选模式） |
| 建库期预算权重 BLOB（查询期每对只做一次加法） | 139.3 ms/问 vs 现方案 54.9 ms/问，**慢 2.5×** | 否决 |
| dense list 累加（代替 dict） | 89.36–99.34 ms vs dict 50.87 ms | 否决 |
| `zip()` 代替 `enumerate()` | 120.61 ms vs 50.87 ms | 否决 |

最后三条都是「看起来能省一次浮点乘法，实际更慢」——CPython 里
`acc.get()` + 单次下标赋值比任何花哨的列表推导都快，因为权重预计算
要额外造一遍数组、多一次迭代、多一次内存分配。

## 8. 回滚

三条独立的路，任一条都能退回去：

1. **改参数**：`open_backend(dir, kind="json")` —— 不用改代码、不用动数据。
2. **删产物**：删掉 `data/arkrag/fts/<gid>.sqlite` —— `kind="auto"` 会自动退回 json。
3. **删模块**：`src/arkrag/fts_backend.py` 与两个新脚本都可以整体删除，
   `search_backend.py` 里的 `from . import fts_backend` 是**延迟 import**（在函数内），
   所以不装/不留 FTS 模块时旧路径照常工作。

旧路径 `index/lexical.json` **一个字节都没动**（`snapshot_baseline.py --verify`
在实验前后各跑一次，11 项全部一致）。

## 9. 接线（给 `scripts/ask.py` 的一行）

`ask.load_generation` 的返回值里，检索器只认 `loaded["index"]` 这一个槽。
所以最小改动是：

```python
loaded["index"] = search_backend.open_backend(loaded["generation"].dir, kind="fts")
```

放在 `load_generation` 返回之前（或 `build_pipeline` 开头）。需要
`from arkrag import search_backend`。

**但要拿到冷启动那 10.5 s，还需省掉 `index_mod.load_index(gen.index_path)` 这一步**：
上面这行是在 JSON 已经解析完之后覆盖，解析成本照付。`bench_h3.py` 的做法是
在载入阶段把 `index_mod.load_index` 打桩成返回已打开的 backend（见
`run_one_backend` 里 `_patched_load_index`），实测 `index_ms` 19.4 ms。
生产接线时可以按同样思路改成「有 `.sqlite` 就不读 `lexical.json`」：

```python
loaded["index"] = search_backend.open_backend(gen.dir, kind="auto")
```

配合 `load_generation` 里把 `index_mod.load_index` 那一步跳过。具体怎么改由
`ask.py` 的维护者定；本模块只保证 `open_backend()` 是唯一入口、
返回对象满足 `Retriever` 的全部依赖（320/320 逐位相同已经证明这点）。

## 10. 遗留与已知限制

1. **产物体积变大**（121.4 → 212.2 MiB，0.57x）。倒排完整保留是「随机访问」的代价。
   若将来要压，方向是把 `freqs` 从 `I`（4 字节）换成 `H`（2 字节）——
   本语料最大词频远小于 65535，可省约 21 MiB。本次没做，因为要先证明
   `max(freq) < 65536` 全库成立。
2. **建库 36 s**（其中打包 13.8 s + FTS5 写入 15.9 s）。这是一次性成本，
   但换世代就要重付一次。若建库要进 CI，建议加缓存（按 `index_content_digest` 判）。
3. **`chunk_texts` 表（FTS5 虚拟表）目前只用于 `score_mode="fts5"`**，
   默认路径走 `postings`。它占了相当一部分体积（约 40%）。若确认永远不用原生 bm25，
   可以砍掉它，体积能降到 ~130 MiB——但那会让「换口径」这条路彻底关闭，故本次保留。
4. **`entities` / `aliases` / `assertions` / `relations` 四张表已建好但检索路径未用**。
   它们是为后续「把图检索也搬进 SQLite」准备的（H4+），本次只验证能落库。
   `aliases=3344` 而 manifest 里别名总数是 1672——因为一条别名会同时登记
   「实体→别名」与「别名→实体」两个方向。
5. **`LLM` / `embedding` 相关的指标一律 `null`**：本机无 LLM 端点、无 embedding 后端，
   这是既成事实，七项指标里的 `abstain_accuracy` 与 `direction_accuracy` 是
   抽取式答句器算的，不含生成质量。
6. **`test_suite_hygiene.py` 的 `N` 已由 Lead 上调到 1056**（原 1019）。
   增量核算：`test_search_backend.py`（9）+ `test_fts_backend.py`（20）= 29 个用例，
   这两个文件进入 `TEST_FILES` 后又让四个 parametrize 各多两项（+8），1019 + 37 = 1056，
   与 `--collect-only` 实测一致。
7. **本世代的 index 是 `t1` 分词建的，不是当前默认的 `t2`**。`fts_backend`
   一律从 index meta 读 `tokenizer_version` 并沿用，**不硬写 `TOKENIZER_VERSION`**
   （否则召回不可比）。这条有测试守着（`test_unknown_tokenizer_version_raises` 的反面）。

## 11. 复跑清单

```powershell
# 环境（PowerShell 默认 GBK 会把中文显示成乱码）
$env:PYTHONIOENCODING='utf-8'

# 0) 实验前确认工作区干净
py -3 scripts\snapshot_baseline.py --verify

# 1) 新测试
py -3 -m pytest tests\test_search_backend.py tests\test_fts_backend.py -q

# 2) 全量（只应有 test_suite_hygiene 的 N 一项失败）
py -3 -m pytest tests -q

# 3) 建库
py -3 scripts\build_fts_index.py --generation g-20261031T035231Z-4eafae

# 4) 双跑对比
py -3 scripts\bench_h3.py --generation g-20261031T035231Z-4eafae --questions 320

# 5) 实验后确认工作区干净
py -3 scripts\snapshot_baseline.py --verify
```
