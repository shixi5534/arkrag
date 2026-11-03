# H 阶段实测：性能基线、低风险优化与 FTS5 迁移

> 基准世代：`g-20261031T035231Z-4eafae`（ark-rag-3，13,781 块 / 6,582 实体 / 28,625 边 / 62,910 断言，READY）
> 题集：`data/goldset/gold-v1.json` 全 320 题（八类别，`_stratified_sample` 固定种子 20261101）
> 环境：Windows，Python 3.14.3，`sqlite3` 3.50.4，无 LLM 端点、无 embedding 后端
> 测量器：`scripts/bench_h1.py`（H1 建立，H2/H3 复用同一把尺子）

## 0. 口径（先说清，否则下面的数字没有意义）

| 项 | 定法 | 为什么 |
|---|---|---|
| 冷查 / 热查 | 冷 = 本进程对该题的首遍；热 = 其后各遍取中位数 | 取平均会把首次的解释器预热抹平，冷启动成本就看不见了 |
| 分位数 | nearest-rank：`rank = max(1, min(len, ceil(pct*len/100)))` | 与 `statistics.quantiles` 的插值口径不同，插值会在小样本上造出不存在的值 |
| 空样本 | 返回 `null` | 返回 `0.0` 会让"没测到"看起来像"极快" |
| 构建耗时 | 读真实 run 的 `events.jsonl`，不重跑 | 重跑会写新 Generation，污染被测产物清单 |
| 峰值内存 | `GetProcessMemoryInfo` 的 `PeakWorkingSetSize` | 不是"当前 RSS 的最大观测值"——采样会漏峰；也不用 psutil（运行时零第三方依赖） |
| LLM token / 成本 | `null` | 本机无 LLM 端点，写 `null` 并附原因，不估算 |

---

## 1. H1：性能基线（优化前）

命令：

```powershell
py -3 scripts\bench_h1.py --generation g-20261031T035231Z-4eafae --questions 320 --warm-reps 3 --json data\bench_h1_4eafae.json
```

### 1.1 冷启动加载

| 产物 | 耗时 | 体积 |
|---|---|---|
| `graph/graph.json` | 295.7 ms | 30.14 MiB |
| `graph/adjacency.json` | 213.6 ms | 25.18 MiB |
| `graph/assertions.jsonl` | 964.3 ms | 65.55 MiB |
| **`index/lexical.json`** | **9,972.5 ms** | **121.35 MiB** |
| `chunks/*.jsonl` | 631.5 ms | 38.16 MiB |
| **合计** | **12,077.7 ms** | — |
| `Retriever` 构造 | 164.8 ms | — |

索引一项占加载时间的 **82.6%**。计划里"仅 `json.loads` 约 23.88 秒"的口径是单文件串行解析，
这里是五个产物逐项计时后的合计——两者不矛盾，但**必须分开报**，否则会误以为完整加载要 24 秒。

### 1.2 内存

| 时点 | 当前 | 峰值高水位 |
|---|---|---|
| 进程起始 | 25.66 MiB | 25.66 MiB |
| 五个产物加载完 | 1,499.94 MiB | — |
| 管线就绪 | 1,504.91 MiB | **1,717.59 MiB** |

计划参照值 692.4 MiB，**实测是它的 2.48 倍**。差距来源不是哪一处泄露，而是
「127 MB JSON → Python 原生 dict/list」的展开比：`postings` 是
`dict[str, dict[int, int]]`，694,304 个 `dict` 条目各自带对象头，光结构开销就远超序列化字节。

### 1.3 端到端延迟（ms）

| | P50 | P95 | P99 | min | max | mean |
|---|---|---|---|---|---|---|
| 冷查 n=320 | 243.868 | 500.088 | 677.653 | 77.449 | 773.792 | — |
| 热查 n=960 | 239.281 | **366.750** | 417.489 | 77.383 | 466.136 | — |

### 1.4 分阶段（热查，ms）

| 阶段 | P50 | P95 |
|---|---|---|
| `query`（问句解析） | 67.902 | **161.505** |
| `search`（BM25 检索） | 146.797 | **251.383** |
| `expand`（图扩展） | 0.572 | 2.792 |
| `context`（上下文组装） | 15.975 | 54.831 |
| `answer`（抽取式作答） | 0.256 | 0.583 |

两个热点合计 412.9 ms，占端到端 P95 的 112%（有重叠，因为 P95 不是逐题相加）。
**其余三阶段合起来不到 60 ms，优化它们没有意义。**

### 1.5 Top-K 成本

| top_k | P50 | P95 | 平均候选数 |
|---|---|---|---|
| 5 | 180.877 | 249.866 | 28.2 |
| 20 | 204.754 | 277.789 | 107.9 |
| 50 | 198.236 | 294.721 | 267.8 |
| 100 | 209.393 | 303.356 | 537.4 |

**耗时与 top_k 几乎无关**（180–210 ms 常量），候选数却从 28 涨到 537。
结论明确：成本在「扫全部倒排 + BM25 打分」，不在排序截断。所以调小 top_k 省不了时间，
瓶颈只能从打分循环或倒排结构上解决。

### 1.6 图扩展与 Token

| 项 | P50 | P95 | max |
|---|---|---|---|
| 扩展节点数 | 2.0 | 2.0 | 5 |
| 扩展边数 | 20.0 | 40.0 | 52 |
| 上下文 token | 5,760 | 5,984 | — |
| 上下文块数 | 5 | — | — |

图扩展的预算（`max_nodes=80`、`max_edges=200`）远未用满——实际只走了 2 个节点、20 条边。
这说明**当前瓶颈不在图扩展，而在它的上游**（能用满预算的前提是先有足够好的种子实体）。

### 1.7 构建耗时

run `r-20261031T035149Z-d7c94b`，10 条事件，总跨度 176.0 s；`snapshot` 23.0 s、`build` 79.0 s。

### 1.8 LLM Token 与成本

**不可测**。本机 `ARKRAG_LLM_MODEL` / `ARKRAG_LLM_BASE_URL` 均为空，`arkrag.llm` 缺配置时抛
`RuntimeError`；回答器是抽取式 `ExtractiveAnswerer`，不产生 LLM 调用。
报告里这一组字段全部写 `null` 并附 `llm_unavailable_reason`，不估算。

---

## 2. H2：低风险优化

计划列出七项。**逐项实测后只有两项真的需要做**，其余四项早已落地、一项不在热路径。

| 计划项 | 核查结论 | 依据 |
|---|---|---|
| 预计算 `avg_doc_len` | **已落地**，无需再做 | `LexicalIndex.avg_len` 是 `@property` 惰性缓存，`add()` 时置 `None` 失效（`index.py:411-421`） |
| 避免每次搜索扫描全部文档长度 | **已落地** | 同上 |
| 预建 `(category, chapter) -> chunks` | **不在查询热路径** | 只用于建库期分片（`chunk.py:277` `pack_shards`、`chunk.py:297-305` `shard_membership`），查询时不走 |
| 名称合并分桶前缀索引 | **已落地** | `NameTable` 的「两字前缀 ∩ 块内 bigram」求交 + trie 最长匹配 |
| QA 预建 `chunk_id -> index` | **已落地** | `answer.py:128` 已有 `by_id = {...}` |
| L3 有界队列 | **已落地** | `extract.run_shard` 的 `window` 参数 + 熔断后停止补料，`skipped_circuit` 与 `miss_fail` 分开计数 |
| 避免重复读取全部缓存 | **未做，本阶段补上** | 见 2.1 |
| 流式分片序列化 | **现状可接受** | `save_index` 用 `atomic_write_json` 流式 `json.dump` 到临时文件后原子替换，不是先拼整串 |

### 2.1 优化一：`QueryContext` 复用（问句解析）

#### 先定位，再动手

H1 显示 `query` 阶段 P95 = 161.5 ms，与 `search` 的 251.4 ms 同一量级。它**不该这么慢**——
`analyze_query` 做的是「在一张实体表里找出问句提到的名字」，是典型的微秒级工作。

320 题 × 3 轮逐项计时：

| 步骤 | 每 320 题 | 每问 |
|---|---|---|
| 全量 `analyze_query` | 29,523.8 ms | 92.3 ms |
| 其中 `_alias_owner_map`（遍历 6,582 实体 / 8,254 文本） | 15,423.6 ms | 48.2 ms |
| 其中 `NameTable` 建表（构建 8,177 项 trie） | 15,925.2 ms | 49.8 ms |
| `_relation_lookup` | 34.8 ms | 0.1 ms |
| `_normalize` | 5.4 ms | 0.02 ms |

**96% 花在反复重建只依赖图、与问句无关的结构上。** 这三样在一个世代内是不变的
（实体表和词表由 Manifest 锁定），"每问重建"没有一处是为了正确性。

#### 改法

`query.QueryContext`：显式构造的只读容器，持有 `claims` / `name_table` / `rel_table`，
`analyze_query(..., context=ctx)` 传了就复用。生命周期跟着 `Retriever` 走（`Retriever.__init__`
建一次，`ask.build_pipeline` 传入 `vocab` 保证关系表口径一致）。

**为什么不做模块级 `lru_cache`**：key 需要可哈希。把 6,582 个 dict 转内容指纹的代价与
本函数相当，等于没省；用 `id(entities)` 作 key 会在对象回收后**命中另一个世代的图**——
那是静默错配，比慢更坏。

#### 结果

| | 改前 | 改后 | 变化 |
|---|---|---|---|
| `query` P95 | 161.505 ms | **1.154 ms** | **−99.3%** |
| `query` P50 | 67.902 ms | 0.678 ms | −99.0% |
| 端到端热查 P95 | 366.750 ms | 303.905 ms | −17.1% |
| 端到端热查 P50 | 239.281 ms | 189.213 ms | −20.9% |
| 端到端冷查 P95 | 500.088 ms | 306.227 ms | −38.8% |

代价：`Retriever` 构造从 164.8 → 227.8 ms（一次性，换来每问省 160 ms）。

#### 正确性验证

不是"跑通就行"。`tests/test_query_context.py`（23 个用例）用**生产形状**的语料
（一对多别名、别名与规范名同文、前缀嵌套名、拉丁名、与否定词同位置的实体）
对 11 个问句逐一断言：复用与就地重建的**整个返回结构深度相等**（含候选顺序、`score` 取值、
`stats` 每一项计数）。

失效形态为什么必须这样测：建错一个别名、少一个两字前缀，结果都不是异常，而是
**某个问句的实体候选少一个**，下游照常检索作答，召回从 20 掉到 19。评测分数会动，
但没人能从一个下降的小数点看出是这里坏的。

### 2.2 优化二：BM25 归一化项预建（检索打分）

#### 定位

先把规模量准（这三个数经常被混为一谈，混了就会把优化收益算错）：

| 量 | 值 |
|---|---|
| 词项数 `len(postings)` | **694,304** |
| **postings 条目总数** `sum(len(p) for p in postings.values())` | **10,525,428** |
| 文档数 `len(doc_id)` | 13,781 |
| `avg_len` | 1,272.3257 |
| 每问命中的 query term | 56.73 |
| **每问内层循环迭代次数** | **193,402.7**（占 postings 条目总数的 1.84%） |

每问扫 19.3 万个位置，每一条都重算一次 `k1*(1 - b + b*length/avg_len)` ——
这个除法**只与文档有关，与问句无关**。

单次 `search("凯尔希和罗德岛的关系")` 的 `cProfile`：总计 0.057 s，
`search` 本体 tottime 0.026 s、`sorted` 0.020 s、`<lambda>` 0.007 s（13,776 次比较）。

真实索引上的 A/B（320 题，同一进程，预热后）：

| | P50 | P95 | mean |
|---|---|---|---|
| 旧实现 | 150.789 ms | 228.979 ms | 152.747 ms |
| 新实现 | **93.378 ms** | **138.992 ms** | **94.573 ms** |
| 变化 | **−38.1%** | **−39.3%** | −38.1% |

> 修正记录：本节初版把「694,304」（词项数）写成了「postings 条目数」，并引用
> `818,941` 当作"全部 postings 条目数"——那个数是 cProfile **20 次迭代**的累计
> 调用次数，既不是单问规模，也不是总数。三个数（词项数 / 条目数 / 迭代数）差了两个
> 数量级，混用会让"为什么优化有效"的解释变成编出来的。上表是重新逐条量准的结果。

#### 改法

`LexicalIndex.norm_table()`：按 slot 预建扁平 `list[float]`，`add()` 时与 `_avg_len` 一同失效。
`search()` 内层循环只剩一次列表索引与一个乘加。

用 `list` 而非 `dict`：slot 由 `enumerate(sorted(chunk_records, ...))` 连续分配，
list 下标就是 slot。有空洞时按原实现的 `doc_len.get(slot, 1)` 落 `len=1` 的默认值补齐。

另把 `k1, b = 1.5, 0.75` 从函数局部变量提为模块常量 `BM25_K1` / `BM25_B`：
H3 的 FTS5 后端必须用同一组参数，让别的模块 import 一个函数内部的名字不可能，
只能复制字面量——那就又成了「同一判定写两处」。**数值未改动。**

#### 结果

端到端口径（`bench_h1.py`，320 题）：

| | 改前 | 改后 | 变化 |
|---|---|---|---|
| `search` P95 | 251.383 ms | **167.523 ms** | **−33.4%** |
| `search` P50 | 146.797 ms | 111.800 ms | −23.8% |
| k=5 P50 | 180.877 ms | 113.171 ms | −37.4% |
| k=100 P50 | 209.393 ms | 122.820 ms | −41.3% |
| 端到端热查 P95 | 303.905 ms | **211.785 ms** | −30.3% |
| 端到端热查 P99 | 357.585 ms | 239.811 ms | −32.9% |

`bench_h1.py` 的 `search` P95（167.5 ms）高于上面 A/B 的新实现 P95（139.0 ms），
因为前者是**完整链路里**的 `search`（含 `Retriever` 的实体通道与归一化），
后者是**裸 `LexicalIndex.search`**。两者不同口径，不能互相引用。

#### 正确性验证

`tests/test_index_norm_table.py`（40 个用例）把**改动前 `search()` 的逐字副本**
`_legacy_search` 留在测试里当标尺，对 8 个问句 × 3 个 top_k 断言新实现输出
**逐位相同**（连 float 的 `repr` 都比）。另覆盖：空洞 slot 兜底、单文档、空索引、
`add()` 后缓存失效、落盘往返（JSON 把 int key 变字符串那条路径）。

为什么必须这样测：`avg_len` 与 `norm_table` 是两份缓存，如果 `add()` 只失效了前者，
新文档会用旧平均长度打分。分数只差千分之几，排序大体不变，召回掉一两个位置——
没有异常、没有日志、`verify` 也不一定红。

### 2.2b 真实数据核查（不进测试套件）

单元测试用合成语料，快且可控，但只证明"逻辑对"。另做三项**真实世代**核查，
固化成脚本 `scripts/verify_h2_equivalence.py`（要加载 121 MB 索引、跑 320 题两遍，
单次约三分半，不适合放进 CI——放进去只会让人想办法跳过它）：

```powershell
py -3 scripts\verify_h2_equivalence.py --generation g-20261031T035231Z-4eafae
```

| 核查 | 规模 | 结果 |
|---|---|---|
| `QueryContext` 复用 vs 就地重建，整个 `analyze_query` 返回结构深度相等 | 320 题；真实 6,582 实体 / 8,254 别名文本 / 257 关系 | **0 处不等** |
| 新 vs 旧 `search` 的 `(chunk_id, score)` 序列，含 float `repr` | 320 题 × top_k=20；真实 13,781 文档 | **0 处不一致** |
| `norm_table()` 逐项按 `k1*(1-b+b*len/avg_len)` 复算 | 13,781 项 | **0 项不一致** |

前者能进 CI，后者才叫"在生产形状下没坏"。脚本里的 `_legacy_search` 是**冻结的标尺**，
一旦有人"顺手优化"它，这个脚本就再也证明不了任何事——注释里写明了这一点。


### 2.3 顺带修掉：`extract.run_shard` 的 `list.pop(0)`

`pending.pop(0)` 每次搬动整个剩余队列。13,781 块时累计约 9,500 万次元素搬移，
且全部发生在持锁的判断之间。改为 `collections.deque` + `popleft()`（O(1)）。
不是主要瓶颈，但没有理由留着。

### 2.4 H2 总账

| 指标 | H1 基线 | H2 完成 | 变化 |
|---|---|---|---|
| 端到端**热查 P50** | 239.281 ms | **131.572 ms** | **−45.0%** |
| 端到端**热查 P95** | 366.750 ms | **211.785 ms** | **−42.3%** |
| 端到端**热查 P99** | 417.489 ms | 239.811 ms | −42.6% |
| 端到端**冷查 P50** | 243.868 ms | 132.394 ms | −45.7% |
| 端到端**冷查 P95** | 500.088 ms | 212.887 ms | −57.4% |
| 端到端**冷查 P99** | 677.653 ms | 253.194 ms | −62.6% |
| `query` P95 | 161.505 ms | 0.958 ms | −99.4% |
| `search` P95 | 251.383 ms | 167.523 ms | −33.4% |
| **检索阶段 P95** | **251.383 ms** | **167.523 ms** | **−33.4%，已进入 H3 目标（<200 ms）** |
| 冷启动加载 | 12,073 ms | 12,073 ms | 未变 |
| 峰值内存 | 1,717.59 MiB | 1,717.77 MiB | 未变 |

### 2.5 召回没有退化（逐位验证）

H2 的两项改动都是纯优化，验收条件就是**评测分数一个字都不许变**：

```powershell
py -3 scripts\ask.py --evaluate --gold-set data\goldset\gold-v1.json --generation g-20261031T035231Z-4eafae --limit 320
```

| 指标 | H1 基线 | H2 后 |
|---|---|---|
| Recall@20 | 0.6575 | **0.6575** |
| Recall@5 | 0.4658 | **0.4658** |
| MRR | 0.4123 | **0.4123** |
| 引用定位正确率 | 1.0000 | **1.0000** |
| 关键事实引用覆盖率 | 0.1567 | **0.1567** |
| 拒答正确率 | 0.8406 | **0.8406** |
| 方向题正确率 | 0.6667 | **0.6667** |
| 引用校验硬失败 | 0 | **0** |

七项指标逐位相同。这是 H2 唯一可信的验收证据——**慢变快了不能证明对，
只有"快了但结果一个字没变"才能证明对。**

---

## 3. 一个被顺手发现的真实缺陷（H1 期间）

`scripts/ask.py::_latest_identity_generation()` 原本：

```python
for info in release.list_generations():
    ...
    return gid
```

而 `release.list_generations()` 按目录名升序（`g-<时间戳>-<hash>`），**最旧在前**。
于是"取最新"实际取到最旧。

**为什么当时看不出后果**：磁盘上三个 ark-rag-3 世代（`b22ca8` / `348206` / `4eafae`）
内容完全一致——`index_content_digest`、`probe_digest`、`graph_build_fp`、
`chunk_manifest_hash` 全部相同，跑谁的数字都一样。

**这是最坏的缺陷形态：结论正确，机制错误。** 只要出现第四个内容不同的世代，
`--evaluate` 不带 `--generation` 就会静默评测旧数据，而报告里没有一个字段会变红。

已改为：跳过无 Manifest / Manifest 损坏的目录，只留 `graph_schema == GRAPH_SCHEMA_IDENTITY`，
按 `(created_at or gid, gid)` 排序取最后一个。`tests/test_ask_generation_selection.py`
（8 个用例）守门，其中 `test_returns_newest_not_oldest` 在旧实现下必红，
`test_directory_order_does_not_decide` 防"改成 `list[-1]` 就行"。

---

## 4. 复现命令

```powershell
# 基线（H1 口径，三个时间点）
py -3 scripts\bench_h1.py --generation g-20261031T035231Z-4eafae --questions 320 --warm-reps 3 --json data\bench_h1_4eafae.json
py -3 scripts\bench_h1.py --generation g-20261031T035231Z-4eafae --questions 320 --warm-reps 3 --json data\bench_h2_4eafae.json
py -3 scripts\bench_h1.py --generation g-20261031T035231Z-4eafae --questions 320 --warm-reps 3 --json data\bench_h2b_4eafae.json

# 召回不变性
py -3 scripts\ask.py --evaluate --gold-set data\goldset\gold-v1.json --generation g-20261031T035231Z-4eafae --limit 320

# 等价性门（合成语料，进 CI）
py -3 -m pytest tests\test_query_context.py tests\test_index_norm_table.py -q

# 等价性门（真实世代，手工跑，三分半）
py -3 scripts\verify_h2_equivalence.py --generation g-20261031T035231Z-4eafae

# 全量
py -3 -m pytest tests -q
py -3 scripts\snapshot_baseline.py --verify
```

---

## 5. 未解决的遗留

> 本节记录的是 **H2 结束时**的遗留。H3（FTS5）落地后第 1、2 条已解决、第 3 条结论有变，
> 每条的处置写在原条目下的「→ H3 后」里。H3 自身的遗留见 `docs/h3-FTS5迁移.md`。

1. **内存 1,717 MiB 没降下来（H2 范围内无解）。** H2 是「少算」不是「少存」，峰值由 `postings` 的
   Python 对象开销决定，只有换存储层才能动它。
   → **H3 后：已解决。** FTS 后端整条管线峰值 **532.64 MiB**（−69.0%），已低于计划参照 692.4 MiB；
   且 fts 侧峰值等于当前值，**没有解析尖峰**（json 的 peak 恒等于 `json.loads` 那一刻，之后降不下来）。
2. **冷启动 12.1 秒没降下来（H2 范围内无解）。** 同上，`json.loads` 121 MB 是硬成本。
   → **H3 后：已解决倒排层这一项。** `index_ms` 5,467.2 → **18.4 ms**，冷启动总量 7,317.7 → **1,750.0 ms**。
   剩余 1.75 s 的主体是 `assertions_ms` 937.5 + `chunks_ms` 296.1 + `graph_ms` 288.0，
   即**图与块正文**，不在 H3 范围，留给下一阶段。
3. **`search` P95 = 167.5 ms 仍未达"端到端 P95 < 200 ms"。** H3 的初始目标写的是
   「热查询**检索阶段** P95 < 200 ms」——按这个口径 H2 已达标；但**端到端** P95 仍是 211.8 ms，
   差距在 `context`（59.1 ms）与 `search` 剩余部分。这两个数字不许混为一谈。
   → **H3 后：口径结论有变。** 换 FTS 后端后检索阶段 P95 是 **205.456 ms，未达 200 ms**
   （交替 A/B 口径 +44%）。**这不是 H2 退步**——json 侧 H2 是 167.5 ms、H3 同批测量是 160.3 ms，
   H2 成果仍在，是换后端把它顶出去 5.5 ms。端到端 P95 在 FTS 上为 267.1 ms（交替口径）。
4. **`context` 阶段 P95 从 54.8 涨到 59.1 ms。** 幅度在噪声内（三次测量间也在
   55.9/59.1 之间浮动），但**没有归因**，记录在此不做结论。
   → **H3 后：** 仍无归因。FTS 侧 52.867 → 61.615（跨进程）、基本持平略高；
   该世代 index 是 t1 建的、`name_terms` 长度 0，`estimate_tokens` 两边等价，
   **不许把这一阶段算成任何一方的功劳**（`docs/h3-FTS5迁移.md` §5.3 早期版本曾把
   context 大幅下降解释成"内存压力副作用"，那是旧数据推论，已撤回）。
5. **Top-K 仍然与耗时解耦。** 说明瓶颈还在打分循环与倒排遍历本身。
   → **H3 后：仍是空白。** H3 只测了固定 `top_k=20` 的检索延迟，**没有按 top_k 重扫**。
   若要让 top_k 重新敏感，需先解决 `fts_backend._search_exact` 每 term 重建 `array("I")` 的开销。
6. **三个 ark-rag-3 世代内容完全冗余**（digest 全同），占约 300 MB。prune 需 Lead 决策，
   本轮未动。
