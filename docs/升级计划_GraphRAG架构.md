# ArkRAG GraphRAG 架构升级实施计划

> 本文件是用户批准的计划原文存档（2026-10-30 记录）。
> 后续所有实现必须对照本文件验收；改动计划必须改本文件并说明理由，不许只在对话里改口径。
> 进度与偏差记账在 `docs/升级台账.md`。

---

## 0. 背景与总目标

现有系统是一条**离线剧情知识加工流水线**：L0 语料验证 → L1 解析 → L2 分块 → L3 实体/关系抽取 → L4 归一化 → L5 建图 → L6 建索引。产物是 `graph.json` 与 `index/lexical.json`，只服务于"建出来"，不服务于"回答问题"。

升级目标：把它变成**数据可追溯、产物可回滚、检索可评测、回答有证据、异常可恢复、服务可观测**的 GraphRAG 系统。

目标链路：

```
语料验证 → 解析与分块 → 严格校验的实体/关系抽取 → 实体消歧与逐观察证据图
  → 词法/向量混合检索 → 有方向的图扩展 → 重排与上下文组装
  → 带引用回答与无依据拒答 → 持续评测、监控与版本回滚
```

---

## 1. 实施原则（八条，逐条有约束力）

1. **不做大爆炸重写。** 保留现有 L0–L6 流水线，逐步替换数据契约与发布机制。
2. **先修正确性再做性能。** 数据身份 / 证据归属 / 失败发布未解决前，不引入向量库和在线服务。
3. **新旧产物并存。** 用新版本目录与 Schema，不直接覆盖现有可用产物。
4. **每阶段可独立验收。** 阶段失败不得污染上一阶段稳定版本。
5. **所有门禁必须能失败。** 禁止"只打印问题但退出码为 0"冒充验收。
6. **所有优化必须有基线对比。** 没有基线数字的优化不算优化。
7. **原始数据与生成数据分离。** 原始语料只读。
8. **现有全 PASS 不作为免检依据。** 新增测试必须覆盖当前门禁的盲区，而不是重复已经通过的项目。

---

## 2. 阶段计划 A–K

| 阶段 | 名称 | 优先级 | 工作量 | 前置 |
|---|---|---|---|---|
| A | 基线冻结、备份与工程可复现 | P0 | 2–3 天 | 无 |
| B | 数据契约、严格回收与缓存身份 | P0 | 4–6 天 | A |
| C | 事务式构建、原子发布与故障恢复 | P0 | 4–6 天 | A（可与 B 部分并行） |
| D | 解析、分块与指纹完整性 | P0/P1 | 3–5 天 | A、B 契约 |
| E | 证据图与实体身份模型升级 | P1 | 6–10 天 | B、C、D |
| F | 测试体系与真实质量评测集 | P1 | 5–8 天 | A（Gold Set 可提前并行） |
| G | 最小可用检索问答闭环 | P1 | 6–10 天 | B–F 核心部分 |
| H | 检索与存储性能优化 | P1/P2 | 4–7 天 | G 有基线 |
| I | 混合检索、重排与图扩展增强 | P2 | 6–10 天 | G、F |
| J | 服务化、安全、监控与持续交付 | P2 | 5–8 天 | G、H |
| K | 全量重建、灰度验收与旧版本退役 | P0（收尾） | 3–5 天 | 全部 |

### A. 基线冻结、备份与工程可复现（P0）

**A1 冻结现场指标。** 必须记录：语料文件数 / 场景数 / 轮次数、Chunk 数、抽取缓存数、图实体数与边数、索引大小 / 词条数 / Posting 数、当前 75 项门禁报告、测试输出、配置与版本常量。

**A1b 保存产物清单。** 每个产物记录：路径、字节数、SHA-256、mtime。

**A1c 标为 legacy-baseline 只读。** 禁止后续实验直接写：

- `data/arkrag/graph.json`
- `data/arkrag/index/lexical.json`
- 当前 `agent_cache`
- 当前 `gates.json`

**A1d 交付物：**

- `data/releases/legacy-baseline/manifest.json`
- `data/releases/legacy-baseline/gates.snapshot.json`
- `docs/baseline-metrics.md`

**A2 建立 Git 初始提交与标签。** 源码 / 测试 / 文档入库；语料 / 缓存 / 索引 / 大图 / 日志 / 密钥不入库。

**A3 固化运行环境。**

- Python 最低版本
- 依赖声明 + 锁文件
- 统一根目录解析，不依赖调用者 cwd
- 统一 src 导入路径
- README 覆盖：环境准备 / L0–L6 命令 / 测试命令 / 数据目录 / 配置项 / 失败恢复 / 发布回滚
- 消除 `E:\图rag` 硬编码

### B. 数据契约、严格回收与缓存身份（P0）

**B1 定义统一抽取输入输出契约。**

- 任务契约 `schema_version: ark-rag-task-2`
- 产物契约 `schema_version: ark-rag-extraction-2`
- 字段含：`run_id`、`chunk_id`、`content_key`、`scene_id`、`source_fingerprint`、`prompt_sha256`、`model`、`generation`
- 实现 `validate_task()` 与 `validate_extraction()`
- 机器可读错误码 + 字段路径

契约**必须拒收**：缺 `entities`、`entities=""`、`entities=False`、非对象记录、缺 `chunk_id`、`content_key` 不匹配、重复 ID、非法置信度、证据无法定位、未知版本。

**B2 修复 Part 回收。**

- 重复相同行可去重并记录
- 内容不同的重复行必须拒绝
- 不允许"最后一行覆盖前面"
- 缺失任何预期 Chunk 则 Part 不完整
- 坏记录进隔离报告
- 硬错误退出码非零
- 汇总区分：收到 / 合法 / 去重 / 缺失 / 冲突 / 隔离 / 正式入库

**B3 统一 L3 试跑与正式回收入口。** 门分**硬完整性门**与**语义质量门**；状态查询不能仅凭"缓存文件存在"。

**B4 缓存键身份化。** 键至少含：`content_key`、`chunk_id` 或上下文身份、`schema_version`、`model`、`prompt_sha256`、`generation` 参数、词表 / 规范化版本。

- Prompt 哈希由**实际内容**计算，版本字符串只用于展示
- 不再依赖人工 bump `PROMPT_VERSION`
- 非法缓存移入隔离区记 `invalid_cache`，不静默删除

### C. 事务式构建、原子发布与故障恢复（P0）

**C1 Run / Generation 目录模型。**

```
runs/<run_id>/            run.json, events.jsonl, errors/
generations/<generation_id>/   manifest.json, graph/, index/, reports/
current.json
```

`manifest.json` 至少记录：`generation_id`、`source_fingerprint`、`chunk_manifest_hash`、`extraction_manifest_hash`、各版本号、`git_revision`、`graph_artifact_id`、`index_artifact_id`、`gate_report_hash`、`created_at`、`status`、`parent_generation`。

**C1b 状态机：** `PREPARING → BUILDING → VALIDATING → READY → PUBLISHED`；任何状态可 → `FAILED`；**只有 READY 可发布**。

**C1c 发布顺序固定为：**

```
读取固定输入快照 → 构建候选 → 计算门禁 → 回读候选产物
  → 校验身份与内容摘要 → 全部硬门通过 → 原子发布
```

**C2 L6 回读不能仅比数量。** 至少验证：Generation 与输入指纹、文档数 / 词条数、Slot→Chunk 映射摘要、Posting 内容摘要、固定探针排序、元数据一致、产物 SHA-256。

**C2b 故障注入**需覆盖：构建到一半 / 门禁 FAIL / 回读损坏 / 写 Manifest 中断 / 发布前终止 / 发布后写报告失败。要求 `current` 始终指向旧有效 Generation。

**C3 修临时文件与共享状态竞态。** 同目录 UUID 临时文件 + flush/fsync + `os.replace` + 目录 fsync + 单飞或锁；`gates.json` 不再无锁读改写；熔断状态带 Owner / 时间 / Generation；运行日志追加写。

**C4 返修改为新 Generation 替换。** 禁止：任务生成前删旧缓存、未验收就覆盖、返修一部分就标整体健康、用当前 chunks 身份解释旧任务。

**C5 重试限流熔断按错误类型分类。**

| 错误类型 | 处置 |
|---|---|
| 401 / 403 / 无效模型 | 快速失败 |
| 429 | `Retry-After` + 指数退避 + 抖动 |
| 5xx | 有界重试，临时熔断 |
| 网络故障 | 有界重试，不立即永久熔断 |
| 内容契约错误 | 隔离记录，不当全局故障 |
| 系统资源错误 | 停止发布，显式报警 |

`extract.py` 不一次性提交全部 Future，用**有界待处理队列**；熔断后停止补充新任务；记录**真实并发峰值**。

### D. 解析、分块与指纹完整性（P0/P1）

**D1 修 node_id 去重导致内容丢失。**

- L1 校验缺失 / 非法 / 重复并明确报错，不用后值覆盖
- L2 用 `(scene_id, ordinal)` 轮次实例身份处理 overlap，`node_id` 保留为源元数据
- 覆盖门升级为**轮次实例 / 原始顺序 / 正文内容**三重覆盖
- 测试两类：`[1,1,2]` 不同正文、完全重复

**D2 修尾块超长。**

- 短尾块不无条件并入前块
- 合并前预测总长度
- 超硬上限时保留短尾块并显式标记原因，或在相邻块间重新平衡
- 必测输入：`[899,120,899,119,1100,1100]`

**D3 建立完整分层指纹。**

`source_fp` / `scene_fp` / `chunk_fp` / `extraction_fp` / `graph_build_fp` / `graph_id` / `index_build_fp` / `index_id`

必须覆盖：`doc_id` / `title` / `chapter` / `category` / `node_id` / `prop`、正文与说话人、分块参数与 T1 范围、实际 raw 抽取结果、归一化词表、索引字段与分词版本。

图身份要求：`graph_build_fp` 表配置、`graph_artifact_id` 表最终内容、索引绑定 `graph_artifact_id`、Manifest 绑定二者。

**D4 修 L0 来源合同。**

- ZIP 存在则验证 ZIP 与释放内容
- 缺失则验证解压树确定性文件清单
- 无法验证标 `source_unverified`，禁止冒充 Release
- 下游默认拒绝未验证来源

### E. 证据图与实体身份模型升级（P1）

**E1 从聚合边升级为逐观察 Assertion。**

```
assertion_id, subject_id, predicate, object_id, chunk_id, scene_id, doc_id,
evidence_text, evidence_start, evidence_end, confidence, raw_relation,
extraction_version, qualifiers
```

聚合边只存派生统计：`assertion_ids`、`support_count`、`independent_scene_count`、`confidence_summary`、`conflict_count`。

证据优先存字符偏移；只返回文本时在对应 Chunk 内精确查找；多处出现保存全部候选；找不到即硬门失败；不允许用其他 Chunk 证据替代。

迁移策略：旧图只能标 Legacy 投影；新 Assertion 图尽量从现有 raw 缓存重建；raw 不全处必须重抽，**不伪造历史证据**。

**E2 保留关系方向。**

- 邻接项加 `direction`（out / in）、`edge_id`、`assertion_id`、`predicate`、场景时间限定
- 出边入边分别处理
- 无向扩展必须显式声明
- 关系名不能反向原样使用
- 保存路径与每跳证据

**E3 实体身份与别名消歧三步。**

1. **先防明显误并**：类型兼容检查；集体与成员禁止直接合并；人名与职务/组织泛称分离；歧义别名不自动合并；合并前检查整个连通分量；冲突时拒绝 Union。
2. **再修类型归一化顺序**：规范值精确匹配 → 精确同义词 → 受限模糊匹配 → 未知类型。
3. **最后稳定实体 ID**：`Mention → EntityCandidate → CanonicalEntity`，内部用 UUID / 稳定哈希，名称只是展示字段；保存别名来源场景范围与审核状态；支持拆分合并历史。

优先审核：8 例泛称锚点吸收专名、46 例角色式别名、"人事干员 <- 乔"等风险、同组类型冲突、冲突关系集合。

风险应对：自动消歧准确率不足时保留候选不自动合并，**宁可拆多不可错并**，高频实体优先人工审核。

**E4 预留时间 / 冲突 / 叙事限定字段。**

`story_time`、`valid_time`、`scene_order`、`certainty`、`modality`、`is_flashback`、`is_hearsay`、`is_hypothetical`

互斥关系不直接删除；无法确定时间标为冲突 / 不同观察。

### F. 测试体系与真实质量评测集（P1）

**F1 修复测试运行方式。**

- 手写 `FAILS` 列表改为真正断言或接入 Pytest
- 临时目录隔离写入
- Mock LLM 网络
- 恢复被修改的全局配置
- CI 固定命令检查退出码

**F2 必补回归测试清单。**

- *回收与缓存*：缺字段错类型 / 重复行冲突 / 旧 Content Key / 错模型错 Chunk / 缓存损坏 / 同键并发 / 重跑幂等
- *构建与发布*：L4/L5/L6 FAIL 不发布 / 回读损坏不发布 / 中途失败保留旧版 / 回滚到父 Generation / 指纹覆盖全部输入
- *调度与网络*：429 / 超时 / 断网 / 401-403 / 5xx / 熔断恢复 / 有界队列 / 返修中断
- *数据语义*：重复 node_id / 超长尾块 / 跨标点分词 / 类型同义词优先级 / 实体类型冲突 / 有向图边 / 每观察证据定位

**F3 Gold Set 第一版 300–500 题。** 八类占比：

| 类别 | 占比 |
|---|---|
| 单块事实 | 20% |
| 实体别名与消歧 | 15% |
| 两实体直接关系 | 15% |
| 多跳关系 | 15% |
| 跨块跨章节综合 | 10% |
| 时间冲突 / 剧情阶段 | 10% |
| 否定与不可回答 | 10% |
| 易混淆对抗 | 5% |

每条至少含：`question_id`、`question`、`answer`、`acceptable_answers`、`gold_chunk_ids`、`gold_assertion_ids`、`required_citations`、`difficulty`、`category`、`should_abstain`。

标注质量控制：双人复核；不接受"答案对但引用错"；训练调参集与最终测试集分离。

**F4 质量门禁指标初始建议。**

| 指标 | 阈值 |
|---|---|
| 金标准证据 Recall@20 | ≥ 90% |
| 金标准证据 Recall@5 | ≥ 75% |
| MRR | 建立基线后提升，不设虚假绝对值 |
| 引用定位正确率 | ≥ 98% |
| 关键事实引用覆盖率 | ≥ 95% |
| 无依据问题拒答正确率 | ≥ 90% |
| 方向性问题正确率 | ≥ 95% |
| 硬完整性错误 | 0 |
| 失败产物发布 | 0 |

Recall 不能靠放宽"相关"定义达标。

### G. 最小可用检索问答闭环（P1）

**G1 先实现无向量版本。**

```
Query → 查询预处理 → 实体/别名候选 → BM25 检索 → 有界图扩展
  → 融合去重 → 上下文预算 → 生成 → 引用校验
```

**G2 查询理解输出结构。**

`original_query`、`normalized_query`、`entity_candidates`、`relation_hints`、`time_constraints`、`requires_graph`、`should_answer`

实体候选必须保留歧义；不因一个别名命中强制唯一实体；低置信度允许澄清或扩大召回；查询中否定和时间词不能丢弃。

**G3 修词法检索。**

- 标点和句子边界禁止跨接 bigram
- 保留英文 / 数字 / 专有编号
- 实体名别名作为完整 Token
- 标题 / 章节 / 正文分字段
- 预计算 `avg_doc_len`
- 查询返回稳定排序与解释字段

回归测试：`「我在罗德岛。凯尔希前往龙门。」` 不得生成"岛凯""希前"。

**G4 有界图扩展默认预算。**

1–2 跳基线；最大节点数 / 边数 / 每实体最大邻接数 / 最大 Assertion 数 / 最大上下文 Token / 超时预算。

扩展优先级：查询实体直接关系 → 与查询关系类型匹配的边 → 多来源支持的 Assertion → 与已召回 Chunk 共现的边 → 时间场景匹配的边。

禁止：全邻接无差别展开、为凑上下文增加弱关系、反向边沿用原关系语义、只返回图结论不返回原文证据。

**G5 上下文单元字段。**

`chunk_id`、`doc_id`/`title`/`chapter`、原文、证据区间、对应实体与 Assertion、召回来源与得分、图扩展路径。

去重压缩规则：同 Chunk 只留一次；相同证据多条 Assertion 合并；优先直接证据再图背景；截断不能截掉引用锚点；记录模型看到的精确上下文摘要。

**G6 回答输出契约。**

```
answer, citations[{claim, chunk_id, evidence, start, end}],
confidence, abstained, warnings, generation_id
```

硬规则：关键事实必须有引用；引用必须支持对应事实；检索不足时拒答；不允许用模型常识补齐剧情事实；冲突材料不得无条件合并；记录模型 Prompt 上下文 Generation 版本。

引用校验：自动检查 Chunk 与偏移存在；自动检查引用文本一致；引用失败不能当高置信答案返回。

**G7 CLI 入口。**

```
arkrag query "问题" --top-k 20 --graph-depth 1
arkrag evaluate --gold-set ...
arkrag inspect chunk|assertion
arkrag release current|rollback <generation_id>
```

验收链路"构建 → 查询 → 图扩展 → 回答 → 引用 → 拒答 → 回滚"不需手工改文件。

### H. 检索与存储性能优化（P1/P2）

**H1 性能基线。** 已测：127.2 MB JSON、仅 `json.loads` 约 23.88 秒、Python 峰值约 692.4 MiB。

进一步记录：完整加载耗时、冷热查询 P50/P95/P99、RSS 高水位、构建耗时、不同 Top-K 成本、图扩展节点边数、单问题 LLM Token 与成本。

**H2 低风险优化先行。** 预计算 `avg_doc_len`；避免每次搜索扫描所有文档长度；预建 `(category, chapter) -> chunks`；名称合并分桶前缀索引；QA 预建 `chunk_id -> index`；L3 有界队列；避免重复读取全部缓存；流式分片序列化。

**H3 SQLite FTS5 迁移。** 表：`chunks` / `chunk_texts`（FTS 虚拟表）/ `entities` / `aliases` / `assertions` / `relations` / `artifacts` / `runs`。

步骤：实现统一 `SearchBackend` 接口 → 保留旧 JSON Backend → 实现 FTS5 Backend → 同 Gold Set 双跑 → 比较召回/排序/延迟/内存 → 小流量灰度 → 达标切默认 → 保留旧 Generation 回滚。

初始性能目标：热查询检索阶段 P95 < 200 ms；冷启动不再依赖 20 秒级巨型 JSON 解析；内存显著低于 692 MiB 峰值。

### I. 混合检索、重排与图扩展增强（P2）

**I1 向量召回。**

- Embedding 模型和维度入 Manifest
- Chunk 内容哈希绑定向量
- 模型变化触发新向量 Generation
- 向量与 Chunk 一一对应
- 删除/修改 Chunk 不允许残留旧向量
- 保存构建时间、模型版本、归一化参数
- 限制批量和内存

融合先用 RRF：`BM25 candidates + vector candidates + entity/name exact matches → RRF → optional reranker`。不要一开始叠加复杂加权评分。

**I2 重排。** 输入特征：BM25/向量/RRF 分数、实体精确匹配、标题章节匹配、Assertion 支持数、图路径相关性、时间场景匹配、文本重复程度。

风险控制：重排器不能无证据创造新事实；保存重排前后排名；低置信不强行进上下文；单独评测重排收益和延迟。

### J. 服务化、安全、监控与持续交付（P2）

**J1 最小 API。**

```
POST /query
GET  /chunks/{id}
GET  /assertions/{id}
GET  /entities/{id}
GET  /health
GET  /ready
GET  /release/current
POST /admin/rollback
POST /admin/evaluate
```

Ready 条件：current Manifest 完整；图索引身份一致；必要数据可读；门禁报告为 READY/PUBLISHED；**不因进程启动就返回 Ready**。

**J2 安全。**

- 剧情文本视为不可信数据
- 防语料指令覆盖系统提示
- 限制 Query / Part / 响应 / 单记录大小
- 限制 Top-K / 图深度 / 上下文 Token
- 设置请求超时和并发限制
- 密钥只从安全配置读取，不写日志和 Manifest
- 鉴权限流；管理接口单独授权
- 错误响应不泄露本地路径 / 密钥 / 堆栈
- 审计回滚发布操作

**J3 可观测性。**

每次构建记录：`run_id`、`generation_id`、`git_revision`、输入输出指纹、配置模型摘要、开始结束时间、处理成功/失败/隔离数量、重试限流、实际并发峰值、门禁产物摘要、错误报告路径。

每次查询记录：`query_id`、`generation_id`、检索候选数、各通道得分、图扩展节点边数、最终上下文摘要、模型 Prompt 版本、延迟、Token 成本、引用校验结果、拒答原因。

默认不记录敏感完整内容。

### K. 全量重建、灰度验收与旧版本退役（P0 收尾）

**K1 全量重建前检查。** 新 Schema 冻结、契约测试通过、故障注入通过、Gold Set 固定、旧产物已备份、磁盘足够、模型预算明确、可从中断恢复、回滚命令已实测。

**K2 全量构建顺序。**

```
验证 Source → 新 Scene Generation → 新 Chunk Generation → 新抽取 Generation
  → 新 Assertion Graph → 新索引 → 全量硬门 → Gold Set 评测 → 性能评测
  → 候选 READY → 灰度查询 → 正式发布
```

全量硬门：无轮次静默丢失、无重复 Chunk 身份、无非法缓存、所有 Assertion 证据可定位、无未解释产物身份变化、无失败 Generation 被发布、索引与图绑定同一 Manifest、现有 8 例和 46 例风险有处理结论。

**K3 新旧对比指标与数量异常处理。**

- 实体突然减少 → 优先怀疑过度合并
- 实体增加 → 检查是否停止误并或消歧退化
- 边数减少 → 区分去重 / 证据校验拒收 / 漏抽
- 索引体积变化 → 结合分词与存储格式解释
- 超阈值必须人工审批

**K4 回滚与退役。**

- 保留至少上一代完整有效 Generation
- 回滚只切换指针，不重建历史
- 回滚后执行固定探针和 Gold Set 冒烟
- 新版稳定运行约定观察期后才归档旧版
- Legacy JSON 不得静默与新模型混用
- 删除旧产物前验证备份和 Manifest

---

## 3. 风险与预案（17 条）

| # | 风险 | 预案 |
|---|---|---|
| 1 | 严格校验导致大量旧缓存失效 | 先影子校验分类，只豁免可证明安全的 Legacy 数据，坏数据重抽 |
| 2 | 新图规模爆炸 | 原文与断言分层、聚合投影、分片存储、不丢弃原始证据 |
| 3 | 实体消歧准确率不足 | 禁止低置信自动合并、高频实体人工审核、保留拆分历史 |
| 4 | 新检索效果不如旧版 | 双 Backend 对比回滚，按问题类型定位，不全局调一个权重 |
| 5 | 向量升级成本过高 | 先缓存和 FTS5，向量仅用于 BM25 弱项，收益不足不上线 |
| 6 | 全量重抽失败或中断 | 新 Generation 续跑、旧版继续服务、分类重试、禁止先删缓存 |
| 7 | 门禁越来越松 | 硬错误零容忍；语义门必须有阈值 Owner 和处置期限 |
| 8 | 评测集过拟合 | 调参集与隐藏测试集分离，定期加入新错误样例 |
| 9 | 多阶段并行接口漂移 | 契约版本化、契约测试、Manifest 作为唯一接口事实源 |
| 10 | 发布时间不可控 | P0 未清零不进入向量和服务化，每阶段独立交付 |

---

## 4. Definition of Done（四大块）

**数据正确性**
所有抽取入库经过统一严格校验；重复 / 缺失 / 漂移 / 畸形记录不静默入库；每条关系观察有对应 Chunk 和证据定位；实体合并有类型与歧义保护；图和索引身份绑定真实输入与产物内容。

**可靠性**
任何门禁失败不覆盖正式产物；返修不先删旧数据；并发写入不产生半文件或丢失更新；429 与网络错误与永久错误区别处理；可从失败点恢复并回滚到上一代。

**检索与回答**
存在真实查询入口；实体消歧 / 检索 / 图扩展 / 重排 / 上下文组装形成闭环；图扩展保留方向；关键事实带准确引用；无依据问题可拒答；Gold Set 指标达阈值。

**性能与运维**
不再依赖每次启动解析 127 MB 巨型 JSON；查询和构建性能有持续记录；Run/Generation/Manifest 和错误报告完整；依赖命令与部署方式可复现；CI 能正确识别失败；源码 / 测试 / 文档有可回滚版本基线。

---

## 5. 推荐立即执行的前三项

1. **A** 冻结当前基线、建立 Git 与可复现环境。
2. **B / C** 严格回收校验、完整缓存身份、门禁后原子发布。
3. **D / E** 完整指纹、逐观察证据图和实体误并修复。

> 这三项完成前**不建议**投入向量数据库、前端或复杂在线服务。
