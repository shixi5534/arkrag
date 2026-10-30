# ArkRAG

离线剧情知识加工 + 检索问答流水线。把明日方舟剧情语料加工成**可追溯、可回滚、可评测**的知识图与词法索引。

当前状态：L0–L6 流水线已升级到第一版（数据契约 / 缓存身份 / 事务式发布），检索问答闭环尚未实现。逐项进度与验收证据见 `docs/升级台账.md`。

---

## 1. 环境准备

**Python**：3.11 或更高（实际开发与验证环境为 3.14.3）。`py -3 --version` 确认。

```powershell
py -3 --version        # 应 >= 3.11
```

**运行时依赖：零第三方库。** `src/`、`scripts/` 只用标准库，这样离线机器上不需要联网装包就能跑全流程。

**开发依赖（只有跑测试需要）**：

```powershell
py -3 -m pip install -r requirements-dev.txt
```

锁定版本的清单在 `requirements.lock.txt`（仅记录实际验证过的版本，便于复现）。

---

## 2. 目录与数据

| 路径 | 内容 | 入库 |
|---|---|---|
| `corpus/zh_CN/` | 只读原始语料（游戏数据投影，约 239 MB） | 否 |
| `zh_CN.zip` | 语料的发布包（35,904,420 B），用于来源校验 | 否 |
| `data/arkrag/` | 生成产物根目录 | 部分 |
| `data/arkrag/cache/` | L3 抽取缓存（13,801 个文件） | 否 |
| `data/arkrag/generations/<gid>/` | **一代完整产物**：`graph/`、`index/`、`reports/`、`manifest.json` | 图与索引否，清单是 |
| `data/arkrag/runs/<run_id>/` | 一次构建的执行记录：`run.json`、`events.jsonl`、`errors/` | 是 |
| `data/arkrag/current.json` | **唯一的"线上是哪一代"指针** | 是 |
| `data/releases/legacy-baseline/` | A 阶段冻结的旧基线快照 | 是 |
| `docs/` | 决策台账、抽取台账、升级计划与执行台账 | 是 |

原始语料只读。所有生成物写 `data/`，任何阶段都不改 `corpus/`。

---

## 3. 流水线命令

按顺序跑。每步都是独立脚本，退出码非零即失败（`0` 通过，`1` 门禁 FAIL，`2` 输入不完整/拒绝执行）。

```powershell
# L0 语料来源校验与清单
py -3 scripts\run_l0.py

# L1 解析：语料 -> scenes.jsonl（场景/轮次）
py -3 scripts\run_l1.py

# L2 分块：scenes -> chunks.jsonl + shards.json
py -3 scripts\run_l2.py

# L3 抽取：子代理派发 / 回收
py -3 scripts\run_l3_agent.py --plan        # 生成派发计划
py -3 scripts\run_l3_agent.py --stratify    # 分层下发
py -3 scripts\pack_pending.py               # 打包待抽块
py -3 scripts\run_l3_agent.py --status      # 查看进度（带 chunk_id 身份核对）
py -3 scripts\run_l3_agent.py --ingest <label>   # 严格回收（硬错误/缺块退出码非零）

# L4+L5 归一化与建图
py -3 scripts\run_l5.py

# L6 建索引
py -3 scripts\run_l6.py
```

### 3.1 推荐：事务式构建与发布

上面 L4/L5/L6 分开跑是**调试用**的：它们直接写 `data/arkrag/graph.json` 与
`data/arkrag/index/lexical.json`（受 `legacy.guard_write` 保护，需显式设
`ARKRAG_ALLOW_LEGACY_WRITE=1` 才允许）。

**正式产物请走一条命令**：

```powershell
py -3 scripts\build_generation.py                 # 构建 -> 门禁 -> 回读 -> 原子发布
py -3 scripts\build_generation.py --no-publish    # 只构建到 READY，不切指针
```

它做的事（顺序不可调换）：固定输入快照 → 建候选到 `generations/<gid>/` → 算门禁 →
回读候选产物并核内容摘要与探针 → 全部硬门通过才 READY → 原子替换 `current.json`。

**任何一步失败，`current.json` 一个字节都不会被改动**，线上仍是上一代。

---

## 4. 发布、回滚与查看

```powershell
py -3 scripts\release_cli.py current            # 线上是哪一代 + 自检
py -3 scripts\release_cli.py list               # 所有世代与状态
py -3 scripts\release_cli.py show <gid>         # 一代的 Manifest 全文 + 自检
py -3 scripts\release_cli.py verify <gid>       # 只校验不改状态
py -3 scripts\release_cli.py publish <gid>      # 把 READY 的一代切成线上
py -3 scripts\release_cli.py rollback <gid>     # 指针切回历史版本
py -3 scripts\release_cli.py prune --keep 2     # 清理旧 FAILED 世代（默认演练，--apply 才删）
```

**回滚只切指针，不重建历史产物。** 重跑几小时抽取才能造回来的东西，不该在回滚时被重新生成一遍。

---

## 5. 测试

```powershell
py -3 -m pytest tests -q
```

当前 **215 项通过**。测试只写临时目录，不碰 `data/` 真实产物（`tests/test_build_generation.py::TestIsolation` 就是守这条的）。

另有 A 阶段手写套件（保留兼容）：

```powershell
py -3 tests\test_negative.py     # 10 项负例
py -3 tests\test_vocab.py        # 词表一致性
```

### 5.1 基线校验

```powershell
py -3 scripts\snapshot_baseline.py --verify
```

`0` = 基线 11 项产物全部一致；`1` = 有产物被改动或丢失；`2` = 快照不存在。

**这条命令应当在每次实验前后各跑一次。** 它抓到过一次真实事故：测试把假的 `corpus_fp` 写进了 `data/arkrag/corpus_meta.json`，产物字节变了但没人发现——`--verify` 报 `字节已变`。

---

## 6. 配置项

全部可用环境变量覆盖（见 `src/arkrag/config.py`）：

| 变量 | 默认 | 含义 |
|---|---|---|
| `ARKRAG_SOURCE_DIR` | `corpus/zh_CN` | 只读语料根 |
| `ARKRAG_DATA_DIR` | `data/arkrag` | 生成物根 |
| `ARKRAG_CHUNK_CHARS` | `900` | 分块目标字数 |
| `ARKRAG_CHUNK_MIN_CHARS` | `120` | 尾块最小字数 |
| `ARKRAG_CHUNK_OVERLAP_TURNS` | `1` | 相邻块重叠轮次数 |
| `ARKRAG_CONCURRENCY` | `4` | 单进程 LLM 并发 |
| `ARKRAG_RETRIES` | `3` | 重试次数 |
| `ARKRAG_LLM_MODEL` / `ARKRAG_LLM_BASE_URL` | 空 | 模型与网关 |
| `ARKRAG_CONF_THRESHOLD` | `0.5` | 关系置信度下限 |
| `ARKRAG_ALLOW_LEGACY_WRITE` | 未设 | 设为 `1` 才允许写冻结路径（仅调试用） |
| `ARKRAG_RUN_OWNER` / `ARKRAG_GENERATION_ID` | 空 | 熔断状态归属，便于追责 |

根目录用 `Path(__file__).resolve().parents[2]` 解析，**不依赖调用者的当前目录**——从任意 cwd 调用都得到同一套路径。

---

## 7. 失败恢复

| 症状 | 处理 |
|---|---|
| 门禁 FAIL 未发布 | 候选留在 `generations/<gid>/`（标 `FAILED`，`fail_reason` 写明原因）。修完重跑 `build_generation.py` 即可，旧版本仍在服务 |
| 构建中断 | 同上。没有半成品会被当成正式版本——`current.json` 未动 |
| 抽取被熔断 | 看 `data/arkrag/circuit_open.json` 的 `kind`：`permanent` 需人工处理鉴权/额度；`transient` 到期自动解除（默认 300 秒）。收据里 `skipped_circuit` 是"没跑"，下次补 |
| 缓存疑似污染 | `py -3 scripts\shadow_cache_check.py` 看四层分布（current/legacy/miss/invalid），invalid 会带原因 |
| 需要返修某批块 | `py -3 scripts\make_repair.py --list <file>`：**新旧两种缓存键都清**，只删新键会让 legacy 记录继续被复用，返修看起来没生效 |
| 产物身份对不上 | `py -3 scripts\release_cli.py verify <gid>` 逐项列出不符的字段 |

**永远不要先删旧产物再重建。** 返修走新 Generation，旧的那一代留着——它是唯一的回滚目标。

---

## 8. 数据契约

- `src/arkrag/contracts.py`：`ark-rag-task-2`（送入模型）与 `ark-rag-extraction-2`（模型返回）。21 个机器可读错误码，每个带字段路径。
- `src/arkrag/cache.py`：缓存键含 `content_key | chunk_id | scene_id | model | prompt_sha256 | schema_version | generation`。`prompt_sha256` 由提示词实际内容算出，改提示词不需要人工改版本号。
- `src/arkrag/release.py`：Generation Manifest 记录 `source_fingerprint`、`chunk_manifest_hash`、`extraction_manifest_hash`、`graph_artifact_id`、`index_artifact_id`、`gate_report_hash`、`index_content_digest`、`probe_digest`。

`Manifest 是唯一接口事实源`：图与索引各自带配置指纹，Manifest 记内容摘要。配置变了要重建，内容被换掉了要能发现。

---

## 9. 文档

| 文件 | 内容 |
|---|---|
| `docs/升级计划_GraphRAG架构.md` | A–K 十一阶段计划原文（改计划必须改这里） |
| `docs/升级台账.md` | 实际做了什么、验收证据、与计划的偏差 |
| `docs/baseline-metrics.md` | A 阶段冻结的现场指标 |
| `docs/决策台账.md` | D-01 ～ D-24：历史决策与踩过的坑 |
| `docs/抽取台账.md` | L3 抽取过程记录 |
| `docs/子代理抽取作业指引.md` | 派发作业给子代理的操作说明 |
