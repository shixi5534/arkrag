# legacy-baseline 基线指标

> 本文件由 `scripts/snapshot_baseline.py` 生成，**不要手改**。
> 冻结时间：`2026-10-30T11:15:13Z`
> 校验命令：`py -3 scripts/snapshot_baseline.py --verify`（不一致时退出码 1）

## 版本常量

| 常量 | 值 |
|---|---|
| schema_version | `ark-rag-1` |
| prompt_version | `v2` |
| normalizer_version | `n2` |
| chunk 目标/下限/重叠 | 900 / 120 / 1 |
| T1 范围 | obt/main、activities、obt/memory、obt/roguelike、obt/rogue |
| 置信度阈值 | 0.5 |
| Python | 3.14.3 |

## 语料

- corpus_fp：`None`
- story 文件数：2292

## L1 场景

- 场景数：**2,291**
- 轮次数：**422,975**

## L2 分块

- 全量块数：**13,786**
- T1 块数：**13,781**

| category | 块数 |
|---|---|
| activities | 7,948 |
| obt/guide | 5 |
| obt/main | 2,272 |
| obt/memory | 3,138 |
| obt/rogue | 373 |
| obt/roguelike | 50 |

## L3 抽取缓存

- 缓存文件数：**13,801**
- 总字节：30,688,245
- 整体摘要：`a244f0362aa423dc78f65db11a2eab0c11b5357114354d84b33eea73f2b797b7`

## L5 图

| 指标 | 值 |
|---|---|
| 实体数 | **6,483** |
| 边数 | **28,234** |
| 孤立实体 | 852 |
| 平均度数 | 8.71 |
| 别名条目 | 1,605 |
| graph_fp | `f90a78f6df41f564` |
| corpus_fp | `188e3218448d6bb4` |
| ev / n | `ev2` / `n2` |

## L6 索引

| 指标 | 值 |
|---|---|
| 文件字节 | 127,244,404（127.2 MB） |
| 文档数 | 13,781 |
| 词条数 | 694,304 |
| 倒排项数 | 10,525,428 |
| 索引图指纹 | `f90a78f6df41f564` |
| 向量通道 | `not_built_no_embedding_backend` |

## 门禁

- 总行数：**75**
- 分布：L0=10、L1=9、L2=7、L3=19、L4=13、L5=9、L6=8
- 判定：PASS=75

**全部 75 项 PASS。** 但依计划原则 8，全 PASS 不作为免检依据；
本基线只冻结「当时的面貌」，不表示已知盲区都被覆盖。

## 产物清单（SHA-256）

| path | kind | bytes | sha256 |
|---|---|---|---|
| `data/arkrag/graph.json` | graph | 17,115,864 | `ee7b1e05a31d61b0ac5c895d42e7b56bb4960b17665a2cc78ce91348cf52cb48` |
| `data/arkrag/index/lexical.json` | index | 127,244,404 | `39911f26426bbe3b79b4582a5df51e5cdb103a862b88bdf61ef783091e9486c2` |
| `data/arkrag/gates.json` | gates | 18,499 | `393ef1dca0466eaacd28a03d507fc44f7f71aa43e0b8f48253470049ff8ddc7e` |
| `data/arkrag/chunks.jsonl` | chunks | 40,016,871 | `56841f32f1736eae912c4799cad9f693cd298d1c940596d07205eab183bf213b` |
| `data/arkrag/scenes.jsonl` | scenes | 71,305,439 | `689b9b9343596bc0b5901f019f19c9d66354adf08f6db8a49342ae25d877cec5` |
| `data/arkrag/shards.json` | shards | 139,498 | `a8e516454f2cea96a0181e32da25bf32f261ba29a248f04a580e0b0461e7e532` |
| `data/arkrag/l0_manifest.json` | l0_manifest | 159,722 | `a4ca9d8d1232148fbe9f09eb96f4765e33da1c34f89f5d0014df23197b6d0ce2` |
| `data/arkrag/l1_stats.json` | l1_stats | 2,950 | `baf1b7e3abb9bdf0380eac27b3b7012c89e11deb41b14fb0a570aa8e05a7295c` |
| `data/arkrag/l1_dropped.json` | l1_dropped | 86 | `34fccd6e5cde55f45473b72ec499fd4abd7eadb230eb464471d968ab9fbef2a2` |
| `data/arkrag/corpus_meta.json` | corpus_meta | 153 | `97d9a477d19b2e0d20363ec2e37999103bee13d9a6b11c331ddee9549a7274c4` |
| `data/arkrag/cache` | extraction_cache_dir | 30,688,245 | `a244f0362aa423dc78f65db11a2eab0c11b5357114354d84b33eea73f2b797b7` |

## 已知能力边界（继承自现有台账，不在本阶段修复）

- L6 只有词法通道，`vector_channels=not_built_no_embedding_backend`。**对外不能说具备语义检索。**
- 图是聚合边模型，一条边只有一条 `evidence`，无法回答「这条边在第几段原文」。
- 图邻接双向且反向边沿用同一关系名，方向语义会丢。
- 无检索/问答/评测入口——系统目前只能建，不能答。
