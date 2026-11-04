# K 阶段实测

> 本文件记录 K 阶段的实测数据。当前覆盖 **K1、K2**；K3/K4 完成后追加。
> 与前几个阶段一样，只写实际跑出来的数，不写"预计"。

## 0. 口径

1. **退出码语义写死在脚本里**：`scripts/k1_checks.py` 退出 **0** = 九项 blocking 全过；**1** = 有 blocking 未过；**2** = 硬错误（结论不可用）。
2. **九项检查不重做 `verify_generation` 的活儿**。它们检查的都是"构建过程本身看不到的东西"——产物自检已经在 `release.verify_generation` 里有一份定义，再写一份就是第二个事实源。
3. **每项必须能失败**。`tests/test_k1_checks.py` 为每一项构造了一个本该拦住它的场景，并断言它确实返回 `ok=False`。一个永远返回 `True` 的检查函数比没有更糟——它会在台账里多出九行 PASS，而没人知道那几行是怎么来的。
4. **只跑一部分不算通过**。`--only` 跑 3 项时输出会多一行"只跑了 3/9 项"，结论也改成"不足以进入 K2"。首轮实测踩到过：`--only schema_frozen,gold_set_fixed,disk_space` 打印了"九项 blocking 检查全部通过，可以进入 K2"。
5. **无 LLM 端点时 `budget_required=False` 是可验证事实，不是跳过**。检查里写的是"无 LLM 端点（model/base_url 皆空）→ 预算为 0"，一旦接上端点而没有 `ARKRAG_K1_LLM_BUDGET`，这项立刻转红。

## 1. 九项检查与判据

| # | id | 判据 | 为什么这项在这里 |
|---|---|---|---|
| 1 | `schema_frozen` | 代码侧 7 个版本号 vs 最新 ark-rag-3 世代的 Manifest/图内记录 | 冻结的含义是"两端写着同一个值"。只比 Manifest 不比代码，改代码后旧产物仍然自洽 |
| 2 | `contract_tests` | 子进程跑 `tests/test_contracts.py` + `tests/test_build_generation.py` | 契约变更必须先在这里红，而不是在 20 分钟后全量重建的最后一步红 |
| 3 | `fault_injection` | 子进程跑 `test_negative.py` / `test_legacy_guard.py` / `test_gate_numbering.py` / `test_gates_l0.py` / `test_gates_e.py` / `test_release.py` | 这些用例守的是"坏输入必须被拒" |
| 4 | `gold_set_fixed` | sha256 前 16 位 + 字节数 + 题数 + 八类配额 | 换题库必须改一行代码——改代码是有人签字的动作，改文件不是 |
| 5 | `old_artifacts_backed_up` | 复用 `snapshot_baseline.py --verify` + 可服务世代 ≥ 2 + 当前线上世代自身 0 问题 | 回滚需要至少一个完好目标。`previous_generation` 指向 `606ea9`，它有 2 项不过，所以"上一代"不能靠指针，只能靠"盘上有几个能用的" |
| 6 | `disk_space` | 剩余 ≥ 2 GiB | 242.2 MiB × 3 代（候选 + 上一代 + 回滚目标）× 3 倍余量 ≈ 2.1 GiB |
| 7 | `model_budget` | 无端点 → 0；有端点 → 要求 `ARKRAG_K1_LLM_BUDGET` 三段 | 挡"悄悄接上 LLM 开始烧钱" |
| 8 | `resumable` | 状态字面量在已知六态内 + **现场**拿盘上真实的非 READY 世代调 `release.publish` 必须被拒 + 当前指针世代 0 问题 | 不靠读代码判断"能不能恢复"，靠真调一次 |
| 9 | `rollback_tested` | 在**隔离指针**上真跑 `release.rollback`，验三件事 | 没人真跑过，"回滚已实测"这句话就是假的 |

## 2. 实跑结果（`py -3 scripts/k1_checks.py`，EXIT=0）

```
K1 全量重建前检查
  时间     : 2026-11-03T11:49:22Z
  修订     : 53f19f71ac64fad1aa5a4ace7c9c81feaa1ae634
  世代     : 9 个

  [PASS] 新 Schema 冻结
         7 项版本与 g-20261031T035231Z-4eafae 记录一致（其中 task/extraction 契约版本两端都不记，列为未验证）
  [PASS] 契约测试通过
         76 passed, 1 warning in 15.93s
  [PASS] 故障注入通过
         146 passed, 1 warning in 22.54s
  [PASS] Gold Set 固定
         sha256=76d66073ff87aab1 字节=284356 题数=320 八类配额全对
  [PASS] 旧产物已备份
         冻结基线 10 文件校验一致；可服务世代 6 个（含线上 g-20261030T154606Z-55dfd7）
  [PASS] 磁盘足够
         剩余 101.3 GiB / 需要 2.00 GiB（最大单代 242.2 MiB）
  [PASS] 模型预算明确
         无 LLM 端点（model/base_url 皆空）→ 预算为 0；抽取由 agent:qoder-subagent 离线产出
  [PASS] 可从中断恢复
         状态集合 FAILED/PUBLISHED/READY；非 READY 发布被拒（refused: IllegalTransition）；线上 g-20261030T154606Z-55dfd7 自检 0 问题
  [PASS] 回滚命令已实测
         隔离指针上 g-20261031T035231Z-4eafae → g-20261030T150101Z-e0aacc 成功，键集同构，回滚到自身被拒

结论：九项 blocking 检查全部通过，可以进入 K2。
      （通过不代表新世代更好，只代表没有发现阻断理由。）
```

**可服务世代 6 个**：`e0aacc`、`65a5c4`、`55dfd7`（线上）、`b22ca8`、`348206`、`4eafae`。不计入的：`606ea9` / `7b0fd3`（各 2 问题）、`1d1b08`（FAILED，7 问题）。

**`task_schema` / `extraction_schema` 单列 `unverified_in_artifacts`**，不当通过：这两项版本号产物里根本不记，比不了就是比不了。

## 3. 首轮实跑的 2 项红——都是我自己写错的判据

首轮 EXIT=1，两条失败信息**都指向了错误的方向**。

### 3.1 16 位展示指纹 vs 64 位摘要

```
[FAIL] 新 Schema 冻结   代码与产物版本不一致 1 项：
       prompt_sha256 代码=75835665be8edb9b
       产物=75835665be8edb9b147c3453827b14dbdf7fbde09e0648c4a7fbb2382c4846f1
```

`_repo_versions()` 里我用了 `prompt.prompt_fingerprint()`（**16 位**）而不是 `fingerprint.prompt_digest()`（**64 位**）。两者前 16 位完全相同，所以报出来的消息看起来像"产物的摘要是好的，只是多了 48 个字符"——**读的人第一反应是产物有问题**。

真相：Manifest 记的一直是 64 位那一份，代码侧取错了长度。

### 3.2 把"总条数"当成"坏条数"

```
[FAIL] 可从中断恢复   抽取缓存里有 13801 条非法记录
```

`cache.invalid_cache_report()` 返回 `{total, ok, invalid, by_reason, samples}`。我读了 `total`。真相是 `total=13801, ok=13801, invalid=0`——**一条坏记录都没有**。

这类错误最危险的地方在于**它看起来像个重大发现**：13801 条非法记录，任谁看到都会先停下来查缓存。它同时也是最容易被"顺手修好"的——只要把检查删掉，"重大发现"就消失了。

两条都各配了一条回归用例（`TestSchemaFrozenCanFail` / `TestResumableCanFail` 内），钉的是"取哪个字段"这件事本身。

## 4. 全量 pytest 抓出的真实缺陷：错误响应被连接重置吞掉

### 4.1 现象

`py -3 -m pytest tests -q` → `1 failed, 1246 passed, 1 skipped in 1261.33s(21 分钟)`：

```
FAILED tests/test_server.py::TestQueryEndpoint::test_oversized_body_is_refused_before_reading
ConnectionAbortedError: [WinError 10053] 你的主机中的软件中止了一个已建立的连接。
  http/client.py:297 _read_status → socket.py:725 recv_into
```

单跑这条通过。再跑**整个** `tests/test_server.py` 时，失败的换成：

```
FAILED tests/test_server.py::TestAdminFailClosed::test_admin_disabled_when_no_token_configured
```

同一个 `WinError 10053`。

### 4.2 根因

`_BodyReader.read()` 在 `Content-Length > max_body_bytes` 时**只抛 413，不读请求体**。接收缓冲区里剩着未读字节，`close()` 发的是 RST 而不是 FIN，客户端内核把已经到达的 413 一起丢掉。

单跑能过，是因为那时客户端早把几百字节发完了；全量跑时序不同就露出来。

**这不是 413 的个性问题。** 同一条路径上还有：`_require_admin()` 在 `_json_body()` 之前跑，所以没配口令时的 503、token 不对时的 401，都发生在"还没轮到解析请求体"的时刻——同样留下未读字节。`test_admin_disabled_when_no_token_configured` 就是这么红的。

### 4.3 为什么这条不能当 flaky 放过

"只在全量跑偶发失败、单跑必过"是 flaky 的典型签名，而 flaky 的标准处置是重跑或加 `-p no:randomly`。但这里重跑也会过——所以它会一直留在 CI 里，直到某天有人把它标成 skip。

线上同一个路径的效果是：**"你的请求太大了"被变成"服务挂了"**。调用方看到连接重置，会重试、会熔断、会报警，而正确的反应其实是改小 payload。

### 4.4 修法

两处，都写进代码注释了：

1. **`_BodyReader.read()` 超限时先排空**：读掉至多 `DRAIN_CAP = 1 MiB` 字节并丢弃（不进内存累积），再抛 413。超过就不读——不能被一个声明了 10 GB 的请求拖住，此时确实会重置连接，但那已经是明确的滥用流量。
2. **`_route_post()` 在任何拒绝分支之前读掉请求体**：把"读请求体"提到路由入口，`_json_body()` 优先用 `self._raw_body`。这样 401/503/404/429 都不会留下未读字节，不用逐个拒绝点记得做这件事。

配套：`_fail()` 在"排空被跳过"时加 `Connection: close` 并置 `close_connection`；`_error_payload()` 剔掉一切 `_` 开头的键——`_drain` 是给 `_fail()` 看的处置指令，不该出现在对外响应里（调用方看到一个 `_drain` 字段只会困惑）。

### 4.5 验证

新增 3 条用例（`tests/test_server.py`，58 → 61）：

| 用例 | 守的是 |
|---|---|
| `test_oversized_body_still_delivers_the_413` | 连续发三次超限请求，每次都必须**真的读到** 413（不是异常） |
| `test_413_does_not_leak_internal_drain_directive` | 对外错误体里不许出现 `_` 开头的内部字段 |
| `test_unreasonably_large_body_closes_the_connection` | 声明 5 MiB 但一字不发 → 仍回 413，且带 `Connection: close` |

`Client.post_raw()` 是新加的能力：走 `http.client` 手工拼头，让声明长度与实际字节无关——`urllib` 会按实际字节改 `Content-Length`，测不到"只发头不发体"这个场景。

改后：
- `pytest tests/test_server.py -q` → **61 passed, 65.16s**
- `pytest tests -q` → **1250 passed, 1 skipped in 583.86s**（9 分 43 秒，比修复前的 21 分钟快了 11 分钟——修复前那 1261 秒里相当一部分是超时重试）
- `snapshot_baseline.py --verify` → PASS，产物 11 项一致

## 5. 新增文件

| 文件 | 规模 | 作用 |
|---|---|---|
| `scripts/k1_checks.py` | 943 行 / 40,590 B | K1 九项检查，退出码 0/1/2 |
| `tests/test_k1_checks.py` | 657 行 / 26,744 B | 36 用例，证明九项每项都能失败 |

`tests/test_suite_hygiene.py` 的 `N`：1208 → 1248（+36 用例 +4 parametrize）→ **1251**（再 +3 个 413 用例）。

## 6. 复现命令

```bash
py -3 scripts/k1_checks.py                  # 九项，退出码 0/1/2
py -3 scripts/k1_checks.py --json           # 同上，机器可读
py -3 scripts/k1_checks.py --list           # 列出九项 id
py -3 scripts/k1_checks.py --only a,b,c     # 只跑指定项（结论会说明覆盖不全）
py -3 -m pytest tests/test_k1_checks.py -q  # 36 用例
py -3 -m pytest tests/test_server.py -q     # 61 用例
py -3 scripts/snapshot_baseline.py --verify # 11 项产物一致性
```

## 7. 遗留

1. **K1 通过不代表新世代更好**。脚本自己在结论里加了这一句，避免被当成验收结论引用。
2. **`previous_generation` 指向的 `606ea9` 有 2 项不过**（缺 `graph_artifact_id` + `source_status=''` 未验证来源），所以线上 `55dfd7` 的"上一代"不能作为回滚目标。当前靠"盘上有 6 个可服务世代"绕过，但**这是 K4「保留至少上一代完整有效 Generation」的现成缺口**，需要在 K4 收口。
3. **`check_resumable` 只验到"非 READY 不能发布"这一层**。真正的"从中断恢复"（构建中途进程被杀、下次从断点续跑）还没有对应的实操演练——那需要 `build_generation.py` 有断点续跑的路径，当前它是一次性事务。
4. **`DRAIN_CAP = 1 MiB` 是经验值**，没有实测依据。选它是因为：正常查询的 payload 是几百字节，1 MiB 足够覆盖任何正常客户端；同时它把最坏情况的排空成本钉在 1 MiB。若将来允许上传类接口，这个值得重新评估。
5. **`_route_post()` 现在无条件先读请求体**，意味着一个声明了 1 MiB 的合法但无用的 POST 会先被读完才被 404 掉。可以加"只对已知路由读"，但那样就得让每个拒绝点自己记着读——正是这次要消除的形态。当前取"永远先读"。

## K2 实测：全量重建

### 0. 口径

1. **构建环境**：`$env:ARKRAG_ARTIFACT_DIR="data/arkrag/staging/k2"`。不用
   `ARKRAG_ALLOW_LEGACY_WRITE=1`——那是「我知道我在覆盖冻结产物」的放行口，而本次
   要做的恰恰是「不覆盖任何冻结产物，另建一份」。
2. **所有数字都是本次实跑读数**，没有引用、估算或推算值。
3. **`scene_fp` / `chunk_fp` / `chunk_manifest_hash` 这类身份值的变化，必须给出解释**
   才算通过；解释不了就是未解释的产物身份漂移，按 K2 硬门第 5 条不许发布。
4. **召回指标的差异必须先过显著性检验**才能说"更好/更差"。本节只报观测值。
5. **失败候选」留在盘上**，不删——它们是「这次真的失败了」的证据。

### 1. L1 场景层（EXIT=0，十一门全 PASS）

| 项 | 值 |
|---|---|
| `corpus_fp` | `188e3218448d6bb4`（**与已发布世代一致**） |
| `scene_fp` | `184b619b8c620a2a`（`4eafae` 记的是 `20b49bb036087012`） |
| 来源状态 | `verified_zip` |
| 场景 / 轮次 | 2,291 / 422,975 |

**`scene_fp` 变化已查明**：逐场景比对 2,291 个 scene_id，only old 0 / only new 0，
但 2,291 个**全部 changed**。唯一差异是 D 阶段给 turn 加了 `ordinal` 字段
（旧 turn keys `['kind','masked','node_id','prop','speaker','text']`，
新 = 那六个 + `ordinal`，`ordinal None => 0`）。`corpus_fp` 未变说明**正文没变**。

这正是分层指纹的意义：`corpus_fp` 变了才是"上游数据变了"，`scene_fp` 变了只说明
"这一层的记录形状变了"。

### 2. L2 分块层（EXIT=0，八门全 PASS）

| 项 | 值 |
|---|---|
| 全量块 | **13,803** |
| T1 块 | **13,798**（线上世代是 13,781） |
| `chunk_fp` | `d29d74aa792176ff` |
| T1 分片 | 69（S00–S68） |

八门实测：L2.1a 原文轮次覆盖 422,975/422,975 漏 **0**；L2.1b 渲染冗余率 1.219（仅申报）；
L2.1c 顺序/正文异常 **0**；L2.2 无原因短块 **0**；L2.3 最大块 1,000（≤1,800）；
L2.4 跨场景块 **0**；L2.5 内容哈希重复 **0**；L2.6 分片均衡 min 198 / max 229。

**13,798 正是 D 阶段那次一直没上线的分块**（D 阶段因 34 块缓存缺口而停在 13,781）。

### 3. 抽取缺口：34 块，无一块可复用

`run_l3_agent.py --status`（staging）：

```
合计 13764/13803 块已入库
来源分层 current 0 legacy 13764 adopted 0 miss 39 invalid 0
```

T1 范围 13,798 里缺 **34** 块（另 5 块在 T1 范围外，`obt/guide`）。

**形态**：17 对「一个正文变了的主块 + 一个新增的短尾块」。主块 `chunk_id` 与旧世代
相同但 `content_key` 变了；尾块是新 id。

| 例 | 新 ck | 旧 ck | 字数 |
|---|---|---|---|
| `activities/a001/level_a001_01_end#000` | `b18a5342ff82021344cb505b` | `469728ddd0774c367533` | 887 vs 992 |
| `activities/a001/level_a001_01_end#001`（新增） | `47020d5c05669e4813e1cc13` | — | 121 |

**没有一块能 `adopt_legacy`**：缓存键绑 `content_key`，正文变了键就变了
（`来源分层 adopted 0` 就是这个意思）。

抽取自产出 34 条后 → `13798/13803`（T1 全齐），`miss 5` 是范围外那 5 块，与线上同构。
`--ingest` 回执：`读到 34 行 合法 34 去重 0 冲突 0 隔离 0 缺块 0 命中 0 新入库 34
硬错误 0 / 原始实体 162 原始关系 129`。

抽取过程中被抓到的唯一一条真错：`activities/act16mini/level_act16mini_st02#009`
里我写 `莱伊：矿场的人都叫我莱伊。`，而该块说话人标签实际是 **`劫车人`**
（`劫车人：矿场的人都叫我莱伊。`），直到最后两行才改成 `莱伊`。本地校验器按
"evidence 必须是块正文逐字子串"拦下了它——若直接写进 outbox 会被 `EVIDENCE_NOT_FOUND` 拒收。

### 4. 构建：先失败一次，原因在我的探针

首次 `build_generation.py --graph-model identity --no-publish` → 候选 `fa22ae`：

```
FAIL L6.3 检索正确性（稀疏词回查来源块） 实测[153/197]
  教能->activities/act52side/level_act52side_06_beg#003; 宗是->...; 起武->...; 间同->...
```

同代其余 37 门全 PASS；`4eafae` 与 `55dfd7` 的 L6.3 都是 199/199。

**根因**：`build_probes()` 调 `index.tokenize(f"{title} {chapter} {text}")` 时
**没传 `name_terms`**，而 `index.build()` 每块 `index.add(..., name_terms=table)` **传了**。
`tokenize_t2` 对专名区间内侧不连跨边界双字（`index.py` 的 `cuts` 逻辑），于是同一段
文本在两侧切出两套 token——探针能挑到「倒排里存在、但那个 bigram 实际落在别的块」的词。

定量（同随机种子 20261029，抽 200 块）：

| 传 `name_terms` | 抽样 199 条里指向错块 |
|---|---|
| 否 | **58** |
| 是 | **0** |

索引本身完全正确。这条例子的价值在于：**门禁报的是索引的错，实际是探针的错**。
若按"索引坏了"去查，会一路查到倒排结构、分词器、建库流程，最后大概会修一个没坏的东西。

### 5. 候选世代（`g-20261103T132152Z-06a741`）

| 项 | `4eafae`（旧） | `06a741`（新） |
|---|---|---|
| `chunk_count` | 13,781 | **13,798** |
| `entity_count` | 6,582 | **6,583** |
| `edge_count` | 28,625 | **28,636** |
| `assertion_count` | 62,910 | **62,924** |
| `coverage_status` | complete | complete |
| `chunk_manifest_hash` | `34e6971d86c72935…` | `e9cf0a92b5a9c369…` |
| `index_content_digest` | `c1fef3588e662271…` | `f8419d479ad3a2ee…` |
| `probe_digest` | `46e600c1f3a9ec9e…` | `b800a2a2b2e273b0…` |
| `graph_build_fp` | `1d276c22c613dc9f` | `1d276c22c613dc9f`（**相同**） |
| `index_build_fp` | `bb5c93f8f1a75d42` | `4f1fc2958f70e526` |
| 门禁行数 | 30（ark-rag-1） | **38 全 PASS** |
| `verify_generation` | 0 问题 | **0 问题** |
| 目录字节 | 254,005,061 | **302,419,455**（288.4 MiB，含自带 `chunks.jsonl` 68,935,400 B） |

`graph_build_fp` 两代完全相同——图构建的输入（抽取记录）对已抽的 13,764 块没变，
新增 34 块只带来 +1 实体 / +11 边。**这是「同样输入给同样产物」的证据。**

`index_build_fp` / `index_content_digest` / `probe_digest` 全都变了，解释见 §8。

### 6. Gold Set 评测（320 题）

| 指标 | `4eafae` | `06a741` | Δ |
|---|---|---|---|
| Recall@20 | 0.7019 | 0.6989 | −0.0030 |
| Recall@5 | 0.5108 | 0.5008 | −0.0100 |
| MRR | 0.4394 | 0.4312 | −0.0082 |
| 引用定位正确率 | 1.0000 (1,589) | 1.0000 (1,642) | — |
| 核心引用覆盖率 | 0.1867 | 0.1700 | −0.0167 |
| 拒答率 | 0.8500 | 0.8562 | +0.0062 |
| 方向题 | 0.6771 | 0.6354 | −0.0417 |
| **引用校验硬失败** | 0 | **0**（修复前 1） | — |

分类（新世代）：adversarial 16 拒1 引用5.06 / alias 48 拒6 4.50 / cross_chunk 32 拒5 3.12 /
direct_relation 48 拒0 5.98 / multihop 48 拒0 6.94 / negative_abstain 32 拒4 4.72 /
single_chunk 64 拒16 4.08 / time_conflict 32 拒0 6.66。

**这批差异尚未做显著性检验**。按本项目纪律，只能说「在 n=300 上观察到这些差，未检验」，
不许说成「K2 更差」或「K2 更好」。K3 要做的事正是这个。

### 7. 两个缺陷及其修复

#### 7.1 缺陷 A：世代不自带块正文（架构级）

`q-sc-000024` 一条引用硬失败：`chunk_id=activities/a001/level_a001_01_end#000`、
`start=95 end=114`、`evidence="可萝尔：刚才，有个人，从窗户外飞了过去"`，
而 `text[95:114] = "尼：是这样没错......唔，我们两个"`。证据确实在该块正文里，
位置在 **966**——不是块错了，是**偏移和正文不是同一版**。

根因：`pipeline.load_generation()` 从 `config.chunks_path()`（全局
`data/arkrag/chunks.jsonl`，被 legacy-baseline 冻结的 L1 旧产物）读块正文，
而 `build_generation.py` 建库时用的是 staging 的新分块。**世代目录里没有自己的
`chunks.jsonl`**（九个旧世代全部没有）。

量化（当时用一个一次性探针 `_tmp_serving_gap.py` 量的，该脚本未入库、**它量的那个状态在缺陷
修好后已经不存在了**——候选世代现在自带 `chunks.jsonl`，`resolve_chunks()` 不会再退到全局文件，
所以这组数字无法再重跑，只作为缺陷当时的证据留档）：

| 项 | 值 |
|---|---|
| 索引块数（世代声称） | 13,798 |
| 服务端块文件 | 13,786（T1 13,781） |
| **索引有、服务端无** | **17 块** |
| **两侧都有但正文变了** | **17 块** |

后 17 块是「不报错」的那种坏：索引按新正文建、偏移按新正文记，服务端拿旧正文比对 →
必然对不上 → 引用被静默剔除、置信度降低，**表现为「答得短」而不是「服务坏了」**。

`chunk_manifest_hash` 能唯一标识每一代那批块（`4eafae` 的 `34e6971d…` 只能由
frozen T1 复现，`27be38` 的 `e9cf0a92…` 只能由 staging T1 复现），**但这层身份
没有任何消费者用来定位块文件**。

严重性：线上 `55dfd7` 是同样形态，它今天能工作只因为「全局 `chunks.jsonl` 恰好就是
它那一批」。下一次 L2 重建一覆盖全局文件，线上立刻静默错位——而 38 门全 PASS、
`verify_generation` 0 问题，**门禁和自检都只在这一代自己的产物内比对，从没人问
「消费者拿哪份块正文」**。

唯一拦住它的是 `/ready` 的 `chunks_cover_generation`（J 阶段设成阻断项）。
对候选跑七项：`ready=False`，只有这一项 FAIL（「索引有 13798 块，块正文缺 17 块」）。
**这个 FAIL 现在复现不出来**：缺陷修好后 `resolve_chunks()` 对候选返回自带文件，实测
`server._compute_chunks_cover('g-20261103T132152Z-06a741')` → `ok=True`、
`索引 13798 块均在块正文中`；`4eafae` / `55dfd7` 借全局文件同样 `ok=True`。

修复（五处）：`Generation.chunks_path` 属性 / `pipeline.resolve_chunks()` fail-closed /
`build_generation.py` 落盘块正文 / `release.verify_generation` 增查 / `server` 改用同一定位。

#### 7.2 缺陷 B：图扩展把外块偏移登记到当前块

`retrieve.build_context()` 原写 `target = (ensure(host_id) if host_id else None) or unit`。
宿主块载入不回来时（缺陷 A 造成的缺 17 块），`or unit` 把该断言的偏移登记到**当前
图扩展命中的那个单元**上。

实测断言 `a-6b8d3ac8bf188882`：属 `…#001`（新增块，缺失）、`start/end=95/114`，
被登记到 `…#000`（旧块，992 字）→ `text[95:114]` 切出另一句话。

改为「宿主块载入不回来就不登记」。没有正确偏移可用时，唯一诚实的做法是不给出。

### 8. `tokenizer_version` 从 `t1` 变 `t2` 是已知的

`index_build_fp` / `index_content_digest` / `probe_digest` 全变，原因有三层叠加：

1. 块数 13,781 → 13,798（+17）；
2. **分词器 `t1` → `t2`**：`TOKENIZER_VERSION = "t2"` 是 F 阶段 `ada2ce2`
   （2026-10-31 15:41）引入的，而已发布世代都是 F 之前建的，索引里记的是 `t1`；
3. 索引文件字节数 127.2 MB → 106.7 MB（`t2` 生成的词项更少）。

**所以这不是未解释的产物身份漂移**，而是「F 阶段的分词器变更第一次真正进到产物里」。

### 9. 顺带修掉：基线校验把「合法追加」判成篡改

34 条新抽取入库后 `snapshot_baseline.py --verify` FAIL：

```
缓存目录已变: data/arkrag/cache count 13801->13835 digest a244f0362aa4->78353ab867af
```

而 `legacy.guard_frozen_artifact()` 的 docstring 明确写着**刻意不拦这个目录**，
理由是「缓存目录里新记录以新键追加写入是正常运行」。两条规矩打架的结果是：
**每跑一次合法抽取，这条门就红一次**——它只有在「永不抽取」时才是绿的。

**先证明差异是纯追加**：冻结块 `content_key` 13,786 个、staging 新增 34 个；按记录里的
`content_key` 给 13,835 个缓存文件分堆，属于新增集合的正好 **34** 个，其余 **13,801** 个
重算摘要 = `a244f0362aa423dc`，**与基线逐字相同**。

修法：给基线补逐文件清单 `data/releases/legacy-baseline/cache-inventory.json`
（相对名 / 字节 / SHA-256），校验分两类判：冻结过的被改/被删 → 红；多出新文件 → 只报数量。

清单不能用来掩盖篡改：`cache_inventory_digest(rows)` 必须逐字等于 Manifest 里的
`digest`（与 `legacy.dir_digest()` 同算法）。**Manifest 一字未动**——`created_at`、
artifacts、gates 冻结副本、metrics 全部原样，只多写一个文件。

补清单也不靠猜：Manifest 里那个 `digest` 早就把清单摘要冻结住了，补出来的必须能
重算回它（实测 13,801 条 → `a244f0362aa423dc`，逐字相同）。

### 10. 顺带修掉：一条用例打在了没人读的文件上

`test_missing_chunk_text_blocks_ready` 截断的是**全局** `data/chunks.jsonl`，
而缺陷 A 修复后消费者改读世代自带的 `chunks.jsonl` → 它打到了一个没人读的文件上，
既拦不住真问题，也测不到它想测的东西。

改成打 `resolve_chunks()` 真正返回的文件后，暴露更深的问题：截断会被**身份校验**
先拦下（`chunk_manifest_hash` 不符，fail-closed），逐块比对那一步**从来没被执行过**，
却一直跟着用例一起算通过。

拆成三条：①截断被身份校验拦下；②直接调 `server._compute_chunks_cover()` 验证
「块齐时绿、少一块时点名缺哪块」；③反向验证截断全局文件**不再**影响就绪
（这是接线真的生效的证据）。

### 11. 回归门

| 命令 | 结果 |
|---|---|
| `py -3 -m pytest tests -q` | **1288 passed, 1 skipped** in 1073.80s（1289 collected），EXIT=0 |
| `py -3 scripts/snapshot_baseline.py --verify` | **PASS 产物 11 项全部一致**，附「新增 34 条缓存记录（正常追加，不计入失败）」，EXIT=0 |
| `py -3 scripts/k1_checks.py` | **九项全 PASS**，可服务世代 8 个，回滚演练 `06a741 → e0aacc` 成功，EXIT=0 |
| `py -3 scripts/ask.py --evaluate --limit 320`（`06a741`） | 见 §6，引用硬失败 **0**，EXIT=0 |
| `py -3 scripts/ask.py --evaluate --limit 320`（`4eafae` 复测） | 与 I/J 阶段逐位相同，**证明修复没改变旧世代行为** |

### 12. 新增与改动文件

| 文件 | 变化 |
|---|---|
| `src/arkrag/pipeline.py` | +`ChunksUnavailable` / +`resolve_chunks()` / `load_generation()` 改调它 |
| `src/arkrag/release.py` | +`Generation.chunks_path` / `verify_generation()` 增查块正文身份 |
| `src/arkrag/retrieve.py` | `build_context()` 删掉 `or unit` 回退 |
| `src/arkrag/server.py` | `_compute_chunks_cover()` / `_CoverCache.key()` 改用 `resolve_chunks()` |
| `scripts/build_generation.py` | 块正文随世代落盘 / `build_probes()` 补 `name_terms` |
| `scripts/snapshot_baseline.py` | 缓存目录改按逐文件清单比对 |
| `data/releases/legacy-baseline/cache-inventory.json` | 新增（13,801 条 / 1,945,066 B） |
| `tests/test_chunk_snapshot.py` | 新增 14 用例 |
| `tests/test_baseline_cache_inventory.py` | 新增 14 用例 |
| `tests/test_server.py` | 58 → 63 用例（拆一条 + 补两条） |

卫生门 `tests/test_suite_hygiene.py` 的 `N`：1251 → 1269 → 1287 → **1289**。

### 13. 复现命令

```bash
# 1. L1 / L2 跑进 staging（不碰冻结产物）
$env:ARKRAG_ARTIFACT_DIR="data/arkrag/staging/k2"
py -3 scripts/run_l1.py
py -3 scripts/run_l2.py
py -3 scripts/run_l3_agent.py --status          # 期望 13764/13803
py -3 scripts/run_l3_agent.py --plan --label K2 --batch-size 4
# ... 抽取产出 data/arkrag/agent_outbox/K2.part*.jsonl ...
py -3 scripts/run_l3_agent.py --ingest --label K2
# 2. 建候选（不发布）
py -3 scripts/build_generation.py --graph-model identity --no-publish
# 3. 验收
py -3 scripts/release_cli.py verify <gid>
py -3 scripts/ask.py --evaluate --gold-set data/goldset/gold-v1.json --limit 320 --generation <gid>
py -3 scripts/snapshot_baseline.py --verify
```

### 14. 遗留

见台账「K2 阶段」§11，五条：召回差异未做显著性检验 / `_compute_chunks_cover` 冷启动
约 6 秒 / 块正文每代多占 68.9 MiB 且旧世代仍借全局文件 / `fa22ae` 与 `27be38` 留盘待
K4 决定去留 / 「自己分词时要记得传 `name_terms`」这条只修了 `build_probes()`。

## K3 实测：新旧对比归因

### 0. 口径

1. **两代**：旧 `g-20261031T035231Z-4eafae`（线上上一代）vs 新候选
   `g-20261103T132152Z-06a741`（K2 产物，READY 未发布）。
2. **判据不是「差异很小」，是「未解释栏 = 0」**。K 阶段要求两代之间的每一处结构差异
   都能指到一个已记录的原因；指不到的那一类叫未解释的产物漂移，按硬门不许发布。
3. **本节所有数字都是实跑读数**，没有引用、估算或推算值。§2 / §3 / §9 / §10 引的是
   同一次归因脚本运行（EXIT=0）的输出，按层拆开贴。
4. **召回差异必须过配对检验才能下结论**。K2 §6 留的正是这句话，本节补上。
5. 本节**不改代码**，只读两代产物与抽取缓存。

### 1. 判据：四层差异各按什么键比

| 层 | 比较键 | 为什么是这个键 |
|---|---|---|
| 块 | `chunk_id` → `content_key` | `content_key` 是正文哈希，正文没变它就一定不变；这是「新增或变更块」的定义 |
| 实体 | 实体 `id` | 同 id 换名是**改名**，不是「删一个 + 加一个」。按名字比会把一次改名记成两条差异 |
| 边 | 展示名三元组（subject 名 / 谓词 / object 名） | `edge_id` 由两端实体 id 派生，见下：两种键在本例里都会骗人 |
| 断言 | `assertion_id`，再按落点分类 | 断言要区分「随新块进来的」与「只是跟着改名换 id 的」 |

第三条是本节唯一一处「不按 id 比」的地方，理由不是「按 id 更好」，而是**两种键在这份数据上
都会给出错形状的报告**。实测（见 `scripts/k3_attribution.py` §5.2 / §5.3）：

```
=== 两种键各得到多少「差异」 ===
  按 edge_id ：仅旧 57 / 仅新 68（合计 125）
  按名三元组 ：仅旧 19 / 仅新 30（合计 49）

同名三元组在两代都出现：28,606 条
  其中 edge_id 相同：28,562
  其中 edge_id 不同：44        ← 全部 44 条都涉及 莱伊 / 劫车人
```

两个方向各错一次：

- **按 `edge_id` 比会把差异数放大到 125。** 因为 `莱伊` 这次拿了新 id
  （`e-6fa1624b83f766b0` → `e-af2211fa9e8022ef`），凡是 `莱伊` 参与的边
  `edge_id` 全换，44 条同名三元组的 id 都对不上了。读报告的人会看到 125 处变化，
  而实际变化只有 49 处（其中 30+19 里还有 7 条是同一条边的两个名字）。
- **只看名字又会漏掉「只改显示名、没改 id」的那一侧。** 那 7 条 rename 边里，
  `劫车人` 一侧用的还是 `e-6fa1624b83f766b0`，所以 `莱伊 -持有-> 弩`（旧）与
  `劫车人 -持有-> 弩`（新）**共享同一个 `edge_id` `r-ed9a0e4db94e7c85`**——
  按 id 比这条边是「完全没变」，改名这件事直接消失。

所以本节的报告口径是：**按展示名三元组出差异 + 显式声明一个 `RENAME` 集
（`{"莱伊","劫车人"}`）把改名涟漪收进去**，这样 49 处才能落到「新块 24+18、
改名 6+1」上，未解释栏才为 0。换任何一个单一键，都拼不出这张表。

每条差异必须落进 `NEW_CHUNK` / `RENAME` / `NEW_ENTITY` / `GONE_ENTITY` 四类之一，
落不进的计入 `UNEXPLAINED`——**这一栏非 0 就不许发布**。

### 2. 块层：34 块，与 K2 §3 的缺口逐一对上

```
===== 块 =====
  旧块 13,786 / 新块 13,803 / 新增或变更块 34
```

旧侧读的是全局 `data/arkrag/chunks.jsonl`（旧世代不自带块正文，K2 §7.1 缺陷 A），
新侧读候选自带的 `chunks.jsonl`。34 与 K2 §3 记录的那 34 块抽取缺口是同一批，
一一对应，没有第五种原因。

### 3. 实体层：净 +1 个实体，外加 1 次同 id 换名

```
===== 实体 =====
  仅旧 id 1 / 仅新 id 2 / 同 id 换名 1
    改名 e-6fa1624b83f766b0: 莱伊 -> 劫车人（mentions 53 -> 6）
    - e-e099352f9ae9ca4c 医疗部外走廊 (地点) chunks=1
    + e-95a35aa77c8a2b20 菲利普伯爵 (角色) chunks=1
    + e-af2211fa9e8022ef 莱伊 (角色) chunks=48
  实体数 old=6,582 new=6,583（+1）
```

| 差异 | 数量 | 内容 |
|---|---|---|
| 仅旧 id | 1 | `e-e099352f9ae9ca4c` 医疗部外走廊（地点），chunks=1 |
| 仅新 id | 2 | `e-95a35aa77c8a2b20` 菲利普伯爵（角色）chunks=1；`e-af2211fa9e8022ef` 莱伊（角色）chunks=48 |
| 同 id 换名 | 1 | `e-6fa1624b83f766b0`：莱伊 → 劫车人（mentions 53 → 6） |

`医疗部外走廊` 与 `菲利普伯爵` 是随 34 块新正文进来的：一进一出，净 +1，符合预期。

`莱伊 / 劫车人` 这一条**既不是新实体也不是消失实体**——它是同一个人在两代里
「合成一个」还是「拆成两个」的差别。§4 到 §8 单独拆。

### 4. 头号归因：`e-6fa1624b83f766b0` 是一次别名合并的去留

先说结论：**这个 id 哈希的从来就是「劫车人」。旧代它显示成「莱伊」，是因为它把
「劫车人」当别名并了进来；新代那次合并没了，显示名回落到 id 本来的出生名。**

```
stable_id( 劫车人 ) = e-6fa1624b83f766b0
stable_id( 莱伊 )   = e-af2211fa9e8022ef
sorted pair: ['劫车人', '莱伊']
```

```
旧 {"id": "e-6fa1624b83f766b0", "name": "莱伊", "type": "角色",
    "aliases": [{"text":"劫车人", ...},
                {"text":"莱叶菈", ...}],
    "merged_from": [{"name":"劫车人","reason":"alias",
                     "scenes":["activities/act16mini/level_act16mini_st02"]}],
    "mentions": 53}
新 {"id": "e-6fa1624b83f766b0", "name": "劫车人", "type": "角色",
    "aliases": [], "merged_from": [], "mentions": 6}
```

`identity.py` 里两件事要分开看：

- **id**：`_UnionFind` 合并时根恒取组内字典序最小的成员名（`if rb < ra: ra, rb = rb, ra`），
  `seed = root`，`stable_id()` 哈希的就是这个根。`sorted(['莱伊','劫车人'])` 的第一个是
  `劫车人`，所以 `劫车人` 那一组出来的 id 就是 `e-6fa1624b83f766b0`。
- **显示名**：另算，`canonical_pick(members, counts)` 按「出现次数多优先 → 含 `·` 优先 →
  短优先 → 字典序」四级排。

于是：**id 稳定，显示名会随「这一代把它跟谁合成一组」而变**。旧代 `莱伊` 的出现次数
远多于 `劫车人`，显示名取 `莱伊`，`劫车人` 退进 `aliases` 并记一条
`merged_from(reason=alias)`；新代两个名字不再合成一组，各自成实体，
`e-6fa1624b83f766b0` 回落到出生名 `劫车人`（mentions 6、只在 `st02` 一个场景），
`莱伊` 拿到新 id `e-af2211fa9e8022ef`（mentions 48、9 个场景）。

两代 `alias_index` 的实测对照（同一脚本）：

```
--- 旧 4eafae alias_index ---
  莱伊 -> (不在别名表)
  劫车人 -> e-6fa1624b83f766b0
  莱叶菈 -> e-6fa1624b83f766b0
  实体 e-6fa1624b83f766b0 name=莱伊 mentions=53 chunks=53 场景数=9

--- 新 06a741 alias_index ---
  莱伊 -> (不在别名表)
  劫车人 -> (不在别名表)
  莱叶菈 -> e-af2211fa9e8022ef
  实体 e-6fa1624b83f766b0 name=劫车人 mentions=6 chunks=6 场景数=1
      来自 activities/act16mini/level_act16mini_st02
  实体 e-af2211fa9e8022ef name=莱伊 mentions=48 chunks=48 场景数=9
```

`劫车人` 在新代不再是任何人的别名，`莱伊` 也不是——两个名字都成了独立实体的正名。

### 5. 触发点：`level_act16mini_st02#009` 的正文从 985 字变成 875 字

```
目标场景 activities/act16mini/level_act16mini_st02

=== 全局 data/arkrag/chunks.jsonl（线上世代借的就是它）===
--- global ---
  ...#000  字数 878  content_key dcfd388503963c3998b62e6b
  ...#001  字数 871  content_key 6694c218bbaf337b28632406
  ...#002  字数 898  content_key a835f1019cc73983ed0e0736
  ...#003  字数 867  content_key bed7c900d85c8a118baba404
  ...#004  字数 888  content_key ffa5de957f223911ea201a66
  ...#005  字数 864  content_key 710a66335c8eb6635d039c2a
  ...#006  字数 853  content_key 933656bd00e6ca49827dd7c3
  ...#007  字数 899  content_key e2419615b335700b8faf9cf7
  ...#008  字数 898  content_key 7d28df3ce0e9f1689ea1dafa
  ...#009  字数 985  content_key 973194376d0ebe71919d64d1

=== 旧世代 g-20261031T035231Z-4eafae ===
  （该世代没有自带 chunks.jsonl，借全局文件）

=== 新候选 g-20261103T132152Z-06a741 ===
--- new 自带 ---
  ...#000  字数 878  content_key dcfd388503963c3998b62e6b
  ...#001  字数 871  content_key 6694c218bbaf337b28632406
  ...#002  字数 898  content_key a835f1019cc73983ed0e0736
  ...#003  字数 867  content_key bed7c900d85c8a118baba404
  ...#004  字数 888  content_key ffa5de957f223911ea201a66
  ...#005  字数 864  content_key 710a66335c8eb6635d039c2a
  ...#006  字数 853  content_key 933656bd00e6ca49827dd7c3
  ...#007  字数 899  content_key e2419615b335700b8faf9cf7
  ...#008  字数 898  content_key 7d28df3ce0e9f1689ea1dafa
  ...#009  字数 875  content_key 1f791cb6ccf7b7fb7e343428
  ...#010  字数 124  content_key de9ec6205a0fe63acee60f65
```

（上面把公共前缀 `activities/act16mini/level_act16mini_st02` 缩成 `...` 以便贴进文档，
字数与 `content_key` 是原文。）

同场景其余 9 块（`#000`–`#008`）的 `content_key` 一字未变，只有 `#009` 变了，
并且多出一个短尾块 `#010`。这正是 K2 §3 记的「17 对：一个正文变了的主块 + 一个新增短尾块」
形态。

缓存键绑 `content_key`（K2 §3 的结论），所以旧 n1 缓存对这一块**失效**，必须重抽；
重抽走的是 n2 归一化路径。**n1 → n2 的交接点就在这一块上。**

### 6. 整条别名关系只有一个来源，而那个来源正好是变了的那一块

```
===== 把 `莱伊` 写成别名的记录 =====
  [n1] activities/act16mini/level_act16mini_st02#009  实体 劫车人  aliases=['莱伊']
  合计 1 条

===== 把 `劫车人` 写成别名的记录 =====
  合计 0 条
```

`劫车人` 作为实体名被 7 条缓存记录声明（`st02#000`–`#003`、`#007`、`#008`、`#009`，
全部 n1），其中只有 `#009` 带 `aliases:["莱伊"]`；`莱伊` 作为实体名被 48 条声明
（46 条 n1 + 2 条 n2，即 `st02#009` / `#010`）。

**「莱伊 = 劫车人」这条知识没有冗余**：它挂在一块的抽取结果上，那块正文一变，
它就没有替代来源。这就是「一次别名合并的去留」——不是模型两次判断不一致，
是这条判断只被写下来过一次，而那一次被作废了。

两份缓存记录的原文对照：

```
旧 [n1] ck=973194376d0ebe71919d64d1  written_at 2026-10-29T11:04:18Z
  E {"name": "阿兰娜",  "type": "角色", "aliases": [], "desc": "运载车驾驶员"}
  E {"name": "杰里",    "type": "角色", "aliases": [], "desc": "自称普通乘客的札拉克人"}
  E {"name": "劫车人",  "type": "角色", "aliases": ["莱伊"], "desc": "前往咧嘴谷寻找巨大阴影的人"}
  E {"name": "小锅盖",  "type": "角色", "aliases": ["温米"], "desc": "被劫车人救下的卡特斯女孩"}
  E {"name": "咧嘴谷",  "type": "地点", "aliases": [], "desc": "被天灾摧毁的荒地废墟"}
  R 杰里 -交谈-> 阿兰娜 / 劫车人 -交谈-> 阿兰娜 / 阿兰娜 -交谈-> 劫车人 / 劫车人 -前往-> 咧嘴谷 / 劫车人 -救助-> 小锅盖

新 [n2] ck=1f791cb6ccf7b7fb7e343428  key_scheme=2  written_at 2026-11-03T12:38:53Z
  E {"name": "阿兰娜", "type": "角色"}
  E {"name": "杰里",   "type": "角色"}
  E {"name": "莱伊",   "type": "角色"}
  E {"name": "小锅盖", "type": "角色"}
  E {"name": "咧嘴谷", "type": "地点"}
  R 阿兰娜 -交谈-> 杰里 / 阿兰娜 -交谈-> 莱伊 / 莱伊 -交谈-> 阿兰娜 / 莱伊 -救助-> 小锅盖 /
    莱伊 -前往-> 咧嘴谷 / 阿兰娜 -前往-> 咧嘴谷
```

新记录里这一块的正文已经是莱伊在报名字了（`阿兰娜：对了，你叫什么？` /
`矿场的人都叫我莱伊。`），和 K2 §3 记的那次说话人标签变动是同一处。

注意新记录**没有任何 `aliases` 键，也没有 `desc` 键**，连 `阿兰娜` / `杰里` 这种
n1 里带 `desc` 的都没有。这不是「模型这次只写了名字」——§7 给出机制。

### 7. 新记录里没有 aliases：契约在入库路径上把字段丢了

`prompt.py` 第 8 条硬规则要求「别名放进 aliases」，输出示例里也带
`"aliases":["罗德岛领袖"]`。但把带别名的实体喂给 `contracts.validate_extraction()`：

```
输入 entities[0]：
  {"name": "劫车人", "type": "角色", "aliases": ["莱伊"], "desc": "前往咧嘴谷的人"}

校验通过。cleaned entities[0]：
  {"name": "劫车人", "type": "角色", "desc": "前往咧嘴谷的人"}

  保留的键 ['desc', 'name', 'type']
  aliases 还在吗 = False
  desc 还在吗    = True

提示级告警（不影响录用）： []
```

**aliases 被静默丢掉，`errors` 为空、没有任何告警。** 根因在
`src/arkrag/contracts.py:205-211`：校验通过后是逐字段重建 `item`——`item = {"name": name}`
起手，再补 `type` / `id` / `desc`，**从不补 `aliases`**。而
`scripts/run_l3_agent.py --ingest` 落缓存时用的正是 `cleaned`（`src/arkrag/agent_tasks.py:355`
的 `raw = {"entities": cleaned["entities"], ...}`）。所以 n2 路径上，模型即使按提示词
写了别名，也到不了图里。

这条不是 K2 引入的：`git log -S aliases -- src/arkrag/contracts.py` 为空，基线上
同样是只补 `desc`。它是**长期存在的丢字段路径，直到 n2 才第一次被走到**——n1 那批
缓存走的是更早的直接落 `raw` 的路径，所以带别名。

本轮**没有修**它（K4 的验收面是发布与回滚，不是抽取契约），登记在 §14 遗留。

### 8. 但别名总量只少了 1 条——不许写成「别名整体丢了」

```
                            旧 4eafae      新 06a741           差
entities                       6,582         6,583           1
with_alias                     1,122         1,122           0
alias_total                    1,672         1,671          -1
pending                          562           562           0
alias_index                    1,672         1,671          -1

有别名实体占比：旧 17.0465%  新 17.0439%
```

净损失**正好 1 条**，就是 `劫车人 → 莱伊`。原因：13,835 条缓存里 13,801 条是 n1，
它们的 `raw.entities` 带别名，而 `cache.adopt_legacy()` 把源记录原样透传
（`write_raw(new_key, src_raw, ...)`，不是写 `cleaned`）。别名没有跟着 n2 一起消失。

这条同时是本节第二个「不许写夸张结论」的例子：§7 的契约丢字段是真的，但若据此说
「n2 把别名整体丢了」，图上 1,671 个别名条目会直接否掉这句话。**一处真缺陷不等于
一个系统性问题，必须数一数再说。**

### 9. 边层：仅新 30 / 仅旧 19，未解释 0

```
===== 边（按展示名三元组比）=====
  旧边 28,625 / 新边 28,636 / 仅新 30 / 仅旧 19
  仅新/NEW_CHUNK: 24
      临光 -敌对-> 莱塔尼亚
      商业联合会 -包含-> 无胄盟
      嘉维尔 -交谈-> 灰掌
      林雨霞 -使用-> 炸弹
      林雨霞 -持有-> 炸弹
      歌蕾蒂娅 -敌对-> 盐风城
      海嗣 -包含-> 歌蕾蒂娅
      海嗣 -敌对-> 乌尔比安
      灰掌 -隶属-> 罗德岛
      爱布拉娜 -保护-> 塔拉
      特尔斐运河 -位于-> 米诺斯
      玛拉帕夏 -属于-> 萨尔贡
      玛莉娅 -隶属-> 临光
      玛露西尔 -同盟-> 阿米娅
      艾丝黛尔 -交谈-> 灰掌
      艾丝黛尔 -同盟-> 特米米
      莱欧斯 -同盟-> 阿米娅
      赫拉格 -交谈-> 博士
      重岳 -前往-> 岁陵
      重岳 -称为-> 朔
      铃兰 -雇用-> 奥斯塔
      锡兰 -受保护于-> 黑
      阿兰娜 -同行-> 小锅盖
      陈 -任职-> 特别督察组
  仅新/RENAME: 6
      劫车人 -交谈-> 小锅盖
      劫车人 -交谈-> 阿兰娜
      劫车人 -前往-> 咧嘴谷
      劫车人 -持有-> 弩
      劫车人 -救助-> 小锅盖
      阿兰娜 -交谈-> 劫车人
  仅旧/NEW_CHUNK: 18
      可萝尔 -雇用-> 格拉尼
      小锅盖 -使用-> 秋千
      海嗣 -敌对-> 幽灵鲨
      爱布拉娜 -属于-> 德拉克
      爱布拉娜 -背叛-> 沃里克伯爵
      玛拉帕夏 -隶属-> 萨尔贡
      瑟奇亚克 -获救于-> 格蕾纳蒂
      索娜 -同盟-> 监正会
      艾丝黛尔 -救助-> 灰掌
      莫佚 -位于-> 岁陵
      莫佚 -赠予-> 重岳
      角峰 -保护-> 阿米娅
      赞索斯 -赠予-> 缇缇
      重岳 -亲属-> 均
      重岳 -亲属-> 望
      陈 -任职-> 龙门
      陈 -师从-> 维多利亚皇家近卫学院
      黑 -源自-> 哥伦比亚
  仅旧/RENAME: 1
      莱伊 -持有-> 弩
  未解释边：0
```

两类各有分工：

- **NEW_CHUNK**（仅新 24 / 仅旧 18）：来自那 34 块新正文。`仅旧 18` 是旧块里的边在新块里
  没抽出来——同一段正文换了抽取就未必抽全，这是抽取的非确定性，不是图构建的错。
- **RENAME**（仅新 6 / 仅旧 1）：这 7 条全是同一件事的两面。`莱伊 -持有-> 弩` 与
  `劫车人 -持有-> 弩` 是同一条边；其余 6 条是旧代里 `劫车人` 作为别名被藏起来，
  按名字看就成了「新出现」。

边数 28,625 → 28,636（+11），与 K2 §5 表一致。

### 10. 断言层：仅新 208 / 仅旧 194，未解释 0

```
===== 断言 =====
  旧 62,910 / 新 62,924（+14）
  仅新 208 = 落新块 117 + 改名涟漪 91 + 未解释 0
  仅旧 194 = 落新块 103 + 改名涟漪 91 + 未解释 0

未解释合计（边 + 断言）：0
结论：全部差异已归因，未解释 0
```

两个 91 是同一批断言：它们的 `assertion_id` 含实体 id 段与 chunk 段，改名/重组后 id 换了，
在老代与新代各出现一次，于是「仅新 91 + 仅旧 91」。这 91 对全部涉及 `莱伊` / `劫车人`。

断言总数 +14，等于「新块带来的净增」：仅新 208 减仅旧 194。

### 11. 召回差异的显著性：配对符号检验 + bootstrap

先复测两代指标（两代各跑一遍 320 题 Gold Set，EXIT=0）：

```
指标                          值           能算      跳过      说明
Recall@20                   0.7019      300     20
Recall@5                    0.5108      300     20
MRR                         0.4394      300     20
引用定位正确率                     1.0000      1589    0
关键事实引用覆盖率                   0.1867      300     20
拒答题正确率                      0.8500      320     0
方向题正确率                      0.6771      96      224
引用校验硬失败 0 条
```
（上：`g-20261031T035231Z-4eafae`）

```
指标                          值           能算      跳过      说明
Recall@20                   0.6989      300     20
Recall@5                    0.5008      300     20
MRR                         0.4312      300     20
引用定位正确率                     1.0000      1642    0
关键事实引用覆盖率                   0.1700      300     20
拒答题正确率                      0.8562      320     0
方向题正确率                      0.6354      96      224
引用校验硬失败 0 条
```
（上：`g-20261103T132152Z-06a741`）

320 题里 20 题没有 gold（`negative_abstain` / `adversarial` 的一部分），
评测器算 R@5 / R@20 / MRR 时分母是这 300 题、跳过 20 题。

**配对检验用两套口径各算一次**，因为这两套在这份数据上给出的 RR 不一致，必须都摊开：

口径 A —— **配对子集口径**，生产者脚本 `scripts/k3_signif.py`（`data/k3_recall_pairs.json`
就是它写的），`pairs()` 里 `if a is None or b is None: continue`：

```
指标           旧均值       新均值         Δ    新胜    旧胜    持平        p  bootstrap 95%
r5        0.5108    0.5008   -0.0100    11    10   279   1.0000  [-0.0333, +0.0125]
r20       0.7019    0.6989   -0.0031    11    12   277   1.0000  [-0.0242, +0.0172]
rr        0.5700    0.5619   -0.0081    55    40   134   0.1505  [-0.0326, +0.0162]
```

这张表三行的**均值分母并不相同**，实测（`scripts/k3_probe_denominator.py` 段一）：

```
共同题 320
  r5：两代都有 300 / 仅旧有 0 / 仅新有 0 / 都没有 20
 r20：两代都有 300 / 仅旧有 0 / 仅新有 0 / 都没有 20
  rr：两代都有 229 / 仅旧有 6 / 仅新有 6 / 都没有 79

旧代 rr 为 None 85（其中无 gold 20）
新代 rr 为 None 85（其中无 gold 20）

配对子集 n=229  rr 旧 0.5700 / 新 0.5619 / Δ -0.0081
0 填充口径 n=320  rr 旧 0.4120 / 新 0.4042 / Δ -0.0077
```

所以口径 A 的准确标签是：**r5 均值 n=300、r20 均值 n=300、rr 均值 n=229**。
`rr` 那一行**不能**与台账里的 `MRR 0.4394 / 0.4312` 混着引用——后者是 0 填充的
300 题口径（`scripts/k3_probe_denominator.py` 段二实测印证：`有 gold 的 300 old: MRR 0.4394`）。
符号检验的 n 比均值分母还小，因为持平不计入：口径 A 是 r5 21（11+10）、r20 23（11+12）、
rr 95（55+40）。**Δ 与 p 值不受分母标注影响**——配对检验本来就只在可比对上做，
受影响的只是均值能不能跨行比。

口径 B —— 与评测器对齐（`scripts/ask.py:279` 是
`rr.append(0.0 if first is None else 1.0/(first+1))`，未命中记 0 而不是丢弃），
`None` 一律 0 填充，n=300：

```
[R@5] n=300
  旧 0.5108 / 新 0.5008 / Δ -0.0100
  新胜 11 / 旧胜 10 / 持平 279 → 符号检验 n=21
  p = 1.0000
  bootstrap 95% [-0.0333, +0.0125]  跨 0=True

[R@20] n=300
  旧 0.7019 / 新 0.6989 / Δ -0.0031
  新胜 11 / 旧胜 12 / 持平 277 → 符号检验 n=23
  p = 1.0000
  bootstrap 95% [-0.0242, +0.0172]  跨 0=True

[RR] n=300
  旧 0.4394 / 新 0.4312 / Δ -0.0082
  新胜 61 / 旧胜 46 / 持平 193 → 符号检验 n=107
  p = 0.1756
  bootstrap 95% [-0.0284, +0.0116]  跨 0=True

=== 对照：RR 把未命中题当缺失丢掉会怎样（错口径）===
  n=229 旧 0.5700 / 新 0.5619 / Δ -0.0081（台账记的是 0.4394 / 0.4312，对不上）
```

最后那段是脚本自带的一行对照（脚本自己把它标成「错口径」，指**与评测器口径不对齐**，
不是说这种算法本身错）。它值得单独说：同一份数据在「丢弃未命中」与「未命中记 0」
两种口径下，RR 的均值差 **0.13**（0.5700 vs 0.4394）。两个数都对，但只要有一个
不写分母，另一个就成了错的。**这是同一份数据能生出两个「事实」的标准形态。**

| 指标 | Δ（A） | p（A） | 95% 区间（A） | Δ（B） | p（B） | 95% 区间（B） | 跨 0 |
|---|---|---|---|---|---|---|---|
| R@5 | −0.0100 | 1.0000 | [−0.0333, +0.0125] | −0.0100 | 1.0000 | [−0.0333, +0.0125] | 是 |
| R@20 | −0.0031 | 1.0000 | [−0.0242, +0.0172] | −0.0031 | 1.0000 | [−0.0242, +0.0172] | 是 |
| RR | −0.0081 | 0.1505 | [−0.0326, +0.0162] | −0.0082 | 0.1756 | [−0.0284, +0.0116] | 是 |

**R@5 / R@20 两套口径逐位相同；RR 不同，因为分母不同（229 vs 300）。**

口径 B 才是评测器打印 `MRR 0.4394 → 0.4312` 的那个口径；口径 A 的 rr 均值
（0.5700 / 0.5619）在评测报告里找不到对应行，因为它把未命中题排除掉了。
两个口径都对，但**报数字时必须说清分母**。

**结论（对两种口径都成立）**：三个指标的 95% 区间全部跨 0，符号检验也没有一项
接近显著。所以唯一许可的读法是——

> 候选在召回上没有可分辨的变化，不是回退也不是提升。

配对样本只有 21 / 23 题在 R@5 上发生变化（其余 279 / 277 题两代一模一样），
这个样本量下用符号检验去分 11:10，本来就分辨不出东西。要把 0.01 量级的差异做实，
需要的是更多题、或者多次重抽取平均，而不是换检验方法。

### 12. 索引体积：词项 −43.76% 不是语料缩水

```
=== 两代索引 meta ===
  old tokenizer_version = 't1'  doc_count=None
  new tokenizer_version = 't2'  doc_count=None

=== 倒排规模 ===
  old: 文档 13781 / 词项 694,304 / 条目 10,525,428 / 平均每词 15.160 / lexical.json 127,244,898 B
  new: 文档 13798 / 词项 390,508 / 条目 8,996,776 / 平均每词 23.039 / lexical.json 106,685,068 B

  文档 13,781 → 13,798（+17）
  词项 694,304 → 390,508（-303,796，-43.76%）
  条目 10,525,428 → 8,996,776（-1,528,652，-14.52%）
  字节 127,244,898 → 106,685,068 B（-16.16%）

=== 同一批正文在 t1 / t2 下的 token 数（前 300 块 T1）===
  专名表 8254 项
  t1: 总 token 371979 / 平均每块 1239.9
  t2: 总 token 344876 / 平均每块 1149.6
  t2/t1 = 0.9271（正文量纲只缩了 7.29%）

=== 专名整词入表贡献（t2 独有）===
  t2 传 name_terms 总 token 344876 / 不传 339289
  差 -5587（专名整词并入后减少的分词条目数）+ 专名整词本身 8177 类词项
  专名表项数 = 8177
```

文档数**增加** 17，词项却掉 43.76%，条目只掉 14.52%，字节只掉 16.16%——
四个降幅互不相同，这本身就说明「词项数」和「语料量」不是一个量纲。

同一批正文在 t1 / t2 下差了 **7.29%**（t2/t1 = 0.9271）。词项数多掉的那 36 个百分点
来自 t2 的两处口径变化：

1. **专名整词入表**：`name_terms` 把 8,177 类专名作为一个整词项直接写进倒排，
   不再靠「首字 + 双字」去覆盖它们；
2. **专名边界不连双字**：`index.py` 的 `cuts` 让专名区间内侧不再跨边界切双字，
   一次专名出现产生的双字条目因此变少（实测传 `name_terms` 后 token 少 5,587 个）。

所以这句话只能这样说：**词项数下降到 390,508 是分词口径变了，不是语料缩水。**
把它写成「索引瘦身 44%」或者「语料少了 44%」都是错的——前者把口径变化当成了优化成果，
后者与「文档数 +17、正文 token 只 −7.3%」直接矛盾。

（专名表的两个数要说清：原始 `names` 集合 **8,254** 项，`_as_table` 去重成**词项类 8,177** 项
——差值是同一专名在不同场景重复登记。）

### 13. 其他指标与它们能说什么

| 指标 | `4eafae` | `06a741` | Δ | 本轮是否做过检验 |
|---|---|---|---|---|
| 引用定位正确率 | 1.0000 (1,589) | 1.0000 (1,642) | — | 分母是引用总条数，两代都 0 失败 |
| 核心引用覆盖率 | 0.1867 | 0.1700 | −0.0167 | **未做** |
| 拒答率 | 0.8500 | 0.8562 | +0.0062 | **未做** |
| 方向题正确率 | 0.6771 | 0.6354 | −0.0417 | **未做** |
| 引用校验硬失败 | 0 | 0（K2 修复前 1） | — | 计数，不是比例 |

`引用定位正确率` 两代都是 1.0000，括号里是**引用总条数**（1,589 → 1,642），
不是失败数——两代失败都是 0。K2 §7.1 修掉「世代不自带块正文」之后，定位校验才第一次
真的能执行；1,642 条引用全部定位成功，是那个修复生效的证据。

后三项**本轮没有做配对检验**，所以本节只报观测值，不说「变差」。
方向题 −0.0417 是这几项里最大的一个，它值得在 K5 做一次配对检验——但今天没有数据支撑
任何方向的结论，写「方向题下降」和写「方向题提升」一样是没有依据的。

分类明细（可作后续检验的分层依据）：

```
        类别        旧(条/拒/引用)     新(条/拒/引用)
adversarial          16 / 1 / 4.56      16 / 1 / 5.06
alias                48 / 7 / 4.19      48 / 6 / 4.50
cross_chunk          32 / 7 / 3.03      32 / 5 / 3.12
direct_relation      48 / 0 / 5.73      48 / 0 / 5.98
multihop             48 / 0 / 6.69      48 / 0 / 6.94
negative_abstain     32 / 4 / 4.59      32 / 4 / 4.72
single_chunk         64 /15 / 4.02      64 /16 / 4.08
time_conflict        32 / 0 / 6.81      32 / 0 / 6.66
```

八类里七类的平均引用数都升了，只有 `time_conflict` 从 6.81 掉到 6.66。这不能拿来
说「新代引用更集中」——引用数是「平均返回几条引用」，它是配置与拒答行为的函数，
不是质量指标，两代 `top_k` 与 `graph_depth` 相同的情况下它随图结构微调而变，
和多召回一条两回事。

### 14. 复现命令与遗留

```bash
# 归因（块/实体/边/断言四层，未解释栏必须为 0）；§5 同场给出「边为什么按展示名比」
py -3 scripts/k3_attribution.py
# 身份机理：stable_id 与别名分组
py -3 scripts/k3_probe_identity.py
# 别名：声明形态、来源唯一性、两代总量、alias_index（四段顺序执行）
py -3 scripts/k3_probe_alias.py
# 契约是否保留 aliases
py -3 scripts/k3_probe_contract.py
# 触发点：同场景逐块 content_key
py -3 scripts/k3_probe_chunk.py
# 显著性（两种口径各一份）
py -3 scripts/k3_signif.py            # 口径 A，且会重写 data/k3_recall_pairs.json
py -3 scripts/k3_signif_eval.py       # 口径 B，只读 data/k3_recall_pairs.json
# 三行指标的分母标注（段一 pair / 段二 mrr）
py -3 scripts/k3_probe_denominator.py
# 索引体积与分词口径
py -3 scripts/k3_probe_index.py
# 两代指标复测
py -3 scripts/ask.py --evaluate --gold-set data/goldset/gold-v1.json --limit 320 --generation g-20261031T035231Z-4eafae
py -3 scripts/ask.py --evaluate --gold-set data/goldset/gold-v1.json --limit 320 --generation g-20261103T132152Z-06a741
```

遗留四条：

1. **`contracts.validate_extraction()` 丢 `aliases`（`src/arkrag/contracts.py:205-211`）未修。**
   影响面是「所有走 n2 路径的新抽取都不带别名」。本轮实测图上的别名净损失只有 1 条
   （§8），但那是靠 13,801 条 n1 老的带别名记录撑着——**下次全量重抽（而不是增量 34 块）
   会把 1,671 条别名一起丢掉**。这是 K5 之前必须修的。
2. **口径 A / B 在 RR 上不一致，`scripts/k3_signif.py` 的 `pairs()` 应当与
   `scripts/ask.py:279` 对齐**（未命中记 0，不丢弃）。本节把两套都留下了，因为
   改脚本会改 `data/k3_recall_pairs.json` 的内容与所有引用它的数字。
3. **核心覆盖、拒答率、方向题三项未做配对检验。** 方向题 −0.0417 是最大的一个，
   需要单独一次配对检验才能下结论。
4. **`仅旧/NEW_CHUNK` 18 条边**是同一段正文重抽后没抽出来的。这是抽取非确定性，
   不是缺陷；但如果哪天这个数变大到十位数，就要怀疑是不是 n2 提示词/契约动了
   抽取的召回力——目前没有证据，只立个观察位。

### 15. 复现脚本正式化：从临时探针到 `scripts/k3_*.py`

上面 §14 的 12 条命令一开始指向 `_tmp_docs_k34/*.py` 与 `_tmp_k3_id_probe2.py`。
**临时脚本不入库，所以那批命令是死的**——换个 cwd、换台机器、或者哪天清一次目录就都跑不了。
本轮把它们提升成 9 个固定文件名的正式脚本（逻辑逐字保留，只改「怎么找到仓库根」）：

| 正式脚本 | 前身 | 行数 |
|---|---|---|
| `scripts/k3_attribution.py` | `attribution.py` + `edge_key_probe3/4/5.py` | 258 |
| `scripts/k3_probe_alias.py` | `alias_{scope,source,graph,index}_probe.py` 合并为四段 | 245 |
| `scripts/k3_probe_contract.py` | `contract_alias_probe.py` | 97 |
| `scripts/k3_probe_denominator.py` | `pair_denominator.py` + `mrr_denominator.py` 合并为两段 | 100 |
| `scripts/k3_probe_index.py` | `index_probe.py` | 100 |
| `scripts/k3_probe_chunk.py` | `chunk_probe.py` | 75 |
| `scripts/k3_probe_identity.py` | `_tmp_k3_id_probe2.py` | 59 |
| `scripts/k3_signif_eval.py` | `signif.py`（口径 B） | 115 |
| `scripts/k3_signif.py` | 上一轮已经提升过，本轮**未改一字** | 125 |

统一形态只有五处必要修正：`ROOT = Path(__file__).resolve().parent.parent`、
路径改基于 `ROOT`、`sys.stdout.reconfigure(encoding="utf-8")`、补 coding 声明与 docstring、
`def main() -> int:` + `raise SystemExit(main())`（0=判据成立 / 1=不成立）。**判据一条都没放宽。**

等价性怎么证明的（不是"看着一样"）：把新脚本的输出与 `_tmp_docs_k34` 里存档的旧输出逐条 diff——

```
k3_attribution.py   vs attribution.txt      前 75 行逐字一致，末尾新增 40 行（= §9 的边键三小节）
k3_signif_eval.py   vs signif.txt           完全一致
k3_probe_alias.py   段一/二/三/四            完全一致（7 / 67 / 8 / 15 行），只多段分隔行
k3_probe_identity  / contract / chunk / index / denominator    均完全一致
数字不一致清单：                              0 处
```

跨 cwd 也验了：从 `E:\` 用绝对路径调这 8 个脚本，输出与仓库根调用 **8 对全部 IDENTICAL**。

搬迁过程中唯一踩的坑是**结构性而非数字性**的：原 `attribution.py` 用模块级全局
`NEW_BLOCKS`，改成 `main()` 局部变量后 `split()` 抛 `NameError`。修法是给
`split(rows, by_id, new_blocks)` 加第三个参数、把 `main()` 里的局部值传进去——
**只搬作用域，判据与算法没动**，改完输出与旧基线逐字一致。

#### 顺带抓出的缺陷：脚本把中文写成 cp936 字节

提升过程中有个脚本的输出**按 UTF-8 解不开**：

```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb9 in position 100:
                    invalid start byte
```

`0xb9 0xb2` 按 gbk 解出来是「共」，整句是「共同题 320」。也就是说该脚本的中文
是按 Windows 控制台默认的 cp936 编码写出去的。**`py -3 x.py > out.txt` 之后那个文件不是
UTF-8**，粘进文档要乱码——而本节所有数字都是从这种输出里抄的。

补上 `sys.stdout.reconfigure(encoding="utf-8")` 后重跑：13 行中文正常，且
`data/k3_recall_pairs.json` 的 SHA-256 仍是 `A155984E…FC0D5`（**没变**，证明只是编码修复、
结果没动）。

顺手把范围扩大查了一遍，同一毛病还有 4 个已入库或在用的脚本：`k1_checks.py`、
`k2_hard_gates.py`、`k2_risk_register.py`、`k4_drill.py`。四个都补了同样的调用，
但**加 `hasattr(sys.stdout, "reconfigure")` 守卫**——`k1_checks.py` 会被
`tests/test_k1_checks.py` 用 `importlib` 加载，那时 `sys.stdout` 是 pytest 的捕获对象，
不该被重配置；`k1_checks` 的那一处放在 `main()` 里而不是模块顶层，理由同上。

补完实测 12 个脚本全部 UTF-8 可重定向（`k2_hard_gates.py` 仍是 8/8 硬门 PASS，
`k1_checks.py` 的 36 条用例仍全过）。

#### 两条死引用按实测改写

清理时发现文档里还有两处指向不存在的临时脚本。**没有简单删掉，而是改成能重跑的说法**：

1. `docs/k阶段实测.md` 的「量化（`_tmp_serving_gap` 的结论）」——注明那是一次性探针、
   未入库，**且它量的那个状态在缺陷修好后已经不存在**（候选世代现在自带 `chunks.jsonl`），
   所以那组数字无法重跑，只作缺陷当时的留档。
2. `docs/升级台账.md` 的「`_tmp_ready_cand.py` 对候选跑七项：`ready=False`，只有这一项 FAIL」
   ——同样注明探针未入库，并补上**现在**的实测读数：

```
server._compute_chunks_cover('g-20261103T132152Z-06a741') → ok=True
  detail: 索引 13798 块均在块正文中（…/06a741/chunks.jsonl，共 13798 条）
server._compute_chunks_cover('g-20261031T035231Z-4eafae') → ok=True
server._compute_chunks_cover('g-20261030T154606Z-55dfd7') → ok=True
  detail: 索引 13781 块均在块正文中（data/arkrag/chunks.jsonl，共 13786 条）
```

三代全是 `ok=True`——**这本身就是缺陷真的修好了的证据**：修之前候选那一代是
`FAIL 索引有 13798 块，块正文缺 17 块`。

#### 清理

删掉 22 项临时脚手架（`_tmp_docs_k34/` 59 个文件、`_tmp_k3_*.py` 四个、
`_verify/` 与 `_verify2/` 两个验证目录、各路 `_*.txt` 抓包输出），**删除前逐项核过
git 跟踪状态，被跟踪的一个都不碰**（实测跳过 0 项）。根目录现在没有任何 `_` 开头的残留。

## K4 实测：回滚与灰度演练

### 0. 口径

1. **候选** = `g-20261103T132152Z-06a741`（K2 产物，READY 未发布）；**上一代** =
   `g-20261031T035231Z-4eafae`；**线上** = `g-20261030T154606Z-55dfd7`（PUBLISHED，
   本节所有演练都不许碰它）。
2. **指针隔离**：演练把 `release.CURRENT_PATH` 指向 `data/arkrag/k4_probe/current.json`，
   `finally` 里还原并删掉。真实线上指针一次都不写。
3. **所有数字都是本次实跑读数**。§2 是 `py -3 scripts/k4_drill.py` 的整段原文。
4. **门数从 14 加到 15 是加严，不是放宽**：新增的第 15 门 `manifest_isolated`
   专门拦「演练把世代 Manifest 改坏」这件事（§8）。
5. 本轮**只动两处产品代码 + 两处测试**：`src/arkrag/release.py` 的 `prune_old()` 候选条件、
   `scripts/k4_drill.py` 的 Manifest 隔离；以及 `tests/test_release.py` 新增的两条用例、
   `tests/test_suite_hygiene.py` 的用例总数常量。其余全是只读演练（清单见 §11）。

### 1. 判据：门必须能失败，申报项不许写成必过项

K4 要回答的是「新世代能不能上、上了出问题能不能退回来」。这两件事都不能靠
「跑一遍没报错」来证明，所以 15 门里每一项都写成**可失败的断言**，而不是
「打印一串数字等人看」：

| 门 | 它拦什么 |
|---|---|
| `gray_candidate` / `gray_baseline` | 候选在固定探针上的可服务性（对照上一代） |
| `abstain_stable` | 换世代把某题的「该拒答 / 该回答」结论翻了 |
| `k4_precondition` | 回滚目标不存在或不完好 |
| `publish_prev` | 发布后 `previous_generation` 指向空或不完好的一代 |
| `rollback_pointer_only` | 回滚顺手重建了历史产物（只该切指针） |
| `rollback_pointer_shape` | 指针缺 `rolled_back` 键 |
| `rollback_probe` | 回滚后服务不出东西，或回答器世代标识对不上 |
| `rollback_reversible` | 回滚过去回不来 |
| `manifest_isolated` | 演练自己把世代 Manifest 改了 |
| `prod_untouched` | 演练动了线上指针 |
| `prod_servable` | 线上那一代能不能被当前链路服务（**申报项**） |
| `gold_smoke` | 候选在 40 题冒烟上出现硬失败 |
| `legacy_not_mixed` | Legacy JSON 与新模型静默混用 |
| `prune_safe` | 清理会删掉可服务世代 |

表里 14 行、15 门（第 1 行是 `gray_candidate` 与 `gray_baseline` 两门合写）。

**`prod_servable` 是申报项，不是必过项。** 它断言的是「申报项必须被如实报出来」，
所以 `可服务=False` 是它的正解，不是它的失败（§9 展开）。如果把它改成「必须可服务」，
这门就永远是红的——红久了就会被加进白名单，那才是最坏的结果。

### 2. 实跑结果（EXIT=0，15/15）

```
固定探针 18 题（类别 ['adversarial', 'alias', 'cross_chunk', 'direct_relation', 'multihop', 'negative_abstain', 'single_chunk', 'time_conflict']）

[PASS] 灰度查询（候选，不发指针）              题 18 / 答 17 / 拒 1 / 引用 94 条（定位失败 0）/ 单题 202.3 ms
[PASS] 灰度查询（上一代，对照）               题 18 / 答 17 / 拒 1 / 引用 88 条（定位失败 0）/ 单题 162.2 ms
[PASS] 拒答结论逐题稳定                   翻转 0 题（两代回答/拒答一致）
[PASS] 保留至少上一代完整有效 Generation     回滚目标 g-20261031T035231Z-4eafae 自检 0 问题；隔离指针起始 = g-20261031T035231Z-4eafae
[PASS] 发布后 previous_generation 指向有效上一代 指针 g-20261103T132152Z-06a741 / previous g-20261031T035231Z-4eafae（该代自检 0 问题）
[PASS] 回滚只切指针、不重建历史               回滚到 g-20261031T035231Z-4eafae 完成；12 份 Manifest 改动 0
[PASS] 回滚指针键集含 rolled_back        指针 = g-20261031T035231Z-4eafae / rolled_back = True
[PASS] 回滚后固定探针                    题 18 / 引用定位失败 0 / 回答器世代标识全对=True / 单题 162.5 ms
[PASS] 回滚可逆（再切回候选）                指针 = g-20261103T132152Z-06a741 / previous = g-20261031T035231Z-4eafae
[PASS] 演练未改坏任何世代 Manifest（隔离生效）   备份 12 份 / 还原 1 份：['g-20261103T132152Z-06a741']
[PASS] 线上指针未被演练改动                 线上仍是 g-20261030T154606Z-55dfd7（PUBLISHED）
[PASS] 线上世代可被当前链路服务（申报项）          g-20261030T154606Z-55dfd7 可服务=False：世代 g-20261030T154606Z-55dfd7 的 graph_schema=ark-rag-1，本工具只支持 ark-rag-3（ark-rag-1 没有断言，做不了带方向的扩展与可定位引用） → 回滚目标只能用 g-20261031T035231Z-4eafae
[PASS] Gold Set 冒烟（40 题，候选）       R@20 0.7436 / R@5 0.5897 / MRR 0.4591 / 引用定位 1.0000 / 硬失败 0
[PASS] Legacy JSON 未与新模型静默混用      graph_schema=ark-rag-3（需 ark-rag-3）/ schema_version=ark-rag-1 / entity_rule=— normalizer=n2
[PASS] 删除旧产物前只清 FAILED、且备份已验证     dry-run 拟清 0 个：[]；其中非 FAILED 0

K4 检查：15/15 通过
结论：K4 灰度验收与回滚演练全过
报告已写 data/k4_drill.json
```

第 12 个 `[PASS]` 行 `prod_servable` 显示 `[PASS]` 而 detail 里是 `可服务=False`——这不是矛盾：
这一门断言的是「申报项必须被如实报出来」，不是「必须可服务」。§9 说清。

注意第 10 个 `[PASS]` 行：**备份 12 份 / 还原 1 份**。还原的那 1 份正是 §8 说的污染，
它说明隔离器不是空转的——它真的救回了一份被改坏的 Manifest。

上面这段是其中一次运行的原文（`data/k4_drill.json` 会被每次运行覆盖，**现在盘上那份是
更晚一次运行的结果**：15 项判定与 detail 文字逐字相同，只有三处耗时变成
`193.7 / 151.5 / 142.4 ms`）。判据是 15 个 `ok` 布尔值，它们每次都必须一致；
耗时每次都不一样，别拿它做比较（§3 末段说清为什么）。

### 3. 灰度：候选能服务，拒答结论逐题稳定

| 项 | 候选 `06a741` | 上一代 `4eafae` |
|---|---|---|
| 题数 | 18 | 18 |
| 回答 / 拒答 | 17 / 1 | 17 / 1 |
| 引用条数 | 94（定位失败 **0**） | 88（定位失败 **0**） |
| 单题耗时 | 202.3 ms | 162.2 ms |

**拒答结论逐题稳定：翻转 0 题。** 这是灰度最重要的一条——换世代把某题从「该答」
变成「该拒」，或反过来，是用户能直接感知到的行为变化。18 题里 0 翻转。

**耗时不是可复现数字。** 同一个脚本我跑过多轮，单题耗时（候选 / 上一代）实测到过
`76.0 / 65.2`、`78.0 / 63.1`、`177.0 / 133.1`、`193.7 / 151.5`、`202.3 / 162.2` ms，
候选**始终**比上一代慢，但差距在 **15–44 ms/题**之间浮动。所以只能说「候选较上一代慢，
量级几十毫秒/题」，不能写「慢 40 ms」——那个数换一台机器、换一次负载就变。慢的原因
没有查，它不影响 §1 的判据（灰度只看能不能服务、结论稳不稳），登记在 §13 遗留。

### 4. 回滚只切指针：12 份 Manifest 改动 0

```
[PASS] 回滚只切指针、不重建历史               回滚到 g-20261031T035231Z-4eafae 完成；12 份 Manifest 改动 0
[PASS] 回滚指针键集含 rolled_back        指针 = g-20261031T035231Z-4eafae / rolled_back = True
[PASS] 回滚可逆（再切回候选）                指针 = g-20261103T132152Z-06a741 / previous = g-20261031T035231Z-4eafae
```

`12 份 Manifest 改动 0` 是这一节的核心证据。盘上 12 个世代，回滚前后逐份比 Manifest
字节，**没有一份变过**。这证明「回滚 = 改一个指针文件」，而不是「重新构建上一代」。
后者在真事故里根本来不及——重建一代要几分钟，而且重建出来的东西和原来那份未必逐字节相同。

`publish → rollback → 再 rollback` 三段都通，且最后能切回候选，说明指针操作可逆。

### 5. 缺陷：`prune_old()` 的候选条件会删掉可服务世代

原实现（`src/arkrag/release.py`）候选条件是 `g["status"] != PUBLISHED`：

```python
def prune_old(*, keep: int = 2, dry_run: bool = True) -> list[str]:
    """归档旧的 FAILED/非发布世代目录。默认只演练不删除。
    ...
    """
    cur = read_current() or {}
    keep_ids = {cur.get("generation_id", ""), cur.get("previous_generation", "")}
    candidates = [
        g for g in list_generations() if g["generation_id"] not in keep_ids and g["status"] != PUBLISHED
    ]
    candidates.sort(key=lambda g: g.get("created_at", ""), reverse=True)
    victims = [g["generation_id"] for g in candidates[keep:]]
```

字面读像是「保留最近 N 个非发布世代」，实际效果是**把「保留集合」当成了
「候选集合的过滤器」**：`keep_ids` 只有 `current` 与 `previous` 两个 id，
而**上一代可服务世代未必落在这两个 id 里**。现场就是这样：
`current=55dfd7`、`previous=606ea9`，真正的上一代 `4eafae` 两个都不是。

按旧规则在当前盘面复算（12 个世代，只算不删）：

```
当前盘面世代 12 个
指针 current=g-20261030T154606Z-55dfd7  previous=g-20261030T114823Z-606ea9

=========== keep=1 ===========
  旧规则 !PUBLISHED：候选 9 / 拟清 8 / 其中非 FAILED 6
    拟清 ['g-20261103T124939Z-27be38', 'g-20261103T124101Z-fa22ae', 'g-20261031T035231Z-4eafae', 'g-20261031T034009Z-348206', 'g-20261031T033200Z-b22ca8', 'g-20261031T032914Z-1d1b08', 'g-20261030T150655Z-65a5c4', 'g-20261030T150101Z-e0aacc']
    非 FAILED ['g-20261103T124939Z-27be38', 'g-20261031T035231Z-4eafae', 'g-20261031T034009Z-348206', 'g-20261031T033200Z-b22ca8', 'g-20261030T150655Z-65a5c4', 'g-20261030T150101Z-e0aacc']
    含上一代 4eafae = True
  新规则 ==FAILED（实调 prune_old）：候选 2 / 拟清 1
    拟清 ['g-20261031T032914Z-1d1b08']
    其中非 FAILED []

=========== keep=2 ===========
  旧规则 !PUBLISHED：候选 9 / 拟清 7 / 其中非 FAILED 5
    拟清 ['g-20261103T124101Z-fa22ae', 'g-20261031T035231Z-4eafae', 'g-20261031T034009Z-348206', 'g-20261031T033200Z-b22ca8', 'g-20261031T032914Z-1d1b08', 'g-20261030T150655Z-65a5c4', 'g-20261030T150101Z-e0aacc']
    非 FAILED ['g-20261031T035231Z-4eafae', 'g-20261031T034009Z-348206', 'g-20261031T033200Z-b22ca8', 'g-20261030T150655Z-65a5c4', 'g-20261030T150101Z-e0aacc']
    含上一代 4eafae = True
  新规则 ==FAILED（实调 prune_old）：候选 2 / 拟清 0
    拟清 []
    其中非 FAILED []

全部 FAILED 世代：['g-20261031T032914Z-1d1b08', 'g-20261103T124101Z-fa22ae']
全部 READY  世代：['g-20261030T150101Z-e0aacc', 'g-20261030T150655Z-65a5c4', 'g-20261031T033200Z-b22ca8', 'g-20261031T034009Z-348206', 'g-20261031T035231Z-4eafae', 'g-20261103T124939Z-27be38', 'g-20261103T132152Z-06a741']
```

**默认 `keep=2` 会拟清 7 个，其中 5 个不是 FAILED，含上一代 `4eafae`。**
`keep=1` 更糟：拟清 8 个，6 个非 FAILED。

计划 K4 的原话是「保留至少上一代完整有效 Generation」。按旧规则真跑一次，
上一代就没了——而上一代正是回滚目标。**这个缺陷的后果不是「多删几个旧目录」，
是「回滚能力被清理动作自己删掉」。**

修法：候选条件改成 `g["status"] == FAILED`（`src/arkrag/release.py:862`）。
理由是 READY 是**可服务但未发布**的合法状态，它存在的意义就是随时能切上线；
能清的只有 FAILED——那些已经跑不出产物、留着只为留证。

修复后实测：`keep=1` 拟清 1 个（`1d1b08`，非 FAILED 0 个），`keep=2` 拟清 0 个
（盘上只有 2 个 FAILED，keep=2 全留下）。

### 6. 原有用例为什么发现不了：它在空目录里断言「没删东西」

`tests/test_release.py` 原有用例 `test_prune_is_dry_run_by_default`：

```python
def test_prune_is_dry_run_by_default(self, sandbox):
    release.prune_old()
    assert len(list(release.GENERATIONS_DIR.iterdir())) == 0
```

`sandbox` 是空目录。**空目录里任何候选条件都返回空集**，于是「拟清 0 个」恒成立，
用例恒绿。它测的是「dry_run 不删文件」，不是「只删该删的」。

这类用例最危险的地方在于**它看起来在保护这个函数**：函数名、用例名、断言方向
（`== 0`）都对，唯独没有构造出「有可服务世代、也有 FAILED 世代」这个真实盘面。
一个永远在空集上成立的断言，和没有断言的区别只在于它会让人以为有人管这块。

### 7. 新增测试，以及反向实测它们确实能红

`tests/test_release.py` 补了两样东西：

- `:613 _seed_three_kinds()`：造出 PUBLISHED / READY / FAILED 四代，`created_at`
  显式递增，产物用 `_build_generation()` 造得能通过 `verify_generation`。
- `:652 class TestPruneOnlyTouchesFailed`，两条：
  `:670 test_dry_run_lists_only_failed`（`keep=1` 断言名单 ⊆ 两条 FAILED，
  且恰好 `== [failed_old]`）、`:681 test_real_prune_keeps_ready_and_published`
  （`dry_run=False` 后 READY 与 PUBLISHED 目录仍在、最近 1 个 FAILED 保留、
  最旧 FAILED 被删）。

修复后两条全绿：

```
2 passed, 54 deselected, 1 warning in 4.61s
```

只有绿是不够的——**一条永远绿的用例和没有用例没区别**。所以把 `release.py:862` 的
`== FAILED` 临时改回 `!= PUBLISHED`，再跑一次（改完按 SHA-256 校验还原：

`BEFORE_SHA=49033212D23C358F3F560085E159C0C0C088381538DE22904C108E089F44B1F1`
→ `AFTER_SHA=49033212D23C358F3F560085E159C0C0C088381538DE22904C108E089F44B1F1`，
`RESTORED=True`）：

```
tests\test_release.py:675: in test_dry_run_lists_only_failed
    assert set(victims) <= {gens["failed_new"].generation_id, gens["failed_old"].generation_id}, (
E   AssertionError: dry-run 名单里出现了非 FAILED 世代：['g-20260103T000000Z-cc0003', 'g-20260102T000000Z-bb0002']
E   assert {'g-20260102T...0000Z-cc0003'} <= {'g-20260103T...0000Z-dd0004'}
E
E     Extra items in the left set:
E     'g-20260102T000000Z-bb0002'
```

```
FAILED tests/test_release.py::TestPruneOnlyTouchesFailed::test_dry_run_lists_only_failed
FAILED tests/test_release.py::TestPruneOnlyTouchesFailed::test_real_prune_keeps_ready_and_published
2 failed, 54 deselected, 1 warning in 6.23s
```

报错里那个多出来的 `bb0002` 正是被误判成候选的 READY 世代——**用例红的原因是缺陷本身，
不是断言写歪了**。这一条比「修复后全绿」重要：能红才证明它拦的是这个问题。

卫生门的计数随之 `tests/test_suite_hygiene.py` 的 `N`：1289 → **1291**。

### 8. 演练本身暴露的缺陷：`publish()` 会改写目标世代的 Manifest

演练第一次跑完，`git status` 里多了一条：

```
 M data/arkrag/generations/g-20261103T132152Z-06a741/manifest.json
```

查下来：`release.publish()` 不只写指针，它还会**改写目标世代的 Manifest**
（`status` READY → PUBLISHED，并加 `published_at`）。而演练只隔离了
`release.CURRENT_PATH`（指针），没隔离 Manifest。于是：

- `06a741/manifest.json` 的 `status` 变成 `PUBLISHED`、多出 `published_at: 2026-11-03T15:35:50Z`；
- 而发布指针从头到尾是 `55dfd7`。

**候选其实从未真正上线。** 这就是「演练在备份盘上制造了一份与事实不符的历史」——
既不是线上状态，也不是演练前状态。真出事时拿这份 Manifest 去判断「这一代上线过没有」，
答案会是「上线过」，而这是假的。

修法：`scripts/k4_drill.py` 新增 `class _ManifestGuard`：

- `__enter__`：把 `GENERATIONS_DIR/*/manifest.json` 的**原始 bytes** 全部备份
  （读失败记进 `failed` 列表，不假装备份成功）；
- `__exit__`：逐份比对，**只有真变了才写回**，并 `return False`（不吃异常）；
- `summary()`：输出「备份 12 份 / 还原 1 份」。

两个「为什么不」要写下来，因为它们看起来更简单：

- **不「改回 READY」**：再走一次 `update()` 会改 `updated_at`，Manifest 仍不是
  演练前那份字节，`git status` 照样脏。「恢复」必须是字节级恢复，不是状态级恢复。
- **不「整脚本搬到临时目录跑」**：`release.publish()` 要读候选的图与索引产物
  （`artifact_digests`），整代复制等于复制数 GB 产物，而且副本一致性还要另外证明。
  隔离该隔离的东西，比整体搬家更小、更容易验证。

这条缺陷同时升级成了第 15 门 `manifest_isolated`：演练结束逐份比 Manifest，
有一份变过就红。**演练自己也要被演练**——否则「演练通过」这件事本身没有可信度。

### 9. 申报项：线上那一代不可被当前链路服务

```
[PASS] 线上世代可被当前链路服务（申报项）  g-20261030T154606Z-55dfd7 可服务=False：
  世代 g-20261030T154606Z-55dfd7 的 graph_schema=ark-rag-1，本工具只支持 ark-rag-3
  （ark-rag-1 没有断言，做不了带方向的扩展与可定位引用）
  → 回滚目标只能用 g-20261031T035231Z-4eafae
```

线上 `55dfd7` 是 `ark-rag-1`：没有断言，做不了带方向的扩展与可定位引用。当前链路
（`pipeline.load_generation()`）读它会直接 `SystemExit`。

**这意味着「回滚到上一代」今天只能回到 `4eafae`。** 这一条必须写成申报项而不是必过项，
理由有两条：

1. 它不是本轮引入的问题——`55dfd7` 是 K 阶段之前就在线上的世代，它的 schema 由当时
   的工具链决定，改不了；
2. 把它写成必过项，这门就永远是红的。**永远是红的门等于没有门**：第一次看到会停下来查，
   第十次看到就学会跳过了。红门只有在「红了就说明有新问题」时才有用。

所以这门断言的是「申报项被如实报出来」，`可服务=False` 是它的正解，不是它的失败。
真正的行动项挂到 §13 遗留：`55dfd7` 的处置（下线 / 补断言 / 保持只读）要在 K5 定。

### 10. Legacy 未混用与 Gold 冒烟

```
[PASS] Legacy JSON 未与新模型静默混用      graph_schema=ark-rag-3（需 ark-rag-3）/ schema_version=ark-rag-1 / entity_rule=— normalizer=n2
[PASS] Gold Set 冒烟（40 题，候选）       R@20 0.7436 / R@5 0.5897 / MRR 0.4591 / 引用定位 1.0000 / 硬失败 0
```

`legacy_not_mixed` 查的是候选 Manifest 声明的那组身份值：`graph_schema=ark-rag-3`
与当前代码要求一致，`normalizer=n2`、`prompt v2`、`vocab_sha` 都对得上，
`entity_rule` 为空（这一代没启实体规则）。

这一门防的是「新旧两套模型混着用还看不出来」：如果 Manifest 写着 `ark-rag-1` 而产物
是 `ark-rag-3` 的形状，读的人会按老 schema 解释新产物，出错时找不到原因。

Gold 冒烟 40 题的 R@20 0.7436 / 硬失败 0，是**冒烟**不是验收：40 题抽自题库，
每类 5 题，用来确保候选不是「能加载但答不出」，不能拿来和 320 题的 K3 数字比较
（同一代 320 题是 R@20 0.6989，两者分母不同，混着引用会造出假的「提升 0.045」）。

### 11. 新增与改动文件

| 文件 | 变化 |
|---|---|
| `src/arkrag/release.py` | `prune_old()` 候选条件 `!= PUBLISHED` → `== FAILED`；docstring 扩写根因 |
| `scripts/k4_drill.py` | 新增（15 门演练）；`class _ManifestGuard` Manifest 字节级隔离；`manifest_isolated` 门 |
| `tests/test_release.py` | +`_seed_three_kinds()`、+`class TestPruneOnlyTouchesFailed`（2 用例） |
| `tests/test_suite_hygiene.py` | `N` 1289 → **1291** |
| `data/k4_drill.json` | 新增（15 门结果，`all_passed: true`） |

### 12. 复现命令

```bash
# 演练（会隔离指针与 Manifest，跑完自动还原）
py -3 scripts/k4_drill.py

# 清理规则：只算不删
py -3 -c "from arkrag import release; print(release.prune_old(keep=1, dry_run=True))"

# 清理规则的单测（含「能失败」的两条）
py -3 -m pytest tests/test_release.py -q

# 演练结果落盘内容逐项看
py -3 -c "import json;d=json.load(open('data/k4_drill.json',encoding='utf-8'));[print(c['id'],c['ok']) for c in d['checks']]"
```

### 13. 遗留

1. **`55dfd7`（ark-rag-1）不可被当前链路服务，回滚目标只能是 `4eafae`。**
   已在 §9 立为申报项。K5 要定 `55dfd7` 的处置：下线、就地冻结只读、还是补一份
   断言桥接层。**在它解决之前，「回滚到线上上一代」这句话在本项目里是不成立的。**
2. **候选比上一代慢几十毫秒/题，原因未查。** 多轮实测在 15–44 ms/题之间浮动，
   不可复现到具体数。不影响本轮判据，但如果 K5 要算吞吐，得先做一次固定负载下的对比。
3. **`prune_old()` 的 `keep` 语义仍有解释空间。** 现在是「保留最近 N 个 FAILED」。
   FAILED 世代留着只为留证，长期看会一直长——真正该做的是「留最近 N 个 + 给一个
   `--all` 显式清全部留证」，但那是运维习惯问题，不在 K4 验收面内。
4. **本次演练的 Manifest 隔离靠「整目录备份 + 逐份比对还原」**，世代变多时
   `__enter__` 的耗时会线性涨（12 份现在可忽略）。若哪天世代到三位数，改成
   「只备份与演练相关的那几代」更合适。
