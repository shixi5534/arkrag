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

## K 阶段补做：对照计划查出的四处漏项

### 0. 为什么会有这一节

K1–K4 的代码与演练在 `c33719c` 提交之后，拿 `docs/升级计划_GraphRAG架构.md` 的 K 阶段条目
**逐句**核对，查出四处**只有计划文字、代码里没有任何判定**的地方。计划原文（行号取自该文件
`Select-String` 的实测输出）：

| 计划行 | 原文 | 属于 |
|---|---|---|
| 424 | `→ 新 Assertion Graph → 新索引 → 全量硬门 → Gold Set 评测 → 性能评测` | K2 构建顺序 |
| 425 | `→ 候选 READY → 灰度查询 → 正式发布` | K2 构建顺序 |
| 436 | `- 超阈值必须人工审批` | K3 五条判据的第 5 条 |
| 443 | `- 新版稳定运行约定观察期后才归档旧版` | K4 六条要求的第 4 条 |

「只有文字」不是印象，是数出来的。**在本节追加之前**，两份文档里这些关键词的出现次数
（`[IO.File]::ReadAllText` + `[regex]::Matches` 逐词计数）：

| 关键词 | `docs/k阶段实测.md` | `docs/升级台账.md` |
|---|---|---|
| 性能 | 0 | 2 |
| bench | 0 | 9 |
| P95 | 0 | 25 |
| 人工审批 | 0 | 0 |
| 观察期 | 0 | 0 |
| 正式发布 | 0 | 0 |
| 灰度查询 | 2 | 0 |

这张表里有两处**不能照字面读**，必须说清，否则它会反过来骗人：

1. **`k阶段实测.md` 里 `灰度查询` 不是 0，是 2。** 那 2 处在 1246/1247 行，是 K4 §2 探针
   输出的两行标签（`[PASS] 灰度查询（候选，不发指针）`），指「拿固定题集问候选、不发指针」，
   **不是** K2 计划里的「灰度查询 → 正式发布」这一整个环节。同名不同物。
2. **台账里 `性能` 2 处、`bench` 9 处、`P95` 25 处，全部来自 H 阶段**（`scripts/bench_h1.py` /
   `bench_h3.py` 的索引层基线，见台账「## H 阶段」）。那是「单个索引后端快不快」的尺子，
   **不是** K2 要求的「整代构建完成后候选 vs 上一代」的性能评测——后者在 K 阶段一次都没量过。
   所以 `k阶段实测.md` 里 `性能`/`bench`/`P95` 三个 0 才是这条漏项的真凭据。

四处漏项的共同点：**它们都是「流程节点」，不是「代码路径」**。写计划时它们是一串箭头里的
一个词，落地时最容易被当成「反正会有人做」。而一条从未被执行过一次的判据，等于没有——
K3 那处尤其典型：本轮实测恰好「略降但不显著」，**从未触发过阈值**，所以「超阈值要审批」
这条规则在补做之前既没定义过阈值、也没定义过审批记录长什么样。

本轮给每处各补一道**能失败的门**（不是补一段说明文字）。下面 4 节逐处说清「门判什么」与
「本次实测读数」，§7–§9 是文件清单、复现命令与遗留。

### 1. K2 性能评测：门只放确定性事实

#### 1.1 口径

`scripts/k2_perf.py`，默认 40 题、热查重复 2 次、`top_k=20`、`graph_depth=1`。
每代各起 **2 个全新子进程**（一个是整代测量、一个是索引路径证明）——冷启动成本与峰值 RSS
只有在全新进程里才是真的；同一进程先后载入两代，第二代踩第一代的页缓存，量出来的是
「我们的测量方法」而不是「这一代有多重」。

关键设计：**门分成两类，只有一类算数**。

| 类别 | 放什么 | 是否进 `all_passed` |
|---|---|---|
| 硬门 `checks`（7 项：`generations_ok` / `index_size_delta` / `cold_load_path` / `index_binding` / `rss_ceiling` / `report_fields` / `p95_backstop`） | 索引与加载字节、探针 digest、冷加载走的函数、峰值 RSS 上界、报告字段完整性，外加唯一碰时序的兜底门 `p95_backstop`（阈值 2×，只为抓「慢了十倍」这种灾难） | **是** |
| 申报项 `declared`（22 项） | 冷/热 P50/P95/P99、加载耗时、构建耗时 | **否**，字段里写死 `enforced=False` |

`p95_backstop` 是 7 硬门里**唯一**碰时序的那一条，所以门数是 7 而不是 6——它虽在硬门里，
阈值取得极宽（2×），作用只是保险丝，不是性能判据（§1.4 末段）。

理由是这条脚本最要紧的一条纪律：**单题耗时在本机不可复现到具体值**。同一份代码、同一批题，
重跑一次 P95 就能挪几十毫秒（`docs/k阶段实测.md` K4 §3 已记 15–44 ms/题的浮动）。把「候选
不许比上一代慢 5%」做成门，等于让门随机变红；随机红的门活不过两周，它会被顺手放宽，
然后连「慢了十倍」也拦不住。所以时序只申报，兜底门取 2×，只为抓灾难性回退。

#### 1.2 实跑结果（`py -3 scripts/k2_perf.py`，EXIT=0）

```
K2 性能评测（候选 vs 上一代）  Python 3.14.3  win32
候选 g-20261103T132152Z-06a741
上一代 g-20261031T035231Z-4eafae
题量 40  热查重复 2  top_k 20  graph_depth 1
------------------------------------------------------------------------------
确定性事实（硬门，条件不成立必红）
[PASS] 两代可加载且产物齐备             候选 g-20261103T132152Z-06a741 schema=ark-rag-3 status=READY / 上一代 g-20261031T035231Z-4eafae schema=ark-rag-3 status=READY / 5 件产物均 > 0 字节：True
[PASS] 索引字节与成因核对              lexical.json 候选 106,685,068 B → 上一代 127,244,898 B，缩小 16.16%（n=2 代）；倒排词条 390,508 ← 694,304（43.76% 更少，成因 t1→t2）；文档数 13798 ← 13781（+17 块，文件反而更小）；每千词条字节 候选 0.273 KiB vs 上一代 0.183 KiB
[PASS] 冷加载只走 load_index       audit hook 实测：06a741 打开 1 份 JSON / 106,685,068 B（lexical.json 106,685,068 B）；4eafae 打开 1 份 JSON / 127,244,898 B（lexical.json 127,244,898 B）；graph.json 打开次数 候选 0 / 上一代 0（n=2 代 × 1 次冷加载）
[PASS] 索引内容绑定 Manifest        probe_digest 重算一致：候选 b800a2a2b2e273b0、上一代 46e600c1f3a9ec9e（n=2 代 × 10 条固定探针 top-10）
[PASS] 峰值 RSS 上界              峰值 RSS 候选 1563.63 MiB / 上一代 1719.73 MiB（比 0.909×，n=2 进程）；上界 4096.0 MiB、相对上限 1.5×
[PASS] 每代报告字段完整性              候选 13 字段齐 + 冷/热各 4 项分位齐（n=40 题）/ 上一代同（n=40 题）
[PASS] 热查 P95 兜底（2×）          热查 P95 候选 130.687 ms / 上一代 89.402 ms = 1.462×（阈值 2.0×，n=候选 40 题 / 上一代 40 题）
------------------------------------------------------------------------------
申报项（时序数字，只记录不判红，不进 all_passed）
  候选   冷查 n=40 P50 73.715 P95 138.987 P99 165.89 min 33.992 max 165.89
  候选   热查 n=40 P50 72.658 P95 130.687 P99 157.719 min 34.303 max 157.719
  候选   加载 12216.52 ms（含五件产物）  管线构造 196.11 ms
  候选   RSS 起始 26.69 MiB  加载后 1359.62 MiB  就绪 1375.5 MiB  峰值 1563.63 MiB
  候选   题量 n=40  类别 adversarial=3  alias=6  cross_chunk=4  direct_relation=6  multihop=6  negative_abstain=4  single_chunk=7  time_conflict=4
  候选   构建 run r-20261103T132055Z-fbdfeb 总跨度 196.0 s （11 条事件，取自事件日志不重跑）
  候选   索引路径 data/arkrag/generations/g-20261103T132152Z-06a741/chunks.jsonl
  上一代 冷查 n=40 P50 51.726 P95 93.795 P99 192.51 min 26.314 max 192.51
  上一代 热查 n=40 P50 48.916 P95 89.402 P99 97.882 min 26.645 max 97.882
  上一代 加载 7757.55 ms（含五件产物）  管线构造 96.72 ms
  上一代 RSS 起始 26.8 MiB  加载后 1502.34 MiB  就绪 1508.13 MiB  峰值 1719.73 MiB
  上一代 题量 n=40  类别 adversarial=3  alias=6  cross_chunk=4  direct_relation=6  multihop=6  negative_abstain=4  single_chunk=7  time_conflict=4
  上一代 构建 run r-20261031T035149Z-d7c94b 总跨度 176.0 s （10 条事件，取自事件日志不重跑）
  上一代 索引路径 data/arkrag/chunks.jsonl
  申报项合计 22 项，全部 enforced=False
  LLM token/成本：不可测（本机无 LLM 端点，抽取式回答器），一律 null
------------------------------------------------------------------------------
硬门 7/7 通过（申报项 22 项不参与判定）
报告已写 data/k2_perf.json
```

#### 1.3 确定性事实：这四组数可以当结论

| 事实 | 候选 `06a741` | 上一代 `4eafae` | 关系 |
|---|---|---|---|
| `lexical.json` 字节 | 106,685,068 B | 127,244,898 B | 缩小 **16.16%** |
| 倒排词条 | 390,508 | 694,304 | 少 **43.76%**（成因 `t1→t2`） |
| 文档数 | 13798 | 13781 | +17 块 |
| 每千词条字节 | 0.273 KiB | 0.183 KiB | 词条单位更贵、总量更省 |
| 峰值 RSS | 1563.63 MiB | 1719.73 MiB | **0.909×** |
| 冷加载打开的 JSON | 1 份（只有 `lexical.json`） | 1 份（同） | `graph.json` 打开 **0** 次 |
| 探针 digest | `b800a2a2b2e273b0` | `46e600c1f3a9ec9e` | 各代自洽 |

这四组为什么能当结论：**它们不随机器忙不忙变**。`127,244,898 B → 106,685,068 B` 换个时间
再跑还是这两个数；`graph.json 打开 0 次` 是 audit hook 抓到的真实 `open()` 调用，不是推断；
探针 digest 是把 10 条固定探针的 top-10 重算一遍比对，索引被换掉一位数就会红。

这三组数字与 K3 §12 的读数**逐个对得上**（同两个世代、同两份文件）：`127,244,898 B` 与
`106,685,068 B`、`694,304` 与 `390,508` 词项、`13781` 与 `13798` 文档，与 K3 §12
「=== 倒排规模 ===」那四行逐字相同。**这一致性是跨脚本的交叉验证**：两个脚本独立读同一份
`lexical.json`，各算各的，得到的字节数与词项数完全相同。

但 **`−16.16%`（字节）与 `−43.76%`（词项）不是同一件事的两种说法**，是两个不同量纲，
K3 §12 已论证：同一批正文在 `t1/t2` 下的 token 数只差 **7.29%**（`t2/t1 = 0.9271`），
词项多掉的那 36 个百分点来自 `t2` 的两处口径变化（专名整词入表、专名边界不连双字）。
所以本节只能说「**字节**缩小 16.16%」，不能说「索引瘦身 16.16%」，更不能把 43.76% 混进来。

`文档 +17 块而文件更小` 不是矛盾，是 `tokenizer_version t1→t2` 的必然：词条少了 43.76%，
多 17 个文档的增量远不足以抵掉词表收缩。**这条必须连 `t1→t2` 一起写**，否则「词条少了
43.76%」会被误读成语料缩水。

#### 1.4 时序读数是「本次读数」，不是性能结论——本次方向还与早先一次相反

这是本节最必须说清的一条。**下面这些数字只能当「我跑这一次看到了什么」，不能当结论**：

| 项 | 候选本次 | 上一代本次 |
|---|---|---|
| 冷查 P50 / P95 | 73.715 / 138.987 ms | 51.726 / 93.795 ms |
| 热查 P50 / P95 | 72.658 / 130.687 ms | 48.916 / 89.402 ms |
| 加载（含五件产物） | 12216.52 ms | 7757.55 ms |
| 管线构造 | 196.11 ms | 96.72 ms |

**本次候选在热查 P95 上比上一代慢 1.462×，在冷查与加载上也慢。** 这与早先一次在干净环境
（无并发）下的读数**方向相反**——据 Lead 转述，那次读到候选热查 P95 127.403 ms / 上一代
223.186 ms = `0.571×`，是候选明显更快。两次的差别不在代码，在**当时机器上有别的进程在抢
CPU 与内存**。

> **这三个数的出处要说清**：`127.403` / `223.186` / `0.571×` 是 Lead 转述的「早先一次干净
> 环境读数」，**不是本节任何一条命令的输出**，仓库里也没有留档（`scripts/` / `tests/` /
> `docs/` / `data/` 全文搜这三个数，只搜到本节与台账里对它们的转述本身）。本节自己实跑、
> 能指回 `py -3 scripts/k2_perf.py` 输出的只有 `130.687` / `89.402` / `1.462×`。
> 留着旧读数只为说明「方向上会翻」，**不能拿它当性能证据**。

同一批题、同一份代码，候选热查 P95 曾被观测到从 167.671 ms 跳到 544.19 ms（3.2 倍），
只因当时有别的进程在跑。所以：

* **「候选热查比上一代快多少」这个问题，本轮的读数回答不了。** 要回答它，必须在无并发时
  重跑，并且把「当时机器负载」一起记下来（`k2_gray.py` 就是这么做的——它先读
  `GlobalMemoryStatusEx` 拿到「可用 16232 MiB / 总 32543 MiB / 负载 50%」再决定模式）。
* **兜底门 `p95_backstop` 仍然有意义**：阈值 2×，本次 1.462× 过了。它的用处不是判断
  「快慢」，是判断「有没有慢到十倍」。若将来候选真慢了十倍，1.462× 这种量级的噪声淹不掉它。
* **别把申报项当门读**。报告里 22 项全部 `enforced=False`，这句是写给未来的自己和读者看的：
  它们进不了 `all_passed`，所以「硬门 7/7 通过」**不代表**「性能没退化」，只代表
  「7 条确定性事实成立」。

### 2. K2 灰度查询：两个真 HTTP 实例，逐题交替问

#### 2.1 口径

`scripts/k2_gray.py` 起**两个真的 `serve.py` 实例**（`--generation … --port 0`，端口由系统分配），
**逐题交替**打两代（不是先把一代问完再问另一代——交替才能让两代吃到同一时刻的机器状态）。
它**只按 `--generation` 显式指定世代启动，不写发布指针，不调 `/admin/rollback`**。

「灰度」在这里的含义必须精确：**它是「候选可以在真 HTTP 上服务、且行为与基线逐题一致」**，
不是「已经对部分流量放量」。放量需要改指针或网关，本轮没做（§9 遗留 ④）。

#### 2.2 实跑结果（`py -3 scripts/k2_gray.py`，EXIT=0）

```
K2 灰度查询：两代真实 HTTP 实例，逐题比对
[i] 候选 g-20261103T132152Z-06a741
[i] 基线 g-20261031T035231Z-4eafae
[i] 只按 --generation 显式指定世代启动；不写发布指针，不调 /admin/rollback
[i] 可用物理内存 16232 MiB（总 32543 MiB，负载 50%） 来源 GlobalMemoryStatusEx
[i] 模式 concurrent（门槛 6144 MiB：两个实例同时活着）
[i] 起服务（serve.py --generation … --port 0，模式 concurrent）
[i] candidate g-20261103T132152Z-06a741 端口 56650 启动 401 ms / 就绪 19777 ms / 阻断项不符 0
[i] baseline g-20261031T035231Z-4eafae 端口 56652 启动 401 ms / 就绪 21053 ms / 阻断项不符 0
[i] candidate 热身（管道构建 g-20261103T132152Z-06a741）：8130 ms
[i] baseline 热身（管道构建 g-20261031T035231Z-4eafae）：8124 ms
[PASS] 探针题数固定               固定探针 20 题（期望 20），题集一致且全部在 Gold Set 中
[PASS] 两代探针皆 200            两代共 40 次探针请求全部 200
[PASS] 世代标识三处核对             candidate 三处标识均 = g-20261103T132152Z-06a741；baseline 三处标识均 = g-20261031T035231Z-4eafae
[PASS] 拒答决策不翻转              翻转 0 题（20 题两代一致）；另：Gold Set 标注应拒答的题被答了（两代相同，非灰度回归） 候选 ['q-na-000013'] / 基线 ['q-na-000013']
[PASS] 引用可定位                引用 184 条全部可在本题命中块上定位（越界/对不上 0，引到命中块外 0）

灰度判据：5/5 通过
结论：候选 g-20261103T132152Z-06a741 与基线 g-20261031T035231Z-4eafae 在同一批 20 题上无拒答翻转、引用全部可定位、世代标识正确（模式 concurrent）
报告已写 data/k2_gray.json
```

#### 2.3 五条判据各拦什么

| 判据 | 它拦什么 | 本次读数 |
|---|---|---|
| `probe_set_fixed` | 题集被悄悄换掉、或题不在 Gold Set 里 | 固定探针 20 题（期望 20） |
| `probes_answered` | 候选起得来但服务不通（4xx/5xx/超时） | 两代共 40 次请求**全部 200** |
| `generation_identity` | 服务自称的世代与实际载入的不是同一代（静默回落到旧版） | 三处标识全对，两代各三处 |
| `abstain_flip` | 换世代把某题「该答/该拒」结论翻了 | **翻转 0 题**（20 题两代一致） |
| `citation_locate` | 引用指向不存在的块，或指到本题命中块之外 | **184 条全部可定位**，越界 0 |

`拒答翻转 0` 是灰度最重要的一条：它是用户能直接感知到的行为变化。20 题里两代回答/拒答
逐题相同。

`引用可定位 184 条` 的分母是 20 题的两次回答加起来（候选 + 基线逐题引用的总数），
`越界/对不上 0`、`引到命中块外 0`——两个 0 是分开判的：前者是「块 id 不存在」，
后者是「块存在但不是这一题检索命中过的块」。后者更严，因为它拦的是「引用了别处的正确块」
这种看着没问题、实则不可定位的引用。

#### 2.4 报告里另外三条硬事实

```
mode=concurrent
production_pointer_untouched=True
production_pointer={"before":"g-20261030T154606Z-55dfd7","after":"g-20261030T154606Z-55dfd7"}
servers: candidate port 56650 boot 401ms ready 19777ms peak_rss 1487 MiB
         baseline  port 56652 boot 401ms ready 21053ms peak_rss 1723 MiB
```

`production_pointer_untouched=true`，且 `before` 与 `after` 都是 `55dfd7`——**演练前后线上
指针逐字相同**。这是「灰度 ≠ 发布」的凭据：整个过程中线上那一代一次都没被碰过。

`模式 concurrent（门槛 6144 MiB）` 是**先测可用内存再选模式**：两个实例同时活着要预留余量，
可用内存不够就退化成串行模式（并如实标注），而不是硬起两个实例把机器打爆——那样量出来的
「候选慢」其实是自己造的。

`就绪 19777 / 21053 ms` 是进程从启动到能服务的时长，含管道构建（热身 8130 / 8124 ms 是
其中一次显式热身）。**这个数字同样是本次读数**：它受页缓存与磁盘负载影响，不要当启动性能。

### 3. K3 超阈值人工审批：阈值登记表与审批记录强制校验

#### 3.1 口径

`scripts/k3_thresholds.py` 把计划 K3 第 5 条「超阈值必须人工审批」变成一道**会变红的门**。

三条要点：

1. **指标自己算，不抄别人打印好的数字。** 指标从 `data/k3_recall_pairs.json` 重算——
   否则阈值比对的是二手数字，跨了也看不出来。
2. **阈值只有一个事实源**：`data/k3_thresholds.json`（`schema=k3-thresholds-1`）。脚本
   **只读不写**阈值，代码里不写死任何阈值常量。
3. **三种结局**：未跨阈值 → PASS（不需要审批）；跨了 + 有该指标的**有效**审批记录 → PASS；
   跨了 + 没有有效审批 → **FAIL**，并点名指标、实测多少、阈值多少、差多少。

「有效审批」的定义（`approval_policy`）不是「记录存在」就算：

* 七个字段齐全：`metric` / `measured` / `threshold` / `approved_by` / `approved_at` /
  `decision` / `rationale`；
* `approved_by` 必须是**自然人**——空串、`TBD`、`agent`、`auto` 之类占位符一律无效。
  **审批是人做的动作，程序不能自批。**
* `approved_at` 可解析为日期；`decision` 是明确的通过/拒绝；
* `measured` 与本次实测差 ≤ `1e-9`——**审批只能覆盖它当时看到的那一次实测，换一次实测就失效**。
* `direction: "red_line"` 的行（未解释差异数）**不接受审批**：归因能力跟不上改动时，
  签字不能替代归因。

#### 3.2 实跑结果（`py -3 scripts/k3_thresholds.py`，EXIT=0）

```
==============================================================================
K3 第 5 条判据：超阈值必须人工审批
==============================================================================
阈值唯一事实源：data/k3_thresholds.json（schema=k3-thresholds-1）
旧代 g-20261031T035231Z-4eafae
新代 g-20261103T132152Z-06a741

== 分母 ==
  共同题 320；有 gold 可评 300（无 gold 剔除 20）
  delta 判据 8 条；absolute 申报 9 条
  审批记录 0 条（可用 0，非对象被丢弃 0）

== 召回指标（两套口径逐条分开，Δ = 新 − 旧）==
  口径 A（丢弃 null）
    R@5   n=300  旧 0.510833 → 新 0.500833 Δ -0.010000
    R@20  n=300  旧 0.701944 → 新 0.698889 Δ -0.003056
    RR    n=229  旧 0.570038 → 新 0.561904 Δ -0.008134
  口径 B（0 填充，与 ask.py 同口径）
    R@5   n=300  旧 0.510833 → 新 0.500833 Δ -0.010000
    R@20  n=300  旧 0.701944 → 新 0.698889 Δ -0.003056
    RR    n=300  旧 0.439430 → 新 0.431197 Δ -0.008233

== 结构量 ==
  实体数          旧        6,582 → 新        6,583  Δ +0.0152%
  边数           旧       28,625 → 新       28,636  Δ +0.0384%
  词法索引字节       旧  127,244,898 → 新  106,685,068  Δ -16.1577%
  未解释差异 0（红线 0）

== delta 判据（逐指标）==
  [PASS] r20_delta                阈值 -0.0200  实测 -0.003056  未跨阈值，不需要审批（距线尚有 0.016944（实测 -0.003056 vs 阈值 -0.020000））
  [PASS] r5_delta                 阈值 -0.0300  实测 -0.010000  未跨阈值，不需要审批（距线尚有 0.020000（实测 -0.010000 vs 阈值 -0.030000））
  [PASS] rr_delta                 阈值 -0.0200  实测 -0.008233  未跨阈值，不需要审批（距线尚有 0.011767（实测 -0.008233 vs 阈值 -0.020000））
  [PASS] entity_count_drop_pct    阈值 -1.0000  实测 +0.015193  未跨阈值，不需要审批（距线尚有 1.015193（实测 +0.015193 vs 阈值 -1.000000））
  [PASS] entity_count_rise_pct    阈值 +5.0000  实测 +0.015193  未跨阈值，不需要审批（距线尚有 4.984807（实测 +0.015193 vs 阈值 +5.000000））
  [PASS] edge_count_drop_pct      阈值 -2.0000  实测 +0.038428  未跨阈值，不需要审批（距线尚有 2.038428（实测 +0.038428 vs 阈值 -2.000000））
  [PASS] lexical_bytes_drop_pct   阈值 -25.0000  实测 -16.157685  未跨阈值，不需要审批（距线尚有 8.842315（实测 -16.157685 vs 阈值 -25.000000））
  [PASS] unexplained_differences  阈值 +0.0000  实测 +0.000000  未跨阈值，不需要审批（距线尚有 0.000000（实测 +0.000000 vs 阈值 +0.000000））

== absolute 声明（F4 的绝对门，K3 不重复判，只申报差距）==
  [申报] recall_at_20                     当前 0.6989（n=300） vs ≥ 0.9000 → 未达标 -0.2011
  [申报] recall_at_5                      当前 0.5008（n=300） vs ≥ 0.7500 → 未达标 -0.2492
  [申报] mrr                              当前 0.4312（n=300）；本行不设绝对阈值，仅申报基线
  [申报] citation_localization            无实测输入（本门不评估这项，也不假装通过）
  [申报] key_fact_citation_coverage       无实测输入（本门不评估这项，也不假装通过）
  [申报] unanswerable_refusal             无实测输入（本门不评估这项，也不假装通过）
  [申报] directional_accuracy             无实测输入（本门不评估这项，也不假装通过）
  [申报] hard_integrity_errors            无实测输入（本门不评估这项，也不假装通过）
  [申报] failed_artifact_published        无实测输入（本门不评估这项，也不假装通过）

结论：8/8 条 delta 判据成立（本轮无指标跨阈值，因此未发生人工审批；这不是「审批流程已验证通过」）

写盘：k3_threshold_gate.json（all_passed=True）
```

#### 3.3 本轮没有任何指标跨阈值 —— 这句话决定这一节能说什么、不能说什么

8 条 delta 判据**全部未跨阈值**，最近的一条是 `lexical_bytes_drop_pct`：实测 −16.157685%，
阈值 −25.0000%，**距线 8.842315 个百分点**。其余七条距线都在 0.011 以上（`unexplained_differences`
实测 0、阈值 0，判据是「未**超过** 0」，所以 0 不构成跨线）。

因此 `data/k3_approvals.json` 的 `approvals` 是**空数组**（文件 5 行、706 B）：

```json
{ "schema": "k3-approvals-1", "note": "…本轮没有发生人工审批，一条记录都不该有。空数组不是「审批流程已验证通过」，只是「没有该审批的事」…", "approvals": [] }
```

脚本自己把这句话印在结论行里，本节照抄不改写：

> **本轮无指标跨阈值，因此未发生人工审批；这不是「审批流程已验证通过」。**

这条区分不是文字游戏，它决定这道门的**证据等级**：

| 已验证 | **未验证** |
|---|---|
| 阈值表存在且被逐条读取（8 条 delta + 9 条 absolute） | 跨阈值时是否真的会变红（没有一次跨线让它红过） |
| 「未跨 → PASS」这条路径 | 「跨了 + 有效审批 → PASS」这条路径 |
| 「距线多少」的算术 | 审批记录格式校验在真实记录上的表现 |
| 审批目录为空的处理（`审批记录 0 条，可用 0，非对象被丢弃 0`） | 七字段校验、`approved_by` 占位符拒绝、`measured` 差 ≤1e-9 这些规则**在真实数据上的**行为 |

**它目前只被验证到「能拦」的一半**：未跨的时候放行是对的，跨了的时候会不会拦住——
只有 `tests/test_k3_thresholds.py` 的 61 条用例做得到（它们构造跨线场景与伪审批记录），
**真实盘面上一次都没发生过**。§9 遗留 ②。

#### 3.4 两套口径为什么必须分开印

`R@5` / `R@20` 在口径 A 与口径 B 下逐位相同，**`RR` 不同**：

| 口径 | n | 旧 RR | 新 RR | Δ |
|---|---|---|---|---|
| A（丢弃 null） | 229 | 0.570038 | 0.561904 | −0.008134 |
| B（未命中填 0，与 `ask.py` 同口径） | 300 | 0.439430 | 0.431197 | −0.008233 |

阈值表里 `rr_delta` 那行**写明按口径 B 判**（与 `scripts/ask.py` 的评测器同口径），口径 A
只作申报。理由很硬：**门判的和线上量的必须是同一件事**。若门用口径 A（分母只算有金标的
229 题）判，而线上监控用口径 B（300 题、未命中填 0），那门永远是宽松的那一侧，跨线也看不出来。

口径 A 的 n=229 与口径 B 的 n=300 相差 71——这就是「丢弃 null」与「填 0」的差别，两个数
都不是错的，但**混着用就会得出相反的结论**。

### 4. K4 观察期与归档：机制先拦住，真实归档尚未发生

#### 4.1 缺口与补法

计划 K4 第 4 条是「新版稳定运行约定观察期后才归档旧版」。原来 K4 的 15 门里**没有一门**
碰这件事：归档函数不存在，观察期天数没有定义，`data/arkrag/archive/` 这个目录从来没被创建过。

本轮加了两样东西、两道门：

| 加的东西 | 位置 | 作用 |
|---|---|---|
| `OBSERVATION_DAYS = 7` + `_observation_days()` | `src/arkrag/release.py` | 观察期天数的唯一定义，可用环境变量 `ARKRAG_OBSERVATION_DAYS` 覆盖（演练与测试要用），但**默认值必须写在代码里**；非整数或负数直接抛 `ReleaseError`，不猜 |
| `observation_status()` | 同上 | 返回 `{generation_id, published_at, days_elapsed, required_days, complete, reason}` |
| `archive_old()` | 同上 | 归档旧版；观察期未满必须拒绝 |
| 门 `observation_window` | `scripts/k4_drill.py` | 观察期判定正确且 **fail-closed** |
| 门 `archive_refused_during_observation` | 同上 | 观察期未满时归档被拒，且**盘上零改动** |

`observation_status()` 的核心设计是**一切异常都落到 `complete=False`**（fail-closed）：

> 把不确定性往「已满」那边倒是 fail-open：指针坏掉、`published_at` 是空串、时间格式换了
> 个写法——任何一处都会让**刚上线几分钟**的新版被判成「已过观察期」，旧版随即被复制归档
> 并从世代目录删掉。反过来（判成「未满」）的代价只是多留几天磁盘。

所以：`published_at` 缺失或无法解析 → `complete=False` 并写明原因，**不当作 0 天，
也不当作已满**；`now` 早于 `published_at`（时钟回拨/手改时间戳）同样按未满处理，
因为时间线本身已经不可信。

两道门是**故意分开**的：观察期算得再准，只要 `archive_old()` 忘了调它，旧版照样会被搬走。
所以 `observation_window` 判「算得对不对」，`archive_refused_during_observation` 判
「有没有真的拿它去拦」。

#### 4.2 实跑结果（`py -3 scripts/k4_drill.py`，EXIT=0，17/17）

门数从 15 加到 17，两道新门：

```
[PASS] 观察期判定正确且 fail-closed（不判是否已满） 线上 g-20261030T154606Z-55dfd7 已 4.6444 天 / 要求 7 天 / complete=False；独立算 4.6444 天 → 一致；fail-closed 三例（无 published_at / 格式非法 / 无指针）complete 全 False=True
[PASS] 观察期未满时归档被拒且盘上零改动           真实 g-20261031T035231Z-4eafae：拒绝=True（4 个文件/242.2 MB SHA-256 前后一致）；合成世代：对照组（30 天）拿到计划 3 个文件、观察期 5.0 天时 dry_run=False 拒绝=True、产物零改动=True

K4 检查：17/17 通过
结论：K4 灰度验收与回滚演练全过
报告已写 data/k4_drill.json
```

线上那一代的观察期读数（`data/k4_drill.json` 的 `observation_window.data.observation`）：

```json
{"generation_id":"g-20261030T154606Z-55dfd7","published_at":"2026-10-30T15:49:29Z",
 "days_elapsed":4.6444,"required_days":7,"complete":false,
 "reason":"线上世代 g-20261030T154606Z-55dfd7 仅运行 4.64 天，未达观察期 7 天（还差 2.36 天），旧版不得归档"}
```

三门课都在这里：

1. **主判据不判「是否已满」**。门名写的是「观察期判定正确且 fail-closed（**不判是否已满**）」。
   若把「线上必须已满 7 天」当判据，这门从加上的第一天就是红的（线上才 4.6444 天），
   红久了必被加白名单。所以门判的是：`days_elapsed` 算得对（**用原始字节里的
   `published_at` 独立算一遍**，不经过 `observation_status()` 内部函数，两路都是 4.6444 天）、
   `required_days` 与 `OBSERVATION_DAYS` 吻合、`complete` 与实际一致。
2. **fail-closed 三例逐例验**：无 `published_at` / 格式非法（`2026/01/01 00:00`）/ 完全没有指针。
   三例 `complete` 全 `False`。这三例是「坏输入不会变成放行」的证据。
3. **归档门要证明的是「拒了 **且** 盘上没少一个字节」**。只证明「返回了拒绝」不够——
   函数可能先把文件搬走再报拒绝。真实那一代 `4eafae`：4 个文件 / 242.2 MB，
   **SHA-256 前后一致**。另一半是 `dry_run=False` 的**真实调用**：合成世代在观察期
   `5.0 天`时被拒、产物零改动；对照组（把观察期覆盖成 30 天）能拿到计划里的 3 个文件，
   证明「拒绝」不是因为「这个函数根本不干活」。

#### 4.3 但「机制能拦」不是「真的归档过」

这一节最必须写清的一句：**历史上从未真的归档过一次**。证据两条：

* `data/arkrag/archive/` 目录**从未被创建**（`Test-Path` → `False`）；
* `git log --all -- data/arkrag/archive` 查出 **0 条**提交——它连被误提交过一次都没有。

所以本轮补的全是**阻止不该发生的动作**的那一侧：

| 已验证 | **未验证** |
|---|---|
| 观察期未满时拒绝归档（真实世代 + 合成世代，两次） | 观察期满后**真的**归档一次 |
| 拒绝时盘上 SHA-256 不变（4 文件 / 242.2 MB） | 归档完成后的目录结构、`archive/` 里的 Manifest、旧版能否从归档恢复 |
| fail-closed 三例（坏指针不会放行） | 归档后 `prune_old()` 与新指针的相互作用 |

这与 §3 是同一类状态：**门是红的必要条件已经具备，绿的那条路径没走过。**
两者都不是「写个 TODO 等以后」，而是「日志里明确记着还没走」——`archive_old()` 的真实调用
在本轮只发生在 `dry_run=True` 与「被拒绝」两种情形下。

#### 4.4 另外 15 门本次读数

新增两门之外，15 门与 K4 §2 同（判据是 17 个 `ok` 布尔值，它们每次都必须一致）：

```
[PASS] 灰度查询（候选，不发指针）   题 18 / 答 17 / 拒 1 / 引用 94 条（定位失败 0）/ 单题 75.1 ms
[PASS] 灰度查询（上一代，对照）     题 18 / 答 17 / 拒 1 / 引用 88 条（定位失败 0）/ 单题 59.9 ms
[PASS] 拒答结论逐题稳定            翻转 0 题（两代回答/拒答一致）
[PASS] 回滚只切指针、不重建历史      回滚到 g-20261031T035231Z-4eafae 完成；12 份 Manifest 改动 0
[PASS] 演练未改坏任何世代 Manifest   备份 12 份 / 还原 1 份：['g-20261103T132152Z-06a741']
[PASS] 线上指针未被演练改动         线上仍是 g-20261030T154606Z-55dfd7（PUBLISHED）
[PASS] 回滚后固定探针              题 18 / 引用定位失败 0 / 回答器世代标识全对=True / 单题 62.4 ms
[PASS] Gold Set 冒烟（40 题，候选）  R@20 0.7436 / R@5 0.5897 / MRR 0.4591 / 引用定位 1.0000 / 硬失败 0
```

`12 份 Manifest 改动 0` 与 `备份 12 份 / 还原 1 份` 两条与 K4 §2/§8 逐字相同——**加了两道门
之后，原有 15 门的结论一个都没变**。这是加门而不是改判据的证据。

单题耗时 `75.1 / 59.9 / 62.4 ms` 是本次读数（K4 §3 已记同样两代测到过
`76.0 / 65.2`、`202.3 / 162.2`、`193.7 / 151.5` 等一串值），不作比较。

### 5. 顺带修掉：`k1_checks` 的回滚门在并发下会假红

这不是四处漏项之一，是本轮验收时**跑出来的真事故**，所以记在这里。

#### 5.1 现象

2026-11-04 15:19 跑 `py -3 scripts/k1_checks.py`，九项里**回滚那一项红**：

```
[FAIL] 回滚命令已实测   回滚到当前世代未被拒，previous_generation 会退化成指向自身
```

而当时另一个子代理正在跑全量 `pytest`。单独重跑这一项是绿的。

#### 5.2 根因：探针目录是写死的共享路径

`scripts/k1_checks.py` 原第 717 行：

```python
tmp_dir = Path(config.CIRCUIT_OPEN_PATH).parent / "_k1_rollback_probe"
```

于是**任意两个进程同时跑这一项就共用同一个 `current.json`**。这一项要做两件事：
先回滚一次（`A → B`），再回滚到自身（必须被拒）。两个进程交错时，A 写完
「当前是候选世代」，B 已经把它回滚成 `e0aacc` 并读出「当前是 e0aacc」，A 接着在
B 的指针上再回滚一次——这一次要么被「已经是当前世代」拒掉，要么干脆被接受。

写 `_race_probe.py` 并发 4 份 `--only rollback_tested` 实测，**修正前 0/4 通过**，
两种错法各出现两次：

```
#0 exit=1 回滚到当前世代未被拒，previous_generation 会退化成指向自身
#1 exit=1 release.rollback 拒绝执行：g-20261030T150101Z-e0aacc 已经是当前世代，无需回滚
#2 exit=1 release.rollback 拒绝执行：g-20261030T150101Z-e0aacc 已经是当前世代，无需回滚
#3 exit=1 回滚到当前世代未被拒，previous_generation 会退化成指向自身
通过 0 / 4
```

**`release.rollback()` 本身是对的**（`src/arkrag/release.py:772` 正确地拒绝回滚到自身，
单元级复现也正确拒绝）。错的是这道检查自己的脚手架——它把「两个进程互不干扰」
这个前提当成了默认成立。

#### 5.3 修法与自证

改成每次运行独一份：

```python
tmp_dir = Path(tempfile.mkdtemp(prefix="k1_rollback_probe_"))
```

不用「在 `data/arkrag` 下加进程号」：进程号会被复用，而且这个目录本来就不该出现在
冻结基线目录里（`snapshot_baseline.py` 会扫到它）。`mkdtemp` 保证唯一且落在系统临时区，
`finally` 里整目录清掉。

修正后并发 **4/4 通过**：

```
并发 4 份 scripts/k1_checks.py --only rollback_tested
  #0 exit=0 ok  [PASS] 回滚命令已实测
  #1 exit=0 ok  [PASS] 回滚命令已实测
  #2 exit=0 ok  [PASS] 回滚命令已实测
  #3 exit=0 ok  [PASS] 回滚命令已实测
通过 4 / 4
结论：探针目录每次运行独一份，并发下互不影响
```

这段并发复现被固化成 `scripts/k1_race_check.py`（不靠一次性临时脚本）：

```
py -3 scripts/k1_race_check.py            # 并发 4 份，EXIT=0 表示 4/4 通过
py -3 scripts/k1_race_check.py --n 8      # 改并发份数
```

它的退出码判据就是「全部通过 = 0，有任一份没通过 = 1」，不含任何「先看看输出」的余地。

新增用例 `tests/test_k1_checks.py::TestRollbackTestedCanFail::test_probe_dir_is_unique_per_run_and_cleaned_up`
钉住的性质是「两次运行拿到两个不同目录、都不在 `CIRCUIT_OPEN_PATH` 同级、跑完都不存在」，
不是「用了 `mkdtemp`」这个实现。

**反向实测一**（把 `k1_checks.py` 那一行改回共享路径后跑单测）：`AssertionError: 两次运行
共用了同一个探针目录：{WindowsPath('E:/图rag/data/arkrag/_k1_rollback_probe')}`，
`1 failed, 36 deselected`，`exit=1`。还原后 `scripts/k1_checks.py` SHA-256 前后同为
`5c692d83e99fc3940a3f27db66944c96623a3c8bc0b0f2fd92bffa10172af486`，逐字节一致；
重跑 `-k Rollback` → `5 passed`。

**反向实测二**（同样的改法，跑 `scripts/k1_race_check.py`）：

```
并发 4 份 scripts/k1_checks.py --only rollback_tested
  #0 exit=1 FAIL  [FAIL] 回滚命令已实测
  #1 exit=1 FAIL  [FAIL] 回滚命令已实测
  #2 exit=1 FAIL  [FAIL] 回滚命令已实测
  #3 exit=1 FAIL  [FAIL] 回滚命令已实测
通过 0 / 4
结论：有并发冲突。…
EXIT=1
```

还原后 `scripts/k1_checks.py` SHA-256 又是 `5c692d83…af486`（逐字节一致）。
两次反向实测覆盖两种入口：单测（钉实现约束）与并发脚本（钉真实场景）。

#### 5.4 为什么这算「真事故」而不是「测试环境的噪声」

因为它**红的方向是错的**：报的是「回滚命令不可用」，而回滚命令一直是好的。一条会在
并发下报假红、且假红内容指向错误结论的门，比没有门更糟——下一次它真红的时候，
第一反应会是「又是那个假红」。

另有一条纪律值得记下来：`tests/test_k1_checks.py::test_real_pointer_is_never_written`
**也调这一项**（它要证明真实指针一个字节没动），所以「跑全量 pytest」与「手动跑
`k1_checks.py`」天然就会撞车。这不是罕见场景，是日常场景。

### 6. 发布前清单门：五道门都跑过、而且跑的是这一代，才允许发布

#### 6.1 为什么另加一道门

`release.publish()` 自己只查三件事：状态是 `READY`、`coverage_status` 完整、
`verify_generation()` 没有不符项。它**看不到那五份门报告**——`data/k2_hard_gates.json`、
`data/k2_perf.json`、`data/k2_gray.json`、`data/k3_threshold_gate.json`、`data/k4_drill.json`。
也就是说，在这个门之前，「五道门全过」只是五份文件里各自写着 `all_passed: true`，
没有任何东西阻止一次越过它们的发布：报告可以缺、可以绑在上一代上、可以自己就是红的。

这不是假设。K4 演练暴露过一次同类事故：`publish()` 会改写目标世代的 Manifest
（K4 §8），当时是靠演练里的 `_ManifestGuard` 把 `06a741` 认回来。**靠人记得去比对
五份报告不是机制**，所以补一道门。

#### 6.2 五份报告的世代字段名不一样，不许统一改它们

这是这道门最容易写错的地方：五份报告不是同一个脚本产出的，各自把「我是哪一代」写在
不同位置。消费者必须逐份按登记的字段路径取，**不许为了好写去改那五份报告的格式**——
它们是各自门的产出物，改动等于让已经跑过的门报告失效。

| 报告 | 世代字段 | 顶层键 |
|---|---|---|
| `data/k2_hard_gates.json` | `generation_id` | `generation_id` / `baseline_generation_id` / `checks`(8) / `all_passed` |
| `data/k2_perf.json` | `generation_id` | 11 键，含 `generated_at` |
| `data/k2_gray.json` | `generation_id` | 17 键，含 `generated_at` / `production_pointer_untouched` |
| `data/k3_threshold_gate.json` | **`generations.new`** | `schema` / `thresholds_file` / `generations` / `checks`(8，键是 `metric` 不是 `id`) / `all_passed` |
| `data/k4_drill.json` | **`candidate`** | `candidate` / `baseline` / `probes`(18) / `checks`(17) / `all_passed` |

`REPORTS` 登记表里每条写 `key` / `label` / `path` / `gid_field`（点号路径）/ `producer`
（失败时告诉人该跑哪个脚本）。`tests/test_k2_publish_check.py` 有一条用例专门
**钉住这五个 `gid_field` 名**——将来谁改了报告格式却忘了改登记表，那条用例会红。

#### 6.3 六条判据，三态，`all_passed` 只统计 FAIL

| 判据 | 红在什么情况 |
|---|---|
| `reports_present` | 五份报告有缺的、或 JSON 读不出来（detail 点名缺哪个文件、该跑哪个脚本） |
| `reports_passed` | 某份报告自己 `all_passed` 不是 `True`（**`None` 不算过**，缺字段不能当通过） |
| `reports_bind` | 某份报告绑的不是本次要发布的世代（detail 同时打出「要发布的是」和两个 id） |
| `candidate_status` | 候选状态不是 `READY` |
| `candidate_verify` | `verify_generation()` 有不符合项（截断到 8 条但仍报总数） |
| `reports_not_stale` | 报告 `generated_at` 早于候选 `created_at` |

三态是关键：`pass` / `fail` / `undetermined`。`all_passed` **只统计 `fail`**；
`undetermined` 的 `ok` 是 `False` 但**不使门失败**。这样「不知道」既不被写成通过，
也不被当成红——把不知道写成知道，正是这道门本来要防的事。

`reports_not_stale` 目前就是 `undetermined`：只有 `k2_perf` 与 `k2_gray` 有
`generated_at`，另三份报告没有时间戳字段。这条**现在无法判定**，脚本如实这么写，
用例 `test_three_reports_without_timestamp_are_undetermined_not_pass` 钉住它。

六条判据都是纯函数，返回 `(verdict, detail, data)`；`evaluate()` **六条永远全跑**，
不是遇到第一条红就返回——一次说清所有问题，省得改一条跑一次。

#### 6.4 实跑

```
$ py -3 scripts/k2_publish_check.py
==============================================================================
发布前清单门  Python 3.14.3  win32
要发布的世代 g-20261103T132152Z-06a741
报告目录 data
口径：五份门报告各自绑的世代字段不同（见 REPORTS 登记表），逐份比对
------------------------------------------------------------------------------
逐份报告
  data/k2_hard_gates.json                  世代字段 generation_id      = g-20261103T132152Z-06a741  all_passed=True  generated_at=None
  data/k2_perf.json                        世代字段 generation_id      = g-20261103T132152Z-06a741  all_passed=True  generated_at=2026-11-04T10:20:59Z
  data/k2_gray.json                        世代字段 generation_id      = g-20261103T132152Z-06a741  all_passed=True  generated_at=2026-11-04T10:03:51Z
  data/k3_threshold_gate.json              世代字段 generations.new    = g-20261103T132152Z-06a741  all_passed=True  generated_at=None
  data/k4_drill.json                       世代字段 candidate          = g-20261103T132152Z-06a741  all_passed=True  generated_at=None
------------------------------------------------------------------------------
清单判定
[PASS] 五份门报告在场且可解析                  5/5 份报告在场且可解析
[PASS] 每份报告自身 all_passed=True       5/5 份报告 all_passed=True
[PASS] 每份报告绑的是本次要发布的世代              5/5 份都绑 g-20261103T132152Z-06a741
[PASS] 候选状态是 READY                  候选 g-20261103T132152Z-06a741 状态 READY
[PASS] 候选 verify_generation() 无不符项  候选 g-20261103T132152Z-06a741 自检 0 项不符
[无法判定] 报告不早于候选建成                    2 份可比且都不早于候选建成；另 3 份无时间戳字段，本条对其无法判定：data/k2_hard_gates.json；data/k3_threshold_gate.json；data/k4_drill.json
------------------------------------------------------------------------------
清单 5/6 条成立（另有 1 条无法判定，不计为通过）
报告已写 data/k2_publish_check.json
==============================================================================
EXIT=0
```

`data/k2_publish_check.json` 顶层 10 键，`all_passed=true`、`failed=[]`、
`undetermined=["reports_not_stale"]`。退出码：0=可发布 / 1=有不成立 / 2=硬错误。

#### 6.5 接进发布路径，以及那一次真的把线上切了

只写一个清单脚本是不够的——脚本写好了没人调用，它只是一份没人拦的报告。所以
`scripts/release_cli.py` 的 `cmd_publish()` 改成**先过清单**，`PublishCheckError` →
「拒绝发布：发布前清单门无法判定」返回 1；`not verdict["all_passed"]` →
「拒绝发布：发布前清单 N 条不成立」+ 逐条 + 「这道清单没有跳过开关」返回 1；
有 `undetermined` 则打印「清单通过（另有 N 条无法判定…）」+ 逐条（**必须说出来**，
否则「清单全过」会被读成六条都验过了）。这些都过了才 `release.publish()`。

判定只写一处：清单脚本里的 `evaluate_generation()`，发布路径调它。按纪律不许
「发布路径与门各写一套」。

**踩了一次真事故**：验证接线的第一反应是跑 `py -3 scripts/release_cli.py publish
g-20261103T132152Z-06a741`，结果它**真的发布了**——`current.json` 被切成 `06a741`、
`published_at=2026-11-04T09:38:57Z`，`06a741/manifest.json` 被改成 `PUBLISHED`，
还多出一个 `data/arkrag/runs/r-20261104T093830Z-266954/`。已用
`git checkout -- data/arkrag/current.json data/arkrag/generations/g-20261103T132152Z-06a741/manifest.json`
（两者都被 git 跟踪）加删除那个 run 目录逐字节还原：`current -> 55dfd7 PUBLISHED`、
`06a741 -> READY` 且 `published_at` 键不存在、`git status --porcelain -- data/arkrag` 为空。

教训写在这里：**`release_cli publish <gid>` 会真的切线上指针，不许当「测一下清单」跑。**
要验证拦截只能跑清单脚本本身（改报告 → 跑 → 还原），或者在 tmp 沙箱里跑（用例里的
`pub_sandbox` 夹具 monkeypatch `release.GENERATIONS_DIR` / `RUNS_DIR` / `CURRENT_PATH`）。

#### 6.6 用例，以及反向实测它们确实能红

`tests/test_k2_publish_check.py` 共 **36 条**：31 条测清单本身（缺报告 / 坏 JSON /
单份红 / 字段缺失不算过 / 绑上一代 / 三个非 `generation_id` 字段都读对 / 候选六种
非 READY 状态 / 自检不符截断 / 三份无时间戳必须 `undetermined` / 补时间戳转 pass /
早于候选建成则红 / 六条全跑 / 钉住五个 `gid_field`），**5 条测发布路径本身**：

* `test_publish_refused_when_a_report_is_missing` —— 缺 `k2_perf` → 拒、err 里含该跑的
  脚本名、不写指针、候选状态仍是 `READY`
* `test_publish_refused_when_a_report_binds_another_generation` —— 报告绑上一代 → 拒
* `test_publish_refused_when_a_report_failed_itself` —— 报告自己红 → 拒
* `test_publish_proceeds_and_says_which_checks_were_undetermined` —— 全齐则真发布，
  且输出必须写明「另有 1 条无法判定」
* `test_publish_still_lets_release_checks_reject_after_checklist_passes` —— 清单过了但
  覆盖不全，`release.publish()` 仍拒（证明清单不是发布判定的替代品）

这 5 条写错过一轮，两个真原因：①`release_cli` 里的 `import k2_publish_check` 与本文件
`spec_from_file_location` 载的是**两份模块实例**，只 patch 一处等于没 patch（症状是
`0/5 份绑对世代`）；②`_build_generation()` 用 `_now()` 造 `created_at`，而夹具里写死了
`generated_at`，于是「报告早于候选建成」误红。

反向实测：把 `cmd_publish()` 里清单那 25 行（1011 字符）短路掉，
`pytest tests/test_k2_publish_check.py -q -k publish` → **`5 failed, 31 passed`**，
红的正是那 5 条；还原后 `36 passed`。`scripts/release_cli.py` 改前/还原后 SHA-256
同为 `51bdc509…c49867`。

反向脚本本身也踩了一次：第一版用 `path.write_text()` 还原，Windows 下把 LF 换成 CRLF，
于是「逐字节一致」判 False——**那不是内容没还原，是换行被改了**。修法：`read_bytes()` /
`write_bytes()` 全程按字节。这一条对以后所有反向实测都适用。

### 7. 新增与改动文件

行数为 `[IO.File]::ReadAllLines(...).Count` 的实测值；改动文件的增删行为 `git diff --numstat`。

| 路径 | 行数 / 增删 | 性质 |
|---|---|---|
| `scripts/k2_perf.py` | 811 行 | 新增：性能评测，7 硬门 + 22 申报项 |
| `tests/test_k2_perf.py` | 475 行（38 用例） | 新增 |
| `data/k2_perf.json` | 1165 行 / 36,540 B | 新增产物 |
| `scripts/k2_gray.py` | 1093 行 | 新增：两实例逐题交替 |
| `tests/test_k2_gray.py` | 608 行（54 用例） | 新增 |
| `data/k2_gray.json` | 2739 行 / 115,745 B | 新增产物 |
| `scripts/k3_thresholds.py` | 615 行 | 新增：阈值审批门 |
| `tests/test_k3_thresholds.py` | 472 行（61 用例） | 新增 |
| `data/k3_thresholds.json` | 215 行 / 11,358 B | 新增：阈值**唯一事实源** |
| `data/k3_approvals.json` | 5 行 / 706 B | 新增：审批记录（`approvals: []` 空） |
| `data/k3_threshold_gate.json` | 303 行 / 8,906 B | 新增产物 |
| `src/arkrag/release.py` | +363 / −0 | 改动：`observation_status()` / `archive_old()` / `OBSERVATION_DAYS=7` |
| `scripts/k4_drill.py` | 889 行（+476 / −1） | 改动：13→17 门 |
| `tests/test_release.py` | +212 / −0（+14 用例，现采 70 条） | 改动 |
| `scripts/k1_checks.py` | 961 行（+14 / −2） | 改动：回滚探针目录改 `tempfile.mkdtemp()`（§5） |
| `scripts/k1_race_check.py` | 新增（并发复现 §5 的修复） | 新增：4 份并发跑回滚那一项，全过 0 / 有不过 1 |
| `scripts/k2_publish_check.py` | 558 行 | 新增：发布前清单门，6 判据三态（§6） |
| `tests/test_k2_publish_check.py` | 622 行（36 用例） | 新增：31 条清单 + 5 条发布路径（§6.6） |
| `data/k2_publish_check.json` | 168 行 / 4,780 B | 新增产物 |
| `scripts/release_cli.py` | 202 行（+42 / −2） | 改动：`cmd_publish()` 先过发布前清单（§6.5） |
| `tests/test_k1_checks.py` | 720 行（+63 / −0，+1 用例，现采 37 条） | 改动：钉住探针目录每次独一份（§5） |
| `tests/test_suite_hygiene.py` | +63 / −1（`N` 1291→1471→1511） | 改动：用例总数常量（§5 +1、§6 +40） |
| `data/k4_drill.json` | 271 行（+82 / −9） | 改动：报告刷新 |

用例增量逐笔对得上：38 + 54 + 61 + 14 = **167**；再加**三个新测试文件进入
`TEST_FILES`** 后让 `tests/test_suite_hygiene.py` 里四个 parametrize（61/130/150/184 行）
各多一项，**+12**；再加 §5 那条并发用例 **+1**；合计 **180**。1291 + 180 = **1471**。

§6 的清单门再加两级：`tests/test_k2_publish_check.py` 先进 31 条，作为**新测试文件**
又让那四个 parametrize 各多一项（**+4**）→ 1471 + 35 = **1506**；随后补 5 条发布路径
用例（该文件已在 `TEST_FILES` 里，不产生 parametrize 增量）→ 1506 + 5 = **1511**。
`tests/test_suite_hygiene.py:361` 的 `N = 1511` 与 `--collect-only` 实测一致。

五个新测试文件单跑合计 38 + 54 + 61 + 70 + 36 = **259 条**（§5 那一条在
`tests/test_k1_checks.py` 里，不在这五个之内）。

口径上补一句：`k4_drill.py` 那两道新门**没有专门的测试文件**（`tests/` 下没有
`test_k4_drill.py`），它们的判据由 `scripts/k4_drill.py` 自己每次运行断言，逻辑单元由
`tests/test_release.py` 的 14 条覆盖——后者测的是 `observation_status()` / `archive_old()`
本身，不是「演练脚本有没有正确调用它们」。这两者不是一回事（§4.1 末段）。

### 8. 复现命令

```powershell
cd E:\图rag
$env:PYTHONIOENCODING="utf-8"

py -3 scripts/k2_perf.py           # 性能评测；本次 EXIT=0，硬门 7/7
py -3 scripts/k2_gray.py           # 灰度；本次 EXIT=0，5/5（会起两个 HTTP 实例，端口 56650/56652）
py -3 scripts/k3_thresholds.py     # 阈值审批；本次 EXIT=0，8/8
py -3 scripts/k4_drill.py          # K4 演练；本次 EXIT=0，17/17
py -3 scripts/k2_publish_check.py  # 发布前清单；本次 EXIT=0，5/6 + 1 条无法判定（§6）
py -3 scripts/k1_checks.py         # K1 九项；本次 EXIT=0，9/9（回滚那项已在 §5 修好）
py -3 scripts/k1_race_check.py     # 并发 4 份跑回滚那一项；EXIT=0 表示 4/4（§5）

py -3 -m pytest tests/test_k2_perf.py tests/test_k2_gray.py tests/test_k3_thresholds.py tests/test_release.py -q
py -3 -m pytest tests/test_k2_publish_check.py -q     # 36 passed（§6.6）
py -3 -m pytest tests -q           # 1510 passed, 1 skipped（1511 collected）
```

注意：

* `k2_perf.py` 的**时序数字要在无并发时重跑**才有比较意义；同机同时跑 pytest 会把 P95 推高
  （§1.4 的两次方向相反就是这个原因）。判据（7 个 `ok`）不受影响。
* `k2_gray.py` 会真的起两个 `serve.py` 进程，需要可用内存 ≥ 6144 MiB（不够会退化成串行模式
  并如实标注）；跑完端口自动释放，不写发布指针。
* `k4_drill.py` 全程在隔离指针上操作，`finally` 里还原；真实线上指针一次都不写。
* PowerShell 下 `py -3 -m py_compile scripts/k*.py` **不会展开通配符**，别当它跑通了。
* **不许用 `py -3 scripts/release_cli.py publish <gid>` 去试清单门拦不拦**——它会真的切线上指针（§6.5 那次事故）。要试就改报告后跑 `scripts/k2_publish_check.py`，或在 tmp 沙箱里跑用例。

### 9. 遗留

1. **性能时序不可复现，只能当「本次读数」。** 同一批题、同一份代码，候选热查 P95 本次读到
   130.687 ms（`py -3 scripts/k2_perf.py` 输出）；`tests/test_suite_hygiene.py:331` 记着另一次
   观测从 167.671 ms 跳到 544.19 ms（3.2 倍）。方向上本次甚至与 Lead 转述的早先干净环境读数
   **相反**（本次候选慢 1.462×，那次据称候选快 0.571×）——但那一对数的原值（127.403 / 223.186）
   本轮查不到留档，只能当传闻。**「候选到底比上一代快多少」这个问题，本轮没能力回答**——要回答
   必须在无并发时重跑，并把当时的机器负载一起记下来。兜底门 2× 只保证「没慢到十倍」这一件事。

2. **`k3_approvals.json` 是空的、`data/arkrag/archive/` 从未创建——两处都只到「机制能拦」。**
   K3 的 8 条 delta 本轮全部未跨阈值（最近的 `lexical_bytes_drop_pct` 距线 8.842315
   个百分点），所以一条审批记录都没有；K4 的线上世代才 4.6444 天 / 要求 7 天，所以一次
   归档都没发生过（`git log --all` 0 条提交、目录不存在）。两处的**绿路径**（真的审批一次、
   真的归档一次）都从未走过，只由 `tests/` 里的构造用例覆盖。换句话说：这两道门目前是
   「红的时候会不会红」被验证了，「绿的时候对不对」还没被验证。

3. **灰度里 `q-na-000013` 两代都答，而 Gold Set 标 `should_abstain=true`。** 两代一致
   （候选与基线都在这个题上答了、各引 1 条、`gold命中0`），所以**不是灰度回归**，
   `abstain_flip` 门判的是「换代前后是否翻转」而不是「是否符合 Gold 标注」，故未把它设成门。
   这是 **Gold Set 质量问题**（该题标注与实际可答性冲突），另开一轮处理。
   顺带记录：20 行探针输出逐行数下来，**15 题 `gold命中1`、5 题 `gold命中0`**
   （`q-dr-000002` / `q-mh-000002` / `q-tc-000002` / `q-na-000013` / `q-na-000014`）——
   命中 0 本身不等于答错（候选可能引到了别处正确块），但这 5 题值得在下轮复核标注。

4. **本轮没做真实发布，线上仍是 `55dfd7`。** 计划 K2 的流程是「灰度查询 → 正式发布」，
   本轮补的是前半段（灰度能在真 HTTP 上跑通、逐题一致），**后半段「正式发布」一行代码都没写**：
   没有改指针、没有放量、没有发布后的观察。`k2_gray.json` 里
   `production_pointer_untouched=true`、`before`/`after` 都是 `55dfd7`，就是这件事的凭据。
   候选 `06a741` 至今仍是 READY 未发布。

5. **`absolute` 那 9 条里 6 条「无实测输入」。** `citation_localization` /
   `key_fact_citation_coverage` / `unanswerable_refusal` / `directional_accuracy` /
   `hard_integrity_errors` / `failed_artifact_published` 六项在本门里**没有实测输入**，
   脚本逐条印「本门不评估这项，也不假装通过」。它们属于 F4 的绝对门，K3 只申报差距——
   这不是本轮引入的问题，但要写在这里，免得后来人以为 K3 把那六项也判过了。

6. **`p95_backstop` 的 2× 阈值是拍的。** 取 2× 的理由是「只为抓灾难性回退」，但这个 2
   没有数据支撑（本轮观测到的噪声量级是 1.462×）。等积累了几次干净环境读数之后应该回头
   校准；现在它只是个极宽的保险丝，不能当性能门读。
