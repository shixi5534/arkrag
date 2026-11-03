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

量化（`_tmp_serving_gap` 的结论）：

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
