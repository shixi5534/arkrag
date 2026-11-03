# K 阶段实测（K1 全量重建前检查）

> 本文件记录 K 阶段的实测数据。当前覆盖 **K1**；K2/K3/K4 完成后追加。
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
