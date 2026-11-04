# Gold Set v2 说明

`data/goldset/gold-v2.json` —— 640 条评测题（v1 320 条切分 + 新增 320 条挖题），
`split` 字段的构造规则、八类配额、污染声明与已知结构局限。

- 生成命令：`py -3 scripts\make_split.py`
- 内置自检：`make_split.py::self_check()`（12 类约束，退出码 1 表示自检失败）
- 外部门：`py -3 -m pytest tests\test_make_split.py -q`（17 passed）
- 切分文档：`data/goldset/split-v2.json`（规则、逐类账、holdout 名单）
- schema：`ark-goldset-2`（题）/ `ark-split-1`（切分规则）
- `split_seed`：`arkrag-split-v1-20261104`（切分）/ `20261111`（新增批挖题）
- 数据源世代：`data/arkrag/generations/g-20261103T132152Z-06a741/`

指纹（2026-11-05 实测）：

| 文件 | 字节 | sha256 |
|---|---|---|
| `data/goldset/gold-v2.json` | 601,492 | `069375f8e355e4b9c3b4253e14959532f87e05bc129d5a7a6da39e07f357a108` |
| `data/goldset/split-v2.json` | 9,145 | `3d586b36efc7ce85dab5906aa1dd2eb6b7b81f9cc0b5071b630add714915b753` |
| `data/goldset/gold-v1.json`（**未改动**） | 284,356 | `76d66073ff87aab1c53f5fcd82277b732362be9372ed311d66d5308c159203d1` |

---

## 一、为什么 v1 原样保留、另建 v2

`data/goldset/gold-v1.json` **一个字节都没动**。理由两条：

1. **它是历史对照锚点。** 2026-10-30～11-04 期间所有实测（K 阶段全量回归、
   引用覆盖漏斗、A 的阶段报告）都报在 v1 上。就地改 v1 等于让那些数字失去对照物。
2. **换题集是建立新契约，不是修改旧契约。** v1 的结构缺陷（见第六节）会由 v2 的
   新增批暴露得更清楚，但修它必须先有「旧的那套长什么样」的完整留痕。

所以 v2 = v1 原题（逐字段保留）+ 新增 320 题 + 每题两个新字段：

- `split`：`dev` / `holdout`
- `origin`：`v1` / `new`

`tests/test_make_split.py::test_v1_half_is_byte_identical_except_split_and_origin`
把 v1 那 320 条剥掉这两个字段后逐字段与 `gold-v1.json` 比对，相等才算过。
`test_gold_v1_is_untouched` 守 v1 文件本身的 sha256 与字节数。

---

## 二、切分规则（可复现，不靠记忆）

规则全文写在 `data/goldset/split-v2.json` 的 `rule` 段，代码在
`scripts/make_split.py`，两者的常量由 `test_split_seed_is_frozen` 钉住。
复现只需三条：

1. **分层键 = `(category, should_abstain)`。**
   不用 `difficulty`，因为**实测 `difficulty` 在每一类内是常量**
   （single_chunk/alias 全 easy、direct_relation/cross_chunk/negative_abstain 全 medium、
   multihop/time_conflict/adversarial 全 hard），它在类内没有信息量。
   唯一在类内混合的维度是 `negative_abstain` 的 `should_abstain`（20 True / 12 False），
   其余七类恒为 False。`test_stratum_ignores_difficulty_and_uses_should_abstain`
   每次跑都会实测这条前提，若哪天 difficulty 不再常量，该用例会红并提示重新论证分层键。

2. **桶 = `sha256(SPLIT_SEED + "|" + tag + "|" + question_id)`，层内按桶升序。**
   `tag` 取 `"v1"` 或 `"new"`，所以两批用同一把尺子但互不干扰。
   `test_split_is_deterministic` + `test_bucket_is_stable_and_depends_on_tag` 守这一条：
   改拼串方式等于重新切分，**没有任何运行时症状，只有指纹会变**。

3. **取前 `ceil(ratio * n)` 进 holdout。** 用 `ceil` 不用 `round`：
   `n=2, ratio=0.30` 时 `round` 给 0 条，整类在 holdout 里缺席；
   `ceil` 保证每类至少有 1 条代表。`test_ceil_keeps_small_strata_in_holdout` 守这条。

比例：v1 `ratio = 0.30`（320 → holdout 100 / dev 220）；
新增批先在批内 `ratio = 0.50`（320 → holdout 160 / dev 160）。
**v1 用 0.30 而不是 0.20 的理由**：0.20 时 `negative_abstain(可答)` 与 `adversarial`
各只剩 3 条，单类结论不可用。

合计：**holdout 260 / dev 380 / 共 640。**

---

## 三、八类配额

### v1 部分（原 320，按 0.30 切）

| category | should_abstain | n | holdout | dev |
|---|---|---|---|---|
| single_chunk | false | 64 | 20 | 44 |
| alias | false | 48 | 15 | 33 |
| direct_relation | false | 48 | 15 | 33 |
| multihop | false | 48 | 15 | 33 |
| cross_chunk | false | 32 | 10 | 22 |
| time_conflict | false | 32 | 10 | 22 |
| negative_abstain | false | 12 | 4 | 8 |
| negative_abstain | true | 20 | 6 | 14 |
| adversarial | false | 16 | 5 | 11 |

### 新增批（seed 20261111 挖 640 → 三条件去重得池 566 → 取 320，按 0.50 切）

| category | n | holdout | dev | 池余量 |
|---|---|---|---|---|
| single_chunk | 64 | 32 | 32 | 123 |
| alias | 48 | 24 | 24 | 86 |
| direct_relation | 48 | 24 | 24 | 84 |
| multihop | 48 | 24 | 24 | 86 |
| cross_chunk | 32 | 16 | 16 | 56 |
| time_conflict | 32 | 16 | 16 | **48** |
| negative_abstain | 32 | 16 | 16 | 52 |
| adversarial | 16 | 8 | 8 | 31 |

去重三条件（对 v1）：问句文本不同、`gold_chunk_ids` 无交集、`gold_assertion_ids` 无交集。
三者同时成立才算「新题」，由 `test_new_half_does_not_overlap_v1` 每次跑实测。

**新增批取 320 不取 480 的理由**：池里 `time_conflict` 只有 48 条，
取 480 会把 tc 用满、零余量；取 320 时 tc 用 32/48 留 16 条备用。
n=160 的 95% 置信区间半宽是 3.4pp，够用。

难度分布 112 easy / 112 medium / 96 hard，`should_abstain` 32 True / 288 False，
与 v1 完全一致（`shortfall = {}`，`gaps = {}`）。

---

## 四、污染声明（对 A 而言，holdout 不等于「从未见过」）

**这一条不许软化：工程师 A 已经看过 v1 全 320 题的逐题结果**
（gold 排名、上下文命中、答案覆盖、引用覆盖状态都是他跑出来的）。
所以从这 320 题里切出来的 v1-holdout 100 条，对 A **不等于「从未见过」**。
切分只解决「同一批题不要再拿去做参数搜索」这一件事，**不解决污染**。

| 部分 | 题数 | 对 A 是否干净 | 用途 |
|---|---|---|---|
| v1-dev | 220 | 否 | 可继续用于诊断 |
| v1-holdout | 100 | **否**（A 已逐题看过） | 只用于验收留痕，不能算「盲测」 |
| new-dev | 160 | 是（新题，A 未见过） | 调参可以报数 |
| new-holdout | 160 | 是（新题，A 未见过） | **唯一对 A 干净的验收证据** |

管控规则：A 后续实验只在 `dev` 侧报数；动 holdout 要记账（第几次、什么目的、结果），
次数上限由 Lead 定。A 的产物里已有 `split_policy: "pre-split-diagnostic..."` 与
`contamination_note` 字段，引用时以那两处为准。

---

## 五、扩题不改变结构天花板（origin × split × category）

这一节回答「新增批是不是更干净的评测集」。**不是。**

判据与指标一致（`scripts/ask.py:283-298`，单向）：
某条 required 可达 ⇔ 同 chunk 内存在一条断言，其 `evidence_text` **以 required 那句为子串**。
这是判据成立的**必要条件**——达不到这一步的题，检索怎么排都判不过。

### 5.1 题级（一道题的全部 required 都可达才算可达）

| category | v1-dev | v1-holdout | new-dev | new-holdout |
|---|---|---|---|---|
| single_chunk | 44/44=1.00 | 20/20=1.00 | 32/32=1.00 | 32/32=1.00 |
| time_conflict | 22/22=1.00 | 10/10=1.00 | 16/16=1.00 | 16/16=1.00 |
| direct_relation | 33/33=1.00 | 14/15=0.93 | 24/24=1.00 | 24/24=1.00 |
| multihop | 33/33=1.00 | 15/15=1.00 | 23/24=0.96 | 23/24=0.96 |
| **alias** | 3/33=0.09 | **0/15=0.00** | **0/24=0.00** | **1/24=0.04** |
| **cross_chunk** | 0/22=0.00 | **0/10=0.00** | **0/16=0.00** | **0/16=0.00** |
| **adversarial** | 0/11=0.00 | **0/5=0.00** | **0/8=0.00** | **0/8=0.00** |
| **negative_abstain** | 0/8=0.00 | 0/4=0.00 | （应拒答，无 required） | （同左） |

**四类恒 1.00、四类恒 0.00，跨 v1 / new 一模一样。** 这不是抽样噪声，也不是污染，
是 `src/arkrag/goldset.py` 那四条挖掘路径的**确定性产物**。

### 5.2 条级（每个 `required_citations` 条目单独判）

口径说明：`multihop` / `cross_chunk` 一题常带 2–4 条 required，所以条级数与题级数不同。
**两个口径都对，但引用时必须写明是哪一个**，否则两个人对不上账。

| category | v1-dev | v1-holdout | new-dev | new-holdout |
|---|---|---|---|---|
| multihop | 66/66=1.00 | 30/30=1.00 | 47/48=0.98 | 47/48=0.98 |
| cross_chunk | 4/53=0.08 | 1/27=0.04 | 3/44=0.07 | 1/42=0.02 |
| alias | 3/33=0.09 | 0/15=0.00 | 0/24=0.00 | 1/24=0.04 |

### 5.3 四个子集的结构天花板

| 子集 | 题数 | 应答题 | 可达 | 天花板 |
|---|---|---|---|---|
| v1-dev | 220 | 206 | 135 | **0.6553** |
| v1-holdout | 100 | 94 | 59 | **0.6277** |
| new-dev | 160 | 144 | 95 | **0.6597** |
| new-holdout | 160 | 144 | 96 | **0.6667** |

四个子集全在 0.63–0.67 之间，**新增批（0.6667）并不比 v1（0.6277）更高**。

### 5.4 结论

> 新增批不是「更干净的评测集」。四条挖掘路径
> （`mine_alias_questions` / `mine_cross_chunk` / `mine_adversarial` /
> `mine_negative_and_unanswerable`）产出的 required 引用与断言层不同源，
> 该缺陷**跨数据集恒定**（new holdout 的 alias 1/24、cross_chunk 0/16、adversarial 0/8）。
> 因此**扩题不改变结构天花板**：v1 dev 0.6553、v1 holdout 0.6277、new dev 0.6597、
> new holdout 0.6667——四个子集都在 0.63–0.67 之间。任何「引用覆盖 ≥0.95」的目标
> 在现有 gold 与抽取上不可达，除非先修挖掘路径或改分母口径。

**注意这句话的边界**：0.6467（v1 300 应答题口径）是**当前 gold + 当前抽取**的天花板，
**不是系统的能力天花板**。不许把它外推成「任务不可达」。

---

## 六、已知局限：required 引用与断言层不同源

`src/arkrag/goldset.py` 的八条挖掘函数分成两组，行为截然不同：

| 挖掘函数 | 行 | `required_citations` 来源 | 与断言层同源？ |
|---|---|---|---|
| `mine_single_chunk_facts` | :397 | 断言 `evidence_text`（8–40 字、块内恰好出现一次） | 是 |
| `mine_direct_relations` | :536 | 边的 `evidence`（`e["assertion_ids"]`） | 是 |
| `mine_multihop` | :619 | 边的 `evidence_assertion_id` 所在块 | 是 |
| `mine_time_conflict` | :830 | 同源 | 是 |
| `mine_alias_questions` | :468 | `_window_text(chunk["text"], alias)` 从**块正文截整句** | **否** |
| `mine_cross_chunk` | :743 | `_window_text(chunk["text"], 实体名)` | **否** |
| `mine_negative_and_unanswerable` | :1069 | `row["sentence"]` | **否** |
| `mine_adversarial` | :1195 | 同类 | **否** |

上四行必然可达，下四行必然不可达，中间没有状态——这就是 5.1 那张表出现
「四类 1.00 / 四类 0.00」断崖的原因。**这是设计产物，不是抽取质量的分布。**

修它有两条路（都还没做，也都不在本次范围内）：

1. **改挖掘路径**：让后四条也从断言层取证据（工程量在 `goldset.py` 内部，
   但要重新出题、重新换签）。
2. **改分母口径**：把「引用覆盖」拆成「同源类」与「非同源类」两套指标分别报，
   不再用一个合计数代表系统能力。

---

## 七、K1 换签记录

`scripts/k1_checks.py` 的 `gold_set_fixed` 门从 v1 换到 v2，四个常量全更新：

```python
GOLD_SET_PATH      = ROOT / "data" / "goldset" / "gold-v2.json"
GOLD_SET_SHA256_16 = "069375f8e355e4b9"
GOLD_SET_BYTES     = 601492
GOLD_SET_TOTAL     = 640
GOLD_SET_QUOTAS    = {single_chunk 128, alias 96, direct_relation 96, multihop 96,
                      cross_chunk 64, time_conflict 64, negative_abstain 64, adversarial 32}
```

换签必须三件事同时成立（缺一不算换签完成）：

1. **v1 一字节未动** —— 由 `tests/test_make_split.py::test_gold_v1_is_untouched` 守。
2. **四个常量与 v2 实测一致** —— 由 `tests/test_k1_checks.py::test_real_gold_set_passes` 守。
3. **有人做过反向实测** —— 见下节；不许只说「改好了」。

新增的门（都是「只改一半」才会踩到的坑）：

- `test_gate_points_at_v2_not_v1`：只改 sha 常量忘了改路径，门会「通过」在 v1 上，
  **绿得毫无意义**。
- `test_v1_half_is_byte_identical_except_split_and_origin`：
  `gold_set_fixed` 只算整文件 sha256，它看得见「文件变了」，看不见「变的是 v1 那半
  还是新增那半」——有人改一条 v1 题又同步更新哈希常量，门是绿的而历史基线已漂移。

反向实测（三条命令级记录在 `%TEMP%\gate0\logs\reverse_gold_v2_gate.txt`）：

| 场景 | 命令 | 结果 |
|---|---|---|
| 改前 | `py -3 -m pytest tests\test_k1_checks.py -q -k GoldSet` | 5 passed, exit 0 |
| sha 末位 `4b9`→`4ba` | 同上 | **1 failed / 4 passed, exit 1**，门自身 `ok=False`、`expected=...4ba`、`actual=...4b9` |
| 恢复 | 同上 | 5 passed, exit 0 |
| 路径指回 v1 | 同上 | **2 failed / 3 passed, exit 1** |
| 恢复 | 同上 | 5 passed, exit 0 |

---

## 八、变更日志

| 日期 | 变更 | 触发 | 影响题数 | 审核人 |
|---|---|---|---|---|
| 2026-11-05 | 新建 `gold-v2.json`（640 题）与 `split-v2.json`；`gold-v1.json` 一字节未动 | P1-a：v1 无 `split` 字段，A 调参等于看答案 | 640（新增 320，其余 320 逐字段沿用 v1） | Lead 已批（m00976） |
| 2026-11-05 | `scripts/make_split.py` 新建 | 让切分规则活在代码里，不只活在文档里 | — | Lead |
| 2026-11-05 | `tests/test_make_split.py` 新建（17 条） | 规则可复现 + v1 防偷改 + 自检能被反向实测 | — | Lead |
| 2026-11-05 | `k1_checks.py` `gold_set_fixed` 换签到 v2（四个常量） | v1 → v2 换签 | — | Lead 已批（含三条硬条件） |
| 2026-11-05 | `tests/test_k1_checks.py` +2 条（`test_gate_points_at_v2_not_v1` / `test_v1_file_still_present_and_untouched`） | 防「只改一半」 | — | Lead |
| 2026-11-05 | `tests/test_suite_hygiene.py` `N` 1550 → 1573 | 新增 17 + 2 + 4（parametrize）题 | — | Lead |
| — | **本次未改任何 gold 题内容** | P1-b 结论修好前禁止改 gold | 0 | — |

---

## 九、教训（写在这里是因为下次还会犯）

**这一节记的不是 gold 的事，是「判据骗人」的事。**

2026-11-05，我第一版 P1-b 报告里写了三个数：「干净分母 229」「0.2227」
「天花板 0.7633，所以 0.95 结构性不可达」。三个数都是从自己的脚本算出来的，
算术没错，**判据错了**：我用的是**双向子串**（证据与 `evidence_text` 互为子串或
区间重叠），而指标的真判据是**单向**（`req.evidence in ev`，见 `ask.py:283-298`）。
判据比指标宽 → 分母变宽、天花板变高 → 结论方向反了。

这与台账里那条老事故**同形**：

> `invalid_cache_report()["total"]` 把 `total=13801 / ok=13801 / invalid=0`
> 报成「13801 条非法记录」。**判据本身在骗人，数字却很好看。**

三次事故的共同形状都是：**检查自己写错，报出来的红（或绿）方向是假的。**
所以本阶段的规矩是：任何门、任何数，**先说清判据是什么、分子分母各自是什么**，
再报数；报数前先找一条能让它变红的路走一遍。
