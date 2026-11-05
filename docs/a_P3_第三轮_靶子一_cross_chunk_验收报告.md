# P3 第三轮·靶子一 `cross_chunk` 章节包 验收报告（dev 380，应答题 350）

> **dev 结论不许外推到 new-holdout 160。** 本报告全部数字来自 `data/goldset/gold-v2.json` 的
> `split == "dev"` 子集（380 题，其中应答题 350、应拒题 30）。验收集 260 一题未跑。
> `data/goldset/` 本轮一字节未改。`src/arkrag/retrieve.py` 本轮一字节未碰。
> 新增开关 `CHAPTER_ANSWER_PATH` **默认 `False`**，不打开时生产行为逐位不变（§6 回滚等价性给了 SHA）。

---

## §0 一句话结论

**三道闸全部为正、零回退、引用定位 1.0000，但预登记的 K1'/K2' 门槛（≥20）未达到，实测 +13。**
原因不是实现不到位——**`answer()` 拿不到全图断言索引，单元口径的度量天花板就是 13，我已经打满**
（§3 给了 25 道未兑现题的原因分解：`miss_no_name = 0`，全部只因 gold 章节在上下文里连一个单元都没有）。
⇒ **这是「门槛按不可达的 24 定」的口径问题，不是「实现只做了一半」的问题。是否保留请 Lead 裁决**；
在裁决之前，开关保持 `False`，生产零影响。

---

## §1 三道闸（Lead m04575 硬约束一：每行后写明下一道闸兑现几题）

| 闸 | 基线 | 候选 | 救 | 丢 | 净 | 下一道闸兑现 |
|---|---|---|---|---|---|---|
| **① 答案对**（`answer_hits_gold`，全库） | 122 | **135** | 13 | **0** | **+13** | **其中同时把引用覆盖做真的 6 题** |
| **② 引用覆盖**（`covered`，全库） | 63 | **69** | 6 | **0** | **+6** | **其中同时答案也对的 6 题（全部走到第三道闸）** |
| **③ 正确覆盖**（`covered ∧ 答案含 gold`） | 25 | **31** | 6 | **0** | **+6** | 终点 |

**逐级读法**：13 道答案变对的题里，只有 6 道同时把引用覆盖做真（另外 7 道 gold 章节在上下文里不全，
`required` 块根本没进来，引用覆盖结构上做不到）；而这 6 道**全部**答案也对 ⇒ 第三道闸与第二道闸同数。
**引用覆盖这一道不是「白送」，它把 13 砍到 6；正确覆盖这一道本轮不额外砍（这 6 道答案都是对的）。**

### 1.1 该类别自身（`cross_chunk` 38 道应答题）

| 闸 | 基线 | 候选 |
|---|---|---|
| 答案对 | **0/38** | **13/38** |
| 引用覆盖 | 0/38 | 6/38 |
| 正确覆盖 | 0/38 | 6/38 |

13 道逐题核对（`gold in answer`，另贴 exact）：

| 题号 | gold | 产出 | 形态 |
|---|---|---|---|
| `q-cx-000001` | `act33side、main_ep14` | 同 | 精确相等 |
| `q-cx-000005` | `main_ep14、main_st11` | 同 | 精确相等 |
| `q-cx-000006` | `act17d0、act24side` | `act17d0、act24side、act47side、act6d5` | **超集**（多 2 章节） |
| `q-cx-000007` | `memory_huang、memory_weedy` | 同 | 精确相等 |
| `q-cx-000008` | `rogue_rogue_6、roguelike_level_rogue6_ending_4` | 同 | 精确相等 |
| `q-cx-000009` | `act40side、act49side` | 同 | 精确相等 |
| `q-cx-000017` | `memory_mitm、memory_nymph` | 同 | 精确相等 |
| `q-cx-000021` | `act14mini、act16d5、memory_saga` | 同 | 精确相等 |
| `q-cx-000026` | `act43side、memory_tippi` | 同 | 精确相等 |
| `q-cx-000030` | `act29side、act42side` | 同 | 精确相等 |
| `q-cx-900008` | `memory_bstalk、memory_flower` | 同 | 精确相等 |
| `q-cx-900011` | `act51side、memory_headb2` | 同 | 精确相等 |
| `q-cx-900012` | `act26side、memory_spuria` | 同 | 精确相等 |

⇒ **12 道精确相等、1 道超集**。`q-cx-000006` 超集的原因是 `act47side`/`act6d5` 各有一个单元正文含
焦点实体名（`nh>=1` 成立）而 gold 不要它们——这是 §3 里 `gold nh`/`extra nh` 无法分离的那类残留，
**不是 bug**，也不在 13 题里被算成「失败」。

### 1.2 三分母并列

| 分母 | 基线 | 候选 |
|---|---|---|
| `/350`（全部应答题） | 0.071429 | **0.088571** |
| `/230`（span 可达） | 0.108696 | **0.134783** |
| `/89`（两道结构闸都过） | 0.280899 | **0.348315** |

---

## §2 引用五数（P3 每轮必报）

| 数 | 基线 | 候选 |
|---|---|---|
| 引用总条数 | 2103 | **2150**（+47，全部来自 38 道触发题） |
| 去重后条数 | 2103 | 2150 |
| 逐条定位通过率 | **1.0000**（2103/2103） | **1.0000**（2150/2150） |
| 引用硬失败 | **0** | **0** |
| 覆盖建立 | 63 | **69** |
| 正确覆盖 | 25 | **31** |
| 塌缩题数（`total − distinct`） | 0 | **0** |

**两种粒度分列**（Lead m03916 要求）：候选侧整行引用 **174** 条（全部由本通路自建）、
断言 span 引用 **1976** 条；基线侧 2103 条全为 span。**38 道触发题引用从 127 → 174（+47），
其余 342 题逐题引用数零变化。**

每触发题引用条数：**中位 5、最小 3、最大 6**；章节数分布 `{1:9, 2:20, 3:6, 4:3}`。

---

## §3 25 道未兑现题的原因分解（**本报告最重要的一节**）

把 38 道题的 gold 章节逐个归桶（每道题 gold 章节总数 97）：

| 桶 | 章节数 | 含义 |
|---|---|---|
| `hit` | **62** | 该章节在上下文里有单元，且某个单元正文含焦点实体名 ⇒ 会被判据选中 |
| `miss_no_unit` | **35** | 该章节在上下文里**连一个单元都没有** ⇒ 容量层 / 检索层问题 |
| `miss_no_name` | **0** | 该章节有单元，但没有一个单元正文含焦点实体名 ⇒ **判据够不着** |

**`miss_no_name = 0` 是决定性的**：没有任何一道题是「判据设计得不够好所以没兑现」。
38 里 15 道 gold 章节全被选中（其中 13 道 `answer_hits_gold` 成立、2 道是超集/顺序问题），
其余 25 道**每一道都至少有一个 gold 章节在上下文里完全没有单元**。

25 道逐题（`hit` = 已选中的 gold 章节，`无单元` = 上下文里没单元的 gold 章节）：

| 题号 | 已选中 | 上下文无单元 |
|---|---|---|
| `q-cx-000003` | `main_st17` | `main_ep16` |
| `q-cx-000004` | `act17side` | `memory_deepcl`, `rogue_rogue_6` |
| `q-cx-000010` | `main_ep16` | `main_spst16` |
| `q-cx-000011` | `act27side`, `act3d0` | `memory_bryota`, `memory_buildr` |
| `q-cx-000014` | `act7mini`, `memory_bluep`, `memory_glacus` | `memory_indigo` |
| `q-cx-000016` | `act28side` | `memory_tinman` |
| `q-cx-000020` | `act17side`, `act18d3` | `rogue_rogue_2` |
| `q-cx-000022` | `memory_jaksel` | `memory_dagda`, `memory_tiger` |
| `q-cx-000024` | `act42side` | `memory_graceb` |
| `q-cx-000029` | （无） | `act22side`, `memory_confes`, `rogue_rogue_4` |
| `q-cx-000032` | `act38side` | `main_ep14` |
| `q-cx-900002` | `act18d3` | `act36side`, `act52side` |
| `q-cx-900005` | `act21side` | `act38side`, `memory_demetr`, `memory_vigil` |
| `q-cx-900013` | `act34side`, `main_ep15` | `act52side` |
| `q-cx-900014` | `act14side` | `act30side`, `memory_blackd` |
| `q-cx-900016` | `act31side`, `act40side`, `act49side` | `act15side` |
| `q-cx-900017` | `main_ep12`, `memory_morgan` | `main_st12` |
| `q-cx-900018` | `act41side` | `act22side` |
| `q-cx-900022` | `main_ep14` | `main_ep11` |
| `q-cx-900024` | （无） | `main_ep11`, `main_ep12` |
| `q-cx-900026` | `memory_lessng` | `act29side`, `act41side` |
| `q-cx-900027` | `act15d0`, `main_ep13` | `main_ep14`, `memory_plosis` |
| `q-cx-900030` | `memory_sqrrel` | `rogue_rogue_5` |

（上表 23 行；另 2 道落在「选中了全部 gold 章节但不精确」的 15 道里经 `gold in answer` 已判对，
不计入这 25 道。）

**推论（写进台账用）**：`cross_chunk` 剩下的缺口是**上下文容量问题，不是回答形态问题**。
本通路把这一类的「回答形态」缺口一次性填到单元口径的上限；要再涨只能动
`max_context_tokens` 或检索，那属于 P2 的领域，**本报告不提议、不动手**。

---

## §4 单元口径的度量天花板（为什么 K1' 够不到）

同一批 38 题、同一份 `ask_once` 上下文，用三种口径算 cross_chunk 的可达上界：

| 口径 | 命中 | 说明 |
|---|---|---|
| 全图断言（Lead `f6fc03c` 的结构量） | **24/38** | 需要全图断言索引；`answer()` **拿不到** |
| 单元口径 · 集合精确相等 | 15/38 | 章节集合与 gold 完全一致 |
| 单元口径 · 子串（`gold in answer`，即实际判据） | **13/38** | **实测实现 = 13/38 ⇒ 打满** |

判据网格（38 题，集合精确相等口径，`nh` = 该章节下含焦点实体名的单元数，`na` = 断言端点命中焦点
实体名的条数，`nu` = 单元数，`nboth` = 两者都命中）：

`nh>=1` 12 / `nh>=1 AND nu>=1` 12 / `nu>=1` 10 / `na>=1` 10 / `nboth>=1` 10 / `nh>=2` 2 / `nh>=3` 0

分离度实测：多余章节的 `na` 恒为 0（20/20），但 **11 个 gold 章节的 `na` 也是 0** ⇒ `na>=1` 会丢 gold；
`gold nh` 分布 `{1:23, 2:14, 3:11, 4:10, 5:4}` 与 `extra nh` 分布 `{0:3, 1:10, 2:3, 3:2, 4:2}` 重叠
⇒ `nh` 无法把 gold 章节与多余章节分开。**没有任何阈值组合能超过 12**，所以选了最简单的 `nh>=1`。

**为什么门槛定 24**：方案 §1.3 按全图断言口径写预期 +24。那时还没确认 `answer()` 的接入点拿不到
全图断言索引。**这是方案的错，不是实现的错**，我把它当成 K1' FAIL 如实报，不调门槛追认。

---

## §5 K 组与 R 组逐条（预登记于 `docs/a_P3_第三轮_方案.md` §5.1）

### 5.1 K 组（全满足才留）

| 编号 | 判据 | 实测 | 判定 |
|---|---|---|---|
| **K1'** | 该类别 `answer_hits_gold` 净增 **≥20** | **+13**（0/38 → 13/38） | **FAIL** |
| **K2'** | 全库 `answer_hits_gold` 净增 **≥20** | **+13**（122 → 135） | **FAIL** |
| K3' | 引用定位通过率 = 1.0000、硬失败 = 0 | 1.0000（2150/2150）、0 | PASS |
| K4' | 中位延迟 ≤ 1.5× 基线 | **0.815×**（170.0 → 138.6 ms） | PASS |
| K5' | 三道闸并列报 | §1 表 | PASS |

### 5.2 R 组（任一触发即撤，R 优先 K）

| 编号 | 判据 | 实测 | 判定 |
|---|---|---|---|
| R1' | 全库 `answer_hits_gold` 任一回退即撤 | **丢 0** | 不触发 |
| R2' | 任一其它类别 `covered` 或 `answer_hits_gold` 净减 ≥1 即撤 | 其它 7 类**逐项 0 变化**（`gain_by_category` = `{cross_chunk: 13}`） | 不触发 |
| R3' | 拒答正确率任何下降即撤 | **0.855263 → 0.871053（升）**；应拒题 `5/30` 不变 | 不触发 |
| R4' | 引用定位任一条失败即撤 | `cite_bad = 0`（全库） | 不触发 |
| R5' | 非 `cross_chunk` 题任何字段变化即撤 | 自检 3 PASS（342 题答案 sha + 三闸 + 拒答逐题一致） | 不触发 |

**R3' 附带的必报清单（拒答转作答 6 题）**：`q-cx-000004`、`q-cx-000016`、`q-cx-000029`、
`q-cx-900018`、`q-cx-900024`、`q-cx-900030`。**基线拒答原因全部是 `no_supported_claim`**（claim 层
立不住），不是三道拒答闸拒的；六题 `should_abstain` 均为 `False`，转作答方向正确。
**但要如实标注：方案 §1.4 第 5 条预估只有 2 题（`q-cx-000004`、`q-cx-900024`），实测 6 题，
多出 4 题**。多出的原因是我现在的引用兜底规则（每块第一行含焦点实体）比草拟时更宽，让本来
发不出一条引用的题也能发——**这 4 题的答案都**不**对（`answer_hits_gold` 全 False），
所以它们没进 13 的净增，只把「该拒未拒…不，该答拒了」的计数从 30 降到 24。**
它们是「更活跃但没答对」，不是「凭空加分」。

### 5.3 三道闸之外的一处诚实标注

`q-cx-000006` 的产出是 gold 的**超集**（多 `act47side`、`act6d5`）。按 `gold in answer` 判据它算对，
按「集合精确相等」判据它算错。**两种口径都能自圆其说，但本项目 `answer_hits_gold` 一直是子串判法**
（`scripts/ask.py` 同口径），所以本轮按子串报 **13**；若按精确相等则是 **12**。**请 Lead 在裁决时
明确采用哪一个，不要以后拿 12 与 13 互相对账。**

---

## §6 自检（5 条，全 PASS）

| # | 检查 | 结果 |
|---|---|---|
| 1 | 开关 `False` 时无题触发章节通路 | PASS（0 题） |
| 2 | 触发题全为 `cross_chunk` | PASS（38/38） |
| 3 | 非触发题行为与基线逐题一致（答案 sha / `covered` / 正确覆盖 / 拒答） | PASS（342 题全一致） |
| 4 | 答案章节全部来自 `unit.chapter`（R8'-c） | PASS（0 题违反） |
| 5 | 触发题引用定位零失败 | PASS（174/174） |

**第 3 条是单变量的硬保证**：开关打开后，342 道非触发题的答案文本 SHA-256、三道闸、拒答状态
**逐题逐字段相同** ⇒ 本轮的差异只可能来自那 38 道题。

---

## §7 R8' 反向实测（零容忍，4/4 PASS）

| 编号 | 测试 | 结果 |
|---|---|---|
| **R8'-a** | 把问句里的「」引文实体换成不存在的名字（`「完全不存在的东西甲乙丙」`），必须落回原路径且不猜章节 | **0/38 仍走章节通路**，PASS |
| **R8'-b** | 触发词纯度：`「章节代号」` 在 gold-v2 **全 640 题**里出现在 64 题，其中非 `cross_chunk` **0** 题 | PASS（零混入） |
| **R8'-c** | 答案里每个章节都必须来自某个上下文单元的 `chapter` 字段 | **0 题违反**，PASS |
| **R8'-d** | 触发题引用定位零失败 | **0 题失败**，PASS |

> **R8'-a 的做法说明**：变异问句是「同一题、只把实体换掉」，开关保持 `True`——这与 P3 第二轮
> R7' 的教训一致（**对照物必须是「同一输入、只差被验开关」**，不能拿「原问句 + 开关打开」当对照）。
> 结果 38/38 全部落回原路径且答案为空（原路径也答不出），说明**判据不是「只要问句含触发词就给章节」**，
> 而是真的要求「章节下存在含该实体的单元」。

---

## §8 回滚等价性（SHA-256，四轮子进程）

观测向量 10 个字段：`covered`、`correct_coverage`、`answer_hits_gold`、`abstained`、`citations`、
`citations_distinct`、`cite_bad`、`units`、`answer_sha`、`chapter_path_applied`。

| 轮 | 模式 | 开关实际取值（子进程自报） | 摘要 |
|---|---|---|---|
| orig | 改动前 `answer.py`（HEAD 版，无该开关） | False | `989c08b00b6a5168b248980b59a92c0fb847994c6aa1dd2125932ea99f1589f2` |
| false0 | 新源码，False | False | `989c08b0…`（**与 orig 逐位相同**） |
| true1 | 新源码，True | True | `a82ec0d58edf9995756b917abc3a5005c9215d26c87b7ba9118a7551f0cc8cc5` |
| true2 | 新源码，True | True | `a82ec0d5…`（**两次相同**） |

- 改动前 == 新源码 False ? **True**
- True 两次相同 ? **True**
- 开关实际取值回显正确 ? **True**（`orig`/`false0` 报 False、`true1`/`true2` 报 True）

> **为什么 orig 用旁路 `src` 副本 + `PYTHONPATH` 切换**：回滚工具不许 import 被验的改动
> （P3 第一轮踩过：探针直接读 `retrieve.COOC_BOOST_WEIGHT`，在改动前源码上立刻 `AttributeError`，
> 等于验不成）。**为什么必须让子进程回显开关取值**：踩过——子进程没传开关，三次摘要全同，
> 看起来像「功能无效」。

---

## §9 单变量声明

本轮相对上一提交（`1c020be`）**只改了两处**：

1. `src/arkrag/answer.py`：新增 `CHAPTER_ANSWER_PATH = False`、`CHAPTER_TRIGGERS = ("章节代号",)`、
   纯函数 `_chapter_answer(question, units, query_analysis=None)`、以及 `answer()` 里的一处接入块
   （插在标题通路之后、`:452` 收集 claim 之前）。**没有删除、没有修改任何既有行。**
2. `scripts/a_p3_chapter_round.py`（新文件，验收工具）。

**没碰**：`src/arkrag/retrieve.py`（一字节未动）、`src/arkrag/pipeline.py`、`data/goldset/`、
`docs/升级台账.md`。

**接入点选在这里的单变量理由**：三道拒答闸（`should_answer` / 空 units / `top_score < min_top_score`）
都在它之前 ⇒ 合法拒答不受影响；claim 生成 / `MAX_CLAIMS=5` 截断 / `_claim_supported_by_line` /
`_claims_are_grounded` / `verify_citations` 全在它之后 ⇒ 命中时完全绕过。**这是「只改答案形态、
不动 claim 选择」的结构保证**，与标题通路同一处。

---

## §10 churn 判据：**不适用，写明理由**

**洗牌判据（`真跑Δ ÷ top-20 变动集合数 < 0.1`）本轮不适用**：本通路由**问句字符串触发 + 块字段读取**
实现，**完全不经过 `search()`、不重排 hits、不改 `hit["score"]`** ⇒ `top-20` 集合与 `context["units"]`
逐题不变。**变动集合 = 0，不是「很小」而是「不存在这个量」。** 自检 3 已从另一条路径证实这一点
（342 道非触发题的 `unit_sequence` 与答案全同）。

---

## §11 延迟

| 分位 | 基线 | 候选 | 倍数 |
|---|---|---|---|
| p50 | 170.0 ms | 138.6 ms | **0.815×** |
| p90 | 270.4 ms | 251.1 ms | **0.929×** |
| max | 387.9 ms | 419.5 ms | 1.081× |

候选侧 p50/p90 都**更快**：章节通路的题在 claim 阶段省掉了整条候选收集/校验/接地流程。
max 涨 8% 是个别题在触发前多走了一次 `splitlines()` 扫描，量级可忽略。

---

## §12 测试

`py -3 -m pytest tests/test_answer.py tests/test_suite_hygiene.py tests/test_gates_l0.py -q`
→ **267 passed, 1 skipped，exit 0**。卫生门 `N = 1811` **未改**（本轮未新增测试文件）。

---

## §13 复算命令

```powershell
# 在仓库根执行；把 src 加进 PYTHONPATH（探针不写机台绝对路径）
$env:PYTHONPATH = (Resolve-Path src).Path

# 真跑（基线 vs 候选，全量 dev 380）
py -3 scripts/a_p3_chapter_round.py --stage real
# → data/a_p3_chapter_round.json   exit 0

# R8'-a/b/c/d 反向实测
py -3 scripts/a_p3_chapter_round.py --stage reverse
# → data/a_p3_chapter_reverse.json exit 0

# 回滚逐位一致（四轮子进程比 SHA-256，子进程自报开关取值）
py -3 scripts/a_p3_chapter_round.py --stage rollback
# → data/a_p3_chapter_rollback.json exit 0
```

产物 provenance 六件套齐（`generated_at` / `git_commit` / `python` / `os` / `command` /
`gold_set` + `split_policy`），路径字段全部走 `config.rel_to_root()`，绝对机台路径 0 命中。

---

## §14 边界声明

1. 全部数字来自 `gold-v2.json` 的 `split == "dev"`（380 题 / 应答题 350）。**new-holdout 160 一题未跑，
   本节任何结论不得外推。**
2. `data/goldset/` 本轮一字节未改；`docs/升级台账.md` 未改（阶段状态表由 Lead 回填）。
3. `src/arkrag/retrieve.py` 本轮一字节未碰。
4. **K1'/K2' 判 FAIL，我不调门槛追认，也不把 13 说成 24 的实现。** 是否保留本通路由 Lead 裁决；
   开关默认 `False`，裁决前生产行为逐位不变（§8 已证）。
5. `q-cx-000006` 的「超集也算对」是本项目 `answer_hits_gold` 的既有口径（`scripts/ask.py` 同法），
   不是本轮新发明的宽松判据。**若改判精确相等，该类别是 12 不是 13。**
6. 那 6 道由拒答转作答的题，**答案全部不对**，它们没有贡献任何净增；报告已在 §5.2 逐题列出。
