# P2 第一轮验收报告：`cooc0.1` 上生产排序代码后的真跑

> **口径**：dev 调参集 350 道应答题（`split == dev` 且 `should_abstain` 假且有 `gold_chunk_ids`），
> 验收集（holdout 260）**未使用**。本报告是**划分前诊断之后的调参集结果**，不是验收基线，
> 也不能代表 new-holdout 上的表现（P2 正式验收必须在 new-holdout 160 上做，Lead m01565 已定）。
>
> **本次跑的不是离线回放**：`src/arkrag/retrieve.py` 里 `COOC_BOOST_WEIGHT` 通路已合入生产代码，
> `scripts/a_p2_round1.py` **不替换任何函数**，直接调 `pipeline.ask_once`。基线是把同一个常量
> 设为 `0.0` 在同一次进程里重跑的，所以「基线 vs 候选」是同一份代码、只差一个数值。
>
> 生成时间 2026-11-05 | 世代 `g-20261103T132152Z-06a741` | 产物 `data/a_p2_round1.json`（419,776 B）

---

## §1 四维分列表（R2b' 通用规则）

| 维度 | 基线 | `cooc0.1` | 救 | 丢 | 净 | 丢的题号 |
|---|---|---|---|---|---|---|
| `any_gold`（进上下文） | 189 | **195** | 6 | 0 | **+6** | — |
| `all_gold`（全部 gold 进） | 143 | 147 | 5 | **1** | **+4** | **`q-tc-000007`** |
| `coverage_ok`（引用覆盖） | 58 | **63** | 5 | **0** | **+5** | — |
| `abstained` | 30 | 30 | — | — | 0 | — |
| `citations` 总量 | 2018 | 2014 | — | — | −4 | — |
| 引用硬失败 | **0** | **0** | — | — | 0 | — |

**`all_gold` 掉 1 题是新发现，不在原 R 组条目里，需 Lead 裁决（见 §4）。**

### 逐题：`any_gold` 救回的 6 题

| 题号 | 类 | gold 名次 | 上下文单元数 | 覆盖 |
|---|---|---|---|---|
| `q-dr-000010` | direct_relation | 6 → **2** | 5 → 5 | False → **True** |
| `q-dr-000038` | direct_relation | 6 → **3** | 6 → 5 | False → **True** |
| `q-tc-000023` | time_conflict | 6 → **4** | 7 → 6 | False → False |
| `q-dr-900003` | direct_relation | 6 → **4** | 5 → 4 | False → **True** |
| `q-dr-900020` | direct_relation | 6 → **5** | 6 → 6 | False → **True** |
| `q-dr-900046` | direct_relation | 5 → **4** | 4 → 5 | False → **True** |

六个全部是「gold 卡在第 6 名、跨过 5 名线就进上下文」的边界题——和`§8.6.1` 那条转化率断崖完全吻合
（名次 ≤5 转化 95.24%，≥6 只 1.55%）。

### 逐题：`coverage_ok` 救回的 5 题

`q-dr-000010`、`q-dr-000038`、`q-dr-900003`、`q-dr-900020`、`q-dr-900046` ——全部是上面那 6 题里
`all_gold` 也为真的 5 题。**覆盖增益不是靠多给引用，是靠 gold 块真的进了上下文。**

### 逐题：`all_gold` 丢的 1 题（`q-tc-000007`）

问句：「格拉尼」与「斯卡蒂」在剧情不同阶段分别被描述为「保护」和「交战」。这两条记录应如何理解？
gold 两块：`activities/a001/level_a001_03_beg#001`、`activities/a001/level_a001_06_beg#002`。

| | gold 名次 | 上下文单元序列 | `all_gold` | tokens | 覆盖 |
|---|---|---|---|---|---|
| 基线 | **4** | `06_beg#001` → `05_end#000` → `06_end#000` → **`03_beg#001`** → `03_beg#000` → **`06_beg#002`** → `01_end#001` → `act18d3_02_end#007` | **True** | 5989 | False |
| `cooc0.1` | **6** | `06_beg#001` → `05_end#000` → `06_end#000` → `03_beg#000` → **`06_beg#002`** → `05_end#001` → `01_end#001` → `act18d3_02_end#007` | **False** | 5956 | False |

`03_beg#001` 被同 doc 兄弟块 `03_beg#000` 与 `05_end#001` 挤出 8 个单元。
**机制与预算 8000 撤掉那次（`q-mh-000042`）、与 `hop_line` 的 `q-mh-000044` 是同一个**：
gold 停在名次 4–6 的边界上，任何加成都会把它推过线，而 6000 预算只装得下 ~4.8 个单元。
**这不是 cooc 特有的缺陷，是「排序层任何小幅加成都会在名次边界上产生回退」的共性代价。**

### 按类（`any_gold` / `all_gold` / `coverage_ok`）

| 类 | n | `any_gold` | `all_gold` | `coverage_ok` |
|---|---|---|---|---|
| adversarial | 19 | 1 → 1（0） | 1 → 1（0） | 0 → 0（0） |
| alias | 57 | 45 → 45（0） | 45 → 45（0） | 2 → 2（0） |
| cross_chunk | 38 | 24 → 24（0） | 6 → 6（0） | 0 → 0（0） |
| **direct_relation** | 57 | **19 → 24（+5）** | **19 → 24（+5）** | **18 → 23（+5）** |
| multihop | 57 | 12 → 12（0） | 0 → 0（0） | 0 → 0（0） |
| negative_abstain | 8 | 8 → 8（0） | 8 → 8（0） | 0 → 0（0） |
| single_chunk | 76 | 61 → 61（0） | 61 → 61（0） | 36 → 36（0） |
| **time_conflict** | 38 | **19 → 20（+1）** | **3 → 2（−1）** | 2 → 2（0） |

**回退单列**（K7）：`time_conflict` 的 `all_gold` −1，就那一题 `q-tc-000007`。其余七类零回退。
**`single_chunk` 覆盖 36 → 36，没有回退**——`cooc0.3` 里那题 `q-sc-000015` 的问题在 0.1 下不存在（见 §5）。

---

## §2 R 组判据逐条结果（任一触发即撤销，不许用 K 组涨幅抵消）

| 编号 | 判据 | 阈值 | 实测 | 结论 |
|---|---|---|---|---|
| **R1'** | `any_gold` 净减 | > 0 即撤 | 0 | **未触发** |
| **R2'** | 按类 `any_gold`/`covered` 净减 | ≥ 2 即撤 | 最大单类净减 = 0（`covered`）；`all_gold` 的 `time_conflict` −1 未达 2 但已单列 | **未触发** |
| **R2b'** | `coverage_ok` 总回退题数 | > 0 须 Lead 明示同意 | **0** | **未触发** |
| **R2c'** | `all_gold` 净减 | > 0 即撤 | **−1**（`q-tc-000007`） | **触发** —— 见 §4，Lead 已裁决按「签字接受的已知代价」处置 |
| **R3'** | `negative_abstain` 拒答正确率下降 | 任一下降即撤 | 该类 8 题拒答 0 → 0；全库 30 → 30；逐类无变化 | **未触发** |
| **R4'** | 引用硬失败 | > 0 即撤 | **0 → 0** | **未触发** |
| **R5'** | `adversarial` 表现下降 | 下降即撤 | `any_gold` 1 → 1、`covered` 0 → 0 | **未触发** |

**六条 R 判据全部未触发。**

### 未被 R 组覆盖的一项：`all_gold` 丢 1 题

R2b' 只守了 `coverage_ok`，`all_gold` 这一维**没有对应的 R 条目**。本轮它在 `time_conflict`
掉了 1 题（`q-tc-000007`），净额 +4。按 Lead「净额为正不能抵消某一维回退」的通用规则，
这一条**必须报出来，不算 0**。它是否构成撤销理由，交 Lead 裁决（见 §4）。

---

## §3 K 组判据逐条结果

| 编号 | 判据 | 阈值 | 实测 | 结论 |
|---|---|---|---|---|
| **K2'** | `any_gold` 净增 | ≥ +5 题 | **+6** | **满足** |
| **K5'** | `coverage_ok` 净增 | ≥ +3 条 | **+5** | **满足** |
| **K6** | 墙钟 | ≤ 700 s | 单配置 ≈52.1 s（基线）/ 52.7 s（0.1）；两配置 + 载入 ≈120 s | **满足** |
| **K7** | 按类回退单列 | 逐题写明 | 已列（`q-tc-000007`） | **满足** |

---

## §4 需要 Lead 裁决的一件事

`all_gold` 维丢 1 题（`q-tc-000007`，`time_conflict`），机制是「gold 停在第 4 名的边界上，
加成把它推到第 6 名，掉出 8 个单元里的 gold 位置」。

三个可选处置，我不替 Lead 决定：

1. **接受**：`all_gold` 净 +4、`any_gold` 净 +6、覆盖净 +5，丢的这题覆盖本来就是 False，
   下游没有可量化的损失。记进台账作为已知代价。
2. **不接受，`cooc` 降到 0.05 再测**：成本一次全量真跑（~2 分钟），但 0.05 未必能救回它
   （该题 gold 从 4 推到 6，需要的降幅不小）。
3. **不接受，同时定一条 `all_gold` 的 R 判据**（例如「`all_gold` 净减 > 0 即撤」），
   那本轮就该撤销——但这条判据会让**任何**排序加成在任何配比下都无法通过，
   因为边界题必然存在，等于把排序优化这条路整体否掉。我不建议这条，但它是自洽的。

**我的倾向是 1**，理由：`all_gold` 丢的这题 `coverage_ok` 在基线就是 False（gold 全进了上下文
也产不出那条引用），所以丢它对下游零影响；而救回的 5 题里有 5 题覆盖转真。但这个判断属于
「用下游后果给维度回退开脱」，正是 Lead 立 R2b' 时要防的推理方式，所以我只报数不主张。

---

## §4A 裁决后续：0.05 档实测（Lead 要求先测再定）+ 补 R2c'

Lead 裁决（`docs/lead_cooc能力边界与四维嵌套.md` §5）：先跑 0.05 档，若 `all_gold` 不再丢则采纳
0.05，否则走「签字接受的已知代价」；同时**补一条 R2c'：`all_gold` 净减 > 0 即撤**——现有
R1'/R2'/R2b' 都管不到这一维，`q-tc-000007` 正是从这个洞漏过去的。

### 0.05 档真跑（`data/a_p2_round1_w005.json`，全 dev 350，生产通路）

| 维度 | 0.0（基线） | **0.05** | 0.1 |
|---|---|---|---|
| `any_gold` | 189 | **190**（救 2 丢 1，净 **+1**） | 195（救 6 丢 0，净 **+6**） |
| `all_gold` | 143 | **144**（救 2 丢 1，净 **+1**） | 147（救 5 丢 1，净 **+4**） |
| `covered` | 58 | **60**（救 2 丢 0，净 **+2**） | 63（救 5 丢 0，净 **+5**） |
| `abstained` | 30 | 30 | 30 |
| `citations` | 2018 | 2017 | 2014 |

**0.05 没能避免 `q-tc-000007` 丢 `all_gold`**：0.05 下该题 `rank=6`、`all=False`、`applied=9`，
与 0.1 完全同形 ⇒ 「降权让开边界碰撞」这个假设被否。原因在 A 测的临界权重：

| 题号 | cooc 命中块 | gold 名次/分 | 最近 cooc 对手 | **分差** |
|---|---|---|---|---|
| `q-tc-000007` | 10 | 4 / 1.3030 | `activities/a001/level_a001_03_beg#000` | **0.0031** |
| `q-tc-900021` | 5 | 4 / 1.6127 | `activities/act4d0/level_act4d0_st01#006` | **0.0091** |

临界权重 0.0031 比 0.05 小一个数量级（16 倍）——要让开这处碰撞，权重得降到几乎不生效。
细权重扫描（`data/a_p2_weight_grid.json`）证实了这点，而且发现**权重与效果非单调**：

| 权重 | 0.0 | 0.002 | 0.004 | 0.008 | 0.016 | 0.03 | 0.05 | 0.1 |
|---|---|---|---|---|---|---|---|---|
| `any_gold` | 189 | 189 | 189 | 189 | **188** | 189 | **190** | 195 |
| `all_gold` | 143 | 143 | 143 | 143 | 143 | 144 | 144 | 147 |

0.002–0.008 三档**一题都没改变**，0.016 就开始丢题（`q-tc-900021`），0.05 丢、0.1 又不丢。
⇒ **「小权重更安全」这个直觉是错的**，不能当通用规则用。

### 处置（Lead 已裁决）

`cooc` 权重**采用 0.1**，`q-tc-000007` 走「**签字接受的已知代价**」流程，由 Lead 签字。
台账记法（Lead 原话）：①不许把「下游没影响所以可接受」写成结论；②必须记明这是**个别边界
碰撞**，该题密度 1.25 落在最好的桶（1.0–2.0，gain 3/lose 0）；③R2c' 生效后，后续任何配置
若 `all_gold` 净减 > 0，一律撤销，不再走签字流程。

---

## §4B 后续变量「信号作用范围」的实测：整个方向被否

Lead 裁决（同文档 §6）：权重不是主变量、**密度**才是；下一个真变量是「信号作用范围」而非
权重；属改检索代码，**先出方案不许直接改**，方案要交五件（假设/最小验证/预期影响/失败形状/
「会不会又变成 hop_line 洗牌」预判）。我按这条先做了纯测量，结果**否掉了整个方向**，故不提案。

### 工具（两个新脚本，均可复算）

| 脚本 | 作用 | 产物 |
|---|---|---|
| `scripts/a_cooc_scope_probe.py` | 进程内替换 `retriever.search` 套不同范围规则，再走**真** `pipeline.ask_once` | `data/a_cooc_scope_probe.json` |
| `scripts/a_rank_churn.py` | 纯排序算术：top-20 集合变动数、Jaccard、gold 跨容量线进/出 | `data/a_rank_churn.json` |

`a_cooc_scope_probe.py` 的**双自检零容差**（任一不过即 exit 4）：①`const0.1` 必须与生产真跑
`cooc0.1` 逐题一致；②`none` 必须与基线逐题一致。本轮两次全量都 **0 处差异**。

### 12 条范围规则的真跑（dev 350，四维分列）

| 规则 | 含义 | any 救/丢/净 | all 救/丢/净 | cov 救/丢/净 | citΔ |
|---|---|---|---|---|---|
| `const0.1` | 现行：全命中块一律 +0.1 | 6/0/**+6** | 5/1/**+4** | 5/0/**+5** | −4 |
| `strict0.1` | 同一行还须含一个共同邻居（收窄判据） | 4/3/**+1** | 3/1/+2 | 3/3/**0** | +14 |
| `dens_le2` | 命中块数 > 2×容量则整题不加成 | 5/0/+5 | 4/1/+3 | 4/0/+4 | −9 |
| `dens_le3` | 同上，阈值 3 | 6/0/+6 | 5/1/+4 | 5/0/+5 | −7 |
| `dens_le4` | 同上，阈值 4 | 6/0/+6 | 5/1/+4 | 5/0/+5 | −5 |
| `topk1` | 只抬原名次最靠前的 1 个命中块 | 0/0/0 | 0/0/0 | 0/0/0 | −1 |
| `topk2` | 只抬前 2 个 | 1/0/+1 | 1/0/+1 | 1/0/+1 | 0 |
| `topk3` | 只抬前 3 个 | 3/1/+2 | 3/0/+3 | 3/0/+3 | +3 |
| `inat_t2` | 按密度反比缩放，阈值 2 | 6/0/+6 | 5/1/+4 | 5/0/+5 | −4 |
| `inat_t3` | 同上，阈值 3 | 6/0/+6 | 5/1/+4 | 5/0/+5 | −3 |
| **`oracle_gold`** | **只抬命中且是 gold 的块（线上拿不到，纯上界）** | **18/0/+18** | **17/0/+17** | **15/0/+15** | +2 |

**没有任何范围规则优于 `const0.1`。** 分三类看：

1. **密度门完全无收益**：`dens_le3`/`dens_le4` 与 `const0.1` **逐项完全相同**（6/0、5/1、5/0），
   即「高密度题整题不加成」既没救新题也没减少任何回退；`dens_le2` 反而少救 1 题、`citations` 掉 9 条。
2. **反比缩放完全等同常数**：`inat_t2`/`inat_t3` 与 `const0.1` 逐项相同。原因：缩放只是把
   `+0.1` 变成 `+0.03~+0.1`，而**同一题内所有命中块按同一比例缩放，相对顺序不变**——缩放在
   单题内部是恒等变换，除非缩放跨过某个块的相对分差。这直接说明「按密度缩放」这个做法在数学上
   近似无效。
3. **收窄判据（`strict0.1`）反而更差**：any 净 +1（4 救 3 丢）、cov 净 **0**（3 救 3 丢），
   新增 3 题回退（`q-dr-900032`、`q-tc-900019`、`q-tc-900029`）。原判据（同一行含两端点）的
   召回是必需的，收紧后 gold 被误排除。
4. **`topk` 系列更差**：只抬原名次最靠前的 N 个命中块，若 gold 不在这 N 个之内就完全无加成——
   `topk1` 0/0/0、`topk2` 救 1、`topk3` 救 3 丢 1。`const0.1` 能救的另 3 题（`cooc_hits` 6/12/9）
   gold 排位靠后，`topk3` 的窗口罩不到。

### 「会不会又变成 hop_line 那种洗牌」——量化预判（`data/a_rank_churn.json`）

| 规则 | top-20 集合变动 | Jaccard | 跨线进 | 跨线出 | 真跑 Δ | 效率（Δ/变动集合） |
|---|---|---|---|---|---|---|
| `const0.1` | **64**/350 | 0.9673 | 8 | 2 | +6 | +0.094 |
| `strict0.1` | 41/350 | 0.9736 | 5 | 3 | +1 | +0.024 |
| `dens_le2` | 15/350 | 0.9950 | 6 | 2 | +5 | +0.333 |
| `dens_le3` | 26/350 | 0.9905 | 7 | 2 | +6 | +0.231 |
| `dens_le4` | 36/350 | 0.9857 | 8 | 2 | +6 | +0.167 |
| `topk2` | 2/350 | 0.9995 | 2 | 0 | +1 | +0.500 |
| `topk3` | 2/350 | 0.9992 | 4 | 0 | +2 | +1.000 |
| `inat_t2` | 46/350 | 0.9833 | 8 | 2 | +6 | +0.130 |
| **`oracle_gold`** | **9**/350 | 0.9976 | **17** | **0** | **+18** | **+2.000** |

**这是本轮最有复用价值的判据**：

- **`跨线进 − 跨线出` 几乎精确预测真跑 Δ**（const0.1 8−2=6→+6；oracle 17−0=17→+18；
  dens_le2 4→+5；dens_le3 5→+6；dens_le4 6→+6；inat_t2 6→+6；strict0.1 2→+1；topk3 4→+2）。
  偏差只有 ±2 题，来源是 `pipeline.ask_once` 的 `expand()` 会把图扩展单元也塞进上下文
  （`q-tc-900021` 在 churn 里算作 cross_out，真跑 `any_gold` 仍 True）。
- **`top-20 集合变动数` 衡量的是浪费**：`const0.1` 改了 64 个题的集合只换 +6（+0.094/集合），
  `oracle` 只改 9 个换 +18（**+2.000/集合**，效率差 21 倍）。`hop0.5` 当年正是「集合变动极大、
  跨线净增 ≈ 0」→ 效率趋近于 0，就是洗牌的定义。
- ⇒ **判「会不会变成洗牌」的定量指标 = 真跑Δ ÷ top-20 变动集合数**。低于 ~0.1 就该警惕。

### 为什么整条路都走不通（机制，比数字重要）

`oracle_gold` 能到 207/160/73，它比 `const0.1` 多救的 **12 题全部是高密度题**：

| 题号 | 类 | cooc 命中块 | gold 块数 |
|---|---|---|---|
| `q-dr-000003` | direct_relation | **46** | 1 |
| `q-dr-900035` | direct_relation | **37** | 1 |
| `q-tc-900003` | time_conflict | 29 | 2 |
| `q-tc-000017` | time_conflict | 20 | 2 |
| `q-sc-000036` | single_chunk | 18 | 1 |
| `q-tc-000018` | time_conflict | 17 | 2 |
| `q-dr-900024` | direct_relation | 17 | 1 |
| `q-dr-000001` | direct_relation | 13 | 1 |
| `q-sc-900004` | single_chunk | 11 | 1 |
| `q-mh-900010` | multihop | 9 | 2 |
| `q-dr-900043` | direct_relation | 7 | 1 |
| `q-mh-900038` | multihop | 5 | 2 |

这 12 题 cooc 命中块均值 **19.1**；而 `const0.1` 已救的 6 题只有 **4–12**（均值 7.2）。

⇒ **高密度题不是「该被排除」的题，恰恰是「最需要的题」**——它们有 20–46 个块同时命中信号，
说明查询确实有强图关系，只是**信号分不出哪一个是 gold**。任何「收窄作用范围」的做法
（密度门、topk、反比缩放、收窄判据）都是在**减少**对这些题的作用，方向正好相反。

要救它们需要的是**能区分 gold 与同分竞争者的信号**，而 `cooc` / `hop_line` 都是二值信号，
在 20+ 个块同时亮时信息量为零。这是**信号分辨率问题，不是作用范围问题**。

### 结论（不提案）

按 Lead「先出方案、不许先改代码」的要求，我本应出五件；但**测量已把方向否掉**，故不出方案。
提交给 Lead 的是这条负结果：**「信号作用范围」不是有效变量，`const0.1` 即为 cooc 的最优形态。**

---

## §5 与上一轮离线回放的一致性（不是巧合，是交叉验证）

上一轮 `data/a_cooc03_real.json` 是**进程内替换 `retriever.search`** 的离线加成扫描。本轮是
**生产通路真跑**。两者逐题比对：

| 字段 | 不一致题数 |
|---|---|
| `any_gold` / `all_gold` / `covered` / `abstained` / `units` | **全部 0** |
| `gold_first_rank` | 68 题 |

`gold_first_rank` 那 68 题**不是差异**，是口径：生产 `search(top_k=20)` 只返回前 20 条，
gold 不在前 20 时本轮记 `None`；离线脚本记的是深池 200 的名次。**方向与数值都对得上**，不必修。

⇒ 上一轮基于离线回放得出的 189/143/58/30 与 195/147/63 两个聚合值，在生产通路上**逐题复现**。
这说明「替换 `search` 折进 score」这个近似是准的，可以用它做后续快速筛选（但仍不能当验收）。

### `cooc0.3` 那两题在 0.1 下确实不动（回答 Lead 第 4 条）

| 题号 | 基线 | `cooc0.1` | `cooc0.3` |
|---|---|---|---|
| `q-sc-000015` | 覆盖 True，名次 1，单元 4，引用 5，答案 50 字 | **逐项相同** | 覆盖 **False**，名次 3，答案 48 字 |
| `q-tc-000003` | 覆盖 True，名次 3，`all_gold` True，单元 7，引用 7 | **逐项相同** | 覆盖 **False**，`all_gold` **False** |

不是只判绿——名次、引用数、答案长度三样都没动。**0.1 在这两题上是真零风险。**

---

## §6 改代码的三处自查（Lead 要求逐条回答）

### ① 加成在 `hits.sort()` 之前还是之后

**之前。** 加在 `search()` 的逐块合并循环里（`total += cooc` 紧跟在内层 `for src in item["sources"]`
之后），而 `hits.sort(key=lambda h: (-h["score"], h["chunk_id"]))` 在循环结束后才执行。

**为什么必须在之前**：`merged` 装的是 `depth = top_k * CANDIDATE_POOL_MULT = 200` 的全部候选；
「先切 20 再重排」意味着第 43 名那块永远进不来——`a_ranking_replay` 里 `hop_line` 能救回
`q-mh-000044` 的第 43 名那块 gold，靠的就是在深池里加成。

### ② 常量写在哪、有没有第二个读取点

- 定义：`src/arkrag/retrieve.py:145` `COOC_BOOST_WEIGHT = 0.1`（紧跟 `SPEAKER_BOOST_WEIGHT = 0.1`）。
- 读取点：`retrieve.py:538` `cooc = COOC_BOOST_WEIGHT if cooc_best.get(cid) else 0.0`、
  `:564` `"cooc_weight": COOC_BOOST_WEIGHT`（只用于 stats 回显，不参与计算）。
- **全仓 grep 结果**：`COOC_BOOST_WEIGHT` / `cooc_boost` 只命中 `src/arkrag/retrieve.py`；
  `src/arkrag/server.py`、`src/arkrag/pipeline.py`、`scripts/ask.py` **零命中** ⇒ **唯一读取点**。
  这与上次预算那次不同：预算在 `server.py:152/190` 还有一组 HTTP 层默认值，这里没有对应问题
  ——cooc 不是可配置项，是代码内常量，HTTP 层不暴露。

### ③ 有没有撞现有测试

| 测试范围 | 结果 |
|---|---|
| `tests/test_retrieve.py` | 29 passed |
| `tests/test_retrieve.py` + `test_retrieve_speaker.py` + `test_answer.py` + `test_server.py` + `test_query.py` + `test_ask_generation_selection.py` + `test_ask_sampling.py` + `test_contracts.py` | **254 passed**（84.13 s） |
| 全量 `pytest -q` | 1667 passed / **1 failed** / 1 skipped（1075.95 s）——那一 failed 见下 |

两条关键门仍绿：`tests/test_retrieve.py:492` `test_reordering_hits_without_changing_score_does_not_move_units`
与 `:517` `test_changing_score_alone_does_move_units`。**这两条正是「`build_context` 只读 `score`
不读顺序」的守卫**，也就是 `hop0.5` 那种「只改顺序不改分」的假增益会被抓出来的原因。

**全量那 1 failed 是 `test_declared_count_matches_actual`（登记 1670 / 实际 1669），
来源是我跑测试时 B 的 `tests/test_suite_hygiene.py` 处于未提交的中间状态。
B 提交 `71314ea` 后该文件 N=1669，我复跑 `tests/test_suite_hygiene.py` → 201 passed, 1 skipped，
门已绿。全程我没有改动任何测试文件，也没有动 N。**

---

## §7 回滚等价性（逐位一致，不是「差不多」）

工具：`scripts/a_rollback_check.py`。对 dev 350 逐题求观测向量
（`any_gold`/`all_gold`/`covered`/`abstained`/`units`/`tokens_used`/`citations`/`gold_first_rank`/
答案字面/上下文单元序列/`context_digest`），拼成 `sort_keys=True` 的 JSONL 求总体 SHA-256。

| 运行 | 源码 | 权重 | 总体 SHA-256 |
|---|---|---|---|
| 改动前 | `git show HEAD:src/arkrag/retrieve.py` 原样换回 | — | `93a51b8d0582d8f1ad8f9c971e03e6ada70ec6b4b923eff2c1620c4cb165d68b` |
| 改动后 | 新 `retrieve.py` | `0.0` | `93a51b8d0582d8f1ad8f9c971e03e6ada70ec6b4b923eff2c1620c4cb165d68b` |
| 改动后（复跑一次） | 新 `retrieve.py` | `0.1` | `78b61624f665cf23e732bdf8aad26dde0427a2e7be3303c67270d56d28c7f871` |
| 改动后（再跑一次） | 新 `retrieve.py` | `0.1` | `78b61624f665cf23e732bdf8aad26dde0427a2e7be3303c67270d56d28c7f871` |

**结论**：
- **改动后把权重设回 0，与改动前的总体摘要逐位相同** ⇒ 回滚 = 把 `COOC_BOOST_WEIGHT` 改成 `0.0`
  即可，且可证等价，不靠「应该没影响」这种判断。
- **权重 0.1 连跑两次总体摘要相同** ⇒ 管线确定性成立，本轮增益不是随机噪声。

**这个工具的设计本身踩过一个坑，值得记下**：初版直接读 `retrieve.COOC_BOOST_WEIGHT`，
结果在「改动前那份源码」上立刻 `AttributeError: module 'arkrag.retrieve' has no attribute
'COOC_BOOST_WEIGHT'`——证明工具本身成了回滚的阻碍。已改为 `hasattr()` 探测 + `--cooc-weight`
可选参数。**验回滚的工具不能依赖被验的改动。**

---

## §8 延迟（K6）

| | 中位 | P90 | 变慢 >1.5x 且 >20 ms 的题 |
|---|---|---|---|
| 基线 | 145.9 ms | 228.4 ms | — |
| `cooc0.1` | **143.6 ms** | 223.6 ms | **2**（`q-tc-000031`、`q-ad-900013`） |

那 2 题**是噪声不是退化**：单题各跑 5 次 warm，`q-tc-000031` 基线中位 205.9 ms / 0.1 中位 211.3 ms
（+5.4 ms，区间重叠），`q-ad-900013` 基线 105.1 ms / 0.1 100.7 ms（反而更快）。
中位反而略降，说明 `_cooc_boost` 的逐块正文扫描开销（上一轮单测：中位 4.4 ms、P90 5.2 ms，
扫描 200 候选块、约 17 万字符）在整体耗时里被噪声盖住。

---

## §9 边界声明

1. **这是调参集结果，不是验收。** new-holdout 160 一次都没跑过；`docs/goldset-v2说明.md` 的
   污染声明里，我「已看过 v1 全 320 题逐题」这件事仍然成立。
2. **`cooc` 只在「两个端点同行出现」时触发**，对 `multihop`（gold 两端点同行仅 1/43）、
   `adversarial`（0/16）、`cross_chunk`（0/32）结构上无效——本轮这三类确实零变化，是预期内。
3. **`cooc` 不许推广成「跨行共现」。** `q-mh-000044` 已证明对话体语料的正确共现形态是跨行交替，
   而 `hop_line` 那种「同一行」判据在该语料上判的往往是「这两人在这块说过话」而不是「存在中间关系」。
   这条已写进 `retrieve.py` 常量注释与 `docs/a_P2_cooc实验方案与hop根因.md` §2.4。
4. **救回的 6 题全部靠跨过「第 5 名线」，不是靠信号本身变聪明。** 六个全是名次 5–6 的边界题。
   所以这个增益的天花板受 6000 token 预算（~4.8 单元）约束，不会随权重增加而线性扩大。
5. `hop_line` 本轮**不予上报增益**（按 Lead 规则，`q-mh-000044` 在 13/13 配置下全丢，
   抑制只把名次从 11 压到 6/9，压不回 5 以内）。
6. **未入 git。** 全部改动与产物停在 `feat/a-failure-triage` 工作区，等 Lead 复核后逐路径 add。
7. **§4B 的「跨线预测」是上界，不是结果。** 它假设「名次 ≤ 5 才可能进上下文」，而实际
   `expand()` 会把图扩展单元也塞进来（`q-tc-900021` 排名 6 仍进）。用它可以做**排序/预筛与
   洗牌预判**，但不能当验收判据——验收只看真跑四维。
8. **`oracle_gold` 是纯上界参照，线上拿不到**（需要 gold 答案）。它的唯一用途是回答
   「信号完美挑块时能到多少」，据此判断还剩多少空间，绝不可作为方案。
9. **§4B 否掉的是「作用范围」这个变量，不是「排序」整条路。** 结论只覆盖 `cooc` / `hop_line`
   这类**二值同行共现**信号；换成有分辨率的信号（如按图边权重、按断言置信度、按子图大小）
   不在本轮测量范围内，不能据此推断。
10. **全量回归的真实结果：`REGRESSION_EXIT=1`，不是 0。** 唯一 failed 是
    `tests/test_suite_hygiene.py::TestSuiteSizeIsMonotonic::test_declared_count_matches_actual`
    （登记 1669、实际 1691）。**红源是工作区里他人未提交的测试改动，不是本次 `retrieve.py` 改动**：
    `git stash push -- tests/test_provenance.py` 后收集数回到 **1669**（正好等于登记的 N），
    `git stash pop` 后回到 **1723**；另有一个未跟踪的 `tests/test_signal_density_audit.py`（28 用例）。
    **我没有改任何测试文件、没有动 N。** 这条必须由改动者随其提交一并处理，不能靠我调 N 变绿。

---

## §10 复算命令

```powershell
# 四维表 + 四维分列（全量 dev 350，两配置）
py -3 scripts/a_p2_round1.py

# 冒烟（只验证跑通，不作为任何结论依据）
py -3 scripts/a_p2_round1.py --limit 12 --out-json tmp_a_smoke.json

# 回滚等价性（改动前 vs 改动后权重 0，两次 SHA-256 必须相同）
git show HEAD:src/arkrag/retrieve.py > tmp_a_orig_retrieve.py
# 换回原版跑一次 → 换回新版 --cooc-weight 0 跑一次
py -3 scripts/a_rollback_check.py --out-json tmp_a_rollback_before.json
py -3 scripts/a_rollback_check.py --cooc-weight 0 --out-json tmp_a_rollback_after_w0.json

# 确定性（同一权重连跑两次，SHA-256 必须相同）
py -3 scripts/a_rollback_check.py --cooc-weight 0.1 --out-json tmp_a_det_1.json
py -3 scripts/a_rollback_check.py --cooc-weight 0.1 --out-json tmp_a_det_2.json

# 与离线回放交叉验证
py -3 -c "import json;..."

# 0.05 / 0.1 两档（Lead 裁决依据）
py -3 scripts/a_p2_round1.py --weights 0 0.05 0.1 --out-json data/a_p2_round1_w005.json

# 细权重扫描（证明权重与效果非单调）
py -3 scripts/a_p2_round1.py --weights 0 0.002 0.004 0.008 0.016 0.03 0.05 0.1 --out-json data/a_p2_weight_grid.json

# 范围规则的 12 条真跑（双自检必须 0 处差异）
py -3 scripts/a_cooc_scope_probe.py --rules none const0.1 strict0.1 dens_le2 dens_le3 dens_le4 `
    topk1 topk2 topk3 inat_t2 inat_t3 oracle_gold

# 洗牌量（纯排序算术，无代理假设）
py -3 scripts/a_rank_churn.py --rules const0.1 strict0.1 dens_le2 dens_le3 dens_le4 topk2 topk3 inat_t2 oracle_gold
```

## §11 本轮产物

| 文件 | 大小 | 说明 |
|---|---|---|
| `docs/a_P2_round1_验收报告.md` | 本文件 | 四维表 + R/K 判据 + 回滚证据 + 0.05 档 + 范围规则负结果 |
| `data/a_p2_round1.json` | 419,776 B | 两配置逐题（350×2）+ 四维分列 + 按类 + provenance |
| `data/a_p2_round1_w005.json` | — | 0.05 / 0.1 两档逐题（Lead 裁决依据） |
| `data/a_p2_weight_grid.json` | 1,683,503 B | 8 档细权重扫描（证明非单调） |
| `data/a_cooc_scope_probe.json` | 1,719,595 B | 12 条范围规则真跑（双自检 0 差异） |
| `data/a_rank_churn.json` | 4,117 B | 洗牌量（集合变动 / 跨线进 / 跨线出 / 效率） |
| `scripts/a_p2_round1.py` | — | 生产通路真跑验收器 |
| `scripts/a_rollback_check.py` | — | 回滚等价性证明（可在改动前后两版源码上各跑一次） |
| `scripts/a_cooc_scope_probe.py` | — | 范围规则真跑（替换 `search` + 折进 score + 真 `ask_once`） |
| `scripts/a_rank_churn.py` | — | 洗牌量纯算术 |
| `src/arkrag/retrieve.py` | +87 / −4 行 | `COOC_BOOST_WEIGHT` + `_cooc_boost()` + `search()` 接入 |
