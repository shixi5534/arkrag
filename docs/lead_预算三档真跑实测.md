# 预算三档真跑（6000 / 7000 / 8000）：容量层是真的，但下游答案质量几乎不动

生成命令：`py -3 scripts/lead_budget_realrun.py --json data/lead_budget_realrun.json`
产物：`data/lead_budget_realrun.json`（逐题行 + 六节汇总 + 判据块 + 七字段 provenance）
世代：`g-20261103T132152Z-06a741`　题库：`data/goldset/gold-v2.json`（sha16 `069375f8e355e4b9`，640 题）
分母：dev 350 应答题（`split == "dev"` 且 `should_abstain == False` 且 `gold_chunk_ids` 非空）
开关状态：`TITLE_ANSWER_PATH` 取仓库默认（False）、`CITATION_WHOLE_LINE` 取仓库默认（False）
单变量：三档只差 `retriever.budget["max_context_tokens"]` 一个整数

---

## 0. 为什么写这份

`docs/lead_预算容量与填充实测.md` 里的预算曲线是**结构量**（直接调 `build_context(token_budget=...)`，绕过了回答层）。
「gold 块进上下文」不等于「答案对」，也不等于「引用了该引的句子」。
所以这次走**生产通路**（`pipeline.ask_once`）真跑三档，把下游后果一并量出来，用来回答一个具体问题：

> 预算 6000 → 8000 到底该不该采纳？

本轮**预登记判据**（写在脚本 docstring 里，跑之前就没改过）：

| 判据 | 门槛 |
|---|---|
| K1（主判据） | `correct_coverage` 净增 ≥ +10 题 |
| K2 | `covered` 净增 ≥ +10 题 |
| K3 | frontier 题（基线 gold 名次 ≤5 却一块都没进上下文）收复 ≥ 6 题 |
| R1 | 任一题 `correct_coverage` 由真变假 → 撤销 |
| R2 | 拒答数任何上升 → 撤销 |
| R3 | 任何一条引用定位失败 → 撤销 |
| R4 | 两道已知非单调题必须单列 |

---

## 1. 三档汇总（dev 350 应答题）

| 预算 | any_gold | all_gold | covered | correct | ansHit | abst | 引用条 | 坏条 | 单元 | 已用tok |
|---|---|---|---|---|---|---|---|---|---|---|
| 6000 | 273 | 147 | 63 | 25 | 122 | 30 | 2014 | 0 | 5.014 | 5701.5 |
| 7000 | 273 | 158 | 71 | 28 | 122 | 26 | 2199 | 0 | 5.76 | 6747.6 |
| 8000 | 273 | 169 | 73 | 29 | 125 | 24 | 2356 | 0 | 6.574 | 7771.8 |

列含义：`covered` = `required_citations` 全部被引用命中（单向子串，同 `scripts/ask.py:283-298`）；`correct` = `covered` ∧ 答案文本含 gold 答案；`ansHit` = 答案含 gold 答案（不管引用）；`abst` = 拒答数；`单元` = 平均进上下文的单元数。

**6000 → 8000 的逐题救 / 丢**：

| 维度 | 救 | 丢 | 净 | 丢的是谁 |
|---|---|---|---|---|
| `all_gold` | 22 | 0 | **+22** | — |
| `covered` | 11 | 1 | **+10** | `q-sc-000028` |
| `correct_coverage` | 4 | 0 | **+4** | — |
| `answer_hits_gold` | 6 | 3 | +3 | `q-al-900035`、`q-mh-000008`、`q-mh-000038` |
| `abstained` | 0 | 6 | −6（少拒 6 题） | — |

`correct_coverage` 救回的 4 题：`q-dr-000001`、`q-dr-900036`、`q-dr-900043`、`q-mh-900009`。
6000 → 7000：`all_gold` 救 13 丢 2（`q-al-000022`、`q-cx-900011`）、`covered` 救 9 丢 1（`q-sc-000028`）、`correct` 救 3 丢 0、`abst` 30 → 26。

**按类（8000 vs 6000，只列有变化的）**：
- `covered`：`direct_relation` 23→26（+3）、`time_conflict` 2→6（+4）、`single_chunk` 36→38（+2）、`multihop` 0→1（+1）
- `correct_coverage`：只有 `direct_relation` 23→26（+3）、`multihop` 0→1（+1）

⇒ **加预算的钱基本花在「覆盖成立」上，答案也跟着对的只有 4 题。** 这与 P3 已确立的结论一致：缺口在回答层（`answer.py` 只会吐「主体 谓词 客体」三元组），不在容量层。

---

## 2. 判据结论

| 判据 | 实测 | 结论 |
|---|---|---|
| **K1（主判据）** | `correct_coverage` 净增 **+4**（门槛 ≥ +10） | **FAIL** |
| K2 | `covered` 净增 **+10**（门槛 ≥ +10） | PASS |
| K3 | frontier 收复 **9/9**（门槛 ≥ 6） | PASS |
| R1 | `correct_coverage` 丢失 **0** 题 | 未触发 |
| R2 | 拒答变化 **−6**（少拒 6 题） | 未触发 |
| R3 | 定位失败 **0** 条 | 未触发 |

**主判据没过。** 按预登记纪律，**预算 8000 不作为「质量改进」采纳**；它最多能算「容量层的结构改进」，且必须同时写明「答案质量只涨 4 题」。

---

## 3. frontier 9 题：7000 就全收复，不是 8000 的功劳

基线 frontier（gold 名次 ≤5 却一块都没进上下文）= **9 题**：
`q-al-000034`、`q-dr-900036`、`q-dr-900043`、`q-mh-900001`、`q-mh-900038`、`q-sc-000036`、`q-sc-000045`、`q-sc-000057`、`q-tc-000018`

| 预算 | 收复 |
|---|---|
| 6000 | **0/9** |
| 7000 | **9/9** |
| 8000 | 9/9 |

⇒ 这 9 题的 token 缺口全在 248–1295 之间，**+1000 token 就够，不需要 +2000**。

但 9/9 进上下文之后下游只兑现了一部分（逐题实测）：

| 题 | any_gold | covered | correct |
|---|---|---|---|
| `q-dr-900036` | False→True | False→**True** | False→**True** |
| `q-dr-900043` | False→True | False→**True** | False→**True** |
| `q-sc-000036` | False→True | False→**True** | False→False |
| `q-sc-000045` | False→True | False→**True** | False→False |
| `q-al-000034`、`q-mh-900001`、`q-mh-900038`、`q-sc-000057`、`q-tc-000018` | False→True | False→False | False→False |

⇒ **9 题进了上下文，只有 4 题发对了引用，只有 2 题答案也正确。** 「进上下文」→「发对引用」→「答案对」是三道各丢一半的闸，任何一道只报上一道的数都会高估收益。这也是 K3 与 K1 结论相反的根源：K3 量的是第一道闸，K1 量的是第三道闸。

---

## 4. 两道非单调题：本轮三档全部没进 top-20

| 题 | 6000 | 7000 | 8000 |
|---|---|---|---|
| `q-mh-000042` | any=0 / all=0 / rank=None / units=5 / 5859t | 同 → units=6 / 6841t | 同 → units=6 / 7771t |
| `q-tc-900022` | any=0 / all=0 / rank=None / units=5 / 5950t | units=5 / 6886t | units=6 / 8000t |

**这两题在本轮口径下三档都没进 top-20**，此前的轨迹「6000 有 → 6400 起全无」属于**离线深池口径**（`top_k=200`），两者不可混引。已写进本节标题，不许只引其中一个。

---

## 5. 本轮修掉的两个自家缺陷（第一版跑出来是错的）

### 5.1 frontier 判据口径混引（第一版 53 题 → 修后 9 题）

第一版用 `required_in_context`（**全部** gold 块都在上下文）当 frontier 判据，跑出 **53 题**，其中大量是 `cross_chunk` 的四块题——那批是「部分层」，不是「一块都没进」。
按 `docs/lead_预算容量与填充实测.md` 的定义（gold 名次 ≤5 却**一块都没进**）重算 = **9 题**，与旧文档逐题一致。

**这一改直接把 K3 从 PASS（14/53）变成 PASS（9/9）**——两个 PASS 的含义完全不同：
- 14/53 = 「多块题里补进了一部分块」，分母混了「部分层」；
- 9/9 = 「名次 ≤5 的那批全部进上下文」，是真正的容量层缺口。

**教训（与项目已有三种形状同族，算新的一种）**：**同一个词（「没进上下文」）在两种口径下指两个集合**（全缺 vs 缺一部分），而两者都能算出「收复 N/M」这种格式正确的数。凡「收复/救回」类判据，必须写明**分母是哪个集合、该集合怎么选的**，且必须与自己上一轮文档里的同名指标逐题对齐。

### 5.2 七字段 provenance 是 `unknown`（第一版）

第一版 `payload["provenance"]` 只是 `provenance.verify({k: payload.get(k) ...})`，而 payload 里根本没有 `code_sha256_16` / `dataset` / `inputs` / `environment` 这几个键 ⇒ 七字段判 **unknown**。
改用 `provenance.collect(generation_id=..., dataset=provenance.dataset_block(gold_path=...), inputs=provenance.inputs_block([...]))` 拼全后 → **pass**，`git_commit=4c1822c5…`、`code_sha256_16=d5c268288d2db716`、`dataset gold=…sha16=069375f8e355e4b9 n=640`、`inputs 2 个文件都已登记 sha16`。

**这正是 `provenance.py` 里写的那个设计意图**：`collect()` 取不到就留空、由 `verify()` 判 unknown；而「自己拼一个 dict 去 verify」等于绕过 `collect()`，缺字段就永远看不出来。**要判「报告可追溯」，必须先把 `collect()` 的产物拿去 `verify()`，不能拿半成品 payload 去 verify。**

---

## 6. 限制

1. 只测 dev 350 应答题，**不外推 new-holdout 160**；本轮所有数都是 dev 观测量，不是验收结论。
2. `TITLE_ANSWER_PATH` 与 `CITATION_WHOLE_LINE` 都取仓库默认（False）——本轮只量预算一个变量。标题通路打开后这三档数字会整体上移，届时需重跑，不能拿今天的数外推。
3. `covered` 与 `correct_coverage` 都是**单向子串**判据，未展开 `acceptable_answers`，绝对值偏低；三档用同一判据，**差值可信**。
4. `p50` 延迟本轮为 101 / 174 / 184 ms，比上一轮（79 / 81 / 82 ms）整体高三倍，**机器上有别的负载**，本轮延迟数不可作性能结论，也不构成 K5 类判据的证据。
5. 预算只测到 8000；10000 以上是结构量已有结论（`any_gold` 231、`all_gold` 182），未做生产通路真跑。
6. R2「拒答任何上升即撤」本轮未触发（拒答反而少 6 题），但**少拒 6 题本身要单独审**：这 6 题是不是本该拒答的（`should_abstain`）？本轮 `load_dev()` 已排除 `should_abstain=True`，所以这 6 题都在应答题里，属「本来该答、之前没答」，方向上不是坏事，但**改变拒答率会影响 `abstain_accuracy` 的分母**，验收时必须重算。

---

## 7. 变更日志

- 2026-11-05：新建。三档生产通路真跑，K1 FAIL / K2 PASS / K3 PASS；修掉 frontier 口径混引（53 → 9）与 provenance 七字段 unknown 两个自家缺陷。
