# cross_chunk 判据分析：gold 口径、通路错因与可机检验收线

- 题目类别：`cross_chunk`（dev 380 内共 38 题，`should_abstain` 全 False）
- 世代：`g-20261103T132152Z-06a741`
- 金标：`data/goldset/gold-v2.json` 的 `split == "dev"` 子集
- 复现脚本：`scripts/diag_cross_judge.py`；产物：`data/diag_cross_judge.json`
- 管线：`load_generation("g-20261103T132152Z-06a741")` -> `build_pipeline(loaded, top_k=20, graph_depth=1)` -> `ask_once`；
  预算由脚本**显式**写入 `pipe["retriever"].budget["max_context_tokens"] = B`，不依赖 `build_pipeline` 默认值。
  （核对：本报告写作时 `retrieve.DEFAULT_BUDGET["max_context_tokens"]` 已被另一处工作改成 **12000**，
  而任务下发时的基线是 6000；正因为默认值处在变动中，本脚本两档都显式赋值，读数不随它漂。）

## 结论摘要（先看这里）

| 问题 | 答案 | 分母 |
|---|---|---|
| 1. gold 是不是「全部章节」 | **是**，但完整口径是 `G_name`（实体关联块 ∩ 正文含规范名），不是问句字面的「正文出现过」 | `38/38 [gold == G_name]`、`37/38 [gold == G_all]`、`12/38 [gold == C_text]` |
| 2. 多列的是真章节还是无关章节 | **两者都不是**：是「字面命中但不属于该实体」的第三类。`unrelated` 0 个、`gold_gap` 0 个 | 多列章节 17 个 @6000、46 个 @12000；分类见下 |
| 3. 该修什么 | **(b) 通路**。判据不改、题型不用换；把「章入选」从字面命中改成图的实体归属 | — |
| 建议判据当前分数 | `12/38 [final_criterion_cross_chunk]` @6000、`11/38 [final_criterion_cross_chunk]` @12000 | 38 |
| 修通路后的同判据分数（可达） | `13/38 [v6_graph_chunks]` @6000、`20/38 [v6_graph_chunks]` @12000（extra 恒 0） | 38 |
| 纯上限（上下文已含 gold 全部章节的题） | `15/38 [ceiling]` @6000、`22/38 [ceiling]` @12000 | 38 |

## 三个口径的定义（后文全部按此命名）

| 记号 | 定义 | 数据来源 |
|---|---|---|
| `gold` | 题目 `answer` 按 `、` 切分出的章节集合（`acceptable_answers` 的 `,` / 空格变体同集合） | gold-v2.json |
| `G_name` | 图实体记录 `entity["chunks"]` 中的块，**只保留正文含该实体规范名（`entity["name"]`）的块**，取其 `chapter` 去重 | graph.json + chunks.jsonl |
| `G_all` | 同上但**不过滤正文**，即实体关联块的全部 `chapter` | graph.json + chunks.jsonl |
| `C_text` | 语料（T1 过滤后 13798 个块）中**任意**块正文含该名字的块的 `chapter` 去重 | chunks.jsonl |

语料章节码总数 494。

重要的口径事实：`src/arkrag/goldset.py::mine_cross_chunk` 的生成判据逐字为
`for cid in e["chunks"]: if not _in_text(chunk, name): continue; by_chapter.setdefault(chunk["chapter"], cid)`，
即 **gold 的构造口径就是 `G_name`**。这决定了下面所有比对的方向。

---

## 问题 1：gold 集合是不是「该实体出现的全部章节」

### 答案

**在 `G_name` 口径下是：`38/38 [gold == G_name]`。**

| 口径 | 命中 | 含义 |
|---|---|---|
| `gold == G_name` | **38/38** | gold 完整等于「实体关联块里正文含规范名的章节」 |
| `gold == G_all` | 37/38 | 不过滤正文时 gold 少 1 章 |
| `gold == C_text` | 12/38 | 「正文出现过该名字」这个口径与 gold 严重不符 |
| `gold ⊆ C_text` | 38/38 | C_text 恒为 gold 的宽松超集 |

所以问题 1 的答案是：**gold 是完整的，但它完整的口径是 `G_name`，不是问句字面上读出来的「正文出现过」（`C_text`）。**
把判据改成「正文出现过的全部章节」是**错的方向** —— 那会让 26/38 题的标准答案从 2~4 章膨胀到十几到几十章。

### 唯一的 `G_all` 反例：q-cx-900016「宁辞秋」

- gold = `G_name` = `{act15side, act31side, act40side, act49side}`（4 章）
- `G_all` = 上述 4 章 + `act11mini`（5 章）
- `extra_in_G_all_not_in_gold` = `["act11mini"]`，`extra_in_G_name_not_in_gold` = `[]`

细节：该实体在图中挂 45 个块，其中 `act11mini` 的 5 个块
`activities/act11mini/level_act11mini_st02#000 / #001 / #002 / #004 / #005`
正文只写「宁小姐」（5 个块逐一核对：`宁小姐 in text == True`、`宁辞秋 in text == False`）。
实体 `aliases` 是空数组，`宁小姐` 实际落在 `pending_aliases` 里且 `review == "pending"`：

```
pending_aliases = [
  {"text": "宁小姐", "source": "upstream", "review": "pending",
   "scenes": ["activities/act11mini/level_act11mini_st02", "activities/act15side/...", ...]},
  {"text": "小尚书", "source": "upstream", "review": "pending",
   "scenes": ["activities/act40side/level_act40side_06_beg"]},
]
```

即：图在构建期按**待审别名**把这块归进了实体（因此 `G_all` 含 `act11mini`），
但 goldset 的 `_in_text(chunk, name)` 只认规范名「宁辞秋」，所以 gold 漏掉 `act11mini`。
注意 `act15side` 的 gold 命中是另有一批正文直接写「宁辞秋」的块，与这条 pending 别名无关。

结论：这是 **gold 少列**（口径更严），不是图错。它同时也是本次诊断中唯一一处「图统计比 gold 更宽」的题；
其余 37 题两口径完全一致。**这条不要拿去做新判据** —— 若按 `G_all` 定义正确答案，第 900016 题的验收标准会与本世代 gold 冲突，
需要在 goldset 生成端决定「别名归并是否计入章节出现」；本报告的建议是保持 `G_name`（与现有 38 题一致）。

### `C_text` 为什么不能用（多列压力实测）

| 题号 | 焦点 | gold 章数 | `C_text` 章数 |
|---|---|---|---|
| q-cx-000004 | 海洋 | 3 | 47 |
| q-cx-900022 | 巫术 | 2 | 32 |
| q-cx-000016 | 银行 | 2 | 24 |
| q-cx-000029 | 收音机 | 3 | 19 |
| q-cx-000032 | 轮胎 | 2 | 17 |

`C_text` 严格大于 `G_name` 的题有 26/38。这些普通名词（海洋 / 银行 / 巫术 / 轮胎 / 收音机 / 动力炉）在剧情里被随口提及，
按「正文出现过」定义就会得到几十个章节，远超 goldset 挖掘时设定的 `_CROSS_CHUNK_MAX_CHAPTERS = 4`。

---

## 问题 2：`_chapter_answer` 精确匹配上不去，多列的是真章节还是无关章节

### 生产判据

`src/arkrag/answer.py::_chapter_answer` 的收录条件（逐字）：

```python
for unit in units:
    chapter = unit.get("chapter")
    if not chapter: continue
    if not any(name in unit["text"] for name in focus_names): continue
    by_chapter.setdefault(chapter, unit)
```

即：**只要上下文里该章节有任意一个块正文含焦点名字，该章节就入选**。这是 `C_text` 口径（在上下文子集上的截断版），
而 gold 是 `G_name` 口径 —— 两者天然不等，这是「多列」的机制性来源。

### 逐题三分类（38 题，两档）

分类规则（脚本内 `extras_detail`，**单元级**判定，不用章节级拼接文本）：
先取该章在上下文里的全部单元，若存在某单元正文里焦点名有**一次不被任何更长实体名区间包含**的独立命中 -> `unattributed_standalone`；
否则若焦点名在该章正文出现过 -> `substring_other_entity`；若该章在 `G_all` 里 -> `alias_only`；若在 `G_name` 里 -> `gold_gap`；都不满足 -> `unrelated`。

| 分类 | B=6000 | B=12000 | 含义 |
|---|---|---|---|
| `unattributed_standalone` | **14** | **42** | 名字**独立**出现在该章正文，但图没把任何一个含该名字的块归给该实体 |
| `substring_other_entity` | **3** | **4** | 该章正文里的「名字」其实是**另一个更长实体名**的子串 |
| `alias_only` | 0 | 0 | 图经别名归并知道它出现，goldset 未计入 |
| `gold_gap` | 0 | 0 | 图统计说该章出现、gold 没列（即 gold 不全） |
| `unrelated` | **0** | **0** | 焦点实体根本与该章无关、名字也没出现 |
| 合计（多列章节数） | 17 | 46 | 上表 14+3 / 42+4 |

**结论（问题 2 的答案）：两类都不是。**
- **不是「gold 没列的真章节」**：`gold_gap` 0 例、`alias_only` 0 例 —— gold 在 `G_name` 口径下完整（见问题 1）。
- **不是「无关章节」**：`unrelated` 0 例 —— 一个都没有出现「名字在该章根本不出现」的情况。
- **是第三类**：名字确实在该章**字面**出现，但**实体归属不成立**。细分两种：
  - `unattributed_standalone`（6000: 14，12000: 42）：说明这个字面串在剧情里是普通名词/说话人标签/泛称，被随口提及，图没把它当该实体的提及；
  - `substring_other_entity`（6000: 3，12000: 4）：字面串根本属于另一个实体。

题级分布：B=6000 共 11 题含多列 —— 9 题只含 `unattributed_standalone`（000004 / 000006 / 000027 / 000029 / 000032 / 900018 / 900021 / 900022 / 900027）、
1 题只含 `substring_other_entity`（900024）、1 题两类都有（900030：act16d5 是 A、memory_nothin 是 S）。
B=12000 共 19 题含多列 —— 16 题只含 `unattributed_standalone`（000001 / 000004 / 000006 / 000009 / 000010 / 000017 / 000020 / 000027 / 000029 / 000032 / 900008 / 900017 / 900018 / 900021 / 900022 / 900027）、
2 题只含 `substring_other_entity`（900024 / 900026）、1 题两类都有（900030）。

### 多列证据原文（B=6000，逐题；原文逐字取自 `chunks.jsonl`）

| 题号 | 焦点 | 多列章节 | 类别 | 命中块与原文行 |
|---|---|---|---|---|
| 000004 | 海洋 | act34side | A | `activities/act34side/level_act34side_08_beg#011`「海流串联起散落在大洋中的每一处巢穴，海嗣源源不断地从海洋各处向陆地汇拢，却也为光的通路画出了曼妙的延长线——」 |
| 000006 | 源石引擎 | act47side | A | `activities/act47side/level_act47side_07_end#002`「高登：那些源石引擎、施术核心都来自他国的军事装备？园区要是因为它们而炸毁，那......」 |
| 000006 | 源石引擎 | act6d5 | A | `activities/act6d5/level_act6d5_st03#004`「魏文月：当然很复杂。光是地表设施的供电系统和源石引擎就足够庞大了。」 |
| 000027 | 动力炉 | act16mini | A | `activities/act16mini/level_act16mini_st01#008`「阿兰娜：然后是备用动力炉管道的安全阀。」 |
| 000029 | 收音机 | act21side | A | `activities/act21side/level_act21side_st03#002`「收音机：我与卡拉奇是同期进入市场监管局的。」（**说话人标签**，不是被提及的实体） |
| 000029 | 收音机 | act38side | A | `activities/act38side/level_act38side_07_end#000`「收音机：亲爱的市民们，中午好。」（同上） |
| 000032 | 轮胎 | act12side | A | `activities/act12side/level_act12side_06_beg#001`「陈忽地压下林雨霞的头，躲开了迎面而来的一个轮胎。」 |
| 000032 | 轮胎 | memory_jnight | A | `obt/memory/story_jnight_1_1#000`「艾沃娜：收货单下面的这张是......轮胎换新申请单？」 |
| 900018 | 雄辩家 | main_ep09 | A | `obt/main/level_main_09-17_end#004`「“雄辩家”：......萨科塔。」（**说话人标签**） |
| 900021 | 法伦 | memory_durnar | A | `obt/memory/story_durnar_1_1#000`「坚雷：格瑞斯、法伦、杜宾，还有其他人。」 |
| 900022 | 巫术 | main_ep13 | A | `obt/main/level_main_13-03_beg#005`「埃芒加德：这甚至不是巫术......绝不是巫术，任何源石技艺都没办法达到这样的效果。唔，也不好说，但至少得是个历史学家。」 |
| 900024 | 帕特 | act42side | S | `activities/act42side/level_act42side_02_end#001`「帕特里奇昂：牧首，您今天叫我来是为了？」（正文全是**帕特里奇昂**，另一实体） |
| 900024 | 帕特 | memory_phenxi | S | `obt/memory/story_phenxi_1_1#002`「帕特里奇昂：不许再说了！以前的小菲可不会说这种话！你父母把你托付给我的时候可没允许你这么没礼貌！」 |
| 900027 | 炎魔 | act19side | A | `activities/act19side/level_act19side_03_end#006`「醉醺醺的男人：“炎魔”实验的最大资助者是谁？洛肯水箱的投资人又是谁？」 |
| 900027 | 炎魔 | rogue_rogue_4 | A | `obt/rogue/rogue_4/endbook/endbook_rogue_4_4_2#001`「“啊，霸迩萨想解释一下，他后来一直为他的冲动后悔，他希望来自他乡的同胞不要把他当成一个不知改悔、善妒又暴躁的炎魔——但我们实话实说吧，他一直就是，你没必要去挽救他的名望——”」 |
| 900030 | 勾吴 | act16d5 | A | `activities/act16d5/level_act16d5_04_beg#002`「嵯峨：想当初，小僧只是途经大炎勾吴地界，突发奇想，想去拜访一下那位大家后人。」（**这一句是让该章被判为 A 的独立命中**；同章的 `level_act16d5_06_end#003`、`#004` 两个单元里「勾吴」全部包在「勾吴城」内，`standalone=False`） |
| 900030 | 勾吴 | memory_nothin | S | `obt/memory/story_nothin_1_1#000`「街头混混：我记得你说，你已经去过勾吴城里面了？」（该单元 `standalone=False`） |

上表只列了 B=6000 的 17 个多列章节（B=12000 有 46 个，同理）。
**两档合计 63 个多列章节，`unit_entities_has_focus` 无一例外全是 False** —— 也就是说生产通路选中的这批块，
`build_context` 自己填的 `unit["entities"]` 里**一个都不含焦点实体**。这是「通路用的是字面而不是归属」的直接证据。

### 反向：漏列完全由上下文容量造成

| 量 | B=6000 | B=12000 |
|---|---|---|
| `ceiling_all_gold_chapters_in_context` | **15/38** | **22/38** |
| gold 章节可达率 | 62/97 (0.6392) | 74/97 (0.7629) |
| 被上下文挡住的题 | 23/38 | 16/38 |
| `superset` | 15/38 | 22/38 |
| 多列章节中「上下文里已有该章单元」 | 17/17 | 46/46 |
| 多列章节中「上下文里根本没有该章」 | 0/17 | 0/46 |

两组关键相等：**`superset` 集合与 `ceiling` 集合逐题完全重合**（6000 都是 15 题，12000 都是 22 题，
两方向的对称差为空）。含义是：凡上下文里能凑齐 gold 全部章节的题，通路**已经**全列出来了（它从不漏列可达的 gold 章），
它唯一的失败是**多列**。反之 23/16 题的漏列是上下文容量，与判据无关。

所以在 6000 档：`12/38` 精确、`15/38` 超集 —— **差的那 3 题（000006 / 000027 / 900021）纯粹是被多列打掉的**。
12000 档：`11/38` 精确、`22/38` 超集 —— **差的那 11 题（000001 / 000006 / 000009 / 000017 / 000020 / 000027 / 900008 / 900017 / 900018 / 900021 / 900030）纯粹是被多列打掉的**。
这就是「预算越大、多列越多、精确几乎不涨」的机制：预算放宽只抬高天花板（15 -> 22），不改变通路的多列倾向，
而多列倾向随上下文变大**同比**增长（mean_extra 0.4474 -> 1.2105）。

---

## 问题 3：该修判据、通路、还是题型

### 答案：**(b) 通路**。判据不需要放宽，题型也不需要移出抽取式回答器。

分量说明：

- **(a) 判据 —— 不改。** gold 在 `G_name` 口径下 `38/38` 完整；`G_all` 只有 1 处小数（q-cx-900016），
  且那一处是 gold 更严而非 gold 不全。把判据改成「正文出现过的全部章节」（`C_text`）会让 26/38 题的标准答案膨胀十几倍，反而使判据失去意义。
  现行判据本应就是**双向集合相等**（既有 `proposed_cross_chunk` 已是），不需要改；
  真正需要落地的是把它写成**正式验收项**（见下节）。附带发现：单边判据可作弊已被证伪 ——
  把语料全部 494 章都列出来，`superset` 判据 `38/38` 通过；只列 gold 中任意一章，`subset` 判据 `38/38` 通过。
  两者在 `set_equal` 下都是 `0/38`。所以**超集判据不能作为验收指标**（它与「全列 494 章」退化解不可区分）。
- **(b) 通路 —— 必须改。** `_chapter_answer` 用 `name in unit["text"]` 判定该章入选，等于在上下文子集上跑 `C_text`，
  而 gold 是 `G_name`。证据是 17/17 与 46/46 的多列章节全部 `unit_entities_has_focus == False`，
  即被选中的块在图的归属里完全不含该实体。修法是把「章入选」的判据从**字面命中**换成**图的实体归属**：
  该章入选当且仅当上下文中存在属于该实体的块（`chunk_id ∈ entity["chunks"]`）。
  这需要 `_chapter_answer` 能拿到图实体记录 —— 目前它只拿 `query_analysis["entity_candidates"]` 的 name，
  拿不到 `entity["chunks"]`，这是一个接口缺口。
- **(c) 题型不该由抽取式承担 —— 不成立。** 因为漏列（23/38 @6000、16/38 @12000）**全部**可解释为上下文容量（`superset` == `ceiling`），
  不是抽取式方法本身的表达力不足；而多列（11 题 / 19 题）是判据口径问题，可修。两类问题都不是「必须换生成式模型」才能解决的。

### 通路改造后的可达上限（证明瓶颈在哪）

脚本在同一批上下文单元上重放了 10 档更严/更松的判据（`per_budget[B]["variants"]`）：

| 变体 | 判据内容 | 6000 exact | 12000 exact | 6000 mean_extra | 12000 mean_extra |
|---|---|---|---|---|---|
| `v0_text` | **生产现状**：该章任一单元正文含名字 | 12/38 | 11/38 | 0.4474 | 1.2105 |
| `v1_text_attributed` | 加：该单元 `unit["entities"]` 含焦点名 | 10/38 | 13/38 | 0.0000 | 0.0000 |
| `v2_text_assertion` | 加：该单元断言 subject/object 命中 | 10/38 | 13/38 | 0.0000 | 0.0000 |
| `v3_attributed_only` | 只要求归属，不要求正文 | 10/38 | 13/38 | 0.0000 | 0.0000 |
| `v4_text_standalone` | 名字须有独立命中（排除更长实体子串） | 12/38 | 11/38 | 0.3684 | 1.1053 |
| `v5_text_standalone_attributed` | v4 + 归属 | 10/38 | 13/38 | 0.0000 | 0.0000 |
| `v6_graph_chunks` | 该章入选当且仅当有块 `chunk_id ∈ entity["chunks"]` | 13/38 | 20/38 | 0.0000 | 0.0000 |
| `v7_graph_chunks_named` | v6 + 该块正文含名字 | 13/38 | 20/38 | 0.0000 | 0.0000 |
| `v8_graph_chapters_in_ctx` | `G_name ∩ 上下文可达章节`（纯集合运算，上限） | **15/38** | **22/38** | 0.0000 | 0.0000 |
| `v9_graph_chapters_in_ctx_named` | v8 + 逐章引用支撑 | **15/38** | **22/38** | 0.0000 | 0.0000 |

三点必须诚实说明：

1. 所有归属类变体（v1 / v2 / v3 / v5 / v6 / v7 / v8 / v9）的 `mean_extra` 都是 **恰好 0.0000**（两档都是）。
   即「多列」是**完全可消除**的，不是抽取式方法的固有代价。
2. `v8` / `v9` 的 exact 题集合与 `ceiling` 题集合**逐题相等**（6000 15 题、12000 22 题，对称差为空）。
   说明只要通路改用图的章节归属，剩下的唯一瓶颈就是上下文容量。
3. **但 `v8`/`v9` 是上限证明，不是可直接交付的实现。** 它直接复用了 `G_name` 的构造口径（实体 chunks + 正文含名）。
   `G_name` 由 `graph.json` 的实体记录与 `chunks.jsonl` 正文算出，生产期同样可得、不读 gold，
   但这条路等于把 goldset 的挖掘规则搬进回答器；工程上更合理的落点是 `v6`（**13/38 / 20/38**，extra 恒 0）——
   只要求「该块在图中归属该实体」，不要求正文含规范名，语义上独立于 goldset 的 `_in_text`。
   `v6` 与 `v8` 的差是：6000 档 2 题（000009 古寺、000027 动力炉），12000 档 2 题（000020 恐鱼、900030 勾吴）。
   这 4 题的情形是「gold 该章在上下文里已有单元，但该单元不在该实体的 `chunks` 里」——
   `v8` 只做章节级交集（`G_name ∩ 上下文可达章节`），`v6` 还要块级归属，所以 `v6` 在这些题上少列一章。
   换句话说 `v8` 的上限里有一部分是**章节级**宽松换来的，块级归属的保守代价是每档各 2 题（合计 4 题）。

---

## 建议的判据（可直接翻译成 assert）

**名称**：`final_criterion_cross_chunk`

**中文定义**（五项，全部必须满足）：

1. 该题**未拒答**（`answer["abstained"] is False`）；
2. 答案切分出的章节集合 `S` **非空**；
3. `S` 与 gold 章节集合**双向相等**（`S == gold`，不是超集也不是子集）；
4. `S` 中**每一章**至少有一条引用，其 `chunk_chapter(cite["chunk_id"]) == 该章`；
5. `S` 中每一章的那条引用，`chunk_text(cite["chunk_id"])[cite["start"]:cite["end"]] == cite["evidence"]`（偏移自检通过）
   且 `cite["evidence"]` 含该题焦点的实体名（引用行必须真的在讲这个实体，不是随手一行）。

附加格式约束：`S` 的每个元素必须是合法章节码（`chapter_code_set` 的成员），不接受格式化垃圾 token。

**可机检伪码**（`item` 是 gold 条目，`gold_set` 由 gold 的 `answer` 字段切出；与 `data/diag_cross_judge.json` 里 `criterion_final.machine_checkable` 一致）：

```python
gold_set   = set(split_chapters(item["answer"]))      # gold 的章节集合，题外常量
focus_name = item_focus_name                          # 问句「」里的实体名

ans = ask_once(pipeline, item["question"])["answer"]
assert ans["abstained"] is False                      # (1) 未拒答
S = {t for t in re.split(r"[、,，/;\s]+", ans["answer"]) if t}
assert S != set()                                     # (2) 非空
assert S == gold_set                                  # (3) 双向集合相等
assert S <= chapter_code_set                          # 格式：每个 token 都是合法章节码
for ch in S:                                          # (4)(5) 逐章引用支撑
    ok = [c for c in ans["citations"]
          if chunk_chapter[c["chunk_id"]] == ch
          and chunk_text(c["chunk_id"])[c["start"]:c["end"]] == c["evidence"]
          and focus_name in c["evidence"]]
    assert ok, f"chapter {ch} has no verified citation"
```

**为什么不能靠多列通过**（三条反证，用 gold 自身穷举，非实测产出）：

- 列全部 494 个章节码：`S ⊋ gold`，第 (3) 项挡。实测该退化解在 `superset` 判据下 `38/38` 通过、在 `set_equal` 下 `0/38`。
- 只列 gold 中任意一章：`S ⊊ gold`，第 (3) 项挡。实测该退化解在 `subset` 判据下 `38/38` 通过、在 `set_equal` 下 `0/38`。
- 想靠凭空的章节凑数：每章都必须有通过偏移自检、且含焦点名的引用，第 (4)(5) 项挡；
  而 ref 只能来自真实块，`chunk_chapter` 由块记录直接读出，无法伪造。

**为什么不能靠拒答通过**：第 (1)(2) 项要求未拒答且非空；38 题 `should_abstain` 全为 False。

---

## 该判据现在的分数

| 预算 `max_context_tokens` | 分数 | 备注 |
|---|---|---|
| **6000** | **12/38** `[final_criterion_cross_chunk]` | 与 `set_equal` 同分：`equal_and_cite_each` 12/38、`equal_and_cite_line_has_focus` 12/38、`super`15/38 |
| **12000** | **11/38** `[final_criterion_cross_chunk]` | 同形：`super` 22/38 |

引用支撑条件在两档下**不额外扣分**（`proposed` 与 `set_equal` 同分 12/38、11/38），说明当前失败**只**来自集合不等，不来自引用缺失。

对照参考（同一判据、同一上下文，只换通路判据口径）：

| 口径 | 6000 | 12000 |
|---|---|---|
| 生产现状 `v0_text`（= 本判据下的实际得分） | 12/38 | 11/38 |
| 通路改为图归属 `v6_graph_chunks`（建议落点） | 13/38 | 20/38 |
| `v8/v9` 上限（`G_name ∩ 上下文可达`） | 15/38 | 22/38 |
| `ceiling`（上下文里能凑齐 gold 全部章节的题数） | 15/38 | 22/38 |
| 超集判据（**不建议**，可被退化解通过） | 15/38 | 22/38 |

注意 `ceiling` 与 `superset` 同值、且 `v8/v9` 与二者逐题相等，这不是巧合：
`superset` 判据的分数恰好等于上下文天花板，所以「超集」指标**无法区分**「通路已经做到最好」和「通路在瞎列」
（生产现状 `superset` 也是 15/38 与 22/38，但它只精确 12/38 与 11/38）。**必须用双向集合相等**。

---

## 关键数字一览（分母均为 38 题）

| 指标 | B=6000 | B=12000 |
|---|---|---|
| `gold == G_name` | 38/38 | 38/38 |
| `gold == G_all` | 37/38 | 37/38 |
| `gold == C_text` | 12/38 | 12/38 |
| `set_equal` | 12/38 | 11/38 |
| `superset` | 15/38 | 22/38 |
| `subset` | 27/38 | 19/38 |
| `final_criterion_cross_chunk` | **12/38** | **11/38** |
| `abstained` | 0/38 | 0/38 |
| `chapter_path_applied` | 38/38 | 38/38 |
| `answer_token_not_a_chapter_code` | 0/38 | 0/38 |
| `ceiling_all_gold_chapters_in_context` | 15/38 | 22/38 |
| gold 章节可达率 | 62/97 | 74/97 |
| mean_listed | 2.0789 | 3.1579 |
| mean_extra | 0.4474 | 1.2105 |
| mean_miss | 0.9211 | 0.6053 |
| 有 extra 的题 | 11/38 | 19/38 |
| 有 miss 的题 | 23/38 | 16/38 |
| 列出的章节数 > 4（超挖掘上限）的题 | 0/38 | 6/38 |

最后一行的 6 题是 000004（列 5）/ 000006（列 8）/ 000032（列 7）/ 900008（列 5）/ 900027（列 5）/ 900030（列 6）。
goldset 出题时限制 `_CROSS_CHUNK_MAX_CHAPTERS = 4`，所以「列出 8 个章节」这种输出在任何情况下都不可能是正确答案 ——
这是「输出量本身超出题型物理上限」的独立硬证据，可用于回归时的快速体检。

---

## 逐题清单

`listed` = 回答实际列出的章节；`extra` 后缀 `A` = `unattributed_standalone`，`S` = `substring_other_entity`；
`final` = 是否通过本报告的 `final_criterion_cross_chunk`。

### B = 6000

| 题号 | 焦点 | gold | listed | extra | miss | final |
|---|---|---|---|---|---|---|
| 000001 | 以勒什 | act33side、main_ep14 | act33side、main_ep14 | - | - | PASS |
| 000003 | 米拉娜 | main_ep16、main_st17 | main_st17 | - | main_ep16 | fail |
| 000004 | 海洋 | act17side、memory_deepcl、rogue_rogue_6 | act17side、act34side | act34side(A) | memory_deepcl、rogue_rogue_6 | fail |
| 000005 | 维多利亚议会 | main_ep14、main_st11 | main_ep14、main_st11 | - | - | PASS |
| 000006 | 源石引擎 | act17d0、act24side | act17d0、act24side、act47side、act6d5 | act47side(A)、act6d5(A) | - | fail |
| 000007 | 蓄水炮 | memory_huang、memory_weedy | memory_huang、memory_weedy | - | - | PASS |
| 000008 | 卡萨斯 | rogue_rogue_6、roguelike_level_rogue6_ending_4 | rogue_rogue_6、roguelike_level_rogue6_ending_4 | - | - | PASS |
| 000009 | 古寺 | act40side、act49side | act40side、act49side | - | - | PASS |
| 000010 | 乌卡边境 | main_ep16、main_spst16 | main_ep16 | - | main_spst16 | fail |
| 000011 | 汐斯塔火山 | act27side、act3d0、memory_bryota、memory_buildr | act27side、act3d0 | - | memory_bryota、memory_buildr | fail |
| 000014 | 格劳克斯 | act7mini、memory_bluep、memory_glacus、memory_indigo | act7mini、memory_bluep、memory_glacus | - | memory_indigo | fail |
| 000016 | 银行 | act28side、memory_tinman | act28side | - | memory_tinman | fail |
| 000017 | 识字班 | memory_mitm、memory_nymph | memory_mitm、memory_nymph | - | - | PASS |
| 000020 | 恐鱼 | act17side、act18d3、rogue_rogue_2 | act17side、act18d3 | - | rogue_rogue_2 | fail |
| 000021 | 嵯峨 | act14mini、act16d5、memory_saga | act14mini、act16d5、memory_saga | - | - | PASS |
| 000022 | 杰克 | memory_dagda、memory_jaksel、memory_tiger | memory_jaksel | - | memory_dagda、memory_tiger | fail |
| 000024 | 潘格尼尼 | act42side、memory_graceb | act42side | - | memory_graceb | fail |
| 000026 | 蒂比 | act43side、memory_tippi | act43side、memory_tippi | - | - | PASS |
| 000027 | 动力炉 | act11d0、act28side | act11d0、act16mini、act28side | act16mini(A) | - | fail |
| 000029 | 收音机 | act22side、memory_confes、rogue_rogue_4 | act21side、act38side | act21side(A)、act38side(A) | act22side、memory_confes、rogue_rogue_4 | fail |
| 000030 | 荒域 | act29side、act42side | act29side、act42side | - | - | PASS |
| 000032 | 轮胎 | act38side、main_ep14 | act12side、act38side、memory_jnight | act12side(A)、memory_jnight(A) | main_ep14 | fail |
| 900002 | 安妮塔 | act18d3、act36side、act52side | act18d3 | - | act36side、act52side | fail |
| 900005 | 贝纳尔多 | act21side、act38side、memory_demetr、memory_vigil | act21side | - | act38side、memory_demetr、memory_vigil | fail |
| 900008 | 豆豆 | memory_bstalk、memory_flower | memory_bstalk、memory_flower | - | - | PASS |
| 900011 | 学生联合大会 | act51side、memory_headb2 | act51side、memory_headb2 | - | - | PASS |
| 900012 | 斯普莉雅 | act26side、memory_spuria | act26side、memory_spuria | - | - | PASS |
| 900013 | 技术院 | act34side、act52side、main_ep15 | act34side、main_ep15 | - | act52side | fail |
| 900014 | 图里卡姆 | act14side、act30side、memory_blackd | act14side | - | act30side、memory_blackd | fail |
| 900016 | 宁辞秋 | act15side、act31side、act40side、act49side | act31side、act40side、act49side | - | act15side | fail |
| 900017 | 日落街酒店 | main_ep12、main_st12、memory_morgan | main_ep12、memory_morgan | - | main_st12 | fail |
| 900018 | 雄辩家 | act22side、act41side | act41side、main_ep09 | main_ep09(A) | act22side | fail |
| 900021 | 法伦 | act7d5、memory_blitz、memory_melan | act7d5、memory_blitz、memory_durnar、memory_melan | memory_durnar(A) | - | fail |
| 900022 | 巫术 | main_ep11、main_ep14 | main_ep13、main_ep14 | main_ep13(A) | main_ep11 | fail |
| 900024 | 帕特 | main_ep11、main_ep12 | act42side、memory_phenxi | act42side(S)、memory_phenxi(S) | main_ep11、main_ep12 | fail |
| 900026 | 蒂姆 | act29side、act41side、memory_lessng | memory_lessng | - | act29side、act41side | fail |
| 900027 | 炎魔 | act15d0、main_ep13、main_ep14、memory_plosis | act15d0、act19side、main_ep13、rogue_rogue_4 | act19side(A)、rogue_rogue_4(A) | main_ep14、memory_plosis | fail |
| 900030 | 勾吴 | memory_sqrrel、rogue_rogue_5 | act16d5、memory_nothin、memory_sqrrel | act16d5(A)、memory_nothin(S) | rogue_rogue_5 | fail |

### B = 12000

| 题号 | 焦点 | gold | listed | extra | miss | final |
|---|---|---|---|---|---|---|
| 000001 | 以勒什 | act33side、main_ep14 | act17mini、act33side、main_ep14 | act17mini(A) | - | fail |
| 000003 | 米拉娜 | main_ep16、main_st17 | main_ep16、main_st17 | - | - | PASS |
| 000004 | 海洋 | act17side、memory_deepcl、rogue_rogue_6 | act17side、act34side、act39side、rogue_rogue_2、rogue_rogue_5 | act34side(A)、act39side(A)、rogue_rogue_2(A)、rogue_rogue_5(A) | memory_deepcl、rogue_rogue_6 | fail |
| 000005 | 维多利亚议会 | main_ep14、main_st11 | main_ep14、main_st11 | - | - | PASS |
| 000006 | 源石引擎 | act17d0、act24side | act13d5、act17d0、act19side、act24side、act38side、act47side、act6d5、main_ep16 | act13d5(A)、act19side(A)、act38side(A)、act47side(A)、act6d5(A)、main_ep16(A) | - | fail |
| 000007 | 蓄水炮 | memory_huang、memory_weedy | memory_huang、memory_weedy | - | - | PASS |
| 000008 | 卡萨斯 | rogue_rogue_6、roguelike_level_rogue6_ending_4 | rogue_rogue_6、roguelike_level_rogue6_ending_4 | - | - | PASS |
| 000009 | 古寺 | act40side、act49side | act40side、act49side、rogue_rogue_5 | rogue_rogue_5(A) | - | fail |
| 000010 | 乌卡边境 | main_ep16、main_spst16 | main_ep16、main_st17 | main_st17(A) | main_spst16 | fail |
| 000011 | 汐斯塔火山 | act27side、act3d0、memory_bryota、memory_buildr | act27side、act3d0、memory_buildr | - | memory_bryota | fail |
| 000014 | 格劳克斯 | act7mini、memory_bluep、memory_glacus、memory_indigo | act7mini、memory_bluep、memory_glacus | - | memory_indigo | fail |
| 000016 | 银行 | act28side、memory_tinman | act28side | - | memory_tinman | fail |
| 000017 | 识字班 | memory_mitm、memory_nymph | main_ep13、memory_mitm、memory_nymph | main_ep13(A) | - | fail |
| 000020 | 恐鱼 | act17side、act18d3、rogue_rogue_2 | act17side、act18d3、act52side、rogue_rogue_2 | act52side(A) | - | fail |
| 000021 | 嵯峨 | act14mini、act16d5、memory_saga | act14mini、act16d5、memory_saga | - | - | PASS |
| 000022 | 杰克 | memory_dagda、memory_jaksel、memory_tiger | memory_jaksel、memory_tiger | - | memory_dagda | fail |
| 000024 | 潘格尼尼 | act42side、memory_graceb | act42side | - | memory_graceb | fail |
| 000026 | 蒂比 | act43side、memory_tippi | act43side、memory_tippi | - | - | PASS |
| 000027 | 动力炉 | act11d0、act28side | act11d0、act16mini、act28side、rogue_rogue_4 | act16mini(A)、rogue_rogue_4(A) | - | fail |
| 000029 | 收音机 | act22side、memory_confes、rogue_rogue_4 | act21mini、act21side、act38side、memory_haak | act21mini(A)、act21side(A)、act38side(A)、memory_haak(A) | act22side、memory_confes、rogue_rogue_4 | fail |
| 000030 | 荒域 | act29side、act42side | act29side、act42side | - | - | PASS |
| 000032 | 轮胎 | act38side、main_ep14 | act12side、act32side、act38side、act45side、memory_coldst、memory_jnight、memory_wscoot | act12side(A)、act32side(A)、act45side(A)、memory_coldst(A)、memory_jnight(A)、memory_wscoot(A) | main_ep14 | fail |
| 900002 | 安妮塔 | act18d3、act36side、act52side | act18d3 | - | act36side、act52side | fail |
| 900005 | 贝纳尔多 | act21side、act38side、memory_demetr、memory_vigil | act21side | - | act38side、memory_demetr、memory_vigil | fail |
| 900008 | 豆豆 | memory_bstalk、memory_flower | act13mini、act31side、memory_bstalk、memory_ctable、memory_flower | act13mini(A)、act31side(A)、memory_ctable(A) | - | fail |
| 900011 | 学生联合大会 | act51side、memory_headb2 | act51side、memory_headb2 | - | - | PASS |
| 900012 | 斯普莉雅 | act26side、memory_spuria | act26side、memory_spuria | - | - | PASS |
| 900013 | 技术院 | act34side、act52side、main_ep15 | act34side、act52side、main_ep15 | - | - | PASS |
| 900014 | 图里卡姆 | act14side、act30side、memory_blackd | act14side、act30side、memory_blackd | - | - | PASS |
| 900016 | 宁辞秋 | act15side、act31side、act40side、act49side | act31side、act40side、act49side | - | act15side | fail |
| 900017 | 日落街酒店 | main_ep12、main_st12、memory_morgan | act37side、main_ep12、main_st12、memory_morgan | act37side(A) | - | fail |
| 900018 | 雄辩家 | act22side、act41side | act22side、act41side、main_ep09、main_ep10 | main_ep09(A)、main_ep10(A) | - | fail |
| 900021 | 法伦 | act7d5、memory_blitz、memory_melan | act7d5、memory_blitz、memory_durnar、memory_melan | memory_durnar(A) | - | fail |
| 900022 | 巫术 | main_ep11、main_ep14 | act29side、act47side、main_ep13、main_ep14 | act29side(A)、act47side(A)、main_ep13(A) | main_ep11 | fail |
| 900024 | 帕特 | main_ep11、main_ep12 | act42side、memory_phenxi | act42side(S)、memory_phenxi(S) | main_ep11、main_ep12 | fail |
| 900026 | 蒂姆 | act29side、act41side、memory_lessng | act41side、memory_beewax、memory_lessng | memory_beewax(S) | act29side | fail |
| 900027 | 炎魔 | act15d0、main_ep13、main_ep14、memory_plosis | act15d0、main_ep13、main_ep14、main_st12、rogue_rogue_4 | main_st12(A)、rogue_rogue_4(A) | memory_plosis | fail |
| 900030 | 勾吴 | memory_sqrrel、rogue_rogue_5 | act16d5、act23side、act49side、memory_nothin、memory_sqrrel、rogue_rogue_5 | act16d5(A)、act23side(A)、act49side(A)、memory_nothin(S) | - | fail |

---

## 复现方法

```powershell
Set-Location E:\图rag
$env:PYTHONIOENCODING = "utf-8"
py -3 scripts\diag_cross_judge.py --budgets 6000,12000
```

脚本耗时约 1 分钟（`load_generation` 单次约 8 秒，两档共用一次加载）。
产物 `data/diag_cross_judge.json` 顶层键：`generation_id / gold_set / category / split / n_items / q1 / items_meta / per_budget / criterion / criterion_final`；
`per_budget[B]` 含 `metrics / ceiling / conditional / extra_class_totals / variants / n_questions_with_extra / n_questions_with_miss / mean_listed / mean_extra / mean_miss / rows`。
逐题的 `extra_detail` 里带每个多列章节的 `chunk_id`、`kind`、命中行原文（`sample_lines`）与 `unit_entities_has_focus`，可直接复核本报告的证据表。

脚本未修改 `src/**`、`tests/**`、`data/goldset/**`、`scripts/ask.py`，未运行 pytest。
