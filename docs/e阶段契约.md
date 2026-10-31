# E 阶段契约：逐观察证据图、关系方向、实体身份、叙事限定

本文件是 E 阶段**唯一 schema 事实源**。任何模块、测试、脚本用的字段名与语义以本文件为准；
要改就改本文件并说明理由，不许只在代码里改。

计划原文见 `docs/升级计划_GraphRAG架构.md` 的 E1–E4。

---

## 0. 现状实测（决定这些改动值不值得做）

全部读盘，非估算。

| 项 | 数值 | 出处 |
|---|---|---|
| T1 块 | 13,781 | `chunks.jsonl` 的 `in_t1` |
| raw relations（逐观察） | **63,172** | 13,781 条缓存记录的 `raw.relations` |
| raw entities（逐提及） | **86,124** | 同上 |
| 当前图的聚合边 | **28,234** | `graph.json` |
| 只有 1 条观察的边 | 18,590 | `len(edge.chunks)` |
| 有多条观察的边 | **9,747** | `edge.repeat > 1` |
| 单边最多观察条数 | **116** | 同上 |
| 每观察证据可在块内定位 | **63,212 / 63,212**（T1 范围内） | `text.count(evidence)` |
| 证据在块内出现多次 | **189** | `count > 1` |
| 缓存重复 chunk_id | **20** | 均为 `prompt_version=v1` 的陈旧记录 |

观察守恒链条（逐级实测，别用估算）：

```
63,172  raw relations（13,781 块的全部 raw）
 −  402  建图期丢弃（endpoint_filtered 231 + self_loop 150 + low_confidence 21）
= 62,770 进入聚合的观察（== sum(edge.repeat)）
 −34,536 与同边其它观察共用一条 evidence
= 28,234 聚合边（== len(edges)）
```

**结论**：9,747 条边的多条观察共用一条 `evidence`，单边最多 **116** 条观察只留 1 条证据；
`sum(len(edge.chunks))` 只有 59,363（同块内多次观察被去重）。
**34,536 次观察在当前图里没有自己的证据位置**——这不是理论缺陷，是 E1 要消灭的东西。

---

## 1. Assertion（逐观察断言）—— E1

每次「某块的抽取结果里出现了一条关系」就是一条 Assertion。**Assertion 是图的原子事实单元**；
聚合边只是它的派生统计。

```jsonc
{
  "assertion_id": "a-3f9c1d2e8b7a4f60",   // 见 1.1 稳定哈希
  "subject_id": "罗德岛",                  // 已消歧的实体 id（不是原始名）
  "predicate": "隶属",                     // 规范化后的谓词（norm_relation 产出）
  "object_id": "凯尔希",
  "direction": "out",                      // out = subject → object；in = 反向（见 1.3）
  "raw_relation": "隶属于",                // 上游原词，永不为归一化丢弃

  "chunk_id": "obt/main/level_main_00-01_beg#000",
  "scene_id": "obt/main/level_main_00-01_beg",   // 场景实例身份 = doc_id（1 doc = 1 scene，实测）
  "doc_id":   "obt/main/level_main_00-01_beg",

  "evidence_text": "你们背叛了王酋",
  "evidence_start": 412,                   // 块内字符偏移，UTF-16→Python str 下标（0 起，右开）
  "evidence_end": 419,
  "evidence_all": [                        // 块内全部出现位置；多个候选必须全留
    {"start": 412, "end": 419},
    {"start": 880, "end": 887}
  ],

  "confidence": 0.85,
  "qualifiers": { /* 见 §4 */ },
  "extraction": { /* 见 1.2 */ }
}
```

### 1.1 `assertion_id` 的构成（必须稳定、可重算）

```
sha256("\x1f".join([
    chunk_id,
    subject_id,
    predicate,
    object_id,
    str(evidence_start),        # 用 primary 偏移而不是全部候选，避免为去重再算一遍
    raw_relation,
]))[:16]  前缀 "a-"
```

**不得**把 `confidence`、`written_at` 之类非身份字段放进哈希——否则同一事实重跑就换 id，
回滚比对全部失效。两条观察即使文本完全一样，只要 `evidence_start` 不同就是两条独立
Assertion（这正是本层要保留的信息）。

### 1.2 `extraction` 块（每层身份都要带）

```jsonc
{
  "schema_version": "ark-rag-1",
  "prompt_sha256": "…64 位…",
  "vocab_sha": "…16 位…",
  "entity_rule_version": "ev2",
  "relation_version": "rv2",
  "normalizer_version": "n2",
  "extraction_manifest_hash": "…",     // release.extraction_manifest_hash()
  "generation_id": ""                  // 构建期填，手写测试可为空
}
```

理由：一条 Assertion 被人质疑时，要能立刻回答「它是在哪版提示词、哪版词表、哪批输入下抽出来的」。
只留一个全局 `extraction_manifest_hash` 在 Manifest 里不够——那是整批的，不是这一条的。

### 1.3 `direction` 与反向邻接的语义边界

* `direction` 在 Assertion 上**恒为 `"out"`**。Assertion 记录的是「抽取时 subject 在左」这个事实。
* 需要反向视角时**不改 Assertion**，由邻接层（§2）给出 `direction: "in"` 的邻接项，
  并把谓词换成逆谓词（`vocab.INVERSE_PAIRS`）。
* 关系没有逆谓词（48 个规范谓词里只有 11 对）：反向邻接项必须带 `"inverse_relation": ""`
  且 `"direction": "in"`，**禁止**把原关系名原样当反向关系用（这正是当前
  `index.build_adjacency` 的缺陷）。

### 1.4 硬规则（门禁要守的）

1. 证据在任何来源块里都定位不到 → **该 Assertion 不许入库**，计入 `dropped:evidence_not_found`。
2. 不允许拿别的块的证据顶替。
3. 不允许只有文本没有偏移。
4. `evidence_text` 长度上限 60（沿用 `contracts.EVIDENCE_MAX_CHARS`）。

---

## 2. 有方向的邻接（E2）

`index.build_adjacency(edges)` 的返回结构改为邻接项列表，**不再是对称的两条同关系名**：

```jsonc
{
  "entity": "罗德岛",
  "neighbors": [
    {
      "neighbor": "凯尔希",
      "direction": "out",              // out / in / undirected
      "relation": "隶属",               // 从 entity 看过去的方向语义（in 时已被换成逆谓词）
      "inverse_relation": "包含",       // relation 的逆谓词；无则空串
      "symmetric": false,               // 该谓词是否对称（vocab.SYMMETRIC_RELATIONS）
      "edge_id": "e-…",                 // 聚合边身份
      "assertion_ids": ["a-…"],         // 支撑这条邻接的全部断言
      "weight": 0.85,                   // 聚合置信度（confidence_summary.max）
      "support_count": 3,
      "independent_scene_count": 2
    }
  ]
}
```

规则：

* **出边入边分别处理**。`direction` 必须显式写出来，不许靠"这条边在谁的列表里"隐含。
* `in` 项的 `relation` 用逆谓词；没有逆谓词时 `relation` 保留原谓词但
  `inverse_relation=""` 且 `direction="in"` —— 调用方必须自己判断能不能反着说。
* 对称谓词（`亲属`/`同盟`/`敌对`/`交战`/`冲突`/`研究`/`称为`/`交谈`/`同行`）标
  `direction="undirected"`、`symmetric=true`。
* 图扩展（G 阶段）**默认只走出边**；需要无向扩展时必须显式传参，
  并在结果里留下 `expansion_mode="undirected"` 的痕迹。
* 每条邻接项都要能回到 `assertion_ids` → `chunk_id` → 原文。**只返回图结论不返回原文证据是禁止的**。

---

## 3. 实体身份（E3）

三层模型，名称只是展示字段：

```
Mention（逐块提及，上游抽取给的原始名）
   ↓  类型归一 + 噪声剔除
EntityCandidate（候选，可能同名多义）
   ↓  防误并检查 + 别名消歧
CanonicalEntity（最终实体，稳定 id）
```

### 3.1 CanonicalEntity 结构

```jsonc
{
  "id": "e-7d2a91c4f0b83e16",        // 稳定哈希，见 3.2
  "name": "凯尔希",                    // 展示字段，可改，不参与身份
  "type": "角色",
  "desc": "…",
  "aliases": [                        // 权威检索表：只含 review ∈ {auto, approved}
    {
      "text": "凯太后",
      "source": "upstream",           // upstream / fullname / alias_chain / manual
      "scenes": ["obt/main/level_main_00-01_beg"],   // 该别名出现过的场景范围
      "review": "auto"                // auto / approved（pending 与 rejected 不在此表）
    }
  ],
  "pending_aliases": [                // 未审候选：检索**不可见**，人工审核后可提升
    {"text": "库兰塔士兵", "source": "upstream", "scenes": ["…"], "review": "pending"}
  ],
  "rejected_aliases": [               // 显式拒绝，留档可追责
    {"text": "…", "source": "upstream", "scenes": ["…"], "review": "rejected"}
  ],
  "merged_from": [
    {"name": "凯尔希医生", "reason": "fullname", "scenes": ["…"]}
  ],
  "split_history": [                  // 拆分/合并历史，支持事后追责与回退
    {"at": "2026-11-01T00:00:00Z", "action": "merge", "members": ["…"], "reason": "alias"}
  ],
  "mentions": 1234,
  "chunks": ["…"]
}
```

#### 3.1.1 为什么别名拆三张表（2026-10-31 实测后加，原契约只有 `aliases`）

原契约只列 `aliases` 一个字段、用 `review` 标记审核状态。**实测证明这个设计没有强制力**：

```
$ Select-String src/arkrag/*.py scripts/*.py -Pattern 'review'
→ 全部命中都在 identity.py 内部（写的那一侧），读取方零个
```

下游建别名表的两处写法都是只看 `aliases`：

```python
# src/arkrag/graph.py:181
"alias_index": {alias: ent["id"] for ent in entities.values() for alias in ent["aliases"]}
# src/arkrag/index.py:134-139
def build_alias_index(entities): ... for alias in ent["aliases"]: out.setdefault(alias, ent["id"])
```

于是出现这样的结果：合并被拒了（实体没并），但**别名表照样指过去**——`库兰塔士兵` 依然能从检索命中 `暴雨`。同一个错误从"并实体"转移到"并别名"，只是晚一步发作。

改法是把审核状态**编码进结构**，而不是依赖下游自觉过滤：

- `aliases` 只放 `review ∈ {auto, approved}` —— 下游拿它建索引是**正确**的写法，不需要懂 `review` 语义；
- `pending_aliases` 放 `pending` —— 保留 `text`/`scenes`/`review` 全部信息，检索不可见，人工审核后提升进 `aliases`；
- `rejected_aliases` 放 `rejected` —— 显式拒绝留档。

`CanonicalEntity` dataclass 自身仍持有全部条目（含 `review`），**只有 `to_dict()` 的产物做分离**，所以 `merged_from ⊆ aliases ∪ pending_aliases ∪ rejected_aliases` 的血缘完备性不受影响。

这条改动依据的是本项目反复踩到的同一类坑：**同一个判定在两处各写一份，只要有任意一处较宽松，整套检查就等于那处最松的**（已发生三次：冻结清单 vs 写保护清单、`corpus_meta` 副本 vs `source_verification` 权威、`release.py` vs `source.py` 的"已验证"判据）。纪律：判定只能有一处事实源，其余地方引用它而不是重写它。

**下游硬性要求**：任何按别名建检索表的地方，只能读 `aliases`，**不得**读 `pending_aliases` 或 `rejected_aliases`。有测试盯着（`tests/test_identity.py::TestAliasVisibility::test_pending_alias_cannot_be_found_by_alias_index` 现场按 `graph.py:181` 的写法建一次索引再断言）。

**实测规模**（真实数据，drop_noise=True，13,786 chunks / 86,124 mentions）：

| 项 | 值 |
|---|---|
| `to_dict()["aliases"]` 条目 | 1,686（全部 `review=auto`） |
| `to_dict()["pending_aliases"]` 条目 | 539 |
| `to_dict()["rejected_aliases"]` 条目 | 0 |
| dataclass 别名行合计 | 2,225（= 1,686 + 539，闭合） |

539 的来源分类（互斥，已核实）：形状判据新增 28（`alias_generic_pending`）+ 别名冲突 70（`alias_conflict_pending`）+ 歧义主张 441（第 4 条 `ambiguous_alias` 挡下的候选名及其上游别名行）。


### 3.2 稳定 id

```
sha256("\x1f".join([entity_rule_version, canonical_name_at_birth]))[:16]  前缀 "e-"
```

**必须满足**：改 `name`（例如把 `凯尔希医生` 的展示名换成 `凯尔希`）**不得**改 `id`。
所以 id 只在实体**第一次被创建**时算一次，之后写进 `merged_from`/`split_history` 就固定下来。
纯名字做 id（当前实现）在 E3 之后不再允许。

### 3.3 合并前的六道检查（顺序不可调换，任一不过就拒绝 Union）

1. **类型兼容**：`type` 不同且两者都不是 `概念` → 拒绝（`rejected:type_conflict`）。
2. **集体 vs 成员**：`is_member_index(short, long)` 命中 → 拒绝（`rejected:member_index`）。
3. **人名 vs 职务/组织泛称**：一方是 `GENERIC_PERSON_NOUNS` 或 `ORG_SUFFIXES` 形态 → 拒绝（`rejected:generic_anchor`）。
4. **歧义别名**：被 ≥2 个候选同时主张 → 不自动合并，标 `review=pending`（`rejected:ambiguous_alias`）。
5. **连通分量检查**：Union 之后整组若出现类型冲突或成员索引关系 → 撤销这次 Union（`rejected:component_conflict`）。
6. **冲突关系**：Union 之后若产生 `MUTUALLY_EXCLUSIVE_PAIRS` 的自环式矛盾 → 撤销（`rejected:relation_conflict`）。

宁可拆多不可错并：任何一道检查不确定时**保留候选不合并**，写进 `review=pending`。

### 3.4 类型归一顺序（E3 要求，当前实现违反）

现状 `norm_entity.norm_type` 的顺序是「精确 → **子串** → 同义词 → 兜底」，
其中子串匹配在第 3 位之前，会把 `人物设定` 这类把"人物"子串命中的词判成 `角色`。
新顺序固定为：

1. 规范值精确匹配（`text in ENTITY_TYPES`）
2. 精确同义词（`TYPE_SYNONYMS[text]`）
3. 受限模糊匹配（**仅当** `text` 长度 ≥ 2 且命中项占 `text` 长度一半以上，才允许子串命中）
4. 未知 → `概念`（兜底，不许给「未分类」）

顺序本身的理由：子串匹配是最松的判据，放在最前面等于让最松的判据覆盖最紧的判据。

---

## 4. 叙事限定字段（E4）

`qualifiers` 是 Assertion 的字段，全部可为空/默认。**空串表示「抽取没有给」，
不等于「没有这个性质」**——所以门禁不校验它们是否填满，只校验类型与互斥一致性。

| 字段 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `story_time` | str | `""` | 剧情内时间（原样文本，如「1096年」），不做日期解析 |
| `valid_time` | str | `""` | 该事实成立的时间范围（原文表述） |
| `scene_order` | int | `-1` | 场景在 doc 内的序（当前 == chunk 的 `scene_index`） |
| `certainty` | str | `"asserted"` | `asserted` / `hedged` / `uncertain` / `negated` |
| `modality` | str | `"actual"` | `actual` / `hypothetical` / `reported` / `remembered` |
| `is_flashback` | bool | `false` | 回忆/闪回 |
| `is_hearsay` | bool | `false` | 转述 |
| `is_hypothetical` | bool | `false` | 假设/未发生 |

**互斥关系不直接删除**：两条互斥谓词（`同盟` vs `敌对`）的 Assertion 若落在不同场景，
两条都保留，由聚合层记 `conflict_count`；若落在同一场景同一条证据，标 `certainty="uncertain"`
交人工，**不许悄悄挑一条留下**。

本轮不实现时间抽取（没有 LLM 端点，也没有可靠的规则），字段先按默认值写满，
让下游结构稳定。这一点必须如实记进台账，不许说成"已支持时间推理"。

---

## 5. 门禁改动

### 5.1 新增（E1）

* `L5.9 每条 Assertion 证据可定位（逐观察，FAIL）` —— 逐条 `chunk_text[start:end] == evidence_text`
  且 `evidence_all` 每个区间都对得上。当前实测 63,172/63,172 应全通过。
* `L5.10 聚合边只存派生统计（FAIL）` —— 边记录**不得**含唯一 evidence（`evidence` 字段必须是
  从某条 assertion 派生出的代表值，且该 assertion 必须在 `assertion_ids` 里）。
* `L5.11 观察守恒（FAIL）` —— `sum(support_count) == len(assertions)`，且每条 assertion 恰好
  被一条聚合边引用（不许丢、不许重复计）。

### 5.2 强化（E2）

* `L6.4 邻接方向完整性（FAIL）` —— 每条邻接项必须有 `direction` ∈ {out,in,undirected}；
  `in` 项的 `relation` 必须是 `relation` 的逆谓词或为空；每条邻接项的 `assertion_ids` 非空。

### 5.3 强化（E3）

* `L4.12 实体 id 不含名字（FAIL）` —— 抽查每个实体 `id != name` 且 id 匹配 `^e-[0-9a-f]{16}$`。
* `L4.13 合并前六道检查有记录（FAIL）` —— 每个 `merged_from` 项必须有 `reason` ∈
  {alias, fullname, alias_chain}，且 `rejected:*` 统计必须在 `stats.identity` 里出现。

---

## 6. 兼容与迁移

* 新图的 `schema_version` 升为 **`ark-rag-3`**（`ark-rag-1` 是聚合边模型，`ark-rag-2` 未使用）。
* 旧图 `data/arkrag/graph.json` 与已发布世代里的图一律视为 **Legacy 投影**：
  只读、可对比、**不得**与新 Assertion 图混用（Manifest 里已经靠 `graph_artifact_id` 分隔）。
* 迁移策略按计划：新 Assertion 图**尽量**从现有 raw 缓存重建（raw 完整，实测 13,781 块全有记录，
  63,172 条关系全可定位）。**raw 不全处必须重抽，不伪造历史证据**——
  当前 raw 覆盖完整，所以本轮不重抽，但代码必须能报出"缺 raw 的块"而不是静默少边。
* 20 条 `prompt_version=v1` 的陈旧重复缓存记录：`cache.lookup_for_chunk` 已按
  `prompt_version == config.PROMPT_VERSION` 过滤，不会进入建图。**不许删它们**（那是历史证据），
  但要在 `invalid_cache_report()` 的计数里能看见。

---

## 7. 纪律

1. 本文件的 schema 是唯一的；代码里出现与本文不符的字段名即为缺陷。
2. 每个新增门禁必须能被至少一条构造输入触发 FAIL（`test_gates_*.py` 的元测试会检查）。
3. 所有改动必须在真实数据上跑一次并写进台账；只跑单元测试不算验收。
4. 不得为了让门变绿而放宽判据。如果数据确实做不到，把门改成「申报」并在台账写明理由。
