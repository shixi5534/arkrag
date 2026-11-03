# I 阶段 · 前项目经验检索（roleplay-ai → ArkRAG）

> 取证对象：`C:\Users\Lenovo\WorkBuddy\2026-07-27-16-45-03\roleplay-ai\`（明日方舟前身项目，含 `compendium-rag` 与 `plot-graph-rag` 两套检索线）
> 取证方式：只读。文档全文/grep、`src/` 与 `scripts/` 源码实读、`data/**/reports/*.json` 产物读数复核。**未运行任何脚本、未修改 roleplay-ai 任何文件、未修改本仓库除本报告以外的任何文件。**
> 引用纪律：以下所有引文均为原文逐字（含原项目错别字与术语，如「扩得越大」「已挤不出水」）。凡我给不出文件+行/节的，一律写「未找到」，不做推断性补全。

---

## 0. 阅读范围与「未找到」清单

已实读（带行号复核）：

| 类别 | 文件 |
|---|---|
| 计划/裁决文档 | `docs/compendium-rag/检索四缺口修复计划_v10_2026-10-21.md`、`P10口径裁决_N_retrieval能力读数.md`、`P10修复计划_CBI收口.md`、`research/四缺口外部设计对照_2026-10-21.md` |
| 基线/编年史 | `docs/compendium-rag/baseline/BASELINE_v7_20260928.md`、`architecture.md`、`CHRONICLE.md`、`优化计划_v9_2026-10-08.md`、`P9_CBI报告.md`、`entity_disambiguation_rules.md`、`判分尺子_CBI统一rubric_v2.md` |
| plot 线文档 | `docs/plot-graph-rag/CHRONICLE.md`、`corpus/剧情时间轴与GraphRAG联动.md`、`corpus/剧情术语对照词表.md` |
| 源码 | `src/compendium_rag/retriever.py`、`src/roleplay/core/knowledge/plot_graph.py`、`plot_rerank.py`、`plot_semantic.py`、`plot_entity_expand.py`、`graph_ppr.py`、`src/roleplay/config.py` |
| 实验脚本 | `scripts/_rank_rerank_lab.py`、`_rank_collect.py`、`_rank_triage.py`、`_rank_stats.py`、`_frozen_prod_eval.py`、`_85_sweep_k.py`、`_rrf_k_regression.py`、`_fusion_restructure.py`、`_fusion_restructure_accept.py`、`_p3_pool_widen.py`、`_p3_vector_weight_scan.py`、`_p3_prf_rm3.py`、`_p3_intervention.py`、`ab_ce_rerank.py`、`ab_rerank_ceiling.py`、`_p4_ce_real_test.py`、`_p4_ce_subset.py`、`_p4_ce_onnx_subset.py`、`ab_query_rewrite.py`、`compendium_rag_eval300.py`、`eval_plot_rag.py` |
| 产物读数 | `data/compendium_rag/reports/_85_sweep.json`、`_85_verify_k1.json`、`_85_lexfirst_safe.json`、`_85_safe_run.log`、`_p4_ce_subset.json`、`_p4_ce_onnx_subset.json`、`rank_rerank_lab_300cn_p30_ce.json`、`rank_rerank_lab_300cn_v6/.v7/.p30/.me60/.pe8.json`、`rank_triage_300cn_v7.json`、`rank_stats_final.json`、`_p3_lexfirst_refit.json`、`_p3_ab_sweep.json`、`_fusion_restructure_accept.json`；`data/knowledge/_audit/ab_ce_rerank_k30.json`、`ce_timing.json`、`ce_batch_timing.json`、`ab_qr_e2e_mh40.json`、`ab_qr_oracle_upper.json` |

**未找到**（用户点名的目标，实查后不存在或无相关内容）：

1. `docs/compendium-rag/ranking/` 目录 —— **未找到**。`docs/compendium-rag/` 实有 14 个 md，无 `ranking/`。CHRONICLE 引用的 `排序侧实验_执行报告_20260926.md`、`docs/目标①_池内miss干预报告_20260928.md`、`docs/目标①_收口报告` **均未找到**（只剩 CHRONICLE/脚本 docstring 里的摘录与复述）。
2. `docs/plot-graph-rag/graph-governance/` 下与排序/图扩展相关的部分 —— **未找到**。该目录只有 `wiki-r1999-relation-reference-20260909.md`（39,850 B，wiki 关系参考），grep「RRF|融合|重排|MRR|排序」**零命中**。`acceptance/剧情图谱验收标准.md`、`corpus/剧情别名与抽取裁决记录.md` 同样零命中。
3. `docs/plot-graph-rag/corpus/剧情时间轴与GraphRAG联动.md` 中的排序/融合/重排内容 —— **未找到**。该文档是「三份新产物的挂接点设计」（术语表/时间轴/人物去向图），只有 G1–G4 四个挂接点定义与「别名表现状」（`:68` `| 实体名去重数 | 6 473 |`、`:69` `| 现有别名表覆盖 | 340 个表层形式（**仅 12.3% 的提及**） |`），不涉及排序。
4. 生产代码里的**在线重排** —— **未找到**。`src/compendium_rag/` 内 grep `class .*Reranker|zfit|full_topN|StaticReranker` 零命中；冻结线性重排只存在于离线脚本 `scripts/_rank_rerank_lab.py` 与 `_frozen_prod_eval.py`。`plot_graph.py` 内也无 CE 调用，`plot_rerank.py` 是独立模块且 `plot_rerank_topup_enabled=False`、CE 从未接入 `retrieve()`。
5. 重排的**跨请求缓存 / 异步** —— **未找到**。只有进程内一次性构建的记忆化（`plot_graph.py:1818 _sem_hash2row`「构建一次后缓存」），无跨查询缓存、无异步、无线程池。

---

## 1. 结论先行

### 可立即用（有实测读数 + 有可直接抄的代码，且我方已有全部依赖）

| # | 做法 | 出处 | 为什么对我方直接可用 |
|---|---|---|---|
| A1 | **RRF 的 `k` 是名次折扣曲线，不是「融合强度」**：k=60 时 rank1/rank6 分值比只有 1.08×，k=1 时 3.50×。调权重无效、调 k 有效。 | `docs/compendium-rag/baseline/BASELINE_v7_20260928.md` §3（行 55–67） | 我方扫到 k=1 时最好成绩仍是 ΔR@20 +0.0125 / ΔMRR −0.0071（两项未同时不劣），说明**下一步不是继续在 k 上找点**，而是换空间（见 7.2）。这条的价值是「把 k 这一维从怀疑名单里划掉，并给出机理」。 |
| A2 | **「池内金标名次分布 + 池天花板」这一组诊断**，把 miss 三分成 `pooled_in_window / corpus_not_pooled / no_corpus_match`，并用 oracle 算「若重排把池内 gold 全部顶进前 N 的天花板」。 | `scripts/_rank_triage.py`（docstring）、`scripts/_rank_rerank_lab.py`（`oracle_ceiling`）、产物 `rank_triage_300cn_v7.json`、`rank_rerank_lab_300cn_v7.json` | 我方「bm25 单路 gold 命中率 0.9367 但 R@20 只有 0.6478」疑似 `pooled_in_window` 占绝大多数，但**这是待验证的推断，不是结论**（我方尚未做这项诊断）。**先量名次分布，再谈方案。** |
| A3 | **离线盲特征线性重排**（`full_topN(a=1.0)`）：5–10 个 z-score 归一盲特征加权求和，替换全池次序。 | `scripts/_rank_rerank_lab.py`（行 45–46、251–283、330–347）；产物 `rank_rerank_lab_300cn_p30_ce.json` | **p95 22.4 ms**（特征）+ **p95 0.244 ms**（重排本体），纯 numpy，零模型。完全落在我方 P95 < 200 ms 预算内，还留一个数量级余量。 |
| A4 | **「保留头部 + 余弦/打分补位」而不是整体重排**：整体重排 strict +1 而 inject −26；保留前 3 + 补位 strict +12、inject +1。 | `src/roleplay/core/knowledge/plot_graph.py`（行 1840–1857 docstring、1899–1913 实现） | 直接对应我方「MRR 比 R@20 值钱」：补位只动尾部，不动榜头，**保 MRR 的同时补召回**。且给出分数带纪律（补位块一律压在头部最低分之下）。 |
| A5 | **「扩池/扩候选无效、召回面已不是瓶颈」的判据与实测**：mention 截断 5→20/10→50，gold 入池 0→19 题，最终 strict 只 +1；gold 名次反而更靠后。 | `scripts/ab_rerank_ceiling.py`（行 6–13）、`data/knowledge/_audit/ab_mention_depth_20260917.json`、`ab_rerank_ceiling_20260917.json` | 直接确认我方第 5 条的判断：**继续扩召回/扩池是负期望**。这条能替我方向 owner 挡掉「再把候选池放大」类提案。 |
| A6 | **配对显著性 + 名次记账工具口径**：hit@6/hit@10/mrr@6 分层，McNemar 精确双侧（`b`/`c`）+ 配对 Bootstrap 10k CI，逐题 up/down 清单。 | `scripts/_rank_stats.py`（docstring 行 1–34）、产物 `rank_stats_final.json` | 我方「两指标一起看」的门禁需要这个：单看 Δ 点数会被 ±1 题噪声骗（他们的 ±1.6pt 就是 `p=0.635828` 不显著）。 |

### 需验证（方向对，但前项目没在「我方条件下」验证过）

| # | 做法 | 出处 | 必须先验什么 |
|---|---|---|---|
| B1 | **词面优先结构改法（`lex_first` / `m2_union(3)`）**：把词面通道前 L 名提到最前，其余按融合序补足、去重。 | `src/compendium_rag/retriever.py`（行 724–747）、`scripts/_fusion_restructure_accept.py` | 前项目**在这条上自相矛盾过**：300cn 选参时两集同升（0.8316→0.8469 / 0.8301→0.9126），但把 `lex_first=3` 拿到生产重排口径的盲测集上**崩 37 题**（`_p3_lexfirst_refit.json`：blind `rerank_hit6 0.7427`，`verdict.blind_no_regress=false`）。**必须双集/留出集验证，且必须先跑可分性诊断**（见 B2）。 |
| B2 | **条件化词面优先的可分性诊断**：穷举「是否提升」的可部署判据（共识闸门 `n_lex3_in_top6 >= k`、分离度 `gap12 >= τ`、`notin-top6` 限量提升…）。 | `data/compendium_rag/reports/_85_lexfirst_safe.json`、`_85_safe_run.log` | 结论是**只有用 gold 的信号可分，部署可用的信号全部不可分**。若我方也要做「条件触发」，必须重复这套穷举，且**禁止**用 gold 派生的信号。 |
| B3 | **补位用什么分数**：前项目用 query–chunk 余弦。 | `plot_graph.py:1896` `scored.append((float(np.dot(v, qn) / nv), idx, chunk, extra))` | 我方 bge-m3 单路 R@20 仅 0.3892 / MRR 0.2101，而 roleplay-ai 明确记录过「真阳余弦低于负样本」（`docs/plot-graph-rag/CHRONICLE.md:328–331`）。**我方补位分母是否该换成 BM25 分数或 BM25 归一化分，必须先量**。 |
| B4 | **执行器分离（`pool_top_n` 与 `top_k` 解耦）**：把「进融合的候选宽度」和「输出条数」拆成两个参数。 | `docs/compendium-rag/检索四缺口修复计划_v10_2026-10-21.md` §1（行 22–26）、`architecture.md` §4.1（行 221–252） | 前项目实测 `top_k=6` 实际返回 **33–105 条**（中位 68）。我方 `top_k=20` 若同样兼职池宽，则「R@20」量到的东西可能不是「模型看到的 20 条」。**先自查我方 `top_k` 是否兼职池宽**——这是口径问题，可能直接解释一部分 MRR 异常。 |
| B5 | **盲特征的 `orig` 项（原 RRF 分）权重可能为 0** | `rank_rerank_lab_300cn_v6.json`：`"orig": 0.0` | 说明「RRF 融合分」作为重排特征**没有增量**。若我方照搬特征集，应把 RRF 分列为待消融项而不是默认项。 |

### 不适用（有明确负结果，别重蹈）

| # | 做法 | 出处 | 负结果 |
|---|---|---|---|
| C1 | **小模型 listwise LLM 重排** | `plot_rerank.py:17–19`（docstring） | `qwen2.5:1.5b` 全量 300 题 inject_pass 62.00% → 60.33%（−1.67 pt），「把证据块往后挪」。 |
| C2 | **ONNX int8 交叉编码器重排** | `docs/compendium-rag/CHRONICLE.md:608`、`_p4_ce_onnx_subset.json` | 「ONNX int8 快 6 倍但**增益完全消失**」；同 26 题子集 fp32 CE 0.5 → int8 0.4231（掉 2 题）。**我方手上正是 int8 那个**。 |
| C3 | **CE 全量重排** | `data/knowledge/_audit/ab_ce_rerank_k30.json` | `ce_latency_p50_ms 990.5 / p95 13005.4`，300 题 `elapsed_s 1520.2`。且 `d_strict +6 / d_inject −3`——召回涨、注入跌，正是我方要避免的形状。 |
| C4 | **绝对余弦阈值做通道过滤** | `检索四缺口修复计划_v10` §1（行 29–34）、`优化计划_v9` §…、`plot-graph CHRONICLE.md:328–331` | BGE 模型卡 [0.6,1] 被证伪（本仓库实测 0.32–0.67，中位≈0.50）；LightRAG 引入后又移除（「绝对阈值容易造成「全空」副作用」）；且「真阳实体的余弦分数**低于**负样本」。 |
| C5 | **可见性/角色域过滤** | `P10口径裁决` §2.4、`entity_disambiguation_rules.md:200–205`、`CHRONICLE.md:145` | 三套 MRR 全负：300cn `Δ −0.12963`、blind206 `Δ −0.09926`。两次独立判负。 |
| C6 | **idf 加权提及打分** | `plot_graph.py:1119–1124` | 「实测在本语料上 **idf 反而更差**（严格通过率 −4pt）」。 |
| C7 | **HippoRAG 式 PPR 引导 dense 排序** | `docs/plot-graph-rag/CHRONICLE.md:476` | 「闸门判定：不达标，不落地」。 |
| C8 | **靠收紧 top-k 改指标** | `P9_CBI报告.md` §2.13（行 587–641） | 「**k 不是修法**」。 |

---

## 2. 他们遇到的同型问题与解法

### 2.1 「候选池里有 gold，但排不进输出窗」——与我方第 5 条形态相同（我方尚未验证是否同构）

**`scripts/ab_rerank_ceiling.py:1–13`（docstring 原文）**：

> 为什么做这个实验
> -----------------
> ``ab_mention_depth.py`` 实测：把 mention 通道截断从 (per_entity=5, cap=10) 放大到
> (20, 50) 后，60 道 never_candidate 题里 **gold 块进入候选池的从 0 题涨到 19 题**，
> 但最终 strict 只救回 **1 题**。
>
> 这说明瓶颈已经从「召回面（gold 进不了池）」转移到「排序（进了池挤不进 top-8）」：
> 候选池扩到 42 块后，RRF 按名次给分，排在 20 名开外的 gold 块拿到的融合分太低，
> 被 edge / mention 前排块挤出预算。**继续扩池是无效的**——扩得越大，gold 的名次
> 反而越靠后。

**`src/roleplay/core/knowledge/plot_rerank.py:4–14`（模块 docstring 原文）**：

> 为什么需要（2026-09-16 通道归因实测）
>     90 道缺口题中 **89 道是"融合失败"**——候选池里有证据块，但没进 final 的前几块。
>     图传播通道每题产生约 17 块候选，而 final 只有 6~8 个位置；证据块被 RRF 压到底部。
>     参数消融（max_edges / weight_semantic / node_specificity / max_hops）与
>     **候选生成量上调**（semantic_top / mention cap / edges / chapter_top，见 IDEAS_LOG【23】）
>     对 ``inject_pass`` **均无实质收益** ⇒ 瓶颈是"排序"，不是"候选量"。
>
>     v2 证据（同日 23:20，`diag_rank_distribution.py`）：产品按 2400 字符装配只装得下
>     约 2 块 ⇒ 证据块必须排进**前 2 名**才算数。当前分布：
>     第 1 名 82 题｜第 2 名 42 题｜**第 3~8 名 76 题**｜无覆盖块 100 题。
>     重排的理论可动靶子 = 76 + 26（第 9~30 名）= 102 题。

**`docs/plot-graph-rag/CHRONICLE.md:73–75`（原文）**：

> - **「让证据块进 top-8」是错误目标**：**25.3%（76 题）** 的证据块进了 top-8 却排在第 3~8 名，提示词装不下；优化 recall / strict_pass 会把这部分当进展，**实际产品零收益**。
> …
> - 结论：靶子应改为**名次分布**（把正解从前 8 名之外拉进第 1–2 名），`scripts/ab_cross_encoder.py` 引用本文作为重排目标的定义。

**`docs/plot-graph-rag/CHRONICLE.md:99`（原文）**：

> - **关系题瓶颈是 top-k 而非召回**：严格命中 top-5 55% → top-10 64% → top-20 71%，正解已在候选池（7,582 边 / 9,021 块）只是排序竞争激烈。

**他们的解法（正解）**——`docs/plot-graph-rag/CHRONICLE.md:61–62`（原文）：

> - 根因定位：**RRF 只按名次给分，深池块永远拿不到分**。全库排名实测 —— 语义向量：top-20 31.7% / top-50 50.0% / top-100 55.0% / top-500 76.7% / 全库 **100%**（best-rank p50=54, p75=379, p90=1665）；BM25 词法：top-20 11.7% / top-50 15.0% / top-100 25.0% / top-500 35.0% / 全库 43.3%（p50=81）。
> - 方案：**保留融合头部 + 余弦补位**。全量 300 题 strict **195 → 207（+12）**，inject +1，负样本零恶化；已落地为**默认关闭**的生产开关并完成回归验证。

### 2.2 他们遇到的同型「召回/精度涨但 MRR 跌」：可见性过滤，两次判负

**`docs/compendium-rag/entity_disambiguation_rules.md:200–205`（原文）**：

> 2. **名字键与章节键都标不住私密信息**。可见性过滤在三套 MRR 上实测判负：300cn 全集角色域
>    **Δ MRR@10 = −0.1296**，"该章出现过即可见"的松键只挽回 0.018；
>    机制上误伤全部来自**转述型块**（A 的事由别人说出，A 的名字不在块里）——
>    那批题 Δ = **−0.2424**，占角色域题 56.2%，而非转述型 +0.0151 几乎不受损。
>    ⇒ **归属必须是显式标注的关系，不能从"名字是否出现"推断**；这也正是消歧表里
>    「面板只渲染规范名会造出假错张」的那同一枚硬币：**别名/转述是表示层的地雷，也是归属层的地雷**。

**`docs/compendium-rag/CHRONICLE.md:145`（原文）**：

> - 结论：可见性过滤**三套 MRR 全负**。300cn n=196 / 可用 153 / MRR 0.66528，**Δ −0.12963**（去最高单题 −0.11151），hit 0.8824 → 0.6340；blind206 n=206 / 181 / 0.68080，**Δ −0.09926**（−0.07922），0.9227 → 0.7459；gen120 n=120 / 117 / 0.82188，Δ +0.00157（−0.00812），1.0000 → 0.9573。

**角色域影子（另一条同型裁决）** —— `docs/compendium-rag/优化计划_v9_2026-10-08.md:326–335`（原文，含表）：

> | 集 | w=0 | w=0.25 | w=0.5 | w=0.75 | w=1.0 | 窗内块 | 判为跨域 | 跨域且 gold | gold 占跨域 |
> | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
> | 300cn | 0.5300 | 0.5267 | 0.5267 | 0.5233 | 0.5267 | 1800 | 1254（69.7%） | 353 | **28.15%** |
> | blind206 | 0.8398 | 0.8398 | 0.8398 | 0.8398 | 0.8350 | 1236 | 701（56.7%） | 122 | **17.40%** |
> | gen120 | 0.9833 | 0.9833 | 0.9833 | 0.9833 | 0.9833 | 720 | 199（27.6%） | 59 | **29.65%** |
>
> - **判负发生在最小非零档**：`w=0.25` 就掉 1 题（300cn/R257），`w=0.75` 掉 2 题，`w=1.0` 在 blind206 掉 1 题；全程只有 1 题净增（R068，且只在 w=1.0），**没有任何一档做到「不掉 gold」**。登记硬检查的写法是「任何 gold 掉出 top-k 即判负」，按它执行就是判负，不许把「只掉 1 题」重新解释成可接受。

### 2.3 「单通道比融合好」——与我方「bm25 单路 MRR 0.4248 > baseline 0.4123」形态相同

**`scripts/_fusion_restructure.py`（docstring 原文）**：

> 关键发现深挖：**词面通道单独 0.9320 > 融合 0.8301（盲测 206）**。为什么会这样（机理）：RRF 给每个通道的前 1 名都是同样的 1/(60+1)。当 graph/mention/vector 把**大量**候选排到各自前列（它们各自的前 6 都在 top-6 里拿高分），融合后的 top-6 会被这些通道"分票"，把词面通道真正命中的块挤出前 6。**实测权重扫描（1→16）无法解决（各档均 ≤ 融合现状）**。

**`src/compendium_rag/retriever.py:724–731`（注释原文，即他们实现的解法）**：

> 融合重构（默认关闭，lex_first=0 时行为与接线前逐位一致）：
> 把词面通道的前 lex_first 名**提到最前**，其余按融合序补足（去重）。
> 动机（2026-09-28 实测）：RRF 是排名制，graph/mention/vector 会把词面通道真正命中的块挤出前 6 —— 盲测 206 上 **lex-only 0.9320 > 融合 0.8301**。
> 实测 lex_first=3（仅在 300cn 上选定）：裸检索 300cn 0.8316->0.8469、盲测 0.8301->0.9126（两集同升）。
> ⚠ 与**冻结重排**的交互是集合相关的（见 docs/目标①_收口报告），故默认关闭，启用需按 §4bis 冷启动重新校准。

**但这条最终被诚实修正** —— `docs/compendium-rag/CHRONICLE.md:603`（原文）：

> - **① `lex_first` 保持关闭**：v7 的 `k=1` 已让四集裸检索全部 ≥0.85，`k=1 + lex_first=3` 在 300cn 裸检索**零增益**（167 不变）、生产口径零变化（0.8673 不变），仅盲测 +3 题。**诚实修正**：前序记载「`lex_first` 曾使 300cn 生产口径 0.8673→0.8418」在 k=1 之上**未复现**——该跌幅结论**仅成立于 k=60**。故本条理由不是「它有害」而是「它零收益」。接线与 `_lexfirst_regression.py` 保留在库，默认关闭。

### 2.4 「扩池无效、权重无效，只有名次折扣 k 有效」

**`docs/compendium-rag/baseline/BASELINE_v7_20260928.md` §3（行 44–67，原文含表）**：

> ## 3. 为什么之前没发现（本轮的核心洞察）
>
> 前序会话（`docs/目标①_池内miss干预报告`）把"调参不可救"的结论建立在**权重扫描**上：
>
> | 干预 | 结果 |
> |---|---|
> | `lex_top` 30→60/100 | 0 救回 |
> | `vector_top` 30→50/100/200 | 0 救回 |
> | **`vector` 权重 2/3/5** | 0 救回 |
> | `lex` 权重 1→16 | "完全无法修复" |
>
> **但它们从未扫过 RRF 的排名折扣常数 k 本身。**
>
> `rrf_fuse` 的打分是 `w × 1/(k + rank)`。k 决定**头部名次的相对权重**：
>
> | k | rank1 | rank6 | 头名相对第 6 名的倍率 |
> |---|---|---|---|
> | 60（历史） | 1/61 = 0.0164 | 1/66 = 0.0152 | **1.08×** |
> | **1** | 1/2 = 0.500 | 1/7 = 0.143 | **3.50×** |
>
> 即：k=60 时"排第 1"与"排第 6"几乎等价，于是**单通道霸榜**的候选能压过
> **在多个通道都排前列**的候选；k=1 把这种区分放大 3.5 倍，让"多通道共识"胜出。
> 这正好解释了为什么调**权重**无效（权重是全局缩放，不改通道内相对名次），
> 而调 **k** 有效（k 改的是名次折扣曲线）。

配套实测 `data/compendium_rag/reports/_85_sweep.json`（裸检索 hit@6，300cn n=196 / blind n=206）：v6 baseline `165/172`；**k=1 → `167/184`**；k=3 → `165/185`；k=5 → `166/181`；k=10 → `166/180`；k=20 → `165/175`；k=45 → `165/172`；k=60 → `165/172`；k=200 → `165/172`。

**同一个结论在另一条产品线（plot）独立得出过，且写进了配置注释** —— `src/roleplay/config.py:232–234`（原文）：

>     # k 必须取**小值**：k=60（RRF 论文默认）时 1/(60+rank) 几乎不随名次变化，
>     # 通道权重会完全压过名次——实测 k=60 时边证据霸占全部名额（路径占比 87%），
>     # 严格通过率只有 42%；k=3 时两路真正交错，回到 46%（=旧实现水平）。

⇒ **两条线、两套语料、两个团队上下文，独立得出「k=60 是错的，必须取小值」。** 这条在我方已成立（k=1 最好），可以就此封档。

**「RRF 是纯排名制」的定性** —— `scripts/_p3_pool_widen.py`（docstring 原文）：

> 机理（本轮实测推出的结构性结论）：**RRF 是纯排名制**融合：某候选若在**每个**通道都排 14 名开外，则无论权重怎么调，它都进不了 top-6（top-6 必然是该通道的前几名）。实测佐证：把 vector 权重提到 5.0，7 道语义可达题**依然 0 命中**，且全量 down 从 0 涨到 7 —— 权重不是解法。
>
> 但 7 道题的 gold 在**向量全库排名 14–50**，即**方向是对的、只是被排名制挡住**。所以正解是：把 RRF 的候选池（top_n）放宽，再把整个池交给**打分制重排器**（其 `cos` 特征直接用余弦相似度，能把向量排名靠前的候选顶上来）。

### 2.5 「整体重排会毁掉互补性」——余弦同质化

**`plot_graph.py:1843–1857`（`_rerank_topup` docstring 原文）**：

> 为什么是补位而不是整体重排（2026-09-17 全量实测，300 题）：
>   - **整体余弦重排**：strict 195→196（+1），但 inject 177→**151（−26）**。
>     原因：余弦偏袒「与 query 整体最像」的块，选出的 8 块高度同质，丢掉了
>     RRF 多路命中带来的**互补证据**；而 inject 口径（2400 字符装配）恰恰
>     需要多块互补才能覆盖全部证据组。
>   - **保留前 3 名 + 余弦补位**：strict 195→**207（+12）**、inject 177→178（+1）、
>     负样本失守 7/53 条与注入 54 块**均无变化**。
>
> 为什么能救回失分题：失败 105 题里 60 题的正解块从未进入最终预算，但它们
> 100% 存在于各通道候选池之外的语义空间里（全库排名 p50=54）。RRF 只按**名次**
> 给分，深池块永远拿不到分；补位绕开名次，直接用余弦把这些块捞进剩余名额。
>
> 分数带：RRF 把分数归一到 ``[FUSE_MIN_SCORE, 1.0]``，补位块的分数必须**低于**
> 被保留的头部块，否则名次被打乱、``rerank_topup_keep`` 就失去意义。

### 2.6 「重排器判别力不足以兑现上界」

**`docs/compendium-rag/CHRONICLE.md:607–608`（原文）**：

> - **③ 不立项 P4，P3 列为唯一后续优先**：v7 有效 miss 19 道 → A 组「gold 在池内、仅名次深」13 道（池内名次 6–24，重排可救）、B 组「gold 不在池内」6 道（R094/R244/R248/R292/R297/R298，重排救不了）。池上限 hit@30 = 183/196 = 0.9337（有效 178/184 = 0.9674）。
>   - P4 理论上界 0.9184（救回 A 全组）**属超额**；成本 fp32 `bge-reranker-large` ~110s/题（206 题 ≈ 6.3 h），ONNX int8 快 6 倍但**增益完全消失**；实测 CE 对池内 rank 34–51 的新进池题只救回 1/6 ⇒ 判别力不足以兑现上界。

### 2.7 「重排后召回涨但注入跌」——与我方 MRR 权衡同类

`data/knowledge/_audit/ab_ce_rerank_k30.json`（CE k=30，300 题，主评测口径）：baseline `strict_pass_n 195 / inject_pass_n 177 / latency_p95_ms 497.0`；CE → `strict_pass_n 201 / inject_pass_n 174 / ce_latency_p50_ms 990.5 / ce_latency_p95_ms 13005.4 / elapsed_s 1520.2`；`delta_vs_baseline {d_strict 6, d_inject −3, d_neg_blocks 0}`。

即：**+6 通过数换来 −3 注入通过，延迟 ×26**。前项目据此没有把它接进生产。

### 2.8 方法学：扫参前先证明离线重放器与官方同源

**`BASELINE_v7` §4（行 71 起，「方法学（本轮建立的信任基础，可复跑）」）**：`scripts/_85_validate_rrf.py` 用 collect 工件 `reasons` 重建 score **2400/2400 精确一致**（最大偏差 0.00e+00）；`_85_build_replay.py`/`_85_validate_replay.py` 四配置逐题一致；**k 只在 300cn 上选**（k=1 与 k=2 同分 167，取更小者），盲测 206 为留出集；最后真冷启动 `_85_verify.py` 逐题 up/down 与离线预测完全一致（产物 `_85_verify_k1.json`：300cn `167/196=0.852`、blind `184/206=0.8932`、gen120 `118/120=0.9833`、`unanswerable_hits 1/104`、`grounded 1.0`）。

**配套的「样本内 ≠ 系统能力」纪律** —— `scripts/_frozen_prod_eval.py`（docstring 原文）：

> **冻结配置外样本检验**：把 300cn 上选定的生产重排配置，原样应用到盲测题集。为什么必须这样做：``_rank_rerank_lab.py`` 会在**传入的任何题集上重新拟合**权重与 α。因此"盲测集上 lab=0.94"是**样本内**数字，不是系统能力。真实问题应当是：把在 300cn 上冻结的配置拿到盲测题上，还能得多少？

---

## 3. 他们的检索排序公式（完整抄出）

### 3.1 plot 线（roleplay 生产路径，`src/roleplay/core/knowledge/plot_graph.py`）

**常量（行 71–76 原文）**：

```python
# 融合后把分数映射到 [0.6, 1.0]：下界必须高于 persona_prompt.MIN_CONTEXT_SCORE(0.05)，
FUSE_MIN_SCORE = 0.6
FUSE_MAX_SCORE = 1.0
FUSE_PATH_PRIORITY = {"edge": 5, "mention": 4, "semantic": 3, "chapter": 2, "lexical": 1}
```

**融合核心（行 1278–1305 原文，节选关键行）**：

```python
        for path, weight, items in channels:
            if weight <= 0:
                continue
            for rank, (_s, chunk, extra) in enumerate(items):
                h = str(getattr(chunk, "hash", "") or "")
                if not h:
                    continue
                if h not in fused:
                    order.append(h)
                    chunk_by_hash[h] = chunk
                    meta_by_hash[h] = dict(extra or {})
                    paths_by_hash[h] = []
                fused[h] = fused.get(h, 0.0) + weight / (self.fuse_k + rank)
                if path not in paths_by_hash[h]:
                    paths_by_hash[h].append(path)
        if not order:
            return []
        top = max(fused.values()) or 1.0
        out: list[tuple[float, object, dict]] = []
        for h in order:
            meta = meta_by_hash[h]
            best = max(paths_by_hash[h], key=lambda p: (FUSE_PATH_PRIORITY.get(p, 0), p))
            meta["path"] = best
            meta["paths"] = sorted(paths_by_hash[h])
            out.append((FUSE_MIN_SCORE + (FUSE_MAX_SCORE - FUSE_MIN_SCORE) * (fused[h] / top),
                        chunk_by_hash[h], meta))
        # 稳定排序：同分保持通道内名次（order 的插入顺序即"边证据优先、再提及"）
        out.sort(key=lambda item: -item[0])
```

**公式（按代码逐项展开）**：

```
fused(h) = Σ_c  w_c / (fuse_k + rank_c(h))            # rank 从 0 起算（enumerate 无 start=1）
score(h) = 0.6 + (1.0 - 0.6) * fused(h) / max_h fused(h)     # 线性映射到 [0.6, 1.0]
排序键   = -score(h)                                   # 稳定排序，同分保留通道插入序
meta.path = argmax_{p∈paths(h)} ( FUSE_PATH_PRIORITY[p], p )
```

**通道权重（`src/roleplay/config.py`，逐行原文）**：

| 参数 | 行 | 默认值 | 注释原文 |
|---|---|---|---|
| `plot_fuse_k` | 235 | `Field(3, ge=1, le=1000)` | 见下方注释块（行 232–234） |
| `plot_weight_edge` | 236 | `1.0` | `# 边证据（结构化关系路径）` |
| `plot_weight_mention` | 237 | `1.0` | `# 实体提及块（属性题）` |
| `plot_weight_lexical` | 238 | `0.6` | `# 词法兜底（BM25）` |
| `plot_weight_semantic` | 316 | `1.0` | — |
| `plot_weight_chapter` | 284 | `1.0` | — |
| `plot_weight_mention_entity` | 266 | `Field(1.0, ge=0.0, le=10.0)` | — |

**`plot_fuse_k` 的定义性注释（`config.py:228–234` 原文，与我方 RRF_K 直接对应）**：

```python
    # ---- 剧情检索排序（P2：三通道 RRF 融合）----
    # 三条通道分数量纲互不可比（边证据≈0.2 / 提及≈1.2 / 词法 BM25 30–75），
    # 旧实现直接混排排序，导致图结构证据被提及路稳压、且进不了提示词（组内 0.85 相对门）。
    # RRF 只用名次，天然免疫量纲问题。
    # k 必须取**小值**：k=60（RRF 论文默认）时 1/(60+rank) 几乎不随名次变化，
    # 通道权重会完全压过名次——实测 k=60 时边证据霸占全部名额（路径占比 87%），
    # 严格通过率只有 42%；k=3 时两路真正交错，回到 46%（=旧实现水平）。
    plot_fuse_k: int = Field(3, ge=1, le=1000)
```

**融合分派与两路配额（行 1782–1799 原文）**：

```python
        if lexical_items and graph_count:
            # 两路共存（topup）：图路占 top_chunks 名额，词法只补 lexical_fallback 个
            fused = self._fuse_channels(
                [
                    ("edge", self.weight_edge, edge_items),
                    ("mention", mention_weight, mention_items),
                    ("semantic", self.weight_semantic, semantic_items),
                    ("chapter", self.weight_chapter, chapter_items),
                    ("lexical", self.weight_lexical, lexical_items),
                ],
                limit=graph_limit + max(0, lexical_fallback),
            )
            graph_kept = [it for it in fused if it[2].get("path") != "lexical"][:graph_limit]
            lex_kept = [it for it in fused if it[2].get("path") == "lexical"][
                : max(0, lexical_fallback)
            ]
            fused = graph_kept + lex_kept
            fused.sort(key=lambda item: -item[0])
```

**补位重排的分数带（行 1899–1913 原文）**：

```python
        scored.sort(key=lambda t: (-t[0], t[1]))
        picked = scored[: budget - len(keep)]
        keep_min = min((float(s) for s, _c, _m in keep), default=1.0)
        raw = [s for s, _i, _c, _e in picked]
        lo, hi = min(raw), max(raw)
        span = (hi - lo) or 1.0
        ceil_score = max(FUSE_MIN_SCORE, keep_min - 1e-3)
        out: list[tuple[float, object, dict]] = list(keep)
        for s, _i, chunk, extra in picked:
            frac = (s - lo) / span
            score = FUSE_MIN_SCORE + (ceil_score - FUSE_MIN_SCORE) * frac
            meta = dict(extra or {})
            meta["rerank_topup"] = True
            out.append((float(score), chunk, meta))
        return out
```

**保底（行 1944）**：`rescue_score = max(FUSE_MIN_SCORE, min_score + 1e-3)`

**提及通道打分（行 1119–1127 原文）**：

> 打分（P2，模式由 ``mention_scoring`` 决定；消融结果见审计报告 §P2）：
> - ``"count"``（默认）：``score = Σ_名字 w(名)·出现次数``，即旧口径；
> - ``"idf"``：``score = Σ_名字 w(名)·idf(名)·(1+0.3·ln 次数)``，稀有名字加权。
>   实测在本语料上 **idf 反而更差**（严格通过率 −4pt：ASR 转写里"反复提到某角色"
>   的块通常正是该角色的主场戏，也就是答案块；稀有别名命中多为顺带一提）。
>   故默认回到旧口径，idf 作为可选模式保留，供换语料时重新评估。
> - ``cooccurrence_bonus``：块命中 k≥2 个目标实体时 ``×(1+0.5·(k−1))``——
>   多实体共现是关系题/多跳题答案块的特征；
> - ``seed_factor``：种子 1.0 / PPR 邻居 0.5（保持"种子优先"的既有语义）；
> - 每实体先取前 ``per_entity`` 块再汇总，与旧实现的候选收敛口径一致。

**预算语义（行 1689 注释原文）**：

> `# 预算语义（与旧实现一致）：**图路可用满整个返回预算**（top_chunks + lexical_fallback）；词法兜底只在图路不足时占用剩余名额。实测教训：把图路硬压到 top_chunks(=3) 会让返回块数从 4.83 掉到 2.86，严格通过率直降 9pt——证据块并集口径下，"少给块"就是"少覆盖"。`

### 3.2 compendium 线（`src/compendium_rag/retriever.py`，与我方 RRF 最像）

**`rrf_fuse` 全文（行 613–638 原文）**：

```python
    def rrf_fuse(channels: dict[str, dict[str, float]], k: int = 60,
                 weights: dict[str, float] | None = None,
                 top_n: int = 20) -> list[Evidence]:
        """Reciprocal Rank Fusion：按各通道排名融合，规避量纲不一致。

        确定性（2026-09-28 追加，作战书 §4bis「排序必须带显式次级键」）：
          两处排序都补了次级键 `chunk_id`。原因：RRF 分数是 `1/(k+rank)` 的加和，
          平局极常见 —— 实测 120 题中 **116 题**存在并列（单题最多 42 个元素并列），
          而原先只按分数排序时，平局先后完全由 **dict 插入序**决定，
          等于 top-6 的边界取舍是任意的（与 #39 Phase 2 修掉的 link() 同类缺陷）。
        """
        weights = weights or {}
        agg: dict[str, Evidence] = {}
        for ch in sorted(channels):          # 通道遍历序确定（不依赖调用方构造顺序）
            scored = channels[ch]
            if not scored:
                continue
            w = weights.get(ch, 1.0)
            ranked = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
            for rank, (cid, raw) in enumerate(ranked, start=1):
                ev = agg.setdefault(cid, Evidence(chunk_id=cid, score=0.0))
                ev.channels[ch] = raw
                ev.score += w * (1.0 / (k + rank))
                ev.reasons.append(f"{ch}#{rank}")
        out = sorted(agg.values(), key=lambda e: (-e.score, e.chunk_id))
        return out[:top_n]
```

**生产配置（`docs/compendium-rag/architecture.md` §5 配置面，行 441–457）**：`TOP_K/ENTITY_LINK_TOP/SEED_SCORE_THRESHOLD = 6/6/0.34`；`VECTOR_WEIGHT/GRAPH_WEIGHT/RRF_K = 1.0/1.0/60`（历史默认，**生产值 `RRF_K=1`**）；`LEX_WEIGHT/LEX_TOP/LEX_FIRST = 0.0（关）/30/0`；`CONTEXT_MAX_CHARS=3000`。生产口径最终为 `lex_weight=1.0 / lex_top=30 / lex_first=0 / rrf_k=1`（`architecture.md` 行 500–511）。

**`lex_first` 结构改法（`retriever.py:736` 原文）**：

```python
            for cid in lex_ranked[:lex_first] + [e.chunk_id for e in res.evidence] + lex_ranked:
```

### 3.3 离线盲特征重排公式（`scripts/_rank_rerank_lab.py`）

**策略族（行 45–46 等 docstring 原文）**：

>   full_topN(a)         按 (1-a)·norm_orig_rank + a·norm_blindscore 升序重排
>                        全池（norm_orig_rank=原位次/(N-1)；并列按原名次稳定

**实现（行 251–283，核心）**：

```python
        key = (1.0 - param) * rn - param * Ss      # rn = np.arange(pool)/max(1,pool-1)；Ss = zscore_row(S)
        return np.argsort(key, axis=1, kind="stable")
```

- `boundary_promote(m)`：`lo = K_MAIN - 1`（槽 6 的 0 基下标）；`best = lo + np.argmax(Ss[:, lo:pool], axis=1)`；`cond = Ss[np.arange(n), best] > Ss[:, lo] + param`；命中则**仅互换槽 6 与胜者**（多重集不变）。
- `top1_thresh(t)`：min-max 归一分 `rel = (S - mn)/span`；`cond = rel[np.arange(n), best] >= param * rel[:, 0]`。
- `zfit` 坐标上升（行 330–347）：`for _p in range(2): for j in range(F): for v in (0.0, 0.5, 1.0, 2.0)`，目标 `obj(w)` = `(int(hit6.sum()), float(rr6.sum()))`；网格 `{0, 0.5, 1, 2}`×2 轮 + LOO-CV 过拟合检查。

**10 个盲特征（行 204–221 逐项）**：`cos`（块向量已 L2 归一化后与 query 点积）、`lex`（query bigram 覆盖率 `len(inter)/len(qb)`）、`lex_idf`（`sum(q_idf[g] for g in inter)/qb_total`，`_idf(g)=log((n_chunks+1)/(1+df[g]))+1.0`）、`lex_title`（篇名 blob 的 bigram 覆盖）、`orig`（原 RRF 分）、`ppr`、`has_vec`、`seed`（种子名子串是否在块文本）、`chunk_len`、`log_deg`（块挂载实体 `log1p(deg)` 最大值）。可选 `extra`（外部逐对分数，CE 走这条路）。

**实测最优权重（`rank_rerank_lab_300cn_p30_ce.json`，pool=30，n=196 answerable）**：

- baseline：`hit6 0.801 / hit10 0.8622 / mrr6 0.6147 / mrr10 0.6225`
- chosen `full_topN[zfit]a=1.0`，weights = `{"cos":2.0,"lex":2.0,"lex_idf":0.0,"lex_title":1.0,"orig":0.0,"ppr":0.0,"has_vec":2.0,"seed":1.0,"chunk_len":0.0,"log_deg":0.0,"extra":1.0}`
- → `hit6 0.8776 / hit10 0.8929 / mrr6 0.7599 / mrr10 0.7617`，up 18 题 / down 3 题
- `loo_cv`：`cv_hit6 0.8622 / cv_mrr6 0.7536 / overfit_gap_hit6 0.0154`
- `holdout`（native 250 题，配置原封不动打过去）：baseline `hit6 0.632 / mrr6 0.4264` → policy `hit6 0.66 / mrr6 0.5485`，`delta_hit6 +0.028 / delta_mrr6 +0.1221`
- `ablation`（逐个丢特征）：`has_vec −0.0256 / lex −0.0154 / seed −0.0154 / extra −0.0154 / cos −0.0103 / lex_title −0.0052`

同族（无 CE 的 `rank_rerank_lab_300cn_p30.json`）：`hit6 0.8776 / mrr6 0.727`，up 17 down 2，`cv 0.8724 / 0.7268`，holdout `delta_hit6 +0.016 / delta_mrr6 +0.0725`。

---

## 4. 他们怎么用重排器

| 维度 | 交叉编码器（CE，`plot_rerank.CrossEncoderReranker`） | 余弦补位（`_rerank_topup`） | 盲特征线性重排（`_rank_rerank_lab` 离线） |
|---|---|---|---|
| 模型 | `bge-reranker-base-onnx` / `onnx/model_int8.onnx`；`_p4_ce_subset` 另试 `bge-reranker-large` fp32 | 无模型，用 `semantic.vectors` 点积 | 无模型，纯 numpy |
| 候选数 | `ab_ce_rerank.py` 标定 `k ∈ {20,30,50}`，实跑 `pool_avg 20.4`（k=30） | 五通道并集去重后取 `budget - keep_n` 个 | `--pool` 默认 10，上限 `POOL_MAX = 30` |
| 截断 | **token 级 `max_len=512`**（`ids = ids[: self.max_len]`，行 126）+ **字符级 `max_chars=1200`**（`texts.append(t[: self.max_chars])`，行 149），两级串联 | 无截断（直接对块全文点积？—— 实为对 `sem.vectors` 行点积，向量是建库时对整块编的） | 无截断（cos 复用块向量；lex 用块 blob 的 bigram） |
| 批处理 | **`batch` 恒为 1**：`np.array([ids])`（行 131–132）逐对串行 | 全池一次性矩阵乘 | 全池一次性矩阵 |
| 保留头部 | 全量重排（替换整个次序） | `keep_n = min(self.rerank_topup_keep, budget)`，**前 3 名不动**（构造函数默认 3；但 `config.py:352 plot_rerank_topup_keep=5`，两处默认值不一致） | 由 `a` 控制：`a=0` 等价基线，`a=1.0` 纯盲分序 |
| 延迟 | `ce_latency_p50_ms 990.5 / p95 13005.4`（300 题，k=30，主评测口径）；`ce_timing.json` 单对 `19.77 ms`；`ce_batch_timing.json` 10/20/30/40 候选 = `176.55/379.64/534.23/727.54 ms`；`_p4_ce_onnx_subset.json` `seconds_per_q 18.481`（batch=16 时） | 一次矩阵乘，未单独计时（`_rank_rerank_lab` 的 reorder p95 = 0.244 ms 属同类量级） | `feature_ms_mean 18.312 / p95 22.412`；`reorder_ms_mean 0.211 / p95 0.244` |
| 异步 | **未找到** | **未找到** | **未找到** |
| 缓存 | **未找到**（无跨请求缓存） | 复用了进程内 `_sem_hash2row` 映射缓存（`:1818`「构建一次后缓存」） | 复用 `_rank_collect.py` 落盘工件（「此后全部实验零检索、零网关」） |
| 失败降级 | 「任何异常（加载失败/推理异常/超时）→ **原序返回**」；`_ensure()` 失败后 `self._err` 记住不再重试（`:78–113`） | 无异常路径，缺向量就 `return fused` | 无（离线） |
| 开关 | 默认不开；`plot_rerank_topup_enabled=False` | 同左，`plot_rerank_topup_enabled: bool = False`（`config.py:351`） | 离线，不进生产包 |

**关键结论（对手方实测）**：

1. **CE 在 int8 上增益归零**：`_p4_ce_subset.json`（fp32 `bge-reranker-large`，n=26）`ce 0.5`；`_p4_ce_onnx_subset.json`（int8，n=26）`ce_hit6 0.4231` ⇒ 掉 2 题（`R248`、`BD-050`）。CHRONICLE:608 的原话是「ONNX int8 快 6 倍但**增益完全消失**」。
2. **同一批题的延迟-收益比**：`ce_timing.json` 单对 19.77 ms，k=30 时约 530 ms；`ab_ce_rerank_k30.json` p50 990.5 ms / p95 13,005.4 ms。CHRONICLE:211 给的成本账：「bge-reranker-large fp32 13/26 @ ~110 s；bge-reranker-base ONNX int8 11/26 @ 18.5 s（全量约 64 分钟）」。
3. **离线脚本 batch=16，生产接口 batch=1**：`_p4_ce_onnx_subset.py:54 def score(self, pairs, batch=16)` / `:131 ce.score(pairs, batch=16)`，而 `plot_rerank.CrossEncoderReranker.score_pair` 一次只喂 `np.array([ids])`。**这是「同一模型、两个延迟数」的一个确定来源**，抄代码时别只看一个数。

---

## 5. 他们的评测口径

### 5.1 Recall / hit / MRR 的定义（按项目分两套）

**(a) compendium 官方 eval300** —— `scripts/compendium_rag_eval300.py:1–23`（docstring 原文）：

> 口径设计（重要）
> ----------------
> 这 300 题是为**旧语料**（B站视频转写 + 灰机 wiki，含英文名）造的，而
> Compendium 是**纯简体中文官方正文**。实测约 44% 的题目标注的答案关键词
> （Zanetta / Charlotte O'Hagan / Marcus …）在 Compendium 里**根本不存在**——…
> 因此本评测把 300 题**分层报告**，而不是只给一个总分：
> 1. ``all``            —— 全部题（含不可答的，最保守的口径）
> 2. ``answerable``     —— 答案关键词在 Compendium 中出现过的题（**主口径**）
> 3. ``cn_keyword``     —— 其中期望关键词是**中文**的题…
> 4. ``unanswerable``   —— 关键词完全不在语料中的题（**只报不评**，…）
>
> 指标：``hit@k``（top-k 证据块中命中任一期望关键词）、``MRR``、
> ``seed_recall``（期望实体是否被链接）、``grounded``（证据是否真出自语料）。

打分实现（行 95–115）：`for rank, ev in enumerate(res.get("evidence", [])[:top_k], start=1)`；命中判 `if b and any(k in b for k in kws)`；`if rr == 0.0: rr = 1.0 / rank` —— **取首个命中名次**（所以 MRR 的分母是「第一个含关键词的块」。`hit@k` 与 `MRR` 的实现分别在 `:139` `:140`）。

**(b) plot 官方 `evaluate_case`** —— `scripts/eval_plot_rag.py:341–353, 389–394`（原文）：

>     """单题检索 + 打分（严格口径与灰机评测一致：并集子串命中）。
> …
>     为什么必须单列这一口径（M7 审计发现）：``strict_pass`` 统计的是**返回块并集**，
>     而产品真正交给模型的是 ``plot_context_max_chars``（默认 1000 字符）以内的文本。
>     语料块约 600 字/块 ⇒ 实际只装得下 1–2 块，两者口径相差甚远。
>     只看 strict_pass 会把"多返回块"当成质量提升（实测 top 3→8 能把 strict 从
>     47% 拉到 57%，但注入文本一个字都没变）——那是**指标灌水**，不是产品变好。
>     """
>
>     # MRR：第一个「全证据组覆盖块」的名次（1 基）；无则 0
>     mrr = 0.0
>     for rank, h in enumerate(hashes, start=1):
>         if h in full:
>             mrr = 1.0 / rank
>             break

同时 `:400–404` 记录了另一个抗灌水设计：「strict 195→214（+19），而 inject **完全不动**——纯灌水。」

### 5.2 名次记账与显著性（`scripts/_rank_stats.py:1–34`，原文）

>   - 按共同 qid 配对（同 mode、同 stratum 过滤后）。
>   - McNemar 精确双侧：b=base 未中而 cand 命中、c=base 命中而 cand 未中
>     （与 _er_report_gates 的 up/down 同序），p = 2·Σ_{i<=min(b,c)} C(n,i)/2^n。
>   - hit 类指标直接做 McNemar；mrr6 是连续值，McNemar 按 rr6>0 二值化
>     （输出注明），主报告为配对 Bootstrap Δ（均值差）百分位 CI。
>   - Bootstrap：10k 次有放回重采样（默认，--boot 可调），固定种子可复现。

`rank_stats_final.json` 应用实例：300cn answerable hit6 `b=17 c=2 p=0.000729 delta +7.65pt CI[3.57,12.24]`（显著）；native hit6 `b=22 c=18 p=0.635828 +1.6pt`（**不显著**）。

### 5.3 「召回涨但 MRR 跌」的裁决记录（共 5 条，全部留档）

| # | 场景 | 裁决 | 原文出处 |
|---|---|---|---|
| 1 | 可见性过滤：hit 与 MRR 同时掉 | **判负，两次** | `entity_disambiguation_rules.md:200–205`；`CHRONICLE.md:145`；`P10口径裁决` §2.4 |
| 2 | 角色域过滤影子：w 五档全部掉题 | **判负，参数不落地** | `优化计划_v9:326–335` |
| 3 | max_edges 截断放宽：MRR 单调涨 +0.0104/+0.0198/+0.0500 | **仍不改默认值** | `优化计划_v9:261–270` |
| 4 | 规范化改动：主口径涨，MRR 两降一升 | **判 PASS，但「记在账上」** | `优化计划_v9:128–140` |
| 5 | 重排：hit 涨 inject 跌 | **不接生产** | `ab_ce_rerank_k30.json`、`plot_rerank docstring` |

**裁决 #2 的方法学原文（最值得抄的一条）** —— `优化计划_v9:332`：

> 登记硬检查的写法是「任何 gold 掉出 top-k 即判负」，按它执行就是判负，不许把「只掉 1 题」重新解释成可接受。

**裁决 #3 的预注册文** —— `优化计划_v9:265, 270`：

> 改默认值属于**口径变更**：真要改，必须按 §4bis 冷启动重校准（三集 + 安全线 + 与冻结重排的交互）且 L2.3/独立量不回退；**不许用 hit@6 的涨幅当理由**（[[fix-data-not-metrics]]）。曲线只用来报告，不用来选档。
>
> **裁决：默认值维持 30，本轮不动。**…不许拿这次的 MRR 涨幅回头论证改默认值。

**裁决 #4 的「记账」文** —— `优化计划_v9:136–140`：

> 副口径与 MRR 一并如实报，不挑好看的：`graph_only` 上 **blind206 掉了 1 题**
> （174→173，0.844660→0.839806），300cn `graph_only` +1 题、gen120 `graph_only` +2 题；
> MRR 三集里两降一升（300cn full −0.0028、gen120 full −0.0046、blind206 full +0.0011）。
> 硬门只看三个主口径，所以判通过；但「规范化让 1 题的图通道邻居块掉出前 6」这条**记在账上**，
> P7 的 v8→v9 delta 表要把它显式列出来，不许被主口径的 +0.005 盖过去。

**MRR 回退线的预注册** —— `P10修复计划_CBI收口.md:96`：

> 所以**必须先在 300cn/blind206/gen120 三套 MRR 上量误伤**，MRR 掉超过 0.02 就回退。

**MRR 闸门的适用域** —— `P9_CBI报告.md` §2.13（行 587–620）原文要点：

> **边界（这一条必须跟着 reciprocal_rank 一起引用）**：`MRR ≥ 0.39 不回退` **不能**在本测试集上读，两个独立理由：① 这套题里唯一的检索金标就是「B 的私密块」，那是我们**不希望**被找到的东西，它的倒数排名是**危险度**（越高＝泄漏越快），与 0.39 的方向相反；② A 侧被注入的私密按纪律只进 prompt 上下文、**不写图** ⇒ 图里没有 A 的对应块，这套题就没有"可答题"，质量 MRR 连分母都不存在。

同节还有一条对我方直接有用的观测：**命中不在榜头**——`v1 控制` 名次分布 `{2:1,3:2,4:3,5:4,6:2}`、均位 4.33、**第 1 名 0**；`v2 控制` 均位 4.21、第 1 名仅 1。即「MRR 低」有时是金标本身的属性，不是排序缺陷。

### 5.4 他们**没有**做的事（诚实标注）

- **没有**在线（请求路径内）的 MRR 门禁。冻结重排与 lab 全是离线；`_frozen_prod_eval.py` 是「把配置打过去量一次」，不是 CI 门禁。
- **没有**「MRR 优先于 hit」的选优函数。相反：`_rank_rerank_lab.py:437–441 pick_best` docstring 原文「确定性选优：hit@6 降序 → MRR@6 降序 → 名字升序（跳过 baseline/PENDING）」⇒ `key=lambda c: (-c["hit6"], -c["mrr6"], c["name"])`。**MRR 只是 tie-breaker。** 这与我方「MRR 比 R@20 值钱」的估值相反，是可**反用**的教训。

---

## 6. 明确无效的东西（他们试过并放弃，附读数）

| # | 做法 | 读数 / 裁决原文 | 出处 |
|---|---|---|---|
| 1 | 小模型 listwise LLM 重排 | 「qwen2.5:1.5b 全量 300 题 inject_pass 62.00% → 60.33%（−1.67 pt），失败模式是**把证据块往后挪**」 | `plot_rerank.py:17–19` |
| 2 | 扩大候选生成量 | 「**扩大候选生成量已实测无效**（9 轮消融，同批次串行）：base inject 0.6200 / strict 0.6767；`plot_semantic_top` 5→10 inject 0.6233 (+0.0033) / strict 0.6800；5→20 inject 0.6267 (+0.0067) / strict 0.6800，返回块数恒为 7.89。」 | `docs/plot-graph-rag/CHRONICLE.md:74` |
| 3 | 权重扫描 | `vector` 权重 2/3/5 → 0 救回；`lex` 权重 1→16「完全无法修复」；`_fusion_restructure.py`「**实测权重扫描（1→16）无法解决（各档均 ≤ 融合现状）**」 | `BASELINE_v7` §3 表；`_fusion_restructure.py` |
| 4 | 扩 RRF 候选池 | `pool 60 → 189/196=0.9643`（新增 7 题，池内排名 34–51），但「**+冻结重排 → 168/196 = 0.8571**，得 7 / 失 3，对 miss 集分别 0/6 与 0/14 —— 即所有干预都**没碰到达标关键的那批 miss**。」 | `CHRONICLE.md:222` |
| 5 | 六类池内 miss 干预 | 「14 道池内 miss 经六类干预（lex_top 扩大 / 章节通道 / vector_top 放宽 / vector 权重提高 / 放宽 RRF 池 / 放宽池+冻结重排）**均未能救回任何一道**。」 | `CHRONICLE.md:221` |
| 6 | ONNX int8 CE | 「ONNX int8 快 6 倍但**增益完全消失**」；同 26 题子集 fp32 `0.5` → int8 `0.4231` | `CHRONICLE.md:608`；`_p4_ce_subset.json` / `_p4_ce_onnx_subset.json` |
| 7 | `lex_first`（无条件版） | 在 k=1 之上「**零增益**（167 不变）、生产口径零变化」；独立复算里 blind 生产口径 `rerank_hit6 0.7427`（vs `pool_hit6 0.9126`）、`down=37`、`verdict.blind_no_regress=false` | `CHRONICLE.md:603`；`_p3_lexfirst_refit.json` |
| 8 | `additive`（逐通道追加 lex 名次） | 全 8 档（L=1…40）结果**完全相同** `165/172` | `_85_sweep.json` |
| 9 | `hoist_missing`（无条件提升缺失块） | L=8 时 `h300 156 / hblind 142`（相对 v6 `165/172` 崩掉） | `_85_sweep.json` |
| 10 | idf 加权提及打分 | 「实测在本语料上 **idf 反而更差**（严格通过率 −4pt）」 | `plot_graph.py:1122–1124` |
| 11 | 余弦**整体**重排 | 「整体余弦重排：strict 195→196（+1），但 inject 177→**151（−26）**」 | `plot_graph.py:1844` |
| 12 | 语义/篇章保底（rescue floor） | 「融合层两个杠杆实测**都没拉起 head 指标**…**救援兜底「无害但无效」**」；`min-semantic-kept 2 + min-chapter-kept 1` → 197/300（vs 195），inject 不变 | `docs/plot-graph-rag/CHRONICLE.md:46` |
| 13 | HippoRAG 式 PPR 引导 dense 排序 | 「闸门判定：不达标，不落地」 | `docs/plot-graph-rag/CHRONICLE.md:476` |
| 14 | bge-reranker 精排（T1 标定） | 「闸门判定：不达标，不落地」；「k=20/50 未跑：k=30 已两项不达标，按闸门逻辑停止」 | `docs/plot-graph-rag/CHRONICLE.md:466, 473` |
| 15 | 绝对余弦阈值过滤实体链接 | 「真阳实体的余弦分数 **低于** 负样本：真阳 p50 = 0.512，负样本 top1 p50 = 0.656（p10 也有 0.554）。任何能接住真阳的阈值（≤ 0.45）会让 **100% 的负样本**被链接上；阈值提到 0.60 能挡掉 25% 负样本，代价是真阳接住率归零。」 | `docs/plot-graph-rag/CHRONICLE.md:328–331` |
| 16 | 查询改写（多路）端到端 | ORACLE 上界 `strict 18 → 23`（+5），但端到端 `strict 18 → 18`、`inject 18 → 15`（−3），`d_strict 0` | `ab_qr_oracle_upper.json` / `ab_qr_e2e_mh40.json` |
| 17 | 收紧 k 来改指标 | 「**k 不是修法**：把 top-6 收到 top-3 会消掉 v1 的 12 条里 9 条…但 `retrieval_leak` 是检索通道在**固定 k=6** 上打的标，改 k 只改我报的暴露数，不改边界」 | `P9_CBI报告.md` §2.13 |
| 18 | 「扩池越大越好」 | 「**继续扩池是无效的**——扩得越大，gold 的名次反而越靠后」；`max_edges` 扫到 300 仍无人入池（「加边引入的竞争块把 RRF 名次稀释回去，非单调」） | `ab_rerank_ceiling.py:12–13`；`CHRONICLE.md:509` |

---

## 7. 对我方第 2–6 条困境的逐条回应

> 前置声明：前项目主口径是 `top_k=6` / hit@6 / MRR@6（`K_MAIN = 6  # 主指标边界：hit@6（对齐 eval300，不可调）`），我方是 `top_k=20` / R@20。**MRR@6 与 MRR@20 不是同一个量**：MRR 对 k 敏感（k 越大分母越大、值越小），跨项目比 Δ 可以，比值不可。以下所有引用都只用来搬「机理与判据」，不用来搬「数值」。

### 7.1 对第 2 条（RRF 三通道 R@20 +0.0361 / MRR −0.0753）

**他们的同型观测**：`ab_ce_rerank_k30.json` 的 `d_strict +6 / d_inject −3`；`plot_rerank docstring` 的「整体余弦重排 strict +1 / inject −26」；`CHRONICLE.md:122` 优化轮的「块精度 0.356 → 0.338（−0.018）；**MRR 0.328 → 0.326**」而延迟改善 49×。

**建议怎么做（三步，按前项目踩坑顺序）**：

1. **先判「这是排序缺陷还是口径缺陷」**。前项目有一个可复用的信号判据（`architecture.md:536–556` §7.2 原文）：`seed_hit` 全 300 题逐题相同、`n_evidence` 也相同，只有 18 题 hit 不同 —— 「这形态指向**排序/融合**而非**检索覆盖**」。我方 baseline 是 `bm25 + 0.5*entity` 两路，RRF 版是三路，`n_evidence` 必然不同，所以**这条信号在我方不成立，必须换判据**：直接用 `_rank_triage.py` 的三分诊（`pooled_in_window / corpus_not_pooled / no_corpus_match`）。若 `pooled_in_window` 占绝大多数（按 bm25 单路 gold 命中率 0.9367 推断应是如此，但**必须实测**），则 MRR 掉了 0.0753 是纯排序问题，与召回无关。
2. **不要用「R@20 涨了 0.036」给自己发许可**。前项目对同类诱惑的裁决是预注册的：「**不许用 hit@6 的涨幅当理由**」（`优化计划_v9:265`）。我方纪律「只有一个指标变好不算数」等价于这条，但前项目把它做成了**可执行的句式**：任何参数/结构变更申请，若主口径涨而副口径（MRR）跌，必须先写「MRR 跌多少、为什么可接受」，而不是写「R@20 涨了多少」。
3. **把 MRR 的「第一名占比」单列出来报**。前项目 `P9_CBI报告.md` §2.13 的做法是给**名次分布直方图**（`{2:1,3:2,4:3,5:4,6:2}`、均位、第 1 名计数）。我方 R@20 在 20 里，MRR 只看第 1 命中；两者对不上时，名次分布能立刻显示是「整体后移」还是「头名塌陷」——**后者才是 RRF 稀释词面通道的典型形状**（见下条）。

**证据来源**：`data/knowledge/_audit/ab_ce_rerank_k30.json`；`src/roleplay/core/knowledge/plot_rerank.py:4–19`；`plot_graph.py:1843–1849`；`docs/plot-graph-rag/CHRONICLE.md:122–123`；`docs/compendium-rag/architecture.md:536–556`；`优化计划_v9_2026-10-08.md:265, 270`；`P9_CBI报告.md` §2.13。

### 7.2 对第 3 条（42 组合 k×子集，无一种能让 R@20 与 MRR 同时不劣）

**他们的同型观测 + 定性结论**：这就是 `scripts/_p3_pool_widen.py` 那句「**RRF 是纯排名制**：某候选若在**每个**通道都排 14 名开外，则无论权重怎么调，它都进不了 top-6」。他们的同类二维/多维网格扫描也做了（`_85_sweep.json` 实测 **59 条**配置：k∈{1,2,3,5,8,10,15,20,30,45,60,80,120,200} × lexfirst L∈{1..15} × additive × interleave×stride × hoist_missing × k×L 组合），**结论是「网格里确实存在一个两集同时不劣的点（k=1），但它在生产重排口径下不可复现地脆弱」**（`_p3_lexfirst_refit.json` `blind_no_regress=false`）。

**建议怎么做**：

1. **不要再加维度，要换空间**。我方 42 组全部落在「RRF 名次折扣空间」内；前项目的结论是这个空间的天花板由 `1/(k+rank)` 的形状决定，改 k 只是改曲线斜率，改子集只是改求和项。**要突破必须离开名次空间**——他们的三条出路（按被验证顺序）：
   - **(i) 结构改法**（`lex_first` / `m2_union(3)`）：300cn 选参下两集同升（0.8316→0.8469 / 0.8301→0.9126），**但盲测生产口径崩 37 题**（`_p3_lexfirst_refit.json`）。→ 我方若试，必须先固定双集 + 先跑 B2 的可分性穷举。
   - **(ii) 打分制重排**（盲特征线性，`full_topN(a=1.0)`）：`hit6 0.801 → 0.8776`、`mrr6 0.6147 → 0.7599`，`cv_mrr6 0.7536`（过拟合 gap 仅 0.0154），holdout `delta_mrr6 +0.1221`。**这是他们唯一兑现且两指标同向的方案。**
   - **(iii) CE**：判别力不足 + 延迟不可接受（见 7.5）。
2. **给我方的「失败证明」补一条可上线判据**：前项目 42 组式扫描的产物里有 `chosen_a: []`（`_p3_ab_sweep.json`，九档 a 无一同时达标）这种**空集结论的明文留档**。我方扫完 42 组如果也是空集，应照此把「空集」作为正式结论登记，并附上「下一步只能换空间」的依据——而不是继续在网格里加 k 的小数。

**证据来源**：`scripts/_p3_pool_widen.py`（docstring）；`data/compendium_rag/reports/_85_sweep.json`、`_85_lexfirst_safe.json`、`_p3_ab_sweep.json`、`_p3_lexfirst_refit.json`、`rank_rerank_lab_300cn_p30_ce.json`；`docs/compendium-rag/baseline/BASELINE_v7_20260928.md` §3。

### 7.3 对第 4 条（bm25 单路 MRR 0.4248 > baseline 0.4123；vector 单路 R@20 0.3892 / MRR 0.2101；entity 单路 MRR 0.2255）

**他们的同型观测**（几乎一模一样）：「**词面通道单独 0.9320 > 融合 0.8301（盲测 206）**」（`_fusion_restructure.py`）；`retriever.py:724–731` 的动机段；以及 `CHRONICLE.md:603` 的诚实修正。

**建议怎么做**：

1. **把我方第 4 条当成「词面通道被融合稀释」的确证，而不是「bm25 更强所以干脆不用融合」**。前项目的教训很关键：他们一度把 lex-only 0.9320 > fused 0.8301 当作「融合有害」的证据，做了 `lex_first`；后来在 k=1 之上**零收益**，原话是「故本条理由不是「它有害」而是「它零收益」」（`CHRONICLE.md:603`）。所以正确的读法是：
   - bm25 单路 MRR 高 ⇒ **词面通道的名次要被保护**；
   - 但「保护名次」不等于「整体改成词面优先」——后者只在融合基线很差时才有效，一旦融合基线（k 或结构）改善，词面优先的增益会被吞掉。
2. **vector 单路（R@20 0.3892 / MRR 0.2101 / gold 命中 0.6467）应被降权，但不要按前项目的「向量可达性」逻辑去救它**。前项目的 `_p3_vector_weight_scan.py` 判据（原文）：「若提高 vector 权重能救回其中若干道，且**全量 down=0** -> 值得落盘；若全量下降或零救回 -> 否决，路由到 P4」。他们实测「把 vector 权重提到 5.0，7 道语义可达题**依然 0 命中**，且全量 down 从 0 涨到 7」（`_p3_pool_widen.py`）。我方 vector 单路质量远低于 bm25（MRR 0.2101 vs 0.4248），**大概率连「被稀释」的资格都没有，是纯噪声源**——建议做一次「去掉 vector 通道」的三档消融（bm25+entity / bm25+entity+vector / 三通道），并明确要求 down=0。
3. **gate 写法抄他们的**：`_rrf_k_regression.py` 的六条判据（原文）：「R1 default … R2 equality … R3 changed : k=1 与 k=60 的排序**确实不同**（证明旋钮生效）；R4 improves … R5 safety … R6 determin : 同一输入重复计算结果一致（带显式次级键）」。其中 **R3（证明旋钮真的生效）+ R6（确定性）** 是我方最容易漏的两条——尤其 R6，因为 RRF 平局极多（他们 `architecture.md:186–190`：120 题中 116 题存在并列，单题最多 42 个元素并列），次级键不显式就不可复现。

**证据来源**：`scripts/_fusion_restructure.py`；`src/compendium_rag/retriever.py:724–747`；`docs/compendium-rag/CHRONICLE.md:603`；`scripts/_p3_vector_weight_scan.py`、`_p3_pool_widen.py`；`scripts/_rrf_k_regression.py`；`docs/compendium-rag/architecture.md:186–190`。

### 7.4 对第 5 条（gold 命中率 0.9367 但 R@20 只有 0.6478 ——「gold 基本都在池里，排不进前 20」）

**我判定这是全篇对我方最有价值的一条**，因为前项目把这件事做成了**可复算的诊断套件**。我方第 5 条的形态与他们的 A 组高度相似，但**我方尚未做这项诊断，所以「同构」是待验证的假设，不是结论**。

**他们的诊断三件套**：

1. **三分诊**（`scripts/_rank_triage.py` docstring 原文）：「(i) pooled_in_window 关键词块就在采集证据池里 → 记最小名次；(ii) corpus_not_pooled 语料里存在关键词块但没进池 → 记 seed_hit、语料关键词块总数、以及"任一关键词块是否与任一池内块共享图实体"（桥接可及=说明池子/召回面的问题，桥接不可及=图覆盖问题）；(iii) no_corpus_match 全库没有任何块含关键词 → **按 answerable 规则必须为 0**」。
   实测（`rank_triage_300cn_v7.json`）：`n_misses 35, classes {pooled_in_window 21, corpus_not_pooled 14, no_corpus_match 0}`。
2. **池天花板**（`rank_rerank_lab_300cn_v7.json` `oracle`）：`pools.10 {recoverable 8, ceiling_hit6 0.8622} / pools.20 {18, 0.9133} / pools.30 {21, 0.9286}`。即「池里的 gold 有 21 个可以被顶进前 6，天花板 0.9286」。
3. **A/B 组分工判据**（`CHRONICLE.md:607`）：A 组「gold 在池内、仅名次深」→ 重排可救；B 组「gold 不在池内」→ 重排救不了。

**建议怎么做**：

1. **先算我方的三个数**（不改任何代码，只加诊断）：
   - `pooled_in_window` 占比（用 bm25 单路 channel_k=100 的候选池当「池」，与 baseline top-20 对比）；
   - `pooled_in_window` 那批题的**池内名次分布**（前项目给出 `stored_rank` 逐题值如 `R001=24 / R011=12 / R070=7 / R090=10 / R288=25`，我方应照此逐题落盘）；
   - **池天花板**：若「池内全部 gold 被顶到第 1」，R@20 与 MRR 各是多少。我方 0.9367 的池级命中率**若**成立，天花板应显著高于 0.6575——但这必须在「被排序的那个池」上重新算，不能拿 channel_k=100 的命中率当上界。**这个天花板才是重排的合理目标**，而不是「R@20 冲 0.75」。
2. **按 A/B 组分头做**：A 组（池内名次深）→ 走 7.5 的盲特征重排/补位；B 组（不在池）→ 走召回（但注意前项目实测「扩池无效」，我方 bm25 单路已覆盖 93.67%，B 组应该很小）。
3. **口径提醒**：前项目明确警告过「候选池命中 ≠ 最终通过」（`scripts/ab_query_rewrite.py` docstring 原文）：「本项目已经踩过一次这个坑（候选池扩到 42 块、gold 入池 19 题，最终 strict 只 +1）」。我方「gold 命中率 0.9367」也是**池级指标**，不能直接读成「只差排序」——必须经 `pooled_in_window` 确认它确实在**被排序的那个池**里，而不是在 channel_k=100 的外层池里。

**证据来源**：`scripts/_rank_triage.py`；`data/compendium_rag/reports/rank_triage_300cn_v7.json`、`rank_rerank_lab_300cn_v7.json`；`docs/compendium-rag/CHRONICLE.md:607`；`scripts/ab_query_rewrite.py`；`scripts/ab_rerank_ceiling.py:6–13`。

### 7.5 对第 6 条（CE 每对 95 ms，20 候选 1.9 s；截到 128 token 仍 362 ms；预算 P95 < 200 ms，差一个数量级）

**他们的实测数字（可直接当交叉验证）**：

| 场景 | 读数 | 出处 |
|---|---|---|
| 单对（生产接口 batch=1，`bge-reranker-base` int8） | `per_call_avg_ms 19.77` | `data/knowledge/_audit/ce_timing.json` |
| 10/20/30/40 候选（某批测口径） | `176.55 / 379.64 / 534.23 / 727.54 ms` | `data/knowledge/_audit/ce_batch_timing.json` |
| 端到端 300 题（k=30，主评测口径，含 `evaluate_case` 全流程） | `ce_latency_p50_ms 990.5 / ce_latency_p95_ms 13005.4`，`elapsed_s 1520.2` | `data/knowledge/_audit/ab_ce_rerank_k30.json` |
| int8 离线脚本（batch=16） | `seconds_per_q 18.481` | `_p4_ce_onnx_subset.json` |
| fp32 `bge-reranker-large` | 「~110 s/题（206 题 ≈ 6.3 h）」 | `CHRONICLE.md:608` |

**建议怎么做**：

1. **判定：CE 全量重排在 P95 < 200 ms 的预算下不可行，且即使可行也拿不到增益。** 两条独立理由：
   - **延迟**：前项目 20 候选 ≈ 380 ms、30 候选 ≈ 534 ms（`ce_batch_timing.json`），且那是**批测口径**；生产接口 `batch=1` 串行，端到端 p50 已到 990.5 ms。我方 20 候选 1.9 s 与他们的量级一致（我方块更长，95 ms/对 vs 他们 ~20–38 ms/对）。**这不是调参能解决的差距**。
   - **增益**：「ONNX int8 快 6 倍但**增益完全消失**」；同 26 题子集 fp32 `0.5` → int8 `0.4231`。我方手上的正是 int8，**前项目的实测直接否掉了我方手里的那个模型**。而 fp32 large 是 6.3 小时/206 题，不在 200 ms 预算的任何解释下可用。
2. **改走「无模型重排」，这是他们验证过且落在我方预算内的路线**：盲特征（cos / lex / lex_idf / lex_title / orig / ppr / has_vec / seed / chunk_len / log_deg）z-score 加权 + `full_topN(a=1.0)`，`feature_ms_p95 22.412` + `reorder_ms_p95 0.244`。读数 `hit6 0.8776 / mrr6 0.7599`，`cv_mrr6 0.7536`，holdout `delta_mrr6 +0.1221`。**这条的 MRR 增量（+0.14 绝对）反而比 CE 更大**，因为 CE 只被当成 11 个特征里的 1 个（`extra` 权重 1.0，ablation `extra −0.0154`）。
3. **若仍想用 CE，唯一可行的接法是「只补位、不整体重排」，且只对极少数候选打分**：前项目的补位版（保留前 3 + 余弦补位）是「strict +12 / inject +1」，而整体重排是「+1 / −26」。补位只对 `budget - keep_n` 个尾部候选打分（我方 20 个候选里补位名额更少），**打分对数是可裁剪的旋钮**。但他们补位用的是**余弦不是 CE**——我方若要上 CE 补位，必须自己量「补位名额 × 单对延迟」是否 < 200 ms，例如 4 个名额 × 40 ms = 160 ms，仍在边缘。
4. **`max_chars` 截断的正确用法（他们踩过）**：`plot_rerank.py` 是**两级**截断——token 级 `max_len=512`（`:126`）**加** 字符级 `max_chars=1200`（`:149`）。`_p4_ce_real_test.py` 用 `--max-chars 1200`。我方「512 token 块正文 P50 898 字」意味着 512 token 大约正好覆盖全文，所以「截到 128 token」是**砍掉 3/4 证据**，掉效果是必然的——前项目从未把 token 截断降到 128 用过。**若要走 CE，截断应保持 512 token / 1200 字符，靠减少候选数省延迟，而不是减少每对的 token。**
5. **别忘批处理这个免费杠杆**：生产接口 `batch=1`（`np.array([ids])`），离线脚本 `batch=16`。同一模型下后者 `seconds_per_q 18.481` 是「全量 ~1 小时可行」的依据。**若我方要上 CE，批量推理是第一个该做的工程优化**（前项目连这个都没在生产接口里实现，他们只在离线脚本里做了）。

**证据来源**：`data/knowledge/_audit/ce_timing.json`、`ce_batch_timing.json`、`ab_ce_rerank_k30.json`；`data/compendium_rag/reports/_p4_ce_subset.json`、`_p4_ce_onnx_subset.json`、`rank_rerank_lab_300cn_p30_ce.json`；`src/roleplay/core/knowledge/plot_rerank.py:65–72, 122–139, 141–157`；`scripts/_p4_ce_onnx_subset.py:54, 131`；`scripts/ab_ce_rerank.py:120–137`；`docs/compendium-rag/CHRONICLE.md:608`；`src/roleplay/core/knowledge/plot_graph.py:1840–1857, 1899–1913`。

### 7.6 附带：一条我方口径该反着抄的地方

前项目 `pick_best` 是 **hit@6 优先、MRR 仅 tie-breaker**（`_rank_rerank_lab.py:437–441`）。我方「上下文只装 4.7 块 ⇒ MRR 比 R@20 值钱」的价值判断与前项目**相反**。所以：

- **不要**照抄 `key=lambda c: (-c["hit6"], -c["mrr6"], c["name"])`；
- 应改成 MRR 优先（或至少并列一等，任何一项跌即判负）；
- 并且照抄他们的**配套**：`K_MAIN` 作为「不可调的主指标边界」常量（原文注释 `# 主指标边界：hit@6（对齐 eval300，不可调）`）、`REF_UNANS_HIT6` 作为安全线参考值、退出码三态（`0=有盲配置胜出且安全底线无违反；1=安全底线被违反；2=没有任何盲配置胜出（合法结论）`）。

**证据来源**：`scripts/_rank_rerank_lab.py:102–113, 437–441` 及其 docstring 退出码段落。

---

## 8. 一句话总结

前项目在**与我方几乎相同的位置**停下来了：他们确认「gold 已进池、只是排不进」（`pooled_in_window 21 / 35`、池天花板 `0.9286`），确认「扩池/权重/六类干预全部无效」，确认「CE 要么太慢要么增益归零」，最后兑现的方案是 **k=1（名次折扣）+ 盲特征线性重排（+7.65pt hit6, p=0.00073）+ 余弦补位（+12 strict / +1 inject）**，全部落在我方延迟预算之内或之外有明确读数。我方应当抄的是**诊断顺序**（先 triage 再谈方案）和**无模型重排**这两件事；**不该抄的是 MRR 只当 tie-breaker 的选优函数**——因为我方的上下文预算比他们更小，MRR 的权重只能更高。
