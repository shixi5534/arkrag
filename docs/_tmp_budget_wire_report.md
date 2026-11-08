# 上下文预算 max_context_tokens 6000 -> 12000：接线、门、反向实测

日期：2026-11-07。口径写法统一为 `<分子>/<分母> [口径名]`。

## 0. 一句话结论

`retrieve.DEFAULT_BUDGET["max_context_tokens"]` 已从 6000 改成 12000，服务端与
`scripts/ask.py` 都改成从这一处派生（不再各抄一个数），新增
`tests/test_context_budget.py` 共 12 条用例把这件事钉住（11 条断言 + 1 条
`xfail(strict=True)` 记已知缺陷）。**反向实测三笔全部按要求变红**，六文件门
`378 passed, 1 skipped, 1 xfailed`。全量套件有 1 条红，**是这次抬预算造成的
真实后果**，见 §7——需要你裁决，因为它落在我的禁改清单外。

## 1. 改动文件清单

| 文件 | 改了什么 |
| --- | --- |
| `src/arkrag/retrieve.py` | `DEFAULT_BUDGET["max_context_tokens"]` 6000 -> 12000，并在其上写 21 行注释：为什么是 12000 而不是更大、为什么不动 `MAX_CLAIMS`、dev/holdout/clean-holdout 三组双报数字（含 holdout 94 题与 gold-v1 重复的说明）、延迟代价、回滚方法。另外把两处 docstring 里手抄的 "6000 token" 改成"不抄第二份，数字只在 `DEFAULT_BUDGET` 那一处"。 |
| `src/arkrag/server.py` | 新增 `from . import retrieve as retrieve_mod`；新增常量 `DEFAULT_MAX_CONTEXT_TOKENS = retrieve_mod.DEFAULT_BUDGET["max_context_tokens"]`（带注释说明这是「同一判定写两处」事故形态）；`Settings.max_context_tokens: int = 6000` -> `= DEFAULT_MAX_CONTEXT_TOKENS`；`from_env()` 里 `_int("ARKRAG_SERVE_MAX_CONTEXT_TOKENS", 6000)` -> `_int(..., DEFAULT_MAX_CONTEXT_TOKENS)`。**文件里再没有第二个手抄的数字。** |
| `scripts/ask.py` | `evaluate()` 新增 `token_budget: int | None = None` 并在 `build_pipeline` 之后应用；CLI 新增 `--token-budget`（默认 `retrieve_mod.DEFAULT_BUDGET["max_context_tokens"]`，help 里写明 40 多个历史脚本吃默认值、台账 6000 档只有 `--token-budget 6000` 能复现）；`--evaluate` 分支把它传给 `evaluate()`；单问分支在 `build_pipeline` 之后覆盖 `pipeline["retriever"].budget["max_context_tokens"]`。共 2 处覆盖赋值（两条通路各一处）。 |
| `tests/test_context_budget.py`（新） | 12 条用例，见 §2。 |
| `tests/test_server.py` | :717 `assert second["limits"]["max_context_tokens"] == 6000` -> `== 12000`（带失败文案）。这条**本来就该变红**——它红了说明默认值真的变了。 |
| `tests/test_suite_hygiene.py` | `N = 1880` -> `N = 1896`，上方按既有格式追加 1880 -> 1896（+16）的注释块：算术、`--collect-only` 实测两数、这次上调换来哪些门。 |

未碰：`answer.py`、`pipeline.py`、`vector_union.py`、`index.py`、`scripts/diag_*.py`、`data/goldset/**`。
`MAX_CLAIMS` 一个字没动（5 -> 12 是 dev +12 而 holdout 只有 +3，判过拟合）。

## 2. 门钉住什么（12 条，每条都能红）

`tests/test_context_budget.py`：

| 类 | 用例 | 钉什么 |
| --- | --- | --- |
| `TestDefaultValueIsPinned` | `test_default_budget_is_12000` | 默认值就是 12000。改回 6000 立刻红，失败文案直接打印三组实测数字。 |
| | `test_default_is_an_int_and_not_a_downgrade` | 不是 bool、不是 float、不小于历史口径 6000（防「悄悄缩水」伪装成改预算）。 |
| `TestDefaultTakesEffectInBuildContext` | `test_default_budget_fits_more_than_five_units` | 默认值真的作用到 `build_context`：24 块 x 600 字语料上装 11 块（>5）。 |
| | `test_explicit_6000_reproduces_the_historical_caliber` | 显式 `token_budget=6000` 仍是「只装 5 块」，且 12000 装得更多——历史口径可复现。 |
| | `test_instance_budget_is_the_entry_ask_py_uses` | `Retriever.budget` 是唯一生效入口：改它 `build_context` 跟着变（否则 `--token-budget` 是摆设）。 |
| | `test_truncation_is_reported_not_silent` | 超预算的块必须进 `truncated_chunks`，且 `units + truncated == candidates`、各 unit 的 `tokens` 之和 == `stats["tokens_used"]`。 |
| `TestKnownDefect` | `test_known_bug_bigger_budget_drops_a_chunk_that_used_to_fit` | `xfail(strict=True)`：`build_context` 超预算时是 `continue` 不是 `break`，于是「加大预算反而丢掉小预算里装得下的块」。真语料 237 题里 4 题出现（q-tc-000016 / q-dr-000041）。 |
| `TestServerDoesNotCopyTheNumber` | `test_server_default_matches_retrieve_default` | `server.DEFAULT_MAX_CONTEXT_TOKENS` / `Settings()` / `Settings.from_env()` 三者都等于 `retrieve.DEFAULT_BUDGET`。 |
| | `test_server_source_has_no_hardcoded_budget_literal` | 静态门：`server.py` 里不许再出现 `max_context_tokens = 数字` 或 `ARKRAG_SERVE_MAX_CONTEXT_TOKENS", 数字`。 |
| `TestCliCanReproduceTheOldCaliber` | `test_token_budget_flag_defaults_to_the_derived_value` | `--token-budget` 存在，且默认值源码段引用 `DEFAULT_BUDGET`（不是写死）。 |
| | `test_cli_default_equals_retrieve_default_at_runtime` | 运行时从参数表读出 `--token-budget` 的 default，必须等于当前 `DEFAULT_BUDGET`。 |
| | `test_evaluate_accepts_and_applies_the_budget` | `evaluate()` 有 `token_budget`（默认 `None`），且全文件覆盖预算的赋值正好 2 处（少一处意味着有一条通路是死的）。 |

为什么用合成块而不用真语料：真语料要一整个世代、跑得慢、数字随语料漂。`estimate_tokens`
是词法口径（约 1.7 token/字），600 字块单价约 1018 token，于是 6000 装 5 块、
12000 装 11 块，正好把两档分开。实测读数（24 块 x 600 字）：

```
unit_cost=1018  default(12000)=11 units/11198 tok   token_budget=6000=5 units/5090 tok
```

## 3. 反向实测读数（原始，逐条）

### 3.1 预算改回 6000 -> 新门变红

`src/arkrag/retrieve.py` 的 `"max_context_tokens": 12000,` 改回 `6000` 后跑
`py -3 -m pytest tests/test_context_budget.py -q`：

```
3 failed, 8 passed, 1 xfailed, 1 warning in 0.91s
FAILED tests/test_context_budget.py::TestDefaultValueIsPinned::test_default_budget_is_12000
FAILED tests/test_context_budget.py::TestDefaultTakesEffectInBuildContext::test_default_budget_fits_more_than_five_units
FAILED tests/test_context_budget.py::TestDefaultTakesEffectInBuildContext::test_instance_budget_is_the_entry_ask_py_uses
```

### 3.2 恢复 12000 -> 全部通过

```
11 passed, 1 xfailed, 1 warning in 0.88s
```

### 3.3 卫生门 N 故意写错成旧值 1880 -> 报错原文

`tests/test_suite_hygiene.py` 的 `N = 1896` 改成 `1880` 后跑
`pytest tests/test_suite_hygiene.py::TestSuiteSizeIsMonotonic -q`：

```
>       assert actual == self.N, (
            f"测试总数变了：登记 {self.N}，实际 {actual}。"
            f"减少说明有门被删掉了（危险）；增加请更新本文件的 N。"
        )
E       AssertionError: 测试总数变了：登记 1880，实际 1896。减少说明有门被删掉了（危险）；增加请更新本文件的 N。
E       assert 1896 == 1880
E        +  where 1880 = <test_suite_hygiene.TestSuiteSizeIsMonotonic object at 0x...>.N

tests\test_suite_hygiene.py:1015: AssertionError
1 failed, 1 warning in 5.72s
```

### 3.4 逐条突变实测（证明「每条都真的能失败」）

每条只改一处、跑完立刻还原（`finally`）。命令：`py -3 tmp_b_mutation_sweep*.py`。

| 突变 | 改了什么 | 结果 | 变红的用例 |
| --- | --- | --- | --- |
| M1 | 预算降到 3000（低于历史口径） | 4 failed, 7 passed, 1 xfailed | `test_default_budget_is_12000`、`test_default_is_an_int_and_not_a_downgrade`、`test_default_budget_fits_more_than_five_units`、`test_instance_budget_is_the_entry_ask_py_uses` |
| M2 | `Settings.max_context_tokens` 手抄 6000 | 1 failed, 10 passed, 1 xfailed | `test_server_default_matches_retrieve_default` |
| M3 | `from_env()` 手抄 6000 | 2 failed, 9 passed, 1 xfailed | `test_server_default_matches_retrieve_default`、`test_server_source_has_no_hardcoded_budget_literal` |
| M4 | 删掉 `--token-budget` 的 `add_argument` | 2 failed, 9 passed, 1 xfailed | `test_token_budget_flag_defaults_to_the_derived_value`、`test_cli_default_equals_retrieve_default_at_runtime` |
| M5 | `evaluate(token_budget: int = 12000)` 默认值写死 | 1 failed, 10 passed, 1 xfailed | `test_evaluate_accepts_and_applies_the_budget` |
| M6 | `evaluate()` 不再应用预算（只剩单问那处赋值） | 1 failed, 10 passed, 1 xfailed | `test_evaluate_accepts_and_applies_the_budget` |
| M7 | `build_context` 不读 `self.budget`（改读常量） | 1 failed, 10 passed, 1 xfailed | `test_instance_budget_is_the_entry_ask_py_uses` |
| M8 | 超预算的块不入 `truncated_chunks` | 1 failed, 10 passed, 1 xfailed | `test_truncation_is_reported_not_silent` |
| M9 | `build_context` 忽略 `token_budget` 实参 | 2 failed, 9 passed, 1 xfailed | `test_explicit_6000_reproduces_the_historical_caliber`、`test_truncation_is_reported_not_silent` |
| X | 模拟「修好」非单调缺陷（单价升序挑选 + 原优先级输出） | 1 failed（`XPASS(strict)`） | `test_known_bug_bigger_budget_drops_a_chunk_that_used_to_fit` |

X 那笔的原始短摘要（`-rX`）：

```
[XPASS(strict)] 已知缺陷：build_context 超预算时用 continue 而非 break，预算变大反而可能丢掉小预算里装得下的块
1 failed, 11 deselected, 1 warning in 0.21s
```

即：`xfail(strict=True)` 是双向的——缺陷被修好时会 XPASS 变红，逼人更新这条记录，
不会永远安静地 xfail 下去。（第一次尝试用「不再丢弃超预算块」来模拟修复，结果仍是
`1 xfailed`：那样小预算也装下三块，小预算那条断言先红，只是换了理由 xfail。要 XPASS
必须两条断言同时成立，所以改用真正的单调化修法。）

### 3.5 CLI 端到端（真生产世代，非合成语料）

`py -3 scripts/ask.py "凯尔希和罗德岛是什么关系"`（吃默认值）：

```
摘要：单元 9  证据锚点 34  tokens 11695/12000  直接证据块 9  纯图扩展 0
```

`py -3 scripts/ask.py "凯尔希和罗德岛是什么关系" --token-budget 6000`：

```
摘要：单元 5  证据锚点 16  tokens 5681/6000  直接证据块 4  纯图扩展 1
```

与台账记录的「5.0 单位/4038 字 -> 9.8 单位/8312 字」同形，且 `--token-budget 6000`
确实把口径拨回旧档。`--token-budget` 出现在 `--help` 的 usage 与 options 两处。

## 4. 双报数字（这次改动的依据，写进代码注释与失败文案）

| 口径 | 6000 档 | 12000 档 |
| --- | --- | --- |
| dev 可判 285 题 | 191/285 (67.02%) | 225/285 (78.95%) |
| holdout 可判 195 题 | 137/195 (70.26%) | 149/195 (76.41%) |
| clean holdout 可判 120 题 | 87/120 (72.50%) | 97/120 (80.83%) |

holdout 238 题里有 **94 题与 gold-v1 重复**，clean holdout 只有 144 题、其中可判
120 题，所以 holdout 整体偏乐观，**两个数字必须一起看**。

为什么停在 12000 而不是更大：18000/24000/40000 档通过数是 236/240/246/251，边际递减；
更要紧的是 cross_chunk 题（判据是 gold 章节**集合相等**）在 40000 档开始「多列章节」
虚增——集合相等判据下 40000 只有 232/285，12000 是 225/285，只多 7 题，而 6000 -> 12000
是 +34。增益从 +34 掉到 +7，说明大预算已从「排得准」转成「堆量」。

延迟：上下文从 5.0 单位/4038 字涨到 9.8 单位/8312 字，检索+装配 P50 从 149 ms 到
148 ms——贵在装配次数，不在 token 数。

不改 `MAX_CLAIMS`：5 -> 12 是 dev +12 但 holdout 只有 +3，判为过拟合。

## 5. 最终 pytest 结果

```
py -3 -m pytest tests/test_context_budget.py tests/test_retrieve.py tests/test_server.py tests/test_suite_hygiene.py tests/test_vector_union.py tests/test_answer_paths.py -q
378 passed, 1 skipped, 1 xfailed, 1 warning in 79.92s (0:01:19)
```

卫生门单跑：`233 passed, 1 skipped`。收集数：`1896 tests collected`（旧值 1880，
+12 新文件 +4 本文件四个按 `TEST_FILES` 参数化的门各多一项）。

全量：`1 failed, 1893 passed, 1 skipped, 1 xfailed in 673.43s`——那 1 条红不是上面六个
文件里的，见 §7。

## 6. 回滚怎么做

1. 把 `src/arkrag/retrieve.py` 里 `"max_context_tokens": 12000,` 改回 `6000,`。
   这一行是全站唯一一处：`server.DEFAULT_MAX_CONTEXT_TOKENS` 与
   `scripts/ask.py --token-budget` 的默认值都从它派生，改这一行即全站生效。
2. `tests/test_context_budget.py` 里把 `DEFAULT_N = 12000` 改成 `6000`，
   `LEGACY_N` 改成对应的旧档；并把 `test_explicit_6000_...` 换成新档对照。
   **不要为了让门变绿而改这些常量**——它们红着就是在说「默认值变了，去重跑题」。
3. `tests/test_server.py:717` 的 `== 12000` 改回 `== 6000`。
4. `tests/test_suite_hygiene.py` 的 `N` 按实测重算（若同时删了用例），并在注释块
   追加一条「1896 -> X」说明为什么回退。
5. 重新刷 `data/p1c_needs_extra_citation.json`（见 §7），否则那条 `--check` 门
   会一直红。

只回滚 1、不改 2-3 的话，门会红——那是门在正常工作。

## 7. 需要裁决的一件事：p1c 探针快照被这次抬预算顶翻了（在我禁改清单外）

全量套件红的唯一一条：

```
FAILED tests/test_needs_extra_citation.py::TestCheckIgnoresVolatileProvenanceFields::test_check_passes_on_the_committed_file
E  AssertionError: ... returncode=1, stdout='引用观测与当前数据不一致（重跑不带 --check 即可刷新）'
```

**归因确认（不是既有失败）**：同一命令在预算 6000 下 `rc=0`（一致），在 12000 下 `rc=1`（不一致）。

该门的口径是 16 条目标 [p1c 引用落点探针]，它比对的是**当前代码重跑出来的引用落点**
与 `data/p1c_needs_extra_citation.json` 里入库的快照。抬预算后 16/16 行的
`units` 都变了（例：q-al-000002 4 -> 9、q-al-000004 5 -> 10、q-al-000010 5 -> 9），
其中 2 行的判定被翻转：

| 题 | 变化 | 6000 档 | 12000 档 |
| --- | --- | --- | --- |
| q-al-000012 (dev) | `required_line_cited` | true | false |
| q-al-000028 (dev) | `abstained` | true | false |

汇总口径 `16 条目标 [p1c]`：`required_cited_but_carrier_not_n` 1 -> 0、
`if_appended_would_cover_n` 3 -> 2、`neither_cited_n` 5 -> 6。
按划分看：dev 13 条里 `required_line_cited` 3 -> 2、`neither_cited` 5 -> 6；
holdout 3 条无变化。

**这不影响 §4 的结论**（那两套题量的是答案正确率，本题量的是「required 行有没有被
引到」这一个窄口径，且在 16 条上动 2 条），但它确实是抬预算的**副作用**，不是噪声。

处理需要两件事，都在我的禁改清单里／需要你点头：

1. `py -3 scripts/needs_extra_citation_probe.py`（不带 `--check`）刷新
   `data/p1c_needs_extra_citation.json` —— 这会改一个 `data/` 下的入库文件。
2. 顺带确认：`scripts/` 下还有 `diag_*.py`、`lead_budget_realrun.py`、
   `eval_union_budget.py`、`audit_redteam.py` 等一票脚本把 `6000` 写死成自己的
   `--budget` 默认值或 `BASE_BUDGET`。它们是**历史测量脚本**，按任务纪律我没碰；
   但台账复现命令如果要走它们，得显式传 `--budget`／`--token-budget 6000`。
   这一条建议在台账里补一句。

## 8. 复现命令

```
py -3 -m pytest tests/test_context_budget.py -q                      # 12 条（11 passed + 1 xfailed）
py -3 -m pytest tests --collect-only -q | Select-Object -Last 3       # 1896 tests collected
py -3 -m pytest tests/test_suite_hygiene.py -q                       # 233 passed, 1 skipped
py -3 scripts/ask.py "凯尔希和罗德岛是什么关系"                        # tokens 11695/12000
py -3 scripts/ask.py "凯尔希和罗德岛是什么关系" --token-budget 6000    # tokens 5681/6000
py -3 tmp_b_mutation_sweep.py                                        # M2/M3/M5/M6/M8
py -3 tmp_b_mutation_sweep2.py                                       # M4/M7
py -3 tmp_b_mutation_sweep3.py                                       # M1/M9
py -3 tmp_b_reverse_final.py                                         # 反向 A/B
```

突变脚本都在仓库根、以 `tmp_b_` 开头，每条自带 `finally` 还原；跑完会打印
「restore check」三行，全 `True` 才算干净。
