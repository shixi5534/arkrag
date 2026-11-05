# 评测快照：改动前后各一份，且必须过 `provenance` 两道判据

生成时间：2026-11-05。对应分支 `p1a-split-v2`，代码版本 `d81227a`（A 的标题通路已入库）。

## 0 为什么写这个

`scripts/ask.py --evaluate` 只把七项指标打到 stdout，除非显式给 `--json`；打出来的东西**没有代码版本、没有输入文件 sha**，换一天、换一份工作区就再也复算不出来。P0 门报告要求的七个溯源字段（`generated_at` / `generation_id` / `git_commit` / `code_sha256_16` / `dataset` / `inputs` / `environment`）不能靠人抄。

所以做了 `scripts/lead_eval_snapshot.py`：跑一次 gold set、把逐题结果和七字段一起写成一份 JSON，并且**写完当场用 `arkrag.provenance` 的两套判据自检**（七字段 `verify()`、指标行 `verify_metric_row()`），任意一条不是 `pass` 就以非零退出码失败。判据本身一行都不在这里重写，全部调用 `provenance`。

## 1 两份快照是什么

同一份 gold set（`data/goldset/gold-v1.json`，320 题 / 300 应答题）、同一个世代（`g-20261103T132152Z-06a741`）、同一份代码（`git_commit=d81227a`），**只差 `TITLE_ANSWER_PATH` 一个开关**：

| 文件 | `TITLE_ANSWER_PATH` | `code_sha256_16` |
|---|---|---|
| `data/eval_snapshot_title_off.json` | `False` | `4b7c510eecc50166` |
| `data/eval_snapshot_title_on.json` | `True` | `1b16025624efe3ac` |

两份的 `code_sha256_16` 不同而 `git_commit` 相同 —— 这是对的：`git_commit` 是 `d81227a`，但工作区里那份 `answer.py` 带 A 的未提交改动，`code_sha256_16` 才抓得住它。若两份 `code_sha256_16` 也相同，说明开关根本没生效。

## 2 两份的差别（gold-v1 320 题口径）

| 指标 | title_off | title_on | 说明 |
|---|---|---|---|
| `required_citation_coverage` | 0.183333（55/300） | **0.246667（74/300）** | +19 题 |
| `correct_coverage`（覆盖且答案含 gold） | 21 | **73** | +52 题 |
| `abstain_accuracy` | 0.85625 | **0.8875** | 274 → 284 |
| `citation_location_accuracy` | 1.0000（n=1651） | **1.0000（n=1482）** | 分母掉 169 条，非变差 |
| `citation_verify_failures` | 0 | 0 | 硬失败两侧都是 0 |
| `recall_at_20` | 0.708889 | 0.708889 | **完全相同** |
| `recall_at_5` | 0.510833 | 0.510833 | **完全相同** |
| `mrr` | 0.438846 | 0.438846 | **完全相同** |
| `direction_accuracy` | 0.666667（n=96） | 0.666667（n=96） | **完全相同** |
| 下游量 | `{citations_total: 1651, answered: 288, abstained: 32}` | `{citations_total: 1482, answered: 298, abstained: 22}` | 多答 10 题 |

**检索四项（R@20 / R@5 / MRR / 方向题）逐位相同**，这是单变量的直接证据：标题通路只改回答层，一个检索相关数字都没动。凡看到「某改动提升了回答指标、同时 R@20 也涨了」，先怀疑它不是单变量。

**引用总条数 1651 → 1482（−169）而定位通过率保持 1.0000**：标题题的引用是「引文所在整行」，每条题通常只发 1 条，而原来的三元组路径每题发好几条。**条数变少不等于质量变差**，判断要看 `covered`（55→74）与 `correct_coverage`（21→73），不许只看条数。

## 3 「快照本身也会被判 unknown」这条教训

第一次去看这份数据时，`data/eval_snapshot_title_on.json` 是**坏的**：

```
七字段 pass | 指标行 unknown | denom {'mixed': 300}
```

它缺 `reachable` / `ungated` 两个分母，也缺 `correct_coverage` 字段 —— 于是 `provenance.verify_metric_row()` 判 `unknown`。同一目录下 title_off 那份是 `pass / pass / {mixed: 300, reachable: 194, ungated: 75}`。

**判据没错，错在产物**：那份文件是脚本早期版本写的（当时还没有三分母与 `correct_coverage`）。用现在的脚本重跑一次：

```
py -3 scripts/lead_eval_snapshot.py --title-path on --out data/eval_snapshot_title_on.json
→ 七字段 pass；指标行 pass ；墙钟 32.316s / CPU 31.703s
→ denom {mixed: 300, reachable: 194, ungated: 75}，correct_coverage 73
```

两件事要记住：

1. **`unknown` 不是「大概是好的」**。这份文件当时如果被拿去当验收证据，结论会是「覆盖率从 0.18 涨到 0.2467」，而那行里的分母是残的、正确覆盖率根本没算 —— 数字看着没问题，判据说它不可比。这正是三态门存在的理由。
2. **报告和被测代码一样要能溯源**。同一份脚本在不同版本下写出的报告字段可以不一样，所以「报告缺字段」往往不是数据坏了，而是**生成它的程序已经变了**。要么把生成程序一起入库（本脚本已入库），要么这份报告作废。

## 4 复算方法

```
py -3 scripts/lead_eval_snapshot.py --title-path off --out data/eval_snapshot_title_off.json
py -3 scripts/lead_eval_snapshot.py --title-path on  --out data/eval_snapshot_title_on.json
```

脚本四参数：`--gold-set`（默认 `data/goldset/gold-v1.json`）、`--limit`（默认 320）、`--title-path {on,off}`（默认 off）、`--out`（必填）。退出码：`0` 通过并写出、`2` 输入不可用、`4` 自检不过（七字段或指标行不是 `pass`）。耗时约 32s（CPU 31.7s，本机无休眠污染）。

## 5 限制

1. 用的是 `gold-v1.json` 320 题，**不是** `gold-v2` 的 dev 380 调参集，也不是 new-holdout 160 验收集。这两份快照只能当「同分布对照」，不能当 P3 验收。
2. `correct_coverage` 与 `answer_hits_gold` 都是子串判据，未展开 `acceptable_answers`，绝对值偏低；**差值**可信，绝对值不可外报。
3. `citation_location_accuracy` 的分母是引用条数（1651 / 1482），两份分母不同，**不许直接比通过率**，要比就比「失败条数都是 0」。
4. 两份快照都是同一台机器同一时刻跑的，不构成性能数据（32s 是 320 题全量，不是单请求延迟）。
5. 工作区带 A 的未提交 `answer.py`，所以 `git_commit` 相同而 `code_sha256_16` 不同；等 A 提交后这两份快照的 `code_sha256_16` 会与提交后重建的不一致，届时需重跑，不许复用。

## 6 变更日志

- 2026-11-05 初版：入库 `scripts/lead_eval_snapshot.py` 与两份快照（`d81227a` 口径），记录「快照被判 unknown」的教训与两份指标的逐项对照。
