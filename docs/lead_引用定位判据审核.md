# 引用定位正确率：它测的是「我们自己写的偏移对不对」，不是「引用指对了地方」

用户目标里有一条「**引用定位 ≥98%**」，现报值一直是 **1.0000**。本轮审的是这个 1.0000 是什么。

结论：**它是构造不变量。** 判据里那三个值（`evidence` / `start` / `end`）是同一条断言记录里一起取出来、一起写进引用对象的，所以只要断言表本身自洽，判据恒真。它证明的是「我们发出去的引用与它自己声明的偏移一致」，**不证明引用指向的内容与问题有关**。

同时必须说清：这个门**确实能红**（换一条文本就假），不是写坏的。问题不在门的实现，在它测的对象。

---

## 0. 为什么写这一节

1.0000 是所有指标里最好看的一个数，最容易被当成「这块已经稳了、不用再管」。而实测 **83.9% 的引用指向的块不是本题 required 的块** —— 它们每一条都「定位正确」，但与问题无关。引用定位这条指标对这类问题一条都抓不到。

## 1. 指标实现（`scripts/ask.py:261-270` 逐字）

```python
        # 引用定位正确率：逐条看 text[start:end] 是否逐字等于 evidence
        for cite in ans.get("citations") or []:
            cite_total += 1
            rec = loaded["chunks"].get(str(cite.get("chunk_id") or ""))
            text = str((rec or {}).get("text") or "")
            start, end = int(cite.get("start", -1)), int(cite.get("end", -1))
            if rec is not None and 0 <= start < end <= len(text) and text[start:end] == cite["evidence"]:
                cite_ok += 1
            else:
                cite_verify_failures += 1
```

判据 = `text[start:end] == evidence`。

## 2. 恒真检验：三个值来自同一条记录

实测（dev 350 应答题，仓库默认开关，2,014 条引用）：

| 项 | 值 |
|---|---|
| 引用条数 | 2,014 |
| 定位判据通过 | **2,014** |
| 定位判据失败 | 0 |
| `(evidence, start, end)` 能在该块断言表里找到**原样**的记录 | **2,014 / 2,014** |
| 溯源命中率 | **1.000000** |

链路：`src/arkrag/retrieve.py::_push_span` 把断言记录的 `evidence_text` / `evidence_start` / `evidence_end` 原样写进 citation → `src/arkrag/answer.py::verify_citations` 再按同样三个值核一遍 → `scripts/ask.py` 第三次按同样三个值核。

**同一条记录的三个字段，被核了三遍。** 只要 `evidence_start/end` 与 `evidence_text` 在断言表里是自洽的（它们本来就是同一次切分产生的），这三遍必然全过。⇒ **1.0000 是构造上的必然，不是质量上的成就。**

## 3. 判别力检验：门是真的能红

把 evidence 换成**同一块里的另一条**断言文本（`start` / `end` 一字不动），判据必须变假：

| 项 | 值 |
|---|---|
| 可换文本样本 | 1,997 |
| 换后判假 | **1,997** |
| 换后仍判真（应当为 0） | **0** |

⇒ 门没有写坏：偏移真的决定判据结果。**能红，只是它红的时候对应的是「偏移与文本不一致」，而不是「引用与问题无关」。**

## 4. 被 1.0000 掩盖的风险：83.9% 的引用指向非 required 块

| 项 | 条数 | 占比 |
|---|---|---|
| 指向本题 required 块 | 324 | 0.1609 |
| **指向非 required 块** | **1,690** | **0.8391** |

按类（dev 350 应答题）：

| 题型 | 题数 | 引用条数 | 指向非 required |
|---|---|---|---|
| `multihop` | 57 | 376 | **361** |
| `direct_relation` | 57 | 372 | **333** |
| `alias` | 57 | 391 | 307 |
| `single_chunk` | 76 | 320 | 243 |
| `time_conflict` | 38 | 265 | 221 |
| `adversarial` | 19 | 91 | **91（全部）** |
| `cross_chunk` | 38 | 127 | 85 |
| `negative_abstain`（应答题 8） | 8 | 72 | 49 |

这些引用每一条都「定位正确」：偏移精确、文本逐字一致、`verify_citations` 全过。问题是**它们支撑的不是本题要的那条事实**。

⇒ 引用定位与引用覆盖测的是两件不同的事，**不能互相替代**：
- 定位指标：这条引用是否与它声明的偏移一致（考察**引用完整性**）。
- 覆盖指标：本题每条 required 是否被某条引用覆盖（考察**引用相关性**）。

P3 一直报 1.0000 定位 + 0.1971 覆盖，这两个数并列才能说明「引用都完整，但大部分不相关」。**只报 1.0000 会得出完全相反的印象。**

## 5. 对门禁的影响（四条）

1. **「引用定位 ≥98%」这一行要保留，但必须在旁边写清它测什么**：`text[start:end] == evidence`，是构造不变量。留它的价值是「偏移写坏了会立刻红」（第 3 节已证），而不是「引用相关」。
2. **不许把 1.0000 当「引用质量已达标」**。发布清单里引用质量必须同时有覆盖那一条，且覆盖那一条要带答案正确性（见 `docs/lead_引用覆盖五闸门实测.md`）。
3. **`citation_verify_failures` 归零不等于没问题**。它是第 1 节那条 else 分支的计数，与「引用是否相关」正交。
4. **新增一条可选的诊断指标「引用非 required 占比」**（当前 0.8391），只作诊断不进发布门——它会把「答案对了但引用在别处」与「引用对了但答案错」区分开。注意它依赖 `required_citations`，而后者对四类 gold（alias / cross_chunk / adversarial / negative_abstain）是块级窗口文本、可用性低，所以**不要用它做门限**，只做观察。

## 6. 限制

1. 只跑 dev 350 应答题，开关取仓库默认（`TITLE_ANSWER_PATH=False` / `CHAPTER_ANSWER_PATH=False` / `CITATION_WHOLE_LINE=False`）。**标题通路开着时引用条数会变（2,103 → 1,905 那一批），本节的 2,014 / 1,690 只能与同状态比。**
2. 「三元组可溯源」查的是**该世代自带** `data/arkrag/generations/<gen>/graph/assertions.jsonl`，不是全局块文件（遵守 `pipeline.resolve_chunks()` 的规矩）。
3. 判别力检验用的是「同一块里的另一条断言文本」；若该块只有一条断言则无样本（本轮 2,014 条里 17 条无样本）。
4. 「指向 required 块」按 `required_citations` 的 **chunk_id 集合**判，不比 evidence，所以它是**上界**（324 是「块对了」的数，不是「那条事实被引了」的数）。
5. 复算：`py -3 scripts/lead_cite_locator_audit.py --json data/lead_cite_locator_audit.json`，产物 73,876 B（含逐题明细）。

## 7. 变更日志

- 2026-11-06：首版。恒真检验 2014/2014 溯源命中、判别力 1997/1997 能红、非 required 引用占 0.8391。
