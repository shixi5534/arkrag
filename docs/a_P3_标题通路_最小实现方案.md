# P3 标题通路：最小实现方案（待批，未改代码）

**状态：方案待批。本轮没改任何生产代码、没改任何参数、没动 `data/goldset/`、没碰 `src/arkrag/retrieve.py`。**

口径：全部数字来自 **dev 380 里的 350 道应答题**（`split=="dev"` ∧ 非应拒 ∧ 有 `gold_chunk_ids`）。验收集 260 未碰。**dev 结论不许外推到 new-holdout 160。**

---

## §0 一句话结论 + 一处方案级修正

**标题通路可以在回答层单点实现，一行都不用碰 `retrieve.py`，也不用碰引用粒度。**

**修正上一份方案的一个判断**：我原先说「B 组 25 题需两轮（先标题通路、再放宽准入）」。**实测不成立——B 组 25 题全部能用「问句引文所在那一行」直接发引用，不需要放行 `_assertion_relevant`。**

理由：覆盖率判据是 `req.evidence in cite.evidence`（单向子串），只要引用的 `evidence` 是**引文所在整行**，而那一行恰好包含 required 的 evidence。实测 60/60 命中题**全部满足**「引文行同时含全部 required evidence」。

⇒ **第二轮「放宽准入」可以整体取消**，`_assertion_relevant` 不用动。这是本轮最重要的方案级收敛。

---

## §1 上界（在线可得口径，不用 gold）

我用**只依赖问句与上下文**的定位逻辑（不含任何 gold 信息）重跑 76 题标题题：

| 项 | 题数 |
|---|---|
| 触发模板（4 个触发词之一） | 76（100% `single_chunk`） |
| 问句含「」引文 | 76 |
| 引文在上下文命中 **≥1** 个块，且命中块 `title` **唯一**、且 `title` 非空 | **61** |
| 其中该 title 恰等于 gold 答案 | **61 / 61**（零误判） |
| 其中引文所在行可作引用（`text[start:end]==line` 且 `len≥MIN_EVIDENCE_CHARS`） | **60** |
| 且该行含全部 required evidence（能同时过覆盖判据） | **60 / 60** |

* 零命中 15 题（引文在上下文里找不到，多半是 required 块没进上下文）→ **不触发，落回原路径，行为不变**。
* 多重命中 1 题（`q-sc-000064`，2 个块含引文但 title 唯一）→ 我按「title 唯一即放行」处理，**不需要 `n_hit==1`**，这样它是第 61 题。

### 1.1 两档并列表（按 Lead 要求：干净口径在前）

| 口径 | 可救题数 | 正确覆盖（`/350`） | 净增 | pt |
|---|---|---|---|---|
| **干净（问句里不含答案，模板 B）** | **22** | 25 → **47** | **+22** | **+6.29pt（0.0714 → 0.1343）** |
| 含问句泄露（模板 A，`《T》` 里就是答案） | 39 | 25 → 86 | +39 | +11.14pt |

合计 61 = 22 干净 + 39 泄露。

**不含泄露的 22 题**：
`q-sc-000017 干员间切磋指南`、`000018 问道`、`000021 归乡`、`000028 火山制造`、`000033 向死寻生`、`000034 护身符`、`000038 毗牟题，或解答`、`000042 第一次接触`、`000046 蒙眬`、`000048 降生`、`000059 崩落`、`000061 升华之夜`、`q-sc-900003 历史的航向`、`900005 命途何在`、`900011 新秩序`、`900017 凌绝顶`、`900019 在卡兹戴尔`、`900020 降生`、`900034 “狼之主”`、`900037 瑟瑟秋华`、`900043 漫游症`、`900062 分身有术`。

### 1.2 与上一份方案三组分解的对应关系（自检）

| 上一份的分组 | 题数 | 本轮可兑现 | 干净 | 泄露 |
|---|---|---|---|---|
| A（准入已过） | 36 | 35（`q-sc-000064` 因多重命中曾落在 C） | 13 | 23 |
| B（准入被挡） | 25 | **25（全部）** | 9 | 16 |
| C（其他，零命中） | 15 | 0 | 0 | 0 |
| 合计 | 76 | **60 → 61** | 22 | 39 |

23+16 = 39、13+9 = 22，与两档表一致。**B 组从「需两轮」变成「一轮可救」，就是 §0 那个修正的来源。**

### 1.3 安全性（为什么不会误伤 274 题）

* 非标题类 274 题里「gold 恰等于上下文某块 title」的 **0 题**（已实测）。
* 更硬的：**触发条件是「4 个触发词之一 + 问句含「」引文 + 引文在上下文命中块 title 唯一」**。274 题里虽有 158 题含「」引文，但无一含这 4 个触发词中的任何一个（触发词集与 `answer_form == needs_metadata` 完全等价，已实测 76 题全 `single_chunk`）。
* 因此 R5'（274 题任何指标变化即撤）预期恒 0，但**仍要报**。

---

## §2 最小实现方案

### 2.1 改哪个函数

**只改 `src/arkrag/answer.py` 一个文件、三个位置**，`retrieve.py` / `pipeline.py` / `server.py` / `scripts/ask.py` 一行不动。

| # | 位置 | 改什么 |
|---|---|---|
| 1 | `answer.py:62` 之后（紧跟 `CITATION_WHOLE_LINE = False`） | 新增模块级开关 `TITLE_ANSWER_PATH = False` + 常量 `TITLE_TRIGGERS`（4 个触发词元组） |
| 2 | `answer.py` 模块级（建议放在 `_citation_span` 之前，`:83` 前） | 新增纯函数 `_title_answer(question, units) -> dict \| None` |
| 3 | `answer.py:307-313` 之后、`:315`（收集候选 claim）之前 | 加一段「开关为真则先试标题通路，命中即返回；未命中继续原路径」 |

**为什么插在 `:313` 之后**：那是三道拒答闸（`should_answer` / `units` 为空 / `top_score < 0.05`）全部通过之后、claim 机制启动之前。放这里有两个含义：

* 合法拒答（问句不可答、零上下文、检索分数不足）**不受影响**，标题通路不会把该拒的翻成作答；
* 命中时**完全绕过 claim 生成、`MAX_CLAIMS` 截断、`_claim_supported_by_line`、`_claims_are_grounded`**——这正是「只改答案形态、不动 claim 选择」的单变量保证。
* 现状 13 道 `no_supported_claim` 拒答的标题题会在这一层被救（因为它们在 `:418` 之前就返回了），这是**预期行为**，必须在报告里单列题号。

### 2.2 `TITLE_ANSWER_PATH` 放哪、怎么写

```python
# ---- 标题型答案通路 --------------------------------------------------
#
# 抽取式回答器只会拼「主体 谓词 客体」三元组，遇到「这句话出自哪个块？请给出该块的
# 标题」这类问法永远答不出——答案是一个 chunk 的 title，既不在块正文里、也不在断言里。
# 实测（dev 350 应答题）这类题 76 道，100% 是 single_chunk，其中 61 道能用「问句里的
# 「」引文」唯一定位到上下文中的某个块，而其 title 恰是 gold 答案（61/61 零误判）。
#
# 开关默认 False：这是**新增一种答案形态**，不是改判据。关掉即与改动前逐位一致。
#
# 硬约束（不许违反）：
#   1. 答案文本**只能**取自 `unit["title"]`。question 参数仅用于提取「」引文去
#      **定位块**，绝不许从中切片/正则出答案字符。
#   2. 引用 evidence 取「引文所在那一行」，并保证 `text[start:end] == evidence`，
#      否则 `verify_citations` 会判 `evidence_does_not_match_offsets`。
#   3. 命中块的 title 不唯一（>1 种）或为空 ⇒ 不触发，落回原路径。
#   4. 不动 CITATION_WHOLE_LINE、不动 MAX_CLAIMS、不动 min_top_score、不动 retrieve。
TITLE_ANSWER_PATH = False

# 触发词与 `answer_form == needs_metadata` 判据完全等价（实测 76/76 全 single_chunk）。
# 刻意收窄：不加「引文」「原文」等泛词，否则会把 alias 类「「X」的规范名是什么」一起吃进来
# （dev 350 里有 158 道含「」引文的非标题题，必须一道都不能触发）。
TITLE_TRIGGERS = (
    "请给出该块的标题",
    "请回答该段的标题",
    "出自哪个文本块",
    "来自哪一段剧情",
)
```

### 2.3 `_title_answer()` 的实现骨架（约 40 行）

```python
_QUOTE_RE = re.compile(r"「([^」]{4,})」")

def _title_answer(question: str, units: list[dict]) -> dict | None:
    """问句在问「出处标题」时，用「」引文定位块，返回该块的 title 与引文行引用。

    只读 `unit["title"]` 作为答案；question 只用来取引文做**定位键**。
    返回 None 表示不适用（落回原有 claim 通路）。
    """
    q = str(question or "")
    if not any(t in q for t in TITLE_TRIGGERS):
        return None
    quotes = [m for m in _QUOTE_RE.findall(q) if m.strip()]
    if not quotes:
        return None

    located: list[tuple[dict, int, int, str]] = []   # (unit, start, end, line)
    for unit in units:
        title = str(unit.get("title") or "").strip()
        if not title:
            continue
        text = str(unit.get("text") or "")
        if not text:
            continue
        # 取第一个「同时含引文」的行；同一行必须能整行定位（offset 自洽）
        for line in text.splitlines():
            if not line.strip():
                continue
            if not any(qq in line for qq in quotes):
                continue
            if len(line) < MIN_EVIDENCE_CHARS:
                continue
            start = text.find(line)
            if start < 0 or text[start:start + len(line)] != line:
                continue
            located.append((unit, start, start + len(line), line))
            break

    if not located:
        return None

    titles = {str(u.get("title") or "").strip() for u, _, _, _ in located}
    if len(titles) != 1:          # title 不唯一 ⇒ 不猜
        return None
    title = titles.pop()
    unit, start, end, line = located[0]

    return {
        "answer": title,
        "unit": unit,
        "start": start,
        "end": end,
        "evidence": line,
        "n_located": len(located),
    }
```

### 2.4 `answer()` 里的接入段（约 25 行）

```python
        # ---- 标题型答案通路（新增答案形态，默认关） ----------------------
        # 放在三道拒答闸之后、claim 机制之前：合法拒答不受影响，命中时完全绕过
        # claim 生成 / MAX_CLAIMS 截断 / 逐行支撑判据 —— 这就是「只改答案形态」的保证。
        if TITLE_ANSWER_PATH:
            hit = _title_answer(question, units)
            if hit is not None:
                base["answer"] = hit["answer"]
                base["citations"] = [{
                    "claim": f"标题：{hit['answer']}",
                    "chunk_id": str(hit["unit"].get("chunk_id") or ""),
                    "evidence": hit["evidence"],
                    "start": hit["start"],
                    "end": hit["end"],
                    "assertion_id": "",
                    "predicate_in_line": False,
                }]
                base["confidence"] = round(min(0.95, float(top_score or 0.0)), 4)
                base["stats"]["title_path"] = {
                    "applied": True,
                    "located_units": hit["n_located"],
                    "title_source": "unit.title",   # 供 R7' 反向实测核查
                }
                return base
            base["stats"]["title_path"] = {"applied": False}
```

* **不写 `abstained=True`**，反向也不写——`base` 初值就是 False，直接返回即「作答」。
* `confidence` 沿用 `top_score`（与现有口径同源），不另造一个数。
* `stats["title_path"]["title_source"]` 是给 R7' 与 grep 用的**可审计标记**。

### 2.5 怎么保证「只从 `chunk["title"]` 取」

三条，缺一不可：

1. **代码层**：答案字符串的唯一来源是 `titles.pop()`，而 `titles` 只由 `unit.get("title")` 构成。`question` 变量在 `_title_answer` 里只出现在 `any(t in q ...)` 与 `_QUOTE_RE.findall(q)` 两处，**没有任何切片/替换/`partition`**。
2. **可 grep 验证**：实现后跑 `grep -n 'title' src/arkrag/answer.py`，`answer` 侧只有 `unit.get("title")` 一处取值；`grep -n 'question' ` 确认没有 `question[` / `question.split` / `question.replace`。
3. **反向实测 R7'（见 §3.3）**：把问句里 `《TITLE》` 换成无关标题重跑，答案必须跟着块 title 走。

---

## §3 判据（按 Lead 加严后的版本）

### 3.1 K 组（全满足才留）

| 编号 | 判据 | 阈值 |
|---|---|---|
| **K1'** | `correct_coverage` 净增，**按 dev 350 应答题算** | **≥ 22 题**（干净口径），同时给 `/350` pt：**+6.29pt（0.0714 → 0.1343）** |
| **K2'** | 覆盖率 `covered` 不降 | ≥ 63（预期 63 不变，A 组本来就 covered） |
| **K3'** | 逐条引用定位通过率 | **必须 = 1.0000** |
| **K4'** | 引用硬失败 | **必须 = 0** |
| **K5'** | 延迟 | 全量 dev 中位 ≤ 1.5× 基线 |
| **K6'** | 报告首行必须是**干净口径 22 题**；含泄露 39 题只能作第二行列在旁边 | 谁在前是硬要求 |

### 3.2 五数必报（不是四数）

1. 引用总条数（并报 `distinct` 去重后条数）
2. 逐条定位通过率
3. 覆盖率 `covered`
4. **正确覆盖**（引用成立 ∧ 答案含 gold）
5. **拒答数**（必须单列 —— `any_predicate` 那条路的教训）

### 3.3 反向实测 R7'（Lead 新增，必须做）

**做法**：取 5 道模板 A 的题（`q-sc-000010`、`q-sc-000017`、`q-sc-000032`、`q-sc-000035`、`q-sc-000052`），把问句里的 `《{真标题}》` 替换成 `《完全不存在的假标题》`（句式一字不改，引文不动），重跑：

* **必须**：答案仍是块的真实 `title`（跟着 `unit["title"]` 走）；
* **失败形状**：答案变成那个假标题 ⇒ 实现是在抄问句，**K1' 全部作废**。

> 结构性说明：这个替换不影响引文，所以定位到的块不变；如果实现没抄问句，答案必然不变。这是一条**零容忍**测试。

### 3.4 R 组（任一触发即撤，R 优先 K）

| 编号 | 判据 | 阈值 |
|---|---|---|
| **R1'** | `correct_coverage` 任一题回退 | **> 0 即撤**，单列题号 |
| **R2'** | 按类任一维净减 | **≥ 2 即撤** |
| **R3'** | 拒答正确率下降 | **任何下降即撤**（注意：本轮预期**拒答数 30 → 17**，13 道标题题从拒答变为作答；这 13 道**必须逐一核对是否都是应答题**，若有一道是应拒题 → 立即撤） |
| **R4'** | 引用定位任一条失败 | **即撤**（不是比例，一条就撤） |
| **R5'** | 非标题类 274 题任何指标变化 | **即撤**（预期 0，但必报） |
| **R6'** | `any_predicate` / `admit_all` 式全库放行 | **不许实现** |
| **R7'** | 反向实测（§3.3）失败 | **即撤，K1' 作废** |

**R1' 与 R5' 的优先级**：R1' 先看正确覆盖回退，R5' 看误伤面，**两条都要报**，不许只报没触发的那条（按 Lead m03828 要求）。

### 3.5 回滚

* 回滚 = `TITLE_ANSWER_PATH = False`。
* **逐位一致证明**（照 P3 第一轮手法）：旁路 `src` 副本 + `PYTHONPATH` 切导入，子进程各跑一遍全量 dev 350，比对观测向量总体 SHA-256。三轮：改动前源码 / 改动后开关 False / 改动后开关 True（连跑两次）。
* **等价性探针必须自己回显开关实际取值**（P3 第一轮踩过：子进程没传参导致三次摘要全同，看起来像「功能无效」）。

---

## §4 单变量声明（对 Lead 四条确认的逐条回应）

| Lead 的约束 | 本方案怎么守 |
|---|---|
| `CITATION_WHOLE_LINE` 已入库默认 False，本轮若动 `answer.py` 要说清是否依赖它 | **不依赖**。标题通路的引用走 `_title_answer` 自己构造的 `start/end/evidence`，**不经过 `_citation_span`**。`CITATION_WHOLE_LINE` 保持 False 不动。 |
| 不许同时改引用粒度 | **不改**。引用粒度只在 `_citation_span` 里生效，标题通路根本不调它。 |
| `src/arkrag/retrieve.py` 一字节不碰 | **不碰**。全程不动检索、不动 cooc、不动 `_assertion_relevant`（§0 已证明不需要）。 |
| 只改一个变量 | 唯一变量 = 「新增一种答案形态」。不动 `MAX_CLAIMS`、不动 claim 选择、不动预算、不动排序。 |
| 报告口径：干净在前 | K6' 已写死。 |
| 新增五数 | §3.2 已列。 |
| 引用 `distinct` 数必须报 | §3.2 第 1 项已含。本轮预期**不塌缩**（每条标题题只有 1 条引用，且 chunk_id 各不相同），但必须实测报数。 |

---

## §5 预期结果（可被事后逐项对照）

对 dev 350 应答题：

| 指标 | 基线 | 预期候选 | 备注 |
|---|---|---|---|
| `covered` | 63 | **63**（不变） | A 组 36 题本来就 covered；B 组 25 题靠引文行**可能**新增覆盖，故也可能是 88 —— 两种都接受，必须实测 |
| **`correct_coverage`** | **25** | **47（干净）/ 86（含泄露）** | 主判据 |
| `/350` | 0.0714 | **0.1343 / 0.2457** | |
| 引用总条数 | 2014 | 2014 或 +61 | 看覆盖是否新增 |
| `distinct` | 2014 | 同左 | 本轮不该塌缩 |
| 定位通过率 | 1.0000 | **1.0000** | |
| 硬失败 | 0 | **0** | |
| **拒答数** | **30** | **17** | 13 道标题题转作答，须逐一核对 |

若 `covered` 变成 88、`correct` 只涨 22 —— 说明引文行覆盖了 required 但答案形态没生效，**先怀疑自己**（查 `title_source` 标记与触发题数是否为 76）。

---

## §6 「零变化先怀疑自己」的同形坑清单

1. **触发题数必须是 76**。若跑出 0 或 <76，先查 `TITLE_TRIGGERS` 是否被写成子串不匹配（4 个触发词都是**整句片段**，不是单个词）。
2. **`unit["title"]` 是否为空**。块无 title 时必须跳过，不能返回空串（空串会让 `verify_citations` 判 `evidence_too_short` 之前就走 `no_supported_claim`）。
3. **`text.find(line)` 的正确性**：同一行文本可能在块内重复出现，`find` 取第一个会让 `text[start:end]` 定位到别处。**必须用 `text.find(line)` 后立即校验 `text[start:start+len(line)] == line`**（骨架里已写）。
4. **行长度**：`MIN_EVIDENCE_CHARS = 4`，短行会被 `verify_citations` 剔除，必须提前过滤（骨架里已写）。
5. **`answer_hits_gold` 判据**是「gold 或任一 `acceptable_answers` 出现在 `answer` 里」。块标题可能带引号（如 `“卡托加区欢迎你”`、`“狼之主”`），要贴原文核对至少 5 题，防止全半角/弯直引号差异造成假阴性。
6. **模板 A 的 `《T》` 与 `「Q」` 都在问句里**：`_QUOTE_RE` 只取「」，不会误取《》。但要确认问句里若同时有多个「」引文，定位是取「任一命中」，会不会把命中块扩大到多个（→ title 不唯一 → 不触发，行为安全）。

---

## §7 复算命令（实现后照此跑）

```powershell
# 1) 反向实测 R7'（假标题替换，5 题）
py -3 scripts/a_title_path_round.py --stage reverse
# → data/a_title_path_reverse.json   exit 0

# 2) 全量真跑（开关 False / True 各一轮，外加回滚三轮）
py -3 scripts/a_title_path_round.py --stage real
# → data/a_title_path_round.json     exit 0

# 3) 回滚逐位一致（旁路 src 副本，三轮 SHA-256）
py -3 scripts/a_title_path_round.py --stage rollback
# → data/a_title_path_rollback.json  exit 0
```

脚本骨架（尚未写）拟照 `scripts/a_p3_line_citation.py`：
* `load_dev()` 按 `split=="dev"` 全量取，**不做 `[:n]`**；
* `coverage_of()` / `citation_scan()` 与 `scripts/ask.py:283-298`、`:262-270` 逐字同口径；
* 自检：开关 False 时必须与基线逐题一致（any/covered/correct/abstained/citations），否则 exit 4；
* provenance 六件套 + `gold_set` 走 `config.rel_to_root()`。

---

## §8 边界声明

1. **本方案未实现任何代码**。上一轮的 `scripts/a_gate1b_probe.py` 是测量工具（进程内临时替换，`try/finally` 还原），不是生产改动；本轮新增的 `scripts/` 产物尚未创建。
2. 全部数字来自 dev 380 的 350 道应答题。验收集 260 未碰。**dev 结论不许外推到 new-holdout 160。**
3. `data/goldset/` 一字节没动。
4. **模板 A 的 39 题答案在问句里**，它们不是「系统读懂了块结构」，是「按引文定位到块后读出 title」。报收益必须两档并列，首行只许写干净口径 22 题。
5. 标题通路只解决 `single_chunk` 的元数据类问法，**对 multihop / adversarial / direct_relation 零作用**（实测这三类标题题为 0）。
6. §0 的「第二轮准入放宽可取消」是基于「引文行同时含全部 required evidence，60/60」这一实测。若实现后发现某题引文行不含 required evidence，则该题覆盖率不会变真（但答案仍会变对）——**答案质量与覆盖率两个维度必须分开报，不能合并成一个「救回」。**
