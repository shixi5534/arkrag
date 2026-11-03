# J 阶段实测：服务化、安全与可观测性

本文件记录计划 `docs/升级计划_GraphRAG架构.md` §J（J1 最小 API / J2 安全 / J3 可观测性）
的实际测量与决策依据。每条数字都可用 §7 的命令复现；不可复现的推断会被明确标注为推断。

对应台账：`docs/升级台账.md` §J 阶段。

---

## §0 口径

1. **服务与命令行必须是同一条链路。** 不是"尽量一致"，而是同一个函数对象：
   `src/arkrag/pipeline.py` 是唯一实现，`scripts/ask.py` 与 `src/arkrag/server.py` 都从它取。
   否则"必需产物清单""拒答判据""引用校验"会有两份定义——本项目已因「同一判定写两处、
   取最松那份生效」出过三次同型事故（见 `tests/test_single_source_of_truth.py`）。
2. **`/health` 与 `/ready` 必须是两个不同的答案。** 进程活着不代表数据齐。计划原文
   「不因进程启动就返回 Ready」在这里落成：`/health` 不查任何数据，`/ready` 查七项。
3. **就绪检查的每一项都必须能单独变红。** 只会在演示里返回 `true` 的探针等于没有探针。
   `tests/test_server.py::TestReadyChecks` 逐项篡改并断言变红。
4. **没测到的写 `null`，不写 `0`。** `/query` 响应里的 `token_cost` 是 `null` 并附
   `token_cost_reason`，因为本机无 LLM 端点、回答器是抽取式，token 成本这一项**没有测**，
   写成 `0` 会被读成"免费"。
5. **绝对延迟有噪声，只做同一次运行内的相对比较。** 本文件里出现跨运行的数字对比时，
   都注明是哪两次运行。

---

## §1 交付物

| 文件 | 行数 | 作用 |
|---|---|---|
| `src/arkrag/pipeline.py` | 200 | 检索链唯一实现（从 `scripts/ask.py` 纯搬移，零判定改动） |
| `src/arkrag/server.py` | 1,499 | HTTP 服务：9 个端点 + 就绪七项 + 限流 + 超时 + 审计 + 查询日志 |
| `scripts/serve.py` | 167 | 启动 CLI（`--print-limits` / `--generation` / `--host` / `--port` / 信号处理） |
| `tests/test_server.py` | 920 | 58 个用例，真起 HTTP 服务、用真实 identity 构建流程造世代 |

`scripts/ask.py` 改为从 `pipeline` 转发四个名字（`load_generation` / `build_pipeline` /
`ask_once` / `_read_jsonl`），因此 `ask_mod.load_generation = …` 这类 monkeypatch 仍然生效
——`scripts/bench_h3.py` 与 12 个扫描脚本依赖这一点。

### 1.1 端点清单（与计划 J1 逐条对应）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 存活探针，**故意不查任何数据**，响应内附"请查 /ready"提示 |
| GET | `/ready` | 就绪七项，见 §3 |
| GET | `/release/current` | 当前发布指针，附 `serving_generation_id` 与 `serving_is_current` |
| GET | `/limits` | 生效的限制摘要（**不含** admin token） |
| GET | `/chunks/{id}` | 单块；`untrusted: true` 标注剧情文本是数据不是指令 |
| GET | `/assertions/{id}` | 单条断言，走 `Retriever.assertions_by_id` |
| GET | `/entities/{id}` | 单实体 + 相关边；接受 id / 规范名 / 权威别名（§4.3） |
| POST | `/query` | 检索问答，默认与 `ask.py` 同参（`top_k=20`、`graph_depth=1`） |
| POST | `/admin/rollback` | 切指针，**未配 token 时 503 fail-closed** |
| POST | `/admin/evaluate` | 跑 Gold Set，`gold_set` 限定仓库内路径 |

---

## §2 为什么抽包而不是在 server 里重写

`load_generation` 的检查顺序本身就是一份判定：世代目录存在 → `manifest.graph_schema`
必须等于 `config.GRAPH_SCHEMA_IDENTITY` → 四个产物存在（图 / 断言 / 邻接 / 索引）→
块正文文件存在。**它刻意不做"缺断言就用边顶替"的降级**（无断言就无法机检证据偏移）。

如果服务自己再写一遍这段，那么"哪些产物是必需的"就有两个答案；一个版本加了一条检查、
另一个忘了加，症状是"命令行拒绝、服务照答"，而两边的日志都不会说这件事。

`pipeline.py` 因此是**纯搬移**：函数体一字未改，只把导入改成相对导入。搬移后的等价性
由既有测试守住——`tests/test_ask_sampling.py`、`tests/test_ask_generation_selection.py`、
`ask.py --evaluate` 的七项指标（§6）都在搬移后逐位复现。

---

## §3 就绪七项（`AppState.readiness()`）

| 检查 | 阻断 | 判据 | 为什么需要 |
|---|---|---|---|
| `current_pointer` | 是 | 发布指针可解析 | 指针坏了就没有"当前世代"这个概念 |
| `manifest_complete` | 是 | 八字段齐备且 `graph_schema == ark-rag-3` | 缺字段的 Manifest 会让下游各自猜默认值 |
| `graph_index_identity` | 是 | **直接调 `release.verify_generation()`** | 图/索引/断言/邻接四者的摘要与分层身份必须自洽 |
| `data_readable` | 是 | 五个文件存在且非空 | 文件在但 0 字节 = 读到空数据后静默答空 |
| `gates_pass` | 是 | 每条 `verdict == "PASS"`；空报告不算 PASS | 门禁报告是"这批产物过了门"的唯一证据 |
| `chunks_cover_generation` | 是 | 索引 `doc_id` 集合被 `chunks.jsonl` 覆盖 | 见 §3.1 |
| `serving_is_current` | **否** | 服务的世代 == 发布指针 | 灰度期服务 READY 未发布世代是合法形态（计划 K2） |

### 3.1 `chunks_cover_generation` 为什么必须阻断

引用校验拿到 `None` 时**不会报错**：它把引用判成"块不存在"、剔除、降置信度。
表现出来是"答案变短了"，不是"服务坏了"。D 阶段 13,798 块 vs 线上 13,781 块的分歧
正说明这个集合不是天然的：换一份块正文进来，索引里的 `doc_id` 就会有 17 个落空，
而 `/query` 会继续返回 200。

实测（`g-20261031T035231Z-4eafae`）：`索引 13781 块均在块正文中（块文件共 13786 条）`
—— 多出的 5 条是 T1 范围外，不影响覆盖。

### 3.2 门禁报告"没有顶层 status 字段"

计划原文写"门禁报告为 READY/PUBLISHED"，但 `reports/gates.json` 的实际结构是
list of `{name, stage, threshold, measured, verdict, note}`，**没有顶层状态**。
所以 `gates_pass` 只能逐条看 `verdict`。**空报告判为"没有可判条目"，不是 PASS**——
`all([]) == True` 是这种检查最容易踩的陷阱。

---

## §4 三个被实测抓出来的缺陷

### 4.1 `/ready` 首次 20 秒

初版每次请求都重算全部摘要。实测拆解：

| 项 | 耗时 |
|---|---|
| Manifest 读取 | 0.00 s |
| 门禁报告读取 | 0.00 s |
| `release.verify_generation()` | **14.07 s** |
| `_compute_chunks_cover()` | **5.96 s** |
| `readiness()` 冷 | **19.93 s** |
| `readiness()` 热（改前） | 0.00 s（因为根本没缓存，第二次也一样慢） |

修法：`_VerificationCache` + `_CoverCache` 两个缓存。改后第二次/第三次 `/ready` 实测 **0.0 s**。

代价是"缓存可能说谎"，所以两个缓存的键都必须是**产物 identity**，见 §4.2。

### 4.2 缓存键只按 Manifest → 产物被换掉后探针继续说好话

`_VerificationCache.key()` 初版是 `(gid, manifest mtime_ns, manifest size)`。实测：

把 `index/lexical.json` 复制一份、翻转第 100 字节（**size 不变**）：

| 实现 | 首次 | 篡改后第二次 |
|---|---|---|
| 只按 Manifest 做键 | 0 问题 | **0 问题**（继续说好话） |
| 按 Manifest + 五个产物 stat | 0 问题 | **2 问题** |

现键：`(gid, manifest_stat, stamp(graph), stamp(assertion), stamp(adjacency), stamp(index), stamp(gate_report))`。

只按 gid 不够的理由也写进 docstring：同一代 Manifest 会被 `advance()` / `update()` 改写
而产物没变，那时**必须**重算。两种失效方向都要挡。

### 4.3 `release.verify_generation` 没接 `UnicodeDecodeError`

`src/arkrag/release.py` 里：

```python
try:
    graph = json.loads(gen.graph_path.read_text("utf-8"))
    idx_payload_meta = json.loads(gen.index_path.read_text("utf-8")).get("meta") or {}
except (OSError, json.JSONDecodeError) as exc:
    ...
```

产物被翻字节后 `read_text("utf-8")` 抛的是 `UnicodeDecodeError`（`ValueError` 的子类，
**不是** `OSError`），于是 `/ready` 返回 500 而不是"未就绪 + 原因"。

这个函数的职责**正是**报告这份产物不合格，自己崩掉就变成探针 500。
已改为 `except (OSError, UnicodeDecodeError, json.JSONDecodeError)`，返回
`产物不可解析：…`。

### 4.4 `/entities/{id}` 只认 id

实体 id 是 `identity.stable_id()` 的摘要（`e-c7091cbe8cb6fe66` 这种）。只认 id 的接口
实际上只有机器能调，而"凯尔希有哪些关系"是最自然的一次人工查询。

现解析顺序：`by_id` → `by_name` → `graph["alias_index"]` 取 owner id 再 `by_id`，
响应里带 `resolved_by ∈ {"id","name","alias"}`。

**复用图里已有的 `alias_index`**，不另建映射：那张表由 `identity.CanonicalEntity.to_dict()`
定，已排除待审/拒绝别名；再写一份就会出现"接口认的别名和抽取阶段认的不是同一批"。

---

## §5 安全（计划 J2 九条逐条落地）

| 计划条目 | 落地 | 验证 |
|---|---|---|
| 剧情文本视为不可信数据 | `/chunks` 与 `/query` 的块载荷都带 `untrusted: true`；`/query` 顶层有 `untrusted_content` 说明 | 响应实测 |
| 防语料指令覆盖系统提示 | 本机无 LLM 端点，回答器是抽取式，此条当前无攻击面；标记已就位供接 LLM 时用 | 无（标注为待接 LLM 后复验） |
| 限制 Query / Part / 响应 / 单记录大小 | `QUERY_MAX_CHARS=512`（**复用 `query.QUERY_MAX_CHARS`，不另写数字**）、`max_body_bytes=64 KiB`、`max_response_bytes=256 KiB`、`max_record_chars=4096` | 400 / 413 / 响应降级 |
| 限制 Top-K / 图深度 / 上下文 Token | `max_top_k=50`、`max_graph_depth=2`（取 `retrieve.DEFAULT_BUDGET["max_depth"]` 同一个数）、`max_context_tokens=6000` | 400 `param_out_of_range` |
| 请求超时与并发 | `timeout_s=20.0` + `BoundedSemaphore(concurrency=4)` + `future.result(timeout=)` | 504；**并写明工作线程不会终止**（Python 无法安全抢占线程），信号量只限制滞留线程数 |
| 密钥只从安全配置读、不写日志和 Manifest | `admin_token` 只从 `ARKRAG_ADMIN_TOKEN` 读，无默认值；`Settings.describe()` 只报 `admin_enabled` 布尔 | 无 token → 503 |
| 鉴权限流、管理接口单独授权 | 管理接口要 `Bearer`；`secrets.compare_digest` 比较；滑动窗口 60 次/60 秒 | 401 / 429 + `Retry-After` |
| 错误响应不泄露本地路径 / 密钥 / 堆栈 | 出口统一 `sanitize_text()`：token→`<redacted>`、`config.ROOT` 四种写法→`<repo>`、绝对路径正则→`<path>`；兜底 `_guard` **只回异常类型名不回 `str(exc)`** | `str(config.ROOT) not in json.dumps(body)` |
| 审计回滚发布操作 | `<DATA_DIR>/audit/server-audit.jsonl`，**写不下去就 500 拒绝执行** | 回滚成功/失败两条都写 |

### 5.1 三条刻意的选择

1. **`/health` 不查数据。** 让它查数据会得到两个都叫"就绪"的探针，而其中一个会在
   数据损坏时仍然 200——那时运维会看错那个。
2. **参数越界 400，不夹紧。** 夹紧会让调用方以为自己的参数生效了，
   下一次它会把 `top_k=500` 当成正常用法。
3. **`log_message` 直接 `return`。** 默认访问日志把完整请求行写进 stderr，
   而 `/query` 的请求行只有路径——但 `/chunks/{id}` 与 `/entities/{name}` 会带上
   查询内容。J3 要求"默认不记录敏感完整内容"，这条得从源头上关掉。
4. **审计日志写不下去就拒绝执行管理操作**，而查询日志写不下去只计数、不阻断。
   代价不对称：前者是"回滚了但没人知道是谁做的"，后者是"磁盘满导致整个服务不可用"。

---

## §6 可观测性（计划 J3）

### 6.1 查询记录（`<DATA_DIR>/logs/queries.jsonl`）

默认**不写问句原文**，只写 `query_sha256[:16]` 与 `query_chars`；要写原文须显式置
`ARKRAG_SERVE_LOG_QUERY_TEXT=1`。

字段：`at` / `query_id`（形如 `q-20261103T075859Z-1dc37615`）/ `generation_id` /
`query_chars` / `query_sha256` / `top_k` / `graph_depth` / `candidates` /
`channels[{channel,score}]` / `expand_nodes` / `expand_edges` / `context_tokens` /
`latency_ms{query,search,expand,context,answer,total,wall}` / `prompt_version` /
`answerer` / `citation_verification{checked,accepted,rejected}` / `abstained` /
`abstain_reason` / `token_cost`。

对照计划 J3 的查询字段清单：候选数、各通道得分、扩展节点边数、上下文摘要
（`context_tokens` + 响应里的 `context_digest`）、Prompt 版本、延迟、Token 成本、
引用校验结果、拒答原因——全部覆盖。

### 6.2 构建记录

沿用既有 `release.Run`（`runs/<run_id>/events.jsonl` + `run.json` + `errors/`），
J 阶段没有新增机制。计划 J3 的构建字段与它的对照见台账 §J 阶段，缺的三项
（实际并发峰值、重试限流次数、错误报告路径**在 API 响应中**的回传）记为遗留。

### 6.3 一次真实查询的实测延迟（HTTP 全链路，`g-20261031T035231Z-4eafae`）

| 阶段 | ms |
|---|---|
| query（问句分析） | 0.52 |
| search（BM25 + 实体 + 说话人三路） | 39.70 |
| expand（图扩展，2 节点 / 40 边） | 2.44 |
| context（上下文装配，5 单元 / 5697 tokens） | 36.91 |
| answer（抽取式） | 0.29 |
| **total** | **79.86**（HTTP wall 80.33） |

候选 516 条。注意 `search_ms` 39.70 明显低于 H 阶段基线的 111.80（P50）：这里
`query_context` 已预建、页缓存全热、且只有一问。**不可与 bench 的分位数互相引用。**

---

## §7 复现命令

```powershell
# 全量测试（1208 collected）
py -3 -m pytest tests -q

# 只跑服务测试（58 用例，约 77 秒）
py -3 -m pytest tests\test_server.py -q

# 产物基线（11 项）
py -3 scripts\snapshot_baseline.py --verify

# Gold Set 七项指标（须显式给 --gold-set）
py -3 scripts\ask.py --evaluate --gold-set data\goldset\gold-v1.json --limit 320

# 起服务（--generation 省略则用发布指针；不猜"最新的那个"）
py -3 scripts\serve.py --print-limits
py -3 scripts\serve.py --generation g-20261031T035231Z-4eafae --port 8765
```

真实 HTTP 冒烟（手工执行过，结果见 §3.1 / §6.3）：

```powershell
# 起服务后
curl.exe -s http://127.0.0.1:8765/health
curl.exe -s http://127.0.0.1:8765/ready
curl.exe -s -X POST http://127.0.0.1:8765/query -H "Content-Type: application/json" `
  -d '{\"question\":\"凯尔希和罗德岛是什么关系？\"}'
curl.exe -s "http://127.0.0.1:8765/chunks/obt/main/level_main_07-02_end%23002"
curl.exe -s "http://127.0.0.1:8765/entities/%E5%87%AF%E5%B0%94%E5%B8%8C"
```

---

## §8 回归门

| 门 | 结果 |
|---|---|
| `pytest tests -q` | **1207 passed, 1 skipped, 1 warning in 200.05s**（1208 collected） |
| `snapshot_baseline.py --verify` | **PASS，产物 11 项全部一致** |
| `ask.py --evaluate --limit 320` | R@20 **0.7019** / R@5 **0.5108** / MRR **0.4394** / 引用定位 1.0000(1589) / 引用覆盖 **0.1867** / 拒答 **0.8500** / 方向题 0.6771 / 引用硬失败 **0** |

七项与 I 阶段提交时逐位相同——J 阶段只加了一层壳，没有碰检索或回答逻辑。
这正是它该有的样子。

卫生门 `tests/test_suite_hygiene.py` 的 `N`：1146 → **1208**（`test_server.py` 58 个用例
+ 该文件进入 `TEST_FILES` 后四个 parametrize 各多一项 = +4）。

---

## §9 遗留

1. **`/admin/evaluate` 在测试里没有跑通完整 Gold Set。** 单测只覆盖了鉴权与参数校验；
   完整 320 题会跑数分钟，超出单测预算。它的正确性依赖 `scripts/ask.py::evaluate`，
   而后者由 §8 的第三条门覆盖。
2. **超时用的是"回 504 但线程还在跑"。** Python 无法安全抢占线程。`BoundedSemaphore`
   只保证滞留线程最多 `concurrency` 个，不保证它们会停。这一点写进了
   `AppState.run_bounded` 的 docstring，不假装做到了。
3. **构建记录缺三项**（实际并发峰值、重试限流次数、错误报告路径在 API 响应中回传）。
   沿用既有 `Run`，本次未扩展。
4. **限流是按客户端地址的进程内滑动窗口**，多进程部署时每个进程各算一份。
   计划没有要求分布式限流，如实记录。
5. **`ARKRAG_SERVE_LOG_QUERY_TEXT=1` 会写问句原文进日志。** 这是有意的开关，
   但一旦打开，日志就变成含敏感内容的文件——运维需要自己决定保留策略。
6. **`/query` 的 `token_cost` 恒为 `null`**，因为本机无 LLM 端点。接 LLM 后这一项
   必须换成实测值，否则"Token 成本"这一栏永远是空的。
7. **`/ready` 的 `graph_index_identity` 冷启要 14.07 秒**（缓存未命中时）。
   这在正常运维中是"进程起来后第一次探针慢"，不是问题；但如果有编排器探针超时
   设成 10 秒并因此重启容器，就会变成循环重启。部署时需要知道这个数。
