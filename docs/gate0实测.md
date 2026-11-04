# Gate 0：编码一致性与回归入口（实测记录）

> 本文件记录 Gate 0 的全部实测数据：命令、退出码、耗时、commit。
> 与 `docs/k阶段实测.md` 一样，只写实际跑出来的数，不写"预计"。
> 阶段状态表的裁决在 `docs/升级台账.md`，由 Lead 汇总，本文件不改那张表。

## 0. 口径与环境

1. **本机编码事实**（`py -3` 实测，未设任何编码环境变量）：

   | 项 | 值 |
   |---|---|
   | `sys.stdout.encoding` | `gbk` |
   | `sys.stderr.encoding` | `gbk` |
   | `sys.getfilesystemencoding()` | `utf-8` |
   | `locale.getpreferredencoding(False)` | `cp936` |
   | `sys.flags.utf8_mode` | `0` |
   | `PYTHONIOENCODING` / `PYTHONUTF8` | 均未设 |
   | `sys.version` | `3.14.3 (tags/v3.14.3:323c59a, Feb 3 2026) [MSC v.1944 64 bit (AMD64)]` |

   关键点：**文件系统是 UTF-8，管道是 cp936**。这两者不一致正是本阶段所有事故的来源。

2. **所有实测都在不设 `PYTHONIOENCODING` 的前提下跑**，也就是线上/本机控制台的
   真实条件。凡是需要对照的地方，都显式写清是哪种环境。

3. **分支**：`gate0-encoding-fix`，从 `d39fbbf`（台账：先不发布，K 阶段到此为止）拉出。
   一个实验一个提交，提交清单见 §11。

## 1. 根因：一条正则、一个缺失的 env

`scripts/k2_gray.py::ServerProc.start()` 起子进程时没有传 `env`，而父进程用
`encoding="utf-8"` 解码它的输出。Windows 子进程 stdout 默认 cp936，于是：

```
子进程 print("[i] 监听 http://127.0.0.1:65439")
  → 写出 GBK 字节 5b695d20 bce0ccfd 20687474 703a2f2f ...
  → 父进程按 UTF-8 解码 → "[i] \ufffd\ufffd\ufffd\ufffd http://127.0.0.1:65439"
  → re.search(r"监听 http://[^:\s]+:(\d+)", line) 永不命中
  → 每个用例干等满 BOOT_TIMEOUT_S = 90 秒才抛 ServeError
```

报错原文是「世代 g-test-candidate 在 90 秒内**没打出监听地址**」——**脚本其实
打印了，端口也确实分配了**（`http://127.0.0.1:65439` 就在报错里）。门红了，
但红的方向是假的。全量回归里 `4 failed / 8 errors` 全部来自这一行。

### 1.1 最小复现（`repr` / `unicode_escape` 打印，不直接 print 中文）

子进程固定打印 `[i] 监听 http://127.0.0.1:12345`，父进程按 UTF-8 解码：

| 环境 | 子进程原始字节 | UTF-8 可解码 | 父进程解出的文本 | `\ufffd` 数 | 含「监听」 | 正则命中 |
|---|---|---|---|---|---|---|
| 不传 env | `5b695d20bce0ccfd20687474703a2f2f...` | false | `[i] \ufffd\ufffd\ufffd\ufffd http://127.0.0.1:12345\r\n` | 4 | false | **false** |
| 传 `PYTHONIOENCODING=utf-8` | `5b695d20e79b91e590ac...` | true | `[i] \u76d1\u542c http://127.0.0.1:12345\r\n` | 0 | true | **true**（端口 12345） |

### 1.2 端到端复现

| 命令 | 退出码 | 结果 |
|---|---|---|
| 不设 env，跑单用例 `test_stub_server_assigns_port_and_reports_ready` | 1 | **1 error in 93.13s**，`ServeError: …在 90 秒内没打出监听地址：[i] \ufffd\ufffd\ufffd\ufffd http://127.0.0.1:52636`（`scripts/k2_gray.py:375`） |
| 设 `PYTHONIOENCODING=utf-8`，跑 `tests/test_k2_gray.py` | 0 | **54 passed in 12.68s** |

### 1.3 第二类危害：子进程直接崩

cp936 编不出某些字符时，不是乱码，是**子进程自己抛异常退出**：

| 子进程打印内容 | 不设 env | 设 `PYTHONIOENCODING=utf-8` |
|---|---|---|
| 纯中文「监听 中文」 | rc=0，输出 `\ufffd\ufffd\ufffd\ufffd \ufffd\ufffd\ufffd\ufffd` | rc=0，原样 |
| 含 ⚠️ / 🚀 | **rc=1**，`UnicodeEncodeError: 'gbk' codec can't encode character '\u26a0'` | rc=0，原样 |

这一类比乱码更难辨认：退出码 1 与「真实业务失败」**无法区分**。

## 2. 修法

新增 `src/arkrag/subproc.py` 作唯一事实源：

```python
PIPE_ENCODING = "utf-8"
ENV_PIPE_ENCODING = "PYTHONIOENCODING"

def child_env(base=None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    env[ENV_PIPE_ENCODING] = PIPE_ENCODING
    return env
```

三条约束写进该模块 docstring：

1. **不清空整个环境**——复制 `os.environ` 后只覆盖一项。子进程仍需 `PATH`、`ARKRAG_*`。
2. **强制覆盖而非 `setdefault`**——若外部已是 `gbk`，保留它就等于主动重建事故组合。
   `scripts/bench_h3.py` 用的是 `setdefault`，在本场景不安全。
3. **不动 `PYTHONUTF8`**——见 §3。

改动五处（判据：父进程按 UTF-8 解码 **且** 用正则/字符串匹配子进程的中文输出）：

| 文件 | 位置 | 影响 |
|---|---|---|
| `scripts/k2_gray.py` | `ServerProc.start()` | 主犯，假红直接来源 |
| `scripts/k1_checks.py` | `_run_pytest()` | 摘要行乱码，诊断不可读 |
| `scripts/k1_checks.py` | `check_old_artifacts_backed_up()` | 同上（读 `snapshot_baseline.py --verify`） |
| `scripts/k1_race_check.py` | `one()` | 挑「最后一个 `[` 开头的行」仍挑得到，只是内容乱码——最易忽略 |
| `scripts/k3_thresholds.py` | `read_unexplained()` | 不命中就 `raise GateError`，见 §4 |

父进程侧的解码**保持 UTF-8 不变**。

### 2.1 为什么只修这五处

`scripts/k2_perf.py:149` 与 `scripts/bench_h3.py:663` 早就设了
`PYTHONIOENCODING=utf-8`，说明这是**已知约定**，`k2_gray.py` 是漏网的。
本次把散落的约定收进一个模块，并加 AST 门（§5）防止再漏。

不属同族、明确不改的两处：

| 位置 | 理由 |
|---|---|
| `src/arkrag/release.py::git_revision` | 只跑 `git rev-parse HEAD`，输出是 40 位十六进制 ASCII，无中文 |
| `tests/test_portable_paths.py::_git_tracked_data_files` | 只跑 `git ls-files` 取路径清单，只当集合元素比较 |
| `tests/test_suite_hygiene.py::test_declared_count_matches_actual` | 只读 `(\d+) tests? collected` 里的 ASCII 数字 |

## 3. `PYTHONUTF8=1`：判断后不加

`PYTHONUTF8=1` 会打开 Python 的 UTF-8 模式，影响范围远大于「管道编码」：

- 改变 `open()` 的默认编码（本项目所有产物读写都走 `atomic.atomic_write_bytes`，
  很多地方依赖默认编码的行为）；
- 改变文件系统路径的编解码与 `sys.argv` 的解码方式；
- 会掩盖真实存在的编码问题，而不是解决它们。

Gate 0 要解决的是**管道**这一层的读写不一致。只设 `PYTHONIOENCODING` 就够了，
范围可控、可解释。**这是判断后不加，不是遗漏。** `tests/test_subprocess_encoding.py`
里有一条 `test_child_env_leaves_pythonutf8_alone` 钉住这个决定。

## 4. `k3_thresholds.py`：脆弱耦合，不是已坏

这一处**当前实际是好的**，原因不是它做对了，而是它的子进程
`scripts/k3_attribution.py:163` 自己做了 `sys.stdout.reconfigure(encoding="utf-8")`。
那是**隐式契约**：任何一方被改动，这道门就会以错误的方向变红。

实测（命令级，2026-11-04）：

| 步骤 | 命令 | 退出码 | 输出 |
|---|---|---|---|
| 改前 | `py -3 scripts/k3_thresholds.py` | **0** | `结论：8/8 条 delta 判据成立`，`写盘：k3_threshold_gate.json（all_passed=True）` |
| 改后 | 临时注释掉 `k3_attribution.py:163` 的 reconfigure，再跑 | **2** | `[HARD ERROR] 归因脚本没有打印「未解释合计」行（退出码 0），无法确认红线，判据不成立` |
| 恢复 | 还原该行，再跑 | **0** | `all_passed=True`，8 秒 |

改后那一步正是 K2 的形状：**归因脚本退出码是 0、那行字确实打出来了，门却报
「没打印」**。该实验的改动只在工作区做过、跑完立刻还原，`git diff
scripts/k3_attribution.py` 为空，未进仓库。

定性：**脆弱耦合，不算「已坏」，不进失败数**。加 `env=` 之后这条隐式契约变成显式
保证，任一方单独改动都不会再翻转判据。

## 5. 两道结构性门

`tests/test_subprocess_encoding.py`（20 条）不只钉「当前这几处坏了」，更钉
「以后新加一处也不会坏」，所以判据写成 **AST 扫描**。

### 5.1 门 A：管道编码必带 env

凡是 `subprocess.run / Popen / call / check_output / check_call`，都要显式传
`env=subproc.child_env()`，除非命中白名单（三条，见 §2.1，每条带理由）。

- 白名单键是 **(相对路径, 函数名)**，不是行号——行号会随无关编辑漂移，漂移就红在
  与它无关的改动上，那正是「门变噪音、被人顺手放宽」的起点。
- `test_whitelist_points_at_real_calls` 断言白名单每条都指向真实调用，防死条目
  让门悄悄变宽。
- AST 取 `node.keywords` 的 kwarg 名，**不取源码行**——多行调用里 `env=` 往往
  不在第一行。

### 5.2 门 B：控制台编不出的字符必先 reconfigure

这是 K2 那个病的**另一侧**。K2 是「父进程 UTF-8 读、子进程 GBK 写」；这边是
「脚本往 GBK 控制台写 UTF-8-only 字符」，`print` 自己抛 `UnicodeEncodeError`、
进程以 1 退出。

| 实测 | 结果 |
|---|---|
| `scripts/eval_fusion.py:297` 那行含 U+2212（减号 −） | 无 `PYTHONIOENCODING` → **rc=1**，`UnicodeEncodeError: 'gbk' codec can't encode character '\u2212' in position 14` |
| 同上，加 `PYTHONIOENCODING=utf-8` | rc=0 |
| 同上，改为脚本内 reconfigure stdout | rc=0（且外部 poison 不再有效） |
| 同样的 print 在 pytest 默认捕获下跑 | **不崩**（写进内存缓冲，不碰控制台编码） |
| 同上但加 `-s`（不捕获，直写控制台） | **rc=1** |

最后两行是关键：**这类崩溃只在「脚本被当命令行工具直接跑」时发生，跑测时被
pytest 捕获掩盖**。所以 AST 门是唯一能提前发现它的手段。

全仓扫描结果：`scripts/*.py` 里「无 reconfigure 且 print 含 cp936 编不出字符」的
**只有 `scripts/eval_fusion.py:297` 一处**，已修。cp936 可编码性实测：
`−`(U+2212) / `⇒`(U+21D2) / `↔`(U+2194) / `⚠`(U+26A0) 均 **FAIL**；
`δ` / `Δ` / 中文字 均 OK。

关于 `CONSOLE_CODEC`：**写死 `"cp936"`**，不用 `locale.getpreferredencoding()`。
后者会让这道门随跑测机器而变，在 UTF-8 控制台上静默变成永远绿。

### 5.3 这道门自己踩过的坑（必须记下来）

第一版用 `"reconfigure" in path.read_text()` 判断有没有保护。反向实测
**没变红**（19 passed / exit=0）——因为我给 `eval_fusion.py` 加的中文注释里
**就有「reconfigure」这个词**，注释替真代码把门撑住了。

这正是 Lead 说的「检查自己写错、红方向是假的」。改成 AST 识别真实调用后，
删掉那两行 → `2 failed, 18 passed`。并留了一条回归用例
`test_reconfigure_detector_ignores_comments`：喂「注释里提到 reconfigure 但没真
调用」的文件必须返回 False，喂真调用的返回 True。

## 6. 回归入口 `scripts/regression.py`

### 6.1 事故形状

```
py -3 -m pytest tests -q | Select-Object -Last 40
$LASTEXITCODE   # -> 0
```

`$LASTEXITCODE` 拿到的是管道**右侧** `Select-Object` 的退出码。逐行 tee 的写法
在 PowerShell 里天然会引入这么一个成功退出的中间进程，于是「1498 passed /
4 failed」的外层命令报了成功。

不是 PowerShell 的问题：`py -3 -c "import sys; sys.exit(3)" | Select-Object -Last 2;
$LASTEXITCODE` 正确给出 3。问题在**调用链的形状**——中间隔了进程，失败被掩盖。

### 6.2 设计与判据

`scripts/regression.py`（纯标准库）只做一件事：跑 pytest、逐行 tee（35 分钟的
全量能实时看）、解析摘要、**以 pytest 的退出码作为自己的退出码**。

两个「看起来绿但该判红」的例外，都用 `EXIT_ENTRY_FAILED = 90`（避开 pytest 的 1–5）：

| 情形 | 处理 |
|---|---|
| pytest 退出码 0 但摘要含 `no tests ran` | 90——空跑不是绿 |
| pytest 退出码 0 但摘要自相矛盾（有 failed/errors） | 90——意味着有人把退出码弄丢了 |
| 起不来 pytest（`OSError`） | 90 |

**摘要解析必须精确，不许裸匹配 `failed`**：`0 failed` 里也有 "failed"，裸匹配会把
「1511 passed, 0 failed」判成失败。那比漏判更糟——它会训练人忽略这道门。
所以用 `(\d+)\s+failed` 与 `(\d+)\s+error(?:s)?\b`，且要求捕获组 **> 0**。

### 6.3 两边都测

| 用例 | 断言 |
|---|---|
| `test_entry_exits_nonzero_when_pytest_fails` | 临时写 `assert 1 == 2`，入口必须 `returncode == 1`（pytest 的码原样透出） |
| `test_entry_exits_zero_when_pytest_passes` | 全绿时必须是 0 |
| `test_entry_is_nonzero_when_nothing_is_collected` | 空跑必须非零 |
| `test_zero_failed_summary_is_not_bad` | `"1511 passed, 0 failed, 0 errors, 1 skipped"` 必须判绿 |
| `test_errors_alone_are_bad` | 只有 `8 errors` 不带 "failed" 也必须算坏 |
| `test_bare_word_failed_in_a_name_does_not_count` | 用例名里带 failed 不算 |

## 7. 反向实测（命令级，每条门都留了改前/改后/恢复后）

日志在 `%TEMP%\gate0\logs\`。

| # | 门 | 改动 | 命令 | 退出码 | 结果 | 恢复后 |
|---|---|---|---|---|---|---|
| 1 | K2 中文监听行 | 删 `k2_gray.py` 的 `env=subproc.child_env(),` | `py -3 -m pytest tests/test_k2_gray.py::test_stub_server_parses_chinese_listen_line_with_poisoned_pipe_encoding -q` | **1** | **1 failed in 22.97s**，`ServeError: 世代 g-test-candidate 在 20 秒内没打出监听地址：[i] \ufffd\ufffd\ufffd\ufffd http://127.0.0.1:65439` | 56 passed, exit 0 |
| 2 | 门 A（AST） | 删 `k1_race_check.py` 的 env 行 | `py -3 -m pytest tests/test_subprocess_encoding.py -q` | **1** | **2 failed / 13 passed in 8.45s** | 15 passed, exit 0 |
| 3 | 门 A | 删 `k1_checks.py` 两处 env | 同上 | **1** | **3 failed / 12 passed in 5.89s** | — |
| 4 | 门 A | 删 `k3_thresholds.py` 一处 env | 同上 | **1** | **2 failed / 13 passed in 5.86s** | — |
| 5 | 门 B（控制台） | 删 `eval_fusion.py` 的 reconfigure 两行 | 同上 | **1** | **2 failed / 18 passed in 10.11s**：`test_scripts_never_print_chars_the_console_cannot_encode` + `test_eval_fusion_protects_its_stdout` | **20 passed in 10.57s, exit 0** |
| 6 | 回归入口·非零 | `return returncode` → `return 0` | `py -3 -m pytest tests/test_regression_entry.py::test_entry_exits_nonzero_when_pytest_fails -q` | **1** | **1 failed in 19.27s** | 通过 |
| 7 | 回归入口·判零 | `summary_looks_bad` → 裸匹配 `"failed" in str(parsed)` | `py -3 -m pytest tests/test_regression_entry.py::test_zero_failed_summary_is_not_bad -q` | **1** | **1 failed in 0.29s** | 通过 |

每次反向实测后都用备份还原；`git diff` 确认过 `scripts/k3_attribution.py` 干净。

## 8. 结果

### 8.1 K2 单文件（验收硬条件之一）

```
py -3 -m pytest tests/test_k2_gray.py -q          # 不设 PYTHONIOENCODING
→ 56 passed, 1 warning in 13.22s                   退出码 0
```

修前：该文件是 1498/4/8/1 那次全量里 **4 failed + 8 errors 的全部来源**。

### 8.2 计数

```
py -3 -m pytest tests --collect-only -q
→ 1550 tests collected
```

算术：`1511（旧 N）+ 20（test_subprocess_encoding）+ 9（test_regression_entry）
+ 4 + 4（两个新文件各让四个 parametrize 各 +1）+ 2（k2_gray 新增）= 1550` ✓

`tests/test_suite_hygiene.py` 的 `N` 已从 1511 上调到 1550，注释里按既有体例
逐笔写了算术等式与每次上调换来的门。

### 8.3 全量回归（验收硬条件之二、之三）

```
py -3 scripts/regression.py tests -q          # 不设 PYTHONIOENCODING
→ pytest 退出码 : 0
  摘要行       : 1549 passed, 1 skipped, 1 warning in 1325.52s (0:22:05)
  passed / failed / errors : 1549 / 0 / 0
  结论         : 全绿
→ 脚本退出码 : 0
→ 墙钟耗时   : 1332.09 秒
```

`1549 + 1(skipped) = 1550`，与 `--collect-only` 的 1550 一致。

三次回归对照：

| | 修前（用户实测） | 修后（本次实测） |
|---|---|---|
| 结果 | 1498 passed / **4 failed / 8 errors** / 1 skipped | **1549 passed** / 0 failed / 0 errors / 1 skipped |
| 用例总数 | 1511 | 1550（新增 39 条门） |
| 耗时 | 2109.12 秒 | 1325.52 秒 |
| 失败来源 | 全部在 `tests/test_k2_gray.py` | 无 |

耗时少了 783 秒，原因不是机器变快：那 12 条异常用例每条都在干等满 90 秒超时
（12 × 90 = 1080 秒的空转），现在它们真实通过、不再空等。新增 39 条门反而让
**总时间变短**——这本身是根因判断正确的一个旁证。

**退出码传播已验**：这次全量走的是 `scripts/regression.py`，不是
`py -3 -m pytest tests -q | Select-Object ...`。脚本退出码 0，而 pytest 也是 0。
反向情形（pytest 失败时入口非零）在 §7 第 6 条用 `assert 1 == 2` 的临时用例
实测过，返回 1。

## 9. `data/arkrag/cache` 那 34 条追加：正常追加，不是篡改

`py -3 scripts/snapshot_baseline.py --verify` 在 stderr 上打一行：

```
注意: data/arkrag/cache 新增 34 条缓存记录（正常追加，不计入失败）
```

**退出码是 0**，stdout 是「产物 11 项全部一致」。这行看着像警告，其实是设计内的
正常路径。独立复核如下（不采信「应该没事」这种判断）。

### 9.1 逐文件比对

清单 `data/releases/legacy-baseline/cache-inventory.json` 记 13801 条；
盘上 `data/arkrag/cache` 实有 13835 个文件：

| 项 | 数 |
|---|---|
| extra（盘上有、清单没有） | **34** |
| missing（清单有、盘上没有） | **0** |
| changed（字节数或 SHA-256 变了） | **0** |

`missing=0` 与 `changed=0` 是关键：**没有任何一条被冻结的记录被删或被改**。

### 9.2 这 34 条是怎么来的

| 属性 | 34 条全部相同 |
|---|---|
| 扩展名 | `.json` |
| 文件 mtime | `2026-11-03 20:38:53` |
| 记录里 `written_at` | `2026-11-03T12:38:53Z` |
| `generation` | `0` |
| `key_scheme` | `2` |
| `__cache_format__` | `"raw"` |
| `prompt_sha256` | `75835665be8edb9b147c3453827b14dbdf7fbde09e0648c4a7fbb2382c4846f1`（= 当前 `cache.prompt_sha256()`） |

最强的一条证据：**34/34 的文件名都能由 `cache.key_for(content_key=..., chunk_id=...,
scene_id=..., model=..., generation=..., schema_version=..., prompt_digest=...)`
精确重算命中**，零偏差。这个函数（`src/arkrag/cache.py:77-102`）只有
`cache.write_raw()`（`:107-149`）会调用。也就是说这些文件名不可能是手写或
其他途径产生的。

另外：34 条的 `content_key` 与清单里已有的路径**冲突 0 条**，34 条**内部重复 0 条**
——没有覆盖任何既有记录。

### 9.3 为什么两代格式并存是设计内

| | 清单里的旧记录 | 这 34 条新记录 |
|---|---|---|
| 字段 | `__cache_format__`, `normalizer`, `prompt_version`, `model`, `chunk_id`, `content_key`, `written_at`, `raw` | 上述 + `key_scheme`, `generation`, `schema_version`, `prompt_sha256` |
| 键的算法 | `sha256(f"{content_key}\|{model}\|{prompt_version}")[:24]`（legacy） | `sha256("v2\x1f…")[:24]`（`key_for`） |

新键把 schema、scene、generation 也纳入哈希，旧键没有。两代并存是键方案升级的
预期结果，不是数据损坏。

### 9.4 为什么它不该被当失败吞掉

`scripts/snapshot_baseline.py:453 _verify_cache_dir()` 的语义是分开判的：

- **冻结过的记录被改/被删 → 红**（含「同长度翻一个字节」与「改清单想掩盖」两种绕法
  ——先核 `cache_inventory_digest(rows) == rec["digest"]`，改清单本身会被抓到）；
- **多出新文件 → 只报数量，不判红**。

这个分判是必要的：`data/arkrag/cache` 的正常运行就是「新抽取记录以新键追加」。
如果按整体摘要比，每跑一次合法抽取就红一次——那条门等于要求「永不抽取」。
`tests/test_baseline_cache_inventory.py` 专门守这条区分（14 个用例）。

所以这行提示是**信息，不是警告**：它告诉我们「有 34 条新数据入库了」，而
「一条旧数据都没被动过」已由 `missing=0 / changed=0 / digest 相符` 三件事证明。

## 10. 未做的两件事与理由

1. **没改 `BOOT_TIMEOUT_S`**（仍为 `90.0`）。它是产品契约不是测试参数；用例要用
   短超时就走显式 `boot_timeout_s=20.0`。另加
   `test_boot_timeout_default_is_still_90` 钉住这个默认值。
2. **没删除、没 skip 任何一条原有异常用例**，也没放宽 `监听` 这个匹配条件。

## 11. 教训

1. **「门红了」和「门红对了」是两件事。** 这次 12 条异常用例红了 2109 秒，报的是
   「脚本没打出监听地址」——脚本其实打出来了。如果只看红/绿，会得出「超时太短
   要调大」的结论，那正好把根因埋掉。
2. **检查自己写错、红方向为假，是同一个病。** 门 B 第一版用字符串匹配
   `"reconfigure"`，被我自己写的注释撑住了，反向实测 19 passed 没变红。**反向
   实测是唯一能发现这件事的手段**——如果不做反向实测，这道门会永远绿，而且
   看起来非常合理。
3. **`0 failed` 里也有 "failed"。** 用裸字符串判断摘要，会把成功判成失败。那比漏判
   更糟，因为它训练人忽略这道门。

## 12. 提交清单（`gate0-encoding-fix`，从 `d39fbbf` 起）

| commit | 内容 | 换来的门 |
|---|---|---|
| `ae056f9` | 新增 `src/arkrag/subproc.py` | 管道编码的唯一事实源 |
| `74d20ed` | 修 `k2_gray.py` + `tests/test_k2_gray.py` 两条用例 | 中文监听行在毒化 env 下仍可解析；`BOOT_TIMEOUT_S` 钉死 90 |
| `fb744f0` | `k1_checks.py` ×2、`k1_race_check.py`、`k3_thresholds.py` 补 env | 同族四处不再依赖隐式契约 |
| `722b234` | `eval_fusion.py` reconfigure + `tests/test_subprocess_encoding.py` | 门 A（AST 管道编码）+ 门 B（控制台可编码性） |
| `9f5d736` | `scripts/regression.py` + `tests/test_regression_entry.py` | 回归入口退出码传播，失败必非零、`0 failed` 必零 |

