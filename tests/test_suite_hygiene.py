"""测试套件自身的卫生门：防止"文件在，但 pytest 一个用例都没收集到"。

## 为什么需要这个文件

`tests/test_vocab.py` 曾经**被 pytest 收集到 0 个用例**——它只有一个
`main()`，所有断言写在 `check()` 里往列表中记，由 `main()` 读。pytest 进来
扫一遍，找不到 `def test_*` 也没有 `class Test`，于是安静地跑完 0 项、
退出码 0、报告"成功"。

`tests/test_negative.py` 更隐蔽：它有 10 个 `def test_*` 函数，所以能收集到
11 项，但里面的断言只往 `FAILS` 列表记、从不抛异常，于是**全部显示通过**。
实测代价：分块器把 `ordinal` 记成块内下标（2,291 个场景里 2,127 个轮次
覆盖不全）时，`pytest` 报 11 passed，只有直接运行该文件才看到那 1 条 FAIL。

这两个都是同一类失效：**测试代码在，判定不在**。CI 绿着，缺陷躺在那里。
这个文件把"判定必须存在且必须能变红"变成机检。

## 检查项

1. 每个 `tests/test_*.py` 必须能被 pytest 收集到用例（有 `def test_` 或
   `class Test`）。
2. 不得存在"只记录、不抛异常"的断言辅助函数被 `def test_*` 使用——
   即：用了 `FAILS.append` / `FAILURES.append` 这类模块级列表的文件，
   它的 `test_*` 函数体内必须有 `assert` 或 `raise`。
3. 测试文件不得用相对 `cwd` 的仓库路径（换目录运行会得出不同结论）。
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"

TEST_FILES = sorted(p for p in TESTS.glob("test_*.py"))
HARNESS_FILES = {"conftest.py"}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _module_tree(path: Path) -> ast.Module:
    return ast.parse(_read(path), filename=str(path))


def _functions(tree: ast.Module, prefix: str) -> list[ast.FunctionDef]:
    return [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith(prefix)
    ]


class TestEveryFileIsCollected:
    """检查 1：文件存在不等于测试存在。"""

    @pytest.mark.parametrize("path", TEST_FILES, ids=lambda p: p.name)
    def test_file_has_pytest_visible_tests(self, path: Path) -> None:
        tree = _module_tree(path)
        has_func = bool(_functions(tree, "test_"))
        has_class = any(
            isinstance(node, ast.ClassDef) and node.name.startswith("Test")
            for node in tree.body
        )
        assert has_func or has_class, (
            f"{path.name} 里没有任何 pytest 能收集的用例"
            f"（既无 def test_* 也无 class Test*）。"
            f"这种文件会显示'成功'，实际上一个断言都没跑。"
            f"如果它只有一个 main()，请补一个调用 main() 并 assert 的 test_* 函数——"
            f"见 tests/test_vocab.py 的 test_vocab() 就是这么修的。"
        )

    def test_known_harness_files_are_not_mistaken_for_tests(self) -> None:
        """`conftest.py` 是夹具，被收集逻辑排除在外，不能算作漏测。"""
        assert HARNESS_FILES == {"conftest.py"}
        for name in HARNESS_FILES:
            assert (TESTS / name).exists(), f"夹具文件 {name} 不见了"


class TestAssertionsActuallyRaise:
    """检查 2：断言不能"只记录、不抛异常"。

    判据分两层，缺一层就会漏：

    - 辅助函数层：`expect()` / `check()` 这类函数体里必须有 `raise` 或 `assert`。
      旧版 `if not cond: FAILS.append(msg)` 就在这一层失效——它记了一笔然后
      继续往下跑，pytest 完全看不见。
    - 调用层：`def test_*` 的函数体里要么自己有 `assert`/`raise`，要么调用了一个
      **会抛异常的**模块级辅助函数。只调用"记账式"辅助函数的用例恒为 PASS。

    两层都要查，因为只查调用层会放过"辅助函数造假"，只查辅助函数层会放过
    "test_* 调了一个根本不判定的函数"。第一版只按模块级列表 + append 判断，
    结果把已经修好的 `test_negative.py`（`expect` 已 raise）误报为违规——
    误报会让人学会忽略这条门，所以判据必须是"能不能真的变红"，不是"看起来像不像"。
    """

    def _raising_helpers(self, tree: ast.Module, src: str) -> set[str]:
        """模块级函数里，**能真正抛异常**的那些名字。

        要做传递闭包，不能只看函数体里有没有 `raise` 字面量：
        `test_vocab()` 调 `_check_all()`，`_check_all()` 调 `check()`，
        `check()` 才 `raise`。只看一层会把这种正确结构误报成"不会失败"——
        实际上它会失败，只是隔着两层。误报会让人学会忽略这条门。
        """
        bodies: dict[str, str] = {}
        for fn in tree.body:
            if isinstance(fn, ast.FunctionDef):
                bodies[fn.name] = ast.get_source_segment(src, fn) or ""

        raising = {
            name
            for name, seg in bodies.items()
            if "raise " in seg or "assert " in seg or "pytest.raises(" in seg
        }
        changed = True
        while changed:
            changed = False
            for name, seg in bodies.items():
                if name in raising:
                    continue
                if any(f"{other}(" in seg for other in raising):
                    raising.add(name)
                    changed = True
        return raising

    @pytest.mark.parametrize("path", TEST_FILES, ids=lambda p: p.name)
    def test_every_test_function_can_fail(self, path: Path) -> None:
        src = _read(path)
        tree = _module_tree(path)
        helpers = self._raising_helpers(tree, src)
        offenders: list[str] = []
        for fn in _functions(tree, "test_"):
            seg = ast.get_source_segment(src, fn) or ""
            # 认三种「真会变红」的构造，不止 assert/raise 两个字面量：
            #   · `with pytest.raises(E): f()`：f() 不抛时 pytest 自己 FAIL；
            #   · `pytest.fail(...)`：直接红。
            # 原先只认 assert/raise，于是把
            # `tests/test_p0_item5_drill.py::test_reassemble_needs_the_kept_files`
            # 判成「恒为 PASS」——实测是假阳性（去掉保留清单校验后它立刻变红）。
            if (
                "assert " in seg
                or "raise " in seg
                or "pytest.raises(" in seg
                or "pytest.fail(" in seg
            ):
                continue
            if any(f"{name}(" in seg for name in helpers):
                continue
            offenders.append(fn.name)
        assert not offenders, (
            f"{path.name} 里这些 test_* 函数既没有 assert/raise，也没有调用会抛异常的辅助函数："
            f"{offenders}。它们在 pytest 下会恒为 PASS。"
            f"把辅助函数改成直接 raise（参考 tests/test_negative.py 的 expect），"
            f"或让 test_* 显式断言辅助函数的结果。"
        )

    @pytest.mark.parametrize("path", TEST_FILES, ids=lambda p: p.name)
    def test_assertion_helpers_raise(self, path: Path) -> None:
        """名字像断言辅助的函数必须真能抛。按名字集合判定，不靠调用点。"""
        HELPER_NAMES = {"expect", "check", "assert_", "require", "verify"}
        src = _read(path)
        tree = _module_tree(path)
        bad: list[str] = []
        for fn in tree.body:
            if not isinstance(fn, ast.FunctionDef):
                continue
            if fn.name not in HELPER_NAMES and not fn.name.startswith("assert_"):
                continue
            seg = ast.get_source_segment(src, fn) or ""
            if "raise " not in seg and "assert " not in seg:
                bad.append(f"{path.name}:{fn.name}()")
        assert not bad, (
            f"这些断言辅助函数只做记录、不抛异常，调用方在 pytest 下会恒 PASS：{bad}"
        )


class TestNoCwdRelativeRepoPaths:
    """检查 3：测试不得依赖调用者 cwd。

    实测教训：`tests/test_vocab.py` 原用 `Path("src/arkrag/vocab.py")` 读源码，
    从 `E:\\图rag` 跑没问题，从别的目录跑就 `FileNotFoundError`——
    结论取决于站在哪，这种测试不能当门。

    本文件自己排除：它的检测正则以字符串形式写在这份源码里，会自我命中。
    自指的豁免必须写明理由，不能靠"反正跑不到"。
    """

    BAD = re.compile(r"""Path\(\s*['"](src|docs|data|scripts|corpus)/""")
    SELF = Path(__file__).name

    @pytest.mark.parametrize("path", TEST_FILES, ids=lambda p: p.name)
    def test_no_repo_relative_path_literal(self, path: Path) -> None:
        if path.name == self.SELF:
            pytest.skip("本文件的正则字面量会自我命中；例外已在此声明")
        offenders = []
        for i, line in enumerate(_read(path).splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if self.BAD.search(line):
                offenders.append(f"{i}: {line.strip()}")
        assert not offenders, (
            f"{path.name} 里有相对 cwd 的仓库路径，换个目录运行就会失败：{offenders}。"
            f"请用 `Path(__file__).resolve().parents[1]` 拼绝对路径。"
        )


class TestSuiteSizeIsMonotonic:
    """检查 4：用例总数不许悄悄变少。

    `N` 是登记值。删除或重命名测试时必须同步更新它——这是有意的摩擦：
    少掉一个门要有人签字，不能让"少跑了几个测试"静默发生。
    """

    # 936 -> 948（H1）：新增 tests/test_ask_generation_selection.py 的 8 个用例，
    # 且该文件进入 `TEST_FILES` 后自身让三个 parametrize 各多一项，故 +12。
    # 这次上调不是"为了让门变绿"：上调换来的是一条能变红的用例
    # （默认世代曾静默取最旧的那个），见该文件的 docstring。
    #
    # 948 -> 1019（H2）：新增 tests/test_query_context.py（23 个）与
    # tests/test_index_norm_table.py（40 个），两个文件进入 `TEST_FILES` 后
    # 又让三个 parametrize 各多两项（+6），另加 2 个显式登记的用例。
    # 这里换来的门同样是能变红的：一个是"复用预建结构与就地重建逐位相同"，
    # 一个是"预建归一化表与旧实现分数逐位相同"——两者都是纯优化改动，
    # 唯一能证明没改坏排序的证据就是这类等价性断言。
    # 1019 -> 1056（H3）：新增 tests/test_search_backend.py（9 个）与
    # tests/test_fts_backend.py（20 个），合计 29 个用例；两个文件进入
    # `TEST_FILES` 后又让四个 parametrize（本文件 61/130/150/184 行）各多两项（+8）。
    # 1019 + 29 + 8 = 1056，与 `--collect-only` 实测一致。
    # 这里换来的门：FTS 后端与 JSON 后端 top-20 逐位相同、SQLite 库结构完备、
    # 建库只从已发布 lexical.json 翻译而不重新分词（否则两份索引会静默不同）。
    # 1056 -> 1093（I）：新增 tests/test_retrieve_speaker.py 的 33 个用例，
    # 该文件进入 `TEST_FILES` 后让四个 parametrize 各多一项（+4）。
    # 1056 + 33 + 4 = 1093，与 `--collect-only` 实测一致。
    # 这里换来的门是**两条真被它抓到的缺陷**：
    #   ① `_speaker_boost` 对同一实体的规范名+别名重复计票（`["凯尔希","凯尔希医生"]`
    #      能拿到 2 倍票）——按实体去重修掉了；
    #   ② 涨分到底来自"多实体在场"还是"别名重复"，是被这个文件里的
    #      「上界 = 命中实体数」断言与配套的 `diagnose_speaker_clamp.py`
    #      一起分辨出来的。分辨之前我曾错误地把分数夹到 1.0，指标掉了 0.0100。
    # 1093 -> 1146（I 续）：新增 tests/test_vector_store.py 的 49 个用例，
    # 该文件进入 `TEST_FILES` 后让四个 parametrize 各多一项（+4）。
    # 1093 + 49 + 4 = 1146，与 `--collect-only` 实测一致。
    # 这里换来的门是**这类模块最容易出的失效形态——不崩、只是静默错**：
    #   ① 向量文件被截断或改一个字节 → `np.frombuffer` 不抛异常，只给垃圾分数，
    #      表现为"召回莫名下降"。用例用手工截断与翻字节证明 sha256 校验真的会红。
    #   ② 块正文改了而向量没重建 → 检索照常工作但语义脱节。逐条哈希比对。
    #   ③ 块删了而向量还在 → 检索可能返回不存在的 chunk_id。
    #   ④ 同一通道输出重复项 → RRF 若按出现次数累加就是奖励上游 bug。
    #   ⑤ 通道缺席若不显式申报 → `channels_used` 少一行而没人会去数。
    # 1146 -> 1208（J）：新增 tests/test_server.py 的 58 个用例，
    # 该文件进入 `TEST_FILES` 后让四个 parametrize 各多一项（+4）。
    # 1146 + 58 + 4 = 1208，与 `--collect-only` 实测一致。
    # 这里换来的门全部是"服务能坏但测试不会叫"的形态：
    #   ① `/health` 与 `/ready` 必须是两个答案——进程活着不代表数据齐；
    #   ② 就绪检查的每一项都要能单独变红（挡住 Manifest、产物、门禁报告、块覆盖）；
    #   ③ 管理接口在没配 token 时必须 503 fail-closed，而不是"没人认证就当管理员"；
    #   ④ 参数越界必须 400 而不是夹紧——夹紧会让调用方以为自己的参数生效了；
    #   ⑤ 校验缓存必须按**产物**失效，否则产物被换掉后探针只会说好话。
    # 1208 -> 1248（K）：新增 tests/test_k1_checks.py 的 36 个用例，
    # 该文件进入 `TEST_FILES` 后让四个 parametrize 各多一项（+4）。
    # 1208 + 36 + 4 = 1248，与 `--collect-only` 实测一致。
    # 这次换来的门是"检查本身必须能失败"：`scripts/k1_checks.py` 的九项
    # 每一项都被构造了一个本该拦住它的场景，并断言它确实返回 ok=False。
    # 一个永远返回 True 的检查函数比没有更糟——它会在台账里多出九行 PASS，
    # 而没人知道那九行是怎么来的。这个文件还钉住了 K1 首轮的两处自伤错误：
    #   ① 把 16 位展示指纹 `prompt.prompt_fingerprint()` 当成 64 位
    #      `fingerprint.prompt_digest()`，两者前 16 位相同，报出的不一致像是
    #      产物坏了；
    #   ② 把 `invalid_cache_report()["total"]` 当成"坏了几条"，于是把
    #      `total=13801 / ok=13801 / invalid=0` 报成"13801 条非法记录"——
    #      一条假重大发现。
    #
    # 1248 → 1251：`tests/test_server.py` 新增 3 个用例
    # （`test_oversized_body_still_delivers_the_413`、`test_413_does_not_leak_internal_drain_directive`、
    # `test_unreasonably_large_body_closes_the_connection`），该文件已在 `TEST_FILES` 里，
    # 不再产生 parametrize 增量。1248 + 3 = 1251，与 `--collect-only` 实测一致。
    # 这次换来的门是"错误响应必须真的到达客户端"：`_route_post()` 在**任何**拒绝分支
    # 之前读掉请求体，否则未读字节让 `close()` 发 RST，客户端收到
    # `ConnectionAbortedError: [WinError 10053]` 而不是那条 413/503。
    # 这条缺陷只在全量跑时偶发，最容易被当成 flaky 放过。
    # 1251 → 1269：新增 tests/test_chunk_snapshot.py 的 14 个用例，
    # 该文件进入 `TEST_FILES` 后让四个 parametrize（本文件 61/130/150/184 行）
    # 各多一项（+4）。1251 + 14 + 4 = 1269，与 `--collect-only` 实测一致。
    # 这次换来的门是「引用偏移必须落在它自己那个块的正文上」：
    #   ① 世代必须自带块正文，或借来的全局文件必须能复现本代
    #      `chunk_manifest_hash`——否则消费者拿错块，`text[start:end]`
    #      切出另一句话，引用被静默剔除（实测 q-sc-000024，17 块正文变了
    #      + 17 块新增，Gold Set 冒出 1 条引用硬失败）；
    #   ② 图扩展登记证据偏移时，宿主块载入不回来就不许登记——
    #      退回当前单元等于把 A 块的偏移套在 B 块正文上。
    # 两条都不崩、只是让答案变短，是典型的静默失效。
    # 1269 → 1287：新增 tests/test_baseline_cache_inventory.py 的 14 个用例，
    # 该文件进入 `TEST_FILES` 后让四个 parametrize（本文件 61/130/150/184 行）
    # 各多一项（+4）。1269 + 14 + 4 = 1287，与 `--collect-only` 实测一致。
    # 这次换来的门是「冻结目录的追加与篡改必须分开判」：
    #   `data/arkrag/cache` 的正常运行就是「新抽取记录以新键追加」，
    #   而旧版 `--verify` 只比整体摘要，于是每跑一次合法抽取就红一次——
    #   这条门等于要求「永不抽取」。实测触发点是 K2：34 条新抽取入库后
    #   `count 13801->13835`，校验 FAIL。现在按逐文件清单分判：
    #   · 冻结过的记录被改/被删 → 红（含「同长度翻一个字节」与
    #     「改清单想掩盖」两种绕法）；
    #   · 多出新文件 → 只报数量，不判红。
    # 1287 → 1289：`tests/test_server.py` 的「块正文缺块阻断就绪」拆成两条
    # （`test_truncated_generation_chunks_blocks_ready`、
    # `test_chunks_cover_check_reports_which_blocks_are_missing`），并新增
    # `test_global_chunk_file_is_no_longer_what_the_consumer_reads`；
    # 该文件已在 `TEST_FILES` 里，不再产生 parametrize 增量。+2 = 1289。
    # 为什么值得拆：原用例把「身份」与「覆盖」两件事混在一句断言里，
    # `resolve_chunks()` 落地后截断先被身份校验 fail-closed 拦下，
    # 逐块比对那一步**从来没被执行过**，却一直跟着用例一起算通过。
    # 1289 → 1291：`tests/test_release.py` 新增 `TestPruneOnlyTouchesFailed`
    # 两条（`test_dry_run_lists_only_failed`、`test_real_prune_keeps_ready_and_published`）。
    # 该文件已在 `TEST_FILES` 里，不再产生 parametrize 增量。1289 + 2 = 1291，
    # 与 `--collect-only` 实测一致。
    # 这次换来的门是「退役只许清 FAILED」：`prune_old()` 旧实现把
    # `status != PUBLISHED` 当选人条件，于是 READY（可服务）也进了候选。
    # 实测 keep=2 时拟清 7 个、其中 5 个非 FAILED，含上一代
    # `g-20261031T035231Z-4eafae`——真跑一次就同时删掉上一代与多个 READY 候选，
    # 违反计划 K4 的「保留至少上一代完整有效 Generation」。
    # 原有用例（`test_prune_is_dry_run_by_default`）只在空目录里断言「演练不删东西」，
    # 空目录下任何候选条件都为空，**永远发现不了这个缺陷**；新用例造
    # PUBLISHED / READY / FAILED 三代并跑到真实 prune，才能把这条拦住。
    # 1291 → 1470：补齐 K 阶段对照计划查出的四处漏项，一共 +179。逐笔核过：
    #   · `tests/test_k2_perf.py` 38 个（性能评测门）；
    #   · `tests/test_k2_gray.py` 54 个（真 HTTP 双实例灰度）；
    #   · `tests/test_k3_thresholds.py` 61 个（阈值登记与审批记录强制校验）；
    #   · `tests/test_release.py` +14 个（观察期判定与归档顺序），该文件已在
    #     `TEST_FILES` 里，不产生 parametrize 增量；
    #   · 前三个是**新文件**，进入 `TEST_FILES` 后让四个 parametrize
    #     （本文件 61/130/150/184 行）各多三项（+12）。
    #   38 + 54 + 61 + 14 + 12 = 179，1291 + 179 = 1470，与 `--collect-only` 实测一致。
    # 这四道门换来的都是「原来根本不存在判定」：
    #   · 性能：K 阶段此前一次性能评测都没跑过（`k阶段实测.md` 里「性能」
    #     「bench」「P95」各 0 处），DoD 的「查询和构建性能有持续记录」无从谈起。
    #     门只放**确定性事实**（字节数、词条数、文档数、探针 digest、RSS 上界、
    #     冷加载打开过哪些文件），时序只申报不入 `all_passed`——实测同一批题
    #     连跑两次，候选热查 P95 从 167.671 ms 跳到 544.19 ms（3.2 倍），
    #     只有并发负载变了。把时序做成门就是让门随机红。
    #   · 灰度：K4 那个「灰度」是进程内对 18 题跑探针，不是 HTTP。现在起两个
    #     真 `serve.py` 实例逐题交替问，判据是拒答翻转、引用可定位、世代标识
    #     三处核对；实测 20 题、40 次请求全 200、翻转 0、引用 184 条全部可定位。
    #   · 审批：K3 第 5 条「超阈值必须人工审批」此前从未被执行过一次。
    #     现在阈值（8 条 delta）与审批记录（七必填字段、`approved_by` 是
    #     `auto`/`TBD` 判无效、`measured` 与实测差 >1e-9 判记录不符）由
    #     `data/k3_thresholds.json` 与 `data/k3_approvals.json` 承担。
    #   · 观察期：K4 第 4 条「观察期后才归档旧版」此前没有观察期也没有归档动作。
    #     现在 `observation_status()` fail-closed（无 `published_at`／格式非法／
    #     无指针一律按未满），`archive_old()` 七条拒绝理由逐条独立，
    #     复制 → 逐文件核 SHA-256 → 写清单 → 最后才删原件，任一步失败不删任何东西。
    # 1470 → 1471：`tests/test_k1_checks.py` 新增
    # `test_probe_dir_is_unique_per_run_and_cleaned_up` 一条。该文件已在
    # `TEST_FILES` 里，不产生 parametrize 增量。1470 + 1 = 1471，
    # 与 `--collect-only` 实测一致。
    # 这次换来的门是「回滚演练的探针目录每次运行独一份」：`check_rollback_tested()`
    # 原先把探针目录写死在 `data/arkrag/_k1_rollback_probe/`，任一进程同时跑这一项
    # 就共用同一个 `current.json`，于是 A 写在 B 的指针上、互相把对方的一步
    # 当成自己的结果。**单独跑永远绿，只有并发时才红**，而且红得不像并发问题：
    # 实测并发 4 份 `--only rollback_tested`，修正前 0/4 通过，两种错法各两次——
    # 「回滚到当前世代未被拒，previous_generation 会退化成指向自身」×2 与
    # 「release.rollback 拒绝执行：… 已经是当前世代，无需回滚」×2。
    # 这条第一次出现是真事故：2026-11-04 15:19 跑 `k1_checks.py`，另一个子代理
    # 正在跑全量 pytest（其中 `test_real_pointer_is_never_written` 也调这一项），
    # 检查报了「回滚命令已实测」红，而代码本身是对的（`release.py:772` 正确地拒绝
    # 回滚到自身）。修法 `tempfile.mkdtemp(prefix="k1_rollback_probe_")` 后
    # 并发 4/4 通过。
    # 1471 → 1506：`tests/test_k2_publish_check.py` 新增 31 个用例；该文件是新文件，
    # 进入 `TEST_FILES` 后让四个 parametrize（本文件 61/130/150/184 行）各多一项（+4）。
    # 31 + 4 = 35，1471 + 35 = 1506，与 `--collect-only` 实测一致。
    # 这次换来的门是「发布前必须证明那五道门真跑过、而且跑的是这一代」：
    # `docs/升级计划_GraphRAG架构.md:420-428` 的 K2 构建顺序末尾是
    # 「… → 候选 READY → 灰度查询 → 正式发布」，五道门各自写报告，但**没有任何一处
    # 要求发布前对账**——`release.publish()` 只查状态 READY、覆盖完整、
    # `verify_generation()`，它看不到那五份报告。也就是说一个 READY 世代可以带着
    # 五份「根本没跑」或「跑的是上一代」的报告直接上线，而没人会发现。
    # `scripts/k2_publish_check.py` 把清单变成六条可失败判据（五份在场／各自
    # all_passed／都绑同一世代／候选 READY／候选自检干净／报告不早于候选建成）。
    # 三份报告没有 `generated_at` 字段，对它们只能记 `undetermined`——
    # 用例专门钉住「无法判定不等于通过」（`test_three_reports_without_timestamp_are_
    # undetermined_not_pass`），因为把不知道写成知道，正是这道门本来要防的事。
    # 1506 → 1511：`tests/test_k2_publish_check.py` 再加 5 条，钉的是「这道清单
    # 真的接在发布路径上了」。该文件已在 `TEST_FILES` 里，不产生 parametrize 增量。
    # 1506 + 5 = 1511，与 `--collect-only` 实测一致。
    # 为什么光有清单门不够：清单脚本写好了却没人调用，它只是一份没人拦的报告。
    # 这 5 条在 tmp 沙箱里跑 `release_cli.cmd_publish()`，证明缺报告 / 报告绑上一代 /
    # 报告自己没过 三种形状都会**拒绝且不写指针**；同时留两条反向对照
    # （全齐则真发布、清单过了但覆盖不全仍被 `release.publish()` 拒），
    # 证明它们不是「凡发布必红」。反向实测：把 `cmd_publish()` 里清单那 25 行
    # 短路掉，5 条全红（`5 failed, 31 passed`），还原后 36 passed。
    # 踩过的坑写进用例注释：`release_cli` 里的 `import k2_publish_check` 与本文件
    # `spec_from_file_location` 载的是**两份模块实例**，只 patch 一处等于没 patch。
    # 1511 → 1550（Gate 0）：一次性记四笔，逐笔核过。
    #   · `tests/test_subprocess_encoding.py` 20 个（新文件，进入 `TEST_FILES`
    #     后让四个 parametrize（本文件 61/130/150/184 行）各多一项，+4）；
    #   · `tests/test_regression_entry.py` 9 个（新文件，同理 +4）；
    #   · `tests/test_k2_gray.py` +2 个，该文件已在 `TEST_FILES` 里，
    #     不产生 parametrize 增量。
    #   20 + 9 + 4 + 4 + 2 = 39，1511 + 39 = 1550，与 `--collect-only` 实测一致。
    # 这四笔换来的门都是「门会报红，但红的方向是假的」这一类——
    # 也就是本项目历史上三次事故的共同形状。它们彼此独立，各自能变红：
    #   ① `test_k2_gray.py::test_stub_server_parses_chinese_listen_line_with_
    #      poisoned_pipe_encoding`：把父进程的 `PYTHONIOENCODING` 毒化成 `gbk`
    #      （模拟 Windows 控制台默认 cp936 的真实条件），仍必须解析出中文
    #      「监听」那一行的端口。修之前这个用例就是全量回归里 4 failed / 8 errors
    #      的真身：子进程按 GBK 写出字节、父进程按 UTF-8 解出 `\ufffd\ufffd\ufffd\ufffd`，
    #      `re.search(r"监听 http://[^:\s]+:(\d+)", line)` 永远不命中，每个用例
    #      干等满 90 秒才抛 `ServeError`，报错原文还写着「没打出监听地址」——
    #      **脚本其实打印了**。它不是「慢」，它是把「读不到」报成了「没写」。
    #      （`BOOT_TIMEOUT_S` 保持 90 秒不变，用例走显式 `boot_timeout_s=20.0`。
    #      另有一条 `test_boot_timeout_default_is_still_90` 钉住这个默认值，
    #      防止有人为了「让它快一点」把产品契约改掉。）
    #   ② `test_subprocess_encoding.py` 的 AST 门
    #      （`test_subprocess_calls_declare_pipe_encoding`）：凡是
    #     `subprocess.run/Popen/call/check_output/check_call` 且父进程用 UTF-8
    #      解码子进程输出的地方，必须显式传 `env=subproc.child_env()`。这条门是
    #      **结构性**的——①②③④ 四道门都只是「当前这几处坏了」，只有它是
    #      「以后新加一处也不会坏」。白名单三条（`git rev-parse` / `git ls-files` /
    #      `--collect-only` 数数）都只读 ASCII，理由逐条写在 `ENV_WHITELIST` 里；
    #      `test_whitelist_points_at_real_calls` 防死条目让门悄悄变宽。
    #   ③ 同文件的控制台门
    #      （`test_scripts_never_print_chars_the_console_cannot_encode`）：
    #      脚本自己往 cp936 控制台写它编不出的字符（U+2212 / U+26A0 / U+2194 等）
    #      会让 `print` 抛 `UnicodeEncodeError`、进程以 1 退出——**与一个真实的
    #      业务失败无法区分**。实测 `scripts/eval_fusion.py:297` 的减号 − 就是这种，
    #      而它在 pytest 里跑是绿的（pytest 捕获 stdout，写进内存缓冲，不碰控制台
    #      编码），只有被当命令行工具直接跑时才崩——所以只有 AST 门抓得住。
    #      这道门自己踩过一次坑，写进用例注释了：第一版用
    #      `"reconfigure" in source` 判断，反向实测 **19 passed 没变红**，因为
    #      我给 eval_fusion 加的中文注释里就有「reconfigure」这个词，注释替真代码
    #      把门撑住了。改成 AST 识别真实调用后，删掉那两行 → `2 failed, 18 passed`。
    #   ④ `test_regression_entry.py` + `scripts/regression.py`：回归入口必须
    #      **以 pytest 的退出码为自己的退出码**。这条门针对的不是代码缺陷，是
    #      **调用链**：`py -3 -m pytest tests -q | Select-Object -Last 40` 这样写，
    #      `$LASTEXITCODE` 是管道右侧 `Select-Object` 的 0，于是「1498 passed /
    #      4 failed」那次的外层命令报了成功。它同时钉住 Lead 特别要求的那条：
    #      摘要里出现 `0 failed` 不许判红——裸匹配 `"failed" in text` 会把
    #      「1511 passed, 0 failed」读成失败，那比漏判更糟，因为它会训练人
    #      忽略这道门。反向实测两笔：把 `return returncode` 改成 `return 0` →
    #      `test_entry_exits_nonzero_when_pytest_fails` 红（1 failed, 19.27s）；
    #      把 `summary_looks_bad` 改成裸匹配 → `test_zero_failed_summary_is_not_bad`
    #      红（0.29s）。
    # 1550 → 1573（P1-a 切分落文件）：一次性记两笔。
    #   · `tests/test_make_split.py` 17 个（新文件，进入 `TEST_FILES` 后让四个
    #     parametrize（本文件 61/130/150/184 行）各多一项，+4）；
    #   · `tests/test_k1_checks.py` +2 个（`test_gate_points_at_v2_not_v1` 与
    #     `test_v1_file_still_present_and_untouched`），该文件已在 `TEST_FILES` 里，
    #     不产生 parametrize 增量。
    #   17 + 4 + 2 = 23，1550 + 23 = 1573，与 `--collect-only` 实测一致
    #   （把新文件临时移走后实测 1552 = 1550 + 2，带回来后 1573，账对得上；
    #   同时本文件从 186 涨到 190，正是那 +4）。
    # 这两笔换来的门是「切分规则本身有守门人」——在此之前，切分只存在于一次性的
    # 试算脚本里，改一个 `ceil` 成 `round`、或者把分层键里的 `should_abstain`
    # 去掉，都不会有任何测试变红，而这两种改动都会让「每类都要有代表」这句话失效：
    #   · `test_ceil_keeps_small_strata_in_holdout`：`n=2, ratio=0.30` 时 `round`
    #     给出 0 条进 holdout，holdout 就缺了这一整类。用 `ceil` 才保证小层至少
    #     有一条。这是「每类都要有代表」的算术前提，不是风格偏好。
    #   · `test_split_is_deterministic` / `test_bucket_is_stable_and_depends_on_tag`：
    #     切分靠 `sha256(SPLIT_SEED|tag|question_id)` 排序，不靠任何全局状态。
    #     改拼串方式等于重新切分，已入库的 holdout 就不再是同一批题——而这件事
    #     没有任何运行时症状，只有指纹会变。
    #   · `test_v1_half_is_byte_identical_except_split_and_origin` +
    #     `test_self_check_catches_tampered_v1_item`：v2 里 `origin=v1` 的那 320 条
    #     必须除 `split`/`origin` 外逐字段等于 `gold-v1.json` 原件。**这条最要紧**：
    #     `gold_set_fixed` 门只算 `gold-v2.json` 整个文件的 sha256，它看得见「文件
    #     变了」，却看不见「变的是 v1 那半还是新增那半」——有人顺手改一条 v1 题的
    #     `required`，恰好又同步更新了哈希常量，门是绿的，历史基线却已经漂移。
    #   · `test_gate_points_at_v2_not_v1`：换签只改 sha 常量、忘了改 `GOLD_SET_PATH`，
    #     门会「通过」在 v1 上，绿得毫无意义。
    # 反向实测两笔（日志 `%TEMP%\gate0\logs\reverse_gold_v2_gate.txt`）：
    #   ① 把 `GOLD_SET_SHA256_16` 的最后一位 `4b9` 改成 `4ba` →
    #      `py -3 -m pytest tests/test_k1_checks.py -q -k GoldSet` →
    #      **1 failed, 4 passed, exit 1**，`check_gold_set_fixed()` 的 detail 是
    #      「sha256 前 16 位 069375f8e355e4b9 ≠ 冻结值 069375f8e355e4ba」——
    #      红在正确的原因上；恢复后 5 passed, exit 0。
    #   ② 把 `GOLD_SET_PATH` 指回 `gold-v1.json`（常量仍是 v2 的）→ 同一命令
    #      **2 failed, 3 passed, exit 1**（`test_real_gold_set_passes` +
    #      `test_gate_points_at_v2_not_v1`）；恢复后 5 passed, exit 0。
    # 1573 → 1576（build_context 顺序门）：一笔。
    #   · `tests/test_retrieve.py` 新增 `TestContextOrderComesFromScore` 三个用例
    #     （该文件已在 `TEST_FILES` 里，不产生 parametrize 增量）；
    #   1573 + 3 = 1576，与 `--collect-only` 实测一致（`test_retrieve.py` 由 26 涨到 29）。
    # 这一笔换来的门是「排序改动的假增益会被拦住」。Lead 2026-11-05 真跑排序改动时
    # 踩到：`build_context` 的 `key=lambda u: (0 if direct(u) else 1, -u["score"],
    # u["chunk_id"])` **只读 `hit["score"]`，完全不读 hits 的先后**。他在 `search()`
    # 层重排 hits 却不同步更新 score，四个配置输出逐位相同——加成对上下文零影响；
    # 而 R@5/R@20 会涨，指标报告显示「优化成功」。这是**假增益**，比一个不红的门
    # 更危险：它长得像通过。三条用例缺一不可：
    #   · `test_reordering_hits_without_changing_score_does_not_move_units`：
    #     三种打乱 + score 不动 → 单元顺序与 `context_digest` 都不许变。
    #   · `test_changing_score_alone_does_move_units`：**反向控制**。若没有这一条，
    #     上一条可能只是在断言一个恒真的东西（比如实现永远按 chunk_id 排）。
    #     这条自己踩过坑，写进用例注释了：第一版用「score 取反」，看着是绿的，
    #     其实是假的——`unit["score"]` 初值 0.0 且用 `max()` 累积，**负分被夹回
    #     0**，同分后退回 chunk_id 定序，两单元时恰好等于整体反转。改成
    #     `CEILING - score`（保正、大小顺序整体颠倒）才有判别力。
    #   · `test_equal_scores_break_ties_by_chunk_id_not_input_order`：同分按
    #     `chunk_id` 升序、与输入顺序无关，否则同样输入两次跑出不同上下文。
    # 反向实测（日志 `%TEMP%\gate0\logs\reverse_context_order_gate.txt`）：
    #   ① 改前 `py -3 -m pytest tests/test_retrieve.py -q -k ContextOrderComesFromScore`
    #      → **3 passed, exit 0**；
    #   ② 把 `ordered = sorted(...)` 临时改成 `ordered = list(units.values())`
    #      （即按 hits 出现顺序）→ 同一命令 **3 failed, exit 1**，
    #      报错原文「只改了 hits 的顺序、score 没动，上下文单元顺序却变了……」
    #      并打印输入顺序 `['doc/b#000','doc/a#001','doc/a#000']` 对基线
    #      `['doc/a#000','doc/a#001','doc/b#000']`——红在正确原因上；
    #   ③ 恢复后 **3 passed, exit 0**，`git diff src/arkrag/retrieve.py` 为空。
    # 1576 → 1631（P0 门报告可追溯）：一次性记两笔。
    #   · `tests/test_provenance.py` 36 个（新文件，进入 `TEST_FILES` 后让四个
    #     parametrize（本文件 61/130/150/184 行）各多一项，+4）；
    #   · `tests/test_chunks_source.py` 11 个（新文件，同样 +4）。
    #   36 + 4 + 11 + 4 = 55，1576 + 55 = 1631，与 `--collect-only` 实测一致
    #   （本文件同时从 190 涨到 198，正是那 +8）。
    # 这两笔换来的门是「门报告的结论可追溯到哪份代码、哪套数据、哪个时刻」——
    # 在此之前，`data/` 下九份门报告里只有 `k2_perf.json` 与 `k2_gray.json` 带
    # `generated_at`，`k2_hard_gates.json` / `k3_threshold_gate.json` /
    # `k4_drill.json` 三份**一样也没有**；而 `scripts/k2_publish_check.py` 的
    # `check_reports_freshness()` 对查不到时间戳的报告记 `undetermined` 之后，
    # **整道门仍然能过**——「查不了」被安静地当成「不用查」。
    #   · `TestUnknownIsNotPass`（7 条参数化）：七个必填字段逐个删掉，逐次都必须
    #     判 `unknown`，且 `gate_ok()` 必须为 False。这条对付的正是上面那个形状：
    #     缺信息不等于没问题。另有一条 `test_fail_outranks_unknown` 钉住
    #     `_fold()` 的优先级——FAIL 要压过 UNKNOWN，否则「有一个字段填错了」
    #     会被「另一个字段没填」稀释成 unknown。
    #   · `TestMissingInputIsDistinguishable`：`verify()` 必须能在 `fields` 里点名
    #     缺的是哪个字段，且 `missing`（没填）与 `invalid`（填了但不合法）是两个
    #     名单。P1-c 那次 `gold_assertion_ids` 在 triage 行里恒为 None、导致全部
    #     落到「无法判定」，就是这个形状：判据没取到输入，安静地降级成一个
    #     看起来合理的类别，不报错、不崩、退出码 0。
    #   · `TestCaliberAndDownstream`：含 recall / R@n 的行必须带 `caliber`，取值只
    #     允许 `gold_only` / `equiv_v2`；`equiv_v1_wide` 已作废，出现即 FAIL；
    #     只报 R@20 而没有下游量的行必须 FAIL。反例是 Lead 的六配置表：
    #     `hop0.5` 的 R@20=0.7379 排全场第二，但真进上下文净增 0、引用覆盖 51→48。
    #     只看 R@20 的报告会推荐它。
    #   · `TestReverseProbeRefusesStaleEvidence`：退出码非 0 时产物一律无效、
    #     每轮开始先清上一轮产物。同族事故已经三次（Gate 0 的管道退出码被
    #     `Select-Object` 掩盖；我自己那轮 patch 锚点写错让脚本崩了、harness 却
    #     读到上一轮残留 JSON 报「结论没变」；邻接表字段名写错导致回放与现状逐位
    #     相同），所以把它做成工具 `src/arkrag/reverse_probe.py`，harness 自己也用它。
    # 第二笔（`test_chunks_source.py`）换来的门是「块正文来源不许选错那一份」。
    # 同一个坑踩过三次：D 阶段 13,798 vs 13,781；P1-b 结构报告分子读全局 13,786、
    # 分母读世代 13,798；P1-c 脚本路径写错又带一句 `if not path.exists(): 回退到
    # 全局`。三次都不报错。`pipeline.resolve_chunks()` 是仓库已有的正解（它会核
    # `chunk_manifest_hash`，`exists()` 检查抓不到这一层）。门分两层：
    #   · 新增违规点位立刻红（`test_no_new_offender_beyond_registered_debt`）；
    #   · **修好一个点位忘了删登记也红**（`test_debt_registry_matches_measured`）——
    #     只做「新增就红」的登记表会越积越厚，最后没人相信里面有活条目；
    #     「修好了也要改登记」才让这份名单有下行压力。
    #   另有 `test_allowed_and_debt_do_not_overlap` 防「为了让门绿把债塞进白名单」，
    #   `test_allowed_entries_point_at_real_calls` 防死条目让门悄悄变宽，
    #   `TestGateItselfCanFail` 四条合成源码用例证明判据不是恒真。
    # 反向实测六笔（日志 `%TEMP%\gate0\logs\reverse_p0_gates.txt`，脚本
    # `tmp_b_p0_reverse.py`；harness 自己用 `reverse_probe.run()` 跑，自证两条规矩）：
    #   A 去掉 `_fold()` 的 UNKNOWN 分支 → `-k test_each_field_missing` **7 failed,
    #     exit 1**；恢复到 36 passed, exit 0。
    #   B `verify()` 里不再 `missing.append(name)` → `test_verify_names_the_missing_field`
    #     **1 failed, exit 1**；恢复绿。
    #   C `reverse_probe.run()` 的 `if proc.returncode != EXIT_OK` 改成 `if False`
    #     → `test_probe_rejects_artifacts_when_command_fails` **1 failed, exit 1**；恢复绿。
    #   D `cleared = clear(arts)` 改成 `cleared = []` → `test_probe_clears_previous_artifacts`
    #     **1 failed, exit 1**；恢复绿。
    #   E 作废口径分支改成 `if False` → `test_retired_wide_caliber_is_fail_not_unknown`
    #     **1 failed, exit 1**；恢复绿。
    #   F 缺下游量分支改成 `if False` → `test_recall_without_downstream_is_fail`
    #     **1 failed, exit 1**；恢复绿。
    # 1631 → 1666（P0 门报告体检命令 `scripts/check_report_provenance.py`）：一笔。
    #   · `tests/test_report_provenance.py` 31 个（新文件，进入 `TEST_FILES` 后让四个
    #     parametrize（本文件 61/130/150/184 行）各多一项，+4）；
    #   31 + 4 = 35，1631 + 35 = 1666，与 `--collect-only` 实测一致（本文件由 198
    #   涨到 202，正是那 +4）。
    # 这一笔换来的门是「报告的可追溯性有一个能跑、有退出码的判据，而不是一段
    # 写在文档里的要求」。`tests/test_provenance.py` 守的是**判据**本身；
    # 这个文件守的是**仓库里那九份真实报告**，两者缺一不可：
    #   · `test_debt_registry_matches_actual_state`：某份报告补上七字段后必须从
    #     `REPORTS_AWAITING_PROVENANCE` 删掉。没有这条，名单只会越来越长，
    #     最后没人相信里面有活条目。
    #   · `test_no_unregistered_noncompliant_report`：新报告不许绕开登记。
    #     **这条第一版写成了空转**：它遍历的是「已登记名单」，所以永远不可能
    #     发现未登记项——第一版反向实测里替它变红的是旁边那条名单覆盖用例。
    #     现在两条都从 `_gate_reports_on_disk()`（磁盘实测）出发，分工是
    #     「有没有被漏掉」与「漏掉的里面有没有不合格的」，各自独立变红（见下 M1/M2）。
    #   · `TestCliExitCodes`：命令的退出码必须跟着报告翻身——缺 `git_commit`
    #     的报告退出码 1 且 stdout 点名缺哪项；报告文件不存在也非零
    #     （「不存在」不能算通过）。
    #   · `test_cli_reports_relative_paths_only`：输出里不许出现机台绝对路径
    #     （2026-11-04 真发生过一次入库 JSON 带 `E:\...` 被 portable_paths 判红）。
    # 实测的存量债（2026-11-05 体检，写进 `docs/p0_门报告可追溯.md`）：
    #   `k2_hard_gates.json` / `k2_publish_check.json` / `k3_threshold_gate.json` /
    #   `k4_drill.json` / `p1c_leak_audit.json` 五份连 `generated_at` 都没有；
    #   `k2_perf.json` / `k2_gray.json` 有 `generated_at` 但没有代码版本。
    #   七份全部 `verdict=unknown`、`gate_ok=False`——按新规矩它们都**不算通过**，
    #   这与「P0 之前的门能过」正是同一件事的两面。
    # 反向实测五笔（日志 `%TEMP%\gate0\logs\reverse_report_provenance.txt`，
    # 脚本 `tmp_b_p0_report_reverse.py`）：
    #   J 删 `provenance._fold()` 的 UNKNOWN 分支 →
    #     `-k test_a_report_missing_a_field_is_unknown` **7 failed, 24 deselected,
    #     exit 1**（七条参数化全红，报错原文 `- unknown / + pass`——缺字段被判成了
    #     通过）；恢复后 7 passed, exit 0。
    #     注：我第一版注释这里写的是「5 failed，前两条仍绿」，那是凭预期写的、
    #     没有实测（日志 tail 把汇总行截掉了）。补跑 `tmp_b_p0_j_count.py` 得到
    #     真实数字是七条全红。**这正是本项目第三次栽的那种「记录比实测好看」
    #     ——现在这行数字有对应的完整日志。**
    #   K 把 `scripts/check_report_provenance.py` 的退出码改成恒 `return EXIT_OK`
    #     → `-k test_cli_exits_nonzero_for_a_noncompliant_report` **1 failed, exit 1**
    #     （报错 `assert 0 == 1`，脚本明明打印了「缺 1 项：git_commit」却退出 0）；
    #     恢复绿。
    #   L 给 `data/k2_perf.json` 临时补上七字段、但**不删**债表登记 →
    #     `-k test_debt_registry_matches_actual_state` **1 failed, exit 1**
    #     （报错「这些报告已经满足七字段了，请从 REPORTS_AWAITING_PROVENANCE
    #     删掉：['k2_perf.json']」）；恢复绿。
    #   M1 往 `data/` 放一份新的、无七字段、未登记的门报告 →
    #     `-k test_no_unregistered_noncompliant_report` **1 failed, exit 1**
    #     （报错点名 `['_gate0_fake_gate_report.json']`）；
    #   M2 同一份假报告，改跑 `-k test_audited_set_covers_every_gate_report_on_disk`
    #     → **1 failed, exit 1**。两条各自独立变红，证明分工真的修好了；
    #     跑完假报告已删除，`[Z]` 恢复后 31 passed, exit 0。
    #
    # 1666 → 1669：`tests/test_provenance.py` 36 → 39（同文件新增 3 条，该文件
    # 已在 `TEST_FILES` 里，不产生 parametrize 增量）。算术：1666 + 3 = 1669，
    # 与 `--collect-only` 实测一致。
    # 这 3 条换来的是「缺输入伪装成通过」的**第二种形态**——它不是「判据没取到
    # 输入后降级成无法判定」（那是 P1-c 那次），而是「判据根本没看这一行，
    # 却报告检查过了」：
    #   `provenance.collect(generation_id="g-x", inputs=["data/goldset/gold-v2.json"])`
    #   产出的 `inputs` 是纯字符串列表，而旧 `judge_inputs()` 只遍历
    #   `isinstance(row, dict)` 的行 → 字符串行**一条都不检查** → 返回
    #   `("pass", "inputs 1 个文件都已登记 sha16")`。**一个 sha16 都没登记，
    #   却报「都已登记」。**
    #   反向实测四段（日志 `%TEMP%\gate0\logs\reverse_inputs_shape.txt`，
    #   脚本 `tmp_b_p0_inputs_shape.py`，用「退回旧实现」而不是「删掉新代码」）：
    #     [1] 修好之后 → 3 passed, exit 0；
    #     [2] 退回旧实现（只检查 dict 行）→ **2 failed, 1 passed, exit 1**；
    #     [3] 恢复后 → 3 passed, exit 0；
    #     [4] 全文件 → 39 passed, exit 0。
    # 1669 → 1724：两笔账。
    # ① `tests/test_provenance.py` 39 → 61（同文件 +22，该文件已在 `TEST_FILES`
    #    里，不产生 parametrize 增量）。这 22 条对应 Lead 2026-11-05 的三条新判据：
    #    · 信号密度 `TestSignalDensityIsObservable`（7 条）：A 的 P2 第一轮实测
    #      cooc 加成在密度 `boosted/units` 4+ 的 35 题上净增益为 0（常数加成在
    #      命中块数远超上下文容量、units 中位 5 时全体同加同量，相对顺序不变，
    #      等于没加）。报告里没有这个数，下一轮调参就还会拿「R@20 涨了」当成功。
    #      缺 `signal_boosted`/`context_units`/`density` 任一 → unknown；
    #      `density` 与 `boosted/units` 不一致 → fail（手填的漂亮数字不算）；
    #      `context_units=0` → fail（没有上下文单元的行不该进指标）。
    #    · `TestSignalDensityBlock`（6 条）：构造器本身。`signal_boosted` 取各信号
    #      计数的 **max 不取 sum**——同一块被两路加成仍只是一块，求和会把密度
    #      1.4 说成 2.4、跨过「2–4 只救 1 题」那一档，结论就反了。
    #    · 耗时 `TestDurationNeedsCpuTime`（4 条）：只报墙钟不报 CPU → unknown。
    #      实测反例：一次全量回归报 24141.60s，而进程 CPU 累计只有 ~778s——机器
    #      休眠 6 小时。判 unknown 不判 fail：那个数**可能**是真的，只是无法证实。
    #      `p95_ms`/`median_ms`/`qps` 不误伤（单请求延迟天生是墙钟，睡眠期间没有
    #      请求在跑）。
    #    · 并列召回 `TestParallelRecallMetricsNeedContainment`（5 条）：
    #      `any_gold +6` / `all_gold +4` / `covered +5` 不是三份独立证据——
    #      `required_citations` 块集恒等于 `gold_chunk_ids`，实测三个增益题集的
    #      差集只有 `q-tc-000023` 一题。2 个以上带 `gain_ids` 的行没给
    #      `relation_table` → unknown；手填一张与实际集合矛盾的表 → fail。
    #      反向实测六笔（日志 `%TEMP%\gate0\logs\reverse_density_duration_gain.txt`）：
    #      A 两个判据调用改恒 PASS → **6 failed**（逐条点名，不是「有红就行」）；
    #      B 一致性检查 `if False` → 1 failed；C 耗时 unknown→pass → 1 failed；
    #      D 缺表分支 unknown→pass → 1 failed；E 矛盾比对 `if False` → 1 failed；
    #      F `max`→`sum` → 1 failed；`[Z]` 恢复后 61 passed, exit 0。
    # ② 新增 `tests/test_signal_density_audit.py` 29 条（新文件进 `TEST_FILES`，
    #    让四个 parametrize（本文件 61/130/150/184 行）各多一项 +4）。
    #    算术：1669 + 22 + 29 + 4 = 1724，与 `--collect-only` 实测一致
    #    （本文件自身 202 → 206，多出的 4 就是 parametrize 增量）。
    # 这里换来的门是「密度分桶必须可复算，且两种分母不许合并」：
    #    · `TestRealReport::test_bucket_counts_match_the_lead_table` 把
    #      `boost_applied_only` 口径下 0-1/1-2/2-4/4+ 的 n 钉死成 38/41/21/35
    #      （合计 135 = 真正把加成写进 score 的题数，不是 350、也不是 141）。
    #      我第一次跑出 253/41/21/35（全 350 题做分母），与 Lead 的表差 215 题；
    #      第二次跑出 44/41/21/35（用 `signal_boosted>0` 筛，多 6 道「命中了但增幅
    #      被夹成零」的题）。**两次都是「分母口径不同」，每个数都真、合起来意思
    #      不对**——这就是本项目栽过四次的同一个坑，所以两种口径必须同时出现。
    #    · `test_the_4plus_bucket_gains_nothing`：整条门的结论就一句——密度 4+ 的
    #      35 题，`any_gold`/`all_gold`/`covered` 三个指标净增益都是 0。
    #    · `test_gain_set_difference_is_the_one_known_question`：并列召回指标的
    #      增益差集只有 `q-tc-000023`，独立复现 Lead 的数。
    #    · `test_self_check_catches_wrong_caliber_declaration`：桶表两口径齐全、
    #      只有 `calibers` 那行声明少一个，必须被点名。**这条是被反向实测逼出来
    #      的**——我最初只写了 `test_self_check_catches_single_caliber_report`，
    #      而它会被「桶表缺口径」那条判据兜住，于是把声明检查关掉它照样绿
    #      （`[C]` 笔 returncode=0）。补了这条之后 `[C-补]` 才指名道姓变红
    #      （日志 `%TEMP%\gate0\logs\reverse_density_caliber_decl.txt`：
    #      改前 1 passed exit 0 / 改后 1 failed exit 1 / 恢复 1 passed exit 0）。
    #      **反向实测没变红不等于门是坏的，等于那条用例测的不是那个判据。**
    #    另三笔（日志 `%TEMP%\gate0\logs\reverse_signal_density_audit.txt`）：
    #      A 砍成单一口径 → **4 failed**（`test_report_declares_both_calibers`、
    #        `test_bucket_counts_match_the_lead_table`、
    #        `test_the_4plus_bucket_gains_nothing`、
    #        `test_self_check_catches_single_caliber_report`）；
    #      B 筛选 `signal_applied`→`signal_boosted` → **2 failed**（分桶数与
    #        `test_caliber_filters_on_applied_not_boosted`）；
    #      D `bucket_of(None)` 落进 0-1 → 1 failed；`[Z]` 恢复 28 passed。
    # 1724 → 1741：P1-c 剩余（A 类 27 条「gold 引用指错行」的 required 判定）
    # 从临时脚本搬进 `scripts/p1c_leak_audit.py`，并加 `tests/test_p1c_leak_audit.py`。
    # 算术：1724 + 13 + 4 = 1741。13 是新文件的用例数；+4 来自新文件进
    # `TEST_FILES` 后让四个 parametrize（本文件 61/130/150/184 行）各多一项。
    # 与 `--collect-only` 实测 1741 一致（本文件自身 206 → 210，多出的 4 就是
    # parametrize 增量）。
    # 这里换来的门是「A 类那 27 条到底能不能只改 required 指向」：
    #    · 这道门守的是**改 gold 的第一步**。P1-c 的判定会直接导致改标注，而改
    #      gold 是改代码改 hash 的动作（要 Lead 签字）；一条判错的 `move_safe`
    #      会让「别名出现」这个唯一证据从 gold 里消失，事后无法回头核对。
    #    · `TestFourVerdicts::test_alias_on_the_answer_line_is_what_makes_it_safe_to_move`
    #      与 `test_answer_on_a_line_without_the_alias_must_not_be_moved` 成对存在，
    #      钉住 `move_safe` 与 `needs_extra` 的分界：两者都表现为「答案在同块
    #      另一行」，但那一行**含不含别名**决定「能挪」还是「挪了更糟」。把这两态
    #      混成一种，就会把它们一起报成「dev 侧 15 条可改」——而实测 dev 侧
    #      `move_safe` 是 **0 条**。
    #    · `test_the_answer_sitting_on_the_required_line_needs_no_change` 钉住
    #      「引用口径窄 ≠ gold 缺陷」：gold 把说话人前缀（「杜林青年：」）也写进了
    #      reference，断言从冒号后面开始，改它等于改指标不是改标注。
    #    · `test_boolean_answers_are_not_judged_by_substring` 钉住布尔答案那 2 条
    #      （`q-na-000004`/`q-na-000007`）：答案「否」会匹配到断言「火龙并没有被
    #      源石感染」里的「没有」，机器判据在这里必然误命中，只能判无法判定。
    #    · `TestRealRun` 把结论钉死：59 条候选 0 真漏标 / 58 不是漏标 / 1 无法判定；
    #      A 类 27 条 dev 21 / holdout 6；dev 侧 `{needs_extra: 13, carrier_ok: 6,
    #      boolean_answer: 2}`；holdout 只记录不改。
    #    · `test_chunks_come_from_the_generation_not_the_corpus_root` 钉住块正文来源
    #      必须是代内的 13798 块（这个坑本项目踩过三次：D 阶段 13,798 vs 13,781、
    #      P1-b 结构报告 13,786 vs 13,798、P1-c 第一版路径写错还静默回退）。
    # 反向实测四笔（日志 `%TEMP%\gate0\logs\reverse_p1c_a27.txt` 与
    # `reverse_p1c_a27_B.txt`），每笔都是改前 0 / 改后 1 / 还原 0，且**指名道姓**
    # 红在预期那一条上：
    #    A 删掉 `move_safe` 分支 → `test_alias_on_the_answer_line_is_what_makes_it_safe_to_move`
    #      （报错里 `+ needs_extra`）；
    #    B 把 `move_safe` 条件放宽成「只要答案在别的行就算可挪」→
    #      `test_answer_on_a_line_without_the_alias_must_not_be_moved`（`+ move_safe`）；
    #    C 把「承载答案的断言与 required 同行」的区间判据改成恒假 →
    #      `test_the_answer_sitting_on_the_required_line_needs_no_change`（`+ no_support`）；
    #    D 清空 `BOOLEAN_ANSWERS` → `test_boolean_answers_are_not_judged_by_substring`
    #      （`+ no_support`）。
    #    **B 第一次没做成实测**：用例选择器写成 `"needs_extra or zero_move_safe"`，
    #    那不是有效的 `-k` 表达式，只选中一条恰好绿的用例，于是记成「没变红」。
    #    重做时才拿到 0/1/0。**「没变红」先怀疑选择器与用例，再怀疑门。**
    # 1741 → 1743：`tests/test_p1c_leak_audit.py` 再加两条（该文件已在
    # `TEST_FILES` 里，不产生 parametrize 增量）。算术：1741 + 2 = 1743，
    # 与 `--collect-only` 实测一致（本文件自身仍是 210）。
    #    · `test_change_log_declares_zero_gold_changes_and_two_impact_counts`：
    #      判定表里有「变更日志」一节，且必须同时写「dev 影响 0 题 / holdout 影响
    #      0 题」与「审核人空缺」。这张表被改过三轮、每轮结论都变过；不记账的话
    #      下一个人分不清哪一版是真的，而「改了 0 题」这句话必须和「审核人尚未
    #      签字」写在一起——两件事互为证据（Lead 的裁决格式要求 dev / holdout
    #      分开写、审核人空缺不许提交）。
    #    · `test_gold_files_are_untouched`：钉住 `gold-v1.json`（284356 字节 /
    #      `76d66073ff87aab1`）与 `gold-v2.json`（601492 字节 / `069375f8e355e4b9`）
    #      一字节未动。三轮判定下来对 gold 的影响是 **0 题**，这条用例让那个「0」
    #      不是一句声明，而是一个可复算的断言。
    # ------------------------------------------------------------------
    # N: 1743 -> 1760（2026-11-05，三道新判据）
    #
    # 一笔账：`tests/test_provenance.py` 61 -> 78，+17 条，分三个新类：
    #   TestCoverageNeedsThreeDenominators（7）：引用覆盖行必须并列三个分母
    #     （/300 混合、/194 断言层可达、/72 两闸都过），且题数单调不增。
    #     换来的门：把 `denominators` 改成一个数或停掉整条判据 → 5 failed；
    #     只停单调性 → 1 failed。
    #   TestShuffleRatioIsAnEliminationCriterion（6）：净增益 ÷ top-K 变动集合
    #     低于 0.1 判洗牌，且必须声明「只作淘汰用」。
    #     换来的门：停整条判据 → 3 failed；只停「必须声明淘汰」→ 1 failed。
    #   TestRank5CliffMustBeStated（4）：报前 5 名内的指标必须附第 5 名转化率
    #     0.4167（实测 1–4 名 1.0000、第 5 名 5/12、第 6 名 1/18、7–20 名 0/44）。
    #     换来的门：停整条判据 → 2 failed；只停「编造的转化率判 fail」→ 1 failed。
    # 该文件**已在** `TEST_FILES` 里，不产生 parametrize 增量。
    #
    # 算术：1743 + 17 = 1760，与 `py -3 -m pytest tests --collect-only -q`
    # 实测 1760 一致。
    #
    # 反向实测（命令级，日志 %TEMP%\gate0\logs\reverse_coverage_shuffle_rank5.txt）：
    #   [0] 基线 78 passed 退出码 0
    #   [A] 三分母判据恒 PASS → 5 failed 退出码 1；恢复 78 passed 退出码 0
    #   [B] 单调性检查 `if False` → 1 failed 退出码 1；恢复 0
    #   [C] 洗牌判据恒 PASS → 3 failed 退出码 1；恢复 0
    #   [D] 「低转换率必须声明淘汰」`if False` → 1 failed 退出码 1；恢复 0
    #   [E] 断崖判据恒 PASS → 2 failed 退出码 1；恢复 0
    #   [F] 编造转化率不判 fail → 1 failed 退出码 1；恢复 0
    # 六笔都指名道姓核过红的是哪一条，不是「有红花就算数」。
    #
    # 踩到的坑（记在这里免得下次重犯）：`r5` 同时是 recall 键，所以只带
    # `rank5_conversion` 而不带 caliber/下游量的行会**先红在口径上**，断崖这条
    # 根本没被执行到 —— 三条断崖用例最初就是因此假绿。补上 caliber 与
    # context_hits 之后才真的测到它。
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # N: 1760 -> 1789（2026-11-05，引用落点观测可复算）
    #
    # 两笔账，算术 1760 + 25 + 4 = 1789：
    #   · 新增 `tests/test_needs_extra_citation.py`，25 条用例；
    #   · 该文件**不在** `TEST_FILES` 的旧名单里，所以本文件四个 parametrize
    #     （test_file_has_pytest_visible_tests / test_every_test_function_can_fail
    #     / test_assertion_helpers_raise / test_no_repo_relative_path_literal）
    #     各 +1，共 +4。
    # 与 `py -3 -m pytest tests --collect-only -q` 实测 1789 一致；
    # 用 `--ignore=tests/test_needs_extra_citation.py` 复跑得 1764 =
    # 1789 - 25，差的 4 条正是上面那四个 parametrize（glob 仍认这个文件名）。
    #
    # 这次换来的门（逐条写明，只报数字不算）：
    #   · `test_an_assertion_inside_the_line_makes_it_frag` +
    #     `test_the_boundary_does_not_use_the_verdict`：钉住 frag / none 的
    #     分界是「required 那一行**有没有被抽成断言**」，不是「答案在不在那一
    #     行」。若按后者分界，frag 群会恒空——一个恒空的群看起来像结论，实际是
    #     判据与定义重复。把 `group_of` 开头改成恒返回 none → 2 failed。
    #   · `test_an_assertion_spanning_past_the_line_does_not_count`：跨出行外的
    #     断言不算「落在这一行」，否则「这一行被切碎了」的判断会失真。把区间
    #     判据从「完全落在行内」放宽成「相交即算」→ 1 failed。
    #   · `test_the_four_cells_sum_to_n` / `test_a_row_cited_on_both_lines_lands_in_the_both_cell`
    #     / `test_required_line_cited_is_the_union_not_a_cell`：一次回答可能同时
    #     引用「required 那一行」与「承载答案的另一行」，所以三个格加不出总数。
    #     漏掉第四格（两行都有）时，三个数相加比总数多，而多出来的恰好是「看
    #     起来最正常」的那种行。把 `required_and_carrier` 记成 0 → 3 failed。
    #   · `test_observations_strip_volatile_fields` + `test_check_passes_on_the_committed_file`：
    #     `--check` 比的是「引用落点变了吗」，不是「文件字节一样吗」。把溯源
    #     字段算进比较 → 这道命令永远红、那等于没有门。改回 `dict(payload)` → 2 failed。
    #
    # 反向实测（命令级，日志 %TEMP%\gate0\logs\reverse_needs_extra_citation2.txt）：
    #   [0] 基线 25 passed 退出码 0
    #   [A] `group_of` 恒返回 none → 2 failed 退出码 1；恢复 25 passed 退出码 0
    #   [B] 断言区间判据放宽成相交即算 → 1 failed 退出码 1；恢复 0
    #   [C] `required_and_carrier` 记 0 → 3 failed 退出码 1；恢复 0
    #   [D] `_observations` 返回整个 payload → 2 failed 退出码 1；恢复 0
    #
    # 踩到的坑（第一版四笔里两笔「没变红」）：那一版的用例全部读**已写出的
    # JSON**，改坏脚本实现不会触发重算，用例读到的还是上一轮数据 —— 也就是说
    # 那两道门在测磁盘状态，不在测实现。修法是把 `group_of` / `tally_rows` 抽成
    # 模块级纯函数，用例喂合成输入：改坏实现必然红，与磁盘无关。
    # 这与「注释替真代码把门撑住」是同族：**门必须直接调用被判定的东西。**
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # N: 1789 -> 1796（2026-11-05，闸门判定顺序必须是因果顺序）
    #
    # 算术 1789 + 3 + 4 = 1796：
    #   · 新增 `tests/test_gate_order.py`，3 条用例；
    #   · 该文件**不在** `TEST_FILES` 的旧名单里，所以本文件四个 parametrize
    #     各 +1，共 +4。
    # 与 `py -3 -m pytest tests --collect-only -q` 实测 1796 一致。
    #
    # 这次换来的门（逐条写明，只报数字不算）：
    #   · `test_block_missing_from_context_is_gate2_even_when_admission_also_fails`：
    #     块不在上下文、断言又过不了准入判据时，必须记**闸门2**而不是闸门1b。
    #     块不在上下文时 `_assertion_relevant()` 一次都没被调用，说它「被准入
    #     挡掉」是给没发生的事编因果；后果是把 P2 的活算进 P3，分母虚增
    #     （v1 上 29 vs 19，多出的 10 道里 5 道 multihop 的 required 块
    #     `in_context=False`）。把 `classify_gate` 改回「先 admit 后 incl」
    #     → 1 failed。
    #   · `test_block_in_context_with_blocked_assertion_is_gate1b`：反例侧控制，
    #     块在上下文时被挡才叫闸门1b。没有它，「先判 incl」可以被写成
    #     「永远返回闸门2」而这条门仍是绿的。
    #   · `test_admission_passes_when_endpoint_is_a_focus_entity`：端点落在焦点
    #     实体上时准入必须过（变成闸门3）。没有它，「闸门1b」可能只是
    #     「准入判据对谁都返回 False」的假象。
    #
    # 反向实测（命令级）：改前 `py -3 -m pytest tests/test_gate_order.py -q`
    #   → 3 passed 退出码 0；把 `scripts/lead_gate_recheck.py::classify_gate`
    #   的两段判定换回「先 admit 后 incl」→ **1 failed, 2 passed 退出码 1**，
    #   红的正是第一条；`Copy-Item` 还原 → 3 passed 退出码 0，
    #   `git diff --stat` 显示该文件只剩本次 6 插 2 删的改动，实验改动没留下。
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # N: 1796 -> 1811（2026-11-05，正确覆盖成对 + 回归入口自采 CPU）
    #
    # 算术 1796 + 7 + 8 = 1811：
    #   · `tests/test_provenance.py` 新增 `TestCorrectCoverageMustPairWithCoverage`
    #     7 条（先前在 Lead 工作区里未提交，本轮一并入库）；
    #   · `tests/test_regression_entry.py` 新增 8 条 CPU 采集门
    #     （该文件原有 9 条 → 现 17 条）。
    #   · 两个文件**都已在** `TEST_FILES` 里（`TEST_FILES` 是
    #     `sorted(TESTS.glob("test_*.py"))`），所以**四个 parametrize 不变**——
    #     这一点和过去几次「新文件 +4」的账不同，不能照抄旧算式。
    # 与 `py -3 -m pytest tests --collect-only -q` 实测 1811 一致。
    #
    # 这次换来的门（逐条写明，只报数字不算）：
    #   · `test_correct_coverage_must_pair_with_coverage`（同类共 7 条）：
    #     只报「覆盖建立」不报「正确覆盖」时判 incomplete。四个口径的假通过率
    #     base 60.3% / cand 52.4% 说明前者的分子里六成答案是错的，单看它会把
    #     「口径修正」读成「质量改进」。
    #   · `test_os_times_would_have_reported_zero_on_this_platform`：
    #     钉死 `os.times()` 在这台机器上对子进程恒返回 0。第一版实现就是用它，
    #     跑出 28.85s 墙钟配 **0.02s CPU**——一个能通过 `_judge_duration()` 的
    #     假数（0.02 < 28.85 不矛盾），比不采更糟。这条门记录「为什么不能改回去」。
    #   · `test_child_cpu_seconds_reads_real_cpu_for_a_burning_child`：
    #     真起一个烧 2 秒 CPU 的子进程，必须读到 > 1.0s，且来源必须是
    #     `proc_stat`（量到被测进程那档）。改成 `return 0.0, "proc_stat"`
    #     → 1 failed。
    #   · `test_cpu_source_is_one_of_three_values`：`cpu_source` 只能取三个值，
    #     拼错的来源名一律落成 `none`。让未知值原样通过 → 1 failed。
    #   · `test_unmeasurable_cpu_stays_none_not_zero` / `test_source_and_number_
    #     must_both_hold` / `test_no_source_means_no_number_either`：来源与数字
    #     必须同时成立，缺一个就都作废。有来源但填 0 → **2 failed**。
    #   · `test_missing_cpu_is_unknown_not_pass`：只有墙钟 → `unknown`。
    #     `duration_verdict` 恒返回 pass → 1 failed。**判据早就在
    #     `provenance._judge_duration()` 里，但入口没接线**——「没接线的判据
    #     等于不存在」，整条门不成立，不是「弱了一点」。
    #   · `test_report_carries_cpu_s_and_seven_provenance_fields`：`--report`
    #     写出的 JSON 同时带 `cpu_s`、`cpu_source` 与七字段。`"timings": {}`
    #     → 1 failed。
    #
    # 反向实测（命令级，结果落在 `data/_tmp_reverse_cpu_gate.json`；本轮起日志路径
    # 用 `tempfile.gettempdir()` 解析——PowerShell 里带引号的 `"%TEMP%"` 不展开，
    # 会在仓库根建一个叫 `%TEMP%` 的字面目录。那次「日志在 %TEMP%」的断言，
    # 路径字面写对了，指向的地方却不是我以为的那个系统临时目录）：
    #   [0] 基线 17 passed 退出码 0
    #   [A] `GetProcessTimes` 分支 `return 0.0, "proc_stat"` → 1 failed 退出码 1
    #   [B] 有来源但数字丢了时填 0 → 2 failed 退出码 1
    #   [C] 未知来源名原样通过 → 1 failed 退出码 1
    #   [D] `"timings": {}` → 1 failed 退出码 1
    #   [E] `duration_verdict` 恒 pass → 1 failed 退出码 1
    #   [Z] 恢复后 17 passed 退出码 0；`.gate0bak` 残留 False
    #
    # 踩到的坑（当场自查，两次）：
    #   1. 第一版实现用 `os.times()` 前后求差，跑出「墙钟 28.85s / CPU 0.02s」。
    #      **我自己在同一个 docstring 里刚写过「只算本进程会得到几乎为 0 的数，
    #      那比不采更糟」，然后照做了。** 探针实测：`os.times()` 四个字段在
    #      Windows 上全是 0（含两个名字专为子进程准备的 `children_*`）；
    #      `GetProcessTimes` 读回 3.0625s，且必须在 `proc.wait()` 之后、句柄
    #      关闭之前读（运行中恒 0.0）。教训：**「我加了这个字段」不等于
    #      「这个字段是真的」**——新采的每个量都要先拿一个已知答案的输入试一遍，
    #      否则它只是把「缺失」换成了「假数」。
    #   2. B 笔第一次**没变红**：测试喂的是 `cpu_s=None, cpu_source="none"`，
    #      而那条路径在两个版本里都返回 `None`（`source == "none"` 分支先
    #      return 了）。「采不到填 0」这个 bug 只在「**有来源、但数字丢了**」
    #      这条更窄的路径上才显形，而我的用例从没走进那条路径——**用例围着
    #      函数的常见返回值写、没围着它的状态空间写**。补
    #      `test_source_and_number_must_both_hold` 后才红。
    # ------------------------------------------------------------------
    # 1811 -> 1830（+19）：`tests/test_p0_item5_drill.py` 此前是未跟踪残留、
    # 从未登记；它自带 15 个用例，进入 `TEST_FILES` 后又让本文件三个按
    # `TEST_FILES` 参数化的门各多一项。登记值以实测为准：
    #   `--collect-only`             1830 tests collected
    #   `--collect-only --ignore=…`  1815 tests collected
    # 两数相减 = 15，正好是该文件自己的用例数。
    # 这次上调不是为了让门变绿：换来的门是 P0 第 5 项那组归档/还原/回滚的
    # 隔离副本演练（15 条），其中最要紧的一条
    # `test_reassemble_needs_the_kept_files` 盯的是「归档可逆」——
    # 少了保留清单，还原出来就是一包没有身份的字节。
    # 1830 -> 1855（+25）：新增 `tests/test_answer_paths.py`，给两条此前**零测试覆盖**
    # 的答案通路（`TITLE_ANSWER_PATH` / `CHAPTER_ANSWER_PATH`）上门。这两个开关各影响
    # 52 / 32 道题，却在 2026-11-05 被写进台账说「已入库」后一直保持 `False`，一次都
    # 没在生产里跑过——本项目把这种形态记作「**没接线的判据等于不存在**」。
    # 登记值以实测为准：
    #   `--collect-only`                     1855 tests collected
    #   `--collect-only --ignore=…answer_paths`  1834 tests collected
    # 两数相减 = 21，正好是该文件自己的用例数；剩下的 +4 是本文件里四个按
    # `TEST_FILES` 参数化的门各多一项（新文件一进目录就自动被扫到，与 `--ignore`
    # 无关，所以这个 4 不是"多收的"，是「多一个被测文件」的必然结果）。
    # 这次上调换来的门：`TestSwitchesAreWired` 两条把开关钉死（改回 `False` 立刻红，
    # 反向实测 17 failed / 21 passed）+ `TestTitleAnswerPath` 9 条 + `TestChapterAnswerPath`
    # 8 条 + `TestPathsDoNotShadowEachOther` 2 条，钉的是「答案只能取自 unit.title /
    # unit.chapter」「引用 `text[start:end] == evidence` 且过 `verify_citations`」
    # 「判据不成立返回 None 落回原路径（不许猜）」。
    # 1855 -> 1880（+25）：新增 `tests/test_vector_union.py`，给「向量通道怎么接」上门。
    # 用户硬约束是「无论如何向量通道都必须存在」，但**怎么接**已被 300 题实测逼到
    # 唯一可行解：所有融合式接法（minmax 加权 / RRF k=1,2 / DBSF / DBSF 尾部补位 /
    # 头部锁定+向量重排余下）全部让 R@20 或 MRR 变差，只有「触发时追加到尾部」
    # 能保证 base 已命中的题名次不变（追加块名次必然 >= 21）。
    # 登记值以实测为准：
    #   `--collect-only`                         1880 tests collected
    #   `--collect-only --ignore=…vector_union`  1859 tests collected
    # 两数相减 = 21，正是该文件自己的用例数；剩下 +4 是本文件四个按 `TEST_FILES`
    # 参数化的门各多一项（新文件一进目录就被 glob 扫到，与 `--ignore` 无关）。
    # 这次上调换来的门：`TestTriggerSignal` 5 条钉触发判据（含「第 20 名分为 0
    # ⇒ ratio=inf ⇒ 不触发」「前 20 名同分 ⇒ 触发」两个反直觉边界）；
    # `TestTriggeredTailUnion` 10 条钉「不触发时一个字节都不许变且绝不调
    # embed_query」「只追加不重排」「追加块 score 必须是 0.0」「sources 的键必须
    # 是 channel 而非 source（写错会在 build_context 的 direct() 上 KeyError）」
    # 「EmbeddingError 必须向上抛不许吞成没有向量」；`TestPipelineWiring` 4 条钉
    # 「build_pipeline 默认 vector_union=None（40 多个历史脚本的读数不许被静默
    # 改掉）」「ask_once 把 union 放在 search 之后、expand 之前，使补进来的块
    # 进入图扩展种子」。
    # 1880 -> 1896（+16）：新增 `tests/test_context_budget.py`，把「上下文预算
    # 6000 改成 12000」这件事从一次性实测变成门。这次改动的依据是两套题：
    # dev 可判 285 题 191/285 (67.02%) -> 225/285 (78.95%)；holdout 可判 195 题
    # 137/195 (70.26%) -> 149/195 (76.41%)（holdout 里 94 题与 gold-v1 重复，
    # clean holdout 只有 144 题、可判 120 题：87/120 (72.50%) -> 97/120 (80.83%)）。
    # 不改 `MAX_CLAIMS`：5 -> 12 是 dev +12 而 holdout 只有 +3，判为过拟合。
    # 登记值以实测为准：
    #   `--collect-only`                              1896 tests collected
    #   `--collect-only --ignore=…test_context_budget.py`  1884 tests collected
    # 两数相减 = 12，正是该文件自己的用例数（11 条断言 + 1 条 `xfail(strict=True)`
    # 的已知缺陷记录）；剩下 +4 是本文件四个按 `TEST_FILES` 参数化的门各多一项。
    # 1880 + 12 + 4 = 1896。
    # 这次上调换来的门：`TestDefaultValueIsPinned` 2 条钉「默认值就是 12000」
    # （改回 6000 立刻红，失败文案直接打印上面那三组实测数字）+「默认值不得小于
    # 历史口径」；`TestDefaultTakesEffectInBuildContext` 4 条钉「默认值真的作用到
    # build_context（24 块 600 字语料上默认装 11 块、显式 6000 装 5 块）」
    # 「`Retriever.budget` 是 `--token-budget` 的入口」「截断必须记账」；
    # `TestKnownDefect` 1 条 `xfail(strict=True)` 钉住预算非单调这个既知缺陷
    # （`build_context` 超预算时是 `continue` 不是 `break`，真语料 237 题里 4 题
    # 出现「加大预算反而丢金块」；修好时 XPASS 变红，逼人更新记录）；
    # `TestServerDoesNotCopyTheNumber` 2 条钉「服务端默认值派生自
    # `retrieve.DEFAULT_BUDGET`」并把「手抄第二个数字」做成静态门——这是本项目
    # 记录在案的「同一判定写两处」事故形态；`TestCliCanReproduceTheOldCaliber`
    # 3 条钉「`scripts/ask.py --token-budget` 存在、默认值派生、`evaluate()` 也收到」
    # （40 多个历史脚本吃默认值，台账里 6000 档读数只有 `--token-budget 6000` 能复现）。
    # 1896 -> 1939 -> 1943（+43，其中 +4 来自判据改严后补的用例）：
    # 新增 `tests/test_judge.py`，把「答对」这件事从各脚本各写一份变成唯一实现
    # `src/arkrag/judge.py`。这一刀不是为了加功能，是因为本轮评测里同一个"答对"
    # 被三种口径各算过一遍、结论方向不同：
    #   1. 「答案里出现 gold 串」——`MAX_CLAIMS` 5 -> 12 让 dev 从 196 涨到 211
    #      (+15)，但换成严判据后只 +0.70pt（p=0.1290，不显著）；多出来的是
    #      "多堆几条断言撞上高频专名"（`隶属` 出现在 102/350 题答案里）。
    #   2. `cross_chunk` 用「答案章节 ⊇ gold」——12000 档比 6000 档高 7 题，
    #      看起来预算有用；但那是"多列章节"堆出来的。实测把语料里全部 494 章
    #      都列出来在超集判据下 38/38 全过，只列 gold 任一章则 subset 38/38 全过
    #      —— 两端退化解都能拿满分，说明单边判据根本不能当判据。
    #   3. 问句自带答案——`single_chunk` 76 题里 46 题的 gold 就写在题干书名号里，
    #      零检索抄标题就能过 46/76（这是"零检索地板"，任何检索侧增益先减掉它）。
    # 判据最终定稿（复议后）：
    #   cross_chunk = gold ⊆ 答案 AND（额外章节逐个查语料正文必须真的提到该实体）
    #                 AND 额外章节数 ≤ gold 章节数
    #     - 只要求集合相等会惩罚正确答案：图登记不全，实测 44/64 题 gold 系统性
    #       漏掉真实提及章节（共 317 章），预算越大错杀越多（12000 档 18 题、
    #       40000 档 28 题）。
    #     - 只要求「超集 + 额外真实」会被普通名词打满：`海洋` 题 gold 3 章而正文
    #       47 章真提及（比值 14.7），`集市` 2→27、`巫术` 15.0、`银行` 11.0。
    #       所以第三条规模上限是必须的，不是保险。
    #     - `chapter_texts` 不传时显式降级成集合相等并写明 `extra_chapters_unverified`，
    #       不许在没查正文时假装做过真实性校验。
    #   multihop = gold 必须与**每一个**问句锚点各自共现（可分散在不同断言里），
    #     且 gold 本身不是锚点。旧判据只要"gold + ≥1 锚点"，实测 c12_12000 通过的
    #     73 题里 57 题只命中 1 个锚点、没有一题体现"经中间实体连接两端"。
    # 登记值以实测为准：
    #   `--collect-only`                          1943 tests collected
    #   `--collect-only --ignore=…test_judge.py`  1900 tests collected
    # 两数相减 = 43，正是该文件自己的用例数；剩下 +4 是本文件四个按 `TEST_FILES`
    # 参数化的门各多一项。1896 + 43 + 4 = 1943。
    # 这次上调换来的门（每条 `test_*_rejects_*` 都对应一次真跑里判错过的案例）：
    # `TestEmptyAndStructural` 5 条钉「空答案 = False（拒答不算答对，哪怕 gold 也空）」
    # 「结构类返回 None 而不是 0 分（必须剔分母，不许当 0 或当 1）」「可判类与结构类
    # 不重叠且并集等于八类全集」「未知类不崩」；`TestSubstringCategories` 8 条钉
    # single_chunk / alias 的子串口径与 `no_gold`；
    # `TestCrossChunkNeedsBoundedRealExtras` 11 条是这次的核心：
    #   `test_accepts_bounded_real_extra_chapters` 钉住"图漏登记时不该判错"
    #     （gold 少 `main_ep16` 但该章正文真的提到实体，额外 1 ≤ gold 2 → 判对）；
    #   `test_rejects_extra_chapter_it_never_mentions` 钉住"编的章节要拒"；
    #   `test_rejects_real_extras_that_outnumber_gold` 单独钉住规模上限——
    #     额外两章都真实提及、只靠"真实"判据挡不住，只能靠上限拒；
    #   `test_rejects_listing_every_chapter` 钉住退化解（全列 494 章）；
    #   `test_without_chapter_texts_it_degrades_to_set_equality` 钉住不许假装校验过；
    #   `test_space_separated_chapters_are_split` 钉住 gold 里那个空格分隔符
    #     （`act33side main_ep14`，第一版漏了 `\s` 导致 38 题全假失败）。
    # `TestDirectRelationNeedsAllAnchorsInOneClaim` 4 条钉「必须是同一条断言同时
    # 含全部锚点 + gold」，反例就是 `q-dr-900010`（问「红」与「罗德岛」，
    # 答案里「温蒂隶属罗德岛」被旧判据判过）。
    # `TestMultihopNeedsGoldLinkedToEveryAnchor` 4 条钉「gold 与每个锚点各自共现」，
    # 反例是 `q-mh-000001`（锚点 [云兽, 变形者集群]，答案里 `云兽` 一个字没有）。
    # `TestCitationsSupport` 6 条是另一件事（引用偏移逐字对上、列出 N 章就要有 N 条
    # 引用），不并入答对判据，否则分不清"答错了"和"引用写错了"。
    # 反向实测（`tmp_mutate_judge.py` 实跑 10 条突变，**10/10 命中**，基线 rc=0、还原后 rc=0）：
    #   去掉「额外必须真实提及」      => test_rejects_extra_chapter_it_never_mentions 红
    #   去掉「额外数 ≤ gold 数」      => test_rejects_real_extras_that_outnumber_gold 红
    #   回到严格集合相等              => test_accepts_bounded_real_extra_chapters 红
    #   direct_relation `all` -> `any` => test_rejects_claim_about_a_different_subject 红
    #   multihop 放宽成任一锚点        => test_rejects_gold_linked_to_only_one_anchor 红
    #   删掉 `if gold in anc`         => test_rejects_gold_that_is_an_anchor 红
    #   章节分隔符去掉空白字符类        => test_space_separated_chapters_are_split 红
    #   结构类 `None` -> `False`       => test_structural_categories_return_none 红
    #   空答案 `False` -> `True`       => test_abstain_never_counts_as_correct 红
    #   没传正文时假装校验过            => test_without_chapter_texts_it_degrades_to_set_equality 红
    # （第一次跑时「去掉额外数上限」那一格漏网，原因是原用例
    #  `test_rejects_listing_every_chapter` 的额外章节里有一章本来就没提到实体，
    #  被"额外必须真实"这条先挡下了 —— 于是补出 `test_rejects_real_extras_that_outnumber_gold`。
    #  这一格就是"一条判据同时被另一个条件遮住"的活标本。）
    #
    # 1943 -> 1957（+14）：`tests/test_max_claims.py` 新增 10 条 + 本文件四个按
    # `TEST_FILES` 参数化的门各多一项（+4）。实测 `--collect-only` = 1957、
    # `--collect-only --ignore=tests/test_max_claims.py` = 1947，相减 10 对得上。
    # 换来什么门：把 `MAX_CLAIMS` 这个**影响 43% 可判题**（dev 285 里 116 题要 >5 条
    # 断言）的参数钉死，且钉了「构造器默认值必须跟着模块常量走」——`TITLE_ANSWER_PATH`
    # 那种「常量改了但实例还用旧值」的没接线事故在这里也拦得住。
    # 另三条是源码注释体检：实测读数（191/198/200、128/132）、拒答伤害、
    # 回滚方法必须留在 `answer.py` 里，删了就没法判断这个数还能不能动。
    # 1957 -> 2017（+60）：新增 `tests/test_bridge.py`（30 条）与
    # `tests/test_quote_locate.py`（22 条），两个文件进入 `TEST_FILES` 后让四个
    # parametrize（本文件各门按 `TEST_FILES` 参数化）各多两项，再 +8。
    # 1957 + 30 + 22 + 8 = 2017，与 `--collect-only` 实测一致。
    # 换来什么门：把两条**把 dev 可判正确率从 69.47% 抬到 80.35%** 的通路钉死——
    # `quote_locate`（题干引文→逐字定位→命中块 title，门控 single_chunk 128/128、
    # 改一个字即 0 命中）与 `bridge`（一跳公共邻居唯一才答、句源必须是真实断言、
    # 候选只许来自本次 ctx、引用偏移逐字可校验）。这两条都是被对抗审查逐轮拷问
    # 才活下来的，删掉任一约束都会让它退回"注水"形态，所以每条否决读数
    # （41.7%、候选内随机 21 > 排序 top1 19、marginal 差 0 题、改字 0/76、
    # 98.7/98.1/96.9% 唯一率、p50=16）都写进了对应源码注释并有机检断言。
    # 突变验证：`tmp_mutate_gates.py` 实跑 13 组突变，**13/13 按预期命中**
    # （其中 1 组是双突变——`s == o` 与 `- set(anc)` 互为备份，单条改都不红）。
    N = 2017

    def test_declared_count_matches_actual(self) -> None:
        """用 pytest 自己数一遍当前收集到的用例数，与本文件登记值比对。"""
        import subprocess
        import sys

        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(TESTS), "--collect-only", "-q"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=180,
        )
        out = proc.stdout + proc.stderr
        match = re.search(r"(\d+) tests? collected", out)
        assert match, f"没能从 pytest 输出里读出用例数：{out[-800:]}"
        actual = int(match.group(1))
        assert actual == self.N, (
            f"测试总数变了：登记 {self.N}，实际 {actual}。"
            f"减少说明有门被删掉了（危险）；增加请更新本文件的 N。"
        )
