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
            if "raise " in seg or "assert " in seg
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
            if "assert " in seg or "raise " in seg:
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
    N = 1251

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
