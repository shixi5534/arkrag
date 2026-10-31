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

    N = 425

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
