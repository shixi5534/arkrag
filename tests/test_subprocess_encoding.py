# -*- coding: utf-8 -*-
"""子进程管道编码约定（`arkrag.subproc`）的门。

## 为什么这些用例必须真起子进程

断言字符串常量（`assert "监听" in STUB`）什么都证明不了：常量里当然有「监听」，
出事的是它经过管道之后变成了什么字节。2026-11-04 Gate 0 的事故就是这么来的——
脚本打印了，父进程读到的却是 `\\ufffd`，于是「按中文匹配子进程输出」的门恒不命中，
报出来的红方向是假的（它报「脚本没打印」，而脚本其实打印了）。

所以这里的形状是：**真的 Popen 一个 Python 子进程，让它 print 中文，再按 UTF-8 解码
父进程这一侧**。一正一反两条：不设 env 就读不出来（机制证明），设了就读得出来。

## 白名单为什么按函数名而不是行号

静态扫描那一条要求「每个 subprocess 调用要么显式传 `env=`，要么命中一条带理由的
白名单」。命中键用**相对路径 + 所在函数名**，不用行号：行号会随任何无关编辑漂移，
一漂移门就会红在一次与它毫无关系的改动上——那正是「门变成噪音、然后被人顺手放宽」
的起点。用函数名定位同样要求人工书写每一条、每一条都出现在 diff 里，但没有那种
假红。理由必须写清「为什么这个调用不需要 UTF-8 管道」，不许写「历史遗留」。
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

from arkrag import subproc  # noqa: E402

LISTEN_LINE = "[i] 监听 http://127.0.0.1:12345"
LISTEN_RE = re.compile(r"监听 http://[^:\s]+:(\d+)")


def _demo_child(monkeypatch, env):
    """真起一个子进程，让它 print 一行中文监听地址；按 UTF-8 解码父进程这侧。

    `env` 为 None 时沿用当前进程环境（不做任何加工），这就是事故发生时的形状。
    """
    argv = [sys.executable, "-u", "-c", f"print({LISTEN_LINE!r})"]
    proc = subprocess.run(
        argv, capture_output=True, text=True,
        encoding=subproc.PIPE_ENCODING, errors="replace",
        env=env,
        timeout=60,
    )
    return proc


# ---------------------------------------------------------------- 约定本身

def test_child_env_forces_pipe_encoding():
    """从外部环境复制、覆盖 `PYTHONIOENCODING`，**不清空**其余变量。"""
    base = {"PATH": "C:/somewhere/bin", "ARKRAG_HOME": "D:/ark", "PYTHONIOENCODING": "gbk"}
    env = subproc.child_env(base)
    assert env[subproc.ENV_PIPE_ENCODING] == subproc.PIPE_ENCODING
    # 外部给的 gbk 必须被覆盖，不是 setdefault：保留它就会重新造出事故组合。
    assert env[subproc.ENV_PIPE_ENCODING] != "gbk"
    # 其余变量原样保留，否则子进程找不到解释器之外的一切。
    assert env["PATH"] == "C:/somewhere/bin"
    assert env["ARKRAG_HOME"] == "D:/ark"
    # 传进来的 dict 不许被就地改写。
    assert base["PYTHONIOENCODING"] == "gbk"


def test_child_env_leaves_pythonutf8_alone():
    """`PYTHONUTF8` 不许被顺手打开。

    UTF-8 模式不只改管道编码，还改 `open()` 默认编码与文件系统路径编解码，
    超出「父进程按 UTF-8 解码子进程输出」这一件事的范围。这是判断后不加。
    """
    assert "PYTHONUTF8" not in subproc.child_env({"PATH": "x"})
    kept = subproc.child_env({"PYTHONUTF8": "0"})
    assert kept["PYTHONUTF8"] == "0"


def test_child_env_default_reads_real_environ():
    env = subproc.child_env()
    assert env[subproc.ENV_PIPE_ENCODING] == "utf-8"
    # 复制自 os.environ：父进程里有的变量（PATH 一定有）必须在结果里。
    if os.environ.get("PATH"):
        assert env["PATH"] == os.environ["PATH"]


# ---------------------------------------------------------------- 机制：一正一反

def test_gbk_pipe_is_unreadable_under_utf8_decoding(monkeypatch):
    """**机制证明**：子进程按 gbk 写管道、父进程按 utf-8 读 → 中文变成替换字符。

    这是「不设 env 就会坏」的那一面。用显式 `PYTHONIOENCODING=gbk` 强制出这个组合，
    而不是依赖跑测机器的默认代码页——依赖默认代码页的用例在别人机器上会静默变成
    永远绿，那比没有这条用例更糟。
    """
    poisoned = dict(os.environ)
    poisoned[subproc.ENV_PIPE_ENCODING] = "gbk"
    proc = _demo_child(monkeypatch, poisoned)

    assert proc.returncode == 0, proc.stderr
    assert "\ufffd" in proc.stdout, repr(proc.stdout)
    assert "监听" not in proc.stdout
    assert LISTEN_RE.search(proc.stdout) is None


def test_utf8_pipe_is_readable_under_utf8_decoding(monkeypatch):
    """**正面对照**：走 `child_env()` 之后同一行中文必须能解析出端口。"""
    poisoned = dict(os.environ)
    poisoned[subproc.ENV_PIPE_ENCODING] = "gbk"      # 父进程继承来的坏值
    proc = _demo_child(monkeypatch, subproc.child_env(poisoned))

    assert proc.returncode == 0, proc.stderr
    assert "\ufffd" not in proc.stdout, repr(proc.stdout)
    match = LISTEN_RE.search(proc.stdout)
    assert match is not None, repr(proc.stdout)
    assert int(match.group(1)) == 12345


def test_unencodable_char_crashes_child_without_env(monkeypatch):
    """第二类危害：cp936 编不出的字符会让子进程**自己崩**，退出码变成 1。

    这种失败最容易被误读成「业务代码出错」——父进程只看到非零退出码，
    和一个真实的业务失败没法区分。
    """
    poisoned = dict(os.environ)
    poisoned[subproc.ENV_PIPE_ENCODING] = "gbk"
    script = "print('\\u26a0 中文')"          # ⚠ 在 cp936 里编不出来
    crashed = subprocess.run(
        [sys.executable, "-u", "-c", script], capture_output=True, text=True,
        encoding=subproc.PIPE_ENCODING, errors="replace", env=poisoned, timeout=60,
    )
    assert crashed.returncode != 0
    assert "UnicodeEncodeError" in (crashed.stderr or "")

    fixed = subprocess.run(
        [sys.executable, "-u", "-c", script], capture_output=True, text=True,
        encoding=subproc.PIPE_ENCODING, errors="replace",
        env=subproc.child_env(poisoned), timeout=60,
    )
    assert fixed.returncode == 0, fixed.stderr
    assert "中文" in fixed.stdout


# ---------------------------------------------------------------- 每个点位都传了 env

def _fake_completed(stdout="", stderr="", returncode=0):
    class _Done:
        pass

    done = _Done()
    done.stdout = stdout
    done.stderr = stderr
    done.returncode = returncode
    done.args = []
    return done


def _capture_env(monkeypatch, module, fake):
    """把 `module` 里看到的 `subprocess.run` 换成假的，返回被喂进去的 kwargs 盒子。"""
    box: dict = {}

    def fake_run(cmd, **kwargs):
        box["cmd"] = cmd
        box.update(kwargs)
        return fake

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    return box


def _load_script(name: str, module_name: str):
    import importlib.util

    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def k1():
    return _load_script("k1_checks", "k1_checks_encoding_under_test")


@pytest.fixture(scope="module")
def k1_race():
    return _load_script("k1_race_check", "k1_race_encoding_under_test")


@pytest.fixture(scope="module")
def k3():
    return _load_script("k3_thresholds", "k3_thresholds_encoding_under_test")


def test_k1_run_pytest_passes_pipe_encoding(monkeypatch, k1):
    """`k1_checks._run_pytest()` 会解析 pytest 的中文摘要行 → 必须传 env。

    去掉那个 `env=`，这条用例红：kwargs 里没有 `PYTHONIOENCODING`。
    """
    box = _capture_env(monkeypatch, k1, _fake_completed(stdout="1 passed\n"))
    k1._run_pytest(("tests/test_subprocess_encoding.py",))
    assert box["env"][subproc.ENV_PIPE_ENCODING] == "utf-8"
    assert box["encoding"] == subproc.PIPE_ENCODING


def test_k1_baseline_verify_passes_pipe_encoding(monkeypatch, k1):
    """`k1_checks` 里跑 `snapshot_baseline.py --verify` 那处同样解析中文摘要行。

    这一项只关心「子进程怎么被启动」，其余判据的输入给最小可用值：两个可服务世代
    （候选 + 回滚目标各一个）加一个指向其中之一的发布指针。
    """
    box = _capture_env(monkeypatch, k1, _fake_completed(stdout="产物 11 项全部一致\n"))
    gen = {"generation_id": "g-x", "status": k1.release.READY, "problems": [], "bytes": 1}
    ctx = {"generations": [gen, dict(gen)], "current": {"generation_id": "g-x"}}
    k1.check_old_artifacts_backed_up(ctx)
    assert box["env"][subproc.ENV_PIPE_ENCODING] == "utf-8"
    assert box["encoding"] == subproc.PIPE_ENCODING


def test_k1_race_check_passes_pipe_encoding(monkeypatch, k1_race):
    """`k1_race_check.one()` 里的注释早在说「必须显式 utf-8」，但只说了读的那一半。"""
    box = _capture_env(monkeypatch, k1_race, _fake_completed(stdout="[ok] 回滚命令已实测\n"))
    index, returncode, last = k1_race.one(0)
    assert (index, returncode) == (0, 0)
    assert last == "[ok] 回滚命令已实测"
    assert box["env"][subproc.ENV_PIPE_ENCODING] == "utf-8"


def test_k3_read_unexplained_passes_pipe_encoding(monkeypatch, k3):
    """`k3_thresholds.read_unexplained()` 按中文正则取数 → 必须传 env。

    这里以前**碰巧**能跑，因为 `k3_attribution.py` 自己 reconfigure 了 stdout；
    那是隐式契约，删掉它就变成 K2 那种「红方向为假」的门。
    """
    line = "未解释合计（边 + 断言）：0\n"
    box = _capture_env(monkeypatch, k3, _fake_completed(stdout=line))
    assert k3.read_unexplained() == 0
    assert box["env"][subproc.ENV_PIPE_ENCODING] == "utf-8"
    assert box["encoding"] == subproc.PIPE_ENCODING


def test_k3_read_unexplained_still_hard_errors_without_the_line(monkeypatch, k3):
    """加 env 不许顺手把「解析不到就硬错」放宽成「当 0」。"""
    _capture_env(monkeypatch, k3, _fake_completed(stdout="完全无关的输出\n"))
    with pytest.raises(k3.GateError) as exc:
        k3.read_unexplained()
    assert "未解释合计" in str(exc.value)


# ---------------------------------------------------------------- 静态扫描（带白名单）

_SUBPROCESS_CALLEES = {"run", "Popen", "call", "check_output", "check_call"}

#: 白名单：`(相对路径, 所在函数名)` → 为什么这个调用**不需要** UTF-8 管道。
#: 只能人工添加；每加一条都会出现在 diff 里。理由必须说明「输出里没有中文匹配」。
ENV_WHITELIST: dict[tuple[str, str], str] = {
    ("src/arkrag/release.py", "git_revision"): (
        "只跑 `git rev-parse HEAD`，输出是 40 位十六进制 ASCII；"
        "没有中文、也不参与任何中文匹配。"
    ),
    ("tests/test_portable_paths.py", "_git_tracked_data_files"): (
        "只跑 `git ls-files`，用于取受版本管理的路径清单；"
        "路径里即便有非 ASCII 也不参与匹配，只当作集合元素比较。"
    ),
    ("tests/test_suite_hygiene.py", "test_declared_count_matches_actual"): (
        "只跑 `pytest --collect-only -q` 读 `(\\d+) tests? collected` 这个 ASCII 数字；"
        "是元测试，被测对象就是收集数量本身。"
    ),
}


def _scan_sources(*patterns: str) -> list[tuple[str, int, str, tuple[str, ...]]]:
    """扫出所有 `subprocess.run/Popen/...` 调用。

    返回 (相对路径, 行号, 函数名, 该调用**自己的**关键字参数名元组)。

    关键字参数名取自 `node.keywords`，不取源码行——多行调用里 `env=` 往往不在
    第一行，按行看会把「已经传了 env」的调用误判成没传。
    函数名取**最内层**包含该调用的 `def`，模块级调用记为 `<module>`。
    """
    found: list[tuple[str, int, str, tuple[str, ...]]] = []
    for pattern in patterns:
        for path in sorted(ROOT.glob(pattern)):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            spans = [
                (node.lineno, node.end_lineno or node.lineno, node.name)
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not (isinstance(func, ast.Attribute)
                        and isinstance(func.value, ast.Name)
                        and func.value.id == "subprocess"
                        and func.attr in _SUBPROCESS_CALLEES):
                    continue
                owners = [(start, name) for start, end, name in spans
                          if start <= node.lineno <= end]
                owner = max(owners, key=lambda item: item[0])[1] if owners else "<module>"
                rel = path.relative_to(ROOT).as_posix()
                kwargs = tuple(k.arg for k in node.keywords if k.arg)
                found.append((rel, node.lineno, owner, kwargs))
    return found


def _missing_env(sites) -> list[str]:
    """没有 `env=` 且不在白名单里的调用。纯函数，方便单独证明它会红。"""
    offenders: list[str] = []
    for rel, lineno, owner, kwargs in sites:
        if "env" in kwargs:
            continue
        if (rel, owner) in ENV_WHITELIST:
            continue
        shown = ", ".join(kwargs) or "（无关键字参数）"
        offenders.append(f"{rel}:{lineno} ({owner}) 关键字参数：{shown}")
    return offenders


def test_subprocess_calls_declare_pipe_encoding():
    """凡是起子进程的地方，要么显式传 `env=`，要么在白名单里且写明理由。

    这条门的方向：**新增一个不传 env 的子进程调用 → 红**，直到有人把它修好或
    写一条白名单并解释清楚。它防的是 Gate 0 那类事故复发。
    """
    sites = _scan_sources("scripts/*.py", "src/arkrag/*.py", "tests/*.py")
    assert sites, "扫描不到任何 subprocess 调用，说明扫描逻辑自己坏了"
    offenders = _missing_env(sites)
    assert offenders == [], (
        "这些子进程调用没有显式传 env=，而父进程可能按 UTF-8 解码它们的输出：\n  "
        + "\n  ".join(offenders)
        + "\n修法：env=arkrag.subproc.child_env()（见 src/arkrag/subproc.py）。"
        + "\n若确认输出里没有中文匹配，往 tests/test_subprocess_encoding.py 的 "
        + "ENV_WHITELIST 里加一条并写清理由。"
    )


def test_missing_env_detector_flags_a_bare_call():
    """证明上面那道门**真的会红**，不是恒绿。

    喂一个人造调用进去，检测器必须把它抓出来；再给它 `env=` 或白名单，必须放行。
    """
    bare = [("scripts/nowhere.py", 42, "boom", ("encoding", "errors"))]
    offenders = _missing_env(bare)
    assert offenders and "scripts/nowhere.py:42" in offenders[0]

    fixed = [("scripts/nowhere.py", 42, "boom", ("encoding", "errors", "env"))]
    assert _missing_env(fixed) == []

    whitelisted = [("src/arkrag/release.py", 42, "git_revision", ("text",))]
    assert _missing_env(whitelisted) == []


def test_whitelist_entries_are_distinct_and_carry_reasons():
    """白名单每条都要人话理由，且不许同键重复——不然「写一条放过它」会变成习惯。"""
    assert ENV_WHITELIST, "白名单空着也可以，但这条断言防的是有人把整张表删掉"
    for (rel, owner), reason in ENV_WHITELIST.items():
        assert rel and owner, (rel, owner)
        assert len(reason.strip()) >= 10, f"{rel}:{owner} 的理由太短，等于没写"
        assert reason.strip() not in {"历史遗留", "暂时", "TODO", "待补"}, (
            f"{rel}:{owner} 的理由是占位符，等于没有理由"
        )
    # 键是 (路径, 函数名)，dict 天然去重；这条断言防的是以后改成 list 时退化。
    assert len(set(ENV_WHITELIST)) == len(ENV_WHITELIST)


def test_whitelist_points_at_real_calls():
    """白名单不许留下已经消失的条目——死条目会让门悄悄变宽。"""
    sites = _scan_sources("scripts/*.py", "src/arkrag/*.py", "tests/*.py")
    live = {(rel, owner) for rel, _lineno, owner, _kwargs in sites}
    stale = sorted(set(ENV_WHITELIST) - live)
    assert stale == [], (
        "白名单里这些条目对应的子进程调用已经不存在了，请删掉：\n  "
        + "\n  ".join(f"{rel}:{owner}" for rel, owner in stale)
    )


# ---------------------------------------------------------------- 控制台那一侧（同一个病）

#: 本机出货目标的控制台编码。用**显式** cp936 而不是 `locale.getpreferredencoding()`：
#: 后者会让这条门随跑测机器的控制台而变——在 UTF-8 控制台上它会静默变成永远绿，
#: 而实际情况是「有人在 GBK 控制台上跑这个脚本就会崩」。写死 cp936 才可复现。
CONSOLE_CODEC = "cp936"

#: 白名单：`(相对路径, 所在函数名)` → 为什么这个 print **不会**炸控制台。
CONSOLE_WHITELIST: dict[tuple[str, str], str] = {}


def _unencodable_in(text: str) -> list[str]:
    """`text` 里有哪些字符在 `CONSOLE_CODEC` 下编不出来。"""
    bad: list[str] = []
    for ch in dict.fromkeys(text):
        if ord(ch) <= 127:
            continue
        try:
            ch.encode(CONSOLE_CODEC)
        except UnicodeEncodeError:
            bad.append(f"U+{ord(ch):04X}")
    return bad


def _scan_console_writes(*patterns: str):
    """扫出所有「往 stdout 写字符串字面量」的调用。

    返回 (相对路径, 行号, 函数名, 编不出的字符列表)。取整段调用源码而不是只取
    `node.args`：隐式字符串拼接（`f"…"  f"…"`）与 f-string 都要算进去，
    而它们都在 `ast.get_source_segment` 的覆盖范围内。
    """
    found = []
    for pattern in patterns:
        for path in sorted(ROOT.glob(pattern)):
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text)
            spans = [
                (node.lineno, node.end_lineno or node.lineno, node.name)
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                is_print = isinstance(func, ast.Name) and func.id == "print"
                is_stdout_write = (
                    isinstance(func, ast.Attribute) and func.attr == "write"
                    and isinstance(func.value, ast.Attribute) and func.value.attr == "stdout"
                )
                if not (is_print or is_stdout_write):
                    continue
                segment = ast.get_source_segment(text, node) or ""
                bad = _unencodable_in(segment)
                if not bad:
                    continue
                owners = [(start, name) for start, end, name in spans
                          if start <= node.lineno <= end]
                owner = max(owners, key=lambda item: item[0])[1] if owners else "<module>"
                found.append((path.relative_to(ROOT).as_posix(), node.lineno, owner, bad))
    return found


def _reconfigures_stdout(path: Path) -> bool:
    """这个文件里是否**真的调用了** `sys.stdout.reconfigure(...)`。

    必须走 AST，不能 `"reconfigure" in source`：注释里提一句「reconfigure 之后 rc=0」
    就会让字符串判断恒为真，门变成假的绿。这条判断 2026-11-04 首次写时就踩过——
    删掉真正的 reconfigure 那三行之后，注释里的那个词把门撑住了，反向实测 19 passed。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "reconfigure"):
            continue
        inner = func.value
        if (isinstance(inner, ast.Attribute) and inner.attr == "stdout"
                and isinstance(inner.value, ast.Name) and inner.value.id == "sys"):
            return True
    return False


def _unguarded_console_writes(writes, reconfigured: dict[str, bool]) -> list[str]:
    """会崩控制台、又没做 stdout 编码保护、且不在白名单里的写点。纯函数。"""
    offenders: list[str] = []
    for rel, lineno, owner, bad in writes:
        if reconfigured.get(rel):
            continue
        if (rel, owner) in CONSOLE_WHITELIST:
            continue
        offenders.append(f"{rel}:{lineno} ({owner}) 含 {', '.join(bad)}")
    return offenders


def test_scripts_never_print_chars_the_console_cannot_encode():
    """第四道门：**脚本自己**往 stdout 写 cp936 编不出的字符时必须先 reconfigure。

    这和 K2 是同一个病的另一侧。K2 那边是「父进程按 UTF-8 读、子进程按 GBK 写」，
    这边是「子进程往 GBK 控制台写 UTF-8-only 的字符」——后者更直接：`print` 自己
    抛 `UnicodeEncodeError`，进程以 1 退出，与一个真实的业务失败**无法区分**。

    2026-11-04 实测（Gate 0）：`scripts/eval_fusion.py:297` 那行含 U+2212（减号 −），
    在 cp936 控制台下 `rc=1`；加上 reconfigure 后 `rc=0`。修之前的报错是
    `UnicodeEncodeError: 'gbk' codec can't encode character '\\u2212'`。
    """
    reconfigured = {
        path.relative_to(ROOT).as_posix(): _reconfigures_stdout(path)
        for path in list(ROOT.glob("scripts/*.py")) + list(ROOT.glob("src/arkrag/*.py"))
    }
    writes = _scan_console_writes("scripts/*.py", "src/arkrag/*.py")
    offenders = _unguarded_console_writes(writes, reconfigured)
    assert offenders == [], (
        f"这些 print 会往 {CONSOLE_CODEC} 控制台写它编不出的字符，进程会以 1 退出：\n  "
        + "\n  ".join(offenders)
        + "\n修法：在脚本 import 之后加\n"
        + '    if hasattr(sys.stdout, "reconfigure"):\n'
        + '        sys.stdout.reconfigure(encoding="utf-8")\n'
        + "若确认那个字符根本不会被打印出来，往本文件的 CONSOLE_WHITELIST 加一条并写清理由。"
    )


def test_reconfigure_detector_ignores_comments(tmp_path):
    """**这条防的是门自己变成假绿**。

    用「注释里提到 reconfigure」的写法喂进去，检测器必须说「没有」，不然
    注释就能替真代码把门撑住。这是本人写这道门时真实踩到的坑。
    """
    faked = tmp_path / "fake.py"
    faked.write_text(
        "# 这里提到 reconfigure 但并没有真的调用它\n"
        "print('−')\n",
        encoding="utf-8",
    )
    assert _reconfigures_stdout(faked) is False

    guarded = tmp_path / "guarded.py"
    guarded.write_text(
        'import sys\n'
        'if hasattr(sys.stdout, "reconfigure"):\n'
        '    sys.stdout.reconfigure(encoding="utf-8")\n'
        "print('−')\n",
        encoding="utf-8",
    )
    assert _reconfigures_stdout(guarded) is True


def test_console_gate_flags_a_bare_print():
    """证明上面那道门会红：人造写点必须被抓出来；加了保护或白名单必须放行。"""
    writes = [("scripts/nowhere.py", 7, "boom", ["U+2212"])]
    assert _unguarded_console_writes(writes, {"scripts/nowhere.py": False})
    assert _unguarded_console_writes(writes, {"scripts/nowhere.py": True}) == []
    assert _unencodable_in("普通中文") == []
    assert _unencodable_in("Δ 和 − 与 ⚠") == ["U+2212", "U+26A0"]


def test_console_whitelist_entries_carry_reasons():
    """白名单同样只能人工添加，且理由不能是占位符。"""
    for (rel, owner), reason in CONSOLE_WHITELIST.items():
        assert rel and owner, (rel, owner)
        assert len(reason.strip()) >= 10, f"{rel}:{owner} 的理由太短，等于没写"


def test_eval_fusion_protects_its_stdout():
    """点名那处实测会崩的写法，防止有人把 reconfigure 顺手删掉。

    AST 门已经覆盖它了；这条多一层的原因很直接：这是**唯一**一处实测到过
    `rc=1` 的写法，值得有一个按名字指着的用例，红的时候不用读扫描报告。
    """
    source = (ROOT / "scripts" / "eval_fusion.py").read_text(encoding="utf-8")
    assert _reconfigures_stdout(ROOT / "scripts" / "eval_fusion.py"), (
        "eval_fusion.py 会打印 U+2212，删掉 stdout reconfigure 后它在 cp936 控制台下 rc=1。"
    )
    assert _unencodable_in(source), "这行断言防的是「字符被悄悄换成 ASCII 连字符」——那就绕过门了"
