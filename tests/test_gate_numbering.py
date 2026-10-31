"""门禁编号唯一性：同一个编号不许被两条门共用。

为什么需要这条门
----------------
2026-10-31 的 identity 接线里，`l6_direction_gates` 新增的邻接方向门被编成了
`L6.4`，而 `l6_gates` 里早有一个 `L6.4 别名倒排覆盖`。两条门编号相同、含义无关，
在合并后的 `reports/gates.json` 里各自出一行：

    L6 | L6.4 别名倒排覆盖（实体名+别名->id）
    L6 | L6.4 邻接方向完整性（出/入分列，反向换逆谓词，两端落位正确）

后果不是"报告多一行"这么轻：所有按编号查门的地方（人肉核对、`grep L6.4`、
交接文档里的"L6.4 已通过"）都会指向错的那一条。`write_report` 按 `stage` 整段
替换、按 `name` 排序，编号重复不会让它报错，所以从流水线角度看完全正常。

这正是本项目反复踩的同一类问题——**同一个标识在两处各写一份**（冻结清单与写
保护清单、来源判定的白名单与黑名单、别名可检索性的两个过滤点）。这次是编号。
纪律不变：标识只能有一处定义，其余地方必须引用它。

判据
----
只统计**真正构造门**的那一处。写法上不是扫所有字符串字面量——首版那样写会把
docstring 开头的 `\"\"\"L4.12 / L4.13：…` 和 note 里提到的编号也算成一条门，
造出三条假重复（`L1.2b`、`L4.12`、`L5.9`），而它们各自的真实门名都只有一条。
误报会让人学会忽略这条门，所以判据必须精确到"门名究竟从哪里来"：

1. 先找出所有**返回 Gate 的函数**（返回标注是 `Gate`），`Gate` 本身也算一个；
2. 只收这些函数的**第一个位置实参**里是字符串字面量的那些。

`gates.py` 里的 `one(name, threshold, hits, note)` 就是这种包装函数，它的
`Gate(name, ...)` 里 name 是变量，所以第 1 步看到的仍是调用点上写死的门名。

`L6.4` 与 `L6.4c` 算两个不同编号，这是既有约定（`L6.4a`/`L0.6a`/`L2.1a` 同理）。
"""

import ast
import collections
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
GATES = ROOT / "src" / "arkrag" / "gates.py"

# 本仓库实际用到的编号形态都要覆盖：
#   L6.4 / L0.6c / L5.7b      —— 常规
#   L1.2b2                     —— 带数字后缀
#   L3.T0 / L3.T17b / L3.T19   —— L3 组用 T 前缀（T 表示 trial 层）
# 首版只写了 `L\d+\.\d+[a-z]?`，把整个 L3 组和 `L1.2b2` 判成"没有编号"。
# 后面必须跟空白、括号或字符串结束，这样 `L6.4.1` 之类不会只匹配到 `L6.4`。
GATE_NUMBER = re.compile(r"^(L\d+\.[0-9A-Za-z]+)(?=[\s（(]|$)")


def _returns_gate(fn: ast.AST) -> bool:
    """函数的返回标注是否是 Gate。"""
    ann = getattr(fn, "returns", None)
    if ann is None:
        return False
    if isinstance(ann, ast.Name):
        return ann.id == "Gate"
    if isinstance(ann, ast.Constant):
        return ann.value == "Gate"
    return False


def _gate_factories(tree: ast.Module) -> set[str]:
    """返回"能构造门"的函数名集合：Gate 本身 + 所有返回标注为 Gate 的函数。"""
    names = {"Gate"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _returns_gate(node):
            names.add(node.name)
    return names


def _gate_names() -> list[tuple[str, int]]:
    """返回 [(门名, 行号)]，只收真正构造门的调用点上的字符串字面量。"""
    tree = ast.parse(GATES.read_text("utf-8"))
    factories = _gate_factories(tree)
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        fname = getattr(fn, "id", None) or getattr(fn, "attr", None)
        if fname not in factories:
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            found.append((first.value, node.lineno))
    return found


def _gate_numbers() -> list[tuple[str, int]]:
    """[(编号, 行号)]。门名开头必须匹配编号形状，否则说明这条门没编号。"""
    out: list[tuple[str, int]] = []
    for name, line in _gate_names():
        m = GATE_NUMBER.match(name)
        if m:
            out.append((m.group(1), line))
    return out


def test_gate_numbers_are_unique():
    """同一个门禁编号只许出现一次。"""
    numbers = _gate_numbers()
    assert numbers, "没有扫到任何门禁编号——扫描判据坏了，不是门禁干净"

    by_number: dict[str, list[int]] = collections.defaultdict(list)
    for number, line in numbers:
        by_number[number].append(line)

    duplicates = {n: lines for n, lines in by_number.items() if len(lines) > 1}
    assert not duplicates, (
        "门禁编号重复："
        + "; ".join(f"{n} 出现在行 {lines}" for n, lines in sorted(duplicates.items()))
        + "。两条无关的门共用一个编号，按编号核对的人会查到错的那一条。"
        " 改号而不是改文档——文档可以照抄旧号，门禁不会。"
    )


def test_every_gate_name_starts_with_a_number():
    """每条门都必须以编号开头，否则报告里无法按编号定位。"""
    unnumbered = [(n, ln) for n, ln in _gate_names() if not GATE_NUMBER.match(n)]
    assert not unnumbered, f"这些门没有编号前缀：{unnumbered}"


def test_scanner_sees_wrapper_built_gates():
    """扫描判据的阳性对照：必须看得见经由 `one(...)` 包装造出来的门。

    `l3_trial_gates` 里绝大多数门是 `one("L3.T1 …", …)` 这样造的，`Gate` 那一层
    拿到的 name 是变量。若扫描只认 `Gate(...)` 字面量，这一整片门会漏掉——
    而漏掉的正是最容易撞号的地方（L3 组有 T0/T1/T2/…/T19 一大串）。
    """
    numbers = {n for n, _ in _gate_numbers()}
    for expected in ("L3.T0", "L3.T16", "L3.T19"):
        assert expected in numbers, f"扫描没看见 L3 组的 {expected}，包装函数那层漏了"


def test_scanner_sees_known_gates():
    """扫描判据的阳性对照：各组各取若干条，确认真的扫到了。"""
    numbers = {n for n, _ in _gate_numbers()}
    for expected in (
        "L0.1", "L0.6c", "L1.2b", "L1.8b", "L2.1c", "L2.3",
        "L4.12", "L4.13", "L5.9", "L5.10", "L5.11",
        "L6.4", "L6.4c", "L6.8", "L6.8a",
    ):
        assert expected in numbers, f"扫描没看见 {expected}，判据或门禁改名了，请同步本用例"


def test_scanner_ignores_prose_and_other_numbers():
    """扫描判据的阴性对照：docstring 与 note 里提到的编号不许被当成门。

    首版就是栽在这里：`\"\"\"L4.12 / L4.13：…` 与 note 里的 `L1.2b` 都被算成
    一条门，于是 `L1.2b`、`L4.12`、`L5.9` 三条真的各自只出现一次的门被误报重复。
    """
    for sample in ("ark-rag-3", "v2", "ev2", "sha256", "L6", "L.4", "6.4", "L6.4.1"):
        assert GATE_NUMBER.match(sample) is None, f"{sample!r} 被误判成门禁编号"
    assert GATE_NUMBER.match("L6.4 别名倒排覆盖").group(1) == "L6.4"
    assert GATE_NUMBER.match("L6.4c 待审别名").group(1) == "L6.4c"

    # 真实反例：这两条字符串在 gates.py 里存在，但不在构造门的调用点上。
    source = GATES.read_text("utf-8")
    assert '"""L4.12 / L4.13' in source, "反例已被改动，请换一条真实的反例"
    prose_hits = {(n, ln) for n, ln in _gate_names() if "L4.12 / L4.13" in n}
    assert not prose_hits, f"docstring 被当成了门名：{prose_hits}"
