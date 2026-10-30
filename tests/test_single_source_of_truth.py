"""判定只能有一处事实源（静态防回归）。

## 为什么需要这个文件

这个项目里同一类缺陷已经出现过三次，机理完全相同：

1. **冻结清单 vs 写保护清单**各写一份 —— Manifest 里 11 项，`legacy.PROTECTED_FILES`
   只有 4 项，于是重跑 L1/L2 直接把冻结产物覆盖了；
2. **`corpus_meta.json` 副本 vs `source_verification.json` 权威**各算一次 ——
   副本缺 `source_status` 字段被当成"没验证"，实际语料是 `verified_zip`；
3. **`release.py` vs `source.py`** 对"什么算已验证"各有一套判据 ——
   发布门只拒绝字面量 `"source_unverified"`，于是 `""`、`"VERIFIED_ZIP"`、
   `"verified_zip "`（尾空格）、`"bogus"` 全部绕过。

**共同机理：同一个判定在两处各写一份，只要有一处较宽松，整套检查就等于那处最松的。**

单元测试抓不到这类回归：如果有人把 `source.VERIFIED_ZIP` 改回手抄的
`"verified_zip"`，取值恰好相等，所有行为测试照样通过。所以这里做**静态检查**。
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "arkrag"
SCRIPTS = ROOT / "scripts"

# 来源状态只有 source.py 有权定义字面量，其余地方必须引用常量。
STATUS_LITERALS = ("source_unverified", "verified_zip", "verified_tree")


def _python_files(*roots: Path) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        out.extend(sorted(root.glob("*.py")))
    return out


def _code_part(line: str) -> str:
    """去掉行内注释，只留代码部分。

    检查"有没有手抄字面量"必须看代码，不能看注释——注释里出现
    `"source_unverified"` 是在解释历史缺陷，那是正常的、也是必要的。
    """
    # 本仓库注释里没有裸 # 跟在字符串内的怪例，简单切分足够，
    # 且这里宁可**少判**（漏掉一个注释）也不能误判注释为违规。
    return line.split("#", 1)[0]


class TestSourceStatusHasOneDefinition:
    def test_only_source_module_declares_status_literals(self):
        offenders: list[str] = []
        for path in _python_files(SRC, SCRIPTS):
            if path.name == "source.py":
                continue
            for lineno, line in enumerate(path.read_text("utf-8").splitlines(), 1):
                code = _code_part(line)
                for lit in STATUS_LITERALS:
                    if f'"{lit}"' not in code and f"'{lit}'" not in code:
                        continue
                    # 允许的唯一情形：把字面量当**数据值**写给日志/事件
                    # （例如 run.finish(reason="source_unverified")），
                    # 这类不是"判定"，而是记录原因文本。
                    if re.search(rf'(reason|note|message)\s*=\s*["\']{lit}["\']', code):
                        continue
                    offenders.append(f"{path.name}:{lineno}: {line.strip()}")
        assert not offenders, (
            "这些位置手抄了来源状态字面量，应改为引用 source.VERIFIED_ZIP / "
            "source.VERIFIED_TREE / source.UNVERIFIED：\n" + "\n".join(offenders)
        )

    def test_status_defaults_use_constant(self):
        """`source_status` 的默认值必须来自 `source.UNVERIFIED`。

        默认值是最容易手抄的地方，而且它决定了"没写等于什么"——写错方向就是
        把未知当成已验证。
        """
        text = (SRC / "release.py").read_text("utf-8")
        assert 'source_status: str = source.UNVERIFIED' in text
        assert 'source_status: str = "source_unverified"' not in text

    def test_release_gate_uses_whitelist_not_literal(self):
        """发布门必须是白名单，不能只拒绝一个特定拼写。

        只拒绝 `"source_unverified"` 的写法等于没有门：把状态写成 `""`、
        `"VERIFIED_ZIP"`、`"verified_zip "` 这些同样**不是已验证**的取值，
        全都绕得过去。实测对照见 `tests/test_release.py` 的参数化用例。
        """
        text = (SRC / "release.py").read_text("utf-8")
        assert "not in (source.VERIFIED_ZIP, source.VERIFIED_TREE)" in text
        assert '== "source_unverified"' not in text

    def test_gates_use_constant_not_literal(self):
        text = (SRC / "gates.py").read_text("utf-8")
        assert "source.UNVERIFIED" in text
        assert "status in (source.VERIFIED_ZIP, source.VERIFIED_TREE)" in text


class TestVerificationGateIsWhitelistNotBlacklist:
    """来源门必须是白名单，不能是「只拦未验证这一个值」的黑名单。

    ## 这两种写法看起来差不多，行为差很远

    ```python
    if status != source.UNVERIFIED:      # 黑名单：只拦一个值
    if status in (VERIFIED_ZIP, VERIFIED_TREE):  # 白名单：只放两个值
    ```

    黑名单的问题不是"错在某个已知输入"，而是**它对未知输入默认放行**。
    实测代价：`release.py` 用黑名单时，`""`、`"VERIFIED_ZIP"`、
    `"verified_zip "`（尾空格）、`"bogus"` 全部通过发布门；
    `run_l0.py` 同样用黑名单，于是来源未验证时退出码仍可能是 0。

    白名单对未知输入默认拒绝，这才是"身份未知不得冒充已验证"该有的方向。

    ## 为什么允许 `source.py` 内部用 ==

    `source.verify()` 里有一处 `if report["status"] == UNVERIFIED and not problems:`
    ——那是在**补一条解释性 problem**，不是门禁判据，方向是对的（未验证就该有说明）。
    本测试只检查门禁位置，不做全仓一刀切，否则会误判这种合法用法。
    """

    # 这些文件里的来源判据直接决定「能不能发布 / 退出码是 0 还是非 0」
    GATE_SITES = ("release.py", "gates.py")

    def test_no_blacklist_comparison_in_gate_modules(self):
        offenders: list[str] = []
        for name in self.GATE_SITES:
            path = SRC / name
            for lineno, line in enumerate(path.read_text("utf-8").splitlines(), 1):
                code = _code_part(line)
                if re.search(r"(==|!=)\s*source\.UNVERIFIED", code):
                    offenders.append(f"{name}:{lineno}: {line.strip()}")
        assert not offenders, (
            "来源门用了黑名单比较（只拦 UNVERIFIED 一个值，对未知值默认放行）。"
            "请改为白名单 `status in (source.VERIFIED_ZIP, source.VERIFIED_TREE)`：\n"
            + "\n".join(offenders)
        )

    def test_run_l0_exit_code_is_whitelist_based(self):
        """L0 的退出码必须由"是否已验证"决定，不能由"是否恰好等于未验证"决定。"""
        text = (SCRIPTS / "run_l0.py").read_text("utf-8")
        code = "\n".join(_code_part(line) for line in text.splitlines())
        assert "== source.UNVERIFIED" not in code, (
            "run_l0.py 的退出码判据仍是黑名单（只拦 UNVERIFIED 一个值）"
        )
        assert "in (source.VERIFIED_ZIP, source.VERIFIED_TREE)" in code


class TestFrozenProtectionHasOneSource:
    def test_protected_paths_derive_from_baseline_manifest(self):
        """写保护清单必须从冻结 Manifest 派生，不能另写一份。

        这是事故 1 的直接修法：`PROTECTED_FILES` 只有 4 项而 Manifest 有 11 项，
        两份清单必然漂移，漂移的部分就是没人保护的部分。
        """
        text = (SRC / "legacy.py").read_text("utf-8")
        assert "def baseline_paths(" in text
        assert "def protected_dirs(" in text or "def baseline_dirs(" in text
        # is_protected 必须走派生清单
        assert "protected_paths()" in text


class TestGateNeverSwallowsFailures:
    def test_no_stage_function_returns_always_pass(self):
        """门禁函数不许出现"恒 PASS"的写法。

        计划原则 5：所有门禁必须能失败。一个恒 PASS 的门比没有门更糟——
        它会让人以为这件事被检查过了。
        """
        text = (SRC / "gates.py").read_text("utf-8")
        # 一条门的结果只能由条件决定；直接写死 "PASS" 而不带条件就是恒 PASS
        suspicious = re.findall(r'Gate\([^)]*?"PASS"\s*,\s*\)', text, re.S)
        assert not suspicious, f"发现恒 PASS 的门：{suspicious[:3]}"
