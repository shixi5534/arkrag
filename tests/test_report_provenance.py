# -*- coding: utf-8 -*-
"""门报告可追溯性的门：七字段、三态、`unknown` 不算通过。

## 为什么这是一道门，而不是一份文档

P0 之前 `data/` 下九份门报告各写各的，实测（2026-11-05）：

| 报告 | 有 `generated_at` | 有世代 | 有代码版本 |
|---|---|---|---|
| `k2_hard_gates.json` | 否 | 是 | 否 |
| `k2_perf.json` | 是 | 是 | 否 |
| `k2_gray.json` | 是 | 是 | 否 |
| `k2_publish_check.json` | 否 | 是 | 否 |
| `k3_threshold_gate.json` | 否 | 是 | 否 |
| `k4_drill.json` | 否 | 否 | 否 |
| `p1c_leak_audit.json` | 否 | 否 | 否 |

更要紧的是 `scripts/k2_publish_check.py` 的 `check_reports_freshness()`：它对
查不到 `generated_at` 的报告记一个 `undetermined`，然后**整道门仍然能过**。
「我查不了」被安静地当成了「不用查」——转述给读者时就变成「这批报告是新鲜的」。

这不是文档能防的事。文档只会写「报告应当包含……」；而这里的每一道门都会
在字段缺失时**让退出码变成 1**。

## 三态与判据

* `pass`：字段填了、格式合法；
* `fail`：填了但不合法（commit 不是 40 位十六进制、口径已作废）；
* `unknown`：没填、读不到、无法判断。

**`unknown` 不算通过**，这条由 `TestUnknownIsNotPass` 逐字段钉住，并由
`TestRealReportsAreAudited` 保证「有问题的报告不会悄悄变多、也不会悄悄变少」。

## 反向实测怎么做

1. `py -3 -m pytest tests/test_report_provenance.py -q` → 全绿；
2. 把 `src/arkrag/provenance.py` 里 `_fold()` 的 UNKNOWN 分支删掉 →
   `-k test_a_report_missing_a_field_is_unknown` 必须红；
3. 恢复 → 必须绿。

第二笔：把 `scripts/check_report_provenance.py` 的
`return EXIT_OK if all(...) else EXIT_NOT_PASS` 改成 `return EXIT_OK` →
`test_cli_exits_nonzero_for_a_noncompliant_report` 必须红。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from arkrag import provenance, subproc  # noqa: E402

SCRIPT = ROOT / "scripts" / "check_report_provenance.py"
DATA = ROOT / "data"

#: 当前**还不满足**七字段的报告。这份名单只能变小：
#: 给某份报告补上可追溯字段后，必须把它从名单里删掉（否则
#: `test_debt_registry_matches_actual_state` 红）；新增一份不合格报告也要登记
#: （否则 `test_no_unregistered_noncompliant_report` 红）。
#:
#: 登记键是文件名，值是「缺什么、为什么今天还没补」。补的时候把值删掉，
#: 别留一行「已修好」——名单的长度就是这笔债的余额。
REPORTS_AWAITING_PROVENANCE: dict[str, str] = {
    "k2_hard_gates.json": "缺 generated_at / git_commit / code_sha256_16 / dataset / inputs / environment",
    "k2_perf.json": "缺 git_commit / code_sha256_16 / dataset / inputs / environment",
    "k2_gray.json": "缺 git_commit / code_sha256_16 / dataset / inputs / environment",
    "k2_publish_check.json": "缺 generated_at / git_commit / code_sha256_16 / dataset / inputs / environment",
    "k3_threshold_gate.json": "缺 generated_at / git_commit / code_sha256_16 / dataset / inputs / environment",
    "k4_drill.json": "缺 generated_at / generation_id / git_commit / code_sha256_16 / dataset / inputs / environment",
    "p1c_leak_audit.json": "缺 generated_at / generation_id / git_commit / code_sha256_16 / dataset / inputs / environment",
}


#: 仓库里**实际存在**的门报告，挨个体检。名单要覆盖全部，不能只挑好对付的：
#: 少列一份，就等于给它一份免检豁免。
AUDITED_REPORTS: tuple[str, ...] = (
    "k2_hard_gates.json",
    "k2_perf.json",
    "k2_gray.json",
    "k2_publish_check.json",
    "k3_threshold_gate.json",
    "k4_drill.json",
    "p1c_leak_audit.json",
    # p0_item5_drill.json 是 P0 第 5 项（隔离副本审批/发布/回滚/归档）的门报告。
    # 它七字段齐备（`provenance.verify` 逐项 pass），所以既不在债表里，
    # 也必须在这里被体检——漏列就等于给它一份免检豁免。
    "p0_item5_drill.json",
)


def _good() -> dict:
    """一份七字段齐全的最小报告。"""
    return {
        "generated_at": "2026-11-05T00:00:00Z",
        "generation_id": "g-20261103T132152Z-06a741",
        "git_commit": "23590edf1dc172aa48798a5c1a89b1ad53212de9",
        "code_sha256_16": "f5a698523142bc9a",
        "dataset": {
            "gold_path": "data/goldset/gold-v2.json",
            "gold_sha16": "069375f8e355e4b9",
            "gold_total": 640,
        },
        "inputs": [{"path": "data/goldset/gold-v2.json", "sha16": "069375f8e355e4b9"}],
        "environment": {
            "python": "3.14.3",
            "platform": "win32",
            "os": "Windows-11-10.0.26200-SP0",
            "machine": "AMD64",
            "cpu_count": 20,
        },
        "all_passed": True,
    }


def _cli(*paths: str) -> subprocess.CompletedProcess:
    """跑检查命令。

    **必须传 `env=subproc.child_env()`**：这个 CLI 打印中文（「不存在」「缺」等），
    而下面的断言就是在找这些中文。父进程按 UTF-8 解码子进程输出，子进程若是默认
    cp936，中文会变成 `\\ufffd`，于是 `assert "不存在" in proc.stdout` 失败——
    实测（2026-11-05，不设 PYTHONIOENCODING）：
        FAILED test_cli_exits_nonzero_for_a_missing_file
        assert '\\u4e0d\\u5b58\\u5728' in '...nope.json unknown   \\ufffd\\ufffd\\ufffd\\ufffd...'
    CLI 明明打印了「不存在」，测试说它没打印。与 Gate 0 的 K2 事故完全同形。
    """
    return subprocess.run(
        [sys.executable, str(SCRIPT), *paths],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        env=subproc.child_env(),
    )


class TestUnknownIsNotPass:
    def test_a_compliant_report_passes(self) -> None:
        """七字段齐全 + 自报通过 → pass。这是正面对照，证明门不是永远红。"""
        assert provenance.verify(_good())["verdict"] == provenance.PASS
        assert provenance.gate_ok(_good()) is True

    @pytest.mark.parametrize("field", provenance.REQUIRED_FIELDS)
    def test_a_report_missing_a_field_is_unknown(self, field: str) -> None:
        """逐个字段删掉，每一次都必须判 unknown。

        参数化七个而不是只测一个，是因为七个判据各写了一段代码——
        漏测某一个，那段代码就可能在 `if` 里短路成永远返回 pass。
        """
        payload = _good()
        del payload[field]
        result = provenance.verify(payload)
        assert result["verdict"] == provenance.UNKNOWN, (
            f"缺 {field} 时判成了 {result['verdict']}——缺信息不等于没问题"
        )
        assert field in result["missing"], "缺哪个字段必须能反查到"

    def test_unknown_is_not_gate_ok(self) -> None:
        payload = _good()
        del payload["generated_at"]
        assert provenance.gate_ok(payload) is False

    def test_a_self_declared_failure_is_not_gate_ok(self) -> None:
        """七字段齐全但报告自报未通过 → 门不能算过。"""
        payload = _good()
        payload["all_passed"] = False
        assert provenance.verify(payload)["verdict"] == provenance.PASS  # 字段层面合法
        assert provenance.gate_ok(payload) is False  # 但整体不能算过


class TestMalformedValuesAreFailNotUnknown:
    def test_bad_commit_is_fail(self) -> None:
        payload = _good()
        payload["git_commit"] = "not-a-commit"
        result = provenance.verify(payload)
        assert result["verdict"] == provenance.FAIL
        assert "git_commit" in result["invalid"]

    def test_bad_code_sha_is_fail(self) -> None:
        payload = _good()
        payload["code_sha256_16"] = "xyz"
        result = provenance.verify(payload)
        assert result["verdict"] == provenance.FAIL
        assert "code_sha256_16" in result["invalid"]

    def test_fail_outranks_unknown(self) -> None:
        """一个字段填错 + 一个字段没填 → 整体是 fail，不是 unknown。

        反过来的话，「填错了」会被「没填」稀释掉，而填错比没填需要更早处理。
        """
        payload = _good()
        payload["git_commit"] = "nope"
        del payload["environment"]
        result = provenance.verify(payload)
        assert result["verdict"] == provenance.FAIL
        assert result["missing"] == ["environment"]
        assert result["invalid"] == ["git_commit"]


class TestCaliberAndDownstream:
    def test_recall_row_without_caliber_is_unknown(self) -> None:
        row = {"r20": 0.7967, "context_hits": 170}
        result = provenance.verify_metric_row(row)
        assert result["verdict"] == provenance.UNKNOWN
        assert result["missing"] == ["caliber"]

    def test_retired_wide_caliber_is_fail(self) -> None:
        """`equiv_v1_wide` 已作废（P1-c 实测它把 47/96 放成 96/96）。"""
        row = {"r20": 0.9271, "context_hits": 170, "caliber": "equiv_v1_wide"}
        result = provenance.verify_metric_row(row)
        assert result["verdict"] == provenance.FAIL
        assert "作废" in result["detail"]

    def test_recall_row_without_downstream_is_fail(self) -> None:
        """只看 R@20 的行必须并列一个下游量。

        反例来自 Lead 的六配置真跑表：`hop0.5` 的 R@20=0.7379 排全场第二，
        但真进上下文净增 0、引用覆盖 51→48。只看 R@20 的报告会推荐它。
        """
        row = {"r20": 0.7379, "caliber": "gold_only"}
        result = provenance.verify_metric_row(row)
        assert result["verdict"] == provenance.FAIL
        assert "hop0.5" in result["detail"]

    def test_a_complete_recall_row_passes(self) -> None:
        row = {"r20": 0.7379, "context_hits": 162, "caliber": "equiv_v2"}
        assert provenance.verify_metric_row(row)["verdict"] == provenance.PASS

    def test_a_throughput_row_does_not_need_a_caliber(self) -> None:
        """没有 recall 键的行（吞吐 / 内存）不适用这条规则，不该被误伤。"""
        row = {"qps": 42.0, "p95_ms": 118}
        assert provenance.verify_metric_row(row)["verdict"] == provenance.PASS


class TestCliExitCodes:
    def test_cli_exits_zero_for_a_compliant_report(self, tmp_path: Path) -> None:
        path = tmp_path / "good.json"
        path.write_text(json.dumps(_good(), ensure_ascii=False), encoding="utf-8")
        proc = _cli(str(path))
        assert proc.returncode == 0, proc.stdout + proc.stderr

    def test_cli_exits_nonzero_for_a_noncompliant_report(self, tmp_path: Path) -> None:
        """缺 `git_commit` 的报告 → 退出码 1，且输出里点名缺哪个字段。

        这条是整道门对外的那一面：别的脚本/人怎么知道这批报告能不能用。
        """
        payload = _good()
        del payload["git_commit"]
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        proc = _cli(str(path))
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "git_commit" in proc.stdout

    def test_cli_exits_nonzero_for_a_missing_file(self, tmp_path: Path) -> None:
        """报告不存在 → unknown → 非零。不存在不能算通过。"""
        proc = _cli(str(tmp_path / "nope.json"))
        assert proc.returncode == 1
        assert "不存在" in proc.stdout

    def test_cli_reports_relative_paths_only(self, tmp_path: Path) -> None:
        """输出里的路径必须是相对仓库根的。

        绝对机台路径入库后会被 `tests/test_portable_paths.py` 判红
        （2026-11-04 真发生过一次，见 Lead 的 `a3a93ae`）。
        """
        proc = _cli("data/k2_perf.json")
        assert proc.returncode in (0, 1)
        assert "data/k2_perf.json" in proc.stdout
        assert "E:\\" not in proc.stdout and "E:/" not in proc.stdout

    def test_cli_json_mode_is_machine_readable(self) -> None:
        proc = _cli("--json", "data/k4_drill.json")
        assert proc.returncode == 1
        rows = json.loads(proc.stdout)
        assert rows[0]["verdict"] == provenance.UNKNOWN
        assert rows[0]["missing"], "缺项名单不能为空——这正是今天的问题"


def _gate_reports_on_disk() -> list[str]:
    """`data/` 下所有「看起来像门报告」的 JSON 文件名。

    识别方式：顶层带 `checks` / `all_passed` / `passed` 之一。不是白名单——
    是新文件也会被认出来的正向判据。**两条门（未登记、名单覆盖）都从它出发**，
    否则就会出现「一条恒真、另一条独自扛」的分工，而那正是 P1-c 那次
    `test_no_unregistered_noncompliant_report` 写成空转的形状。
    """
    found: list[str] = []
    for path in sorted(DATA.glob("*.json")):
        try:
            payload = json.loads(path.read_text("utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(payload, dict) and {"checks", "all_passed", "passed"} & set(payload):
            found.append(path.name)
    return found


class TestRealReportsAreAudited:
    """对真实报告做体检。这份名单就是这笔债的余额。"""

    def _verdicts(self) -> dict[str, str]:
        """磁盘上每一份门报告的判定（不只是已登记的那些）。"""
        out: dict[str, str] = {}
        for name in _gate_reports_on_disk():
            path = DATA / name
            payload = json.loads(path.read_text("utf-8"))
            out[name] = provenance.verify(payload)["verdict"]
        return out

    def test_no_unregistered_noncompliant_report(self) -> None:
        """磁盘上出现一份**未登记**的不可追溯门报告 → 门红。

        遍历的是磁盘实测出来的门报告全集，不是 `REPORTS_AWAITING_PROVENANCE`
        （后者正是「已登记」这份名单）。拿登记名单当遍历对象的话，这条用例
        永远不可能发现未登记项——**它曾经就是这么写的，第一版反向实测里替它
        变红的是旁边那条名单覆盖用例，等于它是恒真的装饰。**

        没有这条，新报告可以随便不带时间戳进来，而债表只在有人想起来时才动。
        """
        unregistered = sorted(
            name
            for name, verdict in self._verdicts().items()
            if verdict != provenance.PASS and name not in REPORTS_AWAITING_PROVENANCE
        )
        assert not unregistered, (
            f"这些报告不满足七字段，且没有登记进 `REPORTS_AWAITING_PROVENANCE`：{unregistered}。\n"
            f"要么补上可追溯字段，要么显式登记（登记要写清缺什么）——"
            f"不许让它安静地躺在那里。"
        )

    def test_audited_set_covers_every_gate_report_on_disk(self) -> None:
        """`AUDITED_REPORTS` 必须覆盖磁盘上每一份门报告。

        少列一份就等于给它一份免检豁免，而豁免是不该悄悄发生的。
        （两条门从同一个 `_gate_reports_on_disk()` 出发，但判的是两件事：
        这一条判「有没有被漏掉」，上一条判「漏掉的里面有没有不合格的」。）
        """
        missing = sorted(set(_gate_reports_on_disk()) - set(AUDITED_REPORTS))
        assert not missing, (
            f"这些文件像门报告（顶层有 checks/all_passed/passed），但不在 `AUDITED_REPORTS` 里：{missing}。"
            f"请把它们加进名单，让它们也接受七字段体检。"
        )

    def test_debt_registry_matches_actual_state(self) -> None:
        """名单**只能缩小**：某份报告补上了七字段，就必须把它从名单里删掉。

        只做「不合格就登记」的名单会越积越厚，最后没人相信里面有活条目；
        「修好了也要改名单」才让这笔债有下行压力。
        """
        # 只看**已登记**的那些。这条门判的是「登记了、但其实已经合格」，
        # 不是「盘上有合格报告」。原实现遍历全量 `_verdicts()`，于是任何一份
        # 从未登记过、七字段齐备的新报告都会让它假红——实测加进
        # `p0_item5_drill.json` 后报 stale=['p0_item5_drill.json']，而它从来
        # 就不在债表里。假红会让人学会忽略这条门。
        stale = [
            name
            for name, verdict in self._verdicts().items()
            if verdict == provenance.PASS and name in REPORTS_AWAITING_PROVENANCE
        ]
        assert not stale, (
            f"这些报告已经满足七字段了，请从 `REPORTS_AWAITING_PROVENANCE` 删掉：{stale}"
        )

    def test_entries_carry_reasons(self) -> None:
        for name, reason in REPORTS_AWAITING_PROVENANCE.items():
            assert reason.strip(), f"{name} 没写缺什么"
            assert len(reason.strip()) >= 10, f"{name} 的理由太短：{reason!r}"

    def test_every_registered_report_file_exists(self) -> None:
        """登记了却不存在 → 说明文件名写错，名单会永远少算一份。

        （`k1_checks.json` / `k3_signif.json` 不在名单里，是因为这两个脚本
        本来就不写盘；这条用例不把它们当漏项。）
        """
        missing = [name for name in REPORTS_AWAITING_PROVENANCE if not (DATA / name).exists()]
        assert not missing, f"这些登记项对应的报告文件不存在：{missing}"


class TestGateItselfCanFail:
    """合成输入直接喂判据，证明三道判据都不是恒真。"""

    def test_verify_flags_a_bare_dict(self) -> None:
        """空字典（什么字段都没有）必须是 unknown 且列出全部七项。"""
        result = provenance.verify({})
        assert result["verdict"] == provenance.UNKNOWN
        assert sorted(result["missing"]) == sorted(provenance.REQUIRED_FIELDS)

    def test_verify_rejects_a_non_dict(self) -> None:
        result = provenance.verify(["not", "a", "dict"])  # type: ignore[arg-type]
        assert result["verdict"] == provenance.FAIL

    def test_each_field_has_its_own_verdict_row(self) -> None:
        """七个字段各有一行判定，缺项也在里面。

        少了任何一行，`verify()` 就没法回答「到底缺哪个」——而 P1-c 那次
        `gold_assertion_ids` 恒为 None 的事故，恰恰是「缺输入」伪装成了
        「无法判定」。
        """
        result = provenance.verify(_good())
        assert {row["field"] for row in result["fields"]} == set(provenance.REQUIRED_FIELDS)
