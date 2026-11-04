# -*- coding: utf-8 -*-
"""P0 的四道门：报告可追溯、三态、缺输入可反查、反向实测支架。

## 为什么这些必须是门而不是文档里的一段话

Lead 定的目标是「任何报告能回答：哪份代码、哪个候选、哪套数据、何时运行、
结果是否可信」。这个目标有两个典型的失效方式，本项目都见过：

1. **字段写了但没人查。** `k2_hard_gates.json` / `k3_threshold_gate.json` /
   `k4_drill.json` 三份报告连 `generated_at` 都没有，而发布门
   `k2_publish_check.py` 只对「有该字段的那两份」比对时间戳，对另外三份记
   `undetermined` 之后继续走。**「查不了」被安静地当成「不用查」。**

2. **判据没取到输入，被降级成一个看起来合理的类别。** P1-c 实测：A 的 triage
   行里 `gold_assertion_ids` 恒为 `None`（59/59），脚本于是把每条候选都归到
   「无法判定」。退出码 0、不报错、不崩——**它把自己降级成了「没法判」，
   而真正的结论是「我没拿到判据要的输入」**。这两件事处置完全不同：前者
   只能等人工，后者是修脚本。

## 第三样东西：反向实测支架

同一形态的第三次出现在反向实测自己身上：patch 锚点写错让被测脚本崩了
（退出码 1），harness 只读 JSON 产物、读到上一轮残留，报出「结论没变」。
`arkrag.reverse_probe` 把「先看退出码、再清残留」写成代码，这里给它两道门：
一条证明它真的会在退出码非 0 时拒绝产物，一条证明它真的会清掉旧文件。

## 四条门各自怎么变红

* 七字段：删掉 `collect()` 里任一字段 → `verify()` 判 unknown → `gate_ok()`
  为 False。（不是「字典里少个键」这种断言，走的是真实判定链。）
* 三态：把 `_fold()` 改成「有 unknown 也算 pass」→ `test_unknown_is_not_pass` 红。
* 缺输入可反查：把 `verify()` 的 `missing` 改成空列表 → `test_missing_input_is_
  distinguishable_from_undecidable` 红。
* 反向实测：把 `run()` 里的 `if proc.returncode != EXIT_OK` 那段删掉 →
  `test_probe_rejects_artifacts_when_command_fails` 红；把 `clear()` 调用删掉 →
  `test_probe_clears_previous_artifacts` 红。

每条命令级反向实测记录见 `docs/p0_门报告可追溯.md`。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from arkrag import provenance  # noqa: E402
from arkrag import reverse_probe  # noqa: E402


# ---------------------------------------------------------------- 夹具


def _good_block() -> dict:
    """一份七字段齐备的报告骨架。字段值都合法，便于逐项改坏。"""
    return {
        "generated_at": "2026-11-05T01:00:00Z",
        "generation_id": "g-20261103T132152Z-06a741",
        "git_commit": "23590edf1dc172aa48798a5c1a89b1ad53212de9",
        "code_sha256_16": "cbab705564922bee",
        "dataset": {
            "gold_path": "data/goldset/gold-v2.json",
            "gold_sha16": "069375f8e355e4b9",
            "split_path": "data/goldset/split-v2.json",
            "split_sha16": "3d586b36efc7ce85",
            "gold_total": 640,
        },
        "inputs": [
            {"path": "data/goldset/gold-v2.json", "sha16": "069375f8e355e4b9", "bytes": 601492}
        ],
        "environment": {
            "python": "3.14.3",
            "platform": "win32",
            "os": "Windows-11-10.0.26200-SP0",
            "machine": "AMD64",
            "cpu_count": 20,
        },
        "all_passed": True,
    }


# ---------------------------------------------------------------- 门 1：七字段


class TestSevenFieldsAreCollected:
    """七字段齐备、值合法，且 `collect()` 真的会把环境读进来。"""

    def test_good_block_passes(self) -> None:
        result = provenance.verify(_good_block())
        assert result["verdict"] == provenance.PASS
        assert result["missing"] == []
        assert result["invalid"] == []

    def test_collect_produces_all_seven_fields(self) -> None:
        """字段名齐、值合法，且采集出来的块自己能过门。

        注意 `inputs` 必须由调用方显式给：`collect()` 默认 `inputs=[]`，那时
        报告判 unknown（「没列出指标吃了哪些文件」）。这一条刻意传了真实输入，
        为的是证明「字段名对」与「这份报告真的能过门」是两件事——
        只断言 `sorted(block) == sorted(REQUIRED_FIELDS)` 的话，
        一个全部填 None 的块也能通过。
        """
        block = provenance.collect(
            generation_id="g-20261103T132152Z-06a741",
            inputs=provenance.inputs_block(
                ["data/goldset/gold-v2.json", "data/goldset/split-v2.json"]
            ),
        )
        assert sorted(block) == sorted(provenance.REQUIRED_FIELDS)
        result = provenance.verify(block)
        assert result["verdict"] == provenance.PASS, result["detail"]

    def test_collect_leaves_missing_values_blank_instead_of_faking_them(self) -> None:
        """采集不到就留空，不许填「大概是这样」的默认值。

        `git_commit` 在非仓库环境里取不到；此时它必须是空串，由 `verify()` 判
        unknown。若 `collect()` 自己填上一个占位串（比如 `"unknown"`），
        `judge_git_commit` 会把它判成 fail 而不是 unknown——错误类型就变了。
        """
        block = provenance.collect(generation_id="g-x", inputs=[])
        assert block["inputs"] == []          # 不猜「大概是 gold 吧」
        result = provenance.verify(block)
        # inputs 空 → 该项 unknown，整份报告就不该算通过。
        assert result["verdict"] != provenance.PASS
        assert "inputs" in result["missing"]

    def test_code_fingerprint_changes_with_content_not_just_commit(self) -> None:
        """`code_sha256_16` 与 `git_commit` 不是重复项。

        有未提交改动时 commit 号不变而指纹变——这正是「改完代码忘了提交、
        两份报告 commit 相同但结果不同」的缺口。这里用两条不同路径的指纹
        必然不同来证明它真的读内容，而不是读什么常量。
        """
        a = provenance.code_sha256_16(("src/arkrag",))
        b = provenance.code_sha256_16(("scripts",))
        assert provenance._SHA16_RE.match(a), f"指纹形状不对：{a!r}"
        assert provenance._SHA16_RE.match(b), f"指纹形状不对：{b!r}"
        assert a != b, "两个不同目录算出同一个指纹，说明没在哈希内容"

    def test_dataset_block_reads_real_gold_and_split(self) -> None:
        block = provenance.dataset_block()
        assert block["gold_path"].endswith("gold-v2.json")
        assert block["gold_sha16"] == "069375f8e355e4b9"
        assert block["gold_total"] == 640
        assert block["split_sha16"] == "3d586b36efc7ce85"

    def test_inputs_block_reports_missing_file_as_blank_sha(self, tmp_path: Path) -> None:
        """文件不在盘上时 sha16 留空，由 `judge_inputs` 判 unknown。

        这一条防的是「inputs 写了但其实是查不到的文件」——报告上有这一行，
        看上去要素齐了，实际指标不可重算。
        """
        ghost = tmp_path / "nope.json"
        rows = provenance.inputs_block([ghost])
        assert rows[0]["sha16"] == ""
        result = provenance.judge_inputs(rows)
        assert result[0] == provenance.UNKNOWN
        assert "算不出 sha16" in result[1]


# ---------------------------------------------------------------- 门 2：三态


class TestUnknownIsNotPass:
    """`unknown` 不是通过。这是本组最要紧的一条。"""

    @pytest.mark.parametrize(
        "field",
        ["generated_at", "generation_id", "git_commit", "code_sha256_16",
         "dataset", "inputs", "environment"],
    )
    def test_each_field_missing_makes_the_report_unknown(self, field: str) -> None:
        """七个字段逐个删掉，每一次都必须落到 unknown（而不是悄悄通过）。"""
        payload = _good_block()
        del payload[field]
        result = provenance.verify(payload)
        assert result["verdict"] == provenance.UNKNOWN, (
            f"删掉 {field} 之后报告仍判 {result['verdict']}：缺字段必须判 unknown"
        )
        assert field in result["missing"]

    def test_gate_ok_is_false_for_unknown(self) -> None:
        """`gate_ok()` 是发布路径唯一的入口，`unknown` 不许过它。"""
        payload = _good_block()
        del payload["dataset"]
        assert provenance.verify(payload)["verdict"] == provenance.UNKNOWN
        assert provenance.gate_ok(payload) is False

    def test_gate_ok_is_true_only_when_everything_holds(self) -> None:
        payload = _good_block()
        assert provenance.gate_ok(payload) is True
        payload["all_passed"] = False
        assert provenance.gate_ok(payload) is False

    def test_fail_outranks_unknown(self) -> None:
        """能确定错了就先说错了，别让「另一项查不了」把确定的错误稀释掉。

        这一条是 `_fold()` 的语义。若改成「unknown 优先」，一份既有错字段、
        又有缺字段的报告会被报成 unknown——处置方式从「去改那个错值」变成
        「等人工」，而错值明明已经看得见。
        """
        payload = _good_block()
        del payload["environment"]              # 缺 → unknown
        payload["git_commit"] = "not-a-commit"  # 不合法 → fail
        result = provenance.verify(payload)
        assert result["verdict"] == provenance.FAIL
        assert "git_commit" in result["invalid"]
        assert "environment" in result["missing"]

    def test_legacy_alias_is_recorded_not_silently_renamed(self) -> None:
        """`k2_publish_check.py` 里的 `undetermined` 是同一概念的历史拼写。

        本模块不改它（消费者改生产者的报告格式，等于让报告与产生它的代码对不上），
        但必须把「两种拼写同时存在」记成一个已知事实。若哪天有人顺手把这段
        删掉，说明差异被遗忘了。
        """
        assert provenance.LEGACY_ALIASES == {"undetermined": provenance.UNKNOWN}
        assert provenance.UNKNOWN == "unknown"


# ---------------------------------------------------------------- 门 3：缺输入可反查


class TestMissingInputIsDistinguishable:
    """「没拿到判据要的输入」必须与「真的判不了」是两种不同的输出。

    P1-c 实测形态：A 的 triage 行里 `gold_assertion_ids` 恒为 None（59/59），
    脚本于是把每条候选归到「无法判定」。退出码 0、不报错——它安静地降级成了
    一个看起来合理的类别。修法不是「让脚本聪明点」，而是**让缺输入这件事
    必须能被反查到具体字段**。
    """

    def test_verify_names_the_missing_field(self) -> None:
        payload = _good_block()
        del payload["inputs"]
        result = provenance.verify(payload)
        # 关键：不是笼统一句「无法判定」，而是能点名缺的是 inputs。
        assert result["missing"] == ["inputs"]
        assert result["fields"], "fields 为空则无法反查是哪一项缺"
        row = next(f for f in result["fields"] if f["field"] == "inputs")
        assert row["verdict"] == provenance.UNKNOWN
        assert "inputs" in row["detail"] or "指标" in row["detail"]

    def test_every_field_has_a_verdict_row(self) -> None:
        """七字段**每一个**都要有独立判定行，缺一不可。

        若有人为了省事把「缺失项」从 `fields` 里剔掉（只留 `missing` 名单），
        报告里就再也看不到「这一项本来该查什么」。名单只能回答「缺什么」，
        不能回答「缺了会怎样」。
        """
        payload = _good_block()
        del payload["dataset"]
        del payload["environment"]
        result = provenance.verify(payload)
        names = [f["field"] for f in result["fields"]]
        assert names == list(provenance.REQUIRED_FIELDS)
        # 缺失项也在里面，而且是 unknown。
        assert {f["field"] for f in result["fields"] if f["verdict"] == provenance.UNKNOWN} == {
            "dataset",
            "environment",
        }

    def test_unknown_and_fail_are_reported_in_separate_lists(self) -> None:
        """处置方式不同，所以必须分开两个名单。

        缺的要补采集（改脚本），不合法的要改值（改数据/改常量）。合成一个
        名单，就会有人用「把值改成空」的方式让 fail 变成 unknown，然后说
        「这一项现在只是查不了」。
        """
        payload = _good_block()
        payload["code_sha256_16"] = "ZZZZ"      # fail
        del payload["generated_at"]             # unknown
        result = provenance.verify(payload)
        assert result["missing"] == ["generated_at"]
        assert result["invalid"] == ["code_sha256_16"]

    def test_not_a_dict_is_fail_not_unknown(self) -> None:
        """报告本身不是 dict（比如被写成一个 list）：这是坏，不是「查不了」。"""
        result = provenance.verify(["not", "a", "report"])
        assert result["verdict"] == provenance.FAIL
        assert result["missing"] == list(provenance.REQUIRED_FIELDS)

    def test_bare_string_input_rows_are_not_pass(self) -> None:
        """`inputs` 里放纯字符串（没算出 sha16）必须判 unknown，不许判 pass。

        这是本项目「缺输入伪装成通过」的第二种形态，实测于 P0：`collect()`
        允许调用方直接传路径列表，于是

            provenance.collect(generation_id="g-x", inputs=["data/goldset/gold-v2.json"])

        产出的 `inputs` 是 `['data/goldset/gold-v2.json']`，而 `judge_inputs()`
        只检查 `isinstance(row, dict)` 的行 —— 字符串行**一条都不检查**，
        于是返回 `("pass", "inputs 1 个文件都已登记 sha16")`。
        **一个 sha16 都没登记，却报「都已登记」。**

        这不是理论问题：判据里的 `if isinstance(row, dict) and ...` 是「跳过
        不认识的形状」，而「不认识」在这里恰好等价于「没登记」。凡是「跳过
        不认识的东西」的判据，都必须再补一条「不认识的东西本身算不算合格」。
        """
        verdict, detail = provenance.judge_inputs(["data/goldset/gold-v2.json"])
        assert verdict == provenance.UNKNOWN, (
            f"纯字符串行没有 sha16，不能判 pass（实际 {verdict}：{detail}）。"
            f"要么让调用方传完整行，要么在这里判 unknown。"
        )
        assert "sha16" in detail or "无法重算" in detail

    def test_input_rows_without_sha16_are_unknown(self) -> None:
        """dict 行缺 `sha16`（文件不在盘上时就是这样）同样必须 unknown。"""
        verdict, detail = provenance.judge_inputs(
            [{"path": "data/goldset/gold-v2.json", "sha16": "", "bytes": None}]
        )
        assert verdict == provenance.UNKNOWN
        assert "sha16" in detail

    def test_a_row_that_is_neither_str_nor_dict_is_unknown(self) -> None:
        """数字 / None 这类行既没路径也没 sha16，不能算合格。"""
        for bad in (42, None, 3.5, ["nested"]):
            verdict, detail = provenance.judge_inputs([bad])
            assert verdict == provenance.UNKNOWN, f"{bad!r} 不该判 pass（{detail}）"


# ---------------------------------------------------------------- 门 4：口径与下游量


class TestCaliberAndDownstream:
    """`caliber` 是硬字段；只报 R@n 的行必须并列一个下游量。"""

    def test_missing_caliber_on_a_recall_row_is_unknown(self) -> None:
        row = {"r20": 0.7833, "context_hits": 162}
        result = provenance.verify_metric_row(row)
        assert result["verdict"] == provenance.UNKNOWN
        assert result["has_recall"] is True

    def test_retired_wide_caliber_is_fail_not_unknown(self) -> None:
        """已作废的 V1 宽口径必须判 fail，不能判 unknown。

        判 unknown 会让人以为「补个字段就好了」，而这一口径本身算出来的数
        是假的（Lead 2026-11-05 已撤回 +30pt）。
        """
        row = {"r20": 0.9271, "caliber": "equiv_v1_wide", "context_hits": 170}
        result = provenance.verify_metric_row(row)
        assert result["verdict"] == provenance.FAIL
        assert "作废" in result["detail"]

    @pytest.mark.parametrize("caliber", ["gold_only", "equiv_v2"])
    def test_both_live_calibers_pass(self, caliber: str) -> None:
        row = {"r20": 0.7833, "caliber": caliber, "context_hits": 162}
        assert provenance.verify_metric_row(row)["verdict"] == provenance.PASS

    def test_recall_without_downstream_is_fail(self) -> None:
        """只报 R@n、不并列下游量：判 fail。

        实测反例写进失败信息，防止下次有人问「为什么非要下游量」。
        """
        row = {"r20": 0.7379, "caliber": "gold_only"}
        result = provenance.verify_metric_row(row)
        assert result["verdict"] == provenance.FAIL
        assert "hop0.5" in result["detail"], "失败信息里必须带那个反例配置的名字"

    @pytest.mark.parametrize(
        "downstream",
        ["context_hits", "citation_covered", "answer_cover", "downstream"],
    )
    def test_any_recognised_downstream_suffices(self, downstream: str) -> None:
        row = {"r20": 0.7379, "caliber": "gold_only", downstream: 162}
        assert provenance.verify_metric_row(row)["verdict"] == provenance.PASS

    def test_row_without_recall_is_not_applicable(self) -> None:
        """吞吐、延迟这类行没有口径可言，不该被口径检查误伤。

        误伤会让人学会忽略这道门——这与「门不红」同样糟糕。
        """
        row = {"p95_ms": 42.0, "qps": 118.0}
        result = provenance.verify_metric_row(row)
        assert result["verdict"] == provenance.PASS
        assert result["has_recall"] is False

    def test_verify_metric_rows_lists_bad_rows_individually(self) -> None:
        rows = [
            {"r20": 0.78, "caliber": "gold_only", "context_hits": 162},
            {"r20": 0.73, "caliber": "gold_only"},              # 缺下游量
            {"r20": 0.92, "caliber": "equiv_v1_wide", "context_hits": 170},  # 作废口径
        ]
        result = provenance.verify_metric_rows(rows)
        assert result["verdict"] == provenance.FAIL
        assert result["bad"] == [1, 2]


# ---------------------------------------------------------------- 门 5：反向实测支架


class TestReverseProbeRefusesStaleEvidence:
    """反向实测支架：先看退出码，再清残留。两条各有一道门。"""

    def _script(self, tmp_path: Path, body: str) -> Path:
        path = tmp_path / "probe_target.py"
        path.write_text(body, encoding="utf-8")
        return path

    def test_probe_rejects_artifacts_when_command_fails(self, tmp_path: Path) -> None:
        """命令退出码非 0 时，产物一律视为无效——**即使产物文件确实存在**。

        这正是实测踩到的形态：patch 锚点写错让被测脚本崩了，脚本崩溃前已经
        把上一轮的文件留在盘上（或者根本没写），harness 只看文件就报出
        「结论没变」。这条门要求 `usable` 为 False，且理由里点名退出码。
        """
        artifact = tmp_path / "out.json"
        artifact.write_text('{"stale": true}', encoding="utf-8")  # 上一轮的残留
        script = self._script(
            tmp_path,
            "import sys\n"
            "from pathlib import Path\n"
            f"Path({str(artifact)!r}).write_text('stale', encoding='utf-8')\n"
            "sys.exit(1)\n",
        )
        result = reverse_probe.run(
            "崩溃的脚本", [sys.executable, str(script)], artifacts=[str(artifact)]
        )
        assert result["returncode"] == 1
        assert result["usable"] is False, "退出码非 0 却判产物可用——这正是那次假结论的形状"
        assert "退出码" in result["reason"]

    def test_probe_clears_previous_artifacts(self, tmp_path: Path) -> None:
        """每轮开始前必须清掉上一轮产物。

        这台脚本**什么都不写**却正常退出（分支没走到）。若不清残留，读到的
        就是上一轮的文件，而退出码是 0——光看退出码堵不住这种。
        """
        artifact = tmp_path / "out.json"
        artifact.write_text('{"stale": true}', encoding="utf-8")
        script = self._script(tmp_path, "print('我没有写任何文件')\n")
        result = reverse_probe.run(
            "不写产物的脚本", [sys.executable, str(script)], artifacts=[str(artifact)]
        )
        assert result["returncode"] == 0
        assert result["cleared"] == [reverse_probe.config.rel_to_root(artifact)]
        assert not artifact.exists(), "上一轮产物没被清掉，本轮会读到它"
        assert result["usable"] is False, "产物不存在却判可用"
        assert "不存在" in result["reason"]

    def test_probe_accepts_a_fresh_artifact_from_a_successful_run(self, tmp_path: Path) -> None:
        """正面对照：退出码 0 且产物是本轮新写的 → usable 为 True。

        缺这一条，上面两条可能只是在断言「永远 unusable」。
        """
        artifact = tmp_path / "out.json"
        script = self._script(
            tmp_path,
            "import json, pathlib\n"
            f"pathlib.Path({str(artifact)!r}).write_text(json.dumps({{'fresh': 1}}), encoding='utf-8')\n",
        )
        result = reverse_probe.run(
            "正常脚本", [sys.executable, str(script)], artifacts=[str(artifact)]
        )
        assert result["returncode"] == 0
        assert result["usable"] is True, result["reason"]
        payload = json.loads(artifact.read_text("utf-8"))
        assert payload == {"fresh": 1}

    def test_probe_output_survives_cp936_console(self, tmp_path: Path) -> None:
        """子进程输出中文能被读回来——反向实测支架自己不许踩 Gate 0 那个坑。

        若把 `subproc.child_env()` 从 `run()` 里去掉，子进程在 cp936 管道下
        把中文写成 GBK 字节，父进程按 UTF-8 解码得 `\\ufffd`，断言字符串
        永远匹配不上，表现为「脚本没输出」——而它其实输出了。
        """
        script = self._script(tmp_path, "print('[i] 监听 http://127.0.0.1:12345')\n")
        result = reverse_probe.run("中文输出", [sys.executable, str(script)])
        assert result["usable"] is True
        assert "监听" in result["stdout"], (
            f"中文没读回来（stdout={result['stdout']!r}）：子进程管道编码没设"
        )
        assert "\ufffd" not in result["stdout"]
