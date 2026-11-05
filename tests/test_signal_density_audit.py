# -*- coding: utf-8 -*-
"""`scripts/signal_density_audit.py` 的门：密度分桶必须可复算，且两种分母不许合并。

## 为什么这个脚本需要门

它不是一次性分析，而是**下一轮调参唯一的读数来源**。P2 第一轮的真跑已经证明：
cooc 加成的成败由密度决定、与题型无关，而原始报告里根本没有这个比值。把这个数
算出来这件事本身，必须能被第三方用同一条命令复算——否则它只是又一份「当时跑过
的人知道怎么算」的表格。

## 三道门各自防什么

1. **分母口径**（`TestBucketCalibers`）：`all_questions` 与 `boost_applied_only`
   必须都在报告里。只给一种的报告结构上能解析、密度三件套也齐，它会照样通过
   检查——而它缺的正是「215 道完全没加成的题把结论稀释了」这件事。
2. **筛选字段**（`test_caliber_filters_on_applied_not_boosted`）：141 题有命中、
   135 题真正把加成写进 score。用 `boosted` 筛会让 0–1 桶从 38 变成 44，与
   Lead 2026-11-05 的表差 6 题。差 6 题不会报错，只会让结论偏。
3. **三个指标分开**（`TestDensityRows`）：`any_gold` / `all_gold` / `covered`
   在同一桶里可以是 3 / 1 / 2。只报一个，拿另一个口径去对账的人会发现数对不上。

## 反向实测怎么做

把 `CALIBERS_DENSITY` 砍成单个 → `test_report_declares_both_calibers` 红；
把 `bucket_summary` 的筛选改回 `signal_boosted` → 
`test_caliber_filters_on_applied_not_boosted` 红；
把 `self_check` 的口径检查删掉 → `test_self_check_catches_single_caliber_report` 红。
命令级记录见 `docs/p0_门报告可追溯.md`。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

SCRIPT = ROOT / "scripts" / "signal_density_audit.py"
REPORT = ROOT / "data" / "signal_density_audit.json"
SOURCE = ROOT / "data" / "a_p2_round1.json"


def _load():
    spec = importlib.util.spec_from_file_location("signal_density_audit_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SD = _load()


# ---------------------------------------------------------------- 分桶


class TestBucketBoundaries:
    """桶边界是左闭右开。`None`（没有上下文单元）必须单列，不能混进 0–1。"""

    @pytest.mark.parametrize(
        "density,expected",
        [
            (0.0, "0-1"),
            (0.999, "0-1"),
            (1.0, "1-2"),
            (1.999, "1-2"),
            (2.0, "2-4"),
            (3.999, "2-4"),
            (4.0, "4+"),
            (20.0, "4+"),
        ],
    )
    def test_boundaries(self, density: float, expected: str) -> None:
        assert SD.bucket_of(density) == expected

    def test_none_is_its_own_bucket(self) -> None:
        """密度未定义（`units=0`）不能算进 0–1：那是「没有上下文」，不是「低密度」。

        混进去会让 0–1 桶的 n 变大、净增益被摊薄，看起来像「低密度也没用」。
        """
        assert SD.bucket_of(None) == "n/a"


# ---------------------------------------------------------------- 口径


class TestBucketCalibers:
    """两种分母必须都在报告里，且 `boost_applied_only` 必须用 `applied` 筛。"""

    def _rows(self) -> list[dict]:
        return [
            # 真正被加成：applied=3，密度 3/2=1.5
            {"density": 1.5, "signal_boosted": 3, "signal_applied": 3,
             "saved": {"any_gold": True, "all_gold": False, "covered": False},
             "lost": {"any_gold": False, "all_gold": False, "covered": False}},
            # 有命中但没加成：boosted=2、applied=0，密度 0/2=0
            {"density": 0.0, "signal_boosted": 2, "signal_applied": 0,
             "saved": {"any_gold": False, "all_gold": False, "covered": False},
             "lost": {"any_gold": False, "all_gold": False, "covered": False}},
        ]

    def test_all_questions_counts_every_row(self) -> None:
        table = SD.bucket_summary(self._rows(), caliber="all_questions")
        assert sum(cell["n"] for cell in table) == 2

    def test_applied_only_drops_rows_that_were_not_boosted(self) -> None:
        table = SD.bucket_summary(self._rows(), caliber="boost_applied_only")
        assert sum(cell["n"] for cell in table) == 1
        assert [cell["bucket"] for cell in table] == ["1-2"]

    def test_caliber_filters_on_applied_not_boosted(self) -> None:
        """用 `boosted` 筛会把「命中了但增幅为零」的题算进有效区。

        实测差 6 题（141 vs 135），0–1 桶 n 从 38 变 44。差 6 题不报错，
        只让「低密度有效」这个结论被稀释。
        """
        rows = self._rows()
        by_applied = SD.bucket_summary(rows, caliber="boost_applied_only")
        assert sum(cell["n"] for cell in by_applied) == 1, (
            "applied=0 的行必须被排除；若这里变成 2，说明筛选用的是 signal_boosted"
        )

    def test_unknown_caliber_raises(self) -> None:
        with pytest.raises(ValueError):
            SD.bucket_summary(self._rows(), caliber="拍脑袋口径")

    def test_three_metrics_are_kept_separate(self) -> None:
        """`any_gold` 救了不等于 `all_gold` 救了——两个数必须各存一份。"""
        table = SD.bucket_summary(self._rows(), caliber="all_questions")
        cell = [c for c in table if c["bucket"] == "1-2"][0]
        assert cell["any_gold"]["saved"] == 1
        assert cell["all_gold"]["saved"] == 0
        assert cell["covered"]["saved"] == 0


# ---------------------------------------------------------------- 每行明细


class TestDensityRows:
    """`density_rows()` 要相对基线算救/丢，且三个指标各算一遍。"""

    def _payload(self) -> dict:
        return {
            "per_question": {
                "base": [
                    {"question_id": "q1", "any_gold": False, "all_gold": False, "covered": False,
                     "cooc_boosted": 0, "cooc_applied": 0, "units": 5},
                    {"question_id": "q2", "any_gold": True, "all_gold": True, "covered": True,
                     "cooc_boosted": 0, "cooc_applied": 0, "units": 5},
                ],
                "cand": [
                    {"question_id": "q1", "any_gold": True, "all_gold": True, "covered": False,
                     "cooc_boosted": 3, "cooc_applied": 3, "units": 2},
                    {"question_id": "q2", "any_gold": False, "all_gold": False, "covered": False,
                     "cooc_boosted": 1, "cooc_applied": 1, "units": 5},
                ],
            }
        }

    def test_saved_and_lost_are_relative_to_baseline(self) -> None:
        rows = SD.density_rows(self._payload()["per_question"])["cand"]
        q1 = [r for r in rows if r["question_id"] == "q1"][0]
        q2 = [r for r in rows if r["question_id"] == "q2"][0]
        assert q1["saved"]["any_gold"] is True
        assert q1["saved"]["covered"] is False, "q1 的 covered 在候选里仍是假，不算救"
        assert q2["lost"]["any_gold"] is True
        assert q2["saved"]["any_gold"] is False

    def test_density_comes_from_boosted_over_units(self) -> None:
        rows = SD.density_rows(self._payload()["per_question"])["cand"]
        q1 = [r for r in rows if r["question_id"] == "q1"][0]
        assert q1["signal_boosted"] == 3
        assert q1["context_units"] == 2
        assert q1["density"] == 1.5

    def test_single_tag_produces_no_table(self) -> None:
        """没有基线就没法说「救」，宁可不产表也不要拿自己跟自己比。"""
        assert SD.density_rows({"only": []}) == {}


# ---------------------------------------------------------------- 真实数据


class TestRealReport:
    """对真实输入跑一遍，数字必须与 Lead 2026-11-05 的表逐格对上。"""

    @pytest.fixture(scope="class")
    @classmethod
    def report(cls) -> dict:
        payload = json.loads(SOURCE.read_text("utf-8"))
        return SD.build(payload)

    def test_report_declares_both_calibers(self, report: dict) -> None:
        assert report["calibers"] == list(SD.CALIBERS_DENSITY)
        assert len(report["calibers"]) == 2, (
            "只声明一种分母的报告无法回答「密度效应是否被没加成的题稀释」"
        )

    def test_bucket_counts_match_the_lead_table(self, report: dict) -> None:
        """`boost_applied_only` 口径下 0-1/1-2/2-4/4+ 的 n 必须是 38/41/21/35。"""
        table = report["bucket_summary"]["cooc0.1"]["boost_applied_only"]
        got = {cell["bucket"]: cell["n"] for cell in table}
        assert got == {"0-1": 38, "1-2": 41, "2-4": 21, "4+": 35}, (
            f"分桶与 Lead 2026-11-05 的表不一致：{got}；"
            "合计应为 135（真正被加成过的题数），不是 350 也不是 141"
        )

    def test_the_4plus_bucket_gains_nothing(self, report: dict) -> None:
        """整条门的结论就这一句：密度 4+ 的 35 题，三个指标净增益都是 0。"""
        table = report["bucket_summary"]["cooc0.1"]["boost_applied_only"]
        cell = [c for c in table if c["bucket"] == "4+"][0]
        assert cell["n"] == 35
        for metric in SD.DENSITY_METRICS:
            assert cell[metric]["net"] == 0, f"{metric} 在 4+ 桶上不该有净增益"

    def test_gain_set_difference_is_the_one_known_question(self, report: dict) -> None:
        """三个召回指标的增益差集只有 `q-tc-000023` 一题。"""
        rels = report["gain_set_relations"]
        assert rels, "三个并列指标必须产出包含关系表"
        only = {tuple(r["only_a"]) for r in rels}
        assert ("q-tc-000023",) in only

    def test_report_passes_its_own_self_check(self, report: dict) -> None:
        assert SD.self_check(report) == []


# ---------------------------------------------------------------- 门本身能红


class TestSelfCheckCatchesProblems:
    """`self_check()` 必须真能判出问题——否则它只是装饰。"""

    def _good(self) -> dict:
        return SD.build(json.loads(SOURCE.read_text("utf-8")))

    def test_self_check_catches_single_caliber_report(self) -> None:
        """砍掉一个口径：结构还能解析、密度三件套也齐，必须仍然判不合规。

        注意这条**不是**在测「声明的口径名单」那一条判据——`self_check()` 里还有
        「桶表每个口径都得在」那条会兜住。专门测声明名单的是下面
        `test_self_check_catches_wrong_caliber_declaration`。
        """
        report = self._good()
        report["calibers"] = ["all_questions"]
        report["bucket_summary"] = {
            tag: {"all_questions": by_caliber["all_questions"]}
            for tag, by_caliber in report["bucket_summary"].items()
        }
        problems = SD.self_check(report)
        assert problems, "只给一种分母必须被判不合规"
        assert any("口径" in p for p in problems)

    def test_self_check_catches_wrong_caliber_declaration(self) -> None:
        """桶表两口径齐全，只有 `calibers` 那行声明少了一个。

        这条存在的理由：上一条会被「桶表缺口径」兜住，于是把声明检查删掉它照样绿
        ——反向实测时那一笔没变红，才补出这条。**反向实测没变红不等于门是坏的，
        等于这条用例测的不是那个判据。**
        """
        report = self._good()
        report["calibers"] = ["all_questions"]          # 声明少一个
        # 桶表保持两口径齐全 → 只有「声明名单」那条判据能发现
        problems = SD.self_check(report)
        assert any("声明的口径" in p for p in problems), (
            f"声明名单少了一个口径，必须被点名；实际 problems={problems}"
        )

    def test_self_check_catches_missing_relation_table(self) -> None:
        report = self._good()
        for row in report["parallel_recall"]:
            row.pop("relation_table", None)
        problems = SD.self_check(report)
        assert any("并列召回" in p for p in problems)

    def test_self_check_catches_missing_provenance(self) -> None:
        report = self._good()
        report.pop("git_commit")
        problems = SD.self_check(report)
        assert any("七字段" in p for p in problems)


# ---------------------------------------------------------------- 命令行


class TestCli:
    """退出码：0 合规 / 1 不合规或与磁盘不一致 / 2 输入不可用。"""

    def _run(self, *args: str):
        import subprocess

        from arkrag import subproc

        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            capture_output=True, text=True, encoding=subproc.PIPE_ENCODING,
            errors="replace", cwd=str(ROOT), timeout=180,
            env=subproc.child_env(),
        )

    def test_written_report_is_consistent_with_disk(self) -> None:
        """仓库里已提交的报告必须与当前数据一致——否则它描述的不是现在。"""
        result = self._run("--check")
        assert result.returncode == 0, (
            f"已提交的 data/signal_density_audit.json 与当前输入不一致："
            f"{result.stdout} {result.stderr}；请重跑 py -3 scripts/signal_density_audit.py"
        )

    def test_check_on_a_missing_report_is_nonzero(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope.json"
        result = self._run("--check", "--out", str(missing))
        assert result.returncode == 1
        assert "不存在" in (result.stdout + result.stderr)

    def test_missing_input_is_exit_2(self, tmp_path: Path) -> None:
        out = tmp_path / "x.json"
        result = self._run("--in", str(tmp_path / "no_such_input.json"), "--out", str(out))
        assert result.returncode == 2
        assert not out.exists(), "输入不可用时不许写出半成品报告"
