"""`scripts/needs_extra_citation_probe.py` 的门：引用落点观测必须可复算。

## 为什么这条要单独立一个测试文件

`docs/p1c_漏标候选判定表.md` 第十节说「frag 6 条里有 3 条 required 行已被引用」，
这句话**不是**从 gold 与块正文推出来的——它来自一次真跑管线，看引用实际落在
哪一行。2026-11-05 第一次是在 `%TEMP%` 里的临时脚本算的：临时文件既不入库、
也不可复现，等于「只有作者知道」。所以搬成入库脚本 + 这里的门。

## 为什么一半用例是喂合成数据

第一版把每条用例都写成「读已写出的 JSON 然后断言」。反向实测时发现：**改坏脚本
实现不会让它们变红**——JSON 是上一轮跑出来的，脚本改了但没重算，用例读到的还是
旧数据。四笔反向实测里有两笔因此「没变红」，那是用例的问题不是门的问题。

所以「分群规则」与「四格统计」这两条判据被抽成模块级纯函数
（`group_of()` / `tally_rows()`），用例直接喂合成输入验证它们。
这样一来，改坏实现就必然会红，与磁盘上的 JSON 无关。

## 这道门挡的四件事

1. **两个布尔不许合成一个。** `required_line_cited`（引用了 gold 标的那一行）与
   `carrier_lines_cited`（引用了承载答案的另一行）回答两个不同问题。合成
   「覆盖到了没有」，就会把「引错了行」与「没引」混成一种失败。
2. **`frag` / `none` 的分界必须按「required 行有没有被抽成断言」判，不许按
   「答案在不在那一行」判。** 后者是 `needs_extra` 的定义，用它当分界线会让
   `frag` 群恒空——一个恒空的群看起来像结论，实际是判据与定义重复。
3. **四格必须互斥且覆盖全部。**「两行都被引用」是第四格，最容易漏。漏掉它，
   三个数相加比总数多，而多出来的恰好是「看起来最正常」的那种行。
4. **`--check` 不许把溯源字段算进比较。** 它们每次运行都合法地变，算进去这道
   命令就永远红，那道门等于没有。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "needs_extra_citation_probe.py"
DATA = ROOT / "data" / "p1c_needs_extra_citation.json"

sys.path.insert(0, str(ROOT / "src"))
from arkrag import subproc  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location("needs_extra_probe_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def probe():
    return _load()


@pytest.fixture(scope="module")
def payload():
    assert DATA.exists(), "先跑 py -3 scripts/needs_extra_citation_probe.py"
    return json.loads(DATA.read_text("utf-8"))


def _assertion(aid: str, start: int, end: int) -> dict:
    return {"assertion_id": aid, "evidence_start": start, "evidence_end": end}


class TestGroupOfIsAPureFunction:
    """分群只看「required 行有没有被抽成断言」，与答案位置无关。"""

    def test_an_assertion_inside_the_line_makes_it_frag(self, probe) -> None:
        group, ids = probe.group_of((100, 200), [_assertion("a-1", 110, 150)])
        assert group == "frag"
        assert ids == ["a-1"]

    def test_no_assertion_inside_the_line_is_none(self, probe) -> None:
        group, ids = probe.group_of((100, 200), [_assertion("a-1", 300, 350)])
        assert group == "none"
        assert ids == []

    def test_an_assertion_spanning_past_the_line_does_not_count(self, probe) -> None:
        """跨出行外的断言不算「落在这一行」——否则「这一行被切碎了」的判断会失真。"""
        group, ids = probe.group_of((100, 200), [_assertion("a-1", 150, 260)])
        assert group == "none"

    def test_missing_span_is_none_not_an_exception(self, probe) -> None:
        assert probe.group_of(None, [_assertion("a-1", 0, 10)]) == ("none", [])

    def test_the_boundary_does_not_use_the_verdict(self, probe) -> None:
        """反例自检：若按「答案在不在 required 行」分群，frag 群会恒空。

        构造一行：required 区间内**有**断言（应为 frag），但答案在块里另一处。
        按答案位置判就会得到 none —— 这正是第一版的分界线错误。
        """
        assertions = [_assertion("a-on-line", 110, 150), _assertion("a-carrier", 400, 450)]
        group, ids = probe.group_of((100, 200), assertions)
        assert group == "frag", "required 行内有断言就必须是 frag，与答案在哪无关"
        assert ids == ["a-on-line"]

    def test_a_row_with_assertions_but_all_outside_is_none(self, probe) -> None:
        group, ids = probe.group_of((100, 200), [_assertion("a-1", 0, 50),
                                                 _assertion("a-2", 210, 260)])
        assert group == "none"
        assert ids == []


class TestTallyRowsIsDisjointAndComplete:
    """四格互斥且覆盖全部——漏掉「两行都有」就是「同一件事算了两遍」。"""

    @staticmethod
    def _row(required: bool, carrier: bool) -> dict:
        return {"required_line_cited": required, "any_carrier_cited": carrier,
                "question_id": "q", "group": "frag", "split": "dev"}

    def test_the_four_cells_sum_to_n(self, probe) -> None:
        rows = [self._row(True, True), self._row(True, False),
                self._row(False, True), self._row(False, False)]
        block = probe.tally_rows(rows)
        assert block["n"] == 4
        assert block["required_and_carrier"] == 1
        assert block["required_cited_only"] == 1
        assert block["carrier_cited_only"] == 1
        assert block["neither_cited"] == 1
        assert (block["required_cited_only"] + block["required_and_carrier"]
                + block["carrier_cited_only"] + block["neither_cited"]) == block["n"]

    def test_a_row_cited_on_both_lines_lands_in_the_both_cell(self, probe) -> None:
        block = probe.tally_rows([self._row(True, True)])
        assert block["required_and_carrier"] == 1
        assert block["required_cited_only"] == 0

    def test_required_line_cited_is_the_union_not_a_cell(self, probe) -> None:
        """`required_line_cited` 是命中数（含两行都有的），不是互斥格。

        这两个字段名容易混：互斥格是 `required_cited_only`，命中数是
        `required_line_cited`。混了就会把「两行都有」算两遍。
        """
        rows = [self._row(True, True), self._row(True, False)]
        block = probe.tally_rows(rows)
        assert block["required_line_cited"] == 2
        assert block["required_cited_only"] == 1
        assert block["required_and_carrier"] == 1

    def test_empty_input_is_all_zero(self, probe) -> None:
        block = probe.tally_rows([])
        assert block["n"] == 0
        assert sum(block[k] for k in ("required_cited_only", "required_and_carrier",
                                      "carrier_cited_only", "neither_cited")) == 0


class TestObservationsAreTwoSeparateFacts:
    """已写出的 JSON 里，两个布尔必须各自存在且能对上汇总数。"""

    def test_every_row_has_both_flags(self, payload) -> None:
        for row in payload["detail"]:
            assert isinstance(row["required_line_cited"], bool)
            assert isinstance(row["carrier_lines_cited"], list)

    def test_the_four_cells_partition_every_row(self, payload, probe) -> None:
        """四个格子互斥且覆盖全部 —— 正是「同一件事被算两遍」的防线。

        「required 行被引用」与「承载行被引用」**可以同时成立**（一次回答引了多行），
        所以三个格子（只有前者 / 只有后者 / 都没有）加不出总数。第一版就是这么写的，
        加起来 17 ≠ 16，红在这里。这条用例保留下来，因为第四格（两行都有）恰恰
        最容易漏掉。
        """
        rows = payload["detail"]
        block = probe.tally_rows(rows)
        assert (block["required_cited_only"] + block["required_and_carrier"]
                + block["carrier_cited_only"] + block["neither_cited"]) == len(rows)
        assert block["required_and_carrier"] == payload["both_lines_cited_n"]
        assert block["required_cited_only"] == payload["required_cited_but_carrier_not_n"]
        assert block["carrier_cited_only"] == payload["carrier_cited_but_required_not_n"]
        assert block["neither_cited"] == payload["neither_cited_n"]
        assert (block["required_cited_only"] + block["required_and_carrier"]
                == payload["if_appended_would_cover_n"])

    def test_tallies_are_also_disjoint(self, payload) -> None:
        """每个拆分格子内部也要满足四格相加等于 n。"""
        for name in ("by_split", "by_group", "by_split_and_group"):
            for key, block in payload[name].items():
                assert (block["required_cited_only"] + block["required_and_carrier"]
                        + block["carrier_cited_only"] + block["neither_cited"]
                        == block["n"]), f"{name}.{key} 四格相加不等于 n"


class TestCommittedRowsMatchThePureFunction:
    """磁盘上的分群结论必须与纯函数重算一致（防「两份实现各说各话」）。"""

    def test_every_row_group_matches_recompute(self, payload, probe) -> None:
        sys.path.insert(0, str(ROOT / "scripts"))
        import p1c_leak_audit as P1C  # noqa: E402

        corpus = P1C.load_corpus()
        gold_v1 = json.loads(P1C.GOLD_V1.read_text("utf-8"))
        by_id = {str(it["question_id"]): it for it in gold_v1["items"]}
        for row in payload["detail"]:
            item = by_id[row["question_id"]]
            chunk_id = row["chunk_id"]
            text = str((corpus["chunks"].get(chunk_id) or {}).get("text") or "")
            evidence = str((item.get("required_citations") or [{}])[0].get("evidence") or "")
            span = P1C._line_span(text, evidence) if evidence else None
            group, ids = probe.group_of(
                (span[0], span[1]) if span else None,
                corpus["by_chunk"].get(chunk_id) or [])
            assert group == row["group"], f"{row['question_id']} 分群与重算不一致"
            assert ids == row["assertions_on_required_line"]

    def test_frag_rows_have_an_assertion_on_the_required_line(self, payload) -> None:
        frag = [r for r in payload["detail"] if r["group"] == "frag"]
        assert frag, "必须有 frag 行，否则这条判据没被验证到"
        for row in frag:
            assert row["assertions_on_required_line"]

    def test_none_rows_have_no_assertion_on_the_required_line(self, payload) -> None:
        none = [r for r in payload["detail"] if r["group"] == "none"]
        assert none
        for row in none:
            assert row["assertions_on_required_line"] == []


class TestSplitAndGroupTalliesAreBothReported:
    """两个维度都要给，否则「dev 13 条里 frag 6 条」对不上账。"""

    def test_by_split_covers_dev_and_holdout(self, payload) -> None:
        assert set(payload["by_split"]) == {"dev", "holdout"}

    def test_by_group_covers_frag_and_none(self, payload) -> None:
        assert set(payload["by_group"]) == {"frag", "none"}

    def test_by_split_and_group_partitions_every_row(self, payload) -> None:
        cells = payload["by_split_and_group"]
        assert set(cells) == {"dev/frag", "dev/none", "holdout/frag", "holdout/none"}
        assert sum(c["n"] for c in cells.values()) == payload["targets_n"]
        for split in ("dev", "holdout"):
            assert (cells[f"{split}/frag"]["n"] + cells[f"{split}/none"]["n"]
                    == payload["by_split"][split]["n"])
        for group in ("frag", "none"):
            assert (cells[f"dev/{group}"]["n"] + cells[f"holdout/{group}"]["n"]
                    == payload["by_group"][group]["n"])

    def test_dev_side_matches_the_documented_six_and_seven(self, payload) -> None:
        """`docs/p1c_漏标候选判定表.md` 第十节写的是 dev 侧 frag 6 / none 7。

        这两条数是文档里的承诺，改了就必须同步改文档，所以钉在这里。
        """
        cells = payload["by_split_and_group"]
        assert cells["dev/frag"]["n"] == 6
        assert cells["dev/none"]["n"] == 7


class TestCheckIgnoresVolatileProvenanceFields:
    """`--check` 比的是「引用落点变了吗」，不是「文件字节一样吗」。"""

    def test_observations_strip_volatile_fields(self, probe, payload) -> None:
        stripped = probe._observations(payload)
        for name in ("generated_at", "git_commit", "code_sha256_16",
                     "dataset", "inputs", "environment", "generation_id"):
            assert name not in stripped, f"{name} 每跑一次都变，不该参与比较"
        assert "detail" in stripped and "by_split" in stripped

    def test_check_passes_on_the_committed_file(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--check"],
            cwd=str(ROOT), capture_output=True, text=True,
            encoding=subproc.PIPE_ENCODING, errors="replace",
            timeout=1800, env=subproc.child_env(),
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "一致" in proc.stdout


class TestRowsCarryRecomputableInputs:
    """每条都要能被第三方用同样命令复算。"""

    def test_every_row_names_its_chunk_and_required_line(self, payload) -> None:
        for row in payload["detail"]:
            assert row["chunk_id"] and row["question_id"]
            assert "required_line" in row
            assert isinstance(row["emitted_n"], int)
            assert isinstance(row["units"], int)

    def test_probe_declares_where_chunks_came_from(self, payload) -> None:
        """块正文必须走 resolve_chunks，不许读语料根。"""
        assert payload["chunks_source_api"] == "pipeline.resolve_chunks(gen, manifest)"
        assert payload["chunks_loaded"] > 0

    def test_probe_carries_the_seven_provenance_fields(self, payload) -> None:
        from arkrag import provenance
        res = provenance.verify(payload)
        assert res["missing"] == [], f"缺字段：{res['missing']}"
        assert res["verdict"] == provenance.PASS
