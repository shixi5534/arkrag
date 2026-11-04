# -*- coding: utf-8 -*-
"""`scripts/k3_thresholds.py` 的自检：**跨了阈值就必须红**。

K3 第 5 条「超阈值必须人工审批」此前一条从未被执行过的判据（两份 md 里
「人工审批」0 处）。一条从未失败过的门和没有门是一回事，所以这里对每一处
判定构造「本该拦住」的场景，断言它确实红，并且 red 的理由说的是那件事。

## 为什么用合成数据而不是真实 320 题

真跑两代召回要 147 秒，而归因脚本 `scripts/k3_attribution.py` 要 12 秒。
本文件的判据输入全部是**合成的小数据**（4 题、2 个世代目录），所以它是毫秒级的。
真实读数由手工跑 `py -3 scripts/k3_thresholds.py` 的实测输出负责（进汇报与台账），
不在这里假装。

## 这一组用例真正固定的四件事

1. 两套口径（A 丢弃 null / B 输 0）**必须被显式分开**，RR 分叉而 R@5、R@20 不分叉；
   若 pairs 文件的 null 形状变了导致 R@5/R@20 也分叉，门必须硬错而不是继续判。
2. 阈值**只从 `data/k3_thresholds.json` 来**：同一份实测，把阈值放宽就不红、
   收紧就红——证明门读文件，不读代码里的常量，也就不会恒真。
3. 审批记录的「有效」是真的会否掉的：占位符审批人、measured 与实测不符、
   缺字段、拒绝结论、空理由，每一种都单独能被判无效。
4. 红线（未解释差异数）**不接受审批**：归因能力跟不上改动时签字不能替代归因。

## 与卫生门的关系

本文件进入 `tests/test_suite_hygiene.py` 的 `TEST_FILES` 后会让四个 parametrize
各多一项（+4），加上本文件用例数，`N` 需要由 Lead 同步上调。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_gate():
    spec = importlib.util.spec_from_file_location(
        "k3_thresholds_under_test", ROOT / "scripts" / "k3_thresholds.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gate = _load_gate()


# ---------------------------------------------------------------- 合成数据


def _rec(r5, r20, rr=None, category="fact"):
    return {"r5": r5, "r20": r20, "rr": rr, "category": category}


def _pairs(old_rows, new_rows, gens=("g-old", "g-new")):
    """把两份逐题记录拼成 `data/k3_recall_pairs.json` 的形状。"""
    qids = [f"q{i:03d}" for i in range(len(old_rows))]
    return {
        "generations": {"old": gens[0], "new": gens[1]},
        "questions": qids,
        "old": dict(zip(qids, old_rows)),
        "new": dict(zip(qids, new_rows)),
    }


def _flat_pairs(r5_old=0.5, r5_new=0.5, r20_old=0.7, r20_new=0.7,
                rr_old=0.4, rr_new=0.4, n=4):
    """n 题、两代完全同值的基线数据（Δ 全 0）。"""
    old = [_rec(r5_old, r20_old, rr_old) for _ in range(n)]
    new = [_rec(r5_new, r20_new, rr_new) for _ in range(n)]
    return _pairs(old, new)


def _generations(root: Path, old=("g-old", 1000, 5000, 100_000),
                 new=("g-new", 1000, 5000, 100_000)):
    """造两代最小世代目录（只含 manifest.json 与 index/lexical.json）。"""
    for gid, entities, edges, lex_bytes in (old, new):
        gdir = root / gid
        (gdir / "index").mkdir(parents=True, exist_ok=True)
        (gdir / "manifest.json").write_text(
            json.dumps({"generation_id": gid, "entity_count": entities,
                        "edge_count": edges}, ensure_ascii=False),
            encoding="utf-8",
        )
        (gdir / "index" / "lexical.json").write_bytes(b"x" * lex_bytes)
    return root


def _row(metric="r20_delta", threshold=-0.02, direction="lower_is_worse", **over):
    row = {"metric": metric, "label": metric, "unit": "比率（0-1）",
           "direction": direction, "threshold": threshold,
           "caliber": "A=B", "owner": "tester", "reason": "测试用"}
    row.update(over)
    return row


def _thresholds(delta, absolute=None):
    return {"schema": "k3-thresholds-1", "delta": delta,
            "absolute": absolute or []}


def _approval(metric="r20_delta", measured=-0.05, threshold=-0.02,
              approved_by="Lenovo.ysc", approved_at="2026-11-04",
              decision="approved", rationale="确认是分词归一化导致的，非召回退化"):
    return {"metric": metric, "measured": measured, "threshold": threshold,
            "approved_by": approved_by, "approved_at": approved_at,
            "decision": decision, "rationale": rationale}


def _approvals(*records):
    return {"schema": "k3-approvals-1", "approvals": list(records)}


# ---------------------------------------------------------------- 口径


class TestCalibers:
    def test_caliber_a_drops_nulls_caliber_b_zero_fills(self):
        """同一份数据：口径 A 丢弃 null（RR n=2），口径 B 输 0（RR n=4）。"""
        old = [_rec(1.0, 1.0, 0.5), _rec(0.0, 1.0, None),
               _rec(0.0, 1.0, None), _rec(0.0, 1.0, None)]
        new = [_rec(1.0, 1.0, 0.5), _rec(0.0, 1.0, None),
               _rec(0.0, 1.0, None), _rec(0.0, 1.0, None)]
        m = gate.compute_recall_metrics(_pairs(old, new))
        assert m["n_graded"] == 4
        assert m["calibers"]["A"]["n"]["rr"] == 1
        assert m["calibers"]["B"]["n"]["rr"] == 4
        # 口径 B 把三个未命中算成 0 → 均值明显更低
        assert m["calibers"]["B"]["old"]["rr"] < m["calibers"]["A"]["old"]["rr"]

    def test_r5_and_r20_identical_across_calibers(self):
        old = [_rec(1.0, 1.0, 0.5), _rec(0.0, 0.0, None), _rec(1.0, 0.0, 0.25)]
        new = [_rec(1.0, 1.0, 1.0), _rec(0.0, 1.0, None), _rec(0.0, 0.0, 0.25)]
        m = gate.compute_recall_metrics(_pairs(old, new))
        for metric in ("r5", "r20"):
            assert m["calibers"]["A"]["delta"][metric] == m["calibers"]["B"]["delta"][metric]

    def test_r5_diverging_across_calibers_is_a_hard_error(self):
        """r5 为 null（r20 非 null）时口径 A 会少一题 → 两套口径分叉 → 硬错。"""
        old = [_rec(None, 1.0, 0.5), _rec(1.0, 1.0, 0.5)]
        new = [_rec(0.0, 1.0, 0.5), _rec(0.0, 1.0, 0.5)]
        with pytest.raises(gate.GateError) as exc:
            gate.compute_recall_metrics(_pairs(old, new))
        assert "口径 A/B" in str(exc.value)

    def test_no_graded_question_is_a_hard_error(self):
        old = [_rec(None, None, None)]
        new = [_rec(None, None, None)]
        with pytest.raises(gate.GateError) as exc:
            gate.compute_recall_metrics(_pairs(old, new))
        assert "无可比题" in str(exc.value)


# ---------------------------------------------------------------- 阈值判定


class TestBreaches:
    def test_lower_is_worse(self):
        row = _row(threshold=-0.02)
        assert gate.breaches(row, -0.0201) is True
        assert gate.breaches(row, -0.0200) is False
        assert gate.breaches(row, -0.0031) is False
        assert gate.breaches(row, +0.5) is False

    def test_higher_is_worse(self):
        row = _row("entity_count_rise_pct", threshold=5.0, direction="higher_is_worse")
        assert gate.breaches(row, 5.01) is True
        assert gate.breaches(row, 5.0) is False
        assert gate.breaches(row, 0.015) is False

    def test_red_line(self):
        row = _row("unexplained_differences", threshold=0, direction="red_line")
        assert gate.breaches(row, 1) is True
        assert gate.breaches(row, 0) is False

    def test_unknown_direction_is_a_hard_error(self):
        row = _row(direction="whatever")
        with pytest.raises(gate.GateError) as exc:
            gate.breaches(row, 0.0)
        assert "direction" in str(exc.value)

    def test_null_threshold_is_a_hard_error(self):
        row = _row(threshold=None)
        with pytest.raises(gate.GateError) as exc:
            gate.breaches(row, 0.0)
        assert "threshold" in str(exc.value)


# ---------------------------------------------------------------- 审批有效性


class TestApprovalValidity:
    def test_valid_record_has_no_problems(self):
        row = _row()
        assert gate.validate_approval(_approval(), row, -0.05) == []

    @pytest.mark.parametrize("who", ["", "TBD", "agent", "auto", "system", "自动化", "n/a"])
    def test_placeholder_approver_is_invalid(self, who):
        """审批是人做的动作，程序不能自批。"""
        problems = gate.validate_approval(_approval(approved_by=who), _row(), -0.05)
        assert any("占位符" in p for p in problems)

    def test_measured_mismatch_is_invalid(self):
        problems = gate.validate_approval(_approval(measured=-0.01), _row(), -0.05)
        assert any("不符" in p for p in problems)

    def test_measured_within_tolerance_is_valid(self):
        rec = _approval(measured=-0.05 + 1e-12)
        assert gate.validate_approval(rec, _row(), -0.05) == []

    def test_threshold_mismatch_is_invalid(self):
        """审批针对的线必须就是当前登记的那条线。"""
        problems = gate.validate_approval(_approval(threshold=-0.5), _row(), -0.05)
        assert any("threshold" in p for p in problems)

    @pytest.mark.parametrize("field", ["metric", "measured", "threshold",
                                       "approved_by", "approved_at",
                                       "decision", "rationale"])
    def test_each_required_field_is_required(self, field):
        rec = _approval()
        rec.pop(field)
        problems = gate.validate_approval(rec, _row(), -0.05)
        assert any("缺字段" in p and field in p for p in problems)

    @pytest.mark.parametrize("stamp", ["", "TBD", "2026-13-40", "昨天", None])
    def test_unparseable_date_is_invalid(self, stamp):
        problems = gate.validate_approval(_approval(approved_at=stamp), _row(), -0.05)
        assert any("approved_at" in p for p in problems)

    @pytest.mark.parametrize("decision", ["rejected", "拒绝", "maybe"])
    def test_rejection_or_vague_decision_is_invalid(self, decision):
        problems = gate.validate_approval(_approval(decision=decision), _row(), -0.05)
        assert any("decision" in p for p in problems)

    def test_empty_rationale_is_invalid(self):
        problems = gate.validate_approval(_approval(rationale="   "), _row(), -0.05)
        assert any("rationale" in p for p in problems)


# ---------------------------------------------------------------- 单行判定


class TestJudge:
    def test_not_breached_needs_no_approval(self):
        """没跨阈值：不需要审批也是 PASS——这是正常态，不是漏洞。"""
        res = gate.judge(_row(), -0.0031, [])
        assert res["ok"] is True
        assert res["breached"] is False
        assert res["approvals_seen"] == 0
        assert "不需要审批" in res["detail"]

    def test_breached_without_approval_fails_and_names_the_metric(self):
        res = gate.judge(_row(), -0.05, [])
        assert res["ok"] is False
        assert res["breached"] is True
        assert "r20_delta" in res["detail"]
        assert "-0.050000" in res["detail"]
        assert "-0.020000" in res["detail"]
        assert "审批记录" in res["detail"]

    def test_breached_with_placeholder_approver_fails(self):
        res = gate.judge(_row(), -0.05, [_approval(approved_by="auto")])
        assert res["ok"] is False
        assert res["approvals_seen"] == 1
        assert "无效" in res["detail"]

    def test_breached_with_mismatched_measured_fails(self):
        res = gate.judge(_row(), -0.05, [_approval(measured=-0.021)])
        assert res["ok"] is False
        assert "不符" in res["detail"]

    def test_breached_with_valid_approval_passes(self):
        res = gate.judge(_row(), -0.05, [_approval()])
        assert res["ok"] is True
        assert res["approval"]["approved_by"] == "Lenovo.ysc"

    def test_approval_for_another_metric_does_not_count(self):
        res = gate.judge(_row(), -0.05, [_approval(metric="r5_delta")])
        assert res["ok"] is False
        assert res["approvals_seen"] == 0

    def test_red_line_cannot_be_approved(self):
        """红线即使有完整有效的人工审批也不放行。"""
        row = _row("unexplained_differences", threshold=0, direction="red_line")
        res = gate.judge(row, 3.0, [_approval(metric="unexplained_differences",
                                              measured=3.0, threshold=0)])
        assert res["ok"] is False
        assert "红线" in res["detail"]


# ---------------------------------------------------------------- 结构量 / 读入


class TestStructureAndLoad:
    def test_structure_delta_pct(self, tmp_path):
        root = _generations(tmp_path, old=("g-old", 1000, 5000, 100_000),
                            new=("g-new", 990, 4900, 75_000))
        st = gate.compute_structure_metrics(root, {"old": "g-old", "new": "g-new"})
        assert st["delta_pct"]["entity_count"] == pytest.approx(-1.0)
        assert st["delta_pct"]["edge_count"] == pytest.approx(-2.0)
        assert st["delta_pct"]["lexical_bytes"] == pytest.approx(-25.0)

    def test_missing_generation_id_is_a_hard_error(self, tmp_path):
        with pytest.raises(gate.GateError) as exc:
            gate.compute_structure_metrics(tmp_path, {"old": "", "new": "g-new"})
        assert "世代 id" in str(exc.value)

    def test_missing_lexical_index_is_a_hard_error(self, tmp_path):
        gdir = tmp_path / "g-old"
        gdir.mkdir()
        (gdir / "manifest.json").write_text(
            json.dumps({"entity_count": 1, "edge_count": 1}), encoding="utf-8")
        (tmp_path / "g-new").mkdir()
        (tmp_path / "g-new" / "manifest.json").write_text(
            json.dumps({"entity_count": 1, "edge_count": 1}), encoding="utf-8")
        with pytest.raises(gate.GateError) as exc:
            gate.compute_structure_metrics(tmp_path, {"old": "g-old", "new": "g-new"})
        assert "词法索引" in str(exc.value)

    def test_missing_file_is_a_hard_error_not_a_pass(self, tmp_path):
        with pytest.raises(gate.GateError) as exc:
            gate.load_json(tmp_path / "nope.json", "阈值表")
        assert "不存在" in str(exc.value)

    def test_broken_json_is_a_hard_error(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        with pytest.raises(gate.GateError) as exc:
            gate.load_json(bad, "阈值表")
        assert "合法 JSON" in str(exc.value)

    def test_missing_attribution_script_is_a_hard_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gate, "ATTRIBUTION", tmp_path / "gone.py")
        with pytest.raises(gate.GateError) as exc:
            gate.read_unexplained()
        assert "归因脚本不存在" in str(exc.value)


# ---------------------------------------------------------------- absolute 只申报


class TestAbsoluteDeclaredOnly:
    def test_unmet_absolute_is_reported_unmet_not_pass(self):
        """绝对门未达标必须写「未达标」，不许因为「K3 不判」而写成通过。"""
        pairs = _flat_pairs(r20_old=0.70, r20_new=0.70)
        recall = gate.compute_recall_metrics(pairs)
        row = {"metric": "recall_at_20", "label": "Recall@20", "direction": "at_least",
               "threshold": 0.9, "measured_from": "r20", "k3_role": "申报，不判"}
        res = gate.judge_absolute(row, recall)
        assert res["declared"] is True
        assert res["met"] is False
        assert "未达标" in res["detail"]

    def test_missing_input_says_not_evaluated(self):
        row = {"metric": "citation_localization", "threshold": 0.98,
               "measured_from": None, "direction": "at_least"}
        res = gate.judge_absolute(row, gate.compute_recall_metrics(_flat_pairs()))
        assert res["met"] is None
        assert "不假装通过" in res["detail"]


# ---------------------------------------------------------------- 端到端（文件是唯一事实源）


def _write(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return path


def _run(tmp_path, thresholds, pairs, approvals, unexplained=0, gens=None):
    """跑一次 main()，全部输入走临时文件；返回 (退出码, 报告 dict or None)。"""
    t = _write(tmp_path / "thresholds.json", thresholds)
    p = _write(tmp_path / "pairs.json", pairs)
    a = _write(tmp_path / "approvals.json", approvals)
    g = gens or _generations(tmp_path)
    out = tmp_path / "gate.json"
    code = gate.main(["--thresholds", str(t), "--pairs", str(p),
                      "--approvals", str(a), "--gens", str(g),
                      "--out", str(out), "--unexplained", str(unexplained)])
    report = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
    return code, report


class TestEndToEnd:
    def test_no_breach_passes(self, tmp_path):
        pairs = _flat_pairs(r20_old=0.70, r20_new=0.6989)
        code, report = _run(tmp_path, _thresholds([_row()]), pairs, _approvals())
        assert code == 0
        assert report["all_passed"] is True
        assert report["failed_metrics"] == []

    def test_breach_without_approval_exits_1(self, tmp_path):
        pairs = _flat_pairs(r20_old=0.75, r20_new=0.70)   # Δ = -0.05，跨 -0.02
        code, report = _run(tmp_path, _thresholds([_row()]), pairs, _approvals())
        assert code == 1
        assert report["all_passed"] is False
        assert report["failed_metrics"] == ["r20_delta"]

    def test_breach_with_valid_approval_exits_0(self, tmp_path):
        pairs = _flat_pairs(r20_old=0.75, r20_new=0.70)
        code, report = _run(tmp_path, _thresholds([_row()]), pairs,
                            _approvals(_approval(measured=-0.05)))
        assert code == 0
        assert report["checks"][0]["approval"]["approved_by"] == "Lenovo.ysc"

    def test_breach_with_placeholder_approval_exits_1(self, tmp_path):
        pairs = _flat_pairs(r20_old=0.75, r20_new=0.70)
        code, _ = _run(tmp_path, _thresholds([_row()]), pairs,
                       _approvals(_approval(measured=-0.05, approved_by="auto")))
        assert code == 1

    def test_same_measurement_flips_with_the_threshold_file_only(self, tmp_path):
        """同一份实测：阈值收紧就红、放宽就不红 —— 证明判据来自文件而非代码常量。"""
        pairs = _flat_pairs(r20_old=0.75, r20_new=0.72)   # Δ = -0.03
        tight, _ = _run(tmp_path, _thresholds([_row(threshold=-0.02)]), pairs, _approvals())
        loose, _ = _run(tmp_path, _thresholds([_row(threshold=-0.10)]), pairs, _approvals())
        assert tight == 1
        assert loose == 0

    def test_red_line_breach_is_fatal_even_with_approval(self, tmp_path):
        pairs = _flat_pairs()
        row = _row("unexplained_differences", threshold=0, direction="red_line")
        appr = _approval(metric="unexplained_differences", measured=4.0, threshold=0)
        code, report = _run(tmp_path, _thresholds([row]), pairs,
                            _approvals(appr), unexplained=4)
        assert code == 1
        assert report["failed_metrics"] == ["unexplained_differences"]

    def test_missing_input_exits_2(self, tmp_path):
        pairs = _flat_pairs()
        p = _write(tmp_path / "pairs.json", pairs)
        a = _write(tmp_path / "approvals.json", _approvals())
        out = tmp_path / "gate.json"
        code = gate.main(["--thresholds", str(tmp_path / "absent.json"),
                          "--pairs", str(p), "--approvals", str(a),
                          "--gens", str(_generations(tmp_path)),
                          "--out", str(out), "--unexplained", "0"])
        assert code == 2
        assert not out.exists()

    def test_empty_delta_section_is_a_hard_error(self, tmp_path):
        """没有阈值就谈不上审批 —— 空表不许当「全部通过」。"""
        pairs = _flat_pairs()
        code, report = _run(tmp_path, _thresholds([]), pairs, _approvals())
        assert code == 2
        assert report is None

    def test_output_is_lf_only(self, tmp_path):
        pairs = _flat_pairs()
        _run(tmp_path, _thresholds([_row()]), pairs, _approvals())
        raw = (tmp_path / "gate.json").read_bytes()
        assert b"\r\n" not in raw
        assert raw.endswith(b"\n")

    def test_denominators_are_recorded(self, tmp_path):
        """每一项都要报分母：题数、判据数、审批记录数。"""
        pairs = _flat_pairs(n=6)
        _code, report = _run(tmp_path, _thresholds([_row()]), pairs,
                             _approvals(_approval()), unexplained=0)
        den = report["denominators"]
        assert den["n_common_questions"] == 6
        assert den["n_graded_questions"] == 6
        assert den["n_delta_metrics"] == 1
        assert den["n_approval_records"] == 1
        assert den["n_unexplained"] == 0
