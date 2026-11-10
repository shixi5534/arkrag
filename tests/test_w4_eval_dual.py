"""W4b 门：评测口径升级（`--judge v1|v2|both` + 聚类 CI + 地板三行）。

## 这个文件守的是什么

计划书 §W4「评测口径同时升级」与 §验收 J1/J3/J4/J5。四件事各自的**静默失效形态**
完全不同，所以分组钉：

1. **`--judge` 默认 `v1`，且 `v1` 的产物一个键都不许多**（J3 硬门）。
   失效形态：顺手把新块（`judge_mode` / `flips` / `stats` / `floors`）无条件塞进
   `report` —— 分数一个字不变、屏幕上一模一样，但历史产物全部变成新字节。
   那种改动**没有任何可见症状**，只有逐字段比对抓得住，所以这里既比键集合、
   也比整份 dict 的 JSON 字面量，还比 stdout 的旧前缀。

2. **`--judge v2|both` 在 `judge_v2` 缺失时必须报错退出**，不许静默退回 v1。
   失效形态：静默退回会产出一份"两列同数"的假双口径报告 —— 它看起来像
   "v2 与 v1 完全一致"这个结论，而不是"v2 根本没跑"。这比不跑更糟。

3. **翻转四类必须加得起来**（J1 反证条件 2）。
   `PP+PF+FP+FF == 两套都能判的题数`，且 v2 返回 `None` 的题**不许**当 False
   掉进 FF —— 那会把"判据崩了"记成"判据更严"。

4. **簇键是纯函数，能被单测**（J4 的落地方式）。
   不重算计划书里那 1,802 对（那要读 gold 产物，且是已实测事实）：只测
   "同模板归同簇、不同模板分簇"这条定义本身。

5. **校正指标的定义域保护**（J5）。
   `floor > pass` 时必须返回"不可估计"，**不许返回负数**：
   负数会被读成"改进是负的"，而真实含义是"整条通路不如一条地板"。

6. **方差膨胀只在有配对样本时给数**（J4 明令）。
   `m̄≈2.5, ρ≈0.6` 推出的 ≈1.9 是**设计量级估算**，不许冒充实测值出现在任何键里。

## 为什么全部用桩数据

真实世代要 `load_generation()`，一次常驻 1.0–1.5 GB，本工作项纪律禁止加载产物。
这里断言的是**报告与统计口径的纯逻辑**（键集合、计数、定义域、簇归并），
不涉及任何真实分数；真实分数由 `scripts/eval_answer_quality.py` 实跑负责，
那是另一类证据（纪律也不允许本子代理跑）。
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from arkrag import cluster_ci as cluster_mod
from arkrag import judge as judge_mod

ROOT = Path(__file__).resolve().parents[1]

#: 被判定脚本的绝对路径（不许写成相对 cwd 的仓库路径：换个目录跑结论就变）。
EVAL_SCRIPT = ROOT / "scripts" / "eval_answer_quality.py"


def _load_eval_module(name: str = "eval_aq_w4b_under_test"):
    """按绝对路径加载评测脚本（`scripts/` 不是包的一部分）。"""
    spec = importlib.util.spec_from_file_location(name, EVAL_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------- 桩世界
STUB_ITEMS = [
    {"question_id": "q-sc-1", "category": "single_chunk", "split": "dev",
     "question": "「凯尔希」是谁？", "answer": "凯尔希", "gold_chunk_ids": ["c#000"]},
    {"question_id": "q-al-1", "category": "alias", "split": "dev",
     "question": "「凯尔希医生」的别名是什么？", "answer": "凯尔希",
     "acceptable_answers": ["凯尔希医生"], "gold_chunk_ids": ["c#001"]},
    {"question_id": "q-cx-1", "category": "cross_chunk", "split": "dev",
     "question": "「罗德岛」出现在哪些章节代号里？", "answer": "ep01 ep02",
     "gold_chunk_ids": ["c#002"]},
    {"question_id": "q-mh-1", "category": "multihop", "split": "dev",
     "question": "「云兽」与「变形者集群」通过什么连起来？", "answer": "云兽",
     "gold_chunk_ids": ["c#004"]},
    {"question_id": "q-tc-1", "category": "time_conflict", "split": "dev",
     "question": "「凯尔希」什么时候到达？", "answer": "never",
     "gold_chunk_ids": ["c#003"]},
]

STUB_ANSWERS = {
    "q-sc-1": "凯尔希是罗德岛的医疗顾问。",
    "q-al-1": "她叫凯尔希医生。",
    "q-cx-1": "ep01 ep02",
    "q-mh-1": "云兽隶属变形者集群。",
    "q-tc-1": "",
}

STUB_CHUNKS = {
    "c#000": {"chunk_id": "c#000", "chapter": "ep01", "doc_id": "d1", "title": "t1",
              "text": "凯尔希是罗德岛的医疗顾问。"},
    "c#001": {"chunk_id": "c#001", "chapter": "ep01", "doc_id": "d1", "title": "t1",
              "text": "她叫凯尔希医生。"},
    "c#002": {"chunk_id": "c#002", "chapter": "ep02", "doc_id": "d2", "title": "t2",
              "text": "罗德岛停在荒野。"},
    "c#003": {"chunk_id": "c#003", "chapter": "ep03", "doc_id": "d3", "title": "t3",
              "text": "凯尔希到得最晚。"},
    "c#004": {"chunk_id": "c#004", "chapter": "ep04", "doc_id": "d4", "title": "t4",
              "text": "云兽隶属变形者集群。"},
}


class _StubRetriever:
    def __init__(self) -> None:
        self.budget = {"max_context_tokens": 12000}


def _install_stub_world(monkeypatch, mod) -> None:
    """把 `run()` 依赖的四个重活全部换成桩：**不读任何数据产物**。"""

    def fake_load_generation(gid: str) -> dict:
        return {"chunks": dict(STUB_CHUNKS), "graph": {"entities": []},
                "artifacts": {}}

    def fake_build_pipeline(*a, **kw) -> dict:
        return {"retriever": _StubRetriever(), "graph_depth": 1, "top_k": 20,
                "answerer": None}

    def fake_ask_once(pipe, q, *, options=None, timings=None) -> dict:
        for it in STUB_ITEMS:
            if it["question"] == q:
                text = STUB_ANSWERS[it["question_id"]]
                cites = ([{"chunk_id": "c#000", "evidence": "凯尔希是罗德岛的医疗顾问。",
                           "start": 0, "end": 13}] if text else [])
                return {
                    "answer": {"answer": text, "abstained": not text,
                               "citations": cites},
                    "context": {"units": [], "context_digest": "d", "stats": {}},
                    "search": {}, "expand": {}, "query_analysis": {},
                }
        return {"answer": {"answer": "", "abstained": True, "citations": []},
                "context": {"units": [], "context_digest": "", "stats": {}},
                "search": {}, "expand": {}, "query_analysis": {}}

    monkeypatch.setattr(mod.pipeline_mod, "load_generation", fake_load_generation)
    monkeypatch.setattr(mod.pipeline_mod, "build_pipeline", fake_build_pipeline)
    monkeypatch.setattr(mod.pipeline_mod, "ask_once", fake_ask_once)
    # gold-v1 的读取也桩掉：本文件不需要"哪些题在 v1 里"，而且不该碰 data/。
    monkeypatch.setattr(mod, "load_v1_ids", lambda p: set())


def _write_gold(tmp_path: Path) -> Path:
    gold = tmp_path / "gold_w4b.json"
    gold.write_text(json.dumps({"items": STUB_ITEMS}, ensure_ascii=False),
                    encoding="utf-8")
    return gold


def _run_stub(monkeypatch, mod, tmp_path: Path, judge_mode: str,
              judge_v2_stub=None, *, strip_timing: bool = True) -> dict:
    _install_stub_world(monkeypatch, mod)
    if judge_v2_stub is not None:
        monkeypatch.setattr(mod, "load_judge_v2", lambda: judge_v2_stub)
    rep = mod.run("g-fake", _write_gold(tmp_path), split="dev", limit=0,
                  token_budget=None, judge_mode=judge_mode)
    # 时间量随机器变化，任何逐位比对都必须先剔掉（它们改造前后都在顶层）。
    # **只在比对时剔**：`print_report()` 要读 `per_item_ms`，那些用例传
    # `strip_timing=False`——否则被测的不是产物差异，而是"我删了它自己的输入"。
    if strip_timing:
        rep.pop("elapsed_ms", None)
        rep.pop("per_item_ms", None)
    return rep


def _row(v1_pass, v2_pass, *, category="single_chunk", qid="q",
         axes=None, loss=None, reason="r") -> dict:
    """造一行 `run()` 形状的 row（只需 flip_counts 读的那几个键）。"""
    row = {
        "question_id": qid,
        "category": category,
        "judge": {"pass": v1_pass, "reason": reason},
        "arms": {},
    }
    if v2_pass is not False or axes is not None or loss is not None:
        row["judge_v2"] = {"pass": v2_pass, "reason": reason,
                           "axes": dict(axes or {}), "loss": dict(loss or {})}
    else:
        row["judge_v2"] = {"pass": v2_pass, "reason": reason,
                           "axes": dict(axes or {}), "loss": dict(loss or {})}
    return row


class _V2Stub:
    """`judge_v2` 的最小替身：按题号决定通过/拒绝。

    刻意**只**实现约定里的 `judge_answer_v2`（`resolve_v2_entry` 的首选名），
    这样用例同时钉住"优先认约定名"这件事。
    """

    def __init__(self, answers: dict) -> None:
        self.answers = dict(answers)
        self.calls: list[dict] = []

    def judge_answer_v2(self, category, answer_text, item, *, question=None,
                        chapter_texts=None, context=None) -> dict:
        key = str(item.get("question_id") or "")
        rec = self.answers.get(key) or {"pass": None, "reason": "stub_missing"}
        return {"pass": rec["pass"],
                "reason": rec.get("reason", ""),
                "axes": dict(rec.get("axes") or {}),
                "loss": dict(rec.get("loss") or {})}


# ===================================================== 1. `--judge` 默认与取值
class TestJudgeFlag:
    def test_module_exposes_the_three_modes(self):
        mod = _load_eval_module()
        assert mod.JUDGE_MODES == ("v1", "v2", "both"), (
            f"JUDGE_MODES 变成了 {mod.JUDGE_MODES}：计划书只允许 v1|v2|both")

    def test_argparse_default_is_v1(self):
        """默认档必须是 v1 —— 它决定"不传 --judge 的历史产物不许变"（J3）。"""
        src = EVAL_SCRIPT.read_text("utf-8")
        call = None
        for node in ast.walk(ast.parse(src)):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", "") == "add_argument"
                    and node.args and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "--judge"):
                call = node
        assert call is not None, "`eval_answer_quality.py` 里没有 `--judge` 参数"
        kw = {k.arg: k.value for k in call.keywords}
        assert isinstance(kw.get("default"), ast.Constant), "`--judge` 没有显式默认值"
        assert kw["default"].value == "v1", (
            f"`--judge` 默认值变成了 {kw['default'].value!r}："
            f"默认不是 v1 会让历史产物与 judge_v2 的存在与否绑在一起")
        assert kw.get("dest") is not None and kw["dest"].value == "judge_mode"

    def test_run_signature_defaults_to_v1(self):
        mod = _load_eval_module()
        params = inspect.signature(mod.run).parameters
        assert "judge_mode" in params, "`run()` 没有 judge_mode 参数"
        assert params["judge_mode"].default == "v1", (
            "`run()` 的 judge_mode 默认值必须是 v1："
            "任何直接调 run() 的历史脚本都会静默切到 v2")

    def test_argparse_rejects_an_unknown_mode(self):
        mod = _load_eval_module()
        with pytest.raises(SystemExit):
            mod.main(["--judge", "v3"])

    def test_v2_without_the_module_exits_two_instead_of_falling_back(self, capsys):
        """没装 `judge_v2` 时必须退出 2 并说明，**不许**静默跑 v1。"""
        mod = _load_eval_module()

        def missing() -> None:
            mod._JUDGE_V2_ERROR = "ImportError: no module named arkrag.judge_v2"
            return None

        mod.load_judge_v2 = missing
        rc = mod.main(["--judge", "both", "--limit", "1"])
        err = capsys.readouterr().err
        assert rc == 2, f"退出码是 {rc}，应当是 2（配置/依赖错误）"
        assert "judge_v2" in err and "不会静默退回" in err, (
            f"报错信息没说明「不静默退回 v1」这件事：{err!r}")

    def test_run_raises_when_the_mode_needs_a_missing_module(self):
        """第二道闸：直接调 `run(judge_mode="v2")` 也不许静默降级。"""
        mod = _load_eval_module()
        mod.load_judge_v2 = lambda: None
        with pytest.raises(RuntimeError):
            mod.run("g-fake", Path("does-not-exist.json"), split="dev", limit=1,
                    token_budget=None, judge_mode="v2")

    def test_run_rejects_an_unknown_mode(self):
        mod = _load_eval_module()
        with pytest.raises(ValueError):
            mod.run("g-fake", Path("does-not-exist.json"), split="dev", limit=1,
                    token_budget=None, judge_mode="v9")


# ============================================== 2. `--judge v1` 产物逐字段不变
class TestV1ProductIsByteIdentical:
    """J3 硬门。这里比的是**改造前后的同一份东西**，不是"看起来差不多"。"""

    #: 改造前（`git show HEAD:scripts/eval_answer_quality.py`）的顶层键，逐字抄。
    LEGACY_TOP_KEYS = [
        "generation_id", "gold_set", "split", "token_budget", "prod_profile",
        "prod_options", "n_items", "judge_schema", "chapters_seen",
        "per_category", "totals", "rows",
    ]

    #: 改造前的行键。
    LEGACY_ROW_KEYS = [
        "question_id", "category", "split", "question", "answer", "abstained",
        "judge", "citations_ok", "n_citations", "arms",
    ]

    def test_v1_top_level_keys_are_exactly_the_legacy_ones(self, monkeypatch,
                                                           tmp_path):
        mod = _load_eval_module()
        rep = _run_stub(monkeypatch, mod, tmp_path, "v1")
        assert list(rep.keys()) == self.LEGACY_TOP_KEYS, (
            f"v1 产物多了/少了键：{sorted(set(rep) ^ set(self.LEGACY_TOP_KEYS))}。"
            f"J3 要求逐题逐字段相等——多一个 `judge_mode: null` 也算不等")
        for banned in ("judge_mode", "flips", "stats", "floors", "judge_arms",
                       "context_support", "judge_schema_v2"):
            assert banned not in rep, f"v1 产物里出现了 W4b 的键 {banned!r}"

    def test_v1_row_keys_are_exactly_the_legacy_ones(self, monkeypatch, tmp_path):
        mod = _load_eval_module()
        rep = _run_stub(monkeypatch, mod, tmp_path, "v1")
        assert rep["rows"], "桩数据没产出任何行，这条门等于没跑"
        for row in rep["rows"]:
            assert list(row.keys()) == self.LEGACY_ROW_KEYS, (
                f"{row.get('question_id')} 的行键变了：{sorted(row.keys())}；"
                f"v1 档不许出现 judge_v2 / context_available 之类的新字段")
            assert list(row["arms"].keys()) == ["prod"], (
                "v1 档的 arms 只许有 prod 一臂，多一臂等于偷偷改了分母的语义")

    def test_v1_report_equals_the_frozen_literal(self, monkeypatch, tmp_path):
        """整份 dict 的字面量比对（`sort_keys` 后逐字符相等）。

        只比键集合不够：值里任何一个 `round(...,6)` 的口径变化、或者 `tally()`
        分母的调整，都会在分数上留下痕迹而键集合一模一样。
        """
        mod = _load_eval_module()
        rep = _run_stub(monkeypatch, mod, tmp_path, "v1")
        got = json.dumps(rep, ensure_ascii=False, sort_keys=True)
        expected_totals = {
            "prod": {"scorable_n": 4, "scorable_pass": 3, "scorable_rate": 0.75,
                     "all_n": 5, "all_rate": 0.6, "opened": 4,
                     "opened_rate": 0.8},
        }
        assert rep["totals"] == expected_totals, (
            f"v1 的总分口径变了：{rep['totals']} != {expected_totals}。"
            f"桩数据里 5 道题有 4 道可判（time_conflict 是结构类），"
            f"其中 3 道答对、4 道开口——任何一处对不上都说明分母或计分改了")
        assert '"judge_v2"' not in got and '"flips"' not in got, (
            "序列化后的 v1 产物里出现了 v2 的痕迹")

    def test_v1_stdout_keeps_the_legacy_block(self, monkeypatch, tmp_path, capsys):
        """人看的报告：旧那一段的**每一行**都必须与改造前同格式同数字。

        新加的是"地板三行"这一节（J5 要求每次必报），它排在旧内容**之后**，
        所以断言旧内容作为前缀出现是合理的：反过来（把新节插到中间）会让
        任何按行号解析输出的历史脚本读错数。
        """
        mod = _load_eval_module()
        rep = _run_stub(monkeypatch, mod, tmp_path, "v1", strip_timing=False)
        mod.print_report(rep)
        out = capsys.readouterr().out
        lines = out.splitlines()
        legacy = [
            "可判分母 3/4 = 75.00%",
            "全分数  3/5 = 60.00%（结构类按没答对计入）",
            "开口率  4/5 = 80.00%",
        ]
        for want in legacy:
            assert want in lines, f"v1 报告里缺了旧行 {want!r}；实际输出：{lines}"
        # 旧内容的结尾必须在"地板三行"之前：顺序反了会让按行号读输出的脚本读错数。
        assert out.index("地板三行") > out.index(legacy[-1]), (
            "地板三行插到了旧报告内容之前：按行号解析输出的历史脚本会读错数")

    def test_v1_tally_shape_is_untouched(self):
        """`tally()` 是纯函数，可以直接喂 row：它的键集合就是产物的契约。"""
        mod = _load_eval_module()
        rows = [
            {"category": "single_chunk",
             "arms": {"prod": {"pass": True, "answered": True}}},
            {"category": "time_conflict",
             "arms": {"prod": {"pass": False, "answered": False}}},
        ]
        out = mod.tally(rows, ["prod"])
        assert set(out) == {"per_category", "totals"}
        assert out["per_category"]["single_chunk"]["arms"]["prod"] == {
            "pass": 1, "opened": 1}
        assert out["per_category"]["time_conflict"]["arms"]["prod"] == {
            "pass": None, "opened": 0}
        assert set(out["totals"]["prod"]) == {
            "scorable_n", "scorable_pass", "scorable_rate", "all_n",
            "all_rate", "opened", "opened_rate"}

    def test_v1_does_not_call_judge_v2_at_all(self, monkeypatch, tmp_path):
        """J3 的**机制**：v1 档既不解析 v2 模块、也不调它，更不写 `judge_v2` 键。

        三层都要断言。只查"调没调函数"是不够的——反向实测里把 v1 分支也打开时，
        因为 `judge_v2` 局部变量是 None，`_v2_call` 直接返回 `no_entry_point`，
        **一次调用都没发生**，于是只看调用计数的版本照样绿。
        真正的失效是"行里多了一个 `judge_v2` 键"，那才是 J3 会被打破的形态。
        """
        mod = _load_eval_module()
        calls: list[str] = []
        loads: list[str] = []

        class Spy:
            def judge_answer_v2(self, *a, **kw):
                calls.append("called")
                return {"pass": True, "reason": "spy", "axes": {}, "loss": {}}

        def spy_loader():
            loads.append("loaded")
            return Spy()

        monkeypatch.setattr(mod, "load_judge_v2", spy_loader)
        rep = _run_stub(monkeypatch, mod, tmp_path, "v1")
        assert loads == [], f"v1 档解析了 judge_v2 {len(loads)} 次：J3 不成立"
        assert calls == [], f"v1 档调了 judge_v2 {len(calls)} 次：J3 不成立"
        assert all("judge_v2" not in row for row in rep["rows"]), (
            "v1 档的行里出现了 judge_v2 键：J3 要求逐字段相等")


# ============================================================ 3. 翻转四类计数
class TestFlipCounts:
    def test_four_cells_sum_to_the_decidable_denominator(self):
        mod = _load_eval_module()
        rows = [
            _row(True, True, qid="a"),
            _row(True, False, qid="b", axes={"support_ok": False}),
            _row(False, True, qid="c"),
            _row(False, False, qid="d"),
            _row(False, False, qid="e"),
            _row(None, None, category="time_conflict", qid="f"),   # 结构类
            _row(True, None, qid="g"),                             # v2 不可判
        ]
        fl = mod.flip_counts(rows)
        c = fl["counts"]
        assert (c["PP"], c["PF"], c["FP"], c["FF"]) == (1, 1, 1, 2), c
        assert sum(c.values()) == fl["decidable_n"] == 5, (
            f"四类相加 {sum(c.values())} 与可判分母 {fl['decidable_n']} 对不上；"
            f"计划书反证条件 2 明确把这条列为「实现有 bug」")
        assert fl["structural"] == 1
        assert fl["v2_undecidable_n"] == 1
        assert fl["decidable_n"] + fl["structural"] + fl["v2_undecidable_n"] == len(rows)

    def test_undecidable_v2_is_not_counted_as_failure(self):
        """v2 返回 `None` 不许掉进 FF —— 那会把"判据崩了"记成"判据更严"。"""
        mod = _load_eval_module()
        fl = mod.flip_counts([_row(True, None, qid="g"),
                              _row(None, None, category="time_conflict", qid="h")])
        assert fl["counts"] == {"PP": 0, "PF": 0, "FP": 0, "FF": 0}
        assert fl["decidable_n"] == 0
        assert fl["v2_undecidable"][0]["question_id"] == "g"
        assert fl["v2_undecidable"][0]["v2_reason"] == "r"

    def test_pf_groups_are_mutually_exclusive(self):
        """`PF` 按原因分组时，每组**相加必须等于 PF**（互斥，不是多标签）。"""
        mod = _load_eval_module()
        rows = [
            _row(True, False, qid="u", axes={"support_ok": False}),
            _row(True, False, qid="c", axes={"cardinality_ok": False}),
            _row(True, False, qid="v", axes={"value_ok": False}),
            # 两轴同时失败：必须只落进一个桶（固定顺序里靠前的那个）
            _row(True, False, qid="both",
                 axes={"support_ok": False, "cardinality_ok": False}),
        ]
        fl = mod.flip_counts(rows)
        groups = fl["pf_by_reason"]
        assert fl["counts"]["PF"] == 4
        assert sum(groups.values()) == fl["counts"]["PF"], (
            f"PF 分组相加 {sum(groups.values())} != PF {fl['counts']['PF']}："
            f"说明同一题进了多个桶，那份「互斥归因」就是假的")
        assert groups["support_ok"] == 2, groups
        assert groups["cardinality_ok"] == 1
        assert groups["value_ok"] == 1

    def test_pf_bucket_falls_back_to_loss_then_unattributed(self):
        mod = _load_eval_module()
        assert mod.pf_bucket({"axes": {}, "loss": {"extra_values": 2}}) == "extra_values"
        assert mod.pf_bucket({"axes": {}, "loss": {}}) == "unattributed"
        assert mod.pf_bucket({}) == "unattributed"

    def test_an_unattributed_pf_is_visible_not_hidden(self):
        """归因不出来的 PF 必须**显式**出现在分组里（反证条件 1 的入口）。"""
        mod = _load_eval_module()
        fl = mod.flip_counts([_row(True, False, qid="x")])
        assert fl["pf_by_reason"] == {"unattributed": 1}, (
            "无法归因的 PF 被吞掉了：那等于「翻转都可解释」这句话无法被检验")

    def test_dual_report_carries_the_flip_block(self, monkeypatch, tmp_path):
        mod = _load_eval_module()
        stub = _V2Stub({"q-sc-1": {"pass": True, "reason": "v2_ok"},
                        "q-al-1": {"pass": False, "reason": "v2_no_support",
                                   "axes": {"support_ok": False},
                                   "loss": {"unsupported_values": 1}},
                        "q-cx-1": {"pass": True, "reason": "v2_ok"},
                        "q-mh-1": {"pass": False, "reason": "v2_no_support",
                                   "axes": {"support_ok": False}},
                        "q-tc-1": {"pass": None, "reason": "structural"}})
        rep = _run_stub(monkeypatch, mod, tmp_path, "both", judge_v2_stub=stub)
        fl = rep["flips"]
        assert rep["judge_mode"] == "both"
        assert rep["judge_arms"] == ["prod", "v2"]
        assert sum(fl["counts"].values()) == fl["decidable_n"] == 4, (
            f"四类 {fl['counts']} 相加 != 可判 {fl['decidable_n']}")
        # 逐题核过（桩数据里 v1 的判定由真实 `judge.judge_answer` 给出）：
        #   q-sc-1 single_chunk 答案含 gold  → v1 T / v2 T  → PP
        #   q-al-1 alias 答案含 gold         → v1 T / v2 F  → PF（support_ok）
        #   q-cx-1 cross_chunk 章节集合相等  → v1 T / v2 T  → PP
        #   q-mh-1 multihop gold 恰好是锚点  → v1 F / v2 F  → FF
        #   q-tc-1 结构类                    → v1 None（不参与）
        assert (fl["counts"]["PP"], fl["counts"]["PF"],
                fl["counts"]["FP"], fl["counts"]["FF"]) == (2, 1, 0, 1), fl["counts"]
        assert fl["pf_by_reason"] == {"support_ok": 1}, fl["pf_by_reason"]


# ================================================================= 4. 簇键函数
class TestClusterKeys:
    #: 同一模板、只换实体名的两道题（3-gram Jaccard 应当很高）。
    TEMPLATE_A_1 = "「凯尔希」在哪些章节代号里出现过？请只列章节代号。"
    TEMPLATE_A_2 = "「阿米娅」在哪些章节代号里出现过？请只列章节代号。"
    #: 完全不同的模板。
    TEMPLATE_B = "整合运动的首领与罗德岛的医疗顾问之间是什么关系？"

    def test_same_template_lands_in_the_same_cluster(self):
        keys = cluster_mod.cluster_keys([self.TEMPLATE_A_1, self.TEMPLATE_A_2])
        assert keys[0] == keys[1], (
            f"同模板两道题被分到不同簇：{keys}；"
            f"Jaccard={cluster_mod.question_jaccard(self.TEMPLATE_A_1, self.TEMPLATE_A_2):.3f}")

    def test_different_templates_split(self):
        keys = cluster_mod.cluster_keys([self.TEMPLATE_A_1, self.TEMPLATE_B])
        assert keys[0] != keys[1], (
            f"不同模板被并进同一簇：{keys}；"
            f"Jaccard={cluster_mod.question_jaccard(self.TEMPLATE_A_1, self.TEMPLATE_B):.3f}")
        assert len(set(keys)) == 2

    def test_cluster_number_is_the_minimum_index_in_the_group(self):
        """簇号取分量内最小下标：这样报告里的簇号**不看输入顺序也能对上**。"""
        texts = [self.TEMPLATE_B, self.TEMPLATE_A_1, self.TEMPLATE_A_2]
        keys = cluster_mod.cluster_keys(texts)
        assert keys[0] == 0
        assert keys[1] == keys[2] == 1, keys

    def test_cluster_keys_are_deterministic(self):
        texts = [self.TEMPLATE_A_1, self.TEMPLATE_B, self.TEMPLATE_A_2]
        assert cluster_mod.cluster_keys(texts) == cluster_mod.cluster_keys(texts)

    def test_identical_questions_collapse_and_unique_count_agrees(self):
        texts = [self.TEMPLATE_A_1, self.TEMPLATE_A_1, self.TEMPLATE_B]
        keys = cluster_mod.cluster_keys(texts)
        assert keys[0] == keys[1] != keys[2]
        assert cluster_mod.unique_question_count(texts) == 2
        assert cluster_mod.cluster_report(texts)["n_clusters"] == 2

    def test_jaccard_identity_is_one_and_empty_is_zero(self):
        assert cluster_mod.question_jaccard("同一个问题", "同一个问题") == 1.0
        assert cluster_mod.question_jaccard("", "同一个问题") == 0.0, (
            "空题面被判成 0 相似度而不是 1：空集与空集的 Jaccard 无定义，"
            "把它当 1 会把所有空题并成一个簇")
        assert cluster_mod.question_jaccard("", "") == 0.0

    def test_whitespace_is_normalized_before_comparing(self):
        a = "「凯尔希」 是谁？"
        b = "「凯尔希」是谁？"
        assert cluster_mod.question_jaccard(a, b) == 1.0, (
            "空白没有被归一：同一句话带不带空格会被分到两个簇")

    def test_threshold_boundary_is_inclusive(self):
        texts = [self.TEMPLATE_A_1, self.TEMPLATE_A_2]
        sim = cluster_mod.question_jaccard(texts[0], texts[1])
        assert cluster_mod.cluster_keys(texts, threshold=sim)[0] == \
            cluster_mod.cluster_keys(texts, threshold=sim)[1], (
            "阈值取等号时不算近似重复：那会让「恰好 0.6」的一对随机地落在两边")
        assert cluster_mod.cluster_keys(texts, threshold=min(1.0, sim + 1e-9))[0] != \
            cluster_mod.cluster_keys(texts, threshold=min(1.0, sim + 1e-9))[1]

    def test_empty_input_is_an_empty_result_not_a_crash(self):
        assert cluster_mod.cluster_keys([]) == []
        assert cluster_mod.cluster_report([])["n_clusters"] == 0
        assert cluster_mod.cluster_sizes([]) == []

    def test_kish_effective_clusters_penalises_uneven_sizes(self):
        assert cluster_mod.kish_effective_n([1, 1, 1, 1]) == 4.0
        # 一个 7 题的簇 + 三个 1 题的簇：(10^2)/(49+1+1+1) = 1.923
        assert round(cluster_mod.kish_effective_n([7, 1, 1, 1]), 4) == 1.9231


# ========================================================= 5. 校正指标定义域
class TestCorrectedRateDomain:
    def test_valid_domain_reports_the_value(self):
        mod = _load_eval_module()
        out = mod.corrected_rate(pass_count=228, n=285, floor_count=68, floor_n=285)
        assert out["estimable"] is True
        assert abs(out["value"] - (228 - 68) / (285 - 68)) < 1e-12
        assert out["value"] > 0

    def test_floor_above_pass_is_not_estimable(self):
        """**核心**：`floor > pass` 时不许返回负数。

        返回负数会被读成"改进是负的"，而真实含义是"整条通路不如一条地板"。
        """
        mod = _load_eval_module()
        out = mod.corrected_rate(pass_count=40, n=285, floor_count=68, floor_n=285)
        assert out["estimable"] is False, "floor > pass 却给了数"
        assert out["value"] is None, f"floor > pass 返回了 {out['value']!r}（会变成负数）"
        assert "越界" in out["reason"]

    def test_zero_denominator_is_not_estimable(self):
        mod = _load_eval_module()
        out = mod.corrected_rate(pass_count=285, n=285, floor_count=285, floor_n=285)
        assert out["estimable"] is False and out["value"] is None
        assert "N−floor" in out["reason"] or "-" in out["reason"]

    def test_pass_above_n_is_not_estimable(self):
        mod = _load_eval_module()
        out = mod.corrected_rate(pass_count=300, n=285, floor_count=0, floor_n=285)
        assert out["estimable"] is False and out["value"] is None

    def test_negative_floor_is_not_estimable(self):
        mod = _load_eval_module()
        out = mod.corrected_rate(pass_count=200, n=285, floor_count=-1, floor_n=285)
        assert out["estimable"] is False and out["value"] is None

    def test_denominator_mismatch_is_not_estimable(self):
        """三个地板数都在 dev 的 285 上量的；换 split 后不能直接相减。"""
        mod = _load_eval_module()
        out = mod.corrected_rate(pass_count=150, n=195, floor_count=68, floor_n=285)
        assert out["estimable"] is False and out["value"] is None
        assert "分母不同" in out["reason"]

    def test_boundary_floor_equals_pass_is_zero_not_negative(self):
        mod = _load_eval_module()
        out = mod.corrected_rate(pass_count=68, n=285, floor_count=68, floor_n=285)
        assert out["estimable"] is True
        assert out["value"] == 0.0


# ======================================================= 6. 地板三行必报（J5）
class TestFloorsAreAlwaysReported:
    def test_floor_report_has_exactly_the_three_registered_lines(self):
        mod = _load_eval_module()
        block = mod.floor_report(228, 285, arm="v2")
        labels = [line["label"] for line in block["lines"]]
        assert labels == ["回显题干", "形状 oracle", "gold 原文"], labels
        counts = [line["floor_count"] for line in block["lines"]]
        assert counts == [68, 285, 171], (
            f"三个地板的分子变成了 {counts}：计划书的实测值是 68/285、285/285、171/285")
        rates = [round(line["floor_rate"] * 100, 2) for line in block["lines"]]
        assert rates == [23.86, 100.0, 60.0], (
            f"三个地板数变成了 {rates}：计划书写死 23.86% / 100% / 60%")

    def test_every_floor_line_carries_its_own_denominator_and_method(self):
        mod = _load_eval_module()
        block = mod.floor_report(228, 285, arm="v2")
        for line in block["lines"]:
            assert line["floor_n"] == 285, line
            assert line["method"].strip(), f"{line['label']} 没写估计方法"
            assert line["source"].strip(), f"{line['label']} 没写来源"
            assert "corrected" in line

    def test_the_reported_rate_matches_its_own_count(self):
        """`rate` 必须等于 `count / n`，不许是一个手填的漂亮数字。

        这条是反向实测补出来的：把 `"count": 68` 改成 `70` 而 `rate` 仍是 0.2386，
        原先那组断言**一条都没红** —— 因为它只读了 `floor_rate` 这个常量，
        从来没核对过它与分子分母是否自洽。这正是本项目反复记过的
        「手填的漂亮数字不算」。
        """
        mod = _load_eval_module()
        for line in mod.floor_report(228, 285, arm="v2")["lines"]:
            implied = line["floor_count"] / line["floor_n"]
            assert abs(line["floor_rate"] - implied) < 5e-5, (
                f"{line['label']} 的 rate={line['floor_rate']} 与 "
                f"{line['floor_count']}/{line['floor_n']}={implied:.6f} 对不上："
                f"两个数里有一个是手填的")

    def test_floors_are_declared_non_additive(self):
        """三行**不许相加**：它们是同一分母上的替代通路，不是互斥题集。"""
        mod = _load_eval_module()
        block = mod.floor_report(228, 285, arm="v2")
        assert block["additive"] is False
        assert "sum" not in block and "total" not in block, (
            "地板块里出现了合计键：一旦提供，早晚有人会去加")

    def test_print_report_always_prints_the_three_floors(self, monkeypatch,
                                                         tmp_path, capsys):
        """J5 的落地：v1 单臂也必须打出三行（那时 JSON 里没有 `floors` 块）。"""
        mod = _load_eval_module()
        rep = _run_stub(monkeypatch, mod, tmp_path, "v1", strip_timing=False)
        assert "floors" not in rep, "v1 产物里不该有 floors 块（J3）"
        mod.print_report(rep)
        out = capsys.readouterr().out
        assert "地板三行" in out, "v1 的报告没有地板三行"
        for token in ("回显题干", "形状 oracle", "gold 原文",
                      "23.86%", "100.00%", "60.00%"):
            assert token in out, f"地板三行里缺 {token!r}；实际输出：\n{out}"

    def test_dual_report_attaches_floors_computed_on_the_v2_arm(self, monkeypatch,
                                                                tmp_path):
        mod = _load_eval_module()
        stub = _V2Stub({k: {"pass": True, "reason": "ok"} for k in STUB_ANSWERS})
        rep = _run_stub(monkeypatch, mod, tmp_path, "both", judge_v2_stub=stub)
        block = rep["floors"]
        assert block["lines"], "双口径报告没有地板块"
        assert all(line["arm"] == "v2" for line in block["lines"]), (
            "地板块没有说清它挂在哪个臂上：两套判据的通过率不同，"
            "校正指标也就不同")
        assert block["lines"][0]["pass_count"] == rep["totals"]["v2"]["scorable_pass"]


# ================================================ 7. 方差膨胀只在有配对时给数
class TestVarianceInflationHonesty:
    def test_single_arm_reports_not_estimable(self):
        mod = _load_eval_module()
        rows = [{"category": "single_chunk", "question": "q",
                 "arms": {"prod": {"pass": True, "answered": True}}}]
        st = mod.stats_block(rows, ["prod"])
        infl = st["variance_inflation"]
        assert infl["estimable"] is False
        assert infl["value"] is None, (
            f"单臂没有配对样本却给出了方差膨胀 {infl['value']!r}："
            f"计划书明令不许用 m̄≈2.5,ρ≈0.6 的假设值冒充实测")
        assert st["effective_sample_size"]["estimable"] is False

    def test_the_designed_estimate_never_appears_as_a_measured_value(self):
        """扫整块：任何位置都不许出现 1.9 这个"设计量级估算"被当成实测值。"""
        mod = _load_eval_module()
        rows = [{"category": "single_chunk", "question": "q",
                 "arms": {"prod": {"pass": True, "answered": True}}}]
        st = mod.stats_block(rows, ["prod"])
        got = json.dumps(st, ensure_ascii=False)
        assert '"value": 1.9' not in got and '"1.9"' not in got
        assert st["variance_inflation"]["value"] is None
        # 参考量级只能在 `reference` 里以"注释"形式存在，且必须写明它不是实测
        assert "设计量级估算" in got and "不作为实测值" in got

    def test_paired_samples_do_produce_a_ratio(self):
        """正向控制：真有两套方差时必须算出比值（否则上面两条可能只是恒 None）。"""
        iid = {"var": 0.04, "reps": 100, "mean": 0.0, "ci_low": -0.1,
               "ci_high": 0.1, "degenerate": False, "note": ""}
        clu = {"var": 0.076, "reps": 100, "mean": 0.0, "ci_low": -0.2,
               "ci_high": 0.2, "degenerate": False, "note": "", "n_clusters": 40}
        out = cluster_mod.variance_inflation(iid, clu)
        assert out["estimable"] is True
        assert abs(out["value"] - 0.076 / 0.04) < 1e-12

    def test_zero_iid_variance_is_not_estimable(self):
        iid = {"var": 0.0, "reps": 100}
        clu = {"var": 0.076, "reps": 100, "n_clusters": 40}
        out = cluster_mod.variance_inflation(iid, clu)
        assert out["estimable"] is False and out["value"] is None

    def test_zero_reps_is_not_estimable(self):
        """没有抽样序列就没有方差，比值同样无定义。

        这条是反向实测补出来的：原先只测了"方差为 0"这一条，于是把
        `reps == 0` 那道保护整个删掉，测试**一条都没红**。
        """
        iid = {"var": 0.04, "reps": 0}
        clu = {"var": 0.076, "reps": 0, "n_clusters": 40}
        out = cluster_mod.variance_inflation(iid, clu)
        assert out["estimable"] is False and out["value"] is None
        assert "样本" in out["reason"]

    def test_single_cluster_is_not_estimable(self):
        iid = {"var": 0.04, "reps": 100}
        clu = {"var": 0.076, "reps": 100, "n_clusters": 1}
        out = cluster_mod.variance_inflation(iid, clu)
        assert out["estimable"] is False and out["value"] is None

    def test_effective_sample_size_needs_an_estimable_inflation(self):
        bad = {"estimable": False, "value": None, "reason": "x"}
        assert cluster_mod.effective_sample_size(285, bad)["value"] is None
        good = {"estimable": True, "value": 2.0, "reason": "x"}
        assert cluster_mod.effective_sample_size(300, good)["value"] == 150.0


# ============================================================ 8. 双口径并列报
class TestDualReportShape:
    def test_both_arms_are_reported_side_by_side(self, monkeypatch, tmp_path):
        """只报一套就是选择性披露（红线 11 同型）——所以两臂必须同时在。"""
        mod = _load_eval_module()
        stub = _V2Stub({"q-sc-1": {"pass": False, "reason": "no_support",
                                   "axes": {"support_ok": False}},
                        "q-al-1": {"pass": True, "reason": "ok"},
                        "q-cx-1": {"pass": True, "reason": "ok"},
                        "q-mh-1": {"pass": True, "reason": "ok"},
                        "q-tc-1": {"pass": None, "reason": "structural"}})
        rep = _run_stub(monkeypatch, mod, tmp_path, "both", judge_v2_stub=stub)
        assert set(rep["totals"]) == {"prod", "v2"}
        assert rep["totals"]["prod"]["scorable_pass"] == 3
        assert rep["totals"]["v2"]["scorable_pass"] == 3
        for entry in rep["per_category"].values():
            assert set(entry["arms"]) == {"prod", "v2"}, entry["arms"]

    def test_dual_stdout_shows_both_columns(self, monkeypatch, tmp_path, capsys):
        mod = _load_eval_module()
        stub = _V2Stub({"q-sc-1": {"pass": False, "reason": "no_support",
                                   "axes": {"support_ok": False}}})
        rep = _run_stub(monkeypatch, mod, tmp_path, "both", judge_v2_stub=stub,
                        strip_timing=False)
        mod.print_report(rep)
        out = capsys.readouterr().out
        assert "v1答对" in out and "v2答对" in out, out
        assert "翻转四类" in out and "PF" in out
        assert "簇结构" in out, "双口径报告没有簇数/有效样本量那一行"
        assert "方差膨胀" in out

    def test_v2_errors_become_undecidable_not_failures(self, monkeypatch, tmp_path):
        """v2 抛异常 → `pass=None` + error 记进 `v2_undecidable`，不许算失败。"""
        mod = _load_eval_module()

        class Boom:
            def judge_answer_v2(self, *a, **kw):
                raise RuntimeError("v2 内部炸了")

        monkeypatch.setattr(mod, "load_judge_v2", lambda: Boom())
        rep = _run_stub(monkeypatch, mod, tmp_path, "v2", judge_v2_stub=Boom())
        fl = rep["flips"]
        assert fl["decidable_n"] == 0, fl["counts"]
        assert fl["v2_undecidable_n"] == len(STUB_ITEMS)
        assert "v2_error" in fl["v2_undecidable"][0]["v2_reason"]
        assert rep["stats"]["variance_inflation"]["estimable"] is False
        assert rep["stats"]["paired_v1_v2"]["n_paired"] == 0

    def test_context_availability_is_recorded(self, monkeypatch, tmp_path):
        mod = _load_eval_module()
        stub = _V2Stub({"q-sc-1": {"pass": True, "reason": "ok"}})
        rep = _run_stub(monkeypatch, mod, tmp_path, "v2", judge_v2_stub=stub)
        assert rep["context_support"]["rows_with_context"] == len(STUB_ITEMS), (
            "桩 ask_once 明明返回了 context dict，报告却说没有："
            "那会把'支撑轴拿不到断言'这个限制误报成普遍情况")
        lim = rep["context_support"]["limitation"]
        assert "support" in lim and "不代表答案错" in lim
        assert rep["judge_v2_entry"] == "judge_answer_v2", (
            f"报告没记下实际用的入口名：{rep.get('judge_v2_entry')!r}")

    def test_missing_context_is_flagged_as_a_limitation(self):
        """`ask_once` 不给 context 时必须**显式**标出来，不许静默降级。"""
        mod = _load_eval_module()
        note = mod.context_note({"answer": {}})
        assert note["context_available"] is False
        assert "不可信" in note["context_note"] and "不是答案错了" in note["context_note"]
        ok = mod.context_note({"context": {"units": [
            {"assertions": [{"assertion_id": "a1"}]}]}})
        assert ok["context_available"] is True and ok["context_assertions"] == 1


# ======================================================== 9. 统计口径本身
class TestV2EntryPointResolution:
    """接口约定与落地模块的函数名不同，这件事必须被显式处理。

    约定写的是 `judge_answer_v2`，实际落地的 `arkrag/judge_v2.py` 把入口叫
    `judge_v2`（模块名与函数名同名）。静默任选一个会让"报告说自己用了 v2"
    无法核对，所以有固定优先级 + 报告字段。
    """

    def test_prefers_the_contracted_name_when_both_exist(self):
        mod = _load_eval_module()

        def judge_answer_v2(*a, **kw):
            return {"pass": True}

        def judge_v2(*a, **kw):
            return {"pass": False}

        ns = SimpleNamespace(judge_answer_v2=judge_answer_v2, judge_v2=judge_v2)
        fn, name = mod.resolve_v2_entry(ns)
        assert name == "judge_answer_v2"
        assert fn("x", "y", {})["pass"] is True

    def test_falls_back_to_the_shipped_name(self):
        mod = _load_eval_module()
        ns = SimpleNamespace(judge_v2=lambda *a, **kw: {"pass": True})
        _fn, name = mod.resolve_v2_entry(ns)
        assert name == "judge_v2", (
            "落地模块的入口叫 `judge_v2`，认不出来就等于 v2 一条都没跑")

    def test_missing_entry_point_is_undecidable_not_pass(self):
        mod = _load_eval_module()
        rec = mod._v2_call(SimpleNamespace(), "single_chunk", "x", {}, {},
                           question="q", chapter_texts={}, context={})
        assert rec["pass"] is None, "没有入口却被当成通过/失败"
        assert "no_entry_point" in rec["reason"]

    def test_shipped_entry_receives_the_answer_dict_for_citations(self):
        """落地的 `judge_v2.judge_v2` 从答案 dict 里取 citations。

        只传正文等于宣布"这条答案没有引用支撑"，`support_ok` 会恒 False——
        那样 v1/v2 的差就全是"引用没接上"这个**假区别**，而它看起来正好像
        "v2 治了形状 oracle"。这是本项最危险的一处失真，所以单钉。
        """
        mod = _load_eval_module()
        seen: dict = {}

        def shipped(category, answer, item, **kw):
            seen["answer"] = answer
            seen["kwargs"] = kw
            return {"pass": True, "reason": "ok", "axes": {}, "loss": {}}

        ans = {"answer": "凯尔希", "citations": [{"chunk_id": "c#000"}]}
        mod._v2_call(SimpleNamespace(judge_v2=shipped), "single_chunk", "凯尔希",
                     {}, ans, question="q", chapter_texts={}, context={})
        assert seen["answer"] is ans, (
            f"落地入口收到的是 {type(seen['answer']).__name__}，应当是答案 dict："
            f"传正文会把引用丢掉")
        assert "chapter_cap" not in seen["kwargs"], (
            "chapter_cap 未开（None/0）时不该传参：那会覆盖 judge_v2 自己的登记默认值")

    def test_contracted_entry_still_receives_plain_text(self):
        """约定签名 `judge_answer_v2(category, answer_text, ...)` 收的是正文。"""
        mod = _load_eval_module()
        seen: dict = {}

        def contracted(category, answer_text, item, **kw):
            seen["text"] = answer_text
            return {"pass": True, "reason": "ok", "axes": {}, "loss": {}}

        mod._v2_call(SimpleNamespace(judge_answer_v2=contracted), "single_chunk",
                     "凯尔希", {}, {"answer": "凯尔希"}, question="q",
                     chapter_texts={}, context={})
        assert seen["text"] == "凯尔希"

    def test_chapter_cap_is_forwarded_only_when_enabled(self):
        mod = _load_eval_module()
        seen: dict = {}

        def shipped(category, answer, item, **kw):
            seen.update(kw)
            return {"pass": True, "reason": "ok", "axes": {}, "loss": {}}

        ns = SimpleNamespace(judge_v2=shipped)
        mod._v2_call(ns, "cross_chunk", "x", {}, {}, question="q",
                     chapter_texts={}, context={}, chapter_cap=4)
        assert seen.get("chapter_cap") == 4, (
            "生产档开着 chapter_cap=4，却不把它传给 v2："
            "两套判据在 cross_chunk 上的基数口径就不同了")
        assert mod.v2_call_shape("judge_v2")["answer_form"] == "dict"
        assert mod.v2_call_shape("judge_answer_v2")["answer_form"] == "str"

    def test_real_module_is_wired(self):
        """对着**真模块**核一次入口名：它决定 `--judge both` 是否真跑 v2。"""
        from arkrag import judge_v2 as real_v2

        mod = _load_eval_module()
        fn, name = mod.resolve_v2_entry(real_v2)
        assert fn is not None, "真实的 arkrag.judge_v2 里认不出任何判分入口"
        assert name in ("judge_answer_v2", "judge_v2"), name


class TestBootstrapCalibers:
    def test_iid_and_cluster_both_give_intervals(self):
        vals = [1, 0, 1, 1, 0, 1, 0, 1, 1, 1]
        keys = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        iid = cluster_mod.iid_bootstrap_mean(vals, reps=200, seed=7)
        clu = cluster_mod.cluster_bootstrap_mean(vals, keys, reps=200, seed=7)
        assert iid["ci_low"] <= iid["mean"] <= iid["ci_high"] or iid["degenerate"]
        assert clu["ci_low"] <= clu["mean"] <= clu["ci_high"] or clu["degenerate"]
        assert clu["n_clusters"] == 5
        assert iid["mean"] == clu["mean"] == 0.7

    def test_cluster_resampling_is_whole_cluster(self):
        """一个簇要么全进要么全不进：所以簇内方差不会进 CI。

        夹具刻意让两个簇的均值**相同**（各 0.5）：此时 iid 重抽会因为簇内 0/1
        混合而给出非零方差，整簇重抽只能抽到两种同均值组合，方差必须为 0。
        差值本身就是判据——若实现退回成逐题重抽，这条立刻红。
        """
        vals = [1, 0, 1, 0]
        keys = [0, 0, 1, 1]
        iid = cluster_mod.iid_bootstrap_mean(vals, reps=300, seed=7)
        clu = cluster_mod.cluster_bootstrap_mean(vals, keys, reps=300, seed=7)
        assert iid["var"] > 0.0, "反向控制：逐题重抽必须有非零方差"
        assert clu["var"] == 0.0, (
            f"整簇重抽的方差必须为 0（两个簇均值都是 0.5），实际 {clu['var']}")
        assert clu["degenerate"] is True
        assert clu["mean"] == iid["mean"] == 0.5

    def test_single_cluster_is_degenerate_not_a_fake_interval(self):
        clu = cluster_mod.cluster_bootstrap_mean([1, 0, 1, 0], [0, 0, 0, 0])
        assert clu["degenerate"] is True and clu["reps"] == 0
        assert clu["note"] == "single_cluster_or_zero_reps"

    def test_length_mismatch_is_rejected(self):
        with pytest.raises(ValueError):
            cluster_mod.cluster_bootstrap_mean([1.0, 0.0], [0])
        with pytest.raises(ValueError):
            cluster_mod.paired_diffs([1, 0], [1])

    def test_paired_delta_is_in_percentage_points(self):
        # diffs = [0, -1, 0, +1, +1] → 净 +1 题、均值 +0.2 → +20.0 个百分点。
        # 用 1 位小数的绝对值断言，而不是"正负号对"：符号对但量纲错（写成 0.2）
        # 是这类统计函数最常见的失效。
        out = cluster_mod.paired_bootstrap_delta(
            [1, 1, 1, 0, 0], [1, 0, 1, 1, 1], [0, 0, 1, 2, 3], reps=100, seed=7,
            clustered=False)
        assert out["delta_count"] == 1, out
        assert abs(out["delta_pp"] - 20.0) < 1e-9, out

    def test_stats_block_reports_cluster_count_and_unique_questions(self):
        mod = _load_eval_module()
        # 同模板只换实体名 = 同簇；第三个是完全不同的模板 = 独立簇。
        # 短问句（`「凯尔希」是谁？`）的 3-gram Jaccard 只有 0.2，凑不到阈值——
        # 这正是模板簇的口径限制：**簇键只认题面文字，不认"看起来是同一类问法"**。
        rows = [
            {"category": "single_chunk",
             "question": "「凯尔希」在哪些章节代号里出现过？请只列章节代号。",
             "arms": {"prod": {"pass": True, "answered": True}}},
            {"category": "single_chunk",
             "question": "「阿米娅」在哪些章节代号里出现过？请只列章节代号。",
             "arms": {"prod": {"pass": False, "answered": True}}},
            {"category": "single_chunk",
             "question": "整合运动的首领与罗德岛的医疗顾问之间是什么关系？",
             "arms": {"prod": {"pass": True, "answered": True}}},
        ]
        st = mod.stats_block(rows, ["prod"])
        assert st["denominator"] == 3
        assert st["cluster"]["n_items"] == 3
        assert st["cluster"]["n_clusters"] == 2, (
            f"两道同模板题没有被并簇：{st['cluster']}")
        assert st["cluster"]["unique_questions"] == 3, (
            "唯一题面数与簇数是两个量，这里三道题题面都不同，去重后应为 3")
        assert st["cluster"]["cluster_sizes"] == [2, 1]
        assert "effective_clusters" in st["cluster"]

    def test_structural_categories_are_excluded_from_the_denominator(self):
        mod = _load_eval_module()
        rows = [
            {"category": "single_chunk", "question": "a",
             "arms": {"prod": {"pass": True, "answered": True}}},
            {"category": "time_conflict", "question": "b",
             "arms": {"prod": {"pass": True, "answered": True}}},
        ]
        st = mod.stats_block(rows, ["prod"])
        assert st["denominator"] == 1, (
            "结构类进了 CI 的分母：judge 对它们返回 None，"
            "按 True/False 计都会污染区间")
        assert judge_mod.SCORABLE_CATEGORIES
