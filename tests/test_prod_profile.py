"""生产档（`pipeline.PROD_PROFILE`）的门。

## 这个文件守的是什么

dev 可判分母从 198/285（69.47%）到 228/285（80.00%）靠的是**四个数同时生效**：

* `quote_locate`：题干里 ≥8 字的引文去全库逐字定位（门控 128/640，只命中
  single_chunk，零误伤）；
* `bridge`：两点共同邻居唯一时才答（候选只许来自本次 context，见 `bridge.py`）；
* `graph_reserve_tokens=12000`：给图证据预留装填额度；
* `max_context_tokens=24000`：24000 以下纯 ctx 装不下 40 个直接单元 + 12000 预留。

它们缺一个就掉档，而且是**静默**掉档：`ask_once(options=None)` 是历史默认档，
服务端或 CLI 不传 options 的话，线上照样出答案、报告照样打印，只是分数退回
69.47%。这类"开关没接到入口"的缺陷不会自己暴露，所以要静态门钉住：

1. `PROD_PROFILE` 的四个数就是实测出来的那四个（改要重跑三份 split）。
2. `apply_prod_profile()` 真的把预算写进实例属性、且不改 `DEFAULT_BUDGET`
   （改了会让 `scripts/` 下 40 多个历史脚本与台账读数一起漂）。
3. `prod_options()` 返回副本（调用方改它不许污染常量）。
4. 三个入口（`server._query`、`scripts/ask.py` 两处、`eval_answer_quality.run`）
   真的把 options 传给了 `ask_once`。
5. 默认路径不变：`prod=False` / `options=None` 必须逐位等于历史默认档。
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path

import pytest

from arkrag import pipeline
from arkrag import retrieve
from arkrag import server

ROOT = Path(__file__).resolve().parents[1]

#: 实测读数（`scripts/eval_answer_quality.py --split {dev,holdout,clean} --prod`）。
#: 三份必须一起报：dev 81.05% / holdout 78.97% / clean 81.67%。
#:
#: **2026-11-10 更新**（W3 `chapter_cap=4` 进生产档）：此前是
#: 228/285、152/195、97/120（80.00% / 77.95% / 80.83%）。新版三份全为正、
#: 零掉分，增益全部来自 `cross_chunk`（23→26 / 16→18 / 12→13）。
#: 依据：`docs/W3-验收结论-20261110.md`。
MEASURED = {"dev": "231/285", "holdout": "154/195", "clean": "98/120"}


def _module_by_path(name: str, rel: str):
    """按路径加载脚本（`scripts/` 不是包的一部分）。"""
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _call_sites(src: str, func_name: str) -> list[ast.Call]:
    tree = ast.parse(src)
    out: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = getattr(f, "attr", "") or getattr(f, "id", "")
        if name == func_name:
            out.append(node)
    return out


class TestProfileNumbersArePinned:
    def test_budget_is_the_measured_value(self):
        """预算是 24000，不是 12000 也不是更大。"""
        got = pipeline.PROD_PROFILE["max_context_tokens"]
        assert got == 24000, (
            f"生产档预算变成了 {got}。台账读数是在 24000 上量的："
            f"12000 档下 r12000 预留会把 direct_cap 压到 0，"
            f"7 档网格最高只到 75.44%（dev 215/285）；"
            f"而 40000 以上没有独立证据（adversary Q2 判该旋钮饱和在 73.68%）。"
            f"改这个数必须重跑 dev/holdout/clean 三份。"
        )

    def test_all_switches_are_on(self):
        """五个开关缺一个就掉档，所以五个都必须点名钉住。"""
        opts = pipeline.PROD_PROFILE["options"]
        assert opts == {
            "quote_locate": True,
            "bridge": True,
            "bridge_search_top_k": 40,
            "graph_reserve_tokens": 12000,
            "chapter_cap": 4,
        }, (
            f"生产档的开关集合变了：{opts}。五个数各有实测依据——"
            f"bridge 与 graph_reserve_tokens 必须同时开（只开桥接不预留，"
            f"dev multihop 只 8→11）；bridge_search_top_k=40 只对桥接题生效"
            f"（全局抬 top_k 会掉 single_chunk 1 题）；"
            f"quote_locate 的门控命中集是 single_chunk 128/640、其余类别 0；"
            f"chapter_cap=4 是 W3 实测最优点（三份 split 全为正、零掉分："
            f"dev +3 / holdout +2 / clean +1），且必须**正好 4**——"
            f"cap=3 是 +4/−6（净 −2），cap=5/6 都是 +0/−1。"
        )
        # W2 的 `bridge_policy` **刻意不在**生产档里：它实测零收益
        # （dev 228→228），见 `docs/W2-验收结论-20261110.md`。
        assert "bridge_policy" not in opts, (
            "`bridge_policy` 被加进了生产档：W2 实测 dev Δ=0、holdout/clean 未跑，"
            "零收益的开关不该进生产默认")
        # W1 的 `max_direct_units` 同理：dev Δ=−2，已按反证条件回滚。
        assert "max_direct_units" not in opts, (
            "`max_direct_units` 被加进了生产档：W1 实测 dev Δ=−2（226/285），"
            "反证条件 1 触发，已回滚")

    def test_profile_is_a_mapping_and_options_is_a_nested_dict(self):
        """形状也要钉：`apply_prod_profile` 会做 `dict(...)` 与 `int(...)`。"""
        assert isinstance(pipeline.PROD_PROFILE, dict)
        assert isinstance(pipeline.PROD_PROFILE["options"], dict)
        assert isinstance(pipeline.PROD_PROFILE["max_context_tokens"], int)
        assert not isinstance(pipeline.PROD_PROFILE["max_context_tokens"], bool)


class TestApplyProdProfile:
    def test_budget_lands_on_the_instance_not_the_module_constant(self):
        """预算只许改实例属性。

        改 `retrieve.DEFAULT_BUDGET` 会让 `scripts/` 下 40 多个历史脚本、
        以及台账里 6000/12000 档的每一格读数一起漂；`tests/test_context_budget.py`
        另有门钉住 DEFAULT_BUDGET 就是 12000。
        """
        before = retrieve.DEFAULT_BUDGET["max_context_tokens"]
        ret = _FakeRetriever()
        pipe = {"retriever": ret}
        pipeline.apply_prod_profile(pipe)
        assert ret.budget["max_context_tokens"] == 24000
        assert retrieve.DEFAULT_BUDGET["max_context_tokens"] == before, (
            "`apply_prod_profile` 动了 `retrieve.DEFAULT_BUDGET`：历史脚本的读数会一起变"
        )

    def test_returns_the_options_to_pass_to_ask_once(self):
        pipe = {"retriever": _FakeRetriever()}
        opts = pipeline.apply_prod_profile(pipe)
        assert opts == pipeline.PROD_PROFILE["options"]
        assert opts is not pipeline.PROD_PROFILE["options"], (
            "返回的是常量本体：调用方改一下就把生产档改了"
        )


class TestProdOptionsIsACopy:
    def test_mutating_the_copy_does_not_pollute_the_profile(self):
        opts = pipeline.prod_options()
        opts["bridge"] = False
        opts["graph_reserve_tokens"] = 0
        assert pipeline.PROD_PROFILE["options"]["bridge"] is True
        assert pipeline.PROD_PROFILE["options"]["graph_reserve_tokens"] == 12000

    def test_two_calls_are_independent(self):
        a = pipeline.prod_options()
        b = pipeline.prod_options()
        assert a == b and a is not b


class TestServerWiresTheOptions:
    def test_server_query_passes_options_to_ask_once(self):
        """`server._query` 的 `ask_once` 必须收到生产档 options。

        不传的话（历史实现就是那样）线上跑的是 69.47% 的历史默认档，
        而用户看到的仍然是"答出来了、有引用"——没有任何迹象表明掉了 10.5 个点。
        """
        src = (ROOT / "src" / "arkrag" / "server.py").read_text("utf-8")
        sites = [c for c in _call_sites(src, "ask_once")]
        assert sites, "`server.py` 里找不到 `ask_once` 调用点"
        wired = [
            c for c in sites
            if any(kw.arg == "options" for kw in c.keywords)
        ]
        assert wired, (
            "`server.py` 调 `ask_once` 时没有传 `options`："
            "线上会跑历史默认档（dev 69.47%）而不是生产档（80.00%）。"
        )

    def test_server_derives_the_budget_from_the_profile(self):
        """服务端默认预算必须等于 `PROD_PROFILE`，不许退回检索库默认。"""
        assert server.DEFAULT_MAX_CONTEXT_TOKENS == \
            pipeline.PROD_PROFILE["max_context_tokens"]
        assert server.DEFAULT_MAX_CONTEXT_TOKENS != \
            retrieve.DEFAULT_BUDGET["max_context_tokens"]


class TestCliWiresTheOptions:
    @pytest.fixture(scope="class")
    def mod(self):
        return _module_by_path("ask_prod_under_test", "scripts/ask.py")

    def test_every_ask_once_call_site_passes_options(self):
        """`scripts/ask.py` 有两处 `ask_once`（evaluate 循环 + 单问），两处都要传。"""
        src = (ROOT / "scripts" / "ask.py").read_text("utf-8")
        sites = _call_sites(src, "ask_once")
        assert len(sites) >= 2, (
            f"`scripts/ask.py` 只有 {len(sites)} 处 `ask_once`，"
            f"预期至少 2 处（evaluate 循环 + 单问）"
        )
        missing = [c.lineno for c in sites if not any(kw.arg == "options" for kw in c.keywords)]
        assert not missing, (
            f"`scripts/ask.py` 第 {missing} 行的 `ask_once` 没传 `options`："
            f"那条通路的 `--prod` 是死开关"
        )

    def test_prod_flag_exists_and_defaults_off(self, mod):
        """`--prod` 必须存在且默认关。

        默认关是硬要求：这个脚本的默认口径要能复现台账里 6000/12000 档的旧读数，
        改成默认开会让几十个历史数字失去可比性。

        ## 为什么改成纯 AST 静态检查，而不是起一个假 parser 调 `mod.main([])`

        原实现替换 `mod.argparse.ArgumentParser` 再调 `mod.main([])`，有两个
        无法回避的缺陷：

        1. **无限递归**：`mod.argparse` 与测试自己 import 的 `argparse` 是
           **同一个模块对象**，替换它等于全局替换。`_Cap` 继承自
           `argparse.ArgumentParser`，而后者此时已是 `_Cap` 本身 —— 构造
           `_Cap()` 要引用 `_Cap`，`RecursionError`。这条门此前一直报假红。
        2. **它验证的东西比看上去少**：即使不递归，`mod.main([])` 也会走到
           `question == ""` 的早退分支，真正想钉的"`--prod` 默认关"根本没被执行。

        静态读源码没有这两个问题：`add_argument("--prod", action="store_true")`
        这件事在 AST 里是确定的，且不依赖任何运行期替换。
        """
        src = (ROOT / "scripts" / "ask.py").read_text("utf-8")
        tree = ast.parse(src)

        prod_args: list[ast.Call] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if (getattr(fn, "attr", "") or getattr(fn, "id", "")) != "add_argument":
                continue
            for arg in node.args:
                if isinstance(arg, ast.Constant) and arg.value == "--prod":
                    prod_args.append(node)

        assert prod_args, "`scripts/ask.py` 没有 `--prod`"
        call = prod_args[0]

        # `store_true` 的默认值语义由 argparse 保证是 False；若显式写了
        # `default=`，则必须是 `False`，不许是 `True`。
        for kw in call.keywords:
            if kw.arg == "default":
                assert isinstance(kw.value, ast.Constant) \
                    and kw.value.value is False, (
                    "`--prod` 的默认值必须是 False："
                    "默认开会让台账里 6000/12000 档的读数失去可比性"
                )
        actions = {
            kw.value.value
            for kw in call.keywords
            if kw.arg == "action" and isinstance(kw.value, ast.Constant)
        }
        has_store_true = "store_true" in actions
        has_false_default = any(
            kw.arg == "default" and isinstance(kw.value, ast.Constant)
            and kw.value.value is False
            for kw in call.keywords)
        assert has_store_true or has_false_default, (
            f"`--prod` 既不是 `action='store_true'` 也没显式给 "
            f"`default=False`（实际 action={actions}）：无法证明它默认关"
        )

    def test_evaluate_accepts_prod_and_defaults_off(self, mod):
        params = inspect.signature(mod.evaluate).parameters
        assert "prod" in params, (
            "`evaluate()` 没有 `prod` 参数：`--evaluate --prod` 会静默跑历史默认档"
        )
        assert params["prod"].default is False, (
            "`evaluate()` 的 `prod` 默认值必须是 False"
        )


class TestEvalScriptWiresTheOptions:
    @pytest.fixture(scope="class")
    def mod(self):
        return _module_by_path("eval_aq_prod_under_test", "scripts/eval_answer_quality.py")

    def test_run_accepts_prod_and_defaults_off(self, mod):
        params = inspect.signature(mod.run).parameters
        assert "prod" in params, (
            "`run()` 没有 `prod` 参数：`--prod` 收不到评测链里"
        )
        assert params["prod"].default is False, (
            "`run()` 的 `prod` 默认值必须是 False："
            "台账里 6000/12000 档的每一格读数都是在这个默认分支下测的"
        )

    def test_run_passes_options_to_ask_once(self, mod):
        src = (ROOT / "scripts" / "eval_answer_quality.py").read_text("utf-8")
        sites = _call_sites(src, "ask_once")
        assert sites, "`eval_answer_quality.py` 里找不到 `ask_once` 调用点"
        assert any(kw.arg == "options" for kw in sites[0].keywords), (
            "`eval_answer_quality.py` 的 `ask_once` 没传 `options`："
            "`--prod` 只改了预算，三个检索开关没生效"
        )

    def test_cli_exposes_prod_and_defaults_off(self):
        src = (ROOT / "scripts" / "eval_answer_quality.py").read_text("utf-8")
        tree = ast.parse(src)
        call = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
                if node.args and isinstance(node.args[0], ast.Constant) \
                        and node.args[0].value == "--prod":
                    call = node
        assert call is not None, "`eval_answer_quality.py` 没有 `--prod`"
        flags = [kw for kw in call.keywords if kw.arg == "action"]
        assert flags and getattr(flags[0].value, "value", "") == "store_true", (
            "`--prod` 应当是 `store_true`（默认关）"
        )

    def test_report_records_whether_the_prod_profile_was_used(self):
        """报告必须自己写清楚跑的是哪一档。

        不然两份 json 放在一起，80.00% 和 69.47% 分不出谁是谁。
        """
        src = (ROOT / "scripts" / "eval_answer_quality.py").read_text("utf-8")
        assert '"prod_profile"' in src, "评测报告里没有 `prod_profile` 字段"
        assert '"prod_options"' in src, "评测报告里没有 `prod_options` 字段"


class TestDefaultPathIsUnchanged:
    def test_options_none_keeps_the_historical_caliber(self):
        """`options=None` 必须逐位等于改前行为。

        `ask_once` 的默认分支就是历史默认档；`scripts/` 下 40 多个历史脚本
        与台账里每一格旧读数都靠它。任何"顺手把默认档也改了"的改动都会让
        那些数字失去可比性，而且不会有任何测试失败提示。
        """
        src = (ROOT / "src" / "arkrag" / "pipeline.py").read_text("utf-8")
        fn = None
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.FunctionDef) and node.name == "ask_once":
                fn = node
        assert fn is not None
        # `options` 是**关键字参数**（在 `*` 之后），所以它在 `kwonlyargs` 里，
        # 不在 `args` 里。原来只查 `fn.args.args` 会恒判「没有 options 参数」——
        # 这条门一直报的是假红（方向错了：它说"参数没了"，其实参数好好的）。
        # 本次修正只改观测面，不放宽门槛：查不到仍然是 FAIL。
        all_params = list(fn.args.args) + list(fn.args.kwonlyargs)
        opts = [a for a in all_params if a.arg == "options"]
        assert opts, "`ask_once` 没有 `options` 参数"
        # 关键字参数的默认值在 `kw_defaults`（与 `args` 位置对齐，可能含 None），
        # 不在 `defaults` 里。
        idx = [a.arg for a in fn.args.kwonlyargs].index("options")
        default = fn.args.kw_defaults[idx]
        assert isinstance(default, ast.Constant) and default.value is None, (
            "`ask_once` 的 `options` 默认值不是 None："
            "默认档一变，台账里 6000/12000 档的读数全部失去可比性"
        )

    def test_prod_profile_is_not_applied_by_build_pipeline(self):
        """`build_pipeline` 不许偷偷应用生产档。

        它被 40 多个脚本当"造一条链"用；偷偷改预算的话，那些脚本的读数会一起漂，
        而它们的源码一个字都没改。
        """
        src = (ROOT / "src" / "arkrag" / "pipeline.py").read_text("utf-8")
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.FunctionDef) and node.name == "build_pipeline":
                body = ast.get_source_segment(src, node) or ""
                assert "apply_prod_profile" not in body, (
                    "`build_pipeline` 里应用了生产档：40 多个历史脚本的读数会静默漂移"
                )
                assert "PROD_PROFILE" not in body, (
                    "`build_pipeline` 读了 `PROD_PROFILE`：同上"
                )


class _FakeRetriever:
    """`apply_prod_profile` 只碰 `budget`，所以桩只要能装它。"""

    def __init__(self) -> None:
        self.budget = dict(retrieve.DEFAULT_BUDGET)
