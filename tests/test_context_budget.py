"""上下文预算 `max_context_tokens` 从 6000 改成 12000 的门。

## 这个文件守的是什么

预算从 6000 抬到 12000 是一次**有实测依据**的改动，不是调参：

* dev 350 题里可判 285 题：6000 档 175/285（61.40%），12000 档 198/285（69.47%）。
  配对 bootstrap 4000 次：+8.07pt，95%CI[+4.56,+11.58]，P(Δ<=0)=0.0000，+26/-3。
  这两个数是**真实管线**量的（`scripts/eval_answer_quality.py`，生产配置
  MAX_CLAIMS=8、`pipeline.ask_once` 全序、`judge.py` 判定），不是回放。
  早先报的 191 -> 225 是 `cross_chunk` 用双向集合相等、`multihop` 只要 ≥1 锚点那套
  判据下的读数，**已作废**（前者惩罚正确答案、后者 57/73 道通过题只命中一个锚点）。
  更早报的 174 -> 200 混进了 `MAX_CLAIMS=12`（回放臂 `c12_*`），也与生产配置不符。
  174 -> 191 那一版是 `MAX_CLAIMS=5` 时代的读数，随 `MAX_CLAIMS` 抬到 8 一起作废。
* holdout 238 题里可判 195 题：6000 档 119/195（61.03%），12000 档 132/195（67.69%）。
  bootstrap：+6.67pt CI[+2.56,+10.77] P=0.0000，+15/-2。
  holdout 里有 94 题与 gold-v1 重复（clean holdout 只有 144 题、可判 120 题，
  77/120 -> 83/120，+5.00pt CI[+0.00,+10.00] P=0.0315，CI 下界正好触 0、
  n=120 对 +5pt 的功效只有约 46%），所以三个数字必须一起看。
* 再往上不再测到 40000（回放臂 `c12_40000` 的 dev 214/285 = 75.09% 含 MAX_CLAIMS=12，
  不能与生产配置比较）。已知代价：12000 档上下文 9.85 单位 / 8,161 字，40000 档
  28.2 单位 / 24,301 字（2.9 倍），延迟却几乎不变（P50 66.4ms vs 6000 档 70.9ms，
  见 `tmp_latency_12k.py`）。停在 12000 的理由是 clean holdout 那一格
  （n=120、+5.00pt、p=0.0315、CI 下界触 0）已经吃力，再往上加没有独立证据支撑。

数字改了但**没有门**，下一个动这个常数的人就不会知道这些代价。本文件钉死：

1. 默认值就是 12000（改回 6000 立刻红，失败文案打印上面那两组数字）。
2. 默认值真的生效在 `build_context` 里（不是只写在常量里没人用）。
3. 历史口径可复现：显式 `token_budget=6000` 仍然是"只装 5 块"。
4. 服务端**不再手抄**这个数（抄两处是本项目记录在案的事故形态），且服务端默认
   取的是 `pipeline.PROD_PROFILE`（生产档），刻意**不等于**检索库默认——两者相等
   就意味着线上跑的是历史默认档（69.47%）而不是生产档。
5. 已知缺陷（预算非单调）用一条 `xfail(strict=True)` 钉住并写清机理。

## 为什么用例自己造块，不用真语料

真语料要一整个世代，跑得慢且数字随语料漂移。这里用可控长度的合成块：
`estimate_tokens` 是词法口径（约 1.7 token/字），块长 600 字时单价约 1018
token，于是 6000 装 5 块、12000 装 11 块——正好把两档分开（实测读数见
`docs/_tmp_budget_wire_report.md`）。
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
import re
from pathlib import Path

import pytest

from arkrag import adjacency as adjacency_mod
from arkrag import config
from arkrag import index as index_mod
from arkrag import pipeline as pipeline_mod
from arkrag import query as query_mod
from arkrag import retrieve
from arkrag import server
from arkrag import vocab

ROOT = Path(__file__).resolve().parents[1]

#: 合成语料用的对话正文。重复到目标长度即可，内容本身不参与判定。
PAD = (
    "凯尔希：切尔诺伯格已经沦陷了，我们必须在天亮之前离开这里。"
    "阿米娅：可是罗德岛的人还在城里，我们不能丢下他们。"
    "凯尔希：博士，罗德岛不会为一个人的牺牲改变航向。"
)

DEFAULT_N = 12000
LEGACY_N = 6000

DEV_TEXT = "dev 175/285 (61.40%) -> 198/285 (69.47%)"
HOLDOUT_TEXT = "holdout 119/195 (61.03%) -> 132/195 (67.69%)"
CLEAN_TEXT = "clean holdout 77/120 (64.17%) -> 83/120 (69.17%)"


def _text(n_chars: int) -> str:
    out: list[str] = []
    while len("".join(out)) < n_chars:
        out.append(PAD)
    return "".join(out)[:n_chars]


def _chunk(cid: str, text: str) -> dict:
    return {
        "chunk_id": cid,
        "doc_id": cid.split("#")[0],
        "title": "测试篇",
        "chapter": "第一章",
        "text": text,
        "chars": len(text),
    }


def _make_retriever(chunks: list[dict], *, budget: dict | None = None):
    """按给定块造索引 + 一个只加载这些块的 Retriever。"""
    entities = [{"id": "e-0000000000000001", "name": "凯尔希", "type": "角色", "aliases": []}]
    edges: list[dict] = []
    graph = {
        "schema_version": config.GRAPH_SCHEMA_IDENTITY,
        "entities": entities,
        "edges": edges,
        "alias_index": index_mod.build_alias_index(entities),
        "fingerprint": "0123456789abcdef",
        "stats": {"entities": 1, "edges": 0},
    }
    adj = adjacency_mod.build_directed_adjacency(edges)
    idx, _meta = index_mod.build(chunks, graph, "corpusfp01234567")
    ret = retrieve.Retriever(
        idx, graph, adj, chunks=chunks, assertions=[], budget=budget, vocab=vocab
    )
    qa = query_mod.analyze_query("凯尔希和罗德岛是什么关系", entities=entities, vocab=vocab)
    return ret, qa


def _hits(chunks: list[dict]) -> list[dict]:
    """全部块都当直接命中（`SOURCE_BM25`），分数递减以固定排序。"""
    return [
        {
            "chunk_id": c["chunk_id"],
            "score": round(1.0 - i * 0.01, 4),
            "sources": [{"channel": retrieve.SOURCE_BM25, "score": 1.0}],
        }
        for i, c in enumerate(chunks)
    ]


def _uniform_corpus(n_units: int = 24, n_chars: int = 600):
    """24 块等长块。6000 装 5 块、12000 装 11 块（实测见报告）。"""
    chunks = [_chunk("doc/a#%03d" % i, _text(n_chars)) for i in range(n_units)]
    ret, qa = _make_retriever(chunks)
    return ret, qa, _hits(chunks)


def _unit_cost(ret, chunk: dict) -> int:
    return retrieve.estimate_tokens(
        f"{chunk['title']} {chunk['chapter']} {chunk['text']}",
        tokenizer_version=getattr(ret.index, "tokenizer_version", index_mod.TOKENIZER_VERSION),
        name_terms=getattr(ret.index, "name_terms", None),
    )


def _derived_default() -> int:
    return int(retrieve.DEFAULT_BUDGET["max_context_tokens"])


class TestDefaultValueIsPinned:
    def test_default_budget_is_12000(self):
        """改回 6000 立刻红：这个数字是拿两套题量出来的，不是随手写的。"""
        got = retrieve.DEFAULT_BUDGET["max_context_tokens"]
        assert got == DEFAULT_N, (
            f"上下文预算变成了 {got}，登记值是 {DEFAULT_N}。"
            f"改回 {LEGACY_N} 会让实测退回 {DEV_TEXT}；{HOLDOUT_TEXT}；{CLEAN_TEXT}。"
            f"（holdout 里 94 题与 gold-v1 重复，clean holdout 只有 144 题、"
            f"可判 120 题，所以那三个数字必须一起报。）"
            f"要改请先重跑 dev 与 holdout 两套题，别只改这个常数。"
        )

    def test_default_is_an_int_and_not_a_downgrade(self):
        """`DEFAULT_BUDGET` 取键会 KeyError（不静默回落），且不许小于历史口径。"""
        assert "max_context_tokens" in retrieve.DEFAULT_BUDGET
        value = retrieve.DEFAULT_BUDGET["max_context_tokens"]
        assert isinstance(value, int) and not isinstance(value, bool), (
            f"预算必须是 int，实际是 {type(value).__name__}："
            f"`build_context` 里要做 `int(...)` 与加法比较，float 会让边界抖动。"
        )
        assert value >= LEGACY_N, (
            f"预算 {value} 小于历史口径 {LEGACY_N}：这不是抬预算，是悄悄缩水，"
            f"会让 dev 与 holdout 两套题一起变差（{DEV_TEXT}）。"
        )


class TestDefaultTakesEffectInBuildContext:
    def test_default_budget_fits_more_than_five_units(self):
        """默认值必须真的作用到 `build_context`，而不是只躺在常量里。

        24 块、每块 600 字、单价约 1018 token：6000 只装得下 5 块，
        12000 装 11 块。装不下 6 块就说明默认值没生效（或被人改回去了）。
        """
        ret, qa, hits = _uniform_corpus()
        stats = ret.build_context(hits, [], query_analysis=qa)["stats"]
        assert stats["token_budget"] == DEFAULT_N, (
            f"build_context 用的预算 {stats['token_budget']} 不是默认值 {DEFAULT_N}"
        )
        assert stats["units"] > 5, (
            f"默认 {DEFAULT_N} token 只装下 {stats['units']} 块"
            f"（用了 {stats['tokens_used']} token，候选 {stats['candidates']} 块）。"
            f"12000 档实测应装 11 块；装不下 6 块说明预算没生效。"
        )
        assert stats["tokens_used"] <= DEFAULT_N

    def test_explicit_6000_reproduces_the_historical_caliber(self):
        """历史口径必须能复现：`token_budget=6000` 仍然是"只装 5 块"。

        台账里 6000 档的全部读数（dev 175/285、holdout 119/195）都是在这个口径下
        量的。如果 6000 现在装得下 6 块，那些数字就再也对不上了。
        """
        ret, qa, hits = _uniform_corpus()
        legacy = ret.build_context(hits, [], token_budget=LEGACY_N, query_analysis=qa)["stats"]
        assert legacy["token_budget"] == LEGACY_N
        assert legacy["units"] == 5, (
            f"显式 6000 装下了 {legacy['units']} 块（用了 {legacy['tokens_used']} token），"
            f"实测应是 5 块。6000 档的历史读数（{DEV_TEXT}；{HOLDOUT_TEXT}）就是在"
            f"这个口径下量的，装块数一变，那些数字即失效。"
        )
        assert legacy["tokens_used"] <= LEGACY_N

        wide = ret.build_context(hits, [], token_budget=DEFAULT_N, query_analysis=qa)["stats"]
        assert wide["units"] > legacy["units"], (
            f"12000 装 {wide['units']} 块，不比 6000 的 {legacy['units']} 块多——预算没起作用。"
        )

    def test_instance_budget_is_the_entry_ask_py_uses(self):
        """`scripts/ask.py --token-budget` 走的是 `Retriever.budget`，不是实参。

        `build_context` 不传 `token_budget` 时读 `self.budget["max_context_tokens"]`，
        所以实例属性必须能改变默认行为；否则 CLI 那条覆盖语句是死的。
        """
        ret, qa, hits = _uniform_corpus()
        before = ret.build_context(hits, [], query_analysis=qa)["stats"]
        assert before["token_budget"] == DEFAULT_N

        ret.budget["max_context_tokens"] = LEGACY_N
        after = ret.build_context(hits, [], query_analysis=qa)["stats"]
        assert after["token_budget"] == LEGACY_N, (
            "改了 retriever.budget['max_context_tokens'] 但 build_context 没跟着变："
            "`scripts/ask.py --token-budget` 会变成一条不生效的摆设。"
        )
        assert after["units"] < before["units"]

    def test_truncation_is_reported_not_silent(self):
        """被预算丢掉的块必须出现在 `truncated_chunks` 里，且账要对得上。"""
        ret, qa, hits = _uniform_corpus()
        ctx = ret.build_context(hits, [], token_budget=LEGACY_N, query_analysis=qa)
        stats = ctx["stats"]
        assert stats["truncated_chunks"], "超预算了却没有任何块被登记为截断"
        assert len(ctx["units"]) + len(stats["truncated_chunks"]) == stats["candidates"], (
            f"装下 {len(ctx['units'])} + 截断 {len(stats['truncated_chunks'])} "
            f"!= 候选 {stats['candidates']}：有块既没进上下文也没记账。"
        )
        accounted = sum(u["tokens"] for u in ctx["units"])
        assert accounted == stats["tokens_used"], (
            f"各 unit 的 tokens 之和 {accounted} != stats['tokens_used'] "
            f"{stats['tokens_used']}：用量账与实际装进去的块对不上，"
            f"预算读数就不可信了。"
        )
        assert stats["tokens_used"] <= LEGACY_N


class TestKnownDefect:
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "已知缺陷：build_context 超预算时用 continue 而非 break，"
            "预算变大反而可能丢掉小预算里装得下的块"
        ),
    )
    def test_known_bug_bigger_budget_drops_a_chunk_that_used_to_fit(self):
        """已知缺陷（未修）：预算非单调，加大预算反而丢掉本来就装下的块。

        机理在 `retrieve.build_context` 的填充循环里：超预算时是 `continue`
        而不是 `break`，"这一块装不下"只是跳过它、继续试后面的块。于是：

        * 小预算：长块 B 装不下 -> 跳过 -> 后面的短块 C 装进来了；
        * 预算加大：长块 B 装进来了 -> 剩余额度不够 C -> C 被丢。

        "预算变大"丢掉了小预算里本来就在的块。真语料实测 237 题里有 4 题
        出现单调性违规（q-tc-000016 / q-dr-000041 等），见 `docs/升级台账.md`
        的两条遗留。

        写成 `xfail(strict=True)`：缺陷如果被修好（改成全局重选而不是 `continue`），
        这条会 XPASS 从而让套件变红，逼迫有人来更新这条记录——不允许它悄悄
        一直 xfail 下去。
        """
        a = _chunk("doc/a#000", _text(600))   # 中块，排第一
        b = _chunk("doc/a#001", _text(900))   # 最大的块，排第二
        c = _chunk("doc/a#002", _text(100))   # 短块，排第三
        chunks = [a, b, c]
        ret, qa = _make_retriever(chunks)
        hits = _hits(chunks)

        cost_a, cost_b, cost_c = _unit_cost(ret, a), _unit_cost(ret, b), _unit_cost(ret, c)
        assert cost_b > cost_c, "构造错误：第二个块必须比第三个块贵，否则造不出这个形状"

        small_budget = cost_a + cost_c   # 装得下 A 与 C，装不下 B
        large_budget = cost_a + cost_b   # 装得下 A 与 B，剩余额度不够 C
        assert large_budget > small_budget

        small = ret.build_context(hits, [], token_budget=small_budget, query_analysis=qa)
        large = ret.build_context(hits, [], token_budget=large_budget, query_analysis=qa)
        small_ids = [u["chunk_id"] for u in small["units"]]
        large_ids = [u["chunk_id"] for u in large["units"]]

        assert small_ids == ["doc/a#000", "doc/a#002"], (
            f"构造失败：小预算 {small_budget} 下应装下 A 与 C，实际 {small_ids}"
        )
        assert "doc/a#002" in large_ids, (
            f"预算从 {small_budget} 涨到 {large_budget}，短块 doc/a#002 却从上下文里"
            f"消失了（大预算装 {large_ids}，小预算装 {small_ids}）。"
            f"这是 build_context 用 continue 而非 break 造成的非单调缺陷。"
        )


class TestServerDoesNotCopyTheNumber:
    def test_server_default_matches_the_production_profile(self):
        """服务端默认值必须与 `pipeline.PROD_PROFILE` 相等（派生，不是手抄）。

        为什么改了对象：服务端服务的是**生产档**（引文定位 + 桥接 + 图证据预留 +
        24000 预算），这一整套只有 `PROD_PROFILE` 一个出处。预算若继续从
        `retrieve.DEFAULT_BUDGET` 取，就会拼出"开了桥接但预算还是 12000"的半套
        配置——实测那组合比不开桥接还差（dev 最高只到 75.44%）。
        `retrieve.DEFAULT_BUDGET` 仍是 12000：那是**检索库的默认**，台账里 6000/
        12000 档的历史读数与 `scripts/` 下 40 多个脚本靠它复现，两者不是同一个数。
        """
        want = int(pipeline_mod.PROD_PROFILE["max_context_tokens"])
        assert server.DEFAULT_MAX_CONTEXT_TOKENS == want, (
            f"`server.DEFAULT_MAX_CONTEXT_TOKENS` = {server.DEFAULT_MAX_CONTEXT_TOKENS}"
            f" != `pipeline.PROD_PROFILE` 的 {want}：两处各写一个数是本项目"
            f"记录在案的事故形态（改一处忘一处）。"
        )
        assert server.Settings().max_context_tokens == want
        assert server.Settings.from_env().max_context_tokens == want, (
            "`Settings.from_env()` 在没配环境变量时也必须取派生值"
        )

    def test_server_never_falls_back_to_the_library_default(self):
        """服务端默认预算**不等于** `retrieve.DEFAULT_BUDGET`，这是刻意的。

        两者相等的话，说明有人把服务端改回了检索库默认，于是"线上服务"与
        "台账 80% 读数"跑的不是同一档：生产档实测 dev 80.00%，历史默认档 69.47%。
        """
        assert server.DEFAULT_MAX_CONTEXT_TOKENS != retrieve.DEFAULT_BUDGET["max_context_tokens"], (
            "服务端默认预算又等于 `retrieve.DEFAULT_BUDGET` 了：那意味着线上跑的是"
            "历史默认档（69.47%），不是生产档。请从 `pipeline.PROD_PROFILE` 派生。"
        )

    def test_server_source_has_no_hardcoded_budget_literal(self):
        """静态门：`server.py` 里不许再跟一个手抄的数字。

        两个形态都要拦：
        * `max_context_tokens: int = 6000` / `max_context_tokens=6000`（字段与赋值）；
        * `_int("ARKRAG_SERVE_MAX_CONTEXT_TOKENS", 6000)`（环境变量默认值）。

        抄两处的代价是真实的：`/query` 返回的 `limits.max_context_tokens` 与
        真正生效的预算各说各话，报告照常打印，看不出异常。
        """
        path = ROOT / "src" / "arkrag" / "server.py"
        patterns = [
            re.compile(r"max_context_tokens\s*\??\s*[:=]\s*(\d+)"),
            re.compile(r'ARKRAG_SERVE_MAX_CONTEXT_TOKENS"\s*,\s*(\d+)'),
        ]
        offenders: list[str] = []
        for i, line in enumerate(path.read_text("utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            for pat in patterns:
                if pat.search(line):
                    offenders.append(f"{i}: {line.strip()}")
                    break
        assert not offenders, (
            f"`server.py` 里又有地方把手抄的数字写死了：{offenders}。"
            f"请改成从 `retrieve.DEFAULT_BUDGET['max_context_tokens']` 派生"
            f"（见 `DEFAULT_MAX_CONTEXT_TOKENS`），不要写第二处。"
        )


def _ask_module():
    """按路径加载 `scripts/ask.py`（它不是包的一部分）。"""
    path = ROOT / "scripts" / "ask.py"
    spec = importlib.util.spec_from_file_location("ask_budget_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestCliCanReproduceTheOldCaliber:
    def test_token_budget_flag_defaults_to_the_derived_value(self):
        """`--token-budget` 必须存在，且默认值派生自 `DEFAULT_BUDGET`。

        理由：`scripts/` 下 40 多个历史脚本调 `build_pipeline` 时吃的都是默认值。
        默认值一改，它们的历史读数就全变了——除非有个开关能把口径拨回去。
        默认值写死成 12000 也不行：那样预算又变成两处判定。
        """
        src = (ROOT / "scripts" / "ask.py").read_text("utf-8")
        tree = ast.parse(src)
        call = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if getattr(node.func, "attr", "") != "add_argument":
                continue
            if node.args and isinstance(node.args[0], ast.Constant) \
                    and node.args[0].value == "--token-budget":
                call = node
        assert call is not None, (
            "`scripts/ask.py` 没有 `--token-budget`：台账里 6000 档的读数再也"
            "复现不出来了（历史上 40 多个脚本吃的都是默认值）。"
        )
        defaults = [kw for kw in call.keywords if kw.arg == "default"]
        assert defaults, "`--token-budget` 没有 default，等于没有默认口径"
        seg = ast.get_source_segment(src, defaults[0].value) or ""
        assert "DEFAULT_BUDGET" in seg, (
            f"`--token-budget` 的默认值是 {seg!r}，不是从 DEFAULT_BUDGET 派生的。"
            f"写死一个数会让预算重新变成两处判定。"
        )

    def test_cli_default_equals_retrieve_default_at_runtime(self):
        """运行时把参数表解析出来，默认值必须等于 `DEFAULT_BUDGET` 的当前值。

        `main([])` 在缺问题时打印一行提示并返回 2（不抛 `SystemExit`），
        所以这里拦 `parse_args` 顺手读出参数表里的 default，再用返回值判定。
        """
        mod = _ask_module()
        parser_calls: list[dict] = []
        real_parse_args = mod.argparse.ArgumentParser.parse_args

        def _capture(self, *a, **kw):
            actions = getattr(self, "_option_string_actions", {}) or {}
            if "--token-budget" in actions:
                parser_calls.append(actions["--token-budget"].default)
            return real_parse_args(self, *a, **kw)

        mod.argparse.ArgumentParser.parse_args = _capture
        try:
            rc = mod.main([])
        finally:
            mod.argparse.ArgumentParser.parse_args = real_parse_args

        assert rc == 2, f"`main([])` 应当返回 2（缺问题），实际返回 {rc}"
        assert parser_calls, "没能拦截到带 --token-budget 的参数表解析"
        assert parser_calls[0] == _derived_default(), (
            f"`--token-budget` 的运行时默认值是 {parser_calls[0]!r}，"
            f"不等于 retrieve.DEFAULT_BUDGET 的 {_derived_default()}"
        )

    def test_evaluate_accepts_and_applies_the_budget(self):
        """`evaluate()` 也必须能收到预算，并在 `build_pipeline` 之后应用它。

        只给 `main()` 加参数而不接进 `evaluate()` 的话，`--evaluate
        --token-budget 6000` 会安静地按 12000 跑——报告照常打印七项指标。
        """
        mod = _ask_module()
        params = inspect.signature(mod.evaluate).parameters
        assert "token_budget" in params, (
            "`evaluate()` 没有 token_budget 参数：`--evaluate --token-budget 6000` "
            "会静默按默认预算跑，报出的数字不是请求的口径。"
        )
        assert params["token_budget"].default is None, (
            "`evaluate()` 的 token_budget 默认值必须是 None（表示不改、沿用默认），"
            "写成具体数字会让'没传'和'传了默认值'两种情况分不开。"
        )
        src = (ROOT / "scripts" / "ask.py").read_text("utf-8")
        applied = src.count('budget["max_context_tokens"] = int(')
        assert applied == 2, (
            f"`scripts/ask.py` 里覆盖预算的赋值只有 {applied} 处，应当是 2 处"
            f"（单问分支 + evaluate 分支）。少一处就意味着有一条通路的预算开关是死的。"
        )
