"""`scripts/k2_perf.py` 的自检：**每一条门都必须能失败**。

`scripts/k2_perf.py` 的价值不在「跑出七条 PASS」，而在「条件不成立时会变红」。
一个永远返回 `ok=True` 的门比没有门更糟：它会让 `data/k2_perf.json` 里多出几行
PASS，而没人知道那几行是怎么来的。

所以这里对每条门各构造一次「本该拦住」的输入，断言它确实返回 `ok=False`，
并且 `detail` 说的是那件事；同时留一条**反向对照**（同样输入、只改一个变量就变绿），
证明它不是「总是红」。

## 为什么用合成数据而不是真跑两代

真跑两代要起 4 个子进程、加载 200+ MB JSON、单次十几分钟，而且时序数字每次不同。
判据本身是**纯函数**（吃字典、出结论），所以这里把输入换掉让它自己下结论。
它证明的是「判据写对了」，不证明「盘上现在是好的」——后者由手工跑一次的实测输出负责。

## 与卫生门的关系

本文件进入 `tests/test_suite_hygiene.py` 的 `TEST_FILES` 后，四个 parametrize 各多一项
（+4）。`N` 由 Lead 统一上调，本文件不改卫生门。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_k2_perf():
    spec = importlib.util.spec_from_file_location(
        "k2_perf_under_test", ROOT / "scripts" / "k2_perf.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


perf = _load_k2_perf()


# ---------------------------------------------------------------- 夹具
#
# 这些形状照抄真实产物（`data/arkrag/generations/<gid>/index/lexical.json` 的 meta
# 与两代 manifest），只是数值缩小到能一眼看完。判据不关心绝对值，只关心关系。

DIG = "a" * 64


def _measure(**over) -> dict:
    """一份「字段完整、数值正常」的单代测量报告。"""
    base = {
        "generation_id": "g-20261103T132152Z-06a741",
        "graph_schema": "ark-rag-3",
        "generation_status": "READY",
        "load_total_ms": 12345.6,
        "pipeline_ms": 300.0,
        "index_tokenizer": "t2",
        "index_chunk_count": 13798,
        "artifact_bytes": {"graph": 10, "adjacency": 10, "assertions": 10,
                           "index": 10, "chunks": 10},
        "chunks_path": "data/arkrag/generations/g-x/chunks.jsonl",
        "questions": 3,
        "category_mix": {"alias": 1, "direct_relation": 1, "single_chunk": 1},
        "cold": {"n": 3, "p50_ms": 200.0, "p95_ms": 210.0, "p99_ms": 210.0,
                 "min_ms": 150.0, "max_ms": 210.0, "mean_ms": 190.0},
        "warm": {"n": 3, "p50_ms": 190.0, "p95_ms": 200.0, "p99_ms": 200.0,
                 "min_ms": 140.0, "max_ms": 200.0, "mean_ms": 180.0},
        "warm_reps": 2,
        "memory": {
            "at_start": {"current_mib": 26.0, "peak_mib": 26.0},
            "after_load": {"current_mib": 1200.0, "peak_mib": 1400.0},
            "after_pipeline": {"current_mib": 1210.0, "peak_mib": 1400.0},
            "at_end": {"current_mib": 1220.0, "peak_mib": 1500.0},
        },
        "tokens": {
            "context_tokens_p50": None, "context_tokens_p95": None,
            "llm_prompt_tokens": None, "llm_completion_tokens": None,
            "llm_cost_usd": None, "llm_unavailable_reason": "本机无端点",
        },
        "per_question": [],
    }
    base.update(over)
    return base


def _probe(**over) -> dict:
    """一份「载入路径正确、绑定一致」的单代索引探针结果。"""
    base = {
        "generation_id": "g-20261103T132152Z-06a741",
        "index_path": "data/arkrag/generations/g-x/index/lexical.json",
        "index_bytes": 106_685_068,
        "graph_path": "data/arkrag/generations/g-x/graph/graph.json",
        "index_ms": 9000.0,
        "index_peak_mib": 1200.0,
        "opened_json_paths": ["data/arkrag/generations/g-x/index/lexical.json"],
        "opened_json_count": 1,
        "opened_json_bytes": 106_685_068,
        "opened_lexical": True,
        "opened_graph": False,
        "doc_count": 13798,
        "term_count": 390_508,
        "tokenizer_version": "t2",
        "meta_chunk_count": 13798,
        "meta_probe_digest": DIG,
        "manifest_probe_digest": DIG,
        "recomputed_probe_digest": DIG,
    }
    base.update(over)
    return base


def _probe_baseline(**over) -> dict:
    """上一代：更大、词条更多、分词器 t1、文档数更少。"""
    base = _probe(
        generation_id="g-20261031T035231Z-4eafae",
        index_bytes=127_244_898,
        opened_json_bytes=127_244_898,
        doc_count=13781,
        term_count=694_304,
        tokenizer_version="t1",
        meta_chunk_count=13781,
    )
    base.update(over)
    return base


def _measure_baseline(**over) -> dict:
    base = _measure(
        generation_id="g-20261031T035231Z-4eafae",
        index_tokenizer="t1",
        index_chunk_count=13781,
        memory={
            "at_start": {"current_mib": 26.0, "peak_mib": 26.0},
            "after_load": {"current_mib": 1300.0, "peak_mib": 1500.0},
            "after_pipeline": {"current_mib": 1310.0, "peak_mib": 1500.0},
            "at_end": {"current_mib": 1320.0, "peak_mib": 1600.0},
        },
    )
    base.update(over)
    return base


# ---------------------------------------------------------------- 分位数口径
#
# 这两条不是重复实现：`k2_perf` 明令复用 `bench_h1.percentile`，不许另造一套。
# 一旦有人把它换成 numpy/statistics 那种插值分位数，下面的期望值就会变。

def test_percentile_empty_returns_none_not_zero():
    """空样本必须返回 None。返回 0.0 会让「没测到」看起来像「极快」。"""
    assert perf.bench_h1.percentile([], 95) is None


def test_percentile_single_sample_returns_that_sample():
    assert perf.bench_h1.percentile([7.5], 99) == 7.5


def test_percentile_is_nearest_rank_not_interpolated():
    """1..10 的 P95 必须是 10（某个真实样本），不是插值出来的 9.55。"""
    values = [float(i) for i in range(1, 11)]
    assert perf.bench_h1.percentile(values, 95) == 10.0
    assert perf.bench_h1.percentile(values, 50) == 5.0
    assert perf.bench_h1.percentile(values, 99) == 10.0


def test_percentile_only_returns_observed_values():
    """每个返回值都必须能在样本里找到——nearest-rank 的定义性质。"""
    values = [3.0, 1.0, 4.0, 1.5, 9.0, 2.6]
    for pct in (1, 25, 50, 75, 95, 99):
        assert perf.bench_h1.percentile(values, pct) in values


def test_summarize_empty_has_zero_n_and_null_quantiles():
    """n 必须是 0、分位数必须是 None，两个都不能被写成 0.0。"""
    block = perf.bench_h1.summarize([])
    assert block["n"] == 0
    for key in ("p50_ms", "p95_ms", "p99_ms"):
        assert block[key] is None


# ---------------------------------------------------------------- 门 1：两代可加载
def test_generations_ok_passes_on_clean_input():
    ok, detail, _ = perf.check_generations_ok(_measure(), _measure_baseline())
    assert ok, detail


def test_generations_ok_fails_on_legacy_schema():
    """ark-rag-1 那一代当前链路加载不了，必须被拦住并点名 schema。"""
    bad = _measure(graph_schema="ark-rag-1")
    ok, detail, data = perf.check_generations_ok(bad, _measure_baseline())
    assert not ok
    assert "ark-rag-1" in detail
    assert data["problems"]


def test_generations_ok_fails_when_same_generation_compared():
    """自己跟自己比必须红：那是没有对比，不是「没有退步」。"""
    same = _measure_baseline(generation_id=_measure()["generation_id"])
    ok, detail, _ = perf.check_generations_ok(_measure(), same)
    assert not ok
    assert "同一个世代" in detail


def test_generations_ok_fails_on_empty_artifact():
    """某件产物 0 字节 = 缺产物，不许当「很小」放过。"""
    bad = _measure(artifact_bytes={"graph": 0, "adjacency": 10, "assertions": 10,
                                   "index": 10, "chunks": 10})
    ok, detail, _ = perf.check_generations_ok(bad, _measure_baseline())
    assert not ok
    assert "graph" in detail


def test_generations_ok_fails_when_status_not_ready():
    ok, detail, _ = perf.check_generations_ok(
        _measure(generation_status="STAGING"), _measure_baseline()
    )
    assert not ok
    assert "STAGING" in detail


# ---------------------------------------------------------------- 门 2：索引字节与成因
def test_index_size_delta_passes_on_clean_input():
    ok, detail, data = perf.check_index_size_delta(
        _probe(), _probe_baseline(), _measure(), _measure_baseline()
    )
    assert ok, detail
    assert data["shrink_pct"] == pytest.approx(16.16, abs=0.01)


def test_index_size_delta_fails_when_candidate_index_grew():
    """方向不符必须红：门的全部意义就是量出「候选更小」。"""
    big = _probe(index_bytes=130_000_000, opened_json_bytes=130_000_000)
    ok, detail, _ = perf.check_index_size_delta(
        big, _probe_baseline(), _measure(), _measure_baseline()
    )
    assert not ok
    assert "方向不符" in detail


def test_index_size_delta_fails_when_chunk_count_dropped():
    """文件变小可能只是「内容变少」——文档数掉了就不许算通过。"""
    fewer = _probe(doc_count=13000, meta_chunk_count=13000)
    mc = _measure(index_chunk_count=13000)
    ok, detail, _ = perf.check_index_size_delta(
        fewer, _probe_baseline(), mc, _measure_baseline()
    )
    assert not ok
    assert "内容变少" in detail


def test_index_size_delta_fails_when_term_count_not_reduced():
    """t1→t2 的成因是「词条更少更长」；词条数没减就说明成因解释不成立。"""
    same_terms = _probe(term_count=_probe_baseline()["term_count"])
    ok, detail, _ = perf.check_index_size_delta(
        same_terms, _probe_baseline(), _measure(), _measure_baseline()
    )
    assert not ok
    assert "t1→t2" in detail


def test_index_size_delta_fails_when_bytes_differ_from_recorded_identity():
    """字节数与已记录的产物身份不符 = 产物被换过，必须重新解释。"""
    swapped = _probe(index_bytes=106_685_069, opened_json_bytes=106_685_069)
    ok, detail, _ = perf.check_index_size_delta(
        swapped, _probe_baseline(), _measure(), _measure_baseline()
    )
    assert not ok
    assert "产物身份变化" in detail


# ---------------------------------------------------------------- 门 3：冷加载路径
def test_cold_load_path_passes_when_only_lexical_opened():
    ok, detail, _ = perf.check_cold_load_path(_probe(), _probe_baseline())
    assert ok, detail


def test_cold_load_path_fails_when_graph_json_opened():
    """这是 DoD「不再依赖每次启动解析 127 MB 巨型 JSON」的直接判据。"""
    bad = _probe(
        opened_json_paths=["data/arkrag/generations/g-x/graph/graph.json"],
        opened_json_count=2,
        opened_json_bytes=106_685_068 + 31_613_199,
        opened_graph=True,
    )
    ok, detail, _ = perf.check_cold_load_path(bad, _probe_baseline())
    assert not ok
    assert "graph.json" in detail


def test_cold_load_path_fails_when_bytes_do_not_match_lexical():
    """打开的文件字节数对不上 lexical.json：读的确实是另一份东西。"""
    bad = _probe(opened_json_bytes=1000)
    ok, detail, _ = perf.check_cold_load_path(bad, _probe_baseline())
    assert not ok
    assert "1,000" in detail or "1000" in detail


# ---------------------------------------------------------------- 门 4：索引绑定
def test_index_binding_passes_on_clean_input():
    ok, detail, _ = perf.check_index_binding(_probe(), _probe_baseline())
    assert ok, detail


def test_index_binding_fails_when_recomputed_digest_differs():
    """盘上这份索引不是 Manifest 记录的那份：回读摘要必须能发现。"""
    bad = _probe(recomputed_probe_digest="b" * 64)
    ok, detail, _ = perf.check_index_binding(bad, _probe_baseline())
    assert not ok
    assert "索引 meta" in detail


def test_index_binding_fails_when_manifest_digest_differs():
    bad = _probe(manifest_probe_digest="c" * 64)
    ok, detail, _ = perf.check_index_binding(bad, _probe_baseline())
    assert not ok
    assert "Manifest" in detail


def test_index_binding_fails_when_probe_digest_missing():
    """旧索引形态没有 probe_digest 就无法绑定，必须红而不是默默放过。"""
    bad = _probe(meta_probe_digest="")
    ok, detail, _ = perf.check_index_binding(bad, _probe_baseline())
    assert not ok
    assert "无法绑定" in detail


# ---------------------------------------------------------------- 门 5：峰值 RSS
def test_rss_ceiling_passes_on_clean_input():
    ok, detail, _ = perf.check_rss_ceiling(_measure(), _measure_baseline())
    assert ok, detail


def test_rss_ceiling_fails_above_absolute_ceiling():
    """把 127 MB JSON 又整份解析一遍那种量级事故必须红。"""
    heavy = _measure(memory={
        "at_start": {"current_mib": 26.0, "peak_mib": 26.0},
        "after_load": {"current_mib": 5000.0, "peak_mib": 5000.0},
        "after_pipeline": {"current_mib": 5000.0, "peak_mib": 5000.0},
        "at_end": {"current_mib": 5000.0, "peak_mib": 5000.0},
    })
    ok, detail, _ = perf.check_rss_ceiling(heavy, _measure_baseline())
    assert not ok
    assert "上界" in detail


def test_rss_ceiling_fails_above_ratio():
    """绝对上界没破，但比上一代高 1.5 倍以上：加载路径换过。"""
    bloated = _measure(memory={
        "at_start": {"current_mib": 26.0, "peak_mib": 26.0},
        "after_load": {"current_mib": 2600.0, "peak_mib": 2600.0},
        "after_pipeline": {"current_mib": 2600.0, "peak_mib": 2600.0},
        "at_end": {"current_mib": 2600.0, "peak_mib": 2600.0},
    })
    ok, detail, _ = perf.check_rss_ceiling(bloated, _measure_baseline())
    assert not ok
    assert "1.5" in detail


def test_rss_ceiling_fails_when_peak_unavailable():
    """拿不到峰值就不许判通过——那等于把「没测到」当「没超」。"""
    blind = _measure(memory={
        "at_start": {"current_mib": None, "peak_mib": None},
        "after_load": {"current_mib": None, "peak_mib": None},
        "after_pipeline": {"current_mib": None, "peak_mib": None},
        "at_end": {"current_mib": None, "peak_mib": None},
    })
    ok, detail, _ = perf.check_rss_ceiling(blind, _measure_baseline())
    assert not ok
    assert "不可得" in detail


# ---------------------------------------------------------------- 门 6：字段完整性
def test_report_fields_passes_on_clean_input():
    ok, detail, _ = perf.check_report_fields(_measure(), _measure_baseline())
    assert ok, detail


def test_report_fields_fails_when_a_top_level_field_missing():
    bad = _measure()
    bad.pop("load_total_ms")
    ok, detail, data = perf.check_report_fields(bad, _measure_baseline())
    assert not ok
    assert "load_total_ms" in detail
    assert "load_total_ms" in "".join(data["problems"])


def test_report_fields_fails_when_quantile_is_none():
    """空样本的 `None` 不许当成 0.0 —— 那会让一代看起来极快。"""
    bad = _measure()
    bad["warm"] = dict(bad["warm"], p95_ms=None)
    ok, detail, _ = perf.check_report_fields(bad, _measure_baseline())
    assert not ok
    assert "warm.p95_ms" in detail


def test_report_fields_fails_when_sample_count_differs_from_question_count():
    bad = _measure()
    bad["cold"] = dict(bad["cold"], n=2)
    ok, detail, _ = perf.check_report_fields(bad, _measure_baseline())
    assert not ok
    assert "题数" in detail


def test_report_fields_fails_when_llm_tokens_invented():
    """本机无 LLM 端点，有值就是编的。"""
    bad = _measure()
    bad["tokens"] = dict(bad["tokens"], llm_prompt_tokens=1234)
    ok, detail, _ = perf.check_report_fields(bad, _measure_baseline())
    assert not ok
    assert "llm_prompt_tokens" in detail


# ---------------------------------------------------------------- 门 7：P95 兜底
def test_p95_backstop_passes_at_small_ratio():
    ok, detail, _ = perf.check_p95_backstop(_measure(), _measure_baseline())
    assert ok, detail


def test_p95_backstop_passes_at_moderate_slowdown():
    """1.4 倍属于本机不可归因的噪声区，不许因此变红（否则门会被顺手放宽）。"""
    slower = _measure()
    slower["warm"] = dict(slower["warm"], p95_ms=280.0)  # 200 * 1.4
    ok, detail, _ = perf.check_p95_backstop(slower, _measure_baseline())
    assert ok, detail


def test_p95_backstop_fails_at_double():
    """超过 2 倍必须红：这条门只负责抓「慢了十倍」那种灾难性回退。"""
    disaster = _measure()
    disaster["warm"] = dict(disaster["warm"], p95_ms=401.0)  # 200 * 2.005
    ok, detail, data = perf.check_p95_backstop(disaster, _measure_baseline())
    assert not ok
    assert "灾难性回退" in detail
    assert data["ratio"] > 2.0


def test_p95_backstop_fails_when_p95_missing():
    bad = _measure()
    bad["warm"] = dict(bad["warm"], p95_ms=None)
    ok, detail, _ = perf.check_p95_backstop(bad, _measure_baseline())
    assert not ok
    assert "无法判定" in detail


# ---------------------------------------------------------------- 申报项
def test_declared_items_are_never_enforced():
    """申报项必须全部 enforced=False，否则它会悄悄变成第八条硬门。"""
    decl = perf.declared_items(_measure(), _measure_baseline(), _probe(), _probe_baseline())
    assert decl
    assert all(item["enforced"] is False for item in decl)
    assert all(item["generation_id"] for item in decl)


def test_declared_items_cover_both_generations_equally():
    """两代申报项数必须相同，否则「对比」会缺一条腿。"""
    decl = perf.declared_items(_measure(), _measure_baseline(), _probe(), _probe_baseline())
    names_c = {i["name"] for i in decl if i["generation_id"] == _measure()["generation_id"]}
    names_b = {i["name"] for i in decl
               if i["generation_id"] == _measure_baseline()["generation_id"]}
    assert names_c == names_b
    assert len(names_c) == len(decl) // 2


def test_declared_items_carry_null_when_block_missing():
    """缺块时申报项写 None，不许填料。"""
    blind = _measure()
    blind.pop("cold")
    decl = perf.declared_items(blind, _measure_baseline(), _probe(), _probe_baseline())
    cold_p95 = [i for i in decl if i["name"] == "cold_p95_ms"
                and i["generation_id"] == blind["generation_id"]]
    assert cold_p95 and cold_p95[0]["value"] is None
