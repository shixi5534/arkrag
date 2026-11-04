"""P1-a 的回归测试：切分规则可复现、`gold-v1.json` 没被碰、坏规则必须能红。

为什么这些用例必须存在：

* 切分规则如果只写在文档里、实现散在一次性脚本里，改动就没有守门人。这里把
  「分层键、桶算法、比例」锁成常量与纯函数的契约，任何一处漂移都会红。
* `gold-v1.json` 是历史对照锚点，`gold_set_fixed` 门拿它的字节数当冻结值。一条
  「v1 那半逐字段等于原件」的用例，能在有人「顺手把 v2 里 v1 的 required 改一下」
  时立刻报出来。
* 反向实测要求：把 `split_by_stratum` 的 `math.ceil` 改成 `round` 或把分层键里的
  `should_abstain` 去掉，本文件必须变红（见 `docs/goldset-v2说明.md` 的实测记录）。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from arkrag import goldset  # noqa: E402


def _load_make_split():
    spec = importlib.util.spec_from_file_location(
        "make_split_under_test", ROOT / "scripts" / "make_split.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MS = _load_make_split()

GOLD_V1 = ROOT / "data" / "goldset" / "gold-v1.json"
GOLD_V2 = ROOT / "data" / "goldset" / "gold-v2.json"
SPLIT_V2 = ROOT / "data" / "goldset" / "split-v2.json"

#: `gold-v1.json` 的冻结指纹。它与 `scripts/k1_checks.py` 里那对常量必须一致——
#: 两边都改才叫「换了契约」，只改一边就是两套事实。
GOLD_V1_SHA256_16 = "76d66073ff87aab1"
GOLD_V1_BYTES = 284356


def test_split_seed_is_frozen():
    """切分种子是契约。改了它，库里所有 holdout 归属全部失效。"""
    assert MS.SPLIT_SEED == "arkrag-split-v1-20261104"
    assert MS.V1_RATIO == 0.30
    assert MS.NEW_RATIO == 0.50
    assert MS.NEW_SEED == 20261111
    assert MS.NEW_MINE_TOTAL == 640


def test_gold_v1_is_untouched():
    """v1 必须一个字节都没动。它是历史对照锚点，换了就没法做跨阶段对比。"""
    blob = GOLD_V1.read_bytes()
    assert hashlib.sha256(blob).hexdigest()[:16] == GOLD_V1_SHA256_16
    assert len(blob) == GOLD_V1_BYTES


def test_gold_v2_and_split_v2_exist_and_parse():
    gold = json.loads(GOLD_V2.read_text("utf-8"))
    split = json.loads(SPLIT_V2.read_text("utf-8"))
    assert gold["schema"] == "ark-goldset-2"
    assert split["schema"] == "ark-split-1"
    assert gold["total"] == 640
    assert len(gold["items"]) == 640
    assert split["split_seed"] == MS.SPLIT_SEED


def test_every_item_declares_split_and_origin():
    gold = json.loads(GOLD_V2.read_text("utf-8"))
    for it in gold["items"]:
        assert it["split"] in ("dev", "holdout"), it["question_id"]
        assert it["origin"] in ("v1", "new"), it["question_id"]


def test_quota_arithmetic_is_exact():
    """八类配额必须逐类严格成立，且等于「v1 配额 + 新增配额」。"""
    gold = json.loads(GOLD_V2.read_text("utf-8"))
    v1 = json.loads(GOLD_V1.read_text("utf-8"))
    for cat in goldset.CATEGORIES:
        want = v1["quotas"][cat] + MS.NEW_QUOTAS[cat]
        assert gold["quotas"][cat] == want, cat
        got = sum(1 for it in gold["items"] if it["category"] == cat)
        assert got == want, f"{cat} 实得 {got} ≠ {want}"


def test_v1_half_is_byte_identical_except_split_and_origin():
    """v2 里 origin=v1 的题，除 split / origin 外必须与 gold-v1 原件逐字段相等。

    这条最容易被人「顺手改一下 required」悄悄破坏——改 v2 里的 v1 题等于改了
    历史基线，而 `gold_set_fixed` 门只看 v1 文件的哈希，看不见 v2 内部的偏离。
    """
    v2 = json.loads(GOLD_V2.read_text("utf-8"))
    v1 = json.loads(GOLD_V1.read_text("utf-8"))
    src = {it["question_id"]: it for it in v1["items"]}
    seen = 0
    for it in v2["items"]:
        if it["origin"] != "v1":
            continue
        stripped = {k: v for k, v in it.items() if k not in ("split", "origin")}
        assert stripped == src[it["question_id"]], it["question_id"]
        seen += 1
    assert seen == 320


def test_new_half_does_not_overlap_v1():
    """新增批与 v1 不得共享问句、gold 块、gold 断言。

    共享其中任何一项都意味着「对 A 干净」这个前提已经不成立：他看过 v1 的逐题结果，
    chunk 排名、答案覆盖都见过。
    """
    v2 = json.loads(GOLD_V2.read_text("utf-8"))
    v1_items = [it for it in v2["items"] if it["origin"] == "v1"]
    new_items = [it for it in v2["items"] if it["origin"] == "new"]
    assert len(new_items) == 320

    v1_q = {it["question"] for it in v1_items}
    assert not ({it["question"] for it in new_items} & v1_q)

    v1_ch: set[str] = set()
    v1_as: set[str] = set()
    for it in v1_items:
        v1_ch |= set(it.get("gold_chunk_ids") or [])
        v1_as |= set(it.get("gold_assertion_ids") or [])
    new_ch: set[str] = set()
    new_as: set[str] = set()
    for it in new_items:
        new_ch |= set(it.get("gold_chunk_ids") or [])
        new_as |= set(it.get("gold_assertion_ids") or [])
    assert not (new_ch & v1_ch)
    assert not (new_as & v1_as)


def test_split_totals_match_the_file():
    """split-v2 里记的账必须与 gold-v2 实际的 split 字段一致。"""
    gold = json.loads(GOLD_V2.read_text("utf-8"))
    split = json.loads(SPLIT_V2.read_text("utf-8"))
    actual = {
        "holdout": sum(1 for it in gold["items"] if it["split"] == "holdout"),
        "dev": sum(1 for it in gold["items"] if it["split"] == "dev"),
    }
    assert split["totals"]["holdout"] == actual["holdout"]
    assert split["totals"]["dev"] == actual["dev"]
    assert actual["holdout"] + actual["dev"] == 640
    assert actual["holdout"] == 260
    assert actual["dev"] == 380


def test_holdout_ids_in_split_file_are_exactly_the_holdout_items():
    """名单与题上的 split 字段必须双向一致——两边各留一份是为了让人能直接读名单。"""
    gold = json.loads(GOLD_V2.read_text("utf-8"))
    split = json.loads(SPLIT_V2.read_text("utf-8"))
    from_file = set(split["v1"]["holdout"]) | set(split["new"]["holdout"])
    from_items = {it["question_id"] for it in gold["items"] if it["split"] == "holdout"}
    assert from_file == from_items


def test_split_is_deterministic():
    """同样的输入跑两次，归属必须逐条相同。切分靠哈希，不靠任何全局状态。"""
    v2 = json.loads(GOLD_V2.read_text("utf-8"))
    v1_items = [it for it in v2["items"] if it["origin"] == "v1"]
    stripped = [{k: v for k, v in it.items() if k not in ("split", "origin")} for it in v1_items]
    a, _ = MS.split_by_stratum(stripped, MS.V1_RATIO, "v1")
    b, _ = MS.split_by_stratum(stripped, MS.V1_RATIO, "v1")
    assert a == b
    assign_in_file = {it["question_id"]: it["split"] for it in v1_items}
    assert a == assign_in_file


def test_stratum_ignores_difficulty_and_uses_should_abstain():
    """分层键必须是 `(category, should_abstain)`。

    `difficulty` 在类内是常量（实测 single_chunk 全 easy、direct_relation 全 medium），
    拿它分层等于没分；`should_abstain` 才是唯一有混合的维度。这条用真实数据把
    两个事实都钉住，防止有人「优化」成分层键。
    """
    v1 = json.loads(GOLD_V1.read_text("utf-8"))
    by_cat: dict[str, set[str]] = {}
    for it in v1["items"]:
        by_cat.setdefault(it["category"], set()).add(it["difficulty"])
    constant_cats = [c for c, ds in by_cat.items() if len(ds) == 1]
    assert len(constant_cats) == len(goldset.CATEGORIES), (
        f"difficulty 不再是类内常量（{len(constant_cats)}/{len(goldset.CATEGORIES)} 类常量），"
        "分层键的选择理由需要重新论证"
    )

    strata = MS._strata(v1["items"])
    assert all(isinstance(k, tuple) and len(k) == 2 for k in strata)
    assert any(k[1] is True for k in strata), "should_abstain=True 的层不存在，分层键失效"


def test_ceil_keeps_small_strata_in_holdout():
    """小层也必须至少有 1 条进 holdout。

    用 `round` 时 `n=2, ratio=0.30` 会得到 0 条 —— 那么 holdout 就缺了这一类，
    而「每类都要有代表」正是分层切分的唯一目的。
    """
    items = [
        {"question_id": f"q-tc-{i:06d}", "category": "time_conflict",
         "should_abstain": False, "difficulty": "hard"}
        for i in range(1, 3)
    ]
    assign, ledger = MS.split_by_stratum(items, 0.30, "unit")
    assert ledger[0]["holdout"] == 1
    assert list(assign.values()).count("holdout") == 1


def test_bucket_is_stable_and_depends_on_tag():
    """桶算法是契约的一部分：改拼串方式等于重新切分。"""
    a = MS.bucket("v1", "q-sc-000001")
    b = MS.bucket("v1", "q-sc-000001")
    c = MS.bucket("new", "q-sc-000001")
    assert a == b
    assert a != c
    assert len(a) == 64


def test_self_check_catches_broken_split():
    """自检本身必须能红：喂一份坏掉的 gold-v2，必须报出问题而不是默默通过。"""
    gold = json.loads(GOLD_V2.read_text("utf-8"))
    split = json.loads(SPLIT_V2.read_text("utf-8"))
    assert MS.self_check(gold, split) == []

    broken = json.loads(json.dumps(gold))
    broken["items"] = broken["items"][:-1]
    assert MS.self_check(broken, split) != []


def test_self_check_catches_tampered_v1_item():
    """把 v2 里某条 v1 题的 required 改一个字，自检必须报出来。"""
    gold = json.loads(GOLD_V2.read_text("utf-8"))
    split = json.loads(SPLIT_V2.read_text("utf-8"))
    broken = json.loads(json.dumps(gold))
    for it in broken["items"]:
        if it["origin"] == "v1" and it["required_citations"]:
            it["required_citations"][0]["evidence"] = "被改过一个字的证据句"
            break
    else:
        pytest.fail("找不到可改的 v1 题，用例失效")
    problems = MS.self_check(broken, split)
    assert any("被改动了" in p for p in problems), problems


def test_self_check_catches_overlap_between_halves():
    """新增批若与 v1 共享 gold 块，自检必须报出来。"""
    gold = json.loads(GOLD_V2.read_text("utf-8"))
    split = json.loads(SPLIT_V2.read_text("utf-8"))
    broken = json.loads(json.dumps(gold))
    v1_chunk = None
    for it in broken["items"]:
        if it["origin"] == "v1" and it.get("gold_chunk_ids"):
            v1_chunk = it["gold_chunk_ids"][0]
            break
    assert v1_chunk
    for it in broken["items"]:
        if it["origin"] == "new":
            it["gold_chunk_ids"] = [v1_chunk] + list(it.get("gold_chunk_ids") or [])
            break
    problems = MS.self_check(broken, split)
    assert any("共享" in p for p in problems), problems


def test_reports_use_relative_paths():
    """入库 JSON 里不许出现绝对机台路径（`tests/test_portable_paths.py` 那条门的延伸）。

    `make_split.py` 写 `origin.v1.path` 与 `clean_note` 时走 `config.rel_to_root()`，
    这条用例把结果钉住：换机器不会因为路径不同而红。
    """
    gold = json.loads(GOLD_V2.read_text("utf-8"))
    split = json.loads(SPLIT_V2.read_text("utf-8"))
    blob = json.dumps(gold, ensure_ascii=False) + json.dumps(split, ensure_ascii=False)
    assert ":\\" not in blob and ":/" not in blob.replace("http://", ""), "出现绝对路径"
    assert gold["origin"]["v1"]["path"] == "data/goldset/gold-v1.json"
    assert split["v1"]["source"] == "data/goldset/gold-v1.json"
