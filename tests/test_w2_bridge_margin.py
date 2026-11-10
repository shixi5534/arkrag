"""W2 门：`bridge.find_bridge(policy=...)` + `bridge_calib` 的概率化决策。

## 这个文件守的是什么

W2 的**唯一**承诺是"多答一些 multihop"；它同时必须做到**默认档一位不变**。
两条都靠测试钉，不靠注释。

1. **默认路径逐位等价。** 本文件内嵌一份 `_legacy_find_bridge()`，是改造前
   （提交 `70f9b24`）`find_bridge` 的**冻结副本**。对一组合成场景逐个比对
   `find_bridge(...)`（不传 `policy`）与 `_legacy_find_bridge(...)` 的返回值。
   失效方式：新分支顺手改了 `unique_only` 的返回结构或 `status`，历史读数漂移。
   （为什么内嵌副本而不是 `git show`：测试不许依赖 git 状态，也不许依赖
   仓库里某个历史文件还在。）

2. **`ask_once` 的默认调用形状不变。** 用一个**记录参数的桩**替换
   `bridge.find_bridge`，断言默认档下桩收到的关键字集合恰好是
   `{"chunk_assertions", "display"}`——一个 `policy=` 都不许多。
   失效方式：`ask_once` 无条件传 `policy="unique_only"`，于是所有按旧签名
   写的桩/替身（`scripts/audit_round6.py` 就按位置+两关键字调用）在默认档上
   表现不同。

3. **`top1_margin` 不许"无脑 top-1"。** 三个方向都要钉：
   候选唯一 → 作答（与 `unique_only` 同路）；概率 > 2/3 → 作答且**选中 top-1
   本人**；概率 ≤ 2/3 / 桶里没样本 / 并列 → 仍然 `ambiguous`。

4. **校准模块的两段纯逻辑。** `fit(样本) -> 单调映射` 与
   `decide(映射, 候选) -> 答/不答`。单调性（含 PAVA 合并）、阈值 2/3、
   `p == 2/3` 的边界口径、`None`（没样本）与 `0.0`（概率为零）的区别、
   代价比被映射文件篡改时必须拒——逐条钉。

## 这个文件**不**证明什么

它**不做校准**，也**不评测分数**。校准要求跑 `load_generation()` 逐题落盘
`data/diag_bridge_margin.json`，那一步必须经 `scripts/run_guarded.py` 串行跑。
本文件只证明"给定样本能算出单调映射"和"给定映射能决策"这两段纯逻辑；
`p ≥ 2/3` 区域在真实数据上的实测精度（计划书 M0）**不在这里**。
"""
from __future__ import annotations

import copy
import json

import pytest

from arkrag import bridge as br
from arkrag import bridge_calib as calib
from arkrag import pipeline as pipeline_mod

Q_TWO = "已知「甲」通过一个中间实体与「乙」相连。这个中间实体是谁？"


# --------------------------------------------------------------- 夹具
def _unit(cid: str, text: str, assertions: list[dict] | None = None) -> dict:
    return {"chunk_id": cid, "text": text, "assertions": list(assertions or ()),
            "title": "", "chapter": "", "sources": [], "score": 1.0,
            "evidence_spans": []}


def _a(subject: str, predicate: str, obj: str, evidence: str) -> dict:
    return {"subject": subject, "predicate": predicate, "object": obj,
            "evidence": evidence}


def _legacy_find_bridge(question: str, units: list[dict], *, chunk_assertions=None,
                        display=None) -> dict:
    """改造前（提交 `70f9b24`）`find_bridge` 的**冻结副本**。

    一个字都不改地抄过来，作为"默认路径逐位等价"的对照物。**不要**顺手把它
    重构得和新实现一样——它存在的意义就是"没有经历过任何重构"。
    """
    import re as _re

    def anchors_of(q: str) -> list[str]:
        return [m.strip() for m in _re.findall(r"「([^」]{1,200})」", str(q or ""))
                if m.strip()]

    def is_bridge_question(q: str) -> bool:
        q = str(q or "")
        return "中间实体" in q and len(anchors_of(q)) == 2

    q = str(question or "")
    if not is_bridge_question(q):
        return {"status": "not_applicable", "x": "", "sentences": [], "citations": []}
    anc = anchors_of(q)

    def name_of(eid: str) -> str:
        raw = str(eid or "")
        return str(display(raw)) if display is not None else raw

    tri: dict[tuple[str, str, str], tuple[str, str, int, int]] = {}
    for unit in units or ():
        cid = str(unit.get("chunk_id") or "")
        body = str(unit.get("text") or "")
        if not cid or not body:
            continue
        if chunk_assertions is not None:
            records = [
                {"subject": name_of(r.get("subject_id")),
                 "predicate": str(r.get("predicate") or ""),
                 "object": name_of(r.get("object_id")),
                 "evidence": str(r.get("evidence_text") or "")}
                for r in (chunk_assertions(cid) or ())
            ]
        else:
            records = list(unit.get("assertions") or ())
        for a in records:
            s = str(a.get("subject") or "").strip()
            p = str(a.get("predicate") or "").strip()
            o = str(a.get("object") or "").strip()
            ev = str(a.get("evidence") or "")
            if not s or not p or not o or s == o:
                continue
            start = body.find(ev) if ev else -1
            if start < 0 or len(ev) < br.MIN_EVIDENCE_CHARS:
                continue
            tri.setdefault((s, p, o), (cid, ev, start, start + len(ev)))
    if not tri:
        return {"status": "no_common_neighbor", "x": "", "sentences": [], "citations": []}

    neigh: dict[str, list[tuple[str, str]]] = {}
    for s, p, o in tri:
        neigh.setdefault(s, []).append((o, p))
        neigh.setdefault(o, []).append((s, p))

    a_n = {o for o, _ in neigh.get(anc[0], ())}
    b_n = {o for o, _ in neigh.get(anc[1], ())}
    cand = (a_n & b_n) - set(anc)
    if not cand:
        return {"status": "no_common_neighbor", "x": "", "sentences": [], "citations": []}
    if len(cand) > 1:
        return {"status": "ambiguous", "x": "", "sentences": [], "citations": []}

    x = next(iter(cand))
    sentences: list[str] = []
    citations: list[dict] = []
    for anchor in anc:
        found = None
        for other, pred in neigh.get(x, ()):
            if other != anchor:
                continue
            key = (x, pred, anchor) if (x, pred, anchor) in tri else (anchor, pred, x)
            if key not in tri:
                continue
            cid, ev, start, end = tri[key]
            if not cid or not ev or start < 0:
                continue
            found = (f"{x}{pred}{anchor}", cid, ev, start, end)
            break
        if found is None:
            return {"status": "missing_edge", "x": x, "sentences": [], "citations": []}
        sentences.append(found[0])
        citations.append({"claim": found[0], "chunk_id": found[1],
                          "evidence": found[2], "start": found[3], "end": found[4]})

    return {"status": "ok", "x": x, "sentences": sentences, "citations": citations}


def _scenarios() -> list[tuple[str, list[dict]]]:
    """覆盖 `find_bridge` 每一个返回分支的合成场景。"""
    out: list[tuple[str, list[dict]]] = []

    # not_applicable
    out.append((Q_TWO, []))

    # no_common_neighbor：只有一条边
    out.append((Q_TWO, [_unit("c#1", "甲与丙同行。", [_a("甲", "同行", "丙", "甲与丙同行")])]))

    # ok：唯一候选
    text = "甲与丙同行。乙与丙交谈。"
    out.append((Q_TWO, [_unit("c#1", text, [
        _a("甲", "同行", "丙", "甲与丙同行"),
        _a("乙", "交谈", "丙", "乙与丙交谈"),
    ])]))

    # ambiguous：两个候选
    text = "甲与丙同行。乙与丙交谈。甲与丁同行。乙与丁交谈。"
    out.append((Q_TWO, [_unit("c#1", text, [
        _a("甲", "同行", "丙", "甲与丙同行"),
        _a("乙", "交谈", "丙", "乙与丙交谈"),
        _a("甲", "同行", "丁", "甲与丁同行"),
        _a("乙", "交谈", "丁", "乙与丁交谈"),
    ])]))

    # ambiguous：四个候选（`unique_only` 下连排序都不许做）
    parts, asserts = [], []
    for name in ("丙", "丁", "戊", "己"):
        parts.append(f"甲与{name}同行。乙与{name}交谈。")
        asserts.append(_a("甲", "同行", name, f"甲与{name}同行"))
        asserts.append(_a("乙", "交谈", name, f"乙与{name}交谈"))
    out.append((Q_TWO, [_unit("c#1", "".join(parts), asserts)]))

    # ambiguous：两个候选，且 support **不相等**（margin=2）。
    # 这条必须有：`test_unique_only_ignores_calibration` 靠它才能真的变红——
    # 若 `_scenarios()` 里的多候选场景全是 margin=0 的并列，那么"唯一才答"
    # 与"按概率决定"在这个测试里恰好给出同一个答案（都拒答），
    # 于是把 `unique_only` 分支删掉测试仍然是绿的。这是反向实测抓到的：
    # 第一版 `_scenarios()` 的四个多候选场景 margin 全是 0，突变 A 没变红。
    out.append((Q_TWO, _units_for(TEXT_TWO_CAND, ASSERTS_TWO_CAND)))

    # missing_edge：候选唯一，但只凑得出一条句
    out.append((Q_TWO, [_unit("c#1", "甲与丙同行。", [
        _a("甲", "同行", "丙", "甲与丙同行"),
        _a("乙", "交谈", "丙", "定位不到这句"),
    ])]))

    # 自环 + 锚点互连：候选必须为空
    out.append((Q_TWO, [_unit("c#1", "甲与自己交谈。甲与乙交谈。", [
        _a("甲", "交谈", "甲", "甲与自己交谈"),
        _a("甲", "交谈", "乙", "甲与乙交谈"),
    ])]))

    # 短证据（能被 find 找到但短于门槛）
    out.append((Q_TWO, [_unit("c#1", "甲丙。乙丙。", [
        _a("甲", "同行", "丙", "甲丙"),
        _a("乙", "交谈", "丙", "乙丙"),
    ])]))

    # 跨块：X—A 在一块、X—B 在另一块
    out.append((Q_TWO, [
        _unit("c#1", "开场。甲与丙同行。", [_a("甲", "同行", "丙", "甲与丙同行")]),
        _unit("c#2", "结尾。乙与丙交谈。", [_a("乙", "交谈", "丙", "乙与丙交谈")]),
    ]))

    # 触发面外
    out.append(("「甲」与「乙」处于哪种关系？", []))

    # 三个锚点（不触发）
    out.append(("「甲」「乙」「丙」之间的中间实体是谁？", []))

    return out


def _callback_scenario():
    """`chunk_assertions` + `display` 那一路（按全量断言求候选）。"""
    units = [_unit("c#1", "甲与丙同行。乙与丙交谈。", [])]
    full = {
        "c#1": [
            {"subject_id": "e-jia", "predicate": "同行", "object_id": "e-bing",
             "evidence_text": "甲与丙同行"},
            {"subject_id": "e-yi", "predicate": "交谈", "object_id": "e-bing",
             "evidence_text": "乙与丙交谈"},
        ]
    }
    names = {"e-jia": "甲", "e-yi": "乙", "e-bing": "丙"}
    return (units,
            lambda cid: full.get(cid, ()),
            lambda eid: names.get(eid, eid))


#: 供 `top1_margin` 用的双候选场景：丙 support 4 / 丁 support 2。
TEXT_TWO_CAND = ("甲与丙同行。乙与丙交谈。甲又见丙。乙又见丙。"
                 "甲与丁同行。乙与丁交谈。")
ASSERTS_TWO_CAND = [
    _a("甲", "同行", "丙", "甲与丙同行"),
    _a("乙", "交谈", "丙", "乙与丙交谈"),
    _a("甲", "又见", "丙", "甲又见丙"),
    _a("乙", "又见", "丙", "乙又见丙"),
    _a("甲", "同行", "丁", "甲与丁同行"),
    _a("乙", "交谈", "丁", "乙与丁交谈"),
]

#: 供"概率 ≤ 2/3"用的场景：两候选 support 相同（margin=0），但候选数不同。
TEXT_MARGIN_ZERO = "甲与丙同行。乙与丙交谈。甲与丁同行。乙与丁交谈。甲遇丙。"
ASSERTS_MARGIN_ZERO = [
    _a("甲", "同行", "丙", "甲与丙同行"),
    _a("乙", "交谈", "丙", "乙与丙交谈"),
    _a("甲", "遇", "丙", "甲遇丙"),
    _a("甲", "同行", "丁", "甲与丁同行"),
    _a("乙", "交谈", "丁", "乙与丁交谈"),
]

#: 供"完全并列"用的场景：丙、丁 的 (support, 谓词数, 度) 完全相同。
TEXT_TIE = "甲与丙同行。乙与丙交谈。甲与丁同行。乙与丁交谈。"
ASSERTS_TIE = [
    _a("甲", "同行", "丙", "甲与丙同行"),
    _a("乙", "交谈", "丙", "乙与丙交谈"),
    _a("甲", "同行", "丁", "甲与丁同行"),
    _a("乙", "交谈", "丁", "乙与丁交谈"),
]


def _units_for(text: str, asserts: list[dict]) -> list[dict]:
    return [_unit("c#1", text, asserts)]


def _sample(n_cands: int, support_top1: int, support_top2: int,
            n: int, correct: int) -> list[dict]:
    """造一个桶的样本：`n` 条，其中 `correct` 条 gold_is_top1。"""
    rows = []
    for i in range(n):
        rows.append({
            "n_cands": n_cands,
            "support_top1": support_top1,
            "support_top2": support_top2,
            "raw_margin": support_top1 - support_top2,
            "gold_is_top1": i < correct,
            "tie": False,
        })
    return rows


def _two_bucket_mapping() -> dict:
    """margin=2 概率高（8/10）、margin=0 概率低（1/7）的映射。"""
    samples = _sample(2, 4, 2, 10, 8) + _sample(2, 3, 3, 7, 1)
    return calib.fit(samples)


# --------------------------------------------------- 1. 默认路径逐位等价
class TestUniqueOnlyIsBitIdentical:
    """检查 1：`unique_only`（默认）= 与改造前逐位相同。"""

    def test_default_matches_the_frozen_legacy_copy(self):
        """不传 `policy` ⇒ 每个场景的返回值与冻结副本逐位相等。"""
        mismatch = []
        for question, units in _scenarios():
            got = br.find_bridge(question, units)
            want = _legacy_find_bridge(question, units)
            if got != want:
                mismatch.append((question, got, want))
        assert not mismatch, (
            f"默认路径与改造前不一致：{mismatch[:3]}。"
            "`unique_only` 是回滚开关，它一旦不等价，历史台账读数就全部失去可比性。")

    def test_explicit_unique_only_equals_omitted(self):
        """显式传 `policy="unique_only"` 必须与完全不传逐位相同。

        这两条路径真的会同时存在（`ask_once` 默认不传），所以它们必须等价。
        """
        for question, units in _scenarios():
            omitted = br.find_bridge(question, units)
            explicit = br.find_bridge(question, units, policy="unique_only")
            assert omitted == explicit, question

    def test_unique_only_ignores_calibration(self):
        """`unique_only` 下传校准表也**不许**改变任何结果。

        这条是有判别力的：`_scenarios()` 里含一个 `n_cands=2 & margin=2` 的场景，
        而 `_two_bucket_mapping()` 对该桶给出 p=0.8 > 2/3。所以如果
        `unique_only` 分支被删（多候选也走排序+概率），这个场景会从
        `ambiguous` 变成 `ok` ——本用例立刻红。
        """
        mapping = _two_bucket_mapping()
        statuses = []
        for question, units in _scenarios():
            base = br.find_bridge(question, units)
            with_calib = br.find_bridge(question, units, calibration=mapping)
            assert base == with_calib, question
            statuses.append(base["status"])
        # 前提断言：场景集里必须同时有 ambiguous 和 ok，否则上面那圈可能是空转
        assert "ambiguous" in statuses and "ok" in statuses, statuses

    def test_callback_path_matches_legacy(self):
        """`chunk_assertions` + `display` 那一路也逐位相同（它是最容易改坏的）。"""
        units, cb, disp = _callback_scenario()
        got = br.find_bridge(Q_TWO, units, chunk_assertions=cb, display=disp)
        want = _legacy_find_bridge(Q_TWO, units, chunk_assertions=cb, display=disp)
        assert got == want
        assert got["status"] == "ok" and got["x"] == "丙"

    def test_ambiguous_candidate_is_still_not_guessed(self):
        """候选不唯一时默认档仍然 `ambiguous`——这是红线 2 本身。"""
        out = br.find_bridge(Q_TWO, _units_for(TEXT_TWO_CAND, ASSERTS_TWO_CAND))
        assert out["status"] == "ambiguous"
        assert out["x"] == "" and out["sentences"] == [] and out["citations"] == []

    def test_unknown_policy_is_rejected_loudly(self):
        """拼错的策略名不许静默降级成默认档。"""
        with pytest.raises(ValueError):
            br.find_bridge(Q_TWO, _units_for(TEXT_TWO_CAND, ASSERTS_TWO_CAND),
                           policy="top1")

    def test_unknown_policy_is_rejected_even_off_trigger(self):
        """非触发题 + 拼错策略也必须炸，不许安静地返回 `not_applicable`。

        否则一个打错的开关只在 multihop 那 96 道题上才显形——本项目把这种形态
        记作"开关没接线"。
        """
        with pytest.raises(ValueError):
            br.find_bridge("「甲」与「乙」处于哪种关系？", [], policy="top1")


# ------------------------------------------- 2. `ask_once` 调用形状不变
class _StubRetriever:
    """只够 `ask_once` 跑通的最小 retriever：验证接线，不验证检索。"""

    def __init__(self) -> None:
        self.graph = {"entities": [], "alias_index": {}}
        self.query_context = None
        self.assertions_by_chunk = {}
        self.context_units: list[dict] = []
        self.bridge_units: list[dict] | None = None
        self.seen_context_kwargs: dict = {}

    def search(self, query_analysis, *, top_k=20):
        return {"hits": [{"chunk_id": "c#1", "score": 1.0, "cooc_boost": 0.0,
                          "sources": [{"channel": "bm25", "score": 1.0}]}],
                "stats": {"top_score": 1.0, "candidates": 1}}

    def expand(self, entity_ids, *, depth=1, query_analysis=None, seed_chunk_ids=None):
        return {"expansions": [], "stats": {}}

    def build_context(self, hits, expansions, *, token_budget=None,
                      query_analysis=None, **kwargs):
        self.seen_context_kwargs = dict(kwargs)
        out = {"units": list(self.context_units), "context_digest": "d" * 64,
               "stats": {"units": len(self.context_units), "tokens_used": 1}}
        if self.bridge_units is not None:
            out["bridge_candidates"] = list(self.bridge_units)
        return out

    def display_name(self, entity_id: str) -> str:
        return str(entity_id)


class _StubAnswerer:
    def answer(self, question, context, **kwargs):
        return {"answer": "", "abstained": True, "citations": []}


def _stub_pipeline() -> dict:
    return {"retriever": _StubRetriever(), "answerer": _StubAnswerer(),
            "top_k": 20, "graph_depth": 1, "vector_union": None}


def _install_assertions(stub: _StubRetriever, cid: str, asserts: list[dict]) -> None:
    """把 `unit["assertions"]` 形式转成 `assertions_by_chunk` 的全量记录形式。

    **必须转**：`ask_once` 总是传 `chunk_assertions=…get`，于是 `find_bridge`
    走的是"上下文块全量断言"那一路，`unit["assertions"]` 根本不读。桩里只放
    `unit["assertions"]` 的话，桥接会看到空记录集、静默退回 `no_common_neighbor`
    ——这条正好是端到端用例最容易写成假绿的地方（它会让"桥接没接线"看起来像
    "桥接拒答"）。
    """
    stub.assertions_by_chunk = {
        cid: [{"subject_id": a["subject"], "predicate": a["predicate"],
               "object_id": a["object"], "evidence_text": a["evidence"]}
              for a in asserts]
    }


class TestAskOnceCallShapeIsUnchanged:
    """检查 2：默认档下 `ask_once` 交给 `find_bridge` 的参数**一个不多**。"""

    def _run(self, monkeypatch, options: dict) -> list[dict]:
        calls: list[dict] = []

        def spy(question, units, **kwargs):
            calls.append({"question": question, "units": list(units),
                          "kwargs": dict(kwargs)})
            return {"status": "no_common_neighbor", "x": "",
                    "sentences": [], "citations": []}

        monkeypatch.setattr(pipeline_mod.bridge, "find_bridge", spy)
        pipe = _stub_pipeline()
        pipe["retriever"].context_units = _units_for(TEXT_TWO_CAND,
                                                     ASSERTS_TWO_CAND)
        pipeline_mod.ask_once(pipe, Q_TWO, options=options)
        return calls

    def test_default_options_do_not_pass_policy(self, monkeypatch):
        """只开 `bridge` 时，桩收到恰好两个关键字：`chunk_assertions`、`display`。"""
        calls = self._run(monkeypatch, {"bridge": True})
        assert len(calls) == 1
        assert set(calls[0]["kwargs"]) == {"chunk_assertions", "display"}, (
            f"默认档多传了 {sorted(set(calls[0]['kwargs']) - {'chunk_assertions', 'display'})}。"
            "`scripts/audit_round6.py` 按位置+两关键字调用 `find_bridge`，"
            "多一个关键字就让那条通路在默认档上换形状。")
        assert calls[0]["question"] == Q_TWO
        assert [u["chunk_id"] for u in calls[0]["units"]] == ["c#1"]

    def test_explicit_unique_only_is_the_same_shape(self, monkeypatch):
        """显式写 `bridge_policy="unique_only"` 也不许多传 `policy=`。"""
        calls = self._run(monkeypatch, {"bridge": True,
                                        "bridge_policy": "unique_only"})
        assert set(calls[0]["kwargs"]) == {"chunk_assertions", "display"}

    def test_top1_margin_passes_policy_and_calibration(self, monkeypatch):
        """非默认策略才拼关键字，且校准表**显式**从 options 传进来。"""
        mapping = _two_bucket_mapping()
        calls = self._run(monkeypatch, {"bridge": True,
                                        "bridge_policy": "top1_margin",
                                        "bridge_calibration": mapping})
        assert calls[0]["kwargs"]["policy"] == "top1_margin"
        assert calls[0]["kwargs"]["calibration"] is mapping

    def test_top1_margin_without_calibration_passes_none(self, monkeypatch):
        """没给校准表时传 `None`，由 `find_bridge` 自己回退拒答（不是空映射）。"""
        calls = self._run(monkeypatch, {"bridge": True,
                                        "bridge_policy": "top1_margin"})
        assert calls[0]["kwargs"]["policy"] == "top1_margin"
        assert calls[0]["kwargs"]["calibration"] is None

    def test_bridge_candidates_pool_wins_over_capped_units(self, monkeypatch):
        """W1 的 cap 不许砍小桥接候选池：桥接必须读 `bridge_candidates`。

        `build_context(max_direct_units=N)` 时 `units` 只有 N 条，而
        `bridge_candidates` 是 cap 之前的完整接地集合。传错那个，cap 就会
        顺手砍掉桥接的候选——计划书 §W1 明写这是"必须二选一并写清"的接口事实。
        """
        calls: list[dict] = []

        def spy(question, units, **kwargs):
            calls.append({"units": list(units)})
            return {"status": "no_common_neighbor", "x": "",
                    "sentences": [], "citations": []}

        monkeypatch.setattr(pipeline_mod.bridge, "find_bridge", spy)
        pipe = _stub_pipeline()
        ret = pipe["retriever"]
        ret.context_units = [_unit("c#capped", "随便一段正文。", [])]
        ret.bridge_units = _units_for(TEXT_TWO_CAND, ASSERTS_TWO_CAND)
        pipeline_mod.ask_once(pipe, Q_TWO, options={"bridge": True,
                                                    "max_direct_units": 1})
        assert [u["chunk_id"] for u in calls[0]["units"]] == ["c#1"], (
            "桥接拿到了 cap 之后的 `units`：cap 会砍小桥接候选池")


# --------------------------------------------------- 3. `top1_margin` 行为
class TestTop1Margin:
    """检查 3：概率 > 2/3 才答，且答的必须是 top-1 本人。"""

    def test_single_candidate_still_answers(self):
        """候选唯一时两条策略同路：直接作答，不看概率。"""
        text = "甲与丙同行。乙与丙交谈。"
        units = [_unit("c#1", text, [
            _a("甲", "同行", "丙", "甲与丙同行"),
            _a("乙", "交谈", "丙", "乙与丙交谈"),
        ])]
        out = br.find_bridge(Q_TWO, units, policy="top1_margin")
        assert out["status"] == "ok" and out["x"] == "丙"

    def test_high_probability_picks_the_true_top1(self):
        """margin=2 的桶校准到 0.8 > 2/3 ⇒ 作答，选中 support 更高的丙。"""
        mapping = _two_bucket_mapping()
        units = _units_for(TEXT_TWO_CAND, ASSERTS_TWO_CAND)
        out = br.find_bridge(Q_TWO, units, policy="top1_margin",
                             calibration=mapping)
        assert out["status"] == "ok", (
            f"校准概率 0.8 > 2/3 时必须作答，实际 {out['status']}")
        assert out["x"] == "丙", f"选的不是 top-1，而是 {out['x']!r}"
        assert len(out["sentences"]) == 2 and len(out["citations"]) == 2

    def test_low_probability_still_abstains(self):
        """margin=0 的桶校准到 1/7 < 2/3 ⇒ 仍然 `ambiguous`。"""
        mapping = _two_bucket_mapping()
        units = _units_for(TEXT_MARGIN_ZERO, ASSERTS_MARGIN_ZERO)
        tri = br._grounded_triples(units)
        neigh = br._neighbourhood(tri)
        anc = br.anchors_of(Q_TWO)
        ranked = br.rank_candidates(tri, neigh, anc,
                                    br._common_neighbours(neigh, anc))
        assert len(ranked) == 2
        assert ranked[0]["support_sum"] == ranked[1]["support_sum"], (
            "前提：这个场景的两个候选 support 相同（margin=0）")
        p = calib.probability(mapping, calib.features_of_ranked(ranked))
        assert p is not None and p <= 2 / 3, f"前提：该桶概率应 ≤ 2/3，实际 {p}"

        out = br.find_bridge(Q_TWO, units, policy="top1_margin",
                             calibration=mapping)
        assert out["status"] == "ambiguous", (
            f"概率 {p} ≤ 2/3 时不许作答，实际 {out['status']} x={out['x']!r}")
        assert out["x"] == ""

    def test_no_calibration_means_no_answer(self):
        """没有校准表就没有概率 ⇒ 拒答（不许"先猜着，反正没校准"）。"""
        units = _units_for(TEXT_TWO_CAND, ASSERTS_TWO_CAND)
        out = br.find_bridge(Q_TWO, units, policy="top1_margin")
        assert out["status"] == "ambiguous" and out["x"] == ""

    def test_missing_bucket_means_no_answer(self):
        """桶里没样本（样本不足）⇒ 拒答，不许把"没见过"当成"概率为零"或"平均"。"""
        mapping = calib.fit(_sample(3, 9, 1, 10, 9))   # 只有 n_cands=3 的桶
        units = _units_for(TEXT_TWO_CAND, ASSERTS_TWO_CAND)  # n_cands=2
        out = br.find_bridge(Q_TWO, units, policy="top1_margin",
                             calibration=mapping)
        assert out["status"] == "ambiguous"

    def test_tie_abstains_even_with_probability_one(self):
        """完全并列时即使映射说 p=1.0 也不许答——名字字典序不是 top-1 的依据。

        映射是**手写的**：只要还有任何一条路径能把并列答出去，这条就会红。
        """
        mapping = {
            "schema": calib.SCHEMA,
            "feature_keys": [calib.MONOTONE_IN],
            "monotone_in": calib.MONOTONE_IN,
            "cost": {"answer": calib.ANSWER_COST, "refuse": calib.REFUSE_COST},
            "threshold": calib.threshold(),
            "n_samples": 10,
            "n_excluded_tie": 0,
            "buckets": [{"key": {calib.MONOTONE_IN: 0}, "n": 10,
                         "correct": 10, "p": 1.0, "pool": 0}],
        }
        units = _units_for(TEXT_TIE, ASSERTS_TIE)
        out = br.find_bridge(Q_TWO, units, policy="top1_margin",
                             calibration=mapping)
        # 前提：这个场景确实是完全并列（否则这条测的是别的东西）
        tri = br._grounded_triples(units)
        neigh = br._neighbourhood(tri)
        anc = br.anchors_of(Q_TWO)
        ranked = br.rank_candidates(tri, neigh, anc,
                                    br._common_neighbours(neigh, anc))
        assert len(ranked) == 2
        assert calib.tie_of_top(ranked) is True
        assert out["status"] == "ambiguous" and out["x"] == ""

    def test_pick_is_bound_to_ranked_top1_only(self, monkeypatch):
        """`_decide_top1_x` 只认 `ranked[0]`：判定层说"选丁"时必须拒答。

        这条用**污染判定层**的方式钉住"不重试次优候选"：如果将来有人加一条
        "top-1 不行就试 top-2"的回退，这个桩会返回丁，函数就必须把它挡下来。
        """
        mapping = _two_bucket_mapping()
        units = _units_for(TEXT_TWO_CAND, ASSERTS_TWO_CAND)
        tri = br._grounded_triples(units)
        neigh = br._neighbourhood(tri)
        anc = br.anchors_of(Q_TWO)
        cand = br._common_neighbours(neigh, anc)

        monkeypatch.setattr(
            br.bridge_calib, "decide_top1",
            lambda m, ranked: {"answer": True, "x": "丁", "p": 0.99,
                               "gain": 1.0, "reason": "calibrated_answer",
                               "features": {}})
        assert br._decide_top1_x(tri, neigh, anc, cand, mapping) == "", (
            "判定层指向了非 top-1 的候选，`_decide_top1_x` 却接受了——"
            "这就是一条未登记的重试策略")

        monkeypatch.setattr(
            br.bridge_calib, "decide_top1",
            lambda m, ranked: {"answer": True, "x": "丙", "p": 0.99,
                               "gain": 1.0, "reason": "calibrated_answer",
                               "features": {}})
        assert br._decide_top1_x(tri, neigh, anc, cand, mapping) == "丙"

    def test_no_answer_without_calibration_at_the_pick_layer(self):
        """`calibration=None` 时判定层不被调用就走拒答（不给"默认表"的机会）。"""
        units = _units_for(TEXT_TWO_CAND, ASSERTS_TWO_CAND)
        tri = br._grounded_triples(units)
        neigh = br._neighbourhood(tri)
        anc = br.anchors_of(Q_TWO)
        cand = br._common_neighbours(neigh, anc)
        assert br._decide_top1_x(tri, neigh, anc, cand, None) == ""

    def test_pipeline_end_to_end_answers_with_the_top1(self, monkeypatch):
        """`ask_once` 真的把 `top1_margin` 的答案接进答案串（不是死开关）。"""
        mapping = _two_bucket_mapping()
        pipe = _stub_pipeline()
        ret = pipe["retriever"]
        ret.context_units = _units_for(TEXT_TWO_CAND, ASSERTS_TWO_CAND)
        _install_assertions(ret, "c#1", ASSERTS_TWO_CAND)
        res = pipeline_mod.ask_once(
            pipe, Q_TWO,
            options={"bridge": True, "bridge_policy": "top1_margin",
                     "bridge_calibration": mapping})
        assert res["answer"]["answer"].startswith("丙。"), (
            f"桥接答案没接进答案串：{res['answer']['answer']!r}"
            "（若这里是空串，先确认桩把断言放进了 `assertions_by_chunk`，"
            "而不是只放进 `unit['assertions']`）")

    def test_pipeline_end_to_end_abstains_at_low_probability(self):
        """同一管道，概率 ≤ 2/3 时答案串不许出现候选名（不猜）。"""
        mapping = _two_bucket_mapping()
        pipe = _stub_pipeline()
        ret = pipe["retriever"]
        ret.context_units = _units_for(TEXT_MARGIN_ZERO, ASSERTS_MARGIN_ZERO)
        _install_assertions(ret, "c#1", ASSERTS_MARGIN_ZERO)
        res = pipeline_mod.ask_once(
            pipe, Q_TWO,
            options={"bridge": True, "bridge_policy": "top1_margin",
                     "bridge_calibration": mapping})
        assert "丙" not in res["answer"]["answer"]
        assert "丁" not in res["answer"]["answer"]


# --------------------------------------------------- 4. 校准模块（纯逻辑）
class TestCalibrationFit:
    """检查 4a：`fit()` 产出的映射必须单调，且并列/坏样本被正确处置。"""

    def test_mapping_is_monotone_in_margin(self):
        """同一层内，margin 越大 `p` 不得越小（这是计划书的单调校准要求）。"""
        samples = (_sample(2, 1, 1, 6, 1)          # margin 0
                   + _sample(2, 2, 1, 6, 3)        # margin 1
                   + _sample(2, 3, 1, 6, 5)        # margin 2
                   + _sample(2, 4, 1, 6, 6))       # margin 3
        mapping = calib.fit(samples)
        by_margin = {}
        for bucket in mapping["buckets"]:
            by_margin[int(bucket["key"][calib.MONOTONE_IN])] = float(bucket["p"])
        ordered = [by_margin[m] for m in sorted(by_margin)]
        assert ordered == sorted(ordered), (
            f"映射不单调：margin→p = {by_margin}。"
            "单调性是计划书 §W2.1 对校准的硬要求，违反它等于允许"
            "'分数更高反而更不可信'。")
        assert ordered[-1] > ordered[0], "构造的样本本该单调递增，映射却是平的"

    def test_violating_buckets_get_pooled(self):
        """margin 增大而经验频率下降时，PAVA 必须把相邻桶拉平（不是丢弃顺序）。"""
        samples = _sample(2, 1, 1, 10, 9) + _sample(2, 2, 1, 10, 1)
        mapping = calib.fit(samples)
        ps = [float(b["p"]) for b in mapping["buckets"]]
        pools = [int(b["pool"]) for b in mapping["buckets"]]
        assert ps[0] == ps[1], f"违反单调的两个桶没被拉平：{ps}"
        assert pools[0] == pools[1], "被拉平的桶必须共享同一个 pool 编号"
        # 拉平后的 p 是合并频率 10/20，不是任何一侧的 9/10
        assert abs(ps[0] - 0.5) < 1e-12, ps

    def test_n_cands_can_be_a_separate_stratum(self):
        """把 `n_cands` 分进桶键时，单调性只在同层内要求（跨层不拉平）。"""
        samples = (_sample(2, 1, 1, 10, 9)      # n=2, margin 0, 高
                   + _sample(3, 1, 1, 10, 1))    # n=3, margin 0, 低
        flat = calib.fit(samples)
        assert abs(float(flat["buckets"][0]["p"]) - 0.5) < 1e-12
        layered = calib.fit(samples, keys=("n_cands", calib.MONOTONE_IN))
        by_key = {(int(b["key"]["n_cands"]), int(b["key"][calib.MONOTONE_IN])):
                  float(b["p"]) for b in layered["buckets"]}
        assert by_key[(2, 0)] == 0.9 and by_key[(3, 0)] == 0.1, (
            f"分层后不该被拉平：{by_key}")

    def test_tie_samples_are_excluded_but_counted(self):
        """并列样本不参与拟合（它永远走不到概率判定），但必须计数落盘。"""
        samples = _sample(2, 4, 2, 10, 8)
        samples += [dict(s, tie=True, gold_is_top1=False) for s in samples[:5]]
        mapping = calib.fit(samples)
        assert mapping["n_excluded_tie"] == 5
        assert mapping["n_samples"] == 10
        assert abs(float(mapping["buckets"][0]["p"]) - 0.8) < 1e-12, (
            "并列样本被算进了频率：会把校准偏向'并列题'")

    def test_inconsistent_margin_is_rejected(self):
        """`raw_margin` 与两个 support 相减不一致的样本不许进校准。"""
        bad = {"n_cands": 2, "support_top1": 4, "support_top2": 2,
               "raw_margin": 5, "gold_is_top1": True}
        with pytest.raises(ValueError):
            calib.fit([bad])

    def test_negative_margin_is_rejected(self):
        """候选按 support 降序排，top1 < top2 说明样本与生产排序不同源。"""
        bad = {"n_cands": 2, "support_top1": 1, "support_top2": 4,
               "raw_margin": 0, "gold_is_top1": True}
        with pytest.raises(ValueError):
            calib.fit([bad])

    def test_missing_label_is_rejected(self):
        bad = {"n_cands": 2, "support_top1": 4, "support_top2": 2}
        with pytest.raises(ValueError):
            calib.fit([bad])

    def test_unregistered_feature_is_rejected(self):
        with pytest.raises(ValueError):
            calib.fit(_sample(2, 4, 2, 4, 3), keys=("gold", calib.MONOTONE_IN))

    def test_keys_must_include_the_monotone_feature(self):
        """桶键里没有 margin 就没有可检验的单调方向，必须显式拒绝。"""
        with pytest.raises(ValueError):
            calib.fit(_sample(2, 4, 2, 4, 3), keys=("n_cands",))

    def test_empty_samples_are_rejected(self):
        with pytest.raises(ValueError):
            calib.fit([])


class TestCalibrationDecision:
    """检查 4b：`decide()` 的概率阈值就是代价比算出的 2/3。"""

    def test_threshold_comes_from_the_cost_ratio(self):
        assert calib.ANSWER_COST == 2 and calib.REFUSE_COST == 1
        assert abs(calib.threshold() - 2 / 3) < 1e-12

    def test_expected_gain_changes_sign_at_two_thirds(self):
        assert calib.expected_gain(1 / 3) < 0
        assert abs(calib.expected_gain(2 / 3)) < 1e-12
        assert calib.expected_gain(0.7) > 0

    def test_boundary_abstains(self):
        """`p == 2/3` 时期望增益恰为 0，按"拿不到正收益就不动"拒答。"""
        assert calib.should_answer(2 / 3) is False
        assert calib.should_answer(2 / 3 + 1e-9) is True
        assert calib.should_answer(2 / 3 - 1e-9) is False

    def test_decide_walks_the_three_directions(self):
        mapping = _two_bucket_mapping()
        high = [{"x": "丙", "support_sum": 4, "assert_count": 2, "degree": 2},
                {"x": "丁", "support_sum": 2, "assert_count": 1, "degree": 2}]
        low = [{"x": "丙", "support_sum": 3, "assert_count": 2, "degree": 2},
               {"x": "丁", "support_sum": 3, "assert_count": 1, "degree": 2}]
        top = calib.decide_top1(mapping, high)
        assert top["answer"] is True and top["x"] == "丙"
        assert top["reason"] == "calibrated_answer"
        assert top["p"] > 2 / 3
        bottom = calib.decide_top1(mapping, low)
        assert bottom["answer"] is False and bottom["x"] == ""
        assert bottom["reason"] == "calibrated_abstain"
        assert bottom["p"] <= 2 / 3

    def test_single_candidate_skips_the_probability_entirely(self):
        only = [{"x": "丙", "support_sum": 1, "assert_count": 1, "degree": 2}]
        out = calib.decide_top1(None, only)
        assert out["answer"] is True and out["reason"] == "single_candidate"
        assert out["p"] is None

    def test_no_candidate_answers_nothing(self):
        out = calib.decide_top1(_two_bucket_mapping(), [])
        assert out["answer"] is False and out["reason"] == "no_candidate"

    def test_uncalibrated_is_refused(self):
        two = [{"x": "丙", "support_sum": 4, "assert_count": 2, "degree": 2},
               {"x": "丁", "support_sum": 2, "assert_count": 1, "degree": 2}]
        out = calib.decide_top1(None, two)
        assert out["answer"] is False and out["reason"] == "uncalibrated"

    def test_features_cover_the_four_registered_quantities(self):
        two = [{"x": "丙", "support_sum": 4, "assert_count": 2, "degree": 2},
               {"x": "丁", "support_sum": 2, "assert_count": 1, "degree": 2}]
        feats = calib.features_of_ranked(two)
        assert set(feats) == set(calib.FEATURES)
        assert feats["n_cands"] == 2 and feats["support_top1"] == 4
        assert feats["support_top2"] == 2 and feats["raw_margin"] == 2


class TestMappingPersistence:
    """检查 4c：映射能从 JSON 存取，且代价比不许被文件改写。"""

    def test_roundtrip_preserves_every_field(self):
        mapping = _two_bucket_mapping()
        back = calib.from_json(calib.to_json(mapping))
        assert back == mapping

    def test_save_and_load_through_disk(self, tmp_path):
        mapping = _two_bucket_mapping()
        path = tmp_path / "bridge_margin_mapping.json"
        calib.save_mapping(path, mapping)
        assert path.exists()
        assert calib.load_mapping(path) == mapping

    def test_wrong_schema_is_rejected(self):
        payload = dict(_two_bucket_mapping())
        payload["schema"] = "something.else/9"
        with pytest.raises(ValueError):
            calib.from_json(json.dumps(payload))

    def test_tampered_cost_ratio_is_rejected(self):
        """代价比是预登记业务常量：映射文件不许把它改掉。"""
        payload = copy.deepcopy(_two_bucket_mapping())
        payload["cost"]["answer"] = 1
        with pytest.raises(ValueError):
            calib.from_json(json.dumps(payload))

    def test_missing_buckets_is_rejected(self):
        payload = copy.deepcopy(_two_bucket_mapping())
        payload["buckets"] = []
        with pytest.raises(ValueError):
            calib.from_json(json.dumps(payload))

    def test_probability_returns_none_not_zero_for_unseen_bucket(self):
        """`None`（没样本）与 `0.0`（概率为零）是两件事，不许混。"""
        mapping = _two_bucket_mapping()
        seen = calib.probability(mapping, {"n_cands": 2, "support_top1": 4,
                                           "support_top2": 2, "raw_margin": 2})
        unseen = calib.probability(mapping, {"n_cands": 7, "support_top1": 9,
                                             "support_top2": 3, "raw_margin": 6})
        assert seen == pytest.approx(0.8)
        assert unseen is None, f"没样本的桶返回了 {unseen!r}，必须返回 None"


class TestRankingKey:
    """检查 4d：排序键是计划书预登记的那四个，且并列有独立判定。"""

    def test_support_then_predicates_then_degree_then_name(self):
        units = _units_for(TEXT_TWO_CAND, ASSERTS_TWO_CAND)
        tri = br._grounded_triples(units)
        neigh = br._neighbourhood(tri)
        anc = br.anchors_of(Q_TWO)
        ranked = br.rank_candidates(tri, neigh, anc,
                                    br._common_neighbours(neigh, anc))
        assert [r["x"] for r in ranked] == ["丙", "丁"]
        assert ranked[0]["support_sum"] == 4 and ranked[1]["support_sum"] == 2

    def test_tie_detection_ignores_the_name(self):
        same = [{"x": "丙", "support_sum": 2, "assert_count": 2, "degree": 2},
                {"x": "丁", "support_sum": 2, "assert_count": 2, "degree": 2}]
        assert calib.tie_of_top(same) is True
        differ = [{"x": "丙", "support_sum": 3, "assert_count": 2, "degree": 2},
                  {"x": "丁", "support_sum": 2, "assert_count": 2, "degree": 2}]
        assert calib.tie_of_top(differ) is False

    def test_candidate_support_matches_the_grounded_assertions(self):
        """support 口径 = 接地断言条数（不是图边权重），写成断言免得被复算错。

        丙 连到两个锚点的接地断言有 4 条（`同行`、`交谈`、`又见`×2），
        不同谓词 3 个；`degree` 是它在本次接地邻域里的不同对端数（甲、乙）。
        """
        units = _units_for(TEXT_TWO_CAND, ASSERTS_TWO_CAND)
        tri = br._grounded_triples(units)
        neigh = br._neighbourhood(tri)
        sup = br.candidate_support(tri, neigh, br.anchors_of(Q_TWO), "丙")
        assert sup == {"assertions": 4, "predicates": 3, "degree": 2}


class TestSourceKeepsTheRecord:
    def test_module_records_the_scope_expiry_of_redline_23(self):
        """源码必须写清"红线 23 的作用域"与当前接地候选集上的新读数。

        删了这些数字，下一个改这个模块的人会以为 W2 是在无证据地推翻一条红线。
        """
        from pathlib import Path as _P
        src = (_P(__file__).resolve().parents[1] / "src" / "arkrag"
               / "bridge.py").read_text("utf-8")
        for token in ("12.57", "37.43", "43.9", "87.7"):
            assert token in src, f"`bridge.py` 里找不到 W2 的实测读数 {token}"
