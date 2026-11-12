"""L4「证据窗口装配」的门。

## 这个文件守的是什么

证据窗口是**唯一一条**能"不加预算就多装证据"的路径：块上限 900 字、median 32 轮，
而一个问句真正相关的证据通常只在 5 轮里。只把含断言的那几轮算进 `budget_tokens`，
同一批块、同一份索引、同一份断言，一块都不用重抽。

但它有三条**违反即失败**的约束，每一条都对应一个"看起来跑过了、其实是错的"形态：

1. **默认路径必须逐位不变。** 开关默认关；关着时 `result` 的 `units` 字段集、
   `stats` 的键集、`context_digest`、`tokens_used` 都必须与改造前完全相同。
   这条如果松了，台账里所有历史读数一起失效，而且没有任何报错会提示这件事。

2. **`unit["text"]` 必须是完整原文，一个字符都不许截。**
   `answer.verify_citations` 逐条查 `chunk.text[start:end] == evidence`，
   `_title_answer` / `_chapter_answer` 用 `unit["text"].splitlines()` 定位引文行。
   就地截断 `unit["text"]` 会让引用校验静默失败——表现为"答得短"，
   没人会去查（这是本项目记录在案的形态）。窗口只能**新增**字段。

3. **窗口必须真的省下 token 且真的多装单元。** 只加了字段、计价没变的实现
   会通过前两条门而毫无作用。所以还要钉住"装填数增加"。

## 为什么用例自己造块，不加载真语料

真语料要一整个世代（`load_generation()` 常驻 1.0–1.5 GB，且会读 288 MB 产物），
单测里跑不起也不该跑。这里用合成的 `turn_spans` + 断言偏移，语义与
`chunk.split_scene` 一致（`"\\n".join` 拼正文，`cursor = end + 1`），
所以窗口计算走的是**同一条代码路径**。
"""
from __future__ import annotations

import json

import pytest

from arkrag import adjacency as adjacency_mod
from arkrag import config
from arkrag import index as index_mod
from arkrag import query as query_mod
from arkrag import retrieve
from arkrag import vocab

#: 一轮 = 一行。10 轮 × 约 60 字 ≈ 600 字，够造出 3–5 倍压缩。
TURNS = [
    "旁白：清晨的营地刚刚苏醒，风从山口吹过来，卷起一层薄薄的尘土。",
    "凯尔希：博士，切尔诺伯格已经沦陷了，我们必须在天亮之前离开这里。",
    "阿米娅：可是罗德岛的人还在城里，我们不能丢下他们不管。",
    "旁白：远处传来低沉的轰鸣，地面开始轻微地震颤，碎石滚落下来。",
    "凯尔希：罗德岛不会为一个人的牺牲改变航向，这一点你应该清楚。",
    "阿米娅：我明白，可是让我什么都不做地离开，我做不到。",
    "旁白：营火在风里摇晃，把两个人的影子拉长又缩短，反复不定。",
    "凯尔希：把伤员先送上车，剩下的物资按顺序清点，别留下任何东西。",
    "阿米娅：我去通知大家，十分钟之后在这里集合，谁都不许掉队。",
    "旁白：天边透出一线灰白，营地里的人开始收拾行装，没有人说话。",
]

ENT_KAL = "e-0000000000000001"
ENT_RHO = "e-0000000000000003"


def _chunk(cid: str, text: str, *, title: str = "测试篇") -> dict:
    """造一个带 `turn_spans` 的块——与 `chunk.split_scene` 同坐标。"""
    spans = []
    cursor = 0
    for i, line in enumerate(TURNS):
        end = cursor + len(line)
        spans.append({"ordinal": i, "node_id": 100 + i, "start": cursor, "end": end})
        cursor = end + 1  # "\n"
    return {
        "chunk_id": cid,
        "doc_id": cid.split("#")[0],
        "title": title,
        "chapter": "第一章",
        "text": text,
        "chars": len(text),
        "turn_spans": spans,
    }


def _body() -> str:
    return "\n".join(TURNS)


def _make(chunks: list[dict], *, assertions: list[dict] | None = None,
          entities: list[dict] | None = None):
    ents = entities or [
        {"id": ENT_KAL, "name": "凯尔希", "type": "角色", "aliases": []},
        {"id": ENT_RHO, "name": "罗德岛", "type": "组织", "aliases": []},
    ]
    edges: list[dict] = []
    graph = {
        "schema_version": config.GRAPH_SCHEMA_IDENTITY,
        "entities": ents,
        "edges": edges,
        "alias_index": index_mod.build_alias_index(ents),
        "fingerprint": "0123456789abcdef",
        "stats": {"entities": len(ents), "edges": 0},
    }
    adj = adjacency_mod.build_directed_adjacency(edges)
    idx, _ = index_mod.build(chunks, graph, "corpusfp01234567")
    ret = retrieve.Retriever(idx, graph, adj, chunks=chunks,
                             assertions=assertions or [], vocab=vocab)
    qa = query_mod.analyze_query("凯尔希和罗德岛是什么关系",
                                 entities=ents, vocab=vocab)
    return ret, qa


def _assertion(aid: str, cid: str, text: str, needle: str,
               subj: str = ENT_KAL, obj: str = ENT_RHO) -> dict:
    start = text.find(needle)
    assert start >= 0, f"造数据错误：{needle!r} 不在正文里"
    return {
        "assertion_id": aid,
        "subject_id": subj,
        "predicate": "隶属",
        "object_id": obj,
        "chunk_id": cid,
        "evidence_text": needle,
        "evidence_start": start,
        "evidence_end": start + len(needle),
        "confidence": 0.9,
    }


def _hits(chunks: list[dict]) -> list[dict]:
    return [
        {"chunk_id": c["chunk_id"], "score": round(1.0 - i * 0.01, 4),
         "sources": [{"channel": retrieve.SOURCE_BM25, "score": 1.0}]}
        for i, c in enumerate(chunks)
    ]


def _corpus(n: int = 18):
    """18 个等长块，**每一块都带一条断言**（断言落在第 1 行/第 8 行）。"""
    body = _body()
    chunks: list[dict] = []
    assertions: list[dict] = []
    for i in range(n):
        cid = "doc/a#%03d" % i
        chunks.append(_chunk(cid, body))
        needle = TURNS[1] if i % 2 == 0 else TURNS[8]
        assertions.append(_assertion("a-%013d" % i, cid, body, needle))
    return chunks, assertions


# ---------------------------------------------------------------- 纯函数
class TestEvidenceWindowHelper:
    def test_window_is_a_contiguous_span_covering_all_evidence_turns(self):
        """窗口必须是**连续**区间：下游一切校验偏移的代码都只认连续区间。"""
        text = _body()
        chunk = _chunk("doc/a#000", text)
        spans = [
            {"start": text.find(TURNS[1]), "end": text.find(TURNS[1]) + len(TURNS[1])},
            {"start": text.find(TURNS[8]), "end": text.find(TURNS[8]) + len(TURNS[8])},
        ]
        win = retrieve._evidence_window_span(text, chunk["turn_spans"], spans)
        assert win is not None
        lo, hi = win
        assert lo == chunk["turn_spans"][1]["start"]
        assert hi == chunk["turn_spans"][8]["end"]
        # 两段证据各自都完整落在窗口内（不再逐字 find）
        for s in spans:
            assert lo <= s["start"] and s["end"] <= hi

    def test_window_is_none_without_turn_spans(self):
        """没有 `turn_spans` 必须返回 None —— 绝不猜一个窗口出来。"""
        text = _body()
        assert retrieve._evidence_window_span(text, None, [{"start": 0, "end": 5}]) is None
        assert retrieve._evidence_window_span(text, [], [{"start": 0, "end": 5}]) is None

    def test_window_is_none_without_legal_offsets(self):
        text = _body()
        chunk = _chunk("doc/a#000", text)
        # -1 / 空区间 / 越界：都不许被当成位置
        assert retrieve._evidence_window_span(
            text, chunk["turn_spans"], [{"start": -1, "end": -1}]) is None
        assert retrieve._evidence_window_span(
            text, chunk["turn_spans"], [{"start": 5, "end": 5}]) is None
        assert retrieve._evidence_window_span(
            text, chunk["turn_spans"], [{"start": 10 ** 9, "end": 10 ** 9 + 3}]) is None
        assert retrieve._evidence_window_span(text, chunk["turn_spans"], []) is None

    def test_evidence_turns_maps_offsets_to_turn_indices(self):
        text = _body()
        chunk = _chunk("doc/a#000", text)
        s = text.find(TURNS[3])
        idxs = retrieve._evidence_turns(
            text, chunk["turn_spans"], [{"start": s, "end": s + len(TURNS[3])}])
        assert idxs == [3]


# ---------------------------------------------------------------- 默认路径
class TestDefaultPathIsByteIdentical:
    def test_switch_off_keeps_stats_keys_and_values(self):
        """默认（`evidence_window=0`）时 `stats` / 单元字段集必须与改造前逐位相同。"""
        chunks, assertions = _corpus(6)
        ret, qa = _make(chunks, assertions=assertions)
        hits = _hits(chunks)
        ctx = ret.build_context(hits, [], query_analysis=qa)

        assert "evidence_window" not in ctx["stats"]
        for key in ("windowed_units", "window_full_units", "window_saved_tokens",
                    "window_head_units", "window_chars_kept_pct"):
            assert key not in ctx["stats"], f"默认路径不该出现 L4 的 stat 键 {key!r}"
        for unit in ctx["units"]:
            assert "window_text" not in unit, (
                f"默认路径不该给单元加 window_text（{unit['chunk_id']}）"
            )
            assert "window_applied" not in unit

    def test_switch_off_tokens_used_equals_whole_chunk_sum(self):
        """关着时 `tokens_used` 必须等于各单元**整块**计价之和（历史口径）。"""
        chunks, assertions = _corpus(6)
        ret, qa = _make(chunks, assertions=assertions)
        hits = _hits(chunks)
        ctx = ret.build_context(hits, [], query_analysis=qa)
        by_id = {c["chunk_id"]: c for c in chunks}
        want = 0
        for u in ctx["units"]:
            rec = by_id[u["chunk_id"]]
            want += retrieve.estimate_tokens(
                f"{rec['title']} {rec['chapter']} {rec['text']}",
                tokenizer_version=getattr(ret.index, "tokenizer_version",
                                          index_mod.TOKENIZER_VERSION),
                name_terms=getattr(ret.index, "name_terms", None),
            )
        assert ctx["stats"]["tokens_used"] == want
        assert sum(u["tokens"] for u in ctx["units"]) == want

    def test_switch_off_digest_unchanged_by_new_kwarg(self):
        """传 `evidence_window=0` 与不传，摘要必须一致（默认值 = 现状）。"""
        chunks, assertions = _corpus(6)
        ret, qa = _make(chunks, assertions=assertions)
        hits = _hits(chunks)
        a = ret.build_context(hits, [], query_analysis=qa)
        b = ret.build_context(hits, [], query_analysis=qa, evidence_window=0)
        assert a["context_digest"] == b["context_digest"]
        assert a["stats"] == b["stats"]
        assert [u["chunk_id"] for u in a["units"]] == [u["chunk_id"] for u in b["units"]]


# ---------------------------------------------------------------- 开启后
class TestWindowKeepsFullText:
    def test_unit_text_is_the_original_body_unchanged(self):
        """★ 预登记反证条件 ③：`unit["text"]` 长度/内容一律不许变。"""
        chunks, assertions = _corpus(18)
        ret, qa = _make(chunks, assertions=assertions)
        hits = _hits(chunks)
        off = ret.build_context(hits, [], query_analysis=qa)
        on = ret.build_context(hits, [], query_analysis=qa,
                               evidence_window=1, window_head_units=0)
        before = {u["chunk_id"]: u["text"] for u in off["units"]}
        assert before, "构造失败：默认臂一个单元都没装"
        for u in on["units"]:
            cid = u["chunk_id"]
            assert cid in before
            assert u["text"] == before[cid], (
                f"{cid} 的 unit['text'] 变了：窗口实现就地截断了正文。"
                f"`verify_citations` 查的就是 unit['text'][start:end]，"
                f"一截整条引用就废——这是预登记的反证条件 ③。"
            )
            assert len(u["text"]) == len(before[cid])

    def test_window_text_and_offsets_are_self_consistent(self):
        """`window_text` 必须逐字等于 `text[window_start:window_end]`。"""
        chunks, assertions = _corpus(18)
        ret, qa = _make(chunks, assertions=assertions)
        hits = _hits(chunks)
        on = ret.build_context(hits, [], query_analysis=qa,
                               evidence_window=1, window_head_units=0)
        saw_windowed = 0
        for u in on["units"]:
            lo = int(u["window_start"])
            hi = int(u["window_end"])
            assert 0 <= lo < hi <= len(u["text"]), (
                f"{u['chunk_id']} 的窗口 {lo}:{hi} 越出正文长度 {len(u['text'])}"
            )
            assert u["window_text"] == u["text"][lo:hi], (
                f"{u['chunk_id']} 的 window_text 与 text[lo:hi] 不一致："
                f"下游按连续区间校验偏移，拼出来的多段文本会立刻对不上"
            )
            if u.get("window_applied"):
                saw_windowed += 1
        assert saw_windowed > 0, "窗口一个都没生效，这条门就没测到东西"

    def test_every_evidence_span_lives_inside_its_window(self):
        """断言安放：窗口绝不能把任何一条证据甩到窗口外。"""
        chunks, assertions = _corpus(18)
        ret, qa = _make(chunks, assertions=assertions)
        hits = _hits(chunks)
        on = ret.build_context(hits, [], query_analysis=qa,
                               evidence_window=1, window_head_units=0)
        n = 0
        for u in on["units"]:
            lo, hi = int(u["window_start"]), int(u["window_end"])
            for span in u["evidence_spans"]:
                if span["start"] < 0 or span["end"] <= span["start"]:
                    continue
                n += 1
                assert lo <= span["start"] and span["end"] <= hi, (
                    f"{u['chunk_id']} 的证据 {span['start']}:{span['end']} "
                    f"落在窗口 {lo}:{hi} 外——窗口裁掉了证据本身"
                )
        assert n > 0, "没有可测的证据区间"


class TestWindowFitsMoreUnits:
    def test_window_packs_more_units_and_uses_less_per_unit(self):
        """★ 主判据：同一预算下装进的单元数必须变多，单价必须变低。

        预算压到 3000 才测得出差别：合成块只有约 300 字（≈430 token），
        默认 12000 把 18 个候选全装下，两臂都"装完了"，门会假绿。
        """
        chunks, assertions = _corpus(18)
        ret, qa = _make(chunks, assertions=assertions)
        hits = _hits(chunks)
        budget = 3000
        off = ret.build_context(hits, [], token_budget=budget, query_analysis=qa)
        on = ret.build_context(hits, [], token_budget=budget, query_analysis=qa,
                               evidence_window=1, window_head_units=0)
        assert off["stats"]["units"] < len(chunks), (
            f"构造失败：预算 {budget} 把 {len(chunks)} 个候选全装下了，测不出差别"
        )
        assert on["stats"]["units"] > off["stats"]["units"], (
            f"窗口开启装了 {on['stats']['units']} 个单元，"
            f"整块计价装了 {off['stats']['units']} 个：窗口没起作用。"
        )
        off_per = off["stats"]["tokens_used"] / off["stats"]["units"]
        on_per = on["stats"]["tokens_used"] / on["stats"]["units"]
        assert on_per < off_per * 0.6, (
            f"窗口臂单价 {on_per:.0f} token/单元，整块臂 {off_per:.0f}："
            f"窗口没省下 token（预期省 40% 以上）"
        )
        assert on["stats"]["tokens_used"] <= on["stats"]["token_budget"]
        assert on["stats"]["windowed_units"] > 0
        assert on["stats"]["window_saved_tokens"] > 0

    def test_head_units_stay_whole_chunks(self):
        """分层：前 N 个单元必须装整块（保住 title/chapter 的引文定位）。"""
        chunks, assertions = _corpus(18)
        ret, qa = _make(chunks, assertions=assertions)
        hits = _hits(chunks)
        on = ret.build_context(hits, [], query_analysis=qa,
                               evidence_window=1, window_head_units=4)
        head = on["units"][:4]
        for u in head:
            assert u["window_applied"] is False, (
                f"头部单元 {u['chunk_id']} 被裁了：头部必须装整块，"
                f"否则 title/chapter 通路的引文会落到被丢弃的轮里"
            )
            assert u["window_start"] == 0
            assert u["window_end"] == len(u["text"])
            assert u["window_text"] == u["text"]
        # 头部之外必须有单元真的被裁（否则 head 参数等于没接上）
        assert any(u["window_applied"] for u in on["units"]), (
            "传了 window_head_units=4 但没有任何单元被裁：头部条数没接上"
        )

    def test_heap_head_zero_windows_everything(self):
        """`window_head_units=0` 是合法的显式取值（整批都按窗口计价）。"""
        chunks, assertions = _corpus(8)
        ret, qa = _make(chunks, assertions=assertions)
        on = ret.build_context(_hits(chunks), [], query_analysis=qa,
                               evidence_window=1, window_head_units=0)
        assert on["stats"]["window_head_units"] == 0
        assert on["stats"]["window_full_units"] == 0, (
            "头部条数为 0 时不该有单元被判成「整块」，实际 "
            f"{on['stats']['window_full_units']} 个"
        )

    def test_units_without_turn_spans_fall_back_to_whole_chunk(self):
        """没有 `turn_spans` 的块必须退回整块计价，且不许标 `window_applied`。"""
        body = _body()
        legacy = {"chunk_id": "doc/b#000", "doc_id": "doc/b", "title": "老世代",
                  "chapter": "第一章", "text": body, "chars": len(body)}  # 无 turn_spans
        assertions = [_assertion("a-0000000000000009", "doc/b#000", body, TURNS[1])]
        ret, qa = _make([legacy], assertions=assertions)
        on = ret.build_context(_hits([legacy]), [], query_analysis=qa,
                               evidence_window=1, window_head_units=0)
        assert on["stats"]["units"] == 1
        u = on["units"][0]
        assert u["window_applied"] is False
        assert u["window_text"] == body
        assert on["stats"]["windowed_units"] == 0
        assert on["stats"]["window_full_units"] == 1


class TestWindowWithReserve:
    def test_reserve_path_still_counts_each_unit_once(self):
        """`graph_reserve_tokens` 路径会把同一单元算两次；账目必须不被重复累计。"""
        chunks, assertions = _corpus(18)
        ret, qa = _make(chunks, assertions=assertions)
        hits = _hits(chunks)
        on = ret.build_context(hits, [], query_analysis=qa,
                               evidence_window=1, window_head_units=0,
                               graph_reserve_tokens=2000)
        assert on["stats"]["windowed_units"] <= len(on["units"])
        assert sum(u["tokens"] for u in on["units"]) == on["stats"]["tokens_used"], (
            "各单元 tokens 之和与 stats['tokens_used'] 对不上："
            "窗口计价在 reserve 路径上被重复累计了"
        )


class TestContextStillSerialisable:
    def test_window_context_is_json_serialisable(self):
        chunks, assertions = _corpus(6)
        ret, qa = _make(chunks, assertions=assertions)
        on = ret.build_context(_hits(chunks), [], query_analysis=qa,
                               evidence_window=1, window_head_units=2)
        blob = json.dumps(on, ensure_ascii=False)
        assert json.loads(blob)["stats"]["evidence_window"] == 1


# ---------------------------------------------------------------- 生产档不被污染
class TestProdProfileUntouched:
    def test_prod_profile_does_not_enable_the_window(self):
        """★ `PROD_PROFILE` 里**不许**出现 `evidence_window`。

        本轮预登记写明：端到端分数还没量，生产档一个字都不许动。这个开关一旦
        进了生产档，`server` / `scripts/ask.py` / 评测脚本三处同时生效，
        而"读数变了"这件事没有任何测试会报——它只会表现为分数悄悄动了。
        要开必须由 Lead 在拿到三份 split 读数之后单独改这一处。
        """
        from arkrag import pipeline

        opts = pipeline.PROD_PROFILE["options"]
        assert "evidence_window" not in opts, (
            "`PROD_PROFILE` 里出现了 `evidence_window`：本轮预登记禁止动生产档。"
            "端到端读数出来、Lead 复核过再单独改。"
        )
        assert "window_head_units" not in opts
        assert pipeline.PROD_PROFILE["max_context_tokens"] == 24000

    def test_pipeline_passes_the_switch_only_when_set(self):
        """`ask_once` 只在 `evidence_window` 为真时才拼关键字（默认不拼）。

        默认路径必须连**调用形状**都和改造前一样：既有 `build_context` 替身
        （测试桩、旧版实现）不会因为多一个关键字而 TypeError。
        """
        from arkrag import pipeline

        src = (__import__("pathlib").Path(pipeline.__file__)).read_text("utf-8")
        # 关键字只在 `opts.get("evidence_window")` 分支里出现
        assert 'if opts.get("evidence_window"):' in src
        assert 'ctx_kwargs["evidence_window"] = 1' in src
        # `window_head_units` 用 `is not None` 判，不许用 `or`：0 是合法取值
        assert 'if opts.get("window_head_units") is not None:' in src
