"""给题干引文定位通路（`src/arkrag/quote_locate.py`）写门。

钉六件事，每一件都对应一次实测事故或一次对抗审查的硬要求：

1. `MIN_PROBE_CHARS` 必须是 8。它是**唯一的**长度阈值，且必须与
   `answer.TITLE_TRIGGERS` 的组合一起决定触发面——两处各写一个数就会出现
   "改了常量但实例还用旧值"那种没接线事故。
2. 触发面必须是 `TITLE_TRIGGERS` 判定的标题题 **且** 有 ≥8 字引文。全 640 题里
   只命中 `single_chunk` 128/128、其余七类 0，这条门用真实 gold 集复核。
3. 最长引文胜出、短引文被忽略。第二长引文若被误选，命中块就会错。
4. 定位结果只含 `{probe, chunk_ids, skipped_reason}`——**不许顺手产出答案文本**。
   一旦它开始产出答案字符，"定位"和"作答"就混在一起，判据之外的信息会漏进分数。
5. 扰动对照：把引文随机改一个汉字，命中必须归零。对抗审查用它证明本通路依赖
   真实原文对齐、而不是在复现金标生成规则（实测改 1 字 → 0/76、0/52、0/32）。
6. 源码里必须留着对抗审查的披露项（引文唯一率 98.7/98.1/96.9%、引文长度 p50=16、
   引文≠gold 0/76）与三组对照读数。删了这些，下一个读这个模块的人会以为
   命中率 100% 是"检索变强了"。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from arkrag import answer as answer_mod
from arkrag import quote_locate as ql

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "arkrag" / "quote_locate.py"

PinnedMinProbe = 8


def _mkchunk(cid: str, text: str, *, title: str = "章节标题") -> dict:
    return {"chunk_id": cid, "text": text, "title": title, "chapter": "ch1"}


def _title_question(probe: str) -> str:
    return f"以下这段剧情出自哪个文本块？「{probe}」请回答该段的标题。"


LONG = "博士把那份名单交到凯尔希手上时，窗外的雪已经停了"   # 25 字
SHORT = "凯尔希"                                          # 3 字


class TestConstantsArePinned:
    def test_module_constant_is_pinned(self) -> None:
        assert ql.MIN_PROBE_CHARS == PinnedMinProbe, (
            f"MIN_PROBE_CHARS 现在是 {ql.MIN_PROBE_CHARS}，登记值 {PinnedMinProbe}。"
            "8 字是实测拐点：截到 4 字时命中会跨块扩散（对照组 40 题仍对），"
            "8 字起唯一率稳定在 96%+（dev/holdout/clean = 98.7/98.1/96.9%）。"
            "改之前先跑 scripts/eval_answer_quality.py 三份 split。"
        )

    def test_locate_default_follows_the_constant(self) -> None:
        """`locate()` 不传 `min_probe_chars` 时必须等于模块常量。"""
        chunks = {"c#1": _mkchunk("c#1", "正文里有" + SHORT + "这三个字")}
        out = ql.locate(_title_question(SHORT), chunks)
        assert out["skipped_reason"] == "no_probe_long_enough", (
            "3 字引文应被 `probe_of` 直接滤掉（长度门槛 ≤ 模块常量）；"
            f"实际 skipped_reason={out['skipped_reason']!r}。"
        )


class TestProbeSelection:
    def test_longest_quote_wins(self) -> None:
        """两段引文都过长度门槛时，必须取更长的那个当定位键。

        取样要点：被比下去的那段**也必须 ≥ `MIN_PROBE_CHARS`**，否则
        `probe_of` 在取最长之前就把它滤掉了，`quotes[0]` 与 `max(...)` 恰好
        相等——这条测试对"取最长"这层逻辑就是假绿（突变实测：换成
        `return quotes[0]` 仍全绿）。
        """
        shorter = "雪已经停了之后凯尔希才开口"          # 14 字
        assert len(shorter) >= PinnedMinProbe and len(shorter) < len(LONG)
        q = f"「{shorter}」出自哪一段剧情？「{LONG}」该块的标题是什么？"
        assert ql.probe_of(q) == LONG, (
            f"多引文时取错定位键：期望 {LONG!r}（{len(LONG)} 字），"
            f"实际 {ql.probe_of(q)!r}。取错就会定位到错的块。"
        )

    def test_short_quote_is_ignored(self) -> None:
        assert ql.probe_of(f"「{SHORT}」请给出该块的标题。") == "", (
            f"短于 {PinnedMinProbe} 字的引文不许当定位键（跨块扩散，实测 40 题仍对，"
            "说明它对定位没有约束力）。"
        )

    def test_no_quote_gives_empty_probe(self) -> None:
        assert ql.probe_of("请给出该块的标题。") == ""

    def test_quote_regex_is_not_shared_with_answer(self) -> None:
        """本模块的引文正则不设 4 字下限，长度门槛由本模块常量单独把。"""
        src = SOURCE.read_text(encoding="utf-8")
        assert "MIN_PROBE_CHARS = 8" in src
        # `answer._QUOTE_RE` 是 `{4,}`；本模块若照抄，`probe_of` 的 `>= 8` 过滤就
        # 成了死代码，而 4~7 字的引文会被静默吞进候选。
        assert r"「([^」]{1,200})」" in src


class TestTitleTrigger:
    def test_trigger_words_come_from_answer(self) -> None:
        for trigger in answer_mod.TITLE_TRIGGERS:
            assert ql.is_title_question(f"「{LONG}」{trigger}"), (
                f"触发词 {trigger!r} 在 answer.TITLE_TRIGGERS 里，"
                "本模块却不认它——两处判定已经分叉。"
            )

    def test_non_title_question_is_rejected(self) -> None:
        assert not ql.is_title_question(f"「{LONG}」这个中间实体是谁？")

    def test_full_gold_set_trigger_surface_is_single_chunk_only(self) -> None:
        """真实 gold 集复核门控：640 题里只有 single_chunk 会触发。

        门是**两个条件的与**：`is_title_question(q) and probe_of(q)`。只用
        「有 ≥8 字引文」这一半会命中 163 题（dev 上 multihop 2、negative_abstain 17
        ……），因为那些问句也带「」；加上标题触发词这一半才收到 single_chunk 128/128。
        """
        import json

        path = ROOT / "data" / "goldset" / "gold-v2.json"
        items = json.loads(path.read_text(encoding="utf-8"))["items"]
        hit: dict[str, int] = {}
        for it in items:
            q = it.get("question") or ""
            if ql.is_title_question(q) and ql.probe_of(q):
                cat = it.get("category") or "?"
                hit[cat] = hit.get(cat, 0) + 1
        assert set(hit) == {"single_chunk"}, (
            f"引文定位在当前 gold 集上命中了 {hit}。门控要求只有 single_chunk 触发——"
            "多命中一类就等于把这条通路扩散到别的题型上，而那些题型的答案不由 "
            "`_title_answer` 产出。"
        )
        assert hit["single_chunk"] == 128, (
            f"single_chunk 应命中 128 题，实际 {hit['single_chunk']}；"
            "说明 MIN_PROBE_CHARS 或触发词被改动了。"
        )

    def test_probe_alone_is_not_enough(self) -> None:
        """只用「有 ≥8 字引文」这一半会误伤其它题型——这门钉住"必须两个条件都满足"。"""
        import json

        path = ROOT / "data" / "goldset" / "gold-v2.json"
        items = json.loads(path.read_text(encoding="utf-8"))["items"]
        loose = {it.get("category") for it in items
                 if ql.probe_of(it.get("question") or "")}
        assert loose != {"single_chunk"}, (
            "只用'有 ≥8 字引文'就够收窄到 single_chunk 了？那说明 `is_title_question` "
            "这一半是多余的——先确认 gold 集没变，再决定要不要删触发词。"
        )


class TestLocate:
    def test_skips_non_title_question(self) -> None:
        out = ql.locate(f"「{LONG}」这个中间实体是谁？", {})
        assert out["chunk_ids"] == [] and out["skipped_reason"] == "not_title_question"

    def test_skips_when_probe_is_absent_from_the_corpus(self) -> None:
        chunks = {"c#1": _mkchunk("c#1", "完全无关的一段正文内容")}
        out = ql.locate(_title_question(LONG), chunks)
        assert out["chunk_ids"] == []
        assert out["skipped_reason"] == "probe_not_in_corpus"

    def test_finds_the_host_chunk(self) -> None:
        chunks = {
            "c#1": _mkchunk("c#1", "无关正文", title="甲"),
            "c#2": _mkchunk("c#2", "开场白。" + LONG + "。收尾。", title="乙"),
        }
        out = ql.locate(_title_question(LONG), chunks)
        assert out["chunk_ids"] == ["c#2"], (
            f"引文只出现在 c#2 里，却定位到 {out['chunk_ids']}。"
        )
        assert out["skipped_reason"] == ""

    def test_multiple_hits_are_sorted(self) -> None:
        chunks = {
            "c#9": _mkchunk("c#9", LONG),
            "c#3": _mkchunk("c#3", "前缀" + LONG),
        }
        out = ql.locate(_title_question(LONG), chunks)
        assert out["chunk_ids"] == ["c#3", "c#9"], (
            "命中多块时顺序必须固定，否则读数不可复现（答案器会因 title 不唯一而拒答）。"
        )

    def test_accepts_dict_and_list_forms(self) -> None:
        recs = [_mkchunk("c#1", "无关正文"), _mkchunk("c#2", LONG)]
        as_list = ql.locate(_title_question(LONG), recs)
        as_dict = ql.locate(_title_question(LONG), {r["chunk_id"]: r for r in recs})
        assert as_list == as_dict == {"probe": LONG, "chunk_ids": ["c#2"],
                                      "skipped_reason": ""}

    def test_result_never_carries_answer_text(self) -> None:
        out = ql.locate(_title_question(LONG), {"c#2": _mkchunk("c#2", LONG, title="乙")})
        assert set(out) == {"probe", "chunk_ids", "skipped_reason"}, (
            f"定位结果多出了键 {sorted(set(out))}。本模块只决定'去哪儿找'，"
            "答案字符必须只由 `answer._title_answer` 从 `unit['title']` 取——"
            "否则'定位'与'作答'混在一起，分数里会混进判据之外的信息。"
        )
        assert "乙" not in str(out.values()), "定位结果里出现了 title（答案文本）。"

    def test_mutating_one_character_breaks_the_hit(self) -> None:
        """扰动对照：改一个汉字必须定位失败（对抗审查的 0/76、0/52、0/32）。"""
        chunks = {"c#2": _mkchunk("c#2", LONG)}
        mutated = LONG[:5] + "雪" + LONG[6:]
        assert mutated != LONG
        ok = ql.locate(_title_question(LONG), chunks)
        bad = ql.locate(_title_question(mutated), chunks)
        assert ok["chunk_ids"] == ["c#2"]
        assert bad["chunk_ids"] == [], (
            "改一个字之后仍然命中——说明匹配不是逐字子串，"
            "那 100% 命中率就不再是'原文对齐'的证据。"
        )


class TestSourceKeepsTheDisclosure:
    def test_records_the_three_controls(self) -> None:
        src = SOURCE.read_text(encoding="utf-8")
        for token in ("0/76", "0/52", "0/32", "76/75/75/74/62/33"):
            assert token in src, f"源码里找不到对抗审查的对照读数 {token}。"

    def test_records_the_unique_rate_and_length(self) -> None:
        src = SOURCE.read_text(encoding="utf-8")
        for token in ("98.7", "98.1", "96.9", "p50 = 16"):
            assert token in src, (
                f"源码里找不到披露项 {token}：命中率几乎 100% 主要来自"
                "'≥8 字中文子串在 13,798 块里几乎唯一'，不写清楚就会被读成检索变强。"
            )

    def test_records_that_the_probe_is_not_the_answer(self) -> None:
        src = SOURCE.read_text(encoding="utf-8")
        assert "0/76" in src and ("答案不是抄引文" in src or "引文与 gold" in src)

    def test_records_the_gate_surface(self) -> None:
        src = SOURCE.read_text(encoding="utf-8")
        assert "128/128" in src, "触发面（single_chunk 128/128、其余 0）必须写在源码里。"
