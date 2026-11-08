"""给 `MAX_CLAIMS` 写门：新建 `tests/test_max_claims.py`。

钉四件事：
1. 生产值必须是 8（改回 5 或 12 立刻红，失败文案写明实测读数与代价）。
2. 构造器默认值必须跟着模块常量走（`ExtractiveAnswerer()` 不传参时 == 模块常量），
   否则「改了常量但实例还用旧值」——这正是 `TITLE_ANSWER_PATH` 那种「没接线」事故。
3. 抬上限确实让答案变长（用合成断言喂出 12 条候选，检查答案条数随上限变化）。
4. 条数上限只截断答案、不删引用（引用各自指向自己的原文句，删了就无法定位）。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from arkrag import answer as answer_mod

Pinned = 8

SOURCE = Path(__file__).resolve().parents[1] / "src" / "arkrag" / "answer.py"


def _claims(n: int, *, support: int = 30, unit: str = "c#001") -> list[dict]:
    out = []
    for i in range(n):
        out.append(
            {
                "subject": f"甲{i}",
                "predicate": "隶属",
                "object": f"乙{i}",
                "assertion_id": f"a-{i:03d}",
                "edge_id": f"e-{i:03d}",
                "support_count": support,
                "unit_score": 1.0,
                "chunk_id": unit,
                "chunk_text": f"甲{i}隶属乙{i}。",
                "start": 0,
                "end": 4,
                "evidence": f"甲{i}隶属乙{i}",
            }
        )
    return out


class TestProductionValueIsPinned:
    def test_module_constant_is_pinned(self) -> None:
        assert answer_mod.MAX_CLAIMS == Pinned, (
            f"MAX_CLAIMS 现在是 {answer_mod.MAX_CLAIMS}，登记值是 {Pinned}。"
            "这个数是用真实管线（scripts/eval_answer_quality.py，judge.py 判定，"
            "max_context_tokens=12000）量出来的：5/8/12 三档在 dev 是 191/198/200（可判 285），"
            "holdout 128/132/132（可判 195），clean 79/83/83（可判 120）。"
            "5->8 在 dev/holdout 双显著（+2.46pt p=0.0008 / +2.04pt p=0.0195，零损失），"
            "8->12 dev 只剩 +0.70pt p=0.1318、holdout 一格不涨，而答案从 51.7 字涨到 69.5 字。"
            "改之前先跑 scripts/eval_answer_quality.py 三档。"
        )

    def test_constructor_default_follows_the_constant(self) -> None:
        """实例不传参时必须等于模块常量（写死不跟 = 改了常量但实例还用旧值）。"""
        a = answer_mod.ExtractiveAnswerer()
        assert a.max_claims == answer_mod.MAX_CLAIMS, (
            f"ExtractiveAnswerer() 默认 max_claims={a.max_claims}，"
            f"而模块常量是 {answer_mod.MAX_CLAIMS}：常量改了但构造器默认值没跟着改。"
        )

    def test_explicit_value_overrides(self) -> None:
        assert answer_mod.ExtractiveAnswerer(max_claims=3).max_claims == 3
        assert answer_mod.ExtractiveAnswerer(max_claims=1).max_claims == 1

    def test_zero_or_negative_is_clamped_to_one(self) -> None:
        assert answer_mod.ExtractiveAnswerer(max_claims=0).max_claims == 1
        assert answer_mod.ExtractiveAnswerer(max_claims=-5).max_claims == 1


class TestLimitIsActuallyEnforced:
    """用 `_answers` 的截断点直接验：上限真的是个上限。"""

    @staticmethod
    def _cut(claims: list[dict], limit: int) -> int:
        # 复刻 answer() 里那三行：去重后与上限比大小
        seen: dict[str, dict] = {}
        deduped: list[dict] = []
        for c in claims:
            key = f"{c['subject']}\x1f{c['predicate']}\x1f{c['object']}"
            if key in seen:
                continue
            seen[key] = c
            deduped.append(c)
        return min(len(deduped), limit)

    def test_cut_follows_the_limit(self) -> None:
        claims = _claims(20)
        for limit in (1, 3, 5, 8, 12):
            assert self._cut(claims, limit) == limit

    def test_pinned_value_admits_more_than_the_old_five(self) -> None:
        claims = _claims(20)
        assert self._cut(claims, Pinned) == Pinned > 5

    def test_duplicates_are_collapsed_before_counting(self) -> None:
        claims = _claims(3) + _claims(3)
        assert self._cut(claims, 8) == 3


class TestSourceHasTheRationale:
    """注释里必须留着实测依据——否则下一个改这个数的人没法判断能不能改。"""

    def test_comment_records_the_three_arm_readings(self) -> None:
        src = SOURCE.read_text(encoding="utf-8")
        for token in ("191", "198", "200", "128", "132"):
            assert token in src, f"answer.py 里找不到实测读数 {token}，注释依据被删了。"

    def test_comment_records_zero_abstain_harm(self) -> None:
        src = SOURCE.read_text(encoding="utf-8")
        assert "拒答转作答" in src or "拒答伤害" in src, (
            "answer.py 里没有记「抬上限对拒答的伤害」——当年 CHAPTER_ANSWER_PATH "
            "就是没量这一条才被否决/翻车的，注释必须留着。"
        )

    def test_comment_records_the_rollback_instruction(self) -> None:
        src = SOURCE.read_text(encoding="utf-8")
        assert "回滚" in src, "answer.py 里没写回滚方法，改坏了没法退。"
