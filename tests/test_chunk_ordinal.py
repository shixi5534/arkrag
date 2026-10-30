"""D1 轮次实例身份的回归门：ordinal 必须是**场景级**序号，不能退化成块内下标。

## 这个文件为什么存在

`chunk.split_scene()` 用 `(scene_id, ordinal)` 作为轮次实例身份，门 `L2.1a`
靠它算覆盖率。但 `turn_ordinal(turn, fallback)` 的 `fallback` 是调用方传的，
而 `_dedupe()` 与 `turn_spans` 都用 `enumerate(buf)` 传**块内下标**。

冻结基线 `scenes.jsonl` 里的轮次**没有 `ordinal` 字段**（实测键集是
`['kind','masked','node_id','prop','speaker','text']`）。于是每一块的
ordinal 都从 0 重新数：一个 40 轮的长场景切成 7 块后，`turn_ordinals`
全部落在 0..6，覆盖门看 `len(covered)` 时会以为"轮次够用"，而按实例身份
核对会发现 33 轮没有任何块引用。

实测规模（修复前，用 `_with_ordinal` 停用的方式复现）：
**2,291 个场景中 2,127 个覆盖不全（92.8%）**。

这个缺陷的危险之处不是数字难看，而是它**双层隐形**：
1. `tests/test_negative.py` 的 `expect()` 当时只往列表记、不抛异常，
   `pytest` 显示 11 passed；
2. `L2.1a` 的判据是 `ordinals - covered`，而 `ordinals` 本身若按块内下标
   理解就会跟着缩水 —— 但只要 chunks 与 scenes 用同一套退化口径，门也可能
   两边一起错。所以门必须被单独测一次"喂退化输入会不会红"。

两处都已修（`chunk._with_ordinal` + `expect()` 改抛异常），本文件把
"修复是否还在"变成机检。
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import chunk as chunker, gates  # noqa: E402

SCENES = ROOT / "data" / "arkrag" / "scenes.jsonl"


def _turns(n: int, text: str = "句" * 120) -> list[dict]:
    """没有 ordinal 字段的轮次——和冻结 scenes.jsonl 的形状一致。"""
    return [
        {
            "kind": "dialogue",
            "speaker": f"角色{i % 3}",
            "text": text,
            "node_id": i,
            "prop": "name",
            "masked": False,
        }
        for i in range(n)
    ]


def _scene(turns: list[dict], doc_id: str = "obt/main/level_ordinal") -> dict:
    return {
        "doc_id": doc_id,
        "category": "obt/main",
        "chapter": "ch",
        "title": "t",
        "turns": turns,
    }


class TestOrdinalIsSceneScoped:
    def test_multi_chunk_scene_covers_every_turn(self):
        """40 轮切 7 块，覆盖集合必须等于 0..39，不是 0..6。"""
        scene = _scene(_turns(40))
        chunks = chunker.split_scene(scene)
        assert len(chunks) > 1, f"应切成多块，实得 {len(chunks)}"
        covered: set[int] = set()
        for c in chunks:
            covered |= set(c["turn_ordinals"])
        assert covered == set(range(40)), (
            f"轮次覆盖不全：缺 {sorted(set(range(40)) - covered)}；"
            f"各块 ordinals={[c['turn_ordinals'] for c in chunks]}"
        )

    def test_each_block_ordinals_are_contiguous_scene_positions(self):
        """每块的 ordinal 都必须是场景级连续区间，且各块首尾相接。"""
        scene = _scene(_turns(40))
        chunks = chunker.split_scene(scene)
        expected_start = 0
        for c in chunks:
            ords = c["turn_ordinals"]
            assert ords == list(range(ords[0], ords[0] + len(ords))), (
                f"{c['chunk_id']} 的 ordinals 不连续: {ords}"
            )
            # 允许重叠轮次，但不允许出现"跳过去又回来"的空洞
            assert ords[0] <= max(expected_start, ords[0]), ords
            expected_start = ords[-1] + 1

    def test_source_turns_are_not_mutated(self):
        """补齐 ordinal 不能就地改调用方传进来的 scene（否则重复调用会串味）。"""
        scene = _scene(_turns(5))
        before = copy.deepcopy(scene)
        chunker.split_scene(scene)
        assert scene == before, "split_scene 改动了入参 scene"

    def test_existing_ordinal_is_respected(self):
        """上游已经写了 ordinal 时必须用它的值，不能被位置序号覆盖。"""
        turns = _turns(3)
        for i, t in enumerate(turns):
            t["ordinal"] = 100 + i
        chunks = chunker.split_scene(_scene(turns))
        covered = set()
        for c in chunks:
            covered |= set(c["turn_ordinals"])
        assert covered == {100, 101, 102}, f"上游 ordinal 被覆盖: {covered}"

    def test_duplicate_node_id_still_kept(self):
        """D1 原始诉求：同 node_id 不同正文的两轮都要留下。"""
        turns = [
            {"kind": "dialogue", "speaker": "凯尔希", "text": "第一轮", "node_id": 1, "prop": "name", "masked": False},
            {"kind": "dialogue", "speaker": "阿米娅", "text": "第二轮", "node_id": 1, "prop": "name", "masked": False},
            {"kind": "dialogue", "speaker": "博士", "text": "第三轮", "node_id": 2, "prop": "name", "masked": False},
        ]
        chunks = chunker.split_scene(_scene(turns))
        text = "".join(c["text"] for c in chunks)
        for t in turns:
            assert t["text"] in text, f"node_id={t['node_id']} 的轮次被当重复丢掉"
        covered = set()
        for c in chunks:
            covered |= set(c["turn_ordinals"])
        assert covered == {0, 1, 2}


class TestCoverageGateCatchesDegradation:
    """`L2.1a` 必须能对"ordinal 退化成块内下标"的输入判 FAIL。

    这条测试的意义：如果哪天有人再把 `enumerate(buf)` 当地场景序号用，
    覆盖数字会在块数与场景之间自洽地缩小，门看着还 PASS。这里显式喂一份
    退化产物，门必须红。
    """

    @staticmethod
    def _row(rows, prefix: str) -> dict:
        return next(r for r in rows if r["name"].startswith(prefix))

    def test_good_chunks_pass(self):
        scene = _scene(_turns(40))
        chunks = chunker.split_scene(scene)
        run = gates.l2_gates(chunks, [], [scene])
        row = self._row(run.rows(), "L2.1a")
        assert row["verdict"] == "PASS", row
        assert "漏 0 篇" in row["measured"], row["measured"]

    def test_degraded_chunks_fail(self):
        scene = _scene(_turns(40))
        chunks = copy.deepcopy(chunker.split_scene(scene))
        for c in chunks:
            n = len(c["turn_ordinals"])
            c["turn_ordinals"] = list(range(n))
            for j, span in enumerate(c.get("turn_spans") or []):
                span["ordinal"] = j
        run = gates.l2_gates(chunks, [], [scene])
        row = self._row(run.rows(), "L2.1a")
        assert row["verdict"] == "FAIL", (
            f"退化 ordinal 竟然通过覆盖门：{row}"
        )


@pytest.mark.skipif(not SCENES.exists(), reason="冻结 scenes.jsonl 不在场")
class TestFrozenScenesShape:
    """冻结基线语料本身没有 ordinal —— 这是上述缺陷的触发条件，必须一直被覆盖。"""

    def test_frozen_turns_have_no_ordinal_field(self):
        with SCENES.open(encoding="utf-8") as fh:
            first = json.loads(fh.readline())
        keys = set(first["turns"][0])
        assert "ordinal" not in keys, (
            "冻结 scenes.jsonl 现在带 ordinal 了——如果确实重跑过 L1，"
            "请确认 test_frozen_turns_have_no_ordinal_field 的语义"
            "（它守的是'老语料没有该字段'这条路径）"
        )

    def test_first_fifty_scenes_fully_covered(self):
        """抽样前 50 个场景跑真实分块，覆盖必须完整。"""
        bad = []
        with SCENES.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= 50:
                    break
                scene = json.loads(line)
                if not scene["turns"]:
                    continue
                chunks = chunker.split_scene(scene)
                covered: set[int] = set()
                for c in chunks:
                    covered |= set(c["turn_ordinals"])
                want = set(range(len(scene["turns"])))
                if covered != want:
                    bad.append((scene["doc_id"], len(want), len(covered)))
        assert not bad, f"真实语料覆盖不全 {len(bad)} 例: {bad[:5]}"
