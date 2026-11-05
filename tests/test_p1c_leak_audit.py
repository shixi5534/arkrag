# -*- coding: utf-8 -*-
"""`scripts/p1c_leak_audit.py` 的门：A 类「gold 引用指错行」的四态判定必须可复算。

## 为什么这个脚本需要门

它是 P1-c 唯一的判定来源，而 P1-c 的产物会直接导致**改 gold**。一条判错的
`move_safe` 会让「别名出现」这个唯一证据从 gold 里消失，而 gold 一改就是
改代码改 hash 的动作（Lead 要签字）。所以判定逻辑本身必须有能变红的用例。

## 四态里最容易错的一处

`move_safe` 与 `needs_extra` 都表现为「答案在同块另一行」，但补救动作相反：

- `move_safe`：那一行**也含**问句里的别名 → 挪过去安全（一行同时证明两件事）；
- `needs_extra`：那一行**不含**别名 → 挪过去就丢掉别名证据，只能追加 required。

把这两态混成一种是这一轮最容易犯的错——它们会一起被报成「15 条可改」，
而实际上改过去会让引用更难被断言支持。本文件的门就是钉住这个区分。

## 反向实测怎么做

删掉 `move_safe` 分支（让它落到 `needs_extra`）→
`test_alias_on_the_answer_line_is_what_makes_it_safe_to_move` 红；
把「承载答案的断言是否与 required 同行」的区间重叠判据改成恒假 →
`test_the_answer_sitting_on_the_required_line_needs_no_change` 红；
把 `BOOLEAN_ANSWERS` 清空 → `test_boolean_answers_are_not_judged_by_substring` 红。
命令级记录见 `docs/p1c_漏标候选判定表.md` 的变更日志与
`docs/p0_门报告可追溯.md` §八。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

SCRIPT = ROOT / "scripts" / "p1c_leak_audit.py"
JSON_OUT = ROOT / "data" / "p1c_leak_audit.json"
DOC = ROOT / "docs" / "p1c_漏标候选判定表.md"


def _load():
    spec = importlib.util.spec_from_file_location("p1c_leak_audit_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = _load()


# ---------------------------------------------------------------- 合成语料


def _corpus(chunks: dict[str, str], assertions: dict[str, list[dict]]) -> dict:
    """造一份最小语料：`chunks` 是 chunk_id→正文，`assertions` 是 chunk_id→断言列表。

    `assertions` 里每条只要 `evidence_text` / `evidence_start` / `evidence_end`，
    其它字段判据不用。span 必须和正文里实际位置一致——判据靠区间重叠认「同一行」，
    span 编错会让门自己失去判别力（那正是本项目栽过的坑）。
    """
    by_chunk = {}
    for cid, rows in assertions.items():
        by_chunk[cid] = [
            {
                "assertion_id": f"a-{cid}-{i}",
                "chunk_id": cid,
                "evidence_text": row["evidence_text"],
                "evidence_start": row["evidence_start"],
                "evidence_end": row["evidence_end"],
            }
            for i, row in enumerate(rows)
        ]
    return {
        "chunks": {cid: {"chunk_id": cid, "text": text} for cid, text in chunks.items()},
        "by_chunk": by_chunk,
    }


def _span(text: str, needle: str) -> tuple[int, int]:
    i = text.index(needle)
    return i, i + len(needle)


def _item(question: str, answer: str, chunk_id: str, required: str,
          split: str = "dev") -> dict:
    return {
        "question_id": "q-al-000001",
        "category": "alias",
        "question": question,
        "answer": answer,
        "acceptable_answers": [answer],
        "gold_chunk_ids": [chunk_id],
        "required_citations": [{"chunk_id": chunk_id, "evidence": required}],
        "should_abstain": False,
    }


def _judge_one(item: dict, corpus: dict) -> dict:
    """跑一条题的判定，返回单行结果（题级判定取最坏的一行）。"""
    result = AUDIT.a_class_requirements(
        [item], corpus, {}, [item["question_id"]], {item["question_id"]: "dev"})
    return result["rows"][0]


# ---------------------------------------------------------------- 四态


class TestFourVerdicts:
    """四态各自的判据，以及它们为什么不能合并。"""

    def test_the_answer_sitting_on_the_required_line_needs_no_change(self) -> None:
        """承载答案的断言与 required 在同一行 → `carrier_ok`，不许报成要改。

        这是「引用口径窄」而不是「gold 指错行」：gold 把说话人前缀也写进了
        reference，断言从冒号后面开始。改它等于改引用口径，不是改标注。
        """
        text = "杜林青年：自我介绍一下，我叫卡奇·叙光，是这座城市的代表。"
        needle = "自我介绍一下，我叫卡奇·叙光，是这座城市的代表。"
        start, end = _span(text, needle)
        corpus = _corpus(
            {"c1": text},
            {"c1": [{"evidence_text": needle, "evidence_start": start, "evidence_end": end}]},
        )
        item = _item("「杜林青年」被用来指代谁？", "卡奇·叙光", "c1",
                     "杜林青年：自我介绍一下，我叫卡奇·叙光，是这座城市的代表。")
        row = _judge_one(item, corpus)
        assert row["verdict"] == "carrier_ok", (
            f"承载答案的断言就在 required 那一行上，不该要求改 gold；实际 {row['verdict']}，"
            f"理由：{row['requirements'][0]['reason']}"
        )

    def test_alias_on_the_answer_line_is_what_makes_it_safe_to_move(self) -> None:
        """答案在另一行、且那一行**也含别名** → `move_safe`（可挪）。

        构造：required 那句含别名但不含答案；答案那行同时含别名与答案。
        挪过去安全，因为那一行独立证明了「别名在这」和「答案是什么」。
        """
        text = ("热情的村民：哟，这不是小缠丸嘛！\n"
                "村民：热情的村民说的是繁藏阿伯。\n")
        answer_needle = "热情的村民说的是繁藏阿伯。"
        start, end = _span(text, answer_needle)
        corpus = _corpus(
            {"c1": text},
            {"c1": [{"evidence_text": "热情的村民说的是繁藏阿伯",
                     "evidence_start": start, "evidence_end": end}]},
        )
        item = _item("「热情的村民」指代谁？", "繁藏", "c1",
                     "热情的村民：哟，这不是小缠丸嘛")
        row = _judge_one(item, corpus)
        assert row["verdict"] == "move_safe", (
            f"答案那一行也含别名，挪过去是安全的；实际 {row['verdict']}，"
            f"理由：{row['requirements'][0]['reason']}"
        )

    def test_answer_on_a_line_without_the_alias_must_not_be_moved(self) -> None:
        """答案在另一行、但那一行**不含别名** → `needs_extra`（不能挪）。

        挪过去就丢掉「别名出现」这个唯一证据，而块里没有任何一行同时给出两者。
        把这一态与 `move_safe` 合并是这一轮最容易犯的错。
        """
        text = ("热情的村民：哟，这不是小缠丸嘛！\n"
                "缠丸：繁藏阿伯，没时间跟你聊了。\n")
        needle = "缠丸：繁藏阿伯，没时间跟你聊了"
        start, end = _span(text, needle)
        corpus = _corpus(
            {"c1": text},
            {"c1": [{"evidence_text": needle, "evidence_start": start, "evidence_end": end}]},
        )
        item = _item("「热情的村民」指代谁？", "繁藏", "c1",
                     "热情的村民：哟，这不是小缠丸嘛")
        row = _judge_one(item, corpus)
        assert row["verdict"] == "needs_extra", (
            f"答案那一行不含别名，挪过去会丢证据；实际 {row['verdict']}，"
            f"理由：{row['requirements'][0]['reason']}"
        )
        supporters = row["requirements"][0]["supporters_other_line"]
        assert supporters, "至少要点名是哪条断言承载了答案"
        assert supporters[0]["line_has_alias"] == [], (
            "这一态的前提就是「那一行不含别名」，若这里非空则上一条断言会自相矛盾"
        )

    def test_boolean_answers_are_not_judged_by_substring(self) -> None:
        """答案「否」不能因为断言里有「没有」就算被支持 → `boolean_answer`。

        真实例子：`q-na-000004` 的答案「否」匹配到断言「火龙并没有被源石感染」
        里的「没有」。这类题机器判据必然误命中，只能判无法判定。
        """
        text = "村民：火龙并没有被源石感染，你别怕。\n"
        needle = "火龙并没有被源石感染"
        start, end = _span(text, needle)
        corpus = _corpus(
            {"c1": text},
            {"c1": [{"evidence_text": needle, "evidence_start": start, "evidence_end": end}]},
        )
        item = _item("火龙被源石感染了吗？", "否", "c1", "火龙被源石感染了吗")
        row = _judge_one(item, corpus)
        assert row["verdict"] == "boolean_answer", (
            f"布尔答案只匹配到子串，必须判无法判定；实际 {row['verdict']}"
        )

    def test_no_assertion_carries_the_answer(self) -> None:
        """gold 块里的断言一条都不含答案 → `no_support`（抽取缺口，不是标注问题）。"""
        text = "村民：今天天气不错。\n"
        needle = "今天天气不错"
        start, end = _span(text, needle)
        corpus = _corpus(
            {"c1": text},
            {"c1": [{"evidence_text": needle, "evidence_start": start, "evidence_end": end}]},
        )
        item = _item("「老张」指代谁？", "张德海", "c1", "村民：今天天气不错")
        row = _judge_one(item, corpus)
        assert row["verdict"] == "no_support", (
            f"没有任何断言含答案时不能判成「可改标注」；实际 {row['verdict']}"
        )


# ---------------------------------------------------------------- 真实数据


class TestRealRun:
    """对仓库里的真实数据跑一遍，结论要与 Lead 复核过的数一致。"""

    @pytest.fixture(scope="class")
    @classmethod
    def payload(cls) -> dict:
        return json.loads(JSON_OUT.read_text("utf-8"))

    def test_committed_json_is_consistent_with_current_data(self) -> None:
        """已提交的 JSON 必须与当前数据一致——否则它描述的不是现在。"""
        from arkrag import subproc

        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--check"],
            capture_output=True, text=True, encoding=subproc.PIPE_ENCODING,
            errors="replace", cwd=str(ROOT), timeout=600,
            env=subproc.child_env(),
        )
        assert result.returncode == 0, (
            f"data/p1c_leak_audit.json 与当前数据不一致：{result.stdout}{result.stderr}"
            "；请重跑 py -3 scripts/p1c_leak_audit.py"
        )

    def test_59_candidates_have_zero_real_misses(self, payload: dict) -> None:
        """59 条候选的结论是 0 真漏标 / 58 不是漏标 / 1 无法判定。"""
        assert payload["n"] == 59
        assert payload["verdicts"].get("真漏标", 0) == 0
        assert payload["verdicts"].get("不是漏标") == 58
        assert payload["verdicts"].get("无法判定") == 1

    def test_a_class_27_splits_21_dev_6_holdout(self, payload: dict) -> None:
        """A 类 27 条的划分要钉住（Lead 用一个口径说过 5 条，完整口径是 6 条）。"""
        req = payload["a_class_requirements"]
        assert req["dev_n"] == 21
        assert req["holdout_n"] == 6
        assert req["dev_n"] + req["holdout_n"] == 27

    def test_dev_side_has_zero_move_safe(self, payload: dict) -> None:
        """dev 侧 `move_safe` 必须是 0——没有一条能靠只改 required 指向救回来。

        这一条是整个 P1-c 剩余部分的核心结论。如果它变成非 0，说明判定逻辑变了，
        必须重看那几条是不是真的「那一行也含别名」。
        """
        dev = payload["a_class_requirements"]["by_split"]["dev"]
        assert dev.get("move_safe", 0) == 0, (
            f"dev 侧出现了可安全改指向的题：{dev}；"
            "若真有，要逐条核「答案那一行是否真的含别名」，不能直接改 gold"
        )
        assert dev == {"needs_extra": 13, "carrier_ok": 6, "boolean_answer": 2}

    def test_holdout_is_recorded_but_not_to_be_changed(self, payload: dict) -> None:
        """holdout 侧只记录不改：表里给出判定，但改动数必须是 0。"""
        req = payload["a_class_requirements"]
        assert req["by_split"]["holdout"] == {"carrier_ok": 3, "needs_extra": 3}
        assert payload["a_class_split"].get("q-al-000001") == "holdout"
        assert payload["a_class_split"].get("q-na-000006") == "holdout"

    def test_every_row_carries_a_criterion_and_inputs(self, payload: dict) -> None:
        """第三方复核的最低要求：每条判定都带判据与输入，不许只有结论。"""
        for row in payload["rows"]:
            assert row.get("criterion"), f"{row['question_id']} 缺判据"
            assert row.get("inputs"), f"{row['question_id']} 缺输入"
        for row in payload["a_class_requirements"]["rows"]:
            assert row["requirements"], f"{row['question_id']} 缺 required 行明细"
            for req in row["requirements"]:
                assert req["reason"], f"{row['question_id']} 的 required 行缺理由"

    def test_chunks_come_from_the_generation_not_the_corpus_root(self, payload: dict) -> None:
        """块正文必须来自代内 `chunks.jsonl`（13,798 块），不是语料根（13,786 块）。

        这个坑本项目踩过三次。`resolve_chunks()` 是唯一正解，脚本里必须写明用的
        是它——写 `config.chunks_path()` 会在语料根重建后静默换源。
        """
        assert payload["chunks_source_api"] == "pipeline.resolve_chunks(gen, manifest)"
        assert payload["chunks_loaded"] == 13798
        assert payload["chunks_source"].endswith("chunks.jsonl")
        assert "generations" in payload["chunks_source"]

    def test_document_exists_and_declares_the_verdicts(self) -> None:
        """判定表必须真的写了那四态——文档缺一节，读的人就只能看 JSON。"""
        text = DOC.read_text("utf-8")
        for marker in ("carrier_ok", "move_safe", "needs_extra", "boolean_answer"):
            assert marker in text, f"判定表里没有 {marker}"
        assert "## 十、" in text, "缺第十节（A 类 27 条的 required 判定）"
        assert "## 十一、" in text, "缺第十一节（本表自己的变更日志）"

    def test_change_log_declares_zero_gold_changes_and_two_impact_counts(self) -> None:
        """变更日志必须写「影响题数（dev / holdout 分开）」且审核人空缺即未改。

        这张表被改过三轮、每轮结论都变过。不记账的话，下一个人分不清哪一版是真的；
        而「改了 0 题」这句话必须和「审核人尚未签字」写在一起——两件事互为证据。
        """
        text = DOC.read_text("utf-8")
        assert "dev 影响 0 题" in text, "变更日志没写 dev 侧影响题数"
        assert "holdout 影响 0 题" in text, "变更日志没写 holdout 侧影响题数"
        assert "审核人空缺" in text, "变更日志没写「审核人空缺即未提交任何 gold 改动」"

    def test_gold_files_are_untouched(self) -> None:
        """三轮判定下来一字节 gold 都没动——这是本次交付的核心事实。"""
        import hashlib

        expect = {
            "gold-v1.json": ("76d66073ff87aab1", 284356),
            "gold-v2.json": ("069375f8e355e4b9", 601492),
        }
        for name, (sha16, size) in expect.items():
            path = ROOT / "data" / "goldset" / name
            raw = path.read_bytes()
            assert len(raw) == size, f"{name} 字节数变了：{len(raw)} ≠ {size}"
            assert hashlib.sha256(raw).hexdigest()[:16] == sha16, f"{name} 内容变了"
