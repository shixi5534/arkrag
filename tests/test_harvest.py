"""Part 回收的严格性测试（计划 B2 / F2「回收与缓存」组）。

每一条都对应旧 `ingest()` 会静默放过的一种坏情况。最核心的一条是
`test_conflicting_duplicate_rejected_not_last_wins`——旧代码这里是「最后一行覆盖」。
"""
from __future__ import annotations

import json

import pytest

from arkrag import agent_tasks, cache, config, contracts


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """把 inbox/outbox/ledger/quarantine/cache 全指向 tmp。"""
    inbox = tmp_path / "inbox"
    outbox = tmp_path / "outbox"
    ledger = tmp_path / "ledger"
    cache_dir = tmp_path / "cache"
    for d in (inbox, outbox, ledger, cache_dir):
        d.mkdir()
    monkeypatch.setattr(agent_tasks, "INBOX_DIR", inbox)
    monkeypatch.setattr(agent_tasks, "OUTBOX_DIR", outbox)
    monkeypatch.setattr(agent_tasks, "QUARANTINE_DIR", tmp_path / "quarantine")
    monkeypatch.setattr(config, "LEDGER_DIR", ledger)
    monkeypatch.setattr(config, "CACHE_DIR", cache_dir)
    return tmp_path


def _chunk(i: int, text: str | None = None) -> dict:
    import hashlib

    body = text or f"凯尔希：第{i}段台词，罗德岛不会改变航向。"
    return {
        "chunk_id": f"obt/main/test#{i:03d}",
        "content_key": hashlib.sha256(body.encode()).hexdigest()[:24],
        "scene_id": "obt/main/test",
        "doc_id": "obt/main/test",
        "source_fingerprint": "fp",
        "text": body,
    }


def _good_rec(chunk: dict, **over) -> dict:
    rec = {
        "chunk_id": chunk["chunk_id"],
        "content_key": chunk["content_key"],
        "scene_id": chunk["scene_id"],
        "entities": [{"name": "凯尔希", "type": "角色"}, {"name": "罗德岛", "type": "组织"}],
        "relations": [
            {
                "source": "凯尔希",
                "relation": "隶属",
                "target": "罗德岛",
                "evidence": "罗德岛不会改变航向",
                "confidence": 0.9,
            }
        ],
    }
    rec.update(over)
    return rec


def _write_outbox(tmp_path, label: str, recs: list) -> None:
    outbox = tmp_path / "outbox"
    with (outbox / f"{label}.jsonl").open("w", encoding="utf-8") as fh:
        for r in recs:
            fh.write((r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)) + "\n")


class TestHappyPath:
    def test_complete_part_is_ingested(self, workdir):
        chunks = [_chunk(i) for i in range(3)]
        _write_outbox(workdir, "T", [_good_rec(c) for c in chunks])
        receipt, code = agent_tasks.ingest_strict("T", chunks)
        assert code == 0
        assert receipt["written"] == 3
        assert receipt["missing"] == 0
        assert receipt["conflicts"] == 0
        assert receipt["complete"] is True
        for c in chunks:
            assert cache.read_for_chunk(c, agent_tasks.AGENT_MODEL) is not None

    def test_second_run_is_idempotent(self, workdir):
        chunks = [_chunk(i) for i in range(2)]
        _write_outbox(workdir, "T", [_good_rec(c) for c in chunks])
        agent_tasks.ingest_strict("T", chunks)
        before = {c["chunk_id"]: cache.read_for_chunk(c, agent_tasks.AGENT_MODEL) for c in chunks}
        # 清空 outbox 后重跑：应从缓存命中，不再是 written
        (workdir / "outbox" / "T.jsonl").unlink(missing_ok=True)
        receipt, code = agent_tasks.ingest_strict("T", chunks)
        assert code == 0
        assert receipt["written"] == 0
        for c in chunks:
            after = cache.read_for_chunk(c, agent_tasks.AGENT_MODEL)
            assert after == before[c["chunk_id"]]


class TestConflictingDuplicates:
    def test_conflicting_duplicate_rejected_not_last_wins(self, workdir):
        """旧代码：got[cid] = rec 直接覆盖，只 duplicate_lines+=1。这是核心缺陷。"""
        c = _chunk(0)
        rec_a = _good_rec(c, entities=[{"name": "凯尔希", "type": "角色"}], relations=[])
        rec_b = _good_rec(c, entities=[{"name": "阿米娅", "type": "角色"}], relations=[])
        _write_outbox(workdir, "T", [rec_a, rec_b])
        receipt, code = agent_tasks.ingest_strict("T", [c])

        assert code != 0, "冲突重复行竟然返回成功"
        assert receipt["conflicts"] == 1
        assert receipt["written"] == 0, "冲突块不该入库任一版本"
        assert cache.read_for_chunk(c, agent_tasks.AGENT_MODEL) is None
        assert receipt["complete"] is False

    def test_identical_duplicate_lines_dedupe_and_succeed(self, workdir):
        c = _chunk(0)
        rec = _good_rec(c)
        _write_outbox(workdir, "T", [rec, rec])
        receipt, code = agent_tasks.ingest_strict("T", [c])
        assert code == 0, f"完全相同的重复行不该判失败: {receipt}"
        assert receipt["deduped"] == 1
        assert receipt["written"] == 1
        assert receipt["conflicts"] == 0

    def test_conflict_is_reported_not_retried_with_last(self, workdir):
        """即使冲突的两行里有一行是好的，也不许挑一个用。"""
        c = _chunk(0)
        good = _good_rec(c)
        bad = _good_rec(c, entities=[], relations=[])
        _write_outbox(workdir, "T", [good, bad])
        receipt, _ = agent_tasks.ingest_strict("T", [c])
        assert receipt["written"] == 0
        assert cache.read_for_chunk(c, agent_tasks.AGENT_MODEL) is None


class TestBadRecords:
    @pytest.mark.parametrize(
        "rec,reason",
        [
            ({"content_key": "x", "entities": [], "relations": []}, "missing_chunk_id"),
            ({"chunk_id": "", "entities": [], "relations": []}, "missing_chunk_id"),
            ({"chunk_id": None, "entities": [], "relations": []}, "missing_chunk_id"),
        ],
    )
    def test_missing_chunk_id_quarantined(self, workdir, rec, reason):
        c = _chunk(0)
        _write_outbox(workdir, "T", [rec])
        receipt, code = agent_tasks.ingest_strict("T", [c])
        assert code != 0
        reasons = {q["reason"] for q in receipt["quarantine"]}
        assert reason in reasons
        # 关键：不许出现 "None" 这种伪 chunk_id 被当成真实块
        assert "None" not in receipt["conflict_ids"]

    def test_non_json_line_quarantined(self, workdir):
        c = _chunk(0)
        _write_outbox(workdir, "T", ["{ not json at all", json.dumps(_good_rec(c))])
        receipt, code = agent_tasks.ingest_strict("T", [c])
        assert code != 0
        reasons = {q["reason"] for q in receipt["quarantine"]}
        assert "line_not_json" in reasons
        # 同一文件里的好行仍应入库
        assert receipt["written"] == 1

    def test_non_object_record_quarantined(self, workdir):
        c = _chunk(0)
        _write_outbox(workdir, "T", [[1, 2, 3]])
        receipt, code = agent_tasks.ingest_strict("T", [c])
        assert code != 0
        assert "not_object" in {q["reason"] for q in receipt["quarantine"]}

    def test_unknown_chunk_id_quarantined(self, workdir):
        c = _chunk(0)
        stray = _good_rec(_chunk(99))
        _write_outbox(workdir, "T", [json.dumps(_good_rec(c), ensure_ascii=False), json.dumps(stray, ensure_ascii=False)])
        receipt, code = agent_tasks.ingest_strict("T", [c])
        assert code != 0
        assert "unknown_chunk_id" in {q["reason"] for q in receipt["quarantine"]}

    def test_contract_violation_quarantined(self, workdir):
        """空 entities 必须走契约拒绝，而不是被 `or []` 洗成空数组存下来。"""
        c = _chunk(0)
        _write_outbox(workdir, "T", [_good_rec(c, entities=[])])
        receipt, code = agent_tasks.ingest_strict("T", [c])
        assert code != 0
        reasons = {q["reason"] for q in receipt["quarantine"]}
        assert "contract_violation" in reasons
        assert cache.read_for_chunk(c, agent_tasks.AGENT_MODEL) is None

    def test_entities_string_quarantined(self, workdir):
        c = _chunk(0)
        _write_outbox(workdir, "T", [_good_rec(c, entities="")])
        receipt, code = agent_tasks.ingest_strict("T", [c])
        assert code != 0
        assert cache.read_for_chunk(c, agent_tasks.AGENT_MODEL) is None

    def test_evidence_not_in_chunk_quarantined(self, workdir):
        c = _chunk(0)
        rec = _good_rec(c)
        rec["relations"][0]["evidence"] = "完全不在这段正文里的句子"
        _write_outbox(workdir, "T", [rec])
        receipt, code = agent_tasks.ingest_strict("T", [c])
        assert code != 0
        assert cache.read_for_chunk(c, agent_tasks.AGENT_MODEL) is None

    def test_quarantine_file_written(self, workdir):
        c = _chunk(0)
        _write_outbox(workdir, "T", ["broken"])
        agent_tasks.ingest_strict("T", [c])
        qpath = workdir / "quarantine" / "T.quarantine.json"
        assert qpath.exists()
        payload = json.loads(qpath.read_text("utf-8"))
        assert payload["count"] >= 1


class TestMissingBlocks:
    def test_missing_block_makes_part_incomplete(self, workdir):
        chunks = [_chunk(0), _chunk(1)]
        _write_outbox(workdir, "T", [_good_rec(chunks[0])])
        receipt, code = agent_tasks.ingest_strict("T", chunks)
        assert code != 0, "缺块竟然返回成功"
        assert receipt["missing"] == 1
        assert receipt["complete"] is False
        assert chunks[1]["chunk_id"] in receipt["missing_ids"]

    def test_cached_block_is_not_missing(self, workdir):
        chunks = [_chunk(0), _chunk(1)]
        _write_outbox(workdir, "T", [_good_rec(chunks[0])])
        agent_tasks.ingest_strict("T", chunks)
        # 第二次只补第二块
        (workdir / "outbox" / "T.jsonl").unlink(missing_ok=True)
        _write_outbox(workdir, "T", [_good_rec(chunks[1])])
        receipt, code = agent_tasks.ingest_strict("T", chunks)
        assert code == 0, receipt
        assert receipt["missing"] == 0


class TestReceiptShape:
    def test_receipt_splits_seven_counts(self, workdir):
        c = _chunk(0)
        _write_outbox(workdir, "T", [_good_rec(c)])
        receipt, _ = agent_tasks.ingest_strict("T", [c])
        for key in (
            "lines_read",
            "accepted",
            "deduped",
            "conflicts",
            "quarantined",
            "missing",
            "written",
        ):
            assert key in receipt, f"收据缺口径 {key}"

    def test_receipt_carries_identity_versions(self, workdir):
        c = _chunk(0)
        _write_outbox(workdir, "T", [_good_rec(c)])
        receipt, _ = agent_tasks.ingest_strict("T", [c])
        assert receipt["prompt_sha256"] == cache.prompt_sha256()
        assert receipt["schema_version"] == config.SCHEMA_VERSION
        assert receipt["normalizer_version"] == config.NORMALIZER_VERSION
