"""契约校验的回归测试（计划 B1 / F2「回收与缓存」组）。

每一条都对应计划里点名必须拒收的一种坏数据。故意用 `parametrize` 逐项独立，
失败时能一眼看出是哪一种坏数据被放行了——而不是"整条 validate 挂了"。
"""
from __future__ import annotations

import copy

import pytest

from arkrag import contracts
from arkrag.contracts import Code


def _rec(task, **over) -> dict:
    rec = {
        "schema_version": contracts.EXTRACTION_SCHEMA,
        "chunk_id": task["chunk_id"],
        "content_key": task["content_key"],
        "scene_id": task["scene_id"],
        "entities": [{"name": "凯尔希", "type": "角色"}, {"name": "阿米娅", "type": "角色"}],
        "relations": [
            {
                "source": "凯尔希",
                "relation": "隶属",
                "target": "罗德岛",
                "evidence": "罗德岛不会为一个人的牺牲改变航向",
                "confidence": 0.9,
            }
        ],
    }
    rec.update(over)
    return rec


def _codes(errors) -> set[str]:
    return {e["code"] for e in errors if e["level"] == "error"}


# ------------------------------------------------------------------ task 契约

class TestValidateTask:
    def test_valid_task_passes(self, sample_task):
        cleaned, errors = contracts.validate_task(sample_task)
        assert errors == []
        assert cleaned["chunk_id"] == sample_task["chunk_id"]

    def test_non_object_rejected(self):
        cleaned, errors = contracts.validate_task(["not", "a", "dict"])
        assert cleaned is None
        assert Code.NOT_OBJECT in _codes(errors)

    @pytest.mark.parametrize("bad", ["ark-rag-1", "ark-rag-task-1", None, "", 2])
    def test_unknown_schema_rejected(self, sample_task, bad):
        task = copy.deepcopy(sample_task)
        task["schema_version"] = bad
        cleaned, errors = contracts.validate_task(task)
        assert cleaned is None
        assert Code.UNKNOWN_SCHEMA in _codes(errors)

    @pytest.mark.parametrize(
        "field",
        ["run_id", "chunk_id", "content_key", "scene_id", "source_fingerprint", "prompt_sha256", "model"],
    )
    def test_missing_identity_field_rejected(self, sample_task, field):
        task = copy.deepcopy(sample_task)
        del task[field]
        cleaned, errors = contracts.validate_task(task)
        assert cleaned is None, f"缺 {field} 竟然通过"
        assert Code.MISSING_FIELD in _codes(errors)

    def test_missing_text_rejected(self, sample_task):
        task = copy.deepcopy(sample_task)
        del task["text"]
        cleaned, errors = contracts.validate_task(task)
        assert cleaned is None
        assert any(e["field"] == "text" for e in errors)

    def test_missing_generation_rejected(self, sample_task):
        task = copy.deepcopy(sample_task)
        del task["generation"]
        cleaned, errors = contracts.validate_task(task)
        assert cleaned is None
        assert any(e["field"] == "generation" for e in errors)

    @pytest.mark.parametrize("bad", [-1, "0", 1.5, True])
    def test_bad_generation_rejected(self, sample_task, bad):
        task = copy.deepcopy(sample_task)
        task["generation"] = bad
        cleaned, errors = contracts.validate_task(task)
        assert cleaned is None
        assert Code.WRONG_TYPE in _codes(errors)


# ------------------------------------------------------------ extraction 契约

class TestValidateExtraction:
    def test_valid_extraction_passes_and_locates_evidence(self, sample_task, sample_chunk_text):
        cleaned, errors = contracts.validate_extraction(_rec(sample_task), sample_task, sample_chunk_text)
        assert [e for e in errors if e["level"] == "error"] == []
        rel = cleaned["relations"][0]
        assert rel["evidence_start"] >= 0
        assert sample_chunk_text[rel["evidence_start"]:rel["evidence_end"]] == rel["evidence"]

    def test_non_object_rejected(self, sample_task, sample_chunk_text):
        cleaned, errors = contracts.validate_extraction("nope", sample_task, sample_chunk_text)
        assert cleaned is None
        assert Code.NOT_OBJECT in _codes(errors)

    @pytest.mark.parametrize("bad", ["ark-rag-1", "ark-rag-extraction-1", None])
    def test_unknown_schema_rejected(self, sample_task, sample_chunk_text, bad):
        cleaned, errors = contracts.validate_extraction(
            _rec(sample_task, schema_version=bad), sample_task, sample_chunk_text
        )
        assert cleaned is None
        assert Code.UNKNOWN_SCHEMA in _codes(errors)

    def test_missing_entities_field_rejected(self, sample_task, sample_chunk_text):
        rec = _rec(sample_task)
        del rec["entities"]
        cleaned, errors = contracts.validate_extraction(rec, sample_task, sample_chunk_text)
        assert cleaned is None
        assert Code.MISSING_FIELD in _codes(errors)

    @pytest.mark.parametrize("bad", ["", False, None, 0, {}, "凯尔希"])
    def test_entities_falsy_shapes_rejected_not_silently_empty(self, sample_task, sample_chunk_text, bad):
        """现有代码 `rec.get("entities") or []` 会把这些洗成 []，必须改为拒收。"""
        cleaned, errors = contracts.validate_extraction(
            _rec(sample_task, entities=bad), sample_task, sample_chunk_text
        )
        assert cleaned is None, f"entities={bad!r} 被静默当成空数组放行了"
        assert _codes(errors), "至少应有一条 error"

    def test_empty_entities_list_rejected(self, sample_task, sample_chunk_text):
        cleaned, errors = contracts.validate_extraction(
            _rec(sample_task, entities=[]), sample_task, sample_chunk_text
        )
        assert cleaned is None
        assert Code.EMPTY_COLLECTION in _codes(errors)

    def test_missing_chunk_id_rejected(self, sample_task, sample_chunk_text):
        rec = _rec(sample_task)
        del rec["chunk_id"]
        cleaned, errors = contracts.validate_extraction(rec, sample_task, sample_chunk_text)
        assert cleaned is None
        assert Code.MISSING_FIELD in _codes(errors)

    def test_chunk_id_mismatch_rejected(self, sample_task, sample_chunk_text):
        rec = _rec(sample_task, chunk_id="obt/main/other#000")
        cleaned, errors = contracts.validate_extraction(rec, sample_task, sample_chunk_text)
        assert cleaned is None
        assert Code.CHUNK_ID_MISMATCH in _codes(errors)

    def test_content_key_mismatch_rejected(self, sample_task, sample_chunk_text):
        rec = _rec(sample_task, content_key="0" * 24)
        cleaned, errors = contracts.validate_extraction(rec, sample_task, sample_chunk_text)
        assert cleaned is None
        assert Code.CONTENT_KEY_MISMATCH in _codes(errors)

    def test_scene_id_mismatch_rejected(self, sample_task, sample_chunk_text):
        rec = _rec(sample_task, scene_id="obt/main/other")
        cleaned, errors = contracts.validate_extraction(rec, sample_task, sample_chunk_text)
        assert cleaned is None
        assert Code.SCENE_ID_MISMATCH in _codes(errors)

    def test_duplicate_entity_id_rejected(self, sample_task, sample_chunk_text):
        rec = _rec(
            sample_task,
            entities=[
                {"name": "凯尔希", "type": "角色", "id": "e1"},
                {"name": "阿米娅", "type": "角色", "id": "e1"},
            ],
        )
        cleaned, errors = contracts.validate_extraction(rec, sample_task, sample_chunk_text)
        assert cleaned is None
        assert Code.DUPLICATE_ENTITY_ID in _codes(errors)

    def test_duplicate_entity_name_dedupes_with_warning(self, sample_task, sample_chunk_text):
        rec = _rec(
            sample_task,
            entities=[
                {"name": "凯尔希", "type": "角色"},
                {"name": "凯尔希", "type": "角色"},
                {"name": "阿米娅", "type": "角色"},
            ],
            relations=[],
        )
        cleaned, errors = contracts.validate_extraction(rec, sample_task, sample_chunk_text)
        assert [e for e in errors if e["level"] == "error"] == []
        assert len(cleaned["entities"]) == 2
        assert Code.DUPLICATE_ENTITY_NAME in {e["code"] for e in errors if e["level"] == "warn"}

    @pytest.mark.parametrize("bad", [1.5, -0.1, 100, "0.9", True])
    def test_bad_confidence_rejected(self, sample_task, sample_chunk_text, bad):
        rec = _rec(sample_task)
        rec["relations"][0]["confidence"] = bad
        cleaned, errors = contracts.validate_extraction(rec, sample_task, sample_chunk_text)
        assert cleaned is None, f"置信度 {bad!r} 被放行"
        assert Code.BAD_CONFIDENCE in _codes(errors)

    def test_evidence_not_in_chunk_rejected(self, sample_task, sample_chunk_text):
        rec = _rec(sample_task)
        rec["relations"][0]["evidence"] = "这句话根本不在正文里"
        cleaned, errors = contracts.validate_extraction(rec, sample_task, sample_chunk_text)
        assert cleaned is None
        assert Code.EVIDENCE_NOT_FOUND in _codes(errors)

    def test_evidence_from_other_chunk_rejected(self, sample_task, sample_chunk_text):
        """计划 E1：不允许用其他 Chunk 的证据替代。"""
        rec = _rec(sample_task)
        rec["relations"][0]["evidence"] = "切尔诺伯格已经不存在了"  # 确实在本 Chunk
        cleaned, errors = contracts.validate_extraction(rec, sample_task, sample_chunk_text)
        assert cleaned is not None
        other_text = "完全无关的另一段正文。"
        cleaned2, errors2 = contracts.validate_extraction(rec, sample_task, other_text)
        assert cleaned2 is None
        assert Code.EVIDENCE_NOT_FOUND in _codes(errors2)

    def test_evidence_too_long_rejected(self, sample_task, sample_chunk_text):
        rec = _rec(sample_task)
        long_text = "凯尔希" * 40
        text = sample_chunk_text + long_text
        task = dict(sample_task, text=text)
        rec["relations"][0]["evidence"] = long_text
        cleaned, errors = contracts.validate_extraction(rec, task, text)
        assert cleaned is None
        assert Code.EVIDENCE_TOO_LONG in _codes(errors)

    def test_self_loop_rejected(self, sample_task, sample_chunk_text):
        rec = _rec(sample_task)
        rec["relations"][0]["source"] = "凯尔希"
        rec["relations"][0]["target"] = "凯尔希"
        cleaned, errors = contracts.validate_extraction(rec, sample_task, sample_chunk_text)
        assert cleaned is None
        assert Code.RELATION_SELF_LOOP in _codes(errors)

    def test_empty_relation_field_rejected(self, sample_task, sample_chunk_text):
        rec = _rec(sample_task)
        rec["relations"][0]["relation"] = ""
        cleaned, errors = contracts.validate_extraction(rec, sample_task, sample_chunk_text)
        assert cleaned is None
        assert Code.MISSING_FIELD in _codes(errors)

    def test_duplicate_relation_dedupes_with_warning(self, sample_task, sample_chunk_text):
        rel = {
            "source": "凯尔希",
            "relation": "隶属",
            "target": "罗德岛",
            "evidence": "罗德岛不会为一个人的牺牲改变航向",
        }
        rec = _rec(sample_task, relations=[dict(rel), dict(rel)])
        cleaned, errors = contracts.validate_extraction(rec, sample_task, sample_chunk_text)
        assert [e for e in errors if e["level"] == "error"] == []
        assert len(cleaned["relations"]) == 1
        assert Code.DUPLICATE_RELATION in {e["code"] for e in errors if e["level"] == "warn"}

    def test_unknown_predicate_is_warning_not_rejection(self, sample_task, sample_chunk_text):
        """关系名不在词表由 L4 归一化处理，不应在 L3 就整条拒收。"""
        rec = _rec(sample_task)
        rec["relations"][0]["relation"] = "某个不在词表的关系"
        cleaned, errors = contracts.validate_extraction(
            rec, sample_task, sample_chunk_text, known_predicates={"隶属", "敌对"}
        )
        assert cleaned is not None
        assert Code.RELATION_UNKNOWN_PREDICATE in {e["code"] for e in errors if e["level"] == "warn"}

    def test_relations_may_be_empty_array(self, sample_task, sample_chunk_text):
        """纯列实体不建边合法（但 entities 必须有）。"""
        cleaned, errors = contracts.validate_extraction(
            _rec(sample_task, relations=[]), sample_task, sample_chunk_text
        )
        assert [e for e in errors if e["level"] == "error"] == []
        assert cleaned["relations"] == []


class TestSummarize:
    def test_summary_counts_only_errors(self, sample_task, sample_chunk_text):
        rec = _rec(sample_task, content_key="0" * 24)
        rec["relations"][0]["confidence"] = 99
        _, errors = contracts.validate_extraction(rec, sample_task, sample_chunk_text)
        s = contracts.summarize_errors(errors)
        assert s["error_count"] == 2
        assert set(s["by_code"]) == {Code.CONTENT_KEY_MISMATCH, Code.BAD_CONFIDENCE}
