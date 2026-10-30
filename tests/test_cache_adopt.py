"""`cache.adopt_legacy()`：带来源的可证明复用。

为什么要有这一层：D 阶段修好旧分块器的尾块长度测量 bug 后，块身份变了，原先
「旧键命中即复用」的路径对这些块一律 miss。能证明安全的那些块不该被迫重抽，
但也不能悄悄当成「本轮抽的」——所以写新键、记来源、单独报一层。

本文件的断言分两类，缺一不可：

* **正向**：四项身份（content_key / scene_id / model / prompt 版本）全等、且 raw
  过得了严格契约时，真的搬出一条新键记录，并把 `adopted_from` / `adopt_reason`
  / `adopted_at` 落在盘上。
* **负向**：任一项不等、或证据在当前正文里定位不到，都必须在**写盘之前**拒掉。
  负向断言同时检查磁盘——只有返回值是 invalid 但文件还是写了，等于没拒。
"""
from __future__ import annotations

import hashlib
import json

import pytest

from arkrag import cache, config, contracts

MODEL = "agent:qoder-subagent"
TEXT = "凯尔希：切尔诺伯格已经沦陷了。\n阿米娅：我们必须去罗德岛。"
RAW = {
    "entities": [
        {"name": "凯尔希", "type": "角色", "aliases": [], "desc": "罗德岛医疗部门负责人"},
        {"name": "阿米娅", "type": "角色", "aliases": [], "desc": "罗德岛领袖"},
    ],
    "relations": [
        {
            "source": "凯尔希",
            "relation": "信赖",
            "target": "阿米娅",
            "evidence": "我们必须去罗德岛。",
            "confidence": 0.9,
        }
    ],
}


@pytest.fixture()
def cache_dir(tmp_path, monkeypatch):
    d = tmp_path / "cache"
    d.mkdir()
    monkeypatch.setattr(config, "CACHE_DIR", d)
    return d


def _chunk(**over) -> dict:
    base = {
        "chunk_id": "obt/main/level_main_01#000",
        "content_key": hashlib.sha256(TEXT.encode("utf-8")).hexdigest()[:24],
        "scene_id": "obt/main/level_main_01",
        "doc_id": "obt/main/level_main_01",
        "text": TEXT,
    }
    base.update(over)
    return base


def _write_legacy(*, chunk_id="obt/main/level_main_01#000", content_key=None, model=MODEL,
                  prompt_version=None, raw=None, fmt="raw") -> dict:
    """写一条**纯 legacy 形态**的记录：和仓库里 13,801 条一样，没有 prompt_sha256。"""
    ck = content_key if content_key is not None else _chunk()["content_key"]
    rec = {
        "__cache_format__": fmt,
        "normalizer": "n1",
        "prompt_version": prompt_version or config.PROMPT_VERSION,
        "model": model,
        "chunk_id": chunk_id,
        "content_key": ck,
        "written_at": "2026-10-28T17:14:16Z",
        "raw": RAW if raw is None else raw,
    }
    key = cache.cache_key(ck, model, config.PROMPT_VERSION)
    path = cache.cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    return rec


def _json_files(root):
    return sorted(p for p in root.rglob("*.json") if not p.name.endswith(".tmp"))


class TestAdoptHappyPath:
    def test_writes_new_key_record_with_provenance(self, cache_dir):
        src = _write_legacy()
        before = _json_files(cache_dir)

        lk = cache.adopt_legacy(src, _chunk(), MODEL)

        assert lk.layer == "adopted"
        assert lk.adopted_from == cache.cache_key(
            _chunk()["content_key"], MODEL, config.PROMPT_VERSION
        )
        new_key = cache.key_for(
            content_key=_chunk()["content_key"],
            chunk_id=_chunk()["chunk_id"],
            scene_id=_chunk()["scene_id"],
            model=MODEL,
        )
        assert lk.key == new_key

        after = _json_files(cache_dir)
        assert len(after) == len(before) + 1, "只应多出一条新键记录"
        rec = json.loads(cache.cache_path(new_key).read_text("utf-8"))
        assert rec["adopted_from"] == lk.adopted_from
        assert rec["adopt_reason"] == cache.ADOPT_REASON_IDENTICAL
        assert rec["adopted_at"].endswith("Z")
        # 新记录同时带齐本轮身份，不能只留来源字段
        assert rec["key_scheme"] == 2
        assert rec["chunk_id"] == _chunk()["chunk_id"]
        assert rec["content_key"] == _chunk()["content_key"]
        assert rec["scene_id"] == _chunk()["scene_id"]
        assert rec["prompt_sha256"] == cache.prompt_sha256()
        assert rec["raw"] == RAW

    def test_lookup_then_reports_adopted_not_current(self, cache_dir):
        src = _write_legacy()

        # 搬之前：lookup 走 legacy 层
        before = cache.lookup_for_chunk(_chunk(), MODEL)
        assert before.layer == "legacy"

        cache.adopt_legacy(src, _chunk(), MODEL)

        after = cache.lookup_for_chunk(_chunk(), MODEL)
        assert after.layer == "adopted", "搬来的记录不能混进 current 冒充本轮产出"
        assert after.raw == RAW
        assert after.adopted_from

    def test_adopt_is_idempotent(self, cache_dir):
        src = _write_legacy()
        first = cache.adopt_legacy(src, _chunk(), MODEL)
        assert first.layer == "adopted"
        n_after_first = len(_json_files(cache_dir))

        second = cache.adopt_legacy(src, _chunk(), MODEL)
        assert second.layer == "adopted"
        assert second.key == first.key
        assert len(_json_files(cache_dir)) == n_after_first, "重复调用不该再写一条"

    def test_does_not_touch_legacy_record(self, cache_dir):
        _write_legacy()
        legacy_key = cache.cache_key(_chunk()["content_key"], MODEL, config.PROMPT_VERSION)
        legacy_path = cache.cache_path(legacy_key)
        before = legacy_path.read_bytes()

        cache.adopt_legacy(cache._load(legacy_path), _chunk(), MODEL)

        assert legacy_path.read_bytes() == before, "来源记录必须只读"

    def test_generation_is_recorded_not_inherited(self, cache_dir):
        src = _write_legacy()
        lk = cache.adopt_legacy(src, _chunk(), MODEL, generation=3)
        assert lk.layer == "adopted"
        rec = json.loads(cache.cache_path(lk.key).read_text("utf-8"))
        assert rec["generation"] == 3


class TestAdoptRefuses:
    def _assert_refused(self, cache_dir, src, chunk, *, layer, needle):
        before = _json_files(cache_dir)
        lk = cache.adopt_legacy(src, chunk, MODEL)
        assert lk.layer == layer
        assert needle in lk.reason
        assert _json_files(cache_dir) == before, "被拒时不许留下任何新记录"
        return lk

    def test_missing_source_is_miss(self, cache_dir):
        self._assert_refused(cache_dir, None, _chunk(), layer="miss", needle="no_source")

    def test_content_key_mismatch(self, cache_dir):
        src = _write_legacy(content_key="f" * 24)
        self._assert_refused(
            cache_dir, src, _chunk(), layer="invalid", needle="content_key_mismatch"
        )

    def test_chunk_text_must_match_its_own_content_key(self, cache_dir):
        # 记录与块记录的 content_key 都写着同一个值，但正文换了 —— 重算立刻发现
        ck = _chunk()["content_key"]
        src = _write_legacy(content_key=ck)
        mutated = _chunk(text=TEXT + "\n追加了一行。")
        self._assert_refused(
            cache_dir, src, mutated, layer="invalid",
            needle="chunk_text_does_not_match_content_key",
        )

    def test_scene_id_mismatch(self, cache_dir):
        src = _write_legacy(chunk_id="obt/main/level_OTHER#000")
        self._assert_refused(cache_dir, src, _chunk(), layer="invalid", needle="scene_id_mismatch")

    def test_model_mismatch(self, cache_dir):
        src = _write_legacy(model="other:model")
        self._assert_refused(cache_dir, src, _chunk(), layer="invalid", needle="model_mismatch")

    def test_prompt_version_mismatch(self, cache_dir):
        src = _write_legacy(prompt_version="v1")
        self._assert_refused(
            cache_dir, src, _chunk(), layer="invalid", needle="prompt_version_mismatch"
        )

    def test_bad_format(self, cache_dir):
        src = _write_legacy(fmt="legacy-v0")
        self._assert_refused(cache_dir, src, _chunk(), layer="invalid", needle="bad_format")

    def test_raw_not_object(self, cache_dir):
        src = _write_legacy(raw=["not", "a", "dict"])
        self._assert_refused(cache_dir, src, _chunk(), layer="invalid", needle="raw_not_object")

    def test_evidence_not_located_in_this_block(self, cache_dir):
        # 这是本层最重的判据：证据必须真的落在**当前**正文里
        src = _write_legacy(
            raw={
                "entities": [
                    {"name": "凯尔希", "type": "角色"},
                    {"name": "阿米娅", "type": "角色"},
                ],
                "relations": [
                    {
                        "source": "凯尔希",
                        "relation": "信赖",
                        "target": "阿米娅",
                        "evidence": "这句原文不在本块里",
                        "confidence": 0.9,
                    }
                ],
            }
        )
        self._assert_refused(
            cache_dir, src, _chunk(), layer="invalid", needle="contract_rejected"
        )

    def test_relation_endpoint_outside_entities(self, cache_dir):
        # 契约的 _check_relations() 收了 entity_names 却没用它，所以 adopt 自己补
        # 这道端点闭合检查——否则边在 L4/L5 会因找不到端点被丢掉，复用等于白做。
        src = _write_legacy(
            raw={
                "entities": [{"name": "凯尔希", "type": "角色"}],
                "relations": [
                    {
                        "source": "凯尔希",
                        "relation": "信赖",
                        "target": "阿米娅",
                        "evidence": "我们必须去罗德岛。",
                        "confidence": 0.9,
                    }
                ],
            }
        )
        self._assert_refused(
            cache_dir, src, _chunk(), layer="invalid", needle="dangling_endpoint"
        )

    def test_empty_entities_rejected(self, cache_dir):
        src = _write_legacy(raw={"entities": [], "relations": []})
        self._assert_refused(
            cache_dir, src, _chunk(), layer="invalid", needle="contract_rejected"
        )


class TestAdoptBoundary:
    def test_scene_prefix_comes_from_record_chunk_id(self):
        assert cache._scene_of_chunk_id("obt/main/level_main_01#007") == "obt/main/level_main_01"
        assert cache._scene_of_chunk_id("no-hash-here") == ""

    def test_only_identical_content_reason_is_defined(self):
        assert cache.ADOPT_REASON_IDENTICAL == "identical_content"

    def test_contract_schema_is_the_strict_one(self):
        # 若契约升级而这里忘了跟着走，adopt 会把旧形状当合法输入
        assert contracts.EXTRACTION_SCHEMA == "ark-rag-extraction-2"


class TestGateReportsAdoptedSeparately:
    """门禁必须「看得见」复用：只申报不判定，但三层数字要分开。"""

    def _record(self, chunk_id, layer):
        return {
            "chunk_id": chunk_id,
            "text": TEXT,
            "raw": RAW,
            "layer": layer,
        }

    def _gate(self, records):
        from arkrag import gates

        run = gates.l3_trial_gates(records, expected_blocks=len(records))
        return next(g for g in run.gates if g.name.startswith("L3.T19"))

    def test_counts_are_split_not_merged(self):
        g = self._gate(
            [
                self._record("c#000", "current"),
                self._record("c#001", "legacy"),
                self._record("c#002", "adopted"),
                self._record("c#003", "adopted"),
            ]
        )
        assert "current 1" in g.measured
        assert "legacy 1" in g.measured
        assert "adopted 2" in g.measured

    def test_gate_only_reports_and_never_fails(self):
        # 复用占比该是多少取决于块身份变了多少，没有普适阈值 —— 不许拿它当门
        g = self._gate([self._record("c#000", "adopted") for _ in range(3)])
        assert g.verdict == "PASS"
        assert "不判定" in g.threshold

    def test_adopted_ids_are_named_in_note(self):
        g = self._gate([self._record("doc/a#007", "adopted")])
        assert "doc/a#007" in g.note

    def test_records_without_layer_are_reported_as_unlabeled(self):
        # 没带 layer 的调用方（如旧测试）不能被静默算成 current
        g = self._gate([{"chunk_id": "c#000", "text": TEXT, "raw": RAW}])
        assert "未标注 1" in g.measured
        assert "current 0" in g.measured

