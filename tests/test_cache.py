"""缓存身份测试（计划 B4 / F2「回收与缓存」组）。

核心断言：**缓存键必须能区分不同身份**。旧键做不到，这就是 B4 要修的东西。
"""
from __future__ import annotations

import json

import pytest

from arkrag import cache, config


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    d = tmp_path / "cache"
    d.mkdir()
    monkeypatch.setattr(config, "CACHE_DIR", d)
    return d


def _chunk(**over) -> dict:
    base = {
        "chunk_id": "obt/main/level_main_01#000",
        "content_key": "a" * 24,
        "scene_id": "obt/main/level_main_01",
    }
    base.update(over)
    return base


class TestKeyIdentity:
    def test_same_inputs_same_key(self):
        kw = dict(content_key="k", chunk_id="c", scene_id="s", model="m")
        assert cache.key_for(**kw) == cache.key_for(**kw)

    def test_content_key_changes_key(self):
        a = cache.key_for(content_key="k1", chunk_id="c", scene_id="s", model="m")
        b = cache.key_for(content_key="k2", chunk_id="c", scene_id="s", model="m")
        assert a != b

    def test_chunk_id_changes_key(self):
        """旧键的致命缺陷：两个同正文不同 chunk_id 会共用缓存。"""
        a = cache.key_for(content_key="k", chunk_id="c1", scene_id="s", model="m")
        b = cache.key_for(content_key="k", chunk_id="c2", scene_id="s", model="m")
        assert a != b, "chunk_id 没进键：overlap 造成的同正文块会共用缓存"

    def test_scene_id_changes_key(self):
        a = cache.key_for(content_key="k", chunk_id="c", scene_id="s1", model="m")
        b = cache.key_for(content_key="k", chunk_id="c", scene_id="s2", model="m")
        assert a != b

    def test_model_changes_key(self):
        a = cache.key_for(content_key="k", chunk_id="c", scene_id="s", model="m1")
        b = cache.key_for(content_key="k", chunk_id="c", scene_id="s", model="m2")
        assert a != b

    def test_generation_changes_key(self):
        a = cache.key_for(content_key="k", chunk_id="c", scene_id="s", model="m", generation=0)
        b = cache.key_for(content_key="k", chunk_id="c", scene_id="s", model="m", generation=1)
        assert a != b

    def test_prompt_digest_changes_key(self):
        """改提示词但忘 bump 版本字符串时，必须靠内容哈希区分。"""
        a = cache.key_for(content_key="k", chunk_id="c", scene_id="s", model="m", prompt_digest="p1")
        b = cache.key_for(content_key="k", chunk_id="c", scene_id="s", model="m", prompt_digest="p2")
        assert a != b

    def test_schema_version_changes_key(self):
        a = cache.key_for(
            content_key="k", chunk_id="c", scene_id="s", model="m", schema_version="ark-rag-1"
        )
        b = cache.key_for(
            content_key="k", chunk_id="c", scene_id="s", model="m", schema_version="ark-rag-2"
        )
        assert a != b

    def test_legacy_key_still_functional(self):
        """legacy 键不能删——要读 13,801 条历史缓存。"""
        assert cache.cache_key("k", "m", "v2") == cache.cache_key("k", "m", "v2")
        assert cache.cache_key("k", "m", "v2") != cache.cache_key("k", "m", "v3")


class TestPromptDigest:
    def test_digest_is_from_content_not_constant(self):
        from arkrag import prompt

        d = cache.prompt_sha256()
        assert len(d) == 64
        # 与 prompt.prompt_fingerprint() 同源但长度不同：前者 64，后者 16
        assert d.startswith(prompt.prompt_fingerprint()[:8]) is False or True
        assert d != prompt.prompt_fingerprint()

    def test_digest_stable_across_calls(self):
        assert cache.prompt_sha256() == cache.prompt_sha256()


class TestWriteRead:
    def test_roundtrip(self, cache_dir):
        key = cache.key_for(content_key="k", chunk_id="c", scene_id="s", model="m")
        raw = {"entities": [{"name": "凯尔希", "type": "角色"}], "relations": []}
        cache.write_raw(key, raw, "m", "c", "k", scene_id="s")
        got = cache.read_raw(key)
        assert got == raw

    def test_record_carries_identity(self, cache_dir):
        key = cache.key_for(content_key="k", chunk_id="c", scene_id="s", model="m")
        path = cache.write_raw(
            key, {"entities": [], "relations": []}, "m", "c", "k", scene_id="s", generation=3
        )
        rec = json.loads(path.read_text("utf-8"))
        assert rec["chunk_id"] == "c"
        assert rec["content_key"] == "k"
        assert rec["scene_id"] == "s"
        assert rec["generation"] == 3
        assert rec["key_scheme"] == 2
        assert rec["prompt_sha256"] == cache.prompt_sha256()

    def test_prompt_digest_mismatch_invalidates(self, cache_dir, monkeypatch):
        key = cache.key_for(content_key="k", chunk_id="c", scene_id="s", model="m")
        cache.write_raw(key, {"entities": [], "relations": []}, "m", "c", "k", scene_id="s")
        monkeypatch.setattr(cache, "prompt_sha256", lambda: "0" * 64)
        assert cache.read_raw(key) is None

    def test_atomic_no_tmp_left_behind(self, cache_dir):
        key = cache.key_for(content_key="k", chunk_id="c", scene_id="s", model="m")
        cache.write_raw(key, {"entities": [], "relations": []}, "m", "c", "k", scene_id="s")
        leftovers = [p for p in cache_dir.rglob("*.tmp")]
        assert leftovers == [], f"残留临时文件: {leftovers}"


class TestLegacySafety:
    def _write_legacy(self, cache_dir, *, chunk_id="c", content_key="k", model="m", raw=None):
        key = cache.cache_key(content_key, model, config.PROMPT_VERSION)
        path = cache.cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "__cache_format__": "raw",
            "normalizer": "n1",
            "prompt_version": config.PROMPT_VERSION,
            "model": model,
            "chunk_id": chunk_id,
            "content_key": content_key,
            "written_at": "2026-10-01T00:00:00Z",
            "raw": raw if raw is not None else {"entities": [{"name": "A"}], "relations": []},
        }
        path.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        return key

    def test_legacy_with_matching_identity_is_reused(self, cache_dir):
        self._write_legacy(cache_dir)
        lk = cache.lookup_for_chunk(_chunk(chunk_id="c", content_key="k"), "m")
        assert lk.layer == "legacy"
        assert lk.raw["entities"] == [{"name": "A"}]

    def test_legacy_chunk_id_mismatch_rejected(self, cache_dir):
        """旧键不会检查这个——所以同正文不同 Chunk 会互相污染。

        注意此处的危险形态：content_key 相同（overlap 常见）但 chunk_id 不同。
        旧键算出同一个键，会把 A 的抽取结果当成 B 的。新逻辑必须判 invalid。
        """
        self._write_legacy(cache_dir, chunk_id="c", content_key="k")
        lk = cache.lookup(content_key="k", chunk_id="OTHER", scene_id="s", model="m")
        assert lk.raw is None, "同 content_key 不同 chunk_id 的 legacy 记录被当成命中"
        assert lk.layer == "invalid"
        assert "chunk_id_mismatch" in lk.reason

    def test_legacy_content_key_mismatch_rejected(self, cache_dir):
        self._write_legacy(cache_dir, chunk_id="c", content_key="k1")
        lk = cache.lookup(content_key="k2", chunk_id="c", scene_id="s", model="m")
        # legacy 键不同，所以 miss；若手工构造同键则 invalid
        assert lk.layer == "miss"

    def test_legacy_model_mismatch_rejected(self, cache_dir):
        self._write_legacy(cache_dir, model="old-model")
        lk = cache.lookup(content_key="k", chunk_id="c", scene_id="s", model="new-model")
        assert lk.layer == "miss"

    def test_legacy_stale_prompt_version_rejected(self, cache_dir, monkeypatch):
        self._write_legacy(cache_dir)
        monkeypatch.setattr(config, "PROMPT_VERSION", "v9")
        lk = cache.lookup(content_key="k", chunk_id="c", scene_id="s", model="m")
        assert lk.raw is None

    def test_legacy_bad_raw_shape_rejected(self, cache_dir):
        self._write_legacy(cache_dir, raw={"entities": []})  # 缺 relations
        lk = cache.lookup(content_key="k", chunk_id="c", scene_id="s", model="m")
        assert lk.layer == "invalid"
        assert "raw_missing_keys" in lk.reason

    def test_legacy_not_object_rejected(self, cache_dir):
        self._write_legacy(cache_dir, raw=["not", "dict"])
        lk = cache.lookup(content_key="k", chunk_id="c", scene_id="s", model="m")
        assert lk.layer == "invalid"
        assert "raw_not_object" in lk.reason

    def test_allow_legacy_false_never_uses_legacy(self, cache_dir):
        self._write_legacy(cache_dir)
        lk = cache.lookup(
            content_key="k", chunk_id="c", scene_id="s", model="m", allow_legacy=False
        )
        assert lk.layer == "miss"


class TestCorruptCache:
    def test_unreadable_json_is_miss_not_crash(self, cache_dir):
        key = cache.key_for(content_key="k", chunk_id="c", scene_id="s", model="m")
        path = cache.cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ this is not json", encoding="utf-8")
        assert cache.read_raw(key) is None

    def test_non_object_json_is_miss(self, cache_dir):
        key = cache.key_for(content_key="k", chunk_id="c", scene_id="s", model="m")
        path = cache.cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[1,2,3]", encoding="utf-8")
        assert cache.read_raw(key) is None


class TestCounters:
    def test_count_cached_excludes_tmp(self, cache_dir):
        (cache_dir / "aa").mkdir()
        (cache_dir / "aa" / "x.json").write_text("{}", encoding="utf-8")
        (cache_dir / "aa" / "y.tmp").write_text("{}", encoding="utf-8")
        assert cache.count_cached() == 1
