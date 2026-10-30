"""C5 回归：有界待处理队列、熔断后停止投喂、并发峰值记录。

旧实现：

    futures = {pool.submit(extract_chunk, client, c) for c in chunks}

一次性把全部 Chunk 提交进线程池。计划 C5 要求「不一次性提交全部 Future，用有界
待处理队列，熔断后停止补充新任务，记录真实并发峰值」。这三条各自都是可失败的。
"""
from __future__ import annotations

import hashlib
import threading
import time

import pytest

from arkrag import cache, config, contracts, extract, llm


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """把缓存、收据、熔断全部指到临时目录。"""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "LEDGER_DIR", tmp_path / "ledger")
    monkeypatch.setattr(config, "CIRCUIT_OPEN_PATH", tmp_path / "circuit_open.json")
    (tmp_path / "cache").mkdir()
    (tmp_path / "ledger").mkdir()
    return tmp_path


def _chunk(i: int) -> dict:
    text = f"凯尔希：第{i}块正文。阿米娅：收到。"
    return {
        "chunk_id": f"doc# {i}".replace(" ", ""),
        "content_key": hashlib.sha256(text.encode("utf-8")).hexdigest()[:24],
        "scene_id": f"doc#{i}",
        "doc_id": f"doc{i}",
        "category": "obt/main",
        "text": text,
        "title": "测试",
        "chapter": "第一章",
        "speakers": ["凯尔希", "阿米娅"],
    }


class SlowClient:
    """记录同时有多少个请求在飞；可按剧本失败。"""

    def __init__(self, *, delay=0.02, fail_after=None, fail_kind=llm.AUTH):
        self.model = "fake-model"
        self.gauge = llm.InflightGauge()
        self.delay = delay
        self.fail_after = fail_after
        self.fail_kind = fail_kind
        self.calls = 0
        self.lock = threading.Lock()
        self.live = 0
        self.peak = 0

    def complete_json(self, system, user):  # noqa: ARG002
        if llm.circuit_state() is not None:
            raise llm.CircuitOpen("circuit_open")
        with self.lock:
            self.calls += 1
            n = self.calls
            self.live += 1
            self.peak = max(self.peak, self.live)
        try:
            time.sleep(self.delay)
            if self.fail_after is not None and n > self.fail_after:
                return llm.Outcome(
                    None, f"{self.fail_kind}:x", True, "", 0, 0, kind=self.fail_kind
                )
            # 抽一条真能在正文里定位的证据
            text = user.split("---", 1)[-1]
            ev = "凯尔希：" if "凯尔希：" in text else text.strip()[:6]
            return llm.Outcome(
                {
                    "entities": [{"name": "凯尔希", "type": "角色"}],
                    "relations": [
                        {
                            "source": "凯尔希",
                            "relation": "交谈",
                            "target": "阿米娅",
                            "evidence": ev,
                            "confidence": 0.9,
                        }
                    ],
                },
                "",
                False,
                "stop",
                10,
                5,
            )
        finally:
            with self.lock:
                self.live -= 1


class TestBoundedQueue:
    def test_window_limits_inflight(self, sandbox):
        """待处理队列不得把全部 Chunk 一起塞进去。"""
        chunks = [_chunk(i) for i in range(30)]
        client = SlowClient(delay=0.01)
        receipt = extract.run_shard(chunks, concurrency=3, label="t", window=6, client=client)
        assert receipt["window_setting"] == 6
        # 同时运行的不超过 min(window, 并发) —— 关键是远小于 30
        assert client.peak <= 6, f"在途峰值 {client.peak} 超过窗口 6"
        assert receipt["requested"] == 30
        assert receipt["extracted_ok"] == 30

    def test_window_never_below_concurrency(self, sandbox):
        chunks = [_chunk(i) for i in range(10)]
        client = SlowClient(delay=0.005)
        receipt = extract.run_shard(chunks, concurrency=8, label="t2", window=2, client=client)
        assert receipt["window_setting"] == 8

    def test_all_chunks_are_processed(self, sandbox):
        chunks = [_chunk(i) for i in range(12)]
        client = SlowClient(delay=0.005)
        receipt = extract.run_shard(chunks, concurrency=2, label="t3", window=4, client=client)
        assert receipt["extracted_ok"] + receipt["cache_hit"] == 12
        assert receipt["skipped_circuit"] == 0


class TestCircuitStopsFeeding:
    def test_circuit_skips_remaining_not_fails_them(self, sandbox):
        """熔断后剩余 Chunk 记为 skipped，不冒充「跑失败」。"""
        chunks = [_chunk(i) for i in range(40)]
        client = SlowClient(delay=0.002, fail_after=2, fail_kind=llm.AUTH)
        receipt = extract.run_shard(chunks, concurrency=1, label="t4", window=2, client=client)
        assert receipt["circuit_stopped"] >= 1
        assert receipt["skipped_circuit"] > 0
        # 熔断后不再有新请求打出去
        assert client.calls <= 4, f"熔断后仍在发请求：calls={client.calls}"
        # 收据必须区分「没跑」与「跑失败」
        assert receipt["skipped_circuit"] + receipt["failed"] + receipt["extracted_ok"] <= 40
        assert len(receipt["skipped_chunk_ids"]) == receipt["skipped_circuit"]

    def test_skipped_receipt_is_written(self, sandbox):
        chunks = [_chunk(i) for i in range(10)]
        client = SlowClient(delay=0.002, fail_after=0, fail_kind=llm.AUTH)
        extract.run_shard(chunks, concurrency=1, label="t5", window=1, client=client)
        receipt_file = config.LEDGER_DIR / "t5.json"
        assert receipt_file.exists()
        import json

        data = json.loads(receipt_file.read_text("utf-8"))
        assert "skipped_circuit" in data
        assert "peak_inflight" in data
        assert "window_setting" in data


class TestContractIntegration:
    def test_bad_model_output_is_rejected_by_contract(self, sandbox):
        """模型吐出 entities="" 时不得写缓存。"""

        class BadClient(SlowClient):
            def complete_json(self, system, user):  # noqa: ARG002
                with self.lock:
                    self.calls += 1
                return llm.Outcome(
                    {"entities": "", "relations": []}, "", False, "stop", 1, 1
                )

        chunks = [_chunk(0)]
        client = BadClient()
        receipt = extract.run_shard(chunks, concurrency=1, label="t6", window=1, client=client)
        assert receipt["failed"] == 1
        assert receipt["contract_rejects"].get(contracts.Code.WRONG_TYPE) == 1
        assert not any(config.CACHE_DIR.rglob("*.json"))

    def test_evidence_not_in_chunk_is_rejected(self, sandbox):
        class LiarClient(SlowClient):
            def complete_json(self, system, user):  # noqa: ARG002
                with self.lock:
                    self.calls += 1
                return llm.Outcome(
                    {
                        "entities": [{"name": "凯尔希", "type": "角色"}],
                        "relations": [
                            {
                                "source": "凯尔希",
                                "relation": "交谈",
                                "target": "阿米娅",
                                "evidence": "这句正文根本不存在",
                                "confidence": 0.9,
                            }
                        ],
                    },
                    "",
                    False,
                    "stop",
                    1,
                    1,
                )

        chunks = [_chunk(0)]
        client = LiarClient()
        receipt = extract.run_shard(chunks, concurrency=1, label="t7", window=1, client=client)
        assert receipt["failed"] == 1
        assert receipt["contract_rejects"].get(contracts.Code.EVIDENCE_NOT_FOUND) == 1
        assert not any(config.CACHE_DIR.rglob("*.json"))

    def test_successful_extraction_writes_identity_bound_cache(self, sandbox):
        chunks = [_chunk(0)]
        client = SlowClient(delay=0)
        receipt = extract.run_shard(chunks, concurrency=1, label="t8", window=1, client=client)
        assert receipt["extracted_ok"] == 1
        files = list(config.CACHE_DIR.rglob("*.json"))
        assert len(files) == 1
        import json

        rec = json.loads(files[0].read_text("utf-8"))
        assert rec["key_scheme"] == 2
        assert rec["chunk_id"] == chunks[0]["chunk_id"]
        assert rec["content_key"] == chunks[0]["content_key"]
        assert rec["prompt_sha256"] == cache.prompt_sha256()

    def test_second_run_hits_cache_with_zero_calls(self, sandbox):
        chunks = [_chunk(0)]
        first = SlowClient(delay=0)
        extract.run_shard(chunks, concurrency=1, label="t9a", window=1, client=first)
        second = SlowClient(delay=0)
        receipt = extract.run_shard(chunks, concurrency=1, label="t9b", window=1, client=second)
        assert receipt["cache_hit"] == 1
        assert second.calls == 0
        assert receipt["cache_hit_layers"]["current"] == 1
