"""C5 回归：错误分类、限流退避、熔断性质与过期。

这些用例直接钉住旧实现的三类错误：429 被当永久故障、断网被当永久故障、
临时熔断永不过期。每条都对应计划 C5 的一句要求。
"""
from __future__ import annotations

import json
import time
import urllib.error

import pytest

from arkrag import config, llm


@pytest.fixture()
def circuit_path(tmp_path, monkeypatch):
    """把熔断标记指到临时目录，避免污染 data/arkrag/circuit_open.json。"""
    p = tmp_path / "circuit_open.json"
    monkeypatch.setattr(config, "CIRCUIT_OPEN_PATH", p)
    return p


class TestClassifyHttp:
    @pytest.mark.parametrize("code", [401, 402, 403])
    def test_auth_codes(self, code):
        assert llm.classify_http(code, "") == llm.AUTH

    def test_429_is_rate_limit_not_auth(self):
        # 旧实现把 429 和 401 归成同一类 fatal，导致限流直接熔断整条流水线
        assert llm.classify_http(429, "") == llm.RATE_LIMIT

    @pytest.mark.parametrize("code", [500, 502, 503, 504, 599])
    def test_5xx_is_server(self, code):
        assert llm.classify_http(code, "") == llm.SERVER

    def test_404_with_model_marker_is_auth(self):
        assert llm.classify_http(404, '{"error":"invalid model"}') == llm.AUTH

    def test_404_without_marker_is_unknown(self):
        assert llm.classify_http(404, "nope") == llm.UNKNOWN

    def test_context_too_long_is_resource(self):
        assert llm.classify_http(400, "prompt is too long") == llm.RESOURCE

    def test_plain_400_is_content(self):
        assert llm.classify_http(400, "bad json body") == llm.CONTENT


class TestPolicy:
    def test_network_never_circuits(self):
        # 断网抖动不该停整条线
        assert llm.POLICY[llm.NETWORK]["circuit"] is None
        assert llm.POLICY[llm.NETWORK]["retry"] is True

    def test_rate_limit_circuit_is_transient(self):
        assert llm.POLICY[llm.RATE_LIMIT]["circuit"] == "transient"

    def test_server_circuit_is_transient(self):
        assert llm.POLICY[llm.SERVER]["circuit"] == "transient"

    def test_auth_does_not_retry(self):
        assert llm.POLICY[llm.AUTH]["retry"] is False
        assert llm.POLICY[llm.AUTH]["circuit"] == "permanent"

    def test_resource_does_not_retry(self):
        assert llm.POLICY[llm.RESOURCE]["retry"] is False

    def test_content_retries_without_circuit(self):
        assert llm.POLICY[llm.CONTENT]["retry"] is True
        assert llm.POLICY[llm.CONTENT]["circuit"] is None


class TestCircuits:
    def test_no_file_means_closed(self, circuit_path):
        assert llm.circuit_state() is None

    def test_permanent_circuit_stays(self, circuit_path):
        llm.open_circuit("auth", "401", kind="permanent")
        state = llm.circuit_state()
        assert state is not None
        assert state["kind"] == "permanent"
        # 永久熔断不该有到期时间（要求人介入）
        assert "expires_at" not in state

    def test_transient_circuit_expires(self, circuit_path):
        llm.open_circuit("rate_limit", "429", kind="transient", ttl_seconds=60)
        assert llm.circuit_state() is not None
        # 把「当前时间」推到到期之后，必须视为未开——旧实现会永久阻塞
        assert llm.circuit_state(now=time.time() + 61) is None

    def test_transient_before_expiry_still_open(self, circuit_path):
        llm.open_circuit("server", "503", kind="transient", ttl_seconds=60)
        assert llm.circuit_state(now=time.time() + 30) is not None

    def test_circuit_records_owner_and_generation(self, circuit_path, monkeypatch):
        monkeypatch.setenv("ARKRAG_RUN_OWNER", "run-42")
        monkeypatch.setenv("ARKRAG_GENERATION_ID", "gen-7")
        llm.open_circuit("auth", "401")
        state = json.loads(circuit_path.read_text("utf-8"))
        assert state["owner"] == "run-42"
        assert state["generation"] == "gen-7"
        assert state["schema"] == 3

    def test_unreadable_marker_counts_as_open(self, circuit_path):
        circuit_path.write_text("not json", encoding="utf-8")
        state = llm.circuit_state()
        assert state is not None
        assert state["reason"] == "unreadable_marker"

    def test_clear_circuit(self, circuit_path):
        llm.open_circuit("auth", "401")
        assert llm.clear_circuit() is True
        assert llm.circuit_state() is None
        assert llm.clear_circuit() is False

    def test_summary_reports_remaining_ttl(self, circuit_path):
        llm.open_circuit("rate_limit", "429", kind="transient", ttl_seconds=120)
        summary = llm.circuit_summary()
        assert summary["open"] is True
        assert summary["kind"] == "transient"
        assert 0 < summary["expires_in_seconds"] <= 120


class TestRetryAfter:
    @pytest.mark.parametrize(
        "raw,expected",
        [("5", 5.0), ("0", 0.0), ("2.5", 2.5)],
    )
    def test_seconds(self, raw, expected):
        assert llm._parse_retry_after(raw) == expected

    def test_none_and_garbage(self):
        assert llm._parse_retry_after(None) is None
        assert llm._parse_retry_after("soon") is None

    def test_http_date_is_parsed(self):
        import email.utils as eu

        future = eu.formatdate(time.time() + 30, usegmt=True)
        got = llm._parse_retry_after(future)
        assert got is not None
        assert 20 <= got <= 35


class TestBackoff:
    def test_rate_limit_honours_retry_after(self):
        delay = llm.backoff_delay(0, llm.RATE_LIMIT, 30.0)
        assert 30.0 <= delay <= 31.0

    def test_rate_limit_without_header_uses_exponential(self):
        delay = llm.backoff_delay(2, llm.RATE_LIMIT, None)
        assert 4.0 <= delay <= 5.0

    def test_exponential_is_capped(self):
        delay = llm.backoff_delay(10, llm.SERVER, None)
        assert delay <= 21.0


# ------------------------------------------------------------------ 假客户端

class FakeClient:
    """按剧本依次返回 Outcome，用于驱动 extract_with_retry。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.model = "fake-model"

    def complete_json(self, system, user):  # noqa: ARG002
        # 真实 LLMClient 在发请求前先看熔断标记；假客户端必须照样复现这一层，
        # 否则「熔断后不再发请求」这条根本没被测到。
        if llm.circuit_state() is not None:
            raise llm.CircuitOpen("circuit_open")
        self.calls += 1
        item = self.script.pop(0) if self.script else self.script_default()
        if isinstance(item, Exception):
            raise item
        return item

    @staticmethod
    def script_default():
        return llm.Outcome({"entities": [{"name": "凯尔希"}], "relations": []}, "", False, "", 1, 1)


def _fail(kind, error="x", retry_after=None):
    return llm.Outcome(
        None, error, kind in (llm.AUTH, llm.RESOURCE), "", 0, 0, kind=kind, retry_after=retry_after
    )


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """退避不该真的等——单测里把 sleep 换成记录器。"""
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)


class TestExtractWithRetry:
    def test_success_first_try(self, circuit_path):
        client = FakeClient([])
        raw, meta = llm.extract_with_retry(client, "s", "u")
        assert raw is not None
        assert meta["attempts"] == 1
        assert llm.circuit_state() is None

    def test_network_retries_but_never_circuits(self, circuit_path):
        client = FakeClient([_fail(llm.NETWORK), _fail(llm.NETWORK)])
        raw, meta = llm.extract_with_retry(client, "s", "u")
        assert raw is not None  # 第三次成功
        assert meta["attempts"] == 3
        assert meta["circuit_opened"] is False
        assert llm.circuit_state() is None  # 关键：断网不开熔断

    def test_network_exhausts_without_circuit(self, circuit_path, monkeypatch):
        monkeypatch.setattr(config, "LLM_RETRIES", 2)
        client = FakeClient([_fail(llm.NETWORK)] * 5)
        raw, meta = llm.extract_with_retry(client, "s", "u")
        assert raw is None
        assert meta["circuit_opened"] is False
        assert llm.circuit_state() is None

    def test_server_opens_transient_circuit_after_threshold(self, circuit_path, monkeypatch):
        monkeypatch.setattr(config, "LLM_RETRIES", 10)
        client = FakeClient([_fail(llm.SERVER)] * 10)
        raw, meta = llm.extract_with_retry(client, "s", "u")
        assert raw is None
        assert meta["circuit_opened"] is True
        assert meta["circuit_kind"] == "transient"
        assert client.calls == 4  # SERVER 阈值 4，达到即停
        state = llm.circuit_state()
        assert state["kind"] == "transient"
        assert "expires_at" in state

    def test_rate_limit_transient_after_threshold(self, circuit_path, monkeypatch):
        monkeypatch.setattr(config, "LLM_RETRIES", 10)
        client = FakeClient([_fail(llm.RATE_LIMIT, retry_after=2.0)] * 10)
        raw, meta = llm.extract_with_retry(client, "s", "u")
        assert raw is None
        assert meta["circuit_kind"] == "transient"
        assert client.calls == 3  # RATE_LIMIT 阈值 3
        assert meta["retry_after"] == 2.0

    def test_auth_fails_fast_with_permanent_circuit(self, circuit_path):
        client = FakeClient([_fail(llm.AUTH), _fail(llm.AUTH)])
        raw, meta = llm.extract_with_retry(client, "s", "u")
        assert raw is None
        assert client.calls == 1  # 不重试
        assert meta["fatal"] is True
        assert meta["circuit_kind"] == "permanent"
        assert llm.circuit_state()["kind"] == "permanent"

    def test_resource_fails_fast(self, circuit_path):
        client = FakeClient([_fail(llm.RESOURCE)])
        raw, meta = llm.extract_with_retry(client, "s", "u")
        assert raw is None
        assert client.calls == 1
        assert meta["circuit_kind"] == "permanent"

    def test_content_error_retries_without_circuit(self, circuit_path):
        client = FakeClient([_fail(llm.CONTENT), _fail(llm.CONTENT)])
        raw, meta = llm.extract_with_retry(client, "s", "u")
        assert raw is not None
        assert llm.circuit_state() is None

    def test_kinds_recorded_per_attempt(self, circuit_path):
        client = FakeClient([_fail(llm.NETWORK), _fail(llm.SERVER)])
        raw, meta = llm.extract_with_retry(client, "s", "u")
        assert raw is not None
        assert meta["kinds"] == [llm.NETWORK, llm.SERVER, "unknown"] or meta["kinds"][:2] == [
            llm.NETWORK,
            llm.SERVER,
        ]

    def test_open_circuit_short_circuits(self, circuit_path):
        llm.open_circuit("auth", "401", kind="permanent")
        client = FakeClient([])
        raw, meta = llm.extract_with_retry(client, "s", "u")
        assert raw is None
        assert meta["circuit_opened"] is True
        assert client.calls == 0

    def test_expired_transient_does_not_block(self, circuit_path):
        llm.open_circuit("server", "503", kind="transient", ttl_seconds=1)
        # 手工把到期时间拨到过去
        state = json.loads(circuit_path.read_text("utf-8"))
        state["expires_at"] = time.time() - 1
        circuit_path.write_text(json.dumps(state), encoding="utf-8")
        client = FakeClient([])
        raw, _meta = llm.extract_with_retry(client, "s", "u")
        assert raw is not None
        assert client.calls == 1
