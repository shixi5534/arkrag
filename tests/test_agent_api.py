"""A4：agent 接口层的离线测试 —— 能力表 / JSON-RPC / MCP 握手 / OpenAPI / 传输纯净性。

## 为什么全部离线

这一层要证明的是**协议与契约**，不是检索质量。检索质量有它自己的门
（`tests/test_server.py`、Gold Set 评测）。所以这里用假 `AppState`：
真的 `server.AppState` 子类，只把 `served()` 与 `readiness()` 换成内存里的
`_Served`，其余（`Settings`、`run_bounded`、`check_rate`、`sanitize_text`、
管理令牌判据）全部是真实现。

这样做的价值：任何一条"参数越界不夹紧""没配 token 就 503""错误消息过清洗"
的判定都仍然由**生产代码**执行——测试断言的是真实行为，不是替身的行为。

## 为什么这些用例各自对应一条真实失败形态

* `TestStdioPurity` —— stdout 里混进一行 `print()` 会让客户端解析失败，
  而客户端通常只报"JSON 解析失败"。这一条用**真子进程**跑 `serve_mcp.py`，
  断言 stdout 的每一行都是合法 JSON-RPC。
* `TestRegistryIsSingleSource` —— 三处（MCP tools/list、OpenAPI paths、
  JSON-RPC 方法名）必须从同一份 `TOOLS` 派生。写成三份的后果是漂移，
  且生效的是最松那份（本项目为此出过三次事故）。
* `TestAdminFailClosed` —— 没配 `ARKRAG_ADMIN_TOKEN` 时必须**拒绝**，
  而不是"没人认证就当管理员"。
* `TestErrorsNeverLeakTraceback` —— 内部异常只给类型名。
"""
from __future__ import annotations

import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
from io import StringIO
from pathlib import Path

import pytest

from arkrag import agent_api, config, mcp_server, release, server

ROOT = Path(__file__).resolve().parents[1]

GID = "g-20260101T000000Z-abcdef"


_COUNTER = iter(range(1, 100_000))


@pytest.fixture()
def tmp_path():
    """**覆盖** pytest 内建的 `tmp_path` 夹具，换成 `os.makedirs` 建的同款目录。

    ## 为什么必须覆盖

    pytest 的 `tmp_path` 走 `tempfile` 之外的一套逻辑：它用 `mkdir(mode=0o700)`
    建目录。在本仓库当前这套 Windows 沙箱里，这种模式位建出来的目录**列举会
    失败**（`PermissionError: [WinError 5]`），于是所有用 `tmp_path` 的用例都在
    setup 阶段就报错——`tests/test_server.py` 在同样环境下也一样跑不起来。

    那是环境问题，不是本文件的；但"验证接口层"不应该以"先修好机器"为前提。
    这里用 `os.makedirs`（默认模式位）建同语义的目录：可写、可清理、不碰真实
    `data/`，在正常环境下与 pytest 那版行为一致。模块级夹具同名会覆盖插件夹具，
    所以调用方写法不用改一个字。
    """

    root = Path(tempfile.gettempdir()) / f"arkrag-agentapi-{os.getpid()}-{next(_COUNTER)}"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root, exist_ok=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)

CHUNKS = {
    "doc/a#000": {
        "chunk_id": "doc/a#000",
        "doc_id": "doc/a",
        "category": "obt/main",
        "chapter": "第一章",
        "title": "切尔诺伯格",
        "scene_index": 0,
        "speakers": ["凯尔希", "阿米娅"],
        "chars": 60,
        "turn_ids": [0, 1],
        "text": "凯尔希：博士，罗德岛不会为一个人改变航向。\n阿米娅：可是我们还有别的办法。",
    }
}


def _answer() -> dict:
    """一条**引用偏移与块正文逐字对齐**的答案。

    不对齐的引用会被 `verify_citations` 剔除——那样测出来的"引用通过"是假通过，
    因为真正被验的是"剔除逻辑没崩"，而不是"引用成立"。
    """
    text = CHUNKS["doc/a#000"]["text"]
    evidence = "凯尔希：博士，罗德岛不会为一个人改变航向。"
    start = text.index(evidence)
    return {
        "answer": "凯尔希隶属罗德岛。",
        "citations": [
            {
                "chunk_id": "doc/a#000",
                "evidence": evidence,
                "start": start,
                "end": start + len(evidence),
            }
        ],
        "confidence": 0.9,
        "abstained": False,
        "abstain_reason": None,
        "warnings": [],
        "versions": {"prompt_version": config.PROMPT_VERSION, "answerer": "extractive"},
    }


class FakeRetriever:
    """只实现 `server._query` 会用到的四个属性。"""

    def __init__(self) -> None:
        self.budget = {"max_context_tokens": int(server.DEFAULT_MAX_CONTEXT_TOKENS)}
        self.query_context = None
        self.assertions_by_id = {
            "as-1": {
                "assertion_id": "as-1",
                "subject_id": "e-aaaa",
                "predicate": "隶属",
                "object_id": "e-bbbb",
                "direction": "forward",
                "chunk_id": "doc/a#000",
                "doc_id": "doc/a",
                "scene_id": "doc/a#0",
                "evidence_start": 0,
                "evidence_end": 19,
                "evidence_text": "凯尔希：博士，罗德岛不会为一个人改变航向。",
                "confidence": 0.9,
                "extraction": "rule",
            }
        }


def make_state(tmp_path, monkeypatch, **overrides) -> server.AppState:
    """造一个真 `AppState`，只把"盘上事实"换成内存事实。"""
    settings = server.Settings(
        host="127.0.0.1",
        port=0,
        rate_limit_calls=1000,
        # 审计/查询日志落临时目录：任何用例都不许碰真实 data/。
        audit_path=tmp_path / "audit" / "server-audit.jsonl",
        query_log_path=tmp_path / "logs" / "queries.jsonl",
        **overrides,
    )
    state = server.AppState(settings, generation_id=GID)

    loaded = {
        "generation_id": GID,
        "manifest": {"generation_id": GID},
        "graph": {
            "entities": [
                {
                    "id": "e-aaaa",
                    "name": "凯尔希",
                    "type": "角色",
                    "desc": "罗德岛的医疗顾问。",
                    "aliases": ["凯尔希医生"],
                    "pending_aliases": [],
                    "chunks": ["doc/a#000"],
                },
                {"id": "e-bbbb", "name": "罗德岛", "type": "组织", "desc": "", "chunks": []},
            ],
            "edges": [
                {
                    "edge_id": "ed-1",
                    "subject_id": "e-aaaa",
                    "object_id": "e-bbbb",
                    "predicate": "隶属",
                    "support_count": 3,
                }
            ],
            "alias_index": {"凯尔希医生": "e-aaaa"},
        },
        "adjacency": {},
        "assertions": [],
        "index": None,
        "index_meta": {},
        "chunks": CHUNKS,
    }
    served = server._Served(
        generation_id=GID,
        loaded=loaded,
        pipeline={
            "retriever": FakeRetriever(),
            "answerer": None,
            "top_k": server.DEFAULT_TOP_K,
            "graph_depth": server.DEFAULT_GRAPH_DEPTH,
            "vector_union": None,
            "generation_id": GID,
            "entities": loaded["graph"]["entities"],
        },
    )
    monkeypatch.setattr(state, "served", lambda: served)

    def _ask_once(pipeline, question, *, timings=None, options=None):
        if timings is not None:
            timings.update({"query_ms": 1.0, "search_ms": 2.0, "wall_ms": 3.0})
        return {
            "query_analysis": {"should_answer": True},
            "search": {
                "hits": [
                    {
                        "chunk_id": "doc/a#000",
                        "score": 4.5,
                        "sources": [{"channel": "bm25", "score": 1.0}],
                    }
                ],
                "stats": {"candidates": 1, "top_score": 4.5},
            },
            "expand": {"expansions": [], "stats": {"nodes_visited": 1}},
            "context": {
                "units": [{"chunk_id": "doc/a#000", "score": 1.0, "kind": "direct"}],
                "context_digest": "digest",
                "stats": {"tokens_used": 100},
            },
            "answer": _answer(),
        }

    monkeypatch.setattr(server.pipeline_mod, "ask_once", _ask_once)
    # 就绪自检要核 250 MB 产物的 sha256；这里给它一个确定的结论，
    # 让"自检失败项会被如实报出来"这件事可以被断言。
    monkeypatch.setattr(
        state,
        "readiness",
        lambda: {
            "ready": False,
            "generation_id": GID,
            "checked_at": "2026-11-12T00:00:00Z",
            "checks": [
                {"id": "manifest_complete", "ok": True, "blocking": True, "detail": "ok"},
                {"id": "data_readable", "ok": False, "blocking": True, "detail": "缺块正文"},
                {"id": "serving_is_current", "ok": False, "blocking": False, "detail": "非当前"},
            ],
        },
    )
    monkeypatch.setattr(release, "read_current", lambda: {"generation_id": GID})
    return state


@pytest.fixture()
def state(tmp_path, monkeypatch) -> server.AppState:
    return make_state(tmp_path, monkeypatch)


def _call(state, name, arguments=None):
    return agent_api.call_tool(state, name, arguments)


def _error_of(result: dict) -> dict:
    assert result["isError"] is True, f"应当失败，实际成功：{result}"
    return result["structuredContent"]["error"]


# ---------------------------------------------------------------- 能力表


class TestRegistryIsSingleSource:
    """三处消费者必须从同一份定义派生。"""

    def test_registry_is_self_consistent(self):
        assert agent_api.validate_registry() == []

    def test_every_tool_has_name_description_and_schemas(self):
        for name, spec in agent_api.TOOLS.items():
            assert spec.name == name
            assert spec.description_en.strip(), f"{name} 缺英文描述"
            assert spec.description_zh.strip(), f"{name} 缺中文描述"
            assert spec.input_schema.get("type") == "object", f"{name} 的 input_schema 不是 object"
            assert spec.output_schema.get("type") == "object", f"{name} 的 output_schema 不是 object"
            assert callable(spec.handler)
            # 描述必须同时含中英两种文字：只有中文的 schema 在英文客户端里
            # 等于没有描述，模型只能靠名字猜。
            assert spec.description == f"{spec.description_en}\n\n{spec.description_zh}"

    def test_required_tools_are_all_registered(self):
        """任务要求的七个只读能力 + 一个管理能力，一个都不能少。"""
        expected = {
            "arkrag_query",
            "arkrag_search",
            "arkrag_get_chunk",
            "arkrag_get_entity",
            "arkrag_get_assertion",
            "arkrag_release_info",
            "arkrag_health",
        }
        assert expected <= set(agent_api.TOOLS)
        assert agent_api.TOOLS["arkrag_admin_rollback"].mutates is True
        assert agent_api.TOOLS["arkrag_admin_rollback"].requires_admin is True

    def test_openapi_paths_match_tools_exactly(self):
        """OpenAPI 的 paths 与 TOOLS **一一对应**（多一个少一个都是漂移）。"""
        doc = agent_api.build_openapi()
        tool_paths = {p for p in doc["paths"] if p.startswith("/tools/")}
        assert tool_paths == {f"/tools/{name}" for name in agent_api.TOOLS}
        for name, spec in agent_api.TOOLS.items():
            op = doc["paths"][f"/tools/{name}"]["post"]
            assert op["operationId"] == name
            assert op["description"] == spec.description
            assert op["x-arkrag-mutates"] is spec.mutates
            assert op["x-arkrag-idempotent"] is spec.idempotent

    def test_openapi_is_serializable_and_31(self):
        doc = agent_api.build_openapi()
        assert doc["openapi"] == "3.1.0"
        text = json.dumps(doc, ensure_ascii=False)
        assert json.loads(text)["info"]["title"] == "ArkRAG Agent API"
        assert agent_api.openapi_json() == json.dumps(doc, ensure_ascii=False, indent=2)

    def test_openapi_component_names_are_referenced(self):
        """每个 `$ref` 都必须在 `components.schemas` 里存在。

        少了就是"文档里写着 schema，解析器一 deref 就炸"——那种文档比没有更难用。
        """
        doc = agent_api.build_openapi()
        text = json.dumps(doc)
        refs = set()
        cursor = 0
        marker = "#/components/schemas/"
        while True:
            idx = text.find(marker, cursor)
            if idx < 0:
                break
            end = text.find('"', idx + len(marker))
            refs.add(text[idx + len(marker):end])
            cursor = idx + len(marker)
        assert refs, "文档里一个 $ref 都没有，说明 schema 没被引用"
        missing = refs - set(doc["components"]["schemas"])
        assert not missing, f"这些 $ref 没有对应 schema：{sorted(missing)}"

    def test_admin_tool_declares_bearer_security(self):
        doc = agent_api.build_openapi()
        admin = doc["paths"]["/tools/arkrag_admin_rollback"]["post"]
        assert admin["security"] == [{"bearerAuth": []}]
        assert "bearerAuth" in doc["components"]["securitySchemes"]
        read_only = doc["paths"]["/tools/arkrag_query"]["post"]
        assert read_only["security"] == []

    def test_jsonrpc_method_names_come_from_tools(self):
        """原生 JSON-RPC 的方法名映射必须与 `TOOLS` 同键（不是另抄一份清单）。"""
        assert set(agent_api._RPC_METHODS) == set(agent_api.TOOLS)
        for name in agent_api.TOOLS:
            assert agent_api._RPC_METHODS[name] == name


# ---------------------------------------------------------------- MCP 握手


class TestMcpInitialize:
    def test_handshake_returns_version_and_capabilities(self, state):
        resp = agent_api.handle_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": agent_api.LATEST_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "0"},
                },
            },
            state,
        )
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 1
        result = resp["result"]
        assert result["protocolVersion"] == agent_api.LATEST_PROTOCOL_VERSION
        assert "tools" in result["capabilities"]
        assert result["capabilities"]["tools"]["listChanged"] is False
        assert result["serverInfo"]["name"] == agent_api.SERVER_NAME
        assert result["serverInfo"]["version"] == agent_api.SERVER_VERSION
        assert result["instructions"]

    def test_negotiates_after_client_version_when_supported(self, state):
        older = agent_api.SUPPORTED_PROTOCOL_VERSIONS[-1]
        resp = agent_api.handle_jsonrpc(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": older}},
            state,
        )
        assert resp["result"]["protocolVersion"] == older

    def test_unknown_client_version_falls_back_to_latest_supported(self, state):
        """对不上就回自己最新支持的那版，让客户端自己决定是否断开（规范要求）。"""
        resp = agent_api.handle_jsonrpc(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "1900-01-01"}},
            state,
        )
        assert resp["result"]["protocolVersion"] == agent_api.LATEST_PROTOCOL_VERSION
        assert resp["result"]["protocolVersion"] in agent_api.SUPPORTED_PROTOCOL_VERSIONS

    def test_initialized_notification_gets_no_response(self, state):
        assert agent_api.handle_jsonrpc({"jsonrpc": "2.0", "method": "notifications/initialized"}, state) is None

    def test_ping_returns_empty_result(self, state):
        resp = agent_api.handle_jsonrpc({"jsonrpc": "2.0", "id": "p", "method": "ping"}, state)
        assert resp["result"] == {}


class TestToolsList:
    def test_lists_every_registered_tool(self, state):
        resp = agent_api.handle_jsonrpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, state)
        tools = resp["result"]["tools"]
        assert [t["name"] for t in tools] == agent_api.tool_names()
        for entry in tools:
            assert entry["description"]
            assert entry["inputSchema"]["type"] == "object"
            assert entry["outputSchema"]["type"] == "object"
            assert set(entry["annotations"]) >= {"readOnlyHint", "destructiveHint", "idempotentHint"}

    def test_order_is_deterministic(self, state):
        """确定性顺序便于客户端缓存（规范 SHOULD）。两次调用必须完全一致。"""
        first = agent_api.handle_jsonrpc({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, state)
        second = agent_api.handle_jsonrpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, state)
        assert json.dumps(first["result"]) == json.dumps(second["result"])


class TestToolsCall:
    def test_query_happy_path(self, state):
        result = _call(state, "arkrag_query", {"question": "凯尔希属于哪个组织？"})
        assert result["isError"] is False
        body = result["structuredContent"]
        assert body["generation_id"] == GID
        assert body["answer"]["abstained"] is False
        assert body["answer"]["citations"], "答案必须带引用"
        assert body["answer"]["citations"][0]["chunk_id"] == "doc/a#000"
        # 引用必须真的能在这代块正文里定位（偏移对齐），否则校验会剔除它。
        cite = body["answer"]["citations"][0]
        text = CHUNKS["doc/a#000"]["text"]
        assert text[cite["start"]:cite["end"]] == cite["evidence"]
        assert body["untrusted_content"] is True
        # 结构化结果与文本内容同时给（新老客户端都能用）
        assert json.loads(result["content"][0]["text"])["query_id"] == body["query_id"]

    def test_query_defaults_come_from_schema(self, state):
        result = _call(state, "arkrag_query", {"question": "凯尔希"})
        limits = result["structuredContent"]["limits"]
        assert limits["top_k"] == agent_api.TOOLS["arkrag_query"].input_schema["properties"]["top_k"]["default"]
        assert limits["graph_depth"] == 1

    def test_search_returns_chunks_without_answer(self, state):
        result = _call(state, "arkrag_search", {"question": "凯尔希", "top_k": 5})
        body = result["structuredContent"]
        assert result["isError"] is False
        assert body["chunks"][0]["chunk_id"] == "doc/a#000"
        assert body["chunks"][0]["score"] == 4.5
        assert "answer" not in body, "search 不该生成答案"
        assert body["limits"]["top_k"] == 5

    def test_get_chunk_returns_text(self, state):
        result = _call(state, "arkrag_get_chunk", {"chunk_id": "doc/a#000"})
        body = result["structuredContent"]
        assert result["isError"] is False
        assert body["chunk_id"] == "doc/a#000"
        assert "罗德岛" in body["text"]
        assert body["untrusted"] is True

    def test_get_entity_accepts_alias(self, state):
        result = _call(state, "arkrag_get_entity", {"entity_id": "凯尔希医生"})
        body = result["structuredContent"]
        assert result["isError"] is False
        assert body["resolved_by"] == "alias"
        assert body["entity"]["name"] == "凯尔希"
        assert body["edges_total"] == 1

    def test_get_assertion_returns_provenance(self, state):
        result = _call(state, "arkrag_get_assertion", {"assertion_id": "as-1"})
        body = result["structuredContent"]
        assert result["isError"] is False
        assert body["chunk_id"] == "doc/a#000"
        assert body["evidence_start"] == 0

    def test_release_info_reports_self_check(self, state):
        result = _call(state, "arkrag_release_info", {})
        body = result["structuredContent"]
        assert result["isError"] is False
        assert body["served_generation_id"] == GID
        assert body["self_check"]["ready"] is False
        # 阻断项与非阻断项必须分开：把它们混在一个列表里，"是否可服务"就没法判断。
        assert [x["id"] for x in body["self_check"]["blocking_failures"]] == ["data_readable"]
        assert [x["id"] for x in body["self_check"]["non_blocking_failures"]] == ["serving_is_current"]

    def test_health_liveness_does_not_probe_readiness_by_default(self, state):
        body = _call(state, "arkrag_health", {})["structuredContent"]
        assert body["ok"] is True
        assert body["readiness_probed"] is False
        assert "readiness" not in body

    def test_health_with_readiness_reports_both_answers(self, state):
        """"活着"与"就绪"必须是两个答案，合并会让数据坏了报成健康。"""
        body = _call(state, "arkrag_health", {"include_readiness": True})["structuredContent"]
        assert body["ok"] is True
        assert body["ready"] is False
        assert body["readiness"]["ready"] is False


# ---------------------------------------------------------------- 错误码


class TestErrorCodes:
    def test_unknown_tool_is_invalid_params(self, state):
        resp = agent_api.handle_jsonrpc(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
             "params": {"name": "arkrag_nope", "arguments": {}}},
            state,
        )
        assert resp["error"]["code"] == agent_api.INVALID_PARAMS
        assert "Unknown tool" in resp["error"]["message"]

    def test_unknown_method_is_method_not_found(self, state):
        resp = agent_api.handle_jsonrpc({"jsonrpc": "2.0", "id": 1, "method": "no/such"}, state)
        assert resp["error"]["code"] == agent_api.METHOD_NOT_FOUND

    def test_missing_param_is_tool_error_not_protocol_error(self, state):
        """缺必填参数属于**工具执行错误**（SEP-1303）：模型要能看见原因并自我纠正。"""
        result = _call(state, "arkrag_query", {})
        err = _error_of(result)
        assert err["code"] == "invalid_arguments"
        assert "question" in err["message"]

    def test_out_of_range_param_is_not_clamped(self, state):
        err = _error_of(_call(state, "arkrag_query", {"question": "凯", "top_k": 999}))
        assert err["code"] == "invalid_arguments"
        assert "999" in err["message"]
        assert str(server.DEFAULT_MAX_TOP_K) in err["message"]

    def test_boolean_is_not_accepted_as_integer(self, state):
        """`True` 在 Python 里是 int：不显式拒绝就会被当成 1 悄悄接受。"""
        err = _error_of(_call(state, "arkrag_query", {"question": "凯", "top_k": True}))
        assert err["code"] == "invalid_arguments"

    def test_unknown_argument_is_rejected(self, state):
        err = _error_of(_call(state, "arkrag_get_chunk", {"chunk_id": "x", "extra": 1}))
        assert err["code"] == "invalid_arguments"
        assert "extra" in err["message"]

    def test_question_over_shared_limit_uses_one_constant(self, state):
        from arkrag import query as query_mod

        too_long = "凯" * (query_mod.QUERY_MAX_CHARS + 1)
        err = _error_of(_call(state, "arkrag_query", {"question": too_long}))
        assert err["code"] == "invalid_arguments"
        assert str(query_mod.QUERY_MAX_CHARS) in err["message"]

    def test_missing_chunk_is_execution_error_with_404(self, state):
        err = _error_of(_call(state, "arkrag_get_chunk", {"chunk_id": "doc/none#000"}))
        assert err["code"] == "chunk_not_found"
        assert err["http_status"] == 404

    def test_arguments_must_be_object(self, state):
        err = _error_of(_call(state, "arkrag_query", ["not", "an", "object"]))
        assert err["code"] == "bad_arguments"

    def test_invalid_jsonrpc_version_rejected(self, state):
        resp = agent_api.handle_jsonrpc({"jsonrpc": "1.0", "id": 1, "method": "ping"}, state)
        assert resp["error"]["code"] == agent_api.INVALID_REQUEST

    def test_filter_none_returns_invalid_request(self, state):
        assert agent_api.handle_jsonrpc(None, state)["error"]["code"] == agent_api.INVALID_REQUEST

    def test_non_object_request_rejected(self, state):
        assert agent_api.handle_jsonrpc("hello", state)["error"]["code"] == agent_api.INVALID_REQUEST

    def test_internal_exception_does_not_leak_traceback_or_path(self, state, monkeypatch):
        """内部异常只给类型名：`str(exc)` 常带本机绝对路径。"""

        def boom(*_a, **_k):
            raise RuntimeError(f"打不开 {config.ROOT}/data/arkrag/secret.json")

        monkeypatch.setattr(server.pipeline_mod, "ask_once", boom)
        err = _error_of(_call(state, "arkrag_query", {"question": "凯尔希"}))
        assert err["code"] == "internal"
        assert "RuntimeError" in err["message"]
        assert str(config.ROOT) not in json.dumps(err, ensure_ascii=False)
        assert "Traceback" not in json.dumps(err, ensure_ascii=False)


class TestBatchRequests:
    def test_batch_returns_one_response_per_request(self, state):
        payload = [
            {"jsonrpc": "2.0", "id": 1, "method": "ping"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "arkrag_health", "params": {"arguments": {}}},
        ]
        out = agent_api.handle_jsonrpc(payload, state)
        assert isinstance(out, list)
        assert [r["id"] for r in out] == [1, 2, 3]

    def test_batch_is_partitioned_not_all_or_nothing(self, state):
        """一条坏请求不能让同批的好请求一起失败（JSON-RPC 2.0 要求）。"""
        payload = [
            {"jsonrpc": "2.0", "id": 1, "method": "nope"},
            {"jsonrpc": "2.0", "id": 2, "method": "ping"},
            {"jsonrpc": "1.0", "id": 3, "method": "ping"},
        ]
        out = agent_api.handle_jsonrpc(payload, state)
        by_id = {r["id"]: r for r in out}
        assert by_id[1]["error"]["code"] == agent_api.METHOD_NOT_FOUND
        assert by_id[2]["result"] == {}
        assert by_id[3]["error"]["code"] == agent_api.INVALID_REQUEST

    def test_empty_batch_is_invalid_request(self, state):
        assert agent_api.handle_jsonrpc([], state)["error"]["code"] == agent_api.INVALID_REQUEST

    def test_notifications_are_dropped_from_batch(self, state):
        """全通知的批量请求必须回**空**——协议规定此时不回任何内容。"""
        out = agent_api.handle_jsonrpc(
            [{"jsonrpc": "2.0", "method": "notifications/initialized"}], state
        )
        assert out is None

    def test_native_method_name_dispatch(self, state):
        """原生 JSON-RPC 客户端可以直接拿工具名当方法名调。"""
        resp = agent_api.handle_jsonrpc(
            {"jsonrpc": "2.0", "id": 9, "method": "arkrag_get_chunk",
             "params": {"chunk_id": "doc/a#000"}},
            state,
        )
        assert "error" not in resp
        assert resp["result"]["chunk_id"] == "doc/a#000"

    def test_native_method_business_failure_uses_api_error_code(self, state):
        """业务失败用自定义错误码，让调用方区分"协议错"与"这次被拒"。"""
        resp = agent_api.handle_jsonrpc(
            {"jsonrpc": "2.0", "id": 9, "method": "arkrag_get_chunk", "params": {"chunk_id": ""}},
            state,
        )
        assert resp["error"]["code"] == agent_api.API_ERROR
        assert resp["error"]["data"]["code"] == "invalid_arguments"


# ---------------------------------------------------------------- 管理接口 fail-closed


class TestAdminFailClosed:
    def test_admin_tool_refused_without_token(self, state):
        """没配 `ARKRAG_ADMIN_TOKEN` 时必须**拒绝**，而不是"没人认证就当管理员"。"""
        assert not state.settings.admin_token
        err = _error_of(_call(state, "arkrag_admin_rollback", {"generation_id": GID}))
        assert err["code"] == "admin_disabled"
        assert err["http_status"] == 503

    def test_admin_tool_reaches_the_gate_when_token_configured(self, tmp_path, monkeypatch):
        """配了令牌也要走真实回滚判据，而不是在工具层另判一次。

        `release.rollback()` 会拒绝这个不存在的世代——错误码是 `rollback_refused`，
        不是 `admin_disabled`。这一条证明"工具层没有绕过管理门、也没有另写一套"。
        """
        state = make_state(tmp_path, monkeypatch, admin_token="s3cr3t")

        def refuse(_gid, **_k):
            raise release.ReleaseError("世代不存在")

        monkeypatch.setattr(release, "rollback", refuse)
        err = _error_of(_call(state, "arkrag_admin_rollback", {"generation_id": GID}))
        assert err["code"] == "rollback_refused"
        assert err["http_status"] == 409
        assert "s3cr3t" not in json.dumps(err, ensure_ascii=False)

    def test_admin_token_never_appears_in_tool_schema(self):
        """令牌不能作为工具参数：它会进模型上下文、调用日志与 tracing。"""
        for spec in agent_api.TOOLS.values():
            props = (spec.input_schema.get("properties") or {})
            assert "token" not in props, f"{spec.name} 把令牌做成了参数"
            assert "admin_token" not in props


# ---------------------------------------------------------------- stdio 传输纯净性


class TestStdioPurity:
    """stdout 只能出现协议消息——混进一行文本就会让客户端解析失败。"""

    def _transport(self, state):
        return mcp_server.StdioTransport(state, reader=StringIO(""), writer=StringIO())

    def test_handle_line_returns_only_protocol_messages(self, state):
        transport = self._transport(state)
        out = transport.handle_line(
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        )
        assert len(out) == 1
        assert out[0]["jsonrpc"] == "2.0"
        # 必须能压成单行（协议：消息内不得含换行）
        line = mcp_server.encode_message(out[0])
        assert "\n" not in line
        assert json.loads(line)["id"] == 1

    def test_notification_line_produces_nothing(self, state):
        transport = self._transport(state)
        assert transport.handle_line('{"jsonrpc":"2.0","method":"notifications/initialized"}') == []

    def test_bad_json_becomes_parse_error_not_exception(self, state):
        transport = self._transport(state)
        out = transport.handle_line("not json at all")
        assert len(out) == 1
        assert out[0]["error"]["code"] == agent_api.PARSE_ERROR
        assert out[0]["id"] is None

    def test_blank_lines_are_ignored(self, state):
        transport = self._transport(state)
        assert transport.handle_line("   \n") == []

    def test_oversized_line_is_refused(self, state):
        transport = self._transport(state)
        huge = '{"jsonrpc":"2.0","id":1,"method":"ping","params":{"x":"' + "啊" * 200000 + '"}}'
        out = transport.handle_line(huge)
        assert out[0]["error"]["code"] == agent_api.PARSE_ERROR

    def test_serve_forever_writes_only_parseable_lines(self, state):
        """整轮读循环跑一遍，逐行断言 stdout 内容。**

        这是"不污染 stdout"的**直接**证据：不是看代码里有没有 print，
        而是看实际写出去的东西长什么样。
        """
        script = "\n".join(
            [
                json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
                json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
                json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
                json.dumps({"jsonrpc": "2.0", "id": 3, "method": "ping"}),
                "",
            ]
        )
        writer = StringIO()
        transport = mcp_server.StdioTransport(state, reader=StringIO(script), writer=writer)
        assert transport.serve_forever() == 0
        written = [line for line in writer.getvalue().splitlines() if line]
        assert len(written) == 3, f"应当只有 3 条响应，实际 {written}"
        for line in written:
            parsed = json.loads(line)  # 解析失败即证明 stdout 被污染
            assert parsed["jsonrpc"] == "2.0"
            assert set(parsed) <= {"jsonrpc", "id", "result", "error"}

    def test_real_subprocess_stdout_is_clean(self):
        """跑**真实** `scripts/serve_mcp.py`：入口脚本本身也不许污染 stdout。

        这一条覆盖"模块里干净、脚本里 print 了一行横幅"这种最常见的形态。
        """
        script = ROOT / "scripts" / "serve_mcp.py"
        assert script.exists()
        messages = "\n".join(
            [
                json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                            "params": {"protocolVersion": agent_api.LATEST_PROTOCOL_VERSION}}),
                json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
                "",
            ]
        )
        proc = subprocess.run(
            [sys.executable, str(script)],
            input=messages,
            capture_output=True,
            text=True,
            encoding="utf-8",
            # 子进程的 stderr 在 Windows 控制台代码页下可能是 GBK（中文字节），
            # 严格 utf-8 解码会让 `subprocess` 的读取线程抛异常、`proc.stdout` 变空，
            # 从而把一个"stdout 干净"的断言变成"读不到 stdout"的假失败。
            errors="replace",
            cwd=str(ROOT),
            timeout=120,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        assert proc.returncode == 0, f"退出码 {proc.returncode}，stderr：{proc.stderr[-2000:]}"
        lines = [x for x in proc.stdout.splitlines() if x.strip()]
        assert len(lines) == 2, f"stdout 行数不对：{proc.stdout!r}"
        for line in lines:
            parsed = json.loads(line)
            assert parsed["jsonrpc"] == "2.0"
        assert json.loads(lines[0])["result"]["serverInfo"]["name"] == agent_api.SERVER_NAME

    def test_writer_stream_is_isolated_from_real_stdout(self, state, capsys):
        """注入 writer 时，真 stdout 必须一个字都没收到。"""
        writer = StringIO()
        transport = mcp_server.StdioTransport(
            state,
            reader=StringIO(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}) + "\n"),
            writer=writer,
        )
        transport.serve_forever()
        captured = capsys.readouterr()
        assert captured.out == "", f"真 stdout 被写入了：{captured.out!r}"
        assert json.loads(writer.getvalue())["result"] == {}


# ---------------------------------------------------------------- HTTP 传输


class _Live:
    def __init__(self, httpd) -> None:
        import threading

        self.httpd = httpd
        self.thread = threading.Thread(target=httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        host, port = self.httpd.server_address[0], self.httpd.server_address[1]
        return f"http://{host}:{port}"

    def __exit__(self, *_exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=10)


def _post(base: str, payload, headers: dict | None = None):
    import urllib.error
    import urllib.request

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    hdrs = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    hdrs.update(headers or {})
    req = urllib.request.Request(
        base + mcp_server.MCP_ENDPOINT, data=body, headers=hdrs, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            text = resp.read().decode("utf-8")
            return resp.status, dict(resp.headers), (json.loads(text) if text.strip() else None)
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = {"raw": text}
        return exc.code, dict(exc.headers), parsed


class TestStreamableHttp:
    @pytest.fixture()
    def base(self, tmp_path, monkeypatch):
        state = make_state(tmp_path, monkeypatch)
        # 注入同一个假 state：这样 HTTP 用例同时覆盖"传输层"与"工具层"，
        # 而不必先造一代真产物（"造真产物"是 tests/test_server.py 的职责）。
        httpd = mcp_server.serve_http(state.settings, generation_id=GID, state=state)
        with _Live(httpd) as url:
            yield url

    def test_initialize_issues_session_header(self, base):
        status, headers, body = _post(
            base,
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": agent_api.LATEST_PROTOCOL_VERSION}},
        )
        assert status == 200
        assert headers.get("Content-Type", "").startswith("application/json")
        assert headers.get(mcp_server.SESSION_HEADER), "initialize 必须发会话 id"
        assert body["result"]["protocolVersion"] == agent_api.LATEST_PROTOCOL_VERSION

    def test_request_without_session_is_rejected(self, base):
        """非 initialize 的请求没带会话 → 400（规范：要求会话的服务端应当这样回）。"""
        status, _h, body = _post(base, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        assert status == 400
        assert body["error"]["code"] == agent_api.INVALID_REQUEST

    def test_unknown_session_is_404_so_client_reinitializes(self, base):
        status, _h, body = _post(
            base,
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            headers={mcp_server.SESSION_HEADER: "does-not-exist"},
        )
        assert status == 404

    def test_session_flow_notification_returns_202(self, base):
        _s, headers, _b = _post(
            base,
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        )
        sid = headers[mcp_server.SESSION_HEADER]
        status, _h, body = _post(
            base,
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers={mcp_server.SESSION_HEADER: sid},
        )
        assert status == 202
        assert body is None

    def test_tools_call_over_http(self, base):
        _s, headers, _b = _post(
            base, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        )
        sid = headers[mcp_server.SESSION_HEADER]
        status, _h, body = _post(
            base,
            {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
             "params": {"name": "arkrag_get_chunk", "arguments": {"chunk_id": "doc/a#000"}}},
            headers={mcp_server.SESSION_HEADER: sid},
        )
        assert status == 200
        assert body["result"]["isError"] is False
        assert body["result"]["structuredContent"]["chunk_id"] == "doc/a#000"

    def test_sse_only_accept_gets_406_with_explanation(self, base):
        """未实现 SSE 必须**明说**，而不是假装成功再断开连接。"""
        status, _h, body = _post(
            base,
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            headers={"Accept": "text/event-stream"},
        )
        assert status == 406
        assert "text/event-stream" in body["error"]["message"]

    def test_get_returns_405_not_a_hanging_stream(self, base):
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            base + mcp_server.MCP_ENDPOINT,
            headers={"Accept": "text/event-stream"},
            method="GET",
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=30)
        assert exc.value.code == 405
        assert "SSE" in json.loads(exc.value.read().decode("utf-8"))["error"]["message"]

    def test_cross_origin_is_403(self, base):
        """DNS rebinding 防护（规范对本传输的 MUST）。"""
        status, _h, _b = _post(
            base,
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            headers={"Origin": "http://evil.example.com"},
        )
        assert status == 403

    def test_loopback_origin_is_allowed(self, base):
        status, _h, _b = _post(
            base,
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            headers={"Origin": "http://127.0.0.1:58281"},
        )
        assert status == 200

    def test_unsupported_protocol_header_is_400(self, base):
        status, _h, body = _post(
            base,
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            headers={mcp_server.PROTOCOL_HEADER: "1999-01-01"},
        )
        assert status == 400
        assert "1999-01-01" in body["error"]["message"]

    def test_batch_over_http(self, base):
        _s, headers, _b = _post(
            base, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        )
        sid = headers[mcp_server.SESSION_HEADER]
        status, _h, body = _post(
            base,
            [
                {"jsonrpc": "2.0", "id": 1, "method": "ping"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            ],
            headers={mcp_server.SESSION_HEADER: sid},
        )
        assert status == 200
        assert [r["id"] for r in body] == [1, 2]

    def test_delete_ends_the_session(self, base):
        import urllib.error
        import urllib.request

        _s, headers, _b = _post(
            base, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        )
        sid = headers[mcp_server.SESSION_HEADER]
        req = urllib.request.Request(
            base + mcp_server.MCP_ENDPOINT,
            headers={mcp_server.SESSION_HEADER: sid},
            method="DELETE",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            assert resp.status == 200
        # 会话已删 → 后续请求 404（客户端据此重新 initialize）
        status, _h, _b = _post(
            base,
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            headers={mcp_server.SESSION_HEADER: sid},
        )
        assert status == 404

    def test_bad_json_body_is_parse_error(self, base):
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            base + mcp_server.MCP_ENDPOINT,
            data=b"{not json",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=30)
        assert exc.value.code == 400
        assert json.loads(exc.value.read().decode("utf-8"))["error"]["code"] == agent_api.PARSE_ERROR


class TestNativeHttpEntrypoints:
    """OpenAPI 文档里声明的路径必须**真的能用**。

    ## 为什么单独钉这一组

    `build_openapi()` 会生成 `/tools/<name>` 与 `/rpc`。如果服务端不实现它们，
    文档就成了一张假的菜单——比没有文档更糟，因为读者会照它写客户端，
    然后拿到 404 而分不清是自己的问题还是文档的问题。
    所以这里逐条打**真实 HTTP**，断言"文档里有的，打过去真的通"。
    """

    @pytest.fixture()
    def base(self, tmp_path, monkeypatch):
        state = make_state(tmp_path, monkeypatch)
        httpd = mcp_server.serve_http(state.settings, generation_id=GID, state=state)
        with _Live(httpd) as url:
            yield url

    def test_every_openapi_tool_path_is_served(self, base):
        """遍历 OpenAPI 里声明的每一个 `/tools/<name>`，确认它真的能调。"""
        import urllib.error
        import urllib.request

        doc = agent_api.build_openapi()
        tool_paths = sorted(p for p in doc["paths"] if p.startswith(mcp_server.TOOLS_PATH_PREFIX))
        assert tool_paths, "OpenAPI 里没有任何工具路径"
        # 每个工具的最小合法入参。管理类跳过（没配令牌必然 503，另有专门用例）。
        minimal = {
            "arkrag_query": {"question": "凯尔希"},
            "arkrag_search": {"question": "凯尔希"},
            "arkrag_get_chunk": {"chunk_id": "doc/a#000"},
            "arkrag_get_entity": {"entity_id": "凯尔希"},
            "arkrag_get_assertion": {"assertion_id": "as-1"},
            "arkrag_release_info": {"include_self_check": False},
            "arkrag_health": {},
        }
        called = 0
        for path in tool_paths:
            name = path[len(mcp_server.TOOLS_PATH_PREFIX):]
            if name not in minimal:
                continue
            req = urllib.request.Request(
                base + path,
                data=json.dumps(minimal[name], ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                assert resp.status == 200, f"{path} 没通"
                payload = json.loads(resp.read().decode("utf-8"))
            assert payload["isError"] is False, f"{path} 返回了错误：{payload}"
            called += 1
        assert called == len(minimal), f"只覆盖了 {called}/{len(minimal)} 个只读工具"

    def test_tool_path_reports_tool_level_error(self, base):
        """参数错要在工具结果里说（isError），而不是 4xx 掉整条 HTTP。

        理由：模型要靠 `structuredContent.error.message` 纠正自己的参数，
        而 HTTP 4xx 在很多客户端里只被当成"服务出错"，body 不被读。
        """
        import urllib.request

        req = urllib.request.Request(
            base + "/tools/arkrag_query",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            assert resp.status == 200
            payload = json.loads(resp.read().decode("utf-8"))
        assert payload["isError"] is True
        assert payload["structuredContent"]["error"]["code"] == "invalid_arguments"

    def test_unknown_tool_path_is_404(self, base):
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            base + "/tools/arkrag_nope",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=30)
        assert exc.value.code == 404

    def test_rpc_endpoint_single_and_batch(self, base):
        import urllib.request

        def _rpc(payload):
            req = urllib.request.Request(
                base + mcp_server.RPC_PATH,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))

        assert _rpc({"jsonrpc": "2.0", "id": 1, "method": "ping"})["result"] == {}
        batch = _rpc(
            [
                {"jsonrpc": "2.0", "id": 1, "method": "ping"},
                {"jsonrpc": "2.0", "id": 2, "method": "arkrag_health", "params": {}},
            ]
        )
        assert [r["id"] for r in batch] == [1, 2]

    def test_openapi_served_by_the_running_server(self, base):
        """文档由**运行中的服务**给出，内容与 `build_openapi()` 同源。"""
        import urllib.request

        with urllib.request.urlopen(base + mcp_server.OPENAPI_PATH, timeout=30) as resp:
            served = json.loads(resp.read().decode("utf-8"))
        assert served["openapi"] == "3.1.0"
        # paths 集合必须与本进程 `TOOLS` 派生的一致（服务端用的是同一份注册表）。
        assert set(served["paths"]) == set(agent_api.build_openapi()["paths"])
        assert set(served["components"]["schemas"]) == set(
            agent_api.build_openapi()["components"]["schemas"]
        )


class TestSessionStore:
    def test_capacity_evicts_oldest_instead_of_refusing(self):
        """满容量时淘汰最久没动的那个：拒绝会让"最多 N 个客户端"变成硬失败。"""
        store = mcp_server.SessionStore(ttl_s=3600.0, capacity=2)
        first = store.create(protocol_version="v1")
        store.create(protocol_version="v1")
        store.create(protocol_version="v1")
        assert len(store) == 2
        assert store.get(first.session_id) is None

    def test_expired_sessions_are_not_returned(self):
        store = mcp_server.SessionStore(ttl_s=0.001, capacity=8)
        session = store.create(protocol_version="v1")
        import time

        time.sleep(0.01)
        assert store.get(session.session_id) is None

    def test_drop(self):
        store = mcp_server.SessionStore()
        session = store.create(protocol_version="v1")
        assert store.drop(session.session_id) is True
        assert store.drop(session.session_id) is False


# ---------------------------------------------------------------- 复用而非重写


class TestReusesExistingServerLogic:
    """钉住"这一层没有另写业务判定"这件事。"""

    def test_call_route_uses_the_real_handler_class(self):
        """`call_route()` 必须走 `server.make_handler()`，不能自带一份路由副本。"""
        source = inspect.getsource(agent_api.call_route)
        assert "server_mod.make_handler" in source

    def test_tool_handlers_delegate_to_named_routes(self):
        """每个工具的执行体都只调一个具名路由（或一个只做这件事的中间层），不自己算。

        `_tool_query` / `_tool_search` 走 `_query_envelope()`——那个函数体只有
        "把参数拼成 `/query` 的请求体"这一步，本身不含任何业务判定。所以这里把
        "直接命中"与"经 `_query_envelope` 命中"都当作合规，但**不允许**两者都不中
        （那说明有人另写了一套检索/校验）。
        """
        expected = {
            "_tool_get_chunk": "_chunk",
            "_tool_get_entity": "_entity",
            "_tool_get_assertion": "_assertion",
            "_tool_release_info": "_release_current",
            "_tool_health": "_health",
            "_tool_admin_rollback": "_admin_rollback",
        }
        for func_name, route in expected.items():
            source = inspect.getsource(getattr(agent_api, func_name))
            assert f'"{route}"' in source, f"{func_name} 没有调 {route}（可能另写了一套实现）"

        envelope = inspect.getsource(agent_api._query_envelope)
        assert '"_query"' in envelope, "_query_envelope 必须只转发到 /query 路由"
        for func_name in ("_tool_query", "_tool_search"):
            source = inspect.getsource(getattr(agent_api, func_name))
            assert "_query_envelope(" in source, f"{func_name} 没走统一的 /query 信封"
            # 这两个工具都**不许**自己去调检索/答案器：那是另一条通路，
            # 会让 search 与 query 的排序各说各话。
            assert "retriever" not in source, f"{func_name} 自己动了检索器"
            assert "answerer" not in source, f"{func_name} 自己动了答案器"

    def test_retrieval_params_are_shared_not_copied(self):
        """`arkrag_query` 与 `arkrag_search` 的检索参数必须是**同一个对象**。

        各写一份的后果是有人只改其中一个的 `maximum`：search 收 60 而 query 收 50,
        两个工具对"合法参数"的说法不一致，调用方没法知道该信哪个。
        """
        q = agent_api.TOOLS["arkrag_query"].input_schema["properties"]
        s = agent_api.TOOLS["arkrag_search"].input_schema["properties"]
        for key in ("top_k", "graph_depth"):
            assert q[key] == s[key], f"{key} 在两个工具里不一致：{q[key]} vs {s[key]}"
            assert q[key] is s[key], f"{key} 是被各自拷贝的，不是共享定义"

    def test_limits_come_from_server_constants_not_literals(self, state):
        """schema 里的上限必须是服务端常量的**值**，不是手抄的数字。"""
        q = agent_api.TOOLS["arkrag_query"].input_schema["properties"]
        assert q["top_k"]["maximum"] == server.DEFAULT_MAX_TOP_K
        assert q["graph_depth"]["maximum"] == server.DEFAULT_MAX_GRAPH_DEPTH
        assert q["max_context_tokens"]["maximum"] == server.DEFAULT_MAX_CONTEXT_TOKENS

    def test_query_profile_comes_from_prod_profile(self, state):
        body = _call(state, "arkrag_query", {"question": "凯尔希"})["structuredContent"]
        assert body["profile"]["max_context_tokens"] == int(server.pipeline_mod.PROD_PROFILE["max_context_tokens"])
        assert body["profile"]["options"] == server.pipeline_mod.prod_options()
