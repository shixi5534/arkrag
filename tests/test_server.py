"""J1/J2/J3 的 HTTP 服务：路由、状态码、就绪判据、安全边界与可观测性。

## 这些测试打的是真实 HTTP

`server.serve()` 只建不启动，正是为了让测试拿到 `server_address` 里的实际端口
（`port=0` 由系统分配）再用 `urllib` 打过去。绕过 HTTP 直接调 `Handler._query()`
测不到路由匹配、状态码、"响应头里有 `Cache-Control: no-store`"这三类东西——
而它们恰恰是安全要求落地的位置。

## 夹具：造一代真的能服务的世代

用 `scripts/build_generation.py` 的**真实构建流程**（identity 图模型）造一代
四块语料的小世代，而不是手写几个产物文件。理由是就绪判据里的
`graph_index_identity` 要核六个 artifact 摘要与分层身份，手写很容易造出一代
"看起来对但其实少一层"的产物——那样测出来的 PASS 是假 PASS。
门禁本身换成替身（理由同 `tests/test_build_generation.py`：真实门禁判据要求
平均度数 > 6 这类全量语料才有的规模）。

## 每个"能通过"的检查都必须有一条"能让它失败"的用例

`/ready` 的七项检查里六项阻断。只有断言"全 PASS 时 ready=true"是不够的——
那只证明探针会说好话。所以每一项都配一条把它弄坏的用例，且**必须在缓存生效
之后再弄坏**（否则测不出缓存键是否包含产物状态）。
"""
from __future__ import annotations

import argparse
import http.client
import importlib.util
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

from arkrag import config, gates, pipeline, release, server

ROOT = Path(__file__).resolve().parents[1]

GID = "g-20260101T000000Z-abcdef"

CHUNKS = [
    {
        "chunk_id": f"doc/a#{i:03d}",
        "content_key": f"ck{i:04d}",
        "text": (
            "凯尔希：切尔诺伯格已经沦陷了。\n"
            "阿米娅：我们必须去罗德岛。\n"
            "凯尔希：博士，罗德岛不会为一个人改变航向。"
        ),
        "title": "切尔诺伯格",
        "chapter": "第一章",
        "category": "obt/main",
        "scene_id": f"doc/a#{i}",
        "doc_id": "doc/a",
        "speakers": ["凯尔希", "阿米娅"],
        "chars": 60,
        "turn_ids": [0, 1, 2],
        "scene_index": i,
    }
    for i in range(4)
]

RECORDS = {
    c["chunk_id"]: {
        "entities": [
            {"name": "凯尔希", "type": "角色"},
            {"name": "阿米娅", "type": "角色"},
            {"name": "罗德岛", "type": "组织"},
            {"name": "切尔诺伯格", "type": "地点"},
        ],
        # `evidence` 必须是块正文的**逐字子串**，否则 `assertion.build_assertions`
        # 会以 `evidence_not_found` 丢弃它——那样夹具看起来"建成了"，其实答案侧
        # 一条断言都没有，`ExtractiveAnswerer` 只会回 `no_supported_claim`。
        "relations": [
            {
                "source": "凯尔希",
                "relation": "隶属",
                "target": "罗德岛",
                "evidence": "凯尔希：博士，罗德岛不会为一个人改变航向。",
                "confidence": 0.9,
            },
            {
                "source": "阿米娅",
                "relation": "隶属",
                "target": "罗德岛",
                "evidence": "阿米娅：我们必须去罗德岛。",
                "confidence": 0.9,
            },
            {
                "source": "凯尔希",
                "relation": "交谈",
                "target": "阿米娅",
                "evidence": "阿米娅：我们必须去罗德岛。",
                "confidence": 0.8,
            },
        ],
    }
    for c in CHUNKS
}


def _passing(stage: str, n: int):
    """替身门：只证明"门禁这一段接上了"，不假装在验真实语料。"""

    def factory(*_a, **_k):
        run = gates.GateRun()
        for i in range(n):
            run.add(gates.Gate(f"{stage}.{i}", stage, "替身门", "1", "PASS", "服务测试的替身"))
        return run

    return factory


def _load_build_script():
    spec = importlib.util.spec_from_file_location(
        "build_generation_under_server_test", ROOT / "scripts" / "build_generation.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _isolate(monkeypatch, tmp_path) -> Path:
    """把 `DATA_DIR` / 世代目录 / 运行目录 / 指针全部指到临时根。"""
    data_dir = (tmp_path / "data").resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    assert str(data_dir).startswith(str(tmp_path.resolve())), "DATA_DIR 必须落在临时根下"
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(release, "GENERATIONS_DIR", data_dir / "generations")
    monkeypatch.setattr(release, "RUNS_DIR", data_dir / "runs")
    monkeypatch.setattr(release, "CURRENT_PATH", data_dir / "current.json")
    (data_dir / "generations").mkdir()
    (data_dir / "runs").mkdir()
    with (data_dir / "chunks.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for c in CHUNKS:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    (data_dir / "source_verification.json").write_text(
        json.dumps({"status": "verified_zip", "checks": ["zip_bytes_and_sha256"], "problems": []}),
        encoding="utf-8",
    )
    (data_dir / "corpus_meta.json").write_text(
        json.dumps({"corpus_fp": "corpusfp01234567", "source_status": "verified_zip"}),
        encoding="utf-8",
    )
    return data_dir


@pytest.fixture()
def servable(tmp_path, monkeypatch):
    """走真实构建流程造一代可服务的世代，返回 (gid, data_dir)。"""
    data_dir = _isolate(monkeypatch, tmp_path)
    mod = _load_build_script()
    monkeypatch.setattr(mod.extract, "load_chunks", lambda *a, **k: [dict(c) for c in CHUNKS])
    monkeypatch.setattr(mod.cache, "is_cached_for_chunk", lambda *a, **k: True)
    monkeypatch.setattr(
        mod.graph,
        "load_records",
        lambda chunks: [{"chunk_id": c["chunk_id"], "raw": RECORDS[c["chunk_id"]]} for c in chunks],
    )
    monkeypatch.setattr(mod.gates, "l4_gates", _passing("L4", 2))
    monkeypatch.setattr(mod.gates, "l5_gates", _passing("L5", 3))
    monkeypatch.setattr(mod.gates, "l6_gates", _passing("L6", 2))

    args = argparse.Namespace(
        probes=5, no_publish=False, allow_incomplete=False, graph_model="identity"
    )
    assert mod.build(args) == 0, "夹具世代必须构建成功"
    gid = release.read_current()["generation_id"]
    assert release.verify_generation(release.Generation(gid)) == []
    return gid, data_dir


# ---------------------------------------------------------------- HTTP 客户端


class Client:
    """极简 HTTP 客户端。`base` 由 `_serve()` 给出。"""

    def __init__(self, base: str) -> None:
        self.base = base

    def get(self, path: str, headers: dict | None = None):
        return self._call("GET", path, None, headers)

    def post(self, path: str, body=None, headers: dict | None = None, raw: bytes | None = None):
        return self._call("POST", path, body, headers, raw)

    def post_raw(self, path: str, body: bytes, *, content_length: int | None = None):
        """发一个**声明长度与实际字节无关**的请求，用来测"只发头不发体"。

        `urllib` 会按实际字节改 `Content-Length`，这条路测不到"声明 5 MB 但一个
        字节都不发"。所以这里直接走 `http.client` 手工拼头。
        """
        parsed = urllib.parse.urlsplit(self.base)
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=30)
        try:
            conn.putrequest("POST", path, skip_accept_encoding=True)
            conn.putheader("Content-Type", "application/json")
            conn.putheader("Content-Length", str(content_length if content_length is not None else len(body)))
            conn.endheaders()
            if body:
                conn.send(body)
            resp = conn.getresponse()
            text = resp.read().decode("utf-8")
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = {"raw": text}
            return resp.status, {k: v for k, v in resp.getheaders()}, payload
        finally:
            conn.close()

    def _call(self, method, path, body, headers, raw=None):
        hdrs = dict(headers or {})
        data = raw
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            hdrs.setdefault("Content-Type", "application/json")
        req = urllib.request.Request(self.base + path, data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.status, dict(resp.headers), json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8")
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = {"raw": text}
            return exc.code, dict(exc.headers), payload


class Live:
    def __init__(self, httpd, client: Client) -> None:
        self.httpd = httpd
        self.client = client
        self._thread = None

    def __enter__(self) -> Client:
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()
        return self.client

    def __exit__(self, *_exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=10)


def _serve(gid: str, **overrides) -> Live:
    kwargs = {"host": "127.0.0.1", "port": 0, "rate_limit_calls": 1000}
    kwargs.update(overrides)
    settings = server.Settings(**kwargs)
    httpd = server.serve(settings, generation_id=gid)
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    return Live(httpd, Client(base))


def _settings(**overrides) -> server.Settings:
    kwargs = {"host": "127.0.0.1", "port": 0, "rate_limit_calls": 1000}
    kwargs.update(overrides)
    return server.Settings(**kwargs)


# ---------------------------------------------------------------- 清洗与设置


class TestSanitize:
    def test_strips_repo_root_and_absolute_paths(self):
        raw = f"打不开 {config.ROOT}\\data\\arkrag\\current.json 以及 C:\\Windows\\x.txt"
        cleaned = server.sanitize_text(raw)
        assert "<repo>" in cleaned
        assert "<path>" in cleaned
        assert str(config.ROOT) not in cleaned
        assert "C:\\Windows" not in cleaned

    def test_redacts_admin_token(self):
        """口令绝不能出现在对外消息里。"""
        settings = server.Settings(admin_token="s3cr3t-token-value")
        cleaned = server.sanitize_text("认证失败 token=s3cr3t-token-value", settings=settings)
        assert "s3cr3t-token-value" not in cleaned
        assert "<redacted>" in cleaned

    def test_describe_reports_admin_enabled_without_token(self):
        off = server.Settings().describe()
        on = server.Settings(admin_token="abc123").describe()
        assert off["admin_enabled"] is False
        assert on["admin_enabled"] is True
        assert "abc123" not in json.dumps(on, ensure_ascii=False)

    def test_settings_never_defaults_admin_open(self):
        """fail-closed 的默认值：没有环境变量就没有口令。"""
        settings = server.Settings.from_env()
        assert settings.admin_token == ""
        assert settings.describe()["admin_enabled"] is False


# ---------------------------------------------------------------- 存活 vs 就绪


class TestHealthIsNotReady:
    def test_health_does_not_touch_data(self, servable, monkeypatch):
        """`/health` 故意不查数据：数据全坏时它仍然 200。

        这与 `/ready` 分开的理由就在这里——合并之后"活着但数据坏了"会被报成健康。
        """
        gid, data_dir = servable
        # 把块正文整个删掉：/ready 会挂，/health 必须仍然 200
        (data_dir / "chunks.jsonl").unlink()
        with _serve(gid) as client:
            status, _h, body = client.get("/health")
            assert status == 200
            assert body["ok"] is True
            assert "不代表就绪" in body["note"]

            ready_status, _h, ready = client.get("/ready")
            assert ready_status == 503
            assert ready["ready"] is False

    def test_health_reports_served_generation_once_loaded(self, servable):
        gid, _data = servable
        with _serve(gid) as client:
            status, _h, body = client.get("/health")
            assert status == 200
            assert body["served_generation_id"] == gid


class TestReadyChecks:
    def test_all_blocking_checks_pass_on_a_good_generation(self, servable):
        gid, _data = servable
        with _serve(gid) as client:
            status, _h, body = client.get("/ready")
            assert status == 200
            assert body["ready"] is True
            ids = {c["id"]: c for c in body["checks"]}
            assert set(ids) == {
                "current_pointer",
                "manifest_complete",
                "graph_index_identity",
                "data_readable",
                "gates_pass",
                "chunks_cover_generation",
                "serving_is_current",
            }
            assert all(c["ok"] for c in body["checks"]), body["checks"]

    def test_serving_is_current_is_reported_but_not_blocking(self, servable):
        """服务的世代不是线上指针那一代时：`ready` 仍为 true，但必须可见。

        灰度期就是这个形态（服务 READY 未发布世代），计划 K2 允许。但如果它悄悄
        通过、什么都不说，那么"线上跑的不是线上指针"就永远没人知道。
        """
        gid, data_dir = servable
        pointer = json.loads((data_dir / "current.json").read_text("utf-8"))
        pointer["generation_id"] = "g-20260101T000000Z-000000"
        (data_dir / "current.json").write_text(json.dumps(pointer), encoding="utf-8")

        with _serve(gid) as client:
            status, _h, body = client.get("/ready")
            assert status == 200
            checks = {c["id"]: c for c in body["checks"]}
            assert checks["serving_is_current"]["ok"] is False
            assert checks["serving_is_current"]["blocking"] is False
            assert body["ready"] is True

    def test_manifest_missing_field_blocks_ready(self, servable):
        gid, _data = servable
        gen = release.Generation(gid)
        manifest = json.loads(gen.manifest_path.read_text("utf-8"))
        del manifest["assertion_count"]
        gen.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with _serve(gid) as client:
            status, _h, body = client.get("/ready")
            assert status == 503
            checks = {c["id"]: c for c in body["checks"]}
            assert checks["manifest_complete"]["ok"] is False
            assert "assertion_count" in checks["manifest_complete"]["detail"]

    def test_gate_report_with_a_fail_blocks_ready(self, servable):
        """门禁报告里只要有一条不是 PASS，就不许 ready。

        先查一遍让缓存生效，再改报告——这样才能证明缓存键包含报告文件本身。
        """
        gid, _data = servable
        with _serve(gid) as client:
            assert client.get("/ready")[2]["ready"] is True
            gen = release.Generation(gid)
            rows = json.loads(gen.gate_report_path.read_text("utf-8"))
            rows[0]["verdict"] = "FAIL"
            gen.gate_report_path.write_text(json.dumps(rows), encoding="utf-8")

            status, _h, body = client.get("/ready")
            assert status == 503
            checks = {c["id"]: c for c in body["checks"]}
            assert checks["gates_pass"]["ok"] is False
            assert "不是 PASS" in checks["gates_pass"]["detail"]

    def test_empty_gate_report_is_not_pass(self, servable):
        """空报告不算 PASS：没有可判条目 ≠ 全部通过。"""
        gid, _data = servable
        with _serve(gid) as client:
            assert client.get("/ready")[2]["ready"] is True
            gen = release.Generation(gid)
            gen.gate_report_path.write_text("[]", encoding="utf-8")
            status, _h, body = client.get("/ready")
            assert status == 503
            checks = {c["id"]: c for c in body["checks"]}
            assert checks["gates_pass"]["ok"] is False
            assert "空" in checks["gates_pass"]["detail"]

    def test_truncated_generation_chunks_blocks_ready(self, servable):
        """块正文被截断必须阻断——它是引用校验的判据。

        结尾**不是**「缺某块」，而是「与 Manifest 不符」：`resolve_chunks()` 先
        核 `chunk_manifest_hash`，对不上就 fail-closed，根本不会走到逐块比对。
        这是有意为之——身份对不上时，讨论"缺哪几块"没有意义。

        第一版这条用例截的是全局 `data/chunks.jsonl`，而缺陷 A 修复后消费者改读
        世代自带的 `<gen>/chunks.jsonl`，于是它打到了一个没人读的文件上。
        """
        gid, _data_dir = servable
        gen = release.Generation(gid)
        chunks_path = Path(pipeline.resolve_chunks(gen, gen.load()))
        assert chunks_path == gen.chunks_path, "消费者必须读世代自带的块正文"
        with _serve(gid) as client:
            assert client.get("/ready")[2]["ready"] is True
            with chunks_path.open("w", encoding="utf-8", newline="\n") as fh:
                for c in CHUNKS[:-1]:
                    fh.write(json.dumps(c, ensure_ascii=False) + "\n")
            status, _h, body = client.get("/ready")
            assert status == 503
            checks = {c["id"]: c for c in body["checks"]}
            assert checks["chunks_cover_generation"]["ok"] is False
            detail = checks["chunks_cover_generation"]["detail"]
            assert "不符" in detail, detail

    def test_chunks_cover_check_reports_which_blocks_are_missing(self, servable, tmp_path, monkeypatch):
        """覆盖校验单独有效：块正文合法但少了索引里的某些块时，必须点名缺哪几块。

        直接打 `server._compute_chunks_cover()` 并让 `resolve_chunks()` 交回一份
        **少一块**的文件——绕开身份校验，专门验证逐块比对这一步本身。若不这样拆，
        截断永远先被身份校验拦下，这条判定就永远没被执行过。
        """
        gid, _data_dir = servable
        gen = release.Generation(gid)
        short = tmp_path / "short-chunks.jsonl"
        with short.open("w", encoding="utf-8", newline="\n") as fh:
            for c in CHUNKS[:-1]:
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
        monkeypatch.setattr(server.pipeline_mod, "resolve_chunks", lambda g, m: short)
        ok, detail = server._compute_chunks_cover(gid)
        assert ok is False
        assert CHUNKS[-1]["chunk_id"] in detail, detail
        assert "缺 1 块" in detail, detail
        # 同一条判定在块齐时必须是绿的，否则它只是个恒假断言
        monkeypatch.setattr(server.pipeline_mod, "resolve_chunks", lambda g, m: gen.chunks_path)
        ok2, detail2 = server._compute_chunks_cover(gid)
        assert ok2 is True, detail2

    def test_global_chunk_file_is_no_longer_what_the_consumer_reads(self, servable):
        """截断全局 `data/chunks.jsonl` 不再影响就绪——证明接线真的生效了。

        这条是缺陷 A 的**反向证据**：修之前消费者读全局文件，截它必红；
        修之后世代自带块正文，全局文件被截了也不该影响这一代的就绪状态。
        若哪天有人把 `resolve_chunks()` 改回读全局文件，这条立刻变红。
        """
        gid, data_dir = servable
        gen = release.Generation(gid)
        chunks_path = Path(pipeline.resolve_chunks(gen, gen.load()))
        assert chunks_path != data_dir / "chunks.jsonl", "消费者不该读全局块文件"
        with _serve(gid) as client:
            assert client.get("/ready")[2]["ready"] is True
            with (data_dir / "chunks.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(CHUNKS[0], ensure_ascii=False) + "\n")
            status, _h, body = client.get("/ready")
            assert status == 200, body
            assert body["ready"] is True

    def test_tampered_artifact_blocks_ready_after_cache_warm(self, servable):
        """产物被改之后，第二次 `/ready` 必须变红。

        这是本文件最关键的一条：缓存键如果只按 Manifest 做，产物换了它还会继续
        说"逐项一致"——探针从此只会说好话。用例刻意先让缓存热起来再动手。
        """
        gid, _data = servable
        with _serve(gid) as client:
            assert client.get("/ready")[2]["ready"] is True
            gen = release.Generation(gid)
            original = gen.index_path.read_bytes()
            gen.index_path.write_bytes(original + b"\n")
            try:
                status, _h, body = client.get("/ready")
                assert status == 503
                checks = {c["id"]: c for c in body["checks"]}
                assert checks["graph_index_identity"]["ok"] is False
            finally:
                gen.index_path.write_bytes(original)

    def test_corrupt_index_does_not_crash_the_probe(self, servable):
        """索引字节被翻坏时必须回"未就绪 + 原因"，而不是 500。

        `release.verify_generation` 曾只接 `OSError`/`JSONDecodeError`，
        而坏掉的 UTF-8 抛的是 `UnicodeDecodeError`——断言就崩在异常里，
        探针返回 500，运维看到的是"服务崩了"而不是"这一代不可用"。
        """
        gid, _data = servable
        with _serve(gid) as client:
            gen = release.Generation(gid)
            original = gen.index_path.read_bytes()
            flipped = bytearray(original)
            flipped[100] ^= 0xFF
            gen.index_path.write_bytes(bytes(flipped))
            try:
                status, _h, body = client.get("/ready")
                assert status == 503, body
                assert body["ready"] is False
                checks = {c["id"]: c for c in body["checks"]}
                assert checks["graph_index_identity"]["ok"] is False
                # 对外文本里不许出现本机绝对路径
                assert str(config.ROOT) not in json.dumps(body, ensure_ascii=False)
            finally:
                gen.index_path.write_bytes(original)

    def test_ready_needs_a_servable_generation(self, tmp_path, monkeypatch):
        """没有发布指针、也没指定世代：`/ready` 必须 503，不许默认 ready。"""
        _isolate(monkeypatch, tmp_path)
        settings = _settings()
        state = server.AppState(settings, generation_id="")
        result = state.readiness()
        assert result["ready"] is False
        assert result["checks"][0]["id"] == "current_pointer"
        assert result["checks"][0]["ok"] is False


class TestServedGeneration:
    def test_legacy_generation_is_not_servable(self, servable):
        """线上指针那一代是 ark-rag-1（无断言/邻接）：/query 必须 503 而不是崩。"""
        gid, _data = servable
        # 造一个只有 Manifest 的 ark-rag-1 世代目录
        legacy_gid = "g-20260101T000000Z-111111"
        legacy_dir = release.GENERATIONS_DIR / legacy_gid
        legacy_dir.mkdir(parents=True)
        (legacy_dir / "manifest.json").write_text(
            json.dumps({"generation_id": legacy_gid, "graph_schema": None}), encoding="utf-8"
        )
        with _serve(legacy_gid) as client:
            status, _h, body = client.post("/query", {"question": "凯尔希是谁？"})
            assert status == 503
            assert body["error"]["code"] == "generation_not_servable"

    def test_invalidate_served_drops_cached_generation(self, servable):
        """回滚后必须丢掉缓存，否则指针切了服务还在用旧代回答。"""
        gid, _data = servable
        state = server.AppState(_settings(), generation_id=gid)
        first = state.served()
        assert state.served() is first
        state.invalidate_served()
        assert state._served is None
        assert state.served_or_none() is not None


# ---------------------------------------------------------------- 查询


class TestQueryEndpoint:
    def test_query_answers_with_citations(self, servable):
        gid, _data = servable
        with _serve(gid) as client:
            status, headers, body = client.post("/query", {"question": "凯尔希属于哪个组织？"})
            assert status == 200
            assert headers["Cache-Control"] == "no-store"
            assert headers["X-Content-Type-Options"] == "nosniff"
            assert body["generation_id"] == gid
            assert body["query_id"].startswith("q-")
            assert body["untrusted_content"] is True
            assert body["answer"]["text"]
            assert body["answer"]["verification"]["rejected"] == 0
            # 本机无 LLM：成本字段必须是 null + 原因，不是 0（0 会被当成"免费"）
            assert body["stats"]["token_cost"] is None
            assert "无 LLM 端点" in body["stats"]["token_cost_reason"]

    def test_query_rejects_empty_question(self, servable):
        gid, _data = servable
        with _serve(gid) as client:
            status, _h, body = client.post("/query", {"question": "   "})
            assert status == 400
            assert body["error"]["code"] == "bad_request"

    def test_query_rejects_question_over_shared_limit(self, servable):
        """上限必须来自 `query.QUERY_MAX_CHARS` 这一处，不是 handler 里的字面量。"""
        from arkrag import query as query_mod

        gid, _data = servable
        with _serve(gid) as client:
            status, _h, body = client.post("/query", {"question": "凯" * (query_mod.QUERY_MAX_CHARS + 1)})
            assert status == 400
            assert body["error"]["code"] == "param_out_of_range"
            assert str(query_mod.QUERY_MAX_CHARS) in body["error"]["message"]

    def test_query_rejects_out_of_range_top_k_without_clamping(self, servable):
        """越界返回 400 并说明范围，**不夹紧**——夹紧会让调用方以为参数生效了。"""
        gid, _data = servable
        with _serve(gid) as client:
            status, _h, body = client.post("/query", {"question": "凯尔希", "top_k": 999})
            assert status == 400
            assert body["error"]["code"] == "param_out_of_range"
            assert "top_k=999" in body["error"]["message"]

    def test_query_rejects_boolean_top_k(self, servable):
        """`true` 在 Python 里是 int，不显式拒绝就会被当成 1 悄悄接受。"""
        gid, _data = servable
        with _serve(gid) as client:
            status, _h, body = client.post("/query", {"question": "凯尔希", "top_k": True})
            assert status == 400
            assert body["error"]["code"] == "bad_param"
            assert "布尔" in body["error"]["message"]

    def test_query_rejects_graph_depth_over_bound(self, servable):
        gid, _data = servable
        with _serve(gid) as client:
            status, _h, body = client.post("/query", {"question": "凯尔希", "graph_depth": 9})
            assert status == 400
            assert body["error"]["code"] == "param_out_of_range"

    def test_query_rejects_non_json_body(self, servable):
        gid, _data = servable
        with _serve(gid) as client:
            status, _h, body = client.post("/query", raw=b"not json at all")
            assert status == 400
            assert body["error"]["code"] == "bad_json"

    def test_query_rejects_json_array_body(self, servable):
        gid, _data = servable
        with _serve(gid) as client:
            status, _h, body = client.post("/query", raw=b"[1,2,3]")
            assert status == 400
            assert body["error"]["code"] == "bad_json"

    def test_oversized_body_is_refused_before_reading(self, servable):
        gid, _data = servable
        with _serve(gid, max_body_bytes=256) as client:
            payload = {"question": "凯" * 400}
            status, _h, body = client.post("/query", payload)
            assert status == 413
            assert body["error"]["code"] == "payload_too_large"

    def test_oversized_body_still_delivers_the_413(self, servable):
        """超限时客户端必须**真的收到 413**，而不是连接被重置。

        实测踩到的形态：`test_oversized_body_is_refused_before_reading` 单跑通过、
        全量跑偶发失败，报 `ConnectionAbortedError: [WinError 10053]`。原因是只回
        413 就关连接，接收缓冲区里还有未读字节，`close()` 发 RST 而不是 FIN，
        客户端内核把已到达的 413 一起丢掉。

        单跑通过正是这条缺陷最容易被放过的地方——它看起来像 flaky。但线上同一个
        路径会把"请求太大"变成"服务挂了"。所以这里单独钉一条：用一个会完整
        发出请求体的客户端（不是流式半发），断言状态码真的读到了。
        """
        gid, _data = servable
        with _serve(gid, max_body_bytes=256) as client:
            payload = {"question": "凯" * 400}
            # 连续发三次：若第一次留下半读的连接，后续会读到脏字节或直接重置
            for _ in range(3):
                status, headers, body = client.post("/query", payload)
                assert status == 413, "超限必须回 413，实际 %r" % (status,)
                assert body["error"]["code"] == "payload_too_large"
                # 连接被复用也没问题（服务端已把请求体读掉）
                assert headers.get("Content-Type", "").startswith("application/json")

    def test_413_does_not_leak_internal_drain_directive(self, servable):
        """`_drain` 是服务端内部处置指令，不该出现在对外错误体里。"""
        gid, _data = servable
        with _serve(gid, max_body_bytes=256) as client:
            _status, _h, body = client.post("/query", {"question": "凯" * 400})
            assert "_drain" not in body["error"]
            assert not [k for k in body["error"] if k.startswith("_")]

    def test_unreasonably_large_body_closes_the_connection(self, servable):
        """声明超出排空上限的请求体：仍回 413，但要显式关连接。

        排空是为了让客户端读得到响应，但不能被一个声明了 10 GB 的请求拖住。
        超过上限就不读，改发 `Connection: close`——重开连接是调用方的责任，
        这只是明确告知，不是让服务端陪着耗。
        """
        gid, _data = servable
        with _serve(gid, max_body_bytes=256) as client:
            # 只发头、不发体：服务端看到 Content-Length 超限就会拒，
            # 且因为超过排空上限，直接标记 close。
            status, headers, body = client.post_raw("/query", b"", content_length=5 * 1024 * 1024)
            assert status == 413
            assert body["error"]["code"] == "payload_too_large"
            assert headers.get("Connection", "").lower() == "close"

    def test_response_is_degraded_in_documented_order(self, servable):
        """超响应上限时按 context.units → citations → channels → hits 顺序降级。"""
        gid, _data = servable
        with _serve(gid, max_response_bytes=1200) as client:
            status, _h, body = client.post("/query", {"question": "凯尔希属于哪个组织？"})
            assert status == 200
            assert body["truncated"], "降级了就必须显式说明丢了什么"
            assert body["truncated"][0] == "context.units"

    def test_custom_context_budget_does_not_leak_between_requests(self, servable):
        """`_with_budget` 必须克隆：原地改预算会让并发请求读到别人的预算。"""
        gid, _data = servable
        with _serve(gid) as client:
            first = client.post("/query", {"question": "凯尔希", "max_context_tokens": 300})[2]
            second = client.post("/query", {"question": "凯尔希"})[2]
            assert first["limits"]["max_context_tokens"] == 300
            assert second["limits"]["max_context_tokens"] == 6000

    def test_query_writes_a_log_record_without_the_question(self, servable):
        """J3：默认不记录问句原文，只记长度与摘要。"""
        gid, data_dir = servable
        with _serve(gid) as client:
            assert client.post("/query", {"question": "凯尔希属于哪个组织？"})[0] == 200
        log_path = data_dir / "logs" / "queries.jsonl"
        assert log_path.exists()
        records = [json.loads(x) for x in log_path.read_text("utf-8").splitlines() if x.strip()]
        assert len(records) == 1
        rec = records[0]
        assert "question" not in rec
        assert rec["query_chars"] == len("凯尔希属于哪个组织？")
        assert len(rec["query_sha256"]) == 16
        assert rec["generation_id"] == gid
        assert rec["citation_verification"]["rejected"] == 0


class TestTimeout:
    def test_slow_query_returns_504(self, servable, monkeypatch):
        """超时返回 504 而不是无限等。工作线程无法被抢占，这一点写在模块 docstring 里。"""
        gid, _data = servable

        def slow(*_a, **_k):
            time.sleep(2.0)
            raise AssertionError("不该走到这里")

        monkeypatch.setattr(server.pipeline_mod, "ask_once", slow)
        with _serve(gid, timeout_s=0.2) as client:
            status, _h, body = client.post("/query", {"question": "凯尔希"})
            assert status == 504
            assert body["error"]["code"] == "timeout"


class TestRateLimit:
    def test_exceeding_rate_limit_returns_429_with_retry_after(self, servable):
        gid, _data = servable
        with _serve(gid, rate_limit_calls=3, rate_limit_window_s=60.0) as client:
            statuses = [client.get("/health")[0] for _ in range(5)]
            assert statuses[:3] == [200, 200, 200]
            assert statuses[3] == 429
            status, headers, body = client.get("/health")
            assert status == 429
            assert body["error"]["code"] == "rate_limited"
            assert int(headers["Retry-After"]) >= 1


# ---------------------------------------------------------------- 单记录


class TestRecordRoutes:
    def test_chunk_id_with_slashes_is_matched_whole(self, servable):
        """`chunk_id` 本身带斜杠（`doc/a#000`），不能用路径分段匹配。"""
        gid, _data = servable
        with _serve(gid) as client:
            status, _h, body = client.get("/chunks/doc/a%23000")
            assert status == 200
            assert body["chunk_id"] == "doc/a#000"
            assert body["untrusted"] is True
            assert body["text_truncated"] is False

    def test_unknown_chunk_returns_404(self, servable):
        gid, _data = servable
        with _serve(gid) as client:
            status, _h, body = client.get("/chunks/doc/a%23999")
            assert status == 404
            assert body["error"]["code"] == "chunk_not_found"

    def test_long_chunk_text_is_clipped_and_flagged(self, servable):
        gid, _data = servable
        with _serve(gid, max_record_chars=10) as client:
            status, _h, body = client.get("/chunks/doc/a%23000")
            assert status == 200
            assert body["text_truncated"] is True
            assert len(body["text"]) == 10

    def test_assertion_lookup_uses_retriever_index(self, servable):
        """断言走 `Retriever.assertions_by_id`，不另建一份字典。"""
        gid, _data = servable
        with _serve(gid) as client:
            _status, _h, entity = client.get("/entities/%E5%87%AF%E5%B0%94%E5%B8%8C")
            edges = entity.get("edges") or []
            assert edges, "夹具世代里凯尔希必须至少有一条边"
            assertion_id = (edges[0].get("assertion_ids") or [None])[0]
            assert assertion_id, edges[0]
            status, _h, body = client.get(f"/assertions/{assertion_id}")
            assert status == 200
            assert body["assertion_id"] == assertion_id
            assert body["evidence_text"]
            assert body["untrusted"] is True

    def test_unknown_assertion_returns_404(self, servable):
        gid, _data = servable
        with _serve(gid) as client:
            status, _h, body = client.get("/assertions/a-ffffffffffffffff")
            assert status == 404
            assert body["error"]["code"] == "assertion_not_found"

    def test_entity_reports_total_and_truncation(self, servable):
        gid, _data = servable
        with _serve(gid, max_top_k=1) as client:
            status, _h, body = client.get("/entities/%E5%87%AF%E5%B0%94%E5%B8%8C")
            assert status == 200
            assert body["entity"]["name"] == "凯尔希"
            assert body["edges_returned"] == 1
            assert body["edges_truncated"] is True
            assert body["edges_total"] >= 1

    def test_unknown_entity_returns_404(self, servable):
        gid, _data = servable
        with _serve(gid) as client:
            status, _h, body = client.get("/entities/%E4%B8%8D%E5%AD%98%E5%9C%A8")
            assert status == 404
            assert body["error"]["code"] == "entity_not_found"

    def test_unknown_path_returns_404(self, servable):
        gid, _data = servable
        with _serve(gid) as client:
            assert client.get("/nope")[0] == 404
            assert client.post("/nope", {})[0] == 404

    def test_release_current_reports_pointer(self, servable):
        gid, _data = servable
        with _serve(gid) as client:
            status, _h, body = client.get("/release/current")
            assert status == 200
            assert body["current"]["generation_id"] == gid
            assert body["serving_is_current"] is True

    def test_limits_endpoint_exposes_bounds_only(self, servable):
        gid, _data = servable
        with _serve(gid, admin_token="top-secret-xyz") as client:
            status, _h, body = client.get("/limits")
            assert status == 200
            assert body["limits"]["admin_enabled"] is True
            assert "top-secret-xyz" not in json.dumps(body, ensure_ascii=False)


# ---------------------------------------------------------------- 管理接口


class TestAdminFailClosed:
    def test_admin_disabled_when_no_token_configured(self, servable):
        """没配口令 = 管理接口关闭。默认放开是这类接口最常见的事故源。"""
        gid, _data = servable
        with _serve(gid) as client:
            for path in ("/admin/rollback", "/admin/evaluate"):
                status, _h, body = client.post(path, {"generation_id": gid})
                assert status == 503
                assert body["error"]["code"] == "admin_disabled"

    def test_wrong_token_is_rejected(self, servable):
        gid, _data = servable
        with _serve(gid, admin_token="right-token") as client:
            status, _h, body = client.post(
                "/admin/rollback", {"generation_id": gid},
                headers={"Authorization": "Bearer wrong-token"},
            )
            assert status == 401
            assert body["error"]["code"] == "unauthorized"

    def test_missing_authorization_header_is_rejected(self, servable):
        gid, _data = servable
        with _serve(gid, admin_token="right-token") as client:
            status, _h, body = client.post("/admin/rollback", {"generation_id": gid})
            assert status == 401
            assert body["error"]["code"] == "unauthorized"

    def test_rollback_to_self_is_refused(self, servable):
        """目标已是当前世代时不许"回滚到自身"（会让 previous 指向自己）。"""
        gid, _data = servable
        with _serve(gid, admin_token="tok") as client:
            status, _h, body = client.post(
                "/admin/rollback", {"generation_id": gid},
                headers={"Authorization": "Bearer tok"},
            )
            assert status == 409
            assert body["error"]["code"] == "rollback_refused"

    def test_rollback_rejects_malformed_generation_id(self, servable):
        gid, _data = servable
        with _serve(gid, admin_token="tok") as client:
            status, _h, body = client.post(
                "/admin/rollback", {"generation_id": "../etc/passwd"},
                headers={"Authorization": "Bearer tok"},
            )
            assert status == 400
            assert body["error"]["code"] == "bad_request"

    def test_evaluate_rejects_gold_set_outside_repo(self, servable):
        """`gold_set` 只允许仓库内路径，否则该接口变成任意文件读取器。"""
        gid, _data = servable
        with _serve(gid, admin_token="tok") as client:
            status, _h, body = client.post(
                "/admin/evaluate", {"limit": 5, "gold_set": "/etc/passwd"},
                headers={"Authorization": "Bearer tok"},
            )
            assert status == 400
            assert body["error"]["code"] == "bad_request"
            assert "仓库内" in body["error"]["message"]

    def test_rollback_is_audited(self, servable):
        """管理操作必须留痕：没有审计就等于"回滚了但没人知道是谁做的"。"""
        gid, data_dir = servable
        # 再造一代当回滚目标
        gen2 = release.Generation("g-20260101T000000Z-222222")
        gen2.create(
            snapshot=release.capture_inputs(
                chunks=CHUNKS, records=RECORDS, source_fp="corpusfp01234567",
                source_status="verified_zip",
            ),
            run_id="r-test",
        )
        gen2.fail("test_only", detail="只作为回滚失败的对象")
        with _serve(gid, admin_token="tok") as client:
            client.post(
                "/admin/rollback", {"generation_id": gen2.generation_id},
                headers={"Authorization": "Bearer tok"},
            )
        audit = data_dir / "audit" / "server-audit.jsonl"
        assert audit.exists()
        rows = [json.loads(x) for x in audit.read_text("utf-8").splitlines() if x.strip()]
        assert rows and rows[-1]["action"] == "rollback"
        assert rows[-1]["result"] == "failed"
        assert rows[-1]["actor"] == "admin"


# ---------------------------------------------------------------- 内部单元


class TestFitPayload:
    def test_under_limit_is_untouched(self):
        payload = {"a": 1, "hits": [1, 2, 3]}
        out, dropped = server._fit_payload(payload, 10000)
        assert out["hits"] == [1, 2, 3]
        assert dropped == []

    def test_over_limit_drops_in_documented_order(self):
        payload = {
            "context": {"units": ["x" * 400], "stats": {}},
            "citations": [{"c": "y" * 400}],
            "stats": {"channels": [{"c": "z" * 400}]},
            "hits": [{"h": 1}],
        }
        out, dropped = server._fit_payload(payload, 200)
        assert dropped, "超过上限必须至少丢一项"
        assert dropped[0] == "context.units"
        assert out["truncated"] == dropped
        assert out["context"]["units"] == []


class TestIntField:
    def test_default_when_absent(self):
        assert server._int_field({}, "top_k", 20, minimum=1, maximum=50) == 20

    def test_accepts_numeric_string(self):
        assert server._int_field({"top_k": "7"}, "top_k", 20, minimum=1, maximum=50) == 7

    def test_rejects_bool(self):
        with pytest.raises(server.ApiError) as exc:
            server._int_field({"top_k": False}, "top_k", 20, minimum=1, maximum=50)
        assert exc.value.status == 400
        assert exc.value.code == "bad_param"

    def test_rejects_out_of_range_without_clamping(self):
        with pytest.raises(server.ApiError) as exc:
            server._int_field({"top_k": 999}, "top_k", 20, minimum=1, maximum=50)
        assert exc.value.status == 400
        assert exc.value.code == "param_out_of_range"


class TestGatewayRows:
    def test_accepts_plain_list(self, tmp_path, monkeypatch):
        _isolate(monkeypatch, tmp_path)
        gen = release.Generation(GID)
        gen.dir.mkdir(parents=True)
        (gen.dir / "reports").mkdir()
        gen.gate_report_path.write_text(json.dumps([{"name": "L4.1", "verdict": "PASS"}]),
                                       encoding="utf-8")
        rows, problem = server._gate_rows(gen)
        assert problem == ""
        assert [r["name"] for r in rows] == ["L4.1"]

    def test_missing_report_is_a_problem_not_a_pass(self, tmp_path, monkeypatch):
        _isolate(monkeypatch, tmp_path)
        gen = release.Generation(GID)
        gen.dir.mkdir(parents=True)
        rows, problem = server._gate_rows(gen)
        assert rows == []
        assert "不存在" in problem


class TestVerificationCacheKey:
    def test_artifact_change_invalidates_cached_verdict(self, servable):
        """缓存的键必须包含产物状态：否则产物换了探针还说"逐项一致"。"""
        gid, _data = servable
        gen = release.Generation(gid)
        cache = server._VerificationCache()
        assert cache.get(gen) == []
        before_key = cache.key(gen)

        original = gen.index_path.read_bytes()
        gen.index_path.write_bytes(original + b"\n")
        try:
            assert cache.key(gen) != before_key, "产物变了缓存键必须变"
            assert cache.get(gen), "产物被改之后必须重新算出问题"
        finally:
            gen.index_path.write_bytes(original)
        assert cache.get(gen) == []

    def test_manifest_change_invalidates_cached_verdict(self, servable):
        """Manifest 改了（产物没改）也必须重算：同一代的 Manifest 会被改写。

        改的字段要选**校验真的会看的**那个。`chunk_count` 不行——它只被
        `/ready` 的 `manifest_complete` 当作"字段齐备"来判，`verify_generation`
        核的是索引自己的 `meta.chunk_count`；用它做这条用例会写出一个
        "缓存没失效也能过"的假断言。`assertion_count` 是逐行核对断言行数的，改它必红。
        """
        gid, _data = servable
        gen = release.Generation(gid)
        cache = server._VerificationCache()
        assert cache.get(gen) == []
        before_key = cache.key(gen)
        manifest = json.loads(gen.manifest_path.read_text("utf-8"))
        manifest["assertion_count"] = int(manifest["assertion_count"]) + 1
        gen.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        assert cache.key(gen) != before_key
        problems = cache.get(gen)
        assert problems, "Manifest 改了必须重新校验"
        assert any("assertion_count" in p for p in problems)
