# -*- coding: utf-8 -*-
"""`scripts/k2_gray.py` 的单元测试：判据函数 + 进程生命周期。

**不加载真实世代**（每代 1.7 GB、冷启动 ~70 s），全部用 `http.server` 造的
最小桩服务代替 `scripts/serve.py`。桩服务模仿真服务的关键契约：
`--port 0` 由系统分配、启动时打印「监听 http://…:PORT」、`/health`、`/ready`
返回 `generation_id`、`/release/current` 返回 `served_generation_id`、
`/chunks/<带上斜杠的 id>` 取块正文、`/query` 回显 `generation_id`。

被测模块用 `importlib.util.spec_from_file_location` 加载（与
`tests/test_k1_checks.py` 同款）——注意那种方式下 `sys.stdout` 是 pytest 的捕获对象，
这正是被测脚本里 `hasattr(sys.stdout, "reconfigure")` 守卫存在的原因。
"""
from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    path = ROOT / "scripts" / "k2_gray.py"
    spec = importlib.util.spec_from_file_location("k2_gray_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


K2 = _load_module()


# ---------------------------------------------------------------- 桩服务

STUB = '''# -*- coding: utf-8 -*-
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

gen = ""
port = 0
argv = sys.argv[1:]
for i, a in enumerate(argv):
    if a == "--generation":
        gen = argv[i + 1]
    elif a == "--port":
        port = int(argv[i + 1])

CHUNKS = {
    "chunk/one": "abcdefghij",
    "chunk/big": "abcdefghij",
}


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, payload):
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": True, "pid": 1})
        if self.path == "/ready":
            return self._send(200, {"ready": True, "generation_id": gen, "checks": []})
        if self.path == "/release/current":
            return self._send(200, {
                "current": {"generation_id": "g-prod-untouched"},
                "served_generation_id": gen,
                "serving_is_current": False,
            })
        if self.path.startswith("/chunks/"):
            cid = unquote(self.path[len("/chunks/"):])
            if cid in CHUNKS:
                return self._send(200, {
                    "chunk_id": cid,
                    "text": CHUNKS[cid],
                    "text_truncated": cid.endswith("big"),
                })
            return self._send(404, {"error": {"code": "chunk_not_found", "message": cid}})
        return self._send(404, {"error": {"code": "not_found", "message": self.path}})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        if self.path != "/query":
            return self._send(404, {"error": {"code": "not_found", "message": self.path}})
        question = str(json.loads(raw.decode("utf-8") or "{}").get("question") or "")
        if question.startswith("RATE"):
            return self._send(429, {"error": {"code": "rate_limited", "message": "太频繁"}})
        if question.startswith("ABSTAIN"):
            return self._send(200, {
                "generation_id": gen,
                "answer": {"abstained": True, "abstain_reason": "no_supported_claim", "text": ""},
                "context": {"units": []},
                "hits": [],
            })
        if question.startswith("MISMATCH"):
            cite = {"chunk_id": "chunk/one", "evidence": "ZZZ", "start": 2, "end": 5}
        elif question.startswith("OUTSIDE"):
            cite = {"chunk_id": "chunk/one", "evidence": "cde", "start": 2, "end": 5}
        elif question.startswith("TRUNC"):
            cite = {"chunk_id": "chunk/big", "evidence": "cde", "start": 2, "end": 5}
        else:
            cite = {"chunk_id": "chunk/one", "evidence": "cde", "start": 2, "end": 5}
        # OUTSIDE：命中块里**没有**被引用那个块，用来触发"引到命中块外"。
        units = [{"chunk_id": "chunk/other"}] if question.startswith("OUTSIDE") \
            else [{"chunk_id": "chunk/one"}]
        return self._send(200, {
            "generation_id": gen,
            "answer": {"abstained": False, "text": "abc", "citations": [cite]},
            "context": {"units": units},
            "hits": [],
        })


httpd = ThreadingHTTPServer(("127.0.0.1", port), H)
print("[i] 监听 http://127.0.0.1:{}".format(httpd.server_address[1]), flush=True)
httpd.serve_forever()
'''

BROKEN = '''# -*- coding: utf-8 -*-
import sys
sys.stderr.write("[2] 世代不可服务：graph_schema=ark-rag-1\\n")
raise SystemExit(2)
'''


@pytest.fixture()
def stub_script(tmp_path):
    path = tmp_path / "stub_serve.py"
    path.write_text(STUB, encoding="utf-8")
    return path


@pytest.fixture()
def stub(stub_script):
    """起一个桩服务实例；无论如何都回收，测试自己不许留孤儿。"""
    proc = K2.ServerProc("g-test-candidate", argv=[
        sys.executable, "-u", str(stub_script),
        "--generation", "g-test-candidate", "--host", "127.0.0.1", "--port", "0",
    ])
    proc.start()
    proc.wait_ready()
    try:
        yield proc
    finally:
        proc.stop()


def _rec(gen_echo="g-x", **over):
    base = {
        "http_status": 200, "error": None, "answered": True, "abstained": False,
        "abstain_reason": None, "citations": 1, "cite_failures": [], "cite_outside_hits": [],
        "cite_unverifiable": [], "hit_chunk_ids": ["chunk/one"], "gold_chunk_ids": ["chunk/one"],
        "gold_hit": True, "gen_echo": gen_echo, "ms": 5,
    }
    base.update(over)
    return base


def _item(qid="q-x-000001", cand=None, base=None, **over):
    row = {
        "question_id": qid, "category": "single_chunk", "should_abstain": False,
        "question": "问", "gold_chunk_ids": ["chunk/one"], "hit_chunk_ids": ["chunk/one"],
        "candidate": cand or _rec(), "baseline": base or _rec(),
    }
    row.update(over)
    return row


# ---------------------------------------------------------------- 内存与模式

def test_memory_status_shape():
    mem = K2.memory_status()
    for key in ("avail_mib", "total_mib", "load_percent", "source"):
        assert key in mem
    if sys.platform == "win32":
        assert mem["source"] == "GlobalMemoryStatusEx"
        assert isinstance(mem["avail_mib"], int) and mem["avail_mib"] > 0


@pytest.mark.parametrize("avail,expected", [
    (6144, "concurrent"),
    (16986, "concurrent"),
    (6143, "sequential"),
    (0, "sequential"),
    (None, "sequential"),
])
def test_pick_mode_threshold(avail, expected):
    assert K2.pick_mode(avail) == expected


def test_concurrent_threshold_is_six_gib():
    # 门槛不许被悄悄调低成"永远并行"。
    assert K2.CONCURRENT_MIN_AVAIL_MIB == 6144


# ---------------------------------------------------------------- 判据：探针集

def test_check_probe_set_passes_on_exact_set():
    gold = set(K2.PROBE_IDS)
    outcome = K2.check_probe_set(K2.PROBE_IDS, gold, list(K2.PROBE_IDS))
    assert outcome["ok"] is True
    assert outcome["data"]["ran"] == K2.EXPECTED_PROBE_COUNT


def test_check_probe_set_fails_when_a_probe_disappears():
    """题变少了必须红——否则"跑 1 题全过"和"跑 20 题全过"没法区分。"""
    gold = set(K2.PROBE_IDS)
    ran = list(K2.PROBE_IDS)[:-1]
    outcome = K2.check_probe_set(K2.PROBE_IDS, gold, ran)
    assert outcome["ok"] is False
    assert outcome["data"]["missing"] == [K2.PROBE_IDS[-1]]


def test_check_probe_set_fails_on_duplicate_and_unknown():
    gold = set(K2.PROBE_IDS)
    ran = list(K2.PROBE_IDS)[:-1] + [K2.PROBE_IDS[0]]
    dup = K2.check_probe_set(K2.PROBE_IDS, gold, ran)
    assert dup["ok"] is False and dup["data"]["duplicates"]
    other = K2.check_probe_set(K2.PROBE_IDS, gold, list(K2.PROBE_IDS) + ["q-zz-999999"])
    assert other["ok"] is False and other["data"]["extra"] == ["q-zz-999999"]


def test_check_probe_set_fails_when_probe_not_in_gold():
    gold = set(K2.PROBE_IDS) - {K2.PROBE_IDS[3]}
    outcome = K2.check_probe_set(K2.PROBE_IDS, gold, list(K2.PROBE_IDS))
    assert outcome["ok"] is False
    assert outcome["data"]["not_in_gold"] == [K2.PROBE_IDS[3]]


def test_check_probe_set_fails_when_planned_set_is_truncated():
    """写死题集被删掉一题也要红（EXPECTED_PROBE_COUNT 是独立字面量）。"""
    trimmed = tuple(K2.PROBE_IDS[:-1])
    outcome = K2.check_probe_set(trimmed, set(K2.PROBE_IDS), list(trimmed))
    assert outcome["ok"] is False
    assert "≠ 期望" in outcome["detail"]


def test_probe_ids_are_unique_and_match_expected_count():
    assert len(set(K2.PROBE_IDS)) == len(K2.PROBE_IDS)
    assert len(K2.PROBE_IDS) == K2.EXPECTED_PROBE_COUNT == 20


# ---------------------------------------------------------------- 判据：HTTP 200

def test_check_probes_answered_passes():
    outcome = K2.check_probes_answered([_item(), _item("q-y-000001")])
    assert outcome["ok"] is True
    assert outcome["data"]["total"] == 4


@pytest.mark.parametrize("status", [429, 500, 503, 504, 0])
def test_check_probes_answered_fails_on_non_200(status):
    cand = _rec(http_status=status, error="x")
    outcome = K2.check_probes_answered([_item(cand=cand)])
    assert outcome["ok"] is False
    assert outcome["data"]["bad"][0]["http_status"] == status


def test_check_probes_answered_fails_on_empty_run():
    outcome = K2.check_probes_answered([])
    assert outcome["ok"] is False


# ---------------------------------------------------------------- 判据：世代标识

def _ident(**over):
    rec = {
        "role": "candidate", "requested": "g-new", "ready_gid": "g-new",
        "served_gid": "g-new", "query_echoes": ["g-new", "g-new"],
    }
    rec.update(over)
    return rec


def test_check_generation_identity_passes():
    outcome = K2.check_generation_identity([
        _ident(),
        _ident(role="baseline", requested="g-old", ready_gid="g-old",
               served_gid="g-old", query_echoes=["g-old"]),
    ])
    assert outcome["ok"] is True


def test_check_generation_identity_fails_when_ready_reports_other_generation():
    """"我以为是候选在服务，其实是旧代"——最隐蔽的那种错。"""
    outcome = K2.check_generation_identity([_ident(ready_gid="g-prod-untouched")])
    assert outcome["ok"] is False
    assert "ready" in outcome["detail"]


def test_check_generation_identity_fails_when_served_gid_differs():
    outcome = K2.check_generation_identity([_ident(served_gid="g-prod-untouched")])
    assert outcome["ok"] is False


def test_check_generation_identity_fails_when_a_query_echo_is_wrong():
    outcome = K2.check_generation_identity([_ident(query_echoes=["g-new", "g-prod-untouched"])])
    assert outcome["ok"] is False
    assert "回显" in outcome["detail"]


def test_check_generation_identity_fails_without_any_echo():
    outcome = K2.check_generation_identity([_ident(query_echoes=[])])
    assert outcome["ok"] is False


def test_check_generation_identity_fails_on_empty_records():
    assert K2.check_generation_identity([])["ok"] is False


# ---------------------------------------------------------------- 判据：拒答翻转

def test_check_abstain_flip_passes_when_both_agree():
    cand = _rec(abstained=True, answered=False)
    base = _rec(abstained=True, answered=False)
    outcome = K2.check_abstain_flip([_item(cand=cand, base=base)])
    assert outcome["ok"] is True
    assert outcome["data"]["flip_count"] == 0


@pytest.mark.parametrize("ca,ba,direction", [
    (True, False, "该答变该拒"),
    (False, True, "该拒变该答"),
])
def test_check_abstain_flip_fails_on_either_direction(ca, ba, direction):
    outcome = K2.check_abstain_flip([_item(cand=_rec(abstained=ca), base=_rec(abstained=ba))])
    assert outcome["ok"] is False
    assert outcome["data"]["flips"][0]["direction"] == direction


def test_check_abstain_flip_records_gold_expectation_mismatch_without_failing():
    """Gold 说该拒却答了：只记录，不设门（两代一样说明不是灰度回归）。"""
    item = _item(should_abstain=True,
                 cand=_rec(abstained=False), base=_rec(abstained=False))
    outcome = K2.check_abstain_flip([item])
    assert outcome["ok"] is True
    assert outcome["data"]["answered_but_should_abstain"]["candidate"] == ["q-x-000001"]


def test_check_abstain_flip_skips_non_200():
    outcome = K2.check_abstain_flip([
        _item(cand=_rec(http_status=504, abstained=True), base=_rec(abstained=False)),
    ])
    assert outcome["ok"] is True  # 非 200 由 probes_answered 判，这里不重复判


# ---------------------------------------------------------------- 判据：引用定位

def test_check_citation_locate_passes():
    outcome = K2.check_citation_locate([_item()])
    assert outcome["ok"] is True
    assert outcome["data"]["citations_total"] == 2


def test_check_citation_locate_fails_when_offsets_do_not_match():
    bad = _rec(cite_failures=[{"chunk_id": "chunk/one", "reason": "evidence_does_not_match_offsets"}])
    outcome = K2.check_citation_locate([_item(cand=bad)])
    assert outcome["ok"] is False
    assert outcome["data"]["unlocatable"][0]["role"] == "candidate"


def test_check_citation_locate_fails_when_citing_outside_hits():
    bad = _rec(cite_outside_hits=[{"chunk_id": "chunk/one", "reason": "cited_chunk_not_in_hits"}])
    outcome = K2.check_citation_locate([_item(base=bad)])
    assert outcome["ok"] is False
    assert outcome["data"]["outside_hits"][0]["role"] == "baseline"


def test_check_citation_locate_records_truncated_chunk_as_unverifiable():
    rec = _rec(cite_unverifiable=[{"chunk_id": "chunk/big", "reason": "chunk_text_truncated"}])
    outcome = K2.check_citation_locate([_item(cand=rec)])
    assert outcome["ok"] is True
    assert outcome["data"]["unverifiable"][0]["reason"] == "chunk_text_truncated"


# ---------------------------------------------------------------- 进程管理

def test_stub_server_assigns_port_and_reports_ready(stub):
    assert stub.port > 0                       # --port 0 由系统分配
    assert stub.base == f"http://127.0.0.1:{stub.port}"
    assert stub.ready_payload["generation_id"] == "g-test-candidate"
    status, body = stub.get("/health")
    assert status == 200 and body["ok"] is True
    status, body = stub.get("/release/current")
    assert body["served_generation_id"] == "g-test-candidate"


def test_stop_reaps_the_process(stub_script):
    proc = K2.ServerProc("g-test-candidate", argv=[
        sys.executable, "-u", str(stub_script),
        "--generation", "g-test-candidate", "--port", "0",
    ])
    proc.start()
    handle = proc.proc
    assert handle is not None and handle.poll() is None
    proc.stop()
    assert proc.proc is None
    assert handle.poll() is not None           # 真的退了，没留孤儿
    assert proc not in K2._LIVE
    proc.stop()                                # 二次 stop 是空操作，不抛


def test_stop_after_still_reads_peak_rss(stub_script):
    proc = K2.ServerProc("g-test-candidate", argv=[
        sys.executable, "-u", str(stub_script), "--generation", "g-test-candidate", "--port", "0",
    ])
    proc.start()
    try:
        assert proc.peak_rss_mib() is not None if sys.platform == "win32" else True
    finally:
        proc.stop()
    # stop() 把 proc 置空，但 pid 还在，所以峰值 RSS 仍可读（Windows 上句柄有效期内）。
    if sys.platform == "win32":
        assert proc.pid is not None


def test_chunk_path_percent_encodes_slashes():
    assert K2.chunk_path("a/b/c") == "/chunks/a%2Fb%2Fc"
    assert K2.chunk_path("t1/main#3") == "/chunks/t1%2Fmain%233"


def test_http_call_raises_serve_error_on_dead_port():
    """连不通必须抛 `ServeError`，且消息里要有方法+路径——否则调用方分不清
    是「进程死了」还是「服务端拒了这个请求」。"""
    with pytest.raises(K2.ServeError) as exc:
        K2.http_call("http://127.0.0.1:1", "GET", "/health", timeout=2.0)
    msg = str(exc.value)
    assert "传输层失败" in msg, msg
    assert "GET /health" in msg, msg


def test_start_raises_serve_error_when_process_exits_early(tmp_path):
    boom = tmp_path / "boom.py"
    boom.write_text("raise SystemExit(2)\n", encoding="utf-8")
    proc = K2.ServerProc("g-nope", argv=[sys.executable, "-u", str(boom), "--port", "0"],
                         boot_timeout_s=10.0)
    with pytest.raises(K2.ServeError) as exc:
        proc.start()
    assert "提前退出" in str(exc.value)
    assert proc not in K2._LIVE


def test_wait_ready_times_out_when_ready_never_true(tmp_path):
    """桩里 /ready 永远 503 —— 必须抛，不许把它当就绪。"""
    script = tmp_path / "never_ready.py"
    script.write_text(STUB.replace('return self._send(200, {"ready": True,',
                                   'return self._send(503, {"ready": False,'),
                        encoding="utf-8")
    proc = K2.ServerProc("g-never", argv=[sys.executable, "-u", str(script), "--port", "0"],
                         ready_timeout_s=1.0)
    proc.start()
    try:
        with pytest.raises(K2.ServeError) as exc:
            proc.wait_ready()
        assert "没就绪" in str(exc.value)
    finally:
        proc.stop()


def test_wait_ready_fails_fast_when_verdict_is_computed_not_ready(tmp_path):
    """503 且带 checks 明细 = 就绪判定已经算完并判否 → 立刻退出码 2，不干等超时。

    这条对应真实事故路径：graph_schema=ark-rag-1 的世代 manifest 缺字段，
    /ready 会稳定回 503。等满 READY_TIMEOUT_S 只是浪费时间。
    """
    script = tmp_path / "computed_no.py"
    script.write_text(
        STUB.replace(
            'return self._send(200, {"ready": True, "generation_id": gen, "checks": []})',
            'return self._send(503, {"ready": False, "generation_id": gen, "checks": ['
            '{"id": "manifest_complete", "ok": False, "blocking": True,'
            ' "detail": "缺字段：graph_schema"}]})',
        ),
        encoding="utf-8",
    )
    proc = K2.ServerProc("g-bad", argv=[sys.executable, "-u", str(script), "--port", "0"],
                         ready_timeout_s=120.0)
    proc.start()
    t0 = time.time()
    try:
        with pytest.raises(K2.ServeError) as exc:
            proc.wait_ready()
        assert "不可服务" in str(exc.value)
        assert "manifest_complete" in str(exc.value)
        assert time.time() - t0 < 30.0       # 远小于 ready_timeout_s
    finally:
        proc.stop()


# ---------------------------------------------------------------- 端到端（桩）

def test_probe_one_records_answer_citations_and_gold_hit(stub):
    item = {"question": "正常问题", "gold_chunk_ids": ["chunk/one"]}
    rec = K2.probe_one(stub, item, {})
    assert rec["http_status"] == 200
    assert rec["abstained"] is False and rec["answered"] is True
    assert rec["citations"] == 1 and rec["cite_ok"] == 1
    assert rec["cite_failures"] == [] and rec["cite_outside_hits"] == []
    assert rec["gold_hit"] is True
    assert rec["gen_echo"] == "g-test-candidate"


def test_probe_one_detects_evidence_mismatch(stub):
    rec = K2.probe_one(stub, {"question": "MISMATCH 题"}, {})
    assert rec["cite_failures"][0]["reason"] == "evidence_does_not_match_offsets"
    assert K2.check_citation_locate([_item(cand=rec)])["ok"] is False


def test_probe_one_detects_citation_outside_hits(stub):
    rec = K2.probe_one(stub, {"question": "OUTSIDE 题"}, {})
    assert rec["cite_outside_hits"][0]["reason"] == "cited_chunk_not_in_hits"
    assert K2.check_citation_locate([_item(cand=rec)])["ok"] is False


def test_probe_one_marks_truncated_chunk_unverifiable(stub):
    rec = K2.probe_one(stub, {"question": "TRUNC 题"}, {})
    assert rec["cite_unverifiable"][0]["reason"] == "chunk_text_truncated"
    assert rec["cite_failures"] == []


def test_probe_one_records_abstain(stub):
    rec = K2.probe_one(stub, {"question": "ABSTAIN 题"}, {})
    assert rec["abstained"] is True and rec["answered"] is False
    assert rec["abstain_reason"] == "no_supported_claim"


def test_probe_one_surfaces_http_error_without_raising(stub):
    rec = K2.probe_one(stub, {"question": "RATE 题"}, {})
    assert rec["http_status"] == 429
    assert "rate_limited" in rec["error"]
    assert K2.check_probes_answered([_item(cand=rec)])["ok"] is False


def test_collect_probes_pairs_both_generations(stub, stub_script):
    """两代交错采集：每题两个 role 都在，title 顺序跟 pattern 走。"""
    other = K2.ServerProc("g-test-baseline", argv=[
        sys.executable, "-u", str(stub_script), "--generation", "g-test-baseline", "--port", "0",
    ])
    other.start()
    other.wait_ready()
    try:
        items = {qid: {"question": "正常问题", "category": "single_chunk",
                       "should_abstain": False, "gold_chunk_ids": ["chunk/one"]}
                 for qid in ("q-a", "q-b")}
        rows = K2.collect_probes({"candidate": stub, "baseline": other}, items, ["q-a", "q-b"])
        assert [r["question_id"] for r in rows] == ["q-a", "q-b"]
        for row in rows:
            assert row["candidate"]["gen_echo"] == "g-test-candidate"
            assert row["baseline"]["gen_echo"] == "g-test-baseline"
    finally:
        other.stop()


# ---------------------------------------------------------------- 退出码

def test_main_returns_2_when_generation_cannot_be_served(tmp_path, monkeypatch):
    """服务起不来必须 2，不是 0 —— 这是最容易写成假绿的那条路径。"""
    broken = tmp_path / "broken_serve.py"
    broken.write_text(BROKEN, encoding="utf-8")
    monkeypatch.setattr(K2, "SERVE", broken)
    monkeypatch.setattr(K2, "OUT", tmp_path / "k2_gray.json")
    assert K2.main([]) == 2
    assert not (tmp_path / "k2_gray.json").exists()


def test_main_returns_2_when_probe_set_missing_from_gold(tmp_path, monkeypatch):
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps({"schema": "x", "items": []}), encoding="utf-8")
    monkeypatch.setattr(K2, "GOLD", gold)
    assert K2.main([]) == 2


def test_load_gold_reports_unreadable_file(tmp_path, monkeypatch):
    """Gold Set 读不出来必须抛，不许当成「没有探针」静默继续。"""
    missing = tmp_path / "nope.json"
    monkeypatch.setattr(K2, "GOLD", missing)
    with pytest.raises(K2.ServeError) as exc:
        K2._load_gold()
    msg = str(exc.value)
    assert "Gold Set 读不出来" in msg, msg
    assert "nope.json" in msg, msg


def test_main_verdict_exit_code_mapping():
    """0=全过 / 1=有判据不成立：这条规则本身要有测试钉住。"""
    src = (ROOT / "scripts" / "k2_gray.py").read_text(encoding="utf-8")
    assert "return 0 if all_passed else 1" in src
    assert "all_passed = not bad and len(results) == 5" in src
