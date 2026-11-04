"""`scripts/k2_publish_check.py` 的自检：**每条判据都必须能失败**。

这道门的价值不在「六条里五条 PASS」，而在「条件不成立时会变红」。一个永远返回
`ok=True` 的清单门比没有门更糟：它会让人以为「发布前该查的都查过了」，而实际
什么都没查——这正是当初漏掉这道门时的那种自信。

所以这里对每条判据各构造一次「本该拦住」的输入，断言它确实不成立，并且 `detail`
说的是那件事；同时留**反向对照**（同样输入只改一个变量就变绿），证明它不是「总是红」。

## 为什么用合成数据

真跑一遍要起两代产物、跑五道门（性能要跑十几分钟），而且报告的世代 id 得指到真实
世代上。判据本身是纯函数（吃字典、出结论），所以这里把输入换掉让它自己下结论。
它证明的是「判据写对了」，不证明「盘上现在是对的」——后者由手工跑一次的实测输出负责。

## 三条不能走捷径的地方

1. `undetermined` 必须与原样输出比对：三份报告本来就没有 `generated_at` 字段，
   对它们只能「无法判定」。**断言它们「通过」就是把已知的不知道写成知道**，
   所以这里断言的是 `verdict == "undetermined"` 且 detail 里出现「无法判定」。
2. `reports_bind_generation` 的 red 输出必须同时出现「要发布的是」与两个 id——
   只报「不一致」而不报「记的是谁、要的是谁」，排查时等于没说。
3. `all_passed` 只统计 FAIL：`undetermined` 既不算通过也不算失败。这条要有专门用例，
   否则「无法判定」要么被偷偷算成通过、要么把整道门拖红，两种都是错。

## 与卫生门的关系

本文件进入 `tests/test_suite_hygiene.py` 的 `TEST_FILES` 后，四个 parametrize 各多一项。
`N` 由 Lead 统一上调，本文件不改卫生门。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import release  # noqa: E402


def _load_publish_check():
    spec = importlib.util.spec_from_file_location(
        "k2_publish_check_under_test", ROOT / "scripts" / "k2_publish_check.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PC = _load_publish_check()

CAND = "g-20261103T132152Z-06a741"
PREV = "g-20261031T035231Z-4eafae"
CREATED = datetime(2026, 11, 3, 13, 21, 52, tzinfo=timezone.utc)


# ---------------------------------------------------------------- 夹具
#
# 形状照抄真实报告：五份各自的世代字段不同（这张表就是脚本里 REPORTS 的意义所在），
# 只有 k2_perf / k2_gray 带 generated_at。


def _five_specs(tmp_path: Path) -> list[dict]:
    """五份报告的登记表，路径落在 tmp_path 里。字段名与真实报告一致。"""
    return [
        {"key": "k2_hard_gates", "label": "硬门", "path": str(tmp_path / "k2_hard_gates.json"),
         "gid_field": "generation_id", "producer": "py -3 scripts/k2_hard_gates.py"},
        {"key": "k2_perf", "label": "性能", "path": str(tmp_path / "k2_perf.json"),
         "gid_field": "generation_id", "producer": "py -3 scripts/k2_perf.py"},
        {"key": "k2_gray", "label": "灰度", "path": str(tmp_path / "k2_gray.json"),
         "gid_field": "generation_id", "producer": "py -3 scripts/k2_gray.py"},
        {"key": "k3_threshold_gate", "label": "阈值审批",
         "path": str(tmp_path / "k3_threshold_gate.json"),
         "gid_field": "generations.new", "producer": "py -3 scripts/k3_thresholds.py"},
        {"key": "k4_drill", "label": "K4 演练", "path": str(tmp_path / "k4_drill.json"),
         "gid_field": "candidate", "producer": "py -3 scripts/k4_drill.py"},
    ]


def _write_reports(
    tmp_path: Path,
    *,
    gid: str = CAND,
    perf_gid: str | None = None,
    all_passed: bool = True,
    skip: str = "",
    generated_at: str = "2026-11-04T07:14:40Z",
) -> list[dict]:
    """写出五份「正常」报告；`skip` 指定哪一份不写，`perf_gid` 单独改性能报告的世代。"""
    specs = _five_specs(tmp_path)
    payloads = {
        "k2_hard_gates": {"generation_id": gid, "all_passed": all_passed},
        "k2_perf": {
            "generation_id": perf_gid or gid,
            "all_passed": all_passed,
            "generated_at": generated_at,
        },
        "k2_gray": {"generation_id": gid, "all_passed": all_passed, "generated_at": generated_at},
        "k3_threshold_gate": {"generations": {"new": gid, "old": PREV}, "all_passed": all_passed},
        "k4_drill": {"candidate": gid, "baseline": PREV, "all_passed": all_passed},
    }
    for spec in specs:
        if spec["key"] == skip:
            continue
        Path(spec["path"]).write_text(
            json.dumps(payloads[spec["key"]], ensure_ascii=False), encoding="utf-8"
        )
    return specs


def _verdicts(items: list[dict]) -> dict[str, str]:
    return {item["id"]: item["verdict"] for item in items}


def _by_id(items: list[dict], check_id: str) -> dict:
    for item in items:
        if item["id"] == check_id:
            return item
    raise AssertionError(f"没有这条判据：{check_id}（实际有 {[i['id'] for i in items]}）")


# ---------------------------------------------------------------- 判据一：报告在场


def test_all_five_present_passes(tmp_path):
    records = PC.load_reports(_write_reports(tmp_path))
    verdict, detail, data = PC.check_reports_present(records)
    assert verdict == "pass"
    assert data["have"] == 5 and data["total"] == 5
    assert "5/5" in detail


def test_missing_report_fails_and_names_the_file_and_producer(tmp_path):
    records = PC.load_reports(_write_reports(tmp_path, skip="k2_gray"))
    verdict, detail, data = PC.check_reports_present(records)
    assert verdict == "fail"
    assert data["have"] == 4
    assert "k2_gray.json" in detail, "点不出缺的是哪份，等于没说"
    assert "py -3 scripts/k2_gray.py" in detail, "不告诉人该跑什么，这道门只是骂人"
    assert len(data["missing"]) == 1


def test_corrupt_json_report_fails(tmp_path):
    specs = _write_reports(tmp_path)
    (tmp_path / "k4_drill.json").write_text("{ 这不是 JSON", encoding="utf-8")
    records = PC.load_reports(specs)
    verdict, detail, data = PC.check_reports_present(records)
    assert verdict == "fail"
    assert len(data["broken"]) == 1
    assert "k4_drill.json" in detail
    assert "坏 1 份" in detail


# ---------------------------------------------------------------- 判据二：各自 all_passed


def test_all_passed_true_passes(tmp_path):
    records = PC.load_reports(_write_reports(tmp_path))
    verdict, detail, data = PC.check_reports_passed(records)
    assert verdict == "pass"
    assert data["judged"] == 5 and data["passed"] == 5
    assert "5/5" in detail


def test_one_report_false_fails_and_names_it(tmp_path):
    specs = _write_reports(tmp_path)
    # 只把灰度那份改成 False，其余四份仍 True——证明这道门不是「看一眼总数」
    gray = tmp_path / "k2_gray.json"
    payload = json.loads(gray.read_text("utf-8"))
    payload["all_passed"] = False
    gray.write_text(json.dumps(payload), encoding="utf-8")
    records = PC.load_reports(specs)
    verdict, detail, data = PC.check_reports_passed(records)
    assert verdict == "fail"
    assert data["passed"] == 4
    assert "k2_gray.json" in detail
    assert data["not_true"][0].endswith("all_passed=False）")


def test_missing_field_is_not_treated_as_pass(tmp_path):
    """报告里没有 `all_passed` 字段时，不许当成 True。

    `None is not True` 这件事必须成立——否则一份只写了 `{"generation_id": ...}` 的
    空报告会被当通过。
    """
    specs = _five_specs(tmp_path)
    (tmp_path / "k4_drill.json").write_text(json.dumps({"candidate": CAND}), encoding="utf-8")
    records = PC.load_reports(specs)
    verdict, detail, data = PC.check_reports_passed(records)
    assert verdict == "fail"
    assert "k4_drill.json" in detail
    assert "all_passed=None" in detail


# ---------------------------------------------------------------- 判据三：绑的是同一代


def test_all_bound_to_candidate_passes(tmp_path):
    records = PC.load_reports(_write_reports(tmp_path))
    verdict, detail, data = PC.check_reports_bind(records, CAND)
    assert verdict == "pass"
    assert data["bound"] == 5
    assert "5/5" in detail


def test_report_bound_to_previous_generation_fails_and_shows_both_ids(tmp_path):
    """这条是「拿旧报告凑数」的专用拦截：报告都在、都过，但绑的是上一代。"""
    records = PC.load_reports(_write_reports(tmp_path, perf_gid=PREV))
    verdict, detail, data = PC.check_reports_bind(records, CAND)
    assert verdict == "fail"
    assert data["bound"] == 4
    assert "要发布的是" in detail, "必须说清要发布的是谁"
    assert PREV in detail and CAND in detail, "两个 id 都要出现，否则看不出差在哪"
    assert "generation_id" in detail


def test_three_different_gid_fields_are_all_read_correctly(tmp_path):
    """三份用非 `generation_id` 字段的报告（`generations.new` / `candidate`）也要读对。

    这是这张登记表存在的全部理由：把它们当成 `generation_id` 读会得到 `None`，
    于是「五份都没绑对」——门红得有道理，但红错了原因。
    """
    records = PC.load_reports(_write_reports(tmp_path))
    mapping = {rec["key"]: rec["generation_id"] for rec in records}
    assert mapping["k3_threshold_gate"] == CAND
    assert mapping["k4_drill"] == CAND
    assert mapping["k2_hard_gates"] == CAND
    verdict, detail, data = PC.check_reports_bind(records, CAND)
    assert verdict == "pass"
    assert data["bound"] == 5


# ---------------------------------------------------------------- 判据四：候选状态


def test_candidate_ready_passes():
    verdict, detail, data = PC.check_candidate_status("READY", CAND)
    assert verdict == "pass"
    assert "READY" in detail


@pytest.mark.parametrize("status", ["FAILED", "BUILDING", "VALIDATING", "PUBLISHED", "", None])
def test_candidate_not_ready_fails(status):
    verdict, detail, data = PC.check_candidate_status(status, CAND)
    assert verdict == "fail"
    assert str(status) in detail or repr(status) in detail
    assert data["want"] == "READY"


# ---------------------------------------------------------------- 判据五：候选自检


def test_candidate_verify_clean_passes():
    verdict, detail, data = PC.check_candidate_verify([], CAND)
    assert verdict == "pass"
    assert "0 项不符" in detail


def test_candidate_verify_problems_fails_and_lists_them():
    problems = ["graph_artifact_id 不符：Manifest=a 实际=b", "index/lexical.json 缺失"]
    verdict, detail, data = PC.check_candidate_verify(problems, CAND)
    assert verdict == "fail"
    assert "2 项不符" in detail
    assert "graph_artifact_id" in detail
    assert len(data["problems"]) == 2


def test_candidate_verify_truncates_but_reports_the_total():
    problems = [f"问题{i}" for i in range(12)]
    verdict, detail, _ = PC.check_candidate_verify(problems, CAND)
    assert verdict == "fail"
    assert "共 12 项" in detail, "截断时必须报总数，否则看起来只坏了 8 处"


# ---------------------------------------------------------------- 判据六：报告不比候选老


def test_fresh_reports_pass_when_all_have_timestamps(tmp_path):
    """反向对照：给三份「本来没时间戳」的报告也补上时间戳，这条就该从
    `undetermined` 变 `pass`——证明它不是恒为「无法判定」。"""
    specs = _write_reports(tmp_path)
    stamp = "2026-11-04T07:14:40Z"
    for name, key in (
        ("k2_hard_gates.json", "generation_id"),
        ("k4_drill.json", "candidate"),
    ):
        path = tmp_path / name
        payload = json.loads(path.read_text("utf-8"))
        payload["generated_at"] = stamp
        path.write_text(json.dumps(payload), encoding="utf-8")
    threshold = tmp_path / "k3_threshold_gate.json"
    payload = json.loads(threshold.read_text("utf-8"))
    payload["generated_at"] = stamp
    threshold.write_text(json.dumps(payload), encoding="utf-8")

    records = PC.load_reports(specs)
    verdict, detail, data = PC.check_reports_freshness(records, CREATED)
    assert verdict == "pass"
    assert data["checked"] == 5
    assert not data["undetermined"]


def test_three_reports_without_timestamp_are_undetermined_not_pass(tmp_path):
    """三份报告本来就没有 `generated_at`。这条必须是 `undetermined`。

    断言「通过」等于把不知道写成知道；断言「失败」等于凭空白拉红。两者都不对。
    """
    records = PC.load_reports(_write_reports(tmp_path))
    verdict, detail, data = PC.check_reports_freshness(records, CREATED)
    assert verdict == "undetermined"
    assert "无法判定" in detail
    assert data["checked"] == 2, "只有 k2_perf 与 k2_gray 带时间戳"
    assert len(data["undetermined"]) == 3
    assert any("k4_drill.json" in item for item in data["undetermined"])


def test_report_generated_before_candidate_fails(tmp_path):
    records = PC.load_reports(
        _write_reports(tmp_path, generated_at="2026-11-01T00:00:00Z")
    )
    verdict, detail, data = PC.check_reports_freshness(records, CREATED)
    assert verdict == "fail"
    assert len(data["stale"]) == 2
    assert "早于候选建成" in detail


def test_unparseable_timestamp_is_undetermined_not_pass(tmp_path):
    records = PC.load_reports(_write_reports(tmp_path, generated_at="昨天下午"))
    verdict, detail, data = PC.check_reports_freshness(records, CREATED)
    assert verdict == "undetermined"
    assert data["checked"] == 0
    assert len(data["undetermined"]) == 5


def test_missing_candidate_created_at_makes_it_undetermined(tmp_path):
    records = PC.load_reports(_write_reports(tmp_path))
    verdict, detail, data = PC.check_reports_freshness(records, None)
    assert verdict == "undetermined"
    assert data["checked"] == 0


# ---------------------------------------------------------------- 汇总：三态怎么算


def test_evaluate_all_green(tmp_path):
    """五份齐全 + 全过 + 全绑对 + 候选 READY 且自检干净 → 只有时间戳那条无法判定。

    `all_passed` 必须为 True：`undetermined` 不参与失败统计。
    """
    records = PC.load_reports(_write_reports(tmp_path))
    items = PC.evaluate(CAND, records, status="READY", problems=[], created_at=CREATED)
    verdicts = _verdicts(items)
    assert verdicts["reports_present"] == "pass"
    assert verdicts["reports_passed"] == "pass"
    assert verdicts["reports_bind_generation"] == "pass"
    assert verdicts["candidate_ready"] == "pass"
    assert verdicts["candidate_verify"] == "pass"
    assert verdicts["reports_not_stale"] == "undetermined"
    assert not [item for item in items if item["verdict"] == "fail"]
    assert _by_id(items, "reports_not_stale")["ok"] is False, (
        "无法判定的条目 ok 必须是 False——它不算通过"
    )


def test_evaluate_any_fail_makes_the_gate_red(tmp_path):
    records = PC.load_reports(_write_reports(tmp_path, perf_gid=PREV))
    items = PC.evaluate(CAND, records, status="READY", problems=[], created_at=CREATED)
    failed = [item["id"] for item in items if item["verdict"] == "fail"]
    assert failed == ["reports_bind_generation"]


def test_evaluate_reports_all_six_checks_always(tmp_path):
    """六条判据永远都在：不许因为前一条红了就少判后面的。

    少判一条的结果是「修完这个再看下一个」，而发布清单必须在一次里说清全部问题。
    """
    records = PC.load_reports(_write_reports(tmp_path, skip="k2_perf", all_passed=False))
    items = PC.evaluate(CAND, records, status="FAILED", problems=["坏了"], created_at=CREATED)
    assert len(items) == 6
    assert {item["id"] for item in items} == {
        "reports_present",
        "reports_passed",
        "reports_bind_generation",
        "candidate_ready",
        "candidate_verify",
        "reports_not_stale",
    }
    verdicts = _verdicts(items)
    assert verdicts["reports_present"] == "fail"
    assert verdicts["candidate_ready"] == "fail"
    assert verdicts["candidate_verify"] == "fail"


# ---------------------------------------------------------------- 工具函数


def test_dig_reads_dotted_path():
    payload = {"generations": {"new": "g-1", "old": "g-0"}, "candidate": "g-2"}
    assert PC.dig(payload, "generations.new") == "g-1"
    assert PC.dig(payload, "candidate") == "g-2"
    assert PC.dig(payload, "generations.missing") is None
    assert PC.dig(payload, "nope.deeper") is None


def test_parse_iso_handles_z_suffix_and_garbage():
    parsed = PC.parse_iso("2026-11-04T07:14:40Z")
    assert parsed is not None
    assert parsed.tzinfo is not None
    assert parsed.year == 2026 and parsed.hour == 7
    assert PC.parse_iso("2026-11-04T07:14:40+08:00") is not None
    assert PC.parse_iso("昨天下午") is None
    assert PC.parse_iso("") is None
    assert PC.parse_iso(None) is None


def test_registry_has_exactly_five_reports_and_known_field_names():
    """登记表本身也要钉住：五份、字段名就是真实报告里那几个。

    写错一个字段名，判据会静默地读成 `None`——报告明明绑对了也会被判「不一致」。
    """
    assert len(PC.REPORTS) == 5
    fields = {spec["key"]: spec["gid_field"] for spec in PC.REPORTS}
    assert fields == {
        "k2_hard_gates": "generation_id",
        "k2_perf": "generation_id",
        "k2_gray": "generation_id",
        "k3_threshold_gate": "generations.new",
        "k4_drill": "candidate",
    }


def test_registry_paths_are_the_real_report_files():
    """登记表指向的必须是 `data/` 下真实存在的五份报告（跨 cwd 也要成立）。"""
    for spec in PC.REPORTS:
        assert spec["path"].startswith("data/")
        assert (ROOT / spec["path"]).exists(), f"{spec['path']} 不存在，登记表过期了"


# ---------------------------------------------------------------- 接进发布路径之后


def _load_release_cli():
    """载入 `scripts/release_cli.py`（它自己会 import `k2_publish_check`）。"""
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location(
        "release_cli_under_test", ROOT / "scripts" / "release_cli.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CLI = _load_release_cli()


@pytest.fixture()
def pub_sandbox(tmp_path, monkeypatch):
    """把 `release` 的三处全局与清单门的报告登记表全部指到 tmp。

    `release.GENERATIONS_DIR` / `RUNS_DIR` / `CURRENT_PATH` 是模块级常量，
    只有 monkeypatch 才能让函数体跟着变（同 `tests/test_release.py` 的 `sandbox`）。
    `REPORTS` 也必须换掉，否则门会去读仓库里真实的五份报告——那样这条测试就
    变成了「今天盘上恰好是绿的」，而不是「条件不成立时会拦」。

    **注意要换两处**：`release_cli` 里 `import k2_publish_check` 拿到的是模块的
    另一份实例（本文件顶上是 `spec_from_file_location` 载的），只 patch 一处，
    命令行那条路径仍然读真报告——这正是第一版 5 条用例全红的原因。
    """
    from test_release import _build_generation  # 复用造世代的那套

    gen_dir = tmp_path / "generations"
    runs_dir = tmp_path / "runs"
    current = tmp_path / "current.json"
    gen_dir.mkdir()
    runs_dir.mkdir()
    monkeypatch.setattr(release, "GENERATIONS_DIR", gen_dir)
    monkeypatch.setattr(release, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(release, "CURRENT_PATH", current)
    specs = _five_specs(tmp_path)
    monkeypatch.setattr(PC, "REPORTS", specs)
    monkeypatch.setattr(CLI.publish_check, "REPORTS", specs)
    return {"tmp": tmp_path, "current": current, "build": _build_generation, "specs": specs}


def _fresh_stamp(gen) -> str:
    """给报告一个「不早于候选建成」的时间戳。

    `_build_generation()` 用 `_now()` 造 `created_at`，所以固定写死一个过去的时间
    会让「报告不早于候选建成」这条误红——那是另一条判据的事，不该混进来。
    """
    from datetime import timedelta

    created = datetime.strptime(gen.load()["created_at"], "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    return (created + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _publish_args(gid: str):
    import argparse

    return argparse.Namespace(generation_id=gid, func=CLI.cmd_publish)


def test_publish_refused_when_a_report_is_missing(pub_sandbox, capsys):
    """五份报告缺一份 → 拒绝发布，**指针一个字节都没写**。

    这条是整个清单门存在的理由：`release.publish()` 自己看不到那五份报告，
    没有这道门，一个 READY 世代可以带着「根本没跑」的报告上线。
    """
    gen = pub_sandbox["build"]("g-20260101T000000Z-c00001")
    _write_reports(
        pub_sandbox["tmp"],
        skip="k2_perf",
        gid=gen.generation_id,
        generated_at=_fresh_stamp(gen),
    )

    code = CLI.cmd_publish(_publish_args(gen.generation_id))
    err = capsys.readouterr().err

    assert code == 1
    assert "拒绝发布" in err
    assert "五份门报告在场且可解析" in err
    assert "k2_perf.json" in err
    assert "py -3 scripts/k2_perf.py" in err, "要告诉人该补跑哪个脚本"
    assert not pub_sandbox["current"].exists(), "被拒时不许写指针"
    assert gen.status() == release.READY, "被拒不该改状态，补齐报告后还能重试"


def test_publish_refused_when_a_report_binds_another_generation(pub_sandbox, capsys):
    """报告都在、都过，但绑的是上一代 → 拒绝。这是「拿旧报告凑数」的形状。"""
    gen = pub_sandbox["build"]("g-20260101T000000Z-c00002")
    _write_reports(
        pub_sandbox["tmp"],
        gid=gen.generation_id,
        perf_gid="g-20251201T000000Z-0ld001",
        generated_at=_fresh_stamp(gen),
    )

    code = CLI.cmd_publish(_publish_args(gen.generation_id))
    err = capsys.readouterr().err

    assert code == 1
    assert "每份报告绑的是本次要发布的世代" in err
    assert "要发布的是" in err
    assert "g-20251201T000000Z-0ld001" in err and gen.generation_id in err
    assert not pub_sandbox["current"].exists()


def test_publish_refused_when_a_report_failed_itself(pub_sandbox, capsys):
    gen = pub_sandbox["build"]("g-20260101T000000Z-c00003")
    _write_reports(
        pub_sandbox["tmp"], gid=gen.generation_id, generated_at=_fresh_stamp(gen)
    )
    gray = pub_sandbox["tmp"] / "k2_gray.json"
    payload = json.loads(gray.read_text("utf-8"))
    payload["all_passed"] = False
    gray.write_text(json.dumps(payload), encoding="utf-8")

    code = CLI.cmd_publish(_publish_args(gen.generation_id))
    err = capsys.readouterr().err

    assert code == 1
    assert "每份报告自身 all_passed=True" in err
    assert "k2_gray.json" in err
    assert not pub_sandbox["current"].exists()


def test_publish_proceeds_and_says_which_checks_were_undetermined(pub_sandbox, capsys):
    """反向对照：五份都齐、都过、都绑对 → 真的发布。

    三份报告没有 `generated_at`，所以 `reports_not_stale` 只能是 `undetermined`。
    这时必须**把它说出来**再发布——否则「清单通过」会被读成「六条都验过了」。
    这条同时钉住「无法判定不阻止发布」：把它当失败会让发布永远做不成
    （除非去改那三份报告的生产者），而那不是这道门该管的事。
    """
    gen = pub_sandbox["build"]("g-20260101T000000Z-c00004")
    _write_reports(
        pub_sandbox["tmp"], gid=gen.generation_id, generated_at=_fresh_stamp(gen)
    )

    code = CLI.cmd_publish(_publish_args(gen.generation_id))
    out = capsys.readouterr().out

    assert code == 0
    assert "清单通过" in out
    assert "另有 1 条无法判定" in out, "无法判定的条数必须报出来"
    assert "报告不早于候选建成" in out
    assert pub_sandbox["current"].exists()
    assert release.read_current()["generation_id"] == gen.generation_id
    assert gen.status() == release.PUBLISHED


def test_publish_still_lets_release_checks_reject_after_checklist_passes(pub_sandbox, capsys):
    """清单过了不等于能发布：`release.publish()` 自己的检查仍然是第二道闸。

    这条防的是「清单过了就无脑切指针」——覆盖不全这类事清单门看不见
    （它的六条判据里没有这一项，也不该有：那是 `release.publish()` 的职责）。
    """
    gen = pub_sandbox["build"]("g-20260101T000000Z-c00005")
    _write_reports(
        pub_sandbox["tmp"], gid=gen.generation_id, generated_at=_fresh_stamp(gen)
    )
    gen.update(coverage_status=release.COVERAGE_INCOMPLETE, coverage_missing=["doc/a#002"])

    code = CLI.cmd_publish(_publish_args(gen.generation_id))
    captured = capsys.readouterr()

    assert code == 1
    assert "清单通过" in captured.out, "清单这一关是过了的"
    assert "覆盖不全" in captured.err
    assert not pub_sandbox["current"].exists()
