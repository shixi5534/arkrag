"""K1 全量重建前检查：计划 §K1 点名的九项，逐项可失败。

    py -3 scripts/k1_checks.py                 # 跑全部九项，人读摘要
    py -3 scripts/k1_checks.py --json          # 机读（写进 docs 与台账用这一份）
    py -3 scripts/k1_checks.py --only schema_frozen
    py -3 scripts/k1_checks.py --list          # 只列出九项 id 与判据，不跑

## 退出码语义（必须能失败，见计划原则 5）

    0  九项 blocking 检查全部通过
    1  有 blocking 检查未通过（明细在 stdout，`--json` 里有逐项 data）
    2  硬错误（读不到世代目录、pytest 起不来之类），此时结论不可用

## 为什么要有这个脚本

K2 会重跑整条流水线并产出新一代候选。跑之前如果"Schema 悄悄变了、Gold Set
被人改过、上一代备份已经烂了、回滚根本切不动"，那么重跑出来的东西既无法与
历史对比，也没有退路——而这些都不是构建过程本身能发现的（构建只看得到自己的输入）。

所以九项检查的共同点是：**它们检查的都是"构建不会检查的东西"**。
九项里没有一项是"重新验证一遍产物内容"，那是 `verify_generation` 的职责，
在这里重写一遍就成了第二份定义。

## 一项检查通过意味着什么，不意味着什么

通过只代表"这一项没有发现阻断理由"。它不代表构建会成功，也不代表新世代比旧的好。
每项的 `data` 里都写了实测值，判断依据以数值为准，不以"PASS"这两个字为准。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arkrag import (  # noqa: E402
    agent_tasks,
    cache,
    config,
    contracts,
    fingerprint,
    legacy,
    prompt,
    release,
    subproc,
    vocab,
)

ROOT = config.ROOT
SCRIPTS = ROOT / "scripts"
TESTS = ROOT / "tests"

# ---------------------------------------------------------------- 冻结值

# Gold Set 的固定承诺。**这个常量就是"Gold Set 已固定"这句话的实体**。
# 取值时刻：2026-11-03 K1 首次实跑，sha256(gold-v1.json) 前 16 位。
# 换题库必须显式改这一行（改一行代码是有人签字的动作，改一个文件不是）。
#
# 2026-11-04 P1-a：题库从 v1（320 题）换成 v2（640 题，含切分与新增批）。
# **这次换签是 Lead 批准的动作**，理由是 v1 没有 split 字段、无法把调参集与
# 验收集分开，A 调参等于看答案。三件事同时成立才算换签完成：
#   1. gold-v1.json 一个字节没动（它仍是历史对照锚点，`gold_set_fixed` 换成 v2
#      之后，v1 的完整性由 tests/test_make_split.py 的 `test_gold_v1_is_untouched`
#      守着：sha256 前 16 位 76d66073ff87aab1、284356 字节）；
#   2. 下面三个常量与 gold-v2.json 实测一致；
#   3. 有人做过反向实测：把 GOLD_SET_SHA256_16 故意改错一位，门必须变红。
GOLD_SET_PATH = ROOT / "data" / "goldset" / "gold-v2.json"
GOLD_SET_SHA256_16 = "069375f8e355e4b9"
GOLD_SET_BYTES = 601492
GOLD_SET_TOTAL = 640
GOLD_SET_QUOTAS = {
    "single_chunk": 128,
    "alias": 96,
    "direct_relation": 96,
    "multihop": 96,
    "cross_chunk": 64,
    "time_conflict": 64,
    "negative_abstain": 64,
    "adversarial": 32,
}

# 磁盘余量下限的推导（不拍脑袋）：
#   实测一代 ark-rag-3 产物 254,005,061 B（242.2 MiB，7 个文件）。
#   一次 K2 全量重建 = 新一代候选 + 保留上一代 + 留一份回滚目标 = 3 代 ≈ 0.71 GiB。
#   再乘 3 倍余量给索引/向量/FTS 副产物与临时文件 → 约 2.1 GiB。
#   取整 2 GiB。低于这个数就停下来，不要"先跑跑看"。
REQUIRED_FREE_BYTES = 2 * 1024**3

# 契约测试：证明"输入形状"与"事务式构建"这两件事仍然成立。
CONTRACT_TESTS = (
    "tests/test_contracts.py",
    "tests/test_build_generation.py",
)

# 故障注入：证明"坏数据/坏操作会被挡住"，而不是只证明好数据能过。
#   test_negative.py    —— 计划书点名的反模式，每条都必须被拒
#   test_legacy_guard.py —— 冻结产物写保护必须真的抛异常
#   test_gate_numbering.py —— 门禁编号不许重号/跳号（少一个门没人发现的事故形态）
#   test_gates_l0.py / test_gates_e.py —— 门自身的负例
#   test_release.py     —— 发布/回滚的状态机不许越级
FAULT_TESTS = (
    "tests/test_negative.py",
    "tests/test_legacy_guard.py",
    "tests/test_gate_numbering.py",
    "tests/test_gates_l0.py",
    "tests/test_gates_e.py",
    "tests/test_release.py",
)

# 构建在途状态：这三态出现就说明有一次构建被打断或正在跑。
IN_FLIGHT = (release.PREPARING, release.BUILDING, release.VALIDATING)
TERMINAL = (release.READY, release.PUBLISHED, release.FAILED)


# ---------------------------------------------------------------- 结果形状


def _result(cid: str, name: str, ok: bool, detail: str, data: dict, *, blocking: bool = True) -> dict:
    return {
        "id": cid,
        "name": name,
        "ok": bool(ok),
        "blocking": bool(blocking),
        "detail": detail,
        "data": data,
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _generations() -> list[dict]:
    """盘上每一代，带自检问题数与目录字节数。自检要重算 250 MB 摘要，只算一次。"""
    rows: list[dict] = []
    for info in release.list_generations():
        gid = str(info.get("generation_id") or "")
        root = Path(str(info.get("path") or ""))
        problems: list[str] = []
        if GEN_ID_RE_OK(gid):
            try:
                problems = release.verify_generation(release.Generation(gid))
            except Exception as exc:  # noqa: BLE001 - 读不动就是问题，不能当 0
                problems = ["自检抛异常：%s: %s" % (type(exc).__name__, exc)]
        else:
            problems = ["目录名不合世代 id 形状"]
        total = 0
        files = 0
        if root.is_dir():
            for p in root.rglob("*"):
                if p.is_file():
                    files += 1
                    total += p.stat().st_size
        rows.append(
            {
                "generation_id": gid,
                "status": str(info.get("status") or ""),
                "path": str(root),
                "problems": problems,
                "files": files,
                "bytes": total,
            }
        )
    return rows


def GEN_ID_RE_OK(gid: str) -> bool:
    return bool(gid) and bool(release.GEN_ID_RE.match(gid))


def _current() -> dict:
    try:
        return release.read_current() or {}
    except Exception as exc:  # noqa: BLE001
        return {"_error": "%s: %s" % (type(exc).__name__, exc)}


def _repo_versions() -> dict:
    """当前代码声明的全部版本号。它们是"新 Schema 已冻结"这句话的对象。

    注意 `prompt_sha256` 必须取 `fingerprint.prompt_digest()`（64 位），**不是**
    `prompt.prompt_fingerprint()`（16 位截断）。两者前 16 位相同，所以看起来像
    同一个东西，但产物里记的是 64 位那一份——取错会得到一条"代码=7583… 产物=7583…
    7583…"这种前半段完全一样的不一致报警，让人以为产物有问题。
    首版就是这么写错的，实测红了一次才分清。
    """
    return {
        "schema_version": config.SCHEMA_VERSION,
        "prompt_version": config.PROMPT_VERSION,
        "prompt_sha256": fingerprint.prompt_digest(),
        "normalizer_version": config.NORMALIZER_VERSION,
        "task_schema": contracts.TASK_SCHEMA,
        "extraction_schema": contracts.EXTRACTION_SCHEMA,
        "entity_rule_version": vocab.ENTITY_RULE_VERSION,
        "relation_version": vocab.RELATION_VERSION,
        "vocab_sha": fingerprint.vocab_digest(),
    }


def _newest_identity_generation(rows: list[dict]) -> dict | None:
    """最新一代 ark-rag-3（用来对照"代码声明的版本"是否与已建成产物一致）。

    刻意不用"文件名最大"：世代目录名带时间戳，字符串序等价于时间序，
    但那是运气而不是契约。这里读 Manifest 里的 `graph_schema` 判断身份，
    再按 `created_at or gid` 排序取最后一个——与 `scripts/ask.py` 的
    `_latest_identity_generation` 同一个判据（那边已被 8 个用例盯着）。
    """
    candidates: list[tuple[str, str]] = []
    by_id = {r["generation_id"]: r for r in rows}
    for info in release.list_generations():
        gid = str(info.get("generation_id") or "")
        if not GEN_ID_RE_OK(gid):
            continue
        try:
            manifest = release.Generation(gid).load()
        except Exception:  # noqa: BLE001 - 读不动就不参与比较
            continue
        if str(manifest.get("graph_schema") or "") != config.GRAPH_SCHEMA_IDENTITY:
            continue
        candidates.append((str(manifest.get("created_at") or gid), gid))
    if not candidates:
        return None
    candidates.sort()
    return by_id.get(candidates[-1][1])


# ---------------------------------------------------------------- 九项检查


def check_schema_frozen(ctx: dict) -> dict:
    """1. 新 Schema 冻结：代码声明的版本必须与最新一代已建成产物记录的一致。

    判据为什么是"代码 vs 产物"而不是"代码里有没有写版本号"：版本号写在哪都行，
    真正会出事的是**两端不一致**——代码改了分词器/提示词/规范化器，产物还是旧的，
    这时重跑出来的新世代与历史世代不可比，而报告里每一项都还是绿的。

    这一项能失败的方式：改 `prompt.py` 一个字 → `prompt_sha256` 变 → 红。
    """
    versions = ctx["versions"]
    rows = ctx["generations"]
    newest = _newest_identity_generation(rows)
    data: dict = {"code": versions, "reference_generation": None, "mismatch": []}

    if newest is None:
        return _result(
            "schema_frozen",
            "新 Schema 冻结",
            False,
            "盘上没有 ark-rag-3 世代，无法证明当前代码版本与任一已建成产物一致",
            data,
        )

    manifest = release.Generation(newest["generation_id"]).load()
    recorded = dict(manifest.get("versions") or {})
    graph_path = release.Generation(newest["generation_id"]).graph_path
    graph_meta: dict = {}
    if graph_path.exists():
        try:
            graph = json.loads(graph_path.read_text("utf-8"))
            graph_meta = {
                "entity_rule_version": graph.get("entity_rule_version"),
                "relation_version": graph.get("relation_version"),
                "normalizer_version": graph.get("normalizer_version"),
                "prompt_version": graph.get("prompt_version"),
                "schema_version": graph.get("schema_version"),
            }
        except (OSError, ValueError) as exc:
            graph_meta = {"_error": "%s: %s" % (type(exc).__name__, exc)}

    # 只比"两端都记了"的键。Manifest 只记 5 个版本，图里多两个规则版本，
    # task/extraction 契约版本两边都不记——没记的东西不能算"一致"，
    # 但也不算不一致，单独列在 unverified 里，不许悄悄当成通过。
    pairs = (
        ("schema_version", versions["schema_version"], recorded.get("schema_version")),
        ("prompt_version", versions["prompt_version"], recorded.get("prompt_version")),
        ("prompt_sha256", versions["prompt_sha256"], recorded.get("prompt_sha256")),
        ("normalizer_version", versions["normalizer_version"], recorded.get("normalizer_version")),
        ("vocab_sha", versions["vocab_sha"], recorded.get("vocab_sha")),
        ("entity_rule_version", versions["entity_rule_version"], graph_meta.get("entity_rule_version")),
        ("relation_version", versions["relation_version"], graph_meta.get("relation_version")),
    )
    mismatch = [
        {"key": key, "code": code_val, "artifact": art_val}
        for key, code_val, art_val in pairs
        if art_val is not None and str(art_val) != str(code_val)
    ]
    unverified = [
        key
        for key in ("task_schema", "extraction_schema")
        if not recorded.get(key) and not graph_meta.get(key)
    ]

    data.update(
        {
            "reference_generation": newest["generation_id"],
            "recorded_versions": recorded,
            "graph_meta": graph_meta,
            "mismatch": mismatch,
            "unverified_in_artifacts": unverified,
        }
    )
    if mismatch:
        detail = "代码与产物版本不一致 %d 项：%s" % (
            len(mismatch),
            "；".join("%s 代码=%s 产物=%s" % (m["key"], m["code"], m["artifact"]) for m in mismatch[:4]),
        )
        return _result("schema_frozen", "新 Schema 冻结", False, detail, data)
    detail = "7 项版本与 %s 记录一致（其中 task/extraction 契约版本两端都不记，列为未验证）" % (
        newest["generation_id"],
    )
    return _result("schema_frozen", "新 Schema 冻结", True, detail, data)


def _run_pytest(paths: tuple[str, ...]) -> dict:
    """跑一组测试文件，返回逐文件 passed/failed 与退出码。

    用子进程而不是 `pytest.main()`：容器里已经有 pytest 的插件状态与断言重写缓存，
    进程内调用会让"这次跑的是不是同一批用例"变得说不清。检查脚本要能被人手工复现。
    """
    existing = [p for p in paths if (ROOT / p).exists()]
    missing = [p for p in paths if not (ROOT / p).exists()]
    cmd = [sys.executable, "-m", "pytest", "-q", "--no-header", *existing]
    started = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding=subproc.PIPE_ENCODING,
        errors="replace",
        # pytest 的摘要行是中文（「1 passed」「1 failed」之间的那些字），子进程默认按
        # cp936 写管道，我们按 UTF-8 读，摘要就会变成一串 \ufffd。判据虽然看的是
        # returncode，但人要靠这一行判断「红在哪」，不能让它不可读。
        env=subproc.child_env(),
    )
    tail = [line for line in (started.stdout or "").strip().splitlines() if line.strip()]
    return {
        "command": " ".join(cmd),
        "files": existing,
        "missing": missing,
        "returncode": started.returncode,
        "summary": tail[-1] if tail else "",
        "stdout_tail": tail[-12:],
        "stderr_tail": (started.stderr or "").strip().splitlines()[-6:],
    }


def check_contract_tests(ctx: dict) -> dict:
    """2. 契约测试通过：抽取契约与事务式构建的测试必须全绿。

    这一项能失败的方式：把 `contracts.validate_extraction` 里某条拒收放宽，
    或让 `build_generation` 在门禁失败时仍然 publish → 红。
    """
    run = _run_pytest(CONTRACT_TESTS)
    ok = run["returncode"] == 0 and not run["missing"]
    detail = run["summary"] or "pytest 无输出"
    if run["missing"]:
        detail = "缺测试文件：%s" % "、".join(run["missing"])
    return _result("contract_tests", "契约测试通过", ok, detail, run)


def check_fault_injection(ctx: dict) -> dict:
    """3. 故障注入通过：计划点名的反模式必须逐条被拒。

    `tests/test_negative.py` 的 `expect()` 失败即抛异常（不是记列表），
    所以它绿就是真绿——这一点由该文件的 docstring 记录过一次假通过事故。
    """
    run = _run_pytest(FAULT_TESTS)
    ok = run["returncode"] == 0 and not run["missing"]
    detail = run["summary"] or "pytest 无输出"
    if run["missing"]:
        detail = "缺测试文件：%s" % "、".join(run["missing"])
    return _result("fault_injection", "故障注入通过", ok, detail, run)


def check_gold_set_fixed(ctx: dict) -> dict:
    """4. Gold Set 固定：字节、条数、类别配额三项都必须与冻结值一致。

    只查 sha256 不够——有人可能只改了 `total` 而字节数刚好相同（换题不换长）。
    三项一起查，任一不符即红。配额表本身也来自 `gold-v2.json` 的实测。

    `gold-v1.json` 从 2026-11-04 起不再是被查对象：题库换成 v2（v1 的 320 条原样
    搬入 + 320 条全新题，每题带 `split`）。v1 文件仍在库、仍能跑，它是历史对照
    锚点；它的完整性由 `tests/test_make_split.py::test_gold_v1_is_untouched` 守。
    """
    data: dict = {
        "path": config.rel_to_root(GOLD_SET_PATH),
        "expected_sha256_16": GOLD_SET_SHA256_16,
        "expected_bytes": GOLD_SET_BYTES,
        "expected_total": GOLD_SET_TOTAL,
        "expected_quotas": GOLD_SET_QUOTAS,
    }
    if not GOLD_SET_PATH.exists():
        return _result("gold_set_fixed", "Gold Set 固定", False, "Gold Set 文件不存在", data)

    blob = GOLD_SET_PATH.read_bytes()
    actual_sha = hashlib.sha256(blob).hexdigest()[:16]
    data["actual_sha256_16"] = actual_sha
    data["actual_bytes"] = len(blob)

    try:
        payload = json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        return _result(
            "gold_set_fixed", "Gold Set 固定", False, "Gold Set 不是合法 UTF-8 JSON：%s" % exc, data
        )
    items = payload.get("items") if isinstance(payload, dict) else payload
    items = items or []
    counts: dict[str, int] = {}
    for item in items:
        cat = str(item.get("category") or "")
        counts[cat] = counts.get(cat, 0) + 1
    data["actual_total"] = len(items)
    data["actual_quotas"] = dict(sorted(counts.items()))

    problems: list[str] = []
    if actual_sha != GOLD_SET_SHA256_16:
        problems.append("sha256 前 16 位 %s ≠ 冻结值 %s" % (actual_sha, GOLD_SET_SHA256_16))
    if len(blob) != GOLD_SET_BYTES:
        problems.append("字节数 %d ≠ 冻结值 %d" % (len(blob), GOLD_SET_BYTES))
    if len(items) != GOLD_SET_TOTAL:
        problems.append("题数 %d ≠ 冻结值 %d" % (len(items), GOLD_SET_TOTAL))
    for cat, want in GOLD_SET_QUOTAS.items():
        got = counts.get(cat, 0)
        if got != want:
            problems.append("%s 配额 %d ≠ 冻结值 %d" % (cat, got, want))
    extra = sorted(set(counts) - set(GOLD_SET_QUOTAS))
    if extra:
        problems.append("出现冻结配额表里没有的类别：%s" % "、".join(extra))

    if problems:
        return _result("gold_set_fixed", "Gold Set 固定", False, "；".join(problems[:4]), data)
    return _result(
        "gold_set_fixed",
        "Gold Set 固定",
        True,
        "sha256=%s 字节=%d 题数=%d 八类配额全对"
        % (actual_sha, len(blob), len(items)),
        data,
    )


def check_old_artifacts_backed_up(ctx: dict) -> dict:
    """5. 旧产物已备份：冻结基线完整 + 至少两代可服务世代（候选与回滚目标）。

    两件事都要查，因为它们防的是不同的损失：
      - 冻结基线（`data/releases/legacy-baseline/`）是**唯一**一份 legacy 产物的
        字节级备份。它烂了，legacy 那条路径就再也重建不出来（写保护会拒绝覆盖，
        而覆盖也意味着丢失原始字节）。
      - 世代目录是**回滚的落点**。只剩一代时，"回滚"这个动作没有目标。

    判据复用既有脚本而不是重写：`snapshot_baseline.py --verify` 的 11 项比对
    就是这一件事的定义，在这里再写一遍就成了第二份定义。
    """
    data: dict = {"baseline_manifest": config.rel_to_root(legacy.BASELINE_MANIFEST)}
    problems: list[str] = []

    if not legacy.BASELINE_MANIFEST.exists():
        problems.append("冻结基线清单不存在：%s" % data["baseline_manifest"])
        data["snapshot_verify_returncode"] = None
    else:
        recorded = legacy.baseline_paths()
        data["baseline_files"] = len(recorded)
        cmd = [sys.executable, str(SCRIPTS / "snapshot_baseline.py"), "--verify"]
        run = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding=subproc.PIPE_ENCODING,
            errors="replace",
            # 同上：`snapshot_baseline.py --verify` 的摘要行是中文，不传 env 会读成乱码。
            env=subproc.child_env(),
        )
        data["snapshot_verify_returncode"] = run.returncode
        tail = [ln for ln in (run.stdout or "").strip().splitlines() if ln.strip()]
        data["snapshot_verify_summary"] = tail[-1] if tail else ""
        if run.returncode != 0:
            problems.append(
                "冻结基线校验未通过（退出码 %d）：%s"
                % (run.returncode, data["snapshot_verify_summary"] or "无输出")
            )

    servable = [r for r in ctx["generations"] if not r["problems"] and r["status"] in TERMINAL]
    data["servable_generations"] = [
        {"generation_id": r["generation_id"], "status": r["status"]} for r in servable
    ]
    data["servable_count"] = len(servable)
    if len(servable) < 2:
        problems.append(
            "可服务世代只有 %d 个（候选与回滚目标各需一个），不满足 K4「保留至少上一代完整有效 Generation」"
            % len(servable)
        )

    current = ctx["current"]
    cur_id = str(current.get("generation_id") or "")
    data["current_generation"] = cur_id
    if not cur_id:
        problems.append("发布指针读不出 generation_id：%s" % (current.get("_error") or "空"))
    elif cur_id not in {r["generation_id"] for r in servable}:
        problems.append("当前线上世代 %s 自身自检不过，不能算「已备份的可回退起点」" % cur_id)

    if problems:
        return _result("old_artifacts_backed_up", "旧产物已备份", False, "；".join(problems[:4]), data)
    return _result(
        "old_artifacts_backed_up",
        "旧产物已备份",
        True,
        "冻结基线 %d 文件校验一致；可服务世代 %d 个（含线上 %s）"
        % (data.get("baseline_files", 0), len(servable), cur_id),
        data,
    )


def check_disk_space(ctx: dict) -> dict:
    """6. 磁盘足够：剩余空间必须 ≥ 2 GiB，且 ≥ 3 倍单代产物实测大小。

    阈值推导写在 `REQUIRED_FREE_BYTES` 的注释里（实测一代 242.2 MiB × 3 代 × 3 倍余量）。
    """
    usage = shutil.disk_usage(str(ROOT))
    biggest = max((r["bytes"] for r in ctx["generations"]), default=0)
    need = max(REQUIRED_FREE_BYTES, 3 * biggest)
    data = {
        "root": config.rel_to_root(ROOT),
        "free_bytes": usage.free,
        "total_bytes": usage.total,
        "largest_generation_bytes": biggest,
        "required_bytes": need,
        "required_rule": "max(2 GiB, 3 × 最大单代产物字节)",
    }
    ok = usage.free >= need
    detail = "剩余 %.1f GiB / 需要 %.2f GiB（最大单代 %.1f MiB）" % (
        usage.free / 1024**3,
        need / 1024**3,
        biggest / 1024**2,
    )
    if not ok:
        detail = "磁盘不足：" + detail
    return _result("disk_space", "磁盘足够", ok, detail, data)


def check_model_budget(ctx: dict) -> dict:
    """7. 模型预算明确：要么没有 LLM 端点（预算为 0），要么显式声明预算。

    本机现状是无 LLM 端点（`ARKRAG_LLM_MODEL` 与 `ARKRAG_LLM_BASE_URL` 皆空，
    `arkrag.llm` 缺配置直接抛 RuntimeError），回答器是抽取式 `ExtractiveAnswerer`，
    所以这一项的正确答案是"不需要预算"，而且这个答案是**可验证的事实**，
    不是"跳过"。

    这一项能失败的方式：把 `ARKRAG_LLM_MODEL` 设成任何非空值而不配
    `ARKRAG_K1_LLM_BUDGET`（币种/上限/预估调用次数三元组）→ 红。
    换句话说，它挡的是"悄悄接上 LLM 然后开始烧钱"。
    """
    model = str(config.LLM_MODEL or "").strip()
    base_url = str(config.LLM_BASE_URL or "").strip()
    budget_raw = str(os.environ.get("ARKRAG_K1_LLM_BUDGET") or "").strip()
    data = {
        "llm_model": model,
        "llm_base_url": base_url,
        "llm_retries": config.LLM_RETRIES,
        "llm_concurrency": config.LLM_CONCURRENCY,
        "budget_declared": budget_raw,
        "answerer": "extractive",
        "extraction_executor": agent_tasks.AGENT_MODEL,
    }
    if not model and not base_url:
        data["budget_required"] = False
        return _result(
            "model_budget",
            "模型预算明确",
            True,
            "无 LLM 端点（model/base_url 皆空）→ 预算为 0；抽取由 %s 离线产出" % agent_tasks.AGENT_MODEL,
            data,
        )
    data["budget_required"] = True
    if not budget_raw:
        return _result(
            "model_budget",
            "模型预算明确",
            False,
            "已配置 LLM 端点（model=%s）但未声明预算：请设 ARKRAG_K1_LLM_BUDGET（币种/上限/预估调用次数）" % model,
            data,
        )
    parts = [p for p in budget_raw.split("/") if p.strip()]
    ok = len(parts) >= 3
    return _result(
        "model_budget",
        "模型预算明确",
        ok,
        "已配置 LLM 端点并声明预算：%s" % budget_raw if ok else "预算声明缺字段（需 币种/上限/预估调用次数 三段）：%s" % budget_raw,
        data,
    )


def check_resumable(ctx: dict) -> dict:
    """8. 可从中断恢复：在途世代可识别、不可发布、线上世代不受影响。

    这条最容易被写成"我们支持续跑"这种无法验证的话。可验证的事实是三件：

    a. 盘上每一代的状态都在已知六态内。出现未知状态字面量说明有人加了状态
       却没同步到发布状态机，此时"它能不能发布"没人知道。
    b. 被打断的世代（在途三态）**不能**被发布。这一条现场实测，不靠读代码：
       拿盘上真实存在的一代非终态/失败世代去调 `release.publish`，必须抛
       `IllegalTransition` 或 `ReleaseError`。
    c. 打断不影响线上：当前指针指向的世代自检 0 问题。

    另附抽取缓存的键控事实（`cache_key` 含 content_key + model + prompt sha），
    说明重跑是"续"而不是"从头"——这一项用 `invalid_cache_report()` 的非法条目数
    来验证，非法条目非空说明缓存复用已经出过错。
    """
    rows = ctx["generations"]
    data: dict = {}
    problems: list[str] = []

    statuses = sorted({r["status"] for r in rows})
    data["statuses"] = statuses
    unknown = [s for s in statuses if s not in (TERMINAL + IN_FLIGHT)]
    in_flight = [r for r in rows if r["status"] in IN_FLIGHT]
    data["in_flight"] = [
        {"generation_id": r["generation_id"], "status": r["status"]} for r in in_flight
    ]
    if unknown:
        problems.append("出现未知世代状态：%s" % "、".join(unknown))

    # b. 现场证明"非 READY 不许发布"。挑一代 FAILED（没有就挑在途的，再没有就跳过）。
    probe = next((r for r in rows if r["status"] == release.FAILED), None) or (
        in_flight[0] if in_flight else None
    )
    data["publish_probe_generation"] = probe["generation_id"] if probe else None
    if probe is None:
        data["publish_probe"] = "skipped_no_non_ready_generation"
    else:
        try:
            release.publish(release.Generation(probe["generation_id"]))
        except release.ReleaseError as exc:
            data["publish_probe"] = "refused: %s" % type(exc).__name__
            data["publish_probe_message"] = str(exc)[:200]
        else:
            data["publish_probe"] = "ACCEPTED"
            problems.append(
                "release.publish 接受了状态为 %s 的 %s，中断恢复的隔离前提不成立"
                % (probe["status"], probe["generation_id"])
            )

    # c. 线上不受影响。
    cur_id = str(ctx["current"].get("generation_id") or "")
    data["current_generation"] = cur_id
    cur_row = next((r for r in rows if r["generation_id"] == cur_id), None)
    if cur_row is None:
        problems.append("当前指针 %s 在盘上不存在" % (cur_id or "(空)"))
    elif cur_row["problems"]:
        problems.append("当前线上世代自检 %d 项问题" % len(cur_row["problems"]))

    # 附：抽取缓存的非法条目。非空说明"复用过的记录"已经出过问题。
    #
    # `invalid_cache_report()` 返回 `{total, ok, invalid, by_reason, samples}`——
    # **`total` 是全部文件数，`invalid` 才是坏的那部分**。首版把 `total` 当成坏的
    # 数量报了 "13801 条非法记录"，而实测 `ok=13801 / invalid=0`：缓存一个坏条目
    # 都没有。这类错误最危险的地方是它看起来像个重大发现。
    try:
        report = cache.invalid_cache_report()
        if isinstance(report, dict):
            bad = int(report.get("invalid") or 0)
            data["invalid_cache"] = {
                "total": int(report.get("total") or 0),
                "ok": int(report.get("ok") or 0),
                "invalid": bad,
                "by_reason": dict(report.get("by_reason") or {}),
            }
        else:
            bad = 0
            data["invalid_cache"] = {"error": "非法返回值：%r" % (type(report).__name__,)}
    except Exception as exc:  # noqa: BLE001
        data["invalid_cache"] = {"error": "%s: %s" % (type(exc).__name__, exc)}
        bad = 0
    if bad:
        problems.append("抽取缓存里有 %d 条非法记录（K2 硬门「无非法缓存」会因此为假）" % bad)

    if problems:
        return _result("resumable", "可从中断恢复", False, "；".join(problems[:4]), data)
    detail = "状态集合 %s；非 READY 发布被拒（%s）；线上 %s 自检 0 问题" % (
        "/".join(statuses),
        data.get("publish_probe"),
        cur_id,
    )
    if in_flight:
        detail += "；有 %d 个在途世代待清理或续跑" % len(in_flight)
    return _result("resumable", "可从中断恢复", True, detail, data)


def check_rollback_tested(ctx: dict) -> dict:
    """9. 回滚命令已实测：在隔离指针上真跑一次 `release.rollback`。

    为什么必须在隔离指针上跑：真跑会把线上指针切走。而"只在文档里说回滚可用"
    正是这一类缺口的标准形态——`previous_generation` 指向的那一代（`606ea9`）
    实测自检 **2 项不过**，如果没人真跑过，"回滚命令已实测"这句话就是假的。

    做法：把 `release.CURRENT_PATH` 临时指到 tmp 文件（`release` 里所有指针读写都
    经过这个模块级常量），写入一个"当前是候选世代"的假指针，然后回滚到一个
    实测 0 问题的历史世代。要验证的是三件事：
      a. 指针真的被改写，且写出的键集与 publish 指针同构（含 `rolled_back`）；
      b. 目标世代是从 `verify_generation` 拿到的真结论，不是"文件存在就算"；
      c. 回滚到"当前已经是"的那一代必须被拒（否则 `previous_generation` 指向自身，
         下次真回滚就没有退路）。

    真跑完把 `CURRENT_PATH` 复原。这中间即使失败也不改线上指针——因为写的
    从来不是那个文件。
    """
    data: dict = {"real_current_path": config.rel_to_root(release.CURRENT_PATH)}
    rows = ctx["generations"]
    servable = [r for r in rows if not r["problems"] and r["status"] in TERMINAL]
    data["servable_generations"] = [r["generation_id"] for r in servable]
    if len(servable) < 2:
        return _result(
            "rollback_tested",
            "回滚命令已实测",
            False,
            "可服务世代不足 2 个，没有可用的回滚演练目标",
            data,
        )

    # 目标是"最新的一代"（模拟候选刚上线），回滚到"较早的一代"（模拟回退）。
    servable_sorted = sorted(servable, key=lambda r: r["generation_id"])
    older, newer = servable_sorted[0], servable_sorted[-1]
    data["simulated_current"] = newer["generation_id"]
    data["rollback_target"] = older["generation_id"]

    # 探针目录必须**每次运行独一份**。原先写死成
    # `data/arkrag/_k1_rollback_probe/`，于是任意两个进程同时跑这一项就会
    # 共用同一个 `current.json`：A 刚写完"当前是候选世代"，B 已经把它回滚成
    # `e0aacc` 并读出"当前是 e0aacc"，A 接着在 B 的指针上再回滚一次——
    # 这一次要么被"已经是当前世代"拒掉（判成"回滚命令不可用"），要么
    # 干脆被接受（判成"回滚到自身未被拒"）。两条都是**假结论**，而且只在
    # 并发时出现：单独跑永远绿。实测并发 4 份 `--only rollback_tested`
    # → 0/4 通过，两种错法各出现两次。
    #
    # 用 `tempfile.mkdtemp()` 而不是在 `data/arkrag` 下加进程号：进程号会被
    # 复用，而这个目录出现在冻结基线目录里本身就不该。`mkdtemp` 保证
    # 唯一且落在系统临时区；`finally` 里整目录清掉。
    tmp_dir = Path(tempfile.mkdtemp(prefix="k1_rollback_probe_"))
    tmp_pointer = tmp_dir / "current.json"
    original = release.CURRENT_PATH
    problems: list[str] = []
    try:
        release.CURRENT_PATH = tmp_pointer
        tmp_pointer.write_text(
            json.dumps(
                {
                    "generation_id": newer["generation_id"],
                    "status": release.PUBLISHED,
                    "published_at": "2026-11-03T00:00:00Z",
                    "previous_generation": "",
                    "source_fingerprint": "",
                    "graph_artifact_id": "",
                    "index_artifact_id": "",
                    "gate_report_hash": "",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        # 回滚命令自己抛错（`ReleaseError`）是一条**结论**，不是脚本崩了：
        # 它意味着"回滚命令不可用"，正是这一项要查的东西。若让它冒出去，
        # `run_checks` 会把它记成"检查自身抛异常"，detail 里只剩个异常名，
        # 读的人不知道是回滚被拒还是磁盘坏了。所以在这里接住并记成问题。
        pointer = None
        try:
            pointer = release.rollback(older["generation_id"])
        except release.ReleaseError as exc:
            problems.append("release.rollback 拒绝执行：%s" % str(exc)[:200])
            data["rollback_refused"] = type(exc).__name__
        except Exception as exc:  # noqa: BLE001 - 非 ReleaseError 也是问题，但要区分开
            problems.append("release.rollback 抛出非 ReleaseError：%s: %s" % (type(exc).__name__, exc))
            data["rollback_refused"] = type(exc).__name__

        if pointer is None:
            data["pointer_on_disk"] = None
            data["pointer_keys"] = []
        else:
            data["written_pointer"] = pointer
            written = json.loads(tmp_pointer.read_text("utf-8"))
            data["pointer_on_disk"] = written

            if written.get("generation_id") != older["generation_id"]:
                problems.append("指针未改写成回滚目标")
            expected_keys = {
                "generation_id",
                "status",
                "published_at",
                "previous_generation",
                "rolled_back",
                "source_fingerprint",
                "graph_artifact_id",
                "index_artifact_id",
                "gate_report_hash",
            }
            data["pointer_keys"] = sorted(written.keys())
            if set(written.keys()) != expected_keys:
                problems.append(
                    "回滚指针键集与 publish 不同构：多 %s 少 %s"
                    % (sorted(set(written) - expected_keys), sorted(expected_keys - set(written)))
                )
            if written.get("rolled_back") is not True:
                problems.append("回滚指针没有 rolled_back=True 标记")

            # 回滚到"当前已经是"的那一代必须被拒。
            try:
                release.rollback(older["generation_id"])
            except release.ReleaseError as exc:
                data["self_rollback_probe"] = "refused: %s" % type(exc).__name__
                data["self_rollback_message"] = str(exc)[:160]
            except Exception as exc:  # noqa: BLE001
                problems.append("回滚到自身抛了非 ReleaseError：%s" % type(exc).__name__)
            else:
                data["self_rollback_probe"] = "ACCEPTED"
                problems.append("回滚到当前世代未被拒，previous_generation 会退化成指向自身")
    finally:
        release.CURRENT_PATH = original
        shutil.rmtree(tmp_dir, ignore_errors=True)

    data["real_pointer_untouched"] = (
        config.rel_to_root(original) == data["real_current_path"] and original == release.CURRENT_PATH
    )
    if problems:
        return _result("rollback_tested", "回滚命令已实测", False, "；".join(problems[:4]), data)
    return _result(
        "rollback_tested",
        "回滚命令已实测",
        True,
        "隔离指针上 %s → %s 成功，键集同构，回滚到自身被拒" % (newer["generation_id"], older["generation_id"]),
        data,
    )


CHECKS = (
    ("schema_frozen", check_schema_frozen, "新 Schema 冻结"),
    ("contract_tests", check_contract_tests, "契约测试通过"),
    ("fault_injection", check_fault_injection, "故障注入通过"),
    ("gold_set_fixed", check_gold_set_fixed, "Gold Set 固定"),
    ("old_artifacts_backed_up", check_old_artifacts_backed_up, "旧产物已备份"),
    ("disk_space", check_disk_space, "磁盘足够"),
    ("model_budget", check_model_budget, "模型预算明确"),
    ("resumable", check_resumable, "可从中断恢复"),
    ("rollback_tested", check_rollback_tested, "回滚命令已实测"),
)
CHECK_IDS = tuple(cid for cid, _, _ in CHECKS)


# ---------------------------------------------------------------- 主流程


def build_context() -> dict:
    """一次性收集检查要用的盘上事实。九项共用，避免某项偷偷用不同的快照。"""
    return {
        "generations": _generations(),
        "current": _current(),
        "versions": _repo_versions(),
        "git_revision": release.git_revision(),
    }


def run_checks(only: tuple[str, ...] = ()) -> dict:
    ctx = build_context()
    results: list[dict] = []
    for cid, fn, name in CHECKS:
        if only and cid not in only:
            continue
        try:
            results.append(fn(ctx))
        except Exception as exc:  # noqa: BLE001 - 单项崩掉不能拖垮整轮
            results.append(
                _result(
                    cid,
                    name,
                    False,
                    "检查自身抛异常：%s: %s" % (type(exc).__name__, exc),
                    {"exception": type(exc).__name__},
                )
            )
    blocking_failed = [r for r in results if r["blocking"] and not r["ok"]]
    return {
        "generated_at": _now(),
        "git_revision": ctx["git_revision"],
        "generations": [
            {k: v for k, v in r.items() if k != "problems"} | {"problem_count": len(r["problems"])}
            for r in ctx["generations"]
        ],
        "checks": results,
        "blocking_failed": [r["id"] for r in blocking_failed],
        "ok": not blocking_failed,
    }


def _now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def print_report(report: dict) -> None:
    print("K1 全量重建前检查")
    print("  时间     : %s" % report["generated_at"])
    print("  修订     : %s" % (report["git_revision"] or "(无 git)"))
    print("  世代     : %d 个" % len(report["generations"]))
    ran = len(report["checks"])
    total = len(CHECK_IDS)
    if ran != total:
        # `--only` 只跑了子集。此时绝不能说"九项全部通过"——那会让人以为
        # 九项都查过了，而实际只查了三项。结论必须带上分母。
        print("  范围     : 只跑了 %d/%d 项（--only），结论仅覆盖这几项" % (ran, total))
    print()
    width = max(len(c["name"]) for c in report["checks"]) if report["checks"] else 10
    for item in report["checks"]:
        mark = "PASS" if item["ok"] else "FAIL"
        flag = "" if item["blocking"] else " [不阻断]"
        print("  [%s] %-*s%s" % (mark, width, item["name"], flag))
        print("         %s" % item["detail"])
    print()
    if report["ok"] and ran == total:
        print("结论：九项 blocking 检查全部通过，可以进入 K2。")
        print("      （通过不代表新世代更好，只代表没有发现阻断理由。）")
    elif report["ok"]:
        print("结论：已跑的 %d 项通过，但只覆盖 %d/%d 项，不足以进入 K2。" % (ran, ran, total))
        print("      进 K2 前请跑全量：py -3 scripts/k1_checks.py")
    else:
        print("结论：%d 项未通过 → %s" % (len(report["blocking_failed"]), "、".join(report["blocking_failed"])))
        print("      不要开始全量重建。")
        print("      不要开始全量重建。")


def main(argv: list[str] | None = None) -> int:
    # Windows 控制台默认 cp936，不重配置会把中文写成乱码字节，
    # 且 `> file` 重定向后文件不是 UTF-8（本项目脚本一律要求输出可原样粘贴进文档）。
    # 放在 `main()` 里而不是模块顶层：本文件会被 `tests/test_k1_checks.py` 用
    # importlib 加载，那时 `sys.stdout` 是 pytest 的捕获对象，不该被重配置。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="K1 全量重建前检查（计划 §K1 九项）")
    ap.add_argument("--json", action="store_true", help="输出机读 JSON")
    ap.add_argument("--only", default="", help="逗号分隔的检查 id，只跑这些")
    ap.add_argument("--list", action="store_true", help="只列出九项 id，不跑")
    args = ap.parse_args(argv)

    if args.list:
        for cid, _, name in CHECKS:
            print("%-24s %s" % (cid, name))
        return 0

    only = tuple(p.strip() for p in args.only.split(",") if p.strip())
    unknown = [p for p in only if p not in CHECK_IDS]
    if unknown:
        print("[2] 未知检查 id：%s（可用：%s）" % ("、".join(unknown), "、".join(CHECK_IDS)))
        return 2

    try:
        report = run_checks(only)
    except Exception as exc:  # noqa: BLE001
        print("[2] 收集盘上事实失败：%s: %s" % (type(exc).__name__, exc))
        return 2

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)
    if not report["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
