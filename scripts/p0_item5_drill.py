"""P0 第 5 项：在**隔离副本**里走一遍真实审批 / 发布 / 回滚 / 归档。

    py -3 scripts\\p0_item5_drill.py            # 全跑，写 data/p0_item5_drill.json
    py -3 scripts\\p0_item5_drill.py --check    # 只复核已写盘的报告，不重跑

退出码：0 = 全部判据成立；1 = 有判据不成立；2 = 硬错误（世代缺产物 / 报告读不出）。

## 为什么必须隔离

四个动作里三个会写真实状态：`publish()` 把候选 Manifest 改成 PUBLISHED 并改写
`current.json`；`rollback()` 改写指针；`archive_old(dry_run=False)` 把 250 MB 产物
复制走再删原件。在共享工作目录里直接跑它们，等于用演练本身制造事故。

做法：把要演练的世代**整目录复制**进临时沙箱，再把 `release` 的四个路径指进沙箱，
然后调**未改一行的真实实现**（`k2_publish_check.evaluate_generation` /
`release.publish` / `release.rollback` / `release.archive_old`）。复制的是真字节，
动作是真动作，被隔离的只有路径。

## 两半的界线（Lead m06795 要求分开做、分开写）

* **半 A —— 真实世代 + `dry_run=True`，证明「未修改任何持久化文件」**：
  在**真实目录**上直接调 `archive_old()`（线上世代 + 历史世代各一次），随后对
  `data/releases/` 与 `data/arkrag/generations/` 做**递归逐文件 SHA-256**，跑前跑后
  必须逐位相同。判据不是「我没看到改动」，是摘要相等。
* **半 B —— `dry_run=False`，证明「复制 → 核对 → 删」的顺序**：
  在沙箱里对真实世代的副本真删，三条判据缺一条即判红：
  ① 复制后源与归档副本**逐文件 SHA-256 全等**；
  ② 核对通过**才**允许删源（本工具核对删后的中间态：源目录只剩非产物文件）；
  ③ **删源后归档副本仍可 `release.verify_generation()` 通过**，且**线上世代仍可
     `release_cli.py current` 通过**——归档旧世代不许把在役世代带下水。
  没有第 ③ 条，「复制→核对→删」只是三个动作，证明不了删完还能用。

## 本轮固定结论（写进报告，不许被读成「演练通过」）

真实线上世代 `g-20261030T154606Z-55dfd7` 的观察期 **不足 7 天**（跑的那一刻约 6.0 天，
每轮实测写进报告的 `online_observation` 里），按 `archive_old()` 的 fail-closed 规则，
**真实归档路径今天必然被拒**。这不是实现问题，是规则生效。报告里
`real_archive_exercised` 恒为 `false`，并带一句固定结构的话（天数由实测填），
`check_report()` 会把缺这句话的报告判红——将来谁想把它改成「归档路径已验过」，
先得把这条门拆掉。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS.parent / "src"))
sys.path.insert(0, str(_SCRIPTS))

from arkrag import atomic, config, provenance, release, subproc  # noqa: E402

import k2_publish_check as publish_check  # noqa: E402

TOOL = "scripts/p0_item5_drill.py"
REPORT_PATH = config.ROOT / "data" / "p0_item5_drill.json"
SCHEMA = "ark-p0-item5-drill-1"

CANDIDATE = publish_check.CANDIDATE_GID
# 回滚目标与归档靶子都必须是**可服务**的世代：线上 `55dfd7` 是 ark-rag-1，
# `pipeline.load_generation()` 会拒；而回滚与归档都不经管线，选它是为了让
# 「previous_generation / 指针形状」与线上真实情况一致。
ROLLBACK_TARGET = "g-20261030T154606Z-55dfd7"
ARCHIVE_TARGET = "g-20261031T035231Z-4eafae"

# 半 A 要证明「一字节未动」的两棵真实目录树。
REAL_TREES = (
    config.ROOT / "data" / "releases",
    release.GENERATIONS_DIR,
)

REAL_ARCHIVE_NOTE_TEMPLATE = (
    "真实归档路径本轮未走通，原因是观察期 {days} 天 < {required} 天，"
    "不是实现问题。本报告里的归档结论全部来自隔离副本上的同一份真实字节。"
)

# 判据用的形状（不含具体天数）：天数每轮都在变，钉死数字就会在下一次跑时
# 变成一句错话——但「未走通 + 观察期 + 天 < 7 天 + 不是实现问题」这个结构必须
# 一字不差地出现。钉结构不钉数字，是这一条与「钉死字符串」的区别。
REAL_ARCHIVE_NOTE_SHAPE = re.compile(
    r"真实归档路径本轮未走通，原因是观察期 [\d.]+ 天 < \d+ 天，不是实现问题"
)

SEVEN = provenance.REQUIRED_FIELDS


# ---------------------------------------------------------------- 工具


def digest_tree(root: Path) -> dict[str, str]:
    """目录下每个文件 → SHA-256，键是相对路径（posix 风格）。"""
    out: dict[str, str] = {}
    root = Path(root)
    if not root.exists():
        return out
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[p.relative_to(root).as_posix()] = release.sha256_file(p)
    return out


def describe_drift(before: dict[str, str], after: dict[str, str]) -> dict:
    """两份摘要的差：少了 / 多了 / 变了。三种分开列，不合并成「有变化」。"""
    gone = sorted(set(before) - set(after))
    added = sorted(set(after) - set(before))
    changed = sorted(k for k in set(before) & set(after) if before[k] != after[k])
    return {"missing": gone, "extra": added, "changed": changed,
            "n_files": len(before), "identical": not (gone or added or changed)}


def manifest_stamps(generations_dir: Path) -> dict[str, tuple[int, int]]:
    """所有世代 Manifest 的 `(st_mtime_ns, st_size)`。回滚不许动产物。"""
    out: dict[str, tuple[int, int]] = {}
    for p in sorted(Path(generations_dir).glob("*/manifest.json")):
        st = p.stat()
        out[p.parent.name] = (st.st_mtime_ns, st.st_size)
    return out


def reassemble(archive_dir: Path, kept_from: Path, target_root: Path) -> Path:
    """把归档目录还原成「一代」的形状：`target_root/<gid>/`。

    `archive_old()` 只搬重量级产物，`manifest.json` 与 `reports/gates.json` 留在原处，
    所以还原 = 归档副本 + 源目录里保留的那几个文件。这一函数是判据 ③ 的全部内容：
    **还原之后必须能过 `verify_generation()`**，否则「归档可逆」只是句口号。
    """
    archive_dir = Path(archive_dir)
    dest = Path(target_root) / archive_dir.name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(archive_dir, dest)
    for p in sorted(Path(kept_from).rglob("*")):
        if p.is_file():
            rel = p.relative_to(kept_from)
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest / rel)
    return dest


def _strings_in(value) -> list[str]:
    """把一个嵌套结构里所有字符串捞出来。

    不能拿 `json.dumps()` 的文本去查机台路径：JSON 里反斜杠会被转义成 `\\\\`，
    `E:\\图rag` 在序列化结果里根本不存在——**查不到不是因为没有，是因为查法错了**。
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings_in(v)]
    if isinstance(value, (list, tuple, set)):
        return [s for v in value for s in _strings_in(v)]
    return []


def check_report(report: dict) -> list[str]:
    """报告的结构判据。返回问题清单，空表示合格。

    刻意不复用单条 `verify_*` 的结论而是**整块**要七字段：缺一个字段就是 `unknown`，
    而 `unknown` 不算通过。
    """
    problems: list[str] = []
    if report.get("schema") != SCHEMA:
        problems.append(f"schema 不是 {SCHEMA}（拿到 {report.get('schema')!r}）")

    verdict = provenance.verify({k: report.get(k) for k in SEVEN})
    if verdict.get("verdict") != provenance.PASS:
        missing = ", ".join(verdict.get("missing") or []) or "(见 detail)"
        problems.append(f"七字段未全部成立：verdict={verdict.get('verdict')}，缺 {missing}")

    if report.get("real_archive_exercised") is not False:
        problems.append(
            "real_archive_exercised 必须是 false：线上世代观察期不足 7 天，"
            "真实归档必然被拒；写成 true 等于宣称一条没走过的路径已经验过"
        )
    if not REAL_ARCHIVE_NOTE_SHAPE.search(str(report.get("real_archive_note") or "")):
        problems.append(
            "缺固定说明「真实归档路径本轮未走通，原因是观察期 X 天 < 7 天，"
            "不是实现问题」——缺这句将来会被读成演练通过"
        )

    checks = report.get("checks") or []
    if not checks:
        problems.append("checks 为空：一份没有判据的报告不能算通过")
    for item in checks:
        if item.get("ok") is not True:
            problems.append(f"判据 {item.get('id')} 不成立：{item.get('detail')}")

    root = str(config.ROOT)
    for text in _strings_in(report):
        if root in text:
            problems.append(
                f"报告里出现仓库绝对根路径（{config.rel_to_root(root)}…）："
                "应用 config.rel_to_root() 转相对"
            )
            break
    return problems


def _record(checks: list[dict], cid: str, ok: bool, detail: str, data=None) -> None:
    checks.append({"id": cid, "ok": bool(ok), "detail": detail, "data": data or {}})
    print(f"  [{'ok' if ok else 'FAIL'}] {cid}: {detail}")


def _run(cmd: list[str], *, cwd: Path, env_extra: dict[str, str] | None = None) -> dict:
    """跑一条命令并留命令级证据（命令 / 退出码 / stdout 摘要）。"""
    env = subproc.child_env()
    env.update(env_extra or {})
    proc = subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True,
        encoding=subproc.PIPE_ENCODING, errors="replace", env=env, timeout=600,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return {"cmd": " ".join(cmd), "exit_code": proc.returncode,
            "stdout_head": out.strip().splitlines()[:6], "chars": len(out)}


# ---------------------------------------------------------------- 阶段 1：审批


def stage_approval(checks: list[dict]) -> dict:
    """审批：只读真实五份门报告，跑与 `release_cli publish` 完全同一条判定。"""
    print("== 阶段 1：审批（真实报告，只读） ==")
    verdict = publish_check.evaluate_generation(CANDIDATE)
    failed = verdict.get("failed") or []
    items = verdict.get("checks") or []
    undetermined = [c for c in items if c.get("verdict") == publish_check.UNDETERMINED]
    _record(
        checks, "approval.all_passed", bool(verdict.get("all_passed")),
        f"清单 {len(items)} 条判据、failed {len(failed)} 条、undetermined {len(undetermined)} 条",
        {"failed": [f.get("name") for f in failed],
         "undetermined": [c.get("name") for c in undetermined]},
    )
    # 无法判定必须被看见：它不算失败，但更不算通过。
    _record(
        checks, "approval.undetermined_are_named",
        all(bool(c.get("detail")) for c in undetermined),
        f"{len(undetermined)} 条无法判定的判据都带 detail，不会被静默当成通过",
        {"names": [c.get("name") for c in undetermined]},
    )
    return {"all_passed": bool(verdict.get("all_passed")),
            "checks": len(items), "failed": len(failed),
            "undetermined": len(undetermined)}


# ---------------------------------------------------------------- 沙箱


def _build_sandbox(sandbox: Path, gids: list[str]) -> Path:
    """复制指定世代进沙箱，返回沙箱的 `ARKRAG_DATA_DIR`（即 `<sandbox>/arkrag`）。"""
    data_dir = sandbox / "arkrag"
    gens = data_dir / "generations"
    gens.mkdir(parents=True, exist_ok=True)
    (data_dir / "runs").mkdir(parents=True, exist_ok=True)
    for gid in gids:
        src = release.GENERATIONS_DIR / gid
        if not src.is_dir():
            raise release.ReleaseError(f"演练需要的世代不存在：{config.rel_to_root(src)}")
        shutil.copytree(src, gens / gid)
    return data_dir


def _enter_sandbox(data_dir: Path) -> dict:
    """把 release 的四个路径指进沙箱，返回原值供还原。"""
    saved = {
        "GENERATIONS_DIR": release.GENERATIONS_DIR,
        "RUNS_DIR": release.RUNS_DIR,
        "CURRENT_PATH": release.CURRENT_PATH,
        "ARCHIVE_DIR": release.ARCHIVE_DIR,
    }
    release.GENERATIONS_DIR = data_dir / "generations"
    release.RUNS_DIR = data_dir / "runs"
    release.CURRENT_PATH = data_dir / "current.json"
    release.ARCHIVE_DIR = data_dir / "archive"
    return saved


def _leave_sandbox(saved: dict) -> None:
    for key, value in saved.items():
        setattr(release, key, value)


def _write_pointer(path: Path, generation_id: str, *, published_days_ago: float) -> None:
    """写一份沙箱指针。只写沙箱，不碰真实指针。"""
    published = datetime.now(timezone.utc) - timedelta(days=published_days_ago)
    atomic.atomic_write_json(
        path,
        {"generation_id": generation_id, "status": release.PUBLISHED,
         "published_at": published.strftime("%Y-%m-%dT%H:%M:%SZ"),
         "previous_generation": ""},
    )


# ---------------------------------------------------------------- 阶段 2：发布 → 回滚


def stage_publish_rollback(data_dir: Path, checks: list[dict]) -> dict:
    """发布 + 回滚：沙箱副本上跑真实 `release.publish()` / `release.rollback()`。"""
    print("== 阶段 2：发布 → 回滚（沙箱副本） ==")
    real_before = release.CURRENT_PATH.read_bytes() if release.CURRENT_PATH.exists() else b""
    saved = _enter_sandbox(data_dir)
    out: dict = {}
    try:
        _write_pointer(release.CURRENT_PATH, ROLLBACK_TARGET, published_days_ago=1)
        gen = release.Generation(CANDIDATE)
        run = release.Run()
        published = release.publish(gen, run=run)
        _record(
            checks, "publish.pointer_points_at_candidate",
            published.get("generation_id") == CANDIDATE and gen.status() == release.PUBLISHED,
            f"指针 {published.get('generation_id')}，候选状态 {gen.status()}，"
            f"previous={published.get('previous_generation')}",
        )
        _record(
            checks, "publish.previous_is_rollback_target",
            published.get("previous_generation") == ROLLBACK_TARGET,
            f"previous_generation={published.get('previous_generation')!r}",
        )
        _record(
            checks, "publish.pointer_has_no_path_key", "path" not in published,
            "指针里不许存会过期的绝对机台路径",
            {"keys": sorted(published)},
        )

        before = manifest_stamps(release.GENERATIONS_DIR)
        rolled = release.rollback(ROLLBACK_TARGET)
        after = manifest_stamps(release.GENERATIONS_DIR)
        _record(
            checks, "rollback.pointer_shape",
            rolled.get("generation_id") == ROLLBACK_TARGET and rolled.get("rolled_back") is True,
            f"generation_id={rolled.get('generation_id')} rolled_back={rolled.get('rolled_back')}",
        )
        _record(
            checks, "rollback.manifests_untouched", before == after,
            f"{len(before)} 份 Manifest 的 (st_mtime_ns, st_size) 前后一致",
            {"changed": [k for k in before if before.get(k) != after.get(k)]},
        )
        _record(
            checks, "rollback.key_set_matches_publish",
            sorted(rolled) == sorted(list(published) + ["rolled_back"]),
            "回滚指针 = 发布指针 + rolled_back，多一个少一个都说明两边会漂移",
            {"publish": sorted(published), "rollback": sorted(rolled)},
        )
        again = release.rollback(CANDIDATE)
        _record(
            checks, "rollback.reversible", again.get("generation_id") == CANDIDATE,
            f"再切回后指针 {again.get('generation_id')}",
        )
        out["published_keys"] = sorted(published)
        out["rollback_keys"] = sorted(rolled)
    finally:
        _leave_sandbox(saved)

    real_after = release.CURRENT_PATH.read_bytes() if release.CURRENT_PATH.exists() else b""
    _record(
        checks, "rollback.real_pointer_untouched", real_before == real_after,
        "真实 current.json 全程一字节未变（沙箱隔离生效）",
        {"real_bytes": len(real_before)},
    )
    return out


# ---------------------------------------------------------------- 阶段 3：半 A


def stage_half_a(checks: list[dict]) -> dict:
    """半 A：真实世代 + `dry_run=True`，证明「未修改任何持久化文件」。

    判据是 `data/releases/` 与 `data/arkrag/generations/` 的递归逐文件 SHA-256
    跑前跑后逐位相同——不是「我没看到改动」。
    """
    print("== 阶段 3：半 A —— 真实世代 dry_run=True（证明盘上一字节不动） ==")
    before = {str(t): digest_tree(t) for t in REAL_TREES}
    n_before = sum(len(v) for v in before.values())
    print(f"  跑前摘要：{n_before} 个文件（data/releases/ + data/arkrag/generations/）")

    attempts = []
    for gid in (ROLLBACK_TARGET, ARCHIVE_TARGET):
        try:
            plan = release.archive_old(gid, dry_run=True)
        except release.ReleaseError as exc:
            attempts.append({"gid": gid, "refused": True, "reason": str(exc),
                             "detail": exc.detail})
            print(f"  {gid}：被拒 —— {exc}")
        else:
            attempts.append({"gid": gid, "refused": False,
                            "file_count": plan["file_count"],
                            "total_bytes": plan["total_bytes"]})
            print(f"  {gid}：给出归档计划 {plan['file_count']} 个文件 / {plan['total_bytes']} B")

    after = {str(t): digest_tree(t) for t in REAL_TREES}
    drifts = {name: describe_drift(before[name], after[name]) for name in before}
    _record(
        checks, "half_a.real_trees_byte_identical",
        all(d["identical"] for d in drifts.values()),
        f"两棵真实目录树 {n_before} 个文件逐位相同（跑前跑后递归 SHA-256 相等）",
        drifts,
    )

    obs = release.observation_status()
    _record(
        checks, "half_a.online_observation_incomplete_so_refused",
        obs["complete"] is False
        and all(a["refused"] for a in attempts)
        and all("观察期" in (a.get("reason") or "") + (a.get("detail") or "")
                for a in attempts),
        f"线上观察期 {obs.get('days_elapsed')} 天 < {obs.get('required_days')} 天，"
        f"两次真实 dry_run 均以观察期为由被拒",
        {"observation": {k: obs.get(k) for k in
                         ("generation_id", "published_at", "days_elapsed",
                          "required_days", "complete")},
         "attempts": attempts},
    )
    return {"attempts": attempts, "drift": drifts, "n_files": n_before}


# ---------------------------------------------------------------- 阶段 4：半 B


def stage_half_b(data_dir: Path, checks: list[dict]) -> dict:
    """半 B：`dry_run=False`，证明「复制 → 核对 → 删」的顺序且删完仍可用。"""
    print("== 阶段 4：半 B —— dry_run=False（真删一份真实世代的副本） ==")
    gen_dir = data_dir / "generations" / ARCHIVE_TARGET
    kept_dir = data_dir / "kept" / ARCHIVE_TARGET
    saved = _enter_sandbox(data_dir)
    try:
        # 指针指向候选并写成 30 天前：观察期门是这条路径的前置条件，
        # 沙箱里合法地满足它；线上世代（候选）不去动。
        _write_pointer(release.CURRENT_PATH, CANDIDATE, published_days_ago=30)
        obs = release.observation_status()
        _record(
            checks, "half_b.observation_complete_in_sandbox", obs["complete"] is True,
            f"沙箱指针 {obs.get('days_elapsed')} 天 / 需 {obs.get('required_days')} 天",
        )

        # 「保留文件」先备份：判据 ③ 还原时要用。
        kept_dir.mkdir(parents=True, exist_ok=True)
        for p in sorted(gen_dir.rglob("*")):
            if p.is_file() and p.relative_to(gen_dir).as_posix() in (
                    "manifest.json", "reports/gates.json", "reports/gates.json.lock"):
                (kept_dir / p.relative_to(gen_dir)).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, kept_dir / p.relative_to(gen_dir))

        source_before = digest_tree(gen_dir)
        res = release.archive_old(ARCHIVE_TARGET, dry_run=False)
        archive_dir = Path(res["archive_dir"])

        # 判据 ①：复制后源与归档副本逐文件 SHA-256 全等。
        pairs = []
        for item in res["files"]:
            dst = archive_dir / item["path"]
            ok = dst.is_file() and release.sha256_file(dst) == item["sha256"]
            pairs.append({"path": item["path"], "bytes": item["bytes"],
                          "sha256": item["sha256"][:16], "match": ok})
        _record(
            checks, "half_b.copy_digests_match",
            bool(pairs) and all(p["match"] for p in pairs),
            f"{len(pairs)} 个文件逐个 SHA-256 与源一致"
            f"（合计 {sum(p['bytes'] for p in pairs)} B）",
            {"files": pairs},
        )

        # 判据 ②：核对通过才允许删源 —— 核对删后的中间态：源里只剩非产物文件。
        # `reports/gates.json.lock` 是写盘锁（0 字节），它在不在取决于上一次写盘，
        # 所以判据是「除锁文件外只剩 manifest 与 gates」，不把锁写死进期望值。
        remaining_all = sorted(
            p.relative_to(gen_dir).as_posix() for p in gen_dir.rglob("*") if p.is_file()
        )
        remaining = [r for r in remaining_all if not r.endswith(".lock")]
        _record(
            checks, "half_b.source_keeps_only_non_artifacts",
            remaining == ["manifest.json", "reports/gates.json"],
            f"删源后源目录剩 {remaining}"
            + ("" if len(remaining) == len(remaining_all)
               else f"（另有 {sorted(set(remaining_all) - set(remaining))} 写盘锁）"),
            {"removed": res["removed"], "all_remaining": remaining_all},
        )
        _record(
            checks, "half_b.heavy_files_gone_from_source",
            not release._heavy_artifacts(release.Generation(ARCHIVE_TARGET)),
            "图 / 索引 / 块正文已从源目录移走（`_heavy_artifacts()` 为空）",
        )

        manifest = json.loads((archive_dir / "archive-manifest.json").read_text("utf-8"))
        _record(
            checks, "half_b.archive_manifest_complete",
            len(manifest["files"]) == res["file_count"]
            and [f["path"] for f in manifest["files"]] == [f["path"] for f in res["files"]]
            and manifest.get("total_bytes") == res["total_bytes"],
            f"archive-manifest.json 记了 {len(manifest['files'])} 个文件 / "
            f"{manifest['total_bytes']} B，与返回的清单逐项一致",
        )

        # 判据 ③a：删源后归档副本还原成世代形状，仍能过 verify_generation()。
        verify_root = data_dir / "restore" / "generations"
        verify_root.mkdir(parents=True, exist_ok=True)
        restored = reassemble(archive_dir, kept_dir, verify_root)
        problems = release.verify_generation(
            release.Generation(ARCHIVE_TARGET, root=verify_root))
        _record(
            checks, "half_b.restored_copy_verifies", problems == [],
            f"归档副本还原后 verify_generation() 返回 {len(problems)} 项不符"
            + ("" if not problems else f"：{problems[:3]}"),
            {"restored_files": len(digest_tree(restored)),
             "archived_files": len(digest_tree(archive_dir))},
        )
    finally:
        _leave_sandbox(saved)

    # 判据 ③b：线上世代（候选）仍在役，`release_cli.py current` 必须仍通过。
    # 用子进程 + `ARKRAG_DATA_DIR` 指进沙箱——这样跑的是**真实的命令行入口**，
    # 不是我在这里重新拼一遍同一套判据。
    cli = _run(
        [sys.executable, str(_SCRIPTS / "release_cli.py"), "current"],
        cwd=config.ROOT, env_extra={"ARKRAG_DATA_DIR": str(data_dir)},
    )
    _record(
        checks, "half_b.serving_generation_still_current",
        cli["exit_code"] == 0 and any("自检 PASS" in line for line in cli["stdout_head"]),
        f"归档旧世代后 `release_cli current` 退出码 {cli['exit_code']}，自检 PASS",
        cli,
    )
    return {"file_count": res["file_count"], "total_bytes": res["total_bytes"],
            "removed": res["removed"], "restored_verify_problems": problems,
            "source_files_before": len(source_before), "cli": cli}


# ---------------------------------------------------------------- 主流程


def collect_provenance() -> dict:
    """七字段：走 `provenance` 的采集器，不手拼。"""
    return provenance.collect(
        generation_id=CANDIDATE,
        dataset=provenance.dataset_block(),
        inputs=provenance.inputs_block([spec["path"] for spec in publish_check.REPORTS]),
    )


def run_drill() -> dict:
    checks: list[dict] = []
    t0 = time.perf_counter()
    sandbox = Path(tempfile.mkdtemp(prefix="p0item5_"))
    stages: dict = {}
    try:
        data_dir = _build_sandbox(sandbox, [CANDIDATE, ROLLBACK_TARGET, ARCHIVE_TARGET])
        n_gens = len(list((data_dir / "generations").iterdir()))
        print(f"== 阶段 0：三个真实世代整目录复制进沙箱（{n_gens} 个） ==")
        stages["approval"] = stage_approval(checks)
        stages["publish_rollback"] = stage_publish_rollback(data_dir, checks)
        stages["half_a_real_dry_run"] = stage_half_a(checks)
        stages["half_b_real_delete"] = stage_half_b(data_dir, checks)
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)

    report: dict = {}
    report.update(collect_provenance())
    obs = release.observation_status()
    report.update({
        "schema": SCHEMA,
        "tool": TOOL,
        "candidate": CANDIDATE,
        "rollback_target": ROLLBACK_TARGET,
        "archive_target": ARCHIVE_TARGET,
        "sandbox": "临时目录（每轮新建、跑完删除；报告里不写机台路径）",
        "stages": stages,
        "checks": checks,
        "real_archive_exercised": False,
        "real_archive_note": REAL_ARCHIVE_NOTE_TEMPLATE.format(
            days=obs.get("days_elapsed"), required=obs.get("required_days")
        ),
        "online_observation": {
            k: obs.get(k)
            for k in ("generation_id", "published_at", "days_elapsed",
                      "required_days", "complete", "reason")
        },
        "all_passed": bool(checks) and all(c["ok"] for c in checks),
        "wall_seconds": round(time.perf_counter() - t0, 2),
    })
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="P0 第 5 项：隔离副本审批/发布/回滚/归档演练")
    ap.add_argument("--check", action="store_true", help="只复核已写盘的报告")
    args = ap.parse_args()

    if args.check:
        if not REPORT_PATH.exists():
            print(f"报告不存在：{config.rel_to_root(REPORT_PATH)}", file=sys.stderr)
            return 2
        report = json.loads(REPORT_PATH.read_text("utf-8"))
        problems = check_report(report)
        if problems:
            print(f"报告不合格（{len(problems)} 项）：", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            return 1
        print(f"报告合格：{len(report.get('checks') or [])} 条判据全部成立，七字段 pass")
        return 0

    try:
        report = run_drill()
    except release.ReleaseError as exc:
        print(f"演练硬错误：{exc}", file=sys.stderr)
        if exc.detail:
            print(f"  明细：{exc.detail}", file=sys.stderr)
        return 2

    problems = check_report(report)
    if problems:
        report["all_passed"] = False
        report["report_problems"] = problems
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    failed = [c for c in report["checks"] if not c["ok"]]
    print(f"\n判据 {len(report['checks'])} 条，不成立 {len(failed)} 条；"
          f"报告 → {config.rel_to_root(REPORT_PATH)}")
    print(report["real_archive_note"])
    if problems:
        print(f"报告自身不合格 {len(problems)} 项，结论记 fail", file=sys.stderr)
        return 1
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
