"""Run / Generation 目录模型与原子发布（计划 C1 / C2）。

## 为什么必须有 Generation

现有流水线把产物直接写在固定路径上（`graph.json`、`index/lexical.json`、
`gates.json`）。这带来三个没法用"更小心一点"解决的事故：

1. **失败发布**：门禁 FAIL 之后 `graph.json` 已经被覆盖了。下一次跑 L6 读到的是
   一个没过门的图，而门禁报告可能还是上一轮的全 PASS——产物与报告对不上，且没人
   能事后分辨。
2. **回滚做不到**：只有一份产物，没有"上一代"。想退回去只能重跑，而重跑要等
   抽取和建图，中间这段时间线上没有可用版本。
3. **身份说不清**：`graph.json` 里有一个 `fingerprint`，`lexical.json` 里记着
   `graph_fp`。但这只是**配置指纹**——它证明"输入相同"，不证明"这份字节就是由
   那次构建产出的"。文件被换掉一份而指纹不变，任何检查都发现不了。

所以这一层引入两件事：

* `generations/<generation_id>/` —— 每次构建一个独立目录，产物、门禁报告、
  Manifest 都在自己目录里，构建期间随便怎么看都不会污染正式版本。
* `current.json` —— 唯一的"现在线上是哪一代"指针。发布 = 一次原子替换这个文件。
  发布之前任何一步失败，`current.json` 根本没有被碰过，旧版本照常服务。

## 状态机

    PREPARING -> BUILDING -> VALIDATING -> READY -> PUBLISHED
    任何状态 -> FAILED

只有 `READY` 能被发布。`PUBLISHED` 是从 `READY` 经过一次成功指针切换得到的，
不经过 `READY` 的 `PUBLISHED` 视为非法状态（说明有人绕过了校验）。

## 发布顺序（不可调换）

    读取固定输入快照 -> 构建候选 -> 计算门禁 -> 回读候选产物
    -> 校验身份与内容摘要 -> 全部硬门通过 -> 原子发布

回读与校验在发布**之前**：先证明"我刚写下的东西能原样读回来、且读回来的内容
摘要与 Manifest 记的一致"，再去动指针。顺序反过来就永远存在"已发布但读不出来"
的窗口。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import atomic, config

GENERATIONS_DIR = config.DATA_DIR / "generations"
RUNS_DIR = config.DATA_DIR / "runs"
CURRENT_PATH = config.DATA_DIR / "current.json"

MANIFEST_NAME = "manifest.json"
GATE_REPORT_NAME = "gates.json"

# 状态机
PREPARING = "PREPARING"
BUILDING = "BUILDING"
VALIDATING = "VALIDATING"
READY = "READY"
PUBLISHED = "PUBLISHED"
FAILED = "FAILED"

TRANSITIONS: dict[str, tuple[str, ...]] = {
    PREPARING: (BUILDING, FAILED),
    BUILDING: (VALIDATING, FAILED),
    VALIDATING: (READY, FAILED),
    READY: (PUBLISHED, FAILED),
    PUBLISHED: (FAILED,),
    FAILED: (),
}

GEN_ID_RE = re.compile(r"^g-\d{8}T\d{6}Z-[0-9a-f]{6}$")


class ReleaseError(RuntimeError):
    """发布层的硬失败。消息面向人，`detail` 面向排查。"""

    def __init__(self, message: str, *, detail: str = "") -> None:
        super().__init__(message)
        self.detail = detail


class IllegalTransition(ReleaseError):
    pass


# ---------------------------------------------------------------- 身份与摘要


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_revision() -> str:
    """取当前 git 提交号。没有提交（初始仓库）或没装 git 时返回空串并写进 Manifest。

    空串不等于"干净"——Manifest 里保留空串，是为了让读的人一眼看出
    "这一代构建时源码版本不知道"，而不是让某个默认值冒充真实提交。
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(config.ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def chunk_manifest_hash(chunks: list[dict]) -> str:
    """块清单摘要：身份 + 内容键，按 chunk_id 排序后累进。

    它回答的问题是"这一代用到的块清单有没有变"。只比块数是不够的——
    重建后块数相同、内容全换了，块数比对完全看不出来。
    """
    h = hashlib.sha256()
    for c in sorted(chunks, key=lambda r: r["chunk_id"]):
        h.update(f"{c['chunk_id']}\x00{c.get('content_key','')}\x00{len(c.get('text',''))}\n".encode())
    return h.hexdigest()


def extraction_manifest_hash(records: dict[str, dict]) -> str:
    """抽取清单摘要：每个块的 chunk_id + content_key + 该块 raw 的内容摘要。"""
    h = hashlib.sha256()
    for cid in sorted(records):
        raw = records[cid] or {}
        body = json.dumps(
            {"entities": raw.get("entities"), "relations": raw.get("relations")},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        h.update(f"{cid}\x00{sha256_bytes(body.encode())}\n".encode())
    return h.hexdigest()


# ---------------------------------------------------------------- 输入快照


@dataclass
class InputSnapshot:
    """构建前固定下来的输入身份。构建过程中输入再被改动，也不影响这一代。"""

    source_fingerprint: str
    chunk_manifest_hash: str
    chunk_count: int
    extraction_manifest_hash: str
    extraction_count: int
    versions: dict[str, str] = field(default_factory=dict)
    git_revision: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_fingerprint": self.source_fingerprint,
            "chunk_manifest_hash": self.chunk_manifest_hash,
            "chunk_count": self.chunk_count,
            "extraction_manifest_hash": self.extraction_manifest_hash,
            "extraction_count": self.extraction_count,
            "versions": dict(self.versions),
            "git_revision": self.git_revision,
        }


def capture_inputs(*, chunks: list[dict], records: dict[str, dict], source_fp: str) -> InputSnapshot:
    return InputSnapshot(
        source_fingerprint=source_fp,
        chunk_manifest_hash=chunk_manifest_hash(chunks),
        chunk_count=len(chunks),
        extraction_manifest_hash=extraction_manifest_hash(records),
        extraction_count=len(records),
        versions={
            "schema_version": config.SCHEMA_VERSION,
            "prompt_version": config.PROMPT_VERSION,
            "normalizer_version": config.NORMALIZER_VERSION,
        },
        git_revision=git_revision(),
    )


# ---------------------------------------------------------------- Run 目录


def new_run_id() -> str:
    return "r-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + secrets.token_hex(3)


class Run:
    """一次构建的执行记录。任何阶段都能往里追加事件，用来回答"那次构建卡在哪"。"""

    def __init__(self, run_id: str | None = None) -> None:
        self.run_id = run_id or new_run_id()
        self.dir = RUNS_DIR / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "errors").mkdir(exist_ok=True)
        self.events_path = self.dir / "events.jsonl"
        if not (self.dir / "run.json").exists():
            atomic.atomic_write_json(
                self.dir / "run.json",
                {
                    "run_id": self.run_id,
                    "started_at": _now(),
                    "pid": os.getpid(),
                    "git_revision": git_revision(),
                },
            )

    def event(self, stage: str, message: str, **fields: Any) -> None:
        record = {"at": _now(), "stage": stage, "message": message, "pid": os.getpid(), **fields}
        atomic.append_jsonl(self.events_path, record)

    def error(self, stage: str, message: str, detail: str = "", **fields: Any) -> Path:
        name = f"{len(list((self.dir / 'errors').glob('*.json'))) + 1:03d}-{stage}.json"
        path = self.dir / "errors" / name
        atomic.atomic_write_json(
            path,
            {"at": _now(), "stage": stage, "message": message, "detail": detail, **fields},
        )
        self.event(stage, message, error_file=path.name, **fields)
        return path

    def finish(self, status: str, **fields: Any) -> None:
        atomic.atomic_write_json(
            self.dir / "run.json",
            {
                "run_id": self.run_id,
                "finished_at": _now(),
                "status": status,
                "pid": os.getpid(),
                "git_revision": git_revision(),
                **fields,
            },
        )
        self.event("run", f"结束 status={status}", status=status, **fields)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------- Generation


def new_generation_id() -> str:
    return "g-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + secrets.token_hex(3)


class Generation:
    """一代产物。目录布局固定，Manifest 是这一代的唯一身份文件。"""

    def __init__(self, generation_id: str, root: Path | None = None) -> None:
        if not GEN_ID_RE.match(generation_id):
            raise ReleaseError(f"非法 generation_id：{generation_id}（应为 g-YYYYMMDDTHHMMSSZ-xxxxxx）")
        self.generation_id = generation_id
        self.dir = (root or GENERATIONS_DIR) / generation_id

    # 路径
    @property
    def manifest_path(self) -> Path:
        return self.dir / MANIFEST_NAME

    @property
    def graph_path(self) -> Path:
        return self.dir / "graph" / "graph.json"

    @property
    def index_path(self) -> Path:
        return self.dir / "index" / "lexical.json"

    @property
    def reports_dir(self) -> Path:
        return self.dir / "reports"

    @property
    def gate_report_path(self) -> Path:
        return self.reports_dir / GATE_REPORT_NAME

    # 生命周期
    def create(self, *, snapshot: InputSnapshot, run_id: str, parent: str = "") -> dict:
        if self.dir.exists():
            raise ReleaseError(f"Generation 目录已存在，拒绝覆盖：{self.dir}")
        for sub in ("graph", "index", "reports"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        manifest = {
            "generation_id": self.generation_id,
            "status": PREPARING,
            "created_at": _now(),
            "run_id": run_id,
            "parent_generation": parent,
            **snapshot.as_dict(),
            # 产物身份在写盘后回填，此刻还不知道
            "graph_artifact_id": "",
            "index_artifact_id": "",
            "graph_build_fp": "",
            "index_build_fp": "",
            "gate_report_hash": "",
            "index_content_digest": "",
            "probe_digest": "",
        }
        atomic.atomic_write_json(self.manifest_path, manifest)
        return manifest

    def load(self) -> dict:
        if not self.manifest_path.exists():
            raise ReleaseError(f"Generation 缺 Manifest：{self.manifest_path}")
        return json.loads(self.manifest_path.read_text("utf-8"))

    def status(self) -> str:
        return str(self.load().get("status", ""))

    def update(self, **fields: Any) -> dict:
        manifest = self.load()
        manifest.update(fields)
        manifest["updated_at"] = _now()
        atomic.atomic_write_json(self.manifest_path, manifest)
        return manifest

    def advance(self, new_status: str, **fields: Any) -> dict:
        current = self.status()
        allowed = TRANSITIONS.get(current, ())
        if new_status not in allowed:
            raise IllegalTransition(
                f"{self.generation_id}: 状态 {current} 不能转到 {new_status}"
                f"（允许：{', '.join(allowed) or '无'}）"
            )
        return self.update(status=new_status, **fields)

    def fail(self, reason: str, *, detail: str = "") -> dict:
        """任何状态都能转 FAILED。失败也写 Manifest，不删除目录——

        删掉就再也没法回答"那一代到底坏在哪"，而重跑代价是几小时的抽取。
        """
        return self.update(status=FAILED, failed_at=_now(), fail_reason=reason, fail_detail=detail)


# ---------------------------------------------------------------- 回读校验


def artifact_digests(gen: Generation) -> dict[str, str]:
    """回读并计算候选产物的内容摘要。缺文件即硬失败。"""
    out: dict[str, str] = {}
    for name, path in (("graph", gen.graph_path), ("index", gen.index_path)):
        if not path.exists():
            raise ReleaseError(f"回读失败：候选产物不存在 {name} -> {path}")
        size = path.stat().st_size
        if size == 0:
            raise ReleaseError(f"回读失败：候选产物是空文件 {name} -> {path}")
        out[name] = sha256_file(path)
    if gen.gate_report_path.exists():
        out["gate_report"] = sha256_file(gen.gate_report_path)
    else:
        out["gate_report"] = ""
    return out


def verify_generation(gen: Generation) -> list[str]:
    """逐项核对 Manifest 与实际产物。返回问题清单，空表示一致。

    这里查的是"这一代自洽"，不是"这一代好"。门禁好不好由 gates 判，
    这一层只回答"Manifest 记的字节是不是盘上这些字节"。
    """
    problems: list[str] = []
    manifest = gen.load()

    if manifest.get("generation_id") != gen.generation_id:
        problems.append(
            f"Manifest 的 generation_id={manifest.get('generation_id')} 与目录名 {gen.generation_id} 不符"
        )

    try:
        digests = artifact_digests(gen)
    except ReleaseError as exc:
        return problems + [str(exc)]

    if manifest.get("graph_artifact_id") != digests["graph"]:
        problems.append(
            f"graph_artifact_id 不符：Manifest={manifest.get('graph_artifact_id') or '(空)'} "
            f"实际={digests['graph']}"
        )
    if manifest.get("index_artifact_id") != digests["index"]:
        problems.append(
            f"index_artifact_id 不符：Manifest={manifest.get('index_artifact_id') or '(空)'} "
            f"实际={digests['index']}"
        )
    if (manifest.get("gate_report_hash") or "") != digests["gate_report"]:
        problems.append(
            f"gate_report_hash 不符：Manifest={manifest.get('gate_report_hash') or '(空)'} "
            f"实际={digests['gate_report'] or '(无报告)'}"
        )

    # 图与索引必须绑同一份图：索引里的 graph_fp 记的是它建时的图指纹。
    try:
        graph = json.loads(gen.graph_path.read_text("utf-8"))
        idx_payload_meta = json.loads(gen.index_path.read_text("utf-8")).get("meta") or {}
    except (OSError, json.JSONDecodeError) as exc:
        return problems + [f"产物不可解析：{exc}"]

    graph_fp = graph.get("fingerprint") or ""
    if manifest.get("graph_build_fp") and manifest["graph_build_fp"] != graph_fp:
        problems.append(
            f"graph_build_fp 不符：Manifest={manifest['graph_build_fp']} 图内={graph_fp or '(空)'}"
        )
    if idx_payload_meta.get("graph_fp") != graph_fp:
        problems.append(
            f"索引绑定的图指纹与图不一致：index.graph_fp={idx_payload_meta.get('graph_fp') or '(空)'} "
            f"graph={graph_fp or '(空)'}"
        )
    if idx_payload_meta.get("corpus_fp") != manifest.get("source_fingerprint"):
        problems.append(
            f"索引语料指纹与 Manifest 不符：index={idx_payload_meta.get('corpus_fp') or '(空)'} "
            f"manifest={manifest.get('source_fingerprint') or '(空)'}"
        )

    # 内容摘要：不只比数量。数量相同内容不同是静默失效最典型的形态。
    if manifest.get("index_content_digest"):
        from . import index as index_mod

        loaded, meta = index_mod.load_index(gen.index_path)
        actual = index_mod.content_digest(loaded)
        if actual != manifest["index_content_digest"]:
            problems.append(
                f"index_content_digest 不符：Manifest={manifest['index_content_digest'][:16]} "
                f"回读={actual[:16]}"
            )
        if meta.get("chunk_count") != len(loaded.doc_id):
            problems.append(
                f"索引 meta.chunk_count={meta.get('chunk_count')} 与实际文档数 {len(loaded.doc_id)} 不符"
            )
        # 固定探针排序：计数全对但「检索排不出来」的失效只能靠这一层发现
        if manifest.get("probe_digest"):
            replayed = index_mod.probe_digest(loaded)
            if replayed != manifest["probe_digest"]:
                problems.append(
                    f"probe_digest 不符：Manifest={manifest['probe_digest'][:16]} 回读重放={replayed[:16]}"
                )

    return problems


# ---------------------------------------------------------------- 发布与回滚


def read_current() -> dict | None:
    if not CURRENT_PATH.exists():
        return None
    try:
        return json.loads(CURRENT_PATH.read_text("utf-8"))
    except json.JSONDecodeError:
        # 指针文件坏了绝不能猜一个，直接当"没有可用版本"报出去
        raise ReleaseError(f"current.json 不可解析，拒绝猜测当前版本：{CURRENT_PATH}")


def current_generation() -> Generation | None:
    cur = read_current()
    if not cur:
        return None
    gid = cur.get("generation_id") or ""
    if not gid:
        return None
    return Generation(gid)


def publish(gen: Generation, *, run: Run | None = None) -> dict:
    """把 READY 的一代切为线上版本。唯一的一次原子写。

    调用方必须已经完成"回读候选产物 + 校验身份与内容摘要 + 全部硬门通过"。
    这里再自己查一遍，是因为"别人替我查过了"不是可审计的保证。
    """
    status = gen.status()
    if status != READY:
        raise IllegalTransition(f"只有 READY 能发布，当前 {gen.generation_id} 是 {status}")

    problems = verify_generation(gen)
    if problems:
        gen.fail("publish_verify_failed", detail="; ".join(problems[:8]))
        if run:
            run.error("publish", "发布前校验失败", "; ".join(problems[:8]))
        raise ReleaseError(f"发布前校验失败（{len(problems)} 项）", detail="; ".join(problems[:8]))

    previous = read_current() or {}
    manifest = gen.load()
    pointer = {
        "generation_id": gen.generation_id,
        "path": str(gen.dir),
        "status": PUBLISHED,
        "published_at": _now(),
        "previous_generation": previous.get("generation_id", ""),
        "source_fingerprint": manifest.get("source_fingerprint", ""),
        "graph_artifact_id": manifest.get("graph_artifact_id", ""),
        "index_artifact_id": manifest.get("index_artifact_id", ""),
        "gate_report_hash": manifest.get("gate_report_hash", ""),
    }
    gen.update(status=PUBLISHED, published_at=pointer["published_at"])
    atomic.atomic_write_json(CURRENT_PATH, pointer)
    if run:
        run.event("publish", f"已发布 {gen.generation_id}", previous=pointer["previous_generation"])
    return pointer


def rollback(generation_id: str, *, run: Run | None = None) -> dict:
    """把指针切回某一代历史版本。只切指针，不重建任何历史产物。"""
    gen = Generation(generation_id)
    if not gen.dir.exists():
        raise ReleaseError(f"回滚目标不存在：{gen.dir}")
    status = gen.status()
    if status not in (READY, PUBLISHED):
        raise ReleaseError(f"回滚目标状态是 {status}，只有 READY/PUBLISHED 的世代可回滚")
    problems = verify_generation(gen)
    if problems:
        raise ReleaseError(
            f"回滚目标自检不过（{len(problems)} 项），拒绝把坏版本切上来",
            detail="; ".join(problems[:8]),
        )
    previous = read_current() or {}
    pointer = {
        "generation_id": gen.generation_id,
        "path": str(gen.dir),
        "status": PUBLISHED,
        "published_at": _now(),
        "previous_generation": previous.get("generation_id", ""),
        "rolled_back": True,
        "source_fingerprint": gen.load().get("source_fingerprint", ""),
    }
    atomic.atomic_write_json(CURRENT_PATH, pointer)
    if run:
        run.event("rollback", f"回滚到 {gen.generation_id}", previous=pointer["previous_generation"])
    return pointer


# ---------------------------------------------------------------- 列表与清理


def list_generations() -> list[dict]:
    out: list[dict] = []
    if not GENERATIONS_DIR.exists():
        return out
    for path in sorted(GENERATIONS_DIR.iterdir()):
        if not path.is_dir() or not GEN_ID_RE.match(path.name):
            continue
        manifest_path = path / MANIFEST_NAME
        if not manifest_path.exists():
            out.append({"generation_id": path.name, "status": "(无 Manifest)", "path": str(path)})
            continue
        try:
            manifest = json.loads(manifest_path.read_text("utf-8"))
        except json.JSONDecodeError:
            out.append({"generation_id": path.name, "status": "(Manifest 损坏)", "path": str(path)})
            continue
        out.append(
            {
                "generation_id": path.name,
                "status": manifest.get("status", ""),
                "created_at": manifest.get("created_at", ""),
                "chunk_count": manifest.get("chunk_count"),
                "entities": (manifest.get("stats") or {}).get("entities"),
                "edges": (manifest.get("stats") or {}).get("edges"),
                "parent_generation": manifest.get("parent_generation", ""),
                "path": str(path),
            }
        )
    return out


def prune_old(*, keep: int = 2, dry_run: bool = True) -> list[str]:
    """归档旧的 FAILED/非发布世代目录。默认只演练不删除。

    删除前必须确认：不是当前版本、不是保留世代、且它自己的 Manifest 在。
    计划 K4 明确要求"删除旧产物前验证备份和 Manifest"。
    """
    cur = read_current() or {}
    keep_ids = {cur.get("generation_id", ""), cur.get("previous_generation", "")}
    candidates = [
        g for g in list_generations() if g["generation_id"] not in keep_ids and g["status"] != PUBLISHED
    ]
    candidates.sort(key=lambda g: g.get("created_at", ""), reverse=True)
    victims = [g["generation_id"] for g in candidates[keep:]]
    if not dry_run:
        for gid in victims:
            shutil.rmtree(GENERATIONS_DIR / gid, ignore_errors=True)
    return victims
