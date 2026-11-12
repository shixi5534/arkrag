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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import atomic, config, fingerprint, source

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

# 覆盖状态。默认是"完整"，只有构建脚本显式申报才可能是"不完整"。
# 这一项不参与「这一代自不自洽」的判定（那是 verify_generation 的事），
# 只拦一件事：把不完整输入建成的一代当完整世代发布出去。
COVERAGE_COMPLETE = "complete"
COVERAGE_INCOMPLETE = "incomplete"


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
    """构建前固定下来的输入身份。构建过程中输入再被改动，也不影响这一代。

    分层是因为一层代表不了另一层（计划 D3）：块数相同内容全换，块清单摘要会变；
    块清单相同而图变了，是抽取结果变了；抽取相同而图仍变了，就是建图配置变了。
    少了任何一层，「产物变了」这件事就落不到具体哪一层上。
    """

    source_fingerprint: str
    chunk_manifest_hash: str
    chunk_count: int
    extraction_manifest_hash: str
    extraction_count: int
    scene_fingerprint: str = ""
    chunk_fingerprint: str = ""
    versions: dict[str, str] = field(default_factory=dict)
    git_revision: str = ""
    # 默认值引用 source.UNVERIFIED 而不是手抄字面量：默认值决定了"没写等于什么"，
    # 偏向哪一侧是有语义的（未验证 = 不得发布），不能在两处各写一份。
    source_status: str = source.UNVERIFIED

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_fingerprint": self.source_fingerprint,
            "source_status": self.source_status,
            "scene_fingerprint": self.scene_fingerprint,
            "chunk_fingerprint": self.chunk_fingerprint,
            "chunk_manifest_hash": self.chunk_manifest_hash,
            "chunk_count": self.chunk_count,
            "extraction_manifest_hash": self.extraction_manifest_hash,
            "extraction_count": self.extraction_count,
            "versions": dict(self.versions),
            "git_revision": self.git_revision,
        }


def capture_inputs(
    *,
    chunks: list[dict],
    records: dict[str, dict],
    source_fp: str,
    scenes: list[dict] | None = None,
    source_status: str = source.UNVERIFIED,
) -> InputSnapshot:
    return InputSnapshot(
        source_fingerprint=source_fp,
        source_status=source_status,
        scene_fingerprint=fingerprint.scene_fingerprint(scenes) if scenes else "",
        chunk_fingerprint=fingerprint.chunk_fingerprint(
            chunks,
            params={
                "target_chars": config.CHUNK_TARGET_CHARS,
                "min_chars": config.CHUNK_MIN_CHARS,
                "max_chars": config.CHUNK_MAX_CHARS,
                "overlap_turns": config.CHUNK_OVERLAP_TURNS,
                "t1_categories": list(config.T1_CATEGORIES),
                "pack_size": 200,
            },
        ),
        chunk_manifest_hash=chunk_manifest_hash(chunks),
        chunk_count=len(chunks),
        extraction_manifest_hash=extraction_manifest_hash(records),
        extraction_count=len(records),
        versions={
            "schema_version": config.SCHEMA_VERSION,
            "prompt_version": config.PROMPT_VERSION,
            # 版本字符串只用于展示；下面两个内容摘要才是身份（计划 D3）
            "prompt_sha256": fingerprint.prompt_digest(),
            "vocab_sha": fingerprint.vocab_digest(),
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
    def assertion_path(self) -> Path:
        """逐观察断言（仅 `ark-rag-3`）。与 graph.json 同目录，同属图产物家族。"""
        return self.dir / "graph" / "assertions.jsonl"

    @property
    def adjacency_path(self) -> Path:
        """有方向邻接（仅 `ark-rag-3`）。"""
        return self.dir / "graph" / "adjacency.json"

    @property
    def index_path(self) -> Path:
        return self.dir / "index" / "lexical.json"

    @property
    def chunks_path(self) -> Path:
        """这一代实际吃进去的块正文（T1 子集，JSONL）。

        为什么必须放在世代目录里：块正文是**引用校验的判据**。引用偏移
        (`evidence_start/end`) 是在这一代自己的块正文上算出来的；如果消费者
        从别处拿块正文，`text[start:end] == evidence` 就会对不上，而那种对不上
        会被下游当成「引用不成立」静默剔除——表现为答得短，而不是报错。
        快照里的 `chunk_manifest_hash` 已经能唯一标识这一批块，但**身份不等于
        位置**：没有这份文件，谁也不知道该去哪个路径取那一批块。
        """
        return self.dir / "chunks.jsonl"

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
    #
    # `UnicodeDecodeError` 必须一起接住：产物被截断或翻字节后 `read_text("utf-8")`
    # 抛的是它（`ValueError` 的子类，不是 `OSError`）。只接 OSError/JSONDecodeError
    # 的话，一份坏掉的索引会让 `verify_generation` 自己崩掉——而它的职责正是
    # "报告这份产物不合格"。实测：把 lexical.json 第 100 字节翻转，旧代码直接抛
    # `UnicodeDecodeError`，`/ready` 于是返回 500 而不是"未就绪 + 原因"。
    try:
        graph = json.loads(gen.graph_path.read_text("utf-8"))
        idx_payload_meta = json.loads(gen.index_path.read_text("utf-8")).get("meta") or {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return problems + [f"产物不可解析：{exc}"]

    graph_fp = graph.get("fingerprint") or ""
    if manifest.get("graph_build_fp") and manifest["graph_build_fp"] != graph_fp:
        problems.append(
            f"graph_build_fp 不符：Manifest={manifest['graph_build_fp']} 图内={graph_fp or '(空)'}"
        )

    # ---- ark-rag-3 专属：断言与邻接是**独立产物**，必须各自能证明身份 ------
    #
    # 只在这一代声明了 `graph_schema == GRAPH_SCHEMA_IDENTITY` 时才查。
    # 旧代（ark-rag-1）根本没有这两个文件，硬查会让每一代历史产物都"不合格"——
    # 那不是发现缺陷，那是把新规则追溯适用于旧产物。
    graph_schema = str(graph.get("schema_version") or "")
    if graph_schema == config.GRAPH_SCHEMA_IDENTITY:
        declared = str(manifest.get("graph_schema") or "")
        if declared != graph_schema:
            problems.append(
                f"Manifest 的 graph_schema={declared or '(空)'} 与图内 schema_version={graph_schema} 不符"
            )
        for key, path, label in (
            ("assertion_artifact_id", gen.assertion_path, "断言"),
            ("adjacency_artifact_id", gen.adjacency_path, "邻接"),
        ):
            want = str(manifest.get(key) or "")
            if not want:
                problems.append(f"Manifest 缺 {key}：这一代声明了 ark-rag-3，{label}产物必须可追踪")
                continue
            if not path.exists():
                problems.append(f"{label}产物不存在：{path}")
                continue
            actual = sha256_file(path)
            if actual != want:
                problems.append(
                    f"{key} 不符：Manifest={want[:16]} 实际={actual[:16]}"
                )
        # 断言条数必须与 Manifest 记的一致：条数是"这一代到底抽出了多少条观察"的
        # 唯一外部可核对口径，缺了它，断言文件被截断一半也没人知道。
        want_count = manifest.get("assertion_count")
        if want_count is None:
            problems.append("Manifest 缺 assertion_count：无法核对断言产物是否完整")
        elif gen.assertion_path.exists():
            try:
                with gen.assertion_path.open("r", encoding="utf-8") as fh:
                    actual_count = sum(1 for line in fh if line.strip())
            except OSError as exc:
                problems.append(f"断言产物不可读：{exc}")
            else:
                if int(want_count) != actual_count:
                    problems.append(
                        f"assertion_count 不符：Manifest={want_count} 实际行数={actual_count}"
                    )
    elif manifest.get("graph_schema"):
        problems.append(
            f"Manifest 的 graph_schema={manifest.get('graph_schema')} 与图内 schema_version={graph_schema or '(空)'} 不符"
        )

    if idx_payload_meta.get("graph_fp") != graph_fp:
        problems.append(
            f"索引绑定的图指纹与图不一致：index.graph_fp={idx_payload_meta.get('graph_fp') or '(空)'} "
            f"graph={graph_fp or '(空)'}"
        )
    # 索引必须绑定**图产物的内容摘要**，不匹配就说明这份倒排是对着另一张图建的（计划 D3）。
    # 只绑 graph_fp（配置指纹）不够：同一份配置重建两次图，配置指纹一样而内容未必一样。
    idx_graph_aid = idx_payload_meta.get("graph_artifact_id") or ""
    if not idx_graph_aid:
        problems.append("索引 meta 缺 graph_artifact_id：无法证明这份倒排对着哪一份图产物")
    elif idx_graph_aid != digests["graph"]:
        problems.append(
            f"索引绑定的图产物摘要与实际图不符：index.graph_artifact_id={idx_graph_aid[:16]} "
            f"实际图={digests['graph'][:16]}"
        )
    if idx_payload_meta.get("corpus_fp") != manifest.get("source_fingerprint"):
        problems.append(
            f"索引语料指纹与 Manifest 不符：index={idx_payload_meta.get('corpus_fp') or '(空)'} "
            f"manifest={manifest.get('source_fingerprint') or '(空)'}"
        )
    # 提示词与词表的内容摘要必须与 Manifest 记的一致：改提示词忘改版本号这类漏洞，
    # 只有在"配置摘要与实际内容比对"时才暴露（计划 D3）。
    versions = manifest.get("versions") or {}
    for key, meta_key in (("prompt_sha256", "prompt_sha256"), ("vocab_sha", "vocab_sha")):
        want = versions.get(key) or ""
        got = idx_payload_meta.get(meta_key) or ""
        if want and got and want != got:
            problems.append(f"索引 {meta_key} 与 Manifest 不符：index={got[:16]} manifest={want[:16]}")
    # 来源未验证的世代不得冒充 Release（计划 D4）
    #
    # 判据必须和 source.strength() 共用同一张表，不能只认字面量
    # "source_unverified"。否则 source_status 写成 ""、"VERIFIED_ZIP"、
    # "verified_zip "（尾空格）、"bogus" 这些**同样不是已验证**的取值，
    # 都能绕过这道门 —— 门只拒绝一个特定拼写，等于没有门。
    # 改成白名单判定：只有明确认定已验证的两个取值才放行。
    _source_status = manifest.get("source_status") or ""
    if _source_status not in (source.VERIFIED_ZIP, source.VERIFIED_TREE):
        problems.append(
            f"source_status={_source_status!r}：语料来源未能验证，"
            f"按计划 D4 不得作为 Release 发布"
        )

    # 分层身份必须逐层对齐（计划 D3）：Manifest 记的是这一代吃进去的每一层输入，
    # 图里记的是它自己认为吃进去的每一层。两者不一致就说明图不是从这一代输入建的。
    identities = graph.get("identities") or {}
    if identities:
        for key, want in (
            ("source_fp", manifest.get("source_fingerprint")),
            ("scene_fp", manifest.get("scene_fingerprint")),
            ("chunk_fp", manifest.get("chunk_fingerprint")),
            ("chunk_manifest_hash", manifest.get("chunk_manifest_hash")),
            ("extraction_manifest_hash", manifest.get("extraction_manifest_hash")),
        ):
            got = identities.get(key) or ""
            if want and got and got != want:
                problems.append(
                    f"图记录的 {key} 与 Manifest 不符：graph={got[:16]} manifest={str(want)[:16]}"
                )
            elif want and not got:
                problems.append(f"图缺 {key}：无法证明这张图吃的是本代输入")
    elif manifest.get("chunk_fingerprint"):
        problems.append("图缺 identities：Manifest 有分层身份而图完全没有记录，无法逐层对齐")

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

    # 块正文身份：世代自带块文件时，它必须就是 Manifest 记的那一批。
    #
    # 只在这一代**有**该文件时检查——旧世代（K2 之前建的）没有，不能因此判红，
    # 它们由 `pipeline.resolve_chunks()` 在消费侧 fail-closed（借全局文件但必须
    # 能复现本代 hash）。这里管的是新形态：文件在盘上不等于内容对。
    if gen.chunks_path.exists():
        try:
            records = [
                json.loads(line)
                for line in gen.chunks_path.read_text("utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            problems.append(f"世代自带块正文不可解析：{exc}")
        else:
            t1 = [r for r in records if config.in_t1(str(r.get("category") or ""))]
            if not t1:
                problems.append(f"世代自带块正文为空（{gen.chunks_path}）")
            want = str(manifest.get("chunk_manifest_hash") or "")
            if want:
                got = chunk_manifest_hash(t1)
                if got != want:
                    problems.append(
                        f"世代自带块正文与 Manifest 不符："
                        f"chunk_manifest_hash Manifest={want[:16]} 文件={got[:16]}"
                    )

    # ---- 门禁结论：**报告里写过非 PASS 就不许发布** ----------------------
    #
    # 此前这一层只核"报告字节还是不是 Manifest 记的那份"（`gate_report_hash`），
    # 从不读报告里的 `verdict`。两个后果：
    #
    #   1. `publish()` 的 docstring 写"发布前自己再查一遍"，但查的全是**字节自洽**，
    #      不看门禁结论——一份 `verdict:"FAIL"` 的报告能 100% 通过校验。标准构建
    #      路径有 `build_generation.py` 的 `hard_fails` 拦着，所以平时不炸；但任何
    #      **旁路**（手工把世代标成 READY 再 publish，或 `rollback` 到一代被改过的
    #      历史产物）就只剩这一层，而这一层原本不判好坏。
    #   2. 报告缺失时 `gate_report_hash` 是空串；若 Manifest 里也是空串，
    #      `"" == ""` 会判"一致"——空对空的假通过。
    #
    # 判定口径与 `build_generation.py` 完全一致（`verdict != "PASS"` 即不通过），
    # 不另造第二套。
    if not gen.gate_report_path.exists():
        problems.append(f"门禁报告不存在：{gen.gate_report_path}")
    else:
        try:
            rows = json.loads(gen.gate_report_path.read_text("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            problems.append(f"门禁报告不可解析：{exc}")
        else:
            if isinstance(rows, dict):
                rows = rows.get("gates") or rows.get("items") or []
            if not isinstance(rows, list) or not rows:
                problems.append("门禁报告为空：没有可核的门禁行")
            else:
                bad = [
                    str(r.get("name") or "?")
                    for r in rows
                    if isinstance(r, dict) and str(r.get("verdict") or "") != "PASS"
                ]
                if bad:
                    problems.append(
                        f"门禁未全部 PASS（{len(bad)}/{len(rows)} 项）：{', '.join(bad[:5])}"
                    )

    return problems


# ---------------------------------------------------------------- 发布与回滚


def read_current() -> dict | None:
    """读发布指针。历史指针里可能带 `path`，读的时候就当它不存在。

    为什么要容忍而不是报错：2026-10 之前的指针写了一个 `"path"` 字段，存的是
    `generations/<gid>` 的**绝对机台路径**。那个字段全仓库没有任何一处读取
    （`release_cli current` 只打印身份字段），因为目录本来就能从 id 推导：
    `Generation(gid).dir == GENERATIONS_DIR / gid`。存一份能推导的东西，等于存
    一份会过期的、机台相关的副本——进 git 后换机器就是错的。

    所以这里**只针对 `path` 特判**：弹出它，其余字段原样保留。不静默丢弃其它
    未知字段，是因为那些可能是未来版本加的（老代码读新指针时把未来字段吃掉，
    等于让"降级一次就把信息丢了"变成静默行为）。
    """
    if not CURRENT_PATH.exists():
        return None
    try:
        cur = json.loads(CURRENT_PATH.read_text("utf-8"))
    except json.JSONDecodeError:
        # 指针文件坏了绝不能猜一个，直接当"没有可用版本"报出去
        raise ReleaseError(f"current.json 不可解析，拒绝猜测当前版本：{CURRENT_PATH}")
    if isinstance(cur, dict):
        cur.pop("path", None)
    return cur


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

    manifest = gen.load()
    # 覆盖不全的一代不许发布。这条不是「自洽性」检查——不完整的一代可以完全自洽
    # （Manifest 记的字节就是盘上这些字节），但把它当完整世代切上线，等于用
    # 13,764 块的图冒充 13,798 块的语料，事后没有任何指针能看出少了 34 块。
    # 判据看两处：显式申报的 coverage_status，以及 coverage_missing 是否非空。
    # 后者不是为了防手滑——它兜住的是"Manifest 里记了缺哪些块、却没写状态"的形态，
    # 缺字段时按 complete 处理是为了兼容历史世代，但记了缺块就是记了。
    coverage = str(manifest.get("coverage_status") or COVERAGE_COMPLETE)
    missing = manifest.get("coverage_missing") or []
    if coverage != COVERAGE_COMPLETE or missing:
        detail = f"覆盖 {coverage}，缺 {len(missing)} 块：{'; '.join(str(c) for c in missing[:5])}"
        if run:
            run.error("publish", "这一代覆盖不全，拒绝发布", detail)
        raise ReleaseError(
            f"{gen.generation_id} 覆盖不全（{coverage}），拒绝发布为完整世代", detail=detail
        )

    problems = verify_generation(gen)
    if problems:
        gen.fail("publish_verify_failed", detail="; ".join(problems[:8]))
        if run:
            run.error("publish", "发布前校验失败", "; ".join(problems[:8]))
        raise ReleaseError(f"发布前校验失败（{len(problems)} 项）", detail="; ".join(problems[:8]))

    previous = read_current() or {}
    # 指针里**不存** `path`：目录能从 id 推导（`Generation(gid).dir`），
    # 存一份就等于存了一份会过期的绝对机台路径——指针进 git，换台机器 clone
    # 下来那个字符串全是错的，而它本来一个读取方都没有。
    pointer = {
        "generation_id": gen.generation_id,
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
    # 回滚到"当前已经是"的那一代是空操作。若照写不误，指针会声称
    # 「这一代替换了它自己」，`previous_generation` 从此指向自身，
    # 下次真回滚时就没有可退的目标了。
    if previous.get("generation_id") == gen.generation_id:
        raise ReleaseError(
            f"{gen.generation_id} 已经是当前世代，无需回滚（previous_generation 会退化成指向自身）"
        )
    manifest = gen.load()
    # 回滚指针必须与 publish 指针同构：只记 generation_id 的话，
    # `release_cli current` 事后查不出线上跑的是哪份图、哪份索引、哪份门禁报告。
    # 同构指**身份字段**：publish 的 8 个键这里一个不少，多出的 `rolled_back`
    # 是回滚独有的语义标记（记录"这一代是被回滚切上来的"），不是身份。
    # 两者都**不含** `path`：目录能从 `Generation(gid).dir` 推导，存绝对路径只会让
    # 指针在别的机器上指向不存在的地方。这一条有测试盯着（键集必须完全相同）。
    pointer = {
        "generation_id": gen.generation_id,
        "status": PUBLISHED,
        "published_at": _now(),
        "previous_generation": previous.get("generation_id", ""),
        "rolled_back": True,
        "source_fingerprint": manifest.get("source_fingerprint", ""),
        "graph_artifact_id": manifest.get("graph_artifact_id", ""),
        "index_artifact_id": manifest.get("index_artifact_id", ""),
        "gate_report_hash": manifest.get("gate_report_hash", ""),
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
    """归档旧的 FAILED 世代目录。默认只演练不删除。

    ## 为什么候选条件必须是 `status == FAILED`，而不是 `!= PUBLISHED`

    旧实现在这里写的是 `status != PUBLISHED`，于是 READY 的世代全都算"可清候选"，
    再按 `keep` 从最近往里留 N 个。实测（12 个世代，线上指针 `55dfd7`、
    `previous` 为 `606ea9`）：`prune_old(keep=2)` 拟清 7 个，其中 5 个是 READY
    可服务世代，含**上一代** `g-20261031T035231Z-4eafae`。真跑一次
    `dry_run=False` 就把上一代和多个候选一起删了，直接违反计划 K4 第一条
    「保留至少上一代完整有效 Generation」。

    根因是「保留集合」被当成了「候选集合的过滤器」：`keep_ids` 只有
    `current` 与 `previous` 两个 id，而**上一代可服务世代未必在这两个 id 里**
    （现场就是：`current=55dfd7`/`previous=606ea9`，真正的上一代 `4eafae`
    两个都不是）。任何靠"从候选里挑 N 个留着"的写法都会在 keep 之外
    再丢世代——因为候选集合本身就不该包含 READY。

    READY 是**可服务但未发布**的合法状态，它存在的意义就是随时能切上线；
    能清的只有 FAILED——那些已经跑不出产物、留着只为留证。所以：
    候选只收 FAILED，`keep` 的含义收窄为"最近 N 个 FAILED 留作证据"，
    READY/PUBLISHED 一个都不进候选，也就永远不会被删。

    删除前必须确认：不是当前版本、不是保留世代、且它自己的 Manifest 在。
    计划 K4 明确要求"删除旧产物前验证备份和 Manifest"。
    """
    cur = read_current() or {}
    keep_ids = {cur.get("generation_id", ""), cur.get("previous_generation", "")}
    candidates = [
        g for g in list_generations() if g["generation_id"] not in keep_ids and g["status"] == FAILED
    ]
    candidates.sort(key=lambda g: g.get("created_at", ""), reverse=True)
    victims = [g["generation_id"] for g in candidates[keep:]]
    if not dry_run:
        for gid in victims:
            shutil.rmtree(GENERATIONS_DIR / gid, ignore_errors=True)
    return victims


# ---------------------------------------------------------------- 观察期与归档

# 归档根。默认与 `generations/` 同级（`data/arkrag/archive/<gid>/`），
# 而不是放进 `generations/` 里面：`list_generations()` 遍历的是
# `GENERATIONS_DIR.iterdir()` 且按 `GEN_ID_RE` 过滤，把归档目录塞进去会被当成
# 一个新世代列出来（目录名同样是 `g-…` 形态），于是"盘上有几代"这件事开始说谎。
ARCHIVE_DIR = config.DATA_DIR / "archive"

# 观察期天数默认值。计划 K4 第 4 条只写"新版稳定运行约定观察期后才归档旧版"，
# 没有给数字；这里的 7 天是**本项目自己定的约定**，理由三条：
#
# 1. 一周覆盖完整的工作日+周末流量形态。Gold Set 探针与灰度对照都是在某个
#    时间窗里取的样，出现过的失效（拒答翻转、引用定位失败）里有相当一部分
#    只在周末的低并发分布下显形——3 天窗口会整段跳过周末。
# 2. 归档是**不可逆的重量级动作**：现场一代 137–288 MB，动作是"复制一遍 +
#    逐字节核对 + 删原件"。多留一周磁盘的代价远小于把还在出问题的旧版归档掉。
# 3. 7 是整数天，用 `_now()` 的秒级 UTC 时间戳判定时不需要月长/闰年特判。
#
# 可用 `ARKRAG_OBSERVATION_DAYS` 覆盖（演练与测试要用），但默认值必须写在
# 代码里：只从环境变量取的话，"没人设过"就等于"没有观察期"，那是 fail-open。
OBSERVATION_DAYS = 7
ENV_OBSERVATION_DAYS = "ARKRAG_OBSERVATION_DAYS"


def _observation_days() -> int:
    """观察期天数。环境变量非法时抛 ReleaseError（由调用方 fail-closed）。"""
    raw = os.environ.get(ENV_OBSERVATION_DAYS)
    if raw is None or not str(raw).strip():
        return OBSERVATION_DAYS
    try:
        value = int(str(raw).strip())
    except ValueError:
        raise ReleaseError(
            f"{ENV_OBSERVATION_DAYS}={raw!r} 不是整数，拒绝猜一个观察期天数"
        ) from None
    if value < 0:
        raise ReleaseError(f"{ENV_OBSERVATION_DAYS}={raw!r} 为负数，观察期不能是负的")
    return value


def _parse_utc(value: str) -> datetime | None:
    """把 `_now()`/`publish()` 产出的时间戳解析成带时区的 UTC 时间。

    `_now()` 的格式是 `time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())`——
    **带 Z 后缀**（Zulu/UTC）。`datetime.fromisoformat()` 在 3.11 之后能直接吃
    它，但这里仍显式把 `Z` 换成 `+00:00` 再解析：解析逻辑不该依赖某个小版本的
    宽松行为，换台机器换条 Python 就"观察期算不出来"是不可接受的。
    解析不了返回 None，由调用方 fail-closed，**不**猜当前时间。
    """
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:  # 不带时区的一律按 UTC 解释（与 _now() 同口径）
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def observation_status(*, now: str | None = None) -> dict:
    """当前线上世代是否已过观察期（计划 K4 第 4 条）。

    返回 `{'generation_id','published_at','days_elapsed','required_days','complete','reason'}`。

    ## 为什么一切异常都落到 `complete=False`

    这个函数的返回值决定"旧版能不能被归档"。把不确定性往"已满"那边倒是
    fail-open：指针坏掉、`published_at` 是空串、时间格式换了个写法——任何一处
    都会让**刚上线几分钟**的新版被判成"已过观察期"，旧版随即被复制归档并从
    世代目录删掉。反过来（判成"未满"）的代价只是多留几天磁盘。
    所以：`published_at` 缺失或无法解析 → `complete=False` 并写明原因，
    **不当作 0 天，也不当作已满**；`now` 早于 `published_at`（时钟回拨/手改
    时间戳）同样按未满处理，因为时间线本身已经不可信。

    `now` 可注入是为可测：否则要验证"满 7 天"就得真等 7 天。
    """
    out: dict = {
        "generation_id": "",
        "published_at": "",
        "days_elapsed": None,
        "required_days": OBSERVATION_DAYS,
        "complete": False,
        "reason": "",
    }
    try:
        required = _observation_days()
    except ReleaseError as exc:
        out["reason"] = f"观察期配置无效：{exc}（按未满处理）"
        return out
    out["required_days"] = required

    # 指针文件本身坏掉时 `read_current()` 抛 ReleaseError——这里不吞它：
    # "指针不可解析"和"指针里没写时间"不是同一件事，前者必须让人去修文件，
    # 后者只是这一代没记发布时间。
    cur = read_current() or {}
    if not cur:
        out["reason"] = "没有发布指针（current.json 不存在）：无法判定观察期，按未满处理"
        return out
    gid = str(cur.get("generation_id") or "")
    published = str(cur.get("published_at") or "")
    out["generation_id"] = gid
    out["published_at"] = published
    if not gid:
        out["reason"] = "发布指针缺 generation_id：无法判定观察期起点，按未满处理"
        return out

    start = _parse_utc(published)
    if start is None:
        out["reason"] = (
            f"发布指针的 published_at={published!r} 缺失或无法解析："
            f"无法判定观察期起点，按未满处理（不当作 0 天，也不当作已满）"
        )
        return out

    now_text = _now() if now is None else str(now)
    now_dt = _parse_utc(now_text)
    if now_dt is None:
        out["reason"] = f"now={now_text!r} 无法解析：无法判定观察期，按未满处理"
        return out

    days = (now_dt - start).total_seconds() / 86400.0
    out["days_elapsed"] = round(days, 4)
    if days < 0:
        out["reason"] = (
            f"published_at={published} 晚于当前时间 {now_text}（差 {-days:.2f} 天）："
            f"时间线不可信，按未满处理"
        )
        return out
    if days >= required:
        out["complete"] = True
        out["reason"] = f"线上世代 {gid} 已运行 {days:.2f} 天，达到观察期 {required} 天"
    else:
        out["reason"] = (
            f"线上世代 {gid} 仅运行 {days:.2f} 天，未达观察期 {required} 天"
            f"（还差 {required - days:.2f} 天），旧版不得归档"
        )
    return out


def _heavy_artifacts(gen: Generation) -> list[Path]:
    """这一代的"重量级产物"清单（图/索引整个目录 + 世代自带块正文）。

    以 `Generation` 的属性为准，而不是硬编码文件名：`graph/` 里除了
    `graph.json` 还有 ark-rag-3 的 `assertions.jsonl`（68.7 MB）与
    `adjacency.json`（26.4 MB），只搬 `graph.json` 会留下两个更大的文件在
    世代目录里，"归档"就成了把目录变小一点而已。
    `index/` 同理（可能还有词表/分片）。`chunks.jsonl` 在 `gen.dir` 根下。
    """
    out: list[Path] = []
    for directory in (gen.graph_path.parent, gen.index_path.parent):
        if directory.is_dir():
            out.extend(sorted(p for p in directory.rglob("*") if p.is_file()))
    if gen.chunks_path.is_file():
        out.append(gen.chunks_path)
    return out


def archive_old(
    generation_id: str,
    *,
    dry_run: bool = True,
    archive_root: Path | None = None,
) -> dict:
    """把旧世代的重量级产物归档到 `archive/<gid>/`，再从世代目录删掉。默认只演练。

    ## 与 `prune_old()` 的职责边界（两者不能互相替代）

    `prune_old()` 清的是 **FAILED**：那些跑不出产物的世代，留着只为留证，
    证据价值随轮次衰减，所以它可以**直接删**（连归档都不做）。
    本函数归档的是 **还能服务/曾经服务过的** 世代：它会先复制、逐字节核对、
    写一份 `archive-manifest.json`，然后才允许删原目录里的重量级产物。

    因此"目标是 FAILED"在这里是**拒绝理由**，不是候选条件：失败世代的产物
    本身可能就是事后定位缺陷的证据（`verify_generation()` 对它们必然不过），
    把一份坏产物复制进归档区再删原件，等于用"归档"这个动作给坏证据盖上
    "已妥善保存"的章。失败世代该由 `prune_old()` 按其规则处理。

    ## 为什么观察期检查放在最前面（计划 K4 第 4 条）

    归档的全部风险来自"删原件"这一步：复制再核对只能保证归档副本是好的，
    保证不了**新版真的稳定**。观察期未满就归档 = 一旦新版出问题，回滚目标
    已经被搬走并删掉图/索引，回滚过去只剩一个读不出产物的目录。
    判据一律走 `observation_status()`，`complete=False` 就拒绝——包括
    `published_at` 缺失这种"判不出来"的情形（fail-closed）。

    ## 顺序不许颠倒

        复制 -> 逐文件核对 SHA-256 -> 写 archive-manifest.json -> 才删原件

    任何一步失败都不删原件。反过来（先删后复制/边删边复制）意味着中途失败
    就同时失去原产物和完整备份。

    ## 保留什么

    `manifest.json` 与 `reports/gates.json` 不删。Manifest 是这一代在
    `list_generations()` 里存在的唯一依据——删了它，这一代就从列表里消失，
    而 `current.json`/回滚历史里可能仍然提到这个 id，指针就查不到东西了。
    `reports/gates.json` 是 Manifest 里 `gate_report_hash` 的被引对象，也是
    "这一代当时过了哪些门"的唯一记录，体积可忽略。

    默认 `dry_run=True`（与 `prune_old()` 一致）：演练只回答"会搬哪些文件、
    多少字节"，盘上一个字节都不动。

    ## 归档之后这一代的 `verify_generation()` 必然不过

    这是设计结果，不是缺陷：图/索引已搬走，`artifact_digests()` 会因为文件
    不存在而报"回读失败"。这一代的完整性证据此后以 `archive-manifest.json`
    为唯一依据（每个文件的相对路径、字节数、SHA-256、原路径）。
    所以归档是**不可逆**的动作——要复原必须人工按 archive-manifest 搬回来，
    这也正是"观察期未满一律拒绝"必须存在的理由。
    """
    gen = Generation(generation_id)
    root = Path(archive_root) if archive_root is not None else ARCHIVE_DIR
    dest = root / generation_id
    cur = read_current() or {}

    # 逐条收集拒绝理由，一条都不许被合并成"不满足条件"：
    # 事后看日志的人必须能分辨是"观察期没过"还是"这一代自己就坏了"。
    problems: list[str] = []

    if not gen.dir.exists():
        raise ReleaseError(f"归档目标不存在：{gen.dir}")

    if cur.get("generation_id") == generation_id:
        problems.append(
            f"目标是当前线上世代（current.json 指向 {generation_id}）："
            f"线上版本不得归档，先发布下一代再考虑"
        )

    status = ""
    if gen.manifest_path.exists():
        try:
            status = gen.status()
        except ReleaseError as exc:
            problems.append(f"Manifest 读不出状态：{exc}")
    else:
        problems.append(f"目标缺 Manifest（{gen.manifest_path}）：无法证明它是什么，拒绝归档")
    if status == FAILED:
        problems.append(
            f"目标状态是 FAILED：失败世代留作失败证据，不是归档对象"
            f"（其清理归 prune_old() 管，见其 docstring）"
        )

    observation = observation_status()
    if not observation["complete"]:
        problems.append(f"观察期未满：{observation['reason']}")

    if not problems and status != FAILED:
        # 坏版本不许归档：它的产物自检不过，说明"原件"本身就不是可复原的证据；
        # 更糟的是把一个 verify 不过的目录搬走，等于让后来人以为它当年是好的。
        try:
            verify_problems = verify_generation(gen)
        except ReleaseError as exc:
            verify_problems = [str(exc)]
        if verify_problems:
            problems.append(
                f"目标自检不过（{len(verify_problems)} 项），坏版本不得归档："
                + "; ".join(verify_problems[:6])
            )

    if dest.exists():
        problems.append(f"归档目标已存在（{dest}）：拒绝覆盖已有归档，先人工确认")

    if problems:
        raise ReleaseError(
            f"{generation_id} 不满足归档条件（{len(problems)} 项），拒绝归档",
            detail=" | ".join(problems),
        )

    files = _heavy_artifacts(gen)
    if not files:
        raise ReleaseError(
            f"{generation_id} 没有可归档的重量级产物（graph/ index/ chunks.jsonl 都不在）",
            detail=f"目标目录 {gen.dir}",
        )

    plan = [
        {
            "path": p.relative_to(gen.dir).as_posix(),
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
            "source": str(p),
        }
        for p in files
    ]
    total_bytes = sum(item["bytes"] for item in plan)
    result: dict = {
        "generation_id": generation_id,
        "dry_run": dry_run,
        "archive_root": str(root),
        "archive_dir": str(dest),
        "files": plan,
        "file_count": len(plan),
        "total_bytes": total_bytes,
        "kept": ["manifest.json", "reports/gates.json"],
        "observation": observation,
        "removed": [],
    }
    if dry_run:
        return result

    # ---- 真归档：复制 -> 核对 -> 写 archive-manifest -> 才删原件 ----
    created_root = not root.exists()
    try:
        for item in plan:
            src = gen.dir / item["path"]
            dst = dest / item["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        # 逐文件核对：不比"复制了几份"，比"归档副本的字节是不是原件那些字节"。
        for item in plan:
            dst = dest / item["path"]
            if not dst.is_file():
                raise ReleaseError(f"归档副本缺失：{item['path']} -> {dst}")
            got_bytes = dst.stat().st_size
            got_sha = sha256_file(dst)
            if got_bytes != item["bytes"] or got_sha != item["sha256"]:
                raise ReleaseError(
                    f"归档副本核对失败：{item['path']} "
                    f"原件 {item['bytes']}B/{item['sha256'][:16]} "
                    f"副本 {got_bytes}B/{got_sha[:16]}"
                )

        archive_manifest = {
            "generation_id": generation_id,
            "archived_at": _now(),
            "source_dir": str(gen.dir),
            "archive_dir": str(dest),
            "status": status,
            "observation": observation,
            "total_bytes": total_bytes,
            "files": plan,
            "kept": ["manifest.json", "reports/gates.json"],
        }
        atomic.atomic_write_json(dest / "archive-manifest.json", archive_manifest)
    except BaseException:
        # 复制/核对/写清单任一步失败：**原件一个字节都不动**。
        # 只清掉我们自己刚造出来、且已知不完整的归档副本（否则下次会被
        # "归档目标已存在"挡住，人还得先去手工删）。
        shutil.rmtree(dest, ignore_errors=True)
        if created_root:
            shutil.rmtree(root, ignore_errors=True)
        raise

    # ---- 核对全部通过，才删原件 ----
    for item in plan:
        src = gen.dir / item["path"]
        src.unlink(missing_ok=True)
        result["removed"].append(item["path"])
    for directory in (gen.graph_path.parent, gen.index_path.parent):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()

    result["archive_manifest"] = str(dest / "archive-manifest.json")
    return result
