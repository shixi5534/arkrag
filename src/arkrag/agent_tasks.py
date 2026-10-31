"""子代理抽取的任务下发与产物回收。

抽取由 Qoder 子代理本体完成（签字七），不走任何模型网关。为了让下游 L4-L7 不用改：
  1. 任务文件带 chunk_id / content_key / 完整 prompt，子代理只负责按提示词输出 JSON；
  2. 回收时仍做结构校验，并写进与云端路径完全相同的缓存（model 固定为 AGENT_MODEL）；
  3. 缓存值是子代理原始输出，归一化版本不进键（决策三）。

## 回收为什么重写（计划 B2）

旧 `ingest()` 的核心缺陷在这一行：

    cid = str(rec.get("chunk_id"))
    got[cid] = rec            # ← 同一个 chunk_id 出现两次，后者静默覆盖前者

后果：一个 Part 里若同一个 `chunk_id` 出现两行**内容不同**的结果，收据只记
`duplicate_lines += 1`，入库的是最后那行——谁也不知道被丢掉的是哪一版，也无法
事后判断「该用哪一版」。更糟的是 `str(rec.get("chunk_id"))` 会把缺失的 chunk_id
变成字符串 `"None"`，与真实 id 混在同一个字典里。

新规则（逐条对应计划 B2）：
  同 id 同行       → 去重并记录（无害）
  同 id 不同内容   → **拒绝该 id，进隔离报告**（不猜）
  缺 chunk_id      → 隔离
  非 JSON 行       → 隔离
  未知 chunk_id    → 隔离
  缺失任何预期块   → Part 不完整
  硬错误           → 退出码非零

收据把「收到 / 合法 / 去重 / 缺失 / 冲突 / 隔离 / 正式入库」分开计数，不再用一个
`failed` 把七种不同的问题揉成一个数。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import atomic, cache, config, contracts, prompt

AGENT_MODEL = "agent:qoder-subagent"

INBOX_DIR = config.DATA_DIR / "agent_inbox"
OUTBOX_DIR = config.DATA_DIR / "agent_outbox"
QUARANTINE_DIR = config.DATA_DIR / "agent_quarantine"

# 计划书 3.6 的 T0 分层。原先按「只抽主线」设计（main10 / memory5 / act3 / rogue2），
# 现在 T1 已扩到主线+活动+密录+肉鸽（config.T1_CATEGORIES），冒烟分层同步补齐，
# 让每一族都有样本，避免「只在主线上验证过的提示词直接套到活动/肉鸽上」。
# 分层只用于冒烟，不影响正式分片的选取。
# 注意 `stratified` 用 startswith 前缀匹配，`obt/rogue` 会把 `obt/roguelike` 一并纳入池子
# （423 块），冒烟取样时属于可接受的宽口径；正式分片走 chunks.jsonl 的 shard 字段，不受影响。
T0_PLAN = [
    ("obt/main", 10),
    ("activities", 4),
    ("obt/memory", 3),
    ("obt/roguelike", 2),
    ("obt/rogue", 1),
]


def prepare_batches(label: str, chunks: list[dict], batch_size: int) -> dict:
    """把一片切成 batch_size 一块的子代理任务文件；已入库的块不再下发。"""
    pending = [c for c in chunks if cache.read_for_chunk(c, AGENT_MODEL) is None]
    (INBOX_DIR).mkdir(parents=True, exist_ok=True)
    (INBOX_DIR / "batch.system.md").write_text(prompt.SYSTEM_PROMPT + "\n", encoding="utf-8", newline="\n")
    plan = []
    for index in range(0, len(pending), batch_size):
        batch = pending[index : index + batch_size]
        tag = f"{label}.part{index // batch_size:02d}"
        path = INBOX_DIR / f"{tag}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for c in batch:
                fh.write(
                    json.dumps(
                        {
                            "chunk_id": c["chunk_id"],
                            "content_key": c["content_key"],
                            "user_prompt": prompt.build_user_prompt(c),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        # 路径写**相对仓库根**的形式（计划 A3 / DoD「部署方式可复现」）：
        # 这份 plan 会被 `run_l3_agent.py --plan` 落进 `data/arkrag/agent_plan.json`，
        # 而那个文件是入库的。写绝对机台路径等于把 `E:\图rag` 硬编码进 git，
        # 换机器后这些路径全部指向不存在的位置。
        # 实测这两个字段零读取方（`run_l3_agent.py:150` 只打印 tag 与 blocks），
        # 所以改成相对路径不会影响任何调用方。
        plan.append(
            {
                "tag": tag,
                "blocks": len(batch),
                "inbox": config.rel_to_root(path),
                "outbox": config.rel_to_root(OUTBOX_DIR / f"{tag}.jsonl"),
            }
        )
    return {"label": label, "cached_before": len(chunks) - len(pending), "pending": len(pending), "batches": plan}


def outbox_path(label: str) -> Path:
    return OUTBOX_DIR / f"{label}.jsonl"


def outbox_files(label: str) -> list[Path]:
    """支持多个子代理并行写分片文件：T0.jsonl 与 T0.part*.jsonl 都会被回收。"""
    OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
    files = [OUTBOX_DIR / f"{label}.jsonl"]
    files.extend(sorted(OUTBOX_DIR.glob(f"{label}.part*.jsonl")))
    return [f for f in files if f.exists()]


def _fingerprint_record(rec: dict) -> str:
    """一条回收记录的内容指纹，用来判断「同 id 两行是否真的相同」。

    只取影响产物的两个字段。其余字段（如模型自带的注释）不同不算冲突。
    """
    payload = json.dumps(
        {"entities": rec.get("entities"), "relations": rec.get("relations")},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class HarvestResult:
    """一次 Part 回收的完整账目。

    七个计数互不重叠，加起来等于「文件里出现的有效行数 + 预期块数」，
    这样任何一块的下落都能被追到唯一一格。
    """

    __slots__ = (
        "label",
        "lines",
        "accepted",
        "deduped",
        "conflicts",
        "quarantined",
        "missing",
        "written",
        "cache_hit",
        "by_id",
        "conflict_ids",
        "quarantine_items",
        "missing_ids",
        "raw_entities",
        "raw_relations",
    )

    def __init__(self, label: str) -> None:
        self.label = label
        self.lines = 0
        self.accepted = 0
        self.deduped = 0
        self.conflicts = 0
        self.quarantined = 0
        self.missing = 0
        self.written = 0
        self.cache_hit = 0
        self.by_id: dict[str, dict] = {}
        self.conflict_ids: list[str] = []
        self.quarantine_items: list[dict] = []
        self.missing_ids: list[str] = []
        self.raw_entities = 0
        self.raw_relations = 0

    @property
    def hard_errors(self) -> int:
        return self.conflicts + self.quarantined

    @property
    def complete(self) -> bool:
        """Part 是否完整：没有隔离、没有冲突、没有缺失块。"""
        return self.hard_errors == 0 and self.missing == 0

    def to_receipt(self) -> dict:
        return {
            "shard": self.label,
            "executor": AGENT_MODEL,
            "prompt_version": config.PROMPT_VERSION,
            "prompt_fingerprint": prompt.prompt_fingerprint(),
            "prompt_sha256": cache.prompt_sha256(),
            "normalizer_version": config.NORMALIZER_VERSION,
            "schema_version": config.SCHEMA_VERSION,
            # 七个口径分开报，禁止再用一个 failed 揉在一起
            "lines_read": self.lines,
            "accepted": self.accepted,
            "deduped": self.deduped,
            "conflicts": self.conflicts,
            "quarantined": self.quarantined,
            "missing": self.missing,
            "written": self.written,
            "cache_hit": self.cache_hit,
            "hard_errors": self.hard_errors,
            "complete": self.complete,
            "conflict_ids": self.conflict_ids[:60],
            "missing_ids": self.missing_ids[:60],
            "quarantine": self.quarantine_items[:60],
            "raw_entities": self.raw_entities,
            "raw_relations": self.raw_relations,
            # 同 `prepare_batches` 的理由：收据会落进入库的 `data/arkrag/ledger/<label>.json`，
            # 写绝对路径等于把机台路径提交进 git。此字段同样零读取方。
            "task_file": config.rel_to_root(INBOX_DIR / f"{self.label}.jsonl"),
        }


def _collect(label: str, result: HarvestResult, files: list[Path] | None = None) -> None:
    """读所有 outbox 文件，按 chunk_id 归并，冲突即拒。

    `files` 可显式指定要读哪些文件：按 part 逐个结算时，绝不能顺手把同 label
    下的其它 part 一起读进来，否则同一条记录会被结算两次。
    """
    for path in (files if files is not None else outbox_files(label)):
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                result.lines += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    result.quarantined += 1
                    result.quarantine_items.append(
                        {
                            "source": f"{path.name}:{lineno}",
                            "reason": "line_not_json",
                            "detail": str(exc)[:120],
                        }
                    )
                    continue

                if not isinstance(rec, dict):
                    result.quarantined += 1
                    result.quarantine_items.append(
                        {"source": f"{path.name}:{lineno}", "reason": "not_object", "detail": type(rec).__name__}
                    )
                    continue

                cid_raw = rec.get("chunk_id")
                if not isinstance(cid_raw, str) or not cid_raw.strip():
                    # 旧代码用 str(None) => "None" 混入字典
                    result.quarantined += 1
                    result.quarantine_items.append(
                        {
                            "source": f"{path.name}:{lineno}",
                            "reason": "missing_chunk_id",
                            "detail": repr(cid_raw),
                        }
                    )
                    continue
                cid = cid_raw.strip()

                prev = result.by_id.get(cid)
                if prev is None:
                    result.by_id[cid] = rec
                    result.accepted += 1
                    continue

                # 同一个 chunk_id 又出现一次：判断内容是否真的相同
                if _fingerprint_record(prev) == _fingerprint_record(rec):
                    result.deduped += 1
                    continue

                # 内容不同：绝不猜哪一版是对的
                result.conflicts += 1
                result.conflict_ids.append(cid)
                result.quarantine_items.append(
                    {
                        "source": f"{path.name}:{lineno}",
                        "reason": "conflicting_duplicate",
                        "detail": f"chunk_id {cid} 已有一份内容不同的结果",
                    }
                )


def ingest(label: str, chunks: list[dict], *, files: list[Path] | None = None) -> dict:
    """回收子代理产物 -> 严格契约校验 -> 写缓存。硬错误时退出码非零由调用方保证。

    `files` 用于「按 part 文件逐个结算」：只读指定的那一个文件，且该文件自己的
    label 就是结算单位。
    """
    result = HarvestResult(label)
    _collect(label, result, files)

    by_id = {c["chunk_id"]: c for c in chunks}
    conflict_set = set(result.conflict_ids)

    for chunk_id, rec in result.by_id.items():
        if chunk_id in conflict_set:
            # 冲突 id 直接隔离，两版都不入库
            continue
        chunk = by_id.get(chunk_id)
        if chunk is None:
            result.quarantined += 1
            result.quarantine_items.append(
                {"source": chunk_id, "reason": "unknown_chunk_id", "detail": "不在本 Part 的预期块内"}
            )
            continue

        existing = cache.lookup_for_chunk(chunk, AGENT_MODEL)
        if existing.raw is not None:
            result.cache_hit += 1
            continue

        # 契约校验：必须先在记录上补齐契约要求的身份字段
        task = {
            "schema_version": contracts.TASK_SCHEMA,
            "run_id": rec.get("run_id") or f"harvest:{label}",
            "chunk_id": chunk["chunk_id"],
            "content_key": chunk["content_key"],
            "scene_id": chunk.get("scene_id") or chunk.get("doc_id") or "",
            "doc_id": chunk.get("doc_id") or "",
            "source_fingerprint": chunk.get("source_fingerprint") or "",
            "prompt_sha256": cache.prompt_sha256(),
            "model": AGENT_MODEL,
            "generation": int(rec.get("generation") or 0),
            "text": chunk.get("text") or "",
        }

        candidate = {
            "schema_version": contracts.EXTRACTION_SCHEMA,
            "chunk_id": rec.get("chunk_id"),
            "content_key": rec.get("content_key") or chunk["content_key"],
            "scene_id": rec.get("scene_id") or task["scene_id"],
            "entities": rec.get("entities"),
            "relations": rec.get("relations") if "relations" in rec else [],
            "generation": rec.get("generation", 0),
            "model": AGENT_MODEL,
        }

        cleaned, errors = contracts.validate_extraction(
            candidate, task, task["text"], known_predicates=None
        )
        if cleaned is None:
            result.quarantined += 1
            summary = contracts.summarize_errors(errors)
            result.quarantine_items.append(
                {
                    "source": chunk_id,
                    "reason": "contract_violation",
                    "detail": json.dumps(summary["by_code"], ensure_ascii=False),
                    "errors": errors[:20],
                }
            )
            continue

        key = cache.key_for(
            content_key=chunk["content_key"],
            chunk_id=chunk["chunk_id"],
            scene_id=task["scene_id"],
            model=AGENT_MODEL,
            generation=task["generation"],
        )
        raw = {"entities": cleaned["entities"], "relations": cleaned["relations"]}
        cache.write_raw(
            key,
            raw,
            AGENT_MODEL,
            chunk["chunk_id"],
            chunk["content_key"],
            scene_id=task["scene_id"],
            generation=task["generation"],
        )
        result.written += 1
        result.raw_entities += len(raw["entities"])
        result.raw_relations += len(raw["relations"])

    # 缺失块：预期里有、既没收到也没缓存
    for c in chunks:
        if c["chunk_id"] in result.by_id:
            continue
        if cache.lookup_for_chunk(c, AGENT_MODEL).raw is not None:
            continue
        result.missing += 1
        result.missing_ids.append(c["chunk_id"])

    receipt = result.to_receipt()
    atomic.atomic_write_json(config.LEDGER_DIR / f"{label}.json", receipt)

    if result.quarantine_items:
        atomic.atomic_write_json(
            QUARANTINE_DIR / f"{label}.quarantine.json",
            {
                "shard": label,
                "count": len(result.quarantine_items),
                "items": result.quarantine_items,
            },
        )

    # 只有完整回收才归档 outbox：有冲突/隔离/缺失时保留原文件，方便复查哪一行坏的。
    # 显式传了 files 时只归档这几个文件——否则按 part 结算会把同 label 下其它
    # 还没收干净的 part 一起搬进 done/，那些块就永远没人再结算了。
    if result.complete and result.lines:
        done = OUTBOX_DIR / "done"
        done.mkdir(exist_ok=True)
        for path in (files if files is not None else outbox_files(label)):
            if path.exists():
                path.replace(done / path.name)
    return receipt


def ingest_strict(label: str, chunks: list[dict]) -> tuple[dict, int]:
    """回收并返回 (收据, 退出码)。硬错误一律非零——这是计划原则 5 的落点。"""
    receipt = ingest(label, chunks)
    if receipt["hard_errors"] or receipt["missing"]:
        return receipt, 1
    return receipt, 0
