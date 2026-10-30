"""回收 agent_outbox 里所有已完成的 part 文件：严格契约校验 -> 写缓存 -> 归档。

## 与旧版的区别（计划 B2 / B3）

旧版本自己另写了一套校验：用 `extract.validate_shape`（只查四个字段是否为非空
字符串），并且

    raw = {"entities": rec.get("entities") or [], "relations": rec.get("relations") or []}

把 `entities=""`、`entities=False`、`entities=null` 全部洗成「没抽到实体」后写进
缓存——坏数据一旦入库就无法在事后区分。同一份数据在 `agent_tasks.ingest()` 里
走的是严格契约，两个入口判据不一致，正是计划 B3 要求消灭的「同一件事两种口径」。

现在本脚本**不再自带校验**，只负责：

1. 找出可以结算的 part 文件（避开子代理正在追加写的文件）；
2. 按 inbox 任务文件确定这个 part 的**预期块集合**；
3. 交给 `agent_tasks.ingest()` 走统一严格契约；
4. 汇总收据，有任何硬错误或缺块就让退出码非零。

计划原则 5：门禁必须能失败。旧版本无论多少坏行都 `return 0`，等于用一个永远
通过的脚本冒充验收。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import agent_tasks, cache, config, extract  # noqa: E402
from arkrag import prompt as prompt_mod  # noqa: E402

INBOX = config.DATA_DIR / "agent_inbox"
OUTBOX = config.DATA_DIR / "agent_outbox"
DONE = OUTBOX / "done"
LOG = config.LEDGER_DIR / "_ingest_log.jsonl"

# 文件最后修改距今不足这个秒数就跳过本轮：子代理追加写期间不回收。
SETTLE_SECONDS = 180


def _expected_ids(task_path: Path) -> list[str] | None:
    """从 inbox 任务文件读出这个 part 的预期 chunk_id 列表。"""
    ids: list[str] = []
    try:
        with task_path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # 任务文件自己坏了：整个 part 不能结算，否则「预期集合」不可信
                    return None
                cid = rec.get("chunk_id") if isinstance(rec, dict) else None
                if not isinstance(cid, str) or not cid.strip():
                    return None
                ids.append(cid.strip())
    except OSError:
        return None
    return ids


def main() -> int:
    DONE.mkdir(parents=True, exist_ok=True)
    config.LEDGER_DIR.mkdir(parents=True, exist_ok=True)

    chunks = {c["chunk_id"]: c for c in extract.load_chunks()}

    totals = {
        "parts_settled": 0,
        "parts_deferred": 0,
        "parts_bad": 0,
        "parts_complete": 0,
        "parts_incomplete": 0,
        "written": 0,
        "cache_hit": 0,
        "deduped": 0,
        "conflicts": 0,
        "quarantined": 0,
        "missing": 0,
        "entities": 0,
        "relations": 0,
        "unknown_chunk_ids": 0,
    }
    bad_report: list[str] = []
    hard_failure = False

    now = time.time()
    for path in sorted(OUTBOX.glob("*.part*.jsonl")) + sorted(OUTBOX.glob("T0.jsonl")):
        stat = path.stat()
        # 子代理是边抽边追加写的，刚动过的文件回收会把「写到一半」误判成缺行。
        # 已写进缓存的行是幂等的，下一轮补完再收即可，这里只做安静跳过。
        if now - stat.st_mtime < SETTLE_SECONDS or stat.st_size == 0:
            totals["parts_deferred"] += 1
            continue

        task_path = INBOX / path.name
        if not task_path.exists():
            bad_report.append(f"{path.name}: 没有对应的 inbox 任务文件，无法确定预期块")
            totals["parts_bad"] += 1
            hard_failure = True
            continue

        expected_ids = _expected_ids(task_path)
        if expected_ids is None:
            bad_report.append(f"{path.name}: inbox 任务文件自身不可解析，拒绝结算")
            totals["parts_bad"] += 1
            hard_failure = True
            continue

        expected: list[dict] = []
        for cid in expected_ids:
            c = chunks.get(cid)
            if c is None:
                totals["unknown_chunk_ids"] += 1
                bad_report.append(f"{path.name}:{cid}: 不在当前 chunks.jsonl 里")
                hard_failure = True
                continue
            expected.append(c)

        label = path.stem
        # 只结算这一个文件——同 label 下可能还有其它 part 未写完或被拒收，
        # 顺手一起读会让同一条记录被结算两次。
        receipt = agent_tasks.ingest(label, expected, files=[path])
        totals["parts_settled"] += 1
        totals["written"] += receipt["written"]
        totals["cache_hit"] += receipt["cache_hit"]
        totals["deduped"] += receipt["deduped"]
        totals["conflicts"] += receipt["conflicts"]
        totals["quarantined"] += receipt["quarantined"]
        totals["missing"] += receipt["missing"]
        totals["entities"] += receipt["raw_entities"]
        totals["relations"] += receipt["raw_relations"]

        if receipt["complete"]:
            totals["parts_complete"] += 1
        else:
            totals["parts_incomplete"] += 1
            bad_report.append(
                f"{path.name}: 不完整 — 冲突 {receipt['conflicts']} 隔离 {receipt['quarantined']} "
                f"缺块 {receipt['missing']}"
            )
            for item in receipt["quarantine"][:10]:
                bad_report.append(f"  {item.get('source')} {item.get('reason')} {item.get('detail','')[:80]}")

        if receipt["hard_errors"] or receipt["missing"]:
            hard_failure = True

        try:
            with LOG.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(
                    json.dumps(
                        {
                            "file": path.name,
                            "expected": len(expected),
                            "written": receipt["written"],
                            "cache_hit": receipt["cache_hit"],
                            "deduped": receipt["deduped"],
                            "conflicts": receipt["conflicts"],
                            "quarantined": receipt["quarantined"],
                            "missing": receipt["missing"],
                            "complete": receipt["complete"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except OSError as exc:  # 日志写不进去不该让整轮结算崩掉，但要说出来
            print(f"WARN 追加结账日志失败: {exc}", file=sys.stderr)

        # outbox 的归档由 agent_tasks.ingest 在「完整」时完成，这里不再重复搬动
        if receipt["complete"] and (DONE / path.name).exists():
            totals["archived"] = totals.get("archived", 0) + 1

    summary = {
        "executor": agent_tasks.AGENT_MODEL,
        "schema_version": config.SCHEMA_VERSION,
        "prompt_version": config.PROMPT_VERSION,
        "prompt_sha256": cache.prompt_sha256(),
        "prompt_fingerprint": prompt_mod.prompt_fingerprint(),
        "normalizer_version": config.NORMALIZER_VERSION,
        **totals,
        "bad_examples": bad_report[:60],
    }
    (config.LEDGER_DIR / "_ingest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(totals, ensure_ascii=False))
    for line in bad_report[:20]:
        print("  BAD " + line)
    if hard_failure:
        print("回收失败：存在冲突/隔离/缺块/未知块，退出码 1", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
