"""子代理抽取的下发/回收入口。

  py -3 scripts/run_l3_agent.py --plan   --label S00 --shard S00 --batch-size 20
  py -3 scripts/run_l3_agent.py --ingest --label S00
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arkrag import agent_tasks, cache, config, extract  # noqa: E402

PLAN_PATH = config.DATA_DIR / "agent_plan.json"


def select_chunks(args: argparse.Namespace) -> list[dict]:
    if args.stratify:
        return stratified(dict(agent_tasks.T0_PLAN))
    if args.shard:
        return [c for c in extract.load_chunks(shard=args.shard) if config.in_t1(c["category"])]
    # 无 --shard 时按 T1 范围兜底：扩范围后只剩 obt/guide 5 块在范围外，
    # 但它们不含人物关系，不该被顺手派发出去。
    return [c for c in extract.load_chunks() if config.in_t1(c["category"])]


def stratified(per_category: dict[str, int]) -> list[dict]:
    """按语料族分层取样，取样规则确定（排序后等距），不靠随机数。

    前缀匹配的副作用：`obt/rogue` 也会把 `obt/roguelike` 选进池子，两族都列时会重复抽到同一块。
    这里按 chunk_id 去重，保证同一块只下发一次。
    """
    all_chunks = extract.load_chunks()
    picked: list[dict] = []
    seen: set[str] = set()
    for category, want in per_category.items():
        pool = [c for c in all_chunks if c["category"].startswith(category)]
        if category in config.T1_CATEGORIES:
            exact = [c for c in pool if c["category"] == category]
            if exact:
                pool = exact
        if not pool:
            continue
        step = max(1, len(pool) // want)
        for i in range(want):
            if i * step >= len(pool):
                break
            chunk = pool[i * step]
            if chunk["chunk_id"] in seen:
                continue
            seen.add(chunk["chunk_id"])
            picked.append(chunk)
    return picked


def load_plan() -> dict:
    if not PLAN_PATH.exists():
        return {}
    plans = json.loads(PLAN_PATH.read_text("utf-8"))
    return plans


def save_plan(plans: dict) -> None:
    PLAN_PATH.write_text(
        json.dumps(plans, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="切批次并下发任务文件")
    ap.add_argument("--ingest", action="store_true", help="回收子代理产物写缓存")
    ap.add_argument("--label", default="")
    ap.add_argument("--shard")
    ap.add_argument("--stratify", action="store_true")
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--status", action="store_true", help="按片汇报已入库/缺口（分 current/legacy/adopted）")
    ap.add_argument(
        "--adopt",
        action="store_true",
        help="对 miss 块尝试带来源复用（cache.adopt_legacy）：四项身份全等且过严格契约才搬",
    )
    args = ap.parse_args()

    if args.status or (not args.label and not args.adopt):
        all_chunks = extract.load_chunks()
        by_shard: dict[str, list[dict]] = {}
        for c in all_chunks:
            by_shard.setdefault(c.get("shard") or "(范围外)", []).append(c)
        # 三层分开报：current=本轮新键抽取，legacy=只读复用旧记录，
        # adopted=从旧记录按四项全等搬进新键。合成一个「已入库」会让
        # 「这一代到底真抽了多少块」永久不可见（计划 D5）。
        totals = {"current": 0, "legacy": 0, "adopted": 0, "miss": 0, "invalid": 0}
        adopted_ids: list[str] = []
        invalid_ids: list[str] = []
        for shard in sorted(by_shard):
            pool = by_shard[shard]
            # 状态查询必须带 chunk_id 核对（计划 B3）：「缓存文件存在」不等于
            # 「这一块抽过」。旧口径按 content_key 拼旧键，overlap 的同正文块会
            # 互相命中，剩余量被系统性低估，调度看着像快完成了其实还差很多。
            counts = {"current": 0, "legacy": 0, "adopted": 0, "miss": 0, "invalid": 0}
            for c in pool:
                lk = cache.lookup_for_chunk(c, agent_tasks.AGENT_MODEL)
                lay = lk.layer if lk.layer in counts else "miss"
                counts[lay] += 1
                totals[lay] += 1
                if lay == "adopted":
                    adopted_ids.append(c["chunk_id"])
                elif lay == "invalid":
                    invalid_ids.append(c["chunk_id"])
            done = counts["current"] + counts["legacy"] + counts["adopted"]
            print(
                f"  {shard} 已入库 {done:4d}/{len(pool):4d}  缺 {len(pool) - done:3d}"
                f"   〔current {counts['current']:4d}  legacy {counts['legacy']:4d}"
                f"  adopted {counts['adopted']:3d}  隔离 {counts['invalid']:3d}〕"
            )
        out_of_t1 = sum(1 for c in all_chunks if not config.in_t1(c["category"]))
        done_total = totals["current"] + totals["legacy"] + totals["adopted"]
        print(
            f"合计 {done_total}/{len(all_chunks)} 块已入库"
            f"（T1 范围 {'、'.join(config.T1_CATEGORIES)}；范围外 {out_of_t1} 块）"
        )
        print(
            f"来源分层 current {totals['current']}  legacy {totals['legacy']}"
            f"  adopted {totals['adopted']}  miss {totals['miss']}  invalid {totals['invalid']}"
        )
        if adopted_ids:
            print(f"  adopted 样例: " + "; ".join(adopted_ids[:5]))
        if invalid_ids:
            print(f"  invalid 样例: " + "; ".join(invalid_ids[:5]))
        return 0

    chunks = select_chunks(args)
    if args.limit:
        chunks = chunks[: args.limit]

    if args.plan:
        plan = agent_tasks.prepare_batches(args.label, chunks, args.batch_size)
        plans = load_plan()
        plans[args.label] = plan
        save_plan(plans)
        print(
            f"[{args.label}] 块数 {len(chunks)} 已入库 {plan['cached_before']} "
            f"待抽 {plan['pending']} -> {len(plan['batches'])} 批 x<={args.batch_size}"
        )
        for b in plan["batches"][:3]:
            print(f"  {b['tag']} blocks={b['blocks']}")
        return 0

    if args.ingest:
        # 统一走严格回收入口（计划 B3）：与 scripts/ingest_all.py 同一套判据，
        # 不再各写一份。硬错误/缺块一律退出码非零——原则 5。
        receipt, code = agent_tasks.ingest_strict(args.label, chunks)
        print(
            f"[{args.label}] 读到 {receipt['lines_read']} 行  合法 {receipt['accepted']}  "
            f"去重 {receipt['deduped']}  冲突 {receipt['conflicts']}  隔离 {receipt['quarantined']}  "
            f"缺块 {receipt['missing']}"
        )
        print(f"  命中 {receipt['cache_hit']}  新入库 {receipt['written']}  硬错误 {receipt['hard_errors']}")
        print(f"  原始实体 {receipt['raw_entities']} 原始关系 {receipt['raw_relations']}")
        for item in receipt["quarantine"][:8]:
            print(f"  隔离 {item.get('source')} {item.get('reason')} {str(item.get('detail',''))[:100]}")
        if receipt["missing_ids"]:
            print("  缺块样例: " + "; ".join(receipt["missing_ids"][:6]))
        return code

    if args.adopt:
        # 带来源的复用（计划 D5）：只对 miss 块尝试，四项身份全等 + 过严格契约才搬。
        # 搬出来的记录写新键并标注 adopted_from/adopt_reason/adopted_at，
        # 因此下游 lookup 会报 layer="adopted" 而不是混进 current。
        targets = chunks
        adopted, skipped, refused = [], [], []
        for chunk in targets:
            lk = cache.lookup_for_chunk(chunk, agent_tasks.AGENT_MODEL)
            if lk.raw is not None:
                skipped.append((chunk["chunk_id"], lk.layer))
                continue
            src = cache.load_legacy_record(chunk["content_key"], agent_tasks.AGENT_MODEL)
            got = cache.adopt_legacy(src, chunk, agent_tasks.AGENT_MODEL)
            if got.layer == "adopted":
                adopted.append((chunk["chunk_id"], got.adopted_from))
            else:
                refused.append((chunk["chunk_id"], got.layer, got.reason))
        print(
            f"复用尝试 {len(targets)} 块：已命中跳过 {len(skipped)}  "
            f"成功搬用 {len(adopted)}  拒绝 {len(refused)}"
        )
        for cid, src_key in adopted[:10]:
            print(f"  adopted {cid}  <- {src_key}")
        for cid, lay, why in refused[:12]:
            print(f"  拒绝 {cid}  {lay}  {why[:110]}")
        return 0 if not refused else 1

    print("需要 --plan 或 --ingest")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
