"""L3 抽取执行入口。

拓扑（签字六）：分片 = 章节装箱后的块集合；子代理只驱动本脚本，不充当抽取模型；
全局并发预算 <= 8，由「并行子代数 x 本脚本并发」共同决定。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arkrag import config, extract, llm, prompt  # noqa: E402


def dry_run(chunks: list[dict], label: str) -> int:
    if not chunks:
        print("没有匹配的块")
        return 2
    sizes = [len(prompt.build_user_prompt(c)) for c in chunks]
    sample = prompt.build_user_prompt(chunks[0])
    print(f"[dry-run] {label}: {len(chunks)} 块")
    print(f"  user prompt 字数 min/avg/max = {min(sizes)}/{sum(sizes) // len(sizes)}/{max(sizes)}")
    print(f"  system prompt 字数 = {len(prompt.SYSTEM_PROMPT)}  指纹 = {prompt.prompt_fingerprint()}")
    print(f"  受控关系名 {len(prompt.CANONICAL_RELATIONS)} 条")
    print(f"  估算输入 token 合计（1.0 字/token 口径）≈ {sum(sizes) + len(sizes) * len(prompt.SYSTEM_PROMPT):,}")
    print("\n---- 首块 user prompt 前 900 字 ----\n" + sample[:900])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", help="chunks.jsonl 中的 shard 字段，如 S00")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk-id", action="append", default=[], help="指定块 id，可重复")
    ap.add_argument("--concurrency", type=int, default=config.LLM_CONCURRENCY)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # 熔断是全局的：任何一片留下的标记都会让后续片立刻停止，不再派发
    state = llm.circuit_state()
    if state and not args.dry_run:
        print(f"熔断标记在场，拒绝派发: {state}", file=sys.stderr)
        return 3

    chunks = extract.load_chunks(shard=args.shard, limit=args.limit or None, ids=args.chunk_id or None)
    label = args.shard or ("custom" if args.chunk_id else "all")

    if args.dry_run:
        return dry_run(chunks, label)
    if not chunks:
        print(f"没有匹配到块: shard={args.shard} ids={args.chunk_id}", file=sys.stderr)
        return 2

    receipt = extract.run_shard(chunks, args.concurrency, label)
    print(
        f"[{label}] 请求 {receipt['requested']} 命中 {receipt['cache_hit']} "
        f"成功 {receipt['extracted_ok']} 失败 {receipt['failed']} 熔断停发 {receipt['circuit_stopped']}"
    )
    print(f"  调用数 {receipt['total_calls']}  在途峰值 {receipt['peak_inflight']} (设置 {receipt['concurrency_setting']})")
    print(f"  token 输入 {receipt['tokens']['prompt']:,} 输出 {receipt['tokens']['completion']:,}")
    print(f"  原始实体 {receipt['raw_entities']:,} 原始关系 {receipt['raw_relations']:,}")
    if receipt["failed_chunks"]:
        print("  失败样例: " + "; ".join(receipt["failed_chunks"][:5]))
    return 0 if receipt["failed"] == 0 and receipt["circuit_stopped"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
