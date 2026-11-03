"""建向量索引：把一代的块正文编码成 1024 维向量并落盘。

用法：

    py -3 scripts\\build_vectors.py --generation g-20261031T035231Z-4eafae
    py -3 scripts\\build_vectors.py --generation ... --limit 200   # 试跑
    py -3 scripts\\build_vectors.py --generation ... --check-only  # 只校验不建

## 为什么从世代目录读 chunks 而不从 `chunks.jsonl`

`chunks.jsonl` 是 L1 的**最新**产物，而一个世代里的块未必是它（D 阶段新分块
13,798 块 vs 线上 13,781 块的分歧就是这么来的）。向量必须绑在**这一代实际在用的块**上，
所以块集合从世代目录解析，而世代目录里只有 `index/lexical.json` 的 `doc_id` 记着
「这一代用了哪些 chunk_id 及其顺序」。

但 `doc_id` 只有 id 没有正文，正文在 `chunks.jsonl`。所以流程是：

1. 从 `index/lexical.json` 取 `doc_id` → 这一代的块 id 列表（顺序即矩阵行序）
2. 从 `chunks.jsonl` 取正文 → 按 id 索引
3. 缺正文的块是硬错误（不许编码空串冒充）

## 幂等

sidecar 已存在且 `index_content_digest` 一致时直接退出（打印复用）。重建要显式 `--force`。
理由：全量编码 10–13 分钟，而三个 ark-rag-3 世代内容完全一致，重复建三遍纯属浪费。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arkrag import embedding, release, vector_store  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def generation_chunks(generation_id: str) -> tuple[list[dict], str, str]:
    """返回 (按 chunk_id 升序的块记录, index_content_digest, tokenizer_version)。

    行序必须与 `index.build()` 一致：它用 `sorted(chunk_records, key=chunk_id)` 分配 slot，
    而 `load_index` 把 `doc_id` 还原成 {slot: chunk_id}。所以这里也按 chunk_id 排序，
    这样矩阵第 i 行就对应 `doc_id[i]`，与词法索引的 slot 完全相同——后面 RRF 融合
    要按 chunk_id 对齐，两套行序一致能省掉一层映射，也少一处出错的机会。
    """
    gen = release.Generation(generation_id)
    manifest = gen.load()
    index_meta_path = gen.index_path
    if not index_meta_path.exists():
        raise SystemExit(f"[2] 世代 {generation_id} 没有索引：{index_meta_path}")

    payload = json.loads(index_meta_path.read_text(encoding="utf-8"))
    meta = payload.get("meta") or {}
    doc_id = payload.get("doc_id") or {}
    if not doc_id:
        raise SystemExit(f"[2] 索引里没有 doc_id，无法确定这一代的块集合")

    # JSON 落盘后 int key 变字符串
    by_slot = {int(k): str(v) for k, v in doc_id.items()}
    ordered_ids = [by_slot[s] for s in sorted(by_slot)]
    content_digest = meta.get("index_content_digest") or ""
    if not content_digest:
        raise SystemExit("[2] 索引 meta 缺 index_content_digest，向量无法绑定世代内容")

    from arkrag import config

    chunks_path = config.chunks_path()
    texts: dict[str, dict] = {}
    for rec in read_jsonl(chunks_path):
        cid = str(rec.get("chunk_id") or "")
        if cid:
            texts[cid] = rec

    missing = [cid for cid in ordered_ids if cid not in texts]
    if missing:
        raise SystemExit(
            f"[2] {len(missing)} 个块在 {chunks_path} 里找不到正文（前 3 个：{missing[:3]}）。"
            f"向量不许编码空串——那会得到一批彼此高度相似的无意义向量。"
        )
    return [texts[cid] for cid in ordered_ids], content_digest, meta.get("tokenizer_version") or ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="建向量索引（Ollama bge-m3）")
    ap.add_argument("--generation", help="世代 id；省略则取 current")
    ap.add_argument("--model", help=f"embedding 模型（默认 {embedding.DEFAULT_MODEL}）")
    ap.add_argument("--batch-size", type=int, default=embedding.DEFAULT_BATCH)
    ap.add_argument("--limit", type=int, help="只编码前 N 块（试跑用，产物不完整故不落盘）")
    ap.add_argument("--check-only", action="store_true", help="只校验已有向量，不建")
    ap.add_argument("--force", action="store_true", help="已存在也重建")
    args = ap.parse_args(argv)

    model = args.model or embedding.model_name()

    gid = args.generation
    if not gid:
        current = release.current_generation()
        if current is None:
            print("[2] 没有 current 世代，且未指定 --generation", file=sys.stderr)
            return 2
        gid = current.generation_id
    print(f"世代 {gid}")

    # 先证明 embedding 服务可用且模型身份没变，再花 10 分钟编码。
    # 反过来的话，服务在跑到一半挂掉，会留下半份向量。
    try:
        info = embedding.probe(model=model)
    except embedding.EmbeddingError as exc:
        print(f"[2] embedding 不可用：{exc}", file=sys.stderr)
        return 2
    print(
        f"模型 {info['model']}  dim={info['dim']}  digest={info['digest'][:16]}…  "
        f"{info['parameter_size']} {info['quantization']}  单条 {info['latency_ms']:.0f} ms"
    )
    if info["problems"]:
        for p in info["problems"]:
            print(f"[2] {p}", file=sys.stderr)
        return 2

    records, content_digest, tok = generation_chunks(gid)
    total = len(records) if args.limit is None else min(args.limit, len(records))
    print(f"块 {len(records)} 条（本次编码 {total} 条），index_content_digest={content_digest[:16]}…，分词器 {tok}")

    bin_path, meta_path = vector_store.vector_paths(
        vector_store._slug(model), content_digest
    )

    if meta_path.exists() and not args.force:
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            _, meta = vector_store.load_vectors(bin_path, meta_path)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            print(f"[!] 已有产物读不出来（{exc}），继续重建")
        else:
            if args.check_only:
                pass
            print(
                f"复用已有向量：{existing['count']} 条 {existing['dim']} 维 "
                f"（{bin_path.name}，{existing['matrix_bytes'] / 1e6:.1f} MB）"
            )
            report = _check(records, meta, model, info)
            _print_report(report)
            return 0 if report["ok"] else 1

    texts = [vector_store.chunk_text_for_embedding(rec) for rec in records[:total]]
    chunk_ids = [str(rec["chunk_id"]) for rec in records[:total]]

    if args.check_only:
        if not meta_path.exists():
            print(f"[2] --check-only 但产物不存在：{meta_path}", file=sys.stderr)
            return 2
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        report = _check(records, meta, model, info)
        _print_report(report)
        return 0 if report["ok"] else 1

    print(f"开始编码：{total} 条，batch={args.batch_size}，预计 {total * 0.043 / 60:.1f} 分钟")
    t0 = time.perf_counter()
    last = [0.0]

    def progress(done: int, total: int = total) -> None:
        now = time.perf_counter()
        if now - last[0] < 5.0 and done < total:
            return
        last[0] = now
        elapsed = now - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(f"  {done}/{total}  {rate:.1f} 条/s  ETA {eta / 60:.1f} 分钟", flush=True)

    vectors = embedding.embed_all(
        texts, model=model, batch_size=args.batch_size, progress=progress
    )
    elapsed = time.perf_counter() - t0
    print(f"编码完成：{len(vectors)} 条，{elapsed:.1f} s（{elapsed / len(vectors) * 1000:.1f} ms/条）")

    if args.limit is not None and total < len(records):
        print("[!] --limit 试跑：产物不完整，不落盘（否则会被当成「已建好」复用）")
        return 0

    bin_path, meta_path, meta = vector_store.save_vectors(
        vectors,
        chunk_ids,
        texts,
        model=info["model"],
        model_digest=info["digest"],
        dim=info["dim"],
        generation_id=gid,
        index_content_digest=content_digest,
        normalized=info["normalized"],
    )
    print(f"落盘 {bin_path}（{meta['matrix_bytes'] / 1e6:.1f} MB）+ {meta_path.name}")

    report = _check(records, meta, model, info)
    _print_report(report)
    return 0 if report["ok"] else 1


def _check(records: list[dict], meta: dict, model: str, info: dict) -> dict:
    """建完立刻自检。不信任「我刚写的」——那正是最该验的东西。"""
    problems = []
    if meta.get("model") != info["model"]:
        problems.append(f"模型不符：侧车 {meta.get('model')}，实测 {info['model']}")
    if meta.get("model_digest") != info["digest"]:
        problems.append(
            f"模型权重 digest 不符：侧车 {str(meta.get('model_digest'))[:16]}…，"
            f"实测 {info['digest'][:16]}…（tag 相同但权重被重打包过）"
        )
    if int(meta.get("dim") or 0) != info["dim"]:
        problems.append(f"维度不符：侧车 {meta.get('dim')}，实测 {info['dim']}")
    if meta.get("count") != len(records):
        problems.append(f"条数不符：侧车 {meta.get('count')}，这一代 {len(records)} 块")

    ids = [str(r["chunk_id"]) for r in records]
    texts_by_id = {str(r["chunk_id"]): vector_store.chunk_text_for_embedding(r) for r in records}
    detail = vector_store.verify_vectors(meta, ids, texts_by_id)
    if not detail["ok"]:
        problems.extend(detail["problems"])
    return {"ok": not problems, "problems": problems, "detail": detail, "meta": meta}


def _print_report(report: dict) -> None:
    meta = report["meta"]
    detail = report["detail"]
    print("-" * 70)
    print(
        f"自检：{meta.get('count')} 条 × {meta.get('dim')} 维  "
        f"model={meta.get('model')}  normalized={meta.get('normalized')}"
    )
    print(
        f"  一一对应：缺 {detail['missing']} / 残留 {detail['orphan']} / "
        f"哈希不符 {detail['mismatched']}"
    )
    if report["ok"]:
        print("  结论：PASS")
    else:
        print("  结论：FAIL")
        for p in report["problems"]:
            print(f"    - {p}")


if __name__ == "__main__":
    raise SystemExit(main())
