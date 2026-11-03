"""L3 抽取驱动：契约校验 + 缓存身份 + 有界队列执行。归一化不在这里做（决策三）。

## 相对旧版改了什么（计划 B1 / B4 / C5）

* **结构校验换成契约**：旧 `validate_shape` 只查「四字段是不是非空字符串」，
  置信度越界、证据根本不在正文里、`entities=""` 都能过。现在走
  `contracts.validate_extraction`，每个拒收原因都有机器可读错误码与字段路径。
* **缓存身份换成新键**：旧 `cache.cache_key(content_key, model, prompt_version)`
  不含 chunk_id / 场景 / 契约版本，overlap 的两块正文相同就会互相命中。现在走
  `cache.lookup_for_chunk`，新键缺失时只复用**身份核对通过**的 legacy 记录。
* **提交方式换成分批**：旧代码
  `futures = {pool.submit(...) for c in chunks}` 一次性把 13,781 个任务全塞进
  队列——内存与在途峰值都不受控，而且熔断后无法停止投喂。现在按 `window` 分批，
  每批完成再补，熔断后**停止补充新任务**，并记录真实并发峰值。
"""
from __future__ import annotations

import json
import threading
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from . import atomic, cache, config, contracts, llm, prompt


def validate_shape(raw: dict) -> tuple[bool, str]:
    """旧的结构检查，保留给不认识契约的调用方做快速粗筛。

    正式入库路径**不再**用它——它不查置信度、不查证据可定位、不查重复 ID，
    只适合在真正下发前拦一下明显崩坏的返回。
    """
    if not isinstance(raw, dict):
        return False, "not_object"
    entities = raw.get("entities")
    relations = raw.get("relations")
    if not isinstance(entities, list) or not isinstance(relations, list):
        return False, "missing_arrays"
    for i, ent in enumerate(entities):
        if not isinstance(ent, dict):
            return False, f"entity[{i}] not object"
        if not isinstance(ent.get("name"), str) or not ent["name"].strip():
            return False, f"entity[{i}] empty name"
    for i, rel in enumerate(relations):
        if not isinstance(rel, dict):
            return False, f"relation[{i}] not object"
        for field in ("source", "relation", "target", "evidence"):
            if not isinstance(rel.get(field), str) or not rel[field].strip():
                return False, f"relation[{i}] field {field} empty"
    return True, ""


def _build_task(chunk: dict, client_model: str, *, generation: int = 0) -> dict:
    return {
        "schema_version": contracts.TASK_SCHEMA,
        "run_id": config.LLM_MODEL or "local",
        "chunk_id": chunk["chunk_id"],
        "content_key": chunk["content_key"],
        "scene_id": chunk.get("scene_id", ""),
        "doc_id": chunk.get("doc_id", ""),
        "source_fingerprint": chunk.get("corpus_fp", ""),
        "prompt_sha256": cache.prompt_sha256(),
        "model": client_model,
        "generation": generation,
        "text": chunk.get("text", ""),
        "title": chunk.get("title", ""),
        "chapter": chunk.get("chapter", ""),
        "category": chunk.get("category", ""),
    }


def extract_chunk(
    client: llm.LLMClient,
    chunk: dict,
    *,
    generation: int = 0,
    known_predicates: set[str] | None = None,
) -> dict:
    """抽一个 Chunk。返回结果含 `cache` 状态与（可选的）已校验 `raw`。"""
    hit = cache.lookup_for_chunk(chunk, client.model, generation=generation)
    if hit:
        return {
            "chunk_id": chunk["chunk_id"],
            "cache": "hit",
            "key": hit.key,
            "layer": hit.layer,
            "raw": hit.raw,
            "meta": {},
        }

    if llm.circuit_state() is not None:
        return {"chunk_id": chunk["chunk_id"], "cache": "circuit", "key": "", "raw": None, "meta": {}}

    user = prompt.build_user_prompt(chunk)
    raw, meta = llm.extract_with_retry(client, prompt.SYSTEM_PROMPT, user)
    if raw is None:
        return {"chunk_id": chunk["chunk_id"], "cache": "miss_fail", "key": "", "raw": None, "meta": meta}

    # 契约校验：把裸返回包成正式契约记录再验，身份字段取自任务本身
    task = _build_task(chunk, client.model, generation=generation)
    candidate = {
        "schema_version": contracts.EXTRACTION_SCHEMA,
        "chunk_id": chunk["chunk_id"],
        "content_key": chunk["content_key"],
        "scene_id": chunk.get("scene_id", ""),
        "generation": generation,
        "model": client.model,
        "entities": raw.get("entities"),
        "relations": raw.get("relations", []),
    }
    cleaned, errors = contracts.validate_extraction(
        candidate, task, chunk.get("text", ""), known_predicates=known_predicates
    )
    if cleaned is None:
        summary = contracts.summarize_errors(errors)
        meta = dict(meta)
        meta["contract_errors"] = summary["by_code"]
        meta["contract_samples"] = summary["samples"]
        meta["errors"] = list(meta.get("errors", [])) + [
            f"contract:{k}" for k in summary["by_code"]
        ]
        return {"chunk_id": chunk["chunk_id"], "cache": "miss_fail", "key": "", "raw": None, "meta": meta}

    key = cache.key_for(
        content_key=chunk["content_key"],
        chunk_id=chunk["chunk_id"],
        scene_id=chunk.get("scene_id", ""),
        model=client.model,
        generation=generation,
    )
    # 只存模型原始输出的可复用部分；证据偏移与校验结论随记录一起留下，
    # 但不写进缓存——缓存是「原始抽取」层，重新校验要能复现。
    cache.write_raw(
        key,
        {"entities": cleaned["entities"], "relations": cleaned["relations"]},
        client.model,
        chunk["chunk_id"],
        chunk["content_key"],
        scene_id=chunk.get("scene_id", ""),
        generation=generation,
    )
    return {"chunk_id": chunk["chunk_id"], "cache": "miss_ok", "key": key, "raw": cleaned, "meta": meta}


def load_chunks(shard: str | None = None, limit: int | None = None, ids: list[str] | None = None) -> list[dict]:
    out = []
    wanted = set(ids) if ids else None
    with config.chunks_path().open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            c = json.loads(line)
            if wanted is not None:
                if c["chunk_id"] in wanted:
                    out.append(c)
            elif shard is not None:
                if c.get("shard") == shard:
                    out.append(c)
            else:
                out.append(c)
    if limit:
        out = out[:limit]
    return out


def run_shard(
    chunks: list[dict],
    concurrency: int,
    label: str,
    *,
    window: int | None = None,
    generation: int = 0,
    known_predicates: set[str] | None = None,
    client: llm.LLMClient | None = None,
) -> dict:
    """跑一个分片。

    `window` 是**待处理队列上限**（不是线程数）：在任何时刻最多有这么多个任务
    已被提交但未完成。旧实现一次性提交全部 Chunk，等于没有队列；熔断发生后也只能
    眼睁睁看着剩余任务继续打向已经不可用的网关。

    熔断一旦出现，立即停止补充新任务，把剩余 Chunk 记为 `skipped_circuit`——
    它们是「没跑」而不是「跑失败」，两者在收据里必须分开。

    `client` 可注入，供测试用假客户端驱动；不传则按配置构造真实客户端。
    """
    window = window or max(concurrency * 2, 8)
    window = max(window, concurrency)

    gauge: llm.InflightGauge = getattr(client, "gauge", None) or llm.InflightGauge()
    if client is None:
        client = llm.LLMClient(gauge=gauge)
    lock = threading.Lock()
    tally = {"hit": 0, "miss_ok": 0, "miss_fail": 0, "circuit": 0}
    failures: list[dict] = []
    tokens = {"prompt": 0, "completion": 0}
    entities_raw = 0
    relations_raw = 0
    contract_rejects: dict[str, int] = {}
    layers = {"current": 0, "legacy": 0}

    pending = deque(chunks)
    inflight: dict = {}
    stopped = False

    def _absorb(res: dict) -> None:
        nonlocal entities_raw, relations_raw, stopped
        tally[res["cache"]] = tally.get(res["cache"], 0) + 1
        meta = res.get("meta") or {}
        tokens["prompt"] += meta.get("prompt_tokens", 0)
        tokens["completion"] += meta.get("completion_tokens", 0)
        layer = res.get("layer")
        if layer in layers:
            layers[layer] += 1
        raw = res.get("raw")
        if raw:
            entities_raw += len(raw.get("entities") or [])
            relations_raw += len(raw.get("relations") or [])
        for code, n in (meta.get("contract_errors") or {}).items():
            contract_rejects[code] = contract_rejects.get(code, 0) + n
        if res["cache"] in ("miss_fail", "circuit"):
            failures.append(
                {
                    "chunk_id": res["chunk_id"],
                    "reason": res["cache"],
                    "errors": (meta.get("errors") or [])[:10],
                }
            )
        if res["cache"] == "circuit":
            stopped = True

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        while pending or inflight:
            # 只在还没熔断时补充新任务
            while not stopped and pending and len(inflight) < window:
                # `deque.popleft()` 是 O(1)；原先是 `list.pop(0)`，每次搬动整个
                # 剩余队列。13,781 块时累计约 9,500 万次元素搬移，且全部发生在
                # 持锁的判断之间——不是主要瓶颈，但是没有理由留着。
                c = pending.popleft()
                fut = pool.submit(
                    extract_chunk,
                    client,
                    c,
                    generation=generation,
                    known_predicates=known_predicates,
                )
                inflight[fut] = c
            if not inflight:
                break
            done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for fut in done:
                inflight.pop(fut, None)
                with lock:
                    _absorb(fut.result())

    skipped = [c["chunk_id"] for c in pending]

    receipt = {
        "shard": label,
        "model": client.model,
        "prompt_version": config.PROMPT_VERSION,
        "prompt_sha256": cache.prompt_sha256(),
        "normalizer_version": config.NORMALIZER_VERSION,
        "schema_version": contracts.EXTRACTION_SCHEMA,
        "generation": generation,
        "requested": len(chunks),
        "cache_hit": tally["hit"],
        "cache_hit_layers": layers,
        "extracted_ok": tally["miss_ok"],
        "failed": tally["miss_fail"],
        "circuit_stopped": tally["circuit"],
        "skipped_circuit": len(skipped),
        "skipped_chunk_ids": skipped[:200],
        "failed_chunks": failures[:50],
        "contract_rejects": contract_rejects,
        "total_calls": gauge.total_calls,
        "peak_inflight": gauge.peak,
        "concurrency_setting": concurrency,
        "window_setting": window,
        "tokens": tokens,
        "raw_entities": entities_raw,
        "raw_relations": relations_raw,
        "circuit_state": llm.circuit_summary(),
    }
    config.LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    atomic.atomic_write_json(config.LEDGER_DIR / f"{label}.json", receipt)
    return receipt
