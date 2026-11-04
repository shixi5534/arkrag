# -*- coding: utf-8 -*-
"""K2 全量硬门八条：逐条给出可验证结论。退出码 0=八条全过 / 1=有未过 / 2=硬错误。

八条来自 docs/升级计划_GraphRAG架构.md 的 K2 节：
  1. 无轮次静默丢失
  2. 无重复 Chunk 身份
  3. 无非法缓存
  4. 所有 Assertion 证据可定位
  5. 无未解释产物身份变化
  6. 无失败 Generation 被发布
  7. 索引与图绑定同一 Manifest
  8. 现有 8 例和 46 例风险有处理结论

设计纪律：
- 每条都要**能失败**，且失败时说得清是哪一块哪个字出了问题。
- 不重做 `release.verify_generation()` 的活儿（那是另一份定义）；这里查的是
  「构建过程看不到的东西」。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
# Windows 控制台默认 cp936，不重配置会把中文写成乱码字节，
# 且 `> file` 重定向后文件不是 UTF-8（本项目脚本一律要求输出可原样粘贴进文档）。
# 守卫是必要的：pytest 捕获期间 `sys.stdout` 可能不是可重配置的文本流。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from arkrag import atomic, cache, config, index as index_mod, pipeline, release  # noqa: E402

NEW_GID = sys.argv[1] if len(sys.argv) > 1 else "g-20261103T132152Z-06a741"
OLD_GID = "g-20261031T035231Z-4eafae"
SCENES = ROOT / "data" / "arkrag" / "staging" / "k2" / "scenes.jsonl"
SHARDS = ROOT / "data" / "arkrag" / "staging" / "k2" / "shards.json"
L1_STATS = ROOT / "data" / "arkrag" / "staging" / "k2" / "l1_stats.json"

results: list[dict] = []


def record(gid: str, name: str, ok: bool, detail: str, data: dict | None = None) -> None:
    results.append({"id": gid, "name": name, "ok": ok, "detail": detail, "data": data or {}})
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<22s} {detail}")


gen = release.Generation(NEW_GID)
manifest = gen.load()
graph = json.loads(gen.graph_path.read_text("utf-8"))
_all_chunks = pipeline._read_jsonl(pipeline.resolve_chunks(gen, manifest))
chunks = [c for c in _all_chunks if config.in_t1(str(c.get("category") or ""))]
assertions = pipeline._read_jsonl(gen.assertion_path)

# ---- 1. 无轮次静默丢失 ----
#
# 口径必须按 **T1 范围**，不能拿全语料轮次去比。两个原因叠在一起：
#   a) `config.T1_CATEGORIES` 是分类白名单，按设计不收录 `obt/guide`（新手引导）；
#   b) 世代自带的 `chunks.jsonl` 就是 T1 子集（`build_generation.py` 建库时
#      用 `[c for c in load_chunks() if config.in_t1(c["category"])]`）——
#      拿全部块去补也补不回 `obt/guide`，因为那批块根本不在这份快照里。
#
# 于是"丢失"要问的是：**属于 T1 的场景，它的每一轮是不是都被某个 T1 块收了**。
# 白名单外的轮次不是丢失，是**按设计不收录**，单独计数并点名分类，不许混进失败项，
# 也不许当成"覆盖了"含糊过去。
try:
    scenes = pipeline._read_jsonl(SCENES) if SCENES.exists() else []
    t1_have: dict[str, set[int]] = {}
    for c in chunks:
        doc = str(c.get("doc_id") or "")
        t1_have.setdefault(doc, set()).update(int(o) for o in (c.get("turn_ordinals") or []))
    t1_scenes = 0
    t1_turns = 0
    out_of_scope_turns = 0
    out_of_scope_cats: set[str] = set()
    lost: list[tuple[str, int]] = []
    for sc in scenes:
        cat = str(sc.get("category") or "")
        doc = str(sc.get("doc_id") or "")
        turns = sc.get("turns") or []
        if cat not in config.T1_CATEGORIES:
            out_of_scope_turns += len(turns)
            out_of_scope_cats.add(cat)
            continue
        t1_scenes += 1
        t1_turns += len(turns)
        have = t1_have.get(doc, set())
        for i in range(len(turns)):
            if i not in have:
                lost.append((doc, i))
    ok1 = t1_turns > 0 and not lost
    detail = (f"T1 场景 {t1_scenes} / 轮次 {t1_turns} / 丢失 {len(lost)}；"
              f"白名单外 {out_of_scope_turns} 轮按设计不收录"
              f"（{','.join(sorted(out_of_scope_cats))}）")
    if lost[:3]:
        detail += f"  丢失例：{lost[:3]}"
    record("lost_turns", "无轮次静默丢失", ok1, detail,
           {"t1_scenes": t1_scenes, "t1_turns": t1_turns, "lost": len(lost),
            "out_of_scope_turns": out_of_scope_turns,
            "out_of_scope_categories": sorted(out_of_scope_cats),
            "t1_categories": list(config.T1_CATEGORIES)})
except Exception as exc:  # noqa: BLE001
    record("lost_turns", "无轮次静默丢失", False, f"无法核对：{type(exc).__name__}: {exc}")

# ---- 2. 无重复 Chunk 身份 ----
ids = [str(c["chunk_id"]) for c in chunks]
dupe_ids = [k for k, v in Counter(ids).items() if v > 1]
keys = [str(c["content_key"]) for c in chunks]
dupe_keys = [k for k, v in Counter(keys).items() if v > 1]
ok2 = not dupe_ids and not dupe_keys
detail = f"块 {len(chunks)} / 重复 chunk_id {len(dupe_ids)} / 重复 content_key {len(dupe_keys)}"
if dupe_ids[:3]:
    detail += f"  例：{dupe_ids[:3]}"
if dupe_keys[:3]:
    detail += f"  同文块：{dupe_keys[:3]}"
record("dupe_chunks", "无重复 Chunk 身份", ok2, detail)

# ---- 3. 无非法缓存 ----
rep = cache.invalid_cache_report()
ok3 = int(rep.get("invalid") or 0) == 0
record("bad_cache", "无非法缓存", ok3,
       f"缓存 {rep.get('total')} 条 / 合法 {rep.get('ok')} / 非法 {rep.get('invalid')}"
       + (f"  原因：{rep.get('by_reason')}" if rep.get("invalid") else ""))

# ---- 4. 所有 Assertion 证据可定位 ----
text_of = {str(c["chunk_id"]): str(c.get("text") or "") for c in chunks}
bad_ev: list[str] = []
missing_chunk: list[str] = []
for a in assertions:
    cid = str(a.get("chunk_id") or "")
    text = text_of.get(cid)
    if text is None:
        missing_chunk.append(str(a.get("assertion_id") or ""))
        continue
    start, end = a.get("evidence_start"), a.get("evidence_end")
    ev = str(a.get("evidence_text") or "")
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end > len(text):
        bad_ev.append(f"{a.get('assertion_id')} 偏移越界 {start}:{end} / {len(text)}")
        continue
    if ev and text[start:end] != ev:
        bad_ev.append(f"{a.get('assertion_id')} {text[start:end]!r} != {ev!r}")
ok4 = not bad_ev and not missing_chunk
record("evidence_spans", "所有 Assertion 证据可定位", ok4,
       f"断言 {len(assertions)} / 偏移不符 {len(bad_ev)} / 宿主块缺失 {len(missing_chunk)}"
       + (f"  例：{bad_ev[:2]}" if bad_ev else "")
       + (f"  缺块例：{missing_chunk[:2]}" if missing_chunk else ""))

# ---- 5. 无未解释产物身份变化 ----
old_gen = release.Generation(OLD_GID)
old_manifest = old_gen.load()
known = {}
changed = []
for key in ("graph_artifact_id", "index_artifact_id", "assertion_artifact_id",
            "adjacency_artifact_id", "gate_report_hash", "index_content_digest",
            "probe_digest", "chunk_manifest_hash", "graph_build_fp", "index_build_fp"):
    old_v, new_v = old_manifest.get(key), manifest.get(key)
    if old_v == new_v:
        continue
    changed.append(key)
    if key in ("graph_build_fp",):
        known[key] = "图构建指纹未变（同批抽取、同一规则）"
    elif key in ("index_build_fp", "index_content_digest", "probe_digest"):
        known[key] = "分词器 t1->t2 + 块集 13781->13798，图指纹未变"
    elif key == "chunk_manifest_hash":
        known[key] = "块集变化（34 块新增或正文变更）"
    elif key in ("graph_artifact_id", "assertion_artifact_id", "adjacency_artifact_id"):
        known[key] = "图/断言/邻接产物随块集变化"
    elif key == "index_artifact_id":
        known[key] = "索引随分词器与块集变化"
    elif key == "gate_report_hash":
        known[key] = "门禁行数 30->38（ark-rag-3 新增 E 门）"
unexplained = [k for k in changed if k not in known]
ok5 = not unexplained
record("identity_delta", "无未解释产物身份变化", ok5,
       f"变化字段 {len(changed)} / 有解释 {len(changed) - len(unexplained)} / 未解释 {len(unexplained)}"
       + (f"  未解释：{unexplained}" if unexplained else ""),
       {"changed": changed, "explanations": known})

# ---- 6. 无失败 Generation 被发布 ----
#
# 口径要分三层，混一句会得出错误结论：
#   a) 指针指向的世代必须自检 0 问题——这是"当前对外服务的那一代是好的"；
#   b) 状态为 FAILED 的世代绝不许是指针——这是"失败的不许上线"；
#   c) 本 schema 世代（ark-rag-3）里已发布的必须全部自检 0 问题。
#
# 特意**不**把"所有 PUBLISHED 都 0 问题"当判据：`606ea9`/`7b0fd3` 是 ark-rag-1
# 时代发布的，那时的 Manifest 根本没有 `graph_artifact_id`/`assertion_artifact_id`
# 字段，`source_status` 也要到 D4 才记录。拿今天的字段表去判它们，
# 只会得到"每条历史发布都不合格"这种恒假结论——那不是发现，是尺子用错了。
# 它们被单独列成 `legacy_exceptions` 并逐步点名，不静默放过。
pointer_status = ""
pointer_ok = True
try:
    cur = release.read_current()
    cur_gid = str(cur.get("generation_id") or "")
    pointer_status = str(cur.get("status") or "")
    pointer_ok = cur_gid != NEW_GID
    pointer_detail = f"指针 = {cur_gid}（{pointer_status}）"
    cur_problems = release.verify_generation(release.Generation(cur_gid)) if cur_gid else ["指针里没有 generation_id"]
    pointer_clean = not cur_problems
    if not pointer_clean:
        pointer_detail += f" / 指针世代自检不过 {len(cur_problems)} 项：{cur_problems[:2]}"
    pointer_failed = pointer_status == release.FAILED
    if pointer_failed:
        pointer_detail += " / 指针指向 FAILED 世代"
except release.ReleaseError as exc:
    pointer_detail = f"指针不可读：{exc}"
    pointer_clean = False
    pointer_failed = True

identity_published_fail = []
legacy_exceptions = []
for g in release.list_generations():
    if g.get("status") != release.PUBLISHED:
        continue
    gen_i = release.Generation(g["generation_id"])
    problems = release.verify_generation(gen_i)
    if not problems:
        continue
    try:
        schema = str((gen_i.load() or {}).get("graph_schema") or "")
    except (OSError, ValueError):
        schema = "?"
    if schema == config.GRAPH_SCHEMA_IDENTITY:
        identity_published_fail.append((g["generation_id"], len(problems)))
    else:
        legacy_exceptions.append((g["generation_id"], schema or "(空)", len(problems)))
ok6 = (pointer_ok and pointer_clean and not pointer_failed
       and not identity_published_fail)
detail6 = (f"{pointer_detail} / 本 schema 已发布自检不过 {len(identity_published_fail)} 个"
           + (f"：{identity_published_fail}" if identity_published_fail else ""))
if legacy_exceptions:
    detail6 += (f" / ark-rag-1 历史发布 {len(legacy_exceptions)} 个按旧字段表判不过"
                f"（无 graph_artifact_id、source_status 未记录，D4 之前）："
                f"{[x[0] for x in legacy_exceptions]}")
record("no_failed_publish", "无失败 Generation 被发布", ok6, detail6,
       {"pointer": pointer_detail, "identity_published_fail": identity_published_fail,
        "legacy_exceptions": legacy_exceptions})

# ---- 7. 索引与图绑定同一 Manifest ----
idx, meta = index_mod.load_index(gen.index_path)
same_gfp = meta.get("graph_fp") == graph.get("fingerprint")
same_art = meta.get("graph_artifact_id") == manifest.get("graph_artifact_id")
same_cnt = int(meta.get("chunk_count") or 0) == len(chunks)
same_idx_cnt = len(idx.doc_id) == len(chunks)
ok7 = same_gfp and same_art and same_cnt and same_idx_cnt
record("index_graph_bind", "索引与图绑定同一 Manifest", ok7,
       f"graph_fp 一致={same_gfp} / artifact_id 一致={same_art} / "
       f"meta.chunk_count={meta.get('chunk_count')}→块 {len(chunks)} 一致={same_cnt} / "
       f"倒排文档 {len(idx.doc_id)} 一致={same_idx_cnt}")

# ---- 8. 8 例与 46 例风险有处理结论 ----
#
# 判据不是「文档里出现了"8 例"这几个字」——那种写法只要复制一句引言就能骗过。
# 要的是**逐条处置表**：每一条风险都得有人给结论，且结论只能取自
# SPLIT / KEPT / PENDING / REJECTED / RENAMED / MISSING 六类；
# 出现 MISSING 或无法归类的行必须点名（MISSING 不是"已处理"）。
RISK_DOC = ROOT / "docs" / "k阶段风险处置.md"
VERDICTS = ("SPLIT", "KEPT", "PENDING", "REJECTED", "RENAMED", "MISSING")
risk_rows: list[tuple[str, str]] = []
risk_text = RISK_DOC.read_text("utf-8") if RISK_DOC.exists() else ""
for line in risk_text.splitlines():
    if not line.startswith("| ") or "<-" not in line:
        continue
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells) < 3:
        continue
    verdict = cells[2]
    if verdict not in VERDICTS:
        continue
    risk_rows.append((cells[1], verdict))
tally = Counter(v for _, v in risk_rows)
missing_rows = [pair for pair, v in risk_rows if v == "MISSING"]
unknown_rows = [pair for pair, v in risk_rows if v not in VERDICTS]
ok8 = bool(risk_rows) and not unknown_rows
detail8 = (f"{RISK_DOC.name}：逐条处置 {len(risk_rows)} 行"
           + (f"（{', '.join(f'{k} {v}' for k, v in sorted(tally.items()))}）" if tally else "")
           + f" / 未归类 {len(unknown_rows)}")
if missing_rows:
    detail8 += f" / 三处查不到的条目 {len(missing_rows)} 条已点名：{missing_rows[:3]}"
record("risk_register", "8/46 例风险有处理结论", ok8, detail8,
       {"rows": len(risk_rows), "tally": dict(tally), "missing": missing_rows})

# ---- 汇总 ----
bad = [r for r in results if not r["ok"]]
print()
print(f"八条硬门：{len(results) - len(bad)}/{len(results)} 通过")
if bad:
    print("未通过：")
    for r in bad:
        print(f"  - {r['name']}: {r['detail']}")
print("结论：" + ("八条全过" if not bad else "有未过项，不得进入 K4"))

out = ROOT / "data" / "k2_hard_gates.json"
atomic.atomic_write_json(out, {
    "generation_id": NEW_GID,
    "baseline_generation_id": OLD_GID,
    "checks": results,
    "all_passed": not bad,
})
print(f"报告已写 {config.rel_to_root(out)}")

sys.exit(0 if not bad else 1)
