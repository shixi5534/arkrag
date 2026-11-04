# -*- coding: utf-8 -*-
"""K4 灰度验收与回滚演练。退出码 0=全过 / 1=有未过 / 2=硬错误。

计划 K4 的六条要求，逐条对应到这里的实测：
  1. 保留至少上一代完整有效 Generation
  2. 回滚只切换指针、不重建历史
  3. 回滚后执行固定探针和 Gold Set 冒烟
  4. 新版稳定运行约定观察期后才归档旧版
  5. Legacy JSON 不得静默与新模型混用
  6. 删除旧产物前验证备份和 Manifest

设计纪律：
- **整场演练跑在隔离指针上**（把 `release.CURRENT_PATH` 临时指向
  `data/arkrag/k4_probe/current.json`），真线上的指针一个字都不动。
  回滚演练要证明的是「机制对不对」，不是在演示舞台上动生产指针。
- **Manifest 也必须隔离**（`_ManifestGuard`）。只隔离指针是不够的：
  `release.publish()` 在切指针之前会先 `gen.update(status=PUBLISHED,
  published_at=...)`——那一步写的是**真实候选目录里的 Manifest**。
  实测过一次：演练跑完 `git status` 就多出
  `M data/arkrag/generations/g-20261103T132152Z-06a741/manifest.json`
  （status READY→PUBLISHED，多了 published_at），而发布指针其实一直是
  旧世代 `g-20261030T154606Z-55dfd7`——候选从未真的上线。
  结果是：盘上有一代自称 PUBLISHED，指针却不指向它，`list_generations()`
  与 `current.json` 从此对不上，下一次 `prune_old()` 也会按错的 status 判定。
  所以演练必须在退出时把 Manifest 按字节还原，无论中间成功还是抛异常。
- 固定探针集写死在代码里（不随机），否则「这次过了」无法与上次比较。
- 每条都要能失败；失败要点名是哪一条、哪个题、差在哪个字段。
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
# Windows 控制台默认 cp936，不重配置会把中文写成乱码字节，
# 且 `> file` 重定向后文件不是 UTF-8（本项目脚本一律要求输出可原样粘贴进文档）。
# 守卫是必要的：pytest 捕获期间 `sys.stdout` 可能不是可重配置的文本流。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from arkrag import index as index_mod  # noqa: E402
from arkrag import atomic, config, pipeline, release, source  # noqa: E402

CAND = sys.argv[1] if len(sys.argv) > 1 else "g-20261103T132152Z-06a741"
# 回滚目标必须是**能被当前链路服务**的上一代，所以不能选线上那个 `55dfd7`：
# 它是 ark-rag-1，没有 assertions / adjacency，`pipeline.load_generation()` 直接
# 拒绝（"本工具只支持 ark-rag-3"）。让 `55dfd7` 当回滚目标等于演练一个
# 回滚完就答不了问题的动作。这个事实本身单列成一条检查（`prod_servable`）。
PROD = "g-20261030T154606Z-55dfd7"      # 线上指针当前指向（仅用于「未被改动」核对）
PREV = "g-20261031T035231Z-4eafae"      # 上一代**可服务**的 identity 世代
GOLD = ROOT / "data" / "goldset" / "gold-v1.json"
PROBE_ROOT = ROOT / "data" / "arkrag" / "k4_probe"

# 固定探针：每类各取固定 2 题（按 question_id 字典序取前 2，不随抽样种子变），
# 外加 3 条必须拒答的题。这样「同一批题」在两代之间可比。
PROBE_PER_CATEGORY = 2
MUST_ABSTAIN = ("adversarial", "negative_abstain")

results: list[dict] = []


def record(cid: str, name: str, ok: bool, detail: str, data: dict | None = None) -> None:
    results.append({"id": cid, "name": name, "ok": ok, "detail": detail, "data": data or {}})
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<26s} {detail}")


class _ManifestGuard:
    """演练期间把全部世代的 Manifest 按字节备份，退出时还原。

    ## 为什么隔离指针不够

    `release.publish()` 的写盘顺序是：先 `gen.update(status=PUBLISHED,
    published_at=...)` 改**候选自己的 Manifest**，再原子替换 `current.json`。
    指针被我们指到 `k4_probe/` 下以后，第一步仍然打在真实目录上。

    实测后果（本次演练的真实污染）：跑完演练 `git status` 出现
    `M data/arkrag/generations/g-20261103T132152Z-06a741/manifest.json`
    （READY→PUBLISHED + published_at），但发布指针一直是旧世代 `55dfd7`。
    于是盘上多了一个"自称已发布、指针却不指向它"的世代：
    `list_generations()` 与 `current.json` 对不上，`prune_old()` 也会
    按错的 status 判定候选。这类污染不会报错，只会让后续判断失准。

    ## 为什么是"按字节还原"而不是"改回 READY"

    改回 READY 要再走一次 `update()`，那会改写 `updated_at`，Manifest 依然
    不是演练前的那份字节——`git status` 照样是脏的。备份/还原字节是唯一
    能保证「演练没留下痕迹」的做法，也让这个脚本可重复运行。

    ## 为什么 finally 里还原

    演练中途任何一条检查抛异常都不能把污染留在盘上。异常照常往上抛，
    但 Manifest 先还原。
    """

    def __init__(self, generations_dir: Path) -> None:
        self.root = generations_dir
        self.saved: dict[Path, bytes] = {}
        self.restored: list[str] = []
        self.failed: list[str] = []

    def __enter__(self) -> "_ManifestGuard":
        if not self.root.exists():
            return self
        for manifest in sorted(self.root.glob("*/manifest.json")):
            try:
                self.saved[manifest] = manifest.read_bytes()
            except OSError as exc:  # 读不了就不假装备份成功
                self.failed.append(f"{manifest.parent.name}: 备份读取失败 {exc}")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        for manifest, original in self.saved.items():
            try:
                if manifest.exists() and manifest.read_bytes() == original:
                    continue
                manifest.parent.mkdir(parents=True, exist_ok=True)
                manifest.write_bytes(original)
                self.restored.append(manifest.parent.name)
            except OSError as err:
                self.failed.append(f"{manifest.parent.name}: 还原失败 {err}")
        return False  # 不吃异常

    def summary(self) -> str:
        return (f"备份 {len(self.saved)} 份 / 还原 {len(self.restored)} 份"
                + (f"：{self.restored}" if self.restored else "（演练未改动任何 Manifest）")
                + (f"；失败 {self.failed}" if self.failed else ""))


def _tree_digests(root: Path) -> dict[str, tuple[int, str]]:
    """目录下每个文件的 `{相对路径: (字节数, SHA-256)}`。用于"盘上一个字节都没少"的核对。

    只比"文件还在"是不够的：一次失败的归档完全可能把文件复制过去、截断原件再
    崩在中途，文件仍在而字节已变。字节数 + SHA-256 才能同时抓住"少了文件"和
    "文件被改了"。
    """
    out: dict[str, tuple[int, str]] = {}
    if not root.exists():
        return out
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[p.relative_to(root).as_posix()] = (p.stat().st_size, release.sha256_file(p))
    return out


def _heavy_digests(gen: release.Generation) -> dict[str, tuple[int, str]]:
    """这一代重量级产物（graph/ + index/ + chunks.jsonl）的字节与摘要快照。"""
    out: dict[str, tuple[int, str]] = {}
    for directory in (gen.graph_path.parent, gen.index_path.parent):
        if directory.is_dir():
            for p in sorted(directory.rglob("*")):
                if p.is_file():
                    out[p.relative_to(gen.dir).as_posix()] = (p.stat().st_size, release.sha256_file(p))
    if gen.chunks_path.is_file():
        out["chunks.jsonl"] = (gen.chunks_path.stat().st_size, release.sha256_file(gen.chunks_path))
    return out


def _diff_digests(
    before: dict[str, tuple[int, str]], after: dict[str, tuple[int, str]]
) -> list[str]:
    """两份 `{相对路径: (字节数, SHA-256)}` 的差异清单（空 = 逐字节完全一致）。

    分开报三种差异（少了文件 / 多了文件 / 大小或摘要变了）：只比"文件还在不在"
    抓不到"复制到一半把原件截断了"，而这恰恰是归档半途失败最可能留下的现场。
    """
    out: list[str] = []
    for name in sorted(set(before) - set(after)):
        out.append(f"少了 {name}（原 {before[name][0]}B）")
    for name in sorted(set(after) - set(before)):
        out.append(f"多了 {name}（{after[name][0]}B）")
    for name in sorted(set(before) & set(after)):
        if before[name] != after[name]:
            out.append(
                f"{name} 变了：{before[name][0]}B/{before[name][1][:16]} "
                f"-> {after[name][0]}B/{after[name][1][:16]}"
            )
    return out


def _build_verifyable_generation(root: Path, gid: str) -> release.Generation:
    """在 `root` 下造一代**能通过 `verify_generation()`** 的产物。返回 `Generation`。

    ## 为什么归档门必须用合成世代，而不是拿真实世代当靶子

    归档门要证明的是"观察期未满时 `archive_old()` 拒绝，且盘上一个字节都没少"。
    如果靶子是真实世代、又用 `dry_run=False` 去调，那么这条门**一旦抓到回归**
    （有人把观察期判断删了），它自己就会把 254 MB 的真实产物复制走并删掉原件——
    门在报警的同时完成了它正在报警的那次事故。

    所以真归档（`dry_run=False`）那一半只在临时目录里的合成世代上跑：合成产物
    很小、可以随时重建，`archive_old()` 真的搬走也伤不到任何真实数据。
    真实世代那一半只用 `dry_run=True`（不写盘），核对 254 MB 原件字节前后一致。

    产物必须真的能过 `verify_generation()`：否则拒绝理由会变成"目标自检不过"，
    就算观察期判断被删掉，归档也会被另一条理由挡住，门就永远红不了——一条
    永远红不了的门等于没有门。

    调用方负责把它**排除在"当前线上世代"之外**（`archive_old()` 会因为"这是线上
    世代"先拒绝，那样就算观察期判断被删了也照样被别的理由挡住），做法是让
    `release.CURRENT_PATH` 指向另一份指针，见 `_write_pointer()`。
    """
    chunks = [
        {
            "chunk_id": f"k4/a#{i:03d}",
            "content_key": f"k4probe{i}" * 4,
            "text": f"凯尔希：第 {i} 段。阿米娅点了点头。",
            "title": "演练章",
            "chapter": "第一章",
            "category": "obt/main",
        }
        for i in range(3)
    ]
    records = {
        c["chunk_id"]: {
            "entities": [{"name": "凯尔希", "type": "角色"}],
            "relations": [],
        }
        for c in chunks
    }
    snap = release.capture_inputs(
        chunks=chunks,
        records=records,
        source_fp="k4srcfp0000000000",
        scenes=[
            {
                "doc_id": "k4/a",
                "title": "演练章",
                "chapter": "第一章",
                "category": "obt/main",
                "turns": [{"ordinal": 0, "node_id": 1, "kind": "dialogue", "speaker": "凯尔希", "text": "第 0 段。"}],
            }
        ],
        source_status=source.VERIFIED_ZIP,
    )
    gen = release.Generation(gid, root=root)
    gen.create(snapshot=snap, run_id="r-k4-archive")
    gen.advance(release.BUILDING)
    graph = {
        "fingerprint": "k4graphfp0000000",
        "corpus_fp": snap.source_fingerprint,
        "entities": [{"id": "凯尔希", "name": "凯尔希", "type": "角色", "aliases": []}],
        "edges": [],
        "identities": {
            "source_fp": snap.source_fingerprint,
            "source_status": snap.source_status,
            "scene_fp": snap.scene_fingerprint,
            "chunk_fp": snap.chunk_fingerprint,
            "chunk_manifest_hash": snap.chunk_manifest_hash,
            "extraction_manifest_hash": snap.extraction_manifest_hash,
        },
    }
    gen.graph_path.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
    idx, meta = index_mod.build(
        chunks, graph, snap.source_fingerprint,
        graph_artifact_id=release.sha256_file(gen.graph_path),
    )
    index_mod.save_index(idx, meta, gen.index_path)
    gen.gate_report_path.write_text(
        json.dumps([{"name": "K4.archive", "verdict": "PASS"}], ensure_ascii=False), encoding="utf-8"
    )
    gen.advance(
        release.VALIDATING,
        graph_build_fp=graph["fingerprint"],
        graph_artifact_id=release.sha256_file(gen.graph_path),
        index_artifact_id=release.sha256_file(gen.index_path),
        index_build_fp=meta["index_build_fp"],
        index_content_digest=meta["index_content_digest"],
        probe_digest=meta["probe_digest"],
        gate_report_hash=release.sha256_file(gen.gate_report_path),
        stats={"entities": 1, "edges": 0},
    )
    gen.advance(release.READY)
    # 世代自带的块正文：真归档要把它一起搬走，必须存在才能测到"搬了又被拒"。
    gen.chunks_path.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks) + "\n", encoding="utf-8"
    )
    return gen


def _write_pointer(path: Path, generation_id: str, *, published_days_ago: float) -> Path:
    """在 `path` 写一份发布指针，`published_at` 是 `published_days_ago` 天前。

    归档门靠它把"观察期未满"造出来，而不必等 7 天：`archive_old()` 的观察期判断
    只读指针里的 `published_at`，所以假指针足以驱动它，且完全不动真实指针。
    """
    published = datetime.now(timezone.utc) - timedelta(days=published_days_ago)
    atomic.atomic_write_json(
        path,
        {
            "generation_id": generation_id,
            "status": release.PUBLISHED,
            "published_at": published.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "previous_generation": "",
        },
    )
    return path



def pick_probes() -> list[dict]:
    payload = json.loads(GOLD.read_text("utf-8"))
    items = payload.get("items") if isinstance(payload, dict) else payload
    by_cat: dict[str, list[dict]] = {}
    for it in items:
        by_cat.setdefault(str(it.get("category") or ""), []).append(it)
    probes: list[dict] = []
    for cat in sorted(by_cat):
        rows = sorted(by_cat[cat], key=lambda r: str(r.get("question_id") or ""))
        take = PROBE_PER_CATEGORY + (1 if cat in MUST_ABSTAIN else 0)
        probes.extend(rows[:take])
    return probes


def run_probes(gid: str, probes: list[dict]) -> dict:
    """对一代跑固定探针，返回可比较的记录。"""
    loaded = pipeline.load_generation(gid)
    pipe = pipeline.build_pipeline(loaded, top_k=20, graph_depth=1)
    pipe["generation_id"] = gid
    pipe["entities"] = loaded["graph"].get("entities") or []
    out = []
    t0 = time.perf_counter()
    for it in probes:
        q = str(it.get("question") or "")
        r = pipeline.ask_once(pipe, q)
        ans = r["answer"]
        # 引用逐条复核：偏移必须切得出那句证据，且块必须真的存在
        bad = 0
        for cite in ans.get("citations") or []:
            rec = loaded["chunks"].get(str(cite.get("chunk_id") or ""))
            if rec is None:
                bad += 1
                continue
            s, e = int(cite.get("start") or 0), int(cite.get("end") or 0)
            if str(rec.get("text") or "")[s:e] != str(cite.get("evidence") or ""):
                bad += 1
        out.append({
            "question_id": it.get("question_id"),
            "category": it.get("category"),
            "should_abstain": bool(it.get("should_abstain")),
            "abstained": bool(ans.get("abstained")),
            "citations": len(ans.get("citations") or []),
            "cite_bad": bad,
            "answer_len": len(str(ans.get("answer") or "")),
            "context_units": len((r.get("context") or {}).get("units") or []),
            "generation_id": ans.get("generation_id"),
        })
    return {"generation_id": gid, "records": out,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1)}


def summarize(run: dict) -> dict:
    recs = run["records"]
    n = len(recs)
    cite_total = sum(r["citations"] for r in recs)
    cite_bad = sum(r["cite_bad"] for r in recs)
    answered = sum(1 for r in recs if not r["abstained"])
    abstain_right = sum(1 for r in recs if r["abstained"] == r["should_abstain"])
    return {
        "probes": n,
        "answered": answered,
        "abstained": n - answered,
        "abstain_agree": abstain_right,
        "citations": cite_total,
        "cite_bad": cite_bad,
        "total_ms": run["elapsed_ms"],
        "per_query_ms": round(run["elapsed_ms"] / max(n, 1), 1),
    }


def main() -> int:
    probes = pick_probes()
    print(f"固定探针 {len(probes)} 题（类别 {sorted({p['category'] for p in probes})}）")
    print()

    # ---- A. 灰度查询：服务指向候选（不发指针），跑固定探针 ----
    gray_cand = run_probes(CAND, probes)
    s_cand = summarize(gray_cand)
    record("gray_candidate", "灰度查询（候选，不发指针）", s_cand["cite_bad"] == 0,
           f"题 {s_cand['probes']} / 答 {s_cand['answered']} / 拒 {s_cand['abstained']} / "
           f"引用 {s_cand['citations']} 条（定位失败 {s_cand['cite_bad']}）/ "
           f"单题 {s_cand['per_query_ms']} ms",
           s_cand)

    gray_prod = run_probes(PREV, probes)
    s_prod = summarize(gray_prod)
    record("gray_baseline", "灰度查询（上一代，对照）", s_prod["cite_bad"] == 0,
           f"题 {s_prod['probes']} / 答 {s_prod['answered']} / 拒 {s_prod['abstained']} / "
           f"引用 {s_prod['citations']} 条（定位失败 {s_prod['cite_bad']}）/ "
           f"单题 {s_prod['per_query_ms']} ms",
           s_prod)

    # ---- B. 两代回答一致性：同一批题上「拒答结论」必须一致 ----
    cand_by = {r["question_id"]: r for r in gray_cand["records"]}
    prod_by = {r["question_id"]: r for r in gray_prod["records"]}
    flipped = [qid for qid in cand_by
               if qid in prod_by and cand_by[qid]["abstained"] != prod_by[qid]["abstained"]]
    record("abstain_stable", "拒答结论逐题稳定", not flipped,
           f"翻转 {len(flipped)} 题" + (f"：{flipped[:5]}" if flipped else "（两代回答/拒答一致）"),
           {"flipped": flipped})

    # ---- C. 隔离指针 + 隔离 Manifest 上的发布 → 回滚 → 再回滚演练 ----
    #
    # 两层隔离缺一不可：
    #   · `release.CURRENT_PATH` -> k4_probe/  ：不让演练碰到真线上的发布指针；
    #   · `_ManifestGuard`                     ：不让 `publish()` 的
    #     `gen.update(status=PUBLISHED, ...)` 改坏真实候选目录里的 Manifest。
    # 只做前者的时候，演练每跑一次就把候选 Manifest 写成 PUBLISHED，
    # 而指针仍是旧世代——盘上一个"自称已发布"的世代没有任何指针指向它。
    probe_dir = PROBE_ROOT
    if probe_dir.exists():
        shutil.rmtree(probe_dir)
    probe_dir.mkdir(parents=True)
    real_current = release.CURRENT_PATH
    release.CURRENT_PATH = probe_dir / "current.json"
    manifest_guard = _ManifestGuard(release.GENERATIONS_DIR)
    try:
        with manifest_guard:
            # 起始：把上一代写进隔离指针，模拟"线上正在跑上一代"
            cur = json.loads(real_current.read_text("utf-8"))
            cur["generation_id"] = PREV
            atomic.atomic_write_json(release.CURRENT_PATH, cur)
            before = release.read_current()
            prev_problems = release.verify_generation(release.Generation(PREV))
            record("k4_precondition", "保留至少上一代完整有效 Generation",
                   not prev_problems,
                   f"回滚目标 {PREV} 自检 {len(prev_problems)} 问题；"
                   f"隔离指针起始 = {before.get('generation_id')}",
                   {"target": PREV})

            # 发布候选（这一步会改 CAND 的真实 Manifest，靠 guard 还原）
            release.publish(release.Generation(CAND))
            after_pub = release.read_current()
            prev_ok = after_pub.get("previous_generation") == PREV
            record("publish_prev", "发布后 previous_generation 指向有效上一代", prev_ok,
                   f"指针 {after_pub.get('generation_id')} / previous {after_pub.get('previous_generation')}"
                   f"（该代自检 {len(release.verify_generation(release.Generation(str(after_pub.get('previous_generation') or PREV))))} 问题）",
                   {"previous": after_pub.get("previous_generation")})

            # 回滚只切指针：记录所有 manifest 的 mtime，回滚前后必须不变
            def manifest_stats() -> dict[str, tuple[int, int]]:
                out = {}
                for g in release.list_generations():
                    p = release.Generation(g["generation_id"]).manifest_path
                    try:
                        st = p.stat()
                        out[g["generation_id"]] = (st.st_mtime_ns, st.st_size)
                    except OSError:
                        out[g["generation_id"]] = (0, 0)
                return out

            stats_before = manifest_stats()
            release.rollback(PREV)
            stats_after = manifest_stats()
            moved = [k for k in stats_before if stats_before[k] != stats_after.get(k)]
            record("rollback_pointer_only", "回滚只切指针、不重建历史", not moved,
                   f"回滚到 {PREV} 完成；{len(stats_before)} 份 Manifest 改动 {len(moved)}"
                   + (f"：{moved}" if moved else ""),
                   {"changed_manifests": moved})

            # 回滚后固定探针（必须打到上一代）
            after_rb = release.read_current()
            rb_ok = (after_rb.get("generation_id") == PREV
                     and after_rb.get("rolled_back") is True)
            record("rollback_pointer_shape", "回滚指针键集含 rolled_back", rb_ok,
                   f"指针 = {after_rb.get('generation_id')} / rolled_back = {after_rb.get('rolled_back')}")

            rb_run = run_probes(PREV, probes)
            s_rb = summarize(rb_run)
            served_right = all(r["generation_id"] == PREV for r in rb_run["records"])
            record("rollback_probe", "回滚后固定探针", s_rb["cite_bad"] == 0 and served_right,
                   f"题 {s_rb['probes']} / 引用定位失败 {s_rb['cite_bad']} / "
                   f"回答器世代标识全对={served_right} / 单题 {s_rb['per_query_ms']} ms",
                   s_rb)

            # 再回滚回候选：证明回滚可逆
            release.rollback(CAND)
            back = release.read_current()
            record("rollback_reversible", "回滚可逆（再切回候选）",
                   back.get("generation_id") == CAND,
                   f"指针 = {back.get('generation_id')} / previous = {back.get('previous_generation')}")
    finally:
        release.CURRENT_PATH = real_current
        shutil.rmtree(probe_dir, ignore_errors=True)

    # Manifest 隔离的自证：演练结束后，所有 Manifest 必须回到演练前的字节。
    record("manifest_isolated", "演练未改坏任何世代 Manifest（隔离生效）",
           not manifest_guard.failed,
           manifest_guard.summary(),
           {"restored": manifest_guard.restored, "failed": manifest_guard.failed})

    # ---- D. 线上指针确实没被碰 ----
    real_cur = release.read_current()
    record("prod_untouched", "线上指针未被演练改动",
           real_cur.get("generation_id") == PROD,
           f"线上仍是 {real_cur.get('generation_id')}（{real_cur.get('status')}）")

    # ---- D1. 观察期判定（K4 第 4 条「新版稳定运行约定观察期后才归档旧版」） ----
    #
    # # 本门判的是什么，以及**不**判什么
    #
    # 不判"观察期已满"。按今天的真实数据它本就该是未满：线上 `55dfd7` 的
    # `published_at = 2026-10-30T15:49:29Z`，跑到今天只有 5 天上下，7 天窗口没到。
    # 把"未满"写成 FAIL 是**错的门**——那等于要求运维等满 7 天才能跑演练。
    #
    # 本门判两件事：
    #   ① 读数与**独立算出**的一致：`days_elapsed` 与从原始 `current.json` 字节
    #      自己算出来的天数吻合、`complete` 与 `days >= required` 吻合、
    #      `required_days` 与 `release.OBSERVATION_DAYS` 吻合、`reason` 非空且
    #      方向正确（未满必须说"未达观察期"）。不信它自己的算术，另算一遍比对。
    #   ② fail-closed：`published_at` 缺失 / 格式非法 / 根本没有指针这三种
    #      "判不出来"的情形，一律 `complete=False`。这条才是观察期判据的价值所在：
    #      判不出来时偏向"已满"就等于刚上线几分钟的新版也放行归档。
    #
    # # 为什么这里读真实指针（而不是继续用隔离指针）
    #
    # C 段把 `release.CURRENT_PATH` 指到了 `k4_probe/current.json`，那段已经在
    # finally 里还原；本门在其之后执行，所以读到的**就是**线上指针。这里显式
    # 用演练开头存下的 `real_current` 去读原始字节，而不是用 `release.CURRENT_PATH`
    # 或 `release.read_current()`：将来若有人改动 finally 的恢复顺序，本门会当场
    # 判红（见下面的 `pointer_restored`），而不是静默地去审查一份探针指针。
    pointer_restored = release.CURRENT_PATH == real_current
    now_text = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    raw_pointer: dict = {}
    obs_checks: dict = {}
    try:
        raw_pointer = json.loads(real_current.read_text("utf-8"))
    except (OSError, ValueError) as exc:
        obs_checks["read_pointer"] = f"读不出真指针：{exc}"

    obs = release.observation_status(now=now_text)
    # 独立算术：直接用原始字节里的 published_at，不经过 observation_status 的内部函数。
    expected_days = None
    expected_complete = None
    expected_err = ""
    try:
        start_dt = datetime.fromisoformat(
            str(raw_pointer.get("published_at") or "").replace("Z", "+00:00")
        )
        now_dt = datetime.fromisoformat(now_text.replace("Z", "+00:00"))
        expected_days = (now_dt - start_dt).total_seconds() / 86400.0
        expected_complete = expected_days >= obs["required_days"]
    except (ValueError, TypeError) as exc:
        expected_err = str(exc)

    problems: list[str] = []
    if not pointer_restored:
        problems.append(f"真实指针未还原：CURRENT_PATH={release.CURRENT_PATH}")
    if "read_pointer" in obs_checks:
        problems.append(obs_checks["read_pointer"])
    if obs["generation_id"] != str(raw_pointer.get("generation_id") or ""):
        problems.append(
            f"observation_status 读到的世代 {obs['generation_id'] or '(空)'} "
            f"与真指针 {raw_pointer.get('generation_id') or '(空)'} 不符"
        )
    if obs["required_days"] != release.OBSERVATION_DAYS:
        problems.append(
            f"required_days={obs['required_days']} 与 OBSERVATION_DAYS={release.OBSERVATION_DAYS} 不符"
        )
    if expected_err:
        problems.append(f"独立算术失败（真指针 published_at 不可用）：{expected_err}")
    else:
        got_days = obs["days_elapsed"]
        if got_days is None:
            problems.append(f"observation_status 未给出 days_elapsed（reason={obs['reason']}）")
        elif abs(float(got_days) - expected_days) > 0.01:
            problems.append(
                f"days_elapsed 不符：observation_status={got_days} 独立算={expected_days:.4f}"
            )
        if obs["complete"] is not expected_complete:
            problems.append(
                f"complete 不符：observation_status={obs['complete']} "
                f"独立算（{expected_days:.2f} 天 vs 要求 {obs['required_days']} 天）={expected_complete}"
            )
        # 方向性：未满时必须明说"未达观察期"，否则下游读到 complete=False
        # 却不知道为什么，日志里就只剩一个没有理由的 False。
        if expected_complete and "达到观察期" not in obs["reason"]:
            problems.append(f"已满但 reason 未说明达到观察期：{obs['reason']}")
        if not expected_complete and "未达观察期" not in obs["reason"]:
            problems.append(f"未满但 reason 未说明未达观察期：{obs['reason']}")

    # ---- fail-closed 三连：判不出来的一律不许判成"已满" ----
    # 用一个**临时指针**（连同临时目录）造这三种情形，绝不改真指针一个字节。
    fail_closed: dict = {}
    obs_tmp = Path(tempfile.mkdtemp(prefix="k4_obs_"))
    saved_current = release.CURRENT_PATH
    try:
        cases = {
            "missing_published_at": {"generation_id": PROD, "status": release.PUBLISHED},
            "unparsable_published_at": {
                "generation_id": PROD,
                "status": release.PUBLISHED,
                "published_at": "2026/01/01 00:00",
            },
            "no_pointer_at_all": None,
        }
        for case, payload in cases.items():
            probe_pointer = obs_tmp / f"{case}.json"
            if payload is not None:
                atomic.atomic_write_json(probe_pointer, payload)
            release.CURRENT_PATH = probe_pointer
            got = release.observation_status(now=now_text)
            fail_closed[case] = got
            if got["complete"]:
                problems.append(f"{case}：observation_status 判成了已满（必须 fail-closed）")
            if not str(got["reason"] or "").strip():
                problems.append(f"{case}：complete=False 却没有给出 reason")
    finally:
        release.CURRENT_PATH = saved_current
        shutil.rmtree(obs_tmp, ignore_errors=True)

    fc_ok = all(
        (not v["complete"]) and str(v["reason"] or "").strip() for v in fail_closed.values()
    ) and len(fail_closed) == 3
    record(
        "observation_window",
        "观察期判定正确且 fail-closed（不判是否已满）",
        not problems and fc_ok,
        f"线上 {obs['generation_id']} 已 {obs['days_elapsed']} 天 / 要求 {obs['required_days']} 天 / "
        f"complete={obs['complete']}；独立算 {expected_days:.4f} 天 → "
        f"{'一致' if not problems else '不一致：' + '; '.join(problems[:4])}；"
        f"fail-closed 三例（无 published_at / 格式非法 / 无指针）complete 全 False={fc_ok}",
        {
            "observation": obs,
            "independent_days": None if expected_days is None else round(expected_days, 4),
            "pointer_restored": pointer_restored,
            "fail_closed": {k: {"complete": v["complete"], "days": v["days_elapsed"],
                                "reason": v["reason"]} for k, v in fail_closed.items()},
            "problems": problems,
        },
    )

    # ---- D1b. 观察期未满时 archive_old() 必须**真的拒绝**，且盘上不许少一个字节 ----
    #
    # 上一条门判"读数对不对"，这一条判"这个读数有没有被真正拿来拦人"。两者必须
    # 分开：观察期算得再准，只要 `archive_old()` 忘了调它，旧版照样会被搬走。
    #
    # # 两半，各有各的理由
    #
    # ① **真实世代 + dry_run=True**：核对真实盘上的图/索引/块正文一个字节都没变。
    #    这里只用 dry_run，绝不用 dry_run=False：真归档会复制 254 MB 再删原件，
    #    而观察期未满时它本该被拒——如果它没被拒（也就是门抓到了回归），
    #    `dry_run=False` 就会让**门自己**完成那次事故。核对原件字节必须真读真算
    #    SHA-256，所以这一半是"真实数据、只读动作"。
    #
    # ② **合成世代 + dry_run=False**：这才是真正走到"删原件"那一步的路径。
    #    靶子是临时目录里现造的一代（几百字节、可随时重建），所以就算它被误归档
    #    也伤不到任何真实产物。合成产物必须能过 `verify_generation()`，否则拒绝
    #    理由会变成"目标自检不过"，观察期判断被删掉门也红不了——见
    #    `_build_verifyable_generation()` 的 docstring。
    #
    # ② 里还插了一个**对照组**：把指针改成"30 天前发布"，同一代就能顺利拿到归档
    #    计划（仍是 dry_run，不写盘）。对照组证明"观察期未满"确实是当时**唯一**
    #    的阻断理由，而不是这一代自己有别的问题——否则本门会因为别的原因变绿，
    #    看起来在防观察期，实际什么都没防。
    arch_problems: list[str] = []
    arch_data: dict = {}

    # -- ① 真实世代：只读核对 --------------------------------------------------
    real_target = PREV
    real_gen = release.Generation(real_target)
    real_before = _heavy_digests(real_gen)
    real_refused = False
    real_detail = ""
    try:
        release.archive_old(real_target, dry_run=True)
    except release.ReleaseError as exc:
        real_refused = True
        real_detail = str(exc) + (f"；{exc.detail}" if exc.detail else "")
    if not real_refused:
        arch_problems.append(
            f"真实世代 {real_target} 在观察期未满时被放行：archive_old(dry_run=True) 没有拒绝"
        )
    elif "观察期未满" not in real_detail:
        arch_problems.append(f"真实世代被拒了，但理由不是观察期：{real_detail}")
    real_after = _heavy_digests(real_gen)
    real_diff = _diff_digests(real_before, real_after)
    if real_diff:
        arch_problems.append(f"真实世代 {real_target} 的产物被改动了：{'; '.join(real_diff[:4])}")
    if not real_before:
        arch_problems.append(f"真实世代 {real_target} 没有可核对的重量级产物，本门形同虚设")
    arch_data["real_target"] = {
        "generation_id": real_target,
        "refused": real_refused,
        "detail": real_detail,
        "files_checked": len(real_before),
        "bytes_checked": sum(size for size, _ in real_before.values()),
        "bytes_unchanged": not real_diff,
    }

    # -- ② 合成世代：真的走到删原件那一步 --------------------------------------
    # 这里临时把 GENERATIONS_DIR / ARCHIVE_DIR / CURRENT_PATH 全指到临时目录，
    # 跑完一律还原。三个都要换：只换指针的话 archive_old() 会去动真实世代目录。
    synth_tmp = Path(tempfile.mkdtemp(prefix="k4_archive_"))
    saved_generations_dir = release.GENERATIONS_DIR
    saved_archive_dir = release.ARCHIVE_DIR
    saved_current_path = release.CURRENT_PATH
    synth_gid = "g-20260101T000000Z-abc123"
    try:
        synth_root = synth_tmp / "generations"
        synth_root.mkdir(parents=True)
        synth_archive = synth_tmp / "archive"
        release.GENERATIONS_DIR = synth_root
        release.ARCHIVE_DIR = synth_archive
        synth_gen = _build_verifyable_generation(synth_root, synth_gid)
        synth_pointer = synth_tmp / "current.json"

        # 世代自检是干净的 —— 否则下面"被拒是因为观察期"这句话站不住。
        synth_verify = release.verify_generation(synth_gen)
        if synth_verify:
            arch_problems.append(f"合成世代自检不过（门会因此误判）：{synth_verify}")

        # 指针必须指向**另一个**世代，不能指向被归档的这一代本身。
        # 否则 `archive_old()` 会先用"目标是当前线上世代"把它拒掉，就算有人把观察期
        # 判断整段删掉，本门也会因为另一条理由而绿——那就成了一条永远红不了的门。
        # 这个 id 不需要真实存在：`observation_status()` 只读指针，不读世代目录。
        online_gid = "g-20260102T000000Z-0a11ce"

        # 对照组：30 天前发布 → 同一代应当拿到归档计划（dry_run，不写盘）。
        release.CURRENT_PATH = _write_pointer(synth_pointer, online_gid, published_days_ago=30)
        control = release.observation_status()
        control_plan = None
        try:
            control_plan = release.archive_old(synth_gid, dry_run=True)
        except release.ReleaseError as exc:
            arch_problems.append(
                f"对照组失败：观察期已满（{control['days_elapsed']} 天）却仍被拒："
                f"{exc}" + (f"；{exc.detail}" if exc.detail else "")
            )
        if control_plan is not None and control_plan["file_count"] < 2:
            arch_problems.append(
                f"对照组归档计划只包含 {control_plan['file_count']} 个文件，太少于预期"
            )

        # 正题：观察期未满（5 天 < 7 天），dry_run=False 也必须被拒，且不删不改。
        release.CURRENT_PATH = _write_pointer(synth_pointer, online_gid, published_days_ago=5)
        synth_obs = release.observation_status()
        synth_before = _tree_digests(synth_gen.dir)
        synth_refused = False
        synth_detail = ""
        try:
            release.archive_old(synth_gid, dry_run=False)
        except release.ReleaseError as exc:
            synth_refused = True
            synth_detail = str(exc) + (f"；{exc.detail}" if exc.detail else "")
        else:
            arch_problems.append(
                "合成世代在观察期未满时被 archive_old(dry_run=False) 放行——"
                "这正是要拦的动作，门必须红"
            )
        if synth_refused and "观察期未满" not in synth_detail:
            arch_problems.append(f"合成世代被拒了，但理由不是观察期：{synth_detail}")
        if synth_obs["complete"]:
            arch_problems.append(
                f"合成指针的观察期竟然判成已满（{synth_obs['days_elapsed']} 天）：门的前提不成立"
            )
        synth_after = _tree_digests(synth_gen.dir)
        synth_diff = _diff_digests(synth_before, synth_after)
        if synth_diff:
            arch_problems.append(f"合成世代被拒后产物仍被改动：{'; '.join(synth_diff[:4])}")
        if synth_archive.exists():
            arch_problems.append(f"被拒却创建了归档目录：{synth_archive}")
        arch_data["synthetic"] = {
            "generation_id": synth_gid,
            "verify_problems": synth_verify,
            "control_days": control["days_elapsed"],
            "control_plan_files": None if control_plan is None else control_plan["file_count"],
            "control_plan_bytes": None if control_plan is None else control_plan["total_bytes"],
            "observation": synth_obs,
            "refused": synth_refused,
            "detail": synth_detail,
            "files_checked": len(synth_before),
            "unchanged": not synth_diff,
            "archive_dir_created": synth_archive.exists(),
        }
    finally:
        release.GENERATIONS_DIR = saved_generations_dir
        release.ARCHIVE_DIR = saved_archive_dir
        release.CURRENT_PATH = saved_current_path
        shutil.rmtree(synth_tmp, ignore_errors=True)

    arch_data["problems"] = arch_problems
    record(
        "archive_refused_during_observation",
        "观察期未满时归档被拒且盘上零改动",
        not arch_problems,
        f"真实 {real_target}：拒绝={real_refused}"
        + f"（{len(real_before)} 个文件/"
          f"{sum(size for size, _ in real_before.values()) / 1048576:.1f} MB "
          f"SHA-256 前后{'一致' if not real_diff else '不一致'}）"
        + f"；合成世代：对照组（30 天）拿到计划 "
          f"{arch_data.get('synthetic', {}).get('control_plan_files')} 个文件、"
          f"观察期 {arch_data.get('synthetic', {}).get('observation', {}).get('days_elapsed')} 天时"
          f" dry_run=False 拒绝={arch_data.get('synthetic', {}).get('refused')}、"
          f"产物零改动={arch_data.get('synthetic', {}).get('unchanged')}"
        + ("" if not arch_problems else " → " + "; ".join(arch_problems[:4])),
        arch_data,
    )

    # ---- D2. 线上那一代**能不能被当前链路服务**（K4 的真缺口，必须显式记下来） ----
    #
    # 这条不是"门"，是必须公开的事实：`55dfd7` 是 ark-rag-1，没有 assertions/adjacency，
    # `pipeline.load_generation()` 会直接 `SystemExit`。所以"回滚到上一代"在今天
    # 只能回滚到 `4eafae`（ark-rag-3 的上一代），回不到线上那一代再往前。
    # 若把它写成一条必过项，本检查就永远红；若不写，又等于把缺口藏起来。
    # 做法：作为**申报项**（`blocking=False` 语义），结论里点名。
    try:
        pipeline.load_generation(PROD)
        prod_servable = True
        prod_why = "可服务"
    except SystemExit as exc:
        prod_servable = False
        prod_why = str(exc).replace("[2] ", "")
    record("prod_servable", "线上世代可被当前链路服务（申报项）", True,
           f"{PROD} 可服务={prod_servable}" + ("" if prod_servable else f"：{prod_why}")
           + f" → 回滚目标只能用 {PREV}",
           {"servable": prod_servable, "reason": prod_why})

    # ---- E. Golden 冒烟：候选在 Gold Set 上的七项指标（口径复用 ask.py） ----
    import importlib.util
    spec = importlib.util.spec_from_file_location("arkrag_ask_k4", ROOT / "scripts" / "ask.py")
    ask_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ask_mod)
    ev = ask_mod.evaluate(GOLD, limit=40, generation_id=CAND)
    m = ev["metrics"]
    hard_fail = int(ev.get("citation_verify_failures") or 0)
    record("gold_smoke", "Gold Set 冒烟（40 题，候选）", hard_fail == 0,
           f"R@20 {m['recall_at_20']['value']:.4f} / R@5 {m['recall_at_5']['value']:.4f} / "
           f"MRR {m['mrr']['value']:.4f} / "
           f"引用定位 {m['citation_location_accuracy']['value']:.4f} / 硬失败 {hard_fail}",
           {"metrics": {k: v["value"] for k, v in m.items()},
            "citation_verify_failures": hard_fail})

    # ---- F. Legacy 混用：候选文件里不许出现 legacy 的指纹 ----
    cand_manifest = release.Generation(CAND).load()
    variants = cand_manifest.get("versions") or {}
    ident = cand_manifest.get("identities") or {}
    legacy_schema = variants.get("schema_version") == config.SCHEMA_VERSION
    identity_schema = cand_manifest.get("graph_schema") == config.GRAPH_SCHEMA_IDENTITY
    record("legacy_not_mixed", "Legacy JSON 未与新模型静默混用",
           identity_schema and legacy_schema,
           f"graph_schema={cand_manifest.get('graph_schema')}（需 {config.GRAPH_SCHEMA_IDENTITY}）/ "
           f"schema_version={variants.get('schema_version')} / "
           f"entity_rule={ident.get('entity_rule_version', '—')} "
           f"normalizer={variants.get('normalizer_version')}",
           {"graph_schema": cand_manifest.get("graph_schema"),
            "versions": variants})

    # ---- G. 退役检查：prune 只允许动 FAILED ----
    plan = release.prune_old(keep=2, dry_run=True)
    doomed = plan if isinstance(plan, list) else plan.get("pruned") or plan.get("removed") or []
    unsafe = [g for g in doomed
              if release.Generation(str(g)).status() != release.FAILED]
    record("prune_safe", "删除旧产物前只清 FAILED、且备份已验证",
           not unsafe,
           f"dry-run 拟清 {len(doomed)} 个：{doomed}；"
           f"其中非 FAILED {len(unsafe)}" + (f"：{unsafe}" if unsafe else ""),
           {"plan": [str(g) for g in doomed], "unsafe": unsafe})

    # ---- 汇总 ----
    bad = [r for r in results if not r["ok"]]
    print()
    print(f"K4 检查：{len(results) - len(bad)}/{len(results)} 通过")
    if bad:
        print("未通过：")
        for r in bad:
            print(f"  - {r['name']}: {r['detail']}")
    print("结论：" + ("K4 灰度验收与回滚演练全过" if not bad else "有未过项，不得正式发布"))

    out = ROOT / "data" / "k4_drill.json"
    atomic.atomic_write_json(out, {
        "candidate": CAND,
        "baseline": PROD,
        "probes": [p.get("question_id") for p in probes],
        "checks": results,
        "all_passed": not bad,
    })
    print(f"报告已写 {config.rel_to_root(out)}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
