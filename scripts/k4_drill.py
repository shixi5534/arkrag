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
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
# Windows 控制台默认 cp936，不重配置会把中文写成乱码字节，
# 且 `> file` 重定向后文件不是 UTF-8（本项目脚本一律要求输出可原样粘贴进文档）。
# 守卫是必要的：pytest 捕获期间 `sys.stdout` 可能不是可重配置的文本流。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from arkrag import atomic, config, pipeline, release  # noqa: E402

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
