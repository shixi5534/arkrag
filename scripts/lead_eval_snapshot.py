"""把一次「正式评测」落成可追溯的快照文件。

为什么需要这个脚本
==================
`scripts/ask.py --evaluate` 只把七项指标打到 stdout，除非显式给 `--json`。
2026-11-05 实测过一次：跑完报了一串数字，回头要写报告时发现**盘上什么都没有**，
只能用肉眼记住的数——那正是本项目反复栽的那类坑（「有个数字不等于那个数字
是测量的」）。所以落盘必须是默认动作，不能是记得住才有。

第二个理由：`ask.py` 的七项指标**不带任何溯源信息**，也没有 `TITLE_ANSWER_PATH`
这类开关的对照。想回答「这个数是哪份代码、哪个候选、哪套数据、何时跑出来的」，
就必须在外面把七字段拼上去，而拼的时候最容易漏的是 `inputs` 的 sha16——
只写路径不写 sha16 的报告，指标不可重算（`provenance.verify()` 判 `unknown`）。

判据契约（全部来自 `arkrag.provenance`，本脚本不另立一套）
==========================================================
* 七字段：`REQUIRED_FIELDS`，缺一项就 `unknown`，**不是 pass**；
* 引用覆盖必须并列三个分母 `COVERAGE_DENOMINATORS`（`mixed` / `reachable` /
  `ungated`），P3 盯 `ungated`；
* 报前 5 名指标必须附 `rank5_conversion = 0.4167`（实测断崖值）；
* 含 recall 的行必须带 `caliber`（本脚本只有 gold 口径，故写 `gold_only`）
  并且**并列至少一个下游量**——`hop0.5` 那次 R@20 六配置第二高而真进上下文
  净 0，就是「只看 R@n 会选错配置」的实证；
* 累计耗时必须另有 CPU 时间（墙钟会被机器休眠污染：实测报过 24141.60s，
  而进程 CPU 只有 ~778s）。

用法
====
    py -3 scripts/lead_eval_snapshot.py --gold-set data/goldset/gold-v1.json \\
        --limit 320 --out data/eval_snapshot_title_off.json
    py -3 scripts/lead_eval_snapshot.py --gold-set data/goldset/gold-v1.json \\
        --limit 320 --title-path on --out data/eval_snapshot_title_on.json

退出码：0 跑通且自洽 / 2 输入不可用 / 4 自检失败（两次跑出的指标不一致）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from arkrag import answer, config, provenance  # noqa: E402

#: 引用覆盖的三个分母在快照里的键名。顺序与 `provenance.COVERAGE_DENOMINATORS`
#: 一致，但这里只是搬运，判据仍在 `provenance` 里。
COVERAGE_KEYS = ("mixed", "reachable", "ungated")


def _sha256_file(path: Path) -> str | None:
    """整个文件的 SHA-256；文件不在盘上返回 None，不抛。"""
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _metric_block(metrics: dict, key: str) -> dict:
    row = metrics.get(key) or {}
    return {
        "value": row.get("value"),
        "n": row.get("n"),
        "skipped": row.get("skipped"),
    }


def _answer_contains(answer_text: str, gold_item: dict) -> bool:
    """答案文本里是否含 gold 答案（含 `acceptable_answers` 展开）。

    这一条是「正确覆盖」的分母来源。为什么必须与「覆盖」分开报：
    实测整行引用让覆盖 63→82（+19 题），但答案质量 122→122（零收益），
    19 题里 5 题是「引用修好了、答案还是错的」。只报覆盖会把口径修正
    当成能力提升。
    """
    text = str(answer_text or "")
    if not text:
        return False
    cands = [str(gold_item.get("answer") or "")]
    for extra in gold_item.get("acceptable_answers") or []:
        cands.append(str(extra or ""))
    return any(c and c in text for c in cands)


def collect_facts(gold_path: Path, *, limit: int, title_path: bool) -> tuple[dict, list]:
    """跑一遍评测，同时把**逐题明细**留下。

    为什么要自己留一遍明细、而不是读 `ask.py` 的 `rows`
    --------------------------------------------------
    `ask.py --evaluate` 的 `rows` 只保留三个字段级摘要：`citations` 是**条数
    （int）**、`answer` 是**前 120 字**、上下文单元根本没留。于是：

    * 判不了「覆盖建立 **且** 答案含 gold」（需要完整答案）；
    * 判不了 `ungated` 那个分母（需要上下文里都有哪些块）；
    * 连 `covered` 都只能从 `metrics.required_citation_coverage.covered` 读到一个总数，
      没法逐题核对。

    本函数用 monkeypatch 把 `ask.ask_once` 换成「先调真函数、再把结果存一份」的
    包装器——`ask.evaluate` 内部解析的正是模块全局名 `ask_once`，所以包装器会被
    真正调用到（`scripts/bench_h1.py` 早就用同一手法替换 `ask.load_generation`）。
    返回 `(report, facts)`，`facts[i]` 与 `report["rows"][i]` 一一对应。
    """
    import ask as ask_mod  # noqa: PLC0415 — 评测脚本在 scripts/ 下，延迟导入
    from arkrag import pipeline as pipeline_mod  # noqa: PLC0415

    captured: list[dict] = []
    original = ask_mod.ask_once

    def recorder(pipe, question, **kwargs):
        result = pipeline_mod.ask_once(pipe, question, **kwargs)
        captured.append(result)
        return result

    answer.TITLE_ANSWER_PATH = bool(title_path)
    ask_mod.ask_once = recorder
    try:
        report = ask_mod.evaluate(gold_path, limit=limit)
    finally:
        ask_mod.ask_once = original
        answer.TITLE_ANSWER_PATH = False

    facts = []
    for result in captured:
        ans = result.get("answer") or {}
        ctx = result.get("context") or {}
        facts.append({
            "citations": [
                {"chunk_id": str(c.get("chunk_id") or ""),
                 "evidence": str(c.get("evidence") or ""),
                 "start": int(c.get("start", -1)), "end": int(c.get("end", -1))}
                for c in ans.get("citations") or []
            ],
            "answer": str(ans.get("answer") or ""),
            "abstained": bool(ans.get("abstained")),
            "context_ids": [str(u.get("chunk_id") or "")
                            for u in ctx.get("units") or []],
        })
    return report, facts


def selected_ids(gold_path: Path, *, limit: int) -> list:
    """复现 `ask.evaluate` 抽样的题号顺序，用来对齐 `facts`。

    必须调 `ask._stratified_sample` 而不是自己写抽样：抽样规则只许有一份定义。
    """
    import ask as ask_mod  # noqa: PLC0415

    doc = json.loads(gold_path.read_text(encoding="utf-8"))
    items = doc.get("items") if isinstance(doc, dict) else doc
    chosen = ask_mod._stratified_sample(items or [], limit)
    return [str(it.get("question_id") or "") for it in chosen]


def load_assertions_by_chunk(generation_id: str) -> dict:
    """按块分组的断言，用于判 `reachable`（断言层能不能支撑那条证据）。"""
    from arkrag import release  # noqa: PLC0415

    gen = release.Generation(generation_id)
    by_chunk: dict[str, list[dict]] = {}
    with gen.assertion_path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            by_chunk.setdefault(str(rec.get("chunk_id") or ""), []).append(rec)
    return by_chunk


def coverage_facts(facts: list, ids: list, gold: dict, by_chunk: dict) -> dict:
    """算出引用覆盖的四件事：三个分母 + 正确覆盖。

    三个分母的包含关系（`provenance._judge_coverage_denominators` 会核这条）：
        `ungated ⊆ reachable ⊆ mixed`
    * `mixed`：全部应答题（混合了三种失败原因，最容易看起来很低）；
    * `reachable`：required 的每条 evidence 都能在 required 块某条断言的
      `evidence_text` 里找到（**断言层**支撑得住）；
    * `ungated`：在 `reachable` 之上，再要求 required 的块**全部进了上下文**
      （两道上游闸都通过，只剩「发不发引用」这一层 —— P3 该盯的就是它）。

    `correct` 是「覆盖建立 **且** 答案里真的含 gold 答案」的题数。为什么要它：
    实测整行引用让覆盖 63→82（+19 题）而正确答案 122→122（零收益），
    19 题里 5 题是「引用修好了、答案还是错的」——只报覆盖会把口径修正
    当成能力提升。
    """
    mixed = 0
    reachable = 0
    ungated = 0
    covered = 0
    correct = 0
    for i, qid in enumerate(ids):
        item = gold.get(qid) or {}
        if item.get("should_abstain"):
            continue
        required = item.get("required_citations") or []
        if not required:
            continue
        mixed += 1
        fact = facts[i] if i < len(facts) else {}
        emitted = [(c["chunk_id"], c["evidence"]) for c in fact.get("citations") or []]
        is_covered = all(
            any(str(rc.get("chunk_id") or "") == cid and str(rc.get("evidence") or "") in ev
                for cid, ev in emitted)
            for rc in required
        )
        # 下面两个闸与「发没发引用」无关，纯看断言与上下文，所以即使没覆盖也要算。
        reachable_ok = all(
            any(str(rc.get("evidence") or "") in str(rec.get("evidence_text") or "")
                for rec in by_chunk.get(str(rc.get("chunk_id") or "")) or [])
            for rc in required
        )
        ctx = set(fact.get("context_ids") or [])
        ungated_ok = reachable_ok and all(
            str(rc.get("chunk_id") or "") in ctx for rc in required
        )
        if reachable_ok:
            reachable += 1
        if ungated_ok:
            ungated += 1
        if is_covered:
            covered += 1
            if _answer_contains(fact.get("answer"), item):
                correct += 1
    return {
        "denominators": {"mixed": mixed, "reachable": reachable, "ungated": ungated},
        "covered": covered,
        "correct_coverage": correct,
        "covered_without_correct": covered - correct,
    }


def load_gold_items(gold_path: Path) -> dict:
    """gold 题按 `question_id` 索引。不存在时返回空 dict，由调用方判缺口。"""
    try:
        doc = json.loads(gold_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    items = doc.get("items") if isinstance(doc, dict) else doc
    return {str(it.get("question_id") or ""): it for it in (items or [])}


def snapshot(report: dict, *, gold_path: Path, title_path: bool,
             elapsed_s: float, cpu_s: float | None,
             cov: dict | None = None) -> dict:
    metrics = report.get("metrics") or {}
    rows = report.get("rows") or []

    covered = int(_metric_block(metrics, "required_citation_coverage").get("n") and
                  (metrics.get("required_citation_coverage") or {}).get("covered") or 0)
    n_answerable = int(_metric_block(metrics, "required_citation_coverage").get("n") or 0)

    # 下游量：报告里唯一同时反映「块进了上下文」与「引用发出去了」的逐题字段是
    # citations 条数与非拒答题数。R@n 单独一行不构成「更好」的证据，所以必须并列。
    #
    # `rows[i]["citations"]` 是**条数（int）**，不是列表——与 `data/goldset/gold-v1.json`
    # 里同名的 `required_citations`（那是列表）形状不同。本项目已在
    # `data/a_triage_all320.jsonl` 的 `required_citations` 上栽过一次同样的坑
    # （把它当列表遍历，报 `TypeError: 'int' object is not iterable`）。
    # 这里两种形状都接住，取不到就记 None 而不是静默当 0。
    def _cite_count(row: dict) -> int:
        raw = row.get("citations")
        if isinstance(raw, list):
            return len(raw)
        if isinstance(raw, int):
            return raw
        return 0

    downstream = {
        "citations_total": sum(_cite_count(r) for r in rows),
        "answered": sum(1 for r in rows if not r.get("abstained")),
        "abstained": sum(1 for r in rows if r.get("abstained")),
    }

    # dataset / inputs 两块的形状**由 provenance 自己拼**，不在这里手搓：
    # 手搓过一次，漏了 `gold_sha16` 与 `bytes`，`verify()` 直接判 unknown。
    # 那次教训是本项目的第三次「两个合法数据源选错一个」的邻居形态——
    # 「形状看起来对、字段名不对」，肉眼很难看出来。
    dataset = provenance.dataset_block(
        gold_path=gold_path,
        gold_total=int(report.get("items_total_in_file") or 0),
    )
    # 切分文件只有 gold-v2 有；gold-v1 跑评测时它不参与，但记上去不损伤可追溯性，
    # 且能让读者看出「这次跑的是没切分的 v1」。文件不在盘上时 sha16 为空。
    inputs = provenance.inputs_block([gold_path])

    blk = provenance.collect(
        generation_id=str(report.get("generation_id") or ""),
        dataset=dataset,
        inputs=inputs,
    )

    row = {
        "generated_at": blk.get("generated_at"),
        "generation_id": blk.get("generation_id"),
        "git_commit": blk.get("git_commit"),
        "code_sha256_16": blk.get("code_sha256_16"),
        "dataset": blk.get("dataset"),
        "inputs": blk.get("inputs"),
        "environment": blk.get("environment"),
        # 口径：本快照只认 gold_chunk_ids，等价集合口径尚未进主评测。
        "caliber": "gold_only",
        # 断崖：报前 5 名指标必须附这个实测值。
        "rank5_conversion": provenance.RANK5_CONVERSION,
        # 引用覆盖三个分母 + 正确覆盖。`reachable`/`ungated` 由本脚本另跑一遍
        # 上下文与断言算出来（`ask.py` 的 rows 不保留上下文单元，读不出来）。
        # 算不出来时给 `None` 让 `provenance` 判 unknown —— 不编数。
        "denominators": (cov or {}).get("denominators") or {"mixed": n_answerable},
        "covered_n": (cov or {}).get("covered", covered),
        "correct_coverage": (cov or {}).get("correct_coverage"),
        "recall_at_20": _metric_block(metrics, "recall_at_20"),
        "recall_at_5": _metric_block(metrics, "recall_at_5"),
        "mrr": _metric_block(metrics, "mrr"),
        "citation_location_accuracy": _metric_block(metrics, "citation_location_accuracy"),
        "required_citation_coverage": _metric_block(metrics, "required_citation_coverage"),
        "abstain_accuracy": _metric_block(metrics, "abstain_accuracy"),
        "direction_accuracy": _metric_block(metrics, "direction_accuracy"),
        "citation_verify_failures": len(report.get("citation_verify_failures") or []),
        "downstream": downstream,
        "elapsed_s": round(elapsed_s, 3),
        "cpu_s": None if cpu_s is None else round(cpu_s, 3),
        "switch": {"TITLE_ANSWER_PATH": title_path, "CITATION_WHOLE_LINE":
                   bool(getattr(answer, "CITATION_WHOLE_LINE", False))},
        "items_evaluated": int(report.get("items_evaluated") or 0),
    }
    return row


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(description="把一次正式评测落成可追溯快照")
    ap.add_argument("--gold-set", default="data/goldset/gold-v1.json")
    ap.add_argument("--limit", type=int, default=320)
    ap.add_argument("--title-path", choices=("on", "off"), default="off",
                    help="answer.TITLE_ANSWER_PATH 的取值；两个值各跑一次才能对照")
    ap.add_argument("--out", required=True, help="快照写到这个路径（JSON）")
    args = ap.parse_args(argv)

    gold_path = Path(args.gold_set)
    if not gold_path.is_absolute():
        gold_path = ROOT / gold_path
    if not gold_path.exists():
        print(f"[错误] gold set 不存在：{gold_path}", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    wall0 = time.process_time()
    report, facts = collect_facts(gold_path, limit=args.limit,
                                  title_path=(args.title_path == "on"))
    elapsed = time.perf_counter() - t0
    cpu = time.process_time() - wall0

    # 覆盖四件事要另算：`ask.py` 的 rows 不保留上下文与完整答案。
    gold = load_gold_items(gold_path)
    ids = selected_ids(gold_path, limit=args.limit)
    cov = None
    if len(ids) == len(facts) and report.get("generation_id"):
        try:
            by_chunk = load_assertions_by_chunk(str(report["generation_id"]))
        except (OSError, SystemExit) as exc:  # 断言读不到就不编数，判 unknown
            print(f"[注意] 断言读不到，reachable/ungated 留空：{exc}", file=sys.stderr)
        else:
            cov = coverage_facts(facts, ids, gold, by_chunk)
    if cov is None:
        print("[注意] 覆盖三分母与正确覆盖未算出，指标行会判 unknown", file=sys.stderr)

    row = snapshot(report, gold_path=gold_path, title_path=(args.title_path == "on"),
                   elapsed_s=elapsed, cpu_s=cpu, cov=cov)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "arkrag-eval-snapshot/1",
        "metrics": [row],
        "rows": report.get("rows") or [],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    # ---- 自检：快照必须能被 provenance 的两套判据读懂 ----
    verdicts = {}
    r1 = provenance.verify({k: row.get(k) for k in provenance.REQUIRED_FIELDS})
    verdicts["seven_fields"] = r1["verdict"]
    r2 = provenance.verify_metric_row(row)
    verdicts["metric_row"] = r2["verdict"]

    print(f"世代 {row['generation_id']}  题集 {args.gold_set}  评测 {row['items_evaluated']} 条")
    for key in ("recall_at_20", "recall_at_5", "mrr", "citation_location_accuracy",
                "required_citation_coverage", "abstain_accuracy", "direction_accuracy"):
        b = row[key]
        print(f"  {key:28s} {b['value']}  n={b['n']} skipped={b['skipped']}")
    print(f"  citation_verify_failures     {row['citation_verify_failures']}")
    print(f"  下游量                       {row['downstream']}")
    print(f"  TITLE_ANSWER_PATH            {row['switch']['TITLE_ANSWER_PATH']}")
    print(f"  耗时                         墙钟 {row['elapsed_s']}s / CPU {row['cpu_s']}s")
    print(f"七字段 {verdicts['seven_fields']}；指标行 {verdicts['metric_row']}")
    for name, res in (("seven_fields", r1), ("metric_row", r2)):
        if res["verdict"] != provenance.PASS:
            print(f"  [注意] {name} 判 {res['verdict']}：{res.get('detail') or res.get('missing')}")
    print(f"快照 -> {config.rel_to_root(out_path)}")

    # 判据：七字段必须 pass；指标行允许 unknown（三分母缺两个是真的缺），
    # 但不许 fail —— fail 说明写错了，那是本脚本的 bug 不是数据缺口。
    if verdicts["seven_fields"] != provenance.PASS:
        return 4
    if verdicts["metric_row"] == provenance.FAIL:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
