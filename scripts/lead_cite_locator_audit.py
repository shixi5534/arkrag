# -*- coding: utf-8 -*-
"""审核「引用定位 ≥98%」这条门限：它是一个构造不变量，测不出「引用指对了地方」。

背景
----
`scripts/ask.py:261-270` 的判据是：

    text[start:end] == cite["evidence"]

而 `start` / `end` / `evidence` **三者是同一条断言记录里一起取出来的**
（`src/arkrag/retrieve.py::_push_span` 写进 `citation`，值来自
`assertions.jsonl` 的 `evidence_text` / `evidence_start` / `evidence_end`）。
也就是说：只要断言表本身自洽，这条判据**恒真**——它不是「引用指对了地方」，而是
「我们发出去的引用与它自己声明的偏移一致」。

本脚本用实测把这件事钉死，并回答真正的风险问题：
1. **恒真检验**：每条引用的 `(start, end, evidence)` 是否都能在该块断言表里找到
   完全相同的三元组（`evidence_start` / `evidence_end` / `evidence_text`）。
   若 100% 命中 ⇒ 该指标是构造不变量。
2. **判别力检验**：把 evidence 换成同一块里的**另一条**断言的文本（保持 start/end
   不动），判据必须变假。这是「门是不是真的能红」的证明。
3. **真正的风险**：这些定位「有效」的引用里，有多少条**指向的块不是本题 required
   的块**（即：引用本身站得住，但与问题无关）。这才是 1.0000 掩盖掉的东西。
4. **能否抓到错**：如果引用指向了错误的块但 evidence 与偏移仍自洽，判据会不会红。
   （答案：不会，这就是为什么要和 required 一起看。）

边界
----
- 只跑 dev 应答题；开关取仓库默认（`TITLE_ANSWER_PATH=False`、`CHAPTER_ANSWER_PATH=False`、
  `CITATION_WHOLE_LINE=False`）。
- 只读，不改代码、不改 gold。
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from arkrag import pipeline, subproc  # noqa: E402

DEFAULT_GENERATION = "g-20261103T132152Z-06a741"
GOLD = ROOT / "data" / "goldset" / "gold-v2.json"


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=20, env=subproc.child_env(),
        )
    except Exception:  # noqa: BLE001
        return ""
    return (out.stdout or "").strip() if out.returncode == 0 else ""


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generation", default=DEFAULT_GENERATION)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", dest="json_out", default="")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not GOLD.exists():
        raise SystemExit(f"gold 不存在：{GOLD}")
    gold = [it for it in json.loads(GOLD.read_text(encoding="utf-8"))["items"]
            if it.get("split") == "dev" and not it.get("should_abstain")]
    if args.limit:
        gold = gold[: args.limit]

    loaded = pipeline.load_generation(args.generation)
    pipe = pipeline.build_pipeline(loaded, top_k=20, graph_depth=1)
    chunks = loaded["chunks"]

    gen_dir = ROOT / "data" / "arkrag" / "generations" / args.generation
    assert_path = gen_dir / "graph" / "assertions.jsonl"
    if not assert_path.exists():
        raise SystemExit(f"世代自带断言文件不存在：{assert_path}")
    # 每块：{(evidence_text, start, end)} 三元组集合，以及 evidence_text -> 其它文本
    triples: dict[str, set[tuple[str, int, int]]] = {}
    texts: dict[str, list[str]] = {}
    for a in load_jsonl(assert_path):
        cid = a["chunk_id"]
        triples.setdefault(cid, set()).add(
            (a["evidence_text"], int(a["evidence_start"]), int(a["evidence_end"])))
        texts.setdefault(cid, []).append(a["evidence_text"])

    cite_total = cite_ok = 0
    triple_hit = triple_miss = 0
    flip_needed = flip_failed = 0          # 判别力：换另一条断言文本，判据应变假
    unrelated_cite = 0                      # 引用指向的块不在 required 里
    related_cite = 0
    per_q = []
    chunk_not_in_asserts = Counter()

    for it in gold:
        qid = it["question_id"]
        res = pipeline.ask_once(pipe, it["question"])
        ans = res["answer"]
        cites = ans.get("citations") or []
        req = it.get("required_citations") or []
        req_chunks = {str(r.get("chunk_id") or "") for r in req}
        req_pairs = {(str(r.get("chunk_id") or ""), str(r.get("evidence") or "")) for r in req}
        n_ok = n_bad = 0
        n_rel = n_unrel = 0
        for c in cites:
            cite_total += 1
            cid = str(c.get("chunk_id") or "")
            ev = str(c.get("evidence") or "")
            start, end = int(c.get("start", -1)), int(c.get("end", -1))
            rec = chunks.get(cid)
            text = str((rec or {}).get("text") or "")
            ok = bool(rec is not None and 0 <= start < end <= len(text) and text[start:end] == ev)
            if ok:
                cite_ok += 1
                n_ok += 1
            else:
                n_bad += 1
            if ok and (ev, start, end) in triples.get(cid, ()):
                triple_hit += 1
            elif ok:
                triple_miss += 1
            # 判别力：同一块里换一条别的断言文本（保持 start/end 不动）
            if ok:
                alts = [t for t in texts.get(cid, ()) if t != ev]
                if alts:
                    alt = alts[0]
                    flip_needed += 1
                    if not (0 <= start < end <= len(text) and text[start:end] == alt):
                        flip_failed += 1
            if not cid:
                chunk_not_in_asserts["空 chunk_id"] += 1
                continue
            if cid in req_chunks:
                related_cite += 1
                n_rel += 1
            else:
                unrelated_cite += 1
                n_unrel += 1
        per_q.append({
            "question_id": qid,
            "category": it["category"],
            "citations": len(cites),
            "cite_ok": n_ok,
            "cite_bad": n_bad,
            "cite_to_required_chunk": n_rel,
            "cite_to_other_chunk": n_unrel,
            "required_n": len(req_pairs),
        })

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_revision(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "command": "py -3 scripts/lead_cite_locator_audit.py",
        "generation": args.generation,
        "gold_set": "data/goldset/gold-v2.json",
        "state": "TITLE_ANSWER_PATH=False / CHAPTER_ANSWER_PATH=False / CITATION_WHOLE_LINE=False",
        "1_恒真检验": {
            "引用条数": cite_total,
            "定位判据通过": cite_ok,
            "定位判据失败": cite_total - cite_ok,
            "三元组可溯源命中": triple_hit,
            "三元组可溯源未命中": triple_miss,
            "溯源命中率": round(triple_hit / cite_total, 6) if cite_total else None,
        },
        "2_判别力检验": {
            "可换文本样本数": flip_needed,
            "换文本后仍判真（应当为 0）": flip_needed - flip_failed,
            "换文本后判假": flip_failed,
            "结论": ("判据能红：同一块换一条断言文本后 text[start:end]!=evidence"
                     if flip_needed and flip_failed == flip_needed else "判据未能变红，需排查"),
        },
        "3_引用指向": {
            "指向 required 块": related_cite,
            "指向非 required 块": unrelated_cite,
            "非 required 占比": round(unrelated_cite / cite_total, 6) if cite_total else None,
        },
        "4_按类": {
            cat: {
                "题数": sum(1 for r in per_q if r["category"] == cat),
                "引用条数": sum(r["citations"] for r in per_q if r["category"] == cat),
                "指向非 required 块": sum(r["cite_to_other_chunk"] for r in per_q if r["category"] == cat),
            }
            for cat in sorted({r["category"] for r in per_q})
        },
        "5_逐题": per_q,
    }

    s1, s2, s3 = payload["1_恒真检验"], payload["2_判别力检验"], payload["3_引用指向"]
    print("== 1. 引用定位判据能不能红？ ==")
    print(f"  引用条数 {s1['引用条数']}  通过 {s1['定位判据通过']}  失败 {s1['定位判据失败']}")
    print(f"  (evidence,start,end) 能在该块断言表里找到原样的：{s1['三元组可溯源命中']}"
          f" / 未命中 {s1['三元组可溯源未命中']}  ⇒ 溯源命中率 {s1['溯源命中率']}")
    print("  ⇒ 三者是同一条断言记录里一起写入的，判据恒真——它是构造自洽，不是「指对了地方」")

    print("\n== 2. 判别力：同一块换一条断言文本，判据必须变假 ==")
    print(f"  可换文本样本 {s2['可换文本样本数']}  换后判假 {s2['换文本后判假']}"
          f"  换后仍判真 {s2['换文本后仍判真（应当为 0）']}")
    print(f"  结论：{s2['结论']}")

    print("\n== 3. 1.0000 掩盖掉的风险：引用指向了哪一块 ==")
    print(f"  指向 required 块 {s3['指向 required 块']}  指向非 required 块 {s3['指向非 required 块']}"
          f"  非 required 占比 {s3['非 required 占比']}")
    print("  ⇒ 这些引用「定位有效、evidence 与偏移自洽」，但与本题 required 无关；")
    print("    引用定位指标一条都抓不到，只有引用覆盖指标能抓。")

    print("\n== 4. 按类：指向非 required 块的引用 ==")
    for cat, t in payload["4_按类"].items():
        print(f"  {cat:<16} 题 {t['题数']:<3} 引用 {t['引用条数']:<5} 非 required {t['指向非 required 块']}")

    if args.json_out:
        p = Path(args.json_out)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                     encoding="utf-8", newline="\n")
        print(f"\n写出 {p}（{p.stat().st_size} B）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
