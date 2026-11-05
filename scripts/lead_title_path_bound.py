#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""标题通路上界复算：把「76 道标题题 / 61 可定位 / 干净 22」这组数变成一条命令。

为什么单独写这个脚本
--------------------
A 交的 `data/a_title_impl_feasible.json` 与 `data/a_title_locator_check.json` 已经把
逐题判定写下来了，但有两件事只有 Lead 这边算了：

1. **引文在「整个语料」里能不能唯一定位**（A 只算了「上下文里的块」）。
   这两件事不是一回事：上下文只装 5 个单元，语料有 13,798 块。
   实测结论：76/76 在语料级唯一（零命中 0、title 歧义 0）。
   => 「61 命中全对」不是运气，是这类题的引文本身在语料里唯一。
   报告里必须写「引文语料级唯一 76/76」，不能写「61/61 零误判」——后者像赌对了 61 次。

2. **「干净 22 / 泄露 39」与「模板 B ∩ 可行 / 模板 A ∩ 可行」是否严格等价**。
   实测严格等价。这条等价关系比数字本身更有用：它说明「泄露」不是一个需要人工判的
   模糊标签，而是**问句骨架决定的**——模板 A 的 `《T》` 里就是答案，模板 B 的答案不在问句里。

3. **15 道不可行题的归因**：它们在语料里能找到，是 **gold 块没进上下文**（闸门2）。
   报告若不写这条，会被读成「定位器有 15 道失灵」，那是给 P2 的活记到 P3 头上。

口径与限制
----------
* 块正文取自**世代自带** `chunks.jsonl`（`resolve_chunks()` 的语义），不是全局
  `data/arkrag/chunks.jsonl`——两者不是同一批块（代内 13,798 / 语料根 13,786）。
* 「答案是否在问句里」用整串子串判定（`answer in question`），不做同义归一：
  这是**泄露的下界**（更保守地标为泄露），因此「干净」是上界口径。
* 只读 `gold-v2.json` 的 `dev` 且 `should_abstain=False` 且 `gold_chunk_ids` 非空的题。
  验收集（holdout 260）**一行不读**。
* 本脚本不做任何真跑、不改任何代码，纯结构量。

用法
----
    py -3 scripts/lead_title_path_bound.py
    py -3 scripts/lead_title_path_bound.py --json data/lead_title_path_bound.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from arkrag import config, subproc  # noqa: E402

DEFAULT_GENERATION = "g-20261103T132152Z-06a741"
GOLD = "data/goldset/gold-v2.json"
TRIAGE = "data/a_p3_line_citation.json"

# 触发词必须与 `answer.py` 里将要写入的 TITLE_TRIGGERS 逐字一致。
# 刻意收窄：不加「引文」「原文」等泛词，否则含「」的非标题题会被吃进来。
TITLE_TRIGGERS = (
    "请给出该块的标题",
    "请回答该段的标题",
    "出自哪个文本块",
    "来自哪一段剧情",
)
# 模板 B 的特征串：问句里没有《》，答案不在问句里。
TEMPLATE_B_MARK = "来自哪一段剧情"
QUOTE_RE = re.compile(r"「([^」]{4,})」")


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
            env=subproc.child_env(),
        )
        return (out.stdout or "").strip() or "unknown"
    except Exception as exc:  # pragma: no cover - 溯源失败不该中断分析
        return f"unknown({exc})"


def load_chunks(generation: str) -> dict[str, dict]:
    """读**世代自带**块正文。文件不在就退出，绝不回退到全局文件。"""
    path = config.DATA_DIR / "generations" / generation / "chunks.jsonl"
    if not path.exists():
        raise SystemExit(f"世代自带块文件不存在：{path}（不许回退到全局 chunks.jsonl）")
    chunks: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            chunks[str(rec.get("chunk_id") or "")] = rec
    return chunks


def load_dev_answerable() -> list[dict]:
    gold = json.loads((ROOT / GOLD).read_text(encoding="utf-8"))
    return [
        it
        for it in gold.get("items") or []
        if it.get("split") == "dev"
        and not it.get("should_abstain")
        and it.get("gold_chunk_ids")
    ]


def quote_line(text: str, quotes: list[str]) -> str | None:
    for line in text.splitlines():
        if not line.strip():
            continue
        if all(q in line for q in quotes):
            return line
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", default=GOLD)
    ap.add_argument("--triage", default=TRIAGE)
    ap.add_argument("--generation", default=DEFAULT_GENERATION)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    dev = load_dev_answerable()
    chunks = load_chunks(args.generation)

    triggered = [it for it in dev if any(t in it["question"] for t in TITLE_TRIGGERS)]
    rest = [it for it in dev if it not in triggered]

    # 骨架统计：把「」与《》都折成占位符，看还剩几种问法。
    def skeleton(q: str) -> str:
        s = QUOTE_RE.sub("<Q>", str(q))
        return re.sub(r"《[^》]*》", "<T>", s)

    skel: dict[str, int] = {}
    for it in triggered:
        skel[skeleton(it["question"])] = skel.get(skeleton(it["question"]), 0) + 1

    # 语料级定位：引文在所有块里命中几个、title 是否唯一。
    rows = []
    for it in triggered:
        quotes = QUOTE_RE.findall(str(it["question"]))
        if not quotes:
            rows.append({"question_id": it["question_id"], "n_quotes": 0,
                         "corpus_hit": 0, "unique_title": False, "title": None,
                         "gold": it.get("answer"), "template": "none"})
            continue
        hits = [
            (cid, str(rec.get("title") or ""))
            for cid, rec in chunks.items()
            if all(q in str(rec.get("text") or "") for q in quotes)
        ]
        titles = {t.strip() for _, t in hits if t.strip()}
        rows.append({
            "question_id": it["question_id"],
            "n_quotes": len(quotes),
            "corpus_hit": len(hits),
            "unique_title": len(titles) == 1,
            "title": titles.pop() if len(titles) == 1 else None,
            "gold": it.get("answer"),
            "template": "B" if TEMPLATE_B_MARK in str(it["question"]) else "A",
        })

    locate_any = [r for r in rows if r["corpus_hit"] > 0]
    unique_at_corpus = [r for r in rows if r["corpus_hit"] > 0 and r["unique_title"]]
    zero_hit = [r for r in rows if r["corpus_hit"] == 0]

    # 两层「可行」必须分开报，混起来会把 76 和 61 说成同一个数：
    #   * 语料级可行（corpus_feasible）= 引文在 13,798 块里定位唯一 且 title 恰等于 gold。
    #     这是**定位器的上界**——它假设目标块一定在上下文里。
    #   * 上下文内可行（ctx_feasible）= 语料级可行 ∩ gold 块确实进了上下文（闸门2 已过）。
    #     这是**这一轮能兑现的上界**。两者之差就是被闸门2 拦掉的那批（P2 的活）。
    # title 与 gold 比对只用于核算上界；实现路径不得读取 gold。
    corpus_feasible_ids = {
        r["question_id"] for r in unique_at_corpus
        if str(r["title"] or "") == str(r["gold"] or "")
    }

    # 干净 / 泄露：按「答案是否是问句子串」判，并用模板归属做交叉验证。
    qmap = {it["question_id"]: it for it in dev}

    def split_leak(ids: set[str]) -> tuple[set[str], set[str]]:
        leak, clean = set(), set()
        for qid in ids:
            ans = str(qmap[qid].get("answer") or "").strip()
            if ans and ans in str(qmap[qid]["question"]):
                leak.add(qid)
            else:
                clean.add(qid)
        return leak, clean

    leak_ids, clean_ids = split_leak(corpus_feasible_ids)

    # 真跑产物里的 required_in_context：决定「上下文内可行」这一层。
    base_rows: dict[str, dict] = {}
    triage_path = ROOT / args.triage
    triage_note = "缺失（无法判定上下文内可行，只报语料级上界）"
    if triage_path.exists():
        tri = json.loads(triage_path.read_text(encoding="utf-8"))
        base_rows = {
            r["question_id"]: r for r in (tri.get("per_question") or {}).get("base") or []
        }
        triage_note = "取 per_question.base[*].required_in_context（cooc0.0 基线档）"

    ctx_feasible_ids = {
        qid for qid in corpus_feasible_ids
        if (base_rows.get(qid) or {}).get("required_in_context")
    }
    ctx_leak_ids, ctx_clean_ids = split_leak(ctx_feasible_ids)   # 注意返回顺序是 (leak, clean)

    tmpl_b_ids = {r["question_id"] for r in rows if r["template"] == "B"}
    tmpl_a_ids = {r["question_id"] for r in rows if r["template"] == "A"}

    # 引文行是否含全部 required evidence（决定「不用放宽准入也能过覆盖判据」）。
    line_ok = 0
    line_bad = []
    for qid in sorted(corpus_feasible_ids):
        it = qmap[qid]
        base = str(it["gold_chunk_ids"][0])
        text = str(chunks.get(base, {}).get("text") or "")
        line = quote_line(text, QUOTE_RE.findall(str(it["question"])))
        reqs = [str(rc.get("evidence") or "") for rc in it.get("required_citations") or []]
        if line is not None and all(r in line for r in reqs):
            line_ok += 1
        else:
            line_bad.append(qid)

    # 「语料级可行但没进上下文」的归因：这批才是这一轮救不到的，且原因是闸门2。
    # 注意不要拿「语料级零命中」当分母——实测零命中是 0，那样归因表会全是 0。
    gap_ids = corpus_feasible_ids - ctx_feasible_ids
    attr = {"gap_total": len(gap_ids), "in_context_false": 0, "in_context_true": 0,
            "unknown": 0, "ids": []}
    if base_rows:
        for qid in sorted(gap_ids):
            row = base_rows.get(qid)
            if not row:
                attr["unknown"] += 1
                continue
            if row.get("required_in_context"):
                attr["in_context_true"] += 1
            else:
                attr["in_context_false"] += 1
                attr["ids"].append(qid)
    else:
        attr["unknown"] = len(gap_ids)

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_revision(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "command": "py -3 scripts/lead_title_path_bound.py",
        "generation": args.generation,
        "gold_set": args.gold,
        "1_口径": {
            "split": "dev 且非应拒 且 gold 块非空",
            "触发词": list(TITLE_TRIGGERS),
            "答案在问句里": "整串子串判定，不做同义归一 => 泄露是下界、干净是上界",
            "块正文来源": "世代自带 chunks.jsonl（不回退全局文件）",
        },
        "2_总量": {
            "dev_应答题": len(dev),
            "触发题数": len(triggered),
            "非标题题数": len(rest),
            "非标题题含引文数": sum(1 for it in rest if QUOTE_RE.search(str(it["question"]))),
            "语料块数": len(chunks),
        },
        "3_问句骨架": {"不同骨架数": len(skel), "分布": skel},
        "4_语料级定位": {
            "有命中": len(locate_any),
            "唯一title": len(unique_at_corpus),
            "零命中": len(zero_hit),
            "title歧义": len(locate_any) - len(unique_at_corpus),
        },
        "5_可行集（两层）": {
            "语料级可行": len(corpus_feasible_ids),
            "语料级_干净": len(clean_ids),
            "语料级_含问句泄露": len(leak_ids),
            "上下文内可行": len(ctx_feasible_ids),
            "上下文内_干净": len(ctx_clean_ids),
            "上下文内_含问句泄露": len(ctx_leak_ids),
            "可取上下文来源": triage_note,
        },
        "6_模板交叉": {
            "模板A": len(tmpl_a_ids),
            "模板B": len(tmpl_b_ids),
            "语料级_干净 == 模板B ∩ 可行": clean_ids == (tmpl_b_ids & corpus_feasible_ids),
            "语料级_泄露 == 模板A ∩ 可行": leak_ids == (tmpl_a_ids & corpus_feasible_ids),
            "上下文内_干净 == 模板B ∩ 可行": ctx_clean_ids == (tmpl_b_ids & ctx_feasible_ids),
            "上下文内_泄露 == 模板A ∩ 可行": ctx_leak_ids == (tmpl_a_ids & ctx_feasible_ids),
        },
        "7_引文行含全部required": {
            "满足": line_ok,
            "不满足": len(line_bad),
            "不满足题号": line_bad,
        },
        "8_上下文缺口归因": attr,
        "9_逐题": rows,
        "10_限制": [
            "只读 dev，验收集 260 一行未读",
            "「答案在问句里」用整串子串，未展开 acceptable_answers",
            "语料级唯一只是必要条件：块若没进上下文，通路仍不生效（15 道即是）",
            "可行集用 gold 核算上界，实现不得读取 gold",
        ],
    }

    if args.json:
        dest = ROOT / args.json if not os.path.isabs(args.json) else Path(args.json)
        dest.write_text(
            json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[写出] {dest}")

    t = out["2_总量"]
    print(f"dev 应答题 {t['dev_应答题']} / 触发 {t['触发题数']} / 非标题 {t['非标题题数']}"
          f"（含引文 {t['非标题题含引文数']}）/ 语料块 {t['语料块数']}")
    print(f"问句骨架 {out['3_问句骨架']['不同骨架数']} 种："
          f"{[v for _, v in sorted(skel.items(), key=lambda kv: -kv[1])]}")
    d = out["4_语料级定位"]
    print(f"语料级定位：有命中 {d['有命中']} / 唯一 title {d['唯一title']}"
          f" / 零命中 {d['零命中']} / title 歧义 {d['title歧义']}")
    f = out["5_可行集（两层）"]
    print(f"语料级可行 {f['语料级可行']} = 干净 {f['语料级_干净']} + 泄露 {f['语料级_含问句泄露']}")
    print(f"上下文内可行 {f['上下文内可行']} = 干净 {f['上下文内_干净']} + 泄露 {f['上下文内_含问句泄露']}"
          f"（差 {f['语料级可行'] - f['上下文内可行']} 道被闸门2 拦掉）")
    c = out["6_模板交叉"]
    print(f"模板 A {c['模板A']} / B {c['模板B']}；"
          f"语料级: 干净==B∩可行 {c['语料级_干净 == 模板B ∩ 可行']} / 泄露==A∩可行 {c['语料级_泄露 == 模板A ∩ 可行']}")
    print(f"上下文内: 干净==B∩可行 {c['上下文内_干净 == 模板B ∩ 可行']} / 泄露==A∩可行 {c['上下文内_泄露 == 模板A ∩ 可行']}")
    l = out["7_引文行含全部required"]
    print(f"引文行含全部 required：{l['满足']} 满足 / {l['不满足']} 不满足")
    a = out["8_上下文缺口归因"]
    print(f"上下文缺口 {a['gap_total']} 道：required_in_context=False {a['in_context_false']}"
          f" / True {a['in_context_true']} / 未知 {a['unknown']}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        # 不 reconfigure 的话，往 cp936 控制台写非 GBK 字符会让进程以 1 退出，
        # 而这个 1 与「分析失败」长得一模一样。
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
