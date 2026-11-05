# -*- coding: utf-8 -*-
"""审核「方向题正确率」这个指标到底测什么（用户六大目标之一：方向题 ≥95%）。

背景
----
`scripts/ask.py:305-311` 里承担「方向题」的指标是这样算的：

    if category in ("direct_relation", "multihop") and not should_abstain:
        ok = (not abstained) and gold_answer and gold_answer in str(ans["answer"])
        direction_ok.append(bool(ok))

它只做一件事：**金标答案字符串是否作为子串出现在答案里**。没有主体、没有客体、
没有谓词绑定、没有主客顺序比较。本脚本回答三个问题：

1. **恒等检验**：`direction_accuracy` 在它的两个题类上，是否与 `answer_hits_gold`
   逐题恒等（恒等 ⇒ 该指标零新增信息，只是同一个量的第二次计数）。
2. **严判据对照（direct_relation）**：改用「答案里必须有一条三元组，谓词=金标谓词；
   对称关系免顺序，非对称关系要求 主体=金标主体 ∧ 客体=金标客体」重算，多少题被降级、
   降级原因是什么（主体错 / 方向反 / 只是关系词同名）。
3. **multihop 的结构性说明**：该类的金标答案是**单一中间实体名**（如「阿米娅」），
   答案文本本身不含方向信息 ⇒ 「方向」在这一类上根本没有可测量的对象。

判据来源（都是项目自己的）
--------------------------
- 金标主体/客体：`gold_assertion_ids` 唯一时，取该断言的 `subject_id` / `object_id`，
  再查 `graph.entities[].id -> name`。
- 对称关系：`arkrag.vocab.SYMMETRIC_RELATIONS`（9 个）。
- 关系词表：`arkrag.vocab.CANONICAL_RELATIONS`（48 个），用于从答案里切三元组。

边界
----
- 只读 dev（`split == "dev"`）应答题；验收集 260 一题不跑。
- 三元组靠关系词切分，实体名里若含关系词会切错；切不出来的题记为「无三元组」，不猜。
- 本脚本不改任何代码、不改 gold、只读。
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from arkrag import subproc, vocab  # noqa: E402

DEFAULT_GENERATION = "g-20261103T132152Z-06a741"
GOLD = ROOT / "data" / "goldset" / "gold-v2.json"
ROUND = ROOT / "data" / "a_p3_chapter_round.json"

QUOTE = re.compile(r"「([^」]+)」")


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
            env=subproc.child_env(),
        )
    except Exception:  # noqa: BLE001 - 溯源字段取不到就留空，由 verify 判 unknown
        return ""
    return (out.stdout or "").strip() if out.returncode == 0 else ""


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def split_triples(answer: str) -> list[tuple[str, str, str]]:
    """把 `S谓词O。S谓词O。` 形式的答案切成三元组；切不出就跳过该段。"""
    out: list[tuple[str, str, str]] = []
    for seg in answer.replace("；", "。").split("。"):
        seg = seg.strip()
        if not seg:
            continue
        for pred in vocab.CANONICAL_RELATIONS:
            i = seg.find(pred)
            if 0 < i and i + len(pred) < len(seg):
                out.append((seg[:i], pred, seg[i + len(pred):]))
                break
    return out


def direction_ok_legacy(r: dict, gold_answer: str) -> bool:
    """逐字复刻 scripts/ask.py:305-311。"""
    return bool(
        (not r["abstained"])
        and gold_answer
        and gold_answer in str(r["answer"] or "")
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generation", default=DEFAULT_GENERATION)
    ap.add_argument("--json", dest="json_out", default="")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not GOLD.exists() or not ROUND.exists():
        raise SystemExit(f"输入不存在：{GOLD} / {ROUND}")

    gold = {it["question_id"]: it
            for it in json.loads(GOLD.read_text(encoding="utf-8"))["items"]}
    rnd = json.loads(ROUND.read_text(encoding="utf-8"))
    base = {r["question_id"]: r for r in rnd["per_question"]["base"]}

    gen_dir = ROOT / "data" / "arkrag" / "generations" / args.generation
    graph_path = gen_dir / "graph" / "graph.json"
    assert_path = gen_dir / "graph" / "assertions.jsonl"
    if not graph_path.exists() or not assert_path.exists():
        raise SystemExit(f"世代自带图文件不存在：{graph_path} / {assert_path}")
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    names = {e["id"]: e["name"] for e in graph["entities"]}
    by_name: dict[str, str] = {}
    for e in graph["entities"]:
        by_name.setdefault(e["name"], e["id"])
    asserts = {a["assertion_id"]: a for a in load_jsonl(assert_path)}
    symmetric = set(vocab.SYMMETRIC_RELATIONS)

    # ---------- 1. 恒等检验 ----------
    identity = {}
    for cat in ("direct_relation", "multihop"):
        sel = [(q, it) for q, it in gold.items()
               if it.get("split") == "dev" and not it.get("should_abstain")
               and it["category"] == cat]
        same, diff = 0, []
        for q, it in sel:
            r = base[q]
            old = direction_ok_legacy(r, str(it.get("answer") or ""))
            if old == bool(r["answer_hits_gold"]):
                same += 1
            else:
                diff.append({"question_id": q,
                             "direction_ok": old,
                             "answer_hits_gold": bool(r["answer_hits_gold"])})
        identity[cat] = {"n": len(sel), "一致": same, "不一致": diff}

    # ---------- 2. direct_relation 严判据 ----------
    dr_sel = [(q, it) for q, it in gold.items()
              if it.get("split") == "dev" and not it.get("should_abstain")
              and it["category"] == "direct_relation"]
    cnt = Counter()
    demoted = []
    for q, it in dr_sel:
        r = base[q]
        ans = str(r["answer"] or "")
        pred = str(it.get("answer") or "")
        old = direction_ok_legacy(r, pred)
        cnt[("old", old)] += 1
        aids = it.get("gold_assertion_ids") or []
        if len(aids) != 1 or aids[0] not in asserts:
            cnt["断言不可用"] += 1
            continue
        a = asserts[aids[0]]
        subj = names.get(a["subject_id"], "?")
        obj = names.get(a["object_id"], "?")
        tp = split_triples(ans)
        if not tp:
            cnt["无三元组"] += 1
        if pred in symmetric:
            strict = any(p == pred and {s, o} == {subj, obj} for s, p, o in tp)
        else:
            strict = any(s == subj and p == pred and o == obj for s, p, o in tp)
        strict = strict and not r["abstained"]
        cnt[("strict", strict)] += 1
        if old and not strict:
            if not tp:
                why = "答案里切不出三元组"
            elif any(p == pred and s == obj and o == subj for s, p, o in tp):
                why = "方向写反（主体客体对调）"
            elif any(p == pred and o == obj for s, p, o in tp):
                why = "主体错（客体对）"
            elif any(p == pred and s == subj for s, p, o in tp):
                why = "客体错（主体对）"
            else:
                why = "主体客体都不对：只是「关系词」同名"
            demoted.append({"question_id": q,
                            "金标断言": f"{subj} --{pred}--> {obj}",
                            "对称关系": pred in symmetric,
                            "答案": ans,
                            "降级原因": why})

    # ---------- 3. multihop 结构性说明 ----------
    mh_sel = [(q, it) for q, it in gold.items()
              if it.get("split") == "dev" and not it.get("should_abstain")
              and it["category"] == "multihop"]
    mh = Counter()
    for q, it in mh_sel:
        r = base[q]
        ans = str(r["answer"] or "")
        mh["n"] += 1
        if r["abstained"]:
            mh["拒答"] += 1
        ends = [by_name.get(x) for x in QUOTE.findall(it["question"])]
        if len(ends) == 2 and all(ends):
            common = set()
            for e in graph["edges"]:
                if e.get("subject_id") == ends[0]:
                    common.add(e.get("object_id"))
                if e.get("object_id") == ends[0]:
                    common.add(e.get("subject_id"))
            second = set()
            for e in graph["edges"]:
                if e.get("subject_id") == ends[1]:
                    second.add(e.get("object_id"))
                if e.get("object_id") == ends[1]:
                    second.add(e.get("subject_id"))
            shared = common & second
            named = {by_name[n] for n in by_name if n and n in ans}
            mh["答案里有共同邻居名"] += bool(named & shared)
            mh["答案里没有任何共同邻居名"] += not bool(named & shared)

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_revision(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "command": "py -3 scripts/lead_direction_audit.py",
        "generation": args.generation,
        "gold_set": "data/goldset/gold-v2.json",
        "split": "dev",
        "1_恒等检验": identity,
        "2_direct_relation_严判据": {
            "n": len(dr_sel),
            "旧判据判真": cnt[("old", True)],
            "严判据判真": cnt[("strict", True)],
            "降级题数": len(demoted),
            "断言不可用": cnt["断言不可用"],
            "无三元组": cnt["无三元组"],
            "对称关系集合": sorted(symmetric),
            "逐题降级": demoted,
        },
        "3_multihop_结构性": dict(mh),
    }

    print("== 1. direction_accuracy 与 answer_hits_gold 的恒等检验 ==")
    for cat, s in identity.items():
        print(f"  {cat:<16} n={s['n']:<4} 逐题一致 {s['一致']:<4} 不一致 {len(s['不一致'])}")
    allsame = all(not s["不一致"] for s in identity.values())
    print(f"  ⇒ {'两个判据逐题恒等：direction_accuracy 在该两类题上就是 answer_hits_gold 的第二次计数'
               if allsame else '两者不等，需分别定义'}")

    print("\n== 2. direct_relation 严判据对照（对称关系免顺序） ==")
    s2 = payload["2_direct_relation_严判据"]
    print(f"  n={s2['n']}  旧判据判真 {s2['旧判据判真']}（{s2['旧判据判真']/s2['n']:.4f}）"
          f"  严判据判真 {s2['严判据判真']}（{s2['严判据判真']/s2['n']:.4f}）")
    print(f"  降级 {s2['降级题数']} 题；断言不可用 {s2['断言不可用']}；无三元组 {s2['无三元组']}")
    for d in demoted:
        print(f"  - {d['question_id']}  {d['金标断言']}  对称={d['对称关系']}")
        print(f"      {d['降级原因']}")
        print(f"      ans={d['答案'][:68]!r}")

    print("\n== 3. multihop：答案里没有方向可言 ==")
    print(f"  n={mh['n']}  拒答 {mh['拒答']}"
          f"  答案里出现共同邻居名 {mh['答案里有共同邻居名']}"
          f"  没有 {mh['答案里没有任何共同邻居名']}")
    print("  ⇒ 该类金标答案是单一中间实体名，答案文本不含方向信息，")
    print("    「方向题正确率」在这 57 题上不存在可测量的对象。")

    if args.json_out:
        out = Path(args.json_out)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                       encoding="utf-8", newline="\n")
        print(f"\n写出 {out}（{out.stat().st_size} B）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
