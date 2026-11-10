"""J2：逐题重判历史逆序对 + cap=4 造成的 5 个翻转。

## 这个脚本回答什么

计划书 §W4 说：

> 实测 345 个配对里有 **5 个逆序对**：`q-cx-000011`（P=1.00, R=0.75, F1=0.857）
> 被判 **FAIL**，`q-cx-900027`（P=0.50, R=1.00, F1=0.667）被判 **PASS**。
> F1 本身满足通常的 P/R 序关系，问题在于 v1 的集合包含与额外项规则并不等价于 F1 阈值。

J2 的要求是：**对这 5 个逆序对逐题给出新判定**，不声称原始 F1 被改写成单调函数。

## 两件事必须分开

1. **计划书说的"5 个逆序对"** 是在某个更早的评测快照上算的（F1 序 vs 判据序）。
2. **`chapter_cap=4` 本身也造成 5 个翻转**（4 题由 FAIL→PASS，1 题 PASS→FAIL）。

这两组"5"**不是同一件事**，混起来会得出矛盾的结论。本脚本分别报。

## 产出

`data/j2_inversions.json`：每题给 v1 判定、v2 判定、gold/输出章节集、
以及"cap 前后"的对比，逐题标注它属于哪一类。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OLD_PROD = ROOT / "tmp_prod_dev.json"
NEW_PROD = ROOT / "data" / "w3_evidence" / "cap4_dev.json"
BOTH = ROOT / "data" / "w4_evidence" / "judge_both_dev.json"
OUT = ROOT / "data" / "j2_inversions.json"


def load(p):
    return json.load(open(p, encoding='utf-8'))


def rows_of(d):
    return {r['question_id']: r for r in d.get('rows') or []}


def main():
    old = rows_of(load(OLD_PROD))
    new = rows_of(load(NEW_PROD))
    both = rows_of(load(BOTH))

    # ---- A 组：计划书点名的逆序对（有 F1 数字的两道）----
    named = ['q-cx-000011', 'q-cx-900027']

    # ---- B 组：cap=4 造成的全部翻转 ----
    cap_flips = []
    for qid, ro in old.items():
        rn = new.get(qid)
        if not rn:
            continue
        po = ro['arms']['prod']['pass']
        pn = rn['arms']['prod']['pass']
        if po != pn:
            cap_flips.append(qid)

    out = {
        "schema": "arkrag.j2_inversions/1",
        "purpose": "J2：逐题重判历史逆序对 + cap=4 造成的翻转（两组必须分开）",
        "warning": (
            "计划书的『5 个逆序对』是按 F1 序 vs 判据序算的；"
            "cap=4 造成的翻转也是 5 个。两组数字相同但含义完全不同，不可混用。"
        ),
        "group_A_named_inversions": {},
        "group_B_cap_flips": {},
    }

    for qid in named:
        ro = old.get(qid)
        rn = new.get(qid)
        rb = both.get(qid)
        entry = {
            "question_id": qid,
            "category": (ro or rn or {}).get('category'),
            "question": (ro or {}).get('question'),
            "old_prod_pass": (ro or {}).get('arms', {}).get('prod', {}).get('pass'),
            "old_prod_reason": (ro or {}).get('arms', {}).get('prod', {}).get('reason'),
            "old_answer": (ro or {}).get('answer'),
            "new_prod_pass": (rn or {}).get('arms', {}).get('prod', {}).get('pass'),
            "new_prod_reason": (rn or {}).get('arms', {}).get('prod', {}).get('reason'),
            "new_answer": (rn or {}).get('answer'),
            "v1_pass_current": (rb or {}).get('judge', {}).get('pass'),
            "v1_reason_current": (rb or {}).get('judge', {}).get('reason'),
            "v2_pass": (rb or {}).get('judge_v2', {}).get('pass'),
            "v2_reason": (rb or {}).get('judge_v2', {}).get('reason'),
            "v2_axes": (rb or {}).get('judge_v2', {}).get('axes'),
            "v2_loss": (rb or {}).get('judge_v2', {}).get('loss'),
        }
        out["group_A_named_inversions"][qid] = entry

    for qid in cap_flips:
        ro = old.get(qid)
        rn = new.get(qid)
        rb = both.get(qid)
        out["group_B_cap_flips"][qid] = {
            "question_id": qid,
            "category": ro.get('category'),
            "old_pass": ro['arms']['prod']['pass'],
            "old_reason": ro['arms']['prod']['reason'],
            "old_answer": ro.get('answer'),
            "new_pass": rn['arms']['prod']['pass'],
            "new_reason": rn['arms']['prod']['reason'],
            "new_answer": rn.get('answer'),
            "v2_pass": (rb or {}).get('judge_v2', {}).get('pass'),
            "v2_reason": (rb or {}).get('judge_v2', {}).get('reason'),
            "direction": "FAIL->PASS" if not ro['arms']['prod']['pass'] else "PASS->FAIL",
        }

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n",
                   encoding='utf-8')
    print(f"-> {OUT}")
    print()
    print("A 组（计划书点名）:")
    for qid, e in out["group_A_named_inversions"].items():
        print(f"  {qid}: 旧={e['old_prod_pass']} 新={e['new_prod_pass']}"
              f" | v2={e['v2_pass']} ({e['v2_reason']})")
    print()
    print("B 组（cap=4 造成的翻转）:")
    for qid, e in out["group_B_cap_flips"].items():
        print(f"  {qid}: {e['direction']}  v2={e['v2_pass']}")


if __name__ == '__main__':
    main()
