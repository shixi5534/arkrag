# -*- coding: utf-8 -*-
"""校验 outbox 里某几个 part 文件的抽取质量（不写缓存，只读数）。

用法：py -3 scripts/qa_part.py B01.part00.jsonl [更多的 part ...]
读数：行数/顺序、evidence 是否落在本行正文、是否超 40 字、关系是否在封闭表内、
端点是否在同行 entities、confidence 是否越界、实体类型是否六选一。
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from arkrag.vocab import CANONICAL_RELATIONS  # noqa: E402

INBOX = os.path.join(ROOT, "data", "arkrag", "agent_inbox")
OUTBOX = os.path.join(ROOT, "data", "arkrag", "agent_outbox")

TYPES = {"角色", "组织", "地点", "物品", "概念", "事件"}
VOCAB = set(CANONICAL_RELATIONS)


def body_of(user_prompt: str) -> str:
    parts = user_prompt.split("---")
    return parts[1] if len(parts) >= 3 else user_prompt


def check(name: str) -> dict:
    task_path = os.path.join(INBOX, name)
    out_path = os.path.join(OUTBOX, name)
    tasks = [json.loads(l) for l in open(task_path, encoding="utf-8") if l.strip()]
    if not os.path.exists(out_path):
        # 收割器会把完成的文件移到 done/
        archived = os.path.join(OUTBOX, "done", name)
        if os.path.exists(archived):
            out_path = archived
        else:
            return {"part": name, "missing": True}
    raw = [l for l in open(out_path, encoding="utf-8") if l.strip()]
    objs, parse_bad = [], 0
    for l in raw:
        try:
            objs.append(json.loads(l))
        except json.JSONDecodeError:
            parse_bad += 1
    by_id = {}
    for o in objs:
        by_id[str(o.get("chunk_id"))] = o

    stat = {
        "part": name,
        "task_lines": len(tasks),
        "out_lines": len(raw),
        "parse_bad": parse_bad,
        "missing_lines": 0,
        "order_match": [str(o.get("chunk_id")) for o in objs] == [t["chunk_id"] for t in tasks],
        "entities": 0,
        "relations": 0,
        "ev_not_in_body": 0,
        "ev_too_long": 0,
        "off_vocab": [],
        "endpoint_missing": 0,
        "bad_type": 0,
        "bad_conf": 0,
        "bad_examples": [],
    }
    for t in tasks:
        cid = t["chunk_id"]
        body = body_of(t["user_prompt"])
        rec = by_id.get(cid)
        if rec is None:
            stat["missing_lines"] += 1
            continue
        ents = rec.get("entities") or []
        rels = rec.get("relations") or []
        stat["entities"] += len(ents)
        stat["relations"] += len(rels)
        names = {e.get("name") for e in ents}
        for e in ents:
            if e.get("type") not in TYPES:
                stat["bad_type"] += 1
        for r in rels:
            ev = r.get("evidence") or ""
            if ev and ev not in body:
                stat["ev_not_in_body"] += 1
                if len(stat["bad_examples"]) < 12:
                    stat["bad_examples"].append([cid, "evidence", ev])
            if len(ev) > 40:
                stat["ev_too_long"] += 1
            rel = r.get("relation")
            if rel not in VOCAB:
                stat["off_vocab"].append(rel)
            if r.get("source") not in names or r.get("target") not in names:
                stat["endpoint_missing"] += 1
                if len(stat["bad_examples"]) < 12:
                    stat["bad_examples"].append([cid, "endpoint", r.get("source"), r.get("target")])
            c = r.get("confidence")
            if not isinstance(c, (int, float)) or not (0.5 <= c <= 0.95):
                stat["bad_conf"] += 1
    return stat


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(json.dumps({"usage": "py -3 scripts/qa_part.py <part.jsonl> [...]"}, ensure_ascii=False))
        return 2
    total = {"entities": 0, "relations": 0, "blocks": 0,
             "ev_not_in_body": 0, "ev_too_long": 0, "endpoint_missing": 0,
             "off_vocab": 0, "bad_type": 0, "bad_conf": 0, "parse_bad": 0, "missing_lines": 0}
    for name in args:
        s = check(name)
        if s.get("missing"):
            print(json.dumps({"part": name, "missing": True}, ensure_ascii=False))
            continue
        print(json.dumps(s, ensure_ascii=False))
        total["blocks"] += s["out_lines"]
        for k in ("entities", "relations", "ev_not_in_body", "ev_too_long",
                  "endpoint_missing", "bad_type", "bad_conf", "parse_bad", "missing_lines"):
            total[k] += s[k]
        total["off_vocab"] += len(s["off_vocab"])
    print(json.dumps({"TOTAL": total}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
