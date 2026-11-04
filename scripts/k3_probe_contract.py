# -*- coding: utf-8 -*-
"""K3：`contracts.validate_extraction` 会不会保留实体上的 `aliases`？

背景：34 条 n2 缓存记录的 `raw.entities` 里 `aliases` 键一个都没有（0/162），
而 13,801 条 n1 记录里有 13,966 条非空别名。要分清两件事：
  (a) 模型没吐 aliases；
  (b) ingest 的契约校验把 aliases 丢了。
(a) 的原始 outbox 已经不在（`S09.part09.jsonl` 0 字节），无法复核；
(b) 可以直接构造一条带 aliases 的抽取记录喂给校验器，看它吐回来什么。

判据：`aliases` 不在 cleaned 里、`desc` 在、且没有任何提示级告警 —— 三条同时成立才说明
「契约静默丢字段」（结论成立，退出码 0）。任一条不成立就说明结论变了，退出码 1。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from arkrag import contracts  # noqa: E402

TEXT = "阿兰娜：对了，你叫什么？\n矿场的人都叫我莱伊。\n莱伊：但是我还要去咧嘴谷。"

task = {
    "schema_version": contracts.TASK_SCHEMA,
    "run_id": "k3-probe",
    "chunk_id": "activities/probe#000",
    "content_key": "0" * 24,
    "scene_id": "activities/probe",
    "doc_id": "activities/probe",
    "source_fingerprint": "",
    "prompt_sha256": "0" * 64,
    "model": "agent:qoder-subagent",
    "generation": 0,
    "text": TEXT,
}

candidate = {
    "schema_version": contracts.EXTRACTION_SCHEMA,
    "chunk_id": "activities/probe#000",
    "content_key": "0" * 24,
    "scene_id": "activities/probe",
    "entities": [
        # 故意带上 aliases 与 desc —— 看校验器留不留
        {"name": "劫车人", "type": "角色", "aliases": ["莱伊"], "desc": "前往咧嘴谷的人"},
        {"name": "咧嘴谷", "type": "地点"},
    ],
    "relations": [
        {
            "source": "劫车人",
            "relation": "前往",
            "target": "咧嘴谷",
            "evidence": "但是我还要去咧嘴谷。",
            "confidence": 0.85,
        }
    ],
    "generation": 0,
    "model": "agent:qoder-subagent",
}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    cleaned, errors = contracts.validate_extraction(candidate, task, TEXT, known_predicates=None)

    print("输入 entities[0]：")
    print("  " + json.dumps(candidate["entities"][0], ensure_ascii=False))
    print()
    if cleaned is None:
        print("校验失败：", json.dumps(errors, ensure_ascii=False)[:400])
        return 1

    print("校验通过。cleaned entities[0]：")
    print("  " + json.dumps(cleaned["entities"][0], ensure_ascii=False))
    print()
    e0 = cleaned["entities"][0]
    print(f"  保留的键 {sorted(e0.keys())}")
    print(f"  aliases 还在吗 = {'aliases' in e0}")
    print(f"  desc 还在吗    = {'desc' in e0}")
    print()
    warns = [e for e in errors if e.get("level") == "warn"]
    print("提示级告警（不影响录用）：",
          json.dumps(warns, ensure_ascii=False)[:300])

    # 判据：aliases 被丢、desc 保留、且无提示级告警
    lost_aliases = "aliases" not in e0
    kept_desc = "desc" in e0
    silent = not warns
    return 0 if (lost_aliases and kept_desc and silent) else 1


if __name__ == "__main__":
    raise SystemExit(main())
