"""L1 解析与清洗：跑解析 + 写语料元数据（含指纹）+ 出门判定 + 产出人工抽验清单。"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arkrag import config, fingerprint, gates, parse  # noqa: E402


def main() -> int:
    dropped_log: list = []
    scenes, stats = parse.parse_corpus(config.DATA_DIR / "scenes.jsonl", dropped_log=dropped_log)
    stats = Counter(stats)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    (config.DATA_DIR / "l1_stats.json").write_text(
        json.dumps(dict(stats.most_common()), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (config.DATA_DIR / "l1_dropped.json").write_text(
        json.dumps(dropped_log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    scene_dicts = [s.as_dict() for s in scenes]
    corpus_fp = fingerprint.corpus_fingerprint(scene_dicts)
    (config.DATA_DIR / "corpus_meta.json").write_text(
        json.dumps(
            {
                "schema_version": config.SCHEMA_VERSION,
                "corpus_fp": corpus_fp,
                "scenes": len(scene_dicts),
                "turns": sum(len(s["turns"]) for s in scene_dicts),
                "source_dir": str(config.SOURCE_DIR),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    run = gates.l1_gates(scene_dicts, stats)
    gates.write_report(run, config.DATA_DIR / "gates.json", "L1")
    gates.sampling_for_human(scene_dicts, Path("docs") / "人工抽验_L1漏检抽样50.md")

    print(f"corpus_fp = {corpus_fp}")
    print("=== L1 门判定 ===")
    for g in run.gates:
        print(f"  {g.verdict:16s} {g.name:34s} 门线[{g.threshold}] 实测[{g.measured[:70]}]")
        if g.note:
            print(f"      note: {g.note[:180]}")
    fails = [g for g in run.gates if g.verdict == "FAIL"]
    human = [g for g in run.gates if g.verdict == "NEEDS_HUMAN"]
    print(f"FAIL {len(fails)} / NEEDS_HUMAN {len(human)} / 共 {len(run.gates)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
