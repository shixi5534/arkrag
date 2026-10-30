"""L1 解析与清洗：跑解析 + 写语料元数据（含分层指纹）+ 出门判定 + 产出人工抽验清单。"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arkrag import atomic, config, fingerprint, gates, parse, source  # noqa: E402


def main() -> int:
    dropped_log: list = []
    root = config.artifact_root()
    scenes, stats = parse.parse_corpus(config.scenes_path(), dropped_log=dropped_log)
    stats = Counter(stats)
    root.mkdir(parents=True, exist_ok=True)
    atomic.atomic_write_json(config.l1_stats_path(), dict(stats.most_common()))
    atomic.atomic_write_json(config.l1_dropped_path(), dropped_log)

    scene_dicts = [s.as_dict() for s in scenes]
    corpus_fp = fingerprint.corpus_fingerprint(scene_dicts)
    scene_fp = fingerprint.scene_fingerprint(scene_dicts)
    source_status, source_report = source.status_for_build()
    atomic.atomic_write_json(
        config.corpus_meta_path(),
        {
            "schema_version": config.SCHEMA_VERSION,
            "corpus_fp": corpus_fp,
            # 场景层身份单独记：正文没变而标题/章节/node_id/prop 变了时，
            # corpus_fp 依旧不变，只有这一层能看出来（计划 D3）。
            "scene_fp": scene_fp,
            "scenes": len(scene_dicts),
            "turns": sum(len(s["turns"]) for s in scene_dicts),
            "source_dir": str(config.SOURCE_DIR),
            # 来源身份随语料一起记：下游不该靠"再跑一次 run_l0"才知道语料是哪来的（计划 D4）
            "source_status": source_status,
            "source_contract_fp": (source_report.get("contract_fp") or ""),
            "source_tree_digest": (source_report.get("tree_digest") or ""),
        },
    )

    run = gates.l1_gates(scene_dicts, stats)
    gates.write_report(run, config.gates_path(), "L1")
    gates.sampling_for_human(scene_dicts, Path("docs") / "人工抽验_L1漏检抽样50.md")

    print(f"corpus_fp = {corpus_fp}   scene_fp = {scene_fp}   source {source_status}")
    print(f"产物根 = {root}")
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
