"""F3：从 `ark-rag-3` 图产物机械构造 Gold Set 第一版，并做可失败的校验。

    py -3 scripts/build_goldset.py                        # 建 320 题并写 data/goldset/gold-v1.json
    py -3 scripts/build_goldset.py --dry-run              # 只打印报告，不写盘
    py -3 scripts/build_goldset.py --total 320 --seed 20261101

为什么要单独一支脚本而不是接进 `build_generation`：Gold Set 是**评测输入**，
它的生命周期与图产物解耦——换了模型/提示词会重跑图，但评测集必须能固定下来做
跨世代对比。所以这里只读世代产物、只写 `data/goldset/`（不在冻结区 `data/arkrag/` 内）。

退出码语义（必须能失败，见计划原则 5）：
    0  全部通过：无校验失败、无重复问句、无缺口（或缺口已在报告里如实记录）
    1  有校验失败项或有重复问句
    2  数据源不可用（世代不存在 / 不是 ark-rag-3 / 缺断言文件）

用法：
    py -3 scripts/build_goldset.py [--generation g-...] [--total N] [--seed N]
                                   [--out path] [--report path] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from arkrag import config, goldset, release  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def _pick_generation(wanted: str) -> Path:
    """定位数据源世代。默认取最新的 `ark-rag-3` 候选世代。

    线上世代是 `ark-rag-1`（聚合边、没有逐观察断言），挖不了「引用可逐字定位」
    的题，所以这里**不退回线上世代**——找不到就直接退出码 2，让人去看构建状态。
    """
    gens_dir = release.GENERATIONS_DIR
    if not gens_dir.exists():
        raise SystemExit(f"世代目录不存在：{gens_dir}")
    if wanted:
        gen = release.Generation(wanted)
        if not gen.dir.exists():
            raise SystemExit(f"世代不存在：{gen.dir}")
        return gen.dir
    candidates: list[Path] = []
    for path in sorted(gens_dir.iterdir()):
        if not path.is_dir():
            continue
        graph_path = path / "graph" / "graph.json"
        if not graph_path.exists():
            continue
        try:
            doc = json.loads(graph_path.read_text("utf-8"))
        except json.JSONDecodeError:
            continue
        if doc.get("schema_version") == "ark-rag-3":
            candidates.append(path)
    if not candidates:
        raise SystemExit("没有可用的 ark-rag-3 世代（需要 graph/assertions.jsonl）")
    latest = candidates[-1]
    if not (latest / "graph" / "assertions.jsonl").exists():
        raise SystemExit(f"世代 {latest.name} 缺 graph/assertions.jsonl")
    return latest


def _load_sources(gen_dir: Path) -> dict:
    """读齐四份输入：图、断言、邻接、语料块。"""
    graph_doc = json.loads((gen_dir / "graph" / "graph.json").read_text("utf-8"))
    assertions = _read_jsonl(gen_dir / "graph" / "assertions.jsonl")
    adjacency = json.loads((gen_dir / "graph" / "adjacency.json").read_text("utf-8"))
    chunks = _read_jsonl(config.chunks_path())
    if not chunks:
        raise SystemExit(f"语料块为空：{config.chunks_path()}")
    return {
        "graph": graph_doc,
        "assertions": assertions,
        "adjacency": adjacency,
        "chunks": chunks,
    }


def _print_report(gold: dict, *, gen_dir: Path, sources: dict, duplicates: list[dict]) -> None:
    rep = gold["report"]
    print(f"数据源世代  {gen_dir.name}")
    print(
        f"输入规模    entities {len(sources['graph'].get('entities') or [])}"
        f" / edges {len(sources['graph'].get('edges') or [])}"
        f" / assertions {len(sources['assertions'])}"
        f" / chunks {len(sources['chunks'])}"
    )
    print(f"\n总题数      {rep['total_actual']} / 请求 {rep['total_requested']}")
    print(f"{'类别':<18}{'配额':>6}{'实得':>6}{'缺口':>6}{'实际占比':>10}")
    for cat in goldset.CATEGORIES:
        q = rep["quotas"][cat]
        p = rep["produced"][cat]
        gap = rep["gaps"].get(cat, 0)
        share = rep["share_actual"][cat]
        print(f"{cat:<18}{q:>6}{p:>6}{gap:>6}{share:>10.4f}")
    gaps = {c: v for c, v in rep["gaps"].items() if v}
    print(f"\n缺口        {gaps if gaps else '无'}")
    print(f"校验失败    {len(rep['validation_failures'])}")
    for f in rep["validation_failures"][:20]:
        print(f"  {f['question_id']}: {f['errors']}")
    if len(rep["validation_failures"]) > 20:
        print(f"  ...（共 {len(rep['validation_failures'])} 条，只列前 20）")
    print(f"重复问句    {len(duplicates)}")
    for d in duplicates[:10]:
        print(f"  {d['first']} == {d['again']}: {d['question'][:60]}")
    ab = Counter(bool(it.get("should_abstain")) for it in gold["items"])
    print(f"应拒答题    {ab.get(True, 0)} / 可答题 {ab.get(False, 0)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="构造 F3 Gold Set 第一版")
    ap.add_argument("--generation", default="", help="指定世代 id（默认取最新 ark-rag-3）")
    ap.add_argument("--total", type=int, default=320, help="目标题数（默认 320）")
    ap.add_argument("--seed", type=int, default=20261101, help="随机种子（默认 20261101）")
    ap.add_argument("--out", default="", help="输出 JSON 路径（默认 data/goldset/gold-v1.json）")
    ap.add_argument("--report", default="", help="把报告另存为 JSON 到这个路径")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写盘")
    args = ap.parse_args()

    gen_dir = _pick_generation(args.generation)
    sources = _load_sources(gen_dir)
    graph_doc = sources["graph"]

    gold = goldset.build_gold_set(
        sources["assertions"],
        sources["chunks"],
        graph_doc.get("edges") or [],
        graph_doc.get("entities") or [],
        sources["adjacency"],
        total=args.total,
        seed=args.seed,
    )

    duplicates = gold["report"]["duplicate_questions"]
    _print_report(gold, gen_dir=gen_dir, sources=sources, duplicates=duplicates)

    if not args.dry_run:
        out_path = Path(args.out) if args.out else goldset.save_gold_set(gold)
        if args.out:
            out_path = goldset.save_gold_set(gold, out_path)
        print(f"\n已写出      {out_path}")
        if args.report:
            rep_path = Path(args.report)
            rep_path.parent.mkdir(parents=True, exist_ok=True)
            rep_path.write_text(
                json.dumps(
                    {
                        "generation": gen_dir.name,
                        "schema": gold["schema"],
                        "seed": gold["seed"],
                        "quotas": gold["quotas"],
                        **gold["report"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"报告已写    {rep_path}")
    else:
        print("\n(dry-run：未写盘)")

    bad = bool(gold["report"]["validation_failures"]) or bool(duplicates)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
