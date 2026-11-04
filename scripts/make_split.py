"""P1-a：把 Gold Set 切成「调参集 / 验收集」两半，并冻结切分规则。

    py -3 scripts/make_split.py              # 写 data/goldset/split-v2.json 与 gold-v2.json
    py -3 scripts/make_split.py --dry-run    # 只打印账，不写盘

产出两份文件，职责分开：

* `data/goldset/split-v2.json` —— **规则与归属**。谁在 holdout、谁在 dev、用的什么
  种子、什么分层键、每类多少条。评审看这一份就能复核切分是否可复现。
* `data/goldset/gold-v2.json`   —— **题本身**。v1 的 320 条原样搬过来（只加 `split`
  与 `origin` 两个字段），再追加 320 条全新题。评测脚本只认这一份。

为什么 `gold-v1.json` 一个字节都不动：它是历史对照的锚点，K1 的 `gold_set_fixed`
门用它的 sha256 前 16 位与字节数当冻结值。换题集是「建立新契约」，不是「修改旧
契约」，所以新增文件、旧文件留着仍可跑。

切分规则（本文件是唯一实现处，文档只描述不实现）：

1. 分层键 `(category, should_abstain)`。**不能用 difficulty 分层**：实测每个类别内部
   difficulty 是常量（single_chunk 全 easy、direct_relation 全 medium…），拿它分层
   等于没分。真正带混合信息的维度只有 negative_abstain 的 should_abstain。
2. 桶 `sha256(SPLIT_SEED + "|" + tag + "|" + question_id)`，层内按桶升序取前
   `ceil(ratio * n)` 进 holdout，其余进 dev。
3. 同一份输入跑两次结果必须逐字节一致 —— 由 `tests/test_make_split.py` 钉住。

新增批为什么要换种子重挖而不是复用 v1：v1 那 320 条工程师 A 已经逐题看过结果
（gold 排名、上下文命中、答案覆盖），从里面切出来的 holdout 对他不等于「从未
见过」。新挖的一批是他没看过的，唯一真正干净的证据。**但这只解决「看过」这一种
污染；新增批在断言覆盖质量上并不比 v1 干净**，见 `docs/goldset-v2说明.md`。

退出码：
    0  切分完成且所有自检通过
    1  自检失败（重叠 / 缺口 / 规则被破坏）
    2  数据源不可用
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from arkrag import config, goldset, release  # noqa: E402

# ---------------------------------------------------------------- 冻结常量

#: 切分种子。**改这个值等于重新切分**，已在库的 holdout 就不再是同一批题。
SPLIT_SEED = "arkrag-split-v1-20261104"

#: v1 那 320 条的 holdout 比例。0.30 而不是 0.20：0.20 时 negative_abstain(可答)
#: 只剩 3 条、adversarial 只剩 3 条，单类结论会完全不可用。
V1_RATIO = 0.30

#: 新增批挖矿种子。与 v1 的 20261101 不同，是为了换出另一批候选。
NEW_SEED = 20261111

#: 新增批先挖这么多，再按「与 v1 完全不重叠」筛。挖 640 是为了给八类都留余量：
#: 实测 time_conflict 池子最小，640 条只筛出 48 条可用，配额取 32 还余 16 条。
NEW_MINE_TOTAL = 640

#: 新增批内部对半切。
NEW_RATIO = 0.50

#: 新增批八类配额，与 v1 的八类占比完全一致（64/48/48/48/32/32/32/16 = 320）。
NEW_QUOTAS = {
    "single_chunk": 64,
    "alias": 48,
    "direct_relation": 48,
    "multihop": 48,
    "cross_chunk": 32,
    "time_conflict": 32,
    "negative_abstain": 32,
    "adversarial": 16,
}

#: 新增批的 question_id 从这一档起编号，避开 v1 已占用的 000001–000064。
#: `goldset.QUESTION_ID_RE` 要求六位数字，所以只能在这种整数档位上做区隔。
NEW_ID_BASE = 900000

GOLD_V1 = ROOT / "data" / "goldset" / "gold-v1.json"
GOLD_V2 = ROOT / "data" / "goldset" / "gold-v2.json"
SPLIT_V2 = ROOT / "data" / "goldset" / "split-v2.json"


# ---------------------------------------------------------------- 小工具


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def bucket(*parts: str) -> str:
    """层内排序键。拼串方式固定，改动会让全部归属变化。"""
    payload = SPLIT_SEED + "|" + "|".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _strata(items: list[dict]) -> dict[tuple[str, bool], list[dict]]:
    out: dict[tuple[str, bool], list[dict]] = collections.defaultdict(list)
    for it in items:
        out[(it["category"], bool(it["should_abstain"]))].append(it)
    return out


def split_by_stratum(items: list[dict], ratio: float, tag: str) -> tuple[dict[str, str], list[dict]]:
    """按 `(category, should_abstain)` 分层切分。

    返回 `({question_id: "dev"|"holdout"}, 每层的账)`。层内按桶升序取前
    `ceil(ratio*n)` 进 holdout —— 用 `ceil` 而不是 `round`，保证小样本层也至少
    有一条进 holdout（否则 `n=4` 的层可能一条都不进，holdout 里就缺了这一类）。
    """
    assign: dict[str, str] = {}
    ledger: list[dict] = []
    strata = _strata(items)
    ordered = sorted(strata, key=lambda k: (goldset.CATEGORIES.index(k[0]), k[1]))
    for key in ordered:
        rows = sorted(strata[key], key=lambda it: bucket(tag, it["question_id"]))
        n = len(rows)
        n_hold = math.ceil(ratio * n)
        for idx, it in enumerate(rows):
            assign[it["question_id"]] = "holdout" if idx < n_hold else "dev"
        ledger.append(
            {
                "category": key[0],
                "should_abstain": key[1],
                "n": n,
                "holdout": n_hold,
                "dev": n - n_hold,
            }
        )
    return assign, ledger


def _fingerprint(obj: object) -> dict:
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()
    return {"sha256": digest, "sha256_16": digest[:16], "bytes": len(blob)}


# ---------------------------------------------------------------- 主流程


def _pick_generation() -> Path:
    """取最新一个 `ark-rag-3` 世代（新题必须从带逐观察断言的世代挖）。"""
    gens_dir = release.GENERATIONS_DIR
    if not gens_dir.exists():
        raise SystemExit(f"世代目录不存在：{gens_dir}")
    candidates: list[Path] = []
    for path in sorted(gens_dir.iterdir()):
        if not path.is_dir():
            continue
        graph_path = path / "graph" / "graph.json"
        if not graph_path.exists() or not (path / "graph" / "assertions.jsonl").exists():
            continue
        try:
            doc = json.loads(graph_path.read_text("utf-8"))
        except json.JSONDecodeError:
            continue
        if doc.get("schema_version") == "ark-rag-3":
            candidates.append(path)
    if not candidates:
        raise SystemExit("没有可用的 ark-rag-3 世代（需要 graph/assertions.jsonl）")
    return candidates[-1]


def _renumber(items: list[dict]) -> None:
    """给新题编号。从 `NEW_ID_BASE` 起，避开 v1 已占用的号段。"""
    counters: dict[str, int] = collections.defaultdict(int)
    for it in items:
        abbr = goldset.CATEGORY_ABBR[it["category"]]
        counters[abbr] += 1
        it["question_id"] = f"q-{abbr}-{NEW_ID_BASE + counters[abbr]:06d}"


def mine_new_batch(gen_dir: Path, v1: dict) -> tuple[list[dict], dict]:
    """挖新增批：换种子重挖 → 三条件去重 → 按八类配额取。"""
    graph_doc = json.loads((gen_dir / "graph" / "graph.json").read_text("utf-8"))
    adjacency = json.loads((gen_dir / "graph" / "adjacency.json").read_text("utf-8"))
    assertions = _read_jsonl(gen_dir / "graph" / "assertions.jsonl")
    chunks = _read_jsonl(config.chunks_path())

    v1_questions = {i["question"] for i in v1["items"]}
    v1_chunks: set[str] = set()
    v1_assertions: set[str] = set()
    for item in v1["items"]:
        v1_chunks |= set(item.get("gold_chunk_ids") or [])
        v1_assertions |= set(item.get("gold_assertion_ids") or [])

    mined = goldset.build_gold_set(
        assertions,
        chunks,
        graph_doc.get("edges") or [],
        graph_doc.get("entities") or [],
        adjacency,
        total=NEW_MINE_TOTAL,
        seed=NEW_SEED,
    )

    # 三条件去重。第三条（gold_assertion_ids 无交集）看起来多余，其实必要：
    # 断言被复用意味着「同一句原文」，A 的 triage 里已经看过它的排名。
    pool = [
        it
        for it in mined["items"]
        if it["question"] not in v1_questions
        and not (set(it.get("gold_chunk_ids") or []) & v1_chunks)
        and not (set(it.get("gold_assertion_ids") or []) & v1_assertions)
    ]

    by_cat: dict[str, list[dict]] = collections.defaultdict(list)
    for it in pool:
        by_cat[it["category"]].append(it)

    picked: list[dict] = []
    shortfall: dict[str, dict] = {}
    for cat in goldset.CATEGORIES:
        rows = sorted(by_cat[cat], key=lambda it: bucket("new", it["question"]))
        want = NEW_QUOTAS[cat]
        if len(rows) < want:
            # 缺口如实记录、如实少给，绝不用别类凑数（与 build_gold_set 同一原则）。
            shortfall[cat] = {"want": want, "have": len(rows)}
        picked.extend(rows[:want])
    _renumber(picked)

    cmap = {c["chunk_id"]: c for c in chunks if c.get("chunk_id")}
    by_entity = {e.get("id"): e for e in (graph_doc.get("entities") or []) if e.get("id")}
    failures = []
    for it in picked:
        errs = goldset.validate_gold_item(
            it, cmap, edges=graph_doc.get("edges") or [], entities_by_id=by_entity
        )
        if errs:
            failures.append({"question_id": it["question_id"], "errors": errs})

    stats = {
        "generation": gen_dir.name,
        "mine_total": NEW_MINE_TOTAL,
        "mined_actual": len(mined["items"]),
        "pool_disjoint": len(pool),
        "pool_per_cat": {c: len(by_cat[c]) for c in goldset.CATEGORIES},
        "quotas": dict(NEW_QUOTAS),
        "produced": len(picked),
        "per_cat": dict(collections.Counter(it["category"] for it in picked)),
        "difficulty": dict(collections.Counter(it["difficulty"] for it in picked)),
        "abstain": {
            ("true" if k else "false"): v
            for k, v in collections.Counter(bool(it["should_abstain"]) for it in picked).items()
        },
        "shortfall": shortfall,
        "validation_failures": failures,
    }
    return picked, stats


def build() -> tuple[dict, dict, dict]:
    """返回 `(gold_v2, split_v2, report)`。纯函数式：不写盘，方便测试直接调。"""
    if not GOLD_V1.exists():
        raise SystemExit(f"缺 gold-v1.json：{GOLD_V1}")
    v1 = json.loads(GOLD_V1.read_text("utf-8"))

    gen_dir = _pick_generation()
    new_items, new_stats = mine_new_batch(gen_dir, v1)

    v1_assign, v1_ledger = split_by_stratum(v1["items"], V1_RATIO, "v1")
    new_assign, new_ledger = split_by_stratum(new_items, NEW_RATIO, "new")

    # ---- 组装 gold-v2：v1 原样搬，只加 split / origin ----
    items: list[dict] = []
    for it in v1["items"]:
        copy = dict(it)
        copy["split"] = v1_assign[it["question_id"]]
        copy["origin"] = "v1"
        items.append(copy)
    for it in new_items:
        copy = dict(it)
        copy["split"] = new_assign[it["question_id"]]
        copy["origin"] = "new"
        items.append(copy)

    quotas = {c: NEW_QUOTAS[c] + v1["quotas"][c] for c in goldset.CATEGORIES}
    produced = dict(collections.Counter(it["category"] for it in items))
    duplicate_questions = goldset.find_duplicate_questions(items)
    all_failures = list(new_stats["validation_failures"])

    gold_v2 = {
        "schema": "ark-goldset-2",
        "split_seed": SPLIT_SEED,
        "origin": {
            "v1": {"path": config.rel_to_root(GOLD_V1), "seed": v1["seed"], "total": len(v1["items"])},
            "new": {"generation": new_stats["generation"], "seed": NEW_SEED,
                    "mine_total": NEW_MINE_TOTAL, "quotas": dict(NEW_QUOTAS)},
        },
        "total": len(items),
        "quotas": quotas,
        "report": {
            "total_actual": len(items),
            "quotas": quotas,
            "produced": produced,
            "gaps": {c: quotas[c] - produced.get(c, 0) for c in goldset.CATEGORIES
                     if quotas[c] > produced.get(c, 0)},
            "by_split": dict(collections.Counter(it["split"] for it in items)),
            "by_origin": dict(collections.Counter(it["origin"] for it in items)),
            "share_actual": {c: round(produced.get(c, 0) / len(items), 4) for c in goldset.CATEGORIES},
            "validation_failures": all_failures,
            "duplicate_questions": duplicate_questions,
            "new_batch": {k: v for k, v in new_stats.items() if k != "validation_failures"},
        },
        "items": items,
    }

    split_doc = {
        "schema": "ark-split-1",
        "split_seed": SPLIT_SEED,
        "rule": {
            "stratum": "(category, should_abstain)",
            "stratum_why": "difficulty 在类内是常量，分层无信息量；唯一混合维度是 negative_abstain 的 should_abstain",
            "bucket": f"sha256(SPLIT_SEED + '|' + tag + '|' + question_id)，层内升序",
            "pick": "层内按桶升序取前 ceil(ratio*n) 进 holdout",
            "v1_ratio": V1_RATIO,
            "new_ratio": NEW_RATIO,
            "tags": {"v1": "v1", "new": "new"},
            "v1_ratio_why": "0.20 时 negative_abstain(可答) 与 adversarial 各只剩 3 条，单类结论不可用",
        },
        "v1": {
            "source": config.rel_to_root(GOLD_V1),
            "n": len(v1["items"]),
            "holdout": sorted(q for q, s in v1_assign.items() if s == "holdout"),
            "ledger": v1_ledger,
        },
        "new": {
            "generation": new_stats["generation"],
            "seed": NEW_SEED,
            "mine_total": NEW_MINE_TOTAL,
            "n": len(new_items),
            "holdout": sorted(q for q, s in new_assign.items() if s == "holdout"),
            "ledger": new_ledger,
        },
        "totals": {
            "v1_holdout": sum(1 for s in v1_assign.values() if s == "holdout"),
            "v1_dev": sum(1 for s in v1_assign.values() if s == "dev"),
            "new_holdout": sum(1 for s in new_assign.values() if s == "holdout"),
            "new_dev": sum(1 for s in new_assign.values() if s == "dev"),
        },
        "clean_note": (
            "v1 的 holdout 只解决『同一批题别再拿去调参』，不解决污染："
            "工程师A 已看过 v1 全 320 题的逐题结果。对 A 真正干净的只有 new 那一半。"
        ),
    }
    split_doc["totals"]["holdout"] = (
        split_doc["totals"]["v1_holdout"] + split_doc["totals"]["new_holdout"]
    )
    split_doc["totals"]["dev"] = split_doc["totals"]["v1_dev"] + split_doc["totals"]["new_dev"]

    report = {
        "gold_v1": {"path": config.rel_to_root(GOLD_V1),
                    "sha256_16": hashlib.sha256(GOLD_V1.read_bytes()).hexdigest()[:16],
                    "bytes": GOLD_V1.stat().st_size},
        "gold_v2": _fingerprint(gold_v2),
        "split_v2": _fingerprint(split_doc),
        "totals": split_doc["totals"],
        "new_batch": gold_v2["report"]["new_batch"],
        "v1_ledger": v1_ledger,
        "new_ledger": new_ledger,
    }
    return gold_v2, split_doc, report


def self_check(gold_v2: dict, split_doc: dict) -> list[str]:
    """切分自检。任何一条不成立都必须让脚本非零退出——这些是切分的定义性约束。"""
    problems: list[str] = []

    items = gold_v2["items"]
    if len(items) != 640:
        problems.append(f"总题数 {len(items)} ≠ 640")
    for cat in goldset.CATEGORIES:
        want = gold_v2["quotas"][cat]
        got = sum(1 for it in items if it["category"] == cat)
        if got != want:
            problems.append(f"{cat} 实得 {got} ≠ 配额 {want}")

    for it in items:
        if it.get("split") not in ("dev", "holdout"):
            problems.append(f"{it['question_id']} 的 split 非法：{it.get('split')!r}")
            break

    # v1 那半必须逐字段等于 gold-v1 原件（只允许 split / origin 两个新字段）
    v1_items = [it for it in items if it.get("origin") == "v1"]
    if len(v1_items) != 320:
        problems.append(f"origin=v1 的题数 {len(v1_items)} ≠ 320")
    v1_src = json.loads(GOLD_V1.read_text("utf-8"))
    src_by_id = {it["question_id"]: it for it in v1_src["items"]}
    for it in v1_items:
        src = src_by_id.get(it["question_id"])
        if src is None:
            problems.append(f"v1 题 {it['question_id']} 在 gold-v1 里找不到")
            break
        stripped = {k: v for k, v in it.items() if k not in ("split", "origin")}
        if stripped != src:
            problems.append(f"v1 题 {it['question_id']} 的字段被改动了（除 split/origin 外）")
            break
    new_q = {it["question"] for it in items if it.get("origin") == "new"}
    v1_q = {it["question"] for it in v1_items}
    if new_q & v1_q:
        problems.append(f"新增批与 v1 有 {len(new_q & v1_q)} 条问句重复")
    ids = [it["question_id"] for it in items]
    if len(set(ids)) != len(ids):
        problems.append("question_id 有重复")

    # gold-v2 里 v1 那半的 gold 块不得与新增批共享（那等于污染新题）
    v1_ch: set[str] = set()
    new_ch: set[str] = set()
    for it in items:
        (v1_ch if it.get("origin") == "v1" else new_ch).update(it.get("gold_chunk_ids") or [])
    if v1_ch & new_ch:
        problems.append(f"新增批与 v1 共享 {len(v1_ch & new_ch)} 个 gold 块")

    if gold_v2["report"]["duplicate_questions"]:
        problems.append(f"重复问句 {len(gold_v2['report']['duplicate_questions'])} 条")
    if gold_v2["report"]["validation_failures"]:
        problems.append(f"校验失败 {len(gold_v2['report']['validation_failures'])} 条")

    totals = split_doc["totals"]
    if totals["holdout"] + totals["dev"] != len(items):
        problems.append(
            f"holdout {totals['holdout']} + dev {totals['dev']} ≠ 总题数 {len(items)}"
        )
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description="P1-a：切分调参集 / 验收集并冻结规则")
    ap.add_argument("--dry-run", action="store_true", help="只打印账，不写盘")
    args = ap.parse_args()

    gold_v2, split_doc, report = build()
    problems = self_check(gold_v2, split_doc)

    t = report["totals"]
    print("=== 切分账 ===")
    print(f"v1    ：总 320 → holdout {t['v1_holdout']} / dev {t['v1_dev']}")
    print(f"新增批：总 320 → holdout {t['new_holdout']} / dev {t['new_dev']}")
    print(f"合计  ：总 640 → holdout {t['holdout']} / dev {t['dev']}")
    nb = report["new_batch"]
    print(
        f"\n新增批挖掘：seed {NEW_SEED} 挖 {nb['mine_total']} 条 → "
        f"三条件去重后池子 {nb['pool_disjoint']} 条 → 取 {nb['produced']} 条"
    )
    print(f"池子分布    {nb['pool_per_cat']}")
    print(f"缺口        {nb['shortfall'] or '无'}")
    print(f"难度        {nb['difficulty']}")
    print(f"应拒答      {nb['abstain']}")

    print("\n=== 指纹 ===")
    print(f"gold-v1   sha256_16 {report['gold_v1']['sha256_16']}  {report['gold_v1']['bytes']} 字节（未改动）")
    print(f"gold-v2   sha256_16 {report['gold_v2']['sha256_16']}  {report['gold_v2']['bytes']} 字节")
    print(f"split-v2  sha256_16 {report['split_v2']['sha256_16']}  {report['split_v2']['bytes']} 字节")

    if problems:
        print("\n[自检失败]")
        for p in problems:
            print("  - " + p)
        return 1
    print("\n自检：全部通过")

    if args.dry_run:
        print("\n(dry-run：未写盘)")
        return 0

    from arkrag import atomic

    atomic.atomic_write_json(SPLIT_V2, split_doc, indent=2)
    atomic.atomic_write_json(GOLD_V2, gold_v2, indent=2)
    print(f"\n已写出 {config.rel_to_root(SPLIT_V2)}")
    print(f"已写出 {config.rel_to_root(GOLD_V2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
