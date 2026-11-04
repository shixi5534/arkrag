# -*- coding: utf-8 -*-
"""K3：把「别名丢在哪一环」这件事钉到可复核的粒度——四段顺序执行。

段一 scope：别名声明在 n1 / n2 两代缓存里各是什么形态——是全都不写了，还是只是这一条。
    只读 data/arkrag/cache 下的 JSON，统计每个 normalizer 版本的记录数、
    raw.entities 里带 `aliases` 字段的条数 / 带非空 aliases 的条数 / 带 `desc` 字段的条数。
段二 source：`劫车人` / `莱伊` 这两个名字分别由**哪些缓存记录**支撑。
    要回答的问题是：新代里 `劫车人` 与 `莱伊` 变成两个实体，是因为哪条记录丢了别名声明。
段三 graph：如果「别名在 n2 入库路径上被整体丢掉」成立，那么新一代图里的别名总量
    应当明显少于旧一代——而不只是 莱伊/劫车人 这一条。只读两代 graph.json，
    数实体、有别名实体、别名条目。
段四 index：把「别名丢在哪一环」钉到可复核粒度。三件事，全部只读：
    1. n2 缓存记录数 == 新增/变更块数？（34）
    2. 两代 alias_index 里 莱伊 / 劫车人 / 莱叶菈 各指向谁？
    3. 新世代里 劫车人 的 chunks 来自哪些场景（说明它为什么还在）。

四段各自按 `===== 段落名 =====` 分隔（原本是四个独立脚本，提升为一条命令）。

退出码：0 = 四段都跑完（判据：段三段四的两代图都可读、段一缓存目录存在）/ 1 = 判据不成立。
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CACHE = ROOT / "data" / "arkrag" / "cache"
GEN = ROOT / "data" / "arkrag" / "generations"
OLD = GEN / "g-20261031T035231Z-4eafae" / "graph" / "graph.json"
NEW = GEN / "g-20261103T132152Z-06a741" / "graph" / "graph.json"

WANT = ("劫车人", "莱伊")
NAMES = ("莱伊", "劫车人", "莱叶菈")


def section_scope() -> None:
    by_norm: Counter[str] = Counter()
    files = 0
    bad = 0
    for p in CACHE.glob("*/*.json"):
        try:
            d = json.loads(p.read_text("utf-8"))
        except Exception:
            bad += 1
            continue
        files += 1
        by_norm[d.get("normalizer", "(无该字段)")] += 1

    print(f"缓存文件 {files} 个（不可解析 {bad}）")
    for k, v in sorted(by_norm.items(), key=lambda kv: -kv[1]):
        print(f"  normalizer={k!r}  {v} 条")

    print()
    has_aliases: Counter[str] = Counter()
    has_alias_nonempty: Counter[str] = Counter()
    has_desc: Counter[str] = Counter()
    ents_total: Counter[str] = Counter()

    for p in CACHE.glob("*/*.json"):
        try:
            d = json.loads(p.read_text("utf-8"))
        except Exception:
            continue
        norm = d.get("normalizer", "(无该字段)")
        raw = d.get("raw") or {}
        for e in raw.get("entities") or []:
            ents_total[norm] += 1
            if "aliases" in e:
                has_aliases[norm] += 1
                if e["aliases"]:
                    has_alias_nonempty[norm] += 1
            if "desc" in e:
                has_desc[norm] += 1

    print("按 normalizer 统计 raw.entities：")
    for k in sorted(ents_total, key=lambda x: -ents_total[x]):
        print(f"  {k!r}: 实体 {ents_total[k]} / 有 aliases 键 {has_aliases[k]}"
              f" / aliases 非空 {has_alias_nonempty[k]} / 有 desc 键 {has_desc[k]}")


def section_source() -> None:
    hits: dict[str, list[tuple[str, str, dict]]] = defaultdict(list)

    for p in CACHE.glob("*/*.json"):
        try:
            d = json.loads(p.read_text("utf-8"))
        except Exception:
            continue
        for e in (d.get("raw") or {}).get("entities") or []:
            nm = str(e.get("name") or "")
            if nm not in WANT:
                continue
            aliases = [a for a in (e.get("aliases") or []) if a]
            # 只报两种情况：名字命中，或别名命中
            if nm in WANT or any(a in WANT for a in aliases):
                hits[nm].append(
                    (d.get("chunk_id", "?"), d.get("normalizer", "?"),
                     {"name": nm, "aliases": aliases, "desc": e.get("desc")})
                )

    for nm in WANT:
        rows = hits[nm]
        print(f"===== 实体名 {nm}：被 {len(rows)} 条缓存记录声明 =====")
        by_norm: dict[str, int] = defaultdict(int)
        for _cid, norm, _e in rows:
            by_norm[norm] += 1
        print(f"  按 normalizer：{dict(by_norm)}")
        for cid, norm, e in rows:
            print(f"  [{norm}] {cid}  aliases={e['aliases']}  desc={e['desc']!r}")
        print()

    # 另外：有多少条记录把 莱伊 写成了别人的别名
    print("===== 把 `莱伊` 写成别名的记录 =====")
    n = 0
    for p in CACHE.glob("*/*.json"):
        try:
            d = json.loads(p.read_text("utf-8"))
        except Exception:
            continue
        for e in (d.get("raw") or {}).get("entities") or []:
            if "莱伊" in [a for a in (e.get("aliases") or []) if a]:
                n += 1
                print(f"  [{d.get('normalizer')}] {d.get('chunk_id')}  "
                      f"实体 {e.get('name')}  aliases={e.get('aliases')}")
    print(f"  合计 {n} 条")

    print()
    print("===== 把 `劫车人` 写成别名的记录 =====")
    n = 0
    for p in CACHE.glob("*/*.json"):
        try:
            d = json.loads(p.read_text("utf-8"))
        except Exception:
            continue
        for e in (d.get("raw") or {}).get("entities") or []:
            if "劫车人" in [a for a in (e.get("aliases") or []) if a]:
                n += 1
                print(f"  [{d.get('normalizer')}] {d.get('chunk_id')}  "
                      f"实体 {e.get('name')}  aliases={e.get('aliases')}")
    print(f"  合计 {n} 条")


def _graph_stat(path: Path) -> dict:
    g = json.loads(path.read_text("utf-8"))
    ents = g["entities"]
    with_alias = 0
    alias_total = 0
    pending = 0
    for e in ents:
        al = e.get("aliases") or []
        texts = [a.get("text") if isinstance(a, dict) else a for a in al]
        texts = [t for t in texts if t]
        if texts:
            with_alias += 1
        alias_total += len(texts)
        pending += len(e.get("pending_aliases") or [])
    ai = g.get("alias_index") or {}
    return {
        "entities": len(ents),
        "with_alias": with_alias,
        "alias_total": alias_total,
        "pending": pending,
        "alias_index": len(ai),
    }


def section_graph() -> None:
    o = _graph_stat(OLD)
    n = _graph_stat(NEW)

    print(f"{'':22}{'旧 4eafae':>14}{'新 06a741':>14}{'差':>12}")
    for k in ("entities", "with_alias", "alias_total", "pending", "alias_index"):
        print(f"{k:22}{o[k]:>14,}{n[k]:>14,}{n[k] - o[k]:>12,}")

    print()
    print(f"有别名实体占比：旧 {o['with_alias'] / o['entities']:.4%}"
          f"  新 {n['with_alias'] / n['entities']:.4%}")


def section_index() -> None:
    # --- 1. n2 记录数 ---
    n1 = n2 = 0
    for p in CACHE.glob("*/*.json"):
        try:
            d = json.loads(p.read_text("utf-8"))
        except Exception:
            continue
        if d.get("normalizer") == "n2":
            n2 += 1
        else:
            n1 += 1
    print(f"缓存记录：n1 {n1} 条 / n2 {n2} 条")
    print()

    def probe(tag: str, path: Path) -> dict:
        g = json.loads(path.read_text("utf-8"))
        ai = g.get("alias_index") or {}
        print(f"--- {tag} alias_index ---")
        for nm in NAMES:
            print(f"  {nm} -> {ai.get(nm, '(不在别名表)')}")
        by_id = {e["id"]: e for e in g["entities"]}
        for e in g["entities"]:
            if e["name"] in NAMES:
                chs = e.get("chunks") or []
                docs = sorted({c.split("#")[0] for c in chs})
                print(f"  实体 {e['id']} name={e['name']} mentions={e.get('mentions')} "
                      f"chunks={len(chs)} 场景数={len(docs)}")
                if e["name"] == "劫车人":
                    for d_ in docs:
                        print(f"      来自 {d_}")
        print()
        return by_id

    probe("旧 4eafae", OLD)
    probe("新 06a741", NEW)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    # 判据：缓存目录与两代图都得在，否则四段的读数没有意义
    if not CACHE.is_dir() or not OLD.is_file() or not NEW.is_file():
        missing = [str(p) for p in (CACHE, OLD, NEW) if not p.exists()]
        print("缺少输入：", "；".join(missing))
        return 1

    print("===== 段一：别名声明在 n1/n2 缓存里的形态（scope）=====")
    section_scope()
    print()
    print("===== 段二：莱伊/劫车人的缓存来源（source）=====")
    section_source()
    print()
    print("===== 段三：两代图上的别名总量（graph）=====")
    section_graph()
    print()
    print("===== 段四：alias_index 指向与 n2 记录数（index）=====")
    section_index()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
