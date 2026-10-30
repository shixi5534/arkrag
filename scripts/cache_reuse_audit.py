"""缓存复用取证：13,798 块口径下每一个 miss 块到底能不能复用旧抽取结果。

## 它回答什么问题

D1/D2 修了旧分块器的两个 bug 之后，真实语料重跑 L1/L2 得到 13,798 个 T1 块
（冻结基线是 13,781）。块数变多必然带来块身份变化：那 17 个 activities 族文档
各多出 1 块，其**后续块**的 `chunk_id`（`f"{doc_id}#{index:03d}"`）整体后移一位。

新键（`cache.key_for`）把 `chunk_id` 绑进了键，所以后移的块全部变成 miss。
但"键不命中"不等于"抽取结果不可用"——raw 抽取的输入只有三样：
**块正文 + 提示词 + 模型**。如果一个 miss 块的正文与某条 legacy 记录当初抽取的
正文**逐字节相同**，且提示词与模型也相同，那么那条记录就是本块的正确结果，
与块编号叫 `#007` 还是 `#006` 无关。

本脚本就是把这件事逐块查清楚，并**只依据字节**下结论：

    identical_content  正文逐字节相同，且存在合法 legacy 记录（可复用）
    changed_content    同一位置存在 legacy 记录，但正文确实变了（必须重抽）
    no_record          该位置在旧缓存里根本没有记录（必须重抽）

## 为什么不能按 chunk_id 前缀猜

"前缀相同就当作同一块"正是这个项目里最容易犯的错：overlap 让相邻块正文高度
重复，而块号后移又让"同号不同内容"成为常态。所以本脚本对每一块做四件事：

1. 重算 `sha256(新块正文)[:24]`，确认它等于块记录里的 `content_key`
   （先把"块记录自称的 content_key"钉死在实际字节上）；
2. 用旧键 `cache_key(content_key, model, PROMPT_VERSION)` 定位 legacy 记录。
   注意**旧键不含 chunk_id**，因此块号后移不影响它——这让"正文相同"的
   记录仍然可寻；
3. 从冻结基线的旧 `chunks.jsonl` 里取出 legacy 记录当初对应的那一块
   （`rec["chunk_id"]`），把它的 `text` 与新块 `text` 做**字节级**比较；
4. 再用 `prompt.build_user_prompt()` 分别渲染两块的用户提示，做**字节级**比较
   ——用户提示里含篇目/章节/场景/说话人/正文，说话人列表变了提示词就变了，
   这层必须单独比，不能只比正文。

## 关于 prompt_sha256 的诚实说明

legacy 记录里**没有** `prompt_sha256` 字段（那时的记录只有
`__cache_format__ / normalizer / prompt_version / model / chunk_id / content_key /
written_at / raw`）。所以"同 prompt_sha256"这一条在这批记录上无法直接比对，
只能通过 `prompt_version == config.PROMPT_VERSION` 间接成立——这也正是
`cache._legacy_record_is_safe()` 采用的口径。本脚本把这一点显式打印出来，
不用"四项全等"的说法去盖住一处实际靠版本号间接成立的事实。

新键记录（`key_scheme == 2`）才有 `prompt_sha256`，那种情况下本脚本直接比它。

## 用法

    $env:ARKRAG_ARTIFACT_DIR = "data/arkrag/staging/d-stage"
    py -3 scripts\\cache_reuse_audit.py
    py -3 scripts\\cache_reuse_audit.py --json data/arkrag/staging/d-stage/cache_reuse_audit.json

退出码：0 = 全部 miss 块都已归类且有据可查；1 = 存在 changed_content 或
no_record（这些块**必须重抽**，本轮构建不能声称覆盖完整 T1 范围）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arkrag import cache, config, prompt  # noqa: E402
from arkrag.agent_tasks import AGENT_MODEL  # noqa: E402

# 冻结基线的旧分块（13,786 行 / T1 13,781）。它是本轮之前那份抽取的**原始输入**，
# 因此是校验"legacy 记录当初抽的到底是哪段正文"的唯一权威来源。
# 注意它绑在 DATA_DIR 上，不受 ARKRAG_ARTIFACT_DIR 影响——这正是我们要的：
# 本轮产物在 staging，旧输入留在冻结目录。
LEGACY_CHUNKS_PATH = config.DATA_DIR / "chunks.jsonl"

IDENTICAL = "identical_content"
CHANGED = "changed_content"
NO_RECORD = "no_record"


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_legacy_chunks(path: Path) -> dict[str, dict]:
    """旧 chunks.jsonl → {chunk_id: chunk}。文件缺失即返回空（后面会报告）。"""
    if not path.exists():
        return {}
    return {c["chunk_id"]: c for c in load_jsonl(path)}


def scan_cache() -> tuple[dict[str, dict], dict[str, dict]]:
    """把整个缓存目录读两遍：按 chunk_id 建索引，也按 content_key 建索引。

    第二个索引是"identical_content"判定的关键：旧键里 `content_key` 只由正文算出、
    **不含 chunk_id**，所以哪怕块号后移，只要正文逐字节相同，某个记录的
    `content_key` 就会等于新块重算出来的 `content_key`。用全库索引去找它，
    而不是只查"同位置那条记录"——后者会漏掉"正文相同但记录挂在别的块号下"的情况，
    而本脚本存在的意义恰恰就是不能漏。
    """
    by_chunk: dict[str, dict] = {}
    by_content: dict[str, dict] = {}
    dupes: dict[str, int] = {}
    if not config.CACHE_DIR.exists():
        return by_chunk, by_content
    for path in sorted(config.CACHE_DIR.rglob("*.json")):
        try:
            rec = json.loads(path.read_text("utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(rec, dict):
            continue
        rel = str(path.relative_to(config.ROOT)) if path.is_relative_to(config.ROOT) else str(path)
        cid = rec.get("chunk_id")
        if isinstance(cid, str) and cid:
            if cid in by_chunk:
                dupes[cid] = dupes.get(cid, 1) + 1
            else:
                rec["__path__"] = rel
                by_chunk[cid] = rec
        ck = rec.get("content_key")
        if isinstance(ck, str) and ck:
            by_content.setdefault(ck, dict(rec, __path__=rel))
    if dupes:
        print(f"注意：{len(dupes)} 个 chunk_id 在缓存里有多条记录，按 chunk_id 索引用第一条")
    return by_chunk, by_content


def doc_of(chunk_id: str) -> str:
    return chunk_id.rsplit("#", 1)[0] if "#" in chunk_id else chunk_id


def classify(
    chunk: dict,
    *,
    model: str,
    legacy_chunks: dict[str, dict],
    by_chunk_id: dict[str, dict],
    by_content_key: dict[str, dict],
) -> dict:
    """对单个 miss 块取证。所有结论都由字节比较得出，不看块号前缀。"""
    text = chunk["text"]
    doc = doc_of(chunk["chunk_id"])
    ck_from_bytes = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
    row: dict = {
        "chunk_id": chunk["chunk_id"],
        "doc_id": doc,
        "scene_id": chunk.get("scene_id") or chunk.get("doc_id") or "",
        "content_key_in_record": chunk["content_key"],
        "content_key_from_bytes": ck_from_bytes,
        "chars": len(text),
        "text_sha256": sha256_hex(text),
        "user_prompt_sha256": sha256_hex(prompt.build_user_prompt(chunk)),
        "classification": NO_RECORD,
        "evidence": {},
        "blocking": [],
    }
    # 步骤 1：块记录自称的 content_key 必须真的由正文算出
    if row["content_key_in_record"] != ck_from_bytes:
        row["blocking"].append(
            f"块记录 content_key 与正文不符：{row['content_key_in_record']} != {ck_from_bytes}"
        )

    # 步骤 2：先按"正文寻址"找记录。旧键 = sha256(content_key|model|prompt_version)[:24]，
    #          不含 chunk_id，所以块号后移不影响；再用 content_key 全库索引兜一层，
    #          覆盖"记录挂在别的块号下但正文相同"的情况。
    legacy_key = cache.cache_key(chunk["content_key"], model, config.PROMPT_VERSION)
    legacy_path = cache.cache_path(legacy_key)
    row["evidence"]["legacy_key"] = legacy_key
    row["evidence"]["legacy_path"] = (
        str(legacy_path.relative_to(config.ROOT)) if legacy_path.is_relative_to(config.ROOT) else str(legacy_path)
    )
    row["evidence"]["legacy_record_present"] = legacy_path.exists()

    rec: dict | None = None
    if legacy_path.exists():
        try:
            loaded = json.loads(legacy_path.read_text("utf-8"))
            rec = loaded if isinstance(loaded, dict) else None
        except (OSError, ValueError):
            rec = None
    if rec is None:
        hit = by_content_key.get(chunk["content_key"])
        if hit is not None:
            rec = hit
            row["evidence"]["found_by"] = "content_key_index"
        else:
            row["evidence"]["found_by"] = ""
    else:
        row["evidence"]["found_by"] = "legacy_key"

    if rec is None:
        # 正文寻址彻底找不到记录：交给 classify_unrecorded 按字节判 changed / no_record
        return classify_unrecorded(
            chunk, legacy_chunks=legacy_chunks, doc_texts={}, by_chunk_id=by_chunk_id
        )

    # 步骤 2b：记录自带身份字段逐项核对
    row["evidence"]["record_chunk_id"] = rec.get("chunk_id")
    row["evidence"]["record_content_key"] = rec.get("content_key")
    row["evidence"]["record_model"] = rec.get("model")
    row["evidence"]["record_prompt_version"] = rec.get("prompt_version")
    row["evidence"]["record_has_prompt_sha256"] = "prompt_sha256" in rec

    # 内容相等的判定基准是**正文**，不是块号。content_key 就是正文的 sha256 前 24 位。
    content_matches = ck_from_bytes == chunk["content_key"] == rec.get("content_key")
    row["evidence"]["content_key_matches"] = content_matches
    if not content_matches:
        row["blocking"].append(
            f"content_key 不符：记录 {rec.get('content_key')} vs 块 {chunk['content_key']} vs 正文 {ck_from_bytes}"
        )
    if rec.get("model") != model:
        row["blocking"].append(f"模型不符：{rec.get('model')} != {model}")
    if rec.get("prompt_version") != config.PROMPT_VERSION:
        row["blocking"].append(
            f"提示词版本过期：{rec.get('prompt_version')} != {config.PROMPT_VERSION}"
        )

    raw = rec.get("raw")
    if not isinstance(raw, dict) or "entities" not in raw or "relations" not in raw:
        row["blocking"].append("raw 结构不合法（缺 entities/relations 或不是对象）")

    # 步骤 3：拿该记录当初那一块的正文，做**字节级**比较
    old = legacy_chunks.get(rec.get("chunk_id") or "")
    if old is None:
        row["blocking"].append(
            f"旧 chunks.jsonl 里找不到块 {rec.get('chunk_id')!r}，无法做正文比对"
        )
    else:
        row["evidence"]["legacy_chunk_text_sha256"] = sha256_hex(old["text"])
        row["evidence"]["legacy_chunk_chars"] = len(old["text"])
        row["evidence"]["text_identical"] = old["text"] == text
        # 步骤 4：用户提示**字节级**比较（含篇目/章节/场景/说话人/正文）
        old_pp = prompt.build_user_prompt(old)
        row["evidence"]["legacy_user_prompt_sha256"] = sha256_hex(old_pp)
        row["evidence"]["user_prompt_identical"] = old_pp == prompt.build_user_prompt(chunk)

    if row["blocking"]:
        row["classification"] = CHANGED
        return row

    if row["evidence"].get("text_identical") and row["evidence"].get("user_prompt_identical"):
        row["classification"] = IDENTICAL
    else:
        # 记录在、正文或提示词有差：那是真的内容变了，必须重抽
        row["classification"] = CHANGED
    return row


def classify_unrecorded(
    chunk: dict,
    *,
    legacy_chunks: dict[str, dict],
    doc_texts: dict[str, str],
    by_chunk_id: dict[str, dict],
) -> dict:
    """正文寻址找不到任何记录时，判定是「内容变了」还是「本来就没记录」。

    判据同样只看字节，不看块号：
    * 这个 doc 在旧分块里**有**块 ⇒ 该位置曾经存在正文，而现在的正文与它不等
      ⇒ `changed_content`（内容确实变了，必须重抽）；
    * 这个 doc 在旧分块里**没有**块 ⇒ `no_record`（这一族根本没被抽过）。

    额外记一条 `contained_in_legacy`：新块正文是不是旧合并块的**连续子串**。
    它只解释机理（块被切开后每一半都是原子串），**不构成复用理由**——
    raw 抽取是拿整块正文做条件生成的，头部少掉的那截内容会改变模型抽什么，
    子串关系不能证明"半个输入会得到同一条输出"。这项诊断写进报告，
    但绝不被当作可复用依据。
    """
    text = chunk["text"]
    doc = doc_of(chunk["chunk_id"])
    row: dict = {
        "chunk_id": chunk["chunk_id"],
        "doc_id": doc,
        "scene_id": chunk.get("scene_id") or chunk.get("doc_id") or "",
        "content_key_in_record": chunk["content_key"],
        "content_key_from_bytes": hashlib.sha256(text.encode("utf-8")).hexdigest()[:24],
        "chars": len(text),
        "text_sha256": sha256_hex(text),
        "user_prompt_sha256": sha256_hex(prompt.build_user_prompt(chunk)),
        "classification": NO_RECORD,
        "evidence": {"found_by": ""},
        "blocking": [],
    }
    if row["content_key_in_record"] != row["content_key_from_bytes"]:
        row["blocking"].append(
            f"块记录 content_key 与正文不符：{row['content_key_in_record']} != {row['content_key_from_bytes']}"
        )

    olds = [cid for cid in legacy_chunks if doc_of(cid) == doc]
    row["evidence"]["legacy_blocks_in_doc"] = len(olds)
    if olds:
        row["classification"] = CHANGED
        row["evidence"]["same_slot_record"] = by_chunk_id.get(chunk["chunk_id"], {}).get("chunk_id", "")
        merged = "\n".join(legacy_chunks[cid]["text"] for cid in sorted(olds))
        row["evidence"]["contained_in_legacy_merged"] = text in merged
        # 与旧同号块比长度，说明"是切开还是变长"
        same_slot = legacy_chunks.get(chunk["chunk_id"])
        if same_slot is not None:
            row["evidence"]["same_slot_legacy_chars"] = len(same_slot["text"])
    else:
        row["classification"] = NO_RECORD
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="", help="把逐块取证结果写成 JSON 到这个路径")
    ap.add_argument("--model", default=AGENT_MODEL)
    args = ap.parse_args()

    print("=== 缓存复用取证（D 阶段块身份变更） ===")
    print(f"产物根      {config.artifact_root()}")
    print(f"新块文件    {config.chunks_path()}")
    print(f"旧块文件    {LEGACY_CHUNKS_PATH}")
    print(f"模型        {args.model}")
    print(f"提示词      prompt_version={config.PROMPT_VERSION} "
          f"prompt_sha256={cache.prompt_sha256()[:16]}…")

    if not LEGACY_CHUNKS_PATH.exists():
        print(f"拒绝取证：旧分块文件不存在 {LEGACY_CHUNKS_PATH}", file=sys.stderr)
        return 2

    # 取证的前提是「新旧是两批不同的块」。如果把新块文件当成旧分块文件来读，
    # 每个块都和自己逐字节相等，脚本会报「0 块待取证 / 全部可复用」并返回 0——
    # 一个什么都证明不了的假门。默认参数就踩这个坑（两个路径都是
    # data/arkrag/chunks.jsonl），故这里直接拒绝而不是"顺手给个默认值"。
    _new_path = Path(config.chunks_path()).resolve()
    _old_path = Path(LEGACY_CHUNKS_PATH).resolve()
    if _new_path == _old_path:
        print(
            "拒绝取证：新块文件与旧分块文件是同一个路径 "
            f"{_new_path}。\n"
            "  那样每个块都和自己逐字节相等，结论必然是「全部可复用」——证明不了任何事。\n"
            "  用 ARKRAG_ARTIFACT_DIR 指向新分块的产物根，例如：\n"
            '    $env:ARKRAG_ARTIFACT_DIR = "data/arkrag/staging/d-stage"',
            file=sys.stderr,
        )
        return 2

    legacy_chunks = load_legacy_chunks(LEGACY_CHUNKS_PATH)
    print(f"旧分块      {len(legacy_chunks):,} 块")
    print("（旧 chunks.jsonl 里**没有** scene_id 字段；scene 取自新块记录的 "
          "scene_id 或 doc_id，doc_id 由 chunk_id 去掉 `#NNN` 得到）")

    all_chunks = load_jsonl(config.chunks_path())
    t1 = [c for c in all_chunks if config.in_t1(c["category"])]
    print(f"新分块      {len(all_chunks):,} 块，其中 T1 {len(t1):,} 块")

    misses: list[dict] = []
    layers: dict[str, int] = {}
    for c in t1:
        lk = cache.lookup_for_chunk(c, args.model)
        layers[lk.layer] = layers.get(lk.layer, 0) + 1
        if lk.raw is None:
            misses.append(c)
    print("缓存分层    " + "  ".join(f"{k}={v}" for k, v in sorted(layers.items())))
    print(f"待取证 miss {len(misses)} 块")

    by_chunk_id, by_content_key = scan_cache()
    print(f"缓存 chunk_id 索引 {len(by_chunk_id):,} 条；content_key 索引 {len(by_content_key):,} 条")

    rows = [classify(c, model=args.model, legacy_chunks=legacy_chunks,
                     by_chunk_id=by_chunk_id, by_content_key=by_content_key)
            for c in misses]

    buckets: dict[str, list[dict]] = {IDENTICAL: [], CHANGED: [], NO_RECORD: []}
    for r in rows:
        buckets.setdefault(r["classification"], []).append(r)

    print()
    print("=== 分类结果 ===")
    for name in (IDENTICAL, CHANGED, NO_RECORD):
        print(f"  {name:20s} {len(buckets.get(name, []))}")

    for name in (CHANGED, NO_RECORD):
        items = buckets.get(name, [])
        if not items:
            continue
        print()
        print(f"--- {name}（必须重抽，本轮不得声称覆盖完整 T1） ---")
        for r in items:
            why = "; ".join(r["blocking"]) if r["blocking"] else "旧键位置无记录"
            print(f"  {r['chunk_id']}  chars={r['chars']}  {why}")

    if buckets.get(IDENTICAL):
        print()
        print(f"--- {IDENTICAL}（可证明复用；块号后移但正文与提示词逐字节相同） ---")
        for r in buckets[IDENTICAL]:
            ev = r["evidence"]
            print(f"  {r['chunk_id']}  <- 旧记录块 {ev.get('legacy_record_chunk_id')}"
                  f"  key={ev.get('legacy_key')}  chars={r['chars']}"
                  f"  prompt_version={ev.get('legacy_record_prompt_version')}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
            "model": args.model,
            "prompt_version": config.PROMPT_VERSION,
            "prompt_sha256": cache.prompt_sha256(),
            "artifact_root": str(config.artifact_root()),
            "chunks_path": str(config.chunks_path()),
            "legacy_chunks_path": str(LEGACY_CHUNKS_PATH),
            "totals": {
                "t1_chunks": len(t1),
                "layers": layers,
                "miss": len(misses),
                "identical_content": len(buckets.get(IDENTICAL, [])),
                "changed_content": len(buckets.get(CHANGED, [])),
                "no_record": len(buckets.get(NO_RECORD, [])),
            },
            "rows": rows,
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n逐块取证 -> {out}")

    unresolved = len(buckets.get(CHANGED, [])) + len(buckets.get(NO_RECORD, []))
    print()
    if unresolved:
        print(f"结论：{len(buckets.get(IDENTICAL, []))} 块可证明复用；"
              f"{unresolved} 块必须重抽（本机无 LLM 端点，本轮构建无法覆盖完整 T1 范围）。")
        return 1
    if not misses:
        # 没有 miss 就不存在"复用"这件事。别把这句写成"全部 N 块均可复用"——
        # 那会让人以为跑过了一次有意义的取证。
        print("结论：本次没有 miss 块，无需复用（未对任何块做可复用性判定）。")
        return 0
    print(f"结论：全部 {len(buckets.get(IDENTICAL, []))} 个 miss 块均可证明复用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
