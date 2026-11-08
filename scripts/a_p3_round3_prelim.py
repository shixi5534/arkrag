"""P3 第三轮方案的前置测量（只读，不改生产代码）。

三个靶子各出一组数，全部在 dev 380（应答题 350）上：

1. **闸门1（required 那一行没被抽成断言）**：按类拆开，并回答结构问题——
   这 120 道里有多少是「抽取器漏了能抽的东西」，有多少是「gold 挑的那一行本来就不该
   被抽成断言」（后者问题在 gold，不归 P3 动）。

2. **闸门2（required 块没进上下文）**：按 gold 名次拆成「检索层无解（名次 > 20）」与
   「容量层（名次 ≤ 5 但被 token 预算截断）」两半，供 Lead 判断 P2 还值不值得投。

3. **`negative_abstain` 拒答缺口**：先把两条候选判据各自做 paraphrase 反向测试，
   跑不过的自己否掉，不往上报。

退出码：0 跑通 / 2 输入不可用 / 4 自检失败。
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import platform
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkrag import config, pipeline, query as query_mod, retrieve, subproc, vocab  # noqa: E402

DEFAULT_GEN = "g-20261103T132152Z-06a741"
DEFAULT_GOLD = ROOT / "data" / "goldset" / "gold-v2.json"

EVAL_TOP_K = 20
EVAL_GRAPH_DEPTH = 1

# 引用关系词（用于判「这一行是不是关系陈述」）。取全图谓词表 + 关系同义词的键。
RELATION_WORDS = sorted(set(vocab.CANONICAL_RELATIONS) | set(vocab.RELATION_SYNONYMS))


def rel(path: Path) -> str:
    return config.rel_to_root(path)


def git_rev() -> str:
    try:
        import subprocess

        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
            capture_output=True, text=True, timeout=20, check=False,
            env=subproc.child_env(),
        )
        return (out.stdout or "").strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def load_dev(gold_path: Path) -> list[dict]:
    doc = json.loads(gold_path.read_text(encoding="utf-8"))
    rows = [it for it in doc["items"] if it.get("split") == "dev"]
    rows.sort(key=lambda it: it["question_id"])
    return rows


def answerable_of(items: list[dict]) -> list[dict]:
    return [it for it in items
            if not it.get("should_abstain") and (it.get("gold_chunk_ids") or [])]


def deep_search(retriever, qa: dict, pool: int) -> dict:
    """与生产 `search(top_k=20)`（depth=200）同源的深池。

    必须把 `CANDIDATE_POOL_MULT` 压到 1 再传 top_k=pool，否则 depth 会变成
    `pool * 10`，三路候选集与 `_norm_scores()` 的归一分母都漂移（本项目踩过两次）。
    """
    old = retrieve.CANDIDATE_POOL_MULT
    retrieve.CANDIDATE_POOL_MULT = 1
    try:
        return retriever.search(qa, top_k=pool)
    finally:
        retrieve.CANDIDATE_POOL_MULT = old


# ---------------------------------------------------------------- 闸门1


def gate1_profile(item: dict, chunks: dict, by_chunk: dict[str, list[dict]],
                  entity_names: dict[str, str]) -> dict:
    """把一道闸门1题的 required 逐条画像：那一行到底能不能被抽成断言。"""
    out = []
    for rc in item.get("required_citations") or []:
        cid = str(rc.get("chunk_id") or "")
        req_ev = str(rc.get("evidence") or "")
        rec = chunks.get(cid) or {}
        text = str(rec.get("text") or "")
        asserts = by_chunk.get(cid) or []

        # required evidence 是否真能逐字定位到块正文
        pos = text.find(req_ev) if req_ev else -1
        locatable = pos >= 0
        # 它所在的整行
        if locatable:
            left = text.rfind("\n", 0, pos) + 1
            right = text.find("\n", pos + len(req_ev))
            right = len(text) if right < 0 else right
            line = text[left:right]
        else:
            line = ""

        # 这一行被抽成断言了吗？分三态，不能把「行内只要有断言」当成已覆盖：
        #   exact —— 某条断言的 evidence_text 含 required evidence（这才是 span 可达，本不该进闸门1）
        #   frag  —— 行内有断言，但断言只是行的一小段，required 那句完整的话没被覆盖
        #   none  —— 整行没有任何断言
        # 闸门1 的定义是「required evidence 不是任何断言 evidence_text 的子串」，所以落在
        # 这里的只可能是 frag 或 none。**初版把「行内已有断言」单列成一类并注「不该算闸门1」，
        # 那是错的**：行内有断言 ≠ 那句话被抽到了。已改。
        line_hit_assertions = []
        exact_hit = False
        for rec_a in asserts:
            a_txt = str(rec_a.get("evidence_text") or "")
            if req_ev and req_ev in a_txt:
                exact_hit = True
            a_start = int(rec_a.get("evidence_start") or -1)
            a_end = int(rec_a.get("evidence_end") or -1)
            on_line = False
            if locatable and 0 <= a_start <= a_end <= len(text):
                l_end = left + len(line)
                if not (a_end <= left or a_start >= l_end):
                    on_line = True
            if not on_line and line and a_txt and a_txt in line:
                on_line = True
            if on_line:
                line_hit_assertions.append(rec_a)

        if exact_hit:
            span_state = "exact（本不该进闸门1）"
        elif line_hit_assertions:
            span_state = "frag（断言只覆盖行的一部分）"
        else:
            span_state = "none（整行无断言）"

        # 这一行里出现的已知实体名（用于判「能不能被抽」）
        line_entities = sorted(
            {nid for name, nid in entity_names.items() if name and name in line})
        # 这一行里出现的关系词
        line_rels = [w for w in RELATION_WORDS if w and w in line]
        # 说话人前缀（对话体）
        speaker = line.split("：", 1)[0] if "：" in line[:20] else ""

        if not locatable:
            kind = "required evidence 在块正文里定位不到"
        elif len(line_entities) >= 2 and line_rels:
            kind = "实体≥2 且含关系词 → 抽取器漏了"
        elif len(line_entities) >= 2:
            kind = "实体≥2 但无关系词 → 关系未成词"
        elif len(line_entities) == 1:
            kind = "仅 1 个实体 → 结构上抽不成（主-谓-宾缺一侧）"
        elif line_rels:
            kind = "无已知实体但含关系词 → 实体未识别"
        else:
            kind = "无实体无关系词 → 不是关系陈述（gold 侧问题）"

        # 「实体≥2 但无关系词」是灰区：行内两个实体都有、却没有 48 谓词里的关系词。
        # 拆开它才能回答「抽取器漏了」还是「这行本就不该被抽成断言」：
        #   pair_extracted —— 块内存在一条断言，其 subject/object 恰好就是行内那对实体
        #                     ⇒ 抽取器抽到了这层关系，只是证据落在别的行；gold 挑行的
        #                       问题（P1-c 已裁决，不改 gold，只记录）
        #   pair_missing   —— 块内没有任何断言连接这对实体 ⇒ 抽取器确实漏了这一层
        pair_extracted = None
        pair_assertion = None
        if len(line_entities) >= 2:
            want = set(line_entities)
            for rec_a in asserts:
                s_id = str(rec_a.get("subject_id") or "")
                o_id = str(rec_a.get("object_id") or "")
                if s_id in want and o_id in want and s_id != o_id:
                    pair_extracted = True
                    pair_assertion = {
                        "assertion_id": str(rec_a.get("assertion_id") or ""),
                        "predicate": str(rec_a.get("predicate") or ""),
                        "evidence_text": str(rec_a.get("evidence_text") or "")[:80],
                    }
                    break
            if pair_extracted is None:
                pair_extracted = False

        out.append({
            "chunk_id": cid,
            "required_len": len(req_ev),
            "locatable": locatable,
            "line": line[:160],
            "line_len": len(line),
            "speaker": speaker,
            "span_state": span_state,
            "n_line_entities": len(line_entities),
            "line_entities": line_entities[:6],
            "line_rels": line_rels[:6],
            "pair_extracted": pair_extracted,
            "pair_assertion": pair_assertion,
            "n_assertions_in_chunk": len(asserts),
            "n_assertions_on_line": len(line_hit_assertions),
            "kind": kind,
        })
    return {"required": out}


# ---------------------------------------------------------------- 闸门2


def make_qa(question: str, retriever, entities: list[dict]) -> dict:
    """按 `pipeline.ask_once` 的同一调用方式解析问句。

    **不能只传 question**：`query.analyze_query(text, *, entities, aliases=None, vocab=None,
    context=None)` 的 `entities` 是**必填 keyword-only**（它是实体/别名的唯一权威来源），
    漏传直接 `TypeError`。也不能自己造 `QueryContext`——那会退化成每问重建图级结构
    （约 98 ms/问）。用 `retriever.query_context` 复用只读结构。
    """
    return query_mod.analyze_query(
        question,
        entities=entities,
        aliases=retriever.graph.get("alias_index"),
        vocab=getattr(retriever, "vocab", None),
        context=getattr(retriever, "query_context", None),
    )


def gate2_profile(item: dict, retriever, entities: list[dict], ctx_ids: set[str],
                  pool: int) -> dict:
    qa = make_qa(str(item.get("question") or ""), retriever, entities)
    deep = deep_search(retriever, qa, pool)
    hits = deep.get("hits") or []
    ranks = {}
    for i, h in enumerate(hits, 1):
        ranks.setdefault(str(h.get("chunk_id") or ""), i)
    gold = [str(c) for c in item.get("gold_chunk_ids") or []]
    gold_ranks = [ranks.get(c) for c in gold]
    found = [(c, r) for c, r in zip(gold, gold_ranks) if r is not None]
    best = min((r for _, r in found), default=None)
    required = [str(rc.get("chunk_id") or "") for rc in item.get("required_citations") or []]
    req_ranks = [ranks.get(c) for c in required]
    req_found = [r for r in req_ranks if r is not None]
    return {
        "question_id": item["question_id"],
        "category": str(item.get("category") or ""),
        "gold_ranks": gold_ranks,
        "gold_best_rank": best,
        "required_ranks": req_ranks,
        "required_best_rank": min(req_found) if req_found else None,
        "required_in_pool": all(r is not None for r in req_ranks),
        "required_best_in_top5": bool(req_found) and min(req_found) <= 5,
        "required_best_in_top20": bool(req_found) and min(req_found) <= 20,
        "required_in_context": all(c in ctx_ids for c in required),
        "pool_size": len(hits),
    }


# ---------------------------------------------------------------- 拒答判据反测


def paraphrase_table() -> dict[str, list[str]]:
    """每个规范关系词的**表外同义说法**（不给答案词本身）。

    这是 paraphrase 反向测试的核心素材：应答题的答案换成表外说法后，任何「必须命中
    关系词」的判据都会把本来能答的题判死。
    """
    return {
        "位于": ["坐落在", "地处", "就在"],
        "交战": ["打过仗", "动过手", "交过火"],
        "交谈": ["聊过天", "说过话", "谈过"],
        "隶属": ["归属", "归", "受辖于"],
        "同盟": ["结盟", "站在一边", "联手"],
        "同行": ["一起走", "结伴而行", "同路"],
        "同行于": ["一起走", "结伴而行"],
        "背叛": ["背弃", "反水", "出卖"],
        "师从": ["从师", "拜师", "跟着学"],
        "源自": ["来自", "起源于", "出自"],
        "交好": ["关系好", "亲近"],
        "敌对": ["对立", "为敌", "势不两立"],
        "保护": ["护着", "守护", "庇护"],
        "使用": ["用", "动用了", "使唤"],
        "持有": ["拿着", "随身带", "握着"],
        "创建": ["创立", "建立", "创办"],
        "制造": ["造出", "做出", "生产"],
        "研究": ["钻研", "探究"],
        "破坏": ["毁掉", "砸了", "损毁"],
        "救助": ["救下", "搭救", "帮了"],
        "治愈": ["治好了", "医好"],
        "感染": ["染上", "被感染"],
        "指导": ["指点", "带教"],
        "指挥": ["率领", "统率"],
        "效忠": ["忠于", "尽忠"],
        "任职": ["在任", "供职", "担任"],
        "出生于": ["生在", "出身于"],
        "发生于": ["发生在", "出现在"],
        "产自": ["出产自", "出产于"],
        "包含": ["含", "包括"],
        "容纳": ["装得下", "内置"],
        "导致": ["引起", "造成", "引发"],
        "属于": ["归", "算作"],
        "接受": ["收下", "接下"],
        "称为": ["叫做", "被称为"],
        "前往": ["去到", "赶赴", "去了"],
        "离开": ["走了", "撤离", "离了"],
        "亲属": ["亲戚", "家里人"],
        "下属": ["手下", "下级"],
        "听命于": ["受命于", "遵命于"],
        "受保护于": ["被护着", "受庇护"],
        "受赠于": ["收到赠礼于"],
        "冲突": ["起冲突", "有摩擦"],
        "参与": ["参加了", "涉入"],
    }


RELATION_ASK_RE = re.compile(r"「([^」]{2,12})」(?:这层)?关系")
BETWEEN_RE = re.compile(r"「([^」]+)」(?:与|和)「([^」]+)」")


def is_quoted_relation_question(question: str) -> tuple[bool, str]:
    m = RELATION_ASK_RE.search(str(question or ""))
    return (bool(m), m.group(1) if m else "")


def endpoints_of(question: str) -> tuple[str, str] | None:
    m = BETWEEN_RE.search(str(question or ""))
    return (m.group(1), m.group(2)) if m else None


# ---------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P3 第三轮前置测量（只读）")
    ap.add_argument("--generation", default=DEFAULT_GEN)
    ap.add_argument("--gold-set", default=str(DEFAULT_GOLD))
    ap.add_argument("--pool", type=int, default=200)
    ap.add_argument("--out-json", default=str(ROOT / "data" / "a_p3_round3_prelim.json"))
    args = ap.parse_args(argv)

    gold_path = Path(args.gold_set)
    if not gold_path.is_absolute():
        gold_path = ROOT / gold_path
    if not gold_path.exists():
        print(f"[2] 输入不可用：{args.gold_set}", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    loaded = pipeline.load_generation(args.generation)
    chunks = loaded["chunks"]
    by_chunk: dict[str, list[dict]] = collections.defaultdict(list)
    for rec in loaded["assertions"]:
        by_chunk[str(rec.get("chunk_id") or "")].append(rec)
    entities = loaded["graph"].get("entities") or []
    entity_names = {str(e.get("name") or ""): str(e.get("id") or "") for e in entities
                    if len(str(e.get("name") or "")) >= 2}
    pipe = pipeline.build_pipeline(loaded, top_k=EVAL_TOP_K, graph_depth=EVAL_GRAPH_DEPTH)
    retriever = pipe["retriever"]

    items = load_dev(gold_path)
    answerable = answerable_of(items)
    print(f"[load] dev={len(items)} 应答题={len(answerable)} 块={len(chunks)} "
          f"断言={len(loaded['assertions'])} 实体={len(entities)}")

    # ---------------- 跑一遍真管线，拿上下文与引用 ----------------
    # **必须跑全部 dev 380 题，不能只跑应答题 350**：拒答判据要在「应拒题」上测触发率，
    # 只跑应答题会让 `should_abstain` 子集恒为空（初版就栽在这里：判据 A 报「应拒共 0」）。
    from arkrag import answer as answer_mod
    rows = []
    for it in items:
        out = pipeline.ask_once(pipe, str(it.get("question") or ""))
        ctx = out.get("context") or {}
        ans = out.get("answer") or {}
        ctx_ids = {str(u.get("chunk_id") or "") for u in ctx.get("units") or []}
        reqs = it.get("required_citations") or []
        emitted = {(str(c.get("chunk_id") or ""), str(c.get("evidence") or ""))
                   for c in ans.get("citations") or []}
        covered = bool(reqs) and all(
            any(cid == str(r.get("chunk_id") or "") and str(r.get("evidence") or "") in ev
                for cid, ev in emitted) for r in reqs)
        span_ok = bool(reqs) and all(
            any(str(r.get("evidence") or "") in str(a.get("evidence_text") or "")
                for a in by_chunk.get(str(r.get("chunk_id") or "")) or [])
            for r in reqs)
        rows.append({
            "item": it,
            "answerable": (not it.get("should_abstain")) and bool(it.get("gold_chunk_ids")),
            "covered": covered,
            "span_reachable": span_ok,
            "required_in_context": bool(reqs) and all(
                str(r.get("chunk_id") or "") in ctx_ids for r in reqs),
            "ctx_ids": ctx_ids,
            "ctx_stats": ctx.get("stats") or {},
            "answer": str(ans.get("answer") or ""),
            "abstained": bool(ans.get("abstained")),
            "citations": ans.get("citations") or [],
        })
    answer_rows = [r for r in rows if r["answerable"]]
    print(f"[run] 真管线 {len(rows)} 题（应答题 {len(answer_rows)}）"
          f"{time.perf_counter()-t0:.1f}s")

    # ---------------- 闸门分区（只在应答题上做） ----------------
    gate = {"已覆盖": [], "闸门1": [], "闸门2": [], "闸门3": []}
    for r in answer_rows:
        if r["covered"]:
            gate["已覆盖"].append(r)
        elif not r["span_reachable"]:
            gate["闸门1"].append(r)
        elif not r["required_in_context"]:
            gate["闸门2"].append(r)
        else:
            gate["闸门3"].append(r)
    gate_sizes = {k: len(v) for k, v in gate.items()}
    print(f"[闸门] {gate_sizes}")
    if sum(gate_sizes.values()) != len(answer_rows):
        print(f"[4] 闸门分区不闭合：{sum(gate_sizes.values())} != {len(answer_rows)}",
              file=sys.stderr)
        return 4

    # ---------------- 靶子 1：闸门1 结构分类 ----------------
    g1_rows = []
    for r in gate["闸门1"]:
        prof = gate1_profile(r["item"], chunks, by_chunk, entity_names)
        kinds = collections.Counter(x["kind"] for x in prof["required"])
        g1_rows.append({
            "question_id": r["item"]["question_id"],
            "category": str(r["item"].get("category") or ""),
            "n_required": len(prof["required"]),
            "kinds": dict(kinds),
            "required": prof["required"],
        })
    g1_kind_total = collections.Counter()
    g1_kind_by_cat: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for row in g1_rows:
        for k, n in row["kinds"].items():
            g1_kind_total[k] += n
            g1_kind_by_cat[row["category"]][k] += n
    print("[靶子1 闸门1] 逐条 required 画像（条级）")
    for k, n in g1_kind_total.most_common():
        print(f"  {n:4d}  {k}")

    # 「实体≥2 但无关系词」的灰区再拆：块内是否已有连接这对实体的断言
    grey = [rq for row in g1_rows for rq in row["required"]
            if rq["kind"] == "实体≥2 但无关系词 → 关系未成词"]
    grey_extracted = [rq for rq in grey if rq.get("pair_extracted")]
    grey_missing = [rq for rq in grey if rq.get("pair_extracted") is False]
    print(f"[靶子1 灰区] 「实体≥2 但无关系词」{len(grey)} 条，其中：")
    print(f"   {len(grey_extracted)} 条 块内已有连接这对实体的断言（抽取器抽到了，"
          f"证据落在别的行 → gold 挑行问题，P1-c 已裁决不改）")
    print(f"   {len(grey_missing)} 条 块内无任何断言连接这对实体（抽取器真漏了这一层）")
    grey_by_cat = collections.defaultdict(lambda: [0, 0])
    for row in g1_rows:
        for rq in row["required"]:
            if rq["kind"] != "实体≥2 但无关系词 → 关系未成词":
                continue
            if rq.get("pair_extracted"):
                grey_by_cat[row["category"]][0] += 1
            else:
                grey_by_cat[row["category"]][1] += 1
    print("[靶子1 灰区] 按类（已有断言 / 真漏）")
    for c in sorted(grey_by_cat):
        a, b = grey_by_cat[c]
        print(f"   {c}: {a} / {b}（共 {a+b}）")

    # 题级：一道题按「最坏一条 required」定性
    def worst_kind(kinds: dict) -> str:
        order = ["required evidence 在块正文里定位不到",
                 "无实体无关系词 → 不是关系陈述（gold 侧问题）",
                 "仅 1 个实体 → 结构上抽不成（主-谓-宾缺一侧）",
                 "无已知实体但含关系词 → 实体未识别",
                 "实体≥2 但无关系词 → 关系未成词",
                 "实体≥2 且含关系词 → 抽取器漏了"]
        for k in order:
            if kinds.get(k):
                return k
        return "未知"
    g1_q = collections.Counter()
    for row in g1_rows:
        row["worst"] = worst_kind(row["kinds"])
        g1_q[row["worst"]] += 1
    print("[靶子1 闸门1] 题级定性（按最坏一条 required）")
    for k, n in g1_q.most_common():
        print(f"  {n:4d}  {k}")

    # span 三态分布（条级）：注意**条级 exact 出现在闸门1里是正常的**——闸门1 是题级
    # 判定（`all(...)`：只要有一条 required 不可达，整题进闸门1），同一题里其它 required
    # 条目可以是 exact。真正的不变量是「每道闸门1题至少有一条 non-exact 的 required」。
    g1_span = collections.Counter()
    for row in g1_rows:
        for rq in row["required"]:
            g1_span[rq["span_state"]] += 1
    print("[靶子1 闸门1] required 条级 span 三态（条级 exact 出现是正常的，见注释）")
    for k, n in g1_span.most_common():
        print(f"  {n:4d}  {k}")
    bad = [row["question_id"] for row in g1_rows
           if not any(rq["span_state"] != "exact（本不该进闸门1）" for rq in row["required"])]
    if bad:
        print(f"[4] 这些闸门1题的 required 全是 exact：闸门分区口径有漏 {bad[:5]}",
              file=sys.stderr)
        return 4
    # 题级：至少一条 none 的题数（「整行都没被抽」是更硬的缺口）
    q_all_none = [row["question_id"] for row in g1_rows
                  if all(rq["span_state"] == "none（整行无断言）" for rq in row["required"])]
    print(f"[靶子1 闸门1] 全部 required 行都没有任何断言的题：{len(q_all_none)} 道")

    # ---------------- 靶子 2：闸门2 名次拆分 ----------------
    g2_rows = []
    for r in gate["闸门2"]:
        prof = gate2_profile(r["item"], retriever, entities, r["ctx_ids"], args.pool)
        prof["tokens_used"] = int((r["ctx_stats"] or {}).get("tokens_used") or 0)
        prof["token_budget"] = int((r["ctx_stats"] or {}).get("token_budget") or 0)
        prof["units"] = int((r["ctx_stats"] or {}).get("units") or 0)
        prof["truncated_chunks"] = list((r["ctx_stats"] or {}).get("truncated_chunks") or [])
        reqs = [str(rc.get("chunk_id") or "") for rc in r["item"].get("required_citations") or []]
        prof["required_in_truncated"] = [c for c in reqs if c in prof["truncated_chunks"]]
        # 定性：检索层无解 / 容量层 / 其他
        best = prof["required_best_rank"]
        if best is None:
            prof["bucket"] = "检索层：required 块不在深池 200 内"
        elif best > 20:
            prof["bucket"] = "检索层：名次 > 20（前 20 都没进）"
        elif best <= 5 and prof["required_in_truncated"]:
            prof["bucket"] = "容量层：名次 ≤5 但被 token 预算截断"
        elif best <= 5 and not prof["required_in_context"]:
            prof["bucket"] = "容量层以外：名次 ≤5 未进上下文但不在 truncated 名单"
        elif best <= 20 and not prof["required_in_truncated"]:
            prof["bucket"] = "装配层：名次 ≤20 但未进上下文（非预算截断）"
        else:
            prof["bucket"] = "容量层：名次 6–20 且被预算截断"
        g2_rows.append(prof)
    g2_b = collections.Counter(r["bucket"] for r in g2_rows)
    g2_b_cat: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in g2_rows:
        g2_b_cat[r["category"]][r["bucket"]] += 1
    print("\n[靶子2 闸门2] 名次/容量拆分（题级）")
    for k, n in g2_b.most_common():
        print(f"  {n:4d}  {k}")
    print("[靶子2 闸门2] 按类")
    for cat in sorted(g2_b_cat):
        print(f"  {cat}: {dict(g2_b_cat[cat])}")

    # ---------------- 靶子 3：拒答判据的 paraphrase 反测 ----------------
    para = paraphrase_table()
    canon = set(vocab.CANONICAL_RELATIONS)
    # 判据 A：问句关系词不在图谓词表 → 拒答
    na_should = [r for r in rows if r["item"].get("should_abstain")]
    na_answer = [r for r in rows if r["answerable"]]
    dev_items = [it for it in items if it.get("should_abstain")]

    hit_should, hit_answer = [], []
    for r in na_should:
        ok, w = is_quoted_relation_question(str(r["item"].get("question") or ""))
        if ok and w not in canon:
            hit_should.append(r["item"]["question_id"])
    for r in na_answer:
        ok, w = is_quoted_relation_question(str(r["item"].get("question") or ""))
        if ok and w not in canon:
            hit_answer.append(r["item"]["question_id"])

    # paraphrase 反测：把应答题的 gold 答案换成表外同义说法，再套判据 A
    para_killed, para_built = [], 0
    for r in na_answer:
        it = r["item"]
        gold = str(it.get("answer") or "")
        if gold not in para:
            continue
        between = endpoints_of(str(it.get("question") or ""))
        if not between:
            continue
        for alias in para[gold]:
            para_built += 1
            # 构造「同义说法」版问句：把答案词换成表外说法，句式不变
            q = str(it.get("question") or "").replace(gold, alias, 1)
            ok, w = is_quoted_relation_question(q)
            if not ok or w not in canon:
                para_killed.append({
                    "question_id": it["question_id"],
                    "gold_relation": gold,
                    "alias": alias,
                    "constructed_question": q,
                })
                break

    # 判据 B：claim 至少要含一个问句端点（抗「捞无关 claim 就交卷」）
    # 反测：多跳题的答案是中间实体，本就不在问句里 —— 看会误杀多少
    q_entities: dict[str, list[str]] = {}
    claim_miss = []
    for r in rows:
        q = str(r["item"].get("question") or "")
        qa = make_qa(q, retriever, entities)
        names = [str(c.get("name") or "") for c in (qa.get("entity_candidates") or [])
                 if str(c.get("name") or "")]
        q_entities[r["item"]["question_id"]] = names
        if not r["answer"]:
            continue
        if names and not any(n and n in r["answer"] for n in names):
            claim_miss.append({
                "question_id": r["item"]["question_id"],
                "category": str(r["item"].get("category") or ""),
                "should_abstain": bool(r["item"].get("should_abstain")),
                "covered": r["covered"],
                "answer": r["answer"][:120],
                "question_entities": names[:4],
            })
    miss_by_cat = collections.Counter(x["category"] for x in claim_miss)

    doc = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_rev(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "command": "py -3 scripts/a_p3_round3_prelim.py",
        "generation": args.generation,
        "gold_set": rel(gold_path),
        "split_policy": "pre-split-diagnostic/dev-only：dev 结论不外推到 new-holdout 160",
        "counts": {"dev": len(items), "answerable": len(answerable),
                   "should_abstain": len(dev_items)},
        "gates": gate_sizes,
        "target1_gate1": {
            "questions": len(g1_rows),
            "kinds_item_level": dict(g1_q),
            "kinds_required_level": dict(g1_kind_total),
            "kinds_by_category": {k: dict(v) for k, v in g1_kind_by_cat.items()},
            "per_question": g1_rows,
        },
        "target2_gate2": {
            "questions": len(g2_rows),
            "buckets": dict(g2_b),
            "buckets_by_category": {k: dict(v) for k, v in g2_b_cat.items()},
            "per_question": g2_rows,
        },
        "target3_abstain": {
            "criterion_A": {
                "name": "问句关系词不在图 48 谓词表 → 拒答",
                "fires_on_should_abstain": hit_should,
                "fires_on_answerable": hit_answer,
                "paraphrase_built": para_built,
                "paraphrase_killed": para_killed,
                "paraphrase_killed_n": len(para_killed),
            },
            "criterion_B": {
                "name": "claim 至少要含一个问句端点",
                "miss_by_category": dict(miss_by_cat),
                "miss_total": len(claim_miss),
                "would_wrongly_abstain_answerable": sum(
                    1 for x in claim_miss if not x["should_abstain"]),
                "per_question": claim_miss,
            },
        },
        "wall_seconds": round(time.perf_counter() - t0, 2),
    }
    out = Path(args.out_json)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[靶子3 判据A] 问句关系词不在谓词表 → 拒答")
    print(f"  在应拒题上触发 {len(hit_should)} 道（应拒共 {len(na_should)}）")
    print(f"  在应答题上触发 {len(hit_answer)} 道（应答题共 {len(na_answer)}）")
    print(f"  paraphrase 反测：构造 {para_built} 条同义问句，被判据打死 "
          f"{len(para_killed)} 道题")
    if para_killed[:3]:
        for x in para_killed[:3]:
            print(f"    {x['question_id']} {x['gold_relation']}→{x['alias']}: "
                  f"{x['constructed_question'][:70]}")

    print("\n[靶子3 判据B] claim 必须含一个问句端点")
    print(f"  当前答案不含任何问句端点的题 {len(claim_miss)} 道，按类 {dict(miss_by_cat)}")
    print(f"  其中本来**应答题**（会被误拒）{sum(1 for x in claim_miss if not x['should_abstain'])} 道")

    print(f"\n输出: {rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
