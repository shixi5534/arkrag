"""检索问答管线的装配与单问执行——**唯一**一份实现。

为什么把它从 `scripts/ask.py` 搬进包内：J 阶段要加一个 HTTP 服务，服务必须
和命令行用**同一条**链路。如果 `scripts/serve.py` 自己写一遍"读哪些产物、
缺哪个算不合格、按什么顺序调 retriever/answerer"，那么"哪些产物是必需的"
这件事就有两份定义——本项目已经因为"同一判定写两处、取最松的那份生效"
出过三次同型事故（见 `tests/test_single_source_of_truth.py`）。

所以这里只做**搬移**，不改任何判定：

* `load_generation()` 的产物清单与检查顺序原样保留；
* `build_pipeline()` 的构造参数原样保留；
* `ask_once()` 的分支与调用顺序原样保留（`timings` 仍是可选旁路）。

`scripts/ask.py` 改成从这里 import 同名函数并继续对外暴露（`ask.load_generation`
等名字不变），所以既有的 `scripts/bench_h1.py` / `scripts/bench_h3.py` 里
`ask_mod.load_generation = …` 这类 monkeypatch 仍然生效——它们改的是
`scripts/ask.py` 模块对象上的名字，而 `ask.evaluate()` 是在 `ask.py` 里定义的，
解析的正是该模块的全局名。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import answer as answer_mod
from . import bridge
from . import config
from . import index as index_mod
from . import query as query_mod
from . import quote_locate
from . import release
from . import retrieve as retrieve_mod
from . import vocab


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    try:
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[2] {path} 不是合法 JSONL：{exc}") from exc
    return out


# ---------------------------------------------------------------- 块正文来源
class ChunksUnavailable(RuntimeError):
    """块正文找不到、或找到的那份证明不了是这一代吃进去的那批。

    为什么必须是异常而不是返回 None：块正文是**引用校验的判据**。
    拿错块不会报错——`text[start:end] != evidence` 只会让引用被静默剔除、
    置信度降低，表现为「答得短」。这种形态没人会去查，必须在这里拒。
    """


def resolve_chunks(gen, manifest: dict) -> Path:
    """决定这一代该从哪个文件读块正文，并**证明**那份文件就是它那一批。

    两代形态：

    1. 世代自带 `<gen>/chunks.jsonl`（K2 起的新形态）。直接用它，但仍要核
       `chunk_manifest_hash`——文件在盘上不等于内容对。
    2. 旧世代没有自带块文件。此时**不许默认"全局块文件就是它的"**：全局
       `data/arkrag/chunks.jsonl` 是 L1 的最新产物，下一次 L2 重建就会覆盖它，
       而世代不会跟着变。只有当它的 T1 子集能复现本代 `chunk_manifest_hash`
       时才允许借用——那不是猜测，是同一批块的证明。

    两种情况都证明不了就抛 `ChunksUnavailable`，由调用方决定怎么呈现。
    """
    want = str(manifest.get("chunk_manifest_hash") or "")

    def t1_of(records: list[dict]) -> list[dict]:
        return [r for r in records if config.in_t1(str(r.get("category") or ""))]

    own = gen.chunks_path
    if own.exists():
        records = t1_of(_read_jsonl(own))
        if want:
            got = release.chunk_manifest_hash(records)
            if got != want:
                raise ChunksUnavailable(
                    f"世代自带块正文与 Manifest 不符："
                    f"chunk_manifest_hash Manifest={want[:16]} 实际={got[:16]}"
                    f"（{own}，{len(records)} 块）"
                )
        if not records:
            raise ChunksUnavailable(f"世代自带块正文是空的：{own}")
        return own

    # ---- 旧世代：只有能证明是同一批块时才借全局文件
    shared = Path(config.chunks_path())
    if not shared.exists():
        raise ChunksUnavailable(f"找不到块文件 {shared}（引用校验需要块正文）")
    records = t1_of(_read_jsonl(shared))
    if want:
        got = release.chunk_manifest_hash(records)
        if got != want:
            raise ChunksUnavailable(
                f"这一代的块正文不在 {shared} 里："
                f"chunk_manifest_hash Manifest={want[:16]} 该文件实测={got[:16]}"
                f"（差 {len(records)} 块的那批）。拒绝用错块做引用校验——"
                "请给这一代补上自带的 chunks.jsonl，或从中断点重建。"
            )
    return shared


def load_generation(generation_id: str) -> dict:
    """读出评测/回答需要的全部产物。缺任何一件都抛 `SystemExit(2)`。

    这里刻意**不**做"缺断言就用边顶替"的降级：`ark-rag-3` 的回答要靠逐观察断言
    给出证据偏移，缺了它只能给出没有位置的引用，那种引用无法机检。
    """
    gen = release.Generation(generation_id)
    if not gen.dir.exists():
        raise SystemExit(f"[2] 世代不存在：{generation_id}（{gen.dir}）")
    manifest = gen.load()
    schema = str(manifest.get("graph_schema") or config.GRAPH_SCHEMA_LEGACY)
    if schema != config.GRAPH_SCHEMA_IDENTITY:
        raise SystemExit(
            f"[2] 世代 {generation_id} 的 graph_schema={schema}，"
            f"本工具只支持 {config.GRAPH_SCHEMA_IDENTITY}（ark-rag-1 没有断言，"
            "做不了带方向的扩展与可定位引用）"
        )
    for path, label in (
        (gen.graph_path, "graph/graph.json"),
        (gen.assertion_path, "graph/assertions.jsonl"),
        (gen.adjacency_path, "graph/adjacency.json"),
        (gen.index_path, "index/lexical.json"),
    ):
        if not path.exists():
            raise SystemExit(f"[2] 世代 {generation_id} 缺 {label}（{path}）")

    graph = json.loads(gen.graph_path.read_text("utf-8"))
    adjacency = json.loads(gen.adjacency_path.read_text("utf-8"))
    assertions = _read_jsonl(gen.assertion_path)
    idx, idx_meta = index_mod.load_index(gen.index_path)

    try:
        chunks_path = resolve_chunks(gen, manifest)
    except ChunksUnavailable as exc:
        raise SystemExit(f"[2] {exc}") from exc
    chunks: dict[str, dict] = {}
    for rec in _read_jsonl(chunks_path):
        cid = str(rec.get("chunk_id") or "")
        if cid:
            chunks[cid] = rec

    return {
        "generation_id": generation_id,
        "generation": gen,
        "manifest": manifest,
        "graph": graph,
        "adjacency": adjacency,
        "assertions": assertions,
        "index": idx,
        "index_meta": idx_meta,
        "chunks": chunks,
    }


# ---------------------------------------------------------------- 管线
def is_chapter_question(question: str) -> bool:
    """问句是否落在「问章节代号」这一类（`cross_chunk` 骨架）。

    单一事实来源在 `answer.CHAPTER_TRIGGERS`：这里**只转发**，不另写一份触发词。
    本项目已因「同一判定写两处、取最松的那份生效」出过三次同型事故。
    """
    q = str(question or "")
    return any(t in q for t in answer_mod.CHAPTER_TRIGGERS)


def build_pipeline(loaded: dict, *, top_k: int, graph_depth: int,
                   vector_union=None) -> dict:
    """装配一条问答链。`vector_union` 为 `None` 时**与改造前完全等价**。

    向量通道为什么是可选注入而不是默认开：它需要向量产物 + embedding backend
    两件外部物，缺任何一件都必须显式报错（`EmbeddingError` / 缺产物），不能
    降级成"没有向量也照跑"——那会让"带向量的评测"和"不带向量的评测"长得
    一样，而本项目已因"判据没接线等于不存在"栽过两次。

    默认 `None` 还有第二层理由：`scripts/` 下 40 多个历史脚本都在调
    `build_pipeline(loaded, top_k=…, graph_depth=…)`，给它们默认接上向量会
    静默改变它们的历史读数，那些台账里的对比就全部失去可比性。
    """
    retriever = retrieve_mod.Retriever(
        loaded["index"],
        loaded["graph"],
        loaded["adjacency"],
        chunks=loaded["chunks"],
        assertions=loaded["assertions"],
        vocab=vocab,
    )
    answerer = answer_mod.ExtractiveAnswerer()
    return {"retriever": retriever, "answerer": answerer, "top_k": top_k,
            "graph_depth": graph_depth, "vector_union": vector_union}

# ---------------------------------------------------------------- 生产档
#: 生产检索档：**唯一**一份定义。服务端 `/query`、`scripts/ask.py`、评测脚本都从
#: 这里取，不各自手抄一套数——本项目已有"同一判定写两处、取最松的那份生效"的三次
#: 同型事故。
#:
#: 为什么是这四个键、这几个数（全部有实测依据，见 `docs/升级台账.md` 本轮章节）：
#:
#: * `quote_locate` —— 题干引文逐字定位。门控在 `quote_locate.is_title_question`，
#:   640 题里只命中 single_chunk 128 道，零误伤；dev 该类 69/76 → 76/76。
#: * `bridge` + `graph_reserve_tokens` —— 多跳桥接实体。**两者必须同时开**：
#:   只开 bridge 时两条路径断言根本进不了上下文（40 个直接单元就吃掉 42k–52k
#:   token，图单元排在尾部被预算整批截掉，实测只有 3/62 能落地），dev multihop
#:   8→11；加上预留额度后才 8→27。这两个数不是互相可拆借的两笔收益。
#: * `bridge_search_top_k=40` —— 只对桥接题抬候选数。全局抬到 40 会让
#:   single_chunk 掉 1 题（逐字定位命中的块被挤到后面），只对桥接题抬则三份都涨。
#: * `max_context_tokens=24000` —— 12000 档装不下"40 个直接单元 + 预留 12000"，
#:   实测 12000 档最高只到 75.44%（dev），只有 24000 档能到 80%。
#:   代价：上下文单位 9.8 → 19.5、token 11.7k → 23.7k，延迟几乎不变
#:   （P50 73 → 81 ms、P95 132 → 137 ms）。
#: * `chapter_cap=4` —— W3：`cross_chunk` 章节答案截断到 4 章。**三份 split 全为正、
#:   零掉分**：dev 228→**231**（+3）、holdout 152→**154**（+2）、clean 97→**98**（+1），
#:   全部增益来自 `cross_chunk`（23→26 / 16→18 / 12→13）。
#:   为什么是 4：这是预登记值，也是实测最优点——cap=3 是 +4/−6（净 −2），
#:   cap=5 与 cap=6 都是 +0/−1。**不是"截得越狠越好"，必须正好 4。**
#:   为什么它有效：`cross_chunk` 判据要求 `|O| ≤ 2|W|`，基线常因顺带列了太多
#:   无关章节而失败；截到 4 章能消掉这一类，且 gold 章节的排序质量已实测
#:   （`data/diag_cx_rank_keys.json`：当前排序键 hit@4 = 55/57 = 96.5%）。
PROD_PROFILE: dict = {
    "max_context_tokens": 24000,
    "options": {
        "quote_locate": True,
        "bridge": True,
        "bridge_search_top_k": 40,
        "graph_reserve_tokens": 12000,
        "chapter_cap": 4,
    },
}


def prod_options() -> dict:
    """生产档的 `ask_once` 开关副本（调用方改副本不会污染 `PROD_PROFILE`）。"""
    return dict(PROD_PROFILE["options"])


def apply_prod_profile(pipeline: dict) -> dict:
    """把生产档落到一条链上，返回该传给 `ask_once` 的 `options`。

    为什么要一个函数而不是让调用方自己拼：服务端、CLI、评测脚本三处都要"用生产
    档"，各写一遍就是三份判定。这里也顺手把预算写进 `Retriever.budget`——那是
    `build_context` 唯一生效的入口（`tests/test_context_budget.py` 钉着这条）。

    **不改 `DEFAULT_BUDGET`**：那是个默认值，改它会让 `scripts/` 下 40 多个历史
    脚本的读数一起变，台账里的对比全部失去可比性。这里只改实例属性。
    """
    pipeline["retriever"].budget["max_context_tokens"] = int(
        PROD_PROFILE["max_context_tokens"])
    return prod_options()


def ask_once(pipeline: dict, question: str, *, timings: dict | None = None,
             options: dict | None = None) -> dict:
    """一问一答的完整链路。返回答案 + 每一层的中间量（供排查与评测）。

    `timings` 传入一个 dict 时，逐阶段毫秒数写进它（H1 性能基线要用）。
    默认 `None` → 不测时间，返回结构与原来完全一致：**计时是可选旁路，
    不是返回值的一部分**，否则每个读 `result["search"]` 的调用方都得改。

    ## `options`（默认 `None` = 与改造前逐位等价）

    | 键 | 默认 | 作用 |
    |---|---|---|
    | `quote_locate` | False | 题干引文逐字定位，命中块前置进 hits（`quote_locate.locate`） |
    | `bridge` | False | 多跳桥接实体通路，唯一候选才答（`bridge.find_bridge` + `merge`） |
    | `graph_reserve_tokens` | 0 | 传给 `build_context` 的图证据预留额度 |
    | `bridge_search_top_k` | 0 | 桥接题单独使用的候选数（0 = 用 `pipeline["top_k"]`） |
    | `search_top_k` | `pipeline["top_k"]` | 覆盖本次检索的候选数 |
    | `max_direct_units` | 0 | W1：直接证据单元进入上下文的条数上限（0 = 现状） |
    | `chapter_cap` | 0 | W3：`cross_chunk` 章节答案的章节数上限（0 = 现状） |
    | `chapter_rank` | `"score"` | L2：`chapter_cap` 打开时的章节排序键（`"score"` = 现状） |
    | `bridge_policy` | `"unique_only"` | W2：桥接候选策略（`unique_only` = 现状；`top1_margin` = 概率化 top-1） |
    | `bridge_calibration` | None | W2：`bridge_calib` 的映射表（仅 `top1_margin` 会读；不做隐式文件读取） |
    | `alias_answer` | False | L3：别名规范化答案通路（「「X」的规范名是什么」按图别名表反查） |
    | `evidence_window` | 0 | L4：证据窗口装填（0 = 现状；1 = 长尾单元只按含断言的轮次计价） |
    | `window_head_units` | None | L4：前 N 个单元装整块（None = `retrieve.WINDOW_HEAD_DEFAULT`；0 合法） |
    | `multihop_conjunction` | False | multihop：答案里补一条「实体与全部锚点相关联」的合取句（**带注水风险**，见 `answer.MULTIHOP_CONJUNCTION`） |
    | `max_claims` | 0 | L4b：按题覆盖 claim 句数上限（0 = 用 answerer 构造时的值） |
    | `chapter_fill` | False | L5：`cross_chunk` 定向补章（补全库里真实含该实体的长尾章节） |

    三个开关全部默认关，且**每一条都只改检索/装配，不改判据**；关掉时返回的
    `search.hits` / `context.stats` 与改造前逐位相同，历史台账读数不受影响。

    `graph_reserve_tokens` 与 `bridge` 都只在桥接题上生效：预留额度对全题生效是
    负收益（dev 213→209），只有在这类题上才是正收益——这不是调出来的阈值，而是
    "这类题需要两条图上的边、而 40 个直接单元已经吃满预算"的机制结果。
    """
    opts = dict(options or {})
    retriever = pipeline["retriever"]
    entities = pipeline.get("entities") or retriever.graph.get("entities") or []
    _t0 = time.perf_counter()
    qa = query_mod.analyze_query(
        question,
        entities=entities,
        aliases=retriever.graph.get("alias_index"),
        vocab=vocab,
        # 复用 `Retriever` 建好的只读结构。**这一行是 H2 的主要提速点**
        # （每问省约 98 ms 的图级重建，见 `query.QueryContext` 的实测表）。
        # 必须传 `retriever.query_context` 而不是自己造：那样又变回每问重建。
        context=getattr(retriever, "query_context", None),
    )
    if timings is not None:
        timings["query_ms"] = (time.perf_counter() - _t0) * 1000.0
    if not qa.get("should_answer", True):
        if timings is not None:
            timings["total_ms"] = (time.perf_counter() - _t0) * 1000.0
            timings["abstained_by_query"] = 1
        return {
            "query_analysis": qa,
            "search": {"hits": [], "stats": {"skipped_reason": qa.get("reason")}},
            "expand": {"expansions": [], "stats": {"skipped_reason": qa.get("reason")}},
            "context": {"units": [], "context_digest": "", "stats": {}},
            "answer": pipeline["answerer"].answer(
                question, {"units": [], "context_digest": ""},
                generation_id=pipeline.get("generation_id", ""),
                top_score=0.0, query_analysis=qa,
            ),
        }

    # 候选数：桥接题单独用 `bridge_search_top_k`，其余题照 `pipeline["top_k"]`。
    # 只在桥接题上抬高是实测结果——全局抬到 40 会让 single_chunk 掉 1 题
    # （逐字定位命中的块被挤到后面），而桥接题确实需要更宽的一跳邻域才有候选。
    top_k = int(opts.get("search_top_k") or 0)
    if not top_k:
        top_k = pipeline["top_k"]
        wide = int(opts.get("bridge_search_top_k") or 0)
        if wide and opts.get("bridge") and bridge.is_bridge_question(question):
            top_k = wide
    found = retriever.search(qa, top_k=top_k)
    if timings is not None:
        timings["search_ms"] = (time.perf_counter() - _t0) * 1000.0 - timings["query_ms"]
    # ---- 引文逐字定位（可选，默认关） ---------------------------------
    # 命中块**前置插入**而不是新增一路通道：它只提供"块 id"，分数与来源照正常命中
    # 的样子写，所以 `build_context` 不用知道这条前置存在。分数取
    # `max(现有 hits 分)+1.0`，保证它排在第一位——这不是排序技巧，是"逐字命中"
    # 本就比 BM25 更强的语义，把两者混进同一个尺度反而会让它被压下去。
    quote_hits = {}
    if opts.get("quote_locate"):
        loc = quote_locate.locate(question, retriever.chunks)
        if loc["chunk_ids"]:
            top = max((float(h.get("score") or 0.0) for h in found["hits"]), default=0.0)
            front = [
                {"chunk_id": cid, "score": top + 1.0, "cooc_boost": 0.0,
                 "sources": [{"channel": retrieve_mod.SOURCE_BM25, "score": 1.0}]}
                for cid in loc["chunk_ids"]
            ]
            seen = {h["chunk_id"] for h in found["hits"]}
            found["hits"] = [h for h in front if h["chunk_id"] not in seen] + list(found["hits"])
            quote_hits = {"probe": loc["probe"], "chunk_ids": list(loc["chunk_ids"])}
    _tvu = time.perf_counter()
    union = pipeline.get("vector_union")
    if union is not None:
        # 向量通道只在触发时**追加到尾部**，不改 base 的名次（见
        # `vector_union` 模块头：任何全局融合都被 300 题实测否证）。
        # 放在 search 之后、expand 之前：expand 的 seed_chunk_ids 必须包含
        # 向量补进来的块，否则"补进来的块"进不了图扩展，白补。
        aug = union.augment(found["hits"], question)
        found["hits"] = aug["hits"]
        found["stats"]["vector_union"] = {
            "used": aug["used"], "reason": aug["reason"],
            "ratio": aug["ratio"], "cut": aug["cut"],
            "vec_hits": aug["vec_hits"], "added": len(aug["added"]),
        }
    if timings is not None:
        timings["vector_union_ms"] = (time.perf_counter() - _tvu) * 1000.0
    _t1 = time.perf_counter()
    seed_ids = [c["entity_id"] for c in qa.get("entity_candidates") or []]
    expanded = retriever.expand(
        seed_ids,
        depth=pipeline["graph_depth"],
        query_analysis=qa,
        seed_chunk_ids=[h["chunk_id"] for h in found["hits"]],
    )
    if timings is not None:
        timings["expand_ms"] = (time.perf_counter() - _t1) * 1000.0
        timings["expand_nodes"] = int(expanded["stats"].get("nodes_visited") or 0)
        timings["expand_edges"] = int(expanded["stats"].get("expansions") or 0)
    _t2 = time.perf_counter()
    # **预留额度只在桥接题上生效**：全题生效实测是负收益（dev 213→209），因为
    # 非桥接题里图单元本来就只是背景，把它们优先装进来会挤掉真正的直接证据。
    # 这不是调出来的阈值——桥接题需要两条图上的边，而 40 个直接单元已经吃满预算，
    # 其余题没有这个结构性需求。
    reserve = int(opts.get("graph_reserve_tokens") or 0)
    if reserve and not bridge.is_bridge_question(question):
        reserve = 0
    # ---- W1：直接单元 cap 的门控（默认 0 = 现状） ------------------------
    # cap **只对「非桥接、非章节」题生效**，这是实测决定的：
    #   * multihop 有 56.7% 的 gold 落在 rank 9–20，而 `bridge_search_top_k=40`
    #     正是为它加的 —— 对它 cap 等于拆掉桥接通路；
    #   * cross_chunk 的章节答案由 W3 单独管，cap 它会与章节截断互相污染。
    # 这与 `graph_reserve_tokens`（只对桥接题生效）恰好互补。
    max_direct = int(opts.get("max_direct_units") or 0)
    if max_direct and (bridge.is_bridge_question(question)
                       or is_chapter_question(question)):
        max_direct = 0
    # 只在真的要传时才拼关键字：默认路径（`reserve == 0` 且 `max_direct == 0`）与
    # 改造前**连调用形状都一样**——这样任何既有 `build_context` 替身（测试桩、
    # 旧版实现）都不会因为多一个关键字参数而 TypeError。
    ctx_kwargs: dict = {}
    if reserve:
        ctx_kwargs["graph_reserve_tokens"] = reserve
    if max_direct:
        ctx_kwargs["max_direct_units"] = max_direct
    # ---- L4：证据窗口（默认 0 = 关，不传关键字） --------------------------
    # 同一套写法：只有非默认值才拼进 `**kwargs`，默认路径连调用形状都与改造前
    # 一样（既有 `build_context` 替身不会 TypeError）。
    #
    # **不门控在题型上**：窗口只改"长尾单元算多少 token"，不改任何判据，
    # 对哪些题型有效是端到端才能回答的问题——在这里按题型门控等于用未验证的
    # 假设去限制一个未验证的改动，两级不确定叠在一起就没人能归因了。
    # 头部整块的条数由 `window_head_units` 单独给（0 是合法值，表示整批按窗口
    # 计价），所以这里只在真的传入时才拼关键字，不用 `or` 把 0 吞掉。
    if opts.get("evidence_window"):
        ctx_kwargs["evidence_window"] = 1
        if opts.get("window_head_units") is not None:
            ctx_kwargs["window_head_units"] = int(opts["window_head_units"])
    context = retriever.build_context(
        found["hits"], expanded["expansions"], query_analysis=qa, **ctx_kwargs)
    if timings is not None:
        timings["context_ms"] = (time.perf_counter() - _t2) * 1000.0
    _t3 = time.perf_counter()
    # ---- W3：章节答案截断（默认 0 = 现状，不传关键字） --------------------
    # 与 `max_direct_units` 同一套写法：**只有非默认值才拼进 `**kwargs`**，
    # 这样默认路径连调用形状都和改造前一样，任何既有 `answer()` 替身
    # （测试桩、旧版实现）都不会因为多一个关键字参数而 TypeError。
    #
    # 门控：只对章节题生效。非章节题根本走不到 `_chapter_answer`，
    # 传进去也只是白挂着——但显式门控能让"这个旋钮影响谁"在代码里可读，
    # 而不是靠读 `answer()` 内部实现才知道。
    chapter_cap = int(opts.get("chapter_cap") or 0)
    if chapter_cap and not is_chapter_question(question):
        chapter_cap = 0
    ans_kwargs: dict = {}
    if chapter_cap:
        ans_kwargs["chapter_cap"] = chapter_cap
    # ---- L2：章节排序键（默认 `"score"` = 现状，不传关键字） ----------------
    # 与 `chapter_cap` 同一套写法：非默认值才拼进 `**kwargs`，这样默认档连
    # `answer()` 的调用形状都与改造前一样（既有替身不会 TypeError）。
    #
    # **不额外门控在 `chapter_cap` 上**：`chapter_rank` 只在真要截断时才被读取
    # （`_chapter_answer` 里 `cap > 0` 分支），所以 cap=0 时传了也是空转。
    # 不门控的代价是"传了个不起作用的键"，门控的代价是**静默吞掉取值** ——
    # 后者更坏：调用方以为开了、读数不变，会去查别处。这里选前者。
    chapter_rank = str(opts.get("chapter_rank") or answer_mod.CHAPTER_RANK_DEFAULT)
    if chapter_rank != answer_mod.CHAPTER_RANK_DEFAULT:
        ans_kwargs["chapter_rank"] = chapter_rank
    # ---- L5：`cross_chunk` 定向补章（默认关，非默认才拼关键字） --------------
    # 为什么需要扫全库：`_chapter_answer` 只看**本次上下文单元**，而长尾章节
    # （某实体在全库只出现 1 次的章）向量检索召不回来。实测 10 道 cross_chunk
    # 失败题全是「缺 1 个 gold 章节」，而那些块在全库里都存在。
    # 回调由这里注入，`answer.py` 不碰磁盘 —— 与 `bridge_calibration` 同一条纪律：
    # 回答层不许有隐式 I/O，否则「同一输入两次跑」会依赖磁盘状态。
    if opts.get("chapter_fill"):
        ans_kwargs["chapter_fill"] = True
        ans_kwargs["chapter_scan"] = retriever.chapter_scan
    # ---- L3：别名规范化答案通路（默认关，不传关键字） --------------------
    # 与 `chapter_cap` / `max_direct_units` 同一处置：**只有显式开启时才拼
    # `alias_context` 关键字**，默认路径连调用形状都与改造前一样，任何既有
    # `answer()` 替身（测试桩、旧版实现）都不会因为多一个关键字参数而 TypeError。
    #
    # 传的是 `retriever.query_context`（`query.QueryContext`）：它持有与本次
    # 查询**同一张**别名表（`_alias_owner_map` 的产物），回答侧反查规范名时
    # 不会出现"索引里有 `罗德岛制药公司`、查询侧拆成两个名字"这种对不齐。
    # **不在这里读任何文件**：pipeline 是每问都走的路径，隐式 I/O 会让
    # "同一输入两次跑"依赖磁盘状态。
    if opts.get("alias_answer"):
        ans_kwargs["alias_context"] = getattr(retriever, "query_context", None)
    # ---- multihop 多锚点合取装配（默认关，不传关键字） --------------------
    # 与 `chapter_cap` / `max_direct_units` / `alias_answer` 同一处置：**只有显式
    # 开启时才拼关键字**，默认路径连调用形状都与改造前一样，任何既有 `answer()`
    # 替身（测试桩、旧版实现）都不会因为多一个关键字参数而 TypeError。
    #
    # 门控：只对**多跳题**生效。判据与 `bridge` 共用同一个 `is_bridge_question`
    # （实测 96/96 全命中 multihop、其余七类 0 道），这里显式门控是为了让
    # "这个旋钮影响谁"在代码里可读，而不是靠读 `answer()` 内部实现才知道。
    #
    # ⚠️ 这条通路**天然带注水风险**（合取句是装配出来的，不是原文三元组），
    # 记账口径见 `answer.MULTIHOP_CONJUNCTION` 上方那段模块级注释。
    if opts.get("multihop_conjunction") and bridge.is_bridge_question(question):
        ans_kwargs["multihop_conjunction"] = True
    # ---- L4b：按题覆盖 claim 句数上限（默认 0 = 不传关键字，调用形状不变）----
    # L4 证据窗口把进上下文的单元数抬高后，`MAX_CLAIMS=8` 变成真正的瓶颈：
    # 参与竞争的 claim 变多，gold 句更容易被挤出前 8。实测 16 题因此从 pass 变 fail。
    # 这个旋钮默认 0，不传就不会有任何行为变化。
    max_claims_opt = int(opts.get("max_claims") or 0)
    if max_claims_opt > 0:
        ans_kwargs["max_claims"] = max_claims_opt
    ans = pipeline["answerer"].answer(
        question,
        context,
        generation_id=pipeline.get("generation_id", ""),
        top_score=float(found["stats"].get("top_score") or 0.0),
        query_analysis=qa,
        retrieval_stats=found["stats"],
        **ans_kwargs,
    )
    # ---- 多跳桥接（可选，默认关） -------------------------------------
    # 放在答案器**之后**：它不替换原答案，只是把两条真实断言句前置，让
    # "gold 与每个锚点各有一句共现"这个判据有机会成立。原答案一句不删（除句数封顶）。
    bridge_info = {}
    if opts.get("bridge"):
        # 候选断言取"上下文块的**全量**断言"，不只用 `unit["assertions"]`：
        # 后者被 `build_context` 按焦点实体谓词过滤，`X—A`/`X—B` 那条边的另一端
        # 不是焦点实体，会被过滤掉，实测少捞 6 题（dev multihop 20 → 26）。
        # 两种来源都只覆盖本次上下文里的块，接地口径不变。
        #
        # **W1 的桥接候选池**：cap 开启时 `build_context` 会额外返回
        # `bridge_candidates`（cap **之前**的完整接地候选单元）。桥接必须读它，
        # 否则 cap 会顺手砍掉桥接的候选池，违背 W1「解耦检索宽度与上下文宽度」。
        # 默认（cap=0）没有这个键，`or` 落回 `units`，逐位等价。
        bridge_units = context.get("bridge_candidates") or context.get("units") or []
        # ---- W2：桥接策略（默认 `unique_only` = 现状） --------------------
        # **只在非默认策略时拼关键字**：默认路径与改造前**连调用形状都一样**
        # （与 W1 的 `ctx_kwargs` 同一处置）。理由不只是洁癖：`tests/test_bridge.py`
        # 与 `scripts/audit_round6.py` 都会 `inspect.signature(find_bridge)` /
        # 直接按位置+两关键字调用；多传一个 `policy=` 会让那些替身与桩在默认档
        # 上表现不同，而默认档正是所有历史读数所在的那一档。
        policy = str(opts.get("bridge_policy") or bridge.DEFAULT_POLICY)
        br_kwargs: dict = {}
        if policy != bridge.DEFAULT_POLICY:
            br_kwargs["policy"] = policy
            # 校准表**只从 options 显式传入**，不在这里读文件：pipeline 是每问都
            # 走的路径，隐式 I/O 会让"同一输入两次跑"依赖磁盘状态。
            br_kwargs["calibration"] = opts.get("bridge_calibration")
        br = bridge.find_bridge(
            question,
            bridge_units,
            chunk_assertions=retriever.assertions_by_chunk.get,
            display=retriever.display_name,
            **br_kwargs,
        )
        bridge_info = {"status": br["status"], "x": br["x"]}
        if br["status"] == "ok":
            ans = bridge.merge(ans, br)
    if timings is not None:
        timings["answer_ms"] = (time.perf_counter() - _t3) * 1000.0
        timings["total_ms"] = (time.perf_counter() - _t0) * 1000.0
        timings["context_tokens"] = int(context["stats"].get("tokens_used") or 0)
        timings["context_units"] = int(context["stats"].get("units") or 0)
        timings["candidates"] = int(found["stats"].get("candidates") or 0)
        if opts.get("quote_locate"):
            timings["quote_locate_hits"] = len(quote_hits.get("chunk_ids") or ())
        if opts.get("bridge"):
            timings["bridge_status"] = bridge_info.get("status") or ""
    return {
        "query_analysis": qa,
        "search": found,
        "expand": expanded,
        "context": context,
        "answer": ans,
    }


__all__ = ["ask_once", "build_pipeline", "load_generation"]
