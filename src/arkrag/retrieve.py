"""检索与有界图扩展 —— 计划 G1 / G4 / G5。

## 分工

`query.analyze_query` 只说"问了什么"，本模块说"去库里找什么、找到多少就停"。
`answer` 只说"根据拿到的材料能说什么"。三件事分开写，是因为它们失败的后果不同：
问句解析错了会让整条链路答偏，检索错了会让答案没有证据，生成错了会编事实。

## 四条不许退让的规矩

1. **扩展必须带原文证据。** 每个扩展项都带 `chunk_id` + `evidence`（逐字来自
   `assertions.jsonl` 的 `evidence_text`）。只返回"凯尔希 --隶属--> 罗德岛"这种
   图结论、不返回那句话在哪里说的，等于要求下游相信一个不可查的断言。图里的
   每条边本来就是从具体观察聚合出来的，不拿它用，等于白扔 62,910 条断言。
2. **默认只走出边。** `adjacency` 把 in/out 分列，就是为了让"读反向"成为一个
   显式动作。要读入边必须 `include_in=True`，结果标 `used_in_edges=True`，
   并且每条入边的 `relation` 换成逆谓词（`adjacency.adjacency_item` 已经算好）。
   反向沿用原谓词是 E2 记录过的真实 bug，这里不重犯。
3. **预算硬性、用尽要说。** `max_depth / max_nodes / max_edges / max_degree_per_entity
   / max_assertions / max_context_tokens / timeout_ms` 全部可配且全部回显实际用量；
   任一项用尽就写 `truncated_reason`。悄悄多取一点，事后没人知道上下文缺了什么。
4. **同 chunk 只留一次，来源合并。** 词法命中与图扩展常指向同一块；拆成两条会让
   答案重复引用同一句话，看起来"证据充分"其实只有一条。

## 五档扩展优先级（`stats["priority_tiers"]` 各档计数）

| 档 | 判据 | 理由 |
|---|---|---|
| 1 | 边的一端**就是**查询实体，且深度 1 | 用户直接问到的关系最相关 |
| 2 | 谓词命中 `relation_hints` | 「凯尔希和罗德岛什么关系」里出现的关系词 |
| 3 | `support_count >= SUPPORT_HIGH` | 出现多次的关系比只出现一次的可信 |
| 4 | 边的 `chunks` 与已召回块有交集 | 已经找到的正文旁边的关系更可能是上下文 |
| 5 | 证据断言有 `qualifiers.scene_order` 且查询含时间词 | **只做标记，不做时序推理**（见下方「已知边界」） |

一条边可能同时命中多档，取**最小档号**作为它的 `priority_tier`，其余档号留在
`matched_tiers` 里——这样排序只看一个键，而计数仍能反映真实覆盖。

## 已知边界（如实写在这里，不靠判据掩盖）

* 第 5 档**不做真实时序推理**。语料里没有可靠的绝对时间轴，`scene_order` 是
  "本场景内第几条观察"，跨场景不可比。所以这里只标出"这条证据带场景序、查询
  提到了时间词"，把是否成立的判断留给 `answer` 的 warnings 与人工复核。
* **不排序不收敛的图游走**。深度上限 2，每实体邻居上限 20，节点上限 80；
  这些数字是为了"一次问答能在两秒内完成"，不是"覆盖整张图"。
* `token` 估算用与建库同一套分词器（`index.tokenize`）。它与任何真实 LLM 的
  分词都不同；写成文档是为了让"6000 token"这个数字可复现，不是声称精确。
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict

from . import adjacency as adjacency_mod
from . import index as index_mod
from . import query as query_mod

# 默认预算。每个数字都写在这里，调用方可逐个覆盖；没有任何一个藏在下游函数里。
DEFAULT_BUDGET: dict = {
    "max_depth": 2,
    "max_nodes": 80,
    "max_edges": 200,
    "max_degree_per_entity": 20,
    "max_assertions": 200,
    # 单条边最多取几条观察做锚点。没有这条上限时，实测第一个实体的一条
    # `交谈` 边就带着 121 条断言，直接吃掉整个 max_assertions 预算，于是
    # 整次扩展只产出 1 条边——预算被一条边独占，其余邻居一个都进不来。
    "max_assertions_per_edge": 5,
    "max_context_tokens": 6000,
    "timeout_ms": 2000,
    # 第 3 档的门槛：出现至少两次才算"可信度较高"。设成 2 而不是 1，是因为
    # 实测 28,625 条边里有相当一部分 support_count=1；全算高可信等于没有这一档。
    "support_high": 2,
}

# 实体加成通道的权重。选 0.5 的理由：词法命中是"正文里确实出现了这些词"，
# 图实体命中是"问句里出现了图里的名字"，后者更可能是问句命中了某个专名而不代表
# 那块的正文更相关。给一半权重正好让它能顶掉弱匹配，顶不掉强匹配。
ENTITY_CHANNEL_WEIGHT = 0.5

# 候选池深度倍数：`index.search` 拉 `top_k * 本值` 个候选再合并排序。
#
# 为什么可以放大到 10：词法检索的成本在**扫描倒排**，不在排序截断。
# H1 实测 `top_k` 从 5 到 100，耗时恒定在 180–210 ms（H2 预建归一化表后约 67 ms），
# 即「多拉候选」几乎不花钱。
#
# 为什么值得放大：多拉的候选在 `bm25 + 0.5*entity` 合并之后，能捞回一批被实体
# 通道分数挤到 20 名之后的块。320 题实测（输出固定 top_k=20）：
#     mult=3（原值）  R@20 0.6575  MRR 0.4123
#     mult=5          R@20 0.6775  MRR 0.4221  (+0.0200 / +0.0098)
#     mult=10         R@20 0.6842  MRR 0.4392  (+0.0267 / +0.0269)  ← 采纳
#     mult=20         R@20 0.6817  MRR 0.4385  (+0.0242 / +0.0262)
#     mult=50         R@20 0.6817  MRR 0.4384
# 10 是唯一同时最大化 R@20 与 MRR 的点；20/50 略回落（候选池已覆盖全部相关块，
# 再深只是把弱匹配也拉进来争名次）。
CANDIDATE_POOL_MULT = 10

# 说话人加成权重。
#
# 信号是什么：`chunks.jsonl` 的 `speakers` 字段是**结构化**的"这个块里谁在说话"。
# 剧情是对话体，问句问到的对象若正是该块的说话人之一，那这块几乎必然相关。
# 这比实体通道的字面子串匹配强得多——后者只说明"正文里提到了这个名字"，
# 一个块提三次「凯尔希」不代表它在讲凯尔希的这件事（实体通道单路 MRR 仅 0.2255
# 就是证据）。
#
# 为什么是 0.1 而且只给 speakers：`chunks.jsonl` 13,786 块中 97.1% 有 speakers，
# 320 题里有 206 题的问句能命中某个块的 speakers（64%）。扫描实测（320 题，
# 输出固定 top_k=20，baseline 为不加任何结构化加成）：
#     ws=0.1  R@20 0.7019 (+0.0177)  MRR 0.4394 (+0.0002)  ← 采纳，唯一两者都不劣
#     ws=0.2  R@20 0.7003 (+0.0161)  MRR 0.4336 (−0.0056)
#     ws=0.3  R@20 0.6964 (+0.0122)  MRR 0.4289 (−0.0103)
#     ws=0.5  R@20 0.6819 (−0.0023)  MRR 0.4076 (−0.0316)
# 权重再大就开始把"提到了这个人"的块硬抬到头部，伤害精度。
# `title`、`chapter` 加成在任何权重下都只降不升，故不实现。
#
# 显著性（配对符号检验，n=300）：R@20 更好 8 题 / 更差 1 题 / 持平 291 题，
# 双侧 p=0.0391，bootstrap 95% 区间 [+0.00167, +0.03611] 不跨 0 → 显著；
# MRR 更好 22 / 更差 12 / 持平 266，p=0.1214，区间 [−0.00458, +0.00432] 跨 0
# → **不显著**。所以准确说法是「召回显著提升，头部精度无显著变化（没有变差）」，
# 不许说成「两项都提升」。
SPEAKER_BOOST_WEIGHT = 0.1

# 两端点同行共现加成权重（P2 第一轮，配置名 `cooc0.1`）。
#
# 信号是什么：问句解析出的**两个端点实体**的规范名，**同时出现在候选块的同一行**。
# 这是「这个块直接讲了这两个人之间的这件事」的强证据，比实体通道的
# 「块里提到了这个名字」精确得多——一个块提三次「凯尔希」不代表它在讲凯尔希的这件事。
#
# 为什么 0.1 够用：`cooc` 是**稀疏信号**。实测（dev 350 应答题）前 20 内命中块数分布
# 为 0 块 215 题 / 1-3 块 40 / 4-9 块 57 / ≥10 块 38，均值 2.90。稀疏意味着
# 可以给得起权重而不至于把大片噪声块抬上来；0.1 已能把目标块从第 6-9 名推进前 5。
#
# **为什么取 0.1 不取 0.3**（dev 350 真管线实测）：
#     cooc0.1  进 ctx 189→195 (+6)、覆盖 58→63 (+5)、**覆盖回退 0 题**
#     cooc0.3  进 ctx 189→198 (+9)、覆盖 58→65 (+7)、**覆盖回退 2 题**
#              （`q-sc-000015`、`q-tc-000003`；机制是 `answer.py` 的
#               `MAX_CLAIMS = 5` claim 截断——gold 单元后移后被同块兄弟单元挤出）
# 判据纪律是「净额为正不能抵消某一维的回退」，故取零回退的 0.1。
# 0.3 留作第二档，须先单独解决 claim 截断。
#
# **不许把本信号在对话体语料上推广成「跨行共现」**：`hop_line` 被判死刑正是因为
# 它的「同一行」前提在对话体（`X：` / `Y：` 交替分行）上判反了，详见
# `docs/a_P2_cooc实验方案与hop根因.md` §2.4。
COOC_BOOST_WEIGHT = 0.1

# 上下文单元的固定来源标记，避免各处手写字符串拼错。
SOURCE_BM25 = "bm25"
SOURCE_ENTITY = "entity"
SOURCE_SPEAKER = "speaker"
SOURCE_GRAPH = "graph"


def _speaker_set(rec: dict) -> set[str]:
    """从块记录取说话人集合，兼容 list 与 JSON/`repr` 字符串两种形态。

    `chunks.jsonl` 里 `speakers` 目前是**字符串形式的 Python list**
    （如 `"['可萝尔', '村民']"`，单引号），不是严格 JSON。所以先试 JSON，
    失败再退回字面量解析，两条都不成就当没有——**不猜、不部分解析**：
    猜错的说话人会让加成把无关块抬进头部，比没有加成更糟。
    """
    raw = rec.get("speakers")
    if isinstance(raw, (list, tuple, set)):
        return {str(s).strip() for s in raw if str(s).strip()}
    if not isinstance(raw, str) or not raw.strip():
        return set()
    text = raw.strip()
    parsed = None
    if text.startswith("["):
        try:
            import json as _json

            parsed = _json.loads(text)
        except ValueError:
            try:
                import ast

                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = None
    if isinstance(parsed, (list, tuple, set)):
        return {str(s).strip() for s in parsed if str(s).strip()}
    return set()


def estimate_tokens(text: str, *, tokenizer_version: str = index_mod.TOKENIZER_VERSION,
                    name_terms=None) -> int:
    """用建库同一套分词器估 token 数。

    与真实 LLM 分词不同，这一点在模块 docstring 里已如实声明。用同一套的好处是
    "上下文 6000 token"这个结论在测试里可复现，不会因为环境不同而漂。
    """
    if not text:
        return 0
    return len(index_mod.tokenize(text, version=tokenizer_version, name_terms=name_terms))


def _norm_scores(hits: list[tuple[str, float]]) -> dict[str, float]:
    """把一路得分归一到 [0, 1]。空输入返回空 dict。"""
    if not hits:
        return {}
    top = max(s for _, s in hits) or 1.0
    return {cid: s / top for cid, s in hits}


class Retriever:
    """检索 + 有界扩展。

    参数
    ----
    index
        `index.LexicalIndex`。分词版本与整词名表随索引走，查询侧不另造口径。
    graph
        `ark-rag-3` 的图 dict（需要 `entities` / `edges`）。
    adjacency
        `adjacency.build_directed_adjacency` 的输出。
    chunks
        `{chunk_id: 块}` 或块列表。**没有它就无法给出原文证据**，所以缺块时
        相关单元会被跳过并计入 `stats["skipped_units"]`，而不是伪造空正文。
    assertions
        `assertions.jsonl` 读出来的列表（可选）。有它才能给出 evidence 的
        绝对偏移；没有时扩展项仍带 `evidence`，但 `evidence_spans` 为空。
    budget
        覆盖 `DEFAULT_BUDGET` 中的任意键。
    """

    def __init__(
        self,
        index,
        graph: dict,
        adjacency: dict,
        *,
        chunks=None,
        assertions=None,
        budget: dict | None = None,
        query_context=None,
        vocab=None,
    ) -> None:
        self.index = index
        self.graph = graph or {}
        self.adjacency = adjacency or {}
        self.budget = dict(DEFAULT_BUDGET)
        self.budget.update(budget or {})

        self.entities_by_id: dict[str, dict] = {}
        self.entities_by_name: dict[str, dict] = {}
        self.alias_index: dict[str, str] = {}
        self._build_entity_tables()

        # 问句解析的只读预构建结构（`query.QueryContext`）。这里**惰性构造**：
        # 不传就用本图的 entities + 默认词表建一份，之后每问复用。
        #
        # 为什么放在 Retriever 上而不是 query 模块里做全局缓存：这些结构必须与
        # 本实例的 `graph` 成对，`Retriever` 已经绑定了那张图，生命周期天然对齐。
        # 见 `query.QueryContext` 的实测表（每问省约 98 ms）。
        if query_context is not None:
            self.query_context = query_context
        else:
            self.query_context = query_mod.QueryContext(
                self.graph.get("entities") or [],
                self.graph.get("alias_index"),
                vocab,
            )

        self.edges_by_id: dict[str, dict] = {}
        for edge in self.graph.get("edges") or []:
            eid = str(edge.get("edge_id") or "")
            if eid:
                self.edges_by_id[eid] = edge

        self.chunks: dict[str, dict] = {}
        if isinstance(chunks, dict):
            self.chunks = {str(k): v for k, v in chunks.items()}
        else:
            for rec in chunks or ():
                cid = str(rec.get("chunk_id") or "")
                if cid:
                    self.chunks[cid] = rec

        self.assertions_by_id: dict[str, dict] = {}
        self.assertions_by_chunk: dict[str, list[dict]] = defaultdict(list)
        for rec in assertions or ():
            aid = str(rec.get("assertion_id") or "")
            if not aid:
                continue
            self.assertions_by_id[aid] = rec
            cid = str(rec.get("chunk_id") or "")
            if cid:
                self.assertions_by_chunk[cid].append(rec)
        for items in self.assertions_by_chunk.values():
            items.sort(key=lambda r: str(r.get("assertion_id") or ""))

    # ------------------------------------------------------------ 实体表
    def _build_entity_tables(self) -> None:
        """建 `id->实体`、`name->实体`、`别名->id` 三张表。

        别名表直接用 `index.build_alias_index`——它按元素类型分派，两代图都能吃，
        且**不含 pending_aliases**（`identity.to_dict()` 已把它们分到另一张表）。
        在这里再写一遍过滤就是同一判定写两处，`tests/test_single_source_of_truth.py`
        记录的三次事故都是这个机理。
        """
        entities = list(self.graph.get("entities") or [])
        for ent in entities:
            eid = str(ent.get("id") or "")
            if eid:
                self.entities_by_id[eid] = ent
            name = str(ent.get("name") or "")
            if name:
                self.entities_by_name.setdefault(name, ent)
        self.alias_index = index_mod.build_alias_index(entities)

    def display_name(self, entity_id: str) -> str:
        ent = self.entities_by_id.get(str(entity_id)) or {}
        return str(ent.get("name") or entity_id)

    def _speaker_boost(
        self, entity_candidates, allowed: set[str]
    ) -> dict[str, float]:
        """说话人加成：问句实体命中了某块的 `speakers` 就给分。

        **只用结构化字段做精确相等匹配**，不做子串模糊匹配——模糊匹配正是实体
        通道已有的信息（"正文里出现了这个名字"），再做一遍不会带来新信号。
        剧情是对话体，`speakers` 列的是这个块里真正开口的人，命中它就说明
        问句问的对象当场在场。

        归一化：`命中实体的权重之和 / 问句里最强实体的权重`。**这个值可以超过 1.0，
        上界是"问句里同时命中该块的实体数"**——这是刻意保留的，不是缺陷：

        * 分子是"有几个问句实体在这个块里真的开口了"，分母只取最强的那一个。
          两个问句实体都在同一块说话，比只有一个在场更能说明该块相关，
          这个单调性正是要被排序看到的东西。
        * 实测：把分数夹到 1.0 后 R@20 从 0.7019 掉到 0.6919（`Δ` 少一半），
          而按实体去重（防止同一实体的规范名+别名重复计票）对分数**毫无影响**
          （0.7019 不变）。所以「超过 1.0」全部来自多实体同时在场，不是别名重复。
        * 因此 `SPEAKER_BOOST_WEIGHT` 的实际强度随命中实体数放大。这是已知且
          被实测支持的取舍：ws=0.1 是唯一让 R@20 与 MRR 都不劣的值，
          而 ws≥0.5 时两项一起掉——放大效应已在扫描里体现，没有失控。

        只在 `allowed`（已进入候选池的 chunk）里算，不扫全表：
        13,786 个块逐个解析 speakers 每题要几十毫秒，而候选池只有几百个。
        """
        if not entity_candidates or not allowed:
            return {}
        # 实体 -> (该实体的全部称呼, 该实体的候选权重)
        # 按实体去重（而非按别名累加）：同一实体的规范名与别名可能同时出现在
        # 某块的 speakers 里（如 ["凯尔希", "凯尔希医生"]），按别名累加会让这一块
        # 白拿双倍票。去重实测对指标**零影响**（R@20 0.7019 不变），
        # 纯粹是防止一种讲不通的计票方式。
        by_entity: dict[str, tuple[set[str], float]] = {}
        for cand in entity_candidates:
            eid = str(cand.get("entity_id") or "")
            ent = self.entities_by_id.get(eid) or {}
            terms = {str(ent.get("name") or cand.get("name") or "").strip()}
            for item in ent.get("aliases") or []:
                t = item.get("text") if isinstance(item, dict) else item
                if str(t or "").strip():
                    terms.add(str(t).strip())
            terms.discard("")
            if not terms:
                continue
            weight = float(cand.get("score") or 0.0)
            # 同一实体可能以多个候选出现（多个别名各成一票），取最大权重
            prev = by_entity.get(eid)
            if prev is None or weight > prev[1]:
                merged_terms = terms | (prev[0] if prev else set())
                by_entity[eid] = (merged_terms, weight)
            else:
                prev[0].update(terms)
        if not by_entity:
            return {}
        strongest = max(w for _terms, w in by_entity.values()) or 1.0
        best: dict[str, float] = {}
        for cid in allowed:
            rec = self.chunks.get(cid) or {}
            speakers = _speaker_set(rec)
            if not speakers:
                continue
            # 每个实体最多贡献一次：只要它的任一个称呼在 speakers 里就算命中
            hit = sum(w for terms, w in by_entity.values() if terms & speakers)
            if hit > 0:
                best[cid] = hit / strongest
        return best

    def _cooc_boost(
        self, entity_candidates, allowed: set[str]
    ) -> tuple[dict[str, int], list[str]]:
        """两端点**同行共现**：问句的前两个实体规范名同时出现在块的同一行。

        与 `_speaker_boost` 的区别：后者看结构化 `speakers` 字段（"谁在这块开口"），
        这个看正文行（"这块哪一行同时写了这两个人"）。两者互补，都只在候选池里算。

        为什么只要"前两个实体"：问句通常有两个端点（"A 与 B 处于哪种关系"），
        `analyze_query` 已按权重排序，前两个即最可能的端点对。**不枚举所有两两组合**
        ——组合数会随候选数平方增长，而实测稀疏性（dev 350 前 20 内 0 块 215 题）
        说明单对信号已经够用。

        返回 `(命中表, 端点名列表)`；端点名回传是为了让调用方能把信号记进 stats，
        便于事后归因"这一题到底有没有端点对"。
        """
        names: list[str] = []
        seen: set[str] = set()
        for cand in (entity_candidates or []):
            eid = str(cand.get("entity_id") or "")
            ent = self.entities_by_id.get(eid) or {}
            nm = str(ent.get("name") or cand.get("name") or "").strip()
            if nm and nm not in seen:
                seen.add(nm)
                names.append(nm)
            if len(names) >= 2:
                break
        if len(names) < 2:
            return {}, names

        best: dict[str, int] = {}
        for cid in allowed:
            rec = self.chunks.get(cid) or {}
            text = str(rec.get("text") or "")
            if not text:
                continue
            for line in text.splitlines():
                if all(n in line for n in names):
                    best[cid] = 1
                    break
        return best, names

    # ------------------------------------------------------------ 检索
    def search(self, query_analysis: dict, *, top_k: int = 20) -> dict:
        """BM25 命中，两路加权合并 + 说话人结构化加成。

        通道与公式（写清是为了让分数可复算）：

        * `bm25`：`index.search(retrieval_query, top_k=top_k*CANDIDATE_POOL_MULT)`，
          再按该路最高分归一到 `[0,1]`。
        * `entity`：把每个实体候选的**规范名 + 权威别名**拼成一个查询串分别检索，
          每路同样归一，按候选 `score` 加权取该 chunk 的最大值。
        * `speaker`：问句实体与块的 `speakers`（结构化字段）精确相等的比例，
          见 `_speaker_boost`。
        * `cooc`：问句**前两个实体**的规范名同时出现在块正文的**同一行**，
          见 `_cooc_boost`。命中格给 `COOC_BOOST_WEIGHT`（常数，非比例）。
        * 合并：`final = bm25_norm + ENTITY_CHANNEL_WEIGHT * entity_norm
          + SPEAKER_BOOST_WEIGHT * speaker_norm + cooc_bonus`。
          各权重的取值理由与实测见各自常量注释。

        返回 `{hits:[{chunk_id, score, cooc_boost, sources:[{channel, score}]}], stats:{...}}`。
        三路的原始分都留在 `sources` 里，谁贡献了多少可以逐条查；`cooc_boost`
        单独列出（`0.0` 或 `COOC_BOOST_WEIGHT`），因为它不在 `sources` 通道体系里，
        混进 `score` 后无法反推是哪个信号抬的。
        """
        top_k = max(1, int(top_k))
        offset = 0
        if not query_analysis.get("should_answer", True):
            return {
                "hits": [],
                "stats": {
                    "skipped_reason": str(query_analysis.get("reason") or "not_answerable"),
                    "bm25_terms": 0,
                    "entity_channels": 0,
                    "candidates": 0,
                    "top_k": top_k,
                },
            }
        text = str(
            query_analysis.get("retrieval_query")
            or query_analysis.get("normalized_query")
            or ""
        )
        # 候选池深度 = `top_k * CANDIDATE_POOL_MULT`。这个倍数不是随手取的：
        # `index.search` 的成本在扫描倒排（与 top_k 几乎无关，H1 实测 P50 在
        # k=5..100 之间恒定 180–210 ms，H2 预建归一化表后降到约 67 ms），
        # 所以「多拉候选」几乎是免费的；而多拉的候选在合并、排序之后能捞回
        # 一批被实体通道分数挤出前 20 的块。实测（320 题，输出固定 top_k=20）：
        #     mult=3（原值）  R@20 0.6575  MRR 0.4123
        #     mult=5          R@20 0.6775  MRR 0.4221
        #     mult=10         R@20 0.6842  MRR 0.4392  ← 采纳
        #     mult=20         R@20 0.6817  MRR 0.4385
        # 10 是拐点：再深不再涨（候选池已覆盖全部相关块），且 R@20 略回落。
        # 注意这**不是**评测技巧：用户看到的仍是前 20 条，只是内部多看几眼再排。
        depth = top_k * CANDIDATE_POOL_MULT
        bm25_raw = self.index.search(text, top_k=depth) if text else []
        bm25 = _norm_scores(bm25_raw)

        entity_best: dict[str, float] = {}
        entity_channels = 0
        for cand in query_analysis.get("entity_candidates") or []:
            eid = str(cand.get("entity_id") or "")
            ent = self.entities_by_id.get(eid) or {}
            terms = [str(ent.get("name") or cand.get("name") or "")]
            for item in ent.get("aliases") or []:
                t = item.get("text") if isinstance(item, dict) else item
                if str(t or "").strip():
                    terms.append(str(t).strip())
            probe = " ".join(sorted({t for t in terms if t}))
            if not probe:
                continue
            entity_channels += 1
            weight = float(cand.get("score") or 0.0)
            for cid, score in _norm_scores(self.index.search(probe, top_k=depth)).items():
                entity_best[cid] = max(entity_best.get(cid, 0.0), score * weight)

        merged: dict[str, dict] = {}
        for cid, score in bm25.items():
            merged.setdefault(cid, {"chunk_id": cid, "sources": []})["sources"].append(
                {"channel": SOURCE_BM25, "score": round(score, 6)}
            )
        for cid, score in entity_best.items():
            merged.setdefault(cid, {"chunk_id": cid, "sources": []})["sources"].append(
                {"channel": SOURCE_ENTITY, "score": round(score, 6)}
            )

        # ---- 说话人通道：结构化精确匹配，不是字面子串 ----
        # 只在已进入候选池的块里算（bm25 ∪ entity），不扫全表。
        speaker_best = self._speaker_boost(
            query_analysis.get("entity_candidates") or (), set(merged)
        )
        for cid, score in speaker_best.items():
            merged.setdefault(cid, {"chunk_id": cid, "sources": []})["sources"].append(
                {"channel": SOURCE_SPEAKER, "score": round(score, 6)}
            )

        # ---- 两端点同行共现通道：P2 第一轮新增（`COOC_BOOST_WEIGHT`）----
        # 与说话人通道同理，只在已进入候选池的块里算（bm25 ∪ entity），不扫全表。
        # **必须在这里算、在下面的 hits.sort() 之前加成**：若改成「先切 top_k 再重排」，
        # 深池里名次靠后的块（实测有 rank 43 的 gold）永远进不来，信号等于没用。
        cooc_best, cooc_names = self._cooc_boost(
            query_analysis.get("entity_candidates") or (), set(merged)
        )

        hits: list[dict] = []
        for cid, item in merged.items():
            total = 0.0
            for src in item["sources"]:
                if src["channel"] == SOURCE_BM25:
                    total += src["score"]
                elif src["channel"] == SOURCE_SPEAKER:
                    total += SPEAKER_BOOST_WEIGHT * src["score"]
                else:
                    total += ENTITY_CHANNEL_WEIGHT * src["score"]
            cooc = COOC_BOOST_WEIGHT if cooc_best.get(cid) else 0.0
            total += cooc
            hits.append(
                {
                    "chunk_id": cid,
                    "score": round(total, 6),
                    "cooc_boost": cooc,
                    "sources": sorted(item["sources"], key=lambda s: s["channel"]),
                }
            )
        hits.sort(key=lambda h: (-h["score"], h["chunk_id"]))
        hits = hits[:top_k]

        # 词法通道单独贡献的 top1 分数，用于 answer 的"检索不足"判据
        top_score = hits[0]["score"] if hits else 0.0
        return {
            "hits": hits,
            "stats": {
                "bm25_terms": len(bm25),
                "bm25_top_score": round(max(bm25.values()), 6) if bm25 else 0.0,
                "entity_channels": entity_channels,
                "entity_top_score": round(max(entity_best.values()), 6) if entity_best else 0.0,
                "speaker_boosted": len(speaker_best),
                "speaker_weight": SPEAKER_BOOST_WEIGHT,
                "cooc_endpoints": cooc_names,
                "cooc_boosted": len(cooc_best),
                "cooc_weight": COOC_BOOST_WEIGHT,
                "candidates": len(merged),
                "top_score": top_score,
                "top_k": top_k,
                "entity_weight": ENTITY_CHANNEL_WEIGHT,
                "candidate_pool_depth": depth,
                "skipped_reason": "",
                "queried_text": text,
            },
        }

    # ------------------------------------------------------------ 扩展
    def expand(
        self,
        entity_ids,
        *,
        depth: int = 1,
        budget: dict | None = None,
        include_in: bool = False,
        query_analysis: dict | None = None,
        seed_chunk_ids=None,
    ) -> dict:
        """从给定实体出发做有界扩展。

        只走出边；`include_in=True` 时才读入边，且入边项的 `relation` 换成逆谓词、
        结果标 `used_in_edges=True`。每条扩展项都带 `chunk_id` + `evidence`。

        返回 `{expansions:[...], stats:{...}}`。`stats` 里既有预算原始值也有实际用量，
        用尽时 `stats["truncated_reason"]` 非空。
        """
        merged_budget = dict(self.budget)
        merged_budget.update(budget or {})
        max_depth = max(0, int(merged_budget["max_depth"]))
        depth = min(max(0, int(depth)), max_depth)
        max_nodes = max(1, int(merged_budget["max_nodes"]))
        max_edges = max(0, int(merged_budget["max_edges"]))
        max_degree = max(1, int(merged_budget["max_degree_per_entity"]))
        max_assertions = max(0, int(merged_budget["max_assertions"]))
        max_per_edge = max(1, int(merged_budget.get("max_assertions_per_edge", 5)))
        support_high = int(merged_budget["support_high"])
        deadline = time.monotonic() + max(1, int(merged_budget["timeout_ms"])) / 1000.0

        qa = query_analysis or {}
        hint_predicates = {str(h.get("canonical") or h.get("raw") or "") for h in (qa.get("relation_hints") or [])}
        hint_predicates.discard("")
        time_constrained = bool(qa.get("time_constraints"))
        seeds = [str(e) for e in (entity_ids or []) if str(e)]
        seed_set = set(seeds)
        seed_chunks = {str(c) for c in (seed_chunk_ids or []) if str(c)}

        priority = {}   # 档号 -> 判据说明
        expansions: list[dict] = []
        seen_edge_keys: set[tuple[str, str]] = set()
        visited_nodes: list[str] = []
        truncated_reason = ""
        used_in_edges = False
        assertions_emitted = 0
        tier_counts = defaultdict(int)
        per_entity_taken: dict[str, int] = defaultdict(int)
        frontier = [(eid, 0) for eid in seeds] if depth > 0 else []
        visited = set(seeds)

        while frontier:
            if not frontier:
                break
            node_id, node_depth = frontier.pop(0)
            if node_id not in visited_nodes:
                visited_nodes.append(node_id)
            if len(visited_nodes) >= max_nodes:
                truncated_reason = truncated_reason or f"max_nodes({max_nodes})"
                break
            if time.monotonic() > deadline:
                truncated_reason = truncated_reason or "timeout_ms"
                break

            directions = ["out"] + (["in"] if include_in else [])
            candidates: list[tuple[dict, str]] = []
            for direction in directions:
                for item in adjacency_mod.neighbors(self.adjacency, node_id, direction=direction):
                    candidates.append((item, direction))
            # 先按档号、再按 support_count、最后按 edge_id 排——
            # 排序键只有这三个，避免"看起来更相关"这类没有判据的排序。
            scored: list[tuple[int, int, str, dict, str, list[int]]] = []
            for item, direction in candidates:
                edge_id = str(item.get("edge_id") or "")
                key = (node_id, edge_id) if direction == "out" else (edge_id, node_id)
                if key in seen_edge_keys:
                    continue
                edge = self.edges_by_id.get(edge_id) or {}
                tiers: list[int] = []
                if node_depth == 0 and node_id in seed_set:
                    tiers.append(1)
                if hint_predicates and item.get("relation") in hint_predicates:
                    tiers.append(2)
                if int(item.get("support_count") or 0) >= support_high:
                    tiers.append(3)
                if seed_chunks and set(edge.get("chunks") or []) & seed_chunks:
                    tiers.append(4)
                if time_constrained:
                    tiers.append(5)
                scored.append(
                    (
                        min(tiers) if tiers else 99,
                        -int(item.get("support_count") or 0),
                        edge_id,
                        item,
                        direction,
                        sorted(tiers),
                    )
                )
            scored.sort(key=lambda row: (row[0], row[1], row[2]))

            stopped = False
            for tier, _neg_support, edge_id, item, direction, tiers in scored:
                if len(expansions) >= max_edges:
                    truncated_reason = truncated_reason or f"max_edges({max_edges})"
                    stopped = True
                    break
                if per_entity_taken[node_id] >= max_degree:
                    # **只停止当前实体的展开，不终止整个遍历。**
                    # 原实现这里置 `stopped = True`，于是 `max_degree_per_entity` 一旦
                    # 用完就 `break` 出整个 while：实测问「凯尔希和罗德岛是什么关系？」
                    # 时凯尔希出边够 20 条即停，**第二个种子罗德岛从未被访问**
                    # （`nodes_visited=1`），问句里另一半实体对应的边根本不会出现在
                    # 上下文里。多实体问句必然踩到这条。
                    truncated_reason = truncated_reason or f"max_degree_per_entity({max_degree})"
                    break
                if time.monotonic() > deadline:
                    truncated_reason = truncated_reason or "timeout_ms"
                    stopped = True
                    break
                edge = self.edges_by_id.get(edge_id) or {}
                remaining = max_assertions - assertions_emitted
                anchors, used = self._edge_anchors(edge, min(max_per_edge, remaining))
                if not anchors:
                    # 没有可定位证据的边不进上下文：图结论没有原句支撑时，
                    # 写进上下文只会让下游更容易编事实。
                    continue
                if assertions_emitted + len(anchors) > max_assertions:
                    truncated_reason = truncated_reason or f"max_assertions({max_assertions})"
                    stopped = True
                    break
                assertions_emitted += len(anchors)
                per_entity_taken[node_id] += 1
                key = (node_id, edge_id) if direction == "out" else (edge_id, node_id)
                seen_edge_keys.add(key)
                if direction == "in":
                    used_in_edges = True
                tier_counts[tier] += 1
                priority.setdefault(tier, _tier_reason(tier))
                neighbor_id = str(item.get("neighbor_id") or "")
                expansions.append(
                    {
                        "from_id": node_id,
                        "from_name": self.display_name(node_id),
                        "neighbor_id": neighbor_id,
                        "neighbor_name": self.display_name(neighbor_id),
                        "relation": str(item.get("relation") or ""),
                        "inverse_relation": str(item.get("inverse_relation") or ""),
                        "direction": str(item.get("direction") or direction),
                        "symmetric": bool(item.get("symmetric")),
                        "edge_id": edge_id,
                        "assertion_ids": list(item.get("assertion_ids") or []),
                        "support_count": int(item.get("support_count") or 0),
                        "independent_scene_count": int(item.get("independent_scene_count") or 0),
                        "priority_tier": tier,
                        "matched_tiers": tiers,
                        "depth": node_depth + 1,
                        "used_in_edges": direction == "in" or bool(used),
                        "chunk_id": anchors[0]["chunk_id"],
                        "evidence": anchors[0]["evidence"],
                        "evidence_spans": [
                            {"chunk_id": a["chunk_id"], "start": a["start"], "end": a["end"]}
                            for a in anchors
                        ],
                        "anchors": anchors,
                    }
                )
                if neighbor_id and neighbor_id not in visited and node_depth + 1 <= depth:
                    visited.add(neighbor_id)
                    if node_depth + 1 < depth:
                        frontier.append((neighbor_id, node_depth + 1))
            if stopped:
                break

        stats = {
            "seeds": len(seeds),
            "nodes_visited": len(visited_nodes),
            "expansions": len(expansions),
            "assertions_used": assertions_emitted,
            "used_in_edges": used_in_edges,
            "include_in": bool(include_in),
            "priority_tiers": {str(k): tier_counts[k] for k in sorted(tier_counts)},
            "priority_legend": {str(k): v for k, v in sorted(priority.items())},
            "truncated_reason": truncated_reason,
            "budget": {
                "max_depth": max_depth,
                "depth_used": depth,
                "max_nodes": max_nodes,
                "max_edges": max_edges,
                "max_degree_per_entity": max_degree,
                "max_assertions": max_assertions,
                "max_assertions_per_edge": max_per_edge,
                "timeout_ms": int(merged_budget["timeout_ms"]),
                "support_high": support_high,
            },
        }
        return {"expansions": expansions, "stats": stats}

    def _edge_anchors(self, edge: dict, remaining: int) -> tuple[list[dict], bool]:
        """取这条边可定位的观察锚点。`remaining` 是还能用几条断言。

        锚点必须来自 `assertions.jsonl`：有 `chunk_id` 与 `evidence_start/end`。
        没有断言数据时退回边自带的 `evidence` + `chunks[0]`，但 `start/end` 记 -1，
        让下游的偏移校验能明确拒绝——不假装知道位置。
        """
        if remaining <= 0:
            return [], False
        anchors: list[dict] = []
        for aid in list(edge.get("assertion_ids") or [])[:remaining]:
            rec = self.assertions_by_id.get(str(aid))
            if not rec:
                continue
            anchors.append(
                {
                    "assertion_id": str(aid),
                    "chunk_id": str(rec.get("chunk_id") or ""),
                    "evidence": str(rec.get("evidence_text") or ""),
                    "start": int(rec.get("evidence_start", -1)),
                    "end": int(rec.get("evidence_end", -1)),
                }
            )
        if anchors:
            return anchors, True
        cid = str((edge.get("chunks") or [""])[0])
        ev = str(edge.get("evidence") or "")
        if cid and ev:
            return (
                [{"assertion_id": "", "chunk_id": cid, "evidence": ev, "start": -1, "end": -1}],
                False,
            )
        return [], False

    # ------------------------------------------------------------ 上下文
    def build_context(
        self,
        hits,
        expansions,
        *,
        token_budget: int | None = None,
        query_analysis: dict | None = None,
    ) -> dict:
        """把检索命中与图扩展拼成上下文单元。

        规则（每条都对着一个真实失效模式）：

        * **同 chunk 只留一次**，`sources` 合并。同一句话被两条通道各带回一次，
          在上下文里出现两遍，会让人误以为有两个独立证据。
        * **相同证据的多条 assertion 合并**成一个 `evidence` 组，`assertion_ids` 累加。
        * **直接证据优先于图背景**：词法命中块排在纯图扩展块之前（`sources` 里有
          `bm25`/`entity` 即算直接证据）。
        * **截断不截引用锚点**：超预算时整单元丢弃，绝不切开某个单元的
          `evidence_spans`。切一半的引用比没有引用更坏——下游会以为自己有证据。
        * 每个单元都带 `chunk_id/doc_id/title/chapter/text`，因为引用校验要拿
          `text[start:end]` 逐字比对。

        返回 `{units:[...], context_digest, stats:{...}}`。
        """
        budget_tokens = int(
            token_budget if token_budget is not None else self.budget["max_context_tokens"]
        )
        qa = query_analysis or {}
        hint_predicates = {
            str(h.get("canonical") or h.get("raw") or "") for h in (qa.get("relation_hints") or [])
        }
        hint_predicates.discard("")
        focus_ids = {str(c.get("entity_id") or "") for c in (qa.get("entity_candidates") or [])}
        focus_ids.discard("")

        units: dict[str, dict] = {}
        skipped: list[dict] = []

        def ensure(chunk_id: str) -> dict | None:
            cid = str(chunk_id or "")
            if not cid:
                return None
            if cid in units:
                return units[cid]
            rec = self.chunks.get(cid)
            if not rec:
                skipped.append({"chunk_id": cid, "reason": "chunk_not_loaded"})
                return None
            unit = {
                "chunk_id": cid,
                "doc_id": str(rec.get("doc_id") or ""),
                "title": str(rec.get("title") or ""),
                "chapter": str(rec.get("chapter") or ""),
                "text": str(rec.get("text") or ""),
                "evidence_spans": [],
                "entities": [],
                "assertions": [],
                "score": 0.0,
                "sources": [],
                "graph_path": [],
            }
            units[cid] = unit
            return unit

        # 1) 检索命中（直接证据）
        for hit in hits or []:
            unit = ensure(hit.get("chunk_id"))
            if unit is None:
                continue
            unit["score"] = max(unit["score"], float(hit.get("score") or 0.0))
            for src in hit.get("sources") or []:
                if src not in unit["sources"]:
                    unit["sources"].append(dict(src))
            # 这块里与焦点实体/关系提示相关的断言，作为证据锚点
            for rec in self.assertions_by_chunk.get(unit["chunk_id"], []):
                if not _assertion_relevant(rec, focus_ids, hint_predicates):
                    continue
                _push_span(unit, rec)
                unit["assertions"].append(_assertion_brief(rec, self.display_name))
                for eid in (rec.get("subject_id"), rec.get("object_id")):
                    name = self.display_name(str(eid or ""))
                    if name and name not in unit["entities"]:
                        unit["entities"].append(name)

        # 2) 图扩展（背景证据）
        for exp in expansions or []:
            unit = ensure(exp.get("chunk_id"))
            if unit is None:
                continue
            if exp not in unit["graph_path"]:
                unit["graph_path"].append(exp)
            for anchor in exp.get("anchors") or []:
                rec = self.assertions_by_id.get(str(anchor.get("assertion_id") or ""))
                if rec is not None:
                    # 证据偏移只在自己的块正文里有效。一条边可能由多个块里的断言支撑，
                    # 把 anchors[1:] 的偏移登记到 anchors[0] 那块上，就等于报一个指向
                    # 别处的位置——下游 `text[start:end] == evidence` 必然对不上
                    # （真实语料实测：105 条引用里 24 条这样坏掉）。所以按锚点自己的
                    # chunk_id 找单元再登记。
                    host_id = str(rec.get("chunk_id") or "")
                    # 宿主块必须自己载入得回来。**不许退回 unit**：退回等于把
                    # A 块的偏移登记到 B 块的正文上，下游 text[start:end] 必然
                    # 对不上，而这条坏引用会被当成「引用不成立」静默剔除，表现
                    # 为「答得短」而不是「服务坏了」（实测：q-sc-000024）。
                    # 此时已经没有正确的偏移可用，只能不登记。
                    target = ensure(host_id) if host_id else None
                    if target is None:
                        continue
                    _push_span(target, rec)
                    brief = _assertion_brief(rec, self.display_name)
                    if brief not in target["assertions"]:
                        target["assertions"].append(brief)
                    for eid in (rec.get("subject_id"), rec.get("object_id")):
                        name = self.display_name(str(eid or ""))
                        if name and name not in target["entities"]:
                            target["entities"].append(name)
            for name in (exp.get("from_name"), exp.get("neighbor_name")):
                if name and name not in unit["entities"]:
                    unit["entities"].append(name)
            if not any(s["channel"] == SOURCE_GRAPH for s in unit["sources"]):
                unit["sources"].append({"channel": SOURCE_GRAPH, "score": 0.0})

        def direct(unit: dict) -> bool:
            return any(s["channel"] in (SOURCE_BM25, SOURCE_ENTITY) for s in unit["sources"])

        ordered = sorted(
            units.values(),
            key=lambda u: (0 if direct(u) else 1, -u["score"], u["chunk_id"]),
        )

        kept: list[dict] = []
        used_tokens = 0
        truncated: list[str] = []
        for unit in ordered:
            cost = estimate_tokens(
                f"{unit['title']} {unit['chapter']} {unit['text']}",
                tokenizer_version=getattr(self.index, "tokenizer_version", index_mod.TOKENIZER_VERSION),
                name_terms=getattr(self.index, "name_terms", None),
            )
            unit["tokens"] = cost
            if used_tokens + cost > budget_tokens:
                truncated.append(unit["chunk_id"])
                continue
            used_tokens += cost
            kept.append(unit)

        payload = [
            {
                "chunk_id": u["chunk_id"],
                "evidence": [s["evidence"] for s in u["evidence_spans"]],
                "assertion_ids": [a["assertion_id"] for a in u["assertions"]],
            }
            for u in kept
        ]
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

        return {
            "units": kept,
            "context_digest": digest,
            "stats": {
                "units": len(kept),
                "candidates": len(units),
                "tokens_used": used_tokens,
                "token_budget": budget_tokens,
                "truncated_chunks": truncated,
                "skipped_units": skipped,
                "direct_units": sum(1 for u in kept if direct(u)),
                "graph_only_units": sum(1 for u in kept if not direct(u)),
                "evidence_spans": sum(len(u["evidence_spans"]) for u in kept),
            },
        }


def _tier_reason(tier: int) -> str:
    return {
        1: "查询实体的直接出边（深度 1）",
        2: "谓词命中 relation_hints",
        3: "support_count >= support_high",
        4: "与已召回块共现",
        5: "证据带 scene_order 且查询含时间词（仅标记，不做时序推理）",
        99: "未命中任何优先级档，仅因预算允许而纳入",
    }.get(tier, "未知档")


def _assertion_relevant(rec: dict, focus_ids: set[str], hint_predicates: set[str]) -> bool:
    """这条断言是否与当前问句相关。

    判据只有两条，都可机检：端点落在焦点实体上，或谓词命中关系提示。
    刻意不按"描述像不像"判断——那会把判据变成主观。
    """
    if str(rec.get("subject_id") or "") in focus_ids:
        return True
    if str(rec.get("object_id") or "") in focus_ids:
        return True
    if hint_predicates and str(rec.get("predicate") or "") in hint_predicates:
        return True
    return False


def _push_span(unit: dict, rec: dict) -> None:
    """登记一条断言的证据区间。相同证据只留一次，偏移取首次出现的合法位置。"""
    ev = str(rec.get("evidence_text") or "")
    start = int(rec.get("evidence_start", -1))
    end = int(rec.get("evidence_end", -1))
    for span in unit["evidence_spans"]:
        if span["evidence"] == ev and span["start"] == start and span["end"] == end:
            return
    unit["evidence_spans"].append(
        {
            "assertion_id": str(rec.get("assertion_id") or ""),
            "evidence": ev,
            "start": start,
            "end": end,
        }
    )


def _assertion_brief(rec: dict, display) -> dict:
    return {
        "assertion_id": str(rec.get("assertion_id") or ""),
        "subject": display(str(rec.get("subject_id") or "")),
        "predicate": str(rec.get("predicate") or ""),
        "object": display(str(rec.get("object_id") or "")),
        "evidence": str(rec.get("evidence_text") or ""),
        "confidence": rec.get("confidence"),
        "chunk_id": str(rec.get("chunk_id") or ""),
    }


__all__ = [
    "DEFAULT_BUDGET",
    "ENTITY_CHANNEL_WEIGHT",
    "SOURCE_BM25",
    "SOURCE_ENTITY",
    "SOURCE_GRAPH",
    "Retriever",
    "estimate_tokens",
]
