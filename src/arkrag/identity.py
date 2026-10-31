"""E3 实体身份：Mention -> EntityCandidate -> CanonicalEntity 三层模型。

本模块只放**判定逻辑**（纯函数）：类型归一、稳定 id、合并前六道检查、合并执行。
它不建图、不写盘、不读缓存——那些是 L5 的事。契约见 `docs/e阶段契约.md` §3。

三条不变量，各自有代价，都写在这里免得后人重犯：

1. **名称只是展示字段**（§3.2）。id 由 `stable_id()` 在建组时算一次；之后改
   `CanonicalEntity.name` 不得改 id。旧实现拿名字直接当 id，改名就等于换实体。
2. **类型归一顺序固定**（§3.4）：规范值精确 -> 精确同义词 -> 受限模糊 -> 兜底。
   旧实现把最松的子串判据放在第 2 位，于是「人物设定」被判成 角色。
3. **宁可拆多不可错并**（§3.3）。六道检查顺序不可调换：先命中的那条是拒绝理由，
   顺序一改，报出来的 reason 就变成别的检查，事后追责时分不清到底为什么没并。

判据只有一份：`normalize_type` 是本仓库唯一的类型归一实现，`norm_entity.norm_type`
转调它（历史上曾经两处各写一份，结果总是较松的那处生效）。
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from . import vocab

ID_PREFIX = "e-"
CONCEPT = "概念"

#: 契约 §3.3 的六道检查，顺序即优先级，不可调换。
REJECT_REASONS = (
    "type_conflict",
    "member_index",
    "generic_anchor",
    "ambiguous_alias",
    "component_conflict",
    "relation_conflict",
)

#: 契约 §3.1 的别名来源。
ALIAS_SOURCES = ("upstream", "fullname", "alias_chain", "manual")

#: 契约 §3.1 的审核状态。
REVIEW_STATES = ("auto", "pending", "approved", "rejected")

#: 合并成功后并进来的名字，落进 `merged_from[].reason`（契约 §5.3 的 L4.13）。
MERGE_REASONS = ("alias", "fullname", "alias_chain")

#: 自环哨兵：合并之后组内成员之间的边变成自指，不能再按名字分桶。
_SELF = "\x00self"


# ------------------------------------------------------------------ 基础工具


def doc_id_of(chunk_id: str) -> str:
    """`doc_id#NNN` -> `doc_id`；没有 `#` 时原样返回。

    1 doc = 1 scene 是实测结论（契约 §1），所以场景身份直接取 doc_id。
    用 `rpartition` 而不是 `split("#")[0]`：doc_id 里若含 `#`，切第一个会切错。
    """
    head, sep, _ = str(chunk_id or "").rpartition("#")
    return head if sep else str(chunk_id or "")


def stable_id(canonical_name: str, *, entity_rule_version: str, salt: str = "") -> str:
    """稳定实体 id：`"e-" + sha256(entity_rule_version \\x1f canonical_name)[:16]`。

    ## 为什么必须有 salt

    两个**不同**的实体可能恰好同名：活动剧情里两个村长都叫「陈」，两个都是真实体。
    纯名字哈希会把它们压成同一个 id，等于在身份层就把人并了，后面六道检查再严也
    无从补救。所以调用方传场景前缀作 salt：

        stable_id("陈", entity_rule_version="ev2", salt="activities/act1/x")

    salt 为空时不进哈希，构成与契约 §3.2 字面完全一致的两个分量——这样「不需要
    salt 的场景」与契约文本仍逐字节对得上；一旦传了 salt，第三个分量接在后面。

    ## 为什么改名不改 id

    本函数只吃 `canonical_name`（**建组时的名字**），产物由调用方固定下来写进
    `split_history`。`CanonicalEntity.rename()` 只改展示名、不动 id。
    契约 §3.2 的硬要求：`凯尔希医生` 的展示名换成 `凯尔希`，id 必须不变。

    ## 一处如实说明

    「建组时的名字」在本实现里取组内字典序最小的候选名，所以**同一批输入**重跑
    得到同一个 id；但若将来有新的、排序更靠前的别名出现并并进这一组，重算会得到
    另一个 id。跨世代真正稳定要靠 `merge_candidates(pinned_ids=...)` 把上一次的
    id 钉住——契约说的「算一次就固定」在无状态纯函数里只能做到这一步。
    """
    parts = [str(entity_rule_version), str(canonical_name)]
    if salt:
        parts.append(str(salt))
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]
    return ID_PREFIX + digest


# ------------------------------------------------------------------ 类型归一


def _fuzzy_hit(text: str) -> str:
    """受限模糊匹配：命中项必须**长于** text 的一半，否则不算命中。

    这是契约 §3.4 第 ③ 步。写成 `2 * len(hit) > len(text)` 而**不是** `>=`：
    契约原文是「占 text 长度一半以上」，「一半以上」按汉字语义是严格大于。
    边界实测：`人物设定`（命中 `人物`，2 字 / 全长 4 字）恰好是一半，因此**不命中**,
    落到兜底 `概念`；旧实现会判成 角色（子串在最前面）。

    多个候选命中时取最长者；等长时规范类型优先于同义词键（同义词表可能把
    `生物` 映射到 角色 而 `角色` 本身是规范值，同一份 text 两个都命中时以规范值为准）。
    """
    if len(text) < 2:
        return ""
    best: tuple[int, int, str] = (0, 1, "")
    for canonical in vocab.ENTITY_TYPES:
        pos = text.find(canonical)
        if pos < 0:
            continue
        if 2 * len(canonical) > len(text):
            cand = (len(canonical), 0, canonical)
            if cand > best:
                best = cand
    for key, val in vocab.TYPE_SYNONYMS.items():
        if key not in text:
            continue
        if 2 * len(key) > len(text):
            cand = (len(key), 1, val)
            if cand > best:
                best = cand
    return best[2]


def normalize_type(raw: object, *, strict_order: bool = True) -> str:
    """实体类型归一。契约 §3.4 的四步，顺序不可调换。

    1. 规范值精确匹配（`text in ENTITY_TYPES`）
    2. 精确同义词（`TYPE_SYNONYMS[text]`）
    3. 受限模糊匹配（仅当 text 长度 >= 2 且命中项**长于** text 一半）
    4. 兜底 `概念`

    顺序本身的理由：子串判据是最松的那个，放在最前面等于让最松的判据覆盖最紧的，
    `人物设定` 就会被判成 角色。这不是理论问题——旧实现 `norm_entity.norm_type`
    就是这个顺序，真实数据里确实有这类词。

    兜底只能是 `概念`，不许给「未分类」：下游所有按类型过滤的代码路径都会因为
    多出一个词表外的值而静默失效（计划书 §4.1）。

    `strict_order=False` 只用于诊断历史行为（复现旧顺序），**生产路径不许用**。
    """
    if not isinstance(raw, str):
        return CONCEPT
    text = raw.strip()
    if not text:
        return CONCEPT
    if text in vocab.ENTITY_TYPES:
        return text
    hit = vocab.TYPE_SYNONYMS.get(text)
    if hit:
        return hit
    if not strict_order:
        # 旧顺序（精确 -> 子串 -> 同义词 -> 兜底）。保留是因为「新旧数字不一致时
        # 要能指出差在哪一条」，不是为了给生产用。
        for canonical in vocab.ENTITY_TYPES:
            if canonical in text or text in canonical:
                return canonical
        for key, val in vocab.TYPE_SYNONYMS.items():
            if key in text:
                return val
        return CONCEPT
    return _fuzzy_hit(text) or CONCEPT


# ------------------------------------------------------------------ 三层数据结构


@dataclass
class Mention:
    """逐块提及：上游抽取给的原始名，一条 = 某块里出现过这个名字。

    `raw_type` 是上游原词（可能空、可能是词表外的同义词），归一在
    `build_candidates()` 里做——Mention 层不做任何删改，这样「原始观察」永远可回查。
    """

    name: str
    chunk_id: str
    raw_type: str = ""
    desc: str = ""
    aliases: tuple[str, ...] = ()

    @property
    def doc_id(self) -> str:
        return doc_id_of(self.chunk_id)

    @property
    def scene_id(self) -> str:
        """场景实例身份：1 doc = 1 scene（契约 §1 实测），故与 doc_id 相同。"""
        return self.doc_id


@dataclass
class EntityCandidate:
    """候选实体：同名提及聚合后的样子，可能同名多义（由 salt / 分层调用方分开）。"""

    name: str
    etype: str = CONCEPT
    desc: str = ""
    aliases: tuple[str, ...] = ()
    #: 别名 -> 该别名出现过的场景集合（从 mention 的 chunk_id 推出）
    alias_scenes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    mentions: int = 0
    scenes: frozenset[str] = frozenset()
    chunks: frozenset[str] = frozenset()

    @property
    def claim_names(self) -> set[str]:
        """本候选主张的全部字符串：自己的名字 + 上游声明的别名。"""
        return {self.name} | set(self.aliases)


@dataclass
class CanonicalEntity:
    """最终实体。`name` 是展示字段，`id` 才是身份（契约 §3.1 / §3.2）。"""

    id: str
    name: str
    etype: str
    desc: str = ""
    aliases: list[dict] = field(default_factory=list)
    merged_from: list[dict] = field(default_factory=list)
    split_history: list[dict] = field(default_factory=list)
    mentions: int = 0
    chunks: list[str] = field(default_factory=list)

    def rename(self, new_name: str) -> "CanonicalEntity":
        """只改展示名。id 不动——这正是契约 §3.2 那条硬要求的落点。"""
        self.name = str(new_name)
        return self

    def to_dict(self) -> dict:
        """契约 §3.1 的 JSON 形态（字段名以契约为准：`type` 而不是 `etype`）。

        **别名按审核状态分成三张表，因为 `review` 本身没有任何强制力**：

        * `aliases` —— 只有 `review in ("auto", "approved")`。这是**权威检索表**，
          下游 `graph.py` 的 `alias_index` / `index.py` 的 `build_alias_index` 直接
          拿它建「别名 → id」是**正确**的写法，不需要懂 `review` 语义。
        * `pending_aliases` —— `review == "pending"`，未审的**候选**别名，检索不可见。
        * `rejected_aliases` —— `review == "rejected"`，显式拒绝，仅留档。

        为什么必须这样分，而不是只留一个 `review` 字段：判据只能有一处事实源，
        其余地方必须**引用**它而不是重写它（本项目这一阶段踩过三次：冻结清单/写保护
        清单两份、corpus_meta/authority 两份、release/source 两套「已验证」判据）。
        实测过一次：拒绝合并 `暴雨 <- 库兰塔士兵` 之后，别名表里仍写着
        `{"text": "库兰塔士兵", "review": "pending"}`，而 `graph.py:181` 的
        `{alias: ent["id"] for alias in ent["aliases"]}` 照样把它指向暴雨——
        **同一个错误从「并实体」转移到了「并别名」，只是晚一步发作**。
        未审的别名不是别名，是候选；放进 `aliases` 就等于默认它是真的。
        所以本函数**让产物不携带危险信息**，而不是指望每个下游都记得过滤。
        """
        authoritative = [
            dict(a) for a in self.aliases if a.get("review") in ("auto", "approved")
        ]
        pending = [dict(a) for a in self.aliases if a.get("review") == "pending"]
        rejected = [dict(a) for a in self.aliases if a.get("review") == "rejected"]
        return {
            "id": self.id,
            "name": self.name,
            "type": self.etype,
            "desc": self.desc,
            "aliases": authoritative,
            "pending_aliases": pending,
            "rejected_aliases": rejected,
            "merged_from": [dict(m) for m in self.merged_from],
            "split_history": [dict(s) for s in self.split_history],
            "mentions": self.mentions,
            "chunks": list(self.chunks),
        }


@dataclass
class MergeDecision:
    """一次防误并检查的结论。`reason` 只在 `allowed=False` 时非空。"""

    allowed: bool
    reason: str = ""
    detail: str = ""
    #: 哪一道检查给出的结论（"pass" / "type" / "member" / "generic" / "alias" /
    #: "component" / "relation"）。只用于解释，不参与判定。
    check: str = ""

    def __post_init__(self) -> None:
        if self.allowed and self.reason:
            raise ValueError("允许合并时 reason 必须为空串（拒绝理由不能带过）")
        if not self.allowed and self.reason not in REJECT_REASONS:
            raise ValueError(f"未知拒绝理由 {self.reason!r}，只允许 {REJECT_REASONS}")


class _UnionFind:
    """最小并查集：**只是容器，不含任何判据**（判据全在 `check_merge`）。

    根恒为组内字典序最小的名字，所以合并结果与遍历顺序无关；这与
    `norm_entity.UnionFind` 同一约定，但本模块不 import 它——identity 是
    norm_entity 的下层（norm_entity.norm_type 要转调本模块），反向依赖会绕成环。
    """

    def __init__(self, names: Iterable[str] = ()) -> None:
        self.parent: dict[str, str] = {n: n for n in names}

    def add(self, x: str) -> None:
        self.parent.setdefault(x, x)

    def find(self, x: str) -> str:
        self.add(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if rb < ra:
            ra, rb = rb, ra
        self.parent[rb] = ra


# ------------------------------------------------------------------ 判定用的谓词


def _generic_sets() -> set[str]:
    return set(vocab.GENERIC_PERSON_NOUNS) | set(vocab.ORG_SUFFIXES) | set(vocab.MEMBER_SUFFIXES)


def is_generic_anchor(name: str) -> bool:
    """这个名字本身是不是「职务/组织泛称」形态（契约 §3.3 第 3 条）。

    判据是「整个名字就是一个泛称标记」，不是「以泛称结尾」：

    * `士兵` / `医生` / `医疗干员` / `协会` -> 泛称锚点（`医疗干员` 以 `干员` 结尾且
      前缀是职务词，实测就是计划书里说的「泛称锚点吸收专名」那一类）；
    * `凯尔希医生` **不是**泛称锚点——它是全名，若按「以职位结尾」判会将
      `凯尔希 <- 凯尔希医生` 这条契约自己举的例子打成拒绝，那是把判据用反了。

    所以：完全等于词表里的泛称，或者「以成员后缀结尾且去掉后缀后是一个泛称/职务词」。

    `ORG_SUFFIXES` 同样按「整名即后缀词」理解（`协会`、`公司`），**不**按结尾判：
    `罗德岛制药公司` 若算泛称锚点，就永远不能与 `罗德岛` 做全名归并，那是把
    §3.1 里 `merged_from: fullname` 的用法整个废掉。

    `人事干员` 这类「部门词 + 职务后缀」的名字本函数判 False——因为 `人事` 不在
    词表里，而词表是只读的，本模块不许再自造第二份词表。这一形态改由
    `is_generic_anchor_by_shape()` 用结构判据补上，不靠扩充词表。
    """
    text = (name or "").strip()
    if not text:
        return False
    if text in _generic_sets():
        return True
    heads = set(vocab.GENERIC_PERSON_NOUNS) | set(vocab.ORG_SUFFIXES) | set(vocab.MEMBER_SUFFIXES)
    for suffix in vocab.MEMBER_SUFFIXES:
        if not text.endswith(suffix) or len(text) <= len(suffix):
            continue
        if text[: -len(suffix)] in heads:
            return True
    return False


def _shape_suffixes() -> tuple[str, ...]:
    """别名一侧的后缀表：既有词表的并集，**不自造新表**（vocab 只读）。"""
    return tuple(vocab.MEMBER_SUFFIXES) + tuple(vocab.ORG_SUFFIXES)


def _role_suffixes() -> tuple[str, ...]:
    """「名字 vs 名字」那一侧只认**职务**后缀，不认组织后缀。

    组织后缀用来判「这个字符串是不是组织泛称」（`公司`、`协会`），拿它去判两个
    组织名之间能不能并会把真东西打成拒绝：实测 `第一集团军` 以 `团军` 结尾
    （`ORG_SUFFIXES` 里那条是 `军团` 的倒序录入，词表只读、不能修）、
    `伊比利亚平民海军舰队` 以 `舰队` 结尾，两者都是正当专名。职务后缀（`干员`、
    `术师`、`士兵`、`队长`）才对应契约 §3.3 第 3 条说的「人名与职务泛称分离」。
    """
    return tuple(vocab.MEMBER_SUFFIXES)


def is_generic_anchor_by_shape(
    alias: str, canonical: str, *, suffixes: Sequence[str] | None = None
) -> bool:
    """「修饰语 + 职务后缀」当别名用 —— 契约 §3.3 第 3 条的结构补充判据。

    剧本给无名路人起的显示名，形态是「描述/种族/职业 + 职务词」：`老练的士兵`、
    `落魄商人`、`拍摄队长`、`黑帮首领`、`罗德岛外勤干员`、`人事干员`。它们是**角色
    属性**，不是专名。把这种字符串当别名并到具名角色名下，等于断言「这里所有无名
    士兵都是暴雨」——契约 §3.3 要拦的正是这一类。

    三条判据，后两条是**实测挡掉真实误伤的前置**，删任何一条都会把合法别名打进拒绝：

    1. `alias` 整体以职务/组织后缀结尾，且**不是后缀本身**
       （`干员` 这种纯后缀交给 `is_generic_anchor`，不在这里重复判）；
    2. 互不包含 —— 互相包含说明这是**全名归并**，放行：
       `凯尔希医生` / `罗德岛制药公司` / `…自治州住宅管理委员会`；
    3. 首字不同 —— 共享首字说明这是**姓氏/缩写式别名**，放行：
       `魏彦吾 <- 魏大长官`、`麟青砚 <- 麟长官`。

    `suffixes` 让调用方按方向换后缀表：别名一侧用全表（`我的家族` / `总公司`
    这类组织泛称也只能在这一侧判），而「名字 vs 名字」那一侧只能用职务词表
    （见 `_ROLE_SUFFIXES` 的说明）。

    纯结构判据，不查任何外部资料（计划书第 9 条原则：联网资料不得进入产物）。
    """
    a = (alias or "").strip()
    c = (canonical or "").strip()
    if not a or not c or a == c:
        return False
    table = tuple(suffixes) if suffixes is not None else _shape_suffixes()
    # 1) 必须以职务/组织后缀结尾，且后缀之外还有内容
    if not any(a.endswith(s) and len(a) > len(s) for s in table):
        return False
    # 2) 互相包含 = 全名归并，放行
    if c.startswith(a) or a.startswith(c) or c.endswith(a) or a.endswith(c):
        return False
    # 3) 共享首字 = 姓氏/缩写式别名，放行
    if c[0] == a[0]:
        return False
    return True


def _member_index_pair(a: str, b: str) -> tuple[bool, str]:
    """「集体 vs 成员」判据，带前缀前置条件（契约 §3.3 第 2 条）。

    前置条件缺不得：`is_member_index` 只切后缀，直接拿「克里斯」和「罗德岛工程干员」
    去问会切出「岛工程干员」并判真（决策台账 D-24 的真实事故，36 条假阳性）。所以
    本函数先要求 `long.startswith(short)`。

    但**只要求前缀，不要求分隔符**：分隔符（`FULLNAME_SEPS`）是 `can_merge_fullname`
    那个问题的前置条件——「全名归并必须是有结构的复合名」。成员索引问的是另一件事：
    `整合运动` 与 `整合运动成员` 之间没有分隔符，却正是契约 §3.3 第 2 条要拦的
    「集体与成员」。把分隔符要求照抄过来会让这道检查对真实数据几乎全盲（实测
    `整合运动成员` / `整合运动干部` / `罗德岛工程干员` 全部漏判）。
    """
    from . import norm_entity  # 延迟导入：norm_entity 顶层依赖本模块

    for short, long in ((a, b), (b, a)):
        if not short or not long or len(long) <= len(short):
            continue
        if not long.startswith(short):
            continue
        if norm_entity.is_member_index(short, long):
            return True, f"{long!r} 是 {short!r} 的成员集合，不是同一个实体"
    return False, ""


def _check_ambiguous_alias(
    text: str, claimants: Mapping[str, Sequence[str]] | None
) -> tuple[bool, str]:
    """歧义别名判据（契约 §3.3 第 4 条）。**唯一实现**：合并边与候选构建都调它。

    被 >= 2 个候选同时主张的字符串没有资格当别名：「殿下」「将军」「老爷」在真实
    数据里注定被不同的人各自认领，当别名边用会把互不相干的人串成一个大组。
    """
    if not text or not claimants:
        return True, ""
    owners = claimants.get(text) or ()
    if len(owners) >= 2:
        return False, f"别名 {text!r} 被 {len(owners)} 个候选同时主张：{sorted(owners)[:4]}"
    return True, ""


def _edge_pairs(edges: Iterable) -> list[tuple[str, str]]:
    """把调用方给的边归一成 `(relation, other_name)`。

    接受 `(relation, other)` 二元组，也接受 `{"relation"/"predicate", "target"/
    "object"}` 形态的 dict——建图期两种都出现过，硬要调用方先转换只会多一处判据。
    """
    out: list[tuple[str, str]] = []
    for item in edges or ():
        if isinstance(item, Mapping):
            rel = item.get("relation") or item.get("predicate") or ""
            other = item.get("target") or item.get("object") or item.get("other") or ""
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            rel, other = item[0], item[1]
        else:
            continue
        rel, other = str(rel or "").strip(), str(other or "").strip()
        if rel:
            out.append((rel, other))
    return out


def _exclusive_pairs() -> list[tuple[str, str]]:
    return [(str(a), str(b)) for a, b in vocab.MUTUALLY_EXCLUSIVE_PAIRS]


# ------------------------------------------------------------------ 六道检查


def _check_type(a: EntityCandidate, b: EntityCandidate) -> MergeDecision:
    """第 1 条：类型兼容。`概念` 是万能兼容项，两个非概念类型不同即拒绝。"""
    if a.etype == b.etype or CONCEPT in (a.etype, b.etype):
        return MergeDecision(True, "", f"{a.etype} / {b.etype}", "pass")
    return MergeDecision(
        False, "type_conflict", f"{a.name}({a.etype}) 与 {b.name}({b.etype}) 类型不同", "type"
    )


def _check_member(a: EntityCandidate, b: EntityCandidate) -> MergeDecision:
    """第 2 条：集体 vs 成员。是成员就绝不并成本体。"""
    bad, detail = _member_index_pair(a.name, b.name)
    if bad:
        return MergeDecision(False, "member_index", detail, "member")
    return MergeDecision(True, "", "", "pass")


def _mutual_epithet(a: EntityCandidate, b: EntityCandidate) -> bool:
    """两个候选**互相**把对方的名字声明为别名，且至少共处一个场景。

    这是「同一人的两个称呼」在数据里的可判定形态，用来把泛称形状判据的误伤
    区分出来（见 `_check_generic` 的说明）：

    * 互相 —— 泛称短语不会反过来把专名写成自己的别名。单向声明
      （`米纳 <- 医疗干员`、`莉塔 <- 医疗干员`、`纳吉尔 <- 外勤干员`、
      `罗莲娜 <- 整合运动术师`）仍然是「描述短语被当别名」，照旧拒绝；
    * 同场共现 —— 同一场景里两个人同时出场、互相这么叫，才谈得上同一人。
      只在各自场景里互相声明的（跨文档同义指代）证据不足，交人工。

    真实数据上这一条**恰好命中两对**（全库扫描：形如「名字 vs 名字」且被形状判据
    命中的候选对共 6 对，其中互相主张的只有这 2 对）：`科瓦尔 <-> 老工匠`、
    `阿卜大叔 <-> 老队长`。PRTS「剧情角色一览」写明「老工匠，乌萨斯人，原名科瓦尔」，
    `阿卜大叔`/`老队长` 同在 `obt/memory/story_zebra_1_1` 由暴雨交替称呼，
    故两对都是真同一人，形状判据在此属于误伤。

    注意这只是 shape 层的放行条件：整名判据 `is_generic_anchor` 在它之前已经
    返回过拒绝，`医疗干员` 这类真泛称名字仍然拦得住。
    """
    an, bn = a.name, b.name
    if not an or not bn:
        return False
    if bn not in a.aliases or an not in b.aliases:
        return False
    return bool(set(a.scenes) & set(b.scenes))


def _check_generic(
    a: EntityCandidate, b: EntityCandidate, link: str = ""
) -> MergeDecision:
    """第 3 条：人名 vs 职务/组织泛称。任一方是泛称形态即拒绝（契约字面：一方是就拒）。

    两层判据，**加强而非替换**：

    * `is_generic_anchor(name)`：整个名字就是泛称标记（`医疗干员`、`士兵`、`协会`）；
    * `is_generic_anchor_by_shape(alias, canonical)`：别名是「修饰语 + 职务后缀」
      （`老练的士兵`、`人事干员`、`黑帮首领`）。这一层需要知道**谁是别名**。

    方向判定必须**对称**：两个名字互相都试一遍，再加 `link`/主张集合里的
    非候选文本各试两个方向。只按「`link` 命中谁」单向判会漏掉反方向的边——
    实测 `科瓦尔 <- 老工匠` 与 `老工匠 <- 科瓦尔` 在数据里**两向都有**，单向判法
    拒掉一条、另一条照样把两组合成一组，等于判据没生效。契约 §3.3 第 3 条的
    字面就是「**一方**是泛称形态即拒」，对称才是字面含义，也不依赖边方向。

    **一处实测得来的例外（限于 shape 层，整名层不放宽）**：两个名字**互相**把对方
    声明为别名，且**至少共处一个场景**（`mutual_epithet`）。真实数据里正好两对：
    `科瓦尔 <-> 老工匠`（PRTS 剧情角色一览：老工匠，乌萨斯人，原名科瓦尔）、
    `阿卜大叔 <-> 老队长`（同在 `obt/memory/story_zebra_1_1`，暴雨对他的两种称呼）。
    这种「互相指认 + 同场共现」是**同一人的两个称呼**，不是「描述短语被当别名」——
    泛称短语不会反过来把专名写成自己的别名。只有单向主张的（`米纳 <- 医疗干员`、
    `莉塔 <- 医疗干员`、`纳吉尔 <- 外勤干员`、`罗莲娜 <- 整合运动术师`）照旧拒绝。
    """
    a_gen, b_gen = is_generic_anchor(a.name), is_generic_anchor(b.name)
    if a_gen or b_gen:
        who = a.name if a_gen else b.name
        return MergeDecision(
            False,
            "generic_anchor",
            f"{who!r} 是职务/组织泛称形态，不能与人名/专名并成一个实体",
            "generic",
        )

    if not _mutual_epithet(a, b):
        pairs: list[tuple[str, str, tuple[str, ...] | None]] = [
            (a.name, b.name, _role_suffixes()),
            (b.name, a.name, _role_suffixes()),
        ]
        texts = {link} if link else (a.claim_names | b.claim_names)
        for text in sorted(t for t in texts if t):
            if text in (a.name, b.name):
                continue
            pairs.append((text, a.name, None))
            pairs.append((text, b.name, None))
        for alias, canonical, table in pairs:
            if is_generic_anchor_by_shape(alias, canonical, suffixes=table):
                return MergeDecision(
                    False,
                    "generic_anchor",
                    f"{alias!r} 是「修饰语 + 职务后缀」的泛称短语，不能当作 {canonical!r} 的别名",
                    "generic",
                )
    return MergeDecision(True, "", "", "pass")


def _check_alias(
    a: EntityCandidate,
    b: EntityCandidate,
    claimants: Mapping[str, Sequence[str]] | None,
    link: str,
) -> MergeDecision:
    """第 4 条：歧义别名。被多方主张的字符串既不能用来合并，也不进任何 aliases。

    `link` 是这次合并的凭据（别名文本）；没给时用两侧主张集合的交集，
    这样调用方漏传也不会让这道检查静默失效。
    """
    texts = {link} if link else (a.claim_names & b.claim_names)
    for text in sorted(t for t in texts if t):
        ok, detail = _check_ambiguous_alias(text, claimants)
        if not ok:
            return MergeDecision(False, "ambiguous_alias", detail, "alias")
    return MergeDecision(True, "", "", "pass")


def _check_component(
    a: EntityCandidate,
    b: EntityCandidate,
    group: Sequence[EntityCandidate],
) -> MergeDecision:
    """第 5 条：连通分量检查。Union 之后整组不许出现类型冲突或成员索引关系。

    只查 a-b 一条边是不够的：真正的误并往往发生在「新成员与组里第三个名字」
    之间（A 概念 吸收了 B，再吸收 C 时 B 与 C 类型冲突）。
    """
    members: list[EntityCandidate] = []
    seen: set[str] = set()
    for cand in list(group) + [a, b]:
        if cand is None or cand.name in seen:
            continue
        seen.add(cand.name)
        members.append(cand)
    for i, left in enumerate(members):
        for right in members[i + 1 :]:
            if left.etype != right.etype and CONCEPT not in (left.etype, right.etype):
                return MergeDecision(
                    False,
                    "component_conflict",
                    f"并成一组的 {left.name}({left.etype}) 与 {right.name}({right.etype}) 类型冲突",
                    "component",
                )
            bad, detail = _member_index_pair(left.name, right.name)
            if bad:
                return MergeDecision(False, "component_conflict", detail, "component")
    return MergeDecision(True, "", "", "pass")


def _check_relations(
    a: EntityCandidate,
    b: EntityCandidate,
    group: Sequence[EntityCandidate],
    edges_by_name: Mapping[str, Iterable] | None,
) -> MergeDecision:
    """第 6 条：冲突关系。**这次 Union 新造出来**的互斥关系才拒绝。

    判据：合并把不同名字的边压到同一个节点上。若某个目标上互斥的两条关系来自
    **不同的名字**，那是合并造成的新矛盾 → 撤销。

    为什么不是「组内出现过互斥对就拒」：真实数据里同一个名字自己就带
    `凯尔希 --同盟--> 博士` 与 `凯尔希 --敌对--> 博士`（两个不同块分别抽出来的），
    实测这类「单人自相矛盾」有 120 处。这种矛盾合并前就存在、合并后也存在，
    撤销合并并不会让它消失，只会让同一个人留下两个实体。判据问的是「并了以后
    多出什么」，不是「并了以后有什么」。

    为什么用「来源名字不同」而不是「落在 a 侧还是 b 侧」：调用方传进来的 `group`
    已经含两侧整个连通分量，按 a/b 分侧会让两侧都以同一个 `group` 为底，最后
    退化成同一侧、判据静默失效（第一版就是这么错的）。按「哪条边来自哪个名字」
    判既不需要知道分组，也对称。

    组内成员之间的边在合并后变成自环，按同一个目标（哨兵 `_SELF`）归桶，所以
    「A 同盟 B 且 B 敌对 A」这种互相矛盾同样会被拦下。
    """
    if not edges_by_name:
        return MergeDecision(True, "", "", "pass")

    members = {c.name for c in list(group) + [a, b] if c is not None}
    # 目标 -> 关系 -> 挂这条边的名字集合
    rel_owners: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for name in members:
        for rel, other in _edge_pairs(edges_by_name.get(name) or ()):
            target = _SELF if other in members else other
            rel_owners[target][rel].add(name)
    for target, rels in sorted(rel_owners.items()):
        for r1, r2 in _exclusive_pairs():
            s1, s2 = rels.get(r1) or set(), rels.get(r2) or set()
            if not s1 or not s2:
                continue
            if s1 == s2 and len(s1) == 1:
                # 互斥的两条关系都来自同一个名字：矛盾合并前就在它身上，
                # 这次合并没有新造任何东西。宁可让那一个人自己带着矛盾，
                # 也不要因为它而把两个实体永远拆开。
                continue
            where = "（合并后自指）" if target == _SELF else f" 到 {target!r}"
            return MergeDecision(
                False,
                "relation_conflict",
                f"合并后在同一实体{where}上同时存在互斥关系 {r1}/{r2}",
                "relation",
            )
    return MergeDecision(True, "", "", "pass")


def check_merge(
    cand_a: EntityCandidate,
    cand_b: EntityCandidate,
    *,
    alias_claimants: Mapping[str, Sequence[str]] | None = None,
    group: Sequence[EntityCandidate] = (),
    edges_by_name: Mapping[str, Iterable] | None = None,
    link: str = "",
) -> MergeDecision:
    """契约 §3.3 的六道检查，**顺序不可调换**，任一不过即拒绝这次 Union。

    | # | 检查 | 拒绝理由 |
    |---|---|---|
    | 1 | 类型兼容（概念万能） | `type_conflict` |
    | 2 | 集体 vs 成员 | `member_index` |
    | 3 | 人名 vs 职务/组织泛称 | `generic_anchor` |
    | 4 | 歧义别名（被 >=2 候选主张） | `ambiguous_alias` |
    | 5 | 连通分量（整组无类型冲突/成员索引） | `component_conflict` |
    | 6 | 冲突关系（互斥谓词对同一目标） | `relation_conflict` |

    顺序的意义：**先命中的那条就是拒绝理由**。构造一个同时违反 1 和 4 的输入，
    返回的必须是 `type_conflict`——类型冲突是结构性的、无需上下文就能判定；
    歧义别名依赖别的主张集合，是更弱的证据。顺序一换，事后追责时看到的理由就错了。

    参数：
      * `alias_claimants`：别名文本 -> 主张它的候选名集合（第 4 条用）。
      * `group`：a 当前所在连通分量的成员（第 5/6 条用）。空集时只查 a、b 两个。
      * `edges_by_name`：名字 -> 该名字的边（`(relation, other)` 或 dict），第 6 条用。
      * `link`：这次合并的凭据文本，第 4 条优先用它。
    """
    if cand_a is None or cand_b is None:
        raise ValueError("check_merge 需要两个实体候选，收到 None")
    if cand_a.name == cand_b.name:
        return MergeDecision(True, "", "同一个候选，无需合并", "pass")

    for step in (
        lambda: _check_type(cand_a, cand_b),
        lambda: _check_member(cand_a, cand_b),
        lambda: _check_generic(cand_a, cand_b, link),
        lambda: _check_alias(cand_a, cand_b, alias_claimants, link),
        lambda: _check_component(cand_a, cand_b, group),
        lambda: _check_relations(cand_a, cand_b, group, edges_by_name),
    ):
        decision = step()
        if not decision.allowed:
            return decision
    return MergeDecision(True, "", "", "pass")


# ------------------------------------------------------------------ 候选层构建


def _vote_type(votes: Counter) -> str:
    """同名的类型投票：优先具体类型，再按票数，最后按词表顺序（确定性）。"""
    specific = {t: n for t, n in votes.items() if t != CONCEPT}
    pool = specific or dict(votes)
    if not pool:
        return CONCEPT
    order = {t: i for i, t in enumerate(vocab.ENTITY_TYPES)}
    return sorted(pool.items(), key=lambda kv: (-kv[1], order.get(kv[0], 99), kv[0]))[0][0]


def build_candidates(
    mentions: Iterable[Mention], *, drop_noise: bool = False
) -> tuple[list[EntityCandidate], dict]:
    """Mention -> EntityCandidate：类型归一 + （可选）噪声剔除。

    返回 `(candidates, stats)`。`stats` 里记了剔除了哪些名字、按哪条判据剔的——
    只给布尔会让人无法回答「为什么这个实体不见了」。

    `drop_noise` 默认关闭：噪声判据（`norm_entity.is_noise_name`）属 L4 过滤，
    是否在本层执行由调用方决定；开着时判据本身仍是那一份，不在这里重写。
    """
    from . import norm_entity  # 延迟导入：见 _member_index_pair 的说明

    slots: dict[str, dict] = {}
    stats: Counter = Counter()
    for m in mentions:
        name = str(m.name or "").strip()
        if not name:
            stats["dropped:empty_name"] += 1
            continue
        slot = slots.setdefault(
            name,
            {
                "types": Counter(),
                "descs": Counter(),
                "aliases": {},
                "mentions": 0,
                "scenes": set(),
                "chunks": set(),
            },
        )
        slot["types"][normalize_type(m.raw_type)] += 1
        slot["mentions"] += 1
        slot["chunks"].add(m.chunk_id)
        slot["scenes"].add(m.doc_id)
        for alias in m.aliases or ():
            text = str(alias or "").strip()
            if text and text != name:
                slot["aliases"].setdefault(text, set()).add(m.doc_id)
        desc = str(m.desc or "").strip()
        if desc:
            slot["descs"][desc] += 1

    out: list[EntityCandidate] = []
    for name in sorted(slots):
        slot = slots[name]
        etype = _vote_type(slot["types"])
        if drop_noise:
            bad, why = norm_entity.is_noise_name(name, etype)
            if bad:
                stats[f"dropped:{why}"] += 1
                continue
        descs = slot["descs"]
        out.append(
            EntityCandidate(
                name=name,
                etype=etype,
                desc=sorted(descs.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if descs else "",
                aliases=tuple(sorted(slot["aliases"])),
                alias_scenes={a: tuple(sorted(s)) for a, s in sorted(slot["aliases"].items())},
                mentions=slot["mentions"],
                scenes=frozenset(slot["scenes"]),
                chunks=frozenset(slot["chunks"]),
            )
        )
    stats["candidates"] = len(out)
    return out, dict(stats)


# ------------------------------------------------------------------ 合并执行


def _pick_desc(members: Sequence[EntityCandidate]) -> str:
    descs: Counter = Counter()
    for m in members:
        if m.desc:
            descs[m.desc] += 1
    if not descs:
        return ""
    return sorted(descs.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _merged_reason(
    name: str, display: str, evidence: Mapping[str, Mapping[str, str]]
) -> str:
    """`merged_from[].reason`：与展示名有**直接**合并边 -> 该边的凭据；否则 `alias_chain`。

    为什么要有 `alias_chain`：A 的别名是 B、B 的别名是 C，则 C 与 A 之间没有直接
    证据，是走两步推出来的。把它记成 alias（一步）会在事后追责时高估证据强度。

    为什么按「与展示名的直接边」而不是「与组根的边」判：组根是 UnionFind 里字典序
    最小的名字，展示名是 `canonical_pick` 挑出来的、通常是更长的那个，两者经常不是
    同一个（`罗德岛` 是根，展示名却是 `罗德岛（制药）`）。按组根判会把一条实打实的
    全名归并边报成 alias_chain，那是把证据强度**说小**了，同样错误。
    """
    direct = evidence.get(display, {}).get(name, "")
    return direct or "alias_chain"


def _reason_to_source(reason: str) -> str:
    return {"alias": "upstream", "fullname": "fullname", "alias_chain": "alias_chain"}.get(
        reason, "alias_chain"
    )


def merge_candidates(
    candidates: Iterable[EntityCandidate],
    *,
    edges_by_name: Mapping[str, Iterable] | None = None,
    entity_rule_version: str | None = None,
    salt: str = "",
    manual_aliases: Mapping[str, Iterable[str]] | None = None,
    pinned_ids: Mapping[str, str] | None = None,
    now: str = "",
) -> tuple[list[CanonicalEntity], dict]:
    """执行合并，产出 CanonicalEntity 列表与统计。

    ## 两种合并凭据，全部走同一套六道检查

    * **上游别名**（`candidate.aliases` 里指向另一个候选名的）——来源 `upstream`；
    * **全名归并**（`can_merge_fullname`、且不是成员索引）——来源 `fullname`；
    * 多步链推出来的记 `alias_chain`。

    歧义别名（被 >=2 个候选主张）在**任何**路径上都不参与合并，也不写进任何实体的
    aliases——写进去只会让按别名检索随机命中错的人（决策台账 D-24）。主张它的候选
    的全部别名条目打 `review="pending"` 交人工。

    ## 统计的键名

    `stats` 里每个拒绝理由都在，键名是 `rejected:<reason>`，且**即使为 0 也存在**——
    门禁 L4.13 要查的正是这些键，缺键与「没发生过」读起来一样，那是缺陷。
    基线口径的 `alias_merged` / `alias_dropped_unknown` / `alias_ambiguous_rejected` /
    `fullname_merged` / `fullname_rejected_member` 一并保留，便于与旧图对比。

    参数：
      * `edges_by_name`：名字 -> 边，只给第 6 条（冲突关系）用；
      * `salt`：见 `stable_id()`——同名不同义时由调用方传场景前缀；
      * `pinned_ids`：名字 -> 上一次已经固定在产物里的 id。跨世代稳定靠它。
      * `now`：写进 `split_history[].at` 的时间戳。默认空串（未知），**不自动取
        当前时间**：那会让同一批输入的产物每次不同，回滚比对全部失效。
    """
    from . import norm_entity  # 延迟导入 + 复用既有判据（can_merge_fullname/canonical_pick）

    rule = entity_rule_version or vocab.ENTITY_RULE_VERSION

    # ---- 候选索引（同名候选只留一个，重复计数而不是静默取后者）
    stats: Counter = Counter({f"rejected:{r}": 0 for r in REJECT_REASONS})
    stats.update(
        {
            "candidates": 0,
            "entities": 0,
            "duplicate_candidate": 0,
            "alias_merged": 0,
            "alias_dropped_unknown": 0,
            "alias_ambiguous_rejected": 0,
            "fullname_merged": 0,
            "fullname_rejected_member": 0,
            "alias_chain_merged": 0,
            "already_same_group": 0,
            "ambiguous_aliases": 0,
            "alias_pending": 0,
            "alias_generic_pending": 0,
            "alias_conflict_pending": 0,
            "aliases": 0,
        }
    )
    by_name: dict[str, EntityCandidate] = {}
    for cand in candidates:
        if cand.name in by_name:
            stats["duplicate_candidate"] += 1
            continue
        by_name[cand.name] = cand
    stats["candidates"] = len(by_name)

    # ---- 别名主张与歧义集合（唯一判据 _check_ambiguous_alias）
    claimants: dict[str, set[str]] = defaultdict(set)
    for cand in by_name.values():
        for alias in cand.aliases:
            text = str(alias or "").strip()
            if text and text != cand.name:
                claimants[text].add(cand.name)
    ambiguous = {t for t, owners in claimants.items() if len(owners) > 1}
    pending_names = {n for t in ambiguous for n in claimants[t]}
    stats["ambiguous_aliases"] = len(ambiguous)
    stats["alias_pending"] = len(pending_names)

    # ---- 合并边：别名先、全名后；各自排序，保证与输入顺序无关
    alias_edges: list[tuple[str, str]] = []
    for cand in by_name.values():
        for alias in sorted({str(a or "").strip() for a in cand.aliases}):
            if not alias or alias == cand.name:
                continue
            ok, _why = _check_ambiguous_alias(alias, claimants)
            if not ok:
                # 歧义别名：走六道检查里的第 4 条，理由记同一个键，口径只有一个
                stats["rejected:ambiguous_alias"] += 1
                stats["alias_ambiguous_rejected"] += 1
                continue
            if alias not in by_name:
                stats["alias_dropped_unknown"] += 1
                continue
            alias_edges.append((cand.name, alias))
    alias_edges.sort()

    fullname_edges: list[tuple[str, str]] = []
    names = sorted(by_name)
    for i, short in enumerate(names):
        for long in names[i + 1 :]:
            if len(long) <= len(short):
                continue
            if norm_entity.can_merge_fullname(short, long):
                fullname_edges.append((short, long))
            elif _member_index_pair(short, long)[0]:
                # 前缀关系成立、但因为成员索引被拒 —— 这一条要单独记账，
                # 它和「名字不相干所以不合并」是两件事。
                stats["fullname_rejected_member"] += 1
    fullname_edges.sort()

    uf = _UnionFind(names)
    #: 直接合并证据：evidence[owner][target] = "alias"/"fullname"（双向都记一份）。
    #: 只用于给 `merged_from[].reason` 定性——没有边的成员名就是 alias_chain。
    evidence: dict[str, dict[str, str]] = defaultdict(dict)
    for source, edges in (("alias", alias_edges), ("fullname", fullname_edges)):
        for owner, target in edges:
            if uf.find(owner) == uf.find(target):
                stats["already_same_group"] += 1
                continue
            # 整组 = 两侧当前的连通分量，不只是 owner 那一侧。只查 owner 侧会把
            # 「target 带着自己一个冲突成员入伙」这种最典型的误并漏掉：A(角色) 先被
            # 概念实体吸收，再吸收 地点实体时，冲突发生在 A 与 地点 之间，而 owner 侧
            # 根本不含 地点。
            group = [
                by_name[n]
                for n in sorted(set(_group_names(uf, owner)) | set(_group_names(uf, target)))
            ]
            decision = check_merge(
                by_name[owner],
                by_name[target],
                alias_claimants=claimants,
                group=group,
                edges_by_name=edges_by_name,
                link=target if source == "alias" else "",
            )
            if not decision.allowed:
                stats[f"rejected:{decision.reason}"] += 1
                continue
            uf.union(owner, target)
            evidence[owner][target] = source
            evidence[target][owner] = source
            stats["alias_merged" if source == "alias" else "fullname_merged"] += 1

    # ---- 组装 CanonicalEntity
    groups: dict[str, list[str]] = defaultdict(list)
    for name in names:
        groups[uf.find(name)].append(name)
    group_of_name = {name: root for root, members in groups.items() for name in members}

    counts = {name: cand.mentions for name, cand in by_name.items()}
    manual = {str(k): tuple(str(x).strip() for x in v if str(x).strip()) for k, v in (manual_aliases or {}).items()}
    pins = {str(k): str(v) for k, v in (pinned_ids or {}).items()}

    entities: list[CanonicalEntity] = []
    for root in sorted(groups):
        members = sorted(groups[root])
        seed = root  # 组根名 = 建组时的规范名（id 的出生名）
        display = norm_entity.canonical_pick(members, counts)
        member_cands = [by_name[m] for m in members]
        entity_id = pins.get(seed) or stable_id(seed, entity_rule_version=rule, salt=salt)

        merged_from: list[dict] = []
        reasons: dict[str, str] = {}
        for name in members:
            if name == display:
                continue
            reason = _merged_reason(name, display, evidence)
            reasons[name] = reason
            merged_from.append(
                {"name": name, "reason": reason, "scenes": sorted(by_name[name].scenes)}
            )
            if reason == "alias_chain":
                stats["alias_chain_merged"] += 1

        aliases_out: list[dict] = []
        member_set = set(members)
        for name in members:
            if name != display:
                aliases_out.append(
                    {
                        "text": name,
                        "source": _reason_to_source(reasons.get(name, "alias_chain")),
                        "scenes": sorted(by_name[name].scenes),
                        "review": "pending" if name in pending_names else "auto",
                    }
                )
            for text in by_name[name].aliases:
                if not text or text in member_set or text == display:
                    continue
                ok, _why = _check_ambiguous_alias(text, claimants)
                if not ok:
                    continue  # 歧义别名不给任何候选（前置已计入 rejected:ambiguous_alias）
                review = "pending" if name in pending_names else "auto"
                # 两种「合并被拒但别名仍要进检索表」的情形，一律标 pending 交人工：
                #
                # 1) 第 3 条挡下的「修饰语 + 职务后缀」短语（`库兰塔士兵` 之于 `暴雨`）;
                # 2) 别名文本本身是**另一个候选实体**、且这次合并被六道检查拒了
                #    （`临光家` 之于 `临光`，type_conflict）。别名表是检索入口，
                #    让 `临光家` 命中 `临光` 与把它并进 `临光` 是同一个错误，只是
                #    晚一步发作——拒绝合并却把它当权威别名，等于绕过了自己的判据。
                if is_generic_anchor_by_shape(text, display):
                    review = "pending"
                    stats["alias_generic_pending"] += 1
                elif text in by_name and by_name[text].name != display:
                    other = by_name[text]
                    hit = group_of_name.get(other.name, other.name)
                    if hit != root:
                        review = "pending"
                        stats["alias_conflict_pending"] += 1
                scenes: set[str] = set()
                for owner in members:
                    scenes |= set(by_name[owner].alias_scenes.get(text) or ())
                if not scenes:
                    scenes = set(by_name[name].scenes)
                aliases_out.append(
                    {
                        "text": text,
                        "source": "upstream",
                        "scenes": sorted(scenes),
                        "review": review,
                    }
                )
        for key in (display, seed):
            for text in manual.get(key, ()):
                if text in member_set or text == display:
                    continue
                aliases_out.append(
                    {
                        "text": text,
                        "source": "manual",
                        "scenes": [],
                        "review": "auto",
                    }
                )

        # 同一文本重复出现（既是成员名又被声明为别名）只留一条，来源取更强的那个
        dedup: dict[str, dict] = {}
        rank = {s: i for i, s in enumerate(ALIAS_SOURCES)}
        for item in aliases_out:
            old = dedup.get(item["text"])
            if old is None or rank[item["source"]] < rank[old["source"]]:
                dedup[item["text"]] = item
        aliases_final = [dedup[t] for t in sorted(dedup)]

        split_history: list[dict] = []
        if len(members) > 1:
            kinds = {reasons.get(n, "alias") for n in members if n != display}
            if "alias_chain" in kinds:
                overall = "alias_chain"
            elif kinds == {"fullname"}:
                overall = "fullname"
            else:
                overall = "alias"
            split_history.append(
                {"at": now, "action": "merge", "members": list(members), "reason": overall}
            )

        chunks: set[str] = set()
        for cand in member_cands:
            chunks |= set(cand.chunks)
        entities.append(
            CanonicalEntity(
                id=entity_id,
                name=display,
                etype=_vote_type(Counter({c.etype: max(1, c.mentions) for c in member_cands})),
                desc=_pick_desc(member_cands),
                aliases=aliases_final,
                merged_from=merged_from,
                split_history=split_history,
                mentions=sum(c.mentions for c in member_cands),
                chunks=sorted(chunks),
            )
        )

    entities.sort(key=lambda e: (e.name, e.id))
    stats["entities"] = len(entities)
    stats["aliases"] = sum(len(e.aliases) for e in entities)
    return entities, dict(stats)


def _group_names(uf: _UnionFind, name: str) -> list[str]:
    root = uf.find(name)
    return [n for n in uf.parent if uf.find(n) == root]
