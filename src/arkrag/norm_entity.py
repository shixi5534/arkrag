"""L4 实体归一：类型闭合三层、噪声名判据、并查集合并、规范名四级优先。

计划书 3.3 的本地二次过滤落在 is_noise_name；§4.2 的合并与拒绝条件落在 merge_names。
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from . import vocab

FULLNAME_SEPS = "·・‧‥ ‪-–—_（("
RE_MASKED = re.compile(r"^[?？]+$")
RE_CODE_NAME = re.compile(r"^(?:[A-Za-z][A-Za-z0-9]*[-_]){0,3}[A-Z0-9]{5,}$|[0-9A-F]{6,}")
RE_PUNCT = re.compile(r"[。！？；，、：:!?;,\u3000]")
RE_LATIN_DIGIT = re.compile(r"[A-Za-z0-9]")


def norm_type(raw: object) -> str:
    """实体类型归一，契约 §3.4 的四步（顺序不可调换）。

    1. 规范值精确匹配
    2. 精确同义词
    3. 受限模糊（仅 len(text) >= 2，且命中项**长于** text 的一半）
    4. 兜底「概念」

    本函数只是 `identity.normalize_type` 的转发入口，判据只有那一份。

    为什么删掉原来的子串步：旧顺序把最松的判据（子串）放在第 2 位，于是
    「人物设定」因为含「人物」被判成 角色、`感染事件报告` 因为含「事件」被判成
    事件——真实数据里这类描述性短语不少。判据一松，第 1 条（类型兼容）就形同虚设。
    「同一个判定在两处各写一份，整套检查就等于最松的那处」正是本项目反复踩过的坑，
    所以这里不留副本，连兜底字面量也不再重复写一个 "概念"。

    兜底不给「未分类」：下游所有按类型的过滤会因为多出一个词表外的值而静默失效。

    延迟导入的原因：`identity` 是本人模块的下层（它的合并逻辑要复用
    `can_merge_fullname` / `is_member_index` / `canonical_pick`），顶层互相 import
    会成环。函数内导入在 import 期零成本，且不受 import 顺序影响。
    """
    from . import identity  # noqa: PLC0415 —— 见 docstring「延迟导入的原因」

    return identity.normalize_type(raw)


def is_structured_proper_name(name: str) -> bool:
    """超长但结构上确定是「专名」而非「描述性短语」的形态。

    MAX_NAME_CHARS=16 的用意是拦住「把一句描述当名字」；
    两类真专名会自然超长，拦它就是误伤：
      - 含间隔号「·」的多段人名/地名（俄式全名：叶莉萨维塔·康斯坦丁诺夫娜·拉齐萨拉）
      - 以机构后缀结尾的机构全名（伦蒂尼姆及卡兹戴尔经济文化交流协会）
    这两类形态一旦命中，名字本身的构造就证明它是专名，不需要靠长度猜。
    注意它只豁免「长度」这一条判据，不豁免泛称/描述性前后缀/句读等其余判据。
    """
    stripped = (name or "").strip()
    if not stripped:
        return False
    if any(sep in stripped for sep in "·・‧‥"):
        return True
    return stripped.endswith(tuple(vocab.ORG_SUFFIXES))


def is_noise_name(name: str, etype: str | None = None) -> tuple[bool, str]:
    """返回 (是否噪声, 判据名)。判据名会进过滤器统计，不许只留布尔。"""
    if not name or not name.strip():
        return True, "empty"
    if RE_MASKED.match(name):
        return True, "masked_speaker"
    if RE_CODE_NAME.match(name) and ("-" in name or any(ch.isdigit() for ch in name)):
        return True, "serial_code"
    if name in vocab.PRONOUN_SET:
        return True, "pronoun"
    if name in set(vocab.GENERIC_PERSON_NOUNS):
        return True, "generic_noun"
    if len(name) > vocab.MAX_NAME_CHARS and not is_structured_proper_name(name):
        return True, "overlong"
    if any(name.startswith(p) for p in vocab.DESC_PREFIX):
        return True, "descriptive_prefix"
    if any(name.endswith(s) for s in vocab.DESC_SUFFIX):
        return True, "descriptive_suffix"
    if etype == "事件":
        if RE_PUNCT.search(name):
            return True, "event_punctuated"
        if len(name) > 12:
            return True, "event_overlong"
        if any(p in name for p in vocab.PARTICLES):
            return True, "event_particle"
        for verb in vocab.ACTION_VERBS:
            pos = name.find(verb)
            if pos > 0 and pos + len(verb) < len(name):
                return True, "event_clause"
    return False, ""


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

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
        # 字典序小者做根：让合并结果与遍历顺序无关（不变量 7）
        if rb < ra:
            ra, rb = rb, ra
        self.parent[rb] = ra


def is_member_index(short: str, long: str) -> bool:
    """「组织·信徒」是一群无名成员，不是组织本体，禁止并成一个节点。"""
    remainder = long[len(short) :]
    if not remainder:
        return False
    stripped = remainder.lstrip(FULLNAME_SEPS)
    if stripped.endswith("们"):
        return True
    return any(stripped.endswith(suffix) for suffix in vocab.MEMBER_SUFFIXES)


def can_merge_fullname(short: str, long: str) -> bool:
    if not long.startswith(short) or len(long) <= len(short):
        return False
    nxt = long[len(short)]
    if nxt not in FULLNAME_SEPS:
        return False
    return not is_member_index(short, long)


def canonical_pick(names: list[str], counts: dict[str, int]) -> str:
    """四级优先，顺序不可调换：出现次数 -> 含全名分隔符 -> 更长 -> 字典序。"""
    seen = names

    def key(name: str):
        return (
            -counts.get(name, 0),
            0 if any(sep in name for sep in "·・") else 1,
            -len(name),
            name,
        )

    return sorted(seen, key=key)[0]


def collect(records: list[dict]) -> dict:
    """把逐块原始抽取结果汇成「按名字的观察」，不做任何删改。"""
    obs: dict[str, dict] = {}
    for rec in records:
        chunk_id = rec["chunk_id"]
        for ent in rec["raw"].get("entities") or []:
            name = (ent.get("name") or "").strip()
            if not name:
                continue
            slot = obs.setdefault(
                name,
                {
                    "name": name,
                    "type_votes": Counter(),
                    "aliases": set(),
                    "descs": Counter(),
                    "chunks": set(),
                    "mention_as_entity": 0,
                },
            )
            slot["type_votes"][norm_type(ent.get("type"))] += 1
            slot["mention_as_entity"] += 1
            slot["chunks"].add(chunk_id)
            for alias in ent.get("aliases") or []:
                alias = str(alias).strip()
                if alias and alias != name:
                    slot["aliases"].add(alias)
            desc = (ent.get("desc") or "").strip()
            if desc:
                slot["descs"][desc] += 1
    return obs


def ambiguous_aliases(obs: dict) -> dict[str, set[str]]:
    """被两个以上「已留名」同时主张的别名 -> 主张者集合。

    「殿下」「将军」「老爷」「父亲」这类称谓在同一批块里注定被不同的人各自认领，
    把它当别名边会通过并查集把互不相干的人串成一个大组（实测凯尔希一组 32 人，
    里面同时有 Logos、Mon3tr、特蕾西娅、特雷西斯、杜比、维娜……）。
    一个人的别名只能由他自己主张；多人主张 = 这个字符串没资格当别名，
    既不能用来合并，也不该写进任何实体的 aliases（写进去会让检索随机命中错的人）。
    """
    owners: dict[str, set[str]] = defaultdict(set)
    for name, slot in obs.items():
        for alias in slot["aliases"]:
            if alias != name:
                owners[alias].add(name)
    return {alias: who for alias, who in owners.items() if len(who) > 1}


def merge_names(obs: dict, alias_edges: list[tuple[str, str]]) -> tuple[dict[str, str], dict]:
    """别名边（两端都存在才合并）+ 全名归并（带成员拒绝）。凭空造实体是禁止的。"""
    names = list(obs)
    uf = UnionFind()
    for name in names:
        uf.add(name)

    stats = {
        "alias_merged": 0,
        "alias_dropped_unknown": 0,
        "alias_ambiguous_rejected": 0,
        "fullname_merged": 0,
        "fullname_rejected_member": 0,
    }
    ambiguous = ambiguous_aliases(obs)
    for src, dst in alias_edges:
        if dst in ambiguous:
            stats["alias_ambiguous_rejected"] += 1
            continue
        if src in obs and dst in obs:
            if uf.find(src) != uf.find(dst):
                uf.union(src, dst)
                stats["alias_merged"] += 1
        else:
            stats["alias_dropped_unknown"] += 1

    by_len = sorted(names, key=len)
    for i, short in enumerate(by_len):
        for long in by_len[i + 1 :]:
            if len(long) <= len(short):
                continue
            if not long.startswith(short):
                continue
            if long[len(short) : len(short) + 1] not in FULLNAME_SEPS:
                continue
            if is_member_index(short, long):
                stats["fullname_rejected_member"] += 1
                continue
            if uf.find(short) != uf.find(long):
                uf.union(short, long)
                stats["fullname_merged"] += 1
    groups: dict[str, list[str]] = defaultdict(list)
    for name in names:
        groups[uf.find(name)].append(name)
    return groups, stats
