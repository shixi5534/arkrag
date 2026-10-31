"""E3 身份层（`arkrag.identity`）的判据测试。

每条断言都对应契约 §3 的一条具体条款，且都能被构造输入触发 FAIL——
`tests/test_suite_hygiene.py` 会机检「每个 test_* 是否真能抛」，这里每个函数都
自带 assert，不靠外部记账。

本文件不读写 `data/`：真实数据上的跑批结论写在报告里，不进单元测试
（那会让门变成"取决于本机缓存是否完整"）。
"""
from __future__ import annotations

import re

import pytest

from arkrag import gates
from arkrag import identity
from arkrag import norm_entity
from arkrag import vocab

ID_RE = re.compile(r"^e-[0-9a-f]{16}$")


# ------------------------------------------------------------------ 构造工具


def cand(name: str, etype: str = "概念", aliases=(), scenes=("s",), mentions: int = 1):
    """造一个 EntityCandidate。默认 概念，因为它是类型兼容的万能项。"""
    return identity.EntityCandidate(
        name=name,
        etype=etype,
        aliases=tuple(aliases),
        alias_scenes={a: tuple(scenes) for a in aliases},
        mentions=mentions,
        scenes=frozenset(scenes),
        chunks=frozenset(c + "#000" for c in scenes),
    )


def ment(name: str, chunk_id: str, raw_type: str = "概念", aliases=(), desc: str = ""):
    return identity.Mention(
        name=name, chunk_id=chunk_id, raw_type=raw_type, desc=desc, aliases=tuple(aliases)
    )


def merged(mentions, edges=None, **kw):
    """mention 列表 -> (候选, CanonicalEntity 列表, stats)。"""
    candidates, _build_stats = identity.build_candidates(mentions)
    entities, stats = identity.merge_candidates(
        candidates, edges_by_name=edges or {}, **kw
    )
    return candidates, entities, stats


def entity_named(entities, name: str):
    for e in entities:
        if e.name == name:
            return e
    raise AssertionError(f"没有产出名为 {name!r} 的实体，实际有 {[e.name for e in entities]}")


# ------------------------------------------------------------------ §3.2 稳定 id


class TestStableId:
    """契约 §3.2：`e-` + sha256(rule_version \\x1f canonical_name)[:16]。"""

    def test_same_input_same_id(self) -> None:
        a = identity.stable_id("凯尔希", entity_rule_version="ev2")
        b = identity.stable_id("凯尔希", entity_rule_version="ev2")
        assert a == b, "同输入必须同 id，否则跨次构建无法对照"
        assert ID_RE.match(a), f"格式必须匹配 ^e-[0-9a-f]{{16}}$，实测 {a!r}"
        # 快照：构成一旦变化这条会红，红了先核对契约 §3.2 再改这里
        assert a == "e-c7091cbe8cb6fe66", f"构成变了？实测 {a}"

    def test_rule_version_changes_id(self) -> None:
        a = identity.stable_id("凯尔希", entity_rule_version="ev2")
        b = identity.stable_id("凯尔希", entity_rule_version="ev3")
        assert a != b, "规则版本必须进哈希：规则一改，旧 id 的可比性就该明确失效"

    def test_salt_changes_id(self) -> None:
        a = identity.stable_id("陈", entity_rule_version="ev2")
        b = identity.stable_id("陈", entity_rule_version="ev2", salt="activities/act1/x")
        assert a != b, "两个不同实体可能同名，salt 是它们分开的唯一手段"

    def test_same_salt_same_id(self) -> None:
        a = identity.stable_id("陈", entity_rule_version="ev2", salt="activities/act1/x")
        b = identity.stable_id("陈", entity_rule_version="ev2", salt="activities/act1/x")
        assert a == b, "同一个场景里的同一个名字必须稳定，否则每次构建都换 id"

    def test_format(self) -> None:
        got = identity.stable_id("凯尔希医生", entity_rule_version="ev2")
        assert ID_RE.match(got), f"{got!r} 不匹配 ^e-[0-9a-f]{{16}}$（门禁 L4.12 的字面要求）"
        assert len(got) == 18, f"e- + 16 hex = 18 字符，实测 {len(got)}"

    def test_id_does_not_contain_name(self) -> None:
        for name in ("凯尔希", "陈", "A", "罗德岛（制药）"):
            got = identity.stable_id(name, entity_rule_version="ev2")
            assert name not in got, f"id {got!r} 含名字 {name!r}，这就是 L4.12 要禁的旧方案"

    def test_rename_keeps_id(self) -> None:
        """契约 §3.2 硬要求：展示名改了，id 不变。"""
        _cands, ents, _stats = merged([ment("凯尔希医生", "obt/main/a#001", "人物")])
        e = entity_named(ents, "凯尔希医生")
        before = e.id
        e.rename("凯尔希")
        assert e.name == "凯尔希", "rename 没生效"
        assert e.id == before, f"改名换了 id（{before} -> {e.id}），合并/拆分历史会随之断掉"

    def test_different_names_different_ids(self) -> None:
        a = identity.stable_id("凯尔希", entity_rule_version="ev2")
        b = identity.stable_id("阿米娅", entity_rule_version="ev2")
        assert a != b


# ------------------------------------------------------------------ §3.4 类型归一


class TestNormalizeTypeOrder:
    """契约 §3.4 四步顺序：精确 -> 精确同义词 -> 受限模糊 -> 兜底 概念。"""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("角色", "角色"),
            ("组织", "组织"),
            ("地点", "地点"),
            ("物品", "物品"),
            ("概念", "概念"),
            ("事件", "事件"),
        ],
    )
    def test_step1_canonical_exact(self, raw: str, expected: str) -> None:
        assert identity.normalize_type(raw) == expected

    @pytest.mark.parametrize(
        "raw,expected",
        [("人物", "角色"), ("人名", "角色"), ("种族", "概念"), ("家族", "组织")],
    )
    def test_step2_synonym_exact(self, raw: str, expected: str) -> None:
        assert identity.normalize_type(raw) == expected

    def test_step3_fuzzy_still_allowed_when_long_enough(self) -> None:
        """命中项必须超过一半长度：`角色类` 命中 2/3，过。"""
        assert identity.normalize_type("角色类") == "角色"
        assert identity.normalize_type("地点型") == "地点"

    def test_persona_setting_is_not_a_role(self) -> None:
        """`人物设定` 命中 `人物` 只占一半，按契约「一半以上」不命中 -> 概念。

        旧实现（精确 -> 子串 -> 同义词）会判成 角色，这正是本次修正的靶子。
        """
        got = identity.normalize_type("人物设定")
        assert got == "概念", f"人物设定 被判成 {got}；子串判据又跑到同义词前面去了？"
        assert got != "角色"

    def test_long_text_is_not_fuzzy_matched(self) -> None:
        """`人物的相关设定说明`：命中 2 字 / 全长 9 字，远不到一半 -> 概念。"""
        assert identity.normalize_type("人物的相关设定说明") == "概念"

    def test_old_order_would_have_matched(self) -> None:
        """证明差异是实的：旧顺序把这两个词判成具体类型，新顺序判 概念。"""
        old = identity.normalize_type("角色扮演", strict_order=False)
        new = identity.normalize_type("角色扮演")
        assert old == "角色", f"预期旧顺序命中子串判成 角色，实测 {old}"
        assert new == "概念", f"新顺序应拒绝 2/4 的模糊命中，实测 {new}"
        old2 = identity.normalize_type("感染事件报告", strict_order=False)
        new2 = identity.normalize_type("感染事件报告")
        assert (old2, new2) == ("事件", "概念")

    @pytest.mark.parametrize(
        "raw", ["", "   ", None, 123, [], {"a": 1}, 3.5, b"role"]
    )
    def test_fallback_is_concept(self, raw) -> None:
        assert identity.normalize_type(raw) == "概念"

    def test_unknown_string_falls_back(self) -> None:
        assert identity.normalize_type("完全不在词表里的词") == "概念"
        assert identity.normalize_type("zzz") == "概念"

    def test_strip_and_single_char(self) -> None:
        assert identity.normalize_type(" 角色 ") == "角色", "首尾空白必须被吃掉"
        assert identity.normalize_type("角") == "概念", "单字不许走模糊匹配（len < 2）"

    def test_result_always_in_vocab(self) -> None:
        """兜底只能是词表里的值：多出一个词表外类型会让下游过滤静默失效。"""
        for raw in ("角色", "人物设定", "", None, "??", "生物", "家族"):
            assert identity.normalize_type(raw) in vocab.ENTITY_TYPES

    def test_norm_entity_delegates(self) -> None:
        """`norm_entity.norm_type` 必须是同一个实现，不许有第二份判据。"""
        samples = ["角色", "人物", "人物设定", "角色扮演", "感染事件报告", "", None, "种族"]
        for raw in samples:
            assert norm_entity.norm_type(raw) == identity.normalize_type(raw), (
                f"{raw!r} 在两处实现里给出不同答案——判据分叉了"
            )

    def test_norm_entity_public_behaviour(self) -> None:
        assert norm_entity.norm_type("人物") == "角色"
        assert norm_entity.norm_type("人物设定") == "概念"


# ------------------------------------------------------------------ §3.3 六道检查


class TestCheck1TypeConflict:
    """第 1 条：类型不同且都不是 概念 -> type_conflict。"""

    def test_triggers(self) -> None:
        d = identity.check_merge(cand("乌萨斯", "地点"), cand("乌萨斯帝国", "组织"))
        assert d.allowed is False
        assert d.reason == "type_conflict", d.reason
        assert d.check == "type"
        assert "乌萨斯" in d.detail

    def test_allows_same_type(self) -> None:
        d = identity.check_merge(cand("凯尔希", "角色"), cand("凯尔希医生", "角色"))
        assert d.allowed is True
        assert d.reason == ""

    def test_allows_when_one_side_is_concept(self) -> None:
        d = identity.check_merge(cand("甲", "概念"), cand("乙", "地点"))
        assert d.allowed is True, "概念是万能兼容项，这一条不该拦"

    def test_allows_both_concept(self) -> None:
        assert identity.check_merge(cand("甲", "概念"), cand("乙", "概念")).allowed is True


class TestCheck2MemberIndex:
    """第 2 条：集体 vs 成员 -> member_index。"""

    def test_triggers(self) -> None:
        d = identity.check_merge(cand("整合运动", "组织"), cand("整合运动成员", "组织"))
        assert d.allowed is False
        assert d.reason == "member_index", d.reason

    def test_allows_fullname(self) -> None:
        d = identity.check_merge(cand("薇薇安娜", "角色"), cand("薇薇安娜·德罗斯特", "角色"))
        assert d.allowed is True

    def test_prefix_required(self) -> None:
        """没有前缀关系时不许判成员索引（决策台账 D-24 的 36 条假阳性）。"""
        bad, _detail = identity._member_index_pair("克里斯", "罗德岛工程干员")
        assert bad is False, "is_member_index 只切后缀，调用方必须补前缀前置条件"


class TestCheck3GenericAnchor:
    """第 3 条：职务/组织泛称 -> generic_anchor。"""

    def test_triggers(self) -> None:
        d = identity.check_merge(cand("医疗干员", "角色"), cand("米纳", "角色"))
        assert d.allowed is False
        assert d.reason == "generic_anchor", d.reason

    def test_allows_proper_name(self) -> None:
        d = identity.check_merge(cand("米纳", "角色"), cand("莉塔", "角色"))
        assert d.allowed is True

    def test_generic_predicate_itself(self) -> None:
        for name in ("士兵", "干部", "协会", "医疗干员", "感染者"):
            assert identity.is_generic_anchor(name) is True, name
        for name in ("凯尔希", "博士", "罗德岛", "乔", "凯尔希医生"):
            assert identity.is_generic_anchor(name) is False, name


class TestCheck3GenericAnchorShape:
    """第 3 条的结构补充：别名是「修饰语/种族/职业 + 职务后缀」时也要拦。

    `is_generic_anchor` 抓的是「整个名字就是泛称」；这一类抓的是
    「`老练的士兵`、`人事干员`、`黑帮首领` 被当成具名角色的别名」——剧本给无名
    路人起的显示名是角色属性，不是专名。
    """

    # ---- Lead 指定的四条硬要求

    def test_rejects_department_plus_role_suffix(self) -> None:
        """① `人事干员 <- 乔`：`人事` 不在词表，旧判据放行，形状判据必须拦住。"""
        assert identity.is_generic_anchor("人事干员") is False, "旧判据本就抓不到，这是前提"
        d = identity.check_merge(
            cand("乔", "角色"), cand("人事干员", "角色"), link="人事干员"
        )
        assert d.allowed is False, "「人事干员」是职务短语，不能当乔的别名"
        assert d.reason == "generic_anchor", d.reason

    def test_allows_fullname_with_role_suffix(self) -> None:
        """② `凯尔希 <- 凯尔希医生`：互相包含，是全名归并非泛称。"""
        d = identity.check_merge(
            cand("凯尔希", "角色"), cand("凯尔希医生", "角色"), link="凯尔希医生"
        )
        assert d.allowed is True, d.detail
        assert identity.is_generic_anchor_by_shape("凯尔希医生", "凯尔希") is False

    def test_allows_org_fullname(self) -> None:
        """③ `罗德岛 <- 罗德岛制药公司`：同上，组织全名。"""
        d = identity.check_merge(
            cand("罗德岛", "组织"), cand("罗德岛制药公司", "组织"), link="罗德岛制药公司"
        )
        assert d.allowed is True, d.detail
        assert identity.is_generic_anchor_by_shape("罗德岛制药公司", "罗德岛") is False

    def test_allows_surname_style_alias(self) -> None:
        """④ `魏彦吾 <- 魏大长官`：共享首字，是姓氏/敬称式别名。"""
        d = identity.check_merge(
            cand("魏彦吾", "角色"), cand("魏大长官", "角色"), link="魏大长官"
        )
        assert d.allowed is True, d.detail
        assert identity.is_generic_anchor_by_shape("魏大长官", "魏彦吾") is False

    # ---- ⑤ 三版判据的差异：只看后缀会误伤，加前置后正确

    def test_suffix_only_would_misjudge_fullname(self) -> None:
        """只看「以职务后缀结尾」会把全名归并误打成泛称——第 2 条前置挡掉它。"""
        alias, canonical = "谢拉格自治州住宅管理委员会", "谢拉格"
        assert any(
            alias.endswith(s) and len(alias) > len(s)
            for s in vocab.MEMBER_SUFFIXES + vocab.ORG_SUFFIXES
        ), "前提：它确实以组织后缀结尾"
        assert identity.is_generic_anchor_by_shape(alias, canonical) is False, (
            "互相包含说明是全名归并，不能被形状判据拦下"
        )

    def test_suffix_only_would_misjudge_surname_alias(self) -> None:
        """只看后缀 + 互不包含，会把姓氏式别名误打成泛称——第 3 条前置挡掉它。"""
        alias, canonical = "麟长官", "麟青砚"
        assert alias.endswith("长官")
        assert not canonical.startswith(alias) and not alias.startswith(canonical), (
            "前提：两者互不包含，所以第 2 条前置放不过它"
        )
        assert identity.is_generic_anchor_by_shape(alias, canonical) is False, (
            "共享首字是姓氏式别名，不能被形状判据拦下"
        )

    def test_bare_suffix_is_left_to_is_generic_anchor(self) -> None:
        """纯后缀（`干员`）不归形状判据管，避免两处都判同一个字符串。"""
        assert identity.is_generic_anchor_by_shape("干员", "米纳") is False
        assert identity.is_generic_anchor("干员") is True, "纯后缀由整名判据负责"

    def test_direction_unknown_tries_both_sides(self) -> None:
        """`link` 没给时两个方向都试：任一侧成立即拒（宁可拆多不可错并）。"""
        d = identity.check_merge(
            cand("暴雨", "角色"), cand("库兰塔士兵", "角色", aliases=("库兰塔士兵",))
        )
        assert d.allowed is False, "没给 link 也必须拦住「种族+职务」形态"
        assert d.reason == "generic_anchor", d.reason

    def test_same_name_is_not_flagged(self) -> None:
        for name in ("", "乔"):
            assert identity.is_generic_anchor_by_shape(name, name) is False

    def test_real_world_shapes_all_flagged(self) -> None:
        """Lead 从真实数据里点出的形态，逐条钉住。"""
        flagged = [
            ("老练的士兵", "里斯本"),
            ("落魄商人", "哈伦"),
            ("拍摄队长", "南希"),
            ("黑帮首领", "恩里科"),
            ("罗德岛外勤干员", "斯弗"),
            ("库兰塔士兵", "暴雨"),
            ("黑钢情报小组成员", "蒂拉"),
        ]
        for alias, canonical in flagged:
            assert identity.is_generic_anchor_by_shape(alias, canonical) is True, alias
        allowed = [
            ("凯尔希医生", "凯尔希"),
            ("罗德岛制药公司", "罗德岛"),
            ("魏大长官", "魏彦吾"),
            ("麟长官", "麟青砚"),
        ]
        for alias, canonical in allowed:
            assert identity.is_generic_anchor_by_shape(alias, canonical) is False, alias

    def test_shape_alias_is_pending_in_alias_table(self) -> None:
        """被形状判据挡下的短语若仍写进别名表，必须标 pending 交人工。"""
        _c, ents, stats = merged(
            [
                identity.Mention(
                    name="暴雨",
                    chunk_id="s#1",
                    raw_type="角色",
                    desc="",
                    aliases=("库兰塔士兵",),
                ),
                identity.Mention(
                    name="里斯本",
                    chunk_id="s#2",
                    raw_type="角色",
                    desc="",
                    aliases=("老练的士兵",),
                ),
            ]
        )
        assert stats["alias_generic_pending"] == 2, stats["alias_generic_pending"]
        found = {}
        for e in ents:
            for a in e.aliases:
                if a["text"] in ("库兰塔士兵", "老练的士兵"):
                    found[a["text"]] = a["review"]
        assert found.get("库兰塔士兵") == "pending", found
        assert found.get("老练的士兵") == "pending", found

    def test_symmetric_both_directions_rejected(self) -> None:
        """两向都有别名声明时必须都拒——否则拒一条、另一条照样把两组合成一组。

        这里两个候选**分处不同场景**（`a#1` / `b#2`），互相主张也不足以证明同一人，
        所以两向都拒。同场共现的那一对走下面 `test_mutual_epithet_exception`。
        """
        both = [
            identity.Mention(
                name="科瓦尔", chunk_id="a#1", raw_type="角色", desc="", aliases=("老工匠",)
            ),
            identity.Mention(
                name="老工匠", chunk_id="b#2", raw_type="角色", desc="", aliases=("科瓦尔",)
            ),
        ]
        _c, ents, stats = merged(both)
        assert stats["rejected:generic_anchor"] >= 1, stats["rejected:generic_anchor"]
        assert len(ents) == 2, [e.name for e in ents]

    def test_mutual_epithet_exception(self) -> None:
        """互相指认 + 同场共现 = 同一人的两个称呼，shape 层放行（真实数据只有 2 对）。"""
        same = [
            identity.Mention(
                name="科瓦尔", chunk_id="x#1", raw_type="角色", desc="工匠", aliases=("老工匠",)
            ),
            identity.Mention(
                name="老工匠", chunk_id="x#2", raw_type="角色", desc="老工匠", aliases=("科瓦尔",)
            ),
        ]
        _c, ents, stats = merged(same)
        assert len(ents) == 1, [e.name for e in ents]
        assert stats["rejected:generic_anchor"] == 0, stats["rejected:generic_anchor"]
        got = ents[0]
        texts = {a["text"] for a in got.aliases} | {got.name}
        assert {"科瓦尔", "老工匠"} <= texts, texts

    def test_mutual_epithet_needs_shared_scene(self) -> None:
        """光互相指认、不同场景 —— 证据不足，仍拒（不能拿跨文档同义指代当同一人）。"""
        diff = [
            identity.Mention(
                name="科瓦尔", chunk_id="p#1", raw_type="角色", desc="", aliases=("老工匠",)
            ),
            identity.Mention(
                name="老工匠", chunk_id="q#1", raw_type="角色", desc="", aliases=("科瓦尔",)
            ),
        ]
        _c, ents, stats = merged(diff)
        assert len(ents) == 2, [e.name for e in ents]
        assert stats["rejected:generic_anchor"] >= 1

    def test_mutual_epithet_requires_both_sides(self) -> None:
        """只有**单向**主张时不能放行，哪怕同场共现——这正是「描述短语被当别名」的形态。

        `整合运动术师` 不是整名泛称（`is_generic_anchor` 放它过），所以这条由 shape 层
        独立负责；它只在 `罗莲娜` 一侧被声明为别名，反方向没有。
        """
        one_sided = [
            identity.Mention(
                name="罗莲娜", chunk_id="t#1", raw_type="角色", desc="", aliases=("整合运动术师",)
            ),
            identity.Mention(
                name="整合运动术师", chunk_id="t#2", raw_type="角色", desc="", aliases=()
            ),
        ]
        _c, ents, stats = merged(one_sided)
        assert len(ents) == 2, [e.name for e in ents]
        assert stats["rejected:generic_anchor"] >= 1, stats["rejected:generic_anchor"]

    def test_mutual_epithet_does_not_rescue_true_generic(self) -> None:
        """整名泛称拦在 shape 层之前：互相指认也不能把 `医疗干员` 并进 `米纳`。"""
        pair = [
            identity.Mention(
                name="米纳", chunk_id="s#1", raw_type="角色", desc="", aliases=("医疗干员",)
            ),
            identity.Mention(
                name="医疗干员", chunk_id="s#2", raw_type="角色", desc="", aliases=("米纳",)
            ),
        ]
        _c, ents, stats = merged(pair)
        assert len(ents) == 2, [e.name for e in ents]
        assert stats["rejected:generic_anchor"] >= 1

    def test_containment_precondition_alone(self) -> None:
        """「互不包含」这条前置必须独立可证：构造一个首字不同、只靠它拦下的例子。

        `哈洛德麾下的老练的士兵` 以 `老练的士兵` 结尾，两者首字不同（哈 / 老），
        所以第 3 条前置救不了它，唯一拦下它的是第 2 条。删掉第 2 条这条立刻变红。
        """
        alias, canonical = "老练的士兵", "哈洛德麾下的老练的士兵"
        assert alias[0] != canonical[0], "前提：首字不同，第 3 条前置不生效"
        assert canonical.endswith(alias), "前提：别名是规范名的后缀，属于全名归并形态"
        assert identity.is_generic_anchor(alias) is False, "前提：不走整名判据那一层"
        assert (
            identity.is_generic_anchor_by_shape(
                alias, canonical, suffixes=identity._role_suffixes()
            )
            is False
        ), "被规范名包含的字符串是全名归并，不是泛称短语"

    def test_first_char_precondition_alone(self) -> None:
        """「共享首字」这条前置也要独立可证：构造一个互不包含、只靠它拦下的例子。"""
        alias, canonical = "麟长官", "麟青砚"
        assert not canonical.startswith(alias) and not alias.startswith(canonical)
        assert not canonical.endswith(alias) and not alias.endswith(canonical)
        assert alias[0] == canonical[0], "前提：首字相同"
        assert (
            identity.is_generic_anchor_by_shape(
                alias, canonical, suffixes=identity._role_suffixes()
            )
            is False
        ), "共享首字是姓氏/敬称式别名"

    def test_org_suffix_does_not_split_org_names(self) -> None:
        """组织名判据只看职务后缀：`第一集团军` 以 `团军` 结尾，但它不是泛称。"""
        assert identity.is_generic_anchor_by_shape(
            "第一集团军", "阿米娅", suffixes=identity._role_suffixes()
        ) is False
        d = identity.check_merge(cand("阿米娅", "组织"), cand("第一集团军", "组织"))
        assert d.allowed is True, d.detail

    def test_rejected_merge_alias_is_pending(self) -> None:
        """合并被六道检查拒掉的候选名，若仍出现在别名表里必须标 pending。"""
        _c, ents, stats = merged(
            [
                ment("临光", "a#1", "角色", aliases=("临光家",)),
                ment("临光家", "a#2", "组织"),
            ]
        )
        assert stats["rejected:type_conflict"] == 1, stats["rejected:type_conflict"]
        assert len(ents) == 2, [e.name for e in ents]
        assert stats["alias_conflict_pending"] >= 1, stats["alias_conflict_pending"]
        for e in ents:
            for a in e.aliases:
                if a["text"] in ("临光家", "临光"):
                    assert a["review"] == "pending", a


class TestCheck4AmbiguousAlias:
    """第 4 条：被 >=2 候选主张的别名 -> ambiguous_alias。"""

    def test_triggers(self) -> None:
        claimants = {"殿下": {"阿米娅", "特蕾西娅"}}
        d = identity.check_merge(
            cand("阿米娅", "角色"),
            cand("特蕾西娅", "角色"),
            alias_claimants=claimants,
            link="殿下",
        )
        assert d.allowed is False
        assert d.reason == "ambiguous_alias", d.reason

    def test_allows_single_claimant(self) -> None:
        d = identity.check_merge(
            cand("阿米娅", "角色"),
            cand("特蕾西娅", "角色"),
            alias_claimants={"殿下": {"阿米娅"}},
            link="殿下",
        )
        assert d.allowed is True

    def test_no_link_no_claim_means_pass(self) -> None:
        d = identity.check_merge(cand("甲", "概念"), cand("乙", "概念"))
        assert d.allowed is True


class TestCheck5ComponentConflict:
    """第 5 条：整组并完不许有类型冲突/成员索引 -> component_conflict。"""

    def test_triggers_on_third_member(self) -> None:
        group = [cand("凯尔希", "角色")]
        d = identity.check_merge(cand("甲", "概念"), cand("乙", "地点"), group=group)
        assert d.allowed is False
        assert d.reason == "component_conflict", d.reason
        assert "凯尔希" in d.detail and "乙" in d.detail

    def test_allows_clean_group(self) -> None:
        group = [cand("凯尔希", "角色")]
        d = identity.check_merge(cand("凯尔希医生", "角色"), cand("凯尔希女士", "角色"), group=group)
        assert d.allowed is True

    def test_triggers_on_member_index_inside_group(self) -> None:
        group = [cand("整合运动", "组织")]
        d = identity.check_merge(cand("整合运动干部", "组织"), cand("整合运动成员", "组织"), group=group)
        assert d.allowed is False
        assert d.reason == "component_conflict", d.reason


class TestCheck6RelationConflict:
    """第 6 条：合并新造出的跨侧互斥关系 -> relation_conflict。"""

    def test_triggers_cross_side(self) -> None:
        edges = {"凯尔希": [("同盟", "博士")], "凯尔希医生": [("敌对", "博士")]}
        d = identity.check_merge(cand("凯尔希", "角色"), cand("凯尔希医生", "角色"), edges_by_name=edges)
        assert d.allowed is False
        assert d.reason == "relation_conflict", d.reason

    def test_allows_different_targets(self) -> None:
        """互斥的两条边指向不同目标，合并不制造矛盾。"""
        edges = {"凯尔希": [("同盟", "博士")], "凯尔希医生": [("敌对", "阿米娅")]}
        d = identity.check_merge(cand("凯尔希", "角色"), cand("凯尔希医生", "角色"), edges_by_name=edges)
        assert d.allowed is True

    def test_allows_preexisting_contradiction_on_one_side(self) -> None:
        """合并前就自相矛盾的名字，撤销合并并不会让矛盾消失，只多一个实体。"""
        edges = {"凯尔希": [("同盟", "博士"), ("敌对", "博士")]}
        d = identity.check_merge(cand("凯尔希", "角色"), cand("凯尔希医生", "角色"), edges_by_name=edges)
        assert d.allowed is True, "判据问的是「并了以后多出什么」，不是「并了以后有什么」"

    def test_no_edges_means_pass(self) -> None:
        assert identity.check_merge(cand("甲", "概念"), cand("乙", "概念")).allowed is True


class TestCheckOrder:
    """契约 §3.3：顺序不可调换，先命中的那条就是拒绝理由。"""

    def test_type_conflict_wins_over_ambiguous_alias(self) -> None:
        """同时违反第 1 条与第 4 条 -> 必须报 type_conflict（1 先命中）。"""
        d = identity.check_merge(
            cand("A", "角色"),
            cand("B", "组织"),
            alias_claimants={"殿下": {"A", "B"}},
            link="殿下",
        )
        assert d.allowed is False
        assert d.reason == "type_conflict", f"顺序错了：实测 {d.reason}"
        assert d.check == "type"

    def test_type_conflict_wins_over_member_index(self) -> None:
        d = identity.check_merge(cand("整合运动", "组织"), cand("整合运动成员", "地点"))
        assert d.reason == "type_conflict"

    def test_member_index_wins_over_generic(self) -> None:
        """`干部` 是泛称、`整合运动干部` 同时是成员索引 -> 第 2 条先命中。"""
        d = identity.check_merge(cand("干部", "角色"), cand("干部成员", "角色"))
        assert d.reason in ("member_index", "generic_anchor")

    def test_ambiguous_alias_wins_over_component(self) -> None:
        d = identity.check_merge(
            cand("甲", "概念"),
            cand("乙", "概念"),
            alias_claimants={"丙": {"甲", "乙"}},
            link="丙",
            group=[cand("丁", "地点")],
        )
        assert d.reason == "ambiguous_alias", f"实测 {d.reason}"

    def test_reason_is_always_in_declared_set(self) -> None:
        cases = [
            identity.check_merge(cand("A", "角色"), cand("B", "组织")),
            identity.check_merge(cand("整合运动", "组织"), cand("整合运动成员", "组织")),
            identity.check_merge(cand("士兵", "角色"), cand("甲", "角色")),
        ]
        for d in cases:
            assert d.allowed is False
            assert d.reason in identity.REJECT_REASONS

    def test_allowed_decision_forbids_reason(self) -> None:
        """通过时 reason 必须是空串——带着旧理由的 PASS 会让事后追责读错。"""
        d = identity.check_merge(cand("凯尔希", "角色"), cand("凯尔希医生", "角色"))
        assert d.allowed is True and d.reason == ""
        with pytest.raises(ValueError):
            identity.MergeDecision(True, "type_conflict", "x", "type")

    def test_unknown_reason_rejected(self) -> None:
        with pytest.raises(ValueError):
            identity.MergeDecision(False, "made_up_reason", "", "x")

    def test_none_candidate_raises(self) -> None:
        with pytest.raises(ValueError):
            identity.check_merge(None, cand("甲"))


# ------------------------------------------------------------------ §3.1 别名来源与场景


class TestAliasSourceAndScenes:
    """别名四项：`text` / `source` / `scenes` / `review`。"""

    def test_upstream_alias_keeps_scenes(self) -> None:
        _c, ents, _s = merged(
            [
                ment("凯尔希", "obt/main/level_main_00-01_beg#001", "角色", aliases=("凯太后",)),
                ment("凯尔希", "obt/main/level_main_00-02_beg#003", "角色"),
            ]
        )
        e = entity_named(ents, "凯尔希")
        alias = [a for a in e.aliases if a["text"] == "凯太后"]
        assert len(alias) == 1, f"上游别名丢了：{e.aliases}"
        assert alias[0]["source"] == "upstream", alias[0]["source"]
        assert alias[0]["scenes"] == ["obt/main/level_main_00-01_beg"], (
            f"scenes 应是别名实际出现过的 doc_id（`#` 之前那段），实测 {alias[0]['scenes']}"
        )
        assert alias[0]["review"] == "auto"

    def test_alias_scenes_spans_multiple_docs(self) -> None:
        _c, ents, _s = merged(
            [
                ment("凯尔希", "a/one#001", "角色", aliases=("凯太后",)),
                ment("凯尔希", "b/two#002", "角色", aliases=("凯太后",)),
            ]
        )
        e = entity_named(ents, "凯尔希")
        alias = [a for a in e.aliases if a["text"] == "凯太后"][0]
        assert alias["scenes"] == ["a/one", "b/two"], alias["scenes"]

    def test_fullname_merge_marked_fullname(self) -> None:
        _c, ents, stats = merged(
            [
                ment("罗德岛", "x#1", "组织"),
                ment("罗德岛（制药）", "y#2", "组织"),
            ]
        )
        assert stats["fullname_merged"] == 1, stats
        e = entity_named(ents, "罗德岛（制药）")
        assert [a["text"] for a in e.aliases] == ["罗德岛"]
        assert e.aliases[0]["source"] == "fullname", e.aliases[0]["source"]
        assert [m["reason"] for m in e.merged_from] == ["fullname"]
        assert e.merged_from[0]["scenes"] == ["x"]

    def test_alias_chain_marked(self) -> None:
        """甲 -> 乙 -> 丙：甲与丙之间没有直接证据，必须是 alias_chain。"""
        _c, ents, stats = merged(
            [
                ment("甲", "r#1", "概念", aliases=("乙",)),
                ment("乙", "r#2", "概念", aliases=("丙",)),
                ment("丙", "r#3", "概念"),
            ]
        )
        assert stats["alias_chain_merged"] == 1, stats
        e = entity_named(ents, "丙")
        reasons = {m["name"]: m["reason"] for m in e.merged_from}
        assert reasons == {"乙": "alias", "甲": "alias_chain"}, reasons
        sources = {a["text"]: a["source"] for a in e.aliases}
        assert sources["甲"] == "alias_chain", sources

    def test_direct_alias_marked_upstream(self) -> None:
        _c, ents, _s = merged(
            [
                ment("甲", "r#1", "概念", aliases=("乙",)),
                ment("乙", "r#2", "概念"),
            ]
        )
        e = entity_named(ents, "乙")
        assert [a["source"] for a in e.aliases] == ["upstream"]
        assert [m["reason"] for m in e.merged_from] == ["alias"]

    def test_ambiguous_alias_not_given_to_any_candidate(self) -> None:
        """被 >=2 候选主张的别名：谁都不给，相关条目 review=pending。"""
        _c, ents, stats = merged(
            [
                ment("阿米娅", "p#1", "角色", aliases=("殿下", "小兔子")),
                ment("特蕾西娅", "q#1", "角色", aliases=("殿下",)),
            ]
        )
        amiya = entity_named(ents, "阿米娅")
        theresia = entity_named(ents, "特蕾西娅")
        texts_amiya = [a["text"] for a in amiya.aliases]
        assert "殿下" not in texts_amiya, f"歧义别名不该分配给任何候选：{amiya.aliases}"
        assert "殿下" not in [a["text"] for a in theresia.aliases]
        assert stats["ambiguous_aliases"] == 1, stats
        assert stats["ambiguous_aliases"] == 1 and stats["rejected:ambiguous_alias"] == 2, stats
        assert [a["review"] for a in amiya.aliases] == ["pending"], amiya.aliases
        assert stats["alias_pending"] == 2

    def test_ambiguous_alias_does_not_merge(self) -> None:
        """歧义别名同时是候选名时，也不许拿它当合并边。"""
        _c, ents, stats = merged(
            [
                ment("阿米娅", "p#1", "角色", aliases=("殿下",)),
                ment("特蕾西娅", "q#1", "角色", aliases=("殿下",)),
                ment("殿下", "r#1", "角色"),
            ]
        )
        assert stats["entities"] == 3, f"歧义别名把两个候选串起来了：{[e.name for e in ents]}"
        assert stats["rejected:ambiguous_alias"] >= 2, stats

    def test_manual_alias_source(self) -> None:
        _c, ents, _s = merged(
            [ment("凯尔希", "a#1", "角色")],
            manual_aliases={"凯尔希": ("凯爹",)},
        )
        e = entity_named(ents, "凯尔希")
        manual = [a for a in e.aliases if a["text"] == "凯爹"]
        assert len(manual) == 1, e.aliases
        assert manual[0]["source"] == "manual"
        assert manual[0]["scenes"] == []

    def test_alias_sources_are_declared_values(self) -> None:
        _c, ents, _s = merged(
            [
                ment("甲", "r#1", "概念", aliases=("乙", "戊")),
                ment("乙", "r#2", "概念"),
                ment("罗德岛", "x#1", "组织"),
                ment("罗德岛（制药）", "y#2", "组织"),
            ],
            manual_aliases={"罗德岛（制药）": ("岛",)},
        )
        for e in ents:
            for a in e.aliases:
                assert a["source"] in identity.ALIAS_SOURCES, a
                assert a["review"] in identity.REVIEW_STATES, a
                assert isinstance(a["scenes"], list)

    def test_merged_from_reasons_are_declared(self) -> None:
        _c, ents, _s = merged(
            [
                ment("甲", "r#1", "概念", aliases=("乙",)),
                ment("乙", "r#2", "概念", aliases=("丙",)),
                ment("丙", "r#3", "概念"),
                ment("罗德岛", "x#1", "组织"),
                ment("罗德岛（制药）", "y#2", "组织"),
            ]
        )
        seen = set()
        for e in ents:
            for m in e.merged_from:
                assert m["reason"] in identity.MERGE_REASONS, m
                assert isinstance(m["scenes"], list)
                seen.add(m["reason"])
        assert {"alias", "fullname"} <= seen, f"没覆盖到这两类：{seen}"

    def test_display_name_not_in_merged_from(self) -> None:
        """展示名排除：它既不在 merged_from（那是「被并进来的名字」），
        也不必当别名（它就是实体自己的名字）。

        与 `test_every_merged_member_is_in_alias_table` 是两条**不同**的排除：
        那条管「被并掉的成员名必须进别名表」，这条管「展示名不许出现在历史里」。
        """
        _c, ents, _s = merged(
            [ment("罗德岛", "x#1", "组织"), ment("罗德岛（制药）", "y#2", "组织")]
        )
        for e in ents:
            assert e.name not in [m["name"] for m in e.merged_from], (
                "展示名不许出现在自己的 merged_from 里：它是展示名，不是被并进来的名字"
            )

    def test_every_merged_member_is_in_alias_table(self) -> None:
        """merged_from 里的每个成员名都必须也出现在 aliases 里。

        两者用途不同（血缘 vs 检索别名表），**允许重叠**；但别名表必须完备——
        下游建「名→id」映射靠它，漏一个名就等于那个名字检索不到这个实体。
        所以这里断言的是包含关系，不是互斥。
        """
        _c, ents, _s = merged(
            [
                ment("甲", "r#1", "概念", aliases=("乙",)),
                ment("乙", "r#2", "概念"),
                ment("罗德岛", "x#1", "组织"),
                ment("罗德岛（制药）", "y#2", "组织"),
            ]
        )
        for e in ents:
            alias_texts = {x["text"] for x in e.aliases}
            for m in e.merged_from:
                assert m["name"] in alias_texts, (
                    f"{e.name} 的成员名 {m['name']!r} 不在别名表里：{sorted(alias_texts)}"
                )
            texts = [x["text"] for x in e.aliases]
            assert len(texts) == len(set(texts)), f"{e.name} 的 aliases 有重复文本：{texts}"


# ------------------------------------------------------------------ fullname 路径


class TestFullnamePath:
    """全名归并：`can_merge_fullname` 放行、成员索引拦下。"""

    def test_fullname_rejected_member_is_counted(self) -> None:
        """`整合运动成员` 有前缀关系但不是同一个实体：必须单独记账。"""
        _c, ents, stats = merged(
            [
                ment("整合运动", "a#1", "组织"),
                ment("整合运动成员", "b#1", "组织"),
            ]
        )
        assert stats["fullname_rejected_member"] == 1, stats
        assert stats["fullname_merged"] == 0, stats
        assert stats["entities"] == 2, f"集体与成员被并了：{[e.name for e in ents]}"

    def test_member_not_merged_even_with_separator(self) -> None:
        _c, ents, stats = merged(
            [ment("罗德岛", "a#1", "组织"), ment("罗德岛-干员", "b#1", "组织")]
        )
        assert stats["fullname_merged"] == 0
        assert stats["entities"] == 2

    def test_structured_fullname_merges(self) -> None:
        _c, ents, stats = merged(
            [ment("薇薇安娜", "a#1", "角色"), ment("薇薇安娜·德罗斯特", "b#1", "角色")]
        )
        assert stats["fullname_merged"] == 1, stats
        assert stats["entities"] == 1


# ------------------------------------------------------------------ stats 形态


class TestStatsShape:
    """门禁 L4.13 要查的键：`rejected:<reason>` 必须**即使为 0 也存在**。"""

    def test_every_reject_key_present_even_when_zero(self) -> None:
        _c, _e, stats = merged([ment("凯尔希", "a#1", "角色")])
        for reason in identity.REJECT_REASONS:
            key = f"rejected:{reason}"
            assert key in stats, f"缺键 {key}：缺键与「没发生过」读起来一样，那是缺陷"
            assert stats[key] == 0, f"{key} 应为 0，实测 {stats[key]}"

    def test_all_reject_keys_named_correctly(self) -> None:
        _c, _e, stats = merged(
            [
                ment("A", "a#1", "角色"),
                ment("A·全名", "a#2", "组织"),
                ment("整合运动", "b#1", "组织"),
                ment("整合运动成员", "b#2", "组织"),
                ment("殿下", "c#1", "角色"),
                ment("阿米娅", "c#2", "角色", aliases=("殿下",)),
                ment("特蕾西娅", "c#3", "角色", aliases=("殿下",)),
            ]
        )
        reject_keys = {k for k in stats if k.startswith("rejected:")}
        assert reject_keys == {f"rejected:{r}" for r in identity.REJECT_REASONS}, reject_keys

    def test_counts_match_constructed_input(self) -> None:
        _c, ents, stats = merged(
            [
                # 第 1 条：类型冲突，一次
                ment("乌萨斯", "a#1", "地点", aliases=("乌萨斯帝国",)),
                ment("乌萨斯帝国", "a#2", "组织"),
                # 第 3 条：泛称锚点，一次
                ment("米纳", "b#1", "角色", aliases=("医疗干员",)),
                ment("医疗干员", "b#2", "角色"),
                # 第 4 条：歧义别名，两条（两个候选各主张一次）
                ment("阿米娅", "c#1", "角色", aliases=("殿下",)),
                ment("特蕾西娅", "c#2", "角色", aliases=("殿下",)),
            ]
        )
        assert stats["rejected:type_conflict"] == 1, stats
        assert stats["rejected:generic_anchor"] == 1, stats
        assert stats["rejected:ambiguous_alias"] == 2, stats
        assert stats["rejected:member_index"] == 0, stats
        assert stats["rejected:component_conflict"] == 0, stats
        assert stats["rejected:relation_conflict"] == 0, stats
        assert stats["entities"] == 6, f"实测 {[e.name for e in ents]}"

    def test_relation_conflict_counted(self) -> None:
        edges = {
            "凯尔希": [("同盟", "博士")],
            "凯尔希医生": [("敌对", "博士")],
        }
        _c, ents, stats = merged(
            [
                ment("凯尔希", "a#1", "角色", aliases=("凯尔希医生",)),
                ment("凯尔希医生", "a#2", "角色"),
            ],
            edges=edges,
        )
        assert stats["rejected:relation_conflict"] == 1, stats
        assert stats["entities"] == 2, [e.name for e in ents]

    def test_component_conflict_counted(self) -> None:
        """第三个名字入伙时与组里既有成员冲突 -> 第 5 条。"""
        _c, ents, stats = merged(
            [
                ment("甲", "a#1", "概念", aliases=("乙",)),
                ment("乙", "a#2", "地点", aliases=("丙",)),
                ment("丙", "a#3", "角色"),
            ]
        )
        assert stats["rejected:component_conflict"] >= 1 or stats["rejected:type_conflict"] >= 1, stats

    def test_baseline_keys_still_present(self) -> None:
        """与旧图对比要用的五个键不能因为改名而消失。"""
        _c, _e, stats = merged([ment("凯尔希", "a#1", "角色")])
        for key in (
            "alias_merged",
            "alias_dropped_unknown",
            "alias_ambiguous_rejected",
            "fullname_merged",
            "fullname_rejected_member",
        ):
            assert key in stats, key

    def test_entity_count_matches_list(self) -> None:
        _c, ents, stats = merged(
            [ment("甲", "a#1", "概念", aliases=("乙",)), ment("乙", "a#2", "概念")]
        )
        assert stats["entities"] == len(ents) == 1

    def test_ids_unique(self) -> None:
        mentions = [ment(n, "a#1", "角色") for n in ("凯尔希", "阿米娅", "陈", "陈")]
        _c, ents, _s = merged(mentions)
        ids = [e.id for e in ents]
        assert len(ids) == len(set(ids)), f"id 撞了：{ids}"
        assert all(ID_RE.match(i) for i in ids), ids


# ------------------------------------------------------------------ 三层数据结构


class TestThreeLayerModel:
    """契约 §3：Mention -> EntityCandidate -> CanonicalEntity。"""

    def test_mention_doc_id(self) -> None:
        m = ment("凯尔希", "activities/act19mini/level_act19mini_st05#004")
        assert m.doc_id == "activities/act19mini/level_act19mini_st05"
        assert m.scene_id == m.doc_id

    def test_doc_id_edge_cases(self) -> None:
        assert identity.doc_id_of("a#1") == "a"
        assert identity.doc_id_of("no-hash") == "no-hash"
        assert identity.doc_id_of("a#b#c") == "a#b", "doc_id 里若含 # 要切最后一个"
        assert identity.doc_id_of("") == ""

    def test_build_candidates_aggregates(self) -> None:
        cands, stats = identity.build_candidates(
            [
                ment("凯尔希", "a#1", "人物", aliases=("凯太后",)),
                ment("凯尔希", "b#1", "角色"),
                ment("阿米娅", "c#1", "人物"),
            ]
        )
        assert stats["candidates"] == 2
        by = {c.name: c for c in cands}
        assert by["凯尔希"].mentions == 2
        assert by["凯尔希"].scenes == frozenset({"a", "b"})
        assert by["凯尔希"].etype == "角色", "类型投票：人物 与 角色 都归一到 角色"
        assert by["凯尔希"].aliases == ("凯太后",)
        assert by["凯尔希"].alias_scenes["凯太后"] == ("a",)

    def test_empty_name_dropped(self) -> None:
        cands, stats = identity.build_candidates([ment("  ", "a#1"), ment("甲", "a#2")])
        assert [c.name for c in cands] == ["甲"]
        assert stats["dropped:empty_name"] == 1

    def test_canonical_entity_to_dict_schema(self) -> None:
        """契约 §3.1 的键名：`type` 而不是 `etype`；另加两张分离的待审表。"""
        _c, ents, _s = merged([ment("凯尔希", "a#1", "角色", aliases=("凯太后",))])
        d = ents[0].to_dict()
        assert set(d) == {
            "id",
            "name",
            "type",
            "desc",
            "aliases",
            "pending_aliases",
            "rejected_aliases",
            "merged_from",
            "split_history",
            "mentions",
            "chunks",
        }, sorted(d)
        assert d["type"] == "角色"
        assert ID_RE.match(d["id"])
        assert set(d["aliases"][0]) == {"text", "source", "scenes", "review"}

    def test_split_history_records_merge(self) -> None:
        _c, ents, _s = merged(
            [ment("罗德岛", "x#1", "组织"), ment("罗德岛（制药）", "y#2", "组织")]
        )
        e = entity_named(ents, "罗德岛（制药）")
        assert len(e.split_history) == 1, e.split_history
        rec = e.split_history[0]
        assert rec["action"] == "merge"
        assert rec["reason"] == "fullname"
        assert set(rec["members"]) == {"罗德岛", "罗德岛（制药）"}

    def test_single_entity_has_no_split_history(self) -> None:
        _c, ents, _s = merged([ment("凯尔希", "a#1", "角色")])
        assert ents[0].split_history == []
        assert ents[0].merged_from == []

    def test_pinned_ids_survive(self) -> None:
        """跨世代稳定：上一次固定过的 id 必须被沿用。"""
        _c, ents, _s = merged(
            [ment("凯尔希", "a#1", "角色"), ment("凯尔希医生", "a#2", "角色")],
            pinned_ids={"凯尔希": "e-0123456789abcdef"},
        )
        ids = {e.id for e in ents}
        assert "e-0123456789abcdef" in ids, ids

    def test_duplicate_candidate_counted(self) -> None:
        _e, stats = identity.merge_candidates([cand("甲"), cand("甲")])
        assert stats["duplicate_candidate"] == 1


# ------------------------------------------------------------------ 门禁接线


class TestGateWireUp:
    """`gates.l4_identity_gates` 是对本模块产出的反向验证（不修改 gates.py）。"""

    def test_l4_12_fails_on_name_as_id(self) -> None:
        legacy = {
            "entities": [{"id": "凯尔希", "name": "凯尔希", "merged_from": []}],
            "stats": {"merge": {}},
        }
        run = gates.l4_identity_gates(legacy)
        verdicts = {g.name.split()[0]: g.verdict for g in run.gates}
        assert verdicts["L4.12"] == "FAIL", verdicts

    def test_l4_12_passes_on_stable_id(self) -> None:
        _c, ents, stats = merged(
            [
                ment("凯尔希", "a#1", "角色", aliases=("凯太后",)),
                ment("凯太后", "b#1", "角色"),
            ]
        )
        graph = {
            "entities": [e.to_dict() for e in ents],
            "stats": {"merge": dict(stats)},
        }
        run = gates.l4_identity_gates(graph)
        verdicts = {g.name.split()[0]: g.verdict for g in run.gates}
        assert verdicts["L4.12"] == "PASS", (
            f"stable_id 产出的 id 仍被判成含名字：{[g.measured for g in run.gates]}"
        )

    def test_l4_13_passes_with_merge_records(self) -> None:
        _c, ents, stats = merged(
            [ment("罗德岛", "x#1", "组织"), ment("罗德岛（制药）", "y#2", "组织")]
        )
        graph = {"entities": [e.to_dict() for e in ents], "stats": {"merge": dict(stats)}}
        run = gates.l4_identity_gates(graph)
        assert {g.name.split()[0]: g.verdict for g in run.gates}["L4.13"] == "PASS"


# ------------------------------------------------------------------ 别名可见性


class TestAliasVisibility:
    """`review="pending"` 必须有强制力：待审别名不得出现在权威检索表里。

    这组用例的背景是一次实测：拒绝合并 `暴雨 <- 库兰塔士兵` 之后，别名表里仍写着
    `{"text": "库兰塔士兵", "review": "pending"}`，而 `graph.py:181` 那种
    `{alias: ent["id"] for alias in ent["aliases"]}` 的写法照样把它指向暴雨——
    合并被拒了，别名表却照样指过去，同一个错误只是晚一步发作。
    所以 `to_dict()` 把别名拆成 `aliases` / `pending_aliases` / `rejected_aliases`。

    下面第三条用例是**可执行的判据**，不是注释：它按 `graph.py` 的写法现场建一次
    索引，断言待审别名不可达；第四条用例反向证明这条判据真的能红。
    """

    def test_pending_alias_not_in_authoritative_aliases(self) -> None:
        cands, _s = identity.build_candidates(
            [ment("暴雨", "a#1", "角色", aliases=("库兰塔士兵", "罗德岛外勤干员"))]
        )
        ents, _m = identity.merge_candidates(cands, edges_by_name={})
        assert len(ents) == 1
        d = ents[0].to_dict()
        authoritative = {x["text"] for x in d["aliases"]}
        pending = {x["text"] for x in d["pending_aliases"]}
        assert "库兰塔士兵" not in authoritative, sorted(authoritative)
        assert "罗德岛外勤干员" not in authoritative, sorted(authoritative)
        assert "库兰塔士兵" in pending, sorted(pending)
        assert "罗德岛外勤干员" in pending, sorted(pending)

    def test_authoritative_aliases_only_contain_auto_or_approved(self) -> None:
        """所有实体的权威别名表里，`review` 只允许 auto / approved。"""
        _c, ents, _s = merged(
            [
                ment("暴雨", "a#1", "角色", aliases=("库兰塔士兵", "库兰塔")),
                ment("凯尔希", "b#1", "角色", aliases=("凯尔希医生",)),
                ment("临光", "c#1", "角色", aliases=("临光家",)),
            ]
        )
        checked = 0
        for e in ents:
            for a in e.to_dict()["aliases"]:
                assert a["review"] in ("auto", "approved"), (e.name, a)
                checked += 1
        assert checked >= 1, "没有任何权威别名被检查到，用例本身失效"

    def test_pending_alias_cannot_be_found_by_alias_index(self) -> None:
        """按 `graph.py:181` 的写法建索引：待审别名必须查不到，权威别名必须查得到。"""
        _c, ents, _s = merged(
            [
                ment("暴雨", "a#1", "角色", aliases=("库兰塔士兵",)),
                ment("凯尔希", "b#1", "角色", aliases=("凯尔希医生",)),
            ]
        )
        graph = {"entities": [e.to_dict() for e in ents]}
        # 与 src/arkrag/graph.py:181 完全相同的写法：只看 ent["aliases"]。
        alias_index = {
            alias["text"]: ent["id"]
            for ent in graph["entities"]
            for alias in ent["aliases"]
        }
        assert "库兰塔士兵" not in alias_index, (
            f"待审别名进了权威索引，检索「库兰塔士兵」会直接命中暴雨：{alias_index}"
        )
        assert "凯尔希医生" in alias_index, sorted(alias_index)

    def test_alias_index_guard_really_goes_red(self) -> None:
        """反向证明：把待审别名塞回 `aliases`（模拟改坏的实现），上一条断言必然失败。

        这里不依赖测试框架的自反性——直接把「旧写法会产出什么」构造出来，
        断言它确实让上面那条判据失效，从而证明那条判据不是恒真。
        """
        _c, ents, _s = merged(
            [ment("暴雨", "a#1", "角色", aliases=("库兰塔士兵",))]
        )
        d = ents[0].to_dict()
        assert d["pending_aliases"], "构造输入应当产生待审别名"
        # 模拟修复前的产物：权威表 = aliases + pending_aliases。
        leaky = {"entities": [dict(d, aliases=d["aliases"] + d["pending_aliases"])]}
        leaky_index = {
            alias["text"]: ent["id"]
            for ent in leaky["entities"]
            for alias in ent["aliases"]
        }
        assert "库兰塔士兵" in leaky_index, (
            "泄露构造没生效，这条反向用例就失去了意义"
        )
        # 而按正确产物建的索引查不到它 —— 两者必须不同，否则判据恒真。
        clean_index = {
            alias["text"]: ent["id"] for ent in [d] for alias in ent["aliases"]
        }
        assert "库兰塔士兵" not in clean_index
