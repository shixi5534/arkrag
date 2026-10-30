"""负例测试：每条都针对计划书的一个反模式，且必须在真实覆盖范围内跑。

这些测试的意义是「门真的能挡住失效」，不是装饰——任何一条 FAIL 都不许进入下一阶段。
"""
from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arkrag import cache, chunk as chunker, config, fingerprint, gates, llm, parse  # noqa: E402

FAILS: list[str] = []


def expect(cond: bool, message: str) -> None:
    """断言。失败时**立刻抛异常**，而不是只往列表里记一笔。

    ## 为什么不能只记录

    旧写法是 `if not cond: FAILS.append(message)`，只有 `main()` 会在末尾读
    `FAILS`。于是 `pytest tests/test_negative.py` 显示全部通过——每个
    `test_*` 函数都正常返回了，pytest 无从知道里面记过失败。

    实测代价：分块器把 ordinal 记成块内下标（见 `chunk._with_ordinal` 的
    docstring）导致 40 轮只覆盖 7 轮时，`pytest` 报 **11 passed**，
    只有直接运行本文件才看到那 1 条 FAIL。这种"假通过"比没有测试更糟：
    CI 绿着，缺陷照样躺在那里。

    现在失败就是异常，两种运行方式的结论必然一致：
    pytest 报 FAILED，`main()` 捕获后汇总成 FAIL 列表。
    """
    if not cond:
        raise AssertionError(message)


def synthetic(payload: dict, name: str) -> Path:
    tmp = Path(tempfile.mkdtemp()) / name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8", newline="\n")
    return tmp


def story(nodes: list[dict], **meta) -> dict:
    base = {"lang": "zh_CN", "eventid": "t", "storyName": "负例", "storyList": nodes}
    base.update(meta)
    return base


def node(index: int, prop: str, **attrs) -> dict:
    return {"id": index, "prop": prop, "attributes": attrs}


def test_player_option_never_becomes_turn() -> None:
    path = synthetic(
        story(
            [
                node(0, "name", content="   阿米娅：我们走。", name="阿米娅"),
                node(1, "Decision", options="（挥挥手）;（鼓鼓掌）;加油——！", values="1;2;3"),
                node(2, "HEADER", key="title_test"),
            ]
        ),
        "neg_option.json",
    )
    stats: Counter = Counter()
    scene = parse.parse_story(path, stats, rel_stem="obt/main/level_neg_00-01_beg")
    expect(scene is not None, "含对话的文件不应被整篇丢弃")
    props = {t.prop for t in scene.turns}
    expect("decision" not in props, "玩家选项 prop 进入了轮次（L1.1 门线为 0）")
    expect(stats["drop_class:player_option"] == 1, f"Decision 未被计入 player_option 计数: {stats}")
    expect(stats["drop_class:title_screen"] == 1, "HEADER 未被计入 title_screen 计数")
    expect(all(not t.text.startswith("   ") for t in scene.turns), "前导空格未剥离，会污染 content_key")


def test_speakerless_content_is_narration() -> None:
    path = synthetic(
        story(
            [
                node(0, "name", content="   ......好热......"),
                node(1, "name", content="   你还好吗", name="阿米娅"),
                node(2, "name", content="   ???", name="？？？"),
                node(3, "subtitle", text="夜幕降临。"),
                node(4, "sticker", text="亲爱的K先生："),
                node(5, "animtext", content="卡兹戴尔南部地区", name="group_location_stamp"),
                node(6, "name", content="   ***！我宰了你！", name="醉汉A"),
            ]
        ),
        "neg_kinds.json",
    )
    stats: Counter = Counter()
    scene = parse.parse_story(path, stats, rel_stem="obt/main/level_neg_00-01_beg")
    kinds = [t.kind for t in scene.turns]
    expect(kinds[0] == "narration", f"无说话人的 content 应判旁白，实得 {kinds[0]}")
    expect(kinds[1] == "dialogue", "有说话人应判对话")
    expect(kinds[3] == "narration", f"subtitle 应判旁白（决策 D-05），实得 {kinds[3]}")
    expect(kinds[4] == "quote", f"sticker 应判引用（决策 D-06），实得 {kinds[4]}")
    expect(kinds[5] == "title_card", f"animtext 应判标题卡，实得 {kinds[5]}")
    expect(scene.turns[2].speaker_masked, "？？？ 说话人未打掩码标记")
    expect(kinds[6] == "dialogue", "被打码台词必须留在库内，不得当残留标记删掉")
    scene_dicts = [scene.as_dict()]
    run = gates.l1_gates(scene_dicts, stats)
    markdown = next(g for g in run.gates if g.name.startswith("L1.3"))
    expect(markdown.verdict == "PASS", f"负例（打码台词）不该触发 markdown 残留门: {markdown.measured}")


def test_unknown_prop_is_counted_not_silent() -> None:
    path = synthetic(
        story(
            [
                node(0, "name", content="   测试台词", name="阿米娅"),
                node(1, "brandNewProp", content="   未来出现的新形态"),
            ]
        ),
        "neg_unknown.json",
    )
    stats: Counter = Counter()
    scene = parse.parse_story(path, stats, rel_stem="obt/main/level_neg_00-01_beg")
    expect(stats["drop_class:unknown_prop"] == 1, "未登记 prop 必须进 unknown 计数而不是静默丢弃")
    expect("brandnewprop" not in {t.prop for t in scene.turns}, "未登记 prop 不得进轮次")


def test_case_folded_props() -> None:
    path = synthetic(
        story(
            [
                node(0, "name", content="   台词", name="阿米娅"),
                node(1, "Subtitle", text="字幕卡形态"),
                node(2, "subtitle", text="小写形态"),
                node(3, "DELAY", time=1),
                node(4, "daley", time=1),
            ]
        ),
        "neg_case.json",
    )
    stats: Counter = Counter()
    scene = parse.parse_story(path, stats, rel_stem="obt/main/level_neg_00-01_beg")
    kinds = [t.kind for t in scene.turns]
    expect(kinds.count("narration") == 2, f"Subtitle/subtitle 两种大小写都要收进旁白，实得 {kinds}")
    expect(stats["drop_class:unknown_prop"] == 0, f"已登记 prop 的大小写变体不该算未知: {stats}")
    expect(stats["drop_class:typo_staging"] == 1, "拼写错误变体 daley 应归 typo_staging")


def test_chunks_never_cross_scene() -> None:
    turns = [
        {"kind": "dialogue", "speaker": f"角色{i%3}", "text": "句" * 120, "node_id": i, "prop": "name", "masked": False}
        for i in range(40)
    ]
    scene = {
        "doc_id": "obt/main/level_main_00-01_beg",
        "category": "obt/main",
        "chapter": "main_ep00",
        "title": "负例",
        "turns": turns,
    }
    chunks = chunker.split_scene(scene)
    expect(len(chunks) > 1, f"40 个长轮次应该切成多块，实得 {len(chunks)}")
    expect(all(c["doc_id"] == scene["doc_id"] for c in chunks), "块跨场景")
    expect(
        all(c["chars"] <= config.CHUNK_MAX_CHARS for c in chunks),
        f"有块超过硬上限 {config.CHUNK_MAX_CHARS}（计划 D2）",
    )
    # 轮次覆盖按"实例序号"算，不按 node_id（计划 D1）：node_id 是源文件给的编号，
    # 会缺失也会重复，拿它当唯一键在重复时会把两轮当成一轮、静默丢内容。
    covered = set()
    for c in chunks:
        covered |= set(c["turn_ordinals"])
    expect(covered == set(range(len(turns))), "有轮次未被任何块覆盖（静默丢内容）")


def test_duplicate_node_id_keeps_both_turns() -> None:
    """计划 D1 必测：node_id 相同的两轮若正文不同，都必须保留。"""
    turns = [
        {"kind": "dialogue", "speaker": "凯尔希", "text": "第一轮正文", "node_id": 1, "prop": "name", "masked": False},
        {"kind": "dialogue", "speaker": "阿米娅", "text": "第二轮正文", "node_id": 1, "prop": "name", "masked": False},
        {"kind": "dialogue", "speaker": "博士", "text": "第三轮正文", "node_id": 2, "prop": "name", "masked": False},
    ]
    scene = {"doc_id": "obt/main/level_dup", "category": "obt/main", "chapter": "c", "title": "t", "turns": turns}
    chunks = chunker.split_scene(scene)
    text = "".join(c["text"] for c in chunks)
    for turn in turns:
        expect(turn["text"] in text, f"node_id={turn['node_id']} 的轮次被当成重复丢掉了: {turn['text']}")
    expect(
        sum(len(c["turn_ordinals"]) for c in chunks) >= 3,
        f"覆盖的轮次实例不足 3（重叠会让它大于 3，但不该小于）: {[c['turn_ordinals'] for c in chunks]}",
    )


def test_fingerprint_tracks_normalizer_version() -> None:
    scenes = [
        {"doc_id": "b", "turns": [{"kind": "dialogue", "speaker": "X", "text": "你好"}]},
        {"doc_id": "a", "turns": [{"kind": "narration", "speaker": "", "text": "夜里"}]},
    ]
    fp = fingerprint.corpus_fingerprint(scenes)
    fp_again = fingerprint.corpus_fingerprint(list(reversed(scenes)))
    expect(fp == fp_again, "语料指纹必须与遍历顺序无关（按 doc_id 排序）")
    base = fingerprint.graph_fingerprint("s1", fp, "m1", "v1", 900, "n1")
    moved = fingerprint.graph_fingerprint("s1", fp, "m1", "v1", 900, "n2")
    expect(base != moved, "归一化版本没进指纹：图变了指纹不变，所有回滚锚点会失效（不变量 5）")


def test_cache_key_excludes_normalizer() -> None:
    k1 = cache.cache_key("abc", "model-a", "v1")
    k2 = cache.cache_key("abc", "model-a", "v1")
    k3 = cache.cache_key("abc", "model-a", "v2")
    k4 = cache.cache_key("abd", "model-a", "v1")
    expect(k1 == k2, "同一输入必须得到同一缓存键（不变量 7）")
    expect(k1 != k3, "提示词版本必须进缓存键，否则改提示词看不到变化")
    expect(k1 != k4, "正文变化必须换键")
    import inspect

    sig = inspect.signature(cache.cache_key).parameters
    expect("normalizer" not in sig, "归一化版本不得进缓存键（决策三，否则改规则要重新付费抽取）")


def test_circuit_open_blocks_dispatch() -> None:
    """熔断命中后剩余块一个都不许发起调用。"""
    with tempfile.TemporaryDirectory() as tmp:
        marker = Path(tmp) / "circuit_open.json"
        marker.write_text(json.dumps({"reason": "quota"}), encoding="utf-8")
        original = config.CIRCUIT_OPEN_PATH
        config.CIRCUIT_OPEN_PATH = marker
        try:
            gauge = llm.InflightGauge()
            state = llm.circuit_state()
            expect(state is not None, "熔断标记未被读到")
            chunks = [{"chunk_id": f"c{i}", "content_key": f"k{i}", "text": "x", "title": "t",
                       "chapter": "c", "doc_id": "d", "speakers": []} for i in range(10)]
            config.LLM_BASE_URL = "http://127.0.0.1:9/v1"
            config.LLM_MODEL = "negative-test-model"
            client = llm.LLMClient(gauge=gauge)
            results = [extract_dispatch(client, c) for c in chunks]
            expect(gauge.total_calls == 0, f"熔断后仍发起了 {gauge.total_calls} 次调用")
            expect(all(r["cache"] == "circuit" for r in results), "熔断后的块未标记为 circuit")
        finally:
            config.CIRCUIT_OPEN_PATH = original


def extract_dispatch(client, chunk):
    from arkrag import extract

    return extract.extract_chunk(client, chunk)


def test_trial_gates_actually_fire() -> None:
    """门必须能挡住失效：注入四类违规，逐条确认对应门翻 FAIL。
    永不触发的门等于没有门——它的 PASS 不具备任何信息量。"""
    text = "阿米娅：凯尔希，我们前往巴别塔。"
    good = {
        "chunk_id": "c0",
        "text": text,
        "raw": {
            "entities": [
                {"name": "阿米娅", "type": "角色", "aliases": [], "desc": "罗德岛领导者"},
                {"name": "巴别塔", "type": "地点", "aliases": [], "desc": "一座飞行要塞"},
            ],
            "relations": [
                {"source": "阿米娅", "relation": "前往", "target": "巴别塔",
                 "evidence": "我们前往巴别塔", "confidence": 0.9}
            ],
        },
    }
    base = gates.l3_trial_gates([good])
    unexpected = [g.name for g in base.gates if g.verdict == "FAIL" and not g.name.startswith("L3.T0 块数")]
    expect(not unexpected, f"干净样本被判 FAIL: {unexpected}")

    bad_records = [
        {**good, "chunk_id": "c1", "raw": {
            "entities": [{"name": "阿米娅", "type": "角色", "aliases": [], "desc": "x"},
                         {"name": "巴别塔", "type": "地点", "aliases": [], "desc": "y"}],
            "relations": [{"source": "阿米娅", "relation": "前往", "target": "巴别塔",
                           "evidence": "阿米娅去了巴别塔", "confidence": 0.9}]}},          # 证据是改写不是原文
        {**good, "chunk_id": "c2", "raw": {
            "entities": [{"name": "阿米娅", "type": "角色", "aliases": [], "desc": "x"},
                         {"name": "巴别塔", "type": "地点", "aliases": [], "desc": "y"}],
            "relations": [{"source": "阿米娅", "relation": "启程前往", "target": "巴别塔",
                           "evidence": "我们前往巴别塔", "confidence": 0.9}]}},          # 词表外关系名
        {**good, "chunk_id": "c3", "raw": {
            "entities": [{"name": "他们", "type": "角色", "aliases": ["他们"], "desc": ""},
                         {"name": "LC0073458", "type": "物品", "aliases": [], "desc": "编号"}],
            "relations": [{"source": "罗德岛", "relation": "包含", "target": "巴别塔",
                           "evidence": "我们前往巴别塔", "confidence": 2.0}]}},          # 代词+代码+悬空端点+置信越界+空desc+别名自含
    ]
    fired = gates.l3_trial_gates([good] + bad_records)
    failed_names = {g.name for g in fired.gates if g.verdict == "FAIL"}
    for expected in ("L3.T1 证据可在块正文定位（逐条比对）", "L3.T3 关系名全部在受控词表内",
                     "L3.T4 关系两端都在 entities 内", "L3.T6 代词被抽成实体",
                     "L3.T8 编号/代码形态成实体", "L3.T12 实体缺 desc",
                     "L3.T13 别名包含规范名自身", "L3.T15 置信度越界"):
        expect(expected in failed_names, f"注入违规后 {expected} 仍然 PASS（门是摆设）")
    # 泛称按 §3.3 归 L4 本地过滤，raw 层不判 FAIL，但必须"看得见"：计数不能是 0
    t9 = next(g for g in fired.gates if g.name.startswith("L3.T9"))
    expect(t9.measured != "0", "注入的泛称未被计数（L4 过滤器将失去输入）")
    generic_gate = next(g for g in base.gates if g.name.startswith("L3.T9"))
    expect(generic_gate.measured == "0", f"干净样本的泛称计数不为 0: {generic_gate.measured}")


def test_relation_adjudication_keeps_facts() -> None:
    """十步裁决的顺序本身就是判据：否定必须早于归约，否则事实整个反过来。"""
    from arkrag import norm_relation as nr

    for raw in ("不隶属于", "非隶属", "拒绝帮助", "未能救助", "无法离开", "不再需要保护", "放弃前往"):
        d = nr.adjudicate(raw)
        expect(d.action == "drop" and d.flag == "negation", f"{raw} 未被否定分支拦住，实得 {d.action}/{d.flag}")
        expect(d.canonical != "隶属", f"{raw} 被归约成了肯定关系：{d.canonical}")

    # 势力归属不能掉进所有权关系：从属于(3) 必须赢过 属于(2)
    expect(nr.adjudicate("从属于罗德岛").canonical == "隶属", "从属于罗德岛 归组错误（应为隶属）")
    expect(nr.adjudicate("隶属于罗德岛").canonical == "隶属", "隶属于 归组错误")
    # 两个方向都在词表里，模型可以直接表达；归约不得把方向抹平
    expect(nr.adjudicate("师从").canonical == "师从", "师从 不应被改写成指导")
    expect(nr.adjudicate("是她的学生").canonical in ("师从", "指导"), "学生类谓语未归组")
    # 捞不回来的必须"保留原值并打标"，不许静默删（不变量 3）
    keep = nr.adjudicate("他们之间的关系")
    expect(keep.action == "keep_flagged" and keep.canonical == keep.raw, f"未映射关系被丢了：{keep}")
    empty = nr.adjudicate("")
    expect(empty.action == "drop" and empty.flag == "empty", "空关系判错")
    junk = nr.adjudicate("镜头")
    expect(junk.action == "drop" and junk.flag == "junk_word", f"噪声词未丢弃：{junk}")


def main() -> int:
    suite = [
        test_player_option_never_becomes_turn,
        test_speakerless_content_is_narration,
        test_unknown_prop_is_counted_not_silent,
        test_case_folded_props,
        test_chunks_never_cross_scene,
        test_fingerprint_tracks_normalizer_version,
        test_cache_key_excludes_normalizer,
        test_circuit_open_blocks_dispatch,
        test_trial_gates_actually_fire,
        test_relation_adjudication_keeps_facts,
    ]
    # 现在 expect() 是抛异常的，所以 main() 必须逐个捕获——否则第一个失败
    # 就会中断整轮，看不到后面还有几条红。这条汇总路径的意义是：单文件直接
    # 运行时能一次看到全部失败，而 pytest 各自独立、天然就是这个行为。
    for fn in suite:
        try:
            fn()
        except AssertionError as exc:
            FAILS.append(f"{fn.__name__}: {exc}")
    if FAILS:
        print(f"负例测试 FAIL（{len(FAILS)} 条，共跑 {len(suite)} 项）:")
        for f in FAILS:
            print("  -", f)
        return 1
    print(f"负例测试 {len(suite)}/{len(suite)} PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
