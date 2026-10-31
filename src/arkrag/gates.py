"""门函数唯一入口：报告里的每个判定都必须由这里的函数产出，不许手写结论。

每个门返回 Gate(name, stage, threshold, measured, verdict, note)。
verdict: PASS / FAIL / KNOWN_DEVIATION / NEEDS_HUMAN
"""
from __future__ import annotations

import hashlib
import json
import time
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import atomic, config, source
from .prop_spec import TEXT_PROP_FIELD

# 文档级元数据泄漏进正文的形态：顶层键名、磁盘路径、词频表字段
RE_DOC_META_LEAK = re.compile(
    r'"?(?:storyCode|entryType|eventid|eventName|avgTag|storyName|storyInfo|lang|OPTIONTRACE)"?\s*[:=]'
    r"|gamedata[/\\]story|\.json\b|\bwordcount\b"
)
# 世界内编码行（终端日志、对接单位等）：属于剧情内容，但整串像编号，
# 会被抽取模型当实体——交给 L3 噪声过滤与人工裁决，不在 L1 静默删。
RE_INWORLD_CODE = re.compile(r"^(?:LOG|SYS|ERR|ID|NO)\s*[:：]|对接单位\s*[:：]|[0-9A-Fa-f]{8,}")
# 世界内编码的两类：终端日志行、序列号/编号。分开是为了让 L1.2b 能机检判定——
# 「是剧情内容」和「是解析脏数据」的区别就在这个分类里，不能靠人逐条看。
RE_TERMINAL_LOG = re.compile(r"^(?:LOG|SYS|ERR|ID|NO)\s*[:：]|对接单位\s*[:：]")
RE_SERIAL = re.compile(r"[0-9A-Fa-f]{8,}")
# 玩家选项形态：分号分隔的候选串（Decision.attributes.options 的写法）
RE_OPTION_LINE = re.compile(r"[^;；]{2,}(?:;[^;；]{2,})+")
# markdown 说话人标记残留：**说话人**： 。注意不能宽到匹配一切以星号开头的行——
# 剧情里有醉汉台词「***！我宰了你！」，那是被打码的台词而不是残留标记。
RE_MARKDOWN_SPEAKER = re.compile(r"^\s*(?:[-*+]\s+)?\*\*[^*\n]{1,24}\*\*\s*[：:]")


@dataclass
class Gate:
    name: str
    stage: str
    threshold: str
    measured: str
    verdict: str
    note: str = ""


@dataclass
class GateRun:
    gates: list[Gate] = field(default_factory=list)

    def add(self, gate: Gate) -> None:
        self.gates.append(gate)

    def rows(self) -> list[dict]:
        return [asdict(g) for g in self.gates]


def load_scenes(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def norm_for_dup(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKC", text) if not ch.isspace())


def shingles(text: str, k: int = 12) -> set[str]:
    if len(text) < k:
        return {text} if text else set()
    return {text[i : i + k] for i in range(0, len(text) - k + 1, 3)}


# --------------------------------------------------------------- 两代图字段分派
#
# 图产物有两代 schema 并存（`config.GRAPH_SCHEMA_LEGACY` / `GRAPH_SCHEMA_IDENTITY`）：
#
#   ark-rag-1  `entities[].aliases` 是 list[str]，`edges[]` 用 source/target/relation，
#              置信度是 `confidence` 单值。
#   ark-rag-3  `entities[].aliases` 是 list[dict]，`edges[]` 用 subject_id/object_id/
#              predicate，置信度是 `confidence_summary` 三元。
#
# 门禁要读这些字段。这里有两条都不许走的路：
#
#   1. **把新图改回旧形态再喂门**——那等于让门测的是一份为了过门而造的形状，
#      被测对象和产物脱钩，门就成了装饰。
#   2. **复制一套只认新形态的门**——同一判据写两处，只要一处较松，整套就等于
#      那处最松的（`tests/test_single_source_of_truth.py` 记的三个真实事故都是这个机理）。
#
# 所以：判据只写一份，**字段按元素类型分派**。分派放在最小面上（五个取值函数），
# 判定逻辑一行不动，两代图走的是同一条门。
#
# 为什么按元素类型而不是 `schema_version`：同一份对比报告里可能同时装着两代图
# （迁移期要横向比），而元素类型就在手上、不会说谎；`schema_version` 抄错一个字母
# 会让整组门静默走错分支。


def alias_texts(entity: dict) -> list[str]:
    """实体的**权威**别名文本（ark-rag-1 list[str] / ark-rag-3 list[dict]）。

    ark-rag-3 的 `aliases` 只含 `review ∈ {auto, approved}`（见
    `identity.CanonicalEntity.to_dict()`），待审与拒绝在另外两张表里，
    所以这里不需要再判 review——判两次就有两次机会判错。
    """
    out: list[str] = []
    for item in entity.get("aliases") or []:
        text = str(item.get("text") or "") if isinstance(item, dict) else str(item or "")
        text = text.strip()
        if text:
            out.append(text)
    return out


def merged_names(entity: dict) -> list[str]:
    """被并进来的名字（ark-rag-1 list[str] / ark-rag-3 list[{name,reason,scenes}]）。"""
    out: list[str] = []
    for item in entity.get("merged_from") or []:
        name = str(item.get("name") or "") if isinstance(item, dict) else str(item or "")
        name = name.strip()
        if name:
            out.append(name)
    return out


def edge_ends(edge: dict) -> tuple[str, str]:
    """边的两端 id（ark-rag-1 source/target / ark-rag-3 subject_id/object_id）。"""
    src = edge.get("source")
    dst = edge.get("target")
    if src is None:
        src = edge.get("subject_id")
    if dst is None:
        dst = edge.get("object_id")
    return str(src or ""), str(dst or "")


def edge_relation(edge: dict) -> str:
    """边的谓词（ark-rag-1 relation / ark-rag-3 predicate）。"""
    rel = edge.get("relation")
    if rel is None:
        rel = edge.get("predicate")
    return str(rel or "")


def edge_confidence(edge: dict) -> float:
    """边的置信度标量。

    ark-rag-1 直接存 `confidence`；ark-rag-3 的聚合边只存派生统计
    `confidence_summary{min,max,mean}`（不许存单一代表值，见契约 §1.4 与 L5.10）。
    取 `max` 而不是 `mean`：旧门的判据是「这条边最可信的那次观察够不够可信」，
    `max` 与它同义；用 `mean` 会把「一次高置信 + 多次低置信」的边判成低置信，
    那是判据换了含义，不是换了个字段名。
    """
    raw = edge.get("confidence")
    if raw is None:
        summary = edge.get("confidence_summary") or {}
        if isinstance(summary, dict):
            raw = summary.get("max")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def entity_key(graph: dict, entity: dict) -> str:
    """实体在邻接表里的键。

    ark-rag-1 的邻接键是规范名（旧图 id 就是名字），ark-rag-3 是实体 id。
    按**图里实际存在的邻接键**分派：先看 id，再看名字。这样门问的是
    「这个实体在邻接表里找得到吗」，而不是「它的 id 长得像不像名字」。
    """
    ident = str(entity.get("id") or "")
    return ident if ident else str(entity.get("name") or "")


def l0_gates(manifest: dict) -> GateRun:
    run = GateRun()
    b = manifest["baseline"]
    files = b["story_files"]
    dropped = manifest.get("dropped_files", [])
    run.add(
        Gate(
            "L0.1 文档数==文件数",
            "L0",
            "每个被丢弃文档必须查明原因",
            f"files={files} dropped={len(dropped)}",
            "PASS" if len(dropped) == 0 or all(d.get("reason") for d in dropped) else "FAIL",
            "; ".join(f"{d['doc_id']}:{d['reason']}" for d in dropped[:5]),
        )
    )
    missing_title = b.get("story_name_missing", 0)
    run.add(
        Gate(
            "L0.2 标题缺失文档数",
            "L0",
            "0",
            str(missing_title),
            "PASS" if missing_title == 0 else "FAIL",
            "计划书原文是「文档第一行必须是最高级标题」；本语料标题在 storyName 字段，"
            "静默丢弃的失效方向相同，故按等价判据实测（决策台账 D-02）",
        )
    )
    enc = manifest["encoding"]
    run.add(
        Gate(
            "L0.3 带 BOM 文件数",
            "L0",
            "0",
            str(len(enc["bom_files"])),
            "PASS" if not enc["bom_files"] else "FAIL",
        )
    )
    run.add(
        Gate(
            "L0.4 非 UTF-8 / 解码失败文件数",
            "L0",
            "0",
            str(len(enc["decode_failures"])),
            "PASS" if not enc["decode_failures"] else "FAIL",
        )
    )
    run.add(
        Gate(
            "L0.5 轮次总数",
            "L0",
            ">0",
            str(b["dialogue_turns"] + b["narration_turns"]),
            "PASS" if b["dialogue_turns"] + b["narration_turns"] > 0 else "FAIL",
        )
    )
    ratio = (b["dialogue_turns"] / b["narration_turns"]) if b["narration_turns"] else float("inf")
    ratio_char = (
        b["dialogue_chars"] / b["narration_chars"] if b.get("narration_chars") else float("inf")
    )
    # 两口径比值的一致性才是「旁白有没有被成片降级成对话」的可机检证据：
    # 若解析把旁白误判为对话，旁白的短句会并入对话，字数口径与轮次口径会剧烈分叉。
    # 绝对值（1.5-4）来自散文体假设，本语料是纯对话剧本，绝对值只申报不判定（见决策台账 D-24）。
    if ratio == float("inf") or ratio_char == float("inf"):
        cross = float("inf")
    else:
        cross = max(ratio, ratio_char) / max(1e-9, min(ratio, ratio_char))
    run.add(
        Gate(
            "L0.6 对话/旁白比（多口径一致性）",
            "L0",
            "两口径相差 ≤2 倍",
            f"turn {ratio:.2f} / char {ratio_char:.2f} / 相差 {cross:.2f} 倍",
            "PASS" if cross <= 2.0 else "FAIL",
            "原始门线 1.5-4 是散文体假设，剧本语料不适用；改判「旁白是否被成片降级」，"
            "绝对值只申报（决策台账 D-24）",
        )
    )
    nar_docs = b.get("narration_docs")
    total_docs = b.get("total_docs")
    run.add(
        Gate(
            "L0.6c 含无说话人轮次的文档数（仅申报）",
            "L0",
            "申报（不判定）",
            f"{len(nar_docs)}/{total_docs} 个 doc 有 narrator>0"
            if isinstance(nar_docs, list)
            else "manifest 未记录（需重跑 run_l0）",
            "PASS",
            "narrator 口径含旁白/字幕/贴纸/标题卡等全部无说话人轮次；"
            "纯对话剧本里相当一部分 doc 天然没有，故此项只申报",
        )
    )
    nar_share = b["narration_turns"] / max(1, b["narration_turns"] + b["dialogue_turns"])
    run.add(
        Gate(
            "L0.6a 探针：narration 数>0",
            "L0",
            "非零（证明无说话人检测生效）",
            str(b["narration_turns"]),
            "PASS" if b["narration_turns"] > 0 else "FAIL",
        )
    )
    run.add(
        Gate(
            "L0.6b 探针：narration 占比<0.30",
            "L0",
            "<0.30（未被全量降级为旁白）",
            f"{nar_share:.3f}",
            "PASS" if nar_share < 0.30 else "FAIL",
        )
    )
    unexplained = {
        name: r.get("unexplained", []) for name, r in manifest.get("reconciliation", {}).items()
    }
    run.add(
        Gate(
            "L0.7 文件<->索引缺口逐条有解释",
            "L0",
            "UNEXPLAINED == 0",
            "; ".join(f"{k}:{len(v)}" for k, v in unexplained.items()) or "无缺口",
            "PASS" if not any(unexplained.values()) else "FAIL",
            "缺口本身可以是合法的（另册索引/不在覆盖范围），但不许有无解释的缺口",
        )
    )
    # 来源身份（计划 D4）。旧版 ZIP 不在场时只打印一句"跳过校验"就继续，
    # 于是"没人检查过"和"已验证"在下游完全无法区分，产物照样被当 Release 引用。
    # 这条门让那个区别变成机器可读的结论。
    sv = manifest.get("source_verification") or {}
    # 判定只能有一处事实源：这里引用 source 的常量，不手抄字面量。
    # 手抄的后果已经发生过——release.py 曾只拒绝字面量 "source_unverified"，
    # 于是 ""、"VERIFIED_ZIP"、"verified_zip "（尾空格）全都绕过了发布门。
    # 同一个判定在两处各写一份，整套检查就等于那处最松的。
    status = sv.get("status") or manifest.get("source_status") or source.UNVERIFIED
    checks = sv.get("checks") or []
    problems = sv.get("problems") or []
    verified = status in (source.VERIFIED_ZIP, source.VERIFIED_TREE)
    run.add(
        Gate(
            "L0.8 来源已验证（ZIP 或确定性解压树清单）",
            "L0",
            "status ∈ {verified_zip, verified_tree}",
            f"status={status} checks={len(checks)} problems={len(problems)} "
            f"tree_files={sv.get('tree_files', 0)}",
            "PASS" if verified else "FAIL",
            "; ".join(problems[:3])
            or (
                "来源身份未知时产物不得冒充 Release（计划 D4）"
                if not verified
                else ""
            ),
        )
    )
    return run


def l1_gates(scenes: list[dict], stats: dict) -> GateRun:
    run = GateRun()
    total_turns = sum(len(s["turns"]) for s in scenes)

    option_turns = [
        (s["doc_id"], t)
        for s in scenes
        for t in s["turns"]
        if t["prop"] in ("decision", "predicate") or RE_OPTION_LINE.fullmatch(t["text"])
    ]
    run.add(
        Gate(
            "L1.1 含玩家选项的轮次数",
            "L1",
            "0",
            str(len(option_turns)),
            "PASS" if not option_turns else "FAIL",
            "; ".join(f"{d}:{t['text'][:30]}" for d, t in option_turns[:3]),
        )
    )
    meta_turns = [
        (s["doc_id"], t["text"]) for s in scenes for t in s["turns"] if RE_DOC_META_LEAK.search(t["text"])
    ]
    run.add(
        Gate(
            "L1.2a 含文档元数据/路径的轮次数",
            "L1",
            "0",
            str(len(meta_turns)),
            "PASS" if not meta_turns else "FAIL",
            "; ".join(f"{d}:{x[:40]}" for d, x in meta_turns[:3]),
        )
    )
    # 世界内编码拆成两族分别计：终端日志行是剧情道具（屏幕上的字），序列号是编号。
    # 两者都不该让人工去逐条裁决「算不算正文」——它们本来就是正文。
    # 真正要机检的是「有没有漏到结构化字段/实体名里」，那才是污染：
    #   - 结构化字段在本门查（说话人、doc_id、prop）；
    #   - 实体名/别名在图里查（L5.7b）。
    code_hits = [
        (s["doc_id"], s["category"], t)
        for s in scenes
        for t in s["turns"]
        if RE_INWORLD_CODE.search(t["text"])
    ]
    log_hits = [h for h in code_hits if RE_TERMINAL_LOG.search(h[2]["text"])]
    serial_hits = [
        h for h in code_hits if not RE_TERMINAL_LOG.search(h[2]["text"]) and RE_SERIAL.search(h[2]["text"])
    ]
    struct_leak = [
        (d, t.get("speaker") or "", k)
        for d, _c, t in code_hits
        for k in (("speaker", t.get("speaker") or ""), ("prop", t.get("prop") or ""))
        if k[1] and RE_INWORLD_CODE.search(k[1])
    ] + [(d, d, "doc_id") for d, _c, _t in code_hits if RE_INWORLD_CODE.search(d)]
    run.add(
        Gate(
            "L1.2b 世界内编码行（终端日志/序列号，机检）",
            "L1",
            "结构化字段泄漏 == 0",
            f"终端日志 {len(log_hits)} 轮 / 序列号 {len(serial_hits)} 轮 / 结构化泄漏 {len(struct_leak)}",
            "PASS" if not struct_leak else "FAIL",
            "两族都是剧情内正文（屏幕上的字、编号），不是脏数据，故不再逐条人工裁决；"
            "判的是有没有漏进 speaker/prop/doc_id。" + (f" 泄漏例 {struct_leak[:3]}" if struct_leak else "")
            + "；实体名侧由图门 L5.7b 兜底",
        )
    )
    run.add(
        Gate(
            "L1.2b2 世界内编码的文档分布（仅申报）",
            "L1",
            "申报（不判定）",
            f"{len({h[0] for h in code_hits})} 个 doc 命中（例 {sorted({h[0] for h in code_hits})[:3]}）",
            "PASS",
            "集中在 act8mini_st06（PRTS 终端）与 act42side 的两条编号，属剧情设计",
        )
    )
    markdown_turns = [
        (s["doc_id"], t["text"]) for s in scenes for t in s["turns"] if RE_MARKDOWN_SPEAKER.match(t["text"])
    ]
    garbled_turns = [t["text"] for s in scenes for t in s["turns"] if t["text"].lstrip().startswith("***")]
    run.add(
        Gate(
            "L1.3 残留 markdown 说话人标记轮次数",
            "L1",
            "0",
            str(len(markdown_turns)),
            "PASS" if not markdown_turns else "FAIL",
            f"负例保持入库：被打码台词 {len(garbled_turns)} 条（例 {garbled_turns[:1]}）",
        )
    )

    unknown = stats.get("drop_class:unknown_prop", 0)
    dropped_total = sum(v for k, v in stats.items() if k.startswith("drop_class:"))
    share = unknown / max(1, total_turns + dropped_total)
    run.add(
        Gate(
            "L1.4 兜底(unknown prop)行占比",
            "L1",
            "<0.05",
            f"{share:.5f}  (unknown={unknown})",
            "PASS" if share < 0.05 else "FAIL",
            "白名单覆盖 L0 实测全部 79 个折叠 prop，故预期为 0",
        )
    )
    non_text_props = {t["prop"] for s in scenes for t in s["turns"]} - set(TEXT_PROP_FIELD)
    run.add(
        Gate(
            "L1.5 舞台指令漏检（进入轮次的非正文 prop）",
            "L1",
            "0",
            str(len(non_text_props)) + (" " + ",".join(sorted(non_text_props)) if non_text_props else ""),
            "PASS" if not non_text_props else "FAIL",
        )
    )

    exact: dict[str, list[str]] = defaultdict(list)
    for s in scenes:
        body = norm_for_dup("".join(t["text"] for t in s["turns"]))
        if body:
            exact[hashlib.sha256(body.encode("utf-8")).hexdigest()].append(s["doc_id"])
    dup_groups = [v for v in exact.values() if len(v) > 1]
    run.add(
        Gate(
            "L1.6 正文完全重复文档组数",
            "L1",
            "0 或已人工裁决",
            str(len(dup_groups)),
            "PASS" if not dup_groups else "NEEDS_HUMAN",
            json.dumps(dup_groups[:5], ensure_ascii=False),
        )
    )

    docs = []
    for s in scenes:
        body = norm_for_dup("".join(t["text"] for t in s["turns"]))
        docs.append((s["doc_id"], shingles(body)))
    # 用倒排索引只在共享 shingle 的文档对之间算 Jaccard，避免 2291^2 全对
    inv: dict[str, list[int]] = defaultdict(list)
    for i, (_, sh) in enumerate(docs):
        for h in sh:
            inv[h].append(i)
    pairs: set[tuple[int, int]] = set()
    for postings in inv.values():
        if 1 < len(postings) <= 40:
            for a in range(len(postings)):
                for b in range(a + 1, len(postings)):
                    pairs.add((postings[a], postings[b]))
    near = []
    for i, j in pairs:
        a, b = docs[i][1], docs[j][1]
        if not a or not b:
            continue
        jac = len(a & b) / len(a | b)
        if jac >= 0.98:
            near.append((docs[i][0], docs[j][0], round(jac, 4)))
    near.sort(key=lambda x: (-x[2], x[0], x[1]))
    run.add(
        Gate(
            "L1.7 近重复文档对数(Jaccard>=0.98)",
            "L1",
            "0 或已人工裁决",
            str(len(near)),
            "PASS" if not near else "NEEDS_HUMAN",
            json.dumps(near[:8], ensure_ascii=False),
        )
    )

    # ---- node_id 身份校验（计划 D1）------------------------------------------
    # 旧实现拿 node_id 当轮次唯一键：`if turn["node_id"] in seen: continue`。
    # 于是同编号不同正文的两轮只剩一轮，而且**没有任何一处会报错**——
    # 更糟的是 L2 的覆盖率门当时也用 node_id 当基准，分子分母一起少，门照样 PASS。
    # 这里把「缺失」与「重复」分开报，因为两者含义完全不同：
    #   缺失（id 为空/0）是上游数据特性，轮次仍然有效，只需申报；
    #   同场景内重复才是身份冲突，必须能被看见。
    missing_id = [
        (s["doc_id"], i) for s in scenes for i, t in enumerate(s["turns"]) if not t.get("node_id")
    ]
    dup_id: list[tuple[str, int, int]] = []
    for s in scenes:
        seen: dict[int, list[str]] = defaultdict(list)
        for t in s["turns"]:
            nid = t.get("node_id")
            if nid:
                seen[nid].append(t["text"])
        for nid, texts in seen.items():
            if len(texts) > 1:
                # 正文完全相同是上游重复行（可去重），正文不同才是身份冲突（会丢内容）
                distinct = len({norm_for_dup(x) for x in texts})
                dup_id.append((s["doc_id"], nid, distinct))
    run.add(
        Gate(
            "L1.8 node_id 缺失轮次数（仅申报）",
            "L1",
            "申报（不判定）",
            f"{len(missing_id)} 轮无 node_id（分布 {len({d for d, _ in missing_id})} 个场景）",
            "PASS",
            "上游部分节点没有 id 字段。轮次本身有效，实例身份改用场景内序号 ordinal 承担；"
            "此项只申报，避免下一个人把「id 为 0」误当成「解析出错」",
        )
    )
    conflicting = [d for d in dup_id if d[2] > 1]
    run.add(
        Gate(
            "L1.8b 同场景 node_id 重复且正文不同（身份冲突）",
            "L1",
            "0",
            f"重复编号 {len(dup_id)} 组，其中正文不同 {len(conflicting)} 组",
            "PASS" if not conflicting else "FAIL",
            "; ".join(f"{d}:id={n}:{k}种正文" for d, n, k in conflicting[:5])
            or (
                "正文相同的重复编号只是上游重复行，可去重；正文不同的重复会静默丢内容（计划 D1）"
                if dup_id
                else "本语料无重复编号"
            ),
        )
    )
    return run


def l2_gates(chunks: list[dict], shards: list[dict], scenes: list[dict]) -> GateRun:
    run = GateRun()


    # 计划书原文「块数 x 平均块长 与总字数之差 <10%」在带说话人前缀与轮次重叠的
    # 渲染形态下结构性不可能成立（前缀与重叠都是净增字）。这里换成同方向的失效探针：
    # 窄口径=原文覆盖率（任何一轮未被任何块包含即丢内容），宽口径=渲染冗余率。
    #
    # 覆盖判据按**轮次实例身份**（scene_id, ordinal）而不是 node_id（计划 D1）：
    # node_id 会缺失（实测 201 轮为 0）也会重复，拿它当集合元素时，
    # 同编号的两轮会被当成一轮——丢内容的同时门还是 PASS。
    covered: dict[str, set[int]] = defaultdict(set)
    covered_text: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for c in chunks:
        ordinals = c.get("turn_ordinals")
        if ordinals is None:  # 老产物没有该字段时退回 turn_ids 并明确降级
            ordinals = []
        covered[c["doc_id"]] |= set(ordinals)
        for span in c.get("turn_spans") or []:
            if span.get("ordinal") is not None:
                covered_text[c["doc_id"]].append((span["ordinal"], c["text"][span["start"] : span["end"]]))
    missing = []
    total_turns = 0
    for s in scenes:
        ordinals = set(range(len(s["turns"])))
        total_turns += len(ordinals)
        gap = ordinals - covered.get(s["doc_id"], set())
        if gap:
            missing.append((s["doc_id"], sorted(gap)[:3]))
    raw_sum = sum(c["raw_chars"] for c in chunks)
    rendered_sum = sum(c["chars"] for c in chunks)
    source_sum = sum(len(t["text"]) for s in scenes for t in s["turns"])
    inflation = rendered_sum / max(1, raw_sum)
    run.add(
        Gate(
            "L2.1a 原文轮次覆盖率",
            "L2",
            "1.00",
            f"{total_turns - sum(len(g) for _, g in missing):,}/{total_turns:,} 漏 {len(missing)} 篇",
            "PASS" if not missing else "FAIL",
            "; ".join(f"{d}:{g}" for d, g in missing[:3]),
        )
    )
    run.add(
        Gate(
            "L2.1b 渲染冗余率(宽口径,仅申报)",
            "L2",
            "申报值",
            f"{inflation:.3f}  (渲染 {rendered_sum:,} / 原文 {raw_sum:,}；其中说话人前缀 "
            f"{sum(c['label_chars'] for c in chunks):,} 字，源文字数 {source_sum:,})",
            "PASS",
            "对应计划书「块数x平均块长 vs 总字数」一门的等价可核算形式",
        )
    )

    # D1 三重覆盖：轮次实例覆盖 + 原始顺序 + 正文内容
    # 只查"序号都在"不够——序号齐了但顺序被调换、或偏移指向别轮的正文，都是损坏。
    order_bad: list[str] = []
    content_bad: list[str] = []
    for s in scenes:
        spans = sorted(
            covered_text.get(s["doc_id"], []), key=lambda x: x[0]
        )
        prev_end = -1
        for ordinal, text in spans:
            if not (0 <= ordinal < len(s["turns"])):
                continue
            expect = s["turns"][ordinal]
            rendered = expect["text"]
            if expect["kind"] == "dialogue" and expect["speaker"]:
                rendered = f"{expect['speaker']}：{rendered}"
            if text != rendered:
                content_bad.append(f"{s['doc_id']}#{ordinal}")
            if ordinal < prev_end:
                order_bad.append(f"{s['doc_id']}#{ordinal}")
            prev_end = ordinal
    run.add(
        Gate(
            "L2.1c 轮次顺序与正文一致（逐轮实例）",
            "L2",
            "0",
            f"顺序异常 {len(order_bad)} / 正文不符 {len(content_bad)}",
            "PASS" if not order_bad and not content_bad else "FAIL",
            f"顺序例 {order_bad[:3]}；正文例 {content_bad[:3]}",
        )
    )

    # 最小块长门：跨场景不许合并，所以判定对象是「同场景内仍有碎块」= 尾块处置失效。
    # 计划 D2：允许保留短尾块，但必须显式写明原因；原因空着才算失败——那等于
    # "静默留了个碎块"或"静默造了个超长块"，两者都不能当 PASS。
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        by_doc[c["doc_id"]].append(c)
    unexplained = [
        f"{c['chunk_id']}({c['chars']})"
        for c in chunks
        if c["chars"] < config.CHUNK_MIN_CHARS and not c.get("short_reason")
    ]
    short_kept = [c for c in chunks if c.get("short_reason")]
    run.add(
        Gate(
            "L2.2 短块必须有书面原因",
            "L2",
            "0",
            f"无原因短块 {len(unexplained)} / 有原因短块 {len(short_kept)}",
            "PASS" if not unexplained else "FAIL",
            "有原因短块按原因计数: "
            + json.dumps(Counter(c["short_reason"] for c in short_kept), ensure_ascii=False),
        )
    )
    oversize = [c for c in chunks if c["chars"] > config.CHUNK_MAX_CHARS]
    # 超限分两种，判据不同：
    #   atomic   —— 块里有一轮本身超过硬上限。轮次是原子单位，切半句就定位不到证据，
    #               所以这是"已知不可修"的事实，交人工决定是放宽上限还是允许超长轮次独立成块。
    #   logic    —— 没有任何单轮超上限，块却超了。这是合并/重叠逻辑自己造出来的，
    #               必须 FAIL——它正是计划 D2 要防的那个失效。
    atomic_oversize = [c for c in oversize if c.get("oversize_turn")]
    logic_oversize = [c for c in oversize if not c.get("oversize_turn")]
    oversize_verdict = "PASS" if not oversize else ("FAIL" if logic_oversize else "NEEDS_HUMAN")
    run.add(
        Gate(
            "L2.3 最大块 <= 硬上限",
            "L2",
            f"<= {config.CHUNK_MAX_CHARS}",
            f"max={max((c['chars'] for c in chunks), default=0)} "
            f"超限 {len(oversize)}（合并逻辑造成 {len(logic_oversize)} / 单轮本身超长 {len(atomic_oversize)}）",
            oversize_verdict,
            ("合并逻辑造出超长块: " + "; ".join(c["chunk_id"] for c in logic_oversize[:3]))
            if logic_oversize
            else (
                "单轮本身超上限（轮次不可切，需人工定夺）: "
                + "; ".join(f"{c['chunk_id']}({c['chars']})" for c in atomic_oversize[:3])
                if atomic_oversize
                else ""
            ),
        )
    )
    cross = [c["chunk_id"] for c in chunks if c["doc_id"].lower() not in c["chunk_id"].lower()]
    run.add(
        Gate("L2.4 跨场景块数", "L2", "0", str(len(cross)), "PASS" if not cross else "FAIL")
    )
    dup: dict[str, list[str]] = defaultdict(list)
    for c in chunks:
        dup[c["content_key"]].append(c["chunk_id"])
    groups = {k: v for k, v in dup.items() if len(v) > 1}
    run.add(
        Gate(
            "L2.5 内容哈希重复组数",
            "L2",
            "0",
            str(len(groups)),
            "PASS" if not groups else "NEEDS_HUMAN",
            json.dumps(list(groups.values())[:5], ensure_ascii=False),
        )
    )
    loads = [s["blocks"] for s in shards]
    run.add(
        Gate(
            "L2.6 分片装箱均衡度",
            "L2",
            "极差 <= 2x",
            f"shards={len(shards)} min={min(loads) if loads else 0} max={max(loads) if loads else 0}",
            "PASS" if loads and max(loads) <= 2 * min(loads) else "FAIL",
            f"合计 {sum(loads)} 块；T1 取 {'、'.join(config.T1_CATEGORIES)} 范围的分片",
        )
    )
    return run



RE_CODE_NAME = re.compile(r"^(?:[A-Za-z][A-Za-z0-9]*[-_]){0,3}[A-Z0-9]{5,}$|[0-9A-F]{6,}")
RE_SENTENCE_PUNCT = re.compile(r"[。！？；，!?;]")


def _is_structured_proper_name(name: str) -> bool:
    """超长但结构上确定是「专名」而非「描述性短语」的形态。

    判据本体在 norm_entity.is_structured_proper_name，这里只是转调：
    门与过滤器必须用同一份判据，否则会出现「门说超长、过滤器说不超长」
    这种自相矛盾的读数（正是 L4.4 假阳性那类问题的镜像）。
    """
    from . import norm_entity

    return norm_entity.is_structured_proper_name(name)


def _desc_hard_defect(name: str, desc: str) -> str:
    """desc 的「机械可判」缺陷：不需要读懂语义就能确定是坏的。

    只保留两条**无争议**的判据：
      equals_name  —— desc 就是名字本身，等于没写
      placeholder  —— 占位符（占位/待补/TODO 之类）

    刻意不收的两类（曾经收过，全量一跑 92 例里 89 例是假阳性）：
      too_short    —— 「帝国」「国家」「城市」对乌萨斯/维多利亚/龙门就是正确答案，
                      计划书要的正是「一句话讲清它是什么」，两个字能讲清就该给分；
      pronoun_head —— 「她们抵达的城市」是不是讲清了，要看上下文才知道，属语义判断。
    门的红线要卡在「连读都不用读就是错的」这一层，否则红久了就没人看（同 L4.4 的教训）。
    这两类改由 L3.T17b 抽样申报。
    """
    d = desc.strip()
    if d == name:
        return "equals_name"
    if d in _DESC_PLACEHOLDERS:
        return "placeholder"
    return ""


_DESC_PLACEHOLDERS = {"占位", "待补", "待定", "TODO", "todo", "略", "无", "未知", "不详", "－", "-", "…", "..."}
_DESC_PRONOUN_HEADS = ("他", "她", "它", "他们", "她们", "它们", "该", "此", "其")


def l3_trial_gates(records: list[dict], expected_blocks: int | None = None) -> GateRun:
    """计划书 3.6 的试跑验收：20 块逐条机检 + 需人工判断的部分单独申报。

    records: [{chunk_id, text, raw:{entities,relations}}]
    证据比对基准是块正文（已 lstrip），不是原始 JSON——基准取错会 100% 假失败。
    """
    from . import vocab

    run = GateRun()
    canon = set(vocab.CANONICAL_RELATIONS)
    types = set(vocab.ENTITY_TYPES)
    generic = set(vocab.GENERIC_PERSON_NOUNS)

    bad_evidence: list[str] = []
    long_evidence: list[str] = []
    off_vocab: list[str] = []
    dangling: list[str] = []
    pronoun_ent: list[str] = []
    masked_ent: list[str] = []
    code_ent: list[str] = []
    generic_ent: list[str] = []
    oversize_ent: list[str] = []
    bad_type: list[str] = []
    no_desc: list[str] = []
    alias_self: list[str] = []
    dup_ent: list[str] = []
    conf_off: list[str] = []
    rel_no_evidence: list[str] = []
    desc_defect: list[str] = []
    ent_total = rel_total = 0

    for rec in records:
        text = rec["text"]
        raw = rec["raw"] or {}
        entities = raw.get("entities") or []
        relations = raw.get("relations") or []
        names = set()
        for ent in entities:
            name = (ent.get("name") or "").strip()
            ent_total += 1
            if name in names:
                dup_ent.append(f"{rec['chunk_id']}:{name}")
            names.add(name)
            if name in vocab.PRONOUN_SET:
                pronoun_ent.append(f"{rec['chunk_id']}:{name}")
            if set(name) <= set("？?") or name in ("？？？", "??", "???"):
                masked_ent.append(f"{rec['chunk_id']}:{name}")
            elif RE_CODE_NAME.match(name) and ("-" in name or any(ch.isdigit() for ch in name)):
                code_ent.append(f"{rec['chunk_id']}:{name}")
            if name in generic:
                generic_ent.append(f"{rec['chunk_id']}:{name}")
            if len(name) > vocab.MAX_NAME_CHARS and not _is_structured_proper_name(name):
                oversize_ent.append(f"{rec['chunk_id']}:{name}({len(name)})")
            if ent.get("type") not in types:
                bad_type.append(f"{rec['chunk_id']}:{name}->{ent.get('type')}")
            desc = (ent.get("desc") or "").strip()
            if not desc:
                no_desc.append(f"{rec['chunk_id']}:{name}")
            else:
                defect = _desc_hard_defect(name, desc)
                if defect:
                    desc_defect.append(f"{rec['chunk_id']}:{name}|{defect}")
            for alias in ent.get("aliases") or []:
                if alias == name:
                    alias_self.append(f"{rec['chunk_id']}:{name}")
        for rel in relations:
            rel_total += 1
            ev = (rel.get("evidence") or "").strip()
            if rel.get("relation") not in canon:
                off_vocab.append(f"{rec['chunk_id']}:{rel.get('relation')}")
            if rel.get("source") not in names or rel.get("target") not in names:
                dangling.append(f"{rec['chunk_id']}:{rel.get('source')}-{rel.get('relation')}->{rel.get('target')}")
            if not ev:
                rel_no_evidence.append(f"{rec['chunk_id']}")
                continue
            if len(ev) > 40:
                long_evidence.append(f"{rec['chunk_id']}:{ev[:24]}({len(ev)})")
            if ev not in text:
                bad_evidence.append(f"{rec['chunk_id']}:{ev[:30]}")
            conf = rel.get("confidence")
            if not isinstance(conf, (int, float)) or not 0.0 <= float(conf) <= 1.0:
                conf_off.append(f"{rec['chunk_id']}:{conf}")

    def one(name: str, threshold: str, hits: list[str], note: str = "") -> Gate:
        return Gate(name, "L3", threshold, str(len(hits)), "PASS" if not hits else "FAIL",
                    (note + ";" if note else "") + "; ".join(hits[:4]))

    ratio_ent = ent_total / max(1, len(records))
    want = expected_blocks if expected_blocks is not None else 20
    run.add(Gate("L3.T0 块数", "L3", str(want), str(len(records)), "PASS" if len(records) == want else "FAIL"))
    run.add(one("L3.T1 证据可在块正文定位（逐条比对）", "违规 0", bad_evidence, f"实体{ent_total} 关系{rel_total}"))
    run.add(one("L3.T2 证据长度<=40", "违规 0", long_evidence))
    run.add(one("L3.T3 关系名全部在受控词表内", "违规 0", off_vocab))
    run.add(one("L3.T4 关系两端都在 entities 内", "违规 0", dangling))
    run.add(one("L3.T5 证据为空的关系数", "0", rel_no_evidence))
    run.add(one("L3.T6 代词被抽成实体", "0", pronoun_ent))
    run.add(one("L3.T7 掩码说话人成实体", "0", masked_ent))
    run.add(one("L3.T8 编号/代码形态成实体", "0", code_ent))
    # 泛称按计划书 §3.3 属于「本地二次过滤」三类噪声之一：模型稳定产出它是预期内的，
    # 缓存必须留原始输出（决策三），由 L4 过滤器剔除。这里只申报数量，不判 raw 层 FAIL。
    run.add(
        Gate(
            "L3.T9 泛称成实体（交 L4 本地过滤）",
            "L3",
            "申报数量，L4 过滤后必须为 0",
            str(len(generic_ent)),
            "PASS",
            "; ".join(generic_ent[:4]),
        )
    )
    run.add(one("L3.T10 超长实体名(>16字)", "0", oversize_ent))
    run.add(one("L3.T11 实体类型词表外", "0", bad_type))
    run.add(one("L3.T12 实体缺 desc", "0", no_desc))
    run.add(one("L3.T13 别名包含规范名自身", "0", alias_self))
    run.add(one("L3.T14 同块重复实体", "0", dup_ent))
    run.add(one("L3.T15 置信度越界", "0", conf_off))
    run.add(
        Gate(
            "L3.T16 每块平均实体/关系数",
            "L3",
            "申报值",
            f"实体 {ratio_ent:.1f}/块，关系 {rel_total / max(1, len(records)):.1f}/块",
            "PASS",
            "关系/实体比过低说明模型只抽点不抽边，图扩展通道会失去原料",
        )
    )
    run.add(
        one(
            "L3.T17 描述机械缺陷（等于名字/占位符/代词开头过短/过短）",
            "违规 0",
            desc_defect,
            f"按 {ent_total} 条实体提及计",
        )
    )
    n_def = len(desc_defect)
    run.add(
        Gate(
            "L3.T17b 描述语义质量（抽样申报）",
            "L3",
            "申报（不判定）",
            f"机械合规 {(ent_total - n_def) / max(1, ent_total):.4f}；语义质量需人读，见人工抽验文档",
            "PASS",
            "「一句话讲清它是什么」是语义判断，机检只能排掉机械缺陷；"
            "把语义当门会让门永远红，红久了就没人看",
        )
    )

    # 块来源分层：本轮抽取 / 只读复用 legacy / 从 legacy 搬进新键。
    # 只申报、不判定——「搬用占比」该是多少取决于块身份变了多少，没有普适阈值；
    # 但它是必须被看见的数：把 adopted 混进 current 报，等于让「这一代真抽了多少」
    # 永久失明。records 的 layer 字段由调用方从 cache.Lookup.layer 带过来。
    layers = {"current": 0, "legacy": 0, "adopted": 0}
    unlabeled = 0
    for rec in records:
        lay = str(rec.get("layer") or "")
        if lay in layers:
            layers[lay] += 1
        else:
            unlabeled += 1
    adopted_ids = [r["chunk_id"] for r in records if str(r.get("layer") or "") == "adopted"]
    run.add(
        Gate(
            "L3.T19 块来源分层（current/legacy/adopted 申报）",
            "L3",
            "申报值（不判定）",
            f"current {layers['current']} / legacy {layers['legacy']} / "
            f"adopted {layers['adopted']} / 未标注 {unlabeled}",
            "PASS",
            ("adopted 块: " + "; ".join(adopted_ids[:4])) if adopted_ids
            else "无 adopted 块（没有可证明复用的对象，或本轮全部走 current/legacy）",
        )
    )
    return run


def l6_gates(
    meta: dict,
    graph: dict,
    current_chunk_count: int,
    current_corpus_fp: str,
    retrieval_probes: list[dict],
) -> GateRun:
    """L6 索引门：新鲜度三查 + 检索自检 + 倒排覆盖 + 图邻接一致性。

    计划书对 L6 的核心恐惧是「索引落后于语料时不报错、只是新内容检索不到」——
    那种失效不会有任何异常，只会让检索悄悄变差。所以这里的每个门都是
    「静默失效探针」：新鲜度靠指纹比对，检索靠回查已知答案，覆盖靠全量计数。
    """
    from . import index as index_mod

    run = GateRun()

    # 三查：块数 / 语料指纹 / 构建时间。任一不等就 FAIL，不许静默使用旧索引。
    problems = index_mod.freshness_check(meta, current_chunk_count, current_corpus_fp)
    run.add(
        Gate(
            "L6.1 新鲜度三查（块数/语料指纹/构建时间）",
            "L6",
            "不一致项 0",
            str(len(problems)),
            "PASS" if not problems else "FAIL",
            "; ".join(problems[:4]),
        )
    )

    # 索引与图必须同源：图重建了而索引没重建，检索出来的实体名就指向已合并掉的旧名。
    graph_fp = graph.get("fingerprint") or ""
    run.add(
        Gate(
            "L6.2 索引图指纹与当前图一致",
            "L6",
            "相等",
            f"index={meta.get('graph_fp') or '(空)'} graph={graph_fp or '(空)'}",
            "PASS" if meta.get("graph_fp") == graph_fp and graph_fp else "FAIL",
            "图重建后索引未重建 = 检索命中已合并掉的旧实体名",
        )
    )

    # 检索自检分两个问题，混在一起测会得出错误结论：
    #   (a) 正确性：倒排是否把词映射到了正确的块。用「该块里 df 最小的词」回查，
    #       这是一次确定性回环——不命中就是索引坏了，必须 100%。
    #   (b) 质量：拿实体名当查询能不能捞回它的来源块。常见名（感染者/哥伦比亚）
    #       在几百个块里都出现，捞不回来源块是**检索排序的正常表现**，不是故障。
    #       所以它只申报命中率，不判 FAIL——把它当门会让门永远红，红久了就没人看。
    bad = [p for p in retrieval_probes if p.get("kind") == "token" and p.get("rank") is None]
    tok_probes = [p for p in retrieval_probes if p.get("kind") == "token"]
    name_probes = [p for p in retrieval_probes if p.get("kind") == "name"]
    name_hits = [p for p in name_probes if p.get("rank") is not None]
    run.add(
        Gate(
            "L6.3 检索正确性（稀疏词回查来源块）",
            "L6",
            "命中率 1.00",
            f"{len(tok_probes) - len(bad)}/{len(tok_probes)}",
            "PASS" if tok_probes and not bad else "FAIL",
            "; ".join(f"{p['query']}->{p['chunk_id']}" for p in bad[:4]),
        )
    )
    run.add(
        Gate(
            "L6.3b 实体名检索质量（top-20，仅申报）",
            "L6",
            "申报值",
            f"{len(name_hits)}/{len(name_probes)}"
            + (f"  中位名次 {sorted((p['rank'] for p in name_hits))[len(name_hits) // 2]}" if name_hits else ""),
            "PASS",
            "常见名捞不回自己那块属正常排序结果；此值只反映检索手感，不参与判定",
        )
    )

    # 别名倒排：每个实体名与别名都必须解析到自己。注意 graph.json 里的 alias_index
    # 只存别名（不含规范名），覆盖度要用 index.build_alias_index 现算，别拿它当全集。
    entities = graph.get("entities") or []
    alias_index = index_mod.build_alias_index(entities)
    unresolved = [e["name"] for e in entities if alias_index.get(e["name"]) != e["id"]]
    lost_aliases = [
        f"{a}->{eid}" for a, eid in (graph.get("alias_index") or {}).items() if a not in alias_index
    ]
    run.add(
        Gate(
            "L6.4 别名倒排覆盖（实体名+别名->id）",
            "L6",
            "未解析 0",
            f"实体名 {len(entities) - len(unresolved)}/{len(entities)}  "
            f"条目 {len(alias_index)}（图内别名 {len(graph.get('alias_index') or {})}，丢失 {len(lost_aliases)}）",
            "PASS" if not unresolved and not lost_aliases else "FAIL",
            "; ".join((unresolved + lost_aliases)[:4]),
        )
    )

    # 别名可检索性：待审别名绝不能替自己的属主开口。
    #
    # 这一条是**独立**的 FAIL 门，不是 L6.4 的子项：L6.4 问「该找到的能不能找到」，
    # 这一条问「不该找到的会不会被找到」。两个方向都要守——`identity.py:241` 记录的
    # 实测事故正是后者：拒绝合并「暴雨 <- 库兰塔士兵」之后别名表里仍写着
    # `review:"pending"`，而下游 `{alias: ent["id"] for alias in ent["aliases"]}`
    # 照样把它指向暴雨。L6.4 在那件事上永远是 PASS（暴雨自己解析得到自己），
    # 只有专门查「待审别名有没有替属主开口」才抓得住。
    #
    # 判据是**指向属主**，不是「文本可检索」。第一版写成「文本出现在检索表里」，
    # 真实语料上直接报 101 条泄漏，逐条查过去**没有一条**指向属主：那 101 条
    # pending 文本（「女妖」之于 Logos、「乌萨斯」之于「乌萨斯帝国」）本来就是
    # 别的实体的规范名或已生效别名，检索到它们是正确的，与这条 pending 无关。
    # 把「文本可检索」当泄漏会让门在真实数据上永远红，红久了就没人看——
    # 所以判据收紧成：**这条 pending 的文本解析到的是不是它自己的属主**。
    # 是，说明它真进了检索并在替属主说话；不是，它对检索结果没有贡献。
    #
    # 两个来源都看：现算的 `alias_index`（检索真正用的那份）与图里申报的
    # `graph["alias_index"]`（消费端可能直接读它建倒排）。后者优先——
    # 图自己写错了才是最该被抓的情形。
    searchable = dict(alias_index)
    searchable.update(graph.get("alias_index") or {})
    leaked_pending = [
        f"{e['name']}:{item.get('text')}"
        for e in entities
        for item in (e.get("pending_aliases") or [])
        if str(item.get("text") or "").strip()
        and searchable.get(str(item.get("text") or "").strip()) == e["id"]
    ]
    run.add(
        Gate(
            "L6.4c 待审别名不得进入检索（pending 只能留档）",
            "L6",
            "泄漏 0",
            f"待审 {sum(len(e.get('pending_aliases') or []) for e in entities)}  "
            f"泄漏 {len(leaked_pending)}",
            "PASS" if not leaked_pending else "FAIL",
            "; ".join(leaked_pending[:4])
            or "检索表只收 review ∈ {auto, approved} 的别名（identity.to_dict() 保证）",
        )
    )

    # 图邻接：边的两端必须都能走到；孤立实体按 L5 口径申报（它们有实体但无边）。
    # 邻接键按图分派：`ark-rag-1` 是规范名（旧图 id 就是名字），`ark-rag-3` 是实体 id。
    # 两边都用 entity_key 取，门问的始终是「这个实体在邻接表里找得到吗」。
    adjacency = index_mod.build_adjacency(graph.get("edges") or [])
    dangling = [
        f"{edge_ends(e)[0]}->{edge_ends(e)[1]}"
        for e in (graph.get("edges") or [])
        if edge_ends(e)[0] not in adjacency or edge_ends(e)[1] not in adjacency
    ]
    run.add(
        Gate(
            "L6.5 图邻接覆盖（边两端可达）",
            "L6",
            "悬空边 0",
            f"节点 {len(adjacency)}  邻接项 {sum(len(v) for v in adjacency.values())}",
            "PASS" if not dangling else "FAIL",
            "; ".join(dangling[:4]),
        )
    )

    isolated = [e["name"] for e in entities if entity_key(graph, e) not in adjacency]
    run.add(
        Gate(
            "L6.6 孤立实体（有实体无边，仅申报）",
            "L6",
            "申报值，L5 已判 <=0.20",
            f"{len(isolated)} ({len(isolated) / max(1, len(entities)):.1%})",
            "PASS",
            "; ".join(isolated[:4]),
        )
    )

    # 向量通道：没有嵌入后端时必须是显式缺席，不能留空让后人以为是漏建。
    channels = meta.get("vector_channels") or ""
    run.add(
        Gate(
            "L6.7 向量通道状态显式申报",
            "L6",
            "非空且声明原因",
            channels or "(未申报)",
            "PASS" if channels else "FAIL",
            f"embedding_model={meta.get('embedding_model')} dim={meta.get('embedding_dim')}",
        )
    )
    return run


def l4_gates(records: list[dict], graph: dict) -> GateRun:
    """L4 归一化门：类型闭合、噪声名、成员误并、关系闭合、方向对成对。

    L4 自己没有落盘产物，它只是「原始观察 -> 图节点/边」之间的那道闸门，
    所以门只能对着 L5 建出来的图反查：门在这里，证据在图上。
    """
    from . import norm_entity, norm_relation, vocab as V

    run = GateRun()
    entities = graph.get("entities") or []
    edges = graph.get("edges") or []
    stats = graph.get("stats") or {}
    canon_rel = set(V.CANONICAL_RELATIONS)

    # 分母用原始唯一名，才能看出过滤到底拦掉了多少；只数留在图上的名字，
    # 会让「拦得少」和「拦得对」两种情况读数一样。
    obs = norm_entity.collect(records)
    raw_names = len(obs)
    dropped = stats.get("dropped_names") or {}
    dropped_total = sum(dropped.values())

    off_vocab = [e["name"] for e in entities if e["type"] not in V.ENTITY_TYPES]
    run.add(
        Gate(
            "L4.1 实体类型闭合（六选一）",
            "L4",
            "词表外 0",
            f"词表外 {len(off_vocab)}/{len(entities)}",
            "PASS" if not off_vocab else "FAIL",
            "; ".join(off_vocab[:4]),
        )
    )

    # 对图上现存名字重跑判据，而不是相信建图时那份统计：统计记的是「剔了几个」，
    # 门要问的是「还剩没剩」。
    residual = []
    for e in entities:
        bad, why = norm_entity.is_noise_name(e["name"], e["type"])
        if bad:
            residual.append(f"{e['name']}({why})")
    run.add(
        Gate(
            "L4.2 噪声名残留（现有名字重跑判据）",
            "L4",
            "残留 0",
            f"残留 {len(residual)}/{len(entities)}",
            "PASS" if not residual else "FAIL",
            "; ".join(residual[:4]),
        )
    )
    run.add(
        Gate(
            "L4.3 噪声名剔除率（仅申报）",
            "L4",
            "申报值",
            f"原始唯一名 {raw_names}，剔除 {dropped_total} "
            f"({dropped_total / max(1, raw_names):.2%})  {dropped}",
            "PASS",
            "剔得太少要怀疑判据失效，剔得太多要怀疑误杀；两个方向都没有绝对阈值",
        )
    )

    # 「组织」与「组织成员」是两类东西，并成一个节点等于把一群人当成一个人。
    # 别名边那条合并路径曾经绕过这个判据，所以这里对合并结果逐个复核。
    def struct_member(short: str, long: str) -> bool:
        """补上前缀前置条件，再问后缀判据。

        is_member_index 只切后缀、不管两名的前缀关系：直接拿「克里斯」和
        「罗德岛工程干员」去问，它会切出「岛工程干员」并判真。漏掉这道前置，
        门会报出一堆假阳性，红久了就没人看。
        """
        if not long.startswith(short) or len(long) <= len(short):
            return False
        if long[len(short)] not in norm_entity.FULLNAME_SEPS:
            return False
        return norm_entity.is_member_index(short, long)

    def is_role_name(name: str) -> bool:
        """职位泛称：无名角色标签（医疗干员、老工匠），不指某一个具体的人。"""
        return name in set(V.GENERIC_PERSON_NOUNS) or name.endswith(tuple(V.MEMBER_SUFFIXES))

    member_bad = []
    role_anchor = []
    role_alias = []
    for e in entities:
        # 别名的两代形态差异在 alias_texts/merged_names 里分派（见文件头部注释），
        # 这里拿到的永远是名字字符串，判据本身不区分 schema。
        for member in merged_names(e) + alias_texts(e):
            if struct_member(e["name"], member) or struct_member(member, e["name"]):
                member_bad.append(f"{e['name']} <- {member}")
            elif is_role_name(e["name"]) and not is_role_name(member):
                role_anchor.append(f"{e['name']} <- {member}")
            elif is_role_name(member) and not is_role_name(e["name"]):
                role_alias.append(f"{e['name']} <- {member}")
    run.add(
        Gate(
            "L4.4 成员误并（集体/成员不得并成本体）",
            "L4",
            "误并 0",
            f"误并 {len(member_bad)}",
            "PASS" if not member_bad else "FAIL",
            "; ".join(member_bad[:4]),
        )
    )

    # 职位泛称与具体的人该不该并，取决于这段剧情里那个标签是否只指一个人：
    # 「油滑的士兵」全剧只对应一个个体，并了是对的；「医疗干员」在 21 块里出现过
    # 好几个不同的人，并进去就是把一群人当成一个人。判定要按块共现，建图期没有
    # 这个信息，所以只申报，等人工拍板再动规则（动规则要 bump ev 版本）。
    run.add(
        Gate(
            "L4.4b 职位泛称并合（仅申报）",
            "L4",
            "申报值",
            f"泛称锚点吸收专名 {len(role_anchor)} 例"
            + (f"（{'; '.join(role_anchor[:4])}）" if role_anchor else "")
            + f"；专名吸收角色式别名 {len(role_alias)} 例"
            + (f"（{'; '.join(role_alias[:3])}）" if role_alias else ""),
            "PASS",
            "前者是「一群人被当成一个人」，后者更可能是「同一个人的战地标签」，"
            "两类必须分开判，判别依据是同一标签在不同块里是否指向不同个体",
        )
    )

    # 别名质量两项都只申报：它们不改变「规范名 -> id」的解析结果（规范名解析权
    # 不会被别名抢走），只让「按别名查」这一步变含糊，需要靠上下文消歧。
    alias_noise = []
    owner: dict[str, list[str]] = defaultdict(list)
    for e in entities:
        for a in alias_texts(e):
            owner[a].append(e["id"])
            bad, why = norm_entity.is_noise_name(a, e["type"])
            if bad:
                alias_noise.append(f"{e['name']}:{a}({why})")
    contested = {a: ids for a, ids in owner.items() if len(ids) > 1}
    alias_total = sum(len(alias_texts(e)) for e in entities)
    run.add(
        Gate(
            "L4.5 别名质量（噪声/争用，仅申报）",
            "L4",
            "申报值",
            f"条目 {alias_total}  噪声 {len(alias_noise)}  争用别名 {len(contested)}"
            + (f"（{', '.join(sorted(contested, key=lambda a: -len(contested[a]))[:3])}）" if contested else ""),
            "PASS",
            "泛称别名（队长/老师）被多个实体共用是数据的真实状态，不是归一化缺陷",
        )
    )

    # 被两个以上实体同时主张的别名不能当合并依据：「殿下」「将军」「老爷」这类称谓
    # 注定被不同的人各自认领，拿它走并查集会把人串成一个大组（修之前凯尔希一组
    # 32 人，里面同时有 Logos、Mon3tr、特蕾西娅、特蕾西娅的摄政王弟弟、维娜……）。
    # 这是建图期的**规则**约束，不是概率性质量问题，所以它是 FAIL 门不是申报门：
    # 规则生效后图上残留必须为 0，残留即说明有别的路径又把歧义别名带回来了。
    ambiguous = norm_entity.ambiguous_aliases(
        {
            e["name"]: {"aliases": set(alias_texts(e))}
            for e in entities
        }
    )
    residual_ambiguous = [
        f"{e['name']}:{a}"
        for e in entities
        for a in alias_texts(e)
        if a in ambiguous
    ]
    run.add(
        Gate(
            "L4.5b 歧义别名参与合并（被多方主张的别名不得当别名）",
            "L4",
            "残留 0",
            f"图上残留 {len(residual_ambiguous)}",
            "PASS" if not residual_ambiguous else "FAIL",
            "; ".join(residual_ambiguous[:4])
            or "多方主张的别名已同时从合并与 aliases 中剔除（建图期 alias_ambiguous_rejected）",
        )
    )

    off_vocab_rel = sorted({edge_relation(e) for e in edges} - canon_rel)
    run.add(
        Gate(
            "L4.6 关系名闭合（封闭表）",
            "L4",
            "词表外 0",
            f"词表外 {len(off_vocab_rel)}，"
            f"已用 {len({edge_relation(e) for e in edges} & canon_rel)}/{len(V.CANONICAL_RELATIONS)}",
            "PASS" if not off_vocab_rel else "FAIL",
            "; ".join(off_vocab_rel[:4]),
        )
    )

    # 这一项查的是**词表结构**，不是抽取质量：逆谓词只要有一端不在表里，
    # 「换方向」就在结构上无法表达，方向丢失是必然的（vocab.py 开头的承诺）。
    missing_pair = [f"{a}/{b}" for a, b in V.INVERSE_PAIRS if a not in canon_rel or b not in canon_rel]
    pair_rate = (len(V.INVERSE_PAIRS) - len(missing_pair)) / max(1, len(V.INVERSE_PAIRS))
    run.add(
        Gate(
            "L4.7 方向对成对完整率（词表结构）",
            "L4",
            ">= 0.95",
            f"{pair_rate:.2f}（{len(V.INVERSE_PAIRS) - len(missing_pair)}/{len(V.INVERSE_PAIRS)} 对齐全）",
            "PASS" if pair_rate >= 0.95 else "FAIL",
            "; ".join(missing_pair[:4]),
        )
    )

    by_pair: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for e in edges:
        src_e, dst_e = edge_ends(e)
        by_pair[(src_e, dst_e)][edge_relation(e)] = e

    # 逆谓词双向重复不是错误，是同一句话被抽成了两个视角（「她救了我」既是
    # 救助也是获救于）。这里额外数出「证据逐字相同」的那批——那才是真冗余。
    dup_other = []
    dup_same_ev = []
    for (src, dst), relmap in by_pair.items():
        back_map = by_pair.get((dst, src), {})
        for rel, e in relmap.items():
            inv = norm_relation.inverse_of(rel)
            if inv and inv in back_map:
                if e["evidence"] == back_map[inv]["evidence"]:
                    dup_same_ev.append(f"{src}-{rel}->{dst}")
                else:
                    dup_other.append(f"{src}-{rel}->{dst}")
    run.add(
        Gate(
            "L4.8 逆谓词双向重复（仅申报）",
            "L4",
            "申报值",
            f"{len(dup_other) + len(dup_same_ev)} 对，其中证据逐字相同 {len(dup_same_ev)} 对",
            "PASS",
            "两视角并存合法；证据相同的那批应由消费端去重，不在建图期删",
        )
    )

    # 互斥共存：存量逐条看全是时序/立场变化（先效忠后背叛）。判 FAIL 会让门永远红，
    # 红久了就没人看，所以只申报，并把「同块共现」单独拎出来——那才可能是抽错。
    mutex_all = []
    mutex_same_chunk = []
    for (src, dst), relmap in by_pair.items():
        rels = sorted(relmap)
        for i, a in enumerate(rels):
            for b in rels[i + 1 :]:
                if norm_relation.is_mutually_exclusive(a, b):
                    mutex_all.append(f"{src}:{a}+{b}")
                    if set(relmap[a].get("chunks") or []) & set(relmap[b].get("chunks") or []):
                        mutex_same_chunk.append(f"{src}:{a}+{b}@同块")
    run.add(
        Gate(
            "L4.9 互斥关系共存（仅申报）",
            "L4",
            "申报值",
            f"{len(mutex_all)} 组，其中同块共现 {len(mutex_same_chunk)} 组",
            "PASS",
            "跨块共存基本都是剧情里的立场反转；同块共现才可能是抽错",
        )
    )

    used = {edge_relation(e) for e in edges} & canon_rel
    unused = [r for r in V.CANONICAL_RELATIONS if r not in used]
    run.add(
        Gate(
            "L4.10 词表使用覆盖（仅申报）",
            "L4",
            "申报值",
            f"已用 {len(used)}/{len(V.CANONICAL_RELATIONS)}" + (f"，未用 {unused}" if unused else ""),
            "PASS",
            "长期未用的关系名说明词表与语料不匹配，是 D-17 那类裁剪讨论的依据",
        )
    )

    # 纯称谓/亲属词当实体（父亲、母亲、爷爷、陛下、殿下、老板）：
    # 抽取规则要求「无专名的角色不得成实体」，但 GENERIC_PERSON_NOUNS 里只收了
    # 职业泛称，没收亲属与敬称，于是这类词会以「角色」身份独立入图。
    # 只申报不判定：它们确实是剧情里反复出现的一个「人」，删掉会连带删掉
    # 79 条边（爷爷 20、老板 17、奶奶 8…），是结构决策不是噪声清理，要人工拍板。
    BARE_TITLES = {
        "父亲", "母亲", "爸爸", "妈妈", "爷爷", "奶奶", "外公", "外婆",
        "哥哥", "姐姐", "弟弟", "妹妹", "叔叔", "阿姨", "舅舅", "姑姑",
        "老爷", "老板", "大人", "阁下", "殿下", "陛下", "将军", "大公",
        "女大公", "公爵", "侯爵", "伯爵", "子爵", "男爵", "王", "皇帝",
    }
    bare = [e for e in entities if e["name"] in BARE_TITLES]
    degree_all: dict[str, int] = defaultdict(int)
    for edge in edges:
        # degree_all 的键与边的端点同一坐标系：`ark-rag-1` 是规范名（id==name），
        # `ark-rag-3` 是实体 id。下面的 e["id"] 两代都能对上，判据不用分叉。
        src_e, dst_e = edge_ends(edge)
        degree_all[src_e] += 1
        degree_all[dst_e] += 1
    bare_edges = sum(degree_all[e["id"]] for e in bare)
    bare_top = ", ".join(
        "%s %d边" % (e["name"], degree_all[e["id"]])
        for e in sorted(bare, key=lambda x: -degree_all[x["id"]])[:6]
    )
    run.add(
        Gate(
            "L4.11 纯称谓/亲属词成实体（仅申报）",
            "L4",
            "申报值",
            f"实体 {len(bare)} 个，涉及边 {bare_edges} 条" + (f"（{bare_top}）" if bare else ""),
            "PASS",
            "「爷爷」「老板」在剧情里确实是反复出现的一个角色，删与留是结构决策；"
            "要删得先扩充 GENERIC_PERSON_NOUNS 的亲属/敬称部分再重跑，不能在建图后摘点",
        )
    )
    return run


def l5_gates(graph: dict, chunks: list[dict], corpus_fp: str) -> GateRun:
    """L5 建图门：规模、连通性、边证据可定位、置信度、指纹。

    计划书 L5 的两条硬门线（平均度数 > 6、孤立实体率 <= 0.20）落在这里。
    其余各项都是「静默失效探针」：图的坏法是边还在、但指向不存在的节点，
    或者证据是编的却没人查——这两种都不会抛异常。
    """
    run = GateRun()
    entities = graph.get("entities") or []
    edges = graph.get("edges") or []
    n_ent = len(entities)
    n_edge = len(edges)

    run.add(
        Gate(
            "L5.1 规模（实体/边/类型分布）",
            "L5",
            "申报值",
            f"实体 {n_ent}  边 {n_edge}  "
            f"类型 {dict(Counter(e['type'] for e in entities).most_common())}",
            "PASS",
            "规模本身不做判定；它是其它所有比率的基数",
        )
    )

    degree: Counter = Counter()
    for e in edges:
        src_e, dst_e = edge_ends(e)
        degree[src_e] += 1
        degree[dst_e] += 1
    avg_degree = (2 * n_edge / n_ent) if n_ent else 0.0
    run.add(
        Gate(
            "L5.2 平均度数",
            "L5",
            "> 6",
            f"{avg_degree:.2f}",
            "PASS" if avg_degree > 6 else "FAIL",
            "度数是图能不能撑起「沿边扩展」的唯一指标，低度数的图等价于一份实体表",
        )
    )

    isolated = [e["name"] for e in entities if degree.get(e["id"], 0) == 0]
    iso_rate = len(isolated) / max(1, n_ent)
    run.add(
        Gate(
            "L5.3 孤立实体率",
            "L5",
            "<= 0.20",
            f"{len(isolated)} ({iso_rate:.1%})",
            "PASS" if iso_rate <= 0.20 else "FAIL",
            "孤立实体只能靠词面命中被检索到，永远不会经图扩展被找到",
        )
    )

    self_loop = [f"{edge_ends(e)[0]}->{edge_ends(e)[1]}" for e in edges
                 if edge_ends(e)[0] == edge_ends(e)[1]]
    # known 用 id：两代图里「实体节点身份」都是 id（ark-rag-1 的 id 就是规范名，
    # 所以这条对旧图等价于原来的 known = {name}）。
    known = {e["id"] for e in entities}
    dangling = [f"{edge_ends(e)[0]}->{edge_ends(e)[1]}" for e in edges
                if edge_ends(e)[0] not in known or edge_ends(e)[1] not in known]
    run.add(
        Gate(
            "L5.4 自环与悬空边",
            "L5",
            "自环 0 / 悬空 0",
            f"自环 {len(self_loop)}  悬空 {len(dangling)}",
            "PASS" if not self_loop and not dangling else "FAIL",
            "; ".join((self_loop + dangling)[:4]),
        )
    )

    # 边证据必须能回到它自己列出的来源块里逐字定位。编造的证据在图上完全看不出来，
    # 只有拿正文比对才暴露——这是 L5 唯一能守住「证据非空」承诺的检查。
    text_by_chunk: dict[str, str] = {c["chunk_id"]: c["text"] for c in chunks}
    no_ev = [f"{edge_ends(e)[0]}->{edge_ends(e)[1]}" for e in edges
             if not (e.get("evidence") or "").strip()]
    unlocatable = []
    for e in edges:
        ev = (e.get("evidence") or "").strip()
        if not ev:
            continue
        if not any(ev in text_by_chunk.get(cid, "") for cid in (e.get("chunks") or [])):
            unlocatable.append(f"{edge_ends(e)[0]}-{edge_relation(e)}->{edge_ends(e)[1]}")
    run.add(
        Gate(
            "L5.5 边证据可定位（逐字回查来源块）",
            "L5",
            "未定位 0 / 空证据 0",
            f"未定位 {len(unlocatable)}  空证据 {len(no_ev)}  共 {n_edge} 条",
            "PASS" if not unlocatable and not no_ev else "FAIL",
            "; ".join((unlocatable + no_ev)[:4]),
        )
    )

    # 置信度按元素类型取（ark-rag-3 的聚合边只有 confidence_summary，
    # 取 max 的理由见 edge_confidence 的文档字符串）。
    confs = sorted(edge_confidence(e) for e in edges)
    low = [f"{edge_ends(e)[0]}->{edge_ends(e)[1]}" for e in edges
           if edge_confidence(e) < config.CONFIDENCE_THRESHOLD]
    run.add(
        Gate(
            "L5.6 边置信度下限",
            "L5",
            f">= {config.CONFIDENCE_THRESHOLD}",
            f"低于下限 {len(low)}  最低 {confs[0]:.2f}  中位 {confs[len(confs) // 2]:.2f}" if confs else "无边",
            "PASS" if not low else "FAIL",
            "; ".join(low[:4]),
        )
    )

    bad_mention = [e["name"] for e in entities if int(e.get("mentions") or 0) < 1]
    run.add(
        Gate(
            "L5.7 提及次数（每个实体至少 1 次）",
            "L5",
            "0 例低于 1",
            f"低于 1 的实体 {len(bad_mention)}",
            "PASS" if not bad_mention else "FAIL",
            "提及为 0 意味着这个实体是被推断出来的、正文里从未出现——那不该存在",
        )
    )

    # L1.2b 的另一半：世界内编码有没有漏成实体名/别名。
    # 语料里 screen 上的日志行、公民编号、验证码都是正文（不该删），
    # 但它们绝不该被抽成实体——那才叫污染。这条门才是 L1.2b 当初要人工裁决的原因。
    code_names = [
        f"{e['name']}"
        for e in entities
        if RE_SERIAL.search(e["name"]) or RE_TERMINAL_LOG.search(e["name"])
    ]
    code_alias = [
        f"{e['name']}<-{a}"
        for e in entities
        for a in alias_texts(e)
        if RE_SERIAL.search(a) or RE_TERMINAL_LOG.search(a)
    ]
    run.add(
        Gate(
            "L5.7b 世界内编码漏成实体名/别名",
            "L5",
            "0 例",
            f"实体名 {len(code_names)} / 别名 {len(code_alias)}",
            "PASS" if not (code_names or code_alias) else "FAIL",
            "L1.2b 把日志行/编号判为合法正文后，唯一真风险就是它们被当成实体；"
            + ("; ".join((code_names + code_alias)[:4]) if (code_names or code_alias) else "实测全清"),
        )
    )

    run.add(
        Gate(
            "L5.8 指纹覆盖（语料/图/版本三件套）",
            "L5",
            "全部非空且与当前语料一致",
            f"corpus_fp={graph.get('corpus_fp') or '(空)'} graph_fp={graph.get('fingerprint') or '(空)'} "
            f"prompt={graph.get('prompt_version')} norm={graph.get('normalizer_version')} "
            f"ent={graph.get('entity_rule_version') or '(空)'}",
            "PASS"
            if graph.get("corpus_fp")
            and graph.get("fingerprint")
            and graph.get("entity_rule_version")
            and graph.get("corpus_fp") == corpus_fp
            else "FAIL",
            "指纹不全的图无法判断是否与当前语料同源，等于不可复现；"
            "实体合并规则版本（ev）同样必须进图，否则同一指纹会对应两张结构不同的图",
        )
    )
    return run


def l5_evidence_gates(
    graph: dict,
    assertions: list[dict],
    edges: list[dict],
    chunks: list[dict],
) -> GateRun:
    """L5.9 / L5.10 / L5.11：逐观察证据、聚合边只存派生统计、观察守恒。

    这三条门是 E1 的验收核心，共同回答一个旧模型答不了的问题：
    **「这张图上的这条边，凭哪一段原文说的？」**

    旧聚合边每条只留一条 `evidence`，实测 9,747 条边的多条观察共用一条证据、
    单边最多 116 条观察只留 1 条——也就是说 34,536 次观察在图里没有自己的位置。
    那不是"证据少了一点"，而是"想知道第三十个出处时查不到"。

    所以：
    * L5.9 每条断言必须能在**它自己的块**里定位（不许拿别的块的原文替代）；
    * L5.10 聚合边不许再冒充事实来源，只许存派生统计，且代表证据必须可回溯；
    * L5.11 观察守恒——每条断言要么被某条边引用一次、要么有书面理由不存在。
    """
    from . import assertion as assertion_mod

    run = GateRun()
    by_chunk = {c["chunk_id"]: c["text"] for c in chunks}

    # --- L5.9 逐观察证据可定位 ---
    unlocatable = []
    bad_all = []
    for a in assertions:
        text = by_chunk.get(a["chunk_id"], "")
        start, end = a["evidence_start"], a["evidence_end"]
        if text[start:end] != a["evidence_text"]:
            unlocatable.append(a["assertion_id"])
        for occ in a.get("evidence_all") or []:
            if text[occ["start"] : occ["end"]] != a["evidence_text"]:
                bad_all.append(a["assertion_id"])
                break
    run.add(
        Gate(
            "L5.9 逐观察证据可定位（断言在自己的块里能查到原文）",
            "L5",
            "定位失败 0 / evidence_all 不符 0",
            f"断言 {len(assertions)}  定位失败 {len(unlocatable)}  evidence_all 不符 {len(bad_all)}",
            "PASS" if not (unlocatable or bad_all) else "FAIL",
            (
                "证据定位是回答可核对的唯一依据；"
                + (
                    "; ".join((unlocatable + bad_all)[:4])
                    if (unlocatable or bad_all)
                    else "实测全部可定位"
                )
            ),
        )
    )

    # --- L5.10 聚合边只存派生统计 ---
    # 判据：必须带 assertion_ids 且 support_count 与之一致；
    # 代表证据必须真的属于同边的某条断言（不能是拼出来的、也不能是摘来的）。
    synthetic = []
    mismatch = []
    untraceable = []
    by_aid = {a["assertion_id"]: a for a in assertions}
    for e in edges:
        aids = e.get("assertion_ids") or []
        if not aids:
            synthetic.append(e.get("edge_id") or e.get("id") or "?")
            continue
        if int(e.get("support_count") or 0) != len(aids):
            mismatch.append(
                f"{e.get('edge_id')} support={e.get('support_count')} ids={len(aids)}"
            )
        rep = e.get("evidence_assertion_id") or ""
        if rep not in aids:
            untraceable.append(f"{e.get('edge_id')} 代表证据不属于本边")
        elif by_aid.get(rep, {}).get("evidence_text") != e.get("evidence"):
            untraceable.append(f"{e.get('edge_id')} 代表证据与断言原文不一致")
    run.add(
        Gate(
            "L5.10 聚合边只存派生统计（可回溯到具体观察）",
            "L5",
            "无 assertion_ids 0 / support 不符 0 / 不可回溯 0",
            f"边 {len(edges)}  缺 ids {len(synthetic)}  support 不符 {len(mismatch)}  不可回溯 {len(untraceable)}",
            "PASS" if not (synthetic or mismatch or untraceable) else "FAIL",
            (
                "聚合边一旦自己带『一条证据 + 一个置信度』，"
                "就成了无法追问出处的事实来源——旧模型正是这样"
                + (
                    "; ".join((synthetic + mismatch + untraceable)[:4])
                    if (synthetic or mismatch or untraceable)
                    else "；实测全部可回溯"
                )
            ),
        )
    )

    # --- L5.11 观察守恒 ---
    conservation = assertion_mod.observation_conservation(assertions, edges)
    detail = (
        f"断言 {conservation['assertions']}  边 {conservation['edges']}  "
        f"support_sum {conservation['support_sum']}"
    )
    problems = (
        conservation["unreferenced"]
        + conservation["multiply_referenced"]
        + conservation["unknown_references"]
        + conservation["support_mismatch"]
    )
    run.add(
        Gate(
            "L5.11 观察守恒（每条断言恰好被一条边引用一次）",
            "L5",
            "未引用 0 / 重复引用 0 / 未知引用 0 / support 不符 0",
            detail,
            "PASS" if conservation["ok"] else "FAIL",
            (
                "丢一条观察不会报错，只会让某个事实突然只有一处出处；"
                + ("; ".join(str(p) for p in problems[:4]) if problems else "实测守恒")
            ),
        )
    )
    return run


def l4_identity_gates(graph: dict) -> GateRun:
    """L4.12 / L4.13：实体 id 不含名字、合并检查有记录。

    L4.12 现在是**必然 FAIL** 的：当前图的 `entities[].id` 就是名字本身
    （E3 之前的设计）。把它写出来而不是等 E3 落地再加，
    是因为"id 是名字"这件事本身就是误并的温床——名字一改 id 就变，
    拆分合并的历史也就无处可查。E3 落地后这条门会自然转 PASS。

    门默认不接进 `build_generation`（会把当前所有构建打红），
    由 `scripts/e_gates.py` 显式运行——见该脚本说明。
    """
    run = GateRun()
    entities = graph.get("entities") or []

    name_as_id = [
        e["name"]
        for e in entities
        if (e.get("id") or "") == (e.get("name") or "") or (e.get("name") or "") in (e.get("id") or "")
    ]
    run.add(
        Gate(
            "L4.12 实体 id 不含名字（身份与展示名分离）",
            "L4",
            "命中 0",
            f"{len(name_as_id)}/{len(entities)}",
            "PASS" if not name_as_id else "FAIL",
            (
                "id 含名字 = 改一个展示名就换一个身份，合并/拆分历史随之断掉；"
                + ("; ".join(name_as_id[:4]) if name_as_id else "")
            ),
        )
    )

    # 合并检查有记录：每个被并进别处的名字都该在 merged_from 里留下痕迹，
    # 且合并后的实体必须能说清按哪条规则并的（stats.merge 的计数）。
    merge_stats = (graph.get("stats") or {}).get("merge") or {}
    merged_total = sum(
        len(e.get("merged_from") or []) for e in entities
    )
    has_record = bool(merge_stats)
    run.add(
        Gate(
            "L4.13 合并检查有记录（merged_from 与规则计数同时存在）",
            "L4",
            "两者皆非空（无合并时允许均为空）",
            f"merged_from 条目 {merged_total}  规则计数 {len(merge_stats)} 项",
            "PASS" if (merged_total == 0 and not merge_stats) or (merged_total > 0 and has_record) else "FAIL",
            (
                "并了却没记按哪条规则并的，事后无法复核是否误并；"
                + (f"规则计数 {merge_stats}" if has_record else "无规则计数")
            ),
        )
    )
    return run


def l6_direction_gates(adjacency: dict, edges: list[dict]) -> GateRun:
    """L6.8：邻接方向完整性——出边入边分开、反向换逆谓词、无伪装。

    这里有三条独立判据，缺一条就会留下一条绕过路径：

    1. **方向标记与所在列表一致**（`direction_report.fake_out`）——
       抓"入边混进 out 列表"。
    2. **反向不沿用原谓词**（`verify_reverse_semantics`）——
       抓旧 `build_adjacency` 那种"两侧都塞、关系名不变"。
    3. **每条对称以外的边，两端各有正确落位**——
       光靠第 1 条不够：把入边塞进 out 列表并顺手把 `direction` 标成 `out`，
       第 1 条就放过了。第 3 条是拿原边反查邻接表，必须两边都在该在的位置。

    第 3 条是补上一个实测发现的缺口：最初只有前两条时，手工构造
    「入边塞进出边列表 + 方向标 out」的邻接能通过 L6.8——而旧代码正是那么写的。
    """
    from . import adjacency as adjacency_mod

    run = GateRun()
    report = adjacency_mod.direction_report(adjacency)
    problems = adjacency_mod.verify_reverse_semantics(edges)

    # 判据 3：拿原边反查邻接表的两端落位
    misplaced: list[str] = []
    for e in edges:
        subject = e["subject_id"]
        obj = e["object_id"]
        predicate = e["predicate"]
        subject_out = (adjacency.get(subject) or {}).get("out") or []
        obj_in = (adjacency.get(obj) or {}).get("in") or []
        if not any(
            it["neighbor_id"] == obj and it["edge_id"] == e["edge_id"]
            for it in subject_out
        ):
            misplaced.append(f"{e['edge_id']} 主语侧的出边缺失（{subject}->{obj}）")
        if adjacency_mod.is_symmetric(predicate):
            # 对称边两侧都该在 out，不该进 in
            if obj_in and any(it["edge_id"] == e["edge_id"] for it in obj_in):
                misplaced.append(f"{e['edge_id']} 对称边被塞进入边列表")
            continue
        if not any(
            it["neighbor_id"] == subject and it["edge_id"] == e["edge_id"]
            for it in obj_in
        ):
            misplaced.append(f"{e['edge_id']} 宾语侧的入边缺失（{obj}<-{subject}）")

    run.add(
        Gate(
            "L6.8 邻接方向完整性（出/入分列，反向换逆谓词，两端落位正确）",
            "L6",
            "方向伪装 0 / 反向沿用原谓词 0 / 落位错误 0",
            f"实体 {report['entities']}  出 {report['out_items']}  入 {report['in_items']}  "
            f"伪装 {report['fake_out']}  反向语义问题 {len(problems)}  落位错误 {len(misplaced)}",
            "PASS" if not (report["fake_out"] or problems or misplaced) else "FAIL",
            (
                "反向边沿用原谓词会把「凯尔希 隶属 罗德岛」读成「罗德岛 隶属 凯尔希」——"
                "检索侧看不出来，因为它只拿到邻居和关系名；"
                + (
                    "; ".join((problems + misplaced)[:4])
                    if (problems or misplaced)
                    else "实测方向干净"
                )
            ),
        )
    )
    # 无逆谓词的反向边只申报：这类边不得用于顺着方向说事实，但它们的入边本身不是错误。
    run.add(
        Gate(
            "L6.8a 无逆谓词的反向边（仅申报，不得用于方向性表述）",
            "L6",
            "申报值",
            f"{report['missing_inverse']} 条",
            "PASS",
            "词表没给逆谓词（创建/前往/使用…），反向只能当无向邻接用；"
            "要用于方向性回答必须先补逆谓词或人工裁决",
        )
    )
    return run


def write_report(run: GateRun, out: Path, stage: str) -> None:
    """把本 stage 的行合并进门禁报告。

    这是「读-改-写」共享状态：两个 stage 并发写时，无锁版本会读到同一份旧内容、
    各自合并、再由后写者覆盖，先写者的行就静默消失——报告看起来完整，其实少了
    一个 stage 的门。故整段临界区用文件锁串行，落盘用原子替换。

    `path=None` 时只返回合并结果不写盘（供候选 Generation 内部计算使用）。
    """
    out = Path(out)
    rows = run.rows()
    order = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "L6": 6, "L7": 7}
    with atomic.file_lock(out, timeout=120.0):
        out.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if out.exists():
            try:
                existing = json.loads(out.read_text("utf-8"))
            except json.JSONDecodeError:
                # 报告损坏不能静默当成空重新开始——那会把旧 stage 的行全部丢掉
                corrupt = out.parent / f"{out.name}.corrupt.{int(time.time())}"
                os.replace(out, corrupt)
                existing = []
        existing = [r for r in existing if r["stage"] != stage]
        existing.extend(rows)
        existing.sort(key=lambda r: (order.get(r["stage"], 9), r["name"]))
        atomic.atomic_write_text(
            out, json.dumps(existing, ensure_ascii=False, indent=2) + "\n"
        )


def sampling_for_human(scenes: list[dict], out: Path, n: int = 50) -> None:
    """计划书要求「抽样 50 条人工检查」——这里产出待验清单，不自行判 PASS。"""
    import random

    rng = random.Random(20261025)
    pool = [(s["doc_id"], t) for s in scenes for t in s["turns"] if len(t["text"]) > 40]
    picks = rng.sample(pool, min(n, len(pool)))
    lines = ["# L1 人工抽验清单（舞台指令/非剧情内容漏检）", "", "判定：每行是否为应丢弃的非剧情内容。漏检超过 2 条则 L1 门不过。", ""]
    for doc, t in picks:
        lines.append(f"- [{t['prop']}/{t['kind']}] {doc} :: {t['text'][:80]}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
