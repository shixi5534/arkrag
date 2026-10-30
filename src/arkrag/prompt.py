"""L3 抽取提示词 v1：六段结构（角色任务 / 硬规则 / 精确性 / 词表 / 格式 / 兜底）。

改这里必须同步改 PROMPT_VERSION，否则旧缓存全部命中，看不到任何变化。
"""
from __future__ import annotations

from . import config
from .vocab import CANONICAL_RELATIONS, ENTITY_TYPES

_TYPE_LINE = "/".join(ENTITY_TYPES)
_RELATION_LINE = "、".join(CANONICAL_RELATIONS)

SYSTEM_PROMPT = f"""你在为《明日方舟》剧情文本构建知识图谱。任务只有一个：从给定剧情块中抽出实体与关系。

硬性规则：
1. 错误的节点比缺失的节点危害大得多。宁可漏抽，不可错抽。
2. 只输出 JSON 对象本身，不要 markdown 围栏，不要解释文字。
3. 每条关系必须在当前块原文中有直接依据。禁止用你的游戏设定知识补写原文没写的关系。
4. 实体类型只能取这六个：{_TYPE_LINE}。取不到就丢弃该实体。
5. 关系名只能取这些：{_RELATION_LINE}。找不到合适的就丢弃该关系，不要自造。
6. evidence 必须是块正文的连续原文片段，不超过 40 字，不得改写、不得概括。
7. 关系的 source 与 target 必须都出现在本次 entities 列表里，名字逐字一致。
8. 同一实体在 entities 里只出现一次；别名放进 aliases，别名不得包含规范名自身。
9. 实体名必须是「名字」，不能是描述性短语；禁止整句当实体名；禁止代词（我/你/他/她/它/我们/你们/某人/对方/敌人/众人）当实体名。
10. 说话人名字已在正文里以「名字：台词」形式给出；说话人若是有专名的角色即可作为实体，但「？？？」「神秘人」这类掩码或无名指代不得成为实体，也不得据其建边。
11. 形如 LC0073458、ZOOT-SYS-INIT、VI03 的编号与代码不得成为实体。
12. 「事件」类型只收有专名的重大事件（如「切尔诺伯格事变」「巴别塔倒塌」），不收「他推开门就走了」这类句子。
13. desc 用一句话说清它是什么，不要复述剧情。
14. 无可抽取内容时返回空结果，不要凑数。
15. 只列实体不建边等于没抽。每个实体应尽量至少参与一条有原文依据的边；确实建不了边的实体可以只列着，但不要为凑边而造没有依据的关系。
16. 台词里最常见的边就是关系：A 与 B 说话用「交谈」，A 与 B 同行的用「同行」，A 命令/指派 B 用「指挥」，A 效忠/忠于 B 用「效忠」。这些是实义关系，不是兜底。

输出格式：
{{"entities":[{{"name":"阿米娅","type":"角色","aliases":["罗德岛领袖"],"desc":"罗德岛制药公司的领导者"}}],"relations":[{{"source":"阿米娅","relation":"隶属","target":"罗德岛","evidence":"阿米娅，罗德岛的leader","confidence":0.9}}]}}

空结果的标准形态：
{{"entities":[],"relations":[]}}"""

USER_TEMPLATE = """剧情块信息
- 篇目：{title}
- 章节：{chapter}
- 场景：{doc_id}
- 说话人：{speakers}

剧情正文（每行一句，对话行为「说话人：台词」，其余为旁白/引文/字幕）：
---
{text}
---

只输出 JSON 对象。"""


def build_user_prompt(chunk: dict) -> str:
    return USER_TEMPLATE.format(
        title=chunk["title"] or "（无名）",
        chapter=chunk["chapter"],
        doc_id=chunk["doc_id"],
        speakers="、".join(chunk["speakers"][:20]) if chunk["speakers"] else "（无对话）",
        text=chunk["text"],
    )


def prompt_fingerprint() -> str:
    import hashlib

    return hashlib.sha256((SYSTEM_PROMPT + "\x1f" + USER_TEMPLATE).encode("utf-8")).hexdigest()[:16]


PROMPT_VERSION = config.PROMPT_VERSION
