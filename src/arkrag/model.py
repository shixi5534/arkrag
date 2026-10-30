"""数据契约：文档 -> 场景 -> 轮次，以及块/实体/边的结构。"""
from __future__ import annotations

from dataclasses import dataclass, field

# 轮次四类互斥（计划书 1.1 术语表）
KINDS = ("dialogue", "narration", "quote", "title_card")


@dataclass
class Turn:
    kind: str
    speaker: str
    text: str
    node_id: int
    prop: str
    speaker_masked: bool = False

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "speaker": self.speaker,
            "text": self.text,
            "node_id": self.node_id,
            "prop": self.prop,
            "masked": self.speaker_masked,
        }


@dataclass
class Scene:
    """一个剧情文件即一个场景（签字四）；场景是分块的硬边界。"""

    doc_id: str
    category: str
    chapter: str
    title: str
    event_id: str
    entry_type: str
    story_code: str
    turns: list[Turn] = field(default_factory=list)
    dropped: dict = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return sum(len(t.text) for t in self.turns)

    def as_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "category": self.category,
            "chapter": self.chapter,
            "title": self.title,
            "event_id": self.event_id,
            "entry_type": self.entry_type,
            "story_code": self.story_code,
            "turns": [t.as_dict() for t in self.turns],
            "dropped": self.dropped,
        }
