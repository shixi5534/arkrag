"""prop 白名单——由 L0 全量实测得出（79 个折叠名），不是抽样推断。

只有 5 个 prop 携带人可读正文；其余全部丢弃并归类，未登记的一律进 unknown 计数。
"""
from __future__ import annotations

# 正文 prop -> 取值字段
TEXT_PROP_FIELD = {
    "name": "content",
    "multiline": "content",
    "subtitle": "text",
    "sticker": "text",
    "animtext": "content",
}

# subtitle 实测正文是「夜幕降临。」「源石锅炉的轰鸣声并未减弱……」这类描述性散文，
# 因此归旁白而不归标题卡；真正的换幕标注是 animtext（地点+年代戳）。
KIND_BY_PROP = {
    "name": None,        # 有说话人 -> dialogue，无说话人 -> narration
    "multiline": None,   # 同上
    "subtitle": "narration",
    "sticker": "quote",
    "animtext": "title_card",
}

PLAYER_OPTION = {"decision"}
BRANCH_CONTROL = {"predicate", "gotopage", "skipnode", "skiptothis"}
GAMEPLAY = {"startbattle", "tutorial"}
TITLE_SCREEN = {"header"}

STAGING = {
    "dialog", "character", "charslot", "blocker", "delay", "background", "image",
    "imagetween", "playmusic", "stopmusic", "stopsound", "playsound", "camerashake",
    "cameraeffect", "characteraction", "charactercutin", "musicvolume", "soundvolume",
    "effect", "bgeffect", "backgroundtween", "largebg", "largebgtween", "verticalbg",
    "gridbg", "curtain", "interlude", "theater", "avgdisplay", "stickerclear",
    "spellsticker", "spellstickerclear", "timersticker", "timerclear", "animtextclean",
    "showitem", "hideitem", "cgitem", "hidecgitem", "focusout", "focusparam",
    "imagerotate", "imgeffect", "glitchtween", "warp", "video", "title", "div",
    "comment",
}

# 游戏数据里的拼写错误变体，语义上都是暂存指令；单列一类是为了让「未知」计数
# 只暴露真正没见过的新形态，而不是被陈旧拼写噪声淹没。
TYPO_STAGING = {
    "dialo", "dialogs", "delat", "dealy", "delau", "daley", "delya", "dalay",
    "delayt", "delay9ti", "stopmucis", "musicvolune", "palysound", "charslsot",
    "chaa", "duration", "fadetime",
}

DROP_CLASS = {}
for _p in PLAYER_OPTION:
    DROP_CLASS[_p] = "player_option"
for _p in BRANCH_CONTROL:
    DROP_CLASS[_p] = "branch_condition"
for _p in GAMEPLAY:
    DROP_CLASS[_p] = "gameplay"
for _p in TITLE_SCREEN:
    DROP_CLASS[_p] = "title_screen"
for _p in STAGING:
    DROP_CLASS[_p] = "staging"
for _p in TYPO_STAGING:
    DROP_CLASS[_p] = "typo_staging"

KNOWN_PROPS = set(TEXT_PROP_FIELD) | set(DROP_CLASS)

COVERED_PROPS = {
    "name": 411033, "charslot": 194993, "character": 135758, "delay": 74692,
    "blocker": 72564, "dialog": 65078, "playsound": 41048, "background": 16823,
    "camerashake": 14472, "stopmusic": 10381, "playmusic": 8726, "subtitle": 8064,
    "sticker": 5857, "image": 5810, "predicate": 5061, "characteraction": 4937,
    "decision": 4152, "stopsound": 3666, "cameraeffect": 2117, "header": 2067,
    "multiline": 1986, "musicvolume": 1831, "bgeffect": 1751, "soundvolume": 1636,
    "curtain": 1520, "imagetween": 1354, "interlude": 1222, "effect": 1212,
    "charactercutin": 652, "focusout": 638, "backgroundtween": 614,
    "animtextclean": 249, "animtext": 248, "gridbg": 221, "hidecgitem": 189,
    "cgitem": 188, "largebgtween": 148, "avgdisplay": 129, "largebg": 123,
    "theater": 39, "verticalbg": 32, "stickerclear": 28, "focusparam": 28,
    "hideitem": 27, "showitem": 26, "title": 24, "div": 24, "imagerotate": 20,
    "spellsticker": 15, "spellstickerclear": 10, "timerclear": 9, "skipnode": 9,
    "timersticker": 8, "warp": 7, "imgeffect": 6, "video": 6, "delat": 3,
    "palysound": 3, "comment": 3, "dealy": 2, "delau": 2, "glitchtween": 2,
    "skiptothis": 2, "daley": 2, "delya": 2, "stopmucis": 1, "musicvolune": 1,
    "dialo": 1, "delayt": 1, "chaa": 1, "charslsot": 1, "dialogs": 1,
    "startbattle": 1, "tutorial": 1, "gotopage": 1, "duration": 1, "fadetime": 1,
    "delay9ti": 1, "dalay": 1,
}
