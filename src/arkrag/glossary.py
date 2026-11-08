"""外挂术语库：把 PRTS 词库（泰拉词库 + 角色真名）变成检索链路能用的别名/整词表。

## 这个模块解决什么

本项目的图（`ark-rag-3`）里，6,583 个实体有 **32 个角色是纯拉丁名**（`Logos`、
`Mon3tr`、`Mechanist`、`Scout`、`Mantra`、`Sharp`、`Touch`、`Pith` …），其中只有 7 个
带中文别名。而语料里这些角色**几乎只以拉丁名出现**（实测 13,798 块正文：
`Logos 853 / 逻各斯 0`、`Mantra 393 / 真言 1`、`Mechanist 360 / 机械师 14`）。

后果：用户用中文问「逻各斯是谁」时，`query.analyze_query` 拿不到实体候选，
`retrieve.Retriever.search` 的实体通道**整路空转**——不是排名差，是这一路根本没被点亮。

PRTS「泰拉词库」正好给出这些角色的中文规范名（`逻各斯 → Logos`、`真言 → Mantra`、
`机械师 → Mechanist`、`电弧 → Raidian` …）。本模块把这类跨语种对应解析成
「中文形态 → 图内实体」的隐藏别名，供两处消费：

* `arkrag.index.name_terms_from_graph(graph, extra_aliases=...)` —— 让中文名进整词表；
* 实体的 `aliases` —— `retrieve.py` 的实体通道会自动把这些中文名拼进 probe
  （那段逻辑读 `ent["aliases"]`，无需改动检索器）。

## 三条纪律（都来自本项目已经踩过的坑）

1. **不修改图产物**。术语库是外挂的，解析结果永远从 `(glossary, graph)` 现算，
   不往 `graph.json` 里写 entity_id —— 世代一换 id 就可能变，写死就是把过期 id
   固化进证据。所以本模块只提供函数，不提供"就地改写 graph"的入口。

2. **一对多的别名一律丢弃，不许猜**。`norm_entity.ambiguous_aliases` 的历史读数：
   「凯尔希」这一个别名曾同时被 32 个实体主张。术语库里同一个中文形态如果映射到
   两个图实体，本模块直接剔除并记进 `ambiguous`，不按热度挑一个——那属于合并判定，
   不是术语库的职权。

3. **不允许静默降级**。源文件缺失、解析出 0 条、章节表头不认识，都抛
   `GlossaryError`。这个仓库出过的最大一类事故就是"看起来跑通了，其实什么都没接上"
   （见 `docs/升级台账.md` 的门规范教训表）。

## 数据源与解析口径

`data/external/prts_terminology.rendered.html` 是 PRTS「泰拉词库」的
`action=parse&prop=text` 渲染结果。**必须用渲染版而不是 wikitext**：干员代号那一节
在 wikitext 里是 `{{#ask:[[分类:干员]]…}}` 语义查询，明文只存在于渲染后的 HTML。

原文结构（实测，111,750 字符）：

    <div class="mw-heading mw-heading2"><h2 id="地名">…</h2>…
    <table class="wikitable">…<tbody><tr><td>龙门</td><td>Lungmen</td>…

* 五列表（基础设定/地名/组织/种族/干员代号）：`中文 | 英文 | 其他 | 日文 | 英文(悠星)`；
* 三列表（其他角色）：`中文 | 日文 | 英文`；
* 四列表（杂项）：`中文 | 日文 | 英文 | 说明`；
* 单元格内的 `/` 分隔多种等义写法；`-` 表示该语言不存在，空串表示未记录；
* 行内 `<ruby><rb>源石</rb>…<rt>オリジニウム</rt></ruby>` 的读音在 `<rt>` 里，
  必须先还原成 `<rb>` 的内容再剥标签，否则「源石(オリジニウム)」会把读音混进中文名。

`data/external/prts_realnames.wikitext` 是「角色真名」页，补的是「代号 → 真名」方向
（`山 → 安东尼·西蒙`、`林 → 林雨霞`）。解析必须先删 `<!-- -->` 注释块，否则被注释掉
的干员行会混进来。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

#: 术语库产物 schema。改解析口径就必须改它，否则读旧文件的调用方会拿到新语义。
GLOSSARY_SCHEMA_VERSION = "ark-rag-gloss-1"

#: 章节名到「这个章节的术语属于什么类型」的映射。`None` 表示不对应实体类型
#: （基础设定/杂项是世界观词，不是实体类型；角色两节都是角色）。
#: `角色` 是**容器节**：h2 本身没有表格，它的内容是挂在下面的两个 h3
#: （干员代号 / 其他角色）。它必须登记，否则会被当成"不认识的章节"。
_SECTION_TYPES = {
    "基础设定": None,
    "地名": "地点",
    "组织": "组织",
    "种族": "种族",
    "角色": "角色",
    "干员代号": "角色",
    "其他角色": "角色",
    "杂项": None,
}

#: 页脚导航章节，不是术语。实测「注释与链接」5 行全是目录链接。
_SKIP_SECTIONS = frozenset({"目录", "注释与链接"})

_HEADING_RE = re.compile(r'<h([23])[^>]*id="([^"]*)"[^>]*>(.*?)</h\1>', re.S)
_RUBY_RE = re.compile(r"<ruby>(.*?)</ruby>", re.S)
_RB_RE = re.compile(r"<rb>(.*?)</rb>", re.S)
# `<ref>…</ref>` 是条目脚注（如 Mon3tr 的"由凯尔希所给予的名字…"），不是术语内容。
# 不剥它会把整段脚注当成一个别名形态塞进实体，实测 `Mon3tr` 因此拿到 4 条垃圾别名。
_REF_PAIR_RE = re.compile(r"<ref\b[^>]*>.*?</ref>", re.S)
_REF_SELF_RE = re.compile(r"<ref\b[^>]*/>", re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_ROW_RE = re.compile(r"<tr>(.*?)</tr>", re.S)
_CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S)
_MULTI_SEP_RE = re.compile(r"\s*[/／]\s*")


class GlossaryError(RuntimeError):
    """术语库不可用或解析不出内容。**调用方必须当硬错误处理。**"""


# ---------------------------------------------------------------- HTML 解析


def _clean_cell(raw: str) -> list[str]:
    """把一个单元格清成一组等义写法。

    顺序不能换：`<ruby>` 必须先于剥标签处理。剥完标签再找 ruby 就只剩
    「源石オリジニウム」，读音与正文粘在一起，再也分不开。
    """
    text = _RUBY_RE.sub(lambda m: "".join(_RB_RE.findall(m.group(1))), raw)
    text = _REF_PAIR_RE.sub("", text)
    text = _REF_SELF_RE.sub("", text)
    text = _TAG_RE.sub("", text)
    text = (
        text.replace("\ufeff", "")
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )
    out: list[str] = []
    for part in _MULTI_SEP_RE.split(text):
        part = re.sub(r"\s+", " ", part).strip()
        if not part or part == "-":
            continue
        out.append(part)
    return out


def parse_terminology_html(html: str) -> list[dict]:
    """把「泰拉词库」渲染 HTML 解析成条目列表。

    每条：`{section, zh:[...], en:[...], ja:[...], fields:[[cell,...],...]}`。
    `zh/en/ja` 是去重后的形态列表；`fields` 保留原始格子，便于事后核对口径。
    """
    if not isinstance(html, str) or not html.strip():
        raise GlossaryError("术语库 HTML 为空")

    # 章节名取**标题正文**（group 3），不取 `id` 属性：目录那一节的 id 是
    # `mw-toc-heading` 而正文是「目录」，按 id 当名字会把它当成一节术语表。
    heads = [
        (m.start(), m.group(1), _TAG_RE.sub("", m.group(3)).strip())
        for m in _HEADING_RE.finditer(html)
    ]
    if not heads:
        raise GlossaryError("术语库 HTML 里一个 <h2 id=…> 都没找到——解析口径或源文件已变")

    out: list[dict] = []
    for i, (pos, level, name) in enumerate(heads):
        if name in _SKIP_SECTIONS:
            continue
        if name not in _SECTION_TYPES:
            # 章节名不认识就炸。踩过的坑：一度只按 `level == "2"` 取节，结果把
            # 挂在「角色」下面的两个 h3（干员代号 427 条、其他角色 88 条）整节漏掉，
            # 而输出"看起来成功"——少了 515 条术语，其中大部分是本模块唯一有价值的部分。
            raise GlossaryError(
                f"术语库出现不认识的章节 {name!r}（h{level}）。"
                f"已知章节：{sorted(_SECTION_TYPES)}。站点结构变了就别猜，先修解析口径。"
            )
        end = heads[i + 1][0] if i + 1 < len(heads) else len(html)
        block = html[pos:end]
        rows = _ROW_RE.findall(block)
        for row in rows:
            cells = [_clean_cell(c) for c in _CELL_RE.findall(row)]
            if not cells or not cells[0]:
                continue  # 表头 / 空行
            first = cells[0]
            if first == ["中文"]:
                continue  # 二级表头
            zh, en, ja = first, [], []

            def _pick(idx: int) -> list[str]:
                return cells[idx] if idx < len(cells) else []

            if len(cells) >= 5:  # 五列：中文|英文|其他|日文|英文
                en, ja = _pick(1), _pick(3)
            elif len(cells) == 3:  # 三列（其他角色）：中文|日文|英文
                ja, en = _pick(1), _pick(2)
            elif len(cells) == 4:  # 四列（杂项）：中文|日文|英文|说明
                ja, en = _pick(1), _pick(2)
            out.append(
                {
                    "section": name,
                    "zh": [z for z in zh if z],
                    "en": [e for e in en if e],
                    "ja": [s for s in ja if s],
                    "fields": cells,
                }
            )

    if not out:
        raise GlossaryError("术语库解析出 0 条——表头列宽或章节名已变，不许当成'空词库'放行")
    return out


# ---------------------------------------------------------------- 真名解析

_REALNAME_ROW_RE = re.compile(r"^\|\{\{干员头像\|([^}|]+)[^}]*\}\}\|\|(.*)$")
_REALNAME_POPUP_RE = re.compile(r"图标=(.*?)\|内容=", re.S)


def _clean_wikitext(text: str) -> str:
    # 顺序有讲究：ref 必须在剥标签之前处理，否则 `<ref name="x" />` 会被当成普通
    # 标签清掉、但 `<ref name="x">正文</ref>` 的正文会留下来，变成一条以脚注为内容
    # 的"别名"。实测 `Mon3tr` 与「岁」阵营五名干员就是这样各拿到一条垃圾别名。
    text = _REF_PAIR_RE.sub("", text)
    text = _REF_SELF_RE.sub("", text)
    text = re.sub(r"\{\{mdi\|[^}]*\}\}", " ", text)  # {{mdi|arrow-right}} 是"又名"箭头图标
    text = text.replace("{{=}}", "=")
    text = re.sub(r"</?(?:del|br|span|div|small|b|i)\b[^>]*>", " ", text)
    text = re.sub(r"\{\{(?:color|Color)\|[^|]*\|", "", text)
    text = text.replace("{{", " ").replace("}}", " ")
    text = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)
    text = text.replace("'''", "").replace("''", "")
    return re.sub(r"\s+", " ", text).strip(" |")


def parse_realnames_wikitext(wikitext: str) -> list[dict]:
    """把「角色真名」页解析成 `[{code, realnames:[...]}]`。

    `cells[0]` 是代号列（`{{Color|#xxx|代号}}`），真名在 `cells[1]`。
    这个偏移错过一次：按 `cells[0]` 取会把代号当成了真名，解析"成功"但全是错的。

    真名列里 `<br>` 分隔**同一角色的多种真名写法**（`Mon3tr` 是
    `AMa-10<br>{{mdi|arrow-right}} Mon3tr`，重岳是 `朔<br>{{mdi|arrow-right}}重岳`），
    所以要拆成列表，不能整格当一个名字。
    """
    if not isinstance(wikitext, str) or not wikitext.strip():
        raise GlossaryError("真名 wikitext 为空")
    body = re.sub(r"<!--.*?-->", "", wikitext, flags=re.S)
    out: list[dict] = []
    for line in body.splitlines():
        m = _REALNAME_ROW_RE.match(line)
        if not m:
            continue
        code = _clean_wikitext(m.group(1))
        cells = m.group(2).split("||")
        if len(cells) < 2:
            continue
        popup = _REALNAME_POPUP_RE.search(cells[1])
        raw = popup.group(1) if popup else cells[1]
        forms: list[str] = []
        for part in re.split(r"<br\s*/?>", raw):
            cleaned = _clean_wikitext(part)
            if cleaned and cleaned != code and cleaned not in forms:
                forms.append(cleaned)
        if not code:
            continue
        out.append({"code": code, "realnames": forms})
    if not out:
        raise GlossaryError("真名页解析出 0 条——行格式已变，不许当成'没有真名'放行")
    return out


# ---------------------------------------------------------------- 与图对齐


def _entity_texts(graph: dict) -> dict[str, str]:
    """图中每个「可被用户打出来的文本」→ entity_id（小写键，供大小写无关匹配）。"""
    out: dict[str, list[str]] = {}
    for ent in graph.get("entities") or []:
        eid = str(ent.get("id") or "")
        if not eid:
            continue
        name = str(ent.get("name") or "").strip()
        if name:
            out.setdefault(name.casefold(), []).append(eid)
        for item in ent.get("aliases") or []:
            text = item.get("text") if isinstance(item, dict) else item
            text = str(text or "").strip()
            if text:
                out.setdefault(text.casefold(), []).append(eid)
    # 一对多保留原样，由调用方决定是否丢弃（本模块自己会丢）
    return {k: v for k, v in out.items()}


_QUOTE_CHARS = "\u201c\u201d\u2018\u2019\"'\u300c\u300d\u300e\u300f"


def _strip_quotes(text: str) -> str:
    return text.strip().strip(_QUOTE_CHARS).strip()


def _looks_like_term(text: str) -> bool:
    """形态是否像一个能进索引的词，而不是解析残渣。

    实测踩过的残渣：`AMa-10<ref name="ama" />mdi|arrow-right Mon3tr<ref>…</ref>`、
    `朔<ref name="岁阵营"/>mdi|arrow-right重岳`。它们全部来自"先剥标签、后处理 ref"
    的次序错误。这里加一道词表入口的兜底：宁可漏一条别名，也不许把一整段脚注
    当成别名灌进整词表——那会让 `tokenize_t2` 的专名表被污染，副作用遍布全库。
    """
    if not text or len(text) > 30:
        return False
    for bad in ("<", ">", "|", "{{", "}}", "ref", "mdi", "\n", "\t"):
        if bad in text:
            return False
    return bool(re.search(r"[\u4e00-\u9fffA-Za-z0-9]", text))


def resolve(
    entries: list[dict],
    graph: dict,
    *,
    extra_pairs: list[dict] | None = None,
) -> dict:
    """把术语条目对齐到图实体，只保留**能唯一确定**的对应。

    返回：`{resolved: {entity_id: [别名文本, ...]}, ambiguous: [...], unresolved: [...]}`。

    * `resolved` 里只放「图里还没有的文本」——已经在 `name`/`aliases` 里的重复添加
      没有意义，只会让 `name_terms_from_graph` 的集合多跑一遍。
    * 同一个中文形态被两个以上实体认领 → 进 `ambiguous`，一条都不加。理由见模块
      docstring 第 2 条。
    """
    owner = _entity_texts(graph)
    by_id = {str(e.get("id") or ""): e for e in graph.get("entities") or []}

    def _type_ok(eid: str, section: str) -> bool:
        """章节期望的类型与命中实体类型必须一致。

        踩过的坑：真名页里有个干员代号就叫「贝洛内」，而图里「贝洛内」是一个**组织**
        （贝洛内家族）。不加类型守卫时，`德米特里·切塔尔多·贝洛内` 会被当别名塞进
        那个组织实体——查询「贝洛内家」时实体通道就会拉出一个人而不是组织。
        章节类型为 None（基础设定/杂项）时不设限。
        """
        want = _SECTION_TYPES.get(section)
        if want is None:
            return True
        got = str((by_id.get(eid) or {}).get("type") or "")
        return got == want
    resolved: dict[str, list[str]] = {}
    ambiguous: list[dict] = []
    unresolved: list[dict] = []

    def _candidates(forms: list[str], section: str) -> dict[str, list[str]]:
        got: dict[str, list[str]] = {}
        for form in forms:
            hits = [e for e in (owner.get(form.casefold()) or []) if _type_ok(e, section)]
            if hits:
                got[form] = sorted(set(hits))
        return got

    for ent in entries:
        zh_forms = [z for z in ent.get("zh") or [] if z]
        # 中文形态优先；没有中文命中时，用英文/日文形态去对上图中拉丁名/日文名
        for forms, note in ((zh_forms, "zh"), (ent.get("en") or [], "en"), (ent.get("ja") or [], "ja")):
            if not forms:
                continue
            cands = _candidates(forms, ent.get("section") or "")
            if not cands:
                continue
            # 一个条目只认一个目标实体：先看中文形态是否全体指向同一实体
            targets = {eid for hits in cands.values() for eid in hits}
            if len(targets) == 1:
                eid = next(iter(targets))
                existing = owner
                added = resolved.setdefault(eid, [])
                for form in list(cands) + zh_forms:
                    form = _strip_quotes(form)
                    if len(form) < 2 or not _looks_like_term(form):
                        continue
                    if form.casefold() in existing:
                        continue
                    if ent.get("section") == "干员代号" and not re.search(r"[\u4e00-\u9fff]", form):
                        # 纯拉丁/日文形态对干员代号没有意义：图上本来就叫那个名字，
                        # 而且 `Logos` 这种词进中文整词表只会污染切分。
                        continue
                    if form not in added:
                        added.append(form)
                if not resolved[eid]:
                    resolved.pop(eid)
                break
            for form, hits in cands.items():
                if len(hits) > 1:
                    ambiguous.append(
                        {"text": form, "entity_ids": hits, "section": ent.get("section"), "via": note}
                    )
                else:
                    unresolved.append({"text": form, "section": ent.get("section"), "via": note})
                break
            break
        else:
            unresolved.append({"zh": zh_forms, "section": ent.get("section")})

    # 真名线：代号已在图中的，把真名补成别名
    for pair in extra_pairs or []:
        code = str(pair.get("code") or "").strip()
        names = [str(n).strip() for n in (pair.get("realnames") or []) if str(n).strip()]
        if not code or not names:
            continue
        code_hits = [
            e for e in sorted(set(owner.get(code.casefold()) or [])) if _type_ok(e, "干员代号")
        ]
        if len(code_hits) != 1:
            continue
        eid = code_hits[0]
        added = resolved.setdefault(eid, [])
        self_name = str((by_id.get(eid) or {}).get("name") or "").casefold()
        for real in names:
            if real.casefold() in owner:
                continue
            # 全名 + 按 `·`/`・`/`-` 拆出的片段（`安东尼·西蒙` → 安东尼 / 西蒙）。
            # 片段也收是因为语料里常只提姓氏或只提名（实测 gold 里「山」的问句写作
            # 「安东尼」，全名反而不出现）。
            for frag in [real] + [p for p in re.split(r"[·・\-]", real) if len(p) >= 2]:
                frag = _strip_quotes(frag)
                if len(frag) < 2 or not _looks_like_term(frag) or frag.casefold() in owner:
                    continue
                if frag.casefold() == self_name:
                    # 自名形态（`阿达·“可露希尔”·丘奇` 里的「可露希尔」）剥引号后正好
                    # 等于实体自己的名字。它不是别名——加进来只会让实体通道的 probe
                    # 里重复一遍自己的名字，白占一次检索。
                    continue
                if frag not in added:
                    added.append(frag)
        if not resolved[eid]:
            resolved.pop(eid)

    # 冲突收口：同一个文本若被多个 entity_id 认领，全部剔除
    claims: dict[str, list[str]] = {}
    for eid, texts in resolved.items():
        for text in texts:
            claims.setdefault(text.casefold(), []).append(eid)
    for text, eids in claims.items():
        if len(set(eids)) > 1:
            ambiguous.append({"text": text, "entity_ids": sorted(set(eids)), "section": None, "via": "conflict"})
    for text, eids in claims.items():
        if len(set(eids)) > 1:
            for eid in set(eids):
                resolved[eid] = [t for t in resolved.get(eid, []) if t.casefold() != text]
    resolved = {eid: texts for eid, texts in resolved.items() if texts}

    return {"resolved": resolved, "ambiguous": ambiguous, "unresolved": unresolved}


def build(
    terminology_html: str,
    realnames_wikitext: str | None,
    graph: dict,
) -> dict:
    """一步到位：解析两个源 → 对齐到图 → 返回可落盘的术语库对象。

    结果结构：

        {schema, counts:{entries, resolved_entities, resolved_aliases, ambiguous, unresolved},
         entries:[{section, zh, en, ja}], aliases:{entity_id:[text,...]},
         ambiguous:[...], unresolved:[...]}
    """
    entries = parse_terminology_html(terminology_html)
    pairs = parse_realnames_wikitext(realnames_wikitext) if realnames_wikitext else []
    aligned = resolve(entries, graph, extra_pairs=pairs)
    return {
        "schema": GLOSSARY_SCHEMA_VERSION,
        "counts": {
            "entries": len(entries),
            "realname_pairs": len(pairs),
            "resolved_entities": len(aligned["resolved"]),
            "resolved_aliases": sum(len(v) for v in aligned["resolved"].values()),
            "ambiguous": len(aligned["ambiguous"]),
            "unresolved": len(aligned["unresolved"]),
        },
        "entries": [
            {"section": e["section"], "zh": e["zh"], "en": e["en"], "ja": e["ja"]}
            for e in entries
        ],
        "aliases": aligned["resolved"],
        "ambiguous": aligned["ambiguous"],
        "unresolved": aligned["unresolved"],
    }


# ---------------------------------------------------------------- 落盘 / 读盘


def save(glossary: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(glossary, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load(path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise GlossaryError(f"术语库文件不存在：{path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    got = data.get("schema")
    if got != GLOSSARY_SCHEMA_VERSION:
        raise GlossaryError(
            f"术语库 schema 不符：文件是 {got!r}，本模块要 {GLOSSARY_SCHEMA_VERSION!r}（{path}）。"
            f"不兼容的旧文件必须重建，不许兼容读——语义变了还按旧结构解释就是静默错。"
        )
    if not data.get("aliases"):
        raise GlossaryError(f"术语库 {path} 的 aliases 是空的，等于没接上")
    return data


def aliases_for(graph: dict, glossary: dict) -> dict[str, list[str]]:
    """给检索层用的「entity_id → 额外别名」，并再次按当前图校验。

    为什么要再校验一次：术语库文件可能是用上一代图对齐出来的。世代一换，某些
    entity_id 就不存在了；把不存在的 id 塞进 `entities_by_id` 只会静默无效。
    本函数只保留**当前图里真实存在的 id**，并把消失的 id 数报给调用方。
    """
    live = {str(e.get("id") or "") for e in graph.get("entities") or []}
    out: dict[str, list[str]] = {}
    for eid, texts in (glossary.get("aliases") or {}).items():
        if eid not in live:
            continue
        keep = [t for t in texts if str(t).strip()]
        if keep:
            out[eid] = keep
    return out


def augment_entities(entities: list[dict], extra: dict[str, list[str]]) -> list[dict]:
    """返回**新列表**，给每个实体补上术语库别名（不修改入参）。

    为什么返回新对象而不是就地 append：`graph["entities"]` 是从 `graph.json` 读出来的，
    同一进程里 `Retriever.entities_by_id` 与别处都指向它。就地改会让"用了术语库"和
    "没用术语库"的两次实验互相污染——本项目在 baseline 对比上栽过这种跟头
    （`docs/i阶段实测.md` 的口径五条里，第 2 条就是"同一进程两次实验必须互不干扰"）。
    """
    out: list[dict] = []
    for ent in entities:
        eid = str(ent.get("id") or "")
        add = extra.get(eid) or []
        if not add:
            out.append(ent)
            continue
        have = set()
        for item in ent.get("aliases") or []:
            text = item.get("text") if isinstance(item, dict) else item
            if text:
                have.add(str(text).casefold())
        merged = list(ent.get("aliases") or [])
        for text in add:
            if text.casefold() in have:
                continue
            merged.append({"text": text, "source": "glossary", "review": "auto"})
            have.add(text.casefold())
        clone = dict(ent)
        clone["aliases"] = merged
        out.append(clone)
    return out
