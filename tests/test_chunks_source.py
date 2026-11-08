# -*- coding: utf-8 -*-
"""块正文来源门：凡按某一代算数的脚本，必须走 `pipeline.resolve_chunks()`。

## 为什么这是一道门

同一个坑在本项目已经被踩过三次，每次形态不同、每次都不报错：

1. **D 阶段**：全局块文件 13,798 块，候选世代 13,781 块——两个「合法的」块文件。
2. **P1-b 结构报告**：分子读全局（13,786）、分母读世代（13,798），算出一个
   谁都不认识的覆盖率，报告里每个数字都真算出来了，% 也是对的。
3. **P1-c 判定脚本**：路径写成 `data/releases/generations/<gen>/chunks.jsonl`
   （不存在），带一句 `if not path.exists(): path = config.chunks_path()` 静默
   回退——59 条候选的正文全部来自另一个文件，而结论看上去完全正常。

第三次是靠 adversarial 那 9 条「两个名字都在正文」与 gold 答案的前提
「原文只出现短名」直接矛盾才暴露的。前两次分别是靠人复核和靠 `--collect-only`
对不上才暴露。**也就是说这个缺陷自己没有症状。**

`src/arkrag/pipeline.py:60-108` 的 `resolve_chunks()` 是仓库已有的唯一正解：
它只认世代自带那份、核 `chunk_manifest_hash`（`exists()` 检查抓不到这一层）、
旧世代借全局文件前必须证明是同一批块。`src/arkrag/server.py:666-671` 有专段
注释解释为什么不能读全局文件。

## 这道门的两层

**第一层（新代码）**：任何文件里出现 `config.chunks_path()`，若该文件同时有
世代上下文（`release.Generation(` / `--generation` / `GENERATIONS_DIR` / `gen_dir`
等等），就必须同时出现 `resolve_chunks`。否则判违规。

**第二层（存量债）**：实测有 20 个点位满足上面的违规条件，集中在
`sweep_*` / `diagnose_*` / `bench_*` / `check_*` 这些「用某代算一个指标」的脚本。
它们不是今天造成的，整改要逐个改代码并重跑报告——不能顺手在同一笔里做。
所以把它们登记进 `CHUNKS_SOURCE_DEBT`，并把登记表做成**只能缩小**：

* 新增一个违规点位 → 门红（必须显式登记，且登记出现在 diff 里）；
* 修好一个点位却忘了删登记 → 门也红（`test_debt_registry_matches_measured`）。

第二条是关键：只做「新增就红」的登记表会越积越厚，而**「修好了也要改登记」
才使得这份名单有下行压力**。`tests/test_subprocess_encoding.py` 的
`test_whitelist_points_at_real_calls` 是同一手法，理由也相同。

## 白名单（真有资格读全局文件的）

`CHUNKS_PATH_ALLOWED` 里的每一条都必须带理由，且必须能在源码里找到对应调用
（`test_allowed_entries_point_at_real_calls`）——死条目会让门悄悄变宽。

三条理由各自不同：

* `pipeline.resolve_chunks`：**它就是实现本身**，旧世代分支必须能拿到全局路径；
* `run_l2.main`：**它是写方**，L1→L2 产出这个文件，读它就是写它；
* `extract.load_chunks`：L1 抽取的输入读取器，此时还没有「世代」这个概念。

其余（`cache_reuse_audit` / `probe_reranker_budget` / `shadow_cache_check`）
读的是缓存的原文对照，不绑世代，也不参与任何按世代算的指标。

## 反向实测怎么做

1. 先跑 `py -3 -m pytest tests/test_chunks_source.py -q` → 全绿；
2. 在 `scripts/eval_fusion.py` 里加一行 `_ = config.chunks_path()`（它已有
   `release.Generation(` 上下文且没用 `resolve_chunks`）→ 门必须红；
3. 恢复 → 必须绿。

第 2 步不能用「已经在债表里的文件」做实验，那样红的是「债表不匹配」而不是
「新增点位被拦住」——两种红要分开验，见 `test_new_offender_is_flagged`。
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src" / "arkrag"

#: 出现这些标记说明这个文件在「按某一代算数」。
#: 判定放在**文件级**而不是函数级，是刻意的：一个文件里 `main()` 用了某一代，
#: 另一个辅助函数读全局块文件，两者相距 200 行——按函数判会漏掉。
GEN_CONTEXT_MARKERS: tuple[str, ...] = (
    "release.Generation(",
    "GENERATIONS_DIR",
    "gen_dir",
    "--generation",
    "generation_id:",
)

#: 唯一正确的块正文解析入口。
RESOLVER = "resolve_chunks"

#: 有资格直接读全局块文件的调用点。键 = (相对路径, 所在函数名)。
CHUNKS_PATH_ALLOWED: dict[tuple[str, str], str] = {
    ("src/arkrag/pipeline.py", "resolve_chunks"): (
        "就是解析器本身的实现：旧世代分支必须能取到全局块文件路径，"
        "并由它自己证明「这批块属于这一代」。此处读全局文件是定义的一部分。"
    ),
    ("scripts/run_l2.py", "main"): (
        "写方：L1→L2 把块正文写进 `config.chunks_path()`。读它等于读自己刚写的东西，"
        "此时还不存在「某一代自带的块文件」。"
    ),
    ("src/arkrag/extract.py", "load_chunks"): (
        "L1 抽取的输入读取器：抽取发生在产世代之前，此刻「本代自带块文件」尚未存在。"
    ),
    ("scripts/cache_reuse_audit.py", "main"): (
        "缓存复用审计：只按 chunk_id 比对缓存原文与块正文，不参与任何按世代算的指标。"
    ),
    ("scripts/probe_reranker_budget.py", "main"): (
        "重排预算探针：直接吃全局块正文构造固定评测输入，不绑世代。"
    ),
    ("scripts/shadow_cache_check.py", "load_chunks"): (
        "影子缓存对照：只比对缓存与块正文的一致性，不按世代算数。"
    ),
}

#: 存量债：按某一代算数、却从全局文件读块正文的点位。
#: **只能缩小**——新增必须显式加进来（出现在 diff 里），修好了必须删掉
#: （否则 `test_debt_registry_matches_measured` 红）。
CHUNKS_SOURCE_DEBT: dict[tuple[str, str], str] = {
    ("scripts/audit_goldset.py", "main"): (
        "按 `--generation` 把 gold 答案对回块正文；该参数的世代与全局块文件可能不是同一批。"
    ),
    ("scripts/bench_h1.py", "timed_load"): (
        "H1 性能基线，`generation_id` 显式入参；块正文读全局，与图中的世代可能不同源。"
    ),
    ("scripts/bench_h3.py", "_load_and_measure"): "H3 性能基线，同 bench_h1。",
    ("scripts/build_goldset.py", "_load_sources"): (
        "按世代挖 gold，但块正文读全局——这正是 v1 里 `required` 与断言层不同源的来源之一。"
    ),
    ("scripts/check_significance.py", "main"): "显著性检验，按世代评测；块源须与世代同源。",
    ("scripts/check_speaker_significance.py", "main"): "说话人显著性检验，同上。",
    ("scripts/diagnose_ranking.py", "main"): "排序诊断，按世代算指标。",
    ("scripts/diagnose_speaker_clamp.py", "main"): "说话人夹紧诊断，按世代算指标。",
    ("scripts/make_split.py", "mine_new_batch"): (
        "P1-a 新增批挖矿。实测：换成 `resolve_chunks()` 会换掉 84 道题"
        "（差异来自集合/顺序而非文本），故须单独一笔改并重出 gold-v2。"
    ),
    ("scripts/rerank_blind.py", "collect"): "盲重排采集，按世代算指标。",
    ("scripts/sweep_entity_weight.py", "main"): "参数扫描，按世代算指标。",
    ("scripts/sweep_fusion.py", "main"): "参数扫描，按世代算指标。",
    ("scripts/sweep_pool_depth.py", "main"): "参数扫描，按世代算指标。",
    ("scripts/sweep_rrf_k.py", "main"): "参数扫描，按世代算指标。",
    ("scripts/sweep_structured_boost.py", "main"): "参数扫描，按世代算指标。",
    ("scripts/sweep_weighted_rrf.py", "main"): "参数扫描，按世代算指标。",
}


# ---------------------------------------------------------------- 扫描（纯函数）


def _owners(tree: ast.Module) -> dict[int, str]:
    """每个 AST 节点所属的**最内层**函数名；模块级记 `<module>`。

    按函数名而不按行号登记，理由同 `tests/test_subprocess_encoding.py`：
    行号会随无关编辑漂移，漂移就红在与它无关的改动上——那正是「门变噪音、
    被人顺手放宽」的起点。
    """
    out: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for inner in ast.walk(node):
                out.setdefault(id(inner), node.name)
    return out


def scan_chunks_path_sites(pairs: list[tuple[Path, str]]) -> list[dict]:
    """找出 `config.chunks_path()` 的调用点。

    `pairs` 是 `(路径, 源码文本)`——源码由调用方读入，纯函数才能喂合成片段做
    「门会不会红」的用例，不必碰真仓库。
    """
    rows: list[dict] = []
    for path, source in pairs:
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            continue
        owners = _owners(tree)
        has_resolve = RESOLVER in source
        gen_context = any(marker in source for marker in GEN_CONTEXT_MARKERS)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "chunks_path"):
                continue
            rel = path.as_posix()
            rows.append(
                {
                    "file": rel,
                    "line": node.lineno,
                    "owner": owners.get(id(node), "<module>"),
                    "gen_context": gen_context,
                    "has_resolver": has_resolve,
                }
            )
    return rows


def offenders(sites: list[dict]) -> set[tuple[str, str]]:
    """违规点位 = 有世代上下文、却没解析器。

    同时把 `CHUNKS_PATH_ALLOWED` 里的合法点位排除——它们读全局文件是定义的一部分。
    """
    out: set[tuple[str, str]] = set()
    for row in sites:
        key = (row["file"], row["owner"])
        if not row["gen_context"] or row["has_resolver"]:
            continue
        if key in CHUNKS_PATH_ALLOWED:
            continue
        out.add(key)
    return out


def _real_pairs() -> list[tuple[Path, str]]:
    pairs: list[tuple[Path, str]] = []
    for base in (SCRIPTS, SRC):
        for path in sorted(base.glob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            pairs.append((Path(rel), path.read_text("utf-8")))
    return pairs


# ---------------------------------------------------------------- 门


class TestChunksSourceIsResolvedPerGeneration:
    """按某一代算数的脚本，不许从全局块文件读正文。"""

    def test_no_new_offender_beyond_registered_debt(self) -> None:
        """任何**未登记**的违规点位都让门红。

        失败信息要能直接照着改：给出文件、函数、以及「该怎么改」。
        """
        measured = offenders(scan_chunks_path_sites(_real_pairs()))
        unregistered = sorted(measured - set(CHUNKS_SOURCE_DEBT))
        assert not unregistered, (
            f"这些点位按某一代算数，却用 `config.chunks_path()` 读块正文，"
            f"且不在已登记的存量债里：{unregistered}。\n"
            f"改法：`pipeline.resolve_chunks(gen, manifest)` —— 它会核 "
            f"`chunk_manifest_hash`，`exists()` 检查抓不到这一层。\n"
            f"若确实不该改（比如它其实不绑世代），把理由加进 "
            f"`CHUNKS_SOURCE_ALLOWED`；那是一次需要写进 diff 的表态。"
        )

    def test_debt_registry_matches_measured(self) -> None:
        """债表**只能缩小**：修好一个点位就必须删掉对应登记。

        只做「新增就红」的登记表会越积越厚，最后没人相信里面有活条目；
        「修好了也要改登记」才让这份名单有下行压力。
        """
        measured = offenders(scan_chunks_path_sites(_real_pairs()))
        dead = sorted(set(CHUNKS_SOURCE_DEBT) - measured)
        assert not dead, (
            f"这些登记项已经不再违规（可能已改用 resolve_chunks，或所在函数改名）：{dead}。\n"
            f"请从 `CHUNKS_SOURCE_DEBT` 删掉它们——修好一处就少一笔债，"
            f"不要让它继续躺在名单里。"
        )

    def test_debt_entries_carry_reasons(self) -> None:
        """每条债都要写清「为什么必须改」，不能只写「以后再说」。"""
        for key, reason in CHUNKS_SOURCE_DEBT.items():
            assert reason.strip(), f"债表条目 {key} 没有理由"
            assert len(reason.strip()) >= 10, f"债表条目 {key} 的理由太短：{reason!r}"

    def test_allowed_entries_carry_reasons(self) -> None:
        """白名单的每一条都要有理由——无理由的白名单就是「永远绿」。
        """
        for key, reason in CHUNKS_PATH_ALLOWED.items():
            assert reason.strip(), f"白名单条目 {key} 没有理由"
            assert len(reason.strip()) >= 10, f"白名单条目 {key} 的理由太短：{reason!r}"

    def test_allowed_and_debt_do_not_overlap(self) -> None:
        """同一个点位不能又是「合法」又是「欠债」。

        同时出现在两边说明有人为了让门变绿把债加进了白名单——这正是
        Lead 说的「没有白名单的绝对门，最后一定会被改成永远绿」。
        """
        overlap = set(CHUNKS_PATH_ALLOWED) & set(CHUNKS_SOURCE_DEBT)
        assert not overlap, f"这些点位同时被登记为合法与欠债：{sorted(overlap)}"

    def test_allowed_entries_point_at_real_calls(self) -> None:
        """白名单条目必须对应真实存在的 `chunks_path()` 调用。

        死条目会让门悄悄变宽：函数改名或调用删掉后，那条豁免仍然生效，
        而没有人会去数白名单里还有多少条是活的。
        """
        measured = {
            (row["file"], row["owner"]) for row in scan_chunks_path_sites(_real_pairs())
        }
        dead = sorted(set(CHUNKS_PATH_ALLOWED) - measured)
        assert not dead, f"这些白名单条目对应的调用已经不存在：{dead}"

    def test_resolver_is_used_by_the_gate_reports(self) -> None:
        """按世代算指标、且**要出报告**的脚本必须真的用解析器。

        这条单独列出来，是因为「用没用 resolve_chunks」和「违规点位数量」
        是两件事：一个脚本可以一处违规都没有（因为它压根没读块正文），
        却仍然在按世代算数。
        """
        must_resolve = [
            ("scripts/k2_hard_gates.py", "K2 全量硬门"),
            ("scripts/k2_perf.py", "K2 性能评测"),
            ("scripts/p1c_leak_audit.py", "P1-c 审计"),
        ]
        missing = [
            f"{path}（{label}）"
            for path, label in must_resolve
            if RESOLVER not in (ROOT / path).read_text("utf-8")
        ]
        assert not missing, f"这些出报告的脚本没有走 resolve_chunks：{missing}"


# ---------------------------------------------------------------- 门自己会不会红


class TestGateItselfCanFail:
    """合成源码直接喂扫描器，证明判据不是恒真。"""

    def test_generation_context_without_resolver_is_flagged(self) -> None:
        source = (
            "import json\n"
            "from arkrag import config, release\n"
            "def main():\n"
            "    gen = release.Generation('g-20260101T000000Z-aaaaaa')\n"
            "    chunks = json.load(open(config.chunks_path()))\n"
        )
        sites = scan_chunks_path_sites([(Path("fake_eval.py"), source)])
        assert offenders(sites) == {("fake_eval.py", "main")}

    def test_generation_context_with_resolver_is_clean(self) -> None:
        source = (
            "from arkrag import config, pipeline, release\n"
            "def main():\n"
            "    gen = release.Generation('g-20260101T000000Z-aaaaaa')\n"
            "    path = pipeline.resolve_chunks(gen, {})\n"
        )
        sites = scan_chunks_path_sites([(Path("fake_eval.py"), source)])
        assert offenders(sites) == set()

    def test_no_generation_context_is_not_flagged(self) -> None:
        """不绑世代的工具（如纯缓存对照）不该被误伤。

        误伤会让人学会忽略这道门——与「门不红」同样糟糕。
        """
        source = (
            "from arkrag import config\n"
            "def load_chunks():\n"
            "    return config.chunks_path().read_text('utf-8')\n"
        )
        sites = scan_chunks_path_sites([(Path("fake_cache.py"), source)])
        assert offenders(sites) == set()

    def test_scan_reports_the_owning_function_not_the_file(self) -> None:
        """登记键是 (文件, 函数)，不是文件——同一文件里两个函数要能分开登记。"""
        source = (
            "from arkrag import config, release\n"
            "def uses_generation():\n"
            "    release.Generation('g-20260101T000000Z-aaaaaa')\n"
            "    return config.chunks_path()\n"
            "def also_uses_generation():\n"
            "    release.Generation('g-20260101T000000Z-bbbbbb')\n"
            "    return config.chunks_path()\n"
        )
        sites = scan_chunks_path_sites([(Path("fake.py"), source)])
        assert offenders(sites) == {
            ("fake.py", "uses_generation"),
            ("fake.py", "also_uses_generation"),
        }
