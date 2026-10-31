"""全局配置与版本常量。所有路径可用 ARKRAG_* 环境变量覆盖。"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SCHEMA_VERSION = "ark-rag-1"
PROMPT_VERSION = "v2"
NORMALIZER_VERSION = "n2"

# 图产物的 schema 有两代，**并存**，不是替代：
#
#   ark-rag-1  聚合边模型：`entities[].id` 就是规范名，`aliases` 是 list[str]，
#              边是 `{source, relation, target, confidence, evidence}` 的压平结果。
#              已发布的每一代产物都是这一版，`data/releases/legacy-baseline` 也是。
#   ark-rag-3  逐观察模型（E 阶段）：实体用 `e-<16 位哈希>` 稳定 id，别名是
#              list[dict] 且权威/待审/拒绝三表分离，边带 `assertion_ids` 可回查
#              每一条观察，邻接分 out/in 且反向换逆谓词。
#
# 为什么不直接把 `SCHEMA_VERSION` 改成 ark-rag-3：那个常量进的是**抽取契约**的
# 身份（`contracts`、`extraction` 块、`chunks.jsonl` 的 schema_version），
# 改它等于宣称缓存里的 13,801 条 raw 抽取结果换了契约——它们没换。
# 图的结构换代是另一件事，所以单独两个常量，由构建脚本按 `--graph-model` 选。
GRAPH_SCHEMA_LEGACY = "ark-rag-1"
GRAPH_SCHEMA_IDENTITY = "ark-rag-3"


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw) if raw else default


def rel_to_root(path) -> str:
    """仓库内路径写成相对仓库根的形式（正斜杠），仓库外保持原样。

    为什么需要：绝对机台路径进 git 后换机器就错。相对路径在任何机器上都指向
    同一个仓库内对象。

    两条边界，都是刻意的：

    * 仓库**外**原样返回 `str(path)`，不写成 `../../..` 逃逸形式。那种字符串
      看着像相对路径，实际把"站在哪个目录"重新塞回语义里——换个 caller 就解析
      到别处，比绝对路径更糟。
    * 大小写只用于**比较**，不用于输出。Windows 上 `E:\\图rag` 与 `e:\\图rag`
      是同一路径但字符串不等，所以"在不在仓库内"要在 casefold 之后判断；
      返回的相对部分取自已 `resolve()` 的路径，保留磁盘上的原始大小写。
      （`Path.resolve()` 在 NTFS 上会顺手把父目录大小写归一，但不能指望这一点：
      大小写敏感文件系统上它保留调用者给的大小写。）

    比较前一律 `resolve()`：`..` 与符号链接会让"字符串像是仓库内"变成假象。
    """
    raw = Path(path)
    root = ROOT.resolve()
    try:
        resolved = raw.resolve()
    except OSError:
        # resolve(strict=False) 正常不抛；真抛了也不该让"写报告"这一步失败，
        # 退一步用不解析的绝对形式——它至少还是这台机器上正确的路径。
        resolved = raw.absolute()
    # 按**路径分量**比较，不按字符串前缀切片：
    #   * 分量比较天然要求边界落在分隔符上——`E:\\图rag2\\x` 的分量是
    #     ('E:\\', '图rag2', 'x')，不会因为前缀相似被错认成仓库内。
    #   * casefold 可能改变字符串长度（如 'ß'.casefold() == 'ss'），按长度切片
    #     会切错位置；分量比较没有这个前提。
    root_parts = root.parts
    got_parts = resolved.parts
    if len(got_parts) >= len(root_parts) and all(
        a.casefold() == b.casefold() for a, b in zip(got_parts, root_parts)
    ):
        rest = got_parts[len(root_parts) :]
        if not rest:
            return "."
        return Path(*rest).as_posix()
    return str(raw)


SOURCE_DIR = _env_path("ARKRAG_SOURCE_DIR", ROOT / "corpus" / "zh_CN")
DATA_DIR = _env_path("ARKRAG_DATA_DIR", ROOT / "data" / "arkrag")
CACHE_DIR = DATA_DIR / "cache"
LEDGER_DIR = DATA_DIR / "ledger"
INDEX_DIR = DATA_DIR / "index"
SIDECAR_DIR = DATA_DIR / "sidecar"

STORY_ROOT = SOURCE_DIR / "gamedata" / "story"

# 语料族 -> story 下的相对目录；Obt 目录在游戏数据里就是首字母大写
CATEGORIES = {
    "obt_main": "Obt/main",
    "obt_memory": "Obt/memory",
    "obt_guide": "Obt/guide",
    "obt_rogue": "Obt/Rogue",
    "roguelike": "Obt/Roguelike",
    "activities": "activities",
}

# T1 抽取范围（唯一真源：L2 分片 / L3 调度 / L5 建图 / 台账都从这里取，别再各处硬编码）
#   主线 obt/main + 活动 activities + 干员密录 obt/memory + 肉鸽 obt/roguelike / obt/rogue
# obt/guide（教学引导，全库 5 块）不在范围内：内容是操作提示，不含人物关系，收进来只会稀释密度。
# 见 docs/决策台账.md D-18 / D-19。
T1_CATEGORIES = (
    "obt/main",
    "activities",
    "obt/memory",
    "obt/roguelike",
    "obt/rogue",
)


def in_t1(category: str) -> bool:
    """category 是否落在 T1 抽取范围内。

    category 形如 `obt/main` / `activities` / `obt/memory`。用前缀匹配是为了
    `activities` 能一并覆盖将来可能出现的 `activities/xxx` 写法。
    """
    return any(category == c or category.startswith(c + "/") for c in T1_CATEGORIES)


# L2 参数（决策推荐区间：目标 800-1000 / 最小 100-150）
CHUNK_TARGET_CHARS = int(os.environ.get("ARKRAG_CHUNK_CHARS", "900"))
CHUNK_MIN_CHARS = int(os.environ.get("ARKRAG_CHUNK_MIN_CHARS", "120"))
# 硬上限：任何块的字符数都不得超过它。尾块合并与重新平衡都以这条为红线，
# 计划 D2 要的正是"合并前预测总长度，超硬上限不许合"。
CHUNK_MAX_CHARS = int(os.environ.get("ARKRAG_CHUNK_MAX_CHARS", str(2 * CHUNK_TARGET_CHARS)))
CHUNK_OVERLAP_TURNS = int(os.environ.get("ARKRAG_CHUNK_OVERLAP_TURNS", "1"))

# L3 并发预算：全局上限 8，由「并行子代数 x 本值」共同决定
LLM_CONCURRENCY = int(os.environ.get("ARKRAG_CONCURRENCY", "4"))
LLM_RETRIES = int(os.environ.get("ARKRAG_RETRIES", "3"))
LLM_MODEL = os.environ.get("ARKRAG_LLM_MODEL", "")
LLM_BASE_URL = os.environ.get("ARKRAG_LLM_BASE_URL", "")

CONFIDENCE_THRESHOLD = float(os.environ.get("ARKRAG_CONF_THRESHOLD", "0.5"))

CIRCUIT_OPEN_PATH = DATA_DIR / "circuit_open.json"


def artifact_root() -> Path:
    """本轮产物根。默认就是 `DATA_DIR`；重跑 L0/L1/L2 时指到 staging。

    为什么需要这层间接：`data/arkrag/` 下那 11 个文件已被冻结为
    `legacy-baseline`（见 `data/releases/legacy-baseline/manifest.json`），
    写保护会拦它们。但 D 阶段修了分块器的两个 bug，必须重跑 L1/L2 才能得到
    真正的 13,798 块——于是产物**必须**有地方落。

    做法是把"产物根"从"冻结基线目录"里拆出来：

        ARKRAG_ARTIFACT_DIR=data/arkrag/staging/d-stage
        py -3 scripts\\run_l1.py && py -3 scripts\\run_l2.py

    这样新产物一律写进 staging，冻结基线一个字节都不动；`snapshot_baseline
    --verify` 因此永远是 PASS，而不是"FAIL 了再去 --force 重建基线"。

    **注意这是函数不是常量**：测试通过 `monkeypatch.setattr(config, "DATA_DIR", …)`
    把产物根指到临时目录，函数体内的 `DATA_DIR` 引用才会跟着变。改成模块级常量
    会让那些测试静默读写真实仓库——`tests/test_build_generation.py` 记着这个教训。
    """
    raw = os.environ.get("ARKRAG_ARTIFACT_DIR")
    return Path(raw) if raw else DATA_DIR


def scenes_path() -> Path:
    return artifact_root() / "scenes.jsonl"


def chunks_path() -> Path:
    return artifact_root() / "chunks.jsonl"


def shards_path() -> Path:
    return artifact_root() / "shards.json"


def corpus_meta_path() -> Path:
    return artifact_root() / "corpus_meta.json"


def l1_stats_path() -> Path:
    return artifact_root() / "l1_stats.json"


def l1_dropped_path() -> Path:
    return artifact_root() / "l1_dropped.json"


def l0_manifest_path() -> Path:
    return artifact_root() / "l0_manifest.json"


def gates_path() -> Path:
    return artifact_root() / "gates.json"


def graph_path() -> Path:
    return artifact_root() / "graph.json"


def index_path() -> Path:
    return artifact_root() / "index" / "lexical.json"


# 兼容别名：老代码按常量名引用。它们绑定在 `DATA_DIR` 上，**只在冻结基线
# 语境下正确**；新代码一律用上面的函数。
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
SHARDS_PATH = DATA_DIR / "shards.json"
GRAPH_PATH = DATA_DIR / "graph.json"
L0_MANIFEST_PATH = DATA_DIR / "l0_manifest.json"
