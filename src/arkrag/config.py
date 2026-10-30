"""全局配置与版本常量。所有路径可用 ARKRAG_* 环境变量覆盖。"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SCHEMA_VERSION = "ark-rag-1"
PROMPT_VERSION = "v2"
NORMALIZER_VERSION = "n2"


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw) if raw else default


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
