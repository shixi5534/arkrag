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
CHUNK_OVERLAP_TURNS = int(os.environ.get("ARKRAG_CHUNK_OVERLAP_TURNS", "1"))

# L3 并发预算：全局上限 8，由「并行子代数 x 本值」共同决定
LLM_CONCURRENCY = int(os.environ.get("ARKRAG_CONCURRENCY", "4"))
LLM_RETRIES = int(os.environ.get("ARKRAG_RETRIES", "3"))
LLM_MODEL = os.environ.get("ARKRAG_LLM_MODEL", "")
LLM_BASE_URL = os.environ.get("ARKRAG_LLM_BASE_URL", "")

CONFIDENCE_THRESHOLD = float(os.environ.get("ARKRAG_CONF_THRESHOLD", "0.5"))

CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
SHARDS_PATH = DATA_DIR / "shards.json"
GRAPH_PATH = DATA_DIR / "graph.json"
L0_MANIFEST_PATH = DATA_DIR / "l0_manifest.json"
CIRCUIT_OPEN_PATH = DATA_DIR / "circuit_open.json"
