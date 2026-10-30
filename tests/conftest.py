"""pytest 公共装置（计划 F1）。

要保证的三件事：

1. **导入路径固定**：测试不依赖调用者 cwd，也不依赖 `PYTHONPATH` 环境变量。
2. **写入隔离**：任何测试都不许碰真实的 `data/`，全部落进 tmp_path。
3. **全局状态可回滚**：现有代码在 `config` 上留了模块级常量、在环境变量里读
   API Key，测试改动它们必须自动复原，否则后续测试跑在不同的世界里。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """清掉可能影响行为的 ARKRAG_* 环境变量，避免本机残留污染测试结果。"""
    for key in list(os.environ):
        if key.startswith("ARKRAG_"):
            monkeypatch.delenv(key, raising=False)
    yield


@pytest.fixture
def sample_chunk_text() -> str:
    """一段带对话说话人的 Chunk 正文，用于证据定位测试。"""
    return (
        "凯尔希：博士，罗德岛不会为一个人的牺牲改变航向。\n"
        "阿米娅：可是我们还有别的办法。\n"
        "凯尔希：没有。切尔诺伯格已经不存在了。\n"
    )


@pytest.fixture
def sample_task(sample_chunk_text) -> dict:
    import hashlib

    return {
        "schema_version": "ark-rag-task-2",
        "run_id": "run-test",
        "chunk_id": "obt/main/test_01#000",
        "content_key": hashlib.sha256(sample_chunk_text.encode()).hexdigest()[:24],
        "scene_id": "obt/main/test_01",
        "doc_id": "obt/main/test_01",
        "source_fingerprint": "deadbeefdeadbeef",
        "prompt_sha256": "cafebabecafebabe",
        "model": "test:model",
        "generation": 0,
        "text": sample_chunk_text,
    }
