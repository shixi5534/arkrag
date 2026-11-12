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
def tmp_path(tmp_path_factory):
    """覆盖 pytest 内建的 `tmp_path`，改用 `os.makedirs` 建同语义目录。

    ## 为什么必须覆盖（实测，非推测）

    pytest 建临时目录走 `Path.mkdir(mode=0o700)`。在 Windows 上，一旦给
    `mkdir` 传了 `mode`，Python 就**不再从父目录继承 DACL**，而是新写一个只
    授权所有者的 ACL。而本机 DSH 受限令牌是靠一条 `S-1-4-*` 授权条目拿到访问权
    的——目录一重写 ACL，那条授权就没了，于是**进程自己列的目录自己都读不了**：

        mkdir(0o700)  -> scandir -> PermissionError [WinError 5]
        makedirs()    -> scandir -> OK

    后果不是"某个用例失败"，而是**所有用 `tmp_path` 的用例在 setup 阶段就
    ERROR**（`tests/` 下 35 个文件用到了它）。这不是本项目引入的缺陷：在干净
    环境（能继承 DACL 的普通用户令牌）下两种写法行为一致，所以覆盖它不改变
    任何真实环境下的语义。

    ## 为什么放在 conftest 而不是各测试文件里

    写一份覆盖就够全项目用。各文件各抄一份，就会出现"漏抄的那个文件在这个环境
    下整片 ERROR"，而那种失败最容易被误读成"代码坏了"。

    ## 为什么不用 `tmp_path_factory` 或 `tempfile.mkdtemp`

    两者内部都走 `mode=0o700`（`mkdtemp` 也是），会踩同一个坑：父目录一开始就
    不可列举，于是连在其中 `mkdir("data")` 都报 WinError 5。所以统一走
    `config.make_temp_dir()`——它用不带 mode 的 `os.makedirs`，并自己保证唯一性。
    这条纪律写在一处，别处（如 `scripts/k1_checks.py`）也复用同一个函数，
    免得以后再多出第六个踩坑点。
    """
    import shutil

    from arkrag import config as _config

    base = _config.make_temp_dir("arkrag-test-")
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


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
