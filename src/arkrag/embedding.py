"""Ollama embedding 客户端。

为什么走 HTTP 而不是直接加载模型：本机的 embedding 能力由 Ollama 提供
（`bge-m3:latest`，1024 维，566.7M 参数 Q4_K_M）。走它的 HTTP 接口意味着
**只需要标准库 `urllib`**，不必引入 torch/transformers/sentence-transformers——
这保住了项目「运行时零第三方依赖」的纪律（`requirements.txt` 只列 pytest 相关）。

代价是网络调用会失败（服务没起、模型没拉）。所以每条失败路径都必须显式抛错，
不许静默返回空向量：一个空的向量通道会让检索悄悄退化成纯 BM25，而评测报告里
没有任何字段会变红。这与 FTS5 后端「库不存在不降级」是同一条纪律。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

# 模型身份是**索引身份的一部分**：换模型必须触发新向量世代，否则向量空间对不上。
# 三个常量都要进 sidecar meta，由 gates 校验（见 gates.py L6.7 的同型门）。
DEFAULT_MODEL = "bge-m3:latest"
DEFAULT_DIM = 1024
# bge-m3 的输出已 L2 归一化（实测 norm=1.000000），所以余弦相似度 = 点积。
# 但这条不能只信文档：`probe()` 会实测一条并断言，避免上游换了模型却没打招呼。
EXPECTS_NORMALIZED = True

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT = 600.0
DEFAULT_BATCH = 16


class EmbeddingError(RuntimeError):
    """embedding 通道不可用。调用方必须当作硬错误处理，不许降级成"没有向量"。"""


def base_url() -> str:
    import os

    return os.environ.get("ARKRAG_EMBED_BASE_URL") or DEFAULT_BASE_URL


def model_name() -> str:
    import os

    return os.environ.get("ARKRAG_EMBED_MODEL") or DEFAULT_MODEL


def _post(path: str, payload: dict, *, timeout: float) -> dict:
    url = base_url().rstrip("/") + path
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise EmbeddingError(f"{url} 返回 HTTP {exc.code}：{detail}") from exc
    except urllib.error.URLError as exc:
        raise EmbeddingError(
            f"连不上 embedding 服务 {url}（{exc.reason}）。"
            f"检查 Ollama 是否在跑：`ollama list`，或用 ARKRAG_EMBED_BASE_URL 改地址。"
        ) from exc
    except TimeoutError as exc:
        raise EmbeddingError(f"{url} 超时（{timeout}s）") from exc


def embed_batch(
    texts: list[str],
    *,
    model: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[list[float]]:
    """一次 HTTP 调用编码一批文本。返回顺序与入参严格一致。"""
    if not texts:
        return []
    payload = {"model": model or model_name(), "input": list(texts)}
    data = _post("/api/embed", payload, timeout=timeout)
    vectors = data.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        got = len(vectors) if isinstance(vectors, list) else "非列表"
        raise EmbeddingError(
            f"embedding 返回条数不符：请求 {len(texts)} 条，返回 {got}"
        )
    for i, vec in enumerate(vectors):
        if not isinstance(vec, list) or not vec:
            raise EmbeddingError(f"第 {i} 条向量为空或类型不对：{type(vec).__name__}")
    return vectors


def embed_all(
    texts: list[str],
    *,
    model: str | None = None,
    batch_size: int = DEFAULT_BATCH,
    progress=None,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[list[float]]:
    """按批编码全部文本。

    实测吞吐（13,781 块、P50 898 字/块）：batch=8 约 47 ms/块、batch=16 约 43.5 ms/块，
    全量约 10–13 分钟。批再大收益递减且有内存峰值风险，故默认 16。
    """
    out: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, batch_size):
        chunk = texts[start : start + batch_size]
        out.extend(embed_batch(chunk, model=model, timeout=timeout))
        if progress is not None:
            progress(min(start + len(chunk), total))
    return out


def model_info(*, model: str | None = None) -> dict:
    """读服务端对该模型的自述：dim、参数量、量化、能力、digest。

    **digest 是模型身份的地面真值**。Ollama 用同样的 tag 也可能拉到不同权重的
    重打包版本，只有 digest 能证明两次建库用的是同一份权重。
    """
    want = model or model_name()
    url = base_url().rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as exc:
        raise EmbeddingError(f"连不上 embedding 服务 {url}（{exc}）") from exc
    for item in data.get("models") or []:
        if item.get("name") == want:
            details = item.get("details") or {}
            return {
                "name": item.get("name"),
                "digest": item.get("digest"),
                "size": item.get("size"),
                "dim": details.get("embedding_length"),
                "parameter_size": details.get("parameter_size"),
                "quantization": details.get("quantization_level"),
                "family": details.get("family"),
                "context_length": details.get("context_length"),
                "capabilities": item.get("capabilities") or [],
            }
    raise EmbeddingError(
        f"服务端没有模型 {want!r}。先 `ollama pull {want}`，或看 `ollama list` 有哪些。"
    )


def probe(*, model: str | None = None) -> dict:
    """实测一条向量：维度、是否 L2 归一化、单条延迟。

    建库前必须调用。理由：`DEFAULT_DIM` / `EXPECTS_NORMALIZED` 是写死的常量，
    而模型可能被换掉。**不实测就断言维度，等于把常量当事实。**
    """
    info = model_info(model=model)
    t0 = time.perf_counter()
    vec = embed_batch(["罗德岛"], model=model)[0]
    ms = (time.perf_counter() - t0) * 1000
    dim = len(vec)
    norm = sum(x * x for x in vec) ** 0.5
    problems = []
    if dim != DEFAULT_DIM:
        problems.append(f"维度实测 {dim}，常量写的是 {DEFAULT_DIM}")
    if EXPECTS_NORMALIZED and abs(norm - 1.0) > 1e-3:
        problems.append(f"期望 L2 归一化（norm=1），实测 {norm:.6f}")
    return {
        "model": info["name"],
        "digest": info["digest"],
        "dim": dim,
        "expected_dim": DEFAULT_DIM,
        "norm": norm,
        "normalized": abs(norm - 1.0) <= 1e-3,
        "latency_ms": ms,
        "parameter_size": info["parameter_size"],
        "quantization": info["quantization"],
        "problems": problems,
    }
