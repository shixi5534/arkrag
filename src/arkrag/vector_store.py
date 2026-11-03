"""向量通道：建、存、校验、检索。

## 三个必须守住的约束（计划 I1 原文）

1. **模型与维度入 Manifest。** 向量空间是模型定义的，换模型等于换空间。所以
   `model` + `digest` + `dim` 三样必须与向量一起落盘，并且**参与身份校验**。
2. **Chunk 内容哈希绑定向量。** 每条向量记它编码的那段文本的 sha256。块文本改一个字，
   哈希就对不上，向量必须重建——不许"向量留着、正文更新"这种静默错配。
3. **删除/修改 Chunk 不允许残留旧向量。** 由「一一对应」强制：`chunk_ids` 必须与
   当前块集合完全相同（既不许缺，也不许多）。多出来的就是残留。

## 为什么不把向量写进世代目录

与 H3 的 SQLite 同理：向量是 56 MB 级的大产物，写进 `generations/<gid>/` 会让
世代摘要变化，`snapshot_baseline.py` 与 `release_cli verify` 立刻变红——那种红是假红。
所以落在 `data/arkrag/vectors/`，由 sidecar 里的 `index_content_digest` 绑回世代。

## 关于 numpy

扫描 13,781 × 1024 需要 numpy：实测一次全量点积 numpy **2.82 ms**，
纯 stdlib 最快的 `math.sumprod` 也要 **938 ms**（`sum(genexpr)` 更慢，1,826 ms）。
差 300 倍以上，搜索引擎不能接受后者。

所以 numpy 是**可选依赖**：没有它时向量通道必须**显式缺席**（`vector_channels`
写 `"not_built_no_numpy"`），绝不允许静默退化成纯 BM25——那会让评测报告里
没有任何字段变红，而召回悄悄掉一截。这与 FTS 后端「库不存在不降级」同一条纪律。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

VECTOR_SCHEMA_VERSION = "ark-rag-vec-1"
# 落盘格式：float32 小端、行优先、无偏移的裸矩阵。
# 选裸矩阵而不是 .npz 是为了让校验可以只读 sidecar 就算出 sha256，
# 也让任何语言都能读（不绑死 numpy 的文件格式）。
ELEMENT_DTYPE = "float32"
ELEMENT_BYTES = 4


def numpy_available() -> bool:
    try:
        import numpy  # noqa: F401
    except ImportError:
        return False
    return True


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - 环境相关
        raise RuntimeError(
            "向量通道需要 numpy（全量扫描 2.82 ms vs stdlib 938 ms，差 300 倍）。"
            "未安装时请把 vector_channels 显式申报为 not_built_no_numpy，"
            "不要静默退化成纯 BM25。"
        ) from exc
    return np


def chunk_text_for_embedding(rec: dict) -> str:
    """**唯一**的「块被编码成什么文本」定义。

    必须与 `index.build()` 喂给词法索引的字符串逐字相同（同样带标题与章节前缀），
    否则向量通道与 BM25 通道看的是两份不同的文本，两者的分数就没法比较，
    RRF 融合也失去意义。**不要在这里另拼一份**：`build_vectors` 与
    `verify_vectors` 都调本函数，只此一处。
    """
    return f"{rec.get('title', '')} {rec.get('chapter', '')} {rec.get('text', '')}"


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def matrix_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def default_vector_dir() -> Path:
    """向量落盘根目录。与 `data/arkrag/fts/` 平级，刻意不在世代目录内。"""
    from . import config

    return config.artifact_root() / "vectors"


def vector_paths(model_slug: str, index_content_digest: str, root: Path | None = None) -> tuple[Path, Path]:
    """一对 (矩阵 .bin, 侧车 .json)。

    文件名带 `index_content_digest` 而不是世代 id：三个 ark-rag-3 世代内容完全一致的
    事实（H 阶段实测）说明**内容相同就该复用同一份向量**，重建一次要 10 分钟。
    世代 id 记在 sidecar 里供追溯。
    """
    base = (root or default_vector_dir())
    stem = f"{model_slug}.{index_content_digest[:16]}"
    return base / f"{stem}.bin", base / f"{stem}.json"


def save_vectors(
    vectors: list[list[float]],
    chunk_ids: list[str],
    texts: list[str],
    *,
    model: str,
    model_digest: str,
    dim: int,
    generation_id: str,
    index_content_digest: str,
    normalized: bool,
    root: Path | None = None,
) -> tuple[Path, Path, dict]:
    """原子写一对 (矩阵, 侧车)。返回 (bin 路径, json 路径, meta)。

    校验顺序有意义：**先证明输入自洽，再落盘**。若条数不匹配却已经写了 .bin，
    磁盘上就留下一个半成品，而 sidecar 里没有它——下次读的人会看到一对不配的文件。
    """
    np = _require_numpy()
    if not (len(vectors) == len(chunk_ids) == len(texts)):
        raise ValueError(
            f"条数不一致：vectors={len(vectors)} chunk_ids={len(chunk_ids)} texts={len(texts)}"
        )
    if not vectors:
        raise ValueError("空向量集不许落盘：那等于申报「有向量通道」却没有向量")
    for i, vec in enumerate(vectors):
        if len(vec) != dim:
            raise ValueError(f"第 {i} 条维度是 {len(vec)}，期望 {dim}")

    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.shape != (len(vectors), dim):
        raise ValueError(f"矩阵形状 {matrix.shape} 与 ({len(vectors)}, {dim}) 不符")
    payload = matrix.tobytes(order="C")

    from . import atomic

    bin_path, meta_path = vector_paths(_slug(model), index_content_digest, root)
    atomic.atomic_write_bytes(bin_path, payload)

    meta = {
        "schema_version": VECTOR_SCHEMA_VERSION,
        "model": model,
        "model_digest": model_digest,
        "dim": dim,
        "count": len(vectors),
        "dtype": ELEMENT_DTYPE,
        "normalized": bool(normalized),
        "generation_id": generation_id,
        "index_content_digest": index_content_digest,
        "matrix_sha256": matrix_sha256(payload),
        "matrix_bytes": len(payload),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "chunk_ids": list(chunk_ids),
        "chunk_text_sha256": [text_sha256(t) for t in texts],
    }
    atomic.atomic_write_json(meta_path, meta)
    return bin_path, meta_path, meta


def _slug(model: str) -> str:
    """把 `bge-m3:latest` 变成文件名安全且可读的 `bge-m3-latest`。"""
    out = []
    for ch in model:
        out.append(ch if (ch.isalnum() or ch in "-_.") else "-")
    return "".join(out).strip("-.") or "model"


def load_vectors(bin_path: Path, meta_path: Path):
    """读回 (matrix, meta)。做三项自检：形状、字节数、sha256。

    sha256 必须验：这个文件在磁盘上被截断或改坏时，`np.frombuffer` 只会得到
    一堆垃圾分数，不会抛异常——那种失败会一路走到"召回莫名下降"。
    """
    np = _require_numpy()
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    payload = Path(bin_path).read_bytes()
    if len(payload) != meta["matrix_bytes"]:
        raise ValueError(
            f"矩阵字节数不符：文件 {len(payload)}，侧车记 {meta['matrix_bytes']}"
        )
    if matrix_sha256(payload) != meta["matrix_sha256"]:
        raise ValueError(
            f"矩阵 sha256 与侧车不符（{bin_path}）：文件可能被截断或改坏"
        )
    dim, count = int(meta["dim"]), int(meta["count"])
    matrix = np.frombuffer(payload, dtype=np.float32)
    if matrix.size != count * dim:
        raise ValueError(f"矩阵元素数 {matrix.size} != {count}×{dim}")
    return matrix.reshape(count, dim), meta


class VectorIndex:
    """只读向量索引：持有矩阵 + 顺序化的 chunk_id。

    刻意**不**持有 chunk 正文或图：它只回答「哪些 chunk_id 与查询最接近」。
    把别的数据放进来会让这个类的生命周期与 Retriever 纠缠，而它应该能独立被
    `bench` 脚本与测试单独构造。
    """

    def __init__(self, matrix, meta: dict) -> None:
        self.matrix = matrix
        self.meta = meta
        self.chunk_ids: list[str] = list(meta["chunk_ids"])
        self.dim: int = int(meta["dim"])
        self.model: str = meta["model"]
        self.model_digest: str = meta.get("model_digest") or ""
        # 行号 → chunk_id 只为调试与校验；检索走矩阵下标，不用这个 dict。
        self._pos: dict[str, int] = {cid: i for i, cid in enumerate(self.chunk_ids)}

    def __len__(self) -> int:
        return len(self.chunk_ids)

    def search(self, query_vector: list[float], *, top_k: int = 20) -> list[tuple[str, float]]:
        """余弦 top-k。query_vector 必须已按同模型编码（本类不负责编码）。

        返回 `[(chunk_id, score)]`，score 越高越相似，降序。
        同分时按 chunk_id 升序——与词法索引的 tie-break 一致，否则 RRF 的
        输入顺序在真实同分场景下会不可复现。

        为什么要对查询向量重新归一化：bge-m3 的输出已 L2 归一化（实测 norm=1.000000），
        但这里不假定——若换了模型没归一化，点积就不再是余弦，排序会静默变错
        而没有任何地方会报错。归一化一次的成本是 1024 次乘加，可以忽略。
        """
        np = _require_numpy()
        if len(query_vector) != self.dim:
            raise ValueError(
                f"查询向量维度 {len(query_vector)} 与索引维度 {self.dim} 不符："
                f"查询与建库必须用同一模型（索引是 {self.model}）"
            )
        q = np.asarray(query_vector, dtype=np.float32)
        qn = float(np.linalg.norm(q))
        if qn > 0:
            q = q / qn
        scores = self.matrix @ q
        k = min(top_k, len(self.chunk_ids))
        # argpartition 取前 k 再精确排序：全排序 13,781 项比取前 k 慢约 8 倍
        idx = np.argpartition(-scores, k - 1)[:k]
        ranked = sorted(
            ((self.chunk_ids[i], float(scores[i])) for i in idx),
            key=lambda kv: (-kv[1], kv[0]),
        )
        return ranked


def verify_vectors(
    meta: dict,
    current_chunk_ids: list[str],
    current_texts: dict[str, str],
) -> dict:
    """三条一一对应检查。返回报告，**不许把失败降级成警告**。

    * 集合相等：多余的向量 = 残留（块已删除但向量还在），缺少的 = 漏建。
    * 顺序无关：`chunk_ids` 的顺序只影响矩阵行号，不影响正确性，但必须与
      `chunk_text_sha256` 逐位对齐——这由它们都按同一份 `sorted(chunk_records)` 产生保证。
    * 内容哈希：每个块的文本哈希必须与建库时一致。改一个字就必须重建。

    为什么不比对「文件行数」或「块数」了事：块数相同而内容不同是最常见的
    静默错配（比如修了分块器却复用旧向量），只有逐条哈希能发现。
    """
    bound_ids = list(meta.get("chunk_ids") or [])
    bound_hashes = list(meta.get("chunk_text_sha256") or [])
    if len(bound_ids) != len(bound_hashes):
        return {
            "ok": False,
            "reason": f"侧车自相矛盾：chunk_ids {len(bound_ids)} 条，哈希 {len(bound_hashes)} 条",
        }
    bound = set(bound_ids)
    current = set(current_chunk_ids)
    missing = sorted(current - bound)
    orphan = sorted(bound - current)
    mismatched: list[str] = []
    for cid, want in zip(bound_ids, bound_hashes):
        text = current_texts.get(cid)
        if text is None:
            continue  # 已计入 missing
        if text_sha256(text) != want:
            mismatched.append(cid)
    problems = []
    if missing:
        problems.append(f"缺 {len(missing)} 条向量（漏建）：{missing[:3]}")
    if orphan:
        problems.append(f"残留 {len(orphan)} 条向量（块已不存在）：{orphan[:3]}")
    if mismatched:
        problems.append(
            f"{len(mismatched)} 条文本哈希不符（块正文变过但向量未重建）：{mismatched[:3]}"
        )
    return {
        "ok": not problems,
        "count": len(bound_ids),
        "current_count": len(current),
        "missing": len(missing),
        "orphan": len(orphan),
        "mismatched": len(mismatched),
        "problems": problems,
    }
