"""重排器的速度/精度权衡：改截断长度能不能把延迟压进预算。

## 问题

`probe_reranker.py` 实测：bge-reranker-base-int8 在 512 token（本项目块正文 P50
898 字，约 500+ token）下 **每对 95 ms**。重排 20 个候选要 1.9 秒，
目标预算是 P95 < 200 ms，差一个数量级。

## 唯一可能的出口

交叉编码器的成本 ≈ O(层数 × 序列长度²)。把输入从 512 token 截到 192/128
理论上能降一个量级，但**截断会丢掉块后半部分的信息**，可能把答案就在后半段的
题判错。所以不能只看延迟，必须同时看「截断后排序是否还和 512 一致」。

这里的判据是 **Spearman 秩相关**：对同一批候选，短截断与全长的排序若高度一致
（ρ > 0.9），说明截断只影响了尾部细节，可以接受；若 ρ 掉下来，说明前 192 token
不足以判断相关性，那就不能截。

## 另一条路：只重排 top-N

如果延迟随候选数线性增长，那重排 top-N（N=10）就够——因为真正要修的是
「gold 排在第 8 名却在上下文里被挤掉」这类近邻错位，不是第 100 名的块要不要上桌。
两种路径都要量，选哪个看数据。
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

MODEL_DIR = Path(
    r"C:\Users\Lenovo\WorkBuddy\2026-07-27-16-45-03\roleplay-ai\models\bge-reranker-base-onnx"
)
ONNX_INT8 = MODEL_DIR / "onnx" / "model_int8.onnx"
TOKENIZER = MODEL_DIR / "tokenizer.json"


def make_scorer(tok: Tokenizer, sess, max_len: int):
    def score(pairs: list[tuple[str, str]]) -> list[float]:
        tok.enable_truncation(max_length=max_len)
        encs = [tok.encode(q, d) for q, d in pairs]
        width = max(len(e.ids) for e in encs)
        ids = np.array([list(e.ids) + [1] * (width - len(e.ids)) for e in encs], dtype=np.int64)
        mask = np.array(
            [list(e.attention_mask) + [0] * (width - len(e.attention_mask)) for e in encs],
            dtype=np.int64,
        )
        feeds = {i.name: v for i, v in zip(sess.get_inputs(), (ids, mask))}
        out = sess.run(None, feeds)[0]
        return [float(x) for x in np.asarray(out).reshape(len(pairs), -1)[:, 0]]

    return score


def spearman(a: list[float], b: list[float]) -> float:
    """秩相关。同分用平均秩，避免 ties 把系数算歪。"""
    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        out = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    ra, rb = ranks(a), ranks(b)
    n = len(ra)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else 1.0


def main() -> int:
    tok = Tokenizer.from_file(str(TOKENIZER))
    sess = ort.InferenceSession(str(ONNX_INT8), providers=["CPUExecutionProvider"])

    # 用真实块正文测（P50 898 字），而不是短样板
    from arkrag import config

    docs = []
    with Path(config.chunks_path()).open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % 500 == 0:
                rec = json.loads(line)
                docs.append(str(rec.get("text") or ""))
            if len(docs) >= 12:
                break
    question = "凯尔希为什么要离开罗德岛"
    pairs = [(question, d) for d in docs]
    avg_chars = round(statistics.mean(len(d) for d in docs))
    print(f"真实块 {len(docs)} 条，平均 {avg_chars} 字")

    # 512 作为基准
    ref_score = make_scorer(tok, sess, 512)
    ref_score(pairs)  # 预热
    ref = ref_score(pairs)

    print(f"\n{'max_len':>8} {'每对ms':>9} {'top20估算ms':>12} {'Spearman':>10} {'结论':>8}")
    print("-" * 56)
    rows = []
    for max_len in (512, 384, 256, 192, 128, 96):
        fn = make_scorer(tok, sess, max_len)
        fn(pairs)  # 预热该形状
        t0 = time.perf_counter()
        reps = 3
        for _ in range(reps):
            got = fn(pairs)
        per_pair = (time.perf_counter() - t0) * 1000 / (reps * len(pairs))
        rho = round(spearman(ref, got), 4)
        est20 = round(per_pair * 20)
        verdict = "可用" if rho >= 0.9 and est20 <= 250 else ("排序变了" if rho < 0.9 else "仍太慢")
        rows.append({"max_len": max_len, "per_pair_ms": round(per_pair, 2),
                     "top20_ms": est20, "spearman": rho, "verdict": verdict})
        print(f"{max_len:>8} {per_pair:>9.1f} {est20:>12} {rho:>10.4f} {verdict:>8}")

    json.dump(rows, sys.stdout, ensure_ascii=False, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
