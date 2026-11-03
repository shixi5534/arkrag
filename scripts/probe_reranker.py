"""探测本机 ONNX 重排器：可用性、速度、打分是否合理。

## 为什么要先探

`scripts/diagnose_channels.py` 的结果是：bm25 单路取前 100 时 gold 命中率
**0.9367**，但 R@20 只有 0.6478、MRR 0.4248。也就是说 **gold 块基本都在候选池里，
问题纯粹是把它们排进前 20**。RRF 解决不了这件事（它只是名次加权，两路都排不进
头部的东西，融合后还是排不进），扫 k 也解不了（实测每个 k 都是召回升/MRR降）。

交叉编码器重排是对的工具：它把 (问句, 块正文) 拼在一起过一遍模型，直接输出相关性
分数，不依赖两路各自的名次。本机 `bge-reranker-base` 的 ONNX 版本在内网离线可用，
所以先确认：能不能加载、单对多快、分数在「显然相关」与「显然无关」之间拉不拉得开。

## 三个必须量的东西

1. **加载耗时**——要进冷启动预算，不能是 20 秒。
2. **单对延迟**——决定一次查询能重排多少候选（预算 P95 < 200 ms，还要留给别的）。
3. **排序正确性**——用「明显相关」vs「明显无关」两组对照，若模型把它们排反了，
   后面所有评测都是噪声。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_DIR = Path(
    r"C:\Users\Lenovo\WorkBuddy\2026-07-27-16-45-03\roleplay-ai\models\bge-reranker-base-onnx"
)
ONNX_INT8 = MODEL_DIR / "onnx" / "model_int8.onnx"
ONNX_FP32 = MODEL_DIR / "onnx" / "model.onnx"
TOKENIZER = MODEL_DIR / "tokenizer.json"
MAX_LEN = 512


def load():
    return ort.InferenceSession(str(ONNX_INT8), providers=["CPUExecutionProvider"])


def encode_batch(tok: Tokenizer, pairs: list[tuple[str, str]]):
    # `length=None` 不会补齐（tokenizers 只按批内最长补齐时要求显式指定），
    # 所以这里手动按批内最长补到 pad_id=1。补齐必须发生在 numpy 化之前，
    # 否则不同长度的 ids 无法堆成矩形矩阵。
    tok.enable_truncation(max_length=MAX_LEN)
    tok.enable_padding(pad_id=1, pad_token="<pad>", length=None, direction="right")
    encs = [tok.encode(q, d) for q, d in pairs]
    width = max(len(e.ids) for e in encs)
    ids, mask = [], []
    for e in encs:
        pad = width - len(e.ids)
        ids.append(list(e.ids) + [1] * pad)
        mask.append(list(e.attention_mask) + [0] * pad)
    return np.array(ids, dtype=np.int64), np.array(mask, dtype=np.int64)


def score(sess, tok, pairs: list[tuple[str, str]]) -> list[float]:
    ids, mask = encode_batch(tok, pairs)
    feeds = {i.name: v for i, v in zip(sess.get_inputs(), (ids, mask))}
    logits = sess.run(None, feeds)[0]
    # XLMRobertaForSequenceClassification 单标签：logits 形状 (B, 1)，原样即相关性分
    return [float(x) for x in np.asarray(logits).reshape(len(pairs), -1)[:, 0]]


def main() -> int:
    print(f"模型 {ONNX_INT8.name}  {ONNX_INT8.stat().st_size / 1e6:.0f} MB")

    t0 = time.perf_counter()
    tok = Tokenizer.from_file(str(TOKENIZER))
    t_tok = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    sess = load()
    t_load = (time.perf_counter() - t0) * 1000
    print(f"加载：tokenizer {t_tok:.0f} ms，onnx session {t_load:.0f} ms")
    print(f"输入 {[i.name for i in sess.get_inputs()]}  输出 {[o.name for o in sess.get_outputs()]}")

    question = "凯尔希为什么离开罗德岛"
    relevant = [
        "凯尔希在离开罗德岛之前，把博士叫到了办公室。她说自己必须去处理一些事情。",
        "凯尔希：罗德岛需要一个能站在明面上的人，而我更适合待在暗处。",
    ]
    irrelevant = [
        "龙门近卫局的陈在雨里站了很久，雨水顺着她的剑鞘往下流。",
        "整合运动的士兵正在切尔诺伯格的街道上分发传单。",
    ]
    pairs = [(question, d) for d in relevant + irrelevant]
    scores = score(sess, tok, pairs)
    print("\n=== 排序正确性（前两条应显著高于后两条）===")
    for d, s in zip(relevant + irrelevant, scores):
        tag = "相关" if d in relevant else "无关"
        print(f"  {tag}  {s:>9.4f}  {d[:40]}")
    ok = min(scores[:2]) > max(scores[2:])
    print(f"  结论：{'PASS' if ok else 'FAIL —— 分数把相关/无关排反了，不能用'}")

    # ---- 延迟：不同批量 ----
    doc = "凯尔希是罗德岛的医疗主管，也是巴别塔时期的元老之一。" * 8
    print("\n=== 单对/批量延迟（同一问句，%d 对）===" % 64)
    for n in (1, 8, 16, 32, 64):
        batch = [(question, doc)] * n
        score(sess, tok, batch)  # 预热
        reps = max(1, 64 // n)
        t0 = time.perf_counter()
        for _ in range(reps):
            score(sess, tok, batch)
        dt = (time.perf_counter() - t0) * 1000 / reps
        print(f"  n={n:<3} 总 {dt:>8.1f} ms  每对 {dt / n:>6.2f} ms  ({n / (dt / 1000):.0f} 对/s)")

    # ---- 真实块长度下的延迟（本项目 P50 898 字）----
    long_doc = doc * 4
    t0 = time.perf_counter()
    score(sess, tok, [(question, long_doc)] * 16)
    dt = (time.perf_counter() - t0) * 1000
    print(f"\n长文本（约 {len(long_doc)} 字，会截断到 {MAX_LEN} token）16 对：{dt:.0f} ms，每对 {dt/16:.1f} ms")

    json.dump(
        {"model": str(ONNX_INT8), "load_ms": round(t_load), "sort_ok": ok,
         "scores": dict(zip([d[:20] for d in relevant + irrelevant], [round(s, 4) for s in scores]))},
        sys.stdout, ensure_ascii=False, indent=2)
    print()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
