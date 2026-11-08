"""交叉编码器重排（CE rerank）：把 (问句, 块正文) 拼起来过一遍模型，直接出相关性分。

## 为什么要这个通道

`scripts/diagnose_channels.py` 的读数是：bm25 单路取前 100 时 gold 命中率 0.9367，
但 R@20 只有 0.6478。**gold 基本都在候选池里，问题只在名次**。RRF 与加权 RRF 都
救不回来（只用名次的东西，名次一旦被弱通道污染就没有信息可恢复）。交叉编码器不看
两路名次，只读原始文本，是对的工具。

## 这是对「运行时零第三方依赖」的**明确破例**

`requirements-dev.txt` 写着「运行时零第三方依赖：`src/` 与 `scripts/` 只用标准库」。
本模块破了这条：重排必须跑 ONNX 模型，离不开 `onnxruntime` 与 `tokenizers`。

**破例理由**：重排的本质是把问句和正文拼成一个序列过 12 层 Transformer，纯标准库
做不到——这不是「实现得慢一点」的问题，是没有实现。而本项目已经证明：不重排时
gold 块就在池子里却进不了前 20，这不是调参能修的缺陷。

**破了之后必须跟着改的门（不写清就会变成静默降级）**：

1. **通道申报要显式**。与 `vector_store` 的 `vector_channels: "not_built_no_numpy"`
   同一条纪律：谁不装 `onnxruntime`/`tokenizers`，就必须在评测报告里把重排通道
   申报为缺席，不许让调用方 `try: ... except: pass` 掉。本模块的 `RerankerUnavailable`
   就是给这个申报用的——它是**硬错误**，不是可以吞掉的警告。
2. **冷启动预算要重算**。实测（本机，见下）tokenizer 加载 0.6–0.7 s、int8 session
   0.3–0.5 s、fp32 session 2 s 级，**首次推理还要额外 1–3 s**（ORT 首次执行才会为
   该形状分配/初始化 kernel）。所以生产必须在启动期构造一次 `warmup()`，不许把
   构造放在请求路径上——否则第一个用户吃满冷启动。
3. **`requirements-dev.txt` 的那句注释需要补一行例外申报**（由项目负责人改，本模块
   无权改它）。否则下一个人读到「零第三方依赖」会以为本文件违规。
4. **模型不在仓库里**（在本机 `roleplay-ai` 项目下的 `models/bge-reranker-base-onnx/`），所以
   `tests/test_portable_paths.py` 扫不到绝对路径，换机器就会失败在运行期而不是
   测试期。任何写入 JSON 的模型身份都必须是**目录名 + 权重文件名 + 加载耗时**，
   不许写绝对路径。

## 长度分桶（为什么不是整批 padding）

本项目的块正文长度**极度集中**：13,798 块里 11,445 块落在 900–1000 字；转成 token
后 P50 = 645、P95 = 702。所以在 `max_length=512` 下几乎所有对都截到 512，
分桶在这个语料上收益很小——**这是实测结论，不是假设**（见
`scripts/probe_rerank_opt.py` 的 `bucket` 行）。但分桶仍然默认开着：它的成本是
一次 `sorted`（20 条数据上可忽略），而在「候选里有短块」的历史会话类查询上它不亏。
桶宽默认 32 token：更细的桶省下的 padding 不到 1 次 attention 的开销，更粗的桶
等于没分。

## 分数方向

原样 logit，**不做 sigmoid**。理由有两条：一是排序不受单调变换影响，sigmoid 只会
丢掉分辨率；二是仓库里已有的读数（`docs/i阶段实测.md` §5.1：相关对 ~4.9951、
无关对 ~-7.6255）都是 logit 尺度，改成概率就再也对不上。
"""

from __future__ import annotations

import time
from pathlib import Path

#: 默认截断长度。取 512 是因为模型 `max_position_embeddings=514`（含 <s> 与 </s>），
#: 再长没有位置编码可用；而项目的真实块正文截到 512 时排序质量最好（ρ 基准）。
DEFAULT_MAX_LENGTH = 512

#: 默认批大小。实测 batch 16 已经吃满 AVX 的收益，32 只多占内存（见 probe 的 batch 行）。
DEFAULT_BATCH_SIZE = 16

#: 长度分桶的桶宽（token）。`None` 表示不分桶——整批 padding 到最长，只用于对照实验。
DEFAULT_BUCKET_WIDTH = 32

_TOKENIZER_FILE = "tokenizer.json"
_CONFIG_FILE = "config.json"
_ONNX_SUBDIR = "onnx"
_FP32_WEIGHTS = "model.onnx"
_INT8_WEIGHTS = "model_int8.onnx"


class RerankerError(RuntimeError):
    """重排失败。**调用方必须当作硬错误处理，不许降级成"没有重排"。**

    这个项目因为静默降级出过多次事故：空的向量通道让 BM25 悄悄退化、假的分块
    结果让评测报告全绿。所以重排的每一条失败路径（缺文件、建不起 session、
    logits 维度不对、输入不是字符串）都在这里抛，一条都不许吞。
    """


class RerankerUnavailable(RerankerError):
    """重排通道不可用（缺 `onnxruntime` / `tokenizers`，或缺模型权重文件）。

    语义与 `vector_store` 的 `not_built_no_numpy` 相同：这不是"降级跑"，是"这个
    通道不存在"。调用方要么显式申报通道缺席（评测报告里必须出现这个字段），
    要么装上依赖——不许 `except: pass` 之后假装重排过了。
    """


def runtime_available() -> bool:
    """重排的第三方依赖是否齐全。给「通道申报」用，不抛异常。"""
    try:
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
    except ImportError:
        return False
    return True


def _import_runtime():
    """惰性导入 onnxruntime / tokenizers / numpy。

    顶层**不许**无条件 import：`src/arkrag/` 的其他模块（检索、融合、发布）在没装
    这两个包的机器上也必须能 import 成功——否则一个可选通道会把整个包拖成不可导入。
    """
    missing = []
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - 环境相关
        np = None
        missing.append("numpy")
    try:
        import onnxruntime as ort
    except ImportError:  # pragma: no cover - 环境相关
        ort = None
        missing.append("onnxruntime")
    try:
        from tokenizers import Tokenizer
    except ImportError:  # pragma: no cover - 环境相关
        Tokenizer = None
        missing.append("tokenizers")
    if missing:
        raise RerankerUnavailable(
            f"重排通道缺少依赖：{', '.join(missing)}。"
            f"重排是可选通道，缺失时必须由调用方显式申报为不可用"
            f"（参考 vector_store 的 not_built_no_numpy），"
            f"不许静默降级成'没有重排'——那会让评测报告全绿而召回悄悄掉一截。"
        )
    return ort, np, Tokenizer


class CrossEncoderReranker:
    """bge-reranker（XLMRobertaForSequenceClassification）的 ONNX 推理封装。

    构造即加载：tokenizer 与 onnx session 都在 `__init__` 里建好，任何一步失败都当场
    抛 `RerankerUnavailable`（带具体路径）。**不做惰性加载**——惰性加载会把
    "模型坏了"推迟到第一次查询，那时调用方已经在等服务了，最容易写出一句
    `except: pass`。

    方法：
    * `score_pairs(pairs)` → 原始 logit，越高越相关，顺序与入参一致。
    * `rerank(query, docs, top_n=...)` → **降序的原始下标**。不返回正文，避免
      调用方为了拿回原文再复制一份 900 字的块。
    * `warmup()` → 强制跑一次短推理，吃掉 ORT 的首形状开销。
    * `info()` → 自述，供评测报告落盘。
    """

    def __init__(
        self,
        model_dir,
        *,
        max_length: int = DEFAULT_MAX_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
        intra_op_threads: int | None = None,
        use_int8: bool = False,
        bucket_width: int | None = DEFAULT_BUCKET_WIDTH,
    ) -> None:
        ort, np, Tokenizer = _import_runtime()

        self._dir = Path(model_dir)
        if not self._dir.is_dir():
            raise RerankerUnavailable(f"重排模型目录不存在：{self._dir}")

        if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length <= 0:
            raise RerankerError(f"max_length 必须是正整数，收到 {max_length!r}")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise RerankerError(f"batch_size 必须是正整数，收到 {batch_size!r}")
        if bucket_width is not None and (
            not isinstance(bucket_width, int) or isinstance(bucket_width, bool) or bucket_width <= 0
        ):
            raise RerankerError(f"bucket_width 必须是正整数或 None，收到 {bucket_width!r}")
        if intra_op_threads is not None and (
            not isinstance(intra_op_threads, int)
            or isinstance(intra_op_threads, bool)
            or intra_op_threads <= 0
        ):
            raise RerankerError(f"intra_op_threads 必须是正整数或 None，收到 {intra_op_threads!r}")

        self._max_length = max_length
        self._batch_size = batch_size
        self._bucket_width = bucket_width
        self._intra_op_threads = intra_op_threads
        self._use_int8 = bool(use_int8)

        # ---- 权重与 tokenizer 的路径解析 -------------------------------------------------
        # 模型包有两种常见落法（`<dir>/onnx/model.onnx` 与 `<dir>/model.onnx`），两种都认，
        # 但必须在 info() 里报出**实际用了哪一个**，否则事后查"我跑的到底是不是 int8"没有证据。
        weights_name = _INT8_WEIGHTS if self._use_int8 else _FP32_WEIGHTS
        candidates = [self._dir / _ONNX_SUBDIR / weights_name, self._dir / weights_name]
        weights = next((p for p in candidates if p.is_file()), None)
        if weights is None:
            raise RerankerUnavailable(
                f"找不到权重文件 {weights_name}（use_int8={self._use_int8}）。"
                f"找过：{[str(p) for p in candidates]}。"
                f"重排通道必须显式申报不可用，不许降级成'没有重排'。"
            )
        tokenizer_path = self._dir / _TOKENIZER_FILE
        if not tokenizer_path.is_file():
            raise RerankerUnavailable(f"找不到 tokenizer：{tokenizer_path}")

        self._weights_path = weights
        self._tokenizer_path = tokenizer_path
        self._pad_id, self._max_positions = self._read_model_config()

        t_all = time.perf_counter()

        # ---- tokenizer ----------------------------------------------------------------
        t0 = time.perf_counter()
        try:
            tok = Tokenizer.from_file(str(tokenizer_path))
        except Exception as exc:  # tokenizers 抛的是 Exception 子类，具体类型随版本变
            raise RerankerUnavailable(
                f"tokenizer 加载失败：{tokenizer_path}（{type(exc).__name__}: {exc}）"
            ) from exc
        # 截断在 encode 时生效；padding 由本模块自己做（见 `_pad_batch`），
        # 因为分桶要求"桶内统一宽度"，而 tokenizers 的 `enable_padding(length=None)`
        # 只会按**整批**最长补齐，那正是我们要避免的行为。
        tok.enable_truncation(max_length=self._max_length)
        tok.no_padding()
        self._tok = tok
        self._tokenizer_load_ms = (time.perf_counter() - t0) * 1000.0

        # ---- onnx session -------------------------------------------------------------
        t0 = time.perf_counter()
        options = ort.SessionOptions()
        if intra_op_threads is not None:
            # 只设 intra_op：inter_op 是图层面的并发，交叉编码器是单链，开了只是多线程抢核。
            options.intra_op_num_threads = intra_op_threads
        try:
            sess = ort.InferenceSession(
                str(weights), options, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            raise RerankerUnavailable(
                f"onnx session 建不起来：{weights}（{type(exc).__name__}: {exc}）"
            ) from exc
        self._sess = sess
        self._session_load_ms = (time.perf_counter() - t0) * 1000.0

        self._input_names = [i.name for i in sess.get_inputs()]
        missing_inputs = {"input_ids", "attention_mask"} - set(self._input_names)
        if missing_inputs:
            raise RerankerError(
                f"模型输入不符合交叉编码器契约：缺 {sorted(missing_inputs)}，"
                f"实际输入 {self._input_names}（{weights}）。"
                f"本模块只喂 input_ids/attention_mask，不做 token_type_ids 之类的适配——"
                f"喂错名字比报错更危险，ORT 可能静默忽略。"
            )
        self._providers = list(sess.get_providers())
        self._ort_version = ort.__version__
        import tokenizers as _tk  # 已经导入过，这里只为拿版本号

        self._tokenizers_version = _tk.__version__
        self._np = np
        self._load_ms = (time.perf_counter() - t_all) * 1000.0

    # ------------------------------------------------------------------ 配置与校验

    def _read_model_config(self) -> tuple[int, int | None]:
        """读 config.json 拿 pad_token_id 与最大位置数。读不到不致命，但要如实报。"""
        self._config_error = ""
        import json

        path = self._dir / _CONFIG_FILE
        pad_id, max_positions = 1, None
        try:
            cfg = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self._config_error = f"缺 {path}，pad_token_id 回落到 1"
            return pad_id, max_positions
        except Exception as exc:  # 坏 JSON
            self._config_error = f"{path} 读不了（{type(exc).__name__}: {exc}），pad_token_id 回落到 1"
            return pad_id, max_positions
        got_pad = cfg.get("pad_token_id")
        if isinstance(got_pad, int) and not isinstance(got_pad, bool):
            pad_id = got_pad
        got_max = cfg.get("max_position_embeddings")
        if isinstance(got_max, int) and not isinstance(got_max, bool):
            max_positions = got_max
            # 位置编码上限是硬约束：超了模型不是变慢，是直接算错（索引越界或广播错）。
            if self._max_length > max_positions:
                raise RerankerError(
                    f"max_length={self._max_length} 超过模型 max_position_embeddings="
                    f"{max_positions}（{path}）。超长会被位置编码截断成错的结果，"
                    f"不报错的模型更危险——请改小 max_length。"
                )
        return pad_id, max_positions

    @staticmethod
    def _validate_pairs(pairs) -> list[tuple[str, str]]:
        if not isinstance(pairs, (list, tuple)):
            raise RerankerError(f"pairs 必须是 (query, doc) 的列表，收到 {type(pairs).__name__}")
        out: list[tuple[str, str]] = []
        for i, item in enumerate(pairs):
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise RerankerError(f"第 {i} 个 pair 不是二元组：{item!r}")
            q, d = item
            if not isinstance(q, str) or not isinstance(d, str):
                # 静默 str() 化会把 None 变成 "None" 喂进模型，打出一个看似正常的分。
                raise RerankerError(
                    f"第 {i} 个 pair 含非字符串：query={type(q).__name__}, doc={type(d).__name__}"
                )
            out.append((q, d))
        return out

    # ------------------------------------------------------------------ 编码与分桶

    def _encode(self, pairs: list[tuple[str, str]]) -> list[tuple[list[int], list[int]]]:
        """把每对编码成 (ids, attention_mask)，**不 padding**。"""
        encs = []
        for q, d in pairs:
            enc = self._tok.encode(q, d)
            encs.append((list(enc.ids), list(enc.attention_mask)))
        return encs

    def _plan_buckets(self, encs: list[tuple[list[int], list[int]]]) -> list[tuple[list[int], int]]:
        """把下标按 token 长度分桶，返回 [(桶内下标列表, 该桶的 padding 宽度)]。

        为什么按长度排序后分桶、而不是整批 padding 到最长：块正文长度差一个数量级的
        时候，把 120 字的块补到 700 token 是纯粹浪费（attention 是 O(L²)）。本语料
        上这个浪费其实很小（块长极度集中），但成本只有一次 sorted，不亏。

        桶宽的作用是让**同一桶内的 padding 宽度一致**，从而使 ORT 反复看到少数几种
        输入形状——形状稳定的收益比少补几十个 pad token 大。
        """
        if not encs:
            return []
        if self._bucket_width is None:
            # 不分桶：一个桶，宽度取全局最长，即"整批 padding 到最长"。留这个分支是因为
            # 没有对照就没法说分桶到底有没有用（probe 会拿它做消融）。
            width = max(len(ids) for ids, _ in encs)
            return [(list(range(len(encs))), width)]

        groups: dict[int, list[int]] = {}
        for i, (ids, _) in enumerate(encs):
            groups.setdefault(len(ids) // self._bucket_width, []).append(i)
        plan = []
        for key in sorted(groups):
            idxs = groups[key]
            width = max(len(encs[i][0]) for i in idxs)
            plan.append((idxs, width))
        return plan

    def _pad_batch(self, encs, idxs: list[int], width: int):
        """把一组编码补成 (B, width) 的两个 int64 矩阵。右侧补齐，与 XLM-R 惯例一致。"""
        np = self._np
        b = len(idxs)
        ids = np.full((b, width), self._pad_id, dtype=np.int64)
        mask = np.zeros((b, width), dtype=np.int64)
        for row, i in enumerate(idxs):
            tok_ids, tok_mask = encs[i]
            n = len(tok_ids)
            if n > width:  # 不可能发生（width 是该桶内最长），发生即说明计划坏了
                raise RerankerError(f"内部错误：第 {i} 条长 {n} 超出桶宽 {width}")
            ids[row, :n] = tok_ids
            mask[row, :n] = tok_mask
        return ids, mask

    # ------------------------------------------------------------------ 推理

    def _run(self, ids, mask) -> list[float]:
        np = self._np
        n = int(ids.shape[0])
        feeds = {"input_ids": ids, "attention_mask": mask}
        out = self._sess.run(None, feeds)
        if not out:
            raise RerankerError("onnx 返回空结果")
        arr = np.asarray(out[0])
        # 维度不对必须当场炸：reshape 一把能"凑出"数字，但那是把形状错误变成随机分数。
        if arr.ndim != 2 or arr.shape[0] != n or arr.shape[1] < 1:
            raise RerankerError(
                f"logits 形状不符：期望 ({n}, >=1)，收到 {tuple(arr.shape)}。"
                f"本模块只支持单标签 XLMRobertaForSequenceClassification。"
            )
        return [float(x) for x in arr[:, 0]]

    def score_pairs(self, pairs) -> list[float]:
        """给 (query, doc) 对打分。返回原始 logit（越高越相关），顺序与入参一致。"""
        pairs = self._validate_pairs(pairs)
        if not pairs:
            return []
        encs = self._encode(pairs)
        scores = [0.0] * len(pairs)
        for idxs, width in self._plan_buckets(encs):
            for start in range(0, len(idxs), self._batch_size):
                chunk = idxs[start : start + self._batch_size]
                ids, mask = self._pad_batch(encs, chunk, width)
                got = self._run(ids, mask)
                for row, original in enumerate(chunk):
                    scores[original] = got[row]
        return scores

    def rerank(self, query: str, docs, *, top_n: int | None = None) -> list[int]:
        """重排：返回按分数**降序**的 `docs` 原始下标列表。

        返回下标而不是正文，是为了让调用方零拷贝地映射回自己的对象——本项目一块正文
        P50 883 字，复制 20 份纯属浪费，更别说复制会让"重排后的文本"与索引里的块
        失去可比性。

        `top_n` 只截断返回长度，不改变打分；同分保持原顺序（用 `-score, index` 排序），
        避免分数相同的两块在两次调用间来回跳。
        """
        if not isinstance(query, str):
            raise RerankerError(f"query 必须是字符串，收到 {type(query).__name__}")
        if not isinstance(docs, (list, tuple)):
            raise RerankerError(f"docs 必须是字符串列表，收到 {type(docs).__name__}")
        if top_n is not None and (
            not isinstance(top_n, int) or isinstance(top_n, bool) or top_n <= 0
        ):
            raise RerankerError(f"top_n 必须是正整数或 None，收到 {top_n!r}")
        if not docs:
            return []
        scores = self.score_pairs([(query, d) for d in docs])
        order = sorted(range(len(docs)), key=lambda i: (-scores[i], i))
        return order[:top_n] if top_n is not None else order

    # ------------------------------------------------------------------ 运维口

    def warmup(self) -> float:
        """跑一次短推理，迫使 ORT 完成该形状的 kernel 初始化。返回耗时 ms。

        为什么必须有这个：本机实测同一个 session 连测 12 次，前 3 次是
        5337 / 5337 / 3777 ms，之后稳定在 ~2050 ms。前几次不是噪声，是首次执行
        与 CPU 升频的真实成本。不预热量出来的 P50 是假的。
        """
        t0 = time.perf_counter()
        self.score_pairs([("预热", "预热")])
        return (time.perf_counter() - t0) * 1000.0

    def info(self) -> dict:
        """自述，供评测报告落盘。**不含绝对路径**（入库 JSON 不许有机台路径）。"""
        return {
            "model": self._dir.name,
            "weights": self._weights_path.name,
            "weights_on_disk": f"{_ONNX_SUBDIR}/{self._weights_path.name}"
            if self._weights_path.parent.name == _ONNX_SUBDIR
            else self._weights_path.name,
            "use_int8": self._use_int8,
            "max_length": self._max_length,
            "max_position_embeddings": self._max_positions,
            "batch_size": self._batch_size,
            "bucket_width": self._bucket_width,
            "intra_op_threads": self._intra_op_threads,
            "providers": self._providers,
            "pad_token_id": self._pad_id,
            "load_ms": round(self._load_ms, 1),
            "tokenizer_load_ms": round(self._tokenizer_load_ms, 1),
            "session_load_ms": round(self._session_load_ms, 1),
            "onnxruntime": self._ort_version,
            "tokenizers": self._tokenizers_version,
            "config_error": self._config_error,
        }
