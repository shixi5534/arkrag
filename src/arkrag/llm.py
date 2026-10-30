"""云端抽取调用：按错误类型分类的重试、限流与熔断。

## 为什么重写（计划 C5）

旧代码把六种性质完全不同的故障压成两个布尔：

    fatal = exc.code in (401, 402, 403, 429) or any(marker)
    except (URLError, TimeoutError, ConnectionError, OSError):
        return Outcome(None, ..., True, ...)      # ← 断网直接判永久故障

后果：

* **429 被当永久错误**：限流本是「等一会儿就能过」的可恢复状态，旧逻辑立刻
  开熔断、整批任务停摆。而按计划要求，429 应读 `Retry-After`、带抖动退避、
  有界重试，并且**不开永久熔断**。
* **断网被当永久错误**：网络抖动一下就把整条流水线熔断，运维只能手工删标记。
* **5xx 与 429 混在一起**：一个该快速退避、一个该临时熔断，处置方式相反。
* **内容契约错误也走同一路径**：模型输出格式不对是「这条记录坏了」，不是
  「系统坏了」——它只该隔离那一条，不该让全局停下。

## 现在的分类

    AUTH        401/403/无效模型/余额不足  → 立即失败，开**永久**熔断，要求人介入
    RATE_LIMIT  429                        → Retry-After + 指数退避 + 抖动，有界重试，
                                             不开永久熔断；连续多次才开**临时**熔断
    SERVER      5xx                        → 有界重试；连续多次开**临时**熔断（带到期时间）
    NETWORK     超时/连接/断网              → 有界重试；不立即熔断
    CONTENT     解析失败/空对象/格式不对     → 由调用方隔离该条，不属于系统故障
    RESOURCE    磁盘满/内存不足             → 停止发布 + 显式报警，绝不静默跳过

「永久」与「临时」的区别落在熔断文件里：临时熔断带 `expires_at`，到期即视为未开，
不会像旧标记那样永远卡住。
"""
from __future__ import annotations

import json
import os
import random
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime

from . import atomic, config

# 只保留真正表示「鉴权/额度」的标记。429 相关的词（如 rate limit）不在此列。
AUTH_MARKERS = (
    "insufficient",
    "quota",
    "credit",
    "余额不足",
    "额度",
    "unauthorized",
    "forbidden",
    "invalid api key",
    "authentication",
    "invalid model",
    "model not found",
)

# 计划 C5 的分类名。下游一律用这些常量，禁止再写裸字符串。
AUTH = "auth"
RATE_LIMIT = "rate_limit"
SERVER = "server"
NETWORK = "network"
CONTENT = "content"
RESOURCE = "resource"
UNKNOWN = "unknown"

# 各分类的处置：可重试 / 是否开熔断 / 熔断是永久还是临时
POLICY = {
    AUTH: {"retry": False, "circuit": "permanent"},
    RATE_LIMIT: {"retry": True, "circuit": "transient", "transient_after": 3},
    SERVER: {"retry": True, "circuit": "transient", "transient_after": 4},
    NETWORK: {"retry": True, "circuit": None},
    CONTENT: {"retry": True, "circuit": None},
    RESOURCE: {"retry": False, "circuit": "permanent"},
    UNKNOWN: {"retry": True, "circuit": None},
}

TRANSIENT_TTL_SECONDS = 300


class CircuitOpen(RuntimeError):
    pass


@dataclass
class Outcome:
    raw: dict | None
    error: str
    fatal: bool
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    kind: str = CONTENT
    retry_after: float | None = None
    detail: str = field(default="", repr=False)


def classify_http(code: int, detail: str) -> str:
    """把 HTTP 状态码映射到故障分类。"""
    lowered = detail.lower()
    if code in (401, 402, 403):
        return AUTH
    if code == 429:
        return RATE_LIMIT
    if code == 404:
        # 模型名错通常表现为 404/400 + invalid model，交给下面的标记判断
        return AUTH if any(m in lowered for m in AUTH_MARKERS) else UNKNOWN
    if 500 <= code < 600:
        return SERVER
    if code in (400, 413, 422):
        # 请求本身有问题（上下文过长、参数非法）：重试无用，但也不是鉴权问题
        return RESOURCE if "too long" in lowered or "context" in lowered else CONTENT
    if any(m in lowered for m in AUTH_MARKERS):
        return AUTH
    return UNKNOWN


class InflightGauge:
    """在途请求峰值探针：分片并行时的「全局并发 <= 8」要靠实测，不靠声明。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.current = 0
        self.peak = 0
        self.total_calls = 0

    def __enter__(self) -> "InflightGauge":
        with self._lock:
            self.current += 1
            self.total_calls += 1
            self.peak = max(self.peak, self.current)
        return self

    def __exit__(self, *exc) -> None:
        with self._lock:
            self.current -= 1


# --------------------------------------------------------------------- 熔断

def circuit_state(now: float | None = None) -> dict | None:
    """读熔断状态。**临时熔断过期后视为未开**——旧标记不会永远卡住。

    旧代码读到文件就返回，导致一次网络抖动留下的标记会永久阻塞后续所有运行，
    只能人工删除。这里把 `expires_at` 纳入判断。
    """
    path = config.CIRCUIT_OPEN_PATH
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {"reason": "unreadable_marker", "at": "unknown", "kind": "permanent"}
    if not isinstance(state, dict):
        return {"reason": "unreadable_marker", "at": "unknown", "kind": "permanent"}

    expires = state.get("expires_at")
    if expires:
        current = now if now is not None else time.time()
        if current >= float(expires):
            return None  # 已过期，不再阻塞
    return state


def open_circuit(
    reason: str,
    detail: str = "",
    *,
    kind: str = "permanent",
    ttl_seconds: int = TRANSIENT_TTL_SECONDS,
    generation: str = "",
    owner: str = "",
) -> None:
    """打开熔断并落盘。

    计划 C3 要求熔断状态带 Owner / 时间 / Generation：否则一个残留的
    `circuit_open.json` 无法判断「是谁、在哪一代、什么时候」打开的，运维只能删掉
    重试——等于把「未知原因的系统性故障」降级成「重启一下」。

    `kind="transient"` 时写 `expires_at`，到期自动失效（计划 C5：5xx/429 用临时熔断）。
    """
    path = config.CIRCUIT_OPEN_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    payload = {
        "reason": reason,
        "detail": detail[:300],
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "owner": owner or os.environ.get("ARKRAG_RUN_OWNER", "") or f"pid:{os.getpid()}",
        "generation": generation or os.environ.get("ARKRAG_GENERATION_ID", ""),
        "run_id": os.environ.get("ARKRAG_RUN_ID", ""),
        "kind": kind,
        "schema": 3,
    }
    if kind == "transient":
        payload["expires_at"] = now + ttl_seconds
        payload["expires_in_seconds"] = ttl_seconds
    atomic.atomic_write_json(path, payload)


def clear_circuit() -> bool:
    """显式清除熔断（人工处置后使用）。返回是否真的删掉了文件。"""
    path = config.CIRCUIT_OPEN_PATH
    if path.exists():
        path.unlink()
        return True
    return False


# ------------------------------------------------------------------ 客户端

class LLMClient:
    def __init__(self, gauge: InflightGauge | None = None) -> None:
        self.base_url = config.LLM_BASE_URL.rstrip("/")
        self.model = config.LLM_MODEL
        self.api_key = os.environ.get("ARKRAG_LLM_API_KEY", "")
        self.gauge = gauge or InflightGauge()
        if not self.base_url or not self.model:
            raise RuntimeError("缺少 ARKRAG_LLM_BASE_URL / ARKRAG_LLM_MODEL 配置")

    def _post(self, body: dict, timeout: int = 180) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def complete_json(self, system: str, user: str, temperature: float = 0.1) -> Outcome:
        if circuit_state() is not None:
            raise CircuitOpen("circuit_open")
        body = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": 2048,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        with self.gauge:
            try:
                data = self._post(body)
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", "ignore")
                except Exception:
                    pass
                kind = classify_http(exc.code, detail)
                retry_after = None
                if kind == RATE_LIMIT:
                    retry_after = _parse_retry_after(exc.headers.get("Retry-After") if exc.headers else None)
                return Outcome(
                    None,
                    f"{kind}:http{exc.code}",
                    kind in (AUTH, RESOURCE),
                    "",
                    0,
                    0,
                    kind=kind,
                    retry_after=retry_after,
                    detail=detail[:200],
                )
            except (TimeoutError, socket.timeout) as exc:
                return Outcome(None, f"{NETWORK}:timeout", False, "", 0, 0, kind=NETWORK, detail=str(exc)[:120])
            except (urllib.error.URLError, ConnectionError, OSError) as exc:
                return Outcome(
                    None, f"{NETWORK}:{type(exc).__name__}", False, "", 0, 0, kind=NETWORK, detail=str(exc)[:120]
                )

        choice = (data.get("choices") or [{}])[0]
        usage = data.get("usage") or {}
        message = choice.get("message") or {}
        content = message.get("content") or ""
        if isinstance(content, list):  # 某些网关返回分段数组
            content = "".join(seg.get("text", "") for seg in content if isinstance(seg, dict))
        finish = choice.get("finish_reason") or ""
        parsed = parse_json_object(content)
        if parsed is None:
            return Outcome(
                None,
                f"{CONTENT}:parse_failed",
                False,
                finish,
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
                kind=CONTENT,
                detail=content[:200],
            )
        if not parsed:
            return Outcome(
                None,
                f"{CONTENT}:empty_object",
                False,
                finish,
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
                kind=CONTENT,
            )
        return Outcome(
            parsed, "", False, finish, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        )


def _parse_retry_after(value: str | None) -> float | None:
    """把 `Retry-After` 解析成秒。支持两种格式：秒数与 HTTP 日期。"""
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
        return max(0.0, when.timestamp() - time.time())
    except (TypeError, ValueError):
        return None


def parse_json_object(content: str) -> dict | None:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def backoff_delay(attempt: int, kind: str, retry_after: float | None) -> float:
    """退避时长。

    * 429 优先听服务端的 `Retry-After`（计划 C5 明确要求），再叠加抖动避免惊群。
    * 其余按 2^attempt 指数退避，上限 20 秒，加 0~1 秒抖动。
    """
    if kind == RATE_LIMIT and retry_after is not None:
        return min(60.0, retry_after + random.random())
    base = min(20.0, 2**attempt)
    return base + random.random()


def extract_with_retry(client: LLMClient, system: str, user: str) -> tuple[dict | None, dict]:
    """按分类处置重试。

    * AUTH/RESOURCE：立即返回，开**永久**熔断（要人介入，重试只是浪费）。
    * RATE_LIMIT：有界重试，听 `Retry-After`；连续多次才开**临时**熔断。
    * SERVER：有界重试；连续多次开**临时**熔断。
    * NETWORK：有界重试，**不**开熔断（抖一下不该停整条线）。
    * CONTENT：有界重试，由调用方决定是否隔离该条。
    """
    meta = {
        "attempts": 0,
        "errors": [],
        "kinds": [],
        "fatal": False,
        "circuit_opened": False,
        "circuit_kind": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "finish_reason": "",
        "retry_after": None,
        "waited_seconds": 0.0,
    }
    kind_streak: dict[str, int] = {}

    for attempt in range(config.LLM_RETRIES):
        meta["attempts"] = attempt + 1
        try:
            outcome = client.complete_json(system, user)
        except CircuitOpen:
            meta["circuit_opened"] = True
            meta["errors"].append("circuit_open")
            meta["kinds"].append("circuit")
            return None, meta

        meta["prompt_tokens"] += outcome.prompt_tokens
        meta["completion_tokens"] += outcome.completion_tokens
        meta["finish_reason"] = outcome.finish_reason
        if outcome.raw is not None:
            return outcome.raw, meta

        kind = outcome.kind or UNKNOWN
        meta["errors"].append(outcome.error or "unknown")
        meta["kinds"].append(kind)
        meta["retry_after"] = outcome.retry_after
        kind_streak[kind] = kind_streak.get(kind, 0) + 1

        policy = POLICY.get(kind, POLICY[UNKNOWN])

        # AUTH / RESOURCE：不重试，永久熔断，要求人介入
        if not policy["retry"]:
            meta["fatal"] = True
            meta["circuit_opened"] = True
            meta["circuit_kind"] = "permanent"
            open_circuit(kind, outcome.error, kind="permanent")
            return None, meta

        # 连续同类的可恢复错误达到阈值 -> 临时熔断
        threshold = policy.get("transient_after")
        if policy["circuit"] == "transient" and threshold and kind_streak[kind] >= threshold:
            meta["circuit_opened"] = True
            meta["circuit_kind"] = "transient"
            open_circuit(kind, outcome.error, kind="transient")
            return None, meta

        if attempt + 1 < config.LLM_RETRIES:
            delay = backoff_delay(attempt, kind, outcome.retry_after)
            meta["waited_seconds"] += delay
            time.sleep(delay)

    return None, meta


def circuit_summary() -> dict:
    """给状态查询用：当前熔断是什么性质、谁开的、还剩多久。"""
    state = circuit_state()
    if state is None:
        return {"open": False}
    summary = {
        "open": True,
        "kind": state.get("kind", "permanent"),
        "reason": state.get("reason"),
        "owner": state.get("owner"),
        "generation": state.get("generation"),
        "at": state.get("at"),
    }
    expires = state.get("expires_at")
    if expires:
        summary["expires_in_seconds"] = max(0.0, float(expires) - time.time())
    return summary
