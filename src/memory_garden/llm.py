"""OpenAI 兼容客户端：chat.completions 工具循环 + embeddings，带重试与密钥脱敏。"""
from __future__ import annotations

import json
import math
import re
import time
from typing import Any
from urllib.parse import urlparse

import httpx

from .budget import remaining
from .config import Settings
from .context_budget import budget_prompt

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


def redact(message: str, api_key: str | None) -> str:
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    return message[:1200]


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    return match.group(1).strip() if match else stripped


class OpenAICompatibleClient:
    def __init__(
        self,
        settings: Settings,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        embedding_model: str | None = None,
        embedding_dimension: int | None = None,
    ):
        self.settings = settings
        self.base_url = settings.llm_base_url if base_url is None else base_url
        self.api_key = settings.llm_api_key if api_key is None else api_key
        self.embedding_model = (
            settings.llm_embedding_model if embedding_model is None else embedding_model
        )
        self.embedding_dimension = (
            settings.llm_embedding_dimension
            if embedding_dimension is None
            else embedding_dimension
        )
        self._http_client: httpx.Client | None = None
        # 仅保存聚合计数，不保存 prompt、响应正文或密钥。Agent/评测可用快照差值
        # 记录真实 token 与网络开销，而不会把私人内容复制进额外日志。
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_requests = 0
        self.total_network_latency_ms = 0
        self.total_context_requests = 0
        self.context_budget_events: list[dict[str, Any]] = []

    def usage_snapshot(self) -> tuple[int, int, int, int]:
        return (
            self.total_prompt_tokens,
            self.total_completion_tokens,
            self.total_requests,
            self.total_network_latency_ms,
        )

    def usage_since(self, before: tuple[int, int, int, int]) -> dict[str, int]:
        after = self.usage_snapshot()
        return {
            "prompt_tokens": max(0, after[0] - before[0]),
            "completion_tokens": max(0, after[1] - before[1]),
            "requests": max(0, after[2] - before[2]),
            "network_latency_ms": max(0, after[3] - before[3]),
        }

    def _record_chat_usage(self, payload: dict[str, Any], elapsed_ms: int) -> None:
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            usage = {}

        def non_negative_int(*names: str) -> int:
            for name in names:
                try:
                    return max(0, int(usage.get(name) or 0))
                except (TypeError, ValueError):
                    continue
            return 0

        self.total_prompt_tokens += non_negative_int("prompt_tokens", "input_tokens")
        self.total_completion_tokens += non_negative_int(
            "completion_tokens", "output_tokens"
        )
        self.total_network_latency_ms += max(0, elapsed_ms)

    def _network_client(self) -> httpx.Client:
        if self._http_client is None:
            self._http_client = httpx.Client(timeout=self.settings.llm_timeout_seconds)
        return self._http_client

    def close(self) -> None:
        if self._http_client is not None:
            self._http_client.close()
            self._http_client = None

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise LLMError("尚未配置 API Key")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _url(self, suffix: str) -> str:
        if not self.base_url:
            raise LLMError("尚未配置 API Base URL")
        return f"{self.base_url.rstrip('/')}/{suffix.lstrip('/')}"

    def _post_json_with_retries(
        self, suffix: str, body: dict[str, Any], operation: str
    ) -> dict[str, Any]:
        """POST JSON with the same bounded retry policy used by chat calls."""
        last_error: Exception | None = None
        for attempt in range(self.settings.llm_max_retries + 1):
            try:
                response = self._network_client().post(
                    self._url(suffix), headers=self._headers(), json=body,
                    timeout=remaining(self.settings.llm_timeout_seconds),
                )
                if response.status_code in _RETRYABLE_STATUS:
                    raise httpx.HTTPStatusError(
                        f"retryable {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("响应顶层必须是对象")
                return payload
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in _RETRYABLE_STATUS:
                    detail = exc.response.text[:600]
                    raise LLMError(
                        redact(
                            f"{operation} 请求失败（HTTP {exc.response.status_code}）："
                            f"{detail}",
                            self.api_key,
                        )
                    ) from exc
                last_error = exc
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
                self.close()
            except (json.JSONDecodeError, ValueError) as exc:
                raise LLMError(f"{operation} 响应不是有效 JSON 对象") from exc
            if attempt < self.settings.llm_max_retries:
                time.sleep(min(0.5 * (2**attempt), remaining(self.settings.llm_timeout_seconds)))
        raise LLMError(redact(f"{operation} 请求失败：{last_error}", self.api_key))

    def chat(self, body: dict[str, Any], timeout_seconds: float | None = None) -> dict[str, Any]:
        """一次 chat.completions 调用，带指数退避重试。返回原始 JSON。

        timeout_seconds 可覆盖默认超时：Harness 用剩余预算收紧单次调用，
        防止流式思考输出不断重置读超时导致整体护栏失效。
        """
        timeout = timeout_seconds or self.settings.llm_timeout_seconds
        if (self.settings.llm_reasoning_effort and "reasoning_effort" not in body
                and body.get('thinking') != {'type': 'disabled'}):
            # DeepSeek Chat Completions uses thinking.type to disable reasoning;
            # reasoning_effort=none belongs to other API formats/providers.
            if (urlparse(self.base_url or '').hostname == 'api.deepseek.com'
                    and self.settings.llm_reasoning_effort == 'none'):
                body = {**body, 'thinking': {'type': 'disabled'}}
            else:
                body = {**body, "reasoning_effort": self.settings.llm_reasoning_effort}
        budgeted = budget_prompt(body, self.settings.llm_max_input_chars)
        self.total_context_requests += 1
        self.context_budget_events.append({'request_number': self.total_context_requests, **budgeted.audit})
        self.context_budget_events = self.context_budget_events[-32:]
        if budgeted.after_chars > budgeted.max_chars:
            raise LLMError('当前输入超过上下文字符预算；已缩减可恢复的工具结果，'
                           '但系统说明、工具定义和用户原话仍需保留。请缩小问题范围或提高 MG_LLM_MAX_INPUT_CHARS。')
        body = budgeted.body
        last_error: Exception | None = None
        started = time.monotonic()
        deadline = started + timeout
        for attempt in range(self.settings.llm_max_retries + 1):
            try:
                attempt_timeout = remaining(deadline - time.monotonic())
                with httpx.Client(timeout=attempt_timeout) as client:
                    self.total_requests += 1
                    response = client.post(
                        self._url("chat/completions"), headers=self._headers(), json=body
                    )
                    if response.status_code in _RETRYABLE_STATUS:
                        raise httpx.HTTPStatusError(
                            f"retryable {response.status_code}", request=response.request,
                            response=response,
                        )
                    response.raise_for_status()
                    payload = response.json()
                    if not isinstance(payload, dict):
                        raise ValueError("响应顶层必须是对象")
                    self._record_chat_usage(
                        payload, int((time.monotonic() - started) * 1000)
                    )
                    return payload
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in _RETRYABLE_STATUS:
                    detail = exc.response.text[:600]
                    raise LLMError(
                        redact(f"{exc} | provider detail: {detail}", self.api_key)
                    ) from exc
                last_error = exc
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
            except (json.JSONDecodeError, ValueError) as exc:
                raise LLMError("模型响应不是有效 JSON 对象") from exc
            if attempt < self.settings.llm_max_retries:
                delay = remaining(deadline - time.monotonic())
                time.sleep(min(0.5 * (2**attempt), delay))
        raise LLMError(redact(f"模型调用失败：{last_error}", self.api_key))

    def chat_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
        timeout_seconds: float | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.settings.llm_chat_model,
            "messages": messages,
            "temperature": 0.2,
        }
        if tools:
            body["tools"] = [{"type": "function", "function": tool} for tool in tools]
            body["tool_choice"] = tool_choice or "auto"
            if tool_choice and tool_choice != 'auto' and urlparse(self.base_url or '').hostname == 'api.deepseek.com':
                # Forced structured decisions/final replies require DeepSeek non-thinking mode.
                body['thinking'] = {'type': 'disabled'}
        raw = self.chat(body, timeout_seconds=timeout_seconds)
        try:
            return raw["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"模型响应缺少 message：{redact(str(raw)[:400], self.api_key)}") from exc

    def chat_json(self, system: str, user: str) -> dict[str, Any]:
        message = self.chat_with_tools(
            [{"role": "system", "content": system}, {"role": "user", "content": user}], []
        )
        content = str(message.get("content") or "")
        try:
            parsed = json.loads(strip_code_fence(content))
        except json.JSONDecodeError as exc:
            raise LLMError(f"无法解析模型 JSON：{exc}") from exc
        if not isinstance(parsed, dict):
            raise LLMError("模型 JSON 顶层不是对象")
        return parsed

    # ── embeddings ─────────────────────────────────────────────────────
    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embed_batch(texts)

    def embed_one(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.embedding_model:
            raise LLMError("尚未配置 MG_LLM_EMBEDDING_MODEL")
        payload = self._post_json_with_retries(
            "embeddings",
            {"model": self.embedding_model, "input": texts},
            "embedding",
        )
        try:
            data = sorted(payload["data"], key=lambda item: int(item["index"]))
            if len(data) != len(texts) or [int(item["index"]) for item in data] != list(
                range(len(texts))
            ):
                raise ValueError("返回条数或 index 与请求不一致")
            vectors: list[list[float]] = []
            for item in data:
                raw_vector = item["embedding"]
                if not isinstance(raw_vector, list) or not raw_vector:
                    raise ValueError("embedding 必须是非空数组")
                vector = [float(value) for value in raw_vector]
                if not all(math.isfinite(value) for value in vector):
                    raise ValueError("embedding 包含非有限数值")
                vectors.append(vector)
            dimensions = {len(vector) for vector in vectors}
            if len(dimensions) != 1:
                raise ValueError("同一响应中的向量维度不一致")
            configured_dim = self.embedding_dimension
            if configured_dim > 0 and dimensions != {configured_dim}:
                raise ValueError(
                    f"返回维度 {next(iter(dimensions))} 与 MG_LLM_EMBEDDING_DIMENSION="
                    f"{configured_dim} 不一致"
                )
            return vectors
        except (KeyError, TypeError, ValueError) as exc:
            raise LLMError(f"embedding 响应格式异常：{exc}") from exc

    # ── cross-encoder rerank ─────────────────────────────────────────
    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        model: str,
    ) -> list[tuple[int, float]]:
        """Call the SiliconFlow-compatible /rerank contract and validate every item."""
        if not query.strip():
            raise LLMError("rerank query 不能为空")
        if not documents:
            return []
        if not model:
            raise LLMError("尚未配置 MG_RERANKER_MODEL")
        body = {
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
            "return_documents": False,
        }
        payload = self._post_json_with_retries("rerank", body, "rerank")
        try:
            raw_results = payload["results"]
            if not isinstance(raw_results, list):
                raise ValueError("results 必须是数组")
            results: list[tuple[int, float]] = []
            for item in raw_results:
                index = int(item["index"])
                score = float(item["relevance_score"])
                if not 0 <= index < len(documents):
                    raise ValueError(f"index={index} 越界")
                if not math.isfinite(score):
                    raise ValueError("relevance_score 不是有限数值")
                results.append((index, score))
            indices = [index for index, _score in results]
            if len(results) != len(documents) or set(indices) != set(range(len(documents))):
                raise ValueError("返回条数、index 或唯一性与请求不一致")
            return sorted(results, key=lambda item: (-item[1], item[0]))
        except (KeyError, TypeError, ValueError) as exc:
            raise LLMError(f"rerank 响应格式异常：{exc}") from exc


class EmbeddingClientAdapter:
    """把 OpenAICompatibleClient 适配为 VectorRetriever 需要的 embed/embed_one 接口。"""

    def __init__(self, client: OpenAICompatibleClient, provider: str = "api"):
        self.client = client
        self.provider = provider
        self.model = client.embedding_model
        self.dimension = client.embedding_dimension or None
        self.is_cloud = True

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.client.embed(texts)

    def embed_one(self, text: str) -> list[float]:
        return self.client.embed_one(text)
