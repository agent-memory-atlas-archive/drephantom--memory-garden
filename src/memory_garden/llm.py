"""OpenAI 兼容客户端：chat.completions 工具循环 + embeddings，带重试与密钥脱敏。"""
from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx

from .config import Settings

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
    def __init__(self, settings: Settings):
        self.settings = settings

    def _headers(self) -> dict[str, str]:
        if not self.settings.llm_api_key:
            raise LLMError("尚未配置 MG_LLM_API_KEY")
        return {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
        }

    def _url(self, suffix: str) -> str:
        if not self.settings.llm_base_url:
            raise LLMError("尚未配置 MG_LLM_BASE_URL")
        return f"{self.settings.llm_base_url.rstrip('/')}/{suffix.lstrip('/')}"

    def chat(self, body: dict[str, Any], timeout_seconds: float | None = None) -> dict[str, Any]:
        """一次 chat.completions 调用，带指数退避重试。返回原始 JSON。

        timeout_seconds 可覆盖默认超时：Harness 用剩余预算收紧单次调用，
        防止流式思考输出不断重置读超时导致整体护栏失效。
        """
        timeout = timeout_seconds or self.settings.llm_timeout_seconds
        if self.settings.llm_reasoning_effort and "reasoning_effort" not in body:
            body = {**body, "reasoning_effort": self.settings.llm_reasoning_effort}
        last_error: Exception | None = None
        for attempt in range(self.settings.llm_max_retries + 1):
            try:
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(
                        self._url("chat/completions"), headers=self._headers(), json=body
                    )
                    if response.status_code in _RETRYABLE_STATUS:
                        raise httpx.HTTPStatusError(
                            f"retryable {response.status_code}", request=response.request,
                            response=response,
                        )
                    response.raise_for_status()
                    return response.json()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in _RETRYABLE_STATUS:
                    detail = exc.response.text[:600]
                    raise LLMError(
                        redact(f"{exc} | provider detail: {detail}", self.settings.llm_api_key)
                    ) from exc
                last_error = exc
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
            time.sleep(0.5 * (2**attempt))
        raise LLMError(redact(f"模型调用失败：{last_error}", self.settings.llm_api_key))

    def chat_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.settings.llm_chat_model,
            "messages": messages,
            "temperature": 0.2,
        }
        if tools:
            body["tools"] = [{"type": "function", "function": tool} for tool in tools]
            body["tool_choice"] = "auto"
        raw = self.chat(body, timeout_seconds=timeout_seconds)
        try:
            return raw["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"模型响应缺少 message：{redact(str(raw)[:400], self.settings.llm_api_key)}") from exc

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
        if not self.settings.llm_embedding_model:
            raise LLMError("尚未配置 MG_LLM_EMBEDDING_MODEL")
        with httpx.Client(timeout=self.settings.llm_timeout_seconds) as client:
            response = client.post(
                self._url("embeddings"),
                headers=self._headers(),
                json={"model": self.settings.llm_embedding_model, "input": texts},
            )
            response.raise_for_status()
            payload = response.json()
        try:
            data = sorted(payload["data"], key=lambda item: int(item["index"]))
            return [item["embedding"] for item in data]
        except (KeyError, TypeError) as exc:
            raise LLMError(f"embedding 响应格式异常：{exc}") from exc


class EmbeddingClientAdapter:
    """把 OpenAICompatibleClient 适配为 VectorRetriever 需要的 embed/embed_one 接口。"""

    def __init__(self, client: OpenAICompatibleClient):
        self.client = client

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.client.embed(texts)

    def embed_one(self, text: str) -> list[float]:
        return self.client.embed_one(text)
