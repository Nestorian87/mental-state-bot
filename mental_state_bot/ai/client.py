from __future__ import annotations

import copy
import hashlib
import json
import time
from typing import Any

import httpx

from mental_state_bot.ai.schemas import ModelCallResult, Usage


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        api_key: str,
        timeout_seconds: int,
        provider_extra: dict[str, Any] | None = None,
    ) -> None:
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.provider_extra = provider_extra or {}

    async def chat(
        self,
        *,
        task_name: str,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        json_schema: dict[str, Any] | None = None,
        thinking: bool = False,
    ) -> ModelCallResult:
        if not self.api_key:
            raise RuntimeError("AI_API_KEY is not configured")

        request_messages = copy.deepcopy(messages)
        payload: dict[str, Any] = {
            "model": model,
            "messages": request_messages,
        }
        if not thinking:
            payload["temperature"] = temperature
        if json_schema is not None:
            _append_json_instruction(request_messages, json_schema)
            if self.provider.lower() == "deepseek":
                payload["response_format"] = {"type": "json_object"}
            else:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": task_name,
                        "strict": True,
                        "schema": json_schema,
                    },
                }

        extra = self.provider_extra.get("thinking_on" if thinking else "thinking_off")
        if isinstance(extra, dict):
            payload.update(extra)
        elif self.provider.lower() == "deepseek":
            payload["thinking"] = {"type": "enabled" if thinking else "disabled"}
            if thinking:
                payload["reasoning_effort"] = "high"

        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                self._url("/chat/completions"),
                headers=self._headers(),
                json=payload,
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError:
                if json_schema is None:
                    raise
                payload["response_format"] = {"type": "json_object"}
                response = await client.post(
                    self._url("/chat/completions"),
                    headers=self._headers(),
                    json=payload,
                )
                response.raise_for_status()
        latency_ms = int((time.perf_counter() - started) * 1000)
        raw = response.json()
        choice = raw["choices"][0]
        text = choice.get("message", {}).get("content") or ""
        data = _parse_json_object(text) if json_schema is not None else None
        usage = _parse_usage(raw.get("usage") or {})
        return ModelCallResult(
            provider=self.provider,
            model=model,
            task_name=task_name,
            text=text,
            data=data,
            usage=usage,
            latency_ms=latency_ms,
            raw=raw,
        )

    async def embed(self, *, model: str, text: str) -> ModelCallResult:
        if not self.api_key:
            raise RuntimeError("Embedding API key is not configured")

        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                self._url("/embeddings"),
                headers=self._headers(),
                json={"model": model, "input": text},
            )
            response.raise_for_status()
        latency_ms = int((time.perf_counter() - started) * 1000)
        raw = response.json()
        embedding = raw["data"][0]["embedding"]
        usage = _parse_usage(raw.get("usage") or {})
        return ModelCallResult(
            provider=self.provider,
            model=model,
            task_name="embed",
            text="",
            data={"embedding": embedding},
            usage=usage,
            latency_ms=latency_ms,
            raw=raw,
        )

    def request_hash(self, payload: Any) -> str:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"


def _parse_usage(raw_usage: dict[str, Any]) -> Usage:
    completion_details = raw_usage.get("completion_tokens_details") or {}
    return Usage(
        prompt_tokens=raw_usage.get("prompt_tokens"),
        completion_tokens=raw_usage.get("completion_tokens"),
        reasoning_tokens=completion_details.get("reasoning_tokens") or raw_usage.get("reasoning_tokens"),
        total_tokens=raw_usage.get("total_tokens"),
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _append_json_instruction(messages: list[dict[str, str]], json_schema: dict[str, Any]) -> None:
    if not messages:
        return
    messages[-1]["content"] += (
        "\n\nПоверни тільки валідний JSON object без Markdown. "
        "JSON має відповідати цій JSON Schema:\n"
        + json.dumps(json_schema, ensure_ascii=False)
    )
