from __future__ import annotations

import json

import httpx
import pytest

from mental_state_bot.ai.client import OpenAICompatibleClient


class _FakeResponse:
    def __init__(self) -> None:
        self._payload = {
            "choices": [{"message": {"content": json.dumps({"value": "ok"})}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class _FakeAsyncClient:
    calls: list[dict] = []

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, url, headers, json):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return _FakeResponse()


@pytest.mark.asyncio
async def test_deepseek_non_thinking_payload_disables_thinking(monkeypatch) -> None:
    _FakeAsyncClient.calls = []
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    client = OpenAICompatibleClient(
        provider="deepseek",
        base_url="https://api.deepseek.com",
        api_key="test",
        timeout_seconds=10,
    )

    await client.chat(
        task_name="test",
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": "x"}],
        temperature=0.2,
        json_schema={"type": "object"},
        thinking=False,
    )

    payload = _FakeAsyncClient.calls[0]["json"]
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["temperature"] == 0.2
    assert payload["response_format"] == {"type": "json_object"}
    assert "JSON Schema" in payload["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_deepseek_thinking_payload_enables_thinking_without_temperature(monkeypatch) -> None:
    _FakeAsyncClient.calls = []
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    client = OpenAICompatibleClient(
        provider="deepseek",
        base_url="https://api.deepseek.com",
        api_key="test",
        timeout_seconds=10,
    )

    await client.chat(
        task_name="test",
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": "x"}],
        temperature=0.2,
        json_schema={"type": "object"},
        thinking=True,
    )

    payload = _FakeAsyncClient.calls[0]["json"]
    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "high"
    assert "temperature" not in payload


@pytest.mark.asyncio
async def test_non_deepseek_json_schema_payload_uses_schema_mode(monkeypatch) -> None:
    _FakeAsyncClient.calls = []
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    client = OpenAICompatibleClient(
        provider="openai-compatible",
        base_url="https://example.test/v1",
        api_key="test",
        timeout_seconds=10,
    )

    await client.chat(
        task_name="test",
        model="some-model",
        messages=[{"role": "user", "content": "x"}],
        temperature=0.2,
        json_schema={"type": "object"},
        thinking=False,
    )

    payload = _FakeAsyncClient.calls[0]["json"]
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["schema"] == {"type": "object"}
