from __future__ import annotations

from decimal import Decimal

from mental_state_bot.ai.pricing import (
    estimate_cost_usd,
    estimate_embedding_cost_usd,
    estimate_transcription_cost_usd,
)
from mental_state_bot.ai.schemas import Usage


def test_deepseek_flash_cost_estimate() -> None:
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000)

    assert estimate_cost_usd("deepseek", "deepseek-v4-flash", usage) == Decimal("0.42")


def test_reasoning_tokens_are_reported_but_not_charged_twice() -> None:
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000, reasoning_tokens=800_000)

    assert estimate_cost_usd("deepseek", "deepseek-v4-flash", usage) == Decimal("0.42")


def test_unknown_provider_has_no_estimate() -> None:
    usage = Usage(prompt_tokens=100, completion_tokens=100)

    assert estimate_cost_usd("unknown", "model", usage) is None


def test_openai_embedding_cost_estimate() -> None:
    usage = Usage(prompt_tokens=1_000_000, total_tokens=1_000_000)

    assert estimate_embedding_cost_usd("text-embedding-3-small", usage) == Decimal("0.02")


def test_openai_mini_transcription_cost_estimate() -> None:
    assert estimate_transcription_cost_usd("gpt-4o-mini-transcribe", 120) == Decimal("0.006")
