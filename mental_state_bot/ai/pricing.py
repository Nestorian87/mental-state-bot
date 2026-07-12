from __future__ import annotations

from decimal import Decimal

from mental_state_bot.ai.schemas import Usage

DEEPSEEK_RATES_PER_MILLION: dict[str, tuple[Decimal, Decimal]] = {
    "deepseek-v4-flash": (Decimal("0.14"), Decimal("0.28")),
    "deepseek-v4-pro": (Decimal("0.435"), Decimal("0.87")),
}

EMBEDDING_RATES_PER_MILLION: dict[str, Decimal] = {
    "text-embedding-3-small": Decimal("0.02"),
    "text-embedding-3-large": Decimal("0.13"),
}

TRANSCRIPTION_RATES_PER_MINUTE: dict[str, Decimal] = {
    "gpt-4o-mini-transcribe": Decimal("0.003"),
}


def estimate_cost_usd(provider: str, model: str, usage: Usage) -> Decimal | None:
    provider = provider.lower()
    model_key = model.lower()
    if provider != "deepseek":
        return None

    rates = None
    for known_model, known_rates in DEEPSEEK_RATES_PER_MILLION.items():
        if known_model in model_key:
            rates = known_rates
            break
    if rates is None:
        return None

    input_rate, output_rate = rates
    input_tokens = usage.prompt_tokens or 0
    # OpenAI-compatible APIs report reasoning tokens as a breakdown of
    # completion tokens, not an additional billed output stream.
    output_tokens = usage.completion_tokens or 0
    return (Decimal(input_tokens) * input_rate + Decimal(output_tokens) * output_rate) / Decimal(1_000_000)


def estimate_embedding_cost_usd(model: str, usage: Usage) -> Decimal | None:
    model_key = model.lower()
    rate = None
    for known_model, known_rate in EMBEDDING_RATES_PER_MILLION.items():
        if known_model in model_key:
            rate = known_rate
            break
    if rate is None:
        return None
    tokens = usage.total_tokens or usage.prompt_tokens or 0
    return Decimal(tokens) * rate / Decimal(1_000_000)


def estimate_transcription_cost_usd(model: str, duration_seconds: int | float | None) -> Decimal | None:
    if duration_seconds is None:
        return None
    model_key = model.lower()
    rate = None
    for known_model, known_rate in TRANSCRIPTION_RATES_PER_MINUTE.items():
        if known_model in model_key:
            rate = known_rate
            break
    if rate is None:
        return None
    minutes = Decimal(str(max(float(duration_seconds), 0.0))) / Decimal(60)
    return minutes * rate
