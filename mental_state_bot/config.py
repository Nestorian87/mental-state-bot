from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    app_timezone: str = "Europe/Kyiv"
    log_level: str = "INFO"
    media_root: Path = Path("./data/media")

    telegram_bot_token: str = ""
    telegram_allowed_user_ids: list[int] = Field(default_factory=list)

    database_url: str = (
        "postgresql+asyncpg://mental_state_bot:mental_state_bot@localhost:5432/mental_state_bot"
    )
    database_sync_url: str = (
        "postgresql+psycopg://mental_state_bot:mental_state_bot@localhost:5432/mental_state_bot"
    )

    snapshot_active_start: str = "09:00"
    snapshot_active_end: str = "23:30"
    snapshot_min_interval_minutes: int = 30
    snapshot_max_interval_minutes: int = 70
    snapshot_reminder_delay_minutes: int = 25
    max_clarifications_per_snapshot: int = 2
    photo_prompt_chance: float = 0.18

    ai_provider: str = "deepseek"
    ai_base_url: str = "https://api.deepseek.com"
    ai_api_key: str = ""
    ai_live_model: str = "deepseek-v4-flash"
    ai_heavy_model: str = "deepseek-v4-pro"
    ai_temperature: float = 0.35
    ai_timeout_seconds: int = 45
    ai_live_thinking: bool = False
    ai_heavy_thinking: bool = False
    ai_provider_extra_json: dict[str, Any] = Field(
        default_factory=lambda: {
            "thinking_off": {"thinking": {"type": "disabled"}},
            "thinking_on": {"thinking": {"type": "enabled"}, "reasoning_effort": "high"},
        }
    )

    embeddings_enabled: bool = True
    embedding_provider: str = "openai-compatible"
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    @field_validator("telegram_allowed_user_ids", mode="before")
    @classmethod
    def parse_allowed_user_ids(cls, value: Any) -> list[int]:
        if value in (None, "", []):
            return []
        if isinstance(value, int):
            return [value]
        if isinstance(value, str):
            return [int(part.strip()) for part in value.split(",") if part.strip()]
        return value

    @field_validator("ai_provider_extra_json", mode="before")
    @classmethod
    def parse_extra_json(cls, value: Any) -> dict[str, Any]:
        if value in (None, ""):
            return {}
        if isinstance(value, str):
            return json.loads(value)
        return value

    @field_validator("photo_prompt_chance")
    @classmethod
    def clamp_photo_prompt_chance(cls, value: float) -> float:
        return max(0.0, min(float(value), 1.0))

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"

    def ensure_runtime_dirs(self) -> None:
        self.media_root.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_runtime_dirs()
    return settings
