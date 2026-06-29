from __future__ import annotations

from mental_state_bot.config import Settings


def test_settings_parse_allowed_user_ids() -> None:
    settings = Settings(
        telegram_allowed_user_ids="123, 456",
        ai_provider_extra_json='{"thinking_off": {"thinking": {"type": "disabled"}}}',
    )

    assert settings.telegram_allowed_user_ids == [123, 456]
    assert settings.ai_provider_extra_json["thinking_off"]["thinking"]["type"] == "disabled"


def test_settings_parse_single_allowed_user_id_from_int() -> None:
    settings = Settings(telegram_allowed_user_ids=664158220)

    assert settings.telegram_allowed_user_ids == [664158220]


def test_settings_empty_allowed_users() -> None:
    settings = Settings(telegram_allowed_user_ids="")

    assert settings.telegram_allowed_user_ids == []
