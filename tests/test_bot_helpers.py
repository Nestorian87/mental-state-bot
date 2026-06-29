from __future__ import annotations

from types import SimpleNamespace

from mental_state_bot.bot.handlers import (
    _archive_export_options,
    _command_argument,
    _format_settings_text,
    _frequency_preset_values,
    _help_text,
    _is_sleep_marker_text,
    _missed_reason_text,
    _split_telegram_text,
    _valid_hhmm,
)
from mental_state_bot.bot.keyboards import missed_prompt_keyboard
from mental_state_bot.services.preferences import (
    custom_interaction_style,
    settings_json_with_custom_interaction_style,
    settings_json_with_snapshot_pause,
    snapshots_paused,
)


def test_command_argument() -> None:
    assert _command_argument("/similar лежу порожньо") == "лежу порожньо"
    assert _command_argument("/similar") == ""


def test_valid_hhmm() -> None:
    assert _valid_hhmm("09:00")
    assert _valid_hhmm("23:59")
    assert not _valid_hhmm("24:00")
    assert not _valid_hhmm("9")


def test_split_telegram_text_keeps_chunks_under_limit() -> None:
    chunks = _split_telegram_text("a\n" * 5000, limit=100)

    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_help_text_mentions_core_commands() -> None:
    text = _help_text()

    assert "/snapshot" in text
    assert "/summary" in text
    assert "/sleep" in text
    assert "/export" in text
    assert "/export_csv" in text
    assert "/export_zip" in text
    assert "/pause" in text
    assert "/resume" in text
    assert "/gaps" in text
    assert "/audit" in text


def test_is_sleep_marker_text_matches_only_narrow_phrases() -> None:
    assert _is_sleep_marker_text("лягаю спати")
    assert _is_sleep_marker_text("Лягаю спати.")
    assert not _is_sleep_marker_text("Йду спати.")
    assert not _is_sleep_marker_text("  спати  ")
    assert not _is_sleep_marker_text("не хочу спати але працюю")
    assert not _is_sleep_marker_text("погано спав, але ще не лягаю")


def test_missed_reason_text_uses_explicit_prefix() -> None:
    assert _missed_reason_text("причина: був у дорозі") == "був у дорозі"
    assert _missed_reason_text("Reason: phone was away") == "phone was away"
    assert _missed_reason_text("причина:   ") is None
    assert _missed_reason_text("бо був у дорозі") is None


def test_missed_prompt_keyboard_exposes_reason_actions() -> None:
    keyboard = missed_prompt_keyboard()
    callbacks = {
        button.callback_data
        for row in keyboard.inline_keyboard
        for button in row
        if button.callback_data
    }

    assert "snapshot:as_is" not in callbacks
    assert "snapshot:stop" in callbacks
    assert "snapshot:later" in callbacks
    assert "missed_reason:custom" in callbacks
    assert "missed_reason:busy" not in callbacks


def test_frequency_preset_values() -> None:
    assert _frequency_preset_values("slow") == {
        "min_interval_minutes": 75,
        "max_interval_minutes": 120,
    }
    assert _frequency_preset_values("normal") == {
        "min_interval_minutes": 30,
        "max_interval_minutes": 70,
    }
    assert _frequency_preset_values("fast") == {
        "min_interval_minutes": 20,
        "max_interval_minutes": 40,
    }
    assert _frequency_preset_values("unknown") == {
        "min_interval_minutes": 30,
        "max_interval_minutes": 70,
    }


def test_format_settings_text() -> None:
    user_settings = SimpleNamespace(
        active_start="09:00",
        active_end="23:30",
        min_interval_minutes=30,
        max_interval_minutes=70,
        reminder_delay_minutes=25,
        max_clarifications=2,
        ask_body_signals=True,
        photo_prompts_enabled=False,
        tone="calm",
        humanity_level="balanced",
        settings_json={"custom_interaction_style": "коротко і без підбадьорювання"},
    )
    text = _format_settings_text(
        user_settings=user_settings,
        snapshots_are_paused=True,
    )

    assert "Автоматичні зрізи: на паузі" in text
    assert "Тон: спокійний" in text
    assert "Стиль: стримано" in text
    assert "Власний стиль: коротко і без підбадьорювання" in text
    assert "Інтервал зрізів: 30-70 хв" in text
    assert "Питати про тіло: так" in text
    assert "Фото-підказки: ні" in text


def test_snapshot_pause_preferences_preserve_settings_json() -> None:
    settings = SimpleNamespace(settings_json={"other": "value"})

    updated = settings_json_with_snapshot_pause(settings, True)

    assert updated == {"other": "value", "snapshots_paused": True}
    assert snapshots_paused(SimpleNamespace(settings_json=updated))


def test_custom_interaction_style_preferences() -> None:
    settings = SimpleNamespace(settings_json={"other": "value"})

    updated = settings_json_with_custom_interaction_style(settings, "  коротко   без порад  ")
    assert updated == {"other": "value", "custom_interaction_style": "коротко без порад"}
    assert custom_interaction_style(SimpleNamespace(settings_json=updated)) == "коротко без порад"

    reset = settings_json_with_custom_interaction_style(SimpleNamespace(settings_json=updated), None)
    assert reset == {"other": "value"}
    assert custom_interaction_style(SimpleNamespace(settings_json=reset)) is None


def test_archive_export_options() -> None:
    assert _archive_export_options("archive:export_json") == (
        "json",
        "json",
        "export",
        "JSON-архів готовий.",
    )
    assert _archive_export_options("archive:export_md") == (
        "markdown",
        "md",
        "export",
        "Markdown-архів готовий.",
    )
    assert _archive_export_options("archive:export_csv") == (
        "csv",
        "csv",
        "metrics",
        "CSV з метриками готовий.",
    )
    assert _archive_export_options("archive:export_zip") == (
        "zip",
        "zip",
        "archive",
        "ZIP-архів з даними готовий.",
    )
