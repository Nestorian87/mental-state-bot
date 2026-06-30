from __future__ import annotations

from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
)

from mental_state_bot.db.models import UserSettings
from mental_state_bot.services.preferences import snapshots_paused


def main_reply_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Новий зріз"), KeyboardButton(text="Сьогодні")],
            [KeyboardButton(text="Метрики"), KeyboardButton(text="Фото дня")],
            [KeyboardButton(text="Прогалини"), KeyboardButton(text="Налаштування")],
            [KeyboardButton(text="Лягаю спати")],
        ],
        resize_keyboard=True,
        input_field_placeholder="Можна написати будь-який момент",
    )


def snapshot_initial_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Не хочу зараз", callback_data="snapshot:stop"),
                InlineKeyboardButton(text="Пізніше", callback_data="snapshot:later"),
            ],
        ]
    )


def snapshot_clarification_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Записати як є", callback_data="snapshot:as_is"),
                InlineKeyboardButton(text="Не хочу зараз", callback_data="snapshot:stop"),
            ],
            [InlineKeyboardButton(text="Пізніше", callback_data="snapshot:later")],
        ]
    )


def correction_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Виправити", callback_data="correction:start")],
        ]
    )


def voice_transcription_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Так, зберегти", callback_data="voice:confirm")],
            [
                InlineKeyboardButton(text="Виправити текст", callback_data="voice:fix"),
                InlineKeyboardButton(text="Скасувати", callback_data="voice:cancel"),
            ],
        ]
    )


def sleep_confirmation_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Так, закрити день", callback_data="sleep:confirm"),
                InlineKeyboardButton(text="Ні, скасувати", callback_data="sleep:cancel"),
            ],
        ]
    )


def missed_prompt_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Не хочу зараз", callback_data="snapshot:stop"),
                InlineKeyboardButton(text="Пізніше", callback_data="snapshot:later"),
            ],
            [InlineKeyboardButton(text="Пояснити пропуск", callback_data="missed_reason:custom")],
        ]
    )


def summary_detail_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Історія", callback_data="summary:story"),
                InlineKeyboardButton(text="Таймлайн", callback_data="summary:timeline"),
            ],
            [
                InlineKeyboardButton(text="Метрики", callback_data="summary:metrics"),
                InlineKeyboardButton(text="Фото дня", callback_data="summary:photos"),
            ],
            [InlineKeyboardButton(text="Сирі записи", callback_data="summary:raw")],
            [InlineKeyboardButton(text="Головне меню", callback_data="nav:home")],
        ]
    )


def settings_keyboard(*, user_settings: UserSettings) -> InlineKeyboardMarkup:
    paused = snapshots_paused(user_settings)
    active_text = "Увімкнути" if paused else "Пауза"
    active_action = "resume" if paused else "pause"
    body_text = "Тіло: on" if user_settings.ask_body_signals else "Тіло: off"
    photo_text = "Фото: on" if user_settings.photo_prompts_enabled else "Фото: off"
    humanity_text = "Людяніше" if user_settings.humanity_level != "warm" else "Стриманіше"
    humanity_action = "warm" if user_settings.humanity_level != "warm" else "balanced"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text=active_text, callback_data=f"settings:{active_action}"),
                InlineKeyboardButton(text="Оновити", callback_data="settings:open"),
            ],
            [
                InlineKeyboardButton(text="Рідше", callback_data="settings:freq:slow"),
                InlineKeyboardButton(text="Норм", callback_data="settings:freq:normal"),
                InlineKeyboardButton(text="Частіше", callback_data="settings:freq:fast"),
            ],
            [
                InlineKeyboardButton(text="Нагад. 15", callback_data="settings:reminder:15"),
                InlineKeyboardButton(text="Нагад. 25", callback_data="settings:reminder:25"),
                InlineKeyboardButton(text="Нагад. 45", callback_data="settings:reminder:45"),
            ],
            [
                InlineKeyboardButton(text=body_text, callback_data="settings:toggle:body"),
                InlineKeyboardButton(text=photo_text, callback_data="settings:toggle:photo"),
            ],
            [
                InlineKeyboardButton(text="Тон: сухий", callback_data="settings:tone:precise"),
                InlineKeyboardButton(text="Тон: спокійний", callback_data="settings:tone:calm"),
            ],
            [
                InlineKeyboardButton(text=humanity_text, callback_data=f"settings:humanity:{humanity_action}"),
            ],
            [InlineKeyboardButton(text="Власний стиль", callback_data="settings:custom_style")],
            [InlineKeyboardButton(text="Контекст про мене", callback_data="settings:profile_context")],
            [InlineKeyboardButton(text="Головне меню", callback_data="nav:home")],
        ]
    )
