from __future__ import annotations

from collections.abc import Sequence

from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
)

from mental_state_bot.db.models import UserSettings
from mental_state_bot.services.preferences import snapshots_paused


def main_reply_keyboard(
    placeholder: str = "Напиши момент або надішли голосове",
) -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Меню"), KeyboardButton(text="Новий зріз")],
            [KeyboardButton(text="Лягаю спати")],
        ],
        resize_keyboard=True,
        input_field_placeholder=placeholder[:64],
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


def correction_keyboard(*, entry_id: str | None = None) -> InlineKeyboardMarkup:
    callback_data = f"correction:start:{entry_id}" if entry_id else "correction:start"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Виправити", callback_data=callback_data)],
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


def manual_entry_confirmation_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Записати як момент", callback_data="manual:save")],
            [InlineKeyboardButton(text="Ігнорувати", callback_data="manual:ignore")],
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


def main_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="День", callback_data="menu:day"),
                InlineKeyboardButton(text="Підсумки", callback_data="menu:summaries"),
            ],
            [
                InlineKeyboardButton(text="Пам’ять", callback_data="menu:memory"),
                InlineKeyboardButton(text="Дані", callback_data="menu:data"),
            ],
            [
                InlineKeyboardButton(text="Налаштування", callback_data="settings:open"),
                InlineKeyboardButton(text="Новий зріз", callback_data="snapshot:new"),
            ],
        ]
    )


def day_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Сьогодні", callback_data="menu:day:today"),
                InlineKeyboardButton(text="Вчора", callback_data="menu:day:yesterday"),
            ],
            [InlineKeyboardButton(text="Ввести дату", callback_data="menu:day:date")],
            [InlineKeyboardButton(text="Головне меню", callback_data="menu:main")],
        ]
    )


def summaries_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Підсумок дня", callback_data="menu:summaries:day")],
            [
                InlineKeyboardButton(text="Тиждень", callback_data="menu:summaries:week"),
                InlineKeyboardButton(text="Місяць", callback_data="menu:summaries:month"),
            ],
            [InlineKeyboardButton(text="Головне меню", callback_data="menu:main")],
        ]
    )


def period_choice_keyboard(*, period: str) -> InlineKeyboardMarkup:
    label = "тиждень" if period == "week" else "місяць"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text=f"Поточний {label}", callback_data=f"menu:period:{period}:current"),
                InlineKeyboardButton(text=f"Попередній {label}", callback_data=f"menu:period:{period}:previous"),
            ],
            [InlineKeyboardButton(text="Назад", callback_data="menu:summaries")],
        ]
    )


def memory_menu_keyboard(*, embeddings_enabled: bool) -> InlineKeyboardMarkup:
    status = "✅ embeddings увімкнені" if embeddings_enabled else "embeddings вимкнені"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Пошук у пам’яті", callback_data="menu:memory:search")],
            [InlineKeyboardButton(text="Схоже на останній запис", callback_data="menu:memory:last")],
            [InlineKeyboardButton(text=status, callback_data="menu:memory:status")],
            [InlineKeyboardButton(text="Головне меню", callback_data="menu:main")],
        ]
    )


def data_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Аудит архіву", callback_data="archive:audit"),
                InlineKeyboardButton(text="Витрати AI", callback_data="ai:costs"),
            ],
            [
                InlineKeyboardButton(text="JSON", callback_data="archive:export"),
                InlineKeyboardButton(text="Markdown", callback_data="archive:export_md"),
            ],
            [
                InlineKeyboardButton(text="CSV", callback_data="archive:export_csv"),
                InlineKeyboardButton(text="ZIP", callback_data="archive:export_zip"),
            ],
            [InlineKeyboardButton(text="Переаналіз AI", callback_data="menu:data:reanalyze")],
            [InlineKeyboardButton(text="Головне меню", callback_data="menu:main")],
        ]
    )


def reanalysis_confirmation_keyboard(*, limit: int = 200) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=f"Так, переаналізувати {limit}", callback_data=f"features:reanalyze:{limit}")],
            [
                InlineKeyboardButton(text="Скасувати", callback_data="features:reanalyze:cancel"),
                InlineKeyboardButton(text="Назад до даних", callback_data="menu:data"),
            ],
        ]
    )


def summary_detail_keyboard(*, summary_id: str | None = None) -> InlineKeyboardMarkup:
    prefix = f"summary:{summary_id}" if summary_id else "summary"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Історія", callback_data=f"{prefix}:story"),
                InlineKeyboardButton(text="Таймлайн", callback_data=f"{prefix}:timeline"),
            ],
            [
                InlineKeyboardButton(text="Метрики", callback_data=f"{prefix}:metrics"),
                InlineKeyboardButton(text="Фото дня", callback_data=f"{prefix}:photos"),
            ],
            [InlineKeyboardButton(text="Сирі записи", callback_data=f"{prefix}:raw")],
            [InlineKeyboardButton(text="Оновити підсумок", callback_data=f"{prefix}:refresh")],
            [InlineKeyboardButton(text="Головне меню", callback_data="nav:home")],
        ]
    )


def day_detail_keyboard(*, day_id: str) -> InlineKeyboardMarkup:
    prefix = f"dayview:{day_id}"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Історія", callback_data=f"{prefix}:story"),
                InlineKeyboardButton(text="Таймлайн", callback_data=f"{prefix}:timeline"),
            ],
            [
                InlineKeyboardButton(text="Метрики", callback_data=f"{prefix}:metrics"),
                InlineKeyboardButton(text="Фото дня", callback_data=f"{prefix}:photos"),
            ],
            [
                InlineKeyboardButton(text="Прогалини", callback_data=f"{prefix}:gaps"),
                InlineKeyboardButton(text="Сирі записи", callback_data=f"{prefix}:raw"),
            ],
            [InlineKeyboardButton(text="Керувати записами", callback_data=f"{prefix}:entries")],
            [InlineKeyboardButton(text="Оновити підсумок", callback_data=f"{prefix}:refresh")],
            [InlineKeyboardButton(text="Головне меню", callback_data="nav:home")],
        ]
    )


def entry_management_keyboard(*, day_id: str, entries: Sequence[tuple[str, str]]) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(text=label, callback_data=f"entry:delete:{entry_id}")]
        for entry_id, label in entries
    ]
    rows.append([InlineKeyboardButton(text="Назад до дня", callback_data=f"dayview:{day_id}:timeline")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def entry_delete_confirmation_keyboard(*, entry_id: str, day_id: str | None = None) -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(text="Так, видалити запис", callback_data=f"entry:confirm_delete:{entry_id}")]]
    if day_id:
        rows.append([InlineKeyboardButton(text="Скасувати", callback_data=f"dayview:{day_id}:entries")])
    else:
        rows.append([InlineKeyboardButton(text="Скасувати", callback_data="nav:home")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def period_detail_keyboard(*, summary_id: str) -> InlineKeyboardMarkup:
    prefix = f"periodview:{summary_id}"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Огляд", callback_data=f"{prefix}:overview"),
                InlineKeyboardButton(text="Таймлайн", callback_data=f"{prefix}:timeline"),
            ],
            [
                InlineKeyboardButton(text="Метрики", callback_data=f"{prefix}:metrics"),
                InlineKeyboardButton(text="Графік", callback_data=f"{prefix}:chart"),
            ],
            [InlineKeyboardButton(text="Дні періоду", callback_data=f"{prefix}:days")],
            [InlineKeyboardButton(text="Головне меню", callback_data="nav:home")],
        ]
    )


def settings_keyboard(*, user_settings: UserSettings) -> InlineKeyboardMarkup:
    settings_json = user_settings.settings_json or {}
    paused = snapshots_paused(user_settings)
    active_text = "✅ На паузі" if paused else "✅ Увімкнено"
    body_text = "✅ Тіло" if user_settings.ask_body_signals else "Тіло"
    photo_text = "✅ Фото" if user_settings.photo_prompts_enabled else "Фото"
    custom_style_text = "✅ Власний стиль" if settings_json.get("custom_interaction_style") else "Власний стиль"
    context_text = "✅ Контекст про мене" if settings_json.get("user_profile_context") else "Контекст про мене"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Ритм зрізів", callback_data="settings:section:rhythm"),
                InlineKeyboardButton(text="Стиль", callback_data="settings:section:style"),
            ],
            [
                InlineKeyboardButton(text="Збір даних", callback_data="settings:section:capture"),
                InlineKeyboardButton(text=active_text, callback_data="settings:toggle:pause"),
            ],
            [
                InlineKeyboardButton(text=custom_style_text, callback_data="settings:custom_style"),
                InlineKeyboardButton(text=context_text, callback_data="settings:profile_context"),
            ],
            [
                InlineKeyboardButton(text=body_text, callback_data="settings:toggle:body"),
                InlineKeyboardButton(text=photo_text, callback_data="settings:toggle:photo"),
            ],
            [InlineKeyboardButton(text="Головне меню", callback_data="menu:main")],
        ]
    )


def settings_rhythm_keyboard(*, user_settings: UserSettings) -> InlineKeyboardMarkup:
    paused = snapshots_paused(user_settings)
    active_text = "Увімкнути" if paused else "Поставити на паузу"
    active_action = "resume" if paused else "pause"
    freq = (user_settings.min_interval_minutes, user_settings.max_interval_minutes)
    reminder = user_settings.reminder_delay_minutes
    def freq_text(label: str, values: tuple[int, int]) -> str:
        return f"✅ {label}" if freq == values else label
    def reminder_text(minutes: int) -> str:
        return f"✅ {minutes} хв" if reminder == minutes else f"{minutes} хв"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text=active_text, callback_data=f"settings:{active_action}"),
                InlineKeyboardButton(text="Оновити", callback_data="settings:open"),
            ],
            [
                InlineKeyboardButton(text=freq_text("Рідше", (75, 120)), callback_data="settings:freq:slow"),
                InlineKeyboardButton(text=freq_text("Норм", (30, 70)), callback_data="settings:freq:normal"),
                InlineKeyboardButton(text=freq_text("Частіше", (20, 40)), callback_data="settings:freq:fast"),
            ],
            [
                InlineKeyboardButton(text=reminder_text(15), callback_data="settings:reminder:15"),
                InlineKeyboardButton(text=reminder_text(25), callback_data="settings:reminder:25"),
                InlineKeyboardButton(text=reminder_text(45), callback_data="settings:reminder:45"),
            ],
            [InlineKeyboardButton(text="Назад", callback_data="settings:open")],
        ]
    )


def settings_style_keyboard(*, user_settings: UserSettings) -> InlineKeyboardMarkup:
    tone = getattr(user_settings, "tone", "calm")
    humanity = getattr(user_settings, "humanity_level", "balanced")
    precise_text = "✅ Сухий" if tone == "precise" else "Сухий"
    calm_text = "✅ Спокійний" if tone == "calm" else "Спокійний"
    balanced_text = "✅ Стримано" if humanity == "balanced" else "Стримано"
    warm_text = "✅ Людяніше" if humanity == "warm" else "Людяніше"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text=precise_text, callback_data="settings:tone:precise"),
                InlineKeyboardButton(text=calm_text, callback_data="settings:tone:calm"),
            ],
            [
                InlineKeyboardButton(text=balanced_text, callback_data="settings:humanity:balanced"),
                InlineKeyboardButton(text=warm_text, callback_data="settings:humanity:warm"),
            ],
            [InlineKeyboardButton(text="Власний стиль", callback_data="settings:custom_style")],
            [InlineKeyboardButton(text="Контекст про мене", callback_data="settings:profile_context")],
            [InlineKeyboardButton(text="Назад", callback_data="settings:open")],
        ]
    )


def settings_capture_keyboard(*, user_settings: UserSettings) -> InlineKeyboardMarkup:
    body_text = "✅ Питати про тіло" if user_settings.ask_body_signals else "Питати про тіло"
    photo_text = "✅ Фото-підказки" if user_settings.photo_prompts_enabled else "Фото-підказки"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text=body_text, callback_data="settings:toggle:body"),
                InlineKeyboardButton(text=photo_text, callback_data="settings:toggle:photo"),
            ],
            [InlineKeyboardButton(text="Назад", callback_data="settings:open")],
        ]
    )
