from __future__ import annotations

from collections.abc import Sequence

from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
)

from mental_state_bot.db.models import UserSettings
from mental_state_bot.emotions import CANONICAL_EMOTIONS
from mental_state_bot.services.preferences import (
    adaptive_observation_enabled,
    context_quiet_enabled,
    quiet_is_active,
    snapshots_paused,
)

EMOTION_CALIBRATION_OPTIONS = CANONICAL_EMOTIONS
EMOTION_INTENSITY_OPTIONS = (
    ("trace", "Ледь фоном"),
    ("mild", "Слабко"),
    ("moderate", "Помірно"),
    ("strong", "Сильно"),
    ("overwhelming", "Дуже сильно"),
)
_EMOTION_INTENSITY_CODES = {
    "trace": "t",
    "mild": "m",
    "moderate": "d",
    "strong": "s",
    "overwhelming": "o",
    "unclear": "u",
}


def main_reply_keyboard(
    placeholder: str = "Напиши момент або надішли голосове",
) -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Меню"), KeyboardButton(text="Новий зріз")],
            [KeyboardButton(text="Пауза"), KeyboardButton(text="Лягаю спати")],
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


def deferred_clarification_keyboard(*, item_id: str, options: Sequence[str] = ()) -> InlineKeyboardMarkup:
    option_buttons = [
        InlineKeyboardButton(text=" ".join(str(option).split())[:64], callback_data=f"clarification:option:{item_id}:{index}")
        for index, option in enumerate(options[:4])
        if str(option).strip()
    ]
    option_rows = [option_buttons[index : index + 2] for index in range(0, len(option_buttons), 2)]
    return InlineKeyboardMarkup(
        inline_keyboard=[
            *option_rows,
            [InlineKeyboardButton(text="Пропустити", callback_data=f"clarification:skip:{item_id}")],
            [InlineKeyboardButton(text="Уточнення", callback_data="clarification_queue:open")],
        ]
    )


def clarifications_menu_keyboard(
    *,
    has_queued: bool,
    has_pending: bool,
    has_clearable: bool,
) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    if has_queued and not has_pending:
        rows.append([InlineKeyboardButton(text="Поставити наступне", callback_data="clarification_queue:next")])
    if has_clearable:
        rows.append([InlineKeyboardButton(text="Пропустити все", callback_data="clarification_queue:skip_all")])
    rows.append([InlineKeyboardButton(text="Головне меню", callback_data="menu:main")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def clarifications_skip_all_confirmation_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Так, пропустити все", callback_data="clarification_queue:skip_all_confirm")],
            [InlineKeyboardButton(text="Скасувати", callback_data="clarification_queue:open")],
        ]
    )


def correction_keyboard(*, entry_id: str | None = None) -> InlineKeyboardMarkup:
    callback_data = f"correction:start:{entry_id}" if entry_id else "correction:start"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Виправити", callback_data=callback_data)],
        ]
    )


def interpretation_keyboard(*, entry_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Виправити словами", callback_data=f"correction:start:{entry_id}"),
            ],
            [
                InlineKeyboardButton(text="Настрій", callback_data=f"metric:start:{entry_id}:mood"),
                InlineKeyboardButton(text="Енергія", callback_data=f"metric:start:{entry_id}:energy"),
            ],
            [InlineKeyboardButton(text="Емоції", callback_data=f"emotion:start:{entry_id}")],
            [InlineKeyboardButton(text="Ок", callback_data="interpretation:ok")],
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


def metric_score_keyboard(
    *, entry_id: str, metric: str, include_correction: bool = False
) -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton(text=str(score), callback_data=f"metric:{entry_id}:{metric}:{score}")
            for score in range(0, 6)
        ],
        [
            InlineKeyboardButton(text=str(score), callback_data=f"metric:{entry_id}:{metric}:{score}")
            for score in range(6, 11)
        ],
        [InlineKeyboardButton(text="Не хочу", callback_data=f"metric:{entry_id}:{metric}:skip")],
    ]
    if include_correction:
        rows.append([InlineKeyboardButton(text="Виправити словами", callback_data=f"correction:start:{entry_id}")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def emotion_calibration_keyboard(
    *, entry_id: str, selected: Sequence[str] = (), include_correction: bool = False
) -> InlineKeyboardMarkup:
    selected_set = {item for item in selected if item in EMOTION_CALIBRATION_OPTIONS}
    selected_indexes = tuple(index for index, emotion in enumerate(EMOTION_CALIBRATION_OPTIONS) if emotion in selected_set)

    def callback(action: str, indexes: Sequence[int] = selected_indexes, toggle_index: int | None = None) -> str:
        compact_entry_id = entry_id.replace("-", "")
        encoded = _encode_emotion_indexes(indexes)
        if toggle_index is None:
            return f"emotion:{action}:{compact_entry_id}:{encoded}"
        return f"emotion:{action}:{compact_entry_id}:{encoded}:{toggle_index}"

    def option_button(index: int, label: str) -> InlineKeyboardButton:
        next_indexes = set(selected_indexes)
        if index in next_indexes:
            next_indexes.remove(index)
        else:
            next_indexes.add(index)
        marker = "✓ " if label in selected_set else ""
        return InlineKeyboardButton(
            text=f"{marker}{label.capitalize()}",
            callback_data=callback("t", tuple(sorted(next_indexes)), index),
        )

    option_rows = [
        [
            option_button(index, label)
            for index, label in enumerate(EMOTION_CALIBRATION_OPTIONS[start : start + 2], start=start)
        ]
        for start in range(0, len(EMOTION_CALIBRATION_OPTIONS), 2)
    ]
    rows = [
        *option_rows,
        [InlineKeyboardButton(text="Зберегти вибір", callback_data=callback("save"))],
        [InlineKeyboardButton(text="Не уточнювати", callback_data=callback("skip"))],
        [InlineKeyboardButton(text="Описати словами", callback_data=callback("custom"))],
    ]
    if include_correction:
        rows.append([InlineKeyboardButton(text="Виправити словами", callback_data=f"correction:start:{entry_id}")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def emotion_intensity_keyboard(
    *,
    entry_id: str,
    selected: Sequence[str],
    intensity_levels: Sequence[str] | None = None,
    position: int = 0,
    time_scope: str = "current",
) -> InlineKeyboardMarkup:
    selected_indexes = tuple(index for index, emotion in enumerate(EMOTION_CALIBRATION_OPTIONS) if emotion in selected)
    compact_entry_id = entry_id.replace("-", "")
    encoded = _encode_emotion_indexes(selected_indexes)
    levels = list(intensity_levels or ())
    if len(levels) != len(selected_indexes):
        levels = ["unclear"] * len(selected_indexes)
    level_codes = "".join(_EMOTION_INTENSITY_CODES.get(level, "u") for level in levels)
    safe_position = min(max(position, 0), max(0, len(selected_indexes) - 1))
    scope_code = "n" if time_scope == "mentioned_not_felt" else "c"

    def callback(action: str, *extra: object) -> str:
        return ":".join(
            [
                "e",
                action,
                compact_entry_id,
                encoded,
                level_codes,
                scope_code,
                str(safe_position),
                *(str(item) for item in extra),
            ]
        )

    intensity_rows = [
        [
            InlineKeyboardButton(
                text=label,
                callback_data=callback("s", _EMOTION_INTENSITY_CODES[level]),
            )
            for level, label in pair
        ]
        for pair in (
            EMOTION_INTENSITY_OPTIONS[:2],
            EMOTION_INTENSITY_OPTIONS[2:4],
            EMOTION_INTENSITY_OPTIONS[4:],
        )
    ]
    rows = [*intensity_rows]
    rows.append(
        [
            InlineKeyboardButton(
                text=("✓ " if scope_code == "n" else "") + "Це не поточні емоції",
                callback_data=callback("x"),
            )
        ]
    )
    navigation: list[InlineKeyboardButton] = []
    if safe_position > 0:
        navigation.append(InlineKeyboardButton(text="← Попередня", callback_data=callback("n", safe_position - 1)))
    if safe_position < len(selected_indexes) - 1:
        navigation.append(InlineKeyboardButton(text="Наступна →", callback_data=callback("n", safe_position + 1)))
    if navigation:
        rows.append(navigation)
    rows.append([InlineKeyboardButton(text="Зберегти", callback_data=callback("d"))])

    return InlineKeyboardMarkup(inline_keyboard=rows)


def _encode_emotion_indexes(indexes: Sequence[int]) -> str:
    mask = 0
    for index in indexes:
        if index >= 0:
            mask |= 1 << index
    return f"h{mask:x}" if mask else ""


def manual_entry_confirmation_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Записати як момент", callback_data="manual:save")],
            [InlineKeyboardButton(text="Ігнорувати", callback_data="manual:ignore")],
        ]
    )


def planned_event_offer_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Так, запам'ятати", callback_data="planned_event:confirm")],
            [
                InlineKeyboardButton(text="Уточнити", callback_data="planned_event:clarify"),
                InlineKeyboardButton(text="Ігнорувати", callback_data="planned_event:ignore"),
            ],
            [InlineKeyboardButton(text="Скасувати", callback_data="planned_event:cancel")],
        ]
    )


def wake_time_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Не уточнювати", callback_data="wake_time:skip")],
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


def sleep_reflection_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Важкий", callback_data="sleep:reflect:hard"),
                InlineKeyboardButton(text="Змішаний", callback_data="sleep:reflect:mixed"),
            ],
            [
                InlineKeyboardButton(text="Нормальний", callback_data="sleep:reflect:okay"),
                InlineKeyboardButton(text="Добрий", callback_data="sleep:reflect:good"),
            ],
            [InlineKeyboardButton(text="Написати своє", callback_data="sleep:reflect:custom")],
            [InlineKeyboardButton(text="Пропустити", callback_data="sleep:reflect:skip")],
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
                InlineKeyboardButton(text="Живий контекст", callback_data="menu:life_context"),
            ],
            [
                InlineKeyboardButton(text="Дані", callback_data="menu:data"),
                InlineKeyboardButton(text="Налаштування", callback_data="settings:open"),
            ],
            [
                InlineKeyboardButton(text="Новий зріз", callback_data="snapshot:new"),
                InlineKeyboardButton(text="Уточнення", callback_data="menu:clarifications"),
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
            [InlineKeyboardButton(text="Візуалізація графа", callback_data="menu:memory:graph")],
            [InlineKeyboardButton(text="Фрази й значення", callback_data="menu:memory:lexicon")],
            [
                InlineKeyboardButton(text="Експорт графа", callback_data="menu:memory:export"),
                InlineKeyboardButton(text="Імпорт графа", callback_data="menu:memory:import"),
            ],
            [InlineKeyboardButton(text="Пошук у пам’яті", callback_data="menu:memory:search")],
            [InlineKeyboardButton(text="Схоже на останній запис", callback_data="menu:memory:last")],
            [InlineKeyboardButton(text="Пам’ять у питаннях", callback_data="menu:memory:influences")],
            [InlineKeyboardButton(text="Обслуговування графа", callback_data="menu:memory:maintain")],
            [InlineKeyboardButton(text="AI-ревізія графа", callback_data="menu:memory:review")],
            [InlineKeyboardButton(text="Перебудувати пам’ять", callback_data="menu:memory:rebuild")],
            [InlineKeyboardButton(text=status, callback_data="menu:memory:status")],
            [InlineKeyboardButton(text="Головне меню", callback_data="menu:main")],
        ]
    )


def memory_maintenance_confirmation_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Так, перевірити граф", callback_data="memory:maintain:confirm")],
            [InlineKeyboardButton(text="Скасувати", callback_data="memory:maintain:cancel")],
        ]
    )


def memory_ai_review_confirmation_keyboard(*, limit: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=f"Так, перевірити {limit} пар", callback_data=f"memory:review:{limit}")],
            [InlineKeyboardButton(text="Скасувати", callback_data="memory:review:cancel")],
        ]
    )


def memory_rebuild_confirmation_keyboard(*, limit: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=f"Так, перебудувати {limit}", callback_data=f"memory:rebuild:{limit}")],
            [InlineKeyboardButton(text="Скасувати", callback_data="memory:rebuild:cancel")],
        ]
    )


def memory_graph_import_confirmation_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Так, замінити граф", callback_data="memory:import:confirm")],
            [InlineKeyboardButton(text="Скасувати", callback_data="memory:import:cancel")],
        ]
    )


def data_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Візуальний PDF-звіт", callback_data="menu:data:visual_report")],
            [
                InlineKeyboardButton(text="Аудит архіву", callback_data="archive:audit"),
                InlineKeyboardButton(text="Витрати AI", callback_data="ai:costs"),
            ],
            [InlineKeyboardButton(text="Аудит емоцій", callback_data="menu:data:affect_audit")],
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


def reanalysis_scope_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Пробно: 10 останніх", callback_data="features:scope:recent:10")],
            [
                InlineKeyboardButton(text="Останні 3 журнальні дні", callback_data="features:scope:days:3"),
                InlineKeyboardButton(text="Останні 7 днів", callback_data="features:scope:days:7"),
            ],
            [InlineKeyboardButton(text="Ввести діапазон дат", callback_data="features:scope:range")],
            [InlineKeyboardButton(text="Увесь архів", callback_data="features:scope:all")],
            [InlineKeyboardButton(text="Назад до даних", callback_data="menu:data")],
        ]
    )


def reanalysis_confirmation_keyboard(*, action: str, selected: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=f"Так, переаналізувати {selected}",
                    callback_data=f"features:reanalyze:{action}",
                )
            ],
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
            [InlineKeyboardButton(text="Повороти дня", callback_data=f"{prefix}:turning_points")],
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
            [InlineKeyboardButton(text="Повороти дня", callback_data=f"{prefix}:turning_points")],
            [InlineKeyboardButton(text="Керувати записами", callback_data=f"{prefix}:entries")],
            [InlineKeyboardButton(text="Оновити підсумок", callback_data=f"{prefix}:refresh")],
            [InlineKeyboardButton(text="Головне меню", callback_data="nav:home")],
        ]
    )


def turning_points_keyboard(*, day_id: str, labels: Sequence[str]) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(text=label[:64], callback_data=f"turning:detail:{day_id}:{index}")]
        for index, label in enumerate(labels)
    ]
    rows.append([InlineKeyboardButton(text="До дня", callback_data=f"dayview:{day_id}:story")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def turning_point_detail_keyboard(*, day_id: str, index: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Показати запис", callback_data=f"turning:entry:{day_id}:{index}")],
            [InlineKeyboardButton(text="Усі повороти", callback_data=f"turning:list:{day_id}")],
            [InlineKeyboardButton(text="До дня", callback_data=f"dayview:{day_id}:story")],
        ]
    )


def entry_management_keyboard(*, day_id: str, entries: Sequence[tuple[str, str]]) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(text=label, callback_data=f"entry:delete:{entry_id}")]
        for entry_id, label in entries
    ]
    rows.append([InlineKeyboardButton(text="Назад до дня", callback_data=f"dayview:{day_id}:timeline")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def life_context_menu_keyboard(*, has_items: bool = False) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(text="Знайти припущення", callback_data="life_context:scan")],
    ]
    if has_items:
        rows.append([InlineKeyboardButton(text="Показати живий контекст", callback_data="life_context:list")])
        rows.append([InlineKeyboardButton(text="Оновити живий контекст", callback_data="life_context:rewrite")])
    rows.append([InlineKeyboardButton(text="Головне меню", callback_data="menu:main")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def life_context_offer_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Так, перевірити", callback_data="life_context:review:start")],
            [
                InlineKeyboardButton(text="Пізніше", callback_data="life_context:review:later"),
                InlineKeyboardButton(text="Не зараз", callback_data="life_context:review:stop"),
            ],
        ]
    )


def life_context_question_keyboard(candidate: dict) -> InlineKeyboardMarkup:
    options = [str(option) for option in candidate.get("options") or [] if str(option).strip()]
    rows = []
    for index, option in enumerate(options[:5]):
        rows.append([InlineKeyboardButton(text=option[:64], callback_data=f"life_context:answer:option:{index}")])
    if not rows and candidate.get("question_type") == "confirm":
        rows.append(
            [
                InlineKeyboardButton(text="Так", callback_data="life_context:answer:yes"),
                InlineKeyboardButton(text="Не зовсім", callback_data="life_context:answer:free"),
                InlineKeyboardButton(text="Ні", callback_data="life_context:answer:no"),
            ]
        )
    else:
        rows.append([InlineKeyboardButton(text="Поясню словами", callback_data="life_context:answer:free")])
    rows.append(
        [
            InlineKeyboardButton(text="Пропустити", callback_data="life_context:answer:skip"),
            InlineKeyboardButton(text="Зупинити", callback_data="life_context:answer:stop"),
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def life_context_continue_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Продовжити перевірку", callback_data="life_context:review:next")],
            [InlineKeyboardButton(text="Зупинити", callback_data="life_context:answer:stop")],
        ]
    )


def life_context_current_question_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Так", callback_data="life_context:answer:yes"),
                InlineKeyboardButton(text="Не зовсім", callback_data="life_context:answer:free"),
                InlineKeyboardButton(text="Ні", callback_data="life_context:answer:no"),
            ],
            [
                InlineKeyboardButton(text="Пропустити", callback_data="life_context:answer:skip"),
                InlineKeyboardButton(text="Зупинити", callback_data="life_context:answer:stop"),
            ],
        ]
    )


def life_context_open_question_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Поясню словами", callback_data="life_context:answer:free")],
            [
                InlineKeyboardButton(text="Пропустити", callback_data="life_context:answer:skip"),
                InlineKeyboardButton(text="Зупинити", callback_data="life_context:answer:stop"),
            ],
        ]
    )


def life_context_rewrite_confirmation_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Так, оновити", callback_data="life_context:rewrite:apply")],
            [InlineKeyboardButton(text="Скасувати", callback_data="life_context:rewrite:cancel")],
        ]
    )


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
            [
                InlineKeyboardButton(text="Емоції", callback_data=f"{prefix}:emotions"),
                InlineKeyboardButton(text="Патерни", callback_data=f"{prefix}:patterns"),
            ],
            [InlineKeyboardButton(text="Повороти", callback_data=f"{prefix}:turning_points")],
            [InlineKeyboardButton(text="Дні періоду", callback_data=f"{prefix}:days")],
            [InlineKeyboardButton(text="Головне меню", callback_data="nav:home")],
        ]
    )


def settings_keyboard(*, user_settings: UserSettings) -> InlineKeyboardMarkup:
    settings_json = user_settings.settings_json or {}
    paused = snapshots_paused(user_settings)
    quiet_text = "✅ Контекстна тиша" if context_quiet_enabled(user_settings) else "Контекстна тиша"
    quiet_status = "Тиша активна" if quiet_is_active(user_settings) else "Тиха пауза"
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
                InlineKeyboardButton(text=quiet_status, callback_data="quiet:menu"),
                InlineKeyboardButton(text=quiet_text, callback_data="settings:toggle:context_quiet"),
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


def quiet_menu_keyboard(*, active: bool = False) -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton(text="1 год", callback_data="quiet:set:1h"),
            InlineKeyboardButton(text="2 год", callback_data="quiet:set:2h"),
        ],
        [
            InlineKeyboardButton(text="До вечора", callback_data="quiet:set:evening"),
            InlineKeyboardButton(text="До завтра", callback_data="quiet:set:tomorrow"),
        ],
        [InlineKeyboardButton(text="Вказати час", callback_data="quiet:custom")],
    ]
    if active:
        rows.append([InlineKeyboardButton(text="Скасувати паузу", callback_data="quiet:cancel")])
    rows.append([InlineKeyboardButton(text="Головне меню", callback_data="menu:main")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def quiet_offer_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="1 год", callback_data="quiet:set:1h"),
                InlineKeyboardButton(text="2 год", callback_data="quiet:set:2h"),
            ],
            [
                InlineKeyboardButton(text="Вказати час", callback_data="quiet:custom"),
                InlineKeyboardButton(text="Не треба", callback_data="quiet:offer:no"),
            ],
        ]
    )


def settings_rhythm_keyboard(*, user_settings: UserSettings) -> InlineKeyboardMarkup:
    paused = snapshots_paused(user_settings)
    active_text = "Увімкнути" if paused else "Поставити на паузу"
    active_action = "resume" if paused else "pause"
    freq = (user_settings.min_interval_minutes, user_settings.max_interval_minutes)
    reminder = user_settings.reminder_delay_minutes
    adaptive_text = "✅ Адаптивно" if adaptive_observation_enabled(user_settings) else "Адаптивно"
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
            [InlineKeyboardButton(text=adaptive_text, callback_data="settings:toggle:adaptive_observation")],
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
