from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from mental_state_bot.bot.handlers import (
    MANUAL_ENTRY_ACTIONS,
    VOICE_TRANSCRIPTION_ACTIONS,
    VoiceNoteTranscription,
    _archive_export_options,
    _command_argument,
    _format_semantic_memory_influences,
    _format_settings_text,
    _frequency_preset_values,
    _help_text,
    _inline_reply_keyboard,
    _is_previous_period_query,
    _is_sleep_marker_text,
    _manual_entry_confirmation_text,
    _missed_reason_text,
    _parse_date_range_query,
    _parse_day_query,
    _parse_quiet_until_text,
    _parse_reanalysis_action,
    _pending_manual_entry_payload,
    _pending_voice_note_payload,
    _split_telegram_text,
    _valid_hhmm,
    _voice_note_from_pending,
    _voice_transcription_preview,
)
from mental_state_bot.bot.keyboards import (
    EMOTION_CALIBRATION_OPTIONS,
    clarifications_menu_keyboard,
    correction_keyboard,
    data_menu_keyboard,
    day_detail_keyboard,
    day_menu_keyboard,
    deferred_clarification_keyboard,
    emotion_calibration_keyboard,
    emotion_intensity_keyboard,
    entry_delete_confirmation_keyboard,
    entry_management_keyboard,
    interpretation_keyboard,
    main_menu_keyboard,
    main_reply_keyboard,
    manual_entry_confirmation_keyboard,
    memory_menu_keyboard,
    memory_rebuild_confirmation_keyboard,
    metric_score_keyboard,
    missed_prompt_keyboard,
    period_detail_keyboard,
    planned_event_offer_keyboard,
    quiet_menu_keyboard,
    quiet_offer_keyboard,
    reanalysis_confirmation_keyboard,
    settings_capture_keyboard,
    settings_rhythm_keyboard,
    settings_style_keyboard,
    sleep_reflection_keyboard,
    summary_detail_keyboard,
    voice_transcription_keyboard,
    wake_time_keyboard,
)
from mental_state_bot.services.preferences import (
    clarification_queue,
    context_quiet_enabled,
    custom_interaction_style,
    pending_clarification,
    pending_correction_entry_id,
    pending_input,
    pending_life_context_review,
    pending_manual_entry,
    pending_voice_transcript,
    quiet_is_active,
    settings_json_with_clarification_queue,
    settings_json_with_context_quiet,
    settings_json_with_custom_interaction_style,
    settings_json_with_pending_clarification,
    settings_json_with_pending_correction_entry_id,
    settings_json_with_pending_input,
    settings_json_with_pending_life_context_review,
    settings_json_with_pending_manual_entry,
    settings_json_with_pending_voice_transcript,
    settings_json_with_quiet_until,
    settings_json_with_snapshot_pause,
    settings_json_without_pending_input,
    settings_json_without_pending_manual_entry,
    settings_json_without_pending_voice_transcript,
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


def test_main_reply_keyboard_uses_contextual_placeholder() -> None:
    keyboard = main_reply_keyboard("Напиши, що я зрозумів не так")

    assert keyboard.input_field_placeholder == "Напиши, що я зрозумів не так"


def test_main_reply_keyboard_limits_long_placeholder() -> None:
    keyboard = main_reply_keyboard("x" * 80)

    assert keyboard.input_field_placeholder == "x" * 64


def test_main_reply_keyboard_stays_small() -> None:
    keyboard = main_reply_keyboard()
    labels = [button.text for row in keyboard.keyboard for button in row]

    assert labels == ["Меню", "Новий зріз", "Пауза", "Лягаю спати"]


def test_quiet_keyboards_expose_pause_actions() -> None:
    menu = quiet_menu_keyboard(active=True)
    offer = quiet_offer_keyboard()
    menu_callbacks = [button.callback_data for row in menu.inline_keyboard for button in row]
    offer_callbacks = [button.callback_data for row in offer.inline_keyboard for button in row]

    assert {"quiet:set:1h", "quiet:set:2h", "quiet:custom", "quiet:cancel"} <= set(menu_callbacks)
    assert {"quiet:set:1h", "quiet:set:2h", "quiet:offer:no"} <= set(offer_callbacks)


def test_help_text_mentions_core_commands() -> None:
    text = _help_text()

    assert "/snapshot" in text
    assert "/summary" in text
    assert "/day 2026-06-30" in text
    assert "/sleep" in text
    assert "/week prev" in text
    assert "/month prev" in text
    assert "/export" in text
    assert "/export_csv" in text
    assert "/export_zip" in text
    assert "/pause" in text
    assert "/resume" in text
    assert "/gaps" in text
    assert "/audit" in text
    assert "/visual_report" in text
    assert "голосові" in text


def test_data_menu_exposes_visual_report() -> None:
    keyboard = data_menu_keyboard()
    callbacks = {
        button.callback_data
        for row in keyboard.inline_keyboard
        for button in row
        if button.callback_data
    }

    assert "menu:data:visual_report" in callbacks


def test_voice_transcription_preview_quotes_and_truncates() -> None:
    text = _voice_transcription_preview("  " + "слово " * 300, limit=20)

    assert text.startswith("«слово")
    assert text.endswith("…»")
    assert len(text) <= 22


def test_voice_transcription_keyboard_exposes_confirm_fix_cancel() -> None:
    keyboard = voice_transcription_keyboard()
    callbacks = {
        button.callback_data
        for row in keyboard.inline_keyboard
        for button in row
        if button.callback_data
    }

    assert callbacks == {"voice:confirm", "voice:fix", "voice:cancel"}
    assert {callback.split(":", maxsplit=1)[1] for callback in callbacks} == VOICE_TRANSCRIPTION_ACTIONS


def test_metric_score_keyboard_uses_zero_to_ten_and_skip() -> None:
    entry_id = str(uuid4())
    keyboard = metric_score_keyboard(entry_id=entry_id, metric="mood")
    callbacks = [
        button.callback_data
        for row in keyboard.inline_keyboard
        for button in row
        if button.callback_data
    ]

    assert callbacks[:11] == [f"metric:{entry_id}:mood:{score}" for score in range(11)]
    assert callbacks[-1] == f"metric:{entry_id}:mood:skip"


def test_emotion_calibration_keyboard_has_basic_emotions_and_skip() -> None:
    entry_id = str(uuid4())
    compact_entry_id = entry_id.replace("-", "")
    keyboard = emotion_calibration_keyboard(entry_id=entry_id, selected=["радість", "сум"])
    selected_indexes = _encoded_emotion_indexes({"радість", "сум"})
    callbacks = {
        button.callback_data
        for row in keyboard.inline_keyboard
        for button in row
        if button.callback_data
    }
    labels = [button.text for row in keyboard.inline_keyboard for button in row]

    assert f"emotion:save:{compact_entry_id}:{selected_indexes}" in callbacks
    assert f"emotion:skip:{compact_entry_id}:{selected_indexes}" in callbacks
    assert f"emotion:custom:{compact_entry_id}:{selected_indexes}" in callbacks
    assert all(len(callback or "") <= 64 for callback in callbacks)
    assert any(label == "✓ Радість" for label in labels)
    assert any(label == "✓ Сум" for label in labels)


def test_deferred_clarification_keyboard_exposes_ai_options() -> None:
    keyboard = deferred_clarification_keyboard(item_id="item-id", options=["Ще тримається", "Стало слабше"])
    callbacks = {
        button.callback_data
        for row in keyboard.inline_keyboard
        for button in row
        if button.callback_data
    }

    assert "clarification:option:item-id:0" in callbacks
    assert "clarification:option:item-id:1" in callbacks
    assert "clarification:skip:item-id" in callbacks


def test_inline_clarification_keyboard_preserves_ai_options() -> None:
    keyboard = _inline_reply_keyboard(
        "clarification:item-id",
        options=("Варіант один", "Варіант два"),
    )
    callbacks = {
        button.callback_data
        for row in keyboard.inline_keyboard
        for button in row
        if button.callback_data
    }

    assert "clarification:option:item-id:0" in callbacks
    assert "clarification:option:item-id:1" in callbacks


def test_emotion_intensity_keyboard_keeps_selected_emotions_in_callback() -> None:
    entry_id = str(uuid4())
    compact_entry_id = entry_id.replace("-", "")
    keyboard = emotion_intensity_keyboard(entry_id=entry_id, selected=["радість", "сум"])
    selected_indexes = _encoded_emotion_indexes({"радість", "сум"})
    callbacks = {
        button.callback_data
        for row in keyboard.inline_keyboard
        for button in row
        if button.callback_data
    }
    labels = [button.text for row in keyboard.inline_keyboard for button in row]

    assert f"e:s:{compact_entry_id}:{selected_indexes}:uu:c:0:m" in callbacks
    assert f"e:s:{compact_entry_id}:{selected_indexes}:uu:c:0:s" in callbacks
    assert f"e:x:{compact_entry_id}:{selected_indexes}:uu:c:0" in callbacks
    assert "Це не поточні емоції" in labels
    assert all(len(callback or "") <= 64 for callback in callbacks)


def test_emotion_intensity_keyboard_keeps_each_selected_emotion_level_in_callback() -> None:
    entry_id = str(uuid4())
    compact_entry_id = entry_id.replace("-", "")
    selected_indexes = _encoded_emotion_indexes({"радість", "сум"})
    keyboard = emotion_intensity_keyboard(
        entry_id=entry_id,
        selected=["радість", "сум"],
        intensity_levels=["strong", "mild"],
        position=1,
        time_scope="mentioned_not_felt",
    )
    callbacks = {
        button.callback_data
        for row in keyboard.inline_keyboard
        for button in row
        if button.callback_data
    }

    assert f"e:d:{compact_entry_id}:{selected_indexes}:sm:n:1" in callbacks
    assert f"e:x:{compact_entry_id}:{selected_indexes}:sm:n:1" in callbacks
    assert all(len(callback or "") <= 64 for callback in callbacks)


def _encoded_emotion_indexes(selected: set[str]) -> str:
    mask = 0
    for index, emotion in enumerate(EMOTION_CALIBRATION_OPTIONS):
        if emotion in selected:
            mask |= 1 << index
    return f"h{mask:x}"


def test_planned_event_offer_keyboard_has_confirm_clarify_ignore_and_cancel() -> None:
    keyboard = planned_event_offer_keyboard()
    callbacks = {
        button.callback_data
        for row in keyboard.inline_keyboard
        for button in row
        if button.callback_data
    }

    assert callbacks == {
        "planned_event:confirm",
        "planned_event:clarify",
        "planned_event:ignore",
        "planned_event:cancel",
    }


def test_wake_time_keyboard_has_skip_action() -> None:
    keyboard = wake_time_keyboard()
    callbacks = {
        button.callback_data
        for row in keyboard.inline_keyboard
        for button in row
        if button.callback_data
    }

    assert callbacks == {"wake_time:skip"}


def test_sleep_reflection_keyboard_exposes_presets_custom_and_skip() -> None:
    keyboard = sleep_reflection_keyboard()
    callbacks = {
        button.callback_data
        for row in keyboard.inline_keyboard
        for button in row
        if button.callback_data
    }

    assert callbacks == {
        "sleep:reflect:hard",
        "sleep:reflect:mixed",
        "sleep:reflect:okay",
        "sleep:reflect:good",
        "sleep:reflect:custom",
        "sleep:reflect:skip",
    }


def test_manual_entry_confirmation_keyboard_exposes_save_and_ignore() -> None:
    keyboard = manual_entry_confirmation_keyboard()
    callbacks = {
        button.callback_data
        for row in keyboard.inline_keyboard
        for button in row
        if button.callback_data
    }

    assert callbacks == {"manual:save", "manual:ignore"}
    assert {callback.split(":", maxsplit=1)[1] for callback in callbacks} == MANUAL_ENTRY_ACTIONS


def test_day_query_parses_explicit_dates() -> None:
    assert _parse_day_query("2026-06-30", "Europe/Kyiv") == date(2026, 6, 30)
    assert _parse_day_query("30.06.2026", "Europe/Kyiv") == date(2026, 6, 30)
    assert _parse_day_query("", "Europe/Kyiv") is None
    assert _parse_day_query("колись", "Europe/Kyiv") is None


def test_day_query_uses_journal_today_when_provided() -> None:
    journal_today = date(2026, 7, 1)

    assert _parse_day_query("сьогодні", "Europe/Kyiv", today=journal_today) == journal_today
    assert _parse_day_query("вчора", "Europe/Kyiv", today=journal_today) == date(2026, 6, 30)


def test_date_range_query_parses_and_orders_dates() -> None:
    journal_today = date(2026, 7, 9)

    assert _parse_date_range_query("2026-07-01 2026-07-09", "Europe/Kyiv") == (
        date(2026, 7, 1),
        date(2026, 7, 9),
    )
    assert _parse_date_range_query("2026-07-09 2026-07-01", "Europe/Kyiv") == (
        date(2026, 7, 1),
        date(2026, 7, 9),
    )
    assert _parse_date_range_query("вчора сьогодні", "Europe/Kyiv", today=journal_today) == (
        date(2026, 7, 8),
        date(2026, 7, 9),
    )
    assert _parse_date_range_query("2026-07-01", "Europe/Kyiv") == (
        date(2026, 7, 1),
        date(2026, 7, 1),
    )
    assert _parse_date_range_query("не дата", "Europe/Kyiv") is None


def test_summary_detail_keyboard_scopes_callbacks_to_summary() -> None:
    summary_id = str(uuid4())
    keyboard = summary_detail_keyboard(summary_id=summary_id)
    callbacks = {
        button.callback_data
        for row in keyboard.inline_keyboard
        for button in row
        if button.callback_data
    }

    assert f"summary:{summary_id}:story" in callbacks
    assert f"summary:{summary_id}:timeline" in callbacks
    assert f"summary:{summary_id}:refresh" in callbacks
    assert "summary:timeline" not in callbacks


def test_day_detail_keyboard_scopes_callbacks_to_day() -> None:
    day_id = str(uuid4())
    keyboard = day_detail_keyboard(day_id=day_id)
    callbacks = {
        button.callback_data
        for row in keyboard.inline_keyboard
        for button in row
        if button.callback_data
    }

    assert f"dayview:{day_id}:story" in callbacks
    assert f"dayview:{day_id}:gaps" in callbacks
    assert f"dayview:{day_id}:entries" in callbacks
    assert f"dayview:{day_id}:refresh" in callbacks


def test_correction_keyboard_can_target_entry() -> None:
    entry_id = str(uuid4())
    keyboard = correction_keyboard(entry_id=entry_id)
    callbacks = {
        button.callback_data
        for row in keyboard.inline_keyboard
        for button in row
        if button.callback_data
    }

    assert callbacks == {f"correction:start:{entry_id}"}


def test_interpretation_keyboard_groups_metric_and_text_fixes() -> None:
    entry_id = str(uuid4())
    keyboard = interpretation_keyboard(entry_id=entry_id)
    callbacks = {
        button.callback_data
        for row in keyboard.inline_keyboard
        for button in row
        if button.callback_data
    }

    assert callbacks == {
        f"correction:start:{entry_id}",
        f"metric:start:{entry_id}:mood",
        f"metric:start:{entry_id}:energy",
        f"emotion:start:{entry_id}",
        "interpretation:ok",
    }


def test_life_context_pending_review_roundtrip() -> None:
    settings = SimpleNamespace(settings_json={})
    review = {"index": 0, "candidates": [{"label": "Проєкт"}]}
    settings.settings_json = settings_json_with_pending_life_context_review(settings, review)

    assert pending_life_context_review(settings) == review

    settings.settings_json = settings_json_with_pending_life_context_review(settings, None)

    assert pending_life_context_review(settings) is None


def test_entry_management_keyboards_require_confirmation() -> None:
    entry_id = str(uuid4())
    day_id = str(uuid4())
    management = entry_management_keyboard(day_id=day_id, entries=[(entry_id, "1. 10:00 - текст")])
    confirmation = entry_delete_confirmation_keyboard(entry_id=entry_id, day_id=day_id)
    management_callbacks = {
        button.callback_data
        for row in management.inline_keyboard
        for button in row
        if button.callback_data
    }
    confirmation_callbacks = {
        button.callback_data
        for row in confirmation.inline_keyboard
        for button in row
        if button.callback_data
    }

    assert f"entry:delete:{entry_id}" in management_callbacks
    assert f"entry:confirm_delete:{entry_id}" not in management_callbacks
    assert f"entry:confirm_delete:{entry_id}" in confirmation_callbacks
    assert f"dayview:{day_id}:entries" in confirmation_callbacks


def test_main_menu_groups_sections() -> None:
    keyboard = main_menu_keyboard()
    callbacks = {
        button.callback_data
        for row in keyboard.inline_keyboard
        for button in row
        if button.callback_data
    }

    assert {
        "menu:day",
        "menu:summaries",
        "menu:memory",
        "menu:life_context",
        "menu:data",
        "settings:open",
        "menu:clarifications",
    } <= callbacks
    assert len(keyboard.inline_keyboard) == 4


def test_submenus_expose_grouped_actions() -> None:
    day_callbacks = {
        button.callback_data
        for row in day_menu_keyboard().inline_keyboard
        for button in row
        if button.callback_data
    }
    memory_callbacks = {
        button.callback_data
        for row in memory_menu_keyboard(embeddings_enabled=True).inline_keyboard
        for button in row
        if button.callback_data
    }
    data_callbacks = {
        button.callback_data
        for row in data_menu_keyboard().inline_keyboard
        for button in row
        if button.callback_data
    }

    assert "menu:day:date" in day_callbacks
    assert "menu:memory:search" in memory_callbacks
    assert "menu:memory:maintain" in memory_callbacks
    assert "menu:memory:review" in memory_callbacks
    assert "menu:memory:rebuild" in memory_callbacks
    assert "archive:export_zip" in data_callbacks
    assert "menu:data:reanalyze" in data_callbacks


def test_reanalysis_confirmation_requires_explicit_confirm() -> None:
    keyboard = reanalysis_confirmation_keyboard(action="recent:10", selected=10)
    buttons = [button for row in keyboard.inline_keyboard for button in row]
    callbacks = {button.callback_data for button in buttons if button.callback_data}
    labels = {button.text for button in buttons}

    assert "features:reanalyze:recent:10" in callbacks
    assert "features:reanalyze:cancel" in callbacks
    assert "Так, переаналізувати 10" in labels


def test_reanalysis_action_supports_new_and_previous_buttons() -> None:
    assert _parse_reanalysis_action("recent:10") == ("recent", 10, None, None)
    assert _parse_reanalysis_action("range:20260701:20260703") == (
        "range",
        None,
        date(2026, 7, 1),
        date(2026, 7, 3),
    )
    assert _parse_reanalysis_action("all") == ("all", None, None, None)
    assert _parse_reanalysis_action("200") == ("recent", 200, None, None)
    assert _parse_reanalysis_action("range:nope:20260703") is None


def test_memory_influence_view_hides_raw_memory_and_shows_hypothesis() -> None:
    snapshot_id = uuid4()
    snapshot = SimpleNamespace(
        id=snapshot_id,
        prompted_at=datetime(2026, 7, 10, 12, 0, tzinfo=UTC),
        context_json={
            "semantic_memory_insight": {
                "used": True,
                "hypothesis": "схожий патерн очікування відповіді",
                "evidence_entry_ids": ["entry-1", "entry-2"],
            }
        },
    )
    prompt = SimpleNamespace(snapshot_id=snapshot_id, prompt_kind="initial", text="Що змінилося після очікування?")

    text = _format_semantic_memory_influences([snapshot], [prompt], timezone="Europe/Kyiv")

    assert "схожий патерн очікування відповіді" in text
    assert "2 схожих попередніх записів" in text
    assert "entry-1" not in text


def test_memory_rebuild_confirmation_requires_explicit_confirm() -> None:
    keyboard = memory_rebuild_confirmation_keyboard(limit=200)
    buttons = [button for row in keyboard.inline_keyboard for button in row]
    callbacks = {button.callback_data for button in buttons if button.callback_data}
    labels = {button.text for button in buttons}

    assert "memory:rebuild:200" in callbacks
    assert "memory:rebuild:cancel" in callbacks
    assert "Так, перебудувати 200" in labels


def test_clarification_queue_keyboards_expose_review_actions() -> None:
    active = deferred_clarification_keyboard(item_id="q1")
    active_callbacks = {
        button.callback_data
        for row in active.inline_keyboard
        for button in row
        if button.callback_data
    }
    menu = clarifications_menu_keyboard(has_queued=True, has_pending=False, has_clearable=True)
    menu_callbacks = {
        button.callback_data
        for row in menu.inline_keyboard
        for button in row
        if button.callback_data
    }

    assert "clarification:skip:q1" in active_callbacks
    assert "clarification_queue:open" in active_callbacks
    assert "clarification_queue:next" in menu_callbacks
    assert "clarification_queue:skip_all" in menu_callbacks


def test_period_detail_keyboard_scopes_callbacks_to_summary() -> None:
    summary_id = str(uuid4())
    keyboard = period_detail_keyboard(summary_id=summary_id)
    callbacks = {
        button.callback_data
        for row in keyboard.inline_keyboard
        for button in row
        if button.callback_data
    }

    assert f"periodview:{summary_id}:overview" in callbacks
    assert f"periodview:{summary_id}:timeline" in callbacks
    assert f"periodview:{summary_id}:chart" in callbacks
    assert f"periodview:{summary_id}:emotions" in callbacks
    assert f"periodview:{summary_id}:patterns" in callbacks
    assert f"periodview:{summary_id}:turning_points" in callbacks


def test_settings_keyboards_mark_active_options() -> None:
    settings = SimpleNamespace(
        settings_json={"custom_interaction_style": "коротко"},
        ask_body_signals=True,
        photo_prompts_enabled=False,
        min_interval_minutes=30,
        max_interval_minutes=70,
        reminder_delay_minutes=25,
        tone="calm",
        humanity_level="warm",
    )
    main_callbacks = {
        button.text
        for row in main_menu_keyboard().inline_keyboard
        for button in row
    }
    rhythm_labels = {
        button.text
        for row in settings_rhythm_keyboard(user_settings=settings).inline_keyboard
        for button in row
    }
    style_labels = {
        button.text
        for row in settings_style_keyboard(user_settings=settings).inline_keyboard
        for button in row
    }
    capture_labels = {
        button.text
        for row in settings_capture_keyboard(user_settings=settings).inline_keyboard
        for button in row
    }

    assert "Налаштування" in main_callbacks
    assert "✅ Норм" in rhythm_labels
    assert "✅ 25 хв" in rhythm_labels
    assert "✅ Спокійний" in style_labels
    assert "✅ Людяніше" in style_labels
    assert "✅ Питати про тіло" in capture_labels


def test_previous_period_query_aliases() -> None:
    assert _is_previous_period_query("prev")
    assert _is_previous_period_query("попередній")
    assert _is_previous_period_query("минула")
    assert not _is_previous_period_query("")
    assert not _is_previous_period_query("сьогодні")


def test_pending_voice_note_payload_roundtrip_keeps_original_transcript() -> None:
    run_id = uuid4()
    voice_note = VoiceNoteTranscription(
        text="В цілому я почуваюся добре.",
        original_text="В цілому я почуваюся добре.",
        original_path=Path("/tmp/voice.ogg"),
        transcription_path=Path("/tmp/voice.webm"),
        transcription_run_id=run_id,
        duration_seconds=3,
        mime_type="audio/ogg",
        file_size=1024,
        telegram_file_id="file-id",
        telegram_file_unique_id="unique-id",
    )

    payload = _pending_voice_note_payload(
        voice_note,
        telegram_message_id=10,
        reply_to_message_id=None,
        target_pending_kind="life_context_free_answer",
    )
    restored = _voice_note_from_pending(payload, text="В цілому ти почуваєшся добре.")

    assert payload["telegram_message_id"] == 10
    assert payload["target_pending_kind"] == "life_context_free_answer"
    assert restored.text == "В цілому ти почуваєшся добре."
    assert restored.original_text == "В цілому я почуваюся добре."
    assert restored.transcription_run_id == run_id
    assert restored.telegram_file_unique_id == "unique-id"


def test_voice_transcript_pending_state_roundtrip() -> None:
    settings = SimpleNamespace(settings_json={})
    settings.settings_json = settings_json_with_pending_voice_transcript(
        settings,
        {"text": "транскрипція"},
    )

    assert pending_input(settings) == "voice_transcript"
    assert pending_voice_transcript(settings) == {"text": "транскрипція"}

    settings.settings_json = settings_json_without_pending_voice_transcript(settings)

    assert pending_input(settings) is None
    assert pending_voice_transcript(settings) is None


def test_pending_correction_target_clears_with_pending_input() -> None:
    entry_id = str(uuid4())
    settings = SimpleNamespace(settings_json={"pending_input": "correction"})
    settings.settings_json = settings_json_with_pending_correction_entry_id(settings, entry_id)

    assert pending_correction_entry_id(settings) == entry_id

    settings.settings_json = settings_json_without_pending_input(settings)

    assert pending_input(settings) is None
    assert pending_correction_entry_id(settings) is None


def test_menu_pending_input_kinds_are_supported() -> None:
    for kind in ("day_date", "memory_search", "visual_report_range"):
        settings = SimpleNamespace(settings_json={})

        settings.settings_json = settings_json_with_pending_input(settings, kind)

        assert pending_input(settings) == kind


def test_manual_entry_pending_state_roundtrip() -> None:
    settings = SimpleNamespace(settings_json={"other": "value"})
    payload = {"text": "просто написав", "telegram_message_id": 42}

    settings.settings_json = settings_json_with_pending_manual_entry(settings, payload)

    assert pending_input(settings) is None
    assert pending_manual_entry(settings) == payload

    settings.settings_json = settings_json_without_pending_manual_entry(settings)

    assert settings.settings_json == {"other": "value"}
    assert pending_manual_entry(settings) is None


def test_manual_entry_payload_and_confirmation_text() -> None:
    message = SimpleNamespace(message_id=15, reply_to_message=None)
    payload = _pending_manual_entry_payload(message, "  важливий момент  ")
    text = _manual_entry_confirmation_text("важливий момент")

    assert payload["text"] == "  важливий момент  "
    assert payload["telegram_message_id"] == 15
    assert "поза відкритим зрізом" in text
    assert "важливий момент" in text


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


def test_deferred_clarification_preferences_roundtrip() -> None:
    settings = SimpleNamespace(settings_json={"other": "value"})
    item = {"id": "q1", "entry_id": "e1", "question": "Що було важливим?", "status": "active"}

    updated = settings_json_with_clarification_queue(settings, [item])
    updated = settings_json_with_pending_clarification(SimpleNamespace(settings_json=updated), item)
    view = SimpleNamespace(settings_json=updated)

    assert clarification_queue(view) == [item]
    assert pending_clarification(view) == item
    assert view.settings_json["other"] == "value"


def test_quiet_until_preferences_preserve_settings_json() -> None:
    settings = SimpleNamespace(settings_json={"other": "value"})
    until = datetime(2026, 7, 4, 15, 0, tzinfo=UTC)

    updated = settings_json_with_quiet_until(settings, until)

    assert updated["other"] == "value"
    assert quiet_is_active(SimpleNamespace(settings_json=updated), datetime(2026, 7, 4, 14, 0, tzinfo=UTC))
    assert not quiet_is_active(SimpleNamespace(settings_json=updated), datetime(2026, 7, 4, 16, 0, tzinfo=UTC))
    assert settings_json_with_quiet_until(SimpleNamespace(settings_json=updated), None) == {"other": "value"}


def test_context_quiet_preferences_default_enabled() -> None:
    settings = SimpleNamespace(settings_json={})

    assert context_quiet_enabled(settings)
    assert not context_quiet_enabled(SimpleNamespace(settings_json=settings_json_with_context_quiet(settings, False)))


def test_parse_quiet_until_text_accepts_time_and_duration() -> None:
    parsed_time = _parse_quiet_until_text("до 18:30", timezone="Europe/Kyiv")
    parsed_duration = _parse_quiet_until_text("на 2 години", timezone="Europe/Kyiv")

    assert parsed_time is not None
    assert parsed_duration is not None


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
