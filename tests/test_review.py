from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from io import BytesIO
from types import SimpleNamespace
from uuid import uuid4
from zoneinfo import ZoneInfo

from PIL import Image

import mental_state_bot.services.review as review_module
from mental_state_bot.emotions import EMOTION_COLORS
from mental_state_bot.services.review import (
    AffectSpectrumPoint,
    DayTurningPoint,
    EmotionSignalPoint,
    PhotoMoment,
    _affect_spectrum_png,
    _affect_spectrum_point,
    _emotion_label,
    _emotion_signal_color,
    _emotion_timeline_png,
    _emotion_timeline_row,
    _entry_affective_states,
    _entry_chart_x_values,
    _entry_emotion_signals,
    _entry_emotions,
    _feature_score,
    _format_counts,
    _format_entry_line,
    _graphable_feature_score,
    _latest_feature_results_by_entry,
    _line_chart_png,
    _meaningful_pleasant_moments,
    _metric_graph_item,
    _metric_graph_report,
    _metrics_label_text,
    _sparkline,
    _story_material_lines,
    _truncate,
    format_cost_report,
    format_day_turning_point,
    format_day_turning_points,
    format_gap_report,
    format_photo_moments_view,
    format_similar_entries,
    format_summary_section,
)


async def test_cost_report_discloses_calls_without_a_known_price(monkeypatch) -> None:
    async def model_run_cost_totals(session, *, user_id, since):
        return {
            "runs": 3,
            "estimated_cost_usd": Decimal("0.012345"),
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "reasoning_tokens": 25,
            "total_tokens": 150,
            "estimated_runs": 2,
        }

    async def list_model_runs_since(session, *, user_id, since):
        return [
            SimpleNamespace(task_name="extract_entry_features", estimated_cost_usd=Decimal("0.012345"), total_tokens=150),
            SimpleNamespace(task_name="unknown", estimated_cost_usd=None, total_tokens=0),
            SimpleNamespace(task_name="unknown", estimated_cost_usd=None, total_tokens=0),
        ]

    monkeypatch.setattr(review_module.repo, "model_run_cost_totals", model_run_cost_totals)
    monkeypatch.setattr(review_module.repo, "list_model_runs_since", list_model_runs_since)

    text = await format_cost_report(object(), user=SimpleNamespace(id=uuid4()), days=7)

    assert "Оцінка вартості: $0.012345 (для 2/3 викликів)" in text
    assert "Без цінової оцінки: 1 з 3 викликів" in text


def test_truncate_compacts_and_limits_text() -> None:
    text = _truncate("дуже   багато     пробілів " + "x" * 50, 24)

    assert "  " not in text
    assert len(text) <= 24
    assert text.endswith("…")


def test_format_entry_line_includes_labels_and_quality() -> None:
    entry_id = "123"
    entry = SimpleNamespace(
        id=entry_id,
        local_timestamp=datetime(2026, 6, 29, 10, 30, tzinfo=ZoneInfo("Europe/Kyiv")),
        created_at=None,
        raw_text="лежу і не можу почати",
    )

    line = _format_entry_line(
        entry,
        quality_by_entry={entry_id: "partial"},
        labels_by_entry={entry_id: ["лежання", "важко почати"]},
    )

    assert line.startswith("10:30 - лежу")
    assert "[лежання, важко почати]" in line
    assert "(частково)" in line


def test_format_entry_line_converts_utc_timestamp_to_user_timezone() -> None:
    entry_id = "123"
    entry = SimpleNamespace(
        id=entry_id,
        local_timestamp=datetime(2026, 6, 29, 19, 45, tzinfo=UTC),
        created_at=None,
        raw_text="лягаю спати",
    )

    line = _format_entry_line(
        entry,
        quality_by_entry={},
        labels_by_entry={},
        timezone="Europe/Kyiv",
    )

    assert line.startswith("22:45 - лягаю спати")


def test_format_similar_entries_empty() -> None:
    assert "поки не знайшов" in format_similar_entries([])


def test_format_similar_entries_summarizes_memory_matches() -> None:
    first_id = uuid4()
    second_id = uuid4()
    records = [
        SimpleNamespace(
            target_type="entry",
            target_id=first_id,
            source_text="Raw: Збираюся гуляти. Настрій змішаний.\nMicro-summary: прогулянка",
            created_at=datetime(2026, 6, 30, 12, 0, tzinfo=UTC),
        ),
        SimpleNamespace(
            target_type="entry",
            target_id=second_id,
            source_text="Raw: Вийшов пройтися й випити кави.",
            created_at=datetime(2026, 6, 29, 12, 0, tzinfo=UTC),
        ),
    ]
    entries = [
        SimpleNamespace(
            id=first_id,
            local_timestamp=datetime(2026, 6, 30, 15, 0, tzinfo=UTC),
            created_at=None,
            raw_text="Збираюся гуляти. Настрій змішаний.",
        ),
        SimpleNamespace(
            id=second_id,
            local_timestamp=datetime(2026, 6, 29, 16, 0, tzinfo=UTC),
            created_at=None,
            raw_text="Вийшов пройтися й випити кави.",
        ),
    ]
    analyses = [
        SimpleNamespace(
            task_name="extract_entry_features",
            target_id=first_id,
            result={"activity_labels": ["прогулянка"], "state_labels": ["змішаний стан"]},
        ),
        SimpleNamespace(
            task_name="extract_entry_features",
            target_id=second_id,
            result={"activity_labels": ["прогулянка"], "state_labels": ["спокій"]},
        ),
    ]

    text = format_similar_entries(
        records,
        query="гуляю",
        entries=entries,
        analyses=analyses,
        timezone="Europe/Kyiv",
    )

    assert "Пам’ять за запитом: «гуляю»" in text
    assert "Що може повторюватися" in text
    assert "прогулянка (2)" in text
    assert "Відкрити день: /day 2026-06-30" in text


def test_feature_score_maps_common_values() -> None:
    assert _feature_score({"value": "very_low"}) == 1
    assert _feature_score({"value": "neutral"}) == 4
    assert _feature_score({"value": "very_high"}) == 8
    assert _feature_score({"value": "низько"}) == 2
    assert _feature_score({"value": "нормальний"}) == 4
    assert _feature_score({"value": "гарний"}) == 7
    assert _feature_score({"value": 6}) == 6
    assert _feature_score({"value": "10"}) == 10
    assert _feature_score({"value": 0}) == 0
    assert _feature_score({"value": "unclear"}) is None
    assert _feature_score(None) is None


def test_graphable_feature_score_respects_new_schema_flags() -> None:
    legacy_result = {"mood": {"value": "high", "confidence": 0.8}}
    assert _graphable_feature_score(legacy_result, "mood") == 7

    blocked_result = {
        "entry_type": "activity_only",
        "mood": {"value": "high", "confidence": 0.8},
        "mood_evidence": "гарний настрій",
        "mood_reasoning_type": "direct_text",
        "should_graph_mood": False,
    }
    assert _graphable_feature_score(blocked_result, "mood") is None

    graphable_result = {
        "entry_type": "current_state",
        "mood": {"value": "high", "confidence": 0.8},
        "mood_evidence": "гарний настрій",
        "mood_reasoning_type": "direct_text",
        "should_graph_mood": True,
    }
    assert _graphable_feature_score(graphable_result, "mood") == 7


def test_metric_graph_report_explains_structural_skip_reasons() -> None:
    results = [
        {
            "entry_type": "current_state",
            "mood": {"value": "high", "confidence": 0.8},
            "mood_evidence": "гарний настрій",
            "mood_reasoning_type": "direct_text",
            "should_graph_mood": True,
        },
        {
            "entry_type": "activity_only",
            "mood": {"value": "unclear", "confidence": 0.0},
            "should_graph_mood": False,
        },
        {
            "entry_type": "current_state",
            "mood": {"value": "low", "confidence": 0.8},
            "mood_reasoning_type": "direct_text",
            "should_graph_mood": False,
        },
    ]
    items = [_metric_graph_item(result, "mood", _graphable_feature_score(result, "mood")) for result in results]
    report = _metric_graph_report(items)

    assert report.total == 3
    assert report.graphable == 1
    assert report.skipped_reasons["лише активність"] == 1
    assert report.skipped_reasons["немає evidence"] == 1


def test_entry_chart_x_values_use_real_time_distance() -> None:
    entries = [
        SimpleNamespace(local_timestamp=datetime(2026, 7, 9, 10, 0, tzinfo=ZoneInfo("Europe/Kyiv")), created_at=None),
        SimpleNamespace(local_timestamp=datetime(2026, 7, 9, 10, 2, tzinfo=ZoneInfo("Europe/Kyiv")), created_at=None),
        SimpleNamespace(local_timestamp=datetime(2026, 7, 9, 10, 42, tzinfo=ZoneInfo("Europe/Kyiv")), created_at=None),
    ]

    assert _entry_chart_x_values(entries, "Europe/Kyiv") == [0.0, 2.0, 42.0]


def test_story_material_lines_separate_story_data_from_graph_points() -> None:
    first = SimpleNamespace(id="first")
    second = SimpleNamespace(id="second")
    lines = _story_material_lines(
        entries=[first, second],
        extraction_by_entry={
            "first": {
                "entry_type": "activity_only",
                "activity_labels": ["прогулянка"],
                "state_labels": [],
                "emotion_labels": [],
            },
            "second": {
                "entry_type": "current_state",
                "activity_labels": [],
                "state_labels": ["втома"],
                "emotion_labels": ["сум"],
            },
        },
        mood_points=[None, 3],
        energy_points=[None, None],
    )

    assert "- AI-аналіз є для 2/2 записів." in lines
    assert "- Мітки активностей/станів/емоцій є у 2 записах." in lines
    assert "- Корисні для історії, але без надійних mood/energy точок: 1." in lines
    assert any("лише активність 1" in line for line in lines)


def test_sparkline_handles_missing_and_known_points() -> None:
    sparkline = _sparkline([1, None, 4, 8])

    assert sparkline.startswith("▁·▃▆")
    assert "сер=4.3" in sparkline
    assert "даних=3/4" in sparkline


def test_sparkline_handles_no_known_points() -> None:
    assert _sparkline([None, None]) == "··  даних мало"


def test_line_chart_png_generates_valid_png() -> None:
    png = _line_chart_png({"mood": [1, None, 4, 8], "energy": [2, 3, None, 7]})

    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png) > 1000


def test_emotion_helpers_extract_multiple_ukrainian_emotions() -> None:
    assert _emotion_label("страх") == "страх"
    assert _emotion_label("безнадія") is None
    assert _emotion_label("fear") is None
    assert _entry_emotions(
        {
            "emotion_labels": ["тривога", "тепло"],
            "state_labels": ["спокій", "fear", "тривога"],
        }
    ) == ["тривога"]
    assert _entry_affective_states(
        {"affective_states": [{"label": "безнадія"}, {"label": "спокій"}]}
    ) == ["безнадія", "спокій"]


def test_entry_emotion_signals_prefers_current_structured_emotions() -> None:
    result = {
        "emotions": [
            {
                "label": "надія",
                "intensity_level": "strong",
                "intensity": 0.2,
                "confidence": 0.8,
                "evidence": "побачив надію",
                "time_scope": "current",
            },
            {
                "label": "страх",
                "intensity_level": "strong",
                "confidence": 0.9,
                "evidence": "говорили про страх",
                "time_scope": "mentioned_not_felt",
            },
        ],
        "emotion_labels": ["сум"],
    }
    signals = _entry_emotion_signals(result)

    assert [(signal.label, signal.intensity) for signal in signals] == [("надія", 0.8)]
    assert _entry_emotions(result) == ["надія"]


def test_non_strict_structured_label_does_not_hide_strict_fallback_emotion() -> None:
    result = {
        "emotions": [
            {
                "label": "самотність",
                "intensity_level": "strong",
                "confidence": 0.8,
                "evidence": "дуже самотньо",
                "time_scope": "current",
            }
        ],
        "emotion_labels": ["сум"],
    }

    signals = _entry_emotion_signals(result)

    assert [(signal.label, signal.intensity_level) for signal in signals] == [("сум", "moderate")]
    assert _entry_emotions(result) == ["сум"]


def test_emotion_timeline_png_generates_valid_png() -> None:
    png = _emotion_timeline_png(
        [["сум"], ["тривога", "радість"], [], ["сум"]],
        labels=["10:00", "10:40", "11:20", "12:00"],
    )

    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png) > 1000


def test_affect_spectrum_uses_only_graphable_mood_points() -> None:
    point = _affect_spectrum_point(
        {
            "entry_type": "current_state",
            "mood": {"value": "high", "confidence": 0.8},
            "mood_evidence": "explicit evidence",
            "mood_reasoning_type": "direct_text",
            "should_graph_mood": True,
            "emotions": [
                {
                    "label": "радість",
                    "intensity": 0.8,
                    "confidence": 0.9,
                    "time_scope": "current",
                    "evidence": "explicit evidence",
                }
            ],
        }
    )

    assert point is not None
    assert point.tone == 0.7
    assert point.color == EMOTION_COLORS["радість"]
    assert _affect_spectrum_point({"should_graph_mood": False, "mood": {"value": "high"}}) is None


def test_affect_spectrum_png_generates_valid_png_with_a_gap() -> None:
    png = _affect_spectrum_png(
        [
            AffectSpectrumPoint(tone=0.3, color=(59, 130, 246), confidence=0.8),
            None,
            AffectSpectrumPoint(tone=0.7, color=(22, 163, 74), confidence=0.8),
        ],
        width=900,
        height=500,
        labels=["09:00", "10:00", "11:00"],
        x_values=[0.0, 60.0, 120.0],
    )

    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png) > 1000


def test_affect_spectrum_png_generates_valid_png_with_turning_markers() -> None:
    png = _affect_spectrum_png(
        [
            AffectSpectrumPoint(tone=0.7, color=(22, 163, 74), confidence=0.8),
            AffectSpectrumPoint(tone=0.25, color=(59, 130, 246), confidence=0.8),
        ],
        width=900,
        height=500,
        turning_point_indexes={1: 1},
    )

    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png) > 1000


def test_format_day_turning_point_anchors_the_summary_to_the_original_entry() -> None:
    entry = SimpleNamespace(
        local_timestamp=datetime(2026, 7, 11, 14, 20, tzinfo=ZoneInfo("Europe/Kyiv")),
        created_at=None,
        raw_text="Короткий запис про помітну зміну стану.",
    )
    point = DayTurningPoint(
        entry=entry,
        title="Помітна зміна",
        change="Стан відчутно змінився.",
        confidence=0.8,
    )

    assert "14:20" in format_day_turning_points([point], timezone="Europe/Kyiv")
    detail = format_day_turning_point(point, timezone="Europe/Kyiv", index=1)
    assert "Опорний запис" in detail
    assert entry.raw_text in detail


def test_emotion_timeline_png_normalizes_real_time_x_values() -> None:
    png = _emotion_timeline_png(
        [["сум"], ["радість"]],
        width=800,
        height=500,
        x_values=[0.0, 600.0],
    )
    image = Image.open(BytesIO(png)).convert("RGB")
    colored_pixels_on_right = sum(
        1
        for x in range(600, 760)
        for y in range(120, 360)
        if image.getpixel((x, y)) not in {(255, 255, 255), (250, 252, 255), (241, 245, 249)}
    )

    assert colored_pixels_on_right > 20


def test_emotion_timeline_row_marks_explicitly_non_emotional_moment_as_observed() -> None:
    row = _emotion_timeline_row({"emotion_observation": "no_current_emotion"})

    assert row.signals == []
    assert row.observed is True


def test_emotion_timeline_row_keeps_unknown_moment_as_gap() -> None:
    row = _emotion_timeline_row({"emotion_observation": "unclear"})

    assert row.signals == []
    assert row.observed is False


def test_emotion_timeline_png_keeps_unknown_moment_as_visual_break() -> None:
    rows = [
        _emotion_timeline_row({"emotion_labels": ["сум"]}),
        _emotion_timeline_row({"emotion_observation": "unclear"}),
        _emotion_timeline_row({"emotion_labels": ["сум"]}),
    ]

    png = _emotion_timeline_png(rows, width=500, height=400)

    assert png.startswith(b"\x89PNG\r\n\x1a\n")


def test_emotion_timeline_png_does_not_treat_missing_emotion_as_zero() -> None:
    png = _emotion_timeline_png(
        [
            [EmotionSignalPoint(label="сум", intensity=0.8, confidence=0.9)],
            [EmotionSignalPoint(label="радість", intensity=0.7, confidence=0.9)],
        ],
        width=800,
        height=500,
        x_values=[0.0, 600.0],
    )
    image = Image.open(BytesIO(png)).convert("RGB")
    blue = EMOTION_COLORS["сум"]
    blue_pixels_near_second_lane = sum(
        1
        for x in range(580, 740)
        for y in range(210, 310)
        if image.getpixel((x, y)) == blue
    )

    assert blue_pixels_near_second_lane == 0


def test_emotion_signal_color_gets_stronger_with_intensity() -> None:
    base = EMOTION_COLORS["сум"]

    weak = _emotion_signal_color(base, intensity=0.2, confidence=0.8)
    strong = _emotion_signal_color(base, intensity=0.9, confidence=0.8)

    weak_distance = sum(abs(weak[index] - base[index]) for index in range(3))
    strong_distance = sum(abs(strong[index] - base[index]) for index in range(3))
    assert strong_distance < weak_distance


def test_format_counts_sorts_by_count_then_name() -> None:
    text = _format_counts({"b": 2, "a": 2, "c": 1})

    assert text.splitlines()[:3] == ["- a: 2", "- b: 2", "- c: 1"]


def test_metrics_label_text_hides_non_normalized_english_labels() -> None:
    assert _metrics_label_text("спокій") == "спокій"
    assert _metrics_label_text("fear") is None
    assert _metrics_label_text("feeling punished") is None


def test_meaningful_pleasant_moments_filters_noise() -> None:
    moments = _meaningful_pleasant_moments(["", "ок", "немає", "теплі обійми уві сні", "спокійне читання"])

    assert moments == ["теплі обійми уві сні", "спокійне читання"]


def test_latest_feature_results_prefers_newer_ai_analysis() -> None:
    entry_id = uuid4()
    analyses = [
        SimpleNamespace(
            task_name="extract_entry_features",
            target_id=entry_id,
            result={"state_labels": ["calm"]},
        ),
        SimpleNamespace(
            task_name="extract_entry_features",
            target_id=entry_id,
            result={"state_labels": ["спокій"]},
        ),
    ]

    assert _latest_feature_results_by_entry(analyses)[str(entry_id)] == {"state_labels": ["спокій"]}


def test_format_photo_moments_view_separates_photos_from_analysis() -> None:
    tz = ZoneInfo("Europe/Kyiv")
    moment = PhotoMoment(
        media=SimpleNamespace(created_at=datetime(2026, 6, 29, 14, 5, tzinfo=tz)),
        entry=SimpleNamespace(
            local_timestamp=datetime(2026, 6, 29, 14, 0, tzinfo=tz),
            created_at=None,
            raw_text="на столі чай і ноутбук",
        ),
    )

    text = format_photo_moments_view([moment])

    assert "Фото дня: 1" in text
    assert "Окремо надсилаю фото нижче" in text
    assert "14:00 — на столі чай" in text


def test_format_gap_report_shows_notable_pauses_and_missed_prompts() -> None:
    tz = ZoneInfo("Europe/Kyiv")
    missed = SimpleNamespace(
        missed_at=datetime(2026, 6, 29, 12, 0, tzinfo=tz),
        status="open",
    )

    text = format_gap_report(
        entries=[
            SimpleNamespace(local_timestamp=datetime(2026, 6, 29, 10, 0, tzinfo=tz), created_at=None),
            SimpleNamespace(local_timestamp=datetime(2026, 6, 29, 13, 30, tzinfo=tz), created_at=None),
        ],
        missed_prompts=[missed],
        window_start=datetime(2026, 6, 29, 9, 0, tzinfo=tz),
        window_end=datetime(2026, 6, 29, 15, 0, tzinfo=tz),
    )

    assert "Прогалини сьогодні" in text
    assert "Записів у вікні: 2" in text
    assert "Пропущених/нагаданих зрізів: 1" in text
    assert "10:00-13:30: 3 год 30 хв" in text
    assert "12:00: відкрито" in text


def test_format_gap_report_converts_entry_times_to_window_timezone() -> None:
    kyiv = ZoneInfo("Europe/Kyiv")

    text = format_gap_report(
        entries=[
            SimpleNamespace(local_timestamp=datetime(2026, 6, 29, 12, 31, tzinfo=UTC), created_at=None),
            SimpleNamespace(local_timestamp=datetime(2026, 6, 29, 16, 45, tzinfo=UTC), created_at=None),
        ],
        missed_prompts=[],
        window_start=datetime(2026, 6, 29, 9, 0, tzinfo=kyiv),
        window_end=datetime(2026, 6, 29, 22, 46, tzinfo=kyiv),
    )

    assert "Перший запис: 15:31" in text
    assert "Останній запис: 19:45" in text


def test_format_gap_report_includes_missed_prompt_reason() -> None:
    tz = ZoneInfo("Europe/Kyiv")
    missed = SimpleNamespace(
        missed_at=datetime(2026, 6, 29, 12, 0, tzinfo=tz),
        status="explained",
        reason_text="не було ресурсу відповідати",
    )

    text = format_gap_report(
        entries=[],
        missed_prompts=[missed],
        window_start=datetime(2026, 6, 29, 9, 0, tzinfo=tz),
        window_end=datetime(2026, 6, 29, 15, 0, tzinfo=tz),
    )

    assert "12:00: пояснено — не було ресурсу відповідати" in text


def test_format_summary_story_section() -> None:
    summary = SimpleNamespace(
        short_text="Коротко.",
        details={
            "story": "День почався повільно.",
            "actual_activities": ["лежання", "трохи телефону"],
            "state_changes": ["після обіду стало трохи ясніше"],
        },
    )

    text = format_summary_section(summary, "story")

    assert "Історія дня" in text
    assert "День почався повільно." in text
    assert "- лежання" in text
    assert "- після обіду стало трохи ясніше" in text


def test_format_summary_section_marks_stale_summary() -> None:
    summary = SimpleNamespace(
        short_text="Старий короткий підсумок.",
        details={
            "story": "Старий текст.",
            "stale": {"reason": "entry_deleted", "marked_at": "2026-07-01T10:00:00+00:00"},
        },
    )

    text = format_summary_section(summary, "story")

    assert text.startswith("Підсумок може бути застарілим: запис було видалено.")
    assert "Оновити підсумок" in text
    assert "Старий текст." in text


def test_format_summary_metrics_section_with_gaps() -> None:
    summary = SimpleNamespace(
        id=uuid4(),
        short_text="Коротко.",
        details={
            "hardest_interval": "13:00-16:00",
            "best_or_stablest_interval": "вечір",
            "pleasant_moments": ["чай"],
            "cautious_observations": ["після паузи важче стартувати"],
            "data_quality": "partial",
        },
    )

    text = format_summary_section(summary, "metrics")

    assert "Найважчий відрізок: 13:00-16:00" in text
    assert "- чай" in text
    assert "Якість даних: частково" in text
