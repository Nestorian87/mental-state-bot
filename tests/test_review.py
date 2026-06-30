from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4
from zoneinfo import ZoneInfo

from mental_state_bot.services.review import (
    PhotoMoment,
    _feature_score,
    _format_counts,
    _format_entry_line,
    _latest_feature_results_by_entry,
    _line_chart_png,
    _sparkline,
    _truncate,
    format_gap_report,
    format_photo_moments_view,
    format_similar_entries,
    format_summary_section,
)


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


def test_feature_score_maps_common_values() -> None:
    assert _feature_score({"value": "very_low"}) == 1
    assert _feature_score({"value": "neutral"}) == 4
    assert _feature_score({"value": "very_high"}) == 8
    assert _feature_score({"value": "unclear"}) is None
    assert _feature_score(None) is None


def test_sparkline_handles_missing_and_known_points() -> None:
    sparkline = _sparkline([1, None, 4, 8])

    assert sparkline.startswith("▁·▄█")
    assert "сер=4.3" in sparkline


def test_sparkline_handles_no_known_points() -> None:
    assert _sparkline([None, None]) == "··  даних мало"


def test_line_chart_png_generates_valid_png() -> None:
    png = _line_chart_png({"mood": [1, None, 4, 8], "energy": [2, 3, None, 7]})

    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png) > 1000


def test_format_counts_sorts_by_count_then_name() -> None:
    text = _format_counts({"b": 2, "a": 2, "c": 1})

    assert text.splitlines()[:3] == ["- a: 2", "- b: 2", "- c: 1"]


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
