from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace
from uuid import uuid4
from zoneinfo import ZoneInfo

from mental_state_bot.services.period_analysis import (
    _build_period_analysis_from_records,
    compare_period_analyses,
)


def test_period_analysis_keeps_single_observation_out_of_repeated_patterns() -> None:
    day_id = uuid4()
    entry_id = uuid4()
    entry = SimpleNamespace(
        id=entry_id,
        day_id=day_id,
        local_timestamp=datetime(2026, 7, 1, 11, 0, tzinfo=ZoneInfo("Europe/Kyiv")),
        created_at=None,
    )
    features = {
        str(entry_id): {
            "should_graph_mood": True,
            "mood": {"value": 7},
            "emotions": [
                {
                    "label": "радість",
                    "intensity": 0.8,
                    "confidence": 0.9,
                    "evidence": "радісно",
                    "time_scope": "current",
                }
            ],
            "activity_labels": ["прогулянка"],
            "data_quality": "sufficient",
        }
    }

    analysis = _build_period_analysis_from_records(
        entries=[entry],
        features_by_entry=features,
        days_by_id={str(day_id): SimpleNamespace(id=day_id, local_date=date(2026, 7, 1))},
        daily_summaries=[],
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 2),
        timezone="Europe/Kyiv",
    )

    assert analysis["coverage"]["mood"] == {"points": 1, "days": 1}
    assert analysis["emotions"]["frequency"] == [{"label": "радість", "observations": 1}]
    assert analysis["repeated_associations"] == []


def test_period_analysis_reports_repeated_same_record_observations_without_causality() -> None:
    day_id = uuid4()
    first_id = uuid4()
    second_id = uuid4()
    entries = [
        SimpleNamespace(
            id=entry_id,
            day_id=day_id,
            local_timestamp=datetime(2026, 7, index, 11, tzinfo=ZoneInfo("Europe/Kyiv")),
            created_at=None,
        )
        for index, entry_id in enumerate((first_id, second_id), start=1)
    ]
    features = {
        str(entry.id): {
            "emotions": [
                {
                    "label": "інтерес",
                    "intensity": 0.55,
                    "confidence": 0.9,
                    "evidence": "цікаво працювати",
                    "time_scope": "current",
                }
            ],
            "activity_labels": ["робота над проєктом"],
            "data_quality": "sufficient",
        }
        for entry in entries
    }

    analysis = _build_period_analysis_from_records(
        entries=entries,
        features_by_entry=features,
        days_by_id={str(day_id): SimpleNamespace(id=day_id, local_date=date(2026, 7, 1))},
        daily_summaries=[],
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 1),
        timezone="Europe/Kyiv",
    )

    assert analysis["repeated_associations"] == [
        {
            "kind": "activity_emotion",
            "activity": "робота над проєктом",
            "emotion": "інтерес",
            "observations": 2,
        }
    ]


def test_period_comparison_keeps_raw_coverage_alongside_metric_change() -> None:
    current = {
        "coverage": {"active_days": 5, "total_days": 7, "mood": {"points": 8}, "energy": {"points": 4}},
        "trajectory": {"mood_mean": 5.5, "energy_mean": 4.0},
        "emotions": {"frequency": [{"label": "сум", "observations": 3}]},
    }
    previous = {
        "coverage": {"active_days": 2, "total_days": 7, "mood": {"points": 2}, "energy": {"points": 1}},
        "trajectory": {"mood_mean": 4.0, "energy_mean": 5.0},
        "emotions": {"frequency": [{"label": "сум", "observations": 1}]},
    }

    comparison = compare_period_analyses(current, previous)

    assert comparison["coverage_ratio"] == 0.4
    assert comparison["mood_mean_change"] == 1.5
    assert comparison["energy_mean_change"] == -1.0
    assert comparison["previous_coverage"]["active_days"] == 2
