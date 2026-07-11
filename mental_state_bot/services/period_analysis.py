from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date
from itertools import combinations
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from mental_state_bot.db import repositories as repo
from mental_state_bot.db.models import Day, Entry, Summary, User
from mental_state_bot.services.review import (
    _data_quality_label,
    _entry_emotion_signals,
    _entry_journal_date,
    _entry_timestamp,
    _graphable_feature_score,
    _latest_feature_results_by_entry,
    _metric_entries,
    _metrics_label_text,
    _summaries_by_journal_date,
)
from mental_state_bot.time_utils import zoneinfo


async def build_period_analysis(
    session: AsyncSession,
    *,
    user: User,
    start_date: date,
    end_date: date,
    period_start,
    period_end,
) -> dict[str, Any]:
    days = list(
        await repo.list_days_between(
            session,
            user_id=user.id,
            start_date=start_date,
            end_date=end_date,
        )
    )
    days_by_id = {str(day.id): day for day in days}
    entries = _metric_entries(
        list(
            await repo.list_entries_between(
                session,
                user_id=user.id,
                start=period_start,
                end=period_end,
            )
        )
    )
    analyses = await repo.list_analyses_for_targets(
        session,
        target_type="entry",
        target_ids=[entry.id for entry in entries],
    )
    summaries = list(
        await repo.list_summaries_between(
            session,
            user_id=user.id,
            period_type="daily",
            start=period_start,
            end=period_end,
        )
    )
    return _build_period_analysis_from_records(
        entries=entries,
        features_by_entry=_latest_feature_results_by_entry(analyses),
        days_by_id=days_by_id,
        daily_summaries=summaries,
        start_date=start_date,
        end_date=end_date,
        timezone=user.timezone,
    )


def compare_period_analyses(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    current_coverage = current.get("coverage") or {}
    previous_coverage = previous.get("coverage") or {}
    current_active = int(current_coverage.get("active_days") or 0)
    previous_active = int(previous_coverage.get("active_days") or 0)
    more_active = max(current_active, previous_active, 1)
    return {
        "previous_coverage": {
            "active_days": previous_active,
            "total_days": int(previous_coverage.get("total_days") or 0),
            "mood_points": int((previous_coverage.get("mood") or {}).get("points") or 0),
            "energy_points": int((previous_coverage.get("energy") or {}).get("points") or 0),
            "emotion_observed_entries": int(previous_coverage.get("emotion_observed_entries") or 0),
        },
        "coverage_ratio": round(min(current_active, previous_active) / more_active, 2),
        "mood_mean_change": _difference(
            (current.get("trajectory") or {}).get("mood_mean"),
            (previous.get("trajectory") or {}).get("mood_mean"),
        ),
        "energy_mean_change": _difference(
            (current.get("trajectory") or {}).get("energy_mean"),
            (previous.get("trajectory") or {}).get("energy_mean"),
        ),
        "emotion_frequency_change": _counter_difference(
            (current.get("emotions") or {}).get("frequency") or [],
            (previous.get("emotions") or {}).get("frequency") or [],
        ),
    }


def _build_period_analysis_from_records(
    *,
    entries: list[Entry],
    features_by_entry: dict[str, dict[str, Any]],
    days_by_id: dict[str, Day],
    daily_summaries: list[Summary],
    start_date: date,
    end_date: date,
    timezone: str,
) -> dict[str, Any]:
    daily: dict[date, dict[str, Any]] = {
        current: _empty_day(current) for current in _date_range(start_date, end_date)
    }
    quality_counts: Counter[str] = Counter()
    emotion_frequency: Counter[str] = Counter()
    emotion_intensity: dict[str, list[float]] = defaultdict(list)
    emotion_pairs: Counter[tuple[str, str]] = Counter()
    activity_emotions: Counter[tuple[str, str]] = Counter()
    activity_mood: dict[str, list[int]] = defaultdict(list)
    activity_energy: dict[str, list[int]] = defaultdict(list)
    rhythm: dict[str, dict[str, Any]] = {
        label: _empty_rhythm_bucket(label) for label in ("ранок", "день", "вечір", "ніч")
    }

    for entry in entries:
        entry_date = _entry_journal_date(entry, days_by_id=days_by_id, timezone=timezone)
        if entry_date not in daily:
            continue
        result = features_by_entry.get(str(entry.id), {})
        day = daily[entry_date]
        day["entry_count"] += 1
        if result:
            day["analysed_entries"] += 1
        else:
            day["entries_without_analysis"] += 1
        quality_counts[_data_quality_label(result.get("data_quality"))] += 1

        mood = _graphable_feature_score(result, "mood")
        energy = _graphable_feature_score(result, "energy")
        if mood is not None:
            day["mood_values"].append(mood)
        if energy is not None:
            day["energy_values"].append(energy)

        signals = _entry_emotion_signals(result)
        labels = sorted({signal.label for signal in signals})
        if signals:
            day["emotion_observed_entries"] += 1
        for signal in signals:
            emotion_frequency[signal.label] += 1
            emotion_intensity[signal.label].append(signal.intensity)
            day["emotion_counts"][signal.label] += 1
        for left, right in combinations(labels, 2):
            emotion_pairs[(left, right)] += 1

        activities = _normalized_labels(result.get("activity_labels") or [])
        for activity in activities:
            day["activity_counts"][activity] += 1
            if mood is not None:
                activity_mood[activity].append(mood)
            if energy is not None:
                activity_energy[activity].append(energy)
            for emotion in labels:
                activity_emotions[(activity, emotion)] += 1

        timestamp = _entry_timestamp(entry, zoneinfo(timezone))
        if timestamp is not None:
            bucket = rhythm[_rhythm_bucket(timestamp.hour)]
            bucket["entry_count"] += 1
            if mood is not None:
                bucket["mood_values"].append(mood)
            if energy is not None:
                bucket["energy_values"].append(energy)
            for label in labels:
                bucket["emotion_counts"][label] += 1

    summaries_by_date = _summaries_by_journal_date(
        daily_summaries,
        days_by_id=days_by_id,
        timezone=timezone,
    )
    turning_points = []
    daily_rows = []
    for current in _date_range(start_date, end_date):
        row = daily[current]
        summary = summaries_by_date.get(current)
        changes = _state_changes(summary)
        if changes:
            turning_points.append({"date": current.isoformat(), "changes": changes[:5]})
        daily_rows.append(
            {
                "date": current.isoformat(),
                "entry_count": row["entry_count"],
                "analysed_entries": row["analysed_entries"],
                "mood_mean": _rounded_mean(row["mood_values"]),
                "energy_mean": _rounded_mean(row["energy_values"]),
                "emotion_observed_entries": row["emotion_observed_entries"],
                "top_emotions": _top_counts(row["emotion_counts"], limit=4),
                "top_activities": _top_counts(row["activity_counts"], limit=4),
                "state_changes": changes[:5],
            }
        )

    mood_values = [value for row in daily.values() for value in row["mood_values"]]
    energy_values = [value for row in daily.values() for value in row["energy_values"]]
    active_days = sum(1 for row in daily.values() if row["entry_count"])
    emotion_observed_entries = sum(row["emotion_observed_entries"] for row in daily.values())
    total_days = len(daily)
    return {
        "schema_version": "period_analysis.v1",
        "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        "coverage": {
            "total_days": total_days,
            "active_days": active_days,
            "entry_count": len(entries),
            "analysed_entries": sum(row["analysed_entries"] for row in daily.values()),
            "entries_without_analysis": sum(row["entries_without_analysis"] for row in daily.values()),
            "emotion_observed_entries": emotion_observed_entries,
            "mood": _metric_coverage(mood_values, daily_rows, "mood_mean"),
            "energy": _metric_coverage(energy_values, daily_rows, "energy_mean"),
            "data_quality": _top_counts(quality_counts, limit=8),
        },
        "trajectory": {
            "mood_mean": _rounded_mean(mood_values),
            "energy_mean": _rounded_mean(energy_values),
            "daily": daily_rows,
        },
        "emotions": {
            "frequency": _top_counts(emotion_frequency, limit=12),
            "mean_intensity": [
                {"emotion": label, "intensity": _rounded_mean(values, digits=2), "observations": len(values)}
                for label, values in sorted(
                    emotion_intensity.items(),
                    key=lambda item: (-len(item[1]), item[0]),
                )[:12]
            ],
            "co_occurrence": [
                {"emotions": [left, right], "observations": count}
                for (left, right), count in sorted(
                    emotion_pairs.items(),
                    key=lambda item: (-item[1], item[0]),
                )
                if count >= 2
            ][:10],
        },
        "rhythm": [
            {
                "period": label,
                "entry_count": item["entry_count"],
                "mood_mean": _rounded_mean(item["mood_values"]),
                "energy_mean": _rounded_mean(item["energy_values"]),
                "top_emotions": _top_counts(item["emotion_counts"], limit=4),
            }
            for label, item in rhythm.items()
        ],
        "repeated_associations": _repeated_associations(
            activity_emotions,
            activity_mood,
            activity_energy,
        ),
        "turning_points": turning_points[:12],
    }


def _empty_day(current: date) -> dict[str, Any]:
    return {
        "date": current,
        "entry_count": 0,
        "analysed_entries": 0,
        "entries_without_analysis": 0,
        "mood_values": [],
        "energy_values": [],
        "emotion_observed_entries": 0,
        "emotion_counts": Counter(),
        "activity_counts": Counter(),
    }


def _empty_rhythm_bucket(label: str) -> dict[str, Any]:
    return {"period": label, "entry_count": 0, "mood_values": [], "energy_values": [], "emotion_counts": Counter()}


def _metric_coverage(values: list[int], daily: list[dict[str, Any]], key: str) -> dict[str, int]:
    return {"points": len(values), "days": sum(1 for item in daily if item.get(key) is not None)}


def _repeated_associations(
    activity_emotions: Counter[tuple[str, str]],
    activity_mood: dict[str, list[int]],
    activity_energy: dict[str, list[int]],
) -> list[dict[str, Any]]:
    observations = [
        {"kind": "activity_emotion", "activity": activity, "emotion": emotion, "observations": count}
        for (activity, emotion), count in activity_emotions.items()
        if count >= 2
    ]
    for activity in set(activity_mood) | set(activity_energy):
        mood_values = activity_mood.get(activity, [])
        energy_values = activity_energy.get(activity, [])
        support = max(len(mood_values), len(energy_values))
        if support < 3:
            continue
        observations.append(
            {
                "kind": "activity_metrics",
                "activity": activity,
                "observations": support,
                "mood_mean": _rounded_mean(mood_values),
                "energy_mean": _rounded_mean(energy_values),
            }
        )
    return sorted(observations, key=lambda item: (-int(item["observations"]), str(item.get("activity") or "")))[:12]


def _normalized_labels(values: list[Any]) -> list[str]:
    return [label for value in values if (label := _metrics_label_text(value)) is not None]


def _state_changes(summary: Summary | None) -> list[str]:
    details = getattr(summary, "details", None) or {}
    values = details.get("state_changes") or []
    return [" ".join(str(value).split())[:360] for value in values if str(value).strip()]


def _top_counts(counter: Counter[str], *, limit: int) -> list[dict[str, Any]]:
    return [
        {"label": label, "observations": count}
        for label, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def _difference(current: Any, previous: Any) -> float | None:
    if not isinstance(current, int | float) or not isinstance(previous, int | float):
        return None
    return round(float(current) - float(previous), 2)


def _counter_difference(current: list[dict[str, Any]], previous: list[dict[str, Any]]) -> list[dict[str, Any]]:
    current_counts = {str(item.get("label") or ""): int(item.get("observations") or 0) for item in current}
    previous_counts = {str(item.get("label") or ""): int(item.get("observations") or 0) for item in previous}
    return [
        {"emotion": label, "change": current_counts.get(label, 0) - previous_counts.get(label, 0)}
        for label in sorted(set(current_counts) | set(previous_counts))
        if current_counts.get(label, 0) != previous_counts.get(label, 0)
    ][:10]


def _rounded_mean(values: list[int | float], *, digits: int = 1) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), digits)


def _rhythm_bucket(hour: int) -> str:
    if hour < 6:
        return "ніч"
    if hour < 12:
        return "ранок"
    if hour < 18:
        return "день"
    return "вечір"


def _date_range(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current = current.fromordinal(current.toordinal() + 1)
