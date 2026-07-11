from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, tzinfo
from decimal import Decimal
from io import BytesIO
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from sqlalchemy.ext.asyncio import AsyncSession

from mental_state_bot.db import repositories as repo
from mental_state_bot.db.models import (
    Day,
    EmbeddingRecord,
    Entry,
    Media,
    MissedPrompt,
    Summary,
    User,
    UserSettings,
)
from mental_state_bot.emotions import (
    CANONICAL_AFFECTIVE_STATES,
    CANONICAL_EMOTIONS,
    EMOTION_COLORS,
    EMOTION_INTENSITY_VALUES,
)
from mental_state_bot.services.journal_day import current_journal_date
from mental_state_bot.time_utils import parse_hhmm, utc_now, zoneinfo


@dataclass(frozen=True)
class PhotoMoment:
    media: Media
    entry: Entry


@dataclass(frozen=True)
class MetricGraphReport:
    total: int
    graphable: int
    skipped_reasons: Counter[str]


@dataclass(frozen=True)
class EmotionSignalPoint:
    label: str
    intensity: float
    confidence: float
    intensity_level: str = "unclear"


@dataclass(frozen=True)
class EmotionTimelineRow:
    signals: list[EmotionSignalPoint]
    observed: bool = False


@dataclass(frozen=True)
class AffectSpectrumPoint:
    tone: float
    color: tuple[int, int, int]
    confidence: float


async def _current_journal_date(session: AsyncSession, *, user: User) -> date:
    user_settings = await repo.get_user_settings(session, user.id)
    return await current_journal_date(session, user=user, user_settings=user_settings)


async def format_today_view(session: AsyncSession, *, user: User, limit: int = 18) -> str:
    target_date = await _current_journal_date(session, user=user)
    day = await repo.get_day_by_date(session, user_id=user.id, local_date_value=target_date)
    if day is None:
        return "За сьогодні ще немає записів."
    return await format_day_view(session, user=user, day=day, limit=limit, title="Сьогодні")


async def format_day_view(
    session: AsyncSession, *, user: User, day: Day, limit: int = 18, title: str | None = None
) -> str:
    entries = list(await repo.list_day_entries(session, day_id=day.id))
    day_title = title or f"День {day.local_date.isoformat()}"
    if not entries:
        return f"{day_title}: ще немає записів."

    visible_entries = entries[-limit:]
    analyses = await repo.list_analyses_for_targets(
        session,
        target_type="entry",
        target_ids=[entry.id for entry in visible_entries],
    )
    latest_features = _latest_feature_results_by_entry(analyses)
    quality_by_entry: dict[str, str] = {}
    labels_by_entry: dict[str, list[str]] = defaultdict(list)
    for entry_id, result in latest_features.items():
        quality_by_entry[entry_id] = result.get("data_quality") or "unknown"
        labels_by_entry[entry_id].extend(result.get("activity_labels") or [])
        labels_by_entry[entry_id].extend(result.get("state_labels") or [])

    lines = [
        f"{day_title}: {_entry_count_text(len(entries))}.",
        "",
    ]
    if len(entries) > limit:
        lines.append(f"Останні {limit}:")

    for entry in visible_entries:
        lines.append(_format_entry_line(entry, quality_by_entry, labels_by_entry, timezone=user.timezone))

    if day.ended_at:
        ended_at = day.ended_at.astimezone(zoneinfo(user.timezone))
        lines.extend(["", f"День закрито: {ended_at.strftime('%H:%M')}"])
    return "\n".join(lines)


async def format_raw_entries_view(session: AsyncSession, *, user: User, limit: int = 30) -> str:
    target_date = await _current_journal_date(session, user=user)
    day = await repo.get_day_by_date(session, user_id=user.id, local_date_value=target_date)
    if day is None:
        return "За сьогодні ще немає сирих записів."
    return await format_raw_entries_for_day(session, user=user, day=day, limit=limit, title="сьогодні")


async def format_raw_entries_for_day(
    session: AsyncSession, *, user: User, day: Day, limit: int = 30, title: str | None = None
) -> str:
    entries = list(await repo.list_day_entries(session, day_id=day.id))
    day_title = title or day.local_date.isoformat()
    if not entries:
        return f"За {day_title} ще немає сирих записів."

    visible_entries = entries[-limit:]
    lines = [f"Сирі записи за {day_title}: {_entry_count_text(len(entries))}", ""]
    if len(entries) > limit:
        lines.append(f"Показую останні {limit}.")
    for entry in visible_entries:
        time_text = _entry_time_text(entry, user.timezone)
        lines.append(f"{time_text} [{_source_label(entry.source)}] {entry.raw_text or '[без тексту]'}")
    return "\n".join(lines)


async def format_metrics_view(session: AsyncSession, *, user: User) -> str:
    target_date = await _current_journal_date(session, user=user)
    day = await repo.get_day_by_date(session, user_id=user.id, local_date_value=target_date)
    if day is None:
        return "За сьогодні ще немає метрик."
    return await format_metrics_for_day(session, user=user, day=day, title="сьогодні")


async def format_metrics_for_day(
    session: AsyncSession, *, user: User, day: Day, title: str | None = None
) -> str:
    entries = _metric_entries(list(await repo.list_day_entries(session, day_id=day.id)))
    day_title = title or day.local_date.isoformat()
    if not entries:
        return f"За {day_title} ще немає метрик."

    analyses = await repo.list_analyses_for_targets(
        session,
        target_type="entry",
        target_ids=[entry.id for entry in entries],
    )
    extraction_by_entry = _latest_feature_results_by_entry(analyses)

    activity_counts: dict[str, int] = defaultdict(int)
    state_counts: dict[str, int] = defaultdict(int)
    emotion_counts: dict[str, int] = defaultdict(int)
    affective_state_counts: dict[str, int] = defaultdict(int)
    emotion_points: list[list[str]] = []
    structured_emotion_points = 0
    fallback_emotion_points = 0
    quality_counts: dict[str, int] = defaultdict(int)
    mood_points: list[int | None] = []
    energy_points: list[int | None] = []
    mood_report_items: list[dict[str, Any]] = []
    energy_report_items: list[dict[str, Any]] = []
    meaningful_pleasant: list[str] = []
    unnormalized_label_count = 0

    for entry in entries:
        result = extraction_by_entry.get(str(entry.id), {})
        for label in result.get("activity_labels") or []:
            normalized_label = _metrics_label_text(label)
            if normalized_label is None:
                unnormalized_label_count += 1
                continue
            activity_counts[normalized_label] += 1
        for label in result.get("state_labels") or []:
            normalized_label = _metrics_label_text(label)
            if normalized_label is None:
                unnormalized_label_count += 1
                continue
            state_counts[normalized_label] += 1
        for state in _entry_affective_states(result):
            affective_state_counts[state] += 1
        structured_signals = _structured_emotion_signals(result)
        emotions = _entry_emotions(result)
        emotion_points.append(emotions)
        if structured_signals:
            structured_emotion_points += 1
        elif emotions:
            fallback_emotion_points += 1
        for emotion in emotions:
            emotion_counts[emotion] += 1
        quality_counts[_data_quality_label(result.get("data_quality"))] += 1
        mood_score = _graphable_feature_score(result, "mood")
        energy_score = _graphable_feature_score(result, "energy")
        mood_points.append(mood_score)
        energy_points.append(energy_score)
        mood_report_items.append(_metric_graph_item(result, "mood", mood_score))
        energy_report_items.append(_metric_graph_item(result, "energy", energy_score))
        meaningful_pleasant.extend(_meaningful_pleasant_moments(result.get("pleasant_moments") or []))
    meaningful_pleasant = _dedupe_texts(meaningful_pleasant)
    mood_report = _metric_graph_report(mood_report_items)
    energy_report = _metric_graph_report(energy_report_items)

    metrics_notes = []
    if unnormalized_label_count:
        metrics_notes.append(
            f"Ненормалізованих міток, прихованих із топів: {unnormalized_label_count}. "
            "Для очищення старих даних можна запустити features-backfill --force."
        )

    lines = [
        f"Метрики за {day_title}: {_entry_count_text(len(entries))}.",
        "",
        "Коротко:",
        *_metrics_overview_lines(
            entries=entries,
            mood_points=mood_points,
            energy_points=energy_points,
            mood_report=mood_report,
            energy_report=energy_report,
            quality_counts=quality_counts,
            timezone=user.timezone,
        ),
        "",
        "Динаміка:",
        *_metric_dimension_lines("Настрій", mood_points, report=mood_report, entries=entries, timezone=user.timezone),
        *_metric_dimension_lines("Енергія", energy_points, report=energy_report, entries=entries, timezone=user.timezone),
        "",
        "Матеріал для історії дня:",
        *_story_material_lines(
            entries=entries,
            extraction_by_entry=extraction_by_entry,
            mood_points=mood_points,
            energy_points=energy_points,
        ),
        "",
        "Емоційна карта:",
        *_emotion_overview_lines(
            emotion_points,
            entries=entries,
            timezone=user.timezone,
            structured_points=structured_emotion_points,
            fallback_points=fallback_emotion_points,
        ),
        _format_counts(emotion_counts, empty="немає явних емоцій", limit=8),
        "Ширші афективні стани:",
        _format_counts(affective_state_counts, empty="немає явних афективних станів", limit=8),
        "",
        f"Значущі приємні/живі моменти: {len(meaningful_pleasant)}",
        _format_bullets(meaningful_pleasant[:5], empty="немає явних значущих моментів"),
        "",
        "Якість даних:",
        _format_counts(quality_counts, limit=6),
        "",
        "Найчастіші стани:",
        _format_counts(state_counts, empty="немає явних станів", limit=6),
        "",
        "Найчастіші активності:",
        _format_counts(activity_counts, empty="немає явних активностей", limit=6),
    ]
    if metrics_notes:
        lines.extend(["", "Примітки:", *metrics_notes])
    return "\n".join(lines)


async def format_affective_vocabulary_audit(
    session: AsyncSession,
    *,
    user: User,
    limit: int = 500,
) -> str:
    entries = list(await repo.list_user_entries(session, user_id=user.id, limit=limit, descending=True))
    if not entries:
        return "Поки немає записів для аудиту емоцій."
    analyses = await repo.list_analyses_for_targets(
        session,
        target_type="entry",
        target_ids=[entry.id for entry in entries],
    )
    features_by_entry = _latest_feature_results_by_entry(analyses)
    emotion_counts: Counter[str] = Counter()
    affective_counts: Counter[str] = Counter()
    legacy_affective: Counter[str] = Counter()
    rejected_labels: Counter[str] = Counter()
    analyzed = 0

    for result in features_by_entry.values():
        analyzed += 1
        emotion_counts.update(_entry_emotions(result))
        affective_counts.update(_entry_affective_states(result))
        for value in result.get("emotion_labels") or []:
            label = _metrics_label_text(value)
            if not label or label in CANONICAL_EMOTIONS:
                continue
            if label in CANONICAL_AFFECTIVE_STATES:
                legacy_affective[label] += 1
            else:
                rejected_labels[label] += 1
        for item in result.get("emotions") or []:
            if not isinstance(item, dict):
                continue
            label = _metrics_label_text(item.get("label"))
            if label and label not in CANONICAL_EMOTIONS and label not in CANONICAL_AFFECTIVE_STATES:
                rejected_labels[label] += 1

    lines = [
        f"Аудит емоцій і станів: останні {len(entries)} записів.",
        f"AI-аналіз є для {analyzed}/{len(entries)}.",
        "",
        "Строгі емоції:",
        _format_counts(emotion_counts, empty="немає підтверджених емоцій", limit=10),
        "",
        "Ширші афективні стани:",
        _format_counts(affective_counts, empty="немає підтверджених афективних станів", limit=10),
        "",
        "Старі мітки, які після переаналізу мають перейти в афективні стани:",
        _format_counts(legacy_affective, empty="немає", limit=10),
        "",
        "Мітки поза контрольованими словниками:",
        _format_counts(rejected_labels, empty="немає", limit=10),
    ]
    if legacy_affective or rejected_labels:
        lines.extend(
            [
                "",
                "Це не означає, що переживання неправильне. Це лише сигнал, що старий аналіз варто "
                "переаналізувати, аби не змішувати його з емоційним графіком.",
            ]
        )
    return "\n".join(lines)


async def build_metrics_chart_png(session: AsyncSession, *, user: User) -> bytes | None:
    target_date = await _current_journal_date(session, user=user)
    day = await repo.get_day_by_date(session, user_id=user.id, local_date_value=target_date)
    if day is None:
        return None
    return await build_metrics_chart_png_for_day(session, user=user, day=day)


async def build_metrics_chart_png_for_day(session: AsyncSession, *, user: User, day: Day) -> bytes | None:
    entries = _metric_entries(list(await repo.list_day_entries(session, day_id=day.id)))
    if not entries:
        return None
    analyses = await repo.list_analyses_for_targets(
        session,
        target_type="entry",
        target_ids=[entry.id for entry in entries],
    )
    extraction_by_entry = _latest_feature_results_by_entry(analyses)
    mood_points = [
        _graphable_feature_score(extraction_by_entry.get(str(entry.id)) or {}, "mood") for entry in entries
    ]
    energy_points = [
        _graphable_feature_score(extraction_by_entry.get(str(entry.id)) or {}, "energy") for entry in entries
    ]
    if not any(point is not None for point in [*mood_points, *energy_points]):
        return None
    labels = [_entry_time_text(entry, user.timezone) for entry in entries]
    return _line_chart_png(
        {"mood": mood_points, "energy": energy_points},
        labels=labels,
        x_values=_entry_chart_x_values(entries, user.timezone),
        title=f"Динаміка дня: {day.local_date.isoformat()}",
        subtitle=None,
    )


async def build_emotion_timeline_png(session: AsyncSession, *, user: User) -> bytes | None:
    target_date = await _current_journal_date(session, user=user)
    day = await repo.get_day_by_date(session, user_id=user.id, local_date_value=target_date)
    if day is None:
        return None
    return await build_emotion_timeline_png_for_day(session, user=user, day=day)


async def build_emotion_timeline_png_for_day(session: AsyncSession, *, user: User, day: Day) -> bytes | None:
    entries = _metric_entries(list(await repo.list_day_entries(session, day_id=day.id)))
    if not entries:
        return None
    analyses = await repo.list_analyses_for_targets(
        session,
        target_type="entry",
        target_ids=[entry.id for entry in entries],
    )
    extraction_by_entry = _latest_feature_results_by_entry(analyses)
    emotion_rows = [_emotion_timeline_row(extraction_by_entry.get(str(entry.id)) or {}) for entry in entries]
    if not any(bool(item.signals) for item in emotion_rows):
        return None
    labels = [_entry_time_text(entry, user.timezone) for entry in entries]
    return _emotion_timeline_png(
        emotion_rows,
        labels=labels,
        x_values=_entry_chart_x_values(entries, user.timezone),
        title=f"Емоційна карта дня: {day.local_date.isoformat()}",
    )


async def build_affect_spectrum_png(session: AsyncSession, *, user: User) -> bytes | None:
    target_date = await _current_journal_date(session, user=user)
    day = await repo.get_day_by_date(session, user_id=user.id, local_date_value=target_date)
    if day is None:
        return None
    return await build_affect_spectrum_png_for_day(session, user=user, day=day)


async def build_affect_spectrum_png_for_day(session: AsyncSession, *, user: User, day: Day) -> bytes | None:
    entries = _metric_entries(list(await repo.list_day_entries(session, day_id=day.id)))
    if not entries:
        return None
    analyses = await repo.list_analyses_for_targets(
        session,
        target_type="entry",
        target_ids=[entry.id for entry in entries],
    )
    extraction_by_entry = _latest_feature_results_by_entry(analyses)
    points = [_affect_spectrum_point(extraction_by_entry.get(str(entry.id)) or {}) for entry in entries]
    if not any(point is not None for point in points):
        return None
    return _affect_spectrum_png(
        points,
        labels=[_entry_time_text(entry, user.timezone) for entry in entries],
        x_values=_entry_chart_x_values(entries, user.timezone),
        title=f"Спектр стану дня: {day.local_date.isoformat()}",
    )


async def get_today_photo_moments(session: AsyncSession, *, user: User) -> list[PhotoMoment]:
    target_date = await _current_journal_date(session, user=user)
    day = await repo.get_day_by_date(session, user_id=user.id, local_date_value=target_date)
    if day is None:
        return []
    return await get_photo_moments_for_day(session, day=day)


async def get_photo_moments_for_day(session: AsyncSession, *, day: Day) -> list[PhotoMoment]:
    rows = await repo.list_day_media_with_entries(session, day_id=day.id, media_type="photo")
    return [PhotoMoment(media=media, entry=entry) for media, entry in rows]


def format_photo_moments_view(
    moments: list[PhotoMoment],
    *,
    limit: int = 12,
    timezone: str | None = None,
    title: str = "Фото дня",
) -> str:
    if not moments:
        if title == "Фото дня":
            return "За сьогодні ще немає фото."
        return f"{title}: немає фото."

    visible = moments[-limit:]
    lines = [f"{title}: {len(moments)}", ""]
    if len(moments) > limit:
        lines.append(f"Показую останні {limit}.")
        lines.append("")
    lines.append("Окремо надсилаю фото нижче, щоб вони не змішувалися з текстом і метриками.")
    lines.append("")
    for index, moment in enumerate(visible, start=1):
        timestamp = moment.entry.local_timestamp or moment.media.created_at or moment.entry.created_at
        time_text = _time_text(timestamp, timezone)
        caption = moment.entry.raw_text
        if caption == "[photo]":
            caption = None
        caption_text = f" — {_truncate(caption, 80)}" if caption else ""
        lines.append(f"{index}. {time_text}{caption_text}")
    return "\n".join(lines)


async def format_gaps_view(session: AsyncSession, *, user: User) -> str:
    today = await _current_journal_date(session, user=user)
    day = await repo.get_day_by_date(session, user_id=user.id, local_date_value=today)
    return await format_gaps_for_day(session, user=user, day=day, target_date=today, title="Прогалини сьогодні")


async def format_gaps_for_day(
    session: AsyncSession,
    *,
    user: User,
    day: Day | None,
    target_date: date | None = None,
    title: str | None = None,
) -> str:
    report_date = target_date or (day.local_date if day else await _current_journal_date(session, user=user))
    user_settings = await repo.get_user_settings(session, user.id)
    window_start, window_end = _active_window(report_date, user.timezone, user_settings)
    now_local = utc_now().astimezone(zoneinfo(user.timezone))
    current_date = await current_journal_date(session, user=user, user_settings=user_settings)
    coverage_end = min(now_local, window_end) if report_date == current_date else window_end

    entries = list(await repo.list_day_entries(session, day_id=day.id)) if day else []
    missed_prompts = list(
        await repo.list_missed_prompts_between(
            session,
            user_id=user.id,
            start=window_start,
            end=coverage_end,
        )
    )
    return format_gap_report(
        entries=entries,
        missed_prompts=missed_prompts,
        window_start=window_start,
        window_end=coverage_end,
        title=title or f"Прогалини за {report_date.isoformat()}",
    )


def format_gap_report(
    *,
    entries: list[Entry],
    missed_prompts: list[MissedPrompt],
    window_start: datetime,
    window_end: datetime,
    title: str = "Прогалини сьогодні",
) -> str:
    if window_end <= window_start:
        return "\n".join(
            [
                title,
                f"Активне вікно починається о {window_start.strftime('%H:%M')}.",
                "Поки рано оцінювати покриття дня.",
            ]
        )

    report_tz = window_start.tzinfo
    entry_times = sorted(
        timestamp for entry in entries if (timestamp := _entry_timestamp(entry, report_tz)) is not None
    )
    entry_times = [timestamp for timestamp in entry_times if window_start <= timestamp <= window_end]
    gaps = _gap_segments(entry_times, window_start=window_start, window_end=window_end)
    notable_gaps = [gap for gap in gaps if gap[2] >= timedelta(minutes=60)]
    largest_gap = max((gap[2] for gap in gaps), default=timedelta())
    coverage_note = _coverage_note(len(entry_times), len(missed_prompts), largest_gap)

    lines = [
        title,
        f"Активне вікно у звіті: {window_start.strftime('%H:%M')}-{window_end.strftime('%H:%M')}",
        f"Записів у вікні: {len(entry_times)}",
        f"Пропущених/нагаданих зрізів: {len(missed_prompts)}",
        f"Покриття: {coverage_note}",
        "",
    ]
    if entry_times:
        lines.extend(
            [
                f"Перший запис: {entry_times[0].strftime('%H:%M')}",
                f"Останній запис: {entry_times[-1].strftime('%H:%M')}",
                "",
            ]
        )
    if notable_gaps:
        lines.append("Помітні паузи:")
        for start, end, duration in notable_gaps[:6]:
            lines.append(f"- {start.strftime('%H:%M')}-{end.strftime('%H:%M')}: {_format_duration(duration)}")
    else:
        lines.append("Помітних пауз понад 1 годину не видно.")

    if missed_prompts:
        lines.extend(["", "Пропущені зрізи:"])
        for missed in missed_prompts[:6]:
            reason_text = getattr(missed, "reason_text", None)
            reason = f" — {reason_text}" if reason_text else ""
            lines.append(
                f"- {missed.missed_at.astimezone(window_start.tzinfo).strftime('%H:%M')}: "
                f"{_status_label(missed.status)}{reason}"
            )
    return "\n".join(lines)


async def format_latest_summary_section(
    session: AsyncSession, *, user: User, section: str
) -> str:
    summary = await repo.get_latest_summary(session, user_id=user.id, period_type="daily")
    if summary is None:
        return "Ще немає згенерованого денного підсумку. Можна натиснути “Підсумок дня”."
    return format_summary_section(summary, section)


async def format_day_summary_section(session: AsyncSession, *, user: User, day: Day, section: str) -> str:
    summary = await repo.get_day_summary(session, user_id=user.id, day_id=day.id, period_type="daily")
    if summary is None:
        return f"Підсумку за {day.local_date.isoformat()} ще немає."
    return format_summary_section(summary, section)


def format_summary_section(summary: Summary, section: str) -> str:
    details = summary.details or {}
    stale_notice = _stale_summary_notice(details)
    if section == "story":
        return "\n\n".join(
            part
            for part in [
                stale_notice,
                "Історія дня",
                details.get("story") or summary.short_text,
                _list_block("Що реально відбувалося", details.get("actual_activities") or []),
                _list_block("Зміни стану", details.get("state_changes") or []),
            ]
            if part
        )
    if section == "metrics":
        return "\n".join(
            [line for line in [
                stale_notice,
                "Спостереження з підсумку",
                "",
                f"Найважчий відрізок: {details.get('hardest_interval') or 'неясно'}",
                f"Найстабільніший/найкращий відрізок: {details.get('best_or_stablest_interval') or 'неясно'}",
                "",
                _list_block("Приємні/живі моменти", details.get("pleasant_moments") or []),
                "",
                _list_block("Обережні спостереження", details.get("cautious_observations") or []),
                "",
                f"Якість даних: {_data_quality_label(details.get('data_quality'))}",
            ] if line is not None]
        )
    if section == "timeline":
        return "\n".join(
            [line for line in [
                stale_notice,
                "Таймлайн з підсумку",
                "",
                details.get("story") or summary.short_text,
                "",
                _list_block("Прогалини", details.get("data_gaps") or []),
            ] if line is not None]
        )
    if stale_notice:
        return f"{stale_notice}\n\n{summary.short_text}"
    return summary.short_text


def _stale_summary_notice(details: dict[str, Any]) -> str | None:
    stale = details.get("stale")
    if not isinstance(stale, dict):
        return None
    reason = {
        "entry_deleted": "запис було видалено",
        "entry_corrected": "запис було виправлено",
    }.get(str(stale.get("reason") or ""), "дані дня змінилися")
    return f"Підсумок може бути застарілим: {reason}. Можна натиснути “Оновити підсумок”."


def format_period_summary(summary: Summary) -> str:
    details = summary.details or {}
    title = "Тижневий підсумок" if summary.period_type == "weekly" else "Місячний підсумок"
    start_label = details.get("journal_start_date") or _display_date(summary.period_start)
    end_label = details.get("journal_end_date") or _display_date(summary.period_end)
    lines = [
        title,
        f"{start_label} - {end_label}",
        "",
        summary.short_text,
        "",
        details.get("period_story") or "",
        "",
        _list_block("Повторювані патерни", details.get("repeated_patterns") or []),
        "",
        _list_block("Зміни проти попереднього періоду", details.get("changes_vs_previous_period") or []),
        "",
        _list_block("Активність + стан", details.get("activity_state_patterns") or []),
        "",
        _list_block("Що допомагало", details.get("what_helped") or []),
        "",
        _list_block("Що погіршувало", details.get("what_worsened") or []),
        "",
        _list_block("Помітні дні", details.get("notable_days") or []),
        "",
        _list_block("Прогалини", details.get("data_gaps") or []),
        "",
        _list_block("Обережні спостереження", details.get("cautious_observations") or []),
        "",
        f"Якість даних: {_data_quality_label(details.get('data_quality'))}",
    ]
    return "\n".join(line for line in lines if line is not None).strip()


def format_period_emotions_view(summary: Summary) -> str:
    analysis = _stored_period_analysis(summary)
    emotions = analysis.get("emotions") or {}
    coverage = analysis.get("coverage") or {}
    frequency = emotions.get("frequency") or []
    intensity = emotions.get("mean_intensity") or []
    co_occurrence = emotions.get("co_occurrence") or []
    lines = [
        f"Емоції: {_period_title(summary).lower()}",
        _period_date_label(summary),
        "",
        (
            "Явні емоційні сигнали: "
            f"{coverage.get('emotion_observed_entries', 0)} записів із {coverage.get('entry_count', 0)}."
        ),
        "",
        _period_named_observations("Найчастіше", frequency, noun="спостережень"),
        "",
        _period_intensity_observations(intensity),
        "",
        _period_co_occurrence_observations(co_occurrence),
    ]
    return "\n".join(line for line in lines if line is not None).strip()


def format_period_patterns_view(summary: Summary) -> str:
    analysis = _stored_period_analysis(summary)
    associations = analysis.get("repeated_associations") or []
    comparison = (summary.details or {}).get("previous_period_comparison") or {}
    ai_patterns = (summary.details or {}).get("repeated_patterns") or []
    lines = [
        f"Патерни: {_period_title(summary).lower()}",
        _period_date_label(summary),
        "",
        "Повторювані збіги в тих самих записах:",
        _format_period_associations(associations),
        "",
        "Порівняння з попереднім періодом:",
        _format_period_comparison(comparison),
        "",
        _list_block("AI-спостереження", ai_patterns),
    ]
    return "\n".join(line for line in lines if line is not None).strip()


def format_period_turning_points_view(summary: Summary) -> str:
    analysis = _stored_period_analysis(summary)
    points = analysis.get("turning_points") or []
    lines = [
        f"Повороти: {_period_title(summary).lower()}",
        _period_date_label(summary),
        "",
    ]
    if not points:
        lines.append("У денних підсумках за цей період ще немає достатньо описаних змін стану.")
        return "\n".join(lines)
    lines.append("Зміни, які AI виділив у денних історіях:")
    for point in points:
        date_text = str(point.get("date") or "дата невідома")
        changes = point.get("changes") or []
        lines.append(f"\n{date_text}:")
        lines.extend(f"- {change}" for change in changes[:5])
    lines.extend(
        [
            "",
            "Це навігація по вже збережених історіях дня, а не окрема діагностика чи новий висновок.",
        ]
    )
    return "\n".join(lines)


async def format_period_timeline_view(session: AsyncSession, *, user: User, summary: Summary) -> str:
    start_date, end_date = _summary_local_date_range(summary, user.timezone)
    days = await repo.list_days_between(session, user_id=user.id, start_date=start_date, end_date=end_date)
    days_by_id = _days_by_id(days)
    entries = await repo.list_entries_between(
        session,
        user_id=user.id,
        start=summary.period_start,
        end=summary.period_end,
    )
    daily_summaries = await repo.list_summaries_between(
        session,
        user_id=user.id,
        period_type="daily",
        start=summary.period_start,
        end=summary.period_end,
    )
    entry_counts = _entry_counts_by_journal_date(entries, days_by_id=days_by_id, timezone=user.timezone)
    summaries_by_date = _summaries_by_journal_date(daily_summaries, days_by_id=days_by_id, timezone=user.timezone)
    day_ids_by_date = {day.local_date: day.id for day in days}
    lines = [
        _period_title(summary),
        f"{start_date.isoformat()} - {end_date.isoformat()}",
        "",
        "Таймлайн періоду",
    ]
    for current in _date_range(start_date, end_date):
        count = entry_counts.get(current, 0)
        daily = summaries_by_date.get(current)
        marker = "•" if count else "·"
        summary_text = _truncate(daily.short_text, 120) if daily else "денного підсумку немає"
        if count == 0:
            summary_text = "немає записів"
        day_hint = f" /day {current.isoformat()}" if current in day_ids_by_date else ""
        lines.append(f"{marker} {current.isoformat()}: {_entry_count_text(count)} — {summary_text}{day_hint}")
    return "\n".join(lines)


async def format_period_metrics_view(session: AsyncSession, *, user: User, summary: Summary) -> str:
    report = await _period_metrics_report(session, user=user, summary=summary)
    start_date, end_date = _summary_local_date_range(summary, user.timezone)
    lines = [
        f"Метрики: {_period_title(summary).lower()}",
        f"{start_date.isoformat()} - {end_date.isoformat()}",
        f"Записів: {report['entry_count']}",
        f"Днів із записами: {report['active_days']}/{report['total_days']}",
        f"Значущі приємні/живі моменти: {report['pleasant_count']}",
        "",
        "Графіки моментів по днях:",
        f"- Настрій: {report['mood_graphable_days']} днів із надійними точками; точок {report['mood_point_count']}.",
        f"- Енергія: {report['energy_graphable_days']} днів із надійними точками; точок {report['energy_point_count']}.",
        f"- Емоції: {report['emotion_observed_days']} днів із явними емоційними сигналами.",
        f"- Днів лише з матеріалом для історії, без mood/energy точок: {report['story_only_days']}.",
        "",
        "Настрій у надійних точках:",
        _sparkline(report["daily_mood"]),
        "Енергія у надійних точках:",
        _sparkline(report["daily_energy"]),
        "Це середні значення надійних моментних точок дня, не автоматична оцінка всього дня.",
        "",
        "Якість даних:",
        _format_counts(report["quality_counts"]),
        "",
        "Найчастіші стани:",
        _format_counts(report["state_counts"], empty="немає явних станів"),
        "",
        "Найчастіші емоції:",
        _format_counts(report["emotion_counts"], empty="немає явних емоцій"),
        "",
        "Найчастіші активності:",
        _format_counts(report["activity_counts"], empty="немає явних активностей"),
    ]
    stored_analysis = _stored_period_analysis(summary)
    rhythm = stored_analysis.get("rhythm") or []
    if rhythm:
        lines.extend(["", "Ритм спостережень:", _format_period_rhythm(rhythm)])
    if report["unnormalized_label_count"]:
        lines.extend(
            [
                "",
                "Примітки:",
                f"Ненормалізованих міток, прихованих із топів: {report['unnormalized_label_count']}.",
            ]
        )
    return "\n".join(lines)


async def build_period_metrics_chart_png(
    session: AsyncSession, *, user: User, summary: Summary
) -> bytes | None:
    report = await _period_metrics_report(session, user=user, summary=summary)
    daily_mood = report["daily_mood"]
    daily_energy = report["daily_energy"]
    if not any(point is not None for point in [*daily_mood, *daily_energy]):
        return None
    start_date, end_date = _summary_local_date_range(summary, user.timezone)
    labels = [current.strftime("%m-%d") for current in _date_range(start_date, end_date)]
    return _line_chart_png(
        {"mood": daily_mood, "energy": daily_energy},
        labels=labels,
        title=f"Динаміка: {_period_title(summary).lower()}",
        subtitle="Кожна точка — середнє надійних моментних оцінок дня, не оцінка всього дня.",
    )


async def build_period_emotion_chart_png(
    session: AsyncSession,
    *,
    user: User,
    summary: Summary,
) -> bytes | None:
    report = await _period_metrics_report(session, user=user, summary=summary)
    rows: list[EmotionTimelineRow] = report["daily_emotion_rows"]
    if not any(row.signals for row in rows):
        return None
    start_date, end_date = _summary_local_date_range(summary, user.timezone)
    labels = [current.strftime("%m-%d") for current in _date_range(start_date, end_date)]
    return _emotion_timeline_png(
        rows,
        labels=labels,
        x_values=[float(index) for index in range(len(rows))],
        title=f"Емоційна динаміка: {_period_title(summary).lower()}",
    )


async def format_period_days_view(session: AsyncSession, *, user: User, summary: Summary) -> str:
    start_date, end_date = _summary_local_date_range(summary, user.timezone)
    days = await repo.list_days_between(session, user_id=user.id, start_date=start_date, end_date=end_date)
    days_by_id = _days_by_id(days)
    daily_summaries = await repo.list_summaries_between(
        session,
        user_id=user.id,
        period_type="daily",
        start=summary.period_start,
        end=summary.period_end,
    )
    entries = await repo.list_entries_between(
        session,
        user_id=user.id,
        start=summary.period_start,
        end=summary.period_end,
    )
    entry_counts = _entry_counts_by_journal_date(entries, days_by_id=days_by_id, timezone=user.timezone)
    lines = [
        f"Дні періоду: {_period_title(summary).lower()}",
        f"{start_date.isoformat()} - {end_date.isoformat()}",
        "",
    ]
    if not daily_summaries and not entry_counts:
        return "\n".join([*lines, "За цей період немає даних."])
    summaries_by_date = _summaries_by_journal_date(daily_summaries, days_by_id=days_by_id, timezone=user.timezone)
    for current in _date_range(start_date, end_date):
        count = entry_counts.get(current, 0)
        daily = summaries_by_date.get(current)
        if not count and daily is None:
            continue
        short_text = _truncate(daily.short_text, 180) if daily else "підсумку немає"
        lines.append(f"{current.isoformat()}: {_entry_count_text(count)}")
        lines.append(short_text)
        lines.append(f"/day {current.isoformat()}")
        lines.append("")
    return "\n".join(lines).strip()


async def format_cost_report(session: AsyncSession, *, user: User, days: int = 7) -> str:
    since = utc_now() - timedelta(days=days)
    totals = await repo.model_run_cost_totals(session, user_id=user.id, since=since)
    runs = await repo.list_model_runs_since(session, user_id=user.id, since=since)
    by_task: dict[str, dict[str, Any]] = defaultdict(lambda: {"runs": 0, "cost": Decimal("0"), "tokens": 0})
    for run in runs:
        bucket = by_task[run.task_name]
        bucket["runs"] += 1
        bucket["cost"] += run.estimated_cost_usd or Decimal("0")
        bucket["tokens"] += run.total_tokens or 0

    lines = [
        f"Витрати за {days} днів:",
        f"Викликів моделі: {totals['runs']}",
        f"Оцінка вартості: ${Decimal(totals['estimated_cost_usd']):.6f}",
        (
            f"Токени: {totals['total_tokens']} загалом, {totals['prompt_tokens']} вхідні, "
            f"{totals['completion_tokens']} вихідні, {totals['reasoning_tokens']} міркування"
        ),
    ]
    if by_task:
        lines.append("")
        lines.append("По задачах:")
        for task_name, bucket in sorted(by_task.items()):
            lines.append(
                f"- {_task_label(task_name)}: {bucket['runs']} викликів, {bucket['tokens']} токенів, ${bucket['cost']:.6f}"
            )
    else:
        lines.append("")
        lines.append("Поки немає записів про виклики моделей.")
    return "\n".join(lines)


def format_similar_entries(
    records: list[EmbeddingRecord],
    *,
    query: str | None = None,
    entries: list[Entry] | None = None,
    analyses=None,
    timezone: str | None = None,
    limit: int = 5,
) -> str:
    if not records:
        return "Схожих записів поки не знайшов. Можливо, ще немає embeddings або даних замало."

    entries_by_id = {str(entry.id): entry for entry in entries or []}
    features_by_entry = _latest_feature_results_by_entry(analyses or [])
    selected_records = _diverse_similar_records(
        records,
        entries_by_id=entries_by_id,
        timezone=timezone,
        limit=limit,
    )
    selected_entry_ids = [str(record.target_id) for record in selected_records]
    activity_counts, state_counts = _similar_label_counts(selected_entry_ids, features_by_entry)
    day_counts = _similar_day_counts(selected_records, entries_by_id=entries_by_id, timezone=timezone)

    title = f"Пам’ять за запитом: «{_truncate(query, 80)}»" if query else "Схожі моменти з пам’яті"
    lines = [
        title,
        f"Знайшов {len(records)} схожих моментів. Показую {len(selected_records)} найкорисніших для згадування.",
        "",
    ]
    pattern_lines = []
    if state_counts:
        pattern_lines.append("Стани: " + _inline_counts(state_counts, limit=4))
    if activity_counts:
        pattern_lines.append("Активності: " + _inline_counts(activity_counts, limit=4))
    if day_counts:
        pattern_lines.append("Дати: " + _inline_counts(day_counts, limit=4))
    if pattern_lines:
        lines.extend(["Що може повторюватися:", *[f"- {line}" for line in pattern_lines], ""])

    lines.append("Приклади:")
    for index, record in enumerate(selected_records, start=1):
        entry = entries_by_id.get(str(record.target_id))
        timestamp = _similar_timestamp(record, entry, timezone)
        date_text = timestamp.strftime("%Y-%m-%d") if timestamp else "дата невідома"
        time_text = timestamp.strftime("%H:%M") if timestamp else "??:??"
        text = _similar_record_text(record, entry)
        labels = _similar_labels(features_by_entry.get(str(record.target_id), {}))
        label_text = f"\n   Мітки: {', '.join(labels[:4])}" if labels else ""
        lines.append(f"{index}. {date_text} {time_text} — {_truncate(text, 170)}{label_text}")
        if timestamp:
            lines.append(f"   Відкрити день: /day {date_text}")
    return "\n".join(lines)


def _feature_score(feature: object) -> int | None:
    if not isinstance(feature, dict):
        return None
    raw_value = feature.get("value")
    if isinstance(raw_value, int | float):
        score = int(round(raw_value))
        return score if 0 <= score <= 10 else None
    value = " ".join(str(raw_value or "unclear").strip().lower().replace("_", " ").split())
    if value.isdigit():
        score = int(value)
        return score if 0 <= score <= 10 else None
    mapping = {
        "very low": 1,
        "дуже низько": 1,
        "дуже низький": 1,
        "дуже погано": 1,
        "дуже поганий": 1,
        "немає сил": 1,
        "low": 2,
        "низько": 2,
        "низький": 2,
        "погано": 2,
        "поганий": 2,
        "мало сил": 2,
        "somewhat low": 3,
        "трохи низько": 3,
        "нижче середнього": 3,
        "mixed": 4,
        "neutral": 4,
        "змішано": 4,
        "нейтрально": 4,
        "нормально": 4,
        "нормальний": 4,
        "medium": 5,
        "moderate": 5,
        "середньо": 5,
        "помірно": 5,
        "somewhat high": 6,
        "трохи високо": 6,
        "вище середнього": 6,
        "high": 7,
        "високо": 7,
        "високий": 7,
        "добре": 7,
        "гарний": 7,
        "very high": 8,
        "дуже високо": 8,
        "дуже високий": 8,
        "дуже добре": 8,
    }
    return mapping.get(value)


def _graphable_feature_score(result: dict[str, Any], metric: str) -> int | None:
    should_graph = result.get(f"should_graph_{metric}")
    if should_graph is False:
        return None
    if should_graph is None and _new_entry_feature_schema(result):
        return None
    return _feature_score(result.get(metric))


def _new_entry_feature_schema(result: dict[str, Any]) -> bool:
    return any(
        key in result
        for key in (
            "entry_type",
            "mood_evidence",
            "energy_evidence",
            "mood_reasoning_type",
            "energy_reasoning_type",
            "should_graph_mood",
            "should_graph_energy",
        )
    )


def _metric_graph_item(result: dict[str, Any], metric: str, score: int | None) -> dict[str, Any]:
    if score is not None:
        return {"graphable": True, "reason": None}
    return {"graphable": False, "reason": _metric_skip_reason(result, metric)}


def _metric_graph_report(items: list[dict[str, Any]]) -> MetricGraphReport:
    skipped_reasons = Counter(str(item.get("reason") or "не для графіка") for item in items if not item.get("graphable"))
    return MetricGraphReport(
        total=len(items),
        graphable=sum(1 for item in items if item.get("graphable")),
        skipped_reasons=skipped_reasons,
    )


def _metric_skip_reason(result: dict[str, Any], metric: str) -> str:
    if not result:
        return "немає AI-аналізу"
    entry_type = str(result.get("entry_type") or "")
    if entry_type in {"activity_only", "photo_only", "sleep", "dream", "reply_fragment", "command_or_system"}:
        return _entry_type_skip_label(entry_type)
    feature = result.get(metric)
    if _feature_score(feature) is None:
        return "оцінка неясна"
    evidence = result.get(f"{metric}_evidence")
    if not isinstance(evidence, str) or not evidence.strip():
        return "немає evidence"
    reasoning = str(result.get(f"{metric}_reasoning_type") or "unclear")
    if reasoning not in {"direct_text", "user_manual"}:
        return _reasoning_skip_label(reasoning)
    confidence = _feature_confidence(feature)
    threshold = 0.7 if metric == "energy" else 0.65
    if confidence is not None and confidence < threshold:
        return "низька впевненість"
    should_graph = result.get(f"should_graph_{metric}")
    if should_graph is False:
        return "відхилено AI-gate"
    if should_graph is None and _new_entry_feature_schema(result):
        return "немає graph gate"
    return "не для графіка"


def _entry_type_skip_label(entry_type: str) -> str:
    labels = {
        "activity_only": "лише активність",
        "photo_only": "лише фото",
        "sleep": "сон/метадані сну",
        "dream": "сон/сновидіння",
        "reply_fragment": "фрагмент відповіді",
        "command_or_system": "службова дія",
    }
    return labels.get(entry_type, "тип запису не для графіка")


def _reasoning_skip_label(reasoning: str) -> str:
    labels = {
        "weak_text": "непряме evidence",
        "context_inferred": "виведено з контексту",
        "metadata_only": "лише метадані",
        "unclear": "reasoning неясний",
    }
    return labels.get(reasoning, "непряме evidence")


def _feature_confidence(feature: object) -> float | None:
    if not isinstance(feature, dict):
        return None
    value = feature.get("confidence")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, Decimal):
        return float(value)
    return None


def _graphable_count_text(report: MetricGraphReport) -> str:
    return f"{report.graphable} надійних точок із {report.total}"


def _format_skip_reasons(reasons: Counter[str], *, limit: int = 3) -> str:
    if not reasons:
        return "немає"
    return ", ".join(f"{reason} {count}" for reason, count in reasons.most_common(limit))


def _metrics_label_text(label: object) -> str | None:
    text = _ai_label_text(label)
    if not text or text == "невідомо":
        return None
    if any("a" <= char.lower() <= "z" for char in text):
        return None
    return _canonical_metric_label(text)


def _primary_emotion(labels: list[object]) -> str | None:
    return next(iter(_emotion_labels_from_values(labels)), None)


def _entry_emotions(result: dict[str, Any]) -> list[str]:
    structured = _structured_emotion_signals(result)
    if structured:
        return _dedupe_labels([signal.label for signal in structured])
    return _fallback_emotion_signals(result)


def _fallback_emotion_signals(result: dict[str, Any]) -> list[str]:
    return _emotion_labels_from_values(
        [
            *(result.get("emotion_labels") or []),
            *(result.get("state_labels") or []),
        ]
    )


def _entry_affective_states(result: dict[str, Any]) -> list[str]:
    states: list[str] = []
    for item in result.get("affective_states") or []:
        if not isinstance(item, dict):
            continue
        label = _metrics_label_text(item.get("label"))
        if label and label not in states:
            states.append(label)
    return states[:8]


def _entry_emotion_signals(result: dict[str, Any]) -> list[EmotionSignalPoint]:
    structured = _structured_emotion_signals(result)
    return structured or [
        EmotionSignalPoint(label=label, intensity=0.55, confidence=0.45, intensity_level="moderate")
        for label in _fallback_emotion_signals(result)[:8]
    ]


def _structured_emotion_signals(result: dict[str, Any]) -> list[EmotionSignalPoint]:
    signals: list[EmotionSignalPoint] = []
    seen: set[str] = set()
    for item in result.get("emotions") or []:
        if not isinstance(item, dict):
            continue
        label = _structured_emotion_label(item.get("label"))
        if not label or label in seen:
            continue
        if str(item.get("time_scope") or "").strip().lower() not in {"current", "recent"}:
            continue
        evidence = str(item.get("evidence") or "").strip()
        if not evidence:
            continue
        confidence = _float_between(item.get("confidence"), default=0.0)
        if confidence < 0.25:
            continue
        level = str(item.get("intensity_level") or "unclear").strip().lower()
        intensity = _emotion_intensity_value(level, item.get("intensity"))
        if intensity <= 0:
            continue
        signals.append(
            EmotionSignalPoint(
                label=label,
                intensity=intensity,
                confidence=confidence,
                intensity_level=level,
            )
        )
        seen.add(label)
    if signals:
        return sorted(signals, key=lambda signal: signal.intensity * signal.confidence, reverse=True)[:8]
    return []


def _dedupe_labels(labels: list[str]) -> list[str]:
    result: list[str] = []
    for label in labels:
        if label and label not in result:
            result.append(label)
    return result


def _structured_emotion_label(label: object) -> str | None:
    text = _metrics_label_text(label)
    return text if text in CANONICAL_EMOTIONS else None


def _emotion_intensity_value(level: str, value: object) -> float:
    if level in EMOTION_INTENSITY_VALUES:
        return EMOTION_INTENSITY_VALUES[level]
    return _float_between(value, default=0.0)


def _float_between(value: object, *, default: float) -> float:
    try:
        number = float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))


def _emotion_labels_from_values(labels: list[object]) -> list[str]:
    emotions: list[str] = []
    seen: set[str] = set()
    for label in labels:
        normalized = _emotion_label(label)
        if normalized and normalized not in seen:
            emotions.append(normalized)
            seen.add(normalized)
    return emotions


def _emotion_label(label: object) -> str | None:
    text = _metrics_label_text(label)
    if not text:
        return None
    if text in CANONICAL_EMOTIONS:
        return text
    aliases = {
        "спокій": {"спокійно"},
        "радість": {"радісно"},
        "тривога": {"тривожно"},
        "сум": {"сумно", "смуток"},
        "злість": {"гнів", "роздратування"},
        "порожнеча": {"пустота"},
    }
    for canonical, variants in aliases.items():
        if text in variants:
            return canonical
    return None


def _emotion_colors() -> dict[str, tuple[int, int, int]]:
    return dict(EMOTION_COLORS)


def _canonical_metric_label(text: str) -> str:
    normalized = " ".join(text.lower().replace("_", " ").split())
    aliases = {
        "гуляти": "прогулянка",
        "гуляння": "прогулянка",
        "ходив гуляти": "прогулянка",
        "йду додому": "дорога додому",
        "walking home": "дорога додому",
        "гарний настрій": "радість",
        "добрий настрій": "радість",
        "енергійний": "енергія",
        "енергійність": "енергія",
        "виснажено": "виснаження",
        "втомлений": "втома",
    }
    return aliases.get(normalized, normalized)


def _feature_value_label(value: str | None) -> str:
    labels = {
        "very_low": "дуже низько",
        "low": "низько",
        "somewhat_low": "трохи низько",
        "mixed": "змішано",
        "neutral": "нейтрально",
        "medium": "середньо",
        "moderate": "помірно",
        "somewhat_high": "трохи високо",
        "high": "високо",
        "very_high": "дуже високо",
        "unclear": "неясно",
        "unknown": "невідомо",
    }
    return labels.get(str(value or "unknown").lower(), _human_label(str(value or "невідомо")))


def _data_quality_label(value: str | None) -> str:
    labels = {
        "empty": "порожньо",
        "very_low": "дуже мало даних",
        "partial": "частково",
        "enough": "достатньо",
        "rich": "багато даних",
        "low": "мало даних",
        "unknown": "невідомо",
    }
    return labels.get(str(value or "unknown").lower(), _human_label(str(value or "невідомо")))


def _status_label(value: str | None) -> str:
    labels = {
        "open": "відкрито",
        "explained": "пояснено",
        "closed": "закрито",
        "closed_by_user": "закрито вручну",
        "missed": "пропущено",
        "missed_explained": "пропуск пояснено",
        "postponed": "відкладено",
        "prompted": "очікує відповіді",
        "in_progress": "у процесі",
    }
    return labels.get(str(value or "unknown").lower(), _human_label(str(value or "невідомо")))


def _source_label(value: str | None) -> str:
    labels = {
        "manual": "ручний запис",
        "snapshot_response": "відповідь на зріз",
        "user_stop": "зупинка зрізу",
        "button_as_is": "кнопка: записати як є",
        "button_stop": "кнопка: зупинитися",
        "button_later": "кнопка: пізніше",
        "missed_reason": "причина пропуску",
        "sleep_marker": "закриття дня",
        "correction": "виправлення",
        "profile_context_update": "оновлення контексту",
    }
    return labels.get(str(value or "unknown").lower(), _human_label(str(value or "невідомо")))


def _task_label(value: str | None) -> str:
    labels = {
        "generate_snapshot_question": "генерація питання",
        "generate_clarification": "уточнення",
        "extract_entry_features": "аналіз запису",
        "generate_micro_summary": "мікро-підсумок",
        "generate_daily_summary": "денний підсумок",
        "generate_weekly_summary": "тижневий підсумок",
        "generate_monthly_summary": "місячний підсумок",
        "daily_summary_semantic_context": "пам’ять для денного підсумку",
        "weekly_summary_semantic_context": "пам’ять для тижневого підсумку",
        "monthly_summary_semantic_context": "пам’ять для місячного підсумку",
        "embed_entry": "вектор пам’яті для запису",
        "similar_entries": "пошук схожих моментів",
        "transcribe_voice": "транскрипція голосового",
    }
    return labels.get(str(value or "unknown").lower(), _human_label(str(value or "невідомо")))


def _human_label(value: str) -> str:
    known = {
        "lying_down": "лежання",
        "inactive": "бездіяльність",
        "stuck": "застрягання",
        "stuck_or_scrolling": "залипання",
        "inability_to_start": "важко почати",
        "emptiness": "порожнеча",
        "avoidance": "уникання",
        "rumination": "думки по колу",
        "social_activity": "соціальна активність",
    }
    normalized = " ".join(str(value or "").strip().lower().split())
    spaced = normalized.replace("_", " ")
    return known.get(normalized) or known.get(spaced) or spaced or "невідомо"


def _latest_feature_results_by_entry(analyses) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for analysis in analyses:
        if analysis.task_name == "extract_entry_features":
            results[str(analysis.target_id)] = analysis.result or {}
    return results


def _metrics_overview_lines(
    *,
    entries: list[Entry],
    mood_points: list[int | None],
    energy_points: list[int | None],
    mood_report: MetricGraphReport,
    energy_report: MetricGraphReport,
    quality_counts: dict[str, int],
    timezone: str,
) -> list[str]:
    mood_known = [point for point in mood_points if point is not None]
    energy_known = [point for point in energy_points if point is not None]
    lines = [
        "- Графіки обережні: "
        f"настрій {_graphable_count_text(mood_report)}, енергія {_graphable_count_text(energy_report)}.",
    ]
    if len(mood_known) < max(3, len(entries) // 3):
        lines.append("- Багато записів допомагають історії дня, але не є надійними точками для графіка.")
    if mood_known:
        avg = sum(mood_known) / len(mood_known)
        lines.append(f"- Настрій у відомих точках: {_score_word(avg)} ({avg:.1f}/10).")
    if energy_known:
        avg = sum(energy_known) / len(energy_known)
        lines.append(f"- Енергія у відомих точках: {_score_word(avg)} ({avg:.1f}/10).")
    weak_quality = quality_counts.get("дуже мало даних", 0) + quality_counts.get("невідомо", 0)
    if weak_quality:
        lines.append(f"- Записів із дуже слабким/невідомим контекстом: {weak_quality}.")
    first_known = _first_known_metric_time(entries, [mood_points, energy_points], timezone)
    last_known = _last_known_metric_time(entries, [mood_points, energy_points], timezone)
    if first_known and last_known and first_known != last_known:
        lines.append(f"- Оцінювана частина дня: приблизно {first_known}-{last_known}.")
    skipped_parts = [
        f"настрій: {_format_skip_reasons(mood_report.skipped_reasons)}",
        f"енергія: {_format_skip_reasons(energy_report.skipped_reasons)}",
    ]
    lines.append("- Чому частина не в графіку: " + "; ".join(skipped_parts) + ".")
    return lines


def _metric_dimension_lines(
    title: str,
    points: list[int | None],
    *,
    report: MetricGraphReport,
    entries: list[Entry],
    timezone: str,
) -> list[str]:
    known = [(index, point) for index, point in enumerate(points) if point is not None]
    if not known:
        return [
            f"{title}: надійних точок немає з {len(points)} записів.",
            f"  Причини: {_format_skip_reasons(report.skipped_reasons)}.",
        ]
    values = [point for _, point in known]
    min_value = min(values)
    max_value = max(values)
    avg = sum(values) / len(values)
    low_times = _metric_times_for_value(entries, known, min_value, timezone)
    high_times = _metric_times_for_value(entries, known, max_value, timezone)
    return [
        f"{title}: {_score_word(avg)}; сер={avg:.1f}/10, мін={min_value}, макс={max_value}, надійних точок={len(known)}/{len(points)}.",
        f"  Найнижче: {', '.join(low_times[:3])}; найвище: {', '.join(high_times[:3])}.",
        f"  Не графилося: {_format_skip_reasons(report.skipped_reasons)}.",
    ]


def _story_material_lines(
    *,
    entries: list[Entry],
    extraction_by_entry: dict[str, dict[str, Any]],
    mood_points: list[int | None],
    energy_points: list[int | None],
) -> list[str]:
    analyzed = 0
    labeled = 0
    story_without_graph_points = 0
    entry_type_counts: Counter[str] = Counter()
    for index, entry in enumerate(entries):
        result = extraction_by_entry.get(str(entry.id), {})
        if not result:
            continue
        analyzed += 1
        entry_type = result.get("entry_type")
        if isinstance(entry_type, str) and entry_type:
            entry_type_counts[_entry_type_display_label(entry_type)] += 1
        has_labels = any(
            result.get(key)
            for key in (
                "activity_labels",
                "state_labels",
                "emotion_labels",
                "pleasant_moments",
                "what_helped",
                "what_worsened",
            )
        )
        if has_labels:
            labeled += 1
        has_graph_point = (
            index < len(mood_points)
            and mood_points[index] is not None
            or index < len(energy_points)
            and energy_points[index] is not None
        )
        if has_labels and not has_graph_point:
            story_without_graph_points += 1
    lines = [
        f"- AI-аналіз є для {analyzed}/{len(entries)} записів.",
        f"- Мітки активностей/станів/емоцій є у {labeled} записах.",
        f"- Корисні для історії, але без надійних mood/energy точок: {story_without_graph_points}.",
    ]
    if entry_type_counts:
        lines.append(f"- Типи записів: {_format_inline_counts(entry_type_counts, limit=5)}.")
    return lines


def _entry_type_display_label(entry_type: str) -> str:
    labels = {
        "current_state": "поточний стан",
        "activity_only": "лише активність",
        "sleep": "сон",
        "dream": "сновидіння",
        "photo_only": "лише фото",
        "reply_fragment": "фрагмент відповіді",
        "reflection": "рефлексія",
        "social_event": "соціальна подія",
        "body_state": "тілесний стан",
        "command_or_system": "службова дія",
        "mixed": "змішаний запис",
        "unknown": "невідомо",
    }
    return labels.get(entry_type, entry_type)


def _format_inline_counts(counts: dict[str, int] | Counter[str], *, limit: int = 5) -> str:
    if not counts:
        return "немає"
    return ", ".join(f"{key} {value}" for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit])


def _emotion_overview_lines(
    emotions: list[list[str]],
    *,
    entries: list[Entry],
    timezone: str,
    structured_points: int = 0,
    fallback_points: int = 0,
) -> list[str]:
    known = [(index, item) for index, item in enumerate(emotions) if item]
    if not known:
        return ["- Явних емоцій для карти поки замало."]
    counts = Counter(emotion for _, item in known for emotion in item)
    dominant, count = counts.most_common(1)[0]
    lines = [f"- Покриття емоцій: {len(known)}/{len(entries)} записів; найчастіше: {dominant} ({count})."]
    transitions = []
    previous: str | None = None
    mixed_count = sum(1 for _, item in known if len(item) > 1)
    for index, item in known:
        emotion = item[0]
        if previous is not None and emotion != previous:
            transitions.append(f"{_entry_time_text(entries[index], timezone)}: {previous} -> {emotion}")
        previous = emotion
    if mixed_count:
        lines.append(f"- Змішаних моментів: {mixed_count}.")
    if transitions:
        lines.append("- Помітні зміни: " + "; ".join(transitions[:3]) + ".")
    return lines


def _score_word(value: float) -> str:
    if value < 2.5:
        return "дуже низько"
    if value < 3.5:
        return "низько"
    if value < 4.75:
        return "змішано/нижче середнього"
    if value < 6.25:
        return "середньо"
    if value < 7.25:
        return "добре"
    return "дуже високо"


def _metric_times_for_value(
    entries: list[Entry],
    known: list[tuple[int, int]],
    value: int,
    timezone: str,
) -> list[str]:
    times = []
    for index, point in known:
        if point == value and index < len(entries):
            times.append(_entry_time_text(entries[index], timezone))
    return times or ["невідомо"]


def _first_known_metric_time(
    entries: list[Entry],
    point_groups: list[list[int | None]],
    timezone: str,
) -> str | None:
    for index, _entry in enumerate(entries):
        if any(index < len(points) and points[index] is not None for points in point_groups):
            return _entry_time_text(entries[index], timezone)
    return None


def _last_known_metric_time(
    entries: list[Entry],
    point_groups: list[list[int | None]],
    timezone: str,
) -> str | None:
    for index in range(len(entries) - 1, -1, -1):
        if any(index < len(points) and points[index] is not None for points in point_groups):
            return _entry_time_text(entries[index], timezone)
    return None


def _entry_chart_x_values(entries: list[Entry], timezone: str) -> list[float]:
    tz = zoneinfo(timezone)
    timestamps = [_entry_timestamp(entry, tz) for entry in entries]
    known = [timestamp for timestamp in timestamps if timestamp is not None]
    if not known:
        return [float(index) for index, _entry in enumerate(entries)]
    start = min(known)
    values: list[float] = []
    previous = 0.0
    for index, timestamp in enumerate(timestamps):
        if timestamp is None:
            value = previous + 1.0 if values else float(index)
        else:
            value = max(0.0, (timestamp - start).total_seconds() / 60)
            if values and value < previous:
                value = previous
        values.append(value)
        previous = value
    return values


def _meaningful_pleasant_moments(items: list[object]) -> list[str]:
    ignored = {"", "немає", "немає даних", "невідомо", "unknown", "none", "нічого"}
    moments = []
    for item in items:
        text = " ".join(str(item or "").strip().split())
        normalized = text.lower().strip(" .,!?:;")
        if normalized in ignored or len(normalized) < 10:
            continue
        moments.append(text)
    return moments


def _dedupe_texts(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for item in items:
        key = " ".join(item.lower().split())
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _format_bullets(items: list[str], *, empty: str = "немає даних") -> str:
    if not items:
        return empty
    return "\n".join(f"- {item}" for item in items)


def _sparkline(points: list[int | None]) -> str:
    if not points:
        return "немає даних"
    blocks = "▁▂▃▄▅▆▇█"
    chars = [
        blocks[min(7, max(0, int(point / 10 * 7)))] if point is not None else "·"
        for point in points
    ]
    known = [point for point in points if point is not None]
    if not known:
        return "·" * len(points) + "  даних мало"
    return (
        "".join(chars)
        + f"  мін={min(known)} сер={sum(known) / len(known):.1f} "
        + f"макс={max(known)} даних={len(known)}/{len(points)}"
    )


def _line_chart_png(
    series: dict[str, list[int | None]],
    width: int = 1800,
    height: int = 1040,
    *,
    labels: list[str] | None = None,
    x_values: list[float] | None = None,
    title: str = "Динаміка стану",
    subtitle: str | None = None,
) -> bytes:
    image = Image.new("RGB", (width, height), (250, 252, 255))
    draw = ImageDraw.Draw(image)
    title_font = _chart_font(42, bold=True)
    text_font = _chart_font(28)
    small_font = _chart_font(22)

    plot_left = 132
    plot_right = width - 88
    plot_top = 170
    plot_bottom = height - 176
    colors = {"mood": (37, 99, 235), "energy": (22, 163, 74)}
    names = {"mood": "настрій", "energy": "енергія"}
    max_len = max((len(points) for points in series.values()), default=0)
    labels = labels or [str(index + 1) for index in range(max_len)]
    if x_values is None or len(x_values) < max_len:
        x_values = [float(index) for index in range(max_len)]
    else:
        x_values = [float(value) for value in x_values[:max_len]]
    x_min = min(x_values, default=0.0)
    x_max = max(x_values, default=0.0)

    draw.rounded_rectangle((36, 32, width - 36, height - 32), radius=34, fill=(255, 255, 255), outline=(226, 232, 240), width=3)
    draw.text((82, 66), title, fill=(15, 23, 42), font=title_font)
    if subtitle:
        draw.text((82, 122), subtitle, fill=(71, 85, 105), font=text_font)

    legend_x = width - 450
    for offset, key in enumerate(("mood", "energy")):
        y = 78 + offset * 48
        color = colors[key]
        draw.line((legend_x, y + 16, legend_x + 68, y + 16), fill=color, width=7)
        draw.ellipse((legend_x + 26, y + 7, legend_x + 42, y + 23), fill=color)
        draw.text((legend_x + 88, y), names[key], fill=(30, 41, 59), font=text_font)

    for level in range(0, 11):
        y = plot_bottom - int(level / 10 * (plot_bottom - plot_top))
        grid_color = (226, 232, 240) if level in {0, 5, 10} else (241, 245, 249)
        draw.line((plot_left, y, plot_right, y), fill=grid_color, width=3 if level in {0, 5, 10} else 2)
        draw.text((82, y - 14), str(level), fill=(100, 116, 139), font=small_font)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=(148, 163, 184), width=3)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=(148, 163, 184), width=3)

    if max_len <= 0:
        return _image_png_bytes(image)

    def point_xy(index: int, value: int) -> tuple[int, int]:
        if max_len == 1 or x_max <= x_min:
            x = (plot_left + plot_right) // 2
        else:
            x = plot_left + int((x_values[index] - x_min) / (x_max - x_min) * (plot_right - plot_left))
        y = plot_bottom - int(value / 10 * (plot_bottom - plot_top))
        return x, y

    x_label_indexes = _chart_label_indexes(max_len)
    for index in x_label_indexes:
        x = point_xy(index, 0)[0]
        label = labels[index] if index < len(labels) else str(index + 1)
        draw.line((x, plot_bottom, x, plot_bottom + 12), fill=(148, 163, 184), width=2)
        draw.text((x - 40, plot_bottom + 24), label, fill=(100, 116, 139), font=small_font)

    for name, points in series.items():
        color = colors.get(name, (15, 23, 42))
        previous: tuple[int, int] | None = None
        previous_index: int | None = None
        for index, value in enumerate(points):
            if value is None:
                continue
            current = point_xy(index, max(0, min(value, 10)))
            if previous is not None:
                if previous_index is not None and index - previous_index == 1:
                    draw.line((*previous, *current), fill=color, width=7)
                else:
                    _draw_dashed_line(draw, previous, current, fill=color, width=5)
            draw.ellipse((current[0] - 10, current[1] - 10, current[0] + 10, current[1] + 10), fill=color, outline=(255, 255, 255), width=3)
            previous = current
            previous_index = index

    return _image_png_bytes(image)


def _emotion_timeline_png(
    emotions: list[EmotionTimelineRow | list[EmotionSignalPoint] | list[str]],
    width: int = 1800,
    height: int = 1040,
    *,
    labels: list[str] | None = None,
    x_values: list[float] | None = None,
    title: str = "Емоційна карта дня",
) -> bytes:
    colors = _emotion_colors()
    timeline_rows = [_coerce_emotion_timeline_row(item) for item in emotions]
    signal_rows = [item.signals for item in timeline_rows]
    known_emotions = [signal.label for item in signal_rows for signal in item if signal.label in colors]
    if not known_emotions:
        image = Image.new("RGB", (width, height), (250, 252, 255))
        return _image_png_bytes(image)

    emotion_scores: Counter[str] = Counter()
    for item in signal_rows:
        for signal in item:
            if signal.label in colors:
                emotion_scores[signal.label] += signal.intensity * max(signal.confidence, 0.35)
    emotion_order = [emotion for emotion, _score in emotion_scores.most_common(7)]

    lane_count = len(emotion_order)
    height = max(height, 300 + lane_count * 92)
    image = Image.new("RGB", (width, height), (250, 252, 255))
    draw = ImageDraw.Draw(image)
    title_font = _chart_font(42, bold=True)
    label_font = _chart_font(24, bold=True)
    small_font = _chart_font(22)

    plot_left = 230
    plot_right = width - 92
    plot_top = 180
    plot_bottom = height - 164
    max_len = len(emotions)
    labels = labels or [str(index + 1) for index in range(max_len)]
    if not x_values or len(x_values) != max_len:
        x_values = [float(index) for index in range(max_len)]
    else:
        x_values = [float(value) for value in x_values]
    x_min = min(x_values, default=0.0)
    x_max = max(x_values, default=0.0)

    draw.rounded_rectangle(
        (36, 32, width - 36, height - 32),
        radius=34,
        fill=(255, 255, 255),
        outline=(226, 232, 240),
        width=3,
    )
    draw.text((82, 66), title, fill=(15, 23, 42), font=title_font)

    if max_len <= 0:
        return _image_png_bytes(image)

    def point_x(index: int) -> int:
        if max_len == 1 or x_max <= x_min:
            return (plot_left + plot_right) // 2
        return plot_left + int((x_values[index] - x_min) / (x_max - x_min) * (plot_right - plot_left))

    def lane_y(index: int) -> int:
        if lane_count == 1:
            return (plot_top + plot_bottom) // 2
        return int(plot_top + index / (lane_count - 1) * (plot_bottom - plot_top))

    x_label_indexes = _chart_label_indexes(max_len)
    for index in x_label_indexes:
        x = point_x(index)
        label = labels[index] if index < len(labels) else str(index + 1)
        draw.line((x, plot_top - 16, x, plot_bottom + 16), fill=(248, 250, 252), width=2)
        draw.line((x, plot_bottom, x, plot_bottom + 12), fill=(148, 163, 184), width=2)
        draw.text((x - 40, plot_bottom + 24), label, fill=(100, 116, 139), font=small_font)

    for lane_index, emotion in enumerate(emotion_order):
        color = colors[emotion]
        y = lane_y(lane_index)
        draw.line((plot_left, y, plot_right, y), fill=(226, 232, 240), width=3)
        draw.ellipse((82, y - 11, 104, y + 11), fill=color)
        draw.text((116, y - 17), emotion, fill=(30, 41, 59), font=label_font)

        points: list[tuple[int, int, int, int, float, tuple[int, int, int]]] = []
        for index, row in enumerate(timeline_rows):
            signal = next((candidate for candidate in row.signals if candidate.label == emotion), None)
            if signal is not None:
                intensity = max(0.12, min(1.0, signal.intensity))
                confidence = max(0.2, min(1.0, signal.confidence))
                radius = int(9 + 16 * intensity)
                signal_color = _emotion_signal_color(color, intensity=intensity, confidence=confidence)
                points.append((index, point_x(index), y, radius, confidence, signal_color))
        previous: tuple[int, int, int, int, float, tuple[int, int, int]] | None = None
        for point in points:
            if previous is not None and point[0] - previous[0] <= 2:
                line_color = _average_color(previous[5], point[5])
                draw.line((previous[1], previous[2], point[1], point[2]), fill=line_color, width=min(previous[3], 16))
            previous = point
        for _index, x, y, radius, confidence, signal_color in points:
            outline = (255, 255, 255) if confidence >= 0.65 else (226, 232, 240)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=signal_color, outline=outline, width=4)

    return _image_png_bytes(image)


def _affect_spectrum_point(result: dict[str, Any]) -> AffectSpectrumPoint | None:
    mood = _graphable_feature_score(result, "mood")
    if mood is None:
        return None
    confidence = _feature_confidence(result.get("mood")) or 0.65
    signals = _entry_emotion_signals(result)
    color = _spectrum_signal_color(signals) or _spectrum_tone_color(mood / 10)
    return AffectSpectrumPoint(
        tone=max(0.0, min(1.0, mood / 10)),
        color=color,
        confidence=max(0.2, min(1.0, confidence)),
    )


def _spectrum_signal_color(signals: list[EmotionSignalPoint]) -> tuple[int, int, int] | None:
    weighted: list[tuple[tuple[int, int, int], float]] = []
    for signal in signals:
        color = EMOTION_COLORS.get(signal.label)
        if color is None:
            continue
        weight = max(0.1, signal.intensity) * max(0.2, signal.confidence)
        weighted.append((color, weight))
    if not weighted:
        return None
    total = sum(weight for _color, weight in weighted)
    return tuple(
        int(round(sum(color[index] * weight for color, weight in weighted) / total))
        for index in range(3)
    )


def _spectrum_tone_color(tone: float) -> tuple[int, int, int]:
    anchors = (
        (0.0, (59, 130, 246)),
        (0.5, (217, 119, 6)),
        (1.0, (22, 163, 74)),
    )
    for (left_position, left_color), (right_position, right_color) in zip(anchors, anchors[1:], strict=True):
        if tone <= right_position:
            ratio = (tone - left_position) / (right_position - left_position)
            return _mix_color(left_color, right_color, ratio)
    return anchors[-1][1]


def _affect_spectrum_png(
    points: list[AffectSpectrumPoint | None],
    width: int = 1800,
    height: int = 860,
    *,
    labels: list[str] | None = None,
    x_values: list[float] | None = None,
    title: str = "Спектр стану дня",
) -> bytes:
    output_width, output_height = width, height
    render_scale = 2
    width *= render_scale
    height *= render_scale
    image = Image.new("RGB", (width, height), (250, 252, 255))
    draw = ImageDraw.Draw(image)
    title_font = _chart_font(42 * render_scale, bold=True)
    small_font = _chart_font(22 * render_scale)
    plot_left, plot_right = 158 * render_scale, width - 92 * render_scale
    plot_top, plot_bottom = 178 * render_scale, height - 142 * render_scale
    max_len = len(points)
    labels = labels or [str(index + 1) for index in range(max_len)]
    if x_values is None or len(x_values) != max_len:
        x_values = [float(index) for index in range(max_len)]
    else:
        x_values = [float(value) for value in x_values]
    x_min = min(x_values, default=0.0)
    x_max = max(x_values, default=0.0)

    draw.rounded_rectangle(
        (36 * render_scale, 32 * render_scale, width - 36 * render_scale, height - 32 * render_scale),
        radius=34 * render_scale,
        fill=(255, 255, 255),
        outline=(226, 232, 240),
        width=3 * render_scale,
    )
    draw.text((82 * render_scale, 66 * render_scale), title, fill=(15, 23, 42), font=title_font)

    midpoint = (plot_top + plot_bottom) // 2
    draw.rounded_rectangle(
        (plot_left, plot_top, plot_right, midpoint),
        radius=20 * render_scale,
        fill=(235, 248, 240),
    )
    draw.rectangle((plot_left, midpoint, plot_right, plot_bottom), fill=(254, 241, 242))
    draw.rectangle(
        (plot_left, midpoint - 54 * render_scale, plot_right, midpoint + 54 * render_scale),
        fill=(250, 247, 237),
    )
    for y in (plot_top, midpoint, plot_bottom):
        draw.line((plot_left, y, plot_right, y), fill=(226, 232, 240), width=2 * render_scale)
    draw.text((58 * render_scale, plot_top - 12 * render_scale), "легше", fill=(100, 116, 139), font=small_font)
    draw.text((58 * render_scale, midpoint - 12 * render_scale), "змішано", fill=(100, 116, 139), font=small_font)
    draw.text((58 * render_scale, plot_bottom - 12 * render_scale), "важче", fill=(100, 116, 139), font=small_font)

    if max_len <= 0:
        return _image_png_bytes(image)

    def point_x(index: int) -> int:
        if max_len == 1 or x_max <= x_min:
            return (plot_left + plot_right) // 2
        return plot_left + int((x_values[index] - x_min) / (x_max - x_min) * (plot_right - plot_left))

    def point_y(tone: float) -> int:
        return plot_bottom - int(max(0.0, min(1.0, tone)) * (plot_bottom - plot_top))

    x_label_indexes = _chart_label_indexes(max_len)
    for index in x_label_indexes:
        x = point_x(index)
        label = labels[index] if index < len(labels) else str(index + 1)
        draw.line((x, plot_bottom, x, plot_bottom + 10 * render_scale), fill=(148, 163, 184), width=2 * render_scale)
        draw.text((x - 38 * render_scale, plot_bottom + 24 * render_scale), label, fill=(100, 116, 139), font=small_font)

    observed = [
        (index, point_x(index), point_y(point.tone), point)
        for index, point in enumerate(points)
        if point is not None
    ]
    known_x_gaps = [
        right[1] - left[1]
        for left, right in zip(observed, observed[1:], strict=False)
        if right[1] > left[1]
    ]
    typical_gap = sorted(known_x_gaps)[len(known_x_gaps) // 2] if known_x_gaps else plot_right - plot_left
    connection_limit = max(100 * render_scale, typical_gap * 3)

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    for previous, current, curve in _spectrum_curve_segments(observed):
        missing_observations = current[0] - previous[0] - 1
        is_long_gap = current[1] - previous[1] > connection_limit
        opacity = 24 if missing_observations or is_long_gap else 48
        _draw_spectrum_ribbon(
            overlay_draw,
            previous=previous,
            current=current,
            curve=curve,
            width=92 * render_scale,
            opacity=opacity,
            muted=bool(missing_observations or is_long_gap),
        )
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)
    for previous, current, curve in _spectrum_curve_segments(observed):
        missing_observations = current[0] - previous[0] - 1
        is_long_gap = current[1] - previous[1] > connection_limit
        _draw_spectrum_ribbon(
            draw,
            previous=previous,
            current=current,
            curve=curve,
            width=48 * render_scale,
            muted=bool(missing_observations or is_long_gap),
        )
        _draw_spectrum_highlight(
            draw,
            previous=previous,
            current=current,
            curve=curve,
            width=7 * render_scale,
            muted=bool(missing_observations or is_long_gap),
        )
    for _index, x, y, point in observed:
        radius = int((10 + 9 * point.confidence) * render_scale)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(255, 255, 255),
            outline=(38, 50, 65),
            width=4 * render_scale,
        )

    return _image_png_bytes(image.resize((output_width, output_height), Image.Resampling.LANCZOS))


def _draw_spectrum_ribbon(
    draw: ImageDraw.ImageDraw,
    *,
    previous: tuple[int, int, int, AffectSpectrumPoint],
    current: tuple[int, int, int, AffectSpectrumPoint],
    curve: list[tuple[int, int]],
    width: int,
    muted: bool,
    opacity: int | None = None,
) -> None:
    radius = width // 2
    last_index = max(1, len(curve) - 1)
    for index, (start, end) in enumerate(zip(curve, curve[1:], strict=False)):
        color = _mix_color(previous[3].color, current[3].color, (index + 0.5) / last_index)
        if muted:
            color = _mix_color((255, 255, 255), color, 0.58)
        fill: tuple[int, int, int] | tuple[int, int, int, int]
        fill = (*color, opacity) if opacity is not None else color
        draw.line((*start, *end), fill=fill, width=width)
        draw.ellipse((end[0] - radius, end[1] - radius, end[0] + radius, end[1] + radius), fill=fill)


def _draw_spectrum_highlight(
    draw: ImageDraw.ImageDraw,
    *,
    previous: tuple[int, int, int, AffectSpectrumPoint],
    current: tuple[int, int, int, AffectSpectrumPoint],
    curve: list[tuple[int, int]],
    width: int,
    muted: bool,
) -> None:
    last_index = max(1, len(curve) - 1)
    for index, (start, end) in enumerate(zip(curve, curve[1:], strict=False)):
        color = _mix_color(previous[3].color, current[3].color, (index + 0.5) / last_index)
        color = _mix_color(color, (255, 255, 255), 0.20 if muted else 0.34)
        draw.line((*start, *end), fill=color, width=width)


def _spectrum_curve_segments(
    points: list[tuple[int, int, int, AffectSpectrumPoint]],
) -> list[tuple[tuple[int, int, int, AffectSpectrumPoint], tuple[int, int, int, AffectSpectrumPoint], list[tuple[int, int]]]]:
    segments = []
    for index in range(len(points) - 1):
        before = points[index - 1] if index else points[index]
        start = points[index]
        end = points[index + 1]
        after = points[index + 2] if index + 2 < len(points) else end
        control_one = (start[1] + (end[1] - before[1]) / 6, start[2] + (end[2] - before[2]) / 6)
        control_two = (end[1] - (after[1] - start[1]) / 6, end[2] - (after[2] - start[2]) / 6)
        curve = []
        for step in range(33):
            t = step / 32
            inverse = 1 - t
            x = inverse**3 * start[1] + 3 * inverse**2 * t * control_one[0] + 3 * inverse * t**2 * control_two[0] + t**3 * end[1]
            y = inverse**3 * start[2] + 3 * inverse**2 * t * control_one[1] + 3 * inverse * t**2 * control_two[1] + t**3 * end[2]
            curve.append((int(round(x)), int(round(y))))
        segments.append((start, end, curve))
    return segments


def _emotion_signal_color(
    color: tuple[int, int, int],
    *,
    intensity: float,
    confidence: float,
) -> tuple[int, int, int]:
    strength = max(0.0, min(1.0, intensity * (0.65 + 0.35 * confidence)))
    return _mix_color((226, 232, 240), color, 0.28 + 0.72 * strength)


def _mix_color(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    amount: float,
) -> tuple[int, int, int]:
    ratio = max(0.0, min(1.0, amount))
    return tuple(int(round(start[index] + (end[index] - start[index]) * ratio)) for index in range(3))


def _average_color(
    first: tuple[int, int, int],
    second: tuple[int, int, int],
) -> tuple[int, int, int]:
    return tuple(int(round((first[index] + second[index]) / 2)) for index in range(3))


def _emotion_timeline_row(result: dict[str, Any]) -> EmotionTimelineRow:
    signals = _entry_emotion_signals(result)
    return EmotionTimelineRow(
        signals=signals,
        observed=bool(signals) or str(result.get("emotion_observation") or "").strip() == "no_current_emotion",
    )


def _coerce_emotion_timeline_row(
    row: EmotionTimelineRow | list[EmotionSignalPoint] | list[str],
) -> EmotionTimelineRow:
    if isinstance(row, EmotionTimelineRow):
        return row
    signals = _coerce_emotion_signal_row(row)
    return EmotionTimelineRow(signals=signals, observed=bool(signals))


def _coerce_emotion_signal_row(row: list[EmotionSignalPoint] | list[str]) -> list[EmotionSignalPoint]:
    signals: list[EmotionSignalPoint] = []
    for item in row:
        if isinstance(item, EmotionSignalPoint):
            signals.append(item)
            continue
        label = _emotion_label(item)
        if not label:
            continue
        signals.append(
            EmotionSignalPoint(
                label=label,
                intensity=0.55,
                confidence=0.45,
                intensity_level="moderate",
            )
        )
    return signals


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    fill: tuple[int, int, int],
    width: int,
    dash: int = 18,
    gap: int = 14,
) -> None:
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    distance = (dx * dx + dy * dy) ** 0.5
    if distance == 0:
        return
    step = dash + gap
    travelled = 0.0
    while travelled < distance:
        segment_start = travelled
        segment_end = min(travelled + dash, distance)
        sx = x1 + dx * (segment_start / distance)
        sy = y1 + dy * (segment_start / distance)
        ex = x1 + dx * (segment_end / distance)
        ey = y1 + dy * (segment_end / distance)
        draw.line((sx, sy, ex, ey), fill=fill, width=width)
        travelled += step


def _chart_font(size: int, *, bold: bool = False):
    font_paths = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _chart_label_indexes(length: int) -> list[int]:
    if length <= 1:
        return [0] if length else []
    if length <= 6:
        return list(range(length))
    indexes = {0, length - 1, length // 2, length // 4, (length * 3) // 4}
    return sorted(indexes)


def _image_png_bytes(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _format_counts(counts: dict[str, int], empty: str = "немає даних", *, limit: int = 8) -> str:
    if not counts:
        return empty
    return "\n".join(f"- {key}: {value}" for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit])


def _metric_entries(entries: list[Entry]) -> list[Entry]:
    excluded_sources = {"correction", "profile_context_update"}
    return [entry for entry in entries if entry.source not in excluded_sources]


async def _period_metrics_report(session: AsyncSession, *, user: User, summary: Summary) -> dict[str, Any]:
    start_date, end_date = _summary_local_date_range(summary, user.timezone)
    days = await repo.list_days_between(session, user_id=user.id, start_date=start_date, end_date=end_date)
    days_by_id = _days_by_id(days)
    entries = _metric_entries(
        list(
            await repo.list_entries_between(
                session,
                user_id=user.id,
                start=summary.period_start,
                end=summary.period_end,
            )
        )
    )
    analyses = await repo.list_analyses_for_targets(
        session,
        target_type="entry",
        target_ids=[entry.id for entry in entries],
    )
    extraction_by_entry = _latest_feature_results_by_entry(analyses)
    activity_counts: dict[str, int] = defaultdict(int)
    state_counts: dict[str, int] = defaultdict(int)
    emotion_counts: dict[str, int] = defaultdict(int)
    emotion_signals_by_day: dict[date, dict[str, list[EmotionSignalPoint]]] = defaultdict(lambda: defaultdict(list))
    emotion_observed_days: set[date] = set()
    structured_emotion_points = 0
    fallback_emotion_points = 0
    quality_counts: dict[str, int] = defaultdict(int)
    mood_by_day: dict[date, list[int]] = defaultdict(list)
    energy_by_day: dict[date, list[int]] = defaultdict(list)
    meaningful_pleasant: list[str] = []
    unnormalized_label_count = 0

    for entry in entries:
        result = extraction_by_entry.get(str(entry.id), {})
        for label in result.get("activity_labels") or []:
            normalized_label = _metrics_label_text(label)
            if normalized_label is None:
                unnormalized_label_count += 1
                continue
            activity_counts[normalized_label] += 1
        for label in result.get("state_labels") or []:
            normalized_label = _metrics_label_text(label)
            if normalized_label is None:
                unnormalized_label_count += 1
                continue
            state_counts[normalized_label] += 1
        structured_signals = _structured_emotion_signals(result)
        signals = _entry_emotion_signals(result)
        for emotion in _entry_emotions(result):
            emotion_counts[emotion] += 1
        quality_counts[_data_quality_label(result.get("data_quality"))] += 1
        entry_date = _entry_journal_date(entry, days_by_id=days_by_id, timezone=user.timezone)
        if entry_date is not None:
            if structured_signals:
                structured_emotion_points += 1
            elif signals:
                fallback_emotion_points += 1
            if signals or str(result.get("emotion_observation") or "").strip() == "no_current_emotion":
                emotion_observed_days.add(entry_date)
            for signal in signals:
                emotion_signals_by_day[entry_date][signal.label].append(signal)
        mood = _graphable_feature_score(result, "mood")
        energy = _graphable_feature_score(result, "energy")
        if entry_date is not None and mood is not None:
            mood_by_day[entry_date].append(mood)
        if entry_date is not None and energy is not None:
            energy_by_day[entry_date].append(energy)
        meaningful_pleasant.extend(_meaningful_pleasant_moments(result.get("pleasant_moments") or []))

    dates = list(_date_range(start_date, end_date))
    entry_counts = _entry_counts_by_journal_date(entries, days_by_id=days_by_id, timezone=user.timezone)
    active_dates = {current for current, count in entry_counts.items() if count}
    mood_graphable_dates = {current for current, points in mood_by_day.items() if points}
    energy_graphable_dates = {current for current, points in energy_by_day.items() if points}
    dates_with_any_graph_points = mood_graphable_dates | energy_graphable_dates
    daily_emotion_rows = [
        _period_emotion_row(
            emotion_signals_by_day.get(current, {}),
            observed=current in emotion_observed_days,
        )
        for current in dates
    ]
    return {
        "entry_count": len(entries),
        "active_days": len(active_dates),
        "total_days": len(dates),
        "pleasant_count": len(_dedupe_texts(meaningful_pleasant)),
        "daily_mood": [_rounded_average(mood_by_day.get(current, [])) for current in dates],
        "daily_energy": [_rounded_average(energy_by_day.get(current, [])) for current in dates],
        "mood_graphable_days": len(mood_graphable_dates),
        "energy_graphable_days": len(energy_graphable_dates),
        "story_only_days": len(active_dates - dates_with_any_graph_points),
        "mood_point_count": sum(len(points) for points in mood_by_day.values()),
        "energy_point_count": sum(len(points) for points in energy_by_day.values()),
        "quality_counts": quality_counts,
        "state_counts": state_counts,
        "emotion_counts": emotion_counts,
        "emotion_observed_days": len(emotion_observed_days),
        "structured_emotion_points": structured_emotion_points,
        "fallback_emotion_points": fallback_emotion_points,
        "daily_emotion_rows": daily_emotion_rows,
        "activity_counts": activity_counts,
        "unnormalized_label_count": unnormalized_label_count,
    }


def _period_emotion_row(
    signals_by_label: dict[str, list[EmotionSignalPoint]],
    *,
    observed: bool,
) -> EmotionTimelineRow:
    signals: list[EmotionSignalPoint] = []
    for label, signals_for_label in signals_by_label.items():
        if not signals_for_label:
            continue
        total_weight = sum(max(signal.confidence, 0.2) for signal in signals_for_label)
        if total_weight <= 0:
            continue
        intensity = sum(
            max(0.0, min(1.0, signal.intensity)) * max(signal.confidence, 0.2)
            for signal in signals_for_label
        ) / total_weight
        confidence = max(signal.confidence for signal in signals_for_label)
        signals.append(
            EmotionSignalPoint(
                label=label,
                intensity=intensity,
                confidence=confidence,
                intensity_level=_emotion_intensity_level(intensity),
            )
        )
    return EmotionTimelineRow(
        signals=sorted(signals, key=lambda signal: signal.intensity * signal.confidence, reverse=True)[:8],
        observed=observed,
    )


def _emotion_intensity_level(value: float) -> str:
    if value <= 0:
        return "unclear"
    if value < 0.22:
        return "trace"
    if value < 0.43:
        return "mild"
    if value < 0.68:
        return "moderate"
    if value < 0.92:
        return "strong"
    return "overwhelming"


def _summary_local_date_range(summary: Summary, timezone: str) -> tuple[date, date]:
    details = summary.details or {}
    start_date = _parse_iso_date(details.get("journal_start_date"))
    end_date = _parse_iso_date(details.get("journal_end_date"))
    if start_date is not None and end_date is not None:
        return start_date, end_date
    journal_date = _parse_iso_date(details.get("journal_date"))
    if journal_date is not None:
        return journal_date, journal_date
    start = _timestamp_in_timezone(summary.period_start, zoneinfo(timezone))
    end = _timestamp_in_timezone(summary.period_end, zoneinfo(timezone))
    if start is None or end is None:
        return date.today(), date.today()
    return start.date(), end.date()


def _parse_iso_date(value: Any) -> date | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _date_range(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _entry_counts_by_local_date(entries, *, timezone: str) -> dict[date, int]:
    counts: dict[date, int] = defaultdict(int)
    for entry in entries:
        entry_date = _entry_local_date(entry, timezone)
        if entry_date is not None:
            counts[entry_date] += 1
    return counts


def _days_by_id(days: list[Day] | tuple[Day, ...] | Any) -> dict[str, Day]:
    return {str(day.id): day for day in days}


def _entry_counts_by_journal_date(entries, *, days_by_id: dict[str, Day], timezone: str) -> dict[date, int]:
    counts: dict[date, int] = defaultdict(int)
    for entry in entries:
        entry_date = _entry_journal_date(entry, days_by_id=days_by_id, timezone=timezone)
        if entry_date is not None:
            counts[entry_date] += 1
    return counts


def _entry_journal_date(entry: Entry, *, days_by_id: dict[str, Day], timezone: str) -> date | None:
    if entry.day_id:
        day = days_by_id.get(str(entry.day_id))
        if day is not None:
            return day.local_date
    return _entry_local_date(entry, timezone)


def _summaries_by_journal_date(
    summaries: list[Summary] | tuple[Summary, ...] | Any,
    *,
    days_by_id: dict[str, Day],
    timezone: str,
) -> dict[date, Summary]:
    result: dict[date, Summary] = {}
    for summary in summaries:
        summary_date = None
        if summary.day_id:
            day = days_by_id.get(str(summary.day_id))
            if day is not None:
                summary_date = day.local_date
        if summary_date is None:
            summary_date, _ = _summary_local_date_range(summary, timezone)
        result[summary_date] = summary
    return result


def _entry_local_date(entry: Entry, timezone: str) -> date | None:
    timestamp = _entry_timestamp(entry, zoneinfo(timezone))
    return timestamp.date() if timestamp else None


def _rounded_average(values: list[int]) -> int | None:
    if not values:
        return None
    return int(round(sum(values) / len(values)))


def _period_title(summary: Summary) -> str:
    return "Тижневий підсумок" if summary.period_type == "weekly" else "Місячний підсумок"


def _stored_period_analysis(summary: Summary) -> dict[str, Any]:
    value = (summary.details or {}).get("deterministic_period_analysis")
    return value if isinstance(value, dict) else {}


def _period_date_label(summary: Summary) -> str:
    start_label = (summary.details or {}).get("journal_start_date") or _display_date(summary.period_start)
    end_label = (summary.details or {}).get("journal_end_date") or _display_date(summary.period_end)
    return f"{start_label} - {end_label}"


def _period_named_observations(title: str, values: list[dict[str, Any]], *, noun: str) -> str:
    if not values:
        return f"{title}: немає надійних даних."
    lines = [f"{title}:"]
    lines.extend(f"- {item.get('label')}: {item.get('observations')} {noun}" for item in values[:8])
    return "\n".join(lines)


def _period_intensity_observations(values: list[dict[str, Any]]) -> str:
    if not values:
        return "Середня вираженість: немає надійних даних."
    lines = ["Середня вираженість:"]
    for item in values[:8]:
        label = item.get("emotion") or "емоція"
        intensity = item.get("intensity")
        observations = item.get("observations")
        lines.append(f"- {label}: {intensity}/1 ({observations} спостережень)")
    return "\n".join(lines)


def _period_co_occurrence_observations(values: list[dict[str, Any]]) -> str:
    if not values:
        return "Поєднання емоцій: повторів поки замало."
    lines = ["Емоції, що фіксувалися разом:"]
    for item in values[:6]:
        labels = ", ".join(str(value) for value in item.get("emotions") or [])
        lines.append(f"- {labels}: {item.get('observations')} спостережень")
    return "\n".join(lines)


def _format_period_associations(values: list[dict[str, Any]]) -> str:
    if not values:
        return "Повторів у тих самих записах поки недостатньо."
    lines = []
    for item in values:
        activity = item.get("activity") or "активність"
        support = item.get("observations") or 0
        if item.get("kind") == "activity_emotion":
            lines.append(f"- {activity} + {item.get('emotion')}: {support} спільних записів")
            continue
        metrics = []
        if item.get("mood_mean") is not None:
            metrics.append(f"настрій {item['mood_mean']}/10")
        if item.get("energy_mean") is not None:
            metrics.append(f"енергія {item['energy_mean']}/10")
        lines.append(f"- {activity}: {support} записів; {', '.join(metrics) or 'метрик мало'}")
    lines.append("Це повторювані збіги, а не доказ того, що одне спричинило інше.")
    return "\n".join(lines)


def _format_period_comparison(value: dict[str, Any]) -> str:
    if not value:
        return "Попередній період ще не зіставлений."
    previous = value.get("previous_coverage") or {}
    raw_ratio = value.get("coverage_ratio")
    ratio = float(raw_ratio) if isinstance(raw_ratio, int | float) else 0.0
    lines = [
        f"- Покриття активних днів: {ratio * 100:.0f}% від меншого до більшого періоду.",
        f"- Попередній період: {previous.get('active_days', 0)}/{previous.get('total_days', 0)} днів із записами.",
    ]
    if value.get("mood_mean_change") is not None:
        lines.append(f"- Зміна середнього настрою в надійних точках: {value['mood_mean_change']:+.1f}.")
    if value.get("energy_mean_change") is not None:
        lines.append(f"- Зміна середньої енергії в надійних точках: {value['energy_mean_change']:+.1f}.")
    return "\n".join(lines)


def _format_period_rhythm(values: list[dict[str, Any]]) -> str:
    lines = []
    for item in values:
        if not item.get("entry_count"):
            continue
        metrics = []
        if item.get("mood_mean") is not None:
            metrics.append(f"настрій {item['mood_mean']}/10")
        if item.get("energy_mean") is not None:
            metrics.append(f"енергія {item['energy_mean']}/10")
        top = item.get("top_emotions") or []
        if top:
            metrics.append("емоції: " + ", ".join(str(value.get("label")) for value in top[:2]))
        lines.append(f"- {item.get('period')}: {item.get('entry_count')} записів; {', '.join(metrics) or 'метрик мало'}.")
    return "\n".join(lines) if lines else "За частинами доби поки мало даних."


def _diverse_similar_records(
    records: list[EmbeddingRecord],
    *,
    entries_by_id: dict[str, Entry],
    timezone: str | None,
    limit: int,
) -> list[EmbeddingRecord]:
    selected: list[EmbeddingRecord] = []
    seen_dates: set[date] = set()
    for record in records:
        timestamp = _similar_timestamp(record, entries_by_id.get(str(record.target_id)), timezone)
        record_date = timestamp.date() if timestamp else None
        if record_date is not None and record_date in seen_dates:
            continue
        selected.append(record)
        if record_date is not None:
            seen_dates.add(record_date)
        if len(selected) >= limit:
            return selected
    for record in records:
        if record not in selected:
            selected.append(record)
        if len(selected) >= limit:
            break
    return selected


def _similar_timestamp(record: EmbeddingRecord, entry: Entry | None, timezone: str | None) -> datetime | None:
    timestamp = (entry.local_timestamp or entry.created_at) if entry is not None else record.created_at
    return _timestamp_in_timezone(timestamp, zoneinfo(timezone)) if timezone else timestamp


def _similar_record_text(record: EmbeddingRecord, entry: Entry | None) -> str:
    if entry is not None and entry.raw_text:
        return entry.raw_text
    raw = _semantic_field(record.source_text, "Raw:")
    if raw:
        return raw
    summary = _semantic_field(record.source_text, "Micro-summary:")
    if summary:
        return summary
    return record.source_text.replace("\n", " ")


def _semantic_field(text: str, prefix: str) -> str | None:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith(prefix):
            continue
        value_lines = [line[len(prefix) :].strip()]
        for next_line in lines[index + 1 :]:
            if next_line.startswith(("Raw:", "Features:", "Micro-summary:")):
                break
            value_lines.append(next_line.strip())
        value = " ".join(part for part in value_lines if part)
        return value or None
    return None


def _similar_label_counts(
    entry_ids: list[str], features_by_entry: dict[str, dict[str, Any]]
) -> tuple[Counter[str], Counter[str]]:
    activity_counts: Counter[str] = Counter()
    state_counts: Counter[str] = Counter()
    for entry_id in entry_ids:
        result = features_by_entry.get(entry_id, {})
        for label in result.get("activity_labels") or []:
            if label_text := _metrics_label_text(label):
                activity_counts[label_text] += 1
        for label in result.get("state_labels") or []:
            if label_text := _metrics_label_text(label):
                state_counts[label_text] += 1
    return activity_counts, state_counts


def _similar_day_counts(
    records: list[EmbeddingRecord], *, entries_by_id: dict[str, Entry], timezone: str | None
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        timestamp = _similar_timestamp(record, entries_by_id.get(str(record.target_id)), timezone)
        if timestamp is not None:
            counts[timestamp.strftime("%Y-%m-%d")] += 1
    return counts


def _similar_labels(result: dict[str, Any]) -> list[str]:
    labels = []
    for label in [*(result.get("state_labels") or []), *(result.get("activity_labels") or [])]:
        label_text = _metrics_label_text(label)
        if label_text and label_text not in labels:
            labels.append(label_text)
    return labels


def _inline_counts(counts: Counter[str], *, limit: int = 4) -> str:
    return ", ".join(f"{label} ({count})" for label, count in counts.most_common(limit))


def _list_block(title: str, items: list[str]) -> str:
    if not items:
        return f"{title}: немає даних"
    return title + ":\n" + "\n".join(f"- {item}" for item in items)


def _display_date(value: object) -> str:
    if hasattr(value, "date"):
        return str(value.date())
    return str(value)


def _active_window(day: date, timezone: str, settings: UserSettings) -> tuple[datetime, datetime]:
    tz = zoneinfo(timezone)
    start_time = parse_hhmm(settings.active_start)
    end_time = parse_hhmm(settings.active_end)
    start = datetime.combine(day, start_time, tzinfo=tz)
    end = datetime.combine(day, end_time, tzinfo=tz)
    if start_time > end_time:
        end += timedelta(days=1)
    return start, end


def _entry_timestamp(entry: Entry, target_tz: tzinfo | None = None) -> datetime | None:
    timestamp = entry.local_timestamp or entry.created_at
    return _timestamp_in_timezone(timestamp, target_tz)


def _gap_segments(
    entry_times: list[datetime], *, window_start: datetime, window_end: datetime
) -> list[tuple[datetime, datetime, timedelta]]:
    if window_end <= window_start:
        return []
    points = [window_start, *entry_times, window_end]
    gaps: list[tuple[datetime, datetime, timedelta]] = []
    for start, end in zip(points, points[1:], strict=False):
        duration = end - start
        if duration > timedelta():
            gaps.append((start, end, duration))
    return gaps


def _coverage_note(entry_count: int, missed_count: int, largest_gap: timedelta) -> str:
    if entry_count == 0:
        return "немає записів у активному вікні"
    if missed_count >= 3 or largest_gap >= timedelta(hours=4):
        return "фрагментарне, є великі паузи"
    if missed_count or largest_gap >= timedelta(hours=2):
        return "часткове, є прогалини"
    return "достатнє для грубого огляду дня"


def _format_duration(duration: timedelta) -> str:
    minutes = int(duration.total_seconds() // 60)
    hours, remainder = divmod(minutes, 60)
    if hours and remainder:
        return f"{hours} год {remainder} хв"
    if hours:
        return f"{hours} год"
    return f"{remainder} хв"


def _format_entry_line(
    entry: Entry,
    quality_by_entry: dict[str, str],
    labels_by_entry: dict[str, list[str]],
    *,
    timezone: str | None = None,
) -> str:
    time_text = _entry_time_text(entry, timezone)
    text = _truncate(entry.raw_text or "[без тексту]", 120)
    labels = labels_by_entry.get(str(entry.id), [])
    label_text = f" [{', '.join(_ai_label_text(label) for label in labels[:3])}]" if labels else ""
    quality = quality_by_entry.get(str(entry.id))
    quality_text = f" ({_data_quality_label(quality)})" if quality else ""
    return f"{time_text} - {text}{label_text}{quality_text}"


def _entry_time_text(entry: Entry, timezone: str | None = None) -> str:
    return _time_text(entry.local_timestamp or entry.created_at, timezone)


def _ai_label_text(value: str) -> str:
    return " ".join(str(value or "").strip().replace("_", " ").split()) or "невідомо"


def _time_text(timestamp: datetime | None, timezone: str | None = None) -> str:
    if timestamp is None:
        return "??:??"
    if timezone:
        timestamp = _timestamp_in_timezone(timestamp, zoneinfo(timezone))
    return timestamp.strftime("%H:%M")


def _timestamp_in_timezone(timestamp: datetime | None, target_tz: tzinfo | None) -> datetime | None:
    if timestamp is None:
        return None
    if target_tz is None:
        return timestamp
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=target_tz)
    return timestamp.astimezone(target_tz)


def _entry_count_text(count: int) -> str:
    return f"{count} {_ukrainian_plural(count, 'запис', 'записи', 'записів')}"


def _ukrainian_plural(count: int, one: str, few: str, many: str) -> str:
    value = abs(count)
    if value % 100 in {11, 12, 13, 14}:
        return many
    if value % 10 == 1:
        return one
    if value % 10 in {2, 3, 4}:
        return few
    return many


def _truncate(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    if limit <= 1:
        return "…"[:limit]
    return compact[: limit - 1].rstrip() + "…"
