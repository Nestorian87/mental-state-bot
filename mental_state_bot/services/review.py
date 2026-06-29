from __future__ import annotations

import struct
import zlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy.ext.asyncio import AsyncSession

from mental_state_bot.db import repositories as repo
from mental_state_bot.db.models import (
    EmbeddingRecord,
    Entry,
    Media,
    MissedPrompt,
    Summary,
    User,
    UserSettings,
)
from mental_state_bot.time_utils import local_date, parse_hhmm, utc_now


@dataclass(frozen=True)
class PhotoMoment:
    media: Media
    entry: Entry


async def format_today_view(session: AsyncSession, *, user: User, limit: int = 18) -> str:
    day = await repo.get_day_by_date(session, user_id=user.id, local_date_value=local_date(user.timezone))
    if day is None:
        return "За сьогодні ще немає записів."

    entries = list(await repo.list_day_entries(session, day_id=day.id))
    if not entries:
        return "За сьогодні ще немає записів."

    visible_entries = entries[-limit:]
    analyses = await repo.list_analyses_for_targets(
        session,
        target_type="entry",
        target_ids=[entry.id for entry in visible_entries],
    )
    quality_by_entry: dict[str, str] = {}
    labels_by_entry: dict[str, list[str]] = defaultdict(list)
    for analysis in analyses:
        if analysis.task_name != "extract_entry_features":
            continue
        result = analysis.result or {}
        quality_by_entry[str(analysis.target_id)] = result.get("data_quality") or "unknown"
        labels_by_entry[str(analysis.target_id)].extend(result.get("activity_labels") or [])
        labels_by_entry[str(analysis.target_id)].extend(result.get("state_labels") or [])

    lines = [
        f"Сьогодні: {len(entries)} записів.",
        "",
    ]
    if len(entries) > limit:
        lines.append(f"Останні {limit}:")

    for entry in visible_entries:
        lines.append(_format_entry_line(entry, quality_by_entry, labels_by_entry))

    if day.ended_at:
        lines.extend(["", f"День закрито: {day.ended_at.astimezone().strftime('%H:%M')}"])
    return "\n".join(lines)


async def format_raw_entries_view(session: AsyncSession, *, user: User, limit: int = 30) -> str:
    day = await repo.get_day_by_date(session, user_id=user.id, local_date_value=local_date(user.timezone))
    if day is None:
        return "За сьогодні ще немає сирих записів."
    entries = list(await repo.list_day_entries(session, day_id=day.id))
    if not entries:
        return "За сьогодні ще немає сирих записів."

    visible_entries = entries[-limit:]
    lines = [f"Сирі записи за сьогодні: {len(entries)}", ""]
    if len(entries) > limit:
        lines.append(f"Показую останні {limit}.")
    for entry in visible_entries:
        timestamp = entry.local_timestamp or entry.created_at
        time_text = timestamp.strftime("%H:%M") if timestamp else "??:??"
        lines.append(f"{time_text} [{_source_label(entry.source)}] {entry.raw_text or '[без тексту]'}")
    return "\n".join(lines)


async def format_metrics_view(session: AsyncSession, *, user: User) -> str:
    day = await repo.get_day_by_date(session, user_id=user.id, local_date_value=local_date(user.timezone))
    if day is None:
        return "За сьогодні ще немає метрик."
    entries = _metric_entries(list(await repo.list_day_entries(session, day_id=day.id)))
    if not entries:
        return "За сьогодні ще немає метрик."

    analyses = await repo.list_analyses_for_targets(
        session,
        target_type="entry",
        target_ids=[entry.id for entry in entries],
    )
    extraction_by_entry = {
        str(analysis.target_id): analysis.result
        for analysis in analyses
        if analysis.task_name == "extract_entry_features"
    }

    activity_counts: dict[str, int] = defaultdict(int)
    state_counts: dict[str, int] = defaultdict(int)
    quality_counts: dict[str, int] = defaultdict(int)
    mood_points: list[int | None] = []
    energy_points: list[int | None] = []
    pleasant_count = 0

    for entry in entries:
        result = extraction_by_entry.get(str(entry.id), {})
        for label in result.get("activity_labels") or []:
            activity_counts[_human_label(label)] += 1
        for label in result.get("state_labels") or []:
            state_counts[_human_label(label)] += 1
        quality_counts[_data_quality_label(result.get("data_quality"))] += 1
        mood_points.append(_feature_score(result.get("mood")))
        energy_points.append(_feature_score(result.get("energy")))
        pleasant_count += len(result.get("pleasant_moments") or [])

    lines = [
        f"Метрики за сьогодні: {len(entries)} записів.",
        f"Приємні/живі моменти, знайдені AI: {pleasant_count}",
        "",
        "Настрій:",
        _sparkline(mood_points),
        "Енергія:",
        _sparkline(energy_points),
        "",
        "Якість даних:",
        _format_counts(quality_counts),
        "",
        "Найчастіші стани:",
        _format_counts(state_counts, empty="немає явних станів"),
        "",
        "Найчастіші активності:",
        _format_counts(activity_counts, empty="немає явних активностей"),
    ]
    return "\n".join(lines)


async def build_metrics_chart_png(session: AsyncSession, *, user: User) -> bytes | None:
    day = await repo.get_day_by_date(session, user_id=user.id, local_date_value=local_date(user.timezone))
    if day is None:
        return None
    entries = _metric_entries(list(await repo.list_day_entries(session, day_id=day.id)))
    if not entries:
        return None
    analyses = await repo.list_analyses_for_targets(
        session,
        target_type="entry",
        target_ids=[entry.id for entry in entries],
    )
    extraction_by_entry = {
        str(analysis.target_id): analysis.result
        for analysis in analyses
        if analysis.task_name == "extract_entry_features"
    }
    mood_points = [_feature_score((extraction_by_entry.get(str(entry.id)) or {}).get("mood")) for entry in entries]
    energy_points = [
        _feature_score((extraction_by_entry.get(str(entry.id)) or {}).get("energy")) for entry in entries
    ]
    if not any(point is not None for point in [*mood_points, *energy_points]):
        return None
    return _line_chart_png({"mood": mood_points, "energy": energy_points})


async def get_today_photo_moments(session: AsyncSession, *, user: User) -> list[PhotoMoment]:
    day = await repo.get_day_by_date(session, user_id=user.id, local_date_value=local_date(user.timezone))
    if day is None:
        return []
    rows = await repo.list_day_media_with_entries(session, day_id=day.id, media_type="photo")
    return [PhotoMoment(media=media, entry=entry) for media, entry in rows]


def format_photo_moments_view(moments: list[PhotoMoment], *, limit: int = 12) -> str:
    if not moments:
        return "За сьогодні ще немає фото."

    visible = moments[-limit:]
    lines = [f"Фото дня: {len(moments)}", ""]
    if len(moments) > limit:
        lines.append(f"Показую останні {limit}.")
        lines.append("")
    lines.append("Окремо надсилаю фото нижче, щоб вони не змішувалися з текстом і метриками.")
    lines.append("")
    for index, moment in enumerate(visible, start=1):
        timestamp = moment.entry.local_timestamp or moment.media.created_at or moment.entry.created_at
        time_text = timestamp.strftime("%H:%M") if timestamp else "??:??"
        caption = moment.entry.raw_text
        if caption == "[photo]":
            caption = None
        caption_text = f" — {_truncate(caption, 80)}" if caption else ""
        lines.append(f"{index}. {time_text}{caption_text}")
    return "\n".join(lines)


async def format_gaps_view(session: AsyncSession, *, user: User) -> str:
    today = local_date(user.timezone)
    day = await repo.get_day_by_date(session, user_id=user.id, local_date_value=today)
    user_settings = await repo.get_user_settings(session, user.id)
    window_start, window_end = _active_window(today, user.timezone, user_settings)
    now_local = utc_now().astimezone(ZoneInfo(user.timezone))
    coverage_end = min(now_local, window_end)

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
    )


def format_gap_report(
    *,
    entries: list[Entry],
    missed_prompts: list[MissedPrompt],
    window_start: datetime,
    window_end: datetime,
) -> str:
    if window_end <= window_start:
        return "\n".join(
            [
                "Прогалини сьогодні",
                f"Активне вікно починається о {window_start.strftime('%H:%M')}.",
                "Поки рано оцінювати покриття дня.",
            ]
        )

    entry_times = sorted(
        timestamp for entry in entries if (timestamp := _entry_timestamp(entry)) is not None
    )
    entry_times = [timestamp for timestamp in entry_times if window_start <= timestamp <= window_end]
    gaps = _gap_segments(entry_times, window_start=window_start, window_end=window_end)
    notable_gaps = [gap for gap in gaps if gap[2] >= timedelta(minutes=60)]
    largest_gap = max((gap[2] for gap in gaps), default=timedelta())
    coverage_note = _coverage_note(len(entry_times), len(missed_prompts), largest_gap)

    lines = [
        "Прогалини сьогодні",
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


def format_summary_section(summary: Summary, section: str) -> str:
    details = summary.details or {}
    if section == "story":
        return "\n\n".join(
            part
            for part in [
                "Історія дня",
                details.get("story") or summary.short_text,
                _list_block("Що реально відбувалося", details.get("actual_activities") or []),
                _list_block("Зміни стану", details.get("state_changes") or []),
            ]
            if part
        )
    if section == "metrics":
        return "\n".join(
            [
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
            ]
        )
    if section == "timeline":
        return "\n".join(
            [
                "Таймлайн з підсумку",
                "",
                details.get("story") or summary.short_text,
                "",
                _list_block("Прогалини", details.get("data_gaps") or []),
            ]
        )
    return summary.short_text


def format_period_summary(summary: Summary) -> str:
    details = summary.details or {}
    title = "Тижневий підсумок" if summary.period_type == "weekly" else "Місячний підсумок"
    lines = [
        title,
        f"{_display_date(summary.period_start)} - {_display_date(summary.period_end)}",
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


def format_similar_entries(records: list[EmbeddingRecord]) -> str:
    if not records:
        return "Схожих записів поки не знайшов. Можливо, ще немає embeddings або даних замало."

    lines = ["Схожі моменти:"]
    for index, record in enumerate(records, start=1):
        source = _truncate(record.source_text.replace("\n", " "), 260)
        lines.append(f"{index}. {record.created_at.strftime('%Y-%m-%d %H:%M')} - {source}")
    return "\n".join(lines)


def _feature_score(feature: object) -> int | None:
    if not isinstance(feature, dict):
        return None
    value = str(feature.get("value") or "unclear").lower()
    mapping = {
        "very_low": 1,
        "low": 2,
        "somewhat_low": 3,
        "mixed": 4,
        "neutral": 4,
        "medium": 5,
        "moderate": 5,
        "somewhat_high": 6,
        "high": 7,
        "very_high": 8,
    }
    return mapping.get(value)


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
    normalized = str(value or "").strip()
    return known.get(normalized, normalized.replace("_", " ") or "невідомо")


def _sparkline(points: list[int | None]) -> str:
    if not points:
        return "немає даних"
    blocks = "▁▂▃▄▅▆▇█"
    chars = [blocks[max(1, min(point, 8)) - 1] if point else "·" for point in points]
    known = [point for point in points if point is not None]
    if not known:
        return "·" * len(points) + "  даних мало"
    return "".join(chars) + f"  мін={min(known)} сер={sum(known) / len(known):.1f} макс={max(known)}"


def _line_chart_png(series: dict[str, list[int | None]], width: int = 900, height: int = 420) -> bytes:
    pixels = bytearray([255, 255, 255] * width * height)
    margin = 54
    plot_left = margin
    plot_right = width - margin
    plot_top = margin
    plot_bottom = height - margin

    def set_pixel(x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            offset = (y * width + x) * 3
            pixels[offset : offset + 3] = bytes(color)

    def draw_line(x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int], thickness: int = 2) -> None:
        dx = abs(x2 - x1)
        dy = -abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx + dy
        x, y = x1, y1
        while True:
            for tx in range(-thickness + 1, thickness):
                for ty in range(-thickness + 1, thickness):
                    set_pixel(x + tx, y + ty, color)
            if x == x2 and y == y2:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

    def draw_circle(cx: int, cy: int, radius: int, color: tuple[int, int, int]) -> None:
        for y in range(cy - radius, cy + radius + 1):
            for x in range(cx - radius, cx + radius + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius**2:
                    set_pixel(x, y, color)

    for level in range(1, 9):
        y = plot_bottom - int((level - 1) / 7 * (plot_bottom - plot_top))
        draw_line(plot_left, y, plot_right, y, (232, 237, 242), thickness=1)
    draw_line(plot_left, plot_bottom, plot_right, plot_bottom, (145, 160, 180), thickness=1)
    draw_line(plot_left, plot_top, plot_left, plot_bottom, (145, 160, 180), thickness=1)

    colors = {"mood": (37, 99, 235), "energy": (22, 163, 74)}
    max_len = max((len(points) for points in series.values()), default=0)
    if max_len <= 0:
        return _encode_png(width, height, bytes(pixels))

    def point_xy(index: int, value: int) -> tuple[int, int]:
        x = (plot_left + plot_right) // 2 if max_len == 1 else plot_left + int(index / (max_len - 1) * (plot_right - plot_left))
        y = plot_bottom - int((value - 1) / 7 * (plot_bottom - plot_top))
        return x, y

    for name, points in series.items():
        color = colors.get(name, (15, 23, 42))
        previous: tuple[int, int] | None = None
        for index, value in enumerate(points):
            if value is None:
                previous = None
                continue
            current = point_xy(index, max(1, min(value, 8)))
            if previous is not None:
                draw_line(*previous, *current, color, thickness=2)
            draw_circle(*current, radius=5, color=color)
            previous = current

    draw_line(70, 24, 125, 24, colors["mood"], thickness=3)
    draw_line(170, 24, 225, 24, colors["energy"], thickness=3)
    return _encode_png(width, height, bytes(pixels))


def _encode_png(width: int, height: int, rgb: bytes) -> bytes:
    raw_rows = [b"\x00" + rgb[y * width * 3 : (y + 1) * width * 3] for y in range(height)]
    raw = b"".join(raw_rows)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    return b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            chunk(b"IDAT", zlib.compress(raw, level=6)),
            chunk(b"IEND", b""),
        ]
    )


def _format_counts(counts: dict[str, int], empty: str = "немає даних") -> str:
    if not counts:
        return empty
    return "\n".join(f"- {key}: {value}" for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:8])


def _metric_entries(entries: list[Entry]) -> list[Entry]:
    excluded_sources = {"correction", "profile_context_update"}
    return [entry for entry in entries if entry.source not in excluded_sources]


def _list_block(title: str, items: list[str]) -> str:
    if not items:
        return f"{title}: немає даних"
    return title + ":\n" + "\n".join(f"- {item}" for item in items)


def _display_date(value: object) -> str:
    if hasattr(value, "date"):
        return str(value.date())
    return str(value)


def _active_window(day: date, timezone: str, settings: UserSettings) -> tuple[datetime, datetime]:
    tz = ZoneInfo(timezone)
    start_time = parse_hhmm(settings.active_start)
    end_time = parse_hhmm(settings.active_end)
    start = datetime.combine(day, start_time, tzinfo=tz)
    end = datetime.combine(day, end_time, tzinfo=tz)
    if start_time > end_time:
        end += timedelta(days=1)
    return start, end


def _entry_timestamp(entry: Entry) -> datetime | None:
    return entry.local_timestamp or entry.created_at


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
    entry: Entry, quality_by_entry: dict[str, str], labels_by_entry: dict[str, list[str]]
) -> str:
    timestamp = entry.local_timestamp or entry.created_at
    time_text = timestamp.strftime("%H:%M") if timestamp else "??:??"
    text = _truncate(entry.raw_text or "[без тексту]", 120)
    labels = labels_by_entry.get(str(entry.id), [])
    label_text = f" [{', '.join(_human_label(label) for label in labels[:3])}]" if labels else ""
    quality = quality_by_entry.get(str(entry.id))
    quality_text = f" ({_data_quality_label(quality)})" if quality else ""
    return f"{time_text} - {text}{label_text}{quality_text}"


def _truncate(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    if limit <= 1:
        return "…"[:limit]
    return compact[: limit - 1].rstrip() + "…"
