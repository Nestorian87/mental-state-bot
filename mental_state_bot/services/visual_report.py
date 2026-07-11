from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from io import BytesIO

from PIL import Image, ImageDraw
from sqlalchemy.ext.asyncio import AsyncSession

from mental_state_bot.db import repositories as repo
from mental_state_bot.db.models import Day, Entry, Summary, User
from mental_state_bot.emotions import EMOTION_COLORS
from mental_state_bot.services.review import (
    _chart_font,
    _data_quality_label,
    _date_range,
    _days_by_id,
    _entry_emotion_signals,
    _entry_emotions,
    _entry_journal_date,
    _graphable_feature_score,
    _latest_feature_results_by_entry,
    _metric_entries,
    _metrics_label_text,
    _summaries_by_journal_date,
)
from mental_state_bot.time_utils import zoneinfo

MAX_VISUAL_REPORT_DAYS = 45


@dataclass(frozen=True)
class VisualReportDay:
    local_date: date
    entry_count: int
    story: str | None = None
    mood: float | None = None
    energy: float | None = None
    emotions: list[str] = field(default_factory=list)
    states: list[str] = field(default_factory=list)
    activities: list[str] = field(default_factory=list)
    emotion_intensities: dict[str, float] = field(default_factory=dict)
    emotion_observed: bool = False


@dataclass(frozen=True)
class VisualReportData:
    start_date: date
    end_date: date
    days: list[VisualReportDay]
    total_entries: int
    active_days: int
    top_emotions: list[tuple[str, int]]
    top_states: list[tuple[str, int]]
    top_activities: list[tuple[str, int]]
    quality_counts: list[tuple[str, int]]


@dataclass(frozen=True)
class _StoryPage:
    day: VisualReportDay
    page_number: int
    story_lines: list[str]
    continuation_index: int = 0
    continuation_total: int = 1


async def build_visual_report_pdf(
    session: AsyncSession,
    *,
    user: User,
    start_date: date,
    end_date: date,
) -> bytes:
    if start_date > end_date:
        raise ValueError("Початкова дата має бути раніше або дорівнювати кінцевій.")
    if (end_date - start_date).days + 1 > MAX_VISUAL_REPORT_DAYS:
        raise ValueError(f"Поки підтримую до {MAX_VISUAL_REPORT_DAYS} днів в одному PDF.")

    report = await collect_visual_report_data(
        session,
        user=user,
        start_date=start_date,
        end_date=end_date,
    )
    return render_visual_report_pdf(report)


async def collect_visual_report_data(
    session: AsyncSession,
    *,
    user: User,
    start_date: date,
    end_date: date,
) -> VisualReportData:
    timezone = zoneinfo(user.timezone)
    days = list(await repo.list_days_between(session, user_id=user.id, start_date=start_date, end_date=end_date))
    days_by_id = _days_by_id(days)
    start = datetime.combine(start_date, time.min, tzinfo=timezone).astimezone(UTC)
    end = datetime.combine(end_date + timedelta(days=1), time.max, tzinfo=timezone).astimezone(UTC)
    entries = _metric_entries(
        list(await repo.list_entries_between(session, user_id=user.id, start=start, end=end))
    )
    entries_by_date = _entries_by_journal_date(
        entries,
        days_by_id=days_by_id,
        timezone=user.timezone,
        start_date=start_date,
        end_date=end_date,
    )
    summaries = list(
        await repo.list_summaries_between(
            session,
            user_id=user.id,
            period_type="daily",
            start=start,
            end=end,
        )
    )
    summaries_by_date = _summaries_by_journal_date(summaries, days_by_id=days_by_id, timezone=user.timezone)
    analyses = await repo.list_analyses_for_targets(
        session,
        target_type="entry",
        target_ids=[entry.id for entry in entries],
    )
    features_by_entry = _latest_feature_results_by_entry(analyses)

    emotion_counts: Counter[str] = Counter()
    state_counts: Counter[str] = Counter()
    activity_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    report_days: list[VisualReportDay] = []

    for current in _date_range(start_date, end_date):
        day_entries = entries_by_date.get(current, [])
        moods: list[int] = []
        energies: list[int] = []
        day_emotions: Counter[str] = Counter()
        day_states: Counter[str] = Counter()
        day_activities: Counter[str] = Counter()
        day_emotion_values: dict[str, list[float]] = defaultdict(list)
        emotion_observed = False
        for entry in day_entries:
            result = features_by_entry.get(str(entry.id), {})
            quality_counts[_data_quality_label(result.get("data_quality"))] += 1
            mood = _graphable_feature_score(result, "mood")
            energy = _graphable_feature_score(result, "energy")
            if mood is not None:
                moods.append(mood)
            if energy is not None:
                energies.append(energy)
            for emotion in _entry_emotions(result):
                day_emotions[emotion] += 1
                emotion_counts[emotion] += 1
            signals = _entry_emotion_signals(result)
            if signals or str(result.get("emotion_observation") or "") == "no_current_emotion":
                emotion_observed = True
            for signal in signals:
                day_emotion_values[signal.label].append(signal.intensity)
            for label in result.get("state_labels") or []:
                normalized = _metrics_label_text(label)
                if normalized:
                    day_states[normalized] += 1
                    state_counts[normalized] += 1
            for label in result.get("activity_labels") or []:
                normalized = _metrics_label_text(label)
                if normalized:
                    day_activities[normalized] += 1
                    activity_counts[normalized] += 1

        summary = summaries_by_date.get(current)
        report_days.append(
            VisualReportDay(
                local_date=current,
                entry_count=len(day_entries),
                story=_summary_story(summary) or _fallback_story(day_entries),
                mood=_average(moods),
                energy=_average(energies),
                emotions=[label for label, _count in day_emotions.most_common(4)],
                states=[label for label, _count in day_states.most_common(4)],
                activities=[label for label, _count in day_activities.most_common(4)],
                emotion_intensities={
                    label: _average(values) or 0.0 for label, values in day_emotion_values.items()
                },
                emotion_observed=emotion_observed,
            )
        )

    active_days = sum(1 for item in report_days if item.entry_count)
    return VisualReportData(
        start_date=start_date,
        end_date=end_date,
        days=report_days,
        total_entries=sum(item.entry_count for item in report_days),
        active_days=active_days,
        top_emotions=emotion_counts.most_common(8),
        top_states=state_counts.most_common(8),
        top_activities=activity_counts.most_common(8),
        quality_counts=quality_counts.most_common(),
    )


def render_visual_report_pdf(report: VisualReportData) -> bytes:
    with _render_scale(2):
        return _render_visual_report_pdf(report)


def _render_visual_report_pdf(report: VisualReportData) -> bytes:
    pages: list[Image.Image] = []
    has_emotion_chart = bool(_emotion_chart_labels(report))
    story_pages = _plan_story_pages(report, front_matter_pages=2 + int(has_emotion_chart))
    first_page = _new_page()
    draw = ImageDraw.Draw(first_page)
    y = _draw_report_header(draw, report)
    y = _draw_period_chart(draw, report, y)
    _draw_overview_block(draw, report, y)
    _draw_footer(draw, 1)
    pages.append(first_page)

    if has_emotion_chart:
        page = _new_page()
        draw = ImageDraw.Draw(page)
        _draw_emotion_chart_page(draw, report)
        _draw_footer(draw, 2)
        pages.append(page)

    toc_pages = _toc_pages(report, story_pages)
    for index, toc_items in enumerate(toc_pages, start=1):
        page = _new_page()
        draw = ImageDraw.Draw(page)
        _draw_toc_page(
            draw,
            toc_items,
            page_number=index + 1 + int(has_emotion_chart),
            part=index,
            total_parts=len(toc_pages),
        )
        pages.append(page)

    for story_page in story_pages:
        page = _new_page()
        draw = ImageDraw.Draw(page)
        _draw_story_page(draw, story_page)
        pages.append(page)

    output = BytesIO()
    rgb_pages = [page.convert("RGB") for page in pages]
    rgb_pages[0].save(
        output,
        format="PDF",
        resolution=288,
        save_all=True,
        append_images=rgb_pages[1:],
    )
    return output.getvalue()


_BASE_PAGE_WIDTH = 1240
_BASE_PAGE_HEIGHT = 1754
_SCALE = 1
_PAGE_WIDTH = _BASE_PAGE_WIDTH
_PAGE_HEIGHT = _BASE_PAGE_HEIGHT
_MARGIN = 76
_BLUE = (37, 99, 235)
_GREEN = (22, 163, 74)
_TEXT = (15, 23, 42)
_MUTED = (71, 85, 105)
_LIGHT = (241, 245, 249)
_BORDER = (226, 232, 240)
_TOC_ITEMS_PER_PAGE = 18
_FIRST_STORY_LINES_PER_PAGE = 32
_CONTINUED_STORY_LINES_PER_PAGE = 40


@contextmanager
def _render_scale(scale: int):
    global _SCALE, _PAGE_WIDTH, _PAGE_HEIGHT, _MARGIN
    previous = (_SCALE, _PAGE_WIDTH, _PAGE_HEIGHT, _MARGIN)
    _SCALE = scale
    _PAGE_WIDTH = _BASE_PAGE_WIDTH * scale
    _PAGE_HEIGHT = _BASE_PAGE_HEIGHT * scale
    _MARGIN = 76 * scale
    try:
        yield
    finally:
        _SCALE, _PAGE_WIDTH, _PAGE_HEIGHT, _MARGIN = previous


def _font(size: int, *, bold: bool = False):
    return _chart_font(size * _SCALE, bold=bold)


def _u(value: int | float) -> int:
    return int(round(value * _SCALE))


def _new_page() -> Image.Image:
    return Image.new("RGB", (_PAGE_WIDTH, _PAGE_HEIGHT), (248, 250, 252))


def _draw_report_header(draw: ImageDraw.ImageDraw, report: VisualReportData) -> int:
    title_font = _font(48, bold=True)
    text_font = _font(28)
    small_font = _font(23)
    title = "Візуальний журнал"
    period = f"{report.start_date.isoformat()} - {report.end_date.isoformat()}"
    total_days = len(report.days)
    meta = (
        f"{report.active_days}/{total_days} днів із записами · "
        f"{report.total_entries} записів · PDF для перегляду з телефону"
    )
    draw.text((_MARGIN, _u(64)), title, fill=_TEXT, font=title_font)
    draw.text((_MARGIN, _u(126)), period, fill=_MUTED, font=text_font)
    draw.text((_MARGIN, _u(168)), meta, fill=(100, 116, 139), font=small_font)
    return _u(230)


def _draw_period_chart(draw: ImageDraw.ImageDraw, report: VisualReportData, y: int) -> int:
    x0 = _MARGIN
    x1 = _PAGE_WIDTH - _MARGIN
    height = _u(500)
    y0 = y
    y1 = y0 + height
    draw.rounded_rectangle((x0, y0, x1, y1), radius=_u(28), fill=(255, 255, 255), outline=_BORDER, width=_u(3))
    title_font = _font(30, bold=True)
    small_font = _font(21)
    draw.text((x0 + _u(34), y0 + _u(28)), "Стан по днях", fill=_TEXT, font=title_font)
    _legend(draw, x0 + _u(34), y0 + _u(76), "настрій", _BLUE)
    _legend(draw, x0 + _u(185), y0 + _u(76), "енергія", _GREEN)

    plot_left = x0 + _u(94)
    plot_right = x1 - _u(58)
    plot_top = y0 + _u(140)
    plot_bottom = y1 - _u(132)
    for score in range(0, 11, 2):
        yy = _score_y(score, plot_top, plot_bottom)
        draw.line((plot_left, yy, plot_right, yy), fill=_LIGHT, width=_u(2))
        draw.text((x0 + _u(36), yy - _u(12)), str(score), fill=(148, 163, 184), font=small_font)

    labels = [item.local_date.strftime("%m-%d") for item in report.days]
    x_positions = _x_positions(len(report.days), plot_left, plot_right)
    for index in _label_indexes(len(labels)):
        x = x_positions[index]
        _draw_centered_text(draw, (x, plot_bottom + _u(28)), labels[index], small_font, fill=(100, 116, 139))

    _draw_series(draw, [item.mood for item in report.days], x_positions, plot_top, plot_bottom, _BLUE)
    _draw_series(draw, [item.energy for item in report.days], x_positions, plot_top, plot_bottom, _GREEN)
    coverage = (
        f"Точок: настрій {_known_count([item.mood for item in report.days])}/{len(report.days)}, "
        f"енергія {_known_count([item.energy for item in report.days])}/{len(report.days)}."
    )
    draw.text((x0 + _u(34), y1 - _u(52)), coverage, fill=_MUTED, font=small_font)
    return y1 + _u(28)


def _draw_overview_block(draw: ImageDraw.ImageDraw, report: VisualReportData, y: int) -> int:
    x0 = _MARGIN
    x1 = _PAGE_WIDTH - _MARGIN
    y0 = y
    title_font = _font(28, bold=True)
    text_font = _font(24)
    small_font = _font(21)
    columns = [
        ("Емоції", report.top_emotions),
        ("Стани", report.top_states),
        ("Активності", report.top_activities),
        ("Якість даних", report.quality_counts),
    ]
    col_gap = _u(18)
    col_width = int((x1 - x0 - col_gap * 3) / 4)
    height = _u(380)
    for index, (title, items) in enumerate(columns):
        left = x0 + index * (col_width + col_gap)
        right = left + col_width
        draw.rounded_rectangle((left, y0, right, y0 + height), radius=_u(24), fill=(255, 255, 255), outline=_BORDER, width=_u(2))
        draw.text((left + _u(24), y0 + _u(22)), title, fill=_TEXT, font=title_font)
        if not items:
            draw.text((left + _u(24), y0 + _u(72)), "немає даних", fill=_MUTED, font=text_font)
            continue
        yy = y0 + _u(76)
        bottom = y0 + height - _u(24)
        for label, count in items[:6]:
            text = f"{label}: {count}"
            if yy + _u(30) > bottom:
                break
            for line in _wrap_text(draw, text, small_font, col_width - _u(48), max_lines=2):
                if yy + _u(30) > bottom:
                    break
                draw.text((left + _u(24), yy), line, fill=_MUTED, font=small_font)
                yy += _u(30)
            yy += _u(6)
    return y0 + height + _u(40)


def _draw_emotion_chart_page(draw: ImageDraw.ImageDraw, report: VisualReportData) -> None:
    x0 = _MARGIN
    x1 = _PAGE_WIDTH - _MARGIN
    title_font = _font(42, bold=True)
    text_font = _font(24)
    small_font = _font(20)
    draw.text((x0, _u(64)), "Емоційна динаміка", fill=_TEXT, font=title_font)
    draw.text(
        (x0, _u(122)),
        f"{report.start_date.isoformat()} - {report.end_date.isoformat()}",
        fill=_MUTED,
        font=text_font,
    )
    labels = _emotion_chart_labels(report)
    legend_y = _u(178)
    legend_x = x0
    for label in labels:
        color = EMOTION_COLORS.get(label, _MUTED)
        width = _u(54) + int(draw.textlength(label, font=small_font))
        if legend_x + width > x1:
            legend_x = x0
            legend_y += _u(38)
        _legend(draw, legend_x, legend_y, label, color)
        legend_x += width + _u(30)

    chart_top = legend_y + _u(64)
    chart_bottom = _PAGE_HEIGHT - _u(180)
    draw.rounded_rectangle(
        (x0, chart_top, x1, chart_bottom),
        radius=_u(28),
        fill=(255, 255, 255),
        outline=_BORDER,
        width=_u(2),
    )
    plot_left = x0 + _u(102)
    plot_right = x1 - _u(54)
    plot_top = chart_top + _u(66)
    plot_bottom = chart_bottom - _u(110)
    for value, label in ((0.0, "0"), (0.3, "слабко"), (0.55, "помірно"), (0.8, "сильно"), (1.0, "дуже")):
        y = _intensity_y(value, plot_top, plot_bottom)
        draw.line((plot_left, y, plot_right, y), fill=_LIGHT, width=_u(2))
        _draw_right_aligned_text(draw, (plot_left - _u(18), y - _u(11)), label, small_font, fill=(100, 116, 139))

    x_positions = _x_positions(len(report.days), plot_left, plot_right)
    for index in _label_indexes(len(report.days)):
        _draw_centered_text(
            draw,
            (x_positions[index], plot_bottom + _u(28)),
            report.days[index].local_date.strftime("%m-%d"),
            small_font,
            fill=(100, 116, 139),
        )
    for label in labels:
        values = [
            day.emotion_intensities.get(label, 0.0) if day.emotion_observed else None
            for day in report.days
        ]
        _draw_intensity_series(
            draw,
            values,
            x_positions,
            plot_top,
            plot_bottom,
            EMOTION_COLORS.get(label, _MUTED),
        )
    observed = sum(1 for day in report.days if day.emotion_observed)
    draw.text(
        (x0 + _u(34), chart_bottom - _u(52)),
        f"Днів із явними емоційними сигналами: {observed}/{len(report.days)}.",
        fill=_MUTED,
        font=small_font,
    )


def _plan_story_pages(report: VisualReportData, *, front_matter_pages: int = 2) -> list[_StoryPage]:
    dummy = ImageDraw.Draw(_new_page())
    text_font = _font(24)
    max_width = _PAGE_WIDTH - _MARGIN * 2 - _u(60)
    toc_page_count = max(1, (len(report.days) + _TOC_ITEMS_PER_PAGE - 1) // _TOC_ITEMS_PER_PAGE)
    page_number = front_matter_pages + toc_page_count
    pages: list[_StoryPage] = []
    for day in report.days:
        story = day.story or "Немає текстового підсумку або записів за цей день."
        story_lines = _wrap_text(dummy, story, text_font, max_width)
        first_chunk = story_lines[:_FIRST_STORY_LINES_PER_PAGE]
        rest = story_lines[_FIRST_STORY_LINES_PER_PAGE:]
        chunks = [first_chunk]
        while rest:
            chunks.append(rest[:_CONTINUED_STORY_LINES_PER_PAGE])
            rest = rest[_CONTINUED_STORY_LINES_PER_PAGE:]
        for index, chunk in enumerate(chunks):
            pages.append(
                _StoryPage(
                    day=day,
                    page_number=page_number,
                    story_lines=chunk,
                    continuation_index=index,
                    continuation_total=len(chunks),
                )
            )
            page_number += 1
    return pages


def _toc_pages(report: VisualReportData, story_pages: list[_StoryPage]) -> list[list[tuple[VisualReportDay, int]]]:
    first_page_by_date = {
        story_page.day.local_date: story_page.page_number
        for story_page in story_pages
        if story_page.continuation_index == 0
    }
    items = [(day, first_page_by_date.get(day.local_date, 0)) for day in report.days]
    if not items:
        return [[]]
    return [items[index : index + _TOC_ITEMS_PER_PAGE] for index in range(0, len(items), _TOC_ITEMS_PER_PAGE)]


def _draw_toc_page(
    draw: ImageDraw.ImageDraw,
    toc_items: list[tuple[VisualReportDay, int]],
    *,
    page_number: int,
    part: int,
    total_parts: int,
) -> None:
    title_font = _font(42, bold=True)
    text_font = _font(24)
    small_font = _font(20)
    draw.text((_MARGIN, _u(68)), "Зміст", fill=_TEXT, font=title_font)
    subtitle = "Історії днів"
    if total_parts > 1:
        subtitle += f" · частина {part}/{total_parts}"
    draw.text((_MARGIN, _u(126)), subtitle, fill=_MUTED, font=text_font)
    y = _u(190)
    for day, target_page in toc_items:
        y = _draw_toc_item(draw, day, target_page, y)
    draw.text(
        (_MARGIN, _PAGE_HEIGHT - _u(96)),
        "Номери сторінок ведуть до секцій із повнішою історією кожного журнального дня.",
        fill=(100, 116, 139),
        font=small_font,
    )
    _draw_footer(draw, page_number)


def _draw_toc_item(
    draw: ImageDraw.ImageDraw,
    day: VisualReportDay,
    target_page: int,
    y: int,
) -> int:
    x0 = _MARGIN
    x1 = _PAGE_WIDTH - _MARGIN
    row_height = _u(72)
    title_font = _font(23, bold=True)
    small_font = _font(19)
    draw.rounded_rectangle((x0, y, x1, y + row_height), radius=_u(18), fill=(255, 255, 255), outline=_BORDER, width=_u(2))
    draw.text((x0 + _u(22), y + _u(18)), day.local_date.isoformat(), fill=_TEXT, font=title_font)
    draw.text((x0 + _u(190), y + _u(20)), f"{day.entry_count} записів", fill=_MUTED, font=small_font)
    markers = _toc_markers(day)
    if markers:
        marker_left = x0 + _u(320)
        marker_right = x1 - _u(74)
        draw.text(
            (marker_left, y + _u(20)),
            _fit_text(draw, markers, small_font, marker_right - marker_left),
            fill=_MUTED,
            font=small_font,
        )
    page_text = str(target_page)
    page_width = draw.textlength(page_text, font=title_font)
    draw.text((x1 - _u(24) - page_width, y + _u(18)), page_text, fill=_BLUE, font=title_font)
    return y + row_height + _u(14)


def _draw_story_page(draw: ImageDraw.ImageDraw, story_page: _StoryPage) -> None:
    day = story_page.day
    x0 = _MARGIN
    x1 = _PAGE_WIDTH - _MARGIN
    title_font = _font(38, bold=True)
    text_font = _font(24)
    small_font = _font(20)
    continuation = ""
    if story_page.continuation_total > 1:
        continuation = f" · продовження {story_page.continuation_index + 1}/{story_page.continuation_total}"
    title = _fit_text(draw, f"{day.local_date.isoformat()}{continuation}", title_font, x1 - x0)
    draw.text((x0, _u(64)), title, fill=_TEXT, font=title_font)
    draw.text((x0, _u(118)), f"{day.entry_count} записів", fill=_MUTED, font=text_font)
    chips = _day_chips(day)
    _draw_chips(draw, chips, x0, _u(162), max_width=x1 - x0)
    y = _u(230)
    if story_page.continuation_index == 0:
        details = _day_details(day)
        if details:
            for line in _wrap_text(draw, details, small_font, x1 - x0, max_lines=3):
                draw.text((x0, y), line, fill=_MUTED, font=small_font)
                y += _u(29)
            y += _u(14)
    draw.rounded_rectangle((x0, y, x1, _PAGE_HEIGHT - _u(112)), radius=_u(28), fill=(255, 255, 255), outline=_BORDER, width=_u(2))
    y += _u(34)
    for line in story_page.story_lines:
        draw.text((x0 + _u(32), y), line, fill=_TEXT, font=text_font)
        y += _u(34)
    _draw_footer(draw, story_page.page_number)


def _toc_markers(day: VisualReportDay) -> str:
    markers = []
    if day.emotions:
        markers.append(", ".join(day.emotions[:2]))
    if day.activities:
        markers.append(", ".join(day.activities[:2]))
    return " · ".join(markers)


def _day_chips(day: VisualReportDay) -> list[str]:
    chips = []
    if day.mood is not None:
        chips.append(f"настрій {day.mood:.1f}/10")
    if day.energy is not None:
        chips.append(f"енергія {day.energy:.1f}/10")
    chips.extend(day.emotions[:4])
    return chips


def _draw_footer(draw: ImageDraw.ImageDraw, page_number: int) -> None:
    font = _font(19)
    draw.text((_PAGE_WIDTH - _MARGIN - _u(120), _PAGE_HEIGHT - _u(54)), f"стор. {page_number}", fill=(148, 163, 184), font=font)


def _legend(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, color: tuple[int, int, int]) -> None:
    font = _font(22)
    draw.line((x, y + _u(13), x + _u(44), y + _u(13)), fill=color, width=_u(7))
    draw.text((x + _u(56), y), label, fill=_MUTED, font=font)


def _draw_series(
    draw: ImageDraw.ImageDraw,
    values: list[float | None],
    x_positions: list[int],
    plot_top: int,
    plot_bottom: int,
    color: tuple[int, int, int],
) -> None:
    previous: tuple[int, int] | None = None
    previous_index: int | None = None
    for index, value in enumerate(values):
        if value is None:
            continue
        current = (x_positions[index], _score_y(value, plot_top, plot_bottom))
        if previous is not None:
            width = _u(5)
            if previous_index is not None and index - previous_index > 1:
                _draw_dashed_line(draw, previous, current, fill=color, width=width)
            else:
                draw.line((previous, current), fill=color, width=width)
        radius = _u(9)
        draw.ellipse(
            (current[0] - radius, current[1] - radius, current[0] + radius, current[1] + radius),
            fill=color,
            outline=(255, 255, 255),
            width=_u(3),
        )
        previous = current
        previous_index = index


def _draw_intensity_series(
    draw: ImageDraw.ImageDraw,
    values: list[float | None],
    x_positions: list[int],
    plot_top: int,
    plot_bottom: int,
    color: tuple[int, int, int],
) -> None:
    previous: tuple[int, int] | None = None
    previous_index: int | None = None
    for index, value in enumerate(values):
        if value is None:
            continue
        current = (x_positions[index], _intensity_y(value, plot_top, plot_bottom))
        if previous is not None:
            if previous_index is not None and index - previous_index > 1:
                _draw_dashed_line(draw, previous, current, fill=color, width=_u(4))
            else:
                draw.line((previous, current), fill=color, width=_u(4))
        radius = _u(6) + int(max(0.0, min(1.0, value)) * _u(7))
        draw.ellipse(
            (current[0] - radius, current[1] - radius, current[0] + radius, current[1] + radius),
            fill=color,
            outline=(255, 255, 255),
            width=_u(3),
        )
        previous = current
        previous_index = index


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    fill: tuple[int, int, int],
    width: int,
) -> None:
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    distance = (dx * dx + dy * dy) ** 0.5
    if distance <= 0:
        return
    dash = _u(18)
    gap = _u(12)
    step = dash + gap
    travelled = 0.0
    while travelled < distance:
        segment_end = min(travelled + dash, distance)
        sx = x1 + dx * (travelled / distance)
        sy = y1 + dy * (travelled / distance)
        ex = x1 + dx * (segment_end / distance)
        ey = y1 + dy * (segment_end / distance)
        draw.line((sx, sy, ex, ey), fill=fill, width=width)
        travelled += step


def _draw_chips(draw: ImageDraw.ImageDraw, labels: list[str], x: int, y: int, *, max_width: int) -> None:
    font = _font(19)
    current_x = x
    current_y = y
    for label in labels[:6]:
        label = _fit_text(draw, label, font, max(_u(80), max_width - _u(30)))
        width = int(draw.textlength(label, font=font)) + _u(30)
        if current_x + width > x + max_width:
            current_x = x
            current_y += _u(38)
        if current_x + width > x + max_width:
            width = max_width
            label = _fit_text(draw, label, font, max_width - _u(30))
        draw.rounded_rectangle((current_x, current_y, current_x + width, current_y + _u(28)), radius=_u(14), fill=_LIGHT)
        draw.text((current_x + _u(15), current_y + _u(3)), label, fill=_MUTED, font=font)
        current_x += width + _u(10)


def _score_y(score: float, plot_top: int, plot_bottom: int) -> int:
    clamped = max(0, min(10, float(score)))
    return int(plot_bottom - (clamped / 10) * (plot_bottom - plot_top))


def _intensity_y(value: float, plot_top: int, plot_bottom: int) -> int:
    clamped = max(0, min(1, float(value)))
    return int(plot_bottom - clamped * (plot_bottom - plot_top))


def _x_positions(length: int, left: int, right: int) -> list[int]:
    if length <= 1:
        return [(left + right) // 2] if length else []
    return [left + int(index / (length - 1) * (right - left)) for index in range(length)]


def _label_indexes(length: int) -> list[int]:
    if length <= 1:
        return [0] if length else []
    if length <= 8:
        return list(range(length))
    indexes = {0, length - 1, length // 2, length // 4, (length * 3) // 4}
    return sorted(indexes)


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    text: str,
    font,
    *,
    fill: tuple[int, int, int],
) -> None:
    width = draw.textlength(text, font=font)
    draw.text((center[0] - width / 2, center[1]), text, fill=fill, font=font)


def _draw_right_aligned_text(
    draw: ImageDraw.ImageDraw,
    anchor: tuple[int, int],
    text: str,
    font,
    *,
    fill: tuple[int, int, int],
) -> None:
    draw.text((anchor[0] - draw.textlength(text, font=font), anchor[1]), text, fill=fill, font=font)


def _fit_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> str:
    normalized = " ".join(str(text or "").split())
    if draw.textlength(normalized, font=font) <= max_width:
        return normalized
    ellipsis = "..."
    result = normalized
    while result and draw.textlength(result + ellipsis, font=font) > max_width:
        result = result[:-1]
    return (result.rstrip() + ellipsis) if result else ellipsis


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font,
    max_width: int,
    *,
    max_lines: int | None = None,
) -> list[str]:
    words = []
    for raw_word in " ".join(str(text or "").split()).split():
        words.extend(_split_word_to_width(draw, raw_word, font, max_width))
    if not words:
        return []
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
        if max_lines is not None and len(lines) >= max_lines:
            break
    if current and (max_lines is None or len(lines) < max_lines):
        lines.append(current)
    if max_lines is not None and len(lines) == max_lines and len(words) > len(" ".join(lines).split()):
        lines[-1] = lines[-1].rstrip(".,;:") + "..."
    return lines


def _split_word_to_width(draw: ImageDraw.ImageDraw, word: str, font, max_width: int) -> list[str]:
    if draw.textlength(word, font=font) <= max_width:
        return [word]
    parts = []
    current = ""
    for char in word:
        candidate = current + char
        if current and draw.textlength(candidate, font=font) > max_width:
            parts.append(current)
            current = char
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def _entries_by_journal_date(
    entries: list[Entry],
    *,
    days_by_id: dict[str, Day],
    timezone: str,
    start_date: date,
    end_date: date,
) -> dict[date, list[Entry]]:
    grouped: dict[date, list[Entry]] = defaultdict(list)
    for entry in entries:
        entry_date = _entry_journal_date(entry, days_by_id=days_by_id, timezone=timezone)
        if entry_date is not None and start_date <= entry_date <= end_date:
            grouped[entry_date].append(entry)
    return grouped


def _summary_story(summary: Summary | None) -> str | None:
    if summary is None:
        return None
    details = summary.details or {}
    for key in ("story", "period_story", "narrative", "day_story"):
        value = details.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    what_happened = details.get("what_happened")
    state_changes = details.get("state_changes")
    parts = []
    if isinstance(what_happened, list) and what_happened:
        parts.append("Що відбувалося: " + ", ".join(str(item) for item in what_happened[:8]))
    if isinstance(state_changes, list) and state_changes:
        parts.append("Зміни стану: " + ", ".join(str(item) for item in state_changes[:8]))
    if parts:
        return ". ".join(parts)
    return summary.short_text.strip() if summary.short_text else None


def _fallback_story(entries: list[Entry]) -> str | None:
    texts = [entry.raw_text for entry in entries if (entry.raw_text or "").strip()]
    if not texts:
        return None
    if len(texts) == 1:
        return texts[0]
    first = texts[0]
    last = texts[-1]
    return f"Перший запис: {first} Останній запис: {last}"


def _day_details(day: VisualReportDay) -> str:
    parts = []
    if day.activities:
        parts.append("активності: " + ", ".join(day.activities[:4]))
    if day.states:
        parts.append("стани: " + ", ".join(day.states[:4]))
    return " · ".join(parts)


def _average(values: list[int]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 1)


def _known_count(values: list[float | None]) -> int:
    return sum(1 for value in values if value is not None)


def _emotion_chart_labels(report: VisualReportData) -> list[str]:
    counts: Counter[str] = Counter()
    for day in report.days:
        for label in day.emotion_intensities:
            counts[label] += 1
    return [label for label, _count in counts.most_common(6)]
