from __future__ import annotations

from datetime import date

from mental_state_bot.services.visual_report import (
    VisualReportData,
    VisualReportDay,
    _plan_story_pages,
    _toc_pages,
    render_visual_report_pdf,
)


def test_render_visual_report_pdf_returns_pdf_bytes() -> None:
    report = VisualReportData(
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 3),
        total_entries=8,
        active_days=2,
        top_emotions=[("спокій", 3), ("тривога", 1)],
        top_states=[("залученість", 2)],
        top_activities=[("прогулянка", 2), ("музика", 1)],
        quality_counts=[("достатньо", 4), ("частково", 2)],
        days=[
            VisualReportDay(
                local_date=date(2026, 7, 1),
                entry_count=5,
                story="День почався з прогулянки, потім була робота над проєктом і спокійний вечір.",
                mood=6.5,
                energy=5.0,
                emotions=["спокій", "радість"],
                states=["залученість"],
                activities=["прогулянка", "музика"],
                emotion_intensities={"радість": 0.8, "тривога": 0.3},
                emotion_observed=True,
            ),
            VisualReportDay(
                local_date=date(2026, 7, 2),
                entry_count=3,
                story="Було менше сил, але кілька моментів допомогли повернутися до нормального стану.",
                mood=4.0,
                energy=3.0,
                emotions=["тривога", "спокій"],
                states=["втома"],
                activities=["відпочинок"],
                emotion_intensities={"тривога": 0.55},
                emotion_observed=True,
            ),
            VisualReportDay(
                local_date=date(2026, 7, 3),
                entry_count=0,
                story=None,
            ),
        ],
    )

    pdf = render_visual_report_pdf(report)

    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 10_000


def test_visual_report_plans_toc_then_day_story_pages() -> None:
    long_story = " ".join(["Ранок був повільний, але день поступово зібрався."] * 120)
    report = VisualReportData(
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 2),
        total_entries=3,
        active_days=2,
        top_emotions=[],
        top_states=[],
        top_activities=[],
        quality_counts=[],
        days=[
            VisualReportDay(local_date=date(2026, 7, 1), entry_count=2, story=long_story),
            VisualReportDay(local_date=date(2026, 7, 2), entry_count=1, story="Короткий день."),
        ],
    )

    story_pages = _plan_story_pages(report)
    toc_pages = _toc_pages(report, story_pages)

    assert len(toc_pages) == 1
    assert story_pages[0].page_number == 3
    assert story_pages[0].continuation_total > 1
    assert toc_pages[0][0][1] == 3


def test_visual_report_emotion_page_shifts_story_page_numbers() -> None:
    report = VisualReportData(
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 1),
        total_entries=1,
        active_days=1,
        top_emotions=[("сум", 1)],
        top_states=[],
        top_activities=[],
        quality_counts=[],
        days=[
            VisualReportDay(
                local_date=date(2026, 7, 1),
                entry_count=1,
                story="Є запис.",
                emotion_intensities={"сум": 0.8},
                emotion_observed=True,
            )
        ],
    )

    story_pages = _plan_story_pages(report, front_matter_pages=3)

    assert story_pages[0].page_number == 4
