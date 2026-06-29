from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from mental_state_bot.services.archive_audit import (
    ArchiveAudit,
    archive_audit_recommendations,
    count_missing_media_files,
    format_archive_audit,
)


def test_format_archive_audit_includes_coverage_and_recommendations() -> None:
    audit = ArchiveAudit(
        days=3,
        entries=10,
        snapshots=5,
        prompts=7,
        media=2,
        missing_media_files=1,
        missed_prompts=2,
        ai_analyses=12,
        feature_analyses_missing=4,
        embeddings=3,
        embeddings_missing=7,
        summaries={"daily": 2},
        snapshot_statuses={"closed": 4, "missed": 1},
        missed_prompt_statuses={"open": 2},
        model_runs=20,
        retrieval_logs=3,
        exports=1,
    )

    text = format_archive_audit(audit)

    assert "Стан архіву" in text
    assert "- записів: 10" in text
    assert "- entries без extract_entry_features: 4" in text
    assert "- entries без embedding: 7" in text
    assert "- daily: 2" in text
    assert "features-backfill" in text
    assert "embed-backfill" in text
    assert "missing_media.json" in text


def test_archive_audit_recommendations_skip_embeddings_when_not_checked() -> None:
    audit = ArchiveAudit(
        days=0,
        entries=0,
        snapshots=0,
        prompts=0,
        media=0,
        missing_media_files=0,
        missed_prompts=0,
        ai_analyses=0,
        feature_analyses_missing=0,
        embeddings=0,
        embeddings_missing=None,
        summaries={},
        snapshot_statuses={},
        missed_prompt_statuses={},
        model_runs=0,
        retrieval_logs=0,
        exports=0,
    )

    assert archive_audit_recommendations(audit) == []
    assert "embeddings вимкнені" in format_archive_audit(audit)


def test_count_missing_media_files(tmp_path: Path) -> None:
    existing = tmp_path / "photo.jpg"
    existing.write_bytes(b"img")
    media_items = [
        SimpleNamespace(file_path=str(existing)),
        SimpleNamespace(file_path=str(tmp_path / "missing.jpg")),
        SimpleNamespace(file_path=None),
    ]

    assert count_missing_media_files(media_items) == 2
