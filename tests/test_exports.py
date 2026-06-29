from __future__ import annotations

import csv
import json
import zipfile
from datetime import date, datetime
from decimal import Decimal
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from mental_state_bot.services.exports import (
    EXPORT_SCHEMA_VERSION,
    _embedding_table,
    _json_default,
    _model_usage_table,
    build_archive_payload,
    render_archive_markdown,
    render_metrics_csv,
    write_archive_bundle,
)


def test_json_default_serializes_common_archive_types() -> None:
    item_id = uuid4()

    assert _json_default(item_id) == str(item_id)
    assert _json_default(date(2026, 6, 29)) == "2026-06-29"
    assert _json_default(Decimal("0.42")) == "0.42"


def test_build_archive_payload_contains_full_sections() -> None:
    user_id = uuid4()
    now = datetime(2026, 6, 29, 12, 0)
    user = SimpleNamespace(
        id=user_id,
        telegram_user_id=123,
        chat_id=456,
        username="me",
        first_name="Me",
        timezone="Europe/Kyiv",
        first_seen_at=now,
        last_seen_at=now,
        is_active=True,
    )
    user_settings = SimpleNamespace(
        id=uuid4(),
        tone="calm",
        humanity_level="balanced",
        active_start="09:00",
        active_end="23:30",
        min_interval_minutes=30,
        max_interval_minutes=70,
        reminder_delay_minutes=25,
        max_clarifications=2,
        ask_body_signals=True,
        photo_prompts_enabled=True,
        settings_json={},
        created_at=now,
        updated_at=now,
    )
    entry = SimpleNamespace(
        id=uuid4(),
        day_id=None,
        snapshot_id=None,
        source="manual",
        raw_text="лежу",
        normalized_text="лежу",
        telegram_message_id=1,
        reply_to_message_id=None,
        local_timestamp=now,
        created_at=now,
        meta={},
    )
    model_run = SimpleNamespace(
        id=uuid4(),
        provider="deepseek",
        model="deepseek-v4-flash",
        task_name="extract_entry_features",
        status="success",
        prompt_tokens=10,
        completion_tokens=20,
        reasoning_tokens=0,
        total_tokens=30,
        estimated_cost_usd=Decimal("0.00001"),
        latency_ms=100,
        error_message=None,
        request_hash="abc",
        meta={},
        created_at=now,
    )
    embedding = SimpleNamespace(
        id=uuid4(),
        target_type="entry",
        target_id=entry.id,
        provider="openai-compatible",
        model="text-embedding-3-small",
        dimensions=1536,
        source_hash="hash",
        source_text="Raw: лежу",
        created_at=now,
    )

    payload = build_archive_payload(
        user=user,
        user_settings=user_settings,
        days=[],
        snapshots=[],
        prompts=[],
        entries=[entry],
        media=[],
        missed_prompts=[],
        analyses=[],
        summaries=[],
        model_runs=[model_run],
        embeddings=[embedding],
        retrieval_logs=[],
        export_jobs=[],
    )

    assert payload["schema_version"] == EXPORT_SCHEMA_VERSION
    assert payload["user_settings"]["active_start"] == "09:00"
    assert payload["entries"][0]["raw_text"] == "лежу"
    assert payload["model_runs"][0]["task_name"] == "extract_entry_features"
    assert payload["embeddings"][0]["vector_exported"] is False


def test_render_archive_markdown_contains_days_entries_and_usage() -> None:
    payload = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "exported_at": "2026-06-29T12:00:00+00:00",
        "user": {"telegram_user_id": 123, "timezone": "Europe/Kyiv"},
        "days": [{"local_date": "2026-06-29"}],
        "entries": [
            {
                "local_timestamp": "2026-06-29 10:30:00+03:00",
                "created_at": "2026-06-29 07:30:00+00:00",
                "source": "manual",
                "raw_text": "лежу\nі не можу почати",
            }
        ],
        "media": [],
        "summaries": [
            {
                "period_type": "daily",
                "period_start": "2026-06-29 00:00:00+03:00",
                "period_end": "2026-06-29 23:59:59+03:00",
                "short_text": "День був повільний.",
            }
        ],
        "ai_analyses": [],
        "model_runs": [
            {
                "task_name": "extract_entry_features",
                "model": "deepseek-v4-flash",
                "status": "success",
                "total_tokens": 42,
                "estimated_cost_usd": "0.00001",
            }
        ],
        "embeddings": [
            {
                "target_type": "entry",
                "target_id": "abc",
                "model": "text-embedding-3-small",
                "dimensions": 1536,
                "source_hash": "hash",
            }
        ],
    }

    markdown = render_archive_markdown(payload)

    assert "# Mental State Bot Archive" in markdown
    assert "### 2026-06-29" in markdown
    assert "День був повільний." in markdown
    assert "`10:30` `manual` лежу і не можу почати" in markdown
    assert "| extract_entry_features | deepseek-v4-flash | success | 42 | 0.00001 |" in markdown
    assert "| entry:abc | text-embedding-3-small | 1536 | hash |" in markdown


def test_render_metrics_csv_flattens_entry_features() -> None:
    payload = {
        "entries": [
            {
                "id": "entry-1",
                "day_id": "day-1",
                "local_timestamp": "2026-06-29 10:30:00+03:00",
                "created_at": "2026-06-29 07:30:00+00:00",
                "source": "manual",
                "raw_text": "лежу, але трохи легше",
            }
        ],
        "ai_analyses": [
            {
                "target_type": "entry",
                "target_id": "entry-1",
                "task_name": "extract_entry_features",
                "result": {
                    "activity_labels": ["лежання"],
                    "state_labels": ["виснаження", "трохи легше"],
                    "mood": {"value": "low", "confidence": 0.7},
                    "energy": {"value": "very_low", "confidence": 0.8},
                    "anxiety": {"value": "unclear", "confidence": 0.2},
                    "social_activity": {"value": "none", "confidence": 0.9},
                    "emptiness": {"present": True, "confidence": 0.6},
                    "avoidance": {"present": False, "confidence": 0.5},
                    "rumination": {"present": None, "confidence": 0.1},
                    "inability_to_start": {"present": True, "confidence": 0.75},
                    "body_signals": ["важкість"],
                    "pleasant_moments": ["трохи легше"],
                    "what_helped": ["лежати без тиску"],
                    "what_worsened": [],
                    "transition": "стан трохи пом'якшав",
                    "data_quality": "partial",
                    "confidence": 0.66,
                    "uncertainty_notes": ["мало контексту"],
                },
            }
        ],
    }

    rows = list(csv.DictReader(StringIO(render_metrics_csv(payload))))

    assert rows == [
        {
            "entry_id": "entry-1",
            "day_id": "day-1",
            "local_date": "2026-06-29",
            "local_time": "10:30",
            "local_timestamp": "2026-06-29 10:30:00+03:00",
            "created_at": "2026-06-29 07:30:00+00:00",
            "source": "manual",
            "raw_text": "лежу, але трохи легше",
            "activity_labels": "лежання",
            "state_labels": "виснаження; трохи легше",
            "mood": "low",
            "mood_confidence": "0.7",
            "energy": "very_low",
            "energy_confidence": "0.8",
            "anxiety": "unclear",
            "anxiety_confidence": "0.2",
            "social_activity": "none",
            "social_activity_confidence": "0.9",
            "emptiness_present": "true",
            "emptiness_confidence": "0.6",
            "avoidance_present": "false",
            "avoidance_confidence": "0.5",
            "rumination_present": "",
            "rumination_confidence": "0.1",
            "inability_to_start_present": "true",
            "inability_to_start_confidence": "0.75",
            "body_signals": "важкість",
            "pleasant_moments": "трохи легше",
            "pleasant_moments_count": "1",
            "what_helped": "лежати без тиску",
            "what_worsened": "",
            "transition": "стан трохи пом'якшав",
            "data_quality": "partial",
            "feature_confidence": "0.66",
            "uncertainty_notes": "мало контексту",
        }
    ]


def test_write_archive_bundle_includes_data_and_available_media(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    photo = media_root / "moment.jpg"
    photo.write_bytes(b"fake-jpeg")
    output = tmp_path / "archive.zip"
    payload = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "exported_at": "2026-06-29T12:00:00+00:00",
        "user": {"telegram_user_id": 123, "timezone": "Europe/Kyiv"},
        "days": [],
        "entries": [],
        "media": [
            {
                "id": "media-1",
                "entry_id": "entry-1",
                "media_type": "photo",
                "file_path": str(photo),
            },
            {
                "id": "media-2",
                "entry_id": "entry-2",
                "media_type": "photo",
                "file_path": str(media_root / "missing.jpg"),
            },
        ],
        "summaries": [],
        "ai_analyses": [],
        "model_runs": [],
        "embeddings": [],
    }

    manifest = write_archive_bundle(payload, output, media_root=media_root)

    assert output.exists()
    assert manifest["missing_media"][0]["media_id"] == "media-2"
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert "archive.json" in names
        assert "archive.md" in names
        assert "metrics.csv" in names
        assert "export_manifest.json" in names
        assert "missing_media.json" in names
        assert "media/entry-1/media-1-moment.jpg" in names
        assert archive.read("media/entry-1/media-1-moment.jpg") == b"fake-jpeg"
        archived_manifest = json.loads(archive.read("export_manifest.json"))

    assert archived_manifest["missing_media"][0]["reason"] == "file_not_found"
    assert any(item["path"] == "media/entry-1/media-1-moment.jpg" for item in archived_manifest["files"])


def test_model_usage_table_empty() -> None:
    assert _model_usage_table([]) == "No model runs exported."


def test_embedding_table_empty() -> None:
    assert _embedding_table([]) == "No embedding metadata exported."
