from __future__ import annotations

import csv
import json
import zipfile
from datetime import UTC, date, datetime
from decimal import Decimal
from io import StringIO
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import select

from mental_state_bot.config import Settings
from mental_state_bot.db.models import (
    AIAnalysis,
    Day,
    EmbeddingRecord,
    Entry,
    ExportJob,
    Media,
    MissedPrompt,
    ModelRun,
    RetrievalLog,
    Snapshot,
    SnapshotPrompt,
    Summary,
    UserSettings,
)
from mental_state_bot.db.repositories import add_export_job, get_user_by_telegram_id
from mental_state_bot.db.session import async_session_factory, create_async_engine_from_settings

EXPORT_SCHEMA_VERSION = "archive.v2"


async def export_user_archive(
    settings: Settings,
    telegram_user_id: int,
    output: Path,
    *,
    format: str | None = None,
) -> None:
    export_format = (format or output.suffix.lstrip(".") or "json").lower()
    if export_format == "md":
        export_format = "markdown"
    if export_format not in {"json", "markdown", "csv", "zip"}:
        raise ValueError(f"Unsupported export format: {export_format}")

    engine = create_async_engine_from_settings(settings)
    sessionmaker = async_session_factory(engine)
    output.parent.mkdir(parents=True, exist_ok=True)
    async with sessionmaker() as session, session.begin():
        user = await get_user_by_telegram_id(session, telegram_user_id)
        if user is None:
            raise ValueError(f"Unknown Telegram user id: {telegram_user_id}")

        days = (await session.execute(_user_query(Day, user.id).order_by(Day.local_date))).scalars().all()
        snapshots = (
            await session.execute(_user_query(Snapshot, user.id).order_by(Snapshot.created_at))
        ).scalars().all()
        snapshot_ids = [snapshot.id for snapshot in snapshots]
        prompts = (
            await session.execute(
                select(SnapshotPrompt)
                .where(SnapshotPrompt.snapshot_id.in_(snapshot_ids) if snapshot_ids else False)
                .order_by(SnapshotPrompt.sent_at)
            )
        ).scalars().all()
        entries = (await session.execute(_user_query(Entry, user.id).order_by(Entry.created_at))).scalars().all()
        media = (await session.execute(_user_query(Media, user.id).order_by(Media.created_at))).scalars().all()
        missed_prompts = (
            await session.execute(_user_query(MissedPrompt, user.id).order_by(MissedPrompt.created_at))
        ).scalars().all()
        analyses = (
            await session.execute(_user_query(AIAnalysis, user.id).order_by(AIAnalysis.created_at))
        ).scalars().all()
        summaries = (
            await session.execute(_user_query(Summary, user.id).order_by(Summary.period_start))
        ).scalars().all()
        model_runs = (
            await session.execute(_user_query(ModelRun, user.id).order_by(ModelRun.created_at))
        ).scalars().all()
        embeddings = (
            await session.execute(_user_query(EmbeddingRecord, user.id).order_by(EmbeddingRecord.created_at))
        ).scalars().all()
        retrieval_logs = (
            await session.execute(_user_query(RetrievalLog, user.id).order_by(RetrievalLog.created_at))
        ).scalars().all()
        export_jobs = (
            await session.execute(_user_query(ExportJob, user.id).order_by(ExportJob.created_at))
        ).scalars().all()
        user_settings = (
            await session.execute(select(UserSettings).where(UserSettings.user_id == user.id))
        ).scalar_one_or_none()

        payload = build_archive_payload(
            user=user,
            user_settings=user_settings,
            days=days,
            snapshots=snapshots,
            prompts=prompts,
            entries=entries,
            media=media,
            missed_prompts=missed_prompts,
            analyses=analyses,
            summaries=summaries,
            model_runs=model_runs,
            embeddings=embeddings,
            retrieval_logs=retrieval_logs,
            export_jobs=export_jobs,
        )
        if export_format == "markdown":
            output.write_text(render_archive_markdown(payload), encoding="utf-8")
        elif export_format == "csv":
            output.write_text(render_metrics_csv(payload), encoding="utf-8")
        elif export_format == "zip":
            bundle_manifest = write_archive_bundle(payload, output, media_root=settings.media_root)
        else:
            output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        await add_export_job(
            session,
            user_id=user.id,
            status="complete",
            format=export_format,
            file_path=str(output),
            meta={
                "schema_version": EXPORT_SCHEMA_VERSION,
                "entries": len(entries),
                "analyses": len(analyses),
                "summaries": len(summaries),
                "model_runs": len(model_runs),
                "embeddings": len(embeddings),
                "bundle": bundle_manifest if export_format == "zip" else None,
            },
        )
    await engine.dispose()


def render_archive_markdown(payload: dict[str, Any]) -> str:
    user = payload["user"]
    entries_by_day: dict[str, list[dict[str, Any]]] = {}
    for entry in payload.get("entries", []):
        day_key = _entry_day_key(entry)
        entries_by_day.setdefault(day_key, []).append(entry)

    daily_summaries = [
        summary for summary in payload.get("summaries", []) if summary.get("period_type") == "daily"
    ]
    daily_summaries_by_day = {
        str(summary.get("period_start", ""))[:10]: summary for summary in daily_summaries
    }
    period_summaries = [
        summary for summary in payload.get("summaries", []) if summary.get("period_type") != "daily"
    ]

    lines = [
        "# Mental State Bot Archive",
        "",
        f"- Schema: `{payload.get('schema_version')}`",
        f"- Exported at: `{payload.get('exported_at')}`",
        f"- Telegram user id: `{user.get('telegram_user_id')}`",
        f"- Timezone: `{user.get('timezone')}`",
        "",
        "## Overview",
        "",
        f"- Days: {len(payload.get('days', []))}",
        f"- Entries: {len(payload.get('entries', []))}",
        f"- Media items: {len(payload.get('media', []))}",
        f"- Summaries: {len(payload.get('summaries', []))}",
        f"- AI analyses: {len(payload.get('ai_analyses', []))}",
        f"- Model runs: {len(payload.get('model_runs', []))}",
        f"- Embedding records: {len(payload.get('embeddings', []))}",
        "",
    ]

    if period_summaries:
        lines.extend(["## Period Summaries", ""])
        for summary in period_summaries:
            lines.extend(
                [
                    f"### {summary.get('period_type')} {str(summary.get('period_start'))[:10]} - {str(summary.get('period_end'))[:10]}",
                    "",
                    _clean_markdown_text(summary.get("short_text") or ""),
                    "",
                ]
            )

    lines.extend(["## Days", ""])
    day_keys = sorted(set(entries_by_day) | set(daily_summaries_by_day))
    if not day_keys:
        lines.extend(["No diary entries exported.", ""])
    for day_key in day_keys:
        lines.extend([f"### {day_key}", ""])
        summary = daily_summaries_by_day.get(day_key)
        if summary:
            lines.extend(["**Summary**", "", _clean_markdown_text(summary.get("short_text") or ""), ""])
        entries = entries_by_day.get(day_key, [])
        if entries:
            lines.extend(["**Entries**", ""])
            for entry in entries:
                time_text = _entry_time_text(entry)
                source = entry.get("source") or "entry"
                raw_text = _clean_markdown_text(entry.get("raw_text") or "[no text]")
                lines.append(f"- `{time_text}` `{source}` {raw_text}")
            lines.append("")
        else:
            lines.extend(["No raw entries for this day.", ""])

    lines.extend(
        [
            "## Model Usage",
            "",
            _model_usage_table(payload.get("model_runs", [])),
            "",
            "## Embedding Metadata",
            "",
            _embedding_table(payload.get("embeddings", [])),
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def render_metrics_csv(payload: dict[str, Any]) -> str:
    output = StringIO()
    fieldnames = [
        "entry_id",
        "day_id",
        "local_date",
        "local_time",
        "local_timestamp",
        "created_at",
        "source",
        "raw_text",
        "activity_labels",
        "state_labels",
        "mood",
        "mood_confidence",
        "energy",
        "energy_confidence",
        "anxiety",
        "anxiety_confidence",
        "social_activity",
        "social_activity_confidence",
        "emptiness_present",
        "emptiness_confidence",
        "avoidance_present",
        "avoidance_confidence",
        "rumination_present",
        "rumination_confidence",
        "inability_to_start_present",
        "inability_to_start_confidence",
        "body_signals",
        "pleasant_moments",
        "pleasant_moments_count",
        "what_helped",
        "what_worsened",
        "transition",
        "data_quality",
        "feature_confidence",
        "uncertainty_notes",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    features_by_entry = _feature_analyses_by_entry(payload.get("ai_analyses", []))
    for entry in payload.get("entries", []):
        entry_id = str(entry.get("id") or "")
        features = features_by_entry.get(entry_id, {})
        timestamp = entry.get("local_timestamp") or entry.get("created_at")
        local_date_text = str(timestamp)[:10] if timestamp else ""
        local_time_text = str(timestamp)[11:16] if timestamp and len(str(timestamp)) >= 16 else ""
        pleasant_moments = features.get("pleasant_moments") or []
        writer.writerow(
            {
                "entry_id": entry_id,
                "day_id": str(entry.get("day_id") or ""),
                "local_date": local_date_text,
                "local_time": local_time_text,
                "local_timestamp": str(entry.get("local_timestamp") or ""),
                "created_at": str(entry.get("created_at") or ""),
                "source": entry.get("source") or "",
                "raw_text": entry.get("raw_text") or "",
                "activity_labels": _csv_list(features.get("activity_labels")),
                "state_labels": _csv_list(features.get("state_labels")),
                "mood": _feature_value(features.get("mood")),
                "mood_confidence": _feature_confidence(features.get("mood")),
                "energy": _feature_value(features.get("energy")),
                "energy_confidence": _feature_confidence(features.get("energy")),
                "anxiety": _feature_value(features.get("anxiety")),
                "anxiety_confidence": _feature_confidence(features.get("anxiety")),
                "social_activity": _feature_value(features.get("social_activity")),
                "social_activity_confidence": _feature_confidence(features.get("social_activity")),
                "emptiness_present": _presence_value(features.get("emptiness")),
                "emptiness_confidence": _feature_confidence(features.get("emptiness")),
                "avoidance_present": _presence_value(features.get("avoidance")),
                "avoidance_confidence": _feature_confidence(features.get("avoidance")),
                "rumination_present": _presence_value(features.get("rumination")),
                "rumination_confidence": _feature_confidence(features.get("rumination")),
                "inability_to_start_present": _presence_value(features.get("inability_to_start")),
                "inability_to_start_confidence": _feature_confidence(features.get("inability_to_start")),
                "body_signals": _csv_list(features.get("body_signals")),
                "pleasant_moments": _csv_list(pleasant_moments),
                "pleasant_moments_count": len(pleasant_moments),
                "what_helped": _csv_list(features.get("what_helped")),
                "what_worsened": _csv_list(features.get("what_worsened")),
                "transition": features.get("transition") or "",
                "data_quality": features.get("data_quality") or "",
                "feature_confidence": features.get("confidence") or "",
                "uncertainty_notes": _csv_list(features.get("uncertainty_notes")),
            }
        )
    return output.getvalue()


def write_archive_bundle(payload: dict[str, Any], output: Path, *, media_root: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "files": [],
        "missing_media": [],
    }

    def add_text(archive: zipfile.ZipFile, archive_path: str, content: str) -> None:
        archive.writestr(archive_path, content)
        manifest["files"].append({"path": archive_path, "kind": "data"})

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        add_text(
            archive,
            "archive.json",
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        )
        add_text(archive, "archive.md", render_archive_markdown(payload))
        add_text(archive, "metrics.csv", render_metrics_csv(payload))

        for media in payload.get("media", []):
            source = _media_source_path(media.get("file_path"), media_root=media_root)
            if source is None:
                manifest["missing_media"].append(
                    {
                        "media_id": str(media.get("id") or ""),
                        "entry_id": str(media.get("entry_id") or ""),
                        "file_path": media.get("file_path"),
                        "reason": "file_path_missing",
                    }
                )
                continue
            if not source.exists() or not source.is_file():
                manifest["missing_media"].append(
                    {
                        "media_id": str(media.get("id") or ""),
                        "entry_id": str(media.get("entry_id") or ""),
                        "file_path": media.get("file_path"),
                        "reason": "file_not_found",
                    }
                )
                continue

            archive_path = _media_archive_path(media, source)
            archive.write(source, archive_path)
            manifest["files"].append(
                {
                    "path": archive_path,
                    "kind": "media",
                    "media_id": str(media.get("id") or ""),
                    "entry_id": str(media.get("entry_id") or ""),
                    "media_type": media.get("media_type") or "",
                }
            )

        if manifest["missing_media"]:
            add_text(
                archive,
                "missing_media.json",
                json.dumps(manifest["missing_media"], ensure_ascii=False, indent=2),
            )
        manifest["files"].append({"path": "export_manifest.json", "kind": "manifest"})
        archive.writestr("export_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def build_archive_payload(
    *,
    user,
    user_settings,
    days,
    snapshots,
    prompts,
    entries,
    media,
    missed_prompts,
    analyses,
    summaries,
    model_runs,
    embeddings,
    retrieval_logs,
    export_jobs,
) -> dict[str, Any]:
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "exported_at": datetime.now(tz=UTC).isoformat(),
        "user": {
            "id": user.id,
            "telegram_user_id": user.telegram_user_id,
            "chat_id": user.chat_id,
            "username": user.username,
            "first_name": user.first_name,
            "timezone": user.timezone,
            "first_seen_at": user.first_seen_at,
            "last_seen_at": user.last_seen_at,
            "is_active": user.is_active,
        },
        "user_settings": _user_settings_to_dict(user_settings) if user_settings else None,
        "days": [_day_to_dict(item) for item in days],
        "snapshots": [_snapshot_to_dict(item) for item in snapshots],
        "snapshot_prompts": [_prompt_to_dict(item) for item in prompts],
        "entries": [_entry_to_dict(item) for item in entries],
        "media": [_media_to_dict(item) for item in media],
        "missed_prompts": [_missed_prompt_to_dict(item) for item in missed_prompts],
        "ai_analyses": [_analysis_to_dict(item) for item in analyses],
        "summaries": [_summary_to_dict(item) for item in summaries],
        "model_runs": [_model_run_to_dict(item) for item in model_runs],
        "embeddings": [_embedding_to_dict(item) for item in embeddings],
        "retrieval_logs": [_retrieval_log_to_dict(item) for item in retrieval_logs],
        "exports": [_export_job_to_dict(item) for item in export_jobs],
    }


def _user_query(model, user_id: UUID):
    return select(model).where(model.user_id == user_id)


def _user_settings_to_dict(settings: UserSettings) -> dict[str, Any]:
    return {
        "id": settings.id,
        "tone": settings.tone,
        "humanity_level": settings.humanity_level,
        "active_start": settings.active_start,
        "active_end": settings.active_end,
        "min_interval_minutes": settings.min_interval_minutes,
        "max_interval_minutes": settings.max_interval_minutes,
        "reminder_delay_minutes": settings.reminder_delay_minutes,
        "max_clarifications": settings.max_clarifications,
        "ask_body_signals": settings.ask_body_signals,
        "photo_prompts_enabled": settings.photo_prompts_enabled,
        "settings_json": settings.settings_json,
        "created_at": settings.created_at,
        "updated_at": settings.updated_at,
    }


def _day_to_dict(day: Day) -> dict[str, Any]:
    return {
        "id": day.id,
        "local_date": day.local_date,
        "started_at": day.started_at,
        "ended_at": day.ended_at,
        "boundary_kind": day.boundary_kind,
        "data_quality": day.data_quality,
        "created_at": day.created_at,
    }


def _snapshot_to_dict(snapshot: Snapshot) -> dict[str, Any]:
    return {
        "id": snapshot.id,
        "day_id": snapshot.day_id,
        "status": snapshot.status,
        "intent": snapshot.intent,
        "scheduled_for": snapshot.scheduled_for,
        "prompted_at": snapshot.prompted_at,
        "closed_at": snapshot.closed_at,
        "clarification_count": snapshot.clarification_count,
        "context_json": snapshot.context_json,
        "created_at": snapshot.created_at,
    }


def _prompt_to_dict(prompt: SnapshotPrompt) -> dict[str, Any]:
    return {
        "id": prompt.id,
        "snapshot_id": prompt.snapshot_id,
        "prompt_kind": prompt.prompt_kind,
        "text": prompt.text,
        "telegram_message_id": prompt.telegram_message_id,
        "sent_at": prompt.sent_at,
        "model_run_id": prompt.model_run_id,
        "created_at": prompt.created_at,
    }


def _entry_to_dict(entry: Entry) -> dict[str, Any]:
    return {
        "id": entry.id,
        "day_id": entry.day_id,
        "snapshot_id": entry.snapshot_id,
        "source": entry.source,
        "raw_text": entry.raw_text,
        "normalized_text": entry.normalized_text,
        "telegram_message_id": entry.telegram_message_id,
        "reply_to_message_id": entry.reply_to_message_id,
        "local_timestamp": entry.local_timestamp,
        "created_at": entry.created_at,
        "metadata": entry.meta,
    }


def _media_to_dict(media: Media) -> dict[str, Any]:
    return {
        "id": media.id,
        "entry_id": media.entry_id,
        "media_type": media.media_type,
        "telegram_file_id": media.telegram_file_id,
        "telegram_file_unique_id": media.telegram_file_unique_id,
        "file_path": media.file_path,
        "metadata": media.meta,
        "created_at": media.created_at,
    }


def _missed_prompt_to_dict(missed: MissedPrompt) -> dict[str, Any]:
    return {
        "id": missed.id,
        "snapshot_id": missed.snapshot_id,
        "prompt_id": missed.prompt_id,
        "status": missed.status,
        "missed_at": missed.missed_at,
        "reminder_sent_at": missed.reminder_sent_at,
        "resolved_at": missed.resolved_at,
        "reason_text": missed.reason_text,
        "created_at": missed.created_at,
    }


def _analysis_to_dict(analysis: AIAnalysis) -> dict[str, Any]:
    return {
        "id": analysis.id,
        "target_type": analysis.target_type,
        "target_id": analysis.target_id,
        "task_name": analysis.task_name,
        "schema_version": analysis.schema_version,
        "provider": analysis.provider,
        "model": analysis.model,
        "result": analysis.result,
        "confidence": analysis.confidence,
        "uncertainty_notes": analysis.uncertainty_notes,
        "model_run_id": analysis.model_run_id,
        "created_at": analysis.created_at,
    }


def _summary_to_dict(summary: Summary) -> dict[str, Any]:
    return {
        "id": summary.id,
        "day_id": summary.day_id,
        "period_type": summary.period_type,
        "period_start": summary.period_start,
        "period_end": summary.period_end,
        "short_text": summary.short_text,
        "details": summary.details,
        "model_run_id": summary.model_run_id,
        "delivered_at": summary.delivered_at,
        "created_at": summary.created_at,
    }


def _model_run_to_dict(run: ModelRun) -> dict[str, Any]:
    return {
        "id": run.id,
        "provider": run.provider,
        "model": run.model,
        "task_name": run.task_name,
        "status": run.status,
        "prompt_tokens": run.prompt_tokens,
        "completion_tokens": run.completion_tokens,
        "reasoning_tokens": run.reasoning_tokens,
        "total_tokens": run.total_tokens,
        "estimated_cost_usd": run.estimated_cost_usd,
        "latency_ms": run.latency_ms,
        "error_message": run.error_message,
        "request_hash": run.request_hash,
        "metadata": run.meta,
        "created_at": run.created_at,
    }


def _embedding_to_dict(record: EmbeddingRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "target_type": record.target_type,
        "target_id": record.target_id,
        "provider": record.provider,
        "model": record.model,
        "dimensions": record.dimensions,
        "source_hash": record.source_hash,
        "source_text": record.source_text,
        "created_at": record.created_at,
        "vector_exported": False,
    }


def _retrieval_log_to_dict(log: RetrievalLog) -> dict[str, Any]:
    return {
        "id": log.id,
        "task_name": log.task_name,
        "query_text": log.query_text,
        "provider": log.provider,
        "model": log.model,
        "retrieved": log.retrieved,
        "created_at": log.created_at,
    }


def _export_job_to_dict(job: ExportJob) -> dict[str, Any]:
    return {
        "id": job.id,
        "status": job.status,
        "format": job.format,
        "file_path": job.file_path,
        "metadata": job.meta,
        "created_at": job.created_at,
    }


def _entry_day_key(entry: dict[str, Any]) -> str:
    value = entry.get("local_timestamp") or entry.get("created_at")
    if value:
        return str(value)[:10]
    return "unknown-date"


def _entry_time_text(entry: dict[str, Any]) -> str:
    value = entry.get("local_timestamp") or entry.get("created_at")
    if value and len(str(value)) >= 16:
        return str(value)[11:16]
    return "??:??"


def _clean_markdown_text(text: str) -> str:
    return " ".join(str(text).replace("\r", " ").replace("\n", " ").split())


def _table_cell(value: Any) -> str:
    return _clean_markdown_text(str(value or "")).replace("|", "\\|")


def _model_usage_table(model_runs: list[dict[str, Any]]) -> str:
    if not model_runs:
        return "No model runs exported."
    lines = [
        "| Task | Model | Status | Tokens | Cost USD |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for run in model_runs:
        lines.append(
            "| "
            + " | ".join(
                [
                    _table_cell(run.get("task_name")),
                    _table_cell(run.get("model")),
                    _table_cell(run.get("status")),
                    str(run.get("total_tokens") or 0),
                    str(run.get("estimated_cost_usd") or ""),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _embedding_table(embeddings: list[dict[str, Any]]) -> str:
    if not embeddings:
        return "No embedding metadata exported."
    lines = [
        "| Target | Model | Dimensions | Source hash |",
        "| --- | --- | ---: | --- |",
    ]
    for record in embeddings:
        lines.append(
            "| "
            + " | ".join(
                [
                    _table_cell(f"{record.get('target_type')}:{record.get('target_id')}"),
                    _table_cell(record.get("model")),
                    str(record.get("dimensions") or ""),
                    _table_cell(record.get("source_hash")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _feature_analyses_by_entry(analyses: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    features_by_entry: dict[str, dict[str, Any]] = {}
    for analysis in analyses:
        if analysis.get("target_type") != "entry":
            continue
        if analysis.get("task_name") != "extract_entry_features":
            continue
        result = analysis.get("result")
        if isinstance(result, dict):
            features_by_entry[str(analysis.get("target_id"))] = result
    return features_by_entry


def _csv_list(value: Any) -> str:
    if not value:
        return ""
    if not isinstance(value, list):
        return str(value)
    return "; ".join(str(item) for item in value if item is not None)


def _feature_value(feature: Any) -> str:
    if isinstance(feature, dict):
        return str(feature.get("value") or "")
    return ""


def _presence_value(feature: Any) -> str:
    if isinstance(feature, dict):
        value = feature.get("present")
        if value is None:
            return ""
        return "true" if value else "false"
    return ""


def _feature_confidence(feature: Any) -> str:
    if isinstance(feature, dict):
        value = feature.get("confidence")
        return "" if value is None else str(value)
    return ""


def _media_source_path(file_path: Any, *, media_root: Path) -> Path | None:
    if not file_path:
        return None
    path = Path(str(file_path))
    if path.is_absolute():
        return path
    if path.exists():
        return path
    candidate = media_root / path
    if candidate.exists():
        return candidate
    return path


def _media_archive_path(media: dict[str, Any], source: Path) -> str:
    media_id = _safe_archive_part(media.get("id") or "media")
    entry_id = _safe_archive_part(media.get("entry_id") or "no-entry")
    filename = _safe_archive_part(source.name or "file")
    return f"media/{entry_id}/{media_id}-{filename}"


def _safe_archive_part(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "-").replace("/", "-")
    return "".join(char if char.isalnum() or char in "._-" else "-" for char in text) or "unknown"


def _json_default(value: Any) -> str:
    if isinstance(value, datetime | date | UUID | Decimal):
        return str(value)
    return str(value)
