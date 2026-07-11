from __future__ import annotations

from dataclasses import dataclass

from mental_state_bot.config import Settings


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    ok: bool
    message: str


def run_doctor(settings: Settings) -> list[DoctorCheck]:
    checks = [
        DoctorCheck(
            name="telegram_bot_token",
            ok=bool(settings.telegram_bot_token),
            message="TELEGRAM_BOT_TOKEN configured" if settings.telegram_bot_token else "TELEGRAM_BOT_TOKEN is missing",
        ),
        DoctorCheck(
            name="allowed_users",
            ok=bool(settings.telegram_allowed_user_ids),
            message=(
                f"restricted to {len(settings.telegram_allowed_user_ids)} Telegram user id(s)"
                if settings.telegram_allowed_user_ids
                else "TELEGRAM_ALLOWED_USER_IDS is empty; any Telegram user who finds the bot can write"
            ),
        ),
        DoctorCheck(
            name="database_url",
            ok=settings.database_url.startswith("postgresql+asyncpg://"),
            message="DATABASE_URL uses async PostgreSQL driver",
        ),
        DoctorCheck(
            name="database_sync_url",
            ok=settings.database_sync_url.startswith("postgresql+psycopg://"),
            message="DATABASE_SYNC_URL uses sync PostgreSQL driver for Alembic",
        ),
        DoctorCheck(
            name="ai_api_key",
            ok=bool(settings.ai_api_key),
            message="AI_API_KEY configured" if settings.ai_api_key else "AI_API_KEY missing; AI tasks will use fallbacks",
        ),
        DoctorCheck(
            name="ai_models",
            ok=bool(settings.ai_live_model and settings.ai_heavy_model),
            message=f"live={settings.ai_live_model}, heavy={settings.ai_heavy_model}",
        ),
        DoctorCheck(
            name="embeddings",
            ok=(not settings.embeddings_enabled) or bool(settings.embedding_api_key),
            message=(
                f"embeddings enabled with model {settings.embedding_model}"
                if settings.embeddings_enabled and settings.embedding_api_key
                else "embeddings enabled but EMBEDDING_API_KEY is missing"
                if settings.embeddings_enabled
                else "embeddings disabled"
            ),
        ),
        DoctorCheck(
            name="media_root",
            ok=settings.media_root.exists(),
            message=f"media root: {settings.media_root}",
        ),
        DoctorCheck(
            name="snapshot_interval",
            ok=settings.snapshot_min_interval_minutes <= settings.snapshot_max_interval_minutes,
            message=(
                f"{settings.snapshot_min_interval_minutes}-{settings.snapshot_max_interval_minutes} minutes"
            ),
        ),
    ]
    return checks


def format_doctor_report(checks: list[DoctorCheck]) -> str:
    lines = ["Mental State Bot doctor:"]
    for check in checks:
        mark = "OK" if check.ok else "WARN"
        lines.append(f"[{mark}] {check.name}: {check.message}")
    warnings = sum(1 for check in checks if not check.ok)
    lines.append("")
    lines.append(f"Warnings: {warnings}")
    return "\n".join(lines)
