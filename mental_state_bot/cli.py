from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from mental_state_bot.config import get_settings
from mental_state_bot.logging import configure_logging

app = typer.Typer(no_args_is_help=True)


@app.command()
def run() -> None:
    """Run Telegram bot with scheduler."""
    settings = get_settings()
    configure_logging(settings.log_level)
    from mental_state_bot.bot.app import run_bot

    asyncio.run(run_bot(settings))


@app.command()
def migrate() -> None:
    """Run database migrations."""
    settings = get_settings()
    configure_logging(settings.log_level)
    env = {"DATABASE_SYNC_URL": settings.database_sync_url}
    config_path = _alembic_config_path()
    try:
        subprocess.run(
            [sys.executable, "-m", "alembic", "-c", str(config_path), "upgrade", "head"],
            cwd=config_path.parent,
            env={**os.environ, **env},
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        typer.echo("Database migration failed. See Alembic/PostgreSQL output above.", err=True)
        raise typer.Exit(exc.returncode) from exc


def _alembic_config_path() -> Path:
    candidates = [
        Path.cwd() / "alembic.ini",
        Path("/app/alembic.ini"),
        Path(__file__).resolve().parent.parent / "alembic.ini",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    typer.echo("Could not find alembic.ini. Run from the project directory or include it in the image.", err=True)
    raise typer.Exit(1)


@app.command()
def export(
    user_id: int,
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("./data/export.json"),
    export_format: Annotated[str | None, typer.Option("--format", "-f")] = None,
) -> None:
    """Export one Telegram user's diary data to JSON, Markdown, metrics CSV, or ZIP bundle."""
    settings = get_settings()
    configure_logging(settings.log_level)
    from mental_state_bot.services.exports import export_user_archive

    asyncio.run(export_user_archive(settings, user_id, output, format=export_format))
    typer.echo(f"Exported archive to {output}")


@app.command("embed-backfill")
def embed_backfill(user_id: int, limit: int = 100) -> None:
    """Generate missing embeddings for one Telegram user's entries."""
    settings = get_settings()
    configure_logging(settings.log_level)
    from mental_state_bot.ai.service import AIService
    from mental_state_bot.db.session import async_session_factory, create_async_engine_from_settings
    from mental_state_bot.services.memory import backfill_entry_embeddings

    engine = create_async_engine_from_settings(settings)
    sessionmaker = async_session_factory(engine)
    processed = asyncio.run(
        backfill_entry_embeddings(
            settings=settings,
            ai_service=AIService(settings),
            sessionmaker=sessionmaker,
            telegram_user_id=user_id,
            limit=limit,
        )
    )
    asyncio.run(engine.dispose())
    typer.echo(f"Backfilled embeddings for {processed} entries")


@app.command("features-backfill")
def features_backfill(user_id: int, limit: int = 100) -> None:
    """Generate missing AI feature analyses for one Telegram user's entries."""
    settings = get_settings()
    configure_logging(settings.log_level)
    from mental_state_bot.ai.service import AIService
    from mental_state_bot.db.session import async_session_factory, create_async_engine_from_settings
    from mental_state_bot.services.analysis_backfill import backfill_entry_features

    engine = create_async_engine_from_settings(settings)
    sessionmaker = async_session_factory(engine)
    result = asyncio.run(
        backfill_entry_features(
            settings=settings,
            ai_service=AIService(settings),
            sessionmaker=sessionmaker,
            telegram_user_id=user_id,
            limit=limit,
        )
    )
    asyncio.run(engine.dispose())
    typer.echo(
        "Backfilled entry features: "
        f"{result.processed}/{result.selected} processed, "
        f"{result.skipped_missing} skipped missing"
    )


@app.command("user-audit")
def user_audit(user_id: int) -> None:
    """Print archive/data coverage for one Telegram user."""
    settings = get_settings()
    configure_logging(settings.log_level)
    from mental_state_bot.db.repositories import get_user_by_telegram_id
    from mental_state_bot.db.session import async_session_factory, create_async_engine_from_settings
    from mental_state_bot.services.archive_audit import build_archive_audit, format_archive_audit

    async def _run() -> str:
        engine = create_async_engine_from_settings(settings)
        sessionmaker = async_session_factory(engine)
        try:
            async with sessionmaker() as session, session.begin():
                user = await get_user_by_telegram_id(session, user_id)
                if user is None:
                    raise ValueError(f"Unknown Telegram user id: {user_id}")
                audit = await build_archive_audit(session, settings=settings, user=user)
                return format_archive_audit(audit)
        finally:
            await engine.dispose()

    typer.echo(asyncio.run(_run()))


@app.command()
def healthcheck() -> None:
    """Small process healthcheck for containers."""
    settings = get_settings()
    settings.ensure_runtime_dirs()
    typer.echo("ok")


@app.command()
def doctor() -> None:
    """Check local configuration readiness."""
    settings = get_settings()
    configure_logging(settings.log_level)
    from mental_state_bot.services.doctor import format_doctor_report, run_doctor

    typer.echo(format_doctor_report(run_doctor(settings)))
