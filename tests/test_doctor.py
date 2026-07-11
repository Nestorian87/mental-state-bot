from __future__ import annotations

from mental_state_bot.config import Settings
from mental_state_bot.services.doctor import format_doctor_report, run_doctor


def test_doctor_warns_without_tokens(tmp_path) -> None:
    settings = Settings(
        telegram_bot_token="",
        ai_api_key="",
        embedding_api_key="",
        media_root=tmp_path,
    )

    checks = run_doctor(settings)
    report = format_doctor_report(checks)

    assert "[WARN] telegram_bot_token" in report
    assert "[WARN] ai_api_key" in report
    assert "[WARN] embeddings" in report


def test_doctor_accepts_core_config(tmp_path) -> None:
    settings = Settings(
        telegram_bot_token="telegram",
        telegram_allowed_user_ids=[123],
        ai_api_key="ai",
        embedding_api_key="emb",
        media_root=tmp_path,
    )

    checks = run_doctor(settings)
    failed = {check.name for check in checks if not check.ok}

    assert failed == set()
