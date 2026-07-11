from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any

from mental_state_bot.db.models import UserSettings
from mental_state_bot.services.preferences import wake_time_records
from mental_state_bot.time_utils import zoneinfo

_CLOCK_RE = re.compile(r"(?:^|\b|о\s+)(\d{1,2})(?::|\.|:)?(\d{2})?\b", re.IGNORECASE)
_RELATIVE_MINUTES_RE = re.compile(r"(\d+)\s*(?:хв|хвилин)\s+тому", re.IGNORECASE)
_RELATIVE_HOURS_RE = re.compile(r"(\d+)?\s*(?:год|годину|години)\s+тому", re.IGNORECASE)
_HALF_HOUR_RE = re.compile(r"пів\s*години\s+тому|півгодини\s+тому", re.IGNORECASE)


def should_offer_wake_time_question(
    *,
    entries,
    current_text: str,
    user_settings: UserSettings,
    local_date: str,
) -> bool:
    _ = current_text
    if wake_time_record_for_date(user_settings, local_date) is not None:
        return False
    meaningful = [entry for entry in entries if _is_meaningful_entry(entry)]
    return len(meaningful) == 1


def parse_wake_time_text(
    text: str,
    *,
    timezone: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    compact = " ".join(str(text or "").split())
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    local_now = current.astimezone(zoneinfo(timezone))
    estimated = _parse_estimated_local_time(compact, local_now=local_now)
    return {
        "raw_text": compact,
        "estimated_woke_at": estimated.astimezone(UTC).isoformat() if estimated else None,
        "local_date": local_now.date().isoformat(),
        "recorded_at": current.astimezone(UTC).isoformat(),
        "status": "recorded" if compact else "skipped",
        "source": "user_optional_morning_answer",
    }


def skipped_wake_time_record(*, timezone: str, now: datetime | None = None) -> dict[str, Any]:
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    local_now = current.astimezone(zoneinfo(timezone))
    return {
        "raw_text": None,
        "estimated_woke_at": None,
        "local_date": local_now.date().isoformat(),
        "recorded_at": current.astimezone(UTC).isoformat(),
        "status": "skipped",
        "source": "user_skipped_optional_morning_answer",
    }


def wake_time_record_for_date(settings: UserSettings, local_date: str) -> dict[str, Any] | None:
    for record in reversed(wake_time_records(settings)):
        if record.get("local_date") == local_date:
            return record
    return None


def append_wake_time_record(settings: UserSettings, record: dict[str, Any]) -> list[dict[str, Any]]:
    records = [item for item in wake_time_records(settings) if item.get("local_date") != record.get("local_date")]
    records.append(record)
    return records[-60:]


def _parse_estimated_local_time(text: str, *, local_now: datetime) -> datetime | None:
    lowered = text.lower()
    if _HALF_HOUR_RE.search(lowered):
        return local_now - timedelta(minutes=30)
    if match := _RELATIVE_MINUTES_RE.search(lowered):
        return local_now - timedelta(minutes=int(match.group(1)))
    if match := _RELATIVE_HOURS_RE.search(lowered):
        return local_now - timedelta(hours=int(match.group(1) or 1))
    if match := _CLOCK_RE.search(lowered):
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        if not 0 <= hour <= 23 or not 0 <= minute <= 59:
            return None
        candidate = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate > local_now + timedelta(hours=2):
            candidate -= timedelta(days=1)
        return candidate
    return None


def _is_meaningful_entry(entry) -> bool:
    source = str(getattr(entry, "source", "") or "")
    if source in {"sleep_marker", "day_reflection", "correction", "profile_context_update", "missed_reason", "wake_time"}:
        return False
    if source.startswith("button_"):
        return False
    return bool(str(getattr(entry, "raw_text", "") or "").strip())
