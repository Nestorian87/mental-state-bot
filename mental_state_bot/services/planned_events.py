from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from mental_state_bot.db.models import UserSettings
from mental_state_bot.services.preferences import planned_events
from mental_state_bot.time_utils import zoneinfo

_EVENT_KEYWORDS = {
    "терапія": "терапія",
    "психотерапія": "терапія",
    "сесія": "сесія",
    "зустріч": "зустріч",
    "дзвінок": "дзвінок",
    "созвон": "дзвінок",
    "консультація": "консультація",
}
_FUTURE_MARKER_RE = re.compile(
    r"(?:через\s+(?:\d+\s*)?(?:хв|хвилин|год|годину|години)|"
    r"о\s+\d{1,2}[:.]\d{2}|"
    r"сьогодні\s+(?:буде|маю|йду|піду)|"
    r"завтра|"
    r"ще\s+не\s+поч|"
    r"ще\s+попереду|"
    r"попереду|"
    r"йду\s+на|"
    r"їду\s+на)",
    re.IGNORECASE,
)
_PAST_MARKER_RE = re.compile(
    r"\b(?:після|вийшов\s+з|вийшла\s+з|закінчил|пройшл|була|був|щойно\s+з)\b",
    re.IGNORECASE,
)
_RELATIVE_HOURS_RE = re.compile(r"через\s+(?:(\d+)\s*)?(год|годину|години)", re.IGNORECASE)
_RELATIVE_MINUTES_RE = re.compile(r"\bчерез\s+(\d+)\s*(хв|хвилин)\b", re.IGNORECASE)
_CLOCK_RE = re.compile(r"\bо\s+(\d{1,2})[:.](\d{2})\b", re.IGNORECASE)


def detect_planned_event_candidate(
    text: str,
    *,
    timezone: str,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    compact = " ".join(str(text or "").split())
    if not compact:
        return None
    lowered = compact.lower()
    if _PAST_MARKER_RE.search(lowered) and not re.search(r"\bще\s+не\s+поч", lowered):
        return None
    if not _FUTURE_MARKER_RE.search(lowered):
        return None
    title = _event_title(lowered)
    if title is None:
        return None
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    local_now = current.astimezone(zoneinfo(timezone))
    starts_at = _starts_at(lowered, local_now=local_now)
    source_text = _source_snippet(compact, title)
    return {
        "id": str(uuid.uuid4()),
        "title": title,
        "source_text": source_text,
        "starts_at": starts_at.astimezone(UTC).isoformat() if starts_at else None,
        "status": "pending",
        "created_at": current.astimezone(UTC).isoformat(),
    }


def planned_event_context(settings: UserSettings, *, now: datetime | None = None) -> list[dict[str, Any]]:
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    visible: list[dict[str, Any]] = []
    for event in planned_events(settings):
        if not isinstance(event, dict) or event.get("status") != "confirmed":
            continue
        starts_at = _parse_datetime(event.get("starts_at"))
        if starts_at is not None and starts_at < current.astimezone(UTC) - timedelta(hours=4):
            continue
        visible.append(
            {
                "title": event.get("title"),
                "starts_at": event.get("starts_at"),
                "source_text": event.get("source_text"),
                "status": event.get("status"),
            }
        )
    return visible[-10:]


def confirm_pending_planned_event(settings: UserSettings) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    pending = (getattr(settings, "settings_json", None) or {}).get("pending_planned_event")
    if not isinstance(pending, dict):
        return None, planned_events(settings)
    confirmed = {**pending, "status": "confirmed", "confirmed_at": datetime.now(UTC).isoformat()}
    events = [event for event in planned_events(settings) if event.get("id") != confirmed.get("id")]
    events.append(confirmed)
    return confirmed, events[-20:]


def planned_event_text(event: dict[str, Any]) -> str:
    title = str(event.get("title") or "подія")
    starts_at = event.get("starts_at")
    if isinstance(starts_at, str) and starts_at:
        return f"{title}, приблизно {starts_at}"
    return title


def _event_title(text: str) -> str | None:
    for keyword, title in _EVENT_KEYWORDS.items():
        if re.search(rf"\b{re.escape(keyword)}\w*\b", text, re.IGNORECASE):
            return title
    return None


def _starts_at(text: str, *, local_now: datetime) -> datetime | None:
    if match := _RELATIVE_HOURS_RE.search(text):
        return local_now + timedelta(hours=int(match.group(1) or 1))
    if match := _RELATIVE_MINUTES_RE.search(text):
        return local_now + timedelta(minutes=int(match.group(1)))
    if match := _CLOCK_RE.search(text):
        hour = int(match.group(1))
        minute = int(match.group(2))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            candidate = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if "завтра" in text or candidate < local_now - timedelta(minutes=15):
                candidate += timedelta(days=1)
            return candidate
    return None


def _source_snippet(text: str, title: str) -> str:
    index = text.lower().find(title.lower())
    if index < 0:
        return text[:220]
    start = max(0, index - 80)
    end = min(len(text), index + 140)
    return text[start:end].strip()


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
