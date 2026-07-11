from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from mental_state_bot.db.models import UserSettings

SNAPSHOTS_PAUSED_KEY = "snapshots_paused"
QUIET_UNTIL_KEY = "quiet_until"
CONTEXT_QUIET_ENABLED_KEY = "context_quiet_enabled"
ADAPTIVE_OBSERVATION_ENABLED_KEY = "adaptive_observation_enabled"
CONTEXT_QUIET_LAST_OFFER_AT_KEY = "context_quiet_last_offer_at"
CONTEXT_QUIET_LAST_CHECK_AT_KEY = "context_quiet_last_check_at"
CUSTOM_INTERACTION_STYLE_KEY = "custom_interaction_style"
USER_PROFILE_CONTEXT_KEY = "user_profile_context"
PENDING_INPUT_KEY = "pending_input"
PENDING_VOICE_TRANSCRIPT_KEY = "pending_voice_transcript"
PENDING_MANUAL_ENTRY_KEY = "pending_manual_entry"
PENDING_CORRECTION_ENTRY_ID_KEY = "pending_correction_entry_id"
PENDING_EMOTION_ENTRY_ID_KEY = "pending_emotion_entry_id"
CLARIFICATION_QUEUE_KEY = "clarification_queue"
PENDING_CLARIFICATION_KEY = "pending_clarification"
PENDING_POST_ENTRY_FOLLOWUP_KEY = "pending_post_entry_followup"
POST_ENTRY_FOLLOWUP_TTL = timedelta(hours=2)
LIFE_CONTEXT_ITEMS_KEY = "life_context_items"
PENDING_LIFE_CONTEXT_REVIEW_KEY = "pending_life_context_review"
PENDING_LIFE_CONTEXT_REWRITE_KEY = "pending_life_context_rewrite"
LIFE_CONTEXT_LAST_AUTO_OFFER_AT_KEY = "life_context_last_auto_offer_at"
PLANNED_EVENTS_KEY = "planned_events"
PENDING_PLANNED_EVENT_KEY = "pending_planned_event"
PENDING_MEMORY_GRAPH_IMPORT_KEY = "pending_memory_graph_import"
WAKE_TIME_RECORDS_KEY = "wake_time_records"
PENDING_INPUT_KINDS = {
    "custom_style",
    "profile_context",
    "correction",
    "life_context_free_answer",
    "sleep_reflection",
    "quiet_until",
    "voice_transcript",
    "voice_transcript_fix",
    "day_date",
    "memory_search",
    "visual_report_range",
    "reanalysis_range",
    "planned_event_clarify",
    "wake_time",
    "memory_graph_import",
}


def snapshots_paused(settings: UserSettings) -> bool:
    return bool((getattr(settings, "settings_json", None) or {}).get(SNAPSHOTS_PAUSED_KEY))


def settings_json_with_snapshot_pause(settings: UserSettings, paused: bool) -> dict[str, Any]:
    return {**(getattr(settings, "settings_json", None) or {}), SNAPSHOTS_PAUSED_KEY: paused}


def quiet_until(settings: UserSettings) -> datetime | None:
    return _datetime_from_settings(settings, QUIET_UNTIL_KEY)


def quiet_is_active(settings: UserSettings, now: datetime | None = None) -> bool:
    value = quiet_until(settings)
    if value is None:
        return False
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    return value > current.astimezone(UTC)


def settings_json_with_quiet_until(settings: UserSettings, until: datetime | None) -> dict[str, Any]:
    current = dict(getattr(settings, "settings_json", None) or {})
    if until is None:
        current.pop(QUIET_UNTIL_KEY, None)
    else:
        current[QUIET_UNTIL_KEY] = until.astimezone(UTC).isoformat()
    return current


def context_quiet_enabled(settings: UserSettings) -> bool:
    return bool((getattr(settings, "settings_json", None) or {}).get(CONTEXT_QUIET_ENABLED_KEY, True))


def adaptive_observation_enabled(settings: UserSettings) -> bool:
    return bool((getattr(settings, "settings_json", None) or {}).get(ADAPTIVE_OBSERVATION_ENABLED_KEY, True))


def settings_json_with_adaptive_observation(settings: UserSettings, enabled: bool) -> dict[str, Any]:
    current = dict(getattr(settings, "settings_json", None) or {})
    current[ADAPTIVE_OBSERVATION_ENABLED_KEY] = enabled
    return current


def settings_json_with_context_quiet(settings: UserSettings, enabled: bool) -> dict[str, Any]:
    current = dict(getattr(settings, "settings_json", None) or {})
    current[CONTEXT_QUIET_ENABLED_KEY] = enabled
    return current


def context_quiet_last_offer_at(settings: UserSettings) -> datetime | None:
    return _datetime_from_settings(settings, CONTEXT_QUIET_LAST_OFFER_AT_KEY)


def settings_json_with_context_quiet_last_offer(settings: UserSettings, offered_at: datetime) -> dict[str, Any]:
    current = dict(getattr(settings, "settings_json", None) or {})
    current[CONTEXT_QUIET_LAST_OFFER_AT_KEY] = offered_at.astimezone(UTC).isoformat()
    return current


def context_quiet_last_check_at(settings: UserSettings) -> datetime | None:
    return _datetime_from_settings(settings, CONTEXT_QUIET_LAST_CHECK_AT_KEY)


def settings_json_with_context_quiet_last_check(settings: UserSettings, checked_at: datetime) -> dict[str, Any]:
    current = dict(getattr(settings, "settings_json", None) or {})
    current[CONTEXT_QUIET_LAST_CHECK_AT_KEY] = checked_at.astimezone(UTC).isoformat()
    return current


def _datetime_from_settings(settings: UserSettings, key: str) -> datetime | None:
    value = (getattr(settings, "settings_json", None) or {}).get(key)
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _stored_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def custom_interaction_style(settings: UserSettings) -> str | None:
    value = (getattr(settings, "settings_json", None) or {}).get(CUSTOM_INTERACTION_STYLE_KEY)
    if not isinstance(value, str):
        return None
    compact = " ".join(value.split())
    return compact or None


def settings_json_with_custom_interaction_style(settings: UserSettings, style: str | None) -> dict[str, Any]:
    current = dict(getattr(settings, "settings_json", None) or {})
    if style is None:
        current.pop(CUSTOM_INTERACTION_STYLE_KEY, None)
        return current
    current[CUSTOM_INTERACTION_STYLE_KEY] = " ".join(style.split())[:800]
    return current


def user_profile_context(settings: UserSettings) -> str | None:
    value = (getattr(settings, "settings_json", None) or {}).get(USER_PROFILE_CONTEXT_KEY)
    if not isinstance(value, str):
        return None
    compact = " ".join(value.split())
    return compact or None


def settings_json_with_user_profile_context(settings: UserSettings, context: str | None) -> dict[str, Any]:
    current = dict(getattr(settings, "settings_json", None) or {})
    if context is None:
        current.pop(USER_PROFILE_CONTEXT_KEY, None)
        return current
    current[USER_PROFILE_CONTEXT_KEY] = " ".join(context.split())[:2000]
    return current


def life_context_items(settings: UserSettings) -> list[dict[str, Any]]:
    value = (getattr(settings, "settings_json", None) or {}).get(LIFE_CONTEXT_ITEMS_KEY)
    return value if isinstance(value, list) else []


def planned_events(settings: UserSettings) -> list[dict[str, Any]]:
    value = (getattr(settings, "settings_json", None) or {}).get(PLANNED_EVENTS_KEY)
    return value if isinstance(value, list) else []


def wake_time_records(settings: UserSettings) -> list[dict[str, Any]]:
    value = (getattr(settings, "settings_json", None) or {}).get(WAKE_TIME_RECORDS_KEY)
    return value if isinstance(value, list) else []


def clarification_queue(settings: UserSettings) -> list[dict[str, Any]]:
    value = (getattr(settings, "settings_json", None) or {}).get(CLARIFICATION_QUEUE_KEY)
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def settings_json_with_clarification_queue(
    settings: UserSettings,
    queue: list[dict[str, Any]],
) -> dict[str, Any]:
    current = dict(getattr(settings, "settings_json", None) or {})
    current[CLARIFICATION_QUEUE_KEY] = queue[-40:]
    return current


def pending_clarification(settings: UserSettings) -> dict[str, Any] | None:
    value = (getattr(settings, "settings_json", None) or {}).get(PENDING_CLARIFICATION_KEY)
    return value if isinstance(value, dict) else None


def pending_post_entry_followup(settings: UserSettings) -> dict[str, Any] | None:
    value = (getattr(settings, "settings_json", None) or {}).get(PENDING_POST_ENTRY_FOLLOWUP_KEY)
    return value if isinstance(value, dict) else None


def post_entry_followup_is_active(settings: UserSettings, now: datetime | None = None) -> bool:
    followup = pending_post_entry_followup(settings)
    if followup is None:
        return False
    created_at = _stored_datetime(followup.get("created_at"))
    if created_at is None:
        return True
    return created_at + POST_ENTRY_FOLLOWUP_TTL > (now or datetime.now(UTC)).astimezone(UTC)


def settings_json_with_pending_post_entry_followup(
    settings: UserSettings,
    followup: dict[str, Any] | None,
) -> dict[str, Any]:
    current = dict(getattr(settings, "settings_json", None) or {})
    if followup is None:
        current.pop(PENDING_POST_ENTRY_FOLLOWUP_KEY, None)
    else:
        current[PENDING_POST_ENTRY_FOLLOWUP_KEY] = followup
    return current


def settings_json_with_pending_clarification(
    settings: UserSettings,
    item: dict[str, Any] | None,
) -> dict[str, Any]:
    current = dict(getattr(settings, "settings_json", None) or {})
    if item is None:
        current.pop(PENDING_CLARIFICATION_KEY, None)
    else:
        current[PENDING_CLARIFICATION_KEY] = item
    return current


def settings_json_with_wake_time_records(
    settings: UserSettings,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    current = dict(getattr(settings, "settings_json", None) or {})
    current[WAKE_TIME_RECORDS_KEY] = records[-60:]
    return current


def pending_planned_event(settings: UserSettings) -> dict[str, Any] | None:
    value = (getattr(settings, "settings_json", None) or {}).get(PENDING_PLANNED_EVENT_KEY)
    return value if isinstance(value, dict) else None


def settings_json_with_pending_planned_event(
    settings: UserSettings,
    event: dict[str, Any] | None,
) -> dict[str, Any]:
    current = dict(getattr(settings, "settings_json", None) or {})
    if event is None:
        current.pop(PENDING_PLANNED_EVENT_KEY, None)
    else:
        current[PENDING_PLANNED_EVENT_KEY] = event
    return current


def settings_json_with_planned_events(
    settings: UserSettings,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    current = dict(getattr(settings, "settings_json", None) or {})
    current[PLANNED_EVENTS_KEY] = events[-20:]
    return current


def pending_life_context_review(settings: UserSettings) -> dict[str, Any] | None:
    value = (getattr(settings, "settings_json", None) or {}).get(PENDING_LIFE_CONTEXT_REVIEW_KEY)
    return value if isinstance(value, dict) else None


def pending_life_context_rewrite(settings: UserSettings) -> dict[str, Any] | None:
    value = (getattr(settings, "settings_json", None) or {}).get(PENDING_LIFE_CONTEXT_REWRITE_KEY)
    return value if isinstance(value, dict) else None


def settings_json_with_life_context_items(
    settings: UserSettings,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    current = dict(getattr(settings, "settings_json", None) or {})
    current[LIFE_CONTEXT_ITEMS_KEY] = items[:80]
    return current


def settings_json_with_pending_life_context_review(
    settings: UserSettings,
    review: dict[str, Any] | None,
) -> dict[str, Any]:
    current = dict(getattr(settings, "settings_json", None) or {})
    if review is None:
        current.pop(PENDING_LIFE_CONTEXT_REVIEW_KEY, None)
    else:
        current[PENDING_LIFE_CONTEXT_REVIEW_KEY] = review
    return current


def settings_json_with_pending_life_context_rewrite(
    settings: UserSettings,
    rewrite: dict[str, Any] | None,
) -> dict[str, Any]:
    current = dict(getattr(settings, "settings_json", None) or {})
    if rewrite is None:
        current.pop(PENDING_LIFE_CONTEXT_REWRITE_KEY, None)
    else:
        current[PENDING_LIFE_CONTEXT_REWRITE_KEY] = rewrite
    return current


def pending_input(settings: UserSettings) -> str | None:
    value = (getattr(settings, "settings_json", None) or {}).get(PENDING_INPUT_KEY)
    if value in PENDING_INPUT_KINDS:
        return value
    return None


def settings_json_with_pending_input(settings: UserSettings, kind: str | None) -> dict[str, Any]:
    current = dict(getattr(settings, "settings_json", None) or {})
    if kind is None:
        current.pop(PENDING_INPUT_KEY, None)
        return current
    if kind not in PENDING_INPUT_KINDS:
        raise ValueError(f"Unsupported pending input kind: {kind}")
    current[PENDING_INPUT_KEY] = kind
    return current


def settings_json_without_pending_input(settings: UserSettings) -> dict[str, Any]:
    current = settings_json_with_pending_input(settings, None)
    current.pop(PENDING_CORRECTION_ENTRY_ID_KEY, None)
    current.pop(PENDING_EMOTION_ENTRY_ID_KEY, None)
    return current


def pending_memory_graph_import(settings: UserSettings) -> dict[str, Any] | None:
    value = (getattr(settings, "settings_json", None) or {}).get(PENDING_MEMORY_GRAPH_IMPORT_KEY)
    return dict(value) if isinstance(value, dict) else None


def settings_json_with_pending_memory_graph_import(
    settings: UserSettings,
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    current = dict(getattr(settings, "settings_json", None) or {})
    if payload is None:
        current.pop(PENDING_MEMORY_GRAPH_IMPORT_KEY, None)
    else:
        current[PENDING_MEMORY_GRAPH_IMPORT_KEY] = payload
    return current


def pending_correction_entry_id(settings: UserSettings) -> str | None:
    value = (getattr(settings, "settings_json", None) or {}).get(PENDING_CORRECTION_ENTRY_ID_KEY)
    return str(value) if value else None


def settings_json_with_pending_correction_entry_id(
    settings: UserSettings,
    entry_id: str | None,
) -> dict[str, Any]:
    current = dict(getattr(settings, "settings_json", None) or {})
    if entry_id is None:
        current.pop(PENDING_CORRECTION_ENTRY_ID_KEY, None)
    else:
        current[PENDING_CORRECTION_ENTRY_ID_KEY] = entry_id
    return current


def settings_json_without_pending_correction_entry_id(settings: UserSettings) -> dict[str, Any]:
    return settings_json_with_pending_correction_entry_id(settings, None)


def pending_emotion_entry_id(settings: UserSettings) -> str | None:
    value = (getattr(settings, "settings_json", None) or {}).get(PENDING_EMOTION_ENTRY_ID_KEY)
    return str(value) if value else None


def settings_json_with_pending_emotion_entry_id(
    settings: UserSettings,
    entry_id: str | None,
) -> dict[str, Any]:
    current = dict(getattr(settings, "settings_json", None) or {})
    if entry_id is None:
        current.pop(PENDING_EMOTION_ENTRY_ID_KEY, None)
    else:
        current[PENDING_EMOTION_ENTRY_ID_KEY] = entry_id
    return current


def pending_voice_transcript(settings: UserSettings) -> dict[str, Any] | None:
    value = (getattr(settings, "settings_json", None) or {}).get(PENDING_VOICE_TRANSCRIPT_KEY)
    return value if isinstance(value, dict) else None


def pending_manual_entry(settings: UserSettings) -> dict[str, Any] | None:
    value = (getattr(settings, "settings_json", None) or {}).get(PENDING_MANUAL_ENTRY_KEY)
    return value if isinstance(value, dict) else None


def settings_json_with_pending_manual_entry(
    settings: UserSettings,
    entry: dict[str, Any],
) -> dict[str, Any]:
    current = dict(getattr(settings, "settings_json", None) or {})
    current[PENDING_MANUAL_ENTRY_KEY] = entry
    return current


def settings_json_without_pending_manual_entry(settings: UserSettings) -> dict[str, Any]:
    current = dict(getattr(settings, "settings_json", None) or {})
    current.pop(PENDING_MANUAL_ENTRY_KEY, None)
    return current


def settings_json_with_pending_voice_transcript(
    settings: UserSettings,
    transcript: dict[str, Any],
    *,
    kind: str = "voice_transcript",
) -> dict[str, Any]:
    current = settings_json_with_pending_input(settings, kind)
    current[PENDING_VOICE_TRANSCRIPT_KEY] = transcript
    return current


def settings_json_without_pending_voice_transcript(settings: UserSettings) -> dict[str, Any]:
    current = settings_json_without_pending_input(settings)
    current.pop(PENDING_VOICE_TRANSCRIPT_KEY, None)
    return current
