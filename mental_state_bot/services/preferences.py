from __future__ import annotations

from typing import Any

from mental_state_bot.db.models import UserSettings

SNAPSHOTS_PAUSED_KEY = "snapshots_paused"
CUSTOM_INTERACTION_STYLE_KEY = "custom_interaction_style"
USER_PROFILE_CONTEXT_KEY = "user_profile_context"
PENDING_INPUT_KEY = "pending_input"


def snapshots_paused(settings: UserSettings) -> bool:
    return bool((getattr(settings, "settings_json", None) or {}).get(SNAPSHOTS_PAUSED_KEY))


def settings_json_with_snapshot_pause(settings: UserSettings, paused: bool) -> dict[str, Any]:
    return {**(getattr(settings, "settings_json", None) or {}), SNAPSHOTS_PAUSED_KEY: paused}


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


def pending_input(settings: UserSettings) -> str | None:
    value = (getattr(settings, "settings_json", None) or {}).get(PENDING_INPUT_KEY)
    if value in {"custom_style", "profile_context", "correction"}:
        return value
    return None


def settings_json_with_pending_input(settings: UserSettings, kind: str | None) -> dict[str, Any]:
    current = dict(getattr(settings, "settings_json", None) or {})
    if kind is None:
        current.pop(PENDING_INPUT_KEY, None)
        return current
    if kind not in {"custom_style", "profile_context", "correction"}:
        raise ValueError(f"Unsupported pending input kind: {kind}")
    current[PENDING_INPUT_KEY] = kind
    return current


def settings_json_without_pending_input(settings: UserSettings) -> dict[str, Any]:
    return settings_json_with_pending_input(settings, None)
