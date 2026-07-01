from __future__ import annotations

from typing import Any

from mental_state_bot.db.models import UserSettings

SNAPSHOTS_PAUSED_KEY = "snapshots_paused"
CUSTOM_INTERACTION_STYLE_KEY = "custom_interaction_style"
USER_PROFILE_CONTEXT_KEY = "user_profile_context"
PENDING_INPUT_KEY = "pending_input"
PENDING_VOICE_TRANSCRIPT_KEY = "pending_voice_transcript"
PENDING_MANUAL_ENTRY_KEY = "pending_manual_entry"
PENDING_CORRECTION_ENTRY_ID_KEY = "pending_correction_entry_id"
PENDING_INPUT_KINDS = {
    "custom_style",
    "profile_context",
    "correction",
    "voice_transcript",
    "voice_transcript_fix",
}


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
