from __future__ import annotations

CANONICAL_EMOTIONS: tuple[str, ...] = (
    "страх",
    "тривога",
    "сум",
    "злість",
    "огида",
    "сором",
    "провина",
    "розчарування",
    "образа",
    "радість",
    "надія",
    "інтерес",
    "ніжність",
    "вдячність",
    "гордість",
    "здивування",
)

# These are meaningful affective experiences, but deliberately not charted as
# discrete emotions. Keeping the category separate prevents energy, mood and
# social states from quietly becoming another emotion series.
CANONICAL_AFFECTIVE_STATES: tuple[str, ...] = (
    "самотність",
    "безнадія",
    "порожнеча",
    "спокій",
    "полегшення",
    "задоволення",
    "натхнення",
    "розгубленість",
    "нудьга",
    "напруга",
    "збудження",
)

EMOTION_INTENSITY_VALUES: dict[str, float] = {
    "trace": 0.15,
    "mild": 0.30,
    "moderate": 0.55,
    "strong": 0.80,
    "overwhelming": 1.0,
}
EMOTION_INTENSITY_LEVELS: tuple[str, ...] = (*EMOTION_INTENSITY_VALUES.keys(), "unclear")

CONTROLLED_EMOTION_TEXT = ", ".join(CANONICAL_EMOTIONS)
CONTROLLED_AFFECTIVE_STATE_TEXT = ", ".join(CANONICAL_AFFECTIVE_STATES)

EMOTION_COLORS: dict[str, tuple[int, int, int]] = {
    "радість": (245, 158, 11),
    "тривога": (239, 68, 68),
    "страх": (220, 38, 38),
    "сум": (59, 130, 246),
    "злість": (190, 18, 60),
    "огида": (22, 163, 74),
    "сором": (168, 85, 247),
    "провина": (126, 34, 206),
    "розчарування": (99, 102, 241),
    "образа": (244, 63, 94),
    "надія": (34, 197, 94),
    "інтерес": (249, 115, 22),
    "ніжність": (236, 72, 153),
    "вдячність": (132, 204, 22),
    "гордість": (217, 119, 6),
    "здивування": (6, 182, 212),
}
