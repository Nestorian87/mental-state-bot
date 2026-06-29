from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Usage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None


class ModelCallResult(BaseModel):
    provider: str
    model: str
    task_name: str
    text: str
    data: dict[str, Any] | None = None
    usage: Usage = Field(default_factory=Usage)
    latency_ms: int | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class QuestionResult(BaseModel):
    question: str
    intent: str = "state_and_activity"
    rationale: str | None = None


class ClarificationResult(BaseModel):
    question: str
    expected_gain: str | None = None
    should_clarify: bool = True


class FeatureValue(BaseModel):
    value: str
    confidence: float = Field(ge=0, le=1)


class PresenceValue(BaseModel):
    present: bool | None
    confidence: float = Field(ge=0, le=1)


class EntryFeatures(BaseModel):
    activity_labels: list[str] = Field(default_factory=list)
    state_labels: list[str] = Field(default_factory=list)
    mood: FeatureValue = Field(default_factory=lambda: FeatureValue(value="unclear", confidence=0.0))
    energy: FeatureValue = Field(default_factory=lambda: FeatureValue(value="unclear", confidence=0.0))
    anxiety: FeatureValue = Field(default_factory=lambda: FeatureValue(value="unclear", confidence=0.0))
    emptiness: PresenceValue = Field(default_factory=lambda: PresenceValue(present=None, confidence=0.0))
    avoidance: PresenceValue = Field(default_factory=lambda: PresenceValue(present=None, confidence=0.0))
    rumination: PresenceValue = Field(default_factory=lambda: PresenceValue(present=None, confidence=0.0))
    inability_to_start: PresenceValue = Field(
        default_factory=lambda: PresenceValue(present=None, confidence=0.0)
    )
    body_signals: list[str] = Field(default_factory=list)
    social_activity: FeatureValue = Field(
        default_factory=lambda: FeatureValue(value="unclear", confidence=0.0)
    )
    pleasant_moments: list[str] = Field(default_factory=list)
    what_helped: list[str] = Field(default_factory=list)
    what_worsened: list[str] = Field(default_factory=list)
    transition: str | None = None
    data_quality: Literal["empty", "very_low", "partial", "enough", "rich"] = "partial"
    uncertainty_notes: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0, le=1)


class MicroSummary(BaseModel):
    text: str
    session_complete: bool = True


class SemanticMemoryText(BaseModel):
    text: str


class DailySummary(BaseModel):
    short_text: str
    story: str
    actual_activities: list[str] = Field(default_factory=list)
    state_changes: list[str] = Field(default_factory=list)
    hardest_interval: str | None = None
    best_or_stablest_interval: str | None = None
    pleasant_moments: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)
    cautious_observations: list[str] = Field(default_factory=list)
    data_quality: str = "unknown"


class PeriodSummary(BaseModel):
    short_text: str
    period_story: str
    repeated_patterns: list[str] = Field(default_factory=list)
    changes_vs_previous_period: list[str] = Field(default_factory=list)
    activity_state_patterns: list[str] = Field(default_factory=list)
    what_helped: list[str] = Field(default_factory=list)
    what_worsened: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)
    notable_days: list[str] = Field(default_factory=list)
    cautious_observations: list[str] = Field(default_factory=list)
    data_quality: str = "unknown"


class Route(BaseModel):
    model: str
    thinking: bool = False
    temperature: float = 0.35
