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


class SemanticMemoryInsight(BaseModel):
    used: bool = False
    hypothesis: str | None = None
    evidence_entry_ids: list[str] = Field(default_factory=list, max_length=4)
    confidence: float = Field(default=0.0, ge=0, le=1)


class QuestionResult(BaseModel):
    question: str
    intent: str = "state_and_activity"
    rationale: str | None = None
    semantic_memory_insight: SemanticMemoryInsight = Field(default_factory=SemanticMemoryInsight)


class QuietSuggestion(BaseModel):
    should_offer: bool = False
    message: str | None = None
    reason: str | None = None
    confidence: float = Field(default=0.0, ge=0, le=1)


class ClarificationResult(BaseModel):
    question: str = ""
    options: list[str] = Field(default_factory=list, max_length=4)
    expected_gain: str | None = None
    focus: str | None = None
    should_clarify: bool = True


class ClarificationQueueReview(BaseModel):
    should_ask: bool = True
    item_ids: list[str] = Field(default_factory=list)
    question: str | None = None
    options: list[str] = Field(default_factory=list, max_length=4)
    reason: str | None = None
    expected_gain: str | None = None
    confidence: float = Field(default=0.0, ge=0, le=1)


class ObservationCadence(BaseModel):
    volatility: Literal["stable", "moving", "volatile", "sensitive", "unclear"] = "unclear"
    change_likelihood: Literal["low", "medium", "high", "unclear"] = "unclear"
    eventfulness: Literal["low", "medium", "high", "unclear"] = "unclear"
    next_checkin_min_minutes: int | None = Field(default=None, ge=5, le=360)
    next_checkin_max_minutes: int | None = Field(default=None, ge=5, le=360)
    confidence: float = Field(default=0.0, ge=0, le=1)
    reason: str | None = None


class EveningReviewPatch(BaseModel):
    entry_id: str
    patch: dict[str, Any] = Field(default_factory=dict)
    evidence: str | None = None
    confidence: float = Field(default=0.0, ge=0, le=1)


class EveningReviewQuestion(BaseModel):
    entry_id: str
    question: str
    reason: str | None = None
    evidence: str | None = None
    expected_gain: str | None = None
    options: list[str] = Field(default_factory=list, max_length=4)
    confidence: float = Field(default=0.0, ge=0, le=1)


class EveningReview(BaseModel):
    patches: list[EveningReviewPatch] = Field(default_factory=list)
    uncertain_items: list[str] = Field(default_factory=list)
    question_candidates: list[EveningReviewQuestion] = Field(default_factory=list, max_length=2)
    notes: list[str] = Field(default_factory=list)
    memory_graph_notes: list[str] = Field(default_factory=list)


class FeatureValue(BaseModel):
    value: str
    confidence: float = Field(ge=0, le=1)


class PresenceValue(BaseModel):
    present: bool | None
    confidence: float = Field(ge=0, le=1)


class EmotionSignal(BaseModel):
    label: str
    intensity_level: Literal["trace", "mild", "moderate", "strong", "overwhelming", "unclear"] = "unclear"
    intensity: float = Field(default=0.0, ge=0, le=1)
    confidence: float = Field(default=0.0, ge=0, le=1)
    evidence: str | None = None
    time_scope: Literal["current", "recent", "past_story", "dream", "mentioned_not_felt", "unclear"] = "unclear"


class AffectiveStateSignal(BaseModel):
    label: str
    intensity_level: Literal["trace", "mild", "moderate", "strong", "overwhelming", "unclear"] = "unclear"
    intensity: float = Field(default=0.0, ge=0, le=1)
    confidence: float = Field(default=0.0, ge=0, le=1)
    evidence: str | None = None
    time_scope: Literal["current", "recent", "past_story", "dream", "mentioned_not_felt", "unclear"] = "unclear"


class EntryFeatures(BaseModel):
    entry_type: Literal[
        "current_state",
        "activity_only",
        "sleep",
        "dream",
        "photo_only",
        "reply_fragment",
        "reflection",
        "social_event",
        "body_state",
        "command_or_system",
        "mixed",
        "unknown",
    ] = "unknown"
    activity_labels: list[str] = Field(default_factory=list)
    state_labels: list[str] = Field(default_factory=list)
    emotion_labels: list[str] = Field(default_factory=list)
    emotions: list[EmotionSignal] = Field(default_factory=list)
    affective_states: list[AffectiveStateSignal] = Field(default_factory=list)
    non_emotional_states: list[str] = Field(default_factory=list)
    mentioned_but_not_felt: list[str] = Field(default_factory=list)
    emotion_observation: Literal["observed", "no_current_emotion", "unclear"] = "unclear"
    emotion_transition: Literal[
        "continued", "weaker", "stronger", "shifted", "ended", "not_applicable", "unclear"
    ] = "unclear"
    emotion_transition_confidence: float = Field(default=0.0, ge=0, le=1)
    emotion_needs_clarification: bool = False
    mood: FeatureValue = Field(default_factory=lambda: FeatureValue(value="unclear", confidence=0.0))
    mood_evidence: str | None = None
    mood_reasoning_type: Literal[
        "direct_text", "weak_text", "context_inferred", "user_manual", "metadata_only", "unclear"
    ] = "unclear"
    energy: FeatureValue = Field(default_factory=lambda: FeatureValue(value="unclear", confidence=0.0))
    energy_evidence: str | None = None
    energy_reasoning_type: Literal[
        "direct_text", "weak_text", "context_inferred", "user_manual", "metadata_only", "unclear"
    ] = "unclear"
    anxiety: FeatureValue = Field(default_factory=lambda: FeatureValue(value="unclear", confidence=0.0))
    anxiety_evidence: str | None = None
    anxiety_reasoning_type: Literal[
        "direct_text", "weak_text", "context_inferred", "user_manual", "metadata_only", "unclear"
    ] = "unclear"
    used_context: bool = False
    context_inference: Literal["none", "weak", "strong"] = "none"
    should_graph_mood: bool = False
    should_graph_energy: bool = False
    observation_cadence: ObservationCadence = Field(default_factory=ObservationCadence)
    needs_clarification: bool = False
    clarification_question: str | None = None
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
    semantic_memory_insight: SemanticMemoryInsight = Field(default_factory=SemanticMemoryInsight)


class SemanticMemoryText(BaseModel):
    text: str
    graph: dict[str, Any] = Field(default_factory=dict)


class MemoryGraphNodeCandidate(BaseModel):
    label: str
    kind: str = "concept"
    aliases: list[str] = Field(default_factory=list)
    summary: str | None = None
    status: Literal["candidate", "hypothesis", "confirmed"] = "hypothesis"
    weight: float = Field(default=0.5, ge=0, le=1)
    confidence: float = Field(default=0.5, ge=0, le=1)
    evidence: str | None = None


class MemoryGraphEdgeCandidate(BaseModel):
    source_label: str
    relation_label: str
    target_label: str
    summary: str | None = None
    status: Literal["candidate", "hypothesis", "confirmed"] = "hypothesis"
    weight: float = Field(default=0.5, ge=0, le=1)
    confidence: float = Field(default=0.5, ge=0, le=1)
    evidence: str | None = None


class MemoryGraphExtraction(BaseModel):
    nodes: list[MemoryGraphNodeCandidate] = Field(default_factory=list)
    edges: list[MemoryGraphEdgeCandidate] = Field(default_factory=list)
    ignored_notes: list[str] = Field(default_factory=list)


class MemoryGraphReviewDecision(BaseModel):
    pair_id: str
    decision: Literal["same", "alias", "separate", "unsure"] = "unsure"
    canonical_node_id: str | None = None
    alias_text: str | None = None
    reason: str | None = None
    confidence: float = Field(default=0.0, ge=0, le=1)
    needs_user_confirmation: bool = False


class MemoryGraphReviewResult(BaseModel):
    decisions: list[MemoryGraphReviewDecision] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class LifeContextCandidate(BaseModel):
    category: Literal["person", "place", "project", "routine", "theme", "term", "other"] = "other"
    label: str
    hypothesis: str
    question: str
    question_type: Literal["confirm", "choice", "open", "boundary", "status", "meaning"] = "open"
    options: list[str] = Field(default_factory=list)
    why_it_matters: str | None = None
    sensitivity: Literal["normal", "sensitive"] = "normal"
    confidence: float = Field(default=0.5, ge=0, le=1)


class LifeContextExtraction(BaseModel):
    candidates: list[LifeContextCandidate] = Field(default_factory=list)


class LifeContextPruneResult(BaseModel):
    keep_item_ids: list[str] = Field(default_factory=list)
    drop_item_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class LifeContextAnswerReview(BaseModel):
    decision: Literal["store", "ask_followup", "skip"] = "ask_followup"
    normalized_answer: str | None = None
    revised_category: Literal["person", "place", "project", "routine", "theme", "term", "other"] | None = None
    revised_label: str | None = None
    revised_hypothesis: str | None = None
    followup_question: str | None = None
    reason: str | None = None
    confidence: float = Field(default=0.5, ge=0, le=1)


class LifeContextRewriteItem(BaseModel):
    id: str
    action: Literal["keep", "rewrite", "drop"] = "keep"
    category: Literal["person", "place", "project", "routine", "theme", "term", "other"] | None = None
    label: str | None = None
    answer: str | None = None
    reason: str | None = None


class LifeContextRewriteResult(BaseModel):
    items: list[LifeContextRewriteItem] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class DailyTurningPoint(BaseModel):
    entry_id: str
    title: str
    change: str
    confidence: float = Field(default=0.5, ge=0, le=1)


class DailySummary(BaseModel):
    short_text: str
    story: str
    actual_activities: list[str] = Field(default_factory=list)
    state_changes: list[str] = Field(default_factory=list)
    turning_points: list[DailyTurningPoint] = Field(default_factory=list)
    hardest_interval: str | None = None
    best_or_stablest_interval: str | None = None
    pleasant_moments: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)
    cautious_observations: list[str] = Field(default_factory=list)
    data_quality: str = "unknown"
    semantic_memory_insight: SemanticMemoryInsight = Field(default_factory=SemanticMemoryInsight)


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
    semantic_memory_insight: SemanticMemoryInsight = Field(default_factory=SemanticMemoryInsight)


class Route(BaseModel):
    model: str
    thinking: bool = False
    temperature: float = 0.35
