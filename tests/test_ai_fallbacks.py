from __future__ import annotations

from mental_state_bot.ai.service import _fallback_features


def test_fallback_features_do_not_keyword_interpret_user_text() -> None:
    features = _fallback_features("лежу, залип, порожнеча і не можу почати")

    assert features.activity_labels == []
    assert features.state_labels == []
    assert features.data_quality in {"partial", "very_low"}
    assert features.uncertainty_notes == [
        "AI feature extraction unavailable; no keyword-based interpretation was applied"
    ]
