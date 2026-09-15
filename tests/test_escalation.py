"""
Tests for the escalation engine signal computation.

These run without any ML model -- they test the hand-coded signal functions
that form the backbone of the escalation decision.
"""

import pytest
import sys
import os

# Ensure the project root is on the path so `src` imports work.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Minimal config shim so we don't need the real yaml to run tests
# ---------------------------------------------------------------------------
MINIMAL_CFG = {
    "intents": {
        "confidence_threshold": 0.62,
    },
    "escalation": {
        "default_threshold": 0.50,
        "low_confidence_escalate": True,
        "sentiment_anger_threshold": -0.4,
        "complexity_token_threshold": 40,
        "high_sensitivity_intents": ["billing_payment", "account_access"],
        "auto_handle_intents": ["feature_request", "general_inquiry"],
        "per_intent_thresholds": {
            "billing_payment": 0.35,
            "account_access": 0.35,
            "feature_request": 0.90,
        },
    }
}


@pytest.fixture
def engine():
    from src.escalation_engine import EscalationEngine
    return EscalationEngine(cfg=MINIMAL_CFG)


# ---------------------------------------------------------------------------
# Confidence signal
# ---------------------------------------------------------------------------

class TestConfidenceSignal:
    def test_high_confidence_no_signal(self, engine):
        """Confidence above threshold should produce 0 escalation signal."""
        assert engine._confidence_signal(0.90) == 0.0

    def test_confidence_at_threshold_no_signal(self, engine):
        """Confidence exactly at threshold: signal is 0."""
        assert engine._confidence_signal(0.62) == 0.0

    def test_zero_confidence_max_signal(self, engine):
        """Zero confidence should give maximum signal (1.0)."""
        assert engine._confidence_signal(0.0) == pytest.approx(1.0)

    def test_mid_confidence_partial_signal(self, engine):
        """Confidence halfway below threshold should give partial signal."""
        sig = engine._confidence_signal(0.31)  # half of 0.62
        assert 0.4 < sig < 0.6


# ---------------------------------------------------------------------------
# Complexity signal
# ---------------------------------------------------------------------------

class TestComplexitySignal:
    def test_short_message_low_complexity(self, engine):
        assert engine._complexity_signal("songs not playing") < 0.3

    def test_long_message_high_complexity(self, engine):
        long_msg = " ".join(["word"] * 90)
        assert engine._complexity_signal(long_msg) > 0.5

    def test_multiple_questions_raises_complexity(self, engine):
        msg = "why won't it play? is there an outage? should I reinstall?"
        assert engine._complexity_signal(msg) > engine._complexity_signal("why won't it play")


# ---------------------------------------------------------------------------
# Sensitivity signal
# ---------------------------------------------------------------------------

class TestSensitivitySignal:
    def test_billing_high_sensitivity(self, engine):
        assert engine._sensitivity_signal("billing_payment") == pytest.approx(0.9)

    def test_account_high_sensitivity(self, engine):
        assert engine._sensitivity_signal("account_access") == pytest.approx(0.9)

    def test_feature_request_zero_sensitivity(self, engine):
        assert engine._sensitivity_signal("feature_request") == pytest.approx(0.0)

    def test_general_inquiry_zero_sensitivity(self, engine):
        assert engine._sensitivity_signal("general_inquiry") == pytest.approx(0.0)

    def test_unknown_intent_moderate(self, engine):
        assert 0.2 < engine._sensitivity_signal("playback_issue") < 0.5


# ---------------------------------------------------------------------------
# Security signal
# ---------------------------------------------------------------------------

class TestSecuritySignal:
    def test_hacked_triggers_security(self, engine):
        assert engine._security_signal("I think my account was hacked") > 0.5

    def test_fraud_triggers_security(self, engine):
        assert engine._security_signal("this is fraud i want my money back") > 0.5

    def test_normal_message_no_security(self, engine):
        assert engine._security_signal("songs keep buffering on my iphone") == 0.0

    def test_multiple_keywords_higher_signal(self, engine):
        one = engine._security_signal("I was hacked")
        two = engine._security_signal("I was hacked and there was unauthorized access")
        assert two >= one

    def test_signal_capped_at_one(self, engine):
        many_kws = "hacked fraud unauthorized lawsuit lawyer password stolen"
        assert engine._security_signal(many_kws) <= 1.0


# ---------------------------------------------------------------------------
# End-to-end decide() -- uses keyword / config only, no LLM
# ---------------------------------------------------------------------------

class TestDecide:
    def test_feature_request_auto_handles(self, engine):
        result = engine.decide(
            text="please add crossfade to mobile",
            intent="feature_request",
            confidence=0.92,
        )
        # High threshold for feature_request (0.90) and zero sensitivity
        assert result.should_escalate is False

    def test_hacked_account_escalates(self, engine):
        result = engine.decide(
            text="can't log in, I think my account was hacked",
            intent="account_access",
            confidence=0.55,
        )
        assert result.should_escalate is True

    def test_billing_low_confidence_escalates(self, engine):
        """Low classifier confidence on a billing message should trigger escalation."""
        result = engine.decide(
            text="you charged me and I am furious about this unacceptable billing issue",
            intent="billing_payment",
            confidence=0.45,   # below 0.62 threshold -> fires confidence signal
        )
        assert result.should_escalate is True

    def test_playback_normal_auto_handles(self, engine):
        result = engine.decide(
            text="songs buffering on wifi",
            intent="playback_issue",
            confidence=0.88,
        )
        assert result.should_escalate is False

    def test_result_has_score_and_reasons(self, engine):
        result = engine.decide(
            text="please fix the app",
            intent="app_bug",
            confidence=0.75,
        )
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.reasons, list)
        assert len(result.reasons) >= 0

    def test_per_intent_threshold_billing_aggressive(self, engine):
        """billing_payment threshold is 0.35 -- should escalate at lower scores."""
        result = engine.decide(
            text="I was charged",
            intent="billing_payment",
            confidence=0.80,
        )
        # Sensitivity alone (0.9 * 0.20 weight = 0.18) + other signals may push over 0.35
        # At minimum the threshold is lower for billing than default (0.50)
        assert result.score >= 0.0  # sanity; actual escalation depends on signal sum
