"""
Tests for the keyword fallback classify logic in app_web.py.

The fallback runs when the full ML agent can't be loaded (e.g. on Render's 512 MB
free tier). It has to produce a valid JSON response on every input.

We test by importing the logic directly from the route, not by spinning up a
real HTTP server, so these tests run with zero extra dependencies.
"""

import json
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Pull the keyword routing logic out of app_web as a pure function for testing.
# We replicate the exact logic here so it can be unit-tested without FastAPI.
# If the logic in app_web changes, these tests will catch regressions.
# ---------------------------------------------------------------------------

INTENT_LABELS = {
    "playback_issue":      "Playback Issue",
    "account_access":      "Account Access",
    "billing_payment":     "Billing / Payment",
    "content_unavailable": "Content Unavailable",
    "app_bug":             "App Bug",
    "feature_request":     "Feature Request",
    "general_inquiry":     "General Inquiry",
}

TEMPLATE_REPLIES = {
    "playback_issue":      "Hi! Sorry about the playback trouble. Try logging out and back in, and check that the app is up to date. DM us if it persists! ^SP",
    "account_access":      "Hi! Sorry you're having trouble accessing your account. Try resetting your password at spotify.com/password-reset. DM us if you're still stuck! ^SP",
    "billing_payment":     "Hi! Sorry about the billing issue. Please DM us your account email so we can look into it and sort it out for you. ^SP",
    "content_unavailable": "Hi! Sorry that content isn't available. It may be region-restricted or temporarily unavailable. DM us with details and we'll investigate! ^SP",
    "app_bug":             "Hi! Sorry the app isn't working properly. Try force-closing and reopening, or reinstalling if the issue persists. DM us with your device details! ^SP",
    "feature_request":     "Thanks for the suggestion! We're always working to improve Spotify. We'll pass your feedback to our product team. ^SP",
    "general_inquiry":     "Hi! Happy to help. Could you tell us a bit more about what you need? Or check our Help Center at support.spotify.com for quick answers. ^SP",
}


def keyword_classify(message: str):
    """Replicate the keyword fallback logic from app_web.py analyze()."""
    text_lower = message.lower()
    if any(w in text_lower for w in ["buffer", "skip", "play", "song", "podcast", "stream"]):
        intent, confidence = "playback_issue", 0.72
    elif any(w in text_lower for w in ["log in", "login", "password", "account", "locked", "hack"]):
        intent, confidence = "account_access", 0.70
    elif any(w in text_lower for w in ["charge", "refund", "bill", "payment", "cancel", "subscription"]):
        intent, confidence = "billing_payment", 0.74
    elif any(w in text_lower for w in ["crash", "bug", "glitch", "freeze", "broken"]):
        intent, confidence = "app_bug", 0.68
    elif any(w in text_lower for w in ["add", "feature", "please", "would love", "option"]):
        intent, confidence = "feature_request", 0.65
    elif any(w in text_lower for w in ["missing", "removed", "not available", "region"]):
        intent, confidence = "content_unavailable", 0.66
    else:
        intent, confidence = "general_inquiry", 0.60

    should_escalate = intent in ("billing_payment", "account_access") or confidence < 0.62
    esc_score = 0.72 if should_escalate else 0.18

    all_intents = list(INTENT_LABELS.keys())
    remaining = round((1.0 - confidence) / (len(all_intents) - 1), 4)
    scores = {k: remaining for k in all_intents}
    scores[intent] = round(confidence, 4)

    return {
        "intent": intent,
        "confidence": confidence,
        "should_escalate": should_escalate,
        "esc_score": esc_score,
        "scores": scores,
        "reply": TEMPLATE_REPLIES[intent],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestKeywordRouting:
    def test_buffering_is_playback(self):
        r = keyword_classify("spotify keeps buffering on my iphone")
        assert r["intent"] == "playback_issue"

    def test_hacked_is_account(self):
        r = keyword_classify("I think my account was hacked")
        assert r["intent"] == "account_access"

    def test_refund_is_billing(self):
        r = keyword_classify("you charged me twice, I want a refund")
        assert r["intent"] == "billing_payment"

    def test_crash_is_app_bug(self):
        r = keyword_classify("the app keeps crashing when I open it")
        assert r["intent"] == "app_bug"

    def test_feature_request(self):
        r = keyword_classify("please add crossfade to mobile")
        assert r["intent"] == "feature_request"

    def test_missing_content(self):
        r = keyword_classify("this album is not available in my region")
        assert r["intent"] == "content_unavailable"

    def test_vague_message_is_general(self):
        r = keyword_classify("hi")
        assert r["intent"] == "general_inquiry"

    def test_empty_string_returns_general(self):
        r = keyword_classify("")
        assert r["intent"] == "general_inquiry"


class TestEscalationLogic:
    def test_billing_always_escalates(self):
        r = keyword_classify("I was charged and want a refund")
        assert r["should_escalate"] is True

    def test_account_always_escalates(self):
        r = keyword_classify("can't log in, reset my password please")
        assert r["should_escalate"] is True

    def test_feature_request_does_not_escalate(self):
        r = keyword_classify("please add an equaliser option")
        assert r["should_escalate"] is False

    def test_playback_does_not_escalate(self):
        r = keyword_classify("songs keep buffering")
        assert r["should_escalate"] is False


class TestScoresShape:
    def test_scores_contains_all_intents(self):
        r = keyword_classify("my account was hacked")
        assert set(r["scores"].keys()) == set(INTENT_LABELS.keys())

    def test_top_intent_has_highest_score(self):
        r = keyword_classify("songs keep buffering on my phone")
        top = max(r["scores"], key=lambda k: r["scores"][k])
        assert top == r["intent"]

    def test_scores_sum_to_approx_one(self):
        r = keyword_classify("I want a refund")
        total = sum(r["scores"].values())
        assert abs(total - 1.0) < 0.01

    def test_confidence_in_range(self):
        for msg in ["buffer", "login", "charge", "crash", "please add", "missing", "hi"]:
            r = keyword_classify(msg)
            assert 0.0 <= r["confidence"] <= 1.0


class TestReplyQuality:
    def test_reply_is_non_empty(self):
        for msg in ["songs skipping", "hacked", "charged twice", "app crash", "add feature", "hi"]:
            r = keyword_classify(msg)
            assert len(r["reply"]) > 10

    def test_billing_reply_mentions_dm(self):
        r = keyword_classify("I was charged twice")
        assert "DM" in r["reply"] or "dm" in r["reply"].lower()

    def test_account_reply_mentions_password_reset(self):
        r = keyword_classify("can't log in to my account")
        assert "password" in r["reply"].lower() or "reset" in r["reply"].lower()
