"""
Integration tests for the FastAPI /analyze endpoint.

These spin up the app with TestClient (no real HTTP, no ML models needed).
The agent fails to load (no model files in the test environment), so all
responses come from the keyword fallback -- which is exactly what we want
to test here.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fastapi.testclient import TestClient
    from app_web import app
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


pytestmark = pytest.mark.skipif(
    not HAS_FASTAPI,
    reason="fastapi not installed -- skipping API tests"
)


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_health_has_agent_loaded_field(self, client):
        r = client.get("/health")
        assert "agent_loaded" in r.json()


class TestAnalyzeEndpoint:
    def test_basic_request_returns_200(self, client):
        r = client.post("/analyze", json={"message": "songs keep buffering"})
        assert r.status_code == 200

    def test_response_has_required_fields(self, client):
        r = client.post("/analyze", json={"message": "I was charged twice"})
        body = r.json()
        for field in ["intent", "confidence", "reply", "should_escalate", "esc_score",
                      "signals", "scores", "latency_ms"]:
            assert field in body, f"Missing field: {field}"

    def test_empty_message_returns_400(self, client):
        r = client.post("/analyze", json={"message": "   "})
        assert r.status_code == 400

    def test_confidence_in_range(self, client):
        r = client.post("/analyze", json={"message": "please add crossfade"})
        c = r.json()["confidence"]
        assert 0.0 <= c <= 1.0

    def test_esc_score_in_range(self, client):
        r = client.post("/analyze", json={"message": "app keeps crashing"})
        s = r.json()["esc_score"]
        assert 0.0 <= s <= 1.0

    def test_scores_has_seven_intents(self, client):
        r = client.post("/analyze", json={"message": "songs buffering"})
        assert len(r.json()["scores"]) == 7

    def test_prior_turn_accepted(self, client):
        r = client.post("/analyze", json={
            "message": "I already paid",
            "prior_turn": "my account is locked"
        })
        assert r.status_code == 200

    def test_billing_message_escalates(self, client):
        r = client.post("/analyze", json={"message": "you charged me and I want a refund"})
        assert r.json()["should_escalate"] is True

    def test_feature_request_does_not_escalate(self, client):
        r = client.post("/analyze", json={"message": "please add a sleep timer feature"})
        assert r.json()["should_escalate"] is False

    def test_reply_is_non_empty_string(self, client):
        r = client.post("/analyze", json={"message": "can't log in"})
        reply = r.json()["reply"]
        assert isinstance(reply, str)
        assert len(reply) > 10

    def test_latency_ms_positive(self, client):
        r = client.post("/analyze", json={"message": "hi"})
        assert r.json()["latency_ms"] > 0


class TestEdgeCases:
    def test_very_long_message(self, client):
        long_msg = "songs buffering " * 30
        r = client.post("/analyze", json={"message": long_msg})
        assert r.status_code == 200

    def test_unicode_message(self, client):
        r = client.post("/analyze", json={"message": "musique ne joue pas \U0001f3b5"})
        assert r.status_code == 200

    def test_message_with_only_special_chars(self, client):
        r = client.post("/analyze", json={"message": "!!!! ??? ###"})
        assert r.status_code == 200
        assert r.json()["intent"] == "general_inquiry"
