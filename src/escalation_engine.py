"""
Escalation engine — decides whether the agent can handle a message automatically
or whether it should be routed to a human.

Decision is a multi-signal fusion:
  Signal 1 — Classifier confidence       (low confidence = uncertain, escalate)
  Signal 2 — Sentiment polarity          (strong anger = escalate)
  Signal 3 — Message complexity          (very long or multi-issue = escalate)
  Signal 4 — Topic sensitivity           (billing, account security = higher bar)
  Signal 5 — PII / security markers      (account numbers, passwords mentioned)

We combine these into a weighted score in [0, 1].
If score >= 0.5 → escalate; else → auto-handle.

Design choice: escalation recall >> precision. A false negative (not escalating
when we should) is worse than a false positive (escalating unnecessarily), so
we bias the threshold conservatively and err toward human review.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from src.utils import get_logger, load_config

# ---------------------------------------------------------------------------
# Sentiment backend — prefer cardiffnlp/twitter-roberta-base-sentiment
# (trained on 124M tweets, handles sarcasm much better than TextBlob).
# Falls back to TextBlob if transformers isn't installed.
# ---------------------------------------------------------------------------
_sentiment_pipeline = None

def _load_sentiment():
    global _sentiment_pipeline
    if _sentiment_pipeline is not None:
        return _sentiment_pipeline
    try:
        from transformers import pipeline as hf_pipeline
        _sentiment_pipeline = hf_pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            top_k=None,
            truncation=True,
            max_length=512,
        )
        log.info("Loaded twitter-roberta-base-sentiment-latest for sentiment.")
    except Exception:
        _sentiment_pipeline = "textblob"
        log.warning("transformers not available; falling back to TextBlob sentiment.")
    return _sentiment_pipeline

log = get_logger(__name__)

# Regex patterns that strongly suggest the customer is dealing with a
# sensitive or security-related issue.
SECURITY_PATTERNS = [
    r"\bpassword\b",
    r"\bhacked?\b",
    r"\bunauthorized\b",
    r"\bstolen\b",
    r"\bfraud\b",
    r"\bchargeback\b",
    r"\brefund\b.*\bargument\b",
    r"\blawyer\b",
    r"\bsue\b",
    r"\bATM\b",
    r"\bcredit card\b",
]
SECURITY_RE = re.compile("|".join(SECURITY_PATTERNS), re.IGNORECASE)

# Words that amplify negative sentiment
AMPLIFIER_WORDS = {
    "extremely", "very", "incredibly", "absolutely", "completely",
    "furious", "unacceptable", "disgusting", "terrible", "worst",
    "never", "always", "ruined", "useless", "awful",
}

ESCALATION_REASONS = {
    "low_confidence": "Classifier is uncertain about intent — needs human review.",
    "high_anger": "Customer message contains strong negative sentiment.",
    "high_complexity": "Message is long/multi-issue — risk of missed context.",
    "sensitive_intent": "Intent category requires account/billing data access.",
    "security_signal": "Message contains security or legal keywords.",
    "amplified_sentiment": "Multiple strong-negative amplifiers detected.",
}

AUTO_REASONS = {
    "high_confidence": "Classifier is confident about intent.",
    "positive_sentiment": "Customer sentiment is neutral or positive.",
    "simple_message": "Short, single-issue message.",
    "safe_intent": "Intent category is routinely auto-handled.",
}


class EscalationDecision(NamedTuple):
    should_escalate: bool
    score: float                  # 0 = definitely auto, 1 = definitely escalate
    reasons: list[str]            # human-readable explanation signals
    signal_breakdown: dict        # raw signal values for debugging


class EscalationEngine:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        esc = self.cfg["escalation"]
        self.sentiment_threshold = esc["sentiment_anger_threshold"]
        self.complexity_threshold = esc["complexity_token_threshold"]
        self.high_sensitivity_intents = set(esc["high_sensitivity_intents"])
        self.auto_handle_intents = set(esc["auto_handle_intents"])
        self.low_confidence_escalate = esc["low_confidence_escalate"]
        self.confidence_threshold = self.cfg["intents"]["confidence_threshold"]

        # Signal weights (must sum to 1.0)
        self._weights = {
            "confidence":   0.25,
            "sentiment":    0.25,
            "complexity":   0.15,
            "sensitivity":  0.20,
            "security":     0.15,
        }

    def decide(
        self,
        text: str,
        intent: str,
        confidence: float,
    ) -> EscalationDecision:
        """
        Run all signals and return a structured escalation decision.
        """
        signals = self._compute_signals(text, intent, confidence)
        score = sum(
            self._weights[k] * v for k, v in signals.items()
        )
        score = max(0.0, min(1.0, score))

        reasons: list[str] = []
        should_escalate = score >= 0.5

        # Determine readable reasons
        if signals["confidence"] > 0.5:
            reasons.append(ESCALATION_REASONS["low_confidence"])
        if signals["sentiment"] > 0.6:
            reasons.append(ESCALATION_REASONS["high_anger"])
        if signals["complexity"] > 0.5:
            reasons.append(ESCALATION_REASONS["high_complexity"])
        if signals["sensitivity"] > 0.5:
            reasons.append(ESCALATION_REASONS["sensitive_intent"])
        if signals["security"] > 0.5:
            reasons.append(ESCALATION_REASONS["security_signal"])

        if not should_escalate:
            if signals["confidence"] < 0.3:
                reasons.append(AUTO_REASONS["high_confidence"])
            if signals["sentiment"] < 0.2:
                reasons.append(AUTO_REASONS["positive_sentiment"])
            if signals["complexity"] < 0.3:
                reasons.append(AUTO_REASONS["simple_message"])
            if signals["sensitivity"] < 0.1:
                reasons.append(AUTO_REASONS["safe_intent"])

        if not reasons:
            reasons.append(
                "Aggregate score %.2f (%s threshold of 0.50)."
                % (score, "above" if should_escalate else "below")
            )

        return EscalationDecision(
            should_escalate=should_escalate,
            score=round(score, 4),
            reasons=reasons,
            signal_breakdown={k: round(v, 4) for k, v in signals.items()},
        )

    def _compute_signals(
        self, text: str, intent: str, confidence: float
    ) -> dict[str, float]:
        """
        Each signal returns a float in [0, 1] where 1 = "escalate strongly".
        """
        return {
            "confidence": self._confidence_signal(confidence),
            "sentiment":  self._sentiment_signal(text),
            "complexity": self._complexity_signal(text),
            "sensitivity": self._sensitivity_signal(intent),
            "security":   self._security_signal(text),
        }

    def _confidence_signal(self, confidence: float) -> float:
        """Low confidence -> high escalation signal."""
        if not self.low_confidence_escalate:
            return 0.0
        # Linear: conf=threshold -> 0.5 signal; conf=0 -> 1.0 signal
        if confidence >= self.confidence_threshold:
            return 0.0
        return 1.0 - (confidence / self.confidence_threshold)

    def _sentiment_signal(self, text: str) -> float:
        """
        Negative polarity + amplifier words -> high escalation signal.
        Uses twitter-roberta-base-sentiment when available; falls back to TextBlob.
        """
        pipeline = _load_sentiment()

        if pipeline != "textblob":
            # RoBERTa returns [{label, score}, ...] for each label
            results = pipeline(text[:512])
            # results is a list of dicts: [{"label": "negative", "score": 0.9}, ...]
            label_map = {r["label"].lower(): r["score"] for r in results[0]}
            neg = label_map.get("negative", 0.0)
            neu = label_map.get("neutral", 0.0)
            # map negative prob to [0, 1] escalation signal
            # high negative -> high signal; neutral is mild; positive -> 0
            base_signal = neg * 0.9 + neu * 0.15
        else:
            from textblob import TextBlob
            polarity = TextBlob(text).sentiment.polarity  # [-1, 1]
            base_signal = max(0.0, (0.3 - polarity) / 1.3)

        # Amplifier bonus
        words = set(text.lower().split())
        n_amplifiers = len(words & AMPLIFIER_WORDS)
        amplifier_boost = min(0.3, n_amplifiers * 0.08)

        return min(1.0, base_signal + amplifier_boost)

    def _complexity_signal(self, text: str) -> float:
        """Long messages with multiple questions are harder to auto-handle."""
        tokens = text.split()
        n_tokens = len(tokens)
        n_questions = text.count("?")

        length_score = min(1.0, n_tokens / (self.complexity_threshold * 2))
        question_score = min(0.5, n_questions * 0.15)
        return min(1.0, length_score + question_score)

    def _sensitivity_signal(self, intent: str) -> float:
        """Certain intents need human access to systems."""
        if intent in self.high_sensitivity_intents:
            return 0.9
        if intent in self.auto_handle_intents:
            return 0.0
        return 0.3   # default moderate sensitivity

    def _security_signal(self, text: str) -> float:
        """Hard keywords that suggest security or legal sensitivity."""
        if SECURITY_RE.search(text):
            # More matches = higher signal
            matches = SECURITY_RE.findall(text)
            return min(1.0, 0.6 + len(matches) * 0.1)
        return 0.0


if __name__ == "__main__":
    engine = EscalationEngine()
    test_cases = [
        ("my spotify keeps buffering", "playback_issue", 0.85),
        ("I was charged twice and I am absolutely furious this is unacceptable", "billing_payment", 0.72),
        ("can't log in and I think my account was hacked please help", "account_access", 0.55),
        ("please add a crossfade feature", "feature_request", 0.91),
        ("hi how do i get student discount", "general_inquiry", 0.88),
        ("i've had 5 issues this week with the app and i want to cancel and get a refund now", "billing_payment", 0.48),
    ]
    for text, intent, conf in test_cases:
        dec = engine.decide(text, intent, conf)
        action = "ESCALATE" if dec.should_escalate else "AUTO"
        print(f"[{action}] score={dec.score:.2f} | {text[:60]}")
        for r in dec.reasons:
            print(f"   - {r}")
        print()
