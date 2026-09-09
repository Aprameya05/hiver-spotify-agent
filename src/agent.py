"""
Spotify Support Agent — orchestrates classify -> generate -> escalate.

Exposes a single `handle(message)` call that returns a structured AgentResponse.
Designed to be stateless across messages (each call is independent) so it can
run in parallel without locking.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import NamedTuple

from src.escalation_engine import EscalationDecision, EscalationEngine
from src.intent_classifier import IntentClassifier, IntentPrediction
from src.reply_generator import GeneratedReply, ReplyGenerator
from src.utils import clean_tweet, get_logger, load_config

log = get_logger(__name__)


class AgentResponse(NamedTuple):
    # Input
    original_message: str
    cleaned_message: str

    # Stage 1 — intent
    intent: IntentPrediction

    # Stage 2 — reply
    reply: GeneratedReply

    # Stage 3 — escalation
    escalation: EscalationDecision

    # Metadata
    latency_ms: float


class SpotifyAgent:
    """
    Full pipeline: clean -> classify -> retrieve+generate -> escalate.

    Usage:
        agent = SpotifyAgent()
        agent.warm_up()           # load models once
        response = agent.handle("my app keeps crashing")
    """

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.classifier = IntentClassifier(cfg)
        self.reply_gen = ReplyGenerator(cfg)
        self.escalation = EscalationEngine(cfg)
        self._warmed_up = False

    def warm_up(
        self,
        qa_pairs: list[dict] | None = None,
        golden_path: Path | None = None,
        force_rebuild_index: bool = False,
    ) -> None:
        """
        Load or build all models. Call once before serving.
        - If FAISS index exists on disk: load it.
        - Else if qa_pairs provided: build index.
        - Classifier: load from disk if saved, else train from seeds.
        """
        log.info("Warming up Spotify agent …")

        # Classifier
        clf_path = Path(self.cfg["paths"]["models_dir"]) / "intent_clf.joblib"
        if clf_path.exists():
            self.classifier.load()
        elif golden_path and Path(golden_path).exists():
            from src.intent_classifier import build_and_save_classifier
            self.classifier = build_and_save_classifier(golden_path, self.cfg)
        else:
            self.classifier.fit_from_seeds()
            self.classifier.save()

        # Reply generator index
        index_path = Path(self.cfg["paths"]["index_path"])
        if index_path.exists() and not force_rebuild_index:
            self.reply_gen.load_index()
        elif qa_pairs:
            self.reply_gen.build_index(qa_pairs)
        else:
            log.warning(
                "No FAISS index found and no qa_pairs provided. "
                "Reply generation will use fallback mode."
            )

        self._warmed_up = True
        log.info("Agent ready.")

    def handle(self, message: str) -> AgentResponse:
        """
        Process a single customer message end-to-end.
        """
        t0 = time.perf_counter()

        cleaned = clean_tweet(message)

        # Stage 1: classify
        intent_pred = self.classifier.predict(cleaned)
        log.debug("Intent: %s (%.2f)", intent_pred.intent, intent_pred.confidence)

        # Stage 2: generate reply
        if self.reply_gen.index is not None:
            generated = self.reply_gen.generate_with_retrieval(cleaned, intent_pred.intent)
        else:
            # Fallback: zero-shot generation without retrieval
            generated = self.reply_gen.generate(cleaned, intent_pred.intent, retrieved=[])

        # Stage 3: escalation
        esc_decision = self.escalation.decide(
            cleaned, intent_pred.intent, intent_pred.confidence
        )

        latency_ms = (time.perf_counter() - t0) * 1000

        return AgentResponse(
            original_message=message,
            cleaned_message=cleaned,
            intent=intent_pred,
            reply=generated,
            escalation=esc_decision,
            latency_ms=round(latency_ms, 1),
        )

    def handle_batch(self, messages: list[str]) -> list[AgentResponse]:
        return [self.handle(msg) for msg in messages]

    def format_response(self, response: AgentResponse) -> str:
        """Pretty-print a response for CLI output."""
        esc = "ESCALATE TO HUMAN" if response.escalation.should_escalate else "AUTO-HANDLE"
        lines = [
            f"Message    : {response.original_message}",
            f"Intent     : {response.intent.intent} (confidence: {response.intent.confidence:.1%})",
            f"Decision   : {esc} (score: {response.escalation.score:.2f})",
            f"Reply draft:",
            f"  {response.reply.reply}",
            f"Escalation reasons:",
        ]
        for r in response.escalation.reasons:
            lines.append(f"  - {r}")
        lines.append(f"Latency: {response.latency_ms:.0f}ms")
        return "\n".join(lines)


if __name__ == "__main__":
    agent = SpotifyAgent()
    agent.warm_up()

    tests = [
        "spotify keeps buffering on my iphone, tried reinstalling twice",
        "you charged me again even though i cancelled. i want a refund NOW",
        "hi is there a student discount for spotify premium",
        "please add a crossfade feature to the mobile app",
        "can't log into my account and i think someone hacked it",
    ]
    for msg in tests:
        resp = agent.handle(msg)
        print("\n" + "="*70)
        print(agent.format_response(resp))
