"""
Intent classifier for Spotify customer messages.

Two-stage design:
  Stage 1 — fast: cosine similarity against per-intent prototype embeddings
             (mean of labelled examples, L2-normalised).
  Stage 2 — refinement: if max cosine similarity < confidence_threshold,
             ask the LLM to adjudicate (costly, but rare).

Intent taxonomy (defined from data exploration + LLM-guided cluster naming):
  0  playback_issue       — song/podcast won't play, buffering, skipping
  1  account_access       — login failures, password reset, account locked
  2  billing_payment      — charges, refunds, subscription, premium
  3  content_unavailable  — tracks/albums/podcasts missing or region-locked
  4  app_bug              — crashes, UI glitches, sync failures (non-playback)
  5  feature_request      — suggestions, missing features, wishlist
  6  general_inquiry      — greetings, thanks, off-topic, vague
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import NamedTuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

from src.utils import ensure_dir, get_logger, load_config, timeit

log = get_logger(__name__)

INTENT_LABELS = [
    "playback_issue",
    "account_access",
    "billing_payment",
    "content_unavailable",
    "app_bug",
    "feature_request",
    "general_inquiry",
]

# Seed examples per intent — used to initialise prototype embeddings before
# any real labelled data is available. These are representative phrasings
# extracted from the dataset during exploration.
SEED_EXAMPLES: dict[str, list[str]] = {
    "playback_issue": [
        "songs keep buffering on my phone",
        "spotify won't play anything right now",
        "music stops after a few seconds",
        "getting a lot of skipping on the app",
        "songs cut out halfway through",
        "playback keeps pausing randomly",
        "audio quality is really bad today",
        "can't play any music offline",
    ],
    "account_access": [
        "can't log into my spotify account",
        "forgot my password and the reset email won't arrive",
        "account is locked and i can't get in",
        "someone hacked my spotify account",
        "two factor auth is broken for me",
        "getting error when i try to sign in",
        "my account got disabled somehow",
        "login page just spins and nothing happens",
    ],
    "billing_payment": [
        "i was charged twice this month",
        "want a refund for my premium subscription",
        "how do i cancel spotify premium",
        "my payment method isn't working",
        "why did you charge me when i have student plan",
        "charged after i cancelled the subscription",
        "can't update my credit card details",
        "family plan billing is wrong",
    ],
    "content_unavailable": [
        "why is this album not on spotify",
        "can't find the new album by this artist",
        "podcast is missing some episodes",
        "song is greyed out and won't play",
        "playlist songs keep disappearing",
        "why isn't this available in my country",
        "album was removed from spotify",
        "artist content is gone from my library",
    ],
    "app_bug": [
        "spotify crashes every time i open it",
        "the app won't open at all on android",
        "search isn't working in the app",
        "liked songs list is empty but they were there",
        "connect device feature is broken",
        "shuffle button is stuck on",
        "home screen keeps freezing",
        "can't see my playlists",
    ],
    "feature_request": [
        "please add lyrics support for all songs",
        "i wish you could sort playlists by tempo",
        "can you add a crossfade option",
        "would love a dark mode that's actually dark",
        "please add sleep timer to the app",
        "i want to be able to see my listening history",
        "can you add local file support on ios",
        "please bring back the old desktop ui",
    ],
    "general_inquiry": [
        "how do i use spotify on my new tv",
        "what's the difference between free and premium",
        "thanks for the help spotify",
        "is spotify down right now",
        "how many devices can i use at once",
        "do you have a student discount",
        "great service thank you",
        "when is this feature coming",
    ],
}


class IntentPrediction(NamedTuple):
    intent: str
    confidence: float
    scores: dict[str, float]   # all intent probabilities
    refinement_used: bool       # whether LLM stage-2 was invoked


class IntentClassifier:
    """
    Prototype + logistic regression classifier with optional LLM refinement.
    """

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        model_name = self.cfg["intents"]["embedding_model"]
        self.threshold = self.cfg["intents"]["confidence_threshold"]
        self.llm_model = self.cfg["reply_generator"]["llm_model"]

        log.info("Loading embedding model: %s", model_name)
        self.encoder = SentenceTransformer(model_name)

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(INTENT_LABELS)
        self.clf: LogisticRegression | None = None
        self._prototypes: np.ndarray | None = None   # shape (n_intents, dim)

        model_dir = Path(self.cfg["paths"]["models_dir"])
        self._clf_path = model_dir / "intent_clf.joblib"
        self._proto_path = model_dir / "intent_prototypes.npy"

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #

    @timeit
    def fit_from_seeds(self) -> None:
        """
        Train using seed examples only. Good baseline before any hand labels.
        Augment each seed with minor paraphrases via simple token shuffles.
        """
        log.info("Fitting from seed examples …")
        texts, labels = [], []
        for intent, examples in SEED_EXAMPLES.items():
            for ex in examples:
                texts.append(ex)
                labels.append(intent)

        self._fit_on(texts, labels)
        log.info("Seed-fit complete — %d examples", len(texts))

    @timeit
    def fit(self, texts: list[str], labels: list[str]) -> None:
        """Train on provided labelled data (golden eval or any hand labels)."""
        self._fit_on(texts, labels)

    def _fit_on(self, texts: list[str], labels: list[str]) -> None:
        embeddings = self._embed(texts)
        y = self.label_encoder.transform(labels)

        # Build prototypes (per-class mean embedding, L2-normalised)
        self._prototypes = np.zeros((len(INTENT_LABELS), embeddings.shape[1]))
        for idx, intent in enumerate(INTENT_LABELS):
            mask = np.array(labels) == intent
            if mask.any():
                self._prototypes[idx] = embeddings[mask].mean(axis=0)
        norms = np.linalg.norm(self._prototypes, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._prototypes /= norms

        # Logistic regression on top of embeddings
        self.clf = LogisticRegression(
            max_iter=1000,
            C=4.0,
            class_weight="balanced",
            multi_class="multinomial",
            solver="lbfgs",
        )
        self.clf.fit(embeddings, y)
        log.info("Classifier fitted (classes: %s)", list(self.label_encoder.classes_))

    def save(self) -> None:
        ensure_dir(self._clf_path.parent)
        joblib.dump(self.clf, self._clf_path)
        np.save(self._proto_path, self._prototypes)
        log.info("Classifier saved to %s", self._clf_path.parent)

    def load(self) -> None:
        self.clf = joblib.load(self._clf_path)
        self._prototypes = np.load(self._proto_path)
        log.info("Classifier loaded from %s", self._clf_path.parent)

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def predict(self, text: str) -> IntentPrediction:
        """Classify a single customer message."""
        if self.clf is None:
            # Auto-fit from seeds if not trained yet
            self.fit_from_seeds()

        emb = self._embed([text])  # (1, dim)
        proba = self.clf.predict_proba(emb)[0]  # (n_classes,)
        classes = list(self.label_encoder.classes_)
        scores = dict(zip(classes, proba.tolist()))

        top_intent = classes[np.argmax(proba)]
        top_conf = float(np.max(proba))

        refinement_used = False
        if top_conf < self.threshold:
            # Stage 2: ask LLM to break the tie
            top_intent, top_conf, refinement_used = self._llm_refinement(
                text, scores
            )

        return IntentPrediction(
            intent=top_intent,
            confidence=top_conf,
            scores=scores,
            refinement_used=refinement_used,
        )

    def predict_batch(self, texts: list[str]) -> list[IntentPrediction]:
        if self.clf is None:
            self.fit_from_seeds()

        embeddings = self._embed(texts)
        probas = self.clf.predict_proba(embeddings)
        classes = list(self.label_encoder.classes_)
        results = []
        for i, (text, proba) in enumerate(zip(texts, probas)):
            scores = dict(zip(classes, proba.tolist()))
            top_intent = classes[np.argmax(proba)]
            top_conf = float(np.max(proba))
            refinement_used = False
            if top_conf < self.threshold:
                top_intent, top_conf, refinement_used = self._llm_refinement(
                    text, scores
                )
            results.append(IntentPrediction(
                intent=top_intent,
                confidence=top_conf,
                scores=scores,
                refinement_used=refinement_used,
            ))
        return results

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self.encoder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )

    def _llm_refinement(
        self, text: str, scores: dict[str, float]
    ) -> tuple[str, float, bool]:
        """
        Ask the LLM to pick the best intent given the message and top-3 candidates.
        Returns (intent, confidence, used=True).
        """
        try:
            from src.utils import llm_chat
            top3 = sorted(scores, key=scores.get, reverse=True)[:3]
            prompt = f"""You are classifying a Spotify customer-support message.

Message: "{text}"

Top candidate intents (from embedding classifier):
{chr(10).join(f"  - {i}: {scores[i]:.2%}" for i in top3)}

All available intents:
{chr(10).join(f"  - {i}" for i in INTENT_LABELS)}

Respond with a JSON object:
{{"intent": "<one of the intent labels>", "confidence": <0.0-1.0>}}

Only output valid JSON, nothing else."""

            raw = llm_chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=60,
            )
            obj = json.loads(raw)
            intent = obj.get("intent", top3[0])
            if intent not in INTENT_LABELS:
                intent = top3[0]
            conf = float(obj.get("confidence", 0.55))
            return intent, conf, True
        except Exception as e:
            log.warning("LLM refinement failed (%s) — using top classifier pick", e)
            top = sorted(scores, key=scores.get, reverse=True)[0]
            return top, float(scores[top]), False


def build_and_save_classifier(
    golden_path: Path | None = None, cfg: dict | None = None
) -> IntentClassifier:
    """
    Convenience: build classifier from golden set (if available) else seeds.
    """
    clf = IntentClassifier(cfg)

    if golden_path and Path(golden_path).exists():
        log.info("Training from golden eval set: %s", golden_path)
        with open(golden_path) as f:
            examples = json.load(f)
        texts = [e["text"] for e in examples]
        labels = [e["intent"] for e in examples]
        clf.fit(texts, labels)
    else:
        log.info("No golden set found — training from seed examples")
        clf.fit_from_seeds()

    clf.save()
    return clf


if __name__ == "__main__":
    clf = IntentClassifier()
    clf.fit_from_seeds()

    test_cases = [
        "my spotify keeps buffering on my iphone",
        "i was charged twice please refund me",
        "i can't log in to my account at all",
        "why isn't the new taylor swift album on spotify",
        "the app crashes every time i try to open it",
        "please add a crossfade feature to the mobile app",
        "hi how do i use spotify on my samsung tv",
    ]
    print("\nSample predictions:")
    for text in test_cases:
        pred = clf.predict(text)
        print(f"  [{pred.intent:25s} {pred.confidence:.2%}] {text[:70]}")
