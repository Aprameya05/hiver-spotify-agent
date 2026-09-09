"""
Evaluation harness — runs the full agent over the golden eval set and
produces a comprehensive results report.

Baselines:
  Trivial  — always predicts the majority class intent, returns a fixed canned reply.
  Simple   — TF-IDF cosine similarity for intent + nearest-neighbour reply (no LLM).
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from eval.metrics import (
    intent_accuracy,
    intent_f1,
    calibration_error,
    reply_metrics,
    escalation_metrics,
)
from eval.llm_judge import LLMJudge
from src.utils import get_logger, load_config, ensure_dir

log = get_logger(__name__)

CANNED_REPLY = (
    "Hi! Thanks for reaching out to Spotify support. "
    "We're sorry to hear you're having trouble. Please DM us your account details and we'll help sort this out."
)


# ------------------------------------------------------------------ #
# Baselines                                                            #
# ------------------------------------------------------------------ #

class TrivialBaseline:
    """Always predicts majority class; returns fixed canned reply."""
    name = "trivial"

    def fit(self, texts: list[str], intents: list[str]) -> None:
        counter = Counter(intents)
        self.majority_class = counter.most_common(1)[0][0]

    def predict(self, text: str) -> tuple[str, float, str]:
        """Returns (intent, confidence, reply)."""
        return self.majority_class, 1.0 / 7, CANNED_REPLY


class TfidfBaseline:
    """TF-IDF + cosine similarity for intent; nearest-neighbour for reply."""
    name = "tfidf_nn"

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        self.train_vecs = None
        self.train_intents: list[str] = []
        self.train_replies: list[str] = []

    def fit(
        self,
        texts: list[str],
        intents: list[str],
        replies: list[str] | None = None,
    ) -> None:
        self.train_vecs = self.vectorizer.fit_transform(texts)
        self.train_intents = intents
        self.train_replies = replies or [CANNED_REPLY] * len(texts)

    def predict(self, text: str) -> tuple[str, float, str]:
        if self.train_vecs is None:
            raise RuntimeError("Call fit() first")
        vec = self.vectorizer.transform([text])
        sims = cosine_similarity(vec, self.train_vecs)[0]
        best_idx = int(np.argmax(sims))
        best_intent = self.train_intents[best_idx]
        best_conf = float(sims[best_idx])
        best_reply = self.train_replies[best_idx]
        return best_intent, best_conf, best_reply


# ------------------------------------------------------------------ #
# Harness                                                              #
# ------------------------------------------------------------------ #

def load_golden(path: str | Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def run_evaluation(
    agent,
    golden_path: str | Path,
    cfg: dict | None = None,
    skip_judge: bool = False,
) -> dict[str, Any]:
    cfg = cfg or load_config()
    golden = load_golden(golden_path)
    log.info("Evaluating on %d golden examples …", len(golden))

    texts = [ex["text"] for ex in golden]
    true_intents = [ex["intent"] for ex in golden]
    # We don't have reference replies in the golden set — use historical
    # replies from the RAG index as a weak reference (proxy for quality eval).
    # For LLM judge we use the agent's retrieved examples' replies as reference.

    # ------------------------------------------------------------------ #
    # A. Main agent                                                        #
    # ------------------------------------------------------------------ #
    t0 = time.perf_counter()
    log.info("Running main agent …")
    agent_responses = []
    for text in texts:
        resp = agent.handle(text)
        agent_responses.append(resp)
    agent_latency = (time.perf_counter() - t0) * 1000 / len(texts)

    pred_intents = [r.intent.intent for r in agent_responses]
    confidences  = [r.intent.confidence for r in agent_responses]
    correctness  = [p == t for p, t in zip(pred_intents, true_intents)]
    pred_esc     = [r.escalation.should_escalate for r in agent_responses]
    pred_replies = [r.reply.reply for r in agent_responses]

    # True escalation: high-sensitivity intents should escalate
    high_sens = set(cfg["escalation"]["high_sensitivity_intents"])
    true_esc = [intent in high_sens for intent in true_intents]

    agent_intent  = intent_f1(true_intents, pred_intents, "macro")
    agent_acc     = intent_accuracy(true_intents, pred_intents)
    agent_ece     = calibration_error(confidences, correctness)
    agent_esc     = escalation_metrics(true_esc, pred_esc)

    # ------------------------------------------------------------------ #
    # B. Trivial baseline                                                  #
    # ------------------------------------------------------------------ #
    log.info("Running trivial baseline …")
    trivial = TrivialBaseline()
    trivial.fit(texts, true_intents)
    trivial_preds = [trivial.predict(t) for t in texts]
    trivial_intents = [p[0] for p in trivial_preds]
    trivial_replies = [p[2] for p in trivial_preds]
    trivial_esc     = [False] * len(texts)   # trivial never escalates

    trivial_acc = intent_accuracy(true_intents, trivial_intents)
    trivial_f1  = intent_f1(true_intents, trivial_intents, "macro")
    trivial_esc_m = escalation_metrics(true_esc, trivial_esc)

    # ------------------------------------------------------------------ #
    # C. TF-IDF baseline                                                   #
    # ------------------------------------------------------------------ #
    log.info("Running TF-IDF baseline …")
    tfidf = TfidfBaseline()
    # Use first 70% as train, last 30% as eval for baseline (simulate train/test split)
    n_train = int(len(texts) * 0.7)
    train_texts, train_intents = texts[:n_train], true_intents[:n_train]
    # Pull nearest-neighbour replies from RAG metadata if available
    rag_meta = []
    rag_meta_path = Path(cfg["paths"]["metadata_path"])
    if rag_meta_path.exists():
        with open(rag_meta_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rag_meta.append(json.loads(line))
    train_replies = [CANNED_REPLY] * n_train
    if rag_meta:
        # Sample replies from RAG metadata matching each intent
        intent_to_replies: dict[str, list[str]] = {}
        for pair in rag_meta:
            # We don't have intent labels in RAG metadata — use canned
            pass
        train_replies = [CANNED_REPLY] * n_train

    tfidf.fit(train_texts, train_intents, train_replies)
    tfidf_results = [tfidf.predict(t) for t in texts[n_train:]]
    tfidf_intents = [r[0] for r in tfidf_results]
    tfidf_replies = [r[2] for r in tfidf_results]

    eval_true_intents = true_intents[n_train:]
    tfidf_acc = intent_accuracy(eval_true_intents, tfidf_intents)
    tfidf_f1  = intent_f1(eval_true_intents, tfidf_intents, "macro")

    # ------------------------------------------------------------------ #
    # D. LLM-as-judge for reply quality                                    #
    # ------------------------------------------------------------------ #
    judge_scores: dict[str, Any] = {}
    if not skip_judge:
        log.info("Running LLM judge …")
        judge = LLMJudge(cfg)
        # Sample 40 examples for judge (cost control)
        sample_idx = list(range(0, len(texts), max(1, len(texts) // 40)))[:40]
        judge_pairs = [(texts[i], pred_replies[i]) for i in sample_idx]
        judge_results = judge.score_batch(judge_pairs)

        mean_overall = float(np.mean([s.overall for s in judge_results]))
        mean_consistency = float(np.mean([s.consistency for s in judge_results]))

        # Compare vs trivial on same sample
        trivial_judge_pairs = [(texts[i], trivial_replies[i]) for i in sample_idx]
        trivial_judge_results = judge.score_batch(trivial_judge_pairs)
        trivial_mean = float(np.mean([s.overall for s in trivial_judge_results]))

        judge_scores = {
            "agent_mean_overall": round(mean_overall, 3),
            "trivial_mean_overall": round(trivial_mean, 3),
            "mean_judge_consistency": round(mean_consistency, 3),
            "n_judged": len(judge_results),
            "dim_means": {
                "relevance":     round(float(np.mean([s.relevance for s in judge_results])), 3),
                "accuracy":      round(float(np.mean([s.accuracy for s in judge_results])), 3),
                "tone":          round(float(np.mean([s.tone for s in judge_results])), 3),
                "conciseness":   round(float(np.mean([s.conciseness for s in judge_results])), 3),
                "actionability": round(float(np.mean([s.actionability for s in judge_results])), 3),
            }
        }

    # ------------------------------------------------------------------ #
    # E. Compile summary                                                   #
    # ------------------------------------------------------------------ #
    summary = {
        "n_examples": len(golden),
        "agent": {
            "intent_accuracy":       round(agent_acc, 4),
            "intent_f1_macro":       round(agent_intent["f1"], 4),
            "intent_precision_macro":round(agent_intent["precision"], 4),
            "intent_recall_macro":   round(agent_intent["recall"], 4),
            "ece":                   round(agent_ece, 4),
            "escalation_precision":  round(agent_esc["precision"], 4),
            "escalation_recall":     round(agent_esc["recall"], 4),
            "escalation_f1":         round(agent_esc["f1"], 4),
            "escalation_rate":       round(agent_esc["escalation_rate"], 4),
            "mean_latency_ms":       round(agent_latency, 1),
            "per_class_f1":          agent_intent["per_class"],
        },
        "baseline_trivial": {
            "intent_accuracy": round(trivial_acc, 4),
            "intent_f1_macro": round(trivial_f1["f1"], 4),
            "escalation_recall": 0.0,
        },
        "baseline_tfidf": {
            "intent_accuracy": round(tfidf_acc, 4),
            "intent_f1_macro": round(tfidf_f1["f1"], 4),
        },
        "judge": judge_scores,
    }

    return summary


if __name__ == "__main__":
    from src.agent import SpotifyAgent
    cfg = load_config()
    agent = SpotifyAgent(cfg)
    agent.warm_up()

    golden_path = cfg["data"]["golden_eval_path"]
    summary = run_evaluation(agent, golden_path, cfg)
    print(json.dumps(summary, indent=2))
