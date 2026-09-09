"""
Automated evaluation metrics.

Intent classification: accuracy, precision, recall, F1 (macro + weighted).
Reply quality: BLEU-1, BLEU-4, ROUGE-L.
Escalation: precision, recall, F1 at decision level.
Calibration: expected calibration error (ECE) for classifier confidence.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Sequence

import numpy as np


# ------------------------------------------------------------------ #
# Intent metrics                                                       #
# ------------------------------------------------------------------ #

def intent_accuracy(y_true: list[str], y_pred: list[str]) -> float:
    if not y_true:
        return 0.0
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def intent_f1(
    y_true: list[str], y_pred: list[str], average: str = "macro"
) -> dict[str, float]:
    """Return precision, recall, F1 for each class + macro/weighted aggregate."""
    classes = sorted(set(y_true) | set(y_pred))
    per_class: dict[str, dict] = {}
    for cls in classes:
        tp = sum(t == p == cls for t, p in zip(y_true, y_pred))
        fp = sum(p == cls and t != cls for t, p in zip(y_true, y_pred))
        fn = sum(t == cls and p != cls for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[cls] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}

    if average == "macro":
        agg_prec = np.mean([v["precision"] for v in per_class.values()])
        agg_rec  = np.mean([v["recall"]    for v in per_class.values()])
        agg_f1   = np.mean([v["f1"]        for v in per_class.values()])
    else:   # weighted
        total = len(y_true)
        agg_prec = sum(v["precision"] * v["support"] for v in per_class.values()) / total
        agg_rec  = sum(v["recall"]    * v["support"] for v in per_class.values()) / total
        agg_f1   = sum(v["f1"]        * v["support"] for v in per_class.values()) / total

    return {
        "precision": float(agg_prec),
        "recall":    float(agg_rec),
        "f1":        float(agg_f1),
        "per_class": per_class,
    }


def calibration_error(
    confidences: list[float], correctness: list[bool], n_bins: int = 10
) -> float:
    """
    Expected Calibration Error (ECE).
    Measures how well confidence scores reflect actual accuracy.
    Lower is better; 0 = perfect calibration.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        low, high = bins[i], bins[i+1]
        mask = [(low <= c < high) for c in confidences]
        if not any(mask):
            continue
        in_bin = [(c, cor) for c, cor, m in zip(confidences, correctness, mask) if m]
        avg_conf = np.mean([c for c, _ in in_bin])
        avg_acc  = np.mean([cor for _, cor in in_bin])
        ece += (len(in_bin) / n) * abs(avg_conf - avg_acc)
    return float(ece)


# ------------------------------------------------------------------ #
# Reply quality metrics                                                #
# ------------------------------------------------------------------ #

def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def bleu_n(hypothesis: list[str], reference: list[str], n: int) -> float:
    """Corpus-level BLEU-n (single reference, uniform weights)."""
    if len(hypothesis) < n:
        return 0.0
    hyp_ngrams = Counter(
        tuple(hypothesis[i:i+n]) for i in range(len(hypothesis) - n + 1)
    )
    ref_ngrams = Counter(
        tuple(reference[i:i+n]) for i in range(len(reference) - n + 1)
    )
    clipped = {ng: min(cnt, ref_ngrams[ng]) for ng, cnt in hyp_ngrams.items()}
    if sum(hyp_ngrams.values()) == 0:
        return 0.0
    precision = sum(clipped.values()) / sum(hyp_ngrams.values())
    # Brevity penalty
    bp = math.exp(1 - len(reference) / len(hypothesis)) if len(hypothesis) < len(reference) else 1.0
    return bp * precision


def bleu_score(hypothesis: str, reference: str, max_n: int = 4) -> dict[str, float]:
    hyp_tokens = _tokenize(hypothesis)
    ref_tokens = _tokenize(reference)
    scores = {}
    for n in range(1, max_n + 1):
        scores[f"bleu_{n}"] = bleu_n(hyp_tokens, ref_tokens, n)
    return scores


def rouge_l(hypothesis: str, reference: str) -> float:
    """ROUGE-L F-score via LCS."""
    hyp = _tokenize(hypothesis)
    ref = _tokenize(reference)
    if not hyp or not ref:
        return 0.0
    m, n = len(hyp), len(ref)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if hyp[i-1] == ref[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    precision = lcs / m
    recall    = lcs / n
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


def reply_metrics(
    hypotheses: list[str], references: list[str]
) -> dict[str, float]:
    """Aggregate reply quality metrics over a list of (hyp, ref) pairs."""
    bleu1_scores, bleu4_scores, rougel_scores = [], [], []
    for hyp, ref in zip(hypotheses, references):
        b = bleu_score(hyp, ref)
        bleu1_scores.append(b["bleu_1"])
        bleu4_scores.append(b["bleu_4"])
        rougel_scores.append(rouge_l(hyp, ref))
    return {
        "bleu_1":  float(np.mean(bleu1_scores)),
        "bleu_4":  float(np.mean(bleu4_scores)),
        "rouge_l": float(np.mean(rougel_scores)),
    }


# ------------------------------------------------------------------ #
# Escalation metrics                                                   #
# ------------------------------------------------------------------ #

def escalation_metrics(
    y_true_esc: list[bool], y_pred_esc: list[bool]
) -> dict[str, float]:
    """Precision, recall, F1 for escalation decision (positive = escalate)."""
    tp = sum(t and p for t, p in zip(y_true_esc, y_pred_esc))
    fp = sum((not t) and p for t, p in zip(y_true_esc, y_pred_esc))
    fn = sum(t and (not p) for t, p in zip(y_true_esc, y_pred_esc))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "escalation_rate": sum(y_pred_esc) / len(y_pred_esc) if y_pred_esc else 0.0,
    }
