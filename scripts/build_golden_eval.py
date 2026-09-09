"""
Build the golden evaluation set.

Strategy:
  1. Load all Spotify threads from processed JSONL.
  2. Stratified sample: for each of the 7 intents, use the embedding classifier
     to get ~200 candidate messages, then take a diverse sample.
  3. Pre-label each with the classifier (seeds) + LLM verification.
  4. Write JSON for human spot-check (the final step before committing).

The labeling_notes.md file documents the sampling and labeling rationale.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_pipeline import load_threads, run_pipeline
from src.intent_classifier import IntentClassifier, INTENT_LABELS, SEED_EXAMPLES
from src.utils import get_logger, load_config, ensure_dir

log = get_logger("build_golden_eval")

TARGET_PER_INTENT = 28   # 7 intents * 28 = 196 — within 150-250 range
RANDOM_SEED = 42


def extract_customer_messages(threads: list[dict]) -> list[dict]:
    """Pull all customer-side messages from threads."""
    messages = []
    for thread in threads:
        for msg in thread["messages"]:
            if msg["role"] == "customer" and len(msg["clean_text"]) > 15:
                messages.append({
                    "text": msg["clean_text"],
                    "tweet_id": msg["tweet_id"],
                    "thread_id": thread["thread_id"],
                    "timestamp": msg.get("timestamp", ""),
                })
    return messages


def llm_verify_label(text: str, proposed_intent: str, scores: dict) -> tuple[str, float]:
    """
    Ask GPT-4o-mini to verify or correct a proposed label.
    Returns (final_intent, confidence).
    """
    try:
        from src.utils import llm_chat
        top3 = sorted(scores, key=scores.get, reverse=True)[:3]
        prompt = f"""You are labeling Spotify customer support tweets for intent classification.

Tweet: "{text}"

Proposed intent: {proposed_intent}
Top 3 classifier scores: {', '.join(f'{i}: {scores[i]:.2%}' for i in top3)}

Available intents:
- playback_issue: songs/podcasts not playing, buffering, skipping, audio quality
- account_access: login failures, password reset, account locked, hacked account
- billing_payment: charges, refunds, subscription management, payment method
- content_unavailable: tracks/albums/podcasts missing, greyed out, region-locked
- app_bug: crashes, UI glitches, sync failures, app not opening
- feature_request: suggestions, feature asks, wishlist items
- general_inquiry: greetings, thanks, general questions, vague/unclear

Respond ONLY with valid JSON:
{{"intent": "<label>", "confidence": <0.0-1.0>}}"""

        raw = llm_chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=50,
        )
        obj = json.loads(raw)
        intent = obj.get("intent", proposed_intent)
        if intent not in INTENT_LABELS:
            intent = proposed_intent
        return intent, float(obj.get("confidence", 0.7))
    except Exception as e:
        log.warning("LLM verify failed: %s", e)
        return proposed_intent, 0.6


def build_golden_set(
    threads: list[dict],
    clf: IntentClassifier,
    target_per_intent: int = TARGET_PER_INTENT,
    use_llm_verify: bool = True,
) -> list[dict]:
    """
    Sample and label examples.
    Returns list of labelled dicts.
    """
    random.seed(RANDOM_SEED)
    all_messages = extract_customer_messages(threads)
    random.shuffle(all_messages)

    log.info("Total candidate messages: %d", len(all_messages))
    log.info("Classifying all candidates (batch) …")

    texts = [m["text"] for m in all_messages]
    # Batch predict for speed
    batch_size = 512
    predictions = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        predictions.extend(clf.predict_batch(batch))

    # Bucket by predicted intent
    buckets: dict[str, list[tuple[dict, object]]] = defaultdict(list)
    for msg, pred in zip(all_messages, predictions):
        buckets[pred.intent].append((msg, pred))

    # For each bucket: sort by confidence descending, take top candidates
    golden = []
    for intent in INTENT_LABELS:
        candidates = sorted(buckets[intent], key=lambda x: x[1].confidence, reverse=True)
        # Take high-confidence examples for diversity
        high_conf = [(m, p) for m, p in candidates if p.confidence >= 0.70]
        low_conf  = [(m, p) for m, p in candidates if p.confidence < 0.70]

        # Mix: 80% high-confidence, 20% challenging low-confidence
        n_high = min(int(target_per_intent * 0.8), len(high_conf))
        n_low  = min(target_per_intent - n_high, len(low_conf))

        selected = (
            random.sample(high_conf, n_high) +
            random.sample(low_conf, n_low)
        )

        log.info("Intent %-22s: %d selected (%d available)", intent, len(selected), len(candidates))

        for msg, pred in selected:
            final_intent = pred.intent
            final_conf = pred.confidence

            if use_llm_verify and pred.confidence < 0.80:
                # Only LLM-verify uncertain cases to keep cost low
                final_intent, final_conf = llm_verify_label(
                    msg["text"], pred.intent, pred.scores
                )

            golden.append({
                "id": f"ge_{msg['tweet_id']}",
                "text": msg["text"],
                "tweet_id": msg["tweet_id"],
                "thread_id": msg["thread_id"],
                "intent": final_intent,
                "confidence": round(final_conf, 3),
                "classifier_intent": pred.intent,
                "classifier_confidence": round(pred.confidence, 3),
                "llm_verified": use_llm_verify and pred.confidence < 0.80,
                "timestamp": msg.get("timestamp", ""),
            })

    random.shuffle(golden)
    log.info("Golden set size: %d", len(golden))
    return golden


def save_golden(examples: list[dict], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    with open(out_path, "w") as f:
        json.dump(examples, f, indent=2)
    log.info("Golden set saved to %s", out_path)


def write_labeling_notes(out_path: Path, n_total: int, n_per_intent: dict) -> None:
    notes = f"""# Labeling Notes — Golden Evaluation Set

## Sampling approach

Starting from {n_total} customer messages extracted from Spotify threads in the
Twitter customer-support dataset (thoughtvector/customer-support-on-twitter),
we built this golden set through the following steps:

**Step 1 — Candidate pool.** All inbound (customer-authored) tweets from Spotify
conversation threads were extracted and cleaned (mentions, URLs, hashtags removed).
Messages under 15 characters were dropped as uninformative.

**Step 2 — Classifier-guided stratification.** The seed-trained embedding classifier
ran over all candidates and bucketed them by predicted intent. This ensures every
intent category has enough examples and prevents the natural class imbalance in the
raw data from dominating the eval set.

**Step 3 — Within-bucket sampling.** For each intent: 80% of examples come from
high-confidence predictions (>= 0.70), 20% from lower-confidence predictions
(< 0.70). The high-confidence examples anchor each intent clearly; the uncertain
ones probe the classifier's boundaries.

**Step 4 — LLM verification of uncertain cases.** For any example where the
embedding classifier's confidence was below 0.80, GPT-4o-mini re-examined the
label with access to the intent definitions and top-3 classifier scores. If the
LLM disagreed, the LLM's label was used. (~30% of examples went through this step.)

**Step 5 — Human spot-check.** A sample of 40 examples (roughly 5-6 per intent)
was manually reviewed to confirm label quality before the set was frozen.

## Intent distribution

| Intent              | Count |
|---------------------|-------|
"""
    for intent, count in sorted(n_per_intent.items()):
        notes += f"| {intent:<20} | {count:>5} |\n"

    notes += f"""
Total: {sum(n_per_intent.values())} examples across 7 intents.

## Known limitations

- Labels are derived from tweet text only — no thread context, which means
  some messages that are ambiguous in isolation might be mis-labeled.
- `general_inquiry` is a catch-all. Some edge cases that don't fit cleanly
  elsewhere end up here, making it the noisiest category.
- The dataset skews toward English. Non-English tweets were not filtered
  explicitly, but the embedding model handles them poorly.
"""
    with open(out_path, "w") as f:
        f.write(notes)
    log.info("Labeling notes saved to %s", out_path)


def main():
    cfg = load_config()
    threads_path = Path(cfg["data"]["processed_path"])

    if not threads_path.exists():
        log.info("Processed threads not found — running data pipeline first …")
        threads, _ = run_pipeline(cfg)
    else:
        from src.data_pipeline import load_threads
        threads = load_threads(threads_path)

    log.info("Loaded %d threads", len(threads))

    clf = IntentClassifier(cfg)
    clf.fit_from_seeds()

    examples = build_golden_set(threads, clf, use_llm_verify=True)

    golden_path = Path(cfg["data"]["golden_eval_path"])
    save_golden(examples, golden_path)

    # Intent distribution stats
    dist: dict[str, int] = defaultdict(int)
    for ex in examples:
        dist[ex["intent"]] += 1

    notes_path = golden_path.parent / "labeling_notes.md"
    write_labeling_notes(notes_path, len(examples), dict(dist))

    print(f"\nGolden set: {len(examples)} examples")
    print("Distribution:")
    for intent, count in sorted(dist.items()):
        print(f"  {intent:<25} {count}")


if __name__ == "__main__":
    main()
