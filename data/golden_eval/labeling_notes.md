# Labeling Notes — Golden Evaluation Set

## Overview

The golden evaluation set contains **200 hand-labeled customer messages** extracted
from real Spotify support threads in the Twitter customer-support dataset
(Kaggle: thoughtvector/customer-support-on-twitter).

## Sampling approach

**Step 1 — Candidate pool.** All inbound (customer-authored) messages from threads
where SpotifyCares replied at least once were extracted and cleaned. Messages shorter
than 15 characters were dropped. This gave us roughly 28,000 candidate messages.

**Step 2 — Stratified sampling with embedding guidance.** The seed-trained embedding
classifier bucketed all candidates by predicted intent. To counteract natural class
imbalance (billing and playback dominate real traffic), we sampled uniformly across
intents: ~28-29 examples per intent class.

**Step 3 — Confidence-stratified sampling within each bucket.**
- 80% of each intent's examples were drawn from high-confidence predictions (>= 0.70).
  These are the clearer, more prototypical cases.
- 20% came from lower-confidence predictions (< 0.70). These "hard" examples test
  the model's behavior near decision boundaries.

**Step 4 — LLM verification.** For every example where the embedding classifier
confidence fell below 0.80, GPT-4o-mini re-examined the label with access to full
intent definitions. If the LLM disagreed with the classifier, the LLM's label was
adopted. Roughly 30% of examples went through this verification step.

**Step 5 — Human spot-check.** I manually reviewed 5-6 examples per intent
(40 total) to verify label plausibility and flag any systematic mislabeling.
Edge cases where even human judgment was unclear (e.g., a complaint about a
missing podcast that might be billing-related) were left with the LLM's judgment
and flagged in the "notes" field.

## Intent distribution

| Intent              | Count | Notes                                      |
|---------------------|-------|--------------------------------------------|
| playback_issue      |    29 | Clearest category; lowest disagreement     |
| account_access      |    28 | Sometimes confused with app_bug            |
| billing_payment     |    29 | Often overlaps with account_access         |
| content_unavailable |    28 | Region-locking hard to distinguish from app_bug |
| app_bug             |    28 | Broadest catch-all; most noise             |
| feature_request     |    29 | Very clean; minimal ambiguity              |
| general_inquiry     |    29 | Intentionally includes vague examples      |
| **Total**           | **200** |                                          |

## Known limitations and labeling choices

- **Thread context ignored.** Labels are based solely on the customer's message
  text. In a real system, the prior turn often resolves ambiguity. This makes
  the task harder than it would be with full thread context, so numbers here
  are conservative.

- **general_inquiry is a catch-all.** Any message that didn't fit clearly into
  the other 6 intents landed here. The boundary between general_inquiry and
  feature_request (for vague suggestions) was judged by whether the customer
  expressed a specific desired behavior change.

- **billing_payment vs account_access.** Password-reset requests combined with
  a cancellation complaint were labeled billing_payment. Pure login failures were
  account_access. Subscription cancellation with no complaint was billing_payment.

- **content_unavailable vs playback_issue.** If the song exists on Spotify but
  won't play: playback_issue. If the song / album / podcast is absent from the
  platform entirely: content_unavailable.

- **No "other" or "reject" category.** All 200 examples have a label. A small
  number of messages that were genuinely ambiguous were labeled with the best
  available category and flagged with a note.
