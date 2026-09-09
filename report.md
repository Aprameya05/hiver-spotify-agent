# Report: Spotify AI Customer Support Agent

**Author:** Aprameya Bharadwaj  
**Brand:** Spotify (@SpotifyCares)  
**Dataset:** thoughtvector/customer-support-on-twitter (~3M tweets, Kaggle)

---

## Problem framing

### What does "good" mean for Spotify specifically?

Spotify's support is primarily a triage + troubleshooting operation. Most customer complaints cluster into a handful of repeatable failure modes: playback issues, app crashes, billing confusion, account lockouts, and missing content. For these, "good" means:

1. **Correct intent identification** — so the right troubleshooting path gets triggered.
2. **A reply that matches Spotify's established voice** — brief, empathetic, one actionable step, a DM invitation if account data is needed.
3. **Conservative escalation** — billing disputes and potential security issues should always reach a human.

"Good" explicitly does not mean maximizing BLEU or ROUGE. A reply that is lexically similar to a historical Spotify tweet but factually wrong (suggesting the wrong troubleshooting step) is worse than a reply with low lexical overlap that actually solves the problem. This is why the LLM-as-judge is the primary quality signal.

### What I chose not to build

- **Entity extraction** (account IDs, device types, OS versions): Useful in production but adds complexity without changing what's evaluable from tweet text alone.
- **Multi-turn state management**: The agent treats each message independently. Full conversation threading would require a session state store and complicates the evaluation harness significantly.
- **Fine-tuned generation model**: GPT-4o-mini with few-shot examples is good enough that fine-tuning isn't justified at this dataset scale. The real gain would come from more and better labelled data, not a fancier model.
- **A/B testing framework**: The only metric that matters in production is customer resolution rate. I couldn't measure that without live traffic, so I used the LLM judge as a proxy.

---

## Results vs baselines

Three systems evaluated on the 200-example golden set:

### Trivial baseline
Always predicts `general_inquiry` (the majority class in balanced evaluation is actually `playback_issue`, so I use whichever is most common in that run). Returns a fixed canned reply for everything. No escalation.

### TF-IDF + nearest-neighbour baseline
TF-IDF vectorization over unigrams and bigrams. Cosine similarity for intent (picks the intent of the nearest training example). Nearest-neighbour reply (copies the most similar historical Spotify reply verbatim). No LLM involved.

### Main agent
Two-stage embedding classifier + GPT-4o-mini refinement for uncertain cases. FAISS RAG with temporal weighting + GPT-4o-mini generation. Multi-signal escalation engine.

| Metric | Main agent | TF-IDF baseline | Trivial |
|--------|-----------|-----------------|---------|
| Intent accuracy | **0.81** | 0.61 | 0.14 |
| Intent F1 macro | **0.80** | 0.59 | 0.02 |
| Calibration (ECE) | **0.06** | N/A | N/A |
| Escalation F1 | **0.72** | N/A | 0.00 |
| Judge score (1-5) | **4.1** | 2.6 | 2.3 |
| Judge consistency | 0.88 | N/A | N/A |
| Avg latency | 820ms | 45ms | 2ms |

The TF-IDF baseline is a meaningful step above trivial — it gets the easy cases right (billing complaints contain "charged" and "refund"; those words don't appear in playback threads). The main agent's 20-point F1 gain over TF-IDF comes primarily from handling boundary cases: sarcastic messages, messages that mention multiple issues, and short messages that don't have strong lexical signals.

---

## Failure analysis

### Failure mode 1: billing_payment vs account_access conflation

About 40% of `billing_payment` errors are predicted as `account_access`. The surface text often contains "I can't access" even when the core complaint is a charge. The two intents share vocabulary ("account", "access", "subscription") and the classifier's decision boundary is weak here.

*Example:* "I can't get into my account and you're still charging me $9.99 every month" → predicted `account_access`, true `billing_payment`.

*Hypothesis:* These messages are genuinely multi-intent. A multi-label classifier or a joint intent + sub-intent taxonomy would handle them better.

### Failure mode 2: Very short messages default to general_inquiry

Messages under 5 words ("broken again", "nothing works", "fix this") produce embedding vectors with low magnitude that cluster near the centroid of the space. The classifier consistently routes these to `general_inquiry` because that intent's prototype is closest to the origin. They should mostly escalate.

*Hypothesis:* Add a "message too short to classify" branch that routes directly to escalation rather than forcing a classification.

### Failure mode 3: Feature requests with strong emotion get escalated

"PLEASE add lyrics support I have been asking for YEARS" — high capitalization and repetition drives up the complexity and amplifier word scores. The escalation engine scores this 0.52 (just above threshold) and escalates it. Feature requests should never escalate; the fix is a hard intent-level override.

### Failure mode 4: LLM generates overly generic replies for content issues

When a customer reports a missing album, the RAG retrieves similar historical replies. But Spotify's historical replies to content issues are often "try searching again" or "this may not be available in your region" — both unhelpful and mildly condescending when the issue is a genuine licensing gap. The agent inherits this pattern from the training data.

*Hypothesis:* Separate the retrieval pool by intent and tune the system prompt for `content_unavailable` to explicitly acknowledge that content removal is a rights/licensing issue.

### Failure mode 5: Sarcasm is invisible to the sentiment signal

TextBlob polarity on "Oh great, another amazing update that broke everything again" returns approximately +0.3 (positive). The escalation engine treats this as a calm message and doesn't escalate. In practice, sarcastic negative feedback is often more serious than direct negative feedback — the customer is past frustration and into mockery.

*Hypothesis:* Use a Twitter-specific sentiment model (cardiffnlp/twitter-roberta-base-sentiment) that was trained to handle social-media sarcasm.

---

## What is misleading about the headline number

Intent accuracy 0.81 is the number I'd put in a slide. Here is what it obscures:

**The golden set was built by the classifier.** Candidate messages were bucketed by the classifier's predicted intent before I sampled and labeled them. High-confidence examples are over-represented because I specifically sampled more from them. The evaluation set is not independent of the classifier; it's partially shaped by it. True out-of-distribution accuracy is likely 5-8 points lower.

**The taxonomy was defined from the data.** The 7 intents were derived by clustering the same corpus the model trains on. This is circular. A model trained and evaluated on its own taxonomy will always look better than one evaluated against a pre-existing external taxonomy.

**general_inquiry absorbs noise.** The classifier can dump uncertain predictions into general_inquiry without penalty to the other classes. If I removed general_inquiry and forced 6-way classification, accuracy drops to ~0.74.

**Evaluation set size is small.** With 200 examples and 7 classes (~28 per class), a 95% confidence interval on per-class F1 is roughly ±0.10. The per-class numbers should be interpreted with that uncertainty in mind.

---

## What I'd do with one more week

1. **Twitter-specific sentiment.** Replace TextBlob with cardiffnlp/twitter-roberta-base-sentiment. Should fix the sarcasm blindspot and reduce escalation false positives by ~30%.

2. **Fine-tune DistilBERT on the golden set.** 200 examples is enough for fine-tuning a pre-trained model. Expected F1 gain: +7-10 points, pushing above 0.88.

3. **Per-intent escalation thresholds.** Hard-suppress escalation for feature_request. Lower the threshold for billing_payment and account_access to 0.35.

4. **Thread-context classification.** Pass the prior 1-2 messages as context to the classifier. Resolves most billing/account conflation at the cost of requiring a conversation state store.

5. **Live evaluation harness.** Route 1% of real Spotify Twitter traffic through the agent and measure re-contact rate (did the customer have to follow up?). All other metrics are proxies; this is the real signal.
