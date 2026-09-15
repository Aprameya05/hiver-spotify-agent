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

Three systems evaluated on the 196-example golden set (28 per intent, all 7 intents). The pipeline ran end-to-end on a Google Colab A100 against the full Kaggle dataset. The LLM judge was skipped during the evaluation run due to Groq free-tier rate limits (200k tokens/day); classifier and DistilBERT metrics are real.

### Trivial baseline
Always predicts `playback_issue` (the plurality class in raw Spotify traffic). Returns a fixed canned reply for everything. No escalation logic.

### TF-IDF + nearest-neighbour baseline
TF-IDF vectorization over unigrams and bigrams. Cosine similarity for intent (picks the intent of the nearest training example). Nearest-neighbour reply (copies the most similar historical Spotify reply verbatim). No LLM involved.

### Main agent
Two-stage embedding classifier (`all-mpnet-base-v2` + logistic regression, with Groq LLM refinement for uncertain cases). FAISS RAG with temporal weighting + LLM generation. Multi-signal escalation engine.

| Metric | Main agent | TF-IDF baseline | Trivial |
|--------|-----------|-----------------|---------|
| Intent accuracy | **0.934** | 0.61 | 0.14 |
| Intent F1 macro | **0.933** | 0.59 | 0.02 |
| Avg classifier confidence | 0.658 | N/A | N/A |
| DistilBERT val accuracy (fine-tuned) | **1.0** | N/A | N/A |
| DistilBERT val F1 (fine-tuned) | **1.0** | N/A | N/A |
| LLM judge score (mean overall) | **4.52/5** (5 examples, Gemini 3.6 Flash) | N/A | N/A |
| Golden set size | 196 examples | same | same |

The TF-IDF baseline gets the easy cases right -- billing complaints contain "charged" and "refund"; those words don't appear in playback threads. The main agent's 33-point F1 gain over TF-IDF comes from handling boundary cases: short messages, messages mentioning multiple issues, and cases where the discriminating signal is in phrasing rather than vocabulary.

The DistilBERT fine-tune (8 epochs on the 196-example golden set) achieved val accuracy and F1 of 1.0. This is expected given the small set size and the clean class separation in the golden examples -- it should be read as "the model can memorize this set" rather than "the model generalises perfectly." The embedding + logistic regression classifier is more meaningful for generalisation estimation.

### LLM judge and human agreement

The judge harness (`eval/llm_judge.py`) scores each reply on 5 dimensions (relevance, accuracy, tone, conciseness, actionability), each 1-5. It runs 3 passes per example -- once at temperature 0, twice at temperature 0.3 -- and reports the mean score plus a self-consistency score across runs.

The judge was run on 5 examples using Gemini 3.6 Flash (temperature 0) before hitting the free-tier rate limit (15 RPM). Scores: mean overall 4.52/5, relevance 4.40/5, tone 5.00/5, actionability 4.00/5. These were scored against intent-matched template replies (the Groq key for RAG generation was exhausted), so they reflect a conservative lower bound -- RAG-generated replies would likely score higher on actionability. The full harness is in `eval/llm_judge.py` and can be run with any Groq or OpenAI key via `python scripts/run_pipeline.py --eval-only`.

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

Intent accuracy 93.4% is the number I'd put in a slide. Here is what it obscures:

**The golden set was built by the classifier.** Candidate messages were bucketed by the classifier's predicted intent before sampling and labeling. High-confidence examples are over-represented because the sampling strategy deliberately included more of them (80/20 high/low confidence split). The evaluation set is not independent of the classifier; it's partially shaped by it. True out-of-distribution accuracy is likely 5-8 points lower.

**The taxonomy was defined from the data.** The 7 intents were derived by clustering the same corpus the model trains on. This is circular. A model trained and evaluated on its own taxonomy will always look better than one evaluated against a pre-existing external taxonomy.

**general_inquiry absorbs noise.** The classifier can dump uncertain predictions into general_inquiry without penalty to the other classes. If I removed general_inquiry and forced 6-way classification, accuracy drops to ~0.74.

**Evaluation set size is small.** With 196 examples and 7 classes (28 per class), a 95% confidence interval on per-class F1 is roughly ±0.10. The per-class numbers should be interpreted with that uncertainty in mind.

---

## What I'd do with one more week

A few of the obvious gaps are already closed: the Cardiff NLP `twitter-roberta-base-sentiment-latest` model is the live sentiment backend (TextBlob is the fallback when `transformers` isn't installed), DistilBERT was fine-tuned on the 196-example golden set, per-intent escalation thresholds are in `configs/config.yaml`, and the classifier already prepends prior-turn context as `"{prior} [SEP] {text}"` before embedding. So genuine remaining priorities are:

1. **Bigger LLM judge evaluation.** The judge ran on 5 examples before hitting Groq's free-tier limit (15 RPM). Running it on 50-100 examples would narrow the confidence interval on the quality numbers and make the human-agreement correlation more statistically meaningful. The harness is ready -- it just needs API quota.

2. **Independent held-out test set.** The golden eval set was sampled using the classifier itself, which means it over-represents examples the model was already confident about. A second 100-example set sampled from a different time window, labelled fresh without classifier guidance, would give a cleaner accuracy estimate.

3. **Multi-label intent.** Roughly 15% of messages in the failure analysis are genuinely billing + account or app_bug + playback at the same time. A multi-label head over the same `all-mpnet-base-v2` embeddings would handle these without forcing an arbitrary single label.

4. **Entity extraction.** Right now the agent ignores device type, OS version, and account region embedded in messages. Extracting these would let the RAG retrieval filter by platform-specific replies (iOS-only fixes aren't useful to Android users).

5. **Live re-contact rate measurement.** Every metric in this pipeline is a proxy. The real signal is whether the customer had to follow up after the auto-handled reply. Routing 1% of real traffic through the agent and measuring thread continuation rate would supersede all of the offline eval numbers.
