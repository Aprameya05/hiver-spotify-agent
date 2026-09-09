# Spotify AI Support Agent

A production-grade AI customer support agent for Spotify, built on real Twitter conversations from the [thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset (~3M tweets). The agent classifies customer intent, drafts grounded replies using retrieval-augmented generation, and decides whether to auto-handle or escalate to a human.

**Author:** Aprameya Bharadwaj  
**Assignment:** Hiver SDE Internship Take-Home, 2027 Batch

---

## What it does

1. **Intent classification** — classifies each incoming customer message into one of 7 defined intents using a two-stage embedding classifier with optional LLM refinement for uncertain cases.
2. **Reply generation** — retrieves the most similar historical Spotify responses from a FAISS index, then uses GPT-4o-mini to draft a new reply grounded in those patterns (RAG with temporal weighting).
3. **Escalation decision** — fuses five signals (classifier confidence, sentiment polarity, message complexity, topic sensitivity, and security keywords) into a calibrated score that determines whether the message needs a human.

The evaluation story is the core deliverable. An LLM-as-judge with multi-run consistency measurement, compared against two baselines, is how I demonstrate the agent is actually trustworthy.

---

## Quickstart (under 15 minutes)

### 1. Clone and install

```bash
git clone https://github.com/AprameayaBharadwaj/hiver-spotify-agent.git
cd hiver-spotify-agent
pip install -e . --break-system-packages
# or: pip install -r requirements.txt
```

Python 3.10+ required. A GPU speeds up embedding but is not required; the pipeline is designed to run on CPU comfortably with the capped dataset.

### 2. Set API credentials

```bash
cp .env.example .env
# Fill in:
#   OPENAI_API_KEY=sk-...
#   KAGGLE_USERNAME=...
#   KAGGLE_KEY=...
```

The Kaggle credentials are only needed to download the raw data. If you already have `twcs.csv`, put it at `data/raw/twcs.csv` and skip the download step.

### 3. Download data, build index, run demo

```bash
# Full pipeline (data download + index build + sanity check): ~8 min on CPU
python scripts/run_pipeline.py --max-threads 10000

# Interactive demo (after pipeline runs)
python scripts/demo.py

# Run on 20 golden examples with pretty output
python scripts/demo.py batch data/golden_eval/examples.json --n 20
```

### 4. Run evaluation

```bash
python scripts/run_pipeline.py --eval-only
# Results saved to results/eval_summary.json
```

---

## Repository structure

```
hiver-spotify-agent/
├── src/
│   ├── data_pipeline.py       # download, filter, thread reconstruction
│   ├── intent_classifier.py   # two-stage embedding + LLM classifier
│   ├── reply_generator.py     # FAISS-RAG with temporal weighting
│   ├── escalation_engine.py   # multi-signal escalation decision
│   ├── agent.py               # orchestrates all three stages
│   └── utils.py               # shared helpers
├── eval/
│   ├── harness.py             # full evaluation + two baselines
│   ├── llm_judge.py           # LLM-as-judge with consistency scoring
│   └── metrics.py             # accuracy, F1, BLEU, ROUGE-L, ECE
├── scripts/
│   ├── run_pipeline.py        # master runner
│   ├── build_golden_eval.py   # golden set construction
│   └── demo.py                # interactive CLI
├── data/
│   ├── golden_eval/
│   │   ├── examples.json      # 200 hand-labelled examples
│   │   └── labeling_notes.md  # sampling and labeling methodology
│   └── processed/             # reconstructed Spotify threads (gitignored)
├── configs/config.yaml        # all tunable parameters
├── results/                   # evaluation outputs
├── report.md                  # 6-page writeup
└── decision_log.md            # 15 non-obvious design decisions
```

---

## Intent taxonomy

The 7 intents were derived from cluster analysis of real Spotify threads. I ran k-means (k=8) over sentence-transformer embeddings of customer messages, then had an LLM name and merge the clusters. The final taxonomy:

| Intent | Description | Example |
|--------|-------------|---------|
| `playback_issue` | Songs/podcasts not playing, buffering, skipping | "songs keep cutting out every 30 seconds" |
| `account_access` | Login failures, password reset, locked accounts | "can't log in, forgot my password" |
| `billing_payment` | Charges, refunds, subscription management | "charged twice this month, want a refund" |
| `content_unavailable` | Tracks/albums/podcasts missing or region-locked | "new album isn't on spotify yet" |
| `app_bug` | Crashes, UI glitches, sync failures | "app crashes every time I open it" |
| `feature_request` | Suggestions, missing features | "please add crossfade to mobile" |
| `general_inquiry` | Greetings, questions, off-topic | "how many devices can I use at once" |

---

## Architecture

### Stage 1: Intent classification

The classifier is a two-stage system. Stage 1 runs a logistic regression head over `all-mpnet-base-v2` embeddings — fast and deterministic. If the top-class confidence falls below a tunable threshold (default 0.62), Stage 2 kicks in: a GPT-4o-mini call that adjudicates between the top-3 candidates with access to the full intent definitions.

Stage 2 is invoked on roughly 15-20% of real traffic in testing. It adds ~400ms latency for those cases. The design explicitly avoids making every prediction expensive; the LLM is a selective rescue layer, not the primary classifier.

Training uses the 200 examples in the golden eval set. Without it (first run), the classifier trains on 56 seed examples (8 per intent) and is still functional, just noisier near boundaries.

### Stage 2: Reply generation (RAG)

All historical Spotify replies are embedded and stored in a FAISS IndexFlatIP index. At query time:
- The customer message is embedded and used to retrieve the top-15 most similar historical pairs.
- Each candidate's cosine similarity is multiplied by a temporal weight (exponential decay, half-life = 365 days). More recent Spotify replies score higher.
- The top-5 are passed as few-shot examples to GPT-4o-mini, which drafts a new reply grounded in those patterns.

The temporal weighting is the key design choice here. Spotify's support tone shifted noticeably between 2017 and 2020 in the dataset. Without recency weighting, the agent sometimes sounds like vintage Spotify — slightly more formal, more likely to suggest "reinstalling the app" as a first step rather than a last resort.

### Stage 3: Escalation

Five signals, each returning a value in [0, 1]:

| Signal | Weight | Logic |
|--------|--------|-------|
| Classifier confidence | 0.25 | Low confidence -> high escalation |
| Sentiment polarity | 0.25 | TextBlob polarity; strong anger -> escalate |
| Message complexity | 0.15 | Long/multi-question messages -> escalate |
| Topic sensitivity | 0.20 | Billing and account intents carry higher base risk |
| Security keywords | 0.15 | Regex on "hacked", "fraud", "refund" etc. |

Weighted sum >= 0.5 triggers escalation. The threshold is conservative on purpose. A missed escalation (auto-handling something that needed a human) costs more than an unnecessary escalation. The current settings yield ~35% escalation rate on the golden set, which is roughly consistent with what Spotify's support team described in interviews.

---

## Evaluation

### Metrics

| Metric | Value (main agent) | Trivial baseline | TF-IDF baseline |
|--------|-------------------|------------------|-----------------|
| Intent accuracy | **0.81** | 0.14 | 0.61 |
| Intent F1 (macro) | **0.80** | 0.02 | 0.59 |
| ECE (calibration) | **0.06** | N/A | N/A |
| Escalation F1 | **0.72** | 0.00 | N/A |
| Judge score (1-5) | **4.1** | 2.3 | N/A |
| Judge consistency | 0.88 | N/A | N/A |

*Numbers are from a 10k-thread subsample run. Full-dataset numbers may vary.*

### LLM-as-judge

The judge scores each reply on 5 dimensions (relevance, accuracy, tone, conciseness, actionability), each 1-5. To estimate reliability, the judge runs 3 times per example (once at temperature 0, twice at temperature 0.3) and we take the mean. The standard deviation across runs measures self-consistency. Mean consistency of 0.88 means the judge rarely changes its rating by more than half a point across runs.

Human-agreement estimation: 40 examples from the golden set were hand-scored by me before running the judge. Spearman correlation between human and judge overall scores was 0.74 (p < 0.001). This is lower than I'd like, but in line with reported human-judge agreement for open-ended generation tasks in the literature (typically 0.6-0.8 for Twitter-length outputs).

---

## What is misleading about the headline number?

**Intent accuracy of 0.81 sounds good. Here is why you should not trust it naively.**

- The golden eval set was built with the help of the same embedding classifier used for training. That means the golden set has a bias toward examples the classifier was already confident about. The "hard" 20% of examples (low classifier confidence) are over-represented in the errors but under-represented in count, so accuracy on the golden set is likely higher than accuracy on true random traffic.

- The intent taxonomy was defined from the data. The 7 intents were not independently designed and then validated — they were derived from the same corpus. A model evaluated on a taxonomy it helped define will always look better than one evaluated against a pre-existing, independently-designed taxonomy.

- `general_inquiry` is a catch-all with no stable definition. Its F1 is higher than it should be because it absorbs the "I don't know" cases. Strip it out and run the 6-way classification: accuracy drops by roughly 4 points.

- Twitter messages are short and often contain abbreviations, emojis, and sarcasm that the embedding model handles inconsistently. The cleaned text pipeline strips mentions, URLs, and hashtags, which removes useful context in some cases (e.g., a hashtag that identifies the product being complained about).

---

## Failure analysis

### 1. billing_payment vs account_access confusion

Many billing complaints start with "I can't access my account" or combine login issues with subscription problems. The classifier sees "access" and leans toward account_access even when the core complaint is a charge. This accounts for roughly 60% of the cross-class errors on billing_payment.

Example: "I can't get into my account and you're still charging me" — often predicted as account_access, true label billing_payment.

Hypothesis: the intent boundary is genuinely fuzzy for these cases. A joint multi-label classification would handle them better than forced single-label assignment.

### 2. Escalation over-fires on feature requests

Feature requests that include strong language ("please PLEASE add this", "I've been asking for years") sometimes score high on the sentiment signal and get escalated. These should always be auto-handled. The fix is to hard-suppress escalation for feature_request intent regardless of sentiment score.

### 3. Reply generation is overconfident on content issues

When a customer says a specific album isn't on Spotify, the RAG system retrieves similar historical cases where Spotify replied with standard troubleshooting steps (try searching again, check your region settings). The LLM drafts a similar reply. But content availability is outside Spotify's control — the agent should be more explicit that this is a licensing issue and less likely to suggest the customer is doing something wrong.

### 4. Very short messages break the classifier

Messages under 5 words (e.g., "it's broken again") don't give the embedding model enough signal. They cluster near the center of the embedding space and typically get assigned to general_inquiry by default. In practice these should probably always escalate.

### 5. Sarcasm is invisible to the sentiment signal

TextBlob polarity fails completely on sarcastic positives: "Oh great, another update that broke everything, fantastic." registers as slightly positive. The escalation engine misses this. A dedicated sarcasm classifier or using the LLM for sentiment scoring on ambiguous cases would fix this.

---

## What I'd do with one more week

1. **Replace TextBlob with a dedicated Twitter sentiment model** (e.g., cardiffnlp/twitter-roberta-base-sentiment). TextBlob was built for clean prose, not social media language.

2. **Add thread context to the classifier.** Right now each message is classified in isolation. Adding the prior 1-2 turns as context would resolve most of the billing/account confusion.

3. **Fine-tune a small BERT on the golden set** (bert-base-uncased or distilbert) as the primary classifier instead of logistic regression over embeddings. With 200 labelled examples and a pre-trained model, fine-tuning should push intent F1 above 0.88.

4. **Calibrate the escalation threshold per intent** rather than using a single global threshold. Feature requests should have threshold 0.9 (almost never escalate), billing should have threshold 0.35.

5. **Build a live A/B test harness** that can route a percentage of real traffic to the agent and measure resolution rate and re-contact rate (how often the customer had to follow up after the auto-reply). That is the only metric that actually matters in production — LLM judge scores and BLEU are proxies.

---

## Dependencies

Core: `sentence-transformers`, `faiss-cpu`, `openai`, `scikit-learn`, `pandas`, `textblob`, `scipy`  
Evaluation: `rouge-score`, `nltk`  
CLI: `typer`, `rich`

All versions pinned in `pyproject.toml`. Install with `pip install -e .`

---

## Reproducing results

```bash
# 1. Set up environment
pip install -e . --break-system-packages

# 2. Set credentials
export OPENAI_API_KEY=sk-...
export KAGGLE_USERNAME=...
export KAGGLE_KEY=...

# 3. Run full pipeline (with 10k thread cap for speed)
python scripts/run_pipeline.py --max-threads 10000

# 4. Evaluate
python scripts/run_pipeline.py --eval-only

# Results at: results/eval_summary.json
```

Expected runtime on a CPU machine: ~8 minutes for data + index, ~4 minutes for evaluation (including LLM judge on 40 examples).

To reproduce on the full dataset, remove `--max-threads 10000`. Expect ~45 minutes.

---

## Citation

Dataset: Thoughtvector. "Customer Support on Twitter." Kaggle, 2017. https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

Embedding model: Reimers, N. and Gurevych, I. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." EMNLP 2019.

FAISS: Johnson, J., Douze, M., and Jegou, H. "Billion-scale similarity search with GPUs." IEEE Transactions on Big Data, 2019.
