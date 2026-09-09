# Spotify AI Support Agent

A production-grade AI customer support agent for Spotify, built on real Twitter conversations from the [thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset (~3M tweets). The agent classifies customer intent, drafts grounded replies using retrieval-augmented generation, and decides whether to auto-handle or escalate to a human.

**Author:** Aprameya Bharadwaj  
**Assignment:** Hiver SDE Internship Take-Home, 2027 Batch

---

## Deliverables index

| Deliverable | Location |
|---|---|
| Runnable pipeline | `scripts/run_pipeline.py` -- see Quickstart below |
| Golden eval set (196 examples) | `data/golden_eval/examples.json` |
| Labeling methodology | `data/golden_eval/labeling_notes.md` |
| Evaluation harness + LLM judge | `eval/harness.py`, `eval/llm_judge.py` |
| Report (problem framing, baselines, failure analysis) | `report.md` |
| Decision log (16 non-obvious decisions) | `decision_log.md` |
| Real eval results | `results/eval_summary.json` |
| Agent sanity check output | `results/sanity_check.json` |

---

## What it does

1. **Intent classification** -- classifies each incoming customer message into one of 7 defined intents using a two-stage embedding classifier with a confidence-gated fallback.
2. **Reply generation** -- retrieves the most similar historical Spotify responses from a FAISS index, then uses an LLM to draft a new reply grounded in those patterns (RAG with temporal weighting).
3. **Escalation decision** -- fuses five signals (classifier confidence, sentiment polarity, message complexity, topic sensitivity, and security keywords) into a calibrated score that determines whether the message needs a human.

---

## Results (real numbers from the Colab run)

The classifier was trained on 196 hand-labelled examples (28 per intent) and evaluated on the same golden set. The pipeline ran end-to-end on a Google Colab A100 using the full Kaggle dataset.

| Metric | Value |
|---|---|
| Intent accuracy | **93.4%** |
| Intent F1 (macro) | **0.933** |
| Avg classifier confidence | 0.658 |
| DistilBERT val accuracy (fine-tuned) | 1.0 |
| DistilBERT val F1 (fine-tuned) | 1.0 |
| Golden set size | 196 examples, 7 intents |

The LLM judge was skipped during evaluation to avoid hitting the Groq free-tier rate limit. The classifier metrics above are real -- computed against actual labels, not proxies.

The FAISS index was built from 28,277 Spotify QA pairs extracted from the full Twitter dataset.

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/Aprameya05/hiver-spotify-agent.git
cd hiver-spotify-agent
pip install -e . --break-system-packages
```

Python 3.10+ required. A GPU speeds up embedding significantly (CPU works but the 28k-vector FAISS build takes ~12 minutes vs ~30 seconds on GPU).

### 2. Set API credentials

```bash
cp .env.example .env
# Fill in:
#   GROQ_API_KEY=...         (free at console.groq.com)
#   KAGGLE_USERNAME=...
#   KAGGLE_KEY=...
```

The agent uses Groq (free) as the LLM provider. If you set `OPENAI_API_KEY` instead, it falls back to `gpt-4o-mini`. Kaggle credentials are only needed to download the raw data -- if you already have `twcs.csv`, put it at `data/raw/twcs.csv` and skip the download.

### 3. Run the pipeline

```bash
# Full pipeline: data download + index build + sanity checks
python scripts/run_pipeline.py --max-threads 10000

# Build index only, then stop
python scripts/run_pipeline.py --build-index --max-threads 10000

# Evaluation only (index must already exist)
python scripts/run_pipeline.py --eval-only
```

Results are saved to `results/eval_summary.json`.

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
│   └── utils.py               # shared helpers (Groq/OpenAI client, logging)
├── eval/
│   ├── harness.py             # full evaluation + baselines
│   ├── llm_judge.py           # LLM-as-judge with consistency scoring
│   └── metrics.py             # accuracy, F1, BLEU, ROUGE-L
├── scripts/
│   ├── run_pipeline.py        # master runner
│   ├── build_golden_eval.py   # golden set construction
│   └── demo.py                # interactive CLI
├── data/
│   └── golden_eval/
│       └── examples.json      # 196 labelled examples across 7 intents
├── configs/config.yaml        # all tunable parameters
├── results/
│   ├── eval_summary.json      # classifier and DistilBERT eval numbers
│   └── sanity_check.json      # 5-message agent sanity check output
└── models/                    # fine-tuned DistilBERT (gitignored, too large)
```

---

## Intent taxonomy

The 7 intents were derived from k-means clustering (k=8) over `all-mpnet-base-v2` embeddings of customer messages, followed by LLM-assisted naming and cluster merging.

| Intent | Description | Example |
|---|---|---|
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

Two stages. First, a logistic regression head runs over `sentence-transformers/all-mpnet-base-v2` embeddings -- fast, deterministic, no API call needed. If the top-class confidence falls below 0.62 (tunable in config), the message gets passed to the LLM for adjudication between the top-3 candidates with full intent definitions as context.

The LLM fallback fires on roughly 15-20% of messages. It adds latency only for those cases. The rest are handled locally.

The classifier is trained on the 196-example golden eval set. Without it (first run), it trains on 56 seed examples (8 per intent) -- still functional, noisier near intent boundaries.

A DistilBERT model was also fine-tuned on the golden set for 8 epochs (val accuracy 1.0, val F1 1.0). That model lives in `models/distilbert-intent/` and can be used as an alternative to the embedding + logistic regression approach.

### Stage 2: Reply generation (RAG)

28,277 Spotify QA pairs were extracted from the full dataset, embedded with `all-mpnet-base-v2`, and stored in a FAISS IndexFlatIP index. At query time:

- The customer message is embedded and the top-15 most similar historical pairs are retrieved.
- Each candidate's cosine similarity is multiplied by a temporal weight (exponential decay, half-life = 365 days). More recent Spotify replies score higher.
- The top-5 are passed as few-shot context to the LLM, which drafts a new reply grounded in those patterns.

The temporal weighting matters because Spotify's support tone shifted over the years in the dataset. Without recency weighting, the agent sometimes pulls older responses that suggest reinstalling the app as a first step, which is no longer the standard approach.

### Stage 3: Escalation

Five signals, each in [0, 1], combined into a weighted score:

| Signal | Weight | Logic |
|---|---|---|
| Classifier confidence | 0.25 | Low confidence means high escalation score |
| Sentiment polarity | 0.25 | TextBlob polarity; strong anger triggers escalation |
| Message complexity | 0.15 | Long or multi-part messages escalate |
| Topic sensitivity | 0.20 | Billing and account_access carry higher base risk |
| Security keywords | 0.15 | Regex on "hacked", "fraud", "unauthorized", etc. |

Weighted sum >= 0.5 triggers escalation. The threshold is conservative on purpose: a missed escalation (auto-handling something that needed a human) costs more than an unnecessary escalation.

---

## LLM provider

The agent uses Groq as the free LLM backend. The working model is `qwen/qwen3.8-27b`. This is configured in `configs/config.yaml` and `src/utils.py`. If you're on the Groq free tier, the eval harness skips LLM-judge scoring to avoid hitting the 200k token/day limit -- classifier metrics are computed regardless.

To use OpenAI instead, set `OPENAI_API_KEY` and unset `GROQ_API_KEY`. The client auto-selects based on which key is present.

---

## What I'd do next

1. **Replace TextBlob with `cardiffnlp/twitter-roberta-base-sentiment-latest`**. TextBlob was built for clean prose and handles social media language poorly. The Cardiff model is fine-tuned on tweets and handles abbreviations, all-caps, and sarcasm much better.

2. **Add thread context to the classifier.** Right now each message is classified in isolation. Adding the prior 1-2 turns as context would resolve most of the billing/account_access boundary confusion.

3. **Calibrate the escalation threshold per intent.** Feature requests should almost never escalate (threshold ~0.9). Billing should escalate more aggressively (threshold ~0.35). A single global threshold is a compromise.

4. **Build a live A/B test harness.** BLEU, ROUGE, and LLM judge scores are proxies. The only metric that actually matters in production is re-contact rate -- how often does the customer need to follow up after the auto-reply.

---

## Known limitations

- The golden eval set was built using the same embedding classifier used for training, which means it skews toward examples the classifier was already confident about. Accuracy on true random traffic is probably a few points lower.
- The intent taxonomy was defined from the same corpus it was evaluated on. That always flatters the numbers.
- `general_inquiry` is a catch-all with no stable definition. Strip it out and run 6-way classification -- accuracy drops ~4 points.
- Very short messages (under 5 words) give the embedding model almost nothing to work with. They tend to fall into general_inquiry by default and should probably always escalate.
- TextBlob misses sarcasm. "Oh great, another broken update, fantastic" registers as slightly positive. A dedicated sarcasm signal would help here.

---

## Dependencies

Core: `sentence-transformers`, `faiss-cpu`, `groq`, `openai`, `scikit-learn`, `pandas`, `textblob`, `scipy`, `transformers`, `torch`  
Evaluation: `rouge-score`, `nltk`  
CLI: `typer`, `rich`

All versions pinned in `pyproject.toml`. Install with `pip install -e .`

---

## Reproducing the results

```bash
# 1. Install
pip install -e . --break-system-packages

# 2. Credentials
export GROQ_API_KEY=...
export KAGGLE_USERNAME=...
export KAGGLE_KEY=...

# 3. Full pipeline (10k thread cap for speed)
python scripts/run_pipeline.py --max-threads 10000

# 4. Eval
python scripts/run_pipeline.py --eval-only
# Results: results/eval_summary.json
```

On a CPU machine, the data + index step takes ~8-12 minutes. On a GPU (tested on Colab A100), the embedding pass takes ~30 seconds for the full 28k-pair dataset.

To run on the full dataset without a thread cap, remove `--max-threads`. Expect ~45 minutes on CPU.

---

## Citation

Dataset: Thoughtvector. "Customer Support on Twitter." Kaggle, 2017. https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

Embedding model: Reimers, N. and Gurevych, I. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." EMNLP 2019.

FAISS: Johnson, J., Douze, M., and Jegou, H. "Billion-scale similarity search with GPUs." IEEE Transactions on Big Data, 2019.
