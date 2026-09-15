# Decision Log

15 non-obvious design decisions and why I made them.

---

**1. Chose Spotify over Amazon, Apple, or Delta**

Spotify has a high tweet volume in the dataset (~30k threads), a clear and consistent brand voice, and well-defined failure modes that map cleanly to distinct intents. Amazon threads are dominated by order-tracking complaints that require backend access to resolve — the agent would be mostly useless. Apple has good volume but the thread structure is messier because @AppleSupport handles iOS, Mac, Watch, and TV all in the same handle. Spotify keeps support to music + app issues, which makes intent taxonomy tractable.

---

**2. Two-stage classifier (embedding + LLM refinement) rather than LLM-first**

Running an LLM on every message adds 600-800ms latency and burns through free-tier quota fast. The embedding classifier is essentially free after the first pass and handles 80-85% of cases with confidence above 0.80. LLM refinement (via Groq, using qwen/qwen3.8-27b) is reserved for the uncertain minority -- roughly 15-20% of messages.

---

**3. `all-mpnet-base-v2` over `all-MiniLM-L6-v2`**

MiniLM is faster and smaller but mpnet produces meaningfully better embeddings on short social-media text (higher Spearman correlation on STS benchmarks). The quality difference shows up in cluster cohesion when I ran k-means. MiniLM would have been fine for a prototype; since this is going to be evaluated, I chose quality over speed.

---

**4. 7 intents, not 5 or 12**

I initially ran k-means with k=8 and got 8 clusters. Two of them were clearly the same thing (app crashes and playback failures were split by the clustering even though they deserve the same escalation path). After LLM-guided merging I landed at 7. Going to 5 would have required collapsing content_unavailable into app_bug, which conflates things that need very different replies. Going to 12 would have created intents with fewer than 20 training examples each — too few for the logistic regression head to generalize.

---

**5. FAISS IndexFlatIP over HNSW**

For a dataset this size (up to 50k vectors at 768 dimensions), exact search with IndexFlatIP is fast enough (< 50ms per query on CPU) and gives exact results. HNSW is an approximate algorithm that trades recall for speed. I'd switch to HNSW if the index grew past ~500k vectors, but for this use case the added complexity and the risk of missing genuinely relevant retrievals isn't worth it.

---

**6. Temporal weighting with exponential decay (half-life 365 days)**

Spotify's support tone changed over the 2017-2020 period in the dataset. Early tweets are more formulaic and verbose; later ones are shorter and more empathetic. Without recency weighting, the retrieval would sometimes surface three-year-old replies as the closest match and the LLM would mimic their outdated tone. The 365-day half-life means a reply from 2 years ago scores 50% of a same-age reply at equal cosine similarity. The parameter is in config.yaml and easy to tune.

---

**7. Escalation recall >> escalation precision (threshold at 0.50, biased low)**

In a support context, the cost of a false negative (not escalating when we should) is higher than the cost of a false positive (escalating unnecessarily). A customer who got a bad auto-reply and had to escalate themselves is angrier than a customer who got a human when they could have been auto-handled. So I set the threshold conservatively and accepted more unnecessary escalations. The current ~35% escalation rate is high for a real deployment — you'd tune it down after watching real outcomes — but it's right for a cautious first version.

---

**8. Cardiff NLP twitter-roberta-base-sentiment over TextBlob**

I started with TextBlob because it has no dependencies and is easy to inspect. It fell apart immediately on sarcasm -- "oh great, another broken update" registered as positive. I swapped in cardiffnlp/twitter-roberta-base-sentiment-latest, which was trained on 124M tweets and handles negation and amplifier language correctly. The tradeoff is a ~300ms cold-start on first load and the full transformers dependency. On the server (Render free tier, 512 MB RAM limit) transformers isn't installed, so it falls back to TextBlob there -- but the escalation engine adds an amplifier-word bonus ("furious", "unacceptable", "worst") on top of the polarity score to partially compensate.

---

**9. Golden set stratification: 80% high-confidence + 20% hard examples**

A golden set built only from high-confidence classifier predictions would be easy and would make the classifier look better than it is. Including 20% low-confidence ("hard") examples means the evaluation has teeth near decision boundaries, which is exactly where real failure modes live. The 80/20 split is a judgment call — I wanted enough hard examples to matter without making the overall eval set too noisy.

---

**10. LLM judge runs 3 times, not once**

Running the judge once and taking the result is hiding variance. LLMs have non-trivial token-sampling randomness even at temperature 0. By running 3 times (once at temp 0, twice at temp 0.3) and reporting the mean plus the consistency score, I make the reliability of the judge measurement explicit. A judge with consistency 0.65 should be trusted less than one at 0.92. Most papers don't report this. In practice the LLM judge run was skipped in the final eval due to Groq's 200k token/day free-tier limit -- the harness code is complete and works, but the final eval_summary reports classifier metrics only.

---

**11. Separate FAISS index from classifier, not a unified retrieval-only approach**

An alternative design would skip intent classification entirely and just do nearest-neighbour retrieval on the full message, trusting that similar messages get similar replies. That works passably for reply generation but gives you no intent signal for escalation. The escalation engine needs a discrete intent label to set the sensitivity threshold. Keeping classification and retrieval separate lets each component be tuned independently.

---

**12. Kept RAG retrieval at k=5, not k=10 or k=1**

k=1 is too brittle — one retrieved example can be oddly specific or off. k=10 makes the prompt very long and introduces noise from lower-similarity examples. k=5 with temporal reranking gives the LLM enough variety to synthesize from without overwhelming the context window or degrading focus. I tested k=3, 5, and 8 on 50 examples and k=5 produced the best judge scores.

---

**13. Regex-based security signal over LLM-based**

Security keyword detection (hacked, fraud, lawsuit) doesn't need a model. Regex is faster, deterministic, and easier to audit. An LLM-based security classifier would be more accurate on edge cases but would add latency and cost to every single message just for a signal that fires on < 5% of traffic. The regex approach has known false-negative failure modes (obfuscated spelling) but those are acceptable given how rare they are.

---

**14. Storing QA pairs (customer + reply) not just embeddings**

The FAISS metadata file stores the full customer text and Spotify reply alongside each vector, not just a tweet ID or a pointer to a database. This makes the index self-contained — you can run the full pipeline without a live database connection. The tradeoff is disk space (a few hundred MB for 50k pairs), which is fine. If this were production infrastructure I'd store IDs and hit a database, but for a reproducible demo self-containment is worth the disk cost.

---

**15. Fine-tuned DistilBERT as a second classifier alongside the embedding approach**

Rather than just describing fine-tuning as a "what I'd do next" item, I actually did it on the 196-example golden set for 8 epochs. Val accuracy and F1 both hit 1.0. I treat this result honestly in the report -- at 196 examples and 28 per class, val F1 of 1.0 means the model can memorize the set, not that it generalises perfectly. The value of doing it is showing the pipeline works end-to-end, not claiming a better headline number. The saved model is in `models/distilbert-intent/` (gitignored due to size).

---

**16. Report focuses on failure over success**

The assignment said "the proof is worth more than the system." I spent proportionally more time on failure analysis, calibration measurement, and the "what is misleading" section than on hyperparameter tuning. A system that scores 93.4% and knows exactly why it fails at the remaining 6.6% is more trustworthy than one that scores higher with no analysis of the errors. The headline number is not the point.
