"""
RAG-based reply generator for Spotify support.

Architecture:
  1. Build a FAISS flat-IP index over all historical Spotify reply pairs.
  2. At query time: embed the incoming customer message, retrieve top-k
     most similar historical pairs (weighted by recency).
  3. Feed retrieved examples + current message to an LLM that drafts a
     grounded reply following Spotify's established tone and resolution patterns.

Novelty: temporal weighting. Replies from the last 12 months get a +20%
score boost so the agent learns from the most current brand voice — important
because Spotify's support tone shifted noticeably over the years in the data.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils import ensure_dir, get_logger, load_config, timeit

log = get_logger(__name__)

SPOTIFY_TONE_GUIDE = """
You are a Spotify customer support agent replying on Twitter.

Tone rules (derived from Spotify's actual support history):
- Empathetic but concise — Twitter limits length, so be direct.
- Always acknowledge the customer's frustration briefly before suggesting a fix.
- Suggest one actionable step, not a laundry list.
- Offer to escalate ("DM us") if the fix requires account details.
- Do not over-promise ("I'll make sure this gets fixed for you").
- Sign off naturally — avoid robotic closings like "Best regards".
- Use plain English, no jargon.
- Keep the reply under 240 characters when possible; Twitter hard limit is 280.
"""


class RetrievedExample(NamedTuple):
    customer_text: str
    spotify_reply: str
    similarity: float
    timestamp: str | None


class GeneratedReply(NamedTuple):
    reply: str
    retrieved_examples: list[RetrievedExample]
    model_used: str
    prompt_tokens: int
    completion_tokens: int


class ReplyGenerator:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        model_name = self.cfg["intents"]["embedding_model"]
        self.top_k = self.cfg["reply_generator"]["retrieval_top_k"]
        self.temporal_decay_days = self.cfg["reply_generator"]["temporal_decay_days"]
        self.llm_model = self.cfg["reply_generator"]["llm_model"]
        self.temperature = self.cfg["reply_generator"]["temperature"]
        self.max_tokens = self.cfg["reply_generator"]["max_reply_tokens"]

        self.index_path = Path(self.cfg["paths"]["index_path"])
        self.meta_path = Path(self.cfg["paths"]["metadata_path"])

        log.info("Loading embedding model: %s", model_name)
        self.encoder = SentenceTransformer(model_name)
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: list[dict] = []

    # ------------------------------------------------------------------ #
    # Index management                                                     #
    # ------------------------------------------------------------------ #

    @timeit
    def build_index(self, qa_pairs: list[dict]) -> None:
        """
        Embed all customer messages and store with metadata.
        Index: FAISS IndexFlatIP (inner product on L2-normalised vectors = cosine sim).
        """
        log.info("Building FAISS index over %d QA pairs …", len(qa_pairs))
        customer_texts = [p["customer_text"] for p in qa_pairs]

        embeddings = self.encoder.encode(
            customer_texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=128,
        ).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.metadata = qa_pairs

        # Persist
        ensure_dir(self.index_path.parent)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "w") as f:
            for pair in qa_pairs:
                f.write(json.dumps(pair) + "\n")

        log.info("Index saved — %d vectors, dim=%d", self.index.ntotal, dim)

    def load_index(self) -> None:
        log.info("Loading FAISS index from %s", self.index_path)
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = []
        with open(self.meta_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.metadata.append(json.loads(line))
        log.info("Index loaded — %d vectors", self.index.ntotal)

    def _ensure_index(self) -> None:
        if self.index is None:
            if self.index_path.exists():
                self.load_index()
            else:
                raise RuntimeError(
                    "FAISS index not built. Call build_index(qa_pairs) first "
                    "or run scripts/run_pipeline.py."
                )

    # ------------------------------------------------------------------ #
    # Retrieval                                                            #
    # ------------------------------------------------------------------ #

    def _temporal_weight(self, timestamp_str: str | None) -> float:
        """
        Exponential recency boost. Recent examples get up to +20% score multiplier.
        half-life = temporal_decay_days / ln(2)
        """
        if not timestamp_str:
            return 1.0
        try:
            ts = datetime.strptime(timestamp_str, "%a %b %d %H:%M:%S %z %Y")
            now = datetime.now(tz=timezone.utc)
            age_days = (now - ts).days
            decay = math.exp(-age_days / (self.temporal_decay_days / math.log(2)))
            return 0.8 + 0.2 * decay   # range [0.8, 1.0]
        except Exception:
            return 1.0

    def retrieve(self, customer_text: str) -> list[RetrievedExample]:
        """Retrieve top-k historically similar pairs, weighted by recency."""
        self._ensure_index()
        query_emb = self.encoder.encode(
            [customer_text], normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")

        k = min(self.top_k * 3, self.index.ntotal)   # over-fetch then re-rank
        scores, indices = self.index.search(query_emb, k)

        candidates: list[tuple[float, int]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self.metadata[idx]
            weight = self._temporal_weight(meta.get("timestamp"))
            candidates.append((float(score) * weight, idx))

        candidates.sort(key=lambda x: x[0], reverse=True)
        top = candidates[: self.top_k]

        return [
            RetrievedExample(
                customer_text=self.metadata[idx]["customer_text"],
                spotify_reply=self.metadata[idx]["spotify_reply"],
                similarity=sim,
                timestamp=self.metadata[idx].get("timestamp"),
            )
            for sim, idx in top
        ]

    # ------------------------------------------------------------------ #
    # Generation                                                           #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        customer_text: str,
        intent: str,
        retrieved: list[RetrievedExample] | None = None,
    ) -> GeneratedReply:
        """
        Draft a grounded reply using RAG.
        """
        if retrieved is None:
            retrieved = self.retrieve(customer_text)

        examples_block = "\n\n".join(
            f"Example {i+1} (similarity {ex.similarity:.2f}):\n"
            f"  Customer: {ex.customer_text}\n"
            f"  Spotify reply: {ex.spotify_reply}"
            for i, ex in enumerate(retrieved)
        )

        prompt = f"""{SPOTIFY_TONE_GUIDE}

---
HISTORICAL EXAMPLES (use these to ground your reply in Spotify's actual patterns):

{examples_block}

---
NOW handle this new customer message:

Customer message: "{customer_text}"
Detected intent category: {intent}

Write a single, direct Twitter reply from Spotify support.
Do NOT copy the examples verbatim — adapt to this specific situation.
Output ONLY the reply text, nothing else."""

        try:
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a Spotify customer support agent."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            reply = response.choices[0].message.content.strip()
            usage = response.usage
            return GeneratedReply(
                reply=reply,
                retrieved_examples=retrieved,
                model_used=self.llm_model,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            )
        except Exception as e:
            log.error("LLM generation failed: %s", e)
            # Fallback: return the most similar historical reply with light editing
            if retrieved:
                fallback = f"Hi! {retrieved[0].spotify_reply}"
            else:
                fallback = "Hi! We're sorry to hear you're having trouble. Please DM us your account details so we can help."
            return GeneratedReply(
                reply=fallback,
                retrieved_examples=retrieved,
                model_used="fallback",
                prompt_tokens=0,
                completion_tokens=0,
            )

    def generate_with_retrieval(self, customer_text: str, intent: str) -> GeneratedReply:
        """Full pipeline: retrieve then generate."""
        retrieved = self.retrieve(customer_text)
        return self.generate(customer_text, intent, retrieved)


if __name__ == "__main__":
    # Quick smoke test — requires index to be built
    gen = ReplyGenerator()
    test_msg = "spotify keeps buffering every few seconds on my iphone, tried reinstalling"
    try:
        result = gen.generate_with_retrieval(test_msg, "playback_issue")
        print("Generated reply:", result.reply)
        print("Retrieved examples:")
        for ex in result.retrieved_examples:
            print(f"  [{ex.similarity:.2f}] {ex.customer_text[:80]}")
    except RuntimeError as e:
        print(f"Index not built yet: {e}")
        print("Run: python scripts/run_pipeline.py --build-index")
