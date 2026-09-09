"""
Data pipeline: download the Kaggle Twitter customer-support dataset,
extract Spotify threads, reconstruct multi-turn conversations, and
persist clean JSONL ready for downstream use.

Design note: we keep raw tweets keyed by tweet_id so thread reconstruction
is a single O(n) pass over the index rather than repeated dataframe lookups.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pandas as pd
from tqdm import tqdm

from src.utils import clean_tweet, ensure_dir, get_logger, load_config, timeit

log = get_logger(__name__)

# Author-id patterns that identify SpotifyCares outbound messages
SPOTIFY_PATTERNS = [
    "spotifycares",
    "spotify",
]

INBOUND_AUTHOR_BLOCKLIST = {
    "spotifycares",
    "spotify",
}


def download_dataset(cfg: dict) -> Path:
    """
    Pull the Kaggle dataset if it is not already present.
    Requires KAGGLE_USERNAME and KAGGLE_KEY env vars (or ~/.kaggle/kaggle.json).
    """
    raw_path = Path(cfg["data"]["raw_path"])
    if raw_path.exists():
        log.info("Raw data already present at %s — skipping download", raw_path)
        return raw_path

    ensure_dir(raw_path.parent)
    dataset_slug = cfg["data"]["kaggle_dataset"]
    log.info("Downloading %s from Kaggle …", dataset_slug)

    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset_slug,
         "--unzip", "-p", str(raw_path.parent)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        log.error("Kaggle download failed:\n%s", result.stderr)
        raise RuntimeError("Kaggle download failed — check your API credentials.")

    # The file inside the zip is twcs.csv
    csv_candidates = list(raw_path.parent.glob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError("No CSV found after unzip.")
    csv_candidates[0].rename(raw_path)
    log.info("Dataset saved to %s", raw_path)
    return raw_path


@timeit
def load_raw(raw_path: Path, max_rows: int | None = None) -> pd.DataFrame:
    log.info("Loading raw CSV …")
    df = pd.read_csv(
        raw_path,
        dtype={
            "tweet_id": str,
            "author_id": str,
            "inbound": bool,
            "response_tweet_id": str,
            "in_response_to_tweet_id": str,
        },
        nrows=max_rows,
    )
    log.info("Loaded %d rows", len(df))
    return df


def is_spotify_outbound(author_id: str) -> bool:
    """Return True if the author looks like SpotifyCares / official Spotify support."""
    if not isinstance(author_id, str):
        return False
    lower = author_id.lower()
    return any(p in lower for p in SPOTIFY_PATTERNS)


@timeit
def filter_spotify(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only tweets that belong to a thread where SpotifyCares replied at least once.
    Strategy:
      1. Find all tweet_ids authored by SpotifyCares.
      2. Walk parent pointers to collect the full thread.
    This is more precise than filtering by author_id alone because Twitter threads
    can start from any brand handle.
    """
    log.info("Identifying Spotify outbound tweets …")

    # Index by tweet_id for O(1) parent lookups
    id_to_row: dict[str, pd.Series] = {
        row["tweet_id"]: row
        for _, row in tqdm(df.iterrows(), total=len(df), desc="indexing")
        if isinstance(row.get("tweet_id"), str)
    }

    # Step 1: collect all spotify outbound tweet_ids
    spotify_outbound_ids: set[str] = {
        tid for tid, row in id_to_row.items()
        if is_spotify_outbound(str(row.get("author_id", "")))
    }
    log.info("Found %d Spotify outbound tweets", len(spotify_outbound_ids))

    # Step 2: for each outbound, walk up to the root customer message
    thread_ids: set[str] = set()
    for sid in spotify_outbound_ids:
        thread_ids.add(sid)
        # Walk parents
        current_id = sid
        for _ in range(10):  # max depth guard
            row = id_to_row.get(current_id)
            if row is None:
                break
            parent = row.get("in_response_to_tweet_id")
            if not isinstance(parent, str) or parent not in id_to_row:
                break
            thread_ids.add(parent)
            current_id = parent

    # Also include direct responses that are children of outbound tweets
    response_map: dict[str, list[str]] = defaultdict(list)
    for _, row in df.iterrows():
        parent = row.get("in_response_to_tweet_id")
        if isinstance(parent, str) and parent in spotify_outbound_ids:
            child_id = row.get("tweet_id")
            if isinstance(child_id, str):
                thread_ids.add(child_id)
                response_map[parent].append(child_id)

    spotify_df = df[df["tweet_id"].isin(thread_ids)].copy()
    log.info("Filtered to %d tweets in Spotify threads", len(spotify_df))
    return spotify_df


def _parse_ts(ts: str | float) -> datetime | None:
    if not isinstance(ts, str):
        return None
    try:
        return datetime.strptime(ts, "%a %b %d %H:%M:%S %z %Y")
    except Exception:
        return None


@timeit
def reconstruct_threads(df: pd.DataFrame, max_threads: int | None = None) -> list[dict]:
    """
    Reconstruct full conversation threads as ordered message lists.

    Returns a list of thread dicts:
    {
        "thread_id": str,          # root tweet_id
        "messages": [
            {
                "tweet_id": str,
                "author_id": str,
                "role": "customer" | "spotify",
                "text": str,       # raw
                "clean_text": str, # normalised
                "timestamp": str | None,
                "in_response_to": str | None,
            },
            ...
        ],
        "n_turns": int,
        "has_spotify_reply": bool,
    }
    """
    log.info("Reconstructing threads …")

    id_to_row: dict[str, dict] = {}
    for _, row in df.iterrows():
        tid = str(row.get("tweet_id", ""))
        if tid:
            id_to_row[tid] = row.to_dict()

    # Build children map
    children: dict[str, list[str]] = defaultdict(list)
    for tid, row in id_to_row.items():
        parent = row.get("in_response_to_tweet_id")
        if isinstance(parent, str) and parent in id_to_row:
            children[parent].append(tid)

    # Find roots: tweets with no parent in the filtered set
    all_ids = set(id_to_row.keys())
    roots: list[str] = []
    for tid, row in id_to_row.items():
        parent = row.get("in_response_to_tweet_id")
        if not isinstance(parent, str) or parent not in all_ids:
            roots.append(tid)

    threads: list[dict] = []

    def dfs(tweet_id: str, messages: list[dict]):
        row = id_to_row.get(tweet_id)
        if not row:
            return
        author = str(row.get("author_id", ""))
        role = "spotify" if is_spotify_outbound(author) else "customer"
        raw_text = str(row.get("text", ""))
        messages.append({
            "tweet_id": tweet_id,
            "author_id": author,
            "role": role,
            "text": raw_text,
            "clean_text": clean_tweet(raw_text),
            "timestamp": str(row.get("created_at", "")),
            "in_response_to": row.get("in_response_to_tweet_id"),
        })
        for child_id in sorted(children.get(tweet_id, [])):
            dfs(child_id, messages)

    for root_id in tqdm(roots[:max_threads] if max_threads else roots,
                        desc="building threads"):
        messages: list[dict] = []
        dfs(root_id, messages)
        has_spotify = any(m["role"] == "spotify" for m in messages)
        if has_spotify and len(messages) >= 2:
            threads.append({
                "thread_id": root_id,
                "messages": messages,
                "n_turns": len(messages),
                "has_spotify_reply": has_spotify,
            })

    log.info("Reconstructed %d valid threads", len(threads))
    return threads


def save_threads(threads: list[dict], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    with open(out_path, "w") as f:
        for thread in threads:
            f.write(json.dumps(thread) + "\n")
    log.info("Saved %d threads to %s", len(threads), out_path)


def load_threads(path: Path) -> list[dict]:
    threads = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                threads.append(json.loads(line))
    return threads


def extract_qa_pairs(threads: list[dict]) -> list[dict]:
    """
    Extract (customer_message, spotify_reply) pairs for RAG indexing.
    Only uses the first Spotify reply in each thread to keep signals clean.
    """
    pairs = []
    for thread in threads:
        msgs = thread["messages"]
        customer_msgs: list[dict] = []
        for msg in msgs:
            if msg["role"] == "customer":
                customer_msgs.append(msg)
            elif msg["role"] == "spotify" and customer_msgs:
                # Take concatenated customer context + first spotify reply
                context = " ".join(m["clean_text"] for m in customer_msgs[-2:])
                pairs.append({
                    "thread_id": thread["thread_id"],
                    "customer_text": context,
                    "spotify_reply": msg["clean_text"],
                    "timestamp": msg["timestamp"],
                })
                break  # one pair per thread
    log.info("Extracted %d QA pairs for RAG", len(pairs))
    return pairs


def run_pipeline(cfg: dict | None = None) -> tuple[list[dict], list[dict]]:
    """
    Full pipeline entry point. Returns (threads, qa_pairs).
    """
    cfg = cfg or load_config()
    raw_path = Path(cfg["data"]["raw_path"])
    processed_path = Path(cfg["data"]["processed_path"])
    max_threads = cfg["data"].get("max_threads")

    if processed_path.exists():
        log.info("Processed threads found at %s — loading directly", processed_path)
        threads = load_threads(processed_path)
    else:
        if not raw_path.exists():
            download_dataset(cfg)
        df = load_raw(raw_path)
        spotify_df = filter_spotify(df)
        threads = reconstruct_threads(spotify_df, max_threads=max_threads)
        save_threads(threads, processed_path)

    qa_pairs = extract_qa_pairs(threads)
    return threads, qa_pairs


if __name__ == "__main__":
    threads, qa_pairs = run_pipeline()
    print(f"Threads: {len(threads)} | QA pairs: {len(qa_pairs)}")
    # Quick sanity print
    sample = threads[0]
    print(f"\nSample thread ({sample['thread_id']}, {sample['n_turns']} turns):")
    for m in sample["messages"][:4]:
        print(f"  [{m['role']}] {m['clean_text'][:120]}")
