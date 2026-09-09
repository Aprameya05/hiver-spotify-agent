"""
Shared utilities: config loading, logging, text cleaning, timing, LLM client.
"""

from __future__ import annotations

import os
import re
import time
import functools
import logging
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def clean_tweet(text: str) -> str:
    """Normalise a raw tweet string."""
    if not isinstance(text, str):
        return ""
    # Remove @mentions
    text = re.sub(r"@\w+", "", text)
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove hashtags
    text = re.sub(r"#\w+", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def timeit(fn):
    """Decorator that logs function wall-clock time."""
    logger = get_logger("timeit")

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        logger.debug("%s finished in %.2fs", fn.__qualname__, elapsed)
        return result

    return wrapper


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_llm_client():
    """
    Return a Groq client (free, no billing).
    Falls back to OpenAI if OPENAI_API_KEY is set and GROQ_API_KEY is not.
    """
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        from groq import Groq
        return Groq(api_key=groq_key), "groq"
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        from openai import OpenAI
        return OpenAI(api_key=openai_key), "openai"
    raise RuntimeError(
        "No LLM API key found. Set GROQ_API_KEY (free) or OPENAI_API_KEY."
    )


def llm_chat(messages: list[dict], model: str | None = None,
             temperature: float = 0.0, max_tokens: int = 512) -> str:
    """
    Send a chat completion request. Automatically picks Groq or OpenAI.
    model defaults to qwen/qwen3-32b on Groq, gpt-4o-mini on OpenAI.
    """
    client, provider = get_llm_client()
    if model is None:
        model = "qwen/qwen3-32b" if provider == "groq" else "gpt-4o-mini"
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()
