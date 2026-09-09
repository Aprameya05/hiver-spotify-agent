"""
Master pipeline runner.

Usage:
    python scripts/run_pipeline.py                  # full pipeline
    python scripts/run_pipeline.py --build-index    # data + index only
    python scripts/run_pipeline.py --eval-only      # evaluation only (index must exist)
    python scripts/run_pipeline.py --max-threads 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_pipeline import run_pipeline, extract_qa_pairs
from src.intent_classifier import build_and_save_classifier
from src.reply_generator import ReplyGenerator
from src.agent import SpotifyAgent
from src.utils import get_logger, load_config, ensure_dir

log = get_logger("run_pipeline")


def main():
    parser = argparse.ArgumentParser(description="Hiver Spotify Agent pipeline runner")
    parser.add_argument("--build-index", action="store_true",
                        help="Build data pipeline and FAISS index, then stop")
    parser.add_argument("--eval-only", action="store_true",
                        help="Run evaluation only (requires index)")
    parser.add_argument("--max-threads", type=int, default=None,
                        help="Cap number of threads for development runs")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_threads:
        cfg["data"]["max_threads"] = args.max_threads

    # ------------------------------------------------------------------ #
    # Step 1: Data pipeline                                                #
    # ------------------------------------------------------------------ #
    if not args.eval_only:
        log.info("=== Step 1: Data Pipeline ===")
        threads, qa_pairs = run_pipeline(cfg)
        log.info("Threads: %d | QA pairs: %d", len(threads), len(qa_pairs))

        # ------------------------------------------------------------------ #
        # Step 2: Build / load FAISS index                                    #
        # ------------------------------------------------------------------ #
        log.info("=== Step 2: Building FAISS Index ===")
        gen = ReplyGenerator(cfg)
        gen.build_index(qa_pairs)

        if args.build_index:
            log.info("Index built. Stopping (--build-index flag set).")
            return

    # ------------------------------------------------------------------ #
    # Step 3: Train classifier                                             #
    # ------------------------------------------------------------------ #
    log.info("=== Step 3: Training Intent Classifier ===")
    golden_path = Path(cfg["data"]["golden_eval_path"])
    clf = build_and_save_classifier(golden_path if golden_path.exists() else None, cfg)

    # ------------------------------------------------------------------ #
    # Step 4: Warm up agent and run quick sanity checks                   #
    # ------------------------------------------------------------------ #
    log.info("=== Step 4: Sanity Checks ===")
    agent = SpotifyAgent(cfg)
    agent.warm_up(golden_path=golden_path)

    test_messages = [
        "spotify keeps buffering on my iphone, tried reinstalling twice",
        "you charged me even though i cancelled. i want a refund",
        "can't log in and i think my account was hacked",
        "please add offline mode to all devices",
        "hi how do i cancel my premium subscription",
    ]
    results_dir = ensure_dir(Path(cfg["paths"]["results_dir"]))
    sanity_results = []
    for msg in test_messages:
        resp = agent.handle(msg)
        sanity_results.append({
            "message": resp.original_message,
            "intent": resp.intent.intent,
            "confidence": resp.intent.confidence,
            "escalate": resp.escalation.should_escalate,
            "escalation_score": resp.escalation.score,
            "reply": resp.reply.reply,
            "latency_ms": resp.latency_ms,
        })
        print(agent.format_response(resp))
        print()

    with open(results_dir / "sanity_check.json", "w") as f:
        json.dump(sanity_results, f, indent=2)
    log.info("Sanity results saved to %s/sanity_check.json", results_dir)

    # ------------------------------------------------------------------ #
    # Step 5: Evaluation                                                   #
    # ------------------------------------------------------------------ #
    if golden_path.exists():
        log.info("=== Step 5: Running Evaluation ===")
        from eval.harness import run_evaluation
        summary = run_evaluation(agent, str(golden_path), cfg)
        print("\n=== EVALUATION SUMMARY ===")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        with open(results_dir / "eval_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        log.info("Eval summary saved.")
    else:
        log.warning(
            "Golden eval set not found at %s. Skipping evaluation.\n"
            "Run: python scripts/build_golden_eval.py", golden_path
        )


def run_eval():
    """Entry point for pyproject.toml script."""
    sys.argv.append("--eval-only")
    main()


if __name__ == "__main__":
    main()
