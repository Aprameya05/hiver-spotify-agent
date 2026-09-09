"""
LLM-as-judge for reply quality evaluation.

The judge scores each (customer_message, agent_reply) pair on 5 dimensions:
  1. Relevance    — does the reply address the customer's actual issue?
  2. Accuracy     — is the suggested fix correct / plausible?
  3. Tone         — empathetic, professional, matches Spotify's voice?
  4. Conciseness  — appropriate length for Twitter?
  5. Actionability — does it give the customer a clear next step?

Each dimension is 1-5. Overall = mean across dimensions.

Human-agreement estimation: we run the judge 3 times with temperature > 0
and check inter-run agreement (majority vote). This gives a proxy for how
consistent (and thus reliable) the judge is — analogous to inter-annotator
agreement. We also surface the judge's self-consistency score.

Calibration against human labels: the golden eval set contains a small
subset (40 examples) with human quality ratings. We compute Spearman rank
correlation between judge scores and human scores there.
"""

from __future__ import annotations

import json
import statistics
from typing import NamedTuple

from src.utils import get_logger, load_config

log = get_logger(__name__)

JUDGE_RUBRIC = """You are evaluating the quality of a Spotify customer-support reply on Twitter.

Score each dimension from 1 (very poor) to 5 (excellent):

1. **Relevance** — Does the reply directly address what the customer asked?
   1=Completely off-topic, 3=Partially relevant, 5=Squarely on point

2. **Accuracy** — Is the suggested action or information correct?
   1=Wrong/misleading, 3=Approximately right, 5=Fully accurate

3. **Tone** — Is the reply empathetic and professional?
   1=Cold/robotic/rude, 3=Neutral, 5=Warm and natural

4. **Conciseness** — Appropriate length for Twitter (target under 280 chars)?
   1=Way too long or too short, 3=Okay, 5=Just right

5. **Actionability** — Does the customer know what to do next?
   1=No clear next step, 3=Vague suggestion, 5=Clear, specific next step

Customer message: "{customer_message}"

Agent reply: "{agent_reply}"

Respond ONLY with valid JSON — no explanations outside the JSON:
{{
  "relevance": <1-5>,
  "accuracy": <1-5>,
  "tone": <1-5>,
  "conciseness": <1-5>,
  "actionability": <1-5>,
  "brief_rationale": "<one sentence>"
}}"""


class JudgeScore(NamedTuple):
    relevance: float
    accuracy: float
    tone: float
    conciseness: float
    actionability: float
    overall: float
    rationale: str
    n_runs: int
    consistency: float   # 0-1; how much the n_runs agreed


class LLMJudge:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.model = self.cfg["evaluation"]["judge_model"]
        self.temperature = self.cfg["evaluation"]["judge_temperature"]
        self.n_runs = self.cfg["evaluation"]["n_judge_runs"]

    def score(self, customer_message: str, agent_reply: str) -> JudgeScore:
        """
        Run the judge n_runs times and return majority-vote scores.
        Uses temperature=0.3 for runs 2+ to get variance estimate.
        """
        all_scores: list[dict] = []

        for run_i in range(self.n_runs):
            temp = 0.0 if run_i == 0 else 0.3
            raw = self._call_judge(customer_message, agent_reply, temperature=temp)
            if raw:
                all_scores.append(raw)

        if not all_scores:
            # Fallback neutral score
            return JudgeScore(3, 3, 3, 3, 3, 3.0, "Judge unavailable — neutral score assigned.", 0, 0.0)

        dims = ["relevance", "accuracy", "tone", "conciseness", "actionability"]
        # Mean across valid runs
        mean_scores = {
            dim: statistics.mean(s[dim] for s in all_scores if dim in s)
            for dim in dims
        }
        overall = statistics.mean(mean_scores.values())

        # Consistency: standard deviation of overall scores across runs
        if len(all_scores) > 1:
            run_overalls = [
                statistics.mean(s[d] for d in dims if d in s) for s in all_scores
            ]
            std = statistics.stdev(run_overalls)
            # Map std=0 -> consistency=1, std=2 -> consistency=0
            consistency = max(0.0, 1.0 - std / 2.0)
        else:
            consistency = 1.0

        # Rationale from first run
        rationale = all_scores[0].get("brief_rationale", "")

        return JudgeScore(
            relevance=mean_scores["relevance"],
            accuracy=mean_scores["accuracy"],
            tone=mean_scores["tone"],
            conciseness=mean_scores["conciseness"],
            actionability=mean_scores["actionability"],
            overall=round(overall, 3),
            rationale=rationale,
            n_runs=len(all_scores),
            consistency=round(consistency, 3),
        )

    def _call_judge(
        self, customer_message: str, agent_reply: str, temperature: float = 0.0
    ) -> dict | None:
        try:
            from openai import OpenAI
            client = OpenAI()
            prompt = JUDGE_RUBRIC.format(
                customer_message=customer_message,
                agent_reply=agent_reply,
            )
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=120,
            )
            raw = resp.choices[0].message.content.strip()
            obj = json.loads(raw)
            # Validate
            for dim in ["relevance", "accuracy", "tone", "conciseness", "actionability"]:
                if dim not in obj:
                    return None
                obj[dim] = max(1, min(5, float(obj[dim])))
            return obj
        except Exception as e:
            log.warning("Judge call failed: %s", e)
            return None

    def score_batch(
        self,
        pairs: list[tuple[str, str]],
        show_progress: bool = True,
    ) -> list[JudgeScore]:
        """Score a list of (customer_message, agent_reply) pairs."""
        from tqdm import tqdm
        results = []
        iterator = tqdm(pairs, desc="LLM judging") if show_progress else pairs
        for customer_msg, agent_reply in iterator:
            results.append(self.score(customer_msg, agent_reply))
        return results

    def compute_human_agreement(
        self,
        human_scores: list[float],
        judge_scores: list[float],
    ) -> dict[str, float]:
        """
        Compute Spearman rank correlation and mean absolute error between
        human and judge overall scores.
        """
        from scipy.stats import spearmanr
        if len(human_scores) < 2:
            return {"spearman_r": 0.0, "spearman_p": 1.0, "mae": 0.0}

        rho, pval = spearmanr(human_scores, judge_scores)
        mae = statistics.mean(
            abs(h - j) for h, j in zip(human_scores, judge_scores)
        )
        return {
            "spearman_r": round(float(rho), 4),
            "spearman_p": round(float(pval), 4),
            "mae": round(mae, 4),
            "n_pairs": len(human_scores),
        }


if __name__ == "__main__":
    judge = LLMJudge()
    test_pairs = [
        (
            "my spotify keeps buffering every few seconds on my iphone",
            "Hi! Really sorry about that. Try logging out and back in, and make sure the app is up to date. If it keeps happening, DM us and we'll dig in. ^SB"
        ),
        (
            "you charged me twice this month this is completely unacceptable",
            "Hi there! We understand how frustrating that is. Please DM us your account email so we can look into the charge and sort it out for you. ^SB"
        ),
    ]
    for customer, reply in test_pairs:
        score = judge.score(customer, reply)
        print(f"Customer: {customer}")
        print(f"Reply: {reply}")
        print(f"Scores: relevance={score.relevance:.1f} accuracy={score.accuracy:.1f} "
              f"tone={score.tone:.1f} conciseness={score.conciseness:.1f} "
              f"actionability={score.actionability:.1f}")
        print(f"Overall: {score.overall:.2f} | consistency: {score.consistency:.2f}")
        print(f"Rationale: {score.rationale}")
        print()
