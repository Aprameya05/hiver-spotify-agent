"""
Spotify AI Support Agent -- Gradio web demo.

Works without a FAISS index or LLM API key:
  - Embedding classifier (all-mpnet-base-v2) for intent
  - Intent-matched template replies
  - Full multi-signal escalation engine

Set GROQ_API_KEY env var to enable LLM-generated replies.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Make src importable
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import gradio as gr

# ---------------------------------------------------------------------------
# Template replies used when no LLM / FAISS index is available
# ---------------------------------------------------------------------------
TEMPLATE_REPLIES: dict[str, str] = {
    "playback_issue": (
        "Hey! Sorry about the playback trouble. Try logging out and back in, "
        "then clear the cache under Settings > Storage. If it keeps happening, "
        "DM us your device model and app version and we'll dig in."
    ),
    "account_access": (
        "Hi! Sorry you're locked out. Use the password reset link at "
        "spotify.com/password-reset. If that doesn't work, DM us and we'll "
        "look into the account directly."
    ),
    "billing_payment": (
        "Hi! Sorry about the charge confusion. DM us your account email and "
        "we'll pull up the billing details and sort it out for you right away."
    ),
    "content_unavailable": (
        "Hey! Some content isn't available in all regions due to licensing. "
        "If you think this is an error, DM us the track or album name and "
        "we'll check on our end."
    ),
    "app_bug": (
        "Sorry the app is acting up! Try uninstalling and reinstalling the "
        "latest version. If the bug persists, DM us your device model and "
        "OS version and we'll escalate it to the team."
    ),
    "feature_request": (
        "Thanks for the suggestion! We pass all feedback to our product team. "
        "You can also vote on ideas and see what's coming at community.spotify.com."
    ),
    "general_inquiry": (
        "Hi! Happy to help. Could you share a bit more detail so we can point "
        "you in the right direction? Or DM us directly."
    ),
}

INTENT_LABELS: dict[str, str] = {
    "playback_issue":     "Playback Issue",
    "account_access":     "Account Access",
    "billing_payment":    "Billing / Payment",
    "content_unavailable":"Content Unavailable",
    "app_bug":            "App Bug",
    "feature_request":    "Feature Request",
    "general_inquiry":    "General Inquiry",
}

ESCALATION_COLOR = {True: "#c0392b", False: "#27ae60"}

# ---------------------------------------------------------------------------
# Load agent once at startup
# ---------------------------------------------------------------------------
print("Loading agent...")
_agent = None
_load_error = None

try:
    from src.agent import SpotifyAgent
    from src.utils import load_config
    cfg = load_config()
    _agent = SpotifyAgent(cfg)
    _agent.warm_up()
    print("Agent ready.")
except Exception as e:
    _load_error = str(e)
    print(f"Agent load failed: {e}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def analyze(message: str):
    if not message or not message.strip():
        return "", "", "", "", "", ""

    t0 = time.perf_counter()

    if _agent is not None:
        try:
            resp = _agent.handle(message)
            intent = resp.intent.intent
            confidence = resp.intent.confidence
            reply = resp.reply.reply or TEMPLATE_REPLIES.get(intent, TEMPLATE_REPLIES["general_inquiry"])
            should_escalate = resp.escalation.should_escalate
            esc_score = resp.escalation.score
            reasons = resp.escalation.reasons
            latency_ms = resp.latency_ms
        except Exception as e:
            return f"Error: {e}", "", "", "", "", ""
    else:
        # Minimal fallback: keyword-based intent when model failed to load
        text_lower = message.lower()
        if any(w in text_lower for w in ["buffer", "skip", "play", "song", "podcast", "stream"]):
            intent, confidence = "playback_issue", 0.72
        elif any(w in text_lower for w in ["log in", "login", "password", "account", "locked", "hack"]):
            intent, confidence = "account_access", 0.70
        elif any(w in text_lower for w in ["charge", "refund", "bill", "payment", "cancel", "subscription"]):
            intent, confidence = "billing_payment", 0.74
        elif any(w in text_lower for w in ["crash", "bug", "glitch", "freeze", "broken"]):
            intent, confidence = "app_bug", 0.68
        elif any(w in text_lower for w in ["add", "feature", "please", "would love", "option"]):
            intent, confidence = "feature_request", 0.65
        elif any(w in text_lower for w in ["missing", "removed", "not available", "region"]):
            intent, confidence = "content_unavailable", 0.66
        else:
            intent, confidence = "general_inquiry", 0.60

        reply = TEMPLATE_REPLIES[intent]
        # Simple escalation heuristic
        should_escalate = intent in ("billing_payment", "account_access") or confidence < 0.62
        esc_score = 0.72 if should_escalate else 0.18
        reasons = ["Intent requires account/billing access." if should_escalate else "Routine inquiry."]
        latency_ms = (time.perf_counter() - t0) * 1000

    intent_display = INTENT_LABELS.get(intent, intent)
    confidence_display = f"{confidence:.0%}"

    if should_escalate:
        esc_display = f"ESCALATE TO HUMAN  (score: {esc_score:.2f})"
    else:
        esc_display = f"AUTO-HANDLE  (score: {esc_score:.2f})"

    reasons_display = "\n".join(f"  {r}" for r in reasons)
    latency_display = f"{latency_ms:.0f} ms"

    return intent_display, confidence_display, reply, esc_display, reasons_display, latency_display


# ---------------------------------------------------------------------------
# Example messages
# ---------------------------------------------------------------------------
EXAMPLES = [
    ["spotify keeps buffering on my iphone, tried reinstalling twice"],
    ["you charged me even though i cancelled. i want a refund NOW"],
    ["can't log in and i think my account was hacked"],
    ["please add crossfade to the mobile app"],
    ["hi how do i get a student discount for premium"],
    ["the new drake album isn't showing up on spotify"],
    ["app crashes every single time i open it on android"],
]

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(
    title="Spotify Support Agent",
    theme=gr.themes.Base(
        primary_hue="green",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css="""
    .header { text-align: center; padding: 24px 0 8px; }
    .header h1 { font-size: 2rem; font-weight: 700; color: #1DB954; margin: 0; }
    .header p  { color: #888; margin: 6px 0 0; font-size: 0.95rem; }
    .badge { display: inline-block; padding: 2px 10px; border-radius: 12px;
             font-size: 0.8rem; font-weight: 600; }
    .result-box { font-size: 1rem; }
    """,
) as demo:

    gr.HTML("""
    <div class="header">
      <h1>Spotify Support Agent</h1>
      <p>AI-powered customer support triage &mdash; classify intent, draft a reply, decide escalation</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=5):
            message_input = gr.Textbox(
                label="Customer message",
                placeholder="e.g. you charged me twice this month, I want a refund",
                lines=3,
                max_lines=6,
            )
            analyze_btn = gr.Button("Analyze", variant="primary", size="lg")

        with gr.Column(scale=5):
            with gr.Row():
                intent_out = gr.Textbox(label="Detected intent", interactive=False, scale=3)
                confidence_out = gr.Textbox(label="Confidence", interactive=False, scale=1)
            reply_out = gr.Textbox(label="Draft reply", interactive=False, lines=3)
            escalation_out = gr.Textbox(label="Escalation decision", interactive=False)
            reasons_out = gr.Textbox(label="Signal breakdown", interactive=False, lines=2)
            latency_out = gr.Textbox(label="Latency", interactive=False)

    gr.Examples(
        examples=EXAMPLES,
        inputs=message_input,
        label="Try an example",
    )

    gr.HTML("""
    <div style="text-align:center; margin-top:24px; color:#888; font-size:0.8rem;">
      Built by <strong>Aprameya Bharadwaj</strong> &mdash; Hiver SDE Internship Take-Home 2027 &nbsp;|&nbsp;
      <a href="https://github.com/Aprameya05/hiver-spotify-agent" target="_blank" style="color:#1DB954;">GitHub</a>
    </div>
    """)

    analyze_btn.click(
        fn=analyze,
        inputs=message_input,
        outputs=[intent_out, confidence_out, reply_out, escalation_out, reasons_out, latency_out],
    )
    message_input.submit(
        fn=analyze,
        inputs=message_input,
        outputs=[intent_out, confidence_out, reply_out, escalation_out, reasons_out, latency_out],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
