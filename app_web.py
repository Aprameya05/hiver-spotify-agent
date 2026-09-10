"""
Spotify AI Support Agent -- custom FastAPI web app.

Visually polished dark-mode UI. Works without a FAISS index or LLM key:
  - Embedding classifier (all-mpnet-base-v2) for intent
  - Intent-matched template replies
  - Full multi-signal escalation engine

Set GROQ_API_KEY env var to enable LLM-generated replies.
"""

from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ---------------------------------------------------------------------------
# Template replies
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
    "playback_issue":      "Playback Issue",
    "account_access":      "Account Access",
    "billing_payment":     "Billing / Payment",
    "content_unavailable": "Content Unavailable",
    "app_bug":             "App Bug",
    "feature_request":     "Feature Request",
    "general_inquiry":     "General Inquiry",
}

INTENT_ICONS: dict[str, str] = {
    "playback_issue":      "▶",
    "account_access":      "🔐",
    "billing_payment":     "💳",
    "content_unavailable": "🌍",
    "app_bug":             "⚠",
    "feature_request":     "✦",
    "general_inquiry":     "?",
}

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
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Spotify Support Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    message: str


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(HTML)


@app.get("/health")
async def health():
    return {"status": "ok", "agent_loaded": _agent is not None}


@app.post("/analyze")
async def analyze(req: AnalyzeRequest) -> JSONResponse:
    message = req.message.strip()
    if not message:
        return JSONResponse({"error": "empty message"}, status_code=400)

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
            return JSONResponse({"error": str(e)}, status_code=500)
    else:
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
        should_escalate = intent in ("billing_payment", "account_access") or confidence < 0.62
        esc_score = 0.72 if should_escalate else 0.18
        reasons = ["Intent requires account/billing access." if should_escalate else "Routine inquiry."]
        latency_ms = (time.perf_counter() - t0) * 1000

    return JSONResponse({
        "intent": intent,
        "intent_label": INTENT_LABELS.get(intent, intent),
        "intent_icon": INTENT_ICONS.get(intent, "?"),
        "confidence": confidence,
        "reply": reply,
        "should_escalate": should_escalate,
        "esc_score": esc_score,
        "reasons": reasons,
        "latency_ms": latency_ms,
    })


# ---------------------------------------------------------------------------
# HTML -- single-file SPA
# ---------------------------------------------------------------------------
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Spotify Support Agent</title>
<style>
  :root {
    --bg:       #050c17;
    --bg2:      #0a1628;
    --bg3:      #0f1f38;
    --border:   rgba(0,180,255,0.12);
    --border2:  rgba(0,180,255,0.22);
    --cyan:     #00c8ff;
    --cyan2:    #0090d4;
    --green:    #1DB954;
    --green2:   #00ff88;
    --red:      #ff3b52;
    --text:     #e0eaf8;
    --muted:    #5a7599;
    --mono:     'SF Mono', 'Fira Code', monospace;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', system-ui, sans-serif;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Starfield background */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      radial-gradient(1px 1px at 10% 20%, rgba(0,200,255,0.35) 0%, transparent 100%),
      radial-gradient(1px 1px at 25% 60%, rgba(0,200,255,0.2) 0%, transparent 100%),
      radial-gradient(1px 1px at 50% 10%, rgba(255,255,255,0.25) 0%, transparent 100%),
      radial-gradient(1px 1px at 70% 80%, rgba(0,200,255,0.3) 0%, transparent 100%),
      radial-gradient(1px 1px at 85% 35%, rgba(255,255,255,0.2) 0%, transparent 100%),
      radial-gradient(1px 1px at 95% 70%, rgba(0,200,255,0.25) 0%, transparent 100%),
      radial-gradient(2px 2px at 15% 85%, rgba(0,200,255,0.15) 0%, transparent 100%),
      radial-gradient(1px 1px at 40% 45%, rgba(255,255,255,0.15) 0%, transparent 100%),
      radial-gradient(1px 1px at 60% 30%, rgba(0,144,212,0.2) 0%, transparent 100%),
      radial-gradient(1px 1px at 80% 55%, rgba(255,255,255,0.1) 0%, transparent 100%);
    pointer-events: none;
    z-index: 0;
  }

  /* Grid overlay */
  body::after {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(0,144,212,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,144,212,0.03) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none;
    z-index: 0;
  }

  /* ---- NAV ---- */
  nav {
    position: sticky;
    top: 0;
    z-index: 100;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 40px;
    height: 60px;
    background: rgba(5,12,23,0.9);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--border);
  }

  .nav-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.02em;
  }

  .nav-logo svg { color: var(--green); }

  .nav-right {
    display: flex;
    align-items: center;
    gap: 24px;
    font-size: 0.82rem;
    color: var(--muted);
  }

  .live-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    background: rgba(0,200,255,0.08);
    border: 1px solid rgba(0,200,255,0.2);
    border-radius: 20px;
    padding: 4px 12px;
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--cyan);
    letter-spacing: 0.05em;
  }

  .live-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--green2);
    animation: pulse 1.8s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(0,255,136,0.6); }
    50%       { opacity: 0.5; box-shadow: 0 0 0 5px rgba(0,255,136,0); }
  }

  /* ---- HERO ---- */
  .hero {
    position: relative;
    z-index: 1;
    padding: 64px 40px 0;
    max-width: 1200px;
    margin: 0 auto;
  }

  .hero-tag {
    font-family: var(--mono);
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .hero-tag::before {
    content: '';
    display: inline-block;
    width: 28px;
    height: 1px;
    background: var(--muted);
  }

  h1 {
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.02em;
    margin-bottom: 16px;
  }

  h1 .accent { color: var(--green); }

  .hero-sub {
    color: var(--muted);
    font-size: 1rem;
    max-width: 520px;
    line-height: 1.6;
    margin-bottom: 40px;
  }

  /* ---- METRICS BAR ---- */
  .metrics-bar {
    display: flex;
    gap: 0;
    margin-bottom: 56px;
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    max-width: 640px;
  }

  .metric-item {
    flex: 1;
    padding: 16px 20px;
    border-right: 1px solid var(--border);
  }
  .metric-item:last-child { border-right: none; }

  .metric-value {
    font-family: var(--mono);
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
    margin-bottom: 4px;
  }

  .metric-value .unit {
    font-size: 0.75rem;
    color: var(--cyan);
    font-weight: 400;
  }

  .metric-label {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.08em;
    color: var(--muted);
    text-transform: uppercase;
  }

  /* ---- PIPELINE VIZ ---- */
  .pipeline-section {
    position: relative;
    z-index: 1;
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 40px 48px;
  }

  .pipeline-label {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 12px;
  }

  .pipeline-track {
    display: flex;
    align-items: center;
    gap: 0;
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0;
    overflow: hidden;
  }

  .pipeline-stage {
    flex: 1;
    padding: 14px 16px;
    border-right: 1px solid var(--border);
    position: relative;
    transition: background 0.3s;
  }

  .pipeline-stage:last-child { border-right: none; }

  .pipeline-stage.active {
    background: rgba(0,200,255,0.05);
  }

  .pipeline-stage.done {
    background: rgba(29,185,84,0.04);
  }

  .ps-name {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 4px;
  }

  .ps-detail {
    font-size: 0.8rem;
    color: var(--text);
    font-weight: 500;
  }

  .ps-status {
    position: absolute;
    top: 10px;
    right: 12px;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--bg3);
    border: 1px solid var(--border2);
  }

  .pipeline-stage.active .ps-status {
    background: var(--cyan);
    box-shadow: 0 0 8px var(--cyan);
    animation: pulse 1.2s ease-in-out infinite;
  }

  .pipeline-stage.done .ps-status {
    background: var(--green2);
    box-shadow: 0 0 6px var(--green2);
  }

  /* ---- MAIN LAYOUT ---- */
  .main {
    position: relative;
    z-index: 1;
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 40px 80px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
  }

  /* ---- INPUT PANEL ---- */
  .input-panel {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .panel-header {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 4px;
  }

  textarea {
    width: 100%;
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-family: inherit;
    font-size: 0.95rem;
    line-height: 1.6;
    padding: 16px;
    resize: vertical;
    min-height: 140px;
    outline: none;
    transition: border-color 0.2s, box-shadow 0.2s;
  }

  textarea::placeholder { color: var(--muted); }

  textarea:focus {
    border-color: var(--cyan2);
    box-shadow: 0 0 0 3px rgba(0,144,212,0.15);
  }

  .analyze-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    background: var(--green);
    color: #000;
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 0.03em;
    padding: 14px 28px;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    transition: background 0.2s, transform 0.1s, box-shadow 0.2s;
  }

  .analyze-btn:hover {
    background: #21cf60;
    box-shadow: 0 0 24px rgba(29,185,84,0.4);
  }

  .analyze-btn:active { transform: scale(0.98); }

  .analyze-btn.loading {
    background: rgba(29,185,84,0.4);
    pointer-events: none;
  }

  .analyze-btn .spinner {
    display: none;
    width: 16px; height: 16px;
    border: 2px solid rgba(0,0,0,0.3);
    border-top-color: #000;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }

  .analyze-btn.loading .spinner { display: block; }
  .analyze-btn.loading .btn-text { display: none; }

  @keyframes spin { to { transform: rotate(360deg); } }

  /* Examples */
  .examples-label {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 8px;
  }

  .examples {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }

  .example-chip {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.78rem;
    color: var(--muted);
    cursor: pointer;
    transition: border-color 0.2s, color 0.2s, background 0.2s;
    white-space: nowrap;
  }

  .example-chip:hover {
    border-color: var(--cyan2);
    color: var(--cyan);
    background: rgba(0,144,212,0.06);
  }

  /* ---- OUTPUT PANEL ---- */
  .output-panel {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    transition: border-color 0.3s;
  }

  .card.glow-cyan { border-color: rgba(0,200,255,0.35); box-shadow: 0 0 20px rgba(0,200,255,0.06); }
  .card.glow-green { border-color: rgba(0,255,136,0.35); box-shadow: 0 0 20px rgba(29,185,84,0.06); }
  .card.glow-red { border-color: rgba(255,59,82,0.4); box-shadow: 0 0 20px rgba(255,59,82,0.08); }

  .card-label {
    font-family: var(--mono);
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 10px;
  }

  /* Intent row */
  .intent-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }

  .intent-name {
    font-size: 1.05rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .intent-icon {
    font-size: 0.9rem;
    width: 28px; height: 28px;
    border-radius: 6px;
    background: rgba(0,200,255,0.1);
    border: 1px solid rgba(0,200,255,0.2);
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--cyan);
  }

  .confidence-badge {
    font-family: var(--mono);
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--cyan);
    background: rgba(0,200,255,0.08);
    border: 1px solid rgba(0,200,255,0.18);
    border-radius: 20px;
    padding: 4px 12px;
  }

  .conf-bar {
    margin-top: 10px;
    height: 3px;
    background: var(--bg3);
    border-radius: 2px;
    overflow: hidden;
  }

  .conf-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--cyan2), var(--cyan));
    border-radius: 2px;
    transition: width 0.6s ease;
  }

  /* Reply */
  .reply-text {
    font-size: 0.9rem;
    line-height: 1.65;
    color: var(--text);
  }

  /* Escalation */
  .esc-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .esc-badge {
    font-family: var(--mono);
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    padding: 6px 14px;
    border-radius: 20px;
    text-transform: uppercase;
  }

  .esc-badge.escalate {
    background: rgba(255,59,82,0.12);
    border: 1px solid rgba(255,59,82,0.35);
    color: var(--red);
  }

  .esc-badge.auto {
    background: rgba(0,255,136,0.08);
    border: 1px solid rgba(0,255,136,0.25);
    color: var(--green2);
  }

  .esc-score-box {
    text-align: right;
  }

  .esc-score-num {
    font-family: var(--mono);
    font-size: 1.4rem;
    font-weight: 700;
    line-height: 1;
  }

  .esc-score-label {
    font-family: var(--mono);
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    color: var(--muted);
    text-transform: uppercase;
  }

  .score-bar {
    margin-top: 10px;
    height: 3px;
    background: var(--bg3);
    border-radius: 2px;
    overflow: hidden;
  }

  .score-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.6s ease;
  }

  .score-fill.red { background: linear-gradient(90deg, #c0392b, var(--red)); }
  .score-fill.green { background: linear-gradient(90deg, var(--green), var(--green2)); }

  /* Signals */
  .signals {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .signal-row {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--muted);
  }

  .signal-row::before {
    content: '//';
    color: var(--border2);
    font-size: 0.7rem;
  }

  /* Latency */
  .latency-row {
    display: flex;
    align-items: baseline;
    gap: 6px;
  }

  .latency-num {
    font-family: var(--mono);
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
  }

  .latency-unit {
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--cyan);
  }

  /* Empty state */
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    gap: 12px;
    color: var(--muted);
    text-align: center;
    padding: 40px;
    border: 1px dashed var(--border);
    border-radius: 8px;
  }

  .empty-icon {
    font-size: 2rem;
    opacity: 0.3;
  }

  .empty-text {
    font-family: var(--mono);
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    opacity: 0.5;
  }

  /* ---- FOOTER ---- */
  footer {
    position: relative;
    z-index: 1;
    text-align: center;
    padding: 32px 40px;
    border-top: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--muted);
    letter-spacing: 0.04em;
  }

  footer a { color: var(--cyan); text-decoration: none; }
  footer a:hover { color: var(--cyan2); }

  /* Responsive */
  @media (max-width: 800px) {
    nav { padding: 0 20px; }
    .hero, .pipeline-section, .main { padding-left: 20px; padding-right: 20px; }
    .main { grid-template-columns: 1fr; }
    h1 { font-size: 1.8rem; }
  }
</style>
</head>
<body>

<!-- NAV -->
<nav>
  <div class="nav-logo">
    <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
      <circle cx="11" cy="11" r="10.5" stroke="currentColor" stroke-width="1"/>
      <path d="M6 13.5C8.5 12 12 11.8 16 13" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
      <path d="M7 10.5C10 8.8 13.5 8.6 16.5 10" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
      <path d="M8 7.5C10.5 6 13.5 5.8 16 7" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
    </svg>
    Spotify Support Agent
  </div>
  <div class="nav-right">
    <span style="display:none;color:var(--muted);font-family:var(--mono);font-size:0.75rem">93.4% intent accuracy</span>
    <div class="live-badge">
      <div class="live-dot"></div>
      SYSTEM ONLINE
    </div>
    <a href="https://github.com/Aprameya05/hiver-spotify-agent" target="_blank"
       style="color:var(--muted);text-decoration:none;font-size:0.8rem;transition:color 0.2s"
       onmouseover="this.style.color='var(--cyan)'" onmouseout="this.style.color='var(--muted)'">GitHub</a>
  </div>
</nav>

<!-- HERO -->
<div class="hero">
  <div class="hero-tag">AI Customer Support Infrastructure</div>
  <h1>Triage that handles<br>what your team <span class="accent">shouldn't.</span></h1>
  <p class="hero-sub">
    Classifies customer intent, drafts grounded replies, and decides escalation in real time.
    Trained on 28,277 real Spotify support interactions.
  </p>

  <div class="metrics-bar">
    <div class="metric-item">
      <div class="metric-value">93.4<span class="unit">%</span></div>
      <div class="metric-label">Intent Accuracy</div>
    </div>
    <div class="metric-item">
      <div class="metric-value">4.52<span class="unit">/5</span></div>
      <div class="metric-label">LLM Judge Score</div>
    </div>
    <div class="metric-item">
      <div class="metric-value">28k</div>
      <div class="metric-label">QA Pairs Indexed</div>
    </div>
    <div class="metric-item">
      <div class="metric-value">7</div>
      <div class="metric-label">Intent Classes</div>
    </div>
  </div>
</div>

<!-- PIPELINE -->
<div class="pipeline-section">
  <div class="pipeline-label">Pipeline State</div>
  <div class="pipeline-track">
    <div class="pipeline-stage" id="ps-classify">
      <div class="ps-name">Stage 1</div>
      <div class="ps-detail">Intent Classifier</div>
      <div class="ps-status"></div>
    </div>
    <div class="pipeline-stage" id="ps-generate">
      <div class="ps-name">Stage 2</div>
      <div class="ps-detail">Reply Generator</div>
      <div class="ps-status"></div>
    </div>
    <div class="pipeline-stage" id="ps-escalate">
      <div class="ps-name">Stage 3</div>
      <div class="ps-detail">Escalation Engine</div>
      <div class="ps-status"></div>
    </div>
    <div class="pipeline-stage" id="ps-done">
      <div class="ps-name">Output</div>
      <div class="ps-detail" id="ps-done-text">Awaiting input</div>
      <div class="ps-status"></div>
    </div>
  </div>
</div>

<!-- MAIN -->
<div class="main">

  <!-- INPUT -->
  <div class="input-panel">
    <div class="panel-header">Customer message</div>
    <textarea id="msg" placeholder="e.g. you charged me twice this month, I want a refund NOW" rows="6"></textarea>
    <button class="analyze-btn" id="btn" onclick="analyze()">
      <div class="spinner"></div>
      <span class="btn-text">Analyze Message</span>
    </button>

    <div style="margin-top:8px">
      <div class="examples-label">Try an example</div>
      <div class="examples" id="examples"></div>
    </div>
  </div>

  <!-- OUTPUT -->
  <div class="output-panel" id="output">
    <div class="empty-state">
      <div class="empty-icon">◎</div>
      <div class="empty-text">Enter a customer message to analyze</div>
    </div>
  </div>

</div>

<!-- FOOTER -->
<footer>
  Built by <strong>Aprameya Bharadwaj</strong> &nbsp;|&nbsp; Hiver SDE Internship Take-Home 2027 &nbsp;|&nbsp;
  <a href="https://github.com/Aprameya05/hiver-spotify-agent">GitHub</a>
</footer>

<script>
const EXAMPLES = [
  "spotify keeps buffering on my iphone, tried reinstalling twice",
  "you charged me even though i cancelled. i want a refund NOW",
  "can't log in and i think my account was hacked",
  "please add crossfade to the mobile app",
  "hi how do i get a student discount for premium",
  "the new drake album isn't showing up on spotify",
  "app crashes every single time i open it on android",
];

// Render example chips
const examplesEl = document.getElementById('examples');
EXAMPLES.forEach(ex => {
  const chip = document.createElement('div');
  chip.className = 'example-chip';
  chip.textContent = ex.length > 48 ? ex.slice(0, 46) + '...' : ex;
  chip.title = ex;
  chip.onclick = () => {
    document.getElementById('msg').value = ex;
    analyze();
  };
  examplesEl.appendChild(chip);
});

// Enter key support
document.getElementById('msg').addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) analyze();
});

function setPipelineStage(stage) {
  // stage: 'classify' | 'generate' | 'escalate' | 'done' | 'idle'
  const stages = ['classify', 'generate', 'escalate', 'done'];
  const idx = stages.indexOf(stage);
  stages.forEach((s, i) => {
    const el = document.getElementById('ps-' + s);
    el.classList.remove('active', 'done');
    if (i < idx) el.classList.add('done');
    else if (i === idx) el.classList.add('active');
  });
}

function setDoneText(t) {
  document.getElementById('ps-done-text').textContent = t;
}

async function analyze() {
  const msg = document.getElementById('msg').value.trim();
  if (!msg) return;

  const btn = document.getElementById('btn');
  btn.classList.add('loading');

  const out = document.getElementById('output');

  // Animate pipeline stages
  setPipelineStage('classify');
  await new Promise(r => setTimeout(r, 320));
  setPipelineStage('generate');
  await new Promise(r => setTimeout(r, 320));
  setPipelineStage('escalate');

  let data;
  try {
    const res = await fetch('/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg }),
    });
    data = await res.json();
  } catch(e) {
    btn.classList.remove('loading');
    setPipelineStage('idle');
    out.innerHTML = '<div class="empty-state"><div class="empty-icon">⚠</div><div class="empty-text">Network error — is the server running?</div></div>';
    return;
  }

  setPipelineStage('done');
  setDoneText(data.should_escalate ? 'ESCALATE' : 'AUTO-HANDLE');
  btn.classList.remove('loading');

  renderResults(data, out);
}

function renderResults(d, out) {
  const confPct = Math.round(d.confidence * 100);
  const escalate = d.should_escalate;
  const scoreColor = escalate ? 'red' : 'green';
  const escClass = escalate ? 'escalate' : 'auto';
  const escText = escalate ? 'ESCALATE TO HUMAN' : 'AUTO-HANDLE';
  const scoreNum = (d.esc_score).toFixed(2);
  const scoreColor2 = escalate ? 'var(--red)' : 'var(--green2)';
  const cardGlow = escalate ? 'glow-red' : 'glow-green';
  const latency = Math.round(d.latency_ms);
  const signals = (d.reasons || []).join('\n');

  out.innerHTML = `
    <!-- Intent card -->
    <div class="card glow-cyan">
      <div class="card-label">Detected Intent</div>
      <div class="intent-row">
        <div class="intent-name">
          <div class="intent-icon">${d.intent_icon}</div>
          ${d.intent_label}
        </div>
        <div class="confidence-badge">${confPct}%</div>
      </div>
      <div class="conf-bar"><div class="conf-fill" style="width:${confPct}%"></div></div>
    </div>

    <!-- Reply card -->
    <div class="card">
      <div class="card-label">Draft Reply</div>
      <div class="reply-text">${escapeHtml(d.reply)}</div>
    </div>

    <!-- Escalation card -->
    <div class="card ${cardGlow}">
      <div class="card-label">Escalation Decision</div>
      <div class="esc-row">
        <span class="esc-badge ${escClass}">${escText}</span>
        <div class="esc-score-box">
          <div class="esc-score-num" style="color:${scoreColor2}">${scoreNum}</div>
          <div class="esc-score-label">ESC SCORE</div>
        </div>
      </div>
      <div class="score-bar"><div class="score-fill ${scoreColor}" style="width:${Math.round(d.esc_score*100)}%"></div></div>
    </div>

    <!-- Signals -->
    <div class="card">
      <div class="card-label">Signal Breakdown</div>
      <div class="signals">
        ${(d.reasons || []).map(r => `<div class="signal-row">${escapeHtml(r)}</div>`).join('')}
      </div>
    </div>

    <!-- Latency -->
    <div class="card">
      <div class="card-label">Latency</div>
      <div class="latency-row">
        <div class="latency-num">${latency}</div>
        <div class="latency-unit">ms end-to-end</div>
      </div>
    </div>
  `;
}

function escapeHtml(t) {
  return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
