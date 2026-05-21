"""
REDLINE-X — Agent Bridge Server (agent_server.py)
Multi-provider LLM chain:
  1. Groq  (primary)   — very high rate limits, ultra-fast inference
  2. Gemini Flash 2.0  (secondary fallback)
  3. Heuristic rules   (always-available offline fallback)
"""

import os, json, time, logging, sys
from datetime import datetime, timezone
from flask import Flask, request, jsonify

# ─── .env loader ──────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    print("[Server] .env file loaded.")
except ImportError:
    print("[Server] python-dotenv not installed — run: pip install python-dotenv")

# ─── UTF-8 stdout (Windows CP1252 safety) ─────────────────────────────────────
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("agent_server.log~", encoding="utf-8")
    ]
)
logger = logging.getLogger("REDLINE-X")

# ─── Config ───────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Groq model options (fallback order)
# - llama-3.3-70b-versatile  : 131k ctx, 6000 TPM, great reasoning
# - llama-3.1-8b-instant     : 131k ctx, 20000 TPM, ultra fast, lower quality
# - mixtral-8x7b-32768       : 32k ctx, 5000 TPM, good balance
GROQ_MODELS = [
    os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
TRACE_LOG_FILE = "agent_trace_log.jsonl~"
PORT           = int(os.environ.get("PORT", os.environ.get("SERVER_PORT", "5050")))

# ─── Provider init ────────────────────────────────────────────────────────────
groq_client   = None
gemini_client = None

if GROQ_API_KEY:
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info(f"[Provider] Groq ready: {GROQ_MODELS[0]} (primary)")
    except ImportError:
        logger.warning("[Provider] groq package not installed — run: pip install groq")
else:
    logger.info("[Provider] GROQ_API_KEY not set — Groq disabled.")

if GEMINI_API_KEY:
    try:
        from google import genai
        from google.genai import types as genai_types
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info(f"[Provider] Gemini ready: {GEMINI_MODEL} (secondary fallback)")
    except ImportError:
        logger.warning("[Provider] google-genai package not installed — run: pip install google-genai")
else:
    logger.info("[Provider] GEMINI_API_KEY not set — Gemini disabled.")

if not groq_client and not gemini_client:
    logger.warning("[Provider] No LLM providers configured — heuristic fallback only.")

# ─── App state ────────────────────────────────────────────────────────────────
app                   = Flask(__name__)
trace_entries: list   = []
_pending_incidents: list = []

# ─── Health ───────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    provider = "heuristic_fallback"
    if groq_client:   provider = f"groq/{GROQ_MODELS[0]}"
    elif gemini_client: provider = f"gemini/{GEMINI_MODEL}"
    return jsonify({
        "status": "ok",
        "primary_provider": provider,
        "groq_ready":   groq_client   is not None,
        "gemini_ready": gemini_client is not None,
        "trace_entries": len(trace_entries),
        "pending_incidents": len(_pending_incidents),
        "timestamp": _now()
    })

# ─── Incident ingestion ───────────────────────────────────────────────────────
@app.route("/agent/incident", methods=["POST"])
def ingest_incident():
    try:
        body = request.get_json(force=True)
        _pending_incidents.append(body)
        logger.info(f"[Incident] Queued: {body.get('type','unknown')} — {body.get('summary','')}")
        return jsonify({"queued": True, "queue_size": len(_pending_incidents)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ─── Main decision endpoint ───────────────────────────────────────────────────
@app.route("/agent/decide", methods=["POST"])
def agent_decide():
    start_ms = time.time() * 1000

    try:
        body = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    agent_id      = body.get("agentId", "unknown")
    agent_role    = body.get("agentRole", "Unknown")
    system_prompt = body.get("systemPrompt", "")
    obs_json      = body.get("observationJson", "{}")

    # Flush pending incidents into Steward observation
    if "steward" in agent_id and _pending_incidents:
        obs_data = _safe_parse(obs_json)
        obs_data["pendingIncidents"] = list(_pending_incidents)
        obs_json = json.dumps(obs_data)
        flushed  = len(_pending_incidents)
        _pending_incidents.clear()
        logger.info(f"[Steward] Flushed {flushed} queued incidents into prompt.")

    logger.info(f"[{agent_role}] Request received ({len(obs_json)} chars)")
    _log_trace({"type": "OBSERVATION", "agent_id": agent_id, "agent_role": agent_role,
                "timestamp": _now(), "observation": _safe_parse(obs_json)})

    decision   = None
    raw        = None
    provider   = None
    error_msg  = None

    # ── 1. Try Groq (primary) ─────────────────────────────────────────────────
    if groq_client and decision is None:
        for model in GROQ_MODELS:
            try:
                raw      = _call_groq(system_prompt, obs_json, model)
                decision = _parse_decision(raw, agent_id)
                provider = f"groq/{model}"
                logger.info(f"[{agent_role}] Groq ({model}) -> {decision.get('action')} | {decision.get('displayMessage','')[:80]}")
                break
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"[{agent_role}] Groq ({model}) failed ({str(e)[:50]}) — trying next...")
        
        if decision is None:
            logger.warning(f"[{agent_role}] All Groq models failed — trying Gemini...")

    # ── 2. Try Gemini (secondary fallback) ────────────────────────────────────
    if gemini_client and decision is None:
        try:
            raw      = _call_gemini(system_prompt, obs_json)
            decision = _parse_decision(raw, agent_id)
            provider = f"gemini/{GEMINI_MODEL}"
            logger.info(f"[{agent_role}] Gemini -> {decision.get('action')} | {decision.get('displayMessage','')[:80]}")
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"[{agent_role}] Gemini failed ({e}) — using heuristic fallback.")

    # ── 3. Heuristic fallback (always works) ──────────────────────────────────
    if decision is None:
        decision = _heuristic_fallback(agent_id, agent_role, obs_json)
        provider = "heuristic"
        logger.info(f"[{agent_role}] Heuristic -> {decision.get('action')}")

    decision["_provider"] = provider

    latency_ms = round((time.time() * 1000) - start_ms, 1)
    _log_trace({"type": "DECISION", "agent_id": agent_id, "agent_role": agent_role,
                "timestamp": _now(), "latency_ms": latency_ms, "provider": provider,
                "raw_response": raw, "decision": decision, "error": error_msg})

    return jsonify(decision)

# ─── Trace ────────────────────────────────────────────────────────────────────
@app.route("/trace", methods=["GET"])
def get_trace():
    return jsonify({"total_entries": len(trace_entries), "entries": trace_entries})

@app.route("/trace/latest/<int:n>", methods=["GET"])
def get_latest_trace(n):
    return jsonify(trace_entries[-n:])

@app.route("/trace/export", methods=["GET"])
def export_trace():
    return jsonify({"exported_at": _now(), "total_entries": len(trace_entries), "entries": trace_entries})

# ─── Provider calls ───────────────────────────────────────────────────────────
def _call_groq(system_prompt: str, observation_json: str, model: str) -> str:
    """Call Groq's OpenAI-compatible API."""
    response = groq_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"=== RACE STATE ===\n{observation_json}\n\nRespond with ONLY valid JSON. No markdown, no code blocks."}
        ],
        temperature=0.2,
        max_tokens=300,   # Short output = low token cost
    )
    return response.choices[0].message.content


def _call_gemini(system_prompt: str, observation_json: str) -> str:
    """Call Gemini via google-genai SDK."""
    prompt = f"{system_prompt}\n\n=== RACE STATE ===\n{observation_json}\n\nRespond with ONLY valid JSON."
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=300,
        )
    )
    return response.text

# ─── Helpers ──────────────────────────────────────────────────────────────────
def _parse_decision(raw: str, agent_id: str) -> dict | None:
    clean = raw.strip()
    # Strip markdown code fences
    if "```" in clean:
        parts = clean.split("```")
        for part in parts:
            p = part.strip()
            if p.startswith("json"): p = p[4:].strip()
            if p.startswith("{"): clean = p; break
    clean = clean.strip()
    try:
        data = json.loads(clean)
        data.setdefault("agentId",           agent_id)
        data.setdefault("action",            "no_action")
        data.setdefault("parameter",         "")
        data.setdefault("targetCar",         "Player")
        data.setdefault("reasoning",         "")
        data.setdefault("displayMessage",    "")
        data.setdefault("confidence",        0.8)
        data.setdefault("requiresImmediate", False)
        return data
    except Exception as e:
        logger.error(f"Both AI providers failed: {e}")
        # Return silent fallback so the game never stalls but doesn't fake AI messages
        fallback = {
            "agentId": agent_id,
            "action": "no_action",
            "parameter": "",
            "targetCar": "Player",
            "reasoning": f"[ERROR] AI offline: {str(e)[:30]}",
            "displayMessage": "",
            "confidence": 0.0,
            "requiresImmediate": False
        }
        _log_trace({"agentId": agent_id, "fallback": True, "error": str(e), "content": fallback})
        return fallback


def _heuristic_fallback(agent_id: str, agent_role: str, obs_json: str) -> dict:
    obs = _safe_parse(obs_json)
    msg = ""
    action = "no_action"
    
    if agent_role == "Race Engineer":
        if obs.get("lastImpactAlert"):
            msg = "Copy that. Sensors nominal."
            
    return {
        "agentId": agent_id,
        "action": action,
        "parameter": "",
        "targetCar": "Player",
        "reasoning": "[HEURISTIC] Server fallback",
        "displayMessage": msg,
        "confidence": 0.0,
        "requiresImmediate": False
    }


def _log_trace(entry: dict):
    trace_entries.append(entry)
    try:
        with open(TRACE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning(f"Trace write failed: {e}")


def _safe_parse(json_str: str) -> dict:
    try:    return json.loads(json_str)
    except: return {"raw": json_str}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─── Entry ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  REDLINE-X Agent Bridge Server")
    logger.info(f"  Groq:   {str(GROQ_MODELS[0]) + ' (+fallbacks)' if groq_client else 'DISABLED'}")
    logger.info(f"  Gemini: {GEMINI_MODEL if gemini_client else 'DISABLED'}")
    logger.info(f"  Port:   {PORT}")
    logger.info(f"  Trace:  {TRACE_LOG_FILE}")
    logger.info("=" * 60)
    app.run(host="0.0.0.0", port=PORT, debug=False)
