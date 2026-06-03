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
        logging.FileHandler("agent_server.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("REDLINE-X")

# ─── Config ───────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

GROQ_MODELS = [
    os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
    "llama-3.1-8b-instant",
    "gemma2-9b-it"
]
TRACE_LOG_FILE = "agent_trace_log.jsonl"
PORT           = int(os.environ.get("PORT", os.environ.get("SERVER_PORT", "5050")))

# ─── Provider init ────────────────────────────────────────────────────────────
groq_client = None

if GROQ_API_KEY:
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info(f"[Provider] Groq ready: {GROQ_MODELS[0]} (primary)")
    except ImportError:
        logger.warning("[Provider] groq package not installed — run: pip install groq")
else:
    logger.info("[Provider] GROQ_API_KEY not set — Groq disabled.")

if not groq_client:
    logger.warning("[Provider] No LLM provider configured — heuristic fallback only.")

# ─── App state ────────────────────────────────────────────────────────────────
app                   = Flask(__name__)
trace_entries: list   = []
_pending_incidents: list = []   # queued from /agent/incident or /agent/incident_batch

# ─── Health ───────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    provider = f"groq/{GROQ_MODELS[0]}" if groq_client else "heuristic_fallback"
    return jsonify({
        "status": "ok",
        "primary_provider": provider,
        "groq_ready": groq_client is not None,
        "trace_entries": len(trace_entries),
        "pending_incidents": len(_pending_incidents),
        "timestamp": _now()
    })

# ─── Incident ingestion — single ─────────────────────────────────────────────
@app.route("/agent/incident", methods=["POST"])
def ingest_incident():
    try:
        body = request.get_json(force=True)
        _pending_incidents.append(body)
        logger.info(f"[Incident] Queued single: {body.get('type','unknown')} — {body.get('summary','')[:80]}")
        return jsonify({"queued": True, "queue_size": len(_pending_incidents)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ─── Incident ingestion — batch (lap-end push from RaceStewardAgent) ─────────
@app.route("/agent/incident_batch", methods=["POST"])
def ingest_incident_batch():
    """
    Receives the end-of-lap batch from RaceStewardAgent.PushLapIncidentsToServer().
    Each incident in the batch may carry a 'telemetryJson' field with 150m of data.
    We queue them all into _pending_incidents so they are picked up on the next
    steward /agent/decide poll.
    """
    try:
        body = request.get_json(force=True)
        incidents = body.get("incidents", [])
        for inc in incidents:
            _pending_incidents.append(inc)
        logger.info(f"[Steward] Batch received: {len(incidents)} incident(s) queued. "
                    f"Total pending: {len(_pending_incidents)}")

        # Immediately trigger a steward decision with the queued incidents
        # by returning a 202 Accepted so the game knows it landed
        return jsonify({"queued": len(incidents), "queue_size": len(_pending_incidents)}), 202
    except Exception as e:
        logger.error(f"[Steward] Batch ingest error: {e}")
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

    # Flush queued incidents into the Steward observation so the LLM sees them
    if "steward" in agent_id.lower() and _pending_incidents:
        obs_data = _safe_parse(obs_json)
        # Merge pending incidents into the pendingIncidents array
        existing = obs_data.get("pendingIncidents", [])
        if isinstance(existing, list):
            obs_data["pendingIncidents"] = existing + [json.dumps(i) for i in _pending_incidents]
        else:
            obs_data["pendingIncidents"] = [json.dumps(i) for i in _pending_incidents]
        obs_json = json.dumps(obs_data)
        flushed  = len(_pending_incidents)
        _pending_incidents.clear()
        logger.info(f"[Steward] Flushed {flushed} incident(s) into LLM prompt.")

    logger.info(f"[{agent_role}] Request received ({len(obs_json)} chars obs)")
    _log_trace({"type": "OBSERVATION", "agent_id": agent_id, "agent_role": agent_role,
                "timestamp": _now(), "observation": _safe_parse(obs_json)})

    decision  = None
    raw       = None
    provider  = None
    error_msg = None

    # ── 1. Try Groq (primary, with model fallback chain) ─────────────────────
    if groq_client and decision is None:
        for model in GROQ_MODELS:
            try:
                raw      = _call_groq(system_prompt, obs_json, model)
                decision = _parse_decision(raw, agent_id)
                provider = f"groq/{model}"
                logger.info(f"[{agent_role}] Groq ({model}) -> {decision.get('action')} | "
                            f"{decision.get('displayMessage','')[:80]}")
                break
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"[{agent_role}] Groq ({model}) failed: {str(e)[:80]}")

    # ── 2. Heuristic fallback ─────────────────────────────────────────────────
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

# ─── Trace endpoints ──────────────────────────────────────────────────────────
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
    response = groq_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"=== RACE STATE ===\n{observation_json}\n\nRespond with ONLY valid JSON. No markdown, no code blocks."}
        ],
        temperature=0.2,
        max_tokens=350,
    )
    return response.choices[0].message.content


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _parse_decision(raw: str, agent_id: str) -> dict | None:
    clean = raw.strip()
    # Strip markdown code fences if model ignores instructions
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
        logger.error(f"[Parser] JSON parse failed: {e} | raw[:100]={raw[:100]}")
        return None   # caller will try next provider or fall to heuristic


def _heuristic_fallback(agent_id: str, agent_role: str, obs_json: str) -> dict:
    obs    = _safe_parse(obs_json)
    msg    = ""
    action = "no_action"

    if agent_role == "Race Engineer":
        if obs.get("lastImpactAlert"):
            msg    = "Checking damage sensors."
            action = "no_action"
        elif obs.get("mandatoryPitDue"):
            msg    = "Box this lap. Mandatory stop."
            action = "recommend_pit"
        elif obs.get("tyreAvgWearPct", 0) > 85:
            msg    = "Tyres critical. Box box box."
            action = "recommend_pit"
        elif obs.get("safetyCarPitAdvised"):
            msg    = "Safety car. Free stop. Box now."
            action = "safety_car_pit"

    elif "steward" in agent_id.lower():
        incidents = obs.get("pendingIncidents", [])
        collisions = obs.get("collisionSummaries", [])
        if collisions:
            action = "warning"
            msg    = "Stewards noted the contact."
        elif obs.get("trackLimitViolationsPlayer", 0) >= 5:
            action = "time_penalty"
            msg    = "Stewards: +5s penalty — Player, track limits."

    return {
        "agentId":          agent_id,
        "action":           action,
        "parameter":        "5" if action == "time_penalty" else "",
        "targetCar":        "Player",
        "reasoning":        "[HEURISTIC] Server fallback — no LLM available.",
        "displayMessage":   msg,
        "confidence":       0.0,
        "requiresImmediate": action != "no_action"
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
    logger.info(f"  Groq:   {GROQ_MODELS[0] + ' (+fallbacks)' if groq_client else 'DISABLED'}")
    logger.info(f"  Port:   {PORT}")
    logger.info(f"  Trace:  {TRACE_LOG_FILE}")
    logger.info("=" * 60)
    app.run(host="0.0.0.0", port=PORT, debug=False)
