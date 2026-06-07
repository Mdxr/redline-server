"""
REDLINE-X — Agent Bridge Server (agent_server.py)  v3  National Finals Edition
─────────────────────────────────────────────────────────────────────────────────
LLM provider chain (all endpoints):
  1. Groq  (primary)   — llama-3.3-70b-versatile  (+fallback models)
  2. Gemini Flash       (secondary) — REST API, no SDK required
  3. Heuristic rules   (always-available offline fallback)

Endpoints:
  GET  /health
  POST /agent/decide          — Race Engineer + Race Steward decisions
  POST /agent/incident        — Single incident ingest
  POST /agent/incident_batch  — Batch ingest (end-of-lap push)
  POST /agent/appeal          — Post-race penalty appeal tribunal
  POST /agent/debrief         — Post-race Engineer debrief
  GET  /trace
  GET  /trace/latest/<n>
  GET  /trace/export
"""

import os, json, time, logging, sys, urllib.request, urllib.error
from datetime import datetime, timezone
from flask import Flask, request, jsonify

# ─── .env loader ──────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    print("[Server] .env loaded.")
except ImportError:
    print("[Server] python-dotenv not installed.")

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
STEWARD_GROQ_API_KEY  = os.environ.get("STEWARD_GROQ_API_KEY", os.environ.get("GROQ_API_KEY_STEWARD", ""))
ENGINEER_GROQ_API_KEY = os.environ.get("ENGINEER_GROQ_API_KEY", os.environ.get("GROQ_API_KEY_ENGINEER", ""))

GROQ_MODELS = [
    os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
    "llama-3.1-8b-instant"
]

SAMBANOVA_API_KEY = os.environ.get("SAMBANOVA_API_KEY", "")
SAMBANOVA_MODEL   = os.environ.get("SAMBANOVA_MODEL", "Meta-Llama-3.3-70B-Instruct")
SAMBANOVA_BASE    = "https://api.sambanova.ai/v1"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL   = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
OPENROUTER_BASE    = "https://openrouter.ai/api/v1"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
gemini_available = bool(GEMINI_API_KEY)

TRACE_LOG_FILE = "agent_trace_log.jsonl"
PORT           = int(os.environ.get("PORT", os.environ.get("SERVER_PORT", "5050")))

# ─── Provider init ────────────────────────────────────────────────────────────
groq_client_default  = None
groq_client_steward  = None
groq_client_engineer = None

try:
    from groq import Groq
    if GROQ_API_KEY:
        groq_client_default = Groq(api_key=GROQ_API_KEY, max_retries=0)
        logger.info(f"[Provider] Default Groq client ready: {GROQ_MODELS[0]}")
    if STEWARD_GROQ_API_KEY:
        groq_client_steward = Groq(api_key=STEWARD_GROQ_API_KEY, max_retries=0)
        logger.info("[Provider] Dedicated Steward Groq client ready")
    if ENGINEER_GROQ_API_KEY:
        groq_client_engineer = Groq(api_key=ENGINEER_GROQ_API_KEY, max_retries=0)
        logger.info("[Provider] Dedicated Engineer Groq client ready")
except ImportError:
    logger.warning("[Provider] groq package not installed — run: pip install groq")

groq_client = groq_client_default or groq_client_steward or groq_client_engineer
if not groq_client:
    logger.warning("[Provider] No Groq client initialized. Groq is disabled.")

# ─── App state ────────────────────────────────────────────────────────────────
app                   = Flask(__name__)
trace_entries: list   = []
_pending_incidents: list = []

# ─── Health ───────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    provider = (f"groq/{GROQ_MODELS[0]}" if groq_client else
                (f"gemini/{GEMINI_MODEL}"  if gemini_available else "heuristic_fallback"))
    return jsonify({
        "status": "ok",
        "primary_provider": provider,
        "groq_ready": groq_client is not None,
        "gemini_ready": gemini_available,
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
        logger.info(f"[Incident] Queued: {body.get('type','?')} — {body.get('summary','')[:80]}")
        return jsonify({"queued": True, "queue_size": len(_pending_incidents)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/agent/incident_batch", methods=["POST"])
def ingest_incident_batch():
    try:
        body      = request.get_json(force=True)
        incidents = body.get("incidents", [])
        for inc in incidents:
            _pending_incidents.append(inc)
        logger.info(f"[Steward] Batch: {len(incidents)} incidents queued. Total: {len(_pending_incidents)}")
        return jsonify({"queued": len(incidents), "queue_size": len(_pending_incidents)}), 202
    except Exception as e:
        logger.error(f"[Steward] Batch error: {e}")
        return jsonify({"error": str(e)}), 400

# Max incidents per LLM prompt — keeps token usage manageable
_MAX_INCIDENTS_PER_PROMPT = 5
# Essential keys we keep per incident — strip raw telemetry to save tokens
_INCIDENT_KEEP_KEYS = {
    "type", "summary", "severity", "aggressor", "victim",
    "impulseNs", "partHit", "corner", "lap", "article",
    "playerInvolved", "timestamp"
}

def _trim_incident(inc: dict) -> dict:
    """Return only the essential fields of an incident for the LLM prompt."""
    return {k: v for k, v in inc.items() if k in _INCIDENT_KEEP_KEYS}

def _downsample_telemetry(data):
    if not isinstance(data, dict):
        return data
    points = data.get("points")
    if isinstance(points, list) and len(points) > 30:
        step = max(1, len(points) // 30)
        downsampled = [points[i] for i in range(0, len(points), step)]
        if (len(points) - 1) % step != 0:
            downsampled.append(points[-1])
        new_data = dict(data)
        new_data["points"] = downsampled
        new_data["totalPoints"] = len(downsampled)
        return new_data
    return data

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

    # Flush queued incidents into Steward observations
    if "steward" in agent_id.lower() and _pending_incidents:
        obs_data = _safe_parse(obs_json)
        existing = obs_data.get("pendingIncidents", [])

        # ① Only keep the N most recent incidents to stay within token limits
        all_incidents = list(_pending_incidents)
        dropped = max(0, len(all_incidents) - _MAX_INCIDENTS_PER_PROMPT)
        recent_incidents = all_incidents[-_MAX_INCIDENTS_PER_PROMPT:]
        if dropped > 0:
            logger.info(f"[Steward] Dropped {dropped} older incident(s) — keeping {len(recent_incidents)} most recent.")

        # ② Strip raw telemetry and keep only essential metadata fields
        flushed_incidents = []
        for inc in recent_incidents:
            if isinstance(inc, dict):
                inc = _trim_incident(inc)  # drop telemetryJson and other heavy fields
            flushed_incidents.append(inc)

        obs_data["pendingIncidents"] = (existing if isinstance(existing, list) else []) + \
                                       [json.dumps(i) for i in flushed_incidents]
        obs_json = json.dumps(obs_data)
        flushed  = len(_pending_incidents)
        _pending_incidents.clear()
        logger.info(f"[Steward] Flushed {len(flushed_incidents)}/{flushed} incident(s) into LLM prompt ({len(obs_json)} chars).")

    logger.info(f"[{agent_role}] Request ({len(obs_json)} chars obs)")
    _log_trace({"type": "OBSERVATION", "agent_id": agent_id, "agent_role": agent_role,
                "timestamp": _now(), "observation": _safe_parse(obs_json)})

    decision, raw, provider, error_msg = None, None, None, None

    # 1. Groq primary (70b then 8b fallback)
    if groq_client and decision is None:
        for model in GROQ_MODELS:
            try:
                raw      = _call_groq(system_prompt, obs_json, model, agent_id)
                decision = _parse_decision(raw, agent_id)
                provider = f"groq/{model}"
                logger.info(f"[{agent_role}] Groq/{model} → {decision.get('action')} | {decision.get('displayMessage','')[:80]}")
                break
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"[{agent_role}] Groq/{model} failed: {str(e)[:80]}")

    # 2. SambaNova fallback
    if decision is None and SAMBANOVA_API_KEY:
        try:
            user_content = f"=== RACE STATE ===\n{obs_json}\n\nRespond with ONLY valid JSON. No markdown, no code blocks."
            raw      = _call_openai_compatible(system_prompt, user_content, SAMBANOVA_API_KEY, SAMBANOVA_BASE, SAMBANOVA_MODEL)
            decision = _parse_decision(raw, agent_id)
            provider = f"sambanova/{SAMBANOVA_MODEL}"
            logger.info(f"[{agent_role}] SambaNova → {decision.get('action')} | {decision.get('displayMessage','')[:80]}")
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"[{agent_role}] SambaNova failed: {str(e)[:80]}")

    # 3. OpenRouter fallback
    if decision is None and OPENROUTER_API_KEY:
        try:
            user_content = f"=== RACE STATE ===\n{obs_json}\n\nRespond with ONLY valid JSON. No markdown, no code blocks."
            raw      = _call_openai_compatible(system_prompt, user_content, OPENROUTER_API_KEY, OPENROUTER_BASE, OPENROUTER_MODEL)
            decision = _parse_decision(raw, agent_id)
            provider = f"openrouter/{OPENROUTER_MODEL}"
            logger.info(f"[{agent_role}] OpenRouter → {decision.get('action')} | {decision.get('displayMessage','')[:80]}")
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"[{agent_role}] OpenRouter failed: {str(e)[:80]}")

    # 4. Gemini fallback
    if decision is None and gemini_available:
        try:
            raw      = _call_gemini(system_prompt, obs_json)
            decision = _parse_decision(raw, agent_id)
            provider = f"gemini/{GEMINI_MODEL}"
            logger.info(f"[{agent_role}] Gemini → {decision.get('action')} | {decision.get('displayMessage','')[:80]}")
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"[{agent_role}] Gemini failed: {str(e)[:80]}")

    # 5. Heuristic fallback
    if decision is None:
        decision = _heuristic_fallback(agent_id, agent_role, obs_json)
        provider = "heuristic"
        logger.info(f"[{agent_role}] Heuristic → {decision.get('action')}")

    decision["_provider"] = provider
    latency_ms = round((time.time() * 1000) - start_ms, 1)
    _log_trace({"type": "DECISION", "agent_id": agent_id, "agent_role": agent_role,
                "timestamp": _now(), "latency_ms": latency_ms, "provider": provider,
                "raw_response": raw, "decision": decision, "error": error_msg})

    return jsonify(decision)

# ─── Appeal endpoint ──────────────────────────────────────────────────────────
@app.route("/agent/appeal", methods=["POST"])
def agent_appeal():
    """
    FIA Court of Appeal agent.
    Input:  { penaltyReason, penaltySec, stewardReasoning, telemetryJson,
              driverProfileSummary, isInnocentIncident }
    Output: { verdict: "overturned"|"upheld", reasoning, confidence }
    """
    start_ms = time.time() * 1000
    try:
        body = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    penalty_reason   = body.get("penaltyReason", "Unknown penalty")
    penalty_sec      = body.get("penaltySec", 0)
    steward_reason   = body.get("stewardReasoning", "")
    telemetry_json   = body.get("telemetryJson", "{}")
    driver_profile   = body.get("driverProfileSummary", "")
    is_innocent      = body.get("isInnocentIncident", False)

    # ── Pre-screen: test-suite innocent flag (strongest signal) ────────────────────
    # If the game engine explicitly tagged this incident as innocent (player was
    # the victim, not the aggressor), bypass the LLM and immediately overturn.
    # This is the test-suite demo path and also fires on real victim collisions.
    if is_innocent:
        logger.info("[Appeal] isInnocentIncident=true → immediate OVERTURN (victim scenario)")
        latency_ms = round((time.time() * 1000) - start_ms, 1)
        logger.info(f"[Appeal] Done in {latency_ms}ms via pre-screen: overturned")
        return jsonify({
            "verdict":    "overturned",
            "reasoning":  "Telemetry confirms player was the victim. Collision was initiated by another car. Penalty overturned.",
            "confidence": 0.97
        })

    # ── Parse collision metadata from stewardReasoning string ──────────────────
    # Format: "COLLISION_MAJOR | Aggressor:Player | Victim:CarXX | Part:Rear Wing | 12000 N·s"
    sr_lower = steward_reason.lower()
    player_is_aggressor_in_text = "aggressor:player" in sr_lower
    player_is_victim_in_text    = "victim:player"    in sr_lower
    aggressor_unknown           = "aggressor:unknown" in sr_lower

    # Extract impulse magnitude if present
    import re
    impulse_match = re.search(r'([\d,]+)\s*n[·*]?s', steward_reason, re.IGNORECASE)
    impulse_ns = 0
    if impulse_match:
        try:
            impulse_ns = float(impulse_match.group(1).replace(',', ''))
        except ValueError:
            impulse_ns = 0

    # Determine part hit for angle analysis
    part_hit = ""
    part_match = re.search(r'Part:([^|]+)', steward_reason, re.IGNORECASE)
    if part_match:
        part_hit = part_match.group(1).strip().lower()

    # Perfect rear-end: player drove into another car’s rear → always uphold
    # “Rear Wing” on the victim = player hit them from behind
    is_perfect_rear_end = (
        player_is_aggressor_in_text and
        ("rear" in part_hit or "rear wing" in part_hit) and
        impulse_ns >= 8000
    )

    if is_perfect_rear_end:
        logger.info(f"[Appeal] Perfect rear-end by Player ({impulse_ns:.0f} N·s) → immediate UPHOLD")
        latency_ms = round((time.time() * 1000) - start_ms, 1)
        logger.info(f"[Appeal] Done in {latency_ms}ms via pre-screen: upheld")
        return jsonify({
            "verdict":    "upheld",
            "reasoning":  f"Player initiated rear collision ({impulse_ns:.0f} N·s into victim’s Rear Wing). Clear aggressor. Penalty upheld.",
            "confidence": 0.95
        })

    # ── High-impulse pre-screen: any significant collision ≥5000 N·s ───────────
    # If the steward penalised a collision with substantial force and the player
    # is NOT explicitly flagged as innocent/victim, uphold immediately.
    # This catches cases where the aggressor label wasn't parsed from the reasoning
    # text but the steward clearly found a significant impact worth penalising.
    HIGH_IMPULSE_THRESHOLD = 5000  # N·s — user-configurable constant
    if impulse_ns >= HIGH_IMPULSE_THRESHOLD and not player_is_victim_in_text:
        logger.info(f"[Appeal] High-impulse collision ({impulse_ns:.0f} N·s >= {HIGH_IMPULSE_THRESHOLD}) — player not victim → immediate UPHOLD")
        latency_ms = round((time.time() * 1000) - start_ms, 1)
        logger.info(f"[Appeal] Done in {latency_ms}ms via pre-screen: upheld (high impulse)")
        return jsonify({
            "verdict":    "upheld",
            "reasoning":  f"Significant collision detected ({impulse_ns:.0f} N·s). Impact force exceeds racing-incident threshold. Penalty upheld.",
            "confidence": 0.90
        })

    # ── LLM review for low-impulse / ambiguous / side-impact / mutual-fault ────
    system_prompt = """You are the FIA Court of Appeal for REDLINE-X racing.
Review the steward’s penalty decision using collision evidence only.

CRITICAL RULES (non-negotiable, read every rule carefully):
1. THROTTLE DATA IS IRRELEVANT. Never use throttle or brake percentages to judge guilt.
   A single zero-throttle frame does not mean the driver avoided the collision.
2. The PRIMARY evidence is the stewardReasoning string which contains:
   - Aggressor (who caused the contact)
   - Victim (who received the contact)
   - Part hit (Rear Wing, Front Wing, Side, etc.)
   - Impulse magnitude in N·s
   - Collision severity (MAJOR / MINOR / CONTACT)
3. If the player is registered as the aggressor (e.g. "Player" is the aggressor): you must UPHOLD the penalty.
4. If the aggressor is "Unknown", or not registered, or if the player is the victim (another car is the aggressor): you must OVERTURN the penalty.
5. Side impacts ("Side", "Sidepod") at high speed are often mutual — consider context.
   If player entered a gap that closed, it may be a racing incident — consider OVERTURN harsh penalty.
6. Player damage (playerDamagePct) does NOT prove innocence; aggressors also get damaged.
7. If the steward’s own reasoning is contradictory or unclear: benefit of doubt → OVERTURN.
8. Driver profile "aggressive" lowers benefit of doubt.
9. Never confuse being penalised for something that happened to you with being the perpetrator.

Return ONLY valid JSON (no markdown):
{
  "verdict": "overturned" | "upheld",
  "reasoning": "<max 35 words citing collision metadata: aggressor, part hit, impulse, or ruling principle>",
  "confidence": <0.0-1.0>
}"""

    obs = {
        "penaltyReason":        penalty_reason,
        "penaltySec":           penalty_sec,
        "stewardReasoning":     steward_reason,
        "isInnocentIncident":   is_innocent,
        "collisionContext": {
            "playerIsAggressor": player_is_aggressor_in_text,
            "playerIsVictim":    player_is_victim_in_text,
            "aggressorUnknown":  aggressor_unknown,
            "impulseNs":         impulse_ns,
            "partHit":           part_hit,
        },
        "telemetrySummary":     _downsample_telemetry(_safe_parse(telemetry_json)),
        "driverProfile":        driver_profile
    }
    obs_json = json.dumps(obs)

    logger.info(f"[Appeal] Reviewing: {penalty_reason} +{penalty_sec}s (playerAggressor={player_is_aggressor_in_text}, impulse={impulse_ns:.0f}N·s)")

    decision, provider = None, None

    if groq_client:
        for model in GROQ_MODELS:
            try:
                raw      = _call_groq(system_prompt, obs_json, model, "steward_appeal")
                decision = _parse_appeal(raw)
                provider = f"groq/{model}"
                logger.info(f"[Appeal] Groq/{model} → {decision.get('verdict')} | {decision.get('reasoning','')[:60]}")
                break
            except Exception as e:
                logger.warning(f"[Appeal] Groq/{model} failed: {str(e)[:80]}")

    if decision is None and SAMBANOVA_API_KEY:
        try:
            user_content = f"=== RACE STATE ===\n{obs_json}\n\nRespond with ONLY valid JSON. No markdown, no code blocks."
            raw      = _call_openai_compatible(system_prompt, user_content, SAMBANOVA_API_KEY, SAMBANOVA_BASE, SAMBANOVA_MODEL)
            decision = _parse_appeal(raw)
            provider = f"sambanova/{SAMBANOVA_MODEL}"
            logger.info(f"[Appeal] SambaNova → {decision.get('verdict')} | {decision.get('reasoning','')[:60]}")
        except Exception as e:
            logger.warning(f"[Appeal] SambaNova failed: {str(e)[:80]}")

    if decision is None and OPENROUTER_API_KEY:
        try:
            user_content = f"=== RACE STATE ===\n{obs_json}\n\nRespond with ONLY valid JSON. No markdown, no code blocks."
            raw      = _call_openai_compatible(system_prompt, user_content, OPENROUTER_API_KEY, OPENROUTER_BASE, OPENROUTER_MODEL)
            decision = _parse_appeal(raw)
            provider = f"openrouter/{OPENROUTER_MODEL}"
            logger.info(f"[Appeal] OpenRouter → {decision.get('verdict')} | {decision.get('reasoning','')[:60]}")
        except Exception as e:
            logger.warning(f"[Appeal] OpenRouter failed: {str(e)[:80]}")

    if decision is None and gemini_available:
        try:
            raw      = _call_gemini(system_prompt, obs_json)
            decision = _parse_appeal(raw)
            provider = f"gemini/{GEMINI_MODEL}"
            logger.info(f"[Appeal] Gemini → {decision.get('verdict')}")
        except Exception as e:
            logger.warning(f"[Appeal] Gemini failed: {str(e)[:80]}")

    if decision is None:
        # Heuristic fallback: use collision metadata we already parsed
        if player_is_victim_in_text or aggressor_unknown or not steward_reason.strip():
            decision = {"verdict": "overturned", "reasoning": "No steward reasoning or evidence provided — benefit of doubt applied.", "confidence": 0.85}
        elif player_is_aggressor_in_text and impulse_ns >= 6000:
            decision = {"verdict": "upheld", "reasoning": "Player tagged as aggressor with significant impulse. Penalty upheld.", "confidence": 0.70}
        else:
            decision = {"verdict": "upheld", "reasoning": "Steward ruling upheld by default (no LLM available).", "confidence": 0.3}
        provider = "heuristic"

    latency_ms = round((time.time() * 1000) - start_ms, 1)
    logger.info(f"[Appeal] Done in {latency_ms}ms via {provider}: {decision.get('verdict')}")
    return jsonify(decision)


# ─── Debrief endpoint ─────────────────────────────────────────────────────────
@app.route("/agent/debrief", methods=["POST"])
def agent_debrief():
    """
    Post-race AI Engineer Debrief.
    Input:  { lapCount, bestLapTime, avgLapTime, trackLimitViolations,
              collisions, totalPenaltySec, finalPosition, totalCars,
              driverProfileSummary }
    Output: { debrief: "<formatted debrief text>" }
    """
    start_ms = time.time() * 1000
    try:
        body = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    lap_count      = body.get("lapCount", "?")
    best_lap       = body.get("bestLapTime", "N/A")
    avg_lap        = body.get("avgLapTime", "N/A")
    violations     = body.get("trackLimitViolations", "0")
    collisions     = body.get("collisions", "0")
    penalties      = body.get("totalPenaltySec", "0")
    final_pos      = body.get("finalPosition", "?")
    driver_profile = body.get("driverProfileSummary", "No profile data.")

    system_prompt = """You are the Race Engineer for REDLINE-X giving a post-race debrief to your driver.
Be direct, brief, technical. No filler. Sound like a real F1 radio debriefing.

Structure your debrief with EXACTLY these headers (use the exact labels):
LAP CONSISTENCY: <1-2 sentences on lap time delta and pace management>
TYRE MANAGEMENT: <1-2 sentences on wear rate and compound choices>
PENALTIES: <1-2 sentences on collisions, penalties, track limits>
BEST SECTOR: <1 sentence on their strongest sector and why it was good>
BIGGEST MISTAKE: <1 sentence on the biggest error>
NEXT-RACE ADVICE: <1-2 sentences adapting to the driver profile history>

Keep every section concise. Total response under 160 words.
Do NOT use markdown, bold, or bullet points — plain text only."""

    obs = {
        "lapsCompleted":        lap_count,
        "bestLapTime":          best_lap,
        "averageLapTime":       avg_lap,
        "consistencyDelta":     f"({avg_lap} avg vs {best_lap} best)",
        "trackLimitViolations": violations,
        "collisions":           collisions,
        "totalTimePenalty":     f"{penalties}s",
        "finalPosition":        final_pos,
        "driverProfile":        driver_profile
    }
    obs_json = json.dumps(obs)

    logger.info(f"[Debrief] Generating for {lap_count} laps, P{final_pos}")

    debrief_text, provider = None, None

    if groq_client:
        for model in GROQ_MODELS:
            try:
                debrief_text = _call_groq_freeform(system_prompt, obs_json, model, "engineer_debrief")
                provider     = f"groq/{model}"
                logger.info(f"[Debrief] Groq/{model} → {len(debrief_text)} chars")
                break
            except Exception as e:
                logger.warning(f"[Debrief] Groq/{model} failed: {str(e)[:80]}")

    if debrief_text is None and SAMBANOVA_API_KEY:
        try:
            user_content = f"=== RACE DATA ===\n{obs_json}\n\nWrite the debrief now."
            debrief_text = _call_openai_compatible(system_prompt, user_content, SAMBANOVA_API_KEY, SAMBANOVA_BASE, SAMBANOVA_MODEL, max_tokens=300, temperature=0.3)
            provider     = f"sambanova/{SAMBANOVA_MODEL}"
            logger.info(f"[Debrief] SambaNova → {len(debrief_text)} chars")
        except Exception as e:
            logger.warning(f"[Debrief] SambaNova failed: {str(e)[:80]}")

    if debrief_text is None and OPENROUTER_API_KEY:
        try:
            user_content = f"=== RACE DATA ===\n{obs_json}\n\nWrite the debrief now."
            debrief_text = _call_openai_compatible(system_prompt, user_content, OPENROUTER_API_KEY, OPENROUTER_BASE, OPENROUTER_MODEL, max_tokens=300, temperature=0.3)
            provider     = f"openrouter/{OPENROUTER_MODEL}"
            logger.info(f"[Debrief] OpenRouter → {len(debrief_text)} chars")
        except Exception as e:
            logger.warning(f"[Debrief] OpenRouter failed: {str(e)[:80]}")

    if debrief_text is None and gemini_available:
        try:
            debrief_text = _call_gemini_freeform(system_prompt, obs_json)
            provider     = f"gemini/{GEMINI_MODEL}"
            logger.info(f"[Debrief] Gemini → {len(debrief_text)} chars")
        except Exception as e:
            logger.warning(f"[Debrief] Gemini failed: {str(e)[:80]}")

    if debrief_text is None:
        debrief_text = (
            f"LAP CONSISTENCY: {lap_count} laps completed. Best lap {best_lap}, avg {avg_lap}.\n"
            f"TYRE MANAGEMENT: Tyre wear managed over the course of the session.\n"
            f"PENALTIES: {collisions} collision(s), {penalties}s in penalties.\n"
            f"BEST SECTOR: Strongest performance seen in Sector 2.\n"
            f"BIGGEST MISTAKE: {violations} track limit violations recorded — watch your lines.\n"
            f"NEXT-RACE ADVICE: Focus on consistency and cleaner racing."
        )
        provider = "heuristic"

    latency_ms = round((time.time() * 1000) - start_ms, 1)
    logger.info(f"[Debrief] Done in {latency_ms}ms via {provider}")
    return jsonify({"debrief": debrief_text, "_provider": provider})


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
def _get_groq_client(agent_id: str):
    agent_id_lower = (agent_id or "").lower()
    if "steward" in agent_id_lower:
        return groq_client_steward or groq_client_default
    elif "engineer" in agent_id_lower:
        return groq_client_engineer or groq_client_default
    return groq_client_default


def _call_groq(system_prompt: str, observation_json: str, model: str, agent_id: str = "") -> str:
    client = _get_groq_client(agent_id)
    if not client:
        raise Exception("Groq client not initialized")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"=== RACE STATE ===\n{observation_json}\n\nRespond with ONLY valid JSON. No markdown, no code blocks."}
        ],
        temperature=0.2,
        max_tokens=400,
    )
    return response.choices[0].message.content


def _call_groq_freeform(system_prompt: str, obs_json: str, model: str, agent_id: str = "") -> str:
    """Like _call_groq but for freeform text (debrief), not JSON."""
    client = _get_groq_client(agent_id)
    if not client:
        raise Exception("Groq client not initialized")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"=== RACE DATA ===\n{obs_json}\n\nWrite the debrief now."}
        ],
        temperature=0.3,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


def _call_openai_compatible(system_prompt: str, user_content: str, api_key: str, base_url: str, model: str, max_tokens: int = 400, temperature: float = 0.2) -> str:
    """Calls any OpenAI-compatible API (SambaNova, OpenRouter) via urllib."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "Mozilla/5.0"
    }
    if "openrouter" in base_url.lower():
        headers["HTTP-Referer"] = "http://localhost:5050"
        headers["X-Title"] = "REDLINE-X Agent"
        
    req_obj = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req_obj, timeout=20) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"]


def _call_gemini(system_prompt: str, observation_json: str) -> str:
    """Calls Gemini REST API — no SDK required, just urllib."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not configured")

    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}")

    payload = {
        "contents": [{
            "parts": [{"text":
                f"{system_prompt}\n\n=== RACE STATE ===\n{observation_json}"
                f"\n\nRespond with ONLY valid JSON. No markdown, no code blocks."
            }]
        }],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 400}
    }

    data     = json.dumps(payload).encode("utf-8")
    req_obj  = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req_obj, timeout=20) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        return result["candidates"][0]["content"]["parts"][0]["text"]


def _call_gemini_freeform(system_prompt: str, obs_json: str) -> str:
    """Gemini REST call for freeform text (debrief)."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not configured")

    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}")

    payload = {
        "contents": [{
            "parts": [{"text": f"{system_prompt}\n\n=== RACE DATA ===\n{obs_json}\n\nWrite the debrief now."}]
        }],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 350}
    }

    data    = json.dumps(payload).encode("utf-8")
    req_obj = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req_obj, timeout=20) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        return result["candidates"][0]["content"]["parts"][0]["text"].strip()


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _extract_json_block(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return text[start:]


def _parse_decision(raw: str, agent_id: str) -> dict | None:
    clean = raw.strip()
    if "```" in clean:
        for part in clean.split("```"):
            p = part.strip()
            if p.startswith("json"): p = p[4:].strip()
            if p.startswith("{"): clean = p; break
    clean = _extract_json_block(clean)
    try:
        data = json.loads(clean.strip())
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
        return None


def _parse_appeal(raw: str) -> dict | None:
    clean = raw.strip()
    if "```" in clean:
        for part in clean.split("```"):
            p = part.strip()
            if p.startswith("json"): p = p[4:].strip()
            if p.startswith("{"): clean = p; break
    clean = _extract_json_block(clean)
    try:
        data = json.loads(clean.strip())
        data.setdefault("verdict",    "upheld")
        data.setdefault("reasoning",  "")
        data.setdefault("confidence", 0.5)
        return data
    except Exception as e:
        logger.error(f"[Appeal parser] Failed: {e} | raw[:100]={raw[:100]}")
        return None


def _heuristic_fallback(agent_id: str, agent_role: str, obs_json: str) -> dict:
    obs    = _safe_parse(obs_json)
    msg    = ""
    action = "no_action"

    if agent_role == "Race Engineer":
        profile = obs.get("driverProfile", "")
        wear    = obs.get("tyreAvgWearPct", 0)
        life    = obs.get("estimatedTyreLifeLaps", 99)
        laps    = obs.get("lapsRemaining", 99)
        weather = obs.get("weather", "Dry")
        compound= obs.get("tyreCompound", "Medium")

        if obs.get("lastImpactAlert"):
            overall = obs.get("overallDamagePct", 0)
            fw      = obs.get("frontWingDmgPct", 0)
            rw      = obs.get("rearWingDmgPct", 0)
            if overall > 70:
                msg    = "Car is badly damaged. Come in for repairs."
                action = "damage_pit"
            elif fw > 60:
                msg    = "Front wing is damaged. Come in for a new nose."
                action = "damage_pit"
            elif rw > 60:
                msg    = "Rear wing took a hit. We need to check the damage."
                action = "damage_pit"
            else:
                msg = "Minor damage, stay out. We are watching it."
        elif obs.get("mandatoryPitDue"):
            msg    = "Box this lap. Mandatory pit stop, come in."
            action = "recommend_pit"
        elif obs.get("safetyCarPitAdvised"):
            msg    = "Safety car out. Free stop. Box box box."
            action = "safety_car_pit"
        elif wear > 90:
            msg    = "Tyres are gone. Box this lap."
            action = "recommend_pit"
        elif wear > 80 and life < laps:
            msg    = "Tyres are critical, we cannot make it. Come in."
            action = "recommend_pit"
        elif wear > 70:
            msg    = "Tyres are getting very worn. Keep an eye on them."

        # Weather compound mismatch
        if weather in ("Wet", "HeavyRain") and "Wet" not in compound:
            msg    = "It is raining. Box for wet tyres now."
            action = "recommend_pit"
        elif weather == "Damp" and "Intermediate" not in compound and "Wet" not in compound:
            msg    = "Track is damp. Consider intermediates next lap."

        # Driver profile nudges (TTS-friendly)
        if not msg:
            if "tyre_destroyer" in profile and wear > 65:
                msg = "You are eating these tyres. Lift and coast a bit."
            elif "weak_in_wet" in profile and weather in ("Wet", "HeavyRain", "Damp"):
                msg = "Rain out there, take care on the brakes."
            elif "corner_cutter" in profile:
                msg = "Watch the track limits, you are close to a penalty."

    elif "steward" in agent_id.lower():
        collisions = obs.get("collisionSummaries", [])
        pending    = obs.get("pendingIncidents", [])

        # Check pendingIncidents for player-aggressor collisions
        player_aggressor_major = False
        player_aggressor_minor = False
        for raw_inc in pending:
            try:
                inc = json.loads(raw_inc) if isinstance(raw_inc, str) else raw_inc
                aggressor  = str(inc.get("aggressor", "")).lower()
                severity   = str(inc.get("severity", "")).lower()
                involved   = inc.get("playerInvolved", False)
                if aggressor in ("player", "player_car") or (involved and aggressor not in ("", "unknown")):
                    if "major" in severity:
                        player_aggressor_major = True
                    else:
                        player_aggressor_minor = True
            except Exception:
                pass

        if player_aggressor_major:
            action = "time_penalty"
            msg    = "Stewards: 10 second time penalty for causing a major collision."
        elif collisions or player_aggressor_minor:
            action = "warning"
            msg    = "Stewards have noted the contact. Watch your driving."
        elif obs.get("trackLimitViolationsPlayer", 0) >= 5:
            action = "time_penalty"
            msg    = "Five second time penalty for repeated track limit violations."

    return {
        "agentId":           agent_id,
        "action":            action,
        "parameter":         "5" if action == "time_penalty" else "",
        "targetCar":         "Player",
        "reasoning":         "[HEURISTIC] Server fallback — no LLM available.",
        "displayMessage":    msg,
        "confidence":        0.0,
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
    logger.info("  REDLINE-X Agent Bridge Server  v3  (National Finals)")
    status_str = "ENABLED" if groq_client else "DISABLED"
    details = []
    if groq_client_default: details.append("default")
    if groq_client_steward: details.append("dedicated steward")
    if groq_client_engineer: details.append("dedicated engineer")
    if details:
        status_str += f" ({', '.join(details)})"
    logger.info(f"  Groq:       {status_str}")
    logger.info(f"  SambaNova:  {'ENABLED' if SAMBANOVA_API_KEY else 'DISABLED'}")
    logger.info(f"  OpenRouter: {'ENABLED' if OPENROUTER_API_KEY else 'DISABLED'}")
    logger.info(f"  Gemini:     {'ENABLED' if gemini_available else 'DISABLED'}")
    logger.info(f"  Port:       {PORT}")
    logger.info(f"  Trace:      {TRACE_LOG_FILE}")
    logger.info("  Endpoints: /agent/decide  /agent/appeal  /agent/debrief")
    logger.info("=" * 60)
    app.run(host="0.0.0.0", port=PORT, debug=False)
