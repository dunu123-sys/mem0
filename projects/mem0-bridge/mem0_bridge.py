"""
Mem0 MCP bridge for OpenClaw (MCP SDK v2).

Exposes the self-hosted Mem0 REST API (add + search) as MCP stdio tools
so OpenClaw agents can use Mem0's extraction + hybrid retrieval.

Mem0 server URL is configurable via MEM0_URL (default http://localhost:8888).
"""
import os
import json
import time
import sqlite3
import urllib.request
import urllib.error
from datetime import datetime, timezone

try:
    from mcp.server import Server as MCPServer
except ImportError:
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError:
        MCPServer = None

MEM0_URL = os.environ.get("MEM0_URL", "http://localhost:8888").rstrip("/")

# Ollama (bridge runs on the host; Ollama is NOT containerized)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

# Classifier (OpenAI-compatible local endpoint; same model/route the mem0 plugin
# uses for fact extraction, reused here for single-label classification).
CLASSIFIER_BASE_URL = os.environ.get("CLASSIFIER_BASE_URL", "http://127.0.0.1:20128/v1").rstrip("/")
CLASSIFIER_MODEL = os.environ.get("CLASSIFIER_MODEL", "Combo_Extractor")
CLASSIFIER_FALLBACK = "misc"

# Max rows the Mem0 backend will return from GET /memories (its ALL_MEMORIES_LIMIT).
# The bridge previously hardcoded top_k=1000, which silently truncated the category
# counts and the categorizer to the first 1000 rows while the table held more.
MEMORY_FETCH_LIMIT = 5000

_OPENCLAW_CONFIG = os.path.join(os.path.expanduser("~"), ".openclaw", "openclaw.json")
_BRIDGE_ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

def _read_bridge_env_key() -> str:
    """Read CLASSIFIER_API_KEY from the bridge-local .env (KEY=VALUE lines). Never raises."""
    try:
        with open(_BRIDGE_ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                if k.strip() == "CLASSIFIER_API_KEY" and v.strip():
                    return v.strip()
    except Exception:
        pass
    return None

def _classifier_api_key() -> str:
    """Resolve the 9router apiKey for the classifier. Never raises; returns None if absent.

    Priority: CLASSIFIER_API_KEY env > OPENAI_API_KEY-compatible env > bridge-local
    .env file > 9router provider apiKey read from openclaw.json. The apiKey is
    never logged/printed; it is used only to build the Authorization header.
    """
    for var in ("CLASSIFIER_API_KEY", "OPENAI_API_KEY", "ROUTER_API_KEY"):
        v = os.environ.get(var)
        if v:
            return v
    from_env_file = _read_bridge_env_key()
    if from_env_file:
        return from_env_file
    try:
        with open(_OPENCLAW_CONFIG, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        key = (((cfg.get("models") or {}).get("providers") or {}).get("9router") or {}).get("apiKey")
        return key or None
    except Exception:
        return None

# Local state DB for bridge-side capabilities (decay access counts, dedup log, flags).
# Lives next to this file; zero changes to the Mem0 server, no new infra.
_BRIDGE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_DB = os.path.join(_BRIDGE_DIR, "mem0_state.db")
CATEGORIES_FILE = os.path.join(_BRIDGE_DIR, "categories.json")

# Cached category registry (loaded lazily from categories.json).
_CATEGORY_REGISTRY = None

def _load_categories() -> dict:
    """Load the taxonomy (name -> description) from categories.json. Never raises."""
    global _CATEGORY_REGISTRY
    if _CATEGORY_REGISTRY is not None:
        return _CATEGORY_REGISTRY
    try:
        with open(CATEGORIES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        cats = data.get("categories", [])
        registry = {}
        for c in cats:
            name = str(c.get("name", "")).strip().lower()
            if name:
                registry[name] = str(c.get("description", "")).strip()
        if CLASSIFIER_FALLBACK not in registry:
            registry[CLASSIFIER_FALLBACK] = "Anything that does not clearly fit the other categories."
        _CATEGORY_REGISTRY = registry
    except Exception:
        _CATEGORY_REGISTRY = {CLASSIFIER_FALLBACK: "Anything that does not clearly fit the other categories."}
    return _CATEGORY_REGISTRY

def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(STATE_DB)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT, updated_at REAL)"
    )
    # Webhook delivery history (dashboard consumes via the bridge-status sidecar).
    conn.execute(
        "CREATE TABLE IF NOT EXISTS webhook_events ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  event TEXT,"
        "  url TEXT,"
        "  at REAL,"
        "  ok INTEGER DEFAULT 0,"
        "  status INTEGER,"
        "  error TEXT"
        ")"
    )
    # Soft-merge log: one row per merge (loser -> canonical).
    conn.execute(
        "CREATE TABLE IF NOT EXISTS dream_log ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  memory_id TEXT,"
        "  canonical_id TEXT,"
        "  text TEXT,"
        "  merged_at REAL"
        ")"
    )
    # Additive migration: intensity_delta column (T4.3). Older DBs lack it; add
    # it idempotently so existing dream_log tables gain the column without a rebuild.
    try:
        conn.execute(
            "ALTER TABLE dream_log ADD COLUMN intensity_delta INTEGER DEFAULT 0"
        )
    except sqlite3.OperationalError:
        # Column already exists (or DB is locked); non-fatal.
        pass
    return conn

def _log_webhook_event(event: str, url: str, ok: bool, status: int = None, error: str = None):
    """Persist one webhook delivery attempt. Never raises; logging must not break delivery."""
    try:
        conn = _db()
        try:
            conn.execute(
                "INSERT INTO webhook_events (event, url, at, ok, status, error)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (event, url, time.time(), 1 if ok else 0, status, error),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass

def get_state(key: str, default: str = None):
    conn = _db()
    try:
        row = conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default
    finally:
        conn.close()

def set_state(key: str, value: str):
    conn = _db()
    try:
        conn.execute(
            "INSERT INTO kv (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            (key, value, time.time()),
        )
        conn.commit()
    finally:
        conn.close()

mcp = None  # MCP server stub (0.9.1 compat: not used by bridge_status_server)

def _request(path: str, method: str = "POST", payload: dict = None, timeout: int = 120) -> dict:
    """Generic HTTP request to the Mem0 server. Returns parsed JSON (dict) or an error dict."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        f"{MEM0_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode()
        return {"error": detail, "status": e.code}
    except Exception as e:
        return {"error": str(e)}

def _post(path: str, payload: dict, timeout: int = 120) -> dict:
    return _request(path, method="POST", payload=payload, timeout=timeout)

def _get(path: str, timeout: int = 120) -> dict:
    return _request(path, method="GET", timeout=timeout)

# ---- Categorizer (single-label classification via local LLM) ----

def _classify_category(text: str) -> str:
    """Classify a text into exactly ONE category from categories.json.

    Uses the OpenAI-compatible classifier endpoint (Combo_Extraction). Returns
    the normalized category name, or CLASSIFIER_FALLBACK ("misc") on any error.
    Never raises.
    """
    try:
        if not text or not str(text).strip():
            return CLASSIFIER_FALLBACK
        registry = _load_categories()
        if not registry or (len(registry) == 1 and CLASSIFIER_FALLBACK in registry):
            # Registry failed to load beyond the fallback; no point calling the LLM.
            return CLASSIFIER_FALLBACK
        system = (
            "You are a strict single-label classifier. Classify the given memory into exactly ONE "
            "of the following categories. Reply with ONLY the category name, nothing else.\n\n"
            "Categories:\n"
            + "\n".join(f"- {name}: {desc}" for name, desc in registry.items())
            + "\n\nDisambiguation rules:\n"
            "- `contacts` (not `technical`): any fact whose POINT is a person's identity, role, "
            "email, phone, handle, or who they are. Tooling/config mentioned only as context "
            "about a person still belongs to `contacts` (e.g. 'Tailscale machine X is under "
            "beau@rizedigital.io').\n"
            "- `project_status` (not `technical`): facts about WORKFLOW STATE, progress, decisions, "
            "milestones, TODOs, or 'still unbuilt / not yet done / to do' — even when they mention "
            "code or tooling. `technical` is for static technical KNOWLEDGE (how a thing works, "
            "config facts), not for what is unfinished.\n"
            "- `personal`: stable human preferences, tastes, habits, and personal details.\n"
            "- If the text is a long heterogeneous list, a mix of unrelated items, or otherwise "
            "cannot be cleanly captured by ONE category, choose `misc` rather than forcing `technical`.\n"
            "- Do not default to `technical` merely because the text mentions code/tools/config; "
            "choose the category that best matches the POINT of the memory."
        )
        body = {
            "model": CLASSIFIER_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": str(text).strip()},
            ],
        }
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        key = _classifier_api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(
            f"{CLASSIFIER_BASE_URL}/chat/completions",
            data=data,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.loads(r.read().decode())
        # OpenAI-compatible response shape: choices[0].message.content
        reply = ""
        choices = out.get("choices", [])
        if choices:
            msg = choices[0].get("message") or {}
            reply = msg.get("content", "") or ""
        reply = str(reply).strip().lower()
        # Normalize: strip surrounding quotes/punctuation, take first token.
        reply = reply.strip('"\'`.,;:!?')
        reply = reply.split()[0] if reply else ""
        if reply in registry:
            return reply
        return CLASSIFIER_FALLBACK
    except Exception:
        return CLASSIFIER_FALLBACK

# ---- Webhooks (runtime-configurable; env WEBHOOK_URLS is the fallback default) ----

# State-DB key where a runtime override (JSON array of URL strings) is stored.
# When present, it wins over the env-var default. Lets agents/users set webhook
# targets at runtime without touching env or restarting the gateway.
WEBHOOK_URLS_KEY = "webhooks:urls"

def _env_webhook_urls():
    """Fallback default: comma-separated WEBHOOK_URLS env var."""
    return [u.strip() for u in os.environ.get("WEBHOOK_URLS", "").split(",") if u.strip()]

def _effective_webhook_urls():
    """Live webhook URL list. Runtime override persisted to state DB wins;
    otherwise the env-var default is used. Read at emit time, never cached."""
    raw = get_state(WEBHOOK_URLS_KEY, None)
    if raw is not None:
        try:
            val = json.loads(raw)
            if isinstance(val, list):
                return [u.strip() for u in val if isinstance(u, str) and u.strip()]
        except Exception:
            pass
    return _env_webhook_urls()

def _emit_webhook(event: str, payload: dict):
    """POST a webhook event to each effective URL. Fire-and-forget, never raises.

    Returns a list of per-URL results. Reads the effective URL list live at emit
    time (runtime override wins over env).
    """
    urls = _effective_webhook_urls()
    if not urls:
        # No URLs configured => no deliveries attempted => no rows logged.
        # The dashboard should expect last_events == [] while webhooks are disabled.
        return []
    body = json.dumps({"event": event, "ts": time.time(), **payload}).encode("utf-8")
    results = []
    for url in urls:
        for attempt in (1, 2, 3):
            try:
                req = urllib.request.Request(
                    url, data=body, headers={"Content-Type": "application/json"}, method="POST"
                )
                with urllib.request.urlopen(req, timeout=5) as r:
                    results.append({"url": url, "status": r.status})
                    _log_webhook_event(event, url, True, status=r.status)
                break
            except Exception as e:
                if attempt == 3:
                    results.append({"url": url, "error": str(e)})
                    _log_webhook_event(event, url, False, error=str(e))
                else:
                    time.sleep(0.5 * attempt)
    return results

def mem0_emit_test(event: str = "test", detail: str = "bridge webhook test") -> str:
    """Send a test webhook event to confirm the effective target(s) receive events."""
    urls = _effective_webhook_urls()
    if not urls:
        return json.dumps({"ok": False, "note": "no webhook URLs configured; set them with mem0_set_webhooks"}, indent=2)
    results = _emit_webhook(event, {"detail": detail})
    return json.dumps({"ok": True, "urls": urls, "results": results}, indent=2)

def mem0_set_webhooks(urls: list) -> str:
    """Set the runtime webhook URL list (persisted to state DB; overrides env).

    Args:
        urls: List of webhook URLs to POST events to. Pass an empty list to
            clear the override and fall back to the WEBHOOK_URLS env var.
    """
    if not isinstance(urls, list):
        return json.dumps({"ok": False, "error": "urls must be a list of strings"}, indent=2)
    cleaned = [u.strip() for u in urls if isinstance(u, str) and u.strip()]
    set_state(WEBHOOK_URLS_KEY, json.dumps(cleaned))
    return json.dumps({"ok": True, "urls": cleaned, "note": "runtime override set; empty list falls back to env"}, indent=2)

def mem0_get_webhooks() -> str:
    """Read the effective webhook URL list (runtime override if set, else env)."""
    return json.dumps({"ok": True, "urls": _effective_webhook_urls(), "override": get_state(WEBHOOK_URLS_KEY, None) is not None}, indent=2)

# ---- Memory Decay (recency re-ranking) ----

DECAY_KEY = "decay_enabled"
DECAY_ACCESS_CAP = 20
DECAY_FRESH = 1.5      # boost for recently accessed
DECAY_FLOOR = 0.3      # lowest scaling factor (idle)
DECAY_HALFLIFE = 7 * 86400  # ~1 week: idleness half-life for the decay curve

# ---- Intensity / reinforcement layer (Merge Intensity) ----
# These compose WITH the decay engine above; they do not replace it.
# intensity = reinforcement counter (persisted). tier = promotion state.
# Resistance: intensity raises the per-memory decay floor up to FLOOR_CAP.
# Promotion: intensity crossing THRESHOLD_CONSOLIDATED / THRESHOLD_CORE
#   graduates memory episodic -> consolidated -> core (with longer half-lives).
# Recency brake: effective strength decays with the tier half-life; intensity is
#   not monotonic. consolidated/core tiers are protected by a salience floor.

# Promotion thresholds (intensity count -> tier)
THRESHOLD_CONSOLIDATED = 3   # episodic -> consolidated
THRESHOLD_CORE = 6           # consolidated -> core

# Decay-floor resistance tunables
FLOOR_STEP = 0.05            # floor gain per intensity point
FLOOR_CAP = 0.85             # max effective decay floor (never un-decayable)

# Salience floor: consolidated/core effective intensity never drops below this
INTENSITY_FLOOR = 1.0

# Restated-detection cosine range: same-fact but below merge threshold
RESTATE_RANGE = (0.78, 0.88)  # [0.78, 0.88) -> bump intensity, no merge

# Tier half-lives (seconds) for recency decay of intensity (NEXO-informed)
TIER_HALF_LIFE = {
    "episodic": 7 * 86400,       # 7 days
    "consolidated": 30 * 86400,  # 30 days
    "core": 60 * 86400,          # 60 days
}
DEFAULT_TIER = "episodic"
TIER_ORDER = ("episodic", "consolidated", "core")

# Access reinforcement delta (Phase 2 concern; reserved, not wired to a tool yet)
ACCESS_DELTA = 0.2

# Cap on the optional intensity_history event log length
INTENSITY_HISTORY_CAP = 20

def decay_enabled() -> bool:
    return get_state(DECAY_KEY, "off") == "on"

def _decay_state(memory_id: str) -> list:
    raw = get_state(f"decay:{memory_id}", "[]")
    try:
        vals = json.loads(raw)
        return vals if isinstance(vals, list) else []
    except Exception:
        return []

def _record_access(memory_id: str):
    accesses = _decay_state(memory_id)
    accesses.append(time.time())
    accesses = accesses[-DECAY_ACCESS_CAP:]
    set_state(f"decay:{memory_id}", json.dumps(accesses))

def _decay_factor(memory_id: str, metadata: dict = None) -> float:
    """Compute recency scaling factor: 1.5x fresh -> 0.3x idle. Clamped [0,1].

    T2.1: the idle floor is intensity-aware. A memory's intensity raises its floor
    (resistance), so frequently-reinforced facts survive decay longer:
        effective_floor = min(DECAY_FLOOR + intensity * FLOOR_STEP, FLOOR_CAP)
    Passing metadata=None preserves the original intensity-free behavior.
    """
    accesses = _decay_state(memory_id)
    if not accesses:
        return 1.0
    now = time.time()
    # Most recent access drives the factor; older accesses add mild reinforcement.
    newest = max(accesses)
    age_days = (now - newest) / 86400.0
    factor = DECAY_FRESH * (0.5 ** (age_days * 86400.0 / DECAY_HALFLIFE))
    floor = DECAY_FLOOR
    if metadata is not None:
        intensity = _effective_intensity(metadata)
        floor = min(DECAY_FLOOR + intensity * FLOOR_STEP, FLOOR_CAP)
    factor = max(factor, floor)
    return min(factor, 1.0)

def _apply_decay(results: list) -> list:
    """Re-rank search results by recency when decay is on. Records accesses.

    T2.1: passes each result's metadata (which includes intensity, if present)
    through to _decay_factor so the idle floor is intensity-aware.
    """
    if not decay_enabled():
        return results
    for r in results:
        mid = r.get("id")
        if mid:
            _record_access(mid)
            orig = r.get("score") or 0.0
            r["score_orig"] = round(orig, 4)
            r["score"] = round(min(orig * _decay_factor(mid, r.get("metadata")), 1.0), 4)
    results.sort(key=lambda x: x.get("score") or 0.0, reverse=True)
    return results

def mem0_decay(mode: str = "status") -> str:
    """Turn Memory Decay on/off, or check status.

    Args:
        mode: "on", "off", or "status" (default status).
    """
    mode = (mode or "status").lower()
    if mode == "on":
        set_state(DECAY_KEY, "on")
    elif mode == "off":
        set_state(DECAY_KEY, "off")
    elif mode != "status":
        return json.dumps({"error": f"mode must be on/off/status, got {mode!r}"}, indent=2)
    return json.dumps({"decay_enabled": decay_enabled(), "mode": mode}, indent=2)

# ---- Dream / consolidation ----

DREAM_MAX = 500          # legacy cap; dream now processes the full fetched list (top_k=MEMORY_FETCH_LIMIT)
DREAM_SIM_THRESHOLD = 0.88  # cosine similarity to be a candidate dupe

def _embed(texts: list) -> list:
    """Embed a list of texts with local Ollama, chunked (Ollama /api/embed rejects
    >~200 items with HTTP 400). Returns one vector per input text, in order (or empty
    on any failure)."""
    vectors = []
    for start in range(0, len(texts), 100):
        chunk = texts[start:start + 100]
        body = json.dumps({"model": "embeddinggemma:300m", "input": chunk}).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/embed",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.loads(r.read().decode())
            vectors.extend(out.get("embeddings", []))
        except Exception:
            return []
    return vectors

def _cos(a: list, b: list) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def _llm_confirm_merge(text_a: str, text_b: str) -> bool:
    """Ask the OpenAI-compatible classifier endpoint (Combo_Extractor) whether B is
    redundant with A (both about the same fact). Fail-safe returns False."""
    try:
        system = (
            "You decide whether two memory statements are about the SAME fact, such that "
            "one makes the other redundant. Answer with ONLY 'YES' or 'NO'."
        )
        body = {
            "model": CLASSIFIER_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"A: {text_a}\nB: {text_b}"},
            ],
        }
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        key = _classifier_api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(
            f"{CLASSIFIER_BASE_URL}/chat/completions",
            data=data,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.loads(r.read().decode())
        reply = ""
        choices = out.get("choices", [])
        if choices:
            msg = choices[0].get("message") or {}
            reply = msg.get("content", "") or ""
        answer = str(reply).strip().upper()
        return answer.startswith("YES")
    except Exception:
        return False

def _get_intensity(meta: dict) -> int:
    """Read the persisted reinforcement count from a memory's metadata.

    Robust to missing/odd values: returns an int >= 0. Never raises.
    """
    try:
        v = (meta or {}).get("intensity")
        i = int(v)
        return i if i >= 0 else 0
    except Exception:
        return 0

def _get_last_reinforced_at(meta: dict):
    """Return the last_reinforced_at epoch (float) from metadata, or None."""
    raw = (meta or {}).get("last_reinforced_at")
    if raw is None:
        return None
    try:
        if isinstance(raw, (int, float)):
            return float(raw)
        return _parse_time(raw)
    except Exception:
        return None

def _get_tier(meta: dict) -> str:
    """Return the memory's tier, defaulting to episodic when absent/invalid."""
    t = str((meta or {}).get("tier") or DEFAULT_TIER).strip().lower()
    return t if t in TIER_ORDER else DEFAULT_TIER

def _tier_from_intensity(intensity: int) -> str:
    """Promotion gate: map an intensity count to a tier.

    episodic (default) < THRESHOLD_CONSOLIDATED <= consolidated < THRESHOLD_CORE <= core.
    """
    if intensity >= THRESHOLD_CORE:
        return "core"
    if intensity >= THRESHOLD_CONSOLIDATED:
        return "consolidated"
    return "episodic"

def _bump_intensity(meta: dict, event: str, delta: int = 1) -> dict:
    """Increment intensity, refresh last_reinforced_at, recompute tier, and append
    a capped intensity_history event. Returns a NEW metadata dict (never mutates
    the input). event is a short tag like "absorb" or "restated".
    """
    out = dict(meta or {})
    new_intensity = _get_intensity(out) + delta
    out["intensity"] = new_intensity
    out["last_reinforced_at"] = datetime.now(timezone.utc).isoformat()
    out["tier"] = _tier_from_intensity(new_intensity)
    hist = out.get("intensity_history")
    if not isinstance(hist, list):
        hist = []
    hist.append({"type": event, "ts": time.time()})
    out["intensity_history"] = hist[-INTENSITY_HISTORY_CAP:]
    return out

def _effective_intensity(meta: dict, now: float = None) -> float:
    """NEXO strength: current intensity decayed by the tier's half-life.

    strength(t) = intensity * e^(-ln(2)/half_life * elapsed), where elapsed =
    now - last_reinforced_at. Recency brake: a stale-but-restated fact fades
    toward the salience floor. consolidated/core tiers are protected by
    INTENSITY_FLOOR (never drop below it). Returns a float >= 0. Never raises.
    """
    import math
    intensity = _get_intensity(meta)
    tier = _get_tier(meta)
    last = _get_last_reinforced_at(meta)
    if last is None or last <= 0:
        # Never reinforced / no anchor: treat as at full current intensity.
        return float(intensity)
    now = now if now is not None else time.time()
    elapsed = max(0.0, now - last)
    half_life = TIER_HALF_LIFE.get(tier, TIER_HALF_LIFE[DEFAULT_TIER])
    strength = float(intensity) * math.exp(-(math.log(2.0) / half_life) * elapsed)
    # Salience floor for promoted tiers.
    if tier in ("consolidated", "core"):
        strength = max(strength, INTENSITY_FLOOR)
    return strength

def _parse_time(value):
    """Parse a flexible time bound (ISO string, date, or human phrase) into an epoch float.

    Returns None if value is None. Raises ValueError if unparseable.
    """
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if s == "":
        return None
    # Try ISO / date string first
    s2 = s.replace("Z", "+00:00")
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(s2, fmt)
            if fmt.endswith("%z"):
                return dt.timestamp()
            return dt.timestamp()
        except ValueError:
            continue
    # Try ISO with microseconds
    try:
        return datetime.fromisoformat(s2).timestamp()
    except ValueError:
        pass
    # Human phrases
    now = time.time()
    low = s.lower()
    import re
    m = re.match(r"(\d+)\s*(second|minute|hour|day|week|month|year)s?\s*ago", low)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        secs = {"second": 1, "minute": 60, "hour": 3600, "day": 86400,
                "week": 604800, "month": 2592000, "year": 31536000}[unit]
        return now - n * secs
    phrase_map = {"yesterday": 86400, "today": 0, "last week": 604800,
                  "last month": 2592000, "last year": 31536000}
    if low in phrase_map:
        return now - phrase_map[low]
    raise ValueError(f"could not parse time: {value!r}")

def _created_epoch(mem: dict):
    iso = mem.get("created_at")
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None

def mem0_add(
    content: str,
    user_id: str = "openclaw",
    agent_id: str = None,
    run_id: str = None,
    infer: bool = True,
) -> str:
    """Store a conversation message or fact in Mem0.

    Mem0 extracts durable facts (preferences, decisions, plans) via an LLM and
    indexes them for later semantic/hybrid retrieval.

    Args:
        content: The message/text to store (user or assistant utterance).
        user_id: Scope the memory to a user (default "openclaw").
        agent_id: Optional agent scope.
        run_id: Optional run/session scope.
        infer: If True (default), Mem0's LLM extracts facts; if False, stores raw text.
    """
    payload = {
        "messages": [{"role": "user", "content": content}],
        "user_id": user_id,
        "infer": infer,
    }
    if agent_id:
        payload["agent_id"] = agent_id
    if run_id:
        payload["run_id"] = run_id
    # Category: user tags override auto-tags. If metadata already carries a
    # category, keep it; otherwise classify and inject.
    existing_meta = payload.get("metadata") or {}
    if not existing_meta.get("category"):
        category = _classify_category(content)
        existing_meta = dict(existing_meta)
        existing_meta["category"] = category
        payload["metadata"] = existing_meta
    result = _post("/memories", payload)
    _emit_webhook("memory_add", {"user_id": user_id, "agent_id": agent_id, "run_id": run_id, "result": result})
    return json.dumps(result, indent=2)

def mem0_search(
    query: str,
    user_id: str = "openclaw",
    agent_id: str = None,
    run_id: str = None,
    top_k: int = 5,
    include_merged: bool = False,
) -> str:
    """Search Mem0 for relevant stored memories.

    Retrieves memories ranked by hybrid retrieval (semantic + keyword + entity).

    Args:
        query: The natural-language search query.
        user_id: Scope search to a user (must match an add-time user_id).
        agent_id: Optional agent scope filter.
        run_id: Optional run/session scope filter.
        top_k: Max results to return (default 5).
        include_merged: If False (default), filter out soft-merged (lifecycle_state="merged") memories.
    """
    filters = {"user_id": user_id}
    if agent_id:
        filters["agent_id"] = agent_id
    if run_id:
        filters["run_id"] = run_id
    payload = {"query": query, "filters": filters, "top_k": top_k}
    result = _post("/search", payload)
    trimmed = []
    for r in result.get("results", []):
        meta = r.get("metadata") or {}
        if not include_merged and meta.get("lifecycle_state") == "merged":
            continue
        score = r.get("score")
        trimmed.append(
            {
                "id": r.get("id"),
                "memory": r.get("memory"),
                "score": score if score is not None else 0.0,
                "user_id": r.get("user_id"),
                "created_at": r.get("created_at"),
                "metadata": _search_meta(meta),
            }
        )
    trimmed = _apply_decay(trimmed)
    for t in trimmed:
        sc = t.get("score")
        t["score"] = round(sc, 4) if sc is not None else None
    return json.dumps({"results": trimmed, "error": result.get("error"), "decay": decay_enabled()}, indent=2)

_SEARCH_META_FIELDS = ("lifecycle_state", "merged_into", "merged_at", "category", "merged_from", "intensity", "tier", "last_reinforced_at")

def _search_meta(meta: dict) -> dict:
    """Extract merge/category-relevant metadata for search results (additive).

    Returns a fresh dict containing only the present merge/category fields, so
    callers can inspect lifecycle_state / merged_into / merged_at / category /
    merged_from even when include_merged=True. Never mutates the input.
    """
    sub = {}
    for k in _SEARCH_META_FIELDS:
        v = (meta or {}).get(k)
        if v is not None:
            sub[k] = v
    return sub

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mem0 MCP bridge")
    parser.add_argument("--dream", action="store_true", help="run consolidation and exit")
    parser.add_argument("--user", default="openclaw", help="scope user for dream")
    args = parser.parse_args()
    if args.dream:
        print(mem0_dream(user_id=args.user, dry_run=False))
    else:
        if mcp:
            mcp.run()
