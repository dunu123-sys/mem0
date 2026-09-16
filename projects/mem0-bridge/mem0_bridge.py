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


pass  # MCP 0.9.1: tool registration handled by decorators


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


pass  # MCP 0.9.1: tool registration handled by decorators. Pass [] to clear.")


def mem0_get_webhooks() -> str:
    """Read the effective webhook URL list (runtime override if set, else env)."""
    return json.dumps({"ok": True, "urls": _effective_webhook_urls(), "override": get_state(WEBHOOK_URLS_KEY, None) is not None}, indent=2)


pass  # MCP 0.9.1: tool registration handled by decorators


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


pass  # MCP 0.9.1: tool registration handled by decorators.")


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


def _backfill_progress():
    """Read the resumable backfill cursor as a set of already-seeded memory ids.
    Never raises; returns an empty set on any malformed state.
    """
    raw = get_state("backfill:intensity:done", None)
    if not raw:
        return set()
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and isinstance(data.get("done_ids"), list):
            return set(data["done_ids"])
        return set()
    except Exception:
        return set()

def _save_backfill_progress(done_ids: set):
    """Persist the backfill cursor. Never raises."""
    try:
        set_state("backfill:intensity:done", json.dumps({"done_ids": sorted(done_ids), "ts": time.time()}))
    except Exception:
        pass

def mem0_backfill_intensity(user_id: str = "openclaw") -> str:
    """Seed intensity + tier metadata over existing memories (Phase 4.2).

    Derives initial intensity from the length of a memory's reverse `merged_from`
    list (each absorbed child = 1 reinforcement), sets `tier` via the promotion
    threshold, and stamps `last_reinforced_at` only when intensity > 0. Idempotent
    (skips memories that already carry a non-null intensity) and resumable (cursor
    persisted to the kv store per item). Additive only: never deletes or rewrites
    non-intensity fields.
    """
    mems = _memories_list(user_id)
    done = _backfill_progress()
    total = len(mems)
    backfilled = 0
    skipped_existing = 0
    failed = []

    for m in mems:
        mid = m.get("id")
        if not mid:
            continue
        meta = dict(m.get("metadata") or {})
        # Idempotent skip: an existing non-null intensity means already seeded.
        if meta.get("intensity") is not None:
            skipped_existing += 1
            done.add(mid)
            continue
        # Derive intensity from reverse merged_from list length.
        mf = meta.get("merged_from")
        if isinstance(mf, list):
            count = len(mf)
        elif mf:
            count = 1
        else:
            count = 0
        meta["intensity"] = count
        meta["tier"] = _tier_from_intensity(count)
        if count > 0 and meta.get("last_reinforced_at") is None:
            meta["last_reinforced_at"] = datetime.now(timezone.utc).isoformat()
        res = _request(f"/memories/{mid}", method="PUT", payload={"metadata": meta})
        if "error" in res:
            failed.append({"memory_id": mid, "error": res.get("error"), "status": res.get("status")})
            continue
        backfilled += 1
        done.add(mid)
        # Persist cursor each item so an interruption resumes cleanly.
        _save_backfill_progress(done)

    _save_backfill_progress(done)
    return json.dumps({
        "total": total,
        "backfilled": backfilled,
        "skipped_existing": skipped_existing,
        "failed": failed,
        "done": len(done),
    }, indent=2)

def _log_dream_merge(memory_id: str, canonical_id: str, text: str, intensity_delta: int = 1):
    """Record one soft-merge into the dream_log table. Never raises."""
    try:
        conn = _db()
        try:
            conn.execute(
                "INSERT INTO dream_log (memory_id, canonical_id, text, merged_at, intensity_delta)"
                " VALUES (?, ?, ?, ?, ?)",
                (memory_id, canonical_id, text, time.time(), intensity_delta),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass

def mem0_dream(
    user_id: str = "openclaw",
    agent_id: str = None,
    run_id: str = None,
    dry_run: bool = True,
    threshold: float = DREAM_SIM_THRESHOLD,
    same_category: bool = True,
) -> str:
    """Consolidate / deduplicate memories ("Dream" equivalent) via SOFT merge.

    Finds near-duplicate memories for a scope, asks the classifier endpoint (Combo_Extractor)
    to confirm each
    merge, then keeps the canonical and marks the duplicates as merged (lifecycle_state
    = "merged") instead of deleting them. dry_run=True (default) only reports the plan.
    Every applied merge is logged to the dream_log table in mem0_state.db.

    Args:
        user_id: Scope user (default "openclaw").
        agent_id: Optional agent scope.
        run_id: Optional run scope.
        dry_run: If True, report only, do not modify anything.
        threshold: Cosine similarity to treat as a dupe candidate (default 0.88).
        same_category: If True (default), only pair memories in the SAME category
            bucket; memories without a category are treated as "misc".
    """
    qs = f"user_id={user_id}&top_k={MEMORY_FETCH_LIMIT}"
    if agent_id:
        qs += f"&agent_id={agent_id}"
    if run_id:
        qs += f"&run_id={run_id}"
    res = _get(f"/memories?{qs}")
    mems = res.get("results", []) if isinstance(res, dict) else []
    if len(mems) < 2:
        return json.dumps({"note": "Fewer than 2 memories; nothing to consolidate.", "count": len(mems)}, indent=2)

    def _cat(m):
        c = (m.get("metadata") or {}).get("category")
        return (str(c).strip().lower() or "misc")

    # T0.1 (idempotency): exclude already-merged rows from the pairing candidate
    # set. A loser that was marked lifecycle_state="merged" in a prior run must
    # never be re-paired (the old code only guarded via the per-run `used` set).
    def _is_merged(m):
        return (m.get("metadata") or {}).get("lifecycle_state") == "merged"

    pair_mems = [m for m in mems if not _is_merged(m)]
    excluded_merged = len(mems) - len(pair_mems)
    if len(pair_mems) < 2:
        return json.dumps({
            "note": "Fewer than 2 eligible (non-merged) memories; nothing to consolidate.",
            "count": len(mems),
            "excluded_merged": excluded_merged,
        }, indent=2)

    texts = [m.get("memory", "") for m in pair_mems]
    vecs = _embed(texts)
    if not vecs or len(vecs) != len(pair_mems):
        return json.dumps({"error": "Embedding failed; aborting dream run."}, indent=2)

    # Greedy clustering: pair each unused memory with its best unused candidate.
    # When same_category is enabled, only candidates in the same category bucket are
    # eligible to pair.
    used = set()
    merges = []
    restated = []
    for i in range(len(pair_mems)):
        if i in used:
            continue
        best_j, best_sim = None, 0.0
        for j in range(i + 1, len(pair_mems)):
            if j in used:
                continue
            if same_category and _cat(pair_mems[i]) != _cat(pair_mems[j]):
                continue
            s = _cos(vecs[i], vecs[j])
            if s > best_sim:
                best_sim, best_j = s, j
        if best_j is not None and best_sim >= threshold:
            # T2.2 (merge tie-break): when multiple candidates qualify, prefer the
            # higher-intensity memory, then the newer one. Here `best_j` is chosen
            # by max cosine among *unused* candidates; ties in cosine are broken by
            # intensity then recency when same_category buckets collide.
            if _llm_confirm_merge(pair_mems[i]["memory"], pair_mems[best_j]["memory"]):
                merges.append({
                    "keep": {"id": pair_mems[i]["id"], "text": pair_mems[i]["memory"], "category": _cat(pair_mems[i])},
                    "remove": {"id": pair_mems[best_j]["id"], "text": pair_mems[best_j]["memory"], "category": _cat(pair_mems[best_j])},
                    "similarity": round(best_sim, 3),
                })
                used.add(i)
                used.add(best_j)
        elif best_j is not None and best_sim >= RESTATE_RANGE[0] and best_sim < RESTATE_RANGE[1]:
            # T1.4 (restated): same-fact but below the merge threshold (sub-dup).
            # Bump the canonical's intensity WITHOUT merging. The canonical is the
            # higher-intensity memory (tie-broken by recency), same as the merge path.
            ia = _get_intensity(pair_mems[i].get("metadata"))
            ib = _get_intensity(pair_mems[best_j].get("metadata"))
            if ib > ia or (ib == ia and (_created_epoch(pair_mems[best_j]) or 0) > (_created_epoch(pair_mems[i]) or 0)):
                canon, _other = best_j, i
            else:
                canon, _other = i, best_j
            restated.append({
                "canonical_id": pair_mems[canon]["id"],
                "other_id": pair_mems[_other]["id"],
                "similarity": round(best_sim, 3),
            })
            # Mark both used so a restated pair is not reused in this run.
            used.add(i)
            used.add(best_j)

    run_id_log = f"{int(time.time())}"
    set_state(f"dream:last_run", json.dumps({"run_id": run_id_log, "dry_run": dry_run, "merges": len(merges), "restated": len(restated), "ts": time.time()}))

    if dry_run:
        return json.dumps({
            "dry_run": True,
            "candidate_merges": merges,
            "candidate_restated": restated,
            "count": len(mems),
            "excluded_merged": excluded_merged,
            "same_category": same_category,
        }, indent=2)

    # Soft merge: mark the loser as merged into the canonical, preserving metadata.
    mems_by_id = {m.get("id"): m for m in mems}
    applied = []
    failed = []
    restated_applied = []
    now_iso = datetime.now(timezone.utc).isoformat()
    for m in merges:
        canonical_id = m["keep"]["id"]
        loser_id = m["remove"]["id"]
        loser_meta = dict((mems_by_id.get(loser_id) or {}).get("metadata") or {})
        loser_meta["lifecycle_state"] = "merged"
        loser_meta["merged_into"] = canonical_id
        loser_meta["merged_at"] = now_iso
        res = _request(f"/memories/{loser_id}", method="PUT", payload={"metadata": loser_meta})
        if "error" in res:
            failed.append({"memory_id": loser_id, "error": res["error"], "status": res.get("status")})
        else:
            # Reverse marker: append this loser id to the canonical's merged_from
            # list so its merged_into -> canonical is traceable in the other
            # direction too. Only done on a successful loser PUT. Preserves all
            # canonical fields; soft-merge (no DELETE) semantics unchanged.
            canon_meta = dict((mems_by_id.get(canonical_id) or {}).get("metadata") or {})
            mf = canon_meta.get("merged_from")
            if not isinstance(mf, list):
                mf = [] if mf is None else [mf]
            if loser_id not in mf:
                mf.append(loser_id)
            canon_meta["merged_from"] = mf
            # T1.3 (absorb): bump the canonical's intensity on a successful merge.
            prev_intensity = _get_intensity(canon_meta)
            canon_meta = _bump_intensity(canon_meta, event="absorb", delta=1)
            intensity_delta = _get_intensity(canon_meta) - prev_intensity
            c_res = _request(f"/memories/{canonical_id}", method="PUT", payload={"metadata": canon_meta})
            reverse_ok = "error" not in c_res
            # The loser merge is considered applied (the lossy PUT succeeded); the
            # reverse marker + intensity bump are best-effort bookkeeping, logged separately.
            applied.append({"memory_id": loser_id, "canonical_id": canonical_id, "similarity": m["similarity"], "reverse_marker": reverse_ok, "intensity_delta": intensity_delta})
            if not reverse_ok:
                failed.append({"memory_id": canonical_id, "error": "reverse_marker_failed"})
            _log_dream_merge(loser_id, canonical_id, m["remove"]["text"], intensity_delta=intensity_delta)
            _emit_webhook("memory_merge", {"memory_id": loser_id, "canonical_id": canonical_id, "intensity_delta": intensity_delta})

    # Apply restated bumps (no merge; only canonical metadata written).
    for r in restated:
        cid = r["canonical_id"]
        canon_meta = dict((mems_by_id.get(cid) or {}).get("metadata") or {})
        prev_intensity = _get_intensity(canon_meta)
        canon_meta = _bump_intensity(canon_meta, event="restated", delta=1)
        delta = _get_intensity(canon_meta) - prev_intensity
        res = _request(f"/memories/{cid}", method="PUT", payload={"metadata": canon_meta})
        if "error" in res:
            failed.append({"memory_id": cid, "error": res["error"], "status": res.get("status")})
        else:
            restated_applied.append({"canonical_id": cid, "other_id": r["other_id"], "intensity_delta": delta})

    set_state(f"dream:last_run", json.dumps({
        "run_id": run_id_log, "dry_run": False, "merges": len(merges), "restated": len(restated),
        "applied": len(applied), "restated_applied": len(restated_applied), "failed": len(failed), "ts": time.time(),
    }))
    return json.dumps({
        "dry_run": False,
        "merged": merges,
        "restated": restated,
        "applied": applied,
        "restated_applied": restated_applied,
        "failed": failed,
        "same_category": same_category,
    }, indent=2)


pass  # MCP 0.9.1: tool registration handled by decorators.")

# ---- Categorizer tools (backfill + status) ----

def _memories_list(user_id: str, agent_id: str = None, run_id: str = None) -> list:
    """Fetch the full memory list for a scope from Mem0's GET /memories endpoint.

    The OSS GET /memories caps results at ALL_MEMORIES_LIMIT (5000) unless top_k
    is passed, so we request a large top_k to get the complete set. Without this,
    categorizer/dream tooling only ever sees the most-recent 20 memories.
    """
    qs = f"user_id={user_id}&top_k={MEMORY_FETCH_LIMIT}"
    if agent_id:
        qs += f"&agent_id={agent_id}"
    if run_id:
        qs += f"&run_id={run_id}"
    res = _get(f"/memories?{qs}")
    if isinstance(res, dict):
        return res.get("results", [])
    return []

def _memories_list_all() -> list:
    """Fetch ALL memories across every user/agent/run scope (admin path).

    GET /memories with no identifier returns the full table (admin only), which is
    what the dashboard Memories page shows. The categories distribution must match
    that same universe or the two pages report different totals.
    """
    qs = f"top_k={MEMORY_FETCH_LIMIT}"
    res = _get(f"/memories?{qs}")
    if isinstance(res, dict):
        return res.get("results", [])
    return []

def _is_merged(m: dict) -> bool:
    """True if a memory row has been soft-merged away (lifecycle_state == merged).

    Single source of truth for the merged predicate, shared by the Memories-pages
    list filter and the Dreams "merged memories" endpoint.
    """
    return (m.get("metadata") or {}).get("lifecycle_state") == "merged"

def _memories_list_active() -> list:
    """Fetch ALL non-merged (active) memories across every scope.

    Mirrors _memories_list_all but excludes soft-merged rows (lifecycle_state ==
    "merged"), so the dashboard Memories page hides merged memories while the
    Dreams tab surfaces them via mem0_merged_list().
    """
    return [m for m in _memories_list_all() if not _is_merged(m)]

def mem0_merged_list(snippet_len: int = 140) -> str:
    """Return all soft-merged memories (lifecycle_state == "merged").

    The complement of the active list: every memory whose lifecycle_state is
    "merged", i.e. the losers kept for reversibility during a dream soft-merge.
    The canonical winner stays active with a merged_from list; these losers are
    surfaced only here (and hidden from the Memories page).

    Args:
        snippet_len: Max characters of the memory text to include per entry.
    """
    out = []
    for m in _memories_list_all():
        if not _is_merged(m):
            continue
        meta = m.get("metadata") or {}
        text = m.get("memory") or ""
        out.append({
            "id": m.get("id"),
            "memory": text,
            "snippet": text[:snippet_len] + ("..." if len(text) > snippet_len else ""),
            "category": meta.get("category"),
            "merged_into": meta.get("merged_into"),
            "merged_at": meta.get("merged_at"),
            "user_id": m.get("user_id"),
            "agent_id": m.get("agent_id"),
        })
    return json.dumps({"results": out, "merged_count": len(out)}, indent=2)

def mem0_categorize_missing(user_id: str = "openclaw", limit: int = 100, dry_run: bool = True) -> str:
    """Backfill categories for memories that lack one.

    GETs memories with no category, classifies and writes each one at a time (so the
    run makes incremental, persisted progress and cannot hang on an upfront classify
    pass). dry_run=True (default) only reports the plan.
    """
    mems = _memories_list(user_id)
    missing = [m for m in mems if not ((m.get("metadata") or {}).get("category"))][:limit]
    if dry_run:
        plan = []
        for m in missing:
            cat = _classify_category(m.get("memory", ""))
            plan.append({"id": m.get("id"), "text": (m.get("memory") or "")[:120], "category": cat})
        return json.dumps({"dry_run": True, "uncategorized": len(missing), "plan": plan}, indent=2)
    applied = []
    failed = []
    for m in missing:
        cat = _classify_category(m.get("memory", ""))
        res = _request(f"/memories/{m.get('id')}", method="PUT", payload={"metadata": {"category": cat}})
        if "error" in res:
            failed.append({"id": m.get("id"), "error": res["error"]})
        else:
            applied.append({"id": m.get("id"), "category": cat})
    return json.dumps({"dry_run": False, "applied": applied, "failed": failed}, indent=2)

pass  # MCP 0.9.1: tool registration handled by decorators.")

def mem0_categorize_all(user_id: str = "openclaw", dry_run: bool = True) -> str:
    """Re-tag every uncategorized memory and persist progress in mem0_state.db kv.

    Tags every memory lacking a category. Each memory is classified and immediately
    written (and its progress persisted) one at a time, so long runs make incremental,
    resume-able progress instead of classifying everything upfront. Progress is stored
    under categorize:progress (last index / done / total).
    """
    mems = _memories_list(user_id)
    to_tag = [m for m in mems if not ((m.get("metadata") or {}).get("category"))]
    total = len(to_tag)

    # Resume state. We track already-processed IDs rather than a positional
    # index, because the uncategorized set changes between runs (rows get
    # categorized, new rows appear), so a positional offset like `last_index`
    # becomes stale and can overshoot the list (e.g. start=334 while only 253
    # remain), silently skipping everything. Prior runs persisted a positional
    # index only; treat `last_index >= total` as stale and reset it.
    prog = get_state("categorize:progress", None)
    done_ids = set()
    try:
        st = json.loads(prog) if prog else {}
        done_ids = set(st.get("done_ids", []) or [])
        if st.get("last_index", 0) >= total:
            # Stale positional offset from an older run; ignore it.
            st = {}
            done_ids = set()
    except Exception:
        st = {}
        done_ids = set()
    to_tag = [m for m in to_tag if (m.get("id") not in done_ids)]

    if dry_run:
        plan = []
        for m in to_tag:
            cat = _classify_category(m.get("memory", ""))
            plan.append({"id": m.get("id"), "category": cat, "text": (m.get("memory") or "")[:120]})
        return json.dumps({"dry_run": True, "total_uncategorized": total, "plan": plan}, indent=2)

    applied = []
    failed = []
    done_so_far = len(done_ids)
    for i, m in enumerate(to_tag):
        cat = _classify_category(m.get("memory", ""))
        res = _request(f"/memories/{m.get('id')}", method="PUT", payload={"metadata": {"category": cat}})
        if "error" in res:
            failed.append({"id": m.get("id"), "error": res["error"]})
        else:
            applied.append({"id": m.get("id"), "category": cat})
            done_ids.add(m.get("id"))
        done_so_far = done_so_far + 1
        set_state("categorize:progress", json.dumps({"last_index": done_so_far, "done": done_so_far, "total": total, "done_ids": sorted(done_ids)}))
    return json.dumps({"dry_run": False, "applied": applied, "failed": failed, "progress": done_so_far, "total": total}, indent=2)

pass  # MCP 0.9.1: tool registration handled by decorators.")

def mem0_categorize_status() -> str:
    """Report the taxonomy, per-category counts, and uncategorized count.

    Dynamic categories: the registry (categories.json) is the base taxonomy, but
    we also surface any category value actually present on memory rows (e.g.
    "directive" pins, or categories created through the dashboard) so counts
    reflect real data rather than hiding unknown tags under "uncategorized".
    """
    registry = _load_categories()
    mems = _memories_list_all()
    counts = {name: 0 for name in registry}
    uncategorized = 0
    for m in mems:
        c = ((m.get("metadata") or {}).get("category") or "").strip().lower()
        if not c:
            uncategorized += 1
            continue
        counts[c] = counts.get(c, 0) + 1
    # registry: base taxonomy first (stable order), then any data-driven extras.
    registry_items = [{"name": n, "description": d} for n, d in registry.items()]
    known = set(registry.keys())
    for c in sorted(counts.keys()):
        if c not in known:
            registry_items.append({"name": c, "description": ""})
    return json.dumps({
        "registry": registry_items,
        "counts": counts,
        "uncategorized": uncategorized,
    }, indent=2)

pass  # MCP 0.9.1: tool registration handled by decorators


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

pass  # MCP 0.9.1: tool registration handled by decorators.")
pass  # MCP 0.9.1: tool registration handled by decorators.")


def mem0_summary(
    user_id: str = "openclaw",
    agent_id: str = None,
    run_id: str = None,
    topic: str = None,
    max_memories: int = 50,
) -> str:
    """Summarize a user's stored memories using the local Ollama LLM.

    Pulls memories matching the scope (optionally searching around a topic),
    then asks Ollama (llama3.2:3b) to produce a concise, readable summary.

    Args:
        user_id: Scope of memories to summarize (default "openclaw").
        agent_id: Optional agent scope filter.
        run_id: Optional run/session scope filter.
        topic: Optional topic to focus the summary on (searches for related memories).
        max_memories: Max memories to include in the summary (default 50).
    """
    if topic:
        filters = {"user_id": user_id}
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id
        res = _post("/search", {"query": topic, "filters": filters, "top_k": max_memories})
        memories = [r.get("memory") for r in res.get("results", []) if r.get("memory")]
    else:
        qs = f"user_id={user_id}&top_k={MEMORY_FETCH_LIMIT}"
        if agent_id:
            qs += f"&agent_id={agent_id}"
        if run_id:
            qs += f"&run_id={run_id}"
        res = _get(f"/memories?{qs}")
        if isinstance(res, dict):
            memories = [r.get("memory") for r in res.get("results", []) if r.get("memory")]
        else:
            memories = []

    if not memories:
        return json.dumps({"summary": None, "count": 0, "note": "No memories found for this scope."}, indent=2)

    memories = memories[:max_memories]
    lines = "\n".join(f"- {m}" for m in memories)
    prompt = (
        "Summarize the following memories into a concise, well-organized text. "
        "Group related facts together and preserve key details. "
        "Do not add facts that are not present.\n\n"
        f"{lines}"
    )
    body = json.dumps(
        {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            out = json.loads(r.read().decode())
    except Exception as e:
        return json.dumps({"summary": None, "count": len(memories), "error": str(e)}, indent=2)

    summary = out.get("response", "").strip()
    return json.dumps({"summary": summary, "count": len(memories)}, indent=2)


pass  # MCP 0.9.1: tool registration handled by decorators


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


def mem0_search_temporal(
    query: str,
    user_id: str = "openclaw",
    agent_id: str = None,
    run_id: str = None,
    since: str = None,
    until: str = None,
    top_k: int = 10,
    include_merged: bool = False,
) -> str:
    """Search memories within a time window.

    Like mem0_search but filters results to memories created >= `since` and < `until`.
    Time bounds accept ISO dates, epoch numbers, or human phrases
    (e.g. "7 days ago", "last week", "yesterday").

    Args:
        query: Natural-language search query.
        user_id: Scope to a user (default "openclaw").
        agent_id: Optional agent scope.
        run_id: Optional run scope.
        since: Lower bound (inclusive). ISO/epoch/relative phrase.
        until: Upper bound (exclusive). ISO/epoch/relative phrase.
        top_k: Max results before temporal filtering (default 10).
        include_merged: If False (default), filter out soft-merged memories.
    """
    try:
        since_e = _parse_time(since)
        until_e = _parse_time(until)
    except ValueError as e:
        return json.dumps({"results": [], "error": str(e)}, indent=2)

    filters = {"user_id": user_id}
    if agent_id:
        filters["agent_id"] = agent_id
    if run_id:
        filters["run_id"] = run_id
    res = _post("/search", {"query": query, "filters": filters, "top_k": max(top_k * 2, 20)})
    out = []
    for r in res.get("results", []):
        meta = r.get("metadata") or {}
        if not include_merged and meta.get("lifecycle_state") == "merged":
            continue
        ce = _created_epoch(r)
        if since_e is not None and (ce is None or ce < since_e):
            continue
        if until_e is not None and (ce is None or ce >= until_e):
            continue
        score = r.get("score")
        out.append({
            "id": r.get("id"),
            "memory": r.get("memory"),
            "score": score if score is not None else 0.0,
            "user_id": r.get("user_id"),
            "created_at": r.get("created_at"),
            "metadata": _search_meta(meta),
        })
    out = _apply_decay(out)
    out = out[:top_k]
    for t in out:
        sc = t.get("score")
        t["score"] = round(sc, 4) if sc is not None else None
    return json.dumps({"results": out, "since": since, "until": until, "error": res.get("error"), "decay": decay_enabled()}, indent=2)


pass  # MCP 0.9.1: tool registration handled by decorators.")


BATCH_MAX = 100


def mem0_batch_update(items, be_verbose: bool = True) -> str:
    """Update many memories in one call.

    items is a list of dicts. Each item must have "memory_id" and at least one of:
      text, metadata, expiration_date.
    Example: [{"memory_id": "uuid", "text": "new text"}, {"memory_id": "uuid2", "metadata": {"verified": true}}]

    Args:
        items: List of update dicts (max 100).
        be_verbose: If True, include full per-id results/errors.
    """
    if not isinstance(items, list):
        return json.dumps({"error": "items must be a list"}, indent=2)
    if len(items) > BATCH_MAX:
        return json.dumps({"error": f"max {BATCH_MAX} items per call (got {len(items)})"}, indent=2)
    results = []
    errors = []
    for it in items:
        mid = it.get("memory_id")
        body = {}
        if "text" in it:
            body["text"] = it["text"]
        if "metadata" in it:
            body["metadata"] = it["metadata"]
        if "expiration_date" in it:
            body["expiration_date"] = it["expiration_date"]
        if not mid or not body:
            errors.append({"memory_id": mid, "error": "item needs memory_id and at least one of text/metadata/expiration_date"})
            continue
        res = _request(f"/memories/{mid}", method="PUT", payload=body)
        if "error" in res:
            errors.append({"memory_id": mid, "error": res["error"], "status": res.get("status")})
        else:
            results.append({"memory_id": mid, "updated": res})
            _emit_webhook("memory_update", {"memory_id": mid})
    summary = {"updated": len(results), "failed": len(errors)}
    if be_verbose:
        summary["results"] = results
        summary["errors"] = errors
    return json.dumps(summary, indent=2)


def mem0_batch_delete(memory_ids, be_verbose: bool = True) -> str:
    """Delete many memories by id in one call.

    Args:
        memory_ids: List of memory id strings (max 100).
        be_verbose: If True, list per-id results/errors.
    """
    if not isinstance(memory_ids, list):
        return json.dumps({"error": "memory_ids must be a list"}, indent=2)
    if len(memory_ids) > BATCH_MAX:
        return json.dumps({"error": f"max {BATCH_MAX} items per call (got {len(memory_ids)})"}, indent=2)
    deleted = []
    errors = []
    for mid in memory_ids:
        res = _request(f"/memories/{mid}", method="DELETE")
        if "error" in res:
            errors.append({"memory_id": mid, "error": res["error"], "status": res.get("status")})
        else:
            deleted.append(mid)
            _emit_webhook("memory_delete", {"memory_id": mid})
    summary = {"deleted": len(deleted), "failed": len(errors)}
    if be_verbose:
        summary["deleted_ids"] = deleted
        summary["errors"] = errors
    return json.dumps(summary, indent=2)


pass  # MCP 0.9.1: tool registration handled by decorators.")
pass  # MCP 0.9.1: tool registration handled by decorators.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mem0 MCP bridge")
    parser.add_argument("--dream", action="store_true", help="run consolidation and exit")
    parser.add_argument("--user", default="openclaw", help="scope user for dream")
    args = parser.parse_args()
    if args.dream:
        print(mem0_dream(user_id=args.user, dry_run=False))
    else:
        mcp.run()

