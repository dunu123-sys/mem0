#!/usr/bin/env python3
"""
Mem0 bridge status sidecar.

Exposes the mem0 MCP bridge's webhook + analytics state (host-side) as JSON
so the self-hosted Mem0 dashboard (running in Docker, cannot read the host
filesystem) can consume real local data.

Python stdlib only (http.server). No pip dependencies.

Endpoints:
    GET /healthz        -> 200 {"ok": true}
    GET /bridge/status  -> 200 JSON (see contract below)
    GET /bridge/categories          -> taxonomy + counts
    POST /bridge/categories         -> create category {name, description}
    PUT  /bridge/categories/{name}  -> rename / update description
    DELETE /bridge/categories/{name}-> delete category
    POST /bridge/recategorize       -> run categorize-all (dry_run=False)
    POST /webhook       -> 200 {"ok": true}  (receiver/sink for bridge delivery)
    OPTIONS *           -> 200 (CORS preflight)

Config:
    BRIDGE_STATUS_PORT  -> listen port (default 18900)
    WEBHOOK_URLS        -> comma-separated webhook URLs (fallback default)

Data sources:
    - Runtime webhook override in mem0_state.db kv (webhooks:urls) wins;
      otherwise WEBHOOK_URLS env var -> webhook URLs + enabled flag.
    - mem0_state.db (kv)  -> decay_enabled, dream:last_run
    - mem0_state.db (webhook_events) -> last_events (written by the bridge)
    - in-memory RECENT_RECEIPTS      -> last N POST /webhook payloads (sink only)
"""
import os
import json
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BRIDGE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_DB = os.path.join(BRIDGE_DIR, "mem0_state.db")
CATEGORIES_FILE = os.path.join(BRIDGE_DIR, "categories.json")
CLASSIFIER_FALLBACK = "misc"

# State-DB key where the bridge persists its runtime webhook URL override
# (matching mem0_bridge.py). The sidecar reads it so /bridge/status reflects
# the effective runtime config, not just the static env default.
WEBHOOK_URLS_KEY = "webhooks:urls"

PORT = int(os.environ.get("BRIDGE_STATUS_PORT", "18900"))

# In-memory sink for POST /webhook deliveries (most recent first). Sink only:
# the webhook_events table (written by the bridge) stays the source of truth.
RECENT_RECEIPTS = []
RECENT_RECEIPTS_MAX = 50

# Background dream dry-run jobs. keyed by job_id -> {"status", "created",
# "finished", "user_id", "same_category", "error", "result"}. Guarded by
# _DREAM_JOBS_LOCK because ThreadingHTTPServer serves requests concurrently.
_DREAM_JOBS = {}
_DREAM_JOBS_LOCK = threading.Lock()
_DREAM_JOBS_MAX = 20
_DREAM_RESULT_TTL = 1800.0  # seconds a finished result is retained

def _dream_jobs_set(job_id, **fields):
    with _DREAM_JOBS_LOCK:
        job = _DREAM_JOBS.setdefault(job_id, {})
        job.update(fields)
        return job

def _dream_jobs_get(job_id):
    with _DREAM_JOBS_LOCK:
        job = _DREAM_JOBS.get(job_id)
        return dict(job) if job else None

def _dream_jobs_prune():
    """Drop the oldest finished jobs if we exceed the cap, and expired finishes."""
    with _DREAM_JOBS_LOCK:
        now = time.time()
        for jid in list(_DREAM_JOBS.keys()):
            job = _DREAM_JOBS[jid]
            if job.get("status") == "done" and job.get("finished"):
                if now - job["finished"] > _DREAM_RESULT_TTL:
                    del _DREAM_JOBS[jid]
        if len(_DREAM_JOBS) > _DREAM_JOBS_MAX:
            # Remove oldest finished jobs until under cap; keep running ones.
            finished = [j for j in _DREAM_JOBS.items() if j[1].get("status") == "done"]
            finished.sort(key=lambda kv: kv[1].get("finished") or 0)
            while len(_DREAM_JOBS) > _DREAM_JOBS_MAX and finished:
                jid, _ = finished.pop(0)
                _DREAM_JOBS.pop(jid, None)

def _get_state(key, default=None):
    """Read a value from the bridge kv table. Never raises on missing/malformed state."""
    try:
        conn = sqlite3.connect(STATE_DB, timeout=5)
        try:
            row = conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
            return row[0] if row else default
        finally:
            conn.close()
    except Exception:
        return default

def _webhook_urls():
    """Effective webhook URL list: runtime override (state DB) wins over env."""
    raw = _get_state(WEBHOOK_URLS_KEY, None)
    if raw is not None:
        try:
            val = json.loads(raw)
            if isinstance(val, list):
                return [u.strip() for u in val if isinstance(u, str) and u.strip()]
        except Exception:
            pass
    return [u.strip() for u in os.environ.get("WEBHOOK_URLS", "").split(",") if u.strip()]

def _bridge_status():
    # ---- webhook ----
    urls = _webhook_urls()
    webhook_enabled = len(urls) > 0
    # No webhook-event log table exists in the bridge DB yet; inspect the actual schema.
    try:
        conn = sqlite3.connect(STATE_DB, timeout=5)
        try:
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            event_table = "webhook_events" if "webhook_events" in tables else None
        finally:
            conn.close()
        last_events = []
        if event_table:
            conn = sqlite3.connect(STATE_DB, timeout=5)
            try:
                rows = conn.execute(
                    "SELECT * FROM webhook_events ORDER BY rowid DESC LIMIT 50").fetchall()
                cols = [r[1] for r in conn.execute("PRAGMA table_info(webhook_events)").fetchall()]
                for r in rows:
                    d = dict(zip(cols, r))
                    last_events.append({
                        "event": d.get("event"),
                        "url": d.get("url"),
                        "at": d.get("at"),
                        "ok": bool(d.get("ok", 0)),
                    })
            finally:
                conn.close()
    except Exception:
        # Fail gracefully: never 500 on missing state.
        event_table = None
        last_events = []

    # ---- bridge / decay / dream ----
    decay_enabled = _get_state("decay_enabled", "off") == "on"

    last_dream_at = None
    last_dream_summary = None
    dream_raw = _get_state("dream:last_run", None)
    if dream_raw:
        try:
            dream = json.loads(dream_raw) if isinstance(dream_raw, str) else None
            if isinstance(dream, dict):
                ts = dream.get("ts")
                if isinstance(ts, (int, float)):
                    last_dream_at = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                # The bridge's dream log stores run_id/dry_run/merges/ts/applied only;
                # there is no human summary string, so we leave summary null rather
                # than fabricate one.
                last_dream_summary = None
        except Exception:
            last_dream_at = None
            last_dream_summary = None

    return {
        "ok": True,
        "webhook": {
            "enabled": webhook_enabled,
            "urls": urls,
            "last_events": last_events,
        },
        "bridge": {
            "decay_enabled": decay_enabled,
            "last_dream_at": last_dream_at,
            "last_dream_summary": last_dream_summary,
        },
    }

def _load_categories_file():
    """Read categories.json into an ordered list of {name, description}. Never raises."""
    try:
        with open(CATEGORIES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        cats = data.get("categories", [])
        out = []
        for c in cats:
            name = str(c.get("name", "")).strip().lower()
            if name:
                out.append({"name": name, "description": str(c.get("description", "")).strip()})
        return out
    except Exception:
        return []

def _write_categories_file(cats):
    """Atomically write the category list back to categories.json. Raises on failure."""
    tmp = CATEGORIES_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"categories": cats}, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, CATEGORIES_FILE)

def _invalidate_bridge_registry_cache():
    """Clear mem0_bridge's cached category registry so it re-reads categories.json.

    Best-effort: only works when the bridge module is importable in this process
    (it is, since the sidecar imports it). Never raises.
    """
    try:
        import mem0_bridge as b
        b._CATEGORY_REGISTRY = None
    except Exception:
        pass

def _create_category(payload):
    """POST /bridge/categories -> add a category. Payload: {name, description}."""
    name = str((payload or {}).get("name", "")).strip().lower()
    description = str((payload or {}).get("description", "")).strip()
    if not name:
        return {"ok": False, "error": "name is required"}
    if name == CLASSIFIER_FALLBACK:
        return {"ok": False, "error": f"cannot modify reserved category '{CLASSIFIER_FALLBACK}'"}
    cats = _load_categories_file()
    existing = next((c for c in cats if c["name"] == name), None)
    if existing is not None:
        return {"ok": True, "category": existing, "already_existed": True}
    cats.append({"name": name, "description": description})
    try:
        _write_categories_file(cats)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    _invalidate_bridge_registry_cache()
    return {"ok": True, "category": {"name": name, "description": description}}

def _bridge_categories():
    """GET /bridge/categories -> registry + counts."""
    try:
        import mem0_bridge as b
        raw = b.mem0_categorize_status()
        data = json.loads(raw)
        categories = [
            {"name": r.get("name"), "description": r.get("description"), "count": int(data.get("counts", {}).get(r.get("name"), 0))}
            for r in data.get("registry", [])
        ]
        return {"ok": True, "categories": categories, "uncategorized": int(data.get("uncategorized", 0))}
    except Exception as e:
        return {"ok": False, "categories": [], "uncategorized": 0, "error": str(e)}

def _bridge_recategorize():
    """POST /bridge/recategorize -> run categorize-all (dry_run=False)."""
    try:
        import mem0_bridge as b
        raw = b.mem0_categorize_all(dry_run=False)
        data = json.loads(raw)
        return {
            "ok": True,
            "categorized": len(data.get("applied", [])),
            "skipped": len(data.get("failed", [])),
            "progress": data.get("progress"),
            "total": data.get("total"),
        }
    except Exception as e:
        return {"ok": False, "categorized": 0, "skipped": 0, "error": str(e)}

class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload, extra_headers=None):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(200, {})

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            return {}

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/") or "/"
        if path == "/healthz":
            self._send(200, {"ok": True})
        elif path == "/bridge/status":
            self._send(200, _bridge_status())
        elif path == "/bridge/categories":
            self._send(200, _bridge_categories())
        else:
            self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/") or "/"
        if path == "/bridge/recategorize":
            self._send(200, _bridge_recategorize())
            return
        if path == "/bridge/categories":
            self._send(200, _create_category(self._read_json_body()))
            return
        if path != "/webhook":
            self._send(404, {"ok": False, "error": "not found"})
            return
        self._send(200, {"ok": True})

    def log_message(self, fmt, *args):
        # Keep it quiet on the console.
        pass

def main():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"bridge-status sidecar listening on 0.0.0.0:{PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("shutting down", flush=True)
        server.shutdown()

if __name__ == "__main__":
    main()
