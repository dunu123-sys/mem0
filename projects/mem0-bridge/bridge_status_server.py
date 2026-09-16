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


def _update_category(name, payload):
    """PUT /bridge/categories/{name} -> rename or update description.

    Payload: {name?: new_name, description?: string}. At least one must be present.
    """
    name = name.strip().lower()
    if not name:
        return {"ok": False, "error": "name is required"}
    if name == CLASSIFIER_FALLBACK:
        return {"ok": False, "error": f"cannot modify reserved category '{CLASSIFIER_FALLBACK}'"}
    payload = payload or {}
    new_name = str(payload.get("name", "")).strip().lower()
    new_desc = str(payload.get("description", "")).strip()
    if "name" not in payload and "description" not in payload:
        return {"ok": False, "error": "provide name and/or description"}
    if new_name == CLASSIFIER_FALLBACK:
        return {"ok": False, "error": f"cannot rename to reserved category '{CLASSIFIER_FALLBACK}'"}
    cats = _load_categories_file()
    idx = next((i for i, c in enumerate(cats) if c["name"] == name), None)
    if idx is None:
        return {"ok": False, "error": f"category '{name}' not found"}
    target_name = new_name or name
    if target_name != name and any(c["name"] == target_name for c in cats):
        return {"ok": False, "error": f"category '{target_name}' already exists"}
    if "name" in payload and new_name:
        cats[idx]["name"] = new_name
    if "description" in payload:
        cats[idx]["description"] = new_desc
    try:
        _write_categories_file(cats)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    _invalidate_bridge_registry_cache()
    return {"ok": True, "category": cats[idx]}


def _delete_category(name):
    """DELETE /bridge/categories/{name} -> remove a category (memory tags are untouched)."""
    name = name.strip().lower()
    if not name:
        return {"ok": False, "error": "name is required"}
    if name == CLASSIFIER_FALLBACK:
        return {"ok": False, "error": f"cannot delete reserved category '{CLASSIFIER_FALLBACK}'"}
    cats = _load_categories_file()
    idx = next((i for i, c in enumerate(cats) if c["name"] == name), None)
    if idx is None:
        return {"ok": False, "error": f"category '{name}' not found"}
    removed = cats.pop(idx)
    try:
        _write_categories_file(cats)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    _invalidate_bridge_registry_cache()
    return {"ok": True, "deleted": removed["name"]}


def _bridge_categories():
    """GET /bridge/categories -> registry + per-category counts (delegates to bridge).

    Returns the category taxonomy (name + description) and per-category counts
    plus the uncategorized count, using the bridge's own status logic so the
    dashboard tab renders real data. Graceful: never raises, returns zeroed
    registry with a note on failure.
    """
    try:
        import mem0_bridge as b
        raw = b.mem0_categorize_status()
        data = json.loads(raw)
        categories = [
            {"name": n, "description": d, "count": int(data.get("counts", {}).get(n, 0))}
            for n, d in [
                (r.get("name"), r.get("description")) for r in data.get("registry", [])
            ]
        ]
        return {"ok": True, "categories": categories, "uncategorized": int(data.get("uncategorized", 0))}
    except Exception as e:
        return {"ok": False, "categories": [], "uncategorized": 0, "error": str(e)}


def _bridge_recategorize():
    """POST /bridge/recategorize -> run categorize-all (dry_run=False) via bridge.

    Kicks off a full re-tag of uncategorized memories. Returns counts; graceful
    on failure (never raises).
    """
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


def _dream_log_rows():
    """Return dream_log rows (most recent first), shaped for the dashboard.

    Contract: {ok, entries:[{memory_id, canonical_id, snippet, merged_at,
    intensity_delta}]}.
    """
    entries = []
    try:
        conn = sqlite3.connect(STATE_DB, timeout=5)
        try:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(dream_log)").fetchall()]
            if cols:
                for r in conn.execute("SELECT * FROM dream_log ORDER BY rowid DESC LIMIT 500").fetchall():
                    d = dict(zip(cols, r))
                    merged_at = d.get("merged_at")
                    try:
                        if isinstance(merged_at, (int, float)):
                            merged_at = datetime.fromtimestamp(merged_at, tz=timezone.utc).isoformat()
                    except Exception:
                        merged_at = None
                    entries.append({
                        "memory_id": d.get("memory_id"),
                        "canonical_id": d.get("canonical_id"),
                        "snippet": d.get("text"),
                        "merged_at": merged_at,
                        "intensity_delta": d.get("intensity_delta"),
                    })
        finally:
            conn.close()
    except Exception:
        pass
    return {"ok": True, "entries": entries}

def _dream_overview():
    """Tier counts + intensity buckets + merged count, shaped for the dashboard.

    Contract: {ok, tier_counts:[{tier,count}], intensity_buckets:[{range,count}],
    merged_count}.
    """
    try:
        import mem0_bridge as b
        mems = b._memories_list_all()
        tier_map = {"episodic": 0, "consolidated": 0, "core": 0, "unknown": 0}
        bucket_map = {"0": 0, "1-2": 0, "3-5": 0, "6+": 0}
        merged = 0
        for m in mems:
            meta = m.get("metadata") or {}
            if meta.get("lifecycle_state") == "merged":
                merged += 1
            t = b._get_tier(meta)
            if t in tier_map:
                tier_map[t] += 1
            else:
                tier_map["unknown"] += 1
            i = b._get_intensity(meta)
            if i == 0:
                bucket_map["0"] += 1
            elif i <= 2:
                bucket_map["1-2"] += 1
            elif i <= 5:
                bucket_map["3-5"] += 1
            else:
                bucket_map["6+"] += 1
        tier_counts = [{"tier": k, "count": v} for k, v in tier_map.items()]
        intensity_buckets = [{"range": k, "count": v} for k, v in bucket_map.items()]
        return {
            "ok": True,
            "tier_counts": tier_counts,
            "intensity_buckets": intensity_buckets,
            "merged_count": merged,
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "tier_counts": [], "intensity_buckets": [], "merged_count": 0}

def _dream_merged():
    """Return all soft-merged memories, shaped for the dashboard Dreams tab.

    Contract: {ok, entries:[{id, memory, snippet, category, merged_into,
    merged_at, user_id, agent_id}], merged_count}.
    """
    try:
        import mem0_bridge as b
        data = json.loads(b.mem0_merged_list())
        return {
            "ok": True,
            "entries": data.get("results", []),
            "merged_count": data.get("merged_count", 0),
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "entries": [], "merged_count": 0}

def _dream_dryrun_compute(user_id: str, same_category: bool) -> dict:
    """Run the full (slow) dream dry-run and return the shaped dashboard payload.

    This is the compute-bound part: b.mem0_dream(dry_run=True) sweeps O(n^2)
    over up to MEMORY_FETCH_LIMIT memories plus a Combo_Extractor confirmation
    call per candidate, so it can take minutes. Never raises; returns {ok, ...}.
    """
    import mem0_bridge as b
    raw = b.mem0_dream(user_id=user_id, same_category=same_category, dry_run=True)
    data = json.loads(raw)
    id_to_text = {}
    try:
        for m in b._memories_list_all():
            if m.get("id") and m.get("memory"):
                id_to_text[m.get("id")] = m.get("memory")
    except Exception:
        id_to_text = {}
    merges = []
    for m in data.get("candidate_merges", []) or []:
        k = m.get("keep") or {}
        r = m.get("remove") or {}
        merges.append({
            "keep_id": k.get("id"),
            "keep_text": k.get("text"),
            "remove_id": r.get("id"),
            "remove_text": r.get("text"),
            "similarity": m.get("similarity"),
        })
    restated = []
    for r in data.get("candidate_restated", []) or []:
        canon_id = r.get("canonical_id")
        restated.append({
            "memory_id": canon_id,
            "text": id_to_text.get(canon_id or "", "") or r.get("other_id") or "",
            "reason": f"same-fact pair (similarity {r.get('similarity')})",
        })
    return {"ok": True, "candidate_merges": merges, "candidate_restated": restated}


def _dream_dryrun_background(job_id: str, user_id: str, same_category: bool):
    """Worker thread: compute the dry-run, then store the result on the job."""
    started = time.time()
    _dream_jobs_set(job_id, status="running", started=started)
    try:
        result = _dream_dryrun_compute(user_id, same_category)
        _dream_jobs_set(
            job_id,
            status="done",
            finished=time.time(),
            result=result,
            error=None,
            elapsed=round(time.time() - started, 3),
        )
    except Exception as e:
        _dream_jobs_set(
            job_id,
            status="error",
            finished=time.time(),
            result=None,
            error=str(e),
            elapsed=round(time.time() - started, 3),
        )
    finally:
        _dream_jobs_prune()


def _dream_dryrun(body: dict):
    """Submit a dream dry-run: return a job_id immediately; the sweep runs in a
    background thread. Clients poll GET /bridge/dreams/dryrun/status?job_id=...
    (or /bridge/dreams/dryrun for the latest job) until status == "done".

    Shape: {ok, job_id, status:"started", user_id, same_category,
            poll_url}.
    """
    try:
        user_id = (body or {}).get("user_id") or "openclaw"
        same_category = (body or {}).get("same_category", True)
        if not isinstance(same_category, bool):
            same_category = str(same_category).lower() not in ("false", "0", "no", "")
        job_id = uuid.uuid4().hex
        _dream_jobs_set(
            job_id,
            status="queued",
            created=time.time(),
            user_id=user_id,
            same_category=same_category,
            result=None,
            error=None,
        )
        t = threading.Thread(
            target=_dream_dryrun_background,
            args=(job_id, user_id, same_category),
            daemon=True,
        )
        t.start()
        return {
            "ok": True,
            "job_id": job_id,
            "status": "started",
            "user_id": user_id,
            "same_category": same_category,
            "poll_url": f"/bridge/dreams/dryrun/status?job_id={job_id}",
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _dream_dryrun_status(job_id: str):
    """Return the current state/result of a background dry-run job.

    - status "queued"|"running" -> progress payload, result None
    - status "done" -> includes result.candidate_merges / candidate_restated
    - status "error" -> includes error
    - unknown job_id -> {ok:false, status:"unknown"}
    """
    job = _dream_jobs_get(job_id) if job_id else None
    if not job_id:
        return {"ok": False, "status": "unknown", "error": "missing job_id"}
    if not job:
        return {"ok": False, "status": "unknown", "error": "unknown or expired job_id"}
    payload = {
        "ok": True,
        "job_id": job_id,
        "status": job.get("status"),
        "user_id": job.get("user_id"),
        "same_category": job.get("same_category"),
        "created": job.get("created"),
        "finished": job.get("finished"),
        "elapsed": job.get("elapsed"),
        "error": job.get("error"),
    }
    if job.get("status") == "done" and job.get("result"):
        res = job["result"]
        payload["candidate_merges"] = res.get("candidate_merges", [])
        payload["candidate_restated"] = res.get("candidate_restated", [])
    return payload


def _dream_dryrun_latest():
    """Return the most-recently-created dry-run job (running or finished)."""
    with _DREAM_JOBS_LOCK:
        if not _DREAM_JOBS:
            return {"ok": False, "status": "none", "error": "no dry-run jobs yet"}
        jid = max(_DREAM_JOBS.keys(), key=lambda k: _DREAM_JOBS[k].get("created") or 0)
    return _dream_dryrun_status(jid)

def _dream_apply(body: dict):
    """Apply: bridge mem0_dream(dry_run=False), shaped as {ok, merged, restated, updated, summary}."""
    try:
        import mem0_bridge as b
        user_id = (body or {}).get("user_id") or "openclaw"
        same_category = (body or {}).get("same_category", True)
        raw = b.mem0_dream(user_id=user_id, same_category=same_category, dry_run=False)
        data = json.loads(raw)
        merged = len(data.get("applied", []) or [])
        restated = len(data.get("restated_applied", []) or [])
        return {
            "ok": True,
            "merged": merged,
            "restated": restated,
            "updated": merged + restated,
            "summary": json.dumps(data, indent=2),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

def _dream_backfill(body: dict):
    """Backfill: bridge mem0_backfill_intensity, shaped as {ok, updated, summary}."""
    try:
        import mem0_bridge as b
        user_id = (body or {}).get("user_id") or "openclaw"
        raw = b.mem0_backfill_intensity(user_id=user_id)
        data = json.loads(raw)
        updated = int(data.get("backfilled", 0))
        return {
            "ok": True,
            "updated": updated,
            "summary": json.dumps(data, indent=2),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

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
        elif path == "/bridge/dreams/log":
            self._send(200, _dream_log_rows())
        elif path == "/bridge/dreams/overview":
            self._send(200, _dream_overview())
        elif path == "/bridge/dreams/merged":
            self._send(200, _dream_merged())
        elif path == "/bridge/dreams/dryrun":
            self._send(200, _dream_dryrun_latest())
        elif path.startswith("/bridge/dreams/dryrun/status"):
            qs = self.path.split("?", 1)[1] if "?" in self.path else ""
            job_id = ""
            if qs:
                import urllib.parse as _up
                job_id = dict(_up.parse_qsl(qs)).get("job_id", "")
            self._send(200, _dream_dryrun_status(job_id))
        else:
            self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        """Local sink for bridge webhook deliveries. Returns 200, does NOT persist
        to the webhook_events table (the bridge writes that as source of truth).
        Only the last N payloads are held in memory for dashboard introspection."""
        global RECENT_RECEIPTS
        path = self.path.split("?")[0].rstrip("/") or "/"
        if path == "/bridge/recategorize":
            self._send(200, _bridge_recategorize())
            return
        if path == "/bridge/categories":
            self._send(200, _create_category(self._read_json_body()))
            return
        if path == "/bridge/dreams/dryrun":
            self._send(200, _dream_dryrun(self._read_json_body()))
            return
        if path == "/bridge/dreams/apply":
            self._send(200, _dream_apply(self._read_json_body()))
            return
        if path == "/bridge/dreams/backfill":
            self._send(200, _dream_backfill(self._read_json_body()))
            return
        if path != "/webhook":
            self._send(404, {"ok": False, "error": "not found"})
            return
        body_bytes = b""
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length > 0:
                body_bytes = self.rfile.read(length)
        except Exception:
            body_bytes = b""
        parsed = None
        try:
            parsed = json.loads(body_bytes.decode("utf-8")) if body_bytes else None
        except Exception:
            parsed = None
        receipt = {
            "event": (parsed or {}).get("event"),
            "ts": (parsed or {}).get("ts"),
            "received_at": time.time(),
            "raw": body_bytes.decode("utf-8", errors="replace")[:2000],
        }
        RECENT_RECEIPTS.insert(0, receipt)
        RECENT_RECEIPTS = RECENT_RECEIPTS[:RECENT_RECEIPTS_MAX]
        self._send(200, {"ok": True})

    def do_PUT(self):
        path = self.path.split("?")[0].rstrip("/") or "/"
        prefix = "/bridge/categories/"
        if path.startswith(prefix):
            name = path[len(prefix):]
            self._send(200, _update_category(name, self._read_json_body()))
            return
        self._send(404, {"ok": False, "error": "not found"})

    def do_DELETE(self):
        path = self.path.split("?")[0].rstrip("/") or "/"
        prefix = "/bridge/categories/"
        if path.startswith(prefix):
            name = path[len(prefix):]
            self._send(200, _delete_category(name))
            return
        self._send(404, {"ok": False, "error": "not found"})

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

