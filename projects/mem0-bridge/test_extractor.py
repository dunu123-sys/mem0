import urllib.request, json, time

SYSTEM = """Your Task: Extract durable, actionable facts from conversations between a user and an AI assistant. Only store information that would be useful to an agent in a FUTURE session, days or weeks later.

Guidelines:
- Use third person.
- Include temporal context: "As of 2026-09-10, ..."
- Extract durable outcomes, decisions, preferences, config changes.
- NEVER store secrets/API keys.
- If nothing durable exists, return an empty list.

Return a JSON object with a key "facts" that is an array of fact strings. Example: {"facts": ["As of 2026-09-10, user prefers ..."]}"""

CONVO = """User: I restarted the OpenClaw gateway manually to make the Mem0 plugin autoCapture take effect.
Assistant: The gateway is now running with pid 7460 and the plugin is registered live.
User: The native openclaw-mem0 plugin autoCapture writes to the mem0_768d table in pgvector.
Assistant: Correct, that is the native plugin store.
User: I decided to keep the native openclaw-mem0 plugin and disable the Mem0 MCP bridge to avoid duplicate memory stores."""


def call(model, system, prompt, fmt=None):
    payload = {"model": model, "prompt": prompt, "stream": False}
    if system:
        payload["system"] = system
    if fmt:
        payload["format"] = fmt
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t = time.time()
    r = urllib.request.urlopen(req, timeout=180)
    d = json.loads(r.read())
    return d.get("response"), time.time() - t


print("=== WARMUP ===", flush=True)
print(call("llama3.2:3b", None, "Reply only with: OK")[0], flush=True)

print("\n=== TEST 1: format=json ===", flush=True)
resp, dt = call("llama3.2:3b", SYSTEM, CONVO, fmt="json")
print(f"LATENCY {dt:.1f}s", flush=True)
print("RAW RESPONSE:", flush=True)
print(resp, flush=True)

print("\n=== TEST 2: no format (plain) ===", flush=True)
resp2, dt2 = call("llama3.2:3b", SYSTEM, CONVO)
print(f"LATENCY {dt2:.1f}s", flush=True)
print("RAW RESPONSE:", flush=True)
print(resp2, flush=True)