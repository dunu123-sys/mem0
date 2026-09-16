# Mem0 MCP Bridge

OpenClaw-facing MCP bridge for self-hosted Mem0. Exposes extraction, categorization, and consolidation as stdio tools.

## Files

- **mem0_bridge.py** — Main bridge (extract, categorize, search, dream, decay, batch ops)
- **bridge_status_server.py** — HTTP sidecar for webhook/analytics (port 18900)
- **bridge_status_service.py** — Windows service wrapper (pywin32)
- **categories.json** — Category registry (taxonomy + counts)

## Setup

1. Install deps: `pip install pywin32` (for Windows service wrapper)
2. Configure MEM0_URL, OLLAMA_URL, CLASSIFIER_BASE_URL in env
3. Start sidecar: `python bridge_status_service.py install && sc start Mem0BridgeStatus`
4. MCP bridge runs in OpenClaw via MCP stdio

## Phase 0 & 1

This directory will receive Phase 0 & 1 instrumentation:
- `capture_log` table (SQLite) for write-time instrumentation
- Importance scoring (LLM-judged 0–1)
- Stability hint classification (fact|preference|decision|rule|project|config|technical|relationship|transient)
- Post-write consolidation analysis (nightly job)

See `PHASE-0-1-CODE-AUDIT.md` in workspace-main for details.
