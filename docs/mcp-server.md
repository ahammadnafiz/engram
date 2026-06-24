# Engram MCP Server — a memory layer for Claude Code

Give Claude Code (or Codex / Gemini CLI) a long-term memory that persists across
conversations. **Engram does the remembering; the host LLM does the reasoning.**

The server exposes **one** tool, `recall_memory`, that runs Engram's
benchmark-proven retrieval pipeline with **no LLM of its own** — free, on-device
embeddings only. Generation never happens here, so there is **no API key and no
cost**. Saving each turn is handled by a Claude Code **Stop hook**.

Source: [`engram.mcp.server`](../src/engram/mcp/server.py).

---

## How it works

```
You ──▶ Claude Code
            │  (decides if memory would help)
            ├──▶ recall_memory(query)  ──▶  Engram: hybrid search + rerank
            │                                + session diversification
            │                                + lineage (superseded values)
            │        ◀── evidence block ◀──  [ACTIVE]/[SUPERSEDED], by date
            ▼
        answers, grounded in the evidence
            │
   (turn ends) ──▶ Stop hook ──▶ saves the exchange verbatim
```

- **Recall is model-driven.** Claude Code decides when a question needs memory
  and calls `recall_memory` itself. It reads the returned evidence and answers.
- **Saving is deterministic.** A Stop hook fires after every turn and stores the
  exchange — you don't rely on the model remembering to save.
- **You hold a kill switch** (see [Controlling capture](#controlling-memory-capture)).

---

## Prerequisites

1. **Postgres + pgvector running**, with `ENGRAM_DATABASE_URL` set. The server
   reads it from the project's `.env` or `~/.engram/.env`. The bundled Docker
   setup already provides a pgvector-enabled Postgres.
2. **Install the package** with the MCP + embedding extras:
   ```bash
   pip install "engram[mcp,sentence-transformers]"
   ```
   This puts an `engram-mcp` command on your PATH. The first run downloads the
   `all-MiniLM-L6-v2` model (~80 MB) once.

---

## Quickstart

From the project where you want memory enabled:

```bash
engram-mcp init      # writes .mcp.json + the Stop hook into this directory
```

`init` wires both configs using **the exact Python interpreter you ran it with**
(`python -m engram.mcp.server`), so there is no PATH guessing — the single most
common reason the server "silently does nothing." It is idempotent and
migration-safe: re-running it, or upgrading from an older path-based setup,
replaces the existing entry instead of stacking a duplicate Stop hook.

Then restart Claude Code in that directory and run `/mcp` to confirm
`engram-memory` is listed. That's the whole setup for the default (STDIO) mode.

> Prefer to wire it by hand, or want the faster daemon? See
> [Run modes](#run-modes) below.

---

## Run modes

### A. STDIO — zero setup (what `init` wires by default)

Claude Code launches the server per session. Recall is fast; **saves use an
in-process fallback that loads the model each turn (~2 s)**. Simple, nothing
extra to run.

The `.mcp.json` that `init` writes:

```json
{
  "mcpServers": {
    "engram-memory": {
      "command": "/abs/path/to/python",
      "args": ["-m", "engram.mcp.server"]
    }
  }
}
```

(`init` fills in the absolute interpreter path for you.)

### B. HTTP daemon — fast (recommended for heavy use)

A persistent process keeps the model **loaded once**. Recall connects over HTTP;
the Stop hook POSTs each turn to the warm process (**~0.2 s**, no reload).

1. Keep the daemon running (macOS `launchd`, auto-start + restart). Use the
   absolute interpreter path (find it with `which python`):

   `~/Library/LaunchAgents/com.engram.mcp.plist`
   ```xml
   <?xml version="1.0" encoding="UTF-8"?>
   <plist version="1.0"><dict>
     <key>Label</key><string>com.engram.mcp</string>
     <key>ProgramArguments</key>
     <array>
       <string>/abs/path/to/python</string>
       <string>-m</string>
       <string>engram.mcp.server</string>
       <string>serve</string>
     </array>
     <key>RunAtLoad</key><true/>
     <key>KeepAlive</key><true/>
     <key>StandardErrorPath</key><string>/Users/you/.engram/daemon.log</string>
   </dict></plist>
   ```
   ```bash
   launchctl load ~/Library/LaunchAgents/com.engram.mcp.plist
   curl -s http://127.0.0.1:8765/health    # {"status":"ok",...}
   ```
   Quick alternative without launchd:
   ```bash
   nohup engram-mcp serve >> ~/.engram/daemon.log 2>&1 &
   ```

2. Point `.mcp.json` at the daemon instead of `command`/`args`:
   ```json
   { "mcpServers": { "engram-memory": { "url": "http://127.0.0.1:8765/mcp" } } }
   ```

The **Stop hook is identical in both modes** — it tries the daemon first and
falls back to an in-process save, so a turn is never silently dropped.

---

## The Stop hook

`init` writes this into `.claude/settings.json`:

```json
{
  "hooks": {
    "Stop": [
      { "hooks": [
        { "type": "command",
          "command": "/abs/path/to/python -m engram.mcp.server hook" }
      ] }
    ]
  }
}
```

Each time Claude Code finishes a turn, the hook reads the transcript, extracts
the latest user message + final assistant message, and saves them — unless
capture is disabled.

---

## Controlling memory capture

Type these anywhere in a message to Claude Code:

| Token | Effect |
|---|---|
| `#nomem` | Skip saving **just this turn** |
| `#mem-off` | **Pause** saving until re-enabled (persists across turns) |
| `#mem-on` | **Resume** saving |

Pause state lives in `~/.engram/paused`. The turn that contains a control token
is itself never stored.

---

## Security: keep it on localhost

The daemon has **no authentication**. `ENGRAM_MCP_HOST` defaults to `127.0.0.1`
and the `/ingest` endpoint accepts any POST. That is correct and safe for a
single user on one machine.

**Do not bind it to `0.0.0.0` or expose the port.** There is no auth layer, so a
networked daemon is an unauthenticated read/write of everyone's memory. Sharing
one memory across people/machines is a hosted-service design, not a config flag —
this server is built for self-hosting, one namespace per person.

---

## Scoping: keeping memories separate

Memory is **hard-isolated by `(agent_id, user_id)`** — a recall scoped to one
namespace never sees another's rows, even though everything is in one database.
Set the namespace with env vars:

| Variable | Default | Purpose |
|---|---|---|
| `ENGRAM_CHATBOT_AGENT_ID` | `engram-chatbot` | Primary namespace knob |
| `ENGRAM_CHATBOT_USER_ID` | `default-user` | Secondary scope |

- **One memory everywhere (personal second brain):** run `engram-mcp init` and
  then move the entries into **user scope** (`~/.claude/settings.json` and a
  user-scoped MCP entry), leaving the default namespace.
- **Per-project memory:** set a different `ENGRAM_CHATBOT_AGENT_ID` in each
  project's `.env` (and, for the daemon, run one per project on its own
  `ENGRAM_MCP_PORT`). The Stop hook and recall **must share the namespace**, or
  you'd save into one bucket and read from another.

---

## Admin CLI

```bash
engram-mcp list --limit 20                 # recent memories
engram-mcp forget --memory-id mem_abc123   # delete one
echo '{"user":"...","assistant":"..."}' | engram-mcp save   # manual save
```

These are intentionally **not** exposed as MCP tools — the model only ever sees
`recall_memory`.

---

## Configuration reference

All optional; sensible defaults shown.

| Variable | Default | Meaning |
|---|---|---|
| `ENGRAM_DATABASE_URL` | from `.env` | Postgres DSN |
| `ENGRAM_MCP_HOST` / `ENGRAM_MCP_PORT` | `127.0.0.1` / `8765` | Daemon address |
| `ENGRAM_MCP_URL` | `http://HOST:PORT` | Where the hook POSTs |
| `ENGRAM_MCP_TIMEOUT` | `5` | Hook→daemon POST timeout (s) |
| `ENGRAM_MCP_SEARCH_LIMIT` | `60` | Evidence rows after diversification |
| `ENGRAM_MCP_CANDIDATE_LIMIT` | `100` | Candidate pool before rerank |
| `ENGRAM_MCP_MAX_PER_SESSION` | `4` | Max rows kept per past session |
| `ENGRAM_MCP_RERANK` | `true` | Cross-encoder rerank on/off |
| `ENGRAM_MCP_PAUSE_FILE` | `~/.engram/paused` | Pause marker path |

---

## Verify it works

1. Restart Claude Code in this repo, then run `/mcp` — `engram-memory` should be
   listed with the tool `recall_memory`.
2. Tell it something (`my favorite editor is neovim`), let the turn finish.
3. New turn: *"what's my favorite editor?"* — it should call `recall_memory` and
   answer `neovim`.
4. Confirm it persisted: `engram-mcp list`.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `/mcp` shows nothing | Re-run `engram-mcp init` so the config uses an absolute interpreter path; a bare command may not be on the subprocess PATH. Check the interpreter has `fastmcp` + `engram`. |
| Recall returns `(no matching memory)` | Nothing saved yet, or wrong namespace (`agent_id`/`user_id` mismatch between hook and recall). |
| Saves don't appear | DB unreachable, or a `#mem-off` pause is still active (delete `~/.engram/paused`). |
| Saves feel slow (~2 s) | You're on STDIO mode (no daemon). Start the daemon (mode B) for ~0.2 s saves. |
| `password authentication failed` | `ENGRAM_DATABASE_URL` in `.env` doesn't match the running Postgres credentials. |
| `command not found: engram-mcp` | Install the extra: `pip install "engram[mcp,sentence-transformers]"`, into the interpreter your configs point at. |
