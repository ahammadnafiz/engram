#!/usr/bin/env python3
"""Engram MCP server — a memory layer for Claude Code / Codex / Gemini CLI.

Engram does the remembering; the host LLM does the reasoning. This process
exposes ONE model-facing tool, ``recall_memory``, that runs the benchmark-proven
retrieval pipeline with NO LLM of its own (free, on-device embeddings only):

    hybrid search (vector + full-text, RRF-fused) + cross-encoder rerank
    + session diversification + lineage expansion (superseded predecessors)
    -> one date-grouped, [ACTIVE]/[SUPERSEDED]-tagged evidence block.

The host (Claude Code) decides when to call it, reads the evidence, and writes
the answer. Generation never happens here, so there is no API key and no cost.

Two ways to run it:

  * HTTP daemon (recommended) — model loaded ONCE, fast saves:
        python engram_mcp_server.py serve            # leave running
    Point Claude Code at  http://127.0.0.1:8765/mcp  and the Stop hook POSTs
    each turn to the same warm process (~50 ms, no model reload).

  * STDIO (zero-setup) — Claude Code launches it per session:
        python engram_mcp_server.py                  # in .mcp.json command/args
    No daemon; the Stop hook falls back to an in-process save (~2 s, loads the
    model each turn). Slower, but works with nothing extra running.

Ingestion is deterministic: a Claude Code ``Stop`` hook runs
``python engram_mcp_server.py hook`` after every turn. It tries the daemon
first and falls back to a direct save, so a turn is never silently dropped.

Memory-capture control (typed anywhere in your message to Claude Code):
    #nomem    skip saving just this turn
    #mem-off  pause saving until re-enabled (persists across turns)
    #mem-on   resume saving

Admin CLI:  list | forget --memory-id <id> | save (stdin {"user","assistant"})

Stack: ENGRAM_DATABASE_URL + on-device sentence-transformers (all-MiniLM-L6-v2).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

with contextlib.suppress(ImportError):
    from dotenv import load_dotenv

    # Load config from standard locations so pip-installed users (who may not
    # have the repo checked out) still get ENGRAM_DATABASE_URL. Repo users keep
    # working: load_dotenv() with no path walks up from the CWD and finds the
    # repo's .env when Claude Code runs the hook from the project root.
    load_dotenv(Path.home() / ".engram" / ".env", override=False)
    load_dotenv(override=False)

if TYPE_CHECKING:
    from fastmcp import FastMCP
    from starlette.requests import Request

# --- identity + retrieval knobs (mirror the benchmark/chatbot defaults) ------
AGENT_ID = os.environ.get("ENGRAM_CHATBOT_AGENT_ID", "engram-chatbot")
USER_ID = os.environ.get("ENGRAM_CHATBOT_USER_ID", "default-user")
# Tuned for Chonkie-chunked ingestion: each stored item is already bounded at
# CHUNK_TOKENS, so SEARCH_LIMIT x CHUNK_TOKENS ≈ total tokens returned to the
# host LLM.  30 x 500 ≈ 15000 tokens — wider recall window for richer context.
SEARCH_LIMIT = int(os.environ.get("ENGRAM_MCP_SEARCH_LIMIT", "30"))
MAX_PER_SESSION = int(os.environ.get("ENGRAM_MCP_MAX_PER_SESSION", "3"))
CANDIDATE_LIMIT = int(os.environ.get("ENGRAM_MCP_CANDIDATE_LIMIT", "120"))
RERANK = os.environ.get("ENGRAM_MCP_RERANK", "true").lower() != "false"

# --- daemon address (shared by `serve` and the Stop hook's fast path) --------
HOST = os.environ.get("ENGRAM_MCP_HOST", "127.0.0.1")
PORT = int(os.environ.get("ENGRAM_MCP_PORT", "8765"))
DAEMON_URL = os.environ.get("ENGRAM_MCP_URL", f"http://{HOST}:{PORT}")
DAEMON_TIMEOUT = float(os.environ.get("ENGRAM_MCP_TIMEOUT", "5"))

# --- memory-capture control ---------------------------------------------------
PAUSE_FILE = Path(
    os.environ.get("ENGRAM_MCP_PAUSE_FILE", str(Path.home() / ".engram" / "paused"))
)
TOKEN_SKIP = "#nomem"
TOKEN_OFF = "#mem-off"
TOKEN_ON = "#mem-on"

# --- Chonkie chunking: token budget per ingested chunk -----------------------
# Long user/assistant turns are split into boundary-aware Chonkie chunks before
# storage.  Each chunk is stored as its own memory row so retrieval can surface
# the most relevant portion without crude mid-sentence truncation.
CHUNK_TOKENS = int(os.environ.get("ENGRAM_MCP_CHUNK_TOKENS", "500"))

_RECALL_HINT = "recall_memory"  # drop our own lookups so recall never feeds itself
_SYSREMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)
_CAVEAT_RE = re.compile(r"<local-command-caveat>.*?</local-command-caveat>", re.DOTALL)
_WRAPPER_TAGS = (
    "<command-name>",
    "</command-name>",
    "<command-message>",
    "</command-message>",
    "<command-args>",
    "</command-args>",
    "<command-stdout>",
    "</command-stdout>",
    "<local-command-stdout>",
    "</local-command-stdout>",
)


# =============================================================================
# Engram connection (embeddings only — no LLM, no API key)
# =============================================================================
async def _connect() -> Any:
    """Connect an Engram client configured for free on-device retrieval."""
    from engram import Engram
    from engram.core.config import get_settings

    settings = get_settings().model_copy(
        update={
            "embedding_provider": "sentence-transformers",
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dimension": 384,
            "llm_provider": None,  # pure memory layer: no generation, no cost
            "near_duplicate_threshold": 1.0,  # keep every turn verbatim
            "allow_embedding_dimension_change": True,
        }
    )
    engram = Engram(settings=settings)
    await engram.connect()
    return engram


# =============================================================================
# Retrieval — ported from the chatbot/benchmark pipeline, minus the LLM surface
# =============================================================================
def _to_human_date(date_str: str) -> str:
    if not date_str:
        return "Unknown Date"
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(date_str[:19], fmt).strftime("%B %d, %Y")
        except ValueError:
            continue
    return date_str


def diversify_by_session(
    results: list[Any], *, limit: int, max_per_session: int, rerank: bool
) -> list[Any]:
    """Round-robin a candidate pool across sessions so one session can't
    monopolize the evidence budget. When reranking, candidate order is already
    a relevance ranking, so the user-turn-first nudge is dropped."""
    if not results:
        return results

    def role_bias(r: Any) -> float:
        if rerank:
            return 0.0
        role = str(r.memory.metadata.get("turn_role", "")).lower()
        if role == "user":
            return -1.0
        if role in ("assistant", "system", "tool"):
            return 1.0
        return 0.0

    def group_key(r: Any) -> str:
        sid = r.memory.metadata.get("original_session_id")
        return str(sid) if sid is not None else r.memory.memory_id

    ranked = [
        r
        for _, r in sorted(
            enumerate(results), key=lambda it: (it[0] + role_bias(it[1]), it[0])
        )
    ]
    groups: dict[str, list[Any]] = {}
    for r in ranked:
        groups.setdefault(group_key(r), []).append(r)

    selected: list[Any] = []
    seen: set[str] = set()
    for depth in range(max_per_session):
        for group in groups.values():
            if depth < len(group):
                r = group[depth]
                selected.append(r)
                seen.add(r.memory.memory_id)
                if len(selected) >= limit:
                    return selected
    for r in ranked:
        if r.memory.memory_id not in seen:
            selected.append(r)
            if len(selected) >= limit:
                break
    return selected


def _build_evidence_block(
    search_results: list[Any],
    lineage_superseded: list[Any],
) -> str:
    """Assemble the evidence block: superseded predecessors first, then the
    hybrid hits grouped by date and tagged [ACTIVE]/[SUPERSEDED].

    No truncation is applied here — stored memories are already Chonkie-chunked
    to CHUNK_TOKENS during ingestion, so each item is already bounded."""
    lines: list[str] = []
    seen: set[str] = set()

    for mem in lineage_superseded:
        if mem.memory_id in seen:
            continue
        seen.add(mem.memory_id)
        when = mem.superseded_at or mem.valid_to or mem.created_at
        stamp = when.date().isoformat() if when else "unknown"
        lines.append(f"SUPERSEDED (until {stamp}): {mem.fact or mem.content}")

    lines.append("\n## RETRIEVED MEMORIES (hybrid search)")
    current_date = None
    for result in search_results:
        mem = result.memory
        if mem.memory_id in seen:
            continue
        seen.add(mem.memory_id)
        mem_date = mem.metadata.get("chat_date", "Unknown Date")
        if mem_date != current_date:
            lines.append(f"\n--- {_to_human_date(mem_date)} ---")
            current_date = mem_date
        status = getattr(mem, "status", None) or mem.metadata.get("status", "active")
        tag = "[SUPERSEDED]" if status == "superseded" else "[ACTIVE]"
        lines.append(f"- {tag} {mem.content}")

    return "\n".join(lines)


async def _search_candidates(
    engram: Any, queries: list[str], rerank: bool
) -> list[Any]:
    """Hybrid search for one or many queries. Multiple queries (supplied by the
    host LLM, e.g. two anchors of a 'days between X and Y' question) are fused
    with Reciprocal Rank Fusion — the high-recall multi-query path, without
    needing an LLM in the server."""
    if len(queries) == 1:
        hits: list[Any] = await engram.search(
            query=queries[0],
            agent_id=AGENT_ID,
            user_id=USER_ID,
            limit=CANDIDATE_LIMIT,
            mode="hybrid",
            rerank=rerank,
            include_superseded=True,
        )
        return hits

    rrf_k = 60
    fused: dict[str, float] = {}
    representative: dict[str, Any] = {}
    for q in queries:
        results = await engram.search(
            query=q,
            agent_id=AGENT_ID,
            user_id=USER_ID,
            limit=CANDIDATE_LIMIT,
            mode="hybrid",
            rerank=False,
            include_superseded=True,
        )
        for rank, r in enumerate(results):
            mid = r.memory.memory_id
            fused[mid] = fused.get(mid, 0.0) + 1.0 / (rrf_k + rank + 1)
            if mid not in representative or r.score > representative[mid].score:
                representative[mid] = r
    return [
        representative[mid]
        for mid, _ in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    ]


async def _retrieve(engram: Any, queries: list[str]) -> str:
    """Full LLM-free retrieval: search/fuse -> diversify -> lineage -> block."""
    candidates = await _search_candidates(engram, queries, RERANK)
    search_results = diversify_by_session(
        candidates, limit=SEARCH_LIMIT, max_per_session=MAX_PER_SESSION, rerank=RERANK
    )
    search_results.sort(key=lambda r: r.memory.metadata.get("chat_date", ""))

    lineage_superseded: list[Any] = []
    seen_lineages: set[str] = set()
    for r in search_results:
        mem = r.memory
        lid = getattr(mem, "lineage_id", None)
        status = getattr(mem, "status", None) or mem.metadata.get("status", "active")
        if (
            status != "superseded"
            and lid
            and lid != mem.memory_id
            and lid not in seen_lineages
        ):
            seen_lineages.add(lid)
            with contextlib.suppress(Exception):
                lineage = await engram.get_lineage(mem.memory_id)
                lineage_superseded.extend(
                    m
                    for m in lineage.memories
                    if getattr(m, "status", None) == "superseded"
                )

    block = _build_evidence_block(search_results, lineage_superseded)
    return block if search_results or lineage_superseded else "(no matching memory)"


# =============================================================================
# Ingestion — verbatim, date-anchored, session-continuous (mirrors the chatbot)
# =============================================================================
async def _session_id(engram: Any) -> str:
    """Resume the active/paused chatbot task's session, or start a new one, so
    every stored turn shares one session_id (diversification depends on it)."""
    tasks = await engram.list_tasks(
        agent_id=AGENT_ID, user_id=USER_ID, status=["active", "paused"], limit=1
    )
    if tasks and tasks[0].session_id:
        return str(tasks[0].session_id)

    session_cm = engram.session(
        AGENT_ID, user_id=USER_ID, metadata={"application": "engram-mcp"}
    )
    session = await session_cm.__aenter__()
    await engram.start_task(
        "Engram MCP memory layer",
        AGENT_ID,
        user_id=USER_ID,
        session_id=session.session_id,
        metadata={"application": "engram-mcp"},
    )
    return str(session.session_id)


def _chunk_text(text: str) -> list[str]:
    """Split ``text`` into Chonkie chunks.  Falls back to a single-item list
    when chonkie is not installed or chunking fails, so ingestion is never
    blocked by a missing optional dependency."""
    try:
        from engram.chunking import chonkie_recursive_spans

        spans = chonkie_recursive_spans(text, max_chunk_tokens=CHUNK_TOKENS)
        if spans:
            return [body for (_, body, _, _) in spans]
    except Exception:
        pass
    return [text]


async def save_turn(engram: Any, user_message: str, assistant_response: str) -> int:
    """Store one exchange as date-anchored episodic rows via add_batch.

    Long texts are split into boundary-aware Chonkie chunks; each chunk
    becomes its own memory row with a ``chunk_index`` / ``chunk_count`` tag.
    Short texts (or when chonkie is unavailable) produce a single row as
    before.  No content is ever truncated and discarded."""
    sid = await _session_id(engram)
    chat_date = datetime.now().date().isoformat()
    human_date = _to_human_date(chat_date)
    # Monotonic, O(1) ordering marker — retrieval orders by chat_date /
    # created_at, not this, so a millisecond stamp is sufficient.
    turn = int(datetime.now().timestamp() * 1000)

    def _rows_for(role_label: str, role: str, text: str) -> list[dict[str, Any]]:
        chunks = _chunk_text(text)
        total = len(chunks)
        rows: list[dict[str, Any]] = []
        for idx, body in enumerate(chunks):
            content = f"[{human_date}] {role_label}: {body}"
            meta: dict[str, Any] = {
                "source": "engram-mcp",
                "original_session_id": sid,
                "chat_date": chat_date,
                "turn_index": turn,
                "turn_role": role,
            }
            if total > 1:
                meta["chunk_index"] = idx
                meta["chunk_count"] = total
            rows.append(
                {
                    "content": content,
                    "main_content": content,
                    "agent_id": AGENT_ID,
                    "user_id": USER_ID,
                    "memory_type": "episodic",
                    "metadata": meta,
                }
            )
        return rows

    rows = _rows_for("USER", "user", user_message)
    if assistant_response.strip():
        rows.extend(_rows_for("ASSISTANT", "assistant", assistant_response))
    memories = await engram.add_batch(rows)
    return len(memories)


# =============================================================================
# MCP server — built lazily so `hook`/CLI paths never import fastmcp or torch
# =============================================================================
_INSTRUCTIONS = """Engram is the user's long-term memory — everything they have told you
across past conversations. You are their "second brain": when you answer from it, talk
like a thoughtful friend with perfect recall, not like a database readout.

WHEN TO CALL recall_memory
- Whenever answering could benefit from something the user told you before:
  preferences, facts, decisions, plans, history, "what did I say about X",
  "what's my ...", "when is my ...", "remind me ...".
- Anything that depends on continuity from earlier sessions. When unsure whether
  you know something about the user, call it rather than guess.
- Skip it only for genuinely self-contained questions (general knowledge, the
  code in front of you) that owe nothing to the user's own history.
- Pass extra_queries when one phrasing won't capture everything — e.g. the two
  anchors of a "how long between X and Y" question, or a fact that might be
  stored under different words.

READING THE EVIDENCE IT RETURNS
The block is your source of truth about the user. Ground every claim in it.
- [ACTIVE] memories are current. [SUPERSEDED] are OLD values that were later
  changed — use them ONLY to answer "what was it before / originally". When a
  value changed, it's natural to mention both ("it's X now — it used to be Y").
- Most recent wins for conflicting values of the SAME fact. Memories about
  different people, places, or contexts are not in conflict — don't merge them.
- Match the exact thing asked. If they ask about one specific entity, variant,
  or role and memory only mentions a different one, don't treat them as the
  same — say you don't have that exact detail.
- Do the math when the memories hold the numbers (ages, prices, durations, date
  differences) instead of refusing, even when facts are scattered across several
  memories. Compute relative time expressions ("next week", "in 3 days") against
  today's date.
- Respect what the user avoids — allergies, dislikes, hard constraints. Never
  suggest something they've said they want to avoid, not even as a fallback.
- For "what do you remember about me" questions, synthesize the durable picture
  (who they are, what they're working on, their preferences) in a few natural
  sentences rather than dumping raw lines.

NEVER invent names, numbers, dates, or details that aren't in the evidence. If
it genuinely isn't there, say you don't have it yet."""

_engram: Any = None
_connect_lock: asyncio.Lock | None = None


async def _ensure_connected() -> Any:
    """Connect lazily on first use so the server can respond to MCP initialize
    immediately, without blocking on model/DB startup (~3-15 s)."""
    global _engram, _connect_lock
    if _engram is not None:
        return _engram
    if _connect_lock is None:
        _connect_lock = asyncio.Lock()
    async with _connect_lock:
        if _engram is None:
            _engram = await _connect()
            with contextlib.suppress(Exception):
                await _engram.warmup()
    return _engram


@contextlib.asynccontextmanager
async def _lifespan(_server: FastMCP):  # type: ignore[no-untyped-def]
    global _connect_lock
    _connect_lock = asyncio.Lock()
    try:
        yield
    finally:
        global _engram
        if _engram is not None:
            with contextlib.suppress(Exception):
                await _engram.close()
            _engram = None


async def recall_memory(query: str, extra_queries: list[str] | None = None) -> str:
    """Retrieve everything Engram remembers that is relevant to a query.

    Runs hybrid search (semantic + keyword) with cross-encoder reranking,
    diversifies across past sessions, and expands lineage so superseded values
    are visible. Returns a date-grouped, [ACTIVE]/[SUPERSEDED]-tagged evidence
    block to ground your answer.

    Args:
        query: What to look up, in natural language.
        extra_queries: Optional extra phrasings or sub-questions (e.g. the two
            events of a "how long between X and Y" question). Results are fused
            for higher recall. Use it when one query won't capture everything.
    """
    engram = await _ensure_connected()
    queries = [query, *(q for q in (extra_queries or []) if q.strip())]
    return await _retrieve(engram, queries)


def build_server() -> FastMCP:
    """Construct the FastMCP server. Imports fastmcp lazily so the hook/CLI
    paths stay light. Exposes recall_memory (MCP tool) plus /ingest and /health
    HTTP routes used by the Stop hook's fast path when running as a daemon."""
    from fastmcp import FastMCP
    from mcp.types import ToolAnnotations
    from starlette.responses import JSONResponse

    mcp = FastMCP("engram-memory", instructions=_INSTRUCTIONS, lifespan=_lifespan)
    mcp.tool(
        annotations=ToolAnnotations(
            title="Recall from memory", readOnlyHint=True, openWorldHint=False
        )
    )(recall_memory)

    @mcp.custom_route("/ingest", methods=["POST"])
    async def ingest(request: Request) -> JSONResponse:
        """Fast warm-process save used by the Stop hook. The daemon owns the
        namespace, so ingest and recall can never target different buckets."""
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            return JSONResponse({"error": "bad json"}, status_code=400)
        user = str(body.get("user", ""))
        assistant = str(body.get("assistant", ""))
        if not user.strip():
            return JSONResponse({"saved": 0})
        try:
            eng = await _ensure_connected()
            n = await save_turn(eng, user, assistant)
        except Exception as exc:  # surface, but never crash the daemon
            return JSONResponse({"error": str(exc)}, status_code=500)
        return JSONResponse({"saved": n})

    @mcp.custom_route("/health", methods=["GET"])
    async def health(_request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok" if _engram is not None else "starting",
                "agent_id": AGENT_ID,
                "user_id": USER_ID,
            }
        )

    return mcp


# =============================================================================
# Stop-hook handler + CLI
# =============================================================================
def _capture_disabled_for(user_text: str) -> bool:
    """Apply memory-capture control tokens; return True to skip saving."""
    lowered = user_text.lower()
    if TOKEN_ON in lowered:
        PAUSE_FILE.unlink(missing_ok=True)
        return True  # the control turn itself is not stored
    if TOKEN_OFF in lowered:
        PAUSE_FILE.parent.mkdir(parents=True, exist_ok=True)
        PAUSE_FILE.touch()
        return True
    if TOKEN_SKIP in lowered:
        return True
    return PAUSE_FILE.exists()


def _clean_text(text: str) -> str:
    """Strip harness scaffolding (system reminders, local-command caveats, slash-command
    wrappers) so we store what was actually said, not host boilerplate."""
    text = _SYSREMINDER_RE.sub("", text)
    text = _CAVEAT_RE.sub("", text)
    for tag in _WRAPPER_TAGS:
        text = text.replace(tag, "")
    return text.strip()


def _format_args(inp: Any) -> str:
    """Render a tool_use input dict compactly: each value clipped to one line."""
    if not isinstance(inp, dict):
        return str(inp)[:80]
    return ", ".join(f"{k}={str(v)[:80]}" for k, v in inp.items())


def _result_text(content: Any) -> str:
    """Flatten a tool_result's content (str or text-block list).

    Full content is preserved here; Chonkie chunking in save_turn() handles
    boundary-aware splitting before anything goes to the database."""
    if isinstance(content, list):
        content = "\n".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content).strip()


def _iter_events(transcript_path: str) -> list[tuple[str, Any]]:
    """Flatten a Claude Code transcript into ordered turn events. Kinds:
    ``user_text``, ``assistant_text``, ``thinking``, ``tool_use``, ``tool_result``.
    Only user/assistant messages are read; ``system``/metadata entries are ignored."""
    events: list[tuple[str, Any]] = []
    with Path(transcript_path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("type") not in ("user", "assistant"):
                continue
            message = entry.get("message") or {}
            role = message.get("role")
            content = message.get("content", "")
            if isinstance(content, str):
                if role == "user":
                    events.append(("user_text", content))
                continue
            if not isinstance(content, list):
                continue
            for b in content:
                if not isinstance(b, dict):
                    continue
                bt = b.get("type")
                if bt == "text":
                    events.append(
                        (
                            "user_text" if role == "user" else "assistant_text",
                            b.get("text", ""),
                        )
                    )
                elif bt == "thinking":
                    events.append(("thinking", b.get("thinking", "")))
                elif bt == "tool_use":
                    events.append(("tool_use", b))
                elif bt == "tool_result":
                    events.append(("tool_result", b))
    return events


def _extract_last_turn(transcript_path: str) -> tuple[str, str]:
    """Return (user_prompt, activity_log) for the most recent turn. The activity log
    is the full end-to-end turn — intermediate replies, thinking, tool calls and their
    (truncated) results — minus harness scaffolding and recall_memory's own output
    (which would otherwise feed back into the store on the next turn)."""
    events = _iter_events(transcript_path)

    # The turn begins at the last user message carrying a real (non-scaffolding) prompt.
    start = None
    user_prompt = ""
    for i, (kind, payload) in enumerate(events):
        if kind == "user_text":
            cleaned = _clean_text(payload)
            if cleaned:
                start, user_prompt = i, cleaned
    if start is None:
        return "", ""

    tool_names: dict[str, str] = {}
    parts: list[str] = []
    for kind, payload in events[start + 1 :]:
        if kind == "assistant_text":
            text = _clean_text(payload)
            if text:
                parts.append(text)
        elif kind == "thinking":
            text = payload.strip()
            if text:
                parts.append(f"(thinking) {text}")
        elif kind == "tool_use":
            name = payload.get("name", "tool")
            tool_names[payload.get("id", "")] = name
            if _RECALL_HINT in name:
                continue  # don't log our own memory lookups
            parts.append(f"→ {name}({_format_args(payload.get('input'))})")
        elif kind == "tool_result":
            if _RECALL_HINT in tool_names.get(payload.get("tool_use_id", ""), ""):
                continue  # recall_memory output → never re-ingest it
            body = _result_text(payload.get("content"))
            if body:
                parts.append(f"  ⤷ {body}")
    return user_prompt, "\n".join(parts)


def _post_to_daemon(user: str, assistant: str) -> bool:
    """Fast path: POST the turn to a running daemon. stdlib only — no model
    load, no fastmcp/torch import. Returns False on any failure so the caller
    can fall back to a direct save."""
    data = json.dumps({"user": user, "assistant": assistant}).encode()
    req = urllib.request.Request(
        f"{DAEMON_URL}/ingest",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=DAEMON_TIMEOUT) as resp:
            return bool(resp.status == 200)
    except (urllib.error.URLError, OSError, TimeoutError, ValueError):
        return False


def _direct_save(user: str, assistant: str) -> None:
    """Reliable fallback: connect, save, close. Loads the model (~2 s) — only
    runs when no daemon is reachable, so saving never depends on the daemon."""

    async def _run() -> None:
        engram = await _connect()
        try:
            await save_turn(engram, user, assistant)
        finally:
            await engram.close()

    with contextlib.suppress(Exception):
        asyncio.run(_run())


def _handle_hook() -> None:
    """Claude Code Stop hook: save the just-finished turn unless disabled.
    Tries the warm daemon first, falls back to a direct save. Always exits 0."""
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return  # nothing to do; never break the host
    if payload.get("stop_hook_active"):
        return  # avoid re-trigger loops
    transcript_path = payload.get("transcript_path")
    if not transcript_path or not Path(transcript_path).exists():
        return

    user_text, assistant_text = _extract_last_turn(transcript_path)
    if not user_text:
        return
    if _capture_disabled_for(user_text):
        return

    if not _post_to_daemon(user_text, assistant_text):
        _direct_save(user_text, assistant_text)


def _cli_save() -> None:
    payload = json.load(sys.stdin)

    async def _run() -> None:
        engram = await _connect()
        try:
            n = await save_turn(engram, payload["user"], payload.get("assistant", ""))
            print(f"stored {n} memories")
        finally:
            await engram.close()

    asyncio.run(_run())


def _cli_list(limit: int) -> None:
    async def _run() -> None:
        engram = await _connect()
        try:
            for m in await engram.list_recent(AGENT_ID, user_id=USER_ID, limit=limit):
                print(
                    f"{m.memory_id[:16]}  [{m.memory_type} {m.status}]  {m.content[:100]}"
                )
        finally:
            await engram.close()

    asyncio.run(_run())


def _cli_forget(memory_id: str) -> None:
    async def _run() -> None:
        engram = await _connect()
        try:
            print("deleted" if await engram.forget(memory_id) else "not found")
        finally:
            await engram.close()

    asyncio.run(_run())


def _load_json(path: Path) -> dict[str, Any]:
    if path.exists():
        with contextlib.suppress(json.JSONDecodeError, ValueError, OSError):
            return dict(json.loads(path.read_text(encoding="utf-8")))
    return {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _references_engram_server(command: str) -> bool:
    """True if a hook command runs this server (current module or legacy shim),
    so init can replace it instead of stacking a duplicate Stop hook."""
    return "engram.mcp.server" in command or "engram_mcp_server" in command


def _cli_init() -> None:
    """Wire Claude Code config into the CWD: register recall_memory in .mcp.json
    and the capture Stop hook in .claude/settings.json — using the exact
    interpreter running this command (no PATH guessing, the #1 setup failure).
    Idempotent and migration-safe: any prior engram hook is replaced, not stacked."""
    py = sys.executable
    module = "engram.mcp.server"
    project = Path.cwd()

    mcp_path = project / ".mcp.json"
    mcp_cfg = _load_json(mcp_path)
    mcp_cfg.setdefault("mcpServers", {})["engram-memory"] = {
        "command": py,
        "args": ["-m", module],
    }
    _write_json(mcp_path, mcp_cfg)

    settings_path = project / ".claude" / "settings.json"
    settings = _load_json(settings_path)
    hook_cmd = f"{py} -m {module} hook"
    hooks = settings.setdefault("hooks", {})
    stop_groups = hooks.get("Stop", [])
    rebuilt: list[Any] = []
    for group in stop_groups:
        kept = [
            h
            for h in group.get("hooks", [])
            if not _references_engram_server(str(h.get("command", "")))
        ]
        if kept:
            rebuilt.append({**group, "hooks": kept})
    rebuilt.append({"hooks": [{"type": "command", "command": hook_cmd}]})
    hooks["Stop"] = rebuilt
    _write_json(settings_path, settings)

    print(f"Wrote {mcp_path}")
    print(f"Wrote {settings_path}")
    print(f"Interpreter: {py}")
    if not os.environ.get("ENGRAM_DATABASE_URL"):
        print(
            "\nNote: ENGRAM_DATABASE_URL is not set. Put it in this project's "
            ".env or ~/.engram/.env before recall/capture will work."
        )
    print("\nRestart Claude Code here, then run /mcp to confirm 'engram-memory'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Engram MCP memory server + admin CLI")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("serve", help="run the HTTP daemon (model loaded once)")
    sub.add_parser(
        "hook", help="Claude Code Stop-hook handler (reads payload on stdin)"
    )
    sub.add_parser("save", help='manual save: stdin JSON {"user", "assistant"}')
    list_p = sub.add_parser("list", help="list recent memories")
    list_p.add_argument("--limit", type=int, default=20)
    forget_p = sub.add_parser("forget", help="delete one memory")
    forget_p.add_argument("--memory-id", required=True)
    sub.add_parser(
        "init", help="write Claude Code config (.mcp.json + Stop hook) into the CWD"
    )
    args = parser.parse_args()

    if args.command == "init":
        _cli_init()
    elif args.command == "hook":
        _handle_hook()
    elif args.command == "save":
        _cli_save()
    elif args.command == "list":
        _cli_list(args.limit)
    elif args.command == "forget":
        _cli_forget(args.memory_id)
    elif args.command == "serve":
        build_server().run(transport="http", host=HOST, port=PORT)
    else:
        build_server().run()  # default: MCP server over stdio


if __name__ == "__main__":
    main()
