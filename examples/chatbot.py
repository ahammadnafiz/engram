#!/usr/bin/env python3
"""Real Engram-backed chatbot — the benchmark retrieval pipeline, live.

This chatbot runs the EXACT pipeline proven in `benchmark/` (LongMemEval,
LoCoMo, BEAM), end to end:

  1. INGEST  — every turn is stored verbatim as a date-anchored episodic
     memory via `add_batch()`. On-device embeddings only: no LLM extraction,
     no cost, no destructive supersession at write time. (We deliberately do
     NOT use `add_conversation()` here: with the raw turns co-located, its
     extractor reads every fact as already-present and NOOPs it, so it adds
     ~2x LLM cost per turn for no lineage benefit — the floor + composer
     answers temporal/overwrite questions correctly on its own.)

  2. RETRIEVE — 3-surface evidence gathering, per turn:
       a) search(mode="hybrid", rerank=True) — vector + full-text, RRF-fused,
          cross-encoder reranked, session-diversified.
       b) recall(compose_answer=False) — structured current/previous/conflict
          lineage evidence.
       c) get_lineage() — superseded predecessors of retrieved active facts.
       (Graph traversal is omitted: add_batch() ingest creates no edges.)

  3. GENERATE — one composer LLM call answers from the assembled evidence
     block, using a warm "second brain" prompt (conversational, grounded).

Stack defaults: free on-device sentence-transformers embeddings
(`all-MiniLM-L6-v2`, 384-d) + Google `gemini-3.1-flash-lite` for composition.
Override with the standard ENGRAM_* environment variables.

Run (Gemini):
    export ENGRAM_DATABASE_URL=postgresql://engram:engram_secret@localhost:5432/engram
    export ENGRAM_GEMINI_API_KEY=...   # or GEMINI_API_KEY
    python examples/chatbot.py

Run (Claude):
    export ENGRAM_DATABASE_URL=postgresql://engram:engram_secret@localhost:5432/engram
    export ENGRAM_ANTHROPIC_API_KEY=...   # or ANTHROPIC_API_KEY
    ENGRAM_LLM_PROVIDER=anthropic python examples/chatbot.py

Switch model at runtime with /model gemini or /model claude.

Non-interactive checks:
    python examples/chatbot.py --once "What do you remember about me?"
    python examples/chatbot.py --demo
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import shutil
import sys
import textwrap
from datetime import date, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Rich — optional but strongly recommended for beautiful terminal output.
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.theme import Theme

    _ENGRAM_THEME = Theme(
        {
            "engram.accent": "cyan",
            "engram.dim": "dim",
            "engram.good": "green",
            "engram.warn": "yellow",
            "engram.bad": "red bold",
            "engram.heading": "bold cyan",
            "engram.key": "dim cyan",
            "engram.value": "",
            "engram.cmd": "bold cyan",
            "engram.cmd_desc": "dim",
            "engram.prompt_you": "bold cyan",
            "engram.prompt_arrow": "dim",
        }
    )
    console = Console(theme=_ENGRAM_THEME, highlight=False)
    HAS_RICH = True
except ImportError:  # pragma: no cover
    HAS_RICH = False
    console = None  # type: ignore[assignment]

from engram.core.exceptions import DatabaseConnectionError

gemini_api_key_alias = os.environ.get("GEMINI_API_KEY")
anthropic_api_key_alias = os.environ.get("ANTHROPIC_API_KEY")
load_dotenv(Path(__file__).parent.parent / ".env", override=False)
gemini_api_key_alias = os.environ.get("GEMINI_API_KEY") or gemini_api_key_alias
anthropic_api_key_alias = os.environ.get("ANTHROPIC_API_KEY") or anthropic_api_key_alias

# Map the bare convention keys onto Engram's namespaced variable.
if gemini_api_key_alias and "ENGRAM_GEMINI_API_KEY" not in os.environ:
    os.environ["ENGRAM_GEMINI_API_KEY"] = gemini_api_key_alias
if anthropic_api_key_alias and "ENGRAM_ANTHROPIC_API_KEY" not in os.environ:
    os.environ["ENGRAM_ANTHROPIC_API_KEY"] = anthropic_api_key_alias

# Optimized default stack: free on-device embeddings + Gemini/Claude for responses.
# setdefault keeps any value already supplied via the environment or .env.
os.environ.setdefault("ENGRAM_EMBEDDING_PROVIDER", "sentence-transformers")
os.environ.setdefault("ENGRAM_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
os.environ.setdefault("ENGRAM_EMBEDDING_DIMENSION", "384")

os.environ.setdefault("ENGRAM_GEMINI_MODEL", "gemini-3.1-flash-lite")
os.environ.setdefault("ENGRAM_ANTHROPIC_MODEL", "claude-sonnet-4-6")

os.environ.setdefault("ENGRAM_LLM_PROVIDER", "gemini")
provider = os.environ["ENGRAM_LLM_PROVIDER"].lower()
if provider in ("gemini", "google"):
    os.environ.setdefault("ENGRAM_LLM_MODEL", os.environ["ENGRAM_GEMINI_MODEL"])
elif provider in ("anthropic", "claude"):
    os.environ.setdefault("ENGRAM_LLM_MODEL", os.environ["ENGRAM_ANTHROPIC_MODEL"])

AGENT_ID = os.environ.get("ENGRAM_CHATBOT_AGENT_ID", "engram-chatbot")
USER_ID = os.environ.get("ENGRAM_CHATBOT_USER_ID", "default-user")

# Conversational continuity: last N raw turns replayed to the composer so it
# keeps the thread without re-retrieving. Separate from memory retrieval.
HISTORY_LIMIT = 8

# Retrieval knobs — tuned for Chonkie-chunked ingestion.
# SEARCH_LIMIT x CHUNK_TOKENS ≈ total tokens of evidence injected per turn.
# 10 x 500 ≈ 5 000 tokens: denser than the old 1 500-token ceiling because
# nothing is truncated. CANDIDATE_LIMIT stays wide so the reranker has room.
SEARCH_LIMIT = int(os.environ.get("ENGRAM_CHATBOT_SEARCH_LIMIT", "10"))
MAX_PER_SESSION = int(os.environ.get("ENGRAM_CHATBOT_MAX_PER_SESSION", "3"))
RERANK_MODE = os.environ.get("ENGRAM_CHATBOT_RERANK", "true").lower()
VALID_RERANK_MODES = {"true", "false"}

# Chonkie chunking: token budget per stored chunk.  Long turns are split into
# boundary-aware chunks so retrieval can surface the most relevant portion.
CHUNK_TOKENS = int(os.environ.get("ENGRAM_CHATBOT_CHUNK_TOKENS", "500"))

# Store the assistant's reply as a memory too (role-tagged). The composer prompt
# treats user turns as authoritative; assistant turns help answer "what did you
# tell me / suggest". Set to 0 to store only user turns.
STORE_ASSISTANT_TURNS = os.environ.get("ENGRAM_CHATBOT_STORE_ASSISTANT", "1") != "0"

COLOR_ENABLED = (
    sys.stdout.isatty()
    and os.environ.get("NO_COLOR") is None
    and os.environ.get("TERM") != "dumb"
)

# ---------------------------------------------------------------------------
# Composer prompt — the "second brain": conversational, grounded in evidence.
# ---------------------------------------------------------------------------
COMPOSER_SYSTEM = """You are the user's "second brain" — a warm, personal assistant that remembers everything they have told you across past conversations. Today's date is {today}.

You answer from the MEMORIES block provided below, which is retrieved from the user's own conversation history. Talk naturally, like a thoughtful friend with perfect recall — not like a database readout. Be concise and direct; skip preambles like "Based on your memories".

How to use the memories:

- The MEMORIES block is your source of truth about the user. If something genuinely isn't there, say you don't have it yet — never invent names, numbers, dates, or details.

- Memories tagged [ACTIVE] are current. Memories tagged [SUPERSEDED] are OLD values that were later changed — use them only to answer "what was it before / originally" questions, never as the current answer. When a value changed, it's natural to mention both ("it's X now — it used to be Y").

- Most recent wins for conflicting values of the same fact. Memories about different people, places, or contexts are not in conflict.

- Do the math when the memories hold the numbers (ages, prices, durations, date differences) instead of refusing — even when the facts are scattered across different memories. Compute relative time expressions against today's date above.

- Respect what the user avoids — allergies, dislikes, hard constraints. Never suggest something they have said they want to avoid, not even as a fallback.

- Match the exact thing asked about. If they ask about one specific entity/variant/role and memory only mentions a different one, don't treat them as the same — say you don't have that exact detail.

- For "what do you remember about me" style questions, synthesize the durable picture (who they are, what they're working on, their preferences) in a few natural sentences rather than dumping raw lines.

Think privately inside <mem_thinking>...</mem_thinking> tags first: list the relevant memories, do any counting / temporal / cross-topic reasoning, check avoidances and context, then decide. The user only sees text OUTSIDE the tags — put your natural, conversational reply there."""

_THINKING_CLOSE = "</mem_thinking>"


def build_system_prompt() -> str:
    return COMPOSER_SYSTEM.format(today=date.today().isoformat())


def strip_thinking(text: str) -> str:
    """Drop the hidden <mem_thinking> scratchpad, keeping the final answer."""
    lowered = text.lower()
    if _THINKING_CLOSE in lowered:
        return text[lowered.rfind(_THINKING_CLOSE) + len(_THINKING_CLOSE) :].strip()
    return text.strip()


# ===========================================================================
# Rendering helpers
# ===========================================================================


def preview(text: str, limit: int = 180) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[:limit]}..."


def format_timestamp(value: Any) -> str:
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat(timespec="minutes")
        except TypeError:
            return value.isoformat()
    return str(value)


def paint(text: str, code: str) -> str:
    if not COLOR_ENABLED:
        return text
    return f"\033[{code}m{text}\033[0m"


def dim(text: str) -> str:
    return paint(text, "2")


def bold(text: str) -> str:
    return paint(text, "1")


def accent(text: str) -> str:
    return paint(text, "36")


def good(text: str) -> str:
    return paint(text, "32")


def warn(text: str) -> str:
    return paint(text, "33")


def bad(text: str) -> str:
    return paint(text, "31")


def terminal_width() -> int:
    return max(72, min(104, shutil.get_terminal_size((88, 20)).columns))


def rule(label: str = "") -> str:
    if HAS_RICH:
        # Return empty — callers that use rule() as a separator will be
        # replaced by richer equivalents.  Keep it functional for fallback
        # callers that print(rule(...)).
        width = terminal_width()
        if not label:
            return dim("─" * width)
        prefix = f" {label} "
        return dim(prefix + "─" * max(0, width - len(prefix)))
    width = terminal_width()
    if not label:
        return dim("-" * width)
    prefix = f" {label} "
    return dim(prefix + "-" * max(0, width - len(prefix)))


def print_header() -> None:
    if HAS_RICH:
        console.print()
        title = Text()
        title.append("🧠 Engram Memory Chat", style="bold cyan")
        title.append("  ·  ", style="dim")
        title.append("benchmark pipeline · second brain", style="dim italic")
        console.print(
            Panel(
                title,
                border_style="cyan",
                padding=(0, 2),
                subtitle="[dim italic]Type a message to chat, or /help for commands[/]",
                subtitle_align="left",
            )
        )
        return
    print()
    print(rule("engram"))
    print(f"{bold('Engram Memory Chat')} | {dim('benchmark pipeline · second brain')}")
    print(dim("Type a message to chat, or /help for commands."))


def print_table(rows: list[tuple[str, Any]]) -> None:
    if HAS_RICH:
        table = Table(
            show_header=False,
            show_edge=False,
            box=None,
            padding=(0, 1, 0, 2),
            expand=False,
        )
        table.add_column("Key", style="engram.key", min_width=18)
        table.add_column("Value", style="engram.value")
        for key, value in rows:
            table.add_row(key, str(value))
        console.print(table)
        return
    for key, value in rows:
        print(f"{dim(f'{key:<20}')} {value}")


def print_status_panel(rows: list[tuple[str, Any]]) -> None:
    if HAS_RICH:
        print_header()
        table = Table(
            show_header=False,
            show_edge=False,
            box=None,
            padding=(0, 1),
            expand=False,
        )
        table.add_column("Key", style="engram.key", min_width=18)
        table.add_column("Value", style="engram.value")
        for key, value in rows:
            # Colorize health value
            if key == "health":
                style = "green bold" if str(value) == "healthy" else "yellow bold"
                table.add_row(key, Text(str(value), style=style))
            else:
                table.add_row(key, str(value))
        console.print(
            Panel(
                table,
                title="[bold]session[/]",
                title_align="left",
                border_style="dim cyan",
                padding=(0, 1),
            )
        )
        return
    print_header()
    print(rule("session"))
    print_table(rows)
    print(rule())


def print_notice(message: str, *, level: str = "info") -> None:
    if HAS_RICH:
        style_map = {
            "info": "engram.accent",
            "ok": "engram.good",
            "warn": "engram.warn",
            "error": "engram.bad",
        }
        prefix_map = {
            "info": "▸ engram",
            "ok": "✓ engram",
            "warn": "⚠ engram",
            "error": "✗ error",
        }
        style = style_map.get(level, "engram.accent")
        prefix = prefix_map.get(level, "▸ engram")
        text = Text()
        text.append(prefix, style=style)
        text.append(f"  {message}", style="dim")
        console.print(text)
        return
    prefix = {
        "info": accent("engram"),
        "ok": good("engram"),
        "warn": warn("engram"),
        "error": bad("error"),
    }.get(level, accent("engram"))
    print(f"{prefix} {dim(message)}")


def print_response(text: str) -> None:
    if HAS_RICH:
        console.print()
        md = Markdown(text, code_theme="monokai")
        console.print(
            Panel(
                md,
                title="[bold cyan]assistant[/]",
                title_align="left",
                border_style="cyan",
                padding=(1, 2),
                expand=True,
            )
        )
        return
    # Fallback: plain text rendering
    width = terminal_width() - 4
    print()
    print(f"{accent('assistant')} {dim('-' * max(1, terminal_width() - 10))}")
    for raw_line in text.splitlines() or [""]:
        if not raw_line.strip():
            print()
            continue
        if raw_line.lstrip().startswith(("-", "*")):
            print(f"  {raw_line}")
            continue
        print(
            textwrap.fill(
                raw_line,
                width=width,
                initial_indent="  ",
                subsequent_indent="  ",
                break_long_words=False,
                break_on_hyphens=False,
            )
        )


def prompt_text() -> str:
    if HAS_RICH:
        # Rich markup is stripped by input(), so we bake ANSI via console.export.
        you = Text("you", style="bold cyan")
        arrow = Text(" > ", style="dim")
        combined = Text()
        combined.append_text(you)
        combined.append_text(arrow)
        with console.capture() as capture:
            console.print(combined, end="")
        return capture.get()
    return f"{accent('you')} {dim('> ')}" if COLOR_ENABLED else "you> "


def require_real_config() -> None:
    if RERANK_MODE not in VALID_RERANK_MODES:
        raise ValueError("Invalid ENGRAM_CHATBOT_RERANK. Use 'true' or 'false'.")

    provider = os.environ.get("ENGRAM_LLM_PROVIDER", "gemini").lower()
    if provider in ("gemini", "google") and not os.environ.get("ENGRAM_GEMINI_API_KEY"):
        raise ValueError(
            "Missing required environment variable: ENGRAM_GEMINI_API_KEY "
            "(or GEMINI_API_KEY).\nGet a key at https://aistudio.google.com/apikey "
            "and set it before running the chatbot."
        )
    elif provider in ("anthropic", "claude") and not os.environ.get(
        "ENGRAM_ANTHROPIC_API_KEY"
    ):
        raise ValueError(
            "Missing required environment variable: ENGRAM_ANTHROPIC_API_KEY "
            "(or ANTHROPIC_API_KEY).\nGet a key at https://console.anthropic.com/ "
            "and set it before running the chatbot."
        )


# ===========================================================================
# Retrieval — ported verbatim from the benchmark scripts (proven pipeline).
# ===========================================================================


def _to_human_date(date_str: str) -> str:
    """Render an ISO 'YYYY-MM-DD' chat date as 'Month D, YYYY'."""
    if not date_str:
        return "Unknown Date"
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(date_str[:19], fmt).strftime("%B %d, %Y")
        except ValueError:
            continue
    return date_str


def diversify_by_session(
    results: list[Any],
    *,
    limit: int,
    max_per_session: int,
    rerank: bool = False,
) -> list[Any]:
    """Round-robin a candidate pool across sessions, user turns first.

    Ported from the benchmark scripts. Prevents the evidence budget from
    collapsing onto one session's near-duplicate turns. When reranking, the
    candidate order is already a relevance ranking, so the user-first nudge is
    dropped (it would bury an assistant turn that IS the answer).
    """
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
            enumerate(results),
            key=lambda it: (it[0] + role_bias(it[1]), it[0]),
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
        if r.memory.memory_id in seen:
            continue
        selected.append(r)
        if len(selected) >= limit:
            break
    return selected


def _build_evidence_block(
    search_results: list[Any],
    recall_answer: Any | None,
    lineage_superseded: list[Any] | None = None,
) -> str:
    """Assemble the evidence block from Engram's retrieval surfaces.

    Identical structure to the benchmark `_build_evidence_block`: recall's
    structured lineage first, then superseded predecessors, then the hybrid
    search hits grouped by date and tagged [ACTIVE]/[SUPERSEDED].
    Graph traversal is excluded: add_batch() ingest creates no edges.
    """
    lines: list[str] = []
    seen: set[str] = set()

    if recall_answer is not None:
        cur = getattr(recall_answer, "current", None)
        if cur is not None:
            seen.add(cur.memory_id)
            lines.append(f"CURRENT: {cur.fact or cur.content}")
        for mem in getattr(recall_answer, "previous", []) or []:
            when = mem.superseded_at or mem.valid_to or mem.created_at
            stamp = when.date().isoformat() if when else "unknown"
            seen.add(mem.memory_id)
            lines.append(f"SUPERSEDED (until {stamp}): {mem.fact or mem.content}")
        # Include temporal_chain and other evidence-intent items (BEAM pipeline pattern).
        for mem in getattr(recall_answer, "evidence", []) or []:
            if mem.memory_id in seen:
                continue
            seen.add(mem.memory_id)
            lines.append(f"RECALL: {mem.fact or mem.content}")
        if getattr(recall_answer, "conflict_note", None):
            lines.append(f"CONFLICT: {recall_answer.conflict_note}")
        if getattr(recall_answer, "answer_text", "").strip():
            lines.append(f"RECALL NOTE: {recall_answer.answer_text.strip()}")

    for mem in lineage_superseded or []:
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

    return "\n".join(lines) if lines else "(no matching memory)"


def _chunk_text(text: str) -> list[str]:
    """Split ``text`` into Chonkie chunks bounded at CHUNK_TOKENS.

    Falls back to a single-item list when chonkie is not installed or chunking
    fails, so ingestion is never blocked by a missing optional dependency."""
    try:
        from engram.chunking import chonkie_recursive_spans

        spans = chonkie_recursive_spans(text, max_chunk_tokens=CHUNK_TOKENS)
        if spans:
            return [body for (_, body, _, _) in spans]
    except Exception:
        pass
    return [text]


class MemoryChatbot:
    def __init__(self) -> None:
        self.engram = None
        self.task_id: str | None = None
        self.session_id: str | None = None
        self._session_context: Any | None = None
        self.history: list[dict[str, str]] = []
        self._turn_index = 0
        self._last_evidence = ""

    # -- lifecycle ---------------------------------------------------------

    async def connect(self) -> None:
        require_real_config()

        from engram import Engram
        from engram.core.config import get_settings

        # Match the benchmark settings: keep every turn verbatim (no near-dup
        # collapse), allow the on-device embedding dimension.
        settings = get_settings()
        settings = settings.model_copy(
            update={
                "near_duplicate_threshold": 1.0,
                "allow_embedding_dimension_change": True,
            }
        )

        self.engram = Engram(settings=settings, memory_policy="default")
        await self.engram.connect()
        if self.engram.llm is None:
            provider = os.environ.get("ENGRAM_LLM_PROVIDER", "gemini")
            raise RuntimeError(
                f"LLM provider '{provider}' is disabled or misconfigured. "
                "Set ENGRAM_LLM_PROVIDER=gemini + ENGRAM_GEMINI_API_KEY, or "
                "ENGRAM_LLM_PROVIDER=anthropic + ENGRAM_ANTHROPIC_API_KEY."
            )

        # Warm the reranker now (off the event loop) so the first turn's search
        # doesn't stall mid-conversation loading model weights.
        if self._rerank_enabled():
            await self.engram.warmup()

        await self._resume_or_start_task()
        health = await self.engram.health_check()
        health_status = str(health.get("status"))
        print_status_panel(
            [
                (
                    "health",
                    good(health_status)
                    if health_status == "healthy"
                    else warn(health_status),
                ),
                ("agent", AGENT_ID),
                ("user", USER_ID),
                ("session", self.session_id),
                (
                    "embedding",
                    f"{os.environ.get('ENGRAM_EMBEDDING_PROVIDER')} / "
                    f"{os.environ.get('ENGRAM_EMBEDDING_MODEL')} "
                    f"({os.environ.get('ENGRAM_EMBEDDING_DIMENSION')}d)",
                ),
                (
                    "llm",
                    f"{os.environ.get('ENGRAM_LLM_PROVIDER')} / "
                    f"{os.environ.get('ENGRAM_LLM_MODEL')}",
                ),
                ("pipeline", "add_batch -> search+recall+lineage -> composer"),
                ("rerank", self._rerank_enabled()),
                ("search_limit", SEARCH_LIMIT),
            ]
        )

    async def _resume_or_start_task(self) -> None:
        assert self.engram is not None

        tasks = await self.engram.list_tasks(
            agent_id=AGENT_ID,
            user_id=USER_ID,
            status=["active", "paused"],
            limit=1,
        )
        if tasks:
            task = tasks[0]
            self.task_id = task.task_run_id
            self.session_id = task.session_id
            return

        self._session_context = self.engram.session(
            AGENT_ID,
            user_id=USER_ID,
            metadata={"application": "examples/chatbot.py"},
        )
        session = await self._session_context.__aenter__()
        self.session_id = session.session_id
        task = await self.engram.start_task(
            "Run a real Engram memory chatbot",
            AGENT_ID,
            user_id=USER_ID,
            session_id=self.session_id,
            metadata={"application": "examples/chatbot.py"},
        )
        self.task_id = task.task_run_id

    async def close(self) -> None:
        if self.engram is None:
            return
        if self.task_id is not None:
            with contextlib.suppress(Exception):
                await self.engram.pause_task(
                    self.task_id,
                    outcome="Chatbot process exited; task can be resumed later.",
                )
        if self._session_context is not None:
            with contextlib.suppress(Exception):
                await self._session_context.__aexit__(None, None, None)
        await self.engram.close()

    def _rerank_enabled(self) -> bool:
        return RERANK_MODE == "true"

    async def switch_model(self, model_choice: str) -> None:
        model_choice = model_choice.lower()
        if model_choice not in ("gemini", "claude", "anthropic"):
            print_notice(
                "Invalid model choice. Use 'gemini' or 'claude'.", level="warn"
            )
            return

        print_notice(f"Switching model to {model_choice}...")

        # Save current state to revert if needed
        prev_provider = os.environ.get("ENGRAM_LLM_PROVIDER")
        prev_model = os.environ.get("ENGRAM_LLM_MODEL")

        from engram.core.config import clear_settings_cache

        if model_choice == "gemini":
            os.environ["ENGRAM_LLM_PROVIDER"] = "gemini"
            os.environ["ENGRAM_LLM_MODEL"] = os.environ.get(
                "ENGRAM_GEMINI_MODEL", "gemini-3.1-flash-lite"
            )
        else:
            os.environ["ENGRAM_LLM_PROVIDER"] = "anthropic"
            os.environ["ENGRAM_LLM_MODEL"] = os.environ.get(
                "ENGRAM_ANTHROPIC_MODEL", "claude-sonnet-4-6"
            )

        try:
            require_real_config()
        except ValueError as exc:
            print_notice(str(exc), level="error")
            print_notice("Reverting to previous model...", level="warn")
            if prev_provider:
                os.environ["ENGRAM_LLM_PROVIDER"] = prev_provider
            if prev_model:
                os.environ["ENGRAM_LLM_MODEL"] = prev_model
            return

        await self.close()
        clear_settings_cache()

        try:
            await self.connect()
        except Exception as exc:
            print_notice(f"Failed to connect with new model: {exc}", level="error")
            print_notice("Reverting to previous model...", level="warn")
            if prev_provider:
                os.environ["ENGRAM_LLM_PROVIDER"] = prev_provider
            if prev_model:
                os.environ["ENGRAM_LLM_MODEL"] = prev_model
            clear_settings_cache()
            await self.connect()

    # -- the pipeline ------------------------------------------------------

    async def reply(self, message: str) -> str:
        """One turn: retrieve over past memory, compose, then store this turn."""
        assert self.engram is not None

        # 1. RETRIEVE — 3 surfaces over everything stored BEFORE this turn.
        evidence, _trace = await self._retrieve_evidence(message)
        self._last_evidence = evidence

        # 2. GENERATE — benchmark-aligned: evidence in user turn, <mem_thinking> primer.
        user_content = (
            f"<engram_memory_evidence>\n{evidence}\n</engram_memory_evidence>\n\n"
            f"{message}\n\n"
            "IMPORTANT: You MUST provide your full thinking in <mem_thinking> tags "
            "BEFORE giving your answer."
        )
        messages = [
            {"role": "system", "content": build_system_prompt()},
            *self.history[-HISTORY_LIMIT:],
            {"role": "user", "content": user_content},
        ]
        llm_response = await self.engram.llm.complete_full(
            messages,
            max_tokens=2000,
            temperature=0.4,
        )
        response = strip_thinking(llm_response.content.strip())

        # 3. INGEST — store this turn verbatim (add_batch, no LLM extraction).
        await self._ingest_turn(message, response)

        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": response})
        self.history = self.history[-HISTORY_LIMIT * 2 :]

        return response

    async def _retrieve_evidence(self, question: str) -> tuple[str, dict[str, Any]]:
        """3-surface retrieval, identical to the benchmark `retrieve_evidence`."""
        assert self.engram is not None
        rerank = self._rerank_enabled()
        n_search = n_lineage = 0
        recall_intent = ""
        evidence = ""

        try:
            # a) Hybrid search, wide candidate pool, diversified to the budget.
            candidate_limit = (
                100 if rerank else min(max(SEARCH_LIMIT * 3, SEARCH_LIMIT), 100)
            )
            candidates = await self.engram.search(
                query=question,
                agent_id=AGENT_ID,
                user_id=USER_ID,
                limit=candidate_limit,
                mode="hybrid",
                rerank=rerank,
                include_superseded=True,
            )
            search_results = diversify_by_session(
                candidates,
                limit=SEARCH_LIMIT,
                max_per_session=MAX_PER_SESSION,
                rerank=rerank,
            )
            n_search = len(search_results)
            search_results.sort(key=lambda r: r.memory.metadata.get("chat_date", ""))

            # b) Recall as a structured lineage aid (not the answer).
            recall_answer = None
            try:
                recall_answer = await self.engram.recall(
                    question,
                    AGENT_ID,
                    user_id=USER_ID,
                    limit=max(SEARCH_LIMIT // 2, 10),
                    compose_answer=False,
                )
                recall_intent = getattr(recall_answer, "intent", "")
            except Exception as exc:
                recall_intent = f"error:{type(exc).__name__}"

            # c) Lineage preservation for retrieved active facts.
            lineage_superseded: list[Any] = []
            seen_lineages: set[str] = set()
            for r in search_results:
                mem = r.memory
                lid = getattr(mem, "lineage_id", None)
                status = getattr(mem, "status", None) or mem.metadata.get(
                    "status", "active"
                )
                if (
                    status != "superseded"
                    and lid
                    and lid != mem.memory_id
                    and lid not in seen_lineages
                ):
                    seen_lineages.add(lid)
                    try:
                        lineage = await self.engram.get_lineage(mem.memory_id)
                        lineage_superseded.extend(
                            m
                            for m in lineage.memories
                            if getattr(m, "status", None) == "superseded"
                        )
                    except Exception:
                        pass
            n_lineage = len(lineage_superseded)

            evidence = _build_evidence_block(
                search_results, recall_answer, lineage_superseded
            )
        except Exception as exc:
            evidence = f"(retrieval error: {type(exc).__name__}: {exc})"

        return evidence, {
            "search_hits": n_search,
            "lineage_superseded": n_lineage,
            "recall_intent": recall_intent,
        }

    async def _ingest_turn(self, user_message: str, assistant_response: str) -> None:
        """Store the turn as date-anchored episodic rows via add_batch.

        Long texts are split into boundary-aware Chonkie chunks; each chunk
        becomes its own memory row tagged with chunk_index / chunk_count.
        Short texts (or when chonkie is unavailable) produce a single row.
        No content is ever truncated and discarded.

        No LLM extraction, no supersession decisioning — the benchmark floor
        behaviour. The user turn is authoritative; the assistant turn is stored
        role-tagged so 'what did you tell me' is answerable.
        """
        assert self.engram is not None
        chat_date = date.today().isoformat()
        human_date = _to_human_date(chat_date)
        turn = self._turn_index
        self._turn_index += 1

        def _rows_for(role_label: str, role: str, text: str) -> list[dict[str, Any]]:
            chunks = _chunk_text(text)
            total = len(chunks)
            rows: list[dict[str, Any]] = []
            for idx, body in enumerate(chunks):
                content = f"[{human_date}] {role_label}: {body}"
                meta: dict[str, Any] = {
                    "source": "chatbot",
                    "original_session_id": str(self.session_id),
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
        if STORE_ASSISTANT_TURNS and assistant_response.strip():
            rows.extend(_rows_for("ASSISTANT", "assistant", assistant_response))

        try:
            await self.engram.add_batch(rows)
        except Exception as exc:
            print_notice(f"failed to store turn: {exc}", level="warn")

    # -- commands ----------------------------------------------------------

    async def remember(self, text: str) -> None:
        assert self.engram is not None
        memory = await self.engram.add(
            text,
            AGENT_ID,
            user_id=USER_ID,
            session_id=self.session_id,
            memory_type="semantic",
            metadata={"source": "manual_chatbot_memory"},
        )
        print(rule("stored"))
        print_table(
            [
                ("memory_id", memory.memory_id),
                ("lineage_id", memory.lineage_id),
                ("revision", memory.revision),
                ("status", memory.status),
                ("content", preview(memory.content)),
            ]
        )

    async def memories(self) -> None:
        assert self.engram is not None
        memories = await self.engram.list_recent(AGENT_ID, user_id=USER_ID, limit=15)
        if not memories:
            print_notice("no memories stored yet", level="warn")
            return
        print(rule("memories"))
        for memory in memories:
            print(
                f"  {accent(memory.memory_id[:16])} "
                f"{dim(f'[{memory.memory_type} r{memory.revision} {memory.status} {memory.importance:.2f}]')} "
                f"{preview(memory.content, 120)}"
            )

    async def evidence(self, query: str) -> None:
        """Show the 3-surface evidence block the composer would see."""
        block, trace = await self._retrieve_evidence(query)
        print(
            rule(
                f"evidence [search={trace['search_hits']} "
                f"lineage={trace['lineage_superseded']} "
                f"recall={trace['recall_intent']}]"
            )
        )
        print(block or "(no matching memory)")

    async def show_history(self, rest: str = "") -> None:
        assert self.engram is not None
        arg = rest.strip()
        if arg.startswith("mem_"):
            await self.lineage(arg)
            return

        include_superseded = True
        limit = 20
        if arg == "active":
            include_superseded = False
        elif arg.isdigit():
            limit = max(1, min(100, int(arg)))
        elif arg:
            print_notice("usage: /history [active|limit|memory_id]", level="warn")
            return

        events = await self.engram.get_history(
            AGENT_ID,
            user_id=USER_ID,
            limit=limit,
            include_superseded=include_superseded,
        )
        if not events:
            print_notice("no memory history yet", level="warn")
            return

        print(rule("history"))
        for event in events:
            marker = {"added": "+", "revised": "~", "superseded": "x"}.get(
                event.event_type, "?"
            )
            relation = ""
            if event.event_type == "revised" and event.previous_memory_id:
                relation = f" prev={event.previous_memory_id[:12]}"
            elif event.event_type == "superseded" and event.superseded_by_memory_id:
                relation = f" by={event.superseded_by_memory_id[:12]}"
            print(
                f"  {accent(marker)} {accent(event.event_type.ljust(10))} "
                f"{dim(format_timestamp(event.occurred_at))} "
                f"{accent(event.memory.memory_id[:16])} "
                f"{dim(f'[{event.memory.memory_type} r{event.memory.revision} {event.memory.status}]')} "
                f"{preview(event.memory.content, 110)}{dim(relation)}"
            )

    async def revise(self, memory_id: str, content: str) -> None:
        assert self.engram is not None
        memory = await self.engram.revise(
            memory_id,
            content=content,
            metadata={"source": "manual_chatbot_revision"},
            reason="manual_chatbot_revision",
        )
        current = await self.engram.get_current(memory_id)
        lineage = await self.engram.get_lineage(memory_id)
        print(rule("revised"))
        print_table(
            [
                ("new_memory_id", memory.memory_id),
                ("current", current.memory_id),
                ("lineage_id", lineage.lineage_id),
                ("revisions", len(lineage.memories)),
                ("content", preview(memory.content)),
            ]
        )

    async def lineage(self, memory_id: str) -> None:
        assert self.engram is not None
        current = await self.engram.get_current(memory_id)
        lineage = await self.engram.get_lineage(memory_id)
        explanation = await self.engram.explain_memory(memory_id)
        print(rule("lineage"))
        print_table(
            [
                ("lineage_id", lineage.lineage_id),
                ("current", current.memory_id),
                ("selected", explanation.memory.memory_id),
                ("selected_status", explanation.memory.status),
                (
                    "superseded_by",
                    explanation.superseded_by.memory_id
                    if explanation.superseded_by
                    else None,
                ),
                ("supersedes", [m.memory_id for m in explanation.supersedes]),
            ]
        )
        print(rule("revisions"))
        for memory in lineage.memories:
            marker = "*" if memory.memory_id == current.memory_id else " "
            print(
                f"{marker} {accent(memory.memory_id[:16])} "
                f"{dim(f'r{memory.revision} {memory.status}')} "
                f"{preview(memory.content, 120)}"
            )

    async def search(self, query: str) -> None:
        assert self.engram is not None
        results = await self.engram.search(
            query,
            AGENT_ID,
            user_id=USER_ID,
            limit=8,
            mode="hybrid",
            rerank=self._rerank_enabled(),
            include_superseded=True,
        )
        if not results:
            print_notice("no matches", level="warn")
            return
        print(rule("search"))
        for result in results:
            print(
                f"  {accent(f'{result.score:.3f}')} "
                f"{dim(result.memory.memory_type)} "
                f"{preview(result.memory.content)}"
            )

    async def recall(self, question: str) -> None:
        assert self.engram is not None
        answer = await self.engram.recall(question, AGENT_ID, user_id=USER_ID)
        print(rule(f"recall [{answer.intent}]"))
        print(f"  {answer.answer_text}")
        if answer.previous:
            prev = "; ".join(preview(m.fact or m.content, 60) for m in answer.previous)
            print(f"  {dim('previously:')} {prev}")
        if answer.when_changed:
            print(f"  {dim('changed:')} {answer.when_changed.date().isoformat()}")
        if answer.conflict_note:
            print(f"  {warn('conflict:')} {answer.conflict_note}")

    async def forget(self, memory_id: str) -> None:
        assert self.engram is not None
        deleted = await self.engram.forget(memory_id)
        print(rule("forget"))
        print_table([("deleted", deleted)])

    async def clear(self) -> None:
        assert self.engram is not None
        count = await self.engram.purge(AGENT_ID, user_id=USER_ID)
        self.history.clear()
        self._turn_index = 0
        print(rule("clear"))
        print_table([("purged", count)])


COMMANDS = [
    ("/remember <fact>", "store a durable fact immediately"),
    ("/revise <memory_id> <fact>", "create a new active revision"),
    ("/lineage <memory_id>", "show current head and revision history"),
    ("/history [active|limit|memory_id]", "show memory add/update timeline"),
    ("/memories", "list recent Engram memories"),
    ("/search <query>", "hybrid search over stored memories"),
    ("/recall <question>", "ask memory: current/historical/event/lineage answer"),
    ("/evidence <query>", "show the 3-surface evidence block for a query"),
    ("/forget <memory_id>", "delete one memory"),
    ("/clear", "purge this chatbot user's memories"),
    ("/model <gemini|claude>", "switch the LLM model"),
    ("/help", "show this command palette"),
    ("/quit", "exit"),
]


def print_help() -> None:
    if HAS_RICH:
        table = Table(
            show_header=False,
            show_edge=False,
            box=None,
            padding=(0, 1),
            expand=False,
        )
        table.add_column("Command", style="engram.cmd", no_wrap=True)
        table.add_column("Description", style="engram.cmd_desc")
        for command, description in COMMANDS:
            table.add_row(command, description)

        group = Text()
        group.append_text(Text.from_markup("\n"))
        footer = Text(
            "Plain text runs the benchmark pipeline: "
            "retrieve (search + recall + lineage) → compose → "
            "store the turn via add_batch.",
            style="dim italic",
        )
        console.print()
        console.print(
            Panel(
                table,
                title="[bold]commands[/]",
                title_align="left",
                border_style="dim cyan",
                padding=(1, 2),
                subtitle=footer,
                subtitle_align="left",
            )
        )
        return
    print()
    print(rule("commands"))
    width = max(len(command) for command, _ in COMMANDS)
    for command, description in COMMANDS:
        print(f"  {accent(command.ljust(width))}  {dim(description)}")
    print()
    print(
        dim(
            "Plain text runs the benchmark pipeline: "
            "retrieve (search + recall + lineage) -> compose -> "
            "store the turn via add_batch."
        )
    )
    print(rule())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an Engram memory chatbot (Gemini or Claude)."
    )
    parser.add_argument(
        "--once",
        metavar="MESSAGE",
        help="send one message, print the response, and exit",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="run one deterministic memory-backed demo turn and exit",
    )
    return parser.parse_args(argv)


async def run_command(bot: MemoryChatbot, line: str) -> bool:
    command, _, rest = line.partition(" ")
    command = command.lower()
    rest = rest.strip()

    if command in {"/quit", "/q"}:
        return False
    if command == "/help":
        print_help()
    elif command == "/remember" and rest:
        await bot.remember(rest)
    elif command == "/revise" and rest:
        memory_id, _, content = rest.partition(" ")
        if memory_id and content.strip():
            await bot.revise(memory_id, content.strip())
        else:
            print_notice("usage: /revise <memory_id> <fact>", level="warn")
    elif command == "/lineage" and rest:
        await bot.lineage(rest)
    elif command == "/history":
        await bot.show_history(rest)
    elif command == "/memories":
        await bot.memories()
    elif command == "/search" and rest:
        await bot.search(rest)
    elif command == "/recall" and rest:
        await bot.recall(rest)
    elif command == "/evidence" and rest:
        await bot.evidence(rest)
    elif command == "/forget" and rest:
        await bot.forget(rest)
    elif command == "/model" and rest:
        await bot.switch_model(rest)
    elif command == "/clear":
        clear_prompt = (
            warn("clear") + " " + dim("Delete this user's chatbot memories? y/N: ")
        )
        confirm = await asyncio.to_thread(input, clear_prompt)
        if confirm.lower() == "y":
            await bot.clear()
    else:
        print_notice("unknown or incomplete command, type /help", level="warn")
    return True


async def run_once(bot: MemoryChatbot, message: str) -> None:
    print(rule("once"))
    print_table([("message", preview(message, 100))])
    response = await bot.reply(message)
    print_response(response)


async def run_demo(bot: MemoryChatbot) -> None:
    assert bot.engram is not None
    old = await bot.engram.add(
        "The user's live chatbot demo city is Dhaka.",
        AGENT_ID,
        user_id=USER_ID,
        session_id=bot.session_id,
        memory_type="profile",
        metadata={
            "source": "examples/chatbot.py --demo",
            "conflict_key": f"{AGENT_ID}:{USER_ID}:demo:city",
        },
    )
    new = await bot.engram.revise(
        old.memory_id,
        content="The user's live chatbot demo city is Singapore.",
        metadata={"source": "examples/chatbot.py --demo"},
        reason="demo_correction",
    )
    current = await bot.engram.get_current(old.memory_id)
    lineage = await bot.engram.get_lineage(old.memory_id)
    explanation = await bot.engram.explain_memory(old.memory_id)
    print(rule("demo lineage"))
    print_table(
        [
            ("old_memory_id", old.memory_id),
            ("new_memory_id", new.memory_id),
            ("lineage_id", lineage.lineage_id),
            ("current", current.memory_id),
            ("old_status", explanation.memory.status),
            (
                "old_superseded_by",
                explanation.superseded_by.memory_id
                if explanation.superseded_by
                else None,
            ),
            (
                "revisions",
                [f"r{memory.revision}:{memory.status}" for memory in lineage.memories],
            ),
        ]
    )
    await run_once(
        bot,
        "Using Engram memory only, what is my live chatbot demo city?",
    )


async def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    bot = MemoryChatbot()
    try:
        await bot.connect()
    except DatabaseConnectionError as exc:
        print_notice(f"database connection failed: {exc}", level="error")
        print_notice(
            "Check that ENGRAM_DATABASE_URL matches the running Postgres "
            "container credentials. If you used docker-setup.sh, the password "
            "in .env may differ from an older existing Docker volume.",
            level="warn",
        )
        return
    except ValueError as exc:
        print_notice(str(exc), level="error")
        return
    try:
        if args.demo:
            await run_demo(bot)
            return
        if args.once:
            await run_once(bot, args.once)
            return

        print_help()
        while True:
            try:
                line = (await asyncio.to_thread(input, prompt_text())).strip()
            except EOFError:
                break
            if not sys.stdin.isatty():
                print()
            if not line:
                continue
            if line.startswith("/"):
                if not await run_command(bot, line):
                    break
            else:
                if COLOR_ENABLED:
                    print_notice("retrieving memory and calling the composer")
                response = await bot.reply(line)
                print_response(response)
    finally:
        await bot.close()
        if not sys.stdin.isatty():
            print()
        print_notice("bye", level="ok")


if __name__ == "__main__":
    asyncio.run(main())
