"""Engram client - Main entry point for the AI memory library.

This module provides the Engram class, the main async client for
interacting with the AI memory system.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import re
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Any

from engram.core.config import EngramSettings, get_settings
from engram.core.exceptions import (
    ConfigurationError,
    DuplicateMemoryError,
    EngramError,
    SessionNotFoundError,
)
from engram.embedding.service import EmbeddingService
from engram.graph.models import TraversalQuery, TraversalResult
from engram.graph.traversal import GraphTraversal
from engram.health.checker import HealthChecker
from engram.llm.service import LLMService, MemoryOperationType
from engram.memory.models import (
    Memory,
    MemoryCreate,
    MemoryExplanation,
    MemoryHistoryEvent,
    MemoryLineage,
    MemoryUpdate,
    RecallTrace,
    SearchQuery,
    SearchResult,
)
from engram.memory.store import MemoryStore
from engram.policy import MemoryPolicy, get_memory_policy
from engram.reranking import CrossEncoderReranker
from engram.session.manager import SessionManager
from engram.storage.postgres import PostgresStorage
from engram.task import (
    AgentEvent,
    ContextBuilder,
    ContextBuildOptions,
    ContextBuildResult,
    EventCreate,
    LongInputChunk,
    LongInputContextResult,
    LongInputIngestionReport,
    MemoryJob,
    TaskCheckpoint,
    TaskMemoryManager,
    TaskRun,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from datetime import datetime

    from engram.core._types import (
        AgentId,
        MemoryId,
        MemoryType,
        Metadata,
        RelationType,
        SearchMode,
        SessionId,
        UserId,
    )
    from engram.session.models import Session
    from engram.task.models import EventRole, EventType

logger = logging.getLogger(__name__)


class Engram:
    """Main async client for the Engram AI memory library.

    Engram provides a unified interface for AI memory management including:
    - Adding and retrieving memories
    - Hybrid search (vector + keyword + decay + importance)
    - Graph relations and traversal
    - Session management
    - Health checking

    Example:
        # Using async context manager (recommended)
        async with Engram() as engram:
            # Add a memory
            memory = await engram.add(
                content="User prefers dark mode",
                agent_id="my_agent",
                importance=0.8,
            )

            # Search memories
            results = await engram.search(
                query="user preferences",
                agent_id="my_agent",
            )

            # Create relations
            await engram.relate(
                source_id=memory.memory_id,
                target_id=other_memory_id,
                relation_type="related_to",
            )

        # Or manual lifecycle management
        engram = Engram()
        await engram.connect()
        try:
            # ... use engram ...
        finally:
            await engram.close()
    """

    def __init__(
        self,
        settings: EngramSettings | None = None,
        *,
        database_url: str | None = None,
        openai_api_key: str | None = None,
        memory_policy: str | MemoryPolicy | None = None,
    ) -> None:
        """Initialize the Engram client.

        Args:
            settings: Full settings object. If None, loads from environment.
            database_url: Override database URL from settings.
            openai_api_key: Override OpenAI API key from settings.
        """
        self._settings = settings or get_settings()
        self._memory_policy = get_memory_policy(memory_policy)

        # Apply overrides
        if database_url:
            self._settings = self._settings.model_copy(
                update={"database_url": database_url}
            )
        if openai_api_key:
            self._settings = self._settings.model_copy(
                update={"openai_api_key": openai_api_key}
            )

        # Initialize components (lazy, connected on connect())
        self._storage: PostgresStorage | None = None
        self._embedding: EmbeddingService | None = None
        self._memory_store: MemoryStore | None = None
        self._graph: GraphTraversal | None = None
        self._sessions: SessionManager | None = None
        self._health: HealthChecker | None = None
        self._llm: LLMService | None = None
        self._task_memory: TaskMemoryManager | None = None
        self._reranker: CrossEncoderReranker | None = None

        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if the client is connected."""
        return self._connected

    @property
    def llm(self) -> LLMService | None:
        """The LLM service, or None if no llm_provider is configured.

        Available after connect(). Use it directly for fact extraction,
        summarization, or completions, or call add_conversation() for the
        full extract-and-store pipeline.
        """
        return self._llm

    async def connect(self) -> None:
        """Connect to the database and initialize services.

        This method must be called before using any other methods
        unless using the async context manager.

        Raises:
            ConnectionError: If connection fails.
            ConfigurationError: If configuration is invalid.
        """
        if self._connected:
            logger.warning("Already connected")
            return

        logger.info("Connecting to Engram")

        try:
            # Initialize embedding service FIRST to get dimension. Provider
            # construction can block for seconds (sentence-transformers loads
            # model weights synchronously), so run it off the event loop.
            self._embedding = await asyncio.to_thread(
                EmbeddingService.from_settings, self._settings
            )
            try:
                embedding_dimension = self._embedding.dimension
            except ConfigurationError as e:
                if "Dimension not known" not in str(e):
                    raise
                await self._embedding.embed("engram dimension probe")
                embedding_dimension = self._embedding.dimension
            logger.info(f"Embedding dimension detected: {embedding_dimension}")

            # Initialize storage
            self._storage = PostgresStorage(self._settings)
            await self._storage.connect()

            # Initialize schema with auto-detected embedding dimension
            await self._storage.init_schema(embedding_dimension=embedding_dimension)

            # Initialize higher-level services
            self._memory_store = MemoryStore(
                self._storage, self._embedding, self._settings
            )
            self._graph = GraphTraversal(self._storage)
            self._sessions = SessionManager(self._storage)
            self._health = HealthChecker(self._storage, self._embedding)
            self._task_memory = TaskMemoryManager(self._storage)

            # Optional LLM service. A misconfigured provider (e.g. missing API key)
            # must not block core memory operations, so failures degrade to disabled.
            try:
                self._llm = LLMService.from_settings(self._settings)
                if self._llm is not None:
                    logger.info(f"LLM service enabled (model={self._llm.model})")
            except ConfigurationError as e:
                logger.warning(
                    f"LLM provider configured but could not be initialized: {e}. "
                    "LLM features (add_conversation) disabled."
                )
                self._llm = None

            self._connected = True
            logger.info("Connected to Engram successfully")
        except Exception:
            if self._embedding:
                with suppress(Exception):
                    await self._close_provider_resource(self._embedding.provider)
            if self._llm:
                with suppress(Exception):
                    await self._close_provider_resource(self._llm.provider)
            if self._storage:
                with suppress(Exception):
                    await self._storage.close()
            self._storage = None
            self._embedding = None
            self._memory_store = None
            self._graph = None
            self._sessions = None
            self._health = None
            self._llm = None
            self._task_memory = None
            self._connected = False
            raise

    async def close(self) -> None:
        """Close connections and cleanup resources.

        This method should be called when done using the client
        unless using the async context manager.
        """
        if not self._connected:
            return

        logger.info("Closing Engram connection")

        if self._embedding:
            await self._close_provider_resource(self._embedding.provider)
        if self._llm:
            await self._close_provider_resource(self._llm.provider)

        if self._storage:
            await self._storage.close()

        self._storage = None
        self._embedding = None
        self._memory_store = None
        self._graph = None
        self._sessions = None
        self._health = None
        self._llm = None
        self._task_memory = None
        self._connected = False

        logger.info("Engram connection closed")

    async def __aenter__(self) -> Engram:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.close()

    def _ensure_connected(self) -> None:
        """Ensure the client is connected."""
        if not self._connected:
            raise EngramError("Not connected. Call connect() first or use async with.")

    async def _close_provider_resource(self, obj: Any) -> None:
        """Close provider resources that expose sync or async close()."""
        close = getattr(obj, "close", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result

    def _policy_metadata(
        self,
        *,
        content: str,
        agent_id: AgentId,
        user_id: UserId | None,
        memory_type: MemoryType,
        metadata: Metadata | None,
    ) -> tuple[MemoryType, Metadata]:
        """Attach policy metadata used for recall, freshness, and tracing."""
        return self._memory_policy.apply_metadata(
            content=content,
            agent_id=agent_id,
            user_id=user_id,
            memory_type=memory_type,
            metadata=metadata,
        )

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    # Frequent English noise words at >= 3 chars that would skew chunk
    # relevance counting.
    _QUERY_STOPWORDS = frozenset(
        {"the", "and", "for", "with", "that", "this", "what", "how", "was", "are"}
    )

    def _query_terms(self, query: str) -> set[str]:
        """Lowercased significant query tokens for chunk relevance ranking.

        Three characters minimum: domain-critical short tokens like "p95",
        "SLA", or "NDA" must be matchable.
        """
        return {
            term.lower()
            for term in re.findall(r"[A-Za-z0-9_-]{3,}", query)
            if term.lower() not in self._QUERY_STOPWORDS
        }

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:24]

    def _classify_long_input_chunk(self, text: str, heading: str | None = None) -> str:
        """Classify a long-input chunk for storage and prompt assembly."""
        combined = f"{heading or ''}\n{text}".lower()
        if re.search(
            r"\b(section|clause|whereas|indemn|liabilit|governing law|agreement)\b",
            combined,
        ):
            return "legal_clause"
        if re.search(
            r"\b(must|shall|required|requirement|acceptance criteria)\b", combined
        ):
            return "requirement"
        if re.search(
            r"\b(never|always|do not|don't|constraint|limit|deadline)\b", combined
        ):
            return "constraint"
        if re.search(
            r"\b(decided|decision|approved|rejected|changed|correction)\b", combined
        ):
            return "decision"
        if re.search(
            r"\b(question|ask|answer|clarify|what should|how should)\b", combined
        ):
            return "question"
        if re.search(
            r"\b(tool result|test result|load test|pytest|api response|error rate|p95)\b",
            combined,
        ):
            return "tool_result"
        if re.search(r"\b(background|context|reference|source)\b", combined):
            return "source_doc"
        if re.search(r"\b(instruction|please|you are|your task)\b", combined):
            return "instruction"
        return "background_context"

    def _memory_type_for_chunk_kind(self, kind: str) -> MemoryType:
        mapping: dict[str, MemoryType] = {
            "instruction": "constraint",
            "requirement": "task",
            "constraint": "constraint",
            "legal_clause": "constraint",
            "source_doc": "semantic",
            "decision": "decision",
            "question": "task",
            "tool_result": "tool_result",
            "background_context": "semantic",
        }
        return mapping.get(kind, "semantic")

    def _split_long_input(
        self,
        text: str,
        *,
        max_chunk_tokens: int = 700,
        metadata: Metadata | None = None,
    ) -> list[LongInputChunk]:
        """Split long input by visible structure, with bounded chunk sizes."""
        max_chars = max(400, max_chunk_tokens * 4)
        parts: list[tuple[str | None, str, int, int]] = []
        heading: str | None = None
        buffer: list[str] = []
        buffer_start: int | None = None
        heading_re = re.compile(
            r"^\s{0,3}(#{1,6}\s+.+|[A-Z][A-Z0-9 _/-]{6,}|(?:Section|Clause)\s+[\w.-]+.*)$"
        )

        def flush() -> None:
            nonlocal buffer, buffer_start
            joined = "\n".join(buffer)
            lead = len(joined) - len(joined.lstrip())
            body = joined.strip()
            if body:
                # body is a contiguous slice of the source text starting at
                # buffer_start + lead; exact offsets keep anchors citable.
                start = (buffer_start or 0) + lead
                parts.append((heading, body, start, start + len(body)))
            buffer = []
            buffer_start = None

        for match in re.finditer(r".*(?:\n|$)", text):
            line = match.group(0)
            if not line and match.start() == len(text):
                break
            stripped = line.strip()
            if stripped and heading_re.match(stripped):
                flush()
                heading = stripped.lstrip("#").strip()
                continue
            if buffer_start is None and stripped:
                buffer_start = match.start()
            if stripped or buffer:
                buffer.append(line.rstrip("\n"))
            if buffer and sum(len(x) + 1 for x in buffer) >= max_chars:
                flush()
        flush()

        if not parts and text.strip():
            lead = len(text) - len(text.lstrip())
            body = text.strip()
            parts = [(None, body, lead, lead + len(body))]

        chunks: list[LongInputChunk] = []
        for heading, body, char_start, _char_end in parts:
            remaining = body
            offset = char_start
            while remaining:
                window = remaining[:max_chars]
                split_at = max(
                    window.rfind("\n\n"), window.rfind("\n"), window.rfind(". ")
                )
                if len(remaining) > max_chars and split_at > max_chars // 2:
                    window = window[: split_at + 1]
                # Track exactly how much of `remaining` this window consumes
                # and where the stripped piece sits inside it, so
                # char_start/char_end stay exact spans into the source text.
                consumed = len(window)
                lead = len(window) - len(window.lstrip())
                piece = window.strip()
                if piece:
                    start = offset + lead
                    end = start + len(piece)
                    kind = self._classify_long_input_chunk(piece, heading)
                    chunks.append(
                        LongInputChunk(
                            chunk_id=f"chunk_{len(chunks) + 1:04d}_{self._hash_text(piece)[:10]}",
                            index=len(chunks),
                            kind=kind,
                            heading=heading,
                            text=piece,
                            char_start=start,
                            char_end=end,
                            token_estimate=self._estimate_tokens(piece),
                            quote_hash=self._hash_text(piece),
                            metadata=metadata or {},
                        )
                    )
                remaining = remaining[consumed:]
                offset += consumed
        return chunks

    def _extract_chunk_facts(self, chunk: LongInputChunk) -> list[str]:
        """Deterministic fallback extraction for long-input chunks."""
        text = re.sub(r"\s+", " ", chunk.text).strip()
        if not text:
            return []
        sentences = re.split(r"(?<=[.!?])\s+", text)
        keepers: list[str] = []
        patterns = {
            "requirement": r"\b(must|shall|required|requires|acceptance criteria|target)\b",
            "constraint": r"\b(never|always|do not|don't|cannot|deadline|limit|constraint)\b",
            "legal_clause": r"\b(shall|must|liability|indemn|terminate|confidential|governing law)\b",
            "decision": r"\b(decided|decision|approved|rejected|changed|correction)\b",
            "tool_result": r"\b(result|p95|error rate|passed|failed|pytest|load test)\b",
            "question": r"\?",
            "instruction": r"\b(please|your task|you are|instruction)\b",
        }
        pattern = patterns.get(chunk.kind)
        for sentence in sentences:
            sentence = sentence.strip(" -\t")
            if len(sentence) < 12:
                continue
            if pattern and not re.search(pattern, sentence, re.I):
                continue
            keepers.append(sentence[:1000])
            if len(keepers) >= 4:
                break
        if keepers:
            return keepers
        return [text[:1000]]

    def _normalize_relative_time_notes(
        self,
        text: str,
        event: AgentEvent,
    ) -> list[str]:
        """Record absolute-date notes for common relative date phrases."""
        lowered = text.lower()
        created = getattr(event, "created_at", None)
        if created is None:
            return []
        notes: list[str] = []
        if "today" in lowered:
            notes.append(f"'today' refers to {created.date().isoformat()}")
        if "tomorrow" in lowered:
            from datetime import timedelta

            notes.append(
                f"'tomorrow' refers to {(created.date() + timedelta(days=1)).isoformat()}"
            )
        weekdays = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }
        for name, idx in weekdays.items():
            if f"next {name}" in lowered:
                from datetime import timedelta

                delta = (idx - created.weekday()) % 7
                if delta == 0:
                    delta = 7
                notes.append(
                    f"'next {name.title()}' refers to {(created.date() + timedelta(days=delta)).isoformat()}"
                )
        return notes

    # =========================================================================
    # Memory Operations
    # =========================================================================

    async def add(
        self,
        content: str,
        agent_id: AgentId,
        *,
        main_content: str | None = None,
        user_id: UserId | None = None,
        session_id: SessionId | None = None,
        memory_type: MemoryType = "semantic",
        metadata: Metadata | None = None,
    ) -> Memory:
        """Add a new memory.

        Two-column memory system:
        - content: The fact to store (embedded for hybrid search)
        - main_content: Optional conversation context (not embedded)

        All memories start with importance=0.5. Use reinforce() to boost
        importance when a memory proves useful.

        Args:
            content: The user fact to store (embedded for search).
            agent_id: ID of the agent this memory belongs to.
            main_content: Optional conversation context [USER]: msg\\n[AI]: summary.
            user_id: Optional user ID.
            session_id: Optional session ID.
            metadata: Additional key-value metadata.

        Returns:
            The created memory.

        Example:
            memory = await engram.add(
                content="User works in finance",
                main_content="[USER]: I work in finance\\n[AI]: Interesting field!",
                agent_id="my_agent",
                metadata={"source": "conversation"}
            )
        """
        self._ensure_connected()
        assert self._memory_store is not None

        memory_type, policy_metadata = self._policy_metadata(
            content=content,
            agent_id=agent_id,
            user_id=user_id,
            memory_type=memory_type,
            metadata=metadata,
        )

        # The store inserts and resolves conflict-key supersedes atomically.
        return await self._memory_store.add(
            MemoryCreate(
                content=content,
                main_content=main_content,
                agent_id=agent_id,
                user_id=user_id,
                session_id=session_id,
                memory_type=memory_type,
                metadata=policy_metadata,
            )
        )

    async def add_batch(
        self,
        memories: list[dict[str, Any]],
    ) -> list[Memory]:
        """Add multiple memories in a batch.

        More efficient than calling add() multiple times due to
        batch embedding. Only content (fact) is embedded, not main_content.

        Args:
            memories: List of memory dictionaries with keys:
                - content (required): The user fact (embedded for search)
                - agent_id (required): Agent ID
                - main_content (optional): Conversation context (not embedded)
                - user_id (optional): User ID
                - session_id (optional): Session ID
                - metadata (optional): Additional metadata

        Returns:
            List of created memories.

        Example:
            memories = await engram.add_batch([
                {"content": "Fact 1", "agent_id": "agent_1"},
                {"content": "Fact 2", "agent_id": "agent_1",
                 "main_content": "[USER]: I work...\\n[AI]: Got it!"},
            ])
        """
        self._ensure_connected()
        assert self._memory_store is not None

        creates: list[MemoryCreate] = []
        for m in memories:
            mtype, policy_metadata = self._policy_metadata(
                content=m["content"],
                agent_id=m["agent_id"],
                user_id=m.get("user_id"),
                memory_type=m.get("memory_type", "semantic"),
                metadata=m.get("metadata", {}),
            )
            creates.append(
                MemoryCreate(
                    content=m["content"],
                    main_content=m.get("main_content"),
                    agent_id=m["agent_id"],
                    user_id=m.get("user_id"),
                    session_id=m.get("session_id"),
                    memory_type=mtype,
                    metadata=policy_metadata,
                )
            )

        # The store inserts and resolves conflict-key supersedes atomically.
        return await self._memory_store.add_batch(creates)

    async def add_conversation(
        self,
        user_message: str,
        assistant_response: str,
        agent_id: AgentId,
        *,
        user_id: UserId | None = None,
        session_id: SessionId | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        conversation_summary: str | None = None,
        metadata: Metadata | None = None,
        search_limit: int = 10,
        update_summary: bool = True,
    ) -> list[Memory]:
        """Intelligently store memories from a conversation exchange.

        Runs the full LLM pipeline: extracts atomic facts from the exchange,
        compares each against existing memories, and applies the resulting
        ADD/UPDATE/DELETE/NOOP operations. The raw exchange is stored as
        main_content; only the extracted facts are embedded.

        Requires an LLM provider to be configured (llm_provider setting).

        Args:
            user_message: The user's message.
            assistant_response: The assistant's reply.
            agent_id: Agent this memory belongs to.
            user_id: Optional user ID.
            session_id: Optional session ID.
            conversation_history: Recent messages for temporal context.
            conversation_summary: Optional summary of earlier conversation.
            metadata: Additional metadata for created memories.
            search_limit: How many similar memories to consider per exchange.
            update_summary: When True and a session_id is given, roll the
                session's stored summary forward with this exchange (one extra
                LLM call). Set False to skip summary maintenance.

        Returns:
            Memories that were created or updated (NOOP/skipped facts excluded).

        Raises:
            EngramError: If no LLM provider is configured.
        """
        self._ensure_connected()
        if self._llm is None:
            raise EngramError(
                "add_conversation() requires an LLM provider. "
                "Set llm_provider (and its API key) in settings."
            )
        assert self._memory_store is not None
        assert self._sessions is not None

        # Resolve conversation summary: an explicit arg wins, otherwise load
        # the session's rolling summary (if a known session is provided).
        # summary_loaded_at is the CAS token for the roll-forward below;
        # it stays unset (False) when no session snapshot was loaded.
        effective_summary = conversation_summary
        summary_loaded_at: Any = False
        if effective_summary is None and session_id is not None:
            try:
                sess = await self._sessions.get(session_id)
                effective_summary = sess.summary
                summary_loaded_at = sess.summary_updated_at
            except SessionNotFoundError:
                effective_summary = None

        # Retrieve dedup/consolidation candidates per extracted fact, so facts
        # spanning multiple topics each see their own real matches (not just
        # memories similar to the raw user message). The 4th element is raw
        # cosine similarity for the duplicate check.
        async def _retrieve(fact: str) -> list[tuple[str, str, float, float]]:
            hits = await self.search(
                query=fact, agent_id=agent_id, user_id=user_id, limit=search_limit
            )
            return [
                (h.memory.memory_id, h.memory.content, h.score, h.semantic_score)
                for h in hits
            ]

        result = await self._llm.process_for_memory(
            user_message,
            assistant_response,
            [],
            conversation_history=conversation_history,
            conversation_summary=effective_summary,
            retrieve_for_fact=_retrieve,
            classify_types=True,
        )

        main_content = f"[USER]: {user_message}\n[AI]: {assistant_response}"
        affected: list[Memory] = []

        for op in result.operations:
            if op.operation == MemoryOperationType.ADD:
                affected.append(
                    await self.add(
                        content=op.content,
                        main_content=main_content,
                        agent_id=agent_id,
                        user_id=user_id,
                        session_id=session_id,
                        memory_type=op.memory_type,  # type: ignore[arg-type]
                        metadata=metadata,
                    )
                )
            elif (
                op.operation
                in (
                    MemoryOperationType.UPDATE,
                    MemoryOperationType.DELETE,
                )
                and op.target_id
            ):
                # Corrections create a new active revision while preserving the
                # old fact in the same lineage for audit/debugging.
                op_type, op_metadata = self._policy_metadata(
                    content=op.content,
                    agent_id=agent_id,
                    user_id=user_id,
                    memory_type=op.memory_type,  # type: ignore[arg-type]
                    metadata=metadata,
                )
                op_metadata["memory_type"] = op_type
                op_metadata["correction_operation"] = op.operation.value
                try:
                    affected.append(
                        await self.revise(
                            op.target_id,
                            content=op.content,
                            metadata=op_metadata,
                            reason=op.operation.value,
                        )
                    )
                except DuplicateMemoryError:
                    # The merged content already exists as another memory's
                    # fact; nothing new to store.
                    logger.debug(
                        f"Merged content for {op.target_id} already stored; skipping"
                    )
            # NOOP: nothing to store

        # Roll the session's summary forward with this exchange. Memories are
        # already written at this point, so summary maintenance is strictly
        # best-effort: failing the call here would make callers retry and
        # double-process a turn whose facts were stored.
        if update_summary and session_id is not None:
            try:
                new_summary = await self._llm.update_conversation_summary(
                    effective_summary, user_message, assistant_response
                )
                if summary_loaded_at is False:
                    # No session snapshot loaded (explicit summary given):
                    # plain last-writer update.
                    await self._sessions.update_summary(session_id, new_summary)
                else:
                    # Compare-and-set against the snapshot this summary was
                    # derived from; on conflict, rebase once on the fresh
                    # summary so the concurrent turn's information survives.
                    written = await self._sessions.try_update_summary(
                        session_id,
                        new_summary,
                        expected_updated_at=summary_loaded_at,
                    )
                    if written is None:
                        latest = await self._sessions.get(session_id)
                        rebased = await self._llm.update_conversation_summary(
                            latest.summary, user_message, assistant_response
                        )
                        await self._sessions.update_summary(session_id, rebased)
            except SessionNotFoundError:
                logger.debug(f"Session {session_id} not found; skipping summary update")
            except Exception as e:
                logger.warning(
                    f"Summary update failed for session {session_id}; "
                    f"memories were stored. Error: {e}"
                )

        return affected

    async def get(self, memory_id: MemoryId, *, track_access: bool = True) -> Memory:
        """Get a memory by ID.

        By default this updates the access timestamp and count (these feed
        time-decay ranking). Pass track_access=False for a pure read with no
        write — appropriate for read-replica routing or read-heavy paths.

        Args:
            memory_id: The memory ID to retrieve.
            track_access: When True (default) update access metadata; when False
                perform a plain read.

        Returns:
            The memory.

        Raises:
            MemoryNotFoundError: If memory doesn't exist.
        """
        self._ensure_connected()
        assert self._memory_store is not None

        return await self._memory_store.get(memory_id, track_access=track_access)

    async def update(
        self,
        memory_id: MemoryId,
        *,
        content: str | None = None,
        importance: float | None = None,
        metadata: Metadata | None = None,
    ) -> Memory:
        """Update an existing memory.

        Args:
            memory_id: The memory to update.
            content: New content (triggers re-embedding).
            importance: New importance score.
            metadata: Metadata to merge (not replace).

        Returns:
            The updated memory.

        Raises:
            MemoryNotFoundError: If memory doesn't exist.
        """
        self._ensure_connected()
        assert self._memory_store is not None

        return await self._memory_store.update(
            memory_id,
            MemoryUpdate(
                content=content,
                importance=importance,
                metadata=metadata,
            ),
        )

    async def revise(
        self,
        memory_id: MemoryId,
        *,
        content: str | None = None,
        importance: float | None = None,
        metadata: Metadata | None = None,
        reason: str | None = None,
    ) -> Memory:
        """Create a new active revision for an existing memory lineage.

        Use this for user corrections and LLM-extracted UPDATE/DELETE
        operations. ``update()`` remains the legacy in-place edit API.
        """
        self._ensure_connected()
        assert self._memory_store is not None

        return await self._memory_store.revise(
            memory_id,
            MemoryUpdate(
                content=content,
                importance=importance,
                metadata=metadata,
            ),
            reason=reason,
        )

    async def get_current(self, memory_id: MemoryId) -> Memory:
        """Return the current active head for a memory's lineage."""
        self._ensure_connected()
        assert self._memory_store is not None

        return await self._memory_store.get_current(memory_id)

    async def get_lineage(self, memory_id: MemoryId) -> MemoryLineage:
        """Return all revisions for a memory lineage, newest first."""
        self._ensure_connected()
        assert self._memory_store is not None

        return await self._memory_store.get_lineage(memory_id)

    async def explain_memory(self, memory_id: MemoryId) -> MemoryExplanation:
        """Return lineage and supersession details for one memory."""
        self._ensure_connected()
        assert self._memory_store is not None

        return await self._memory_store.explain_memory(memory_id)

    async def reinforce(
        self,
        memory_id: MemoryId,
        importance_boost: float = 0.1,
    ) -> Memory:
        """Reinforce a memory by boosting its importance.

        Args:
            memory_id: The memory to reinforce.
            importance_boost: Amount to increase importance (capped at 1.0).

        Returns:
            The reinforced memory.
        """
        self._ensure_connected()
        assert self._memory_store is not None

        return await self._memory_store.reinforce(memory_id, importance_boost)

    async def forget(self, memory_id: MemoryId) -> bool:
        """Delete a single memory.

        Args:
            memory_id: The memory to delete.

        Returns:
            True if deleted, False if not found.
        """
        self._ensure_connected()
        assert self._memory_store is not None

        return await self._memory_store.forget(memory_id)

    async def purge(
        self,
        agent_id: AgentId,
        user_id: UserId | None = None,
    ) -> int:
        """Delete all memories for an agent.

        Args:
            agent_id: The agent whose memories to delete.
            user_id: Optional user to filter by.

        Returns:
            Number of memories deleted.
        """
        self._ensure_connected()
        assert self._memory_store is not None

        return await self._memory_store.purge(agent_id, user_id)

    async def list_recent(
        self,
        agent_id: AgentId,
        user_id: UserId | None = None,
        limit: int = 10,
    ) -> list[Memory]:
        """List recent memories for an agent.

        Args:
            agent_id: The agent ID to filter by.
            user_id: Optional user ID to filter by.
            limit: Maximum number of results.

        Returns:
            List of memories ordered by creation time (newest first).
        """
        self._ensure_connected()
        assert self._memory_store is not None

        return await self._memory_store.list_recent(agent_id, user_id, limit)

    async def get_history(
        self,
        agent_id: AgentId,
        *,
        user_id: UserId | None = None,
        limit: int = 50,
        include_superseded: bool = True,
        memory_types: list[MemoryType] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[MemoryHistoryEvent]:
        """Return a user-facing memory timeline for an agent/user.

        The feed contains ``added``, ``revised``, and ``superseded`` events.
        Normal recall/search hides superseded memories, but history keeps them
        visible for audit, debugging, and user-facing "what changed?" views.
        """
        self._ensure_connected()
        assert self._memory_store is not None

        return await self._memory_store.get_history(
            agent_id,
            user_id,
            limit=limit,
            include_superseded=include_superseded,
            memory_types=memory_types,
            since=since,
            until=until,
        )

    # =========================================================================
    # Search Operations
    # =========================================================================

    async def search(
        self,
        query: str,
        agent_id: AgentId,
        *,
        user_id: UserId | None = None,
        limit: int = 10,
        min_score: float = 0.0,
        metadata_filter: Metadata | None = None,
        memory_types: list[MemoryType] | None = None,
        mode: SearchMode = "hybrid",
        rerank: bool = False,
    ) -> list[SearchResult]:
        """Search memories.

        The default hybrid mode combines vector similarity, keyword matching,
        time decay, and importance scoring. "semantic" is pure vector
        similarity; "keyword" is full-text only.

        Args:
            query: The search query text.
            agent_id: Filter by agent ID.
            user_id: Optional filter by user ID.
            limit: Maximum number of results.
            min_score: Minimum score threshold.
            metadata_filter: Optional JSONB containment filter; only memories
                whose metadata contains these key/values are returned.
            memory_types: Optional list of memory types to restrict to
                (e.g. ["episodic"] for events only).
            mode: "hybrid" (default), "semantic", or "keyword".
            rerank: When True, overfetch candidates and re-order them with a
                local cross-encoder before returning the top ``limit``.
                Requires the optional sentence-transformers dependency.

        Returns:
            List of search results with scores.

        Example:
            results = await engram.search(
                query="user preferences for UI",
                agent_id="my_agent",
                limit=5,
                metadata_filter={"source": "conversation"},
            )
            for result in results:
                print(f"{result.score:.2f}: {result.memory.content}")
        """
        self._ensure_connected()
        assert self._memory_store is not None

        fetch_limit = limit
        if rerank:
            fetch_limit = min(
                limit * self._settings.search_candidate_multiplier,
                self._settings.max_search_limit,
            )

        results = await self._memory_store.search(
            SearchQuery(
                query=query,
                agent_id=agent_id,
                user_id=user_id,
                limit=fetch_limit,
                min_score=min_score,
                metadata_filter=metadata_filter,
                memory_types=memory_types,
                mode=mode,
            )
        )
        if rerank:
            results = await self._get_reranker().rerank(query, results, top_k=limit)
        return results

    def _get_reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(
                self._settings.reranker_model,
                backend=self._settings.reranker_backend,
            )
        return self._reranker

    async def get_memories(
        self,
        agent_id: AgentId,
        *,
        user_id: UserId | None = None,
        session_id: SessionId | None = None,
        metadata_filter: Metadata | None = None,
        memory_types: list[MemoryType] | None = None,
        limit: int = 200,
    ) -> list[Memory]:
        """List active memories by filter, without relevance ranking.

        Unlike search(), this is a plain filtered read: no query, no scores,
        no access-count update. Useful for fetching a known group of memories,
        e.g. everything from one source conversation via metadata_filter.

        Args:
            agent_id: Agent to scope the read to.
            user_id: Optional user filter.
            session_id: Optional session filter.
            metadata_filter: Optional JSONB containment filter.
            memory_types: Optional list of memory types to restrict to.
            limit: Maximum number of memories returned.

        Returns:
            Matching memories, oldest first.
        """
        self._ensure_connected()
        assert self._memory_store is not None

        return await self._memory_store.list_memories(
            agent_id,
            user_id=user_id,
            session_id=session_id,
            metadata_filter=metadata_filter,
            memory_types=memory_types,
            limit=limit,
        )

    async def deep_search(
        self,
        query: str,
        agent_id: AgentId,
        *,
        user_id: UserId | None = None,
        limit: int = 10,
        min_score: float = 0.0,
        metadata_filter: Metadata | None = None,
        memory_types: list[MemoryType] | None = None,
        mode: SearchMode = "hybrid",
        n_queries: int = 4,
        rerank: bool = False,
    ) -> list[SearchResult]:
        """High-recall multi-query search (HyDE).

        Rewrites the query into several variants with the LLM, runs a hybrid
        search for each concurrently, and merges the results (deduped by
        memory_id, keeping the best score). Better recall for broad or
        aggregation-style questions than a single ``search()``.

        Falls back to a single ``search()`` when no LLM provider is configured.

        Args:
            query: The search query.
            agent_id: Agent to scope retrieval to.
            user_id: Optional user filter.
            limit: Maximum results to return.
            min_score: Minimum relevance score.
            metadata_filter: Optional metadata containment filter.
            memory_types: Optional list of memory types to restrict to.
            mode: Search mode used for each query variant.
            n_queries: Number of rewritten variants to add to the original.
            rerank: When True, re-order the merged candidate pool with the
                local cross-encoder against the original query before
                returning the top ``limit``. The cross-encoder scores all
                variants' results on one comparable scale.

        Returns:
            Merged, relevance-sorted results (at most ``limit``).
        """
        self._ensure_connected()

        queries = [query]
        if self._llm is not None:
            queries += await self._llm.expand_query(query, n_queries)

        # Dedupe queries (case-insensitive), preserving order
        seen: set[str] = set()
        unique_queries: list[str] = []
        for q in queries:
            key = q.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique_queries.append(q)

        # Tolerate per-variant failures: one transient error must not discard
        # the variants that succeeded. Only raise when every variant failed.
        gathered = await asyncio.gather(
            *[
                self.search(
                    q,
                    agent_id,
                    user_id=user_id,
                    limit=limit,
                    min_score=min_score,
                    metadata_filter=metadata_filter,
                    memory_types=memory_types,
                    mode=mode,
                )
                for q in unique_queries
            ],
            return_exceptions=True,
        )
        result_lists: list[list[SearchResult]] = []
        errors: list[BaseException] = []
        for item in gathered:
            if isinstance(item, BaseException):
                errors.append(item)
            else:
                result_lists.append(item)
        if errors and not result_lists:
            raise errors[0]
        for error in errors:
            logger.warning(f"deep_search variant failed (kept others): {error}")

        # Fuse the variants with Reciprocal Rank Fusion (rank-based), not raw
        # score. Scores from different query variants are not comparable: a
        # distractor that one HyDE variant ranks #1 can out-score — and evict —
        # evidence the original question ranked just inside its own top-k, so a
        # score-max merge can make deep_search recall *worse* than a single
        # search. RRF rewards cross-variant consensus and bounds any one
        # variant's influence. (The single-query hybrid path already fuses
        # semantic+keyword via RRF internally; this applies the same operator
        # across variants.)
        rrf_k = 60
        fused: dict[str, float] = {}
        representative: dict[str, SearchResult] = {}
        for results in result_lists:
            for rank, r in enumerate(results):
                memory_id = r.memory.memory_id
                fused[memory_id] = fused.get(memory_id, 0.0) + 1.0 / (rrf_k + rank + 1)
                current = representative.get(memory_id)
                if current is None or r.score > current.score:
                    representative[memory_id] = r

        merged = [
            representative[memory_id]
            for memory_id, _ in sorted(
                fused.items(), key=lambda item: item[1], reverse=True
            )
        ]
        if rerank:
            return await self._get_reranker().rerank(query, merged, top_k=limit)
        return merged[:limit]

    async def recall_critical(
        self,
        agent_id: AgentId,
        *,
        user_id: UserId | None = None,
        limit: int = 50,
        memory_types: list[MemoryType] | None = None,
    ) -> list[Memory]:
        """Deterministically recall active critical memories.

        This bypasses vector ranking entirely. Use it for facts that should
        not disappear from a long-running agent prompt just because a broad
        query ranked other facts higher.
        """
        self._ensure_connected()
        assert self._memory_store is not None

        return await self._memory_store.list_policy_memories(
            agent_id,
            user_id,
            limit=limit,
            critical_only=True,
            include_superseded=False,
            memory_types=memory_types,
        )

    async def trace_recall(
        self,
        query: str,
        agent_id: AgentId,
        *,
        user_id: UserId | None = None,
        limit: int = 20,
        min_score: float = 0.0,
        max_tokens: int = 2000,
        expected_terms: list[str] | None = None,
        use_deep_search: bool = True,
        memory_types: list[MemoryType] | None = None,
        token_counter: Callable[[str], int] | None = None,
    ) -> RecallTrace:
        """Build a prompt memory block with retrieval observability.

        Trace fields show whether memories were deterministically recalled,
        search-ranked, kept inside the prompt budget, trimmed, or hidden by
        conflict resolution as superseded.
        """
        self._ensure_connected()
        assert self._memory_store is not None

        count = token_counter or (lambda text: max(1, len(text) // 4))
        critical = await self.recall_critical(
            agent_id,
            user_id=user_id,
            limit=max(limit, 50),
            memory_types=memory_types,
        )
        search_results = (
            await self.deep_search(
                query,
                agent_id,
                user_id=user_id,
                limit=limit,
                min_score=min_score,
                memory_types=memory_types,
            )
            if use_deep_search
            else await self.search(
                query,
                agent_id,
                user_id=user_id,
                limit=limit,
                min_score=min_score,
                memory_types=memory_types,
            )
        )
        superseded = await self._memory_store.list_policy_memories(
            agent_id,
            user_id,
            limit=100,
            critical_only=False,
            include_superseded=True,
            memory_types=memory_types,
        )
        superseded = [
            memory
            for memory in superseded
            if memory.metadata.get("status") == "superseded"
        ]

        ranked: list[Memory] = []
        seen: set[str] = set()
        for memory in critical:
            if memory.memory_id not in seen:
                ranked.append(memory)
                seen.add(memory.memory_id)
        for result in search_results:
            if result.memory.memory_id not in seen:
                ranked.append(result.memory)
                seen.add(result.memory.memory_id)

        lines: list[str] = []
        kept_ids: list[str] = []
        trimmed_ids: list[str] = []
        used = 0
        for memory in ranked:
            slot = memory.metadata.get("critical_slot")
            prefix = f"[{memory.memory_type}]"
            if slot:
                prefix += f"[{slot}]"
            line = f"- {prefix} {memory.content}"
            line_cost = count(line)
            if used + line_cost > max_tokens:
                trimmed_ids.append(memory.memory_id)
                continue
            lines.append(line)
            kept_ids.append(memory.memory_id)
            used += line_cost

        context = "## Memory Recall\n" + "\n".join(lines) if lines else ""
        corpus = context.lower()
        missing_terms = [
            term for term in (expected_terms or []) if term.lower() not in corpus
        ]
        notes: list[str] = []
        if missing_terms:
            notes.append("expected_terms_missing_from_context")
        if trimmed_ids:
            notes.append("some_ranked_memories_trimmed_by_token_budget")
        if superseded:
            notes.append("superseded_memories_hidden_from_active_recall")

        return RecallTrace(
            query=query,
            agent_id=agent_id,
            user_id=user_id,
            critical_memory_ids=[m.memory_id for m in critical],
            search_memory_ids=[r.memory.memory_id for r in search_results],
            ranked_memory_ids=[m.memory_id for m in ranked],
            kept_memory_ids=kept_ids,
            trimmed_memory_ids=trimmed_ids,
            superseded_memory_ids=[m.memory_id for m in superseded],
            missing_expected_terms=missing_terms,
            context=context,
            notes=notes,
            metadata={
                "critical_count": len(critical),
                "search_count": len(search_results),
                "ranked_count": len(ranked),
                "kept_count": len(kept_ids),
                "trimmed_count": len(trimmed_ids),
                "superseded_count": len(superseded),
            },
        )

    async def get_context_block(
        self,
        query: str,
        agent_id: AgentId,
        *,
        user_id: UserId | None = None,
        session_id: SessionId | None = None,
        limit: int = 10,
        min_score: float = 0.0,
        max_tokens: int | None = None,
        header: str = "## Relevant memories",
        token_counter: Callable[[str], int] | None = None,
        memory_types: list[MemoryType] | None = None,
        group_by_type: bool = False,
        rerank: bool = False,
    ) -> str:
        """Assemble a compact, injection-ready memory block for a prompt.

        Searches memories for ``query`` and renders them as a deterministic,
        relevance-ordered bullet list ready to drop into a system/context
        prompt. When ``session_id`` is given and the session has a rolling
        summary, it is prepended. Ordering is stable so the block stays
        prompt-cache friendly across turns when memories are unchanged.

        Args:
            query: Query to retrieve relevant memories for.
            agent_id: Agent to scope retrieval to.
            user_id: Optional user filter.
            session_id: Optional session whose rolling summary is prepended.
            limit: Max memories to retrieve.
            min_score: Minimum relevance score.
            max_tokens: Optional token budget; lines are added in relevance
                order until the budget would be exceeded.
            header: Heading for the memory list ("" to omit).
            token_counter: text -> token count. Defaults to a ~4-chars-per-token
                heuristic (no tokenizer dependency).
            memory_types: Optional list of memory types to restrict to.
            group_by_type: When True, render memories grouped under
                Semantic / Episodic / Procedural headings instead of one list.
            rerank: When True, re-order retrieved memories with the local
                cross-encoder before applying the token budget.

        Returns:
            The rendered block, or "" if there is nothing to include.

        Example:
            block = await engram.get_context_block(
                query=user_message,
                agent_id="my_agent",
                session_id=sess.session_id,
                max_tokens=400,
            )
            system_prompt = f"{base_prompt}\\n\\n{block}" if block else base_prompt
        """
        self._ensure_connected()
        assert self._sessions is not None

        count = token_counter or (lambda text: max(1, len(text) // 4))
        sections: list[str] = []

        # Optional rolling summary from the session
        if session_id is not None:
            try:
                summary = (await self._sessions.get(session_id)).summary
            except SessionNotFoundError:
                summary = None
            if summary:
                sections.append(f"## Conversation summary\n{summary}")

        results = await self.search(
            query=query,
            agent_id=agent_id,
            user_id=user_id,
            limit=limit,
            min_score=min_score,
            memory_types=memory_types,
            rerank=rerank,
        )

        if results:
            # Budget includes already-built sections AND the heading that will
            # wrap the memory list — otherwise the block overruns max_tokens.
            used = count("\n\n".join(sections)) if sections else 0
            if header and not group_by_type:
                used += count(header)
            kept: list[SearchResult] = []
            for r in results:
                line = f"- {r.memory.content}"
                if max_tokens is not None and used + count(line) > max_tokens:
                    break
                kept.append(r)
                used += count(line)

            if kept and group_by_type:
                labels = {
                    "semantic": "## Semantic — user facts",
                    "episodic": "## Episodic — events",
                    "procedural": "## Procedural — rules",
                }
                for mtype in ("semantic", "episodic", "procedural"):
                    group = [r for r in kept if r.memory.memory_type == mtype]
                    if group:
                        body = "\n".join(f"- {r.memory.content}" for r in group)
                        sections.append(f"{labels[mtype]}\n{body}")
            elif kept:
                body = "\n".join(f"- {r.memory.content}" for r in kept)
                sections.append(f"{header}\n{body}" if header else body)

        return "\n\n".join(sections)

    # =========================================================================
    # Long-Running Task Memory Operations
    # =========================================================================

    async def start_task(
        self,
        goal: str,
        agent_id: AgentId,
        *,
        user_id: UserId | None = None,
        session_id: SessionId | None = None,
        metadata: Metadata | None = None,
    ) -> TaskRun:
        """Start a durable long-running agent task."""
        self._ensure_connected()
        assert self._task_memory is not None

        return await self._task_memory.start_task(
            goal=goal,
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
        )

    async def get_task(
        self,
        task_run_id: str,
        *,
        include_deleted: bool = False,
    ) -> TaskRun:
        """Get a task run by ID."""
        self._ensure_connected()
        assert self._task_memory is not None

        return await self._task_memory.get_task(
            task_run_id,
            include_deleted=include_deleted,
        )

    async def list_tasks(
        self,
        *,
        agent_id: AgentId | None = None,
        user_id: UserId | None = None,
        status: str | list[str] | None = None,
        limit: int = 100,
        include_deleted: bool = False,
    ) -> list[TaskRun]:
        """List task runs, typically to resume active long-running work."""
        self._ensure_connected()
        assert self._task_memory is not None

        return await self._task_memory.list_tasks(
            agent_id=agent_id,
            user_id=user_id,
            status=status,  # type: ignore[arg-type]
            limit=limit,
            include_deleted=include_deleted,
        )

    async def pause_task(
        self,
        task_run_id: str,
        *,
        outcome: str | None = None,
    ) -> TaskRun:
        """Mark a task as paused so it can be resumed later."""
        self._ensure_connected()
        assert self._task_memory is not None

        return await self._task_memory.set_task_status(
            task_run_id,
            "paused",
            outcome=outcome,
        )

    async def complete_task(
        self,
        task_run_id: str,
        *,
        outcome: str | None = None,
    ) -> TaskRun:
        """Mark a task as completed."""
        self._ensure_connected()
        assert self._task_memory is not None

        return await self._task_memory.set_task_status(
            task_run_id,
            "completed",
            outcome=outcome,
        )

    async def fail_task(
        self,
        task_run_id: str,
        *,
        outcome: str | None = None,
    ) -> TaskRun:
        """Mark a task as failed."""
        self._ensure_connected()
        assert self._task_memory is not None

        return await self._task_memory.set_task_status(
            task_run_id,
            "failed",
            outcome=outcome,
        )

    async def cancel_task(
        self,
        task_run_id: str,
        *,
        outcome: str | None = None,
    ) -> TaskRun:
        """Mark a task as cancelled."""
        self._ensure_connected()
        assert self._task_memory is not None

        return await self._task_memory.set_task_status(
            task_run_id,
            "cancelled",
            outcome=outcome,
        )

    async def soft_delete_task(self, task_run_id: str) -> TaskRun:
        """Soft-delete a task and hide its events from normal reads."""
        self._ensure_connected()
        assert self._task_memory is not None

        return await self._task_memory.soft_delete_task(task_run_id)

    async def record_event(
        self,
        *,
        agent_id: AgentId,
        role: EventRole,
        event_type: EventType,
        content: str = "",
        task_run_id: str | None = None,
        session_id: SessionId | None = None,
        user_id: UserId | None = None,
        payload: dict[str, Any] | None = None,
        metadata: Metadata | None = None,
    ) -> AgentEvent:
        """Append one event to the durable task/session ledger."""
        self._ensure_connected()
        assert self._task_memory is not None

        return await self._task_memory.record_event(
            EventCreate(
                task_run_id=task_run_id,
                session_id=session_id,
                agent_id=agent_id,
                user_id=user_id,
                role=role,
                event_type=event_type,
                content=content,
                payload=payload or {},
                metadata=metadata or {},
            )
        )

    async def list_events(
        self,
        *,
        task_run_id: str | None = None,
        session_id: SessionId | None = None,
        agent_id: AgentId | None = None,
        limit: int = 100,
        include_deleted: bool = False,
    ) -> list[AgentEvent]:
        """List recent events in chronological order."""
        self._ensure_connected()
        assert self._task_memory is not None

        return await self._task_memory.list_events(
            task_run_id=task_run_id,
            session_id=session_id,
            agent_id=agent_id,
            limit=limit,
            include_deleted=include_deleted,
        )

    async def redact_event(self, event_id: str) -> AgentEvent:
        """Redact an event payload and content while retaining audit metadata."""
        self._ensure_connected()
        assert self._task_memory is not None

        return await self._task_memory.redact_event(event_id)

    async def record_turn(
        self,
        task_run_id: str,
        user_message: str,
        assistant_response: str,
        *,
        agent_id: AgentId | None = None,
        user_id: UserId | None = None,
        session_id: SessionId | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        artifacts: list[dict[str, Any]] | None = None,
        metadata: Metadata | None = None,
        enqueue_processing: bool = True,
    ) -> list[AgentEvent]:
        """Record a full user/assistant turn and optionally enqueue ingestion."""
        self._ensure_connected()
        assert self._task_memory is not None

        task = await self._task_memory.get_task(task_run_id)
        resolved_agent_id = agent_id or task.agent_id
        resolved_user_id = user_id if user_id is not None else task.user_id
        resolved_session_id = session_id if session_id is not None else task.session_id
        event_metadata = metadata or {}

        def _create(
            role: EventRole,
            event_type: EventType,
            content: str,
            payload: dict[str, Any] | None = None,
        ) -> EventCreate:
            return EventCreate(
                task_run_id=task_run_id,
                session_id=resolved_session_id,
                agent_id=resolved_agent_id,
                user_id=resolved_user_id,
                role=role,
                event_type=event_type,
                content=content,
                payload=payload or {},
                metadata=event_metadata,
            )

        creates = [
            _create("user", "user_message", user_message),
            _create("assistant", "assistant_message", assistant_response),
        ]
        for call in tool_calls or []:
            creates.append(
                _create("tool", "tool_call", self._event_item_label(call), call)
            )
        for artifact in artifacts or []:
            creates.append(
                _create("agent", "artifact", self._event_item_label(artifact), artifact)
            )

        def _job_payload(events: list[AgentEvent]) -> dict[str, Any]:
            return {
                "task_run_id": task_run_id,
                "agent_id": resolved_agent_id,
                "user_id": resolved_user_id,
                "session_id": resolved_session_id,
                "user_message": user_message,
                "assistant_response": assistant_response,
                "user_event_id": events[0].event_id,
                "assistant_event_id": events[1].event_id,
                "event_ids": [event.event_id for event in events],
            }

        # Events and the ingestion job commit in one transaction: a crash in
        # between must not leave a recorded turn that never gets ingested.
        events, _job = await self._task_memory.record_events(
            creates,
            job_type="turn_ingest" if enqueue_processing else None,
            job_payload=_job_payload,
        )
        return events

    async def create_checkpoint(
        self,
        task_run_id: str,
        summary: str,
        *,
        completed_steps: list[str] | None = None,
        pending_steps: list[str] | None = None,
        decisions: list[str] | None = None,
        blockers: list[str] | None = None,
        artifacts: list[dict[str, Any]] | None = None,
        source_event_ids: list[str] | None = None,
        metadata: Metadata | None = None,
    ) -> TaskCheckpoint:
        """Create a compact state snapshot for future task continuation."""
        self._ensure_connected()
        assert self._task_memory is not None

        task = await self._task_memory.get_task(task_run_id)
        checkpoint = TaskCheckpoint(
            task_run_id=task_run_id,
            agent_id=task.agent_id,
            user_id=task.user_id,
            summary=summary,
            completed_steps=completed_steps or [],
            pending_steps=pending_steps or [],
            decisions=decisions or [],
            blockers=blockers or [],
            artifacts=artifacts or [],
            source_event_ids=source_event_ids or [],
            metadata=metadata or {},
        )
        return await self._task_memory.create_checkpoint(checkpoint)

    async def record_long_input(
        self,
        task_run_id: str,
        text: str,
        *,
        title: str | None = None,
        agent_id: AgentId | None = None,
        user_id: UserId | None = None,
        session_id: SessionId | None = None,
        metadata: Metadata | None = None,
        max_chunk_tokens: int = 700,
        extract_with_llm: bool = True,
        max_facts_per_chunk: int = 6,
    ) -> LongInputIngestionReport:
        """Record, chunk, anchor, and distill a long user input/document.

        The raw input is stored unchanged as a ledger event. Structured chunks
        are stored as artifact events. Extracted memories carry source anchors
        so later answers can trace back to exact chunk, character span, and
        content hash.
        """
        self._ensure_connected()
        assert self._task_memory is not None

        if not text.strip():
            raise EngramError("record_long_input() requires non-empty text")

        task = await self._task_memory.get_task(task_run_id)
        resolved_agent_id = agent_id or task.agent_id
        resolved_user_id = user_id if user_id is not None else task.user_id
        resolved_session_id = session_id if session_id is not None else task.session_id
        base_metadata: Metadata = {
            **(metadata or {}),
            "long_input": True,
            "title": title,
        }

        source_event = await self.record_event(
            task_run_id=task_run_id,
            session_id=resolved_session_id,
            agent_id=resolved_agent_id,
            user_id=resolved_user_id,
            role="user",
            event_type="user_message",
            content=text,
            payload={
                "kind": "long_input_source",
                "title": title,
                "char_count": len(text),
                "token_estimate": self._estimate_tokens(text),
                "quote_hash": self._hash_text(text),
            },
            metadata=base_metadata,
        )

        time_notes = self._normalize_relative_time_notes(text, source_event)
        chunks = [
            chunk.model_copy(
                update={
                    "source_event_id": source_event.event_id,
                    "metadata": {
                        **chunk.metadata,
                        **base_metadata,
                        "time_notes": time_notes,
                    },
                }
            )
            for chunk in self._split_long_input(
                text,
                max_chunk_tokens=max_chunk_tokens,
                metadata=base_metadata,
            )
        ]

        chunk_events: list[AgentEvent] = []
        chunk_type_counts: dict[str, int] = {}
        extracted_counts: dict[str, int] = {}
        # Collect every chunk's extracted facts and write them in a single
        # add_batch at the end: one transaction and one set of advisory locks
        # for the whole document, instead of a round-trip per fact.
        memory_creates: list[dict[str, Any]] = []

        for chunk in chunks:
            chunk_type_counts[chunk.kind] = chunk_type_counts.get(chunk.kind, 0) + 1
            chunk_event = await self.record_event(
                task_run_id=task_run_id,
                session_id=resolved_session_id,
                agent_id=resolved_agent_id,
                user_id=resolved_user_id,
                role="user",
                event_type="artifact",
                content=chunk.text,
                payload={
                    "kind": "long_input_chunk",
                    "chunk": chunk.model_dump(),
                },
                metadata={
                    **base_metadata,
                    "long_input_chunk": True,
                    "chunk_id": chunk.chunk_id,
                    "chunk_kind": chunk.kind,
                    "source_event_id": source_event.event_id,
                },
            )
            chunk_events.append(chunk_event)

            facts: list[str] = []
            if extract_with_llm and self._llm is not None:
                try:
                    facts = await self._llm.extract_document_facts(
                        chunk.text,
                        kind=chunk.kind,
                        heading=chunk.heading,
                        max_facts=max_facts_per_chunk,
                    )
                except Exception as exc:
                    logger.warning("Long-input LLM extraction failed: %s", exc)
                    facts = []
            if not facts:
                facts = self._extract_chunk_facts(chunk)

            facts = facts[:max_facts_per_chunk]
            extracted_counts[chunk.chunk_id] = len(facts)
            for fact in facts:
                memory_creates.append(
                    {
                        "content": fact,
                        "agent_id": resolved_agent_id,
                        "main_content": chunk.text,
                        "user_id": resolved_user_id,
                        "session_id": resolved_session_id,
                        "memory_type": self._memory_type_for_chunk_kind(chunk.kind),
                        "metadata": {
                            **base_metadata,
                            "source": "long_input",
                            "source_event_id": source_event.event_id,
                            "chunk_event_id": chunk_event.event_id,
                            "chunk_id": chunk.chunk_id,
                            "chunk_index": chunk.index,
                            "chunk_kind": chunk.kind,
                            "section": chunk.heading,
                            "char_start": chunk.char_start,
                            "char_end": chunk.char_end,
                            "quote_hash": chunk.quote_hash,
                            "time_notes": time_notes,
                        },
                    }
                )

        memories = await self.add_batch(memory_creates) if memory_creates else []
        memory_ids = [memory.memory_id for memory in memories]

        manifest = {
            "title": title,
            "source_event_id": source_event.event_id,
            "chunks": len(chunks),
            "chunk_type_counts": chunk_type_counts,
            "memory_count": len(memory_ids),
            "time_notes": time_notes,
            "chunk_ids": [chunk.chunk_id for chunk in chunks],
        }
        summary = (
            f"Long input recorded"
            f"{f': {title}' if title else ''}. "
            f"{len(chunks)} chunks, {len(memory_ids)} anchored memories."
        )
        checkpoint = await self.create_checkpoint(
            task_run_id,
            summary,
            pending_steps=[
                "Use source chunks before summaries for legal or exact-document answers",
                "Use trace_recall to verify required memories are not missing",
            ],
            artifacts=[
                {
                    "type": "long_input_manifest",
                    "title": title,
                    "source_event_id": source_event.event_id,
                    "chunk_count": len(chunks),
                    "memory_count": len(memory_ids),
                }
            ],
            source_event_ids=[source_event.event_id]
            + [event.event_id for event in chunk_events],
            metadata={
                "source": "long_input",
                "long_input_manifest": manifest,
            },
        )

        return LongInputIngestionReport(
            task_run_id=task_run_id,
            source_event_id=source_event.event_id,
            chunks=chunks,
            memory_ids=memory_ids,
            checkpoint_id=checkpoint.checkpoint_id,
            manifest=manifest,
            trace={
                "chunk_event_ids": [event.event_id for event in chunk_events],
                "extracted_counts": extracted_counts,
                "time_notes": time_notes,
            },
            metadata=base_metadata,
        )

    async def build_long_input_context(
        self,
        task_run_id: str,
        *,
        query: str,
        max_tokens: int = 4000,
        source_chunk_limit: int = 6,
        expected_terms: list[str] | None = None,
        token_counter: Callable[[str], int] | None = None,
    ) -> LongInputContextResult:
        """Build answer context from critical memory plus anchored source chunks."""
        self._ensure_connected()
        assert self._task_memory is not None

        task = await self._task_memory.get_task(task_run_id)
        count = token_counter or self._estimate_tokens
        trace = await self.trace_recall(
            query,
            task.agent_id,
            user_id=task.user_id,
            limit=30,
            max_tokens=max(500, int(max_tokens * 0.45)),
            expected_terms=expected_terms,
        )

        events = await self.list_events(task_run_id=task_run_id, limit=500)
        chunk_events = [
            event
            for event in events
            if event.metadata.get("long_input_chunk")
            or event.payload.get("kind") == "long_input_chunk"
        ]
        query_terms = self._query_terms(query)

        def relevance(event: AgentEvent) -> int:
            haystack = f"{event.content} {event.payload}".lower()
            return sum(1 for term in query_terms if term in haystack)

        ranked_chunks = sorted(
            chunk_events,
            key=lambda event: (relevance(event), event.created_at),
            reverse=True,
        )
        selected_chunks = ranked_chunks[:source_chunk_limit]

        sections: list[str] = []
        if trace.context:
            sections.append(trace.context)

        if selected_chunks:
            lines = ["## Source Chunks"]
            for event in selected_chunks:
                chunk = event.payload.get("chunk", {})
                heading = chunk.get("heading") or chunk.get("kind") or "chunk"
                char_start = chunk.get("char_start")
                char_end = chunk.get("char_end")
                quote_hash = chunk.get("quote_hash")
                anchor = (
                    f"chunk_id={chunk.get('chunk_id')}; "
                    f"chars={char_start}-{char_end}; hash={quote_hash}"
                )
                lines.append(f"### {heading}\n[{anchor}]\n{event.content}")
            sections.append("\n\n".join(lines))

        checkpoints = await self._task_memory.list_checkpoints(task_run_id, limit=3)
        manifest = next(
            (
                checkpoint.metadata.get("long_input_manifest")
                for checkpoint in checkpoints
                if checkpoint.metadata.get("long_input_manifest")
            ),
            None,
        )
        if manifest:
            sections.append(
                "## Long Input Manifest\n"
                f"- title: {manifest.get('title')}\n"
                f"- chunks: {manifest.get('chunks')}\n"
                f"- anchored memories: {manifest.get('memory_count')}\n"
                f"- time notes: {manifest.get('time_notes') or []}"
            )

        kept_sections: list[str] = []
        used = 0
        trimmed_sections: list[str] = []
        for section in sections:
            cost = count(section)
            if used + cost > max_tokens:
                trimmed_sections.append(
                    section.splitlines()[0] if section else "section"
                )
                continue
            kept_sections.append(section)
            used += cost

        text = "\n\n".join(kept_sections)
        missing_terms = [
            term for term in (expected_terms or []) if term.lower() not in text.lower()
        ]

        return LongInputContextResult(
            text=text,
            token_estimate=count(text) if text else 0,
            trace={
                "recall": trace.model_dump(),
                "source_chunk_event_ids": [event.event_id for event in selected_chunks],
                "trimmed_sections": trimmed_sections,
                "missing_expected_terms": missing_terms,
            },
            metadata={
                "task_run_id": task_run_id,
                "source_chunks_considered": len(chunk_events),
                "source_chunks_kept": len(selected_chunks),
            },
        )

    async def build_context(
        self,
        task_run_id: str,
        *,
        query: str = "",
        max_tokens: int = 200000,
        token_counter: Callable[[str], int] | None = None,
        recent_event_limit: int = 40,
        memory_limit: int = 25,
        checkpoint_limit: int = 3,
        include_graph: bool = True,
    ) -> ContextBuildResult:
        """Build a deterministic context block for resuming a long task."""
        self._ensure_connected()
        assert self._task_memory is not None

        task = await self._task_memory.get_task(task_run_id)
        builder = ContextBuilder(self._task_memory, self.search, self.traverse)
        return await builder.build(
            task_run_id=task_run_id,
            agent_id=task.agent_id,
            user_id=task.user_id,
            options=ContextBuildOptions(
                query=query,
                max_tokens=max_tokens,
                recent_event_limit=recent_event_limit,
                memory_limit=memory_limit,
                checkpoint_limit=checkpoint_limit,
                include_graph=include_graph,
            ),
            token_counter=token_counter,
        )

    async def process_memory_jobs(self, *, limit: int = 10) -> list[MemoryJob]:
        """Process queued task memory jobs using the configured services."""
        self._ensure_connected()
        assert self._task_memory is not None

        claimed = await self._task_memory.claim_jobs(limit=limit)
        processed: list[MemoryJob] = []

        for job in claimed:
            try:
                if job.job_type == "turn_ingest":
                    await self._process_turn_ingest_job(job)
                processed.append(await self._task_memory.complete_job(job.job_id))
            except Exception as exc:
                logger.exception("Failed to process memory job %s", job.job_id)
                processed.append(await self._task_memory.fail_job(job.job_id, str(exc)))

        return processed

    async def run_memory_worker(
        self,
        *,
        batch_size: int = 10,
        interval_seconds: float = 1.0,
        stop_event: asyncio.Event | None = None,
        max_iterations: int | None = None,
    ) -> int:
        """Run the durable memory job processor until stopped.

        Returns the number of jobs claimed and finalized by this worker.
        """
        self._ensure_connected()

        processed_count = 0
        iterations = 0
        while stop_event is None or not stop_event.is_set():
            if max_iterations is not None and iterations >= max_iterations:
                break
            iterations += 1

            processed = await self.process_memory_jobs(limit=batch_size)
            processed_count += len(processed)
            if processed:
                continue

            if stop_event is not None:
                with suppress(TimeoutError):
                    await asyncio.wait_for(
                        stop_event.wait(),
                        timeout=interval_seconds,
                    )
            else:
                await asyncio.sleep(interval_seconds)

        return processed_count

    async def _process_turn_ingest_job(self, job: MemoryJob) -> None:
        assert self._task_memory is not None
        payload = job.payload
        task_run_id = str(payload["task_run_id"])
        task = await self._task_memory.get_task(task_run_id)
        user_message = str(payload.get("user_message") or "")
        assistant_response = str(payload.get("assistant_response") or "")
        session_id = payload.get("session_id") or task.session_id
        user_id = payload.get("user_id") or task.user_id
        event_ids = [
            str(event_id)
            for event_id in payload.get("event_ids", [])
            if event_id is not None
        ]

        if self._llm is not None:
            await self.add_conversation(
                user_message,
                assistant_response,
                task.agent_id,
                user_id=user_id,
                session_id=session_id,
                metadata={"source": "task_memory", "task_run_id": task_run_id},
                update_summary=session_id is not None,
            )

        latest = await self._task_memory.latest_checkpoint(task_run_id)
        previous = latest.summary if latest is not None else None
        summary = await self._summarize_task_turn(
            previous,
            user_message,
            assistant_response,
        )
        await self.create_checkpoint(
            task_run_id,
            summary,
            source_event_ids=event_ids,
            metadata={"source": "memory_job", "job_id": job.job_id},
        )

    async def _summarize_task_turn(
        self,
        previous_summary: str | None,
        user_message: str,
        assistant_response: str,
    ) -> str:
        if self._llm is not None:
            return await self._llm.update_conversation_summary(
                previous_summary,
                user_message,
                assistant_response,
                max_length=250,
            )

        lines: list[str] = []
        if previous_summary:
            lines.append(previous_summary)
        lines.append(f"User: {user_message}")
        lines.append(f"Assistant: {assistant_response}")
        return "\n".join(lines)[-2000:]

    def _event_item_label(self, item: dict[str, Any]) -> str:
        for key in ("name", "title", "path", "id", "type"):
            value = item.get(key)
            if value:
                return str(value)
        return str(item)

    # =========================================================================
    # Graph Operations
    # =========================================================================

    async def relate(
        self,
        source_id: MemoryId,
        target_id: MemoryId,
        relation_type: RelationType = "related_to",
        weight: float = 1.0,
        metadata: Metadata | None = None,
    ) -> None:
        """Create a relation between two memories.

        Args:
            source_id: Source memory ID.
            target_id: Target memory ID.
            relation_type: Type of relation.
            weight: Relation weight (0.0 to 1.0).
            metadata: Optional relation metadata.

        Raises:
            MemoryNotFoundError: If either memory doesn't exist.
        """
        self._ensure_connected()
        assert self._graph is not None

        await self._graph.relate(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            metadata=metadata,
        )

    async def traverse(
        self,
        start_memory_id: MemoryId,
        max_depth: int = 3,
        direction: str = "outbound",
        relation_types: list[RelationType] | None = None,
        min_weight: float = 0.0,
        limit: int = 50,
    ) -> list[TraversalResult]:
        """Traverse the memory graph from a starting point.

        Uses recursive CTEs for efficient multi-hop traversal.

        Args:
            start_memory_id: The memory to start from.
            max_depth: Maximum traversal depth.
            direction: Direction (outbound, inbound, any).
            relation_types: Optional filter by relation types.
            min_weight: Minimum relation weight to follow.
            limit: Maximum results.

        Returns:
            List of traversal results with depth and path info.

        Example:
            results = await engram.traverse(
                start_memory_id="mem_abc123",
                max_depth=2,
                direction="outbound",
            )
            for r in results:
                print(f"Depth {r.depth}: {r.content}")
        """
        self._ensure_connected()
        assert self._graph is not None

        return await self._graph.traverse(
            TraversalQuery(
                start_memory_id=start_memory_id,
                max_depth=max_depth,
                direction=direction,
                relation_types=relation_types,
                min_weight=min_weight,
                limit=limit,
            )
        )

    async def traverse_many(
        self,
        start_memory_ids: list[MemoryId],
        *,
        max_depth: int = 2,
        direction: str = "any",
        relation_types: list[RelationType] | None = None,
        min_weight: float = 0.0,
        limit_per_seed: int = 25,
        total_limit: int = 100,
        skip_missing: bool = True,
    ) -> list[TraversalResult]:
        """Traverse the graph from multiple seed memories.

        Useful for prompt assembly, where retrieval often returns several
        relevant memories and the prompt should include their shared graph.
        """
        self._ensure_connected()
        assert self._graph is not None

        return await self._graph.traverse_many(
            start_memory_ids,
            max_depth=max_depth,
            direction=direction,
            relation_types=relation_types,
            min_weight=min_weight,
            limit_per_seed=limit_per_seed,
            total_limit=total_limit,
            skip_missing=skip_missing,
        )

    def render_graph_context(
        self,
        results: list[TraversalResult],
        *,
        max_tokens: int | None = None,
        token_counter: Callable[[str], int] | None = None,
        include_paths: bool = False,
        header: str = "## Related memory graph",
    ) -> str:
        """Render traversal results into a prompt-ready context block."""
        self._ensure_connected()
        assert self._graph is not None

        return self._graph.render_context(
            results,
            max_tokens=max_tokens,
            token_counter=token_counter,
            include_paths=include_paths,
            header=header,
        )

    # =========================================================================
    # Session Operations
    # =========================================================================

    @asynccontextmanager
    async def session(
        self,
        agent_id: AgentId,
        user_id: UserId | None = None,
        metadata: Metadata | None = None,
    ) -> AsyncIterator[Session]:
        """Create a session context manager.

        The session is automatically ended when the context exits.

        Args:
            agent_id: The agent ID.
            user_id: Optional user ID.
            metadata: Optional session metadata.

        Yields:
            The active session.

        Example:
            async with engram.session(agent_id="my_agent") as sess:
                memory = await engram.add(
                    content="In-session memory",
                    agent_id="my_agent",
                    session_id=sess.session_id,
                )
        """
        self._ensure_connected()
        assert self._sessions is not None

        async with self._sessions.session(agent_id, user_id, metadata) as sess:
            yield sess

    # =========================================================================
    # Health Operations
    # =========================================================================

    async def health_check(self) -> dict[str, Any]:
        """Perform a comprehensive health check.

        Returns:
            Dictionary with health status and component details.

        Example:
            status = await engram.health_check()
            if status["status"] == "healthy":
                print("All systems operational")
        """
        self._ensure_connected()
        assert self._health is not None

        return await self._health.check()
