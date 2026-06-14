"""Session manager for Engram.

This module provides session management with async context manager support.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from engram.core.exceptions import SessionNotFoundError, StorageError
from engram.core.serialization import json_dumps
from engram.session.models import Session, SessionCreate

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from datetime import datetime

    from engram.core._types import AgentId, SessionId, UserId
    from engram.storage.postgres import PostgresStorage

logger = logging.getLogger(__name__)


class SessionManager:
    """Manager for agent sessions.

    This class handles session lifecycle including creation, retrieval,
    and termination. It provides async context manager support for
    automatic session cleanup.

    Example:
        manager = SessionManager(storage)

        # Manual session management
        session = await manager.create(SessionCreate(
            agent_id="agent_123",
            user_id="user_456",
        ))
        # ... do work ...
        await manager.end(session.session_id)

        # Or use context manager
        async with manager.session(agent_id="agent_123") as session:
            # Session is automatically ended when context exits
            pass
    """

    def __init__(self, storage: PostgresStorage) -> None:
        """Initialize session manager.

        Args:
            storage: PostgreSQL storage backend.
        """
        self._storage = storage

    async def _ensure_agent_exists(self, agent_id: AgentId) -> None:
        """Ensure agent exists, creating if necessary."""
        await self._storage.execute(
            """
            INSERT INTO agents (agent_id, name)
            VALUES ($1, $2)
            ON CONFLICT (agent_id) DO NOTHING
            """,
            agent_id,
            agent_id,  # Use agent_id as default name
        )

    async def _ensure_user_exists(self, user_id: UserId) -> None:
        """Ensure user exists, creating if necessary."""
        await self._storage.execute(
            """
            INSERT INTO users (user_id)
            VALUES ($1)
            ON CONFLICT (user_id) DO NOTHING
            """,
            user_id,
        )

    async def create(self, create: SessionCreate) -> Session:
        """Create a new session.

        Args:
            create: Session creation parameters.

        Returns:
            The created session.

        Raises:
            StorageError: If session creation fails.
        """
        session = Session(
            agent_id=create.agent_id,
            user_id=create.user_id,
            metadata=create.metadata,
        )

        try:
            # Auto-create agent and user so the session FKs are satisfied
            await self._ensure_agent_exists(create.agent_id)
            if create.user_id:
                await self._ensure_user_exists(create.user_id)

            await self._storage.execute(
                """
                INSERT INTO agent_sessions (
                    session_id, agent_id, user_id,
                    started_at, metadata
                ) VALUES ($1, $2, $3, $4, $5)
                """,
                session.session_id,
                session.agent_id,
                session.user_id,
                session.started_at,
                json_dumps(session.metadata),
            )

            logger.debug(f"Created session {session.session_id}")
            return session

        except Exception as e:
            raise StorageError(f"Failed to create session: {e}") from e

    async def get(self, session_id: SessionId) -> Session:
        """Get a session by ID.

        Args:
            session_id: The session ID to retrieve.

        Returns:
            The session object.

        Raises:
            SessionNotFoundError: If session doesn't exist.
        """
        row = await self._storage.fetchone(
            """
            SELECT
                session_id, agent_id, user_id,
                started_at, ended_at, summary, summary_updated_at, metadata
            FROM agent_sessions
            WHERE session_id = $1
            """,
            session_id,
        )

        if row is None:
            raise SessionNotFoundError(session_id)

        return self._row_to_session(row)

    async def end(self, session_id: SessionId) -> Session:
        """End an active session.

        Args:
            session_id: The session to end.

        Returns:
            The ended session.

        Raises:
            SessionNotFoundError: If session doesn't exist.
        """
        row = await self._storage.fetchone(
            """
            UPDATE agent_sessions
            SET ended_at = NOW()
            WHERE session_id = $1
            RETURNING
                session_id, agent_id, user_id,
                started_at, ended_at, summary, summary_updated_at, metadata
            """,
            session_id,
        )

        if row is None:
            raise SessionNotFoundError(session_id)

        logger.debug(f"Ended session {session_id}")
        return self._row_to_session(row)

    async def update_summary(self, session_id: SessionId, summary: str) -> Session:
        """Update a session's rolling conversation summary.

        Args:
            session_id: The session to update.
            summary: The new summary text.

        Returns:
            The updated session.

        Raises:
            SessionNotFoundError: If session doesn't exist.
        """
        row = await self._storage.fetchone(
            """
            UPDATE agent_sessions
            SET summary = $2, summary_updated_at = NOW()
            WHERE session_id = $1
            RETURNING
                session_id, agent_id, user_id,
                started_at, ended_at, summary, summary_updated_at, metadata
            """,
            session_id,
            summary,
        )

        if row is None:
            raise SessionNotFoundError(session_id)

        logger.debug(f"Updated summary for session {session_id}")
        return self._row_to_session(row)

    async def try_update_summary(
        self,
        session_id: SessionId,
        summary: str,
        *,
        expected_updated_at: datetime | None,
    ) -> Session | None:
        """Compare-and-set update of the rolling summary.

        Writes only if summary_updated_at still equals expected_updated_at
        (None = no summary written yet). Two concurrent turns both roll the
        summary forward from the same snapshot; without this guard the
        second write silently discards the first turn's information.

        Args:
            session_id: The session to update.
            summary: The new summary text.
            expected_updated_at: The summary_updated_at value the new summary
                was derived from.

        Returns:
            The updated session, or None when a concurrent writer won.

        Raises:
            SessionNotFoundError: If session doesn't exist.
        """
        row = await self._storage.fetchone(
            """
            UPDATE agent_sessions
            SET summary = $2, summary_updated_at = NOW()
            WHERE session_id = $1
              AND summary_updated_at IS NOT DISTINCT FROM $3
            RETURNING
                session_id, agent_id, user_id,
                started_at, ended_at, summary, summary_updated_at, metadata
            """,
            session_id,
            summary,
            expected_updated_at,
        )
        if row is not None:
            logger.debug(f"Updated summary for session {session_id} (CAS)")
            return self._row_to_session(row)

        exists = await self._storage.fetchval(
            "SELECT EXISTS(SELECT 1 FROM agent_sessions WHERE session_id = $1)",
            session_id,
        )
        if not exists:
            raise SessionNotFoundError(session_id)
        return None

    async def list_active(
        self,
        agent_id: AgentId | None = None,
        user_id: UserId | None = None,
        limit: int = 50,
    ) -> list[Session]:
        """List active (non-ended) sessions.

        Args:
            agent_id: Optional filter by agent.
            user_id: Optional filter by user.
            limit: Maximum results.

        Returns:
            List of active sessions.
        """
        conditions = ["ended_at IS NULL"]
        params: list[Any] = []
        param_idx = 1

        if agent_id:
            conditions.append(f"agent_id = ${param_idx}")
            params.append(agent_id)
            param_idx += 1

        if user_id:
            conditions.append(f"user_id = ${param_idx}")
            params.append(user_id)
            param_idx += 1

        params.append(limit)

        query = f"""
            SELECT
                session_id, agent_id, user_id,
                started_at, ended_at, summary, summary_updated_at, metadata
            FROM agent_sessions
            WHERE {" AND ".join(conditions)}
            ORDER BY started_at DESC
            LIMIT ${param_idx}
        """

        rows = await self._storage.fetchall(query, *params)
        return [self._row_to_session(row) for row in rows]

    @asynccontextmanager
    async def session(
        self,
        agent_id: AgentId,
        user_id: UserId | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[Session]:
        """Context manager for automatic session lifecycle.

        Creates a session on entry and ends it on exit, even if
        an exception occurs.

        Args:
            agent_id: The agent ID.
            user_id: Optional user ID.
            metadata: Optional session metadata.

        Yields:
            The active session.

        Example:
            async with manager.session(agent_id="agent_123") as session:
                # Session is active here
                print(f"Session: {session.session_id}")
            # Session is automatically ended here
        """
        session = await self.create(
            SessionCreate(
                agent_id=agent_id,
                user_id=user_id,
                metadata=metadata or {},
            )
        )

        try:
            yield session
        finally:
            try:
                await self.end(session.session_id)
            except Exception as e:
                logger.warning(f"Failed to end session {session.session_id}: {e}")

    def _row_to_session(self, row: Any) -> Session:
        """Convert a database row to a Session object."""
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        return Session(
            session_id=row["session_id"],
            agent_id=row["agent_id"],
            user_id=row["user_id"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            summary=row.get("summary"),
            summary_updated_at=row.get("summary_updated_at"),
            metadata=metadata or {},
        )
