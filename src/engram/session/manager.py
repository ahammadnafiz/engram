"""Session manager for Engram.

This module provides session management with async context manager support.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Any, AsyncIterator

from engram.core._types import AgentId, SessionId, UserId
from engram.core.exceptions import SessionNotFoundError, StorageError
from engram.session.models import Session, SessionCreate

if TYPE_CHECKING:
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
                json.dumps(session.metadata),
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
                started_at, ended_at, metadata
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
                started_at, ended_at, metadata
            """,
            session_id,
        )

        if row is None:
            raise SessionNotFoundError(session_id)

        logger.debug(f"Ended session {session_id}")
        return self._row_to_session(row)

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
                started_at, ended_at, metadata
            FROM agent_sessions
            WHERE {" AND ".join(conditions)}
            ORDER BY started_at DESC
            LIMIT ${param_idx}
        """

        rows = await self._storage.fetchall(query, *params)
        return [self._row_to_session(row) for row in rows]

    async def list_by_agent(
        self,
        agent_id: AgentId,
        include_ended: bool = True,
        limit: int = 50,
    ) -> list[Session]:
        """List sessions for an agent.

        Args:
            agent_id: The agent ID to filter by.
            include_ended: Whether to include ended sessions.
            limit: Maximum results.

        Returns:
            List of sessions.
        """
        if include_ended:
            rows = await self._storage.fetchall(
                """
                SELECT 
                    session_id, agent_id, user_id,
                    started_at, ended_at, metadata
                FROM agent_sessions
                WHERE agent_id = $1
                ORDER BY started_at DESC
                LIMIT $2
                """,
                agent_id,
                limit,
            )
        else:
            rows = await self._storage.fetchall(
                """
                SELECT 
                    session_id, agent_id, user_id,
                    started_at, ended_at, metadata
                FROM agent_sessions
                WHERE agent_id = $1 AND ended_at IS NULL
                ORDER BY started_at DESC
                LIMIT $2
                """,
                agent_id,
                limit,
            )

        return [self._row_to_session(row) for row in rows]

    @asynccontextmanager
    async def session(
        self,
        agent_id: AgentId,
        user_id: UserId | None = None,
        metadata: dict | None = None,
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
            metadata=metadata or {},
        )
