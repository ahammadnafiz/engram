"""Unit tests for session management."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestSessionManager:
    """Tests for SessionManager."""

    @pytest.fixture
    def mock_storage(self) -> MagicMock:
        storage = MagicMock()
        storage.execute = AsyncMock(return_value="INSERT 1")
        storage.fetchone = AsyncMock()
        storage.fetchall = AsyncMock(return_value=[])
        return storage

    @pytest.mark.asyncio
    async def test_create_session(self, mock_storage: MagicMock) -> None:
        """Test creating a new session."""
        from engram.session.manager import SessionManager
        from engram.session.models import SessionCreate

        mock_storage.fetchone = AsyncMock(
            return_value={
                "session_id": "sess_123",
                "agent_id": "agent_1",
                "user_id": "user_1",
                "started_at": datetime.now(timezone.utc),
                "ended_at": None,
                "metadata": "{}",
            }
        )

        manager = SessionManager(storage=mock_storage)

        create = SessionCreate(agent_id="agent_1", user_id="user_1")
        session = await manager.create(create)

        assert session.agent_id == "agent_1"
        assert session.user_id == "user_1"
        assert session.is_active

    @pytest.mark.asyncio
    async def test_end_session(self, mock_storage: MagicMock) -> None:
        """Test ending a session."""
        from engram.session.manager import SessionManager

        mock_storage.fetchone = AsyncMock(
            return_value={
                "session_id": "sess_123",
                "agent_id": "agent_1",
                "user_id": None,
                "started_at": datetime.now(timezone.utc),
                "ended_at": datetime.now(timezone.utc),
                "metadata": "{}",
            }
        )

        manager = SessionManager(storage=mock_storage)

        session = await manager.end("sess_123")

        assert session is not None
        assert not session.is_active
        mock_storage.fetchone.assert_called()

    @pytest.mark.asyncio
    async def test_get_session(self, mock_storage: MagicMock) -> None:
        """Test getting an existing session."""
        from engram.session.manager import SessionManager

        mock_storage.fetchone = AsyncMock(
            return_value={
                "session_id": "sess_123",
                "agent_id": "agent_1",
                "user_id": "user_1",
                "started_at": datetime.now(timezone.utc),
                "ended_at": None,
                "metadata": '{"key": "value"}',
            }
        )

        manager = SessionManager(storage=mock_storage)

        session = await manager.get("sess_123")

        assert session is not None
        assert session.session_id == "sess_123"
        assert session.metadata == {"key": "value"}

    @pytest.mark.asyncio
    async def test_update_summary(self, mock_storage: MagicMock) -> None:
        """Test updating a session's rolling summary."""
        from engram.session.manager import SessionManager

        now = datetime.now(timezone.utc)
        mock_storage.fetchone = AsyncMock(
            return_value={
                "session_id": "sess_1",
                "agent_id": "agent_1",
                "user_id": None,
                "started_at": now,
                "ended_at": None,
                "summary": "rolled summary",
                "summary_updated_at": now,
                "metadata": "{}",
            }
        )

        manager = SessionManager(storage=mock_storage)
        session = await manager.update_summary("sess_1", "rolled summary")

        assert session.summary == "rolled summary"
        assert session.summary_updated_at == now

    @pytest.mark.asyncio
    async def test_update_summary_nonexistent_raises(
        self, mock_storage: MagicMock
    ) -> None:
        """Test updating summary of a missing session raises."""
        from engram.core.exceptions import SessionNotFoundError
        from engram.session.manager import SessionManager

        mock_storage.fetchone = AsyncMock(return_value=None)
        manager = SessionManager(storage=mock_storage)

        with pytest.raises(SessionNotFoundError):
            await manager.update_summary("missing", "x")

    @pytest.mark.asyncio
    async def test_get_nonexistent_session_raises_error(
        self, mock_storage: MagicMock
    ) -> None:
        """Test getting non-existent session raises SessionNotFoundError."""
        from engram.core.exceptions import SessionNotFoundError
        from engram.session.manager import SessionManager

        mock_storage.fetchone = AsyncMock(return_value=None)

        manager = SessionManager(storage=mock_storage)

        with pytest.raises(SessionNotFoundError):
            await manager.get("nonexistent")

    @pytest.mark.asyncio
    async def test_list_active_sessions(self, mock_storage: MagicMock) -> None:
        """Test listing active sessions for an agent."""
        from engram.session.manager import SessionManager

        mock_storage.fetchall = AsyncMock(
            return_value=[
                {
                    "session_id": "sess_1",
                    "agent_id": "agent_1",
                    "user_id": None,
                    "started_at": datetime.now(timezone.utc),
                    "ended_at": None,
                    "metadata": "{}",
                },
                {
                    "session_id": "sess_2",
                    "agent_id": "agent_1",
                    "user_id": "user_1",
                    "started_at": datetime.now(timezone.utc),
                    "ended_at": None,
                    "metadata": "{}",
                },
            ]
        )

        manager = SessionManager(storage=mock_storage)

        sessions = await manager.list_active("agent_1")

        assert len(sessions) == 2
        assert all(s.is_active for s in sessions)


class TestSessionManagerContextManager:
    """Tests for SessionManager context manager."""

    @pytest.fixture
    def mock_storage(self) -> MagicMock:
        storage = MagicMock()
        storage.execute = AsyncMock(return_value="INSERT 1")
        storage.fetchone = AsyncMock()
        return storage

    @pytest.mark.asyncio
    async def test_context_manager_auto_ends_session(
        self, mock_storage: MagicMock
    ) -> None:
        """Test that context manager automatically ends session."""
        from engram.session.manager import SessionManager

        now = datetime.now(timezone.utc)

        # create uses execute, end uses fetchone
        ended_session = {
            "session_id": "sess_ctx",
            "agent_id": "agent_1",
            "user_id": None,
            "started_at": now,
            "ended_at": datetime.now(timezone.utc),
            "metadata": "{}",
        }
        mock_storage.fetchone = AsyncMock(return_value=ended_session)

        manager = SessionManager(storage=mock_storage)

        async with manager.session("agent_1") as session:
            assert session.is_active

        # Session should be ended after context exit
        # Verify: create calls execute (auto-create agent + insert session),
        # end calls fetchone. No user_id, so no user-ensure call.
        assert mock_storage.execute.call_count == 2  # ensure agent + insert session
        mock_storage.fetchone.assert_called_once()  # from end


class TestSessionModel:
    """Tests for Session model."""

    def test_session_id_generation(self) -> None:
        """Test that session IDs are properly formatted."""
        from engram.session.models import Session

        session = Session(agent_id="agent_1")

        assert session.session_id.startswith("sess_")
        assert len(session.session_id) > 10

    def test_session_is_active(self) -> None:
        """Test is_active property."""
        from engram.session.models import Session

        # Active session (no ended_at)
        active = Session(agent_id="agent_1")
        assert active.is_active

        # Ended session
        ended = Session(
            agent_id="agent_1",
            ended_at=datetime.now(timezone.utc),
        )
        assert not ended.is_active

    def test_session_duration(self) -> None:
        """Test duration_seconds property."""
        from datetime import timedelta

        from engram.session.models import Session

        start = datetime.now(timezone.utc)
        end = start + timedelta(seconds=120)

        session = Session(
            agent_id="agent_1",
            started_at=start,
            ended_at=end,
        )

        assert session.duration_seconds == 120.0

    def test_session_duration_active_is_none(self) -> None:
        """Test duration is None for active session."""
        from engram.session.models import Session

        session = Session(agent_id="agent_1")

        assert session.duration_seconds is None

    def test_session_datetime_is_utc(self) -> None:
        """Test that session timestamps are UTC."""
        from engram.session.models import Session

        session = Session(agent_id="agent_1")

        assert session.started_at.tzinfo is not None


class TestSessionCreate:
    """Tests for SessionCreate model."""

    def test_session_create_minimal(self) -> None:
        """Test minimal session creation."""
        from engram.session.models import SessionCreate

        create = SessionCreate(agent_id="agent_1")

        assert create.agent_id == "agent_1"
        assert create.user_id is None
        assert create.metadata == {}

    def test_session_create_with_metadata(self) -> None:
        """Test session creation with metadata."""
        from engram.session.models import SessionCreate

        create = SessionCreate(
            agent_id="agent_1",
            user_id="user_1",
            metadata={"channel": "web", "device": "mobile"},
        )

        assert create.metadata == {"channel": "web", "device": "mobile"}

    def test_session_create_immutable(self) -> None:
        """Test that SessionCreate is immutable."""
        from pydantic import ValidationError

        from engram.session.models import SessionCreate

        create = SessionCreate(agent_id="agent_1")

        with pytest.raises(ValidationError):
            create.agent_id = "other"  # type: ignore
