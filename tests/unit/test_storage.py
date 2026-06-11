"""Unit tests for PostgresStorage."""

from __future__ import annotations

import pytest


class TestDsnRedaction:
    """Credentials must never appear in logs or exception messages."""

    def test_redact_dsn_masks_password(self) -> None:
        from engram.storage.postgres import _redact_dsn

        dsn = "postgresql://engram:s3cret-pw@localhost:5432/engram"
        redacted = _redact_dsn(dsn)

        assert "s3cret-pw" not in redacted
        assert "engram" in redacted  # user and db kept for debuggability
        assert "localhost:5432" in redacted

    def test_redact_dsn_without_password(self) -> None:
        from engram.storage.postgres import _redact_dsn

        dsn = "postgresql://localhost:5432/engram"
        assert _redact_dsn(dsn) == dsn

    def test_redact_dsn_malformed_url(self) -> None:
        from engram.storage.postgres import _redact_dsn

        # Must not crash on garbage; must not echo a password-looking part
        assert isinstance(_redact_dsn("not a url"), str)


class TestVectorDimensionGuard:
    """Changing embedding dimension must never silently destroy embeddings."""

    def _storage(self, *, current_dim: int, row_count: int, allow: bool):
        from unittest.mock import AsyncMock

        from engram.core.config import EngramSettings
        from engram.storage.postgres import PostgresStorage

        settings = EngramSettings(allow_embedding_dimension_change=allow)
        storage = PostgresStorage(settings)
        # fetchval is called for: current dimension probe, then row count
        storage.fetchval = AsyncMock(side_effect=[current_dim, row_count])  # type: ignore[method-assign]
        storage.execute = AsyncMock()  # type: ignore[method-assign]
        return storage

    @pytest.mark.asyncio
    async def test_mismatch_with_data_raises_by_default(self) -> None:
        from engram.core.exceptions import ConfigurationError

        storage = self._storage(current_dim=1536, row_count=42, allow=False)

        with pytest.raises(ConfigurationError) as exc_info:
            await storage._ensure_vector_dimension(384)

        assert "42" in str(exc_info.value)
        storage.execute.assert_not_called()  # nothing destructive ran

    @pytest.mark.asyncio
    async def test_mismatch_with_empty_table_adjusts(self) -> None:
        storage = self._storage(current_dim=1536, row_count=0, allow=False)

        await storage._ensure_vector_dimension(384)

        executed = " ".join(str(c) for c in storage.execute.call_args_list)
        assert "ALTER TABLE agent_memory" in executed

    @pytest.mark.asyncio
    async def test_mismatch_with_data_and_explicit_opt_in_adjusts(self) -> None:
        storage = self._storage(current_dim=1536, row_count=42, allow=True)

        await storage._ensure_vector_dimension(384)

        executed = " ".join(str(c) for c in storage.execute.call_args_list)
        assert "embedding = NULL" in executed

    @pytest.mark.asyncio
    async def test_matching_dimension_is_noop(self) -> None:
        storage = self._storage(current_dim=384, row_count=0, allow=False)

        await storage._ensure_vector_dimension(384)

        storage.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_error_does_not_leak_password(self) -> None:
        """Connection failures must not expose the password in str(exc)."""
        from engram.core.config import EngramSettings
        from engram.core.exceptions import ConnectionError as EngramConnectionError
        from engram.storage.postgres import PostgresStorage

        settings = EngramSettings(
            database_url="postgresql://user:supersecretpw@127.0.0.1:1/nodb",
            connection_timeout=0.5,
        )
        storage = PostgresStorage(settings)

        with pytest.raises(EngramConnectionError) as exc_info:
            await storage.connect()

        assert "supersecretpw" not in str(exc_info.value)
