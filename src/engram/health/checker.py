"""Health check for Engram.

This module provides health checking and diagnostics functionality.
"""

from __future__ import annotations

import logging
import platform
import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from engram._version import __version__

if TYPE_CHECKING:
    from engram.embedding.service import EmbeddingService
    from engram.storage.postgres import PostgresStorage

logger = logging.getLogger(__name__)


class HealthChecker:
    """Health checker for Engram components.

    Provides health status and diagnostics for the database connection,
    embedding service, and overall system.

    Example:
        checker = HealthChecker(storage, embedding_service)
        status = await checker.check()
        print(status["status"])  # "healthy" or "unhealthy"
    """

    def __init__(
        self,
        storage: PostgresStorage,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        """Initialize health checker.

        Args:
            storage: PostgreSQL storage backend.
            embedding_service: Optional embedding service.
        """
        self._storage = storage
        self._embedding = embedding_service

    async def check(self, skip_embedding_test: bool = False) -> dict[str, Any]:
        """Perform a comprehensive health check.

        Args:
            skip_embedding_test: If True, skip the actual embedding API call.
                Useful to avoid costs on pay-per-call embedding APIs.
                The embedding service will still be checked for configuration validity.

        Returns:
            Dictionary with health status and component details.
        """
        start_time = datetime.now(timezone.utc)

        # Check database
        db_status = await self._check_database()

        # Check embedding service
        embedding_status = await self._check_embedding(skip_test=skip_embedding_test)

        # Determine overall status
        is_healthy = db_status.get("status") == "healthy" and (
            embedding_status is None or embedding_status.get("status") == "healthy"
        )

        end_time = datetime.now(timezone.utc)
        check_duration_ms = (end_time - start_time).total_seconds() * 1000

        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "version": __version__,
            "timestamp": end_time.isoformat(),
            "check_duration_ms": round(check_duration_ms, 2),
            "components": {
                "database": db_status,
                "embedding": embedding_status,
            },
            "system": self._get_system_info(),
        }

    async def _check_database(self) -> dict[str, Any]:
        """Check database health.

        Returns:
            Dictionary with database status.
        """
        try:
            return await self._storage.health_check()
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    async def _check_embedding(self, skip_test: bool = False) -> dict[str, Any] | None:
        """Check embedding service health.

        Args:
            skip_test: If True, skip the actual embedding test to avoid API costs.

        Returns:
            Dictionary with embedding status, or None if no service.
        """
        if self._embedding is None:
            return None

        result: dict[str, Any] = {
            "status": "healthy",
            "model": self._embedding.model,
            "dimension": self._embedding.dimension,
            "cache": self._embedding.cache_info,
        }

        if skip_test:
            result["test_skipped"] = True
            return result

        try:
            # Try a simple embedding (uses cache if available)
            test_vector = await self._embedding.embed("health check test")
            result["test_vector_length"] = len(test_vector)
            return result
        except Exception as e:
            return {
                "status": "unhealthy",
                "model": self._embedding.model,
                "error": str(e),
            }

    def _get_system_info(self) -> dict[str, Any]:
        """Get system information.

        Returns:
            Dictionary with system details.
        """
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
        }

    async def check_quick(self) -> bool:
        """Quick health check returning just a boolean.

        Returns:
            True if healthy, False otherwise.
        """
        try:
            result = await self._storage.fetchval("SELECT 1")
            return bool(result == 1)
        except Exception:
            return False
