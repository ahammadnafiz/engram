"""PostgreSQL storage backend for Engram.

This module provides the async PostgreSQL connection pool and storage
interface using asyncpg for high-performance database operations.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit, urlunsplit

import asyncpg

from engram.core.config import EngramSettings, get_settings
from engram.core.exceptions import (
    ConfigurationError,
    ConnectionError,
    ConnectionPoolExhaustedError,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from asyncpg import Connection, Pool

logger = logging.getLogger(__name__)

# SQL directory path
SQL_DIR = Path(__file__).parent.parent / "sql"


def _redact_dsn(dsn: str) -> str:
    """Mask the password component of a DSN for safe logging."""
    try:
        parsed = urlsplit(dsn)
        if parsed.password is None:
            return dsn
        netloc = parsed.netloc.replace(f":{parsed.password}@", ":***@", 1)
        return urlunsplit(parsed._replace(netloc=netloc))
    except (ValueError, AttributeError):
        return "<unparseable dsn>"


class PostgresStorage:
    """Async PostgreSQL storage backend using asyncpg.

    This class manages the connection pool and provides low-level
    database operations. It uses asyncpg for optimal performance
    with PostgreSQL.

    Example:
        async with PostgresStorage() as storage:
            result = await storage.fetchone("SELECT 1")

        # Or explicit lifecycle management
        storage = PostgresStorage()
        await storage.connect()
        try:
            result = await storage.fetchone("SELECT 1")
        finally:
            await storage.close()
    """

    def __init__(self, settings: EngramSettings | None = None) -> None:
        """Initialize storage with settings.

        Args:
            settings: Engram settings instance. If None, loads from environment.
        """
        self._settings = settings or get_settings()
        self._pool: Pool | None = None
        self._connected = False

    @property
    def pool(self) -> Pool:
        """Get the connection pool.

        Raises:
            ConnectionError: If not connected.
        """
        if self._pool is None:
            raise ConnectionError("Not connected. Call connect() first.")
        return self._pool

    @property
    def is_connected(self) -> bool:
        """Check if storage is connected."""
        return self._connected and self._pool is not None

    async def connect(self) -> None:
        """Establish database connection pool.

        This method creates the asyncpg connection pool with the
        configured settings. It should be called before any database
        operations.

        Raises:
            ConnectionError: If connection fails.
        """
        if self._connected:
            logger.warning("Already connected, skipping connect()")
            return

        try:
            logger.info(
                "Connecting to PostgreSQL",
                extra={"url": _redact_dsn(self._settings.database_url)},
            )

            self._pool = await asyncpg.create_pool(
                dsn=self._settings.database_url,
                min_size=self._settings.min_pool_size,
                max_size=self._settings.max_pool_size,
                timeout=self._settings.connection_timeout,
                command_timeout=self._settings.command_timeout,
                setup=self._setup_connection,
            )
            self._connected = True
            logger.info("Connected to PostgreSQL successfully")

        except asyncpg.PostgresError as e:
            raise ConnectionError(
                f"Failed to connect to PostgreSQL: {e}",
                dsn=_redact_dsn(self._settings.database_url),
            ) from e
        except TimeoutError as e:
            raise ConnectionError(
                "Connection timeout",
                timeout=self._settings.connection_timeout,
            ) from e
        except OSError as e:
            raise ConnectionError(
                f"Failed to connect to PostgreSQL: {e}",
                dsn=_redact_dsn(self._settings.database_url),
            ) from e

    async def _setup_connection(self, conn: Connection) -> None:
        """Per-connection session settings for filtered vector search recall.

        One global HNSW index is filtered by agent_id after the scan; with
        many agents, ef_search candidates can all belong to other agents and
        recall collapses. Iterative scans (pgvector >= 0.8) keep scanning
        until enough rows survive the filter; strict_order preserves exact
        distance ordering. Both SETs are best-effort: older pgvector versions
        reject them and search still works (with the old recall behavior).
        """
        try:
            await conn.execute("SET hnsw.iterative_scan = strict_order")
        except asyncpg.PostgresError:
            logger.debug("pgvector iterative scans unavailable (< 0.8); skipped")
        if self._settings.hnsw_ef_search is not None:
            try:
                await conn.execute(
                    f"SET hnsw.ef_search = {int(self._settings.hnsw_ef_search)}"
                )
            except asyncpg.PostgresError:
                logger.debug("hnsw.ef_search not supported; skipped")

    async def close(self) -> None:
        """Close the connection pool.

        This method gracefully closes all connections in the pool.
        It should be called when shutting down the application.
        """
        if self._pool is not None:
            logger.info("Closing PostgreSQL connection pool")
            await self._pool.close()
            self._pool = None
            self._connected = False
            logger.info("PostgreSQL connection pool closed")

    async def __aenter__(self) -> PostgresStorage:
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

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[Connection]:
        """Acquire a connection from the pool.

        This context manager ensures connections are properly returned
        to the pool after use.

        Yields:
            A database connection.

        Raises:
            ConnectionError: If not connected.
            ConnectionPoolExhaustedError: If pool is exhausted.

        Example:
            async with storage.acquire() as conn:
                await conn.execute("SELECT 1")
        """
        if not self.is_connected:
            raise ConnectionError("Not connected. Call connect() first.")

        try:
            async with self.pool.acquire() as conn:
                yield conn
        except TimeoutError as e:
            raise ConnectionPoolExhaustedError(
                "Connection pool exhausted",
                pool_size=self._settings.max_pool_size,
                timeout=self._settings.connection_timeout,
            ) from e

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[Connection]:
        """Acquire a connection and open a transaction on it.

        Statements executed on the yielded connection commit together and
        roll back together. Use this for multi-statement operations that
        must be atomic (e.g. insert + conflict supersede).

        Example:
            async with storage.transaction() as conn:
                await conn.execute("INSERT ...")
                await conn.execute("UPDATE ...")
        """
        async with self.acquire() as conn, conn.transaction():
            yield conn

    async def execute(self, query: str, *args: Any) -> str:
        """Execute a query without returning results.

        Args:
            query: SQL query to execute.
            *args: Query parameters.

        Returns:
            Status string from the command.
        """
        async with self.acquire() as conn:
            result: str = await conn.execute(query, *args)
            return result

    async def executemany(self, query: str, args: list[tuple[Any, ...]]) -> None:
        """Execute a query with multiple parameter sets.

        Args:
            query: SQL query to execute.
            args: List of parameter tuples.
        """
        async with self.acquire() as conn:
            await conn.executemany(query, args)

    async def fetchone(self, query: str, *args: Any) -> asyncpg.Record | None:
        """Fetch a single row.

        Args:
            query: SQL query to execute.
            *args: Query parameters.

        Returns:
            A single record or None if no results.
        """
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchall(self, query: str, *args: Any) -> list[asyncpg.Record]:
        """Fetch all rows.

        Args:
            query: SQL query to execute.
            *args: Query parameters.

        Returns:
            List of records.
        """
        async with self.acquire() as conn:
            return cast("list[asyncpg.Record]", await conn.fetch(query, *args))

    async def fetchval(self, query: str, *args: Any, column: int = 0) -> Any:
        """Fetch a single value.

        Args:
            query: SQL query to execute.
            *args: Query parameters.
            column: Column index to return.

        Returns:
            The value from the specified column.
        """
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args, column=column)

    def load_sql(self, filename: str) -> str:
        """Load SQL from a file in the sql directory.

        Args:
            filename: Name of the SQL file (without path).

        Returns:
            Contents of the SQL file.

        Raises:
            FileNotFoundError: If the SQL file doesn't exist.
        """
        sql_path = SQL_DIR / filename
        if not sql_path.exists():
            raise FileNotFoundError(f"SQL file not found: {sql_path}")
        return sql_path.read_text()

    async def init_schema(self, embedding_dimension: int | None = None) -> None:
        """Initialize the database schema.

        This method creates all required tables and indices if they
        don't exist. It's safe to call multiple times.

        If embedding_dimension is provided, the schema will be adjusted
        to use that dimension for the vector column.

        Args:
            embedding_dimension: The embedding dimension to use. If None,
                uses the default from schema.sql (1536).
        """
        logger.info("Initializing database schema")
        schema_sql = self.load_sql("schema.sql")
        await self.execute(schema_sql)

        # Pre-0.2 databases lack the fact/main_content columns; migration 001
        # adds them idempotently (it no-ops on fresh and current schemas).
        await self.execute(self.load_sql("migrations/001_add_fact_columns.sql"))

        # Idempotent column adds for existing databases (CREATE TABLE IF NOT
        # EXISTS does not add columns to a pre-existing table).
        await self.execute(
            "ALTER TABLE agent_sessions ADD COLUMN IF NOT EXISTS summary TEXT;"
        )
        await self.execute(
            "ALTER TABLE agent_sessions "
            "ADD COLUMN IF NOT EXISTS summary_updated_at TIMESTAMPTZ;"
        )
        await self.execute(
            "ALTER TABLE agent_memory "
            "ADD COLUMN IF NOT EXISTS memory_type TEXT NOT NULL DEFAULT 'semantic';"
        )
        await self.execute(
            "ALTER TABLE agent_memory "
            "DROP CONSTRAINT IF EXISTS agent_memory_memory_type_check;"
        )
        await self.execute(
            """
            ALTER TABLE agent_memory
            ADD CONSTRAINT agent_memory_memory_type_check
            CHECK (
                memory_type IN (
                    'semantic', 'episodic', 'procedural',
                    'profile', 'project', 'task', 'preference',
                    'constraint', 'decision', 'tool_result'
                )
            );
            """
        )
        # Index created after the column exists (works for both fresh and existing DBs)
        await self.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_agent_type "
            "ON agent_memory(agent_id, memory_type);"
        )

        # First-class memory lineage/current-head columns. This runs after
        # memory_type setup because older databases may have been created before
        # migrations were applied uniformly.
        await self.execute(self.load_sql("migrations/006_add_memory_lineage.sql"))

        # Hybrid search over the event ledger. The migration is intentionally
        # fast-only: no stored tsvector column, no embedding backfill, and no
        # blocking index build inside the migration transaction.
        await self.execute(self.load_sql("migrations/007_add_event_search.sql"))

        # Upgrade pre-existing unique fact indexes to the md5 form (raw-fact
        # entries fail for facts larger than the ~2704-byte btree row limit) and
        # to the active-row partial form used by memory lineage history.
        unique_fact_def = await self.fetchval(
            "SELECT indexdef FROM pg_indexes WHERE indexname = 'idx_unique_memory_fact'"
        )
        if unique_fact_def is not None and (
            "md5" not in unique_fact_def
            or "WHERE (status = 'active'::text)" not in unique_fact_def
        ):
            logger.info("Rebuilding idx_unique_memory_fact on active md5(fact)")
            await self.execute("DROP INDEX idx_unique_memory_fact;")
            await self.execute(
                "CREATE UNIQUE INDEX idx_unique_memory_fact "
                "ON agent_memory(agent_id, COALESCE(user_id, ''), md5(fact)) "
                "WHERE status = 'active';"
            )
        logger.info("Database schema initialized")

        # Align the generated memory tsvector columns with the configured language
        await self._ensure_text_search_config(self._settings.text_search_config)

        # Adjust vector dimension if specified
        if embedding_dimension is not None:
            await self._ensure_vector_dimension(embedding_dimension)

        await self._ensure_event_search_indexes(self._settings.text_search_config)

    async def _ensure_text_search_config(self, config: str) -> None:
        """Rebuild the generated tsvector columns when the language changed.

        The memory schema ships with 'english'; non-English deployments set
        ENGRAM_TEXT_SEARCH_CONFIG and the memory columns are recreated with
        that configuration (a table rewrite — cheap on fresh DBs, logged on
        populated ones). Event search uses expression indexes instead of a
        stored tsvector column, so large event ledgers avoid a table rewrite.
        """
        import re as _re

        if not _re.fullmatch(r"[a-z_]+", config):
            raise ConfigurationError(
                f"Invalid text_search_config {config!r}: must match [a-z_]+"
            )

        expr = await self.fetchval(
            """
            SELECT pg_get_expr(d.adbin, d.adrelid)
            FROM pg_attrdef d
            JOIN pg_attribute a
                ON a.attrelid = d.adrelid AND a.attnum = d.adnum
            WHERE d.adrelid = 'agent_memory'::regclass
                AND a.attname = 'fact_tsv'
            """
        )
        current = None
        if expr:
            match = _re.search(r"'([a-z_]+)'::regconfig", expr)
            current = match.group(1) if match else None

        if current == config:
            return

        logger.warning(
            f"Rebuilding tsvector columns for text search config "
            f"{current!r} -> {config!r} (table rewrite)"
        )
        await self.execute("DROP INDEX IF EXISTS idx_memory_fact_tsv;")
        await self.execute("DROP INDEX IF EXISTS idx_memory_main_content_tsv;")
        await self.execute("ALTER TABLE agent_memory DROP COLUMN IF EXISTS fact_tsv;")
        await self.execute(
            "ALTER TABLE agent_memory DROP COLUMN IF EXISTS main_content_tsv;"
        )
        # config is validated against [a-z_]+ above; safe to interpolate.
        await self.execute(
            f"""
            ALTER TABLE agent_memory ADD COLUMN fact_tsv TSVECTOR
                GENERATED ALWAYS AS (to_tsvector('{config}', fact)) STORED;
            """
        )
        await self.execute(
            f"""
            ALTER TABLE agent_memory ADD COLUMN main_content_tsv TSVECTOR
                GENERATED ALWAYS AS (
                    CASE WHEN main_content IS NOT NULL
                    THEN to_tsvector('{config}', main_content)
                    ELSE NULL END
                ) STORED;
            """
        )
        await self.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_fact_tsv "
            "ON agent_memory USING GIN (fact_tsv);"
        )
        await self.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_main_content_tsv "
            "ON agent_memory USING GIN (main_content_tsv) "
            "WHERE main_content IS NOT NULL;"
        )
        logger.info(f"tsvector columns now use text search config '{config}'")

    async def _ensure_event_search_indexes(self, config: str) -> None:
        """Build event recall indexes online.

        These statements are deliberately kept outside migration 007. On large
        ledgers, adding a nullable column is fast, while building GIN/HNSW
        indexes can take time. ``CONCURRENTLY`` keeps reads and writes flowing
        during that build in production.
        """
        import re as _re

        if not _re.fullmatch(r"[a-z_]+", config):
            raise ConfigurationError(
                f"Invalid text_search_config {config!r}: must match [a-z_]+"
            )

        indexdef = await self.fetchval(
            "SELECT indexdef FROM pg_indexes WHERE indexname = 'idx_events_content_tsv'"
        )
        expected = f"to_tsvector('{config}'::regconfig, content)"
        if indexdef is None or expected not in str(indexdef):
            await self.execute(
                "DROP INDEX CONCURRENTLY IF EXISTS idx_events_content_tsv;"
            )
            await self.execute(
                f"""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_content_tsv
                ON agent_events USING GIN (to_tsvector('{config}', content));
                """
            )
        await self.execute(
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_embedding
            ON agent_events USING hnsw (event_embedding vector_cosine_ops)
            WHERE event_embedding IS NOT NULL;
            """
        )

    async def _get_current_vector_dimension(
        self, table: str = "agent_memory", column: str = "embedding"
    ) -> int | None:
        """Get the current vector column dimension from the database.

        Returns:
            The current dimension, or None if the column doesn't exist.
        """
        query = """
        SELECT atttypmod
        FROM pg_attribute
        WHERE attrelid = $1::regclass
        AND attname = $2;
        """
        try:
            result = await self.fetchval(query, table, column)
            if result is not None and result > 0:
                return int(result)
            return None
        except Exception as e:
            # Log the error but return None since this is a probing query
            # that may fail on first run before schema exists
            logger.debug(f"Could not get current vector dimension: {e}")
            return None

    async def _ensure_vector_dimension(self, target_dimension: int) -> None:
        """Ensure the vector column has the correct dimension.

        This method checks the current dimension and adjusts it if needed.
        It handles the case where dimensions don't match by dropping and
        recreating the index, then altering the column type.

        Args:
            target_dimension: The desired embedding dimension.
        """
        current_dimension = await self._get_current_vector_dimension()
        event_dimension = await self._get_current_vector_dimension(
            "agent_events", "event_embedding"
        )

        if (
            current_dimension == target_dimension
            and event_dimension == target_dimension
        ):
            logger.debug(f"Vector dimension already set to {target_dimension}")
            return

        # Check if there's existing data that would be lost
        memory_row_count = await self.fetchval(
            "SELECT COUNT(*) FROM agent_memory WHERE embedding IS NOT NULL"
        )
        event_row_count = await self.fetchval(
            "SELECT COUNT(*) FROM agent_events WHERE event_embedding IS NOT NULL"
        )
        row_count = int(memory_row_count or 0) + int(event_row_count or 0)

        if (
            row_count
            and row_count > 0
            and not self._settings.allow_embedding_dimension_change
        ):
            raise ConfigurationError(
                f"Embedding dimension changed from memory={current_dimension}, "
                f"events={event_dimension} to {target_dimension}, which would "
                f"clear {row_count} stored embeddings and make those rows invisible to vector "
                "search. If this is intentional, set "
                "ENGRAM_ALLOW_EMBEDDING_DIMENSION_CHANGE=true and re-embed "
                "existing memories afterwards; otherwise restore the previous "
                "embedding provider/model configuration.",
                current_dimension=current_dimension,
                event_dimension=event_dimension,
                target_dimension=target_dimension,
                affected_embeddings=row_count,
            )

        logger.info(
            "Adjusting vector dimension from "
            f"memory={current_dimension}, events={event_dimension} "
            f"to {target_dimension}"
        )

        if row_count and row_count > 0:
            logger.warning(
                f"Changing vector dimension will clear {row_count} existing embeddings! "
                "They will need to be re-embedded."
            )

        # Drop the HNSW index first (required before altering column type)
        logger.info("Dropping existing vector indexes")
        await self.execute("DROP INDEX IF EXISTS idx_memory_embedding;")
        await self.execute("DROP INDEX IF EXISTS idx_events_embedding;")

        # Clear existing embeddings (they're incompatible with new dimension)
        if row_count and row_count > 0:
            logger.info("Clearing incompatible embeddings")
            await self.execute("UPDATE agent_memory SET embedding = NULL;")
            await self.execute("UPDATE agent_events SET event_embedding = NULL;")

        # Alter the column to the new dimension
        logger.info(f"Altering embedding columns to VECTOR({target_dimension})")
        await self.execute(
            f"ALTER TABLE agent_memory ALTER COLUMN embedding TYPE VECTOR({target_dimension});"
        )
        await self.execute(
            "ALTER TABLE agent_events "
            f"ALTER COLUMN event_embedding TYPE VECTOR({target_dimension});"
        )

        # Recreate the HNSW index
        logger.info("Recreating vector index")
        await self.execute(
            """CREATE INDEX IF NOT EXISTS idx_memory_embedding ON agent_memory
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);"""
        )

        logger.info(f"Vector dimension successfully set to {target_dimension}")

    async def health_check(self) -> dict[str, Any]:
        """Perform a health check on the database connection.

        Returns:
            Dictionary with health status and metrics.
        """
        try:
            result = await self.fetchone("SELECT 1 as check, NOW() as timestamp")
            pool_info = {
                "min_size": self._settings.min_pool_size,
                "max_size": self._settings.max_pool_size,
                "size": self.pool.get_size() if self._pool else 0,
                "free_size": self.pool.get_idle_size() if self._pool else 0,
            }
            return {
                "status": "healthy",
                "database": "connected",
                "timestamp": result["timestamp"] if result else None,
                "pool": pool_info,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "database": "disconnected",
                "error": str(e),
            }
