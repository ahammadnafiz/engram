"""Storage module for Engram.

This module provides database storage backends for memory persistence.
"""

from engram.storage.postgres import PostgresStorage

__all__ = [
    "PostgresStorage",
]
