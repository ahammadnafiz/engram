"""LLM service for Engram.

This module provides the LLM service for fact extraction and other AI tasks.
Implements Mem0-style intelligent memory operations.
"""

from engram.llm.service import (
    ExtractionResult,
    LLMService,
    MemoryOperation,
    MemoryOperationType,
)

__all__ = [
    "ExtractionResult",
    "LLMService",
    "MemoryOperation",
    "MemoryOperationType",
]

