"""Engram MCP server — a memory layer for Claude Code / Codex / Gemini CLI.

The server and its CLI live in :mod:`engram.mcp.server`; the ``engram-mcp``
console script points at ``engram.mcp.server:main``. Kept import-light so
``python -m engram.mcp.server`` runs without a double-import warning.
"""
