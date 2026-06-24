#!/usr/bin/env python3
"""Backward-compatible shim — the Engram MCP server now lives in the package.

Kept so existing ``.mcp.json`` / Stop-hook configs that point at this file path
keep working unchanged. New setups should instead install the ``mcp`` extra and
use the ``engram-mcp`` command:

    pip install -e ".[mcp,sentence-transformers]"
    engram-mcp serve        # or: engram-mcp hook | list | forget | save

See docs/mcp-server.md.
"""

from engram.mcp.server import main

if __name__ == "__main__":
    main()
