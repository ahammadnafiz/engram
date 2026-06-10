"""JSON serialization helpers for storage operations."""

from __future__ import annotations

import json
from typing import Any


def json_dumps(value: Any) -> str:
    """json.dumps with a str() fallback for non-native values.

    User-supplied metadata routinely contains datetime, UUID, or Path
    values; coercing them to strings beats crashing deep inside a storage
    operation with an opaque StorageError.
    """
    return json.dumps(value, default=str)
