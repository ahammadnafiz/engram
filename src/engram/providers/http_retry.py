"""Minimal retry helper for HTTP-based providers.

Used by the httpx-backed providers (Ollama, Groq, HuggingFace) which have no
SDK-level retries. The OpenAI/Anthropic/Cohere SDK clients retry internally
and don't need this.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

# Transient statuses worth a retry; 4xx other than 429 are caller errors.
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


async def request_with_retries(
    send: Callable[[], Awaitable[Any]],
    *,
    retries: int = 2,
    base_delay: float = 0.5,
) -> Any:
    """Run an async HTTP call, retrying transient failures with backoff.

    Args:
        send: Async callable performing the request and returning an
            httpx.Response (raise_for_status is called here).
        retries: Maximum number of retries after the first attempt.
        base_delay: First backoff delay in seconds (doubles per retry).

    Returns:
        The successful httpx.Response.

    Raises:
        httpx.HTTPStatusError: Non-retryable status, or retries exhausted.
        httpx.TransportError: Connection failures after retries exhausted.
    """
    import httpx

    attempt = 0
    while True:
        try:
            response = await send()
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status not in RETRYABLE_STATUS_CODES or attempt >= retries:
                raise
            logger.debug(f"Retrying HTTP {status} (attempt {attempt + 1}/{retries})")
        except httpx.TransportError as e:
            if attempt >= retries:
                raise
            logger.debug(f"Retrying transport error: {e} (attempt {attempt + 1})")
        attempt += 1
        await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
