"""Unit tests for the HTTP retry helper used by httpx-based providers."""

from __future__ import annotations

import httpx
import pytest


def _response(status: int) -> httpx.Response:
    return httpx.Response(
        status_code=status, request=httpx.Request("POST", "http://test")
    )


class TestProviderDefaults:
    def test_anthropic_default_model_is_current(self) -> None:
        """The shipped default must be a current, non-retired model id."""
        import inspect

        from engram.providers.llm.builtin import AnthropicLLMProvider

        default = (
            inspect.signature(AnthropicLLMProvider.__init__).parameters["model"].default
        )
        assert default == "claude-haiku-4-5-20251001"


class TestRequestWithRetries:
    @pytest.mark.asyncio
    async def test_retries_transient_500_then_succeeds(self) -> None:
        from engram.providers.http_retry import request_with_retries

        calls = 0

        async def send() -> httpx.Response:
            nonlocal calls
            calls += 1
            return _response(500 if calls < 3 else 200)

        response = await request_with_retries(send, retries=2, base_delay=0.001)

        assert response.status_code == 200
        assert calls == 3

    @pytest.mark.asyncio
    async def test_non_retryable_400_raises_immediately(self) -> None:
        from engram.providers.http_retry import request_with_retries

        calls = 0

        async def send() -> httpx.Response:
            nonlocal calls
            calls += 1
            return _response(400)

        with pytest.raises(httpx.HTTPStatusError):
            await request_with_retries(send, retries=2, base_delay=0.001)

        assert calls == 1

    @pytest.mark.asyncio
    async def test_exhausted_retries_raise(self) -> None:
        from engram.providers.http_retry import request_with_retries

        calls = 0

        async def send() -> httpx.Response:
            nonlocal calls
            calls += 1
            return _response(503)

        with pytest.raises(httpx.HTTPStatusError):
            await request_with_retries(send, retries=2, base_delay=0.001)

        assert calls == 3  # initial + 2 retries

    @pytest.mark.asyncio
    async def test_connect_error_retried(self) -> None:
        from engram.providers.http_retry import request_with_retries

        calls = 0

        async def send() -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise httpx.ConnectError("refused")
            return _response(200)

        response = await request_with_retries(send, retries=2, base_delay=0.001)

        assert response.status_code == 200
        assert calls == 2
