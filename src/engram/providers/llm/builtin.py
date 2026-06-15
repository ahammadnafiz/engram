"""Built-in LLM providers for Engram.

This module registers the built-in LLM providers:
- openai: OpenAI GPT models
- anthropic: Anthropic Claude models
- ollama: Ollama local LLMs
- groq: Groq inference API
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from engram.core.exceptions import ConfigurationError, LLMProviderError
from engram.providers.http_retry import request_with_retries
from engram.providers.llm.protocol import LLMMessage, LLMProvider, LLMResponse
from engram.providers.llm.registry import llm_registry

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic
    from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


def _to_dict_messages(
    messages: list[LLMMessage] | list[dict[str, str]],
) -> list[dict[str, str]]:
    """Convert messages to dict format."""
    return [m.to_dict() if isinstance(m, LLMMessage) else m for m in messages]


# =============================================================================
# OpenAI Provider
# =============================================================================


@llm_registry.register("openai", aliases=["gpt", "chatgpt"])
class OpenAILLMProvider(LLMProvider):
    """LLM provider using OpenAI's API.

    Supports GPT-4, GPT-4o, GPT-3.5, and other OpenAI models.

    Args:
        api_key: OpenAI API key (required).
        model: Model name (default: "gpt-4o-mini").
        base_url: Custom API base URL (optional, for Azure or proxies).

    Example:
        provider = OpenAILLMProvider(
            api_key="sk-...",
            model="gpt-4o-mini",
        )

        response = await provider.complete([
            {"role": "user", "content": "Hello!"}
        ])
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        **_kwargs: Any,
    ) -> None:
        if not api_key:
            raise ConfigurationError(
                "OpenAI API key is required. Pass api_key parameter or set "
                "ENGRAM_OPENAI_API_KEY environment variable."
            )

        try:
            import openai as openai_module

            self._openai = openai_module
        except ImportError as e:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            ) from e

        # Only pass base_url if specified (None would override defaults)
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client: AsyncOpenAI = self._openai.AsyncOpenAI(**client_kwargs)
        self._model = model

        logger.info(f"Initialized OpenAI LLM provider: {model}")

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self,
        messages: list[LLMMessage] | list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        try:
            call_kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": _to_dict_messages(messages),
            }
            if temperature is not None:
                call_kwargs["temperature"] = temperature
            call_kwargs.update(kwargs)

            # Model-compatibility fallbacks. Newer/reasoning models require
            # max_completion_tokens (not max_tokens) and may reject an explicit
            # temperature (they only allow the default). Swap or drop the
            # offending parameter on a BadRequestError and retry, so one config
            # works across gpt-4o and reasoning models without special-casing.
            if max_tokens is not None:
                call_kwargs["max_completion_tokens"] = max_tokens
            response = None
            for _ in range(3):
                try:
                    response = await self._client.chat.completions.create(**call_kwargs)
                    break
                except self._openai.BadRequestError as e:
                    msg = str(e)
                    if (
                        "max_completion_tokens" in msg
                        and "max_completion_tokens" in call_kwargs
                    ):
                        call_kwargs["max_tokens"] = call_kwargs.pop(
                            "max_completion_tokens"
                        )
                        continue
                    if "temperature" in msg and "temperature" in call_kwargs:
                        call_kwargs.pop("temperature")
                        continue
                    raise
            if response is None:
                raise LLMProviderError(
                    "OpenAI request failed after parameter fallbacks",
                    model=self._model,
                )

            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens
                    if response.usage
                    else 0,
                    "completion_tokens": response.usage.completion_tokens
                    if response.usage
                    else 0,
                    "total_tokens": response.usage.total_tokens
                    if response.usage
                    else 0,
                    # Cached prefix tokens billed at a discount. Non-zero here
                    # confirms automatic prompt caching is firing across the
                    # repeated memory-context prefix of the multi-call reader.
                    "cached_tokens": (
                        getattr(
                            getattr(response.usage, "prompt_tokens_details", None),
                            "cached_tokens",
                            0,
                        )
                        or 0
                    )
                    if response.usage
                    else 0,
                },
                finish_reason=response.choices[0].finish_reason,
                raw=response,
            )
        except self._openai.APIError as e:
            raise LLMProviderError(f"OpenAI API error: {e}", model=self._model) from e
        except Exception as e:
            raise LLMProviderError(f"Failed to complete: {e}", model=self._model) from e


# =============================================================================
# Anthropic Provider
# =============================================================================


@llm_registry.register("anthropic", aliases=["claude"])
class AnthropicLLMProvider(LLMProvider):
    """LLM provider using Anthropic's API.

    Supports current Claude models (Haiku, Sonnet, Opus families).

    Args:
        api_key: Anthropic API key (required).
        model: Model name (default: "claude-haiku-4-5-20251001").

    Example:
        provider = AnthropicLLMProvider(
            api_key="sk-ant-...",
            model="claude-haiku-4-5-20251001",
        )
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-haiku-4-5-20251001",
        **_kwargs: Any,
    ) -> None:
        if not api_key:
            raise ConfigurationError(
                "Anthropic API key is required. Pass api_key parameter or set "
                "ENGRAM_ANTHROPIC_API_KEY environment variable."
            )

        try:
            import anthropic as anthropic_module

            self._anthropic = anthropic_module
        except ImportError as e:
            raise ImportError(
                "Anthropic package not installed. Install with: pip install anthropic"
            ) from e

        self._client: AsyncAnthropic = self._anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model

        logger.info(f"Initialized Anthropic LLM provider: {model}")

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self,
        messages: list[LLMMessage] | list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        try:
            dict_messages = _to_dict_messages(messages)

            # Extract and combine all system messages (Anthropic handles it separately)
            # Multiple system messages are concatenated with newlines
            system_parts: list[str] = []
            filtered_messages = []
            for msg in dict_messages:
                if msg["role"] == "system":
                    system_parts.append(msg["content"])
                else:
                    filtered_messages.append(msg)

            call_kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": filtered_messages,
                "max_tokens": max_tokens or 1024,
            }
            if system_parts:
                call_kwargs["system"] = "\n\n".join(system_parts)
            if temperature is not None:
                call_kwargs["temperature"] = temperature
            call_kwargs.update(kwargs)

            response = await self._client.messages.create(**call_kwargs)

            return LLMResponse(
                content=response.content[0].text if response.content else "",
                model=response.model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens
                    + response.usage.output_tokens,
                    "cache_read_tokens": getattr(
                        response.usage, "cache_read_input_tokens", 0
                    )
                    or 0,
                    "cache_creation_tokens": getattr(
                        response.usage, "cache_creation_input_tokens", 0
                    )
                    or 0,
                },
                finish_reason=response.stop_reason,
                raw=response,
            )
        except self._anthropic.APIError as e:
            raise LLMProviderError(
                f"Anthropic API error: {e}", model=self._model
            ) from e
        except Exception as e:
            raise LLMProviderError(f"Failed to complete: {e}", model=self._model) from e


# =============================================================================
# Ollama Provider
# =============================================================================


@llm_registry.register("ollama", aliases=["ollama-llm", "local"])
class OllamaLLMProvider(LLMProvider):
    """LLM provider using Ollama's local server.

    Ollama runs LLMs locally on your machine.

    Args:
        model: Model name (required, e.g., "llama3.2", "mistral", "codellama").
        base_url: Ollama server URL (default: "http://localhost:11434").

    Example:
        provider = OllamaLLMProvider(model="llama3.2")
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        **_kwargs: Any,
    ) -> None:
        try:
            import httpx

            self._httpx = httpx
        except ImportError as e:
            raise ImportError(
                "httpx package not installed. Install with: pip install httpx"
            ) from e

        self._client = self._httpx.AsyncClient(base_url=base_url, timeout=120.0)
        self._model = model
        self._base_url = base_url

        logger.info(f"Initialized Ollama LLM provider: {model} at {base_url}")

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self,
        messages: list[LLMMessage] | list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **_kwargs: Any,
    ) -> LLMResponse:
        try:
            call_kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": _to_dict_messages(messages),
                "stream": False,
            }

            options: dict[str, Any] = {}
            if max_tokens is not None:
                options["num_predict"] = max_tokens
            if temperature is not None:
                options["temperature"] = temperature
            if options:
                call_kwargs["options"] = options

            response = await request_with_retries(
                lambda: self._client.post("/api/chat", json=call_kwargs)
            )
            data = response.json()

            return LLMResponse(
                content=data.get("message", {}).get("content", ""),
                model=data.get("model", self._model),
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0)
                    + data.get("eval_count", 0),
                },
                finish_reason=data.get("done_reason"),
                raw=data,
            )
        except self._httpx.HTTPError as e:
            raise LLMProviderError(f"Ollama API error: {e}", model=self._model) from e
        except Exception as e:
            raise LLMProviderError(f"Failed to complete: {e}", model=self._model) from e

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


# =============================================================================
# Groq Provider
# =============================================================================


@llm_registry.register("groq")
class GroqLLMProvider(LLMProvider):
    """LLM provider using Groq's fast inference API.

    Groq provides extremely fast inference for Llama, Mixtral, and other models.

    Args:
        api_key: Groq API key (required).
        model: Model name (default: "llama-3.1-8b-instant").

    Example:
        provider = GroqLLMProvider(
            api_key="gsk_...",
            model="llama-3.1-8b-instant",
        )
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "llama-3.1-8b-instant",
        **_kwargs: Any,
    ) -> None:
        if not api_key:
            raise ConfigurationError(
                "Groq API key is required. Pass api_key parameter or set "
                "ENGRAM_GROQ_API_KEY environment variable."
            )

        try:
            import httpx

            self._httpx = httpx
        except ImportError as e:
            raise ImportError(
                "httpx package not installed. Install with: pip install httpx"
            ) from e

        self._api_key = api_key
        self._model = model
        self._client = self._httpx.AsyncClient(
            base_url="https://api.groq.com/openai/v1",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60.0,
        )

        logger.info(f"Initialized Groq LLM provider: {model}")

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self,
        messages: list[LLMMessage] | list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        try:
            call_kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": _to_dict_messages(messages),
            }
            if max_tokens is not None:
                call_kwargs["max_tokens"] = max_tokens
            if temperature is not None:
                call_kwargs["temperature"] = temperature
            call_kwargs.update(kwargs)

            response = await request_with_retries(
                lambda: self._client.post("/chat/completions", json=call_kwargs)
            )
            data = response.json()

            return LLMResponse(
                content=data["choices"][0]["message"]["content"],
                model=data.get("model", self._model),
                usage={
                    "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": data.get("usage", {}).get(
                        "completion_tokens", 0
                    ),
                    "total_tokens": data.get("usage", {}).get("total_tokens", 0),
                },
                finish_reason=data["choices"][0].get("finish_reason"),
                raw=data,
            )
        except self._httpx.HTTPError as e:
            raise LLMProviderError(f"Groq API error: {e}", model=self._model) from e
        except Exception as e:
            raise LLMProviderError(f"Failed to complete: {e}", model=self._model) from e

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


# =============================================================================
# LiteLLM Provider (Universal)
# =============================================================================


@llm_registry.register("litellm", aliases=["universal", "any"])
class LiteLLMLLMProvider(LLMProvider):
    """LLM provider using LiteLLM for universal model access.

    LiteLLM provides a unified interface to 100+ LLM providers.
    See: https://docs.litellm.ai/docs/providers

    Args:
        model: Model name in LiteLLM format (e.g., "gpt-4", "claude-3-opus", "ollama/llama3").
        api_key: API key for the provider (if required).
        api_base: Custom API base URL (optional).

    Example:
        # OpenAI via LiteLLM
        provider = LiteLLMLLMProvider(model="gpt-4o-mini", api_key="sk-...")

        # Anthropic via LiteLLM
        provider = LiteLLMLLMProvider(
            model="claude-haiku-4-5-20251001", api_key="sk-ant-..."
        )

        # Ollama via LiteLLM
        provider = LiteLLMLLMProvider(model="ollama/llama3")
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            import litellm

            self._litellm = litellm
        except ImportError as e:
            raise ImportError(
                "LiteLLM package not installed. Install with: pip install litellm"
            ) from e

        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._extra_kwargs = kwargs

        logger.info(f"Initialized LiteLLM provider: {model}")

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self,
        messages: list[LLMMessage] | list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        try:
            call_kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": _to_dict_messages(messages),
            }
            if self._api_key:
                call_kwargs["api_key"] = self._api_key
            if self._api_base:
                call_kwargs["api_base"] = self._api_base
            if max_tokens is not None:
                call_kwargs["max_tokens"] = max_tokens
            if temperature is not None:
                call_kwargs["temperature"] = temperature
            call_kwargs.update(self._extra_kwargs)
            call_kwargs.update(kwargs)

            response = await self._litellm.acompletion(**call_kwargs)

            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model or self._model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens
                    if response.usage
                    else 0,
                    "completion_tokens": response.usage.completion_tokens
                    if response.usage
                    else 0,
                    "total_tokens": response.usage.total_tokens
                    if response.usage
                    else 0,
                    # Cached prefix tokens billed at a discount. Non-zero here
                    # confirms automatic prompt caching is firing across the
                    # repeated memory-context prefix of the multi-call reader.
                    "cached_tokens": (
                        getattr(
                            getattr(response.usage, "prompt_tokens_details", None),
                            "cached_tokens",
                            0,
                        )
                        or 0
                    )
                    if response.usage
                    else 0,
                },
                finish_reason=response.choices[0].finish_reason,
                raw=response,
            )
        except Exception as e:
            raise LLMProviderError(f"LiteLLM error: {e}", model=self._model) from e
