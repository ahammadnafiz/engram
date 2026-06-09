"""LLM provider system for Engram.

This module provides a pluggable LLM provider architecture.

Built-in providers:
- openai: OpenAI GPT models
- anthropic: Anthropic Claude models
- ollama: Ollama local LLMs
- groq: Groq inference API

Example:
    # Use built-in provider
    from engram.providers import get_llm_provider

    provider = get_llm_provider(
        "openai",
        api_key="sk-...",
        model="gpt-4o-mini",
    )

    response = await provider.complete([
        {"role": "user", "content": "Hello!"}
    ])

    # Register custom provider
    from engram.providers import llm_registry, LLMProvider

    @llm_registry.register("my-llm")
    class MyLLMProvider(LLMProvider):
        ...
"""

from engram.core.exceptions import LLMProviderError

# Import built-in providers to register them
from engram.providers.llm import builtin  # noqa: F401
from engram.providers.llm.protocol import LLMMessage, LLMProvider, LLMResponse
from engram.providers.llm.registry import get_llm_provider, llm_registry

__all__ = [
    "LLMMessage",
    "LLMProvider",
    "LLMProviderError",
    "LLMResponse",
    "get_llm_provider",
    "llm_registry",
]
