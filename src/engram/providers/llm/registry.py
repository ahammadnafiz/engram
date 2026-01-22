"""LLM provider registry for Engram.

This module provides the global LLM provider registry.
"""

from __future__ import annotations

import logging
from typing import Any

from engram.providers.registry import ProviderRegistry
from engram.providers.llm.protocol import LLMProvider

logger = logging.getLogger(__name__)

# Global LLM provider registry
llm_registry: ProviderRegistry[LLMProvider] = ProviderRegistry("llm")


def get_llm_provider(
    provider_name: str,
    **kwargs: Any,
) -> LLMProvider:
    """Create an LLM provider instance.
    
    This is the main factory function for creating LLM providers.
    
    Args:
        provider_name: Name of the provider (e.g., "openai", "anthropic").
        **kwargs: Provider-specific configuration:
        
            For "openai":
                - api_key: OpenAI API key (required)
                - model: Model name (default: "gpt-4o-mini")
                - base_url: Custom API base URL (optional)
                
            For "anthropic":
                - api_key: Anthropic API key (required)
                - model: Model name (default: "claude-3-haiku-20240307")
                
            For "ollama":
                - model: Model name (required)
                - base_url: Ollama server URL (default: "http://localhost:11434")
                
            For "groq":
                - api_key: Groq API key (required)
                - model: Model name (default: "llama-3.1-8b-instant")
                
    Returns:
        An initialized LLM provider.
        
    Raises:
        KeyError: If provider is not registered.
        ConfigurationError: If required configuration is missing.
        
    Example:
        # OpenAI
        provider = get_llm_provider(
            "openai",
            api_key="sk-...",
            model="gpt-4o-mini",
        )
        
        # Anthropic
        provider = get_llm_provider(
            "anthropic",
            api_key="sk-ant-...",
            model="claude-3-haiku-20240307",
        )
        
        # Local with Ollama
        provider = get_llm_provider("ollama", model="llama3.2")
    """
    logger.info(f"Creating LLM provider: {provider_name}")
    return llm_registry.create(provider_name, **kwargs)


def list_llm_providers() -> list[str]:
    """List all registered LLM providers.
    
    Returns:
        List of provider names.
    """
    return llm_registry.available_providers

