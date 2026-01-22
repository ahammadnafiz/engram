"""Provider registry system for Engram.

This module provides a generic registry for pluggable providers.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ProviderRegistry(Generic[T]):
    """Generic registry for provider classes.
    
    This registry allows dynamic registration and retrieval of provider
    implementations. Providers can be registered via decorator or explicit
    registration.
    
    Example:
        # Create a registry
        registry = ProviderRegistry[EmbeddingProvider]("embedding")
        
        # Register via decorator
        @registry.register("openai")
        class OpenAIProvider:
            ...
        
        # Register explicitly  
        registry.register("custom", MyCustomProvider)
        
        # Get a provider class
        provider_cls = registry.get("openai")
        
        # Create instance
        provider = registry.create("openai", api_key="sk-...")
    """
    
    def __init__(self, name: str) -> None:
        """Initialize the registry.
        
        Args:
            name: Human-readable name for this registry (e.g., "embedding", "llm").
        """
        self._name = name
        self._providers: dict[str, type[T]] = {}
        self._aliases: dict[str, str] = {}
        
    @property
    def name(self) -> str:
        """Get the registry name."""
        return self._name
    
    @property
    def available_providers(self) -> list[str]:
        """Get list of all registered provider names."""
        return list(self._providers.keys())
    
    def register(
        self,
        name: str,
        provider_cls: type[T] | None = None,
        *,
        aliases: list[str] | None = None,
    ) -> type[T] | Callable[[type[T]], type[T]]:
        """Register a provider class.
        
        Can be used as a decorator or called directly.
        
        Args:
            name: Unique provider name (e.g., "openai", "anthropic").
            provider_cls: Provider class to register. If None, returns decorator.
            aliases: Optional list of alternative names for this provider.
            
        Returns:
            The provider class, or a decorator if provider_cls is None.
            
        Example:
            # As decorator
            @registry.register("my-provider")
            class MyProvider:
                ...
            
            # Direct registration
            registry.register("my-provider", MyProvider)
        """
        def _register(cls: type[T]) -> type[T]:
            if name in self._providers:
                logger.warning(
                    f"Overwriting existing {self._name} provider: {name}"
                )
            
            self._providers[name] = cls
            logger.debug(f"Registered {self._name} provider: {name}")
            
            # Register aliases
            if aliases:
                for alias in aliases:
                    self._aliases[alias] = name
                    logger.debug(f"Registered alias {alias} -> {name}")
            
            return cls
        
        if provider_cls is not None:
            return _register(provider_cls)
        return _register
    
    def unregister(self, name: str) -> None:
        """Unregister a provider.
        
        Args:
            name: Provider name to unregister.
        """
        if name in self._providers:
            del self._providers[name]
            # Remove any aliases pointing to this provider
            self._aliases = {k: v for k, v in self._aliases.items() if v != name}
            logger.debug(f"Unregistered {self._name} provider: {name}")
    
    def get(self, name: str) -> type[T]:
        """Get a provider class by name.
        
        Args:
            name: Provider name or alias.
            
        Returns:
            The provider class.
            
        Raises:
            KeyError: If provider is not registered.
        """
        # Check for alias
        resolved_name = self._aliases.get(name, name)
        
        if resolved_name not in self._providers:
            available = ", ".join(self.available_providers)
            raise KeyError(
                f"Unknown {self._name} provider: '{name}'. "
                f"Available providers: {available}"
            )
        
        return self._providers[resolved_name]
    
    def create(self, name: str, **kwargs: Any) -> T:
        """Create a provider instance.
        
        Args:
            name: Provider name or alias.
            **kwargs: Arguments to pass to the provider constructor.
            
        Returns:
            A new provider instance.
            
        Raises:
            KeyError: If provider is not registered.
        """
        provider_cls = self.get(name)
        return provider_cls(**kwargs)
    
    def has(self, name: str) -> bool:
        """Check if a provider is registered.
        
        Args:
            name: Provider name or alias.
            
        Returns:
            True if provider exists, False otherwise.
        """
        resolved_name = self._aliases.get(name, name)
        return resolved_name in self._providers
    
    def __contains__(self, name: str) -> bool:
        """Support 'in' operator."""
        return self.has(name)
    
    def __repr__(self) -> str:
        return f"ProviderRegistry({self._name!r}, providers={self.available_providers})"

