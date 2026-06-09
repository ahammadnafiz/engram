"""LLM provider protocol for Engram.

This module defines the abstract base class for all LLM providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class LLMMessage:
    """A message in a conversation.

    Attributes:
        role: The role of the message sender ("system", "user", "assistant").
        content: The message content.
    """

    role: Literal["system", "user", "assistant"]
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary format."""
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    """Response from an LLM provider.

    Attributes:
        content: The generated text content.
        model: The model that generated the response.
        usage: Token usage information (if available).
        finish_reason: Why generation stopped (if available).
        raw: Raw response from the provider (for debugging).
    """

    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str | None = None
    raw: Any = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    All LLM providers must implement this interface. The provider
    is responsible for generating text completions.

    Example:
        class MyLLMProvider(LLMProvider):
            def __init__(self, model: str = "my-model"):
                self._model = model

            @property
            def model(self) -> str:
                return self._model

            async def complete(
                self,
                messages: list[LLMMessage] | list[dict],
                **kwargs,
            ) -> LLMResponse:
                # Your implementation
                return LLMResponse(content="Hello!", model=self._model)
    """

    @property
    @abstractmethod
    def model(self) -> str:
        """Get the model name.

        Returns:
            The model identifier being used.
        """
        ...

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage] | list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion for a conversation.

        Args:
            messages: List of messages in the conversation.
            max_tokens: Maximum tokens to generate (optional).
            temperature: Sampling temperature (optional).
            **kwargs: Provider-specific parameters.

        Returns:
            The generated response.

        Raises:
            LLMProviderError: If generation fails.
        """
        ...

    async def complete_text(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Simple text completion helper.

        Args:
            prompt: The user prompt.
            system: Optional system message.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Provider-specific parameters.

        Returns:
            The generated text content.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self.complete(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        return response.content

    def get_info(self) -> dict[str, Any]:
        """Get provider information.

        Returns:
            Dictionary with provider details.
        """
        return {
            "provider": self.__class__.__name__,
            "model": self.model,
        }
