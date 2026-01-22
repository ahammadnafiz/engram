"""Configuration management for Engram.

This module provides configuration classes using pydantic-settings for
type-safe, validated configuration with environment variable support.

Provider Architecture:
    Engram uses a pluggable provider system. Any embedding or LLM provider
    can be used by specifying the provider name and relevant configuration.
    
    Built-in embedding providers: openai, sentence-transformers, cohere, ollama, huggingface
    Built-in LLM providers: openai, anthropic, ollama, groq, litellm
    
    Custom providers can be registered via the provider registry system.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """PostgreSQL database connection settings.

    All settings can be configured via environment variables with the
    ENGRAM_ prefix (e.g., ENGRAM_DATABASE_URL).
    """

    model_config = SettingsConfigDict(
        env_prefix="ENGRAM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Connection settings
    database_url: str = Field(
        default="postgresql://localhost:5432/engram",
        description="PostgreSQL connection URL",
    )
    min_pool_size: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Minimum connection pool size",
    )
    max_pool_size: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum connection pool size",
    )
    connection_timeout: float = Field(
        default=30.0,
        gt=0,
        description="Connection timeout in seconds",
    )
    command_timeout: float = Field(
        default=60.0,
        gt=0,
        description="Command timeout in seconds",
    )

    @field_validator("max_pool_size")
    @classmethod
    def validate_pool_sizes(cls, v: int, info: dict) -> int:  # type: ignore[type-arg]
        """Ensure max_pool_size >= min_pool_size."""
        min_size = info.data.get("min_pool_size", 5)
        if v < min_size:
            msg = f"max_pool_size ({v}) must be >= min_pool_size ({min_size})"
            raise ValueError(msg)
        return v


class EmbeddingSettings(BaseSettings):
    """Embedding provider settings.
    
    Supports any registered embedding provider. Built-in providers:
    - openai: OpenAI text-embedding models
    - sentence-transformers (aliases: st, local, sbert): Local Sentence Transformers
    - cohere: Cohere embed models
    - ollama: Ollama local embeddings
    - huggingface (aliases: hf): HuggingFace Inference API
    """

    model_config = SettingsConfigDict(
        env_prefix="ENGRAM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Provider settings - any registered provider name
    embedding_provider: str = Field(
        default="openai",
        description="Embedding provider name (e.g., 'openai', 'sentence-transformers', 'cohere', 'ollama')",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Model name for embeddings (provider-specific)",
    )
    embedding_dimension: int | None = Field(
        default=None,
        description="Embedding vector dimension (auto-detected if None)",
    )

    @field_validator("embedding_dimension", mode="before")
    @classmethod
    def coerce_embedding_dimension(cls, v: int | str | None) -> int | None:
        """Coerce string to int for embedding_dimension (env vars are strings)."""
        if v is None or v == "":
            return None
        if isinstance(v, str):
            return int(v)
        return v

    # API Keys for various providers
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key",
    )
    openai_base_url: str | None = Field(
        default=None,
        description="Custom OpenAI API base URL",
    )
    cohere_api_key: str | None = Field(
        default=None,
        description="Cohere API key",
    )
    hf_api_key: str | None = Field(
        default=None,
        description="HuggingFace API key",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL",
    )

    # Batching settings
    embedding_batch_size: int = Field(
        default=100,
        ge=1,
        le=2048,
        description="Batch size for embedding requests",
    )

    # Caching settings
    embedding_cache_size: int = Field(
        default=1000,
        ge=0,
        description="LRU cache size for embeddings (0 to disable)",
    )


class LLMSettings(BaseSettings):
    """LLM provider settings for fact extraction and other AI tasks.
    
    Supports any registered LLM provider. Built-in providers:
    - openai (aliases: gpt, chatgpt): OpenAI GPT models
    - anthropic (aliases: claude): Anthropic Claude models
    - ollama (aliases: local): Ollama local LLMs
    - groq: Groq fast inference
    - litellm (aliases: universal, any): LiteLLM universal interface
    """

    model_config = SettingsConfigDict(
        env_prefix="ENGRAM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Provider settings
    llm_provider: str | None = Field(
        default=None,
        description="LLM provider name (e.g., 'openai', 'anthropic', 'ollama'). None to disable LLM features.",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model name (provider-specific)",
    )

    # API Keys (reuses embedding keys where applicable)
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key",
    )
    groq_api_key: str | None = Field(
        default=None,
        description="Groq API key",
    )


class SearchSettings(BaseSettings):
    """Search and scoring settings."""

    model_config = SettingsConfigDict(
        env_prefix="ENGRAM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Hybrid search weights (must sum to 1.0)
    weight_semantic: float = Field(
        default=0.40,
        ge=0,
        le=1,
        description="Weight for semantic similarity score",
    )
    weight_keyword: float = Field(
        default=0.20,
        ge=0,
        le=1,
        description="Weight for keyword/BM25 score",
    )
    weight_decay: float = Field(
        default=0.25,
        ge=0,
        le=1,
        description="Weight for time decay score",
    )
    weight_importance: float = Field(
        default=0.15,
        ge=0,
        le=1,
        description="Weight for importance score",
    )

    # Decay settings
    decay_rate: float = Field(
        default=0.995,
        gt=0,
        lt=1,
        description="Decay rate per hour (higher = slower decay)",
    )

    # Search limits
    default_search_limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Default number of results to return",
    )
    max_search_limit: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of results allowed",
    )

    @field_validator("weight_importance")
    @classmethod
    def validate_weights_sum(cls, v: float, info: dict) -> float:  # type: ignore[type-arg]
        """Ensure all weights sum to approximately 1.0."""
        total = (
            info.data.get("weight_semantic", 0.40)
            + info.data.get("weight_keyword", 0.20)
            + info.data.get("weight_decay", 0.25)
            + v
        )
        if not (0.99 <= total <= 1.01):
            msg = f"Search weights must sum to 1.0, got {total}"
            raise ValueError(msg)
        return v


class EngramSettings(BaseSettings):
    """Main Engram configuration combining all settings.

    This is the primary configuration class that aggregates all settings
    and provides a unified interface for configuration management.
    
    Provider Architecture:
        Engram uses a pluggable provider system for both embeddings and LLMs.
        Specify any registered provider by name - no hardcoded limitations.
        
        Embedding providers: openai, sentence-transformers, cohere, ollama, huggingface
        LLM providers: openai, anthropic, ollama, groq, litellm

    Example:
        # Load from environment variables
        settings = EngramSettings()

        # OpenAI for both embedding and LLM
        settings = EngramSettings(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            llm_provider="openai",
            llm_model="gpt-4o-mini",
            openai_api_key="sk-...",
        )
        
        # Local embeddings, Ollama for LLM
        settings = EngramSettings(
            embedding_provider="sentence-transformers",
            embedding_model="all-MiniLM-L6-v2",
            llm_provider="ollama",
            llm_model="llama3.2",
        )
        
        # Cohere embeddings, Anthropic for LLM
        settings = EngramSettings(
            embedding_provider="cohere",
            cohere_api_key="...",
            llm_provider="anthropic",
            anthropic_api_key="sk-ant-...",
        )
    """

    model_config = SettingsConfigDict(
        env_prefix="ENGRAM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Database Settings
    # -------------------------------------------------------------------------
    database_url: str = Field(
        default="postgresql://localhost:5432/engram",
        description="PostgreSQL connection URL",
    )
    min_pool_size: int = Field(default=5, ge=1, le=100)
    max_pool_size: int = Field(default=20, ge=1, le=100)
    connection_timeout: float = Field(default=30.0, gt=0)
    command_timeout: float = Field(default=60.0, gt=0)

    # -------------------------------------------------------------------------
    # Embedding Provider Settings
    # -------------------------------------------------------------------------
    embedding_provider: str = Field(
        default="openai",
        description="Embedding provider name (any registered provider)",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name (provider-specific)",
    )
    embedding_dimension: int | None = Field(
        default=None,
        description="Embedding dimension (auto-detected if None)",
    )
    embedding_batch_size: int = Field(default=100, ge=1, le=2048)
    embedding_cache_size: int = Field(default=1000, ge=0)

    # -------------------------------------------------------------------------
    # LLM Provider Settings
    # -------------------------------------------------------------------------
    llm_provider: str | None = Field(
        default=None,
        description="LLM provider name (None to disable LLM features)",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model name (provider-specific)",
    )

    # -------------------------------------------------------------------------
    # API Keys (shared across embedding and LLM where applicable)
    # -------------------------------------------------------------------------
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    openai_base_url: str | None = Field(default=None, description="Custom OpenAI base URL")
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key")
    cohere_api_key: str | None = Field(default=None, description="Cohere API key")
    groq_api_key: str | None = Field(default=None, description="Groq API key")
    hf_api_key: str | None = Field(default=None, description="HuggingFace API key")
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")

    # -------------------------------------------------------------------------
    # Search Settings
    # -------------------------------------------------------------------------
    weight_semantic: float = Field(default=0.40, ge=0, le=1)
    weight_keyword: float = Field(default=0.20, ge=0, le=1)
    weight_decay: float = Field(default=0.25, ge=0, le=1)
    weight_importance: float = Field(default=0.15, ge=0, le=1)
    decay_rate: float = Field(default=0.995, gt=0, lt=1)
    default_search_limit: int = Field(default=10, ge=1, le=100)
    max_search_limit: int = Field(default=100, ge=1, le=1000)

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    log_level: str = Field(default="INFO")
    log_sql_queries: bool = Field(default=False)

    @field_validator("embedding_dimension", mode="before")
    @classmethod
    def coerce_embedding_dimension(cls, v: int | str | None) -> int | None:
        """Coerce string to int for embedding_dimension (env vars are strings)."""
        if v is None or v == "":
            return None
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError as e:
                raise ValueError(f"Invalid embedding_dimension: {v!r}") from e
        return v

    @field_validator("max_pool_size")
    @classmethod
    def validate_pool_sizes(cls, v: int, info: dict) -> int:  # type: ignore[type-arg]
        """Ensure max_pool_size >= min_pool_size."""
        min_size = info.data.get("min_pool_size", 5)
        if v < min_size:
            msg = f"max_pool_size ({v}) must be >= min_pool_size ({min_size})"
            raise ValueError(msg)
        return v

    @field_validator("weight_importance")
    @classmethod
    def validate_weights_sum(cls, v: float, info: dict) -> float:  # type: ignore[type-arg]
        """Ensure all search weights sum to approximately 1.0."""
        total = (
            info.data.get("weight_semantic", 0.40)
            + info.data.get("weight_keyword", 0.20)
            + info.data.get("weight_decay", 0.25)
            + v
        )
        if not (0.99 <= total <= 1.01):
            msg = f"Search weights must sum to 1.0, got {total:.3f}"
            raise ValueError(msg)
        return v
    
    def get_embedding_provider_kwargs(self) -> dict[str, Any]:
        """Get kwargs for creating an embedding provider.
        
        Returns:
            Dictionary of provider-specific configuration.
        """
        kwargs: dict[str, Any] = {"model": self.embedding_model}
        
        if self.embedding_dimension:
            kwargs["dimension"] = self.embedding_dimension
            
        # Add provider-specific config based on embedding_provider
        if self.embedding_provider in ("openai", "openai-embedding"):
            if self.openai_api_key:
                kwargs["api_key"] = self.openai_api_key
            if self.openai_base_url:
                kwargs["base_url"] = self.openai_base_url
        elif self.embedding_provider == "cohere":
            if self.cohere_api_key:
                kwargs["api_key"] = self.cohere_api_key
        elif self.embedding_provider in ("huggingface", "hf"):
            if self.hf_api_key:
                kwargs["api_key"] = self.hf_api_key
        elif self.embedding_provider in ("ollama", "ollama-embedding"):
            if self.ollama_base_url:
                kwargs["base_url"] = self.ollama_base_url
            
        return kwargs
    
    def get_llm_provider_kwargs(self) -> dict[str, Any]:
        """Get kwargs for creating an LLM provider.
        
        Returns:
            Dictionary of provider-specific configuration.
        """
        kwargs: dict[str, Any] = {"model": self.llm_model}
        
        # Add provider-specific API keys based on llm_provider
        if self.llm_provider in ("openai", "gpt", "chatgpt"):
            if self.openai_api_key:
                kwargs["api_key"] = self.openai_api_key
            if self.openai_base_url:
                kwargs["base_url"] = self.openai_base_url
        elif self.llm_provider in ("anthropic", "claude"):
            if self.anthropic_api_key:
                kwargs["api_key"] = self.anthropic_api_key
        elif self.llm_provider == "groq":
            if self.groq_api_key:
                kwargs["api_key"] = self.groq_api_key
        elif self.llm_provider in ("ollama", "local"):
            if self.ollama_base_url:
                kwargs["base_url"] = self.ollama_base_url
                
        return kwargs


@lru_cache(maxsize=1)
def get_settings() -> EngramSettings:
    """Get cached settings instance.

    This function returns a cached instance of EngramSettings,
    ensuring configuration is only loaded once per process.

    Returns:
        Cached EngramSettings instance.

    Example:
        settings = get_settings()
        print(settings.database_url)
    """
    return EngramSettings()


def clear_settings_cache() -> None:
    """Clear the settings cache.
    
    Useful for testing or when configuration needs to be reloaded.
    """
    get_settings.cache_clear()
