"""Unit tests for integration-test environment setup."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_integration_environment_keeps_explicit_database_url(
    monkeypatch, tmp_path: Path
) -> None:
    """A local .env must not redirect integration tests away from the caller DB."""
    from conftest import configure_integration_environment

    dotenv = tmp_path / ".env"
    dotenv.write_text("ENGRAM_DATABASE_URL=postgresql://dotenv/db\n")
    monkeypatch.setenv("ENGRAM_DATABASE_URL", "postgresql://explicit/db")
    monkeypatch.delenv("ENGRAM_TEST_DATABASE_URL", raising=False)

    database_url = configure_integration_environment(env_path=dotenv)

    assert database_url == "postgresql://explicit/db"
    assert database_url == os.environ["ENGRAM_DATABASE_URL"]


def test_integration_environment_prefers_test_database_url(
    monkeypatch, tmp_path: Path
) -> None:
    """ENGRAM_TEST_DATABASE_URL can isolate integration tests from app config."""
    from conftest import configure_integration_environment

    monkeypatch.setenv("ENGRAM_DATABASE_URL", "postgresql://app/db")
    monkeypatch.setenv("ENGRAM_TEST_DATABASE_URL", "postgresql://test/db")

    database_url = configure_integration_environment(env_path=tmp_path / ".env")

    assert database_url == "postgresql://test/db"
    assert database_url == os.environ["ENGRAM_DATABASE_URL"]


def test_integration_environment_sets_local_embedding_defaults(
    monkeypatch, tmp_path: Path
) -> None:
    from conftest import configure_integration_environment

    monkeypatch.delenv("ENGRAM_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("ENGRAM_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("ENGRAM_EMBEDDING_DIMENSION", raising=False)

    configure_integration_environment(local_embeddings=True, env_path=tmp_path / ".env")

    assert os.environ["ENGRAM_EMBEDDING_PROVIDER"] == "sentence-transformers"
    assert os.environ["ENGRAM_EMBEDDING_MODEL"] == "all-MiniLM-L6-v2"
    assert os.environ["ENGRAM_EMBEDDING_DIMENSION"] == "384"
    assert os.environ["ENGRAM_ALLOW_EMBEDDING_DIMENSION_CHANGE"] == "false"
