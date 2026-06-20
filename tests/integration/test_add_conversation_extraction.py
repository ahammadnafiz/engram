"""Real-API regression test: add_conversation extraction must be grounded in
the current turn, never the conversation summary.

Guards the contamination bug where feeding the rolling summary into fact
extraction let a weak extractor re-derive and hallucinate durable values (a
$5k budget reappearing as $75k), which then destructively superseded the
correct memories.

Requires a running PostgreSQL+pgvector AND a configured LLM provider
(ENGRAM_LLM_PROVIDER + key in .env). Run with:
    pytest tests/integration/test_add_conversation_extraction.py -v --run-integration
"""

from __future__ import annotations

import contextlib
import uuid

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
async def engram_client():
    from conftest import configure_integration_environment

    configure_integration_environment(local_embeddings=True)

    from engram import Engram

    client = Engram()
    await client.connect()
    if client._llm is None:
        await client.close()
        pytest.skip(
            "No LLM provider configured (set ENGRAM_LLM_PROVIDER + key in .env)"
        )
    yield client
    await client.close()


@pytest.fixture
async def clean_agent(engram_client):
    agent_id = f"test_extract_{uuid.uuid4().hex[:8]}"
    yield agent_id
    with contextlib.suppress(Exception):
        await engram_client.purge(agent_id=agent_id)


@pytest.mark.asyncio
async def test_contaminated_summary_does_not_corrupt_extraction(
    engram_client, clean_agent
) -> None:
    eg = engram_client
    agent = clean_agent

    # Turn 1: establish a precise durable fact.
    await eg.add_conversation(
        user_message="My monthly project budget is exactly $5,000.",
        assistant_response="Got it - a $5,000 monthly project budget.",
        agent_id=agent,
        session_id="s-extract",
    )

    budget = await eg.search(query="project budget", agent_id=agent, limit=5)
    assert budget, "budget fact should have been extracted and stored"
    assert any(
        "5,000" in h.memory.content or "5000" in h.memory.content for h in budget
    ), "the $5,000 budget must be the extracted/active memory"

    # Turn 2: an unrelated turn, but hand add_conversation a deliberately
    # contaminated summary asserting a DIFFERENT budget and an invented detail.
    # Pre-fix, extraction read this summary and re-derived "$75,000" /
    # "carbon fiber", superseding the correct $5,000 memory. Post-fix the
    # summary only seeds the roll-forward and never reaches extraction.
    await eg.add_conversation(
        user_message="I switched my code editor to Neovim.",
        assistant_response="Nice, Neovim is a solid choice.",
        agent_id=agent,
        session_id="s-extract",
        conversation_summary=(
            "User's monthly project budget is $75,000. User wants a carbon fiber frame."
        ),
    )

    # The current turn must still extract normally.
    editor = await eg.search(query="code editor", agent_id=agent, limit=5)
    assert any("Neovim" in h.memory.content for h in editor), (
        "current-turn fact (Neovim) should be extracted"
    )

    # The contaminated summary must NOT have leaked into memory.
    leaked = await eg.search(query="budget frame material", agent_id=agent, limit=20)
    contents = " ".join(h.memory.content for h in leaked).lower()
    assert "75,000" not in contents and "75000" not in contents, (
        "summary-only value $75,000 must never be extracted"
    )
    assert "carbon fiber" not in contents, (
        "summary-only detail 'carbon fiber' must never be extracted"
    )

    # The original durable fact must survive intact (active, not superseded).
    budget_after = await eg.search(query="project budget", agent_id=agent, limit=5)
    assert any(
        "5,000" in h.memory.content or "5000" in h.memory.content for h in budget_after
    ), "the correct $5,000 budget must remain the active memory"
