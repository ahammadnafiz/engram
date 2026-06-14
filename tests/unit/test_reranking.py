"""Unit tests for cross-encoder reranking."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from engram.core.exceptions import ConfigurationError
from engram.memory.models import Memory, SearchResult
from engram.reranking import CrossEncoderReranker


def make_result(memory_id: str, content: str, score: float) -> SearchResult:
    now = datetime.now(timezone.utc)
    return SearchResult(
        memory=Memory(
            memory_id=memory_id,
            agent_id="agent",
            content=content,
            fact=content,
            created_at=now,
            last_accessed_at=now,
        ),
        score=score,
    )


class TestCrossEncoderReranker:
    @pytest.mark.asyncio
    async def test_reorders_by_cross_encoder_score(self) -> None:
        reranker = CrossEncoderReranker()
        # Hybrid ranked c first, but the cross-encoder prefers a.
        results = [
            make_result("c", "irrelevant chatter", 0.9),
            make_result("a", "the user's dog is named Biscuit", 0.5),
            make_result("b", "weather talk", 0.4),
        ]
        model = MagicMock()
        model.predict = MagicMock(return_value=[0.1, 0.95, 0.2])
        reranker._model = model

        reranked = await reranker.rerank("what is the dog's name?", results, top_k=2)

        assert [r.memory.memory_id for r in reranked] == ["a", "b"]
        # Original hybrid scores are preserved.
        assert reranked[0].score == 0.5
        pairs = model.predict.call_args[0][0]
        assert pairs[0] == ("what is the dog's name?", "irrelevant chatter")

    @pytest.mark.asyncio
    async def test_short_circuits_without_model_for_single_result(self) -> None:
        reranker = CrossEncoderReranker()
        results = [make_result("a", "only one", 0.5)]

        reranked = await reranker.rerank("q", results, top_k=10)

        assert reranked == results
        assert reranker._model is None  # model never loaded

    @pytest.mark.asyncio
    async def test_missing_dependency_raises_configuration_error(self) -> None:
        reranker = CrossEncoderReranker()
        results = [make_result("a", "x", 0.5), make_result("b", "y", 0.4)]
        with (
            patch.dict("sys.modules", {"sentence_transformers": None}),
            pytest.raises(ConfigurationError, match="sentence-transformers"),
        ):
            await reranker.rerank("q", results, top_k=2)


class TestClientRerankPath:
    @pytest.mark.asyncio
    async def test_search_overfetches_then_returns_top_limit(self) -> None:
        from unittest.mock import AsyncMock

        from engram import Engram

        engram = Engram(database_url="postgresql://u:p@localhost/db")
        engram._connected = True
        candidates = [make_result(str(i), f"fact {i}", 0.5) for i in range(20)]
        engram._memory_store = MagicMock()
        engram._memory_store.search = AsyncMock(return_value=candidates)
        reranker = MagicMock()
        reranker.rerank = AsyncMock(return_value=candidates[:3])
        engram._reranker = reranker

        results = await engram.search("q", "agent", limit=3, rerank=True)

        assert len(results) == 3
        query = engram._memory_store.search.call_args[0][0]
        multiplier = engram._settings.search_candidate_multiplier
        assert query.limit == min(3 * multiplier, engram._settings.max_search_limit)
        reranker.rerank.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_search_without_rerank_keeps_limit_and_skips_reranker(self) -> None:
        from unittest.mock import AsyncMock

        from engram import Engram

        engram = Engram(database_url="postgresql://u:p@localhost/db")
        engram._connected = True
        engram._memory_store = MagicMock()
        engram._memory_store.search = AsyncMock(return_value=[])

        await engram.search("q", "agent", limit=3)

        query = engram._memory_store.search.call_args[0][0]
        assert query.limit == 3
        assert engram._reranker is None


class TestRerankerBackend:
    @pytest.mark.asyncio
    async def test_backend_passed_to_cross_encoder(self) -> None:
        reranker = CrossEncoderReranker("some-model", backend="onnx")
        results = [make_result("a", "x", 0.5), make_result("b", "y", 0.4)]
        st_module = MagicMock()
        st_module.CrossEncoder.return_value.predict = MagicMock(return_value=[0.2, 0.8])
        with patch.dict("sys.modules", {"sentence_transformers": st_module}):
            reranked = await reranker.rerank("q", results, top_k=2)

        st_module.CrossEncoder.assert_called_once_with("some-model", backend="onnx")
        assert [r.memory.memory_id for r in reranked] == ["b", "a"]

    @pytest.mark.asyncio
    async def test_client_builds_reranker_with_configured_backend(self) -> None:
        from unittest.mock import AsyncMock

        from engram import Engram

        engram = Engram(database_url="postgresql://u:p@localhost/db")
        engram._connected = True
        engram._settings = engram._settings.model_copy(
            update={"reranker_backend": "onnx"}
        )
        engram._memory_store = MagicMock()
        engram._memory_store.search = AsyncMock(return_value=[])

        with patch("engram.client.CrossEncoderReranker") as reranker_cls:
            reranker_cls.return_value.rerank = AsyncMock(return_value=[])
            await engram.search("q", "agent", limit=3, rerank=True)

        reranker_cls.assert_called_once_with(
            engram._settings.reranker_model, backend="onnx"
        )


class TestDeepSearchRerank:
    @pytest.mark.asyncio
    async def test_deep_search_reranks_merged_pool(self) -> None:
        from unittest.mock import AsyncMock

        from engram import Engram

        engram = Engram(database_url="postgresql://u:p@localhost/db")
        engram._connected = True
        engram._llm = None  # falls back to the original query only
        candidates = [make_result(str(i), f"fact {i}", 0.9 - i * 0.1) for i in range(6)]
        engram._memory_store = MagicMock()
        engram._memory_store.search = AsyncMock(return_value=candidates)
        reranker = MagicMock()
        reranker.rerank = AsyncMock(return_value=candidates[:2])
        engram._reranker = reranker

        results = await engram.deep_search("q", "agent", limit=2, rerank=True)

        assert len(results) == 2
        # Reranker sees the merged pool (6 unique) and cuts to limit.
        args, kwargs = reranker.rerank.call_args
        assert args[0] == "q"
        assert len(args[1]) == 6
        assert kwargs["top_k"] == 2
