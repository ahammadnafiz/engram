# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-01-23

### Added
- **Two-Column Memory System** — Separates `fact` (embedded for search) and `main_content` (conversation context, not embedded) for cost-effective storage
- **LLM Error Handling** — New `LLMError`, `LLMConnectionError`, `LLMRateLimitError` exceptions for robust error handling
- **Async HTTP Client Cleanup** — Ollama and HuggingFace providers now support proper async `close()` for HTTP clients

### Changed
- **README Overhaul** — Updated project description, installation instructions, architecture overview, and documentation links
- **Chatbot Improvements**:
  - Default embedding provider switched to `sentence-transformers` (local, free)
  - Enhanced system prompt with structured memory rules, personality traits, and response formatting
  - Refined memory extraction rules for better fact retrieval and relationship preservation
  - Increased similarity thresholds: duplicate detection (0.90), relevance filtering (0.55)
- **Documentation Updates**:
  - New diagrams for two-column memory storage and retrieval flows
  - Added database schema documentation
  - Cost comparison and search vs storage analysis
  - Removed outdated CHATBOT_CONCEPT.md

### Removed
- Unused OpenAI and Sentence Transformers embedding provider modules (consolidated into main providers)
- "Mem0-style" references in documentation for clarity

### Fixed
- Integration tests improved with better error handling and memory retrieval
- Environment setup in tests enhanced for better configuration management

## [0.1.1] - 2026-01-22

### Fixed
- **High Priority**
  - `embed_batch` now validates output length matches input, raises `EmbeddingError` on mismatch
  - Fixed inconsistent tuple format in chatbot `consolidate_memories` (was 3-tuple, now 2-tuple)

- **Medium Priority**
  - Fixed task list memory leak in chatbot - added cleanup for completed background tasks
  - Added validation for negative `importance_boost` in `reinforce()` method
  - Implemented full search mode support (`hybrid`, `semantic`, `keyword`) in memory store
  - Added `validate_pool_sizes` and `validate_weights_sum` validators to `EngramSettings`
  - Fixed embedding cache to use proper LRU eviction (was FIFO)
  - Fixed HuggingFace `embed_batch` dimension auto-detection and response parsing
  - Replaced deprecated `asyncio.get_event_loop()` with `asyncio.get_running_loop()`
  - Added `skip_embedding_test` parameter to health check to avoid API costs
  - Added `verify_memories` parameter to `relate_batch()` for memory existence validation
  - Fixed `find_path()` to properly raise `GraphError` instead of silently returning None
  - Fixed silent duplicate memory handling - now returns existing memory on conflict
  - Fixed multiple system messages in Anthropic provider - now concatenates them

- **Low Priority**
  - Replaced deprecated `datetime.utcnow()` with timezone-aware `datetime.now(timezone.utc)`
  - Added `close()` method for `ThreadPoolExecutor` cleanup in SentenceTransformers provider
  - Renamed `ConnectionError` to `DatabaseConnectionError` to avoid shadowing Python built-in
  - Improved DELETE result parsing with safe fallback on unexpected format
  - Added logging for silent exceptions in `_get_current_vector_dimension`

### Changed
- Docker setup script now passes PostgreSQL password via `PGPASSWORD` environment variable
- Improved password generation in docker-setup.sh with multiple secure fallbacks
- Schema verification now validates numeric result before comparison

## [0.1.0] - 2026-01-22

### Added
- Initial release with MVP features
