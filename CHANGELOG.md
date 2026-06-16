# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Batched memory-operation decisions: `add_conversation` now consolidates all
  extracted facts in a single LLM call, chunked into bounded concurrent
  sub-batches for large turns, instead of one round-trip per fact.
- Optional Chonkie recursive chunker for `record_long_input`
  (`ENGRAM_LONG_INPUT_CHUNKER=chonkie`, extra `engram[chunking]`), with
  automatic fallback to the builtin structure-aware splitter.
- Opt-in structured rolling-summary template
  (`ENGRAM_SUMMARY_STYLE=structured`): Goal / Constraints / Progress /
  Decisions / Next Steps / Critical Context, iteratively updated.
- Schema-version fast-path so `connect()` skips structural migration work when
  the database is already at the current version.
- `PostgresStorage.settings` accessor and `health_check(skip_embedding_test=...)`
  to skip the metered embedding probe.

### Changed
- Configured search weights now apply to all search modes. Hybrid and semantic
  defaults are unchanged; keyword mode now derives its weights from settings
  instead of hardcoded constants.
- `add_batch` in-batch near-duplicate detection vectorized with numpy
  (previously O(n^2) Python cosine).
- Memory-operation `DELETE` documented as a history-preserving supersede,
  matching `add_conversation`'s behavior.
- Per-scope write advisory lock keyed with 64-bit `hashtextextended` to reduce
  false sharing across scopes.

### Fixed
- `forget()` repoints the lineage head to the newest surviving revision instead
  of leaving `memory_lineages` pointing at a deleted row.
- `mypy --strict` clean across the package (typed the `revise()` memory_type).

## [0.3.0a2] - 2026-06-15

### Added
- Real OpenAI-backed memory chatbot example with command UI, Engram recall, task context, and memory cleanup commands.
- End-to-end API examples and tested documentation coverage for public code snippets.
- MkDocs Material theme customization with Mermaid diagram rendering and polished code block styling.

### Changed
- Reorganized documentation around quickstart, core concepts, long-running memory, API reference, examples, and operations.
- Improved Docker setup so repeated runs preserve existing `.env` secrets and only add missing Docker defaults.

### Fixed
- Ruff formatting drift in source, tests, and examples.
- Integration test environment setup no longer lets local `.env` redirect the caller database.

## [0.3.0a1] - 2026-06-09

### Added
- **Long-running task memory** with task runs, raw event ledgers, checkpoints, and memory jobs.
- **Typed production memory** for profile, project, task, preference, constraint, decision, and tool-result facts.
- **Deterministic critical recall** so critical user/project/task facts do not rely only on vector ranking.
- **Configurable memory policies** via `MemoryPolicy`, `TypeRule`, and `SlotRule`, with `default`, `legal`, and `coding_agent` presets.
- **Conflict and freshness metadata** using `critical_slot`, `conflict_key`, `status`, `version`, and `previous_versions`.
- **Recall observability** with `RecallTrace` and `trace_recall()` to inspect ranked, kept, trimmed, superseded, and missing memories.
- **Long-input ingestion** with `record_long_input()` and `build_long_input_context()` for source-anchored prompts and documents.
- **Long-input example** in `examples/long_input_usage.py`.
- **Security, contributing, and release docs** for OSS alpha publishing.

### Changed
- `Engram` now accepts `memory_policy="default" | "legal" | "coding_agent"` or a custom `MemoryPolicy`.
- README now documents alpha status, policies, long-input workflows, recall trace, and production caveats.
- Optional dependencies now expose provider-specific extras for Anthropic, Cohere, HTTP providers, LiteLLM, examples, and all providers.

### Fixed
- Positive allergy facts and negative allergy exclusions now use separate conflict slots, preventing accidental supersession.
- Search filters out superseded memories by default while trace APIs can still report superseded records.
- Environment examples now consistently use `ENGRAM_`-prefixed variables.

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
