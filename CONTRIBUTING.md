# Contributing

Thanks for helping improve Engram.

## Development Setup

```bash
git clone https://github.com/ahammadnafiz/engram.git
cd engram
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,examples]"
docker compose up -d postgres
pytest tests/unit -q
```

For local embeddings:

```bash
pip install -e ".[sentence-transformers]"
export ENGRAM_EMBEDDING_PROVIDER=sentence-transformers
export ENGRAM_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

For OpenAI:

```bash
pip install -e ".[openai]"
export ENGRAM_EMBEDDING_PROVIDER=openai
export ENGRAM_EMBEDDING_MODEL=text-embedding-3-small
export ENGRAM_OPENAI_API_KEY=...
```

## Test Commands

```bash
pytest tests/unit -q
pytest tests/integration --run-integration -q
ruff check src tests examples
ruff format --check src tests examples
```

Integration tests require PostgreSQL with pgvector and `ENGRAM_DATABASE_URL`.

## Pull Request Expectations

- Keep changes focused.
- Add or update tests for behavior changes.
- Update README/examples when public APIs change.
- Do not commit secrets, local `.env`, database dumps, or generated caches.
- For memory behavior changes, include traceability: stored, ranked, kept,
  trimmed, superseded, or missing.

## API Stability

Engram is currently alpha. Public APIs may change, but changes should be
intentional and documented in `CHANGELOG.md`.
