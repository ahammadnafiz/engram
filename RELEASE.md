# Release Checklist

Engram is currently suitable for alpha/developer-preview releases.

## Before Tagging

1. Ensure the working tree contains only intended release files.
2. Update `src/engram/_version.py`.
3. Update `CHANGELOG.md`.
4. Run:

   ```bash
   python -m py_compile src/engram/client.py
   pytest tests/unit -q
   python -m build
   python -m twine check dist/*
   ```

5. Verify a fresh install:

   ```bash
   python -m venv /tmp/engram-release-venv
   source /tmp/engram-release-venv/bin/activate
   pip install dist/*.whl
   python -c "import engram; print(engram.__version__)"
   ```

6. Run one database-backed smoke test with PostgreSQL + pgvector:

   ```bash
   docker compose up -d postgres
   ENGRAM_DATABASE_URL=postgresql://engram:engram_secret@localhost:5432/engram \
     ENGRAM_EMBEDDING_PROVIDER=sentence-transformers \
     ENGRAM_EMBEDDING_MODEL=all-MiniLM-L6-v2 \
     python examples/long_input_usage.py
   ```

## Tagging

Use alpha tags until the public API stabilizes:

```bash
git tag v0.3.0a2
git push origin v0.3.0a2
```

GitHub Actions publishes tagged builds through trusted publishing when PyPI is
configured for this repository.

## Post-Release

- Confirm the PyPI package page renders README correctly.
- Confirm wheel includes `engram/sql/*.sql`.
- Create a GitHub release from the tag.
- Open follow-up issues for API stabilization, docs site, and integration test
  expansion.
