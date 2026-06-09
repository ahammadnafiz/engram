# Security and Privacy

Engram is a memory system. By design, it can store sensitive user facts,
conversation history, source documents, tool results, and derived summaries.
Treat every deployment as sensitive data infrastructure.

## Reporting Vulnerabilities

Please report security issues privately by emailing the maintainer listed in
`pyproject.toml`. Do not open a public issue for vulnerabilities involving data
exposure, credential handling, tenant isolation, deletion, or prompt/data
exfiltration.

## Data Handling Guidance

- Scope memory by `agent_id` and `user_id`; do not share user IDs across tenants.
- Store only data your application is allowed to retain.
- Define retention rules before production use.
- Use `forget()` for single memories and `purge()` for agent/user memory removal.
- Use `redact_event()` when the raw event ledger contains content that must be
  removed while retaining audit metadata.
- For source documents, keep source chunk anchors (`source_event_id`,
  `chunk_id`, character range, and `quote_hash`) so answers can be audited.
- Encrypt database disks/backups and restrict direct database access.
- Keep API keys in environment variables or secret managers; never commit `.env`.

## Provider and API-Key Notes

Engram can call external embedding and LLM providers. When using cloud
providers, memory contents and prompts sent for embedding/completion may leave
your infrastructure according to that provider's terms.

For stricter privacy, use a local embedding provider such as
`sentence-transformers` and a local LLM provider such as Ollama.

## Current Alpha Caveats

- Public APIs are still stabilizing.
- The policy system is configurable, but domain policies should be reviewed
  before production use.
- Legal/source-grounded use cases should retrieve source chunks before relying
  on derived memory summaries.
