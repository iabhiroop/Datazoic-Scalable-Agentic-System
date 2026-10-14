# Scaffold scope and LLD coverage

Status: database/auth/run-API foundation, scoped catalog queries, registry lifecycle,
binding/hash helpers, embeddings and generic read-only graph/worker implemented.
Clarification/resume, final projection repair and single-worker locking are
database-tested with mocked models. Business writes and durable retry timers remain pending.
[The supplied LLD](LLD.md) is the design reference; its code samples
are not assumed to be complete application code.

## Component ownership

- API submission/status/input/cancel and health: `app/api/`, `app/main.py`.
- Settings/lifecycle/version checks: `app/settings.py`, `app/main.py`, `app/db/session.py`.
- Trusted principal, scopes, ownership, approval and redaction: `app/security.py`.
- All 13 ORM mappings: `app/db/models.py`; scoped queries: `repositories.py`.
- Application migrations: `migrations/`; checkpointer setup: `Database.setup_checkpoints`.
- Tool cards, immutable registry, adapter allowlist: `app/tools/contracts.py`, `registry.py`.
- Tool/document onboarding, extraction and chunks: `app/knowledge/ingestion.py`.
- Offline clustering proposals and reviewed labels: `app/knowledge/taxonomy.py`.
- Embeddings, lexical/vector retrieval and fusion: `app/knowledge/retrieval.py`.
- Context budgeting and schema preservation: `app/knowledge/context.py`.
- Intent/plan/state/resume contracts: `app/agent/state.py`.
- Bindings, plan validation and stable action slots: `app/agent/planning.py`.
- Twelve nodes, routing and checkpoint graph: `app/agent/graph.py`, `nodes.py`.
- Model tasks: `app/agent/llm.py`; provider gateway: `app/agent/gateway.py`;
  Google AI Studio adapter: `app/agent/gemini.py`. Prompt resources remain drafts.
- Operation guard, preconditions, verification, credentials and reconciliation: `app/tools/gateway.py`.
- External invocation/error normalization: `app/tools/adapters.py`.
- Separate registered internal tools: `rag_pipeline.py`, `system_search.py`.
- Singleton worker, claims, heartbeat, durable waits and recovery: `app/worker/`.
- Trajectory summary/review/publication: `app/knowledge/trajectories.py`.
- Authorized payload references and retention: `app/storage.py`.
- Scope/revision-aware optional Redis caching: `app/cache.py`.
- Correlated Langfuse/infrastructure telemetry: `app/observability.py`.
- Administrative command skeletons: `app/cli.py`.
- Contract/auth tests: `tests/unit/`; PostgreSQL transaction/constraint tests and remaining scenarios: `tests/integration/`.

Four small modules (cache, storage, taxonomy, trajectories) extend the agreed
layout to represent every LLD component without introducing extra packages.

## Implementation checkpoints

- [x] Map all tables with tenant FKs, lifecycle constraints, uniqueness and indexes.
- [x] Create pgvector migrations requiring an explicit fixed embedding dimension.
- [x] Configure stable Gemini embedding model, retrieval recipe version and dimension.
- [x] Implement initial token-hash authentication, ownership and scoped run queries.
- [ ] Implement operation-bound approval and external identity-provider integration if needed.
- [x] Initialize/close database/checkpoint/HTTP resources and normalize API errors.
- [x] Wire optional Langfuse lifecycle and redacted LLM/API observations.
- [ ] Wire optional Redis/object-storage clients.
- [x] Wire MinIO payload storage and Markdown/plain-text ingestion through admin CLI.
      Verify source-byte integrity, scoped retrieval/signed access and atomic document activation.
- [ ] Implement registry onboarding, reviewed execution metadata and taxonomy.
- [x] Implement embedded draft storage, local schema checks, allowlisted exact lookup,
      immutable publication/current-version switch and revocation with tenant revisions.
- [x] Implement authorized tool catalog and active/effective document queries.
- [ ] Implement version-compatible hybrid retrieval and bounded context.
- [x] Implement authorized tool/knowledge hybrid retrieval and conservative bounded
      context with an injectable model-specific counter; preserve whole tool schemas.
- [x] Implement/verify async Gemini embeddings and deterministic reciprocal-rank fusion.
- [ ] Verify RAG Pipeline and System Search as standalone read tools.
- [x] Test standalone RAG source validation, insufficient evidence and stale/revoked
      evidence rejection. Live context query passed; live generated answer is pending.
- [x] Verify standalone System Search with fresh permissions, lexical capability
      search, persisted status and redacted cursor-paginated event history.
- [ ] Implement strict model outputs, deterministic plan checks and action slots.
- [x] Implement safe JSON Pointer resolution and canonical JSON payload hashing.
- [x] Validate sequential read plans and fully schema-check resolved dispatch inputs.
      Business write plans remain blocked until approval/action policies are implemented.
- [x] Add a swappable async LLM provider interface and Gemini Interactions adapter.
- [x] Validate classification/planning/answer contracts at the LLM boundary.
- [ ] Confirm live Interactions generation (model access verified; generation timed out).
- [ ] Wire all graph nodes with allowlisted routes, pinned versions and sync durability.
- [x] Wire the generic read-only path with allowlisted routes, executable-version
      checks, sequential plans and sync checkpoints; write reconciliation stays disabled.
- [x] Implement idempotent submission and cooperative cancellation transactions.
- [ ] Implement checkpoint-validated input transactions and durable wait projection.
- [x] Implement checkpoint-matched clarification input, deduplication, wait projection and resume.
- [ ] Implement single-worker ownership, recovery, timers and acknowledgment repair.
- [x] Implement advisory locking, confirmed prior-owner recovery, checkpoint continuation,
      final projection repair and consumed clarification acknowledgment. Timers remain pending.
- [ ] Implement operation CAS guard, provider policies, verification and reconciliation.
- [ ] Enable optional payload offload, exact caches and curated trajectories.
- [x] Capture redacted model I/O, usage, validation errors and correlated API/run IDs.
- [x] Verify Langfuse credentials and a stored synthetic observation.
- [ ] Add graph/retrieval/tool/recovery observations with their implementation.
- [ ] Add infrastructure metrics and measure tuning hypotheses.
- [ ] Run the LLD acceptance and worker fault-injection scenarios.
- [ ] Lock compatible dependencies and deployment images after integration checks.

## Boundaries to preserve

PostgreSQL owns application evidence; LangGraph owns graph progress. Redis is
optional and cannot authorize or schedule execution. One worker invokes one
sequential graph at a time. Model output cannot select identity, credentials,
SQL, endpoint URLs, policy validators or authoritative business action identity.

No business transaction stays open across an LLM/provider call or user wait.
Ambiguous writes reconcile existing operations; they never cause a fresh write
merely because the node replayed. Cancellation cannot erase submitted effects.
Telemetry/cache outages must not invalidate recovery. Checkpoint completion
does not imply a verified financial outcome.
