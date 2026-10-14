# Implementation todo

Working plan: build one sequential, durable workflow using the existing modules.
First milestone: System Search and document Q&A, followed by business actions.
Use Google AI Studio with the existing credentials and a synthetic document.
No new packages
or services are needed merely to organize this work.

## Confirmed decisions and remaining selection

- [x] First milestone: System Search and document Q&A; business actions follow.
- [x] Embedding provider: Google AI Studio using the existing Gemini credentials.
- [x] Acceptance document: small synthetic fixture at `tests/fixtures/policy.md`.
- [x] Confirm embedding access and inspect existing database vector columns.
- [x] Set embedding model/version/dimension before applying migrations (local .env only).

Verified choice: `gemini-embedding-2`, 768 dimensions. Google's official documentation
lists it as stable and recommends 768 among its supported dimensions:
https://ai.google.dev/gemini-api/docs/embeddings
The network-enabled retry returned one 768-dimensional embedding. Local embedding
configuration is now set, application migrations and checkpoints are initialized,
and startup checks confirm the stored vector dimension. The asynchronous adapter
uses matching document/query retrieval formatting and validates returned vectors.

Immediate next batch: live read-workflow validation when Gemini is available,
durable retry timers/fault injection, and remaining recovery hardening. Business
write policies still require a concrete provider/action. No business flow is
selected in application code; the model proposes plans from registered metadata.

First milestone acceptance:

- System Search returns the requesting user's last relevant run and verified evidence.
- Document Q&A answers the fixture's return-window question with a source reference.
- A question absent from the fixture produces insufficient-evidence behavior.
- A second tenant cannot retrieve the first tenant's document or run.
- A submitted run reaches a persisted final result through the worker and graph.

## 1. Prepare and verify the foundation

- [x] Restore a reachable development PostgreSQL instance using the pgvector image.
- [x] Check application migrations and LangGraph checkpoint setup against that instance.
- [x] Create and verify a local development identity without exposing its token.
- [x] Run database integration tests against a disposable database.
- [ ] Verify a live Gemini Interactions generation using the configured model.
- [x] Add bounded, cancellable model/embedding retries for retryable errors.
      Defaults: three total attempts, waits of 10 then 20 seconds, capped at 30.
- [x] Select `gemini-embedding-2` at 768 dimensions and inspect existing database vector columns.
- [x] Configure embedding model/version/dimension; enable pgvector through application migrations.

Completion: database tests pass and live LLM generation succeeds. Embedding
configuration is explicit and compatible with stored vectors.

## 2. Complete deterministic core and catalog

- [ ] Implement scoped catalog, operation, input, event and status-projection repository methods.
- [x] Implement scope/adapter-filtered tool catalog and active/effective document queries.
- [x] Implement exact-version tool lookup, schema validation, publication and revocation.
- [x] Persist embedded drafts without network calls in the publication transaction.
- [ ] Implement plan validation, binding resolution, payload hashing and stable action identities.
- [x] Implement JSON Pointer/literal bindings and deterministic strict-JSON payload hashing.
- [x] Validate read-plan length, distinct sequential dependencies, exact authorized
      tool versions and literal input schemas; fully validate resolved arguments at dispatch.
- [ ] Complete plan-wide validation and operation-bound write action assignment.
- [x] Implement scoped redacted event history with run-bound keyset pagination.
- [ ] Test invalid scopes/versions, cycles, forward bindings and invalid arguments.

Completion: invalid plans cannot reach dispatch; registry versions and action
identities remain stable across retries.

## 3. Deliver standalone read tools

- [x] Implement and test standalone System Search capabilities/status/history.
      Refresh permissions, exclude the current run for last-request lookup, and
      return selected run identity/timestamp; status reads bypass caches.
- [x] Implement Markdown/plain-text import, UTF-8 extraction, bounded overlapping
      chunks with line locators, embeddings and atomic active-version replacement.
- [x] Archive originals in local MinIO and verify upload/read integrity.
- [x] Add `setup-storage` and authenticated `import-document` commands; unchanged
      active imports are idempotent and replacements preserve existing document scopes.
- [x] Implement swappable embedding-provider contract and Google AI Studio async adapter.
- [x] Verify live async embeddings and deterministic rank fusion.
- [x] Implement authorized lexical/vector tool and document retrieval with shared
      scope/version filters, rank fusion and soft tool-category relevance.
- [x] Include complete selected tool schemas and bounded retrieval results in
      model context. The application no longer applies a token budget; reported
      model usage remains available per call in run history and Langfuse.
- [x] Implement standalone RAG Pipeline with source-bound answers, no-evidence
      handling and permission/document-version rechecks around model calls.
- [x] Add authenticated `query-knowledge` CLI in context and answer modes.
- [ ] Verify successful live answer generation through configured Gemini transport.
- [x] Implement authenticated tool import, exact-version publication and revocation CLI.
      Imports validate approved adapters/schemas before embedding and remain drafts.
- [x] Verify approved live onboarding of System Search and RAG Pipeline version 1:
      embeddings, draft import, local publication, exact lookup and authorized discovery.
- [x] Verify tenant/scope/corpus/effective-date/embedding-version isolation,
      stale evidence rejection and source reference checks.

Completion: both tools work independently with deterministic fixtures and
cannot expose another tenant's data.

## 4. Deliver the first end-to-end workflow

- [x] Build generic read-only graph nodes and allowlisted routing around LLM task wrappers.
- [x] Implement sequential worker locking, short queue claims, heartbeats and resource shutdown.
- [x] Connect submission through read-tool processing to checkpointed and projected results/status.
- [x] Add a local Streamlit chat for conversations, status, clarification, cancellation
      and source display using the existing authenticated API.
- [x] Implement durable clarification waits, checkpoint-validated input acceptance,
      duplicate/conflicting input handling, resume and checkpoint-based acknowledgment.
- [x] Implement explicitly confirmed previous-owner recovery and final/wait projection repair.
- [ ] Implement durable retry/reconciliation timers and full worker crash/fault-injection coverage.
- [x] Test cancellation before dispatch and preserve queued/waiting API cancellation semantics.
- [ ] Extend cancellation/recovery scenarios for in-flight tools and business writes.
- [ ] Add correlated graph/retrieval/tool/recovery Langfuse observations.
- [x] Add worker/run and graph-stage observations, reusing existing LLM/read-tool telemetry.
- [x] Add optional shared Langfuse observations for standalone System Search outcomes.
- [x] Add optional shared Langfuse observations for standalone RAG outcomes.
- [x] Add graph v2 model-directed per-subgoal function calling, plan-only results,
      checkpointed Gemini stateless continuation and v1 recovery compatibility.
- [x] Save redacted LLM requests/responses, tool choices and tool results to local
      per-run JSONL traces; display function calls and results in chat activity.

Completion: an authenticated submission produces a verified answer; restart,
resume and cancellation tests pass.

Read-tool dispatch refreshes permissions, resolves/validates arguments, verifies
output schemas and rechecks evidence access before later nodes consume results.
New v2 runs execute requested reads and pause for an operation-bound approval before
each write; preconditions, idempotent operation records and ambiguous-outcome review apply.
V1 remains available for old checkpoints. Live Gemini function-call sandbox
acceptance and integration coverage for multi-step stateless continuation remain
to be run outside the constrained local test environment.

## 5. Add business actions safely

- [x] Choose PayPal sandbox Invoicing as the first external provider/action and inspect its API contract.
- [ ] Add local PayPal sandbox credentials and run the live create/send acceptance test.
- [x] Implement operation-bound approval for PayPal invoice create/send.
- [x] Implement guarded single dispatch, invoice preconditions and sandbox credential resolution.
- [x] Implement PayPal adapter, safe error normalization and outcome verification.
- [x] Reuse saved write outcomes and send ambiguous writes to manual review without retry.
- [ ] Add automated provider-specific reconciliation where evidence can prove an outcome.
- [ ] Fault-test crashes before dispatch, after provider success and before checkpoint persistence.
- [ ] Test permission revocation after approval and cancellation after dispatch.

Completion: retries/restarts cannot create duplicate effects; uncertain writes
remain pending or require review until supported by verified evidence.

## 6. Complete supporting LLD features

- [ ] Implement optional scoped Redis cache and outage behavior.
- [ ] Implement authorized object-storage offload, signed access and orphan cleanup.
- [x] Implement scoped payload upload/read and five-minute authorized signed access.
- [ ] Complete orphan cleanup and workflow payload-offload integration.
- [x] Implement embedding-based taxonomy clustering, Gemini metadata review and guarded batch publication.
- [ ] Curate and expand tool-retrieval evaluation queries as API families are added.
- [ ] Implement curated trajectory publication.
- [ ] Complete remaining admin commands and resource prompt drafts.
- [ ] Add infrastructure metrics and remaining LLD acceptance tests.
- [ ] Lock compatible dependencies/images and document reproducible startup.

## Blockers and verification gaps

- Docker/PostgreSQL, application migrations, vector extension/dimensions and
  checkpoint setup are verified. Database readiness is no longer a blocker.
- MinIO is running from the approved official source-release build. Bucket setup,
  live synthetic document import, original-byte readback and signed access passed.
  Credentials remain in ignored `.env`; Docker build contexts exclude `.env` and runtime tokens.
- Gemini generation: model access was verified, but a successful live Interactions
  response remains unverified. The latest sanitized diagnostic returned HTTP 503
  (`InternalServerError`); retrieval context mode works independently of generation.
  The bounded retry loop is implemented/tested; a live three-attempt check with
  15-second attempt timeouts exhausted its budget. Production attempt timeout
  remains configurable (default 60 seconds), without an infinite retry loop.
- Live tool onboarding: user explicitly approved descriptive-text transfer to
  Google. Both built-in read tools are published and discoverable locally.
  Embedding text excluded schemas, run data and credentials.
- Embeddings: Google AI Studio returned a 768-dimensional `gemini-embedding-2`
  embedding with the existing .env key. Configuration and the async adapter are
  now implemented and verified. Ingestion and hybrid retrieval are wired; live
  context retrieval of the synthetic policy passed.
- Authentication: local development identity created and verified through the API.
  Its bearer token is stored in ignored `runtime/local-auth.json`; only its hash
  is stored in `.env`. Restart running API/worker processes to load the new binding.
- Business actions: a concrete provider/action, API contract and sandbox credentials
  are needed before real adapter implementation.
- Queue execution: read-only graph and worker are implemented and database-tested.
  Run the separate worker process to consume submissions; no worker daemon was
  started as part of this implementation. Live model-backed workflow remains unverified.

Planning, repository and registry work can proceed while external setup is
resolved. Redis, object storage and external identity-provider integration are
not blockers for the initial read workflow.

Latest unit validation: 120 tests passed, including the Streamlit chat smoke test.
Last PostgreSQL integration validation: 28 tests passed using a disposable
migrated/checkpoint-initialized database. Ruff checks passed.
No secrets were printed; temporary test databases were removed.
