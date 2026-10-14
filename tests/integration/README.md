Implemented PostgreSQL foundation tests cover submission deduplication/conflict,
concurrent rollback, owner/tenant boundaries, last-request selection, cancellation,
vector dimensions, taxonomy depth, append-only events and resource cleanup.
Registry/catalog tests additionally cover immutable version publication,
current-version switching, pinned older versions, revocation, tenant/scope/adapter
filters and active/effective document access.
Standalone System Search tests cover owner/conversation-wide last selection,
current-run exclusion, fresh status, tied-timestamp pagination, run-bound cursors,
redacted evidence, management boundaries and refreshed scope revocation.
Document-ingestion tests cover idempotent active imports, embedding/upload failure
preservation, atomic replacement, inherited document scopes and payload integrity.
Hybrid retrieval tests verify both branches reject wrong tenant, restricted/inactive/
future/expired documents, incompatible embeddings and unconfigured/revoked tool versions.
Tool-import tests verify unapproved adapters are rejected before embeddings,
imports remain drafts and immutable version conflicts cannot overwrite publication.
Graph/worker tests use real PostgreSQL checkpoints and mocked models to verify
submission-to-result execution, clarification/resume, input deduplication,
projection repair without reexecution, cancellation, queue claims, heartbeat
fencing, advisory-lock exclusion and composed worker resource shutdown.
Set `TEST_DATABASE_URL` to a disposable migrated database with checkpoint setup
completed; test vectors use dimension 3 (`TEST_EMBEDDING_DIMENSION`).

Remaining integration scenarios for graph/provider implementation:

- Crash before dispatch resumes the eligible checkpoint work.
- Provider success with lost acknowledgment reconciles without a fresh write.
- Saved operation result with lost checkpoint returns stored verified evidence.
- Consumed input with lost acknowledgment is applied once using last_input_id.
- Due retry survives restart and retains budgets.
- Accepted refund remains pending until verification.
- Missing idempotency/lookup after ambiguous dispatch yields needs_review.
- Revoked permission after approval prevents dispatch.
- Missing/stale/restricted RAG evidence cannot leak or support an invented answer.
- Cancellation reconciles existing effects and prevents later business dispatch.
- Redis/Langfuse outages preserve database recovery and show degradation.
- Bad tool versions/bindings/cyclic plans are rejected before dispatch.
- Fault injection terminates worker at dispatch/result/checkpoint boundaries.

The remaining scenarios above are pending acceptance criteria, not passing tests.
